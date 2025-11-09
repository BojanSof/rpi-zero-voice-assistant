import os
import signal
import time
import threading
import queue  # Import the queue module

from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector

from eff_word_net.audio_processing import Resnet50_Arc_loss

# from eff_word_net import samples_loc

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, ConversationInitiationData
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

convai_active = False

elevenlabs = ElevenLabs()
agent_id = os.getenv("ELEVENLABS_AGENT_ID")
api_key = os.getenv("ELEVENLABS_API_KEY")

dynamic_vars = {
    'user_name': 'Thor',
    'greeting': 'Hey'
}

config = ConversationInitiationData(
    dynamic_variables=dynamic_vars
)

# --- Load Hotword Model ---
# Ensure you have the model files and reference file in the correct paths
try:
    base_model = Resnet50_Arc_loss()

    eleven_hw = HotwordDetector(
        hotword="hey_eleven",
        model=base_model,
        # Make sure this path is correct
        reference_file=os.path.join("hotword_refs", "hey_eleven_ref.json"),
        threshold=0.7,
        relaxation_time=2
    )
    print("Hotword detector loaded.")
except Exception as e:
    print(f"Error loading hotword detector. Make sure 'hotword_refs/hey_eleven_ref.json' exists.")
    print(f"Error: {e}")
    exit()

# Global variable for the microphone stream
mic_stream = None

# --- Threading Globals ---
frame_queue = queue.Queue()  # A queue to hold audio frames
wakeword_detected = False      # Flag to signal main loop
stop_worker = threading.Event() # Flag to stop the worker thread

def create_conversation():
    """Create a new conversation instance"""
    print("Creating new ElevenLabs conversation...")
    return Conversation(
        # API client and agent ID.
        elevenlabs,
        agent_id,
        config=config,

        # Assume auth is required when API_KEY is set.
        requires_auth=bool(api_key),

        # Use the default audio interface.
        audio_interface=DefaultAudioInterface(),

        # Simple callbacks that print the conversation to the console.
        callback_agent_response=lambda response: print(f"Agent: {response}"),
        callback_agent_response_correction=lambda original, corrected: print(f"Agent: {original} -> {corrected}"),
        callback_user_transcript=lambda transcript: print(f"User: {transcript}"),

        # Uncomment if you want to see latency measurements.
        # callback_latency_measurement=lambda latency: print(f"Latency: {latency}ms"),
    )

def processing_worker():
    """
    Runs in a separate thread.
    Continuously gets frames from the queue and processes them.
    This is the only place `eleven_hw.scoreFrame` is called.
    """
    global wakeword_detected
    print("Processing worker thread started.")
    while not stop_worker.is_set():
        try:
            # Block until a frame is available
            frame = frame_queue.get(timeout=1.0)
            if frame is None:
                continue

            # This is the slow, CPU-intensive part
            result = eleven_hw.scoreFrame(frame)
            
            if result and result["match"]:
                print(f"Wakeword uttered (Confidence: {result['confidence']:.2f})")
                wakeword_detected = True  # Set the flag for the main loop
                
                # Clear the queue so we don't process old frames
                with frame_queue.mutex:
                    frame_queue.queue.clear()
            
            frame_queue.task_done()

        except queue.Empty:
            # This is normal, just loop again if the queue is empty
            continue
        except Exception as e:
            if not stop_worker.is_set():
                print(f"Error in processing worker: {e}")
                time.sleep(0.1)

    print("Processing worker thread stopped.")

def start_mic_stream():
    """Start or restart the microphone stream"""
    global mic_stream
    # Don't start if it's already running
    if mic_stream:
        print("Microphone stream is already active.")
        return

    try:
        # Create a new stream instance
        mic_stream = SimpleMicStream(
            window_length_secs=1.5,
            # *** FIX: Use a smaller, more stable sliding window ***
            sliding_window_secs=0.25,
        )
        mic_stream.start_stream()
        print("Microphone stream started for wake word detection.")
    except Exception as e:
        print(f"Error starting microphone stream: {e}")
        mic_stream = None
        time.sleep(1)  # Wait a bit before retrying

def stop_mic_stream():
    """Stop the microphone stream safely"""
    global mic_stream
    try:
        if mic_stream:
            # 1. Check for the PyAudio stream object
            if hasattr(mic_stream, 'stream') and mic_stream.stream:
                # 2. Stop and close the stream
                mic_stream.stream.stop_stream()
                mic_stream.stream.close()
                print("PyAudio stream stopped and closed.")
            
            # 3. Check for the main PyAudio instance
            if hasattr(mic_stream, 'p') and mic_stream.p:
                # 4. Terminate the PyAudio instance
                mic_stream.p.terminate()
                print("PyAudio instance terminated.")
            
            print("Microphone stream fully shut down.")
        else:
            print("Microphone stream was not running.")
    except Exception as e:
        print(f"Error stopping microphone stream: {e}")
    finally:
        # Ensure the global variable is reset
        mic_stream = None

# --- Main Application Logic ---

# Start the persistent processing worker thread
stop_worker.clear()
worker_thread = threading.Thread(target=processing_worker, daemon=True)
worker_thread.start()

# Initial start of the microphone stream
start_mic_stream()

print("\nSay 'Hey Eleven' to start the conversation.")
while True:
    try:
        # --- State 1: Conversation is Active ---
        # This state is now handled inside the 'wakeword_detected' block.
        if convai_active:
            time.sleep(0.1)
            continue

        # --- State 2: Wakeword was Detected by the Thread ---
        if wakeword_detected:
            convai_active = True
            wakeword_detected = False  # Reset flag

            # Stop the wake word microphone stream to avoid conflicts
            stop_mic_stream()

            # Give the OS a moment to fully release the audio device
            print("Audio device released, waiting 0.5s before starting conversation...")
            time.sleep(0.5)

            # Start ConvAI Session
            print("Starting ConvAI Session...")
            try:
                # Create a new conversation instance
                conversation = create_conversation()

                # Start the session
                conversation.start_session()

                # Set up signal handler for graceful shutdown (e.g., Ctrl+C)
                def signal_handler(sig, frame):
                    print("\nReceived interrupt signal, ending session...")
                    try:
                        conversation.end_session()
                    except Exception as e:
                        print(f"Error ending session: {e}")

                signal.signal(signal.SIGINT, signal_handler)

                # Wait for session to end (either by user interrupt or natural end)
                conversation_id = conversation.wait_for_session_end()
                print(f"Conversation session finished. ID: {conversation_id}")

            except Exception as e:
                print(f"Error during conversation: {e}")
            finally:
                # Cleanup after conversation
                convai_active = False
                print("Conversation ended, cleaning up...")
                
                # Remove the signal handler to avoid conflicts
                signal.signal(signal.SIGINT, signal.SIG_DFL) 

                # Give some time for audio resources to be released
                time.sleep(1)

                # Restart microphone stream for wake word detection
                print("Restarting wake word detection...")
                # The worker thread is still running, just clear the queue
                with frame_queue.mutex:
                    frame_queue.queue.clear()
                wakeword_detected = False # Reset flag
                start_mic_stream()
                print("\nSay 'Hey Eleven' to start the conversation.")
            
            continue # Go back to the start of the while loop

        # --- State 3: Ready for Wakeword ---
        
        # Make sure we have a valid mic stream
        if mic_stream is None:
            print("Mic stream is not running, attempting to restart...")
            start_mic_stream()
            # Skip to the next loop iteration to avoid errors
            if mic_stream is None: # If it failed again, wait
                time.sleep(1)
                continue

        # Get a frame and queue it up.
        # This call will block until a new frame is ready,
        # perfectly servicing the audio buffer.
        try:
            frame = mic_stream.getFrame()
            if frame is not None:
                frame_queue.put(frame)
        except Exception as e:
            # This will happen if stop_mic_stream() closes the stream
            if not convai_active: # Only log if we weren't expecting it
                print(f"Error getting frame: {e}")
                stop_mic_stream() # Ensure it's stopped
                time.sleep(1)

    except Exception as e:
        # This handles errors in the main wake word detection loop
        print(f"Error in wake word detection loop: {e}")
        # Try to restart microphone stream if there's an error
        stop_mic_stream() # Ensure it's fully stopped
        
        # Clear queue and reset flags
        with frame_queue.mutex:
            frame_queue.queue.clear()
        wakeword_detected = False
        
        time.sleep(1)
        # The loop will attempt to restart it on the next iteration
