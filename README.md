# RPI Zero 2W AI chat gateway

![hardware setup](/imgs/hw.jpeg)

Based on ElevenLabs tutorial [Build a Voice Assistant with Agents Platform on a Raspberry Pi](https://elevenlabs.io/docs/cookbooks/agents-platform/raspberry-pi-voice-assistant), but resolving issues with running the code on [Raspberry Pi Zero 2W](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/).

## Summary of changes

- Python 3.11
- numpy version to 1.24.2
- Change logic for handling microphone input (resolving `Input overflowed` errors with original code)

## Hardware

- Raspberry Pi Zero 2 W
- USB sound card
- Microphone
- Speakers
- USB micro to USB A adapter

## Steps

1. Install the dependencies

```
sudo apt-get update
sudo apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev libasound-dev libsndfile1-dev sox -y
```

2. Check if you can record and play audio

```
# download sample audio for audio playback testing
wget https://www.kozco.com/tech/piano2.wav
play piano2.wav
# check microphone
arecord --format S16_LE --rate 44100 --duration 3 test.wav
play test.wav
```
3. I prefer using [`uv`](https://docs.astral.sh/uv/) for managing Python versions and dev environments. Install `uv` and run `uv sync` to init environment.

4. Run `uv run main.py` to run the code. Say `Hey Eleven` to start chat session. Stop the session by pressins `Ctrl+C`.
