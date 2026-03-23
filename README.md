MIDI Pitch Player FX Looper + MasterRecorder

This project is a Python-based real-time sample player and looper controlled by MIDI and optional PC keyboard input.

FEATURES
- Pitch-shifted sample playback across a MIDI range
- Real-time MIDI note triggering
- Audio effects: distortion, delay, spring reverb, repeater
- Multiple loop recording and playback slots
- Configurable PC keyboard control via JSON
- Multiple special notes with independent samples and behaviors
- Master output recording to WAV
- Live config reload without restarting

REQUIREMENTS
Python 3.10+
Install dependencies:
pip install numpy sounddevice soundfile pygame pynput

RUN
python MidiKeyboard.py AudioPiano.wav

OPTIONAL PARAMETERS
root_note, midi_device_id, skip_ms, threshold

CONFIG.JSON CONTROLS
- instrument settings
- note range
- effects
- loops
- keyboard bindings
- special notes
- master recording

SPECIAL NOTES
Each special note can:
- trigger from MIDI or keyboard
- play once, hold, or loop
- be toggled independently

LOOPS
- record toggle via MIDI or keyboard
- play toggle via MIDI or keyboard
- volume via MIDI CC

KEYBOARD ACTIONS
- note
- special_note
- loop_record_toggle
- loop_play_toggle
- master_record_toggle
- panic

MASTER RECORDING
- toggle on/off
- saves to WAV automatically
- optional auto timestamp filename

PROJECT STRUCTURE
- script.py
- config.json
- audio files
- README.txt

USE CASES
- live performance
- sample triggering
- loop-based composition
- experimental instruments

TROUBLESHOOTING
- check MIDI device ID
- verify audio files exist
- install pynput for keyboard support
- ensure notes are within range

Additional demonstration videos will be published soon to showcase the system in real-world scenarios.

First Video demo: https://www.youtube.com/watch?v=o6R80QeBPDw


--------------Upgrade-------------

Add the BPM Special Function for change bpm on the Loop
New Loop COnfiguration on json: 

 {
      "name": "LOOP1",
      "channel": 10,
      "record_note": 40,
      "play_note": 36,
      "volume_cc": 30,
      "volume_channel": 1,
      "tempo_correction": true ,
      "target_bpm": 5.0,
      "beats": 4.0,
      "keep_original_if_invalid": true     
    },
---------------------Upgrade --------
Add the option on json special note 

"special_notes": [
    {
      "name": "bass_LOOP",
      "enabled": true,
      "midi_note": 0,
      "keyboard_key": "q",
      "file": "DrumLong.wav",
      "volume": 1.0,
      "loop": true,
      "repeat_count": 1,   ///////////new repeat count on loop.
      "stop_on_release": false
    },