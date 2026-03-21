import sys
import os
import time
import json
import threading
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
import pygame
import pygame.midi

try:
    from pynput import keyboard as pynput_keyboard   # need to install pynput if isnt installed the sketch will work without key recognition
    PYNPUT_AVAILABLE = True
except Exception:
    PYNPUT_AVAILABLE = False


DEFAULT_ROOT_NOTE = 60
DEFAULT_MIDI_DEVICE_ID = 17
DEFAULT_SKIP_MS = 0.0
DEFAULT_THRESHOLD = 0.01
DEFAULT_BLOCKSIZE = 64
BASE_VOLUME = 0.8

NOTE_MIN = 36
NOTE_MAX = 96

# -----------------------------
# EFFETTI - DEFAULT
# -----------------------------
DEFAULT_DRIVE = 0.0
DEFAULT_OUTPUT_GAIN = 1.0

DEFAULT_DELAY_MS = 0.0
DEFAULT_DELAY_FEEDBACK = 0.0
DEFAULT_DELAY_MIX = 0.0

DEFAULT_SPRING_MIX = 0.25
DEFAULT_SPRING_DECAY = 0.6
DEFAULT_SPRING_TONE = 0.7

DEFAULT_REPEATER_MS = 80.0
DEFAULT_REPEATER_REPEATS = 4
DEFAULT_REPEATER_DECAY = 0.55
DEFAULT_REPEATER_MIX = 0.45


def note_name(note_number):
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (note_number // 12) - 1
    return f"{names[note_number % 12]}{octave}"


def list_midi_devices():
    print("\nDispositivi MIDI disponibili:\n")
    for i in range(pygame.midi.get_count()):
        info = pygame.midi.get_device_info(i)
        interface = info[0].decode(errors="ignore")
        name = info[1].decode(errors="ignore")
        is_input = info[2]
        is_output = info[3]
        opened = info[4]

        io_type = []
        if is_input:
            io_type.append("INPUT")
        if is_output:
            io_type.append("OUTPUT")

        print(f"[{i}] {name} ({interface}) - {'/'.join(io_type)} - aperto:{opened}")
    print()


def trim_leading_silence(audio, threshold=0.01):
    above = np.where(np.abs(audio) > threshold)[0]
    if len(above) == 0:
        return audio
    return audio[above[0]:]


def skip_initial_ms(audio, samplerate, ms=0.0):
    if ms <= 0:
        return audio

    samples_to_skip = int((ms / 1000.0) * samplerate)
    if samples_to_skip >= len(audio):
        return audio
    return audio[samples_to_skip:]


def smooth_loop_edges(audio, fade_len=256):
    if audio is None or len(audio) < 4:
        return audio

    out = audio.copy()
    fade_len = min(fade_len, len(out) // 2)
    if fade_len <= 0:
        return out

    out[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    out[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    return out.astype(np.float32)


def resample_audio(audio, src_sr, dst_sr):
    if src_sr == dst_sr:
        return audio.astype(np.float32)

    new_length = max(1, int(len(audio) * dst_sr / src_sr))
    old_idx = np.arange(len(audio), dtype=np.float32)
    new_idx = np.linspace(0, len(audio) - 1, new_length, dtype=np.float32)
    return np.interp(new_idx, old_idx, audio).astype(np.float32)


def load_audio(filename, threshold=0.01, skip_ms=0.0):
    data, samplerate = sf.read(filename, dtype="float32")

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    data = trim_leading_silence(data, threshold=threshold)
    data = skip_initial_ms(data, samplerate, ms=skip_ms)

    fade_len = min(32, len(data))
    if fade_len > 1:
        data[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)

    return data.astype(np.float32), samplerate


def pitch_shift_resample(audio, semitones):
    factor = 2 ** (semitones / 12.0)

    old_indices = np.arange(len(audio), dtype=np.float32)
    new_length = max(1, int(len(audio) / factor))
    new_indices = np.linspace(0, len(audio) - 1, new_length, dtype=np.float32)

    shifted = np.interp(new_indices, old_indices, audio).astype(np.float32)
    return shifted


def preload_notes(sample, root_note, note_min=36, note_max=96):
    cache = {}
    print("Precaricamento note...")
    for midi_note in range(note_min, note_max + 1):
        semitones = midi_note - root_note
        cache[midi_note] = pitch_shift_resample(sample, semitones)
    print(f"Note precaricate da {note_min} a {note_max}")
    return cache


def get_default_loops():
    return [
        {
            "name": "LOOP1",
            "channel": 10,
            "record_note": 40,
            "play_note": 36,
            "volume_cc": 30,
            "volume_channel": 10
        },
        {
            "name": "LOOP2",
            "channel": 10,
            "record_note": 41,
            "play_note": 37,
            "volume_cc": 31,
            "volume_channel": 10
        }
    ]


def get_default_keyboard_bindings():
    return [
        {
            "key": "z",
            "action": "note",
            "midi_note": 60,
            "velocity": 120,
            "stop_on_release": False
        },
        {
            "key": "x",
            "action": "note",
            "midi_note": 62,
            "velocity": 120,
            "stop_on_release": False
        },
        {
            "key": "c",
            "action": "note",
            "midi_note": 64,
            "velocity": 120,
            "stop_on_release": False
        },
        {
            "key": "f1",
            "action": "loop_record_toggle",
            "loop_name": "LOOP1"
        },
        {
            "key": "f2",
            "action": "loop_play_toggle",
            "loop_name": "LOOP1"
        },
        {
            "key": "f9",
            "action": "master_record_toggle"
        },
        {
            "key": "esc",
            "action": "panic"
        }
    ]


def get_default_special_notes():
    return [
        {
            "name": "SPECIAL1",
            "enabled": True,
            "midi_note": 72,
            "keyboard_key": "q",
            "file": "OrganGame.wav",
            "volume": 1.0,
            "loop": False,
            "stop_on_release": True
        }
    ]


def load_config(config_path="config.json"):
    defaults = {
        "instrument_file": None,
        "instrument_root_note": DEFAULT_ROOT_NOTE,
        "midi_device_id": DEFAULT_MIDI_DEVICE_ID,
        "skip_ms": DEFAULT_SKIP_MS,
        "threshold": DEFAULT_THRESHOLD,
        "note_min": NOTE_MIN,
        "note_max": NOTE_MAX,

        "special_notes": get_default_special_notes(),

        "base_volume": BASE_VOLUME,
        "drive": DEFAULT_DRIVE,
        "output_gain": DEFAULT_OUTPUT_GAIN,
        "delay_ms": DEFAULT_DELAY_MS,
        "delay_feedback": DEFAULT_DELAY_FEEDBACK,
        "delay_mix": DEFAULT_DELAY_MIX,
        "spring_mix": DEFAULT_SPRING_MIX,
        "spring_decay": DEFAULT_SPRING_DECAY,
        "spring_tone": DEFAULT_SPRING_TONE,
        "repeater_ms": DEFAULT_REPEATER_MS,
        "repeater_repeats": DEFAULT_REPEATER_REPEATS,
        "repeater_decay": DEFAULT_REPEATER_DECAY,
        "repeater_mix": DEFAULT_REPEATER_MIX,

        "loops": get_default_loops(),

        "keyboard_enabled": False,
        "keyboard_bindings": get_default_keyboard_bindings(),

        "master_record_enabled": False,
        "master_record_file": "master_record.wav",
        "master_record_autoname": True
    }

    if not os.path.isfile(config_path):
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)

        defaults.update(user_cfg)

        if "loops" not in defaults or not isinstance(defaults["loops"], list) or len(defaults["loops"]) == 0:
            defaults["loops"] = get_default_loops()

        if "keyboard_bindings" not in defaults or not isinstance(defaults["keyboard_bindings"], list):
            defaults["keyboard_bindings"] = get_default_keyboard_bindings()

        if "special_notes" not in defaults or not isinstance(defaults["special_notes"], list):
            defaults["special_notes"] = get_default_special_notes()

        return defaults

    except Exception as e:
        print(f"Errore lettura config: {e}")
        return defaults


def normalize_loops_config(loops_cfg):
    normalized = []

    if not isinstance(loops_cfg, list) or len(loops_cfg) == 0:
        loops_cfg = get_default_loops()

    for i, loop in enumerate(loops_cfg, start=1):
        try:
            channel = int(loop.get("channel", 10))
            normalized.append({
                "name": str(loop.get("name", f"LOOP{i}")),
                "channel": channel,
                "record_note": int(loop.get("record_note")),
                "play_note": int(loop.get("play_note")),
                "volume_cc": int(loop.get("volume_cc")),
                "volume_channel": int(loop.get("volume_channel", channel)),
            })
        except Exception as e:
            print(f"[LOOPS CONFIG] Loop ignorato: {loop} -> {e}")

    if len(normalized) == 0:
        normalized = get_default_loops()

    return normalized


def normalize_keyboard_bindings(bindings):
    if not isinstance(bindings, list):
        return get_default_keyboard_bindings()

    out = []
    for item in bindings:
        try:
            out.append({
                "key": str(item.get("key")).lower(),
                "action": str(item.get("action")).lower(),
                "midi_note": int(item.get("midi_note", 60)),
                "velocity": int(item.get("velocity", 120)),
                "stop_on_release": bool(item.get("stop_on_release", False)),
                "loop_name": item.get("loop_name"),
            })
        except Exception as e:
            print(f"[KEYBOARD CONFIG] binding ignorato: {item} -> {e}")
    return out


def normalize_special_notes_config(special_notes_cfg):
    if not isinstance(special_notes_cfg, list):
        special_notes_cfg = get_default_special_notes()

    normalized = []
    for i, item in enumerate(special_notes_cfg, start=1):
        try:
            normalized.append({
                "name": str(item.get("name", f"SPECIAL{i}")),
                "enabled": bool(item.get("enabled", True)),
                "midi_note": item.get("midi_note", None),
                "keyboard_key": str(item.get("keyboard_key")).lower() if item.get("keyboard_key") is not None else None,
                "file": item.get("file"),
                "volume": float(item.get("volume", 1.0)),
                "loop": bool(item.get("loop", False)),
                "stop_on_release": bool(item.get("stop_on_release", True)),
            })
        except Exception as e:
            print(f"[SPECIAL NOTES CONFIG] special note ignorata: {item} -> {e}")

    return normalized


def key_to_string(key):
    try:
        if hasattr(key, "char") and key.char is not None:
            return str(key.char).lower()
    except Exception:
        pass

    try:
        s = str(key)
        if s.startswith("Key."):
            return s.split(".", 1)[1].lower()
    except Exception:
        pass

    return None


class SampleVoice:
    def __init__(self, data, velocity_gain, midi_note=None, stop_on_noteoff=False, loop_enabled=False, special_name=None):
        self.data = data
        self.pos = 0
        self.gain = velocity_gain
        self.midi_note = midi_note
        self.stop_on_noteoff = stop_on_noteoff
        self.loop_enabled = loop_enabled
        self.special_name = special_name
        self.stopped = False

    def stop(self):
        self.stopped = True

    def render(self, frames):
        if self.stopped:
            return np.zeros(frames, dtype=np.float32), True

        out = np.zeros(frames, dtype=np.float32)

        if not self.loop_enabled:
            remaining = len(self.data) - self.pos
            if remaining <= 0:
                return np.zeros(frames, dtype=np.float32), True

            n = min(frames, remaining)
            out[:n] = self.data[self.pos:self.pos + n] * self.gain
            self.pos += n
            finished = self.pos >= len(self.data)
            return out, finished

        if len(self.data) == 0:
            return out, True

        for i in range(frames):
            out[i] = self.data[self.pos] * self.gain
            self.pos += 1
            if self.pos >= len(self.data):
                self.pos = 0

        return out, False


class LoopSlot:
    def __init__(self, name):
        self.name = name
        self.is_recording = False
        self.record_buffer = []
        self.sample = None
        self.enabled = False
        self.pos = 0
        self.gain = 1.0

    def clear_recording(self):
        self.record_buffer = []

    def has_sample(self):
        return self.sample is not None and len(self.sample) > 0

    def set_sample(self, audio):
        self.sample = audio
        self.pos = 0

    def toggle_play(self):
        if not self.has_sample():
            return False, "Nessun loop registrato"

        if not self.enabled:
            self.enabled = True
            self.pos = 0
            return True, "PLAY"
        else:
            self.enabled = False
            return True, "STOP"

    def render(self, frames):
        out = np.zeros(frames, dtype=np.float32)

        if not self.enabled or not self.has_sample():
            return out

        for i in range(frames):
            out[i] = self.sample[self.pos] * self.gain
            self.pos += 1
            if self.pos >= len(self.sample):
                self.pos = 0

        return out


class LowLatencySamplePlayer:
    def __init__(
        self,
        note_cache,
        samplerate,
        loops_config,
        base_volume=0.8,
        drive=0.0,
        output_gain=1.0,
        delay_ms=0.0,
        delay_feedback=0.0,
        delay_mix=0.0,
        spring_mix=0.0,
        spring_decay=0.6,
        spring_tone=0.7,
        repeater_ms=0.0,
        repeater_repeats=0,
        repeater_decay=0.5,
        repeater_mix=0.0,
    ):
        self.note_cache = note_cache
        self.samplerate = samplerate
        self.base_volume = base_volume
        self.voices = []
        self.lock = threading.Lock()
        self.peak_guard = 0.98

        self.delay_buffer = None
        self.delay_pos = 0
        self.spring_buffers = []
        self.spring_pos = []
        self.spring_prev = []
        self.repeater_delay_samples = 0

        self.loops_config = normalize_loops_config(loops_config)
        self.loops = [LoopSlot(loop_cfg["name"]) for loop_cfg in self.loops_config]

        # special notes multiple
        self.special_notes_cfg = []
        self.special_notes_by_midi = {}
        self.special_notes_by_key = {}
        self.special_note_active_voices = {}

        # keyboard
        self.keyboard_enabled = False
        self.keyboard_bindings = []
        self.keyboard_pressed_keys = set()

        # master record
        self.master_record_enabled = False
        self.master_record_file = "master_record.wav"
        self.master_record_autoname = True
        self.master_record_buffer = []

        self.update_effects(
            base_volume=base_volume,
            drive=drive,
            output_gain=output_gain,
            delay_ms=delay_ms,
            delay_feedback=delay_feedback,
            delay_mix=delay_mix,
            spring_mix=spring_mix,
            spring_decay=spring_decay,
            spring_tone=spring_tone,
            repeater_ms=repeater_ms,
            repeater_repeats=repeater_repeats,
            repeater_decay=repeater_decay,
            repeater_mix=repeater_mix,
        )

    def update_loops_config(self, loops_config):
        new_cfg = normalize_loops_config(loops_config)

        with self.lock:
            old_slots = {slot.name: slot for slot in self.loops}
            new_slots = []

            for loop_cfg in new_cfg:
                name = loop_cfg["name"]
                if name in old_slots:
                    new_slots.append(old_slots[name])
                else:
                    new_slots.append(LoopSlot(name))

            self.loops_config = new_cfg
            self.loops = new_slots

        print("[LOOPS] Config aggiornata")
        for loop_cfg in self.loops_config:
            print(
                f"  {loop_cfg['name']} | CH:{loop_cfg['channel']} | "
                f"REC:{loop_cfg['record_note']} ({note_name(loop_cfg['record_note'])}) | "
                f"PLAY:{loop_cfg['play_note']} ({note_name(loop_cfg['play_note'])}) | "
                f"VOL_CC:{loop_cfg['volume_cc']} | VOL_CH:{loop_cfg['volume_channel']}"
            )

    def update_keyboard_config(self, enabled, bindings):
        normalized = normalize_keyboard_bindings(bindings)
        with self.lock:
            self.keyboard_enabled = bool(enabled)
            self.keyboard_bindings = normalized
            self.keyboard_pressed_keys.clear()

        print(f"[KEYBOARD] enabled: {self.keyboard_enabled}")
        for b in normalized:
            print(f"  key:{b['key']} -> action:{b['action']}")

    def update_master_record_config(self, enabled=None, filename=None, autoname=None):
        with self.lock:
            if filename is not None:
                self.master_record_file = str(filename)
            if autoname is not None:
                self.master_record_autoname = bool(autoname)

            if enabled is not None:
                enabled = bool(enabled)
                if enabled and not self.master_record_enabled:
                    self.start_master_recording()
                elif (not enabled) and self.master_record_enabled:
                    self.stop_master_recording(save_file=True)

    def start_master_recording(self):
        self.master_record_buffer = []
        self.master_record_enabled = True
        print("[MASTER REC] START")

    def stop_master_recording(self, save_file=True):
        if not self.master_record_enabled:
            return

        self.master_record_enabled = False
        print("[MASTER REC] STOP")

        if save_file:
            self.save_master_recording()

    def _build_master_filename(self):
        base = self.master_record_file or "master_record.wav"

        if self.master_record_autoname:
            name, ext = os.path.splitext(base)
            if ext.strip() == "":
                ext = ".wav"
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{name}_{stamp}{ext}"

        return base

    def save_master_recording(self):
        if len(self.master_record_buffer) == 0:
            print("[MASTER REC] Nessun audio da salvare")
            return

        try:
            audio = np.concatenate(self.master_record_buffer).astype(np.float32)
            filename = self._build_master_filename()

            ext = os.path.splitext(filename)[1].lower()
            if ext not in [".wav", ".flac", ".ogg"]:
                filename = os.path.splitext(filename)[0] + ".wav"

            sf.write(filename, audio, self.samplerate)
            duration_sec = len(audio) / self.samplerate
            print(f"[MASTER REC] Salvato: {filename} ({duration_sec:.2f} sec)")
        except Exception as e:
            print(f"[MASTER REC] Errore salvataggio: {e}")

    def toggle_master_recording(self):
        with self.lock:
            if self.master_record_enabled:
                self.stop_master_recording(save_file=True)
            else:
                self.start_master_recording()

    def update_instrument(self, sample, root_note, note_min=36, note_max=96):
        new_cache = preload_notes(sample, root_note, note_min, note_max)
        with self.lock:
            self.note_cache = new_cache
        print(f"[INSTRUMENT] Nuovo strumento caricato con root note {root_note} ({note_name(root_note)})")

    def update_special_notes(self, cfg, threshold=DEFAULT_THRESHOLD, skip_ms=DEFAULT_SKIP_MS):
        special_notes_cfg = normalize_special_notes_config(cfg.get("special_notes", get_default_special_notes()))

        loaded_cfg = []
        by_midi = {}
        by_key = {}

        for item in special_notes_cfg:
            item_copy = dict(item)
            item_copy["sample"] = None

            if not item_copy["enabled"]:
                loaded_cfg.append(item_copy)
                continue

            file = item_copy.get("file")
            if file and os.path.isfile(file):
                try:
                    data, sr = load_audio(file, threshold=threshold, skip_ms=skip_ms)
                    if sr != self.samplerate:
                        data = resample_audio(data, sr, self.samplerate)
                        print(f"[SPECIAL NOTE {item_copy['name']}] Resampled to {self.samplerate} Hz")

                    item_copy["sample"] = data
                    print(
                        f"[SPECIAL NOTE {item_copy['name']}] caricata | "
                        f"midi:{item_copy.get('midi_note')} | key:{item_copy.get('keyboard_key')} | "
                        f"loop:{item_copy.get('loop')} | stop_on_release:{item_copy.get('stop_on_release')}"
                    )
                except Exception as e:
                    print(f"[SPECIAL NOTE {item_copy['name']}] errore caricamento: {e}")
            else:
                print(f"[SPECIAL NOTE {item_copy['name']}] file non trovato: {file}")

            loaded_cfg.append(item_copy)

            midi_note = item_copy.get("midi_note")
            if midi_note is not None:
                by_midi[int(midi_note)] = item_copy

            keyboard_key = item_copy.get("keyboard_key")
            if keyboard_key:
                by_key[str(keyboard_key).lower()] = item_copy

        with self.lock:
            self.special_notes_cfg = loaded_cfg
            self.special_notes_by_midi = by_midi
            self.special_notes_by_key = by_key
            self.special_note_active_voices = {}

    def start_special_note(self, special_cfg, velocity=127):
        sample = special_cfg.get("sample")
        if sample is None:
            return

        special_name = special_cfg.get("name")
        midi_note = special_cfg.get("midi_note")
        loop_enabled = bool(special_cfg.get("loop", False))
        stop_on_release = bool(special_cfg.get("stop_on_release", True))
        volume = float(special_cfg.get("volume", 1.0))
        gain = (velocity / 127.0) * volume

        with self.lock:
            if loop_enabled and special_name in self.special_note_active_voices:
                old_voice = self.special_note_active_voices[special_name]
                old_voice.stop()
                del self.special_note_active_voices[special_name]
                print(f"[SPECIAL NOTE {special_name}] LOOP STOP")
                return

            voice = SampleVoice(
                sample.copy(),
                gain,
                midi_note=midi_note,
                stop_on_noteoff=stop_on_release or loop_enabled,
                loop_enabled=loop_enabled,
                special_name=special_name
            )
            self.voices.append(voice)

            if loop_enabled:
                self.special_note_active_voices[special_name] = voice
                print(f"[SPECIAL NOTE {special_name}] LOOP START")
            else:
                print(f"[SPECIAL NOTE {special_name}] START")

    def stop_special_note_by_name(self, special_name):
        with self.lock:
            if special_name in self.special_note_active_voices:
                self.special_note_active_voices[special_name].stop()
                del self.special_note_active_voices[special_name]
                print(f"[SPECIAL NOTE {special_name}] STOP")

            for voice in self.voices:
                if voice.special_name == special_name and voice.stop_on_noteoff:
                    voice.stop()

    def stop_special_note_by_midi(self, midi_note):
        with self.lock:
            for voice in self.voices:
                if voice.midi_note == midi_note and voice.stop_on_noteoff:
                    voice.stop()

            to_remove = []
            for special_name, voice in self.special_note_active_voices.items():
                if voice.midi_note == midi_note:
                    voice.stop()
                    to_remove.append(special_name)

            for special_name in to_remove:
                del self.special_note_active_voices[special_name]
                print(f"[SPECIAL NOTE {special_name}] STOP")

    def update_effects(
        self,
        base_volume=None,
        drive=None,
        output_gain=None,
        delay_ms=None,
        delay_feedback=None,
        delay_mix=None,
        spring_mix=None,
        spring_decay=None,
        spring_tone=None,
        repeater_ms=None,
        repeater_repeats=None,
        repeater_decay=None,
        repeater_mix=None,
    ):
        with self.lock:
            if base_volume is not None:
                self.base_volume = max(0.0, float(base_volume))

            if drive is not None:
                self.drive = max(0.0, float(drive))

            if output_gain is not None:
                self.output_gain = max(0.0, float(output_gain))

            if delay_feedback is not None:
                self.delay_feedback = min(max(0.0, float(delay_feedback)), 0.99)

            if delay_mix is not None:
                self.delay_mix = min(max(0.0, float(delay_mix)), 1.0)

            if delay_ms is not None:
                self.delay_ms = max(0.0, float(delay_ms))

            self.delay_enabled = self.delay_ms > 0 and self.delay_mix > 0
            if self.delay_enabled:
                delay_samples = max(1, int((self.delay_ms / 1000.0) * self.samplerate))
                if self.delay_buffer is None or len(self.delay_buffer) != delay_samples:
                    self.delay_buffer = np.zeros(delay_samples, dtype=np.float32)
                    self.delay_pos = 0
            else:
                self.delay_buffer = None
                self.delay_pos = 0

            if spring_mix is not None:
                self.spring_mix = min(max(0.0, float(spring_mix)), 1.0)

            if spring_decay is not None:
                self.spring_decay = min(max(0.0, float(spring_decay)), 0.99)

            if spring_tone is not None:
                self.spring_tone = min(max(0.0, float(spring_tone)), 1.0)

            self.spring_enabled = self.spring_mix > 0.0
            if self.spring_enabled and not self.spring_buffers:
                self.spring_buffers = [
                    np.zeros(max(1, int(0.020 * self.samplerate)), dtype=np.float32),
                    np.zeros(max(1, int(0.031 * self.samplerate)), dtype=np.float32),
                    np.zeros(max(1, int(0.047 * self.samplerate)), dtype=np.float32),
                ]
                self.spring_pos = [0, 0, 0]
                self.spring_prev = [0.0, 0.0, 0.0]
            elif not self.spring_enabled:
                self.spring_buffers = []
                self.spring_pos = []
                self.spring_prev = []

            if repeater_decay is not None:
                self.repeater_decay = min(max(0.0, float(repeater_decay)), 1.0)

            if repeater_mix is not None:
                self.repeater_mix = min(max(0.0, float(repeater_mix)), 1.0)

            if repeater_repeats is not None:
                self.repeater_repeats = max(0, int(repeater_repeats))

            if repeater_ms is not None:
                self.repeater_ms = max(0.0, float(repeater_ms))

            self.repeater_enabled = (
                self.repeater_ms > 0 and
                self.repeater_repeats > 0 and
                self.repeater_mix > 0
            )

            if self.repeater_enabled:
                self.repeater_delay_samples = max(1, int((self.repeater_ms / 1000.0) * self.samplerate))
            else:
                self.repeater_delay_samples = 0

    def find_loop_by_record(self, channel, note):
        for idx, loop_cfg in enumerate(self.loops_config):
            if loop_cfg["channel"] == channel and loop_cfg["record_note"] == note:
                return idx
        return None

    def find_loop_by_play(self, channel, note):
        for idx, loop_cfg in enumerate(self.loops_config):
            if loop_cfg["channel"] == channel and loop_cfg["play_note"] == note:
                return idx
        return None

    def find_loop_by_cc(self, channel, cc):
        for idx, loop_cfg in enumerate(self.loops_config):
            vol_ch = int(loop_cfg.get("volume_channel", loop_cfg["channel"]))
            if vol_ch == channel and loop_cfg["volume_cc"] == cc:
                return idx
        return None

    def find_loop_by_name(self, loop_name):
        for idx, loop_cfg in enumerate(self.loops_config):
            if str(loop_cfg["name"]).upper() == str(loop_name).upper():
                return idx
        return None

    def set_loop_volume(self, loop_index, midi_value):
        gain = min(max(float(midi_value) / 127.0, 0.0), 1.0)
        with self.lock:
            if 0 <= loop_index < len(self.loops):
                self.loops[loop_index].gain = gain
                print(f"[{self.loops[loop_index].name}] Volume: {gain:.2f}")

    def note_on(self, midi_note, velocity, force_stop_on_noteoff=None):
        special_cfg = self.special_notes_by_midi.get(int(midi_note))
        if special_cfg and special_cfg.get("enabled", False):
            self.start_special_note(special_cfg, velocity=velocity)
            return

        if midi_note not in self.note_cache:
            return

        source = self.note_cache[midi_note].copy()

        if self.repeater_enabled:
            source = self.apply_repeater_to_sample(source)

        gain = (velocity / 127.0) * self.base_volume
        stop_mode = bool(force_stop_on_noteoff) if force_stop_on_noteoff is not None else False
        voice = SampleVoice(source, gain, midi_note=midi_note, stop_on_noteoff=stop_mode)

        with self.lock:
            self.voices.append(voice)

    def note_off(self, midi_note):
        self.stop_special_note_by_midi(midi_note)

        with self.lock:
            for voice in self.voices:
                if voice.midi_note == midi_note and voice.stop_on_noteoff:
                    voice.stop()
                    print(f"[NOTE STOP] {midi_note} ({note_name(midi_note)})")

    def stop_all_notes(self):
        with self.lock:
            for voice in self.voices:
                voice.stop()
            self.voices = []
            self.special_note_active_voices = {}
        print("[PANIC] Tutte le note fermate")

    def _toggle_recording_slot(self, slot):
        if not slot.is_recording:
            slot.clear_recording()
            slot.is_recording = True
            print(f"\n[{slot.name}] Recording START\n")
        else:
            slot.is_recording = False

            if len(slot.record_buffer) == 0:
                slot.sample = None
                slot.pos = 0
                print(f"\n[{slot.name}] Recording STOP - nessun audio registrato\n")
                return

            recorded = np.concatenate(slot.record_buffer).astype(np.float32)

            if len(recorded) < 32:
                slot.sample = None
                slot.pos = 0
                print(f"\n[{slot.name}] Recording STOP - loop troppo corto\n")
                return

            recorded = smooth_loop_edges(recorded, fade_len=256)
            slot.set_sample(recorded)

            duration_sec = len(slot.sample) / self.samplerate
            print(f"\n[{slot.name}] Recording STOP - loop salvato ({duration_sec:.2f} sec)\n")

    def toggle_recording(self, loop_index):
        with self.lock:
            if 0 <= loop_index < len(self.loops):
                self._toggle_recording_slot(self.loops[loop_index])

    def toggle_loop_playback(self, loop_index):
        with self.lock:
            if 0 <= loop_index < len(self.loops):
                slot = self.loops[loop_index]
                ok, msg = slot.toggle_play()
                if not ok:
                    print(f"\n[{slot.name}] {msg}\n")
                else:
                    print(f"\n[{slot.name}] {msg}\n")

    def apply_repeater_to_sample(self, sample):
        dry = sample.astype(np.float32)
        total_length = len(dry) + self.repeater_delay_samples * self.repeater_repeats
        out = np.zeros(total_length, dtype=np.float32)

        out[:len(dry)] += dry * (1.0 - self.repeater_mix)

        for r in range(1, self.repeater_repeats + 1):
            start = r * self.repeater_delay_samples
            gain = (self.repeater_decay ** r) * self.repeater_mix
            end = start + len(dry)
            out[start:end] += dry * gain

        return out.astype(np.float32)

    def apply_distortion(self, audio):
        if self.drive <= 0:
            return audio

        driven = np.tanh(audio * (1.0 + self.drive * 10.0))
        driven *= self.output_gain
        return driven.astype(np.float32)

    def apply_delay(self, audio):
        if not self.delay_enabled:
            return audio

        out = np.copy(audio)
        for i in range(len(audio)):
            delayed = self.delay_buffer[self.delay_pos]
            dry = audio[i]

            out[i] = dry * (1.0 - self.delay_mix) + delayed * self.delay_mix
            self.delay_buffer[self.delay_pos] = dry + delayed * self.delay_feedback

            self.delay_pos += 1
            if self.delay_pos >= len(self.delay_buffer):
                self.delay_pos = 0

        return out.astype(np.float32)

    def apply_spring(self, audio):
        if not self.spring_enabled:
            return audio

        out = np.copy(audio)

        for i in range(len(audio)):
            dry = audio[i]
            spring_sum = 0.0

            for j, buf in enumerate(self.spring_buffers):
                pos = self.spring_pos[j]
                delayed = buf[pos]

                filtered = (delayed * (0.35 + self.spring_tone * 0.65)) + (self.spring_prev[j] * 0.2)
                self.spring_prev[j] = filtered

                spring_sum += filtered
                buf[pos] = dry + filtered * self.spring_decay

                pos += 1
                if pos >= len(buf):
                    pos = 0

                self.spring_pos[j] = pos

            spring_sum /= max(1, len(self.spring_buffers))
            out[i] = dry * (1.0 - self.spring_mix) + spring_sum * self.spring_mix

        return out.astype(np.float32)

    def execute_keyboard_action_press(self, key_str):
        with self.lock:
            if not self.keyboard_enabled:
                return

            matching = [b for b in self.keyboard_bindings if b["key"] == key_str]
            if not matching:
                return

        for binding in matching:
            action = binding["action"]

            if action == "note":
                midi_note = int(binding.get("midi_note", 60))
                velocity = int(binding.get("velocity", 120))
                stop_on_release = bool(binding.get("stop_on_release", False))
                print(f"[KEYBOARD] NOTE ON {key_str} -> {midi_note} ({note_name(midi_note)})")
                self.note_on(midi_note, velocity, force_stop_on_noteoff=stop_on_release)

            elif action == "special_note":
                special_cfg = self.special_notes_by_key.get(key_str)
                if special_cfg and special_cfg.get("enabled", False):
                    print(f"[KEYBOARD] SPECIAL NOTE {key_str} -> {special_cfg.get('name')}")
                    self.start_special_note(special_cfg, velocity=127)

            elif action == "loop_record_toggle":
                loop_name = binding.get("loop_name")
                idx = self.find_loop_by_name(loop_name)
                if idx is not None:
                    print(f"[KEYBOARD] LOOP REC TOGGLE {key_str} -> {loop_name}")
                    self.toggle_recording(idx)

            elif action == "loop_play_toggle":
                loop_name = binding.get("loop_name")
                idx = self.find_loop_by_name(loop_name)
                if idx is not None:
                    print(f"[KEYBOARD] LOOP PLAY TOGGLE {key_str} -> {loop_name}")
                    self.toggle_loop_playback(idx)

            elif action == "master_record_toggle":
                print(f"[KEYBOARD] MASTER REC TOGGLE {key_str}")
                self.toggle_master_recording()

            elif action == "panic":
                print(f"[KEYBOARD] PANIC {key_str}")
                self.stop_all_notes()

    def execute_keyboard_action_release(self, key_str):
        with self.lock:
            if not self.keyboard_enabled:
                return

            matching = [b for b in self.keyboard_bindings if b["key"] == key_str]
            if not matching:
                return

        for binding in matching:
            action = binding["action"]

            if action == "note":
                stop_on_release = bool(binding.get("stop_on_release", False))
                if stop_on_release:
                    midi_note = int(binding.get("midi_note", 60))
                    print(f"[KEYBOARD] NOTE OFF {key_str} -> {midi_note} ({note_name(midi_note)})")
                    self.note_off(midi_note)

            elif action == "special_note":
                special_cfg = self.special_notes_by_key.get(key_str)
                if special_cfg and special_cfg.get("enabled", False):
                    if bool(special_cfg.get("stop_on_release", True)) and not bool(special_cfg.get("loop", False)):
                        print(f"[KEYBOARD] SPECIAL NOTE OFF {key_str} -> {special_cfg.get('name')}")
                        self.stop_special_note_by_name(special_cfg.get("name"))

    def audio_callback(self, outdata, frames, time_info, status):
        if status:
            print(f"[AUDIO STATUS] {status}")

        live_mix = np.zeros(frames, dtype=np.float32)
        loop_mix = np.zeros(frames, dtype=np.float32)
        alive = []

        with self.lock:
            for voice in self.voices:
                chunk, finished = voice.render(frames)
                live_mix += chunk
                if not finished:
                    alive.append(voice)
            self.voices = alive

            for loop_slot in self.loops:
                loop_mix += loop_slot.render(frames)

        processed_live = self.apply_distortion(live_mix)
        processed_live = self.apply_delay(processed_live)
        processed_live = self.apply_spring(processed_live)

        with self.lock:
            for loop_slot in self.loops:
                if loop_slot.is_recording:
                    loop_slot.record_buffer.append(processed_live.copy())

        final_mix = processed_live + loop_mix

        peak = np.max(np.abs(final_mix)) if len(final_mix) else 0.0
        if peak > self.peak_guard:
            final_mix *= (self.peak_guard / peak)

        with self.lock:
            if self.master_record_enabled:
                self.master_record_buffer.append(final_mix.copy())

        outdata[:, 0] = final_mix


def config_watcher(player, config_path, stop_event, threshold, skip_ms):
    last_mtime = None
    last_instrument_file = None
    last_instrument_root = None
    last_special_notes_json = None
    last_loops_json = None
    last_keyboard_json = None
    last_keyboard_enabled = None
    last_master_rec_enabled = None
    last_master_rec_file = None
    last_master_rec_autoname = None
    last_note_min = None
    last_note_max = None

    while not stop_event.is_set():
        try:
            if os.path.isfile(config_path):
                mtime = os.path.getmtime(config_path)

                if last_mtime is None or mtime != last_mtime:
                    last_mtime = mtime
                    cfg = load_config(config_path)

                    player.update_effects(
                        base_volume=cfg.get("base_volume"),
                        drive=cfg.get("drive"),
                        output_gain=cfg.get("output_gain"),
                        delay_ms=cfg.get("delay_ms"),
                        delay_feedback=cfg.get("delay_feedback"),
                        delay_mix=cfg.get("delay_mix"),
                        spring_mix=cfg.get("spring_mix"),
                        spring_decay=cfg.get("spring_decay"),
                        spring_tone=cfg.get("spring_tone"),
                        repeater_ms=cfg.get("repeater_ms"),
                        repeater_repeats=cfg.get("repeater_repeats"),
                        repeater_decay=cfg.get("repeater_decay"),
                        repeater_mix=cfg.get("repeater_mix"),
                    )

                    instrument_file = cfg.get("instrument_file")
                    instrument_root = int(cfg.get("instrument_root_note", DEFAULT_ROOT_NOTE))
                    note_min = int(cfg.get("note_min", NOTE_MIN))
                    note_max = int(cfg.get("note_max", NOTE_MAX))

                    if instrument_file and (
                        instrument_file != last_instrument_file or
                        instrument_root != last_instrument_root or
                        note_min != last_note_min or
                        note_max != last_note_max
                    ):
                        if os.path.isfile(instrument_file):
                            sample, samplerate = load_audio(
                                instrument_file,
                                threshold=threshold,
                                skip_ms=skip_ms
                            )

                            if samplerate != player.samplerate:
                                sample = resample_audio(sample, samplerate, player.samplerate)
                                print(f"[INSTRUMENT] Resampled to {player.samplerate} Hz")

                            player.update_instrument(sample, instrument_root, note_min=note_min, note_max=note_max)
                            last_instrument_file = instrument_file
                            last_instrument_root = instrument_root
                            last_note_min = note_min
                            last_note_max = note_max
                            print(f"[CONFIG] Strumento aggiornato da {instrument_file}")
                        else:
                            print(f"[CONFIG] File strumento non trovato: {instrument_file}")

                    special_notes_json = json.dumps(cfg.get("special_notes", get_default_special_notes()), sort_keys=True)
                    if special_notes_json != last_special_notes_json:
                        player.update_special_notes(cfg, threshold=threshold, skip_ms=skip_ms)
                        last_special_notes_json = special_notes_json

                    loops_json = json.dumps(cfg.get("loops", get_default_loops()), sort_keys=True)
                    if loops_json != last_loops_json:
                        player.update_loops_config(cfg.get("loops", get_default_loops()))
                        last_loops_json = loops_json

                    keyboard_enabled = bool(cfg.get("keyboard_enabled", False))
                    keyboard_json = json.dumps(cfg.get("keyboard_bindings", get_default_keyboard_bindings()), sort_keys=True)
                    if keyboard_enabled != last_keyboard_enabled or keyboard_json != last_keyboard_json:
                        player.update_keyboard_config(keyboard_enabled, cfg.get("keyboard_bindings", get_default_keyboard_bindings()))
                        last_keyboard_enabled = keyboard_enabled
                        last_keyboard_json = keyboard_json

                    master_rec_enabled = bool(cfg.get("master_record_enabled", False))
                    master_rec_file = str(cfg.get("master_record_file", "master_record.wav"))
                    master_rec_autoname = bool(cfg.get("master_record_autoname", True))

                    if (
                        master_rec_enabled != last_master_rec_enabled or
                        master_rec_file != last_master_rec_file or
                        master_rec_autoname != last_master_rec_autoname
                    ):
                        player.update_master_record_config(
                            enabled=master_rec_enabled,
                            filename=master_rec_file,
                            autoname=master_rec_autoname
                        )
                        last_master_rec_enabled = master_rec_enabled
                        last_master_rec_file = master_rec_file
                        last_master_rec_autoname = master_rec_autoname

                    print("\n[CONFIG] Parametri aggiornati da config.json\n")

            time.sleep(0.5)

        except Exception as e:
            print(f"Errore config watcher: {e}")
            time.sleep(1.0)


def print_usage():
    print("Uso:")
    print("  python midi_pitch_player_fx_looper_dual.py file_audio")
    print("  python midi_pitch_player_fx_looper_dual.py file_audio root_note")
    print("  python midi_pitch_player_fx_looper_dual.py file_audio root_note midi_device_id")
    print("  python midi_pitch_player_fx_looper_dual.py file_audio root_note midi_device_id skip_ms")
    print("  python midi_pitch_player_fx_looper_dual.py file_audio root_note midi_device_id skip_ms threshold")
    print()
    print("Nuove funzioni:")
    print("  - tastiera PC configurabile da JSON")
    print("  - funzioni assegnabili a tasti da JSON")
    print("  - master recording da JSON")
    print("  - salvataggio automatico WAV")
    print("  - multiple special notes da JSON")
    print()


def midi_loop(midi_input, player, stop_event):
    while not stop_event.is_set():
        try:
            if midi_input.poll():
                events = midi_input.read(32)

                for event in events:
                    data = event[0]
                    status = data[0]
                    data1 = data[1]
                    data2 = data[2]

                    msg_type = status & 0xF0
                    channel = (status & 0x0F) + 1

                    if msg_type == 0x90 and data2 > 0:
                        note = data1
                        velocity = data2

                        rec_idx = player.find_loop_by_record(channel, note)
                        if rec_idx is not None:
                            loop_name = player.loops[rec_idx].name
                            print(f"{loop_name} REC TOGGLE | Canale:{channel} | Nota:{note} ({note_name(note)})")
                            player.toggle_recording(rec_idx)
                            continue

                        play_idx = player.find_loop_by_play(channel, note)
                        if play_idx is not None:
                            loop_name = player.loops[play_idx].name
                            print(f"{loop_name} PLAY TOGGLE | Canale:{channel} | Nota:{note} ({note_name(note)})")
                            player.toggle_loop_playback(play_idx)
                            continue

                        print(f"NOTE ON  | Canale:{channel} | Nota:{note} ({note_name(note)}) | Velocity:{velocity}")
                        player.note_on(note, velocity)

                    elif msg_type == 0x80 or (msg_type == 0x90 and data2 == 0):
                        note = data1
                        print(f"NOTE OFF | Canale:{channel} | Nota:{note} ({note_name(note)})")
                        player.note_off(note)

                    elif msg_type == 0xB0:
                        cc = data1
                        value = data2

                        print(f"CC       | Canale:{channel} | Controller:{cc} | Value:{value}")

                        cc_idx = player.find_loop_by_cc(channel, cc)
                        if cc_idx is not None:
                            loop_name = player.loops[cc_idx].name
                            print(f"{loop_name} VOLUME | Canale:{channel} | CC:{cc} | Value:{value}")
                            player.set_loop_volume(cc_idx, value)
                            continue

            time.sleep(0.0005)

        except Exception as e:
            print(f"Errore nel loop MIDI: {e}")
            break


def start_keyboard_listener(player, stop_event):
    if not PYNPUT_AVAILABLE:
        print("[KEYBOARD] pynput non disponibile. Installa con: pip install pynput")
        return None

    def on_press(key):
        try:
            key_str = key_to_string(key)
            if not key_str:
                return

            with player.lock:
                if key_str in player.keyboard_pressed_keys:
                    return
                player.keyboard_pressed_keys.add(key_str)

            player.execute_keyboard_action_press(key_str)
        except Exception as e:
            print(f"[KEYBOARD] Errore on_press: {e}")

    def on_release(key):
        try:
            key_str = key_to_string(key)
            if not key_str:
                return

            with player.lock:
                if key_str in player.keyboard_pressed_keys:
                    player.keyboard_pressed_keys.remove(key_str)

            player.execute_keyboard_action_release(key_str)
        except Exception as e:
            print(f"[KEYBOARD] Errore on_release: {e}")

    listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    print("[KEYBOARD] Listener avviato")
    return listener


def main():
    cfg = load_config("config.json")

    if len(sys.argv) < 2:
        print_usage()
        return

    audio_file = sys.argv[1]
    root_note = int(sys.argv[2]) if len(sys.argv) > 2 else int(cfg.get("instrument_root_note", DEFAULT_ROOT_NOTE))
    midi_device_id = int(sys.argv[3]) if len(sys.argv) > 3 else int(cfg.get("midi_device_id", DEFAULT_MIDI_DEVICE_ID))
    skip_ms = float(sys.argv[4]) if len(sys.argv) > 4 else float(cfg.get("skip_ms", DEFAULT_SKIP_MS))
    threshold = float(sys.argv[5]) if len(sys.argv) > 5 else float(cfg.get("threshold", DEFAULT_THRESHOLD))
    note_min = int(cfg.get("note_min", NOTE_MIN))
    note_max = int(cfg.get("note_max", NOTE_MAX))

    if not os.path.isfile(audio_file):
        print(f"Errore: file non trovato -> {audio_file}")
        return

    sample, samplerate = load_audio(
        audio_file,
        threshold=threshold,
        skip_ms=skip_ms
    )

    note_cache = preload_notes(sample, root_note, note_min, note_max)

    player = LowLatencySamplePlayer(
        note_cache=note_cache,
        samplerate=samplerate,
        loops_config=cfg.get("loops", get_default_loops()),
        base_volume=cfg.get("base_volume", BASE_VOLUME),
        drive=cfg.get("drive", DEFAULT_DRIVE),
        output_gain=cfg.get("output_gain", DEFAULT_OUTPUT_GAIN),
        delay_ms=cfg.get("delay_ms", DEFAULT_DELAY_MS),
        delay_feedback=cfg.get("delay_feedback", DEFAULT_DELAY_FEEDBACK),
        delay_mix=cfg.get("delay_mix", DEFAULT_DELAY_MIX),
        spring_mix=cfg.get("spring_mix", DEFAULT_SPRING_MIX),
        spring_decay=cfg.get("spring_decay", DEFAULT_SPRING_DECAY),
        spring_tone=cfg.get("spring_tone", DEFAULT_SPRING_TONE),
        repeater_ms=cfg.get("repeater_ms", DEFAULT_REPEATER_MS),
        repeater_repeats=cfg.get("repeater_repeats", DEFAULT_REPEATER_REPEATS),
        repeater_decay=cfg.get("repeater_decay", DEFAULT_REPEATER_DECAY),
        repeater_mix=cfg.get("repeater_mix", DEFAULT_REPEATER_MIX),
    )

    player.update_special_notes(cfg, threshold=threshold, skip_ms=skip_ms)
    player.update_loops_config(cfg.get("loops", get_default_loops()))
    player.update_keyboard_config(cfg.get("keyboard_enabled", False), cfg.get("keyboard_bindings", get_default_keyboard_bindings()))
    player.update_master_record_config(
        enabled=cfg.get("master_record_enabled", False),
        filename=cfg.get("master_record_file", "master_record.wav"),
        autoname=cfg.get("master_record_autoname", True)
    )

    pygame.init()
    pygame.midi.init()

    midi_input = None
    stream = None
    stop_event = threading.Event()
    midi_thread = None
    config_thread = None
    kb_listener = None

    try:
        list_midi_devices()

        midi_input = pygame.midi.Input(midi_device_id)

        print(f"Dispositivo MIDI aperto: {midi_device_id}")
        print(f"File audio iniziale: {audio_file}")
        print(f"Nota base iniziale: {root_note} ({note_name(root_note)})")
        print(f"Range note: {note_min} - {note_max}")
        print(f"Skip attacco: {skip_ms} ms")
        print(f"Soglia trim silenzio: {threshold}")
        print(f"Sample rate: {samplerate}")
        print()
        print("----- CONFIG ATTUALE -----")
        for key, value in cfg.items():
            print(f"{key}: {value}")
        print("--------------------------")
        print("Looper controls:")
        for loop_cfg in cfg.get("loops", get_default_loops()):
            print(
                f"  {loop_cfg['name']} | CH {loop_cfg['channel']} | "
                f"REC {loop_cfg['record_note']} ({note_name(loop_cfg['record_note'])}) | "
                f"PLAY {loop_cfg['play_note']} ({note_name(loop_cfg['play_note'])}) | "
                f"CC {loop_cfg['volume_cc']} | VOL_CH {loop_cfg.get('volume_channel', loop_cfg['channel'])}"
            )
        print()
        print("Special notes:")
        for sn in cfg.get("special_notes", []):
            print(
                f"  {sn.get('name')} | enabled:{sn.get('enabled')} | "
                f"midi:{sn.get('midi_note')} | key:{sn.get('keyboard_key')} | "
                f"file:{sn.get('file')} | loop:{sn.get('loop')} | "
                f"stop_on_release:{sn.get('stop_on_release')}"
            )
        print()
        print("Keyboard:")
        print(f"  enabled: {cfg.get('keyboard_enabled', False)}")
        print(f"  pynput disponibile: {PYNPUT_AVAILABLE}")
        print()
        print("Master recording:")
        print(f"  enabled: {cfg.get('master_record_enabled', False)}")
        print(f"  file: {cfg.get('master_record_file', 'master_record.wav')}")
        print(f"  autoname: {cfg.get('master_record_autoname', True)}")
        print()
        print("CTRL+C per uscire.\n")

        stream = sd.OutputStream(
            samplerate=samplerate,
            channels=1,
            dtype='float32',
            callback=player.audio_callback,
            blocksize=DEFAULT_BLOCKSIZE,
            latency='low'
        )
        stream.start()

        midi_thread = threading.Thread(
            target=midi_loop,
            args=(midi_input, player, stop_event),
            daemon=True
        )
        midi_thread.start()

        config_thread = threading.Thread(
            target=config_watcher,
            args=(player, "config.json", stop_event, threshold, skip_ms),
            daemon=True
        )
        config_thread.start()

        kb_listener = start_keyboard_listener(player, stop_event)

        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nUscita...")

    except Exception as e:
        print(f"Errore: {e}")

    finally:
        stop_event.set()

        with player.lock:
            if player.master_record_enabled:
                player.stop_master_recording(save_file=True)

        if midi_thread is not None and midi_thread.is_alive():
            midi_thread.join(timeout=1.0)

        if config_thread is not None and config_thread.is_alive():
            config_thread.join(timeout=1.0)

        if kb_listener is not None:
            try:
                kb_listener.stop()
            except Exception:
                pass

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

        if midi_input is not None:
            try:
                midi_input.close()
            except Exception:
                pass

        pygame.midi.quit()
        pygame.quit()


if __name__ == "__main__":
    main()