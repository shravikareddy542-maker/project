"""
Voice/text greeter helper for Leo.
Handles recognized guest greetings only.
"""

import threading
import queue
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class Greeter:
    def __init__(self):
        self.enabled = getattr(config, "ENABLE_GREETING", True)
        self.voice_enabled = getattr(config, "ENABLE_VOICE_GREETING", True) and pyttsx3 is not None

        self._queue = queue.Queue(maxsize=5)
        self._engine = None
        self._worker_started = False
        self._lock = threading.Lock()

        self._last_spoken_text = None
        self._last_spoken_at = 0.0
        self._dedupe_window_sec = getattr(config, "TTS_DEDUPE_WINDOW_SECONDS", 3.0)

        if self.voice_enabled:
            self._start_worker()

    def _start_worker(self):
        if self._worker_started:
            return
        self._worker_started = True
        thread = threading.Thread(target=self._worker, daemon=True)
        thread.start()

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                break

            text = item
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", getattr(config, "TTS_RATE", 165))
                engine.setProperty("volume", getattr(config, "TTS_VOLUME", 1.0))
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as e:
                print(f"[WARN] TTS speak failed: {e}")

    def _should_skip_duplicate(self, text: str) -> bool:
        now = time.time()
        with self._lock:
            if self._last_spoken_text == text and (now - self._last_spoken_at) < self._dedupe_window_sec:
                return True

            self._last_spoken_text = text
            self._last_spoken_at = now
            return False

    def speak(self, text: str):
        if not self.enabled or not text:
            return
        if self._should_skip_duplicate(text):
            return
        if not self.voice_enabled:
            return

        try:
            self._queue.put_nowait(text)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
                self._queue.put_nowait(text)
            except Exception as e:
                print(f"[WARN] TTS queue overflow handling failed: {e}")
        except Exception as e:
            print(f"[WARN] TTS queue failed: {e}")

    def greet_recognized(self, name: str) -> str:
        text = config.GREETING_TEMPLATE.format(name=name)
        self.speak(text)
        return text

    def speak_generated(self, text: str) -> str:
        """Speak arbitrary pre-generated text. Returns the text spoken."""
        if not text:
            return ""
        self.speak(text)
        return text

    def shutdown(self):
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass