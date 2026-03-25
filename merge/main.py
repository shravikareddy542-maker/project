# file: main.py
from __future__ import annotations

import platform
import re
import sys
from dataclasses import dataclass
from typing import Optional, Protocol

import speech_recognition as sr
import pywhatkit

import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


def _safe_stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def ensure_nltk() -> None:
    packages = [
        "punkt",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "punkt_tab",
        "averaged_perceptron_tagger_eng",
    ]
    for p in packages:
        try:
            nltk.download(p, quiet=True)
        except Exception:
            try:
                nltk.download(p)
            except Exception:
                continue


@dataclass(frozen=True)
class LeoConfig:
    name: str = "Leo"
    wake_word: str = "leo"


class TTSSpeaker(Protocol):
    def speak(self, text: str) -> None: ...


class PrintOnlySpeaker:
    def __init__(self, cfg: LeoConfig) -> None:
        self.cfg = cfg

    def speak(self, text: str) -> None:
        print(f"\n🤖 {self.cfg.name}: {text}\n")


class SapiSpeaker:
    def __init__(self, cfg: LeoConfig) -> None:
        self.cfg = cfg
        from win32com.client import Dispatch  # type: ignore

        self._voice = Dispatch("SAPI.SpVoice")

    def speak(self, text: str) -> None:
        print(f"\n🤖 {self.cfg.name}: {text}\n")
        self._voice.Speak(text)


class Pyttsx3Speaker:
    def __init__(self, cfg: LeoConfig) -> None:
        self.cfg = cfg
        import pyttsx3  # type: ignore

        system = platform.system().lower()
        drivers = ["sapi5", None] if "windows" in system else (["nsss", None] if "darwin" in system else ["espeak", None])

        last_err: Optional[Exception] = None
        for d in drivers:
            try:
                self._engine = pyttsx3.init(driverName=d) if d else pyttsx3.init()
                self._engine.setProperty("rate", 170)
                self._engine.setProperty("volume", 1.0)
                try:
                    voices = self._engine.getProperty("voices")
                    if voices:
                        self._engine.setProperty("voice", voices[0].id)
                except Exception:
                    pass
                return
            except Exception as e:
                last_err = e

        raise RuntimeError(f"pyttsx3 init failed: {last_err}")

    def speak(self, text: str) -> None:
        print(f"\n🤖 {self.cfg.name}: {text}\n")
        try:
            if self._engine.isBusy():
                self._engine.stop()
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception:
            return


def build_speaker(cfg: LeoConfig) -> TTSSpeaker:
    system = platform.system().lower()

    if "windows" in system:
        try:
            return SapiSpeaker(cfg)
        except Exception:
            pass

    try:
        return Pyttsx3Speaker(cfg)
    except Exception:
        return PrintOnlySpeaker(cfg)


def listen_once(recognizer: sr.Recognizer, mic: sr.Microphone) -> Optional[str]:
    with mic as source:
        try:
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=8)
        except sr.WaitTimeoutError:
            return None

    try:
        return recognizer.recognize_google(audio).strip()
    except (sr.UnknownValueError, sr.RequestError):
        return None


def should_exit(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in ["stop", "exit", "quit", "close", "bye"])


def is_intro_intent(text: str) -> bool:
    t = text.lower().strip()
    name_intent = ("name" in t) and any(k in t for k in ["your", "you", "what", "tell", "who"])
    about_intent = any(
        p in t
        for p in [
            "who are you",
            "tell me about yourself",
            "tell about yourself",
            "introduce yourself",
            "about yourself",
            "self intro",
            "self-intro",
        ]
    )
    return name_intent or about_intent


def intro_response(text: str, cfg: LeoConfig) -> str:
    t = text.lower().strip()

    if ("name" in t) and any(k in t for k in ["your", "you", "what", "tell", "who"]):
        return f"Yo fam, I'm {cfg.name} — that's the vibe!"

    if any(
        p in t
        for p in [
            "tell me about yourself",
            "tell about yourself",
            "who are you",
            "introduce yourself",
            "about yourself",
            "self intro",
            "self-intro",
        ]
    ):
        return (
            f"Hiiiii! I'm {cfg.name}, your friendly humanoid robot buddy! "
            "I love talking with you and playing your favorite songs anytime you ask. "
            "If you ever need help, just call my name — I'll be right here with a big robot smile, ready to assist you!"
        )

    return "Sorry bestie, I didn't catch that. Try again!"


def penn_to_wn(tag: str):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


class LeoNLU:
    def __init__(self, wake_word: str = "leo") -> None:
        self.wake = wake_word.lower()
        self.lemmatizer = WordNetLemmatizer()

        base = set(stopwords.words("english"))
        extra = {
            "please",
            "kindly",
            "hey",
            "hi",
            "hello",
            "can",
            "could",
            "would",
            "will",
            "you",
            "me",
            "my",
            "your",
            "now",
            "just",
            "then",
            "ok",
            "okay",
        }
        self.stops = (base | extra) - {"play"}

        self.generic_music = {"song", "music", "video", "track", "youtube", "yt"}
        self.play_words = {"play", "start", "put"}
        self.stop_intent = {"stop", "exit", "quit", "close"}

    def normalize(self, text: str) -> list[str]:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s']+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        toks = word_tokenize(text)
        tagged = pos_tag(toks)

        lemmas: list[str] = []
        for w, t in tagged:
            lemmas.append(self.lemmatizer.lemmatize(w, penn_to_wn(t)))

        lemmas = [w for w in lemmas if w != self.wake]
        return [w for w in lemmas if w not in self.stops]

    def parse_play_query(self, raw: str) -> Optional[str]:
        raw_l = raw.lower()
        tokens = self.normalize(raw)
        if not tokens:
            return ""

        if any(w in self.stop_intent for w in tokens):
            return "__STOP__"

        play_idx = None
        for i, w in enumerate(tokens):
            if w in self.play_words:
                play_idx = i
                break
        if play_idx is None and "play" in tokens:
            play_idx = tokens.index("play")

        if self.wake not in raw_l:
            if play_idx is None or play_idx != 0:
                return None

        if play_idx is None:
            return ""

        query_tokens = tokens[play_idx + 1 :]
        if len(query_tokens) >= 2:
            query_tokens = [w for w in query_tokens if w not in self.generic_music]
        return " ".join(query_tokens).strip()


def main() -> None:
    _safe_stdout_utf8()
    ensure_nltk()

    cfg = LeoConfig(name="Leo", wake_word="leo")
    speaker = build_speaker(cfg)
    nlu = LeoNLU(cfg.wake_word)

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.2)

    while True:
        try:
            print("🎤 Listening…")
            heard = listen_once(recognizer, mic)
            if not heard:
                continue

            print(f"👤 You said: {heard}")

            if should_exit(heard):
                speaker.speak("Bye!")
                break

            if is_intro_intent(heard):
                speaker.speak(intro_response(heard, cfg))
                continue

            play_q = nlu.parse_play_query(heard)

            if play_q is None:
                continue

            if play_q == "__STOP__":
                speaker.speak("Bye!")
                break

            if play_q:
                speaker.speak(f"Playing {play_q}")
                pywhatkit.playonyt(play_q)

        except KeyboardInterrupt:
            speaker.speak("Bye!")
            break


if __name__ == "__main__":
    main()