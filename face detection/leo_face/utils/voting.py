"""
Per-track frame voting and greeting cooldown logic.
Ensures stable, flicker-free name assignment.
"""
import time
from collections import defaultdict, deque

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class TrackVoter:
    """
    Maintains per-track_id sliding window of recognition results.
    A name is *confirmed* only when the same name appears >= VOTE_REQUIRED
    times in the last VOTE_WINDOW frames.
    """

    def __init__(self,
                 window: int = None,
                 required: int = None):
        self.window = window or config.VOTE_WINDOW
        self.required = required or config.VOTE_REQUIRED
        # track_id → deque of (name, score)
        self._votes: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.window)
        )
        # track_id → confirmed (name, avg_score) or None
        self._confirmed: dict[int, tuple[str, float] | None] = {}

    def cast_vote(self, track_id: int, name: str, score: float):
        self._votes[track_id].append((name, score))
        self._evaluate(track_id)

    def _evaluate(self, track_id: int):
        votes = self._votes[track_id]
        if len(votes) < self.required:
            return
        # Count name occurrences in the window
        name_counts: dict[str, list[float]] = defaultdict(list)
        for n, s in votes:
            name_counts[n].append(s)
        for name, scores in name_counts.items():
            if len(scores) >= self.required:
                avg = sum(scores) / len(scores)
                self._confirmed[track_id] = (name, avg)
                return
        # No name reached the threshold yet – keep old confirmation if any

    def get_confirmed(self, track_id: int) -> tuple[str, float] | None:
        return self._confirmed.get(track_id)

    def remove_track(self, track_id: int):
        self._votes.pop(track_id, None)
        self._confirmed.pop(track_id, None)

    def cleanup_stale(self, active_ids: set[int]):
        """Remove data for tracks no longer active."""
        stale = [tid for tid in self._votes if tid not in active_ids]
        for tid in stale:
            self.remove_track(tid)


class GreetCooldown:
    """
    Presence-aware greeting cooldown.

    Rules:
      1. First time a guest is seen → greet.
      2. Guest leaves (absent > GUEST_ABSENCE_THRESHOLD_SEC) and comes back
         → greet again immediately.
      3. Guest stays continuously in frame → re-greet only after
         GREET_COOLDOWN_SECONDS (10 min) since the last greet.
    """

    def __init__(self, cooldown_sec: float = None, absence_sec: float = None):
        self.cooldown = cooldown_sec or config.GREET_COOLDOWN_SECONDS
        self.absence_threshold = absence_sec or getattr(
            config, "GUEST_ABSENCE_THRESHOLD_SEC", 5
        )
        # guest_id → last time we SAW them (updated every frame)
        self._last_seen: dict[int, float] = {}
        # guest_id → last time we GREETED them
        self._last_greet: dict[int, float] = {}

    def mark_seen(self, guest_id: int):
        """Call every frame for every recognized guest."""
        self._last_seen[guest_id] = time.time()

    def should_greet(self, guest_id: int) -> bool:
        now = time.time()
        last_seen = self._last_seen.get(guest_id)
        last_greet = self._last_greet.get(guest_id)

        # 1. Never greeted → greet
        if last_greet is None:
            return True

        # 2. Was absent long enough → treat as new appearance → greet
        if last_seen is not None:
            gap = now - last_seen
            if gap >= self.absence_threshold:
                return True

        # 3. Continuously present → re-greet after cooldown (10 min)
        if (now - last_greet) >= self.cooldown:
            return True

        return False

    def mark_greeted(self, guest_id: int):
        self._last_greet[guest_id] = time.time()

    def reset(self):
        self._last_seen.clear()
        self._last_greet.clear()
