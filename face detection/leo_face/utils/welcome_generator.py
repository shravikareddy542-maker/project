"""
Welcome Generator – Deterministic, formal guest introduction builder for Leo.

Produces ~1 minute spoken introductions (~130-150 words at TTS pace).
Each introduction is unique based on the guest's specific profile data.

Rules enforced:
  • Uses only DB values: name, designation, achievements
  • No gender inference, no sensitive attributes
  • Neutral, respectful, formal language
  • Deterministic template logic – no LLM, no external APIs
  • Varies structure based on available data for uniqueness
"""

import re
import html
import sqlite3
import hashlib
from dataclasses import dataclass
from typing import Optional


@dataclass
class GuestProfile:
    guest_id: int
    name: str
    designation: str
    achievements: str


@dataclass
class WelcomeMessage:
    on_screen_text: str   # Full display text for the GUI
    spoken_text: str       # ~1 min spoken text for TTS


class WelcomeGenerator:
    """Builds formal, unique, ~1-minute introductions from DB-sourced guest data."""

    MAX_ACHIEVEMENTS_USED = 4

    # ── opening lines (rotated by guest_id hash) ────────────
    _OPENINGS = [
        (
            "Distinguished guests, it is my honour and privilege to present "
            "a truly remarkable individual to you today. "
            "Every gathering is enriched by the presence of those who have "
            "dedicated themselves to excellence, and today is no exception."
        ),
        (
            "Ladies and gentlemen, allow me to introduce an esteemed individual "
            "whose contributions speak volumes about dedication and vision. "
            "We are fortunate to be in the company of someone whose work "
            "has left a meaningful mark."
        ),
        (
            "It is with great respect and admiration that I present to you "
            "a distinguished personality whose journey serves as an inspiration "
            "to many. Occasions like this become truly special when graced by "
            "individuals of such calibre."
        ),
        (
            "Dear guests, it is my pleasure to bring to your attention someone "
            "whose work has made a lasting impact on the world around us. "
            "There are few individuals whose presence alone elevates an "
            "occasion, and today we are privileged to have one such person."
        ),
        (
            "We are honoured to have among us today a truly accomplished "
            "individual who embodies the very essence of dedication, commitment, "
            "and professional excellence. It is occasions like these that "
            "remind us of the power of human endeavour."
        ),
        (
            "Allow me the privilege of introducing someone whose dedication "
            "and excellence are widely recognised, and whose journey continues "
            "to inspire those around them. We are gathered here today in the "
            "presence of a truly distinguished individual."
        ),
    ]

    # ── designation phrasing (rotated) ──────────────────────
    _DESIG_PHRASES = [
        (
            "{name} holds the distinguished position of {desig}. "
            "This role is a testament to years of unwavering commitment, deep expertise, "
            "and a relentless pursuit of excellence in the chosen field of work."
        ),
        (
            "Currently serving as {desig}, {name} brings remarkable knowledge, "
            "leadership, and a wealth of experience to this important role. "
            "It is a position that reflects both professional stature and earned trust."
        ),
        (
            "In the esteemed capacity of {desig}, {name} has demonstrated outstanding "
            "dedication and professionalism, earning the admiration and respect of peers, "
            "colleagues, and the wider professional community."
        ),
        (
            "{name} serves with distinction as {desig}, contributing meaningfully "
            "to the advancement of the field. This role represents a career built on "
            "integrity, hard work, and an enduring passion for making a difference."
        ),
    ]

    # ── no-designation filler ───────────────────────────────
    _NO_DESIG = [
        (
            "Today, we are pleased to welcome {name}, an individual whose presence "
            "adds great value to this occasion. While titles and designations may "
            "often define introductions, the true measure of a person lies in the "
            "impact of their work and character."
        ),
        (
            "{name} joins us today as someone whose quiet contributions have "
            "made a meaningful difference. Not all achievements are measured by "
            "titles alone, and the presence of {name} here today is a reminder "
            "of that important truth."
        ),
    ]

    # ── achievement lead-ins (rotated) ──────────────────────
    _ACH_LEADINS = [
        "Among the many notable accomplishments that define this remarkable journey,",
        "The impressive and inspiring list of contributions includes",
        "The remarkable body of work that has been built over the years encompasses",
        "The professional journey is marked by several significant milestones and achievements, including",
    ]

    # ── single achievement elaboration patterns ─────────────
    _ACH_ELABORATE = [
        (
            "{ach}. This achievement stands as a powerful testament to sustained "
            "excellence, unwavering dedication, and the ability to create lasting impact."
        ),
        (
            "{ach}. This accomplishment reflects a deep commitment to pushing "
            "boundaries, challenging the status quo, and achieving remarkable distinction."
        ),
        (
            "{ach}. This is a milestone that highlights exceptional talent, creative "
            "vision, and the perseverance required to turn ambitious goals into reality."
        ),
    ]

    # ── subsequent achievement connectors ───────────────────
    _ACH_CONNECTORS = [
        "Furthermore, the record includes ",
        "In addition to this notable accomplishment, we must also recognise ",
        "Adding to this already impressive portfolio of achievements is ",
        "Equally noteworthy and deserving of recognition is ",
    ]

    # ── closing lines (rotated) ─────────────────────────────
    _CLOSINGS_FULL = [
        (
            "We are truly privileged to have {name} with us today. The contributions "
            "made to date are a source of pride and inspiration, and we look forward "
            "to the continued impact of such distinguished work. Please join me in "
            "extending a warm and respectful welcome."
        ),
        (
            "The presence of {name} among us today is indeed an honour. Let us take a "
            "moment to acknowledge and appreciate the remarkable journey, and welcome "
            "this distinguished individual with the respect and admiration that such "
            "accomplishments truly deserve."
        ),
        (
            "It is individuals like {name} who inspire us all through their dedication, "
            "hard work, and relentless pursuit of excellence. On behalf of everyone "
            "gathered here, please join me in offering a heartfelt and resounding welcome."
        ),
    ]

    _CLOSINGS_MINIMAL = [
        (
            "We are truly delighted to have {name} with us on this occasion. "
            "Every individual brings a unique perspective and energy, and we are "
            "richer for this presence among us today. Please join me in extending "
            "a warm and respectful welcome to our esteemed guest."
        ),
        (
            "It is a genuine pleasure to welcome {name} to this gathering. "
            "The warmth and distinction that each guest brings makes occasions "
            "like this truly memorable. Let us greet our guest with the respect "
            "and appreciation that is well deserved."
        ),
        (
            "On behalf of everyone present here today, I extend a sincere and "
            "heartfelt welcome to {name}. We are honoured and enriched by this "
            "presence, and we trust that today's gathering will be a meaningful "
            "and enjoyable experience for all."
        ),
    ]

    # ── sanitisation ────────────────────────────────────────

    @staticmethod
    def sanitize(text: str) -> str:
        """Strip HTML, excess whitespace, and non-printable characters."""
        if not text:
            return ""
        text = html.unescape(text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[^\x20-\x7E\w\s.,;:!?()\-–—'\"&]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ── deterministic rotation ──────────────────────────────

    @staticmethod
    def _pick(options: list, seed: int):
        """Deterministically pick from a list using a seed."""
        return options[seed % len(options)]

    @staticmethod
    def _seed_from(name: str, guest_id: int) -> int:
        """Create a deterministic seed from guest data for unique selection."""
        raw = f"{guest_id}:{name}"
        return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)

    # ── achievement parsing ─────────────────────────────────

    @classmethod
    def _parse_achievements(cls, raw: str) -> list[str]:
        """Split comma/semicolon/newline-delimited achievements."""
        if not raw:
            return []
        parts = re.split(r"[;\n]+", raw)
        cleaned = [cls.sanitize(p) for p in parts if p.strip()]
        return cleaned[: cls.MAX_ACHIEVEMENTS_USED]

    # ── text generation ─────────────────────────────────────

    def generate(self, profile: GuestProfile) -> WelcomeMessage:
        """Build a formal ~1 minute introduction. Purely deterministic."""
        name = self.sanitize(profile.name) or "our esteemed guest"
        designation = self.sanitize(profile.designation)
        achievements = self._parse_achievements(profile.achievements)
        seed = self._seed_from(name, profile.guest_id)

        paragraphs: list[str] = []

        # ── paragraph 1: opening ───────────────────────────
        paragraphs.append(self._pick(self._OPENINGS, seed))

        # ── paragraph 2: name + designation ────────────────
        if designation:
            paragraphs.append(
                self._pick(self._DESIG_PHRASES, seed + 1).format(
                    name=name, desig=designation
                )
            )
        else:
            paragraphs.append(
                self._pick(self._NO_DESIG, seed + 1).format(name=name)
            )

        # ── paragraph 3: achievements ──────────────────────
        if achievements:
            leadin = self._pick(self._ACH_LEADINS, seed + 2)
            first_elab = self._pick(self._ACH_ELABORATE, seed + 3).format(
                ach=achievements[0]
            )
            ach_parts = [f"{leadin} {first_elab}"]

            for i, ach in enumerate(achievements[1:], start=1):
                connector = self._pick(self._ACH_CONNECTORS, seed + 3 + i)
                ach_parts.append(f"{connector}{ach}.")

            paragraphs.append(" ".join(ach_parts))

        # ── paragraph 4: closing ───────────────────────────
        if designation or achievements:
            paragraphs.append(
                self._pick(self._CLOSINGS_FULL, seed + 7).format(name=name)
            )
        else:
            paragraphs.append(
                self._pick(self._CLOSINGS_MINIMAL, seed + 7).format(name=name)
            )

        # ── assemble ───────────────────────────────────────
        spoken_text = " ".join(paragraphs)
        on_screen_text = "\n\n".join(paragraphs)

        return WelcomeMessage(
            on_screen_text=on_screen_text,
            spoken_text=spoken_text,
        )


# ── DB helper ──────────────────────────────────────────────

def load_guest_profile(
    conn: sqlite3.Connection, guest_id: int
) -> Optional[GuestProfile]:
    """Fetch a guest from the database and return a GuestProfile, or None."""
    row = conn.execute(
        "SELECT guest_id, name, designation, achievements "
        "FROM guests WHERE guest_id = ?",
        (guest_id,),
    ).fetchone()
    if row is None:
        return None
    if isinstance(row, sqlite3.Row):
        return GuestProfile(
            guest_id=row["guest_id"],
            name=row["name"],
            designation=row["designation"] or "",
            achievements=row["achievements"] or "",
        )
    return GuestProfile(
        guest_id=row[0],
        name=row[1],
        designation=row[2] or "",
        achievements=row[3] or "",
    )
