"""What a recording's filename says about it."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional


@dataclass(frozen=True)
class BrainGamesName:
    """Fields of a tablet filename, ``<Task>_<Kind>_Session<N>_<Subject>_<YYYYMMDD>_<HHMMSS>``.

    Attributes
    ----------
    task : str
        Canonical paradigm (``NBack``, ``Flanker``, ``GoNogo``, ``Rest``), or
        ``''`` when the word is not recognised.
    app_task : str
        The word the tablet wrote (``NBack``, ``Fish``, ``SkipX``, ``Rest``).
        Needed to find the BehavioralData sibling, which is named with the
        app's word (``Fish_BehavioralData_*``, never ``Flanker_...``).
    kind : str
        ``'sensor'``, ``'behaviour'`` or ``''``.
    session : str
        The number in the filename. It comes from an app counter that does
        not always advance, so it is not an identifier on its own: the
        recording time is what separates two "Session4" files.
    subject : str
        The id typed at setup on the tablet. A typo lands in the wrong
        person's file, which is why discovery prefers the enclosing folder.
    recorded_at : datetime or None
        When the file was written.
    unfinished : bool
        The app appended ``_unfinished`` because it did not close the file
        cleanly. On this hardware that says nothing about the length.
    """
    task: str = ""
    app_task: str = ""
    kind: str = ""
    session: str = ""
    subject: str = ""
    recorded_at: Optional[datetime] = None
    unfinished: bool = False


def canonical_task(word: str) -> str:
    """Map any of a task's names (paradigm, tablet word, portal sheet) to the paradigm."""
    w = word.strip().lower()
    if "nback" in w or "n-back" in w:
        return "NBack"
    if "fish" in w or "flanker" in w:
        return "Flanker"
    if "skipx" in w or "gonogo" in w or "skip" in w:
        return "GoNogo"
    if "rest" in w:
        return "Rest"
    return ""


def parse_brain_games_name(stem: str) -> BrainGamesName:
    """Read the fields out of a Brain Games tablet filename (without extension).

    The subject is taken as whatever sits between the session number and the
    trailing date and time, rather than positionally: an id that itself holds
    an underscore (``testuser_0327``) would otherwise shift every field and
    the six-digit time would be read as the subject.
    """
    unfinished = re.search(r"_unfinished\s*$", stem, re.IGNORECASE) is not None
    s = re.sub(r"_unfinished\s*$", "", stem, flags=re.IGNORECASE)
    parts = s.split("_")
    if len(parts) < 2:
        return BrainGamesName(unfinished=unfinished)

    kind_word = parts[1].lower()
    kind = "sensor" if "sensor" in kind_word else "behaviour" if "behav" in kind_word else ""

    m = re.search(r"Session\s*([0-9]+)", s, re.IGNORECASE)
    session = m.group(1) if m else ""

    subject = ""
    for pattern in (r"Session\s*\d+_(.+?)_20\d{6}_\d{6}$",
                    r"Session\s*\d+_(.+?)_20\d{6}$",
                    r"Session\s*\d+_(.+)$"):
        m = re.search(pattern, s)
        if m:
            subject = m.group(1).strip()
            break

    return BrainGamesName(task=canonical_task(parts[0]), app_task=parts[0].strip(), kind=kind,
                          session=session, subject=subject, recorded_at=_timestamp(s),
                          unfinished=unfinished)


def _timestamp(s: str) -> Optional[datetime]:
    m = re.search(r"(20\d{6})_(\d{6})\s*$", s)
    try:
        if m:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        m = re.search(r"(20\d{6})\s*$", s)
        if m:
            return datetime.strptime(m.group(1), "%Y%m%d")
    except ValueError:
        pass
    return None


def parse_name(stem: str, which: Literal["subject", "session", "task"]) -> str:
    """Subject, session or task from a generic recording filename.

    Recognises ``90606_session_1_Report``, ``wNIRS_90599_BH1`` and
    ``dataLog_NBack``. Returns a fallback rather than raising, so an unusual
    filename never stops a batch: the whole stem for a subject, ``'1'`` for a
    session, ``'task'`` for a task.
    """
    if which == "subject":
        m = re.search(r"^\s*(\d{3,})", stem) or re.search(r"(\d{3,})", stem)
        return m.group(1) if m else stem
    if which == "session":
        m = (re.search(r"session[_\-\s]*(\d+)", stem, re.IGNORECASE)
             or re.search(r"(?:^|[_\-])[sS](\d+)(?:[_\-]|$)", stem))
        return m.group(1) if m else "1"
    if which == "task":
        m = re.search(r"(BH\d*|NBack|N-Back|Flanker|GoNogo|SkipX|Rest|MIST)", stem, re.IGNORECASE)
        return m.group(1) if m else "task"
    raise ValueError(f'Unknown field "{which}".')
