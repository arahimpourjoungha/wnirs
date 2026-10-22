"""Reading wNIRS recordings and discovering the sessions of a study.

The entry points:

- :func:`read_brain_games` reads a Brain Games tablet recording (SensorData
  CSV plus its BehavioralData workbook) into a :class:`~wnirs.Raw`, trimming
  runaway recordings to the task's length;
- :func:`read_csv` reads a plain headband dataLog CSV;
- :func:`find_sessions` and :func:`build_manifest` turn a study folder into
  one row per subject, session and task, with the de-duplication rules and
  the tracksheet / participants annotations;
- :func:`behaviour_table` summarises task performance per session.

Not yet supported, because this dataset does not use them: the portal
session report (multi-sheet .xlsx), Hitachi ETG, NIRx, Homer .nirs, SNIRF and
OpenBCI readers, and a format dispatcher. Discovery therefore looks at .csv
recordings only.
"""
from .behaviour import behaviour_table, summarise_trials
from .braingames import TASK_LENGTH, TRIM_FACTOR, find_companion, read_brain_games, read_trials
from .layout import builtin_layouts, load_layout
from .manifest import (build_manifest, data_quality, join_tracksheet, read_participants,
                       session_times)
from .names import BrainGamesName, canonical_task, parse_brain_games_name, parse_name
from .raw import build_raw, conditions_from_stim, conditions_from_trials, read_csv
from .sensor import SensorChannels, SensorTable, all_pairs, parse_sensor, read_sensor_table
from .sessions import find_sessions, has_companion
from .tracksheet import Tracksheet, read_tracksheet

__all__ = [
    "TASK_LENGTH", "TRIM_FACTOR",
    "BrainGamesName", "SensorChannels", "SensorTable", "Tracksheet",
    "all_pairs", "behaviour_table", "build_manifest", "build_raw", "builtin_layouts",
    "canonical_task", "conditions_from_stim", "conditions_from_trials", "data_quality",
    "find_companion", "find_sessions", "has_companion", "join_tracksheet", "load_layout",
    "parse_brain_games_name", "parse_name", "parse_sensor", "read_brain_games", "read_csv",
    "read_participants", "read_sensor_table", "read_tracksheet", "read_trials",
    "session_times", "summarise_trials",
]
