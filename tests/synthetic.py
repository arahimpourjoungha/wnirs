"""Synthetic fNIRS recordings with known ground truth, for tests and fixtures.

Nothing here is real participant data. The preprocessing generators build a
recording from the forward model the pipeline inverts: block-design HRFs on
the long channels, a systemic oscillation shared by every channel and carried
alone by the short channels, then optical density via Beer-Lambert, then
intensity on top of the detector dark level, with motion spikes, a baseline
shift and the saturation cases the channel flagging has to tell apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats

from wnirs.analysis import SessionMeasure
from wnirs.preproc import dpf, extinction
from wnirs.types import Conditions, Hb, Raw

DARK = np.array([1.511, 1.507])          # zero-light reading of the headband detector
SAT_LEVEL = 5.0


def canonical_hrf(fs: float, length: float = 32.0) -> np.ndarray:
    """Double-gamma HRF (peak ~5 s, undershoot ~15 s), unit peak."""
    t = np.arange(0.0, length, 1.0 / fs)
    h = stats.gamma.pdf(t, 6) - stats.gamma.pdf(t, 16) / 6
    return h / h.max()


def boxcar(time: np.ndarray, onsets, duration: float) -> np.ndarray:
    on = np.zeros(time.size)
    for t0 in onsets:
        on[(time >= t0) & (time < t0 + duration)] = 1.0
    return on


def full_layout() -> pd.DataFrame:
    """The 36-channel wNIRS montage (columns S, D, C, Row, Column, Type, Comparison)."""
    with resources.as_file(resources.files("wnirs") / "data" / "wnirs_layout_full.csv") as f:
        return pd.read_csv(f)


def forward_od(hbo: np.ndarray, hbr: np.ndarray, wavelengths=(750.0, 860.0),
               sd_distance: float = 30.0, age: float = 28.0) -> np.ndarray:
    """Optical density ``(T, C, W)`` produced by concentrations in micromolar."""
    lam = np.asarray(wavelengths, dtype=float)
    E = extinction(lam) * ((sd_distance / 10) * dpf(age, lam))[:, None]
    return np.stack([hbo, hbr], axis=-1) * 1e-6 @ E.T


@dataclass
class SyntheticRecording:
    raw: Raw
    hbo_brain: np.ndarray          # (T, C) true cortical HbO, uM
    hbr_brain: np.ndarray
    systemic: np.ndarray           # (T,) shared systemic HbO, uM
    task: np.ndarray               # (T,) HRF-convolved block regressor, unit peak
    spike_channel: int
    spike_time: float
    shift_channel: int
    expected_status: np.ndarray


def make_recording(seed: int = 0, fs: float = 5.0, duration: float = 200.0) -> SyntheticRecording:
    """A 13-channel recording on sources 1, 2 and 4 of the wNIRS montage.

    Fifteen-second task blocks start every 40 s. A motion spike sits at half
    the duration and a baseline shift at 70 % of it.

    Sources 1 and 2 each carry a short channel; source 4 has none, so
    same-source SSR has to fall back to the mean short there. Channel roles:

    ====  =========  ======================================================
    idx   channel    role
    ====  =========  ======================================================
    1     S1-D9      motion spike (``spike_time``)
    2     S1-D12     baseline shift
    8     S2-D15     intermittently clipped (repaired, kept)
    7     S2-D16     clipped most of the time (saturated, dropped)
    11    S4-D13     close to the ceiling without clipping (nearCeiling)
    12    S4-D14     no light above the dark level (dark)
    ====  =========  ======================================================
    """
    rng = np.random.default_rng(seed)
    layout = full_layout()
    layout = layout[layout["S"].isin([1, 2, 4])].reset_index(drop=True)
    C = len(layout)
    is_short = layout["Type"].to_numpy() == 2

    T = int(round(duration * fs))
    time = np.arange(T) / fs
    onsets = np.arange(30.0, duration - 40, 40.0)
    task = np.convolve(boxcar(time, onsets, 15.0), canonical_hrf(fs))[:T]
    task /= task.max()

    gain = np.where(is_short, 0.0, rng.uniform(0.6, 1.4, C))
    hbo_brain = np.outer(task, gain)
    hbr_brain = -0.3 * hbo_brain
    systemic = (0.8 * np.sin(2 * np.pi * 0.07 * time + 0.3)
                + 0.4 * np.sin(2 * np.pi * 0.03 * time + 1.1))
    coupling = rng.uniform(0.7, 1.3, C)
    hbo = hbo_brain + np.outer(systemic, coupling) + 0.05 * rng.standard_normal((T, C))
    hbr = hbr_brain - 0.25 * np.outer(systemic, coupling) + 0.03 * rng.standard_normal((T, C))
    cardiac = 0.003 * np.sin(2 * np.pi * 1.1 * time)

    od = forward_od(hbo, hbr) + cardiac[:, None, None]
    light = rng.uniform(0.3, 0.9, (C, 2))
    light[11] = 3.1                         # nearCeiling: mean about 4.6
    light[8] = 3.4                          # intermittentClip: peaks reach 5
    light[7] = 3.6                          # saturated
    light[12] = 0.0                         # dark
    ambient = 0.004 * np.sin(2 * np.pi * 0.005 * time) + 0.002 * rng.standard_normal(T)

    spike_channel, shift_channel = 1, 2
    spike_time = 0.5 * duration
    od[:, spike_channel] += 0.4 * np.exp(-0.5 * ((time - spike_time) / 0.5) ** 2)[:, None]
    od[time >= 0.7 * duration, shift_channel] += 0.15

    intensity = light * 10.0 ** (-od) + DARK + ambient[:, None, None]
    intensity += 0.001 * rng.standard_normal(intensity.shape)
    intensity[:, 12] = DARK - 0.004 + 0.002 * rng.standard_normal((T, 2))
    intensity = np.minimum(intensity, SAT_LEVEL)

    reference = 0.1 * light * 10.0 ** (-od) + DARK + ambient[:, None, None]
    reference += 0.001 * rng.standard_normal(reference.shape)

    all_pairs, pair_index = _all_pairs(layout, intensity, rng)
    status = np.full(C, "ok", dtype=object)
    status[[7, 8, 11, 12]] = ["saturated", "intermittentClip", "nearCeiling", "dark"]

    raw = Raw(subject="synthetic", task="blocks", session="1", file=Path("synthetic.csv"),
              time=time, fs=fs, intensity=intensity, layout=layout, wavelengths=(750.0, 860.0),
              stim=boxcar(time, onsets, 15.0), reference=reference, all_pairs=all_pairs,
              all_pairs_index=pair_index,
              conditions=Conditions(["task"], [onsets], [np.full(onsets.size, 15.0)]))
    return SyntheticRecording(raw, hbo_brain, hbr_brain, systemic, task,
                              spike_channel, spike_time, shift_channel, status)


def _all_pairs(layout: pd.DataFrame, intensity: np.ndarray,
               rng: np.random.Generator) -> tuple[np.ndarray, pd.DataFrame]:
    """Montage channels plus ten never-illuminated pairs, one of them catching
    stray light from a neighbouring optode."""
    unused_sd = np.array([[8, d] for d in range(1, 11)])
    T = intensity.shape[0]
    unused = DARK + 0.002 * rng.standard_normal((T, len(unused_sd), 2))
    unused[:, 0] += 0.3
    pairs = np.concatenate([intensity, unused], axis=1)
    index = pd.DataFrame(np.vstack([layout[["S", "D"]].to_numpy(), unused_sd]), columns=["S", "D"])
    index["in_layout"] = np.arange(len(index)) < len(layout)
    return pairs, index


def motion_signals(seed: int = 1, fs: float = 10.0,
                   duration: float = 120.0) -> tuple[np.ndarray, np.ndarray]:
    """Four OD-scale traces: clean, spikes, steps, spikes and steps.

    Returns ``(x, clean)``, both ``(T, 4)``; ``clean`` is the slow
    haemodynamic-like part plus the same white noise, so a perfect correction
    returns ``clean``.
    """
    rng = np.random.default_rng(seed)
    T = int(round(duration * fs))
    t = np.arange(T) / fs
    slow = 0.02 * np.sin(2 * np.pi * 0.02 * t) + 0.01 * np.sin(2 * np.pi * 0.07 * t + 1)
    clean = slow[:, None] + 0.002 * rng.standard_normal((T, 4))
    spikes = np.zeros(T)
    for s in (0.15, 0.4, 0.62, 0.85):
        i = int(s * T)
        spikes[i:i + 6] += 0.08 * np.hanning(6)
    steps = 0.05 * (t >= 0.3 * duration) - 0.03 * (t >= 0.7 * duration)
    x = clean.copy()
    x[:, 1] += spikes
    x[:, 2] += steps
    x[:, 3] += spikes + steps
    return x, clean


# ---------------------------------------------------------------------------
# Analysis-stage generators: Hb recordings for the GLM and connectivity, and
# per-session maps with known structure for the reliability family.
# ---------------------------------------------------------------------------
def analysis_layout(n_channels: int, n_short: int = 0) -> pd.DataFrame:
    """A minimal layout: channel k is S(k//4+1)-D(k+1); the last ``n_short`` are shorts."""
    k = np.arange(n_channels)
    return pd.DataFrame({"S": k // 4 + 1, "D": k + 1, "C": k + 1, "Row": k // 6 + 1, "Column": k % 6 + 1,
                         "Type": np.where(k >= n_channels - n_short, 2, 1)})


def ar1_noise(rng: np.random.Generator, n: int, n_channels: int, rho: float) -> np.ndarray:
    """Unit-variance AR(1) noise, (n, n_channels)."""
    e = signal.lfilter([1.0], [1.0, -rho], rng.standard_normal((n + 200, n_channels)), axis=0)[200:]
    return e / e.std(axis=0)


def _hb(time, fs, hbo, hbr, layout, conditions, subject="sim", task="sim") -> Hb:
    return Hb(subject=subject, task=task, session="1", time=time, fs=fs, hbo=hbo, hbr=hbr, layout=layout,
              conditions=conditions, channel_status=np.full(layout.shape[0], "ok", dtype=object))


def null_recording(seed: int, fs: float = 7.0, duration: float = 300.0) -> Hb:
    """One channel of realistic null fNIRS noise with a rapid two-condition design.

    Strongly autocorrelated noise (AR(1), rho 0.95) plus a slow oscillation,
    and no response at all: what an estimator calls significant here is a
    false positive.
    """
    rng = np.random.default_rng(100 + seed)
    T = int(round(duration * fs))
    time = np.arange(T) / fs
    onsets = np.arange(10.0, 280.0 + 1e-9, 4.5)
    cond = np.arange(onsets.size) % 2
    y = ar1_noise(rng, T, 1, 0.95)[:, 0] + 0.5 * np.sin(2 * np.pi * 0.012 * time + rng.uniform(0, 2 * np.pi))
    conditions = Conditions(["c1", "c2"], [onsets[cond == 0], onsets[cond == 1]],
                            [np.full((cond == 0).sum(), 2.0), np.full((cond == 1).sum(), 2.0)])
    return _hb(time, fs, y[:, None], -0.4 * y[:, None], analysis_layout(1), conditions)


def glm_recording(seed: int = 0, fs: float = 5.0, duration: float = 300.0) -> Hb:
    """Five channels of a two-load block design with known betas.

    Channel 0 responds to both loads (0.5 and 1.5 uM), channel 1 to the high
    load only, channel 2 carries a motion spike that robust fitting should
    down-weight, channel 3 is noise only and channel 4 was excluded by QC
    (all NaN). Noise is AR(1) with rho 0.9 plus a slow drift.
    """
    rng = np.random.default_rng(seed)
    T = int(round(duration * fs))
    time = np.arange(T) / fs
    on0, on1 = np.arange(20.0, duration - 40, 60.0), np.arange(50.0, duration - 40, 60.0)
    conditions = Conditions(["0-Back", "2-Back"], [on0, on1], [np.full(on0.size, 20.0), np.full(on1.size, 20.0)])
    h = canonical_hrf(fs)
    regs = []
    for on in (on0, on1):
        r = np.convolve(boxcar(time, on, 20.0), h)[:T]
        regs.append(r / r.max())
    X = np.column_stack(regs)
    betas = np.array([[0.5, 1.5], [0.0, 1.0], [0.3, 0.8], [0.0, 0.0], [0.0, 0.0]])
    drift = 0.3 * np.sin(2 * np.pi * 0.004 * time)[:, None]
    hbo = X @ betas.T + 0.2 * ar1_noise(rng, T, 5, 0.9) + drift
    hbr = -0.4 * X @ betas.T + 0.1 * ar1_noise(rng, T, 5, 0.9) - 0.3 * drift
    hbo[int(150 * fs):int(153 * fs), 2] += 6.0
    hbo[:, 4] = hbr[:, 4] = np.nan
    return _hb(time, fs, hbo, hbr, analysis_layout(5), conditions, task="NBack")


def connectivity_recording(seed: int = 0, fs: float = 5.0, duration: float = 240.0) -> Hb:
    """Eight long channels and one short sharing a common factor; channels 2 and 5 excluded (NaN)."""
    rng = np.random.default_rng(seed)
    T = int(round(duration * fs))
    time = np.arange(T) / fs
    common = ar1_noise(rng, T, 1, 0.8)
    load = np.linspace(0.2, 0.9, 9)
    hbo = common * load + ar1_noise(rng, T, 9, 0.7)
    hbr = -0.3 * hbo + 0.2 * ar1_noise(rng, T, 9, 0.5)
    hbo[:, [2, 5]] = hbr[:, [2, 5]] = np.nan
    return _hb(time, fs, hbo, hbr, analysis_layout(9, n_short=1), Conditions(), task="Rest")


def correlation_sessions(seed: int, n_subjects: int, n_sessions: int, n_channels: int, noise: float = 1.2,
                         dropout: float = 0.0, task: str = "sim") -> list:
    """Connectivity-like matrices: a stable individual pattern under larger session noise.

    Values are squashed with tanh so they behave like correlations; with
    ``dropout`` a random fraction of channels is set NaN in each session, as
    QC exclusion leaves them.
    """
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(n_channels, 1)
    out = []
    for s in range(n_subjects):
        individual = rng.standard_normal(iu[0].size)
        for e in range(n_sessions):
            M = np.zeros((n_channels, n_channels))
            M[iu] = np.tanh(0.3 * (individual + noise * rng.standard_normal(iu[0].size)))
            M = M + M.T
            np.fill_diagonal(M, np.nan)
            if dropout:
                dead = rng.choice(n_channels, int(round(dropout * n_channels)), replace=False)
                M[dead, :] = M[:, dead] = np.nan
            out.append(SessionMeasure(subject=f"S{s + 1:02d}", session=str(e + 1), measure=M, task=task))
    return out


def gtheory_sessions(n_sub: int, n_ses: int, n_ch: int, vp: float, vps: float, vc: float, vpc: float, ve: float,
                     seed: int, task: str = "rest") -> list:
    """Per-channel vectors with planted variance components."""
    rng = np.random.default_rng(seed)
    p = np.sqrt(vp) * rng.standard_normal(n_sub)
    ch = np.sqrt(vc) * rng.standard_normal(n_ch)
    pc = np.sqrt(vpc) * rng.standard_normal((n_sub, n_ch))
    out = []
    for s in range(n_sub):
        for e in range(n_ses):
            y = p[s] + np.sqrt(vps) * rng.standard_normal() + ch + pc[s] + np.sqrt(ve) * rng.standard_normal(n_ch)
            out.append(SessionMeasure(subject=f"S{s + 1:02d}", session=str(e + 1), measure=y, task=task))
    return out


def growth_sessions(n_sub: int, n_ses: int, n_el: int, slope: float, slope_sd: float, seed: int) -> list:
    """Sessions whose elements grow linearly with person-specific slopes."""
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_sub):
        u0, u1 = rng.standard_normal(), slope_sd * rng.standard_normal()
        for e in range(n_ses):
            y = u0 + (slope + u1) * e + rng.standard_normal(n_el)
            out.append(SessionMeasure(subject=f"S{s + 1:02d}", session=str(e + 1), measure=y, task="x"))
    return out


def series_sessions(M: np.ndarray, subject: str = "s", task: str = "x") -> list:
    """One SessionMeasure per column of M (elements x sessions), sessions numbered 1..n."""
    return [SessionMeasure(subject=subject, session=str(i + 1), measure=M[:, i], task=task)
            for i in range(M.shape[1])]


def behaviour_trials(seed: int, n: int, p_correct: float, mean_rt: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Trial RTs (s, shifted log-normal, a few zero non-responses) and accuracy."""
    rng = np.random.default_rng(seed)
    rt = 0.25 + rng.lognormal(np.log(mean_rt - 0.25), 0.35, n)
    rt[rng.random(n) < 0.03] = 0.0
    return rt, rng.random(n) < p_correct


# =============================================================================
# Brain Games tablet files: what wnirs.io reads and wnirs.qc judges.
#
# The writers below reproduce the tablet's on-disk formats exactly (header
# spellings and spacing, float32 readings, the TEST sheet's column types),
# because the reader's agreement with the reference outputs is only
# meaningful on files shaped like the real ones.
# =============================================================================

TABLET_FS = 7.05
APP_WORD = {"Flanker": "Fish", "NBack": "NBack", "GoNogo": "SkipX", "Rest": "Rest"}
N_SOURCES, N_DETECTORS = 8, 16
LED_RP, LED_LP = (100, 69), (20, 20)


@dataclass
class TabletRecording:
    """A synthetic tablet recording and the ground truth it was built from.

    ``hbo`` / ``hbr`` are the concentration changes (uM, ``(T, C)`` over the
    full layout) that the intensities encode, systemic and noise included.
    ``roles`` names the deliberately faulty channels by layout index.
    """
    sensor_file: Path
    behaviour_file: Path | None
    time: np.ndarray
    trials: pd.DataFrame
    hbo: np.ndarray
    hbr: np.ndarray
    roles: dict[str, int]


#: Layout indices (into wnirs_layout_full.csv) of the faulty channels.
FAULTY_CHANNELS = {"saturated": 3, "dark": 6, "motion": 10, "weak": 12}


def tablet_name(task: str, kind: str, session: int | str, subject: str, when: datetime,
                unfinished: bool = False) -> str:
    """``<AppTask>_<Kind>_Session<N>_<Subject>_<YYYYMMDD>_<HHMMSS>[_unfinished]``."""
    stem = f"{APP_WORD[task]}_{kind}_Session{session}_{subject}_{when:%Y%m%d_%H%M%S}"
    return stem + ("_unfinished" if unfinished else "")


def write_tablet_recording(folder: Path, task: str = "Flanker", *, subject: str = "90001",
                           session: int | str = 1,
                           start: datetime = datetime(2026, 1, 5, 10, 0, 0),
                           duration: float = 180.0, seed: int = 0, behaviour: bool = True,
                           unfinished: bool = False) -> TabletRecording:
    """Write a full-montage SensorData CSV (and its BehavioralData workbook).

    Every long channel carries an event-related haemodynamic response to the
    task's trials (bigger for the harder condition), a systemic oscillation,
    a 1.1 Hz cardiac pulse and a head movement at 40 % of the recording. Short
    channels carry the systemic part only and sit brighter. Four long
    channels are faulty on purpose (:data:`FAULTY_CHANNELS`): one clipped at
    the 5 V ceiling for a third of the recording, one with no light above the
    dark level, one with a loose optode that jumps every 8 s, and one whose
    light is above the dark floor but not resolvable above its own noise. The
    91 source-detector pairs the montage does not use read the dark level.
    """
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    n = int(round(duration * TABLET_FS))
    time = np.round(0.138 + np.arange(n) / TABLET_FS + rng.uniform(-0.002, 0.002, n), 5)
    trials = make_trials(task, duration, rng) if task != "Rest" else pd.DataFrame()

    layout = full_layout()
    n_ch = len(layout)
    is_short = layout["Type"].to_numpy() == 2
    gain = np.where(is_short, 0.0, rng.uniform(0.5, 1.5, n_ch))
    evoked = _evoked_response(time, trials)
    systemic = 0.6 * np.sin(2 * np.pi * 0.08 * time + 0.4) + 0.3 * np.sin(2 * np.pi * 0.03 * time)
    hbo = np.outer(evoked, gain) + np.outer(systemic, rng.uniform(0.7, 1.3, n_ch))
    hbo += 0.05 * rng.standard_normal((n, n_ch))
    hbr = -0.35 * np.outer(evoked, gain) - 0.2 * np.outer(systemic, np.ones(n_ch))
    hbr += 0.03 * rng.standard_normal((n, n_ch))

    od = forward_od(hbo, hbr) + 0.004 * np.sin(2 * np.pi * 1.1 * time)[:, None, None]
    od[(time > 0.4 * duration) & (time < 0.4 * duration + 1.0)] += 0.12      # head movement
    for t_jump in np.arange(5.0, duration, 8.0):
        od[(time >= t_jump) & (time < t_jump + 1.0), FAULTY_CHANNELS["motion"]] += 0.5
    # Brighter for a third of the recording, which pushes it into the ceiling.
    od[(time > 0.3 * duration) & (time < 0.63 * duration), FAULTY_CHANNELS["saturated"]] -= 0.1

    light = np.where(is_short[:, None], rng.uniform(2.0, 2.8, (n_ch, 2)),
                     rng.uniform(0.05, 0.9, (n_ch, 2)))
    light[FAULTY_CHANNELS["saturated"]] = 3.0
    light[FAULTY_CHANNELS["dark"]] = 0.0004
    # Raw-intensity OD is not dark-corrected, so a jump only registers as motion
    # on a channel bright enough to dwarf the ~1.5 V pedestal.
    light[FAULTY_CHANNELS["motion"]] = 2.5
    light[FAULTY_CHANNELS["weak"]] = 0.010
    signal_ = light * 10.0 ** (-od)
    intensity = DARK + signal_ + 0.0006 * rng.standard_normal((n, n_ch, 2))
    intensity = np.minimum(intensity, 4.99)
    reference = DARK + 0.05 * signal_ + 0.0006 * rng.standard_normal((n, n_ch, 2))

    pairs = _all_pair_columns(layout, intensity, reference, rng)
    dc = DARK[0] + 0.02 * rng.random(N_DETECTORS) + 0.0005 * rng.standard_normal((n, N_DETECTORS))
    sensor = folder / (tablet_name(task, "SensorData", session, subject, start, unfinished)
                       + ".csv")
    _write_sensor_csv(sensor, time, pairs, dc)

    beh = None
    if behaviour and task != "Rest":
        beh = folder / (tablet_name(task, "BehavioralData", session, subject,
                                    start + timedelta(seconds=duration + 25)) + ".xlsx")
        write_behaviour_workbook(beh, task, trials, start)
    return TabletRecording(sensor, beh, time, trials, hbo, hbr, dict(FAULTY_CHANNELS))


def write_sensor_stub(path: Path, n_samples: int, *, header: bool = True) -> Path:
    """A one-pair sensor CSV, for discovery tests where only names and sizes matter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Time,Stimulus, S1-D1-740nm-RP, S1-D1-850nm-RP, S1-D1-740nm-LP, S1-D1-850nm-LP"]
    if not header:
        lines = []
    lines += [f"{(k + 1) / TABLET_FS:.3f}, 0,1.512,1.511,1.513,1.512" for k in range(n_samples)]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_behaviour_workbook(path: Path, task: str, trials: pd.DataFrame,
                             start: datetime) -> Path:
    """A BehavioralData workbook in the tablet's TEST-sheet layout for ``task``.

    Column order and cell types follow the real exports: labels and the
    wall-clock ``stimOnTime`` are text, onsets are seconds on the sensor clock,
    and the GoNogo export stores response times as text, with its misspelt
    ``Commison`` column beside ``commissionError``.
    """
    import openpyxl

    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "TEST"
    columns = {
        "Flanker": ["stimuliPresented", "correctResponse", "stimuliType", "stimOnTime",
                    "stimOn_ms", "stimOffTime", "accuracy", "omission", "responseTime",
                    "recordedResponse", "commissionError"],
        "NBack": ["stimuliPresented", "stimOnTime", "stimOn_ms", "stimOffTime", "block",
                  "responseTime", "accuracy", "commissionError", "omission"],
        "GoNogo": ["stimuliPresented", "stimOnTime", "stimOn_ms", "stimOffTime", "accuracy",
                   "omission", "Commison", "responseTime", "responseCommission",
                   "commissionError"],
    }[task]
    sheet.append(columns)
    for t in trials.itertuples():
        clock = (start + timedelta(seconds=float(t.Onset))).strftime("%m/%d/%Y %H:%M:%S")
        rt = float(t.ResponseTime)
        common = {"stimOnTime": clock, "stimOn_ms": float(t.Onset),
                  "stimOffTime": round(float(t.Onset + t.Duration), 3),
                  "accuracy": int(t.Accuracy), "omission": int(t.Omission),
                  "commissionError": int(t.CommissionError)}
        if task == "Flanker":
            cell = {**common, "stimuliPresented": t.Shown, "correctResponse": int(t.Shown[2]),
                    "stimuliType": t.Condition, "responseTime": rt,
                    "recordedResponse": int(t.Shown[2]) if t.Accuracy else 3 - int(t.Shown[2])}
        elif task == "NBack":
            cell = {**common, "stimuliPresented": t.Shown, "block": t.Condition,
                    "responseTime": rt if rt > 0 else 0}
        else:
            cell = {**common, "stimuliPresented": t.Shown, "Commison": 0,
                    "responseTime": f"{rt:.3f}" if rt > 0 else 0,
                    "responseCommission": f"{rt:.3f}" if t.CommissionError else 0}
        sheet.append([cell[c] for c in columns])
    book.save(path)
    return path


def make_trials(task: str, duration: float, rng: np.random.Generator) -> pd.DataFrame:
    """Trial timing, labels and responses for one task, on the sensor clock."""
    if task == "Flanker":
        onset = np.arange(10.0, duration - 15, 7.0) + rng.uniform(0, 1.5)
        cond = rng.choice(["congruent", "incongruent"], onset.size)
        shown = ["22222" if c == "congruent" else "11211" for c in cond]
        dur = rng.uniform(2.0, 2.1, onset.size)
        rt = rng.uniform(0.7, 1.3, onset.size) + 0.15 * (cond == "incongruent")
    elif task == "NBack":
        onset = np.arange(15.0, duration - 15, 2.5)
        level = np.minimum((onset - 15.0) // (duration / 3 - 5), 2).astype(int)
        cond = np.array(["0-Back", "1-Back", "2-Back"])[level]
        shown = [str(d) for d in rng.integers(1, 10, onset.size)]
        dur = np.full(onset.size, 1.96)
        rt = np.where(rng.random(onset.size) < 0.7, 0.0, rng.uniform(0.4, 0.9, onset.size))
    elif task == "GoNogo":
        onset = np.arange(12.0, duration - 12, 1.13)
        shown = [("X" if rng.random() < 0.15 else rng.choice(list("ABCDEFGHJKMR")))
                 for _ in onset]
        cond = np.where(np.array(shown) == "X", "NoGo", "Go")
        dur = np.full(onset.size, 0.322)
        rt = rng.uniform(0.35, 0.7, onset.size)
    else:
        raise ValueError(f"no trial structure for {task!r}")
    accuracy = (rng.random(onset.size) < 0.85).astype(int)
    commission = np.zeros(onset.size, dtype=int)
    if task == "GoNogo":
        nogo = cond == "NoGo"
        commission[nogo] = 1 - accuracy[nogo]
        rt[nogo & (accuracy == 1)] = 0.0           # withheld: no response recorded
    omission = ((accuracy == 0) & (commission == 0) & (rng.random(onset.size) < 0.5)).astype(int)
    return pd.DataFrame({"Onset": np.round(onset, 3), "Duration": np.round(dur, 3),
                         "Condition": cond, "Shown": shown, "Accuracy": accuracy,
                         "ResponseTime": np.round(rt, 3), "CommissionError": commission,
                         "Omission": omission})


def _evoked_response(time: np.ndarray, trials: pd.DataFrame) -> np.ndarray:
    """HRF-convolved trial train, uM, scaled by how demanding each condition is."""
    weight = {"congruent": 0.6, "incongruent": 1.0, "0-Back": 0.4, "1-Back": 0.7,
              "2-Back": 1.0, "Go": 0.3, "NoGo": 1.0}
    fs = 1.0 / np.median(np.diff(time))
    drive = np.zeros(time.size)
    for t in trials.itertuples():
        drive += weight[t.Condition] * boxcar(time, [t.Onset], t.Duration)
    if not drive.any():
        return drive
    out = np.convolve(drive, canonical_hrf(fs))[:time.size]
    return 0.8 * out / out.max()


def _all_pair_columns(layout: pd.DataFrame, intensity: np.ndarray, reference: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """``(T, 8 * 16, 4)``: RP 740, RP 850, LP 740, LP 850 for every source-detector pair."""
    n = intensity.shape[0]
    pairs = np.empty((n, N_SOURCES * N_DETECTORS, 4))
    pairs[:] = np.tile(DARK, 2) + 0.004 * rng.random((1, N_SOURCES * N_DETECTORS, 1))
    pairs += 0.0008 * rng.standard_normal(pairs.shape)
    for ch, (s, d) in enumerate(zip(layout["S"], layout["D"])):
        k = (s - 1) * N_DETECTORS + (d - 1)
        pairs[:, k, :2] = intensity[:, ch]
        pairs[:, k, 2:] = reference[:, ch]
    return pairs


def _write_sensor_csv(path: Path, time: np.ndarray, pairs: np.ndarray, dc: np.ndarray) -> None:
    header = ["Time", "Stimulus"]
    for s in range(1, N_SOURCES + 1):
        for d in range(1, N_DETECTORS + 1):
            header += [f" S{s}-D{d}-740nm-RP", f" S{s}-D{d}-850nm-RP", "ledPowerLevel-740nm-RP",
                       " ledPowerLevel-850nm-RP", f" S{s}-D{d}-740nm-LP", f" S{s}-D{d}-850nm-LP",
                       " ledPowerLevel-740nm-LP", " ledPowerLevel-850nm-LP"]
    header += [f"D{d}-DC" for d in range(1, N_DETECTORS + 1)]

    # The device writes float32 readings; their shortest float64 repr is what lands on disk.
    as_text = pairs.astype(np.float32).astype(np.float64).astype(str)
    dc_text = dc.astype(np.float32).astype(np.float64).astype(str)
    led = f" {LED_RP[0]},{LED_RP[1]}"
    lp_led = f" {LED_LP[0]},{LED_LP[1]}"
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for i in range(time.size):
            row = as_text[i]
            cells = [f"{row[k, 0]}, {row[k, 1]},{led},{row[k, 2]}, {row[k, 3]},{lp_led}"
                     for k in range(row.shape[0])]
            f.write(f"{float(time[i])!r}, 0," + ",".join(cells) + "," + ",".join(dc_text[i])
                    + "\n")


#: Task lengths the reader fixtures use: short enough that a 160 s NBack is
#: over-long and a 60 s GoNogo overruns the row cap.
FIXTURE_TASK_LENGTH = {"Flanker": 450.0, "GoNogo": 10.0, "NBack": 100.0, "Rest": 444.0}


def write_reader_study(root: Path) -> dict[str, TabletRecording]:
    """Four full-montage recordings of subject 90001, session 1.

    ``flanker``  180 s with its behavioural file: the QC and conditions case
    ``nback``    160 s, trimmed to 100 s under :data:`FIXTURE_TASK_LENGTH`
    ``rest``     90 s, ``_unfinished``, no behavioural file by design
    ``gonogo``   60 s, whose 10 s task length caps the read at 401 rows
    """
    folder = root / "90001" / "Tablet_backup"
    day = datetime(2026, 1, 5, 10, 0, 0)
    return {
        "flanker": write_tablet_recording(folder, "Flanker", start=day, duration=180, seed=0),
        "nback": write_tablet_recording(folder, "NBack", start=day + timedelta(minutes=10),
                                        duration=160, seed=1),
        "rest": write_tablet_recording(folder, "Rest", start=day + timedelta(minutes=20),
                                       duration=90, seed=2, unfinished=True),
        "gonogo": write_tablet_recording(folder, "GoNogo", start=day + timedelta(minutes=25),
                                         duration=60, seed=3),
    }


def write_discovery_study(root: Path) -> Path:
    """A study folder exercising every rule of session discovery and the manifest.

    Sensor files are one-pair stubs whose sample count stands in for length.
    Subject 90001 (``90001/Tablet_backup``):

    ======  =================================================================
    S1      Fish, NBack, SkipX with behaviour; Rest ``_unfinished`` (none needed)
    S2      NBack twice: 600 samples on Jan 7 with no behaviour, 300 on Jan 8
            with its behaviour 8 min later (the smaller one must win); and a
            650-sample ``testuser_0327`` rehearsal (never a subject)
    S3      Fish twice on Jan 9, neither with behaviour (the bigger wins)
    S4      SkipX header-only (never started) and a real SkipX with behaviour
    S5      NBack without behaviour, beside NBack behaviour for session 15
    S6      NBack of 5 samples (a stub, dropped although unique)
    S7      SkipX fragment with no header
    S8      Fish whose behavioural file is 8 h away (does not count)
    S9      Fish on a date the tracksheet numbers session 10
    ======  =================================================================

    Subject 90002 sits one folder deeper (``90002/Brain games/Tablet_backup``)
    and its NBack file carries the mistyped id 90003. Beside the recordings:
    questionnaire exports, a wearable export, a QC table, participants.csv and
    a tracksheet.
    """
    tab = root / "90001" / "Tablet_backup"
    jan = lambda d, h=9, m=0: datetime(2026, 1, d, h, m, 0)  # noqa: E731
    rng = np.random.default_rng(7)

    def sensor(task, session, when, n, subject="90001", folder=tab, **kw):
        name = tablet_name(task, "SensorData", session, subject, when, kw.pop("unfinished", False))
        return write_sensor_stub(folder / f"{name}.csv", n, **kw)

    def behaviour(task, session, when, subject="90001", folder=tab):
        name = tablet_name(task, "BehavioralData", session, subject, when)
        write_behaviour_workbook(folder / f"{name}.xlsx", task, make_trials(task, 60, rng), when)

    for i, task in enumerate(("Flanker", "NBack", "GoNogo")):
        sensor(task, 1, jan(5, 9, 10 * i), 300)
        behaviour(task, 1, jan(5, 9, 10 * i + 8))
    sensor("Rest", 1, jan(5, 9, 40), 300, unfinished=True)
    sensor("NBack", 2, jan(7), 600)
    sensor("NBack", 2, jan(8), 300)
    behaviour("NBack", 2, jan(8, 9, 8))
    sensor("NBack", 2, jan(8, 7), 650, subject="testuser_0327")
    sensor("Flanker", 3, jan(9), 300)
    sensor("Flanker", 3, jan(9, 11), 500)
    sensor("GoNogo", 4, jan(10, 8), 0)
    sensor("GoNogo", 4, jan(10), 300)
    behaviour("GoNogo", 4, jan(10, 9, 8))
    sensor("NBack", 5, jan(12), 300)
    behaviour("NBack", 15, jan(20, 9, 8))
    sensor("NBack", 6, jan(13), 5)
    sensor("GoNogo", 7, jan(14), 300, header=False)
    sensor("Flanker", 8, jan(15), 300)
    behaviour("Flanker", 8, jan(15, 17, 0))
    sensor("Flanker", 9, jan(16), 300)

    deep = root / "90002" / "Brain games" / "Tablet_backup"
    sensor("NBack", 1, jan(6), 300, subject="90003", folder=deep)
    behaviour("NBack", 1, jan(6, 9, 8), subject="90003", folder=deep)

    for name in ("90001_EMA_raw.csv", "90001_CAARS_labels.csv", "EMA_90001.csv"):
        (root / "90001" / name).write_text(
            "experiment_date,experiment_time,ema_focus\n2026-01-05,09:00,3\n")
    (root / "90001" / "wearable").mkdir()
    (root / "90001" / "wearable" / "hr_export.csv").write_text("time,hr\n1,61\n2,63\n")
    (root / "qc_all_recordings.csv").write_text("Subject,Channel,Pass\n90001,1,1\n")
    (root / "participants.csv").write_text("Subject,Group,Age\n90001,Patient,31\n"
                                           "90002,Control,29\n")
    write_tracksheet(root / "synthetic_tracksheet.xlsx")
    return root


def write_tracksheet(path: Path) -> Path:
    """A tracking workbook in the study's layout, for subject 90001.

    Rows: S1 OFF, S2 ON after the dose, S3 CBT, an unnumbered attempt marked
    Incomplete on the S4 date, S5 ON before the dose, S6-S8 OFF, and session
    10 on the date of the S9 recording.
    """
    import openpyxl

    book = openpyxl.Workbook()
    summary = book.active
    summary.title = "Summary"
    summary.append(["Subject", "Group", "Box ", "Start date", "End date"])
    summary.append([90001, "ADHDMed", 4, datetime(2026, 1, 5), datetime(2026, 1, 31)])
    summary.append([90002, "Control", 2, datetime(2026, 1, 6), datetime(2026, 1, 6)])

    header = ["Session ", "Date ", "Weekday", "On/off Med", "Before/after Med", "Task 1", None,
              "Task 2", None, "Task 3", None, "Rest", "EMA", "Comments", "Portal report",
              "Notes", "Tablet backup", "Notes", "Data uploaded box", "Notes"]
    rows = [(1, 5, "OFF", None, ""), (2, 8, "ON", "After med", ""), (3, 9, "CBT", None, ""),
            (None, 10, "OFF", None, "Incomplete"), (5, 12, "ON", "Before med", ""),
            (6, 13, "OFF", None, ""), (7, 14, "OFF", None, ""), (8, 15, "OFF", None, ""),
            (10, 16, "OFF", None, "2 sessions")]
    for name in ("90001", "Template"):
        sheet = book.create_sheet(name)
        sheet.append(header)
        for col in ("F1:G1", "H1:I1", "J1:K1"):
            sheet.merge_cells(col)
        if name == "Template":
            sheet.append([1] + [None] * 4 + [False, None] * 3 + [False] * 2)
            continue
        for session, day, med, timing, note in rows:
            when = datetime(2026, 1, day)
            sheet.append([session, when, f"{when:%a}", med, timing, True, "NBack", True, "SkipX",
                          True, "Flanker", True, False, None, False, None, True, note or None,
                          True, None])
    book.save(path)
    return path


#: Medication arm of each session in :func:`write_behaviour_study`.
BEHAVIOUR_ARMS = {"1": 0, "2": 0, "3": 0, "4": 1, "5": 1, "6": 1}


def write_behaviour_study(root: Path) -> Path:
    """Six sessions of all three tasks' behaviour for subject 90001.

    Session 5's Flanker is written twice on consecutive days (the later file
    must win), and a ``testuser_0327`` workbook sits alongside (never read).
    """
    folder = root / "90001" / "Tablet_backup"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(11)
    for s in BEHAVIOUR_ARMS:
        for j, task in enumerate(("Flanker", "NBack", "GoNogo")):
            when = datetime(2026, 1, 5 + 2 * int(s), 9, 10 * j)
            write_behaviour_workbook(
                folder / (tablet_name(task, "BehavioralData", s, "90001", when) + ".xlsx"),
                task, make_trials(task, 150, rng), when)
    write_behaviour_workbook(
        folder / (tablet_name("Flanker", "BehavioralData", 5, "90001", datetime(2026, 1, 16, 9))
                  + ".xlsx"), "Flanker", make_trials("Flanker", 150, rng), datetime(2026, 1, 16))
    write_behaviour_workbook(
        folder / (tablet_name("NBack", "BehavioralData", 2, "testuser_0327",
                              datetime(2026, 1, 9, 7)) + ".xlsx"),
        "NBack", make_trials("NBack", 150, rng), datetime(2026, 1, 9, 7))
    return root


def study_fingerprint(root: Path) -> dict[str, str]:
    """SHA-256 of every file under ``root``, keyed by relative path.

    Workbooks are hashed by their cell values, since an .xlsx also stores the
    time it was saved.
    """
    import hashlib

    import openpyxl

    prints = {}
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        if f.suffix == ".xlsx":
            book = openpyxl.load_workbook(f, read_only=True)
            content = repr([list(book[s].iter_rows(values_only=True))
                            for s in book.sheetnames]).encode()
            book.close()
        else:
            content = f.read_bytes()
        prints[f.relative_to(root).as_posix()] = hashlib.sha256(content).hexdigest()
    return prints
