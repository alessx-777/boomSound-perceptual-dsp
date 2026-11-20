"""Offline audio I/O and basic signal utilities for boomSound."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

PCM_SUBTYPES = {"PCM_16", "PCM_24", "PCM_32"}
FLOAT_SUBTYPES = {"FLOAT", "DOUBLE"}


def _validate_rate(sample_rate: object, name: str = "sample_rate") -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(sample_rate)


def _validate_samples(samples: object) -> np.ndarray:
    """Single source of truth for what a valid buffer contains."""
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] not in (1, 2):
        raise ValueError("Audio must have one or two channels")
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise ValueError(f"Audio samples must be finite ({bad} NaN/inf values found)")
    return array


@dataclass(frozen=True, eq=False)
class AudioBuffer:
    """Immutable audio: float64 samples, shape (frames, channels), 1 or 2 channels.

    Every buffer, whichever way it was built, passes the same validation, and
    its sample array is read-only. Buffers can therefore share memory safely
    (to_mono on a mono buffer returns the same object) and no code path can
    smuggle in NaN, inf or a wrong shape after construction.
    """

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        original = self.samples
        samples = _validate_samples(original)
        # Copy only when we would share memory with an array the caller can still write to.
        if isinstance(original, np.ndarray) and np.shares_memory(samples, original):
            samples = samples.copy()
        samples.flags.writeable = False
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "sample_rate", _validate_rate(self.sample_rate))

    @classmethod
    def _wrap(cls, samples: np.ndarray, sample_rate: int) -> "AudioBuffer":
        """Internal constructor for arrays this package just computed.

        Same validation as the public constructor; it only skips the defensive
        copy, because nobody else holds a reference to the new array.
        """
        array = _validate_samples(samples)
        array.flags.writeable = False
        buffer = cls.__new__(cls)
        object.__setattr__(buffer, "samples", array)
        object.__setattr__(buffer, "sample_rate", _validate_rate(sample_rate))
        return buffer

    @property
    def channels(self) -> int:
        return self.samples.shape[1]

    @property
    def frames(self) -> int:
        return self.samples.shape[0]

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.samples))) if self.samples.size else 0.0

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.samples ** 2))) if self.samples.size else 0.0

    def channel_peaks(self) -> np.ndarray:
        if not self.samples.size:
            return np.zeros(self.channels)
        return np.max(np.abs(self.samples), axis=0)


def load_audio(path: str | Path) -> AudioBuffer:
    samples, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if samples.shape[1] > 2:
        raise ValueError(
            f"{Path(path).name} has {samples.shape[1]} channels; boomSound supports mono or stereo"
        )
    try:
        return AudioBuffer._wrap(samples, sample_rate)
    except ValueError as err:
        raise ValueError(f"{Path(path).name}: {err}") from err


def save_audio(audio: AudioBuffer, path: str | Path, subtype: str = "PCM_24") -> None:
    """Write a WAV file.

    PCM formats cannot store values above 0 dBFS, so export is refused instead
    of clipping silently. FLOAT/DOUBLE keep overs intact, which is what you want
    for intermediate candidates that will be level-matched later.
    """
    subtype = subtype.upper()
    if subtype not in PCM_SUBTYPES | FLOAT_SUBTYPES:
        raise ValueError(f"Unsupported WAV subtype {subtype!r}")
    if subtype in PCM_SUBTYPES and audio.peak > 1.0:
        raise ValueError("Peak exceeds 0 dBFS; reduce gain before PCM export or use FLOAT")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Note: no dither is applied. For PCM_16 masters add TPDF dither upstream.
    sf.write(path, audio.samples, audio.sample_rate, format="WAV", subtype=subtype)


def apply_gain(audio: AudioBuffer, gain_db: float) -> AudioBuffer:
    if not np.isfinite(gain_db):
        raise ValueError("gain_db must be finite")
    with np.errstate(over="ignore"):
        factor = np.power(10.0, np.float64(gain_db) / 20.0)
        result = audio.samples * factor
    if not np.isfinite(factor) or not np.all(np.isfinite(result)):
        raise ValueError(f"gain of {gain_db} dB overflows float64; the result would be infinite")
    return AudioBuffer._wrap(result, audio.sample_rate)


def to_mono(audio: AudioBuffer) -> AudioBuffer:
    """Average the channels (equal-weight downmix, no level compensation)."""
    if audio.channels == 1:
        return audio
    return AudioBuffer._wrap(audio.samples.mean(axis=1), audio.sample_rate)


def to_stereo(audio: AudioBuffer) -> AudioBuffer:
    """Duplicate a mono signal on both channels."""
    if audio.channels == 2:
        return audio
    return AudioBuffer._wrap(np.repeat(audio.samples, 2, axis=1), audio.sample_rate)


def resample(audio: AudioBuffer, target_rate: int) -> AudioBuffer:
    """Polyphase resampling with SciPy's anti-aliasing filter."""
    target_rate = _validate_rate(target_rate, "target_rate")
    if target_rate == audio.sample_rate:
        return audio
    if audio.frames == 0:
        # Explicit: do not rely on resample_poly's behaviour with empty input,
        # which differs across SciPy versions.
        return AudioBuffer._wrap(np.zeros((0, audio.channels)), target_rate)
    ratio = Fraction(int(target_rate), audio.sample_rate)
    out = resample_poly(audio.samples, ratio.numerator, ratio.denominator, axis=0)
    return AudioBuffer._wrap(out, target_rate)


def conform(audio: AudioBuffer, sample_rate: int, channels: int) -> AudioBuffer:
    """Bring a buffer to a common format so candidates are comparable."""
    # bool is an int subclass (True == 1) and 1.0 == 1, so test the type first.
    if isinstance(channels, bool) or not isinstance(channels, (int, np.integer)) or channels not in (1, 2):
        raise ValueError("channels must be the integer 1 or 2")
    audio = resample(audio, sample_rate)
    return to_mono(audio) if channels == 1 else to_stereo(audio)