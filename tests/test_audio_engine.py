import dataclasses

import numpy as np
import pytest
import soundfile as sf

from backend.dsp.audio_engine import (
    AudioBuffer, apply_gain, conform, load_audio, resample, save_audio, to_mono, to_stereo,
)


def sine(freq=1000.0, fs=48000, seconds=1.0, amp=0.5, channels=1):
    t = np.arange(int(fs * seconds)) / fs
    x = amp * np.sin(2 * np.pi * freq * t)
    return AudioBuffer(np.tile(x[:, None], (1, channels)), fs)


# --- construction and validation ---------------------------------------------

@pytest.mark.parametrize("shape", [(10, 3), (10, 0), (2, 5, 1), ()])
def test_rejects_bad_shapes(shape):
    with pytest.raises(ValueError):
        AudioBuffer(np.zeros(shape), 48000)


@pytest.mark.parametrize("rate", [0, -48000, 48000.0, True, "48000", None])
def test_rejects_bad_sample_rates(rate):
    with pytest.raises(ValueError):
        AudioBuffer(np.zeros(10), rate)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_samples(bad):
    with pytest.raises(ValueError, match="finite"):
        AudioBuffer(np.array([0.0, bad]), 48000)


def test_accepts_lists_ints_and_numpy_rates():
    buf = AudioBuffer([0, 1, -1], np.int64(44100))
    assert buf.samples.dtype == np.float64 and buf.samples.shape == (3, 1)
    assert type(buf.sample_rate) is int


def test_empty_buffer_is_valid():
    buf = AudioBuffer(np.zeros((0, 2)), 48000)
    assert (buf.frames, buf.duration, buf.peak, buf.rms) == (0, 0.0, 0.0, 0.0)
    np.testing.assert_array_equal(buf.channel_peaks(), [0.0, 0.0])


def test_properties():
    buf = AudioBuffer(np.array([[0.5, -0.25], [-1.0, 0.0]]), 2)
    assert (buf.channels, buf.frames, buf.duration, buf.peak) == (2, 2, 1.0, 1.0)
    np.testing.assert_array_equal(buf.channel_peaks(), [1.0, 0.25])


# --- immutability ------------------------------------------------------------

def test_caller_array_is_copied_and_left_writable():
    data = np.zeros(100)
    buf = AudioBuffer(data, 48000)
    data[0] = 1.0
    assert buf.samples[0, 0] == 0.0
    assert data.flags.writeable


def test_samples_are_read_only():
    buf = sine()
    with pytest.raises(ValueError):
        buf.samples[0, 0] = 1.0


def test_fields_cannot_be_reassigned():
    buf = sine()
    with pytest.raises(dataclasses.FrozenInstanceError):
        buf.samples = "ciao"
    with pytest.raises(dataclasses.FrozenInstanceError):
        buf.sample_rate = 1


# --- the internal constructor validates exactly like the public one ----------

@pytest.mark.parametrize(
    "samples, rate",
    [(np.array([0.0, np.nan]), 48000), (np.zeros((4, 3)), 48000), (np.zeros(4), 0), (np.zeros(4), True)],
)
def test_wrap_cannot_bypass_validation(samples, rate):
    with pytest.raises(ValueError):
        AudioBuffer._wrap(samples, rate)


def test_wrap_output_is_read_only():
    buf = AudioBuffer._wrap(np.zeros(4), 48000)
    assert not buf.samples.flags.writeable


# --- gain ------------------------------------------------------------------------

def test_gain_plus_6db_doubles_amplitude():
    assert apply_gain(sine(amp=0.25), 20 * np.log10(2)).peak == pytest.approx(0.5, rel=1e-12)


def test_gain_zero_db_is_identity_and_input_untouched():
    buf = sine()
    before = buf.samples.copy()
    np.testing.assert_array_equal(apply_gain(buf, 0.0).samples, before)
    apply_gain(buf, 12.0)
    np.testing.assert_array_equal(buf.samples, before)


@pytest.mark.parametrize("gain", [np.nan, np.inf, -np.inf])
def test_gain_rejects_non_finite_gain(gain):
    with pytest.raises(ValueError, match="finite"):
        apply_gain(sine(), gain)


@pytest.mark.parametrize("gain", [1e6, np.float64(1e6), 6200.0, 6165.0])
def test_gain_rejects_overflow(gain):
    # 6165 dB keeps the factor finite (~1.8e308) but 1.5 * factor overflows:
    # the check must look at the result, not only at the factor.
    buf = AudioBuffer(np.full(8, 1.5), 48000)
    with pytest.raises(ValueError, match="overflow"):
        apply_gain(buf, gain)


def test_huge_attenuation_underflows_to_silence():
    assert apply_gain(sine(), -1e6).peak == 0.0


# --- file I/O --------------------------------------------------------------------

def test_roundtrip_float_is_exact(tmp_path):
    buf = sine(channels=2)
    save_audio(buf, tmp_path / "x.wav", subtype="FLOAT")
    back = load_audio(tmp_path / "x.wav")
    assert (back.sample_rate, back.channels, back.frames) == (48000, 2, buf.frames)
    np.testing.assert_allclose(back.samples, buf.samples, atol=1e-7)


@pytest.mark.parametrize("subtype, bits", [("PCM_16", 16), ("PCM_24", 24), ("pcm_24", 24)])
def test_roundtrip_pcm_within_quantisation(tmp_path, subtype, bits):
    buf = sine()
    save_audio(buf, tmp_path / "x.wav", subtype=subtype)
    np.testing.assert_allclose(load_audio(tmp_path / "x.wav").samples, buf.samples, atol=2.0 ** -(bits - 2))


def test_pcm_accepts_exactly_full_scale(tmp_path):
    save_audio(AudioBuffer(np.array([1.0, -1.0, 0.0]), 48000), tmp_path / "fs.wav")
    assert load_audio(tmp_path / "fs.wav").peak == pytest.approx(1.0, abs=1e-6)


def test_pcm_export_refuses_overs_but_float_keeps_them(tmp_path):
    hot = sine(amp=1.5)
    with pytest.raises(ValueError, match="0 dBFS"):
        save_audio(hot, tmp_path / "hot.wav")
    assert not (tmp_path / "hot.wav").exists()
    save_audio(hot, tmp_path / "hot_float.wav", subtype="FLOAT")
    assert load_audio(tmp_path / "hot_float.wav").peak == pytest.approx(1.5, rel=1e-6)


def test_save_rejects_unknown_subtype_and_creates_folders(tmp_path):
    with pytest.raises(ValueError):
        save_audio(sine(), tmp_path / "x.wav", subtype="MP3")
    save_audio(sine(), tmp_path / "a" / "b" / "x.wav")
    assert (tmp_path / "a" / "b" / "x.wav").exists()


def test_load_rejects_nan_wav_and_names_the_file(tmp_path):
    sf.write(tmp_path / "broken.wav", np.array([0.1, np.nan, np.inf, 0.2]), 48000, subtype="FLOAT")
    with pytest.raises(ValueError, match=r"broken\.wav.*2 NaN/inf"):
        load_audio(tmp_path / "broken.wav")


def test_load_rejects_multichannel(tmp_path):
    sf.write(tmp_path / "surround.wav", np.zeros((100, 6)), 48000)
    with pytest.raises(ValueError, match="6 channels"):
        load_audio(tmp_path / "surround.wav")


def test_load_mono_and_int_pcm(tmp_path):
    sf.write(tmp_path / "m.wav", np.array([0, 16384, -32768], dtype=np.int16), 44100)
    buf = load_audio(tmp_path / "m.wav")
    assert (buf.channels, buf.sample_rate) == (1, 44100)
    np.testing.assert_allclose(buf.samples[:, 0], [0.0, 0.5, -1.0])
    assert not buf.samples.flags.writeable


# --- channel and rate conversion ---------------------------------------------

def test_to_mono_averages_channels():
    st = AudioBuffer(np.array([[1.0, 0.0], [0.5, -0.5]]), 48000)
    np.testing.assert_allclose(to_mono(st).samples[:, 0], [0.5, 0.0])


def test_conversions_return_same_object_when_nothing_to_do():
    # Safe because buffers are immutable.
    mono, stereo = sine(), sine(channels=2)
    assert to_mono(mono) is mono and to_stereo(stereo) is stereo
    assert resample(mono, 48000) is mono


def test_to_stereo_duplicates():
    st = to_stereo(sine())
    np.testing.assert_array_equal(st.samples[:, 0], st.samples[:, 1])


@pytest.mark.parametrize("src, dst", [(44100, 48000), (48000, 44100), (48000, 96000)])
def test_resample_keeps_duration_frequency_and_level(src, dst):
    buf = resample(sine(freq=1000, fs=src, amp=0.5), dst)
    assert buf.sample_rate == dst
    assert buf.duration == pytest.approx(1.0, abs=1e-3)
    spectrum = np.abs(np.fft.rfft(buf.samples[:, 0]))
    assert np.argmax(spectrum) * dst / buf.frames == pytest.approx(1000, abs=2)
    middle = buf.samples[dst // 4: 3 * dst // 4, 0]  # away from edge effects
    assert np.max(np.abs(middle)) == pytest.approx(0.5, abs=0.005)


@pytest.mark.parametrize("rate", [0, 44100.0, True])
def test_resample_rejects_bad_rate(rate):
    with pytest.raises(ValueError):
        resample(sine(), rate)


def test_resample_empty_buffer(monkeypatch):
    import backend.dsp.audio_engine as engine

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("resample_poly called on empty input")

    monkeypatch.setattr(engine, "resample_poly", must_not_be_called)
    for channels in (1, 2):
        out = resample(AudioBuffer(np.zeros((0, channels)), 44100), 48000)
        assert (out.frames, out.channels, out.sample_rate) == (0, channels, 48000)
        assert not out.samples.flags.writeable


def test_resample_empty_still_validates_rate():
    with pytest.raises(ValueError):
        resample(AudioBuffer(np.zeros((0, 1)), 44100), True)


@pytest.mark.parametrize("channels", [1, 2, np.int64(1), np.int32(2)])
def test_conform(channels):
    out = conform(sine(fs=44100, channels=1), 48000, channels)
    assert (out.sample_rate, out.channels) == (48000, int(channels))


@pytest.mark.parametrize("channels", [True, False, 1.0, 2.0, 0, 3, "2", None, np.bool_(True)])
def test_conform_rejects_bad_channels(channels):
    with pytest.raises(ValueError, match="channels"):
        conform(sine(), 48000, channels)


@pytest.mark.parametrize("rate", [True, 48000.0, 0])
def test_conform_rejects_bad_rate(rate):
    with pytest.raises(ValueError):
        conform(sine(), rate, 2)


def test_conform_empty_buffer():
    out = conform(AudioBuffer(np.zeros((0, 1)), 44100), 48000, 2)
    assert (out.frames, out.channels, out.sample_rate) == (0, 2, 48000)