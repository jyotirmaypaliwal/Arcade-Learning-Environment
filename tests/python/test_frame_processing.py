"""Check processed frames against independent single-frame observations."""

import numpy as np
import pytest
from ale_py import ALEInterface, LoggerMode, roms


def make_ale(**settings):
    ALEInterface.setLoggerMode(LoggerMode.Error)
    ale = ALEInterface()
    ale.setInt("random_seed", 0)
    ale.setInt("frame_skip", 1)
    ale.setFloat("repeat_action_probability", 0.0)
    for key, value in settings.items():
        ale.setBool(key, value)
    ale.loadROM(roms.get_rom_path("breakout"))
    return ale


@pytest.mark.parametrize("grayscale", [False, True])
def test_frame_max_matches_successive_raw_frames(grayscale):
    raw = make_ale()
    processed = make_ale(frame_max=True)
    get_raw = raw.getScreenGrayscale if grayscale else raw.getScreenRGB
    get_processed = (
        processed.getScreenGrayscale if grayscale else processed.getScreenRGB
    )
    previous = get_raw()
    for i in range(100):
        action = raw.getMinimalActionSet()[i % len(raw.getMinimalActionSet())]
        raw.act(action)
        processed.act(action)
        current = get_raw()
        expected = np.maximum(previous, current)
        np.testing.assert_array_equal(get_processed(), expected)
        buffer = np.empty_like(expected)
        get_processed(buffer)
        np.testing.assert_array_equal(buffer, expected)
        previous = current


def test_frame_average_matches_phosphor_formula():
    raw = make_ale()
    processed = make_ale(frame_average=True)
    previous = raw.getScreenRGB().astype(np.int32)
    for i in range(100):
        action = raw.getMinimalActionSet()[i % len(raw.getMinimalActionSet())]
        raw.act(action)
        processed.act(action)
        current = raw.getScreenRGB().astype(np.int32)
        expected = (
            np.minimum(previous, current) + np.abs(previous - current) * 77 // 100
        )
        np.testing.assert_array_equal(processed.getScreenRGB(), expected)
        # Match C++ round (ties away from zero), rather than NumPy's ties-to-even.
        luminance = np.floor(
            expected[..., 0] * 0.2989
            + expected[..., 1] * 0.5870
            + expected[..., 2] * 0.1140
            + 0.5
        ).astype(np.uint8)
        np.testing.assert_array_equal(processed.getScreenGrayscale(), luminance)
        previous = current


@pytest.mark.parametrize("setting", ["frame_average", "color_averaging"])
def test_conflicting_frame_processors_raise(setting):
    with pytest.raises(RuntimeError, match="Cannot enable both"):
        make_ale(frame_max=True, **{setting: True})


@pytest.mark.parametrize("setting", ["frame_average", "frame_max"])
def test_restore_preserves_processed_observations(setting):
    ale = make_ale(**{setting: True})
    ale.act(1)
    state = ale.cloneSystemState()
    rgb, grayscale, ram = ale.getScreenRGB(), ale.getScreenGrayscale(), ale.getRAM()
    ale.act(0)
    ale.restoreSystemState(state)
    np.testing.assert_array_equal(ale.getScreenRGB(), rgb)
    np.testing.assert_array_equal(ale.getScreenGrayscale(), grayscale)
    np.testing.assert_array_equal(ale.getRAM(), ram)
