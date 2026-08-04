"""Tests for unified R1 (stage-summary) feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from envindex.r1_unified import stage_summaries


def _weather(dap_values, tmax, precip):
    return pd.DataFrame(
        {
            "environment_id": ["e1"] * len(dap_values),
            "day_after_planting": dap_values,
            "tmax": tmax,
            "tmin": np.asarray(tmax) - 5,
            "tmean": np.asarray(tmax) - 2,
            "precipitation": precip,
            "solar_radiation": 15.0,
            "relative_humidity": 60.0,
            "vpd": 1.0,
            "gdd": np.maximum(0, np.asarray(tmax) - 10),
        }
    )


def test_stage_summaries_shape_and_stats():
    # two stages; put hot days in stage 1, rain days in stage 2
    w = _weather(
        dap_values=[10, 20, 40, 50, 70, 80],
        tmax=[20.0, 32.0, 20.0, 20.0, 20.0, 20.0],
        precip=[0.0, 2.0, 0.0, 0.0, 1.0, 0.0],
    )
    windows = [("s1", 0, 30), ("s2", 31, 90)]
    mat = stage_summaries(w, windows, heat_threshold=30.0)
    # n_stages x (8 vars * 5 stats + 3)
    assert mat.shape == (2, 8 * 5 + 3)
    # stage 1 (days 10,20): tmax mean = 26, one heat day (32 > 30)
    s1 = mat[0]
    assert abs(s1[0] - 26.0) < 1e-6  # tmax mean
    assert abs(s1[3] - 52.0) < 1e-6  # tmax sum (20+32)
    assert abs(s1[8 * 5 + 0] - 1.0) < 1e-6  # heat_days count (only day 20, tmax 32 > 30)
    # stage 2 (days 40,50,70,80): precip mean = 0.25
    s2 = mat[1]
    assert abs(s2[3 * 5 + 0] - 0.25) < 1e-6  # precipitation mean


def test_stage_summaries_empty_stage_is_nan():
    w = _weather(dap_values=[5, 10], tmax=[20.0, 22.0], precip=[0.0, 0.0])
    windows = [("s1", 0, 30), ("s2", 100, 130)]  # stage 2 empty
    mat = stage_summaries(w, windows)
    assert np.isnan(mat[1, 0])
