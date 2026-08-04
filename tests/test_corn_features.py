"""Tests for the corn stage-feature builder (G2F weather -> stage matrix)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from envindex.corn_features import (
    CORN_STAGE_WINDOWS,
    WEATHER_VARS,
    anchor_planting_date,
    build_corn_stage_features,
)


def _weather_with_season():
    """Synthetic full-year weather with a clear warm season."""
    dates = pd.date_range("2015-01-01", periods=365, freq="D")
    tmean = 5 + 20 * np.sin(2 * np.pi * (np.arange(365) - 100) / 365)  # cold winter, warm summer
    return pd.DataFrame(
        {
            "_weather_date": dates,
            "tmax": tmean + 6,
            "tmin": tmean - 6,
            "tmean": tmean,
            "precipitation": 2.0,
            "solar_radiation": 15.0,
            "relative_humidity": 60.0,
        }
    )


def test_anchor_planting_date_in_warm_season():
    w = _weather_with_season()
    anchor = anchor_planting_date(w)
    assert anchor is not None
    assert anchor.month in (4, 5)  # first 7-day rolling mean >= 10C in spring


def test_stage_matrix_shape():
    w = _weather_with_season()
    mat, anchor = build_corn_stage_features(w)
    assert mat.shape == (len(CORN_STAGE_WINDOWS), len(WEATHER_VARS) * 4)
    assert anchor is not None
    # warm season means tmean is positive everywhere; no NaN
    assert np.isfinite(mat).all()


def test_empty_weather_returns_zeros():
    mat, anchor = build_corn_stage_features(pd.DataFrame())
    assert mat.shape == (len(CORN_STAGE_WINDOWS), len(WEATHER_VARS) * 4)
    assert anchor is None
    assert np.all(mat == 0)
