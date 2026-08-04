"""Tests for the corn stage-feature builder (G2F weather -> stage matrix)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from envindex.corn_features import (
    CORN_STAGE_GDD_WINDOWS,
    WEATHER_VARS,
    anchor_planting_date,
    build_corn_stage_features,
    _gdd,
)


def _weather_with_season():
    """Synthetic full-year weather: cold winter, hot long summer, cold autumn.

    The hot block (days 90-270, tmean 28 C => GDD ~18/day, ~3240 total) covers
    the full corn profile up to 2450 GDD so every stage has data.
    """
    n = 365
    tmean = np.full(n, 5.0)
    tmean[90:271] = 28.0  # hot summer
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
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
    assert mat.shape == (len(CORN_STAGE_GDD_WINDOWS), len(WEATHER_VARS) * 4)
    assert anchor is not None
    # warm season means tmean is positive everywhere; no NaN
    assert np.isfinite(mat).all()


def test_empty_weather_returns_zeros():
    mat, anchor = build_corn_stage_features(pd.DataFrame())
    assert mat.shape == (len(CORN_STAGE_GDD_WINDOWS), len(WEATHER_VARS) * 4)
    assert anchor is None
    assert np.all(mat == 0)


def test_gdd_calculation():
    import pandas as pd
    tmax = pd.Series([25.0, 35.0])
    tmin = pd.Series([15.0, 25.0])
    # day1: (25+15)/2 - 10 = 10; day2: tmax capped at 30 -> (30+25)/2 - 10 = 17.5
    g = _gdd(tmax, tmin).tolist()
    assert abs(g[0] - 10.0) < 1e-6
    assert abs(g[1] - 17.5) < 1e-6
