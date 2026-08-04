"""envindex.corn_features — build corn stage-summary features from G2F weather.

G2F weather_daily records cover the full calendar year with no planting date.
For the pilot, the growing season is anchored by the first day the 7-day
rolling mean tmean >= 10 C (corn base temperature), and the crop profile's
five DAP stage windows are applied to compute per-stage weather summaries.

Stage summaries mirror the protocol R1 representation: per stage and variable
mean/min/max/std.  This is an approximation for the v0 pilot; exact DAP-based
alignment (protocol §3.4) is the W3 production path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# corn stage windows (DAP) — from the maize crop profile (crop_profiles.py)
CORN_STAGE_WINDOWS = [
    ("early", 0, 30),
    ("vegetative", 31, 60),
    ("flowering", 61, 90),
    ("grain_fill", 91, 130),
    ("late", 131, 180),
]

WEATHER_VARS = ["tmax", "tmin", "tmean", "precipitation", "solar_radiation", "relative_humidity"]
STATS = ["mean", "min", "max", "std"]

BASE_TEMP_C = 10.0


def anchor_planting_date(env_weather: pd.DataFrame, base_temp: float = BASE_TEMP_C) -> pd.Timestamp | None:
    """Return the date where the 7-day rolling mean tmean first >= base_temp."""
    if env_weather.empty or "tmean" not in env_weather.columns:
        return None
    w = env_weather.sort_values("_weather_date").copy()
    w = w.dropna(subset=["tmean", "_weather_date"])
    if w.empty:
        return None
    w = w.set_index("_weather_date")
    roll = w["tmean"].rolling(7, min_periods=5).mean()
    above = roll[roll >= base_temp]
    if above.empty:
        return None
    return above.index[0]


def build_corn_stage_features(env_weather: pd.DataFrame) -> tuple[np.ndarray, pd.Timestamp | None]:
    """Build a (n_stages, F) stage-summary matrix for one corn environment.

    Returns (matrix, planting_anchor).
    """
    if env_weather.empty or "_weather_date" not in env_weather.columns:
        return np.zeros((len(CORN_STAGE_WINDOWS), len(WEATHER_VARS) * len(STATS)), dtype=np.float32), None

    anchor = anchor_planting_date(env_weather)
    if anchor is None:
        return np.zeros((len(CORN_STAGE_WINDOWS), len(WEATHER_VARS) * len(STATS)), dtype=np.float32), None

    w = env_weather.dropna(subset=["_weather_date"]).copy()
    w["_dap"] = (pd.to_datetime(w["_weather_date"]) - anchor).dt.days
    mat = np.zeros((len(CORN_STAGE_WINDOWS), len(WEATHER_VARS) * len(STATS)), dtype=np.float32)
    for si, (_name, start, end) in enumerate(CORN_STAGE_WINDOWS):
        stage = w[(w["_dap"] >= start) & (w["_dap"] <= end)]
        col = 0
        for var in WEATHER_VARS:
            vals = pd.to_numeric(stage[var], errors="coerce").dropna().to_numpy()
            if len(vals) == 0:
                mat[si, col : col + 4] = np.nan
            else:
                mat[si, col] = float(np.mean(vals))
                mat[si, col + 1] = float(np.min(vals))
                mat[si, col + 2] = float(np.max(vals))
                mat[si, col + 3] = float(np.std(vals)) if len(vals) > 1 else 0.0
            col += 4
    return mat, anchor


def load_corn_envs(
    weather_path: str,
    env_meta_path: str,
    n_envs: int = 20,
    seed: int = 0,
) -> dict:
    """Load a subset of G2F corn environments with stage features.

    Returns {env_id: {"x": (S,F), "anchor": timestamp}} plus lists.
    """
    from gxe_budget.data.preprocessing import normalize_weather_dates

    weather = pd.read_parquet(weather_path)
    weather = normalize_weather_dates(weather)
    weather["environment_id"] = weather["environment_id"].astype(str)

    env_meta = pd.read_parquet(env_meta_path)
    env_ids = sorted(weather["environment_id"].unique())
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(env_ids, size=min(n_envs, len(env_ids)), replace=False))

    out = {}
    for env_id in chosen:
        sub = weather[weather["environment_id"] == env_id]
        mat, anchor = build_corn_stage_features(sub)
        out[env_id] = {"x": mat, "anchor": anchor}
    return out
