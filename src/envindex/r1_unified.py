"""envindex.r1_unified — unified stage-summary (R1) feature extraction.

Mirrors the protocol R1 representation (Paper 1 stage_summary) so that any
source with daily weather — G2F, T3, CIMMYT AgERA5 — produces the SAME R1
feature vocabulary.  This is the "3B" path of
specs/cimmyt_climate_normalization.md: recompute consistent features rather
than use each source's native summaries.

Feature layout (per stage, per weather variable): mean / min / max / sum / std
plus heat_days (tmax > threshold), rain_days, dry_days — identical to
gxe_budget.data.preprocessing._build_stage_weather_sequence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEATHER_COLS = ["tmax", "tmin", "tmean", "precipitation", "solar_radiation",
                "relative_humidity", "vpd", "gdd"]
STATS = ["mean", "min", "max", "sum", "std"]


def stage_summaries(
    env_weather: pd.DataFrame,
    stage_windows: list[tuple[str, int, int]],
    heat_threshold: float = 25.0,
    weather_cols: list[str] | None = None,
) -> np.ndarray:
    """Compute (n_stages, n_features) stage-summary matrix for one environment.

    env_weather: daily weather for one environment with `day_after_planting`
                 (or `_dap`) and weather columns.
    Returns a float32 (S, F) matrix; missing stage data is NaN.
    """
    cols = weather_cols or WEATHER_COLS
    w = env_weather.copy()
    if "day_after_planting" in w.columns:
        w["_dap"] = pd.to_numeric(w["day_after_planting"], errors="coerce")
    elif "_dap" not in w.columns:
        raise ValueError("weather must have day_after_planting or _dap")

    mat = np.full((len(stage_windows), len(cols) * len(STATS) + 3), np.nan, dtype=np.float32)
    for si, (_name, start, end) in enumerate(stage_windows):
        stage = w[(w["_dap"] >= start) & (w["_dap"] <= end)]
        feats: list[float] = []
        for c in cols:
            vals = pd.to_numeric(stage[c], errors="coerce").dropna().to_numpy()
            if len(vals) == 0:
                feats.extend([np.nan] * 5)
            else:
                feats.extend([
                    float(np.mean(vals)),
                    float(np.min(vals)),
                    float(np.max(vals)),
                    float(np.sum(vals)),
                    float(np.std(vals)) if len(vals) > 1 else 0.0,
                ])
        tmax = pd.to_numeric(stage.get("tmax"), errors="coerce") if "tmax" in stage.columns else pd.Series(dtype=float)
        precip = pd.to_numeric(stage.get("precipitation"), errors="coerce") if "precipitation" in stage.columns else pd.Series(dtype=float)
        feats.extend([
            float((tmax > heat_threshold).sum()) if len(tmax) else np.nan,
            float((precip > 0.0).sum()) if len(precip) else np.nan,
            float((precip <= 0.0).sum()) if len(precip) else np.nan,
        ])
        mat[si, : len(feats)] = np.asarray(feats[: mat.shape[1]], dtype=np.float32)
    return mat


def build_r1_from_daily(
    weather: pd.DataFrame,
    crop_profile,
    env_meta: pd.DataFrame | None = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Build R1 stage-summary matrices for all environments from daily weather.

    weather  : protocol §3.3 weather_daily.parquet (environment_id,
               day_after_planting optional; if absent and env_meta has
               planting_date, DAP is derived).
    crop_profile : crop_profiles.CropProfile (stage_windows, heat threshold).
    Returns ({env_id: (S, F) matrix}, env_ids).
    """
    w = weather.copy()
    w["environment_id"] = w["environment_id"].astype(str)

    if "day_after_planting" not in w.columns:
        if env_meta is None or "planting_date" not in env_meta.columns:
            raise ValueError("weather has no day_after_planting and env_meta lacks planting_date")
        env = env_meta[["environment_id", "planting_date"]].copy()
        env["environment_id"] = env["environment_id"].astype(str)
        env["planting_date"] = pd.to_datetime(env["planting_date"], errors="coerce")
        w["date"] = pd.to_datetime(w["date"], errors="coerce")
        w = w.merge(env, on="environment_id", how="left")
        w["day_after_planting"] = (w["date"] - w["planting_date"]).dt.days
        w = w.dropna(subset=["day_after_planting"])

    out: dict[str, np.ndarray] = {}
    for env_id, group in w.groupby("environment_id", sort=False):
        out[str(env_id)] = stage_summaries(
            group,
            crop_profile.stage_windows,
            heat_threshold=crop_profile.heat_day_tmax_threshold,
        )
    return out, sorted(out)
