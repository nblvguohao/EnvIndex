"""envid — unified multi-source environment identifiers.

CIMMYT, T3, G2F and SoyNAM each use their own environment keys (study names,
site codes, `loc x year`, `environ`).  This module defines a single, stable,
parseable `environment_id` used across the EnvIndex pipeline and the
protocol's parquet schema (protocol_freeze_paper2.md §3.3).

Format
------
    {crop}:{source}:{native_env_id}

  crop      lowercase crop key (maize | wheat | soybean | ...)
  source    lowercase data source (g2f | t3 | iwin | soynam | ...)
  native    the original environment key, sanitized to [a-z0-9_]

Examples
--------
    maize:wheat:t3:...
    wheat:t3:SDS_2017_AUR_AYT
    wheat:iwin:ESWYT_19103_2017
    soybean:soynam:IA_2012

Native IDs are sanitized (non [A-Za-z0-9_] -> "_") so the result is
filesystem- and parquet-friendly and lossless for round-tripping the source
key (parse returns the sanitized form).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SEP = ":"
_INVALID_CHARS = re.compile(r"[^A-Za-z0-9_]+")
_MULTI_UNDERSCORE = re.compile(r"_+")


def sanitize_native(native: str) -> str:
    """Normalize a native environment key to [a-z0-9_]+."""
    out = _INVALID_CHARS.sub("_", native.strip())
    out = _MULTI_UNDERSCORE.sub("_", out)
    return out.strip("_")


def make_environment_id(crop: str, source: str, native: str) -> str:
    """Build a unified environment id."""
    crop = crop.lower().strip()
    source = source.lower().strip()
    if not crop or not source or not native:
        raise ValueError(f"crop/source/native all required: {crop=} {source=} {native=}")
    return _SEP.join([crop, source, sanitize_native(native)])


@dataclass(frozen=True)
class ParsedEnvironmentId:
    crop: str
    source: str
    native: str

    def to_string(self) -> str:
        return make_environment_id(self.crop, self.source, self.native)


def parse_environment_id(env_id: str) -> ParsedEnvironmentId:
    """Split a unified environment id back into (crop, source, native)."""
    parts = env_id.split(_SEP)
    if len(parts) < 3:
        raise ValueError(f"Not a valid environment id (need crop:source:native): {env_id!r}")
    return ParsedEnvironmentId(crop=parts[0], source=parts[1], native=_SEP.join(parts[2:]))


def is_environment_id(candidate: str) -> bool:
    try:
        parse_environment_id(candidate)
        return True
    except ValueError:
        return False


# Registry of known sources so tooling can validate / document them.
KNOWN_SOURCES: dict[str, str] = {
    "g2f": "maize",
    "t3": "wheat",
    "iwin": "wheat",
    "soynam": "soybean",
    "cimmyt": "wheat",  # alias for iwin
}
