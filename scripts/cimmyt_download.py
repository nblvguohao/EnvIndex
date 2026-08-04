"""cimmyt_download.py — Download CIMMYT IWIN cleaned wheat trial data.

Source: Harvard Dataverse "Clean and formatted IWIN wheat breeding trial
data - version 2" (Xiong et al., DOI 10.7910/DVN/3GAKGY), covering the
ESWYT / HTWYT / IDYN / IWWYT_IRR / IWWYT_SA nurseries 1979-2020.

The .tab files are Dataverse TSV exports; filenames carry the observation +
simulated yield + phenology + climate bundle ("Obs_Sim_Yld_Phe_Climate_All").

Downloads are resumable with retry (the host connection is flaky).

Usage:
    python scripts/cimmyt_download.py --out-dir data/cimmyt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DATASET_DOI = "doi:10.7910/DVN/3GAKGY"
DATAVERSE_BASE = "https://dataverse.harvard.edu"
DEFAULT_OUT = Path("data/cimmyt")

_METADATA_URL = (
    f"{DATAVERSE_BASE}/api/datasets/:persistentId/"
    f"?persistentId={DATASET_DOI}"
)
_FILE_URL = f"{DATAVERSE_BASE}/api/access/datafile/{{file_id}}"


def _fetch_json(url: str, retries: int = 6) -> dict:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            wait = 2.0 * (2 ** (attempt - 1))
            print(f"[cimmyt] retry {attempt}/{retries} metadata ({wait:.0f}s): {type(exc).__name__}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url}: {last}") from last


def _download(url: str, dest: Path, retries: int = 6) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[cimmyt] skip (exists): {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "Mozilla/5.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            req = urllib.request.Request(url, headers=headers)
            mode = "ab" if offset else "wb"
            with urllib.request.urlopen(req, timeout=120) as r, open(partial, mode) as out:
                while True:
                    block = r.read(1 << 16)
                    if not block:
                        break
                    out.write(block)
            if partial.stat().st_size == 0:
                raise RuntimeError("empty download")
            partial.replace(dest)
            print(f"[cimmyt] downloaded: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError, RuntimeError) as exc:
            last = exc
            wait = 2.0 * (2 ** (attempt - 1))
            print(f"[cimmyt] retry {attempt}/{retries} download ({wait:.0f}s): {type(exc).__name__}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Download failed after {retries} retries: {last}") from last


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--file", default=None, help="Download only files whose name contains this substring")
    args = parser.parse_args(argv)

    meta = _fetch_json(_METADATA_URL)
    files = meta.get("data", {}).get("latestVersion", {}).get("files", [])
    print(f"[cimmyt] dataset metadata OK, {len(files)} files")

    targets = []
    for f in files:
        df = f.get("dataFile", {})
        name = df.get("filename")
        fid = df.get("id")
        if not name or not fid:
            continue
        if args.file and args.file not in name:
            continue
        targets.append((name, fid))

    if not targets:
        print("[cimmyt] no files matched", file=sys.stderr)
        return 1

    for name, fid in targets:
        _download(_FILE_URL.format(file_id=fid), args.out_dir / name)

    print(f"[cimmyt] done: {len(targets)} files -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
