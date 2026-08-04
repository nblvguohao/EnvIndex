"""t3_brapi_export.py — Export T3 (Triticeae Toolbox) wheat/barley/oat trial
data via BrAPI V2.

Supports the Paper 2 W1-W3 data census (protocol_freeze_paper2.md §3.1, §3.2):
  - census mode:  list breeding programs and trials, flag which trials record
                   phenology traits (heading/anthesis/flowering/maturity), and
                   emit a candidate-environment catalog
  - export mode:  pull per-plot phenology observations from selected trials
                   into a long-format parquet table

Access path follows the official T3 BrAPI R package
(https://github.com/TriticeaeToolbox/BrAPI.R/blob/main/TUTORIAL.md):
  GET /programs  -> breeding programs
  GET /studies   -> trials (with location, year, planting date)
  GET /studies/{id}/observationvariables -> traits measured in a trial
  GET /studies/{id}/observations         -> per-plot observations

Dependencies: stdlib urllib + pandas + pyarrow (both already available in the
project env). No `requests` dependency.

Usage:
  # Census: enumerate programs + trials for spring-wheat, flag phenology traits
  python scripts/t3_brapi_export.py --crop wheat --base-url https://wheat.triticeaetoolbox.org \
      --mode census --out-dir data/t3

  # Export phenology observations from trials whose name contains "spring"
  python scripts/t3_brapi_export.py --crop wheat --base-url https://wheat.triticeaetoolbox.org \
      --mode export --trial "spring" --out-dir data/t3

Authentication: public T3 data is generally readable anonymously; if an OIDC
token is required, pass it via --token or the T3_TOKEN env var.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

# ---------------------------------------------------------------- constants

# BrAPI base URLs per crop.  The wheat-sandbox is the commonly used read
# instance; production WheatCAP is listed as an alternative.
DEFAULT_BASE_URLS = {
    "wheat": "https://wheat.triticeaetoolbox.org/",
    "barley": "https://barley.triticeaetoolbox.org/",
    "oat": "https://oat.triticeaetoolbox.org/",
}

# Token file written by t3_login.py; read automatically if T3_TOKEN is unset.
DEFAULT_TOKEN_FILE = Path("data/t3/.t3_token")

# Trait-name keywords used to identify phenology traits in a trial's
# observationVariable list.  Kept broad because T3 trials use heterogeneous
# naming (e.g. "Heading date", "Days to heading", "Anthesis", "Zadoks").
PHENOLOGY_KEYWORDS = [
    "heading",
    "anthes",  # anthesis / anthesi
    "flower",  # flowering / flower date
    "zadoks",
    "feekes",
    "booting",
    "maturity",
    "grain fill",
    "grain filling",
    "gdd",
    "growing degree",
    "thermal",
    "days to",
    "spike emergence",
]

# Planting/harvest metadata keys that different BrAPI instances nest
# differently; probed defensively.
PLANTING_DATE_KEYS = ["plantingDate", "Planting Date", "planting_date"]
HARVEST_DATE_KEYS = ["harvestDate", "Harvest Date", "harvest_date"]

PAGE_SIZE = 1000


# ---------------------------------------------------------------- client

@dataclass
class BrApiClient:
    """Minimal BrAPI V2 client with pagination and optional bearer auth.

    `_getter` is injectable for tests; the default hits the real HTTP API.
    """

    base_url: str
    token: str | None = None
    sleep_seconds: float = 0.25
    max_retries: int = 5
    retry_backoff: float = 2.0
    _getter: Callable[[str, dict | None], Any] | None = None

    def __post_init__(self) -> None:
        if not self.base_url.endswith("/"):
            self.base_url += "/"
        # BrAPI v2 paths hang off <root>/brapi/v2/.  Accept either a root URL
        # (https://host/) or a full base (https://host/brapi/v2/).
        if "brapi/" not in self.base_url:
            self.base_url += "brapi/v2/"
        self._call = self._getter or self._http_get

    # -- low-level ---------------------------------------------------------

    def _http_get(self, url: str, params: dict | None = None) -> Any:
        if params:
            url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def _call_with_retry(self, url: str, params: dict | None = None) -> Any:
        """Call self._call with exponential-backoff retry on transient errors.

        Retries ConnectionReset / timeouts / truncated JSON and HTTP 429/5xx;
        fatal 401/403 and other 4xx are raised immediately.
        """
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call(url, params)
            except (urllib.error.URLError, ConnectionResetError, TimeoutError,
                    urllib.error.HTTPError, json.JSONDecodeError) as exc:
                if isinstance(exc, urllib.error.HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                    raise
                last_error = exc
                wait = self.retry_backoff * (2 ** (attempt - 1))
                print(f"[t3_brapi_export] retry {attempt}/{self.max_retries} after "
                      f"{type(exc).__name__} ({wait:.1f}s)", file=sys.stderr)
                time.sleep(wait)
        raise RuntimeError(f"Request failed after {self.max_retries} retries: {last_error}") from last_error

    def _paginated(self, path: str, params: dict | None = None) -> list[dict]:
        """Walk all pages of a BrAPI list endpoint."""
        out: list[dict] = []
        params = dict(params or {})
        params.setdefault("pageSize", PAGE_SIZE)
        page = 0
        while True:
            params["page"] = page
            payload = self._call_with_retry(self.base_url + path.lstrip("/"), params)
            result = payload.get("result", {})
            data = result.get("data", [])
            out.extend(data if isinstance(data, list) else [])
            # BrAPI v2 servers place pagination under metadata.pagination;
            # some implementations put it under result.pagination. Probe both.
            pagination = payload.get("metadata", {}).get("pagination", {}) or result.get("pagination", {})
            total_pages = pagination.get("totalPages")
            if total_pages is None:
                # Fall back to totalCount-based inference.
                total = pagination.get("totalCount")
                size = pagination.get("pageSize") or params.get("pageSize", PAGE_SIZE)
                total_pages = ((total or 0) + size - 1) // size if size else 1
            if page + 1 >= total_pages or not data:
                break
            page += 1
            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)
        return out

    # -- domain ------------------------------------------------------------

    def list_programs(self) -> list[dict]:
        return self._paginated("programs")

    def list_studies(self, program_db_id: str | None = None) -> list[dict]:
        params = {}
        if program_db_id:
            params["programDbId"] = program_db_id
        return self._paginated("studies", params)

    def list_locations(self) -> list[dict]:
        return self._paginated("locations")

    def study_observation_variables(self, study_db_id: str) -> list[dict]:
        return self._paginated(f"studies/{urllib.parse.quote(study_db_id)}/observationvariables")

    def study_observations(
        self, study_db_id: str, variable_db_id: str | None = None
    ) -> list[dict]:
        params = {}
        if variable_db_id:
            params["observationVariableDbId"] = variable_db_id
        return self._paginated(f"studies/{urllib.parse.quote(study_db_id)}/observations", params)


# ---------------------------------------------------------------- parsing

def _get(variable: dict, *keys: str, default: Any = None) -> Any:
    """Nested-safe key probe."""
    for key in keys:
        if key in variable and variable[key] is not None:
            return variable[key]
    return default


def is_phenology_trait(var: dict) -> bool:
    """Heuristic: does this observation variable look like a phenology trait?"""
    name = str(
        _get(var, "observationVariableName", "traitName", "name", default="")
    ).lower()
    trait_name = str(_get(var, "trait", "traitName", default="")).lower()
    method = str(_get(var, "method", "methodName", default="")).lower()
    haystack = " ".join([name, trait_name, method])
    return any(kw in haystack for kw in PHENOLOGY_KEYWORDS)


def variable_id(var: dict) -> str:
    return str(_get(var, "observationVariableDbId", "variableDbId", "traitDbId", default=""))


def variable_name(var: dict) -> str:
    return str(_get(var, "observationVariableName", "traitName", "name", default=""))


def variable_unit(var: dict) -> str:
    scale = _get(var, "scale", default={})
    if isinstance(scale, dict):
        return str(scale.get("unit", ""))
    return str(scale or "")


def _study_env_fields(study: dict) -> dict:
    """Extract environment-candidate fields from a BrAPI study object.

    Returns keys: program_name, trial_name, study_name, study_db_id, year,
    location_name, location_db_id, planting_date, harvest_date.
    """
    location = _get(study, "location", default={}) or {}
    if not isinstance(location, dict):
        location = {}
    seasons = _get(study, "seasons", default=[]) or []
    year = None
    if seasons:
        year = _get(seasons[0], "year", default=None) if isinstance(seasons[0], dict) else None
    # planting date may be nested in additionalInfo / environmentParameters
    planting = harvest = None
    additional = _get(study, "additionalInfo", default={}) or {}
    if isinstance(additional, dict):
        for key in PLANTING_DATE_KEYS:
            if key in additional:
                planting = additional[key]
                break
        for key in HARVEST_DATE_KEYS:
            if key in additional:
                harvest = additional[key]
                break
    env_params = _get(study, "environmentParameters", default=[]) or []
    if isinstance(env_params, list):
        for param in env_params:
            if not isinstance(param, dict):
                continue
            pname = str(param.get("parameterName", "")).lower()
            for key in PLANTING_DATE_KEYS:
                if key.lower() in pname:
                    planting = param.get("value")
                    break
            for key in HARVEST_DATE_KEYS:
                if key.lower() in pname:
                    harvest = param.get("value")
                    break

    return {
        "program_name": _get(study, "programName", default=""),
        "trial_name": _get(study, "trialName", default=""),
        "study_name": _get(study, "studyName", default=""),
        "study_db_id": _get(study, "studyDbId", default=""),
        "study_type": _get(study, "studyTypeName", default=""),
        "year": year,
        "location_name": _get(location, "locationName", _get(study, "locationName", default="")),
        "location_db_id": _get(location, "locationDbId", _get(study, "locationDbId", default="")),
        "planting_date": planting,
        "harvest_date": harvest,
    }


# ---------------------------------------------------------------- exports

def _collect_studies(
    client: BrApiClient, program_filter: str | None, trial_filter: str | None
) -> list[tuple[dict, dict]]:
    """Enumerate programs and their studies (all pages).

    Returns a flat list of (program, study) tuples.  Study enumeration is
    sequential (programs are few); the per-study variable fetch happens later
    in parallel.
    """
    programs = client.list_programs()
    records: list[tuple[dict, dict]] = []
    for program in programs:
        pname = str(_get(program, "programName", default=""))
        if program_filter and program_filter.lower() not in pname.lower():
            continue
        pdbid = str(_get(program, "programDbId", default=""))
        studies = client.list_studies(pdbid) if pdbid else []
        for study in studies:
            env = _study_env_fields(study)
            if trial_filter and trial_filter.lower() not in env["study_name"].lower():
                continue
            records.append((program, study))
    return records


def _fetch_study_variables(client: BrApiClient, study_db_id: str, study_name: str) -> tuple[str, list[dict], str]:
    """Fetch a study's observation variables; returns (study_id, variables, error)."""
    try:
        variables = client.study_observation_variables(study_db_id)
        return study_db_id, variables, ""
    except Exception as exc:  # noqa: BLE001 - census is best-effort
        error = f"{type(exc).__name__}: {exc}"
        print(f"[t3_brapi_export] WARN study {study_db_id} ({study_name}) "
              f"variable fetch failed: {error}", file=sys.stderr)
        return study_db_id, [], error


def build_trials_catalog(
    client: BrApiClient,
    program_filter: str | None = None,
    trial_filter: str | None = None,
    workers: int = 6,
) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    """Census: enumerate trials, flag phenology-variable availability.

    Study-variable fetches run in a thread pool (`workers`) because the
    per-study observationvariables call is the dominant cost and the T3
    connection is flaky.  A single study failing server-side is recorded in
    `variables_error` and does not abort the census.

    Returns (catalog DataFrame, per-trial phenology variable lists).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    records = _collect_studies(client, program_filter, trial_filter)
    total = len(records)
    print(f"[t3_brapi_export] collected {total} studies across filtered programs; "
          f"fetching variables with {workers} workers...")

    catalog_rows: list[dict] = []
    phenology_by_study: dict[str, list[dict]] = {}
    done = 0
    env_by_study: dict[str, dict] = {}
    program_by_study: dict[str, dict] = {}
    for program, study in records:
        env = _study_env_fields(study)
        env_by_study[env["study_db_id"]] = env
        program_by_study[env["study_db_id"]] = program

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_study_variables, client, sid, env["study_name"]): sid
            for sid, env in env_by_study.items()
        }
        for future in as_completed(futures):
            study_db_id, variables, error = future.result()
            pheno = [v for v in variables if is_phenology_trait(v)]
            phenology_by_study[study_db_id] = [
                {"variable_db_id": variable_id(v), "name": variable_name(v), "unit": variable_unit(v)}
                for v in pheno
            ]
            env = env_by_study[study_db_id]
            program = program_by_study[study_db_id]
            catalog_rows.append(
                {
                    **env,
                    "program_name": str(_get(program, "programName", default="")),
                    "crop": "wheat",
                    "n_variables_total": len(variables),
                    "n_phenology_variables": len(pheno),
                    "phenology_traits": "; ".join(sorted({variable_name(v) for v in pheno})),
                    "has_phenology": len(pheno) > 0,
                    "variables_error": error,
                }
            )
            done += 1
            if done % 100 == 0 or done == total:
                print(f"[t3_brapi_export] progress {done}/{total} studies "
                      f"({len(phenology_by_study)} cached)", flush=True)

    catalog = pd.DataFrame(catalog_rows)
    return catalog, phenology_by_study


def export_phenology_observations(
    client: BrApiClient,
    catalog: pd.DataFrame,
    phenology_by_study: dict[str, list[dict]],
) -> pd.DataFrame:
    """Export per-plot phenology observations for catalog trials.

    Long-format rows: study_db_id, trial_name, location_name, year, plot,
    germplasm, trait, unit, value.
    """
    rows: list[dict] = []
    for _, env in catalog.iterrows():
        study_db_id = str(env["study_db_id"])
        pheno_vars = phenology_by_study.get(study_db_id, [])
        for var in pheno_vars:
            try:
                observations = client.study_observations(study_db_id, var["variable_db_id"])
            except Exception as exc:  # noqa: BLE001 - best-effort export
                print(f"[t3_brapi_export] WARN study {study_db_id} trait "
                      f"{var['name']} observations failed: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                continue
            for obs in observations:
                rows.append(
                    {
                        "study_db_id": study_db_id,
                        "trial_name": env.get("trial_name", ""),
                        "study_name": env.get("study_name", ""),
                        "location_name": env.get("location_name", ""),
                        "year": env.get("year"),
                        "plot": str(_get(obs, "observationUnitName", default="")),
                        "germplasm": str(_get(obs, "germplasmName", default="")),
                        "trait": var["name"],
                        "unit": var["unit"],
                        "value": _get(obs, "value", default=None),
                        "observation_time": str(_get(obs, "observationTimeStamp", default="")),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- CLI

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--crop", choices=["wheat", "barley", "oat"], default="wheat",
                        help="Crop portal to query (default wheat)")
    parser.add_argument("--base-url", default=None, help="BrAPI base URL (defaults to crop portal)")
    parser.add_argument("--mode", choices=["census", "export"], default="census",
                        help="census: list trials + flag phenology traits; export: pull observations")
    parser.add_argument("--program", default=None, help="Substring filter on breeding program name")
    parser.add_argument("--trial", default=None, help="Substring filter on study/trial name")
    parser.add_argument("--out-dir", type=Path, default=Path("data/t3"), help="Output directory")
    parser.add_argument("--token", default=None, help="BrAPI bearer token (or set T3_TOKEN)")
    parser.add_argument("--sleep", type=float, default=0.25, help="Seconds between paged requests")
    parser.add_argument("--retries", type=int, default=5,
                        help="Max attempts per request on transient failures (default 5)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel study-variable fetch workers (default 6)")
    return parser.parse_args(argv)


def _resolve_token(token: str | None) -> str | None:
    """Token priority: --token > T3_TOKEN env > token file from t3_login.py."""
    if token:
        return token
    token = os.environ.get("T3_TOKEN")
    if token:
        return token
    if DEFAULT_TOKEN_FILE.exists():
        return DEFAULT_TOKEN_FILE.read_text(encoding="utf-8").strip()
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    base_url = args.base_url or DEFAULT_BASE_URLS[args.crop]
    token = _resolve_token(args.token)
    client = BrApiClient(base_url=base_url, token=token, sleep_seconds=args.sleep,
                         max_retries=args.retries)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[t3_brapi_export] base_url={base_url} mode={args.mode} crop={args.crop}")

    catalog, phenology_by_study = build_trials_catalog(
        client, program_filter=args.program, trial_filter=args.trial, workers=args.workers
    )
    catalog_path = args.out_dir / "trials_catalog.parquet"
    catalog.to_parquet(catalog_path, index=False)
    print(f"[t3_brapi_export] catalog: {len(catalog)} trials -> {catalog_path}")
    if len(catalog) == 0:
        print("[t3_brapi_export] no trials matched; nothing to export")
        return 0

    n_with_pheno = int(catalog["has_phenology"].sum()) if "has_phenology" in catalog else 0
    print(f"[t3_brapi_export] trials with phenology traits: {n_with_pheno}/{len(catalog)}")
    for _, row in catalog.head(15).iterrows():
        marker = "P" if row.get("has_phenology") else "-"
        print(f"  [{marker}] {row.get('study_name','')} "
              f"(loc={row.get('location_name','')}, year={row.get('year','')}) "
              f"[{row.get('n_phenology_variables',0)} pheno vars]")

    if args.mode == "export":
        pheno = export_phenology_observations(client, catalog, phenology_by_study)
        pheno_path = args.out_dir / "phenotype_pheno.parquet"
        pheno.to_parquet(pheno_path, index=False)
        print(f"[t3_brapi_export] phenology observations: {len(pheno)} rows -> {pheno_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
