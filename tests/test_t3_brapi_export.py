"""Unit tests for scripts/t3_brapi_export.py.

These tests inject a fake `_getter` so no network access is required.  They
verify BrAPI pagination walking, phenology-trait detection, study-metadata
parsing, and the census/export output shapes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "t3_brapi_export.py"
_spec = importlib.util.spec_from_file_location("t3_export", _SCRIPT)
t3 = importlib.util.module_from_spec(_spec)
# Register in sys.modules so @dataclass can resolve the module for type
# annotations (otherwise _is_type fails on sys.modules.get(None)).
sys.modules[_spec.name] = t3
_spec.loader.exec_module(t3)


def _paginated(pages: list[list[dict]], page_size: int = 1000):
    """Build a BrAPI-style paginated response for `pages`."""
    total = sum(len(p) for p in pages)
    def _respond(url: str, params: dict | None = None) -> dict:
        page = int((params or {}).get("page", 0))
        size = int((params or {}).get("pageSize", page_size))
        data = pages[page] if page < len(pages) else []
        total_pages = len(pages)
        return {
            "result": {
                "data": data,
                "pagination": {
                    "page": page,
                    "pageSize": size,
                    "totalCount": total,
                    "totalPages": total_pages,
                },
            }
        }
    return _respond


class FakeClient:
    """A BrApiClient whose _getter routes by URL path to canned responses."""

    def __init__(self, routes: dict[str, dict]):
        self.calls: list[str] = []
        self.routes = routes

    def __call__(self, url: str, params: dict | None = None) -> dict:
        path = url.split("/")[-1].split("?")[0]
        self.calls.append(path)
        key = path if path in self.routes else "default"
        return self.routes[key]


def _client_with(routes: dict[str, dict]) -> t3.BrApiClient:
    fake = FakeClient(routes)
    return t3.BrApiClient(base_url="https://example.test/", _getter=fake)


# ---------------------------------------------------------------- fixtures

def _program(name: str, db_id: str) -> dict:
    return {"programName": name, "programDbId": db_id}


def _study(name: str, db_id: str, **kw) -> dict:
    study = {
        "studyName": name,
        "trialName": f"trial_of_{name}",
        "studyDbId": db_id,
        "programName": "Nebraska",
        "location": {"locationName": "Mead", "locationDbId": "L1"},
        "seasons": [{"year": "2021"}],
        "additionalInfo": {"plantingDate": "2021-04-15"},
    }
    study.update(kw)
    return study


def _pheno_var(name: str, db_id: str, unit: str = "days") -> dict:
    return {
        "observationVariableDbId": db_id,
        "observationVariableName": name,
        "trait": {"traitName": name},
        "scale": {"unit": unit},
    }


# ---------------------------------------------------------------- tests

def test_paginated_walks_all_pages():
    client = t3.BrApiClient(
        base_url="https://example.test/",
        sleep_seconds=0,
        _getter=_paginated([[_program("A", "1")], [_program("B", "2")]], page_size=1),
    )
    programs = client.list_programs()
    assert [p["programName"] for p in programs] == ["A", "B"]


def test_is_phenology_trait_detection():
    assert t3.is_phenology_trait(_pheno_var("Days to heading", "v1"))
    assert t3.is_phenology_trait(_pheno_var("Heading date", "v2"))
    assert t3.is_phenology_trait(_pheno_var("Anthesis", "v3"))
    assert t3.is_phenology_trait(_pheno_var("Grain yield - kg/ha", "v4")) is False
    assert t3.is_phenology_trait(_pheno_var("Plant height - cm", "v5")) is False


def test_study_env_fields_parsing():
    env = t3._study_env_fields(_study("Spring01", "S1"))
    assert env["study_db_id"] == "S1"
    assert env["year"] == "2021"
    assert env["location_name"] == "Mead"
    assert env["planting_date"] == "2021-04-15"
    assert env["study_name"] == "Spring01"


def test_study_env_fields_t3_real_structure():
    """Parses T3's actual /studies shape (seasons as string list, top-level
    locationName/startDate, programName nested in additionalInfo)."""
    raw = {
        "studyName": "CSR-Val_2015_Mead",
        "trialName": "CSR Validation",
        "studyDbId": "5710",
        "seasons": ["2015"],
        "locationName": "Ithaca, NE",
        "locationDbId": "117",
        "startDate": "2014-10-06T00:00:00Z",
        "endDate": "2015-07-18T00:00:00Z",
        "studyType": "phenotyping_trial",
        "additionalInfo": {"programName": "University of Nebraska", "programDbId": "349"},
    }
    env = t3._study_env_fields(raw)
    assert env["year"] == "2015"
    assert env["location_name"] == "Ithaca, NE"
    assert env["location_db_id"] == "117"
    assert env["planting_date"] == "2014-10-06T00:00:00Z"
    assert env["harvest_date"] == "2015-07-18T00:00:00Z"
    assert env["program_name"] == "University of Nebraska"
    assert env["study_type"] == "phenotyping_trial"


def test_is_phenology_trait_t3_ontology_names():
    """T3 variable names carry a |CO_321:... ontology suffix; detection and
    display must still work, and the suffix is stripped."""
    var = _pheno_var("Anthesis time - Julian date (JD)|CO_321:0501001", "v1", unit="JD")
    assert t3.is_phenology_trait(var)
    assert t3.variable_name(var) == "Anthesis time - Julian date (JD)"


def test_is_phenology_trait_ignores_description_keywords():
    """Trait descriptions contain stage words ('...at maturity...'); they must
    not trigger detection (regression: Spike shattering false positive)."""
    var = {
        "observationVariableName": "Spike shattering - 0-9 percentage scale|CO_321:0501143",
        "trait": {
            "traitName": "Spike shattering - 0-9 percentage scale",
            "traitDescription": "Observation of grains dehiscence from spike at maturity.",
        },
        "scale": {"unit": "0-9"},
    }
    assert t3.is_phenology_trait(var) is False
    # Same variable with a genuinely phenological name still matches.
    var2 = dict(var)
    var2["observationVariableName"] = "Heading time - Julian date (JD)|CO_321:0001233"
    var2["trait"]["traitName"] = "Heading time - Julian date (JD)"
    assert t3.is_phenology_trait(var2) is True


def test_census_flags_phenology_and_filters_program():
    routes = {
        "programs": {"result": {"data": [_program("University of Nebraska", "P1"), _program("Other", "P2")], "pagination": {}}},
        "studies": {"result": {"data": [_study("Spring Trial", "S1")], "pagination": {}}},
        "variables": {
            "result": {
                "data": [
                    _pheno_var("Days to heading", "VH"),
                    _pheno_var("Grain yield - kg/ha", "VY", unit="kg/ha"),
                ],
                "pagination": {},
            }
        },
    }
    client = _client_with(routes)
    catalog, pheno_by_study = t3.build_trials_catalog(client, program_filter="Nebraska")
    assert len(catalog) == 1
    row = catalog.iloc[0]
    assert bool(row["has_phenology"]) is True
    assert row["n_phenology_variables"] == 1
    assert "Days to heading" in row["phenology_traits"]
    assert pheno_by_study["S1"][0]["name"] == "Days to heading"


def test_export_phenology_observations_shape():
    routes = {
        "programs": {"result": {"data": [_program("Nebraska", "P1")], "pagination": {}}},
        "studies": {"result": {"data": [_study("Spring Trial", "S1")], "pagination": {}}},
        "variables": {"result": {"data": [_pheno_var("Days to heading", "VH")], "pagination": {}}},
        "observations": {
            "result": {
                "data": [
                    {"observationUnitName": "plot1", "germplasmName": "G1", "value": "58"},
                    {"observationUnitName": "plot2", "germplasmName": "G2", "value": "61"},
                ],
                "pagination": {},
            }
        },
    }
    client = _client_with(routes)
    catalog, pheno_by_study = t3.build_trials_catalog(client)
    obs = t3.export_phenology_observations(client, catalog, pheno_by_study)
    assert len(obs) == 2
    assert set(obs.columns) >= {"plot", "germplasm", "trait", "value", "year"}
    assert obs.iloc[0]["trait"] == "Days to heading"
    assert obs.iloc[0]["value"] == "58"


def test_base_url_normalizes_to_brapi_v2():
    """Root URLs get /brapi/v2/ appended (bug: was hitting web UI HTML)."""
    captured: dict = {}

    def capture_get(url: str, params: dict | None = None) -> dict:
        captured["url"] = url
        return {"result": {"data": [], "pagination": {}}}

    client = t3.BrApiClient(base_url="https://wheat.triticeaetoolbox.org/", _getter=capture_get)
    client.list_programs()
    assert captured["url"].startswith("https://wheat.triticeaetoolbox.org/brapi/v2/programs")

    # Passing a full base already containing brapi/v2 is not double-appended.
    client2 = t3.BrApiClient(base_url="https://wheat.triticeaetoolbox.org/brapi/v2/", _getter=capture_get)
    client2.list_programs()
    assert captured["url"].startswith("https://wheat.triticeaetoolbox.org/brapi/v2/programs")


def test_census_degrades_gracefully_on_study_variable_failure():
    """A 500 on one study's variable fetch must not abort the census."""
    import urllib.error

    routes = {
        "programs": {"result": {"data": [_program("Nebraska", "P1")], "pagination": {}}},
        "studies": {"result": {"data": [_study("S1", "S1"), _study("S2", "S2")], "pagination": {}}},
        "variables": {"result": {"data": [], "pagination": {}}},
    }

    def flaky_get(url: str, params: dict | None = None) -> dict:
        # S1's variable fetch fails on BOTH the primary and fallback endpoints;
        # S2 and everything else succeed.  (The mock receives url and params
        # separately, so check params and the fallback path.)
        if (params or {}).get("studyDbId") == "S1" or "studies/S1/observationvariables" in url:
            raise urllib.error.HTTPError(url, 500, "Internal Server Error", {}, None)
        path = url.split("/")[-1].split("?")[0]
        return routes.get(path, routes["variables"])

    client = t3.BrApiClient(base_url="https://example.test/", sleep_seconds=0,
                            max_retries=1, retry_backoff=0, _getter=flaky_get)
    catalog, _ = t3.build_trials_catalog(client)
    assert len(catalog) == 2  # census continues past the failing study
    s1 = catalog[catalog["study_db_id"] == "S1"].iloc[0]
    s2 = catalog[catalog["study_db_id"] == "S2"].iloc[0]
    assert s1["variables_error"] != ""
    assert s2["variables_error"] == ""


def test_http_get_retries_on_transient_errors():
    """Retries ConnectionReset / 502, but not fatal 401."""
    attempts = {"n": 0}

    def flaky_get(url: str, params: dict | None = None) -> dict:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionResetError("forcefully closed")
        return {"result": {"data": [], "pagination": {}}}

    client = t3.BrApiClient(base_url="https://example.test/", sleep_seconds=0,
                            max_retries=5, _getter=flaky_get)
    assert client.list_programs() == []  # empty data after successful retry
    assert attempts["n"] == 3  # two failures then success


def test_http_get_does_not_retry_on_401():
    attempts = {"n": 0}

    def auth_fail_get(url: str, params: dict | None = None) -> dict:
        attempts["n"] += 1
        raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)

    client = t3.BrApiClient(base_url="https://example.test/", sleep_seconds=0,
                            max_retries=5, _getter=auth_fail_get)
    import urllib.error
    with pytest.raises(urllib.error.HTTPError):
        client.list_programs()
    assert attempts["n"] == 1


def test_no_trials_matched_returns_empty_catalog():
    routes = {
        "programs": {"result": {"data": [_program("Nebraska", "P1")], "pagination": {}}},
        "studies": {"result": {"data": [], "pagination": {}}},
    }
    client = _client_with(routes)
    catalog, _ = t3.build_trials_catalog(client)
    assert len(catalog) == 0


def test_login_fetch_token_and_extract():
    """Verify the login helper parses the real T3 v1 token response shape."""
    import importlib
    import importlib.util as iu

    # Load t3_login.py as a sibling module under a registered name.
    login_script = Path(__file__).resolve().parents[1] / "scripts" / "t3_login.py"
    lspec = iu.spec_from_file_location("t3_login", login_script)
    login_mod = iu.module_from_spec(lspec)
    sys.modules["t3_login"] = login_mod
    lspec.loader.exec_module(login_mod)

    # Response shape matches what T3 v1 returned in the connectivity probe.
    payload = {
        "expires_in": 7200,
        "access_token": "tok123",
        "userDisplayName": "user",
        "metadata": {"status": []},
    }
    assert login_mod._extract_access_token(payload) == "tok123"
    # Nested variant (some v2 deployments).
    nested = {"result": {"access_token": "tok456"}}
    assert login_mod._extract_access_token(nested) == "tok456"
    assert login_mod._extract_access_token({"result": {}}) is None


def test_resolve_token_priority(tmp_path, monkeypatch):
    """--token beats env beats token file; each fallback works."""
    # 1. explicit token wins
    assert t3._resolve_token("explicit") == "explicit"

    # 2. env var used when no explicit token
    monkeypatch.setenv("T3_TOKEN", "from_env")
    assert t3._resolve_token(None) == "from_env"

    # 3. token file used when nothing else set
    monkeypatch.delenv("T3_TOKEN", raising=False)
    token_file = tmp_path / ".t3_token"
    token_file.write_text("from_file", encoding="utf-8")
    monkeypatch.setattr(t3, "DEFAULT_TOKEN_FILE", token_file)
    assert t3._resolve_token(None) == "from_file"

    # 4. nothing set -> None
    monkeypatch.setattr(t3, "DEFAULT_TOKEN_FILE", tmp_path / "missing_token")
    assert t3._resolve_token(None) is None
