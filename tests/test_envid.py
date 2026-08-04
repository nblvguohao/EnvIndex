"""Tests for the unified environment-id scheme."""

from __future__ import annotations

import pytest

from envindex import (
    is_environment_id,
    make_environment_id,
    parse_environment_id,
    sanitize_native,
)


def test_make_and_parse_roundtrip():
    eid = make_environment_id("wheat", "iwin", "ESWYT_19103_2017")
    assert eid == "wheat:iwin:ESWYT_19103_2017"
    parsed = parse_environment_id(eid)
    assert parsed.crop == "wheat"
    assert parsed.source == "iwin"
    assert parsed.native == "ESWYT_19103_2017"
    assert parsed.to_string() == eid


def test_sanitize_native():
    assert sanitize_native("SDS_2017_AUR_AYT") == "SDS_2017_AUR_AYT"
    assert sanitize_native("2014_MO-Tarkio") == "2014_MO_Tarkio"
    assert sanitize_native("  IA_2012  ") == "IA_2012"
    assert sanitize_native("CIMMYT (El Batan)") == "CIMMYT_El_Batan"


def test_crop_source_lowercased():
    eid = make_environment_id("WHEAT", "CIMMYT", "ESWYT-2017")
    assert eid == "wheat:cimmyt:ESWYT_2017"


def test_missing_field_raises():
    with pytest.raises(ValueError):
        make_environment_id("wheat", "", "x")


def test_parse_invalid():
    assert is_environment_id("wheat:iwin:ESWYT_1") is True
    assert is_environment_id("not-an-env-id") is False
    with pytest.raises(ValueError):
        parse_environment_id("too-short")


def test_native_special_chars_sanitized():
    # Native keys with special characters are sanitized to [a-z0-9_]+.
    eid = make_environment_id("soybean", "soynam", "IA:2012")
    assert eid == "soybean:soynam:IA_2012"
    parsed = parse_environment_id(eid)
    assert parsed.native == "IA_2012"
    assert parsed.to_string() == eid
