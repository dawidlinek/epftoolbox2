import functools
import json
import re
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from epftoolbox2.data.sources import entsoe as entsoe_module
from epftoolbox2.data.sources.entsoe import (
    EntsoeSource,
    lookup_area,
    AREAS,
)

KEY = "test-key-0000-secret"
PL = AREAS["PL"]
AT = AREAS["AT"]
DE_LU = AREAS["DE_LU"]
FIXTURES = Path(__file__).parent / "fixtures" / "entsoe"

# ---------------------------------------------------------------------------------------------------------------
# Synthetic XML builders and a fake HTTP session
# ---------------------------------------------------------------------------------------------------------------


def _period(start, end, resolution, points, tag="quantity"):
    pts = "".join(f"<Point><position>{p}</position><{tag}>{v}</{tag}></Point>" for p, v in points.items())
    return (
        f"<Period><timeInterval><start>{start}</start><end>{end}</end></timeInterval>"
        f"<resolution>{resolution}</resolution>{pts}</Period>"
    )


def _series(*periods, curve="A03", business_type=None, psr=None, in_bz=None, out_bz=None, sequence=None):
    parts = ["<TimeSeries><mRID>1</mRID>"]
    if business_type:
        parts.append(f"<businessType>{business_type}</businessType>")
    if in_bz:
        parts.append(f'<inBiddingZone_Domain.mRID codingScheme="A01">{in_bz}</inBiddingZone_Domain.mRID>')
    if out_bz:
        parts.append(f'<outBiddingZone_Domain.mRID codingScheme="A01">{out_bz}</outBiddingZone_Domain.mRID>')
    if sequence is not None:
        parts.append(
            "<classificationSequence_AttributeInstanceComponent.position>"
            f"{sequence}</classificationSequence_AttributeInstanceComponent.position>"
        )
    if curve:
        parts.append(f"<curveType>{curve}</curveType>")
    if psr:
        parts.append(f"<MktPSRType><psrType>{psr}</psrType></MktPSRType>")
    parts.extend(periods)
    parts.append("</TimeSeries>")
    return "".join(parts)


def _doc(*series, root="GL_MarketDocument"):
    ns = "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"
    return f'<?xml version="1.0" encoding="utf-8"?>\n<{root} xmlns="{ns}"><mRID>x</mRID>{"".join(series)}</{root}>'


def _ack(text):
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<Acknowledgement_MarketDocument '
        'xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0"><mRID>1</mRID>'
        f"<Reason><code>999</code><text>{text}</text></Reason></Acknowledgement_MarketDocument>"
    )


NO_DATA = _ack("No matching data found for Data item X and interval Y.")


def _response(params, text, status=200, headers=None, reason="OK"):
    """A real requests.Response, with the URL requests would build from `params`."""
    response = requests.Response()
    response.status_code = status
    response._content = text.encode()
    response.headers.update({"Content-Type": "text/xml", **(headers or {})})
    response.url = requests.Request("GET", EntsoeSource.API_URL, params=params).prepare().url
    response.reason = reason
    return response


class FakeSession:
    """Routes session.get calls to a function of the query parameters and records them."""

    def __init__(self, route):
        self.route = route
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        result = self.route(dict(params or {}))
        if isinstance(result, Exception):
            raise result
        if isinstance(result, requests.Response):
            return result
        return _response(params, result if result is not None else NO_DATA)

    def close(self):
        pass


def _source(route, country="PL", type=("price",), prefer_15min=False, **kwargs):
    source = EntsoeSource(country, KEY, type=list(type), prefer_15min=prefer_15min, **kwargs)
    source.session = FakeSession(route)
    source.logger = Mock()
    return source


def _route_by_document(docs):
    """docs: {(documentType, processType or None): xml}"""

    def route(params):
        return docs.get((params.get("documentType"), params.get("processType")))

    return route


def _utc(ts):
    return pd.Timestamp(ts, tz="UTC")


def _hours(start, n):
    return {p: float(start + p) for p in range(1, n + 1)}


# ---------------------------------------------------------------------------------------------------------------
# Area lookup and configuration
# ---------------------------------------------------------------------------------------------------------------


class TestAreaLookup:
    def test_lookup_valid_area(self):
        name, code = lookup_area("PL")
        assert name == "PL"
        assert code == "10YPL-AREA-----S"

    def test_lookup_is_case_insensitive(self):
        assert lookup_area("pl") == lookup_area("PL")

    def test_lookup_invalid_area(self):
        with pytest.raises(ValueError, match="Invalid country code"):
            lookup_area("INVALID")

    def test_all_areas_have_codes(self):
        for name, code in AREAS.items():
            assert isinstance(name, str)
            assert isinstance(code, str)
            assert len(code) > 0


class TestEntsoeSourceInit:
    def test_init_valid_params(self):
        source = EntsoeSource("PL", "test-api-key", type=["load"])
        assert source._area_name == "PL"
        assert source._area_code == "10YPL-AREA-----S"
        assert source.api_key == "test-api-key"
        assert source.type == ["load"]

    def test_init_multiple_types(self):
        source = EntsoeSource("DE_LU", "test-key", type=["load", "generation", "price"])
        assert set(source.type) == {"load", "generation", "price"}

    def test_init_invalid_country(self):
        with pytest.raises(ValueError, match="Invalid country code"):
            EntsoeSource("XX", "test-key", type=["load"])

    def test_init_empty_api_key(self):
        with pytest.raises(ValueError, match="API key cannot be empty"):
            EntsoeSource("PL", "", type=["load"])

    def test_init_empty_types(self):
        with pytest.raises(ValueError, match="At least one data type must be specified"):
            EntsoeSource("PL", "test-key", type=[])

    def test_init_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid type"):
            EntsoeSource("PL", "test-key", type=["invalid_type"])

    def test_validate_config_returns_true(self):
        assert EntsoeSource("PL", "test-api-key", type=["load"])._validate_config() is True

    def test_fetch_validates_timestamps(self):
        source = EntsoeSource("PL", "test-key", type=["load"])
        with pytest.raises(ValueError, match="End timestamp.*must be after start timestamp"):
            source.fetch(_utc("2024-01-02"), _utc("2024-01-01"))

    def test_price_for_country_that_is_not_a_bidding_zone_raises_with_hint(self):
        # "DE" (Germany as a country) has no day-ahead price; the API answers "No matching data" after a long wait.
        with pytest.raises(ValueError, match="DE_LU"):
            EntsoeSource("DE", "test-key", type=["price"])

    def test_load_for_country_code_is_allowed(self):
        EntsoeSource("DE", "test-key", type=["load"])

    def test_cache_config_includes_every_output_parameter(self):
        hourly = EntsoeSource("PL", "k", type=["price", "load"]).get_cache_config()
        quarter = EntsoeSource("PL", "k", type=["load", "price"], prefer_15min=True).get_cache_config()
        assert hourly["prefer_15min"] is False and quarter["prefer_15min"] is True
        assert hourly["types"] == ["load", "price"]
        assert hourly["schema_version"] >= 2
        assert "api_key" not in hourly and "k" not in json.dumps(hourly)


class TestEntsoeSourceChunking:
    def test_generate_chunks_single_month(self):
        source = EntsoeSource("PL", "test-key", type=["load"])
        chunks = source._generate_chunks(_utc("2024-01-01"), _utc("2024-01-15"), months=3)
        assert chunks == [(_utc("2024-01-01"), _utc("2024-01-15"))]

    def test_generate_chunks_multiple_months(self):
        source = EntsoeSource("PL", "test-key", type=["load"])
        chunks = source._generate_chunks(_utc("2024-01-01"), _utc("2024-07-01"), months=3)
        assert len(chunks) == 2
        assert chunks[0][0] == _utc("2024-01-01")
        assert chunks[-1][1] == _utc("2024-07-01")

    def test_generate_chunks_exact_boundary(self):
        source = EntsoeSource("PL", "test-key", type=["load"])
        assert len(source._generate_chunks(_utc("2024-01-01"), _utc("2024-04-01"), months=3)) == 1

    def test_generate_chunks_validates_months_parameter(self):
        source = EntsoeSource("PL", "test-key", type=["load"])
        for months in (0, -1):
            with pytest.raises(ValueError, match="months parameter must be positive"):
                source._generate_chunks(_utc("2024-01-01"), _utc("2024-07-01"), months=months)


# ---------------------------------------------------------------------------------------------------------------
# HTTP: authentication, errors, retries
# ---------------------------------------------------------------------------------------------------------------


class TestApiRequest:
    def _request(self, source, start="2024-01-01", end="2024-01-02"):
        return source._api_request({"documentType": "A65", "processType": "A16"}, _utc(start), _utc(end))

    def test_returns_text_on_success(self):
        source = _source(lambda p: "<data>test</data>", type=["load"])
        assert self._request(source) == "<data>test</data>"

    def test_no_matching_data_returns_none_and_warns(self):
        source = _source(lambda p: NO_DATA, type=["load"])
        assert self._request(source) is None
        assert source.logger.warning.called
        assert "no matching data" in source.logger.warning.call_args[0][0].lower()

    def test_no_matching_data_with_http_400_returns_none(self):
        source = _source(lambda p: _response(p, NO_DATA, status=400, reason="Bad Request"), type=["load"])
        assert self._request(source) is None

    def test_api_key_is_sent_as_header_not_in_url(self):
        source = _source(lambda p: "<ok/>", type=["load"])
        self._request(source)
        call = source.session.calls[0]
        assert call["headers"].get("SECURITY_TOKEN") == KEY
        assert "securityToken" not in call["params"]
        assert KEY not in json.dumps(call["params"])

    def test_period_keeps_minutes(self):
        source = _source(lambda p: "<ok/>", type=["load"])
        self._request(source, "2024-01-01 00:15", "2024-01-01 23:45")
        params = source.session.calls[0]["params"]
        assert params["periodStart"] == "202401010015"
        assert params["periodEnd"] == "202401012345"

    def test_http_error_carries_acknowledgement_reason_and_not_the_key(self):
        source = _source(lambda p: _response(p, _ack("Mandatory parameter ProcessType is missing."), status=400))
        with pytest.raises(Exception) as info:
            self._request(source)
        message = str(info.value)
        assert "Mandatory parameter ProcessType is missing." in message
        assert KEY not in message
        assert KEY not in repr(info.value.__cause__) and KEY not in repr(info.value.__context__)

    def test_authentication_failure_is_not_retried(self):
        source = _source(lambda p: _response(p, _ack("Authentication failed."), status=401))
        with patch.object(entsoe_module.time, "sleep") as sleep, pytest.raises(Exception, match="Authentication"):
            self._request(source)
        assert len(source.session.calls) == 1
        sleep.assert_not_called()

    def test_retries_429_honouring_retry_after(self):
        responses = iter(["429", "ok"])

        def route(p):
            if next(responses) == "429":
                return _response(p, "Too many requests", status=429, headers={"Retry-After": "7"})
            return "<ok/>"

        source = _source(route, type=["load"])
        with patch.object(entsoe_module.time, "sleep") as sleep:
            assert self._request(source) == "<ok/>"
        sleep.assert_called_once_with(7.0)
        assert len(source.session.calls) == 2

    def test_retries_server_errors_with_backoff_then_raises(self):
        source = _source(lambda p: _response(p, "Service Unavailable", status=503), type=["load"])
        with patch.object(entsoe_module.time, "sleep") as sleep, pytest.raises(Exception, match="503"):
            self._request(source)
        assert len(source.session.calls) == EntsoeSource.MAX_RETRIES + 1
        waits = [c.args[0] for c in sleep.call_args_list]
        assert waits == sorted(waits) and waits[0] > 0

    @pytest.mark.parametrize(
        "error_type", [requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError]
    )
    def test_network_errors_are_retried_and_sanitised(self, error_type):
        error = error_type(f"HTTPSConnectionPool: Max retries exceeded (token {KEY})")
        source = _source(lambda p: error, type=["load"])
        with patch.object(entsoe_module.time, "sleep"), pytest.raises(Exception) as info:
            self._request(source)
        assert len(source.session.calls) == EntsoeSource.MAX_RETRIES + 1
        assert KEY not in str(info.value)
        assert info.value.__cause__ is None and info.value.__context__ is None

    def test_failed_chunk_makes_fetch_raise(self):
        # A chunk that keeps failing must not silently produce a frame with a gap.
        def route(p):
            if p["periodStart"].startswith("202404"):
                return _response(p, "Bad Gateway", status=502)
            return _doc(_series(_period("2024-01-01T00:00Z", "2024-01-01T01:00Z", "PT60M", {1: 1.0}, "price.amount")))

        source = _source(route)
        with patch.object(entsoe_module.time, "sleep"), pytest.raises(Exception, match="502"):
            source.fetch(_utc("2024-01-01"), _utc("2024-06-01"))

    def test_xml_with_doctype_is_rejected(self):
        evil = '<?xml version="1.0"?><!DOCTYPE a [<!ENTITY b "c">]><Publication_MarketDocument/>'
        source = _source(lambda p: evil)
        with pytest.raises(Exception, match="DTD"):
            source.fetch(_utc("2024-01-01"), _utc("2024-01-02"))


# ---------------------------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------------------------


class TestCurveTypeA03:
    def test_omitted_positions_repeat_previous_value_including_trailing(self):
        xml = _doc(_series(_period("2024-01-01T00:00Z", "2024-01-01T06:00Z", "PT60M", {1: 10, 2: 20, 5: 50})))
        source = _source(_route_by_document({("A65", "A16"): xml}), type=["load"])
        df = source.fetch(_utc("2024-01-01"), _utc("2024-01-02"))
        assert df["load_actual"].tolist() == [10, 20, 20, 20, 50, 50]
        assert df.index[0] == _utc("2024-01-01 00:00") and df.index[-1] == _utc("2024-01-01 05:00")

    def test_a01_curve_is_not_filled(self):
        xml = _doc(_series(_period("2024-01-01T00:00Z", "2024-01-01T03:00Z", "PT60M", {1: 10, 3: 30}), curve="A01"))
        source = _source(_route_by_document({("A65", "A16"): xml}), type=["load"])
        df = source.fetch(_utc("2024-01-01"), _utc("2024-01-02"))
        assert df["load_actual"].dropna().tolist() == [10, 30]

    def test_a03_fill_applies_to_prices_and_generation(self):
        price = _doc(
            _series(_period("2024-01-01T00:00Z", "2024-01-01T04:00Z", "PT60M", {1: 1, 3: 3}, "price.amount")),
            root="Publication_MarketDocument",
        )
        gen = _doc(_series(_period("2024-01-01T00:00Z", "2024-01-01T04:00Z", "PT60M", {1: 5}), psr="B16", in_bz=PL))
        source = _source(_route_by_document({("A44", None): price, ("A75", "A16"): gen}), type=["price", "generation"])
        df = source.fetch(_utc("2024-01-01"), _utc("2024-01-02"))
        assert df["price"].tolist() == [1, 1, 3, 3]
        assert df["generation_solar"].tolist() == [5, 5, 5, 5]

    def test_daily_positions_follow_the_local_calendar_across_dst(self):
        # A week period with the autumn switch lasts 169 hours; (end - start) / 24 h would give 7.04 positions and
        # fixed 24-hour steps would put Sunday's value at 23:00 local on Saturday.
        week = _doc(
            _series(
                _period("2024-10-20T22:00Z", "2024-10-27T23:00Z", "P1D", {1: 100, 2: 200, 6: 600, 7: 700}),
                business_type="A60",
                out_bz=PL,
            )
        )
        source = _source(_route_by_document({("A65", "A31"): week}), type=["load"])
        df = source.fetch(_utc("2024-10-20 22:00"), _utc("2024-10-27 23:00"))
        values = df["load_forecast_daily_min"].dropna()
        local = values.index.tz_convert("Europe/Warsaw")
        assert (local == local.normalize()).all()
        assert list(local.day) == [21, 22, 23, 24, 25, 26, 27]
        assert values.tolist() == [100, 200, 200, 200, 200, 600, 700]


class TestPrices:
    def _two_days(self, prefer_15min, sequence=None):
        hourly = _period("2025-09-29T22:00Z", "2025-09-30T22:00Z", "PT60M", _hours(0, 24), "price.amount")
        quarters = {p: float(1000 + p) for p in range(1, 97)}
        quarter = _period("2025-09-30T22:00Z", "2025-10-01T22:00Z", "PT15M", quarters, "price.amount")
        xml = _doc(_series(hourly, sequence=sequence), _series(quarter, sequence=sequence), root="P")
        source = _source(_route_by_document({("A44", None): xml}), prefer_15min=prefer_15min)
        return source.fetch(_utc("2025-09-29 22:00"), _utc("2025-10-01 22:00"))["price"]

    def test_hourly_output_keeps_both_sides_of_the_15min_switch(self):
        price = self._two_days(prefer_15min=False)
        assert len(price) == 48 and price.notna().all()
        assert (price.index[1:] - price.index[:-1] == pd.Timedelta("1h")).all()
        assert price[_utc("2025-09-29 22:00")] == 1.0
        # the first hour after the switch is the mean of its four quarter-hours 1001..1004
        assert price[_utc("2025-09-30 22:00")] == pytest.approx(1002.5)

    def test_15min_output_keeps_both_sides_of_the_15min_switch(self):
        price = self._two_days(prefer_15min=True)
        assert len(price) == 192 and price.notna().all()
        assert (price.index[1:] - price.index[:-1] == pd.Timedelta("15min")).all()
        # hourly periods are step-filled: every quarter-hour gets its hour's price
        assert price[_utc("2025-09-29 22:00") : _utc("2025-09-29 22:45")].tolist() == [1.0] * 4
        assert price[_utc("2025-09-30 22:00")] == 1001.0

    def test_single_classification_sequence_one_is_kept(self):
        assert len(self._two_days(prefer_15min=False, sequence=1)) == 48

    def test_only_the_sdac_sequence_is_used(self):
        # DE_LU/AT publish a second 15-minute series (sequence 2) on the same timestamps; it is a different auction.
        sdac = _period("2024-07-01T22:00Z", "2024-07-02T22:00Z", "PT60M", _hours(0, 24), "price.amount")
        other = _period(
            "2024-07-01T22:00Z", "2024-07-02T22:00Z", "PT15M", {p: 999.0 for p in range(1, 97)}, "price.amount"
        )
        xml = _doc(_series(other, sequence=2), _series(sdac, sequence=1), root="P")
        for prefer_15min, rows in ((False, 24), (True, 96)):
            source = _source(_route_by_document({("A44", None): xml}), country="DE_LU", prefer_15min=prefer_15min)
            price = source.fetch(_utc("2024-07-01 22:00"), _utc("2024-07-02 22:00"))["price"]
            assert len(price) == rows
            assert 999.0 not in price.values
            assert price.iloc[0] == 1.0 and price.iloc[-1] == 24.0

    def test_incomplete_hour_is_not_averaged(self):
        quarters = {1: 1.0, 2: 2.0, 3: 3.0, 5: 5.0, 6: 6.0, 7: 7.0, 8: 8.0}
        xml = _doc(
            _series(_period("2025-10-01T00:00Z", "2025-10-01T02:00Z", "PT15M", quarters, "price.amount"), curve="A01"),
            root="P",
        )
        source = _source(_route_by_document({("A44", None): xml}))
        price = source.fetch(_utc("2025-10-01"), _utc("2025-10-02"))["price"]
        assert price.dropna().tolist() == [6.5]

    def test_chunk_boundary_does_not_drop_the_straddled_hour(self):
        # A 3-month chunk boundary that lands mid-hour (e.g. a request that doesn't start on the hour) splits one
        # hour's four quarter-hours across two separate API responses. Gridding per chunk sees only two of the four
        # quarter-hours in each response, so neither chunk can average a complete hour and it is dropped entirely.
        boundary = _utc("2024-01-01 00:30")
        first_half = _doc(
            _series(
                _period("2024-01-01T00:00Z", "2024-01-01T00:30Z", "PT15M", {1: 10.0, 2: 20.0}, "price.amount"),
                curve="A01",
            )
        )
        second_half = _doc(
            _series(
                _period("2024-01-01T00:30Z", "2024-01-01T01:00Z", "PT15M", {1: 30.0, 2: 40.0}, "price.amount"),
                curve="A01",
            )
        )

        def route(params):
            req_start = pd.to_datetime(params["periodStart"], format="%Y%m%d%H%M", utc=True)
            return first_half if req_start < boundary else second_half

        source = _source(route)
        source._generate_chunks = lambda start, end, months=3: [(start, boundary), (boundary, end)]

        df = source.fetch(_utc("2024-01-01 00:00"), _utc("2024-01-01 01:00"))
        assert "price" in df.columns
        price = df["price"]
        assert list(price.index) == [_utc("2024-01-01 00:00")]
        assert price.iloc[0] == pytest.approx(25.0)


class TestGeneration:
    def test_generation_and_consumption_are_separate_columns(self):
        period = "2024-03-30T00:00Z", "2024-03-30T02:00Z", "PT60M"
        xml = _doc(
            _series(_period(*period, {1: 100, 2: 110}), psr="B10", in_bz=AT),
            _series(_period(*period, {1: 0, 2: 55}), psr="B10", out_bz=AT),
            _series(_period(*period, {1: 7}), psr="B01", in_bz=AT),
            _series(_period(*period, {1: 0}), psr="B01", out_bz=AT),
        )
        source = _source(_route_by_document({("A75", "A16"): xml}), country="AT", type=["generation"])
        df = source.fetch(_utc("2024-03-30"), _utc("2024-03-31"))
        assert df["generation_hydro_pumped_storage"].tolist() == [100, 110]
        assert df["generation_hydro_pumped_storage_consumption"].tolist() == [0, 55]
        assert df["generation_biomass"].tolist() == [7, 7]
        assert df["generation_biomass_consumption"].tolist() == [0, 0]

    @pytest.mark.parametrize(
        "psr, column",
        [
            ("B02", "generation_fossil_brown_coal_lignite"),
            ("B03", "generation_fossil_coal_derived_gas"),
            ("B07", "generation_fossil_oil_shale"),
            ("B08", "generation_fossil_peat"),
            ("B11", "generation_hydro_run_of_river_and_pondage"),
            ("B25", "generation_energy_storage"),
            ("B99", "generation_b99"),
        ],
    )
    def test_production_type_column_names(self, psr, column):
        xml = _doc(_series(_period("2024-01-01T00:00Z", "2024-01-01T01:00Z", "PT60M", {1: 1}), psr=psr, in_bz=PL))
        source = _source(_route_by_document({("A75", "A16"): xml}), type=["generation"])
        assert list(source.fetch(_utc("2024-01-01"), _utc("2024-01-02")).columns) == [column]

    def test_wind_solar_forecast_columns(self):
        xml = _doc(_series(_period("2024-01-01T00:00Z", "2024-01-01T01:00Z", "PT60M", {1: 3}), psr="B19", in_bz=PL))
        source = _source(_route_by_document({("A69", "A01"): xml}), type=["generation"])
        df = source.fetch(_utc("2024-01-01"), _utc("2024-01-02"))
        assert list(df.columns) == ["generation_wind_onshore_forecast"]


class TestFetchRange:
    def test_returns_half_open_range(self):
        xml = _doc(_series(_period("2024-01-01T00:00Z", "2024-01-01T04:00Z", "PT60M", _hours(0, 4), "price.amount")))
        source = _source(_route_by_document({("A44", None): xml}))
        price = source.fetch(_utc("2024-01-01 00:00"), _utc("2024-01-01 02:00"))["price"]
        assert list(price.index) == [_utc("2024-01-01 00:00"), _utc("2024-01-01 01:00")]

    def test_daily_value_covering_start_is_reported_at_start(self):
        # The local day 2024-01-02 (Warsaw) starts at 2024-01-01 23:00Z, before a request that starts at UTC midnight.
        week = _doc(
            _series(_period("2024-01-01T23:00Z", "2024-01-03T23:00Z", "P1D", {1: 111, 2: 222}), business_type="A61")
        )
        source = _source(_route_by_document({("A65", "A31"): week}), type=["load"])
        df = source.fetch(_utc("2024-01-02 00:00"), _utc("2024-01-04 00:00"))
        daily = df["load_forecast_daily_max"].dropna()
        assert daily.to_dict() == {_utc("2024-01-02 00:00"): 111, _utc("2024-01-02 23:00"): 222}
        assert df.index.min() >= _utc("2024-01-02 00:00")

    def test_warns_when_coverage_starts_long_after_start(self):
        xml = _doc(_series(_period("2024-03-01T00:00Z", "2024-03-01T01:00Z", "PT60M", {1: 1.0}, "price.amount")))
        source = _source(_route_by_document({("A44", None): xml}))
        source.fetch(_utc("2024-01-01"), _utc("2024-03-02"))
        messages = " ".join(str(c.args[0]) for c in source.logger.warning.call_args_list)
        assert "2024-03-01" in messages and "starts" in messages

    def test_parses_real_response_with_two_sequences(self):
        # 3 days of DE_LU prices: sequence 1 (SDAC, hourly then 15-minute) and sequence 2 (15-minute) per day.
        text = (FIXTURES / "DE_LU_A44_2025-09-30_3d.xml").read_text()
        source = EntsoeSource("DE_LU", KEY, type=["price"])
        parts = source._parse_timeseries(text, "price.amount")
        assert len(parts) == 6
        assert {meta["sequence"] for meta, _ in parts} == {"1", "2"}


# ---------------------------------------------------------------------------------------------------------------
# Known answers: recorded API responses vs. an independent decoding (tests/data/sources/fixtures/entsoe/)
# ---------------------------------------------------------------------------------------------------------------

_FIXTURE_NAME = re.compile(r"^(?P<zone>[A-Z_]+?)_(?P<doc>A\d\d)(?:-(?P<proc>A\d\d))?_\d{4}-\d\d-\d\d_\d+d\.xml$")
_INTERVAL = re.compile(r"[pP]eriod\.timeInterval>\s*<start>(.*?)</start>\s*<end>(.*?)</end>")
REFERENCE = json.loads((FIXTURES / "reference.json").read_text())["cases"]


@functools.lru_cache(maxsize=1)
def _fixture_index():
    index = []
    for path in sorted(FIXTURES.glob("*.xml")):
        m = _FIXTURE_NAME.match(path.name)
        text = path.read_text()
        start, end = (pd.Timestamp(t) for t in _INTERVAL.search(text).groups())
        index.append((AREAS[m["zone"]], m["doc"], m["proc"], start, end, text))
    return index


def _fixture_route(params):
    """Serve the smallest recorded response that covers the requested period, else "No matching data"."""
    start = pd.Timestamp(params["periodStart"], tz="UTC")
    end = pd.Timestamp(params["periodEnd"], tz="UTC")
    area = params.get("in_Domain") or params.get("outBiddingZone_Domain")
    hits = [
        (e - s, text)
        for code, doc, proc, s, e, text in _fixture_index()
        if code == area and doc == params["documentType"] and proc == params.get("processType") and s <= start
        if end <= e
    ]
    return min(hits, key=lambda h: h[0])[1] if hits else NO_DATA


def _local_window(tz, day, days):
    start = pd.Timestamp(day, tz=tz)
    end = pd.Timestamp(pd.Timestamp(day) + pd.Timedelta(days=days), tz=tz)
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def _reference_series(case):
    values = {pd.Timestamp(t): v for t, v in case["values"].items()}
    return pd.Series(values, dtype=float).sort_index()


def _on_price_grid(series, resolution, prefer_15min):
    """What the source should return for a reference price series at its native resolution."""
    native = pd.Timedelta("1h") if resolution == "PT60M" else pd.Timedelta("15min")
    target = pd.Timedelta("15min") if prefer_15min else pd.Timedelta("1h")
    if native == target:
        return series
    if native > target:
        k = int(native / target)
        index = series.index.repeat(k) + np.tile(np.arange(k) * target, len(series))
        return pd.Series(np.repeat(series.values, k), index=index)
    return series.resample(target).mean()


def _column_type(column):
    return "price" if column == "price" else ("load" if column.startswith("load") else "generation")


def _known_answer_params():
    params = []
    for case in REFERENCE:
        for prefer_15min in (False, True) if case["library_column"] == "price" else (False,):
            params.append(pytest.param(case, prefer_15min, id=f"{case['case']}-p15={prefer_15min}"))
    return params


@pytest.mark.parametrize("case, prefer_15min", _known_answer_params())
def test_known_answer_single_local_day(case, prefer_15min):
    days = 7 if "local_week_start" in case else 1
    start, end = _local_window(case["local_timezone"], case.get("local_day") or case["local_week_start"], days)
    source = _source(_fixture_route, country=case["zone"], type=[_column_type(case["library_column"])])
    source.prefer_15min = prefer_15min
    df = source.fetch(start, end)

    expected = _reference_series(case)
    if case["library_column"] == "price":
        expected = _on_price_grid(expected, case["resolution"], prefer_15min)
    got = df[case["library_column"]].dropna()
    pd.testing.assert_series_equal(got, expected, check_names=False, check_freq=False, check_index_type=False)
    assert df.index.min() >= start and df.index.max() < end


@pytest.mark.parametrize("zone", ["PL", "DE_LU"])
@pytest.mark.parametrize("prefer_15min", [False, True])
def test_known_answer_chunk_across_15min_go_live(zone, prefer_15min):
    """One request covering 2025-09-30..10-02 (hourly day, then 15-minute days) must give all three days."""
    start, end = _local_window("Europe/Warsaw", "2025-09-30", 3)
    source = _source(_fixture_route, country=zone, prefer_15min=prefer_15min)
    assert len(source._generate_chunks(start, end)) == 1
    price = source.fetch(start, end)["price"]

    cases = [
        c for c in REFERENCE if c["zone"] == zone and c["library_column"] == "price" and c["local_day"] >= "2025-09-30"
    ]
    expected = pd.concat(
        [_on_price_grid(_reference_series(c), c["resolution"], prefer_15min) for c in cases]
    ).sort_index()
    assert len(price) == (288 if prefer_15min else 72)
    assert price.notna().all()
    pd.testing.assert_series_equal(price, expected, check_names=False, check_freq=False, check_index_type=False)


def test_fixtures_do_not_contain_credentials():
    for path in FIXTURES.iterdir():
        text = path.read_text()
        assert "securityToken" not in text and "SECURITY_TOKEN" not in text, path.name


# ---------------------------------------------------------------------------------------------------------------
# Real API
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.integration
class TestEntsoeSourceIntegration:
    def test_fetch_load_data_poland(self, entsoe_api_key):
        source = EntsoeSource("PL", entsoe_api_key, type=["load"])
        df = source.fetch(_utc("2024-01-01"), _utc("2024-01-03"))
        assert len(df) > 0
        assert "load_actual" in df.columns or "load_forecast" in df.columns
        assert df.index.tz is not None

    def test_fetch_price_data_germany(self, entsoe_api_key):
        source = EntsoeSource("DE_LU", entsoe_api_key, type=["price"])
        df = source.fetch(_utc("2024-01-01"), _utc("2024-01-02"))
        assert "price" in df.columns and len(df) == 24

    def test_fetch_price_across_15min_go_live(self, entsoe_api_key):
        for prefer_15min, rows in ((False, 72), (True, 288)):
            source = EntsoeSource("PL", entsoe_api_key, type=["price"], prefer_15min=prefer_15min)
            df = source.fetch(_utc("2025-09-29 22:00"), _utc("2025-10-02 22:00"))
            assert len(df) == rows and df["price"].notna().all()

    def test_fetch_generation_data_france(self, entsoe_api_key):
        source = EntsoeSource("FR", entsoe_api_key, type=["generation"])
        df = source.fetch(_utc("2024-01-01"), _utc("2024-01-02"))
        assert any(col.startswith("generation_") for col in df.columns)

    def test_fetch_all_types(self, entsoe_api_key):
        source = EntsoeSource("PL", entsoe_api_key, type=["load", "generation", "price"])
        df = source.fetch(_utc("2024-01-01"), _utc("2024-01-02"))
        assert any("load" in col for col in df.columns)
        assert any(col.startswith("generation_") for col in df.columns)
        assert "price" in df.columns
