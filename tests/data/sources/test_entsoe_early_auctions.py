"""The auctions that close before SDAC: EXAA (`price_exaa`, AT and DE_LU) and the Swiss auction (CH's `price`).

Offline: synthetic A44 documents in the real shape (two TimeSeries told apart by classificationSequence position, 60-minute
and 15-minute periods, both DST switches) behind a fake session. The live checks are the integration tests at the end.
"""

import pandas as pd
import pytest

from epftoolbox2.data.sources.entsoe import AREAS, EntsoeSource
from epftoolbox2.information_set import InformationSet, default_rules, source_rules
from tests.data.sources.test_entsoe import KEY, _doc, _period, _series, _source, _utc

POSITION = "classificationSequence_AttributeInstanceComponent.position"
OLD_REQUEST = {"documentType": "A44", "in_Domain": AREAS["AT"], "out_Domain": AREAS["AT"]}


def _hourly(n, base=0.0):
    return {p: base + p - 1 for p in range(1, n + 1)}


def _quarters(n):
    """Quarter p of hour h = 1000 + 10 h + (p mod 4), so an hour's mean is 1000 + 10 h + 1.5."""
    return {p: 1000.0 + 10 * ((p - 1) // 4) + (p - 1) % 4 for p in range(1, n + 1)}


def _a44(start, end, hours, sequence_one=True, sequence_two=True):
    """An A44 document as ENTSO-E returns it for AT and DE_LU: SDAC hourly (position 1), EXAA 15-minute (position 2)."""
    series = []
    if sequence_two:
        quarters = _period(start, end, "PT15M", _quarters(hours * 4), "price.amount")
        series.append(_series(quarters, curve="A01", sequence=2))
    if sequence_one:
        hourly = _period(start, end, "PT60M", _hourly(hours), "price.amount")
        series.append(_series(hourly, curve="A01", sequence=1))
    return _doc(*series, root="Publication_MarketDocument")


def _route(document, only_position_two=None):
    """Answers a request without a position with `document` and one for position 2 with `only_position_two` (or the
    same document, which the source must then filter itself)."""

    def route(params):
        if params.get(POSITION) == "2":
            return only_position_two if only_position_two is not None else document
        return document

    return route


def _exaa_source(document, country="AT", only_position_two=None, **kwargs):
    kwargs.setdefault("price_exaa", True)
    return _source(_route(document, only_position_two), country=country, **kwargs)


# --------------------------------------------------------------------------------------------------------------
# Request shape, parsing, column naming
# --------------------------------------------------------------------------------------------------------------


class TestExaaColumn:
    def test_adds_a_second_request_for_position_two_and_keeps_the_plain_one(self):
        source = _exaa_source(_a44("2024-07-01T22:00Z", "2024-07-02T22:00Z", 24))
        source.fetch(_utc("2024-07-01 22:00"), _utc("2024-07-02 22:00"))
        calls = [c["params"] for c in source.session.calls]
        assert len(calls) == 2
        plain, exaa = calls
        assert POSITION not in plain
        assert exaa[POSITION] == "2"
        for params in calls:
            assert params["documentType"] == "A44"
            assert params["in_Domain"] == params["out_Domain"] == AREAS["AT"]
            assert params["periodStart"] == "202407012200" and params["periodEnd"] == "202407022200"
        assert {k: v for k, v in exaa.items() if k != POSITION} == plain
        assert all(c["headers"] == {"SECURITY_TOKEN": KEY} for c in source.session.calls)

    def test_hourly_columns_price_is_sdac_and_price_exaa_is_the_mean_of_the_quarters(self):
        source = _exaa_source(_a44("2024-07-01T22:00Z", "2024-07-02T22:00Z", 24))
        df = source.fetch(_utc("2024-07-01 22:00"), _utc("2024-07-02 22:00"))
        assert list(df.columns) == ["price", "price_exaa"]
        assert df["price"].tolist() == [float(h) for h in range(24)]
        assert df["price_exaa"].tolist() == [1000.0 + 10 * h + 1.5 for h in range(24)]

    def test_15min_grid_gives_the_auction_as_published_and_steps_the_hourly_sdac(self):
        source = _exaa_source(_a44("2024-07-01T22:00Z", "2024-07-02T22:00Z", 24), prefer_15min=True)
        df = source.fetch(_utc("2024-07-01 22:00"), _utc("2024-07-02 22:00"))
        assert len(df) == 96 and df.notna().all().all()
        assert df["price_exaa"].tolist() == list(_quarters(96).values())
        assert df["price"].iloc[:4].tolist() == [0.0] * 4

    def test_the_plain_price_never_takes_the_auction_when_position_two_comes_first(self):
        document = _a44("2024-07-01T22:00Z", "2024-07-02T22:00Z", 24)
        df = _exaa_source(document).fetch(_utc("2024-07-01 22:00"), _utc("2024-07-02 22:00"))
        assert df["price"].max() == 23.0 and df["price_exaa"].min() > 1000

    def test_the_position_two_request_keeps_only_position_two_even_if_the_answer_has_both(self):
        document = _a44("2024-07-01T22:00Z", "2024-07-02T22:00Z", 24)
        only_sdac = _a44("2024-07-01T22:00Z", "2024-07-02T22:00Z", 24, sequence_two=False)
        source = _exaa_source(document, only_position_two=only_sdac)
        df = source.fetch(_utc("2024-07-01 22:00"), _utc("2024-07-02 22:00"))
        assert list(df.columns) == ["price"]  # SDAC is never passed off as the auction

    def test_prefix_names_the_column_like_the_others(self):
        source = _exaa_source(
            _a44("2024-07-01T22:00Z", "2024-07-02T22:00Z", 24), country="DE_LU", prefix="de", price_exaa=True
        )
        df = source.fetch(_utc("2024-07-01 22:00"), _utc("2024-07-02 22:00"))
        assert list(df.columns) == ["de_price", "de_price_exaa"]
        assert source.session.calls[0]["params"]["in_Domain"] == AREAS["DE_LU"]

    def test_no_auction_published_leaves_price_alone(self):
        sdac_only = _a44("2024-07-01T22:00Z", "2024-07-02T22:00Z", 24, sequence_two=False)
        df = _exaa_source(sdac_only).fetch(_utc("2024-07-01 22:00"), _utc("2024-07-02 22:00"))
        assert list(df.columns) == ["price"] and len(df) == 24

    @pytest.mark.parametrize(
        "start, end, hours",
        [("2024-03-30T23:00Z", "2024-03-31T22:00Z", 23), ("2024-10-26T22:00Z", "2024-10-27T23:00Z", 25)],
        ids=["spring-23h", "autumn-25h"],
    )
    def test_both_dst_switch_days_are_complete(self, start, end, hours):
        source = _exaa_source(_a44(start, end, hours))
        df = source.fetch(_utc(start.replace("T", " ").rstrip("Z")), _utc(end.replace("T", " ").rstrip("Z")))
        assert len(df) == hours and df.notna().all().all()
        assert (df.index[1:] - df.index[:-1] == pd.Timedelta("1h")).all()
        assert df["price_exaa"].tolist() == [1000.0 + 10 * h + 1.5 for h in range(hours)]
        assert df.index[0].tz_convert("Europe/Vienna").hour == 0  # the local day starts at midnight

    def test_a_chunk_boundary_in_the_middle_of_an_hour_loses_no_auction_hour(self):
        boundary = _utc("2024-07-01 22:30")
        first = _doc(
            _series(
                _period("2024-07-01T22:00Z", "2024-07-01T22:30Z", "PT15M", {1: 10.0, 2: 20.0}, "price.amount"),
                curve="A01",
                sequence=2,
            ),
            root="P",
        )
        second = _doc(
            _series(
                _period("2024-07-01T22:30Z", "2024-07-01T23:00Z", "PT15M", {1: 30.0, 2: 40.0}, "price.amount"),
                curve="A01",
                sequence=2,
            ),
            root="P",
        )
        sdac = _doc(
            _series(
                _period("2024-07-01T22:00Z", "2024-07-01T23:00Z", "PT60M", {1: 5.0}, "price.amount"),
                curve="A01",
                sequence=1,
            ),
            root="P",
        )

        def route(params):
            start = pd.to_datetime(params["periodStart"], format="%Y%m%d%H%M", utc=True)
            if params.get(POSITION) == "2":
                return first if start < boundary else second
            return sdac

        source = _source(route, country="AT", price_exaa=True)
        source._generate_chunks = lambda start, end, months=3: [(start, boundary), (boundary, end)]
        df = source.fetch(_utc("2024-07-01 22:00"), _utc("2024-07-01 23:00"))
        assert df["price_exaa"].tolist() == [25.0]


class TestExaaConfiguration:
    @pytest.mark.parametrize("country", ["AT", "DE_LU", "de_lu", "DE_AT_LU"])
    def test_available_for_austria_and_germany_luxembourg(self, country):
        assert EntsoeSource(country, KEY, type=["price"], price_exaa=True).price_exaa is True

    @pytest.mark.parametrize("country", ["PL", "CH", "FR", "CZ", "HU"])
    def test_other_zones_raise(self, country):
        with pytest.raises(ValueError, match="AT, DE_LU and DE_AT_LU only"):
            EntsoeSource(country, KEY, type=["price"], price_exaa=True)

    def test_needs_price_in_type(self):
        with pytest.raises(ValueError, match="include 'price'"):
            EntsoeSource("AT", KEY, type=["load"], price_exaa=True)

    @pytest.mark.parametrize("value", ["yes", 1, None])
    def test_must_be_a_bool(self, value):
        with pytest.raises(ValueError, match="True or False"):
            EntsoeSource("AT", KEY, type=["price"], price_exaa=value)

    def test_cache_key_changes_only_when_set(self):
        plain = EntsoeSource("AT", KEY, type=["price"]).get_cache_config()
        with_exaa = EntsoeSource("AT", KEY, type=["price"], price_exaa=True).get_cache_config()
        assert "price_exaa" not in plain  # the keys of existing caches stay valid
        assert with_exaa == {**plain, "price_exaa": True}
        assert plain == {
            "source_type": "entsoe",
            "area_code": AREAS["AT"],
            "types": ["price"],
            "prefer_15min": False,
            "schema_version": plain["schema_version"],
        }

    def test_survives_the_yaml_round_trip(self):
        source = EntsoeSource("AT", KEY, type=["price"], price_exaa=True, prefix="at")
        params = {name: getattr(source, name) for name in ("country_code", "api_key", "type", "price_exaa", "prefix")}
        assert EntsoeSource(**params).get_cache_config() == source.get_cache_config()


# --------------------------------------------------------------------------------------------------------------
# The default output does not move
# --------------------------------------------------------------------------------------------------------------


class TestDefaultOutputUnchanged:
    def test_default_request_and_frame_are_those_of_the_release_before(self):
        # A document that carries both auctions, as the API returns it for AT and DE_LU.
        document = _a44("2024-07-01T22:00Z", "2024-07-02T22:00Z", 24)
        source = _source(lambda params: document, country="AT")
        df = source.fetch(_utc("2024-07-01 22:00"), _utc("2024-07-02 22:00"))

        assert [c["params"] for c in source.session.calls] == [
            {**OLD_REQUEST, "periodStart": "202407012200", "periodEnd": "202407022200"}
        ]
        expected = pd.DataFrame(
            {"price": [float(h) for h in range(24)]},
            index=pd.date_range("2024-07-01 22:00", periods=24, freq="h", tz="UTC"),
        )
        pd.testing.assert_frame_equal(df, expected, check_exact=True, check_freq=False)
        assert list(df.columns) == ["price"] and df["price"].dtype == "float64" and df.index.tz is not None

    def test_a_source_with_price_exaa_off_makes_the_same_calls_as_one_that_never_heard_of_it(self):
        document = _a44("2024-07-01T22:00Z", "2024-07-02T22:00Z", 24)
        off = _source(lambda params: document, country="AT", price_exaa=False)
        off.fetch(_utc("2024-07-01 22:00"), _utc("2024-07-02 22:00"))
        assert len(off.session.calls) == 1 and POSITION not in off.session.calls[0]["params"]


# --------------------------------------------------------------------------------------------------------------
# Switzerland
# --------------------------------------------------------------------------------------------------------------


class TestSwissAuction:
    def test_price_for_ch_is_requested_with_the_swiss_domain_and_kept_as_price(self):
        document = _doc(
            _series(
                _period("2024-07-01T22:00Z", "2024-07-02T22:00Z", "PT60M", _hourly(24, 50.0), "price.amount"),
                curve="A01",
            ),
            root="Publication_MarketDocument",
        )
        source = _source(lambda params: document, country="CH")
        df = source.fetch(_utc("2024-07-01 22:00"), _utc("2024-07-02 22:00"))
        assert list(df.columns) == ["price"] and df["price"].tolist() == [50.0 + h for h in range(24)]
        params = source.session.calls[0]["params"]
        assert params["in_Domain"] == params["out_Domain"] == "10YCH-SWISSGRIDZ"
        assert POSITION not in params

    def test_ch_is_a_bidding_zone_and_prefixes_like_any_other(self):
        assert EntsoeSource("CH", KEY, type=["price"], prefix="ch").prefix == "ch"


# --------------------------------------------------------------------------------------------------------------
# Availability rules
# --------------------------------------------------------------------------------------------------------------


def _unknown_to_the_issue(info, predictor, zone, start="2024-01-10", end="2024-01-23"):
    index = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz=zone)
    found, _ = info.violations({(p, 1): [predictor] for p in range(24)}, "price", "1h", index, start, end)
    return {(v.predictor, v.period) for v in found}


class TestAvailability:
    def test_rule_of_the_auction_is_11_15_on_d_minus_1_in_vienna_as_a_snapshot(self):
        rule = source_rules([EntsoeSource("AT", KEY, type=["price"], price_exaa=True)])["price_exaa"]
        assert (rule.kind, rule.time, rule.days_before, rule.timezone, rule.vintage) == (
            "published", "11:15", 1, "Europe/Vienna", "snapshot"
        )  # fmt: skip
        assert "Art. 12.1.d" in rule.note
        assert (
            source_rules([EntsoeSource("DE_LU", KEY, type=["price"], price_exaa=True)])["price_exaa"].timezone
            == "Europe/Berlin"
        )
        assert default_rules()["price_exaa"].time == "11:15"

    def test_a_forecast_issued_at_11_45_sees_the_auction_of_the_next_day_and_one_at_10_30_does_not(self):
        info = InformationSet("11:45", "Europe/Vienna").with_source_rules(
            [EntsoeSource("AT", KEY, type=["price"], price_exaa=True)]
        )
        assert _unknown_to_the_issue(info, "price_exaa_d+1", "Europe/Vienna") == set()
        early = InformationSet("10:30", "Europe/Vienna").with_source_rules(
            [EntsoeSource("AT", KEY, type=["price"], price_exaa=True)]
        )
        assert {predictor for predictor, _ in _unknown_to_the_issue(early, "price_exaa_d+1", "Europe/Vienna")} == {
            "price_exaa_d+1"
        }
        # between the 10:15 gate and the legal bound it is not yet counted as out
        mid = InformationSet("11:00", "Europe/Vienna").with_source_rules(
            [EntsoeSource("AT", KEY, type=["price"], price_exaa=True)]
        )
        assert _unknown_to_the_issue(mid, "price_exaa_d+1", "Europe/Vienna") != set()

    def test_the_same_holds_through_both_dst_switches(self):
        sources = [EntsoeSource("DE_LU", KEY, type=["price"], price_exaa=True, prefix="de")]
        for start, end in (("2024-03-25", "2024-04-05"), ("2024-10-21", "2024-11-01")):
            late = InformationSet("11:45", "Europe/Berlin").with_source_rules(sources)
            assert _unknown_to_the_issue(late, "de_price_exaa_d+1", "Europe/Berlin", start, end) == set()
            early = InformationSet("10:30", "Europe/Berlin").with_source_rules(sources)
            assert len(_unknown_to_the_issue(early, "de_price_exaa_d+1", "Europe/Berlin", start, end)) == 24

    def test_the_auction_of_the_issue_day_itself_is_known_and_the_one_two_days_ahead_is_not(self):
        info = InformationSet("11:45", "Europe/Vienna").with_source_rules(
            [EntsoeSource("AT", KEY, type=["price"], price_exaa=True)]
        )
        assert _unknown_to_the_issue(info, "price_exaa_d+0", "Europe/Vienna") == set()
        assert len(_unknown_to_the_issue(info, "price_exaa_d+2", "Europe/Vienna")) == 24

    def test_swiss_price_is_published_by_12_00_zurich(self):
        rule = source_rules([EntsoeSource("CH", KEY, type=["price"])])["price"]
        assert (rule.kind, rule.time, rule.days_before, rule.timezone, rule.vintage) == (
            "published", "12:00", 1, "Europe/Zurich", "snapshot"
        )  # fmt: skip
        assert "unmeasured" in rule.note and "CH is not in SDAC" in rule.note
        sources = [EntsoeSource("CH", KEY, type=["price"])]
        at_noon = InformationSet("12:00", "Europe/Zurich").with_source_rules(sources)
        assert _unknown_to_the_issue(at_noon, "price_d+1", "Europe/Zurich") == set()
        at_11_45 = InformationSet("11:45", "Europe/Zurich").with_source_rules(sources)
        assert len(_unknown_to_the_issue(at_11_45, "price_d+1", "Europe/Zurich")) == 24

    def test_prefixed_swiss_price_gets_the_rule_and_other_zones_keep_the_13_00_one(self):
        rules = source_rules(
            [EntsoeSource("CH", KEY, type=["price"], prefix="ch"), EntsoeSource("AT", KEY, type=["price"], prefix="at")]
        )
        assert rules["ch_price"].time == "12:00" and "at_price" not in rules
        assert default_rules()["price"].time == "13:00"
        assert "price" not in source_rules([EntsoeSource("PL", KEY, type=["price"])])

    def test_load_rules_are_untouched_when_prices_come_with_them(self):
        rules = source_rules([EntsoeSource("AT", KEY, type=["load", "price"], price_exaa=True, prefix="at")])
        assert {"at_load_forecast_daily_min", "at_load_forecast_daily_max", "at_price_exaa"} <= set(rules)


class TestPlainInformationSet:
    """Without `Workflow` (`ModelPipeline(issue_time=...)`, a bare `InformationSet`) only the built-in rules apply. These are
    the same checks as above without `with_source_rules`; `Workflow` adds the source rules through exactly that call."""

    @pytest.mark.parametrize("zone", ["Europe/Vienna", "Europe/Berlin"])
    def test_price_exaa_resolves_to_the_11_15_rule(self, zone):
        rule = InformationSet().resolve("price_exaa_d+1").rule
        assert (rule.time, rule.days_before, rule.vintage) == ("11:15", 1, "snapshot")
        assert _unknown_to_the_issue(InformationSet("11:45", zone), "price_exaa_d+1", zone) == set()
        assert len(_unknown_to_the_issue(InformationSet("10:30", zone), "price_exaa_d+1", zone)) == 24

    @pytest.mark.parametrize("name", ["de_price_exaa_d+1", "at_price_exaa_d+1", "de_price_exaa_h05_d+1"])
    def test_prefixed_price_exaa_resolves_to_the_11_15_rule(self, name):
        rule = InformationSet().resolve(name).rule
        assert (rule.time, rule.days_before, rule.timezone, rule.vintage) == ("11:15", 1, "Europe/Vienna", "snapshot")

    def test_the_serve_time_pattern_uses_prefixed_exaa_on_none_of_the_days_at_10_30(self):
        # The docs' pattern adds the columns only where they are published: horizon 1 at 10:30 and horizon 2 at 11:45 are not.
        index = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="Europe/Berlin")
        for time, horizon in (("10:30", 1), ("10:30", 2), ("11:45", 2)):
            info = InformationSet(time, "Europe/Berlin")
            predictors = {(p, horizon): [f"de_price_exaa_h{h:02d}_d+{horizon}" for h in range(24)] for p in range(24)}
            found, unknown = info.violations(predictors, "price", "1h", index, "2024-01-10", "2024-01-14")
            assert not unknown and {v.predictor for v in found} == set(predictors[(0, horizon)]), (time, horizon)
        info = InformationSet("11:45", "Europe/Berlin")
        predictors = {(p, 1): [f"de_price_exaa_h{h:02d}_d+1" for h in range(24)] for p in range(24)}
        assert info.violations(predictors, "price", "1h", index, "2024-01-10", "2024-01-14")[0] == []

    def test_a_prefixed_swiss_price_has_no_built_in_rule(self):
        assert InformationSet().resolve("ch_price_d+1").rule is None

    def test_price_exaa_profile_columns_follow_the_same_rule(self):
        assert InformationSet().resolve("price_exaa_h05_d+1").rule.time == "11:15"

    def test_plain_ch_price_gets_the_13_00_default_never_earlier_than_12_00_zurich(self):
        rule = InformationSet().resolve("price_d+1").rule
        assert (rule.time, rule.timezone) == (
            "13:00",
            "Europe/Brussels",
        )  # = 13:00 Zurich, an hour later than the source rule
        for time in ("11:45", "12:00"):
            info = InformationSet(time, "Europe/Zurich")
            assert len(_unknown_to_the_issue(info, "price_d+1", "Europe/Zurich")) == 24
        assert _unknown_to_the_issue(InformationSet("13:00", "Europe/Zurich"), "price_d+1", "Europe/Zurich") == set()


# --------------------------------------------------------------------------------------------------------------
# Real API (skipped without ENTSOE_API_KEY)
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.integration
class TestEarlyAuctionsIntegration:
    @pytest.mark.parametrize("country", ["AT", "DE_LU"])
    def test_exaa_is_published_beside_sdac_and_differs_from_it(self, entsoe_api_key, country):
        source = EntsoeSource(country, entsoe_api_key, type=["price"], price_exaa=True)
        df = source.fetch(_utc("2024-06-01 22:00"), _utc("2024-06-03 22:00"))
        assert list(df.columns) == ["price", "price_exaa"] and len(df) == 48
        assert df.notna().all().all() and (df["price"] != df["price_exaa"]).any()

    def test_exaa_across_both_dst_switches_on_the_hourly_grid(self, entsoe_api_key):
        source = EntsoeSource("AT", entsoe_api_key, type=["price"], price_exaa=True)
        for start, end, rows in (
            ("2024-03-30 23:00", "2024-03-31 22:00", 23),
            ("2024-10-26 22:00", "2024-10-27 23:00", 25),
        ):
            df = source.fetch(_utc(start), _utc(end))
            assert len(df) == rows and df.notna().all().all()

    def test_exaa_of_germany_before_the_split_of_the_zone(self, entsoe_api_key):
        # Unverified until run with a key: DE_AT_LU is the German zone before 2018-10-01.
        source = EntsoeSource("DE_AT_LU", entsoe_api_key, type=["price"], price_exaa=True)
        df = source.fetch(_utc("2018-06-01 22:00"), _utc("2018-06-03 22:00"))
        assert list(df.columns) == ["price", "price_exaa"] and len(df) == 48 and df.notna().all().all()

    def test_swiss_auction_price(self, entsoe_api_key):
        df = EntsoeSource("CH", entsoe_api_key, type=["price"]).fetch(
            _utc("2024-06-01 22:00"), _utc("2024-06-03 22:00")
        )
        assert list(df.columns) == ["price"] and len(df) == 48 and df["price"].notna().all()
