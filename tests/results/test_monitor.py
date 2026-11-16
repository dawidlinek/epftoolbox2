"""Monitoring a live directory (step 8h, `epftoolbox2.results.monitor`, `ModelPipeline.monitor`).

The fixtures are hand-made live directories, written through 8c's live-store writers (`LiveStore.begin` / `LiveRevision.open` /
`commit`), never through `forecast()`: ten issue days of Europe/Berlin, 2026-10-18 to 2026-10-27, a switch day among them (the
autumn one, Sunday 10-25), hourly, horizons 1 and 2, three members: `lear`, `naive` (the benchmark) and `best`, a
`FallbackEnsemble` of the two. What happened is planted:

- day 2 has two revisions before the deadline, the first delivered; day 3's second revision is committed after the deadline;
  day 4's third was never committed; the first revision of day 4 has no `naive` forecast of hour 5;
- day 5 is a fallback day (`best` served it from `naive`), day 6 has no call at all, day 7 a late call that committed nothing;
- the data holds the price, `price` known by the rule `published 13:00 on the day before`, and the clock is 14:00 on the last
  issue day, so the keys of horizon 2 of the last two days are not scored yet.

The oracle (`Oracle`) knows nothing of the library: it recomputes every metric with pandas from the records it wrote and the
data, by the definitions of the module's docstring.
"""

import datetime
import json
import math

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.clock import FakeClock, use_clock
from epftoolbox2.ensembles import FallbackEnsemble
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import NaiveModel, OLSModel
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.live import LiveStore
from epftoolbox2.results.monitor import MonitorReport, load_alerts
from epftoolbox2.results.store import LiveDirectoryError, with_status

pytestmark = pytest.mark.usefixtures("no_network")

ZONE = "Europe/Berlin"
DAYS = [(datetime.date(2026, 10, 18) + datetime.timedelta(days=i)).isoformat() for i in range(10)]
HOURS = tuple(range(24))
HORIZONS = (1, 2)
NOW = pd.Timestamp("2026-10-27T14:00", tz=ZONE)
DEADLINE = "11:55"
FALLBACK_DAY = DAYS[5]
MISSING_DAY = DAYS[6]
LATE_DAY = DAYS[7]


def local(day: str, time: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day}T{time}", tz=ZONE).tz_convert("UTC")


def make_data() -> pd.DataFrame:
    """The price, hourly, in Berlin time over the issue days and the days after them: a smooth daily shape plus noise."""
    index = pd.date_range("2026-10-10", "2026-11-03", freq="h", tz=ZONE, inclusive="left")
    rng = np.random.default_rng(3)
    shape = 20 * np.sin((index.hour.to_numpy() - 6) * np.pi / 12)
    trend = 2.0 * (index.normalize() - index[0].normalize()).days.to_numpy()
    return pd.DataFrame({"price": 60 + shape + trend + rng.normal(0, 4, len(index))}, index=index)


DATA = make_data()


def wall_actuals() -> dict:
    """(local date, hour) -> the price; a repeated autumn hour has two prices and so none, as a run's actual (no market cleared
    one of them)."""
    held: dict = {}
    for stamp, price in DATA["price"].items():
        held.setdefault((stamp.date().isoformat(), stamp.hour), []).append(float(price))
    return {key: values[0] for key, values in held.items() if len(values) == 1}


ACTUALS = wall_actuals()


def actual_of(target: str, hour: int) -> float:
    return ACTUALS.get((target, hour), math.nan)


def known(target: str, now: pd.Timestamp) -> bool:
    """The rule `published 13:00 on the day before`: the price of a day is out at 13:00 Berlin on the day before it."""
    day = datetime.date.fromisoformat(target) - datetime.timedelta(days=1)
    return local(day.isoformat(), "13:00") <= now


def plus(day: str, days: int) -> str:
    return (datetime.date.fromisoformat(day) + datetime.timedelta(days=days)).isoformat()


# --- what was forecast ---------------------------------------------------------------------------------------------------------


def planted(member: str, day: str, revision: int, horizon: int, hour: int) -> float:
    """The forecast of a member: the actual (or a smooth price where there is none) plus a planted error of its own.

    `lear`'s error has a level (its day's mean) and a shape (the sign pattern of the hours inside the day), `naive`'s is larger. The
    last four issue days of `lear` get worse (a skill alert's planted condition: lear's error over naive's is high on the last
    three, low on the one before). A later revision of a day moves the forecast by a constant."""
    target = plus(day, horizon)
    base = actual_of(target, hour)
    if math.isnan(base):
        base = 60.0 + hour
    d = DAYS.index(day) if day in DAYS else 0
    sign = 1.0 if hour % 2 == 0 else -1.0
    if member == "naive":
        error = 12.0 + 3.0 * ((d + horizon) % 3) + 4.0 * sign + 0.5 * hour / 24
    else:
        level = 2.0 + 0.5 * d - 1.5 * horizon
        shape = 3.0 + 0.1 * d
        error = level + shape * sign + 0.05 * hour
        if d >= len(DAYS) - 3:
            error += 10.0
        elif d == len(DAYS) - 4:
            error -= 5.0
    return base + error + 0.75 * (revision - 1)


# revision -> (committed, issued) per day; None: never committed. Local Berlin times.
REVISIONS = {
    DAYS[0]: [("10:05", True)],
    DAYS[1]: [("10:15", True)],
    DAYS[2]: [("10:20", True), ("11:30", True)],
    DAYS[3]: [("10:25", True), ("12:30", True)],  # the second after the 11:55 deadline
    DAYS[4]: [("10:30", True), ("11:00", True), ("11:20", False)],  # the third crashed
    DAYS[5]: [("10:35", True)],
    DAYS[8]: [("10:40", True)],
    DAYS[9]: [("10:45", True)],
}
DELIVERIES = [(DAYS[2], 1, "10:50"), (DAYS[3], 1, "11:10"), (DAYS[9], 1, "11:00")]
# (member, day, revision, hour) forecasts a revision lacks, at every horizon: the first revision of day 4 has no naive forecast of
# hour 5 and the second of day 2 no lear forecast of hour 7, so each of the two has keys the other lacks (skill is on the common ones)
GAPS = {("naive", DAYS[4], 1, 5), ("lear", DAYS[2], 2, 7)}


def record(member: str, day: str, revision: int, horizon: int, hour: int) -> dict:
    run = datetime.date.fromisoformat(day)
    out = {
        "run_date": day,
        "target_date": plus(day, horizon),
        "run_weekday": run.weekday(),
        "hour": hour,
        "horizon": horizon,
        "day_in_test": DAYS.index(day) if day in DAYS else 0,
        "prediction": planted(member, day, revision, horizon, hour),
        "actual": None,  # a live record has no actual at issue
        "coefficients": [],
        "freq": "1h",
    }
    if member == "best":
        served_by_naive = day == FALLBACK_DAY
        out["prediction"] = planted("naive" if served_by_naive else "lear", day, revision, horizon, hour)
        out["rung"] = 1 if served_by_naive else 0
        out["member"] = "naive" if served_by_naive else "lear"
    return with_status(out, "ok", list_hash="abc", refit_anchor=day)


def write_live(directory, *, receipts=True, save_format="jsonl") -> None:
    store = LiveStore(directory)
    store.init(mode="live", issue_timezone=ZONE, deadline=DEADLINE)
    with store.hold():
        for day in DAYS:
            if day == LATE_DAY:
                store.record_call(day, "late", 30, call="late", finished=local(day, "12:10"))
            for number, (at, commits) in enumerate(REVISIONS.get(day, []), 1):
                rev = store.begin(day, issued_at=local(day, "09:30"), snapshot=f"s-{day}-{number}")
                assert rev.revision == number
                for member in ("lear", "naive", "best"):
                    out = rev.open(member, "jsonl" if member == "best" else save_format)  # an ensemble is JSONL
                    for hour in HOURS:
                        for horizon in HORIZONS:
                            if (member, day, number, hour) in GAPS:
                                continue
                            out.save({**record(member, day, number, horizon, hour), "revision": 1})
                if commits:
                    on_time = local(day, at) <= local(
                        day, DEADLINE
                    )  # a late commit is the writer's own, with no deadline
                    rev.commit(finished=local(day, at), deadline=local(day, DEADLINE) if on_time else None)
                else:
                    rev.abandon()
        if receipts:
            for day, revision, at in DELIVERIES:
                receipt = store.save_dir / f"receipt-{day}.json"
                receipt.write_text("{}")
                store.mark_delivered(day, revision, receipt, delivered_at=local(day, at))


def pipeline(*, with_ensemble=True) -> ModelPipeline:
    info = InformationSet("11:45", ZONE, rules={"price": Availability.published("13:00", 1, ZONE)})
    pipe = ModelPipeline(information_set=info)
    pipe.add_model(OLSModel(["price_d-1"], name="lear"))
    pipe.add_model(NaiveModel(["price_d-1"], name="naive"))
    if with_ensemble:
        pipe.add_ensemble(FallbackEnsemble(["lear", "naive"], name="best"))
    return pipe


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    directory = tmp_path_factory.mktemp("live") / "de_price"
    write_live(directory)
    return directory


def run_monitor(directory, *, now=NOW, **kwargs) -> MonitorReport:
    with use_clock(FakeClock(now)):
        return pipeline().monitor(DATA, directory, benchmarks=kwargs.pop("benchmarks", ["naive"]), **kwargs)


# --- the oracle -------------------------------------------------------------------------------------------------------------


class Oracle:
    """Every metric by pandas from what the fixture planted, by the definitions of the module's docstring."""

    def __init__(self, policy="latest", now=NOW):
        self.policy = policy
        self.now = now

    def committed(self, day):
        """Number -> committed instant of the revisions of `day` committed by the deadline."""
        limit = local(day, DEADLINE)
        return {
            number: local(day, at)
            for number, (at, commits) in enumerate(REVISIONS.get(day, []), 1)
            if commits and local(day, at) <= limit
        }

    def revision(self, day, policy=None):
        seen = self.committed(day)
        policy = policy or self.policy
        if not seen:
            return None
        if policy == "first":
            return min(seen)
        if policy == "delivered":
            heads = [r for d, r, at in DELIVERIES if d == day and r in seen and local(day, at) <= local(day, DEADLINE)]
            return max(heads) if heads else None
        return max(seen)

    def due(self):
        today = self.now.tz_convert(ZONE).date()
        last = today if local(today.isoformat(), DEADLINE) <= self.now else today - datetime.timedelta(days=1)
        return [d for d in DAYS if d <= last.isoformat()]

    def window(self, n):
        due = self.due()
        return [d for d in due if d >= plus(due[-1], -(n - 1))]

    def keys(self, member, days, policy=None, scored_only=True):
        rows = []
        for day in days:
            revision = self.revision(day, policy)
            if revision is None:
                continue
            for horizon in HORIZONS:
                for hour in HOURS:
                    if (member, day, revision, hour) in GAPS:
                        continue
                    target = plus(day, horizon)
                    actual = actual_of(target, hour)
                    prediction = planted(
                        "naive"
                        if member == "best" and day == FALLBACK_DAY
                        else ("lear" if member == "best" else member),
                        day,
                        revision,
                        horizon,
                        hour,
                    )
                    rows.append(
                        dict(
                            day=day,
                            horizon=horizon,
                            hour=hour,
                            prediction=prediction,
                            actual=actual,
                            known=known(target, self.now),
                            revision=revision,
                        )
                    )
        frame = pd.DataFrame(rows)
        if scored_only and len(frame):
            frame = frame[frame["known"] & frame["actual"].notna()].copy()
        if len(frame):
            frame["error"] = frame["prediction"] - frame["actual"]
        return frame

    def metrics(self, member, n, policy=None):
        frame = self.keys(member, self.window(n), policy)
        by_day = frame.groupby(["day", "horizon"])["error"]
        shape_error = frame["error"] - by_day.transform("mean")
        return dict(
            forecasts=len(frame),
            mae=frame["error"].abs().mean(),
            rmse=math.sqrt((frame["error"] ** 2).mean()),
            rmse_daily=by_day.apply(lambda e: math.sqrt((e**2).mean())).mean(),
            bias=frame["error"].mean(),
            level=by_day.mean().abs().mean(),
            shape=shape_error.abs().mean(),
        )

    def skill(self, member, bench, n, policy=None):
        a = self.keys(member, self.window(n), policy).set_index(["day", "horizon", "hour"])
        b = self.keys(bench, self.window(n), policy).set_index(["day", "horizon", "hour"])
        both = a.index.intersection(b.index)
        a, b = a.loc[both], b.loc[both]
        day_a = a["error"].abs().groupby(level=["day", "horizon"]).mean()
        day_b = b["error"].abs().groupby(level=["day", "horizon"]).mean()
        return dict(
            skill=a["error"].abs().mean() / b["error"].abs().mean(),
            common=len(both),
            days=len(day_a),
            wins=int((day_a < day_b).sum()),
        )


def row(report, window, member):
    table = report.tables()
    chosen = table[(table["window"] == window) & (table["member"] == member)]
    assert len(chosen) == 1
    return chosen.iloc[0]


# --- every metric against the oracle ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["latest", "first", "delivered"])
def test_every_metric_matches_the_oracle_under_each_policy(live, policy):
    report = run_monitor(live, policy=policy, windows=(3, 7, 30))
    oracle = Oracle(policy)
    for window in (3, 7, 30):
        for member in ("lear", "naive", "best"):
            frame = oracle.keys(member, oracle.window(window))
            got = row(report, window, member)
            if not len(frame):
                assert got["forecasts"] == 0
                continue
            expected = oracle.metrics(member, window)
            for name, value in expected.items():
                assert got[name] == pytest.approx(value, rel=1e-12, abs=1e-12), (policy, window, member, name)
            if member != "naive":
                skill = oracle.skill(member, "naive", window)
                assert got["skill_naive"] == pytest.approx(skill["skill"], rel=1e-12)
                assert (got["common_naive"], got["days_naive"], got["wins_naive"]) == (
                    skill["common"],
                    skill["days"],
                    skill["wins"],
                )


def test_the_revision_a_key_is_read_from_follows_the_policy(live):
    """Day 2 has r1 (delivered) and r2, day 3 r1 (delivered) and a late r2, day 4 r1 and r2: the policies differ by day."""
    expected = {
        "latest": {DAYS[2]: 2, DAYS[3]: 1, DAYS[4]: 2},
        "first": {DAYS[2]: 1, DAYS[3]: 1, DAYS[4]: 1},
        "delivered": {DAYS[2]: 1, DAYS[3]: 1},
    }
    for policy, revisions in expected.items():
        oracle = Oracle(policy)
        assert {d: oracle.revision(d) for d in DAYS if oracle.revision(d) is not None and d in revisions} == revisions
        report = run_monitor(live, policy=policy, windows=(10,))
        quality = report.quality().set_index("issue")["revision"].to_dict()
        for day, revision in revisions.items():
            assert quality[day] == revision
        if policy == "delivered":
            assert set(quality) == {DAYS[2], DAYS[3], DAYS[9]}
    # the predictions are those of the revision: r2 of day 4 adds 0.75 to every forecast
    daily = run_monitor(live, policy="latest", windows=(10,)).daily()
    first = run_monitor(live, policy="first", windows=(10,)).daily()
    key = (daily["member"] == "lear") & (daily["run_date"] == DAYS[4]) & (daily["horizon"] == 1)
    other = (first["member"] == "lear") & (first["run_date"] == DAYS[4]) & (first["horizon"] == 1)
    assert daily[key]["bias"].iloc[0] - first[other]["bias"].iloc[0] == pytest.approx(0.75)


def test_the_delivered_share_and_participation_are_reported_whatever_the_policy(live):
    for column, share in (("delivered", 3 / 10), ("participation", 8 / 10)):
        shares = {
            p: run_monitor(live, policy=p, windows=(10,)).tables()[column].iloc[0]
            for p in ("latest", "first", "delivered")
        }
        assert set(shares.values()) == {share}, column


def test_participation_counts_the_days_with_a_revision_committed_by_the_deadline(live):
    table = run_monitor(live, windows=(3, 10, 30)).tables()
    for window in (3, 10, 30):
        chosen = table[table["window"] == window].iloc[0]
        due = Oracle().window(window)
        committed = [d for d in due if Oracle().committed(d)]
        assert chosen["issue_days"] == len(due) == min(window, 10)
        assert chosen["committed_days"] == len(committed)
        assert chosen["participation"] == pytest.approx(len(committed) / len(due))
    # 10 days: the missing day 6, the late day 7 and day 8... have no committed revision before the deadline by the oracle's rule
    assert Oracle().window(10) == DAYS
    assert [d for d in DAYS if not Oracle().committed(d)] == [MISSING_DAY, LATE_DAY]


def test_a_member_s_own_days_are_counted(live):
    table = run_monitor(live, windows=(10,)).tables().set_index("member")
    assert table.loc["lear", "member_days"] == 8 and table.loc["best", "member_days"] == 8


def test_forecasts_not_yet_scored_are_counted_not_scored(live):
    table = run_monitor(live, windows=(10,)).tables().set_index("member")
    oracle = Oracle()
    every = oracle.keys("lear", DAYS, scored_only=False)
    scored = oracle.keys("lear", DAYS)
    assert table.loc["lear", "forecasts"] == len(scored)
    # at 14:00 on the last issue day its horizon 2 (the price of the 29th, out at 13:00 on the 28th) is not known yet: 24 keys;
    # and the repeated autumn hour of the 25th has no actual: 1 key
    unknown = every[~every["known"]]
    assert list(zip(unknown["day"], unknown["horizon"])) == [(DAYS[9], 2)] * 24
    assert (every["known"] & every["actual"].isna()).sum() == 1
    assert table.loc["lear", "unscored"] == len(every) - len(scored) == 25


def test_the_clock_decides_what_is_scored(live):
    """Before 13:00 on the last issue day its horizon 1 isn't published yet, and the day is not even due before its deadline."""
    early = run_monitor(live, windows=(10,), now=pd.Timestamp("2026-10-27T11:00", tz=ZONE))
    assert early.through == DAYS[8]
    oracle = Oracle(now=pd.Timestamp("2026-10-27T11:00", tz=ZONE))
    assert row(early, 10, "lear")["forecasts"] == len(oracle.keys("lear", oracle.window(10)))
    between = run_monitor(live, windows=(10,), now=pd.Timestamp("2026-10-27T12:30", tz=ZONE))
    assert between.through == DAYS[9]  # past the deadline, so due; the day-ahead price isn't out until 13:00
    oracle = Oracle(now=pd.Timestamp("2026-10-27T12:30", tz=ZONE))
    assert row(between, 10, "lear")["forecasts"] == len(oracle.keys("lear", oracle.window(10)))
    assert row(between, 10, "lear")["forecasts"] < row(run_monitor(live, windows=(10,)), 10, "lear")["forecasts"]


# --- the decomposition and the daily numbers --------------------------------------------------------------------------------


def test_level_and_shape_decompose_the_error_as_defined(live):
    daily = run_monitor(live, windows=(10,)).daily()
    table = run_monitor(live, windows=(10,)).tables()
    for member in ("lear", "naive"):
        chosen = table[table["member"] == member].iloc[0]
        # per day mean|e| <= |mean e| + mean|e - mean e|; weighing every day alike this gives MAE <= level + shape up to the day sizes
        frame = Oracle().keys(member, DAYS)
        by_day = frame.groupby(["day", "horizon"])
        weighted_level = (by_day["error"].mean().abs() * by_day.size()).sum() / len(frame)
        assert chosen["mae"] <= weighted_level + chosen["shape"] + 1e-12
        assert chosen["level"] == pytest.approx(by_day["error"].mean().abs().mean())
    assert (daily["rmse"] >= daily["bias"].abs() - 1e-12).all()


def test_a_constant_error_is_all_level_and_a_zero_mean_error_all_shape(tmp_path):
    directory = tmp_path / "live"
    store = LiveStore(directory)
    store.init(mode="live", issue_timezone=ZONE, deadline=DEADLINE)
    days = DAYS[:4]
    with store.hold():
        for day in days:
            rev = store.begin(day, issued_at=local(day, "09:30"), snapshot="s")
            for member, kind in (("flat", "level"), ("swing", "shape"), ("naive", "naive")):
                out = rev.open(member)
                for hour in HOURS:
                    actual = actual_of(plus(day, 1), hour)
                    if kind == "level":
                        error = 5.0 + days.index(day)
                    elif kind == "shape":
                        error = 4.0 if hour % 2 == 0 else -4.0
                    else:
                        error = 20.0
                    if math.isnan(actual):
                        continue
                    out.save({**record("lear", day, 1, 1, hour), "prediction": actual + error, "revision": 1})
            rev.commit(finished=local(day, "10:00"), deadline=local(day, DEADLINE))
    pipe = ModelPipeline(
        information_set=InformationSet("11:45", ZONE, rules={"price": Availability.published("13:00", 1, ZONE)})
    )
    for name in ("flat", "swing"):
        pipe.add_model(OLSModel(["price_d-1"], name=name))
    pipe.add_model(NaiveModel(["price_d-1"], name="naive"))
    with use_clock(FakeClock(pd.Timestamp("2026-10-23T14:00", tz=ZONE))):
        table = pipe.monitor(DATA, directory, benchmarks=["naive"], windows=(4,)).tables().set_index("member")
    assert table.loc["flat", "level"] == pytest.approx(table.loc["flat", "mae"]) and table.loc[
        "flat", "shape"
    ] == pytest.approx(0.0)
    assert table.loc["flat", "bias"] == pytest.approx(table.loc["flat", "mae"])
    assert table.loc["swing", "level"] == pytest.approx(0.0, abs=1e-12) and table.loc[
        "swing", "shape"
    ] == pytest.approx(4.0)
    assert table.loc["swing", "bias"] == pytest.approx(0.0, abs=1e-12)
    assert table.loc["swing", "rmse"] == pytest.approx(4.0) and table.loc["swing", "rmse_daily"] == pytest.approx(4.0)


def test_daily_rmse_is_the_mean_of_the_days_rmse_over_the_switch_day(live):
    """The autumn switch day (10-25, 25 prices) is a target of issue day 10-23 at horizon 2 (issue day 10-24 has no call): its
    curve has 24 periods and the repeated hour has no actual, so 23 forecasts are scored, and the day's RMSE is over 23."""
    table = run_monitor(live, windows=(10,)).tables().set_index("member")
    oracle = Oracle().metrics("lear", 10)
    assert table.loc["lear", "rmse_daily"] == pytest.approx(oracle["rmse_daily"], rel=1e-12)
    assert table.loc["lear", "rmse"] != pytest.approx(table.loc["lear", "rmse_daily"], rel=1e-9)  # days differ in size
    daily = run_monitor(live, windows=(10,)).daily()
    switch = daily[(daily["member"] == "lear") & (daily["target_date"] == "2026-10-25")]
    assert len(switch) == 1 and switch["horizon"].iloc[0] == 2 and switch["forecasts"].iloc[0] == 23
    frame = Oracle().keys("lear", [DAYS[5]])
    day = frame[(frame["horizon"] == 2)]
    assert len(day) == 23 and switch["rmse"].iloc[0] == pytest.approx(math.sqrt((day["error"] ** 2).mean()), rel=1e-12)
    assert (daily[(daily["member"] == "lear")]["forecasts"].isin([23, 24])).all()


def test_a_spring_day_has_23_hours_in_the_daily_rmse(tmp_path):
    days = ["2026-03-27", "2026-03-28", "2026-03-29"]
    index = pd.date_range("2026-03-20", "2026-04-03", freq="h", tz=ZONE, inclusive="left")
    rng = np.random.default_rng(5)
    data = pd.DataFrame({"price": 50 + rng.normal(0, 5, len(index))}, index=index)
    held = {(s.date().isoformat(), s.hour): float(v) for s, v in data["price"].items()}
    directory = tmp_path / "live"
    store = LiveStore(directory)
    store.init(mode="live", issue_timezone=ZONE, deadline=DEADLINE)
    errors = {}
    with store.hold():
        for member, scale in (("a", 1.0), ("naive", 3.0)):
            pass
        for day in days:
            rev = store.begin(day, issued_at=local(day, "09:30"), snapshot="s")
            for member, scale in (("a", 1.0), ("naive", 3.0)):
                out = rev.open(member)
                for hour in HOURS:
                    target = plus(day, 1)
                    if (target, hour) not in held or (day, hour) not in held:  # the hour the clock skips: not planned
                        continue
                    error = scale * (1.0 + 0.1 * hour)
                    errors.setdefault((member, day), []).append(error)
                    out.save(
                        {**record("lear", day, 1, 1, hour), "prediction": held[(target, hour)] + error, "revision": 1}
                    )
            rev.commit(finished=local(day, "10:00"), deadline=local(day, DEADLINE))
    pipe = ModelPipeline(
        information_set=InformationSet("11:45", ZONE, rules={"price": Availability.published("13:00", 1, ZONE)})
    )
    pipe.add_model(OLSModel(["price_d-1"], name="a")).add_model(NaiveModel(["price_d-1"], name="naive"))
    with use_clock(FakeClock(pd.Timestamp("2026-03-30T14:00", tz=ZONE))):
        report = pipe.monitor(data, directory, benchmarks=["naive"], windows=(4,))
    daily = report.daily()
    counts = daily[daily["member"] == "a"].set_index("target_date")["forecasts"].to_dict()
    assert counts == {pd.Timestamp("2026-03-28"): 24, pd.Timestamp("2026-03-29"): 23, pd.Timestamp("2026-03-30"): 23}
    # issue day 03-29 itself has no 02:00 either, and 03-28's target (03-29) lacks it: 23 each
    expected = np.mean([math.sqrt(np.mean(np.square(errors[("a", day)]))) for day in days])
    assert report.tables().set_index("member").loc["a", "rmse_daily"] == pytest.approx(expected, rel=1e-12)


# --- skill, wins and common keys -------------------------------------------------------------------------------------------------


def test_skill_is_on_the_keys_both_members_hold_with_counts(live):
    """Each member lacks keys the other has: the first revision of day 4 has no naive forecast of hour 5, the second of day 2 no
    lear forecast of hour 7 (24 hours x 2 horizons, so 2 keys each). The skill of lear is on the keys the two share, whichever of
    them lacks keys, and each member's own MAE is on all of its keys."""
    for policy in ("first", "latest"):
        report = run_monitor(live, policy=policy, windows=(10,))
        got = row(report, 10, "lear")
        oracle = Oracle(policy)
        own = oracle.keys("lear", DAYS)
        other = oracle.keys("naive", DAYS)
        assert got["forecasts"] == len(own) and row(report, 10, "naive")["forecasts"] == len(other)
        assert got["common_naive"] == len(own.merge(other, on=["day", "horizon", "hour"]))
        # "first": the naive lacks 2 keys of lear's (the common ones are the naive's); "latest": lear lacks 2 the naive has
        if policy == "first":
            assert got["common_naive"] == len(other) == len(own) - 2
        else:
            assert got["common_naive"] == len(own) == len(other) - 2
        skill = oracle.skill("lear", "naive", 10)
        assert got["skill_naive"] == pytest.approx(skill["skill"], rel=1e-12)
        # not the ratio of the two MAEs over their own keys
        assert got["skill_naive"] != pytest.approx(got["mae"] / row(report, 10, "naive")["mae"], rel=1e-9)
        assert math.isnan(row(report, 10, "naive")["skill_naive"])  # a benchmark has none against itself


def test_the_tables_lead_with_skill(live):
    columns = list(run_monitor(live, windows=(7,)).tables().columns)
    assert columns[:2] == ["window", "member"] and columns[2].startswith("skill_")
    assert columns.index("skill_naive") < columns.index("mae")


def test_by_window_cuts_the_tables(live):
    report = run_monitor(live, windows=(3, 7))
    by = report.by_window()
    assert set(by) == {3, 7} and "window" not in by[3]
    assert by[7].equals(report.tables().query("window == 7").drop(columns="window").reset_index(drop=True))


# --- the E3 example: raw error up, skill up ----------------------------------------------------------------------------------------


def test_rmse_can_rise_while_skill_improves(tmp_path):
    """Usage report E3: DE's live RMSE rose 41 % from August to September while skill improved. Fourteen issue days; the last
    seven are volatile: both members' errors are larger, the benchmark's much more."""
    start = datetime.date(2026, 10, 10)
    days = [(start + datetime.timedelta(days=i)).isoformat() for i in range(14)]
    directory = tmp_path / "live"
    store = LiveStore(directory)
    store.init(mode="live", issue_timezone=ZONE, deadline=DEADLINE)

    def scale(member, day):
        calm = days.index(day) < 7
        return {("model", True): 2.0, ("naive", True): 5.0, ("model", False): 8.0, ("naive", False): 30.0}[
            (member, calm)
        ]

    with store.hold():
        for day in days:
            rev = store.begin(day, issued_at=local(day, "09:30"), snapshot="s")
            for member in ("model", "naive"):
                out = rev.open(member)
                for hour in HOURS:
                    held = DATA["price"].loc[pd.Timestamp(f"{plus(day, 1)}T{hour:02d}:00", tz=ZONE)]
                    sign = 1.0 if hour % 2 == 0 else -1.0
                    out.save(
                        {
                            **record("lear", day, 1, 1, hour),
                            "prediction": float(held) + sign * scale(member, day),
                            "revision": 1,
                        }
                    )
            rev.commit(finished=local(day, "10:00"), deadline=local(day, DEADLINE))
    pipe = ModelPipeline(
        information_set=InformationSet("11:45", ZONE, rules={"price": Availability.published("13:00", 1, ZONE)})
    )
    pipe.add_model(OLSModel(["price_d-1"], name="model")).add_model(NaiveModel(["price_d-1"], name="naive"))

    def report_at(last_day, windows):
        with use_clock(FakeClock(pd.Timestamp(f"{last_day}T14:00", tz=ZONE))):
            return pipe.monitor(DATA, directory, benchmarks=["naive"], windows=windows)

    before = row(report_at(days[6], (7,)), 7, "model")  # after the calm week
    after = row(report_at(days[13], (7, 14)), 7, "model")  # after the volatile one
    whole = row(report_at(days[13], (7, 14)), 14, "model")
    assert after["rmse"] == pytest.approx(8.0) and before["rmse"] == pytest.approx(2.0)  # raw error up fourfold
    assert after["rmse_daily"] > before["rmse_daily"] * 1.41
    assert after["skill_naive"] == pytest.approx(8.0 / 30.0) and before["skill_naive"] == pytest.approx(2.0 / 5.0)
    assert after["skill_naive"] < before["skill_naive"]  # and skill better (a lower ratio)
    assert (after["wins_naive"], after["days_naive"]) == (7, 7)
    assert whole["skill_naive"] == pytest.approx((7 * 2.0 + 7 * 8.0) / (7 * 5.0 + 7 * 30.0), rel=1e-9)


# --- fallbacks, rungs, quality ----------------------------------------------------------------------------------------------------


def test_the_fallback_day_and_its_score(live):
    table = run_monitor(live, windows=(10,)).tables().set_index("member")
    oracle = Oracle()
    frame = oracle.keys("best", [FALLBACK_DAY])
    assert table.loc["best", "fallback_days"] == len(HORIZONS)  # (run date, horizon) days
    assert table.loc["best", "fallback_mae"] == pytest.approx(frame["error"].abs().mean(), rel=1e-12)
    assert table.loc["lear", "fallback_days"] == 0 and math.isnan(table.loc["lear", "fallback_mae"])


def test_rung_counts_per_policy(live):
    rungs = run_monitor(live, windows=(10,)).rungs()
    oracle = Oracle()
    assert set(rungs["policy"]) == {"latest", "first", "delivered"}
    for policy in ("latest", "first", "delivered"):
        shown = rungs[(rungs["policy"] == policy) & (rungs["window"] == 10)].set_index("rung")
        days = [d for d in DAYS if oracle.revision(d, policy) is not None]
        served = [d for d in days if d != FALLBACK_DAY]
        assert shown.loc[0, "days"] == len(served) * len(HORIZONS) and shown.loc[0, "forecasts"] == len(served) * 48
        assert shown.loc[1, "days"] == (len(HORIZONS) if FALLBACK_DAY in days else 0)
        assert shown.loc[1, "forecasts"] == (48 if FALLBACK_DAY in days else 0)
        assert shown.loc[0, "rung_member"] == "lear" and shown.loc[1, "rung_member"] == "naive"
        both = shown["days"].sum()
        assert shown["share"].sum() == pytest.approx(1.0 if both else 0.0)
    latest = rungs[(rungs["policy"] == "latest") & (rungs["window"] == 10)].set_index("rung")
    frame = oracle.keys("best", [FALLBACK_DAY])
    assert latest.loc[1, "scored"] == len(frame)
    assert latest.loc[1, "mae"] == pytest.approx(frame["error"].abs().mean(), rel=1e-12)


def write_quality(directory, day, revision, record):
    path = LiveStore(directory).revision_dir(day, revision) / "quality.json"
    path.write_text(json.dumps(record))


def quality_record(
    day, *, missing=0, imputed=0, failed=False, last_valid="2026-10-20T23:00:00+02:00", column="load_forecast"
):
    return {
        "issue": day,
        "sources": {"entsoe": {"status": "failed" if failed else "ok", "last_valid": {column: last_valid}}},
        "columns": {
            column: {
                "issue_rows": 24,
                "missing": missing,
                "imputed": imputed,
                "flag": None,
                "last_valid": last_valid,
                "vintage": "revised",
            },
            "price_d-1": {
                "issue_rows": 24,
                "missing": 0,
                "imputed": None,
                "flag": None,
                "last_valid": None,
                "vintage": None,
            },
        },
    }


@pytest.fixture()
def with_quality(tmp_path):
    directory = tmp_path / "live"
    write_live(directory)
    write_quality(directory, DAYS[2], 2, quality_record(DAYS[2], missing=3))
    write_quality(directory, DAYS[4], 2, quality_record(DAYS[4], imputed=2))
    write_quality(directory, DAYS[5], 1, quality_record(DAYS[5]))
    write_quality(directory, DAYS[8], 1, quality_record(DAYS[8], failed=True, last_valid="2026-10-22T23:00:00+02:00"))
    write_quality(directory, DAYS[9], 1, quality_record(DAYS[9], last_valid="2026-10-22T23:00:00+02:00"))
    # day 2's r1 has a bad record that the policy "latest" must not read
    write_quality(directory, DAYS[2], 1, quality_record(DAYS[2], missing=99))
    return directory


def test_the_quality_days_come_from_the_policy_s_revision(with_quality):
    report = run_monitor(with_quality, windows=(10, 4))
    quality = report.quality().set_index("issue")
    assert quality.loc[DAYS[2], ["revision", "missing_cells", "problem"]].tolist() == [2, 3, True]
    assert quality.loc[DAYS[4], ["imputed_cells", "imputed_columns"]].tolist() == [2, "load_forecast"]
    assert quality.loc[DAYS[5], "problem"] == False  # noqa: E712
    assert quality.loc[DAYS[8], "failed_sources"] == "entsoe"
    assert quality.loc[DAYS[3], "record"] == False  # noqa: E712  (a committed day with no file)
    assert MISSING_DAY not in quality.index
    table = report.tables().set_index(["window", "member"])
    assert table.loc[(10, "lear"), ["quality_records", "quality_days"]].tolist() == [5, 3]
    assert table.loc[(4, "lear"), ["quality_records", "quality_days"]].tolist() == [2, 1]
    first = run_monitor(with_quality, windows=(10,), policy="first").quality().set_index("issue")
    assert first.loc[DAYS[2], "missing_cells"] == 99


# --- the backtest comparison ------------------------------------------------------------------------------------------------------


def backtest_of(scale_lear, scale_naive, days):
    frames = {}
    for member, scale in (("lear", scale_lear), ("naive", scale_naive)):
        rows = []
        for day in days:
            for horizon in HORIZONS:
                for hour in HOURS:
                    target = plus(day, horizon)
                    actual = actual_of(target, hour)
                    if math.isnan(actual):
                        continue
                    sign = 1.0 if hour % 2 == 0 else -1.0
                    rows.append(
                        dict(
                            run_date=pd.Timestamp(day),
                            target_date=pd.Timestamp(target),
                            hour=hour,
                            horizon=horizon,
                            prediction=actual + sign * scale,
                            actual=actual,
                        )
                    )
        frames[member] = pd.DataFrame(rows)
    return frames


def test_the_live_over_backtest_skill_ratio_on_matched_target_days(live):
    days = [
        plus(DAYS[0], i) for i in range(-3, 6)
    ]  # the backtest covers 10-15 to 10-23: overlapping the live days in part
    frames = backtest_of(4.0, 10.0, days)
    report = run_monitor(live, windows=(10,), backtest=frames)
    got = row(report, 10, "lear")
    oracle = Oracle()
    live_lear = oracle.keys("lear", DAYS).set_index(["day", "horizon", "hour"])
    live_naive = oracle.keys("naive", DAYS).set_index(["day", "horizon", "hour"])
    both = live_lear.index.intersection(live_naive.index)
    target = lambda idx: [plus(d, h) for d, h, _ in idx]  # noqa: E731
    bt_days = set(plus(r.run_date.date().isoformat(), r.horizon) for r in frames["lear"].itertuples())
    matched = sorted(set(target(both)) & bt_days)
    assert matched and got["matched_days_naive"] == len(matched)
    live_sel = [k for k in both if plus(k[0], k[1]) in matched]
    live_skill = live_lear.loc[live_sel, "error"].abs().mean() / live_naive.loc[live_sel, "error"].abs().mean()
    assert got["backtest_skill_naive"] == pytest.approx(4.0 / 10.0, rel=1e-9)
    assert got["live_vs_backtest_naive"] == pytest.approx(live_skill / (4.0 / 10.0), rel=1e-9)
    # a backtest with no day in common matches nothing
    none = run_monitor(live, windows=(10,), backtest=backtest_of(4.0, 10.0, ["2026-09-01"]))
    assert row(none, 10, "lear")["matched_days_naive"] == 0 and math.isnan(
        row(none, 10, "lear")["live_vs_backtest_naive"]
    )
    assert "backtest_skill_naive" not in run_monitor(live, windows=(10,)).tables()


# --- alerts ----------------------------------------------------------------------------------------------------------------------


def daily_skill(report, member="lear", bench="naive"):
    daily = report.daily()
    return (
        daily[(daily["member"] == member)]
        .dropna(subset=[f"skill_{bench}"])
        .sort_values(["run_date", "horizon"])[f"skill_{bench}"]
        .to_numpy()
    )


def fired(report, rule):
    return [alert for alert in report.alerts if alert.rule == rule]


def test_skill_above_fires_exactly_when_the_last_n_days_all_exceed_it(live):
    """lear's last three issue days are planted worse than the benchmark's relative scale, the one before them better."""
    skill = daily_skill(run_monitor(live, windows=(10,)))
    # the last scored days, in order (day 9's horizon 1 and day 8's two horizons...): take n = 3 and n = 4 from the tail
    for n in (1, 2, 3, 4):
        tail = skill[-n:]
        weakest = tail.min()
        report = run_monitor(
            live, windows=(10,), alerts=[{"skill_above": float(weakest) - 1e-9, "for_days": n, "member": "lear"}]
        )
        assert len(fired(report, "skill_above")) == 1, n
        assert fired(report, "skill_above")[0].value == pytest.approx(weakest)
        none = run_monitor(
            live, windows=(10,), alerts=[{"skill_above": float(weakest) + 1e-9, "for_days": n, "member": "lear"}]
        )
        assert fired(none, "skill_above") == [], n
        equal = run_monitor(
            live, windows=(10,), alerts=[{"skill_above": float(weakest), "for_days": n, "member": "lear"}]
        )
        assert fired(equal, "skill_above") == [], n  # above is strict
    # a window longer than the days with a score doesn't fire; and by default every non-benchmark member is checked
    assert (
        fired(
            run_monitor(live, windows=(10,), alerts=[{"skill_above": 0.0, "for_days": len(skill) + 1}]), "skill_above"
        )
        == []
    )
    both = run_monitor(live, windows=(10,), alerts=[{"skill_above": 0.0, "for_days": 1}])
    assert {a.member for a in fired(both, "skill_above")} == {"lear", "best"}
    assert all(a.benchmark == "naive" for a in both.alerts)


def test_participation_and_delivered_alerts(live):
    report = run_monitor(
        live,
        windows=(10, 3),
        alerts=[
            {"participation_below": 0.81, "window": 10},  # 8 of 10 days
            {"participation_below": 0.8},
            {"delivered_below": 0.31, "window": 10},  # 3 of 10
            {"delivered_below": 0.3, "window": 10},
            {"participation_below": 0.7, "window": 3},  # days 8, 9 of the last three committed: 2/3
            {"participation_below": 0.6, "window": 3},
        ],
    )
    kinds = [(a.rule, round(a.value, 6)) for a in report.alerts]
    assert kinds == [("participation_below", 0.8), ("delivered_below", 0.3), ("participation_below", 0.666667)]


def test_a_rung_alert_fires_on_the_fallback_days_alone(tmp_path):
    directory = tmp_path / "live"
    write_live(directory)
    # The fallback day is not the latest: nothing fires for the last day, and a deeper rung than there is never fires.
    report = run_monitor(
        directory,
        windows=(10,),
        alerts=[{"rung_at_least": 1}, {"rung_at_least": 1, "for_days": 5}, {"rung_at_least": 2, "for_days": 10}],
    )
    assert report.alerts == []
    # as of the fallback day itself (the last due day then) it fires: a day is an issue day at one horizon, and the fallback day
    # has two of them, the day before it none
    for_day = run_monitor(
        directory,
        windows=(10,),
        now=pd.Timestamp(f"{FALLBACK_DAY}T12:30", tz=ZONE),
        alerts=[
            {"rung_at_least": 1, "for_days": 1},
            {"rung_at_least": 1, "for_days": 2},
            {"rung_at_least": 1, "for_days": 3},
        ],
    )
    alerts = fired(for_day, "rung_at_least")
    assert [a.message[-10:] for a in alerts] == [FALLBACK_DAY] * 2 and {a.member for a in alerts} == {"best"}
    assert [a.through for a in alerts] == [FALLBACK_DAY] * 2
    assert (
        fired(
            run_monitor(
                directory,
                windows=(10,),
                now=pd.Timestamp(f"{plus(FALLBACK_DAY, -1)}T12:30", tz=ZONE),
                alerts=[{"rung_at_least": 1}],
            ),
            "rung_at_least",
        )
        == []
    )


def test_a_rung_alert_names_the_member(live):
    with pytest.raises(ValueError, match="member"):
        run_monitor(live, windows=(10,), alerts=[{"rung_at_least": 1, "member": "lear"}])


def test_stale_input_fires_on_the_column_s_last_valid_stamp(with_quality):
    # day 8 and 9's records: last valid 10-22 23:00, 3.04 and 4.04 days before their midnight; day 5's is 2.04
    def fired_for(**kw):
        report = run_monitor(with_quality, windows=(10,), alerts=[{"stale_input": "load_forecast", **kw}])
        return fired(report, "stale_input")

    assert len(fired_for(age=3.0, for_days=2)) == 1
    assert len(fired_for(age=3.5, for_days=2)) == 0  # day 8 is 3.04 days stale, no more than 3.5
    assert len(fired_for(age=3.5, for_days=1)) == 1  # day 9 alone: 4.04
    assert len(fired_for(age=3.0, for_days=3)) == 0  # the day before them has a fresher value
    assert len(fired_for(age=5.0, for_days=1)) == 0
    assert fired_for(age=3.0, for_days=2)[0].through == DAYS[9]
    unknown = fired(
        run_monitor(with_quality, windows=(10,), alerts=[{"stale_input": "no_such_column", "age": 0}]), "stale_input"
    )
    assert unknown == []  # a column no record names is no information


def test_each_rule_fires_on_its_planted_condition_alone(with_quality):
    """All five rules over one directory: the thresholds are set just beyond what was planted, and nothing else fires."""
    skill = daily_skill(run_monitor(with_quality, windows=(10,)))
    rules = [
        {"skill_above": float(skill[-3:].min()) - 1e-9, "for_days": 3, "member": "lear", "benchmark": "naive"},
        {"participation_below": 0.81},
        {"delivered_below": 0.31},
        {"rung_at_least": 3},
        {"stale_input": "load_forecast", "age": 6.0},
    ]
    report = run_monitor(with_quality, windows=(10,), alerts=rules)
    assert [a.rule for a in report.alerts] == ["skill_above", "participation_below", "delivered_below"]
    assert all(a.message for a in report.alerts) and report.alerts[0].to_dict()["rule"] == "skill_above"


def test_alerts_are_checked(live):
    for bad in (
        [{"skill_above": 0.5, "for_dayz": 3}],
        [{"skill_above": 0.5, "participation_below": 0.9}],
        [{"for_days": 3}],
        [{"skill_above": "high"}],
        [{"participation_below": 0.9, "window": 0}],
        [{"stale_input": 3}],
        [["skill_above", 0.5]],
    ):
        with pytest.raises((ValueError, TypeError)):
            load_alerts(bad)
    with pytest.raises(ValueError, match="benchmark"):
        run_monitor(live, windows=(10,), alerts=[{"skill_above": 0.5, "benchmark": "tso"}])
    assert load_alerts(None) == [] and load_alerts([]) == []


def test_alerts_from_yaml_text_and_file(live, tmp_path):
    text = "- skill_above: 0.0\n  for_days: 1\n- participation_below: 0.5\n"
    assert load_alerts(text) == [{"skill_above": 0.0, "for_days": 1}, {"participation_below": 0.5}]
    path = tmp_path / "alerts.yaml"
    path.write_text(text)
    report = run_monitor(live, windows=(10,), alerts=path)
    assert {a.rule for a in report.alerts} == {"skill_above"}


# --- misc ------------------------------------------------------------------------------------------------------------------------


def test_a_backtest_directory_and_bad_arguments_raise(tmp_path, live):
    backtest = tmp_path / "backtest"
    backtest.mkdir()
    with pytest.raises(LiveDirectoryError):
        pipeline().monitor(DATA, backtest)
    with pytest.raises(ValueError, match="policy"):
        run_monitor(live, policy="newest")
    with pytest.raises(ValueError, match="windows"):
        run_monitor(live, windows=(0,))
    with pytest.raises(ValueError, match="benchmarks"):
        run_monitor(live, benchmarks=["tso"])
    with pytest.raises(KeyError, match="price"):
        with use_clock(FakeClock(NOW)):
            pipeline().monitor(DATA.rename(columns={"price": "p"}), live)


def test_a_pipeline_without_an_information_set_cannot_tell_what_is_known(live):
    pipe = ModelPipeline()
    pipe.add_model(OLSModel(["price_d-1"], name="lear")).add_model(NaiveModel(["price_d-1"], name="naive"))
    with use_clock(FakeClock(NOW)), pytest.raises(ValueError, match="information set"):
        pipe.monitor(DATA, live)


def test_members_without_results_are_listed_and_the_default_benchmark_is_the_naive_model(live):
    pipe = pipeline()
    pipe.add_model(OLSModel(["price_d-1"], name="elsewhere"))
    with use_clock(FakeClock(NOW)):
        report = pipe.monitor(DATA, live, windows=(7,))
    assert report.missing == ("elsewhere",) and report.benchmarks == ("naive",)
    assert "elsewhere" not in report.members and report.members == ("lear", "naive", "best")
    assert report.policy == "latest" and report.through == DAYS[9]


def test_an_empty_directory_gives_empty_tables(tmp_path):
    directory = tmp_path / "live"
    LiveStore(directory).init(mode="live", issue_timezone=ZONE, deadline=DEADLINE)
    with use_clock(FakeClock(NOW)):
        report = pipeline().monitor(DATA, directory)
    assert report.tables().empty and report.alerts == [] and report.through is None
    assert report.members == () and set(report.missing) == {"lear", "naive", "best"}


def test_monitoring_writes_nothing(live):
    before = sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in live.rglob("*") if p.is_file())
    run_monitor(live, windows=(7, 30), policy="delivered")
    assert before == sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in live.rglob("*") if p.is_file())


def test_parquet_results_read_the_same_as_jsonl(tmp_path):
    jsonl, parquet = tmp_path / "jsonl", tmp_path / "parquet"
    write_live(jsonl)
    write_live(parquet, save_format="parquet")
    a = run_monitor(jsonl, windows=(7, 30)).tables()
    b = run_monitor(parquet, windows=(7, 30)).tables()
    pd.testing.assert_frame_equal(a, b)
    assert not b.empty


@pytest.mark.parametrize(
    "start, days, test, zone",
    [
        ("2023-03-01", 45, ("2023-03-22", "2023-03-31"), "Europe/Warsaw"),  # the spring switch, 03-26
        ("2023-10-10", 40, ("2023-10-24", "2023-11-01"), "Europe/Warsaw"),  # the autumn switch, 10-29
    ],
)
def test_the_actual_of_a_key_is_the_one_a_run_scores(start, days, test, zone, monkeypatch):
    """The monitor reads each key's actual from the data: it must be the `actual` of the record a run of the same data wrote,
    on every key including the switch days (NaN where a run has none: the repeated hour, the skipped one)."""
    from epftoolbox2.results.monitor import _Actuals
    from tests.synthetic import PREDICTORS, make_market_data

    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    data = make_market_data(start=start, days=days, tz=zone, horizon=2, noise=1.0)
    report = (
        ModelPipeline(issue_time="11:45", issue_timezone="Europe/Warsaw")
        .add_model(OLSModel(predictors=PREDICTORS, training_window=14, name="OLS"))
        .run(data, test=test, horizon=2)
    )
    frame = report._model_frames()["OLS"].copy()
    assert len(frame) > 400
    frame["run_date"] = pd.to_datetime(frame["run_date"]).astype("datetime64[ns]")
    info = InformationSet("11:45", "Europe/Warsaw", rules={"price": Availability.realised("1h")})
    far = pd.Timestamp("2100-01-01", tz="UTC")
    actual, known = _Actuals(data, "price", "1h", info.in_data_zone(data.index), far).lookup(frame)
    expected = frame["actual"].to_numpy(dtype=float)
    assert np.array_equal(actual, expected, equal_nan=True)
    assert known.sum() == np.isfinite(expected).sum() > 400
    # the run itself leaves out a key whose target doesn't exist; a key it keeps with no actual has none here either
    assert not (known & ~np.isfinite(expected)).any()


def test_the_default_rule_of_the_price_serves_when_no_rule_is_given(live):
    """An information set with no rule of its own for the target reads the library's default for `price` (published 13:00 the day
    before), which is the rule the fixtures give explicitly."""
    pipe = pipeline()
    pipe.information_set = InformationSet("11:45", ZONE)
    with use_clock(FakeClock(NOW)):
        plain = pipe.monitor(DATA, live, benchmarks=["naive"], windows=(10,))
    pd.testing.assert_frame_equal(plain.tables(), run_monitor(live, windows=(10,)).tables())


def test_a_long_one_line_list_of_rules_is_yaml_not_a_file_name():
    text = "[" + ", ".join('{"participation_below": 0.%d}' % (i % 9 + 1) for i in range(40)) + "]"
    assert len(text) > 300 and "\n" not in text
    assert len(load_alerts(text)) == 40


def test_monitor_takes_one_frame(live):
    with pytest.raises(TypeError, match="DataFrame"):
        pipeline().monitor({"DE": DATA}, live)


def test_whether_a_daily_key_is_known_does_not_depend_on_its_batch():
    """On a daily frame two horizons with one target date repeat an instant, which made the reader take the daily rows for hourly
    ones: a key was known or not by what it was looked up next to."""
    from epftoolbox2.results.monitor import _Actuals

    index = pd.date_range("2026-03-01", "2026-03-20", freq="D", tz="Europe/Lisbon")
    data = pd.DataFrame({"price": np.arange(len(index), dtype=float)}, index=index)
    info = InformationSet(
        "11:45", "Europe/Brussels", rules={"price": Availability.published("13:00", 1, "Europe/Brussels")}
    ).in_data_zone(data.index)
    for now in ("2026-03-10T12:30", "2026-03-10T13:30"):
        look = _Actuals(data, "price", "1D", info, pd.Timestamp(now, tz="Europe/Brussels")).lookup
        alone = pd.DataFrame({"run_date": pd.to_datetime(["2026-03-10"]), "hour": [0], "horizon": [1]})
        batch = pd.DataFrame(
            {"run_date": pd.to_datetime(["2026-03-10", "2026-03-09"]), "hour": [0, 0], "horizon": [1, 2]}
        )
        assert look(alone)[1][0] == look(batch)[1][0] == look(batch)[1][1]  # the same target: one answer
        assert look(batch.iloc[::-1])[1].tolist() == look(batch)[1].tolist()[::-1]  # whatever the order


@pytest.mark.parametrize("at", ["11:00", "11:58", "13:30"])
def test_a_delivery_recorded_after_the_forecast_deadline_still_counts(tmp_path, monkeypatch, at):
    """The revision must have been committed by the deadline (11:55); when its delivery was recorded doesn't matter: deliveries
    happen between that deadline and the gate, or are marked later from a receipt."""
    monkeypatch.setitem(globals(), "DELIVERIES", [(DAYS[9], 1, at)])
    directory = tmp_path / "live"
    write_live(directory)
    got = run_monitor(directory, windows=(1,), policy="delivered")
    assert row(got, 1, "lear")["delivered"] == 1.0 and row(got, 1, "lear")["delivered_days"] == 1
    assert row(got, 1, "lear")["forecasts"] > 0 and row(got, 1, "lear")["member_days"] == 1
    # the as-of reader of 8c, which live readers need, still asks the delivery to be recorded by the cut-off
    visible = LiveStore(directory).visible("delivered", "deadline")
    assert (DAYS[9] in visible) == (at == "11:00")


def test_a_delivered_revision_must_still_have_been_committed_by_the_deadline(tmp_path, monkeypatch):
    # day 3's second revision was committed after the deadline: delivering it makes it neither visible nor delivered
    monkeypatch.setitem(globals(), "DELIVERIES", [(DAYS[3], 2, "12:40")])
    directory = tmp_path / "live"
    LiveStore(directory)
    write_live(directory, receipts=False)
    store = LiveStore(directory)
    receipt = directory / "r.json"
    receipt.write_text("{}")
    with store.hold():
        store.mark_delivered(DAYS[3], 2, receipt, delivered_at=local(DAYS[3], "12:40"))
    got = run_monitor(directory, windows=(10,), policy="delivered")
    assert got.tables()["delivered_days"].iloc[0] == 0 and got.tables()["forecasts"].iloc[0] == 0
