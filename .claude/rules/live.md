---
paths:
  - "epftoolbox2/_clock.py"
  - "epftoolbox2/pipelines/_live.py"
  - "epftoolbox2/pipelines/_live_settings.py"
  - "epftoolbox2/pipelines/_issue_records.py"
  - "epftoolbox2/results/live.py"
  - "epftoolbox2/results/forecast_result.py"
  - "epftoolbox2/results/monitor.py"
  - "epftoolbox2/results/checks.py"
  - "epftoolbox2/models/_fits.py"
  - "epftoolbox2/ensembles/fallback.py"
  - "epftoolbox2/data/archive.py"
  - "epftoolbox2/data/sources/archive.py"
  - "epftoolbox2/data/sources/_http.py"
  - "epftoolbox2/cli.py"
  - "tests/live_week.py"
  - "tests/test_simulated_week.py"
  - "tests/test_cli_live.py"
  - "tests/pipelines/test_forecast_issue.py"
  - "tests/models/test_stored_fits.py"
  - "tests/results/test_live_store.py"
  - "tests/results/test_monitor.py"
  - "docs/src/content/docs/operations/**"
---

# Live operations (step 8): what must stay true

A live call (`ModelPipeline.forecast`, `pipelines/_live.py`) issues one day into a live directory and commits numbered revisions.
Its one promise: **a live forecast is the backtest's record of the same pipeline over the same frame, to the bit** (but
`day_in_test` and the live keys `revision`, `issued_at`, `mode`, `snapshot`). Only which fit (stored or inline) and which frame
differ. Never give the live path its own predict, plan, mask or list rule. `tests/test_simulated_week.py` checks it end to end over
a week; `tests/pipelines/test_forecast_issue.py` per list kind, refit and engine.

## The decisions (step 8 spec, "Decisions"), as invariants

2. **A revision means something changed.** A call whose records equal the latest committed revision's, key by key, writes no
   revision and reports `unchanged` with that revision's status (`_live._canonical`: prediction, status, `detail`, `rung`, `member`
   of every model and ensemble; not `list_hash`). Two consequences the week pins: a skip's list isn't compared, so a member skipped
   in r1 (made before EXAA) keeps r1's list when a later call only changed which list it would have read (Friday: main's final
   skips carry the list without EXAA, the backtest's the list with it); and a `check_failed` detail names the failing value, so
   the same failure with another list writes a revision (Tuesday's r2). Changing what `_canonical` compares changes both.
3. **Live and backtest directories are separate.** `run()` into a live directory and `forecast()`/`refit()` into a backtest's
   raise (`LiveDirectoryError`). A live manifest compares the configuration (`LIVE_UNCOMPARED` leaves out `data_columns_hash`,
   which grows every day); each revision's inputs are hashed in its snapshot instead. `live.json` holds the issue zone, the issue
   days and frozen choices, compared on every call (a dry run's too); a deadline is each call's own.
4. **Stored fits are arrays, never pickles.** `units.npz` + `fit.json` (sha256 of the npz, the configuration digest, the rows'
   hashes), checked on load; OLS, Ridge, LassoCV and LEAR store (`_fit_state`/`_from_state`, scalers' `state()`), every other
   family fits at serve time. A stored fit of another digest is neither served nor taken as an older fit.
5. **No fit before its newest target is out.** A fit made now (refit, inline at serve time, a model run through its own run of
   the day) is refused with `FitTooEarlyError` while a training target it reads, or an input of its newest training row, isn't
   known by its rule at the clock's now (`check_published`, `refuse_too_early`). By the rule, not by what the frame holds.
6. **A dry run writes nothing anywhere**: no revision, snapshot, call line, delivered line, lock file, live directory, source
   cache or answer cache (`DataPipeline.run(cache_writes=False)`); it still refuses what the real call would. The week checks it by
   bytes and mtimes of every file.
7. **Live `ctx.published(c)` is "known by the issue time under the rules" and "present in the frame now"**: the backtest's
   `_Cells` over the refreshed frame. A revision before an input is out uses the list without it; no list ever uses a column the
   backtest couldn't (an early value whose rule says after the issue time stays unused).
8. **A revision is visible only through its commit line, written last; numbers are never reused.** Order: a `begun` call line,
   the revision folder, every model's and ensemble's records, the snapshot files, the live manifests, then the `committed` line
   (fsync). Readers take per key the highest committed revision by their cut-off and read a day from one revision only (a crash
   between models never mixes two revisions into one curve). `next_revision` counts call lines, folders and records, so a crashed
   revision's number is skipped (Sunday: r1, crashed r2, r3).
9. **`fallback` and `late` are issue statuses, not record statuses.** A fallback record is a forecast (`status` "ok", `rung`,
   `member`); a late call writes its call line and nothing else; exit codes 0 ok, 10 fallback, 20 skipped, 30 late, 75 busy,
   1 error (`results/forecast_result.py`), 3 for a monitor alert.
10. **`FallbackEnsemble(per="day")`**: a whole target day from the first member whose keys of that day are all ok (none
    skipped, none `check_failed`), so a delivered curve is never stitched from two models. Computed from the members' records,
    live as in a backtest, so each rung's backtest score is a plain report (`report.rungs()`).

## Around them

- **One clock.** Every clock read and sleep of the package goes through `epftoolbox2.clock` (`tests/test_clock.py` scans the
  source); pool workers never read it. The issue day is "today" in the issue zone on the clock. Tests use `FakeClock` +
  `use_clock`, and `cli.main(argv, clock=)`; `--as-of` needs `--refresh never` (a replay must not fetch today's data as of then).
- **`wait_for` reads the issue rows' cells.** `{"load_forecast_d+1": "10:45"}` waits for tomorrow's load forecast; the plain
  `load_forecast` on the issue row is today's, out since yesterday, so waiting for it waits for nothing (`presets.live_defaults`
  still names the plain columns: an open question for Dawid).
- **An issue row with a missing input is a `missing_test_predictors` skip, never a NaN into a model**, whatever the family (the
  week's instrumented `_predict_fitted` counts every row it sees).
- **Stored anchors and predictor functions** (fixed in 8l): a refit planned before a late input was out stores only the list
  without it. Units the exact anchor lacks are fitted inline at the anchor under `fit_missing="previous"` (the default) and
  `"fit"`; `"skip"`, and an older anchor standing in, leave them as `no_stored_fit` skips. Before the fix, every last revision
  of a day whose function switches on EXAA lost its first rung.
- **The refresh range** (`data_span`): from the oldest row a fit of the day reads (an older stored fit's `previous_cap` included),
  less the horizon, the transformers' summed `lookback_days()` and 31 days, to the last target day. The week's is 125 days.
- **The cache never holds the last two days** before the time it was written: `on_failure="cache"` gives settled rows only, so
  during an outage recent cells are missing, and the ladder decides (Saturday: naive serves). A refresh that runs out of its
  budget records the source as `timeout`, one that fails otherwise as `failed`.

## The simulated week (`tests/test_simulated_week.py`, `tests/live_week.py`)

- The harness: a seeded truth with a publication schedule; `FakeEntsoe` patches `requests.Session.get` and answers A44 (both
  classification positions) and A65 through the real parser, 1 s per request, 20 s and 503 in the outage, `requests.Timeout`
  after the timeout when the latency doesn't fit it; the data and model pipelines of the spec; the CLI on a YAML saved from the
  same objects (one configuration digest); the lock held from a second open file description; the backtest twin over the
  server's archive at the end of the week.
- What keeps it near 21 s a pass on one process (the module about 52 s): the worker pools run in process, the package's INFO
  logging is off for the week (rich's handler took more than half of it), `EntsoeSource`'s progress bar is a stand-in, and the
  cache's Parquet pieces are read once per unchanged file (`ParquetReads`). None of them touches a number. Don't add a third pass
  (each costs about 21 s): the second pass runs in Parquet and checks both the byte-identity and the format.
- The table in the spec is the source; where the implementation gives another value the test asserts it and says why in a
  comment (Tuesday's r2, Friday's skip lists, Saturday's `timeout`, the input-quality days).
