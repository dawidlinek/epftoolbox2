"""`models/_sums.py`: exact span sums. Bitwise invariance, exactness of the integer sums, accuracy, edge cases.

The bitwise property is the point of the module: `over(rows, cols)` must not move a bit with the other columns the span holds,
their order, the tiling into blocks, the BLAS thread count, the row chunking, or whether it sums the rows or subtracts the
others. `test_over_is_bitwise_the_same_in_every_configuration` checks 100 random shapes; `library_check_step5/exact_sums/` in
the research tree runs `bitwise_case` on 1,000 larger ones (n up to 40,000, p up to 1,000). Exactness is checked directly too:
the module's integer sums equal int64 arithmetic on its own slices, on data built to sit at the bounds, which catches a β one bit
too large, an unchunked P'P or float runs longer than 2^15 rows even where the bitwise comparisons happen to agree.
"""

import math
from fractions import Fraction

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from epftoolbox2.models import _sums

FIELDS = ("n", "shift", "yshift", "u", "S", "v", "sy", "syy")


def _bits(raw) -> list:
    return [np.asarray(getattr(raw, f), dtype=np.float64 if f != "n" else np.int64).tobytes() for f in FIELDS]


def _assert_same(a, b, what=""):
    for f, x, y in zip(FIELDS, _bits(a), _bits(b)):
        assert x == y, f"{what}: field {f} differs"


# --- random spans ---------------------------------------------------------------------------------------------------


BENIGN = (0, 1, 2, 3, 4, 7, 8)  # the kinds whose Σ|d_j||d_k| isn't dwarfed by r·max|d_j|·max|d_k| (no spike, no Cauchy)


def _column(rng, n, kinds=None):
    kind = rng.integers(0, 9) if kinds is None else rng.choice(kinds)
    if kind == 0:
        return rng.normal(size=n) * 10 ** rng.uniform(-3, 4) + rng.uniform(-1e3, 1e3)
    if kind == 1:
        return np.round(rng.normal(40, 25, size=n), 2)  # prices with two decimals
    if kind == 2:
        return 1e4 + rng.integers(0, 5000, size=n).astype(float)  # loads
    if kind == 3:
        return (rng.random(n) < rng.uniform(0.05, 0.5)).astype(float)  # a dummy
    if kind == 4:
        return np.full(n, rng.choice([0.0, 0.1, 0.3, -7.25]))  # constant
    if kind == 5:
        return rng.standard_cauchy(n)
    if kind == 6:
        col = rng.normal(size=n) * 1e-3
        col[rng.integers(0, n)] = 1e5  # one spike
        return col
    if kind == 7:
        return rng.uniform(-1, 1, size=n) * 2.0 ** rng.integers(-60, 60)
    return rng.normal(size=n)


def _span(rng, n, p, kinds=None):
    """x (n, p) with NaN and inf patterns and y (n,) with a few NaN."""
    x = np.stack([_column(rng, n, kinds) for _ in range(p)], axis=1) if p else np.zeros((n, 0))
    if rng.random() < 0.8:
        x[rng.random((n, p)) < rng.uniform(0, 0.05)] = np.nan
    if rng.random() < 0.4 and n > 4:  # a gap of rows in some columns, as missing weather
        lo = rng.integers(0, n - 1)
        cols = rng.random(p) < 0.3
        x[lo : lo + rng.integers(1, max(2, n // 4)), cols] = np.nan
    if rng.random() < 0.2 and p:
        x[rng.integers(0, n), rng.integers(0, p)] = rng.choice([np.inf, -np.inf])
    if rng.random() < 0.1 and p:
        x[:, rng.integers(0, p)] = np.nan  # a column with no value in the span
    y = rng.normal(size=n) * 20 + 50
    if rng.random() < 0.7:
        y[rng.random(n) < 0.03] = np.nan
    return x, y


def _request(rng, x, y):
    """Requested columns (random subset, random order) and the rows where they and y are finite, some dropped."""
    p = x.shape[1]
    cols = rng.permutation(p)[: rng.integers(1, p + 1)]
    ok = np.isfinite(x[:, cols]).all(axis=1) & np.isfinite(y)
    if rng.random() < 0.5:
        ok &= rng.random(x.shape[0]) < rng.uniform(0.3, 1.0)
    return np.flatnonzero(ok), cols


def _block(rng, p):
    """A block width, log-uniform from max(1, p // 64) to p (at most about 2,000 block pairs)."""
    lo = max(1, p // 64)
    return int(np.exp(rng.uniform(np.log(lo), np.log(max(p, lo) + 1)))) if p else 1


def bitwise_case(seed: int, n_max: int, p_max: int, cells_max: int = 12_000_000) -> int:
    """Build one random span and request, compute `over` six ways, assert all bitwise equal. Returns n·p for the log."""
    rng = np.random.default_rng(seed)
    n = int(np.exp(rng.uniform(np.log(2), np.log(n_max))))
    p = max(1, min(int(np.exp(rng.uniform(0, np.log(p_max)))), cells_max // n))
    x, y = _span(rng, n, p)
    rows, cols = _request(rng, x, y)

    with threadpool_limits(1):
        ref = _sums.SpanSums(x, y).over(rows, cols, method="direct")

    # 2. the same union, as a downdate, other tiling and row chunks, 4 BLAS threads
    with threadpool_limits(4):
        alt = _sums.SpanSums(x, y, block_cols=_block(rng, p), chunk_rows=int(rng.integers(1, n + 1)))
        _assert_same(ref, alt.over(rows, cols, method="downdate"), f"seed {seed}: downdate")

    # 3. another union: the requested columns shuffled among new ones, Fortran order, "auto"
    extra = int(rng.integers(0, 40))
    other, _ = _span(rng, n, extra)
    order = rng.permutation(cols.size + extra)
    wide = np.concatenate([x[:, cols], other], axis=1)[:, order]
    where = np.argsort(order)[: cols.size]
    with threadpool_limits(4):
        mixed = _sums.SpanSums(np.asfortranarray(wide), y, block_cols=_block(rng, wide.shape[1]))
        _assert_same(ref, mixed.over(rows, where), f"seed {seed}: other union")

    # 4. a span whose cache other requests have filled in part, then the request, with a row mask
    with threadpool_limits(1):
        warm = _sums.SpanSums(x, y, block_cols=_block(rng, p), chunk_rows=int(rng.integers(1, n + 1)))
        for _ in range(2):
            some = rng.permutation(p)[: rng.integers(1, p + 1)]
            ok = np.flatnonzero(np.isfinite(x[:, some]).all(axis=1) & np.isfinite(y))
            warm.over(ok, some)
        mask = np.zeros(n, dtype=bool)
        mask[rows] = True
        _assert_same(ref, warm.over(mask, cols), f"seed {seed}: warm cache")
        _assert_same(ref, warm.over(rows[::-1].copy(), cols, method="direct"), f"seed {seed}: rows reversed")

    # 5. the request's own columns in another order: the same sums, permuted
    perm = rng.permutation(cols.size)
    with threadpool_limits(4):
        shuffled = _sums.SpanSums(x, y, block_cols=_block(rng, p)).over(rows, cols[perm])
    expected = ref._replace(shift=ref.shift[perm], u=ref.u[perm], S=ref.S[np.ix_(perm, perm)], v=ref.v[perm])
    _assert_same(expected, shuffled, f"seed {seed}: request order")
    return n * p


def test_over_is_bitwise_the_same_in_every_configuration():
    """100 random shapes: direct at 1 thread = downdate at 4 = another union = a warm cache, all fields, all bits."""
    for seed in range(100):
        bitwise_case(seed, n_max=2500, p_max=120)


def test_rows_past_2_15_are_summed_in_runs_and_any_chunking_moves_no_bit():
    rng = np.random.default_rng(7)
    n = 40_000
    x = rng.normal(size=(n, 5)) * [1, 1e3, 1e-2, 50, 7] + [0, 5e4, 0, 40, -3]
    x[rng.random((n, 5)) < 0.01] = np.nan
    y = rng.normal(size=n) * 30 + 45
    rows = np.flatnonzero(np.isfinite(x[:, [0, 1, 3]]).all(axis=1))
    cols = [3, 0, 1]
    assert _sums.slice_bits(n) == 19
    ref = _sums.SpanSums(x, y).over(rows, cols, method="direct")
    for chunk in (1 << 15, 4096, 777):
        for method in ("direct", "downdate"):
            _assert_same(ref, _sums.SpanSums(x, y, chunk_rows=chunk, block_cols=2).over(rows, cols, method=method))
    _assert_same(ref, _sums.SpanSums(x[:, [3, 0, 1]], y).over(rows, [0, 1, 2]))


# --- the integer sums are exact ---------------------------------------------------------------------------------------


def _at_the_bounds(rng, n, p):
    """Columns of ± values (shift exactly 0) whose w = d·2^(β-e) sits just under 2^β with a fractional part near +0.49,
    so Q1 is near 2^β and Q2 near 2^(β-1) with the same sign: every slice sum is as large as the bounds allow. All columns are
    positive in the first half of the rows and negative in the second, so cross products add up too; each column's second
    half is its own permutation of the first, so they don't pair up into even integers that float rounding would leave alone."""
    beta = _sums.slice_bits(n)
    half = (n + 1) // 2
    cols = []
    for _ in range(p):
        w = 2.0**beta - 1 + 0.49 + rng.uniform(-0.004, 0.004, size=half) - rng.integers(0, 3, size=half)
        mag = np.ldexp(w, int(rng.integers(-20, 20)) - beta)
        if n == 1:
            cols.append(mag)
        else:  # an odd n gets a 0 in the middle, so the values stay symmetric and the shift exactly 0
            cols.append(np.concatenate([mag[: n // 2], [0.0] * (n % 2), -rng.permutation(mag[: n // 2])]))
    return np.stack(cols, axis=1)


def _exact_terms(qa, qb):
    """(T2, T3, T4) of slice arrays (3 × columns × rows) in int64 arithmetic, no BLAS."""
    a, b = qa.astype(np.int64), qb.astype(np.int64)
    t2 = a[0] @ b[0].T
    t3 = a[0] @ b[1].T + a[1] @ b[0].T
    t4 = a[0] @ b[2].T + a[2] @ b[0].T + a[1] @ b[1].T
    return np.stack([t2, t3, t4])


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 17, 1024, 1449, 2048, 1 << 15, 60_000])
def test_slice_sums_equal_int64_arithmetic_at_the_bounds(n):
    rng = np.random.default_rng(n)
    p = 6
    x = _at_the_bounds(rng, n, p)
    y = _at_the_bounds(rng, n, 1)[:, 0]
    ss = _sums.SpanSums(x, y)
    ss.over(np.arange(n), np.arange(p))  # fills slices and totals
    q = ss._q
    assert np.abs(q[0]).max() <= 2.0**ss.beta and np.abs(q[1:]).max() <= 2.0 ** (ss.beta - 1)
    if n >= 1024:  # the data does reach the bounds: Q1 within 3 of 2^β, Q2 near 2^(β-1)
        assert np.abs(q[0]).max() >= 2.0**ss.beta - 3 and np.abs(q[1]).max() >= 0.48 * 2.0**ss.beta
    exact = _exact_terms(q, q)
    np.testing.assert_array_equal(ss._gram, exact)
    np.testing.assert_array_equal(_sums._gram_terms(q, None, _sums.CHUNK_ROWS, ss._pair_chunk), exact)
    np.testing.assert_array_equal(_sums._gram_terms(q, None, 97, 0), exact)  # Q1'Q2 instead of P'P
    np.testing.assert_array_equal(_sums._gram_terms(q[:, :2], q[:, 2:], 1000, ss._pair_chunk), exact[:, :2, 2:])
    qy = ss._qy[:, None, :]
    np.testing.assert_array_equal(ss._vec, _exact_terms(q, qy)[:, :, 0])
    np.testing.assert_array_equal(ss._yy, _exact_terms(qy, qy)[:, 0, 0])
    np.testing.assert_array_equal(ss._lin, q.astype(np.int64).sum(axis=2))


def test_slice_bits_and_pair_rows():
    assert [_sums.slice_bits(n) for n in (0, 1, 2, 3, 4, 5, 1449, 2048, 2049, 1 << 15, 40_000)] == [
        26,
        26,
        26,
        25,
        25,
        25,
        21,
        21,
        20,
        19,
        19,
    ]
    for beta in range(19, 27):
        k = _sums.pair_rows(beta)
        assert k * 2.25 * 4.0**beta <= 2.0**53 < (k + 1) * 2.25 * 4.0**beta
    assert _sums.pair_rows(26) == 0


# --- accuracy against exact rational sums ------------------------------------------------------------------------------


def _exact_raw(x, y, rows, cols, shift, yshift):
    """u, S, v, sy, syy as Fractions of the floats d = fl(x - c), dy = fl(y - c_y) the module sums."""
    d = x[np.ix_(rows, cols)] - shift
    dy = y[rows] - yshift
    fd = [[Fraction(v) for v in col] for col in d.T]
    fy = [Fraction(v) for v in dy]
    p = len(cols)
    S = [[sum((a * b for a, b in zip(fd[j], fd[k])), Fraction(0)) for k in range(p)] for j in range(p)]
    u = [sum(col, Fraction(0)) for col in fd]
    v = [sum((a * b for a, b in zip(col, fy)), Fraction(0)) for col in fd]
    return d, dy, u, S, v, sum(fy, Fraction(0)), sum((a * a for a in fy), Fraction(0))


def _ulps(value, exact, scale):
    """|value - exact| in ulps of `scale` (a nonnegative float); 0 when both are 0."""
    err = abs(Fraction(float(value)) - exact)
    if scale == 0:
        assert err == 0
        return 0.0
    return float(err / Fraction(math.ulp(scale)))


def test_accuracy_within_4_ulp_and_closer_than_a_blas_gram():
    """Small spans: every element within 4 ulp of Σ|d_j||d_k| of the exact sum, and no farther than `D.T @ D` from it."""
    rng = np.random.default_rng(11)
    ours_total = blas_total = 0.0
    for case in range(40):
        n, p = int(rng.integers(2, 60)), int(rng.integers(1, 7))
        x, y = _span(rng, n, p, BENIGN)
        x[~np.isfinite(x)] = np.nan
        rows, cols = _request(rng, x, y)
        if rows.size == 0:
            continue
        raw = _sums.SpanSums(x, y).over(rows, cols)
        d, dy, u, S, v, sy, syy = _exact_raw(x, y, rows, cols, raw.shift, raw.yshift)
        T = np.abs(d).T @ np.abs(d)
        blas = d.T @ d
        ours_max = blas_max = 0.0
        for j in range(len(cols)):
            for k in range(len(cols)):
                o, b = _ulps(raw.S[j, k], S[j][k], T[j, k]), _ulps(blas[j, k], S[j][k], T[j, k])
                assert o <= 4, f"case {case}: S[{j},{k}] {o} ulp"
                ours_max, blas_max = max(ours_max, o), max(blas_max, b)
                ours_total, blas_total = ours_total + o, blas_total + b
            assert _ulps(raw.u[j], u[j], np.abs(d[:, j]).sum()) <= 4
            assert _ulps(raw.v[j], v[j], np.abs(d[:, j]) @ np.abs(dy)) <= 4
        assert ours_max <= blas_max, f"case {case}: {ours_max} ulp against BLAS's {blas_max}"
        assert _ulps(raw.sy, sy, np.abs(dy).sum()) <= 4
        assert _ulps(raw.syy, syy, dy @ dy) <= 4
    assert ours_total < blas_total


def _fsum_dot(a, b):
    """Σ a_i b_i correctly rounded: Dekker's exact products, then `math.fsum` of their two parts."""
    split = 134217729.0  # 2^27 + 1
    ah = a * split - (a * split - a)
    bh = b * split - (b * split - b)
    al, bl = a - ah, b - bh
    prod = a * b
    err = ((ah * bh - prod) + ah * bl + al * bh) + al * bl
    return math.fsum(np.concatenate([prod, err]).tolist())


@pytest.mark.parametrize("n", [1449, 40_000])
def test_accuracy_on_longer_spans_against_a_correctly_rounded_sum(n):
    """Within 4 ulp of Σ|d_j||d_k| (plus the reference's own half ulp), at the β of LEAR's window and past 2^15 rows."""
    rng = np.random.default_rng(n)
    x = np.stack([_column(rng, n, BENIGN) for _ in range(6)], axis=1)
    x[:, 4] = rng.normal(size=n) * 3 + 100
    y = rng.normal(size=n) * 25 + 50
    raw = _sums.SpanSums(x, y).over(np.arange(n), np.arange(6))
    d = x - raw.shift
    for j in range(6):
        for k in range(j, 6):
            T = np.abs(d[:, j]) @ np.abs(d[:, k])
            assert abs(raw.S[j, k] - _fsum_dot(d[:, j], d[:, k])) <= 4.5 * math.ulp(T)


def _assert_fixed_point_bound(x, y, rows, cols):
    raw = _sums.SpanSums(x, y).over(rows, cols)
    d, _, _, S, _, _, _ = _exact_raw(x, y, rows, cols, raw.shift, raw.yshift)
    span = x[:, cols] - raw.shift
    span[~np.isfinite(span)] = 0.0
    e = np.frexp(np.abs(span).max(axis=0))[1]  # the span's exponents, not the subset's
    beta, r = _sums.slice_bits(x.shape[0]), len(rows)
    for j in range(len(cols)):
        for k in range(len(cols)):
            scale = 2.0 ** (int(e[j]) + int(e[k]))
            bound = r * scale * (2.0 ** (1 - 3 * beta) + 2.0 ** (-52 - beta)) + math.ulp(abs(float(S[j][k])))
            assert abs(Fraction(raw.S[j, k]) - S[j][k]) < Fraction(bound)


def test_fixed_point_bound_holds_on_columns_whose_products_hide_in_small_values():
    """The guarantee: |error| < r·2^(e_j+e_k)·(2^(1-3β) + 2^(-52-β)) + an ulp of the result, relative to each column's
    largest |x - c| in the span. Built case: each column's large values meet the other's tiny ones, so Σ|d_j||d_k| is small
    beside that scale; random cases of every kind (spikes, Cauchy, a subset that drops the large rows) too."""
    rng = np.random.default_rng(5)
    n = 64
    a = rng.normal(size=n) * 1e-9
    b = rng.normal(size=n) * 1e-9
    a[:4], b[4:8] = [1e6, -1e6, 2e6, -2e6], [3e3, -3e3, 1e3, -1e3]
    _assert_fixed_point_bound(np.stack([a, b], axis=1), rng.normal(size=n), np.arange(n), [0, 1])
    _assert_fixed_point_bound(np.stack([a, b], axis=1), rng.normal(size=n), np.arange(8, n), [0, 1])
    for _ in range(30):
        n, p = int(rng.integers(2, 50)), int(rng.integers(1, 5))
        x, y = _span(rng, n, p)
        rows, cols = _request(rng, x, y)
        _assert_fixed_point_bound(x, y, rows, cols)


# --- edge cases ---------------------------------------------------------------------------------------------------------


def test_a_nan_or_inf_in_a_requested_row_raises_and_elsewhere_does_not():
    x = np.arange(30, dtype=float).reshape(10, 3)
    y = np.linspace(0, 1, 10)
    x[2, 1], x[5, 2], y[7] = np.nan, np.inf, np.nan
    ss = _sums.SpanSums(x, y)
    with pytest.raises(ValueError, match="x has a NaN"):
        ss.over([0, 1, 2], [0, 1])
    with pytest.raises(ValueError, match="x has a NaN"):
        ss.over([5, 6], [2], method="direct")
    with pytest.raises(ValueError, match="y has a NaN"):
        ss.over([6, 7], [0])
    ok = [0, 1, 3, 4, 6, 8, 9]
    raw = ss.over(ok, [0])  # the gaps in columns 1 and 2 and the removed rows don't matter
    _assert_same(raw, _sums.SpanSums(x[:, :1], y).over(ok, [0], method="direct"))
    assert raw.n == 7


def test_a_constant_column_has_zero_sums_exactly():
    rng = np.random.default_rng(1)
    n = 50
    x = np.stack([np.full(n, 0.1), np.full(n, -3.0), rng.normal(size=n), np.zeros(n)], axis=1)
    x[[3, 9], 0] = np.nan
    assert math.fsum([0.1] * 48) / 48 != 0.1  # fsum's mean of the 48 values isn't the value itself
    y = np.full(n, 42.5)
    raw = _sums.SpanSums(x, y).over(np.flatnonzero(np.isfinite(x[:, 0])), [0, 1, 2, 3])
    assert raw.shift[0] == 0.1 and raw.shift[1] == -3.0 and raw.yshift == 42.5
    for j in (0, 1, 3):
        assert not raw.S[j].any() and not raw.S[:, j].any() and raw.u[j] == 0 and raw.v[j] == 0
    assert raw.sy == 0 and raw.syy == 0 and raw.S[2, 2] > 0


def test_one_row_one_column_and_empty_requests():
    one = _sums.SpanSums(np.array([[2.5]]), np.array([1.0])).over([0], [0])
    assert one.n == 1 and one.shift[0] == 2.5 and one.S[0, 0] == 0 and one.u[0] == 0 and one.syy == 0
    x = np.array([[1.0], [4.0]])
    two = _sums.SpanSums(x, np.array([0.0, 2.0])).over([0, 1], [0])
    assert two.S[0, 0] == 4.5 and two.u[0] == 0 and two.v[0] == 3.0 and two.sy == 0 and two.syy == 2.0
    ss = _sums.SpanSums(np.ones((4, 3)), np.ones(4))
    none = ss.over([], [0, 2])
    assert none.n == 0 and none.S.shape == (2, 2) and not none.S.any() and not none.u.any()
    nothing = ss.over(np.ones(4, dtype=bool), [])
    assert nothing.S.shape == (0, 0) and nothing.n == 4


def test_s_is_exactly_symmetric_and_duplicated_columns_repeat():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(300, 9)) * 100, rng.normal(size=300)
    raw = _sums.SpanSums(x, y, block_cols=4).over(np.arange(0, 300, 2), [8, 1, 4, 1, 0])
    assert raw.S.tobytes() == raw.S.T.copy().tobytes()
    assert raw.S[1].tobytes() == raw.S[3].tobytes() and raw.u[1] == raw.u[3]


def test_the_shift_is_the_fsum_mean_of_the_finite_values():
    """`_shifts` forms exact integer sums where it can and calls `math.fsum` elsewhere; both equal fsum's mean bit for bit."""
    rng = np.random.default_rng(2)
    rows = [
        rng.normal(size=1449) * 30 + 45,
        np.round(rng.normal(size=1449) * 30, 2),
        rng.normal(size=500) * 10.0 ** rng.integers(-25, 25, size=500),  # too wide for two slices: fsum
        rng.normal(size=300) * 1e-310,  # subnormals
        np.concatenate([[2.0**53, 1.0], np.zeros(298)]),  # a tie, rounded to even
        np.concatenate([[1e308, 1e308, -1e308], np.zeros(297)]),
        np.where(rng.random(300) < 0.5, 1e-30, 1e10),
        rng.choice([0.1, 0.2, 0.3], size=300),
    ]
    for row in rows:
        values = np.concatenate([row, [np.nan, np.inf, -np.inf]])
        rng.shuffle(values)
        finite = np.isfinite(values)
        expected = math.fsum(values[finite].tolist()) / finite.sum()
        got = _sums._shifts(values[None, :], finite[None, :])[0]
        assert np.float64(got).tobytes() == np.float64(expected).tobytes()
    assert _sums._shifts(np.full((1, 5), np.nan), np.zeros((1, 5), dtype=bool))[0] == 0.0
    zeros = np.array([[0.0, -0.0, 0.0]])
    assert np.signbit(_sums._shifts(zeros, np.ones_like(zeros, dtype=bool))).tolist() == [False]


def test_ordinary_columns_need_no_fsum(monkeypatch):
    class NoFsum:
        @staticmethod
        def fsum(_):
            raise AssertionError("fell back to math.fsum")

    monkeypatch.setattr(_sums, "math", NoFsum)
    rng = np.random.default_rng(4)
    x = np.stack(
        [rng.normal(40, 25, 1449), np.round(rng.normal(40, 25, 1449), 2), 1e4 + rng.integers(0, 9, 1449)], axis=1
    )
    _sums._shifts(x.T, np.isfinite(x.T))


def test_the_sums_read_the_span_rows_only():
    """A span given as a view of a larger frame equals its copy, and rows of the frame outside it move no bit."""
    rng = np.random.default_rng(6)
    frame = rng.normal(size=(500, 6)) * [1, 10, 100, 1e3, 0.1, 5] + [0, 50, -20, 1e4, 0, 3]
    y = rng.normal(size=500)
    lo, hi = 100, 400
    rows, cols = np.arange(0, hi - lo, 3), [5, 0, 3]
    view = _sums.SpanSums(frame[lo:hi], y[lo:hi]).over(rows, cols)
    _assert_same(view, _sums.SpanSums(frame[lo:hi].copy(), y[lo:hi].copy()).over(rows, cols))
    later = frame.copy()
    later[hi:] *= 1000.0
    later[:lo] = np.nan
    _assert_same(view, _sums.SpanSums(later[lo:hi], y[lo:hi]).over(rows, cols))


def test_bad_requests_raise():
    ss = _sums.SpanSums(np.ones((5, 2)), np.ones(5))
    for rows, cols, match in [
        ([0, 0, 1], [0], "distinct"),
        ([5], [0], "row indices"),
        ([-1], [0], "row indices"),
        (np.ones(4, dtype=bool), [0], "length 5"),
        ([0.5], [0], "boolean mask or integer"),
        ([0], [2], "column indices"),
    ]:
        with pytest.raises(ValueError, match=match):
            ss.over(rows, cols)
    with pytest.raises(ValueError, match="method"):
        ss.over([0], [0], method="blas")
    with pytest.raises(ValueError, match="chunk_rows"):
        _sums.SpanSums(np.ones((5, 2)), np.ones(5), chunk_rows=(1 << 15) + 1)
    with pytest.raises(ValueError, match="x must be"):
        _sums.SpanSums(np.ones((5, 2)), np.ones(4))
