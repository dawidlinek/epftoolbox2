"""Exact sums over one training span, for the step-5 engine: a fit's statistics that no grouping can move by a bit.

`SpanSums(x, y)` holds one span's rows (x: n rows by p union columns, NaN where a value is missing; y: the target) and
`over(rows, cols)` returns `RawSums`: Σ(x - c), Σ(x - c)(x - c)', Σ(x - c)(y - c_y), Σ(y - c_y) and Σ(y - c_y)² over a subset
of the span's rows, for a subset of its columns, about per-column shifts c of the span. Every element is a function of the two
columns' values in those rows and of the span: the same whatever other columns the `SpanSums` holds, in any column order
or tiling, at any BLAS thread count, with rows chunked or not, and summed directly or as all rows minus the others (a downdate).
A BLAS Gram has none of these properties: its blocking, and so its rounding, follows the shape of the whole matrix.

Nothing calls this module yet; the engine (5i) will. numpy only.

How (per span; n rows, m = min(n, 2^15), L = ceil(log2 m)):

- **Shift** c_j: the `math.fsum` mean of column j's finite values in the span; 0 if it has none, the value itself when they are
  all equal (so a constant column has d = 0 and S = 0 exactly; fsum's mean of 0.1 three times is 0.1 + 1 ulp). `_shifts`
  computes it from exact integer sums, equal to `math.fsum`'s correctly rounded sum, and calls `math.fsum` where those can't hold
  a column exactly. d = fl(x - c) is a float; NaN and ±inf are 0 from here on. A column whose sum or some x - c overflows
  float64 is unusable: its slices are 0, the rest of its block is computed as usual, and only a request including it raises
  (naming it), so whether a fit runs never depends on which other columns share the union.
- **Exponent** e_j: `frexp` of max |d| over the span, so |d| < 2^e_j (0 for an all-zero column).
- **Slices**, β = floor((53 - L) / 2) bits each: w = d·2^(β-e) (a power-of-two scaling, exact) has |w| < 2^β; Q1 = rint(w),
  Q2 = rint(2^β(w - Q1)), Q3 = rint(2^β(2^β(w - Q1) - Q2)). Each remainder is exact (w - rint(w) is a multiple of w's ulp of
  magnitude at most 1/2, so representable, and IEEE subtraction returns a representable difference exactly), so
  d = 2^e (Q1 2^-β + Q2 2^-2β + Q3 2^-3β + ε 2^-3β) exactly, with integers |Q1| <= 2^β, |Q2|, |Q3| <= 2^(β-1) and |ε| <= 1/2.
  A w below the normal range (where the scaling rounds) has every slice 0 whatever the rounding, |w| < 2^-1022 being far
  below the 2^-2β the last slice resolves.
- **Kept pairs** s + t <= 4, as integer matrices over the rows summed: T2 = ΣQ1Q1', T3 = Σ(Q1Q2' + Q2Q1'),
  T4 = Σ(Q1Q3' + Q3Q1' + Q2Q2'). With the Gram's symmetry a diagonal block takes four products: Q1'Q1, Q2'Q2 and P'P with
  P = Q1 + Q2 (three `syrk`s; T3 = P'P - Q1'Q1 - Q2'Q2) and Q1'Q3 (added to its transpose). An off-diagonal block takes the
  same five as `gemm`s (Q1a'Q3b and Q3a'Q1b both). Against y, the six pairs are matrix-vector products.
- **Re-slicing** for a request: a column (or y) none of whose values in the requested rows is large, |d| >= 2^(e - δ - 1)
  with δ = floor((3β - 54) / 2), has its largest |d| there more than δ binades under 2^e. It is sliced again on those rows with
  their own exponent (same shift, same β), and its sums with every requested column are formed over the rows directly; the
  other pairs come from the span's slices (direct or downdate). Whether a column is re-sliced is decided by an exact count of
  its large values in the rows (all large values minus those in the removed rows, or those in the rows), from its own values,
  so the same in every way of summing. Without it, a request that leaves out a column's large values (rows with a glitch and a
  missing target, say) would hold the rest to the span's scale: 1e10 ulp off where BLAS is 2 ulp off, on a test case.

Why the integers are exact, in any order. Every kept slice-pair product is an integer of magnitude at most 2^β·2^β = 2^2β, and
a sum of at most m of them, or any part of one, an integer of magnitude at most m·2^2β <= 2^L·2^(53-L) = 2^53. Every integer
up to 2^53 is a float64, so each multiplication, addition and fused multiply-add BLAS performs on the way is exact whatever
its blocking, its kernel for the shape and its thread count. (This assumes what dgemm, dsyrk and dgemv do: each output
element is some float64 sum of the float64 products; a Strassen-like product would break it, and no BLAS numpy links uses one
for dgemm.) P has |P| <= 1.5·2^β, so P'P is summed over chunks of at most 2^(53-2β)·4/9 rows, where |P_j P_k| summed stays
<= 2^53; when that bound is under one row (β = 26, n <= 2), T3 comes from Q1'Q2 instead. P'P - Q1'Q1 is then at most
1.25 of that bound, exact too. The linear sums ΣQ_s are at most m·2^β. Within each run of at most 2^15 rows the chunks'
exact results are added in float64 (their totals obey the same m·2^2β bound), converted to int64 (exactly), and the runs
added in int64, which stays exact while 1.25·n·2^2β < 2^63, i.e. for n up to 2^24 (checked); past 2^15 rows β stays 19. A
downdate subtracts the removed rows' int64 sums from the span's totals, so it equals the direct sum over the kept rows exactly.
Chunk boundaries, the tiling into column blocks and the order of the columns therefore change no integer.

The float result is made from these integers by one fixed sequence, the same for every element:
S_jk = 2^(e_j + e_k - 2β) · fl(fl(fl(T4)·2^-β + fl(T3))·2^-β + fl(T2)), e the exponent the column was sliced with, the last
scaling exact (by multiplication while every intermediate is a normal float, else `ldexp`). Equal integers and exponents give
equal bits, which proves the invariances above: every input to an element is a function of the two columns' values in the
requested rows and of the span. Σ(x - c) uses (ΣQ1, ΣQ2, ΣQ3) and 2^(e - β) the same way; Σ(x - c)(y - c_y), Σ(y - c_y) and
Σ(y - c_y)² use y's own slices.

Which pairs, and how accurate. Per row z = d / 2^e is held to |ε|·2^-3β <= 2^(-3β-1). The dropped pairs (2,3), (3,2) are at most
2^(2β-2)·2^-5β = 2^(-3β-2) each and (3,3) at most 2^(-4β-2): the order of that residual, so keeping them (three more products)
would not make the sums more accurate, while dropping a kept one would (the smallest kept, (1,3) and (2,2), are of order
2^-2β, about n times float64's rounding). Per row the error of z_j z_k is at most 2^(-3β-1)(1 + |z_j| + |z_k|) + 2^(-4β)
< 2^(1-3β), an absolute error at the scale 2^(e_j + e_k) whatever the size of the values: the dropped pairs are below
float64's rounding of a product only where the product is not far below that scale. Re-slicing bounds how far: with m_j the
largest |x_j - c_j| over the r requested rows, e_j <= frexp(m_j) + δ, so per row the error is under
2^(1-3β+2δ)·2^(ẽ_j + ẽ_k) <= 2^-53·4·m_j·m_k, and over the rows under r·2^-51·m_j·m_k: one float64 rounding per row at the
scale of the largest product. The combination adds half an ulp of the result, at most r·2^(-53-β+2δ)·4·m_j·m_k from its inner
sum and, past 2^53 (so past 2^15 rows), the int64-to-float roundings, about 2^-53 of Σ|x_j - c_j||x_k - c_k|. In practice the
rint slices' errors cancel and the result is within an ulp of the exact sum (tests: at most 0.5 ulp of Σ|x_j - c_j||x_k - c_k|
where `X.T @ X` is up to 12 ulp off). The guarantee is still relative to m_j·m_k, not to Σ|x_j - c_j||x_k - c_k|: in rows
where a column is heavy-tailed (its few large values meeting the other's small ones) that sum can be far below r·m_j·m_k, and
the bound above, not "a few ulp of it", is what holds. In standardised units (x - μ)/σ over the rows, the units the solvers
see, the bound is r·2^-51·(m_j/σ_j)(m_k/σ_k). A BLAS Gram's bound is n·2^-53·Σ|x_j||x_k|.
`tests/models/test_sums.py` checks both against exact rational sums.

The shift comes from the span's rows, the exponent from the span's rows or the requested ones, all dated before the issue day,
so data after it, or the start of the frame, moves no bit.

Cost: per pair of columns 2.5 times the multiply-adds of a `gemm` over the rows, five times those of `X.T @ X` (`syrk`), plus the
slicing. The column blocks of the span's totals are computed when an `over` first needs them and kept. Memory per span:
3·n·p float64 for the slices of the blocks used and 3·p² int64 for the totals.
"""

import math
from typing import NamedTuple

import numpy as np

CHUNK_ROWS = 1 << 15
"""The most rows whose slice products are summed in float64; β is set by min(n, CHUNK_ROWS), runs beyond it add in int64."""

BLOCK_COLS = 256
"""Columns per block of the span's cached totals."""

MAX_ROWS = 1 << 24
"""The most rows a span may have: the int64 sums stay exact up to here."""

_SAFE_EXPONENT = 400
"""Column exponents within ±this of β scale a Gram by two multiplications, every intermediate a normal float."""


class RawSums(NamedTuple):
    """Sums over a subset of a span's rows, about the span's shifts (`SpanSums.over`).

    `u` = Σ(x - c), `S` = Σ(x - c)(x - c)', `v` = Σ(x - c)(y - c_y), `sy` = Σ(y - c_y), `syy` = Σ(y - c_y)², over `n` rows,
    `shift` holds c for the requested columns, in their order, and `yshift` c_y.
    """

    n: int
    shift: np.ndarray
    yshift: float
    u: np.ndarray
    S: np.ndarray
    v: np.ndarray
    sy: float
    syy: float


def slice_bits(n: int) -> int:
    """β for a span of `n` rows: floor((53 - ceil(log2 min(n, CHUNK_ROWS))) / 2)."""
    m = min(max(int(n), 1), CHUNK_ROWS)
    return (53 - (m - 1).bit_length()) // 2


def pair_rows(beta: int) -> int:
    """The most rows over which P'P (P = Q1 + Q2, |P| <= 1.5·2^β) sums exactly: floor(2^(53-2β)·4/9); 0 for none."""
    return (4 << (53 - 2 * beta)) // 9


def rebase_margin(beta: int) -> int:
    """δ = floor((3β - 54) / 2): how many binades a column's largest |x - c| over the requested rows may fall below its
    span's before the column is sliced again for that request (then 2^(1-3β+2δ) <= 2^-53, see the module)."""
    return (3 * beta - 54) // 2


# --- shifts and slices ------------------------------------------------------------------------------------------------


def _shifts(xt: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """Per row of `xt` (columns × rows): the `math.fsum` mean of its finite values, 0.0 for none, the value if all are equal,
    NaN when their sum overflows float64 (where `math.fsum` raises).

    The sum is formed exactly where two integer slices of W = 53 - ceil(log2 n) bits hold every value: t = v·2^(2W - E)
    (|v| < 2^E) must be an integer, which makes t = 2^W·T + R with integers |T| <= 2^W, |R| <= 2^(W-1) whose column sums
    stay under 2^53 and are exact; their total, an exact Python integer times 2^(E - 2W), is rounded once by CPython's correctly
    rounded int-to-float conversion or division, as `math.fsum` rounds. Any other column goes to `math.fsum`.
    """
    p, n = xt.shape
    out = np.zeros(p)
    if n == 0 or p == 0:
        return out
    if finite.all():
        count = np.full(p, n)
        lo, hi, v = xt.min(axis=1), xt.max(axis=1), xt
    else:
        count = finite.sum(axis=1)
        lo = np.min(xt, axis=1, where=finite, initial=np.inf)
        hi = np.max(xt, axis=1, where=finite, initial=-np.inf)
        v = np.where(finite, xt, 0.0)
    width = 53 - (n - 1).bit_length()
    varies = (count > 0) & (lo < hi)
    largest = np.where(varies, np.maximum(hi, -lo), 0.0)
    k = 2 * width - np.frexp(largest)[1].astype(np.int64)
    fast = np.flatnonzero(varies & (np.abs(k) <= 1000))
    exact = np.zeros(p, dtype=bool)
    if fast.size:
        vf = v if fast.size == p else v[fast]
        t = vf * np.ldexp(1.0, k[fast])[:, None]
        top = t * float(2.0**-width)
        np.rint(top, out=top)
        low = top * float(2.0**width)
        np.subtract(t, low, out=low)
        held = (np.rint(low) == low).all(axis=1)  # t is an integer exactly when its low part is
        scaled_down = k[fast] < 0
        if scaled_down.any():
            lost = (t[scaled_down] == 0) & (vf[scaled_down] != 0)
            held[scaled_down] &= ~lost.any(axis=1)
        s_top, s_low = top.sum(axis=1), low.sum(axis=1)
        for i in np.flatnonzero(held).tolist():
            j = int(fast[i])
            num = int(s_top[i]) * (1 << width) + int(s_low[i])
            kk = int(k[j])
            exact[j] = True
            try:
                total = float(num << -kk) if kk <= 0 else num / (1 << kk)
            except OverflowError:  # the sum is past float64's range: this column is unusable, the others go on
                out[j] = np.nan
                continue
            out[j] = total / int(count[j])
    for j in np.flatnonzero(varies & ~exact).tolist():
        values = xt[j][finite[j]]
        try:
            out[j] = math.fsum(memoryview(np.ascontiguousarray(values))) / values.shape[0]
        except OverflowError:
            out[j] = np.nan
    equal = (count > 0) & (lo == hi)
    out[equal] = lo[equal] + 0.0  # -0.0 becomes 0.0, whichever zero a reduction met first
    return out


def _slice_into(xt: np.ndarray, beta: int, q: np.ndarray, shift=None):
    """Shifts, exponents, non-finite, large-value and unusable masks of columns given as rows of `xt`; slices into `q`.

    `shift` given (a re-slice for a request) skips the shift. A value is large when |d| >= 2^(e - δ - 1): a request whose rows
    hold no large value of a column has that column's largest |d| more than δ binades under 2^e. A column is unusable when its
    sum or some x - c overflows float64; its slices are 0 and only a request that includes it raises, so no column's sums
    depend on whether such a column shares its block.
    """
    finite = np.isfinite(xt)
    if shift is None:
        shift = _shifts(xt, finite)
    shift = np.asarray(shift, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        d = xt - shift[:, None]
    bad = ~finite
    if bad.any():
        d[bad] = 0.0
    largest = np.abs(d).max(axis=1) if xt.shape[1] else np.zeros(xt.shape[0])
    unusable = ~(np.isfinite(largest) & np.isfinite(shift))
    if unusable.any():
        d[unusable] = 0.0
        largest[unusable] = 0.0
    exponent = np.frexp(largest)[1].astype(np.int64)
    big = np.abs(d) >= np.ldexp(1.0, exponent - rebase_margin(beta) - 1)[:, None]
    k = beta - exponent
    if k.size and np.abs(k).max() <= 1000:
        d *= np.ldexp(1.0, k)[:, None]
    else:
        d = np.ldexp(d, k[:, None])
    step = float(2.0**beta)
    np.rint(d, out=q[0])
    d -= q[0]
    d *= step
    np.rint(d, out=q[1])
    d -= q[1]
    d *= step
    np.rint(d, out=q[2])
    return shift, exponent, bad, big, unusable


# --- integer sums of slice products -----------------------------------------------------------------------------------


def _pieces(lo: int, hi: int, size: int):
    """[lo, hi) cut into ceil((hi - lo) / size) consecutive pieces of nearly equal length (at most `size` each)."""
    k = max(1, -(-(hi - lo) // size))
    edges = [lo + ((hi - lo) * i) // k for i in range(k + 1)]
    return list(zip(edges[:-1], edges[1:]))


def _gram_terms(a: np.ndarray, b, chunk: int, pair_chunk: int) -> np.ndarray:
    """(T2, T3, T4) as int64 between the columns of slices `a` and `b` (3 × columns × rows each), `b` None for `a` with itself.

    Rows run in pieces of at most `CHUNK_ROWS`, within which Q1'Q1, Q2'Q2 and Q1'Q3 are summed in float64 over chunks of at
    most `chunk` rows and P'P in int64 over chunks of at most `pair_chunk` (all exact, see the module); runs add in int64.
    """
    sym = b is None
    if sym:
        b = a
    p, q, r = a.shape[1], b.shape[1], a.shape[2]
    out = np.zeros((3, p, q), dtype=np.int64)
    for lo in range(0, r, CHUNK_ROWS):
        hi = min(lo + CHUNK_ROWS, r)
        f11, f22, f13, f12 = _float_terms(a, b, sym, _pieces(lo, hi, chunk), pair_chunk == 0)
        t2, t22 = f11.astype(np.int64), f22.astype(np.int64)
        if pair_chunk:
            t3 = np.zeros((p, q), dtype=np.int64)
            for c0, c1 in _pieces(lo, hi, min(chunk, pair_chunk)):
                pa = a[0, :, c0:c1] + a[1, :, c0:c1]
                pb = pa if sym else b[0, :, c0:c1] + b[1, :, c0:c1]
                t3 += (pa @ pb.T).astype(np.int64)
            t3 -= t2
            t3 -= t22
        else:
            t3 = f12.astype(np.int64)
        if sym:
            f13 += f13.T.copy()
        out[0] += t2
        out[1] += t3
        out[2] += f13.astype(np.int64)
        out[2] += t22
    return out


def _float_terms(a: np.ndarray, b: np.ndarray, sym: bool, pieces, with_x12: bool):
    """Q1'Q1, Q2'Q2, Q1'Q3 (plus Q3'Q1 off the diagonal) and, if asked, Q1'Q2 + Q2'Q1, summed in float64 over `pieces`."""
    f11 = f22 = f13 = f12 = None
    for c0, c1 in pieces:
        a1, a2, a3 = a[0, :, c0:c1], a[1, :, c0:c1], a[2, :, c0:c1]
        if sym:
            terms = [a1 @ a1.T, a2 @ a2.T, a1 @ a3.T]
            if with_x12:
                m12 = a1 @ a2.T
                terms.append(m12 + m12.T)
        else:
            b1, b2, b3 = b[0, :, c0:c1], b[1, :, c0:c1], b[2, :, c0:c1]
            x13 = a1 @ b3.T
            x13 += a3 @ b1.T
            terms = [a1 @ b1.T, a2 @ b2.T, x13]
            if with_x12:
                x12 = a1 @ b2.T
                x12 += a2 @ b1.T
                terms.append(x12)
        if f11 is None:
            f11, f22, f13 = terms[0], terms[1], terms[2]
            f12 = terms[3] if with_x12 else None
        else:
            f11 += terms[0]
            f22 += terms[1]
            f13 += terms[2]
            if with_x12:
                f12 += terms[3]
    if f11 is None:
        zero = np.zeros((a.shape[1], b.shape[1]))
        return zero, zero.copy(), zero.copy(), zero.copy()
    return f11, f22, f13, f12


def _vector_terms(a: np.ndarray, qy: np.ndarray, chunk: int) -> np.ndarray:
    """(T2, T3, T4) as int64 of each column of slices `a` (3 × columns × rows) against y's slices `qy` (3 × rows)."""
    t = np.zeros((3, a.shape[1]), dtype=np.int64)
    for lo in range(0, a.shape[2], chunk):
        prod = [(a[s, :, lo : lo + chunk] @ qy[:, lo : lo + chunk].T).astype(np.int64) for s in range(3)]
        t[0] += prod[0][:, 0]
        t[1] += prod[0][:, 1]
        t[1] += prod[1][:, 0]
        t[2] += prod[0][:, 2]
        t[2] += prod[2][:, 0]
        t[2] += prod[1][:, 1]
    return t


def _linear_terms(a: np.ndarray) -> np.ndarray:
    """(ΣQ1, ΣQ2, ΣQ3) per column of slices `a` (3 × columns × rows): float64 sums of integers under 2^53, exact."""
    if a.shape[2] == 0:
        return np.zeros((3, a.shape[1]), dtype=np.int64)
    return a.sum(axis=2).astype(np.int64)


# --- from integers to floats, in one fixed order ----------------------------------------------------------------------


def _combine(t2, t3, t4, beta: int) -> np.ndarray:
    """fl(fl(fl(t4)·2^-β + fl(t3))·2^-β + fl(t2)), elementwise: the one order in which every sum is formed."""
    inv = float(2.0**-beta)
    w = np.array(t4, dtype=np.float64)
    w *= inv
    w += np.asarray(t3, dtype=np.float64)
    w *= inv
    w += np.asarray(t2, dtype=np.float64)
    return w


def _scale_gram(w: np.ndarray, k: np.ndarray) -> np.ndarray:
    """w_jk·2^(k_j + k_k), exactly: two multiplications by normal powers of two when every |k| is small, else `ldexp`.

    With |k| <= `_SAFE_EXPONENT` and w either 0 or between 2^-52 and 2^63 in magnitude, every intermediate is a normal float,
    so both multiplications are exact and the result is `ldexp`'s.
    """
    if k.size == 0:
        return w
    if np.abs(k).max() <= _SAFE_EXPONENT:
        f = np.ldexp(1.0, k)
        w *= f[:, None]
        w *= f[None, :]
        return w
    return np.ldexp(w, k[:, None] + k[None, :])


class SpanSums:
    """Exact sums over one span's rows (`x`: rows × union columns, NaN where missing; `y`: the target), see the module.

    `x` and `y` are read, not copied, and must not change while this object is used; it is not thread-safe. `block_cols` and
    `chunk_rows` move no bit (the tests vary them); `chunk_rows` may not exceed `CHUNK_ROWS`.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, *, block_cols: int = BLOCK_COLS, chunk_rows: int = CHUNK_ROWS):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError(f"x must be (n, p) and y (n,); got {x.shape} and {y.shape}")
        if x.shape[0] > MAX_ROWS:
            raise ValueError(f"a span of {x.shape[0]} rows is over the {MAX_ROWS} exact sums allow")
        if not 1 <= chunk_rows <= CHUNK_ROWS:
            raise ValueError(f"chunk_rows must be in [1, {CHUNK_ROWS}]")
        if block_cols < 1:
            raise ValueError("block_cols must be positive")
        self._x = x
        self.n, self.p = x.shape
        self.beta = slice_bits(self.n)
        self._chunk = int(chunk_rows)
        self._pair_chunk = pair_rows(self.beta)
        self._block = int(block_cols)
        n_blocks = -(-self.p // self._block)
        self._ready = np.zeros(n_blocks, dtype=bool)
        self._pair_done = np.zeros((n_blocks, n_blocks), dtype=bool)
        self._shift = np.zeros(self.p)
        self._exponent = np.zeros(self.p, dtype=np.int64)
        self._q = np.zeros((3, self.p, self.n))
        self._bad = np.zeros((self.p, self.n), dtype=bool)
        self._n_bad = np.zeros(self.p, dtype=np.int64)
        self._big = np.zeros((self.p, self.n), dtype=bool)
        self._n_big = np.zeros(self.p, dtype=np.int64)
        self._unusable = np.zeros(self.p, dtype=bool)
        self._lin = np.zeros((3, self.p), dtype=np.int64)
        self._vec = np.zeros((3, self.p), dtype=np.int64)
        self._gram = np.zeros((3, self.p, self.p), dtype=np.int64)
        self._y = y
        qy = np.zeros((3, 1, self.n))
        yshift, yexp, ybad, ybig, yunusable = _slice_into(y[None, :], self.beta, qy)
        self._y_unusable = bool(yunusable[0])
        self.yshift = float(yshift[0])
        self._yexp = int(yexp[0])
        self._qy = np.ascontiguousarray(qy[:, 0, :])
        self._ybad = ybad[0]
        self._ybig = ybig[0]
        self._ylin = _linear_terms(qy)[:, 0]
        self._yy = _gram_terms(qy, None, self._chunk, self._pair_chunk)[:, 0, 0]

    # --- per-block state, computed on demand --------------------------------------------------------------------------

    def _cols_of(self, b: int) -> slice:
        return slice(b * self._block, min((b + 1) * self._block, self.p))

    def _prepare(self, blocks) -> None:
        for b in blocks:
            if self._ready[b]:
                continue
            cols = self._cols_of(b)
            q = self._q[:, cols]
            shift, exponent, bad, big, unusable = _slice_into(np.ascontiguousarray(self._x[:, cols].T), self.beta, q)
            self._unusable[cols] = unusable
            self._shift[cols] = shift
            self._exponent[cols] = exponent
            self._bad[cols] = bad
            self._n_bad[cols] = bad.sum(axis=1)
            self._big[cols] = big
            self._n_big[cols] = big.sum(axis=1)
            self._lin[:, cols] = _linear_terms(q)
            self._vec[:, cols] = _vector_terms(q, self._qy, self._chunk)
            self._ready[b] = True

    def _totals(self, blocks) -> None:
        """The span's totals for every pair of `blocks` (sorted, distinct), computing the missing ones."""
        if not blocks or self._pair_done[np.ix_(blocks, blocks)].all():
            return
        fresh = not self._pair_done[np.ix_(blocks, blocks)].any()
        if fresh and blocks == list(range(blocks[0], blocks[-1] + 1)):
            cols = slice(self._cols_of(blocks[0]).start, self._cols_of(blocks[-1]).stop)
            self._gram[:, cols, cols] = _gram_terms(self._q[:, cols], None, self._chunk, self._pair_chunk)
            self._pair_done[np.ix_(blocks, blocks)] = True
            return
        for i in blocks:
            for j in blocks:
                if j < i or self._pair_done[i, j]:
                    continue
                ci, cj = self._cols_of(i), self._cols_of(j)
                if i == j:
                    self._gram[:, ci, ci] = _gram_terms(self._q[:, ci], None, self._chunk, self._pair_chunk)
                else:
                    t = _gram_terms(self._q[:, ci], self._q[:, cj], self._chunk, self._pair_chunk)
                    self._gram[:, ci, cj] = t
                    self._gram[:, cj, ci] = t.transpose(0, 2, 1)
                self._pair_done[i, j] = self._pair_done[j, i] = True

    # --- the one public query -----------------------------------------------------------------------------------------

    def over(self, rows, cols, *, method: str = "auto") -> RawSums:
        """Sums over the span's `rows` (a boolean mask of length n, or distinct indices) for the columns `cols` (indices).

        Every value of `cols` and of y in those rows must be finite (callers filter first), else ValueError. `method` "auto"
        sums the removed rows and subtracts them from the span's totals when they are fewer than the kept ones; "direct" and
        "downdate" force one way. All three give the same bits.
        """
        rows = self._rows(rows)
        cols = np.asarray(cols, dtype=np.int64).reshape(-1)
        if cols.size and (cols.min() < 0 or cols.max() >= self.p):
            raise ValueError(f"column indices must be in [0, {self.p})")
        if method not in ("auto", "direct", "downdate"):
            raise ValueError(f"method must be 'auto', 'direct' or 'downdate', got {method!r}")
        blocks = np.unique(cols // self._block).tolist()
        self._prepare(blocks)
        self._check_finite(rows, cols)
        removed = np.setdiff1d(np.arange(self.n), rows, assume_unique=True)
        fewer = removed.size < rows.size
        if method == "auto":
            method = "downdate" if fewer else "direct"
        rebase, y_rebase = self._rebased(rows, removed, cols, fewer)
        e, ey = self._exponent[cols].copy(), self._yexp
        if method == "direct" or rebase.all():
            a, qy, e, ey = self._rows_slices(rows, cols, rebase, y_rebase)
            gram, lin, vec, ylin, yy = self._terms(a, qy)
            return self._raw(rows.size, cols, gram, lin, vec, ylin, yy, e, ey)
        keep = np.flatnonzero(~rebase)
        kept = cols[keep]
        self._totals(np.unique(kept // self._block).tolist())
        run = _as_run(kept)
        gram = self._gram[:, run, run] if run is not None else self._gram[:, kept[:, None], kept[None, :]]
        lin, vec, ylin, yy = self._lin[:, kept], self._vec[:, kept], self._ylin, self._yy
        if removed.size:
            g_r, lin_r, vec_r, ylin_r, yy_r = self._terms(
                self._q[:, kept[:, None], removed[None, :]], self._qy[:, removed]
            )
            gram, lin, vec, ylin, yy = gram - g_r, lin - lin_r, vec - vec_r, ylin - ylin_r, yy - yy_r
        if not rebase.any() and not y_rebase:
            return self._raw(rows.size, cols, gram, lin, vec, ylin, yy, e, ey)
        # Columns sliced again for these rows (and y, if so): their sums with every requested column, over the rows directly.
        a, qy, e, ey = self._rows_slices(rows, cols, rebase, y_rebase)
        moved = np.flatnonzero(rebase)
        full = np.zeros((3, cols.size, cols.size), dtype=np.int64)
        full[:, keep[:, None], keep[None, :]] = gram
        if moved.size:
            cross = _gram_terms(a[:, moved], a, self._chunk, self._pair_chunk)
            full[:, moved, :] = cross
            full[:, :, moved] = cross.transpose(0, 2, 1)
        line = np.zeros((3, cols.size), dtype=np.int64)
        line[:, keep] = lin
        line[:, moved] = _linear_terms(a[:, moved])
        if y_rebase:
            vec = _vector_terms(a, qy, self._chunk)
            ylin = _linear_terms(qy[:, None, :])[:, 0]
            yy = _gram_terms(qy[:, None, :], None, self._chunk, self._pair_chunk)[:, 0, 0]
        else:
            vec_all = np.zeros((3, cols.size), dtype=np.int64)
            vec_all[:, keep] = vec
            vec_all[:, moved] = _vector_terms(a[:, moved], qy, self._chunk)
            vec = vec_all
        return self._raw(rows.size, cols, full, line, vec, ylin, yy, e, ey)

    def _rebased(self, rows: np.ndarray, removed: np.ndarray, cols: np.ndarray, fewer: bool):
        """Which requested columns (and whether y) have no large value in `rows`, by an exact count either way round.

        A column with no large value in the whole span is 0 there (finite part) and needs nothing.
        """
        if fewer:
            big = self._n_big[cols] - self._big[cols[:, None], removed[None, :]].sum(axis=1)
            ybig = int(self._ybig.sum()) - int(self._ybig[removed].sum())
        else:
            big = self._big[cols[:, None], rows[None, :]].sum(axis=1)
            ybig = int(self._ybig[rows].sum())
        return (big == 0) & (self._n_big[cols] > 0), ybig == 0 and bool(self._ybig.any())

    def _rows_slices(self, rows: np.ndarray, cols: np.ndarray, rebase: np.ndarray, y_rebase: bool):
        """The requested columns' and y's slices on `rows`, those to rebase sliced again with these rows' own exponent."""
        a = self._q[:, cols[:, None], rows[None, :]]
        qy = self._qy[:, rows]
        e, ey = self._exponent[cols].copy(), self._yexp
        for i in np.flatnonzero(rebase).tolist():
            j = int(cols[i])
            part = np.zeros((3, 1, rows.size))
            _, exponent, _, _, _ = _slice_into(self._x[rows, j][None, :], self.beta, part, shift=self._shift[j : j + 1])
            a[:, i, :] = part[:, 0, :]
            e[i] = exponent[0]
        if y_rebase:
            part = np.zeros((3, 1, rows.size))
            _, exponent, _, _, _ = _slice_into(self._y[rows][None, :], self.beta, part, shift=[self.yshift])
            qy = np.ascontiguousarray(part[:, 0, :])
            ey = int(exponent[0])
        return a, qy, e, ey

    def _rows(self, rows) -> np.ndarray:
        rows = np.asarray(rows)
        if rows.dtype == bool:
            if rows.shape != (self.n,):
                raise ValueError(f"a row mask must have length {self.n}")
            return np.flatnonzero(rows)
        if rows.size == 0:
            return np.zeros(0, dtype=np.int64)
        if not np.issubdtype(rows.dtype, np.integer):
            raise ValueError("rows must be a boolean mask or integer indices")
        rows = rows.astype(np.int64).reshape(-1)
        if rows.min() < 0 or rows.max() >= self.n:
            raise ValueError(f"row indices must be in [0, {self.n})")
        unique = np.unique(rows)
        if unique.size != rows.size:
            raise ValueError("row indices must be distinct")
        return unique

    def _check_finite(self, rows: np.ndarray, cols: np.ndarray) -> None:
        too_large = np.unique(cols[self._unusable[cols]])
        if too_large.size:
            raise ValueError(
                f"column(s) {too_large.tolist()} of x have values too large for exact sums (their sum or x - shift "
                "overflows float64)"
            )
        if self._y_unusable:
            raise ValueError("y has values too large for exact sums (its sum or y - shift overflows float64)")
        if self._ybad[rows].any():
            raise ValueError("y has a NaN or inf in the requested rows")
        gappy = np.unique(cols[self._n_bad[cols] > 0])
        if gappy.size and self._bad[gappy[:, None], rows[None, :]].any():
            raise ValueError("x has a NaN or inf in the requested rows and columns")

    def _terms(self, a: np.ndarray, qy: np.ndarray):
        """The integer sums of slices `a` (3 × columns × rows) and y's `qy` (3 × rows): Gram, linear, against y, y's."""
        return (
            _gram_terms(a, None, self._chunk, self._pair_chunk),
            _linear_terms(a),
            _vector_terms(a, qy, self._chunk),
            _linear_terms(qy[:, None, :])[:, 0],
            _gram_terms(qy[:, None, :], None, self._chunk, self._pair_chunk)[:, 0, 0],
        )

    def _raw(self, n: int, cols: np.ndarray, gram, lin, vec, ylin, yy, e, ey) -> RawSums:
        beta = self.beta
        return RawSums(
            n=int(n),
            shift=self._shift[cols].copy(),
            yshift=self.yshift,
            u=np.ldexp(_combine(lin[0], lin[1], lin[2], beta), e - beta),
            S=_scale_gram(_combine(gram[0], gram[1], gram[2], beta), e - beta),
            v=np.ldexp(_combine(vec[0], vec[1], vec[2], beta), e + ey - 2 * beta),
            sy=float(np.ldexp(_combine(ylin[0], ylin[1], ylin[2], beta), ey - beta)),
            syy=float(np.ldexp(_combine(yy[0], yy[1], yy[2], beta), 2 * ey - 2 * beta)),
        )


def _as_run(cols: np.ndarray):
    """`cols` as a slice when it is one ascending run of consecutive indices (a cheaper gather), else None."""
    if cols.size and cols[-1] - cols[0] == cols.size - 1 and (cols.size == 1 or (np.diff(cols) == 1).all()):
        return slice(int(cols[0]), int(cols[-1]) + 1)
    return None
