"""
Reproducibility checks for the section "Loose paths and cycles" of
"Discounted Interaction-Hitting Domination on Hypergraphs".

Exact checks (rational arithmetic, fractions.Fraction): all equilibria, the
reflection identity, feasibility criteria, domination numbers by exhaustive
search over source sets, fault-tolerant values by exhaustive search over
source and failure sets, transfer-matrix counts, and the displayed examples.

High-precision numerical check: the hyperbolic closed form (eq. phi-closed) is
evaluated with mpmath at 80 significant digits and compared with the exact
rational run minimum to within 1e-60.

Solver-based check: larger instances of the fault-tolerance formulas are
checked against an integer covering formulation, built from the window
criterion (Lemma "Window criterion"), solved by scipy.optimize.milp (a
numerical MILP solver, not a certified exact solver). The formulation is
itself validated against exhaustive search on the small instances.

Requirements: see requirements.txt (mpmath, numpy, scipy >= 1.9 for milp).
Run: python3 verify_loose.py   (fixed seed: random.Random(1))
"""

from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction as Fr

import mpmath
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


# ---------------------------------------------------------------------------
# Loose hypergraphs and exact equilibria
# ---------------------------------------------------------------------------

def loose_cycle(m, k):
    """Junctions are vertices 0..m-1; junction i lies in edges i and i+1 (mod m)."""
    edges, nxt = [], m
    for i in range(m):
        edges.append([i] + list(range(nxt, nxt + k - 2)) + [(i + 1) % m])
        nxt += k - 2
    return nxt, edges


def loose_path(m, k):
    """Edges e_1..e_m; junction j_i (vertex i-1) lies in e_i and e_{i+1}."""
    edges, nxt = [], m - 1
    for i in range(1, m + 1):
        e = []
        if i >= 2:
            e.append(i - 2)
        if i <= m - 1:
            e.append(i - 1)
        need = k - len(e)
        e += list(range(nxt, nxt + need))
        nxt += need
        edges.append(e)
    return nxt, edges


def support(n, edges, S, lam):
    """Exact interaction-hitting supports h^S (Fractions)."""
    S = set(S)
    if not S:
        return [Fr(0)] * n
    U = [v for v in range(n) if v not in S]
    inc = [[e for e in edges if v in e] for v in range(n)]
    pos = {u: i for i, u in enumerate(U)}
    N = len(U)
    M = [[Fr(0)] * (N + 1) for _ in range(N)]
    for i, v in enumerate(U):
        M[i][i] = Fr(1)
        d = len(inc[v])
        for e in inc[v]:
            if S.intersection(e):
                M[i][N] += lam / d
            else:
                for u in e:
                    if u != v:
                        M[i][pos[u]] -= lam / d / (len(e) - 1)
    for c in range(N):
        p = next(r for r in range(c, N) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        for r in range(N):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    h = [Fr(1)] * n
    for i, v in enumerate(U):
        h[v] = M[i][N]
    return h


def dominating(n, edges, S, lam, tau):
    return bool(S) and min(support(n, edges, S, lam)) >= tau


def fault_tolerant(n, edges, S, f, lam, tau):
    if len(S) < f + 1:
        return False
    return all(dominating(n, edges, [v for v in S if v not in F], lam, tau)
               for r in range(f + 1) for F in itertools.combinations(S, r))


def brute_opt(n, edges, f, lam, tau):
    for q in range(1, n + 1):
        for S in itertools.combinations(range(n), q):
            if fault_tolerant(n, edges, list(S), f, lam, tau):
                return q
    return math.inf


# ---------------------------------------------------------------------------
# Run profile and scales
# ---------------------------------------------------------------------------

def phi_exact(r, k, lam):
    """Minimum support on an interior run of r edges (exact)."""
    if r == 0:
        return lam
    m = r + 4
    n, E = loose_cycle(m, k)
    h = support(n, E, [0, r + 2], lam)  # verifies e_{m-1}, e_0 and e_{r+1}, e_{r+2}
    run = set().union(*[set(E[i]) for i in range(1, r + 1)])
    return min(h[v] for v in run)


def phi_closed(r, k, lam):
    """Closed form (eq. phi-closed), evaluated with mpmath at mp.dps = 80."""
    mp = mpmath.mp
    lam = mpmath.mpf(lam.numerator) / lam.denominator
    if r == 0:
        return lam
    lk = lam / (k - 1 - (k - 2) * lam)
    mu = mpmath.acosh(1 / lk)
    ck = lk / (1 + lk)
    kap = (k - 2) * ck
    C = mpmath.cosh((r + 2) * mu / 2)
    g = mpmath.cosh(r * mu / 2) / C
    v = (k - 1) / (1 + kap * (1 + g))
    if r % 2 == 0:
        eta = mp.mpf(1)
    else:
        eta = mpmath.cosh(mu / 2) * (1 if k == 2 else 2 * ck)
    return eta * v / C


def as_mpf(x):
    return mpmath.mpf(x.numerator) / x.denominator


def scales(k, lam, tau):
    r = 0
    while phi_exact(r + 1, k, lam) >= tau:
        r += 1
    L = r + 2
    return L, L // 2


# ---------------------------------------------------------------------------
# Formulas of the section
# ---------------------------------------------------------------------------

def delta_cycle(m, L):
    return -(-m // L)


def delta_path(m, L, B):
    return 1 + -(-max(0, m - 2 * B) // L)


def ft_cycle(m, k, L, f):
    if f + 1 > m * (k - 1):
        return math.inf
    if m <= L - 1:
        return f + 1
    if f > (L - 1) * (k - 1):
        return math.inf
    return max(-(-(f + 1) * m // L), -(-f * m // (L - 1)))


def ft_graph_path(n, L, f):
    B = L // 2
    N = n - 1
    if f > B:
        return math.inf
    if N <= 2 * B:
        return max(f + 1, n + 2 * f - 2 * B)
    w, H, R = f + 1, L - f - 1, N - 2 * B
    a = (R - 1) // L
    rho = R - a * L
    return 2 * f + 1 + a * w + max(1, rho - H)


# ---------------------------------------------------------------------------
# Covering programs from the window criterion
# ---------------------------------------------------------------------------

def cover_cycle(m, k, L, f):
    w, nv = L - 1, 2 * m
    caps = [1] * m + [k - 2] * m
    rows = []
    if m <= w:
        rows.append(np.ones(nv))
    else:
        for s in range(m):
            es = {(s + t) % m for t in range(w)}
            r = np.zeros(nv)
            for i in range(m):          # junction i lies in edges i and i+1
                if i in es or (i + 1) % m in es:
                    r[i] = 1
            for i in es:
                r[m + i] = 1
            rows.append(r)
    res = milp(c=np.ones(nv), constraints=LinearConstraint(np.array(rows), lb=f + 1, ub=np.inf),
               integrality=np.ones(nv), bounds=Bounds(0, caps))
    return round(res.fun) if res.status == 0 else math.inf


def cover_path(m, k, L, f):
    B, w = L // 2, L - 1
    nj = m - 1
    nv = nj + m
    caps = [1] * nj + [(k if m == 1 else (k - 1 if i in (1, m) else k - 2)) for i in range(1, m + 1)]

    def row(es):
        r = np.zeros(nv)
        es = set(es)
        for i in range(1, m):           # junction j_i in e_i and e_{i+1}
            if i in es or i + 1 in es:
                r[i - 1] = 1
        for i in es:
            r[nj + i - 1] = 1
        return r

    rows = [row(range(1, min(B, m) + 1)), row(range(max(1, m - B + 1), m + 1))]
    if m >= w:
        rows += [row(range(s, s + w)) for s in range(1, m - w + 2)]
    res = milp(c=np.ones(nv), constraints=LinearConstraint(np.array(rows), lb=f + 1, ub=np.inf),
               integrality=np.ones(nv), bounds=Bounds(0, caps))
    return round(res.fun) if res.status == 0 else math.inf


# ---------------------------------------------------------------------------
# Transfer-matrix count on loose cycles
# ---------------------------------------------------------------------------

def _pmul(a, b):
    c = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                c[i + j] += x * y
    return c


def _padd(a, b):
    if len(a) < len(b):
        a, b = b, a
    return [x + (b[i] if i < len(b) else 0) for i, x in enumerate(a)]


def nu_cycle_transfer(m, k, L):
    def poly(ds, w):
        p = [0] * (max(ds) + 1)
        for d in ds:
            p[d] += w
        return p
    T = [[poly(range(2, 2 * L + 1, 2), 1), poly(range(1, 2 * L, 2), k - 2)],
         [poly(range(1, 2 * L, 2), 1), poly(range(2, 2 * L - 1, 2), k - 2)]]
    q = -(-m // L)
    P = T
    for _ in range(q - 1):
        P = [[_padd(_pmul(P[i][0], T[0][j]), _pmul(P[i][1], T[1][j])) for j in range(2)] for i in range(2)]
    tr = sum(P[i][i][2 * m] if 2 * m < len(P[i][i]) else 0 for i in range(2))
    assert (m * tr) % q == 0
    return m * tr // q


def nu_cycle_direct(m, k, L, lam, tau):
    n, E = loose_cycle(m, k)
    q = -(-m // L)
    return sum(1 for S in itertools.combinations(range(n), q) if dominating(n, E, list(S), lam, tau))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main():
    mpmath.mp.dps = 80
    params = [(Fr(4, 5), Fr(1, 2)), (Fr(3, 5), Fr(9, 20)), (Fr(9, 10), Fr(37, 100)),
              (Fr(8, 9), Fr(1, 2)), (Fr(19, 20), Fr(31, 100))]

    print("[1] closed-form run profile and reflection identity")
    for k in (2, 3, 4):
        for lam in (Fr(1, 2), Fr(4, 5), Fr(8, 9)):
            for r in range(1, 7):
                require(abs(as_mpf(phi_exact(r, k, lam)) - phi_closed(r, k, lam)) < mpmath.mpf(10) ** -60,
                        f"profile k={k} lam={lam} r={r}")
            for a in range(1, 4):
                n, E = loose_path(a + 3, k)
                h = support(n, E, [a], lam)      # junction j_{a+1} verifies e_{a+1}, e_{a+2}
                run = set().union(*[set(E[i]) for i in range(a)])
                require(min(h[v] for v in run) == phi_exact(2 * a, k, lam),
                        f"reflection k={k} lam={lam} a={a}")
    print("   PASS")

    print("[2] feasibility criterion on random source sets")
    rng = random.Random(1)
    for k in (2, 3, 4):
        for lam, tau in params:
            L, B = scales(k, lam, tau)
            for _ in range(40):
                m = rng.randint(3, 7)
                cyc = rng.random() < 0.5
                n, E = loose_cycle(m, k) if cyc else loose_path(m, k)
                S = set(rng.sample(range(n), rng.randint(1, max(1, n // 3))))
                ver = [bool(S & set(e)) for e in E]
                if cyc:
                    i0 = ver.index(True)
                    runs, cur = [], 0
                    for t in range(1, m + 1):
                        if ver[(i0 + t) % m]:
                            runs.append(cur)
                            cur = 0
                        else:
                            cur += 1
                    pred = max(runs) <= L - 2
                else:
                    first, last = ver.index(True), m - 1 - ver[::-1].index(True)
                    inner, cur = [], 0
                    for i in range(first + 1, last + 1):
                        if ver[i]:
                            inner.append(cur)
                            cur = 0
                        else:
                            cur += 1
                    pred = first <= B - 1 and m - 1 - last <= B - 1 and all(x <= L - 2 for x in inner)
                require(pred == dominating(n, E, list(S), lam, tau), f"criterion k={k} m={m} S={S}")
    print("   PASS")

    print("[3] domination numbers and cycle fault tolerance by exhaustive search")
    for k, lam, tau in [(3, Fr(4, 5), Fr(1, 2)), (3, Fr(3, 5), Fr(9, 20)), (2, Fr(4, 5), Fr(1, 2)),
                        (4, Fr(9, 10), Fr(37, 100))]:
        L, B = scales(k, lam, tau)
        for m in range(3, 6):
            n, E = loose_cycle(m, k)
            if n > 12:
                continue
            for f in range(0, 3):
                if f < n:
                    require(brute_opt(n, E, f, lam, tau) == ft_cycle(m, k, L, f),
                            f"cycle k={k} m={m} f={f}")
        for m in range(1, 5):
            n, E = loose_path(m, k)
            if n > 12:
                continue
            require(brute_opt(n, E, 0, lam, tau) == delta_path(m, L, B), f"path k={k} m={m}")
            for f in range(1, 3):
                if f < n:
                    require(brute_opt(n, E, f, lam, tau) == cover_path(m, k, L, f),
                            f"path cover k={k} m={m} f={f}")
    print("   PASS")

    print("[4] transfer-matrix counts of minimum placements")
    for k, lam, tau in [(3, Fr(4, 5), Fr(1, 2)), (2, Fr(4, 5), Fr(1, 2)), (3, Fr(8, 9), Fr(1, 2))]:
        L, _ = scales(k, lam, tau)
        for m in range(3, 7):
            if loose_cycle(m, k)[0] <= 12:
                require(nu_cycle_direct(m, k, L, lam, tau) == nu_cycle_transfer(m, k, L),
                        f"count k={k} m={m}")
    print("   PASS")

    print("[5] larger orders against the covering programs")
    for k in (2, 3, 4):
        for L in range(2, 8):
            for m in range(3, 30):
                for f in range(0, 2 * L):
                    require(cover_cycle(m, k, L, f) == ft_cycle(m, k, L, f), f"cycle cover k={k} L={L} m={m} f={f}")
    for L in range(2, 9):
        for n in range(2, 40):
            for f in range(0, min(n, L + 2)):
                require(cover_path(n - 1, 2, L, f) == ft_graph_path(n, L, f), f"graph path L={L} n={n} f={f}")
                if n - 1 >= max(L, 3) and f <= (L // 2) * 2:
                    for k in (3, 4):
                        if f <= (L // 2) * (k - 1):
                            M = ft_cycle(n - 1, k, L, f)
                            v = cover_path(n - 1, k, L, f)
                            require(M - (f + 1) <= v <= M + 2 * f + 3, f"path bound k={k} L={L} m={n-1} f={f}")
    print("   PASS")

    print("[6] Example values and the P^(3)_10 example")
    lam = Fr(8, 9)
    require([phi_exact(r, 3, lam) for r in range(4)] == [Fr(8, 9), Fr(32, 47), Fr(16, 29), Fr(64, 167)], "phi values")
    require(scales(3, lam, Fr(1, 2)) == (4, 2), "scales k=3")
    require(scales(2, Fr(4, 5), Fr(1, 2)) == (3, 1), "scales graph")
    require(nu_cycle_transfer(12, 3, 4) == 4, "nu(C^3_12)")
    require([ft_cycle(12, 3, 4, f) for f in range(8)] == [3, 6, 9, 12, 16, 20, 24, math.inf], "FT values")
    require(scales(3, lam, Fr(1, 4)) == (6, 3), "scales tau=1/4")
    n, E = loose_path(10, 3)
    # sources: junctions j_2, j_3, j_7, j_8 (vertices 1, 2, 6, 7) and the free vertex of e_2, e_3, e_8, e_9
    free = lambda i: [v for v in E[i - 1] if v >= 9][0]
    S = [1, 2, 6, 7, free(2), free(3), free(8), free(9)]
    require(fault_tolerant(n, E, S, 3, lam, Fr(1, 4)), "8-source 3-fault-tolerant set")
    require(ft_graph_path(11, 6, 3) == 9, "graph-path expression gives 9")
    print("   PASS")
    print("All checks passed.")


if __name__ == "__main__":
    main()
