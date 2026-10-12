"""
Reproducibility checks for "Discounted Interaction-Hitting Domination on Hypergraphs":
sharp recovery windows (rank-sensitive), the staircase on regular hypergraphs,
the codegree bound, and the quantitative projection separation.

Exact checks (rational arithmetic, fractions.Fraction): every equilibrium,
exhaustive feasibility searches, extremal support values, the staircase and
cubic characterizations, and all rational identities and inequalities.

Numerical checks (double precision): only the logarithmic analytic bounds
q^2 ln(1/R)/(q + ln(1/R)) <= delta(K_n^(q+1)) <= ceil((q+1) ln(1/R)) in check [6]
use math.log, compared with a tolerance of 1e-12.

Random instances use fixed seeds (random.Random(11), and seeds for networkx
drawn from it), so every run is identical.

Requirements: see requirements.txt (networkx).
Run: python3 verify_recovery.py
"""

from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction as Fr

import networkx as nx


# ---------------------------------------------------------------- equilibrium

def support(n, edges, S, lam):
    """Exact solution of the pinned equilibrium (eq:equilibrium)."""
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


def feasible(n, E, S, lam, tau):
    return min(support(n, E, S, lam)) >= tau


def dint(n, E, lam, tau, smax=None):
    for s in range(1, (smax or n) + 1):
        for S in itertools.combinations(range(n), s):
            if feasible(n, E, S, lam, tau):
                return s
    return None


# ---------------------------------------------------------------- thresholds

def theta(D, lam):
    return lam * (D - 1) / (D - lam)


def U1(D, lam):
    return lam / (D - (D - 1) * lam)


def Y(D, rho, lam):
    return lam * (D - 1) * (rho - 1) / (D * (rho - 1) - lam * (rho - 2 + lam))


def sigma(D, rho, lam):
    return lam * Y(D, rho, lam)


def vartheta(D, t, lam):
    return lam * (D - t) / (D - lam * t)


def ell(D, t, lam):
    return lam * (D - t - 1 + (t + 1) * theta(D, lam)) / D


def Rq(lam, tau):
    return (lam - tau) / (lam * (1 - tau))


# ---------------------------------------------------------------- helpers

def degrees(n, E):
    return [sum(1 for e in E if v in e) for v in range(n)]


def distances(n, E, S):
    adj = [set() for _ in range(n)]
    for e in E:
        for u in e:
            adj[u] |= set(e) - {u}
    d = [math.inf] * n
    frontier = list(S)
    for s in S:
        d[s] = 0
    while frontier:
        nxt = []
        for u in frontier:
            for w in adj[u]:
                if d[w] == math.inf:
                    d[w] = d[u] + 1
                    nxt.append(w)
        frontier = nxt
    return d


def is_linear(E):
    return all(len(set(a) & set(b)) <= 1 for a, b in itertools.combinations(E, 2))


def is_transversal(E, S):
    return all(set(e) & set(S) for e in E)


def random_hypergraph(rng, n, D, rho, m):
    deg = [0] * n
    E = []
    for _ in range(10 * m):
        if len(E) >= m:
            break
        k = rng.randint(2, rho)
        cand = [v for v in range(n) if deg[v] < D]
        if len(cand) < k:
            break
        e = sorted(rng.sample(cand, k))
        if e in E:
            continue
        E.append(e)
        for v in e:
            deg[v] += 1
    used = sorted({v for e in E for v in e})
    idx = {v: i for i, v in enumerate(used)}
    return len(used), [[idx[v] for v in e] for e in E]


def random_regular_uniform(rng, n, D, k, tries=2000):
    """Random D-regular k-uniform simple hypergraph via the configuration model."""
    assert (n * D) % k == 0
    for _ in range(tries):
        pts = [v for v in range(n) for _ in range(D)]
        rng.shuffle(pts)
        E = [sorted(pts[i:i + k]) for i in range(0, len(pts), k)]
        if all(len(set(e)) == k for e in E) and len({tuple(e) for e in E}) == len(E):
            return E
    return None


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------- constructions

def padded_gadget(D, rho):
    """Linear rho-uniform hypergraph of maximum hyperdegree D attaining sigma(D,rho) (Thm dom-sharp)."""
    us = list(range(rho - 1))
    v = rho - 1
    E = [us + [v]]
    nxt, W = rho, []
    for i in us:
        for _ in range(D - 1):
            w = nxt
            pad = list(range(nxt + 1, nxt + rho - 1))
            nxt += rho - 1
            E.append([i, w] + pad)
            W.append(w)
    return nxt, E, W, v


def grid_witness(D, rho):
    """Linear rho-uniform hypergraph, max hyperdegree D, dominating set with all supports U1."""
    pts = list(itertools.product(range(rho), repeat=D - 1))
    idx = {}
    n = 0
    for c in range(rho - 1):
        for x in pts:
            idx[(c, x)] = n
            n += 1
    E = []
    for c in range(rho - 1):
        for axis in range(D - 1):
            seen = set()
            for x in pts:
                key = x[:axis] + x[axis + 1:]
                if key in seen:
                    continue
                seen.add(key)
                E.append(sorted(idx[(c, x[:axis] + (a,) + x[axis + 1:])] for a in range(rho)))
    S = []
    for x in pts:
        s = n
        n += 1
        S.append(s)
        E.append(sorted([s] + [idx[(c, x)] for c in range(rho - 1)]))
    return n, E, S


def regular_witness(D, t, rho):
    """D-regular hypergraph H_{D,t,rho} of the staircase theorem, single source w."""
    w, v = 0, 1
    U = [[2 + i * (rho - 1) + j for j in range(rho - 1)] for i in range(t + 1)]
    allu = [u for blk in U for u in blk]
    n = 2 + len(allu)
    E = [[v] + blk for blk in U]
    jmap = {u: (k % D) for k, u in enumerate(allu)}
    for j in range(D):
        f = [w] + [u for u in allu if jmap[u] != j] + ([v] if j < D - t - 1 else [])
        E.append(sorted(f))
    return n, E, [w], v


def projective_plane_prime(q):
    """PG(2,q) by arithmetic modulo q. Valid ONLY for prime q (the suite uses q = 2, 3);
    for a non-prime prime power such as q = 4, integers mod q do not form a field."""
    assert q in (2, 3, 5, 7, 11, 13), "projective_plane_prime requires a prime q"
    pts = [(x, y, 1) for x in range(q) for y in range(q)] + [(x, 1, 0) for x in range(q)] + [(1, 0, 0)]
    idx = {p: i for i, p in enumerate(pts)}
    return len(pts), [[idx[p] for p in pts if (L[0] * p[0] + L[1] * p[1] + L[2] * p[2]) % q == 0] for L in pts]


# ---------------------------------------------------------------- checks

def main():
    rng = random.Random(11)
    lams = [Fr(1, 4), Fr(1, 2), Fr(3, 5), Fr(4, 5), Fr(9, 10)]

    print("[1] universal bounds on random hypergraphs (Lemmas sourcefree, distant, dominated; Prop codegree)")
    cnt = 0
    for _ in range(500):
        D0, r0 = rng.randint(2, 4), rng.randint(2, 4)
        lam = rng.choice(lams)
        n, E = random_hypergraph(rng, rng.randint(5, 9), D0, r0, rng.randint(3, 8))
        if n < 3:
            continue
        deg = degrees(n, E)
        D, rho = max(deg), max(len(e) for e in E)
        S = rng.sample(range(n), rng.randint(1, max(1, n // 2)))
        h = support(n, E, S, lam)
        dist = distances(n, E, S)
        for e in E:
            if not set(e) & set(S):
                require(all(h[v] <= theta(D, lam) for v in e), "source-free bound")
        for v in range(n):
            if 2 <= dist[v]:
                bound = sigma(D, rho, lam) * lam ** (dist[v] - 2) if dist[v] < math.inf else 0
                require(h[v] <= bound, "rank-sensitive distant bound")
                require(D < 2 or sigma(D, rho, lam) < lam * theta(D, lam), "sigma < lam*theta")
        if all(x <= 1 for x in dist):
            require(min(h) >= U1(D, lam), "dominated bound")
        if n <= 8 and cnt < 120:
            cnt += 1
            tau = rng.choice([lam / 2, lam * 3 / 4, lam])
            d0 = min(deg)
            mu = max(sum(1 for e in E if a in e and b in e) for a, b in itertools.combinations(range(n), 2))
            dv = dint(n, E, lam, tau)
            require(dv >= min(n, (1 - Rq(lam, tau)) * d0 / mu), "codegree bound")
    print("   PASS")

    print("[2] extremal constructions")
    for D in (2, 3, 4):
        for lam in (Fr(1, 3), Fr(3, 4)):
            # K_{D+1}: D-t sources give support vartheta_t; D-1 sources give theta and a non-transversal
            K = [[a, b] for a, b in itertools.combinations(range(D + 1), 2)]
            for t in range(D):
                h = support(D + 1, K, list(range(D - t)), lam)
                require(min(h) == vartheta(D, t, lam), "K_{D+1} attains vartheta_t")
            require(vartheta(D, 1, lam) == theta(D, lam) and vartheta(D, D - 1, lam) == U1(D, lam), "endpoints")
            require(not is_transversal(K, range(D - 1)), "Delta-1 vertices do not cover K_{D+1}")
            require(min(len(T) for r in range(1, D + 2) for T in itertools.combinations(range(D + 1), r)
                        if is_transversal(K, T)) == D, "tr(K_{D+1}) = Delta")
            for rho in (2, 3, 4):
                n, E, W, v = padded_gadget(D, rho)
                require(is_linear(E) and all(len(e) == rho for e in E) and max(degrees(n, E)) == D, "gadget class")
                h = support(n, E, W, lam)
                require(distances(n, E, W)[v] == 2, "gadget: v undominated")
                require(min(h) == h[v] == sigma(D, rho, lam), "gadget attains sigma")
                n, E, S = grid_witness(D, rho)
                require(is_linear(E) and all(len(e) == rho for e in E) and max(degrees(n, E)) == D, "grid class")
                require(max(distances(n, E, S)) <= 1, "grid: dominating")
                require(min(support(n, E, S, lam)) == U1(D, lam), "grid attains U1")
            prev = Fr(0)
            for rho in (3, 6, 12, 24):
                require(prev < sigma(D, rho, lam) < lam * theta(D, lam), "sigma increases to lam*theta")
                prev = sigma(D, rho, lam)
    print("   PASS")

    print("[3] staircase: regular witnesses H_{D,t,rho}")
    for D in (2, 3, 4):
        for t in range(1, D):
            for lam in (Fr(1, 3), Fr(3, 4)):
                prev = Fr(0)
                for rho in (3, 5, 8):
                    if (t + 1) * (rho - 1) < D:
                        continue
                    n, E, S, v = regular_witness(D, t, rho)
                    require(set(degrees(n, E)) == {D}, "regular")
                    require(len({tuple(e) for e in E}) == len(E), "simple")
                    h = support(n, E, S, lam)
                    sf = sum(1 for e in E if v in e and not set(e) & set(S))
                    require(sf == t + 1, "v lies in t+1 source-free hyperedges")
                    require(min(h) == h[v] and prev < h[v] < ell(D, t, lam), "approach ell_t from below")
                    prev = h[v]
    print("   PASS")

    print("[4] staircase characterization on random regular graphs and hypergraphs")
    trials = 0
    for _ in range(90):
        kind = rng.choice(["g3", "g4", "g5", "h33", "h43"])
        if kind[0] == "g":
            D = int(kind[1])
            n = rng.choice([8, 10]) if D != 5 else 8
            G = nx.random_regular_graph(D, n, seed=rng.randint(0, 10 ** 6))
            E = [sorted(e) for e in G.edges()]
        else:
            D, k = int(kind[1]), int(kind[2])
            n = 9 if D == 3 else 6
            E = random_regular_uniform(rng, n, D, k)
            if E is None:
                continue
        lam = rng.choice(lams)
        for t in range(D):
            if t >= 1 and lam * t * t > D:
                continue
            hi = vartheta(D, t, lam)
            lo = ell(D, t, lam)
            taus = [hi, (lo + hi) / 2] + ([lo] if t >= 1 else [lo + (hi - lo) / 1000])
            for tau in taus:
                for _ in range(4):
                    S = set(rng.sample(range(n), rng.randint(1, n - 1)))
                    cond = all(sum(1 for e in E if v in e and not set(e) & S) <= t for v in range(n) if v not in S)
                    require(cond == feasible(n, E, S, lam, tau), f"staircase {kind} t={t}")
                    trials += 1
    print(f"   PASS ({trials} tests)")

    print("[4b] cubic domination range (2lam^2/(3-lam^2), lam/(3-2lam)] on random cubic graphs")
    for _ in range(40):
        n = rng.choice([8, 10])
        G = nx.random_regular_graph(3, n, seed=rng.randint(0, 10 ** 6))
        E = [sorted(e) for e in G.edges()]
        lam = rng.choice(lams + [Fr(19, 20)])
        lo, hi = sigma(3, 2, lam), U1(3, lam)
        for tau in (hi, (lo + hi) / 2, lo + (hi - lo) / 1000):
            for _ in range(5):
                S = set(rng.sample(range(n), rng.randint(1, n - 1)))
                dom = all(v in S or any(u in S for u in G[v]) for v in range(n))
                require(dom == feasible(n, E, S, lam, tau), "cubic domination range")
    print("   PASS")

    print("[5] algebraic identities at random rational points")
    for _ in range(400):
        lam = Fr(rng.randint(1, 99), 100)
        D = rng.randint(2, 7)
        rho = rng.randint(2, 9)
        for t in range(D):
            lhs = vartheta(D, t, lam) - ell(D, t, lam)
            rhs = lam * (1 - lam) * (D - lam * t * t) / ((D - lam * t) * (D - lam))
            require(lhs == rhs, "nonemptiness identity")
            if t + 1 <= D - 1 and t >= 1:
                require(vartheta(D, t + 1, lam) < ell(D, t, lam), "gaps between steps")
        require(ell(D, 0, lam) == theta(D, lam) and ell(D, D - 1, lam) == lam * theta(D, lam), "end steps")
        Lam = Fr((rho - 1) * D, (rho - 1) * D * (D - 2) + rho - 2) if (rho - 1) * D * (D - 2) + rho - 2 > 0 else Fr(10)
        require((sigma(D, rho, lam) < U1(D, lam)) == (lam < Lam), "rank-rho window nonempty")
        require(ell(3, 1, lam) == lam * (1 + lam) / (3 - lam), "cubic step 1")
        require(U1(3, lam) < ell(3, 1, lam), "cubic gap nonempty")
        require(sigma(3, 2, lam) == 2 * lam * lam / (3 - lam * lam) < U1(3, lam), "cubic domination range")
    print("   PASS")

    print("[6] projective planes, complete graphs, complete uniform hypergraphs")
    for q in (2, 3):
        n, E = projective_plane_prime(q)
        require(is_linear(E) and set(degrees(n, E)) == {q + 1}, "plane")
        for lam, tau in [(Fr(3, 5), Fr(2, 5)), (Fr(9, 10), Fr(17, 20)), (Fr(9, 10), Fr(89, 100))]:
            R = Rq(lam, tau)
            dpg = dint(n, E, lam, tau, q + 1)
            require((1 - R) * (q + 1) <= dpg <= q + 1, "plane bounds")
            dK = min(n, math.ceil((1 - R) * (n - 1)))
            dU = next(s for s in range(1, n) if Fr(math.comb(n - 1 - s, q), math.comb(n - 1, q)) <= R)
            L = math.log(1 / R)
            require(q * q * L / (q + L) - 1e-12 <= dU <= max(1, math.ceil((q + 1) * L)), "uniform bounds")
            if q == 2:
                Kn = [[a, b] for a, b in itertools.combinations(range(n), 2)]
                require(dint(n, Kn, lam, tau) == dK, "complete graph formula")
            if q == 2 and (lam, tau) == (Fr(3, 5), Fr(2, 5)):
                require((dpg, dU) == (3, 2), "Fano example")
                h = support(n, E, [0, 1], lam)
                require(Fr(17, 43) in h, "Fano x = 17/43")
            print(f"   q={q} lam={lam} tau={tau}: plane {dpg} (>= {float((1 - R) * (q + 1)):.3f}), "
                  f"K_n {dK}, K_n^({q + 1}) {dU}")
    print("   PASS")
    print("All checks passed.")


if __name__ == "__main__":
    main()
