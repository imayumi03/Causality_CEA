"""
cri_ts.py — Time-series Causal Relevance Index (CRI_TS): article-faithful toolkit.

Implements, with numpy/scipy only:
  * stable VAR(p) simulation with exact per-(i, j, tau) ground truth;
  * the Fisher-z (partial) correlation CI test, naive or conditioned;
  * a PCMCI-style lagged discovery: PC1 condition selection + MCI step,
    restricted to lagged edges (tau >= 1), matching the scope of the paper;
  * the max-aggregated edge p-value  p''_e = max_{S tested} p_e(S)   (R2);
  * CRI_TS(G) = 1 - mean_e p''_e                                     (Def. 2.4);
  * closed-form calibration theory for null lagged edges between
    independent AR(1) processes:
        kappa(phi_i, phi_j) = (1 + phi_i phi_j) / (1 - phi_i phi_j)
        E[p | H0](kappa)    = (2/pi) * arctan(1/sqrt(kappa))
    used as overlays in the calibration experiments.

The tigramite backend (for the CausalChamber wind-tunnel section) is optional
and only needed in the notebook's final section.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# 1. Simulation: structural VAR with exact lag-indexed ground truth
# --------------------------------------------------------------------------


@dataclass
class VARSystem:
    """Structural VAR: X^j_t = sum_edges a * X^i_{t-tau} + phi_j X^j_{t-1} + eps."""

    d: int
    edges: list  # list of (i, j, tau, coef) with i != j, tau >= 1
    phi: np.ndarray  # self-AR(1) coefficient per variable (autocorrelation knob)
    tau_max: int
    noise_sd: float = 1.0

    def __post_init__(self):
        self.phi = np.asarray(self.phi, dtype=float)
        assert self.phi.shape == (self.d,)
        for (i, j, tau, a) in self.edges:
            assert i != j and 1 <= tau <= self.tau_max

    # -- ground truth ------------------------------------------------------
    def gt_lagged(self) -> set:
        """Exact per-(i, j, tau) ground-truth set of present lagged edges."""
        return {(i, j, tau) for (i, j, tau, _a) in self.edges}

    def gt_pairs(self) -> set:
        """Summary-graph (pair-level) ground truth."""
        return {(i, j) for (i, j, tau, _a) in self.edges}

    # -- simulation --------------------------------------------------------
    def simulate(self, T: int, rng: np.random.Generator, burn: int = 300) -> np.ndarray:
        n = T + burn
        X = np.zeros((n, self.d))
        eps = rng.normal(0.0, self.noise_sd, size=(n, self.d))
        for t in range(n):
            for j in range(self.d):
                v = eps[t, j]
                if t >= 1:
                    v += self.phi[j] * X[t - 1, j]
                for (i, jj, tau, a) in self.edges:
                    if jj == j and t >= tau:
                        v += a * X[t - tau, i]
                X[t, j] = v
        out = X[burn:]
        # guard against explosive systems
        if not np.isfinite(out).all() or np.abs(out).max() > 1e6:
            raise RuntimeError("VAR system appears unstable; reduce coefficients.")
        return out


def two_independent_ar1(phi_i: float, phi_j: float, T: int,
                        rng: np.random.Generator) -> np.ndarray:
    """d=2 system with NO cross edges: every lagged cross edge is null (H0)."""
    sys2 = VARSystem(d=2, edges=[], phi=np.array([phi_i, phi_j]), tau_max=1)
    return sys2.simulate(T, rng)


# --------------------------------------------------------------------------
# 2. CI test: Fisher-z partial correlation on lagged columns
# --------------------------------------------------------------------------


def _lagged_column(X: np.ndarray, var: int, lag: int, t_index: np.ndarray) -> np.ndarray:
    return X[t_index - lag, var]


def parcorr_pvalue(X: np.ndarray, edge: tuple, cond: list,
                   tau_max: int) -> tuple:
    """
    Fisher-z test of  X^i_{t-tau} independent of X^j_t | cond.

    edge : (i, j, tau) with tau >= 1
    cond : list of (var, lag) pairs, lags counted from time t (lag >= 0 allowed
           for variables other than j; in practice all conditioners are lagged).
    Returns (p_value, z_statistic, r).
    """
    i, j, tau = edge
    T = X.shape[0]
    max_needed = max([tau] + [lag for (_v, lag) in cond] + [tau_max])
    t_index = np.arange(max_needed, T)
    n = len(t_index)

    x = _lagged_column(X, i, tau, t_index)
    y = _lagged_column(X, j, 0, t_index)

    if cond:
        Z = np.column_stack([_lagged_column(X, v, lag, t_index) for (v, lag) in cond])
        Z = np.column_stack([np.ones(n), Z])
        # residualize both series on the conditioning set
        beta_x, *_ = np.linalg.lstsq(Z, x, rcond=None)
        beta_y, *_ = np.linalg.lstsq(Z, y, rcond=None)
        x = x - Z @ beta_x
        y = y - Z @ beta_y
    else:
        x = x - x.mean()
        y = y - y.mean()

    denom = np.sqrt((x @ x) * (y @ y))
    r = float(x @ y / denom) if denom > 0 else 0.0
    r = np.clip(r, -0.999999, 0.999999)
    dof = n - len(cond) - 3
    if dof <= 0:
        return 1.0, 0.0, r
    z = np.arctanh(r) * np.sqrt(dof)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(p), float(z), r


# --------------------------------------------------------------------------
# 3. Lagged discovery: PC1 condition selection + MCI step (PCMCI-style)
# --------------------------------------------------------------------------


@dataclass
class LaggedDiscoveryResult:
    var_names: list
    tau_max: int
    candidate_edges: list                    # all (i, j, tau), i != j, tau >= 1
    p_agg: dict = field(default_factory=dict)     # edge -> p''_e (max over tested sets)
    p_mci: dict = field(default_factory=dict)     # edge -> MCI-step p-value only
    z_mci: dict = field(default_factory=dict)     # edge -> MCI-step z statistic
    parents: dict = field(default_factory=dict)   # j -> list of (var, lag)

    # -- CRI and derived quantities -----------------------------------------
    def cri(self, which: str = "agg") -> float:
        p = self.p_agg if which == "agg" else self.p_mci
        vals = np.array([p[e] for e in self.candidate_edges])
        return float(1.0 - vals.mean())

    def pvalues(self, which: str = "agg") -> np.ndarray:
        p = self.p_agg if which == "agg" else self.p_mci
        return np.array([p[e] for e in self.candidate_edges])

    def split_by_truth(self, gt_lagged: set, which: str = "agg"):
        p = self.p_agg if which == "agg" else self.p_mci
        p_true = np.array([p[e] for e in self.candidate_edges if e in gt_lagged])
        p_abs = np.array([p[e] for e in self.candidate_edges if e not in gt_lagged])
        return p_true, p_abs

    def metrics(self, gt_lagged: set, alpha: float = 0.05, which: str = "agg") -> dict:
        """Per-(i, j, tau) confusion metrics: exact lag-indexed ground truth."""
        p = self.p_agg if which == "agg" else self.p_mci
        TP = FP = FN = TN = 0
        for e in self.candidate_edges:
            present = e in gt_lagged
            declared = p[e] <= alpha
            TP += present and declared
            FP += (not present) and declared
            FN += present and (not declared)
            TN += (not present) and (not declared)
        prec = TP / (TP + FP) if TP + FP else 0.0
        rec = TP / (TP + FN) if TP + FN else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        fpr = FP / (FP + TN) if FP + TN else 0.0
        return dict(TP=TP, FP=FP, FN=FN, TN=TN, precision=prec, recall=rec,
                    F1=f1, FPR=fpr, SHD=FP + FN, CRI=self.cri(which))


def run_lagged_pcmci(X: np.ndarray, tau_max: int, pc_alpha: float = 0.2,
                     max_conds_dim: int = 3, max_parents: int = 8,
                     var_names: list | None = None) -> LaggedDiscoveryResult:
    """
    Lagged-only PCMCI (the tau >= 1 scope of the paper):

      Phase 1 (PC1). For each target j, select lagged conditions among ALL
        (var, lag) pairs, lag in 1..tau_max — including j's own past, which is
        what whitens autocorrelation.  Candidates are removed as soon as one
        tested conditioning set renders them independent (p > pc_alpha).

      Phase 2 (MCI). For every candidate lagged edge (i -> j, tau), test
        X^i_{t-tau} ⟂ X^j_t | pa(X^j_t) \\ {(i,tau)},  pa(X^i_t) shifted by tau.

    Every p-value ever computed for a candidate cross edge (i != j) is recorded,
    and p''_e is their maximum (the paper's aggregation (R2), eq. (2)).
    """
    d = X.shape[1]
    if var_names is None:
        var_names = [f"X{k}" for k in range(d)]
    cand_edges = [(i, j, tau) for i in range(d) for j in range(d) if i != j
                  for tau in range(1, tau_max + 1)]
    res = LaggedDiscoveryResult(var_names=var_names, tau_max=tau_max,
                                candidate_edges=cand_edges)
    tested_p: dict = {e: [] for e in cand_edges}

    def record(i, j, tau, p):
        if i != j:
            tested_p[(i, j, tau)].append(p)

    # ---------------- Phase 1: PC1 per target ----------------
    parents: dict = {}
    for j in range(d):
        cands = [(v, tau) for v in range(d) for tau in range(1, tau_max + 1)]
        strength: dict = {}
        # iteration 0: unconditional
        surviving = []
        for (v, tau) in cands:
            p, z, _ = parcorr_pvalue(X, (v, j, tau), [], tau_max)
            record(v, j, tau, p)
            if p <= pc_alpha:
                surviving.append((v, tau))
                strength[(v, tau)] = abs(z)
        # iterations q = 1, 2, ...: condition on the q strongest OTHER survivors
        for q in range(1, max_conds_dim + 1):
            if len(surviving) <= q:
                break
            order = sorted(surviving, key=lambda c: -strength[c])
            new_surviving = []
            for c in order:
                others = [o for o in order if o != c][:q]
                p, z, _ = parcorr_pvalue(X, (c[0], j, c[1]), others, tau_max)
                record(c[0], j, c[1], p)
                if p <= pc_alpha:
                    new_surviving.append(c)
                    strength[c] = abs(z)
            surviving = new_surviving
        surviving = sorted(surviving, key=lambda c: -strength[c])[:max_parents]
        parents[j] = surviving
    res.parents = parents

    # ---------------- Phase 2: MCI for every candidate edge ----------------
    for (i, j, tau) in cand_edges:
        cond = [c for c in parents[j] if c != (i, tau)]
        cond += [(v, lag + tau) for (v, lag) in parents[i]]
        # de-duplicate while preserving order
        seen, cond_u = set(), []
        for c in cond:
            if c not in seen:
                seen.add(c)
                cond_u.append(c)
        p, z, _ = parcorr_pvalue(X, (i, j, tau), cond_u, tau_max)
        record(i, j, tau, p)
        res.p_mci[(i, j, tau)] = p
        res.z_mci[(i, j, tau)] = z

    for e in cand_edges:
        res.p_agg[e] = max(tested_p[e])
    return res


def naive_all_edges(X: np.ndarray, tau_max: int) -> LaggedDiscoveryResult:
    """The uncorrected baseline: unconditional Fisher-z on every lagged edge."""
    d = X.shape[1]
    cand = [(i, j, tau) for i in range(d) for j in range(d) if i != j
            for tau in range(1, tau_max + 1)]
    res = LaggedDiscoveryResult(var_names=[f"X{k}" for k in range(d)],
                                tau_max=tau_max, candidate_edges=cand)
    for e in cand:
        p, z, _ = parcorr_pvalue(X, e, [], tau_max)
        res.p_agg[e] = p
        res.p_mci[e] = p
        res.z_mci[e] = z
    return res


# --------------------------------------------------------------------------
# 4. Calibration theory (overlays for the experiments)
# --------------------------------------------------------------------------


def kappa_theory(phi_i: float, phi_j: float = None) -> float:
    """
    Long-run/reference variance ratio for the sample (lagged) correlation of two
    INDEPENDENT AR(1) processes:  gamma_xi(k) = (phi_i phi_j)^|k|  hence
    kappa = sum_k (phi_i phi_j)^|k| = (1 + phi_i phi_j) / (1 - phi_i phi_j).
    """
    if phi_j is None:
        phi_j = phi_i
    q = phi_i * phi_j
    return (1.0 + q) / (1.0 - q)


def null_mean_p_theory(kappa: float) -> float:
    """
    E[p | H0] when z ~ N(0, kappa) and p = 2(1 - Phi(|z|)):
        E[p] = P(|Z'| > sqrt(kappa) |Z|) = (2/pi) arctan(1/sqrt(kappa)).
    Equals 1/2 at kappa = 1; < 1/2 for kappa > 1 (anti-conservative).
    """
    return (2.0 / np.pi) * np.arctan(1.0 / np.sqrt(kappa))


def null_cri_theory(kappa: float) -> float:
    """Expected CRI under the GLOBAL NULL when every edge has calibration kappa."""
    return 1.0 - null_mean_p_theory(kappa)


def G_kappa(u: np.ndarray, kappa: float) -> np.ndarray:
    """Limiting null CDF of the p-value:  G_kappa(u) (Prop. 'Calibration')."""
    return 2.0 * (1.0 - stats.norm.cdf(stats.norm.ppf(1.0 - u / 2.0) / np.sqrt(kappa)))
