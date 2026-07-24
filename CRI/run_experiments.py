"""Run the full CRI_TS experiment suite and save figures + CSV results.

Each experiment (E1-E4) validates a specific statement of the article:
  E1  Prop. 'Calibration' + Cor. 'Sign'  : null p-value ECDF vs G_kappa(u)
  E2  Cor. 'Sign' + Thm 'Master' (iv)    : kappa(phi), E[p|H0](phi), null-CRI(phi)
  E3  Thm 'Master' (ii)-(v) + Cor. 5.3   : full pipeline, consistency, power cond.
  E4  Thm 'Master' (v) use-case          : CRI ranking valid among CALIBRATED tests
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import cri_ts as C

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
t_start = time.time()

# ==========================================================================
# E1 — Null calibration: ECDF of null-edge p-values, naive vs MCI-conditioned
# ==========================================================================
print("E1: null calibration ...")
PHIS = [0.0, 0.5, 0.8, 0.95]
R, T = 300, 500
u = np.linspace(1e-4, 1, 400)

fig, axes = plt.subplots(1, len(PHIS), figsize=(4.1 * len(PHIS), 3.8), sharey=True)
e1_rows = []
for ax, phi in zip(axes, PHIS):
    p_naive, p_cond = [], []
    for r in range(R):
        rng = np.random.default_rng(1000 * int(100 * phi) + r)
        X = C.two_independent_ar1(phi, phi, T, rng)
        p1, _, _ = C.parcorr_pvalue(X, (0, 1, 1), [], 1)
        p2, _, _ = C.parcorr_pvalue(X, (0, 1, 1), [(1, 1), (0, 2)], 2)  # own pasts
        p_naive.append(p1); p_cond.append(p2)
    p_naive, p_cond = np.sort(p_naive), np.sort(p_cond)
    kap = C.kappa_theory(phi)
    ecdf = np.arange(1, R + 1) / R
    ax.plot(u, u, "k--", lw=1, label="Uniform (calibrated)")
    ax.plot(u, C.G_kappa(u, kap), color="#B71C1C", lw=1.6,
            label=fr"$G_\kappa(u)$, $\kappa$={kap:.1f}")
    ax.step(p_naive, ecdf, color="#F44336", lw=1.8, label="naive (empirical)")
    ax.step(p_cond, ecdf, color="#1565C0", lw=1.8, label="MCI-conditioned")
    ax.set_title(fr"$\phi$ = {phi}", fontsize=12)
    ax.set_xlabel("u"); ax.grid(alpha=0.2)
    if ax is axes[0]:
        ax.set_ylabel(r"$\hat{P}(p \leq u \mid H_0)$")
    ax.legend(fontsize=7, loc="lower right")
    e1_rows.append(dict(phi=phi, kappa_theory=kap,
                        mean_p_naive=p_naive.mean(), mean_p_cond=p_cond.mean(),
                        mean_p_naive_theory=C.null_mean_p_theory(kap),
                        fpr05_naive=(p_naive <= .05).mean(),
                        fpr05_cond=(p_cond <= .05).mean()))
fig.suptitle("E1 — Null-edge p-value calibration: anti-conservatism of the naive test "
             "and its repair by MCI conditioning", fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(f"{OUT}/E1_null_calibration.png", dpi=150, bbox_inches="tight")
e1 = pd.DataFrame(e1_rows); e1.to_csv(f"{OUT}/E1_calibration.csv", index=False)
print(e1.round(3).to_string(index=False))

# ==========================================================================
# E2 — kappa(phi), E[p|H0](phi), null-CRI(phi): empirical vs closed form
# ==========================================================================
print("\nE2: kappa / null-CRI vs phi ...")
phis = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95])
R2, T2, D2, TAU2 = 150, 500, 4, 2       # d=4 independent AR(1)s -> global null
rows = []
for phi in phis:
    k_n, k_c, cri_n, cri_c = [], [], [], []
    for r in range(R2):
        rng = np.random.default_rng(777 + 10_000 * int(100 * phi) + r)
        sysn = C.VARSystem(d=D2, edges=[], phi=np.full(D2, phi), tau_max=TAU2)
        X = sysn.simulate(T2, rng)
        rn = C.naive_all_edges(X, TAU2)
        rc = C.run_lagged_pcmci(X, TAU2, pc_alpha=0.2)
        k_n.append(np.array([rn.z_mci[e] for e in rn.candidate_edges]))
        k_c.append(np.array([rc.z_mci[e] for e in rc.candidate_edges]))
        cri_n.append(rn.cri()); cri_c.append(rc.cri("mci"))
    z_n, z_c = np.concatenate(k_n), np.concatenate(k_c)
    rows.append(dict(phi=phi,
                     kappa_naive=z_n.var(), kappa_cond=z_c.var(),
                     kappa_theory=C.kappa_theory(phi),
                     cri_naive=np.mean(cri_n), cri_naive_se=np.std(cri_n)/np.sqrt(R2),
                     cri_cond=np.mean(cri_c), cri_cond_se=np.std(cri_c)/np.sqrt(R2),
                     cri_naive_theory=C.null_cri_theory(C.kappa_theory(phi))))
e2 = pd.DataFrame(rows); e2.to_csv(f"{OUT}/E2_kappa_nullcri.csv", index=False)
print(e2.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
pp = np.linspace(0, 0.96, 200)
axes[0].plot(pp, [C.kappa_theory(v) for v in pp], color="#B71C1C", lw=1.6,
             label=r"theory $\kappa=(1+\phi^2)/(1-\phi^2)$")
axes[0].plot(e2.phi, e2.kappa_naive, "o", color="#F44336", ms=7, label="naive (empirical)")
axes[0].plot(e2.phi, e2.kappa_cond, "s", color="#1565C0", ms=7, label="MCI-conditioned")
axes[0].axhline(1, color="gray", ls=":", lw=1)
axes[0].set_xlabel(r"autocorrelation $\phi$")
axes[0].set_ylabel(r"$\hat\kappa=\widehat{\mathrm{Var}}(z_n)$")
axes[0].set_title("Calibration constant vs autocorrelation")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.2)

axes[1].plot(pp, [C.null_cri_theory(C.kappa_theory(v)) for v in pp], color="#B71C1C",
             lw=1.6, label=r"theory $1-\frac{2}{\pi}\arctan(\kappa^{-1/2})$")
axes[1].errorbar(e2.phi, e2.cri_naive, yerr=2*e2.cri_naive_se, fmt="o", color="#F44336",
                 ms=7, capsize=3, label="naive (empirical)")
axes[1].errorbar(e2.phi, e2.cri_cond, yerr=2*e2.cri_cond_se, fmt="s", color="#1565C0",
                 ms=7, capsize=3, label="PCMCI (MCI p-values)")
axes[1].axhline(0.5, color="gray", ls="--", lw=1.2, label=r"global-null ceiling $1/2$ (Thm 1 iv)")
axes[1].set_xlabel(r"autocorrelation $\phi$")
axes[1].set_ylabel(r"$\mathrm{CRI_{TS}}$ under the global null")
axes[1].set_title("Global-null ceiling: naive test violates it, MCI restores it")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.2)
fig.suptitle("E2 — The sign of the autocorrelation effect (Cor. 'Sign'): "
             "variance inflation is ANTI-conservatism", fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(f"{OUT}/E2_kappa_nullcri.png", dpi=150, bbox_inches="tight")

# ==========================================================================
# E3 — Full pipeline on a known VAR: consistency + Corollary 5.3
# ==========================================================================
print("\nE3: full pipeline vs sample size ...")
SYS = C.VARSystem(
    d=4,
    edges=[(0, 1, 1, 0.20), (0, 2, 2, 0.18), (1, 2, 1, 0.16),
           (1, 3, 2, 0.18), (2, 3, 1, 0.20), (3, 0, 2, 0.15)],
    phi=np.array([0.6, 0.7, 0.6, 0.7]),
    tau_max=2,
)
GT = SYS.gt_lagged()
N_CAND = SYS.d * (SYS.d - 1) * SYS.tau_max
Ts = [150, 250, 500, 1000, 2000]
REPS = 12
rows, per_edge_rows = [], []
for T3 in Ts:
    for r in range(REPS):
        rng = np.random.default_rng(50_000 + 97 * T3 + r)
        X = SYS.simulate(T3, rng)
        res = C.run_lagged_pcmci(X, SYS.tau_max, pc_alpha=0.2)
        m = res.metrics(GT, alpha=0.05)
        p_true, p_abs = res.split_by_truth(GT)
        Xn = C.VARSystem(d=SYS.d, edges=[], phi=SYS.phi, tau_max=SYS.tau_max).simulate(
            T3, np.random.default_rng(90_000 + 97 * T3 + r))
        cri_null_twin = C.run_lagged_pcmci(Xn, SYS.tau_max, pc_alpha=0.2).cri()
        rows.append(dict(T=T3, rep=r, CRI=m["CRI"], CRI_mci=res.cri("mci"),
                         CRI_null_twin=cri_null_twin,
                         TPR=m["recall"], FPR=m["FPR"], F1=m["F1"],
                         mean_p_true=p_true.mean(), mean_p_absent=p_abs.mean()))
        for e in res.candidate_edges:
            if e in GT:
                per_edge_rows.append(dict(T=T3, edge=str(e), p_agg=res.p_agg[e]))
e3 = pd.DataFrame(rows); e3.to_csv(f"{OUT}/E3_pipeline.csv", index=False)
g = e3.groupby("T").agg(["mean", "sem"])
print(g[["CRI", "CRI_null_twin", "TPR", "F1", "mean_p_absent"]].round(3).to_string())

wins = (e3.CRI > e3.CRI_null_twin).mean()
print(f"\nCorollary 5.3 (paired): CRI(G*) > CRI(null twin) in {100*wins:.0f}% of runs")

pe = pd.DataFrame(per_edge_rows).groupby(["T", "edge"]).p_agg.mean().reset_index()
viol = pe[pe.p_agg > 0.5]
print(f"Power condition E[p''|H1] <= 1/2: {len(viol)} violations "
      f"out of {len(pe)} (T, edge) cells")
print(pe[pe["T"] == pe["T"].min()].round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
gm = e3.groupby("T").mean(numeric_only=True); gs = e3.groupby("T").sem(numeric_only=True)
axes[0].errorbar(gm.index, gm.CRI, yerr=2*gs.CRI, fmt="o-", color="#E91E63",
                 lw=2, capsize=3, label=r"$\mathrm{CRI_{TS}}(G^*)$")
axes[0].errorbar(gm.index, gm.CRI_null_twin, yerr=2*gs.CRI_null_twin, fmt="s--",
                 color="#616161", lw=1.6, capsize=3,
                 label=r"$\mathrm{CRI_{TS}}$ of matched null twin (Cor. 5.3 baseline)")
axes[0].set_xscale("log"); axes[0].set_xlabel("sample size T"); axes[0].set_ylabel("CRI")
axes[0].set_title(rf"$E[\mathrm{{CRI}}(G^*)] > E[\mathrm{{CRI}}(G_{{\mathrm{{null}}}})]$"
                  f"  (paired win rate {100*wins:.0f}%)")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.2)

axes[1].errorbar(gm.index, gm.CRI_mci, yerr=2*gs.CRI_mci, fmt="o-", color="#E91E63", lw=2,
                 capsize=3, label="CRI (MCI p-values, threshold-free)")
axes[1].errorbar(gm.index, gm.TPR, yerr=2*gs.TPR, fmt="s--", color="#2196F3", lw=2,
                 capsize=3, label=r"TPR at $\alpha=0.05$")
axes[1].errorbar(gm.index, gm.F1, yerr=2*gs.F1, fmt="^--", color="#009688", lw=2,
                 capsize=3, label=r"F1 at $\alpha=0.05$")
axes[1].errorbar(gm.index, gm.FPR, yerr=2*gs.FPR, fmt="v--", color="#F44336", lw=1.6,
                 capsize=3, label=r"FPR at $\alpha=0.05$")
axes[1].set_xscale("log"); axes[1].set_xlabel("sample size T"); axes[1].set_ylabel("score")
axes[1].set_title("CRI tracks discovery quality (Thm 1 v)")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.2)
fig.suptitle("E3 — Full lagged-PCMCI pipeline on a known VAR "
             f"({len(GT)} true edges / {N_CAND} candidates, exact per-(i,j,τ) ground truth)",
             fontsize=12, y=1.03)
fig.tight_layout(); fig.savefig(f"{OUT}/E3_pipeline.png", dpi=150, bbox_inches="tight")

Xb = SYS.simulate(2000, np.random.default_rng(4242))
rb = C.run_lagged_pcmci(Xb, SYS.tau_max, pc_alpha=0.2)
pt, pa = rb.split_by_truth(GT)
fig, ax = plt.subplots(figsize=(7.5, 4))
bins = np.linspace(0, 1, 30)
ax.hist(pa, bins=bins, alpha=0.6, color="#F44336", edgecolor="white",
        label=fr"absent edges (n={len(pa)}), mean={pa.mean():.2f}")
ax.hist(pt, bins=bins, alpha=0.8, color="#4CAF50", edgecolor="white",
        label=fr"present edges (n={len(pt)}), mean={pt.mean():.2f}")
ax.axvline(0.5, color="gray", ls=":", lw=1.5, label=r"$E[p''\mid H_0]\geq 1/2$ (Thm 1 ii)")
ax.set_xlabel(r"max-aggregated edge p-value $p''_e$"); ax.set_ylabel("count")
ax.set_title("Per-(i, j, τ) p-value split at T = 2000")
ax.legend(fontsize=9); ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig(f"{OUT}/E3_pvalue_split.png", dpi=150, bbox_inches="tight")

# ==========================================================================
# E4 — CRI as a ranking criterion, and why (R1) is a precondition
# ==========================================================================
print("\nE4: CRI ranking among calibrated tests; miscalibration cheats the index ...")
T_rank = [150, 300, 600]
rows4, cheat_rows = [], []
for r in range(12):
    for T4 in T_rank:
        X = SYS.simulate(T4, np.random.default_rng(313_000 + 17 * T4 + r))
        res = C.run_lagged_pcmci(X, SYS.tau_max, pc_alpha=0.2)
        m = res.metrics(GT, alpha=0.05)
        rows4.append(dict(rep=r, T=T4, CRI_mci=res.cri("mci"), F1=m["F1"]))
        if T4 == max(T_rank):
            rn = C.naive_all_edges(X, SYS.tau_max)
            mn = rn.metrics(GT, alpha=0.05)
            cheat_rows.append(dict(rep=r,
                                   CRI_pcmci=res.cri("mci"), F1_pcmci=m["F1"],
                                   CRI_naive=rn.cri("mci"), F1_naive=mn["F1"],
                                   FPR_naive=mn["FPR"], FPR_pcmci=m["FPR"]))
e4 = pd.DataFrame(rows4); e4.to_csv(f"{OUT}/E4_ranking.csv", index=False)
rhos = [spearmanr(sub.CRI_mci, sub.F1).statistic
        for _, sub in e4.groupby("rep") if sub.F1.nunique() > 1]
print(e4.groupby("T").mean(numeric_only=True).round(3).to_string())
print(f"Spearman rho (CRI vs F1 within replicate): mean = {np.mean(rhos):.2f} "
      f"over {len(rhos)} replicates with non-tied F1")

ch = pd.DataFrame(cheat_rows); ch.to_csv(f"{OUT}/E4_cheat.csv", index=False)
print("\n(R1) precondition check — same data, T=600:")
print(ch.mean(numeric_only=True).round(3).to_string())
frac_cheat = (ch.CRI_naive > ch.CRI_pcmci).mean()
print(f"Naive (miscalibrated) test reports HIGHER CRI than PCMCI in "
      f"{100*frac_cheat:.0f}% of runs, despite F1 {ch.F1_naive.mean():.2f} vs "
      f"{ch.F1_pcmci.mean():.2f} and FPR {ch.FPR_naive.mean():.2f} vs {ch.FPR_pcmci.mean():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
gm4 = e4.groupby("T").mean(numeric_only=True); sm4 = e4.groupby("T").sem(numeric_only=True)
axes[0].errorbar(gm4.index, gm4.CRI_mci, yerr=2*sm4.CRI_mci, fmt="o-", color="#E91E63",
                 capsize=3, lw=2, label="CRI (MCI p-values, threshold-free)")
axes[0].errorbar(gm4.index, gm4.F1, yerr=2*sm4.F1, fmt="s--", color="#009688",
                 capsize=3, lw=2, label=r"F1 at $\alpha=0.05$")
axes[0].set_xlabel("sample size T"); axes[0].set_ylabel("score")
axes[0].set_title(f"(a) Among calibrated pipelines, CRI ranks like F1\n"
                  fr"mean within-replicate Spearman $\rho$ = {np.mean(rhos):.2f}")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.2)

x = np.arange(2); w = 0.35
axes[1].bar(x - w/2, [ch.CRI_pcmci.mean(), ch.CRI_naive.mean()], w,
            yerr=2*ch[["CRI_pcmci", "CRI_naive"]].sem().values, capsize=4,
            color="#E91E63", label="CRI")
axes[1].bar(x + w/2, [ch.F1_pcmci.mean(), ch.F1_naive.mean()], w,
            yerr=2*ch[["F1_pcmci", "F1_naive"]].sem().values, capsize=4,
            color="#009688", label="F1")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["PCMCI (calibrated)", "naive (anti-conservative)"])
axes[1].set_ylabel("score")
axes[1].set_title("(b) A miscalibrated test CHEATS the index:\n"
                  "higher CRI, worse graph — (R1) is a precondition")
axes[1].legend(fontsize=9); axes[1].grid(axis="y", alpha=0.2)
fig.suptitle("E4 — CRI as a threshold-free ranking criterion (Thm 1 v): "
             "valid among calibrated tests only", fontsize=12, y=1.05)
fig.tight_layout(); fig.savefig(f"{OUT}/E4_ranking.png", dpi=150, bbox_inches="tight")

print(f"\nTotal runtime: {time.time()-t_start:.0f}s. Figures + CSVs in {OUT}/")
