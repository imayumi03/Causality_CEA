
<div align="center">
  <p><b>Causal Discovery in Time Series for Trustworthy AI — 2025/2026</b></p>
  <hr>
</div>

<br />

<p align="center">
  <img src="LogoCS.png" height="120" alt="CentraleSupélec" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="cea.png" height="120" alt="CEA" />
</p>

<br />
<br />

<div align="center">
  <h1>Causal Discovery in Time Series<br>for Trustworthy AI</h1>
  <h3><i>A Comparative Study</i></h3>
</div>

<br />
<br />

<div align="center">
  <table>
      <td></td>
      <td><i>Mounia ABDELMOUMNI</i></td>
      <td><a href="mailto:mounia.abdelmoumni@student-cs.fr">mounia.abdelmoumni@student-cs.fr</a></td>
    </tr>
    <tr>
      <td align="right"><b>Supervisor:</b></td>
      <td><i>Dr. Aurore LOMET</i></td>
      <td><a href="mailto:aurore.lomet@cea.fr">aurore.lomet@cea.fr</a></td>
    </tr>
    <tr>
      <td align="right"><b>Maintainer:</b></td>
      <td><i>Dr. Myriam Tami</i></td>
      <td><a href="mailto:myriam.tami@centralesupelec.fr">myriam.tami@centralesupelec.fr</a></td>
    </tr>
    <tr>
      <td align="right"><b>Date:</b></td>
      <td><i>April 13, 2026</i></td>
      <td></td>
    </tr>
  </table>
</div>

<br />


## Comparative benchmark and CRI\(_{TS}\) extension on the CausalChamber wind tunnel

This repository studies **causal discovery in physical time-series data**. The project has two complementary objectives:

1. Benchmark the **PCMCI algorithmic family** on a real wind-tunnel system with a known causal structure.
2. Extend the **Causal Relevance Index (CRI)** from static graphs to lagged time-series graphs, obtaining **CRI\(_{TS}\)**, and evaluate whether it behaves as predicted by the theory.

The main report contains the complete scientific discussion and the mathematical demonstrations. This README is a practical guide to the repository: it explains what was done, how the files relate to each other, how to run the code, and how the CRI\(_{TS}\) notebook empirically validates the theoretical results.

> **Important distinction:** the report provides the mathematical proofs. The code provides an empirical validation of the theoretical predictions on the CausalChamber dataset. Numerical experiments support the theory; they do not replace the proofs.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Why time-series causal discovery is difficult](#2-why-time-series-causal-discovery-is-difficult)
3. [Main contributions](#3-main-contributions)
4. [Repository structure](#4-repository-structure)
5. [Benchmarking causal discovery on CausalChamber](#5-benchmarking-causal-discovery-on-causalchamber)
6. [Focus: the CRI\(_{TS}\) contribution](#6-focus-the-crits-contribution)
7. [How the notebook validates the CRI\(_{TS}\) theory](#7-how-the-notebook-validates-the-crits-theory)
8. [Installation](#8-installation)
9. [How to run the project](#9-how-to-run-the-project)
10. [Generated outputs](#10-generated-outputs)
11. [Reproducibility and implementation notes](#11-reproducibility-and-implementation-notes)
12. [Where to find the demonstrations in the report](#12-where-to-find-the-demonstrations-in-the-report)
13. [References](#13-references)

---

## 1. Project overview

Many machine-learning models can identify correlations and make predictions. However, a reliable scientific or industrial decision often requires a stronger statement: **which variable is a cause, which variable is an effect, and at what delay does the influence appear?**

This project investigates that question using the **CausalChamber wind tunnel**, a controlled physical device with known causal relationships between actuators and sensors. The study compares causal-discovery methods on both synthetic and physical data and highlights the gap between idealized simulations and noisy real-world dynamics.

The second part of the work addresses a limitation of standard evaluation metrics. Metrics such as precision, recall, F1-score, and Structural Hamming Distance (SHD) compare a learned graph with a known ground-truth graph. In practice, the true graph is rarely available. Moreover, these metrics reduce each edge decision to a binary answer and discard the statistical evidence behind it.

The **Causal Relevance Index** addresses this problem by aggregating edge-wise p-values into a graph-level score. This project extends that idea to time series.

---

## 2. Why time-series causal discovery is difficult

Time-series causal discovery is more difficult than static causal discovery for several reasons:

- **Autocorrelation:** a variable is often correlated with its own recent past. This can generate spurious associations with other variables.
- **Unknown causal delays:** an influence may appear after one or several time steps. The algorithm must search over both the variable pair and the lag.
- **Indirect paths:** a variable may appear related to another one only because the effect propagates through intermediate variables.
- **Contemporaneous effects:** physical interactions can occur faster than the sensor sampling rate and may appear instantaneous in the recorded data.
- **Noise and continuous dynamics:** real physical systems do not always fit neatly into discrete lag buckets.

The repository evaluates methods designed to address these issues rather than relying only on pairwise correlation or standard static causal-discovery algorithms.

---

## 3. Main contributions

### 3.1 Comparative benchmark on a physical system

The benchmarking work compares four configurations:

| Causal-discovery algorithm | Conditional-independence test | Main characteristic |
|---|---|---|
| PCMCI | RobustParCorr | Robust linear baseline for lagged effects |
| PCMCI+ | RobustParCorr | Robust linear baseline including contemporaneous effects |
| PCMCI | CMIknn | Non-parametric conditional-mutual-information test |
| PCMCI+ | CMIknn | Non-parametric test including contemporaneous effects |

The scripts evaluate the learned causal graphs using strict and relaxed protocols. This makes it possible to distinguish a genuine failure to detect a physical relationship from a more limited error in the estimated direction or discrete lag.

### 3.2 Extension of CRI to time-series graphs

The CRI workstream extends the static Causal Relevance Index to **directed, lag-indexed candidate edges**. The extension is denoted **CRI\(_{TS}\)**.

The report contains the theoretical development. In particular, it explains:

- how the static framework is adapted to time-series graphs;
- why the original i.i.d. assumption can be replaced by stationarity and mixing assumptions;
- why correlations among p-values do not invalidate CRI\(_{TS}\);
- why CRI\(_{TS}\) remains useful when Type-II errors occur;
- which conditional-independence tests have a theoretical super-uniformity guarantee in the proposed setting.

The notebook then evaluates the predictions of this theory on the CausalChamber wind-tunnel data.

---

## 4. Repository structure

| File | Purpose |
|---|---|
| `LABPROJECT_Report.pdf` | Complete report: motivation, benchmark, CRI\(_{TS}\) definition, theoretical demonstrations, results, and appendices. |
| `causalchamber_better.py` | Main benchmark script. Compares PCMCI and PCMCI+ with RobustParCorr and CMIknn. Evaluates both strict lag-specific recovery and a lag-collapsed representation. |
| `causalchamber_undirected.py` | Complementary benchmark script. Compares directed and undirected evaluation to quantify the effect of direction and lag ambiguity. |
| `CRI_experiment.ipynb` | Notebook dedicated to the CRI\(_{TS}\) workstream: data loading, diagnostics, stationarity analysis, PCMCI+ runs with several CI tests, CRI\(_{TS}\) calculation, plots, and empirical validation. |
| `data/` | Created by the notebook. Stores the downloaded CausalChamber dataset. |
| `outputs/` | Created by the notebook. Stores generated plots. |
| `results/` | Created by the notebook. Stores comparison tables such as CSV summaries. |

The two Python scripts and the CRI notebook answer related but distinct questions:

- The **benchmark scripts** focus on comparing causal-discovery algorithms and evaluation protocols.
- The **CRI notebook** focuses on the behavior of CRI\(_{TS}\) and its relationship with the theoretical results derived in the report.

---

## 5. Benchmarking causal discovery on CausalChamber

### 5.1 Four-variable benchmark used by the Python scripts

The standalone benchmark scripts use four wind-tunnel variables:

```text
hatch
load_in
rpm_in
pressure_upwind
```

The expected physical structure contains four directed edges at lag 1:

```text
hatch   --> rpm_in
hatch   --> pressure_upwind
load_in --> rpm_in
load_in --> pressure_upwind
```

The scripts use:

```python
N_SAMPLES = 2000
TAU_MAX = 2
ALPHA = 0.05
GROUND_TRUTH_LAG = 1
```

### 5.2 `causalchamber_better.py`

This script performs the main causal-discovery benchmark. Its workflow is:

1. Load the selected variables from `wt_combined_data.csv`.
2. Build the lag-specific ground-truth adjacency tensor and the lag-collapsed adjacency matrix.
3. Run four algorithm/test combinations:
   - `PCMCI_RobustParCorr`
   - `PCMCI+_RobustParCorr`
   - `PCMCI_CMIknn`
   - `PCMCI+_CMIknn`
4. Evaluate the results in two ways:
   - **3D lag-specific evaluation:** an edge is correct only when its direction and lag match the ground truth.
   - **2D collapsed evaluation:** an edge is counted once it is detected at any lag.
5. Save the numerical metrics, discovered edges, adjacency comparisons, and plots.

The 2D collapsed evaluation is useful because a physical mechanism can be detected at a neighboring discrete lag even when the exact lag does not match the manually encoded reference.

### 5.3 `causalchamber_undirected.py`

This script isolates a second source of error: **direction ambiguity**.

It compares:

- **Directed evaluation:** `A --> B` and `B --> A` are different predictions.
- **Undirected evaluation:** `A --> B` and `B --> A` are treated as the same physical connection `A -- B`.

The undirected score is calculated only on the upper triangle of the symmetric adjacency matrix to avoid counting each connection twice.

This relaxed protocol answers a practical question:

> Did the method identify that two physical variables interact, even when the direction or the discrete lag is imperfectly estimated?

---

## 6. Focus: the CRI\(_{TS}\) contribution

### 6.1 Motivation

Constraint-based causal-discovery algorithms run many conditional-independence tests. Behind a learned graph lies a large collection of p-values. A final binary graph discards most of that information.

Two edges may both be kept at a significance level of `0.05`, while one edge is supported by very strong evidence and the other barely passes the threshold. Standard metrics do not capture this difference.

CRI\(_{TS}\) uses the raw p-values to measure the **strength of causal evidence** at the graph level.

### 6.2 Candidate edge set for time series

For a process with `d` variables and a maximum lag `tau_max`, the candidate directed lagged edges are:

```math
E = \{(i \rightarrow j, \tau): i \neq j,\; \tau \in \{1, \ldots, \tau_{\max}\}\}.
```

The number of candidates is:

```math
|E| = d(d-1)\tau_{\max}.
```

### 6.3 Edge-wise aggregation

For each candidate edge `e`, several conditional-independence tests may be performed using different conditioning sets `S`. The edge-wise p-value is defined as:

```math
p''_e = \max_{S \in \mathcal{S}^{\mathrm{tested}}_e} p_e(S).
```

The maximum is deliberate. For an absent edge, it is enough that at least one conditioning set separates the variables. Using the maximum provides a conservative edge-wise value and is robust to accidentally small p-values produced by some conditioning sets.

### 6.4 Definition of CRI\(_{TS}\)

The time-series Causal Relevance Index is:

```math
\mathrm{CRI}_{TS}(G)
= 1 - \frac{1}{|E|}\sum_{e \in E} p''_e.
```

A higher score indicates that the method assigns stronger statistical evidence to causal structure. However, CRI\(_{TS}\) should primarily be used **to compare methods or configurations**, not as a universal accuracy score.

### 6.5 Why CRI\(_{TS}\) does not generally reach 1

In a sparse graph, most candidate edges are absent. Under the null hypothesis, valid p-values have an expected value of at least `0.5`. Therefore, absent edges contribute a non-zero baseline to the average.

If `s` denotes the edge density, the theoretical sparsity-dependent ceiling is:

```math
\mathrm{CRI}^{\max}_{TS} = \frac{1+s}{2}.
```

This is why an absolute CRI\(_{TS}\) value must be interpreted in the context of graph sparsity.

### 6.6 The theoretical results established in the report

The complete demonstrations are provided in **Chapter 4** and **Appendices A–B** of the report. The main ideas are summarized below.

#### A. Replacing i.i.d. sampling with time-series assumptions

The extension assumes:

1. **Strict stationarity** of the process.
2. **Strong mixing:** the temporal dependence weakens sufficiently fast as the lag increases.

Under these assumptions, the report shows that the key p-value property needed by CRI can still hold for time series.

#### B. Super-uniform null p-values

For an absent edge, the required property is:

```math
\Pr\left(p_e(S^*) \leq u \mid H_{0,e}(S^*)\right) \leq u,
\qquad \forall u \in [0,1].
```

This means that false causal claims are not encouraged by systematically underestimated p-values.

The report provides theoretical demonstrations for:

- **RobustParCorr**, under the stated stationarity and mixing assumptions for the positively autocorrelated class considered in the study;
- **GPDC**, under the stated assumptions and a correctly specified or consistently approximated Gaussian-process regression model.

CMIknn is included as an empirical comparison, but the report does **not** claim the same formal super-uniformity guarantee for CMIknn under temporal dependence.

#### C. Why correlated p-values are not a problem for CRI\(_{TS}\)

P-values calculated from the same time series are generally correlated. The report shows that this does not invalidate CRI\(_{TS}\), because CRI\(_{TS}\) is an average and its guarantees rely on **linearity of expectation**, not on independence among p-values.

#### D. Why CRI\(_{TS}\) remains useful with Type-II errors

A Type-II error occurs when a true causal edge is missed. This is common in real time-series data.

The report proves that CRI\(_{TS}\) remains a meaningful plausibility and ranking criterion even in this realistic regime. In simple terms: when at least one true edge receives sufficiently strong evidence, its p-value pulls the graph-level average downward and increases CRI\(_{TS}\) relative to a global-null graph.

---

## 7. How the notebook validates the CRI\(_{TS}\) theory

The notebook `CRI_experiment.ipynb` implements the empirical CRI\(_{TS}\) analysis.

### 7.1 Dataset and selected variables

The notebook downloads the CausalChamber dataset:

```python
DATASET_NAME = "wt_walks_v1"
EXPERIMENT_NAME = "actuators_random_walk_1"
```

It selects six variables:

```text
load_in
load_out
rpm_in
rpm_out
pressure_downwind
pressure_ambient
```

The microphone signal is excluded from the CRI analysis because its non-stationarity is difficult to remove without altering the relevant relationships.

The ground-truth graph used by the notebook contains five directed physical relationships:

```text
load_in          --> rpm_in
load_out         --> rpm_out
rpm_in           --> pressure_downwind
rpm_out          --> pressure_downwind
pressure_ambient --> pressure_downwind
```

### 7.2 Data diagnostics

Before running causal discovery, the notebook explores:

- time-series plots;
- Pearson-correlation matrices;
- mutual-information matrices;
- Shapiro–Wilk normality tests;
- Augmented Dickey–Fuller stationarity tests;
- first-order differencing strategies;
- optional downsampling experiments.

These diagnostics are important because the CRI\(_{TS}\) theory is derived under stationarity assumptions and because physical sensor dynamics can be slow relative to the sampling rate.

### 7.3 PCMCI+ configurations

The notebook compares three conditional-independence tests within PCMCI+:

| CI test | Type | Role in this project |
|---|---|---|
| RobustParCorr | Robust linear test | Theoretically analyzed and computationally efficient |
| GPDC | Gaussian-process distance-correlation test | Theoretically analyzed under GP-model assumptions |
| CMIknn | Non-parametric nearest-neighbor test | Empirical comparison; no formal time-series super-uniformity claim |

For the main comparison, the notebook uses:

```python
TAU_MAX = 5
pc_alpha = 0.2
```

The implementation extracts the returned lagged p-values from the PCMCI+ p-value matrix and calculates:

```python
cri = 1.0 - pvalues.mean()
```

This is the practical notebook implementation of the CRI\(_{TS}\) graph-level aggregation.

### 7.4 Validation 1: absent-edge p-values follow the expected null behavior

The notebook separates p-values associated with true and absent edges and plots their distributions.

The report states that absent-edge p-values have means above the `0.5` null baseline for the three tested CI methods, while true-edge p-values are lower on average. This is consistent with the expected null behavior derived in the theory.

### 7.5 Validation 2: CRI\(_{TS}\) provides a threshold-free method ranking

The notebook compares the three CI tests and obtains the following reported values:

| CI test | CRI\(_{TS}\) | Precision | Recall | F1 | FDR | TP / FP / FN | Approx. runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| RobustParCorr | 0.341 | 1.000 | 0.400 | 0.571 | 0.000 | 2 / 0 / 3 | 4 s |
| GPDC | 0.381 | 0.333 | 0.400 | 0.364 | 0.667 | 2 / 4 / 3 | 52 min |
| CMIknn | 0.263 | 1.000 | 0.200 | 0.333 | 0.000 | 1 / 0 / 4 | 29 min |

The CRI\(_{TS}\) ranking is:

```text
GPDC > RobustParCorr > CMIknn
```

This result illustrates the purpose of CRI\(_{TS}\): **GPDC receives the highest CRI\(_{TS}\)** because it assigns stronger statistical evidence to the causal edges it identifies, even though **RobustParCorr receives the best F1-score**. CRI\(_{TS}\) and F1 measure complementary aspects of the result.

The notebook also sweeps the significance threshold `alpha` and shows that threshold-dependent metrics such as F1 and SHD can change when the threshold changes. CRI\(_{TS}\) uses raw p-values directly and does not require selecting a decision threshold first.

### 7.6 Validation 3: CRI\(_{TS}\) tracks causal-discovery power

The notebook studies the relationship between CRI\(_{TS}\) and the true-positive rate (TPR) across sample-size experiments using RobustParCorr.

The report gives a Pearson correlation of:

```text
r = 0.956
```

This supports the theoretical result that CRI\(_{TS}\) remains useful as a ranking criterion even when Type-II errors occur.

### 7.7 What the code validates, precisely

The notebook empirically checks the main predictions derived in the report:

1. **Null behavior:** absent-edge p-values are concentrated around or above the theoretical null baseline.
2. **Ranking behavior:** CRI\(_{TS}\) distinguishes CI tests according to the strength of their statistical evidence.
3. **Power tracking:** CRI\(_{TS}\) co-evolves with TPR across the tested configurations.
4. **Threshold independence:** CRI\(_{TS}\) can be computed without fixing the final edge-selection threshold `alpha`.

The proofs themselves remain in the report. The notebook is the experimental counterpart of those proofs.

---

## 8. Installation

### 8.1 Create an isolated environment

Using `venv`:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 8.2 Install the required packages

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn networkx scikit-learn statsmodels dcor jupyter causalchamber tigramite
```

The notebook also contains an installation cell, but using a dedicated environment before launching Jupyter is recommended for reproducibility.

---

## 9. How to run the project

### 9.1 Run the main benchmark

The standalone scripts expect a CSV file named `wt_combined_data.csv` unless another path is provided.

```bash
python causalchamber_better.py \
  --csv_path wt_combined_data.csv \
  --output_dir causalchamber_better
```

### 9.2 Run the directed-versus-undirected evaluation

```bash
python causalchamber_undirected.py \
  --csv_path wt_combined_data.csv \
  --output_dir causalchamber_undirected
```

### 9.3 Run the CRI\(_{TS}\) notebook

```bash
jupyter lab CRI_experiment.ipynb
```

Run the notebook from top to bottom in a fresh kernel. GPDC and CMIknn can require substantially more computation time than RobustParCorr.

---

## 10. Generated outputs

### 10.1 Outputs from `causalchamber_better.py`

The main script generates files such as:

```text
causalchamber_better/
├── timeseries.png
├── ground_truth_graph.png
├── tigramite_results.png
├── comparison_bars.png
├── adjacency_comparison.png
├── discovered_edges.txt
└── benchmark_results.csv
```

### 10.2 Outputs from `causalchamber_undirected.py`

The directed-versus-undirected script generates files such as:

```text
causalchamber_undirected/
├── tigramite_results.png
├── comparison_directed_vs_undirected.png
├── adjacency_comparison.png
└── benchmark_results.csv
```

### 10.3 Outputs from `CRI_experiment.ipynb`

The notebook generates plots and tables including:

```text
outputs/
├── p-value distributions
├── true-vs-absent edge comparisons
├── CRI_TS-vs-TPR plots
├── alpha-sensitivity plots
├── adjacency comparisons
├── confusion matrices
└── Tigramite causal graphs

results/
└── ci_test_comparison.csv
```

---

## 11. Reproducibility and implementation notes

The repository contains both polished scripts and an exploratory notebook. The following points should be kept in mind when reproducing or extending the study.

### 11.1 The scripts and the CRI notebook use different experimental views

The standalone scripts use a four-variable graph focused on `hatch`, `load_in`, `rpm_in`, and `pressure_upwind`.

The CRI notebook uses a six-variable graph focused on fan loads, fan speeds, and downwind/ambient pressure.

These are complementary experiments. Their ground-truth edge lists and hyperparameters should not be mixed unintentionally.

### 11.2 Use one clearly documented preprocessing pipeline

The notebook contains exploratory cells testing several stationarity and differencing strategies. For a publication-grade rerun:

1. restart the kernel;
2. select the intended preprocessing strategy;
3. apply exactly the same preprocessing to all CI tests;
4. rerun all comparison cells from a clean state;
5. report the final preprocessing choice explicitly.

### 11.3 Verify the effective sample size in sample-size sweeps

The loaded experiment contains a finite number of rows. A slice such as:

```python
df_full.iloc[:5000]
```

cannot produce 5000 observations when the selected experiment contains fewer rows. Before interpreting a sample-size sweep, print the effective number of rows actually passed to PCMCI+ at each iteration. A genuine large-`T` analysis requires a longer experiment or a documented method for combining compatible sequences.

### 11.4 Practical and formal edge-wise p-values

The theory defines:

```math
p''_e = \max_{S \in \mathcal{S}^{\mathrm{tested}}_e} p_e(S).
```

The notebook uses the lagged `p_matrix` returned by PCMCI+ as its practical edge-wise p-value matrix. This is appropriate for the empirical analysis implemented in the notebook. For a strict equation-level implementation of the theoretical aggregation, store the intermediate p-values for all tested conditioning sets and explicitly calculate the maximum for each candidate edge.

### 11.5 Interpret CRI\(_{TS}\) as a comparative score

CRI\(_{TS}\) depends on graph sparsity and on the distribution of raw p-values. It is most informative when comparing methods, hyperparameters, preprocessing choices, or datasets under a consistent protocol.

---

## 12. Where to find the demonstrations in the report

The mathematical details are intentionally kept in the report rather than repeated in full in this README.

| Topic | Location in `LABPROJECT_Report.pdf` |
|---|---|
| Motivation for CRI and edge-specific p-values | Chapter 4, Section 4.1 |
| Definition of CRI\(_{TS}\) | Chapter 4, Definition 4.1 |
| Sparsity-dependent maximum and worked example | Chapter 4, Section 4.1.2 |
| Inherited CRI properties | Chapter 4, Proposition 4.1 |
| CRI\(_{TS}\) as a ranking criterion with Type-II errors | Chapter 4, Proposition 4.2 |
| Empirical validation on CausalChamber | Chapter 4, Section 4.2 |
| Static Strobl framework and portable components | Appendix A |
| Replacement of i.i.d. by stationarity and strong mixing | Appendix B |
| Super-uniformity argument for RobustParCorr | Appendix B, Proposition B.1 |
| Super-uniformity argument for GPDC | Appendix B, Proposition B.2 |
| Why dependence among p-values does not invalidate CRI | Appendix B, Proposition B.3 |

---

## 13. References

The report contains the complete bibliography. The central references for understanding the repository are:

1. Gamella et al., **CausalChamber: a physical testbed for causal discovery**.
2. Runge et al., work on **PCMCI** and **PCMCI+** for causal discovery in time series.
3. Kunitomo-Jacquin et al., work introducing the **Causal Relevance Index**.
4. Strobl, Spirtes, and Visweswaran, **Estimating and controlling the false discovery rate of the PC algorithm using edge-specific p-values**, *ACM Transactions on Intelligent Systems and Technology*, 2019.
5. Ibragimov and Ibragimov–Linnik, limit theorems for stationary and mixing sequences.
6. Benjamini and Yekutieli, false-discovery-rate control under dependency.

For the full theoretical derivation, experimental discussion, and bibliography, read [`LABPROJECT_Report.pdf`](./LABPROJECT_Report.pdf).
