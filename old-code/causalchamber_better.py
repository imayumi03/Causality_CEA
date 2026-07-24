# -*- coding: utf-8 -*-
"""
Improved Causal Discovery for CausalChamber Wind Tunnel Dataset
================================================================
Applies best practices:
1. Higher sample size (5000)
2. tau_max=2 (detect lag-2 effects)
3. Better CMIknn parameters (knn=10, shuffle_neighbors=100, transform='ranks')
4. Collapsed evaluation (forgiving on lag mismatch)
5. Tigramite's built-in visualization
6. Ground truth at lag 1 (physical systems have delays)

Variables: hatch, load_in, rpm_in, pressure_upwind
Expected causal structure:
  hatch -> rpm_in (lag 1)
  hatch -> pressure_upwind (lag 1)
  load_in -> rpm_in (lag 1)
  load_in -> pressure_upwind (lag 1)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time
import argparse

# Tigramite imports
import tigramite.data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.cmiknn import CMIknn

from sklearn.metrics import precision_score, recall_score, f1_score


# CONFIGURATION (IMPROVED)

N_SAMPLES = 2000          # Reduced for faster CMIknn
TAU_MAX = 2               # Increased from 1 (can detect lag-2)
ALPHA = 0.05
TARGET_VARS = ['hatch', 'load_in', 'rpm_in', 'pressure_upwind']
GROUND_TRUTH_LAG = 1      # Physical systems have delay (not lag 0)


# DATA LOADING

def load_wind_tunnel_data(csv_path="wt_combined_data.csv", n_samples=5000):
    """
    Load wind tunnel data from pre-saved CSV.

    Args:
        csv_path: Path to CSV file
        n_samples: Number of samples to use

    Returns:
        df: DataFrame with target variables
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            f"Please ensure wt_combined_data.csv exists in the working directory."
        )

    print(f"Loading data from: {csv_path}")
    df_full = pd.read_csv(csv_path)

    # Select target variables and limit samples
    df = df_full[TARGET_VARS].iloc[:n_samples].copy()

    print(f"Loaded {len(df)} samples, {len(TARGET_VARS)} variables")
    print(f"Variables: {TARGET_VARS}")

    # Basic statistics
    print("\nData statistics:")
    print(df.describe().round(2))

    return df


# GROUND TRUTH (IMPROVED - LAG 1)

def build_ground_truth(var_names, tau_max, causal_lag=1):
    """
    Build ground truth adjacency matrix.

    IMPROVED: Ground truth at lag 1 (physical delay), not lag 0.

    Expected causal structure:
      hatch -> rpm_in (lag 1)
      hatch -> pressure_upwind (lag 1)
      load_in -> rpm_in (lag 1)
      load_in -> pressure_upwind (lag 1)

    Args:
        var_names: List of variable names
        tau_max: Maximum lag
        causal_lag: Lag at which causal effects occur (default 1)

    Returns:
        true_matrix_3d: Shape (n_vars, n_vars, tau_max+1)
        true_matrix_2d: Shape (n_vars, n_vars) collapsed
    """
    n_vars = len(var_names)
    var_idx = {v: i for i, v in enumerate(var_names)}

    # 3D matrix (lag-specific)
    true_matrix_3d = np.zeros((n_vars, n_vars, tau_max + 1), dtype=int)

    # Define edges at specified lag
    edges = [
        ('hatch', 'rpm_in', causal_lag),
        ('hatch', 'pressure_upwind', causal_lag),
        ('load_in', 'rpm_in', causal_lag),
        ('load_in', 'pressure_upwind', causal_lag),
    ]

    for cause, effect, lag in edges:
        if lag <= tau_max:
            true_matrix_3d[var_idx[cause], var_idx[effect], lag] = 1

    # 2D collapsed matrix (any lag)
    true_matrix_2d = np.zeros((n_vars, n_vars), dtype=int)
    true_matrix_2d[var_idx['hatch'], var_idx['rpm_in']] = 1
    true_matrix_2d[var_idx['hatch'], var_idx['pressure_upwind']] = 1
    true_matrix_2d[var_idx['load_in'], var_idx['rpm_in']] = 1
    true_matrix_2d[var_idx['load_in'], var_idx['pressure_upwind']] = 1

    print(f"\nGround truth: 4 edges at lag {causal_lag}")
    for cause, effect, lag in edges:
        print(f"  {cause} -> {effect} (lag {lag})")

    return true_matrix_3d, true_matrix_2d


# CAUSAL DISCOVERY 

def get_algorithms():
    """Define all 4 algorithms."""
    return [
        {'name': 'PCMCI_RobustParCorr', 'method': 'pcmci', 'ci_test': 'robust_parcorr'},
        {'name': 'PCMCI+_RobustParCorr', 'method': 'pcmciplus', 'ci_test': 'robust_parcorr'},
        {'name': 'PCMCI_CMIknn', 'method': 'pcmci', 'ci_test': 'cmiknn'},
        {'name': 'PCMCI+_CMIknn', 'method': 'pcmciplus', 'ci_test': 'cmiknn'},
    ]


def run_discovery(dataframe, method, ci_test, tau_max, alpha):
    """
    Run causal discovery with IMPROVED parameters.

    CMIknn improvements:
    - knn=10 (fixed, more stable than fraction)
    - shuffle_neighbors=100 (more robust p-values)
    - transform='ranks' (robust to outliers and scale)
    """

    if ci_test == 'robust_parcorr':
        cond_ind_test = RobustParCorr(significance='analytic')
    elif ci_test == 'cmiknn':
        # IMPROVED CMIknn parameters
        cond_ind_test = CMIknn(
            knn=10,                    # Fixed k (not fraction)
            shuffle_neighbors=100,     # More shuffles for robust p-values
            significance='shuffle_test',
            transform='ranks'          # Robust to outliers and scale
        )
    else:
        raise ValueError(f"Unknown ci_test: {ci_test}")

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)

    if method == 'pcmci':
        results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha)
    elif method == 'pcmciplus':
        results = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=alpha)
    else:
        raise ValueError(f"Unknown method: {method}")

    return results, pcmci


# EVALUATION 3D and 2D

def get_adjacency_from_graph(graph):
    """
    Convert Tigramite 3D graph to 2D adjacency matrix.
    Collapses lags: if link exists at ANY lag, mark as 1.
    """
    n_vars = graph.shape[0]
    adj = np.zeros((n_vars, n_vars), dtype=int)

    for i in range(n_vars):
        for j in range(n_vars):
            for tau in range(graph.shape[2]):
                if graph[i, j, tau] != '':
                    adj[i, j] = 1
                    break

    return adj


def get_binary_matrix(results, alpha):
    """Convert p-values to binary matrix."""
    p_matrix = np.nan_to_num(results['p_matrix'], nan=1.0)
    return (p_matrix < alpha).astype(int)


def evaluate_3d(true_matrix, est_matrix, exclude_self=True):
    """
    Evaluate with lag-specific comparison (strict).
    Excludes self-loops (diagonal) by default.
    """
    n = true_matrix.shape[0]

    if exclude_self:
        mask = np.ones_like(true_matrix, dtype=bool)
        for i in range(n):
            mask[i, i, :] = False
        y_true = true_matrix[mask].flatten()
        y_pred = est_matrix[mask].flatten()
    else:
        y_true = true_matrix.flatten()
        y_pred = est_matrix.flatten()

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    shd = FP + FN

    return {
        'Precision_3D': round(precision, 4),
        'Recall_3D': round(recall, 4),
        'F1_3D': round(f1, 4),
        'SHD_3D': shd,
        'TP_3D': TP, 'FP_3D': FP, 'FN_3D': FN
    }


def evaluate_collapsed(true_matrix_2d, pred_matrix_2d, exclude_self=True):
    """
    Evaluate with collapsed 2D matrices (forgiving on lag).
    If edge exists at ANY lag, counts as detected.
    """
    n = true_matrix_2d.shape[0]

    if exclude_self:
        mask = ~np.eye(n, dtype=bool)
        y_true = true_matrix_2d[mask].flatten()
        y_pred = pred_matrix_2d[mask].flatten()
    else:
        y_true = true_matrix_2d.flatten()
        y_pred = pred_matrix_2d.flatten()

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    shd = int(np.sum(true_matrix_2d != pred_matrix_2d))

    return {
        'Precision_2D': round(precision, 4),
        'Recall_2D': round(recall, 4),
        'F1_2D': round(f1, 4),
        'SHD_2D': shd,
        'TP_2D': TP, 'FP_2D': FP, 'FN_2D': FN
    }


# VISUALIZATION (TIGRAMITE BUILT-IN)

def plot_time_series(df, output_dir):
    """Plot time series of all variables."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    for i, (var, color) in enumerate(zip(TARGET_VARS, colors)):
        axes[i].plot(df[var].values[:1000], color=color, linewidth=1, alpha=0.8)
        axes[i].set_ylabel(var, fontsize=11, fontweight='bold')
        axes[i].grid(alpha=0.3)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time Step', fontsize=12)
    plt.suptitle('Wind Tunnel Time Series (First 1000 samples)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/timeseries.png")


def plot_ground_truth(true_matrix_2d, var_names, output_dir):
    """Plot ground truth causal graph."""
    import networkx as nx

    G = nx.DiGraph()
    for var in var_names:
        G.add_node(var)

    # Add edges
    edges = [
        ('hatch', 'rpm_in'),
        ('hatch', 'pressure_upwind'),
        ('load_in', 'rpm_in'),
        ('load_in', 'pressure_upwind'),
    ]
    G.add_edges_from(edges)

    plt.figure(figsize=(10, 8))
    pos = {
        'hatch': (0, 1),
        'load_in': (0, 0),
        'rpm_in': (2, 1),
        'pressure_upwind': (2, 0)
    }

    nx.draw(G, pos, with_labels=True,
            node_color='lightgreen',
            node_size=4000,
            arrowsize=25,
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            width=2,
            arrows=True)

    plt.title('Ground Truth Causal Graph\n(Wind Tunnel: Causes on left, Effects on right)',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ground_truth_graph.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/ground_truth_graph.png")

    # Print adjacency matrix
    print("\nGround Truth Adjacency Matrix:")
    print(pd.DataFrame(true_matrix_2d, index=var_names, columns=var_names))


def plot_tigramite_results(all_results, var_names, output_dir):
    """Plot results using Tigramite's built-in visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    algo_names = ['PCMCI_RobustParCorr', 'PCMCI+_RobustParCorr',
                  'PCMCI_CMIknn', 'PCMCI+_CMIknn']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for algo_name, (row, col) in zip(algo_names, positions):
        if algo_name in all_results:
            results = all_results[algo_name]['results']

            tp.plot_graph(
                val_matrix=results['val_matrix'],
                graph=results['graph'],
                var_names=var_names,
                link_colorbar_label='MCI',
                figsize=(7, 6),
                node_size=0.4,
                arrowhead_size=20,
                curved_radius=0.2,
                label_fontsize=10,
                fig_ax=(fig, axes[row, col])
            )
            axes[row, col].set_title(algo_name.replace('_', ' '),
                                      fontsize=12, fontweight='bold', pad=10)

    plt.suptitle('Causal Discovery Results: Wind Tunnel Data\n(All 4 Algorithms)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tigramite_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/tigramite_results.png")


def plot_comparison_bars(results_df, output_dir):
    """Plot bar charts comparing algorithms."""

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    # Row 1: 3D evaluation (strict)
    metrics_3d = ['Precision_3D', 'Recall_3D', 'F1_3D', 'SHD_3D']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    for ax, metric, color in zip(axes[0], metrics_3d, colors):
        values = results_df[metric].values
        algos = results_df['Algorithm'].values
        bars = ax.bar(range(len(algos)), values, color=color, edgecolor='white')
        ax.set_title(metric.replace('_3D', ' (3D)'), fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels([a.replace('_', '\n') for a in algos], fontsize=8)

        if 'SHD' not in metric:
            ax.set_ylim(0, 1.15)

        for bar, val in zip(bars, values):
            label = f'{val:.2f}' if 'SHD' not in metric else f'{int(val)}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Row 2: 2D evaluation (collapsed)
    metrics_2d = ['Precision_2D', 'Recall_2D', 'F1_2D', 'SHD_2D']

    for ax, metric, color in zip(axes[1], metrics_2d, colors):
        values = results_df[metric].values
        algos = results_df['Algorithm'].values
        bars = ax.bar(range(len(algos)), values, color=color, edgecolor='white')
        ax.set_title(metric.replace('_2D', ' (Collapsed)'), fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels([a.replace('_', '\n') for a in algos], fontsize=8)

        if 'SHD' not in metric:
            ax.set_ylim(0, 1.15)

        for bar, val in zip(bars, values):
            label = f'{val:.2f}' if 'SHD' not in metric else f'{int(val)}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Algorithm Performance Comparison\nTop: 3D (lag-specific), Bottom: 2D (collapsed)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/comparison_bars.png")


def plot_adjacency_comparison(all_results, true_matrix_2d, var_names, output_dir):
    """Plot discovered vs ground truth adjacency matrices."""

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    # Ground truth
    axes[0].imshow(true_matrix_2d, cmap='Greens', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth', fontsize=11, fontweight='bold')
    axes[0].set_xticks(range(4))
    axes[0].set_yticks(range(4))
    axes[0].set_xticklabels(var_names, fontsize=8, rotation=45, ha='right')
    axes[0].set_yticklabels(var_names, fontsize=8)

    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, str(true_matrix_2d[i, j]), ha='center', va='center',
                        fontsize=11, fontweight='bold')

    # Discovered graphs
    algo_names = ['PCMCI_RobustParCorr', 'PCMCI+_RobustParCorr',
                  'PCMCI_CMIknn', 'PCMCI+_CMIknn']

    for idx, algo_name in enumerate(algo_names):
        ax = axes[idx + 1]
        if algo_name in all_results:
            adj = all_results[algo_name]['adj_2d']

            # Color: Green=TP, Red=FP, Blue=FN, White=TN
            colored = np.zeros((4, 4, 3))
            for i in range(4):
                for j in range(4):
                    if true_matrix_2d[i, j] == 1 and adj[i, j] == 1:
                        colored[i, j] = [0.2, 0.8, 0.2]  # Green - TP
                    elif true_matrix_2d[i, j] == 0 and adj[i, j] == 1:
                        colored[i, j] = [0.9, 0.2, 0.2]  # Red - FP
                    elif true_matrix_2d[i, j] == 1 and adj[i, j] == 0:
                        colored[i, j] = [0.2, 0.4, 0.9]  # Blue - FN
                    else:
                        colored[i, j] = [0.95, 0.95, 0.95]  # Light gray - TN

            ax.imshow(colored)
            ax.set_title(algo_name.replace('_', '\n'), fontsize=9, fontweight='bold')
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(var_names, fontsize=7, rotation=45, ha='right')
            ax.set_yticklabels(var_names, fontsize=7)

            for i in range(4):
                for j in range(4):
                    ax.text(j, i, str(adj[i, j]), ha='center', va='center',
                           fontsize=10, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#33cc33', label='TP (Correct)'),
        Patch(facecolor='#e63333', label='FP (Spurious)'),
        Patch(facecolor='#3366e6', label='FN (Missed)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.08))

    plt.suptitle('Adjacency Matrices: Ground Truth vs Discovered (Collapsed)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/adjacency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/adjacency_comparison.png")


def print_discovered_edges(all_results, var_names, output_dir):
    """Print and save discovered edges for each algorithm."""

    lines = []
    lines.append("DISCOVERED EDGES BY ALGORITHM")

    for algo_name, data in all_results.items():
        lines.append(f"\n{algo_name}:")
        est_matrix = data['est_matrix']

        edge_count = 0
        for i in range(4):
            for j in range(4):
                for lag in range(est_matrix.shape[2]):
                    if est_matrix[i, j, lag] == 1 and i != j:
                        lines.append(f"  {var_names[i]} -> {var_names[j]} (lag {lag})")
                        edge_count += 1

        if edge_count == 0:
            lines.append("  No edges detected")
        lines.append(f"  Total: {edge_count} edges")

    output = '\n'.join(lines)
    print(output)

    with open(f'{output_dir}/discovered_edges.txt', 'w') as f:
        f.write(output)
    print(f"Saved: {output_dir}/discovered_edges.txt")


# MAIN BENCHMARK

def run_benchmark(csv_path='wt_combined_data.csv', output_dir='causalchamber_better'):
    """Run the improved causalchamber benchmark."""

    os.makedirs(output_dir, exist_ok=True)

    print("IMPROVED CAUSAL DISCOVERY: WIND TUNNEL DATA")
    print(f"Data: {csv_path}")
    print(f"Samples: {N_SAMPLES}")
    print(f"tau_max: {TAU_MAX}")
    print(f"alpha: {ALPHA}")
    print(f"Ground truth lag: {GROUND_TRUTH_LAG}")
    print(f"Variables: {TARGET_VARS}")
    print("\nIMPROVEMENTS APPLIED:")
    print("  1. Higher sample size (5000)")
    print("  2. tau_max=2 (detect lag-2 effects)")
    print("  3. Better CMIknn (knn=10, shuffle=100, ranks)")
    print("  4. Both 3D and collapsed evaluation")
    print("  5. Ground truth at lag 1 (physical delay)")
    sys.stdout.flush()

    # Load data
    print("\nLoading wind tunnel data...")
    df = load_wind_tunnel_data(csv_path, N_SAMPLES)
    dataframe = pp.DataFrame(df.values, var_names=TARGET_VARS)

    # Build ground truth
    print("\nBuilding ground truth...")
    true_matrix_3d, true_matrix_2d = build_ground_truth(TARGET_VARS, TAU_MAX, GROUND_TRUTH_LAG)

    # Visualizations
    print("\nGenerating initial visualizations...")
    plot_time_series(df, output_dir)
    plot_ground_truth(true_matrix_2d, TARGET_VARS, output_dir)

    # Run all algorithms
    print("RUNNING CAUSAL DISCOVERY ALGORITHMS")
    sys.stdout.flush()

    algorithms = get_algorithms()
    all_results = {}
    results_list = []

    for algo in algorithms:
        print(f"\n{algo['name']}...")
        print(f"  Method: {algo['method']}")
        print(f"  CI Test: {algo['ci_test']}")
        sys.stdout.flush()

        start = time.time()
        results, pcmci = run_discovery(dataframe, algo['method'], algo['ci_test'],
                                        TAU_MAX, ALPHA)
        runtime = round(time.time() - start, 2)

        # Get matrices
        est_matrix = get_binary_matrix(results, ALPHA)
        adj_2d = get_adjacency_from_graph(results['graph'])

        # Evaluate both ways
        metrics_3d = evaluate_3d(true_matrix_3d, est_matrix)
        metrics_2d = evaluate_collapsed(true_matrix_2d, adj_2d)

        print(f"  Runtime: {runtime}s")
        print(f"  3D Eval: F1={metrics_3d['F1_3D']:.3f}, TP/FP/FN={metrics_3d['TP_3D']}/{metrics_3d['FP_3D']}/{metrics_3d['FN_3D']}")
        print(f"  2D Eval: F1={metrics_2d['F1_2D']:.3f}, TP/FP/FN={metrics_2d['TP_2D']}/{metrics_2d['FP_2D']}/{metrics_2d['FN_2D']}")

        all_results[algo['name']] = {
            'results': results,
            'pcmci': pcmci,
            'est_matrix': est_matrix,
            'adj_2d': adj_2d,
            'metrics_3d': metrics_3d,
            'metrics_2d': metrics_2d,
            'runtime': runtime
        }

        results_list.append({
            'Algorithm': algo['name'],
            'Runtime': runtime,
            **metrics_3d,
            **metrics_2d
        })

        sys.stdout.flush()

    # Create results DataFrame
    results_df = pd.DataFrame(results_list)

    # Print summary
    print("RESULTS SUMMARY")

    print("\n3D Evaluation (lag-specific, strict):")
    cols_3d = ['Algorithm', 'Precision_3D', 'Recall_3D', 'F1_3D', 'SHD_3D', 'TP_3D', 'FP_3D', 'FN_3D']
    print(results_df[cols_3d].to_string(index=False))

    print("\n2D Evaluation (collapsed, forgiving):")
    cols_2d = ['Algorithm', 'Precision_2D', 'Recall_2D', 'F1_2D', 'SHD_2D', 'TP_2D', 'FP_2D', 'FN_2D']
    print(results_df[cols_2d].to_string(index=False))

    # Best algorithms
    best_3d = results_df.loc[results_df['F1_3D'].idxmax()]
    best_2d = results_df.loc[results_df['F1_2D'].idxmax()]
    print(f"\nBest (3D strict): {best_3d['Algorithm']} with F1={best_3d['F1_3D']:.3f}")
    print(f"Best (2D collapsed): {best_2d['Algorithm']} with F1={best_2d['F1_2D']:.3f}")

    # Generate visualizations
    print("\nGenerating result visualizations...")
    plot_tigramite_results(all_results, TARGET_VARS, output_dir)
    plot_comparison_bars(results_df, output_dir)
    plot_adjacency_comparison(all_results, true_matrix_2d, TARGET_VARS, output_dir)
    print_discovered_edges(all_results, TARGET_VARS, output_dir)

    # Save results
    results_df.to_csv(f'{output_dir}/benchmark_results.csv', index=False)
    print(f"Saved: {output_dir}/benchmark_results.csv")

    print("BENCHMARK COMPLETE!")
    print(f"All outputs saved to: {output_dir}/")
    sys.stdout.flush()

    return results_df, all_results


# MAIN

def main():
    parser = argparse.ArgumentParser(description='Improved CausalChamber Benchmark')
    parser.add_argument('--csv_path', type=str, default='wt_combined_data.csv',
                        help='Path to wind tunnel CSV data')
    parser.add_argument('--output_dir', type=str, default='causalchamber_better',
                        help='Output directory')
    args = parser.parse_args()

    run_benchmark(csv_path=args.csv_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
