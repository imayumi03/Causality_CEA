# -*- coding: utf-8 -*-
"""
Causal Discovery for CausalChamber - UNDIRECTED Evaluation
===========================================================
Same as causalchamber_better.py but evaluates UNDIRECTED links.

Undirected evaluation:
- A -> B and B -> A are treated as the same edge (A -- B)
- More forgiving: if algorithm finds wrong direction, still counts as correct
- Useful when we care about "is there a relationship?" not "what's the direction?"

Example:
  Ground truth: hatch -> rpm_in
  Algorithm:    rpm_in -> hatch (wrong direction)

  Directed eval:   FP + FN (wrong)
  Undirected eval: TP (correct - relationship exists)
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


# CONFIGURATION

N_SAMPLES = 2000
TAU_MAX = 2
ALPHA = 0.05
TARGET_VARS = ['hatch', 'load_in', 'rpm_in', 'pressure_upwind']
GROUND_TRUTH_LAG = 1


# DATA LOADING

def load_wind_tunnel_data(csv_path="wt_combined_data.csv", n_samples=2000):
    """Load wind tunnel data from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    print(f"Loading data from: {csv_path}")
    df_full = pd.read_csv(csv_path)
    df = df_full[TARGET_VARS].iloc[:n_samples].copy()

    print(f"Loaded {len(df)} samples, {len(TARGET_VARS)} variables")
    return df


# GROUND TRUTH

def build_ground_truth(var_names, tau_max, causal_lag=1):
    """
    Build ground truth adjacency matrices (directed and undirected).

    Returns:
        true_matrix_3d: Directed 3D matrix (n_vars, n_vars, tau_max+1)
        true_matrix_2d_directed: Directed 2D matrix
        true_matrix_2d_undirected: Undirected 2D matrix (symmetric)
    """
    n_vars = len(var_names)
    var_idx = {v: i for i, v in enumerate(var_names)}

    # 3D directed matrix (considers lag time step + direction)
    true_matrix_3d = np.zeros((n_vars, n_vars, tau_max + 1), dtype=int)

    edges = [
        ('hatch', 'rpm_in', causal_lag),
        ('hatch', 'pressure_upwind', causal_lag),
        ('load_in', 'rpm_in', causal_lag),
        ('load_in', 'pressure_upwind', causal_lag),
    ]

    for cause, effect, lag in edges:
        if lag <= tau_max:
            true_matrix_3d[var_idx[cause], var_idx[effect], lag] = 1

    # 2D directed matrix (collapsed: meaning doesn't consider lag time step)
    true_matrix_2d_directed = np.zeros((n_vars, n_vars), dtype=int)
    true_matrix_2d_directed[var_idx['hatch'], var_idx['rpm_in']] = 1
    true_matrix_2d_directed[var_idx['hatch'], var_idx['pressure_upwind']] = 1
    true_matrix_2d_directed[var_idx['load_in'], var_idx['rpm_in']] = 1
    true_matrix_2d_directed[var_idx['load_in'], var_idx['pressure_upwind']] = 1

    # 2D UNDIRECTED matrix (symmetric)
    # Edge exists if A->B OR B->A
    true_matrix_2d_undirected = np.maximum(true_matrix_2d_directed,
                                            true_matrix_2d_directed.T)

    print(f"\nGround truth: 4 directed edges at lag {causal_lag}")
    for cause, effect, lag in edges:
        print(f"  {cause} -> {effect} (lag {lag})")

    print(f"\nUndirected ground truth: 4 undirected edges")
    print("  hatch -- rpm_in")
    print("  hatch -- pressure_upwind")
    print("  load_in -- rpm_in")
    print("  load_in -- pressure_upwind")

    return true_matrix_3d, true_matrix_2d_directed, true_matrix_2d_undirected


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
    """Run causal discovery."""
    if ci_test == 'robust_parcorr':
        cond_ind_test = RobustParCorr(significance='analytic')
    elif ci_test == 'cmiknn':
        cond_ind_test = CMIknn(
            knn=10,
            shuffle_neighbors=100,
            significance='shuffle_test',
            transform='ranks'
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


# EVALUATION - DIRECTED AND UNDIRECTED

def get_adjacency_from_graph(graph):
    """Convert Tigramite 3D graph to 2D adjacency (collapsed over lags)."""
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


def to_undirected(matrix):
    """
    Convert directed matrix to undirected (symmetric).

    Edge exists if A->B OR B->A.
    """
    return np.maximum(matrix, matrix.T)


def get_upper_triangle(matrix, exclude_diagonal=True):
    """
    Get upper triangle of matrix as flat array.

    For undirected evaluation, we only look at upper triangle
    to avoid counting each edge twice.
    """
    n = matrix.shape[0]
    if exclude_diagonal:
        indices = np.triu_indices(n, k=1)
    else:
        indices = np.triu_indices(n, k=0)
    return matrix[indices]


def evaluate_directed(true_matrix, pred_matrix, exclude_self=True):
    """Evaluate DIRECTED edges (A->B different from B->A)."""
    n = true_matrix.shape[0]

    if exclude_self:
        mask = ~np.eye(n, dtype=bool)
        y_true = true_matrix[mask].flatten()
        y_pred = pred_matrix[mask].flatten()
    else:
        y_true = true_matrix.flatten()
        y_pred = pred_matrix.flatten()

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    shd = FP + FN

    return {
        'Precision_dir': round(precision, 4),
        'Recall_dir': round(recall, 4),
        'F1_dir': round(f1, 4),
        'SHD_dir': shd,
        'TP_dir': TP, 'FP_dir': FP, 'FN_dir': FN
    }


def evaluate_undirected(true_matrix_undir, pred_matrix_undir):
    """
    Evaluate UNDIRECTED edges (A--B same as B--A).

    Only looks at upper triangle to avoid double counting.
    """
    # Get upper triangle (excluding diagonal)
    y_true = get_upper_triangle(true_matrix_undir)
    y_pred = get_upper_triangle(pred_matrix_undir)

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    shd = FP + FN

    return {
        'Precision_undir': round(precision, 4),
        'Recall_undir': round(recall, 4),
        'F1_undir': round(f1, 4),
        'SHD_undir': shd,
        'TP_undir': TP, 'FP_undir': FP, 'FN_undir': FN
    }


# VISUALIZATION

def plot_comparison_bars(results_df, output_dir):
    """Plot bar charts comparing directed vs undirected evaluation."""

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    algorithms = results_df['Algorithm'].values
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    # Row 1: Directed evaluation
    metrics_dir = ['Precision_dir', 'Recall_dir', 'F1_dir', 'SHD_dir']
    titles_dir = ['Precision (Directed)', 'Recall (Directed)',
                  'F1 (Directed)', 'SHD (Directed)']

    for ax, metric, title, color in zip(axes[0], metrics_dir, titles_dir, colors):
        values = results_df[metric].values
        bars = ax.bar(range(len(algorithms)), values, color=color, edgecolor='white')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels([a.replace('_', '\n') for a in algorithms], fontsize=8)

        if 'SHD' not in metric:
            ax.set_ylim(0, 1.15)

        for bar, val in zip(bars, values):
            label = f'{val:.2f}' if 'SHD' not in metric else f'{int(val)}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Row 2: Undirected evaluation
    metrics_undir = ['Precision_undir', 'Recall_undir', 'F1_undir', 'SHD_undir']
    titles_undir = ['Precision (Undirected)', 'Recall (Undirected)',
                    'F1 (Undirected)', 'SHD (Undirected)']

    for ax, metric, title, color in zip(axes[1], metrics_undir, titles_undir, colors):
        values = results_df[metric].values
        bars = ax.bar(range(len(algorithms)), values, color=color, edgecolor='white')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels([a.replace('_', '\n') for a in algorithms], fontsize=8)

        if 'SHD' not in metric:
            ax.set_ylim(0, 1.15)

        for bar, val in zip(bars, values):
            label = f'{val:.2f}' if 'SHD' not in metric else f'{int(val)}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Algorithm Performance: Directed vs Undirected Evaluation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_directed_vs_undirected.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/comparison_directed_vs_undirected.png")


def plot_adjacency_comparison(all_results, true_dir, true_undir, var_names, output_dir):
    """Plot directed and undirected adjacency matrices."""

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))

    # Row 1: Directed
    axes[0, 0].imshow(true_dir, cmap='Greens', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth\n(Directed)', fontsize=10, fontweight='bold')
    _add_matrix_labels(axes[0, 0], true_dir, var_names)

    algo_names = ['PCMCI_RobustParCorr', 'PCMCI+_RobustParCorr',
                  'PCMCI_CMIknn', 'PCMCI+_CMIknn']

    for idx, algo_name in enumerate(algo_names):
        ax = axes[0, idx + 1]
        if algo_name in all_results:
            adj = all_results[algo_name]['adj_2d_directed']
            _plot_colored_matrix(ax, adj, true_dir, var_names)
            ax.set_title(f'{algo_name.replace("_", chr(10))}\n(Directed)',
                        fontsize=9, fontweight='bold')

    # Row 2: Undirected
    axes[1, 0].imshow(true_undir, cmap='Greens', vmin=0, vmax=1)
    axes[1, 0].set_title('Ground Truth\n(Undirected)', fontsize=10, fontweight='bold')
    _add_matrix_labels(axes[1, 0], true_undir, var_names)

    for idx, algo_name in enumerate(algo_names):
        ax = axes[1, idx + 1]
        if algo_name in all_results:
            adj = all_results[algo_name]['adj_2d_undirected']
            _plot_colored_matrix(ax, adj, true_undir, var_names)
            ax.set_title(f'{algo_name.replace("_", chr(10))}\n(Undirected)',
                        fontsize=9, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#33cc33', label='TP (Correct)'),
        Patch(facecolor='#e63333', label='FP (Spurious)'),
        Patch(facecolor='#3366e6', label='FN (Missed)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Adjacency Matrices: Directed (top) vs Undirected (bottom)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/adjacency_directed_vs_undirected.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/adjacency_directed_vs_undirected.png")


def _add_matrix_labels(ax, matrix, var_names):
    """Add labels to matrix plot."""
    n = len(var_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(var_names, fontsize=7, rotation=45, ha='right')
    ax.set_yticklabels(var_names, fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                   fontsize=10, fontweight='bold')


def _plot_colored_matrix(ax, pred, true, var_names):
    """Plot matrix with TP/FP/FN coloring."""
    n = len(var_names)
    colored = np.zeros((n, n, 3))

    for i in range(n):
        for j in range(n):
            if true[i, j] == 1 and pred[i, j] == 1:
                colored[i, j] = [0.2, 0.8, 0.2]  # Green - TP
            elif true[i, j] == 0 and pred[i, j] == 1:
                colored[i, j] = [0.9, 0.2, 0.2]  # Red - FP
            elif true[i, j] == 1 and pred[i, j] == 0:
                colored[i, j] = [0.2, 0.4, 0.9]  # Blue - FN
            else:
                colored[i, j] = [0.95, 0.95, 0.95]  # Light gray - TN

    ax.imshow(colored)
    _add_matrix_labels(ax, pred, var_names)


def plot_tigramite_results(all_results, var_names, output_dir):
    """Plot using Tigramite's built-in visualization."""

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

    plt.suptitle('Causal Discovery Results: Wind Tunnel Data',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tigramite_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/tigramite_results.png")


# MAIN BENCHMARK

def run_benchmark(csv_path='wt_combined_data.csv', output_dir='causalchamber_undirected'):
    """Run benchmark with both directed and undirected evaluation."""

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("CAUSAL DISCOVERY: DIRECTED vs UNDIRECTED EVALUATION")
    print("=" * 70)
    print(f"Data: {csv_path}")
    print(f"Samples: {N_SAMPLES}")
    print(f"tau_max: {TAU_MAX}")
    print(f"alpha: {ALPHA}")
    print("=" * 70)
    print("\nEVALUATION MODES:")
    print("  Directed:   A->B different from B->A")
    print("  Undirected: A->B same as B->A (symmetric)")
    print("=" * 70)
    sys.stdout.flush()

    # Load data
    print("\nLoading data...")
    df = load_wind_tunnel_data(csv_path, N_SAMPLES)
    dataframe = pp.DataFrame(df.values, var_names=TARGET_VARS)

    # Build ground truth (directed and undirected)
    print("\nBuilding ground truth...")
    true_3d, true_2d_dir, true_2d_undir = build_ground_truth(
        TARGET_VARS, TAU_MAX, GROUND_TRUTH_LAG)

    print("\nDirected Ground Truth Matrix:")
    print(pd.DataFrame(true_2d_dir, index=TARGET_VARS, columns=TARGET_VARS))

    print("\nUndirected Ground Truth Matrix (symmetric):")
    print(pd.DataFrame(true_2d_undir, index=TARGET_VARS, columns=TARGET_VARS))

    # Run algorithms
    print("\n" + "=" * 70)
    print("RUNNING CAUSAL DISCOVERY ALGORITHMS")
    print("=" * 70)
    sys.stdout.flush()

    algorithms = get_algorithms()
    all_results = {}
    results_list = []

    for algo in algorithms:
        print(f"\n{algo['name']}...")
        print(f"  Method: {algo['method']}, CI Test: {algo['ci_test']}")
        sys.stdout.flush()

        start = time.time()
        results, pcmci = run_discovery(dataframe, algo['method'], algo['ci_test'],
                                        TAU_MAX, ALPHA)
        runtime = round(time.time() - start, 2)

        # Get matrices
        est_matrix = get_binary_matrix(results, ALPHA)
        adj_2d_dir = get_adjacency_from_graph(results['graph'])
        adj_2d_undir = to_undirected(adj_2d_dir)

        # Evaluate both ways
        metrics_dir = evaluate_directed(true_2d_dir, adj_2d_dir)
        metrics_undir = evaluate_undirected(true_2d_undir, adj_2d_undir)

        print(f"  Runtime: {runtime}s")
        print(f"  Directed:   F1={metrics_dir['F1_dir']:.3f}, "
              f"TP/FP/FN={metrics_dir['TP_dir']}/{metrics_dir['FP_dir']}/{metrics_dir['FN_dir']}")
        print(f"  Undirected: F1={metrics_undir['F1_undir']:.3f}, "
              f"TP/FP/FN={metrics_undir['TP_undir']}/{metrics_undir['FP_undir']}/{metrics_undir['FN_undir']}")

        all_results[algo['name']] = {
            'results': results,
            'pcmci': pcmci,
            'est_matrix': est_matrix,
            'adj_2d_directed': adj_2d_dir,
            'adj_2d_undirected': adj_2d_undir,
            'metrics_dir': metrics_dir,
            'metrics_undir': metrics_undir,
            'runtime': runtime
        }

        results_list.append({
            'Algorithm': algo['name'],
            'Runtime': runtime,
            **metrics_dir,
            **metrics_undir
        })

        sys.stdout.flush()

    # Create results DataFrame
    results_df = pd.DataFrame(results_list)

    # Print summary

    print("RESULTS SUMMARY")

    print("\nDIRECTED Evaluation (A->B ≠ B->A):")
    cols_dir = ['Algorithm', 'Precision_dir', 'Recall_dir', 'F1_dir',
                'SHD_dir', 'TP_dir', 'FP_dir', 'FN_dir']
    print(results_df[cols_dir].to_string(index=False))

    print("\nUNDIRECTED Evaluation (A->B = B->A):")
    cols_undir = ['Algorithm', 'Precision_undir', 'Recall_undir', 'F1_undir',
                  'SHD_undir', 'TP_undir', 'FP_undir', 'FN_undir']
    print(results_df[cols_undir].to_string(index=False))

    # Best algorithms
    best_dir = results_df.loc[results_df['F1_dir'].idxmax()]
    best_undir = results_df.loc[results_df['F1_undir'].idxmax()]
    print(f"\nBest (Directed):   {best_dir['Algorithm']} with F1={best_dir['F1_dir']:.3f}")
    print(f"Best (Undirected): {best_undir['Algorithm']} with F1={best_undir['F1_undir']:.3f}")

    # F1 improvement from directed to undirected
    print("\nF1 Improvement (Undirected vs Directed):")
    for _, row in results_df.iterrows():
        improvement = row['F1_undir'] - row['F1_dir']
        print(f"  {row['Algorithm']}: {improvement:+.3f}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_tigramite_results(all_results, TARGET_VARS, output_dir)
    plot_comparison_bars(results_df, output_dir)
    plot_adjacency_comparison(all_results, true_2d_dir, true_2d_undir,
                              TARGET_VARS, output_dir)

    # Save results
    results_df.to_csv(f'{output_dir}/benchmark_results.csv', index=False)
    print(f"Saved: {output_dir}/benchmark_results.csv")


    print("BENCHMARK COMPLETE!")
    print(f"All outputs saved to: {output_dir}/")
    sys.stdout.flush()

    return results_df, all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='CausalChamber - Undirected Evaluation')
    parser.add_argument('--csv_path', type=str, default='wt_combined_data.csv',
                        help='Path to wind tunnel CSV data')
    parser.add_argument('--output_dir', type=str, default='causalchamber_undirected',
                        help='Output directory')
    args = parser.parse_args()

    run_benchmark(csv_path=args.csv_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
