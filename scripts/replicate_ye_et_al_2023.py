"""
Replication of experiments from:
  "Analyzing Robustness of Angluin's L* Algorithm in Presence of Noise"
  Ye et al., Logical Methods in Computer Science, Vol 20(1), 2024.
  https://arxiv.org/pdf/2306.08266

Uses the authors' released code from:
  https://github.com/LeaRNNify/Noisy_Learning

Run from repo root:
  python scripts/replicate_ye_et_al_2023.py [--table N] [--num-benchmarks N]
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

# Add the paper's source to the path
PAPER_CODE_DIR = "/tmp/noisy_learning/source"
if not os.path.isdir(PAPER_CODE_DIR):
    print(
        "Paper code not found. Clone it first:\n"
        "  git clone https://github.com/LeaRNNify/Noisy_Learning.git /tmp/noisy_learning\n"
        "  cd /tmp/noisy_learning/source && python setup.py build_ext --inplace"
    )
    sys.exit(1)
sys.path.insert(0, PAPER_CODE_DIR)

from benchmarking_noisy_dfa import BenchmarkingNoise
from benchmarking_subsuper_dfa import BenchmarkingSubSuper
from counter_dfa import CounterDFA
from dfa import DFANoisy, DFAsubSuper
from noisy_input_dfa import NoisyInputDFA

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "ye_et_al_2023"
)


def ensure_results_dir(subdir=""):
    path = os.path.join(RESULTS_DIR, subdir) if subdir else RESULTS_DIR
    os.makedirs(path, exist_ok=True)
    return path


# ── Table 1: Impact of ε and δ ──────────────────────────────────────────────
def run_table1(num_benchmarks=35):
    """
    Table 1: Evaluation of the impact of ε and δ.
    35 DFAs, 5 noise levels, 5 ε=δ values.
    """
    print("\n" + "=" * 70)
    print("TABLE 1: Impact of ε and δ")
    print("=" * 70)

    save_dir = ensure_results_dir("table1")
    eps_values = [0.05, 0.01, 0.005, 0.001, 0.0005]
    p_noise_values = [0.01, 0.005, 0.0025, 0.0015, 0.001]

    all_results = []
    for eps in eps_values:
        np.random.seed(seed=2)
        bench = BenchmarkingNoise(
            pac_epsilons=(eps,),
            pac_deltas=(eps,),
            p_noise=p_noise_values,
            max_eq=250,
            min_dfa_state=20,
            max_dfa_states=60,
            min_alphabet_size=4,
            max_alphabet_size=20,
        )
        bench.benchmarks_noise_model(
            num_benchmarks, save_dir=os.path.join(save_dir, f"eps_{eps}")
        )

    print(f"Table 1 results saved to {save_dir}")


# ── Table 2: Noisy output ────────────────────────────────────────────────────
def run_table2(num_benchmarks=50):
    """
    Table 2: Evaluation of the algorithm w.r.t. the noisy output.
    50 DFAs with noisy output L(A→p) for p ∈ {0.01, 0.005, 0.0025, 0.0015, 0.001}.
    """
    print("\n" + "=" * 70)
    print("TABLE 2: DFA with noisy output (A→p)")
    print("=" * 70)

    save_dir = ensure_results_dir("table2")
    np.random.seed(seed=2)

    bench = BenchmarkingNoise(
        pac_epsilons=(0.005,),
        pac_deltas=(0.005,),
        p_noise=[0.01, 0.005, 0.0025, 0.0015, 0.001],
        dfa_noise=DFANoisy,
        max_eq=250,
        min_dfa_state=20,
        max_dfa_states=60,
        min_alphabet_size=4,
        max_alphabet_size=20,
    )
    bench.benchmarks_noise_model(num_benchmarks, save_dir=save_dir)
    print(f"Table 2 results saved to {save_dir}")


# ── Table 3: Noisy input ────────────────────────────────────────────────────
def run_table3(num_benchmarks=45):
    """
    Table 3: Evaluation of the algorithm w.r.t. the noisy input.
    45 DFAs with noisy input L(A←p) for p ∈ {1e-4, 5e-4, 1e-3, 5e-3}.
    """
    print("\n" + "=" * 70)
    print("TABLE 3: DFA with noisy input (A←p)")
    print("=" * 70)

    save_dir = ensure_results_dir("table3")
    np.random.seed(seed=2)

    bench = BenchmarkingNoise(
        pac_epsilons=(0.005,),
        pac_deltas=(0.005,),
        p_noise=[0.005, 0.001, 0.0005, 0.0001],
        dfa_noise=NoisyInputDFA,
        max_eq=250,
        min_dfa_state=20,
        max_dfa_states=60,
        min_alphabet_size=4,
        max_alphabet_size=20,
    )
    bench.benchmarks_noise_model(num_benchmarks, save_dir=save_dir)
    print(f"Table 3 results saved to {save_dir}")


# ── Table 4: Counter DFA ────────────────────────────────────────────────────
def run_table4(num_benchmarks=160):
    """
    Table 4: Evaluation of the algorithm w.r.t. the 'noisy' counter.
    160 DFAs with counter DFA noise.
    """
    print("\n" + "=" * 70)
    print("TABLE 4: Counter DFA")
    print("=" * 70)

    save_dir = ensure_results_dir("table4")
    np.random.seed(seed=2)

    bench = BenchmarkingNoise(
        pac_epsilons=(0.005,),
        pac_deltas=(0.005,),
        p_noise=[1],  # dummy value; counter DFA noise doesn't use p_noise
        dfa_noise=CounterDFA,
        max_eq=250,
        min_dfa_state=20,
        max_dfa_states=60,
        min_alphabet_size=4,
        max_alphabet_size=20,
    )
    bench.benchmarks_noise_model(num_benchmarks, save_dir=save_dir)
    print(f"Table 4 results saved to {save_dir}")


# ── Table 5: Pathological behaviors ─────────────────────────────────────────
def run_table5(num_benchmarks=300):
    """
    Table 5: Elimination of pathological behaviors.
    ~300 pairs (A, A+).
    """
    print("\n" + "=" * 70)
    print("TABLE 5: Pathological behaviors (A^n)")
    print("=" * 70)

    save_dir = ensure_results_dir("table5")
    np.random.seed(seed=2)

    bench = BenchmarkingSubSuper(
        pac_epsilons=(0.005,),
        pac_deltas=(0.005,),
        p_noise=[0.005, 0.001, 0.0005, 0.0001],
        dfa_noise=DFAsubSuper,
        min_dfa_state=21,
        max_dfa_states=60,
        min_alphabet_size=5,
        max_alphabet_size=15,
    )
    bench.benchmarks_subsuper_model(num_benchmarks, save_dir=save_dir)
    print(f"Table 5 results saved to {save_dir}")


# ── Table 6: Word distribution ──────────────────────────────────────────────
def run_table6(num_benchmarks=22):
    """
    Table 6: Analysis of different distributions on Σ*.
    For each (p, μ) pair, run experiments on 22+ DFAs.
    μ ∈ {0.001, 0.005, 0.01, 0.05, 0.1}
    p ∈ {0.01, 0.005, 0.0025, 0.0015, 0.001}
    """
    print("\n" + "=" * 70)
    print("TABLE 6: Word distribution analysis")
    print("=" * 70)

    save_dir = ensure_results_dir("table6")
    mu_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    p_noise_values = [0.01, 0.005, 0.0025, 0.0015, 0.001]

    for mu in mu_values:
        np.random.seed(seed=2)
        bench = BenchmarkingNoise(
            pac_epsilons=(0.005,),
            pac_deltas=(0.005,),
            word_probs=(mu,),
            p_noise=p_noise_values,
            dfa_noise=DFANoisy,
            max_eq=250,
            min_dfa_state=20,
            max_dfa_states=60,
            min_alphabet_size=4,
            max_alphabet_size=20,
        )
        bench.benchmarks_noise_model(
            num_benchmarks, save_dir=os.path.join(save_dir, f"mu_{mu}")
        )

    print(f"Table 6 results saved to {save_dir}")


# ── Table summary printer ───────────────────────────────────────────────────
def print_summary(results_dir):
    """Read and print saved CSV summaries."""
    for csv_file in sorted(
        f for f in os.listdir(results_dir) if f.endswith("_summary.csv")
    ):
        path = os.path.join(results_dir, csv_file)
        df = pd.read_csv(path)
        print(f"\n{csv_file}:")
        print(df.to_string(index=False))


# ── Main ────────────────────────────────────────────────────────────────────
TABLE_RUNNERS = {
    1: run_table1,
    2: run_table2,
    3: run_table3,
    4: run_table4,
    5: run_table5,
    6: run_table6,
}

# Paper's benchmark counts per table
PAPER_COUNTS = {
    1: 35,
    2: 50,
    3: 45,
    4: 160,
    5: 300,
    6: 22,
}


def main():
    parser = argparse.ArgumentParser(
        description="Replicate experiments from Ye et al. 2023"
    )
    parser.add_argument(
        "--table",
        type=int,
        choices=list(TABLE_RUNNERS.keys()),
        help="Run a specific table (default: all)",
    )
    parser.add_argument(
        "--num-benchmarks",
        type=int,
        default=None,
        help="Override number of benchmarks per table (default: paper values)",
    )
    args = parser.parse_args()

    ensure_results_dir()

    if args.table:
        tables = [args.table]
    else:
        tables = sorted(TABLE_RUNNERS.keys())

    for t in tables:
        n = args.num_benchmarks or PAPER_COUNTS[t]
        start = time.time()
        print(f"\n{'#' * 70}")
        print(f"# Running Table {t} with {n} benchmarks")
        print(f"{'#' * 70}")
        TABLE_RUNNERS[t](n)
        elapsed = time.time() - start
        print(f"Table {t} completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
