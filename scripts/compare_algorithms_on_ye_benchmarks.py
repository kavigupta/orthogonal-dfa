"""
Compare orthogonal L* vs paper's PAC L* (KV) on the Ye et al. noisy DFA benchmarks.

Uses the paper's code for:
  - Random DFA generation
  - Noise models (DFANoisy, NoisyInputDFA, CounterDFA)
  - Distance estimation (Chernoff-Hoeffding)
  - PAC L* baseline

Uses our codebase for:
  - Orthogonal L* algorithm
  - Baseline L* (aalpy wrapper)

Run:
  python scripts/compare_algorithms_on_ye_benchmarks.py [--num-benchmarks N] [--noise-type noisy_output|noisy_input|counter]
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

# ── Paper code imports ───────────────────────────────────────────────────────
PAPER_CODE_DIR = "/tmp/noisy_learning/source"
if not os.path.isdir(PAPER_CODE_DIR):
    print(
        "Paper code not found. Clone it first:\n"
        "  git clone https://github.com/LeaRNNify/Noisy_Learning.git /tmp/noisy_learning\n"
        "  cd /tmp/noisy_learning/source && python setup.py build_ext --inplace"
    )
    sys.exit(1)
sys.path.insert(0, PAPER_CODE_DIR)

from benchmarking_noisy_dfa import minimize_dfa as ye_minimize_dfa
from counter_dfa import CounterDFA
from dfa import DFA as YeDFA
from dfa import DFANoisy, random_dfa as ye_random_dfa
from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from noisy_input_dfa import NoisyInputDFA
from pac_teacher import PACTeacher
from random_words import confidence_interval_many_cython

# ── Our code imports ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orthogonal_dfa.baseline_lstar.baseline_lstar import run_baseline_lstar
from orthogonal_dfa.l_star.lstar import do_counterexample_driven_synthesis
from orthogonal_dfa.l_star.prefix_suffix_tracker import (
    PrefixSuffixTracker,
    SearchConfig,
)
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.statistics import (
    compute_prefix_set_size,
    compute_suffix_size_counterexample_gen,
    population_size_and_evidence_margin,
)
from orthogonal_dfa.l_star.structures import Oracle, SymmetricBernoulli

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "comparison"
)


# ── Oracle adapter: wrap a paper DFA (noisy or clean) for our L* ─────────
class YeDFAOracle(Oracle):
    """Wraps a Ye et al. DFA/NoisyDFA as an Oracle for orthogonal L*."""

    def __init__(self, ye_dfa, alphabet_size_int):
        self._ye_dfa = ye_dfa
        self._alphabet_size = alphabet_size_int
        self._alphabet = ye_dfa.alphabet  # e.g. ['a', 'b', 'c', ...]

    @property
    def alphabet_size(self):
        return self._alphabet_size

    def membership_query(self, string):
        # Convert int list -> letter tuple for the paper's DFA
        word = tuple(self._alphabet[i] for i in string)
        return self._ye_dfa.is_word_in(word)


# ── Convert our automata-lib DFA back to paper format ────────────────────
def our_dfa_to_ye_dfa(dfa, alphabet_letters):
    """Convert automata-lib DFA (int symbols) to Ye et al. DFA (letter symbols)."""
    transitions = {}
    for state in dfa.states:
        transitions[state] = {}
        for sym_int, letter in enumerate(alphabet_letters):
            transitions[state][letter] = dfa.transitions[state][sym_int]

    return YeDFA(dfa.initial_state, dfa.final_states, transitions)


# ── Run paper's PAC L* ───────────────────────────────────────────────────
def run_ye_pac_lstar(noisy_dfa, epsilon=0.005, delta=0.005, word_prob=0.01):
    """Run the paper's PAC L* (KV algorithm) on a noisy DFA."""
    teacher = PACTeacher(noisy_dfa, epsilon, delta, word_probability=word_prob)
    student = DecisionTreeLearner(teacher)
    teacher.teach_fix_round(student)
    extracted = ye_minimize_dfa(student.dfa)
    return extracted, teacher.num_equivalence_asked


# ── Run our orthogonal L* ────────────────────────────────────────────────
def run_orthogonal_lstar(
    noisy_dfa, alphabet_letters, signal_strength, seed=0
):
    """Run orthogonal L* on a noisy DFA via the Oracle adapter."""
    alphabet_size = len(alphabet_letters)
    oracle = YeDFAOracle(noisy_dfa, alphabet_size)

    effective_p_acc = 0.5 + signal_strength
    n, eps = population_size_and_evidence_margin(
        signal_strength=signal_strength,
        acceptable_fpr=0.01,
        acceptable_fnr=0.01,
    )
    suffix_size = compute_suffix_size_counterexample_gen(0.01, effective_p_acc)

    min_suffix_freq = 0.001

    config = SearchConfig(
        suffix_family_size=n,
        evidence_margin=eps,
        decision_rule_fpr=0.01,
        suffix_size_counterexample_gen=suffix_size,
        min_signal_strength=signal_strength,
        num_addtl_prefixes=200,
        min_suffix_frequency=min_suffix_freq,
    )

    # Use longer samples to compensate for large alphabets — random
    # strings need to be long enough to exercise DFA transitions.
    sampler = UniformSampler(60)
    pst = PrefixSuffixTracker.create(
        sampler,
        np.random.default_rng(seed),
        oracle,
        config,
        num_prefixes=300,
    )

    dfa, dt = do_counterexample_driven_synthesis(
        pst, additional_counterexamples=300, acc_threshold=0.98
    )

    # Convert back to paper format for distance measurement
    ye_dfa = our_dfa_to_ye_dfa(dfa, alphabet_letters)
    return ye_dfa, len(dfa.states)


# ── Run our baseline L* (aalpy) ──────────────────────────────────────────
def run_our_baseline_lstar(noisy_dfa, alphabet_letters, max_states=None):
    """Run baseline L* (aalpy wrapper) on a noisy DFA."""
    alphabet_size = len(alphabet_letters)
    oracle = YeDFAOracle(noisy_dfa, alphabet_size)
    dfa = run_baseline_lstar(oracle, max_states=max_states)
    ye_dfa = our_dfa_to_ye_dfa(dfa, alphabet_letters)
    return ye_dfa, len(dfa.states)


# ── Distance measurement using paper's code ──────────────────────────────
def measure_distances(models, epsilon=0.005, word_prob=0.01):
    """Compute pairwise distances using the paper's Chernoff-Hoeffding code."""
    output, _ = confidence_interval_many_cython(
        models, width=epsilon, confidence=epsilon, word_prob=word_prob
    )
    return output


# ── Single benchmark ─────────────────────────────────────────────────────
def run_single_benchmark(
    dfa_original,
    noisy_dfa,
    noise_type,
    p_noise,
    signal_strength,
    seed,
):
    """Run all algorithms on one (original DFA, noisy DFA) pair."""
    alphabet_letters = dfa_original.alphabet
    result = {
        "dfa_states": len(dfa_original.states),
        "alphabet_size": len(alphabet_letters),
        "noise_type": noise_type,
        "p_noise": p_noise,
        "signal_strength": signal_strength,
    }

    # 1) Paper's PAC L*
    t0 = time.time()
    try:
        ye_extracted, ye_rounds = run_ye_pac_lstar(noisy_dfa)
        result["ye_time"] = time.time() - t0
        result["ye_states"] = len(ye_extracted.states)
        result["ye_rounds"] = ye_rounds
    except Exception as e:
        logging.warning(f"Paper L* failed: {e}")
        ye_extracted = None
        result["ye_time"] = time.time() - t0
        result["ye_states"] = -1
        result["ye_rounds"] = -1

    # 2) Our orthogonal L*
    t0 = time.time()
    try:
        orth_extracted, orth_states = run_orthogonal_lstar(
            noisy_dfa, alphabet_letters, signal_strength, seed=seed
        )
        result["orth_time"] = time.time() - t0
        result["orth_states"] = orth_states
    except Exception as e:
        logging.warning(f"Orthogonal L* failed: {e}")
        orth_extracted = None
        result["orth_time"] = time.time() - t0
        result["orth_states"] = -1

    # 3) Our baseline L* (aalpy)
    t0 = time.time()
    try:
        baseline_extracted, baseline_states = run_our_baseline_lstar(
            noisy_dfa, alphabet_letters, max_states=50
        )
        result["baseline_time"] = time.time() - t0
        result["baseline_states"] = baseline_states
    except Exception as e:
        logging.warning(f"Baseline L* failed: {e}")
        baseline_extracted = None
        result["baseline_time"] = time.time() - t0
        result["baseline_states"] = -1

    # 4) Measure distances
    models = [dfa_original, noisy_dfa]
    labels = ["original", "noisy"]
    extracted_models = []
    extracted_labels = []

    if ye_extracted is not None:
        extracted_models.append(ye_extracted)
        extracted_labels.append("ye")
    if orth_extracted is not None:
        extracted_models.append(orth_extracted)
        extracted_labels.append("orth")
    if baseline_extracted is not None:
        extracted_models.append(baseline_extracted)
        extracted_labels.append("baseline")

    all_models = models + extracted_models
    try:
        dist_matrix, _ = confidence_interval_many_cython(
            all_models, width=0.005, confidence=0.005, word_prob=0.01
        )
        # dist_matrix[i][j] = distance between model i and model j
        result["dist_orig_noisy"] = dist_matrix[0][1]

        for idx, label in enumerate(extracted_labels):
            model_idx = 2 + idx
            result[f"dist_orig_{label}"] = dist_matrix[0][model_idx]
            result[f"dist_noisy_{label}"] = dist_matrix[1][model_idx]
            if dist_matrix[0][model_idx] > 0:
                result[f"gain_{label}"] = (
                    dist_matrix[0][1] / dist_matrix[0][model_idx]
                )
            else:
                result[f"gain_{label}"] = float("inf")
    except Exception as e:
        logging.warning(f"Distance measurement failed: {e}")

    return result


# ── Main benchmarking loop ───────────────────────────────────────────────
def run_noisy_output_benchmarks(
    num_benchmarks=10,
    p_noise_values=(0.01, 0.005, 0.0025, 0.0015, 0.001),
    min_states=20,
    max_states=60,
    min_alphabet=4,
    max_alphabet=20,
):
    """Run benchmarks for DFA with noisy output."""
    results = []

    for bench_idx in range(num_benchmarks):
        logging.info(f"Benchmark {bench_idx + 1}/{num_benchmarks}")

        # Generate and minimize a random DFA (same as paper)
        full_alphabet = "abcdefghijklmnopqrstuvwxyz"
        alph_size = np.random.randint(min_alphabet, max_alphabet)
        alphabet = full_alphabet[:alph_size]

        while True:
            dfa_rand = ye_random_dfa(alphabet, min_state=min_states, max_states=max_states)
            dfa = ye_minimize_dfa(dfa_rand)
            if len(dfa.states) > min_states:
                break

        logging.info(
            f"  DFA: {len(dfa.states)} states, alphabet size {len(dfa.alphabet)}"
        )

        for p in p_noise_values:
            logging.info(f"  Noise p={p}")

            # Create noisy DFA (same as paper)
            noisy_dfa = DFANoisy(
                dfa.init_state, dfa.final_states, dfa.transitions,
                mistake_prob=p,
            )

            # # Signal strength for noisy output: p_correct = 1 - p
            # signal_strength = 0.5 - p

            result = run_single_benchmark(
                dfa, noisy_dfa, "noisy_output", p, 0.05,
                seed=bench_idx,
            )
            results.append(result)

            # Print interim result
            gains = []
            for algo in ["ye", "orth", "baseline"]:
                g = result.get(f"gain_{algo}", "N/A")
                if isinstance(g, float) and g != float("inf"):
                    gains.append(f"{algo}={g:.3f}")
                else:
                    gains.append(f"{algo}={g}")
            logging.info(f"    Gains: {', '.join(gains)}")

    return pd.DataFrame(results)


def run_noisy_input_benchmarks(
    num_benchmarks=10,
    p_noise_values=(0.005, 0.001, 0.0005, 0.0001),
    min_states=20,
    max_states=60,
    min_alphabet=4,
    max_alphabet=20,
):
    """Run benchmarks for DFA with noisy input."""
    results = []

    for bench_idx in range(num_benchmarks):
        logging.info(f"Benchmark {bench_idx + 1}/{num_benchmarks}")

        full_alphabet = "abcdefghijklmnopqrstuvwxyz"
        alph_size = np.random.randint(min_alphabet, max_alphabet)
        alphabet = full_alphabet[:alph_size]

        while True:
            dfa_rand = ye_random_dfa(alphabet, min_state=min_states, max_states=max_states)
            dfa = ye_minimize_dfa(dfa_rand)
            if len(dfa.states) > min_states:
                break

        logging.info(
            f"  DFA: {len(dfa.states)} states, alphabet size {len(dfa.alphabet)}"
        )

        for p in p_noise_values:
            logging.info(f"  Noise p={p}")

            noisy_dfa = NoisyInputDFA(
                dfa.init_state, dfa.final_states, dfa.transitions,
                mistake_prob=p,
            )

            # For noisy input, signal strength is harder to estimate.
            # The effective error rate depends on DFA structure and p.
            # Use a conservative estimate: treat it like noisy output
            # with somewhat lower signal (input noise is harder).
            signal_strength = max(0.5 - p * 10, 0.1)

            result = run_single_benchmark(
                dfa, noisy_dfa, "noisy_input", p, signal_strength,
                seed=bench_idx,
            )
            results.append(result)

            gains = []
            for algo in ["ye", "orth", "baseline"]:
                g = result.get(f"gain_{algo}", "N/A")
                if isinstance(g, float) and g != float("inf"):
                    gains.append(f"{algo}={g:.3f}")
                else:
                    gains.append(f"{algo}={g}")
            logging.info(f"    Gains: {', '.join(gains)}")

    return pd.DataFrame(results)


def summarize_results(df, algo_labels=("ye", "orth", "baseline")):
    """Print a comparison table grouped by noise level."""
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)

    # Group by p_noise
    for p in sorted(df["p_noise"].unique(), reverse=True):
        sub = df[df["p_noise"] == p]
        print(f"\np = {p} ({len(sub)} benchmarks)")
        print(f"  {'Algorithm':<15} {'d(A,A_E)':>12} {'d(noisy,A_E)':>14} {'gain':>10} {'states':>8} {'time(s)':>8}")
        print(f"  {'-'*13:<15} {'-'*12:>12} {'-'*14:>14} {'-'*10:>10} {'-'*8:>8} {'-'*8:>8}")

        for algo in algo_labels:
            dist_col = f"dist_orig_{algo}"
            noisy_col = f"dist_noisy_{algo}"
            gain_col = f"gain_{algo}"
            states_col = f"{algo}_states"
            time_col = f"{algo}_time"

            if dist_col not in sub.columns:
                continue

            valid = sub[dist_col].notna()
            if not valid.any():
                continue

            d_orig = sub.loc[valid, dist_col].mean()
            d_noisy = sub.loc[valid, noisy_col].mean() if noisy_col in sub.columns else float("nan")
            gain = sub.loc[valid, gain_col].mean() if gain_col in sub.columns else float("nan")
            states = sub.loc[valid, states_col].mean() if states_col in sub.columns else float("nan")
            t = sub.loc[valid, time_col].mean() if time_col in sub.columns else float("nan")

            algo_name = {
                "ye": "Paper (KV)",
                "orth": "Orthogonal L*",
                "baseline": "Baseline L*",
            }.get(algo, algo)

            print(f"  {algo_name:<15} {d_orig:>12.6f} {d_noisy:>14.6f} {gain:>10.3f} {states:>8.1f} {t:>8.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare algorithms on Ye et al. benchmarks"
    )
    parser.add_argument("--num-benchmarks", type=int, default=5)
    parser.add_argument(
        "--noise-type",
        choices=["noisy_output", "noisy_input"],
        default="noisy_output",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(seed=2)

    if args.noise_type == "noisy_output":
        df = run_noisy_output_benchmarks(num_benchmarks=args.num_benchmarks)
    elif args.noise_type == "noisy_input":
        df = run_noisy_input_benchmarks(num_benchmarks=args.num_benchmarks)

    save_path = os.path.join(RESULTS_DIR, f"comparison_{args.noise_type}.csv")
    df.to_csv(save_path, index=False)
    logging.info(f"Results saved to {save_path}")

    summarize_results(df)


if __name__ == "__main__":
    main()
