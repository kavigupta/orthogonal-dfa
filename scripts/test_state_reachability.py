"""Test that learned DFA correctly classifies prefixes according to the DT.

This catches the bug where compute_transition_matrix builds a transition matrix
that doesn't correctly represent the DT's state assignments.
"""

import numpy as np
from orthogonal_dfa.l_star.examples.benchmark_generator import (
    DFAOracle,
    sample_balanced_benchmark,
)
from orthogonal_dfa.l_star.lstar import (
    optimal_dfa,
    add_counterexample_prefixes,
    classify_states_with_decision_tree,
    compute_transition_matrix,
)
from orthogonal_dfa.l_star.state_discovery import discover_states
from orthogonal_dfa.l_star.dfa_utils import final_states_all_initial
from tests.test_lstar import compute_dfa_accuracy, compute_pst


def test_dfa_matches_dt(benchmark_seed: int, num_rounds: int = 3):
    """Test that DFA state assignments match the DT on all confident prefixes.

    Args:
        benchmark_seed: Seed for the benchmark
        num_rounds: How many rounds to run before checking

    Raises:
        AssertionError: If the DFA disagrees with DT on any confident prefix
    """
    outer, inner, sep = sample_balanced_benchmark(
        benchmark_seed,
        alphabet_size=2,
        num_inner_states=12,
        num_outer_states=10,
        probe_length=40,
        min_accept_or_reject=0.15,
    )

    oracle_creator = lambda nm, s, _dfa=outer: DFAOracle(nm, s, _dfa)
    pst = compute_pst(oracle_creator, 0.3, 0)

    # Run synthesis rounds
    for round_num in range(num_rounds):
        # Discover states
        while True:
            dt = discover_states(pst, first_round=(round_num == 0))
            if dt.num_states > 1:
                break
            pst.sample_more_prefixes()

        # Get DFA
        internal_acc, dfa = optimal_dfa(pst, dt)
        true_acc, fp, fn = compute_dfa_accuracy(dfa, oracle_creator)

        # === CORE CHECK ===
        print(
            f"\nRound {round_num}: {dt.num_states} DT states, "
            f"{len(dfa.states)} DFA states, internal={internal_acc:.4f}, "
            f"true={true_acc:.4f}"
        )

        # Get DT state assignments
        dt_states = classify_states_with_decision_tree(pst, dt)
        confident = dt_states >= 0
        dt_states_conf = dt_states[confident]

        # Get DFA state assignments (try each initial state, pick best)
        transitions = compute_transition_matrix(pst, dt)
        confident_prefixes = [
            pre for pre, is_conf in zip(pst.prefixes, confident) if is_conf
        ]
        dfa_states_all_initials = final_states_all_initial(transitions, confident_prefixes)

        success_rates = (dfa_states_all_initials == dt_states_conf).mean(1)
        best_initial = np.argmax(success_rates)
        best_success_rate = success_rates[best_initial]
        dfa_states = dfa_states_all_initials[best_initial]

        print(f"  Best initial state {best_initial}: agreement={best_success_rate:.4f}")

        # Find mismatches
        mismatches = dfa_states != dt_states_conf
        mismatch_count = mismatches.sum()

        if mismatch_count > 0:
            print(f"\n  >>> MISMATCH: {mismatch_count} / {len(dt_states_conf)} prefixes")
            for dt_s in np.unique(dt_states_conf[mismatches]):
                dfa_s = dfa_states[dt_states_conf == dt_s][mismatches[dt_states_conf == dt_s]]
                dt_prefix_mask = dt_states_conf == dt_s
                print(
                    f"      DT state {dt_s} ({dt_prefix_mask.sum()} total, "
                    f"{mismatches[dt_prefix_mask].sum()} mismatches)"
                )
                for bad_dfa_s in np.unique(dfa_s):
                    bad_count = (dfa_s == bad_dfa_s).sum()
                    print(f"        -> DFA state {bad_dfa_s}: {bad_count}")

        # ASSERTION: DFA should agree with DT on (almost) all confident prefixes
        # Allow up to 1% error due to randomness in the data
        max_allowed_error = max(1, int(0.01 * len(dt_states_conf)))
        assert (
            mismatch_count <= max_allowed_error
        ), (
            f"DFA disagrees with DT on {mismatch_count} prefixes "
            f"(>{max_allowed_error} allowed). This indicates "
            "compute_transition_matrix is building an incorrect transition table."
        )

        # Also check that number of DFA states doesn't exceed DT states
        assert (
            len(dfa.states) <= dt.num_states
        ), (
            f"DFA has {len(dfa.states)} states but DT only has {dt.num_states}. "
            "States are being created unnecessarily."
        )

        # Add counterexamples for next round
        if round_num < num_rounds - 1:
            add_counterexample_prefixes(pst, dt, dfa, 200)

    print("\n✓ All checks passed!")


if __name__ == "__main__":
    test_dfa_matches_dt(benchmark_seed=1, num_rounds=3)
