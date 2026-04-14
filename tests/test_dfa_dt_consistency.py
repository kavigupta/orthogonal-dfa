"""Test that learned DFA consistently represents the decision tree.

This catches bugs where the transition matrix estimation produces a DFA that
disagrees with the DT on prefixes it was trained on.
"""

import numpy as np
import unittest

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
from orthogonal_dfa.l_star.structures import TriPredicate
from orthogonal_dfa.l_star.state_discovery import discover_states
from orthogonal_dfa.l_star.dfa_utils import final_states_all_initial
from tests.test_lstar import compute_dfa_accuracy, compute_pst


class TestDFADTConsistency(unittest.TestCase):
    """Verify that DFA state assignments match the decision tree."""

    def test_dfa_matches_dt_on_confident_prefixes(self):
        """After synthesis, DFA should agree with DT on all confident prefixes."""
        outer, inner, sep = sample_balanced_benchmark(
            seed=1,
            alphabet_size=2,
            num_inner_states=12,
            num_outer_states=10,
            probe_length=40,
            min_accept_or_reject=0.15,
        )

        oracle_creator = lambda nm, s, _dfa=outer: DFAOracle(nm, s, _dfa)
        pst = compute_pst(oracle_creator, 0.3, 0)

        # Run 3 rounds of synthesis
        for round_num in range(3):
            # Discover states
            while True:
                dt = discover_states(pst, first_round=(round_num == 0))
                if dt.num_states > 1:
                    break
                pst.sample_more_prefixes()

            # Get DFA
            internal_acc, dfa = optimal_dfa(pst, dt)
            true_acc, _, _ = compute_dfa_accuracy(dfa, oracle_creator)

            # Get DT state assignments on confident prefixes
            dt_states = classify_states_with_decision_tree(pst, dt)
            confident = dt_states >= 0
            dt_states_conf = dt_states[confident]

            # Compute DFA state assignments
            transitions = compute_transition_matrix(pst, dt)
            confident_prefixes = [
                pre for pre, is_conf in zip(pst.prefixes, confident) if is_conf
            ]
            dfa_states_all = final_states_all_initial(transitions, confident_prefixes)

            # Find best initial state
            success_rates = (dfa_states_all == dt_states_conf).mean(1)
            best_initial = np.argmax(success_rates)
            dfa_states = dfa_states_all[best_initial]

            # Check agreement
            mismatches = dfa_states != dt_states_conf
            mismatch_count = mismatches.sum()
            total_prefixes = len(dt_states_conf)

            # Allow up to 3% error (due to randomness/estimation and DT confidence bounds)
            max_allowed_error = max(1, int(0.03 * total_prefixes))

            if mismatch_count > max_allowed_error:
                mismatch_msg = (
                    f"Round {round_num}: DFA disagrees with DT on {mismatch_count} "
                    f"/ {total_prefixes} confident prefixes "
                    f"(internal_acc={internal_acc:.4f}, true_acc={true_acc:.4f})\n\n"
                )
                # Add detail on which states are involved
                for dt_s in sorted(np.unique(dt_states_conf[mismatches])):
                    mask = (dt_states_conf == dt_s) & mismatches
                    bad_dfa_states = np.unique(dfa_states[mask])
                    mismatch_msg += (
                        f"  DT state {dt_s} ({(dt_states_conf==dt_s).sum()} total, "
                        f"{mask.sum()} mismatches) -> DFA states {sorted(bad_dfa_states)}\n"
                    )

                # Add detailed analysis of transition matrix estimation
                mismatch_msg += "\n=== TRANSITION MATRIX ANALYSIS ===\n"

                # Recompute the transition matrix vote counts to diagnose
                num_states = dt.num_states
                vote_counts = np.zeros(
                    (num_states, pst.alphabet_size, num_states), dtype=int
                )
                for c in range(pst.alphabet_size):
                    # Create modified DT with [c] + suffix
                    dt_modified = dt.map_over_predicates(
                        lambda p, c=c: TriPredicate(
                            [[c] + x for x in p.vs],
                            p.accept_threshold,
                            p.reject_threshold,
                        )
                    )
                    states_after_c = classify_states_with_decision_tree(
                        pst, dt_modified
                    )
                    valid = states_after_c >= 0

                    # Count votes: (source_state, symbol, target_state)
                    for i, (src, tgt) in enumerate(zip(dt_states, states_after_c)):
                        if src >= 0 and valid[i]:
                            vote_counts[src, c, tgt] += 1

                mismatch_msg += (
                    "Vote counts show how many times each (source, symbol, target) "
                    "transition was observed:\n"
                    "(argmax picks the target with most votes; ties go to lower index)\n\n"
                )

                for source_dt_s in sorted(np.unique(dt_states_conf[mismatches])):
                    mask_source = (dt_states_conf == source_dt_s) & mismatches

                    mismatch_msg += f"Source DT state {source_dt_s} ({mask_source.sum()} mismatches):\n"

                    for sym in range(2):
                        votes = vote_counts[source_dt_s, sym]
                        chosen_target = transitions[source_dt_s, sym]
                        votes_for_chosen = votes[chosen_target]

                        mismatch_msg += f"  Symbol {sym}: votes = {votes}, chosen = {chosen_target}"
                        mismatch_msg += f" ({votes_for_chosen} votes)\n"
                        if votes_for_chosen == 0:
                            mismatch_msg += (
                                "    ^^ WARNING: CHOSEN TARGET HAS ZERO VOTES! "
                                "(likely tied with argmax=0)\n"
                            )

                    mismatch_msg += "\n"

                self.fail(mismatch_msg)

            # Also verify state count doesn't explode
            self.assertLessEqual(
                len(dfa.states),
                dt.num_states,
                f"Round {round_num}: DFA has {len(dfa.states)} states "
                f"but DT has {dt.num_states}",
            )

            # Add counterexamples for next round
            if round_num < 2:
                add_counterexample_prefixes(pst, dt, dfa, 200)


if __name__ == "__main__":
    unittest.main()
