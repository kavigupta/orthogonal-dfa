"""
Thin wrapper around aalpy's L* loop that supports a max_states cap.

The only difference from aalpy's run_Lstar is:
- Uses CappedObservationTable (subclass) instead of ObservationTable
- Drops features we don't use (mealy/moore, timing, multiple cex strategies)
"""

from aalpy.automata import Dfa, DfaState
from aalpy.base import SUL
from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import CacheSUL
from aalpy.learning_algs.deterministic.CounterExampleProcessing import (
    rs_cex_processing,
)
from aalpy.learning_algs.deterministic.ObservationTable import ObservationTable
from aalpy.utils.HelperFunctions import extend_set


class CappedObservationTable(ObservationTable):
    """ObservationTable subclass that caps the number of states and handles
    unclosed rows gracefully."""

    def __init__(self, alphabet, sul, max_states=None):
        super().__init__(alphabet, sul, "dfa", prefixes_in_cell=True)
        self.max_states = max_states

    def get_rows_to_close(self, closing_strategy="shortest_first"):
        if self.max_states is not None and len(self.S) >= self.max_states:
            return None
        return super().get_rows_to_close(closing_strategy)

    def gen_hypothesis(self, no_cex_processing_used=False):
        state_distinguish = {}
        states_dict = {}
        initial_state = None

        for i, prefix in enumerate(self.S):
            state = DfaState(f"s{i}")
            state.is_accepting = self.T[prefix][0]
            state.prefix = prefix
            states_dict[prefix] = state
            state_distinguish[tuple(self.T[prefix])] = state
            if not prefix:
                initial_state = state

        for prefix in self.S:
            for a in self.A:
                row = tuple(self.T[prefix + a])
                target = state_distinguish.get(row)
                if target is None:
                    # Unclosed row: find nearest S-row by suffix signature
                    target = _closest_row(row, state_distinguish)
                states_dict[prefix].transitions[a[0]] = target

        return Dfa(initial_state, list(states_dict.values()))


def _closest_row(row, state_distinguish):
    """Find the S-row most similar to `row` by Hamming distance."""
    _, state = min(
        state_distinguish.items(),
        key=lambda item: sum(a != b for a, b in zip(row, item[0])),
    )
    return state


MAX_LEARNING_ROUNDS = 20


def run_lstar_with_max_states(
    alphabet: list,
    sul: SUL,
    eq_oracle: Oracle,
    *,
    max_states=None,
):
    """Run L* with an optional state cap. Mirrors aalpy's run_Lstar."""
    sul = CacheSUL(sul)
    eq_oracle.sul = sul

    obs_table = CappedObservationTable(alphabet, sul, max_states=max_states)
    obs_table.update_obs_table()

    hypothesis = None
    learning_rounds = 0

    while learning_rounds < MAX_LEARNING_ROUNDS:
        # Close the table (respects max_states via subclass)
        rows_to_close = obs_table.get_rows_to_close("shortest_first")
        while rows_to_close is not None:
            rows_to_query = []
            for row in rows_to_close:
                obs_table.S.append(row)
                rows_to_query.extend([row + (a,) for a in alphabet])
            obs_table.update_obs_table(s_set=rows_to_query)
            rows_to_close = obs_table.get_rows_to_close("shortest_first")

        # Build hypothesis (handles unclosed rows via subclass)
        hypothesis = obs_table.gen_hypothesis()
        learning_rounds += 1

        cex = eq_oracle.find_cex(hypothesis)
        if cex is None:
            break

        cex = tuple(cex)
        cex_suffixes = rs_cex_processing(
            sul, cex, hypothesis, False, closedness="suffix"
        )
        added_suffixes = extend_set(obs_table.E, cex_suffixes)
        obs_table.update_obs_table(e_set=added_suffixes)

    return hypothesis
