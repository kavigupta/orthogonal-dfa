"""The prefixes a hypothesis spells out letter by letter, with their own states."""

from typing import Callable, Optional, Set

from automata.fa.dfa import DFA


class Trie:
    """A set of prefixes, closed under prefixes by its callers."""

    def __init__(self):
        self.prefixes: Set[bytes] = set()
        #: Bumped whenever prefixes are spelled out.
        self.version = 0

    @property
    def depth(self) -> int:
        """One past the longest prefix."""
        return 1 + max(len(u) for u in self.prefixes)

    def walk_start(self, x: bytes) -> int:
        """Past every prefix of x in the trie."""
        k = 0
        while k < len(x) and x[:k] in self.prefixes:
            k += 1
        return k

    def spell_out(self, prefixes):
        self.prefixes |= set(prefixes)
        self.version += 1

    def prepend(
        self,
        leaves: DFA,
        classify: Callable[[bytes], Optional[int]],
        accepts: Callable[[bytes], Optional[bool]],
    ) -> DFA:
        """leaves with a state per prefix, accepting where accepts says so. A child
        outside the trie goes to the leaf classify gives it, or to the initial state
        of leaves where it abstains."""
        if not self.prefixes:
            return leaves
        transitions = {s: dict(row) for s, row in leaves.transitions.items()}
        first = max(transitions) + 1
        trie = {u: first + i for i, u in enumerate(sorted(self.prefixes))}

        def child(v):
            if v in trie:
                return trie[v]
            leaf = classify(v)
            return leaves.initial_state if leaf is None else leaf

        for u, state in trie.items():
            transitions[state] = {
                c: child(u + bytes([c])) for c in sorted(leaves.input_symbols)
            }
        final_states = set(leaves.final_states) | {
            state for u, state in trie.items() if accepts(u)
        }
        return DFA(
            states=set(transitions),
            input_symbols=set(leaves.input_symbols),
            transitions=transitions,
            initial_state=trie.get(b"", leaves.initial_state),
            final_states=final_states,
            allow_partial=False,
        )
