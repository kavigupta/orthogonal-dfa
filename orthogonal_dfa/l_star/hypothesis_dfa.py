"""HypothesisDFA: a discrimination tree and the edges learned between its leaves."""

from typing import Callable, Dict, Optional, Set, Tuple

from automata.fa.dfa import DFA

from .midfix_tree import MidfixTree

Path = Tuple[bool, ...]
Edge = Tuple[Path, int]
Cut = Callable[[bytes], Optional[bool]]


class HypothesisDFA:
    """A tree it splits in place, read through cut."""

    def __init__(self, cut: Cut, tree: MidfixTree):
        self.cut, self.tree = cut, tree
        self.leaves = set(tree.leaf_paths())
        self.edges: Dict[Edge, Tuple[Path, bytes]] = {}
        #: Prefixes the DFA spells out letter by letter instead of sifting.
        self.expanded: Set[bytes] = set()
        #: Bumped at every change.
        self.version = 0

    def sift(self, z: bytes) -> Optional[Path]:
        return self.tree.sift_with_path(self.cut, z)[1]

    def set_edge(self, edge: Edge, value: Tuple[Path, bytes]):
        self.edges[edge] = value
        self.version += 1

    def expand(self, starts):
        self.expanded |= set(starts)
        self.version += 1

    def split_leaf(self, leaf: Path, midfix: bytes):
        """HypothesisDFA.split of RLStar/Code/HypothesisDFA.lean, which keeps every learned
        edge witnessed (SplitAndLearnCorrect). This retargets every edge; the Lean split keeps
        only edges at leaves, and along a run drops none (RunsKeepEdgesAtLeaves).
        """
        self.tree.split(self.tree.leaf_at(leaf), midfix)
        self.leaves = set(self.tree.leaf_paths())
        self._retarget_edges(leaf)
        self.version += 1

    def _retarget_edges(self, leaf: Path):
        """retarget: drop the split leaf's out-edges, and re-sift the
        witnesses of the edges into it."""
        edges = {}
        for (source, letter), (target, witness) in self.edges.items():
            if source[: len(leaf)] == leaf:
                continue
            if target == leaf:
                target = self.sift(witness + bytes([letter]))
                if target is None:
                    continue
            edges[(source, letter)] = (target, witness)
        self.edges = edges

    def to_dfa(
        self,
        alphabet_size: int,
        initial_state: int,
        classify: Optional[Callable[[bytes], Optional[int]]] = None,
    ) -> DFA:
        """An unlearned edge loops on its leaf. Each expanded prefix gets a state of
        its own, whose children not expanded go to the leaf ``classify`` gives them,
        or to ``initial_state`` where it abstains."""
        ids = {path: self.tree.leaf_at(path) for path in self.leaves}
        transitions = {
            ids[path]: {
                c: ids[self.edges.get((path, c), (path,))[0]]
                for c in range(alphabet_size)
            }
            for path in self.leaves
        }
        if self.expanded:
            first = max(transitions) + 1
            trie = {u: first + i for i, u in enumerate(sorted(self.expanded))}

            def child(v):
                if v in trie:
                    return trie[v]
                leaf = classify(v)
                return initial_state if leaf is None else leaf

            for u, state in trie.items():
                transitions[state] = {
                    c: child(u + bytes([c])) for c in range(alphabet_size)
                }
            if b"" in trie:
                initial_state = trie[b""]
        return DFA(
            states=set(transitions),
            input_symbols=set(range(alphabet_size)),
            transitions=transitions,
            initial_state=initial_state,
            final_states=self.tree.accepting_leaves(),
            allow_partial=False,
        )
