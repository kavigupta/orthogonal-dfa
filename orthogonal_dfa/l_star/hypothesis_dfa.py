"""``HypothesisDFA``: a discrimination tree and the edges learned between its leaves."""

from typing import Callable, Dict, Optional, Tuple

from automata.fa.dfa import DFA

from .midfix_tree import MidfixTree

Path = Tuple[bool, ...]
Edge = Tuple[Path, int]
Cut = Callable[[bytes], Optional[bool]]


class HypothesisDFA:
    """A tree it splits in place, read through ``cut``."""

    def __init__(self, cut: Cut, tree: MidfixTree):
        self.cut, self.tree = cut, tree
        self.leaves = set(tree.leaf_paths())
        self.edges: Dict[Edge, Tuple[Path, bytes]] = {}
        #: Bumped at every change.
        self.version = 0

    def sift(self, z: bytes) -> Optional[Path]:
        return self.tree.sift_with_path(self.cut, z)[1]

    def set_edge(self, edge: Edge, value: Tuple[Path, bytes]):
        self.edges[edge] = value
        self.version += 1

    def split_leaf(self, leaf: Path, midfix: bytes):
        self.tree.split(self.tree.leaf_at(leaf), midfix)
        self.leaves = set(self.tree.leaf_paths())
        self._retarget_edges(leaf)
        self.version += 1

    def _retarget_edges(self, leaf: Path):
        """``retargetBy``: drop the split leaf's out-edges, and re-sift the
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

    def to_dfa(self, alphabet_size: int, initial_state: int) -> DFA:
        """An unlearned edge loops on its leaf."""
        ids = {path: self.tree.leaf_at(path) for path in self.leaves}
        transitions = {
            ids[path]: {
                c: ids[self.edges.get((path, c), (path,))[0]]
                for c in range(alphabet_size)
            }
            for path in self.leaves
        }
        return DFA(
            states=set(ids.values()),
            input_symbols=set(range(alphabet_size)),
            transitions=transitions,
            initial_state=initial_state,
            final_states=self.tree.accepting_leaves(),
            allow_partial=False,
        )
