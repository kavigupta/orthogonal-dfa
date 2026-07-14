"""TTT-style discrimination tree over E_core.

Each internal node is labeled with a discriminator e in E_core; its left
child holds words w with y(w + e) = False, its right child holds w with
y(w + e) = True. Leaves correspond to states.

The DT is used for:
- classify(word): walk the tree to find the leaf (state) for any word.
- LCA(C1, C2): the lowest common ancestor's discriminator -- used to
  refine E_core when consistency fails (Algorithm 1, line 10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .observation import ObservationTable, Word


@dataclass
class _Node:
    discr: Optional[Word]  # None for leaves
    left: Optional["_Node"] = None  # y = False child
    right: Optional["_Node"] = None  # y = True child
    members: List[Word] = field(default_factory=list)  # only set on leaves
    leaf_id: int = -1  # only set on leaves
    parent: Optional["_Node"] = None

    @property
    def is_leaf(self) -> bool:
        return self.discr is None


class DiscriminationTree:
    """A DT over the prefixes in S, built by partitioning them using E_core.

    The leaves are numbered 0..n-1 in left-to-right DFS order; each leaf
    corresponds to a hypothesis state."""

    def __init__(self, table: ObservationTable) -> None:
        self.table = table
        self.root = self._build(list(table.S), list(table.E_core), parent=None)
        self.leaves: List[_Node] = []
        self._collect_leaves(self.root)
        # map each word in S to its leaf id
        self.word_to_leaf: Dict[Word, int] = {}
        for leaf in self.leaves:
            for w in leaf.members:
                self.word_to_leaf[w] = leaf.leaf_id

    # -- construction -------------------------------------------------------

    def _build(
        self, words: List[Word], avail: List[Word], parent: Optional[_Node]
    ) -> _Node:
        if len(words) <= 1:
            leaf = _Node(discr=None, members=list(words), parent=parent)
            return leaf
        # Find first e in avail (preserves the order in which discriminators
        # were added to E_core, matching the paper's near-minimal DT bound).
        for idx, e in enumerate(avail):
            labels = [self.table.y(w + e) for w in words]
            if False in labels and True in labels:
                left_words = [w for w, lab in zip(words, labels) if not lab]
                right_words = [w for w, lab in zip(words, labels) if lab]
                rest = avail[idx + 1 :]
                node = _Node(discr=e, parent=parent)
                node.left = self._build(left_words, rest, parent=node)
                node.right = self._build(right_words, rest, parent=node)
                return node
        # No remaining e splits these words: collapse into a single leaf.
        return _Node(discr=None, members=list(words), parent=parent)

    def _collect_leaves(self, node: _Node) -> None:
        if node.is_leaf:
            node.leaf_id = len(self.leaves)
            self.leaves.append(node)
            return
        assert node.left is not None and node.right is not None
        self._collect_leaves(node.left)
        self._collect_leaves(node.right)

    # -- queries ------------------------------------------------------------

    def num_leaves(self) -> int:
        return len(self.leaves)

    def classify(self, word: Word) -> int:
        """Return the leaf id reached by walking the DT for `word`."""
        node = self.root
        while not node.is_leaf:
            assert node.discr is not None
            if self.table.y(word + node.discr):
                assert node.right is not None
                node = node.right
            else:
                assert node.left is not None
                node = node.left
        return node.leaf_id

    def rep(self, leaf_id: int) -> Word:
        """Canonical representative for the leaf: the shortest member (lex-
        breaking ties)."""
        members = self.leaves[leaf_id].members
        return min(members, key=lambda w: (len(w), w))

    def lca_discriminator(self, leaf_a: int, leaf_b: int) -> Optional[Word]:
        """Lowest common ancestor's discriminator for two leaves.

        This is the suffix e such that y(rep(A) + e) != y(rep(B) + e), so the
        column (a + e) introduced by line 10 of Algorithm 1 separates the
        members of A and B."""
        if leaf_a == leaf_b:
            return None
        path_a = self._path_from_root(self.leaves[leaf_a])
        path_b = self._path_from_root(self.leaves[leaf_b])
        last_common: Optional[_Node] = None
        for x, y in zip(path_a, path_b):
            if x is y:
                last_common = x
            else:
                break
        if last_common is None or last_common.is_leaf:
            return None
        return last_common.discr

    def _path_from_root(self, node: _Node) -> List[_Node]:
        path: List[_Node] = []
        cur: Optional[_Node] = node
        while cur is not None:
            path.append(cur)
            cur = cur.parent
        path.reverse()
        return path
