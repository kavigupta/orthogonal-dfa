"""CAPAL main learner.

Implements Algorithm 1 of Chen, Trivedi, Velasquez (ICLR 2026). The main loop:

    1. Build SAMESTATE equivalence classes on S via union-find.
    2. Closedness: extend S until every s*a is SAMESTATE-related to some u in S.
    3. Consistency: for any pair (rep, u) in the same union-find class whose
       a-successors land in different classes, add a separator column
       (a,) + e to E_core (with e drawn from E_core, then E_pool, then ()).
    4. Build hypothesis H using one rep per union-find class; acceptance is
       gold-overridden if any class member has a CE label, else majority of y.
    5. EQ(H). If OK return; else Rivest-Schapire decomposition of the CE with
       label-only-CE gating per App. A.3.3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.structures import Oracle

from .eq_oracle import RandomWordEqOracle
from .observation import ObservationTable, Word

log = logging.getLogger(__name__)


class _UnionFind:
    def __init__(self, elems: List[Word]) -> None:
        self.parent: Dict[Word, Word] = {w: w for w in elems}

    def find(self, w: Word) -> Word:
        cur = w
        while self.parent[cur] != cur:
            self.parent[cur] = self.parent[self.parent[cur]]
            cur = self.parent[cur]
        return cur

    def union(self, a: Word, b: Word) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Prefer lex-shorter root so reps stay short.
        if (len(rb), rb) < (len(ra), ra):
            ra, rb = rb, ra
        self.parent[rb] = ra

    def classes(self) -> Dict[Word, List[Word]]:
        out: Dict[Word, List[Word]] = {}
        for w in self.parent:
            out.setdefault(self.find(w), []).append(w)
        return out


@dataclass
class CAPAL:
    table: ObservationTable
    eq: RandomWordEqOracle
    max_eq_rounds: int = 60
    max_inner_rounds: int = 200
    max_states: int = 200
    label_only_seen: Set[Word] = field(default_factory=set)

    # -- equivalence classes via union-find ---------------------------------

    def _build_classes(self) -> _UnionFind:
        elems = sorted(self.table.S, key=lambda w: (len(w), w))
        uf = _UnionFind(elems)
        for i in range(len(elems)):
            for j in range(i):
                if uf.find(elems[i]) == uf.find(elems[j]):
                    continue
                if self.table.same_state(elems[i], elems[j]):
                    uf.union(elems[i], elems[j])
        return uf

    @staticmethod
    def _rep_of_root(uf: _UnionFind, root: Word) -> Word:
        members = uf.classes()[root]
        return min(members, key=lambda w: (len(w), w))

    def _classify(self, w: Word, uf: _UnionFind) -> Optional[Word]:
        """Return the union-find class root that w belongs to. Words in S use
        their stored root; for w not in S we SAMESTATE against each class rep
        and take the first match. Returns None when no class matches (this
        should not happen after closedness)."""
        if w in uf.parent:
            return uf.find(w)
        for root in uf.classes():
            rep = self._rep_of_root(uf, root)
            if self.table.same_state(w, rep):
                return root
        return None

    # -- closedness ---------------------------------------------------------

    def _try_close(self, uf: _UnionFind) -> bool:
        """Extend S if some s*a has no SAMESTATE match in S. Returns True if S
        was extended (caller should restart classes).

        Algorithm 1 line 6: '!exists u in S with sa ~ u'. We check sa against
        every u in S because union-find can chain SAMESTATE-True pairs into a
        class whose canonical rep is not itself SAMESTATE-True with every
        sibling -- using only class reps would let sa fall through the
        closedness check even when its own original S-entry is a valid match."""
        if not self.table.S:
            return False
        S_sorted = sorted(self.table.S, key=lambda w: (len(w), w))
        for s in S_sorted:
            for a in range(self.table.alphabet_size):
                sa = s + (a,)
                if sa in uf.parent:
                    continue  # already in S
                if any(self.table.same_state(sa, u) for u in S_sorted):
                    continue
                self.table.add_to_S(sa)
                return True
        return False

    # -- consistency --------------------------------------------------------

    def _try_consistency_repair(self, uf: _UnionFind) -> bool:
        """For each union-find class C, for each pair (rep, u) in C, and each
        symbol a, check whether rep*a and u*a land in the same class. If not,
        add a column (a,) + e to E_core where e is the shortest existing
        suffix witnessing y(rep + a + e) != y(u + a + e). Returns True if
        E_core grew."""
        classes_dict = uf.classes()
        for root, members in classes_dict.items():
            if len(members) < 2:
                continue
            rep = self._rep_of_root(uf, root)
            for u in members:
                if u == rep:
                    continue
                for a in range(self.table.alphabet_size):
                    cls_rep_a = self._classify(rep + (a,), uf)
                    cls_u_a = self._classify(u + (a,), uf)
                    if cls_rep_a is None or cls_u_a is None:
                        continue
                    if cls_rep_a == cls_u_a:
                        continue
                    new_col = self._find_separator_column(rep, u, a)
                    if new_col is not None and new_col not in self.table.E_core:
                        self.table.add_to_E_core(new_col)
                        return True
        return False

    def _find_separator_column(self, rep: Word, u: Word, a: int) -> Optional[Word]:
        """Look for the shortest column (a,) + e with y(rep + col) != y(u + col).
        Searches the empty suffix, then current E_core, then E_pool."""
        candidates: List[Word] = (
            [()] + list(self.table.E_core) + list(self.table.E_pool)
        )
        # Stable de-dup preserving insertion order.
        seen: Set[Word] = set()
        for e in candidates:
            if e in seen:
                continue
            seen.add(e)
            col = (a,) + e
            if self.table.y(rep + col) != self.table.y(u + col):
                return col
        return None

    # -- hypothesis construction --------------------------------------------

    def _build_hypothesis(self, uf: _UnionFind) -> Tuple[DFA, Dict[int, Word]]:
        classes_dict = uf.classes()
        roots = list(classes_dict.keys())
        root_to_state = {r: i for i, r in enumerate(roots)}
        n = len(roots)
        state_to_rep: Dict[int, Word] = {}
        transitions: Dict[int, Dict[int, int]] = {}
        for root, members in classes_dict.items():
            rep = self._rep_of_root(uf, root)
            state_to_rep[root_to_state[root]] = rep
            row: Dict[int, int] = {}
            for a in range(self.table.alphabet_size):
                target = self._classify(rep + (a,), uf)
                if target is None:
                    # Closedness should preclude this; fall back to self-loop.
                    target = root
                row[a] = root_to_state[target]
            transitions[root_to_state[root]] = row
        final_states: Set[int] = set()
        for root, members in classes_dict.items():
            state = root_to_state[root]
            gold_votes = [self.table.gold[m] for m in members if m in self.table.gold]
            if gold_votes:
                if 2 * sum(gold_votes) > len(gold_votes):
                    final_states.add(state)
            else:
                votes = [self.table.y(m) for m in members]
                if 2 * sum(votes) > len(votes):
                    final_states.add(state)
        init_root = self._classify((), uf)
        if init_root is None:
            init_root = uf.find(())
        init_state = root_to_state[init_root]
        dfa = DFA(
            states=set(range(n)),
            input_symbols=set(range(self.table.alphabet_size)),
            transitions=transitions,
            initial_state=init_state,
            final_states=final_states,
            allow_partial=False,
        )
        return dfa, state_to_rep

    # -- Rivest-Schapire CE processing --------------------------------------

    def _process_counterexample(
        self,
        ce: Word,
        gold_label: bool,
        dfa: DFA,
        state_to_rep: Dict[int, Word],
    ) -> bool:
        """Returns True if the algorithm made structural progress (S or E_core
        grew), False if the CE was processed as label-only on its first
        occurrence."""
        self.table.set_gold(ce, gold_label)

        # state_i = state reached in dfa on ce[:i]
        states: List[int] = [dfa.initial_state]
        for sym in ce:
            states.append(dfa.transitions[states[-1]][sym])

        def f(i: int) -> bool:
            if i == 0:
                return gold_label
            return self.table.y(state_to_rep[states[i]] + ce[i:])

        n = len(ce)
        split_i: Optional[int] = None
        for i in range(n):
            if f(i) != f(i + 1):
                split_i = i
                break

        if split_i is not None:
            e = ce[split_i + 1 :]
            if e not in self.table.E_core:
                self.table.add_to_E_core(e)
            new_word = state_to_rep[states[split_i]] + ce[split_i:]
            self.table.add_to_S(new_word)
            return True

        # Label-only counterexample.
        if ce in self.label_only_seen:
            for k in range(len(ce) + 1):
                suf = ce[k:]
                if suf not in self.table.E_core:
                    self.table.add_to_E_core(suf)
            return True
        self.label_only_seen.add(ce)
        return False

    # -- main loop ----------------------------------------------------------

    def learn(self) -> DFA:
        last_dfa: Optional[DFA] = None
        for round_ix in range(self.max_eq_rounds):
            # Inner loop: enforce closedness and consistency.
            inner = 0
            while True:
                inner += 1
                if inner > self.max_inner_rounds:
                    raise RuntimeError(
                        "CAPAL: inner loop did not converge "
                        f"(|S|={len(self.table.S)}, |E_core|={len(self.table.E_core)})"
                    )
                uf = self._build_classes()
                num_classes = len(uf.classes())
                if num_classes > self.max_states:
                    raise RuntimeError(
                        f"CAPAL: exceeded max_states={self.max_states}; "
                        f"SAMESTATE produced {num_classes} classes."
                    )
                if self._try_close(uf):
                    continue
                if self._try_consistency_repair(uf):
                    continue
                break

            dfa, state_to_rep = self._build_hypothesis(uf)
            last_dfa = dfa
            log.debug(
                "CAPAL round %d: |S|=%d, |E_core|=%d, states=%d, mq=%d",
                round_ix,
                len(self.table.S),
                len(self.table.E_core),
                len(dfa.states),
                self.table.mq_count,
            )
            ce = self.eq.find_counterexample(dfa)
            if ce is None:
                return dfa
            ce_word: Word = tuple(ce)
            gold_label = not bool(dfa.accepts_input(list(ce)))
            self._process_counterexample(ce_word, gold_label, dfa, state_to_rep)
        assert last_dfa is not None
        return last_dfa


def run_capal(
    oracle: Oracle,
    truth_oracle: Oracle,
    *,
    eta: float,
    num_eq_walks: int = 5000,
    max_walk_len: int = 30,
    alpha: float = 0.01,
    tau_max: float = 0.1,
    pool_max_len: int = 6,
    pool_long_len: int = 10,
    pool_num_long: int = 40,
    pool_cap: int = 256,
    max_eq_rounds: int = 60,
    max_inner_rounds: int = 200,
    max_states: int = 200,
    eq_seed: int = 0,
) -> DFA:
    table = ObservationTable(
        oracle=oracle,
        alphabet_size=oracle.alphabet_size,
        eta=eta,
        alpha=alpha,
        tau_max=tau_max,
        pool_max_len=pool_max_len,
        pool_long_len=pool_long_len,
        pool_num_long=pool_num_long,
        pool_cap=pool_cap,
    )
    eq = RandomWordEqOracle(
        truth=truth_oracle,
        alphabet_size=oracle.alphabet_size,
        num_walks=num_eq_walks,
        max_walk_len=max_walk_len,
        seed=eq_seed,
    )
    learner = CAPAL(
        table=table,
        eq=eq,
        max_eq_rounds=max_eq_rounds,
        max_inner_rounds=max_inner_rounds,
        max_states=max_states,
    )
    return learner.learn()
