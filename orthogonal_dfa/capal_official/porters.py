"""Port this repo's example targets into upstream CAPAL's `capal.DFA` format.

Each builds the noiseless target DFA for one benchmark family, so upstream's
PerfectEQ can run its product-BFS counterexample search against it.
"""

from __future__ import annotations

from typing import Any, Iterable

from .adapter import import_capal


def build_modulo_dfa(modulo: int, allowed: Iterable[int]) -> Any:
    """The 'sum mod N in allowed?' DFA over {'0','1'}, in upstream format."""
    M = import_capal()
    delta = {}
    for q in range(modulo):
        delta[(q, "0")] = q
        delta[(q, "1")] = (q + 1) % modulo
    return M.DFA(
        alphabet=["0", "1"],
        num_states=modulo,
        start=0,
        accept={int(x) for x in allowed},
        delta=delta,
    )


def build_regex_dfa(regex: str, alphabet_size: int = 2) -> Any:
    """Compile `regex` to a minimal DFA in upstream format. Symbols are the
    characters '0', '1', ... matching BernoulliRegex's int->str convention."""
    from automata.fa.dfa import DFA as AutDFA
    from automata.fa.nfa import NFA

    M = import_capal()
    syms = {str(i) for i in range(alphabet_size)}
    nfa = NFA.from_regex(regex, input_symbols=syms)
    aut = AutDFA.from_nfa(nfa, minify=True)

    state_list = sorted(aut.states, key=lambda s: (str(type(s).__name__), str(s)))
    sidx = {s: i for i, s in enumerate(state_list)}
    delta = {}
    for s in state_list:
        for a, dest in aut.transitions[s].items():
            delta[(sidx[s], a)] = sidx[dest]
    return M.DFA(
        alphabet=sorted(aut.input_symbols),
        num_states=len(state_list),
        start=sidx[aut.initial_state],
        accept={sidx[s] for s in aut.final_states},
        delta=delta,
    )
