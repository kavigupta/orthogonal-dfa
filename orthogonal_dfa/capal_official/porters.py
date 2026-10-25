"""Convert target languages between this repo's DFA format and upstream's.

`build_*` go this repo -> upstream `capal.DFA`, so CAPAL's PerfectEQ can run
its product-BFS counterexample search against them. `to_automata_dfa` goes back
the other way, for the oracles E-L* reads.
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

    alphabet = sorted(aut.input_symbols)
    state_list = sorted(aut.states, key=lambda s: (str(type(s).__name__), str(s)))
    sidx = {s: i for i, s in enumerate(state_list)}

    # automata-lib rejects by dying, so any language with a dead end (e.g. `1*`,
    # `0*1*`) comes back partial, while upstream's DFA requires a transition for
    # every (state, symbol). Route the missing ones to an explicit sink.
    sink = len(state_list)
    delta = {}
    for s in state_list:
        for a in alphabet:
            dest = aut.transitions[s].get(a)
            delta[(sidx[s], a)] = sink if dest is None else sidx[dest]
    num_states = len(state_list)
    if sink in delta.values():
        num_states += 1
        for a in alphabet:
            delta[(sink, a)] = sink

    return M.DFA(
        alphabet=alphabet,
        num_states=num_states,
        start=sidx[aut.initial_state],
        accept={sidx[s] for s in aut.final_states},
        delta=delta,
    )


def from_automata_dfa(dfa: Any) -> Any:
    """Complete automata-lib DFA over symbols ``0..k-1`` -> upstream `capal.DFA`
    over the characters ``'0'..str(k-1)``; the inverse of `to_automata_dfa`."""
    M = import_capal()
    symbols = sorted(dfa.input_symbols)
    assert symbols == list(range(len(symbols))), symbols
    states = sorted(dfa.states, key=str)
    sidx = {q: i for i, q in enumerate(states)}
    return M.DFA(
        alphabet=[str(c) for c in symbols],
        num_states=len(states),
        start=sidx[dfa.initial_state],
        accept={sidx[q] for q in dfa.final_states},
        delta={
            (sidx[q], str(c)): sidx[dfa.transitions[q][c]]
            for q in states
            for c in symbols
        },
    )


def to_automata_dfa(target: Any) -> Any:
    """Upstream `capal.DFA` -> automata-lib DFA over integer symbols.

    `DFAOracle` (and hence E-L*) works in symbol *indices*; upstream works in
    characters. Index i always means `target.alphabet[i]`, the same mapping the
    CAPAL side is handed.
    """
    from automata.fa.dfa import DFA as AutDFA

    alphabet = list(target.alphabet)
    idx = {c: i for i, c in enumerate(alphabet)}
    transitions = {
        q: {idx[c]: target.step(q, c) for c in alphabet}
        for q in range(target.num_states)
    }
    return AutDFA(
        states=set(range(target.num_states)),
        input_symbols=set(idx.values()),
        transitions=transitions,
        initial_state=target.start,
        final_states=set(target.accept),
    )
