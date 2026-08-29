"""The lifted oracle's language, as a DFA over the super alphabet.

A super-string's compilations are the base strings that parse back to it, so a
fill may not start a kmer.  Whether it does depends on the symbols after it, which
a left-to-right walk has not read yet, so an element carries the last
``max kmer length - 1`` base symbols and which of them a wildcard produced; a fill
dies when a kmer completes over one of those.  The super state is the set of base
states the surviving fills are in, and it accepts when they all do.
"""

from typing import Optional, Tuple

from automata.fa.dfa import DFA

from .vocabulary import KmerVocabulary

#: (base state, recent base symbols, which of them a wildcard produced)
Element = Tuple[int, Tuple[int, ...], Tuple[bool, ...]]


def _emitter(vocabulary: KmerVocabulary, base_dfa: DFA):
    width = max((len(k) for k in vocabulary.kmers), default=1) - 1

    def emit(element: Element, symbol: int, from_wildcard: bool) -> Optional[Element]:
        state, window, flags = element
        state = base_dfa.transitions[state][symbol]
        window, flags = window + (symbol,), flags + (from_wildcard,)
        for kmer in vocabulary.kmers:
            if window[-len(kmer) :] == kmer and flags[-len(kmer)]:
                return None
        return (state, window[-width:], flags[-width:])

    return emit


def super_target_dfa(vocabulary: KmerVocabulary, base_dfa: DFA) -> DFA:
    """``base_dfa``'s language read over ``vocabulary``'s alphabet.

    Raises when some super-string's fills disagree about acceptance -- then the
    lifted label is not a function of the super-string and no such DFA exists.
    """
    emit = _emitter(vocabulary, base_dfa)

    def step(elements, super_symbol):
        out = set()
        for element in elements:
            if vocabulary.is_unknown(super_symbol):
                for symbol in range(vocabulary.base_alphabet_size):
                    moved = emit(element, symbol, True)
                    if moved is not None:
                        out.add(moved)
                continue
            moved = element
            for symbol in vocabulary.kmers[super_symbol]:
                moved = emit(moved, symbol, False)
                if moved is None:
                    break
            if moved is not None:
                out.add(moved)
        return frozenset(out)

    start = frozenset({(base_dfa.initial_state, (), ())})
    index, order, transitions = {start: 0}, [start], {}
    while len(transitions) < len(order):
        current = order[len(transitions)]
        row = {}
        for symbol in range(vocabulary.alphabet_size):
            moved = step(current, symbol)
            if moved not in index:
                index[moved] = len(order)
                order.append(moved)
            row[symbol] = index[moved]
        transitions[index[current]] = row

    final = set()
    for elements in order:
        accepting = {q in base_dfa.final_states for q, _, _ in elements}
        assert len(accepting) <= 1, (
            "the base oracle reads how the wildcards were filled, so the lifted "
            "language is not a function of the super-string"
        )
        if accepting == {True}:
            final.add(index[elements])

    return DFA(
        states=set(range(len(order))),
        input_symbols=set(range(vocabulary.alphabet_size)),
        transitions=transitions,
        initial_state=0,
        final_states=final,
        allow_partial=False,
    ).minify()
