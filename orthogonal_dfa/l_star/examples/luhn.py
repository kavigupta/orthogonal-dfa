"""The Luhn checksum, which credit card numbers satisfy, as a 100-state DFA."""

from automata.fa.dfa import DFA


def luhn_double(digit: int) -> int:
    """The digit sum of twice ``digit``."""
    return 2 * digit - 9 if digit >= 5 else 2 * digit


def luhn_valid(digits) -> bool:
    """Doubling every second digit from the right,

        sum_{i odd} d_i + sum_{i even} luhn_double(d_i) = 0 (mod 10)

    with d_1 the rightmost digit."""
    return (
        sum(d if i % 2 == 0 else luhn_double(d) for i, d in enumerate(reversed(digits)))
        % 10
        == 0
    )


def luhn_dfa() -> DFA:
    """`luhn_valid` over the digits 0-9.

    State ``10 a + b``: ``a`` is the checksum if the string ends here and ``b`` is
    it with the doubled and undoubled digits swapped, since which digits are
    doubled is fixed only by where the string ends.
    """

    def step(state, c):
        a, b = divmod(state, 10)
        return 10 * ((b + c) % 10) + (a + luhn_double(c)) % 10

    return DFA(
        states=set(range(100)),
        input_symbols=set(range(10)),
        transitions={q: {c: step(q, c) for c in range(10)} for q in range(100)},
        initial_state=0,
        final_states=set(range(10)),
        allow_partial=False,
    )
