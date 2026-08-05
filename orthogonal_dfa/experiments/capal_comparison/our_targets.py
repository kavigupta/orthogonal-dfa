"""Benchmark family 1: this repo's own oracles, ported for CAPAL.

The oracle is the source of truth; `build_modulo_dfa` / `build_regex_dfa` port
it to an upstream `capal.DFA` denoting the same language.
"""

from __future__ import annotations

from typing import List

from orthogonal_dfa.capal_official import build_modulo_dfa, build_regex_dfa

from .benchmark import FAMILY_OURS, Benchmark


def our_benchmarks() -> List[Benchmark]:
    """The oracles from `tests/test_lstar.py` that the findings doc reports on."""
    from orthogonal_dfa.l_star.examples.bernoulli_parity import (
        BernoulliParityOracle,
        BernoulliRegex,
    )

    def regex_case(name: str, regex: str, symbols: int = 2) -> Benchmark:
        target = build_regex_dfa(regex, symbols)
        return Benchmark(
            name=name,
            family=FAMILY_OURS,
            target=target,
            oracle_creator=(
                lambda nm, s, _r=regex, _k=symbols: BernoulliRegex(
                    nm, s, regex=_r, alphabet_size=_k
                )
            ),
            alphabet=[str(i) for i in range(symbols)],
            symbols=symbols,
            target_states=target.num_states,
        )

    modulo_target = build_modulo_dfa(9, (3, 6))
    return [
        Benchmark(
            name="parity_mod9_allowed_3_6",
            family=FAMILY_OURS,
            target=modulo_target,
            oracle_creator=lambda nm, s: BernoulliParityOracle(
                nm, s, modulo=9, allowed_moduluses=(3, 6)
            ),
            alphabet=["0", "1"],
            symbols=2,
            target_states=modulo_target.num_states,
        ),
        regex_case("regex_subseq_1010101", r".*1010101.*"),
        regex_case("regex_two_1111", r".*1111.*1111.*"),
        regex_case("regex_alt_1111_or_0000_11", r".*(1111|0000)11.*"),
        regex_case("regex_alt_111_or_000_3sym", r".*(111|000).*", symbols=3),
    ]
