"""Benchmark family 2: CAPAL's own shipped `.taf` dataset.

The `.taf` file is the source of truth; `to_automata_dfa` ports it to the
automata-lib DFA that `DFAOracle` (and hence E-L*) reads.
"""

from __future__ import annotations

from typing import List

from orthogonal_dfa.capal_official import (
    import_capal,
    resolve_capal_dir,
    to_automata_dfa,
)

from .benchmark import FAMILY_CAPAL, Benchmark


def capal_benchmarks() -> List[Benchmark]:
    """CAPAL's shipped `.taf` targets, each wrapped for both learners."""
    from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle

    M = import_capal()
    d = resolve_capal_dir() / "dataset"
    if not d.is_dir():
        raise RuntimeError(f"no dataset/ directory in the CAPAL checkout at {d}")

    out: List[Benchmark] = []
    for taf in sorted(d.glob("*.taf")):
        name = taf.stem
        target = M.load_dfa_from_taf(str(taf))
        aut = to_automata_dfa(target)
        out.append(
            Benchmark(
                name=name,
                family=FAMILY_CAPAL,
                target=target,
                oracle_creator=lambda nm, s, _d=aut: DFAOracle(nm, s, _d),
                alphabet=list(target.alphabet),
                symbols=len(target.alphabet),
                target_states=target.num_states,
            )
        )
    return out
