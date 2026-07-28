"""Benchmark family 2: CAPAL's own shipped `.taf` dataset.

The `.taf` file is the source of truth; `taf_to_automata_dfa` ports it to the
automata-lib DFA that `DFAOracle` (and hence E-L*) reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from orthogonal_dfa.capal_official import import_capal, resolve_capal_dir

from .benchmark import FAMILY_CAPAL, Benchmark, taf_to_automata_dfa


def capal_dataset_dir() -> Path:
    return resolve_capal_dir() / "dataset"


def capal_benchmarks(names: Optional[Sequence[str]] = None) -> List[Benchmark]:
    """CAPAL's shipped `.taf` targets, each wrapped for both learners."""
    from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle

    M = import_capal()
    d = capal_dataset_dir()
    if not d.is_dir():
        raise RuntimeError(f"no dataset/ directory in the CAPAL checkout at {d}")

    out: List[Benchmark] = []
    for taf in sorted(d.glob("*.taf")):
        name = taf.stem
        if names is not None and name not in names:
            continue
        target = M.load_dfa_from_taf(str(taf))
        aut = taf_to_automata_dfa(target)
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
    if names is not None:
        missing = set(names) - {b.name for b in out}
        if missing:
            raise RuntimeError(f"unknown target(s): {sorted(missing)}")
    return out
