"""The two benchmark families, each exposed to *both* learners.

A benchmark has to be presentable two ways: as an upstream `capal.DFA` (what
CAPAL learns) and as one of this repo's `Oracle`s (what E-L* learns). The two
views must denote the same language under the same symbol ordering, or the
head-to-head is meaningless -- so both are derived from a single source of
truth per family.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from orthogonal_dfa.capal_official import (
    build_modulo_dfa,
    build_regex_dfa,
    import_capal,
    resolve_capal_dir,
)
from orthogonal_dfa.l_star import preconditions

FAMILY_OURS = "ours"
FAMILY_CAPAL = "capal_dataset"

#: E-L*'s default sampling length, and the candidates we tune over.
DEFAULT_SAMPLE_LENGTH = 40

#: Shorter lengths are excluded on purpose. A "balanced" acceptance rate at
#: length 3 is worthless: the whole word space is 8 strings, and E-L* draws
#: hundreds of distinct prefixes and suffixes, so synthesis spins forever
#: instead of converging. Difficult01 (b*a*) is the case in point -- balanced
#: only at lengths 2-3, degenerate by length 10, measurable at neither.
MIN_VIABLE_SAMPLE_LENGTH = 8
CANDIDATE_LENGTHS = [8, 10, 12, 16, 20, 30, 40]

#: A target is unusable at a given length when nearly every sampled word gets
#: the same label -- there is then no acceptance-rate signal to separate
#: prefixes on, whatever the noise level.
DEGENERATE_LO = 0.02
DEGENERATE_HI = 0.98

#: E-L*'s designed operating regime, taken from the filters this repo's own
#: benchmark generator applies (`sample_balanced_benchmark`) with the threshold
#: values its callers in tests/test_lstar.py actually pass. A target outside
#: this regime is one the generator would have discarded, so measuring E-L* on
#: it says more about the benchmark than about the learner.
MIN_ACCEPT_OR_REJECT = 0.15  # tests/test_lstar.py passes this
MIN_CLASS_PRESERVING_FRAC = 0.05  # sample_balanced_benchmark default


@dataclass
class Benchmark:
    """One target, in both learners' terms.

    `alphabet` is the index -> symbol mapping shared by both views: E-L* works
    in symbol indices, CAPAL in characters, and this is what ties them.
    """

    name: str
    family: str
    target: Any  # upstream capal.DFA
    oracle_creator: Callable[[Any, int], Any]
    alphabet: Sequence[str]
    symbols: int
    target_states: int

    def truth(self) -> Callable[[List[int]], bool]:
        """Noiseless ground truth, taken from the upstream DFA so that both
        learners are scored against one definition of the language."""
        dfa, alpha = self.target, self.alphabet
        return lambda w: bool(dfa.run("".join(alpha[i] for i in w)))

    def accept_rate(self, length: int, *, count: int = 2000) -> float:
        """Fraction of uniform words of exactly `length` that the target accepts."""
        return preconditions.acceptance_rate(
            taf_to_automata_dfa(self.target), length=length, num_samples=count
        )

    def tune_sample_length(self) -> tuple:
        """Pick E-L*'s sampling length for this target, and say why.

        All of E-L*'s signal comes from words it samples, so a length at which
        the language is near-empty or near-saturating gives it nothing to work
        with -- 13 of CAPAL's 28 targets are degenerate at the default 40.
        Prefer the default when it is usable; otherwise take the candidate
        whose acceptance rate is closest to balanced.

        Some languages cannot be rescued by any length -- e.g. Difficult08 is
        {w : |w| != 1}, which is constant at every fixed length. Those are
        reported so the sweep can record them instead of burning hours
        sampling a distribution with no signal in it.

        Returns (length, accept_rate_at_that_length, rates_by_length).
        """
        rates = {n: self.accept_rate(n) for n in CANDIDATE_LENGTHS}
        default = rates.get(DEFAULT_SAMPLE_LENGTH, 0.0)
        if DEGENERATE_LO <= default <= DEGENERATE_HI:
            return DEFAULT_SAMPLE_LENGTH, default, rates
        best = min(CANDIDATE_LENGTHS, key=lambda n: abs(rates[n] - 0.5))
        return best, rates[best], rates

    def degenerate_at_all_lengths(self, rates: Dict[int, float]) -> bool:
        """True when no candidate length yields a usable acceptance rate."""
        return not any(DEGENERATE_LO <= r <= DEGENERATE_HI for r in rates.values())

    def class_preserving_frac(self, length: int, *, count: int = 2000) -> float:
        """Fraction of random length-`length` suffixes that map every state to a
        state of the same accept/reject class (preconditions.class_preserving_fraction).

        This repo's benchmark generator rejects candidates below
        MIN_CLASS_PRESERVING_FRAC because a low value "confuses L* synthesis".
        """
        return preconditions.class_preserving_fraction(
            taf_to_automata_dfa(self.target), length=length, num_samples=count
        )

    def regime_report(self) -> Dict[str, Any]:
        """Is this target inside E-L*'s designed regime, and if not, why not?

        Applies the three conditions of preconditions.satisfies_preconditions:
        acceptance balance and class-preservation (sampled at the tuned length)
        plus the exact infinite-reachability check. The measured values and the
        failure reasons go into the experiment JSON so exclusions are auditable.
        """
        aut = taf_to_automata_dfa(self.target)
        length, rate, rates = self.tune_sample_length()
        cp = self.class_preserving_frac(length)
        infinite = preconditions.infinitely_reachable_states(aut)
        transient = sorted(
            str(q) for q in aut.states if q != aut.initial_state and q not in infinite
        )
        reasons = []
        if not MIN_ACCEPT_OR_REJECT <= rate <= 1 - MIN_ACCEPT_OR_REJECT:
            reasons.append(
                f"acceptance rate {rate:.3f} outside "
                f"[{MIN_ACCEPT_OR_REJECT}, {1 - MIN_ACCEPT_OR_REJECT}]"
            )
        if cp < MIN_CLASS_PRESERVING_FRAC:
            reasons.append(
                f"class-preserving fraction {cp:.3f} below "
                f"{MIN_CLASS_PRESERVING_FRAC}"
            )
        if transient:
            reasons.append(
                f"{len(transient)} non-start state(s) reached by finitely many "
                f"strings (transient): {', '.join(transient)}"
            )
        return {
            "sample_length": length,
            "tuned_from_default": length != DEFAULT_SAMPLE_LENGTH,
            "accept_rate_at_sample_length": round(rate, 4),
            "class_preserving_frac": round(cp, 4),
            "transient_states": transient,
            "in_regime": not reasons,
            "excluded_because": reasons,
            "accept_rate_by_length": {str(k): round(v, 4) for k, v in rates.items()},
        }


# -- family 1: this repo's oracles --------------------------------------------


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


# -- family 2: CAPAL's own .taf dataset ---------------------------------------


def taf_to_automata_dfa(target: Any) -> Any:
    """Upstream `capal.DFA` -> automata-lib DFA over integer symbols.

    `DFAOracle` (and hence E-L*) works in symbol *indices*; upstream works in
    characters. Index i always means `target.alphabet[i]`, which is the same
    mapping `Benchmark.alphabet` hands to the CAPAL side.
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
        allow_partial=False,
    )


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
