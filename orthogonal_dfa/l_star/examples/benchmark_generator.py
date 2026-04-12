"""Benchmark generator for noisy L* algorithm.

Generates languages of the form ``Σ*LΣ*`` (contains a substring in *L*) where
*L* is a regular language satisfying the **separator property**: there exists a
character *c* such that no string in *L* ends with *c*.

(Any longer separator ``s`` witnesses the same property via its first
character, so length-1 separators are w.l.o.g.)

Separator constraint at the DFA level
--------------------------------------
For separator character *c* and inner DFA ``(Q, Σ, δ, q0, F)``:

    ``∀q ∈ Q:  δ(q, c) ∉ F``

This single structural constraint is equivalent (after minimisation) to the
language-level separator property.

Sampling the inner DFA
-----------------------
Transitions on *c* are drawn uniformly from *Q\\F*; all other transitions are
drawn uniformly from *Q*.  This is uniform over the unconstrained degrees of
freedom.
"""

from typing import List, Tuple

import numpy as np
from automata.fa.dfa import DFA
from automata.fa.nfa import NFA

from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.structures import NoiseModel, Oracle
from orthogonal_dfa.utils.dfa import al_dfa_symbols_to_int, al_dfa_symbols_to_str


def sample_inner_dfa(
    rng: np.random.Generator,
    *,
    num_states: int,
    alphabet_size: int,
    separator_char: int,
    num_accepting: int | None = None,
) -> DFA:
    """Sample a random minimal DFA for *L* satisfying the separator property.

    The constraint ``∀q: δ(q, separator_char) ∉ F`` is enforced by drawing
    transitions on *separator_char* uniformly from *Q\\F*.  All other
    transitions are uniform over *Q*.

    After construction the DFA is minimised; if the resulting language is
    empty the draw is discarded and a new one is taken.

    Parameters
    ----------
    num_states : state count of the pre-minimisation inner DFA.
    alphabet_size : |Σ| (must be ≥ 2).
    separator_char : the forbidden final character.
    num_accepting : number of accepting states (default: random in 1..n−1).
    """
    if num_states < 2:
        raise ValueError("Need num_states >= 2")

    while True:
        n_acc = num_accepting
        if n_acc is None:
            n_acc = int(rng.integers(1, num_states))  # 1 .. n-1

        # q0 = 0 is always non-accepting (so ε ∉ L)
        candidates = np.arange(1, num_states)
        chosen = rng.choice(candidates, size=min(n_acc, len(candidates)), replace=False)
        accepting = frozenset(int(c) for c in chosen)
        non_accepting = [s for s in range(num_states) if s not in accepting]

        transitions: dict = {}
        for q in range(num_states):
            transitions[q] = {}
            for c in range(alphabet_size):
                if c == separator_char:
                    transitions[q][c] = non_accepting[
                        int(rng.integers(len(non_accepting)))
                    ]
                else:
                    transitions[q][c] = int(rng.integers(num_states))

        dfa = DFA(
            states=set(range(num_states)),
            input_symbols=set(range(alphabet_size)),
            transitions=transitions,
            initial_state=0,
            final_states=accepting,
        ).minify()

        if dfa.final_states and dfa.initial_state not in dfa.final_states:
            return dfa


def build_star_l_star_dfa(inner_dfa: DFA) -> DFA:
    """Build the minimal DFA for ``Σ*LΣ*`` given a DFA for *L*.

    Computes ``Σ* · L · Σ*`` via NFA concatenation then determinises.
    """
    str_dfa = al_dfa_symbols_to_str(inner_dfa)
    str_syms = str_dfa.input_symbols
    sigma_star = NFA(
        states={"q"},
        input_symbols=str_syms,
        transitions={"q": {c: {"q"} for c in str_syms}},
        initial_state="q",
        final_states={"q"},
    )
    inner_nfa = NFA.from_dfa(str_dfa)
    nfa = sigma_star.concatenate(inner_nfa).concatenate(sigma_star)
    return al_dfa_symbols_to_int(DFA.from_nfa(nfa, minify=True))


def sample_star_l_star(
    rng: np.random.Generator,
    *,
    num_inner_states: int | None = None,
    alphabet_size: int = 2,
    num_accepting: int | None = None,
) -> Tuple[DFA, DFA, int]:
    """Sample a random ``Σ*LΣ*`` benchmark.

    Returns ``(outer_dfa, inner_dfa, separator_char)``.
    """
    separator_char = int(rng.integers(0, alphabet_size))

    if num_inner_states is None:
        num_inner_states = int(rng.integers(3, 7))  # 3–6

    inner = sample_inner_dfa(
        rng,
        num_states=num_inner_states,
        alphabet_size=alphabet_size,
        separator_char=separator_char,
        num_accepting=num_accepting,
    )
    outer = build_star_l_star_dfa(inner)
    return outer, inner, separator_char


def sample_balanced_benchmark(
    seed: int,
    *,
    alphabet_size: int,
    num_inner_states: int,
    num_accepting: int,
    max_outer_states: int,
    probe_length: int,
    min_accept_or_reject: float,
    num_probe_samples: int = 200,
    max_attempts: int = 10_000,
) -> Tuple[DFA, DFA, int]:
    """Sample a ``Σ*LΣ*`` benchmark whose outer DFA passes the given filters.

    Tries successive sub-seeds derived from *seed* until one produces a DFA
    whose state count and accept rate fall within the requested bounds.

    Each candidate gets a fresh RNG so that the filtering process does not
    contaminate the randomness of the chosen benchmark.

    Parameters
    ----------
    seed : top-level seed; the i-th candidate uses ``np.random.default_rng((seed, i))``.
    alphabet_size : |Σ| of the inner / outer DFAs.
    num_inner_states : pre-minimisation state count for the inner DFA.
    num_accepting : number of accepting states in the inner DFA.
    max_outer_states : inclusive upper bound on the size of the minimised
        ``Σ*LΣ*`` DFA.
    probe_length : length of random strings used to estimate the accept rate.
    min_accept_or_reject : minimum fraction of probe strings that must be in
        each class — i.e. the empirical accept rate must lie in
        ``[min_accept_or_reject, 1 - min_accept_or_reject]``.
    num_probe_samples : how many strings to sample when estimating the rate.
    max_attempts : maximum number of candidate benchmarks to try.

    Raises
    ------
    RuntimeError if no candidate passes the filters within ``max_attempts``.
    """
    # Hardcode the lower bound on outer DFA size — anything smaller is trivial
    # (1 state = constant function, 2 states = depends on a single character).
    min_outer_states = 3
    sampler = UniformSampler(probe_length)
    probe_rng = np.random.default_rng(seed)
    for sub in range(max_attempts):
        rng = np.random.default_rng((seed, sub))
        outer, inner, sep = sample_star_l_star(
            rng,
            num_inner_states=num_inner_states,
            alphabet_size=alphabet_size,
            num_accepting=num_accepting,
        )
        if not min_outer_states <= len(outer.states) <= max_outer_states:
            continue
        rate = (
            sum(
                outer.accepts_input(sampler.sample(probe_rng, alphabet_size))
                for _ in range(num_probe_samples)
            )
            / num_probe_samples
        )
        if min_accept_or_reject <= rate <= 1 - min_accept_or_reject:
            return outer, inner, sep
    raise RuntimeError(
        f"Could not find a balanced benchmark in {max_attempts} attempts"
    )


class DFAOracle(Oracle):
    """Oracle backed by a pre-built DFA (e.g. from ``build_star_l_star_dfa``)."""

    def __init__(self, noise_model: NoiseModel, seed: int, dfa: DFA):
        self._noise_model = noise_model
        self._seed = seed
        self._dfa = dfa
        self._alphabet_size = len(dfa.input_symbols)

    @property
    def alphabet_size(self) -> int:
        return self._alphabet_size

    def membership_query(self, string: List[int]) -> bool:
        correct = self._dfa.accepts_input(string)
        return self._noise_model.apply_noise(correct, string, self._seed)
