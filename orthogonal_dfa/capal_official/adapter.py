"""Bridge between this repo's Oracle/test-harness and the official CAPAL repo.

Loads the upstream `capal` module from the pinned checkout (see
`resolve_capal_dir`), then exposes:

- build_modulo_dfa(modulo, allowed): the explicit modulo-counting DFA used by
  BernoulliParityOracle, in the upstream `capal.DFA` format.
- build_regex_dfa(regex, alphabet_size): regex -> upstream DFA via automata-lib
  (NFA.from_regex -> DFA.from_nfa, then state-relabel).
- run_official_capal(target, eta, ...): instantiate CAPALLearner with the
  PersistentNoisyMQ + PerfectEQ defaults and call .fit().
- evaluate_official_dfa(dfa, oracle_creator, alphabet_chars, symbols): sample
  uniformly random words, query the noiseless oracle for ground truth, count
  matches.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Set

import numpy as np

UPSTREAM_URL = "https://github.com/lkwargs/CAPAL"

#: Commit the findings doc was measured against. `scripts/capal_upstream.py`
#: keeps its own copy (the scripts folder is standalone w.r.t. this package);
#: bump both together, and re-measure.
PINNED_COMMIT = "57d877f6a083d58852660fac388ff49c052dc2d2"

#: Env var to point at a checkout elsewhere; otherwise a sibling of the repo.
CAPAL_DIR_ENV = "ORTHO_CAPAL_DIR"

_official: Any = None


def resolve_capal_dir() -> Path:
    """Upstream checkout location: $ORTHO_CAPAL_DIR, else `../capal`."""
    override = os.environ.get(CAPAL_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2].parent / "capal"


def _git(path: Path, *args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "git not found on PATH; cannot verify the pinned CAPAL checkout."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"`git {' '.join(args)}` failed in {path}: "
            f"{exc.stderr.strip() or exc}. Expected a clone of {UPSTREAM_URL}."
        ) from exc
    return out.stdout.strip()


def verify_pinned(path: Path) -> None:
    """Raise unless `path` is a clean checkout at PINNED_COMMIT.

    Reproducibility guard: `data/capal_findings.md` is only meaningful against
    this exact commit with no local modifications.
    """
    if not path.exists():
        raise RuntimeError(
            f"No CAPAL checkout at {path}. Clone {UPSTREAM_URL} there and "
            f"`git checkout {PINNED_COMMIT}`, or set ${CAPAL_DIR_ENV}."
        )
    if not (path / "capal.py").exists():
        raise RuntimeError(
            f"{path} contains no capal.py; expected a clone of {UPSTREAM_URL}."
        )

    head = _git(path, "rev-parse", "HEAD")
    if head != PINNED_COMMIT:
        raise RuntimeError(
            f"CAPAL checkout at {path} is at the wrong commit "
            f"(expected {PINNED_COMMIT}, found {head}). data/capal_findings.md "
            f"was measured against the expected commit; others are not "
            f"comparable. Run: git -C {path} checkout {PINNED_COMMIT}"
        )

    dirty = _git(path, "status", "--porcelain")
    if dirty:
        raise RuntimeError(
            f"CAPAL checkout at {path} has local modifications, so results "
            f"would not be reproducible:\n{dirty}"
        )


def _require_official() -> Any:
    """Verify the pin and import upstream on first use.

    Deliberately lazy: importing this package must not fail just because the
    checkout is missing, but *using* it against an unpinned tree must.
    """
    global _official
    if _official is not None:
        return _official
    path = resolve_capal_dir()
    verify_pinned(path)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import capal  # type: ignore[import-not-found]

    _official = capal
    return _official


def build_modulo_dfa(modulo: int, allowed: Iterable[int]) -> Any:
    """The 'sum mod N in allowed?' DFA over {'0','1'}, in upstream format."""
    M = _require_official()
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


def build_all_frames_closed_dfa() -> Any:
    """The DFA matching `AllFramesClosedOracle`: 4-symbol input ('0'..'3'
    corresponding to A/C/G/T), accepted iff for every phase k in {0,1,2}, the
    substring starting at position k contains at least one of TAG/TGA/TAA as a
    non-overlapping length-3 codon. We construct the product of three
    single-phase machines via BFS over reachable joint states; the resulting
    DFA is exact, then automata-lib minifies it."""
    from collections import deque

    from automata.fa.dfa import DFA as AutDFA

    M = _require_official()
    sigma = ["0", "1", "2", "3"]  # A,C,G,T
    stops = {(3, 0, 2), (3, 2, 0), (3, 0, 0)}  # TAG, TGA, TAA

    # A per-phase state is either ("waiting", skips_left) before the phase has
    # started, or (buffer_tuple, found_bool) once it has.
    def frame_initial(k: int):
        if k == 0:
            return ((), False)
        return ("waiting", k)

    def frame_step(state, c: int):
        if state[0] == "waiting":
            skips_left = state[1]
            if skips_left > 1:
                return ("waiting", skips_left - 1)
            return ((c,), False)
        buf, found = state
        new_buf = buf + (c,)
        if len(new_buf) == 3:
            return ((), bool(found or new_buf in stops))
        return (new_buf, found)

    initial = (frame_initial(0), frame_initial(1), frame_initial(2))
    idx: dict = {initial: 0}
    delta_int: dict = {}
    queue: deque = deque([initial])
    while queue:
        s = queue.popleft()
        sid = idx[s]
        for a in sigma:
            c = int(a)
            new_s = tuple(frame_step(s[k], c) for k in range(3))
            if new_s not in idx:
                idx[new_s] = len(idx)
                queue.append(new_s)
            delta_int[(sid, a)] = idx[new_s]

    accept: set = set()
    for s, sid in idx.items():
        if all(s[k][0] != "waiting" and s[k][1] for k in range(3)):
            accept.add(sid)

    # Hand this through automata-lib to minify before returning.
    transitions_aut = {sid: {} for sid in range(len(idx))}
    for (sid, a), dst in delta_int.items():
        transitions_aut[sid][a] = dst
    aut = AutDFA(
        states={i for i in range(len(idx))},
        input_symbols=set(sigma),
        transitions=transitions_aut,
        initial_state=0,
        final_states=accept,
        allow_partial=False,
    ).minify()
    state_list = sorted(aut.states, key=lambda s: (str(type(s).__name__), str(s)))
    sidx = {s: i for i, s in enumerate(state_list)}
    delta = {}
    for s in state_list:
        for a, dst in aut.transitions[s].items():
            delta[(sidx[s], a)] = sidx[dst]
    return M.DFA(
        alphabet=sorted(aut.input_symbols),
        num_states=len(state_list),
        start=sidx[aut.initial_state],
        accept={sidx[s] for s in aut.final_states},
        delta=delta,
    )


def build_regex_dfa(regex: str, alphabet_size: int = 2) -> Any:
    """Compile `regex` to a minimal DFA in upstream format. Symbols are the
    characters '0', '1', ... matching BernoulliRegex's int->str convention."""
    from automata.fa.dfa import DFA as AutDFA
    from automata.fa.nfa import NFA

    M = _require_official()
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


def run_official_capal(
    target: Any,
    eta: float,
    *,
    max_iters: int = 200,
    seed: int = 0,
    verbose: bool = False,
    K_pos: int = 10,
    K_neg: int = 10,
    max_same_samples: int = 60,
    tau_cap: float = 0.2,
    suffix_pool_init: int = 32,
    suffix_pool_len_max: int = 8,
    alpha: float = 1e-3,
) -> Any:
    """Instantiate CAPALLearner with upstream defaults (PersistentNoisyMQ +
    PerfectEQ are built automatically from `target`) and return the learned
    `capal.DFA`."""
    M = _require_official()
    cfg = M.LearnerConfig(
        K_pos=K_pos,
        K_neg=K_neg,
        max_iters=max_iters,
        seed=seed,
        eta=eta,
        alpha=alpha,
        max_same_samples=max_same_samples,
        tau_cap=tau_cap,
        suffix_pool_init=suffix_pool_init,
        suffix_pool_len_max=suffix_pool_len_max,
        verbose=verbose,
    )
    learner = M.CAPALLearner(target=target, cfg=cfg)
    try:
        return learner.fit()
    except RuntimeError as exc:
        # `fit()` raises when max_iters elapses without PerfectEQ accepting --
        # under noise this is common because SAMESTATE may make a handful of
        # false-DIFFERENT decisions, leaving the hypothesis larger than the
        # target. Return the latest hypothesis so we can still measure its
        # accuracy (the user wants accuracy at noise level, not just the
        # converged/not bit).
        last = getattr(learner, "_last_hyp", None)
        if last is not None and getattr(last, "dfa", None) is not None:
            last.dfa.converged = False  # type: ignore[attr-defined]
            return last.dfa
        raise


def evaluate_official_dfa(
    dfa: Any,
    oracle_creator,
    alphabet_chars: Sequence[str],
    symbols: int,
    *,
    count: int = 5000,
    rng_seed: int = 0x1234,
) -> float:
    """Sample random words via the repo's UniformSampler, ask the noiseless
    oracle for the truth label, and ask the upstream DFA for its label.
    Returns accuracy."""
    from orthogonal_dfa.l_star.sampler import UniformSampler
    from orthogonal_dfa.l_star.structures import SymmetricBernoulli

    us = UniformSampler(40)
    truth = oracle_creator(SymmetricBernoulli(p_correct=1.0), 0)
    rng = np.random.default_rng(rng_seed)
    correct = 0
    for _ in range(count):
        s = us.sample(rng, symbols)
        truth_label = bool(truth.membership_query(s))
        word = "".join(alphabet_chars[c] for c in s)
        pred = bool(dfa.run(word))
        if pred == truth_label:
            correct += 1
    return correct / count
