"""Run the official CAPAL learner against this repo's targets.

`make_learner` returns the learner unfitted so callers can instrument it (count
queries, time the fit) before `fit_with_fallback`. Scoring stays with the
caller: a fair comparison scores every learner on one shared word list.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

UPSTREAM_URL = "https://github.com/lkwargs/CAPAL"

#: The single source of truth for the commit every number in
#: `data/capal_findings.md` was measured against. Bumping it means re-measuring.
PINNED_COMMIT = "57d877f6a083d58852660fac388ff49c052dc2d2"

#: What upstream's fit() says when it runs out of iterations (capal.py:1294).
CAP_MESSAGE = "Maximum iterations reached without convergence"

#: Default checkout location, resolved relative to the repo root rather than
#: the cwd, so it does not matter where a caller is invoked from.
DEFAULT_CAPAL_DIR = Path(__file__).resolve().parents[2].parent / "capal"

_official: Any = None


def resolve_capal_dir() -> Path:
    """The upstream checkout: a `capal` sibling of the repo root."""
    return DEFAULT_CAPAL_DIR


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
    """Raise unless `path` is a clean checkout at PINNED_COMMIT -- the numbers
    in data/capal_findings.md are only comparable against this exact tree."""
    if not path.exists():
        raise RuntimeError(
            f"No CAPAL checkout at {path}. Clone {UPSTREAM_URL} there and "
            f"`git checkout {PINNED_COMMIT}`."
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


def import_capal() -> Any:
    """Verify the pin and load upstream's `capal`, cached.

    Lazy so importing this package never requires the checkout, only using it
    does. Loaded by exact file path rather than onto sys.path, so upstream's
    loose sibling modules cannot shadow anything.
    """
    global _official  # pylint: disable=global-statement
    if _official is not None:
        return _official
    path = resolve_capal_dir()
    verify_pinned(path)
    spec = importlib.util.spec_from_file_location("capal", path / "capal.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["capal"] = module  # so capal.DFA etc. resolve by module name
    spec.loader.exec_module(module)
    _official = module
    return _official


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


def make_learner(
    target: Any,
    eta: float,
    *,
    max_iters: int = 200,
    seed: int = 0,
    verbose: bool = False,
    k_pos: int = 10,
    k_neg: int = 10,
    max_same_samples: int = 60,
    tau_cap: float = 0.2,
    suffix_pool_init: int = 32,
    suffix_pool_len_max: int = 8,
    alpha: float = 1e-3,
    enum_depth: int = 3,
    extra_len_max: int = 8,
) -> Any:
    """An unfitted CAPALLearner over `target`.

    `enum_depth`/`extra_len_max` are the matched-query-budget knob; LearnerConfig
    does not forward them, so they go straight on the live SameStateConfig.
    """
    official = import_capal()
    cfg = official.LearnerConfig(
        K_pos=k_pos,
        K_neg=k_neg,
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
    learner = official.CAPALLearner(target=target, cfg=cfg)
    learner.ss.cfg.enum_depth = enum_depth
    learner.ss.cfg.extra_len_max = extra_len_max
    return learner


def fit_with_fallback(learner: Any) -> Tuple[Optional[Any], bool]:
    """Fit `learner`, returning (dfa, converged).

    Under noise the iteration cap is a normal non-convergent outcome, not an
    error: the last hypothesis comes back with converged=False, or dfa None if
    there is none.
    """
    try:
        return learner.fit(), True
    except RuntimeError as exc:
        # RecursionError is a RuntimeError subclass; without this guard the
        # interpreter's own failures would look like non-convergent runs.
        # Matching the message is safe only because the pin locks it.
        if CAP_MESSAGE not in str(exc):
            raise
        last = getattr(learner, "_last_hyp", None)
        return getattr(last, "dfa", None), False
