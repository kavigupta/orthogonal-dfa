"""Bridge between this repo's Oracle/test-harness and the official CAPAL repo.

Loads the upstream `capal` module from the pinned checkout (see
`resolve_capal_dir`), then exposes:

- make_learner(target, eta, ...): a CAPALLearner wired to the PersistentNoisyMQ
  + PerfectEQ defaults, returned unfitted so callers can instrument it.
- fit_with_fallback(learner): .fit(), plus whether it converged.

The two are split because callers need to reach the learner in between -- to
count queries by wrapping `learner.mq.query`, to time the fit, or to suppress
upstream's stdout. Scoring is deliberately not here: a comparison must score
every learner on one shared word list, which only the caller can build.

Building the target DFAs upstream needs lives in `porters`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

UPSTREAM_URL = "https://github.com/lkwargs/CAPAL"

#: Upstream CAPAL commit the checked-in results were measured against.
PINNED_COMMIT = "57d877f6a083d58852660fac388ff49c052dc2d2"

CAPAL_DIR_ENV = "ORTHO_CAPAL_DIR"

#: Default checkout: a sibling of the repo root.
DEFAULT_CAPAL_DIR = Path(__file__).resolve().parents[2].parent / "capal"

_official: Any = None


def resolve_capal_dir(capal_dir: Optional[str] = None) -> Path:
    """Upstream checkout: explicit `capal_dir`, else $ORTHO_CAPAL_DIR, else
    `../capal`."""
    override = capal_dir or os.environ.get(CAPAL_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
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


def import_capal(capal_dir: Optional[str] = None) -> Any:
    """Verify the pin, then import upstream's single-file `capal` module.

    Deliberately lazy and cached: importing this package must not fail just
    because the checkout is missing, but *using* it against an unpinned tree
    must.
    """
    global _official  # pylint: disable=global-statement
    if _official is not None:
        return _official
    path = resolve_capal_dir(capal_dir)
    verify_pinned(path)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import capal  # type: ignore[import-not-found]  # pylint: disable=import-error,import-outside-toplevel

    # A stray `capal` elsewhere on sys.path would import silently and give wrong
    # results; confirm we loaded the pinned checkout.
    loaded = getattr(capal, "__file__", None)
    if loaded is None or Path(loaded).resolve().parent != path:
        raise RuntimeError(
            f"`import capal` resolved to {loaded or '<namespace package>'}, "
            f"not the pinned checkout at {path}. Something on sys.path is "
            f"shadowing upstream CAPAL."
        )
    _official = capal
    return _official


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
    """A CAPALLearner over `target`, unfitted (PersistentNoisyMQ + PerfectEQ are
    built from `target` by upstream).

    `enum_depth` / `extra_len_max` bound how many and how long the SAMESTATE
    suffixes are -- the matched-query-budget knob. LearnerConfig does not
    forward them, so they are set on the live SameStateConfig, which SAMESTATE
    reads lazily during fit().
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

    fit() raises when max_iters elapses without PerfectEQ accepting, which under
    noise is the common case rather than an error; the last hypothesis is still
    worth scoring, so it comes back with converged=False. Only if there is no
    hypothesis at all is the dfa None. Anything but the iteration cap
    propagates.
    """
    try:
        return learner.fit(), True
    except RuntimeError:
        last = getattr(learner, "_last_hyp", None)
        return getattr(last, "dfa", None), False
