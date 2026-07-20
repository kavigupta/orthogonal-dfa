"""Locate and verify the pinned upstream CAPAL checkout.

Used by the scripts in `scripts/capal/`. The folder is standalone with respect
to this repo's `orthogonal_dfa` package: the scripts here import this sibling
module and nothing else from the repo, so they run against a bare CAPAL clone.
`orthogonal_dfa/capal_official/adapter.py` keeps its own copy of the constants
below for the package-side sweep -- if the pin ever moves, both must be
updated.

Every number in `data/capal_findings.md` was produced against
github.com/lkwargs/CAPAL @ 57d877f. Running against a different commit, or
against a checkout with local edits, silently produces numbers that no longer
match the doc -- so we refuse to run at all in either case.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

UPSTREAM_URL = "https://github.com/lkwargs/CAPAL"

#: Commit the findings doc was measured against. Bump this and the copy in
#: `orthogonal_dfa/capal_official/adapter.py` together, and re-measure.
PINNED_COMMIT = "57d877f6a083d58852660fac388ff49c052dc2d2"

#: scripts/capal/upstream.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default checkout location: a sibling of this repo. Resolved relative to the
#: repo root, not to the current working directory, so it does not matter where
#: the scripts are invoked from.
DEFAULT_CAPAL_DIR = REPO_ROOT.parent / "capal"


def resolve_capal_dir(capal_dir: Optional[str] = None) -> Path:
    """Path to the upstream checkout: `capal_dir` if given, else the default."""
    if capal_dir is not None:
        return Path(capal_dir).expanduser().resolve()
    return DEFAULT_CAPAL_DIR


def _git(path: Path, *args: str) -> str:
    """Run git in `path`, returning stdout. Exits on any git failure."""
    try:
        out = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        sys.exit("git not found on PATH; cannot verify the pinned CAPAL checkout.")
    except subprocess.CalledProcessError as exc:
        sys.exit(
            f"`git {' '.join(args)}` failed in {path}: "
            f"{exc.stderr.strip() or exc}\n"
            f"Expected a clone of {UPSTREAM_URL} there."
        )
    return out.stdout.strip()


def verify_pinned(path: Path) -> None:
    """Hard-fail unless `path` is a clean checkout at PINNED_COMMIT.

    Reproducibility guard: the findings doc's numbers are only meaningful
    against this exact commit with no local modifications.
    """
    if not path.exists():
        sys.exit(
            f"No CAPAL checkout at {path}.\n"
            f"  git clone {UPSTREAM_URL} {path}\n"
            f"  git -C {path} checkout {PINNED_COMMIT}\n"
            f"Or pass --capal-dir to point at an existing clone."
        )
    if not (path / "capal.py").exists():
        sys.exit(f"{path} contains no capal.py; expected a clone of {UPSTREAM_URL}.")

    head = _git(path, "rev-parse", "HEAD")
    if head != PINNED_COMMIT:
        sys.exit(
            f"CAPAL checkout at {path} is at the wrong commit.\n"
            f"  expected: {PINNED_COMMIT}\n"
            f"  found:    {head}\n"
            f"data/capal_findings.md was measured against the expected commit; "
            f"other commits are not comparable.\n"
            f"  git -C {path} checkout {PINNED_COMMIT}"
        )

    dirty = _git(path, "status", "--porcelain")
    if dirty:
        sys.exit(
            f"CAPAL checkout at {path} has local modifications:\n"
            f"{dirty}\n"
            f"Results would not be reproducible. Stash or discard them:\n"
            f"  git -C {path} stash --include-untracked"
        )


def import_capal(capal_dir: Optional[str] = None) -> Any:
    """Verify the pin, then import upstream's single-file `capal` module."""
    path = resolve_capal_dir(capal_dir)
    verify_pinned(path)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import capal  # type: ignore[import-not-found]

    # This folder is itself named `capal`, so if scripts/ ever lands on
    # sys.path it can shadow upstream as a namespace package -- which would
    # silently import the wrong thing rather than fail. Check what we got.
    loaded = getattr(capal, "__file__", None)
    if loaded is None or Path(loaded).resolve().parent != path:
        sys.exit(
            f"`import capal` resolved to {loaded or '<namespace package>'}, "
            f"not the pinned checkout at {path}. Something on sys.path is "
            f"shadowing upstream CAPAL."
        )
    return capal
