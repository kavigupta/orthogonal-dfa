"""
summary.tex, the `random_minimal.analyze` summary of the results `run` caches.

    python -m orthogonal_dfa.experiments.luhn.analyze
"""

from ..capal_comparison.core import REPO_ROOT
from ..random_minimal.analyze import fragments, load_results
from .run import DATA_DIR

OUT_DIR = REPO_ROOT / "figures" / "luhn"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = OUT_DIR / "summary.tex"
    tex_path.write_text(fragments(load_results(DATA_DIR))["summary"] + "\n")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
