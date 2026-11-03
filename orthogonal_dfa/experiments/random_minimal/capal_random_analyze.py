"""
How many targets CAPAL succeeds on per size and eta, from the results
`capal_random` caches, as LaTeX macros: e.g. 12 states at eta = 0.10 is
``\\capalSuccessesTwelveEtaTen``.

    python -m orthogonal_dfa.experiments.random_minimal.capal_random_analyze
"""

from __future__ import annotations

import json

from .analyze import OUT_DIR, succeeded
from .capal_random import capal_result_path
from .run import COUNT, DATA_DIR, ETAS, SEEDS, SIZES

_ONES = (
    "Zero One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve "
    "Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen"
).split()
_TENS = "_ _ Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety".split()


def _number_name(n: int) -> str:
    """For n < 100, since LaTeX macro names cannot contain digits."""
    if n < 20:
        return _ONES[n]
    return _TENS[n // 10] + (_ONES[n % 10] if n % 10 else "")


def main() -> None:
    lines = []
    for size in SIZES:
        for eta in ETAS:
            successes = sum(
                succeeded(
                    json.loads(
                        capal_result_path(DATA_DIR, size, index, eta, seed).read_text()
                    )
                )
                for index in range(COUNT)
                for seed in SEEDS
            )
            name = (
                f"capalSuccesses{_number_name(size)}Eta{_number_name(round(eta * 100))}"
            )
            lines.append(f"\\newcommand{{\\{name}}}{{{successes}}}\n")
    path = OUT_DIR / "capal_successes.tex"
    path.write_text("".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
