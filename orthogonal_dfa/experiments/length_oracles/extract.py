"""Every run's run.json and restarts.json, collected into
``data/neural/length_oracles/runs.json``.

    python -m orthogonal_dfa.experiments.length_oracles.extract RUN_DIR..."""

import json
import sys

from ..capal_comparison.core import REPO_ROOT

OUT = REPO_ROOT / "data" / "neural" / "length_oracles" / "runs.json"


def main():
    runs, restarts = [], []
    for run in sys.argv[1:]:
        with open(f"{run}/run.json") as f:
            row = json.load(f)
        with open(f"{run}/restarts.json") as f:
            restarts += [
                dict(oracle=row["oracle"], seed=row["seed"], **r) for r in json.load(f)
            ]
        runs.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(runs=runs, restarts=restarts), f, separators=(",", ":"))


if __name__ == "__main__":
    main()
