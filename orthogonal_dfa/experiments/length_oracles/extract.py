"""Every given run's run.json and restarts.json, added to
``data/neural/length_oracles/runs.json`` in place of any saved rows of the same run.

    python -m orthogonal_dfa.experiments.length_oracles.extract RUN_DIR..."""

import json
import sys

from ..capal_comparison.core import REPO_ROOT

OUT = REPO_ROOT / "data" / "neural" / "length_oracles" / "runs.json"


def saved_runs():
    if not OUT.exists():
        return dict(runs=[], restarts=[])
    with open(OUT) as f:
        return json.load(f)


def extract(run_dirs):
    runs, restarts = [], []
    for run in run_dirs:
        with open(f"{run}/run.json") as f:
            row = json.load(f)
        with open(f"{run}/restarts.json") as f:
            restarts += [
                dict(oracle=row["oracle"], seed=row["seed"], **r) for r in json.load(f)
            ]
        runs.append(row)
    fresh = {(r["oracle"], r["seed"]) for r in runs}
    saved = saved_runs()
    runs, restarts = [
        [r for r in saved[key] if (r["oracle"], r["seed"]) not in fresh] + new
        for key, new in (("runs", runs), ("restarts", restarts))
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(f"{OUT.name}.tmp")
    with open(tmp, "w") as f:
        json.dump(dict(runs=runs, restarts=restarts), f, separators=(",", ":"))
    tmp.replace(OUT)


if __name__ == "__main__":
    extract(sys.argv[1:])
