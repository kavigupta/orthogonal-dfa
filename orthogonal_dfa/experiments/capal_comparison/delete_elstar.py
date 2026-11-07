"""Deletes E-L*'s cells from the comparison to CAPAL, so the next run reruns them.

    python -m orthogonal_dfa.experiments.capal_comparison.delete_elstar"""

import json

from .core import LEARNER_ELSTAR, REPO_ROOT


def main():
    for name in ("capal_benchmarks", "our_benchmarks"):
        path = REPO_ROOT / "data" / "capal" / f"{name}.json"
        payload = json.loads(path.read_text())
        payload["cells"] = [
            c for c in payload["cells"] if c["learner"] != LEARNER_ELSTAR
        ]
        path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
