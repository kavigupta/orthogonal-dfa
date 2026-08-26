"""Print the test targets belonging to one CI shard.

Targets are packed longest-first by their recorded duration, so the shard count
can be tuned against the slowest single test rather than against the job count.
A target with no recorded duration is packed as if it were the median one.
"""

import argparse
import json
import os
import subprocess
import sys

DURATIONS = os.path.join(os.path.dirname(__file__), "test_durations.json")

#: Per-test sharding pays off here and nowhere else; every other file is packed
#: whole, since splitting one costs a collection pass to save a few seconds.
PER_TEST_FILE = "tests/test_lstar.py"


def _targets(mode):
    if mode == "files":
        out = []
        for name in sorted(os.listdir("tests")):
            path = os.path.join("tests", name)
            if name.startswith("test_") and name.endswith(".py"):
                if path != PER_TEST_FILE:
                    out.append(path)
        return out
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", PER_TEST_FILE, "--collect-only", "-q"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ln.strip() for ln in collected.splitlines() if "::" in ln]


def _pack(targets, durations, num_shards):
    """Longest-processing-time-first. Returns a list of ``num_shards`` lists."""
    known = sorted(durations.values())
    default = known[len(known) // 2] if known else 1
    shards = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for target in sorted(targets, key=lambda t: -durations.get(t, default)):
        i = loads.index(min(loads))
        shards[i].append(target)
        loads[i] += durations.get(target, default)
    return shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["files", "lstar"], required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    with open(DURATIONS) as f:
        durations = json.load(f)
    shards = _pack(_targets(args.mode), durations, args.num_shards)
    print(" ".join(shards[args.shard]))


if __name__ == "__main__":
    main()
