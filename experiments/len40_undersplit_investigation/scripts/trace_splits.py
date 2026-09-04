"""Structural trace of round 1's wildcard splits: access strings of the states each
distinguisher separates, and whether the split is within one accept/reject class."""
import pickle
from collections import deque

NAME = {0: "TAG", 1: "TAA", 2: "TGA", 3: "X", 4: "Y"}
DUMP = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/round_dumps"


def named(seq):
    return "".join("[" + NAME.get(int(s), str(s)) + "]" for s in seq) or "eps"


def access_strings(dfa):
    """Shortest super-string reaching each state (BFS from the initial state)."""
    T = {s: {int(k): v for k, v in dfa.transitions[s].items()} for s in dfa.states}
    acc = {dfa.initial_state: []}
    q = deque([dfa.initial_state])
    while q:
        s = q.popleft()
        for a in sorted(T[s]):
            nx = T[s][a]
            if nx not in acc:
                acc[nx] = acc[s] + [a]
                q.append(nx)
    return acc


for r in (0, 1):
    d = pickle.load(open(f"{DUMP}/round_0{r}.pkl", "rb"))
    dfa, dt = d["dfa"], d["dt"]
    acc = access_strings(dfa)
    rej = set(dfa.states) - set(dfa.final_states)
    print(f"\n===== ROUND {r}: {len(dfa.states)} states, reject={sorted(rej)} =====")
    print("  access strings (shortest super-string reaching each state):")
    for s in sorted(dfa.states):
        cls = "REJ" if s in rej else "acc"
        a = named(acc[s]) if s in acc else "(unreachable)"
        print(f"    state {s} [{cls}]: {a}")
    # for each internal tree node, which two leaves does its distinguisher separate,
    # and are they the same accept/reject class?
    print("  distinguisher splits (midfix -> False-leaf / True-leaf, classes):")

    def walk(node, depth=0):
        if isinstance(node, int):
            return
        midfix, lookup = node
        f, t = lookup[False], lookup[True]

        def leafclass(n):
            if isinstance(n, int):
                return f"state {n} [{'REJ' if n in rej else 'acc'}]"
            return "(subtree)"
        # only report leaf/leaf or leaf-bearing splits with a wildcard-only midfix
        mf = midfix if isinstance(midfix, (list, tuple)) else [midfix]
        flat = mf[0] if mf and isinstance(mf[0], (list, tuple)) else mf
        only_wild = all(int(x) >= 3 for x in flat) if flat else True
        tag = "  <-- WILDCARD-ONLY" if only_wild and flat else ""
        print(f"      {named(flat):<22} F:{leafclass(f):18} T:{leafclass(t):18}{tag}")
        walk(f, depth + 1)
        walk(t, depth + 1)

    walk(dt.root)
