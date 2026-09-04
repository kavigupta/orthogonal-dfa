"""Compare round 0 and round 1 of the stop-codon run: DFA + decision tree + behavior."""
import pickle, numpy as np
from orthogonal_dfa.superlanguage import KmerVocabulary, SuperSampler

NAME = {0: "TAG", 1: "TAA", 2: "TGA", 3: "X", 4: "Y"}
STOPS = {(3, 0, 2), (3, 0, 0), (3, 2, 0)}
DUMP = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/round_dumps"
V = KmerVocabulary(kmers=(3, 0, 2), base_alphabet_size=4) if False else \
    KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)


def named(seq):
    return "".join("[" + NAME.get(int(s), str(s)) + "]" for s in seq) or "eps"


def render_midfix(m):
    # a midfix is a tuple; its first element is the distinguishing suffix (a seq)
    if isinstance(m, (tuple, list)) and m and isinstance(m[0], (tuple, list)):
        return "  |  ".join(named(x) for x in m)
    return named(m)


def print_dfa(dfa):
    print(f"  DFA: {len(dfa.states)} states, initial {dfa.initial_state}, "
          f"accepting {sorted(dfa.final_states)}")
    for s in sorted(dfa.states):
        row = {int(k): v for k, v in dfa.transitions[s].items()}
        parts = [f"{NAME.get(k, k)}->{row[k]}" for k in sorted(row)]
        mark = "ACC" if s in dfa.final_states else "REJ"
        init = "*" if s == dfa.initial_state else " "
        print(f"    {init}{s} [{mark}]: " + "  ".join(parts))


def closed(seq):
    fs = set()
    for ph in range(3):
        sub = seq[ph:]
        if any(tuple(sub[i:i+3]) in STOPS for i in range(0, len(sub)-2, 3)):
            fs.add(ph)
    return frozenset(fs)


samp = SuperSampler(V, 36)
rng = np.random.default_rng(11)
ws = [samp.sample(rng, V.alphabet_size) for _ in range(8000)]
bs = V.compile_many(ws, [np.random.default_rng(i) for i in range(8000)])
cf = [closed(b) for b in bs]

for r in (0, 1):
    d = pickle.load(open(f"{DUMP}/round_0{r}.pkl", "rb"))
    dfa, dt = d["dfa"], d["dt"]
    print(f"\n########## ROUND {r}  (est {d.get('est'):.3f}, boundary {d.get('decision_boundary'):.4f}) ##########")
    print_dfa(dfa)
    call = np.array([bool(dfa.accepts_input(w)) for w in ws])
    print(f"  accept-rate {call.mean():.3f}")
    groups = {}
    for c, a in zip(cf, call):
        groups.setdefault(c, []).append(a)
    print("  reject-rate by closed-frame set:")
    for k in sorted(groups, key=lambda s: (len(s), sorted(s))):
        a = np.array(groups[k])
        fs = "{" + ",".join(map(str, sorted(k))) + "}"
        print(f"    {fs:<9}: reject {1-a.mean():.3f} (n={len(a)})")
    print(f"  decision tree: {dt.num_states} states, depth {dt.depth}, "
          f"base_family size {len(dt.base_family)}")
    print("  tree (each internal node = distinguishing midfix; False-child then True-child):")
    for line in dt.render(render_midfix, indent=4):
        print("  " + line)
