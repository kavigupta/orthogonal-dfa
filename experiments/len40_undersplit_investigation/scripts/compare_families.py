"""Compare the suffix family (dt.base_family) of round 0 vs round 1 of the
stop-codon run. Symbols: 0,1,2 = TAG/TAA/TGA (stops), 3,4 = X,Y (wildcards)."""
import pickle
from collections import Counter
import numpy as np

DUMP = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/round_dumps"
NAME = {0: "TAG", 1: "TAA", 2: "TGA", 3: "X", 4: "Y"}


def summarize(fam, label):
    n = len(fam)
    n_stops = [sum(s < 3 for s in v) for v in fam]      # kmer (stop) symbols per suffix
    lens = [len(v) for v in fam]
    pure_wild = sum(k == 0 for k in n_stops)
    has_stop = n - pure_wild
    print(f"\n{label}: {n} suffixes")
    print(f"  pure-wildcard (X/Y only):     {pure_wild:5d}  ({pure_wild/n:.1%})")
    print(f"  contains >=1 stop codon:      {has_stop:5d}  ({has_stop/n:.1%})")
    print(f"  #stops per suffix: mean {np.mean(n_stops):.2f}, distribution {dict(sorted(Counter(n_stops).items()))}")
    print(f"  length: mean {np.mean(lens):.1f}, min {min(lens)}, max {max(lens)}, dist {dict(sorted(Counter(lens).items()))}")
    return {tuple(v) for v in fam}


fams = {}
for r in (0, 1):
    d = pickle.load(open(f"{DUMP}/round_0{r}.pkl", "rb"))
    fams[r] = summarize(d["dt"].base_family, f"ROUND {r} suffix family")

inter = fams[0] & fams[1]
print(f"\noverlap: {len(inter)} suffixes shared "
      f"({len(inter)/len(fams[0]):.1%} of round0, {len(inter)/len(fams[1]):.1%} of round1)")
print(f"  only in round0: {len(fams[0]-fams[1])}   only in round1: {len(fams[1]-fams[0])}")

# among the NEW suffixes in round 1 (not in round 0), how many are pure-wildcard?
new1 = fams[1] - fams[0]
pw_new = sum(all(s >= 3 for s in v) for v in new1)
print(f"  of the {len(new1)} suffixes new to round1, pure-wildcard: {pw_new} ({pw_new/max(len(new1),1):.1%})")
dropped = fams[0] - fams[1]
pw_drop = sum(all(s >= 3 for s in v) for v in dropped)
print(f"  of the {len(dropped)} dropped (round0-only), pure-wildcard: {pw_drop} ({pw_drop/max(len(dropped),1):.1%})")

# show a few pure-wildcard suffixes present in each
for r in (0, 1):
    pw = [v for v in fams[r] if all(s >= 3 for s in v)]
    ex = ["".join(NAME[s] for s in v) for v in list(pw)[:6]]
    print(f"  round{r} pure-wildcard examples: {ex}")
