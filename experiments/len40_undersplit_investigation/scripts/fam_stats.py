import pickle, sys, numpy as np
# usage: fam_stats.py <dump.pkl> <label>
d = pickle.load(open(sys.argv[1], "rb"))
label = sys.argv[2]
fam = d["dt"].base_family
def syms(s):  # bytes or list -> list[int]
    return list(s)
fam = [syms(s) for s in fam]
n = len(fam)
lens = np.array([len(s) for s in fam])
# kmers are symbols 0,1,2; wildcards 3,4
nk = np.array([sum(1 for c in s if c < 3) for s in fam])       # kmers per suffix
# kmer identity counts across whole family
idcount = {0: 0, 1: 0, 2: 0}
# suffix-internal position (index) parity of each kmer
posmod = {0: 0, 1: 0, 2: 0}
for s in fam:
    for i, c in enumerate(s):
        if c < 3:
            idcount[c] += 1
            posmod[i % 3] += 1
tot = sum(idcount.values())
print(f"{label}: n={n} len={lens.min()}-{lens.max()} "
      f"kmers/suffix mean={nk.mean():.3f} sd={nk.std():.3f} "
      f"(min {nk.min()} max {nk.max()}, frac_with_0={np.mean(nk==0):.3f})")
print(f"    kmer identity frac: TAG={idcount[0]/tot:.3f} TAA={idcount[1]/tot:.3f} TGA={idcount[2]/tot:.3f}")
print(f"    kmer suffix-index %3: p0={posmod[0]/tot:.3f} p1={posmod[1]/tot:.3f} p2={posmod[2]/tot:.3f}")
