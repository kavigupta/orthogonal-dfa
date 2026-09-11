"""Are self-loops 'confidently unconfident'?  For v2_s1's round-0 DFA (13 self-loops,
accept-all), compute the leaf occupancy of each state -- how many representative
prefixes sift to it -- and cross-reference the states that were self-loop SOURCES.
_MEMBER_LIMIT=1500 is the CEILING; the question is the floor: do the self-looped
source states have enough members for 'indecisive' to be well-powered, or are they
transient leaves decided on a handful of members?
"""
import pickle, re, numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

D = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad"
V = KmerVocabulary(kmers=((3, 0, 2), (2, 2, 3, 0), (3, 2, 0), (3, 0, 0), (0, 2, 2, 3)), base_alphabet_size=4)
base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)

d = pickle.load(open(f"{D}/rounds_v2_s1/round_00.pkl", "rb"))
dt, dfa, boundary = d["dt"], d["dfa"], float(d["decision_boundary"])
prefixes = [p for p, rep in zip(d["prefixes"], d["representative"]) if rep]
print(f"v2_s1 round 0: {dfa} states={len(dfa.states)}, {len(prefixes)} representative prefixes, boundary {boundary:.4f}")

# self-loop source states from the run log (round-0 export is the first block)
log = open(f"{D}/run_v2_s1.log").read()
srcs = [int(m) for m in re.findall(r"no decisive edge for \(state (\d+)", log)]
from collections import Counter
src_ct = Counter(srcs)
print(f"self-loop source states (all rounds): {dict(src_ct)}")

# decide callback = suffix-family is_accept over the round's family
fam = [dt.base_family[i] if False else v for i, v in enumerate(dt.base_family)]  # base_family suffixes
def decide(seq, midfix):
    vs = [list(seq) + list(midfix) + list(v) for v in dt.base_family]
    m = np.asarray(base.membership_queries(
        [V.compile(s, np.random.default_rng(hash((tuple(s), 7)) % (2**32))) for s in vs])).mean()
    if m >= boundary: return True
    if m < boundary: return False
    return None

# sift each prefix through the tree -> leaf; count occupancy
occ = Counter()
for p in prefixes:
    leaf = dt.classify(p, decide)
    occ[leaf] += 1
print("\nleaf occupancy (state -> #prefixes sifting there):")
for s in sorted(dfa.states):
    mark = " <-- SELF-LOOP SOURCE" if s in src_ct else ""
    rej = " [REJECT]" if s not in dfa.final_states else ""
    print(f"  state {s}: {occ.get(s,0)} members{rej}{mark}")
src_occ = [occ.get(s, 0) for s in src_ct]
print(f"\nself-loop source states occupancy: {src_occ}  (median {int(np.median(src_occ)) if src_occ else 0})")
print(f"MEMBER_LIMIT ceiling is 1500; total prefixes {len(prefixes)}")
