"""Definitive (real-oracle) member counts for v2_s1's self-loops, from the round-0
dump -- bounded (no full FNR re-run).  Sift the dumped representative prefixes to
their tree leaves with the real gate oracle (batched per level via classify_many),
count members per leaf, and cross-reference the self-loop source states.  Also, for
each self-loop edge (state,c), check whether member+[c] sifts indecisively.
"""
import pickle, re, numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.l_star.midfix_tree import oracle_decider
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary
from orthogonal_dfa.superlanguage.oracle import LiftedOracle

D = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad"
V = KmerVocabulary(kmers=((3, 0, 2), (2, 2, 3, 0), (3, 2, 0), (3, 0, 0), (0, 2, 2, 3)), base_alphabet_size=4)
gate = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
oracle = LiftedOracle(gate, V, num_compilations=1, seed=1, noise_model=None)

d = pickle.load(open(f"{D}/rounds_v2_s1/round_00.pkl", "rb"))
dt, dfa, boundary = d["dt"], d["dfa"], float(d["decision_boundary"])
prefixes = [p for p, rep in zip(d["prefixes"], d["representative"]) if rep]
srcs = sorted(set(int(m) for m in re.findall(r"no decisive edge for \(state (\d+)", open(f"{D}/run_v2_s1.log").read())))
print(f"round-0: {len(dfa.states)} states, {len(prefixes)} rep prefixes, boundary {boundary:.4f}")
print(f"self-loop source states (all rounds): {srcs}\n", flush=True)

# decide over the round's suffix family, read decisively at the boundary
decide, decide_level = oracle_decider(oracle, dt.base_family, boundary, boundary)

# leaf occupancy = # prefixes sifting to each leaf (batched per level via classify_many)
leaves = dt.classify_many(prefixes, decide_level)
from collections import Counter
occ = Counter(l for l in leaves if l is not None)
undecided = sum(1 for l in leaves if l is None)
print("leaf occupancy (members) per state  [* = self-loop source]:")
for s in sorted(dfa.states):
    star = " *" if s in srcs else ""
    rej = " REJECT" if s not in dfa.final_states else ""
    print(f"  state {s}: {occ.get(s,0)} members{rej}{star}")
print(f"  (prefixes that could not be placed: {undecided})")
src_members = [occ.get(s, 0) for s in srcs if s in dfa.states]
print(f"\nself-loop source member counts (round-0 states): {src_members}")
print(f"MEMBER_LIMIT ceiling 1500; {len([m for m in src_members if m < 10])} of {len(src_members)} "
      f"self-loop sources have <10 members (thin), "
      f"{len([m for m in src_members if m >= 30])} have >=30 (well-powered)")
