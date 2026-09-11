"""Thread-1 extension: does the ACTUAL decisive_target decision get made on too few
samples, counting PROBE-added members (not just the prefix pool)?

Reconstruct the real LeafPopulation for v2_s1's round-0 dump: seed it with the 400
representative prefixes AND ~3000 sampled probe strings (the real run also anchors
probe substrings into leaves), using the real gate oracle + the round's suffix
family as classify -- then call population.members(path, 1500) for each self-loop
source state, exactly as EdgeResolver.leaf_members does.  Members that go indecisive
while sifting DROP OUT (LeafPopulation._push_chunk), so this measures the true
member pool reaching each leaf.

If the self-loop states still have few members with probes added -> the self-loop
decision really is made on too few samples.  If probes fill them up -> my earlier
prefix-only proxy undercounted.
"""
import pickle, re, numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.l_star.leaf_population import LeafPopulation
from orthogonal_dfa.l_star.midfix_tree import oracle_decider
from orthogonal_dfa.l_star.split_evidence import _MEMBER_LIMIT
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.oracle import LiftedOracle

D = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad"
V = KmerVocabulary(kmers=((3, 0, 2), (2, 2, 3, 0), (3, 2, 0), (3, 0, 0), (0, 2, 2, 3)), base_alphabet_size=4)
gate = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
oracle = LiftedOracle(gate, V, num_compilations=1, seed=1, noise_model=None)

d = pickle.load(open(f"{D}/rounds_v2_s1/round_00.pkl", "rb"))
dt, dfa, boundary = d["dt"], d["dfa"], float(d["decision_boundary"])
prefixes = [p for p, rep in zip(d["prefixes"], d["representative"]) if rep]
srcs = sorted(set(int(m) for m in re.findall(r"no decisive edge for \(state (\d+)", open(f"{D}/run_v2_s1.log").read())))
srcs = [s for s in srcs if s in dfa.states]
print(f"round-0: {len(dfa.states)} states, boundary {boundary:.4f}; self-loop sources {srcs}", flush=True)

# batched classify(strings, midfix) = suffix-family is_accept (matches _classify)
fam = dt.base_family
def classify(strings, midfix):
    if not strings:
        return []
    flat = [list(s) + list(midfix) + list(v) for s in strings for v in fam]
    ans = np.asarray(oracle.membership_queries(flat)).reshape(len(strings), len(fam))
    means = ans.mean(1)
    return [True if m >= boundary else (False if m < boundary else None) for m in means]

# real population: seed with prefixes (root) + sampled probes (root), like the run
pop = LeafPopulation(dt, classify)
for p in prefixes:
    pop.add(list(p))
NPROBE = 3000
samp = SuperSampler(V, 36)
rng = np.random.default_rng(1)
for _ in range(NPROBE):
    pop.add(samp.sample(rng, V.alphabet_size))
print(f"seeded population: {len(prefixes)} prefixes + {NPROBE} probes = {len(prefixes)+NPROBE} strings", flush=True)

print("\nleaf_members(state) via the REAL LeafPopulation (prefixes+probes, dropout applied):")
for s in sorted(dfa.states):
    m = pop.members(dt.path_of(s), _MEMBER_LIMIT)
    star = " <-- SELF-LOOP SOURCE" if s in srcs else ""
    rej = " REJECT" if s not in dfa.final_states else ""
    print(f"  state {s}: {len(m)} members{rej}{star}", flush=True)
src_counts = [len(pop.members(dt.path_of(s), _MEMBER_LIMIT)) for s in srcs]
print(f"\nself-loop source member counts (with probes): {dict(zip(srcs, src_counts))}")
print(f"thin (<10): {sum(c<10 for c in src_counts)}/{len(src_counts)}  |  well-powered (>=30): {sum(c>=30 for c in src_counts)}/{len(src_counts)}")
