"""THE open question: does round 1's grown boundary-prefix pool lift the per-cell
stop-vs-wildcard signal above the ~0.45 compilation-noise floor -- which would let
identify_cluster_around start selecting stop-heavy suffixes (explaining the
stop-enriched round-1 family)?

For round-0 representative prefixes and for round-1's NEWLY-GROWN boundary prefixes,
measure how much a stop-heavy vs a pure-wildcard suffix disagrees with the empty
seed column.  If (stop - wild) disagreement is ~0 on round-0 prefixes but positive
on the grown ones, that's the mechanism.
"""
import pickle, numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary
from orthogonal_dfa.superlanguage.sampler import SuperSampler

D = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/rounds_v1_s1"
V = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
X, Y = V.wildcard_symbols
base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)

r0 = pickle.load(open(f"{D}/round_00.pkl", "rb"))
r1 = pickle.load(open(f"{D}/round_01.pkl", "rb"))
p0 = [tuple(p) for p, rep in zip(r0["prefixes"], r0["representative"]) if rep]
all1 = {tuple(p) for p in r1["prefixes"]}
grown = [list(p) for p in (all1 - set(p0))]          # boundary prefixes added for round 1
p0 = [list(p) for p in p0]
print(f"round-0 representative prefixes: {len(p0)}")
print(f"round-1 total prefixes: {len(all1)}, NEWLY grown (boundary): {len(grown)}")

samp = SuperSampler(V, 36)
rng = np.random.default_rng(0)
wild = [[int(rng.choice([X, Y])) for _ in range(36)] for _ in range(20)]
stopheavy = []
while len(stopheavy) < 20:
    w = samp.sample(rng, V.alphabet_size)
    if sum(s < 3 for s in w) >= 3:
        stopheavy.append(w)

def compile1(w, seed):
    return V.compile(w, np.random.default_rng(seed))

def gap_on(prefixes, tag):
    if len(prefixes) < 20:
        print(f"{tag}: only {len(prefixes)} prefixes, skipping"); return
    P = min(len(prefixes), 300)
    pr = prefixes[:P]
    e = np.asarray(base.membership_queries([compile1(p, i) for i, p in enumerate(pr)])).astype(int)
    def dis(suffixes, off):
        d = []
        for j, s in enumerate(suffixes):
            col = np.asarray(base.membership_queries(
                [compile1(list(p) + list(s), off + j * P + i) for i, p in enumerate(pr)])).astype(int)
            d.append(float((col != e).mean()))
        return np.mean(d)
    dw = dis(wild, 100000); ds = dis(stopheavy, 500000)
    print(f"{tag} (P={P}, seed accept {e.mean():.3f}): wild-disagree {dw:.3f}, "
          f"stop-disagree {ds:.3f}, GAP(stop-wild) {ds-dw:+.3f}", flush=True)

gap_on(p0, "round-0 representative")
gap_on(grown, "round-1 GROWN boundary")
