"""phi-score each round's dumped DFA (over the super alphabet) against the real
oracle and against all-frames-closed -- the recovery metric.  Reads the round_*.pkl
dumps so we don't have to wait for the whole run to finish.
"""
import glob, os, pickle
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage import KmerVocabulary, SuperSampler

TAG, TAA, TGA = (3, 0, 2), (3, 0, 0), (3, 2, 0)
STOPS = {TAG, TAA, TGA}
DUMP = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/round_dumps"


def n_frames_closed(seq):
    n = 0
    for phase in range(3):
        sub = seq[phase:]
        if any(tuple(sub[i:i+3]) in STOPS for i in range(0, len(sub)-2, 3)):
            n += 1
    return n


def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
vocab = KmerVocabulary(kmers=(TAG, TAA, TGA), base_alphabet_size=4)
samp = SuperSampler(vocab, 36)
rng = np.random.default_rng(0x5EED)
N = 4000
supers = [samp.sample(rng, vocab.alphabet_size) for _ in range(N)]
bases = [vocab.compile(w, rng) for w in supers]
ora = np.asarray(base.membership_queries(bases)).astype(float)
nfr = np.array([n_frames_closed(b) for b in bases])
afc = (nfr == 3).astype(float)
print(f"eval N={N}: oracle accept {ora.mean():.3f}, "
      f"phi(oracle, all-frames-closed) = {phi(ora, afc):+.3f}  [baseline ceiling]", flush=True)

for path in sorted(glob.glob(os.path.join(DUMP, "round_*.pkl"))):
    with open(path, "rb") as f:
        d = pickle.load(f)
    dfa = d.get("dfa")
    if dfa is None:
        print(f"{os.path.basename(path)}: no dfa ({d.get('dump_error')})")
        continue
    call = np.array([bool(dfa.accepts_input(w)) for w in supers]).astype(float)
    tag = ""
    if call.std() == 0:
        tag = " ACCEPT-ALL" if call.mean() > 0.5 else " REJECT-ALL"
    print(f"{os.path.basename(path)}: round {d.get('round')}, {len(dfa.states)} states, "
          f"est {d.get('est'):.3f}, accept-rate {call.mean():.3f}{tag}")
    print(f"    phi(DFA, oracle) = {phi(call, ora):+.3f}    phi(DFA, all-frames-closed) = {phi(call, afc):+.3f}", flush=True)
