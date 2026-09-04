"""Per-target-state diagnostic across rounds for the len40 spliceai multi-round run.

The DFA we're going for is the frame-0-and-1 rule; its states are the signatures
sig(s) = (f0-kmer-closed, f1-kmer-closed, phase), with (f0&f1)->SINK.  For each
saved round we ask, per target state:
  (A) how many harvested `indecisive` strings fall in it -- i.e. which states the
      family still can't place -- and whether that shrinks round over round;
  (B) how the produced DFA covers it on a fresh eval set: is it one produced state
      correctly labelled, or split across several / mislabelled.
"""
import pickle, sys, collections
import numpy as np
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

SEED = 2
DUMP = "mr_dumps/seed%d" % SEED

def sig(s):
    wc = 0; f0 = f1 = False
    for c in s:
        if c >= 3:
            wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return "SINK" if (f0 and f1) else (f0, f1, wc % 3)

def load(r):
    with open("%s/round_%02d.pkl" % (DUMP, r), "rb") as f:
        return pickle.load(f)

rounds = []
r = 0
while True:
    try:
        rounds.append(load(r)); r += 1
    except FileNotFoundError:
        break
print(f"loaded rounds: {len(rounds)}  (seed {SEED})")

# (A) indecisive strings per target signature, per round
print("\n=== (A) |indecisive| per target state, by round ===")
allsigs = set()
per = []
for rec in rounds:
    c = collections.Counter(sig(bytes(x)) for x in rec["indecisive"])
    per.append(c); allsigs |= set(c)
order = sorted(allsigs, key=lambda k: (k == "SINK", k))
hdr = "  target-state       " + "".join(f"  r{i}" for i in range(len(rounds)))
print(hdr)
for k in order:
    print(f"  {str(k):18}" + "".join(f"  {per[i].get(k,0):3d}" for i in range(len(rounds))))
print(f"  {'TOTAL':18}" + "".join(f"  {sum(per[i].values()):3d}" for i in range(len(rounds))))

# (B) produced-DFA coverage of each target state on a fresh eval set
print("\n=== (B) produced-DFA states per target state, by round ===")
vocab = KmerVocabulary(kmers=((3,0,2),(3,0,0),(3,2,0)), base_alphabet_size=4)
samp = SuperSampler(vocab, 36); rng = np.random.default_rng(SEED + 999)
supers = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(4000)]
sigs = [sig(w) for w in supers]
def walk(dfa, w):
    s = dfa.initial_state
    for c in w: s = dfa.transitions[s][c]
    return s
for ri, rec in enumerate(rounds):
    dfa = rec["dfa"]
    st = [walk(dfa, w) for w in supers]
    print(f"\n  -- round {ri}: {len(dfa.states)} states, accept {rec['accept_rate']:.3f}, "
          f"phi(oracle) {rec['phi_oracle']:+.3f} --")
    bytarget = collections.defaultdict(collections.Counter)
    for g, s in zip(sigs, st):
        bytarget[g][s] += 1
    for g in sorted(bytarget, key=lambda k: (k == "SINK", k)):
        cc = bytarget[g]; n = sum(cc.values())
        states = sorted(cc, key=lambda s: -cc[s])
        acc = {s: (s in dfa.final_states) for s in states}
        frag = "  <-- split" if len(states) > 1 and cc[states[1]] / n > 0.15 else ""
        top = ", ".join(f"s{s}[{'A' if acc[s] else 'R'}]:{cc[s]}" for s in states[:4])
        print(f"     {str(g):16} n={n:4d}: {top}{frag}")
