import pickle, numpy as np, collections
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

d1 = pickle.load(open("seedsweep/dumps_s1/round_00.pkl", "rb")); D1 = d1["dfa"]
d3 = pickle.load(open("seedsweep/dumps_s3/round_00.pkl", "rb")); D3 = d3["dfa"]

def walk(dfa, w):
    s = dfa.initial_state
    for c in w:
        s = dfa.transitions[s][c]
    return s

STOPS = {(3, 0, 2), (3, 0, 0), (3, 2, 0)}
def f01(seq):
    c = lambda ph: any(tuple(seq[ph:][i:i+3]) in STOPS for i in range(0, len(seq[ph:])-2, 3))
    return c(0) and c(1)

vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
samp = SuperSampler(vocab, 36)
rng = np.random.default_rng(7)
supers = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(6000)]
bases = vocab.compile_many(supers, [np.random.default_rng(i) for i in range(6000)])
tgt = np.array([0 if f01(b) else 1 for b in bases])

s1 = np.array([walk(D1, w) for w in supers])
s3 = np.array([walk(D3, w) for w in supers])
acc1 = np.array([int(x in D1.final_states) for x in s1])
acc3 = np.array([int(x in D3.final_states) for x in s3])
print(f"seed1(crisp 8st): acc {acc1.mean():.3f}  agree-with-frame-rule {np.mean(acc1==tgt):.3f}")
print(f"seed3(coll 11st): acc {acc3.mean():.3f}  agree-with-frame-rule {np.mean(acc3==tgt):.3f}")

refine = collections.defaultdict(set)
for a, b in zip(s3, s1):
    refine[int(a)].add(int(b))
pure = sum(1 for v in refine.values() if len(v) == 1)
print(f"\nseed3: {len(refine)} reachable states; {pure}/{len(refine)} map to a UNIQUE seed1 "
      f"state (is-refinement={pure==len(refine)})")

byS1 = collections.defaultdict(list)
for s3st, s1set in refine.items():
    if len(s1set) == 1:
        byS1[next(iter(s1set))].append(s3st)
print("\nseed1 state (acc) -> seed3 states refining it (with their accept):")
for s1st in sorted(byS1):
    grp = sorted(byS1[s1st])
    accs = [(g, g in D3.final_states) for g in grp]
    over = len(grp) > 1
    flip = len({g in D3.final_states for g in grp}) > 1
    print(f"  s1={s1st}(acc={s1st in D1.final_states}) -> {accs}"
          f"{'  OVER-SPLIT' if over else ''}{'  +ACCEPT-FLIP!' if flip else ''}")
