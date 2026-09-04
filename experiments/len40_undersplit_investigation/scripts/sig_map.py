import pickle, numpy as np, collections
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary
D1=pickle.load(open("seedsweep/dumps_s1/round_00.pkl","rb"))["dfa"]
D3=pickle.load(open("seedsweep/dumps_s3/round_00.pkl","rb"))["dfa"]
def walk(dfa,w):
    s=dfa.initial_state
    for c in w: s=dfa.transitions[s][c]
    return s
def sig(w):
    wc=0; f0=f1=False
    for c in w:
        if c>=3: wc+=1
        else:
            ph=wc%3
            if ph==0: f0=True
            elif ph==1: f1=True
    return "SINK" if (f0 and f1) else (f0,f1,wc%3)
vocab=KmerVocabulary(kmers=((3,0,2),(3,0,0),(3,2,0)),base_alphabet_size=4)
samp=SuperSampler(vocab,20); rng=np.random.default_rng(3)
# use many short prefixes to populate all signatures
pf=[list(samp.sample(rng,vocab.alphabet_size)) for _ in range(20000)]
sig2s1=collections.defaultdict(collections.Counter)
s12sig=collections.defaultdict(collections.Counter)
for w in pf:
    g=sig(w); a=walk(D1,w)
    sig2s1[g][a]+=1; s12sig[a][g]+=1
print("=== signature -> seed1 state (dominant) ===")
for g in sorted(sig2s1,key=str):
    tot=sum(sig2s1[g].values()); dom,cnt=sig2s1[g].most_common(1)[0]
    print(f"  {str(g):16} -> s1={dom} ({cnt/tot*100:.0f}% of {tot})")
print("\n=== seed1 state -> which signatures it MERGES (its true MN-class) ===")
for a in sorted(s12sig):
    sigs=[g for g,_ in s12sig[a].most_common()]
    acc=a in D1.final_states
    print(f"  s1={a}(acc={acc}) merges signatures: {sigs}")
