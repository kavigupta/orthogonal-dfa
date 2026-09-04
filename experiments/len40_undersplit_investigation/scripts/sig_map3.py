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
pf=[list(samp.sample(rng,vocab.alphabet_size)) for _ in range(30000)]
m=collections.defaultdict(collections.Counter)
for w in pf:
    m[sig(w)][ (walk(D1,w), walk(D3,w)) ]+=1
print("signature      -> (seed1_state[acc], seed3_state[acc])   dominant")
for g in sorted(m,key=str):
    (a,b),c=m[g].most_common(1)[0]; tot=sum(m[g].values())
    print(f"  {str(g):16} -> s1={a}[{'A' if a in D1.final_states else 'R'}]  s3={b}[{'A' if b in D3.final_states else 'R'}]   ({c/tot*100:.0f}%/{tot})")
# check merged pairs
print("\nTRUE-merged pairs (seed1 merges) -- does seed3 keep them together?")
pairs=[((True,False,1),(False,True,0)),((True,False,0),(False,True,2)),((False,True,1),(True,False,2))]
def s3of(g):
    for w in pf:
        if sig(w)==g: return walk(D3,w)
for x,y in pairs:
    sx,sy=s3of(x),s3of(y)
    print(f"  {x} & {y}: seed3 states {sx}[{'A' if sx in D3.final_states else 'R'}] vs {sy}[{'A' if sy in D3.final_states else 'R'}]  -> {'SPLIT APART (spurious)' if sx!=sy else 'kept together'}")
