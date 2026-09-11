import pickle, numpy as np
from orthogonal_dfa.superlanguage import KmerVocabulary, SuperSampler
NAME = {0: "TAG", 1: "GGTA", 2: "TGA", 3: "TAA", 4: "AGGT", 5: "X", 6: "Y"}
STOPS_SYM = {0, 2, 3}   # TAG, TGA, TAA super-symbols
FOURMERS = {1, 4}       # GGTA, AGGT
d = pickle.load(open("/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/rounds_v2/round_00.pkl", "rb"))
dfa = d["dfa"]
T = {s: {int(k): v for k, v in dfa.transitions[s].items()} for s in dfa.states}
rej = [s for s in dfa.states if s not in dfa.final_states]
print("reject (non-accepting) states:", rej)
# absorbing states
for s in dfa.states:
    if all(T[s][a] == s for a in T[s]):
        print(f"  state {s} is ABSORBING ({'reject' if s in rej else 'accept'})")
# which (state,symbol) enter a reject state
print("transitions that ENTER a reject state:")
for s in dfa.states:
    for a, nx in T[s].items():
        if nx in rej:
            print(f"  state {s} --{NAME[a]}--> {nx}")
# pure-wildcard trace from init (symbol 5 = X)
s = dfa.initial_state; seq = [s]
for _ in range(9):
    s = T[s][5]; seq.append(s)
print("pure-wildcard state cycle from init:", seq)
# empirically: among frame0-closed strings the DFA ACCEPTS, how many contain a 4-mer?
V = KmerVocabulary(kmers=((3,0,2),(2,2,3,0),(3,2,0),(3,0,0),(0,2,2,3)), base_alphabet_size=4)
STOPS = {(3,0,2),(3,0,0),(3,2,0)}
def f0_closed(seq):
    return any(tuple(seq[i:i+3]) in STOPS for i in range(0, len(seq)-2, 3))
samp = SuperSampler(V, 36); rng = np.random.default_rng(3)
ws = [samp.sample(rng, V.alphabet_size) for _ in range(8000)]
bs = V.compile_many(ws, [np.random.default_rng(i) for i in range(8000)])
call = np.array([bool(dfa.accepts_input(w)) for w in ws])
f0 = np.array([f0_closed(b) for b in bs])
has4 = np.array([any(x in FOURMERS for x in w) for w in ws])
m = f0 & call            # frame0 closed BUT accepted (the ~15% exceptions)
m2 = f0 & ~call          # frame0 closed and rejected
print(f"\nframe0 closed: n={f0.sum()}")
print(f"  of these, DFA rejects {(~call[f0]).mean():.3f}, accepts {call[f0].mean():.3f}")
print(f"  among frame0-closed-but-ACCEPTED: contain a 4-mer (GGTA/AGGT)? {has4[m].mean():.3f}  (n={m.sum()})")
print(f"  among frame0-closed-and-REJECTED: contain a 4-mer?             {has4[m2].mean():.3f}  (n={m2.sum()})")
print(f"  baseline: any string contains a 4-mer? {has4.mean():.3f}")
