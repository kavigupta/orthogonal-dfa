"""Reconstruct round 1's learned rule offline from the DFA printed in the log,
scored against the SAME eval set round_00.pkl stored (shared seed => shared
bases/fpat/ora).  No oracle / torch needed for the DFA-acc column."""
import pickle
import numpy as np
from automata.fa.dfa import DFA
from orthogonal_dfa.superlanguage import KmerVocabulary, SuperSampler

TAG, TAA, TGA = (3, 0, 2), (3, 0, 0), (3, 2, 0)
STOPS = {TAG, TAA, TGA}


def frames_closed(seq):
    out = []
    for ph in range(3):
        sub = seq[ph:]
        out.append(any(tuple(sub[i:i + 3]) in STOPS for i in range(0, len(sub) - 2, 3)))
    return tuple(out)


def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


# --- round-1 DFA, transcribed verbatim from the run log ---
R1 = DFA(
    states=set(range(12)),
    input_symbols=set(range(5)),
    transitions={
        0: {0: 0, 1: 0, 2: 0, 3: 9, 4: 9},
        1: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        2: {0: 1, 1: 1, 2: 1, 3: 4, 4: 4},
        3: {0: 11, 1: 3, 2: 3, 3: 2, 4: 2},
        4: {1: 4, 2: 4, 0: 0, 3: 3, 4: 3},
        5: {3: 8, 4: 10, 0: 3, 1: 3, 2: 3},
        6: {0: 6, 1: 6, 2: 6, 3: 10, 4: 10},
        7: {0: 4, 1: 4, 2: 4, 3: 6, 4: 6},
        8: {0: 4, 1: 4, 2: 4, 3: 0, 4: 0},
        9: {0: 6, 1: 6, 2: 0, 3: 8, 4: 8},
        10: {3: 7, 4: 7, 0: 3, 1: 11, 2: 11},
        11: {0: 3, 1: 11, 2: 3, 3: 1, 4: 1},
    },
    initial_state=10,
    final_states={0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    allow_partial=False,
)

# --- rebuild the SAME eval set the run used (seed 0 => +999 / range(i)) ---
kmers = ((3, 0, 2), (3, 0, 0), (3, 2, 0))
vocab = KmerVocabulary(kmers=kmers, base_alphabet_size=4)
samp = SuperSampler(vocab, 36)
rng = np.random.default_rng(0 + 999)
supers = [samp.sample(rng, vocab.alphabet_size) for _ in range(4000)]
bases = vocab.compile_many(supers, [np.random.default_rng(i) for i in range(4000)])
fpat = [frames_closed(b) for b in bases]

# sanity: the stored eval set must match what we regenerated
with open("rule_dumps/round_00.pkl", "rb") as f:
    d0 = pickle.load(f)
assert list(map(tuple, d0["fpat"])) == fpat, "eval set drift -- seeds differ!"
ora = np.asarray(d0["ora"], float)
print(f"eval set matches round_00.pkl ({len(fpat)} strings); reusing stored oracle labels")

nfc = np.array([sum(p) for p in fpat])
afc = (nfc == 3).astype(float)
call = np.array([bool(R1.accepts_input(w)) for w in supers], float)

print(f"\n[round 1] DFA: 12 states, accept-rate {call.mean():.3f}")
print(f"  phi(DFA, oracle)            = {phi(call, ora):+.3f}")
print(f"  phi(DFA, all-frames-closed) = {phi(call, afc):+.3f}")
print("  rule by frame pattern (f0f1f2): count  DFA-acc  oracle-acc")
for pat in sorted({p for p in fpat}, key=lambda p: (sum(p), p)):
    m = np.array([p == pat for p in fpat])
    if m.sum() == 0:
        continue
    bits = "".join("C" if f else "." for f in pat)
    print(f"    {bits}  ({sum(pat)}closed): {int(m.sum()):5d}  {call[m].mean():.3f}    {ora[m].mean():.3f}")
print("  rule by #frames closed: n  count  DFA-acc  oracle-acc")
for n in range(4):
    m = nfc == n
    if m.sum():
        print(f"    n={n}: {int(m.sum()):5d}  {call[m].mean():.3f}    {ora[m].mean():.3f}")

# dump round 1 too, mirroring the in-process format
with open("rule_dumps/round_01.pkl", "wb") as f:
    pickle.dump(dict(dfa=R1, boundary=0.5949, call=call, fpat=fpat, ora=ora), f)
print("\nwrote rule_dumps/round_01.pkl")
