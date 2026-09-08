"""Is the round-0 vs round-1 difference a MISSING SPLIT or a MIS-RESOLVED EDGE?

Walk the eval set through BOTH DFAs, cross-tabulate (r0_state, r1_state), and for each
IMPURE round-0 state (mixes SINK & live) show where round 1 routes its SINK vs live
members.

- If round 1 sends an impure round-0 state's SINK members to one (pure) round-1 state and
  its live members to another -> round 1 makes a state distinction round 0 lacks (round 0
  is missing that split / has too few states along the SINK dimension).
- If round 1 also merges them into one impure state -> the difference is NOT this split.

Also report whether round 1's partition REFINES round 0's (each round-1 state comes from a
single round-0 state) or cross-cuts it, in both directions -- refinement one way = the
other undersplit.  Oracle-free: pure DFA walking.
"""
import pickle, collections
import numpy as np
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

SEED = 2
NEVAL = 4000
D = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/mr_dumps/seed2"


def sig(s):
    wc = 0; f0 = f1 = False
    for c in s:
        if c >= 3: wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return "SINK" if (f0 and f1) else "live"


def walk(dfa, w):
    s = dfa.initial_state
    for c in w:
        s = dfa.transitions[s][c]
    return s


def refinement(a_states, b_states):
    """Fraction of B-states that map to a single A-state (B refines A on that fraction)."""
    by = collections.defaultdict(set)
    for a, b in zip(a_states, b_states):
        by[b].add(a)
    pure = sum(1 for b, aset in by.items() if len(aset) == 1)
    return pure, len(by)


def main():
    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    d0 = pickle.load(open(f"{D}/round_00.pkl", "rb"))["dfa"]
    d1 = pickle.load(open(f"{D}/round_01.pkl", "rb"))["dfa"]
    a0 = getattr(d0, "final_states", set()); a1 = getattr(d1, "final_states", set())
    samp = SuperSampler(vocab, 36); rng = np.random.default_rng(SEED + 999)
    supers = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(NEVAL)]
    sk = [sig(w) for w in supers]
    s0 = [walk(d0, w) for w in supers]
    s1 = [walk(d1, w) for w in supers]

    # round-0 state purity
    by0 = collections.defaultdict(lambda: {"SINK": 0, "live": 0})
    for a, g in zip(s0, sk):
        by0[a][g] += 1
    impure = [q for q in by0 if min(by0[q]["SINK"], by0[q]["live"]) >= 10]
    impure.sort(key=lambda q: -min(by0[q]["SINK"], by0[q]["live"]))
    desc = ["s%d(%dS/%dL,%s)" % (q, by0[q]["SINK"], by0[q]["live"], "A" if q in a0 else "r") for q in impure]
    print("round 0 impure states (both>=10): " + ", ".join(desc))

    # refinement both directions
    p_b_in_a, nb = refinement(s0, s1)   # r1 refines r0?
    p_a_in_b, na = refinement(s1, s0)   # r0 refines r1?
    print(f"\nrefinement: {p_b_in_a}/{nb} round-1 states come from a single round-0 state "
          f"(round1 refines round0)")
    print(f"            {p_a_in_b}/{na} round-0 states come from a single round-1 state "
          f"(round0 refines round1)")

    # for each impure round-0 state, where round 1 routes SINK vs live members
    for q in impure[:4]:
        sinkR1 = collections.Counter(r1 for r1, r0, g in zip(s1, s0, sk) if r0 == q and g == "SINK")
        liveR1 = collections.Counter(r1 for r1, r0, g in zip(s1, s0, sk) if r0 == q and g == "live")
        def fmt(cnt):
            return "  ".join(f"s{r}({'A' if r in a1 else 'r'}):{n}" for r, n in cnt.most_common(4))
        print(f"\nround-0 s{q} ({'A' if q in a0 else 'r'}): its members in round 1 ->")
        print(f"    SINK -> {fmt(sinkR1)}")
        print(f"    live -> {fmt(liveR1)}")
        # purity of those round-1 targets
        tgts = set(sinkR1) | set(liveR1)
        pur = []
        for r in tgts:
            ss = sum(1 for r1, g in zip(s1, sk) if r1 == r and g == "SINK")
            ll = sum(1 for r1, g in zip(s1, sk) if r1 == r and g == "live")
            pur.append(f"s{r}:{ss}S/{ll}L")
        print(f"    (round-1 target purity: {'  '.join(pur)})")


if __name__ == "__main__":
    main()
