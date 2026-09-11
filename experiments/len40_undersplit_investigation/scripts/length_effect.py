"""Test the length-effect hypothesis (no oracle needed).

Super-strings are fixed super-length 36, but kmer-symbols expand to 3 bases, so the
COMPILED base length varies by composition, and spliceai depends on it.  If the DFA's
walk partition tracks compiled length rather than frame structure, that's a length
effect.  We measure:
  - compiled-length distribution;
  - homogeneity of the walk partition vs 6 equal-count length buckets, vs SINK/live,
    vs frame-sig (higher = the partition explains that variable better);
  - per walk-state mean compiled length + its dominant frame-sig, to see if states are
    length-stratified.
"""
import argparse, pickle, collections, math
import numpy as np
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary


def sig(s):
    wc = 0; f0 = f1 = False
    for c in s:
        if c >= 3: wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return "SINK" if (f0 and f1) else (f0, f1, wc % 3)


def entropy(cs):
    n = sum(cs)
    return -sum((c / n) * math.log(c / n) for c in cs if c > 0) if n else 0.0


def homogeneity(states, labels):
    by = collections.defaultdict(collections.Counter); tot = collections.Counter(); lt = collections.Counter()
    for s, g in zip(states, labels):
        by[s][g] += 1; tot[s] += 1; lt[g] += 1
    H = entropy(list(lt.values())); n = len(states)
    Hc = sum((tot[s] / n) * entropy(list(sub.values())) for s, sub in by.items())
    return 1 - Hc / H if H else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--dump-dir", default="mr_dumps/seed2")
    ap.add_argument("--n-eval", type=int, default=8000)
    args = ap.parse_args()

    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    samp = SuperSampler(vocab, 36)
    rng = np.random.default_rng(args.seed + 999)
    supers = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(args.n_eval)]
    bases = vocab.compile_many(supers, [np.random.default_rng(i) for i in range(args.n_eval)])
    blen = np.array([len(b) for b in bases])
    sigs = [sig(w) for w in supers]
    sink = ["SINK" if g == "SINK" else "live" for g in sigs]
    # equal-count length buckets
    qs = np.quantile(blen, np.linspace(0, 1, 7))
    lbucket = np.clip(np.digitize(blen, qs[1:-1]), 0, 5).tolist()

    print(f"compiled base length: min {blen.min()} med {int(np.median(blen))} max {blen.max()} "
          f"mean {blen.mean():.1f} std {blen.std():.1f}")
    # length vs frame content
    for g in ["SINK", "live"]:
        m = np.array([s == g for s in sink])
        print(f"  mean compiled len | {g:4}: {blen[m].mean():.1f}")

    def walk(dfa, w):
        s = dfa.initial_state
        for c in w: s = dfa.transitions[s][c]
        return s

    r = 0
    while True:
        try:
            rec = pickle.load(open(f"{args.dump_dir}/round_{r:02d}.pkl", "rb"))
        except FileNotFoundError:
            break
        dfa = rec["dfa"]
        st = [walk(dfa, w) for w in supers]
        print(f"\nseed {args.seed} round {r}: {len(set(st))} states | walk-partition homogeneity vs:")
        print(f"   LENGTH-bucket {homogeneity(st, lbucket):.3f}   SINK/live {homogeneity(st, sink):.3f}   "
              f"frame-sig {homogeneity(st, sigs):.3f}")
        # per-state mean length + dominant sig
        by = collections.defaultdict(list); dom = collections.defaultdict(collections.Counter)
        for s, bl, g in zip(st, blen, sigs):
            by[s].append(bl); dom[s][g] += 1
        print("   state  n     meanLen  dominant-sig")
        for s in sorted(by, key=lambda k: -len(by[k]))[:8]:
            g = dom[s].most_common(1)[0][0]
            gg = "SINK" if g == "SINK" else f"{'T' if g[0] else 'F'}{'T' if g[1] else 'F'}p{g[2]}"
            print(f"   s{s:<3} {len(by[s]):5d}  {np.mean(by[s]):6.1f}   {gg}")
        r += 1


if __name__ == "__main__":
    main()
