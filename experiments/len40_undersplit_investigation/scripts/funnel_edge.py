"""Find the specific mis-resolved edge that funnels s10's SINK into s10, and show it is
resolved over a HETEROGENEOUS source state (so edge_resolver's arbitrary-first-member
pick lands on the wrong target).

1. members = s10's SINK-subset that SIFT to leaf 1 (round-0 tree separates them) + the
   live-subset.  Walk each through round-0's DFA recording the full state path (cheap, no
   oracle).  They all end at s10.
2. Per depth, occupancy of each state by SINK vs live.  The MERGE state M is the first
   state that holds both SINK and live on the way to s10 -- the undersplit source state.
3. At M: collect the prefixes that walk to M, take their next symbol c, and SIFT
   prefix+[c].  If those successors sift to DIFFERENT leaves (SINK-lineage -> s1-ish,
   live-lineage -> s10), the edge (M,c) is resolved over a heterogeneous set and the
   arbitrary-member rule mis-points it.  That is the incorrect edge resolution.
"""
import pickle, collections
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

SEED = 2
NEVAL = 4000
DUMP = "/tmp/claude-25787/-mnt-md0-orthogonal-dfa-3/e4e9621c-a36b-4b7e-a24d-819fbb2cab69/scratchpad/mr_dumps/seed2"


def sig(s):
    wc = 0; f0 = f1 = False
    for c in s:
        if c >= 3: wc += 1
        else:
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    return "SINK" if (f0 and f1) else "live"


def walk_path(dfa, w):
    s = dfa.initial_state; path = [s]
    for c in w:
        s = dfa.transitions[s][c]; path.append(s)
    return path


def sift_leaves(root, strings, fam, at, rt, oracle):
    """Route each string to its tree leaf (or None), batched per node."""
    n = len(strings); leaves = [None] * n
    def rec(node, idxs):
        if not isinstance(node, tuple):
            for i in idxs: leaves[i] = node
            return
        midfix, lk = node
        combos = [bytes(strings[i]) + midfix + v for i in idxs for v in fam]
        m = np.asarray(oracle.membership_queries(combos), float).reshape(len(idxs), len(fam)).mean(1)
        t_idx, f_idx = [], []
        for k, i in enumerate(idxs):
            if m[k] >= at: t_idx.append(i)
            elif m[k] < rt: f_idx.append(i)
        rec(lk[True], t_idx); rec(lk[False], f_idx)
    rec(root, list(range(n)))
    return leaves


def main():
    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
    oracle = LiftedOracle(base, vocab, seed=SEED)
    r0 = pickle.load(open(f"{DUMP}/round_00.pkl", "rb"))
    dfa0, dt0 = r0["dfa"], r0["dt"]
    fam0 = [bytes(v) for v in dt0.base_family]
    at0, rt0 = r0["accept_thresh"], r0["reject_thresh"]

    samp = SuperSampler(vocab, 36); rng = np.random.default_rng(SEED + 999)
    supers = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(NEVAL)]
    sk = [sig(w) for w in supers]
    st0 = [walk_path(dfa0, w)[-1] for w in supers]
    by = collections.defaultdict(lambda: {"SINK": [], "live": []})
    for w, g, s in zip(supers, sk, st0):
        by[s][g].append(w)
    S, groups = max(by.items(), key=lambda kv: min(len(kv[1]["SINK"]), len(kv[1]["live"])))
    print(f"target undersplit state s{S}: {len(groups['SINK'])} SINK + {len(groups['live'])} live", flush=True)

    # SINK members that the tree SIFTS to a pure-SINK leaf (i.e. tree separates them)
    sink_all = groups["SINK"]
    sink_leaves = sift_leaves(dt0.root, sink_all, fam0, at0, rt0, oracle)
    leafcnt = collections.Counter(l for l in sink_leaves if l is not None)
    pure_sink_leaf = leafcnt.most_common(1)[0][0]
    sink_sep = [w for w, l in zip(sink_all, sink_leaves) if l == pure_sink_leaf]
    live = groups["live"]
    print(f"round-0 tree sifts SINK to leaves {dict(leafcnt)}; using leaf {pure_sink_leaf} "
          f"({len(sink_sep)} SINK) as the tree-separated set vs {len(live)} live", flush=True)

    # walk paths (cheap)
    sink_paths = [walk_path(dfa0, w) for w in sink_sep]
    live_paths = [walk_path(dfa0, w) for w in live]
    maxd = max(max(len(p) for p in sink_paths), max(len(p) for p in live_paths))
    print(f"\nper-depth state occupancy (state: #SINK/#live), showing where they MERGE:")
    merge = None
    for d in range(maxd):
        sc = collections.Counter(p[d] for p in sink_paths if d < len(p))
        lc = collections.Counter(p[d] for p in live_paths if d < len(p))
        both = [(st, sc[st], lc[st]) for st in set(sc) & set(lc) if sc[st] >= 5 and lc[st] >= 5]
        if both and merge is None:
            merge = (d, max(both, key=lambda x: min(x[1], x[2]))[0])
        if d < 8 or both:
            top = sorted(set(sc) | set(lc), key=lambda st: -(sc[st] + lc[st]))[:5]
            desc = "  ".join(f"s{st}:{sc[st]}/{lc[st]}" for st in top)
            flag = "  <-- MERGE" if merge and merge[0] == d and both else ""
            print(f"  d{d:2d}: {desc}{flag}", flush=True)
    if merge is None:
        print("no clean shared merge state found in first pass"); return
    md, M = merge
    print(f"\nmerge: at depth {md}, state s{M} holds both SINK and live; both then flow to s{S}", flush=True)

    # at M: the edge each takes (next symbol) and where the successor prefix SIFTS
    def succ_info(paths):
        rows = []
        for w, p in zip(paths, range(len(paths))):
            pass
        return rows
    # collect (prefix up to M, next symbol) for SINK and live members passing through s M at depth md
    def at_M(members, paths):
        pre, nextc = [], []
        for w, p in zip(members, paths):
            if md < len(p) and p[md] == M and md < len(w):
                pre.append(list(w[:md]) + [w[md]]); nextc.append(w[md])
        return pre, nextc
    sink_pre, sink_c = at_M(sink_sep, sink_paths)
    live_pre, live_c = at_M(live, live_paths)
    print(f"  members through s{M} at d{md}: {len(sink_pre)} SINK, {len(live_pre)} live", flush=True)
    print(f"  next-symbol dist  SINK: {dict(collections.Counter(sink_c))}  live: {dict(collections.Counter(live_c))}", flush=True)
    # sift the successors prefix+[c]
    allpre = sink_pre + live_pre
    tags = ["SINK"] * len(sink_pre) + ["live"] * len(live_pre)
    succ_leaves = sift_leaves(dt0.root, allpre, fam0, at0, rt0, oracle)
    print(f"\n  where the SUCCESSORS (prefix+[c]) sift -- if heterogeneous, the edge is mis-resolved:")
    tab = collections.defaultdict(lambda: collections.Counter())
    for t, l in zip(tags, succ_leaves):
        tab[t][l if l is not None else -1] += 1
    for t in ("SINK", "live"):
        print(f"    {t}: {dict(tab[t])}", flush=True)
    print(f"\n  s{S} label: {'ACCEPT' if S in getattr(dfa0,'final_states',set()) else 'reject'}")


if __name__ == "__main__":
    main()
