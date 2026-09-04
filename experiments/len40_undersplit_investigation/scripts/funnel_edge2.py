"""Show the incorrect edge resolution directly, anchored on the accept-cycle state s10.

For prefixes that walk to s10, take each out-symbol c and SIFT the successor prefix+[c].
If, for the SAME symbol c, SINK-lineage successors sift to a reject leaf (s1) while
live-lineage successors sift to a cycle state, then edge (s10,c) is resolved over a
HETEROGENEOUS member set -- and edge_resolver's arbitrary-first-member rule committed it
to one target (visible as dfa.transitions[s10][c]), trapping SINK in the accept-cycle.
"""
import pickle, collections
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

SEED = 2; NEVAL = 4000; NMEM = 60; MAXVISIT = 4
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
    acc = getattr(dfa0, "final_states", set())

    samp = SuperSampler(vocab, 36); rng = np.random.default_rng(SEED + 999)
    supers = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(NEVAL)]
    sk = [sig(w) for w in supers]
    ends = [walk_path(dfa0, w)[-1] for w in supers]
    cnt = collections.defaultdict(lambda: {"SINK": 0, "live": 0})
    for e, g in zip(ends, sk):
        cnt[e][g] += 1
    S = max(cnt, key=lambda q: min(cnt[q]["SINK"], cnt[q]["live"]))  # most-conflated state
    print(f"anchor state s{S} (label {'ACCEPT' if S in acc else 'reject'}): "
          f"{cnt[S]['SINK']} SINK + {cnt[S]['live']} live end here", flush=True)
    print(f"cycle labels: " + " ".join(f"s{q}={'A' if q in acc else 'r'}" for q in (1, 4, 10, 13)), flush=True)
    print(f"DFA edges out of s{S}: " + " ".join(
        f"{c}->s{dfa0.transitions[S][c]}" for c in sorted(dfa0.transitions[S])), flush=True)

    sink = [w for w, g, e in zip(supers, sk, ends) if g == "SINK" and e == S][:NMEM]
    live = [w for w, g, e in zip(supers, sk, ends) if g == "live" and e == S][:NMEM]

    # collect successors prefix+[c] at each visit to s{S}
    succ, tag, sym = [], [], []
    for grp, members in (("SINK", sink), ("live", live)):
        for w in members:
            path = walk_path(dfa0, w); visits = 0
            for i in range(len(w)):
                if path[i] == S:
                    succ.append(list(w[:i]) + [w[i]]); tag.append(grp); sym.append(w[i])
                    visits += 1
                    if visits >= MAXVISIT: break
    print(f"\nsifting {len(succ)} successors (prefix+[c]) leaving s{S}...", flush=True)
    leaves = sift_leaves(dt0.root, succ, fam0, at0, rt0, oracle)

    # per out-symbol c: where SINK-succ vs live-succ sift, and the DFA's committed target
    print(f"\n c | DFA edge | SINK successors sift-> | live successors sift->")
    bysym = collections.defaultdict(lambda: {"SINK": collections.Counter(), "live": collections.Counter()})
    for t, c, l in zip(tag, sym, leaves):
        bysym[c][t][l if l is not None else -1] += 1
    for c in sorted(bysym):
        tgt = dfa0.transitions[S].get(c, "?")
        sk_d = dict(bysym[c]["SINK"]); lv_d = dict(bysym[c]["live"])
        print(f" {c} | s{S}-{c}->s{tgt} | SINK {sk_d} | live {lv_d}", flush=True)
    print(f"\n(leaf 1 label: {'ACCEPT' if 1 in acc else 'reject'}; a heterogeneous row -- "
          f"SINK->1(reject) but live->cycle -- is the mis-resolved edge.)")


if __name__ == "__main__":
    main()
