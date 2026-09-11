"""Does round 0's family actually have better labeling quality, or is its lower FPR/FNR an
artifact of abstaining more?

FPR/FNR are deflated by indecision (band strings count as neither accept nor reject), so a
more-abstaining family shows lower FPR AND FNR for free. The BAND-INDEPENDENT quality is how
well the family's accept-rate SEPARATES SINK from live -- measured by AUC (P(mean_live >
mean_SINK)), which doesn't depend on the band. For round 0 and round 1 (seed 2), at the root
(empty midfix = the SINK/live split), report AUC (real separation) alongside the band-based
FPR/FNR/indecision (which mixes quality with abstention).
"""
import pickle
import numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

SEED = 2; N = 1500; NSUF = 150
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


def auc(mean, is_live):
    """P(mean_live > mean_SINK) via rank statistic. 1.0 = perfect separation (live high)."""
    live = mean[is_live]; sink = mean[~is_live]
    if len(live) == 0 or len(sink) == 0:
        return float("nan")
    order = np.argsort(mean)
    ranks = np.empty(len(mean)); ranks[order] = np.arange(1, len(mean) + 1)
    r_live = ranks[is_live].sum()
    u = r_live - len(live) * (len(live) + 1) / 2
    return u / (len(live) * len(sink))


def main():
    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
    oracle = LiftedOracle(base, vocab, seed=SEED)
    samp = SuperSampler(vocab, 36); rng = np.random.default_rng(SEED + 999)
    supers = [list(samp.sample(rng, vocab.alphabet_size)) for _ in range(N)]
    is_live = np.array([sig(w) == "live" for w in supers])
    print(f"eval: {is_live.sum()} live, {(~is_live).sum()} SINK\n")
    print(f"{'round':6} {'band':16} {'AUC(sep)':9} {'FPR':6} {'FNR':6} {'indec':6} "
          f"{'meanSINK':9} {'meanLIVE':9}")
    for rd in (0, 1):
        rec = pickle.load(open(f"{D}/round_{rd:02d}.pkl", "rb"))
        fam = [bytes(v) for v in rec["dt"].base_family]
        vs = [fam[i] for i in np.random.default_rng(7).choice(len(fam), min(NSUF, len(fam)), replace=False)]
        at, rt = rec["accept_thresh"], rec["reject_thresh"]
        combos = [bytes(s) + v for s in supers for v in vs]
        m = np.asarray(oracle.membership_queries(combos), float).reshape(len(supers), len(vs)).mean(1)
        a = auc(m, is_live)
        fpr = np.mean(m[~is_live] >= at) * 100
        fnr = np.mean(m[is_live] < rt) * 100
        ind = np.mean((m >= rt) & (m < at)) * 100
        print(f"{rd:6} [{rt:.3f},{at:.3f})  {a:7.3f}   {fpr:5.1f}  {fnr:5.1f}  {ind:5.1f}  "
              f"{m[~is_live].mean():7.3f}   {m[is_live].mean():7.3f}", flush=True)


if __name__ == "__main__":
    main()
