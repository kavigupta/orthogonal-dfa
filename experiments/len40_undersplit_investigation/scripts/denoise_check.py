"""Apply the REAL denoise_accept_labels to the saved per-round DFAs and measure the
actual post-denoise phi (vs the opt-relabel ceiling we already have).  Reconstructs
only the pst attributes denoise touches: sampler, config.min_signal_strength,
decision_boundary, alphabet_size, rng, table.prefixes, oracle.
"""
import argparse, pickle, types
import numpy as np
import scipy.stats  # noqa: F401  (module code references scipy.stats)
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.l_star.lstar import denoise_accept_labels
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

STOPS = {(3, 0, 2), (3, 0, 0), (3, 2, 0)}


def frames_closed(seq):
    return tuple(
        any(tuple(seq[ph:][i:i + 3]) in STOPS for i in range(0, len(seq[ph:]) - 2, 3))
        for ph in range(3)
    )


def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def best_relabel_phi(call_states, states_list, ora, final_now):
    """Optimal-relabel ceiling: per produced state pick accept/reject to maximise
    agreement with the oracle, return phi of the resulting call vs oracle."""
    # call_states: array of produced-state id per eval string
    acc_label = {}
    for s in states_list:
        mask = call_states == s
        if mask.sum() == 0:
            acc_label[s] = s in final_now
            continue
        acc_label[s] = ora[mask].mean() >= 0.5
    call = np.array([acc_label[s] for s in call_states], float)
    return phi(call, ora)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--dump-dir", default="mr_dumps/seed2")
    args = ap.parse_args()

    vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4)
    base = gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)
    oracle = LiftedOracle(base, vocab, seed=args.seed)

    # fixed eval set identical to instrument_multiround (seed+999)
    samp = SuperSampler(vocab, 36)
    rng = np.random.default_rng(args.seed + 999)
    supers = [samp.sample(rng, vocab.alphabet_size) for _ in range(4000)]
    bases = vocab.compile_many(supers, [np.random.default_rng(i) for i in range(4000)])
    ora = np.asarray(base.membership_queries([bytes(b) for b in bases])).astype(float)
    fpat = [frames_closed(b) for b in bases]
    f01 = np.array([1.0 if not (p[0] and p[1]) else 0.0 for p in fpat])
    print(f"[seed {args.seed}] eval n=4000  phi(oracle,frame01)={phi(ora, f01):+.3f}", flush=True)

    def walk_state(dfa, w):
        s = dfa.initial_state
        for c in w:
            s = dfa.transitions[s][c]
        return s

    r = 0
    while True:
        try:
            rec = pickle.load(open(f"{args.dump_dir}/round_{r:02d}.pkl", "rb"))
        except FileNotFoundError:
            break
        dfa = rec["dfa"]

        # minimal pst stub for denoise
        cfg = types.SimpleNamespace(min_signal_strength=0.05)
        tbl = types.SimpleNamespace(prefixes=[list(p) for p in rec["prefixes"]])
        pst = types.SimpleNamespace(
            sampler=samp, config=cfg, decision_boundary=rec["boundary"],
            alphabet_size=vocab.alphabet_size, rng=np.random.default_rng(1234 + r),
            table=tbl, oracle=oracle,
        )

        # pre-denoise call + phi
        st = np.array([walk_state(dfa, w) for w in supers])
        pre = np.array([bool(dfa.accepts_input(w)) for w in supers], float)
        states_list = sorted(set(st))
        ceil_phi = best_relabel_phi(st, states_list, ora, set(dfa.final_states))

        # denoise
        dn = denoise_accept_labels(pst, dfa)
        post = np.array([bool(dn.accepts_input(w)) for w in supers], float)

        print(
            f"\nround {r}: {len(dfa.states)} st, {len(states_list)} reached | boundary {rec['boundary']:.3f} | "
            f"est {rec['true_acc']:.3f}\n"
            f"   PRE-denoise : phi(ora)={phi(pre, ora):+.3f}  phi(frame01)={phi(pre, f01):+.3f}  accept={pre.mean():.3f}\n"
            f"   POST-denoise: phi(ora)={phi(post, ora):+.3f}  phi(frame01)={phi(post, f01):+.3f}  accept={post.mean():.3f}\n"
            f"   opt-relabel ceiling: phi(ora)={ceil_phi:+.3f}   (n_flips accept-set: "
            f"{len(set(dn.final_states) ^ set(dfa.final_states))})",
            flush=True,
        )
        r += 1


if __name__ == "__main__":
    main()
