# Level5 counterexample search on the len40 real spliceai oracle

Run every script from the repo root with `PYTHONPATH=.`.

Full loop, with each round's family checkpointed (`family_r<n>.pkl`) and each round
ending logged (`diag.jsonl`):

    python -u orthogonal_dfa/experiments/level5_real_oracle/harvest_diag.py \
        --plateau --max-rounds 3 --seed 0 --out OUT

`rounds.json` has each round's DFA states and held-out phi (fit on one half of
2x6000 strings drawn with seeds 1111/2222, read on the other); `summary.json` the
best round.  `--plateau` turns on every opt-in setting the PR's runs used: score
patience 500, FNR-scaled harvests, `fnr_plateau_patience=3`, 3 restarts, and ranking
rounds by held-out phi.

Replay one round's counterexample search from a checkpoint (minutes instead of the
hours the family search takes), logging every probe and every hypothesis:

    python -u orthogonal_dfa/experiments/level5_real_oracle/replay.py OUT/family_r0.pkl \
        [--set max_probes=1500] [--set harvest_multiple=20] [--probe-seed N] [--tag T]

writes `OUT/replay_<tag>/{probes,snapshots}.jsonl`, `snap_<n>.pkl` and
`summary.json`.  `close_snap.py CKPT SNAP...` closes a snapshot's hypothesis and
scores it; `leaf_spread.py CKPT SNAP` counts where pool and held-out prefixes sift.
`source_yield.py CKPT` reruns the round's search, then measures how often the next
round's boundary source finds a new string, per reader and per hypothesis variant.
`rescore.py OUT...` scores each saved round DFA of finished runs on 60,000 fresh strings.

These scripts use the len40 oracle (`length=40`, `len_lo=35`, `len_hi=85`) from
`gate_composition_residual_len40.py`, the version of `gate_composition_residual.py`
#501's runs were fitted with; main's version is fitted around the default exon's query
length (189) and rejects that band.

`SAWTOOTH_COUNTING.md` records what the current learner finds on the old oracle
(`gate_residual_oracle`): DFAs that count wildcards modulo 2 or 3.

`LENGTH_ORACLES.md` compares round 0 on that oracle with one thresholded at each compiled
length's own median (`length_oracles.py`, `length_oracles_label.py`,
`length_oracles_figure.py`): wildcard counters on the first, the frame-0 rule on the second.
