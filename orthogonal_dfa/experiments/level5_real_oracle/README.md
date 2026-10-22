# Level5 counterexample search on the len40 real spliceai oracle

Run every script from the repo root with `PYTHONPATH=.`.

Full loop, with each round's family checkpointed (`family_r<n>.pkl`) and each round
ending logged (`diag.jsonl`):

    python -u orthogonal_dfa/experiments/level5_real_oracle/harvest_diag.py \
        --plateau --max-rounds 3 --seed 0 --out OUT

`rounds.json` has each round's DFA states and held-out phi (fit on one half of
2x6000 strings drawn with seeds 1111/2222, read on the other); `summary.json` the
best round.  `--plateau` sets `fnr_plateau_patience=3`.

Replay one round's counterexample search from a checkpoint (minutes instead of the
hours the family search takes), logging every probe and every hypothesis:

    python -u orthogonal_dfa/experiments/level5_real_oracle/replay.py OUT/family_r0.pkl \
        [--set max_probes=1500] [--set harvest_multiple=20] [--probe-seed N] [--tag T]

writes `OUT/replay_<tag>/{probes,snapshots}.jsonl`, `snap_<n>.pkl` and
`summary.json`.  `close_snap.py CKPT SNAP...` closes a snapshot's hypothesis and
scores it; `leaf_spread.py CKPT SNAP` counts where pool and held-out prefixes sift.

These scripts use the len40 oracle (`length=40`, `len_lo=35`, `len_hi=85`). Main's
`gate_composition_residual.py` is fitted around the default exon's query length (189)
and rejects that band, so check out #501's version of that file to run them.
