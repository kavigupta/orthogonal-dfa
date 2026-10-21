Family-read slack plots (A-F) and the CSVs they are drawn from.

Regenerate (about 3 minutes):

    python scripts/family_read_slack.py all --targets bench:0 bench:1 bench:2 bench:3 bench:4 modulo subseq overlap parity_toggle --learn-seed 0 --read-seed 1 --signal 0.3 --strings-per-state 1000 --workers 2 --out figures/family-read-slack

`collect` writes the CSVs; `plot` redraws the figures from them.
