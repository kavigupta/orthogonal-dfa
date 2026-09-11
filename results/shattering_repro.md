# Reproducing E-L* suffix shattering with a simple grammar

No SpliceAI, no GPU, no learned weights.  Over 300 random length-48 prefixes we count the **distinct observation-table rows** (= states E-L* would create by exact deterministic distinguishing) as the random suffix set grows.  A regular target saturates at its state count; a non-regular one climbs toward one state per prefix -- that runaway is the shattering, the 397k-suffix SpliceAI explosion in miniature.

| oracle (grammar) | 10 suf | 100 suf | 1000 suf | 4000 suf | final frac |
|---|---:|---:|---:|---:|---:|
| randomDFA (6 states, regular) | 1 | 1 | 5 | 5 | 0.02 |
| all-frames-closed (regular) | 9 | 20 | 20 | 20 | 0.07 |
| half count-compare (statistic) | 8 | 16 | 19 | 19 | 0.06 |
| half lex-compare (whole prefix) | 10 | 81 | 231 | 273 | 0.91 |

**Reading.** Both comparison grammars ask "is the first half bigger than the second half?" -- the only difference is *bigger how*:

- **count-compare** (more A's) compares a *statistic*, so E-L* only needs the running A-count -> it collapses to ~length/2 states and saturates.
- **lex-compare** compares the *whole* prefix, so two prefixes are E-L*-equivalent only if identical -> every prefix becomes its own state, and the table shatters (final frac -> 1.0 as suffixes grow; still climbing where the others are flat).

The random DFA (~5) and all-frames-closed (~20) also saturate -- so shattering is not "many states", it is **unbounded, non-regular** state growth.  And all-frames-closed is regular and learnable in isolation; it is only lost when embedded in a shattering function (as in SpliceAI's residual), which explodes E-L* before it can isolate the frame automaton.
