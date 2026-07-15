# Incremental transition-driven DFA discovery

## Motivation

The current discovery algorithm (`state_discovery.discover_states`) builds the
discrimination tree (DT) and then computes transitions as a *separate* step.
States are found by a BFS that, for each discovered family `vs`, prepends every
symbol (`[c]+vs`) and tests that prepended family against **every** state in the
tree. This has a structural flaw: a prepended distinguisher `[c]+vs` can split a
state anywhere in the tree, not just the one it "came from" (the cross-branch
split, e.g. the parity automaton). Any attempt to restrict *where* a prepend is
evaluated is therefore unsound, and the fixups (restricted prepend + a
transition-informed resplit fixpoint, on the `restrict-prepend-queries` branch)
are patches on a base that was never a proper discrimination tree.

This document proposes a different shape: **build the states and the transitions
together, incrementally, driven by transition resolution.** A distinguisher is
only ever created and applied at the exact state whose transition is ambiguous.
There is no global "test this family against all states" step, so the
cross-branch problem cannot arise by construction — this is a genuine
discrimination tree with lazy closedness (the Kearns–Vazirani / TTT shape),
adapted to the noisy, family-of-suffixes setting this codebase already uses.

## The core idea in one paragraph

We keep a discrimination tree whose leaves are states and whose internal nodes
are distinguisher families. Each state owns the set of prefixes that sift to it.
We do **not** have a transition function yet. We work a queue of unresolved
`(state, symbol)` pairs. Resolving `(s, c)` is trivial: push every prefix
`p ∈ s`, extended to `p·c`, through the **existing** decision tree and read off
which leaf it lands in — this is the ordinary DT `classify`/`sift`, no new
distinguishers involved. Early-stop as soon as the landing leaves settle the
question:

- **All of `s`'s prefixes land in the same leaf `t`** → record `s -c-> t` and
  drop `(s, c)` from the queue.
- **They land in different leaves** → `s` was secretly more than one state. Split
  `s` in the tree, enqueue the new children's `(state, symbol)` pairs, and
  re-enqueue any transition that pointed *at* `s` (its target just became
  ambiguous). Then move on.

When the queue empties, every `(state, symbol)` is resolved and we have the full
transition table plus the tree — i.e. the DFA.

The split distinguisher is `[c]+w`, where `w` is the distinguisher at which the
diverging `p·c` sift paths first part company (the LCA node of the differing
landing leaves). It is produced by, and applied only to, `s` — so it never
touches an unrelated state. That is the whole reason cross-branch splits cannot
happen here.

## Vocabulary and starting point

- **Distinguisher family** `v`: a set of suffixes with an accept/reject
  threshold band, exactly the existing `TriPredicate`. Applied to a prefix `p`,
  it returns accept / reject / undecided based on `mean_i oracle(p + v_i)`
  against the band. This is the unit of "a question we can ask about a prefix,"
  and it is inherently statistical (a family, not one suffix) to survive oracle
  noise.
- **State**: a leaf of the discrimination tree. Identified by its access path
  (the sequence of `(distinguisher, outcome)` pairs from the root). Owns the set
  of pool prefixes currently sifting to it.
- **The root distinguisher `v_eps`**: the initial suffix family we already
  sample (`cluster.sample_suffix_family`), i.e. "is this string accepted?" The
  tree starts as a single internal node `v_eps` with two leaves:
  - `s_acc` = prefixes classified accept by `v_eps`,
  - `s_rej` = prefixes classified reject by `v_eps`.
  The accept/reject labelling of every future state is inherited from which side
  of `v_eps` (transitively) it descends from — accept/reject is just the
  root-level outcome, so labels are free.
- **Transition table**: partial map `(state, symbol) -> state`, initially empty.
- **Worklist**: the set of `(state, symbol)` pairs whose transition is not yet
  resolved. Initially `{(s_acc, c), (s_rej, c) : c in alphabet}`.

## The transition-resolution step (the heart)

Resolving `(s, c)` is a single sweep over `s`'s prefixes — no recursion, no
per-node coherence walk:

1. For each prefix `p ∈ s`, sift `p·c` through the current decision tree with the
   ordinary `classify` (evaluate each node's distinguisher `w` on `p·c`, i.e.
   `w(p·c) = mean_i oracle(p·c + w_i) = mean_i oracle(p + [c]+w_i)`, and follow
   the accept/reject branch). Record the landing leaf `t(p)`.
2. Track the set of distinct landing leaves as we go, and **early-stop**:
   - the moment we can call the transition coherent (all decided prefixes so far
     agree on one leaf, with enough evidence), record `s -c-> t` and finish;
   - the moment two prefixes land in *different* leaves with enough evidence,
     stop sweeping and split (below).
3. Coherent → record `s -c-> t`, remove `(s, c)` from the queue. Incoherent →
   split `s` and requeue (below).

The only queries this makes are the `oracle(p + [c]+w_i)` needed to sift `p·c` —
and only for as many prefixes `p` as it takes to settle the question, which is
often a handful. That is where the query saving comes from, for free.

### Noise wrinkle

A landing leaf `t(p)` is itself a noisy classification: each node on the sift
path applies a family band, and `p·c` can be *undecided* at some node (its
`w(p·c)` falls in the band). Two refinements handle this:

- Prefixes that go undecided partway down don't get a landing leaf; they abstain
  from this round. If too few prefixes decide to call coherence, **enrich** `s`
  (sample more pool prefixes that sift to `s`) and retry.
- "They land in different leaves" must mean *significantly* different, not one
  noise-flipped prefix — reuse the existing binomial split test
  (`overlapping_states`, specialised to `s`) over the decided prefixes to decide
  coherent-vs-split, rather than reacting to a single disagreement.

The mask cache (`PrefixSuffixTracker.corresponding_masks`) records every
`oracle(p + [c]+w_i)` sift query, so re-resolving an edge after a later split
reuses them.

## Splitting

When resolving `(s, c)` finds `s`'s prefixes landing in different leaves, let `w`
be the distinguisher of the LCA node of those differing leaves — the node where
the `p·c` sift paths first diverge. Then:

- Create a new internal node with distinguisher `d = [c]+w`, replacing the leaf
  `s`. (If the landing leaves span more than two subtrees, this single binary
  split by the highest divergence is enough — the children are requeued and
  refined further by the same mechanism, so we never build a multi-way split by
  hand.)
- Partition `s`'s prefixes by `d`: the reject side becomes `s0`, the accept side
  `s1`. (Undecided prefixes are held out until enriched — see noise section.)
- Both children inherit `s`'s accept/reject label (splitting refines a state, it
  does not change which side of `v_eps` it is on).
- Worklist bookkeeping:
  - Remove any resolved transitions *out of* `s`; add `(s0, c')` and `(s1, c')`
    for every symbol `c'`.
  - **Closedness restoration**: any recorded transition `s' -c'-> s` is now
    ambiguous (does `s'·c'` land in `s0` or `s1`?). Re-open it: add `(s', c')`
    back to the worklist. (Track incoming edges, or conservatively re-open all
    transitions whose target was `s`.)

Splitting is monotone (states only ever increase) and bounded by the number of
Nerode classes reachable with the current prefix pool, so the inner loop
terminates.

## Outer loop (counterexamples and noise robustness)

Resolving the worklist to empty yields a closed hypothesis DFA: every state has
a coherent transition on every symbol, and accept/reject labels come from the
root. Then, exactly as today:

1. Estimate accuracy against fresh oracle samples (`estimate_agreement_rate`).
2. If good enough, stop.
3. Otherwise, get counterexamples (`generate_counterexamples`) — strings where
   the hypothesis and the oracle disagree. Process each TTT-style: locate the
   prefix along the counterexample where the hypothesis' state and the true
   behaviour diverge, extract the distinguishing suffix, and use it to split the
   responsible state (re-opening affected transitions). Counterexamples also add
   prefixes to the pool, which is what lets rare states accumulate enough
   evidence.

Why we still need the outer loop even though closure "should" give the right
DFA: (a) the oracle is noisy and the prefix pool is finite, so a coherence test
can be wrong and needs the accuracy check as a backstop; (b) rare states may not
be reachable from the initial pool and need counterexample-supplied prefixes
before they can be split out.

## Noise / statistics details

- Every "does `d` split `s`?" is the existing binomial split test over `s`'s
  decided prefixes (reuse `overlapping_states`, restricted to one state), with
  the family size / evidence margin / `decision_rule_fpr` / `split_pval`
  machinery already tuned in `SearchConfig`.
- A prefix whose `[c]+w` value lands in the undecided band contributes to
  neither side; if a state has too few decided prefixes to call coherence, we
  **enrich**: sample more pool prefixes that sift to `s` (the existing
  `enrich_underrepresented_leaves` / prefix-sampling machinery) and retry. This
  replaces "sample more prefixes globally" with "sample more prefixes *for the
  state we're currently stuck on*."
- `v_eps` remains a full family; prepending forms `[c]+v_eps`, `[c'][c]+v_eps`,
  … The prepend-closure of `v_eps` is exactly the set of "read some prefix, then
  check acceptance" suffixes, which separates all Nerode classes — so the
  distinguishers this process generates are sufficient in principle.

## Why this is better than the current algorithm

- **Sound by construction**: a distinguisher `[c]+w` is created while resolving
  `s`'s `c`-transition and is applied only to `s`. There is no global family-vs-
  all-states test, so cross-branch splits (the thing that made the restricted
  prepend unsound and forced the resplit fixpoint) cannot occur.
- **Transitions and states co-evolve**: no separate `compute_transition_matrix`
  pass that re-derives `[c]+predicate` over everything; the transition *is* the
  sift, and the sift *is* what discovers splits.
- **Queries are naturally targeted and early-stopped**: we query `[c]+w` only on
  the state we are resolving, only until the coherence question is decided. The
  saving the `restrict-prepend-queries` branch engineered explicitly is inherent
  here.
- **It is a real discrimination tree**, so the mature literature (TTT
  counterexample decomposition, discriminator finalization) applies directly if
  we want to go further.

## What we can reuse from the existing code

- `structures.TriPredicate`, `DecisionTree*` node types, and the mask cache in
  `PrefixSuffixTracker` (`corresponding_masks`, `record_suffix`,
  `compute_decision_from_strings`) — the storage layer is unchanged.
- `cluster.sample_suffix_family` for `v_eps`.
- `state_discovery.overlapping_states`' binomial split test — specialised to a
  single state — as the coherence test.
- `estimate_agreement_rate`, `generate_counterexamples`, `denoise_accept_labels`,
  and the whole `counterexample_driven_synthesis` outer loop, essentially
  unchanged.

The genuinely new code is a `TransitionResolver` (worklist of `(state, symbol)`,
per-edge sift with early stop, split + closedness restoration) that replaces
`discover_states` + `compute_transition_matrix` with a single interleaved pass.

## Open questions / risks to settle before implementing

1. **Closedness restoration cost.** Splitting a heavily-targeted state re-opens
   every edge that pointed at it. Track incoming edges explicitly so we only
   requeue the affected `(s', c')` pairs rather than rescanning all transitions.
2. **Queue ordering.** BFS from `s_acc`/`s_rej` (reachability order) is the
   natural choice, so the reachable part of the DFA is built first and rare
   states surface via counterexamples.
3. **Interaction with counterexample-supplied prefixes.** New prefixes must be
   sifted into the existing tree (assigned to a leaf) before they can inform a
   transition's coherence; this is the same `sift` used everywhere else.
4. **Early-stop threshold vs noise.** How much agreement is "enough" to call a
   transition coherent, and how much divergence is "enough" to split — reuse the
   existing family-size / `split_pval` / evidence-margin settings, but the
   sequential (add-a-prefix-at-a-time) framing may want its own stopping rule.
5. **Validation.** Compare against the `restrict-prepend-queries` branch on the
   same benchmarks (`modulo`, `subseq`, `two_subseq`, `poor_case`, generated
   benchmarks) by **accuracy across seeds** and **oracle-query count** — never by
   DFA identity.
