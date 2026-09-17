import Mathlib.Tactic
import Mathlib.Probability.ProductMeasure
import Mathlib.Probability.Independence.InfinitePi

/-!
# The model: the oracle, the algorithm, and the theorem

**This file is the whole statement.**  It contains the oracle the algorithm queries, every
definition the algorithm is built from, and `ClusteringCorrect` — the proposition
`OrthoDFA.clustering_correct` proves.  Nothing that the theorem says is stated anywhere
else, so a reviewer checking that the Lean development says what it claims to say need read
only this file.  Everything in the other files is a lemma used to get there.

Two things are worth checking here, and they are different kinds of check:

* **Against the code.**  Do `clusterAround`, `screened`, `ret` and the rest describe what
  `identify_cluster_around`, `_screen_cohort` and `judge_family` actually do?  Each
  definition's doc comment names the Python it models and the place where the two could
  come apart.

* **Against the claim.**  Does `ClusteringCorrect` say the algorithm is PAC-correct, with
  no hypothesis that assumes the conclusion and no parameter tuned to make the proof go
  through?  Its hypotheses are listed in one place, in the order they are consumed.

The budget is *computed*, not quantified over: `schedule` is the concrete ladder
`prefCount, prefCount/2, …` derived from the statement's own `εcov`, `δ`, `α` and `pAP`,
and `stoppable` filters it to the rungs that carry their share of the error budget.  So
there is no hidden "for a large enough sample size" — the sizes are terms in this file.
`sample_suffix_family` stops at a data-dependent time, so the guarantee has to hold at
*whatever* rung the loop picks; that is what quantifying over `stoppable` says, and `ret`
is what "whatever it returns" means — the states passing both gates.  Nothing is claimed
at a state the loop rejects.

The conclusion is about the family's **cut** (`cutCorrect`), not about each of its members
being accept-preserving.  Per-member preservation is the premise the algorithm works from
(`pAP`) and what its screening reaches for, but no finite test certifies it of a sampled
family, and the vote survives a drifting member.  What is certified — by the gate, by
#215's test, and by what E-L\* actually consumes downstream — is that the family
classifies prefixes correctly.

**Known modelling gap (flagged, not hidden).**  The draws here are i.i.d. from each
distribution and deduplicated downstream (`poolAt`, `prefixesAt`), whereas `_draw_cohort`
and `sample_more_prefixes` *redraw* on a duplicate — they sample without replacement.
Deduplicating i.i.d. draws yields a pool at most as large, so this is the conservative
model; but the collision mass enters as `m²ρ`, so the statement caps `ρ` at
`collisionCap` — `δ` over the populations cubed, the prefix and pool counts squared, and
the ladder's length.  Removing that cap needs either the without-replacement concentration
(Hoeffding 1963) or a non-atomic prefix distribution.  That is a gap in the *proof*,
recorded rather than papered over by weakening the claim.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-! ## The oracle

What the algorithm is allowed to ask, and the only randomness that is not a draw. -/

/-- The persistent signal oracle: random classification noise on query strings.
Every field is a function of a **single** query string `w : S` — the oracle answers
membership on one string at a time.  `label w = 1[w ∈ L]` is the noiseless
membership bit and `noise w` the persistent RCN bit; the membership query is
`label ⊕ noise`.  Concatenation and the notion of one suffix flipping a prefix are
*not* in the oracle: strings are Mathlib's theory (`[Mul S]` concatenation,
`[IsRightCancelMul S]` right-cancellation), and `flip` is *derived* below. -/
structure Oracle {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (S : Type*) [MeasurableSpace S] where
  /-- The noiseless membership bit `ℓ(w) = 1[w ∈ L]`. -/
  label : S → ℝ
  label_bit : ∀ w, label w = 0 ∨ label w = 1
  label_meas : Measurable label
  /-- The random classification noise, one persistent bit per query string. -/
  noise : S → Ω → ℝ
  /-- The noise level. -/
  η : ℝ
  hη : η ≤ 1 / 2
  /-- Noise is independent and identically distributed as `Bernoulli(η)`.  Measurability
  is *joint* in the query string and the sample, which is what lets the oracle be
  composed with a **randomly drawn** query string; the per-string version is derived. -/
  noise_meas : Measurable (fun z : S × Ω => noise z.1 z.2)
  noise_indep : iIndepFun noise μ
  noise_bit : ∀ w, ∀ᵐ ω ∂μ, noise w ω = 0 ∨ noise w ω = 1
  noise_mean : ∀ w, μ[noise w] = η

namespace Oracle
variable {S : Type*} [MeasurableSpace S] [Mul S] (O : Oracle μ S)

/-- Whether suffix `v` flips prefix `p`'s acceptance: the XOR `ℓ(p·v) ⊕ ℓ(p)` of the
two membership bits (`a ⊕ b = a + b − 2ab`).  **Derived** from the single-string
label, so it is `0` exactly when `p·v` and `p` agree — accept-preservation. -/
def flip (v p : S) : ℝ :=
  O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p

/-- The disagreement read `flip ⊕ noise = flip + (1−2·flip)·noise` of suffix `v` on
the `i`-th prefix `pref i` (both strings), where the noise is that of the
concatenated query string `pref i · v` (`·` = the string `Mul`). -/
noncomputable def read {ι : Type*} (pref : ι → S) (v : S) (i : ι) : Ω → ℝ :=
  fun ω => O.flip v (pref i) + (1 - 2 * O.flip v (pref i)) * O.noise (pref i * v) ω

end Oracle

open scoped Classical in
/-- The greedy's output: a least-total-loss `k`-subset of `cands` (argmin over
`k`-subsets of `∑ ℓ`).  This *defines* the greedy, so its subset/cardinality/
pairwise-least-loss properties become lemmas rather than assumptions. -/
noncomputable def leastLossSubset {S : Type*} (ℓ : S → ℝ) (cands : Finset S) (k : ℕ) :
    Finset S :=
  if h : (cands.powersetCard k).Nonempty then
    (Finset.exists_min_image (cands.powersetCard k) (fun T => ∑ x ∈ T, ℓ x) h).choose
  else ∅

variable {S : Type*} [MeasurableSpace S] [Monoid S] [IsCancelMul S] [MeasurableMul S]
  [Countable S] [MeasurableSingletonClass S] [DecidableEq S]

/-- The membership query the oracle actually answers: `MQ w = ℓ(w) ⊕ noise(w)`.

At `w = p` this is the **seed column**, the oracle's answer at `p · ε = p`, which is what the
gate reads its verdict off.  Across *distinct* prefixes these are independent — one noise
bit per query string — which is why `prefixesOf` and `certOf` being `Finset`s is what makes
the gate's binomial null honest. -/
noncomputable def mq (O : Oracle μ S) (w : S) (ω : Ω) : ℝ :=
  O.label w + (1 - 2 * O.label w) * O.noise w ω

variable {J : Type*} [Fintype J]

/-! ## The run space

One sample is the noise together with the three draw streams; `runLaw` is its law, so the
independence the proof uses is a fact about a concrete measure rather than a hypothesis. -/

/-- One run of the algorithm: the oracle's persistent noise, the suffix draws, the prefix
draws of each population, and the **certification** draws the gate judges on.

The certification stream is separate because the gate must not be read on the prefixes the
family was selected from — see `certOf`. -/
abbrev Run (Ω S J : Type*) := Ω × ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S))

/-- The law of a run: the three components jointly independent, each stream i.i.d. -/
noncomputable def runLaw (μ : Measure Ω) (D : J → Measure S) (Dsf : Measure S) :
    Measure (Run Ω S J) :=
  μ.prod ((((Measure.infinitePi fun _ : ℕ => Dsf).prod
      (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j))).prod
    (Measure.infinitePi fun z : J × ℕ => D z.1))

instance (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] : IsProbabilityMeasure (runLaw μ D Dsf) := by
  unfold runLaw; infer_instance

/-- The run's persistent noise. -/
def nz (x : Run Ω S J) : Ω := x.1

/-- The `i`-th suffix drawn. -/
def sfx (i : ℕ) (x : Run Ω S J) : S := x.2.1.1 i

/-- The `i`-th prefix drawn from population `j`. -/
def prf (j : J) (i : ℕ) (x : Run Ω S J) : S := x.2.1.2 j i

/-- The `i`-th **certification** prefix from population `j`: drawn only to read the split
on, never added to the table, and independent of everything the family was chosen from
(`certification_sample`). -/
def cert (j : J) (i : ℕ) (x : Run Ω S J) : S := x.2.2 (j, i)

/-- **The loop's state**: the two budgets, the family size, the cluster's centre cutoff,
and the reject/accept cutoffs.  All of it integer data.

There is no history and no real-valued boundary.  The boundary existed to be threaded
along the rounds, and it entered every event only through the count it cut at — so the cut
itself is the state.  The guarantee is quantified over *every* state, so whatever the loop
computes for its boundary, size and margin is covered, and unlike a history this index is
countable. -/
@[ext]
structure Budget where
  /-- Suffix budget: how many suffixes have been drawn. -/
  M : ℕ
  /-- Prefix budget: how many prefixes each population has drawn. -/
  m : ℕ
  /-- Family size. -/
  k : ℕ
  /-- The cluster centre's cutoff, as a ratio `cn/cd`: a prefix is on the centre's accept
  side when `cn · #F < cd · (how many of `F` answer accept)`.

  A ratio rather than a count because `identify_cluster_around` centres on
  `masks[cluster].mean(0) > decision_boundary` — a *fraction* of the current cluster, which
  has one member at the first iteration and `k` afterwards.  A fixed count cannot serve
  both. -/
  cn : ℕ
  cd : ℕ
  /-- Reject at or below this count. -/
  lo : ℕ
  /-- Accept above this count. -/
  hi : ℕ
  /-- The screen's cutoff: a candidate disagreeing with the seed's column on more than this
  many representative prefixes never becomes a clustering candidate.

  `_screen_cohort` tests each drawn suffix against `same_family_rate = 2η(1−η)` and only
  survivors are promoted to fully observed — and `identify_cluster_around` clusters over
  `fully_observed()`.  So the screen, not the Lloyd ranking, is what bounds the family's
  flip mass; the ranking chooses among candidates that already passed.  (Issue #288.) -/
  sc : ℕ
  /-- Its denominator: the screen is a rate, since `_screen_cohort` tests against
  `same_family_rate` on however many representative prefixes the table holds. -/
  scd : ℕ
  /-- The gate's minimum side size: a side of the cut holding fewer prefixes than this is
  not tested.

  A side below it cannot pass its test whatever the family does — the binomial tail at a
  handful of prefixes is above any `α` — so demanding it is what would make a population
  lying entirely on one side of the language untestable.  The prefixes on a skipped side go
  uncertified, which is why `gmin` has to stay small against the prefix count. -/
  gmin : ℕ

deriving instance DecidableEq for Budget

instance : Countable Budget :=
  Function.Injective.countable
    (f := fun b => (b.M, b.m, b.k, b.cn, b.cd, b.lo, b.hi, b.sc, b.scd, b.gmin))
    (by rintro ⟨⟩ ⟨⟩ h; simp_all)

/-! ### The budget, solved rather than searched for

Every field is read off the condition it has to meet.  A condition is always a tail
`exp (-a) ≤ ε`, which asks only that `a` clear `log (1/ε)` — so a field is a logarithm of the
error budget, not a power of a search parameter.  Where a field's own condition mentions the
share it will be given (which depends on that field), it mentions it only through *its*
logarithm, and `log x ≤ 2√x` closes that loop in one step. -/

/-- The oracle's signal. -/
noncomputable def sig (O : Oracle μ S) : ℝ := 1 / 2 - O.η

/-- The budget a round charges wrongly-cut prefixes at: what the gate's margin can absorb. -/
noncomputable def cutBudget (εcov : ℝ) : ℝ := εcov / 64

/-- **The family size.**  A clean family's vote fails at `exp (-κ·s²/2)`, and the round pays
that at the cut budget, so `κ` is the logarithm of the two together. -/
noncomputable def famCount (O : Oracle μ S) (populations : Finset J) (εcov δ : ℝ) : ℕ :=
  ⌈2 * Real.log (32 * (populations.card : ℝ) / (cutBudget εcov * δ)) / sig O ^ 2⌉₊
    + ⌈1 / sig O⌉₊ + 1

/-- **The flip budget** each family member is allowed: the cut budget spread over the family
and the populations. -/
noncomputable def flipBudget (O : Oracle μ S) (populations : Finset J) (εcov δ : ℝ) : ℝ :=
  cutBudget εcov / (8 * (populations.card : ℝ) * (famCount O populations εcov δ : ℝ))

/-- **The screen's margin**, at the flip budget. -/
noncomputable def screenMargin (O : Oracle μ S) (populations : Finset J) (εcov δ : ℝ) : ℝ :=
  flipBudget O populations εcov δ * sig O ^ 2

/-- **The pool**: enough suffixes that a family of `k` fits inside the findable fraction. -/
noncomputable def poolCount (O : Oracle μ S) (populations : Finset J) (εcov δ pAP : ℝ) : ℕ :=
  ⌈2 * ((famCount O populations εcov δ : ℝ) + 1) / pAP⌉₊
    + ⌈Real.log (32 * (populations.card : ℝ) / δ) / (2 * (pAP / 2) ^ 2)⌉₊

/-- The slower of the two rates the state's own share is measured at. -/
noncomputable def shareRate (O : Oracle μ S) (εcov : ℝ) : ℝ :=
  εcov * (sig O * εcov / 16) ^ 2 / 16

/-- The prefix count the *share* asks for.  The ladder's length is logarithmic in the prefix
count, so the share's condition refers to `log` of the count itself; `(√(A/r) + 2/r)²` is
the closed form that clears `A + log (m + 2)` at rate `r`, because the logarithm is under
the square root. -/
noncomputable def shareCount (O : Oracle μ S) (populations : Finset J) (εcov δ : ℝ) : ℕ :=
  ⌈(Real.sqrt ((Real.log (16 * (populations.card : ℝ) / δ) + 4) / shareRate O εcov)
    + 2 / shareRate O εcov) ^ 2⌉₊

/-- **The prefix count**: the counts the round's tails ask for, summed so each is met. -/
noncomputable def prefCount (O : Oracle μ S) (populations : Finset J)
    (εcov δ α pAP : ℝ) : ℕ :=
  ⌈Real.log (32 * (populations.card : ℝ)
      * ((poolCount O populations εcov δ pAP : ℝ) + 1) / δ)
      / (2 * screenMargin O populations εcov δ ^ 2)⌉₊
    + ⌈Real.log (32 * (populations.card : ℝ) / δ) / (2 * (cutBudget εcov / 4) ^ 2)⌉₊
    + ⌈64 * Real.log (1 / α) / (εcov * (sig O * εcov / 4) ^ 2)⌉₊
    + ⌈64 * Real.log (64 * (populations.card : ℝ) / δ)
        / (εcov * (sig O * εcov / 4) ^ 2)⌉₊
    + ⌈Real.log (32 * (populations.card : ℝ)
        * ((poolCount O populations εcov δ pAP : ℝ) + 1) / δ)
        / (2 * ((populations.card : ℝ) * flipBudget O populations εcov δ) ^ 2)⌉₊
    + shareCount O populations εcov δ
    + ⌈64 / εcov⌉₊
    + 1

open scoped Classical in
/-- **The state at a given prefix count.**  Every other field is read off the condition it
has to meet: the family size is a logarithm of the error budget, the pool follows from the
family, the screen's rate from the margins the family fixes, and the gate's floor from the
prefix count itself. -/
noncomputable def solvedBudgetAt (O : Oracle μ S) (populations : Finset J)
    (εcov δ pAP : ℝ) (mi : ℕ) : Budget where
  M := poolCount O populations εcov δ pAP
  m := mi
  k := famCount O populations εcov δ + 1
  cn := 1
  cd := 2
  lo := ⌈(famCount O populations εcov δ : ℝ) * (1 / 2 - sig O / 2)⌉₊ - 1
  hi := ⌈(famCount O populations εcov δ : ℝ) * (1 / 2 - sig O / 2)⌉₊ + 1
  sc := ⌈((⌈1 / (2 * screenMargin O populations εcov δ)⌉₊ + 1 : ℕ) : ℝ)
    * (2 * O.η * (1 - O.η) + screenMargin O populations εcov δ)⌉₊
  scd := ⌈1 / (2 * screenMargin O populations εcov δ)⌉₊ + 1
  gmin := ⌊εcov * (mi : ℝ) / 32⌋₊

/-- **How many times the loop runs the gate**: the prefix count halves down to one, so the
ladder is logarithmic in it.  This is the number of states the error budget is divided
among — there is no other state the loop can return at. -/
noncomputable def ladderLen (O : Oracle μ S) (populations : Finset J)
    (εcov δ α pAP : ℝ) : ℕ :=
  Nat.log 2 (prefCount O populations εcov δ α pAP) + 1

open scoped Classical in
/-- **The states the loop runs the gate at**: the ladder `m, m/2, m/4, …`. -/
noncomputable def schedule (O : Oracle μ S) (populations : Finset J)
    (εcov δ α pAP : ℝ) : Finset Budget :=
  (Finset.range (ladderLen O populations εcov δ α pAP)).image
    (fun i => solvedBudgetAt O populations εcov δ pAP
      (prefCount O populations εcov δ α pAP / 2 ^ i))

/-! ## The algorithm

`poolAt` through `cutCorrect` are `sample_suffix_family` one step at a time: draw, screen,
cluster, and read the cut off the family's vote. -/

/-- The candidate pool at a suffix budget: the first `M` suffixes drawn, **with the seed**.

`identify_cluster_around` requires the seed to be among the candidates and asserts it —
`pst.table.column(v)` promotes it to fully observed at the top of every round, precisely so
that `candidate = pst.table.fully_observed()` contains it.  Without it `leastLossSubset`
could never return a set containing `ε`, `lloydStep` would never fire, and every family
would be the degenerate `{ε}`. -/
noncomputable def poolAt (M : ℕ) (x : Run Ω S J) : Finset S :=
  insert 1 ((Finset.range M).image (fun i => sfx i x))

/-- Population `j`'s own representative prefixes at a budget: its first `m` draws, as a
set — the table interns prefixes, so a repeated draw is one column, not two. -/
noncomputable def prefixesOf (j : J) (m : ℕ) (x : Run Ω S J) : Finset S :=
  (Finset.range m).image (fun i => prf j i x)

/-- Population `j`'s **certification** prefixes at a budget: `certification_sample`'s
draws, which the family was never selected from.

The gate must be judged here and not on `prefixesOf`.  The cluster is chosen to minimise
Hamming loss against a centre anchored at `ε`, i.e. to agree with `ε`'s *noisy* column on
the representative prefixes — the very agreement the gate then measures.  With a large
enough pool the cluster can match that column exactly, at which point the accept side is
`{p | mq p = 1}`, the hit rate is `1`, and the gate admits a family whose cut is the noise.
On prefixes the selection never saw there is no such coupling.  (Issue #284.) -/
noncomputable def certOf (j : J) (m : ℕ) (x : Run Ω S J) : Finset S :=
  (Finset.range m).image (fun i => cert j i x)

open scoped Classical in
/-- The representative prefixes at a prefix budget: every population's, pooled. -/
noncomputable def prefixesAt (populations : Finset J) (m : ℕ)
    (x : Run Ω S J) : Finset S :=
  populations.biUnion (fun j => prefixesOf j m x)

open scoped Classical in
/-- **The family's vote as the count it is**: how many of the family answer accept at `p`.

Every comparison in the algorithm is against a threshold on this count — the real-valued
mean `vote` carries no more information, which is `vote_mem_grid` in `OrthoDFA.Adaptive` —
so the configuration is integer data and the state stays countable. -/
noncomputable def voteCount (O : Oracle μ S) (F : Finset S) (p : S) (ω : Ω) : ℕ :=
  (F.filter (fun v => mq O (p * v) ω = 1)).card

open scoped Classical in
/-- `identify_cluster_around`'s loss: the Hamming distance from a candidate's mask row to
the cluster's **own** thresholded mean (`masks[cluster].mean(0) > decision_boundary`). -/
noncomputable def hammingLoss (O : Oracle μ S) (F : Finset S) (cn cd : ℕ) (P : Finset S)
    (ω : Ω) (v : S) : ℝ :=
  ((P.filter (fun p =>
    ¬ ((mq O (p * v) ω = 1) ↔ cn * F.card < cd * voteCount O F p ω))).card : ℝ)

open scoped Classical in
/-- The loss the greedy minimises, zeroed off the candidate pool.

`leastLossSubset` is an argmin picked by `Classical.choose`, so it depends on the loss as a
*function*, not merely on its values over `cands`.  Zeroing it elsewhere makes two draws
that agree on the reads give literally the same loss, which is what `clusterAround_congr`
needs. -/
noncomputable def clusterLoss (O : Oracle μ S) (F : Finset S) (cn cd : ℕ) (P cands : Finset S)
    (ω : Ω) (v : S) : ℝ :=
  if v ∈ cands then hammingLoss O F cn cd P ω v else 0

open scoped Classical in
/-- One Lloyd step: recentre on the current cluster, then retake the `k` least-loss
candidates — but only while the seed is among them.  `identify_cluster_around` breaks out
(`if seed_local not in nearest`) rather than let the centre drift off `ε`, keeping the
cluster it had.

The cohort is the seed together with the `k−1` next best, and it is taken only when that
really is a least-loss subset.  `np.argsort` is stable and the seed is the table's first
column, so the seed wins ties for the `k`-th place — and at the first step, where the centre
is the seed's own column, the seed's loss is `0` and the cohort is always taken.  Modelling
the ranking as an arbitrary argmin instead would let a tie throw the seed out and stall the
clustering at `{ε}`, which is not what the code does. -/
noncomputable def lloydStep (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (F : Finset S) : Finset S :=
  if ∀ w ∈ cands, w ∉ insert (1 : S)
        (leastLossSubset (clusterLoss O F cn cd P cands ω) (cands.erase 1) (k - 1)) →
      ∀ v ∈ insert (1 : S)
        (leastLossSubset (clusterLoss O F cn cd P cands ω) (cands.erase 1) (k - 1)),
      clusterLoss O F cn cd P cands ω v ≤ clusterLoss O F cn cd P cands ω w
  then insert (1 : S)
    (leastLossSubset (clusterLoss O F cn cd P cands ω) (cands.erase 1) (k - 1)) else F

/-- `identify_cluster_around` iterated to its fixed point.  The total loss is a natural
number bounded by `k·#P` that strictly decreases at each improving step, so `k·#P + 1`
iterations from the seed `ε` already sit at the fixed point — the bound is derived, not a
knob. -/
noncomputable def clusterAround (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω)
    (k : ℕ) : Finset S :=
  (lloydStep O cn cd P cands ω k)^[k * P.card + 1] {(1 : S)}

open scoped Classical in
/-- **The screen's statistic**: how many representative prefixes the candidate's column
disagrees with the seed's on.

This is `_screen_cohort`'s disagreement count.  Its mean separates an accept-preserving
candidate from one carrying flip mass `φ` by `φ(1−2η)²` — the same statistic the first Lloyd
step ranks by, which is why `seedLoss` serves both. -/
noncomputable def screenCount (O : Oracle μ S) (P : Finset S) (v : S) (ω : Ω) : ℕ :=
  (P.filter (fun p => ¬ ((mq O (p * v) ω = 1) ↔ (mq O p ω = 1)))).card

/-- **The candidates the clustering actually sees.**  A suffix the screen rejects never
becomes a fully observed column, so `identify_cluster_around` never ranks it.  This is what
bounds the family's flip mass — the Lloyd ranking only chooses among what is left.  (Issue
#288.) -/
noncomputable def screened (O : Oracle μ S) (sc scd : ℕ) (P cands : Finset S) (ω : Ω) :
    Finset S :=
  cands.filter (fun v => scd * screenCount O P v ω ≤ sc * P.card)

open scoped Classical in
/-- The screen at one budget state. -/
noncomputable def screenedAt (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (x : Run Ω S J) : Finset S :=
  screened O B.sc B.scd (prefixesAt populations B.m x) (poolAt B.M x) (nz x)

/-- The cluster at one budget state.

Nothing here is derived from a real-valued boundary or margin: the centre's cutoff, the
family size and the two gate cutoffs are all part of the state, and the guarantee is
quantified over every state.  `vote_mem_grid` is why that loses nothing — a threshold can
only matter through the count it cuts at. -/
noncomputable def clusterAt (O : Oracle μ S) (populations : Finset J)
    (x : Run Ω S J) (B : Budget) : Finset S :=
  clusterAround O B.cn B.cd (prefixesAt populations B.m x) (screenedAt O populations B x)
    (nz x) B.k

/-- A prefix is *decided* when the family's vote clears the state's accept or reject
threshold; otherwise it lands in the indecisive band and counts towards the FNR. -/
def decided (O : Oracle μ S) (lo hi : ℕ) (F : Finset S) (p : S) (ω : Ω) : Prop :=
  hi < voteCount O F p ω ∨ voteCount O F p ω ≤ lo

/-- **The family's cut is right at `p`**: where it decides, it decides the way the
oracle's noiseless label does.

Indecisive prefixes are excluded — they hold vacuously — which is what the round claims
and no more: they are boundary strings, and #215's test likewise scores only
`classifier.decisive`.  The FNR gate separately caps how much of a population can be
indecisive, so the two together say most of a population is decided, and decided right.

This is the *cut*, not per-member accept preservation.  A family of `k` suffixes votes
correctly while a member drifts — one member moves the vote by `1/k`, inside the margin —
so per-member preservation is the mechanism the algorithm reaches for (`pAP`, the
screening) and correct classification is the end.  It is also what the rest of E-L\*
consumes: the mask rows are read through this cut to identify states. -/
def cutCorrect (O : Oracle μ S) (lo hi : ℕ) (F : Finset S) (p : S) (ω : Ω) : Prop :=
  (hi < voteCount O F p ω → O.label p = 1) ∧ (voteCount O F p ω ≤ lo → O.label p = 0)

/-- A set of prefixes is **flat** when two of them never extend to the same query string.

`UniformSampler(DEFAULT_SAMPLE_LENGTH)` draws every probe at one fixed length — *"All of
E-L*'s signal comes from words drawn at this length"* — so `p * v = p' * v'` between two
probes forces `p = p'` on length alone.  Flatness is exactly what that buys, stated without
needing a length function: `Monoid` on its own does not know that strings factor. -/
def Flat (Pre : Set S) : Prop :=
  ∀ p ∈ Pre, ∀ p' ∈ Pre, ∀ v v' : S, p * v = p' * v' → p = p'

/-! ## The gate

`judge_family`: the FNR test and the accept-preserving test, and the binomial tail the
second is read against. -/

/-- `P[Bin(N,p) ≥ j]` — `scipy.stats.binom.sf(j-1, N, p)`. -/
noncomputable def binomSfGe (N : ℕ) (p : ℝ) (j : ℕ) : ℝ :=
  ∑ i ∈ Finset.Icc j N, (N.choose i : ℝ) * p ^ i * (1 - p) ^ (N - i)

open scoped Classical in
/-- The cut's two sides at a state: the prefixes it accepts, and all the prefixes it
decides. -/
noncomputable def cutSides (O : Oracle μ S) (lo hi : ℕ) (F P : Finset S) (ω : Ω) :
    Finset S × Finset S :=
  (P.filter (fun p => hi - 1 < voteCount O F p ω),
    P.filter (fun p => hi - 1 < voteCount O F p ω ∨ voteCount O F p ω ≤ lo))

open scoped Classical in
/-- The agreement count read off the cut's two sides and the hit set: the accept side's
hits plus the reject side's misses. -/
noncomputable def agreeOf (A Dset U : Finset S) : ℕ :=
  (A ∩ U).card + ((Dset \ A) \ U).card

open scoped Classical in
/-- **The gate's statistic**: `(agreements, decided count)` over the prefixes the cut
decides.

A prefix *agrees* when the seed's column reads the way the cut calls it — `mq p = 1` on the
accept side, `mq p ≠ 1` on the reject side.  Its mean is `1 − η` where the cut is right and
`η` where it is wrong, so over the decided prefixes the statistic concentrates at
`n·(1 − η) − W·(1 − 2η)` with `W` the number of mis-cut prefixes.

One statistic rather than one per side.  Two side-wise rate tests are each only as sharp as
their own side, and a side may hold as little as an `εcov` fraction of the sample, which
costs a second factor of `εcov` in everything the round has to achieve.  The decided count
is bounded below by the FNR test instead — it is `(1 − indecisionLimit)·m` — so the
combined test has no such floor to clear. -/
noncomputable def agreeCount (O : Oracle μ S) (lo hi : ℕ) (F P : Finset S) (ω : Ω) : ℕ × ℕ :=
  (agreeOf (cutSides O lo hi F P ω).1 (cutSides O lo hi F P ω).2
      (P.filter (fun p => mq O p ω = 1)),
    (cutSides O lo hi F P ω).2.card)

/-- **The rate a correct cut's accept side reads at, less the drift the gate must catch.**

A prefix the cut calls accepting reads as accepting on the seed's column with probability
`1 − η` when the cut is right and `η` when it is not, so a cut wrong on an `εcov` fraction
of that side reads at `(1 − η) − εcov(1 − 2η)`.  The null sits midway, which is the
placement that minimises the worse of the two failure modes at a given sample size: validity
spends the distance down to the drifted rate, termination the distance up to the clean one.

The *vote* cutoffs cannot serve here.  `hi/k` sits a fixed distance below `1 − η`, so drift
finer than that reads as clean however many prefixes are certified on — a floor set by the
threshold rather than by the sample.  (PR #286.)  Naming the signal is not new: the screen
already does it. -/
noncomputable def gateAcc (O : Oracle μ S) (εcov : ℝ) : ℝ :=
  (1 - O.η) - (1 / 2 - O.η) * εcov

/-- `drift_verdict`'s **ADMITTED**: the cut reads as its own class on the seed's column
across the prefixes it decides, at error rate `α` (`ACCEPT_PRESERVING_ERROR_RATE = 0.05`).

The rate tested against is `gateAcc`, derived from the oracle's signal and the coverage
being certified — not configuration.  `ACCEPT_PRESERVING_DRIFT` is what names it in the
code, and tying it to the coverage the caller wants rather than fixing it is PR #286's
remaining item.

`n₀` is the skip guard, `Budget.gmin`: below it the binomial tail is above any `α` whatever
the family does, so demanding the test there would make a population lying entirely on one
side of the language untestable.  The prefixes of a skipped test go uncertified, which is
why `Capped.gfloor` keeps `gmin` small against the prefix count.

Applied in `ret` to the family with `ε` removed and to `certOf`, the certification draws.
Removing `ε` matters because it is in every family, so the vote would otherwise contain
`mq p` — the very bit the split is scored against.  Judging on `certOf` matters because
the family was selected against `prefixesOf`.  (Issue #284.) -/
def admitted (O : Oracle μ S) (lo hi n₀ : ℕ) (εcov α : ℝ) (F P : Finset S) (ω : Ω) : Prop :=
  n₀ ≤ (agreeCount O lo hi F P ω).2 →
    binomSfGe (agreeCount O lo hi F P ω).2 (gateAcc O εcov)
      (agreeCount O lo hi F P ω).1 ≤ α

open scoped Classical in
/-- **The loop's return test** at a state: the FNR gate (PR #257: held per population, not
over their union) and the accept-preserving gate.  A family failing either is not
returned — `judge_family` sets its FNR to 1 and the loop samples more.

Both gates read the certification draws (`certOf`), which the family was never selected
from.  For the accept-preserving gate that is issue #284; for the FNR it is PR #289, and the
reason is the same — a family fitted to the prefixes it is then judged on votes more
decisively there than it will on fresh ones, so an FNR read off the table comes out
optimistic.  That matters because a cut is graded only where it decides.  The accept-
preserving split additionally drops the seed, whose own read is the bit being scored — and
the FNR is read off that same seed-dropped vote, so the two gates grade one cut. -/
noncomputable def ret (O : Oracle μ S) (populations : Finset J)
    (indecisionLimit εcov α : ℝ) (B : Budget) : Set (Run Ω S J) :=
  {x | (∀ j ∈ populations,
      (((certOf j B.m x).filter (fun p => ¬ decided O B.lo (B.hi - 1)
          ((clusterAt O populations x B).erase 1) p (nz x))).card : ℝ)
        ≤ indecisionLimit * (certOf j B.m x).card)
    ∧ ∀ j ∈ populations, admitted O B.lo B.hi B.gmin εcov α
        ((clusterAt O populations x B).erase 1) (certOf j B.m x) (nz x)}

/-! ## What the input distributions must satisfy -/

/-- **The collision mass of a prefix population**: the chance that two independent draws
from it coincide, `∑ₐ D({a})²`.

This is the one thing the populations must satisfy beyond being probability measures, and
it is irreducible rather than derivable — the same status as `pAP` for `Dsf`.  The gate's
certification draws carry independent noise only where they are *distinct*, because the
oracle is persistent; with a point-mass population every draw is the same string and no
amount of sampling certifies anything.  `S` is countable, so every `D j` is purely atomic
and this is never zero — the requirement is that it be small, not that it vanish.

Unlike the pool-size coupling that #284 retired, the bound needed here is *fixed*: small
relative to `εcov` and `δ`, not growing with the budget. -/
noncomputable def collisionMass (Dj : Measure S) : ℝ := ∑' a : S, (Dj.real {a}) ^ 2

/-- **What one tested state may cost.**  The five events `measureReal_admitFail_le` charges,
summed over the populations: the certification draws repeating or meeting the table, the
sample missing the wrong set, and the two gate sides. -/
noncomputable def stateFail (O : Oracle μ S) (populations : Finset J) (εcov ρ : ℝ)
    (B : Budget) : ℝ :=
  (populations.card : ℝ) * (((populations.card : ℝ) + 1) * (B.m : ℝ) ^ 2 * ρ
    + (Real.exp (-2 * (B.m : ℝ) * (εcov / 4) ^ 2)
      + 2 * Real.exp (-2 * (εcov / 32 * (B.m : ℝ)) * ((1 - 2 * O.η) * εcov / 32) ^ 2)))

/-- **The states the loop may stop at.**  The thresholds in order, and enough prefixes drawn
to carry the state's share of the error budget.

The share is what the computed ladder is for.  At a handful of prefixes the gate cannot be
sound — a wrong family passes a two-prefix test at constant probability — so the loop cannot
test there, and the guarantee cannot cover it.  The ladder has `L` rungs and they divide
`δ/2` between them, so a state carries its share once its prefix count is logarithmic in
`L` — and `L` is logarithmic in the prefix count. -/
structure Capped (O : Oracle μ S) (populations : Finset J) (εcov δ ρ : ℝ) (L : ℕ)
    (B : Budget) : Prop where
  /-- Reject strictly below accept, so the two gate sides are disjoint. -/
  lohi : B.lo < B.hi
  /-- The gate's floor sits under any side the soundness argument has to test, so skipping
  a side below it costs no coverage. -/
  gfloor : (B.gmin : ℝ) ≤ εcov / 32 * (B.m : ℝ)
  /-- Enough prefixes for the state's share of the budget: the ladder has `L` rungs and
  they divide `δ/2` between them. -/
  share : stateFail O populations εcov ρ B ≤ δ / (2 * L)

open scoped Classical in
/-- **The states the loop may stop at.**  The rungs of the computed ladder that carry their
share.  It is a `Finset`, so the union bound over it is a finite sum of `δ/(2·L)` terms and
no summable weight over all budgets is needed. -/
noncomputable def stoppable (O : Oracle μ S) (populations : Finset J)
    (εcov δ α pAP ρ : ℝ) : Finset Budget :=
  (schedule O populations εcov δ α pAP).filter
    (Capped O populations εcov δ ρ (ladderLen O populations εcov δ α pAP))

/-- The ladder's length, as it enters the collision allowance.  Each rung carries
`δ/(2·L)`, so the collision terms have to fit `L` times smaller. -/
noncomputable def capScale (O : Oracle μ S) (populations : Finset J)
    (εcov δ α pAP : ℝ) : ℝ :=
  (ladderLen O populations εcov δ α pAP : ℝ)

/-- **How much collision mass the populations may carry** at the solved budget: the round
pays `m²ρ` for prefix collisions, so the mass is capped against the prefix count and the
state's own share. -/
noncomputable def collisionCap (O : Oracle μ S) (populations : Finset J)
    (εcov δ α pAP : ℝ) : ℝ :=
  δ / (64 * ((populations.card : ℝ) + 3) ^ 3
    * ((prefCount O populations εcov δ α pAP : ℝ) ^ 2
      + (poolCount O populations εcov δ pAP : ℝ) ^ 2 + 1)
    * capScale O populations εcov δ α pAP)

/-! ## The theorem

Everything above is machinery; this is the claim. -/

/-- **The E-L\* clustering algorithm is PAC-correct.**

With probability `≥ 1 − δ` the adaptive loop **terminates**, and the family it returns — at
whatever state it chooses to stop — preserves acceptance on `≥ 1 − εcov` of **each** prefix
population.

The hypotheses, in the order they appear:

* the oracle has signal (`η < 1/2`) and there is a population to certify;
* the populations are supported on a `Flat` prefix set — one fixed probe length, which is
  what `UniformSampler(DEFAULT_SAMPLE_LENGTH)` draws at;
* their collision mass is at most `ρ`, and `pAP` of the suffix measure is
  accept-preserving.  These two are the irreducible facts about the input distributions:
  without the first, repeated draws carry no fresh noise, and without the second there is
  nothing for the pool to find;
* `indecisionLimit`, `α`, `εcov`, `δ` are in range, and the coverage the round charges
  (`cutBudget εcov`) fits inside the indecision the FNR gate tolerates;
* `ρ` and `Dsf`'s own collision mass fit the allowance `collisionCap`, which is computed
  from the budget rather than assumed.

Nothing here is a parameter of the algorithm tuned to the proof.  `Budget` is *computed*:
the rungs are `solvedBudgetAt` applied along `schedule`, and the guarantee is uniform over
the rungs that carry their share (`stoppable`), so the loop may stop wherever on the ladder
it likes.  The cluster is `clusterAround`, the Lloyd fixed point against its own
thresholded mean, seeded at `ε` and never drifting off it; the gates are the ones
`judge_family` applies; and the run space is the concrete `runLaw`, not an abstract space
assumed to exist. -/
def ClusteringCorrect (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (Pre : Set S) (indecisionLimit εcov α δ ρ pAP : ℝ) : Prop :=
  O.η < 1 / 2 →
  populations.Nonempty →
  Flat Pre →
  (∀ j ∈ populations, D j Preᶜ = 0) →
  (∀ j ∈ populations, collisionMass (D j) ≤ ρ) →
  0 < pAP →
  pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p} →
  0 < indecisionLimit →
  indecisionLimit ≤ 1 / 2 →
  0 < α →
  α < 1 / 2 →
  0 < εcov →
  εcov ≤ 1 →
  0 < δ →
  cutBudget εcov ≤ indecisionLimit / 2 →
  ρ ≤ collisionCap O populations εcov δ α pAP →
  collisionMass Dsf ≤ collisionCap O populations εcov δ α pAP →
  1 - δ ≤ (runLaw μ D Dsf).real
    {x | (∃ B : {B : Budget // B ∈ stoppable O populations εcov δ α pAP ρ},
        x ∈ ret O populations indecisionLimit εcov α B.val) ∧
      ∀ B : {B : Budget // B ∈ stoppable O populations εcov δ α pAP ρ},
        x ∈ ret O populations indecisionLimit εcov α B.val →
        ∀ j ∈ populations, 1 - εcov
          ≤ (D j).real {p | cutCorrect O B.val.lo B.val.hi
              (clusterAt O populations x B.val) p (nz x)}}

end OrthoDFA
