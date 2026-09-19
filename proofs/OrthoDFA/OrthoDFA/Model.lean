import Mathlib.Tactic
import Mathlib.Probability.ProductMeasure
import Mathlib.Probability.Independence.InfinitePi

/-!
# The model: the oracle, the algorithm, and the theorem

Everything the statement says is here: the oracle, every definition the algorithm is built
from, and `ClusteringCorrect`, which `OrthoDFA.Main` proves.  The other files hold only
lemmas, so checking that the development claims what it says it claims means reading this
file and no other.

Each definition's doc names the Python it models and the place the two could come apart.

Known modelling gap.  The draws here are i.i.d. and deduplicated downstream (`poolAt`,
`prefixesAt`), whereas `_draw_cohort` and `sample_more_prefixes` redraw on a duplicate —
sampling without replacement.  Deduplicated i.i.d. draws give a pool at most as large, so
this is the conservative model, but the collision mass enters as `m²ρ` and the statement
therefore caps `ρ` at `collisionCap`.  Lifting that cap needs without-replacement
concentration (Hoeffding 1963) or a non-atomic prefix distribution.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-! ## The oracle -/

/-- The persistent signal oracle: random classification noise on query strings. -/
structure Oracle {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (S : Type*) [MeasurableSpace S] where
  /-- The language.  Membership in it is the bit the oracle is asked for. -/
  L : Set S
  L_meas : MeasurableSet L
  /-- One persistent RCN bit per query string. -/
  noise : S → Ω → ℝ
  η : ℝ
  hη : η ≤ 1 / 2
  /-- The noise is an IID bernoulli with parameter η -/
  noise_meas : ∀ w, Measurable (noise w)
  noise_indep : iIndepFun noise μ
  noise_bit : ∀ w, ∀ᵐ ω ∂μ, noise w ω = 0 ∨ noise w ω = 1
  noise_mean : ∀ w, μ[noise w] = η

/-- Membership as a bit, `ℓ(w) = 1[w ∈ L]`.  The arithmetic form, since every use sums or
averages it. -/
noncomputable def Oracle.label {S : Type*} [MeasurableSpace S] (O : Oracle μ S) : S → ℝ :=
  Set.indicator O.L 1

/-- The membership query the oracle answers, `MQ w = ℓ(w) ⊕ noise(w)`. -/
noncomputable def Oracle.mq {S : Type*} [MeasurableSpace S] (O : Oracle μ S) (w : S) (ω : Ω) : ℝ :=
  O.label w + (1 - 2 * O.label w) * O.noise w ω

/-- General string-like type restriction. Satisfied by all strings over a finite alphabet. -/
class Stringlike (S : Type*) extends MeasurableSpace S, Monoid S, IsCancelMul S,
    MeasurableMul S, Countable S, MeasurableSingletonClass S where
  decEq : DecidableEq S

attribute [instance] Stringlike.decEq

variable {S : Type*} [Stringlike S]
variable {J : Type*} [Fintype J]

/-! ## The run space

One sample of the algorithm's randomness.  `runMeasure` is a concrete measure, so the
independence the proof runs on is a lemma about it rather than a hypothesis. -/

/-- Randomness associated with a run of the algorithm. -/
abbrev Run (Ω S J : Type*) :=
  Ω ×                       -- the oracle's persistent noise
  (((ℕ → S) ×               -- the suffix draws
    (J → ℕ → S)) ×          -- the prefix draws, one stream per population
   (J → ℕ → S))             -- the certification draws, one stream per population

/-- The three components jointly independent, each stream i.i.d. -/
noncomputable def runMeasure (μ : Measure Ω) (D : J → Measure S) (Dsf : Measure S) :
    Measure (Run Ω S J) :=
  μ.prod ((((Measure.infinitePi fun _ : ℕ => Dsf).prod
      (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j))).prod
    (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j))

instance (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] : IsProbabilityMeasure (runMeasure μ D Dsf) := by
  unfold runMeasure; infer_instance

/-- The run's persistent noise. -/
def oracleNoise (x : Run Ω S J) : Ω := x.1

/-- The `i`-th suffix drawn. -/
def suffixDraw (i : ℕ) (x : Run Ω S J) : S := x.2.1.1 i

/-- The `i`-th prefix drawn from population `j`. -/
def prefixDraw (j : J) (i : ℕ) (x : Run Ω S J) : S := x.2.1.2 j i

/-- The `i`-th prefix from `certification_sample`: read only by the gate, never added to
the table. -/
def certPrefix (j : J) (i : ℕ) (x : Run Ω S J) : S := x.2.2 j i

/-! ## The loop's state

The field docs point forward at the screen and the gate, which the next section defines;
how the fields are *chosen* is the solved ladder, further down. -/

/-- The loop's state, all of it integer data.  There is no history and no real-valued
boundary: a boundary enters every event only through the count it cuts at, so the cut is
the state, and unlike a history this index is countable. -/
@[ext]
structure State where
  /-- How many suffixes have been drawn. -/
  nsuff : ℕ
  /-- How many prefixes each population has drawn. -/
  npref : ℕ
  k : ℕ
  /-- The centre's decision boundary, as the ratio `cn/cd`: `p` is on the accept side when
  more than a `cn/cd` fraction of `F` answers accept at `p`, written cross-multiplied as
  `cn · #F < cd · voteCount F p` so the state stays integral.

  This is `identify_cluster_around`'s `decision_boundary`, which it compares the cluster's
  thresholded mean `masks[cluster].mean(0)` against.  A ratio and not a count because that
  cluster has one member at the first iteration and `k` afterwards, so no fixed count serves
  both; `solvedStateAt` sets it to `1/2`, a majority vote. -/
  cn : ℕ
  cd : ℕ
  /-- Reject at or below this count. -/
  lo : ℕ
  /-- Accept above this count. -/
  hi : ℕ
  /-- The screen's cutoff, as the ratio `sc/scd`: a candidate disagreeing with the seed's
  column on more than this fraction never becomes a clustering candidate.

  `_screen_cohort` tests each drawn suffix against `same_family_rate = 2η(1−η)` and only
  survivors reach `fully_observed()`, which is what `identify_cluster_around` clusters over.
  So the screen and not the Lloyd ranking is what bounds the family's flip mass. -/
  sc : ℕ
  scd : ℕ
  /-- The gate skips a sample below this size.

  The binomial tail at a handful of prefixes is above any `α`, so without a floor a
  population lying entirely on one side of the language could never be tested.  Prefixes of
  a skipped test go uncertified, so `gmin` has to stay small against the prefix count. -/
  gmin : ℕ

deriving instance DecidableEq for State

instance : Countable State :=
  Function.Injective.countable
    (f := fun b => (b.nsuff, b.npref, b.k, b.cn, b.cd, b.lo, b.hi, b.sc, b.scd, b.gmin))
    (by rintro ⟨⟩ ⟨⟩ h; simp_all)

/-! ## The algorithm

`sample_suffix_family` in the order it runs: draw, screen, cluster, read the cut off the
family's vote. -/

/-! ### Draw -/

/-- The first `M` suffixes drawn, with the seed.

`identify_cluster_around` asserts the seed is among the candidates — `pst.table.column(v)`
promotes it to fully observed at the top of every round.  Without it `lloydStep` could never
fire and every family would be the degenerate `{ε}`. -/
noncomputable def poolAt (M : ℕ) (x : Run Ω S J) : Finset S :=
  insert 1 ((Finset.range M).image (fun i => suffixDraw i x))

/-- Population `j`'s first `m` draws, as a set: the table interns prefixes, so a repeated
draw is one column and not two. -/
noncomputable def prefixesOf (j : J) (m : ℕ) (x : Run Ω S J) : Finset S :=
  (Finset.range m).image (fun i => prefixDraw j i x)

/-- Every population's representative prefixes, pooled. -/
noncomputable def prefixesAt (populations : Finset J) (m : ℕ)
    (x : Run Ω S J) : Finset S :=
  populations.biUnion (fun j => prefixesOf j m x)

/-- Population `j`'s `certification_sample` draws, which the family was never selected from.

The gate must be judged here and not on `prefixesOf`.  The cluster is chosen to agree with
the seed's *noisy* column on the representative prefixes — the very agreement the gate then
measures — so with a large enough pool it can match that column exactly, at which point the
accept side is `{p | O.mq p = 1}`, the agreement is total, and the gate admits a family whose
cut is the noise. -/
noncomputable def certOf (j : J) (m : ℕ) (x : Run Ω S J) : Finset S :=
  (Finset.range m).image (fun i => certPrefix j i x)

/-! ### Screen -/

/-- `_screen_cohort`'s statistic: how many representative prefixes the candidate's column
disagrees with the seed's on.  Its mean separates an accept-preserving candidate from one
carrying flip mass `φ` by `φ(1−2η)²`. -/
noncomputable def screenCount (mq : S → Ω → ℝ) (P : Finset S) (v : S) (ω : Ω) : ℕ :=
  (P.filter (fun p => ¬ ((mq (p * v) ω = 1) ↔ (mq p ω = 1)))).card

/-- The pool's own disagreement floor: the least disagreement any non-seed candidate shows
against the seed's column.

A candidate's disagreement rate is `2η(1−η) + φ(1−2η)²` for its flip mass `φ`, so the floor
sits at `2η(1−η)` once the pool holds an accept-preserving suffix — which is what `pAP` and
`poolCount` buy.  The screen's cutoff is read off this, so the noise rate never enters
the screen: `_screen_cohort`'s `same_family_rate` is measured,
not assumed.

The seed is excluded because its two reads are the *same* query string, so it disagrees on
nothing and would pin the floor at zero. -/
noncomputable def screenBase (mq : S → Ω → ℝ) (P cands : Finset S) (ω : Ω) : ℕ :=
  if h : (cands.erase 1).Nonempty then
    (cands.erase 1).inf' h (fun v => screenCount mq P v ω)
  else 0

open scoped Classical in
/-- The candidates the clustering actually sees: those within `sc/scd` of the pool's floor.
A suffix the screen rejects never becomes a fully observed column, so
`identify_cluster_around` never ranks it. -/
noncomputable def screened (mq : S → Ω → ℝ) (sc scd : ℕ) (P cands : Finset S) (ω : Ω) :
    Finset S :=
  cands.filter (fun v =>
    scd * screenCount mq P v ω ≤ scd * screenBase mq P cands ω + sc * P.card)

/-- The screen at one budget state. -/
noncomputable def screenedAt (mq : S → Ω → ℝ) (populations : Finset J) (B : State)
    (x : Run Ω S J) : Finset S :=
  screened mq B.sc B.scd (prefixesAt populations B.npref x) (poolAt B.nsuff x) (oracleNoise x)

/-! ### Vote -/

/-- How many of the family answer accept at `p`.

Every comparison in the algorithm is against a threshold on this count; the real-valued mean
carries no more information (`vote_mem_grid`, in `OrthoDFA.Adaptive`), which is what keeps
`State` integer data. -/
noncomputable def voteCount (mq : S → Ω → ℝ) (F : Finset S) (p : S) (ω : Ω) : ℕ :=
  (F.filter (fun v => mq (p * v) ω = 1)).card

/-! ### Cluster -/

/-- The greedy's output: a `k`-subset of `cands` minimising `∑ ℓ`. -/
noncomputable def leastLossSubset {S : Type*} (ℓ : S → ℝ) (cands : Finset S) (k : ℕ) :
    Finset S :=
  if h : (cands.powersetCard k).Nonempty then
    (Finset.exists_min_image (cands.powersetCard k) (fun T => ∑ x ∈ T, ℓ x) h).choose
  else ∅

/-- `identify_cluster_around`'s loss: the Hamming distance from a candidate's mask row to the
cluster's own thresholded mean, `masks[cluster].mean(0) > decision_boundary`. -/
noncomputable def hammingLoss (mq : S → Ω → ℝ) (F : Finset S) (cn cd : ℕ) (P : Finset S)
    (ω : Ω) (v : S) : ℝ :=
  ((P.filter (fun p =>
    ¬ ((mq (p * v) ω = 1) ↔ cn * F.card < cd * voteCount mq F p ω))).card : ℝ)

/-- `hammingLoss` zeroed off the candidate pool.

`leastLossSubset` is an argmin picked by `Classical.choose`, so it depends on the loss as a
function and not only on its values over `cands`.  Zeroing it elsewhere makes two draws
agreeing on the reads give literally the same loss, which `clusterAround_congr_mq` needs. -/
noncomputable def clusterLoss (mq : S → Ω → ℝ) (F : Finset S) (cn cd : ℕ) (P cands : Finset S)
    (ω : Ω) (v : S) : ℝ :=
  if v ∈ cands then hammingLoss mq F cn cd P ω v else 0

/-- One Lloyd step: recentre on the current cluster, then retake the `k` least-loss
candidates — but only while the seed is among them.  `identify_cluster_around` breaks out
(`if seed_local not in nearest`) rather than let the centre drift off `ε`.

The cohort is the seed with the `k−1` next best, taken only when that really is a least-loss
subset.  `np.argsort` is stable and the seed is the table's first column, so the seed wins
ties for the `k`-th place; modelling the ranking as an arbitrary argmin would let a tie throw
the seed out and stall the clustering at `{ε}`. -/
noncomputable def lloydStep (mq : S → Ω → ℝ) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (F : Finset S) : Finset S :=
  if ∀ w ∈ cands, w ∉ insert (1 : S)
        (leastLossSubset (clusterLoss mq F cn cd P cands ω) (cands.erase 1) (k - 1)) →
      ∀ v ∈ insert (1 : S)
        (leastLossSubset (clusterLoss mq F cn cd P cands ω) (cands.erase 1) (k - 1)),
      clusterLoss mq F cn cd P cands ω v ≤ clusterLoss mq F cn cd P cands ω w
  then insert (1 : S)
    (leastLossSubset (clusterLoss mq F cn cd P cands ω) (cands.erase 1) (k - 1)) else F

/-- `identify_cluster_around` iterated to its fixed point.  The total loss is a natural
number bounded by `k·#P` that strictly decreases at each improving step, so `k·#P + 1`
iterations from the seed already sit at the fixed point. -/
noncomputable def clusterAround (mq : S → Ω → ℝ) (cn cd : ℕ) (P cands : Finset S) (ω : Ω)
    (k : ℕ) : Finset S :=
  (lloydStep mq cn cd P cands ω k)^[k * P.card + 1] {(1 : S)}

/-- The cluster at one budget state. -/
noncomputable def clusterAt (mq : S → Ω → ℝ) (populations : Finset J)
    (x : Run Ω S J) (B : State) : Finset S :=
  clusterAround mq B.cn B.cd (prefixesAt populations B.npref x) (screenedAt mq populations B x)
    (oracleNoise x) B.k

/-! ### The cut -/

/-- `p` is decided when the family's vote clears the accept or reject threshold; otherwise
it lands in the indecisive band and counts towards the FNR. -/
def decided (mq : S → Ω → ℝ) (lo hi : ℕ) (F : Finset S) (p : S) (ω : Ω) : Prop :=
  hi < voteCount mq F p ω ∨ voteCount mq F p ω ≤ lo

/-- Where the family decides `p`, it decides the way the noiseless label does.

Indecisive prefixes hold vacuously, which is what the round claims and no more; the FNR gate
separately caps how much of a population can be indecisive.  Note this is the family's cut
and not per-member accept preservation: a family of `k` votes correctly while one member
drifts, since one member moves the vote by `1/k`. -/
def cutCorrect (O : Oracle μ S) (lo hi : ℕ) (F : Finset S) (p : S) (ω : Ω) : Prop :=
  (hi < voteCount O.mq F p ω → O.label p = 1) ∧ (voteCount O.mq F p ω ≤ lo → O.label p = 0)

/-! ## The gate

`judge_family`'s two tests. -/

/-- `P[Bin(N,p) ≥ j]` — `scipy.stats.binom.sf(j-1, N, p)`. -/
noncomputable def binomSfGe (N : ℕ) (p : ℝ) (j : ℕ) : ℝ :=
  ∑ i ∈ Finset.Icc j N, (N.choose i : ℝ) * p ^ i * (1 - p) ^ (N - i)

/-- The prefixes the cut accepts, and all the prefixes it decides. -/
noncomputable def cutSides (mq : S → Ω → ℝ) (lo hi : ℕ) (F P : Finset S) (ω : Ω) :
    Finset S × Finset S :=
  (P.filter (fun p => hi - 1 < voteCount mq F p ω),
    P.filter (fun p => hi - 1 < voteCount mq F p ω ∨ voteCount mq F p ω ≤ lo))

/-- The accept side's hits plus the reject side's misses. -/
noncomputable def agreeOf (A Dset U : Finset S) : ℕ :=
  (A ∩ U).card + ((Dset \ A) \ U).card

/-- The gate's statistic: `(agreements, decided count)`, where `p` agrees when the seed's
column reads the way the cut calls it.  Its mean is

    n·(1 − η) − W·(1 − 2η)

over `n` decided prefixes of which `W` are mis-cut.

One statistic and not one per side: a side may hold as little as an `εcov` fraction of the
sample, and a rate test is only as sharp as its own denominator.  The decided count is
floored by the FNR test at `(1 − indecisionLimit)·m` instead. -/
noncomputable def agreeCount (mq : S → Ω → ℝ) (lo hi : ℕ) (F P : Finset S) (ω : Ω) : ℕ × ℕ :=
  (agreeOf (cutSides mq lo hi F P ω).1 (cutSides mq lo hi F P ω).2
      (P.filter (fun p => mq p ω = 1)),
    (cutSides mq lo hi F P ω).2.card)

/-- `drift_verdict`'s ADMITTED, at error rate `α` (`ACCEPT_PRESERVING_ERROR_RATE`): the cut
agrees with the seed's own read significantly more often than a coin flip.

`drift_verdict` holds each side to the family's own thresholds.  The sides are pooled here,
since a side can be a vanishing fraction of the sample, and held to the centre they straddle.
Nothing is certified by this test: validity comes from how the family is built, and the test
only has to pass a family that was built well.

`n₀` is `State.gmin`, the size below which the test is skipped rather than failed.

`ret` applies this to the family with `ε` removed, because `ε` is in every family and the
vote would otherwise contain `mq p` — the very bit the agreement is scored against. -/
def admitted (mq : S → Ω → ℝ) (lo hi n₀ : ℕ) (α : ℝ) (F P : Finset S) (ω : Ω) : Prop :=
  n₀ ≤ (agreeCount mq lo hi F P ω).2 →
    binomSfGe (agreeCount mq lo hi F P ω).2 (1 / 2) (agreeCount mq lo hi F P ω).1 ≤ α

open scoped Classical in
/-- `judge_family`: a family smaller than the round asked for is not used whatever it would
measure; otherwise the FNR gate and the accept-preserving gate, both held per population.

Both read `certOf`.  A family fitted to the prefixes it is then judged on votes more
decisively there than on fresh ones, so an FNR read off the table comes out optimistic — and
a cut is graded only where it decides.  The FNR is read off the same seed-dropped vote as the
agreement gate, so the two grade one cut. -/
noncomputable def ret (mq : S → Ω → ℝ) (populations : Finset J)
    (indecisionLimit α : ℝ) (B : State) : Set (Run Ω S J) :=
  {x | B.k ≤ (clusterAt mq populations x B).card
    ∧ (∀ j ∈ populations,
      (((certOf j B.npref x).filter (fun p => ¬ decided mq B.lo (B.hi - 1)
          ((clusterAt mq populations x B).erase 1) p (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * (certOf j B.npref x).card)
    ∧ ∀ j ∈ populations, admitted mq B.lo B.hi B.gmin α
        ((clusterAt mq populations x B).erase 1) (certOf j B.npref x) (oracleNoise x)}

/-! ## The budget, solved rather than searched for

Every field of `State` is read off the condition it has to meet.  A condition is always a
tail `exp (-a) ≤ ε`, which asks only that `a` clear `log (1/ε)`, so a field is a logarithm
of the error budget.  No condition refers to the count solving it: a rung is charged in
proportion to its own prefix count, so the ladder's length — which is `log` of that count —
never enters. -/

/-- `s = 1/2 − η`. -/
noncomputable def sig (η : ℝ) : ℝ := 1 / 2 - η

/-- What the gate's margin can absorb, so what a round charges wrongly-cut prefixes at. -/
noncomputable def cutBudget (εcov : ℝ) : ℝ := εcov / 64

/-- A family's vote fails at `exp (-κ·s²/32)`: the vote sits `s` from either clean mean and
a flipping `3/8` of the family spends `3s/4` of that, leaving the margin `s/8` the count is
read at.  The round pays that tail at the cut budget, so `κ` is the logarithm of the two
together.  `⌈8/s⌉` is what makes `κ·s/8` clear the `1` that rounding `κ/2` to a count
costs. -/
noncomputable def famCount (η : ℝ) (populations : Finset J) (εcov δ : ℝ) : ℕ :=
  ⌈32 * Real.log (128 * (populations.card : ℝ) / (cutBudget εcov * δ)) / sig η ^ 2⌉₊
    + ⌈8 / sig η⌉₊ + 1

/-- What one family member may flip.  The vote absorbs three eighths of the family flipping,
so Markov charges the cut budget at a constant and not at the family's size. -/
noncomputable def flipBudget (_η : ℝ) (populations : Finset J) (εcov _δ : ℝ) : ℝ :=
  cutBudget εcov / (8 * (populations.card : ℝ))

/-- The screen's margin, at the flip budget. -/
noncomputable def screenMargin (η : ℝ) (populations : Finset J) (εcov δ : ℝ) : ℝ :=
  flipBudget η populations εcov δ * sig η ^ 2

/-- Enough suffixes that a family of `k` fits inside the findable fraction. -/
noncomputable def poolCount (η : ℝ) (populations : Finset J) (εcov δ pAP : ℝ) : ℕ :=
  ⌈2 * ((famCount η populations εcov δ : ℝ) + 1) / pAP⌉₊
    + ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (pAP / 2) ^ 2)⌉₊

/-- The counts the round's tails ask for, summed so each is met. -/
noncomputable def prefCount (η : ℝ) (populations : Finset J)
    (εcov δ α pAP : ℝ) : ℕ :=
  ⌈Real.log (128 * (populations.card : ℝ)
      * ((poolCount η populations εcov δ pAP : ℝ) + 2) ^ 2 / δ)
      / (2 * (screenMargin η populations εcov δ / 2) ^ 2)⌉₊
    + ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (cutBudget εcov / 4) ^ 2)⌉₊
    + ⌈64 * Real.log (1 / α) / (εcov * (sig η * εcov / 4) ^ 2)⌉₊
    + ⌈64 * Real.log (256 * (populations.card : ℝ) / δ)
        / (εcov * (sig η * εcov / 4) ^ 2)⌉₊
    + ⌈Real.log (128 * (populations.card : ℝ)
        * ((poolCount η populations εcov δ pAP : ℝ) + 1) / δ)
        / (2 * ((populations.card : ℝ) * flipBudget η populations εcov δ) ^ 2)⌉₊
    + ⌈64 / εcov⌉₊
    + 1

/-- The state at a given prefix count: every other field read off the condition it has to
meet. -/
noncomputable def solvedStateAt (η : ℝ) (populations : Finset J)
    (εcov δ pAP : ℝ) (mi : ℕ) : State where
  nsuff := poolCount η populations εcov δ pAP
  npref := mi
  k := famCount η populations εcov δ + 1
  cn := 1
  cd := 2
  lo := ⌈(famCount η populations εcov δ : ℝ) / 2⌉₊ - 1
  hi := ⌈(famCount η populations εcov δ : ℝ) / 2⌉₊ + 1
  sc := ⌈((⌈1 / (2 * screenMargin η populations εcov δ)⌉₊ + 1 : ℕ) : ℝ)
    * screenMargin η populations εcov δ⌉₊
  scd := ⌈1 / (2 * screenMargin η populations εcov δ)⌉₊ + 1
  gmin := ⌊εcov * (mi : ℝ) / 32⌋₊

/-- How many times the loop runs the gate: the prefix count halves down to one.  There is no
other state the loop can return at. -/
noncomputable def ladderLen (η : ℝ) (populations : Finset J)
    (εcov δ α pAP : ℝ) : ℕ :=
  Nat.log 2 (prefCount η populations εcov δ α pAP) + 1

/-- The states the loop runs the gate at: the ladder `m, m/2, m/4, …`. -/
noncomputable def schedule (η : ℝ) (populations : Finset J)
    (εcov δ α pAP : ℝ) : Finset State :=
  (Finset.range (ladderLen η populations εcov δ α pAP)).image
    (fun i => solvedStateAt η populations εcov δ pAP
      (prefCount η populations εcov δ α pAP / 2 ^ i))

/-- What one tested state may cost, summed over the populations: the certification draws
repeating or meeting the table, the sample missing the wrong set, a family member flipping more
than the screen allows, the sample holding too many prefixes the family flips, and the family's
vote misfiring on too many of the rest — and then, once, the pool's draws colliding.

The pool's *findability* is not here: that event does not mention the prefixes, so it is the
same at every rung and is charged once rather than per rung. -/
noncomputable def stateFail (η₀ : ℝ) (populations : Finset J) (εcov δ ρ ρsf : ℝ)
    (B : State) : ℝ :=
  (populations.card : ℝ) * (((populations.card : ℝ) + 1) * (B.npref : ℝ) ^ 2 * ρ
    + (Real.exp (-2 * (B.npref : ℝ) * (εcov / 4) ^ 2)
      + (((B.npref : ℝ) ^ 2 * ρ
          + ((B.nsuff : ℝ) + 2) ^ 2
            * Real.exp (-2 * (B.npref : ℝ) * (screenMargin η₀ populations εcov δ / 2) ^ 2)
          + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ)
            * ((populations.card : ℝ) * flipBudget η₀ populations εcov δ) ^ 2))
        + (Real.exp (-2 * (B.npref : ℝ) * (cutBudget εcov / 4) ^ 2)
          + Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * (sig η₀ / 8) ^ 2) / cutBudget εcov))))
    + (B.nsuff : ℝ) ^ 2 * ρsf

/-- A state can be stopped at when its thresholds are in order and it has drawn enough
prefixes to carry its share of the error budget.

At a handful of prefixes neither the screen nor the certification sample can see a dirty
family, so the guarantee cannot cover a stop there. -/
structure Capped (η₀ : ℝ) (populations : Finset J) (εcov δ ρ ρsf pAP : ℝ) (N : ℕ)
    (B : State) : Prop where
  /-- Reject strictly below accept, so the two gate sides are disjoint. -/
  lohi : B.lo < B.hi
  /-- The centre's boundary is a proper fraction. -/
  bdry : B.cn < B.cd
  /-- There are prefixes to read the rate off. -/
  mpos : 0 < B.npref
  /-- Two accept-preserving draws are expected in the pool, so one of them is not the seed
  and the floor has a clean candidate to sit at. -/
  found : (2 : ℝ) ≤ (B.nsuff : ℝ) * (pAP / 2)
  /-- The rungs divide `δ/4` between them in proportion to their prefix counts, which halve
  down the ladder and so sum to at most `2N`; the other `δ/4` of the validity half pays for
  the pool's findability, once.

  Proportional and not uniform: a rung's tails are exponential in its own count, so a
  `δ/(4·L)` split would make every count clear `log L`, and `L` is read off the top count. -/
  share : stateFail η₀ populations εcov δ ρ ρsf B ≤ δ * (B.npref : ℝ) / (8 * N)

open scoped Classical in
/-- The rungs of the ladder that carry their share.  A `Finset`, so the union bound over it is
a finite sum and no summable weight over all budgets is needed. -/
noncomputable def stoppable (η₀ : ℝ) (populations : Finset J)
    (εcov δ α pAP ρ ρsf : ℝ) : Finset State :=
  (schedule η₀ populations εcov δ α pAP).filter
    (Capped η₀ populations εcov δ ρ ρsf pAP (prefCount η₀ populations εcov δ α pAP))

/-- How much collision mass the populations may carry: the round pays `m²ρ` for prefix
collisions, so the mass is capped against the prefix count and the state's own share. -/
noncomputable def collisionCap (η : ℝ) (populations : Finset J)
    (εcov δ α pAP : ℝ) : ℝ :=
  δ / (64 * ((populations.card : ℝ) + 3) ^ 3
    * ((prefCount η populations εcov δ α pAP : ℝ) ^ 2
      + (poolCount η populations εcov δ pAP : ℝ) ^ 2 + 1))

/-! ## What the input distributions must satisfy -/

/-- Two prefixes of a flat set never extend to the same query string.

`UniformSampler(DEFAULT_SAMPLE_LENGTH)` draws every probe at one fixed length, so
`p * v = p' * v'` forces `p = p'` on length alone.  Stated this way because `Monoid` on its
own does not know that strings factor. -/
def Flat (Pre : Set S) : Prop :=
  ∀ p ∈ Pre, ∀ p' ∈ Pre, ∀ v v' : S, p * v = p' * v' → p = p'

/-- The chance two independent draws coincide, `∑ₐ D({a})²`.

Irreducible rather than derivable, the same status as `pAP` for `Dsf`: the oracle is
persistent, so certification draws carry independent noise only where they are distinct, and
with a point-mass population no amount of sampling certifies anything.  `S` is countable, so
every `D j` is purely atomic and this is never zero — it must be small, not vanish. -/
noncomputable def collisionMass (Dj : Measure S) : ℝ := ∑' a : S, (Dj.real {a}) ^ 2

/-! ## The theorem -/

/-- The E-L\* clustering algorithm is PAC-correct: with probability `≥ 1 − δ` the loop
terminates, and the family it returns — at whatever state it stops — cuts `≥ 1 − εcov` of
each prefix population the way the noiseless oracle does.

The algorithm is not told the noise rate, only an upper bound `η₀` on it — `η₀` is what
`min_signal_strength` gives `build_pst`, and every computed field of `State` is solved from
it, never from `O.η`.  Nothing asks the bound to be tight: the screen reads its cutoff off
`screenBase`, a quantity it measures, and the gate is held to a coin flip, so over-estimating
the noise only costs prefixes.

The hypotheses, in the order they appear: the oracle's noise is at most `η₀`, which has
signal; there is a population to certify; the populations are supported on a `Flat` prefix
set; their collision mass is at most `ρ` and `pAP` of the suffix measure is
accept-preserving; `indecisionLimit`, `α`, `εcov` and `δ` are in range with `cutBudget εcov`
inside the indecision the FNR gate tolerates; and `ρ` and `Dsf`'s collision mass fit
`collisionCap`.

No hypothesis is a parameter of the algorithm: `State` is computed (`solvedStateAt` along
`schedule`), and the guarantee is uniform over the rungs that carry their share, so the loop
may stop wherever on the ladder it likes.

Closed: the spaces, their instances and the data are all quantified here, so
`clustering_correct : ClusteringCorrect` is the whole claim and nothing is hidden in a
binder on the theorem. -/
def ClusteringCorrect : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {J : Type*} [Fintype J]
    (O : Oracle μ S) (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (Pre : Set S) (η₀ indecisionLimit εcov α δ ρ pAP : ℝ),
  O.η ≤ η₀ →
  η₀ < 1 / 2 →
  populations.Nonempty →
  Flat Pre →
  (∀ j ∈ populations, D j Preᶜ = 0) →
  (∀ j ∈ populations, collisionMass (D j) ≤ ρ) →
  0 < pAP →
  pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  0 < indecisionLimit →
  indecisionLimit ≤ 1 / 2 →
  0 < α →
  α < 1 / 2 →
  0 < εcov →
  εcov ≤ 1 →
  0 < δ →
  cutBudget εcov ≤ indecisionLimit / 2 →
  ρ ≤ collisionCap η₀ populations εcov δ α pAP →
  collisionMass Dsf ≤ collisionCap η₀ populations εcov δ α pAP →
  1 - δ ≤ (runMeasure μ D Dsf).real
    {x | (∃ B : {B : State // B ∈ stoppable η₀ populations εcov δ α pAP ρ (collisionMass Dsf)},
        x ∈ ret O.mq populations indecisionLimit α B.val) ∧
      ∀ B : {B : State // B ∈ stoppable η₀ populations εcov δ α pAP ρ (collisionMass Dsf)},
        x ∈ ret O.mq populations indecisionLimit α B.val →
        ∀ j ∈ populations, 1 - εcov
          ≤ (D j).real {p | cutCorrect O B.val.lo B.val.hi
              (clusterAt O.mq populations x B.val) p (oracleNoise x)}}

end OrthoDFA
