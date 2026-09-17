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
noncomputable def mq {S : Type*} [MeasurableSpace S] (O : Oracle μ S) (w : S) (ω : Ω) : ℝ :=
  O.label w + (1 - 2 * O.label w) * O.noise w ω

/-- General string-like type restriction. Satisfied by all strings over a finite alphabet. -/
class Stringlike (S : Type*) extends MeasurableSpace S, Monoid S, IsCancelMul S,
    MeasurableMul S, Countable S, MeasurableSingletonClass S where
  decEq : DecidableEq S

attribute [instance] Stringlike.decEq

variable {S : Type*} [Stringlike S]
variable {J : Type*} [Fintype J]

/-! ## The run space

One sample of the algorithm's randomness.  `runLaw` is a concrete measure, so the
independence the proof runs on is a lemma about it rather than a hypothesis. -/

/-- One run: the persistent noise, the suffix draws, each population's prefix draws, and the
certification draws.  The certification stream is separate because the gate must not be read
on the prefixes the family was selected from — see `certOf`. -/
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

/-- The `i`-th prefix from `certification_sample`: read only by the gate, never added to
the table. -/
def cert (j : J) (i : ℕ) (x : Run Ω S J) : S := x.2.2 (j, i)

/-! ## The loop's state

The field docs point forward at the screen and the gate, which the next section defines;
how the fields are *chosen* is the solved ladder, further down. -/

/-- The loop's state, all of it integer data.  There is no history and no real-valued
boundary: a boundary enters every event only through the count it cuts at, so the cut is
the state, and unlike a history this index is countable. -/
@[ext]
structure Budget where
  /-- How many suffixes have been drawn. -/
  M : ℕ
  /-- How many prefixes each population has drawn. -/
  m : ℕ
  k : ℕ
  /-- The cluster centre's cutoff, as the ratio `cn/cd`: `p` is on the accept side when
  `cn · #F < cd · voteCount F p`.

  A ratio and not a count because `identify_cluster_around` centres on
  `masks[cluster].mean(0) > decision_boundary`, a fraction of the current cluster — which
  has one member at the first iteration and `k` afterwards. -/
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

deriving instance DecidableEq for Budget

instance : Countable Budget :=
  Function.Injective.countable
    (f := fun b => (b.M, b.m, b.k, b.cn, b.cd, b.lo, b.hi, b.sc, b.scd, b.gmin))
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
  insert 1 ((Finset.range M).image (fun i => sfx i x))

/-- Population `j`'s first `m` draws, as a set: the table interns prefixes, so a repeated
draw is one column and not two. -/
noncomputable def prefixesOf (j : J) (m : ℕ) (x : Run Ω S J) : Finset S :=
  (Finset.range m).image (fun i => prf j i x)

open scoped Classical in
/-- Every population's representative prefixes, pooled. -/
noncomputable def prefixesAt (populations : Finset J) (m : ℕ)
    (x : Run Ω S J) : Finset S :=
  populations.biUnion (fun j => prefixesOf j m x)

/-- Population `j`'s `certification_sample` draws, which the family was never selected from.

The gate must be judged here and not on `prefixesOf`.  The cluster is chosen to agree with
the seed's *noisy* column on the representative prefixes — the very agreement the gate then
measures — so with a large enough pool it can match that column exactly, at which point the
accept side is `{p | mq p = 1}`, the agreement is total, and the gate admits a family whose
cut is the noise. -/
noncomputable def certOf (j : J) (m : ℕ) (x : Run Ω S J) : Finset S :=
  (Finset.range m).image (fun i => cert j i x)

/-! ### Screen -/

open scoped Classical in
/-- `_screen_cohort`'s statistic: how many representative prefixes the candidate's column
disagrees with the seed's on.  Its mean separates an accept-preserving candidate from one
carrying flip mass `φ` by `φ(1−2η)²`. -/
noncomputable def screenCount (O : Oracle μ S) (P : Finset S) (v : S) (ω : Ω) : ℕ :=
  (P.filter (fun p => ¬ ((mq O (p * v) ω = 1) ↔ (mq O p ω = 1)))).card

/-- The candidates the clustering actually sees.  A suffix the screen rejects never becomes
a fully observed column, so `identify_cluster_around` never ranks it. -/
noncomputable def screened (O : Oracle μ S) (sc scd : ℕ) (P cands : Finset S) (ω : Ω) :
    Finset S :=
  cands.filter (fun v => scd * screenCount O P v ω ≤ sc * P.card)

open scoped Classical in
/-- The screen at one budget state. -/
noncomputable def screenedAt (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (x : Run Ω S J) : Finset S :=
  screened O B.sc B.scd (prefixesAt populations B.m x) (poolAt B.M x) (nz x)

/-! ### Vote -/

open scoped Classical in
/-- How many of the family answer accept at `p`.

Every comparison in the algorithm is against a threshold on this count; the real-valued mean
carries no more information (`vote_mem_grid`, in `OrthoDFA.Adaptive`), which is what keeps
`Budget` integer data. -/
noncomputable def voteCount (O : Oracle μ S) (F : Finset S) (p : S) (ω : Ω) : ℕ :=
  (F.filter (fun v => mq O (p * v) ω = 1)).card

/-! ### Cluster -/

open scoped Classical in
/-- The greedy's output: a `k`-subset of `cands` minimising `∑ ℓ`. -/
noncomputable def leastLossSubset {S : Type*} (ℓ : S → ℝ) (cands : Finset S) (k : ℕ) :
    Finset S :=
  if h : (cands.powersetCard k).Nonempty then
    (Finset.exists_min_image (cands.powersetCard k) (fun T => ∑ x ∈ T, ℓ x) h).choose
  else ∅

open scoped Classical in
/-- `identify_cluster_around`'s loss: the Hamming distance from a candidate's mask row to the
cluster's own thresholded mean, `masks[cluster].mean(0) > decision_boundary`. -/
noncomputable def hammingLoss (O : Oracle μ S) (F : Finset S) (cn cd : ℕ) (P : Finset S)
    (ω : Ω) (v : S) : ℝ :=
  ((P.filter (fun p =>
    ¬ ((mq O (p * v) ω = 1) ↔ cn * F.card < cd * voteCount O F p ω))).card : ℝ)

open scoped Classical in
/-- `hammingLoss` zeroed off the candidate pool.

`leastLossSubset` is an argmin picked by `Classical.choose`, so it depends on the loss as a
function and not only on its values over `cands`.  Zeroing it elsewhere makes two draws
agreeing on the reads give literally the same loss, which `clusterAround_congr_mq` needs. -/
noncomputable def clusterLoss (O : Oracle μ S) (F : Finset S) (cn cd : ℕ) (P cands : Finset S)
    (ω : Ω) (v : S) : ℝ :=
  if v ∈ cands then hammingLoss O F cn cd P ω v else 0

open scoped Classical in
/-- One Lloyd step: recentre on the current cluster, then retake the `k` least-loss
candidates — but only while the seed is among them.  `identify_cluster_around` breaks out
(`if seed_local not in nearest`) rather than let the centre drift off `ε`.

The cohort is the seed with the `k−1` next best, taken only when that really is a least-loss
subset.  `np.argsort` is stable and the seed is the table's first column, so the seed wins
ties for the `k`-th place; modelling the ranking as an arbitrary argmin would let a tie throw
the seed out and stall the clustering at `{ε}`. -/
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
iterations from the seed already sit at the fixed point. -/
noncomputable def clusterAround (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω)
    (k : ℕ) : Finset S :=
  (lloydStep O cn cd P cands ω k)^[k * P.card + 1] {(1 : S)}

/-- The cluster at one budget state. -/
noncomputable def clusterAt (O : Oracle μ S) (populations : Finset J)
    (x : Run Ω S J) (B : Budget) : Finset S :=
  clusterAround O B.cn B.cd (prefixesAt populations B.m x) (screenedAt O populations B x)
    (nz x) B.k

/-! ### The cut -/

/-- `p` is decided when the family's vote clears the accept or reject threshold; otherwise
it lands in the indecisive band and counts towards the FNR. -/
def decided (O : Oracle μ S) (lo hi : ℕ) (F : Finset S) (p : S) (ω : Ω) : Prop :=
  hi < voteCount O F p ω ∨ voteCount O F p ω ≤ lo

/-- Where the family decides `p`, it decides the way the noiseless label does.

Indecisive prefixes hold vacuously, which is what the round claims and no more; the FNR gate
separately caps how much of a population can be indecisive.  Note this is the family's cut
and not per-member accept preservation: a family of `k` votes correctly while one member
drifts, since one member moves the vote by `1/k`. -/
def cutCorrect (O : Oracle μ S) (lo hi : ℕ) (F : Finset S) (p : S) (ω : Ω) : Prop :=
  (hi < voteCount O F p ω → O.label p = 1) ∧ (voteCount O F p ω ≤ lo → O.label p = 0)

/-! ## The gate

`judge_family`'s two tests. -/

/-- `P[Bin(N,p) ≥ j]` — `scipy.stats.binom.sf(j-1, N, p)`. -/
noncomputable def binomSfGe (N : ℕ) (p : ℝ) (j : ℕ) : ℝ :=
  ∑ i ∈ Finset.Icc j N, (N.choose i : ℝ) * p ^ i * (1 - p) ^ (N - i)

open scoped Classical in
/-- The prefixes the cut accepts, and all the prefixes it decides. -/
noncomputable def cutSides (O : Oracle μ S) (lo hi : ℕ) (F P : Finset S) (ω : Ω) :
    Finset S × Finset S :=
  (P.filter (fun p => hi - 1 < voteCount O F p ω),
    P.filter (fun p => hi - 1 < voteCount O F p ω ∨ voteCount O F p ω ≤ lo))

open scoped Classical in
/-- The accept side's hits plus the reject side's misses. -/
noncomputable def agreeOf (A Dset U : Finset S) : ℕ :=
  (A ∩ U).card + ((Dset \ A) \ U).card

open scoped Classical in
/-- The gate's statistic: `(agreements, decided count)`, where `p` agrees when the seed's
column reads the way the cut calls it.  Its mean is

    n·(1 − η) − W·(1 − 2η)

over `n` decided prefixes of which `W` are mis-cut.

One statistic and not one per side: a side may hold as little as an `εcov` fraction of the
sample, and a rate test is only as sharp as its own denominator.  The decided count is
floored by the FNR test at `(1 − indecisionLimit)·m` instead. -/
noncomputable def agreeCount (O : Oracle μ S) (lo hi : ℕ) (F P : Finset S) (ω : Ω) : ℕ × ℕ :=
  (agreeOf (cutSides O lo hi F P ω).1 (cutSides O lo hi F P ω).2
      (P.filter (fun p => mq O p ω = 1)),
    (cutSides O lo hi F P ω).2.card)

/-- The gate's null rate, `ACCEPT_PRESERVING_DRIFT`.  A correct cut reads at `1 − η` and one
wrong on an `εcov` fraction at `(1 − η) − εcov(1 − 2η)`; the null sits midway, which
minimises the worse of the two failure modes at a given sample size.

The vote cutoffs cannot serve here: `hi/k` sits a fixed distance below `1 − η`, so drift
finer than that reads as clean however many prefixes are certified on. -/
noncomputable def gateAcc (O : Oracle μ S) (εcov : ℝ) : ℝ :=
  (1 - O.η) - (1 / 2 - O.η) * εcov

/-- `drift_verdict`'s ADMITTED, at error rate `α` (`ACCEPT_PRESERVING_ERROR_RATE`).

`n₀` is `Budget.gmin`, the size below which the test is skipped rather than failed.

`ret` applies this to the family with `ε` removed, because `ε` is in every family and the
vote would otherwise contain `mq p` — the very bit the agreement is scored against. -/
def admitted (O : Oracle μ S) (lo hi n₀ : ℕ) (εcov α : ℝ) (F P : Finset S) (ω : Ω) : Prop :=
  n₀ ≤ (agreeCount O lo hi F P ω).2 →
    binomSfGe (agreeCount O lo hi F P ω).2 (gateAcc O εcov)
      (agreeCount O lo hi F P ω).1 ≤ α

open scoped Classical in
/-- `judge_family`'s two tests, both held per population: the FNR gate and the
accept-preserving gate.  A family failing either is not returned and the loop samples more.

Both read `certOf`.  A family fitted to the prefixes it is then judged on votes more
decisively there than on fresh ones, so an FNR read off the table comes out optimistic — and
a cut is graded only where it decides.  The FNR is read off the same seed-dropped vote as the
agreement gate, so the two grade one cut. -/
noncomputable def ret (O : Oracle μ S) (populations : Finset J)
    (indecisionLimit εcov α : ℝ) (B : Budget) : Set (Run Ω S J) :=
  {x | (∀ j ∈ populations,
      (((certOf j B.m x).filter (fun p => ¬ decided O B.lo (B.hi - 1)
          ((clusterAt O populations x B).erase 1) p (nz x))).card : ℝ)
        ≤ indecisionLimit * (certOf j B.m x).card)
    ∧ ∀ j ∈ populations, admitted O B.lo B.hi B.gmin εcov α
        ((clusterAt O populations x B).erase 1) (certOf j B.m x) (nz x)}

/-! ## The budget, solved rather than searched for

Every field of `Budget` is read off the condition it has to meet.  A condition is always a
tail `exp (-a) ≤ ε`, which asks only that `a` clear `log (1/ε)`, so a field is a logarithm
of the error budget.  Where a field's condition mentions the share it will be given — which
depends on that field — it mentions it only through *its* logarithm, and `log x ≤ 2√x`
closes the loop in one step. -/

/-- `s = 1/2 − η`. -/
noncomputable def sig (O : Oracle μ S) : ℝ := 1 / 2 - O.η

/-- What the gate's margin can absorb, so what a round charges wrongly-cut prefixes at. -/
noncomputable def cutBudget (εcov : ℝ) : ℝ := εcov / 64

/-- A clean family's vote fails at `exp (-κ·s²/2)` and the round pays that at the cut
budget, so `κ` is the logarithm of the two together. -/
noncomputable def famCount (O : Oracle μ S) (populations : Finset J) (εcov δ : ℝ) : ℕ :=
  ⌈2 * Real.log (32 * (populations.card : ℝ) / (cutBudget εcov * δ)) / sig O ^ 2⌉₊
    + ⌈1 / sig O⌉₊ + 1

/-- What one family member may flip: the cut budget spread over the family and the
populations. -/
noncomputable def flipBudget (O : Oracle μ S) (populations : Finset J) (εcov δ : ℝ) : ℝ :=
  cutBudget εcov / (8 * (populations.card : ℝ) * (famCount O populations εcov δ : ℝ))

/-- The screen's margin, at the flip budget. -/
noncomputable def screenMargin (O : Oracle μ S) (populations : Finset J) (εcov δ : ℝ) : ℝ :=
  flipBudget O populations εcov δ * sig O ^ 2

/-- Enough suffixes that a family of `k` fits inside the findable fraction. -/
noncomputable def poolCount (O : Oracle μ S) (populations : Finset J) (εcov δ pAP : ℝ) : ℕ :=
  ⌈2 * ((famCount O populations εcov δ : ℝ) + 1) / pAP⌉₊
    + ⌈Real.log (32 * (populations.card : ℝ) / δ) / (2 * (pAP / 2) ^ 2)⌉₊

/-- The slower of the two rates the state's own share is measured at. -/
noncomputable def shareRate (O : Oracle μ S) (εcov : ℝ) : ℝ :=
  εcov * (sig O * εcov / 16) ^ 2 / 16

/-- The prefix count the share asks for.  The ladder's length is logarithmic in the prefix
count, so the condition refers to `log` of the count itself; `(√(A/r) + 2/r)²` is the closed
form clearing `A + log (m + 2)` at rate `r`, the logarithm being under the square root. -/
noncomputable def shareCount (O : Oracle μ S) (populations : Finset J) (εcov δ : ℝ) : ℕ :=
  ⌈(Real.sqrt ((Real.log (16 * (populations.card : ℝ) / δ) + 4) / shareRate O εcov)
    + 2 / shareRate O εcov) ^ 2⌉₊

/-- The counts the round's tails ask for, summed so each is met. -/
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
/-- The state at a given prefix count: every other field read off the condition it has to
meet. -/
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

/-- How many times the loop runs the gate: the prefix count halves down to one.  This is
the number of states the error budget is divided among; there is no other state the loop
can return at. -/
noncomputable def ladderLen (O : Oracle μ S) (populations : Finset J)
    (εcov δ α pAP : ℝ) : ℕ :=
  Nat.log 2 (prefCount O populations εcov δ α pAP) + 1

open scoped Classical in
/-- The states the loop runs the gate at: the ladder `m, m/2, m/4, …`. -/
noncomputable def schedule (O : Oracle μ S) (populations : Finset J)
    (εcov δ α pAP : ℝ) : Finset Budget :=
  (Finset.range (ladderLen O populations εcov δ α pAP)).image
    (fun i => solvedBudgetAt O populations εcov δ pAP
      (prefCount O populations εcov δ α pAP / 2 ^ i))

/-- What one tested state may cost: the three events `measureReal_admitFail_le` charges,
summed over the populations — the certification draws repeating or meeting the table, the
sample missing the wrong set, and the gate passing on a wrong cut. -/
noncomputable def stateFail (O : Oracle μ S) (populations : Finset J) (εcov ρ : ℝ)
    (B : Budget) : ℝ :=
  (populations.card : ℝ) * (((populations.card : ℝ) + 1) * (B.m : ℝ) ^ 2 * ρ
    + (Real.exp (-2 * (B.m : ℝ) * (εcov / 4) ^ 2)
      + 2 * Real.exp (-2 * (εcov / 32 * (B.m : ℝ)) * ((1 - 2 * O.η) * εcov / 32) ^ 2)))

/-- A state can be stopped at when its thresholds are in order and it has drawn enough
prefixes to carry its share of the error budget.

At a handful of prefixes the gate cannot be sound — a wrong family passes a two-prefix test
at constant probability — so the loop cannot test there and the guarantee cannot cover it. -/
structure Capped (O : Oracle μ S) (populations : Finset J) (εcov δ ρ : ℝ) (L : ℕ)
    (B : Budget) : Prop where
  /-- Reject strictly below accept, so the two gate sides are disjoint. -/
  lohi : B.lo < B.hi
  /-- The skip guard sits under any sample the soundness argument has to test, so skipping
  below it costs no coverage. -/
  gfloor : (B.gmin : ℝ) ≤ εcov / 32 * (B.m : ℝ)
  /-- The ladder has `L` rungs and they divide `δ/2` between them. -/
  share : stateFail O populations εcov ρ B ≤ δ / (2 * L)

open scoped Classical in
/-- The rungs of the ladder that carry their share.  A `Finset`, so the union bound over it
is a finite sum of `δ/(2·L)` terms and no summable weight over all budgets is needed. -/
noncomputable def stoppable (O : Oracle μ S) (populations : Finset J)
    (εcov δ α pAP ρ : ℝ) : Finset Budget :=
  (schedule O populations εcov δ α pAP).filter
    (Capped O populations εcov δ ρ (ladderLen O populations εcov δ α pAP))

/-- The ladder's length as it enters the collision allowance: each rung carries `δ/(2·L)`,
so the collision terms have to fit `L` times smaller. -/
noncomputable def capScale (O : Oracle μ S) (populations : Finset J)
    (εcov δ α pAP : ℝ) : ℝ :=
  (ladderLen O populations εcov δ α pAP : ℝ)

/-- How much collision mass the populations may carry: the round pays `m²ρ` for prefix
collisions, so the mass is capped against the prefix count and the state's own share. -/
noncomputable def collisionCap (O : Oracle μ S) (populations : Finset J)
    (εcov δ α pAP : ℝ) : ℝ :=
  δ / (64 * ((populations.card : ℝ) + 3) ^ 3
    * ((prefCount O populations εcov δ α pAP : ℝ) ^ 2
      + (poolCount O populations εcov δ pAP : ℝ) ^ 2 + 1)
    * capScale O populations εcov δ α pAP)

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

The hypotheses, in the order they appear: the oracle has signal and there is a population to
certify; the populations are supported on a `Flat` prefix set; their collision mass is at
most `ρ` and `pAP` of the suffix measure is accept-preserving; `indecisionLimit`, `α`, `εcov`
and `δ` are in range with `cutBudget εcov` inside the indecision the FNR gate tolerates; and
`ρ` and `Dsf`'s collision mass fit `collisionCap`.

No hypothesis is a parameter of the algorithm: `Budget` is computed (`solvedBudgetAt` along
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
    (Pre : Set S) (indecisionLimit εcov α δ ρ pAP : ℝ),
  O.η < 1 / 2 →
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
