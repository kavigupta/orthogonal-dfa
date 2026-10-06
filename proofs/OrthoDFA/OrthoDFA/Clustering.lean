import Mathlib.Tactic
import Mathlib.Probability.ProductMeasure
import Mathlib.Probability.Independence.InfinitePi

/-!
# The model: the oracle, the algorithm, and the theorem

Everything `ClusteringGuarantee` mentions is defined here, so auditing the claim means reading
this file and no other.  Each definition's doc names the Python it models.

Known modelling gap.  The draws here are i.i.d. and deduplicated downstream (`poolAt`,
`prefixesAt`), whereas `_draw_cohort` and `sample_more_prefixes` redraw on a duplicate —
sampling without replacement.  Deduplicated i.i.d. draws give a pool at most as large, so this
is the conservative model, but it is why the claim caps the collision mass.

Known modelling gap.  The Python re-estimates `pst.decision_boundary` from its reads
(`transition_resolver`, `counterexample_synthesis`, `identify_cluster_around`) and cuts the
family's vote there; here the cut is centred at half the family (`lo`, `hi`, and the gate's
`cutAccepts`).  So the signal is the worse rate's margin `½ − max(ηIn, ηOut)` rather than the
half-gap `(1 − ηIn − ηOut)/2`, and `misShare` reads the classes at `η₀` and `1 − η₀` where
`round_rates` reads them at `decision_boundary ∓ min_signal_strength`.

Known modelling gap.  `identify_cluster_around` groups the pool by k-means and a merge test,
takes the group whose rows read in another class than the seed least often on the prefixes the
screen never saw, and fills the family from the rest by distance to that group's mean;
`clusterAround` does none of these.  The proof only uses that the
cluster holds the seed, lies in the screened pool, has `k` members when the pool holds that many
and at most `k` otherwise, and is a function of the pool's reads on the prefixes, all of which
`identify_cluster_around` also satisfies -- the last given `pst.rng`, which k-means draws its
start from and which is independent of the certification sample.

Known modelling gap.  The Python sizes the family and its band from the boundary it estimates
(`smallest_readable_family`, `readable_size_and_margin`); here the family is `famCount + 1` and
the band is sized so that a vote whose mean lies outside it lands on its far side at most
`crossLimit` of the time.  The returned family, seed included, is cut at `lo` and `hi + 1`, wider
by the seed's one read than the gate's cut over the family without it, where the Python cuts both
at one rate.

Known modelling gap.  The FNR test and the gate here both read the first `npref` certification
draws of each population, counting a repeated draw each time.  The Python reads the gate on
`min(alignment_size, certification_budget)` draws from every population, drawn without
replacement, then once more on as many as `prefixes_to_certify` asks for from the one left
undecided.

Known modelling gap.  `judge_family` reads the FNR on the table's own prefixes, the ones
`identify_cluster_around` clustered the family on; `ret` reads it on the certification sample.
The table's votes are fitted to its noise and read as more decisive than they are, so the
claim's bound on the undecided mass holds of the test `ret` runs, not of the Python's.

Known modelling gap.  Each state is its own look at the gate, read at `lookLevel α look`, so
the looks a run makes spend `α` between them.  The Python reads every verdict at
`ACCEPT_PRESERVING_VERDICT_ERROR_RATE`, half of `α`, however many it makes.  The Python's FNR
also reads 1 for a family that decides no prefix one of the two ways, which `ret` does not.

Known modelling gap.  `_screen_cohort` screens each cohort once, when it is drawn, against the
table as it then stands, by a staircase of binomial tests against a floor fitted to the cohort;
`screened` screens the whole pool at the state's prefix count, against a fixed cutoff above the
pool's least count.  `_screen_cohort` also reads only the prefixes `seed_scoring` leaves it, half
of them by a hash of the string, so that what the screen admitted on is not what the anchor
scores on.

Known modelling gap.  The states here are a ladder of prefix counts halving from `prefCount` at
one pool size, and the claim covers a stop only at a rung of at least `validCount` prefixes.  The
Python grows the table as it goes, by `num_addtl_prefixes` prefixes or a cohort of suffixes after
each refusal, and stops at whatever table a round first passes on, which the claim need not
cover.  It also gives up after `ACCEPT_PRESERVING_GIVE_UP` refusals by the gate, where the loop
here never does: the give-up bounds a search on a target with no accept-preserving family, so a
run cannot hang.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-! ## The oracle -/

/-- The persistent signal oracle: random classification noise on query strings. -/
structure Oracle {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (S : Type*) [MeasurableSpace S] where
  /-- The language.  Membership in it is the bit the oracle is asked for. -/
  L : Set S
  L_meas : MeasurableSet L
  /-- One persistent noise bit per query string. -/
  noise : S → Ω → ℝ
  ηIn : ℝ
  ηOut : ℝ
  /-- The bits are independent Bernoullis, at one rate on the language and another off it. -/
  noise_meas : ∀ w, Measurable (noise w)
  noise_indep : iIndepFun noise μ
  noise_bit : ∀ w, ∀ᵐ ω ∂μ, noise w ω = 0 ∨ noise w ω = 1
  noise_mean_in : ∀ w ∈ L, μ[noise w] = ηIn
  noise_mean_out : ∀ w ∉ L, μ[noise w] = ηOut

/-- The worse of the two noise rates.  `½ − η` is the signal: every read leans toward its
true bit by at least that much. -/
noncomputable def Oracle.η {S : Type*} [MeasurableSpace S] (O : Oracle μ S) : ℝ :=
  max O.ηIn O.ηOut

/-- Membership as a bit, `ℓ(w) = 1[w ∈ L]`. -/
noncomputable def Oracle.label {S : Type*} [MeasurableSpace S] (O : Oracle μ S) : S → ℝ :=
  Set.indicator O.L 1

/-- The membership query the oracle answers, `MQ w = ℓ(w) ⊕ noise(w)`. -/
noncomputable def Oracle.mq {S : Type*} [MeasurableSpace S] (O : Oracle μ S) (w : S) (ω : Ω) : ℝ :=
  O.label w + (1 - 2 * O.label w) * O.noise w ω

/-- Satisfied by the strings over a finite alphabet. -/
class Stringlike (S : Type*) extends MeasurableSpace S, Monoid S, IsCancelMul S,
    MeasurableMul S, Countable S, MeasurableSingletonClass S where
  decEq : DecidableEq S

attribute [instance] Stringlike.decEq

variable {S : Type*} [Stringlike S]
variable {J : Type*} [Fintype J]

/-! ## The run space

`runMeasure` is a concrete measure, so the independence the proof uses is a lemma about it
rather than a hypothesis. -/

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

def oracleNoise (x : Run Ω S J) : Ω := x.1

def suffixDraw (i : ℕ) (x : Run Ω S J) : S := x.2.1.1 i

def prefixDraw (j : J) (i : ℕ) (x : Run Ω S J) : S := x.2.1.2 j i

/-- The `i`-th `certification_sample` draw: read only by the gate, never added to the
table. -/
def certPrefix (j : J) (i : ℕ) (x : Run Ω S J) : S := x.2.2 j i

/-! ## The loop's state -/

/-- All of it integer data: a boundary enters every event only through the count it cuts at,
so the cut is the state. -/
@[ext]
structure State where
  /-- How many suffixes have been drawn. -/
  nsuff : ℕ
  /-- How many prefixes each population has drawn. -/
  npref : ℕ
  /-- The cluster's size, seed included. -/
  k : ℕ
  /-- Reject at or below this count. -/
  lo : ℕ
  /-- Accept above this count. -/
  hi : ℕ
  /-- `_screen_cohort`'s cutoff, as the ratio `sc/scd`: a candidate disagreeing with the seed's
  column on more than this fraction of prefixes above the pool's floor never reaches
  `fully_observed()`, so `identify_cluster_around` never clusters over it. -/
  sc : ℕ
  scd : ℕ
  /-- Which look at the gate this is. -/
  look : ℕ

/-! ## The algorithm

`sample_suffix_family` in the order it runs: draw, screen, cluster, read the cut off the
family's vote. -/

/-! ### Draw -/

/-- The first `M` suffixes drawn, with the seed `ε`: `pst.table.column(v)` promotes it to
fully observed at the top of every round. -/
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

/-- Population `j`'s `certification_sample` draws, which the family was never clustered on. -/
noncomputable def certOf (j : J) (m : ℕ) (x : Run Ω S J) : Finset S :=
  (Finset.range m).image (fun i => certPrefix j i x)

/-! ### Screen -/

/-- `_screen_cohort`'s statistic: how many prefixes the candidate's column disagrees with the
seed's on. -/
noncomputable def screenCount (mq : S → Ω → ℝ) (P : Finset S) (v : S) (ω : Ω) : ℕ :=
  (P.filter (fun p => ¬ ((mq (p * v) ω = 1) ↔ (mq p ω = 1)))).card

/-- The least `screenCount` of any candidate but the seed, whose two reads are the same query
string.  `_screen_cohort` reads its `same_family_rate` off the cohort in the same way rather
than from the declared noise rate. -/
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

noncomputable def screenedAt (mq : S → Ω → ℝ) (populations : Finset J) (B : State)
    (x : Run Ω S J) : Finset S :=
  screened mq B.sc B.scd (prefixesAt populations B.npref x) (poolAt B.nsuff x) (oracleNoise x)

/-! ### Vote -/

/-- How many of the family answer accept at `p`. -/
noncomputable def voteCount (mq : S → Ω → ℝ) (F : Finset S) (p : S) (ω : Ω) : ℕ :=
  (F.filter (fun v => mq (p * v) ω = 1)).card

/-- What `voteCount` averages to over the oracle's noise. -/
noncomputable def meanVote (O : Oracle μ S) (F : Finset S) (p : S) : ℝ :=
  ∑ v ∈ F, μ.real {ω | O.mq (p * v) ω = 1}

/-! ### Cluster -/

/-- The greedy's output: a `k`-subset of `cands` minimising `∑ ℓ`. -/
noncomputable def leastLossSubset {S : Type*} (ℓ : S → ℝ) (cands : Finset S) (k : ℕ) :
    Finset S :=
  if h : (cands.powersetCard k).Nonempty then
    (Finset.exists_min_image (cands.powersetCard k) (fun T => ∑ x ∈ T, ℓ x) h).choose
  else ∅

/-- The Hamming distance from a candidate's mask row to the cluster's own thresholded mean,
`masks[cluster].mean(0) > decision_boundary`, the boundary written `cn/cd`. -/
noncomputable def hammingLoss (mq : S → Ω → ℝ) (F : Finset S) (cn cd : ℕ) (P : Finset S)
    (ω : Ω) (v : S) : ℝ :=
  ((P.filter (fun p =>
    ¬ ((mq (p * v) ω = 1) ↔ cn * F.card < cd * voteCount mq F p ω))).card : ℝ)

/-- `hammingLoss`, zeroed off the candidates: `leastLossSubset` picks by `Classical.choose`,
so it sees the loss as a function, and two draws agreeing on the reads must give the same one. -/
noncomputable def clusterLoss (mq : S → Ω → ℝ) (F : Finset S) (cn cd : ℕ) (P cands : Finset S)
    (ω : Ω) (v : S) : ℝ :=
  if v ∈ cands then hammingLoss mq F cn cd P ω v else 0

/-- One Lloyd step: recentre on the current cluster, then retake the `k` least-loss
candidates, but only while the seed is among them.  The seed wins ties for the `k`-th
place. -/
noncomputable def lloydStep (mq : S → Ω → ℝ) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (F : Finset S) : Finset S :=
  if ∀ w ∈ cands, w ∉ insert (1 : S)
        (leastLossSubset (clusterLoss mq F cn cd P cands ω) (cands.erase 1) (k - 1)) →
      ∀ v ∈ insert (1 : S)
        (leastLossSubset (clusterLoss mq F cn cd P cands ω) (cands.erase 1) (k - 1)),
      clusterLoss mq F cn cd P cands ω v ≤ clusterLoss mq F cn cd P cands ω w
  then insert (1 : S)
    (leastLossSubset (clusterLoss mq F cn cd P cands ω) (cands.erase 1) (k - 1)) else F

/-- `lloydStep` iterated to its fixed point, which `k·#P + 1` steps reach: the
total loss is a natural number at most `k·#P` and falls at every improving step. -/
noncomputable def clusterAround (mq : S → Ω → ℝ) (cn cd : ℕ) (P cands : Finset S) (ω : Ω)
    (k : ℕ) : Finset S :=
  (lloydStep mq cn cd P cands ω k)^[k * P.card + 1] {(1 : S)}

/-- The cluster without its seed, recentred at half the family. -/
noncomputable def clusterAt (mq : S → Ω → ℝ) (populations : Finset J)
    (x : Run Ω S J) (B : State) : Finset S :=
  (clusterAround mq 1 2 (prefixesAt populations B.npref x) (screenedAt mq populations B x)
    (oracleNoise x) B.k).erase 1

/-- The family the round returns, `vs`: the cluster with its seed. -/
noncomputable def familyAt (mq : S → Ω → ℝ) (populations : Finset J)
    (x : Run Ω S J) (B : State) : Finset S :=
  insert 1 (clusterAt mq populations x B)

/-! ### The cut -/

/-- `p` is decided when the family's vote clears the accept or reject threshold; otherwise
it lands in the indecisive band and counts towards the FNR. -/
def decided (mq : S → Ω → ℝ) (lo hi : ℕ) (F : Finset S) (p : S) (ω : Ω) : Prop :=
  hi < voteCount mq F p ω ∨ voteCount mq F p ω ≤ lo

/-- Where the family decides `p`, it decides the way the noiseless label does.  Indecisive
prefixes hold vacuously.  This is the family's cut, not accept preservation by each member. -/
def cutCorrect (O : Oracle μ S) (lo hi : ℕ) (F : Finset S) (p : S) (ω : Ω) : Prop :=
  (hi < voteCount O.mq F p ω → O.label p = 1) ∧ (voteCount O.mq F p ω ≤ lo → O.label p = 0)

/-! ## The gate

`judge_family`'s FNR test, and `drift_verdict` read at every population. -/

/-- `P[Bin(N,p) ≥ j]` — `scipy.stats.binom.sf(j-1, N, p)`. -/
noncomputable def binomSfGe (N : ℕ) (p : ℝ) (j : ℕ) : ℝ :=
  ∑ i ∈ Finset.Icc j N, (N.choose i : ℝ) * p ^ i * (1 - p) ^ (N - i)

/-- `P[Bin(N,p) ≤ j]` — `scipy.stats.binom.cdf(j, N, p)`. -/
noncomputable def binomCdfLe (N : ℕ) (p : ℝ) (j : ℕ) : ℝ :=
  ∑ i ∈ Finset.range (j + 1), (N.choose i : ℝ) * p ^ i * (1 - p) ^ (N - i)

/-- `p` lies in `clopper_pearson`'s interval at `level` for `hits` of `trials`: neither binomial
tail at `p` falls below `level / 2`. -/
def cpCovers (trials hits : ℕ) (level p : ℝ) : Prop :=
  level / 2 ≤ binomSfGe trials p hits ∧ level / 2 ≤ binomCdfLe trials p hits

/-- `misclassified_bounds`' `wrong`, weighted by the sides' masses: the accept side holds `m` of
a population and reads 1 at rate `a`, the reject side at rate `r`, and the share `e` of each side
in the other class is solved from

    a = p₁ − e (p₁ − p₀),   r = p₀ + e (p₁ − p₀),   p₀, p₁ = η₀, 1 − η₀,

then clipped to `[0, 1]`. -/
noncomputable def misShare (η₀ m a r : ℝ) : ℝ :=
  m * max 0 (min 1 ((1 - η₀ - a) / (1 - 2 * η₀)))
    + (1 - m) * max 0 (min 1 ((r - η₀) / (1 - 2 * η₀)))

/-- `_split_counts`' side of the cut: accept where at least half the family reads accept. -/
def cutAccepts (mq : S → Ω → ℝ) (F : Finset S) (p : S) (ω : Ω) : Prop :=
  F.card ≤ 2 * voteCount mq F p ω

open scoped Classical in
/-- `_split_counts` on population `j`'s first `n` certification draws, `((a₁, n₁), (a₀, n₀))`:
how many draws the cut accepts and how many of those the seed reads as 1, then the same for the
ones it rejects.  A draw repeated is counted each time. -/
noncomputable def splitCounts (mq : S → Ω → ℝ) (F : Finset S) (j : J) (n : ℕ)
    (x : Run Ω S J) : (ℕ × ℕ) × (ℕ × ℕ) :=
  ((((Finset.range n).filter (fun i => cutAccepts mq F (certPrefix j i x) (oracleNoise x)
        ∧ mq (certPrefix j i x) (oracleNoise x) = 1)).card,
      ((Finset.range n).filter (fun i =>
        cutAccepts mq F (certPrefix j i x) (oracleNoise x))).card),
    (((Finset.range n).filter (fun i => ¬ cutAccepts mq F (certPrefix j i x) (oracleNoise x)
        ∧ mq (certPrefix j i x) (oracleNoise x) = 1)).card,
      ((Finset.range n).filter (fun i =>
        ¬ cutAccepts mq F (certPrefix j i x) (oracleNoise x))).card))

/-- `misclassified_bounds`' bound is at most `limit`: so is `misShare` at every mass and pair of
rates that the counts' Clopper-Pearson intervals at `level` hold, the mass read off both sides'
counts as `ends` reads it. -/
def misAdmits (η₀ level limit : ℝ) (c : (ℕ × ℕ) × (ℕ × ℕ)) : Prop :=
  ∀ m a r : ℝ, m ∈ Set.Icc (0 : ℝ) 1 → a ∈ Set.Icc (0 : ℝ) 1 → r ∈ Set.Icc (0 : ℝ) 1 →
    cpCovers (c.1.2 + c.2.2) c.1.2 level m → cpCovers (c.1.2 + c.2.2) c.2.2 level (1 - m) →
    cpCovers c.1.2 c.1.1 level a → cpCovers c.2.2 c.2.1 level r →
    misShare η₀ m a r ≤ limit

/-- `look_level`: look `k` at the gate is held to `6α/(π(k+1))²`, which sums to `α`. -/
noncomputable def lookLevel (α : ℝ) (k : ℕ) : ℝ := α * 6 / (Real.pi * (k + 1)) ^ 2

open scoped Classical in
/-- `misclassification_limit`. -/
noncomputable def missLimit (uni j : J) (εcov : ℝ) : ℝ := if j = uni then εcov else 1 / 2

open scoped Classical in
/-- `judge_family`: a family smaller than the round asked for is not used; otherwise the FNR
test per population on `certOf`, and `drift_verdict` on every population's first `npref`
certification draws, at the state's look, each interval at a quarter of the look's level per
population as `misclassified_bounds` spends it.  The FNR reads the family with its seed; the
gate reads it without, since the seed's read is the bit the gate scores. -/
noncomputable def ret (mq : S → Ω → ℝ) (populations : Finset J) (uni : J)
    (η₀ indecisionLimit εcov α : ℝ) (B : State) : Set (Run Ω S J) :=
  {x | B.k ≤ (clusterAt mq populations x B).card + 1
    ∧ (∀ j ∈ populations,
      (((certOf j B.npref x).filter (fun p => ¬ decided mq B.lo (B.hi + 1)
          (familyAt mq populations x B) p (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * (certOf j B.npref x).card)
    ∧ ∀ j ∈ populations, misAdmits η₀ (lookLevel α B.look / (4 * populations.card))
        (missLimit uni j εcov) (splitCounts mq (clusterAt mq populations x B) j B.npref x)}

/-- `misShare` of a population itself: `Dj`'s mass on the cut's accept side, and the rates at
which the seed reads 1 on each side, at the noise the run drew. -/
noncomputable def misShareOf (η₀ : ℝ) (mq : S → Ω → ℝ) (Dj : Measure S) (F : Finset S)
    (ω : Ω) : ℝ :=
  misShare η₀ (Dj.real {p | cutAccepts mq F p ω})
    (Dj.real {p | cutAccepts mq F p ω ∧ mq p ω = 1} / Dj.real {p | cutAccepts mq F p ω})
    (Dj.real {p | ¬ cutAccepts mq F p ω ∧ mq p ω = 1} / Dj.real {p | ¬ cutAccepts mq F p ω})

/-! ## What the input distributions must satisfy -/

/-- Two prefixes never extend, by drawn suffixes or by `ε`, to the same query string.

`sample_more_suffixes` draws every suffix at the sampler's length `L`, so `p * v = p' * v'`
with both suffixes drawn forces `p = p'` on length alone.  Against the seed `ε` it would need
`p` to be longer than `p'` by `L`, which prefixes whose lengths lie within a window narrower than
`L` never are.  Stated this way because `Monoid` on its own does not know that strings factor. -/
def Flat (Pre Suf : Set S) : Prop :=
  ∀ p ∈ Pre, ∀ p' ∈ Pre, ∀ v ∈ insert 1 Suf, ∀ v' ∈ insert 1 Suf, p * v = p' * v' → p = p'

/-- The chance two independent draws coincide, `∑ₐ D({a})²`.  The oracle is persistent, so
draws carry independent noise only where they are distinct; `S` is countable, so this is never
zero and the claim asks only that it be small. -/
noncomputable def collisionMass (Dj : Measure S) : ℝ := ∑' a : S, (Dj.real {a}) ^ 2

/-! ## The theorem -/

/-- The E-L\* clustering algorithm is correct at a polynomial cost.  With probability
`≥ 1 − δ − α` the loop stops at one of `states`, and at whichever it stops the family's cut reads,
on every population, the way a cut misclassifying at most `missLimit` of it would read at the
signal the algorithm was told -- `misShareOf`, at most `εcov` on the uniform pool and a half on
any other -- and the family leaves at most `2·indecisionLimit` of each population undecided.  No
state draws more prefixes than the first count below, nor asks for a family or a suffix pool
larger than the sizes below.  Some state draws no more than the second count, so a stop is
covered from there; the first count is what the loop may need before a round passes.  Every
state's band is wide enough that a vote over a family no larger than its own, whose mean lies
above the band, lands at or below `lo`, or one whose mean lies at or below `lo` lands above `hi`,
at most `crossLimit` of the time.

`misShareOf` reads the oracle at the noise the run drew, and assumes nothing about its rates
beyond `η₀`: an oracle cleaner than that reads as misclassifying less than it does on the side
its noise has room on.  With both rates exactly `η₀` and the noise averaged over, it is the share
of the population the cut misclassifies.

The algorithm is told only an upper bound `η₀` on the noise rate.  `uni` is the uniform pool.
`pAP` lower-bounds the share of suffixes that preserve membership for every prefix, and `cap`
bounds the collision mass. -/
def ClusteringGuarantee : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {J : Type*} [Fintype J]
    (O : Oracle μ S) (populations : Finset J) (uni : J) (Pre Suf : Set S)
    (η₀ indecisionLimit εcov α δ pAP crossLimit : ℝ),
  O.η ≤ η₀ →
  η₀ < 1 / 2 →
  uni ∈ populations →
  Flat Pre Suf →
  0 < pAP →
  0 < indecisionLimit →
  indecisionLimit ≤ 1 / 2 →
  0 < α →
  α < 1 / 2 →
  0 < εcov →
  εcov ≤ 1 →
  0 < δ →
  δ ≤ 1 →
  0 < crossLimit →
  crossLimit ≤ 1 →
  ∃ cap : ℝ,
    0 < cap ∧
    ∀ (D : J → Measure S) (Dsf : Measure S),
      (∀ j, IsProbabilityMeasure (D j)) → IsProbabilityMeasure Dsf →
      (∀ j ∈ populations, D j Preᶜ = 0) →
      Dsf Sufᶜ = 0 →
      pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
      ∀ ρ : ℝ,
      (∀ j ∈ populations, collisionMass (D j) ≤ ρ) →
      ρ ≤ cap →
      collisionMass Dsf ≤ cap →
      ∃ states : Finset State,
        (∀ B ∈ states, (B.npref : ℝ) ≤
          4000
          * (populations.card : ℝ) ^ 2
          * Real.log (
            ((populations.card : ℝ) + 2) * B.nsuff
            / (δ * α * pAP * min ((1 / 2 - η₀) * εcov) indecisionLimit)
          )
          / ((1 / 2 - η₀) ^ 4 * min ((1 / 2 - η₀) * εcov) indecisionLimit ^ 2)
        ) ∧
        (∃ B ∈ states, (B.npref : ℝ) ≤
          104
          * (populations.card : ℝ) ^ 2
          * Real.log (((populations.card : ℝ) + 2) * ((B.nsuff : ℝ) + 2) / δ)
          / ((1 / 2 - η₀) ^ 4 * min εcov indecisionLimit ^ 2)
        ) ∧
        (∀ B ∈ states,
          (B.k : ℝ) ≤ 64 * Real.log (2 / (min ((1 / 2 - η₀) * εcov) indecisionLimit * crossLimit))
            / (1 / 2 - η₀) ^ 2
          ∧ (B.nsuff : ℝ) ≤ 128 * Real.log
                (2 / (min ((1 / 2 - η₀) * εcov) indecisionLimit * crossLimit))
              / ((1 / 2 - η₀) ^ 2 * pAP)
            + 16 * Real.log (((populations.card : ℝ) + 2) / δ) / pAP ^ 2) ∧
        (∀ B ∈ states, ∀ F : Finset S, F.card + 1 ≤ B.k → ∀ p,
          (B.hi < meanVote O F p → μ.real {ω | voteCount O.mq F p ω ≤ B.lo} ≤ crossLimit)
          ∧ (meanVote O F p ≤ B.lo → μ.real {ω | B.hi < voteCount O.mq F p ω} ≤ crossLimit)) ∧
        1 - δ - α ≤ (runMeasure μ D Dsf).real
          {x | (∃ B : {B : State // B ∈ states},
                x ∈ ret O.mq populations uni η₀ indecisionLimit εcov α B.val)
            ∧ ∀ B : {B : State // B ∈ states},
              x ∈ ret O.mq populations uni η₀ indecisionLimit εcov α B.val →
              ∀ j ∈ populations,
                misShareOf η₀ O.mq (D j) (clusterAt O.mq populations x B.val) (oracleNoise x)
                  ≤ missLimit uni j εcov
                ∧ (D j).real {p | ¬ decided O.mq B.val.lo (B.val.hi + 1)
                    (familyAt O.mq populations x B.val) p (oracleNoise x)}
                  ≤ 2 * indecisionLimit}

end OrthoDFA
