import Mathlib.Tactic
import Mathlib.Probability.ProductMeasure
import Mathlib.Probability.Independence.InfinitePi

/-!
# The model: the oracle, the algorithm, and the theorem

Everything `ClusteringGuarantee` mentions is defined here, so auditing the claim means reading
this file and no other.  Each definition's doc names the Python it models.

Known modelling gap.  `prefixes_for_split` draws the certification prefixes with `draw_many`,
which on a `UniformSource` redraws neither a repeat nor a prefix the table holds, and a
`RejectionSource` redraws only what it has served itself, not another population's prefixes.
`certOf` redraws both.  The pool's draws merge with the seed `ε` when one is `ε`, where
`_draw_cohort` would redraw it.

Known modelling gap.  For a named population, `split_draws` sends each draw to a held-out
reserve with chance `RESERVE_SHARE` and `prefixes_for_split` certifies on that reserve first, in
draw order, then on fresh draws; `take` puts a round's harvested strings in the table.  Here
every population's certification sample is drawn from a stream of its own, outside the tables,
and no population holds harvested strings.

Known modelling gap.  `grow_population` retires a population whose source fails its yield
test, and `prefixes_for_split` reads a split without a population whose source draws nothing;
`keptAt` drops a population from the tests only when its table or its certification sample
cannot be filled at all.

Known modelling gap.  The Python re-estimates `pst.decision_boundary` from its reads
(`transition_resolver`, `counterexample_synthesis`, `identify_cluster_around`) and cuts the
family's vote there; here the cut is centred at half the family (`lo`, `hi`).  So the signal is
the worse rate's margin `½ − max(ηIn, ηOut)` rather than the half-gap `(1 − ηIn − ηOut)/2`.

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

Known modelling gap.  The FNR test here reads `npref` draws from each population, and the gate
reads each population's first `npref + e`, for any `e ≤ npref`.  The Python reads the gate on
`min(alignment_size, certification_budget)` draws from every population, then once more on as
many as `prefixes_to_certify` asks for from the one left undecided.

Known modelling gap.  `judge_family` reads the FNR on the table's own prefixes, the ones
`identify_cluster_around` clustered the family on; `ret` reads it on the certification sample.
The table's votes are fitted to its noise and read as more decisive than they are, so the
claim's bound on the undecided mass holds of the test `ret` runs, not of the Python's.

Known modelling gap.  `drift_verdict` admits a family when `misclassified_bounds` bounds the
share of the uniform pool's distribution its cut at the boundary misclassifies by
`max_coverage_error`, and every other population's by a half: Clopper-Pearson intervals on each
side's share and on the seed's rate there, read at the rates `decision_boundary ∓
min_signal_strength`, which it takes to be the oracle's.  `certifiesAt` bounds each side's wrong
share through the side's spread against references outside the family, at the worst rate `η₀`,
so it is sound whatever the rates are; the claim covers that test and not the Python's.  The
Python's FNR also reads 1 for a family that decides no prefix one of the two ways, which `ret`
does not.

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

/-! ## The algorithm

`sample_suffix_family` in the order it runs: draw, screen, cluster, read the cut off the
family's vote. -/

/-! ### Draw -/

open scoped Classical in
/-- The first `m` distinct draws of `y` outside `T`, sampling without replacement: a draw
already taken, or in `T`, is drawn again, as `_draw_cohort` and `_distinct_prefixes` do.  Empty
when `y` never offers `m` of them. -/
noncomputable def freshDraws (T : Finset S) (y : ℕ → S) (m : ℕ) : Finset S :=
  if h : ∃ n, m ≤ ((Finset.range n).image y \ T).card then
    (Finset.range (Nat.find h)).image y \ T
  else ∅

/-- `M` suffixes drawn without replacement, with the seed `ε`: `pst.table.column(v)` promotes
it to fully observed at the top of every round. -/
noncomputable def poolAt (M : ℕ) (x : Run Ω S J) : Finset S :=
  insert 1 (freshDraws ∅ (fun i => suffixDraw i x) M)

/-- Population `j`'s `m` prefixes, drawn without replacement. -/
noncomputable def prefixesOf (j : J) (m : ℕ) (x : Run Ω S J) : Finset S :=
  freshDraws ∅ (fun i => prefixDraw j i x) m

/-- Every population's representative prefixes, pooled. -/
noncomputable def prefixesAt (populations : Finset J) (m : ℕ)
    (x : Run Ω S J) : Finset S :=
  populations.biUnion (fun j => prefixesOf j m x)

/-- Population `j`'s `m` `certification_sample` prefixes, drawn without replacement and outside
the table `T`, so the family was never clustered on them. -/
noncomputable def certOf (T : Finset S) (j : J) (m : ℕ) (x : Run Ω S J) : Finset S :=
  freshDraws T (fun i => certPrefix j i x) m

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

`judge_family`'s two tests. -/

/-- `P[Bin(N,p) ≥ j]` — `scipy.stats.binom.sf(j-1, N, p)`. -/
noncomputable def binomSfGe (N : ℕ) (p : ℝ) (j : ℕ) : ℝ :=
  ∑ i ∈ Finset.Icc j N, (N.choose i : ℝ) * p ^ i * (1 - p) ^ (N - i)

/-- Hoeffding's deviation for a sum of `n` independent reads in `[0, 1]`, exceeded with chance at
most `a`. -/
noncomputable def hoeffDev (n : ℕ) (a : ℝ) : ℝ := Real.sqrt (n * Real.log (1 / a) / 2)

open scoped Classical in
/-- The fewest prefixes of `A` on which any reference's read of `p · v` and the seed's read of
`p` disagree. -/
noncomputable def refApart (mq : S → Ω → ℝ) (A R : Finset S) (ω : Ω) : ℕ :=
  if h : R.Nonempty then
    R.inf' h (fun v => (A.filter (fun p => ¬ ((mq (p * v) ω = 1) ↔ (mq p ω = 1)))).card)
  else 0

/-- The level a test at sample size `n` spends out of `a`, so that every size together spends at
most `a`. -/
noncomputable def sizeLevel (a : ℝ) (n : ℕ) : ℝ := a / (((n : ℝ) + 1) * ((n : ℝ) + 2))

/-- The prefixes the family's cut accepts. -/
noncomputable def acceptSide (mq : S → Ω → ℝ) (hi : ℕ) (F P : Finset S) (ω : Ω) : Finset S :=
  P.filter (fun p => hi < voteCount mq F p ω)

/-- The prefixes the family's cut rejects. -/
noncomputable def rejectSide (mq : S → Ω → ℝ) (lo : ℕ) (F P : Finset S) (ω : Ω) : Finset S :=
  P.filter (fun p => voteCount mq F p ω ≤ lo)

/-- How much of the spread in a side's reads its noise leaves unexplained, at centre `c`.  Of the
`n` prefixes on the side, `against` read on the seed against the side and `apart` is the fewest
any reference disagrees with the seed on, each within `d` of its mean; with `t_p` the chance
`p` reads against the side and `r_p` its noise rate,

    (1/n) ∑ (t_p − c)² = c(1 − c) + (1 − 2c)(t̄ − c) − (1/n) ∑ r_p(1 − r_p)

and the reference disagrees at `2r_p(1 − r_p)`. -/
noncomputable def spreadBound (n against apart : ℕ) (d c : ℝ) : ℝ :=
  c * (1 - c) + max ((1 - 2 * c) * ((against - d) / n - c)) ((1 - 2 * c) * ((against + d) / n - c))
    - (apart - d) / (2 * n)

/-- An upper bound on the fraction of a side the cut puts there wrongly, given that the wrong
class's noise rate is at most `y`.  A wrong prefix reads against the side at `1 − y` or more,
so it contributes `(1 − y − c)²` to the spread. -/
noncomputable def wrongCap (n against apart : ℕ) (d y : ℝ) : ℝ :=
  if 0 < n ∧ 0 < 1 - y - (against : ℝ) / n then
    min 1 (max 0 (spreadBound n against apart d ((against : ℝ) / n))
      / (1 - y - (against : ℝ) / n) ^ 2)
  else 1

/-- How many of a side the seed reads against it: `0` on the accept side, `1` on the reject
side. -/
noncomputable def againstOn (mq : S → Ω → ℝ) (accept : Bool) (A : Finset S) (ω : Ω) : ℕ :=
  (A.filter (fun p => (mq p ω = 1) ↔ accept = false)).card

/-- `wrongCap` on one side of the cut, against references `R` at level `a`. -/
noncomputable def sideCap (mq : S → Ω → ℝ) (accept : Bool) (A R : Finset S) (a y : ℝ) (ω : Ω) :
    ℝ :=
  wrongCap A.card (againstOn mq accept A ω) (refApart mq A R ω) (hoeffDev A.card a) y

/-- An upper bound on how many of `P` the cut puts on the wrong side: the accept side's wrong
prefixes lie outside the language and the reject side's inside it, each class's rate at most
`η₀`. -/
noncomputable def miscutBound (mq : S → Ω → ℝ) (a : ℝ) (lo hi : ℕ) (F R P : Finset S) (η₀ : ℝ)
    (ω : Ω) : ℝ :=
  (acceptSide mq hi F P ω).card * sideCap mq true (acceptSide mq hi F P ω) R a η₀ ω
    + (rejectSide mq lo F P ω).card * sideCap mq false (rejectSide mq lo F P ω) R a η₀ ω

/-- `drift_verdict`'s ADMITTED, as modelled: the family's cut misclassifies at most `lim` of the
population.  `P` is the sample, drawn without replacement, and `R` the references; `miscutBound`
and the passage from the sample to the population are both read at `sizeLevel a #P`. -/
def certifiesAt (mq : S → Ω → ℝ) (η₀ lim a : ℝ) (lo hi : ℕ) (F R P : Finset S) (ω : Ω) :
    Prop :=
  0 < P.card ∧ miscutBound mq (sizeLevel a P.card) lo hi F R P η₀ ω
    ≤ P.card * lim - hoeffDev P.card (sizeLevel a P.card)

/-- Population `j` is read at a state of `m` prefixes: its table and its certification sample
both find their `m` strings. -/
def keptAt (populations : Finset J) (j : J) (m : ℕ) (x : Run Ω S J) : Prop :=
  (prefixesOf j m x).card = m ∧ (certOf (prefixesAt populations m x) j m x).card = m

open scoped Classical in
/-- The pool's suffixes outside the family: the gate's references. -/
noncomputable def refsAt (mq : S → Ω → ℝ) (populations : Finset J) (x : Run Ω S J) (B : State) :
    Finset S :=
  poolAt B.nsuff x \ familyAt mq populations x B

open scoped Classical in
/-- `misclassification_limit`: `εcov` on the uniform pool, a half elsewhere. -/
noncomputable def gateLimit (uni : J) (εcov : ℝ) (j : J) : ℝ := if j = uni then εcov else 1 / 2

/-- The level the gate spends at a state: `α` over the gate's seven tails per population,
and over the states by their prefix counts. -/
noncomputable def gateLevel (α : ℝ) (populations : Finset J) (B : State) : ℝ :=
  sizeLevel (α / (7 * populations.card)) B.npref

open scoped Classical in
/-- `judge_family`: a family smaller than the round asked for is not used; otherwise the FNR
test on `certOf`, and the gate, at each population `keptAt` keeps, the gate drawing that
population further when the split reads uncertified, to at most twice its size.  The FNR reads
the family with its seed; the gate reads it without, since the seed's read is the bit the gate
scores. -/
noncomputable def ret (mq : S → Ω → ℝ) (populations : Finset J) (uni : J)
    (η₀ indecisionLimit εcov α : ℝ) (B : State) : Set (Run Ω S J) :=
  {x | B.k ≤ (clusterAt mq populations x B).card + 1
    ∧ (∀ j ∈ populations, keptAt populations j B.npref x →
      (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
          ¬ decided mq B.lo (B.hi + 1) (familyAt mq populations x B) p
            (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * (certOf (prefixesAt populations B.npref x) j B.npref x).card)
    ∧ ∀ j ∈ populations, keptAt populations j B.npref x → ∃ e ≤ B.npref,
      certifiesAt mq η₀ (gateLimit uni εcov j) (gateLevel α populations B) B.lo B.hi
        (clusterAt mq populations x B) (refsAt mq populations x B)
        (certOf (prefixesAt populations B.npref x) j (B.npref + e) x) (oracleNoise x)}

/-! ## What the input distributions must satisfy -/

/-- Two prefixes never extend, by drawn suffixes or by `ε`, to the same query string.

`sample_more_suffixes` draws every suffix at the sampler's length `L`, so `p * v = p' * v'`
with both suffixes drawn forces `p = p'` on length alone.  Against the seed `ε` it would need
`p` to be longer than `p'` by `L`, which prefixes whose lengths lie within a window narrower than
`L` never are.  Stated this way because `Monoid` on its own does not know that strings factor. -/
def Flat (Pre Suf : Set S) : Prop :=
  ∀ p ∈ Pre, ∀ p' ∈ Pre, ∀ v ∈ insert 1 Suf, ∀ v' ∈ insert 1 Suf, p * v = p' * v' → p = p'

/-- The chance two independent draws coincide, `∑ₐ D({a})²`: `1/h` for a population uniform
over `h` strings.  Drawing without replacement redraws a repeat, and the claim asks that a state's
draws redraw only a small fraction of the time. -/
noncomputable def collisionMass (Dj : Measure S) : ℝ := ∑' a : S, (Dj.real {a}) ^ 2

/-! ## The theorem -/

/-- The rate the counts resolve:

    min(εcov·a², indecisionLimit),   a = (1 − 2η₀)/(1 − η₀)

The gate charges each wrongly cut prefix at most `7/(3a²)` of a read, so the share a family may
cut wrongly is held at a multiple of `εcov·a²`. -/
noncomputable def rateScale (η₀ indecisionLimit εcov : ℝ) : ℝ :=
  min (εcov * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) indecisionLimit

/-- The E-L\* clustering algorithm is correct at a polynomial cost.  With probability
`≥ 1 − δ − α` the loop stops at one of `states` when every population is tame, and whenever it
stops, the family it returns there cuts `≥ 1 − (513/512)·εcov` of the uniform pool the way the
noiseless oracle does, reads no tame population backwards (cuts at least `1/2 − εcov/512` of each
that way), and leaves at most `2·indecisionLimit` of each tame population undecided; no state
draws more prefixes than the first count below, nor asks for a family or a suffix pool larger
than the sizes below.  Some state draws no more than the second count, which
is what validity alone costs, so a stop is covered from there; the first count is what the loop
may need before a round passes.  Every state's band is wide enough that a vote over a family no
larger than its own, whose mean lies above the band, lands at or below `lo`, or one whose mean
lies at or below `lo` lands above `hi`, at most `crossLimit` of the time.

The algorithm is told only an upper bound `η₀` on the noise rate.  `uni` is the uniform pool,
the one population every prefix of the learner's distribution is drawn from.  `pAP` lower-bounds
the share of suffixes that preserve membership for every prefix.  `α` is what the gate spends:
the chance it passes a family wrong on more than its limit, over every state.  The states depend
on none of the distributions.  A tame population need only have no string heavy against the
prefixes a state draws, and the suffix measure a collision mass small against the suffixes;
the other populations are arbitrary.  A population is read only while `keptAt` keeps it. -/
def ClusteringGuarantee : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {J : Type*} [Fintype J]
    (O : Oracle μ S) (populations tame : Finset J) (uni : J) (Pre Suf : Set S)
    (η₀ indecisionLimit εcov α δ pAP crossLimit : ℝ),
  O.η ≤ η₀ →
  η₀ < 1 / 2 →
  uni ∈ tame →
  tame ⊆ populations →
  Flat Pre Suf →
  0 < pAP →
  0 < indecisionLimit →
  indecisionLimit ≤ 1 / 2 →
  0 < α →
  α < 1 / 2 →
  0 < εcov →
  εcov ≤ 1 / 2 →
  0 < δ →
  δ ≤ 1 →
  0 < crossLimit →
  crossLimit ≤ 1 →
  ∃ states : Finset State,
    (∀ B ∈ states, (B.npref : ℝ) ≤
      60000
      * (populations.card : ℝ) ^ 2
      * Real.log (
        ((populations.card : ℝ) + 2) * B.nsuff
        / (δ * α * pAP * rateScale η₀ indecisionLimit εcov)
      )
      / ((1 / 2 - η₀) ^ 4 * rateScale η₀ indecisionLimit εcov ^ 2)
    ) ∧
    (∃ B ∈ states, (B.npref : ℝ) ≤
      5 * Real.log (128 * (populations.card : ℝ) / δ) / indecisionLimit ^ 2
    ) ∧
    (∀ B ∈ states,
      (B.k : ℝ) ≤ 64 * Real.log (2 / (rateScale η₀ indecisionLimit εcov * crossLimit))
          / (1 / 2 - η₀) ^ 2
      ∧ (B.nsuff : ℝ) ≤ 128 * Real.log (2 / (rateScale η₀ indecisionLimit εcov * crossLimit))
          / ((1 / 2 - η₀) ^ 2 * pAP)
        + 16 * Real.log (((populations.card : ℝ) + 2) / δ) / pAP ^ 2) ∧
    (∀ B ∈ states, ∀ F : Finset S, F.card + 1 ≤ B.k → ∀ p,
      (B.hi < meanVote O F p → μ.real {ω | voteCount O.mq F p ω ≤ B.lo} ≤ crossLimit)
      ∧ (meanVote O F p ≤ B.lo → μ.real {ω | B.hi < voteCount O.mq F p ω} ≤ crossLimit)) ∧
    ∀ (D : J → Measure S) (Dsf : Measure S),
      (∀ j, IsProbabilityMeasure (D j)) → IsProbabilityMeasure Dsf →
      (∀ j ∈ populations, D j Preᶜ = 0) →
      Dsf Sufᶜ = 0 →
      pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
      (∀ B ∈ states, ∀ j ∈ tame, ∀ t : S,
        2 ^ 20 * ((populations.card : ℝ) + 1) ^ 2 * (B.npref : ℝ) * (D j).real {t}
          ≤ δ * rateScale η₀ indecisionLimit εcov ^ 2) →
      (∀ B ∈ states,
        2 ^ 12 * ((populations.card : ℝ) + 1) * (B.nsuff : ℝ) * collisionMass Dsf ≤ δ * pAP) →
      1 - δ - α ≤ (runMeasure μ D Dsf).real
        {x | ((∀ j ∈ populations, j ∈ tame) → ∃ B : {B : State // B ∈ states},
              x ∈ ret O.mq populations uni η₀ indecisionLimit εcov α B.val)
          ∧ ∀ B : {B : State // B ∈ states},
            x ∈ ret O.mq populations uni η₀ indecisionLimit εcov α B.val →
            1 - 513 / 512 * εcov
              ≤ (D uni).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
                  (familyAt O.mq populations x B.val) p (oracleNoise x)}
            ∧ ∀ j ∈ tame, 1 / 2 - εcov / 512
              ≤ (D j).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
                  (familyAt O.mq populations x B.val) p (oracleNoise x)}
              ∧ (D j).real {p | ¬ decided O.mq B.val.lo (B.val.hi + 1)
                  (familyAt O.mq populations x B.val) p (oracleNoise x)}
                ≤ 2 * indecisionLimit}

end OrthoDFA
