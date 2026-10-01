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
family's vote there; here the cut is centred at half the family (`lo`, `hi`).  So the signal is
the worse rate's margin `½ − max(ηIn, ηOut)` rather than the half-gap `(1 − ηIn − ηOut)/2`.

Known modelling gap.  `identify_cluster_around` scores a candidate by its worst population's
share of `hammingLoss`, stops once the total loss stops falling, and recentres at the boundary;
`clusterAround` does none of these.  The proof only uses that the cluster holds the seed, lies in the screened pool, has
`k` members when the pool holds that many and at most `k` otherwise, and is a function of the
pool's reads on the prefixes, all of which `identify_cluster_around` also satisfies.

Known modelling gap.  The Python sizes the family and its band from the boundary it estimates
(`smallest_readable_family`, `readable_size_and_margin`); here the family is `famCount + 1` and
the band is sized so that a vote whose mean lies outside it lands on its far side at most
`crossLimit` of the time.  The returned family, seed included, is cut at `lo` and `hi + 1`, wider
by the seed's one read than the gate's cut over the family without it, where the Python cuts both
at one rate.

Known modelling gap.  The gate here reads the uniform pool's first `n + e` draws, for any `e`;
the Python reads it on `n`, then once more on as many as `prefixes_to_certify` asks for.

Known modelling gap.  `judge_family` reads the FNR on the table's own prefixes, the ones
`identify_cluster_around` clustered the family on; `ret` reads it on the certification sample.
The table's votes are fitted to its noise and read as more decisive than they are, so the
claim's bound on the undecided mass holds of the test `ret` runs, not of the Python's.

Known modelling gap.  `drift_verdict` also lets the state populations veto a family, testing
each side at `α/num_tests` on `veto_size` draws; `ret` has no veto.  The Python's FNR also reads
1 for a family that decides no prefix one of the two ways, which `ret` does not.

Known modelling gap.  `_screen_cohort` screens each cohort once, when it is drawn, against the
table as it then stands, by a staircase of binomial tests against a floor fitted to the cohort;
`screened` screens the whole pool at the state's prefix count, against a fixed cutoff above the
pool's least count.

Known modelling gap.  The states here are a ladder of prefix counts halving from `prefCount` at
one pool size, and the claim covers a stop only at a rung of at least `validCount` prefixes.  The
Python grows the table as it goes, by `num_addtl_prefixes` prefixes or a cohort of suffixes after
each refusal, and stops at whatever table a round first passes on, which the claim need not
cover.  It also gives up after `ACCEPT_PRESERVING_GIVE_UP` refusals by the gate, where the loop
here never does.
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
`masks[cluster].mean(0) > decision_boundary`, the boundary written `cn/cd`.
`identify_cluster_around` takes it per population and scores the worst share. -/
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
candidates, but only while the seed is among them -- `identify_cluster_around` breaks out
(`if seed_local not in nearest`).  The seed wins ties for the `k`-th place, as it does under the
stable `argsort`. -/
noncomputable def lloydStep (mq : S → Ω → ℝ) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (F : Finset S) : Finset S :=
  if ∀ w ∈ cands, w ∉ insert (1 : S)
        (leastLossSubset (clusterLoss mq F cn cd P cands ω) (cands.erase 1) (k - 1)) →
      ∀ v ∈ insert (1 : S)
        (leastLossSubset (clusterLoss mq F cn cd P cands ω) (cands.erase 1) (k - 1)),
      clusterLoss mq F cn cd P cands ω v ≤ clusterLoss mq F cn cd P cands ω w
  then insert (1 : S)
    (leastLossSubset (clusterLoss mq F cn cd P cands ω) (cands.erase 1) (k - 1)) else F

/-- `identify_cluster_around` iterated to its fixed point, which `k·#P + 1` steps reach: the
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

/-- The prefixes the cut accepts, and all the prefixes it decides. -/
noncomputable def cutSides (mq : S → Ω → ℝ) (lo hi : ℕ) (F P : Finset S) (ω : Ω) :
    Finset S × Finset S :=
  (P.filter (fun p => hi < voteCount mq F p ω),
    P.filter (fun p => hi < voteCount mq F p ω ∨ voteCount mq F p ω ≤ lo))

/-- The accept side's hits plus the reject side's misses. -/
noncomputable def agreeOf (A Dset U : Finset S) : ℕ :=
  (A ∩ U).card + ((Dset \ A) \ U).card

/-- The gate's statistic, `(agreements, decided count)`: `p` agrees when the seed's column
reads the way the cut calls it. -/
noncomputable def agreeCount (mq : S → Ω → ℝ) (lo hi : ℕ) (F P : Finset S) (ω : Ω) : ℕ × ℕ :=
  (agreeOf (cutSides mq lo hi F P ω).1 (cutSides mq lo hi F P ω).2
      (P.filter (fun p => mq p ω = 1)),
    (cutSides mq lo hi F P ω).2.card)

/-- `drift_verdict`'s ADMITTED at error rate `α` (`ACCEPT_PRESERVING_ERROR_RATE`): the cut
decides some prefix, and agrees with the seed's read significantly more often than it would if
every read came back 1 at the decision boundary, which is `1/2` for this oracle. -/
def admitted (mq : S → Ω → ℝ) (lo hi : ℕ) (α : ℝ) (F P : Finset S) (ω : Ω) : Prop :=
  0 < (agreeCount mq lo hi F P ω).2
    ∧ binomSfGe (agreeCount mq lo hi F P ω).2 (1 / 2) (agreeCount mq lo hi F P ω).1 ≤ α

/-- `certification_budget`: both tests read at most the prefixes one round of table growth, `a`
prefixes over the whole pool, would cost in reads of the family, and never more than the table
holds of a population. -/
def certSize (a : ℕ) (B : State) : ℕ := min B.npref (max 1 (a * B.nsuff / B.k))

open scoped Classical in
/-- `judge_family`: a family smaller than the round asked for is not used; otherwise the FNR
test per population on `n` fresh draws, and the accept-preserving gate on as many from the
uniform pool `uni`, which may draw that pool further when the split reads uncertified.  The FNR
reads the family with its seed; the gate reads it without, since the seed's read is the bit the
gate scores. -/
noncomputable def ret (mq : S → Ω → ℝ) (populations : Finset J) (uni : J)
    (indecisionLimit α : ℝ) (n : ℕ) (B : State) : Set (Run Ω S J) :=
  {x | B.k ≤ (clusterAt mq populations x B).card + 1
    ∧ (∀ j ∈ populations,
      (((certOf j n x).filter (fun p => ¬ decided mq B.lo (B.hi + 1)
          (familyAt mq populations x B) p (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * (certOf j n x).card)
    ∧ ∃ e : ℕ, admitted mq B.lo B.hi α
        (clusterAt mq populations x B) (certOf uni (n + e) x) (oracleNoise x)}

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

/-- The prefix count `ClusteringGuarantee` allows a state, at the largest pool it allows. -/
noncomputable def prefixNeed (populations : Finset J)
    (η₀ indecisionLimit εcov δ α pAP crossLimit : ℝ) : ℝ :=
  2048 * (populations.card : ℝ) ^ 2
    * Real.log (((populations.card : ℝ) + 2)
      * (128 * Real.log (2 / (min εcov (min (1 / 2 - η₀) indecisionLimit) * crossLimit))
          / ((1 / 2 - η₀) ^ 2 * pAP)
        + 16 * Real.log (((populations.card : ℝ) + 2) / δ) / pAP ^ 2)
      / (δ * α * pAP * min εcov (min (1 / 2 - η₀) indecisionLimit)))
    / ((1 / 2 - η₀) ^ 6 * min εcov (min (1 / 2 - η₀) indecisionLimit) ^ 3)

/-! ## The theorem -/

/-- The E-L\* clustering algorithm is correct at a polynomial cost.  With probability
`≥ 1 − δ` the loop stops at one of `states`, and the family it returns there cuts `≥ 1 − εcov`
of each population the way the noiseless oracle does and leaves at most `2·indecisionLimit`
of it undecided; no state draws more prefixes than the first count below, nor asks for a family
or a suffix pool larger than the sizes below.  Some
state draws no more than the second count, which is what validity alone costs, so a stop is
covered from there; the first count is what the loop may need before a round passes.  Every
state's band is wide enough that a vote over a family no larger than its own, whose mean lies
above the band, lands at or below `lo`, or one whose mean lies at or below `lo` lands above `hi`,
at most `crossLimit` of the time.

The algorithm is told only an upper bound `η₀` on the noise rate.  `uni` is the uniform pool,
the one population the gate admits on.  `pAP` lower-bounds the share of suffixes that preserve
membership for every prefix, and `cap` bounds the collision mass. -/
def ClusteringGuarantee : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {J : Type*} [Fintype J]
    (O : Oracle μ S) (populations : Finset J) (uni : J) (Pre Suf : Set S)
    (η₀ indecisionLimit εcov α δ pAP crossLimit : ℝ) (a : ℕ),
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
  -- one round of table growth, `a` prefixes, buys a certification sample as large as any state's
  prefixNeed populations η₀ indecisionLimit εcov δ α pAP crossLimit ≤ 2 * a / pAP →
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
          2048
          * (populations.card : ℝ) ^ 2
          * Real.log (
            ((populations.card : ℝ) + 2) * B.nsuff
            / (δ * α * pAP * min εcov (min (1 / 2 - η₀) indecisionLimit))
          )
          / ((1 / 2 - η₀) ^ 6 * min εcov (min (1 / 2 - η₀) indecisionLimit) ^ 3)
        ) ∧
        (∃ B ∈ states, (B.npref : ℝ) ≤
          32
          * (populations.card : ℝ) ^ 2
          * Real.log (((populations.card : ℝ) + 2) * ((B.nsuff : ℝ) + 2) / δ)
          / ((1 / 2 - η₀) ^ 6 * min εcov indecisionLimit ^ 2)
        ) ∧
        (∀ B ∈ states,
          (B.k : ℝ) ≤ 64 * Real.log (2 / (min εcov (min (1 / 2 - η₀) indecisionLimit) * crossLimit))
            / (1 / 2 - η₀) ^ 2
          ∧ (B.nsuff : ℝ) ≤ 128 * Real.log
                (2 / (min εcov (min (1 / 2 - η₀) indecisionLimit) * crossLimit))
              / ((1 / 2 - η₀) ^ 2 * pAP)
            + 16 * Real.log (((populations.card : ℝ) + 2) / δ) / pAP ^ 2) ∧
        (∀ B ∈ states, ∀ F : Finset S, F.card + 1 ≤ B.k → ∀ p,
          (B.hi < meanVote O F p → μ.real {ω | voteCount O.mq F p ω ≤ B.lo} ≤ crossLimit)
          ∧ (meanVote O F p ≤ B.lo → μ.real {ω | B.hi < voteCount O.mq F p ω} ≤ crossLimit)) ∧
        1 - δ ≤ (runMeasure μ D Dsf).real
          {x | (∃ B : {B : State // B ∈ states},
                x ∈ ret O.mq populations uni indecisionLimit α (certSize a B.val) B.val)
            ∧ ∀ B : {B : State // B ∈ states},
              x ∈ ret O.mq populations uni indecisionLimit α (certSize a B.val) B.val →
              ∀ j ∈ populations, 1 - εcov
                ≤ (D j).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
                    (familyAt O.mq populations x B.val) p (oracleNoise x)}
                ∧ (D j).real {p | ¬ decided O.mq B.val.lo (B.val.hi + 1)
                    (familyAt O.mq populations x B.val) p (oracleNoise x)}
                  ≤ 2 * indecisionLimit}

end OrthoDFA
