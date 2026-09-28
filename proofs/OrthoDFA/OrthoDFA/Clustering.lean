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
(`transition_resolver`, `counterexample_synthesis`); here it is a field of `State`
(`cn/cd`).  So the signal is the worse rate's margin `½ − max(ηIn, ηOut)` rather than the
half-gap `(1 − ηIn − ηOut)/2`.

Known modelling gap.  The Python reads the family with a calibrated band around the
boundary; here the band is one count.
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
  k : ℕ
  /-- `identify_cluster_around`'s `decision_boundary`, as the ratio `cn/cd`: `p` is on the
  accept side when `cn · #F < cd · voteCount F p`. -/
  cn : ℕ
  cd : ℕ
  /-- Reject at or below this count. -/
  lo : ℕ
  /-- Accept above this count. -/
  hi : ℕ
  /-- `_screen_cohort`'s cutoff, as the ratio `sc/scd`: a candidate disagreeing with the seed's
  column on more than this fraction of prefixes above the pool's floor never reaches
  `fully_observed()`, so `identify_cluster_around` never clusters over it. -/
  sc : ℕ
  scd : ℕ
  /-- The gate skips a sample smaller than this. -/
  gmin : ℕ

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

noncomputable def clusterAt (mq : S → Ω → ℝ) (populations : Finset J)
    (x : Run Ω S J) (B : State) : Finset S :=
  clusterAround mq B.cn B.cd (prefixesAt populations B.npref x) (screenedAt mq populations B x)
    (oracleNoise x) B.k

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
  (P.filter (fun p => hi - 1 < voteCount mq F p ω),
    P.filter (fun p => hi - 1 < voteCount mq F p ω ∨ voteCount mq F p ω ≤ lo))

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
agrees with the seed's read significantly more often than a coin flip, skipped below `n₀`
decided prefixes.  Unlike `drift_verdict`, which holds each side to the family's thresholds,
the two sides are pooled. -/
def admitted (mq : S → Ω → ℝ) (lo hi n₀ : ℕ) (α : ℝ) (F P : Finset S) (ω : Ω) : Prop :=
  n₀ ≤ (agreeCount mq lo hi F P ω).2 →
    binomSfGe (agreeCount mq lo hi F P ω).2 (1 / 2) (agreeCount mq lo hi F P ω).1 ≤ α

open scoped Classical in
/-- `judge_family`: a family smaller than the round asked for is not used; otherwise the FNR
gate and the accept-preserving gate, each per population on `certOf`.  Both read the vote with
the seed dropped, since `ε` would put `mq p` itself in it. -/
noncomputable def ret (mq : S → Ω → ℝ) (populations : Finset J)
    (indecisionLimit α : ℝ) (B : State) : Set (Run Ω S J) :=
  {x | B.k ≤ (clusterAt mq populations x B).card
    ∧ (∀ j ∈ populations,
      (((certOf j B.npref x).filter (fun p => ¬ decided mq B.lo (B.hi - 1)
          ((clusterAt mq populations x B).erase 1) p (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * (certOf j B.npref x).card)
    ∧ ∀ j ∈ populations, admitted mq B.lo B.hi B.gmin α
        ((clusterAt mq populations x B).erase 1) (certOf j B.npref x) (oracleNoise x)}

/-! ## What the input distributions must satisfy -/

/-- Two prefixes of a flat set never extend to the same query string.

`UniformSampler(DEFAULT_SAMPLE_LENGTH)` draws every probe at one fixed length, so
`p * v = p' * v'` forces `p = p'` on length alone.  Stated this way because `Monoid` on its
own does not know that strings factor. -/
def Flat (Pre : Set S) : Prop :=
  ∀ p ∈ Pre, ∀ p' ∈ Pre, ∀ v v' : S, p * v = p' * v' → p = p'

/-- The chance two independent draws coincide, `∑ₐ D({a})²`.  The oracle is persistent, so
draws carry independent noise only where they are distinct; `S` is countable, so this is never
zero and the claim asks only that it be small. -/
noncomputable def collisionMass (Dj : Measure S) : ℝ := ∑' a : S, (Dj.real {a}) ^ 2

/-! ## The theorem -/

/-- The E-L\* clustering algorithm is correct at a polynomial cost.  With probability
`≥ 1 − δ` the loop stops at one of `states`, and the family it returns there cuts `≥ 1 − εcov`
of each population the way the noiseless oracle does and leaves at most `2·indecisionLimit`
of it undecided; no state draws more prefixes than the count below, for one constant `k`
across every input.

The algorithm is told only an upper bound `η₀` on the noise rate.  `pAP` lower-bounds the share
of suffixes that preserve membership for every prefix, and `cap` bounds the collision mass. -/
def ClusteringGuarantee : Prop :=
  ∃ k : ℝ,
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {J : Type*} [Fintype J]
    (O : Oracle μ S) (populations : Finset J) (Pre : Set S)
    (η₀ indecisionLimit εcov α δ pAP : ℝ),
  O.η ≤ η₀ →
  η₀ < 1 / 2 →
  populations.Nonempty →
  Flat Pre →
  0 < pAP →
  0 < indecisionLimit →
  indecisionLimit ≤ 1 / 2 →
  0 < α →
  α < 1 / 2 →
  0 < εcov →
  εcov ≤ 1 →
  0 < δ →
  δ ≤ 1 →
  ∃ cap : ℝ,
    0 < cap ∧
    ∀ (D : J → Measure S) (Dsf : Measure S),
      (∀ j, IsProbabilityMeasure (D j)) → IsProbabilityMeasure Dsf →
      (∀ j ∈ populations, D j Preᶜ = 0) →
      pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
      ∀ ρ : ℝ,
      (∀ j ∈ populations, collisionMass (D j) ≤ ρ) →
      ρ ≤ cap →
      collisionMass Dsf ≤ cap →
      ∃ states : Finset State,
        (∀ B ∈ states, (B.npref : ℝ) ≤
          k
          * (populations.card : ℝ) ^ 2
          * Real.log (
            ((populations.card : ℝ) + 2)
            / (δ * α * pAP * min εcov (min (1 / 2 - η₀) indecisionLimit))
          )
          / ((1 / 2 - η₀) ^ 6 * min εcov (min (1 / 2 - η₀) indecisionLimit) ^ 3)
        ) ∧
        1 - δ ≤ (runMeasure μ D Dsf).real
          {x | (∃ B : {B : State // B ∈ states},
                x ∈ ret O.mq populations indecisionLimit α B.val)
            ∧ ∀ B : {B : State // B ∈ states},
              x ∈ ret O.mq populations indecisionLimit α B.val →
              ∀ j ∈ populations, 1 - εcov
                ≤ (D j).real {p | cutCorrect O B.val.lo B.val.hi
                    (clusterAt O.mq populations x B.val) p (oracleNoise x)}
                ∧ (D j).real {p | ¬ decided O.mq B.val.lo (B.val.hi - 1)
                    ((clusterAt O.mq populations x B.val).erase 1) p (oracleNoise x)}
                  ≤ 2 * indecisionLimit}

end OrthoDFA
