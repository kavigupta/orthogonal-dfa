import OrthoDFA.Distributional
import OrthoDFA.BinomTail
import Mathlib.Probability.ProductMeasure
import Mathlib.Probability.Independence.InfinitePi

/-!
# The adaptive clustering loop: the integrated theorem

`sample_suffix_family` is a retry loop: cluster, measure the FNR **per population**, and
if any population is too indecisive grow the pool and try again — stopping at a
data-dependent time.  This file states and proves the guarantee **for that loop**, with
no fixed budget:

> with probability `≥ 1 − δ` the loop **terminates**, and **whatever** family it returns
> classifies `≥ 1 − εcov` of **each** prefix population the way the noiseless oracle does,
> wherever it decides at all.

The loop's state is a `Budget`: two growth budgets, a family size, a cluster centre and the
two gate cutoffs, all integers.  There is no history and no real-valued boundary — the
boundary entered every event only through the count it cut at, so the cut is the state, and
the index is countable.

"Whatever it returns" is `ret`: the states that pass both gates.  The guarantee is not
claimed at states the loop rejects.

The conclusion is about the family's **cut** (`cutCorrect`), not about each of its members
being accept-preserving.  Per-member preservation is the premise the algorithm works from
(`hpAPBound`) and what its screening reaches for, but no finite test certifies it of a
sampled family, and the vote survives a drifting member.  What is certified — by the gate,
by #215's test, and by what E-L\* actually consumes downstream — is that the family
classifies prefixes correctly.

Everything the statement needs is present and constrained: the persistent RCN oracle, the
collection of prefix populations, the suffix distribution with its findability `pAP`, the
loop's own return test — the FNR gate *and* the accept-preserving gate, both defined
rather than abstract — and an unbounded growing schedule.
The run space is concrete (`Run`, `runLaw`), so its law is a lemma rather than a
hypothesis; the returned family is *defined* by the clustering.

Proof: the two-part decomposition —

* `validity_of_returned` — whatever is returned is valid, *whenever* it is returned,
  except w.p. `δ/2`;
* `loop_terminates` — the loop returns at some round, except w.p. `δ/2`;
* `sound_and_terminating` composes them.

Both halves are proved.  What is still `sorry` is generic mathematics, none of it about the
algorithm: four binomial facts (`lt_of_binomSfGe_le`, `lt_of_binomCdf_le`, `binomSfGe_le`,
`binomCdf_le`), one independence lemma (`iIndepFun_blocks`), and the existence of an
admissible pair of cutoffs (`exists_admissibleCut`).

Termination is stated at a state that meets `PassableAt` — the arithmetic a round has to
satisfy for its two tests to pass, which is where findability (`pAP`) and the populations'
class balance enter.  That such a state exists under a large enough cap is the loop's
growth schedule's job and is *not* proved here; it is the hypothesis `hwit`.

**Known modelling gap (flagged, not hidden).**  The draws here are i.i.d. from each
distribution and deduplicated downstream (`poolAt`, `prefixesAt`), whereas `_draw_cohort`
and `sample_more_prefixes` *redraw* on a duplicate — they sample without replacement.
Deduplicating i.i.d. draws yields a pool at most as large, so this is the conservative
model; but the collision mass grows with the number of draws, so closing
`validity_of_returned` at unbounded budgets will need either the without-replacement
concentration (Hoeffding 1963) or a non-atomic prefix distribution.  That is a gap in the
*proof*, recorded rather than papered over by weakening the claim.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [MeasurableSpace S] [Monoid S] [IsCancelMul S] [MeasurableMul S]
  [Countable S] [MeasurableSingletonClass S] [DecidableEq S]

/-! ## The derived configuration

`build_pst` does not take the family size; it *computes* it:

```python
n, eps = population_size_and_evidence_margin(
    signal_strength=min_signal_strength, acceptable_fpr=0.01, acceptable_fnr=0.01)
config = SearchConfig(suffix_family_size=n, evidence_margin=eps, ...)
```

and `min_signal_strength` is `½ − η`, already carried by the `Oracle`.  So the family
size and the evidence margin are **derived** from the oracle's signal together with the
two acceptable rates — they are not free parameters. -/

/-- Binomial CDF: `P[Bin(N,p) ≤ j]`. -/
noncomputable def binomCdf (N : ℕ) (p : ℝ) (j : ℕ) : ℝ :=
  ∑ i ∈ Finset.range (j + 1), (N.choose i : ℝ) * p ^ i * (1 - p) ^ (N - i)

/-- **Admissibility as the pair of integers it really is.**  A family of `N` suffixes votes
in `{0, 1/N, …, 1}`, so a threshold matters only through the count it cuts at, and
`admissibleMargin` already says so: `eps` occurs nowhere except inside
`⌊N(center−eps)⌋₊` and `⌈N(center+eps)⌉₊ - 1`.  `lo` and `hi` are those two counts — reject
at or below `lo`, accept at or above `hi` — and the two conditions are the binomial false
positive rate under the null and false negative rate under the signal.

Everything downstream should use this, not a chosen `eps`.  A real-valued margin forces the
proof to reason about `Classical.choose` as a function of the boundary, which has no
structure; the integer cutoffs are a bounded range, so the algorithm's configuration stays
a countable thing. -/
def admissibleCut (s fpr accFnr center : ℝ) (N lo hi : ℕ) : Prop :=
  (binomCdf N center lo + (1 - binomCdf N center hi) ≤ fpr) ∧
    (binomCdf N (s + center) hi - binomCdf N (s + center) lo ≤ accFnr)

/-- **The existence property.**  For any positive signal and any positive error budgets,
some population size admits a pair of cutoffs.

This is the whole content of `population_size_and_evidence_margin`: as `N` grows, the null
`Bin(N, center)` and the signal `Bin(N, s + center)` separate, so cutoffs exist that spend
at most `fpr` on the null's tails and at most `accFnr` on the signal's middle.  Everything
else about the configuration is a search for a witness to this. -/
theorem exists_admissibleCut (s fpr accFnr center : ℝ) (hs : 0 < s)
    (hfpr : 0 < fpr) (haccFnr : 0 < accFnr) :
    ∃ N, 0 < N ∧ ∃ lo hi, admissibleCut s fpr accFnr center N lo hi :=
  sorry

open scoped Classical in
/-- **The search terminates**: the least population size admitting a pair of cutoffs.
`population_size_and_evidence_margin` binary-searches for this; the value is whatever the
search lands on, and `cutSize_spec` is all anything downstream may use about it. -/
noncomputable def cutSize (s fpr accFnr center : ℝ) : ℕ :=
  if h : ∃ N, 0 < N ∧ ∃ lo hi, admissibleCut s fpr accFnr center N lo hi then Nat.find h else 1

theorem cutSize_spec (s fpr accFnr center : ℝ) (hs : 0 < s) (hfpr : 0 < fpr)
    (haccFnr : 0 < accFnr) :
    0 < cutSize s fpr accFnr center ∧
      ∃ lo hi, admissibleCut s fpr accFnr center (cutSize s fpr accFnr center) lo hi := by
  classical
  have h := exists_admissibleCut s fpr accFnr center hs hfpr haccFnr
  rw [cutSize, dif_pos h]
  exact Nat.find_spec h

/-- Any size the search could return is one it may: the spec is all that is used, so a
different search finding a different witness changes nothing downstream. -/
theorem admissibleCut_of_le {s fpr accFnr center : ℝ} {N lo hi : ℕ}
    (h : admissibleCut s fpr accFnr center N lo hi) :
    ∃ lo' hi', admissibleCut s fpr accFnr center N lo' hi' := ⟨lo, hi, h⟩

/-- The membership query the oracle actually answers: `MQ w = ℓ(w) ⊕ noise(w)`. -/
noncomputable def mq (O : Oracle μ S) (w : S) (ω : Ω) : ℝ :=
  O.label w + (1 - 2 * O.label w) * O.noise w ω

/-! ### The seed column

`mq O p` is the oracle's answer at `p · ε = p`: the column the gate reads its verdict off.
Across *distinct* prefixes these are independent — one noise bit per query string — which
is why `prefixesOf` being a `Finset` is what makes the gate's binomial null honest. -/

lemma mq_meas (O : Oracle μ S) (p : S) : Measurable (mq O p) := by
  show Measurable (fun ω => O.label p + (1 - 2 * O.label p) * O.noise p ω)
  exact measurable_const.add (measurable_const.mul (O.noise_meas' p))

lemma mq_indep (O : Oracle μ S) : iIndepFun (fun p : S => mq O p) μ :=
  O.noise_indep.comp (fun p x => O.label p + (1 - 2 * O.label p) * x)
    (fun _ => measurable_const.add (measurable_const.mul measurable_id))

lemma mq_bit (O : Oracle μ S) (p : S) : ∀ᵐ ω ∂μ, mq O p ω = 0 ∨ mq O p ω = 1 := by
  filter_upwards [O.noise_bit p] with ω hω
  rcases O.label_bit p with hl | hl <;> rcases hω with hn | hn <;>
    norm_num [mq, hl, hn]

lemma mq_icc (O : Oracle μ S) (p : S) : ∀ᵐ ω ∂μ, mq O p ω ∈ Set.Icc (0 : ℝ) 1 := by
  filter_upwards [mq_bit O p] with ω hω
  rcases hω with h | h <;> rw [h] <;> norm_num

/-- `E[mq p] = η + (1−2η)·ℓ(p)`: a truly-accepting prefix reads accepting with probability
`1 − η`, a truly-rejecting one with probability `η`. -/
lemma mq_mean (O : Oracle μ S) (p : S) : μ[mq O p] = O.η + (1 - 2 * O.η) * O.label p := by
  have : μ[mq O p] = ∫ ω, (O.label p + (1 - 2 * O.label p) * O.noise p ω) ∂μ := rfl
  rw [this, integral_add (integrable_const _) ((O.noise_int p).const_mul _), integral_const,
    integral_const_mul, O.noise_mean p]
  simp only [measureReal_def, measure_univ, ENNReal.toReal_one, smul_eq_mul, one_mul]
  ring

open scoped Classical in
/-- The gate's hit count *is* the sum of the column's reads: they are `0/1`. -/
lemma hits_eq_sum (O : Oracle μ S) (A : Finset S) :
    ∀ᵐ ω ∂μ, ((A.filter (fun p => mq O p ω = 1)).card : ℝ) = ∑ p ∈ A, mq O p ω := by
  filter_upwards [(ae_ball_iff A.countable_toSet).2 (fun p _ => mq_bit O p)] with ω hω
  rw [← Finset.sum_filter_add_sum_filter_not A (fun p => mq O p ω = 1)]
  have h1 : ∑ p ∈ A.filter (fun p => mq O p ω = 1), mq O p ω
      = ((A.filter (fun p => mq O p ω = 1)).card : ℝ) := by
    rw [Finset.sum_congr rfl (fun p hp => (Finset.mem_filter.mp hp).2), Finset.sum_const,
      nsmul_eq_mul, mul_one]
  have h0 : ∑ p ∈ A.filter (fun p => ¬ (mq O p ω = 1)), mq O p ω = 0 := by
    refine Finset.sum_eq_zero (fun p hp => ?_)
    obtain ⟨hpA, hne⟩ := Finset.mem_filter.mp hp
    rcases hω p hpA with h | h
    · exact h
    · exact absurd h hne
  rw [h1, h0, add_zero]

/-! ## The run space

A run is exactly what the algorithm consumes: the oracle's persistent noise, the stream of
suffix draws, and one stream of prefix draws per population.  That is a concrete space with
a concrete law, so it is built here rather than axiomatised — `law_block` is a *lemma*.

The code deduplicates its draws — `_draw_cohort` skips suffixes already interned and
`sample_more_prefixes` skips prefixes already drawn — and so does this development:
`poolAt` and `prefixesAt` take `Finset.image` of the stream.  Deduplicating `n` i.i.d.
draws is not the same as `n` draws *without replacement*; it yields a pool that is at most
as large, so the guarantee proved here is the conservative one.  Do **not** be tempted to
model the without-replacement law as "i.i.d. conditioned on the block being injective":
those conditioned laws are inconsistent across `n` (for `Dsf = (½,¼,¼)` the first marginal
of the `n = 2` law puts mass `⅖` on the first atom, not `½`), so no space carries them all
and everything built on them would be vacuous. -/

open scoped Classical in
/-- Reading the first `n` coordinates of an i.i.d. stream. -/
lemma measurePreserving_finRestrict (D : Measure S) [IsProbabilityMeasure D] (n : ℕ) :
    MeasurePreserving (fun (p : ℕ → S) (i : Fin n) => p i.val)
      (Measure.infinitePi fun _ : ℕ => D) (Measure.pi fun _ : Fin n => D) where
  measurable := by fun_prop
  map_eq := by
    refine (Measure.pi_eq (μ := fun _ : Fin n => D) fun t ht => ?_).symm
    have hpre : (fun (p : ℕ → S) (i : Fin n) => p i.val) ⁻¹' Set.univ.pi t
        = Set.pi ↑(Finset.range n) (fun j => if h : j < n then t ⟨j, h⟩ else Set.univ) := by
      ext p
      simp only [Set.mem_preimage, Set.mem_pi, Set.mem_univ, forall_const, Finset.coe_range,
        Set.mem_Iio]
      refine ⟨fun h j hj => by rw [dif_pos hj]; exact h ⟨j, hj⟩, fun h i => ?_⟩
      simpa [dif_pos i.isLt] using h i.val i.isLt
    rw [Measure.map_apply (by fun_prop) (MeasurableSet.univ_pi ht), hpre,
      Measure.infinitePi_pi]
    · rw [← Fin.prod_univ_eq_prod_range]
      exact Finset.prod_congr rfl fun i _ => by simp [dif_pos i.isLt]
    · intro j _
      split_ifs with h
      exacts [ht ⟨j, h⟩, .univ]

variable {J : Type*} [Fintype J]

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

lemma measurable_nz : Measurable (nz : Run Ω S J → Ω) := measurable_fst

lemma measurable_sfx (i : ℕ) : Measurable (sfx (Ω := Ω) (S := S) (J := J) i) := by
  unfold sfx; fun_prop

lemma measurable_prf (j : J) (i : ℕ) : Measurable (prf (Ω := Ω) (S := S) j i) := by
  unfold prf; fun_prop

lemma measurable_cert (j : J) (i : ℕ) : Measurable (cert (Ω := Ω) (S := S) j i) := by
  unfold cert; fun_prop

/-- **The joint law of the first `n` draws.**  This is what the concentration arguments
consume, and it is a theorem about `runLaw`, not a hypothesis about an abstract space. -/
lemma law_block (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (n : ℕ) :
    Measure.map (fun x : Run Ω S J => ((fun i : Fin n => sfx i.val x),
        (fun (j : J) (i : Fin n) => prf j i.val x))) (runLaw μ D Dsf)
      = (Measure.pi fun _ : Fin n => Dsf).prod
          (Measure.pi (fun j : J => Measure.pi fun _ : Fin n => D j)) := by
  have h1 : (fun x : Run Ω S J => ((fun i : Fin n => sfx i.val x),
      (fun (j : J) (i : Fin n) => prf j i.val x)))
      = (fun y : (ℕ → S) × (J → ℕ → S) => ((fun i : Fin n => y.1 i.val),
          (fun (j : J) (i : Fin n) => y.2 j i.val))) ∘ (fun x : Run Ω S J => x.2.1) := rfl
  rw [h1, ← Measure.map_map (by fun_prop) (by fun_prop), runLaw]
  rw [show (fun x : Run Ω S J => x.2.1) = Prod.fst ∘ Prod.snd from rfl,
    ← Measure.map_map measurable_fst measurable_snd, Measure.map_snd_prod]
  simp only [measure_univ, one_smul]
  rw [Measure.map_fst_prod]
  simp only [measure_univ, one_smul]
  exact ((measurePreserving_finRestrict Dsf n).prod
    (measurePreserving_pi _ _ fun j => measurePreserving_finRestrict (D j) n)).map_eq

section Loop

variable {D : J → Measure S} {Dsf : Measure S}

/-! ## The loop, with the schedule left to the algorithm

The growth schedule is not a user parameter — it is something the algorithm optimizes as
it sees fit (`sample_suffix_family` alternates suffix- and prefix-growth based on whether
the FNR improved, and even the suffix increment is the *screened* `kept` count).  So the
statement fixes no schedule.  Instead a **history** records the sequence of budget states
the loop has passed through, and the guarantee is uniform over *all* histories and *all*
budgets — whatever the algorithm chooses, it is covered.  Histories are countable, so the
union bound still closes. -/

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

instance : Countable Budget :=
  Function.Injective.countable (f := fun b => (b.M, b.m, b.k, b.cn, b.cd, b.lo, b.hi, b.sc))
    (by rintro ⟨⟩ ⟨⟩ h; simp_all)

/-- The candidate pool at a suffix budget: the first `M` suffixes drawn, **with the seed**.

`identify_cluster_around` requires the seed to be among the candidates and asserts it —
`pst.table.column(v)` promotes it to fully observed at the top of every round, precisely so
that `candidate = pst.table.fully_observed()` contains it.  Without it `leastLossSubset`
could never return a set containing `ε`, `lloydStep` would never fire, and every family
would be the degenerate `{ε}`. -/
noncomputable def poolAt (M : ℕ) (x : Run Ω S J) : Finset S :=
  insert 1 ((Finset.range M).image (fun i => sfx i x))

lemma one_mem_poolAt (M : ℕ) (x : Run Ω S J) : (1 : S) ∈ poolAt M x :=
  Finset.mem_insert_self _ _

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

Every comparison in the algorithm is against a threshold on this count — `vote_mem_grid`
says the real-valued mean carries no more information — so the configuration is integer
data and the state stays countable. -/
noncomputable def voteCount (O : Oracle μ S) (F : Finset S) (p : S) (ω : Ω) : ℕ :=
  (F.filter (fun v => mq O (p * v) ω = 1)).card

/-- The family's vote on a prefix: the mean membership query over the family. -/
noncomputable def vote (O : Oracle μ S) (F : Finset S) (p : S) (ω : Ω) : ℝ :=
  (∑ v ∈ F, mq O (p * v) ω) / F.card

open scoped Classical in
/-- **Votes live on a grid.**  Every membership query is `0` or `1`, so a family of `k`
suffixes votes in `{0, 1/k, …, 1}`.

This is what collapses the union over boundaries.  Every comparison the algorithm makes —
`b < vote` in the cluster centre, `cfgAcc ≤ vote` and `vote < cfgRej` in the gates — comes
down to *which grid cell the threshold sits in*, an integer in `{0, …, k+1}`.  So the whole
event depends on the boundary and the margin only through finitely many integers, however
the margin is derived. -/
lemma vote_mem_grid (O : Oracle μ S) (F : Finset S) (p : S) :
    ∀ᵐ ω ∂μ, ∃ j : ℕ, j ≤ F.card ∧ vote O F p ω = (j : ℝ) / F.card := by
  filter_upwards [(ae_ball_iff F.countable_toSet).2 (fun v _ => mq_bit O (p * v))] with ω hω
  refine ⟨(F.filter (fun v => mq O (p * v) ω = 1)).card,
    Finset.card_le_card (Finset.filter_subset _ _), ?_⟩
  have hsum : ∑ v ∈ F, mq O (p * v) ω
      = ((F.filter (fun v => mq O (p * v) ω = 1)).card : ℝ) := by
    rw [← Finset.sum_filter_add_sum_filter_not F (fun v => mq O (p * v) ω = 1)]
    have h1 : ∑ v ∈ F.filter (fun v => mq O (p * v) ω = 1), mq O (p * v) ω
        = ((F.filter (fun v => mq O (p * v) ω = 1)).card : ℝ) := by
      rw [Finset.sum_congr rfl (fun v hv => (Finset.mem_filter.mp hv).2), Finset.sum_const,
        nsmul_eq_mul, mul_one]
    have h0 : ∑ v ∈ F.filter (fun v => ¬ (mq O (p * v) ω = 1)), mq O (p * v) ω = 0 := by
      refine Finset.sum_eq_zero (fun v hv => ?_)
      obtain ⟨hvF, hne⟩ := Finset.mem_filter.mp hv
      rcases hω v hvF with h | h
      · exact h
      · exact absurd h hne
    rw [h1, h0, add_zero]
  show (∑ v ∈ F, mq O (p * v) ω) / F.card = _
  rw [hsum]

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

/-- **The cluster never drifts off the seed.**  `identify_cluster_around` stops the moment
`ε` would leave, so every family the loop proposes contains it — which is what lets the
gate read the split off `ε`'s own column. -/
lemma one_mem_clusterAround (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ) :
    (1 : S) ∈ clusterAround O cn cd P cands ω k := by
  classical
  unfold clusterAround
  generalize k * P.card + 1 = n
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Function.iterate_succ_apply']
      unfold lloydStep
      split_ifs
      · exact Finset.mem_insert_self _ _
      · exact ih

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
noncomputable def screened (O : Oracle μ S) (sc : ℕ) (P cands : Finset S) (ω : Ω) :
    Finset S :=
  cands.filter (fun v => screenCount O P v ω ≤ sc)

open scoped Classical in
/-- The screen at one budget state. -/
noncomputable def screenedAt (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (x : Run Ω S J) : Finset S :=
  screened O B.sc (prefixesAt populations B.m x) (poolAt B.M x) (nz x)

lemma screened_subset (O : Oracle μ S) (sc : ℕ) (P cands : Finset S) (ω : Ω) :
    screened O sc P cands ω ⊆ cands :=
  Finset.filter_subset _ _

lemma screenCount_one (O : Oracle μ S) (P : Finset S) (ω : Ω) : screenCount O P 1 ω = 0 := by
  classical
  unfold screenCount
  rw [Finset.card_eq_zero]
  exact Finset.filter_eq_empty_iff.2 (fun p _ => by simp [mul_one])

open scoped Classical in
lemma one_mem_screened (O : Oracle μ S) (sc : ℕ) (P cands : Finset S) (ω : Ω)
    (hone : (1 : S) ∈ cands) : (1 : S) ∈ screened O sc P cands ω := by
  classical
  refine Finset.mem_filter.2 ⟨hone, ?_⟩
  show screenCount O P 1 ω ≤ sc
  rw [screenCount_one]
  exact Nat.zero_le _

lemma screenedAt_subset (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (x : Run Ω S J) : screenedAt O populations B x ⊆ poolAt B.M x :=
  screened_subset _ _ _ _ _

open scoped Classical in
/-- The seed always survives: it is the reference, so its disagreement count is zero. -/
lemma one_mem_screenedAt (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (x : Run Ω S J) : (1 : S) ∈ screenedAt O populations B x :=
  one_mem_screened O B.sc _ _ _ (one_mem_poolAt B.M x)

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

/-- **The seed's column is read at a different string from the split.**  The gate counts
`mq p`, the oracle at `p`; the split reads `p · v` for the family members `v`.  With `ε`
dropped from the family those strings are all distinct from `p`, so the persistent oracle's
bits at them are independent of the bit being scored.

This is what makes the accept side usable in `splitAcc_sound`.  That lemma needs the side
`A` fixed, but `A` is `ω`-dependent; it is determined by the reads at `p · v`, and those are
independent of the reads at `p`, so conditioning on the votes fixes `A` without disturbing
the law of the hits.  Without dropping `ε` the side would be partly determined by the very
bit the gate counts, and no conditioning would separate them. -/
lemma mul_ne_self (p v : S) (hv : v ≠ 1) : p * v ≠ p := fun h =>
  hv (mul_left_cancel (a := p) (by rw [h, mul_one]))

/-! ### Bounding an event whose index set is chosen elsewhere

`splitAcc_sound` bounds a *fixed* accept side, but the side the gate scores is
`ω`-dependent.  The way through is not a union bound over the possible sides — that would
cost `2^m` — but the observation that the side is decided by randomness independent of the
bits being scored.  Decomposing over the side's values then pays nothing: the probabilities
of the values sum to one, not to `2^m`. -/

/-- **A worst-case bound survives an independently chosen index.**  If `sel ω` always lands
in the finite set `T` and, for each value `t`, the event `sel ω = t` is independent of
`Bad t`, then a bound `E` holding for every fixed `t` holds for `Bad (sel ω)` itself.

The decomposition over `t` costs nothing: the probabilities of the values sum to one, not to
`#T`.  That is the whole reason the gate's `ω`-dependent accept side is usable — a union
bound over its possible values would cost `2^m`. -/
theorem measureReal_selection_le {β : Type*} [DecidableEq β] (T : Finset β)
    (sel : Ω → β) (hsel : ∀ ω, sel ω ∈ T)
    (hmeasSel : ∀ t, MeasurableSet {ω | sel ω = t})
    (Bad : β → Set Ω) (hmeasBad : ∀ t, MeasurableSet (Bad t)) (E : ℝ)
    (hindep : ∀ t ∈ T, μ.real ({ω | sel ω = t} ∩ Bad t) = μ.real {ω | sel ω = t} * μ.real (Bad t))
    (hbad : ∀ t ∈ T, μ.real (Bad t) ≤ E) (hE : 0 ≤ E) :
    μ.real {ω | ω ∈ Bad (sel ω)} ≤ E := by
  classical
  have hdisj : (T : Set β).PairwiseDisjoint (fun t => {ω | sel ω = t} ∩ Bad t) := by
    intro a _ b _ hab
    simp only [Function.onFun, Set.disjoint_left]
    rintro ω ⟨ha, -⟩ ⟨hb, -⟩
    exact hab (ha.symm.trans hb)
  have hdisj' : (T : Set β).PairwiseDisjoint (fun t => {ω | sel ω = t}) := by
    intro a _ b _ hab
    simp only [Function.onFun, Set.disjoint_left]
    exact fun ω ha hb => hab (ha.symm.trans hb)
  have hcover : {ω | ω ∈ Bad (sel ω)} = ⋃ t ∈ T, ({ω | sel ω = t} ∩ Bad t) := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Finset.mem_coe, exists_prop]
    exact ⟨fun h => ⟨sel ω, hsel ω, rfl, h⟩, fun ⟨t, _, he, hb⟩ => he ▸ hb⟩
  have htotal : ∑ t ∈ T, μ.real {ω | sel ω = t} = 1 := by
    have huniv : (Set.univ : Set Ω) = ⋃ t ∈ T, {ω | sel ω = t} := by
      ext ω
      simp only [Set.mem_univ, Set.mem_iUnion, Finset.mem_coe, Set.mem_setOf_eq, exists_prop,
        true_iff]
      exact ⟨sel ω, hsel ω, rfl⟩
    have hb := measureReal_biUnion_finset (μ := μ) hdisj' (fun t _ => hmeasSel t)
    rw [← hb, ← huniv, measureReal_def, measure_univ, ENNReal.toReal_one]
  calc μ.real {ω | ω ∈ Bad (sel ω)}
      = ∑ t ∈ T, μ.real ({ω | sel ω = t} ∩ Bad t) := by
        rw [hcover, measureReal_biUnion_finset hdisj (fun t _ => (hmeasSel t).inter (hmeasBad t))]
    _ = ∑ t ∈ T, μ.real {ω | sel ω = t} * μ.real (Bad t) := Finset.sum_congr rfl hindep
    _ ≤ ∑ t ∈ T, μ.real {ω | sel ω = t} * E :=
        Finset.sum_le_sum (fun t ht =>
          mul_le_mul_of_nonneg_left (hbad t ht) measureReal_nonneg)
    _ = E := by rw [← Finset.sum_mul, htotal, one_mul]

/-! ### What the clustering reads

`measureReal_selection_le` needs the gate's accept side decided by randomness independent of
the bits the gate scores.  These lemmas pin down which bits decide it: the clustering and
the votes read the oracle only at `p · v` for a representative prefix `p` and a candidate
suffix `v` — never at a bare prefix. -/

open scoped Classical in
/-- The query strings the clustering reads. -/
noncomputable def readSet (P cands : Finset S) : Finset S :=
  (P ×ˢ cands).image (fun z => z.1 * z.2)

lemma mem_readSet {P cands : Finset S} {p v : S} (hp : p ∈ P) (hv : v ∈ cands) :
    p * v ∈ readSet P cands := by
  classical
  exact Finset.mem_image.2 ⟨(p, v), Finset.mem_product.2 ⟨hp, hv⟩, rfl⟩

lemma mq_congr (O : Oracle μ S) {w : S} {ω ω' : Ω} (h : O.noise w ω = O.noise w ω') :
    mq O w ω = mq O w ω' := by simp [mq, h]

lemma voteCount_congr (O : Oracle μ S) (F : Finset S) (p : S) {ω ω' : Ω}
    (h : ∀ v ∈ F, (mq O (p * v) ω = 1 ↔ mq O (p * v) ω' = 1)) :
    voteCount O F p ω = voteCount O F p ω' := by
  classical
  unfold voteCount
  exact congrArg Finset.card (Finset.filter_congr (fun v hv => h v hv))

lemma hammingLoss_congr (O : Oracle μ S) (F : Finset S) (cn cd : ℕ) {P cands : Finset S}
    (hF : F ⊆ cands) {ω ω' : Ω} {v : S} (hv : v ∈ cands)
    (h : ∀ w ∈ readSet P cands, (mq O w ω = 1 ↔ mq O w ω' = 1)) :
    hammingLoss O F cn cd P ω v = hammingLoss O F cn cd P ω' v := by
  classical
  unfold hammingLoss
  refine congrArg _ (congrArg Finset.card (Finset.filter_congr (fun p hp => ?_)))
  rw [voteCount_congr O F p (fun v' hv' => h _ (mem_readSet hp (hF hv')))]
  exact not_congr (iff_congr (h _ (mem_readSet hp hv)) Iff.rfl)

lemma leastLossSubset_subset' (l : S → ℝ) (cands : Finset S) (k : ℕ) :
    leastLossSubset l cands k ⊆ cands := by
  classical
  unfold leastLossSubset
  split_ifs with hne
  · exact (Finset.mem_powersetCard.mp (Finset.exists_min_image _ _ hne).choose_spec.1).1
  · exact Finset.empty_subset _

lemma clusterLoss_congr (O : Oracle μ S) (F : Finset S) (cn cd : ℕ) (P cands : Finset S)
    (hF : F ⊆ cands) {ω ω' : Ω} (h : ∀ w ∈ readSet P cands, (mq O w ω = 1 ↔ mq O w ω' = 1)) :
    clusterLoss O F cn cd P cands ω = clusterLoss O F cn cd P cands ω' := by
  classical
  funext v
  unfold clusterLoss
  split_ifs with hv
  · exact hammingLoss_congr O F cn cd hF hv h
  · rfl

lemma lloydStep_congr (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (k : ℕ) {F : Finset S}
    (hF : F ⊆ cands) {ω ω' : Ω} (h : ∀ w ∈ readSet P cands, (mq O w ω = 1 ↔ mq O w ω' = 1)) :
    lloydStep O cn cd P cands ω k F = lloydStep O cn cd P cands ω' k F := by
  classical
  unfold lloydStep
  rw [clusterLoss_congr O F cn cd P cands hF h]

lemma clusterLoss_nonneg (O : Oracle μ S) (F : Finset S) (cn cd : ℕ) (P cands : Finset S)
    (ω : Ω) (v : S) : 0 ≤ clusterLoss O F cn cd P cands ω v := by
  classical
  unfold clusterLoss hammingLoss
  split_ifs
  · exact Nat.cast_nonneg _
  · exact le_rfl

open scoped Classical in
/-- **The seed's own loss against its own column is zero**, so the first step always ranks
it first — which is what stops the clustering from stalling at `{ε}`. -/
lemma clusterLoss_seed_zero (O : Oracle μ S) {cn cd : ℕ} (hcd : cn < cd) (P cands : Finset S)
    (ω : Ω) (hone : (1 : S) ∈ cands) :
    clusterLoss O {(1 : S)} cn cd P cands ω 1 = 0 := by
  classical
  unfold clusterLoss
  rw [if_pos hone, hammingLoss]
  have hempty : P.filter (fun p => ¬ ((mq O (p * 1) ω = 1)
      ↔ cn * ({(1 : S)} : Finset S).card < cd * voteCount O {(1 : S)} p ω)) = ∅ := by
    refine Finset.filter_eq_empty_iff.2 (fun p _ => ?_)
    simp only [Classical.not_not, Finset.card_singleton, mul_one, mul_comm]
    unfold voteCount
    by_cases h : mq O p ω = 1
    · have hp : ({(1 : S)} : Finset S).filter (fun v => mq O (p * v) ω = 1) = {(1 : S)} := by
        refine Finset.filter_eq_self.2 (fun v hv => ?_)
        rw [Finset.mem_singleton.1 hv, mul_one]
        exact h
      rw [hp]
      simp only [Finset.card_singleton, mul_one]
      exact ⟨fun _ => hcd, fun _ => h⟩
    · have hp : ({(1 : S)} : Finset S).filter (fun v => mq O (p * v) ω = 1) = ∅ := by
        refine Finset.filter_eq_empty_iff.2 (fun v hv => ?_)
        rw [Finset.mem_singleton.1 hv, mul_one]
        exact h
      rw [hp]
      simp only [Finset.card_empty, mul_zero]
      exact ⟨fun hc => absurd hc h, fun hc => absurd hc (by omega)⟩
  rw [hempty]
  simp

open scoped Classical in
/-- The first step takes its cohort: the seed ranks first and the rest are the `k−1` next. -/
lemma lloydStep_seed_card (O : Oracle μ S) {cn cd : ℕ} (hcd : cn < cd) (P cands : Finset S)
    (ω : Ω) (k : ℕ) (hone : (1 : S) ∈ cands) (hk : k ≤ cands.card) (hkpos : 0 < k) :
    (lloydStep O cn cd P cands ω k {(1 : S)}).card = k := by
  classical
  have hkm : k - 1 ≤ (cands.erase 1).card := by
    rw [Finset.card_erase_of_mem hone]; omega
  have hguard : ∀ w ∈ cands, w ∉ insert (1 : S)
        (leastLossSubset (clusterLoss O {(1 : S)} cn cd P cands ω) (cands.erase 1) (k - 1)) →
      ∀ v ∈ insert (1 : S)
        (leastLossSubset (clusterLoss O {(1 : S)} cn cd P cands ω) (cands.erase 1) (k - 1)),
      clusterLoss O {(1 : S)} cn cd P cands ω v
        ≤ clusterLoss O {(1 : S)} cn cd P cands ω w := by
    intro w hw hwn v hv
    rcases Finset.mem_insert.1 hv with rfl | hv'
    · rw [clusterLoss_seed_zero O hcd P cands ω hone]
      exact clusterLoss_nonneg O _ cn cd P cands ω w
    · refine leastLossSubset_least _ (cands.erase 1) (k - 1) hkm v hv' w
        (Finset.mem_erase.2 ⟨fun hc => hwn (hc ▸ Finset.mem_insert_self _ _), hw⟩)
        (fun hc => hwn (Finset.mem_insert_of_mem hc))
  unfold lloydStep
  rw [if_pos hguard, Finset.card_insert_of_notMem (fun hc =>
    (Finset.mem_erase.1 (leastLossSubset_subset' _ _ _ hc)).1 rfl),
    leastLossSubset_card _ _ _ hkm]
  omega

open scoped Classical in
/-- A step from a `k`-member family keeps `k` members. -/
lemma lloydStep_card_keep (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (hone : (1 : S) ∈ cands) (hk : k ≤ cands.card) (hkpos : 0 < k) {F : Finset S}
    (hF : F.card = k) : (lloydStep O cn cd P cands ω k F).card = k := by
  classical
  have hkm : k - 1 ≤ (cands.erase 1).card := by
    rw [Finset.card_erase_of_mem hone]; omega
  unfold lloydStep
  split_ifs
  · rw [Finset.card_insert_of_notMem (fun hc =>
      (Finset.mem_erase.1 (leastLossSubset_subset' _ _ _ hc)).1 rfl),
      leastLossSubset_card _ _ _ hkm]
    omega
  · exact hF

open scoped Classical in
/-- **The clustering does not stall.**  The seed's loss against its own column is zero, so
the first step is taken and every later one either keeps its `k` members or retakes `k`. -/
theorem clusterAround_card (O : Oracle μ S) {cn cd : ℕ} (hcd : cn < cd) (P cands : Finset S)
    (ω : Ω) (k : ℕ) (hone : (1 : S) ∈ cands) (hk : k ≤ cands.card) (hkpos : 0 < k) :
    (clusterAround O cn cd P cands ω k).card = k := by
  classical
  unfold clusterAround
  have hiter : ∀ (n : ℕ) (F : Finset S), F.card = k →
      ((lloydStep O cn cd P cands ω k)^[n] F).card = k := by
    intro n
    induction n with
    | zero => intro F hF; rwa [Function.iterate_zero_apply]
    | succ n ih =>
        intro F hF
        rw [Function.iterate_succ_apply]
        exact ih _ (lloydStep_card_keep O cn cd P cands ω k hone hk hkpos hF)
  rw [show k * P.card + 1 = (k * P.card) + 1 from rfl, Function.iterate_succ_apply]
  exact hiter _ _ (lloydStep_seed_card O hcd P cands ω k hone hk hkpos)

lemma lloydStep_subset (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (hone : (1 : S) ∈ cands) {F : Finset S} (hF : F ⊆ cands) :
    lloydStep O cn cd P cands ω k F ⊆ cands := by
  classical
  unfold lloydStep
  split_ifs
  · exact Finset.insert_subset hone
      (le_trans (leastLossSubset_subset' _ _ _) (Finset.erase_subset _ _))
  · exact hF

lemma lloydIterate_subset (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (hone : (1 : S) ∈ cands) :
    ∀ (n : ℕ) (F : Finset S), F ⊆ cands → (lloydStep O cn cd P cands ω k)^[n] F ⊆ cands := by
  intro n
  induction n with
  | zero => intro F hF; rw [Function.iterate_zero_apply]; exact hF
  | succ n ih =>
      intro F hF
      rw [Function.iterate_succ_apply]
      exact ih _ (lloydStep_subset O cn cd P cands ω k hone hF)

lemma lloydIterate_congr (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands)
    {ω ω' : Ω} (h : ∀ w ∈ readSet P cands, (mq O w ω = 1 ↔ mq O w ω' = 1)) :
    ∀ (n : ℕ) (F : Finset S), F ⊆ cands →
      (lloydStep O cn cd P cands ω k)^[n] F = (lloydStep O cn cd P cands ω' k)^[n] F := by
  intro n
  induction n with
  | zero => intro F _; rw [Function.iterate_zero_apply, Function.iterate_zero_apply]
  | succ n ih =>
      intro F hF
      calc (lloydStep O cn cd P cands ω k)^[n + 1] F
          = (lloydStep O cn cd P cands ω k)^[n] (lloydStep O cn cd P cands ω k F) :=
            Function.iterate_succ_apply _ _ _
        _ = (lloydStep O cn cd P cands ω k)^[n] (lloydStep O cn cd P cands ω' k F) :=
            congrArg (fun z => (lloydStep O cn cd P cands ω k)^[n] z)
              (lloydStep_congr O cn cd P cands k hF h)
        _ = (lloydStep O cn cd P cands ω' k)^[n] (lloydStep O cn cd P cands ω' k F) :=
            ih _ (lloydStep_subset O cn cd P cands ω' k hone hF)
        _ = (lloydStep O cn cd P cands ω' k)^[n + 1] F :=
            (Function.iterate_succ_apply _ _ _).symm

/-- **The cluster reads only `readSet`.**  Two noise draws agreeing at `p · v` for every
representative prefix and candidate suffix give the same family — so neither the family nor
any vote cast with it is decided by the oracle's bit at a bare prefix, which is the bit the
gate scores. -/
lemma clusterAround_congr_mq (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (k : ℕ)
    {ω ω' : Ω} (hone : (1 : S) ∈ cands)
    (h : ∀ w ∈ readSet P cands, (mq O w ω = 1 ↔ mq O w ω' = 1)) :
    clusterAround O cn cd P cands ω k = clusterAround O cn cd P cands ω' k :=
  lloydIterate_congr O cn cd P cands k hone h _ _ (by simpa using hone)

/-- The same from agreeing noise bits, which is how the independence arguments supply it. -/
lemma clusterAround_congr (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (k : ℕ)
    {ω ω' : Ω} (hone : (1 : S) ∈ cands)
    (h : ∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') :
    clusterAround O cn cd P cands ω k = clusterAround O cn cd P cands ω' k :=
  clusterAround_congr_mq O cn cd P cands k hone (fun w hw => by rw [mq_congr O (h w hw)])

/-- **The screen reads only `readSet`.**  Its two reads at a prefix are `p · v` and
`p · ε = p`, and the seed is a candidate, so both are already there. -/
lemma screenCount_congr (O : Oracle μ S) {P cands : Finset S} (hone : (1 : S) ∈ cands)
    {v : S} (hv : v ∈ cands) {ω ω' : Ω}
    (h : ∀ w ∈ readSet P cands, (mq O w ω = 1 ↔ mq O w ω' = 1)) :
    screenCount O P v ω = screenCount O P v ω' := by
  classical
  unfold screenCount
  refine congrArg Finset.card (Finset.filter_congr (fun p hp => ?_))
  have h1 := h _ (mem_readSet hp hv)
  have h0 : (mq O p ω = 1 ↔ mq O p ω' = 1) := by
    have := h _ (mem_readSet hp hone)
    rwa [mul_one] at this
  exact not_congr (iff_congr h1 h0)

lemma screenedAt_congr (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (d : ((ℕ → S) × (J → ℕ → S)) × (J × ℕ → S)) {ω ω' : Ω}
    (h : ∀ w ∈ readSet (prefixesAt populations B.m ((ω, d) : Run Ω S J))
      (poolAt B.M ((ω, d) : Run Ω S J)), (mq O w ω = 1 ↔ mq O w ω' = 1)) :
    screenedAt O populations B (ω, d) = screenedAt O populations B (ω', d) := by
  classical
  unfold screenedAt
  refine Finset.filter_congr (fun v hv => ?_)
  show screenCount O (prefixesAt populations B.m ((ω, d) : Run Ω S J)) v ω ≤ B.sc
    ↔ screenCount O (prefixesAt populations B.m ((ω', d) : Run Ω S J)) v ω' ≤ B.sc
  rw [screenCount_congr O (one_mem_poolAt _ _) hv h]
  rfl

/-! ### The prefix alphabet

The gate scores the oracle's bit at a prefix `p`.  Everything that decides *which side* `p`
falls on — the family, and `p`'s own vote — is read at strings `q · v` with `v ≠ ε`.  For
the gate's null to be honest those must be different strings, and that is a property of
where prefixes come from, not of the algorithm. -/

/-- A set of prefixes is **flat** when two of them never extend to the same query string.

`UniformSampler(DEFAULT_SAMPLE_LENGTH)` draws every probe at one fixed length — *"All of
E-L*'s signal comes from words drawn at this length"* — so `p * v = p' * v'` between two
probes forces `p = p'` on length alone.  Flatness is exactly what that buys, stated without
needing a length function: `Monoid` on its own does not know that strings factor. -/
def Flat (Pre : Set S) : Prop :=
  ∀ p ∈ Pre, ∀ p' ∈ Pre, ∀ v v' : S, p * v = p' * v' → p = p'

/-- No prefix is another prefix extended. -/
lemma flat_eq_one {Pre : Set S} (hflat : Flat Pre) {p p' : S} (hp : p ∈ Pre) (hp' : p' ∈ Pre)
    {v : S} (h : p * v = p') : v = 1 := by
  have hpp : p = p' := hflat p hp p' hp' v 1 (by rw [h, mul_one])
  exact mul_left_cancel (a := p) (by rw [h, hpp, mul_one])

/-- On a flat alphabet the gate's query string is never one of the split's. -/
lemma flat_ne_of_ne_one {Pre : Set S} (hflat : Flat Pre) {p p' : S} (hp : p ∈ Pre)
    (hp' : p' ∈ Pre) {v : S} (hv : v ≠ 1) : p * v ≠ p' :=
  fun h => hv (flat_eq_one hflat hp hp' h)

lemma clusterAround_subset (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (hone : (1 : S) ∈ cands) : clusterAround O cn cd P cands ω k ⊆ cands := by
  classical
  unfold clusterAround
  exact lloydIterate_subset O cn cd P cands ω k hone _ _ (by simpa using hone)

lemma noise_eq_of_mq_eq (O : Oracle μ S) {w : S} {ω ω' : Ω} (h : mq O w ω = mq O w ω') :
    O.noise w ω = O.noise w ω' := by
  rcases O.label_bit w with hl | hl <;>
    · simp only [mq, hl] at h; linarith

open scoped Classical in
/-- Every query string the state's clustering and votes read at population `j`.  A function
of the draws alone — the noise does not enter. -/
noncomputable def gateReads (populations : Finset J) (j : J) (B : Budget) (x : Run Ω S J) :
    Finset S :=
  readSet (prefixesAt populations B.m x) (poolAt B.M x)
    ∪ readSet (certOf j B.m x) ((poolAt B.M x).erase 1)

lemma clusterAt_subset (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (x : Run Ω S J) : clusterAt O populations x B ⊆ poolAt B.M x :=
  fun v hv => screenedAt_subset O populations B x
    (clusterAround_subset O B.cn B.cd _ _ (nz x) B.k (one_mem_screenedAt O populations B x) hv)

/-- **The family is decided by the bits on `readSet`** — the screen's reads and the
clustering's alike. -/
lemma clusterAt_congr (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (d : ((ℕ → S) × (J → ℕ → S)) × (J × ℕ → S)) {ω ω' : Ω}
    (h : ∀ w ∈ readSet (prefixesAt populations B.m ((ω, d) : Run Ω S J))
      (poolAt B.M ((ω, d) : Run Ω S J)), O.noise w ω = O.noise w ω') :
    clusterAt O populations (ω, d) B = clusterAt O populations (ω', d) B := by
  classical
  have hbit : ∀ w ∈ readSet (prefixesAt populations B.m ((ω, d) : Run Ω S J))
      (poolAt B.M ((ω, d) : Run Ω S J)), (mq O w ω = 1 ↔ mq O w ω' = 1) :=
    fun w hw => by rw [mq_congr O (h w hw)]
  have hcands : screenedAt O populations B (ω, d) = screenedAt O populations B (ω', d) :=
    screenedAt_congr O populations B d hbit
  unfold clusterAt
  rw [show nz ((ω, d) : Run Ω S J) = ω from rfl, show nz ((ω', d) : Run Ω S J) = ω' from rfl,
    ← hcands]
  refine clusterAround_congr_mq O B.cn B.cd _ _ B.k (one_mem_screenedAt O populations B (ω, d))
    (fun w hw => hbit w ?_)
  exact Finset.mem_image.2 (by
    obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hw
    obtain ⟨hp, hvs⟩ := Finset.mem_product.1 hpv
    exact ⟨(p, v), Finset.mem_product.2 ⟨hp,
      screenedAt_subset O populations B (ω, d) hvs⟩, rfl⟩)

open scoped Classical in
/-- **Dropping the seed costs the vote one count.**  `1 ∈ F` always, so the full family's
vote at `p` is the erased family's plus the seed's own read there. -/
lemma voteCount_le_erase_succ (O : Oracle μ S) (F : Finset S) (p : S) (ω : Ω) :
    voteCount O F p ω ≤ voteCount O (F.erase 1) p ω + 1 := by
  classical
  unfold voteCount
  have hsub : F.filter (fun v => mq O (p * v) ω = 1)
      ⊆ insert (1 : S) ((F.erase 1).filter (fun v => mq O (p * v) ω = 1)) := by
    intro v hv
    obtain ⟨hvF, hvm⟩ := Finset.mem_filter.1 hv
    by_cases h1 : v = 1
    · rw [h1]
      exact Finset.mem_insert_self _ _
    · exact Finset.mem_insert_of_mem (Finset.mem_filter.2 ⟨Finset.mem_erase.2 ⟨h1, hvF⟩, hvm⟩)
  exact le_trans (Finset.card_le_card hsub) (Finset.card_insert_le _ _)

open scoped Classical in
/-- The prefixes the gate's family accepts at population `j`. -/
noncomputable def sideAcc (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (x : Run Ω S J) : Finset S :=
  (certOf j B.m x).filter
    (fun p => B.hi - 1 < voteCount O ((clusterAt O populations x B).erase 1) p (nz x))

open scoped Classical in
/-- The prefixes it rejects. -/
noncomputable def sideRej (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (x : Run Ω S J) : Finset S :=
  (certOf j B.m x).filter
    (fun p => voteCount O ((clusterAt O populations x B).erase 1) p (nz x) ≤ B.lo)

lemma sideAcc_subset (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (x : Run Ω S J) : sideAcc O populations j B x ⊆ certOf j B.m x :=
  Finset.filter_subset _ _

lemma sideRej_subset (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (x : Run Ω S J) : sideRej O populations j B x ⊆ certOf j B.m x :=
  Finset.filter_subset _ _

/-- **Which side each prefix falls on is decided off the gate's own column.**  Both sides
are determined by the oracle's bits at `gateReads`, and `disjoint_readSet` puts those
strings off the certification prefixes the gate scores. -/
lemma sideAcc_congr (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (d : ((ℕ → S) × (J → ℕ → S)) × (J × ℕ → S)) {ω ω' : Ω}
    (h : ∀ w ∈ gateReads populations j B ((ω, d) : Run Ω S J),
      O.noise w ω = O.noise w ω') :
    sideAcc O populations j B (ω, d) = sideAcc O populations j B (ω', d) := by
  classical
  have hfam : clusterAt O populations ((ω, d) : Run Ω S J) B
      = clusterAt O populations ((ω', d) : Run Ω S J) B :=
    clusterAt_congr O populations B d (fun w hw => h w (Finset.mem_union_left _ hw))
  have hsub : clusterAt O populations ((ω, d) : Run Ω S J) B
      ⊆ poolAt B.M ((ω, d) : Run Ω S J) :=
    clusterAt_subset O populations B _
  unfold sideAcc
  refine Finset.filter_congr (fun p hp => ?_)
  have hvc : voteCount O ((clusterAt O populations ((ω, d) : Run Ω S J) B).erase 1) p ω
      = voteCount O ((clusterAt O populations ((ω, d) : Run Ω S J) B).erase 1) p ω' := by
    refine voteCount_congr O _ p (fun v hv => ?_)
    exact mq_congr O (h _ (Finset.mem_union_right _ (mem_readSet hp
      (Finset.mem_erase.2 ⟨(Finset.mem_erase.1 hv).1, hsub (Finset.mem_erase.1 hv).2⟩)))) ▸ Iff.rfl
  simp only [nz, ← hfam, hvc]

lemma sideRej_congr (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (d : ((ℕ → S) × (J → ℕ → S)) × (J × ℕ → S)) {ω ω' : Ω}
    (h : ∀ w ∈ gateReads populations j B ((ω, d) : Run Ω S J),
      O.noise w ω = O.noise w ω') :
    sideRej O populations j B (ω, d) = sideRej O populations j B (ω', d) := by
  classical
  have hfam : clusterAt O populations ((ω, d) : Run Ω S J) B
      = clusterAt O populations ((ω', d) : Run Ω S J) B :=
    clusterAt_congr O populations B d (fun w hw => h w (Finset.mem_union_left _ hw))
  have hsub : clusterAt O populations ((ω, d) : Run Ω S J) B
      ⊆ poolAt B.M ((ω, d) : Run Ω S J) :=
    clusterAt_subset O populations B _
  unfold sideRej
  refine Finset.filter_congr (fun p hp => ?_)
  have hvc : voteCount O ((clusterAt O populations ((ω, d) : Run Ω S J) B).erase 1) p ω
      = voteCount O ((clusterAt O populations ((ω, d) : Run Ω S J) B).erase 1) p ω' := by
    refine voteCount_congr O _ p (fun v hv => ?_)
    exact mq_congr O (h _ (Finset.mem_union_right _ (mem_readSet hp
      (Finset.mem_erase.2 ⟨(Finset.mem_erase.1 hv).1, hsub (Finset.mem_erase.1 hv).2⟩)))) ▸ Iff.rfl
  simp only [nz, ← hfam, hvc]

/-- **The gate's own query strings are not read by the clustering.**  A prefix is never
`p · v` for a prefix `p` and any suffix, on a flat alphabet — so the bits the gate scores
are untouched by everything that decides which side each prefix falls on. -/
lemma disjoint_readSet {Pre : Set S} (hflat : Flat Pre) {P cands C : Finset S}
    (hP : ∀ p ∈ P, p ∈ Pre) (hC : ∀ p ∈ C, p ∈ Pre) (hPC : Disjoint P C) :
    Disjoint C (readSet P cands) := by
  classical
  refine Finset.disjoint_left.2 (fun z hz hmem => ?_)
  obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hmem
  obtain ⟨hp, -⟩ := Finset.mem_product.1 hpv
  have hv1 : v = 1 := flat_eq_one hflat (hP p hp) (hC _ hz) rfl
  rw [hv1, mul_one] at hz
  exact (Finset.disjoint_left.1 hPC hp) hz

/-- The same, for the gate's own prefixes against their own votes: those read `p · v` with
`v ≠ ε`, which is never a prefix. -/
lemma disjoint_readSet_erase {Pre : Set S} (hflat : Flat Pre) {cands C : Finset S}
    (hC : ∀ p ∈ C, p ∈ Pre) : Disjoint C (readSet C (cands.erase 1)) := by
  classical
  refine Finset.disjoint_left.2 (fun z hz hmem => ?_)
  obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hmem
  obtain ⟨hp, hv⟩ := Finset.mem_product.1 hpv
  exact (Finset.mem_erase.1 hv).1 (flat_eq_one hflat (hC p hp) (hC _ hz) rfl)

/-- **A population prefix the table does not hold is read nowhere by the clustering.**  Its
query strings `p · v` collide with the clustering's `q · v'` only if `p = q`. -/
lemma disjoint_image_readSet {Pre : Set S} (hflat : Flat Pre) {P cands : Finset S} {p : S}
    (hP : ∀ q ∈ P, q ∈ Pre) (hp : p ∈ Pre) (hpP : p ∉ P) :
    Disjoint (↑(cands.image (fun v => p * v)) : Set S) (↑(readSet P cands) : Set S) := by
  classical
  rw [Finset.disjoint_coe, Finset.disjoint_left]
  rintro z hz hmem
  obtain ⟨v, -, rfl⟩ := Finset.mem_image.1 hz
  obtain ⟨⟨q, v'⟩, hqv, hq⟩ := Finset.mem_image.1 hmem
  obtain ⟨hqP, -⟩ := Finset.mem_product.1 hqv
  exact hpP (hflat p hp q (hP q hqP) v v' hq.symm ▸ hqP)

/-! ### Independence of the score from the side

Both events are structurally measurable over disjoint sub-families of the oracle's bits, so
this needs no factorisation theorem: the side is decided by the bits on `gateReads`, the
score reads the bits on the certification prefixes, and `disjoint_readSet` separates them. -/

/-- The σ-algebra the oracle's bits on a set of query strings generate. -/
def noiseAlg (O : Oracle μ S) (T : Set S) : MeasurableSpace Ω :=
  ⨆ w ∈ T, MeasurableSpace.comap (O.noise w) inferInstance

lemma noiseAlg_le (O : Oracle μ S) (T : Set S) : noiseAlg O T ≤ ‹MeasurableSpace Ω› :=
  iSup₂_le (fun w _ => (O.noise_meas' w).comap_le)

lemma measurableSet_noise_preimage (O : Oracle μ S) {T : Set S} {w : S} (hw : w ∈ T)
    {s : Set ℝ} (hs : MeasurableSet s) :
    MeasurableSet[noiseAlg O T] (O.noise w ⁻¹' s) := by
  have hle : MeasurableSpace.comap (O.noise w) inferInstance ≤ noiseAlg O T :=
    le_iSup₂ (f := fun w (_ : w ∈ T) => MeasurableSpace.comap (O.noise w) inferInstance) w hw
  exact hle _ ⟨s, hs, rfl⟩

lemma measurableSet_mq_eq_one (O : Oracle μ S) {T : Set S} {w : S} (hw : w ∈ T) :
    MeasurableSet[noiseAlg O T] {ω | mq O w ω = 1} := by
  have hpre : {ω | mq O w ω = 1}
      = O.noise w ⁻¹' {r : ℝ | O.label w + (1 - 2 * O.label w) * r = 1} := rfl
  rw [hpre]
  exact measurableSet_noise_preimage O hw
    (measurableSet_eq_fun (by fun_prop) measurable_const)

/-- Two independent blocks of the oracle's bits. -/
lemma indep_noiseAlg (O : Oracle μ S) {T T' : Set S} (h : Disjoint T T') :
    Indep (noiseAlg O T) (noiseAlg O T') μ :=
  indep_iSup_of_disjoint (fun w => (O.noise_meas' w).comap_le) O.noise_indep h

open scoped Classical in
/-- A block's bits decide which of its strings satisfy any condition they decide. -/
lemma measurableSet_filter_fiber' (O : Oracle μ S) {T : Set S} {A : Finset S}
    (Pr : S → Ω → Prop) [inst : ∀ ω, DecidablePred (fun p => Pr p ω)]
    (hPr : ∀ p ∈ A, MeasurableSet[noiseAlg O T] {ω | Pr p ω}) (U : Finset S) :
    MeasurableSet[noiseAlg O T] {ω | A.filter (fun p => Pr p ω) = U} := by
  classical
  have hfib : {ω | A.filter (fun p => Pr p ω) = U}
      = if U ⊆ A then (⋂ p ∈ U, {ω | Pr p ω}) ∩ ⋂ p ∈ A \ U, {ω | Pr p ω}ᶜ
        else ∅ := by
    split_ifs with hUA
    · ext ω
      simp only [Set.mem_setOf_eq, Set.mem_inter_iff, Set.mem_iInter, Set.mem_compl_iff,
        Finset.mem_sdiff]
      constructor
      · rintro rfl
        exact ⟨fun p hp => (Finset.mem_filter.1 hp).2,
          fun p hp => fun hc => hp.2 (Finset.mem_filter.2 ⟨hp.1, hc⟩)⟩
      · rintro ⟨h1, h2⟩
        ext p
        simp only [Finset.mem_filter]
        exact ⟨fun ⟨hpA, hpm⟩ => by_contra (fun hpU => h2 p ⟨hpA, hpU⟩ hpm),
          fun hpU => ⟨hUA hpU, h1 p hpU⟩⟩
    · ext ω
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
      exact fun hc => hUA (hc ▸ Finset.filter_subset _ _)
  rw [hfib]
  split_ifs with hUA
  · apply MeasurableSet.inter
    · apply Finset.measurableSet_biInter
      exact fun p hp => hPr p (hUA hp)
    · apply Finset.measurableSet_biInter
      exact fun p hp => (hPr p (Finset.mem_sdiff.1 hp).1).compl
  · exact (noiseAlg O T).measurableSet_empty

lemma measurableSet_noise_eq_one (O : Oracle μ S) {T : Set S} {w : S} (hw : w ∈ T) :
    MeasurableSet[noiseAlg O T] {ω | O.noise w ω = 1} :=
  measurableSet_noise_preimage O hw (measurableSet_singleton 1)

open scoped Classical in
/-- Which of a block's prefixes read as accepting is decided by that block's bits. -/
lemma measurableSet_filter_fiber (O : Oracle μ S) {T : Set S} {A : Finset S} (hA : ↑A ⊆ T)
    (U : Finset S) :
    MeasurableSet[noiseAlg O T] {ω | A.filter (fun p => mq O p ω = 1) = U} :=
  measurableSet_filter_fiber' O _ (fun p hp => measurableSet_mq_eq_one O (hA hp)) U

open scoped Classical in
/-- The bit pattern a block's strings show. -/
noncomputable def noisePattern (O : Oracle μ S) (Q : Finset S) (ω : Ω) : Finset S :=
  Q.filter (fun w => O.noise w ω = 1)

lemma noisePattern_mem (O : Oracle μ S) (Q : Finset S) (ω : Ω) :
    noisePattern O Q ω ∈ Q.powerset :=
  Finset.mem_powerset.2 (Finset.filter_subset _ _)

lemma measurableSet_noisePattern (O : Oracle μ S) (Q : Finset S) (t : Finset S) :
    MeasurableSet[noiseAlg O ↑Q] {ω | noisePattern O Q ω = t} :=
  measurableSet_filter_fiber' O _ (fun w hw => measurableSet_noise_eq_one O hw) t

open scoped Classical in
/-- The runs on which a block's bits are all genuinely `0` or `1`. -/
def noiseClean (O : Oracle μ S) (Q : Finset S) : Set Ω :=
  {ω | ∀ w ∈ Q, O.noise w ω = 0 ∨ O.noise w ω = 1}

lemma measurableSet_noiseClean (O : Oracle μ S) (Q : Finset S) :
    MeasurableSet[noiseAlg O ↑Q] (noiseClean O Q) := by
  classical
  have hset : noiseClean O Q = ⋂ w ∈ Q, (O.noise w ⁻¹' ({0, 1} : Set ℝ)) := by
    ext ω; simp [noiseClean, Set.mem_iInter]
  rw [hset]
  apply Finset.measurableSet_biInter
  exact fun w hw => measurableSet_noise_preimage O hw
    ((measurableSet_singleton 0).union (measurableSet_singleton 1))

lemma noiseClean_ae (O : Oracle μ S) (Q : Finset S) : μ.real (noiseClean O Q)ᶜ = 0 := by
  classical
  have hae : ∀ᵐ ω ∂μ, ω ∈ noiseClean O Q := by
    filter_upwards [(ae_ball_iff Q.countable_toSet).2 (fun w _ => O.noise_bit w)] with ω hω
    exact fun w hw => hω w hw
  have hz : μ (noiseClean O Q)ᶜ = 0 := by
    have h1 := MeasureTheory.ae_iff.1 hae
    rwa [show {a | a ∉ noiseClean O Q} = (noiseClean O Q)ᶜ from rfl] at h1
  simp [measureReal_def, hz]

/-- On a clean run the bit pattern pins the bits down. -/
lemma noise_eq_of_pattern (O : Oracle μ S) {Q : Finset S} {ω ω' : Ω}
    (hω : ω ∈ noiseClean O Q) (hω' : ω' ∈ noiseClean O Q)
    (h : noisePattern O Q ω = noisePattern O Q ω') :
    ∀ w ∈ Q, O.noise w ω = O.noise w ω' := by
  classical
  intro w hw
  have hiff : (O.noise w ω = 1) ↔ (O.noise w ω' = 1) := by
    constructor
    · intro h1
      have : w ∈ noisePattern O Q ω := Finset.mem_filter.2 ⟨hw, h1⟩
      rw [h] at this
      exact (Finset.mem_filter.1 this).2
    · intro h1
      have : w ∈ noisePattern O Q ω' := Finset.mem_filter.2 ⟨hw, h1⟩
      rw [← h] at this
      exact (Finset.mem_filter.1 this).2
  rcases hω w hw with h0 | h1
  · rcases hω' w hw with h0' | h1'
    · rw [h0, h0']
    · exact absurd (hiff.2 h1') (by rw [h0]; norm_num)
  · rcases hω' w hw with h0' | h1'
    · exact absurd (hiff.1 h1) (by rw [h0']; norm_num)
    · rw [h1, h1']

open scoped Classical in
/-- Hence any condition on that count is: it takes finitely many values. -/
lemma measurableSet_filter_pred (O : Oracle μ S) {T : Set S} {A : Finset S} (hA : ↑A ⊆ T)
    (P : Finset S → Prop) :
    MeasurableSet[noiseAlg O T] {ω | P (A.filter (fun p => mq O p ω = 1))} := by
  classical
  have hcover : {ω | P (A.filter (fun p => mq O p ω = 1))}
      = ⋃ U ∈ A.powerset.filter P, {ω | A.filter (fun p => mq O p ω = 1) = U} := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Finset.mem_coe, Finset.mem_filter,
      Finset.mem_powerset, exists_prop]
    refine ⟨fun h => ⟨_, ⟨Finset.filter_subset _ _, h⟩, rfl⟩, ?_⟩
    rintro ⟨U, ⟨-, hPU⟩, rfl⟩
    exact hPU
  rw [hcover]
  apply Finset.measurableSet_biUnion
  exact fun U _ => measurableSet_filter_fiber O hA U

open scoped Classical in
/-- The same for an arbitrary per-element predicate on the bits. -/
lemma measurableSet_filter_pred' (O : Oracle μ S) {T : Set S} {A : Finset S}
    (Pr : S → Ω → Prop) [inst : ∀ ω, DecidablePred (fun p => Pr p ω)]
    (hPr : ∀ p ∈ A, MeasurableSet[noiseAlg O T] {ω | Pr p ω}) (Q : Finset S → Prop) :
    MeasurableSet[noiseAlg O T] {ω | Q (A.filter (fun p => Pr p ω))} := by
  classical
  have hcover : {ω | Q (A.filter (fun p => Pr p ω))}
      = ⋃ U ∈ A.powerset.filter Q, {ω | A.filter (fun p => Pr p ω) = U} := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Finset.mem_coe, Finset.mem_filter,
      Finset.mem_powerset, exists_prop]
    refine ⟨fun h => ⟨_, ⟨Finset.filter_subset _ _, h⟩, rfl⟩, ?_⟩
    rintro ⟨U, ⟨-, hQU⟩, rfl⟩
    exact hQU
  rw [hcover]
  exact Finset.measurableSet_biUnion _ (fun U _ => measurableSet_filter_fiber' O Pr hPr U)

open scoped Classical in
/-- The same when the reads are taken at shifted strings `r v` rather than at `v` itself:
the vote at a population prefix reads `p · v`, not `v`. -/
lemma measurableSet_filter_pred_map (O : Oracle μ S) {T : Set S} {A : Finset S} (r : S → S)
    (hA : ∀ v ∈ A, r v ∈ T) (P : Finset S → Prop) :
    MeasurableSet[noiseAlg O T] {ω | P (A.filter (fun v => mq O (r v) ω = 1))} := by
  classical
  have hcover : {ω | P (A.filter (fun v => mq O (r v) ω = 1))}
      = ⋃ U ∈ A.powerset.filter P, {ω | A.filter (fun v => mq O (r v) ω = 1) = U} := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Finset.mem_coe, Finset.mem_filter,
      Finset.mem_powerset, exists_prop]
    refine ⟨fun h => ⟨_, ⟨Finset.filter_subset _ _, h⟩, rfl⟩, ?_⟩
    rintro ⟨U, ⟨-, hPU⟩, rfl⟩
    exact hPU
  rw [hcover]
  refine Finset.measurableSet_biUnion _ (fun U _ => ?_)
  exact measurableSet_filter_fiber' O _ (fun v hv => measurableSet_mq_eq_one O (hA v hv)) U

open scoped Classical in
/-- **The family is a measurable function of the run.**  It is decided by which of the
clustering's finitely many query strings read accepting, and each of those patterns is a
measurable event. -/
lemma measurableSet_clusterAround (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands) (A₀ : Finset S) :
    MeasurableSet {ω | clusterAround O cn cd P cands ω k = A₀} := by
  classical
  set Pred : Finset S → Prop := fun U => ∃ ω', (readSet P cands).filter
    (fun w => mq O w ω' = 1) = U ∧ clusterAround O cn cd P cands ω' k = A₀ with hPred
  have hcov : {ω | clusterAround O cn cd P cands ω k = A₀}
      = {ω | Pred ((readSet P cands).filter (fun w => mq O w ω = 1))} := by
    ext ω
    simp only [Set.mem_setOf_eq, hPred]
    refine ⟨fun h => ⟨ω, rfl, h⟩, ?_⟩
    rintro ⟨ω', hU, hA⟩
    refine (clusterAround_congr_mq O cn cd P cands k hone (fun w hw => ?_)).trans hA
    have := Finset.ext_iff.1 hU w
    simp only [Finset.mem_filter, hw, true_and] at this
    exact this.symm
  rw [hcov]
  exact noiseAlg_le O Set.univ _
    (measurableSet_filter_pred O (T := Set.univ) (by simp) Pred)

open scoped Classical in
/-- **Congruence becomes measurability.**  A side decided by a block's bits is, on the clean
runs, a union of that block's pattern fibres. -/
lemma measurableSet_side_clean (O : Oracle μ S) (Q : Finset S) (side : Ω → Finset S)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → side ω = side ω')
    (A₀ : Finset S) :
    MeasurableSet[noiseAlg O ↑Q] ({ω | side ω = A₀} ∩ noiseClean O Q) := by
  classical
  have hcover : {ω | side ω = A₀} ∩ noiseClean O Q
      = ⋃ t ∈ Q.powerset.filter
          (fun t => ∃ ω, ω ∈ noiseClean O Q ∧ noisePattern O Q ω = t ∧ side ω = A₀),
        ({ω | noisePattern O Q ω = t} ∩ noiseClean O Q) := by
    ext ω
    simp only [Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_iUnion, Finset.mem_coe,
      Finset.mem_filter, Finset.mem_powerset, exists_prop]
    constructor
    · rintro ⟨hsd, hcl⟩
      exact ⟨noisePattern O Q ω,
        ⟨Finset.mem_powerset.1 (noisePattern_mem O Q ω), ω, hcl, rfl, hsd⟩, rfl, hcl⟩
    · rintro ⟨t, ⟨-, ω₀, hcl₀, hpat₀, hsd₀⟩, hpat, hcl⟩
      refine ⟨?_, hcl⟩
      rw [← hsd₀]
      exact hcongr ω ω₀ (noise_eq_of_pattern O hcl hcl₀ (hpat.trans hpat₀.symm))
  rw [hcover]
  apply Finset.measurableSet_biUnion
  exact fun t _ => (measurableSet_noisePattern O Q t).inter (measurableSet_noiseClean O Q)

open scoped Classical in
/-- **A worst case survives the side being chosen elsewhere.**

`hbad` bounds the score's failure for each *fixed* side; the conclusion bounds it for the
side the run actually produces.  What makes that free is that the side is decided by the
block `Q` while the score reads the strings `r '' C`, and those are disjoint.  The gate
takes `r = id` (it scores prefixes); the vote at a population prefix `p` takes
`r v = p * v`. -/
theorem selection_side_bound (O : Oracle μ S) (C Q : Finset S) (r : S → S)
    (hdisj : Disjoint (↑(C.image r) : Set S) (↑Q : Set S))
    (T : Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T) (hTC : ∀ t ∈ T, t ⊆ C)
    (side : Ω → Finset S) (hside : ∀ ω, side ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → side ω = side ω')
    (P : Finset S → Finset S → Prop) (E : ℝ) (hE : 0 ≤ E)
    (hbad : ∀ A₀ ∈ T, μ.real {ω | P A₀ (A₀.filter (fun v => mq O (r v) ω = 1))} ≤ E) :
    μ.real {ω | P (side ω) ((side ω).filter (fun v => mq O (r v) ω = 1))} ≤ E := by
  classical
  set Bad : Finset S → Set Ω :=
    fun A₀ => {ω | P A₀ (A₀.filter (fun v => mq O (r v) ω = 1))} with hBaddef
  set side' : Ω → Finset S := fun ω => if ω ∈ noiseClean O Q then side ω else t₀ with hside'def
  have hmeasBad : ∀ A₀, MeasurableSet (Bad A₀) := fun A₀ =>
    noiseAlg_le O Set.univ _ (measurableSet_filter_pred_map O (T := Set.univ) r (by simp) _)
  have hmeasBadC : ∀ A₀ ∈ T, MeasurableSet[noiseAlg O ↑(C.image r)] (Bad A₀) :=
    fun A₀ hA₀ => measurableSet_filter_pred_map O r
      (fun v hv => Finset.mem_image_of_mem r (hTC A₀ hA₀ hv)) _
  have hsel' : ∀ ω, side' ω ∈ T := by
    intro ω
    rw [hside'def]
    by_cases hc : ω ∈ noiseClean O Q
    · simpa [hc] using hside ω
    · simpa [hc] using ht₀
  have hsplit : ∀ A₀, {ω | side' ω = A₀}
      = ({ω | side ω = A₀} ∩ noiseClean O Q) ∪ (if A₀ = t₀ then (noiseClean O Q)ᶜ else ∅) := by
    intro A₀
    ext ω
    by_cases hc : ω ∈ noiseClean O Q <;> by_cases he : A₀ = t₀ <;>
      simp [hside'def, hc, he, Set.mem_setOf_eq, eq_comm (a := t₀)]
  have hmeasSelQ : ∀ A₀, MeasurableSet[noiseAlg O ↑Q] {ω | side' ω = A₀} := by
    intro A₀
    rw [hsplit A₀]
    refine MeasurableSet.union (measurableSet_side_clean O Q side hcongr A₀) ?_
    split_ifs
    · exact (measurableSet_noiseClean O Q).compl
    · exact (noiseAlg O ↑Q).measurableSet_empty
  have hmeasSel : ∀ A₀, MeasurableSet {ω | side' ω = A₀} := fun A₀ =>
    noiseAlg_le O ↑Q _ (hmeasSelQ A₀)
  have hindep : ∀ A₀ ∈ T,
      μ.real ({ω | side' ω = A₀} ∩ Bad A₀) = μ.real {ω | side' ω = A₀} * μ.real (Bad A₀) := by
    intro A₀ hA₀
    have hI := (indep_noiseAlg O hdisj.symm).indepSet_of_measurableSet (hmeasSelQ A₀)
      (hmeasBadC A₀ hA₀)
    have := hI.measure_inter_eq_mul
    simp only [measureReal_def, this, ENNReal.toReal_mul]
  have hmain := measureReal_selection_le (μ := μ) T side' hsel' hmeasSel Bad hmeasBad E
    hindep (fun A₀ hA₀ => hbad A₀ hA₀) hE
  have hsub : {ω | P (side ω) ((side ω).filter (fun v => mq O (r v) ω = 1))}
      ⊆ {ω | ω ∈ Bad (side' ω)} ∪ (noiseClean O Q)ᶜ := by
    intro ω hω
    by_cases hc : ω ∈ noiseClean O Q
    · refine Or.inl ?_
      change ω ∈ Bad (side' ω)
      rw [hside'def]
      simp only [hc, if_pos]
      exact hω
    · exact Or.inr hc
  calc μ.real {ω | P (side ω) ((side ω).filter (fun v => mq O (r v) ω = 1))}
      ≤ μ.real ({ω | ω ∈ Bad (side' ω)} ∪ (noiseClean O Q)ᶜ) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ μ.real {ω | ω ∈ Bad (side' ω)} + μ.real (noiseClean O Q)ᶜ := measureReal_union_le _ _
    _ = μ.real {ω | ω ∈ Bad (side' ω)} := by rw [noiseClean_ae O Q, add_zero]
    _ ≤ E := hmain

/-- The gate's instance: it scores the prefixes themselves. -/
theorem gate_side_bound (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    (side : Ω → Finset S) (hside : ∀ ω, side ω ⊆ C)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → side ω = side ω')
    (P : Finset S → Finset S → Prop) (E : ℝ) (hE : 0 ≤ E)
    (hbad : ∀ A₀ ∈ C.powerset, μ.real {ω | P A₀ (A₀.filter (fun p => mq O p ω = 1))} ≤ E) :
    μ.real {ω | P (side ω) ((side ω).filter (fun p => mq O p ω = 1))} ≤ E :=
  selection_side_bound O C Q id (by simpa using hdisj) C.powerset ∅ (Finset.empty_mem_powerset C)
    (fun t ht => Finset.mem_powerset.1 ht) side (fun ω => Finset.mem_powerset.2 (hside ω))
    hcongr P E hE hbad

open scoped Classical in
/-- The label sum over a block splits into its truly-rejecting and truly-accepting parts. -/
lemma sum_label_eq (O : Oracle μ S) (A : Finset S) :
    ∑ p ∈ A, O.label p = ((A.card : ℝ) - ((A.filter (fun p => O.label p = 0)).card : ℝ)) := by
  classical
  rw [← Finset.sum_filter_add_sum_filter_not A (fun p => O.label p = 0)]
  have h0 : ∑ p ∈ A.filter (fun p => O.label p = 0), O.label p = 0 :=
    Finset.sum_eq_zero (fun p hp => (Finset.mem_filter.1 hp).2)
  have hone : ∀ p ∈ A.filter (fun p => ¬ (O.label p = 0)), O.label p = 1 := by
    intro p hp
    rcases O.label_bit p with hl | hl
    · exact absurd hl (Finset.mem_filter.1 hp).2
    · exact hl
  have h1 : ∑ p ∈ A.filter (fun p => ¬ (O.label p = 0)), O.label p
      = ((A.filter (fun p => ¬ (O.label p = 0))).card : ℝ) := by
    rw [Finset.sum_congr rfl hone, Finset.sum_const, nsmul_eq_mul, mul_one]
  rw [h0, h1, zero_add]
  have hc := Finset.card_filter_add_card_filter_not (s := A) (fun p => O.label p = 0)
  have : ((A.filter (fun p => ¬ (O.label p = 0))).card : ℝ)
      = (A.card : ℝ) - ((A.filter (fun p => O.label p = 0)).card : ℝ) := by
    have : ((A.filter (fun p => O.label p = 0)).card : ℝ)
        + ((A.filter (fun p => ¬ (O.label p = 0))).card : ℝ) = (A.card : ℝ) := by
      exact_mod_cast congrArg (Nat.cast : ℕ → ℝ) hc
    linarith
  rw [this]

/-! ### The certification draws see the wrong set

The family and its wrong-set live in the run's first factor; the certification draws are the
second and are independent of them.  So with the wrong-set held fixed this is plain
Hoeffding over i.i.d. draws — no noise enters, which is the second thing the fresh stream
buys. -/

lemma measurableSet_of_countable (W : Set S) : MeasurableSet W :=
  (Set.to_countable W).measurableSet

open scoped Classical in
/-- **A wrong set of mass `≥ εcov` is hit by all but `t` of that fraction of the draws**,
except with probability `exp(−2mt²)`.

The margin `t` is free for the same reason the gate's rates are: it trades against what the
gate then has to resolve.  Charging it a fixed `εcov/2`, as an earlier version did, costs
the gate a factor of four in the drift it must see and sixteen in the prefixes that takes. -/
lemma cert_hits_wrongSet (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    (j : J) (m : ℕ) (W : Set S) (εcov t : ℝ) (hεcov : 0 ≤ εcov) (ht : 0 ≤ t)
    (hW : εcov ≤ (D j).real W) :
    (Measure.infinitePi fun z : J × ℕ => D z.1).real
        {c | (((Finset.range m).filter (fun i => c (j, i) ∈ W)).card : ℝ)
          ≤ (m : ℝ) * (εcov - t)}
      ≤ Real.exp (-2 * (m : ℝ) * t ^ 2) := by
  classical
  set ν : Measure (J × ℕ → S) := Measure.infinitePi fun z : J × ℕ => D z.1 with hνdef
  have hWm : MeasurableSet W := measurableSet_of_countable W
  set ind : S → ℝ := W.indicator 1 with hinddef
  have hindm : Measurable ind := measurable_const.indicator hWm
  set X : ℕ → (J × ℕ → S) → ℝ := fun i c => ind (c (j, i)) with hXdef
  have hcoord : iIndepFun (fun (z : J × ℕ) (c : J × ℕ → S) => c z) ν :=
    iIndepFun_infinitePi (fun _ => measurable_id)
  have hinj : Function.Injective (fun i : ℕ => (j, i)) := fun a b hab => (Prod.mk.inj hab).2
  have hindep : iIndepFun X ν :=
    (hcoord.precomp hinj).comp (fun _ => ind) (fun _ => hindm)
  have hmeas : ∀ i, AEMeasurable (X i) ν := fun i =>
    (hindm.comp (measurable_pi_apply _)).aemeasurable
  have hicc : ∀ i, ∀ᵐ c ∂ν, X i c ∈ Set.Icc (0 : ℝ) 1 := by
    intro i
    filter_upwards with c
    by_cases h : c (j, i) ∈ W
    · simp [hXdef, hinddef, Set.indicator_of_mem h]
    · simp [hXdef, hinddef, Set.indicator_of_notMem h]
  have hmean : ∀ i, ν[X i] = (D j).real W := by
    intro i
    have hmp : MeasurePreserving (fun c : J × ℕ → S => c (j, i)) ν (D j) :=
      measurePreserving_eval_infinitePi _ (j, i)
    calc ν[X i] = ∫ s, ind s ∂(D j) := by
          rw [← hmp.map_eq,
            integral_map (measurable_pi_apply _).aemeasurable hindm.aestronglyMeasurable]
      _ = (D j).real W := by rw [hinddef, integral_indicator_one hWm]
  have hsum : ((Finset.range m).card : ℝ) * εcov ≤ ∑ i ∈ Finset.range m, ν[X i] := by
    rw [Finset.sum_congr rfl (fun i _ => hmean i), Finset.sum_const, nsmul_eq_mul,
      Finset.card_range]
    have : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
    nlinarith [hW]
  have hmain := sumLower_le X (Finset.range m) εcov t hmeas hindep hicc hsum ht
  have hcount : ∀ c : J × ℕ → S, ∑ i ∈ Finset.range m, X i c
      = (((Finset.range m).filter (fun i => c (j, i) ∈ W)).card : ℝ) := by
    intro c
    rw [← Finset.sum_filter_add_sum_filter_not (Finset.range m) (fun i => c (j, i) ∈ W)]
    have h1 : ∑ i ∈ (Finset.range m).filter (fun i => c (j, i) ∈ W), X i c
        = (((Finset.range m).filter (fun i => c (j, i) ∈ W)).card : ℝ) := by
      have hone : ∀ i ∈ (Finset.range m).filter (fun i => c (j, i) ∈ W), X i c = (1 : ℝ) := by
        intro i hi
        simp [hXdef, hinddef, Set.indicator_of_mem (Finset.mem_filter.1 hi).2]
      rw [Finset.sum_congr rfl hone, Finset.sum_const, nsmul_eq_mul, mul_one]
    have h0 : ∑ i ∈ (Finset.range m).filter (fun i => ¬ (c (j, i) ∈ W)), X i c = 0 :=
      Finset.sum_eq_zero (fun i hi => by
        simp [hXdef, hinddef, Set.indicator_of_notMem (Finset.mem_filter.1 hi).2])
    rw [h1, h0, add_zero]
  refine le_trans (le_trans (measureReal_mono ?_ (measure_ne_top _ _)) hmain) ?_
  · intro c hc
    simp only [Set.mem_setOf_eq] at hc ⊢
    rw [hcount c, Finset.card_range]
    linarith [hc]
  · rw [Finset.card_range]

open scoped Classical in
/-- The upper-tail twin: a set of mass at most `q` is hit at most `m(q + t)` times.  This
is what prices the certification prefixes the family flips too much of, whose set is chosen
by the noise while the draws are not. -/
lemma cert_hits_upper (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    (j : J) (m : ℕ) (W : Set S) (q t : ℝ) (ht : 0 ≤ t)
    (hW : (D j).real W ≤ q) :
    (Measure.infinitePi fun z : J × ℕ => D z.1).real
        {c | (m : ℝ) * (q + t)
          ≤ (((Finset.range m).filter (fun i => c (j, i) ∈ W)).card : ℝ)}
      ≤ Real.exp (-2 * (m : ℝ) * t ^ 2) := by
  classical
  set ν : Measure (J × ℕ → S) := Measure.infinitePi fun z : J × ℕ => D z.1 with hνdef
  have hWm : MeasurableSet W := measurableSet_of_countable W
  set ind : S → ℝ := W.indicator 1 with hinddef
  have hindm : Measurable ind := measurable_const.indicator hWm
  set X : ℕ → (J × ℕ → S) → ℝ := fun i c => ind (c (j, i)) with hXdef
  have hcoord : iIndepFun (fun (z : J × ℕ) (c : J × ℕ → S) => c z) ν :=
    iIndepFun_infinitePi (fun _ => measurable_id)
  have hinj : Function.Injective (fun i : ℕ => (j, i)) := fun a b hab => (Prod.mk.inj hab).2
  have hindep : iIndepFun X ν :=
    (hcoord.precomp hinj).comp (fun _ => ind) (fun _ => hindm)
  have hmeas : ∀ i, AEMeasurable (X i) ν := fun i =>
    (hindm.comp (measurable_pi_apply _)).aemeasurable
  have hicc : ∀ i, ∀ᵐ c ∂ν, X i c ∈ Set.Icc (0 : ℝ) 1 := by
    intro i
    filter_upwards with c
    by_cases h : c (j, i) ∈ W
    · simp [hXdef, hinddef, Set.indicator_of_mem h]
    · simp [hXdef, hinddef, Set.indicator_of_notMem h]
  have hmean : ∀ i, ν[X i] = (D j).real W := by
    intro i
    have hmp : MeasurePreserving (fun c : J × ℕ → S => c (j, i)) ν (D j) :=
      measurePreserving_eval_infinitePi _ (j, i)
    calc ν[X i] = ∫ s, ind s ∂(D j) := by
          rw [← hmp.map_eq,
            integral_map (measurable_pi_apply _).aemeasurable hindm.aestronglyMeasurable]
      _ = (D j).real W := by rw [hinddef, integral_indicator_one hWm]
  have hsum : ∑ i ∈ Finset.range m, ν[X i] ≤ ((Finset.range m).card : ℝ) * q := by
    rw [Finset.sum_congr rfl (fun i _ => hmean i), Finset.sum_const, nsmul_eq_mul,
      Finset.card_range]
    have : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
    nlinarith [hW]
  have hmain := sumUpper_le X (Finset.range m) q t hmeas hindep hicc hsum ht
  have hcount : ∀ c : J × ℕ → S, ∑ i ∈ Finset.range m, X i c
      = (((Finset.range m).filter (fun i => c (j, i) ∈ W)).card : ℝ) := by
    intro c
    rw [← Finset.sum_filter_add_sum_filter_not (Finset.range m) (fun i => c (j, i) ∈ W)]
    have h1 : ∑ i ∈ (Finset.range m).filter (fun i => c (j, i) ∈ W), X i c
        = (((Finset.range m).filter (fun i => c (j, i) ∈ W)).card : ℝ) := by
      have hone : ∀ i ∈ (Finset.range m).filter (fun i => c (j, i) ∈ W), X i c = (1 : ℝ) := by
        intro i hi
        simp [hXdef, hinddef, Set.indicator_of_mem (Finset.mem_filter.1 hi).2]
      rw [Finset.sum_congr rfl hone, Finset.sum_const, nsmul_eq_mul, mul_one]
    have h0 : ∑ i ∈ (Finset.range m).filter (fun i => ¬ (c (j, i) ∈ W)), X i c = 0 :=
      Finset.sum_eq_zero (fun i hi => by
        simp [hXdef, hinddef, Set.indicator_of_notMem (Finset.mem_filter.1 hi).2])
    rw [h1, h0, add_zero]
  refine le_trans (le_trans (measureReal_mono ?_ (measure_ne_top _ _)) hmain) ?_
  · intro c hc
    simp only [Set.mem_setOf_eq] at hc ⊢
    rw [hcount c, Finset.card_range]
    linarith [hc]
  · rw [Finset.card_range]

/-! ### The accept-preserving gate

`AcceptPreservingGate` runs after the FNR test, right before the family is returned.  It
splits the prefixes by the family's *own* cut and counts, on the **seed's own column**, how
many read as accepting — membership of `p · ε` is membership of `p`, which is why the gate
is read off `ε` and why `one_mem_clusterAround` matters.  A family is admitted only when
each side reads as its own class (`drift_verdict`).

The cutoffs are `B.lo` and `B.hi`, and the binomial nulls are the rates they cut at,
`lo/k` and `hi/k`.  `admissibleCut` is exactly the statement that those rates keep the
false positive and false negative budgets. -/

/-- `P[Bin(N,p) ≥ j]` — `scipy.stats.binom.sf(j-1, N, p)`. -/
noncomputable def binomSfGe (N : ℕ) (p : ℝ) (j : ℕ) : ℝ :=
  ∑ i ∈ Finset.Icc j N, (N.choose i : ℝ) * p ^ i * (1 - p) ^ (N - i)

open scoped Classical in
/-- `_split_counts` on the accept side: `(hits, n)` over the prefixes the family accepts,
counted on the seed's column. -/
noncomputable def splitAcc (O : Oracle μ S) (hi : ℕ) (F P : Finset S) (ω : Ω) : ℕ × ℕ :=
  let side := P.filter (fun p => hi - 1 < voteCount O F p ω)
  ((side.filter (fun p => mq O p ω = 1)).card, side.card)

open scoped Classical in
/-- `_split_counts` on the reject side. -/
noncomputable def splitRej (O : Oracle μ S) (lo : ℕ) (F P : Finset S) (ω : Ω) : ℕ × ℕ :=
  let side := P.filter (fun p => voteCount O F p ω ≤ lo)
  ((side.filter (fun p => mq O p ω = 1)).card, side.card)

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

/-- The mirror for the reject side. -/
noncomputable def gateRej (O : Oracle μ S) (εcov : ℝ) : ℝ :=
  O.η + (1 / 2 - O.η) * εcov

/-- `drift_verdict`'s **ADMITTED**: each side of the cut reads as its own class on the
seed's column, at error rate `α` (`ACCEPT_PRESERVING_ERROR_RATE = 0.05`).

The two sides are held to `gateAcc` and `gateRej`, which are derived from the oracle's
signal and the coverage being certified — not configuration.  `ACCEPT_PRESERVING_DRIFT` is
what names them in the code, and tying it to the coverage the caller wants rather than
fixing it is PR #286's remaining item.

The *vote* cutoffs `hi`/`lo` cannot serve as these rates: `hi/k` sits a fixed distance
below `1 − η`, so drift finer than that reads as clean however many prefixes are certified
on — a floor set by the threshold, not by the sample.

Applied in `ret` to the family with `ε` removed and to `certOf`, the certification draws.
Removing `ε` matters because it is in every family, so the vote would otherwise contain
`mq p` — the very bit the split is scored against.  Judging on `certOf` matters because
the family was selected against `prefixesOf`.  (Issue #284.) -/
def admitted (O : Oracle μ S) (lo hi : ℕ) (εcov α : ℝ) (F P : Finset S) (ω : Ω) : Prop :=
  binomSfGe (splitAcc O hi F P ω).2 (gateAcc O εcov) (splitAcc O hi F P ω).1 ≤ α
    ∧ binomCdf (splitRej O lo F P ω).2 (gateRej O εcov) (splitRej O lo F P ω).1 ≤ α

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
    ∧ ∀ j ∈ populations, admitted O B.lo B.hi εcov α
        ((clusterAt O populations x B).erase 1) (certOf j B.m x) (nz x)}

lemma binomSfGe_zero (θ : ℝ) : binomSfGe 0 θ 0 = 1 := by
  simp [binomSfGe]

open scoped Classical in
/-- **The seed alone is never returned.**  The iterate can stall at `{ε}`, whose vote is one
noisy read and whose coverage really is bad — but with the seed dropped the split has an
empty accept side, whose survival function is `1`, and no error rate below `1` admits it. -/
lemma not_ret_of_seed_family (O : Oracle μ S) (populations : Finset J)
    (indecisionLimit εcov α : ℝ) (B : Budget) (hα : α < 1)
    (hpop : populations.Nonempty) (x : Run Ω S J)
    (hfam : clusterAt O populations x B = {(1 : S)}) :
    x ∉ ret O populations indecisionLimit εcov α B := by
  classical
  obtain ⟨j, hj⟩ := hpop
  rintro ⟨-, hadm⟩
  have hacc := (hadm j hj).1
  rw [hfam, Finset.erase_singleton] at hacc
  have hside : (∅ : Finset S).card = 0 := rfl
  have hempty : (certOf j B.m x).filter
      (fun p => B.hi - 1 < voteCount O (∅ : Finset S) p (nz x)) = ∅ := by
    refine Finset.filter_eq_empty_iff.2 (fun p _ => ?_)
    simp [voteCount]
  have hsplit : splitAcc O B.hi (∅ : Finset S) (certOf j B.m x) (nz x) = (0, 0) := by
    simp [splitAcc, hempty]
  rw [hsplit, binomSfGe_zero] at hacc
  exact absurd hacc (not_le.2 hα)

/-- The family at a reachable state is **invalid**: on some population its cut is wrong on
more than an `εcov` fraction. -/
def FailAt (O : Oracle μ S) (populations : Finset J) (D : J → Measure S) (εcov : ℝ)
    (B : Budget) : Set (Run Ω S J) :=
  {x | ¬ ∀ j ∈ populations, 1 - εcov
        ≤ (D j).real {p | cutCorrect O B.lo B.hi (clusterAt O populations x B) p (nz x)}}

open scoped Classical in
/-- **The gate's accept side is sound.**  The prefixes the family accepts read as accepting
on the seed's column at rate `η + (1−2η)·(true accepting fraction)`; so if enough of that
side is *truly rejecting*, the count clears `|A|·accept_thresh` only with probability
`exp(−2|A|τ²)`, where `τ` is the gap the drift opens.

`A` is a fixed `Finset`, so the reads are independent (`mq_indep`) and this is Hoeffding.
Taking `A` to be the family's own accept side needs the union over reachable cuts, which
is the `M^k` cost `validity_of_returned` records. -/
lemma splitAcc_sound (O : Oracle μ S) (A : Finset S) (θ τ : ℝ) (hτ : 0 ≤ τ)
    (hmean : ∑ p ∈ A, (O.η + (1 - 2 * O.η) * O.label p) ≤ (A.card : ℝ) * (θ - τ)) :
    μ.real {ω | (A.card : ℝ) * θ ≤ ((A.filter (fun p => mq O p ω = 1)).card : ℝ)}
      ≤ Real.exp (-2 * (A.card : ℝ) * τ ^ 2) := by
  have hsum := wrongDecisive_le (fun p => mq O p) A (θ - τ) τ
    (fun p => (mq_meas O p).aemeasurable) (mq_indep O) (fun p => mq_icc O p)
    (by rw [Finset.sum_congr rfl (fun p _ => mq_mean O p)]; exact hmean) hτ
  have hsub : {ω | (A.card : ℝ) * θ ≤ ((A.filter (fun p => mq O p ω = 1)).card : ℝ)}
      ≤ᵐ[μ] {ω | (A.card : ℝ) * ((θ - τ) + τ) ≤ ∑ p ∈ A, mq O p ω} := by
    filter_upwards [hits_eq_sum O A] with ω hω hmem
    have hge : (A.card : ℝ) * θ ≤ ∑ p ∈ A, mq O p ω := hω ▸ hmem
    show (A.card : ℝ) * ((θ - τ) + τ) ≤ ∑ p ∈ A, mq O p ω
    have hrw : (A.card : ℝ) * ((θ - τ) + τ) = (A.card : ℝ) * θ := by ring
    rw [hrw]; exact hge
  exact le_trans (ENNReal.toReal_mono (measure_ne_top _ _) (measure_mono_ae hsub)) hsum

open scoped Classical in
/-- **The gate's reject side is sound**, the mirror of `splitAcc_sound`: if enough of the
side the family rejects is *truly accepting*, its accepting-read count falls below
`|R|·reject_thresh` only with probability `exp(−2|R|τ²)`. -/
lemma splitRej_sound (O : Oracle μ S) (R : Finset S) (θ τ : ℝ) (hτ : 0 ≤ τ)
    (hmean : (R.card : ℝ) * (θ + τ) ≤ ∑ p ∈ R, (O.η + (1 - 2 * O.η) * O.label p)) :
    μ.real {ω | ((R.filter (fun p => mq O p ω = 1)).card : ℝ) ≤ (R.card : ℝ) * θ}
      ≤ Real.exp (-2 * (R.card : ℝ) * τ ^ 2) := by
  have hsum := sumLower_le (fun p => mq O p) R (θ + τ) τ
    (fun p => (mq_meas O p).aemeasurable) (mq_indep O) (fun p => mq_icc O p)
    (by rw [Finset.sum_congr rfl (fun p _ => mq_mean O p)]; exact hmean) hτ
  have hsub : {ω | ((R.filter (fun p => mq O p ω = 1)).card : ℝ) ≤ (R.card : ℝ) * θ}
      ≤ᵐ[μ] {ω | ∑ p ∈ R, mq O p ω ≤ (R.card : ℝ) * ((θ + τ) - τ)} := by
    filter_upwards [hits_eq_sum O R] with ω hω hmem
    have hle : ∑ p ∈ R, mq O p ω ≤ (R.card : ℝ) * θ := hω ▸ hmem
    show ∑ p ∈ R, mq O p ω ≤ (R.card : ℝ) * ((θ + τ) - τ)
    have hrw : (R.card : ℝ) * ((θ + τ) - τ) = (R.card : ℝ) * θ := by ring
    rw [hrw]; exact hle
  exact le_trans (ENNReal.toReal_mono (measure_ne_top _ _) (measure_mono_ae hsub)) hsum

open scoped Classical in
/-- **The gate cannot admit a drifted accept side.**  If a `γ` fraction of the prefixes the
family accepts are truly rejecting, then by `mq_mean` the side's accepting-read rate sits
`γ(1−2η)` below the clean `1−η`, and the count clears `|A|·θ` only with probability
`exp(−2·c₀·τ²)` for `τ = θ − (1−η) + γ(1−2η)`, the gap the drift opens. -/
theorem gate_accept_sound (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    (side : Ω → Finset S) (hside : ∀ ω, side ω ⊆ C)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → side ω = side ω')
    (θ γ c₀ : ℝ) (hc₀ : 0 ≤ c₀) (hγ : 0 ≤ γ)
    (hτ : 0 ≤ θ - (1 - O.η) + γ * (1 - 2 * O.η)) (hsig : O.η ≤ 1 / 2) :
    μ.real {ω | c₀ ≤ ((side ω).card : ℝ)
        ∧ γ * ((side ω).card : ℝ) ≤ (((side ω).filter (fun p => O.label p = 0)).card : ℝ)
        ∧ ((side ω).card : ℝ) * θ ≤ (((side ω).filter (fun p => mq O p ω = 1)).card : ℝ)}
      ≤ Real.exp (-2 * c₀ * (θ - (1 - O.η) + γ * (1 - 2 * O.η)) ^ 2) := by
  classical
  set τ : ℝ := θ - (1 - O.η) + γ * (1 - 2 * O.η) with hτdef
  refine gate_side_bound O C Q hdisj side hside hcongr
    (fun A₀ T => c₀ ≤ (A₀.card : ℝ)
      ∧ γ * (A₀.card : ℝ) ≤ ((A₀.filter (fun p => O.label p = 0)).card : ℝ)
      ∧ (A₀.card : ℝ) * θ ≤ (T.card : ℝ)) _ (Real.exp_pos _).le (fun A₀ _ => ?_)
  by_cases hbig : c₀ ≤ (A₀.card : ℝ) ∧
      γ * (A₀.card : ℝ) ≤ ((A₀.filter (fun p => O.label p = 0)).card : ℝ)
  · have hmean : ∑ p ∈ A₀, (O.η + (1 - 2 * O.η) * O.label p) ≤ (A₀.card : ℝ) * (θ - τ) := by
      rw [Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum,
        sum_label_eq O A₀]
      have h2 : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith
      nlinarith [hbig.2, hτdef]
    refine le_trans (le_trans (measureReal_mono ?_ (measure_ne_top _ _))
      (splitAcc_sound O A₀ θ τ hτ hmean)) ?_
    · exact fun ω hω => hω.2.2
    · refine Real.exp_le_exp.2 ?_
      have : c₀ ≤ (A₀.card : ℝ) := hbig.1
      nlinarith [sq_nonneg τ]
  · have hempty : {ω | c₀ ≤ (A₀.card : ℝ)
        ∧ γ * (A₀.card : ℝ) ≤ ((A₀.filter (fun p => O.label p = 0)).card : ℝ)
        ∧ (A₀.card : ℝ) * θ ≤ (((A₀.filter (fun p => mq O p ω = 1))).card : ℝ)} = ∅ := by
      ext ω
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
      exact fun h => hbig ⟨h.1, h.2.1⟩
    rw [hempty]
    simpa using (Real.exp_pos _).le

open scoped Classical in
/-- **The gate cannot admit a drifted reject side**, the mirror of `gate_accept_sound`: a
`γ` fraction of truly-accepting prefixes on the rejected side lifts its accepting-read rate
`γ(1−2η)` above the clean `η`, and the count stays below `|R|·θ` only with probability
`exp(−2·c₀·τ²)` for `τ = η + γ(1−2η) − θ`. -/
theorem gate_reject_sound (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    (side : Ω → Finset S) (hside : ∀ ω, side ω ⊆ C)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → side ω = side ω')
    (θ γ c₀ : ℝ) (hc₀ : 0 ≤ c₀) (hγ : 0 ≤ γ)
    (hτ : 0 ≤ O.η + γ * (1 - 2 * O.η) - θ) (hsig : O.η ≤ 1 / 2) :
    μ.real {ω | c₀ ≤ ((side ω).card : ℝ)
        ∧ γ * ((side ω).card : ℝ) ≤ ∑ p ∈ side ω, O.label p
        ∧ (((side ω).filter (fun p => mq O p ω = 1)).card : ℝ) ≤ ((side ω).card : ℝ) * θ}
      ≤ Real.exp (-2 * c₀ * (O.η + γ * (1 - 2 * O.η) - θ) ^ 2) := by
  classical
  set τ : ℝ := O.η + γ * (1 - 2 * O.η) - θ with hτdef
  refine gate_side_bound O C Q hdisj side hside hcongr
    (fun A₀ T => c₀ ≤ (A₀.card : ℝ)
      ∧ γ * (A₀.card : ℝ) ≤ ∑ p ∈ A₀, O.label p
      ∧ (T.card : ℝ) ≤ (A₀.card : ℝ) * θ) _ (Real.exp_pos _).le (fun A₀ _ => ?_)
  by_cases hbig : c₀ ≤ (A₀.card : ℝ) ∧ γ * (A₀.card : ℝ) ≤ ∑ p ∈ A₀, O.label p
  · have hmean : (A₀.card : ℝ) * (θ + τ) ≤ ∑ p ∈ A₀, (O.η + (1 - 2 * O.η) * O.label p) := by
      rw [Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum]
      have h2 : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith
      nlinarith [hbig.2, hτdef]
    refine le_trans (le_trans (measureReal_mono ?_ (measure_ne_top _ _))
      (splitRej_sound O A₀ θ τ hτ hmean)) ?_
    · exact fun ω hω => hω.2.2
    · refine Real.exp_le_exp.2 ?_
      have : c₀ ≤ (A₀.card : ℝ) := hbig.1
      nlinarith [sq_nonneg τ]
  · have hempty : {ω | c₀ ≤ (A₀.card : ℝ)
        ∧ γ * (A₀.card : ℝ) ≤ ∑ p ∈ A₀, O.label p
        ∧ (((A₀.filter (fun p => mq O p ω = 1))).card : ℝ) ≤ (A₀.card : ℝ) * θ} = ∅ := by
      ext ω
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
      exact fun h => hbig ⟨h.1, h.2.1⟩
    rw [hempty]
    simpa using (Real.exp_pos _).le

/-! ### What the gate sees when the cut is wrong

A prefix the cut gets wrong is *decided*, and decided against its label — `cutCorrect` is
vacuous on the indecisive band, so a failure is always a confident mistake.  It therefore
lands on one of the two sides the gate scores, carrying the wrong label with it. -/

/-- A wrong prefix sits on the accept side with label `0`, or the reject side with label
`1`. -/
lemma wrong_mem_side (O : Oracle μ S) (lo hi : ℕ) (F : Finset S) (p : S) (ω : Ω)
    (h : ¬ cutCorrect O lo hi F p ω) :
    (hi < voteCount O F p ω ∧ O.label p = 0) ∨ (voteCount O F p ω ≤ lo ∧ O.label p = 1) := by
  unfold cutCorrect at h
  rw [not_and_or] at h
  rcases h with h | h
  · rw [Classical.not_imp] at h
    refine Or.inl ⟨h.1, ?_⟩
    rcases O.label_bit p with hl | hl
    · exact hl
    · exact absurd hl h.2
  · rw [Classical.not_imp] at h
    refine Or.inr ⟨h.1, ?_⟩
    rcases O.label_bit p with hl | hl
    · exact absurd hl h.2
    · exact hl

open scoped Classical in
/-- The wrong prefixes on the accept side: the gate scores these as accepting, and they are
not. -/
noncomputable def wrongAcc (O : Oracle μ S) (hi : ℕ) (F P : Finset S) (ω : Ω) : Finset S :=
  P.filter (fun p => hi < voteCount O F p ω ∧ O.label p = 0)

open scoped Classical in
/-- The wrong prefixes on the reject side. -/
noncomputable def wrongRej (O : Oracle μ S) (lo : ℕ) (F P : Finset S) (ω : Ω) : Finset S :=
  P.filter (fun p => voteCount O F p ω ≤ lo ∧ O.label p = 1)

open scoped Classical in
lemma wrongAcc_subset (O : Oracle μ S) (hi : ℕ) (F P : Finset S) (ω : Ω) :
    wrongAcc O hi F P ω ⊆ P.filter (fun p => hi < voteCount O F p ω) :=
  fun p hp => by
    obtain ⟨hpP, hlt, -⟩ := Finset.mem_filter.1 hp
    exact Finset.mem_filter.2 ⟨hpP, hlt⟩

open scoped Classical in
lemma wrongRej_subset (O : Oracle μ S) (lo : ℕ) (F P : Finset S) (ω : Ω) :
    wrongRej O lo F P ω ⊆ P.filter (fun p => voteCount O F p ω ≤ lo) :=
  fun p hp => by
    obtain ⟨hpP, hle, -⟩ := Finset.mem_filter.1 hp
    exact Finset.mem_filter.2 ⟨hpP, hle⟩

open scoped Classical in
/-- Every wrong prefix is counted on one side or the other. -/
lemma card_wrong_le (O : Oracle μ S) (lo hi : ℕ) (F P : Finset S) (ω : Ω) :
    (P.filter (fun p => ¬ cutCorrect O lo hi F p ω)).card
      ≤ (wrongAcc O hi F P ω).card + (wrongRej O lo F P ω).card := by
  classical
  refine le_trans (Finset.card_le_card ?_) (Finset.card_union_le _ _)
  intro p hp
  obtain ⟨hpP, hbad⟩ := Finset.mem_filter.1 hp
  rcases wrong_mem_side O lo hi F p ω hbad with h | h
  · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpP, h⟩)
  · exact Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hpP, h⟩)

open scoped Classical in
/-- **The accept side cannot read as accepting when enough of it is truly rejecting.**
`splitAcc_sound` with the drift expressed as a count of wrong prefixes. -/
lemma splitAcc_sound_of_wrong (O : Oracle μ S) (A : Finset S) (θ τ w : ℝ) (hτ : 0 ≤ τ)
    (hsig : O.η ≤ 1 / 2) (hw : w ≤ ((A.filter (fun p => O.label p = 0)).card : ℝ))
    (hθ : (A.card : ℝ) * (1 - O.η) - (1 - 2 * O.η) * w ≤ (A.card : ℝ) * (θ - τ)) :
    μ.real {ω | (A.card : ℝ) * θ ≤ ((A.filter (fun p => mq O p ω = 1)).card : ℝ)}
      ≤ Real.exp (-2 * (A.card : ℝ) * τ ^ 2) := by
  classical
  refine splitAcc_sound O A θ τ hτ ?_
  have hsum : ∑ p ∈ A, (O.η + (1 - 2 * O.η) * O.label p)
      = (A.card : ℝ) * (1 - O.η)
        - (1 - 2 * O.η) * ((A.filter (fun p => O.label p = 0)).card : ℝ) := by
    rw [Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum,
      sum_label_eq O A]
    ring
  rw [hsum]
  have h2 : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith
  nlinarith [hw, hθ]

open scoped Classical in
/-- The mirror: the reject side cannot read as rejecting when enough of it is truly
accepting. -/
lemma splitRej_sound_of_wrong (O : Oracle μ S) (R : Finset S) (θ τ w : ℝ) (hτ : 0 ≤ τ)
    (hsig : O.η ≤ 1 / 2) (hw : w ≤ ((R.filter (fun p => ¬ (O.label p = 0))).card : ℝ))
    (hθ : (R.card : ℝ) * (θ + τ) ≤ (R.card : ℝ) * O.η + (1 - 2 * O.η) * w) :
    μ.real {ω | ((R.filter (fun p => mq O p ω = 1)).card : ℝ) ≤ (R.card : ℝ) * θ}
      ≤ Real.exp (-2 * (R.card : ℝ) * τ ^ 2) := by
  classical
  refine splitRej_sound O R θ τ hτ ?_
  have hsum : ∑ p ∈ R, (O.η + (1 - 2 * O.η) * O.label p)
      = (R.card : ℝ) * O.η
        + (1 - 2 * O.η) * ((R.filter (fun p => ¬ (O.label p = 0))).card : ℝ) := by
    rw [Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum,
      sum_label_eq O R]
    have hcard : ((R.filter (fun p => O.label p = 0)).card : ℝ)
        + ((R.filter (fun p => ¬ (O.label p = 0))).card : ℝ) = (R.card : ℝ) := by
      exact_mod_cast congrArg (fun n : ℕ => (n : ℝ))
        (Finset.card_filter_add_card_filter_not (s := R) (fun p => O.label p = 0))
    nlinarith [hcard]
  rw [hsum]
  have h2 : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith
  nlinarith [hw, hθ]

lemma voteCount_mono (O : Oracle μ S) {F F' : Finset S} (h : F ⊆ F') (p : S) (ω : Ω) :
    voteCount O F p ω ≤ voteCount O F' p ω := by
  classical
  exact Finset.card_le_card (Finset.filter_subset_filter _ h)

open scoped Classical in
/-- **The gate sees every prefix the cut gets wrong.**  A wrong prefix is decided against
its label, and with the accept side shifted by the seed's own vote it lands on the side the
gate scores.  Without the shift the gate would be blind to exactly the prefixes the seed's
own misread pushed over the line. -/
lemma wrong_mem_gate_side (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (hhi : 1 ≤ B.hi) (x : Run Ω S J) {p : S} (hp : p ∈ certOf j B.m x)
    (h : ¬ cutCorrect O B.lo B.hi (clusterAt O populations x B) p (nz x)) :
    (p ∈ sideAcc O populations j B x ∧ O.label p = 0)
      ∨ (p ∈ sideRej O populations j B x ∧ O.label p = 1) := by
  classical
  rcases wrong_mem_side O B.lo B.hi (clusterAt O populations x B) p (nz x) h with ⟨hlt, hl⟩ | ⟨hle, hl⟩
  · refine Or.inl ⟨Finset.mem_filter.2 ⟨hp, ?_⟩, hl⟩
    have hstep := voteCount_le_erase_succ O (clusterAt O populations x B) p (nz x)
    omega
  · refine Or.inr ⟨Finset.mem_filter.2 ⟨hp, ?_⟩, hl⟩
    exact le_trans (voteCount_mono O (Finset.erase_subset _ _) p (nz x)) hle

open scoped Classical in
/-- Every wrong certification prefix is counted against one of the gate's two sides. -/
lemma card_cert_wrong_le (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (hhi : 1 ≤ B.hi) (x : Run Ω S J) :
    ((certOf j B.m x).filter
        (fun p => ¬ cutCorrect O B.lo B.hi (clusterAt O populations x B) p (nz x))).card
      ≤ ((sideAcc O populations j B x).filter (fun p => O.label p = 0)).card
        + ((sideRej O populations j B x).filter (fun p => ¬ (O.label p = 0))).card := by
  classical
  refine le_trans (Finset.card_le_card ?_) (Finset.card_union_le _ _)
  intro p hp
  obtain ⟨hpC, hbad⟩ := Finset.mem_filter.1 hp
  rcases wrong_mem_gate_side O populations j B hhi x hpC hbad with ⟨hs, hl⟩ | ⟨hs, hl⟩
  · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hs, hl⟩)
  · refine Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hs, ?_⟩)
    rw [hl]; norm_num

open scoped Classical in
/-- **The gate cannot admit an accept side that is `w`-wrong.**  `splitAcc_sound_of_wrong`
for the `ω`-dependent side the run actually produces. -/
theorem gate_acc_side_bound (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    (side : Ω → Finset S) (hside : ∀ ω, side ω ⊆ C)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → side ω = side ω')
    (θacc τ w : ℝ) (n₀ : ℕ) (hτ : 0 ≤ τ) (hsig : O.η ≤ 1 / 2)
    (hθ : ∀ n : ℕ, n₀ ≤ n → n ≤ C.card →
      (n : ℝ) * (1 - O.η) - (1 - 2 * O.η) * w ≤ (n : ℝ) * (θacc - τ)) :
    μ.real {ω | n₀ ≤ (side ω).card
        ∧ w ≤ (((side ω).filter (fun p => O.label p = 0)).card : ℝ)
        ∧ ((side ω).card : ℝ) * θacc ≤ (((side ω).filter (fun p => mq O p ω = 1)).card : ℝ)}
      ≤ Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := by
  classical
  refine gate_side_bound O C Q hdisj side hside hcongr
    (fun A₀ U => n₀ ≤ A₀.card ∧ w ≤ ((A₀.filter (fun p => O.label p = 0)).card : ℝ)
      ∧ (A₀.card : ℝ) * θacc ≤ (U.card : ℝ)) _ (Real.exp_nonneg _) ?_
  intro A₀ hA₀
  by_cases hn : n₀ ≤ A₀.card
  · by_cases hw : w ≤ ((A₀.filter (fun p => O.label p = 0)).card : ℝ)
    · refine le_trans (measureReal_mono (fun ω hω => hω.2.2) (measure_ne_top _ _))
        (le_trans (splitAcc_sound_of_wrong O A₀ θacc τ w hτ hsig hw
          (hθ A₀.card hn (Finset.card_le_card (Finset.mem_powerset.1 hA₀)))) ?_)
      refine Real.exp_le_exp.2 ?_
      have hc : (n₀ : ℝ) ≤ (A₀.card : ℝ) := by exact_mod_cast hn
      nlinarith [sq_nonneg τ]
    · have hz : {ω : Ω | n₀ ≤ A₀.card ∧ w ≤ ((A₀.filter (fun p => O.label p = 0)).card : ℝ)
          ∧ (A₀.card : ℝ) * θacc
            ≤ (((A₀.filter (fun p => mq O p ω = 1)).card : ℝ))} = (∅ : Set Ω) := by
        ext ω; simp [hw]
      rw [hz]; simpa using Real.exp_nonneg _
  · have hz : {ω : Ω | n₀ ≤ A₀.card ∧ w ≤ ((A₀.filter (fun p => O.label p = 0)).card : ℝ)
        ∧ (A₀.card : ℝ) * θacc
          ≤ (((A₀.filter (fun p => mq O p ω = 1)).card : ℝ))} = (∅ : Set Ω) := by
      ext ω; simp [hn]
    rw [hz]; simpa using Real.exp_nonneg _

open scoped Classical in
/-- The mirror for the reject side. -/
theorem gate_rej_side_bound (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    (side : Ω → Finset S) (hside : ∀ ω, side ω ⊆ C)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → side ω = side ω')
    (θrej τ w : ℝ) (n₀ : ℕ) (hτ : 0 ≤ τ) (hsig : O.η ≤ 1 / 2)
    (hθ : ∀ n : ℕ, n₀ ≤ n → n ≤ C.card →
      (n : ℝ) * (θrej + τ) ≤ (n : ℝ) * O.η + (1 - 2 * O.η) * w) :
    μ.real {ω | n₀ ≤ (side ω).card
        ∧ w ≤ (((side ω).filter (fun p => ¬ (O.label p = 0))).card : ℝ)
        ∧ (((side ω).filter (fun p => mq O p ω = 1)).card : ℝ) ≤ ((side ω).card : ℝ) * θrej}
      ≤ Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := by
  classical
  refine gate_side_bound O C Q hdisj side hside hcongr
    (fun A₀ U => n₀ ≤ A₀.card ∧ w ≤ ((A₀.filter (fun p => ¬ (O.label p = 0))).card : ℝ)
      ∧ (U.card : ℝ) ≤ (A₀.card : ℝ) * θrej) _ (Real.exp_nonneg _) ?_
  intro A₀ hA₀
  by_cases hn : n₀ ≤ A₀.card
  · by_cases hw : w ≤ ((A₀.filter (fun p => ¬ (O.label p = 0))).card : ℝ)
    · refine le_trans (measureReal_mono (fun ω hω => hω.2.2) (measure_ne_top _ _))
        (le_trans (splitRej_sound_of_wrong O A₀ θrej τ w hτ hsig hw
          (hθ A₀.card hn (Finset.card_le_card (Finset.mem_powerset.1 hA₀)))) ?_)
      refine Real.exp_le_exp.2 ?_
      have hc : (n₀ : ℝ) ≤ (A₀.card : ℝ) := by exact_mod_cast hn
      nlinarith [sq_nonneg τ]
    · have hz : {ω : Ω | n₀ ≤ A₀.card
          ∧ w ≤ ((A₀.filter (fun p => ¬ (O.label p = 0))).card : ℝ)
          ∧ (((A₀.filter (fun p => mq O p ω = 1)).card : ℝ))
            ≤ (A₀.card : ℝ) * θrej} = (∅ : Set Ω) := by
        ext ω; simp [hw]
      rw [hz]; simpa using Real.exp_nonneg _
  · have hz : {ω : Ω | n₀ ≤ A₀.card
        ∧ w ≤ ((A₀.filter (fun p => ¬ (O.label p = 0))).card : ℝ)
        ∧ (((A₀.filter (fun p => mq O p ω = 1)).card : ℝ))
          ≤ (A₀.card : ℝ) * θrej} = (∅ : Set Ω) := by
      ext ω; simp [hn]
    rw [hz]; simpa using Real.exp_nonneg _

open scoped Classical in
/-- The mirror: a mostly-rejecting side reads accepting rarely enough. -/
theorem gate_rej_admit_bound' (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    (side : Ω → Finset S) (hside : ∀ ω, side ω ⊆ C)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → side ω = side ω')
    (θ τ w : ℝ) (n₀ : ℕ) (hτ : 0 ≤ τ) (hsig : O.η ≤ 1 / 2)
    (hθ : ∀ n : ℕ, n₀ ≤ n → n ≤ C.card →
      (n : ℝ) * O.η + (1 - 2 * O.η) * w ≤ (n : ℝ) * (θ - τ)) :
    μ.real {ω | n₀ ≤ (side ω).card
        ∧ ((((side ω).filter (fun p => ¬ (O.label p = 0))).card : ℝ) ≤ w)
        ∧ ((side ω).card : ℝ) * θ ≤ (((side ω).filter (fun p => mq O p ω = 1)).card : ℝ)}
      ≤ Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := by
  classical
  refine gate_side_bound O C Q hdisj side hside hcongr
    (fun A₀ U => n₀ ≤ A₀.card
      ∧ (((A₀.filter (fun p => ¬ (O.label p = 0))).card : ℝ) ≤ w)
      ∧ (A₀.card : ℝ) * θ ≤ (U.card : ℝ)) _ (Real.exp_nonneg _) ?_
  intro A₀ hA₀
  by_cases hn : n₀ ≤ A₀.card
  · by_cases hw : (((A₀.filter (fun p => ¬ (O.label p = 0))).card : ℝ) ≤ w)
    · have hsplit : (((A₀.filter (fun p => O.label p = 0)).card : ℝ))
          + (((A₀.filter (fun p => ¬ (O.label p = 0))).card : ℝ)) = (A₀.card : ℝ) := by
        exact_mod_cast congrArg (fun n : ℕ => (n : ℝ))
          (Finset.card_filter_add_card_filter_not (s := A₀) (fun p => O.label p = 0))
      have hmean : (A₀.card : ℝ) * (1 - O.η)
          - (1 - 2 * O.η) * (((A₀.filter (fun p => O.label p = 0)).card : ℝ))
            ≤ (A₀.card : ℝ) * (θ - τ) := by
        have h2 : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith
        have := hθ A₀.card hn (Finset.card_le_card (Finset.mem_powerset.1 hA₀))
        nlinarith [hsplit, hw]
      refine le_trans (measureReal_mono (fun ω hω => hω.2.2) (measure_ne_top _ _))
        (le_trans (splitAcc_sound_of_wrong O A₀ θ τ _ hτ hsig le_rfl hmean) ?_)
      refine Real.exp_le_exp.2 ?_
      have hc : (n₀ : ℝ) ≤ (A₀.card : ℝ) := by exact_mod_cast hn
      nlinarith [sq_nonneg τ]
    · have hz : {ω : Ω | n₀ ≤ A₀.card
          ∧ (((A₀.filter (fun p => ¬ (O.label p = 0))).card : ℝ) ≤ w)
          ∧ (A₀.card : ℝ) * θ
            ≤ (((A₀.filter (fun p => mq O p ω = 1)).card : ℝ))} = (∅ : Set Ω) := by
        ext ω; simp [hw]
      rw [hz]; simpa using Real.exp_nonneg _
  · have hz : {ω : Ω | n₀ ≤ A₀.card
        ∧ (((A₀.filter (fun p => ¬ (O.label p = 0))).card : ℝ) ≤ w)
        ∧ (A₀.card : ℝ) * θ
          ≤ (((A₀.filter (fun p => mq O p ω = 1)).card : ℝ))} = (∅ : Set Ω) := by
      ext ω; simp [hn]
    rw [hz]; simpa using Real.exp_nonneg _

open scoped Classical in
/-- **A side member carrying the wrong label is a mis-cut prefix.**  So one count — how
often the cut is wrong on the certification sample — bounds the wrong-member budget on both
sides at once. -/
lemma sideAcc_wrong_subset (O : Oracle μ S) (lo ha : ℕ) (F C : Finset S) (ω : Ω) :
    ((C.filter (fun p => ha < voteCount O F p ω)).filter (fun p => ¬ (O.label p = 1)))
      ⊆ C.filter (fun p => ¬ cutCorrect O lo ha F p ω) := by
  classical
  intro p hp
  obtain ⟨hpS, hl⟩ := Finset.mem_filter.1 hp
  obtain ⟨hpC, hlt⟩ := Finset.mem_filter.1 hpS
  exact Finset.mem_filter.2 ⟨hpC, fun hc => hl (hc.1 hlt)⟩

open scoped Classical in
lemma sideRej_wrong_subset (O : Oracle μ S) (lo ha : ℕ) (F C : Finset S) (ω : Ω) :
    ((C.filter (fun p => voteCount O F p ω ≤ lo)).filter (fun p => ¬ (O.label p = 0)))
      ⊆ C.filter (fun p => ¬ cutCorrect O lo ha F p ω) := by
  classical
  intro p hp
  obtain ⟨hpS, hl⟩ := Finset.mem_filter.1 hp
  obtain ⟨hpC, hle⟩ := Finset.mem_filter.1 hpS
  exact Finset.mem_filter.2 ⟨hpC, fun hc => hl (hc.2 hle)⟩

open scoped Classical in
/-- **A correct cut has clean sides.**  Each side is carved out by the very implication
`cutCorrect` asserts, so there is nothing to prove beyond unfolding — but it is what turns
the cut being right into the hypothesis the admission bounds want. -/
lemma sides_clean (O : Oracle μ S) (lo ha : ℕ) (F P : Finset S) (ω : Ω)
    (h : ∀ p ∈ P, cutCorrect O lo ha F p ω) :
    (∀ p ∈ P.filter (fun p => ha < voteCount O F p ω), O.label p = 1)
      ∧ (∀ p ∈ P.filter (fun p => voteCount O F p ω ≤ lo), O.label p = 0) := by
  classical
  constructor
  · intro p hp
    obtain ⟨hpP, hlt⟩ := Finset.mem_filter.1 hp
    exact (h p hpP).1 hlt
  · intro p hp
    obtain ⟨hpP, hle⟩ := Finset.mem_filter.1 hp
    exact (h p hpP).2 hle

open scoped Classical in
/-- The cut is right at every prefix of a finite set, except on the union of the per-prefix
failures.  `certOf` is a function of the draws alone, so at a fixed table this is a plain
finite union. -/
lemma cutRight_all_whp (O : Oracle μ S) (P : Finset S) (lo ha : ℕ) (E : ℝ) (hE : 0 ≤ E)
    (F : Ω → Finset S)
    (hper : ∀ p ∈ P, μ.real {ω | ¬ cutCorrect O lo ha (F ω) p ω} ≤ E) :
    μ.real {ω | ¬ ∀ p ∈ P, cutCorrect O lo ha (F ω) p ω} ≤ (P.card : ℝ) * E := by
  classical
  have hsub : {ω | ¬ ∀ p ∈ P, cutCorrect O lo ha (F ω) p ω}
      ⊆ ⋃ p ∈ P, {ω | ¬ cutCorrect O lo ha (F ω) p ω} := by
    intro ω hω
    simp only [Set.mem_setOf_eq, not_forall] at hω
    obtain ⟨p, hp, hbad⟩ := hω
    exact Set.mem_biUnion hp hbad
  calc μ.real {ω | ¬ ∀ p ∈ P, cutCorrect O lo ha (F ω) p ω}
      ≤ μ.real (⋃ p ∈ P, {ω | ¬ cutCorrect O lo ha (F ω) p ω}) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ p ∈ P, μ.real {ω | ¬ cutCorrect O lo ha (F ω) p ω} :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _p ∈ P, E := Finset.sum_le_sum hper
    _ = (P.card : ℝ) * E := by rw [Finset.sum_const, nsmul_eq_mul]

open scoped Classical in
/-- **A truly-accepting side reads accepting often enough to clear the test.**  The mirror
of `gate_acc_side_bound`: there the side is wrong and must fail, here it is right and must
pass. -/
theorem gate_acc_admit_bound (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    (side : Ω → Finset S) (hside : ∀ ω, side ω ⊆ C)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → side ω = side ω')
    (θ τ : ℝ) (n₀ : ℕ) (hτ : 0 ≤ τ) (hθ : θ + τ ≤ 1 - O.η) :
    μ.real {ω | n₀ ≤ (side ω).card ∧ (∀ p ∈ side ω, O.label p = 1)
        ∧ (((side ω).filter (fun p => mq O p ω = 1)).card : ℝ) ≤ ((side ω).card : ℝ) * θ}
      ≤ Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := by
  classical
  refine gate_side_bound O C Q hdisj side hside hcongr
    (fun A₀ U => n₀ ≤ A₀.card ∧ (∀ p ∈ A₀, O.label p = 1)
      ∧ (U.card : ℝ) ≤ (A₀.card : ℝ) * θ) _ (Real.exp_nonneg _) ?_
  intro A₀ hA₀
  by_cases hn : n₀ ≤ A₀.card
  · by_cases hcl : ∀ p ∈ A₀, O.label p = 1
    · have hmean : (A₀.card : ℝ) * (θ + τ)
          ≤ ∑ p ∈ A₀, (O.η + (1 - 2 * O.η) * O.label p) := by
        rw [Finset.sum_congr rfl (fun p hp => by rw [hcl p hp]), Finset.sum_const, nsmul_eq_mul]
        have : O.η + (1 - 2 * O.η) * 1 = 1 - O.η := by ring
        rw [this]
        exact mul_le_mul_of_nonneg_left hθ (Nat.cast_nonneg _)
      refine le_trans (measureReal_mono (fun ω hω => hω.2.2) (measure_ne_top _ _))
        (le_trans (splitRej_sound O A₀ θ τ hτ hmean) ?_)
      refine Real.exp_le_exp.2 ?_
      have hc : (n₀ : ℝ) ≤ (A₀.card : ℝ) := by exact_mod_cast hn
      nlinarith [sq_nonneg τ]
    · have hz : {ω : Ω | n₀ ≤ A₀.card ∧ (∀ p ∈ A₀, O.label p = 1)
          ∧ (((A₀.filter (fun p => mq O p ω = 1)).card : ℝ))
            ≤ (A₀.card : ℝ) * θ} = (∅ : Set Ω) := by
        ext ω; simp [hcl]
      rw [hz]; simpa using Real.exp_nonneg _
  · have hz : {ω : Ω | n₀ ≤ A₀.card ∧ (∀ p ∈ A₀, O.label p = 1)
        ∧ (((A₀.filter (fun p => mq O p ω = 1)).card : ℝ))
          ≤ (A₀.card : ℝ) * θ} = (∅ : Set Ω) := by
      ext ω; simp [hn]
    rw [hz]; simpa using Real.exp_nonneg _

open scoped Classical in
/-- **A mostly-accepting side still reads accepting often enough.**  `gate_acc_admit_bound`
with a budget `w` of prefixes on the side that are not truly accepting, so the cut only has
to be right on a *fraction* of the certification sample rather than all of it. -/
theorem gate_acc_admit_bound' (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    (side : Ω → Finset S) (hside : ∀ ω, side ω ⊆ C)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → side ω = side ω')
    (θ τ w : ℝ) (n₀ : ℕ) (hτ : 0 ≤ τ) (hsig : O.η ≤ 1 / 2)
    (hθ : ∀ n : ℕ, n₀ ≤ n → n ≤ C.card →
      (n : ℝ) * (θ + τ) ≤ (n : ℝ) * (1 - O.η) - (1 - 2 * O.η) * w) :
    μ.real {ω | n₀ ≤ (side ω).card
        ∧ ((((side ω).filter (fun p => ¬ (O.label p = 1))).card : ℝ) ≤ w)
        ∧ (((side ω).filter (fun p => mq O p ω = 1)).card : ℝ) ≤ ((side ω).card : ℝ) * θ}
      ≤ Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := by
  classical
  refine gate_side_bound O C Q hdisj side hside hcongr
    (fun A₀ U => n₀ ≤ A₀.card
      ∧ (((A₀.filter (fun p => ¬ (O.label p = 1))).card : ℝ) ≤ w)
      ∧ (U.card : ℝ) ≤ (A₀.card : ℝ) * θ) _ (Real.exp_nonneg _) ?_
  intro A₀ hA₀
  by_cases hn : n₀ ≤ A₀.card
  · by_cases hw : (((A₀.filter (fun p => ¬ (O.label p = 1))).card : ℝ) ≤ w)
    · have hsplit : (((A₀.filter (fun p => ¬ (O.label p = 0))).card : ℝ))
          + (((A₀.filter (fun p => ¬ (O.label p = 1))).card : ℝ)) = (A₀.card : ℝ) := by
        have hone : ∀ p ∈ A₀, (¬ (O.label p = 0)) ↔ (O.label p = 1) := by
          intro p _
          rcases O.label_bit p with h | h <;> simp [h]
        rw [Finset.filter_congr hone]
        exact_mod_cast congrArg (fun n : ℕ => (n : ℝ))
          (Finset.card_filter_add_card_filter_not (s := A₀) (fun p => O.label p = 1))
      have hmean : (A₀.card : ℝ) * (θ + τ)
          ≤ (A₀.card : ℝ) * O.η
            + (1 - 2 * O.η) * (((A₀.filter (fun p => ¬ (O.label p = 0))).card : ℝ)) := by
        have h2 : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith
        have := hθ A₀.card hn (Finset.card_le_card (Finset.mem_powerset.1 hA₀))
        nlinarith [hsplit, hw]
      refine le_trans (measureReal_mono (fun ω hω => hω.2.2) (measure_ne_top _ _))
        (le_trans (splitRej_sound_of_wrong O A₀ θ τ _ hτ hsig le_rfl hmean) ?_)
      refine Real.exp_le_exp.2 ?_
      have hc : (n₀ : ℝ) ≤ (A₀.card : ℝ) := by exact_mod_cast hn
      nlinarith [sq_nonneg τ]
    · have hz : {ω : Ω | n₀ ≤ A₀.card
          ∧ (((A₀.filter (fun p => ¬ (O.label p = 1))).card : ℝ) ≤ w)
          ∧ (((A₀.filter (fun p => mq O p ω = 1)).card : ℝ))
            ≤ (A₀.card : ℝ) * θ} = (∅ : Set Ω) := by
        ext ω; simp [hw]
      rw [hz]; simpa using Real.exp_nonneg _
  · have hz : {ω : Ω | n₀ ≤ A₀.card
        ∧ (((A₀.filter (fun p => ¬ (O.label p = 1))).card : ℝ) ≤ w)
        ∧ (((A₀.filter (fun p => mq O p ω = 1)).card : ℝ))
          ≤ (A₀.card : ℝ) * θ} = (∅ : Set Ω) := by
      ext ω; simp [hn]
    rw [hz]; simpa using Real.exp_nonneg _

open scoped Classical in
/-- The mirror: a truly-rejecting side reads accepting rarely enough to clear its test. -/
theorem gate_rej_admit_bound (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    (side : Ω → Finset S) (hside : ∀ ω, side ω ⊆ C)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → side ω = side ω')
    (θ τ : ℝ) (n₀ : ℕ) (hτ : 0 ≤ τ) (hθ : O.η ≤ θ - τ) :
    μ.real {ω | n₀ ≤ (side ω).card ∧ (∀ p ∈ side ω, O.label p = 0)
        ∧ ((side ω).card : ℝ) * θ ≤ (((side ω).filter (fun p => mq O p ω = 1)).card : ℝ)}
      ≤ Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := by
  classical
  refine gate_side_bound O C Q hdisj side hside hcongr
    (fun A₀ U => n₀ ≤ A₀.card ∧ (∀ p ∈ A₀, O.label p = 0)
      ∧ (A₀.card : ℝ) * θ ≤ (U.card : ℝ)) _ (Real.exp_nonneg _) ?_
  intro A₀ hA₀
  by_cases hn : n₀ ≤ A₀.card
  · by_cases hcl : ∀ p ∈ A₀, O.label p = 0
    · have hmean : ∑ p ∈ A₀, (O.η + (1 - 2 * O.η) * O.label p)
          ≤ (A₀.card : ℝ) * (θ - τ) := by
        rw [Finset.sum_congr rfl (fun p hp => by rw [hcl p hp]), Finset.sum_const, nsmul_eq_mul]
        have : O.η + (1 - 2 * O.η) * 0 = O.η := by ring
        rw [this]
        exact mul_le_mul_of_nonneg_left hθ (Nat.cast_nonneg _)
      refine le_trans (measureReal_mono (fun ω hω => hω.2.2) (measure_ne_top _ _))
        (le_trans (splitAcc_sound O A₀ θ τ hτ hmean) ?_)
      refine Real.exp_le_exp.2 ?_
      have hc : (n₀ : ℝ) ≤ (A₀.card : ℝ) := by exact_mod_cast hn
      nlinarith [sq_nonneg τ]
    · have hz : {ω : Ω | n₀ ≤ A₀.card ∧ (∀ p ∈ A₀, O.label p = 0)
          ∧ (A₀.card : ℝ) * θ
            ≤ (((A₀.filter (fun p => mq O p ω = 1)).card : ℝ))} = (∅ : Set Ω) := by
        ext ω; simp [hcl]
      rw [hz]; simpa using Real.exp_nonneg _
  · have hz : {ω : Ω | n₀ ≤ A₀.card ∧ (∀ p ∈ A₀, O.label p = 0)
        ∧ (A₀.card : ℝ) * θ
          ≤ (((A₀.filter (fun p => mq O p ω = 1)).card : ℝ))} = (∅ : Set Ω) := by
      ext ω; simp [hn]
    rw [hz]; simpa using Real.exp_nonneg _

/-! ### The gate in counting form

The soundness argument does not need the binomial tails themselves, only what they force
about the counts.  `admitted` is kept faithful to `drift_verdict`; `admittedCount` is the
consequence everything downstream uses. -/

/-- A tail at most `α < ½` sits strictly above the mean: the median of `Bin(n,θ)` is at
least `⌊nθ⌋`, so `P[X ≥ j] ≥ ½` for any `j ≤ ⌊nθ⌋`.  (Kaas–Buhrman; not in Mathlib.) -/
theorem lt_of_binomSfGe_le (n : ℕ) (θ α : ℝ) (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1)
    (hα : α < 1 / 2) (h : binomSfGe n θ j ≤ α) : (n : ℝ) * θ < j :=
  sorry

/-- The lower-tail counterpart: `P[X ≤ j] ≤ α < ½` forces `j` strictly below the mean. -/
theorem lt_of_binomCdf_le (n : ℕ) (θ α : ℝ) (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1)
    (hα : α < 1 / 2) (h : binomCdf n θ j ≤ α) : (j : ℝ) < n * θ :=
  sorry

/-- What `admitted` forces about the counts: the accepted side reads as accepting at least
as often as `accept_thresh` claims, the rejected side at most as often as `reject_thresh`
does.  This is all the soundness argument uses. -/
def admittedCount (O : Oracle μ S) (lo hi : ℕ) (εcov : ℝ) (F P : Finset S) (ω : Ω) :
    Prop :=
  ((splitAcc O hi F P ω).2 : ℝ) * gateAcc O εcov ≤ (splitAcc O hi F P ω).1
    ∧ ((splitRej O lo F P ω).1 : ℝ) ≤ (splitRej O lo F P ω).2 * gateRej O εcov

lemma admittedCount_of_admitted (O : Oracle μ S) (lo hi : ℕ) (εcov α : ℝ)
    (F P : Finset S) (ω : Ω) (hα : α < 1 / 2)
    (hacc0 : 0 ≤ gateAcc O εcov) (hacc1 : gateAcc O εcov ≤ 1)
    (hrej0 : 0 ≤ gateRej O εcov) (hrej1 : gateRej O εcov ≤ 1)
    (h : admitted O lo hi εcov α F P ω) : admittedCount O lo hi εcov F P ω :=
  ⟨le_of_lt (lt_of_binomSfGe_le _ _ _ hacc0 hacc1 hα h.1),
    le_of_lt (lt_of_binomCdf_le _ _ _ hrej0 hrej1 hα h.2)⟩

/-- Hoeffding's bound on the binomial upper tail, as a fact about `binomSfGe`.  Validity
needs the tails to force the counts; termination needs the counts to force the tails, which
is this direction. -/
theorem binomSfGe_le (n j : ℕ) (θ τ : ℝ) (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (hτ : 0 ≤ τ)
    (h : (n : ℝ) * (θ + τ) ≤ j) :
    binomSfGe n θ j ≤ Real.exp (-2 * (n : ℝ) * τ ^ 2) :=
  sorry

/-- The lower-tail counterpart for `binomCdf`. -/
theorem binomCdf_le (n j : ℕ) (θ τ : ℝ) (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (hτ : 0 ≤ τ)
    (h : (j : ℝ) ≤ (n : ℝ) * (θ - τ)) :
    binomCdf n θ j ≤ Real.exp (-2 * (n : ℝ) * τ ^ 2) :=
  sorry

/-- The gate's rates are probabilities, with no premise beyond the coverage being a
fraction: `gateAcc` runs from `1 − η` down to `½` as `εcov` runs from `0` to `1`. -/
lemma eta_nonneg (O : Oracle μ S) : 0 ≤ O.η := by
  rw [← O.noise_mean (1 : S)]
  refine integral_nonneg_of_ae ?_
  filter_upwards [O.noise_icc (1 : S)] with ω hω
  exact hω.1

lemma gateAcc_mem (O : Oracle μ S) {εcov : ℝ} (h0 : 0 ≤ εcov) (h1 : εcov ≤ 1)
    (hsig : O.η ≤ 1 / 2) : 0 ≤ gateAcc O εcov ∧ gateAcc O εcov ≤ 1 := by
  have hη0 := eta_nonneg O
  unfold gateAcc
  constructor <;> nlinarith [mul_le_of_le_one_right (by linarith : (0:ℝ) ≤ 1 / 2 - O.η) h1,
    mul_nonneg (by linarith : (0:ℝ) ≤ 1 / 2 - O.η) h0]

lemma gateRej_mem (O : Oracle μ S) {εcov : ℝ} (h0 : 0 ≤ εcov) (h1 : εcov ≤ 1)
    (hsig : O.η ≤ 1 / 2) : 0 ≤ gateRej O εcov ∧ gateRej O εcov ≤ 1 := by
  have hη0 := eta_nonneg O
  unfold gateRej
  constructor <;> nlinarith [mul_le_of_le_one_right (by linarith : (0:ℝ) ≤ 1 / 2 - O.η) h1,
    mul_nonneg (by linarith : (0:ℝ) ≤ 1 / 2 - O.η) h0]

/-- **A cut that reads as its own class is admitted.**  The counting form of the gate's
other direction: enough hits on the accept side and few enough on the reject side put both
binomial tails under `α`. -/
lemma admitted_of_counts (O : Oracle μ S) (lo hi : ℕ) (εcov α τ : ℝ) (F P : Finset S) (ω : Ω)
    (hτ : 0 ≤ τ) (hε0 : 0 ≤ εcov) (hε1 : εcov ≤ 1) (hsig : O.η ≤ 1 / 2)
    (hacc : ((splitAcc O hi F P ω).2 : ℝ) * (gateAcc O εcov + τ) ≤ ((splitAcc O hi F P ω).1 : ℝ))
    (hrej : ((splitRej O lo F P ω).1 : ℝ) ≤ ((splitRej O lo F P ω).2 : ℝ) * (gateRej O εcov - τ))
    (hαa : Real.exp (-2 * ((splitAcc O hi F P ω).2 : ℝ) * τ ^ 2) ≤ α)
    (hαr : Real.exp (-2 * ((splitRej O lo F P ω).2 : ℝ) * τ ^ 2) ≤ α) :
    admitted O lo hi εcov α F P ω :=
  ⟨le_trans (binomSfGe_le _ _ _ τ (gateAcc_mem O hε0 hε1 hsig).1 (gateAcc_mem O hε0 hε1 hsig).2
      hτ hacc) hαa,
    le_trans (binomCdf_le _ _ _ τ (gateRej_mem O hε0 hε1 hsig).1
      (gateRej_mem O hε0 hε1 hsig).2 hτ hrej) hαr⟩

open scoped Classical in
/-- **A decisive correct cut puts every accepting prefix on its accept side.**  `cutCorrect`
gives one direction, `decided` the other: an accepting prefix cannot be on the reject side
without the cut being wrong there, and it is on one side or the other. -/
lemma card_le_sideAcc (O : Oracle μ S) (lo ha : ℕ) (F C : Finset S) (ω : Ω)
    (hcut : ∀ p ∈ C, cutCorrect O lo ha F p ω) (hdec : ∀ p ∈ C, decided O lo ha F p ω) :
    (C.filter (fun p => O.label p = 1)).card
      ≤ (C.filter (fun p => ha < voteCount O F p ω)).card := by
  classical
  refine Finset.card_le_card (fun p hp => ?_)
  obtain ⟨hpC, hl⟩ := Finset.mem_filter.1 hp
  refine Finset.mem_filter.2 ⟨hpC, ?_⟩
  rcases hdec p hpC with h | h
  · exact h
  · exact absurd ((hcut p hpC).2 h) (by rw [hl]; norm_num)

open scoped Classical in
/-- The mirror: every rejecting prefix is on the reject side. -/
lemma card_le_sideRej (O : Oracle μ S) (lo ha : ℕ) (F C : Finset S) (ω : Ω)
    (hcut : ∀ p ∈ C, cutCorrect O lo ha F p ω) (hdec : ∀ p ∈ C, decided O lo ha F p ω) :
    (C.filter (fun p => O.label p = 0)).card
      ≤ (C.filter (fun p => voteCount O F p ω ≤ lo)).card := by
  classical
  refine Finset.card_le_card (fun p hp => ?_)
  obtain ⟨hpC, hl⟩ := Finset.mem_filter.1 hp
  refine Finset.mem_filter.2 ⟨hpC, ?_⟩
  rcases hdec p hpC with h | h
  · exact absurd ((hcut p hpC).1 h) (by rw [hl]; norm_num)
  · exact h

open scoped Classical in
/-- **The accept side is short only by the prefixes that were indecisive or mis-cut.**  The
counting form of `card_le_sideAcc`, which is what the assembly needs: both gates bound those
two counts by a *fraction*, so requiring every prefix to behave is never necessary. -/
lemma card_le_sideAcc_add (O : Oracle μ S) (lo ha : ℕ) (F C : Finset S) (ω : Ω) :
    (C.filter (fun p => O.label p = 1)).card
      ≤ (C.filter (fun p => ha < voteCount O F p ω)).card
        + ((C.filter (fun p => ¬ decided O lo ha F p ω)).card
          + (C.filter (fun p => ¬ cutCorrect O lo ha F p ω)).card) := by
  classical
  refine le_trans (Finset.card_le_card ?_)
    (le_trans (Finset.card_union_le _ _) (by
      exact Nat.add_le_add_left (Finset.card_union_le _ _) _))
  intro p hp
  obtain ⟨hpC, hl⟩ := Finset.mem_filter.1 hp
  by_cases hd : decided O lo ha F p ω
  · by_cases hc : cutCorrect O lo ha F p ω
    · refine Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpC, ?_⟩)
      rcases hd with h | h
      · exact h
      · exact absurd (hc.2 h) (by rw [hl]; norm_num)
    · exact Finset.mem_union_right _
        (Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hpC, hc⟩))
  · exact Finset.mem_union_right _ (Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpC, hd⟩))

open scoped Classical in
/-- The mirror for the reject side. -/
lemma card_le_sideRej_add (O : Oracle μ S) (lo ha : ℕ) (F C : Finset S) (ω : Ω) :
    (C.filter (fun p => O.label p = 0)).card
      ≤ (C.filter (fun p => voteCount O F p ω ≤ lo)).card
        + ((C.filter (fun p => ¬ decided O lo ha F p ω)).card
          + (C.filter (fun p => ¬ cutCorrect O lo ha F p ω)).card) := by
  classical
  refine le_trans (Finset.card_le_card ?_)
    (le_trans (Finset.card_union_le _ _) (by
      exact Nat.add_le_add_left (Finset.card_union_le _ _) _))
  intro p hp
  obtain ⟨hpC, hl⟩ := Finset.mem_filter.1 hp
  by_cases hd : decided O lo ha F p ω
  · by_cases hc : cutCorrect O lo ha F p ω
    · refine Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpC, ?_⟩)
      rcases hd with h | h
      · exact absurd (hc.1 h) (by rw [hl]; norm_num)
      · exact h
    · exact Finset.mem_union_right _
        (Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hpC, hc⟩))
  · exact Finset.mem_union_right _ (Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpC, hd⟩))

open scoped Classical in
/-- The two sides of the gate's split, as the counts `admitted` reads them. -/
lemma splitAcc_card (O : Oracle μ S) (hi : ℕ) (F P : Finset S) (ω : Ω) :
    (splitAcc O hi F P ω).2 = (P.filter (fun p => hi - 1 < voteCount O F p ω)).card := rfl

open scoped Classical in
lemma splitRej_card (O : Oracle μ S) (lo : ℕ) (F P : Finset S) (ω : Ω) :
    (splitRej O lo F P ω).2 = (P.filter (fun p => voteCount O F p ω ≤ lo)).card := rfl

open scoped Classical in
/-- **Markov on a per-prefix failure count.**  Both gates ask for a *fraction* of the
certification sample, not for every prefix to behave, so a per-prefix bound `E` only has to
beat the limit `l`: the cost is `E / l`, with no union over the sample.

`G` is the event on which the per-prefix failures are covered by the sets `Bad` — in use it
is where the family is light enough for the per-prefix bound to apply at all. -/
theorem count_frac_le (C : Finset S) (Bad : S → Set Ω) (Pr : Ω → S → Prop)
    [inst : ∀ ω, DecidablePred (fun p => Pr ω p)] (G : Set Ω)
    (hmeas : ∀ p, MeasurableSet (Bad p)) (E l : ℝ) (hE : 0 ≤ E) (hl : 0 < l)
    (hCpos : 0 < C.card) (hper : ∀ p ∈ C, μ.real (Bad p) ≤ E)
    (hsub : ∀ ω ∈ G, ∀ p ∈ C, Pr ω p → ω ∈ Bad p) :
    μ.real (G ∩ {ω | l * (C.card : ℝ) < ((C.filter (fun p => Pr ω p)).card : ℝ)})
      ≤ E / l := by
  classical
  set Y : Ω → ℝ := fun ω => ∑ p ∈ C, (Bad p).indicator (fun _ => (1 : ℝ)) ω with hY
  have hYnn : 0 ≤ᵐ[μ] Y := by
    filter_upwards with ω
    exact Finset.sum_nonneg (fun p _ => Set.indicator_nonneg (fun _ _ => zero_le_one) ω)
  have hYint : Integrable Y μ :=
    integrable_finset_sum C (fun p _ => (integrable_const (1 : ℝ)).indicator (hmeas p))
  have hYmean : ∫ ω, Y ω ∂μ ≤ (C.card : ℝ) * E := by
    rw [hY, integral_finset_sum _ (fun p _ => (integrable_const (1 : ℝ)).indicator (hmeas p))]
    calc ∑ p ∈ C, ∫ ω, (Bad p).indicator (fun _ => (1 : ℝ)) ω ∂μ
        = ∑ p ∈ C, μ.real (Bad p) :=
          Finset.sum_congr rfl (fun p _ => integral_indicator_one (hmeas p))
      _ ≤ ∑ _p ∈ C, E := Finset.sum_le_sum hper
      _ = (C.card : ℝ) * E := by rw [Finset.sum_const, nsmul_eq_mul]
  have hsub' : G ∩ {ω | l * (C.card : ℝ) < ((C.filter (fun p => Pr ω p)).card : ℝ)}
      ⊆ {ω | l * (C.card : ℝ) ≤ Y ω} := by
    rintro ω ⟨hG, hlt⟩
    have hlt' : l * (C.card : ℝ) < ((C.filter (fun p => Pr ω p)).card : ℝ) := hlt
    refine le_trans (le_of_lt hlt') ?_
    show ((C.filter (fun p => Pr ω p)).card : ℝ)
      ≤ ∑ p ∈ C, (Bad p).indicator (fun _ => (1 : ℝ)) ω
    rw [← Finset.sum_filter_add_sum_filter_not C (fun p => Pr ω p)]
    have h1 : ∀ p ∈ C.filter (fun p => Pr ω p),
        (Bad p).indicator (fun _ => (1 : ℝ)) ω = 1 := by
      intro p hp
      obtain ⟨hpC, hpr⟩ := Finset.mem_filter.1 hp
      simp [Set.indicator_of_mem (hsub ω hG p hpC hpr)]
    have h2 : 0 ≤ ∑ p ∈ C.filter (fun p => ¬ Pr ω p),
        (Bad p).indicator (fun _ => (1 : ℝ)) ω :=
      Finset.sum_nonneg (fun p _ => Set.indicator_nonneg (fun _ _ => zero_le_one) ω)
    rw [Finset.sum_congr rfl h1, Finset.sum_const, nsmul_eq_mul, mul_one]
    linarith
  have hCR : (0 : ℝ) < (C.card : ℝ) := by exact_mod_cast hCpos
  have hstep : (l * (C.card : ℝ)) * μ.real {ω | l * (C.card : ℝ) ≤ Y ω} ≤ (C.card : ℝ) * E :=
    le_trans (mul_meas_ge_le_integral_of_nonneg hYnn hYint (l * (C.card : ℝ))) hYmean
  have hfin := measureReal_mono hsub' (measure_ne_top (μ := μ) _)
  have hkey : (C.card : ℝ) * (l * μ.real {ω | l * (C.card : ℝ) ≤ Y ω}) ≤ (C.card : ℝ) * E := by
    calc (C.card : ℝ) * (l * μ.real {ω | l * (C.card : ℝ) ≤ Y ω})
        = (l * (C.card : ℝ)) * μ.real {ω | l * (C.card : ℝ) ≤ Y ω} := by ring
      _ ≤ (C.card : ℝ) * E := hstep
  have hdiv := le_of_mul_le_mul_left hkey hCR
  rw [le_div_iff₀ hl]
  nlinarith [hfin, hdiv, hl.le]

open scoped Classical in
/-- **A mostly-correct cut on sides that carry prefixes is admitted.**  The fractional form:
the cut has to be right on all but `w` of the certification sample, not on all of it, and
`count_frac_le` is what supplies that `w`.  Both sides draw their wrong-member budget from
the same count, since a side member carrying the wrong label *is* a mis-cut prefix. -/
theorem admitted_whp (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    (lo hi : ℕ) (εcov α τ w : ℝ) (n₀ : ℕ) (fam : Ω → Finset S)
    (hQ : ∀ ω, ∀ p ∈ C, ∀ v ∈ fam ω, p * v ∈ Q)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (hτ : 0 ≤ τ) (hε0 : 0 ≤ εcov) (hε1 : εcov ≤ 1) (hsig : O.η ≤ 1 / 2)
    (hga : ∀ n : ℕ, n₀ ≤ n → n ≤ C.card →
      (n : ℝ) * (gateAcc O εcov + τ + τ) ≤ (n : ℝ) * (1 - O.η) - (1 - 2 * O.η) * w)
    (hgr : ∀ n : ℕ, n₀ ≤ n → n ≤ C.card →
      (n : ℝ) * O.η + (1 - 2 * O.η) * w ≤ (n : ℝ) * (gateRej O εcov - τ - τ))
    (hα : Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) ≤ α) :
    μ.real {ω | (((C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ) ≤ w)
        ∧ n₀ ≤ (splitAcc O hi (fam ω) C ω).2 ∧ n₀ ≤ (splitRej O lo (fam ω) C ω).2
        ∧ ¬ admitted O lo hi εcov α (fam ω) C ω}
      ≤ 2 * Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := by
  classical
  set sA : Ω → Finset S := fun ω => C.filter (fun p => hi - 1 < voteCount O (fam ω) p ω) with hsA
  set sR : Ω → Finset S := fun ω => C.filter (fun p => voteCount O (fam ω) p ω ≤ lo) with hsR
  have hsideA : ∀ ω, sA ω ⊆ C := fun ω => Finset.filter_subset _ _
  have hsideR : ∀ ω, sR ω ⊆ C := fun ω => Finset.filter_subset _ _
  have hvc : ∀ (ω ω' : Ω), (∀ w ∈ Q, O.noise w ω = O.noise w ω') → ∀ p ∈ C,
      voteCount O (fam ω) p ω = voteCount O (fam ω') p ω' := by
    intro ω ω' h p hp
    rw [← hcongr ω ω' h]
    exact voteCount_congr O _ p (fun v hv => by rw [mq_congr O (h _ (hQ ω p hp v hv))])
  have hcA : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → sA ω = sA ω' := by
    intro ω ω' h
    exact Finset.filter_congr (fun p hp => by rw [hvc ω ω' h p hp])
  have hcR : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → sR ω = sR ω' := by
    intro ω ω' h
    exact Finset.filter_congr (fun p hp => by rw [hvc ω ω' h p hp])
  have hbA := gate_acc_admit_bound' O C Q hdisj sA hsideA hcA (gateAcc O εcov + τ) τ w n₀ hτ
    hsig (by intro n h1 h2; have := hga n h1 h2; linarith)
  have hbR := gate_rej_admit_bound' O C Q hdisj sR hsideR hcR (gateRej O εcov - τ) τ w n₀ hτ
    hsig (by intro n h1 h2; have := hgr n h1 h2; linarith)
  have hsub : {ω | (((C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ) ≤ w)
      ∧ n₀ ≤ (splitAcc O hi (fam ω) C ω).2 ∧ n₀ ≤ (splitRej O lo (fam ω) C ω).2
      ∧ ¬ admitted O lo hi εcov α (fam ω) C ω}
      ⊆ {ω | n₀ ≤ (sA ω).card
            ∧ ((((sA ω).filter (fun p => ¬ (O.label p = 1))).card : ℝ) ≤ w)
            ∧ (((sA ω).filter (fun p => mq O p ω = 1)).card : ℝ)
              ≤ ((sA ω).card : ℝ) * (gateAcc O εcov + τ)}
        ∪ {ω | n₀ ≤ (sR ω).card
            ∧ ((((sR ω).filter (fun p => ¬ (O.label p = 0))).card : ℝ) ≤ w)
            ∧ ((sR ω).card : ℝ) * (gateRej O εcov - τ)
              ≤ (((sR ω).filter (fun p => mq O p ω = 1)).card : ℝ)} := by
    rintro ω ⟨hw, hnA, hnR, hadm⟩
    have hcardA : ((sA ω).filter (fun p => ¬ (O.label p = 1))).card
        ≤ (C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card :=
      Finset.card_le_card (sideAcc_wrong_subset O lo (hi - 1) (fam ω) C ω)
    have hcardR : ((sR ω).filter (fun p => ¬ (O.label p = 0))).card
        ≤ (C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card :=
      Finset.card_le_card (sideRej_wrong_subset O lo (hi - 1) (fam ω) C ω)
    have hwA : (((sA ω).filter (fun p => ¬ (O.label p = 1))).card : ℝ) ≤ w := by
      refine le_trans ?_ hw
      exact_mod_cast hcardA
    have hwR : (((sR ω).filter (fun p => ¬ (O.label p = 0))).card : ℝ) ≤ w := by
      refine le_trans ?_ hw
      exact_mod_cast hcardR
    have hexpA : Real.exp (-2 * ((sA ω).card : ℝ) * τ ^ 2) ≤ α := by
      refine le_trans (Real.exp_le_exp.2 ?_) hα
      have : (n₀ : ℝ) ≤ ((sA ω).card : ℝ) := by exact_mod_cast hnA
      nlinarith [sq_nonneg τ]
    have hexpR : Real.exp (-2 * ((sR ω).card : ℝ) * τ ^ 2) ≤ α := by
      refine le_trans (Real.exp_le_exp.2 ?_) hα
      have : (n₀ : ℝ) ≤ ((sR ω).card : ℝ) := by exact_mod_cast hnR
      nlinarith [sq_nonneg τ]
    by_contra hnot
    simp only [Set.mem_union, not_or, Set.mem_setOf_eq, not_and] at hnot
    refine hadm (admitted_of_counts O lo hi εcov α τ (fam ω) C ω hτ hε0 hε1 hsig ?_ ?_
      hexpA hexpR)
    · exact not_lt.1 (fun hlt => (hnot.1 hnA hwA) (le_of_lt hlt))
    · exact not_lt.1 (fun hlt => (hnot.2 hnR hwR) (le_of_lt hlt))
  calc μ.real {ω | (((C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ) ≤ w)
        ∧ n₀ ≤ (splitAcc O hi (fam ω) C ω).2 ∧ n₀ ≤ (splitRej O lo (fam ω) C ω).2
        ∧ ¬ admitted O lo hi εcov α (fam ω) C ω}
      ≤ μ.real (_ ∪ _) := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ _ + _ := measureReal_union_le _ _
    _ ≤ Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) + Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) :=
        add_le_add hbA hbR
    _ = 2 * Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := by ring

/-- A countable union bound in real form: Mathlib has `measure_iUnion_le` in `ℝ≥0∞` and
`measureReal_iUnion_fintype_le` for finite index, but not this. -/
lemma measureReal_iUnion_le_tsum {A : Type*} [MeasurableSpace A] {ρ : Measure A}
    [IsFiniteMeasure ρ] {T : Type*} [Countable T] (E : T → Set A) (w : T → ℝ)
    (hw0 : ∀ t, 0 ≤ w t) (hper : ∀ t, ρ.real (E t) ≤ w t) (hsum : Summable w) :
    ρ.real (⋃ t, E t) ≤ ∑' t, w t := by
  have hle : ∀ t, ρ (E t) ≤ ENNReal.ofReal (w t) := by
    intro t
    calc ρ (E t) = ENNReal.ofReal (ρ.real (E t)) :=
          (ENNReal.ofReal_toReal (measure_ne_top ρ _)).symm
      _ ≤ ENNReal.ofReal (w t) := ENNReal.ofReal_le_ofReal (hper t)
  have hunion : ρ (⋃ t, E t) ≤ ENNReal.ofReal (∑' t, w t) := by
    calc ρ (⋃ t, E t) ≤ ∑' t, ρ (E t) := measure_iUnion_le _
      _ ≤ ∑' t, ENNReal.ofReal (w t) := ENNReal.tsum_le_tsum hle
      _ = ENNReal.ofReal (∑' t, w t) := (ENNReal.ofReal_tsum_of_nonneg hw0 hsum).symm
  calc ρ.real (⋃ t, E t) ≤ (ENNReal.ofReal (∑' t, w t)).toReal := by
        refine ENNReal.toReal_mono (by simp) hunion
    _ = ∑' t, w t := ENNReal.toReal_ofReal (tsum_nonneg hw0)

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

/-- Two independent draws, from possibly different populations, coincide with probability
`∑ₐ D₁{a}·D₂{a}`. -/
lemma prod_diagonal_eq (D₁ D₂ : Measure S) [IsProbabilityMeasure D₁] [IsProbabilityMeasure D₂] :
    (D₁.prod D₂).real {q : S × S | q.1 = q.2} = ∑' a : S, D₁.real {a} * D₂.real {a} := by
  classical
  have hdiag : {q : S × S | q.1 = q.2} = ⋃ a : S, {((a, a) : S × S)} := by
    ext q; simp [Prod.ext_iff, eq_comm]
  have hdisj : Pairwise (Function.onFun Disjoint (fun a : S => ({(a, a)} : Set (S × S)))) := by
    intro a b hab
    simp only [Function.onFun, Set.disjoint_singleton]
    exact fun h => hab (congrArg Prod.fst h)
  rw [measureReal_def, hdiag, measure_iUnion hdisj (fun a => measurableSet_singleton _),
    ENNReal.tsum_toReal_eq (fun a => by
      simp only [← Set.singleton_prod_singleton, Measure.prod_prod]
      exact ENNReal.mul_ne_top (measure_ne_top _ _) (measure_ne_top _ _))]
  exact tsum_congr (fun a => by
    rw [← Set.singleton_prod_singleton, Measure.prod_prod, ENNReal.toReal_mul, measureReal_def,
      measureReal_def])

/-- Two independent draws coincide with probability exactly the collision mass. -/
lemma prod_diagonal_eq_collisionMass (Dj : Measure S) [IsProbabilityMeasure Dj] :
    (Dj.prod Dj).real {q : S × S | q.1 = q.2} = collisionMass Dj := by
  rw [prod_diagonal_eq Dj Dj, collisionMass]
  exact tsum_congr (fun a => (sq _).symm)

/-- **Draws from two populations collide no more often than within one.**  By `ab ≤ (a²+b²)/2`
pointwise, so the certification stream's cross-collisions with the table stream are paid for
by the same `ρ`. -/
lemma cross_collision_le (D₁ D₂ : Measure S) [IsProbabilityMeasure D₁] [IsProbabilityMeasure D₂]
    (ρ : ℝ) (h₁ : collisionMass D₁ ≤ ρ) (h₂ : collisionMass D₂ ≤ ρ)
    (hs₁ : Summable (fun a : S => D₁.real {a} ^ 2)) (hs₂ : Summable (fun a : S => D₂.real {a} ^ 2)) :
    (D₁.prod D₂).real {q : S × S | q.1 = q.2} ≤ ρ := by
  rw [prod_diagonal_eq D₁ D₂]
  have hle : ∀ a : S, D₁.real {a} * D₂.real {a}
      ≤ (D₁.real {a} ^ 2 + D₂.real {a} ^ 2) / 2 := by
    intro a; nlinarith [sq_nonneg (D₁.real {a} - D₂.real {a})]
  have hsum : Summable (fun a : S => (D₁.real {a} ^ 2 + D₂.real {a} ^ 2) / 2) :=
    (hs₁.add hs₂).div_const 2
  have hprod : Summable (fun a : S => D₁.real {a} * D₂.real {a}) :=
    Summable.of_nonneg_of_le
      (fun a => mul_nonneg measureReal_nonneg measureReal_nonneg) hle hsum
  calc ∑' a : S, D₁.real {a} * D₂.real {a}
      ≤ ∑' a : S, (D₁.real {a} ^ 2 + D₂.real {a} ^ 2) / 2 := hprod.tsum_le_tsum hle hsum
    _ = (collisionMass D₁ + collisionMass D₂) / 2 := by
        rw [tsum_div_const, hs₁.tsum_add hs₂, collisionMass, collisionMass]
    _ ≤ ρ := by linarith

open scoped Classical in
/-- Two fixed coordinates of an i.i.d. block coincide with exactly the collision mass. -/
lemma pi_coord_eq (Dj : Measure S) [IsProbabilityMeasure Dj] {m : ℕ} {i i' : Fin m}
    (hii : i ≠ i') :
    (Measure.pi fun _ : Fin m => Dj).real {p : Fin m → S | p i = p i'} = collisionMass Dj := by
  have hset : {p : Fin m → S | p i = p i'}
      = ⋃ a : S, Set.univ.pi (fun z => if z = i ∨ z = i' then ({a} : Set S) else Set.univ) := by
    ext p
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_univ_pi]
    refine ⟨fun h => ⟨p i, fun z => ?_⟩, fun ⟨a, ha⟩ => ?_⟩
    · by_cases hz : z = i ∨ z = i'
      · rw [if_pos hz]
        rcases hz with hz | hz <;> rw [hz] <;> simp [h]
      · rw [if_neg hz]; trivial
    · have h1 := ha i; have h2 := ha i'
      rw [if_pos (Or.inl rfl)] at h1
      rw [if_pos (Or.inr rfl)] at h2
      rw [Set.mem_singleton_iff] at h1 h2
      rw [h1, h2]
  have hdisj : Pairwise (Function.onFun Disjoint (fun a : S =>
      Set.univ.pi (fun z => if z = i ∨ z = i' then ({a} : Set S) else Set.univ))) := by
    intro a b hab
    simp only [Function.onFun, Set.disjoint_left]
    intro p ha hb
    have h1 := ha i; have h2 := hb i
    rw [Set.mem_univ_pi] at ha hb
    have h1' := ha i; have h2' := hb i
    rw [if_pos (Or.inl rfl), Set.mem_singleton_iff] at h1' h2'
    exact hab (h1'.symm.trans h2')
  have hmeas : ∀ a : S, MeasurableSet
      (Set.univ.pi (fun z => if z = i ∨ z = i' then ({a} : Set S) else Set.univ)) := by
    intro a
    refine MeasurableSet.univ_pi (fun z => ?_)
    by_cases hz : z = i ∨ z = i'
    · rw [if_pos hz]; exact measurableSet_singleton _
    · rw [if_neg hz]; exact MeasurableSet.univ
  have hbox : ∀ a : S, (Measure.pi fun _ : Fin m => Dj)
      (Set.univ.pi (fun z => if z = i ∨ z = i' then ({a} : Set S) else Set.univ))
      = Dj {a} * Dj {a} := by
    intro a
    rw [Measure.pi_pi]
    have hfac : ∀ z : Fin m, Dj (if z = i ∨ z = i' then ({a} : Set S) else Set.univ)
        = if z ∈ ({i, i'} : Finset (Fin m)) then Dj {a} else 1 := by
      intro z
      by_cases hz : z = i ∨ z = i'
      · rw [if_pos hz, if_pos (by simpa using hz)]
      · rw [if_neg hz, if_neg (by simpa using hz), measure_univ]
    rw [Finset.prod_congr rfl (fun z _ => hfac z), Finset.prod_ite_mem, Finset.univ_inter,
      Finset.prod_const, Finset.card_pair hii, sq]
  rw [measureReal_def, hset, measure_iUnion hdisj hmeas, collisionMass,
    ENNReal.tsum_toReal_eq (fun a => by
      rw [hbox a]; exact ENNReal.mul_ne_top (measure_ne_top _ _) (measure_ne_top _ _))]
  exact tsum_congr (fun a => by rw [hbox a, ENNReal.toReal_mul, sq, measureReal_def])

open scoped Classical in
/-- **The certification block is distinct except for the collision mass.**  This is what
the `ρ` premise buys: below this event the `m` draws are distinct strings, so their noise
bits are independent and the gate's binomial null is honest. -/
lemma pi_not_injective_le (Dj : Measure S) [IsProbabilityMeasure Dj] (m : ℕ) (ρ : ℝ)
    (hρ : collisionMass Dj ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (Measure.pi fun _ : Fin m => Dj).real {p : Fin m → S | ¬ Function.Injective p}
      ≤ (m : ℝ) ^ 2 * ρ := by
  classical
  set κ := (Finset.univ : Finset (Fin m × Fin m)).filter (fun q => q.1 ≠ q.2) with hκ
  have hsub : {p : Fin m → S | ¬ Function.Injective p}
      ⊆ ⋃ q ∈ κ, {p : Fin m → S | p q.1 = p q.2} := by
    intro p hp
    simp only [Set.mem_setOf_eq, Function.Injective, not_forall] at hp
    obtain ⟨a, b, hab, hne⟩ := hp
    exact Set.mem_biUnion (show (a, b) ∈ κ by simp [hκ, hne]) hab
  calc (Measure.pi fun _ : Fin m => Dj).real {p : Fin m → S | ¬ Function.Injective p}
      ≤ (Measure.pi fun _ : Fin m => Dj).real (⋃ q ∈ κ, {p : Fin m → S | p q.1 = p q.2}) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ q ∈ κ, (Measure.pi fun _ : Fin m => Dj).real {p : Fin m → S | p q.1 = p q.2} :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _q ∈ κ, ρ := Finset.sum_le_sum (fun q hq => by
        rw [pi_coord_eq Dj (Finset.mem_filter.mp hq).2]; exact hρ)
    _ = (κ.card : ℝ) * ρ := by rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ (m : ℝ) ^ 2 * ρ := by
        refine mul_le_mul_of_nonneg_right ?_ hρ0
        have h1 : κ.card ≤ (Finset.univ : Finset (Fin m × Fin m)).card :=
          Finset.card_le_card (Finset.filter_subset _ _)
        have h2 : (Finset.univ : Finset (Fin m × Fin m)).card = m * m := by simp
        rw [h2] at h1
        calc (κ.card : ℝ) ≤ ((m * m : ℕ) : ℝ) := by exact_mod_cast h1
          _ = (m : ℝ) ^ 2 := by push_cast; ring

/-- **The budgets the loop may reach**: every component under a cap.

Capping is not a convenience, it is forced twice over.

*Freshness.*  `disjoint_readSet` needs the certification prefixes distinct from the table
prefixes, and the chance they are not is at most `|populations|·m²·ρ` — which *grows* with
the budget.  `S` is countable, so every `D j` is atomic, and once the draws exceed roughly
`1/ρ` the sample has exhausted the support: the certification prefixes are no longer fresh,
the gate scores bits the family was selected on, and the claim is false rather than
unproven.

*Summability.*  The gate's per-budget failure probability depends on the certification
count, not on the suffix budget, so it does not decay in `M` — and a union over
unboundedly many budgets with a constant failure per budget diverges.

A cap settles both at once, and makes the index finite so no summable envelope is needed
at all.  Part 2's obligation becomes: reach a passing state *within* the cap. -/
structure CapOnly (cap B : Budget) : Prop where
  M : B.M ≤ cap.M
  m : B.m ≤ cap.m
  k : B.k ≤ cap.k
  cn : B.cn ≤ cap.cn
  cd : B.cd ≤ cap.cd
  lo : B.lo ≤ cap.lo
  hi : B.hi ≤ cap.hi
  sc : B.sc ≤ cap.sc

instance instFiniteCapOnly (cap : Budget) : Finite {B : Budget // CapOnly cap B} := by
  refine Finite.of_injective
    (fun B => ((⟨B.val.M, Nat.lt_succ_of_le B.property.M⟩ : Fin (cap.M + 1)),
      (⟨B.val.m, Nat.lt_succ_of_le B.property.m⟩ : Fin (cap.m + 1)),
      (⟨B.val.k, Nat.lt_succ_of_le B.property.k⟩ : Fin (cap.k + 1)),
      (⟨B.val.cn, Nat.lt_succ_of_le B.property.cn⟩ : Fin (cap.cn + 1)),
      (⟨B.val.cd, Nat.lt_succ_of_le B.property.cd⟩ : Fin (cap.cd + 1)),
      (⟨B.val.lo, Nat.lt_succ_of_le B.property.lo⟩ : Fin (cap.lo + 1)),
      (⟨B.val.hi, Nat.lt_succ_of_le B.property.hi⟩ : Fin (cap.hi + 1)),
      (⟨B.val.sc, Nat.lt_succ_of_le B.property.sc⟩ : Fin (cap.sc + 1)))) ?_
  intro B B' hb
  simpa [Subtype.ext_iff, Budget.ext_iff, Prod.ext_iff, Fin.ext_iff] using hb

/-- **What one tested state may cost.**  The five events `measureReal_admitFail_le` charges,
summed over the populations: the certification draws repeating or meeting the table, the
sample missing the wrong set, and the two gate sides. -/
noncomputable def stateFail (O : Oracle μ S) (populations : Finset J) (εcov ρ : ℝ)
    (B : Budget) : ℝ :=
  (populations.card : ℝ) * (((populations.card : ℝ) + 1) * (B.m : ℝ) ^ 2 * ρ
    + (Real.exp (-2 * (B.m : ℝ) * (εcov / 4) ^ 2)
      + 2 * Real.exp (-2 * (εcov / 32 * (B.m : ℝ)) * ((1 - 2 * O.η) * εcov / 16) ^ 2)))

/-- **The states the loop may stop at.**  Under the cap, with the thresholds in order, and
with enough prefixes drawn to carry the state's share of the error budget.

The share is what the schedule is for.  At a handful of prefixes the gate cannot be sound —
a wrong family passes a two-prefix test at constant probability — so the loop cannot test
there, and the guarantee cannot cover it.  Dividing `δ/2` equally among the states under the
cap is the `X, X/2, X/4, …` schedule's obligation written out: every attempt is at a prefix
count large enough for its own share. -/
structure Capped (O : Oracle μ S) (populations : Finset J) (εcov δ ρ : ℝ)
    (cap B : Budget) : Prop where
  capOnly : CapOnly cap B
  /-- Reject strictly below accept, so the two gate sides are disjoint. -/
  lohi : B.lo < B.hi
  /-- Enough prefixes for the state's share of the budget. -/
  share : stateFail O populations εcov ρ B
    ≤ δ / (2 * (Nat.card {B : Budget // CapOnly cap B} : ℝ))

instance instFiniteCapped (O : Oracle μ S) (populations : Finset J) (εcov δ ρ : ℝ)
    (cap : Budget) : Finite {B : Budget // Capped O populations εcov δ ρ cap B} :=
  Finite.of_injective (fun B => (⟨B.val, B.property.capOnly⟩ : {B : Budget // CapOnly cap B}))
    (fun B B' h => Subtype.ext (congrArg (fun z : {B : Budget // CapOnly cap B} => z.val) h))

/-! ### The first Lloyd step

Its centre is `{ε}`, so the loss is the disagreement with the seed's own column:
`#{p ∈ P | mq (p·v) ≠ mq p}`.  That is the screening statistic, and its mean separates an
accept-preserving candidate from one carrying flip mass `φ` by `φ(1−2η)²` — two noisy reads
compared, hence the square. -/

/-- **Functions of disjoint blocks of an independent family are independent.**

Mathlib has the two-block case (`iIndepFun.indepFun_finset`) but not this.  It is needed
because the first step's loss at a prefix reads *two* strings, `p` and `p · v`, and those
pairs are disjoint across prefixes — by cancellation for `p · v`, and by flatness for the
bare prefixes. -/
theorem iIndepFun_blocks {ι κ : Type*} {X : κ → Ω → ℝ}
    (hX : ∀ w, Measurable (X w)) (hindep : iIndepFun X μ) (A : ι → Finset κ)
    (hdisj : Pairwise (fun i j => Disjoint (A i) (A j)))
    (g : ι → Ω → ℝ) (hg : ∀ i, Measurable (g i))
    (hblock : ∀ i, ∀ ω ω', (∀ w ∈ A i, X w ω = X w ω') → g i ω = g i ω') :
    iIndepFun g μ :=
  sorry

open scoped Classical in
/-- The first step's loss at one prefix, in the exact form `hammingLoss` uses. -/
noncomputable def seedLoss (O : Oracle μ S) (cn cd : ℕ) (v p : S) (ω : Ω) : ℝ :=
  if ((mq O (p * v) ω = 1) ↔ cn * ({(1 : S)} : Finset S).card
      < cd * voteCount O {(1 : S)} p ω) then 0 else 1

open scoped Classical in
/-- The first step's loss *is* the sum of `seedLoss` over the prefixes — definitionally,
not merely almost everywhere, which is what lets the argmin be transported. -/
lemma hammingLoss_seed (O : Oracle μ S) (cn cd : ℕ) (P : Finset S) (ω : Ω) (v : S) :
    hammingLoss O {(1 : S)} cn cd P ω v = ∑ p ∈ P, seedLoss O cn cd v p ω := by
  classical
  unfold hammingLoss seedLoss
  rw [Finset.sum_ite]
  simp

lemma seedLoss_icc (O : Oracle μ S) (cn cd : ℕ) (v p : S) :
    ∀ᵐ ω ∂μ, seedLoss O cn cd v p ω ∈ Set.Icc (0 : ℝ) 1 := by
  filter_upwards with ω
  unfold seedLoss
  split_ifs <;> norm_num

lemma measurable_voteCount (O : Oracle μ S) (F : Finset S) (p : S) :
    Measurable (fun ω => voteCount O F p ω) := by
  classical
  have hfun : (fun ω => voteCount O F p ω)
      = fun ω => ∑ v ∈ F, (if mq O (p * v) ω = 1 then 1 else 0) := by
    funext ω; unfold voteCount; rw [Finset.card_filter]
  rw [hfun]
  exact Finset.measurable_sum _ (fun v _ =>
    Measurable.ite (measurableSet_eq_fun (mq_meas O _) measurable_const)
      measurable_const measurable_const)

lemma seedLoss_meas (O : Oracle μ S) (cn cd : ℕ) (v p : S) :
    Measurable (seedLoss O cn cd v p) := by
  classical
  unfold seedLoss
  refine Measurable.ite ?_ measurable_const measurable_const
  exact MeasurableSet.iff (measurableSet_eq_fun (mq_meas O _) measurable_const)
    (measurableSet_lt measurable_const (measurable_const.mul (measurable_voteCount O _ p)))

/-- On a proper centre ratio the first step's loss is the disagreement with the seed's own
column: `cn < cd` makes `cn·1 < cd·voteCount {ε}` say exactly `mq p = 1`. -/
lemma seedLoss_eq_disagree (O : Oracle μ S) {cn cd : ℕ} (hcd : cn < cd) (v p : S) :
    ∀ᵐ ω ∂μ, seedLoss O cn cd v p ω
      = mq O (p * v) ω + mq O p ω - 2 * (mq O (p * v) ω * mq O p ω) := by
  classical
  filter_upwards [mq_bit O (p * v), mq_bit O p] with ω h1 h0
  have hvc : voteCount O {(1 : S)} p ω = if mq O p ω = 1 then 1 else 0 := by
    unfold voteCount
    rcases h0 with h | h <;> simp [voteCount, Finset.filter_singleton, mul_one, h]
  unfold seedLoss
  rw [hvc]
  rcases h1 with h1 | h1 <;> rcases h0 with h0 | h0 <;>
    · rw [h1, h0]
      norm_num [hcd, Nat.not_lt.2 (Nat.zero_le cn)]

lemma mq_integrable (O : Oracle μ S) (w : S) : Integrable (mq O w) μ :=
  MeasureTheory.Integrable.of_mem_Icc 0 1 (mq_meas O w).aemeasurable (mq_icc O w)

lemma mq_mul_integrable (O : Oracle μ S) (w w' : S) :
    Integrable (fun ω => mq O w ω * mq O w' ω) μ := by
  refine MeasureTheory.Integrable.of_mem_Icc 0 1
    (((mq_meas O w).mul (mq_meas O w')).aemeasurable) ?_
  filter_upwards [mq_icc O w, mq_icc O w'] with ω h1 h0
  rw [Set.mem_Icc] at h1 h0 ⊢
  exact ⟨mul_nonneg h1.1 h0.1, by nlinarith [h1.1, h1.2, h0.1, h0.2]⟩

lemma seedLoss_integrable (O : Oracle μ S) (cn cd : ℕ) (v p : S) :
    Integrable (seedLoss O cn cd v p) μ :=
  MeasureTheory.Integrable.of_mem_Icc 0 1 (seedLoss_meas O cn cd v p).aemeasurable
    (seedLoss_icc O cn cd v p)

/-- **The first step's mean separates by the square of the signal.**  Two noisy reads are
compared, so an accept-preserving candidate disagrees at `2η(1−η)` and one that flips at `p`
at `1 − 2η(1−η)`; the difference is `(1−2η)²`.  This is exactly the statistic
`_screen_cohort` tests against, and the square is why its power is weaker than a comparison
against the truth would be. -/
lemma seedLoss_mean (O : Oracle μ S) {cn cd : ℕ} (hcd : cn < cd) (v p : S) (hv : p * v ≠ p) :
    μ[seedLoss O cn cd v p] = 2 * O.η * (1 - O.η) + O.flip v p * (1 - 2 * O.η) ^ 2 := by
  have hprod : μ[fun ω => mq O (p * v) ω * mq O p ω] = μ[mq O (p * v)] * μ[mq O p] :=
    ProbabilityTheory.IndepFun.integral_mul_eq_mul_integral
      ((mq_indep O).indepFun hv) (mq_meas O _).aestronglyMeasurable
      (mq_meas O _).aestronglyMeasurable
  have hsplit : μ[seedLoss O cn cd v p]
      = μ[mq O (p * v)] + μ[mq O p] - 2 * (μ[mq O (p * v)] * μ[mq O p]) := by
    rw [integral_congr_ae (seedLoss_eq_disagree O hcd v p),
      integral_sub (f := fun ω => mq O (p * v) ω + mq O p ω)
        (g := fun ω => 2 * (mq O (p * v) ω * mq O p ω))
        ((mq_integrable O _).add (mq_integrable O _))
        ((mq_mul_integrable O (p * v) p).const_mul 2),
      integral_add (mq_integrable O _) (mq_integrable O _), integral_const_mul, hprod]
  rw [hsplit, mq_mean, mq_mean]
  show _ = 2 * O.η * (1 - O.η)
    + (O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p) * (1 - 2 * O.η) ^ 2
  ring

open scoped Classical in
/-- The screen's statistic is the summed `seedLoss`, exactly rather than almost everywhere:
`voteCount` over the singleton `{ε}` is a card, so it is a bit by construction. -/
lemma screenCount_eq_sum (O : Oracle μ S) {cn cd : ℕ} (hcd : cn < cd) (P : Finset S) (v : S)
    (ω : Ω) : ((screenCount O P v ω : ℝ)) = ∑ p ∈ P, seedLoss O cn cd v p ω := by
  classical
  have hcond : ∀ p : S, (cn * ({(1 : S)} : Finset S).card < cd * voteCount O {(1 : S)} p ω)
      ↔ (mq O p ω = 1) := by
    intro p
    have hvc : voteCount O {(1 : S)} p ω = if mq O p ω = 1 then 1 else 0 := by
      unfold voteCount
      by_cases h : mq O p ω = 1 <;> simp [Finset.filter_singleton, mul_one, h]
    rw [hvc]
    by_cases h : mq O p ω = 1 <;> simp [h, hcd]
  unfold screenCount seedLoss
  rw [Finset.sum_ite]
  simp only [Finset.sum_const, smul_zero, zero_add, nsmul_eq_mul, mul_one]
  refine congrArg (fun n : ℕ => (n : ℝ)) (congrArg Finset.card ?_)
  ext p
  simp only [Finset.mem_filter, hcond p]

/-- **The seed's own reads are independent across prefixes.**  Each prefix contributes a
function of two strings, `p` and `p · v`, and those pairs are pairwise disjoint: `p · v` by
right-cancellation, the bare prefixes by distinctness, and the two kinds from each other by
flatness. -/
lemma seedLoss_indep {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S) (cn cd : ℕ)
    {P : Finset S} (hP : ∀ p ∈ P, p ∈ Pre) (v : S) :
    iIndepFun (fun p : {p // p ∈ P} => seedLoss O cn cd v p.val) μ := by
  refine iIndepFun_blocks (X := O.noise) (fun w => O.noise_meas' w) O.noise_indep
    (fun p : {p // p ∈ P} => {p.val, p.val * v}) ?_ _
    (fun p => seedLoss_meas O cn cd v p.val) ?_
  · intro a b hab
    refine Finset.disjoint_left.2 (fun w hw hw' => ?_)
    simp only [Finset.mem_insert, Finset.mem_singleton] at hw hw'
    rcases hw with rfl | rfl <;> rcases hw' with hb | hb
    · exact hab (Subtype.ext hb)
    · by_cases hv1 : v = 1
      · exact hab (Subtype.ext (by rw [hb, hv1, mul_one]))
      · exact flat_ne_of_ne_one hflat (hP _ b.property) (hP _ a.property) hv1 hb.symm
    · by_cases hv1 : v = 1
      · exact hab (Subtype.ext (by rw [← hb, hv1, mul_one]))
      · exact flat_ne_of_ne_one hflat (hP _ a.property) (hP _ b.property) hv1 hb
    · exact hab (Subtype.ext (mul_right_cancel hb))
  · intro p ω ω' h
    have h1 : O.noise (p.val * v) ω = O.noise (p.val * v) ω' := h _ (by simp)
    have h0 : O.noise p.val ω = O.noise p.val ω' := h _ (by simp)
    unfold seedLoss
    have hvc : ∀ w ∈ ({(1 : S)} : Finset S),
        (mq O (p.val * w) ω = 1 ↔ mq O (p.val * w) ω' = 1) := by
      intro w hw
      rw [Finset.mem_singleton] at hw
      subst hw
      rw [mul_one, mq_congr O h0]
    rw [mq_congr O h1, voteCount_congr O {(1 : S)} p.val hvc]

/-! ### The first Lloyd step

Its centre is `{ε}`, so its loss is `seedLoss`, whose mean separates a candidate that never
flips from one that flips on a `Δ` fraction by `Δ(1−2η)²` — the screen's statistic.  The
ranking is *not* what bounds the family's flip mass: `clusterAt_flip_bound` reads that off
the screen, which every candidate has already passed (issue #288).  What the ranking has to
deliver is only that the seed survives it, and `lloydStep`'s tie-break gives that outright.
-/

/-! ### Part 1 comes from the clustering, not the gate

`hpAPBound` is a *premise*: accept-preserving suffixes are drawn with probability `≥ pAP`,
so a pool of `M` holds about `pAP·M` of them and, once `M ≳ k/pAP`, enough to fill the
family.  `identify_cluster_around` then keeps the `k` least-loss candidates, and a
candidate carrying flip mass `Δ` sits `2sΔm` above an accept-preserving one in expected
loss.  So every selected member carries little flip mass, and `coverage_of_summed_flip`
turns that into the per-population coverage.  No certification draw enters, and the gate is
not used: its job is detecting that `hpAPBound` is *false* for a target, which the theorem
excludes by hypothesis.

Two things make this work, and both are recorded rather than assumed silently.

*The pool must not outgrow the prefixes.*  Taking the best of `M` candidates buys
`√(2 log M)` of the loss's spread `√(m(¼−s²))` for free, so the ranking is decided by luck
rather than by flip mass unless `m ≳ (¼−s²)·log M / (2s²Δ²)`.  That is the guard added in
the algorithm; the theorem needs it as `hpool`.

*The centre is the previous iterate.*  The first Lloyd step centres on `{ε}`, which is
deterministic, so `chosen_accept_preserving_whp` applies to it directly.  Later steps centre
on the family the previous step chose, which is `ω`-dependent — the same difficulty the
gate had, and solvable the same way, since a candidate's own reads are at `p · v` while the
centre is read at `p · v'` for the members `v'`, and those are disjoint strings for
`v ∉ F`.  `measureReal_selection_le` and `noiseAlg` are what that needs. -/

/-- The prefix budget the pool needs before its least-loss selection tracks flip mass
rather than luck: `m ≥ (¼ − s²)·log M / (2s²Δ²)`.  The guard in `sample_suffix_family`
switches to prefix growth rather than cross it. -/
def PoolRanked (O : Oracle μ S) (Δ : ℝ) (B : Budget) : Prop :=
  (1 / 4 - (1 / 2 - O.η) ^ 2) * Real.log (max (B.M : ℝ) 2)
    ≤ 2 * (1 / 2 - O.η) ^ 2 * Δ ^ 2 * (B.m : ℝ)

/-! ### Why the iteration cannot drift, and what the seed check is for

The later steps centre on the previous iterate, and the loss they minimise is, up to noise,
the symmetric difference `|Φ_v Δ B|` between a candidate's flip set and the set `B` where
the centre disagrees with the truth.  That has a consequence worth stating plainly, because
it is a property of the algorithm rather than of the proof:

*Every common flip set is a perfect fixed point.*  If every member of the family flips on
the same set `B`, then the centre is wrong exactly on `B`, a candidate flipping on `B`
scores `|Φ_v Δ B| = 0`, and the iteration is stationary at minimal loss — **for any `B`**.
So the Lloyd loss cannot by itself distinguish the truth from a family that is uniformly
wrong on a whole set of prefixes, however large.  Worse, moving toward such a family
*decreases* the loss, so the code's `if new_loss >= loss: break` does not prevent it.

What prevents it is the **seed check**.  At such a fixed point the seed, which never flips,
disagrees with the centre on all of `B` and so scores `|B| > 0` while the drifted members
score `0`.  With `k` drifted candidates available the seed is pushed out of the `k`
least-loss, and `identify_cluster_around` refuses the step — `if seed_local not in nearest:
break`, modelled here as `lloydStep` returning the family it had.

So `lloydStep`'s seed check is not a tidiness measure: it is the only thing standing
between the iteration and an arbitrarily drifted fixed point that the loss actively prefers.
That is also why `one_mem_clusterAround` is worth having as a lemma.

The invariant to carry through the iteration is therefore about the seed's standing, not
about flip mass directly: while the seed survives the ranking, the members cannot be much
worse than it is. -/

/-- **The iteration cannot leave the seed behind**: either the step keeps it, or the step
is refused and the family is unchanged.  This is the disjunction the induction runs on. -/
theorem lloyd_step_seed_or_refused (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (k : ℕ)
    (ω : Ω) (F : Finset S) :
    lloydStep O cn cd P cands ω k F = F ∨ (1 : S) ∈ lloydStep O cn cd P cands ω k F := by
  classical
  unfold lloydStep
  split_ifs
  · exact Or.inr (Finset.mem_insert_self _ _)
  · exact Or.inl rfl

/-- **A kept step ranks every member against every candidate it left out.**  With `≥ k`
accept-preserving candidates in the pool, either all `k` members are accept-preserving or
one was left out, and then every member scores at least as well as it does — which is the
bound on `|Φ_v Δ B|` the induction needs. -/
theorem lloyd_step_ranked_by_excluded (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S)
    (k : ℕ) (hk : k ≤ cands.card) (ω : Ω) (F : Finset S) {w : S} (hw : w ∈ cands)
    (hwn : w ∉ leastLossSubset (clusterLoss O F cn cd P cands ω) cands k) :
    ∀ v ∈ leastLossSubset (clusterLoss O F cn cd P cands ω) cands k,
      clusterLoss O F cn cd P cands ω v ≤ clusterLoss O F cn cd P cands ω w :=
  fun v hv => leastLossSubset_least (clusterLoss O F cn cd P cands ω) cands k hk v hv w hw hwn

open scoped Classical in
/-- **An accept-preserving candidate passes the screen.**  Its disagreement with the seed's
column has mean exactly `2η(1−η)` — two noisy reads of the same bit — so a cutoff `γ` above
that is cleared except in the upper tail.  This is `screen_tail`'s mirror, and it is what
keeps the candidate pool from emptying. -/
theorem screen_pass {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S) {cn cd : ℕ}
    (hcd : cn < cd) {P : Finset S} (hP : ∀ p ∈ P, p ∈ Pre) (v : S) (hv : v ≠ 1)
    (γ : ℝ) (sc : ℕ) (hγ : 0 ≤ γ)
    (hclean : ∀ p ∈ P, O.flip v p = 0)
    (hsc : (P.card : ℝ) * (2 * O.η * (1 - O.η) + γ) ≤ (sc : ℝ)) :
    μ.real {ω | ¬ (screenCount O P v ω ≤ sc)} ≤ Real.exp (-2 * (P.card : ℝ) * γ ^ 2) := by
  classical
  set b : ℝ := 2 * O.η * (1 - O.η) with hb
  have hcard : ((Finset.univ : Finset {p // p ∈ P}).card : ℝ) = (P.card : ℝ) := by
    simp [Finset.card_univ]
  have hmean : ∑ i : {p // p ∈ P}, μ[seedLoss O cn cd v i.val]
      ≤ ((Finset.univ : Finset {p // p ∈ P}).card : ℝ) * b := by
    rw [Finset.sum_congr rfl
      (fun i _ => seedLoss_mean O hcd v i.val (mul_ne_self i.val v hv)),
      Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ]
    simp only [nsmul_eq_mul, Fintype.card_coe]
    have hzero : ∑ i : {p // p ∈ P}, O.flip v i.val * (1 - 2 * O.η) ^ 2 = 0 :=
      Finset.sum_eq_zero (fun i _ => by rw [hclean i.val i.property]; ring)
    rw [hzero, hb]
    simp
  have htail := sumUpper_le (fun (i : {p // p ∈ P}) => seedLoss O cn cd v i.val)
    (Finset.univ : Finset {p // p ∈ P}) b γ
    (fun i => (seedLoss_meas O cn cd v i.val).aemeasurable)
    (seedLoss_indep hflat O cn cd hP v)
    (fun i => seedLoss_icc O cn cd v i.val) hmean hγ
  rw [hcard] at htail
  refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) htail
  intro ω hω
  show (P.card : ℝ) * (b + γ) ≤ ∑ i : {p // p ∈ P}, seedLoss O cn cd v i.val ω
  have hsum : ∑ i : {p // p ∈ P}, seedLoss O cn cd v i.val ω
      = ∑ p ∈ P, seedLoss O cn cd v p ω :=
    Finset.sum_attach P (fun p => seedLoss O cn cd v p ω)
  rw [hsum, ← screenCount_eq_sum O hcd P v ω]
  have : (sc : ℝ) < ((screenCount O P v ω : ℕ) : ℝ) := by
    have := not_le.1 hω
    exact_mod_cast this
  linarith

open scoped Classical in
/-- **The values the family can take.**  The iterate starts at the seed and every step
either keeps its argument or returns a `k`-subset, so the family is one of finitely many
`Finset`s and each has `k` members unless it is the seed alone. -/
lemma clusterAround_mem_values (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω)
    (k : ℕ) (hone : (1 : S) ∈ cands) :
    clusterAround O cn cd P cands ω k ∈ insert ({(1 : S)}) (cands.powersetCard k) := by
  classical
  unfold clusterAround
  generalize k * P.card + 1 = n
  have hstep : ∀ F ∈ insert ({(1 : S)}) (cands.powersetCard k),
      lloydStep O cn cd P cands ω k F ∈ insert ({(1 : S)}) (cands.powersetCard k) := by
    intro F hF
    unfold lloydStep
    split_ifs
    · by_cases hk : k - 1 ≤ (cands.erase 1).card ∧ 0 < k
      · refine Finset.mem_insert_of_mem (Finset.mem_powersetCard.2 ⟨?_, ?_⟩)
        · exact Finset.insert_subset hone
            (le_trans (leastLossSubset_subset' _ _ _) (Finset.erase_subset _ _))
        · rw [Finset.card_insert_of_notMem (fun hc =>
            (Finset.mem_erase.1 (leastLossSubset_subset' _ _ _ hc)).1 rfl),
            leastLossSubset_card _ _ _ hk.1]
          omega
      · rcases Nat.eq_zero_or_pos k with hk0 | hkpos
        · rw [hk0, show leastLossSubset (clusterLoss O F cn cd P cands ω) (cands.erase 1)
              (0 - 1) = ∅ from Finset.card_eq_zero.1
            (leastLossSubset_card _ _ _ (Nat.zero_le _))]
          exact Finset.mem_insert_self _ _
        · have hbig : (cands.erase 1).card < k - 1 := by
            by_contra hc
            exact hk ⟨not_lt.1 hc, hkpos⟩
          rw [leastLossSubset, dif_neg (by
            simp only [Finset.powersetCard_nonempty, not_le]
            omega)]
          exact Finset.mem_insert_self _ _
    · exact hF
  induction n with
  | zero => exact Finset.mem_insert_self _ _
  | succ n ih =>
      rw [Function.iterate_succ_apply']
      exact hstep _ ih

open scoped Classical in
/-- **A badly-flipping candidate rarely passes the screen.**  Its disagreement with the
seed's column has mean `2η(1−η) + φ(1−2η)²`, so a cutoff `γ` below that is cleared only in
the lower tail. -/
theorem screen_tail {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S) {cn cd : ℕ}
    (hcd : cn < cd) {P : Finset S} (hP : ∀ p ∈ P, p ∈ Pre) (v : S) (hv : v ≠ 1)
    (Δ γ : ℝ) (sc : ℕ) (hγ : 0 ≤ γ) (hsig : O.η ≤ 1 / 2)
    (hflip : Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p)
    (hsc : (sc : ℝ) ≤ (P.card : ℝ) * ((2 * O.η * (1 - O.η) + Δ * (1 - 2 * O.η) ^ 2) - γ)) :
    μ.real {ω | screenCount O P v ω ≤ sc} ≤ Real.exp (-2 * (P.card : ℝ) * γ ^ 2) := by
  classical
  set b : ℝ := 2 * O.η * (1 - O.η) + Δ * (1 - 2 * O.η) ^ 2 with hb
  have hcard : ((Finset.univ : Finset {p // p ∈ P}).card : ℝ) = (P.card : ℝ) := by
    simp [Finset.card_univ]
  have hmean : ((Finset.univ : Finset {p // p ∈ P}).card : ℝ) * b
      ≤ ∑ i : {p // p ∈ P}, μ[seedLoss O cn cd v i.val] := by
    rw [Finset.sum_congr rfl
      (fun i _ => seedLoss_mean O hcd v i.val (mul_ne_self i.val v hv)),
      Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ]
    simp only [nsmul_eq_mul, Fintype.card_coe]
    rw [← Finset.sum_mul, hb]
    have hattach : ∑ i : {p // p ∈ P}, O.flip v i.val = ∑ p ∈ P, O.flip v p :=
      Finset.sum_attach P (fun p => O.flip v p)
    rw [hattach]
    have hsq : (0 : ℝ) ≤ (1 - 2 * O.η) ^ 2 := sq_nonneg _
    nlinarith [hflip]
  have htail := sumLower_le (fun (i : {p // p ∈ P}) => seedLoss O cn cd v i.val)
    (Finset.univ : Finset {p // p ∈ P}) b γ
    (fun i => (seedLoss_meas O cn cd v i.val).aemeasurable)
    (seedLoss_indep hflat O cn cd hP v)
    (fun i => seedLoss_icc O cn cd v i.val) hmean hγ
  rw [hcard] at htail
  refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) htail
  intro ω hω
  show ∑ i : {p // p ∈ P}, seedLoss O cn cd v i.val ω ≤ (P.card : ℝ) * (b - γ)
  have hsum : ∑ i : {p // p ∈ P}, seedLoss O cn cd v i.val ω
      = ((screenCount O P v ω : ℝ)) := by
    rw [screenCount_eq_sum O hcd P v ω]
    exact (Finset.sum_attach P (fun p => seedLoss O cn cd v p ω)).symm ▸ rfl
  rw [hsum]
  exact le_trans (by exact_mod_cast hω) hsc

/-! ### The iterate, and what actually bounds its flips

`lloyd_first_step_ranked` covers the first step, whose centre is the seed's own column.
Every step after that ranks candidates against the *current* centre's majority vote, and
that ranking is by agreement with the centre's drift rather than by flipping little: writing
`Dset` for the centre's error set, a candidate scores mean `η·#P + (1−2η)·#(Φ_v Δ Dset)`, so
one that flips exactly `Dset` scores zero.  Chasing the bound through the majority vote gives
`d' ≤ (2 / c) · d` with `c = (s + eps) / (2 * s)`, about `3` at the usual settings — no
contraction.

None of that matters, because the ranking is not what bounds the flips: the **screen** is.
A suffix that fails it never becomes a fully observed column and so is never a clustering
candidate at all, and the iterate can only choose among what is left.  (Issue #288.) -/

/-- **The family flips no more than the screened pool does.** -/
theorem clusterAt_flip_bound (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (x : Run Ω S J) (Δ : ℝ)
    (hscreen : ∀ v ∈ screenedAt O populations B x,
      ¬ (Δ * ((prefixesAt populations B.m x).card : ℝ)
        ≤ ∑ p ∈ prefixesAt populations B.m x, O.flip v p)) :
    ∀ w ∈ clusterAt O populations x B,
      ¬ (Δ * ((prefixesAt populations B.m x).card : ℝ)
        ≤ ∑ p ∈ prefixesAt populations B.m x, O.flip w p) :=
  fun w hw => hscreen w (clusterAround_subset O B.cn B.cd _ _ (nz x) B.k
    (one_mem_screenedAt O populations B x) hw)

/-! ### From flip mass to a correct cut

The band is what turns "few members flip" into "the cut is right", and it is far more
generous than a member count suggests.  A rejecting prefix is accepted only when the vote
clears `hi ≈ k(½ + eps)`, and members preserving at `p` contribute only through noise, so
with `f` members flipping the count is at most `f + Bin(k − f, η)`.  In the mean that
needs

    f · 2s > k · (s + eps)

— two *thirds* of the family at `s = 0.3, eps = 0.1`, not the `eps · k` a naive reading of
the margin suggests.  Markov over the members' flip masses then gives a misclassified mass
of about `1.5 · Δ` rather than `Δ / eps`, which is a factor of six. -/

open scoped Classical in
/-- The number of the family's members that flip at a prefix, as a real. -/
noncomputable def flipCount (O : Oracle μ S) (F : Finset S) (p : S) : ℝ :=
  ((F.filter (fun v => O.flip v p = 1)).card : ℝ)

lemma flipCount_eq_sum (O : Oracle μ S) (F : Finset S) (p : S) :
    flipCount O F p = ∑ v ∈ F, O.flip v p := by
  classical
  unfold flipCount
  rw [← Finset.sum_filter_add_sum_filter_not F (fun v => O.flip v p = 1)]
  have h1 : ∑ v ∈ F.filter (fun v => O.flip v p = 1), O.flip v p
      = ((F.filter (fun v => O.flip v p = 1)).card : ℝ) := by
    rw [Finset.sum_congr rfl (fun v hv => (Finset.mem_filter.1 hv).2), Finset.sum_const,
      nsmul_eq_mul, mul_one]
  have h0 : ∑ v ∈ F.filter (fun v => ¬ (O.flip v p = 1)), O.flip v p = 0 := by
    refine Finset.sum_eq_zero (fun v hv => ?_)
    obtain ⟨-, hne⟩ := Finset.mem_filter.1 hv
    rcases O.flip_bit v p with h | h
    · exact h
    · exact absurd h hne
  rw [h1, h0, add_zero]

lemma flip_integrable (O : Oracle μ S) (Dj : Measure S) [IsProbabilityMeasure Dj] (v : S) :
    Integrable (fun p => O.flip v p) Dj :=
  MeasureTheory.Integrable.of_mem_Icc 0 1 (flip_meas O v).aemeasurable
    (Filter.Eventually.of_forall (fun p => flip_icc O v p))

/-- **Markov over the family's flip masses.**  If no member flips more than a `Δ` mass of
the population, the mass of prefixes where a `c` fraction of the family flips is at most
`Δ / c`.

With `voteCount_le_flips_add_noise` this is what turns per-member flip mass into
misclassified mass: the band puts `c = (s + eps) / (2 * s)`, so the price is a constant
near `3 / 2`, not the `1 / eps` a naive reading of the band charges. -/
theorem flipCount_mass_le (O : Oracle μ S) (Dj : Measure S) [IsProbabilityMeasure Dj]
    (F : Finset S) (Δ c : ℝ) (hc : 0 < c) (hFne : 0 < F.card)
    (hF : ∀ v ∈ F, flipMass O Dj v ≤ Δ) :
    Dj.real {p | c * (F.card : ℝ) ≤ flipCount O F p} ≤ Δ / c := by
  classical
  have hcard : (0 : ℝ) < (F.card : ℝ) := by exact_mod_cast hFne
  have hsum : ∀ p, flipCount O F p = ∑ v ∈ F, O.flip v p := flipCount_eq_sum O F
  have hint : Integrable (fun p => flipCount O F p) Dj := by
    refine ((integrable_finset_sum F (fun v _ => flip_integrable O Dj v)).congr ?_)
    exact Filter.Eventually.of_forall (fun p => (hsum p).symm)
  have hnn : 0 ≤ᵐ[Dj] fun p => flipCount O F p :=
    Filter.Eventually.of_forall (fun p => Nat.cast_nonneg _)
  have hInt : ∫ p, flipCount O F p ∂Dj = ∑ v ∈ F, flipMass O Dj v := by
    rw [integral_congr_ae (Filter.Eventually.of_forall hsum),
      integral_finset_sum _ (fun v _ => flip_integrable O Dj v)]
    rfl
  have hmark := mul_meas_ge_le_integral_of_nonneg hnn hint (c * (F.card : ℝ))
  rw [hInt] at hmark
  have hle : ∑ v ∈ F, flipMass O Dj v ≤ (F.card : ℝ) * Δ := by
    calc ∑ v ∈ F, flipMass O Dj v ≤ ∑ _v ∈ F, Δ := Finset.sum_le_sum hF
      _ = (F.card : ℝ) * Δ := by rw [Finset.sum_const, nsmul_eq_mul]
  have hstep : (F.card : ℝ) * (c * Dj.real {p | c * (F.card : ℝ) ≤ flipCount O F p})
      ≤ (F.card : ℝ) * Δ := by
    calc (F.card : ℝ) * (c * Dj.real {p | c * (F.card : ℝ) ≤ flipCount O F p})
        = c * (F.card : ℝ) * Dj.real {p | c * (F.card : ℝ) ≤ flipCount O F p} := by ring
      _ ≤ ∑ v ∈ F, flipMass O Dj v := hmark
      _ ≤ (F.card : ℝ) * Δ := hle
  have := le_of_mul_le_mul_left hstep hcard
  rw [le_div_iff₀ hc]
  linarith

open scoped Classical in
/-- The vote as a sum of reads, which is what concentrates; a.e. equal to `voteCount`. -/
noncomputable def voteSum (O : Oracle μ S) (F : Finset S) (p : S) (ω : Ω) : ℝ :=
  ∑ v ∈ F, mq O (p * v) ω

lemma voteCount_eq_voteSum (O : Oracle μ S) (F : Finset S) (p : S) :
    ∀ᵐ ω ∂μ, ((voteCount O F p ω : ℝ)) = voteSum O F p ω := by
  classical
  filter_upwards [(ae_ball_iff F.countable_toSet).2 (fun v _ => mq_bit O (p * v))] with ω hω
  unfold voteCount voteSum
  rw [← Finset.sum_filter_add_sum_filter_not F (fun v => mq O (p * v) ω = 1)]
  have h1 : ∑ v ∈ F.filter (fun v => mq O (p * v) ω = 1), mq O (p * v) ω
      = ((F.filter (fun v => mq O (p * v) ω = 1)).card : ℝ) := by
    rw [Finset.sum_congr rfl (fun v hv => (Finset.mem_filter.1 hv).2), Finset.sum_const,
      nsmul_eq_mul, mul_one]
  have h0 : ∑ v ∈ F.filter (fun v => ¬ (mq O (p * v) ω = 1)), mq O (p * v) ω = 0 := by
    refine Finset.sum_eq_zero (fun v hv => ?_)
    obtain ⟨hvF, hne⟩ := Finset.mem_filter.1 hv
    rcases hω v hvF with h | h
    · exact h
    · exact absurd h hne
  rw [h1, h0, add_zero]

lemma voteSum_meanSum (O : Oracle μ S) (F : Finset S) (p : S) :
    ∑ v ∈ F, μ[mq O (p * v)]
      = (F.card : ℝ) * O.η + (1 - 2 * O.η) * ∑ v ∈ F, O.label (p * v) := by
  rw [Finset.sum_congr rfl (fun v _ => mq_mean O (p * v)), Finset.sum_add_distrib,
    Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum]

lemma mq_indep_shift (O : Oracle μ S) (p : S) :
    iIndepFun (fun v : S => mq O (p * v)) μ :=
  (mq_indep O).precomp (mul_right_injective p)

/-- **Upper tail on a rejecting prefix.**  A flip fraction of `f` lifts the vote's mean only
to `η + 2 * s * f`. -/
theorem voteSum_upper (O : Oracle μ S) (F : Finset S) (p : S) (hp : O.label p = 0)
    (f γ : ℝ) (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ) :
    μ.real {ω | (F.card : ℝ) * ((O.η + (1 - 2 * O.η) * f) + γ) ≤ voteSum O F p ω}
      ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  have hlab : ∀ v ∈ F, O.label (p * v) = O.flip v p := by
    intro v _
    show O.label (p * v) = O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p
    rw [hp]; ring
  have hmean : ∑ v ∈ F, μ[mq O (p * v)] ≤ (F.card : ℝ) * (O.η + (1 - 2 * O.η) * f) := by
    rw [voteSum_meanSum, Finset.sum_congr rfl hlab, ← flipCount_eq_sum]
    have h2 : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith [O.hη]
    nlinarith [hf]
  exact sumUpper_le (fun v : S => mq O (p * v)) F (O.η + (1 - 2 * O.η) * f) γ
    (fun v => (mq_meas O _).aemeasurable) (mq_indep_shift O p) (fun v => mq_icc O _) hmean hγ

/-- **Lower tail on an accepting prefix.**  Mirror of `voteSum_upper`: the vote's mean only
falls to `1 - η - 2 * s * f`. -/
theorem voteSum_lower (O : Oracle μ S) (F : Finset S) (p : S) (hp : O.label p = 1)
    (f γ : ℝ) (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ) :
    μ.real {ω | voteSum O F p ω ≤ (F.card : ℝ) * ((O.η + (1 - 2 * O.η) * (1 - f)) - γ)}
      ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  have hlab : ∀ v ∈ F, O.label (p * v) = 1 - O.flip v p := by
    intro v _
    show O.label (p * v) = 1 - (O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p)
    rw [hp]; ring
  have hmean : (F.card : ℝ) * (O.η + (1 - 2 * O.η) * (1 - f)) ≤ ∑ v ∈ F, μ[mq O (p * v)] := by
    rw [voteSum_meanSum, Finset.sum_congr rfl hlab, Finset.sum_sub_distrib, Finset.sum_const,
      nsmul_eq_mul, mul_one, ← flipCount_eq_sum]
    have h2 : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith [O.hη]
    nlinarith [hf]
  exact sumLower_le (fun v : S => mq O (p * v)) F (O.η + (1 - 2 * O.η) * (1 - f)) γ
    (fun v => (mq_meas O _).aemeasurable) (mq_indep_shift O p) (fun v => mq_icc O _) hmean hγ

/-- **The cut survives the family being chosen by the clustering.**  At a population prefix
whose query strings the clustering never read, the family is decided by bits independent of
the ones the vote reads, so the fixed-family bound carries over.

`good` is there because the fixed-family bound holds only for families this prefix is light
for: a prefix a lot of the family flips is charged to `flipCount_mass_le` instead. -/
theorem cutCorrect_selected_whp (O : Oracle μ S) (cands Q : Finset S) (p : S) (lo hi : ℕ)
    (hdisj : Disjoint (↑(cands.image (fun v => p * v)) : Set S) (↑Q : Set S))
    (T good : Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T) (hTC : ∀ t ∈ T, t ⊆ cands)
    (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E : ℝ) (hE : 0 ≤ E)
    (hbad : ∀ A₀ ∈ T, A₀ ∈ good → μ.real {ω | ¬ cutCorrect O lo hi A₀ p ω} ≤ E) :
    μ.real {ω | fam ω ∈ good ∧ ¬ cutCorrect O lo hi (fam ω) p ω} ≤ E := by
  classical
  refine selection_side_bound O cands Q (fun v => p * v) hdisj T t₀ ht₀ hTC fam hfam hcongr
    (fun A₀ U => A₀ ∈ good ∧ ¬ ((hi < Finset.card U → O.label p = 1)
      ∧ (Finset.card U ≤ lo → O.label p = 0)))
    E hE ?_
  intro A₀ hA₀
  by_cases hg : A₀ ∈ good
  · refine le_trans (le_of_eq ?_) (hbad A₀ hA₀ hg)
    congr 1
    ext ω
    simp only [Set.mem_setOf_eq, cutCorrect, voteCount, hg, true_and]
  · rw [show {ω | A₀ ∈ good ∧ ¬ ((hi < (A₀.filter (fun v => mq O (p * v) ω = 1)).card
      → O.label p = 1) ∧ ((A₀.filter (fun v => mq O (p * v) ω = 1)).card ≤ lo
      → O.label p = 0))} = (∅ : Set Ω) by ext ω; simp [hg]]
    simpa using hE

open scoped Classical in
/-- **Decisiveness survives the family being chosen by the clustering**, by the same
argument as `cutCorrect_selected_whp`: both are predicates of the vote's count, and the
vote's reads are ones the clustering never made. -/
theorem decided_selected_whp (O : Oracle μ S) (cands Q : Finset S) (p : S) (lo ha : ℕ)
    (hdisj : Disjoint (↑(cands.image (fun v => p * v)) : Set S) (↑Q : Set S))
    (T good : Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T) (hTC : ∀ t ∈ T, t ⊆ cands)
    (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E : ℝ) (hE : 0 ≤ E)
    (hbad : ∀ A₀ ∈ T, A₀ ∈ good → μ.real {ω | ¬ decided O lo ha A₀ p ω} ≤ E) :
    μ.real {ω | fam ω ∈ good ∧ ¬ decided O lo ha (fam ω) p ω} ≤ E := by
  classical
  refine selection_side_bound O cands Q (fun v => p * v) hdisj T t₀ ht₀ hTC fam hfam hcongr
    (fun A₀ U => A₀ ∈ good ∧ ¬ (ha < Finset.card U ∨ Finset.card U ≤ lo)) E hE ?_
  intro A₀ hA₀
  by_cases hg : A₀ ∈ good
  · refine le_trans (le_of_eq ?_) (hbad A₀ hA₀ hg)
    congr 1
    ext ω
    simp only [Set.mem_setOf_eq, decided, voteCount, hg, true_and]
  · rw [show {ω | A₀ ∈ good
      ∧ ¬ (ha < (A₀.filter (fun v => mq O (p * v) ω = 1)).card
        ∨ (A₀.filter (fun v => mq O (p * v) ω = 1)).card ≤ lo)} = (∅ : Set Ω) by
      ext ω; simp [hg]]
    simpa using hE

open scoped Classical in
/-- Decisiveness at every prefix of a set the clustering never read. -/
theorem decided_all_whp {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hCPre : ∀ p ∈ C, p ∈ Pre)
    (hPC : Disjoint P C) (lo ha : ℕ)
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E : ℝ) (hE : 0 ≤ E)
    (hbad : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p → μ.real {ω | ¬ decided O lo ha A₀ p ω} ≤ E) :
    μ.real {ω | ∃ p ∈ C, fam ω ∈ good p ∧ ¬ decided O lo ha (fam ω) p ω}
      ≤ (C.card : ℝ) * E := by
  classical
  have hsub : {ω | ∃ p ∈ C, fam ω ∈ good p ∧ ¬ decided O lo ha (fam ω) p ω}
      ⊆ ⋃ p ∈ C, {ω | fam ω ∈ good p ∧ ¬ decided O lo ha (fam ω) p ω} := by
    rintro ω ⟨p, hp, hω⟩
    exact Set.mem_biUnion hp hω
  have hper : ∀ p ∈ C, μ.real {ω | fam ω ∈ good p ∧ ¬ decided O lo ha (fam ω) p ω} ≤ E := by
    intro p hp
    have hpP : p ∉ P := Finset.disjoint_right.1 hPC hp
    exact decided_selected_whp O cands (readSet P cands) p lo ha
      (disjoint_image_readSet hflat hP (hCPre p hp) hpP) T (good p) t₀ ht₀ hTC fam hfam
      hcongr E hE (fun A₀ hA₀ hg => hbad p hp A₀ hA₀ hg)
  calc μ.real {ω | ∃ p ∈ C, fam ω ∈ good p ∧ ¬ decided O lo ha (fam ω) p ω}
      ≤ μ.real (⋃ p ∈ C, {ω | fam ω ∈ good p ∧ ¬ decided O lo ha (fam ω) p ω}) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ p ∈ C, μ.real {ω | fam ω ∈ good p ∧ ¬ decided O lo ha (fam ω) p ω} :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _p ∈ C, E := Finset.sum_le_sum hper
    _ = (C.card : ℝ) * E := by rw [Finset.sum_const, nsmul_eq_mul]

open scoped Classical in
/-- **The cut is right at every certification prefix the family is light for.**
`cutCorrect_selected_whp` unioned over `C`.  The certification prefixes are disjoint from
the table's, so each one's query strings are ones the clustering never read, and the
per-prefix bound survives the family being chosen. -/
theorem cutRight_cert_whp {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hCPre : ∀ p ∈ C, p ∈ Pre)
    (hPC : Disjoint P C) (lo ha : ℕ)
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E : ℝ) (hE : 0 ≤ E)
    (hbad : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p →
      μ.real {ω | ¬ cutCorrect O lo ha A₀ p ω} ≤ E) :
    μ.real {ω | ∃ p ∈ C, fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω}
      ≤ (C.card : ℝ) * E := by
  classical
  have hsub : {ω | ∃ p ∈ C, fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω}
      ⊆ ⋃ p ∈ C, {ω | fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω} := by
    rintro ω ⟨p, hp, hω⟩
    exact Set.mem_biUnion hp hω
  have hper : ∀ p ∈ C,
      μ.real {ω | fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω} ≤ E := by
    intro p hp
    have hpP : p ∉ P := Finset.disjoint_right.1 hPC hp
    exact cutCorrect_selected_whp O cands (readSet P cands) p lo ha
      (disjoint_image_readSet hflat hP (hCPre p hp) hpP) T (good p) t₀ ht₀ hTC fam hfam
      hcongr E hE (fun A₀ hA₀ hg => hbad p hp A₀ hA₀ hg)
  calc μ.real {ω | ∃ p ∈ C, fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω}
      ≤ μ.real (⋃ p ∈ C, {ω | fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω}) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ p ∈ C, μ.real {ω | fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω} :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _p ∈ C, E := Finset.sum_le_sum hper
    _ = (C.card : ℝ) * E := by rw [Finset.sum_const, nsmul_eq_mul]

lemma measureReal_le_of_ae_imp {A B : Set Ω} (h : ∀ᵐ ω ∂μ, ω ∈ A → ω ∈ B) :
    μ.real A ≤ μ.real B :=
  ENNReal.toReal_mono (measure_ne_top μ B) (measure_mono_ae h)

/-- **The cut is correct at a prefix the family barely flips.**  Only the side the prefix
actually sits on can fail, so one tail — not two — pays for it. -/
theorem cutCorrect_whp (O : Oracle μ S) (F : Finset S) (p : S) (lo hi : ℕ) (f γ : ℝ)
    (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ)
    (hhi : (F.card : ℝ) * ((O.η + (1 - 2 * O.η) * f) + γ) ≤ (hi : ℝ))
    (hlo : (lo : ℝ) < (F.card : ℝ) * ((O.η + (1 - 2 * O.η) * (1 - f)) - γ)) :
    μ.real {ω | ¬ cutCorrect O lo hi F p ω} ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  rcases O.label_bit p with hp | hp
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_upper O F p hp f γ hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hacc : ¬ (hi < voteCount O F p ω → O.label p = 1) := by
      intro hacc
      exact hbad ⟨hacc, fun _ => hp⟩
    have hgt : hi < voteCount O F p ω := by
      by_contra hc
      exact hacc (fun h => absurd h hc)
    have : (hi : ℝ) < (voteCount O F p ω : ℝ) := by exact_mod_cast hgt
    show (F.card : ℝ) * ((O.η + (1 - 2 * O.η) * f) + γ) ≤ voteSum O F p ω
    rw [← heq]; linarith
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_lower O F p hp f γ hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hrej : ¬ (voteCount O F p ω ≤ lo → O.label p = 0) := by
      intro hrej
      exact hbad ⟨fun _ => hp, hrej⟩
    have hle : voteCount O F p ω ≤ lo := by
      by_contra hc
      exact hrej (fun h => absurd h hc)
    have : (voteCount O F p ω : ℝ) ≤ (lo : ℝ) := by exact_mod_cast hle
    show voteSum O F p ω ≤ (F.card : ℝ) * ((O.η + (1 - 2 * O.η) * (1 - f)) - γ)
    rw [← heq]; linarith

/-! ### From a per-prefix bound to a population bound

The population's prefixes are countable, so the mass of bad ones is a sum rather than an
integral over a product, and the exchange is `lintegral_tsum` — no product measure and no
joint measurability in the pair. -/

lemma tsum_singleton_eq (Dj : Measure S) (A : Set S) : ∑' a : A, Dj {(a : S)} = Dj A := by
  have h := tsum_measure_preimage_singleton (μ := Dj) A.to_countable
    (f := (id : S → S)) (fun y _ => measurableSet_singleton y)
  simpa using h

lemma measure_setOf_eq_tsum (Dj : Measure S) (A : Set S) :
    Dj A = ∑' a : S, A.indicator (fun a => Dj {a}) a := by
  rw [← tsum_singleton_eq Dj A]
  exact tsum_subtype A (fun a => Dj {a})

/-- **The expected bad mass is the worst per-prefix bound.**  Each prefix's failure has
`μ`-probability at most `E`, and the prefix masses sum to one. -/
lemma badMass_eq_tsum (Dj : Measure S) (Bad : S → Set Ω) (ω : Ω) :
    Dj {p | ω ∈ Bad p} = ∑' p : S, (Bad p).indicator (fun _ => Dj {p}) ω := by
  rw [measure_setOf_eq_tsum]
  exact tsum_congr (fun p => by by_cases hb : ω ∈ Bad p <;> simp [hb])

lemma measurable_badMass (Dj : Measure S) (Bad : S → Set Ω)
    (hmeas : ∀ p, MeasurableSet (Bad p)) : Measurable (fun ω => Dj {p | ω ∈ Bad p}) := by
  simp only [badMass_eq_tsum Dj Bad]
  exact Measurable.ennreal_tsum (fun p => measurable_const.indicator (hmeas p))

lemma lintegral_badMass_le (Dj : Measure S) [IsProbabilityMeasure Dj] (Bad : S → Set Ω)
    (hmeas : ∀ p, MeasurableSet (Bad p)) (E : ℝ≥0∞) (h : ∀ p, μ (Bad p) ≤ E) :
    ∫⁻ ω, Dj {p | ω ∈ Bad p} ∂μ ≤ E := by
  classical
  have hpt : ∀ ω, Dj {p | ω ∈ Bad p}
      = ∑' p : S, (Bad p).indicator (fun _ => Dj {p}) ω := badMass_eq_tsum Dj Bad
  calc ∫⁻ ω, Dj {p | ω ∈ Bad p} ∂μ
      = ∫⁻ ω, ∑' p : S, (Bad p).indicator (fun _ => Dj {p}) ω ∂μ :=
        lintegral_congr hpt
    _ = ∑' p : S, ∫⁻ ω, (Bad p).indicator (fun _ => Dj {p}) ω ∂μ :=
        lintegral_tsum (fun p => ((measurable_const.indicator (hmeas p))).aemeasurable)
    _ = ∑' p : S, Dj {p} * μ (Bad p) := by
        refine tsum_congr (fun p => ?_)
        rw [lintegral_indicator (hmeas p), lintegral_const, Measure.restrict_apply_univ]
    _ ≤ ∑' p : S, Dj {p} * E := ENNReal.tsum_le_tsum (fun p => by gcongr; exact h p)
    _ = E := by
        rw [ENNReal.tsum_mul_right, show (∑' p : S, Dj {p}) = 1 by
          simpa using (measure_setOf_eq_tsum Dj Set.univ).symm, one_mul]

/-- **Markov on the bad mass.**  A per-prefix failure probability of `E` leaves at most an
`E / ε` fraction of runs with more than `ε` of the population misclassified. -/
lemma measure_badMass_ge_le (Dj : Measure S) [IsProbabilityMeasure Dj] (Bad : S → Set Ω)
    (hmeas : ∀ p, MeasurableSet (Bad p)) (E ε : ℝ≥0∞) (hε : ε ≠ 0) (hεtop : ε ≠ ∞)
    (h : ∀ p, μ (Bad p) ≤ E) :
    μ {ω | ε ≤ Dj {p | ω ∈ Bad p}} ≤ E / ε := by
  rw [ENNReal.le_div_iff_mul_le (Or.inl hε) (Or.inl hεtop), mul_comm]
  exact le_trans (mul_meas_ge_le_lintegral₀ (measurable_badMass Dj Bad hmeas).aemeasurable ε)
    (lintegral_badMass_le Dj Bad hmeas E h)

/-- Markov in real form. -/
lemma measureReal_badMass_ge_le (Dj : Measure S) [IsProbabilityMeasure Dj] (Bad : S → Set Ω)
    (hmeas : ∀ p, MeasurableSet (Bad p)) (E ε : ℝ) (hE : 0 ≤ E) (hε : 0 < ε)
    (h : ∀ p, μ.real (Bad p) ≤ E) :
    μ.real {ω | ε ≤ Dj.real {p | ω ∈ Bad p}} ≤ E / ε := by
  have hset : {ω | ε ≤ Dj.real {p | ω ∈ Bad p}}
      = {ω | ENNReal.ofReal ε ≤ Dj {p | ω ∈ Bad p}} := by
    ext ω
    exact (ENNReal.ofReal_le_iff_le_toReal (measure_ne_top Dj _)).symm
  have hEnn : ∀ p, μ (Bad p) ≤ ENNReal.ofReal E := by
    intro p
    rw [← ENNReal.ofReal_toReal (measure_ne_top μ (Bad p))]
    exact ENNReal.ofReal_le_ofReal (h p)
  have hmark := measure_badMass_ge_le (μ := μ) Dj Bad hmeas (ENNReal.ofReal E)
    (ENNReal.ofReal ε) (by simpa using hε) (by simp) hEnn
  rw [measureReal_def, hset]
  calc (μ {ω | ENNReal.ofReal ε ≤ Dj {p | ω ∈ Bad p}}).toReal
      ≤ (ENNReal.ofReal E / ENNReal.ofReal ε).toReal :=
        ENNReal.toReal_mono
          (ENNReal.div_ne_top ENNReal.ofReal_ne_top (by simpa using hε)) hmark
    _ = E / ε := by
        rw [ENNReal.toReal_div, ENNReal.toReal_ofReal hE, ENNReal.toReal_ofReal hε.le]

/-- The law of the draws alone. -/
noncomputable def drawLaw (D : J → Measure S) (Dsf : Measure S) :
    Measure ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) :=
  (((Measure.infinitePi fun _ : ℕ => Dsf).prod
      (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j))).prod
    (Measure.infinitePi fun z : J × ℕ => D z.1)

instance (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] : IsProbabilityMeasure (drawLaw D Dsf) := by
  unfold drawLaw; infer_instance

lemma runLaw_eq_prod (D : J → Measure S) (Dsf : Measure S) :
    runLaw μ D Dsf = μ.prod (drawLaw D Dsf) := rfl

/-- One table coordinate has the population's own law. -/
lemma map_drawCoord (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (j : J) (i : ℕ) :
    Measure.map (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => d.1.2 j i)
        (drawLaw D Dsf) = D j := by
  have hstep : (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => d.1.2 j i)
      = (fun p : ℕ → S => p i) ∘ ((fun q : J → ℕ → S => q j) ∘ (Prod.snd ∘ Prod.fst)) := rfl
  rw [hstep, ← Measure.map_map (by fun_prop) (by fun_prop),
    ← Measure.map_map (by fun_prop) (by fun_prop),
    ← Measure.map_map measurable_snd measurable_fst, drawLaw, Measure.map_fst_prod]
  simp only [measure_univ, one_smul]
  rw [Measure.map_snd_prod]
  simp only [measure_univ, one_smul]
  rw [(measurePreserving_eval (fun j : J => Measure.infinitePi fun _ : ℕ => D j) j).map_eq,
    (measurePreserving_eval_infinitePi (fun _ : ℕ => D j) i).map_eq]

/-- **A bound at almost every fixed draw is a bound on the run.**  The clustering's prefixes
and candidates are draws, so its guarantees are stated for the noise at a fixed table; this
is what lifts them.  The `a.e.` is what lets the table be assumed inside the flat set. -/
lemma runLaw_slice_le (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (A : Set (Run Ω S J)) (hA : MeasurableSet A) (E : ℝ≥0∞)
    (h : ∀ᵐ d ∂(drawLaw D Dsf), μ {ω | ((ω, d) : Run Ω S J) ∈ A} ≤ E) :
    runLaw μ D Dsf A ≤ E := by
  rw [runLaw_eq_prod, Measure.prod_apply_symm hA]
  exact le_trans (lintegral_mono_ae h) (by simp)

/-- The table's prefixes land in the flat set, since that is where the populations live. -/
lemma ae_draws_mem_Pre (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (Pre : Set S)
    (populations : Finset J) (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) :
    ∀ᵐ d ∂(drawLaw D Dsf), ∀ j ∈ populations, ∀ i : ℕ, d.1.2 j i ∈ Pre := by
  have hmeasPre : MeasurableSet (Preᶜ : Set S) := (Set.to_countable _).measurableSet
  have hcoord : ∀ z : J × ℕ, ∀ᵐ d ∂(drawLaw D Dsf), z.1 ∈ populations → d.1.2 z.1 z.2 ∈ Pre := by
    rintro ⟨j, i⟩
    by_cases hj : j ∈ populations
    · have hz : drawLaw D Dsf {d | ¬ (j ∈ populations → d.1.2 j i ∈ Pre)} = 0 := by
        have hset : {d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S))
            | ¬ (j ∈ populations → d.1.2 j i ∈ Pre)}
            = (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => d.1.2 j i) ⁻¹' Preᶜ := by
          ext d; simp [hj]
        rw [hset, ← Measure.map_apply (by fun_prop) hmeasPre, map_drawCoord D Dsf j i]
        exact hsupp j hj
      exact ae_iff.2 hz
    · filter_upwards with d hjj
      exact absurd hjj hj
  filter_upwards [ae_all_iff.2 hcoord] with d hd j hj i
  exact hd (j, i) hj

/-- **A bound at every fixed noise-and-table slice is a bound on the run.**  The
certification draws are the last factor, so they can be sliced off on their own — which is
what lets the gate be judged on prefixes the family was never selected from. -/
lemma runLaw_slice_cert_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (A : Set (Run Ω S J)) (hA : MeasurableSet A) (E : ℝ≥0∞)
    (h : ∀ y : Ω × ((ℕ → S) × (J → ℕ → S)),
      (Measure.infinitePi fun z : J × ℕ => D z.1)
        {c | ((y.1, (y.2, c)) : Run Ω S J) ∈ A} ≤ E) :
    runLaw μ D Dsf A ≤ E := by
  set νsq : Measure ((ℕ → S) × (J → ℕ → S)) :=
    (Measure.infinitePi fun _ : ℕ => Dsf).prod
      (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) with hνsq
  set νc : Measure (J × ℕ → S) := Measure.infinitePi fun z : J × ℕ => D z.1 with hνc
  have hmap : Measure.map (MeasurableEquiv.prodAssoc : (Ω × ((ℕ → S) × (J → ℕ → S)))
      × (J × ℕ → S) ≃ᵐ Ω × (((ℕ → S) × (J → ℕ → S)) × (J × ℕ → S)))
      ((μ.prod νsq).prod νc) = runLaw μ D Dsf :=
    (measurePreserving_prodAssoc μ νsq νc).map_eq
  have hpre : runLaw μ D Dsf A = ((μ.prod νsq).prod νc)
      ((MeasurableEquiv.prodAssoc : (Ω × ((ℕ → S) × (J → ℕ → S)))
        × (J × ℕ → S) ≃ᵐ Run Ω S J) ⁻¹' A) := by
    rw [← hmap, Measure.map_apply (MeasurableEquiv.prodAssoc).measurable hA]
  rw [hpre, Measure.prod_apply ((MeasurableEquiv.prodAssoc).measurable hA)]
  calc ∫⁻ y, νc (Prod.mk y ⁻¹' (MeasurableEquiv.prodAssoc ⁻¹' A)) ∂(μ.prod νsq)
      ≤ ∫⁻ _, E ∂(μ.prod νsq) := lintegral_mono (fun y => h y)
    _ = E := by simp

/-- The certification stream's own law. -/
lemma map_certStream (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] :
    Measure.map (fun x : Run Ω S J => x.2.2) (runLaw μ D Dsf)
      = Measure.infinitePi fun z : J × ℕ => D z.1 := by
  rw [show (fun x : Run Ω S J => x.2.2) = Prod.snd ∘ Prod.snd from rfl,
    ← Measure.map_map measurable_snd measurable_snd, runLaw, Measure.map_snd_prod]
  simp only [measure_univ, one_smul]
  rw [Measure.map_snd_prod]
  simp

open scoped Classical in
/-- How many of the first `m` certification draws land in a set, as a measurable function of
the stream. -/
lemma measurable_certHits (W : Set S) (j : J) (m : ℕ) :
    Measurable (fun c : J × ℕ → S =>
      ((((Finset.range m).filter (fun i => c (j, i) ∈ W)).card : ℝ))) := by
  classical
  have hW : MeasurableSet W := (Set.to_countable _).measurableSet
  have hrw : (fun c : J × ℕ → S =>
      ((((Finset.range m).filter (fun i => c (j, i) ∈ W)).card : ℝ)))
      = fun c => ∑ i ∈ Finset.range m, W.indicator (fun _ => (1 : ℝ)) (c (j, i)) := by
    funext c
    rw [Finset.card_filter, Nat.cast_sum]
    refine Finset.sum_congr rfl (fun i _ => ?_)
    by_cases h : c (j, i) ∈ W <;> simp [h]
  rw [hrw]
  exact Finset.measurable_sum _ (fun i _ =>
    (measurable_const.indicator hW).comp (measurable_pi_apply (j, i)))

open scoped Classical in
/-- **The certification sample carries its share of each class.**  `cert_hits_wrongSet` at
the label classes: a class of mass `q` is hit at least `m(q − t)` times, off an
`exp(-2 m t²)` set.  This is what discharges `ret_at_whp`'s class-count hypotheses — the
sides are populated because the population is. -/
theorem cert_class_count_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (j : J) (m : ℕ) (b q t : ℝ) (hq : 0 ≤ q) (ht : 0 ≤ t)
    (hmass : q ≤ (D j).real {p | O.label p = b}) :
    (runLaw μ D Dsf).real {x : Run Ω S J | (((Finset.range m).filter
        (fun i => O.label (cert j i x) = b)).card : ℝ) ≤ (m : ℝ) * (q - t)}
      ≤ Real.exp (-2 * (m : ℝ) * t ^ 2) := by
  classical
  set W : Set S := {p : S | O.label p = b} with hWdef
  have hmeasSet : MeasurableSet {c : J × ℕ → S | (((Finset.range m).filter
      (fun i => c (j, i) ∈ W)).card : ℝ) ≤ (m : ℝ) * (q - t)} :=
    measurableSet_le (measurable_certHits W j m) measurable_const
  have hpre : {x : Run Ω S J | (((Finset.range m).filter
      (fun i => O.label (cert j i x) = b)).card : ℝ) ≤ (m : ℝ) * (q - t)}
      = (fun x : Run Ω S J => x.2.2) ⁻¹' {c : J × ℕ → S | (((Finset.range m).filter
        (fun i => c (j, i) ∈ W)).card : ℝ) ≤ (m : ℝ) * (q - t)} := rfl
  rw [hpre, measureReal_def, Measure.map_apply (by fun_prop) hmeasSet
    |>.symm.trans (congrArg (fun ν : Measure (J × ℕ → S) => ν _) (map_certStream D Dsf)),
    ← measureReal_def]
  exact cert_hits_wrongSet D j m W q t hq ht hmass

/-- One certification coordinate has the population's own law. -/
lemma map_certCoord (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (j : J) (i : ℕ) :
    Measure.map (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => d.2 (j, i))
        (drawLaw D Dsf) = D j := by
  have hstep : (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => d.2 (j, i))
      = (fun c : J × ℕ → S => c (j, i)) ∘ Prod.snd := rfl
  rw [hstep, ← Measure.map_map (by fun_prop) measurable_snd, drawLaw, Measure.map_snd_prod]
  simp only [measure_univ, one_smul]
  exact (measurePreserving_eval_infinitePi (fun z : J × ℕ => D z.1) (j, i)).map_eq

/-- The certification prefixes land in the flat set too, for the same reason the table's
do. -/
lemma ae_cert_mem_Pre (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (Pre : Set S)
    (populations : Finset J) (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) :
    ∀ᵐ d ∂(drawLaw D Dsf), ∀ j ∈ populations, ∀ i : ℕ, d.2 (j, i) ∈ Pre := by
  have hmeasPre : MeasurableSet (Preᶜ : Set S) := (Set.to_countable _).measurableSet
  have hcoord : ∀ z : J × ℕ, ∀ᵐ d ∂(drawLaw D Dsf),
      z.1 ∈ populations → d.2 (z.1, z.2) ∈ Pre := by
    rintro ⟨j, i⟩
    by_cases hj : j ∈ populations
    · have hz : drawLaw D Dsf {d | ¬ (j ∈ populations → d.2 (j, i) ∈ Pre)} = 0 := by
        have hset : {d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S))
            | ¬ (j ∈ populations → d.2 (j, i) ∈ Pre)}
            = (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => d.2 (j, i)) ⁻¹' Preᶜ := by
          ext d; simp [hj]
        rw [hset, ← Measure.map_apply (by fun_prop) hmeasPre, map_certCoord D Dsf j i]
        exact hsupp j hj
      exact ae_iff.2 hz
    · filter_upwards with d hjj
      exact absurd hjj hj
  filter_upwards [ae_all_iff.2 hcoord] with d hd j hj i
  exact hd (j, i) hj

open scoped Classical in
lemma map_certBlock (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (j : J) (m : ℕ) :
    Measure.map (fun x : Run Ω S J => (fun i : Fin m => cert j i.val x)) (runLaw μ D Dsf)
      = Measure.pi (fun _ : Fin m => D j) := by
  have hstep : (fun x : Run Ω S J => (fun i : Fin m => cert j i.val x))
      = (fun c : J × ℕ → S => (fun i : Fin m => c (j, i.val)))
        ∘ (fun x : Run Ω S J => x.2.2) := rfl
  rw [hstep, ← Measure.map_map (by fun_prop) (by fun_prop), map_certStream D Dsf]
  refine (Measure.pi_eq (μ := fun _ : Fin m => D j) fun t ht => ?_).symm
  have hpre : (fun c : J × ℕ → S => (fun i : Fin m => c (j, i.val))) ⁻¹' Set.univ.pi t
      = Set.pi ↑((Finset.range m).image (fun i => (j, i)))
        (fun z => if h : z.2 < m then t ⟨z.2, h⟩ else Set.univ) := by
    ext c
    simp only [Set.mem_preimage, Set.mem_pi, Set.mem_univ, forall_const, Finset.coe_image,
      Finset.coe_range, Set.mem_image, Set.mem_Iio]
    constructor
    · rintro h z ⟨a, ha, rfl⟩
      rw [dif_pos ha]
      exact h ⟨a, ha⟩
    · intro h i
      have := h (j, i.val) ⟨i.val, i.isLt, rfl⟩
      simpa [dif_pos i.isLt] using this
  rw [Measure.map_apply (by fun_prop) (MeasurableSet.univ_pi ht), hpre,
    Measure.infinitePi_pi]
  · rw [Finset.prod_image (fun a _ b _ h => (Prod.mk.inj h).2), ← Fin.prod_univ_eq_prod_range]
    exact Finset.prod_congr rfl fun i _ => by simp [dif_pos i.isLt]
  · intro z _
    split_ifs with h
    exacts [ht ⟨z.2, h⟩, .univ]

/-- A table prefix and a certification prefix are drawn from independent streams, so their
joint law is the product — which is what lets `cross_collision_le` price a collision between
the two. -/
lemma map_prefCertPair (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (j' j : J) (i i' : ℕ) :
    Measure.map (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => (d.1.2 j' i, d.2 (j, i')))
        (drawLaw D Dsf) = (D j').prod (D j) := by
  have hf : Measurable (fun y : (ℕ → S) × (J → ℕ → S) => y.2 j' i) := by fun_prop
  have hg : Measurable (fun c : J × ℕ → S => c (j, i')) := by fun_prop
  have hmapf : Measure.map (fun y : (ℕ → S) × (J → ℕ → S) => y.2 j' i)
      ((Measure.infinitePi fun _ : ℕ => Dsf).prod
        (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j)) = D j' := by
    rw [show (fun y : (ℕ → S) × (J → ℕ → S) => y.2 j' i)
        = (fun q : J → ℕ → S => q j' i) ∘ Prod.snd from rfl,
      ← Measure.map_map (by fun_prop) measurable_snd, Measure.map_snd_prod]
    simp only [measure_univ, one_smul]
    rw [show (fun q : J → ℕ → S => q j' i)
        = (fun r : ℕ → S => r i) ∘ (fun q : J → ℕ → S => q j') from rfl,
      ← Measure.map_map (by fun_prop) (by fun_prop),
      (measurePreserving_eval (fun j : J => Measure.infinitePi fun _ : ℕ => D j) j').map_eq,
      (measurePreserving_eval_infinitePi (fun _ : ℕ => D j') i).map_eq]
  have hmapg : Measure.map (fun c : J × ℕ → S => c (j, i'))
      (Measure.infinitePi fun z : J × ℕ => D z.1) = D j :=
    (measurePreserving_eval_infinitePi (fun z : J × ℕ => D z.1) (j, i')).map_eq
  rw [show (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => (d.1.2 j' i, d.2 (j, i')))
      = Prod.map (fun y : (ℕ → S) × (J → ℕ → S) => y.2 j' i)
        (fun c : J × ℕ → S => c (j, i')) from rfl,
    drawLaw, ← Measure.map_prod_map _ _ hf hg, hmapf, hmapg]

/-! ### Unioning over a drawn pool

The candidates are drawn, so a union bound over them is a union over an `x`-dependent set.
What makes it cost `M` rather than everything is that a candidate's index is drawn from the
suffix stream while the event it indexes lives on the prefix streams — a different factor of
the same product — so slicing costs nothing. -/

lemma map_drawBlock (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] :
    Measure.map (fun x : Run Ω S J => x.2.1) (runLaw μ D Dsf)
      = (Measure.infinitePi fun _ : ℕ => Dsf).prod
          (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) := by
  rw [show (fun x : Run Ω S J => x.2.1) = Prod.fst ∘ Prod.snd from rfl,
    ← Measure.map_map measurable_fst measurable_snd, runLaw, Measure.map_snd_prod]
  simp only [measure_univ, one_smul]
  rw [Measure.map_fst_prod]
  simp only [measure_univ, one_smul]

/-- **A drawn index costs nothing.**  The event is indexed by a suffix draw and decided by
the prefix draws, so the worst case over candidates bounds the run. -/
theorem runLaw_draw_selection_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (C : S → Set (J → ℕ → S)) (hC : ∀ v, MeasurableSet (C v)) (i : ℕ) (E : ℝ≥0∞)
    (hbad : ∀ v, (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) (C v) ≤ E) :
    runLaw μ D Dsf {x : Run Ω S J | x.2.1.2 ∈ C (sfx i x)} ≤ E := by
  classical
  set νs : Measure (ℕ → S) := Measure.infinitePi fun _ : ℕ => Dsf with hνs
  set νq : Measure (J → ℕ → S) := Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j
    with hνq
  have hmeasSet : MeasurableSet {y : (ℕ → S) × (J → ℕ → S) | y.2 ∈ C (y.1 i)} := by
    have hcov : {y : (ℕ → S) × (J → ℕ → S) | y.2 ∈ C (y.1 i)}
        = ⋃ a : S, ({y : (ℕ → S) × (J → ℕ → S) | y.1 i = a} ∩ {y | y.2 ∈ C a}) := by
      ext y
      simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff]
      exact ⟨fun h => ⟨y.1 i, rfl, h⟩, fun ⟨a, ha, h⟩ => ha ▸ h⟩
    rw [hcov]
    exact MeasurableSet.iUnion (fun a =>
      (measurableSet_eq_fun ((measurable_pi_apply i).comp measurable_fst) measurable_const).inter
        (measurable_snd (hC a)))
  have hpre : {x : Run Ω S J | x.2.1.2 ∈ C (sfx i x)}
      = (fun x : Run Ω S J => x.2.1) ⁻¹' {y : (ℕ → S) × (J → ℕ → S) | y.2 ∈ C (y.1 i)} := rfl
  rw [hpre, ← Measure.map_apply (by fun_prop) hmeasSet, map_drawBlock D Dsf,
    Measure.prod_apply hmeasSet]
  calc ∫⁻ a, νq (Prod.mk a ⁻¹' {y : (ℕ → S) × (J → ℕ → S) | y.2 ∈ C (y.1 i)}) ∂νs
      = ∫⁻ a, νq (C (a i)) ∂νs := by rfl
    _ ≤ ∫⁻ _, E ∂νs := lintegral_mono (fun a => hbad (a i))
    _ = E := by simp

/-! ### Level 2: the empirical flip rate is honest

The clustering only ever sees a candidate's flips on the prefixes actually drawn.  Those
are `m` i.i.d. draws from the population, so the count concentrates around `m · flipMass`
and a candidate that looks clean on them is clean. -/

lemma map_prefixBlock (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (j : J) (m : ℕ) :
    Measure.map (fun x : Run Ω S J => (fun i : Fin m => prf j i.val x)) (runLaw μ D Dsf)
      = Measure.pi (fun _ : Fin m => D j) := by
  have hstep : (fun x : Run Ω S J => (fun i : Fin m => prf j i.val x))
      = (fun y : ((Fin m → S) × (J → Fin m → S)) => y.2 j)
        ∘ (fun x : Run Ω S J => ((fun i : Fin m => sfx i.val x),
            (fun (j : J) (i : Fin m) => prf j i.val x))) := rfl
  have hmeasBlock : Measurable (fun x : Run Ω S J => ((fun i : Fin m => sfx i.val x),
      (fun (j : J) (i : Fin m) => prf j i.val x))) :=
    (measurable_pi_lambda _ (fun i : Fin m => measurable_sfx i.val)).prodMk
      (measurable_pi_lambda _ (fun j : J =>
        measurable_pi_lambda _ (fun i : Fin m => measurable_prf j i.val)))
  rw [hstep, ← Measure.map_map (by fun_prop) hmeasBlock, law_block D Dsf m,
    show (fun y : ((Fin m → S) × (J → Fin m → S)) => y.2 j)
      = (fun q : J → Fin m → S => q j) ∘ Prod.snd from rfl,
    ← Measure.map_map (by fun_prop) measurable_snd, Measure.map_snd_prod]
  simp only [measure_univ, one_smul]
  exact (measurePreserving_eval (fun j : J => Measure.pi fun _ : Fin m => D j) j).map_eq

lemma pi_flip_mean (Dj : Measure S) [IsProbabilityMeasure Dj] (O : Oracle μ S) (m : ℕ)
    (v : S) (i : Fin m) :
    (Measure.pi fun _ : Fin m => Dj)[fun q : Fin m → S => O.flip v (q i)]
      = flipMass O Dj v := by
  have hmap : Measure.map (fun q : Fin m → S => q i) (Measure.pi fun _ : Fin m => Dj) = Dj :=
    (measurePreserving_eval (fun _ : Fin m => Dj) i).map_eq
  calc (Measure.pi fun _ : Fin m => Dj)[fun q : Fin m → S => O.flip v (q i)]
      = ∫ w, O.flip v w ∂(Measure.map (fun q : Fin m → S => q i)
          (Measure.pi fun _ : Fin m => Dj)) := by
        rw [integral_map (measurable_pi_apply i).aemeasurable
          (flip_meas O v).aestronglyMeasurable]
    _ = flipMass O Dj v := by rw [hmap]; rfl

/-- **Level 2 for one population.**  A candidate the drawn prefixes say is clean really is. -/
theorem prefix_flip_lower (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (j : J) (m : ℕ) (v : S) (g : ℝ) (hg : 0 ≤ g) :
    (runLaw μ D Dsf).real {x : Run Ω S J | ∑ i : Fin m, O.flip v (prf j i.val x)
        ≤ (m : ℝ) * (flipMass O (D j) v - g)}
      ≤ Real.exp (-2 * (m : ℝ) * g ^ 2) := by
  classical
  have hcard : ((Finset.univ : Finset (Fin m)).card : ℝ) = (m : ℝ) := by simp
  have htail := sumLower_le (μ := Measure.pi fun _ : Fin m => D j)
    (fun (i : Fin m) (q : Fin m → S) => O.flip v (q i)) (Finset.univ : Finset (Fin m))
    (flipMass O (D j) v) g
    (fun i => ((flip_meas O v).comp (measurable_pi_apply i)).aemeasurable)
    (iIndepFun_pi (fun _ => (flip_meas O v).aemeasurable))
    (fun i => Filter.Eventually.of_forall (fun q => flip_icc O v (q i)))
    (by rw [Finset.sum_congr rfl (fun i _ => pi_flip_mean (D j) O m v i), Finset.sum_const,
      nsmul_eq_mul, hcard]) hg
  rw [hcard] at htail
  have hpre : {x : Run Ω S J | ∑ i : Fin m, O.flip v (prf j i.val x)
      ≤ (m : ℝ) * (flipMass O (D j) v - g)}
      = (fun x : Run Ω S J => (fun i : Fin m => prf j i.val x)) ⁻¹'
        {q : Fin m → S | ∑ i : Fin m, O.flip v (q i) ≤ (m : ℝ) * (flipMass O (D j) v - g)} := rfl
  have hmeasPrf : Measurable (fun x : Run Ω S J => (fun i : Fin m => prf j i.val x)) :=
    measurable_pi_lambda _ (fun i : Fin m => measurable_prf j i.val)
  rw [hpre, measureReal_def,
    Measure.map_apply hmeasPrf (measurableSet_le (by fun_prop) measurable_const)
      |>.symm.trans (congrArg (fun ν : Measure (Fin m → S) => ν _) (map_prefixBlock D Dsf j m)),
    ← measureReal_def]
  exact htail

lemma runLaw_prefix_apply (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (C : Set (J → ℕ → S)) (hC : MeasurableSet C) :
    runLaw μ D Dsf {x : Run Ω S J | x.2.1.2 ∈ C}
      = (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) C := by
  have hpre : {x : Run Ω S J | x.2.1.2 ∈ C} = (fun x : Run Ω S J => x.2.1) ⁻¹' (Prod.snd ⁻¹' C) :=
    rfl
  rw [hpre, ← Measure.map_apply (by fun_prop) (measurable_snd hC), map_drawBlock D Dsf,
    ← Measure.map_apply measurable_snd hC, Measure.map_snd_prod]
  simp

open scoped Classical in
/-- The candidates the drawn prefixes understate: really flipping more than `Δ`, but
empirically inside `Δ - g`. -/
noncomputable def understated (O : Oracle μ S) (D : J → Measure S) (j : J) (m : ℕ)
    (Δ g : ℝ) (v : S) : Set (J → ℕ → S) :=
  {q | Δ < flipMass O (D j) v ∧
    ∑ i : Fin m, O.flip v (q j i.val) ≤ (m : ℝ) * (Δ - g)}

lemma measurableSet_understated (O : Oracle μ S) (D : J → Measure S) (j : J) (m : ℕ)
    (Δ g : ℝ) (v : S) : MeasurableSet (understated O D j m Δ g v) := by
  classical
  have hsum : MeasurableSet
      {q : J → ℕ → S | ∑ i : Fin m, O.flip v (q j i.val) ≤ (m : ℝ) * (Δ - g)} :=
    measurableSet_le
      (Finset.measurable_sum _ (fun i _ => (flip_meas O v).comp
        ((measurable_pi_apply i.val).comp (measurable_pi_apply j)))) measurable_const
  by_cases h : Δ < flipMass O (D j) v
  · simpa [understated, h] using hsum
  · simpa [understated, h] using MeasurableSet.empty

lemma measure_understated_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (j : J) (m : ℕ) (Δ g : ℝ) (hg : 0 ≤ g) (v : S) :
    (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) (understated O D j m Δ g v)
      ≤ ENNReal.ofReal (Real.exp (-2 * (m : ℝ) * g ^ 2)) := by
  rw [← runLaw_prefix_apply (μ := μ) D Dsf _ (measurableSet_understated O D j m Δ g v)]
  have hsub : {x : Run Ω S J | x.2.1.2 ∈ understated O D j m Δ g v}
      ⊆ {x : Run Ω S J | ∑ i : Fin m, O.flip v (prf j i.val x)
          ≤ (m : ℝ) * (flipMass O (D j) v - g)} := by
    rintro x ⟨hΔ, hcount⟩
    have hm : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
    exact le_trans hcount (by nlinarith)
  refine le_trans (measure_mono hsub) ?_
  rw [← ENNReal.ofReal_toReal (measure_ne_top (runLaw μ D Dsf) _), ← measureReal_def]
  exact ENNReal.ofReal_le_ofReal (prefix_flip_lower D Dsf O j m v g hg)

/-- **The whole drawn pool is honest at once.**  A pool member the drawn prefixes say sits
inside `Δ - g` really flips at most `Δ` of the population, off an `M · exp(-2 m g²)` set. -/
theorem pool_flipMass_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (j : J) (m M : ℕ) (Δ g : ℝ) (hg : 0 ≤ g) (hΔ : 0 ≤ Δ) :
    (runLaw μ D Dsf).real {x : Run Ω S J | ¬ ∀ v ∈ poolAt M x,
        (∑ i : Fin m, O.flip v (prf j i.val x) ≤ (m : ℝ) * (Δ - g)) → flipMass O (D j) v ≤ Δ}
      ≤ (M : ℝ) * Real.exp (-2 * (m : ℝ) * g ^ 2) := by
  classical
  set E : ℝ≥0∞ := ENNReal.ofReal (Real.exp (-2 * (m : ℝ) * g ^ 2)) with hE
  have hseed : flipMass O (D j) (1 : S) = 0 := by
    have : ∀ p : S, O.flip (1 : S) p = 0 := by
      intro p
      show O.label (p * 1) + O.label p - 2 * O.label (p * 1) * O.label p = 0
      rw [mul_one]
      rcases O.label_bit p with hl | hl <;> rw [hl] <;> ring
    simp [flipMass, this]
  have hsub : {x : Run Ω S J | ¬ ∀ v ∈ poolAt M x,
      (∑ i : Fin m, O.flip v (prf j i.val x) ≤ (m : ℝ) * (Δ - g)) → flipMass O (D j) v ≤ Δ}
      ⊆ ⋃ i ∈ Finset.range M, {x : Run Ω S J | x.2.1.2 ∈ understated O D j m Δ g (sfx i x)} := by
    intro x hx
    simp only [Set.mem_setOf_eq, not_forall] at hx
    obtain ⟨v, hv, hcount, hmass⟩ := hx
    rcases Finset.mem_insert.1 hv with rfl | hv'
    · exact absurd (hseed ▸ hΔ) hmass
    · obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hv'
      exact Set.mem_biUnion hi ⟨not_le.1 hmass, hcount⟩
  have hbound : runLaw μ D Dsf {x : Run Ω S J | ¬ ∀ v ∈ poolAt M x,
      (∑ i : Fin m, O.flip v (prf j i.val x) ≤ (m : ℝ) * (Δ - g)) → flipMass O (D j) v ≤ Δ}
      ≤ (M : ℝ≥0∞) * E := by
    refine le_trans (measure_mono hsub) (le_trans (measure_biUnion_finset_le _ _) ?_)
    calc ∑ i ∈ Finset.range M,
          runLaw μ D Dsf {x : Run Ω S J | x.2.1.2 ∈ understated O D j m Δ g (sfx i x)}
        ≤ ∑ _i ∈ Finset.range M, E :=
          Finset.sum_le_sum (fun i _ => runLaw_draw_selection_le D Dsf _
            (measurableSet_understated O D j m Δ g) i E
            (fun v => measure_understated_le (μ := μ) D Dsf O j m Δ g hg v))
      _ = (M : ℝ≥0∞) * E := by rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  rw [measureReal_def]
  calc (runLaw μ D Dsf {x : Run Ω S J | ¬ ∀ v ∈ poolAt M x,
          (∑ i : Fin m, O.flip v (prf j i.val x) ≤ (m : ℝ) * (Δ - g))
            → flipMass O (D j) v ≤ Δ}).toReal
      ≤ ((M : ℝ≥0∞) * E).toReal :=
        ENNReal.toReal_mono (by simp [hE, ENNReal.mul_eq_top]) hbound
    _ = (M : ℝ) * Real.exp (-2 * (m : ℝ) * g ^ 2) := by
        rw [ENNReal.toReal_mul, hE, ENNReal.toReal_ofReal (Real.exp_nonneg _)]
        simp

/-! ### The run's set-valued data is measurable

The table, the pool and the family are `Finset`s read off finitely many draws and finitely
many oracle bits.  `Finset S` is countable, so every event about them is a countable union
of coordinate fibres — no factorisation theorem, the same move as `measurableSet_filter_pred`
one level up. -/

lemma measurableSet_finData {ι : Type*} [Fintype ι] (c : ι → Run Ω S J → S)
    (hc : ∀ z, Measurable (c z)) (g : (ι → S) → Finset S) (Q : Finset S) :
    MeasurableSet {x : Run Ω S J | g (fun z => c z x) = Q} := by
  classical
  have hcov : {x : Run Ω S J | g (fun z => c z x) = Q}
      = ⋃ t : {t : ι → S // g t = Q}, ⋂ z : ι, {x : Run Ω S J | c z x = t.val z} := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_iInter]
    refine ⟨fun h => ⟨⟨fun z => c z x, h⟩, fun z => rfl⟩, ?_⟩
    rintro ⟨⟨t, ht⟩, hx⟩
    rw [show (fun z => c z x) = t from funext hx]
    exact ht
  rw [hcov]
  exact MeasurableSet.iUnion (fun t => MeasurableSet.iInter (fun z =>
    (hc z) (measurableSet_singleton (t.val z))))

open scoped Classical in
lemma image_range_eq_image_univ {α : Type*} [DecidableEq α] (m : ℕ) (f : ℕ → α) :
    (Finset.range m).image f = (Finset.univ : Finset (Fin m)).image (fun i => f i.val) := by
  classical
  ext a
  simp only [Finset.mem_image, Finset.mem_range, Finset.mem_univ, true_and]
  exact ⟨fun ⟨i, hi, h⟩ => ⟨⟨i, hi⟩, h⟩, fun ⟨i, h⟩ => ⟨i.val, i.isLt, h⟩⟩

open scoped Classical in
lemma measurableSet_prefixesAt (populations : Finset J) (m : ℕ) (Q : Finset S) :
    MeasurableSet {x : Run Ω S J | prefixesAt populations m x = Q} := by
  classical
  have hrw : ∀ x : Run Ω S J, prefixesAt populations m x
      = populations.biUnion (fun j => (Finset.univ : Finset (Fin m)).image
          (fun i => (fun z : J × Fin m => prf z.1 z.2.val x) (j, i))) := by
    intro x
    exact Finset.biUnion_congr rfl (fun j _ => image_range_eq_image_univ m (fun i => prf j i x))
  simp only [hrw]
  exact measurableSet_finData (fun z : J × Fin m => prf z.1 z.2.val)
    (fun z => measurable_prf z.1 z.2.val)
    (fun t => populations.biUnion (fun j => (Finset.univ : Finset (Fin m)).image
      (fun i => t (j, i)))) Q

open scoped Classical in
lemma measurableSet_poolAt (M : ℕ) (C : Finset S) :
    MeasurableSet {x : Run Ω S J | poolAt M x = C} := by
  classical
  have hrw : ∀ x : Run Ω S J, poolAt M x
      = insert 1 ((Finset.univ : Finset (Fin M)).image
          (fun i => (fun z : Fin M => sfx z.val x) i)) := by
    intro x
    exact congrArg (insert 1) (image_range_eq_image_univ M (fun i => sfx i x))
  simp only [hrw]
  exact measurableSet_finData (fun z : Fin M => sfx z.val) (fun z => measurable_sfx z.val)
    (fun t => insert 1 ((Finset.univ : Finset (Fin M)).image (fun i => t i))) C

open scoped Classical in
/-- **Events about the table and the pool are measurable.**  Decompose over their values,
of which there are countably many. -/
lemma measurableSet_of_run_data (populations : Finset J) (B : Budget)
    (R : Finset S → Finset S → Set (Run Ω S J)) (hR : ∀ P C, MeasurableSet (R P C)) :
    MeasurableSet {x : Run Ω S J | x ∈ R (prefixesAt populations B.m x) (poolAt B.M x)} := by
  classical
  have hcov : {x : Run Ω S J | x ∈ R (prefixesAt populations B.m x) (poolAt B.M x)}
      = ⋃ z : Finset S × Finset S, (({x : Run Ω S J | prefixesAt populations B.m x = z.1}
          ∩ {x : Run Ω S J | poolAt B.M x = z.2}) ∩ R z.1 z.2) := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff]
    refine ⟨fun h => ⟨(prefixesAt populations B.m x, poolAt B.M x), ⟨rfl, rfl⟩, h⟩, ?_⟩
    rintro ⟨⟨P, C⟩, ⟨hP, hC⟩, hx⟩
    simp only at hP hC
    rw [hP, hC]
    exact hx
  rw [hcov]
  exact MeasurableSet.iUnion (fun z =>
    ((measurableSet_prefixesAt populations B.m z.1).inter
      (measurableSet_poolAt B.M z.2)).inter (hR z.1 z.2))

lemma measureReal_le_one' (Dj : Measure S) [IsProbabilityMeasure Dj] (A : Set S) :
    Dj.real A ≤ 1 := by
  have h := measureReal_mono (μ := Dj) (Set.subset_univ A) (by finiteness)
  simpa using h

lemma summable_singleton_real (Dj : Measure S) [IsProbabilityMeasure Dj] :
    Summable (fun a : S => Dj.real {a}) := by
  classical
  refine summable_of_sum_le (c := 1) (fun a => measureReal_nonneg) (fun Q => ?_)
  rw [sum_measureReal_singleton]
  exact measureReal_le_one' Dj _

lemma summable_singleton_sq (Dj : Measure S) [IsProbabilityMeasure Dj] :
    Summable (fun a : S => Dj.real {a} ^ 2) := by
  refine Summable.of_nonneg_of_le (fun a => sq_nonneg _) (fun a => ?_) (summable_singleton_real Dj)
  nlinarith [measureReal_nonneg (μ := Dj) (s := ({a} : Set S)), measureReal_le_one' Dj ({a} : Set S)]

/-- **The table carries little of the population.**  The clustering read the oracle at the
table's prefixes, so the cut there is not covered by the independence argument.  No single
prefix can carry more than `√ρ`, since its own square is already inside the collision
mass. -/
lemma measureReal_singleton_le (Dj : Measure S) [IsProbabilityMeasure Dj] (ρ : ℝ)
    (hρ : collisionMass Dj ≤ ρ) (a : S) : Dj.real {a} ≤ Real.sqrt ρ := by
  have hmem : Dj.real {a} ^ 2 ≤ collisionMass Dj := by
    refine le_trans (le_of_eq ?_) (Summable.le_tsum (summable_singleton_sq Dj) a
      (fun b _ => sq_nonneg _))
    rfl
  calc Dj.real {a} = Real.sqrt (Dj.real {a} ^ 2) := (Real.sqrt_sq measureReal_nonneg).symm
    _ ≤ Real.sqrt ρ := Real.sqrt_le_sqrt (le_trans hmem hρ)

lemma measureReal_finset_le (Dj : Measure S) [IsProbabilityMeasure Dj] (Q : Finset S) (ρ : ℝ)
    (hρ : collisionMass Dj ≤ ρ) : Dj.real ↑Q ≤ (Q.card : ℝ) * Real.sqrt ρ := by
  classical
  rw [← sum_measureReal_singleton (μ := Dj) Q]
  calc ∑ a ∈ Q, Dj.real {a} ≤ ∑ _a ∈ Q, Real.sqrt ρ :=
        Finset.sum_le_sum (fun a _ => measureReal_singleton_le Dj ρ hρ a)
    _ = (Q.card : ℝ) * Real.sqrt ρ := by rw [Finset.sum_const, nsmul_eq_mul]

/-! ### The pool's accept-preserving candidates

The clustering's ranking is only as good as what the pool offers it: the argument needs `k`
candidates that flip nothing.  They are a `pAP` fraction of `Dsf`, so `M` draws deliver
them — up to the usual two corrections, the binomial tail and the draws being distinct. -/

/-- The indicator of accept-preservation. -/
noncomputable def apBit (O : Oracle μ S) (v : S) : ℝ :=
  Set.indicator {w : S | ∀ p : S, O.label (p * w) = O.label p} (fun _ => (1 : ℝ)) v

lemma measurableSet_ap (O : Oracle μ S) :
    MeasurableSet {w : S | ∀ p : S, O.label (p * w) = O.label p} :=
  (Set.to_countable _).measurableSet

lemma apBit_meas (O : Oracle μ S) : Measurable (apBit O) :=
  measurable_const.indicator (measurableSet_ap O)

lemma apBit_icc (O : Oracle μ S) (v : S) : apBit O v ∈ Set.Icc (0 : ℝ) 1 := by
  by_cases h : v ∈ {w : S | ∀ p : S, O.label (p * w) = O.label p} <;>
    simp [apBit, Set.indicator_apply, h]

lemma integral_apBit (O : Oracle μ S) (Dsf : Measure S) [IsProbabilityMeasure Dsf] :
    Dsf[apBit O] = Dsf.real {w : S | ∀ p : S, O.label (p * w) = O.label p} :=
  integral_indicator_one (measurableSet_ap O)

lemma map_suffixBlock (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (M : ℕ) :
    Measure.map (fun x : Run Ω S J => (fun i : Fin M => sfx i.val x)) (runLaw μ D Dsf)
      = Measure.pi (fun _ : Fin M => Dsf) := by
  have hstep : (fun x : Run Ω S J => (fun i : Fin M => sfx i.val x))
      = Prod.fst ∘ (fun x : Run Ω S J => ((fun i : Fin M => sfx i.val x),
          (fun (j : J) (i : Fin M) => prf j i.val x))) := rfl
  have hmeasBlock : Measurable (fun x : Run Ω S J => ((fun i : Fin M => sfx i.val x),
      (fun (j : J) (i : Fin M) => prf j i.val x))) :=
    (measurable_pi_lambda _ (fun i : Fin M => measurable_sfx i.val)).prodMk
      (measurable_pi_lambda _ (fun j : J =>
        measurable_pi_lambda _ (fun i : Fin M => measurable_prf j i.val)))
  rw [hstep, ← Measure.map_map measurable_fst hmeasBlock, law_block D Dsf M,
    Measure.map_fst_prod]
  simp

/-- **Enough of the pool preserves acceptance.**  The suffix draws are i.i.d., so the count
falls below `M(pAP − g)` only on an `exp(-2 M g²)` set. -/
theorem pool_ap_count_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S) (M : ℕ)
    (pAP g : ℝ) (hg : 0 ≤ g)
    (hpAP : pAP ≤ Dsf.real {w : S | ∀ p : S, O.label (p * w) = O.label p}) :
    (runLaw μ D Dsf).real {x : Run Ω S J | ∑ i : Fin M, apBit O (sfx i.val x)
        ≤ (M : ℝ) * (pAP - g)}
      ≤ Real.exp (-2 * (M : ℝ) * g ^ 2) := by
  classical
  have hcard : ((Finset.univ : Finset (Fin M)).card : ℝ) = (M : ℝ) := by simp
  have hmean : ∀ i : Fin M,
      (Measure.pi fun _ : Fin M => Dsf)[fun q : Fin M → S => apBit O (q i)]
        = Dsf.real {w : S | ∀ p : S, O.label (p * w) = O.label p} := by
    intro i
    have hmap : Measure.map (fun q : Fin M → S => q i) (Measure.pi fun _ : Fin M => Dsf) = Dsf :=
      (measurePreserving_eval (fun _ : Fin M => Dsf) i).map_eq
    calc (Measure.pi fun _ : Fin M => Dsf)[fun q : Fin M → S => apBit O (q i)]
        = ∫ w, apBit O w ∂(Measure.map (fun q : Fin M → S => q i)
            (Measure.pi fun _ : Fin M => Dsf)) := by
          rw [integral_map (measurable_pi_apply i).aemeasurable
            (apBit_meas O).aestronglyMeasurable]
      _ = Dsf.real {w : S | ∀ p : S, O.label (p * w) = O.label p} := by
          rw [hmap, integral_apBit O Dsf]
  have htail := sumLower_le (μ := Measure.pi fun _ : Fin M => Dsf)
    (fun (i : Fin M) (q : Fin M → S) => apBit O (q i)) (Finset.univ : Finset (Fin M)) pAP g
    (fun i => ((apBit_meas O).comp (measurable_pi_apply i)).aemeasurable)
    (iIndepFun_pi (fun _ => (apBit_meas O).aemeasurable))
    (fun i => Filter.Eventually.of_forall (fun q => apBit_icc O (q i)))
    (by
      rw [Finset.sum_congr rfl (fun i _ => hmean i), Finset.sum_const, nsmul_eq_mul, hcard]
      exact mul_le_mul_of_nonneg_left hpAP (Nat.cast_nonneg M)) hg
  rw [hcard] at htail
  have hmeasSfx : Measurable (fun x : Run Ω S J => (fun i : Fin M => sfx i.val x)) :=
    measurable_pi_lambda _ (fun i : Fin M => measurable_sfx i.val)
  have hpre : {x : Run Ω S J | ∑ i : Fin M, apBit O (sfx i.val x) ≤ (M : ℝ) * (pAP - g)}
      = (fun x : Run Ω S J => (fun i : Fin M => sfx i.val x)) ⁻¹'
        {q : Fin M → S | ∑ i : Fin M, apBit O (q i) ≤ (M : ℝ) * (pAP - g)} := rfl
  rw [hpre, measureReal_def,
    Measure.map_apply hmeasSfx (measurableSet_le (by fun_prop) measurable_const)
      |>.symm.trans (congrArg (fun ν : Measure (Fin M → S) => ν _) (map_suffixBlock D Dsf M)),
    ← measureReal_def]
  exact htail

/-- The suffix draws are distinct, except on an `M²ρ` set — the pool is interned, so the
indices that preserve acceptance only become that many *candidates* when they differ. -/
theorem suffix_not_injective_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (M : ℕ) (ρ : ℝ)
    (hρ : collisionMass Dsf ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runLaw μ D Dsf).real
        {x : Run Ω S J | ¬ Function.Injective (fun i : Fin M => sfx i.val x)}
      ≤ (M : ℝ) ^ 2 * ρ := by
  classical
  have hmeasSfx : Measurable (fun x : Run Ω S J => (fun i : Fin M => sfx i.val x)) :=
    measurable_pi_lambda _ (fun i : Fin M => measurable_sfx i.val)
  have hmeasSet : MeasurableSet {q : Fin M → S | ¬ Function.Injective q} := by
    have hcov : {q : Fin M → S | ¬ Function.Injective q}
        = ⋃ z : {z : Fin M × Fin M // z.1 ≠ z.2}, {q : Fin M → S | q z.val.1 = q z.val.2} := by
      ext q
      simp only [Set.mem_setOf_eq, Set.mem_iUnion, Function.not_injective_iff]
      constructor
      · rintro ⟨a, b, hab, hne⟩; exact ⟨⟨(a, b), hne⟩, hab⟩
      · rintro ⟨⟨⟨a, b⟩, hne⟩, hab⟩; exact ⟨a, b, hab, hne⟩
    rw [hcov]
    exact MeasurableSet.iUnion (fun z =>
      measurableSet_eq_fun (measurable_pi_apply _) (measurable_pi_apply _))
  have hpre : {x : Run Ω S J | ¬ Function.Injective (fun i : Fin M => sfx i.val x)}
      = (fun x : Run Ω S J => (fun i : Fin M => sfx i.val x)) ⁻¹'
        {q : Fin M → S | ¬ Function.Injective q} := rfl
  rw [hpre, measureReal_def, Measure.map_apply hmeasSfx hmeasSet
    |>.symm.trans (congrArg (fun ν : Measure (Fin M → S) => ν _) (map_suffixBlock D Dsf M)),
    ← measureReal_def]
  exact pi_not_injective_le Dsf M ρ hρ hρ0

open scoped Classical in
/-- **From accept-preserving indices to accept-preserving candidates.**  On distinct draws
the pool holds one candidate per index, so the index count is a lower bound on the
candidates the clustering's ranking can draw on.  Preserving acceptance everywhere is
stronger than flipping nothing on the table, which is what the ranking asks for. -/
lemma card_good_pool_ge (O : Oracle μ S) (M : ℕ) (P : Finset S) (x : Run Ω S J)
    (hinj : Function.Injective (fun i : Fin M => sfx i.val x)) :
    ∑ i : Fin M, apBit O (sfx i.val x)
      ≤ (((poolAt M x).filter (fun v => ∑ p ∈ P, O.flip v p = 0)).card : ℝ) := by
  classical
  set G : Finset (Fin M) := Finset.univ.filter
    (fun i => sfx i.val x ∈ {w : S | ∀ p : S, O.label (p * w) = O.label p}) with hG
  have hsum : ∑ i : Fin M, apBit O (sfx i.val x) = (G.card : ℝ) := by
    rw [hG, Finset.card_filter, Nat.cast_sum]
    refine Finset.sum_congr rfl (fun i _ => ?_)
    by_cases h : sfx i.val x ∈ {w : S | ∀ p : S, O.label (p * w) = O.label p} <;>
      simp [apBit, Set.indicator_apply, h]
  rw [hsum]
  refine Nat.cast_le.2 (Finset.card_le_card_of_injOn (fun i : Fin M => sfx i.val x) ?_ ?_)
  · intro i hi
    have hap : ∀ p : S, O.label (p * sfx i.val x) = O.label p := (Finset.mem_filter.1 hi).2
    refine Finset.mem_filter.2 ⟨Finset.mem_insert_of_mem (Finset.mem_image.2
      ⟨i.val, Finset.mem_range.2 i.isLt, rfl⟩), Finset.sum_eq_zero (fun p _ => ?_)⟩
    show O.label (p * sfx i.val x) + O.label p - 2 * O.label (p * sfx i.val x) * O.label p = 0
    rw [hap p]
    rcases O.label_bit p with hl | hl <;> rw [hl] <;> ring
  · intro a _ b _ hab
    exact hinj hab

/-! ### Part 2's first requirement: the vote is decisive

A family that preserves acceptance at `p` votes `Bernoulli(1−η)` there when `p` is
accepting and `Bernoulli(η)` when it is not, so the vote sits `s` away from the centre and
the band `[lo, hi]` only catches it on a deviation of `s − eps`.  This is `voteSum_upper`
and `voteSum_lower` at zero flips. -/

open scoped Classical in
theorem decided_whp (O : Oracle μ S) (F : Finset S) (p : S) (lo hi : ℕ) (γ : ℝ) (hγ : 0 ≤ γ)
    (hclean : flipCount O F p = 0)
    (hhi : (hi : ℝ) ≤ (F.card : ℝ) * ((1 - O.η) - γ))
    (hlo : (F.card : ℝ) * (O.η + γ) ≤ (lo : ℝ) + 1) :
    μ.real {ω | ¬ decided O lo hi F p ω} ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  have hf : flipCount O F p ≤ (F.card : ℝ) * 0 := by rw [hclean, mul_zero]
  rcases O.label_bit p with hp | hp
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_upper O F p hp 0 γ hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hgt : lo < voteCount O F p ω := by
      by_contra hc
      exact hbad (Or.inr (not_lt.1 hc))
    have hcast : (lo : ℝ) + 1 ≤ (voteCount O F p ω : ℝ) := by exact_mod_cast hgt
    show (F.card : ℝ) * ((O.η + (1 - 2 * O.η) * 0) + γ) ≤ voteSum O F p ω
    rw [← heq, mul_zero, add_zero]
    linarith
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_lower O F p hp 0 γ hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hle : voteCount O F p ω ≤ hi := by
      by_contra hc
      exact hbad (Or.inl (not_le.1 hc))
    have hcast : (voteCount O F p ω : ℝ) ≤ (hi : ℝ) := by exact_mod_cast hle
    show voteSum O F p ω ≤ (F.card : ℝ) * ((O.η + (1 - 2 * O.η) * (1 - 0)) - γ)
    rw [← heq]
    have : O.η + (1 - 2 * O.η) * (1 - 0) = 1 - O.η := by ring
    rw [this]
    linarith

/-! ### From the clustering's empirical bound to the population's

The clustering scores a candidate on the *deduplicated* table, the sampler draws `m` times
with replacement, and `flipMass` is about a fresh draw.  On the event that a population's
draws are distinct the three agree; `pi_not_injective_le` prices the rest at `m²ρ`. -/

open scoped Classical in
lemma sum_eq_sum_prefixesOf (O : Oracle μ S) (j : J) (m : ℕ) (x : Run Ω S J) (v : S)
    (hinj : Function.Injective (fun i : Fin m => prf j i.val x)) :
    ∑ i : Fin m, O.flip v (prf j i.val x) = ∑ p ∈ prefixesOf j m x, O.flip v p := by
  classical
  unfold prefixesOf
  rw [Finset.sum_image ?_, ← Fin.sum_univ_eq_sum_range]
  intro a ha b hb hab
  have := hinj (show (fun i : Fin m => prf j i.val x) ⟨a, Finset.mem_range.1 ha⟩
    = (fun i : Fin m => prf j i.val x) ⟨b, Finset.mem_range.1 hb⟩ from hab)
  simpa using congrArg Fin.val this

open scoped Classical in
lemma sum_prefixesOf_le_prefixesAt (O : Oracle μ S) (populations : Finset J) (j : J)
    (hj : j ∈ populations) (m : ℕ) (x : Run Ω S J) (v : S) :
    ∑ p ∈ prefixesOf j m x, O.flip v p ≤ ∑ p ∈ prefixesAt populations m x, O.flip v p := by
  classical
  refine Finset.sum_le_sum_of_subset_of_nonneg ?_ (fun p _ _ => (flip_icc O v p).1)
  exact fun p hp => Finset.mem_biUnion.2 ⟨j, hj, hp⟩

open scoped Classical in
lemma card_prefixesAt_le (populations : Finset J) (m : ℕ) (x : Run Ω S J) :
    (prefixesAt populations m x).card ≤ populations.card * m := by
  classical
  refine le_trans (Finset.card_biUnion_le) ?_
  calc ∑ j ∈ populations, (prefixesOf j m x).card
      ≤ ∑ _j ∈ populations, m :=
        Finset.sum_le_sum (fun j _ => le_trans Finset.card_image_le (by simp))
    _ = populations.card * m := by rw [Finset.sum_const, smul_eq_mul]

/-- The population's own draws are distinct, except on an `m²ρ` set. -/
theorem prefix_not_injective_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (j : J) (m : ℕ) (ρ : ℝ)
    (hρ : collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runLaw μ D Dsf).real
        {x : Run Ω S J | ¬ Function.Injective (fun i : Fin m => prf j i.val x)}
      ≤ (m : ℝ) ^ 2 * ρ := by
  classical
  have hmeasPrf : Measurable (fun x : Run Ω S J => (fun i : Fin m => prf j i.val x)) :=
    measurable_pi_lambda _ (fun i : Fin m => measurable_prf j i.val)
  have hpre : {x : Run Ω S J | ¬ Function.Injective (fun i : Fin m => prf j i.val x)}
      = (fun x : Run Ω S J => (fun i : Fin m => prf j i.val x)) ⁻¹'
        {p : Fin m → S | ¬ Function.Injective p} := rfl
  have hmeasSet : MeasurableSet {p : Fin m → S | ¬ Function.Injective p} := by
    have hcov : {p : Fin m → S | ¬ Function.Injective p}
        = ⋃ z : {z : Fin m × Fin m // z.1 ≠ z.2}, {p : Fin m → S | p z.val.1 = p z.val.2} := by
      ext p
      simp only [Set.mem_setOf_eq, Set.mem_iUnion, Function.not_injective_iff]
      constructor
      · rintro ⟨a, b, hab, hne⟩; exact ⟨⟨(a, b), hne⟩, hab⟩
      · rintro ⟨⟨⟨a, b⟩, hne⟩, hab⟩; exact ⟨a, b, hab, hne⟩
    rw [hcov]
    exact MeasurableSet.iUnion (fun z =>
      measurableSet_eq_fun (measurable_pi_apply _) (measurable_pi_apply _))
  rw [hpre, measureReal_def, Measure.map_apply hmeasPrf hmeasSet
    |>.symm.trans (congrArg (fun ν : Measure (Fin m → S) => ν _) (map_prefixBlock D Dsf j m)),
    ← measureReal_def]
  exact pi_not_injective_le (D j) m ρ hρ hρ0

/-- The certification draws are distinct too, on the same `m²ρ`.  Needed because the
gate's split counts *prefixes*, and a repeated draw is one prefix, not two. -/
theorem cert_not_injective_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (j : J) (m : ℕ) (ρ : ℝ)
    (hρ : collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runLaw μ D Dsf).real
        {x : Run Ω S J | ¬ Function.Injective (fun i : Fin m => cert j i.val x)}
      ≤ (m : ℝ) ^ 2 * ρ := by
  classical
  have hmeasPrf : Measurable (fun x : Run Ω S J => (fun i : Fin m => cert j i.val x)) :=
    measurable_pi_lambda _ (fun i : Fin m => measurable_cert j i.val)
  have hpre : {x : Run Ω S J | ¬ Function.Injective (fun i : Fin m => cert j i.val x)}
      = (fun x : Run Ω S J => (fun i : Fin m => cert j i.val x)) ⁻¹'
        {p : Fin m → S | ¬ Function.Injective p} := rfl
  have hmeasSet : MeasurableSet {p : Fin m → S | ¬ Function.Injective p} := by
    have hcov : {p : Fin m → S | ¬ Function.Injective p}
        = ⋃ z : {z : Fin m × Fin m // z.1 ≠ z.2}, {p : Fin m → S | p z.val.1 = p z.val.2} := by
      ext p
      simp only [Set.mem_setOf_eq, Set.mem_iUnion, Function.not_injective_iff]
      constructor
      · rintro ⟨a, b, hab, hne⟩; exact ⟨⟨(a, b), hne⟩, hab⟩
      · rintro ⟨⟨⟨a, b⟩, hne⟩, hab⟩; exact ⟨a, b, hab, hne⟩
    rw [hcov]
    exact MeasurableSet.iUnion (fun z =>
      measurableSet_eq_fun (measurable_pi_apply _) (measurable_pi_apply _))
  rw [hpre, measureReal_def, Measure.map_apply hmeasPrf hmeasSet
    |>.symm.trans (congrArg (fun ν : Measure (Fin m → S) => ν _) (map_certBlock D Dsf j m)),
    ← measureReal_def]
  exact pi_not_injective_le (D j) m ρ hρ hρ0

lemma map_prefCertPairRun (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (j' j : J) (i i' : ℕ) :
    Measure.map (fun x : Run Ω S J => (prf j' i x, cert j i' x)) (runLaw μ D Dsf)
      = (D j').prod (D j) := by
  rw [show (fun x : Run Ω S J => (prf j' i x, cert j i' x))
      = (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J × ℕ → S)) => (d.1.2 j' i, d.2 (j, i')))
        ∘ Prod.snd from rfl,
    ← Measure.map_map (by fun_prop) measurable_snd, runLaw_eq_prod, Measure.map_snd_prod]
  simp only [measure_univ, one_smul]
  exact map_prefCertPair D Dsf j' j i i'

open scoped Classical in
/-- **The gate's prefixes are fresh.**  A certification draw repeating a table prefix costs
the same `ρ` as a repeat inside one stream, and there are `|populations|·m²` pairs. -/
theorem prefix_cert_disjoint_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (populations : Finset J) (j : J) (m : ℕ) (ρ : ℝ)
    (hρ : ∀ j' ∈ populations, collisionMass (D j') ≤ ρ) (hρj : collisionMass (D j) ≤ ρ)
    (hρ0 : 0 ≤ ρ) :
    (runLaw μ D Dsf).real
        {x : Run Ω S J | ¬ Disjoint (prefixesAt populations m x) (certOf j m x)}
      ≤ (populations.card : ℝ) * (m : ℝ) ^ 2 * ρ := by
  classical
  set κ := populations ×ˢ (Finset.range m ×ˢ Finset.range m) with hκ
  have hsub : {x : Run Ω S J | ¬ Disjoint (prefixesAt populations m x) (certOf j m x)}
      ⊆ ⋃ z ∈ κ, {x : Run Ω S J | prf z.1 z.2.1 x = cert j z.2.2 x} := by
    intro x hx
    obtain ⟨a, ha, ha'⟩ := Finset.not_disjoint_iff.1 hx
    obtain ⟨j', hj', hja⟩ := Finset.mem_biUnion.1 ha
    obtain ⟨i, hi, hia⟩ := Finset.mem_image.1 hja
    obtain ⟨i', hi', hia'⟩ := Finset.mem_image.1 ha'
    exact Set.mem_biUnion (show (j', i, i') ∈ κ by simp [hκ, hj', Finset.mem_range.1 hi,
      Finset.mem_range.1 hi']) (by simpa using hia.trans hia'.symm)
  have hone : ∀ z : J × ℕ × ℕ, z ∈ κ →
      (runLaw μ D Dsf).real {x : Run Ω S J | prf z.1 z.2.1 x = cert j z.2.2 x} ≤ ρ := by
    intro z hz
    have hj' : z.1 ∈ populations := (Finset.mem_product.1 hz).1
    have hdiag : MeasurableSet {q : S × S | q.1 = q.2} :=
      measurableSet_eq_fun measurable_fst measurable_snd
    have hpre : {x : Run Ω S J | prf z.1 z.2.1 x = cert j z.2.2 x}
        = (fun x : Run Ω S J => (prf z.1 z.2.1 x, cert j z.2.2 x)) ⁻¹' {q : S × S | q.1 = q.2} :=
      rfl
    rw [hpre, measureReal_def,
      Measure.map_apply ((measurable_prf z.1 z.2.1).prodMk (measurable_cert j z.2.2)) hdiag
        |>.symm.trans (congrArg (fun ν : Measure (S × S) => ν _)
          (map_prefCertPairRun D Dsf z.1 j z.2.1 z.2.2)),
      ← measureReal_def]
    exact cross_collision_le (D z.1) (D j) ρ (hρ z.1 hj') hρj
      (summable_singleton_sq (D z.1)) (summable_singleton_sq (D j))
  calc (runLaw μ D Dsf).real
        {x : Run Ω S J | ¬ Disjoint (prefixesAt populations m x) (certOf j m x)}
      ≤ (runLaw μ D Dsf).real (⋃ z ∈ κ, {x : Run Ω S J | prf z.1 z.2.1 x = cert j z.2.2 x}) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ z ∈ κ, (runLaw μ D Dsf).real {x : Run Ω S J | prf z.1 z.2.1 x = cert j z.2.2 x} :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _z ∈ κ, ρ := Finset.sum_le_sum hone
    _ = (κ.card : ℝ) * ρ := by rw [Finset.sum_const, nsmul_eq_mul]
    _ = (populations.card : ℝ) * (m : ℝ) ^ 2 * ρ := by
        have hcard : κ.card = populations.card * (m * m) := by
          rw [hκ, Finset.card_product, Finset.card_product, Finset.card_range]
        rw [hcard]; push_cast; ring

/-- The certification prefixes are read by neither the clustering nor the split. -/
lemma disjoint_gateReads {Pre : Set S} (hflat : Flat Pre) (P C A : Finset S)
    (hP : ∀ q ∈ P, q ∈ Pre) (hA : ∀ p ∈ A, p ∈ Pre) (hPA : Disjoint P A) :
    Disjoint (↑A : Set S) (↑(readSet P C ∪ readSet A (C.erase 1)) : Set S) := by
  classical
  rw [Finset.coe_union, Set.disjoint_union_right]
  exact ⟨Finset.disjoint_coe.2 (disjoint_readSet hflat hP hA hPA),
    Finset.disjoint_coe.2 (disjoint_readSet_erase hflat hA)⟩

open scoped Classical in
/-- **A wrong cut is a wrong *fraction* of one gate side.**  Either side may be small, or
clean on its own; what cannot happen is both at once, once the two together carry
`(c + 2β)·|C|` wrong prefixes.  `β` is the size floor the gate's tail bound is charged at,
and paying for it twice — once per side — is what fixes the constants. -/
lemma exists_wrong_side (O : Oracle μ S) (C A R : Finset S) (hA : A ⊆ C) (hR : R ⊆ C)
    (hAR : Disjoint A R) (wtot c β : ℝ) (hc : 0 ≤ c) (hβ : 0 ≤ β)
    (hcount : c * (C.card : ℝ) + 2 * (β * (C.card : ℝ)) < wtot)
    (hle : wtot ≤ ((A.filter (fun p => O.label p = 0)).card : ℝ)
        + ((R.filter (fun p => ¬ (O.label p = 0))).card : ℝ)) :
    (β * (C.card : ℝ) ≤ (A.card : ℝ)
        ∧ c * (A.card : ℝ) ≤ ((A.filter (fun p => O.label p = 0)).card : ℝ))
      ∨ (β * (C.card : ℝ) ≤ (R.card : ℝ)
        ∧ c * (R.card : ℝ) ≤ ∑ p ∈ R, O.label p) := by
  classical
  have hRsum : ∑ p ∈ R, O.label p
      = ((R.filter (fun p => ¬ (O.label p = 0))).card : ℝ) := by
    rw [sum_label_eq O R]
    have hcf := Finset.card_filter_add_card_filter_not (s := R) (fun p => O.label p = 0)
    have : ((R.filter (fun p => O.label p = 0)).card : ℝ)
        + ((R.filter (fun p => ¬ (O.label p = 0))).card : ℝ) = (R.card : ℝ) := by
      exact_mod_cast congrArg (fun n : ℕ => (n : ℝ)) hcf
    linarith
  rw [hRsum]
  by_contra hcon
  push_neg at hcon
  obtain ⟨hAfail, hRfail⟩ := hcon
  have hcard : (A.card : ℝ) + (R.card : ℝ) ≤ (C.card : ℝ) := by
    have hu := Finset.card_le_card (Finset.union_subset hA hR)
    rw [Finset.card_union_of_disjoint hAR] at hu
    exact_mod_cast hu
  have hbound : ∀ (U : Finset S) (Q : S → Prop) [DecidablePred Q],
      (β * (C.card : ℝ) ≤ (U.card : ℝ) → ((U.filter Q).card : ℝ) < c * (U.card : ℝ)) →
      ((U.filter Q).card : ℝ) < c * (U.card : ℝ) + β * (C.card : ℝ) := by
    intro U Q _ hfail
    by_cases hbig : β * (C.card : ℝ) ≤ (U.card : ℝ)
    · have hcU : (0 : ℝ) ≤ c * (U.card : ℝ) := mul_nonneg hc (Nat.cast_nonneg _)
      have hβC : (0 : ℝ) ≤ β * (C.card : ℝ) := mul_nonneg hβ (Nat.cast_nonneg _)
      linarith [hfail hbig]
    · have hfil : ((U.filter Q).card : ℝ) ≤ (U.card : ℝ) := by
        exact_mod_cast Finset.card_filter_le U Q
      have hcU : (0 : ℝ) ≤ c * (U.card : ℝ) := mul_nonneg hc (Nat.cast_nonneg _)
      push_neg at hbig
      linarith
  have hAb := hbound A (fun p => O.label p = 0) hAfail
  have hRb := hbound R (fun p => ¬ (O.label p = 0)) hRfail
  have hcc : (0 : ℝ) ≤ c := hc
  nlinarith [hAb, hRb, hcard, hle, hcount]

/-! ### The population bound for one state

Three things can go wrong at a population prefix: the clustering read its query strings
(`p ∈ P`), the family flips too much of it, or the vote misses despite few flips.  The first
is charged to the prefix mass of the table, the second to `flipCount_mass_le`, and only the
third to the tails. -/

lemma measurableSet_of_fam {fam : Ω → Finset S} {T : Finset (Finset S)}
    (hfam : ∀ ω, fam ω ∈ T) (hfamMeas : ∀ A₀, MeasurableSet {ω | fam ω = A₀})
    (R : Finset S → Set Ω) (hR : ∀ A₀, MeasurableSet (R A₀)) :
    MeasurableSet {ω | ω ∈ R (fam ω)} := by
  classical
  have hcover : {ω | ω ∈ R (fam ω)} = ⋃ A₀ ∈ T, ({ω | fam ω = A₀} ∩ R A₀) := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Finset.mem_coe, exists_prop]
    refine ⟨fun h => ⟨fam ω, hfam ω, rfl, h⟩, ?_⟩
    rintro ⟨A₀, -, he, hr⟩
    exact he ▸ hr
  rw [hcover]
  exact Finset.measurableSet_biUnion _ (fun A₀ _ => (hfamMeas A₀).inter (hR A₀))

open scoped Classical in
/-- The families a prefix is light for: those the vote's tails cover. -/
noncomputable def lightFams (O : Oracle μ S) (T : Finset (Finset S)) (f : ℝ) (p : S) :
    Finset (Finset S) :=
  T.filter (fun t => flipCount O t p ≤ (t.card : ℝ) * f)

open scoped Classical in
/-- The vote's own failures: at a prefix the clustering never read, for a family it is light
for. -/
noncomputable def lightBad (Pre : Set S) (O : Oracle μ S) (P : Finset S) (lo hi : ℕ)
    (T : Finset (Finset S)) (f : ℝ) (fam : Ω → Finset S) (p : S) : Set Ω :=
  if p ∈ Pre ∧ p ∉ P then
    {ω | fam ω ∈ lightFams O T f p ∧ ¬ cutCorrect O lo hi (fam ω) p ω}
  else ∅

/-- **Every prefix's own failure is exponentially unlikely.** -/
theorem measureReal_lightBad_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (P cands : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (lo hi : ℕ)
    (T : Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T) (hTC : ∀ t ∈ T, t ⊆ cands)
    (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (f γ : ℝ) (hγ : 0 ≤ γ) (kmin : ℕ) (hk : ∀ t ∈ T, kmin ≤ t.card)
    (hhi : ∀ t ∈ T, (t.card : ℝ) * ((O.η + (1 - 2 * O.η) * f) + γ) ≤ (hi : ℝ))
    (hlo : ∀ t ∈ T, (lo : ℝ) < (t.card : ℝ) * ((O.η + (1 - 2 * O.η) * (1 - f)) - γ))
    (p : S) :
    μ.real (lightBad Pre O P lo hi T f fam p) ≤ Real.exp (-2 * (kmin : ℝ) * γ ^ 2) := by
  classical
  unfold lightBad
  split_ifs with hcase
  · refine cutCorrect_selected_whp O cands (readSet P cands) p lo hi
      (disjoint_image_readSet hflat hP hcase.1 hcase.2) T (lightFams O T f p) t₀ ht₀ hTC
      fam hfam hcongr _ (Real.exp_nonneg _) ?_
    intro A₀ hA₀ hgood
    have hlight : flipCount O A₀ p ≤ (A₀.card : ℝ) * f := (Finset.mem_filter.1 hgood).2
    refine le_trans (cutCorrect_whp O A₀ p lo hi f γ hlight hγ (hhi A₀ hA₀) (hlo A₀ hA₀)) ?_
    have hkc : (kmin : ℝ) ≤ (A₀.card : ℝ) := by exact_mod_cast hk A₀ hA₀
    exact Real.exp_le_exp.2 (by nlinarith [sq_nonneg γ])
  · simp [Real.exp_nonneg]

lemma measurableSet_cutCorrect (O : Oracle μ S) (lo hi : ℕ) (A₀ : Finset S) (p : S) :
    MeasurableSet {ω | ¬ cutCorrect O lo hi A₀ p ω} :=
  noiseAlg_le O Set.univ _ (measurableSet_filter_pred_map O (T := Set.univ) (fun v => p * v)
    (by simp) (fun U => ¬ ((hi < Finset.card U → O.label p = 1)
      ∧ (Finset.card U ≤ lo → O.label p = 0))))

lemma measurableSet_decided (O : Oracle μ S) (lo ha : ℕ) (A₀ : Finset S) (p : S) :
    MeasurableSet {ω | ¬ decided O lo ha A₀ p ω} :=
  noiseAlg_le O Set.univ _ (measurableSet_filter_pred_map O (T := Set.univ) (fun v => p * v)
    (by simp) (fun U => ¬ (ha < Finset.card U ∨ Finset.card U ≤ lo)))

open scoped Classical in
/-- The indecision event at one prefix, for the family the run produces. -/
lemma measurableSet_indecisive {fam : Ω → Finset S} {T : Finset (Finset S)}
    (hfam : ∀ ω, fam ω ∈ T) (hfamMeas : ∀ A₀, MeasurableSet {ω | fam ω = A₀})
    (O : Oracle μ S) (lo ha : ℕ) (good : Finset (Finset S)) (p : S) :
    MeasurableSet {ω | fam ω ∈ good ∧ ¬ decided O lo ha (fam ω) p ω} := by
  classical
  refine measurableSet_of_fam hfam hfamMeas
    (fun A₀ => {ω | A₀ ∈ good ∧ ¬ decided O lo ha A₀ p ω}) (fun A₀ => ?_)
  by_cases hg : A₀ ∈ good
  · simpa [hg] using measurableSet_decided O lo ha A₀ p
  · simpa [hg] using MeasurableSet.empty

open scoped Classical in
/-- The mis-cut event at one prefix, for the family the run produces. -/
lemma measurableSet_miscut {fam : Ω → Finset S} {T : Finset (Finset S)}
    (hfam : ∀ ω, fam ω ∈ T) (hfamMeas : ∀ A₀, MeasurableSet {ω | fam ω = A₀})
    (O : Oracle μ S) (lo ha : ℕ) (good : Finset (Finset S)) (p : S) :
    MeasurableSet {ω | fam ω ∈ good ∧ ¬ cutCorrect O lo ha (fam ω) p ω} := by
  classical
  refine measurableSet_of_fam hfam hfamMeas
    (fun A₀ => {ω | A₀ ∈ good ∧ ¬ cutCorrect O lo ha A₀ p ω}) (fun A₀ => ?_)
  by_cases hg : A₀ ∈ good
  · simpa [hg] using measurableSet_cutCorrect O lo ha A₀ p
  · simpa [hg] using MeasurableSet.empty

open scoped Classical in
/-- **The indecision rate on the certification sample is below the limit**, off an `E / l`
set.  `count_frac_le` fed by the per-prefix bound, with no union over the sample. -/
theorem indecision_frac_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hCPre : ∀ p ∈ C, p ∈ Pre)
    (hPC : Disjoint P C) (lo ha : ℕ)
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hfamMeas : ∀ A₀, MeasurableSet {ω | fam ω = A₀})
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E l : ℝ) (hE : 0 ≤ E) (hl : 0 < l) (hCpos : 0 < C.card)
    (hbad : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p → μ.real {ω | ¬ decided O lo ha A₀ p ω} ≤ E) :
    μ.real {ω | l * (C.card : ℝ)
        < ((C.filter (fun p => fam ω ∈ good p ∧ ¬ decided O lo ha (fam ω) p ω)).card : ℝ)}
      ≤ E / l := by
  classical
  have hrw : {ω | l * (C.card : ℝ)
      < ((C.filter (fun p => fam ω ∈ good p ∧ ¬ decided O lo ha (fam ω) p ω)).card : ℝ)}
      = (Set.univ : Set Ω) ∩ {ω | l * (C.card : ℝ)
        < ((C.filter (fun p => fam ω ∈ good p ∧ ¬ decided O lo ha (fam ω) p ω)).card : ℝ)} := by
    rw [Set.univ_inter]
  rw [hrw]
  refine count_frac_le (μ := μ) C (fun p => {ω | fam ω ∈ good p ∧ ¬ decided O lo ha (fam ω) p ω})
    (fun ω p => fam ω ∈ good p ∧ ¬ decided O lo ha (fam ω) p ω) Set.univ
    (fun p => measurableSet_indecisive hfam hfamMeas O lo ha (good p) p) E l hE hl hCpos ?_ ?_
  · intro p hp
    have hpP : p ∉ P := Finset.disjoint_right.1 hPC hp
    exact decided_selected_whp O cands (readSet P cands) p lo ha
      (disjoint_image_readSet hflat hP (hCPre p hp) hpP) T (good p) t₀ ht₀ hTC fam hfam
      hcongr E hE (fun A₀ hA₀ hg => hbad p hp A₀ hA₀ hg)
  · intro ω _ p _ hpr
    exact hpr

open scoped Classical in
/-- The same for the rate at which the cut is wrong. -/
theorem miscut_frac_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hCPre : ∀ p ∈ C, p ∈ Pre)
    (hPC : Disjoint P C) (lo ha : ℕ)
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hfamMeas : ∀ A₀, MeasurableSet {ω | fam ω = A₀})
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E l : ℝ) (hE : 0 ≤ E) (hl : 0 < l) (hCpos : 0 < C.card)
    (hbad : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p → μ.real {ω | ¬ cutCorrect O lo ha A₀ p ω} ≤ E) :
    μ.real {ω | l * (C.card : ℝ)
        < ((C.filter (fun p => fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω)).card : ℝ)}
      ≤ E / l := by
  classical
  have hrw : {ω | l * (C.card : ℝ)
      < ((C.filter (fun p => fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω)).card : ℝ)}
      = (Set.univ : Set Ω) ∩ {ω | l * (C.card : ℝ)
        < ((C.filter (fun p => fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω)).card : ℝ)} := by
    rw [Set.univ_inter]
  rw [hrw]
  refine count_frac_le (μ := μ) C (fun p => {ω | fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω})
    (fun ω p => fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω) Set.univ
    (fun p => measurableSet_miscut hfam hfamMeas O lo ha (good p) p) E l hE hl hCpos ?_ ?_
  · intro p hp
    have hpP : p ∉ P := Finset.disjoint_right.1 hPC hp
    exact cutCorrect_selected_whp O cands (readSet P cands) p lo ha
      (disjoint_image_readSet hflat hP (hCPre p hp) hpP) T (good p) t₀ ht₀ hTC fam hfam
      hcongr E hE (fun A₀ hA₀ hg => hbad p hp A₀ hA₀ hg)
  · intro ω _ p _ hpr
    exact hpr

open scoped Classical in
/-- **At a fixed table, the state returns.**  Each failure count splits into the prefixes
the family is light for — bounded by `count_frac_le` — and the heavy ones, whose number is
itself a fraction.  Lightness at a *fresh* prefix only holds for most of them, never all, so
carrying it as a per-prefix condition rather than a global guard is what makes the argument
available at all.

The certification sample's class counts are what the sides are measured against — an
all-accepting population leaves the reject side empty and nothing can admit — so they stay
as hypotheses about the draws rather than becoming a knob. -/
theorem ret_at_whp {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hCPre : ∀ p ∈ C, p ∈ Pre)
    (hPC : Disjoint P C) (Q : Finset S) (hQsup : readSet P cands ⊆ Q)
    (hdisjQ : Disjoint (↑C : Set S) (↑Q : Set S))
    (lo hi : ℕ) (εcov α τ l : ℝ) (n₀ : ℕ)
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hfamMeas : ∀ A₀, MeasurableSet {ω | fam ω = A₀})
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (hQ : ∀ ω, ∀ p ∈ C, ∀ v ∈ fam ω, p * v ∈ Q)
    (E : ℝ) (hE : 0 ≤ E) (hl : 0 < l) (hCpos : 0 < C.card)
    (hτ : 0 ≤ τ) (hε0 : 0 ≤ εcov) (hε1 : εcov ≤ 1) (hsig : O.η ≤ 1 / 2)
    (hdec : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p →
      μ.real {ω | ¬ decided O lo (hi - 1) A₀ p ω} ≤ E)
    (hcut : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p →
      μ.real {ω | ¬ cutCorrect O lo (hi - 1) A₀ p ω} ≤ E)
    (hga : ∀ n : ℕ, n₀ ≤ n → n ≤ C.card →
      (n : ℝ) * (gateAcc O εcov + τ + τ)
        ≤ (n : ℝ) * (1 - O.η) - (1 - 2 * O.η) * (2 * l * (C.card : ℝ)))
    (hgr : ∀ n : ℕ, n₀ ≤ n → n ≤ C.card →
      (n : ℝ) * O.η + (1 - 2 * O.η) * (2 * l * (C.card : ℝ))
        ≤ (n : ℝ) * (gateRej O εcov - τ - τ))
    (hα : Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) ≤ α)
    (hclassA : (n₀ : ℝ) + 2 * l * (C.card : ℝ) + 2 * l * (C.card : ℝ)
      ≤ ((C.filter (fun p => O.label p = 1)).card : ℝ))
    (hclassR : (n₀ : ℝ) + 2 * l * (C.card : ℝ) + 2 * l * (C.card : ℝ)
      ≤ ((C.filter (fun p => O.label p = 0)).card : ℝ)) :
    μ.real {ω | ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) ≤ l * (C.card : ℝ)
        ∧ ¬ ((((C.filter (fun p => ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ)
                ≤ 2 * l * (C.card : ℝ))
            ∧ admitted O lo hi εcov α (fam ω) C ω)}
      ≤ E / l + (E / l + 2 * Real.exp (-2 * (n₀ : ℝ) * τ ^ 2)) := by
  classical
  have hsplit : ∀ (ω : Ω) (Q : S → Prop) [DecidablePred Q],
      (C.filter (fun p => Q p)).card
        ≤ (C.filter (fun p => fam ω ∈ good p ∧ Q p)).card
          + (C.filter (fun p => fam ω ∉ good p)).card := by
    intro ω Q _
    refine le_trans (Finset.card_le_card ?_) (Finset.card_union_le _ _)
    intro p hp
    obtain ⟨hpC, hq⟩ := Finset.mem_filter.1 hp
    by_cases hg : fam ω ∈ good p
    · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpC, hg, hq⟩)
    · exact Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hpC, hg⟩)
  have hsub : {ω | ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) ≤ l * (C.card : ℝ)
      ∧ ¬ ((((C.filter (fun p => ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ)
              ≤ 2 * l * (C.card : ℝ))
          ∧ admitted O lo hi εcov α (fam ω) C ω)}
      ⊆ {ω | l * (C.card : ℝ)
            < ((C.filter (fun p => fam ω ∈ good p
                ∧ ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ)}
        ∪ ({ω | l * (C.card : ℝ)
              < ((C.filter (fun p => fam ω ∈ good p
                  ∧ ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ)}
          ∪ {ω | (((C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ)
                ≤ 2 * l * (C.card : ℝ))
              ∧ n₀ ≤ (splitAcc O hi (fam ω) C ω).2 ∧ n₀ ≤ (splitRej O lo (fam ω) C ω).2
              ∧ ¬ admitted O lo hi εcov α (fam ω) C ω}) := by
    rintro ω ⟨hheavy, hbad⟩
    by_cases hindL : ((C.filter (fun p => fam ω ∈ good p
        ∧ ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ) ≤ l * (C.card : ℝ)
    · by_cases hmisL : ((C.filter (fun p => fam ω ∈ good p
          ∧ ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ) ≤ l * (C.card : ℝ)
      · have hind : (((C.filter (fun p => ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ))
            ≤ 2 * l * (C.card : ℝ) := by
          have := hsplit ω (fun p => ¬ decided O lo (hi - 1) (fam ω) p ω)
          have hc : (((C.filter (fun p => ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ))
              ≤ ((C.filter (fun p => fam ω ∈ good p
                  ∧ ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ)
                + ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) := by exact_mod_cast this
          linarith
        have hmis : (((C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ))
            ≤ 2 * l * (C.card : ℝ) := by
          have := hsplit ω (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)
          have hc : (((C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ))
              ≤ ((C.filter (fun p => fam ω ∈ good p
                  ∧ ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ)
                + ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) := by exact_mod_cast this
          linarith
        have hadm : ¬ admitted O lo hi εcov α (fam ω) C ω := fun h => hbad ⟨hind, h⟩
        refine Or.inr (Or.inr ⟨hmis, ?_, ?_, hadm⟩)
        · have hcard := card_le_sideAcc_add O lo (hi - 1) (fam ω) C ω
          have hcastR : ((C.filter (fun p => O.label p = 1)).card : ℝ)
              ≤ ((C.filter (fun p => hi - 1 < voteCount O (fam ω) p ω)).card : ℝ)
                + (((C.filter (fun p => ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ)
                  + ((C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ)) := by
            exact_mod_cast hcard
          have : (n₀ : ℝ)
              ≤ ((C.filter (fun p => hi - 1 < voteCount O (fam ω) p ω)).card : ℝ) := by linarith
          exact_mod_cast this
        · have hcard := card_le_sideRej_add O lo (hi - 1) (fam ω) C ω
          have hcastR : ((C.filter (fun p => O.label p = 0)).card : ℝ)
              ≤ ((C.filter (fun p => voteCount O (fam ω) p ω ≤ lo)).card : ℝ)
                + (((C.filter (fun p => ¬ decided O lo (hi - 1) (fam ω) p ω)).card : ℝ)
                  + ((C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ)) := by
            exact_mod_cast hcard
          have : (n₀ : ℝ)
              ≤ ((C.filter (fun p => voteCount O (fam ω) p ω ≤ lo)).card : ℝ) := by linarith
          exact_mod_cast this
      · exact Or.inr (Or.inl (not_le.1 hmisL))
    · exact Or.inl (not_le.1 hindL)
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) (add_le_add ?_ ?_)
  · exact indecision_frac_le hflat O P cands C hP hCPre hPC lo (hi - 1) T good t₀ ht₀ hTC
      fam hfam hfamMeas hcongr E l hE hl hCpos hdec
  · refine le_trans (measureReal_union_le _ _) (add_le_add ?_ ?_)
    · exact miscut_frac_le hflat O P cands C hP hCPre hPC lo (hi - 1) T good t₀ ht₀ hTC
        fam hfam hfamMeas hcongr E l hE hl hCpos hcut
    · exact admitted_whp O C Q hdisjQ lo hi εcov α τ (2 * l * (C.card : ℝ)) n₀
        fam hQ (fun ω ω' h => hcongr ω ω' (fun w hw => h w (hQsup hw)))
        hτ hε0 hε1 hsig hga hgr hα

lemma measurableSet_lightBad (Pre : Set S) (O : Oracle μ S) (P : Finset S) (lo hi : ℕ)
    {T : Finset (Finset S)} (f : ℝ) {fam : Ω → Finset S} (hfam : ∀ ω, fam ω ∈ T)
    (hfamMeas : ∀ A₀, MeasurableSet {ω | fam ω = A₀}) (p : S) :
    MeasurableSet (lightBad Pre O P lo hi T f fam p) := by
  classical
  unfold lightBad
  split_ifs
  · refine measurableSet_of_fam hfam hfamMeas
      (fun A₀ => {ω | A₀ ∈ lightFams O T f p ∧ ¬ cutCorrect O lo hi A₀ p ω}) (fun A₀ => ?_)
    by_cases hg : A₀ ∈ lightFams O T f p
    · simpa [hg] using measurableSet_cutCorrect O lo hi A₀ p
    · simpa [hg] using MeasurableSet.empty
  · exact MeasurableSet.empty

/-- **One population's coverage.**  Off the table's own prefixes and off the prefixes the
family flips too much of, the cut fails only through the vote's tails, and those are
exponentially unlikely at each prefix.  The three slacks are what the caller has to buy:
`Dj Preᶜ` is the population's mass outside the sampler's reach, `Dj P` the mass the
clustering already read, and `Δ / f` the Markov price of the per-member flip bound. -/
theorem coverage_of_run {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (Dj : Measure S) [IsProbabilityMeasure Dj]
    (P cands : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (lo hi : ℕ)
    (T : Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T) (hTC : ∀ t ∈ T, t ⊆ cands)
    (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hfamMeas : ∀ A₀, MeasurableSet {ω | fam ω = A₀})
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (f γ Δ ε : ℝ) (hγ : 0 ≤ γ) (hf : 0 < f) (hε : 0 < ε)
    (kmin : ℕ) (hkpos : 0 < kmin) (hk : ∀ t ∈ T, kmin ≤ t.card)
    (hhi : ∀ t ∈ T, (t.card : ℝ) * ((O.η + (1 - 2 * O.η) * f) + γ) ≤ (hi : ℝ))
    (hlo : ∀ t ∈ T, (lo : ℝ) < (t.card : ℝ) * ((O.η + (1 - 2 * O.η) * (1 - f)) - γ)) :
    μ.real {ω | (∀ v ∈ fam ω, flipMass O Dj v ≤ Δ)
        ∧ Dj.real Preᶜ + Dj.real ↑P + Δ / f + ε
            ≤ Dj.real {p | ¬ cutCorrect O lo hi (fam ω) p ω}}
      ≤ Real.exp (-2 * (kmin : ℝ) * γ ^ 2) / ε := by
  classical
  refine le_trans (measureReal_mono ?_ (measure_ne_top _ _))
    (measureReal_badMass_ge_le (μ := μ) Dj (lightBad Pre O P lo hi T f fam)
      (measurableSet_lightBad Pre O P lo hi f hfam hfamMeas)
      (Real.exp (-2 * (kmin : ℝ) * γ ^ 2)) ε (Real.exp_nonneg _) hε
      (measureReal_lightBad_le hflat O P cands hP lo hi T t₀ ht₀ hTC fam hfam hcongr f γ hγ
        kmin hk hhi hlo))
  rintro ω ⟨hgood, hshort⟩
  have hcardpos : 0 < (fam ω).card := lt_of_lt_of_le hkpos (hk _ (hfam ω))
  have hheavy : Dj.real {p | ¬ (fam ω ∈ lightFams O T f p)} ≤ Δ / f := by
    refine le_trans (measureReal_mono ?_ (measure_ne_top _ _))
      (flipCount_mass_le O Dj (fam ω) Δ f hf hcardpos hgood)
    intro q hq
    have : ¬ (flipCount O (fam ω) q ≤ ((fam ω).card : ℝ) * f) := by
      intro hle
      exact hq (Finset.mem_filter.2 ⟨hfam ω, hle⟩)
    show f * ((fam ω).card : ℝ) ≤ flipCount O (fam ω) q
    rw [mul_comm]
    exact le_of_lt (not_le.1 this)
  have hcover : {p | ¬ cutCorrect O lo hi (fam ω) p ω}
      ⊆ (Preᶜ ∪ ↑P) ∪ ({p | ¬ (fam ω ∈ lightFams O T f p)}
        ∪ {p | ω ∈ lightBad Pre O P lo hi T f fam p}) := by
    intro q hq
    by_cases hqP : q ∈ Pre ∧ q ∉ P
    · by_cases hg : fam ω ∈ lightFams O T f q
      · refine Or.inr (Or.inr ?_)
        simp only [Set.mem_setOf_eq, lightBad, if_pos hqP]
        exact ⟨hg, hq⟩
      · exact Or.inr (Or.inl hg)
    · rcases not_and_or.1 hqP with h | h
      · exact Or.inl (Or.inl h)
      · exact Or.inl (Or.inr (by simpa using h))
  have hsub : Dj.real {p | ¬ cutCorrect O lo hi (fam ω) p ω}
      ≤ (Dj.real Preᶜ + Dj.real ↑P)
        + (Dj.real {p | ¬ (fam ω ∈ lightFams O T f p)}
          + Dj.real {p | ω ∈ lightBad Pre O P lo hi T f fam p}) :=
    le_trans (measureReal_mono hcover (measure_ne_top _ _))
      (le_trans (measureReal_union_le _ _)
        (add_le_add (measureReal_union_le _ _) (measureReal_union_le _ _)))
  show ε ≤ Dj.real {p | ω ∈ lightBad Pre O P lo hi T f fam p}
  linarith [hsub, hshort, hheavy]

open scoped Classical in
/-- The family once the screen's verdict is fixed. -/
noncomputable def famCore (O : Oracle μ S) (cn cd : ℕ) (P C cands : Finset S) (k : ℕ)
    (ω : Ω) : Finset S :=
  if clusterAround O cn cd P C ω k = {(1 : S)}
  then leastLossSubset (fun _ : S => (0 : ℝ)) cands k
  else clusterAround O cn cd P C ω k

open scoped Classical in
lemma measurableSet_famCore (O : Oracle μ S) (cn cd : ℕ) (P C cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ C) (A₀ : Finset S) :
    MeasurableSet {ω | famCore O cn cd P C cands k ω = A₀} := by
  classical
  have hstall := measurableSet_clusterAround O cn cd P C k hone ({(1 : S)} : Finset S)
  have htarget := measurableSet_clusterAround O cn cd P C k hone A₀
  set Stall : Set Ω := {ω | clusterAround O cn cd P C ω k = ({(1 : S)} : Finset S)} with hStall
  set Hit : Set Ω := {ω | clusterAround O cn cd P C ω k = A₀} with hHit
  have hcov : {ω | famCore O cn cd P C cands k ω = A₀}
      = (Stall ∩ {_ω : Ω | leastLossSubset (fun _ : S => (0 : ℝ)) cands k = A₀})
        ∪ (Stallᶜ ∩ Hit) := by
    ext ω
    constructor
    · intro hω
      have hf : famCore O cn cd P C cands k ω = A₀ := hω
      by_cases hs : ω ∈ Stall
      · refine Or.inl ⟨hs, ?_⟩
        rw [famCore, if_pos (show clusterAround O cn cd P C ω k = ({(1 : S)} : Finset S) from hs)]
          at hf
        exact hf
      · refine Or.inr ⟨hs, ?_⟩
        rw [famCore, if_neg (show ¬ clusterAround O cn cd P C ω k = ({(1 : S)} : Finset S)
          from hs)] at hf
        exact hf
    · rintro (⟨hs, ht⟩ | ⟨hs, hh⟩)
      · show famCore O cn cd P C cands k ω = A₀
        rw [famCore, if_pos (show clusterAround O cn cd P C ω k = ({(1 : S)} : Finset S) from hs)]
        exact ht
      · show famCore O cn cd P C cands k ω = A₀
        rw [famCore, if_neg (show ¬ clusterAround O cn cd P C ω k = ({(1 : S)} : Finset S)
          from hs)]
        exact hh
  rw [hcov]
  refine MeasurableSet.union (hstall.inter ?_) (hstall.compl.inter htarget)
  by_cases ht : leastLossSubset (fun _ : S => (0 : ℝ)) cands k = A₀
  · simpa only [ht, Set.setOf_true] using MeasurableSet.univ
  · simpa only [ht, Set.setOf_false] using MeasurableSet.empty

open scoped Classical in
/-- `famAt` as a function of the noise alone, with the table and the *drawn* pool fixed —
which is what they are once the draws are.  The screen is applied inside, since it reads the
oracle; the seed-only stall is replaced by a fixed `k`-subset so that the value set is
`powersetCard k` of the drawn pool. -/
noncomputable def famOf (O : Oracle μ S) (cn cd sc : ℕ) (P cands : Finset S) (k : ℕ) (ω : Ω) :
    Finset S :=
  famCore O cn cd P (screened O sc P cands ω) cands k ω

lemma famOf_eq_famCore (O : Oracle μ S) (cn cd sc : ℕ) (P cands : Finset S) (k : ℕ) (ω : Ω) :
    famOf O cn cd sc P cands k ω
      = famCore O cn cd P (screened O sc P cands ω) cands k ω := rfl

open scoped Classical in
lemma famOf_mem (O : Oracle μ S) (cn cd sc : ℕ) (P cands : Finset S) (k : ℕ) (ω : Ω)
    (hone : (1 : S) ∈ cands) (hk : k ≤ cands.card) :
    famOf O cn cd sc P cands k ω ∈ cands.powersetCard k := by
  classical
  unfold famOf famCore
  split_ifs with h
  · exact leastLossSubset_mem _ _ _ hk
  · rcases Finset.mem_insert.1
      (clusterAround_mem_values O cn cd P (screened O sc P cands ω) ω k
        (one_mem_screened O sc P cands ω hone)) with h1 | h1
    · exact absurd h1 h
    · exact Finset.powersetCard_mono (screened_subset O sc P cands ω) h1

open scoped Classical in
lemma famOf_congr_mq (O : Oracle μ S) (cn cd sc : ℕ) (P cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands) {ω ω' : Ω}
    (hbit : ∀ w ∈ readSet P cands, (mq O w ω = 1 ↔ mq O w ω' = 1)) :
    famOf O cn cd sc P cands k ω = famOf O cn cd sc P cands k ω' := by
  classical
  have hscr : screened O sc P cands ω = screened O sc P cands ω' :=
    Finset.filter_congr (fun v hv => by rw [screenCount_congr O hone hv hbit])
  have hsub : ∀ w ∈ readSet P (screened O sc P cands ω'), (mq O w ω = 1 ↔ mq O w ω' = 1) := by
    intro w hw
    obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hw
    obtain ⟨hp, hvs⟩ := Finset.mem_product.1 hpv
    exact hbit _ (mem_readSet hp (screened_subset O sc P cands ω' hvs))
  unfold famOf famCore
  rw [hscr, clusterAround_congr_mq O cn cd P _ k
    (one_mem_screened O sc P cands ω' hone) hsub]

lemma famOf_congr (O : Oracle μ S) (cn cd sc : ℕ) (P cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands) {ω ω' : Ω}
    (h : ∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') :
    famOf O cn cd sc P cands k ω = famOf O cn cd sc P cands k ω' :=
  famOf_congr_mq O cn cd sc P cands k hone (fun w hw => by rw [mq_congr O (h w hw)])

open scoped Classical in
/-- The screen's verdict on one candidate is decided by the bits. -/
lemma measurableSet_screenCount_le' (O : Oracle μ S) (P : Finset S) (v : S) (sc : ℕ) :
    MeasurableSet[noiseAlg O Set.univ] {ω | screenCount O P v ω ≤ sc} := by
  have hPr : ∀ p ∈ P, MeasurableSet[noiseAlg O Set.univ]
      {ω | ¬ ((mq O (p * v) ω = 1) ↔ (mq O p ω = 1))} := by
    intro p _
    have h1 := measurableSet_mq_eq_one O (T := Set.univ) (w := p * v) (Set.mem_univ _)
    have h0 := measurableSet_mq_eq_one O (T := Set.univ) (w := p) (Set.mem_univ _)
    have hiff : {ω | (mq O (p * v) ω = 1) ↔ (mq O p ω = 1)}
        = ({ω | mq O (p * v) ω = 1} ∩ {ω | mq O p ω = 1})
          ∪ ({ω | mq O (p * v) ω = 1}ᶜ ∩ {ω | mq O p ω = 1}ᶜ) := by
      ext ω
      by_cases ha : mq O (p * v) ω = 1 <;> by_cases hb : mq O p ω = 1 <;> simp [ha, hb]
    have : MeasurableSet[noiseAlg O Set.univ] {ω | (mq O (p * v) ω = 1) ↔ (mq O p ω = 1)} := by
      rw [hiff]
      exact ((h1.inter h0).union (h1.compl.inter h0.compl))
    exact this.compl
  exact measurableSet_filter_pred' O (fun p ω => ¬ ((mq O (p * v) ω = 1) ↔ (mq O p ω = 1)))
      hPr (fun U => U.card ≤ sc)

lemma measurableSet_screenCount_le (O : Oracle μ S) (P : Finset S) (v : S) (sc : ℕ) :
    MeasurableSet {ω | screenCount O P v ω ≤ sc} :=
  noiseAlg_le O Set.univ _ (measurableSet_screenCount_le' O P v sc)

open scoped Classical in
lemma measurableSet_screened (O : Oracle μ S) (sc : ℕ) (P cands : Finset S) (C' : Finset S) :
    MeasurableSet {ω | screened O sc P cands ω = C'} :=
  noiseAlg_le O Set.univ _
    (by
      exact measurableSet_filter_pred' (A := cands) O (fun v ω => screenCount O P v ω ≤ sc)
          (fun v _ => measurableSet_screenCount_le' O P v sc) (fun U => U = C'))

open scoped Classical in
lemma measurableSet_famOf (O : Oracle μ S) (cn cd sc : ℕ) (P cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands) (A₀ : Finset S) :
    MeasurableSet {ω | famOf O cn cd sc P cands k ω = A₀} := by
  classical
  set Pred : Finset S → Prop := fun U => ∃ ω', (readSet P cands).filter
    (fun w => mq O w ω' = 1) = U ∧ famOf O cn cd sc P cands k ω' = A₀ with hPred
  have hcov : {ω | famOf O cn cd sc P cands k ω = A₀}
      = {ω | Pred ((readSet P cands).filter (fun w => mq O w ω = 1))} := by
    ext ω
    simp only [Set.mem_setOf_eq, hPred]
    refine ⟨fun h => ⟨ω, rfl, h⟩, ?_⟩
    rintro ⟨ω', hU, hA⟩
    refine (famOf_congr_mq O cn cd sc P cands k hone (fun w hw => ?_)).trans hA
    have := Finset.ext_iff.1 hU w
    simp only [Finset.mem_filter, hw, true_and] at this
    exact this.symm
  rw [hcov]
  exact noiseAlg_le O Set.univ _
    (measurableSet_filter_pred O (T := Set.univ) (by simp) Pred)

open scoped Classical in
/-- `clusterAt` with the draws fixed: the family is a function of the noise alone. -/
noncomputable def clusterOf (O : Oracle μ S) (cn cd sc : ℕ) (P cands : Finset S) (k : ℕ)
    (ω : Ω) : Finset S :=
  clusterAround O cn cd P (screened O sc P cands ω) ω k

lemma clusterAt_eq_clusterOf (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (x : Run Ω S J) :
    clusterAt O populations x B
      = clusterOf O B.cn B.cd B.sc (prefixesAt populations B.m x) (poolAt B.M x) B.k (nz x) :=
  rfl

lemma clusterOf_subset (O : Oracle μ S) (cn cd sc : ℕ) (P cands : Finset S) (k : ℕ) (ω : Ω)
    (hone : (1 : S) ∈ cands) : clusterOf O cn cd sc P cands k ω ⊆ cands :=
  fun v hv => screened_subset O sc P cands ω
    (clusterAround_subset O cn cd P _ ω k (one_mem_screened O sc P cands ω hone) hv)

lemma clusterOf_congr_mq (O : Oracle μ S) (cn cd sc : ℕ) (P cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands) {ω ω' : Ω}
    (hbit : ∀ w ∈ readSet P cands, (mq O w ω = 1 ↔ mq O w ω' = 1)) :
    clusterOf O cn cd sc P cands k ω = clusterOf O cn cd sc P cands k ω' := by
  classical
  have hscr : screened O sc P cands ω = screened O sc P cands ω' :=
    Finset.filter_congr (fun v hv => by rw [screenCount_congr O hone hv hbit])
  have hsub : ∀ w ∈ readSet P (screened O sc P cands ω'), (mq O w ω = 1 ↔ mq O w ω' = 1) := by
    intro w hw
    obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hw
    obtain ⟨hp, hvs⟩ := Finset.mem_product.1 hpv
    exact hbit _ (mem_readSet hp (screened_subset O sc P cands ω' hvs))
  unfold clusterOf
  rw [hscr, clusterAround_congr_mq O cn cd P _ k (one_mem_screened O sc P cands ω' hone) hsub]

open scoped Classical in
lemma measurableSet_clusterOf (O : Oracle μ S) (cn cd sc : ℕ) (P cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands) (A₀ : Finset S) :
    MeasurableSet {ω | clusterOf O cn cd sc P cands k ω = A₀} := by
  classical
  set Pred : Finset S → Prop := fun U => ∃ ω', (readSet P cands).filter
    (fun w => mq O w ω' = 1) = U ∧ clusterOf O cn cd sc P cands k ω' = A₀ with hPred
  have hcov : {ω | clusterOf O cn cd sc P cands k ω = A₀}
      = {ω | Pred ((readSet P cands).filter (fun w => mq O w ω = 1))} := by
    ext ω
    simp only [Set.mem_setOf_eq, hPred]
    refine ⟨fun h => ⟨ω, rfl, h⟩, ?_⟩
    rintro ⟨ω', hU, hA⟩
    refine (clusterOf_congr_mq O cn cd sc P cands k hone (fun w hw => ?_)).trans hA
    have := Finset.ext_iff.1 hU w
    simp only [Finset.mem_filter, hw, true_and] at this
    exact this.symm
  rw [hcov]
  exact noiseAlg_le O Set.univ _
    (measurableSet_filter_pred O (T := Set.univ) (by simp) Pred)

open scoped Classical in
/-- The family at a state, with the seed-only stall replaced by a fixed `k`-subset.  On
`ret` the two agree (`not_ret_of_seed_family`); off it, this is what keeps the family's
value set inside `powersetCard k`, which is what the selection bound consumes. -/
noncomputable def famAt (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (x : Run Ω S J) : Finset S :=
  famOf O B.cn B.cd B.sc (prefixesAt populations B.m x) (poolAt B.M x) B.k (nz x)

lemma famAt_mem (O : Oracle μ S) (populations : Finset J) (B : Budget) (x : Run Ω S J)
    (hk : B.k ≤ (poolAt B.M x).card) :
    famAt O populations B x ∈ (poolAt B.M x).powersetCard B.k :=
  famOf_mem O B.cn B.cd B.sc _ _ B.k _ (one_mem_poolAt B.M x) hk

lemma famAt_eq_of_ret (O : Oracle μ S) (populations : Finset J)
    (indecisionLimit εcov α : ℝ) (B : Budget) (hα : α < 1)
    (hpop : populations.Nonempty) {x : Run Ω S J}
    (hx : x ∈ ret O populations indecisionLimit εcov α B) :
    famAt O populations B x = clusterAt O populations x B := by
  classical
  show famCore O B.cn B.cd (prefixesAt populations B.m x) (screenedAt O populations B x)
    (poolAt B.M x) B.k (nz x) = _
  have hne : ¬ (clusterAround O B.cn B.cd (prefixesAt populations B.m x)
      (screenedAt O populations B x) (nz x) B.k = ({(1 : S)} : Finset S)) := fun hseed =>
    not_ret_of_seed_family O populations indecisionLimit εcov α B hα hpop x hseed hx
  rw [famCore, if_neg hne]
  rfl

/-- **One state's coverage, with the table and the pool fixed.**  `coverage_of_run` with the
value set the iterate actually lands in. -/
theorem coverage_of_famOf {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (Dj : Measure S) [IsProbabilityMeasure Dj] (P cands : Finset S)
    (hP : ∀ q ∈ P, q ∈ Pre) (hsupp : Dj Preᶜ = 0) (hone : (1 : S) ∈ cands)
    (cn cd sc k : ℕ) (hk : k ≤ cands.card) (hkpos : 0 < k) (lo hi : ℕ) (f γ Δ ε : ℝ)
    (hγ : 0 ≤ γ) (hf : 0 < f) (hε : 0 < ε)
    (hhi : (k : ℝ) * ((O.η + (1 - 2 * O.η) * f) + γ) ≤ (hi : ℝ))
    (hlo : (lo : ℝ) < (k : ℝ) * ((O.η + (1 - 2 * O.η) * (1 - f)) - γ)) :
    μ.real {ω | (∀ v ∈ famOf O cn cd sc P cands k ω, flipMass O Dj v ≤ Δ)
        ∧ Dj.real ↑P + Δ / f + ε
            ≤ Dj.real {p | ¬ cutCorrect O lo hi (famOf O cn cd sc P cands k ω) p ω}}
      ≤ Real.exp (-2 * (k : ℝ) * γ ^ 2) / ε := by
  classical
  have hcardT : ∀ t ∈ cands.powersetCard k, t.card = k :=
    fun t ht => (Finset.mem_powersetCard.1 ht).2
  have hzero : Dj.real Preᶜ = 0 := by rw [measureReal_def, hsupp]; simp
  refine le_trans (le_of_eq ?_)
    (coverage_of_run hflat O Dj P cands hP lo hi (cands.powersetCard k)
      (leastLossSubset (fun _ : S => (0 : ℝ)) cands k) (leastLossSubset_mem _ _ _ hk)
      (fun t ht => (Finset.mem_powersetCard.1 ht).1)
      (famOf O cn cd sc P cands k) (fun ω => famOf_mem O cn cd sc P cands k ω hone hk)
      (measurableSet_famOf O cn cd sc P cands k hone)
      (fun ω ω' h => famOf_congr O cn cd sc P cands k hone h)
      f γ Δ ε hγ hf hε k hkpos (fun t ht => le_of_eq (hcardT t ht).symm)
      (fun t ht => by rw [hcardT t ht]; exact hhi)
      (fun t ht => by rw [hcardT t ht]; exact hlo))
  congr 1
  ext ω
  simp only [Set.mem_setOf_eq, hzero, zero_add]

lemma measurable_badMassReal (O : Oracle μ S) (Dj : Measure S) (lo hi : ℕ) (A₀ : Finset S) :
    Measurable (fun ω => Dj.real {p | ¬ cutCorrect O lo hi A₀ p ω}) :=
  ENNReal.measurable_toReal.comp (measurable_badMass Dj
    (fun p => {ω | ¬ cutCorrect O lo hi A₀ p ω}) (fun p => measurableSet_cutCorrect O lo hi A₀ p))

open scoped Classical in
/-- The runs at one state whose coverage is short although every family member is clean. -/
noncomputable def shortCoverage (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (B : Budget) (Δ f ε : ℝ) : Set (Run Ω S J) :=
  {x | (∀ v ∈ famAt O populations B x, flipMass O Dj v ≤ Δ)
    ∧ Dj.real ↑(prefixesAt populations B.m x) + Δ / f + ε
        ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi (famAt O populations B x) p (nz x)}
    ∧ B.k ≤ (poolAt B.M x).card}

open scoped Classical in
lemma measurableSet_shortCoverage (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (B : Budget) (Δ f ε : ℝ) :
    MeasurableSet (shortCoverage O populations Dj B Δ f ε) := by
  classical
  have hR : ∀ P C : Finset S, MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | (∀ v ∈ famOf O B.cn B.cd B.sc P C B.k (nz x), flipMass O Dj v ≤ Δ)
        ∧ Dj.real ↑P + Δ / f + ε
            ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi (famOf O B.cn B.cd B.sc P C B.k (nz x)) p (nz x)}
        ∧ B.k ≤ C.card} else ∅) := by
    intro P C
    split_ifs with hone
    · have hcov : {x : Run Ω S J | (∀ v ∈ famOf O B.cn B.cd B.sc P C B.k (nz x), flipMass O Dj v ≤ Δ)
          ∧ Dj.real ↑P + Δ / f + ε
              ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi (famOf O B.cn B.cd B.sc P C B.k (nz x)) p (nz x)}
          ∧ B.k ≤ C.card}
          = ⋃ A₀ : Finset S, ({x : Run Ω S J | famOf O B.cn B.cd B.sc P C B.k (nz x) = A₀}
            ∩ {x : Run Ω S J | (∀ v ∈ A₀, flipMass O Dj v ≤ Δ)
              ∧ Dj.real ↑P + Δ / f + ε
                  ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi A₀ p (nz x)}
              ∧ B.k ≤ C.card}) := by
        ext x
        simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff]
        refine ⟨fun h => ⟨_, rfl, h⟩, ?_⟩
        rintro ⟨A₀, hA, hx⟩
        rw [hA]
        exact hx
      rw [hcov]
      refine MeasurableSet.iUnion (fun A₀ => MeasurableSet.inter ?_ ?_)
      · exact measurable_nz (measurableSet_famOf O B.cn B.cd B.sc P C B.k hone A₀)
      · by_cases h1 : ∀ v ∈ A₀, flipMass O Dj v ≤ Δ
        · by_cases h3 : B.k ≤ C.card
          · have hset : {x : Run Ω S J | (∀ v ∈ A₀, flipMass O Dj v ≤ Δ)
                ∧ Dj.real ↑P + Δ / f + ε ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi A₀ p (nz x)}
                ∧ B.k ≤ C.card}
                = nz ⁻¹' {ω : Ω | Dj.real ↑P + Δ / f + ε
                    ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi A₀ p ω}} := by
              ext x
              simp only [Set.mem_setOf_eq, Set.mem_preimage, h3, and_true]
              exact ⟨fun h => h.2, fun h => ⟨h1, h⟩⟩
            rw [hset]
            exact measurable_nz (measurableSet_le measurable_const
              (measurable_badMassReal O Dj B.lo B.hi A₀))
          · simpa [h3] using MeasurableSet.empty
        · simpa [h1] using MeasurableSet.empty
    · exact MeasurableSet.empty
  have hrw : shortCoverage O populations Dj B Δ f ε
      = {x : Run Ω S J | x ∈ (fun P C => if (1 : S) ∈ C then
          {x : Run Ω S J | (∀ v ∈ famOf O B.cn B.cd B.sc P C B.k (nz x), flipMass O Dj v ≤ Δ)
            ∧ Dj.real ↑P + Δ / f + ε
                ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
                    (famOf O B.cn B.cd B.sc P C B.k (nz x)) p (nz x)}
            ∧ B.k ≤ C.card} else ∅)
        (prefixesAt populations B.m x) (poolAt B.M x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.M x)]
    rfl
  rw [hrw]
  exact measurableSet_of_run_data populations B _ hR

open scoped Classical in
/-- **One state's coverage, over the run.**  `coverage_of_famOf` holds at every table inside
the flat set, and that is almost every table. -/
theorem measureReal_shortCoverage_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (j : J) (hj : j ∈ populations)
    (B : Budget) (hkpos : 0 < B.k) (f γ Δ ε : ℝ) (hγ : 0 ≤ γ) (hf : 0 < f) (hε : 0 < ε)
    (hhi : (B.k : ℝ) * ((O.η + (1 - 2 * O.η) * f) + γ) ≤ (B.hi : ℝ))
    (hlo : (B.lo : ℝ) < (B.k : ℝ) * ((O.η + (1 - 2 * O.η) * (1 - f)) - γ)) :
    (runLaw μ D Dsf).real (shortCoverage O populations (D j) B Δ f ε)
      ≤ Real.exp (-2 * (B.k : ℝ) * γ ^ 2) / ε := by
  classical
  set E : ℝ := Real.exp (-2 * (B.k : ℝ) * γ ^ 2) / ε with hEdef
  have hEnn : runLaw μ D Dsf (shortCoverage O populations (D j) B Δ f ε) ≤ ENNReal.ofReal E := by
    refine runLaw_slice_le D Dsf _ (measurableSet_shortCoverage O populations (D j) B Δ f ε) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp] with d hd
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.m).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.M).image (fun i => d.1.1 i)) with hCd
    have hPeq : ∀ ω : Ω, prefixesAt populations B.m ((ω, d) : Run Ω S J) = Pd := fun _ => rfl
    have hCeq : ∀ ω : Ω, poolAt B.M ((ω, d) : Run Ω S J) = Cd := fun _ => rfl
    have hP : ∀ q ∈ Pd, q ∈ Pre := by
      intro q hq
      obtain ⟨j', hj', hq'⟩ := Finset.mem_biUnion.1 hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hq'
      exact hd j' hj' i
    have hone : (1 : S) ∈ Cd := Finset.mem_insert_self _ _
    by_cases hk : B.k ≤ Cd.card
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ shortCoverage O populations (D j) B Δ f ε}
          ⊆ {ω : Ω | (∀ v ∈ famOf O B.cn B.cd B.sc Pd Cd B.k ω, flipMass O (D j) v ≤ Δ)
            ∧ (D j).real ↑Pd + Δ / f + ε
                ≤ (D j).real {p | ¬ cutCorrect O B.lo B.hi
                    (famOf O B.cn B.cd B.sc Pd Cd B.k ω) p ω}} := by
        rintro ω ⟨h1, h2, -⟩
        exact ⟨h1, h2⟩
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
      exact ENNReal.ofReal_le_ofReal
        (coverage_of_famOf hflat O (D j) Pd Cd hP (hsupp j hj) hone B.cn B.cd B.sc B.k hk hkpos
          B.lo B.hi f γ Δ ε hγ hf hε hhi hlo)
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ shortCoverage O populations (D j) B Δ f ε}
          = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
        rintro ⟨-, -, h3⟩
        exact hk h3
      simp [hsec]
  rw [measureReal_def]
  calc (runLaw μ D Dsf (shortCoverage O populations (D j) B Δ f ε)).toReal
      ≤ (ENNReal.ofReal E).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = E := ENNReal.toReal_ofReal (by positivity)

open scoped Classical in
/-- The runs at one state whose screen lets through a candidate the table says flips more
than `Δ` of it. -/
noncomputable def screenBad (O : Oracle μ S) (populations : Finset J) (B : Budget) (Δ : ℝ) :
    Set (Run Ω S J) :=
  {x | B.m ≤ (prefixesAt populations B.m x).card
    ∧ ¬ ∀ v ∈ screenedAt O populations B x,
      ¬ (Δ * ((prefixesAt populations B.m x).card : ℝ)
        ≤ ∑ p ∈ prefixesAt populations B.m x, O.flip v p)}

open scoped Classical in
lemma measurableSet_screenBad (O : Oracle μ S) (populations : Finset J) (B : Budget) (Δ : ℝ) :
    MeasurableSet (screenBad O populations B Δ) := by
  classical
  have hR : ∀ P C : Finset S, MeasurableSet
      {x : Run Ω S J | B.m ≤ P.card ∧ ∃ v ∈ C, screenCount O P v (nz x) ≤ B.sc
        ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p} := by
    intro P C
    by_cases hm : B.m ≤ P.card
    swap
    · have : {x : Run Ω S J | B.m ≤ P.card ∧ ∃ v ∈ C, screenCount O P v (nz x) ≤ B.sc
          ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p} = (∅ : Set (Run Ω S J)) := by
        ext x; simp [hm]
      rw [this]; exact MeasurableSet.empty
    have hsimp : {x : Run Ω S J | B.m ≤ P.card ∧ ∃ v ∈ C, screenCount O P v (nz x) ≤ B.sc
        ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p}
        = {x : Run Ω S J | ∃ v ∈ C, screenCount O P v (nz x) ≤ B.sc
          ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p} := by
      ext x; simp [hm]
    rw [hsimp]
    have hcov : {x : Run Ω S J | ∃ v ∈ C, screenCount O P v (nz x) ≤ B.sc
        ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p}
        = ⋃ v ∈ C.filter (fun v => Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p),
            nz ⁻¹' {ω : Ω | screenCount O P v ω ≤ B.sc} := by
      ext x
      simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_preimage, Finset.mem_coe,
        Finset.mem_filter, exists_prop]
      exact ⟨fun ⟨v, hv, h1, h2⟩ => ⟨v, ⟨hv, h2⟩, h1⟩,
        fun ⟨v, ⟨hv, h2⟩, h1⟩ => ⟨v, hv, h1, h2⟩⟩
    rw [hcov]
    exact Finset.measurableSet_biUnion _
      (fun v _ => measurable_nz (measurableSet_screenCount_le O P v B.sc))
  have hrw : screenBad O populations B Δ
      = {x : Run Ω S J | x ∈ (fun P C => {x : Run Ω S J | B.m ≤ P.card ∧ ∃ v ∈ C,
          screenCount O P v (nz x) ≤ B.sc ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p})
        (prefixesAt populations B.m x) (poolAt B.M x)} := by
    ext x
    simp only [screenBad, screened, screenedAt, Set.mem_setOf_eq, not_forall, Finset.mem_filter,
      not_not, exists_prop, and_congr_right_iff]
    intro _
    exact ⟨fun ⟨v, hv, h⟩ => ⟨v, hv.1, hv.2, h⟩, fun ⟨v, hv, h1, h2⟩ => ⟨v, ⟨hv, h1⟩, h2⟩⟩
  rw [hrw]
  exact measurableSet_of_run_data populations B _ hR

/-- **The screen rarely lets a badly-flipping candidate through.**  One `screen_tail` per
pool member, at a fixed table: the flip counts are deterministic once the draws are, so this
is a plain union bound and costs `M + 1`. -/
theorem measureReal_screenBad_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (B : Budget) (hcd : B.cn < B.cd)
    (hsig : O.η ≤ 1 / 2) (hmpos : 0 < B.m) (Δ γ : ℝ) (hΔ : 0 < Δ) (hγ : 0 ≤ γ)
    (hsc : ∀ n : ℕ, B.m ≤ n → (B.sc : ℝ)
      ≤ (n : ℝ) * ((2 * O.η * (1 - O.η) + Δ * (1 - 2 * O.η) ^ 2) - γ)) :
    (runLaw μ D Dsf).real (screenBad O populations B Δ)
      ≤ ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γ ^ 2) := by
  classical
  set E : ℝ := ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γ ^ 2) with hEdef
  have hE0 : 0 ≤ E := by positivity
  have hEnn : runLaw μ D Dsf (screenBad O populations B Δ) ≤ ENNReal.ofReal E := by
    refine runLaw_slice_le D Dsf _ (measurableSet_screenBad O populations B Δ) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp] with d hd
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.m).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.M).image (fun i => d.1.1 i)) with hCd
    have hP : ∀ q ∈ Pd, q ∈ Pre := by
      intro q hq
      obtain ⟨j', hj', hq'⟩ := Finset.mem_biUnion.1 hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hq'
      exact hd j' hj' i
    by_cases hm : B.m ≤ Pd.card
    · have hPpos : 0 < Pd.card := lt_of_lt_of_le hmpos hm
      have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ screenBad O populations B Δ}
          ⊆ ⋃ v ∈ Cd.filter (fun v => Δ * (Pd.card : ℝ) ≤ ∑ p ∈ Pd, O.flip v p),
              {ω : Ω | screenCount O Pd v ω ≤ B.sc} := by
        rintro ω ⟨-, hbad⟩
        simp only [not_forall, not_not] at hbad
        obtain ⟨v, hv, hflip⟩ := hbad
        obtain ⟨hvC, hvs⟩ := Finset.mem_filter.1 hv
        exact Set.mem_biUnion (Finset.mem_filter.2 ⟨hvC, hflip⟩) hvs
      refine le_trans (measure_mono hsec) (le_trans (measure_biUnion_finset_le _ _) ?_)
      have hper : ∀ v ∈ Cd.filter (fun v => Δ * (Pd.card : ℝ) ≤ ∑ p ∈ Pd, O.flip v p),
          μ {ω : Ω | screenCount O Pd v ω ≤ B.sc}
            ≤ ENNReal.ofReal (Real.exp (-2 * (B.m : ℝ) * γ ^ 2)) := by
        intro v hv
        obtain ⟨-, hflip⟩ := Finset.mem_filter.1 hv
        have hv1 : v ≠ 1 := by
          intro h
          rw [h] at hflip
          have hz : ∑ p ∈ Pd, O.flip (1 : S) p = 0 := by
            refine Finset.sum_eq_zero (fun p _ => ?_)
            show O.label (p * 1) + O.label p - 2 * O.label (p * 1) * O.label p = 0
            rw [mul_one]
            rcases O.label_bit p with hl | hl <;> rw [hl] <;> ring
          rw [hz] at hflip
          exact absurd hflip (not_le.2 (mul_pos hΔ (by exact_mod_cast hPpos)))
        have hcardR : (B.m : ℝ) ≤ (Pd.card : ℝ) := by exact_mod_cast hm
        have htail := screen_tail hflat O hcd hP v hv1 Δ γ B.sc hγ hsig hflip (hsc Pd.card hm)
        rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
        refine ENNReal.ofReal_le_ofReal (le_trans htail (Real.exp_le_exp.2 ?_))
        nlinarith [sq_nonneg γ]
      refine le_trans (Finset.sum_le_sum hper) ?_
      rw [Finset.sum_const, nsmul_eq_mul, hEdef, ENNReal.ofReal_mul (by positivity),
        ENNReal.ofReal_add (by positivity) zero_le_one, ENNReal.ofReal_one,
        ENNReal.ofReal_natCast]
      refine mul_le_mul' ?_ le_rfl
      have hCard : Cd.card ≤ B.M + 1 :=
        le_trans (Finset.card_insert_le _ _)
          (by simpa using le_trans Finset.card_image_le (by simp : (Finset.range B.M).card ≤ B.M))
      have hfc : (Cd.filter (fun v => Δ * (Pd.card : ℝ) ≤ ∑ p ∈ Pd, O.flip v p)).card
          ≤ B.M + 1 := le_trans (Finset.card_filter_le _ _) hCard
      exact_mod_cast Nat.cast_le.2 hfc
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ screenBad O populations B Δ} = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
        rintro ⟨h1, -⟩
        exact hm h1
      simp [hsec]
  rw [measureReal_def]
  calc (runLaw μ D Dsf (screenBad O populations B Δ)).toReal
      ≤ (ENNReal.ofReal E).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = E := ENNReal.toReal_ofReal hE0

open scoped Classical in
/-- **A block of i.i.d. draws hits a set about as often as its mass.**  The lower tail, for
the suffix pool: a mass-`q` set is hit at least `M(q − t)` times. -/
lemma pi_hits_lower (Dsf : Measure S) [IsProbabilityMeasure Dsf] (M : ℕ) (W : Set S)
    (q t : ℝ) (hq : 0 ≤ q) (ht : 0 ≤ t) (hW : q ≤ Dsf.real W) :
    (Measure.pi fun _ : Fin M => Dsf).real
        {r : Fin M → S | (((Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W)).card : ℝ)
          ≤ (M : ℝ) * (q - t)}
      ≤ Real.exp (-2 * (M : ℝ) * t ^ 2) := by
  classical
  set ν : Measure (Fin M → S) := Measure.pi fun _ : Fin M => Dsf with hνdef
  have hWm : MeasurableSet W := measurableSet_of_countable W
  set ind : S → ℝ := W.indicator 1 with hinddef
  have hindm : Measurable ind := measurable_const.indicator hWm
  set X : Fin M → (Fin M → S) → ℝ := fun i r => ind (r i) with hXdef
  have hindep : iIndepFun X ν := iIndepFun_pi (fun _ => hindm.aemeasurable)
  have hmeas : ∀ i, AEMeasurable (X i) ν := fun i =>
    (hindm.comp (measurable_pi_apply _)).aemeasurable
  have hicc : ∀ i, ∀ᵐ r ∂ν, X i r ∈ Set.Icc (0 : ℝ) 1 := by
    intro i
    filter_upwards with r
    by_cases h : r i ∈ W
    · simp [hXdef, hinddef, Set.indicator_of_mem h]
    · simp [hXdef, hinddef, Set.indicator_of_notMem h]
  have hmean : ∀ i, ν[X i] = Dsf.real W := by
    intro i
    have hmp : MeasurePreserving (fun r : Fin M → S => r i) ν Dsf :=
      measurePreserving_eval (fun _ : Fin M => Dsf) i
    calc ν[X i] = ∫ s, ind s ∂Dsf := by
          rw [← hmp.map_eq,
            integral_map (measurable_pi_apply _).aemeasurable hindm.aestronglyMeasurable]
      _ = Dsf.real W := by rw [hinddef, integral_indicator_one hWm]
  have hsum : ((Finset.univ : Finset (Fin M)).card : ℝ) * q ≤ ∑ i, ν[X i] := by
    rw [Finset.sum_congr rfl (fun i _ => hmean i), Finset.sum_const, nsmul_eq_mul]
    have hc : (0 : ℝ) ≤ ((Finset.univ : Finset (Fin M)).card : ℝ) := Nat.cast_nonneg _
    nlinarith [hW]
  have hmain := sumLower_le X (Finset.univ : Finset (Fin M)) q t hmeas hindep hicc hsum ht
  have hcount : ∀ r : Fin M → S, ∑ i, X i r
      = (((Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W)).card : ℝ) := by
    intro r
    rw [← Finset.sum_filter_add_sum_filter_not (Finset.univ : Finset (Fin M))
      (fun i => r i ∈ W)]
    have h1 : ∑ i ∈ (Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W), X i r
        = ((((Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W)).card : ℝ)) := by
      have hone : ∀ i ∈ (Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W),
          X i r = (1 : ℝ) := by
        intro i hi
        simp [hXdef, hinddef, Set.indicator_of_mem (Finset.mem_filter.1 hi).2]
      rw [Finset.sum_congr rfl hone, Finset.sum_const, nsmul_eq_mul, mul_one]
    have h0 : ∑ i ∈ (Finset.univ : Finset (Fin M)).filter (fun i => ¬ (r i ∈ W)), X i r = 0 :=
      Finset.sum_eq_zero (fun i hi => by
        simp [hXdef, hinddef, Set.indicator_of_notMem (Finset.mem_filter.1 hi).2])
    rw [h1, h0, add_zero]
  have hcard : ((Finset.univ : Finset (Fin M)).card : ℝ) = (M : ℝ) := by simp
  refine le_trans (le_trans (measureReal_mono ?_ (measure_ne_top _ _)) hmain) ?_
  · intro r hr
    simp only [Set.mem_setOf_eq] at hr ⊢
    rw [hcount r, hcard]
    linarith [hr]
  · rw [hcard]

open scoped Classical in
/-- **The pool holds accept-preserving suffixes.**  Findability says a draw is
accept-preserving with probability at least `pAP`, so `M` draws hold about `pAP·M` of them. -/
theorem measureReal_apShort_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S) (M : ℕ)
    (pAP t : ℝ) (hpAP0 : 0 ≤ pAP) (ht : 0 ≤ t)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p}) :
    (runLaw μ D Dsf).real {x : Run Ω S J |
        (((Finset.univ : Finset (Fin M)).filter
          (fun i => ∀ p, O.label (p * sfx i.val x) = O.label p)).card : ℝ)
        ≤ (M : ℝ) * (pAP - t)}
      ≤ Real.exp (-2 * (M : ℝ) * t ^ 2) := by
  classical
  set W : Set S := {v | ∀ p, O.label (p * v) = O.label p} with hW
  have hmeasHits : Measurable (fun r : Fin M → S =>
      (((Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W)).card : ℝ)) := by
    have hWm : MeasurableSet W := (Set.to_countable _).measurableSet
    have hrw : (fun r : Fin M → S =>
        (((Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W)).card : ℝ))
        = fun r => ∑ i : Fin M, W.indicator (fun _ => (1 : ℝ)) (r i) := by
      funext r
      rw [Finset.card_filter, Nat.cast_sum]
      refine Finset.sum_congr rfl (fun i _ => ?_)
      by_cases h : r i ∈ W <;> simp [h]
    rw [hrw]
    exact Finset.measurable_sum _ (fun i _ =>
      (measurable_const.indicator hWm).comp (measurable_pi_apply i))
  have hmeasSet : MeasurableSet {r : Fin M → S |
      (((Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W)).card : ℝ)
        ≤ (M : ℝ) * (pAP - t)} := measurableSet_le hmeasHits measurable_const
  have hpre : {x : Run Ω S J |
      (((Finset.univ : Finset (Fin M)).filter
        (fun i => ∀ p, O.label (p * sfx i.val x) = O.label p)).card : ℝ)
        ≤ (M : ℝ) * (pAP - t)}
      = (fun x : Run Ω S J => (fun i : Fin M => sfx i.val x)) ⁻¹'
        {r : Fin M → S | (((Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W)).card : ℝ)
          ≤ (M : ℝ) * (pAP - t)} := rfl
  have hmeasBlock : Measurable (fun x : Run Ω S J => (fun i : Fin M => sfx i.val x)) :=
    measurable_pi_lambda _ (fun i : Fin M => measurable_sfx i.val)
  rw [hpre, measureReal_def, Measure.map_apply hmeasBlock hmeasSet
    |>.symm.trans (congrArg (fun ν : Measure (Fin M → S) => ν _) (map_suffixBlock D Dsf M)),
    ← measureReal_def]
  exact pi_hits_lower Dsf M W pAP t hpAP0 ht hpAPBound

open scoped Classical in
/-- The runs where an accept-preserving candidate the pool holds is thrown out by the
screen.  Off this event the pool's accept-preserving draws all survive to be clustered. -/
noncomputable def screenFail (O : Oracle μ S) (populations : Finset J) (B : Budget) :
    Set (Run Ω S J) :=
  {x | B.m ≤ (prefixesAt populations B.m x).card
    ∧ ¬ ∀ v ∈ poolAt B.M x, v ≠ 1 → (∀ p, O.label (p * v) = O.label p) →
      screenCount O (prefixesAt populations B.m x) v (nz x) ≤ B.sc}

open scoped Classical in
lemma measurableSet_screenFail (O : Oracle μ S) (populations : Finset J) (B : Budget) :
    MeasurableSet (screenFail O populations B) := by
  classical
  have hR : ∀ P C : Finset S, MeasurableSet (if B.m ≤ P.card then
      {x : Run Ω S J | ¬ ∀ v ∈ C, v ≠ 1 → (∀ p, O.label (p * v) = O.label p) →
        screenCount O P v (nz x) ≤ B.sc} else ∅) := by
    intro P C
    split_ifs with hm
    · have hset : {x : Run Ω S J | ¬ ∀ v ∈ C, v ≠ 1 → (∀ p, O.label (p * v) = O.label p) →
          screenCount O P v (nz x) ≤ B.sc}
          = nz ⁻¹' (⋃ v ∈ C.filter (fun v => v ≠ 1 ∧ ∀ p, O.label (p * v) = O.label p),
            {ω : Ω | ¬ (screenCount O P v ω ≤ B.sc)}) := by
        ext x
        simp only [Set.mem_setOf_eq, Set.mem_preimage, Set.mem_iUnion, Finset.mem_coe,
          Finset.mem_filter, exists_prop, not_forall]
        constructor
        · rintro ⟨v, hv, hv1, hap, hfail⟩
          exact ⟨v, ⟨hv, hv1, hap⟩, hfail⟩
        · rintro ⟨v, ⟨hv, hv1, hap⟩, hfail⟩
          exact ⟨v, hv, hv1, hap, hfail⟩
      rw [hset]
      exact measurable_nz (Finset.measurableSet_biUnion _
        (fun v _ => (measurableSet_screenCount_le O P v B.sc).compl))
    · exact MeasurableSet.empty
  have hrw : screenFail O populations B
      = {x : Run Ω S J | x ∈ (fun P C => if B.m ≤ P.card then
          {x : Run Ω S J | ¬ ∀ v ∈ C, v ≠ 1 → (∀ p, O.label (p * v) = O.label p) →
            screenCount O P v (nz x) ≤ B.sc} else ∅)
        (prefixesAt populations B.m x) (poolAt B.M x)} := by
    ext x
    simp only [screenFail, Set.mem_setOf_eq]
    by_cases hm : B.m ≤ (prefixesAt populations B.m x).card
    · rw [if_pos hm]
      exact ⟨fun h => h.2, fun h => ⟨hm, h⟩⟩
    · rw [if_neg hm]
      exact ⟨fun h => absurd h.1 hm, fun h => absurd h (Set.notMem_empty x)⟩
  rw [hrw]
  exact measurableSet_of_run_data populations B _ hR

open scoped Classical in
/-- **The screen keeps the accept-preserving candidates.**  One `screen_pass` per pool
member; the cutoff sits above the clean rate `2η(1−η)` by `γ`. -/
theorem measureReal_screenFail_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (B : Budget) (hcd : B.cn < B.cd)
    (γ : ℝ) (hγ : 0 ≤ γ)
    (hsc : ∀ n : ℕ, n ≤ populations.card * B.m →
      (n : ℝ) * (2 * O.η * (1 - O.η) + γ) ≤ (B.sc : ℝ)) :
    (runLaw μ D Dsf).real (screenFail O populations B)
      ≤ ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γ ^ 2) := by
  classical
  set E : ℝ := ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γ ^ 2) with hEdef
  have hE0 : 0 ≤ E := by positivity
  have hEnn : runLaw μ D Dsf (screenFail O populations B) ≤ ENNReal.ofReal E := by
    refine runLaw_slice_le D Dsf _ (measurableSet_screenFail O populations B) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp] with d hd
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.m).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.M).image (fun i => d.1.1 i)) with hCd
    have hP : ∀ q ∈ Pd, q ∈ Pre := by
      intro q hq
      obtain ⟨j', hj', hq'⟩ := Finset.mem_biUnion.1 hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hq'
      exact hd j' hj' i
    have hPle : Pd.card ≤ populations.card * B.m := by
      rw [hPd]
      refine le_trans Finset.card_biUnion_le ?_
      calc ∑ j ∈ populations, ((Finset.range B.m).image (fun i => d.1.2 j i)).card
          ≤ ∑ _j ∈ populations, B.m :=
            Finset.sum_le_sum (fun j _ => le_trans Finset.card_image_le (by simp))
        _ = populations.card * B.m := by rw [Finset.sum_const, smul_eq_mul]
    by_cases hm : B.m ≤ Pd.card
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ screenFail O populations B}
          ⊆ ⋃ v ∈ Cd.filter (fun v => v ≠ 1 ∧ ∀ p, O.label (p * v) = O.label p),
              {ω : Ω | ¬ (screenCount O Pd v ω ≤ B.sc)} := by
        rintro ω ⟨-, hbad⟩
        simp only [not_forall] at hbad
        obtain ⟨v, hv, hv1, hap, hfail⟩ := hbad
        exact Set.mem_biUnion (Finset.mem_filter.2 ⟨hv, hv1, hap⟩) hfail
      refine le_trans (measure_mono hsec) (le_trans (measure_biUnion_finset_le _ _) ?_)
      have hper : ∀ v ∈ Cd.filter (fun v => v ≠ 1 ∧ ∀ p, O.label (p * v) = O.label p),
          μ {ω : Ω | ¬ (screenCount O Pd v ω ≤ B.sc)}
            ≤ ENNReal.ofReal (Real.exp (-2 * (B.m : ℝ) * γ ^ 2)) := by
        intro v hv
        obtain ⟨-, hv1, hap⟩ := Finset.mem_filter.1 hv
        have hclean : ∀ p ∈ Pd, O.flip v p = 0 := by
          intro p _
          show O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p = 0
          rw [hap p]
          rcases O.label_bit p with hl | hl <;> rw [hl] <;> ring
        have htail := screen_pass hflat O hcd hP v hv1 γ B.sc hγ hclean (hsc Pd.card hPle)
        have hcardR : (B.m : ℝ) ≤ (Pd.card : ℝ) := by exact_mod_cast hm
        rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
        refine ENNReal.ofReal_le_ofReal (le_trans htail (Real.exp_le_exp.2 ?_))
        nlinarith [sq_nonneg γ]
      refine le_trans (Finset.sum_le_sum hper) ?_
      rw [Finset.sum_const, nsmul_eq_mul, hEdef, ENNReal.ofReal_mul (by positivity),
        ENNReal.ofReal_add (by positivity) zero_le_one, ENNReal.ofReal_one,
        ENNReal.ofReal_natCast]
      refine mul_le_mul' ?_ le_rfl
      have hCard : Cd.card ≤ B.M + 1 :=
        le_trans (Finset.card_insert_le _ _)
          (by simpa using le_trans Finset.card_image_le (by simp : (Finset.range B.M).card ≤ B.M))
      have hfc : (Cd.filter (fun v => v ≠ 1 ∧ ∀ p, O.label (p * v) = O.label p)).card
          ≤ B.M + 1 := le_trans (Finset.card_filter_le _ _) hCard
      exact_mod_cast Nat.cast_le.2 hfc
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ screenFail O populations B}
          = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
        rintro ⟨h1, -⟩
        exact hm h1
      simp [hsec]
  rw [measureReal_def]
  calc (runLaw μ D Dsf (screenFail O populations B)).toReal
      ≤ (ENNReal.ofReal E).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = E := ENNReal.toReal_ofReal hE0

open scoped Classical in
/-- **The screened pool is big enough to cluster.**  Findability puts `k` accept-preserving
suffixes in the pool, the screen keeps them, and distinct draws keep them distinct — so the
clustering has `k` candidates to choose from and does not stall. -/
theorem measureReal_smallScreen_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (B : Budget) (hcd : B.cn < B.cd)
    (j₀ : J) (hj₀ : j₀ ∈ populations)
    (γ pAP t ρsf ρ : ℝ) (hγ : 0 ≤ γ) (hpAP0 : 0 ≤ pAP) (ht : 0 ≤ t)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hsc : ∀ n : ℕ, n ≤ populations.card * B.m →
      (n : ℝ) * (2 * O.η * (1 - O.η) + γ) ≤ (B.sc : ℝ))
    (hcount : (B.k : ℝ) ≤ (B.M : ℝ) * (pAP - t))
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hρj : collisionMass (D j₀) ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runLaw μ D Dsf).real
        {x : Run Ω S J | ¬ (B.k ≤ (screenedAt O populations B x).card)}
      ≤ (B.M : ℝ) ^ 2 * ρsf + (Real.exp (-2 * (B.M : ℝ) * t ^ 2)
        + ((B.m : ℝ) ^ 2 * ρ + ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γ ^ 2))) := by
  classical
  set E1 : Set (Run Ω S J) :=
    {x | ¬ Function.Injective (fun i : Fin B.M => sfx i.val x)} with hE1
  set E2 : Set (Run Ω S J) := {x | (((Finset.univ : Finset (Fin B.M)).filter
    (fun i => ∀ p, O.label (p * sfx i.val x) = O.label p)).card : ℝ)
      ≤ (B.M : ℝ) * (pAP - t)} with hE2
  set E3 : Set (Run Ω S J) :=
    {x | ¬ Function.Injective (fun i : Fin B.m => prf j₀ i.val x)} with hE3
  set E4 : Set (Run Ω S J) := screenFail O populations B with hE4
  have hsub : {x : Run Ω S J | ¬ (B.k ≤ (screenedAt O populations B x).card)}
      ⊆ E1 ∪ (E2 ∪ (E3 ∪ E4)) := by
    intro x hx
    by_contra hcon
    simp only [Set.mem_union, not_or] at hcon
    obtain ⟨h1, h2, h3, h4⟩ := hcon
    have hinj : Function.Injective (fun i : Fin B.M => sfx i.val x) := by
      by_contra h; exact h1 h
    have hapcount : (B.M : ℝ) * (pAP - t)
        < (((Finset.univ : Finset (Fin B.M)).filter
          (fun i => ∀ p, O.label (p * sfx i.val x) = O.label p)).card : ℝ) := not_le.1 h2
    have hmP : B.m ≤ (prefixesAt populations B.m x).card := by
      have hinjP : Function.Injective (fun i : Fin B.m => prf j₀ i.val x) := by
        by_contra h; exact h3 h
      have hinjOn : Set.InjOn (fun i => prf j₀ i x) ↑(Finset.range B.m) := by
        intro a ha b hb hab
        have := hinjP (show (fun i : Fin B.m => prf j₀ i.val x)
            ⟨a, Finset.mem_range.1 (by simpa using ha)⟩
          = (fun i : Fin B.m => prf j₀ i.val x)
            ⟨b, Finset.mem_range.1 (by simpa using hb)⟩ from hab)
        simpa using congrArg Fin.val this
      have hcardOf : (prefixesOf j₀ B.m x).card = B.m := by
        unfold prefixesOf
        rw [Finset.card_image_of_injOn hinjOn, Finset.card_range]
      calc B.m = (prefixesOf j₀ B.m x).card := hcardOf.symm
        _ ≤ (prefixesAt populations B.m x).card :=
            Finset.card_le_card (fun q hq => Finset.mem_biUnion.2 ⟨j₀, hj₀, hq⟩)
    have hscreen : ∀ v ∈ poolAt B.M x, v ≠ 1 → (∀ p, O.label (p * v) = O.label p) →
        screenCount O (prefixesAt populations B.m x) v (nz x) ≤ B.sc := by
      by_contra h
      exact h4 ⟨hmP, h⟩
    -- the accept-preserving draws, as distinct strings, all survive the screen
    set I : Finset (Fin B.M) := (Finset.univ : Finset (Fin B.M)).filter
      (fun i => ∀ p, O.label (p * sfx i.val x) = O.label p) with hI
    have hImg : I.image (fun i => sfx i.val x) ⊆ screenedAt O populations B x := by
      intro v hv
      obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hv
      have hpool : sfx i.val x ∈ poolAt B.M x :=
        Finset.mem_insert_of_mem (Finset.mem_image.2 ⟨i.val,
          Finset.mem_range.2 i.isLt, rfl⟩)
      refine Finset.mem_filter.2 ⟨hpool, ?_⟩
      by_cases hv1 : sfx i.val x = 1
      · rw [hv1, screenCount_one]
        exact Nat.zero_le _
      · exact hscreen _ hpool hv1 (Finset.mem_filter.1 hi).2
    have hcardI : I.card = (I.image (fun i => sfx i.val x)).card :=
      (Finset.card_image_of_injective I hinj).symm
    have hk : (B.k : ℝ) ≤ ((I.image (fun i => sfx i.val x)).card : ℝ) := by
      rw [← hcardI]
      exact le_trans hcount (le_of_lt hapcount)
    have : B.k ≤ (screenedAt O populations B x).card := by
      have hcast : ((I.image (fun i => sfx i.val x)).card : ℝ)
          ≤ ((screenedAt O populations B x).card : ℝ) := by
        exact_mod_cast Finset.card_le_card hImg
      have : (B.k : ℝ) ≤ ((screenedAt O populations B x).card : ℝ) := le_trans hk hcast
      exact_mod_cast this
    exact hx this
  calc (runLaw μ D Dsf).real {x : Run Ω S J | ¬ (B.k ≤ (screenedAt O populations B x).card)}
      ≤ (runLaw μ D Dsf).real (E1 ∪ (E2 ∪ (E3 ∪ E4))) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ (runLaw μ D Dsf).real E1 + ((runLaw μ D Dsf).real E2
        + ((runLaw μ D Dsf).real E3 + (runLaw μ D Dsf).real E4)) := by
        have h34 := measureReal_union_le (μ := runLaw μ D Dsf) E3 E4
        have h234 := measureReal_union_le (μ := runLaw μ D Dsf) E2 (E3 ∪ E4)
        have hall := measureReal_union_le (μ := runLaw μ D Dsf) E1 (E2 ∪ (E3 ∪ E4))
        linarith
    _ ≤ (B.M : ℝ) ^ 2 * ρsf + (Real.exp (-2 * (B.M : ℝ) * t ^ 2)
        + ((B.m : ℝ) ^ 2 * ρ + ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γ ^ 2))) := by
        gcongr
        · exact suffix_not_injective_le D Dsf B.M ρsf hρsf hρsf0
        · exact measureReal_apShort_le D Dsf O B.M pAP t hpAP0 ht hpAPBound
        · exact prefix_not_injective_le D Dsf j₀ B.m ρ hρj hρ0
        · exact measureReal_screenFail_le hflat O populations D Dsf hsupp B hcd γ hγ hsc

open scoped Classical in
/-- **Every member the clustering keeps is clean.**  Three things have to go right: the
population's draws distinct, the screen holding, and the drawn prefixes not understating a
candidate's flip mass.  The Lloyd ranking does not appear — the family is a subset of what
the screen left, so it inherits the bound.  (Issue #288.) -/
theorem measureReal_dirtyMember_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (j : J) (hj : j ∈ populations)
    (B : Budget) (hcd : B.cn < B.cd) (hsig : O.η ≤ 1 / 2) (hmpos : 0 < B.m)
    (Δ γ g ρ : ℝ) (hΔ : 0 < Δ) (hγ : 0 ≤ γ) (hg : 0 ≤ g) (hρ0 : 0 ≤ ρ)
    (hρD : collisionMass (D j) ≤ ρ)
    (hsc : ∀ n : ℕ, B.m ≤ n → (B.sc : ℝ)
      ≤ (n : ℝ) * ((2 * O.η * (1 - O.η) + Δ * (1 - 2 * O.η) ^ 2) - γ)) :
    (runLaw μ D Dsf).real {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O populations x B,
        flipMass O (D j) v ≤ (populations.card : ℝ) * Δ + g}
      ≤ (B.m : ℝ) ^ 2 * ρ + ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γ ^ 2)
        + (B.M : ℝ) * Real.exp (-2 * (B.m : ℝ) * g ^ 2) := by
  classical
  set Δp : ℝ := (populations.card : ℝ) * Δ + g with hΔp
  set E1 : Set (Run Ω S J) :=
    {x | ¬ Function.Injective (fun i : Fin B.m => prf j i.val x)} with hE1
  set E2 : Set (Run Ω S J) := screenBad O populations B Δ with hE2
  set E3 : Set (Run Ω S J) := {x | ¬ ∀ v ∈ poolAt B.M x,
    (∑ i : Fin B.m, O.flip v (prf j i.val x) ≤ (B.m : ℝ) * (Δp - g)) →
      flipMass O (D j) v ≤ Δp} with hE3
  have hsub : {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O populations x B,
      flipMass O (D j) v ≤ Δp} ⊆ (E1 ∪ E2) ∪ E3 := by
    intro x hx
    by_contra hnot
    simp only [Set.mem_union, not_or] at hnot
    obtain ⟨⟨h1, h2⟩, h3⟩ := hnot
    have hinjP : Function.Injective (fun i : Fin B.m => prf j i.val x) := by
      by_contra h; exact h1 h
    have hcardPre : B.m ≤ (prefixesAt populations B.m x).card := by
      have hinjOn : Set.InjOn (fun i => prf j i x) ↑(Finset.range B.m) := by
        intro a ha b hb hab
        have := hinjP (show (fun i : Fin B.m => prf j i.val x)
            ⟨a, Finset.mem_range.1 (by simpa using ha)⟩
          = (fun i : Fin B.m => prf j i.val x)
            ⟨b, Finset.mem_range.1 (by simpa using hb)⟩ from hab)
        simpa using congrArg Fin.val this
      have hcardOf : (prefixesOf j B.m x).card = B.m := by
        unfold prefixesOf
        rw [Finset.card_image_of_injOn hinjOn, Finset.card_range]
      calc B.m = (prefixesOf j B.m x).card := hcardOf.symm
        _ ≤ (prefixesAt populations B.m x).card :=
            Finset.card_le_card (fun q hq => Finset.mem_biUnion.2 ⟨j, hj, hq⟩)
    -- the screen held, so nothing it left flips much of the table
    have hscreen : ∀ v ∈ screenedAt O populations B x,
        ¬ (Δ * ((prefixesAt populations B.m x).card : ℝ)
          ≤ ∑ p ∈ prefixesAt populations B.m x, O.flip v p) := by
      by_contra h
      exact h2 ⟨hcardPre, h⟩
    have hrank := clusterAt_flip_bound O populations B x Δ hscreen
    obtain ⟨v, hv, hbad⟩ : ∃ v ∈ clusterAt O populations x B, ¬ (flipMass O (D j) v ≤ Δp) := by
      by_contra h
      exact hx (fun v hv => by
        by_contra hc
        exact h ⟨v, hv, hc⟩)
    have hvpool : v ∈ poolAt B.M x := clusterAt_subset O populations B x hv
    have hcount : ∑ i : Fin B.m, O.flip v (prf j i.val x) ≤ (B.m : ℝ) * (Δp - g) := by
      rw [sum_eq_sum_prefixesOf O j B.m x v hinjP]
      have hle1 := sum_prefixesOf_le_prefixesAt O populations j hj B.m x v
      have hle2 := not_le.1 (hrank v hv)
      have hle3 : ((prefixesAt populations B.m x).card : ℝ)
          ≤ (populations.card : ℝ) * (B.m : ℝ) := by
        exact_mod_cast card_prefixesAt_le populations B.m x
      have hΔ0 : (0 : ℝ) ≤ Δ := le_of_lt hΔ
      have hrw : (B.m : ℝ) * (Δp - g) = (populations.card : ℝ) * Δ * (B.m : ℝ) := by
        rw [hΔp]; ring
      rw [hrw]
      nlinarith
    exact hbad (by
      by_contra h
      exact h3 (by
        simp only [hE3, Set.mem_setOf_eq, not_forall]
        exact ⟨v, hvpool, hcount, h⟩))
  calc (runLaw μ D Dsf).real {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O populations x B,
        flipMass O (D j) v ≤ Δp}
      ≤ (runLaw μ D Dsf).real ((E1 ∪ E2) ∪ E3) := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ((runLaw μ D Dsf).real E1 + (runLaw μ D Dsf).real E2) + (runLaw μ D Dsf).real E3 := by
        have h12 := measureReal_union_le (μ := runLaw μ D Dsf) E1 E2
        have h123 := measureReal_union_le (μ := runLaw μ D Dsf) (E1 ∪ E2) E3
        linarith
    _ ≤ _ := by
        gcongr
        · exact prefix_not_injective_le D Dsf j B.m ρ hρD hρ0
        · exact measureReal_screenBad_le hflat O populations D Dsf hsupp B hcd hsig hmpos Δ γ
            hΔ hγ hsc
        · exact pool_flipMass_le D Dsf O j B.m B.M Δp g hg (by positivity)

lemma measurableSet_cutCorrect' (O : Oracle μ S) (lo hi : ℕ) (A₀ : Finset S) (p : S) :
    MeasurableSet[noiseAlg O Set.univ] {ω | ¬ cutCorrect O lo hi A₀ p ω} :=
  measurableSet_filter_pred_map O (T := Set.univ) (fun v => p * v)
    (by simp) (fun U => ¬ ((hi < Finset.card U → O.label p = 1)
      ∧ (Finset.card U ≤ lo → O.label p = 0)))

open scoped Classical in
/-- Events about the table, the pool and the certification draws are measurable: all three
take countably many values. -/
lemma measurableSet_of_run_data_cert (populations : Finset J) (j : J) (B : Budget)
    (R : Finset S → Finset S → (Fin B.m → S) → Set (Run Ω S J))
    (hR : ∀ P C t, MeasurableSet (R P C t)) :
    MeasurableSet {x : Run Ω S J | x ∈ R (prefixesAt populations B.m x) (poolAt B.M x)
      (fun i : Fin B.m => cert j i.val x)} := by
  classical
  have hcov : {x : Run Ω S J | x ∈ R (prefixesAt populations B.m x) (poolAt B.M x)
        (fun i : Fin B.m => cert j i.val x)}
      = ⋃ z : Finset S × Finset S × (Fin B.m → S),
          ((({x : Run Ω S J | prefixesAt populations B.m x = z.1}
            ∩ {x : Run Ω S J | poolAt B.M x = z.2.1})
            ∩ ⋂ i : Fin B.m, {x : Run Ω S J | cert j i.val x = z.2.2 i})
          ∩ R z.1 z.2.1 z.2.2) := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_iInter]
    refine ⟨fun h => ⟨(prefixesAt populations B.m x, poolAt B.M x,
      fun i : Fin B.m => cert j i.val x), ⟨⟨rfl, rfl⟩, fun _ => rfl⟩, h⟩, ?_⟩
    rintro ⟨⟨P, C, t⟩, ⟨⟨hP, hC⟩, ht⟩, hx⟩
    simp only at hP hC ht
    rw [hP, hC, show (fun i : Fin B.m => cert j i.val x) = t from funext ht]
    exact hx
  rw [hcov]
  exact MeasurableSet.iUnion (fun z =>
    (((measurableSet_prefixesAt populations B.m z.1).inter
      (measurableSet_poolAt B.M z.2.1)).inter
        (MeasurableSet.iInter (fun i : Fin B.m =>
          measurableSet_eq_fun (measurable_cert j i.val) measurable_const))).inter
      (hR z.1 z.2.1 z.2.2))

lemma certOf_eq_image (j : J) (m : ℕ) (x : Run Ω S J) :
    certOf j m x = (Finset.univ : Finset (Fin m)).image (fun i => cert j i.val x) :=
  image_range_eq_image_univ m (fun i => cert j i x)

open scoped Classical in
/-- **The certification sample misses a wrong cut.**  The family is fixed before the sample
is drawn, so a cut wrong on `εcov` of the population is hit `εcov·m` times up to the
Hoeffding slack `t`. -/
noncomputable def hitShort (O : Oracle μ S) (populations : Finset J) (Dj : Measure S) (j : J)
    (B : Budget) (εcov t : ℝ) : Set (Run Ω S J) :=
  {x | Function.Injective (fun i : Fin B.m => cert j i.val x)
    ∧ εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi (clusterAt O populations x B) p (nz x)}
    ∧ (((certOf j B.m x).filter (fun p =>
        ¬ cutCorrect O B.lo B.hi (clusterAt O populations x B) p (nz x))).card : ℝ)
      ≤ (B.m : ℝ) * (εcov - t)}

open scoped Classical in
lemma measurableSet_hitShort (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (j : J) (B : Budget) (εcov t : ℝ) :
    MeasurableSet (hitShort O populations Dj j B εcov t) := by
  classical
  have hR : ∀ (P C : Finset S) (tt : Fin B.m → S), MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | Function.Injective tt
        ∧ εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
            (clusterOf O B.cn B.cd B.sc P C B.k (nz x)) p (nz x)}
        ∧ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
            ¬ cutCorrect O B.lo B.hi (clusterOf O B.cn B.cd B.sc P C B.k (nz x)) p (nz x))).card
              : ℝ)
          ≤ (B.m : ℝ) * (εcov - t)} else ∅) := by
    intro P C tt
    split_ifs with hone
    · by_cases hinj : Function.Injective tt
      · have hω : MeasurableSet {ω : Ω |
            εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
                (clusterOf O B.cn B.cd B.sc P C B.k ω) p ω}
            ∧ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
                ¬ cutCorrect O B.lo B.hi (clusterOf O B.cn B.cd B.sc P C B.k ω) p ω)).card : ℝ)
              ≤ (B.m : ℝ) * (εcov - t)} := by
          refine measurableSet_of_fam (T := C.powerset)
            (fun ω => Finset.mem_powerset.2 (clusterOf_subset O B.cn B.cd B.sc P C B.k ω hone))
            (fun A₀ => measurableSet_clusterOf O B.cn B.cd B.sc P C B.k hone A₀)
            (fun A₀ => {ω | εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi A₀ p ω}
              ∧ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
                  ¬ cutCorrect O B.lo B.hi A₀ p ω)).card : ℝ) ≤ (B.m : ℝ) * (εcov - t)})
            (fun A₀ => MeasurableSet.inter
              (measurableSet_le measurable_const (measurable_badMassReal O Dj B.lo B.hi A₀))
              (noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O
                (fun p ω => ¬ cutCorrect O B.lo B.hi A₀ p ω)
                (fun p _ => measurableSet_cutCorrect' O B.lo B.hi A₀ p)
                (fun U => ((U.card : ℝ) ≤ (B.m : ℝ) * (εcov - t))))))
        have hset : {x : Run Ω S J | Function.Injective tt
            ∧ εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
                (clusterOf O B.cn B.cd B.sc P C B.k (nz x)) p (nz x)}
            ∧ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
                ¬ cutCorrect O B.lo B.hi
                  (clusterOf O B.cn B.cd B.sc P C B.k (nz x)) p (nz x))).card : ℝ)
              ≤ (B.m : ℝ) * (εcov - t)}
            = nz ⁻¹' {ω : Ω |
              εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
                  (clusterOf O B.cn B.cd B.sc P C B.k ω) p ω}
              ∧ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
                  ¬ cutCorrect O B.lo B.hi (clusterOf O B.cn B.cd B.sc P C B.k ω) p ω)).card : ℝ)
                ≤ (B.m : ℝ) * (εcov - t)} := by
          ext x
          simp only [Set.mem_setOf_eq, Set.mem_preimage, hinj, true_and]
        rw [hset]
        exact measurable_nz hω
      · simp [hinj]
    · exact MeasurableSet.empty
  have hrw : hitShort O populations Dj j B εcov t
      = {x : Run Ω S J | x ∈ (fun P C tt => if (1 : S) ∈ C then
          {x : Run Ω S J | Function.Injective tt
            ∧ εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
                (clusterOf O B.cn B.cd B.sc P C B.k (nz x)) p (nz x)}
            ∧ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
                ¬ cutCorrect O B.lo B.hi
                  (clusterOf O B.cn B.cd B.sc P C B.k (nz x)) p (nz x))).card : ℝ)
              ≤ (B.m : ℝ) * (εcov - t)} else ∅)
        (prefixesAt populations B.m x) (poolAt B.M x)
        (fun i : Fin B.m => cert j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.M x), hitShort,
      ← certOf_eq_image j B.m x, ← clusterAt_eq_clusterOf O populations B x]
    tauto
  rw [hrw]
  exact measurableSet_of_run_data_cert populations j B _ hR

lemma measurableSet_voteCount_gt (O : Oracle μ S) (F : Finset S) (n : ℕ) (p : S) :
    MeasurableSet[noiseAlg O Set.univ] {ω | n < voteCount O F p ω} :=
  measurableSet_filter_pred_map O (T := Set.univ) (fun v => p * v) (by simp)
    (fun V => n < V.card)

lemma measurableSet_voteCount_le (O : Oracle μ S) (F : Finset S) (n : ℕ) (p : S) :
    MeasurableSet[noiseAlg O Set.univ] {ω | voteCount O F p ω ≤ n} :=
  measurableSet_filter_pred_map O (T := Set.univ) (fun v => p * v) (by simp)
    (fun V => V.card ≤ n)

open scoped Classical in
lemma measurableSet_sideOf_gt (O : Oracle μ S) (A F : Finset S) (n : ℕ) (U : Finset S) :
    MeasurableSet {ω | A.filter (fun p => n < voteCount O F p ω) = U} :=
  noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O
    (fun p ω => n < voteCount O F p ω)
    (fun p _ => measurableSet_voteCount_gt O F n p) (fun V => V = U))

open scoped Classical in
lemma measurableSet_sideOf_le (O : Oracle μ S) (A F : Finset S) (n : ℕ) (U : Finset S) :
    MeasurableSet {ω | A.filter (fun p => voteCount O F p ω ≤ n) = U} :=
  noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O
    (fun p ω => voteCount O F p ω ≤ n)
    (fun p _ => measurableSet_voteCount_le O F n p) (fun V => V = U))

open scoped Classical in
lemma measurableSet_hits_ge (O : Oracle μ S) (U : Finset S) (r : ℝ) :
    MeasurableSet {ω | r ≤ ((U.filter (fun p => mq O p ω = 1)).card : ℝ)} :=
  noiseAlg_le O Set.univ _
    (measurableSet_filter_pred O (T := Set.univ) (by simp) (fun V => r ≤ (V.card : ℝ)))

open scoped Classical in
lemma measurableSet_hits_le (O : Oracle μ S) (U : Finset S) (r : ℝ) :
    MeasurableSet {ω | ((U.filter (fun p => mq O p ω = 1)).card : ℝ) ≤ r} :=
  noiseAlg_le O Set.univ _
    (measurableSet_filter_pred O (T := Set.univ) (by simp) (fun V => (V.card : ℝ) ≤ r))

open scoped Classical in
/-- **The gate admits an accept side that is a `c` fraction wrong.**  The side is scored on
prefixes the clustering never read, so `gate_accept_sound` applies to it as it stands. -/
noncomputable def gateBadAcc (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (εcov c β : ℝ) : Set (Run Ω S J) :=
  {x | Disjoint (prefixesAt populations B.m x) (certOf j B.m x)
    ∧ β * (B.m : ℝ) ≤ ((sideAcc O populations j B x).card : ℝ)
    ∧ c * ((sideAcc O populations j B x).card : ℝ)
        ≤ (((sideAcc O populations j B x).filter (fun p => O.label p = 0)).card : ℝ)
    ∧ ((sideAcc O populations j B x).card : ℝ) * gateAcc O εcov
        ≤ (((sideAcc O populations j B x).filter (fun p => mq O p (nz x) = 1)).card : ℝ)}

open scoped Classical in
/-- The mirror: it rejects a side a `c` fraction of which is truly accepting. -/
noncomputable def gateBadRej (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (εcov c β : ℝ) : Set (Run Ω S J) :=
  {x | Disjoint (prefixesAt populations B.m x) (certOf j B.m x)
    ∧ β * (B.m : ℝ) ≤ ((sideRej O populations j B x).card : ℝ)
    ∧ c * ((sideRej O populations j B x).card : ℝ) ≤ ∑ p ∈ sideRej O populations j B x, O.label p
    ∧ (((sideRej O populations j B x).filter (fun p => mq O p (nz x) = 1)).card : ℝ)
        ≤ ((sideRej O populations j B x).card : ℝ) * gateRej O εcov}

/-- Distinct draws are counted once each, so the sample's hit count is the draw count. -/
lemma card_filter_certOf (j : J) (m : ℕ) (x : Run Ω S J) (Q : S → Prop)
    (instA : DecidablePred Q) (instB : DecidablePred (fun i : ℕ => Q (cert j i x)))
    (hinj : Function.Injective (fun i : Fin m => cert j i.val x)) :
    (@Finset.filter _ (fun i : ℕ => Q (cert j i x)) instB (Finset.range m)).card
      = (@Finset.filter _ Q instA (certOf j m x)).card := by
  refine Finset.card_nbij (fun i => cert j i x) (fun i hi => ?_) (fun a ha b hb hab => ?_)
    (fun p hp => ?_)
  · obtain ⟨hi, hQ⟩ := Finset.mem_filter.1 hi
    exact Finset.mem_filter.2 ⟨Finset.mem_image.2 ⟨i, hi, rfl⟩, hQ⟩
  · simp only [Finset.coe_filter, Set.mem_setOf_eq, Finset.mem_range] at ha hb
    have := hinj (show (fun i : Fin m => cert j i.val x) ⟨a, ha.1⟩
      = (fun i : Fin m => cert j i.val x) ⟨b, hb.1⟩ from hab)
    simpa using congrArg Fin.val this
  · simp only [Finset.coe_filter, Set.mem_setOf_eq] at hp
    obtain ⟨hpC, hQ⟩ := hp
    unfold certOf at hpC
    obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hpC
    exact ⟨i, by simp only [Finset.coe_filter, Set.mem_setOf_eq, Finset.mem_range,
      Finset.mem_range.1 hi, true_and]; exact hQ, rfl⟩

open scoped Classical in
/-- The gate's accept side with the draws fixed. -/
noncomputable def sideAccOf (O : Oracle μ S) (B : Budget) (P C A : Finset S) (ω : Ω) :
    Finset S :=
  A.filter (fun p => B.hi - 1 < voteCount O ((clusterOf O B.cn B.cd B.sc P C B.k ω).erase 1) p ω)

open scoped Classical in
/-- Its reject twin. -/
noncomputable def sideRejOf (O : Oracle μ S) (B : Budget) (P C A : Finset S) (ω : Ω) :
    Finset S :=
  A.filter (fun p => voteCount O ((clusterOf O B.cn B.cd B.sc P C B.k ω).erase 1) p ω ≤ B.lo)

lemma sideAcc_eq_sideAccOf (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (x : Run Ω S J) :
    sideAcc O populations j B x
      = sideAccOf O B (prefixesAt populations B.m x) (poolAt B.M x) (certOf j B.m x) (nz x) :=
  rfl

lemma sideRej_eq_sideRejOf (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (x : Run Ω S J) :
    sideRej O populations j B x
      = sideRejOf O B (prefixesAt populations B.m x) (poolAt B.M x) (certOf j B.m x) (nz x) :=
  rfl

open scoped Classical in
/-- Events about the accept side are measurable: decompose over the family, then over the
side it cuts out. -/
lemma measurableSet_of_sideAccOf (O : Oracle μ S) (B : Budget) (P C A : Finset S)
    (hone : (1 : S) ∈ C) (R : Finset S → Set Ω) (hR : ∀ U, MeasurableSet (R U)) :
    MeasurableSet {ω | ω ∈ R (sideAccOf O B P C A ω)} := by
  classical
  exact measurableSet_of_fam (T := C.powerset)
    (fun ω => Finset.mem_powerset.2 (clusterOf_subset O B.cn B.cd B.sc P C B.k ω hone))
    (fun A₀ => measurableSet_clusterOf O B.cn B.cd B.sc P C B.k hone A₀)
    (fun A₀ => {ω | ω ∈ R (A.filter (fun p => B.hi - 1 < voteCount O (A₀.erase 1) p ω))})
    (fun A₀ => measurableSet_of_fam (T := A.powerset)
      (fun ω => Finset.mem_powerset.2 (Finset.filter_subset _ _))
      (fun U => measurableSet_sideOf_gt O A (A₀.erase 1) (B.hi - 1) U) R hR)

open scoped Classical in
lemma measurableSet_of_sideRejOf (O : Oracle μ S) (B : Budget) (P C A : Finset S)
    (hone : (1 : S) ∈ C) (R : Finset S → Set Ω) (hR : ∀ U, MeasurableSet (R U)) :
    MeasurableSet {ω | ω ∈ R (sideRejOf O B P C A ω)} := by
  classical
  exact measurableSet_of_fam (T := C.powerset)
    (fun ω => Finset.mem_powerset.2 (clusterOf_subset O B.cn B.cd B.sc P C B.k ω hone))
    (fun A₀ => measurableSet_clusterOf O B.cn B.cd B.sc P C B.k hone A₀)
    (fun A₀ => {ω | ω ∈ R (A.filter (fun p => voteCount O (A₀.erase 1) p ω ≤ B.lo))})
    (fun A₀ => measurableSet_of_fam (T := A.powerset)
      (fun ω => Finset.mem_powerset.2 (Finset.filter_subset _ _))
      (fun U => measurableSet_sideOf_le O A (A₀.erase 1) B.lo U) R hR)

open scoped Classical in
lemma measurableSet_gateBadAcc (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (εcov c β : ℝ) : MeasurableSet (gateBadAcc O populations j B εcov c β) := by
  classical
  have hR : ∀ (P C : Finset S) (t : Fin B.m → S), MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.m)).image t)
        ∧ nz x ∈ {ω : Ω | ω ∈ (fun U : Finset S => {ω : Ω |
            β * (B.m : ℝ) ≤ (U.card : ℝ)
            ∧ c * (U.card : ℝ) ≤ ((U.filter (fun p => O.label p = 0)).card : ℝ)
            ∧ (U.card : ℝ) * gateAcc O εcov
                ≤ ((U.filter (fun p => mq O p ω = 1)).card : ℝ)})
          (sideAccOf O B P C ((Finset.univ : Finset (Fin B.m)).image t) ω)}} else ∅) := by
    intro P C t
    split_ifs with hone
    · by_cases hdisj : Disjoint P ((Finset.univ : Finset (Fin B.m)).image t)
      · have hset : {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.m)).image t)
            ∧ nz x ∈ {ω : Ω | ω ∈ (fun U : Finset S => {ω : Ω |
                β * (B.m : ℝ) ≤ (U.card : ℝ)
                ∧ c * (U.card : ℝ) ≤ ((U.filter (fun p => O.label p = 0)).card : ℝ)
                ∧ (U.card : ℝ) * gateAcc O εcov
                    ≤ ((U.filter (fun p => mq O p ω = 1)).card : ℝ)})
              (sideAccOf O B P C ((Finset.univ : Finset (Fin B.m)).image t) ω)}}
            = nz ⁻¹' {ω : Ω | ω ∈ (fun U : Finset S => {ω : Ω |
                β * (B.m : ℝ) ≤ (U.card : ℝ)
                ∧ c * (U.card : ℝ) ≤ ((U.filter (fun p => O.label p = 0)).card : ℝ)
                ∧ (U.card : ℝ) * gateAcc O εcov
                    ≤ ((U.filter (fun p => mq O p ω = 1)).card : ℝ)})
              (sideAccOf O B P C ((Finset.univ : Finset (Fin B.m)).image t) ω)} := by
          ext x
          simp only [Set.mem_setOf_eq, Set.mem_preimage, hdisj, true_and]
        rw [hset]
        refine measurable_nz (measurableSet_of_sideAccOf O B P C _ hone
          (fun U : Finset S => {ω : Ω |
            β * (B.m : ℝ) ≤ (U.card : ℝ)
            ∧ c * (U.card : ℝ) ≤ ((U.filter (fun p => O.label p = 0)).card : ℝ)
            ∧ (U.card : ℝ) * gateAcc O εcov
                ≤ ((U.filter (fun p => mq O p ω = 1)).card : ℝ)}) (fun U => ?_))
        by_cases h1 : β * (B.m : ℝ) ≤ (U.card : ℝ)
        · by_cases h2 : c * (U.card : ℝ) ≤ ((U.filter (fun p => O.label p = 0)).card : ℝ)
          · have hrw : {ω : Ω | β * (B.m : ℝ) ≤ (U.card : ℝ)
                ∧ c * (U.card : ℝ) ≤ ((U.filter (fun p => O.label p = 0)).card : ℝ)
                ∧ (U.card : ℝ) * gateAcc O εcov
                    ≤ ((U.filter (fun p => mq O p ω = 1)).card : ℝ)}
                = {ω : Ω | (U.card : ℝ) * gateAcc O εcov
                    ≤ ((U.filter (fun p => mq O p ω = 1)).card : ℝ)} := by
              ext ω; simp only [Set.mem_setOf_eq, h1, h2, true_and]
            rw [hrw]
            exact measurableSet_hits_ge O U _
          · simpa [h2] using MeasurableSet.empty
        · simpa [h1] using MeasurableSet.empty
      · simpa [hdisj] using MeasurableSet.empty
    · exact MeasurableSet.empty
  have hrw : gateBadAcc O populations j B εcov c β
      = {x : Run Ω S J | x ∈ (fun P C t => if (1 : S) ∈ C then
          {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.m)).image t)
            ∧ nz x ∈ {ω : Ω | ω ∈ (fun U : Finset S => {ω : Ω |
                β * (B.m : ℝ) ≤ (U.card : ℝ)
                ∧ c * (U.card : ℝ) ≤ ((U.filter (fun p => O.label p = 0)).card : ℝ)
                ∧ (U.card : ℝ) * gateAcc O εcov
                    ≤ ((U.filter (fun p => mq O p ω = 1)).card : ℝ)})
              (sideAccOf O B P C ((Finset.univ : Finset (Fin B.m)).image t) ω)}} else ∅)
        (prefixesAt populations B.m x) (poolAt B.M x)
        (fun i : Fin B.m => cert j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.M x), gateBadAcc,
      ← certOf_eq_image j B.m x, ← sideAcc_eq_sideAccOf O populations j B x]
  rw [hrw]
  exact measurableSet_of_run_data_cert populations j B _ hR

open scoped Classical in
lemma measurableSet_gateBadRej (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (εcov c β : ℝ) : MeasurableSet (gateBadRej O populations j B εcov c β) := by
  classical
  have hR : ∀ (P C : Finset S) (t : Fin B.m → S), MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.m)).image t)
        ∧ nz x ∈ {ω : Ω | ω ∈ (fun U : Finset S => {ω : Ω |
            β * (B.m : ℝ) ≤ (U.card : ℝ)
            ∧ c * (U.card : ℝ) ≤ ∑ p ∈ U, O.label p
            ∧ ((U.filter (fun p => mq O p ω = 1)).card : ℝ)
                ≤ (U.card : ℝ) * gateRej O εcov})
          (sideRejOf O B P C ((Finset.univ : Finset (Fin B.m)).image t) ω)}} else ∅) := by
    intro P C t
    split_ifs with hone
    · by_cases hdisj : Disjoint P ((Finset.univ : Finset (Fin B.m)).image t)
      · have hset : {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.m)).image t)
            ∧ nz x ∈ {ω : Ω | ω ∈ (fun U : Finset S => {ω : Ω |
                β * (B.m : ℝ) ≤ (U.card : ℝ)
                ∧ c * (U.card : ℝ) ≤ ∑ p ∈ U, O.label p
                ∧ ((U.filter (fun p => mq O p ω = 1)).card : ℝ)
                    ≤ (U.card : ℝ) * gateRej O εcov})
              (sideRejOf O B P C ((Finset.univ : Finset (Fin B.m)).image t) ω)}}
            = nz ⁻¹' {ω : Ω | ω ∈ (fun U : Finset S => {ω : Ω |
                β * (B.m : ℝ) ≤ (U.card : ℝ)
                ∧ c * (U.card : ℝ) ≤ ∑ p ∈ U, O.label p
                ∧ ((U.filter (fun p => mq O p ω = 1)).card : ℝ)
                    ≤ (U.card : ℝ) * gateRej O εcov})
              (sideRejOf O B P C ((Finset.univ : Finset (Fin B.m)).image t) ω)} := by
          ext x
          simp only [Set.mem_setOf_eq, Set.mem_preimage, hdisj, true_and]
        rw [hset]
        refine measurable_nz (measurableSet_of_sideRejOf O B P C _ hone
          (fun U : Finset S => {ω : Ω |
            β * (B.m : ℝ) ≤ (U.card : ℝ)
            ∧ c * (U.card : ℝ) ≤ ∑ p ∈ U, O.label p
            ∧ ((U.filter (fun p => mq O p ω = 1)).card : ℝ)
                ≤ (U.card : ℝ) * gateRej O εcov}) (fun U => ?_))
        by_cases h1 : β * (B.m : ℝ) ≤ (U.card : ℝ)
        · by_cases h2 : c * (U.card : ℝ) ≤ ∑ p ∈ U, O.label p
          · have hrw : {ω : Ω | β * (B.m : ℝ) ≤ (U.card : ℝ)
                ∧ c * (U.card : ℝ) ≤ ∑ p ∈ U, O.label p
                ∧ ((U.filter (fun p => mq O p ω = 1)).card : ℝ)
                    ≤ (U.card : ℝ) * gateRej O εcov}
                = {ω : Ω | ((U.filter (fun p => mq O p ω = 1)).card : ℝ)
                    ≤ (U.card : ℝ) * gateRej O εcov} := by
              ext ω; simp only [Set.mem_setOf_eq, h1, h2, true_and]
            rw [hrw]
            exact measurableSet_hits_le O U _
          · simpa [h2] using MeasurableSet.empty
        · simpa [h1] using MeasurableSet.empty
      · simpa [hdisj] using MeasurableSet.empty
    · exact MeasurableSet.empty
  have hrw : gateBadRej O populations j B εcov c β
      = {x : Run Ω S J | x ∈ (fun P C t => if (1 : S) ∈ C then
          {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.m)).image t)
            ∧ nz x ∈ {ω : Ω | ω ∈ (fun U : Finset S => {ω : Ω |
                β * (B.m : ℝ) ≤ (U.card : ℝ)
                ∧ c * (U.card : ℝ) ≤ ∑ p ∈ U, O.label p
                ∧ ((U.filter (fun p => mq O p ω = 1)).card : ℝ)
                    ≤ (U.card : ℝ) * gateRej O εcov})
              (sideRejOf O B P C ((Finset.univ : Finset (Fin B.m)).image t) ω)}} else ∅)
        (prefixesAt populations B.m x) (poolAt B.M x)
        (fun i : Fin B.m => cert j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.M x), gateBadRej,
      ← certOf_eq_image j B.m x, ← sideRej_eq_sideRejOf O populations j B x]
  rw [hrw]
  exact measurableSet_of_run_data_cert populations j B _ hR

open scoped Classical in
/-- **The gate does not admit a side that is a `c` fraction wrong.**  `gate_accept_sound` at
the certification sample, which the clustering never read. -/
theorem measureReal_gateBadAcc_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (j : J) (hj : j ∈ populations) (B : Budget)
    (εcov c β : ℝ) (hβ : 0 ≤ β) (hc : 0 ≤ c) (hsig : O.η ≤ 1 / 2)
    (hτ : 0 ≤ gateAcc O εcov - (1 - O.η) + c * (1 - 2 * O.η)) :
    (runLaw μ D Dsf).real (gateBadAcc O populations j B εcov c β)
      ≤ Real.exp (-2 * (β * (B.m : ℝ))
          * (gateAcc O εcov - (1 - O.η) + c * (1 - 2 * O.η)) ^ 2) := by
  classical
  set τ : ℝ := gateAcc O εcov - (1 - O.η) + c * (1 - 2 * O.η) with hτdef
  set E : ℝ := Real.exp (-2 * (β * (B.m : ℝ)) * τ ^ 2) with hEdef
  have hEnn : runLaw μ D Dsf (gateBadAcc O populations j B εcov c β) ≤ ENNReal.ofReal E := by
    refine runLaw_slice_le D Dsf _ (measurableSet_gateBadAcc O populations j B εcov c β) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
      ae_cert_mem_Pre D Dsf Pre populations hsupp] with d hdP hdC
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.m).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.M).image (fun i => d.1.1 i)) with hCd
    set Ad : Finset S := (Finset.range B.m).image (fun i => d.2 (j, i)) with hAd
    have hP : ∀ q ∈ Pd, q ∈ Pre := by
      intro q hq
      obtain ⟨j', hj', hq'⟩ := Finset.mem_biUnion.1 hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hq'
      exact hdP j' hj' i
    have hA : ∀ p ∈ Ad, p ∈ Pre := by
      intro p hp
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hp
      exact hdC j hj i
    by_cases hdisj : Disjoint Pd Ad
    · have hmain := gate_accept_sound O Ad (readSet Pd Cd ∪ readSet Ad (Cd.erase 1))
        (disjoint_gateReads hflat Pd Cd Ad hP hA hdisj)
        (fun ω => sideAcc O populations j B ((ω, d) : Run Ω S J))
        (fun ω => sideAcc_subset O populations j B _)
        (fun ω ω' h => sideAcc_congr O populations j B d h)
        (gateAcc O εcov) c (β * (B.m : ℝ)) (mul_nonneg hβ (Nat.cast_nonneg _)) hc hτ hsig
      have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ gateBadAcc O populations j B εcov c β}
          ⊆ {ω : Ω | β * (B.m : ℝ)
                ≤ ((sideAcc O populations j B ((ω, d) : Run Ω S J)).card : ℝ)
              ∧ c * ((sideAcc O populations j B ((ω, d) : Run Ω S J)).card : ℝ)
                  ≤ (((sideAcc O populations j B ((ω, d) : Run Ω S J)).filter
                    (fun p => O.label p = 0)).card : ℝ)
              ∧ ((sideAcc O populations j B ((ω, d) : Run Ω S J)).card : ℝ) * gateAcc O εcov
                  ≤ (((sideAcc O populations j B ((ω, d) : Run Ω S J)).filter
                    (fun p => mq O p ω = 1)).card : ℝ)} := by
        rintro ω ⟨-, h1, h2, h3⟩
        exact ⟨h1, h2, h3⟩
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
      exact ENNReal.ofReal_le_ofReal hmain
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ gateBadAcc O populations j B εcov c β}
          = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨h1, -⟩
        exact hdisj h1
      simp [hsec]
  rw [measureReal_def]
  calc (runLaw μ D Dsf (gateBadAcc O populations j B εcov c β)).toReal
      ≤ (ENNReal.ofReal E).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = E := ENNReal.toReal_ofReal (Real.exp_nonneg _)

open scoped Classical in
/-- The mirror for the reject side. -/
theorem measureReal_gateBadRej_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (j : J) (hj : j ∈ populations) (B : Budget)
    (εcov c β : ℝ) (hβ : 0 ≤ β) (hc : 0 ≤ c) (hsig : O.η ≤ 1 / 2)
    (hτ : 0 ≤ O.η + c * (1 - 2 * O.η) - gateRej O εcov) :
    (runLaw μ D Dsf).real (gateBadRej O populations j B εcov c β)
      ≤ Real.exp (-2 * (β * (B.m : ℝ))
          * (O.η + c * (1 - 2 * O.η) - gateRej O εcov) ^ 2) := by
  classical
  set τ : ℝ := O.η + c * (1 - 2 * O.η) - gateRej O εcov with hτdef
  set E : ℝ := Real.exp (-2 * (β * (B.m : ℝ)) * τ ^ 2) with hEdef
  have hEnn : runLaw μ D Dsf (gateBadRej O populations j B εcov c β) ≤ ENNReal.ofReal E := by
    refine runLaw_slice_le D Dsf _ (measurableSet_gateBadRej O populations j B εcov c β) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
      ae_cert_mem_Pre D Dsf Pre populations hsupp] with d hdP hdC
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.m).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.M).image (fun i => d.1.1 i)) with hCd
    set Ad : Finset S := (Finset.range B.m).image (fun i => d.2 (j, i)) with hAd
    have hP : ∀ q ∈ Pd, q ∈ Pre := by
      intro q hq
      obtain ⟨j', hj', hq'⟩ := Finset.mem_biUnion.1 hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hq'
      exact hdP j' hj' i
    have hA : ∀ p ∈ Ad, p ∈ Pre := by
      intro p hp
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hp
      exact hdC j hj i
    by_cases hdisj : Disjoint Pd Ad
    · have hmain := gate_reject_sound O Ad (readSet Pd Cd ∪ readSet Ad (Cd.erase 1))
        (disjoint_gateReads hflat Pd Cd Ad hP hA hdisj)
        (fun ω => sideRej O populations j B ((ω, d) : Run Ω S J))
        (fun ω => sideRej_subset O populations j B _)
        (fun ω ω' h => sideRej_congr O populations j B d h)
        (gateRej O εcov) c (β * (B.m : ℝ)) (mul_nonneg hβ (Nat.cast_nonneg _)) hc hτ hsig
      have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ gateBadRej O populations j B εcov c β}
          ⊆ {ω : Ω | β * (B.m : ℝ)
                ≤ ((sideRej O populations j B ((ω, d) : Run Ω S J)).card : ℝ)
              ∧ c * ((sideRej O populations j B ((ω, d) : Run Ω S J)).card : ℝ)
                  ≤ ∑ p ∈ sideRej O populations j B ((ω, d) : Run Ω S J), O.label p
              ∧ (((sideRej O populations j B ((ω, d) : Run Ω S J)).filter
                    (fun p => mq O p ω = 1)).card : ℝ)
                  ≤ ((sideRej O populations j B ((ω, d) : Run Ω S J)).card : ℝ)
                    * gateRej O εcov} := by
        rintro ω ⟨-, h1, h2, h3⟩
        exact ⟨h1, h2, h3⟩
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
      exact ENNReal.ofReal_le_ofReal hmain
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ gateBadRej O populations j B εcov c β}
          = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨h1, -⟩
        exact hdisj h1
      simp [hsec]
  rw [measureReal_def]
  calc (runLaw μ D Dsf (gateBadRej O populations j B εcov c β)).toReal
      ≤ (ENNReal.ofReal E).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = E := ENNReal.toReal_ofReal (Real.exp_nonneg _)

open scoped Classical in
/-- **A cut wrong on the population is wrong on the sample.**  The family is a function of
the noise and the table, so the certification draws are independent of it and plain
Hoeffding applies. -/
theorem measureReal_hitShort_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (populations : Finset J) (j : J) (B : Budget) (εcov t : ℝ) (hε : 0 ≤ εcov) (ht : 0 ≤ t) :
    (runLaw μ D Dsf).real (hitShort O populations (D j) j B εcov t)
      ≤ Real.exp (-2 * (B.m : ℝ) * t ^ 2) := by
  classical
  set E : ℝ := Real.exp (-2 * (B.m : ℝ) * t ^ 2) with hEdef
  have hEnn : runLaw μ D Dsf (hitShort O populations (D j) j B εcov t) ≤ ENNReal.ofReal E := by
    refine runLaw_slice_cert_le D Dsf _
      (measurableSet_hitShort O populations (D j) j B εcov t) _ ?_
    intro y
    set F : Finset S := clusterAt O populations ((y.1, (y.2, fun _ => (1 : S))) : Run Ω S J) B
      with hF
    set W : Set S := {p | ¬ cutCorrect O B.lo B.hi F p y.1} with hW
    have hFeq : ∀ c : J × ℕ → S,
        clusterAt O populations ((y.1, (y.2, c)) : Run Ω S J) B = F := fun c => rfl
    by_cases hmass : εcov ≤ (D j).real W
    · have hsec : {c : J × ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
            ∈ hitShort O populations (D j) j B εcov t}
          ⊆ {c | (((Finset.range B.m).filter (fun i => c (j, i) ∈ W)).card : ℝ)
            ≤ (B.m : ℝ) * (εcov - t)} := by
        rintro c ⟨hinj, -, hcount⟩
        show (((Finset.range B.m).filter (fun i => c (j, i) ∈ W)).card : ℝ)
          ≤ (B.m : ℝ) * (εcov - t)
        calc (((Finset.range B.m).filter (fun i => c (j, i) ∈ W)).card : ℝ)
            = (((certOf j B.m ((y.1, (y.2, c)) : Run Ω S J)).filter
                (fun p => ¬ cutCorrect O B.lo B.hi F p y.1)).card : ℝ) :=
              congrArg (fun n : ℕ => (n : ℝ)) (card_filter_certOf j B.m
                ((y.1, (y.2, c)) : Run Ω S J)
                (fun p => ¬ cutCorrect O B.lo B.hi F p y.1) _ _ hinj)
          _ ≤ (B.m : ℝ) * (εcov - t) := hcount
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top (Measure.infinitePi fun z : J × ℕ => D z.1) _),
        ← measureReal_def]
      refine ENNReal.ofReal_le_ofReal ?_
      rw [hEdef]
      have hcert := cert_hits_wrongSet D j B.m W εcov t hε ht hmass
      convert hcert using 3
      funext c
      congr!
    · have hsec : {c : J × ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
          ∈ hitShort O populations (D j) j B εcov t} = (∅ : Set (J × ℕ → S)) := by
        ext c
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨-, hm, -⟩
        rw [hFeq c] at hm
        exact hmass hm
      simp [hsec]
  rw [measureReal_def]
  calc (runLaw μ D Dsf (hitShort O populations (D j) j B εcov t)).toReal
      ≤ (ENNReal.ofReal E).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = E := ENNReal.toReal_ofReal (Real.exp_nonneg _)

/-- The sample counts draws, the table counts strings: a repeated draw is one string. -/
lemma card_filter_certOf_le (j : J) (m : ℕ) (x : Run Ω S J) (Q : S → Prop)
    (instA : DecidablePred Q) (instB : DecidablePred (fun i : ℕ => Q (cert j i x))) :
    (@Finset.filter _ Q instA (certOf j m x)).card
      ≤ (@Finset.filter _ (fun i : ℕ => Q (cert j i x)) instB (Finset.range m)).card := by
  refine Finset.card_le_card_of_surjOn (fun i => cert j i x) (fun p hp => ?_)
  simp only [Finset.coe_filter, Set.mem_setOf_eq] at hp
  obtain ⟨hpC, hQ⟩ := hp
  unfold certOf at hpC
  obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hpC
  exact ⟨i, by simp only [Finset.coe_filter, Set.mem_setOf_eq, Finset.mem_range,
    Finset.mem_range.1 hi, true_and]; exact hQ, rfl⟩

open scoped Classical in
/-- **The sample is not mostly prefixes the family flips.**  The heavy set is chosen by the
noise and the table; the certification draws are neither, so its hit count is the upper
Hoeffding tail at its mass. -/
noncomputable def heavyHits (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (j : J) (B : Budget) (f q t : ℝ) : Set (Run Ω S J) :=
  {x | Dj.real {p | ¬ (flipCount O ((clusterAt O populations x B).erase 1) p
        ≤ (((clusterAt O populations x B).erase 1).card : ℝ) * f)} ≤ q
    ∧ (B.m : ℝ) * (q + t)
        ≤ (((certOf j B.m x).filter (fun p =>
          ¬ (flipCount O ((clusterAt O populations x B).erase 1) p
            ≤ (((clusterAt O populations x B).erase 1).card : ℝ) * f))).card : ℝ)}

open scoped Classical in
lemma measurableSet_heavyHits (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (j : J) (B : Budget) (f q t : ℝ) :
    MeasurableSet (heavyHits O populations Dj j B f q t) := by
  classical
  have hR : ∀ (P C : Finset S) (tt : Fin B.m → S), MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | Dj.real {p | ¬ (flipCount O
            ((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1) p
          ≤ (((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1).card : ℝ) * f)} ≤ q
        ∧ (B.m : ℝ) * (q + t)
            ≤ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
              ¬ (flipCount O ((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1) p
                ≤ (((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1).card : ℝ) * f))).card
                  : ℝ)} else ∅) := by
    intro P C tt
    split_ifs with hone
    · refine measurable_nz (measurableSet_of_fam (T := C.powerset)
        (fun ω => Finset.mem_powerset.2 (clusterOf_subset O B.cn B.cd B.sc P C B.k ω hone))
        (fun A₀ => measurableSet_clusterOf O B.cn B.cd B.sc P C B.k hone A₀)
        (fun A₀ => {_ω : Ω |
          Dj.real {p | ¬ (flipCount O (A₀.erase 1) p ≤ ((A₀.erase 1).card : ℝ) * f)} ≤ q
          ∧ (B.m : ℝ) * (q + t)
              ≤ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
                ¬ (flipCount O (A₀.erase 1) p ≤ ((A₀.erase 1).card : ℝ) * f))).card : ℝ)})
        (fun A₀ => ?_))
      by_cases hcond : Dj.real {p | ¬ (flipCount O (A₀.erase 1) p
            ≤ ((A₀.erase 1).card : ℝ) * f)} ≤ q
          ∧ (B.m : ℝ) * (q + t)
            ≤ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
              ¬ (flipCount O (A₀.erase 1) p ≤ ((A₀.erase 1).card : ℝ) * f))).card : ℝ)
      · simpa [hcond] using MeasurableSet.univ
      · simpa [hcond] using MeasurableSet.empty
    · exact MeasurableSet.empty
  have hrw : heavyHits O populations Dj j B f q t
      = {x : Run Ω S J | x ∈ (fun P C tt => if (1 : S) ∈ C then
          {x : Run Ω S J | Dj.real {p | ¬ (flipCount O
                ((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1) p
              ≤ (((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1).card : ℝ) * f)} ≤ q
            ∧ (B.m : ℝ) * (q + t)
                ≤ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
                  ¬ (flipCount O ((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1) p
                    ≤ (((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1).card : ℝ)
                      * f))).card : ℝ)} else ∅)
        (prefixesAt populations B.m x) (poolAt B.M x)
        (fun i : Fin B.m => cert j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.M x), heavyHits,
      ← certOf_eq_image j B.m x, ← clusterAt_eq_clusterOf O populations B x]
    tauto
  rw [hrw]
  exact measurableSet_of_run_data_cert populations j B _ hR

open scoped Classical in
/-- The flip-heavy prefixes are a `q` fraction of the population, so the sample sees at most
`m(q + t)` of them. -/
theorem measureReal_heavyHits_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (populations : Finset J) (j : J) (B : Budget) (f q t : ℝ) (ht : 0 ≤ t) :
    (runLaw μ D Dsf).real (heavyHits O populations (D j) j B f q t)
      ≤ Real.exp (-2 * (B.m : ℝ) * t ^ 2) := by
  classical
  have hEnn : runLaw μ D Dsf (heavyHits O populations (D j) j B f q t)
      ≤ ENNReal.ofReal (Real.exp (-2 * (B.m : ℝ) * t ^ 2)) := by
    refine runLaw_slice_cert_le D Dsf _
      (measurableSet_heavyHits O populations (D j) j B f q t) _ ?_
    intro y
    set F : Finset S :=
      (clusterAt O populations ((y.1, (y.2, fun _ => (1 : S))) : Run Ω S J) B).erase 1 with hF
    set W : Set S := {p | ¬ (flipCount O F p ≤ (F.card : ℝ) * f)} with hW
    have hFeq : ∀ c : J × ℕ → S,
        (clusterAt O populations ((y.1, (y.2, c)) : Run Ω S J) B).erase 1 = F := fun c => rfl
    by_cases hmass : (D j).real W ≤ q
    · have hsec : {c : J × ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
            ∈ heavyHits O populations (D j) j B f q t}
          ⊆ {c | (B.m : ℝ) * (q + t)
            ≤ (((Finset.range B.m).filter (fun i => c (j, i) ∈ W)).card : ℝ)} := by
        rintro c ⟨-, hcount⟩
        refine le_trans hcount ?_
        exact_mod_cast card_filter_certOf_le j B.m ((y.1, (y.2, c)) : Run Ω S J)
          (fun p => p ∈ W) _ _
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top (Measure.infinitePi fun z : J × ℕ => D z.1) _),
        ← measureReal_def]
      refine ENNReal.ofReal_le_ofReal ?_
      have hcert := cert_hits_upper D j B.m W q t ht hmass
      convert hcert using 3
      funext c
      congr!
    · have hsec : {c : J × ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
          ∈ heavyHits O populations (D j) j B f q t} = (∅ : Set (J × ℕ → S)) := by
        ext c
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨hm, -⟩
        exact hmass hm
      simp [hsec]
  rw [measureReal_def]
  calc (runLaw μ D Dsf (heavyHits O populations (D j) j B f q t)).toReal
      ≤ (ENNReal.ofReal (Real.exp (-2 * (B.m : ℝ) * t ^ 2))).toReal :=
        ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = Real.exp (-2 * (B.m : ℝ) * t ^ 2) := ENNReal.toReal_ofReal (Real.exp_nonneg _)

lemma measurableSet_decided' (O : Oracle μ S) (lo ha : ℕ) (A₀ : Finset S) (p : S) :
    MeasurableSet[noiseAlg O Set.univ] {ω | ¬ decided O lo ha A₀ p ω} :=
  measurableSet_filter_pred_map O (T := Set.univ) (fun v => p * v)
    (by simp) (fun U => ¬ (ha < Finset.card U ∨ Finset.card U ≤ lo))

open scoped Classical in
lemma measurableSet_indecisionCount (O : Oracle μ S) (lo ha : ℕ) (A₀ A : Finset S) (r : ℝ) :
    MeasurableSet {ω | (((A.filter (fun p => ¬ decided O lo ha A₀ p ω)).card : ℝ) ≤ r)} :=
  noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O
    (fun p ω => ¬ decided O lo ha A₀ p ω)
    (fun p _ => measurableSet_decided' O lo ha A₀ p) (fun U => ((U.card : ℝ) ≤ r)))

open scoped Classical in
lemma measurableSet_binomSfGe_hits (O : Oracle μ S) (U : Finset S) (θ α : ℝ) :
    MeasurableSet {ω | binomSfGe U.card θ ((U.filter (fun p => mq O p ω = 1)).card) ≤ α} :=
  noiseAlg_le O Set.univ _
    (measurableSet_filter_pred O (T := Set.univ) (by simp)
      (fun V => binomSfGe U.card θ V.card ≤ α))

open scoped Classical in
lemma measurableSet_binomCdf_hits (O : Oracle μ S) (U : Finset S) (θ α : ℝ) :
    MeasurableSet {ω | binomCdf U.card θ ((U.filter (fun p => mq O p ω = 1)).card) ≤ α} :=
  noiseAlg_le O Set.univ _
    (measurableSet_filter_pred O (T := Set.univ) (by simp)
      (fun V => binomCdf U.card θ V.card ≤ α))

open scoped Classical in
/-- The gate's verdict is measurable: each side is a fibre of the votes, and the count it
scores is a fibre of the bits at the certification prefixes. -/
lemma measurableSet_admittedFixed (O : Oracle μ S) (lo hi : ℕ) (F A : Finset S)
    (εcov α : ℝ) : MeasurableSet {ω | admitted O lo hi εcov α F A ω} := by
  classical
  have hacc : MeasurableSet {ω : Ω |
      binomSfGe (A.filter (fun p => hi - 1 < voteCount O F p ω)).card (gateAcc O εcov)
        (((A.filter (fun p => hi - 1 < voteCount O F p ω)).filter
          (fun p => mq O p ω = 1)).card) ≤ α} :=
    measurableSet_of_fam (T := A.powerset)
      (fun ω => Finset.mem_powerset.2 (Finset.filter_subset _ _))
      (fun U => measurableSet_sideOf_gt O A F (hi - 1) U)
      (fun U => {ω : Ω | binomSfGe U.card (gateAcc O εcov)
        ((U.filter (fun p => mq O p ω = 1)).card) ≤ α})
      (fun U => measurableSet_binomSfGe_hits O U (gateAcc O εcov) α)
  have hrej : MeasurableSet {ω : Ω |
      binomCdf (A.filter (fun p => voteCount O F p ω ≤ lo)).card (gateRej O εcov)
        (((A.filter (fun p => voteCount O F p ω ≤ lo)).filter
          (fun p => mq O p ω = 1)).card) ≤ α} :=
    measurableSet_of_fam (T := A.powerset)
      (fun ω => Finset.mem_powerset.2 (Finset.filter_subset _ _))
      (fun U => measurableSet_sideOf_le O A F lo U)
      (fun U => {ω : Ω | binomCdf U.card (gateRej O εcov)
        ((U.filter (fun p => mq O p ω = 1)).card) ≤ α})
      (fun U => measurableSet_binomCdf_hits O U (gateRej O εcov) α)
  exact hacc.inter hrej

/-- The family is **usable at `p`**: no member flips it, and the family is neither too
small for the thresholds nor larger than the round allows. -/
def famGood (O : Oracle μ S) (kmin kmax : ℕ) (F : Finset S) (p : S) : Prop :=
  flipCount O F p ≤ (F.card : ℝ) * 0 ∧ kmin ≤ F.card ∧ F.card ≤ kmax

open scoped Classical in
/-- **The draws were good and the state still did not return.**  Everything the state needs
of its draws — the certification prefixes fresh, nonempty, both classes represented, and the
family light on all but an `l` fraction of them — holds, and the round's two tests still fail.
This is the event `ret_at_whp` prices; the rest of the lift charges the draws. -/
noncomputable def retMiss (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (εcov α l : ℝ) (n₀ kmin kmax : ℕ) : Set (Run Ω S J) :=
  {x | Disjoint (prefixesAt populations B.m x) (certOf j B.m x)
    ∧ 0 < (certOf j B.m x).card
    ∧ (n₀ : ℝ) + 2 * l * ((certOf j B.m x).card : ℝ) + 2 * l * ((certOf j B.m x).card : ℝ)
        ≤ (((certOf j B.m x).filter (fun p => O.label p = 1)).card : ℝ)
    ∧ (n₀ : ℝ) + 2 * l * ((certOf j B.m x).card : ℝ) + 2 * l * ((certOf j B.m x).card : ℝ)
        ≤ (((certOf j B.m x).filter (fun p => O.label p = 0)).card : ℝ)
    ∧ (((certOf j B.m x).filter (fun p =>
        ¬ famGood O kmin kmax ((clusterAt O populations x B).erase 1) p)).card : ℝ)
        ≤ l * ((certOf j B.m x).card : ℝ)
    ∧ ¬ ((((certOf j B.m x).filter (fun p => ¬ decided O B.lo (B.hi - 1)
            ((clusterAt O populations x B).erase 1) p (nz x))).card : ℝ)
          ≤ 2 * l * ((certOf j B.m x).card : ℝ)
        ∧ admitted O B.lo B.hi εcov α ((clusterAt O populations x B).erase 1)
            (certOf j B.m x) (nz x))}

open scoped Classical in
lemma measurableSet_retMiss (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (εcov α l : ℝ) (n₀ kmin kmax : ℕ) :
    MeasurableSet (retMiss O populations j B εcov α l n₀ kmin kmax) := by
  classical
  have hR : ∀ (P C : Finset S) (tt : Fin B.m → S), MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.m)).image tt)
        ∧ 0 < ((Finset.univ : Finset (Fin B.m)).image tt).card
        ∧ (n₀ : ℝ) + 2 * l * (((Finset.univ : Finset (Fin B.m)).image tt).card : ℝ)
            + 2 * l * (((Finset.univ : Finset (Fin B.m)).image tt).card : ℝ)
            ≤ ((((Finset.univ : Finset (Fin B.m)).image tt).filter
              (fun p => O.label p = 1)).card : ℝ)
        ∧ (n₀ : ℝ) + 2 * l * (((Finset.univ : Finset (Fin B.m)).image tt).card : ℝ)
            + 2 * l * (((Finset.univ : Finset (Fin B.m)).image tt).card : ℝ)
            ≤ ((((Finset.univ : Finset (Fin B.m)).image tt).filter
              (fun p => O.label p = 0)).card : ℝ)
        ∧ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
            ¬ famGood O kmin kmax
              ((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1) p)).card : ℝ)
            ≤ l * (((Finset.univ : Finset (Fin B.m)).image tt).card : ℝ)
        ∧ ¬ (((((Finset.univ : Finset (Fin B.m)).image tt).filter
                (fun p => ¬ decided O B.lo (B.hi - 1)
                  ((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1) p (nz x))).card : ℝ)
              ≤ 2 * l * (((Finset.univ : Finset (Fin B.m)).image tt).card : ℝ)
            ∧ admitted O B.lo B.hi εcov α
                ((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1)
                ((Finset.univ : Finset (Fin B.m)).image tt) (nz x))} else ∅) := by
    intro P C tt
    split_ifs with hone
    · set A : Finset S := (Finset.univ : Finset (Fin B.m)).image tt with hA
      by_cases hdraw : Disjoint P A ∧ 0 < A.card
          ∧ (n₀ : ℝ) + 2 * l * (A.card : ℝ) + 2 * l * (A.card : ℝ)
              ≤ ((A.filter (fun p => O.label p = 1)).card : ℝ)
          ∧ (n₀ : ℝ) + 2 * l * (A.card : ℝ) + 2 * l * (A.card : ℝ)
              ≤ ((A.filter (fun p => O.label p = 0)).card : ℝ)
      · have hω : MeasurableSet {ω : Ω |
            ((A.filter (fun p => ¬ famGood O kmin kmax
              ((clusterOf O B.cn B.cd B.sc P C B.k ω).erase 1) p)).card : ℝ)
              ≤ l * (A.card : ℝ)
            ∧ ¬ (((A.filter (fun p => ¬ decided O B.lo (B.hi - 1)
                    ((clusterOf O B.cn B.cd B.sc P C B.k ω).erase 1) p ω)).card : ℝ)
                  ≤ 2 * l * (A.card : ℝ)
                ∧ admitted O B.lo B.hi εcov α
                    ((clusterOf O B.cn B.cd B.sc P C B.k ω).erase 1) A ω)} := by
          refine measurableSet_of_fam (T := C.powerset)
            (fun ω => Finset.mem_powerset.2 (clusterOf_subset O B.cn B.cd B.sc P C B.k ω hone))
            (fun A₀ => measurableSet_clusterOf O B.cn B.cd B.sc P C B.k hone A₀)
            (fun A₀ => {ω : Ω |
              ((A.filter (fun p => ¬ famGood O kmin kmax (A₀.erase 1) p)).card : ℝ)
                ≤ l * (A.card : ℝ)
              ∧ ¬ (((A.filter (fun p => ¬ decided O B.lo (B.hi - 1) (A₀.erase 1) p ω)).card : ℝ)
                    ≤ 2 * l * (A.card : ℝ)
                  ∧ admitted O B.lo B.hi εcov α (A₀.erase 1) A ω)})
            (fun A₀ => ?_)
          by_cases hlight : ((A.filter (fun p =>
              ¬ famGood O kmin kmax (A₀.erase 1) p)).card : ℝ) ≤ l * (A.card : ℝ)
          · have hrw : {ω : Ω |
                ((A.filter (fun p => ¬ famGood O kmin kmax (A₀.erase 1) p)).card : ℝ)
                  ≤ l * (A.card : ℝ)
                ∧ ¬ (((A.filter (fun p =>
                      ¬ decided O B.lo (B.hi - 1) (A₀.erase 1) p ω)).card : ℝ)
                      ≤ 2 * l * (A.card : ℝ)
                    ∧ admitted O B.lo B.hi εcov α (A₀.erase 1) A ω)}
                = ({ω : Ω | ((A.filter (fun p =>
                      ¬ decided O B.lo (B.hi - 1) (A₀.erase 1) p ω)).card : ℝ)
                        ≤ 2 * l * (A.card : ℝ)}
                  ∩ {ω : Ω | admitted O B.lo B.hi εcov α (A₀.erase 1) A ω})ᶜ := by
              ext ω
              simp only [Set.mem_setOf_eq, Set.mem_compl_iff, Set.mem_inter_iff, hlight, true_and]
            rw [hrw]
            exact ((measurableSet_indecisionCount O B.lo (B.hi - 1) (A₀.erase 1) A
              (2 * l * (A.card : ℝ))).inter
                (measurableSet_admittedFixed O B.lo B.hi (A₀.erase 1) A εcov α)).compl
          · have hz : {ω : Ω |
                ((A.filter (fun p => ¬ famGood O kmin kmax (A₀.erase 1) p)).card : ℝ)
                  ≤ l * (A.card : ℝ)
                ∧ ¬ (((A.filter (fun p =>
                      ¬ decided O B.lo (B.hi - 1) (A₀.erase 1) p ω)).card : ℝ)
                      ≤ 2 * l * (A.card : ℝ)
                    ∧ admitted O B.lo B.hi εcov α (A₀.erase 1) A ω)} = (∅ : Set Ω) := by
              ext ω
              simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
              rintro ⟨h, -⟩
              exact hlight h
            rw [hz]
            exact MeasurableSet.empty
        have hset : {x : Run Ω S J | Disjoint P A ∧ 0 < A.card
            ∧ (n₀ : ℝ) + 2 * l * (A.card : ℝ) + 2 * l * (A.card : ℝ)
                ≤ ((A.filter (fun p => O.label p = 1)).card : ℝ)
            ∧ (n₀ : ℝ) + 2 * l * (A.card : ℝ) + 2 * l * (A.card : ℝ)
                ≤ ((A.filter (fun p => O.label p = 0)).card : ℝ)
            ∧ ((A.filter (fun p => ¬ famGood O kmin kmax
                ((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1) p)).card : ℝ)
                  ≤ l * (A.card : ℝ)
            ∧ ¬ (((A.filter (fun p => ¬ decided O B.lo (B.hi - 1)
                    ((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1) p (nz x))).card : ℝ)
                  ≤ 2 * l * (A.card : ℝ)
                ∧ admitted O B.lo B.hi εcov α
                    ((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1) A (nz x))}
            = nz ⁻¹' {ω : Ω |
              ((A.filter (fun p => ¬ famGood O kmin kmax
                ((clusterOf O B.cn B.cd B.sc P C B.k ω).erase 1) p)).card : ℝ)
                ≤ l * (A.card : ℝ)
              ∧ ¬ (((A.filter (fun p => ¬ decided O B.lo (B.hi - 1)
                      ((clusterOf O B.cn B.cd B.sc P C B.k ω).erase 1) p ω)).card : ℝ)
                    ≤ 2 * l * (A.card : ℝ)
                  ∧ admitted O B.lo B.hi εcov α
                      ((clusterOf O B.cn B.cd B.sc P C B.k ω).erase 1) A ω)} := by
          ext x
          simp only [Set.mem_setOf_eq, Set.mem_preimage, hdraw.1, hdraw.2.1, hdraw.2.2.1,
            hdraw.2.2.2, true_and]
        rw [hset]
        exact measurable_nz hω
      · have hempty : {x : Run Ω S J | Disjoint P A ∧ 0 < A.card
            ∧ (n₀ : ℝ) + 2 * l * (A.card : ℝ) + 2 * l * (A.card : ℝ)
                ≤ ((A.filter (fun p => O.label p = 1)).card : ℝ)
            ∧ (n₀ : ℝ) + 2 * l * (A.card : ℝ) + 2 * l * (A.card : ℝ)
                ≤ ((A.filter (fun p => O.label p = 0)).card : ℝ)
            ∧ ((A.filter (fun p => ¬ famGood O kmin kmax
                ((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1) p)).card : ℝ)
                  ≤ l * (A.card : ℝ)
            ∧ ¬ (((A.filter (fun p => ¬ decided O B.lo (B.hi - 1)
                    ((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1) p (nz x))).card : ℝ)
                  ≤ 2 * l * (A.card : ℝ)
                ∧ admitted O B.lo B.hi εcov α
                    ((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1) A (nz x))}
            = (∅ : Set (Run Ω S J)) := by
          ext x
          simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
          rintro ⟨h1, h2, h3, h4, -⟩
          exact hdraw ⟨h1, h2, h3, h4⟩
        rw [hempty]
        exact MeasurableSet.empty
    · exact MeasurableSet.empty
  have hrw : retMiss O populations j B εcov α l n₀ kmin kmax
      = {x : Run Ω S J | x ∈ (fun P C tt => if (1 : S) ∈ C then
          {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.m)).image tt)
            ∧ 0 < ((Finset.univ : Finset (Fin B.m)).image tt).card
            ∧ (n₀ : ℝ) + 2 * l * (((Finset.univ : Finset (Fin B.m)).image tt).card : ℝ)
                + 2 * l * (((Finset.univ : Finset (Fin B.m)).image tt).card : ℝ)
                ≤ ((((Finset.univ : Finset (Fin B.m)).image tt).filter
                  (fun p => O.label p = 1)).card : ℝ)
            ∧ (n₀ : ℝ) + 2 * l * (((Finset.univ : Finset (Fin B.m)).image tt).card : ℝ)
                + 2 * l * (((Finset.univ : Finset (Fin B.m)).image tt).card : ℝ)
                ≤ ((((Finset.univ : Finset (Fin B.m)).image tt).filter
                  (fun p => O.label p = 0)).card : ℝ)
            ∧ ((((Finset.univ : Finset (Fin B.m)).image tt).filter (fun p =>
                ¬ famGood O kmin kmax
                  ((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1) p)).card : ℝ)
                ≤ l * (((Finset.univ : Finset (Fin B.m)).image tt).card : ℝ)
            ∧ ¬ (((((Finset.univ : Finset (Fin B.m)).image tt).filter
                    (fun p => ¬ decided O B.lo (B.hi - 1)
                      ((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1) p (nz x))).card : ℝ)
                  ≤ 2 * l * (((Finset.univ : Finset (Fin B.m)).image tt).card : ℝ)
                ∧ admitted O B.lo B.hi εcov α
                    ((clusterOf O B.cn B.cd B.sc P C B.k (nz x)).erase 1)
                    ((Finset.univ : Finset (Fin B.m)).image tt) (nz x))} else ∅)
        (prefixesAt populations B.m x) (poolAt B.M x)
        (fun i : Fin B.m => cert j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.M x), retMiss,
      ← certOf_eq_image j B.m x, ← clusterAt_eq_clusterOf O populations B x]
    tauto
  rw [hrw]
  exact measurableSet_of_run_data_cert populations j B _ hR

open scoped Classical in
lemma measurableSet_clusterOf_erase (O : Oracle μ S) (cn cd sc : ℕ) (P cands : Finset S)
    (k : ℕ) (hone : (1 : S) ∈ cands) (A₀ : Finset S) :
    MeasurableSet {ω | (clusterOf O cn cd sc P cands k ω).erase 1 = A₀} := by
  classical
  have hrw : {ω | (clusterOf O cn cd sc P cands k ω).erase 1 = A₀}
      = {ω | ω ∈ (fun A₁ : Finset S =>
        if A₁.erase 1 = A₀ then (Set.univ : Set Ω) else ∅) (clusterOf O cn cd sc P cands k ω)} := by
    ext ω
    by_cases h : (clusterOf O cn cd sc P cands k ω).erase 1 = A₀ <;> simp [h]
  rw [hrw]
  refine measurableSet_of_fam (T := cands.powerset)
    (fun ω => Finset.mem_powerset.2 (clusterOf_subset O cn cd sc P cands k ω hone))
    (fun A₁ => measurableSet_clusterOf O cn cd sc P cands k hone A₁)
    (fun A₁ => if A₁.erase 1 = A₀ then Set.univ else ∅) (fun A₁ => ?_)
  split_ifs
  · exact MeasurableSet.univ
  · exact MeasurableSet.empty

open scoped Classical in
/-- **The lift of `ret_at_whp`.**  At every table the state's own round fails only as often
as the two fractional counts and the two gate tails allow; the draws are charged elsewhere. -/
theorem measureReal_retMiss_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (j : J) (hj : j ∈ populations) (B : Budget)
    (εcov α τ l E : ℝ) (n₀ kmin kmax : ℕ)
    (hE : 0 ≤ E) (hl : 0 < l) (hτ : 0 ≤ τ) (hε0 : 0 ≤ εcov) (hε1 : εcov ≤ 1)
    (hsig : O.η ≤ 1 / 2)
    (hdec : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * 0 →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ decided O B.lo (B.hi - 1) F p ω} ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * 0 →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo (B.hi - 1) F p ω} ≤ E)
    (hga : ∀ n c : ℕ, n₀ ≤ n → n ≤ c → c ≤ B.m →
      (n : ℝ) * (gateAcc O εcov + τ + τ)
        ≤ (n : ℝ) * (1 - O.η) - (1 - 2 * O.η) * (2 * l * (c : ℝ)))
    (hgr : ∀ n c : ℕ, n₀ ≤ n → n ≤ c → c ≤ B.m →
      (n : ℝ) * O.η + (1 - 2 * O.η) * (2 * l * (c : ℝ))
        ≤ (n : ℝ) * (gateRej O εcov - τ - τ))
    (hα : Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) ≤ α) :
    (runLaw μ D Dsf).real (retMiss O populations j B εcov α l n₀ kmin kmax)
      ≤ E / l + (E / l + 2 * Real.exp (-2 * (n₀ : ℝ) * τ ^ 2)) := by
  classical
  set R : ℝ := E / l + (E / l + 2 * Real.exp (-2 * (n₀ : ℝ) * τ ^ 2)) with hR
  have hR0 : 0 ≤ R := by
    rw [hR]
    have : (0 : ℝ) ≤ E / l := div_nonneg hE hl.le
    have h2 : (0 : ℝ) ≤ Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := Real.exp_nonneg _
    linarith
  have hEnn : runLaw μ D Dsf (retMiss O populations j B εcov α l n₀ kmin kmax)
      ≤ ENNReal.ofReal R := by
    refine runLaw_slice_le D Dsf _
      (measurableSet_retMiss O populations j B εcov α l n₀ kmin kmax) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
      ae_cert_mem_Pre D Dsf Pre populations hsupp] with d hdP hdC
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.m).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.M).image (fun i => d.1.1 i)) with hCd
    set Ad : Finset S := (Finset.range B.m).image (fun i => d.2 (j, i)) with hAd
    have hP : ∀ q ∈ Pd, q ∈ Pre := by
      intro q hq
      obtain ⟨j', hj', hq'⟩ := Finset.mem_biUnion.1 hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hq'
      exact hdP j' hj' i
    have hA : ∀ p ∈ Ad, p ∈ Pre := by
      intro p hp
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hp
      exact hdC j hj i
    have hAcard : Ad.card ≤ B.m := by
      rw [hAd]
      exact le_trans Finset.card_image_le (by simp)
    by_cases hdraw : Disjoint Pd Ad ∧ 0 < Ad.card
        ∧ (n₀ : ℝ) + 2 * l * (Ad.card : ℝ) + 2 * l * (Ad.card : ℝ)
            ≤ ((Ad.filter (fun p => O.label p = 1)).card : ℝ)
        ∧ (n₀ : ℝ) + 2 * l * (Ad.card : ℝ) + 2 * l * (Ad.card : ℝ)
            ≤ ((Ad.filter (fun p => O.label p = 0)).card : ℝ)
    · obtain ⟨hdisj, hCpos, hclassA, hclassR⟩ := hdraw
      set fam : Ω → Finset S :=
        fun ω => (clusterAt O populations ((ω, d) : Run Ω S J) B).erase 1 with hfamdef
      set T : Finset (Finset S) := Cd.powerset with hT
      set good : S → Finset (Finset S) := fun p => T.filter (fun t =>
        flipCount O t p ≤ (t.card : ℝ) * 0 ∧ kmin ≤ t.card ∧ t.card ≤ kmax) with hgood
      have hfamT : ∀ ω, fam ω ∈ T := fun ω => Finset.mem_powerset.2
        (fun v hv => clusterAt_subset O populations B _ (Finset.mem_erase.1 hv).2)
      have hmain := ret_at_whp hflat O Pd Cd Ad hP hA hdisj
        (readSet Pd Cd ∪ readSet Ad (Cd.erase 1)) Finset.subset_union_left
        (disjoint_gateReads hflat Pd Cd Ad hP hA hdisj)
        B.lo B.hi εcov α τ l n₀ T good ∅ (Finset.mem_powerset.2 (Finset.empty_subset _))
        (fun t ht => Finset.mem_powerset.1 ht) fam hfamT
        (fun A₀ => measurableSet_clusterOf_erase O B.cn B.cd B.sc Pd Cd B.k
          (Finset.mem_insert_self 1 _) A₀)
        (fun ω ω' h => congrArg (fun t : Finset S => t.erase 1)
          (clusterAt_congr O populations B d h))
        (fun ω p hp v hv => Finset.mem_union_right _ (mem_readSet hp
          (Finset.mem_erase.2 ⟨(Finset.mem_erase.1 hv).1,
            clusterAt_subset O populations B _ (Finset.mem_erase.1 hv).2⟩)))
        E hE hl hCpos hτ hε0 hε1 hsig
        (fun p _ A₀ _ hgp => hdec A₀ p (Finset.mem_filter.1 hgp).2.1
          (Finset.mem_filter.1 hgp).2.2.1 (Finset.mem_filter.1 hgp).2.2.2)
        (fun p _ A₀ _ hgp => hcut A₀ p (Finset.mem_filter.1 hgp).2.1
          (Finset.mem_filter.1 hgp).2.2.1 (Finset.mem_filter.1 hgp).2.2.2)
        (fun n hn hnc => hga n Ad.card hn hnc hAcard)
        (fun n hn hnc => hgr n Ad.card hn hnc hAcard)
        hα hclassA hclassR
      have hsec : {ω : Ω | ((ω, d) : Run Ω S J)
            ∈ retMiss O populations j B εcov α l n₀ kmin kmax}
          ⊆ {ω : Ω | ((Ad.filter (fun p => fam ω ∉ good p)).card : ℝ) ≤ l * (Ad.card : ℝ)
            ∧ ¬ ((((Ad.filter (fun p =>
                    ¬ decided O B.lo (B.hi - 1) (fam ω) p ω)).card : ℝ)
                  ≤ 2 * l * (Ad.card : ℝ))
              ∧ admitted O B.lo B.hi εcov α (fam ω) Ad ω)} := by
        rintro ω ⟨-, -, -, -, hlight, hbad⟩
        refine ⟨le_trans (le_of_eq ?_) hlight, hbad⟩
        refine congrArg (fun t : Finset S => (t.card : ℝ)) (Finset.filter_congr ?_)
        intro p _
        simp only [hgood, Finset.mem_filter, hfamT ω, true_and]
        exact Iff.rfl
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
      exact ENNReal.ofReal_le_ofReal hmain
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J)
          ∈ retMiss O populations j B εcov α l n₀ kmin kmax} = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨h1, h2, h3, h4, -⟩
        exact hdraw ⟨h1, h2, h3, h4⟩
      simp [hsec]
  rw [measureReal_def]
  calc (runLaw μ D Dsf (retMiss O populations j B εcov α l n₀ kmin kmax)).toReal
      ≤ (ENNReal.ofReal R).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = R := ENNReal.toReal_ofReal hR0

open scoped Classical in
/-- The round's own test at one population: the FNR count and the gate. -/
noncomputable def retAt (O : Oracle μ S) (populations : Finset J)
    (indecisionLimit εcov α : ℝ) (B : Budget) (j : J) : Set (Run Ω S J) :=
  {x | (((certOf j B.m x).filter (fun p => ¬ decided O B.lo (B.hi - 1)
          ((clusterAt O populations x B).erase 1) p (nz x))).card : ℝ)
        ≤ indecisionLimit * ((certOf j B.m x).card : ℝ)
    ∧ admitted O B.lo B.hi εcov α ((clusterAt O populations x B).erase 1)
        (certOf j B.m x) (nz x)}

lemma mem_ret_iff (O : Oracle μ S) (populations : Finset J) (indecisionLimit εcov α : ℝ)
    (B : Budget) (x : Run Ω S J) :
    x ∈ ret O populations indecisionLimit εcov α B
      ↔ ∀ j ∈ populations, x ∈ retAt O populations indecisionLimit εcov α B j := by
  constructor
  · rintro ⟨h1, h2⟩ j hj
    exact ⟨h1 j hj, h2 j hj⟩
  · intro h
    exact ⟨fun j hj => (h j hj).1, fun j hj => (h j hj).2⟩

open scoped Classical in
/-- The runs where the clustering stalls on the seed, or overshoots the round's size.  The
gate refuses a stalled family outright (`not_ret_of_seed_family`), so this is the liveness
half's obligation, not the round's. -/
noncomputable def stalled (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (kmin kmax : ℕ) : Set (Run Ω S J) :=
  {x | ¬ (kmin ≤ ((clusterAt O populations x B).erase 1).card
      ∧ ((clusterAt O populations x B).erase 1).card ≤ kmax)}

open scoped Classical in
/-- **A pool of `k` screened candidates is a family of `k`.**  The seed is one of them and
the ranking keeps it, so the family the gate sees has `k − 1` members besides the seed. -/
lemma stalled_subset (O : Oracle μ S) (populations : Finset J) (B : Budget)
    (hcd : B.cn < B.cd) (hkpos : 0 < B.k) :
    stalled O populations B (B.k - 1) (B.k - 1)
      ⊆ {x : Run Ω S J | ¬ (B.k ≤ (screenedAt O populations B x).card)} := by
  intro x hx
  by_contra hk
  simp only [Set.mem_setOf_eq, Classical.not_not] at hk
  have hcard : (clusterAt O populations x B).card = B.k :=
    clusterAround_card O hcd (prefixesAt populations B.m x) (screenedAt O populations B x)
      (nz x) B.k (one_mem_screenedAt O populations B x) hk hkpos
  have hone : (1 : S) ∈ clusterAt O populations x B :=
    one_mem_clusterAround O B.cn B.cd _ _ (nz x) B.k
  have herase : ((clusterAt O populations x B).erase 1).card = B.k - 1 := by
    rw [Finset.card_erase_of_mem hone, hcard]
  exact hx ⟨le_of_eq herase.symm, le_of_eq herase⟩

open scoped Classical in
/-- The liveness half's obligation, discharged: the clustering has a family of the round's
own size except on the events `measureReal_smallScreen_le` prices. -/
theorem measureReal_stalled_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (B : Budget) (hcd : B.cn < B.cd)
    (hkpos : 0 < B.k) (j₀ : J) (hj₀ : j₀ ∈ populations)
    (γ pAP t ρsf ρ : ℝ) (hγ : 0 ≤ γ) (hpAP0 : 0 ≤ pAP) (ht : 0 ≤ t)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hsc : ∀ n : ℕ, n ≤ populations.card * B.m →
      (n : ℝ) * (2 * O.η * (1 - O.η) + γ) ≤ (B.sc : ℝ))
    (hcount : (B.k : ℝ) ≤ (B.M : ℝ) * (pAP - t))
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hρj : collisionMass (D j₀) ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runLaw μ D Dsf).real (stalled O populations B (B.k - 1) (B.k - 1))
      ≤ (B.M : ℝ) ^ 2 * ρsf + (Real.exp (-2 * (B.M : ℝ) * t ^ 2)
        + ((B.m : ℝ) ^ 2 * ρ + ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γ ^ 2))) :=
  le_trans (measureReal_mono (stalled_subset O populations B hcd hkpos) (measure_ne_top _ _))
    (measureReal_smallScreen_le hflat O populations D Dsf hsupp B hcd j₀ hj₀ γ pAP t ρsf ρ
      hγ hpAP0 ht hpAPBound hsc hcount hρsf hρsf0 hρj hρ0)

open scoped Classical in
/-- **The round returns at one population.**  Everything outside `retMiss` is a fact about
the draws: the certification prefixes repeat, or meet the table, or under-represent a class,
or the family is dirty and the sample sees it.  The cluster's own size is the liveness
half's business and is carried as `Estall`. -/
theorem measureReal_notRetAt_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (j : J) (hj : j ∈ populations) (B : Budget)
    (hmpos : 0 < B.m) (hsig : O.η ≤ 1 / 2)
    (εcov α τ l E Δp ρ tcls qacc qrej th : ℝ) (n₀ kmin kmax : ℕ)
    (hE : 0 ≤ E) (hl : 0 < l) (hτ : 0 ≤ τ) (hε0 : 0 ≤ εcov) (hε1 : εcov ≤ 1)
    (hρ : ∀ j' ∈ populations, collisionMass (D j') ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hΔp : 0 ≤ Δp) (hth : 0 ≤ th) (htcls : 0 ≤ tcls)
    (hqacc : qacc ≤ (D j).real {p | O.label p = 1})
    (hqrej : qrej ≤ (D j).real {p | O.label p = 0})
    (hqacc0 : 0 ≤ qacc) (hqrej0 : 0 ≤ qrej)
    (hclsnum : (n₀ : ℝ) + 2 * l * (B.m : ℝ) + 2 * l * (B.m : ℝ)
      ≤ (B.m : ℝ) * (min qacc qrej - tcls))
    (hheavy : Δp * (kmax : ℝ) + th ≤ l)
    (Estall : ℝ) (hstall : (runLaw μ D Dsf).real (stalled O populations B kmin kmax) ≤ Estall)
    (Edirty : ℝ) (hdirty : (runLaw μ D Dsf).real
      {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O populations x B, flipMass O (D j) v ≤ Δp}
        ≤ Edirty)
    (hdec : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * 0 →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ decided O B.lo (B.hi - 1) F p ω} ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * 0 →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo (B.hi - 1) F p ω} ≤ E)
    (hga : ∀ n c : ℕ, n₀ ≤ n → n ≤ c → c ≤ B.m →
      (n : ℝ) * (gateAcc O εcov + τ + τ)
        ≤ (n : ℝ) * (1 - O.η) - (1 - 2 * O.η) * (2 * l * (c : ℝ)))
    (hgr : ∀ n c : ℕ, n₀ ≤ n → n ≤ c → c ≤ B.m →
      (n : ℝ) * O.η + (1 - 2 * O.η) * (2 * l * (c : ℝ))
        ≤ (n : ℝ) * (gateRej O εcov - τ - τ))
    (hα : Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) ≤ α) :
    (runLaw μ D Dsf).real
        {x : Run Ω S J | x ∉ retAt O populations (2 * l) εcov α B j}
      ≤ ((populations.card : ℝ) + 1) * (B.m : ℝ) ^ 2 * ρ
        + (2 * Real.exp (-2 * (B.m : ℝ) * tcls ^ 2)
          + (Estall + (Edirty + (Real.exp (-2 * (B.m : ℝ) * th ^ 2)
            + (E / l + (E / l + 2 * Real.exp (-2 * (n₀ : ℝ) * τ ^ 2))))))) := by
  classical
  set E1 : Set (Run Ω S J) :=
    {x | ¬ Function.Injective (fun i : Fin B.m => cert j i.val x)} with hE1
  set E2 : Set (Run Ω S J) :=
    {x | ¬ Disjoint (prefixesAt populations B.m x) (certOf j B.m x)} with hE2
  set E3 : Set (Run Ω S J) := {x | (((Finset.range B.m).filter
    (fun i => O.label (cert j i x) = 1)).card : ℝ) ≤ (B.m : ℝ) * (qacc - tcls)} with hE3
  set E4 : Set (Run Ω S J) := {x | (((Finset.range B.m).filter
    (fun i => O.label (cert j i x) = 0)).card : ℝ) ≤ (B.m : ℝ) * (qrej - tcls)} with hE4
  set E5 : Set (Run Ω S J) := stalled O populations B kmin kmax with hE5
  set E6 : Set (Run Ω S J) :=
    {x | ¬ ∀ v ∈ clusterAt O populations x B, flipMass O (D j) v ≤ Δp} with hE6
  set E7 : Set (Run Ω S J) := heavyHits O populations (D j) j B 0 (Δp * (kmax : ℝ)) th with hE7
  set E8 : Set (Run Ω S J) := retMiss O populations j B εcov α l n₀ kmin kmax with hE8
  have hsub : {x : Run Ω S J | x ∉ retAt O populations (2 * l) εcov α B j}
      ⊆ (E1 ∪ E2) ∪ ((E3 ∪ E4) ∪ (E5 ∪ (E6 ∪ (E7 ∪ E8)))) := by
    intro x hx
    by_cases h1 : Function.Injective (fun i : Fin B.m => cert j i.val x)
    · by_cases h2 : Disjoint (prefixesAt populations B.m x) (certOf j B.m x)
      · by_cases h3 : (((Finset.range B.m).filter
            (fun i => O.label (cert j i x) = 1)).card : ℝ) ≤ (B.m : ℝ) * (qacc - tcls)
        · exact Or.inr (Or.inl (Or.inl h3))
        · by_cases h4 : (((Finset.range B.m).filter
              (fun i => O.label (cert j i x) = 0)).card : ℝ) ≤ (B.m : ℝ) * (qrej - tcls)
          · exact Or.inr (Or.inl (Or.inr h4))
          · by_cases h5 : kmin ≤ ((clusterAt O populations x B).erase 1).card
                ∧ ((clusterAt O populations x B).erase 1).card ≤ kmax
            · by_cases h6 : ∀ v ∈ clusterAt O populations x B, flipMass O (D j) v ≤ Δp
              · -- the certification sample is good and the family is clean
                have hcard : ((certOf j B.m x).card : ℝ) = (B.m : ℝ) := by
                  have hinjOn : Set.InjOn (fun i => cert j i x) ↑(Finset.range B.m) := by
                    intro a ha b hb hab
                    have := h1 (show (fun i : Fin B.m => cert j i.val x)
                        ⟨a, Finset.mem_range.1 (by simpa using ha)⟩
                      = (fun i : Fin B.m => cert j i.val x)
                        ⟨b, Finset.mem_range.1 (by simpa using hb)⟩ from hab)
                    simpa using congrArg Fin.val this
                  unfold certOf
                  rw [Finset.card_image_of_injOn hinjOn, Finset.card_range]
                have hCpos : 0 < (certOf j B.m x).card := by
                  have : (0 : ℝ) < ((certOf j B.m x).card : ℝ) := by
                    rw [hcard]; exact_mod_cast hmpos
                  exact_mod_cast this
                have hclassA : (n₀ : ℝ) + 2 * l * ((certOf j B.m x).card : ℝ)
                    + 2 * l * ((certOf j B.m x).card : ℝ)
                    ≤ (((certOf j B.m x).filter (fun p => O.label p = 1)).card : ℝ) := by
                  rw [hcard]
                  have hmin : (B.m : ℝ) * (min qacc qrej - tcls) ≤ (B.m : ℝ) * (qacc - tcls) := by
                    have : min qacc qrej ≤ qacc := min_le_left _ _
                    nlinarith [Nat.cast_nonneg (α := ℝ) B.m]
                  have : (B.m : ℝ) * (qacc - tcls)
                      < (((Finset.range B.m).filter
                        (fun i => O.label (cert j i x) = 1)).card : ℝ) := not_le.1 h3
                  rw [show (((certOf j B.m x).filter (fun p => O.label p = 1)).card : ℝ)
                      = (((Finset.range B.m).filter
                        (fun i => O.label (cert j i x) = 1)).card : ℝ) from
                    congrArg (fun n : ℕ => (n : ℝ))
                      (card_filter_certOf j B.m x (fun p => O.label p = 1) _ _ h1).symm]
                  linarith
                have hclassR : (n₀ : ℝ) + 2 * l * ((certOf j B.m x).card : ℝ)
                    + 2 * l * ((certOf j B.m x).card : ℝ)
                    ≤ (((certOf j B.m x).filter (fun p => O.label p = 0)).card : ℝ) := by
                  rw [hcard]
                  have hmin : (B.m : ℝ) * (min qacc qrej - tcls) ≤ (B.m : ℝ) * (qrej - tcls) := by
                    have : min qacc qrej ≤ qrej := min_le_right _ _
                    nlinarith [Nat.cast_nonneg (α := ℝ) B.m]
                  have : (B.m : ℝ) * (qrej - tcls)
                      < (((Finset.range B.m).filter
                        (fun i => O.label (cert j i x) = 0)).card : ℝ) := not_le.1 h4
                  rw [show (((certOf j B.m x).filter (fun p => O.label p = 0)).card : ℝ)
                      = (((Finset.range B.m).filter
                        (fun i => O.label (cert j i x) = 0)).card : ℝ) from
                    congrArg (fun n : ℕ => (n : ℝ))
                      (card_filter_certOf j B.m x (fun p => O.label p = 0) _ _ h1).symm]
                  linarith
                have hmass : (D j).real {p | ¬ (flipCount O
                      ((clusterAt O populations x B).erase 1) p
                    ≤ ((((clusterAt O populations x B).erase 1).card : ℝ)) * 0)}
                    ≤ Δp * (kmax : ℝ) := by
                  set F : Finset S := (clusterAt O populations x B).erase 1 with hFdef
                  rcases Nat.eq_zero_or_pos F.card with hF0 | hFpos
                  · have hz : {p : S | ¬ (flipCount O F p ≤ ((F.card : ℝ)) * 0)}
                        = (∅ : Set S) := by
                      ext p
                      simp only [Set.mem_empty_iff_false, iff_false, not_not]
                      have : F = ∅ := Finset.card_eq_zero.1 hF0
                      simp [flipCount, this]
                    rw [hz]
                    simp only [measureReal_empty]
                    positivity
                  · have hsetle : {p : S | ¬ (flipCount O F p ≤ ((F.card : ℝ)) * 0)}
                        ⊆ {p | (1 / (F.card : ℝ)) * (F.card : ℝ) ≤ flipCount O F p} := by
                      intro p hp
                      have hFc : (0 : ℝ) < (F.card : ℝ) := by exact_mod_cast hFpos
                      have : (0 : ℝ) < flipCount O F p := by
                        have := not_le.1 hp
                        simpa using this
                      have hone : (1 / (F.card : ℝ)) * (F.card : ℝ) = 1 := by
                        field_simp
                      rw [Set.mem_setOf_eq, hone]
                      unfold flipCount at this ⊢
                      exact_mod_cast this
                    have hFc : (0 : ℝ) < (F.card : ℝ) := by exact_mod_cast hFpos
                    refine le_trans (measureReal_mono hsetle (measure_ne_top _ _)) ?_
                    refine le_trans (flipCount_mass_le O (D j) F Δp (1 / (F.card : ℝ))
                      (by positivity) hFpos (fun v hv => h6 v (Finset.mem_erase.1 hv).2)) ?_
                    have hdiv : Δp / (1 / (F.card : ℝ)) = Δp * (F.card : ℝ) := by
                      field_simp
                    rw [hdiv]
                    have hkm : (F.card : ℝ) ≤ (kmax : ℝ) := by exact_mod_cast h5.2
                    nlinarith
                by_cases h7 : (B.m : ℝ) * (Δp * (kmax : ℝ) + th)
                    ≤ (((certOf j B.m x).filter (fun p =>
                      ¬ (flipCount O ((clusterAt O populations x B).erase 1) p
                        ≤ ((((clusterAt O populations x B).erase 1).card : ℝ)) * 0))).card : ℝ)
                · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨hmass, h7⟩))))
                · refine Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨h2, hCpos, hclassA, hclassR,
                    ?_, ?_⟩))))
                  · have hlt := not_le.1 h7
                    have hle : (((certOf j B.m x).filter (fun p =>
                        ¬ famGood O kmin kmax ((clusterAt O populations x B).erase 1) p)).card : ℝ)
                        = (((certOf j B.m x).filter (fun p =>
                          ¬ (flipCount O ((clusterAt O populations x B).erase 1) p
                            ≤ ((((clusterAt O populations x B).erase 1).card : ℝ))
                              * 0))).card : ℝ) := by
                      refine congrArg (fun t : Finset S => (t.card : ℝ))
                        (Finset.filter_congr (fun p _ => ?_))
                      unfold famGood
                      simp only [h5.1, h5.2, and_true, true_and]
                    rw [hle, hcard]
                    have hm0 : (0 : ℝ) ≤ (B.m : ℝ) := Nat.cast_nonneg _
                    nlinarith
                  · intro hgood
                    exact hx hgood
              · exact Or.inr (Or.inr (Or.inr (Or.inl h6)))
            · exact Or.inr (Or.inr (Or.inl h5))
      · exact Or.inl (Or.inr h2)
    · exact Or.inl (Or.inl h1)
  calc (runLaw μ D Dsf).real {x : Run Ω S J | x ∉ retAt O populations (2 * l) εcov α B j}
      ≤ (runLaw μ D Dsf).real ((E1 ∪ E2) ∪ ((E3 ∪ E4) ∪ (E5 ∪ (E6 ∪ (E7 ∪ E8))))) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ((runLaw μ D Dsf).real E1 + (runLaw μ D Dsf).real E2)
        + (((runLaw μ D Dsf).real E3 + (runLaw μ D Dsf).real E4)
          + ((runLaw μ D Dsf).real E5 + ((runLaw μ D Dsf).real E6
            + ((runLaw μ D Dsf).real E7 + (runLaw μ D Dsf).real E8)))) := by
        have h12 := measureReal_union_le (μ := runLaw μ D Dsf) E1 E2
        have h34 := measureReal_union_le (μ := runLaw μ D Dsf) E3 E4
        have h78 := measureReal_union_le (μ := runLaw μ D Dsf) E7 E8
        have h678 := measureReal_union_le (μ := runLaw μ D Dsf) E6 (E7 ∪ E8)
        have h5678 := measureReal_union_le (μ := runLaw μ D Dsf) E5 (E6 ∪ (E7 ∪ E8))
        have hrest := measureReal_union_le (μ := runLaw μ D Dsf) (E3 ∪ E4) (E5 ∪ (E6 ∪ (E7 ∪ E8)))
        have hall := measureReal_union_le (μ := runLaw μ D Dsf) (E1 ∪ E2)
          ((E3 ∪ E4) ∪ (E5 ∪ (E6 ∪ (E7 ∪ E8))))
        linarith
    _ ≤ ((B.m : ℝ) ^ 2 * ρ + (populations.card : ℝ) * (B.m : ℝ) ^ 2 * ρ)
        + ((Real.exp (-2 * (B.m : ℝ) * tcls ^ 2) + Real.exp (-2 * (B.m : ℝ) * tcls ^ 2))
          + (Estall + (Edirty + (Real.exp (-2 * (B.m : ℝ) * th ^ 2)
            + (E / l + (E / l + 2 * Real.exp (-2 * (n₀ : ℝ) * τ ^ 2))))))) := by
        gcongr
        · exact cert_not_injective_le D Dsf j B.m ρ (hρ j hj) hρ0
        · exact prefix_cert_disjoint_le D Dsf populations j B.m ρ hρ (hρ j hj) hρ0
        · exact cert_class_count_le D Dsf O j B.m 1 qacc tcls hqacc0 htcls hqacc
        · exact cert_class_count_le D Dsf O j B.m 0 qrej tcls hqrej0 htcls hqrej
        · exact measureReal_heavyHits_le D Dsf O populations j B 0 (Δp * (kmax : ℝ)) th hth
        · exact measureReal_retMiss_le hflat O populations D Dsf hsupp j hj B εcov α τ l E
            n₀ kmin kmax hE hl hτ hε0 hε1 hsig hdec hcut hga hgr hα
    _ = ((populations.card : ℝ) + 1) * (B.m : ℝ) ^ 2 * ρ
        + (2 * Real.exp (-2 * (B.m : ℝ) * tcls ^ 2)
          + (Estall + (Edirty + (Real.exp (-2 * (B.m : ℝ) * th ^ 2)
            + (E / l + (E / l + 2 * Real.exp (-2 * (n₀ : ℝ) * τ ^ 2))))))) := by ring

open scoped Classical in
/-- **Part 1 at one state and one population.**  A family the gate admits is right on all
but `εcov` of the population, except on five events: the certification draws repeat, they
meet the table, the sample misses the wrong set, or the gate admits a side a `9εcov/16`
fraction of which is wrong — twice, once per side.

The constants are forced.  The gate's margin `(½−η)εcov` is a wrong *fraction* of `εcov/2`,
the sample delivers `3εcov/4` of the wrong mass, and `exists_wrong_side` spends `2β` of that
on the side that may be too small to charge a tail to; `9εcov/16` and `β = εcov/32` leave
both inequalities strict. -/
theorem measureReal_admitFail_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (j : J) (hj : j ∈ populations) (B : Budget)
    (hlohi : B.lo < B.hi) (εcov α ρ : ℝ) (hε0 : 0 ≤ εcov) (hε1 : εcov ≤ 1) (hα : α < 1 / 2)
    (hsig : O.η ≤ 1 / 2) (hρ : ∀ j' ∈ populations, collisionMass (D j') ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runLaw μ D Dsf).real ({x : Run Ω S J | admitted O B.lo B.hi εcov α
          ((clusterAt O populations x B).erase 1) (certOf j B.m x) (nz x)}
        ∩ {x : Run Ω S J | ¬ (1 - εcov ≤ (D j).real
            {p | cutCorrect O B.lo B.hi (clusterAt O populations x B) p (nz x)})})
      ≤ ((populations.card : ℝ) + 1) * (B.m : ℝ) ^ 2 * ρ
        + (Real.exp (-2 * (B.m : ℝ) * (εcov / 4) ^ 2)
          + 2 * Real.exp (-2 * (εcov / 32 * (B.m : ℝ))
              * ((1 - 2 * O.η) * εcov / 16) ^ 2)) := by
  classical
  have hη := eta_nonneg O
  set c : ℝ := 9 * εcov / 16 with hc
  set β : ℝ := εcov / 32 with hβ
  set E1 : Set (Run Ω S J) :=
    {x | ¬ Function.Injective (fun i : Fin B.m => cert j i.val x)} with hE1
  set E2 : Set (Run Ω S J) :=
    {x | ¬ Disjoint (prefixesAt populations B.m x) (certOf j B.m x)} with hE2
  set E3 : Set (Run Ω S J) := hitShort O populations (D j) j B εcov (εcov / 4) with hE3
  set E4 : Set (Run Ω S J) := gateBadAcc O populations j B εcov c β with hE4
  set E5 : Set (Run Ω S J) := gateBadRej O populations j B εcov c β with hE5
  have hsub : ({x : Run Ω S J | admitted O B.lo B.hi εcov α
        ((clusterAt O populations x B).erase 1) (certOf j B.m x) (nz x)}
      ∩ {x : Run Ω S J | ¬ (1 - εcov ≤ (D j).real
          {p | cutCorrect O B.lo B.hi (clusterAt O populations x B) p (nz x)})})
      ⊆ (E1 ∪ E2) ∪ (E3 ∪ (E4 ∪ E5)) := by
    rintro x ⟨hadm, hfail⟩
    by_cases hinj : Function.Injective (fun i : Fin B.m => cert j i.val x)
    · by_cases hdisj : Disjoint (prefixesAt populations B.m x) (certOf j B.m x)
      · have hWmass : εcov ≤ (D j).real
            {p | ¬ cutCorrect O B.lo B.hi (clusterAt O populations x B) p (nz x)} := by
          rw [show {p : S | ¬ cutCorrect O B.lo B.hi (clusterAt O populations x B) p (nz x)}
              = {p : S | cutCorrect O B.lo B.hi (clusterAt O populations x B) p (nz x)}ᶜ from rfl,
            measureReal_compl (measurableSet_of_countable _), measureReal_def, measure_univ,
            ENNReal.toReal_one]
          push_neg at hfail
          simp only [Set.mem_setOf_eq] at hfail
          linarith
        by_cases hcount : (((certOf j B.m x).filter (fun p =>
            ¬ cutCorrect O B.lo B.hi (clusterAt O populations x B) p (nz x))).card : ℝ)
              ≤ (B.m : ℝ) * (εcov - εcov / 4)
        · exact Or.inr (Or.inl ⟨hinj, hWmass, hcount⟩)
        · push_neg at hcount
          have hcard : ((certOf j B.m x).card : ℝ) = (B.m : ℝ) := by
            have hinjOn : Set.InjOn (fun i => cert j i x) ↑(Finset.range B.m) := by
              intro a ha b hb hab
              have := hinj (show (fun i : Fin B.m => cert j i.val x)
                  ⟨a, Finset.mem_range.1 (by simpa using ha)⟩
                = (fun i : Fin B.m => cert j i.val x)
                  ⟨b, Finset.mem_range.1 (by simpa using hb)⟩ from hab)
              simpa using congrArg Fin.val this
            unfold certOf
            rw [Finset.card_image_of_injOn hinjOn, Finset.card_range]
          have hsides : Disjoint (sideAcc O populations j B x) (sideRej O populations j B x) := by
            refine Finset.disjoint_left.2 (fun p hp hp' => ?_)
            have h1 := (Finset.mem_filter.1 hp).2
            have h2 := (Finset.mem_filter.1 hp').2
            omega
          have hwrong := card_cert_wrong_le O populations j B (by omega) x
          have hwrongR : (((certOf j B.m x).filter (fun p =>
                ¬ cutCorrect O B.lo B.hi (clusterAt O populations x B) p (nz x))).card : ℝ)
              ≤ (((sideAcc O populations j B x).filter (fun p => O.label p = 0)).card : ℝ)
                + (((sideRej O populations j B x).filter
                    (fun p => ¬ (O.label p = 0))).card : ℝ) := by
            exact_mod_cast hwrong
          have hchoice := exists_wrong_side O (certOf j B.m x) (sideAcc O populations j B x)
            (sideRej O populations j B x) (sideAcc_subset O populations j B x)
            (sideRej_subset O populations j B x) hsides
            ((((certOf j B.m x).filter (fun p =>
              ¬ cutCorrect O B.lo B.hi (clusterAt O populations x B) p (nz x))).card : ℝ))
            c β (by positivity) (by positivity) (by rw [hcard]; nlinarith) hwrongR
          have hacc := admittedCount_of_admitted O B.lo B.hi εcov α
            ((clusterAt O populations x B).erase 1) (certOf j B.m x) (nz x) hα
            (gateAcc_mem O hε0 hε1 hsig).1 (gateAcc_mem O hε0 hε1 hsig).2
            (gateRej_mem O hε0 hε1 hsig).1 (gateRej_mem O hε0 hε1 hsig).2 hadm
          rcases hchoice with ⟨h1, h2⟩ | ⟨h1, h2⟩
          · refine Or.inr (Or.inr (Or.inl ⟨hdisj, ?_, h2, hacc.1⟩))
            rw [hcard] at h1
            exact h1
          · refine Or.inr (Or.inr (Or.inr ⟨hdisj, ?_, h2, hacc.2⟩))
            rw [hcard] at h1
            exact h1
      · exact Or.inl (Or.inr hdisj)
    · exact Or.inl (Or.inl hinj)
  have hτacc : gateAcc O εcov - (1 - O.η) + c * (1 - 2 * O.η)
      = (1 - 2 * O.η) * εcov / 16 := by
    unfold gateAcc; rw [hc]; ring
  have hτrej : O.η + c * (1 - 2 * O.η) - gateRej O εcov
      = (1 - 2 * O.η) * εcov / 16 := by
    unfold gateRej; rw [hc]; ring
  have hτ0 : (0 : ℝ) ≤ (1 - 2 * O.η) * εcov / 16 := by nlinarith
  calc (runLaw μ D Dsf).real ({x : Run Ω S J | admitted O B.lo B.hi εcov α
        ((clusterAt O populations x B).erase 1) (certOf j B.m x) (nz x)}
      ∩ {x : Run Ω S J | ¬ (1 - εcov ≤ (D j).real
          {p | cutCorrect O B.lo B.hi (clusterAt O populations x B) p (nz x)})})
      ≤ (runLaw μ D Dsf).real ((E1 ∪ E2) ∪ (E3 ∪ (E4 ∪ E5))) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ((runLaw μ D Dsf).real E1 + (runLaw μ D Dsf).real E2)
        + ((runLaw μ D Dsf).real E3
          + ((runLaw μ D Dsf).real E4 + (runLaw μ D Dsf).real E5)) := by
        have h12 := measureReal_union_le (μ := runLaw μ D Dsf) E1 E2
        have h45 := measureReal_union_le (μ := runLaw μ D Dsf) E4 E5
        have h345 := measureReal_union_le (μ := runLaw μ D Dsf) E3 (E4 ∪ E5)
        have hall := measureReal_union_le (μ := runLaw μ D Dsf) (E1 ∪ E2) (E3 ∪ (E4 ∪ E5))
        linarith
    _ ≤ ((B.m : ℝ) ^ 2 * ρ + (populations.card : ℝ) * (B.m : ℝ) ^ 2 * ρ)
        + (Real.exp (-2 * (B.m : ℝ) * (εcov / 4) ^ 2)
          + (Real.exp (-2 * (β * (B.m : ℝ)) * ((1 - 2 * O.η) * εcov / 16) ^ 2)
            + Real.exp (-2 * (β * (B.m : ℝ)) * ((1 - 2 * O.η) * εcov / 16) ^ 2))) := by
        gcongr
        · exact cert_not_injective_le D Dsf j B.m ρ (hρ j hj) hρ0
        · exact prefix_cert_disjoint_le D Dsf populations j B.m ρ hρ (hρ j hj) hρ0
        · exact measureReal_hitShort_le D Dsf O populations j B εcov (εcov / 4) hε0
            (by positivity)
        · exact le_trans (measureReal_gateBadAcc_le hflat O populations D Dsf hsupp j hj B
            εcov c β (by positivity) (by positivity) hsig (by rw [hτacc]; exact hτ0))
            (le_of_eq (by rw [hτacc]))
        · exact le_trans (measureReal_gateBadRej_le hflat O populations D Dsf hsupp j hj B
            εcov c β (by positivity) (by positivity) hsig (by rw [hτrej]; exact hτ0))
            (le_of_eq (by rw [hτrej]))
    _ = ((populations.card : ℝ) + 1) * (B.m : ℝ) ^ 2 * ρ
        + (Real.exp (-2 * (B.m : ℝ) * (εcov / 4) ^ 2)
          + 2 * Real.exp (-2 * (εcov / 32 * (B.m : ℝ))
              * ((1 - 2 * O.η) * εcov / 16) ^ 2)) := by
        rw [hβ]; ring

/-- **Part 1, reduced to one state.**  States under the cap are a *finite* set, so Part 1 is a
per-state bound at any weight summing under `δ/2`.  There is no union over boundaries and
no union over histories: the boundary and the margin are cutoffs, and the cutoffs are in
the state.

What remains of Part 1 is `hper`: at one state, a family that passes both gates is valid on
every population except with probability `w`. -/
theorem validity_of_budget (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (indecisionLimit εcov α δ ρ : ℝ)
    (w : Budget → ℝ) (hw0 : ∀ B, 0 ≤ w B) (hsum : Summable w) (hle : ∑' B, w B ≤ δ / 2)
    (cap : Budget)
    (hper : ∀ B : {B : Budget // Capped O populations εcov δ ρ cap B}, (runLaw μ D Dsf).real
      (ret O populations indecisionLimit εcov α B.val
        ∩ FailAt O populations D εcov B.val) ≤ w B.val) :
    (runLaw μ D Dsf).real (⋃ B : {B : Budget // Capped O populations εcov δ ρ cap B},
        ret O populations indecisionLimit εcov α B.val
          ∩ FailAt O populations D εcov B.val) ≤ δ / 2 :=
  le_trans (measureReal_iUnion_le_tsum _
      (fun B : {B : Budget // Capped O populations εcov δ ρ cap B} => w B.val)
      (fun B => hw0 B.val) hper (hsum.subtype _))
    (le_trans (hsum.tsum_subtype_le w {B | Capped O populations εcov δ ρ cap B} hw0) hle)

/-- **Part 1 — whatever is returned is valid, whenever it is returned.**

Except with probability `δ/2`, no reachable state is *both* returned and invalid — over
every history the algorithm might follow and every budget it might stop at.  So the loop
may grow and stop however it likes: neither its schedule nor its stopping rule has to be
modelled or itself proved correct.

The intersection with `ret` is load-bearing, not bookkeeping.  Validity at *every* state,
returned or not, is a strictly stronger claim and a false one: at a state whose candidate
pool has outgrown the prefixes, the clustering really can produce a drifted family.  The
algorithm does not return it — that is what the gate is for — and the guarantee is about
what it returns.

Proof plan: this rests on the **accept-preserving gate**, and the gate certifies exactly
what `cutCorrect` asks for.  A returned family has passed `admitted`, which tests — on the
seed's own column, where membership of `p · ε` is membership of `p` — that each side of
its cut reads as its own class.  Reads at distinct prefixes are independent
(`read_indep`, and `prefixesOf` is a `Finset`), so:

* `admitted` forces the accepted side's hit count into the upper tail of
  `Bin(n, accept_thresh)` and the rejected side's into the lower tail of
  `Bin(n, reject_thresh)`;
* a truly-accepting prefix reads accepting with probability `1 − η`, a truly-rejecting one
  with probability `η`, and `admissibleMargin` keeps the margin below the signal, so a
  wrong side drags the count off its tail — Hoeffding at the gap `(1 − 2η)`;
* hence w.h.p. the cut is right on the *sampled* prefixes, and level 2 carries that to
  `D j` over the raw draws `prf j i x`, whose empirical measure is `D j`.

`validity_of_per_state` has already reduced the state union to this.

No slack is carried.  The gate is judged on `certOf` — draws the family was never
selected from — so `splitAcc_sound` applies to the realised cut with no union over
reachable families, and `ε` is dropped from the split so the scored bit does not sit
inside the vote that sorts it.  Both are issue #284; the model here is the fixed
algorithm, and the docstrings on `certOf` and `admitted` say what goes wrong without
them.

Deliberately *not* via `clustering_budget`: inferring validity from the cluster's loss
concentration would need a union bound over every candidate suffix, and the persistent
oracle's fixed noise bits make the per-candidate error floor at the prefix collision
entropy `∑ₐ D_j({a})²` — so that route fails once the candidate pool outgrows
`exp(c/ρ)`.  The gate tests the conclusion instead of inferring it, so the pool size
does not enter, and no collision bound is needed: the gate reads `prefixesOf`, which is
distinct by construction.  `Distributional.lean`'s `clustering_pac` remains the statement
about one budget with a collision bound supplied; it is not what carries this. -/
theorem exists_budget_weight (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (indecisionLimit εcov α : ℝ) (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (Pre : Set S) (hflat : Flat Pre) (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (cap : Budget) (ρ : ℝ) (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ)
    (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) (hα : α < 1 / 2) :
    ∃ w : Budget → ℝ, (∀ B, 0 ≤ w B) ∧ Summable w ∧ (∑' B, w B ≤ δ / 2) ∧
      ∀ B : {B : Budget // Capped O populations εcov δ ρ cap B}, (runLaw μ D Dsf).real
        (ret O populations indecisionLimit εcov α B.val
          ∩ FailAt O populations D εcov B.val) ≤ w B.val  := by
  classical
  have hρ0 : 0 ≤ ρ := by
    obtain ⟨j₀, hj₀⟩ := hpop
    exact le_trans (tsum_nonneg (fun a => sq_nonneg _)) (hρ j₀ hj₀)
  have hfin : {B : Budget | CapOnly cap B}.Finite :=
    Set.finite_coe_iff.1 (inferInstanceAs (Finite {B : Budget // CapOnly cap B}))
  set F : Finset Budget := hfin.toFinset with hF
  have hNcard : (Nat.card {B : Budget // CapOnly cap B} : ℝ) = (F.card : ℝ) := by
    rw [hF]
    exact_mod_cast congrArg (fun n : ℕ => (n : ℝ))
      (Set.ncard_eq_toFinset_card {B : Budget | CapOnly cap B} hfin)
  set v : ℝ := δ / (2 * (Nat.card {B : Budget // CapOnly cap B} : ℝ)) with hv
  have hv0 : 0 ≤ v := by
    rw [hv]
    positivity
  refine ⟨fun B => if Capped O populations εcov δ ρ cap B then v else 0, ?_, ?_, ?_, ?_⟩
  · intro B
    dsimp only
    split_ifs
    · exact hv0
    · exact le_rfl
  · refine summable_of_ne_finset_zero (s := F) (fun B hB => ?_)
    have : ¬ Capped O populations εcov δ ρ cap B := by
      intro hc
      exact hB (by rw [hF]; simpa using hc.capOnly)
    simp [this]
  · have hzero : ∀ B ∉ F, (if Capped O populations εcov δ ρ cap B then v else 0) = 0 := by
      intro B hB
      have : ¬ Capped O populations εcov δ ρ cap B := by
        intro hc
        exact hB (by rw [hF]; simpa using hc.capOnly)
      simp [this]
    rw [tsum_eq_sum hzero]
    calc ∑ B ∈ F, (if Capped O populations εcov δ ρ cap B then v else 0)
        ≤ ∑ _B ∈ F, v := Finset.sum_le_sum (fun B _ => by split_ifs; exacts [le_rfl, hv0])
      _ = (F.card : ℝ) * v := by rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ δ / 2 := by
          rcases Nat.eq_zero_or_pos F.card with hc | hc
          · rw [hc]
            simp only [Nat.cast_zero, zero_mul]
            linarith
          · have hne : (F.card : ℝ) ≠ 0 := by
              have hcpos : (0 : ℝ) < (F.card : ℝ) := by exact_mod_cast hc
              exact ne_of_gt hcpos
            have key : (F.card : ℝ) * (δ / (2 * (F.card : ℝ))) = δ / 2 := by field_simp
            rw [hv, hNcard, key]
  · intro B
    have hgoal : (if Capped O populations εcov δ ρ cap B.val then v else 0) = v := by
      simp [B.property]
    dsimp only
    rw [hgoal]
    by_cases hε1 : εcov ≤ 1
    · have hsub : ret O populations indecisionLimit εcov α B.val
          ∩ FailAt O populations D εcov B.val
          ⊆ ⋃ j ∈ populations, ({x : Run Ω S J | admitted O B.val.lo B.val.hi εcov α
              ((clusterAt O populations x B.val).erase 1) (certOf j B.val.m x) (nz x)}
            ∩ {x : Run Ω S J | ¬ (1 - εcov ≤ (D j).real
                {p | cutCorrect O B.val.lo B.val.hi (clusterAt O populations x B.val) p (nz x)})})
          := by
        rintro x ⟨⟨-, hadm⟩, hfail⟩
        simp only [FailAt, Set.mem_setOf_eq, not_forall] at hfail
        obtain ⟨j, hj, hfj⟩ := hfail
        exact Set.mem_biUnion hj ⟨hadm j hj, hfj⟩
      calc (runLaw μ D Dsf).real (ret O populations indecisionLimit εcov α B.val
            ∩ FailAt O populations D εcov B.val)
          ≤ (runLaw μ D Dsf).real (⋃ j ∈ populations,
              ({x : Run Ω S J | admitted O B.val.lo B.val.hi εcov α
                ((clusterAt O populations x B.val).erase 1) (certOf j B.val.m x) (nz x)}
              ∩ {x : Run Ω S J | ¬ (1 - εcov ≤ (D j).real
                  {p | cutCorrect O B.val.lo B.val.hi
                    (clusterAt O populations x B.val) p (nz x)})})) :=
            measureReal_mono hsub (measure_ne_top _ _)
        _ ≤ ∑ j ∈ populations, (runLaw μ D Dsf).real
              ({x : Run Ω S J | admitted O B.val.lo B.val.hi εcov α
                ((clusterAt O populations x B.val).erase 1) (certOf j B.val.m x) (nz x)}
              ∩ {x : Run Ω S J | ¬ (1 - εcov ≤ (D j).real
                  {p | cutCorrect O B.val.lo B.val.hi
                    (clusterAt O populations x B.val) p (nz x)})}) :=
            measureReal_biUnion_finset_le _ _
        _ ≤ ∑ _j ∈ populations, (((populations.card : ℝ) + 1) * (B.val.m : ℝ) ^ 2 * ρ
              + (Real.exp (-2 * (B.val.m : ℝ) * (εcov / 4) ^ 2)
                + 2 * Real.exp (-2 * (εcov / 32 * (B.val.m : ℝ))
                    * ((1 - 2 * O.η) * εcov / 16) ^ 2))) :=
            Finset.sum_le_sum (fun j hj => measureReal_admitFail_le hflat O populations D Dsf
              hsupp j hj B.val B.property.lohi εcov α ρ hεcov.le hε1 hα hsig.le hρ hρ0)
        _ = stateFail O populations εcov ρ B.val := by
            rw [Finset.sum_const, nsmul_eq_mul]; rfl
        _ ≤ v := B.property.share
    · have hempty : FailAt O populations D εcov B.val = (∅ : Set (Run Ω S J)) := by
        ext x
        simp only [FailAt, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_not]
        intro j _
        exact le_trans (by linarith [not_le.1 hε1]) measureReal_nonneg
      rw [hempty, Set.inter_empty]
      simpa using hv0


/-! ### The argument `exists_budget_weight` needs

At a state `B`, suppose the family `F = clusterAt O populations x B` passes both gates and
yet some population `j` has `(D j) {p | ¬ cutCorrect …} > εcov`.  Write `W` for that
wrong-set.  `W` depends on `ω` and on the pool and table draws, but **not** on the
certification draws, which are fresh.

1. *The sample sees the wrongness.*  Conditionally on `ω` and the table draws, `W` is a
   fixed set and `cert j 0 … cert j (m-1)` are i.i.d. from `D j`, so at least `εcov·m/2` of
   them land in `W` except with probability `exp(−2m(εcov/2)²)`.  Plain Hoeffding over the
   draws; no noise enters, which is why the fresh stream matters twice over.

2. *The draws are distinct.*  Reads at a repeated string are the same bit, so step 3 needs
   the certification prefixes distinct from each other and from the table prefixes.
   `pi_not_injective_le` bounds the first at `m²ρ`; the cross-collisions need the same
   computation against `prefixesOf`.  This is what `hρsmall` pays for.

3. *The gate cannot pass on a wrong cut.*  A prefix in `W` on the accept side has
   `label = 0`, so its seed read is accepting with probability `η` rather than `1 − η`;
   by `mq_mean` a `γ` fraction of such prefixes drags the side's mean down by
   `γ(1 − 2η)`.  `splitAcc_sound` then bounds the chance the count still clears
   `|A|·(hi/k)`, and `splitRej_sound` the mirror.  `admittedCount_of_admitted` is what turns
   the gate's binomial tails into that count condition.

   The side `A` is `ω`-dependent, which is what `mul_ne_self` resolves: `A` is determined by
   the reads at `p · v` for `v ∈ F.erase 1`, all distinct from `p`, so conditioning on the
   votes fixes `A` while leaving the hits' law alone.  `splitAcc_sound` then applies to the
   conditioned law.

4. *Assemble.*  Steps 1–3 bound the per-state failure by
   `exp(−2m(εcov/2)²) + C·m²ρ + exp(−2|A|τ²)` with `τ` the drift gap from step 3.  Any
   summable envelope over `Budget` dominating that serves as `w`; `hδ` and `hεcov` are spent
   choosing it.

Steps 1 and 2 rest on lemmas already proved here (`mq_*`, `pi_coord_eq`,
`pi_not_injective_le`); step 3 rests on `splitAcc_sound`/`splitRej_sound` plus the two
binomial-median facts.  What is not yet written is the conditioning in step 3 and the
envelope in step 4. -/

theorem validity_of_returned (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (indecisionLimit εcov α : ℝ) (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (Pre : Set S) (hflat : Flat Pre) (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (cap : Budget) (ρ : ℝ) (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ)
    (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) (hα : α < 1 / 2) :
    (runLaw μ D Dsf).real (⋃ B : {B : Budget // Capped O populations εcov δ ρ cap B},
        ret O populations indecisionLimit εcov α B.val
          ∩ FailAt O populations D εcov B.val) ≤ δ / 2 := by
  obtain ⟨w, hw0, hsum, hle, hper⟩ := exists_budget_weight O populations D Dsf
    indecisionLimit εcov α hsig hpop Pre hflat hsupp cap ρ hρ hεcov δ hδ hα
  exact validity_of_budget O populations D Dsf indecisionLimit εcov α δ ρ w hw0 hsum hle
    cap hper

/-- What one round at one population can cost: the draws, the family's size and cleanliness,
the sample's two class counts and its heavy fraction, and the round's own two tests. -/
noncomputable def roundFail (populations : Finset J)
    (l τ tcls th E γscr γdirty gdirty tap ρ ρsf : ℝ) (n₀ : ℕ) (B : Budget) : ℝ :=
  ((populations.card : ℝ) + 1) * (B.m : ℝ) ^ 2 * ρ
    + (2 * Real.exp (-2 * (B.m : ℝ) * tcls ^ 2)
      + (((B.M : ℝ) ^ 2 * ρsf + (Real.exp (-2 * (B.M : ℝ) * tap ^ 2)
          + ((B.m : ℝ) ^ 2 * ρ + ((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γscr ^ 2))))
        + (((B.m : ℝ) ^ 2 * ρ + (((B.M : ℝ) + 1) * Real.exp (-2 * (B.m : ℝ) * γdirty ^ 2)
            + (B.M : ℝ) * Real.exp (-2 * (B.m : ℝ) * gdirty ^ 2)))
          + (Real.exp (-2 * (B.m : ℝ) * th ^ 2)
            + (E / l + (E / l + 2 * Real.exp (-2 * (n₀ : ℝ) * τ ^ 2)))))))

/-- **A state whose round can pass.**  Every clause is an inequality among the state's
budgets, the oracle's rates, the populations' class masses and the error budget — no
probability enters, and nothing here is a free parameter of the algorithm.  Reaching such a
state is what the loop's growth schedule is for, and `exists_admissibleCut` is the small
case of the arithmetic being satisfiable at all.

`qcls` is the mass each population puts on each label.  It has to be positive: a population
that never rejects leaves the gate's reject side empty, and an empty side reads as
`binomCdf 0 = 1 > α`, so nothing can be admitted.  That is a property of the populations,
not a knob. -/
def PassableAt (O : Oracle μ S) (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    (indecisionLimit εcov α δ ρ ρsf pAP qcls : ℝ) (B : Budget) : Prop :=
  ∃ (τ tcls th tap γdec γscr γdirty gdirty Δ : ℝ) (n₀ : ℕ),
    0 < B.m ∧ 0 < B.k ∧ B.cn < B.cd ∧ 0 < indecisionLimit ∧ εcov ≤ 1
    ∧ 0 ≤ τ ∧ 0 ≤ tcls ∧ 0 ≤ th ∧ 0 ≤ tap ∧ 0 ≤ γdec ∧ 0 ≤ γscr ∧ 0 ≤ γdirty ∧ 0 ≤ gdirty
    ∧ 0 < Δ ∧ 0 ≤ qcls ∧ 0 ≤ pAP ∧ 0 ≤ ρsf
    -- the populations' classes and the pool's findability
    ∧ (∀ j ∈ populations, qcls ≤ (D j).real {p | O.label p = 1})
    ∧ (∀ j ∈ populations, qcls ≤ (D j).real {p | O.label p = 0})
    ∧ pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p}
    ∧ collisionMass Dsf ≤ ρsf
    -- the certification sample carries both classes and few heavy prefixes
    ∧ ((n₀ : ℝ) + 2 * (indecisionLimit / 2) * (B.m : ℝ)
        + 2 * (indecisionLimit / 2) * (B.m : ℝ) ≤ (B.m : ℝ) * (qcls - tcls))
    ∧ (((populations.card : ℝ) * Δ + gdirty) * ((B.k - 1 : ℕ) : ℝ) + th
        ≤ indecisionLimit / 2)
    -- the screen sits above the clean rate and below the dirty one
    ∧ (∀ n : ℕ, n ≤ populations.card * B.m →
        (n : ℝ) * (2 * O.η * (1 - O.η) + γscr) ≤ (B.sc : ℝ))
    ∧ (∀ n : ℕ, B.m ≤ n → (B.sc : ℝ)
        ≤ (n : ℝ) * ((2 * O.η * (1 - O.η) + Δ * (1 - 2 * O.η) ^ 2) - γdirty))
    -- the pool holds a family
    ∧ ((B.k : ℝ) ≤ (B.M : ℝ) * (pAP - tap))
    -- the thresholds decide, and decide right, on a clean family of the round's size
    ∧ (((B.hi - 1 : ℕ) : ℝ) ≤ ((B.k - 1 : ℕ) : ℝ) * ((1 - O.η) - γdec))
    ∧ (((B.k - 1 : ℕ) : ℝ) * (O.η + γdec) ≤ (B.lo : ℝ) + 1)
    ∧ (((B.k - 1 : ℕ) : ℝ) * (O.η + γdec) ≤ ((B.hi - 1 : ℕ) : ℝ))
    ∧ ((B.lo : ℝ) < ((B.k - 1 : ℕ) : ℝ) * ((1 - O.η) - γdec))
    -- the gate's two sides clear their thresholds
    ∧ (∀ n c : ℕ, n₀ ≤ n → n ≤ c → c ≤ B.m →
        (n : ℝ) * (gateAcc O εcov + τ + τ)
          ≤ (n : ℝ) * (1 - O.η) - (1 - 2 * O.η) * (2 * (indecisionLimit / 2) * (c : ℝ)))
    ∧ (∀ n c : ℕ, n₀ ≤ n → n ≤ c → c ≤ B.m →
        (n : ℝ) * O.η + (1 - 2 * O.η) * (2 * (indecisionLimit / 2) * (c : ℝ))
          ≤ (n : ℝ) * (gateRej O εcov - τ - τ))
    ∧ (Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) ≤ α)
    -- and the whole round, over every population, fits in the budget
    ∧ ((populations.card : ℝ)
        * roundFail populations (indecisionLimit / 2) τ tcls th
            (Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * γdec ^ 2)) γscr γdirty gdirty tap ρ ρsf n₀ B
      ≤ δ / 2)

/-- **Part 2 — the loop terminates.**

Except with probability `δ/2`, some reachable state passes the FNR test, so the loop
returns.

Proof plan: each growth step draws fresh suffixes, and by findability a draw is
accept-preserving with probability `≥ pAP`; once the pool holds enough accept-preserving
suffixes and the prefix count is large enough, every population's vote is decisive on all
but `indecisionLimit` of its mass, so the per-population test passes.  The per-step trigger is
block-local, so `geometric_miss_triggered` gives `(1 − p)^N` and `geom_le` drives it under
`δ/2`.

Termination now also has to clear the accept-preserving gate, which is where the work
moved: `admitted` has to *pass*, not merely be sound.  A family that is genuinely
accept-preserving on the populations reads as such on the seed's column up to the noise,
so this is the gate's own power — the complement of the `α` it spends — and it is why
`pAP > 0` is needed rather than just useful.  `ACCEPT_PRESERVING_GIVE_UP = 20` caps the
refusals; the statement carries no cap, so it is the stronger claim.

`hslack : accFnr < indecisionLimit` is **necessary**, not decoration.  `accFnr` is the
binomial probability that one prefix's count lands inside the indecisive band, so it
bounds the *expected* indecision fraction; if the loop's limit were at or below it the
test could essentially never pass and the loop would not terminate.  The code keeps the
slack: `0.01 < 0.02` (`0.10` after PR #257). -/
theorem loop_terminates {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (indecisionLimit εcov α : ℝ) (cap : Budget) (ρ ρsf pAP qcls δ : ℝ)
    (hsig : O.η ≤ 1 / 2) (hεcov : 0 ≤ εcov) (hpop : populations.Nonempty)
    (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hwit : ∃ B : Budget, Capped O populations εcov δ ρ cap B
      ∧ PassableAt O populations D Dsf indecisionLimit εcov α δ ρ ρsf pAP qcls B) :
    (runLaw μ D Dsf).real {x | ∀ B : {B : Budget // Capped O populations εcov δ ρ cap B},
      x ∉ ret O populations indecisionLimit εcov α B.val} ≤ δ / 2 := by
  classical
  obtain ⟨B, hB, hpass⟩ := hwit
  obtain ⟨τ, tcls, th, tap, γdec, γscr, γdirty, gdirty, Δ, n₀, hmpos, hkpos, hcd, hindLim,
    hε1, hτ, htcls, hth, htap, hγdec, hγscr, hγdirty, hgdirty, hΔ, hqcls0, hpAP0, hρsf0,
    hqacc, hqrej, hpAPBound, hρsf, hclsnum, hheavy, hscLow, hscHigh, hcount,
    hhiUp, hloUp, hhiLo, hloLo, hga, hgr, hα, hbudget⟩ := hpass
  set l : ℝ := indecisionLimit / 2 with hl
  have hlpos : 0 < l := by rw [hl]; linarith
  have hl2 : 2 * l = indecisionLimit := by rw [hl]; ring
  set κ : ℕ := B.k - 1 with hκ
  set E : ℝ := Real.exp (-2 * (κ : ℝ) * γdec ^ 2) with hE
  obtain ⟨j₀, hj₀⟩ := hpop
  -- the whole failure at the one state, population by population
  have hsub : {x : Run Ω S J | ∀ B' : {B : Budget // Capped O populations εcov δ ρ cap B},
        x ∉ ret O populations indecisionLimit εcov α B'.val}
      ⊆ ⋃ j ∈ populations,
        {x : Run Ω S J | x ∉ retAt O populations indecisionLimit εcov α B j} := by
    intro x hx
    by_contra hc
    simp only [Set.mem_iUnion, not_exists, exists_prop, Set.mem_setOf_eq, not_and] at hc
    refine hx ⟨B, hB⟩ ((mem_ret_iff O populations indecisionLimit εcov α B x).2 (fun j hj => ?_))
    by_contra hcj
    exact hc j hj hcj
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_biUnion_finset_le _ _) ?_
  have hper : ∀ j ∈ populations,
      (runLaw μ D Dsf).real
          {x : Run Ω S J | x ∉ retAt O populations indecisionLimit εcov α B j}
        ≤ roundFail populations l τ tcls th E γscr γdirty gdirty tap ρ ρsf n₀ B := by
    intro j hj
    have hstall := measureReal_stalled_le hflat O populations D Dsf hsupp B hcd hkpos j₀ hj₀
      γscr pAP tap ρsf ρ hγscr hpAP0 htap hpAPBound hscLow hcount hρsf hρsf0 (hρ j₀ hj₀) hρ0
    have hdirty := measureReal_dirtyMember_le hflat O populations D Dsf hsupp j hj B hcd hsig
      hmpos Δ γdirty gdirty ρ hΔ hγdirty hgdirty hρ0 (hρ j hj) hscHigh
    have hdec : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * 0 →
        κ ≤ F.card → F.card ≤ κ →
        μ.real {ω | ¬ decided O B.lo (B.hi - 1) F p ω} ≤ E := by
      intro F p hf hmin hmax
      have hcardF : F.card = κ := le_antisymm hmax hmin
      have hclean : flipCount O F p = 0 := by
        have h0 : (0 : ℝ) ≤ flipCount O F p := Nat.cast_nonneg _
        have := hf
        rw [mul_zero] at this
        linarith
      refine le_trans (decided_whp O F p B.lo (B.hi - 1) γdec hγdec hclean ?_ ?_) ?_
      · rw [hcardF]; exact hhiUp
      · rw [hcardF]; exact hloUp
      · rw [hE, hcardF]
    have hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * 0 →
        κ ≤ F.card → F.card ≤ κ →
        μ.real {ω | ¬ cutCorrect O B.lo (B.hi - 1) F p ω} ≤ E := by
      intro F p hf hmin hmax
      have hcardF : F.card = κ := le_antisymm hmax hmin
      refine le_trans (cutCorrect_whp O F p B.lo (B.hi - 1) 0 γdec hf hγdec ?_ ?_) ?_
      · rw [hcardF]
        have : (κ : ℝ) * ((O.η + (1 - 2 * O.η) * 0) + γdec) = (κ : ℝ) * (O.η + γdec) := by ring
        rw [this]
        exact hhiLo
      · rw [hcardF]
        have : (κ : ℝ) * ((O.η + (1 - 2 * O.η) * (1 - 0)) - γdec)
            = (κ : ℝ) * ((1 - O.η) - γdec) := by ring
        rw [this]
        exact hloLo
      · rw [hE, hcardF]
    have hmain := measureReal_notRetAt_le hflat O populations D Dsf hsupp j hj B hmpos hsig
      εcov α τ l E ((populations.card : ℝ) * Δ + gdirty) ρ tcls qcls qcls th n₀ κ κ
      (Real.exp_nonneg _) hlpos hτ hεcov hε1 hρ hρ0
      (by positivity) hth htcls (hqacc j hj) (hqrej j hj) hqcls0 hqcls0
      (by rw [min_self]; exact hclsnum) hheavy _ hstall _ hdirty hdec hcut hga hgr hα
    rw [hl2] at hmain
    refine le_trans hmain (le_of_eq ?_)
    unfold roundFail
    ring
  calc ∑ j ∈ populations, (runLaw μ D Dsf).real
        {x : Run Ω S J | x ∉ retAt O populations indecisionLimit εcov α B j}
      ≤ ∑ _j ∈ populations,
          roundFail populations l τ tcls th E γscr γdirty gdirty tap ρ ρsf n₀ B :=
        Finset.sum_le_sum hper
    _ = (populations.card : ℝ)
          * roundFail populations l τ tcls th E γscr γdirty gdirty tap ρ ρsf n₀ B := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ δ / 2 := hbudget

/-- **The E-L\* clustering algorithm is PAC-correct.**

With probability `≥ 1 − δ` the adaptive loop **terminates**, and the family it returns —
at whatever state it chooses to stop — preserves acceptance on `≥ 1 − εcov` of **each**
prefix population.

Nothing is fixed or idealised.  The guarantee is uniform over every `Budget` — every pair
of growth budgets, every family size, every cluster centre and every pair of gate cutoffs —
so the loop may compute its boundary, margin and size however it likes and stop wherever it
likes.  The cluster is the Lloyd fixed point against its own thresholded mean, seeded at
`ε` and never drifting off it; the gates are the ones `judge_family` applies; and the run
space is the concrete `runLaw`, not an abstract space assumed to exist.

The configuration is *integer* data because that is all it ever was: `vote_mem_grid` says a
threshold matters only through the count it cuts at, and `admissibleCut` is the spec the
search in `population_size_and_evidence_margin` is looking for a witness to. -/
theorem clustering_correct (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (indecisionLimit α : ℝ) (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (Pre : Set S) (hflat : Flat Pre) (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (cap : Budget) (ρ ρsf pAP qcls : ℝ)
    (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ)
    (εcov : ℝ) (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) (hα : α < 1 / 2)
    (hwit : ∃ B : Budget, Capped O populations εcov δ ρ cap B
      ∧ PassableAt O populations D Dsf indecisionLimit εcov α δ ρ ρsf pAP qcls B) :
    1 - δ ≤ (runLaw μ D Dsf).real
      {x | (∃ B : {B : Budget // Capped O populations εcov δ ρ cap B},
          x ∈ ret O populations indecisionLimit εcov α B.val) ∧
        ∀ B : {B : Budget // Capped O populations εcov δ ρ cap B},
          x ∈ ret O populations indecisionLimit εcov α B.val →
          ∀ j ∈ populations, 1 - εcov
            ≤ (D j).real {p | cutCorrect O B.val.lo B.val.hi
                (clusterAt O populations x B.val) p (nz x)}} := by
  have h := sound_and_terminating (runLaw μ D Dsf)
    (fun B : {B : Budget // Capped O populations εcov δ ρ cap B} =>
      ret O populations indecisionLimit εcov α B.val ∩ FailAt O populations D εcov B.val)
    (fun B : {B : Budget // Capped O populations εcov δ ρ cap B} =>
      ret O populations indecisionLimit εcov α B.val) δ
    (validity_of_returned O populations D Dsf indecisionLimit εcov α hsig hpop
      Pre hflat hsupp cap ρ hρ hεcov δ hδ hα)
    (loop_terminates hflat O populations D Dsf hsupp indecisionLimit εcov α cap ρ ρsf pAP
      qcls δ hsig.le hεcov.le hpop hρ
      (le_trans (tsum_nonneg (fun a => sq_nonneg _)) (hρ hpop.choose hpop.choose_spec)) hwit)
  refine le_trans h (le_of_eq ?_)
  congr 1
  ext x
  simp only [Set.mem_setOf_eq, Set.mem_inter_iff, FailAt, not_and, not_not]

#print axioms validity_of_returned
#print axioms loop_terminates
#print axioms clustering_correct

end Loop

end OrthoDFA
