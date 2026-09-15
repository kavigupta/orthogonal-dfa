import OrthoDFA.Distributional
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

The composition is proved; the two halves are `sorry`, with their proof plans recorded.

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

instance : Countable Budget :=
  Function.Injective.countable (f := fun b => (b.M, b.m, b.k, b.cn, b.cd, b.lo, b.hi))
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
cluster it had. -/
noncomputable def lloydStep (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (F : Finset S) : Finset S :=
  if (1 : S) ∈ leastLossSubset (clusterLoss O F cn cd P cands ω) cands k
  then leastLossSubset (clusterLoss O F cn cd P cands ω) cands k else F

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
      split_ifs with h
      · exact h
      · exact ih

/-- The cluster at one budget state.

Nothing here is derived from a real-valued boundary or margin: the centre's cutoff, the
family size and the two gate cutoffs are all part of the state, and the guarantee is
quantified over every state.  `vote_mem_grid` is why that loses nothing — a threshold can
only matter through the count it cuts at. -/
noncomputable def clusterAt (O : Oracle μ S) (populations : Finset J)
    (x : Run Ω S J) (B : Budget) : Finset S :=
  clusterAround O B.cn B.cd (prefixesAt populations B.m x) (poolAt B.M x) (nz x) B.k

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
    (h : ∀ v ∈ F, O.noise (p * v) ω = O.noise (p * v) ω') :
    voteCount O F p ω = voteCount O F p ω' := by
  classical
  unfold voteCount
  exact congrArg Finset.card (Finset.filter_congr (fun v hv => by rw [mq_congr O (h v hv)]))

lemma hammingLoss_congr (O : Oracle μ S) (F : Finset S) (cn cd : ℕ) {P cands : Finset S}
    (hF : F ⊆ cands) {ω ω' : Ω} {v : S} (hv : v ∈ cands)
    (h : ∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') :
    hammingLoss O F cn cd P ω v = hammingLoss O F cn cd P ω' v := by
  classical
  unfold hammingLoss
  refine congrArg _ (congrArg Finset.card (Finset.filter_congr (fun p hp => ?_)))
  rw [mq_congr O (h _ (mem_readSet hp hv)),
    voteCount_congr O F p (fun v' hv' => h _ (mem_readSet hp (hF hv')))]

lemma leastLossSubset_subset' (l : S → ℝ) (cands : Finset S) (k : ℕ) :
    leastLossSubset l cands k ⊆ cands := by
  classical
  unfold leastLossSubset
  split_ifs with hne
  · exact (Finset.mem_powersetCard.mp (Finset.exists_min_image _ _ hne).choose_spec.1).1
  · exact Finset.empty_subset _

lemma clusterLoss_congr (O : Oracle μ S) (F : Finset S) (cn cd : ℕ) (P cands : Finset S)
    (hF : F ⊆ cands) {ω ω' : Ω} (h : ∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') :
    clusterLoss O F cn cd P cands ω = clusterLoss O F cn cd P cands ω' := by
  classical
  funext v
  unfold clusterLoss
  split_ifs with hv
  · exact hammingLoss_congr O F cn cd hF hv h
  · rfl

lemma lloydStep_congr (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (k : ℕ) {F : Finset S}
    (hF : F ⊆ cands) {ω ω' : Ω} (h : ∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') :
    lloydStep O cn cd P cands ω k F = lloydStep O cn cd P cands ω' k F := by
  classical
  unfold lloydStep
  rw [clusterLoss_congr O F cn cd P cands hF h]

lemma lloydStep_subset (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    {F : Finset S} (hF : F ⊆ cands) : lloydStep O cn cd P cands ω k F ⊆ cands := by
  classical
  unfold lloydStep
  split_ifs
  · exact leastLossSubset_subset' _ _ _
  · exact hF

lemma lloydIterate_subset (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ) :
    ∀ (n : ℕ) (F : Finset S), F ⊆ cands → (lloydStep O cn cd P cands ω k)^[n] F ⊆ cands := by
  intro n
  induction n with
  | zero => intro F hF; rw [Function.iterate_zero_apply]; exact hF
  | succ n ih =>
      intro F hF
      rw [Function.iterate_succ_apply]
      exact ih _ (lloydStep_subset O cn cd P cands ω k hF)

lemma lloydIterate_congr (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (k : ℕ)
    {ω ω' : Ω} (h : ∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') :
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
            ih _ (lloydStep_subset O cn cd P cands ω' k hF)
        _ = (lloydStep O cn cd P cands ω' k)^[n + 1] F :=
            (Function.iterate_succ_apply _ _ _).symm

/-- **The cluster reads only `readSet`.**  Two noise draws agreeing at `p · v` for every
representative prefix and candidate suffix give the same family — so neither the family nor
any vote cast with it is decided by the oracle's bit at a bare prefix, which is the bit the
gate scores. -/
lemma clusterAround_congr (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (k : ℕ)
    {ω ω' : Ω} (hone : (1 : S) ∈ cands)
    (h : ∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') :
    clusterAround O cn cd P cands ω k = clusterAround O cn cd P cands ω' k :=
  lloydIterate_congr O cn cd P cands k h _ _ (by simpa using hone)

/-! ### The prefix alphabet

The gate scores the oracle's bit at a prefix `p`.  Everything that decides *which side* `p`
falls on — the family, and `p`'s own vote — is read at strings `q · v` with `v ≠ ε`.  For
the gate's null to be honest those must be different strings, and that is a property of
where prefixes come from, not of the algorithm. -/

/-- A set of prefixes is **flat** when no prefix is another prefix extended.

`UniformSampler(DEFAULT_SAMPLE_LENGTH)` draws every probe at one fixed length — *"All of
E-L*'s signal comes from words drawn at this length"* — so `p * v = p'` between two probes
forces `v = ε` on length alone.  Flatness is exactly what that buys, stated without needing
a length function. -/
def Flat (Pre : Set S) : Prop := ∀ p ∈ Pre, ∀ p' ∈ Pre, ∀ v : S, p * v = p' → v = 1

/-- On a flat alphabet the gate's query string is never one of the split's. -/
lemma flat_ne_of_ne_one {Pre : Set S} (hflat : Flat Pre) {p p' : S} (hp : p ∈ Pre)
    (hp' : p' ∈ Pre) {v : S} (hv : v ≠ 1) : p * v ≠ p' :=
  fun h => hv (hflat p hp p' hp' v h)

lemma clusterAround_subset (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (hone : (1 : S) ∈ cands) : clusterAround O cn cd P cands ω k ⊆ cands := by
  classical
  unfold clusterAround
  exact lloydIterate_subset O cn cd P cands ω k _ _ (by simpa using hone)

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

open scoped Classical in
/-- The prefixes the gate's family accepts at population `j`. -/
noncomputable def sideAcc (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (x : Run Ω S J) : Finset S :=
  (certOf j B.m x).filter
    (fun p => B.hi < voteCount O ((clusterAt O populations x B).erase 1) p (nz x))

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
      = clusterAt O populations ((ω', d) : Run Ω S J) B := by
    refine clusterAround_congr O B.cn B.cd _ _ B.k (one_mem_poolAt _ _) (fun w hw => h w ?_)
    exact Finset.mem_union_left _ hw
  have hsub : clusterAt O populations ((ω, d) : Run Ω S J) B
      ⊆ poolAt B.M ((ω, d) : Run Ω S J) :=
    clusterAround_subset _ _ _ _ _ _ _ (one_mem_poolAt _ _)
  unfold sideAcc
  refine Finset.filter_congr (fun p hp => ?_)
  have hvc : voteCount O ((clusterAt O populations ((ω, d) : Run Ω S J) B).erase 1) p ω
      = voteCount O ((clusterAt O populations ((ω, d) : Run Ω S J) B).erase 1) p ω' := by
    refine voteCount_congr O _ p (fun v hv => h _ ?_)
    refine Finset.mem_union_right _ (mem_readSet hp ?_)
    exact Finset.mem_erase.2 ⟨(Finset.mem_erase.1 hv).1, hsub (Finset.mem_erase.1 hv).2⟩
  simp only [nz, ← hfam, hvc]

lemma sideRej_congr (O : Oracle μ S) (populations : Finset J) (j : J) (B : Budget)
    (d : ((ℕ → S) × (J → ℕ → S)) × (J × ℕ → S)) {ω ω' : Ω}
    (h : ∀ w ∈ gateReads populations j B ((ω, d) : Run Ω S J),
      O.noise w ω = O.noise w ω') :
    sideRej O populations j B (ω, d) = sideRej O populations j B (ω', d) := by
  classical
  have hfam : clusterAt O populations ((ω, d) : Run Ω S J) B
      = clusterAt O populations ((ω', d) : Run Ω S J) B := by
    refine clusterAround_congr O B.cn B.cd _ _ B.k (one_mem_poolAt _ _) (fun w hw => h w ?_)
    exact Finset.mem_union_left _ hw
  have hsub : clusterAt O populations ((ω, d) : Run Ω S J) B
      ⊆ poolAt B.M ((ω, d) : Run Ω S J) :=
    clusterAround_subset _ _ _ _ _ _ _ (one_mem_poolAt _ _)
  unfold sideRej
  refine Finset.filter_congr (fun p hp => ?_)
  have hvc : voteCount O ((clusterAt O populations ((ω, d) : Run Ω S J) B).erase 1) p ω
      = voteCount O ((clusterAt O populations ((ω, d) : Run Ω S J) B).erase 1) p ω' := by
    refine voteCount_congr O _ p (fun v hv => h _ ?_)
    refine Finset.mem_union_right _ (mem_readSet hp ?_)
    exact Finset.mem_erase.2 ⟨(Finset.mem_erase.1 hv).1, hsub (Finset.mem_erase.1 hv).2⟩
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
  have hv1 : v = 1 := hflat p (hP p hp) _ (hC _ hz) v rfl
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
  exact (Finset.mem_erase.1 hv).1 (hflat p (hC p hp) _ (hC _ hz) v rfl)

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
    (Pr : S → Ω → Prop) (hPr : ∀ p ∈ A, MeasurableSet[noiseAlg O T] {ω | Pr p ω})
    (U : Finset S) :
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
/-- **The gate's worst case survives the side being chosen by the clustering.**

`hbad` bounds the score's failure for each *fixed* side; the conclusion bounds it for the
side the run actually produces.  What makes that free is that the side is decided by the
block `Q` and the score reads the block `C`, and those are disjoint. -/
theorem gate_side_bound (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    (side : Ω → Finset S) (hside : ∀ ω, side ω ⊆ C)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → side ω = side ω')
    (P : Finset S → Finset S → Prop) (E : ℝ) (hE : 0 ≤ E)
    (hbad : ∀ A₀ ∈ C.powerset, μ.real {ω | P A₀ (A₀.filter (fun p => mq O p ω = 1))} ≤ E) :
    μ.real {ω | P (side ω) ((side ω).filter (fun p => mq O p ω = 1))} ≤ E := by
  classical
  set Bad : Finset S → Set Ω :=
    fun A₀ => {ω | P A₀ (A₀.filter (fun p => mq O p ω = 1))} with hBaddef
  set side' : Ω → Finset S := fun ω => if ω ∈ noiseClean O Q then side ω else ∅ with hside'def
  have hmeasBad : ∀ A₀, MeasurableSet (Bad A₀) := fun A₀ =>
    noiseAlg_le O Set.univ _ (measurableSet_filter_pred O (T := Set.univ) (by simp) _)
  have hmeasBadC : ∀ A₀ ∈ C.powerset, MeasurableSet[noiseAlg O ↑C] (Bad A₀) := fun A₀ hA₀ =>
    measurableSet_filter_pred O (by exact_mod_cast Finset.mem_powerset.1 hA₀) _
  have hsel' : ∀ ω, side' ω ∈ C.powerset := by
    intro ω
    rw [hside'def]
    by_cases hc : ω ∈ noiseClean O Q
    · simp [hc, Finset.mem_powerset, hside ω]
    · simp [hc, Finset.empty_mem_powerset]
  have hsplit : ∀ A₀, {ω | side' ω = A₀}
      = ({ω | side ω = A₀} ∩ noiseClean O Q) ∪ (if A₀ = ∅ then (noiseClean O Q)ᶜ else ∅) := by
    intro A₀
    ext ω
    by_cases hc : ω ∈ noiseClean O Q <;> by_cases he : A₀ = ∅ <;>
      simp [hside'def, hc, he, Set.mem_setOf_eq, eq_comm (a := (∅ : Finset S))]
  have hmeasSelQ : ∀ A₀, MeasurableSet[noiseAlg O ↑Q] {ω | side' ω = A₀} := by
    intro A₀
    rw [hsplit A₀]
    refine MeasurableSet.union (measurableSet_side_clean O Q side hcongr A₀) ?_
    split_ifs
    · exact (measurableSet_noiseClean O Q).compl
    · exact (noiseAlg O ↑Q).measurableSet_empty
  have hmeasSel : ∀ A₀, MeasurableSet {ω | side' ω = A₀} := fun A₀ =>
    noiseAlg_le O ↑Q _ (hmeasSelQ A₀)
  have hindep : ∀ A₀ ∈ C.powerset,
      μ.real ({ω | side' ω = A₀} ∩ Bad A₀) = μ.real {ω | side' ω = A₀} * μ.real (Bad A₀) := by
    intro A₀ hA₀
    have hI := (indep_noiseAlg O hdisj.symm).indepSet_of_measurableSet (hmeasSelQ A₀)
      (hmeasBadC A₀ hA₀)
    have := hI.measure_inter_eq_mul
    simp only [measureReal_def, this, ENNReal.toReal_mul]
  have hmain := measureReal_selection_le (μ := μ) C.powerset side' hsel' hmeasSel Bad hmeasBad E
    hindep (fun A₀ hA₀ => hbad A₀ hA₀) hE
  have hsub : {ω | P (side ω) ((side ω).filter (fun p => mq O p ω = 1))}
      ⊆ {ω | ω ∈ Bad (side' ω)} ∪ (noiseClean O Q)ᶜ := by
    intro ω hω
    by_cases hc : ω ∈ noiseClean O Q
    · refine Or.inl ?_
      change ω ∈ Bad (side' ω)
      rw [hside'def]
      simp only [hc, if_pos]
      exact hω
    · exact Or.inr hc
  calc μ.real {ω | P (side ω) ((side ω).filter (fun p => mq O p ω = 1))}
      ≤ μ.real ({ω | ω ∈ Bad (side' ω)} ∪ (noiseClean O Q)ᶜ) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ μ.real {ω | ω ∈ Bad (side' ω)} + μ.real (noiseClean O Q)ᶜ := measureReal_union_le _ _
    _ = μ.real {ω | ω ∈ Bad (side' ω)} := by rw [noiseClean_ae O Q, add_zero]
    _ ≤ E := hmain

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
  let side := P.filter (fun p => hi < voteCount O F p ω)
  ((side.filter (fun p => mq O p ω = 1)).card, side.card)

open scoped Classical in
/-- `_split_counts` on the reject side. -/
noncomputable def splitRej (O : Oracle μ S) (lo : ℕ) (F P : Finset S) (ω : Ω) : ℕ × ℕ :=
  let side := P.filter (fun p => voteCount O F p ω ≤ lo)
  ((side.filter (fun p => mq O p ω = 1)).card, side.card)

/-- `drift_verdict`'s **ADMITTED**: each side of the cut reads as its own class on the
seed's column, at error rate `α` (`ACCEPT_PRESERVING_ERROR_RATE = 0.05`).

`θacc` and `θrej` are the rates the two sides are held to (PR #286), and they are left
free.  A prefix the cut calls accepting reads as accepting on the seed's column with
probability `1 − η` when the cut is right and `η` when it is not, so a cut wrong on a `γ`
fraction reads at `(1 − η) − γ(1 − 2η)`, and `θacc` has to separate those.  Where in that
window it sits is a **trade**: the distance down to the drifted rate is what validity
spends, the distance up to the clean rate is what termination spends, and the two sum to
`γ(1 − 2η)`.  Fixing `θacc` midway charges validity four times what it needs, and
termination is the half that can buy more prefixes — so the placement stays a parameter
and the assembly picks it.

The *vote* cutoffs `hi`/`lo` cannot serve as these rates: `hi/k` sits a fixed distance
below `1 − η`, so drift finer than that reads as clean however many prefixes are certified
on — a floor set by the threshold, not by the sample.

Applied in `ret` to the family with `ε` removed and to `certOf`, the certification draws.
Removing `ε` matters because it is in every family, so the vote would otherwise contain
`mq p` — the very bit the split is scored against.  Judging on `certOf` matters because
the family was selected against `prefixesOf`.  (Issue #284.) -/
def admitted (O : Oracle μ S) (lo hi : ℕ) (θacc θrej α : ℝ) (F P : Finset S) (ω : Ω) : Prop :=
  binomSfGe (splitAcc O hi F P ω).2 θacc (splitAcc O hi F P ω).1 ≤ α
    ∧ binomCdf (splitRej O lo F P ω).2 θrej (splitRej O lo F P ω).1 ≤ α

open scoped Classical in
/-- **The loop's return test** at a state: the FNR gate (PR #257: held per population, not
over their union) and the accept-preserving gate.  A family failing either is not
returned — `judge_family` sets its FNR to 1 and the loop samples more.

Both gates read the *distinct* prefixes, as the code does; the accept-preserving gate reads
the certification draws (`certOf`), which the family was never selected from, with the seed
dropped from the split. -/
noncomputable def ret (O : Oracle μ S) (populations : Finset J)
    (indecisionLimit θacc θrej α : ℝ) (B : Budget) : Set (Run Ω S J) :=
  {x | (∀ j ∈ populations,
      (((prefixesOf j B.m x).filter (fun p => ¬ decided O B.lo B.hi
          (clusterAt O populations x B) p (nz x))).card : ℝ)
        ≤ indecisionLimit * (prefixesOf j B.m x).card)
    ∧ ∀ j ∈ populations, admitted O B.lo B.hi θacc θrej α
        ((clusterAt O populations x B).erase 1) (certOf j B.m x) (nz x)}

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
def admittedCount (O : Oracle μ S) (lo hi : ℕ) (θacc θrej : ℝ) (F P : Finset S) (ω : Ω) :
    Prop :=
  ((splitAcc O hi F P ω).2 : ℝ) * θacc ≤ (splitAcc O hi F P ω).1
    ∧ ((splitRej O lo F P ω).1 : ℝ) ≤ (splitRej O lo F P ω).2 * θrej

lemma admittedCount_of_admitted (O : Oracle μ S) (lo hi : ℕ) (θacc θrej α : ℝ)
    (F P : Finset S) (ω : Ω) (hα : α < 1 / 2)
    (hacc0 : 0 ≤ θacc) (hacc1 : θacc ≤ 1) (hrej0 : 0 ≤ θrej) (hrej1 : θrej ≤ 1)
    (h : admitted O lo hi θacc θrej α F P ω) : admittedCount O lo hi θacc θrej F P ω :=
  ⟨le_of_lt (lt_of_binomSfGe_le _ _ _ hacc0 hacc1 hα h.1),
    le_of_lt (lt_of_binomCdf_le _ _ _ hrej0 hrej1 hα h.2)⟩

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
structure Capped (cap B : Budget) : Prop where
  M : B.M ≤ cap.M
  m : B.m ≤ cap.m
  k : B.k ≤ cap.k
  cn : B.cn ≤ cap.cn
  cd : B.cd ≤ cap.cd
  lo : B.lo ≤ cap.lo
  hi : B.hi ≤ cap.hi

instance instFiniteCapped (cap : Budget) : Finite {B : Budget // Capped cap B} := by
  refine Finite.of_injective
    (fun B => ((⟨B.val.M, Nat.lt_succ_of_le B.property.M⟩ : Fin (cap.M + 1)),
      (⟨B.val.m, Nat.lt_succ_of_le B.property.m⟩ : Fin (cap.m + 1)),
      (⟨B.val.k, Nat.lt_succ_of_le B.property.k⟩ : Fin (cap.k + 1)),
      (⟨B.val.cn, Nat.lt_succ_of_le B.property.cn⟩ : Fin (cap.cn + 1)),
      (⟨B.val.cd, Nat.lt_succ_of_le B.property.cd⟩ : Fin (cap.cd + 1)),
      (⟨B.val.lo, Nat.lt_succ_of_le B.property.lo⟩ : Fin (cap.lo + 1)),
      (⟨B.val.hi, Nat.lt_succ_of_le B.property.hi⟩ : Fin (cap.hi + 1)))) ?_
  intro B B' hb
  simpa [Subtype.ext_iff, Budget.ext_iff, Prod.ext_iff, Fin.ext_iff] using hb

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

/-- **The seed's own reads are independent across prefixes.**  Each prefix contributes a
function of two strings, `p` and `p · v`, and those pairs are pairwise disjoint: `p · v` by
right-cancellation, the bare prefixes by distinctness, and the two kinds from each other by
flatness. -/
lemma seedLoss_indep {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S) (cn cd : ℕ)
    {P : Finset S} (hP : ∀ p ∈ P, p ∈ Pre) (v : S) (hv : v ≠ 1) :
    iIndepFun (fun p : {p // p ∈ P} => seedLoss O cn cd v p.val) μ := by
  refine iIndepFun_blocks (X := O.noise) (fun w => O.noise_meas' w) O.noise_indep
    (fun p : {p // p ∈ P} => {p.val, p.val * v}) ?_ _
    (fun p => seedLoss_meas O cn cd v p.val) ?_
  · intro a b hab
    refine Finset.disjoint_left.2 (fun w hw hw' => ?_)
    simp only [Finset.mem_insert, Finset.mem_singleton] at hw hw'
    rcases hw with rfl | rfl <;> rcases hw' with hb | hb
    · exact hab (Subtype.ext hb)
    · exact flat_ne_of_ne_one hflat (hP _ b.property) (hP _ a.property) hv hb.symm
    · exact flat_ne_of_ne_one hflat (hP _ a.property) (hP _ b.property) hv hb
    · exact hab (Subtype.ext (mul_right_cancel hb))
  · intro p ω ω' h
    have h1 : O.noise (p.val * v) ω = O.noise (p.val * v) ω' := h _ (by simp)
    have h0 : O.noise p.val ω = O.noise p.val ω' := h _ (by simp)
    unfold seedLoss
    have hvc : ∀ w ∈ ({(1 : S)} : Finset S),
        O.noise (p.val * w) ω = O.noise (p.val * w) ω' := by
      intro w hw
      rw [Finset.mem_singleton] at hw
      subst hw
      rw [mul_one]
      exact h0
    rw [mq_congr O h1, voteCount_congr O {(1 : S)} p.val hvc]

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

/-- **The first Lloyd step keeps only low-flip candidates.**  Its centre is `{ε}`, so the
loss is the disagreement with the seed's own column and the selection is a fixed-loss
argmin — `chosen_accept_preserving_whp` applies with no conditioning.  A candidate carrying
flip mass `Δ` disagrees with the seed on `Δ(1−2η)²` more of the prefixes than an
accept-preserving one. -/
theorem lloyd_first_step_ranked (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (B : Budget) (Δ : ℝ) (hΔ : 0 < Δ) (hsig : O.η < 1 / 2)
    (pAP : ℝ) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hpool : PoolRanked O Δ B) (hfill : (B.k : ℝ) ≤ pAP * B.M / 2) (η : ℝ) :
    (runLaw μ D Dsf).real
      {x | ¬ ∀ v ∈ lloydStep O B.cn B.cd (prefixesAt populations B.m x) (poolAt B.M x) (nz x) B.k
              {(1 : S)},
          ∀ j ∈ populations, flipMass O (D j) v ≤ Δ}
      ≤ η :=
  sorry

/-- **The iteration keeps what the first step gave it.**  If every member of the current
family carries flip mass `≤ Δ`, its thresholded mean is the majority of `k` mostly-correct
columns, so the next step ranks against something at least as good as the seed's column and
its selection is no worse.

The centre being `ω`-dependent is what stops `chosen_accept_preserving_whp` applying
directly, and it is the same shape as the gate's `ω`-dependent side: a candidate's reads
sit at `p · v` while the centre is read at `p · v'` for `v' ∈ F`, disjoint strings whenever
`v ∉ F`.  `measureReal_selection_le` over the centre's pattern is the route. -/
theorem lloyd_step_preserves_ranked (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (B : Budget) (Δ : ℝ) (hΔ : 0 < Δ) (hsig : O.η < 1 / 2)
    (hpool : PoolRanked O Δ B) (η : ℝ) :
    (runLaw μ D Dsf).real
      {x | ∃ F : Finset S, (∀ v ∈ F, ∀ j ∈ populations, flipMass O (D j) v ≤ Δ) ∧
          ¬ ∀ v ∈ lloydStep O B.cn B.cd (prefixesAt populations B.m x) (poolAt B.M x) (nz x) B.k F,
              ∀ j ∈ populations, flipMass O (D j) v ≤ Δ}
      ≤ η :=
  sorry

/-- **Part 1, reduced to one state.**  States under the cap are a *finite* set, so Part 1 is a
per-state bound at any weight summing under `δ/2`.  There is no union over boundaries and
no union over histories: the boundary and the margin are cutoffs, and the cutoffs are in
the state.

What remains of Part 1 is `hper`: at one state, a family that passes both gates is valid on
every population except with probability `w`. -/
theorem validity_of_budget (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (indecisionLimit θacc θrej α εcov δ : ℝ)
    (w : Budget → ℝ) (hw0 : ∀ B, 0 ≤ w B) (hsum : Summable w) (hle : ∑' B, w B ≤ δ / 2)
    (cap : Budget)
    (hper : ∀ B : {B : Budget // Capped cap B}, (runLaw μ D Dsf).real
      (ret O populations indecisionLimit θacc θrej α B.val
        ∩ FailAt O populations D εcov B.val) ≤ w B.val) :
    (runLaw μ D Dsf).real (⋃ B : {B : Budget // Capped cap B},
        ret O populations indecisionLimit θacc θrej α B.val
          ∩ FailAt O populations D εcov B.val) ≤ δ / 2 :=
  le_trans (measureReal_iUnion_le_tsum _
      (fun B : {B : Budget // Capped cap B} => w B.val)
      (fun B => hw0 B.val) hper (hsum.subtype _))
    (le_trans (hsum.tsum_subtype_le w {B | Capped cap B} hw0) hle)

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
    (indecisionLimit θacc θrej α : ℝ) (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (pAP : ℝ) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (cap : Budget) (ρ : ℝ) (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ)
    (εcov : ℝ) (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) (hα : α < 1 / 2)
    (hρsmall : ρ ≤ εcov ^ 2 * δ) :
    ∃ w : Budget → ℝ, (∀ B, 0 ≤ w B) ∧ Summable w ∧ (∑' B, w B ≤ δ / 2) ∧
      ∀ B : {B : Budget // Capped cap B}, (runLaw μ D Dsf).real
        (ret O populations indecisionLimit θacc θrej α B.val
          ∩ FailAt O populations D εcov B.val) ≤ w B.val :=
  sorry

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
    (indecisionLimit θacc θrej α : ℝ) (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (pAP : ℝ) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (cap : Budget) (ρ : ℝ) (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ)
    (εcov : ℝ) (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) (hα : α < 1 / 2)
    (hρsmall : ρ ≤ εcov ^ 2 * δ) :
    (runLaw μ D Dsf).real (⋃ B : {B : Budget // Capped cap B},
        ret O populations indecisionLimit θacc θrej α B.val
          ∩ FailAt O populations D εcov B.val) ≤ δ / 2 := by
  obtain ⟨w, hw0, hsum, hle, hper⟩ := exists_budget_weight O populations D Dsf
    indecisionLimit θacc θrej α hsig hpop pAP hpAPPositive hpAPBound cap ρ hρ εcov hεcov δ hδ hα
    hρsmall
  exact validity_of_budget O populations D Dsf indecisionLimit θacc θrej α εcov δ w hw0 hsum hle
    cap hper

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
theorem loop_terminates (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (accFnr indecisionLimit θacc θrej α : ℝ) (cap : Budget)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (pAP : ℝ) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (δ : ℝ) (hδ : 0 < δ) (hindLim : 0 < indecisionLimit)
    (hslack : accFnr < indecisionLimit) :
    (runLaw μ D Dsf).real {x | ∀ B : {B : Budget // Capped cap B},
      x ∉ ret O populations indecisionLimit θacc θrej α B.val} ≤ δ / 2 :=
  sorry

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
    (accFnr indecisionLimit θacc θrej α : ℝ) (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (pAP : ℝ) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (cap : Budget) (ρ : ℝ) (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ)
    (εcov : ℝ) (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) (hindLim : 0 < indecisionLimit)
    (hslack : accFnr < indecisionLimit) (hα : α < 1 / 2) (hρsmall : ρ ≤ εcov ^ 2 * δ) :
    1 - δ ≤ (runLaw μ D Dsf).real
      {x | (∃ B : {B : Budget // Capped cap B},
          x ∈ ret O populations indecisionLimit θacc θrej α B.val) ∧
        ∀ B : {B : Budget // Capped cap B},
          x ∈ ret O populations indecisionLimit θacc θrej α B.val →
          ∀ j ∈ populations, 1 - εcov
            ≤ (D j).real {p | cutCorrect O B.val.lo B.val.hi
                (clusterAt O populations x B.val) p (nz x)}} := by
  have h := sound_and_terminating (runLaw μ D Dsf)
    (fun B : {B : Budget // Capped cap B} =>
      ret O populations indecisionLimit θacc θrej α B.val ∩ FailAt O populations D εcov B.val)
    (fun B : {B : Budget // Capped cap B} =>
      ret O populations indecisionLimit θacc θrej α B.val) δ
    (validity_of_returned O populations D Dsf indecisionLimit θacc θrej α hsig hpop
      pAP hpAPPositive hpAPBound cap ρ hρ εcov hεcov δ hδ hα hρsmall)
    (loop_terminates O populations D Dsf accFnr indecisionLimit θacc θrej α cap hsig hpop
      pAP hpAPPositive hpAPBound δ hδ hindLim hslack)
  refine le_trans h (le_of_eq ?_)
  congr 1
  ext x
  simp only [Set.mem_setOf_eq, Set.mem_inter_iff, FailAt, not_and, not_not]

#print axioms validity_of_returned
#print axioms loop_terminates
#print axioms clustering_correct

end Loop

end OrthoDFA
