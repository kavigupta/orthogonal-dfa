import OrthoDFA.Distributional
import Mathlib.Probability.ProductMeasure

/-!
# The adaptive clustering loop: the integrated theorem

`sample_suffix_family` is a retry loop: cluster, measure the FNR **per population**, and
if any population is too indecisive grow the pool and try again — stopping at a
data-dependent time.  This file states and proves the guarantee **for that loop**, with
no fixed budget:

> with probability `≥ 1 − δ` the loop **terminates**, and **whatever** family it returns
> classifies `≥ 1 − εcov` of **each** prefix population the way the noiseless oracle does,
> wherever it decides at all.

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

/-- `evidence_margin_for_population_size`: at population size `N`, the margin `eps`
around `center` is *admissible* when the binomial false-positive rate under the null and
false-negative rate under the signal are both within budget. -/
def admissibleMargin (s fpr accFnr center : ℝ) (N : ℕ) (eps : ℝ) : Prop :=
  0 < eps ∧ eps ≤ s ∧
    (binomCdf N center ⌊(N : ℝ) * (center - eps)⌋₊
        + (1 - binomCdf N center (⌈(N : ℝ) * (center + eps)⌉₊ - 1)) ≤ fpr) ∧
    (binomCdf N (s + center) (⌈(N : ℝ) * (center + eps)⌉₊ - 1)
        - binomCdf N (s + center) ⌊(N : ℝ) * (center - eps)⌋₊ ≤ accFnr)

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

/-- The search's real-valued output names a pair of integer cutoffs. -/
lemma admissibleCut_of_admissibleMargin {s fpr accFnr center : ℝ} {N : ℕ} {eps : ℝ}
    (h : admissibleMargin s fpr accFnr center N eps) :
    admissibleCut s fpr accFnr center N ⌊(N : ℝ) * (center - eps)⌋₊
      (⌈(N : ℝ) * (center + eps)⌉₊ - 1) :=
  ⟨h.2.2.1, h.2.2.2⟩

/-- A large enough population always admits a margin, for any positive signal.  (The
binary search in `population_size_and_evidence_margin` terminates.) -/
theorem exists_admissibleMargin (s fpr accFnr center : ℝ) (hs : 0 < s)
    (hfpr : 0 < fpr) (haccFnr : 0 < accFnr) :
    ∃ N, 0 < N ∧ ∃ eps, admissibleMargin s fpr accFnr center N eps :=
  sorry

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

open scoped Classical in
/-- **The suffix family size, derived.**  `population_size_and_evidence_margin` returns the
*least* population size admitting a margin; this is that `N`. -/
noncomputable def suffixFamilySize (s fpr accFnr center : ℝ) : ℕ :=
  if h : ∃ N, 0 < N ∧ ∃ eps, admissibleMargin s fpr accFnr center N eps then Nat.find h else 1

open scoped Classical in
/-- **The evidence margin, derived**: the margin admissible at that population size. -/
noncomputable def evidenceMargin (s fpr accFnr center : ℝ) : ℝ :=
  if h : ∃ eps, admissibleMargin s fpr accFnr center (suffixFamilySize s fpr accFnr center) eps
  then h.choose else 0

theorem suffixFamilySize_pos (s fpr accFnr center : ℝ) (hs : 0 < s)
    (hfpr : 0 < fpr) (haccFnr : 0 < accFnr) : 0 < suffixFamilySize s fpr accFnr center := by
  classical
  rw [suffixFamilySize, dif_pos (exists_admissibleMargin s fpr accFnr center hs hfpr haccFnr)]
  exact (Nat.find_spec (exists_admissibleMargin s fpr accFnr center hs hfpr haccFnr)).1

/-- The algorithm's family size, as `build_pst` computes it from the oracle's signal
`½ − η` and the two acceptable rates. -/
noncomputable def cfgK (O : Oracle μ S) (fpr accFnr center : ℝ) : ℕ :=
  suffixFamilySize (1 / 2 - O.η) fpr accFnr center

/-- The algorithm's evidence margin, likewise derived. -/
noncomputable def cfgMargin (O : Oracle μ S) (fpr accFnr center : ℝ) : ℝ :=
  evidenceMargin (1 / 2 - O.η) fpr accFnr center

/-- The gate's accept threshold: `decision_boundary + evidence_margin`. -/
noncomputable def cfgAcc (O : Oracle μ S) (fpr accFnr center : ℝ) : ℝ :=
  center + cfgMargin O fpr accFnr center

/-- The gate's reject threshold: `decision_boundary − evidence_margin`. -/
noncomputable def cfgRej (O : Oracle μ S) (fpr accFnr center : ℝ) : ℝ :=
  center - cfgMargin O fpr accFnr center

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
abbrev Run (Ω S J : Type*) := Ω × ((ℕ → S) × ((J → ℕ → S) × (J → ℕ → S)))

/-- The law of a run: the three components jointly independent, each stream i.i.d. -/
noncomputable def runLaw (μ : Measure Ω) (D : J → Measure S) (Dsf : Measure S) :
    Measure (Run Ω S J) :=
  μ.prod ((Measure.infinitePi fun _ : ℕ => Dsf).prod
    ((Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j).prod
      (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j)))

instance (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] : IsProbabilityMeasure (runLaw μ D Dsf) := by
  unfold runLaw; infer_instance

/-- The run's persistent noise. -/
def nz (x : Run Ω S J) : Ω := x.1

/-- The `i`-th suffix drawn. -/
def sfx (i : ℕ) (x : Run Ω S J) : S := x.2.1 i

/-- The `i`-th prefix drawn from population `j`. -/
def prf (j : J) (i : ℕ) (x : Run Ω S J) : S := x.2.2.1 j i

/-- The `i`-th **certification** prefix from population `j`: drawn only to read the split
on, never added to the table, and independent of everything the family was chosen from
(`certification_sample`). -/
def cert (j : J) (i : ℕ) (x : Run Ω S J) : S := x.2.2.2 j i

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
    Measure.map (fun x : Run Ω S J => (nz x, (fun i : Fin n => sfx i.val x),
        ((fun (j : J) (i : Fin n) => prf j i.val x),
          (fun (j : J) (i : Fin n) => cert j i.val x)))) (runLaw μ D Dsf)
      = μ.prod
          ((Measure.pi fun _ : Fin n => Dsf).prod
            ((Measure.pi (fun j : J => Measure.pi fun _ : Fin n => D j)).prod
              (Measure.pi (fun j : J => Measure.pi fun _ : Fin n => D j)))) :=
  ((MeasurePreserving.id μ).prod
    ((measurePreserving_finRestrict Dsf n).prod
      ((measurePreserving_pi _ _ fun j => measurePreserving_finRestrict (D j) n).prod
        (measurePreserving_pi _ _ fun j => measurePreserving_finRestrict (D j) n)))).map_eq

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

/-- The loop's growth history: the budget states it has passed through, in order, each a
suffix budget, a prefix budget and a family size.  The algorithm picks this however it
likes. -/
abbrev Hist := List (ℕ × ℕ × ℕ)

/-- The candidate pool at a suffix budget: the first `M` suffixes drawn. -/
noncomputable def poolAt (M : ℕ) (x : Run Ω S J) : Finset S :=
  (Finset.range M).image (fun i => sfx i x)

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
noncomputable def hammingLoss (O : Oracle μ S) (F : Finset S) (b : ℝ) (P : Finset S)
    (ω : Ω) (v : S) : ℝ :=
  ((P.filter (fun p => ¬ ((mq O (p * v) ω = 1) ↔ b < vote O F p ω))).card : ℝ)

open scoped Classical in
/-- One Lloyd step: recentre on the current cluster, then retake the `k` least-loss
candidates — but only while the seed is among them.  `identify_cluster_around` breaks out
(`if seed_local not in nearest`) rather than let the centre drift off `ε`, keeping the
cluster it had. -/
noncomputable def lloydStep (O : Oracle μ S) (b : ℝ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (F : Finset S) : Finset S :=
  if (1 : S) ∈ leastLossSubset (hammingLoss O F b P ω) cands k
  then leastLossSubset (hammingLoss O F b P ω) cands k else F

/-- `identify_cluster_around` iterated to its fixed point.  The total loss is a natural
number bounded by `k·#P` that strictly decreases at each improving step, so `k·#P + 1`
iterations from the seed `ε` already sit at the fixed point — the bound is derived, not a
knob. -/
noncomputable def clusterAround (O : Oracle μ S) (b : ℝ) (P cands : Finset S) (ω : Ω)
    (k : ℕ) : Finset S :=
  (lloydStep O b P cands ω k)^[k * P.card + 1] {(1 : S)}

/-- **The cluster never drifts off the seed.**  `identify_cluster_around` stops the moment
`ε` would leave, so every family the loop proposes contains it — which is what lets the
gate read the split off `ε`'s own column. -/
lemma one_mem_clusterAround (O : Oracle μ S) (b : ℝ) (P cands : Finset S) (ω : Ω) (k : ℕ) :
    (1 : S) ∈ clusterAround O b P cands ω k := by
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

open scoped Classical in
/-- The boundary update at the end of `identify_cluster_around`: the midpoint of the
accept-side and reject-side prefix means, falling back to whichever side is nonempty, then
**clamped to `[s, 1−s]`** for `s = ½ − η`:

```python
signal = pst.config.min_signal_strength
decision_boundary = min(max(decision_boundary, signal), 1 - signal)
```

The clamp is not cosmetic.  A cluster all on one side estimates a boundary whose implied
rates leave `[0,1]`, and for the proof it is what bounds the gate's detection gap away
from zero *uniformly in the boundary* — without which Part 1's union over states diverges,
since infinitely many histories reach the same budget and the gate's error does not decay
in history length. -/
noncomputable def newBoundary (O : Oracle μ S) (F P : Finset S) (ω : Ω) (b : ℝ) : ℝ :=
  let acc := P.filter (fun p => b < vote O F p ω)
  let rej := P.filter (fun p => ¬ (b < vote O F p ω))
  let am := (∑ p ∈ acc, vote O F p ω) / acc.card
  let rm := (∑ p ∈ rej, vote O F p ω) / rej.card
  let raw := if acc.Nonempty then (if rej.Nonempty then (am + rm) / 2 else am)
             else (if rej.Nonempty then rm else b)
  max (1 / 2 - O.η) (min (1 / 2 + O.η) raw)

/-- The cluster at one budget state, at the boundary carried in.

The family size is the state's own `Mm.2.2`, **not** `cfgK` of the boundary.  That matches
`sample_suffix_family`, where `family_size` is carried and recomputed *between* rounds —
`identify_cluster_around(pst, v, family_size, decision_boundary)` takes it as its own
argument, derived from the previous round's boundary, and `readable_size_and_margin` then
steps it down from what the pool actually holds.

It also matters for the proof.  The guarantee is uniform over the boundary, so the union
over `b` has to collapse; `b` enters the clustering only through `b < vote`, and votes live
on the grid `{0, 1/k, …, 1}`, so that dependence is piecewise constant with `≤ k+2` pieces.
Deriving the size from `b` instead would put `suffixFamilySize`'s `Nat.find` — a least `N`
over conditions containing `⌊N(b±eps)⌋` — inside the union, with no bound on the number of
pieces. Carrying `k` keeps the budget index `ℕ × ℕ × ℕ`, still countable. -/
noncomputable def clusterAt (O : Oracle μ S) (populations : Finset J)
    (x : Run Ω S J) (b : ℝ) (Mm : ℕ × ℕ × ℕ) : Finset S :=
  clusterAround O b (prefixesAt populations Mm.2.1 x) (poolAt Mm.1 x) (nz x) Mm.2.2

/-- The decision boundary carried along a history: it starts at `1/2`
(`decision_boundary : float = 0.5`) and each state replaces it with the boundary its own
cluster induces. -/
noncomputable def boundaryFold (O : Oracle μ S) (populations : Finset J)
    (fpr accFnr : ℝ) (x : Run Ω S J) : ℝ → Hist → ℝ
  | b, [] => b
  | b, Mm :: h =>
      boundaryFold O populations fpr accFnr x
        (newBoundary O (clusterAt O populations x b Mm)
          (prefixesAt populations Mm.2.1 x) (nz x) b) h

/-- The boundary after a history. -/
noncomputable def boundaryAfter (O : Oracle μ S) (populations : Finset J)
    (fpr accFnr : ℝ) (x : Run Ω S J) (h : Hist) : ℝ :=
  boundaryFold O populations fpr accFnr x (1 / 2) h

lemma eta_nonneg (O : Oracle μ S) : 0 ≤ O.η := by
  rw [← O.noise_mean 1]
  exact integral_nonneg_of_ae (by filter_upwards [O.noise_icc 1] with ω hω using hω.1)

/-- **The boundary never leaves `[s, 1−s]`.**  It starts at `½`, which is in range, and
every update is clamped.

This is what makes Part 1's union bound finite.  The failure event at a state depends on
the history *only* through the boundary, and infinitely many histories reach any given
budget — so a union over states cannot converge.  Bounded boundaries let the union be
taken over budgets alone, with the boundary handled uniformly. -/
lemma boundaryFold_mem_Icc (O : Oracle μ S) (populations : Finset J) (fpr accFnr : ℝ)
    (x : Run Ω S J) (h : Hist) (b : ℝ)
    (hb : b ∈ Set.Icc (1 / 2 - O.η) (1 / 2 + O.η)) :
    boundaryFold O populations fpr accFnr x b h ∈ Set.Icc (1 / 2 - O.η) (1 / 2 + O.η) := by
  induction h generalizing b with
  | nil => exact hb
  | cons Mm h ih =>
      refine ih _ ?_
      have hle : 1 / 2 - O.η ≤ 1 / 2 + O.η := by linarith [eta_nonneg O]
      constructor
      · exact le_max_left _ _
      · exact max_le hle (min_le_left _ _)

lemma boundaryAfter_mem_Icc (O : Oracle μ S) (populations : Finset J) (fpr accFnr : ℝ)
    (x : Run Ω S J) (h : Hist) :
    boundaryAfter O populations fpr accFnr x h ∈ Set.Icc (1 / 2 - O.η) (1 / 2 + O.η) :=
  boundaryFold_mem_Icc O populations fpr accFnr x h _
    ⟨by linarith [eta_nonneg O], by linarith [eta_nonneg O]⟩

/-- The family the loop proposes at budget `Mm`, having come through history `h`. -/
noncomputable def famAt (O : Oracle μ S) (populations : Finset J)
    (fpr accFnr : ℝ) (x : Run Ω S J) (h : Hist) (Mm : ℕ × ℕ × ℕ) : Finset S :=
  clusterAt O populations x (boundaryAfter O populations fpr accFnr x h) Mm

/-! ## The accept-preserving gate

`AcceptPreservingGate` runs after the FNR test, right before the family is returned.  It
splits the prefixes by the family's *own* cut and counts, on the **seed's own column**,
how many read as accepting — and membership of `p · ε` is membership of `p`, which is why
the gate is read off `ε` and why `one_mem_clusterAround` matters.  A family is admitted
only when each side reads as its own class, by the same thresholds the family itself is
read with (`drift_verdict`).  Anything else sets the round's FNR to 1 and the loop keeps
sampling.

This is what makes a *returned* family valid.  Validity does not have to be inferred from
the clustering's loss concentration union-bounded over every candidate suffix — the gate
tests the conclusion directly, so the candidate pool may be as large as it likes. -/

/-- `P[Bin(N,p) ≥ j]` — `scipy.stats.binom.sf(j-1, N, p)`. -/
noncomputable def binomSfGe (N : ℕ) (p : ℝ) (j : ℕ) : ℝ :=
  ∑ i ∈ Finset.Icc j N, (N.choose i : ℝ) * p ^ i * (1 - p) ^ (N - i)

open scoped Classical in
/-- `_split_counts` on the accept side: `(hits, n)` over the prefixes the family accepts,
counted on the seed's column (`mq O p`, the membership query at `p · ε = p`). -/
noncomputable def splitAcc (O : Oracle μ S) (b fpr accFnr : ℝ) (F P : Finset S) (ω : Ω) :
    ℕ × ℕ :=
  let side := P.filter (fun p => cfgAcc O fpr accFnr b ≤ vote O F p ω)
  ((side.filter (fun p => mq O p ω = 1)).card, side.card)

open scoped Classical in
/-- `_split_counts` on the reject side. -/
noncomputable def splitRej (O : Oracle μ S) (b fpr accFnr : ℝ) (F P : Finset S) (ω : Ω) :
    ℕ × ℕ :=
  let side := P.filter (fun p => vote O F p ω < cfgRej O fpr accFnr b)
  ((side.filter (fun p => mq O p ω = 1)).card, side.card)

/-- `drift_verdict`'s **ADMITTED**: the accepted prefixes read as accepting on the seed's
column significantly above `accept_thresh`, and the rejected ones significantly below
`reject_thresh`, at error rate `α` (`ACCEPT_PRESERVING_ERROR_RATE = 0.05`).

Applied in `ret` to the family with `ε` **removed** and to `certOf`, the certification
draws.  Removing `ε` matters because it is in every family (`one_mem_clusterAround`), so
`vote p` would otherwise contain `mq (p · ε) = mq p` — the very bit the split is scored
against, pushing prefixes within `1/k` of a threshold across it.  With `ε` dropped the
vote reads `p · v` for `v ≠ ε` and the gate reads `p`: distinct strings, independent
bits.  (Issue #284.) -/
def admitted (O : Oracle μ S) (b fpr accFnr α : ℝ) (F P : Finset S) (ω : Ω) : Prop :=
  binomSfGe (splitAcc O b fpr accFnr F P ω).2 (cfgAcc O fpr accFnr b)
      (splitAcc O b fpr accFnr F P ω).1 ≤ α
    ∧ binomCdf (splitRej O b fpr accFnr F P ω).2 (cfgRej O fpr accFnr b)
        (splitRej O b fpr accFnr F P ω).1 ≤ α

/-- A prefix is *decided* when the family's vote clears the state's accept or reject
threshold; otherwise it lands in the indecisive band and counts towards the FNR. -/
def decided (O : Oracle μ S) (b fpr accFnr : ℝ) (F : Finset S) (p : S) (ω : Ω) : Prop :=
  cfgAcc O fpr accFnr b ≤ vote O F p ω ∨ vote O F p ω < cfgRej O fpr accFnr b

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
def cutCorrect (O : Oracle μ S) (b fpr accFnr : ℝ) (F : Finset S) (p : S) (ω : Ω) : Prop :=
  (cfgAcc O fpr accFnr b ≤ vote O F p ω → O.label p = 1) ∧
    (vote O F p ω < cfgRej O fpr accFnr b → O.label p = 0)

open scoped Classical in
/-- The return test and the failure event at an explicit boundary rather than a history. -/
noncomputable def retB (O : Oracle μ S) (populations : Finset J)
    (fpr accFnr indecisionLimit α : ℝ) (b : ℝ) (Mm : ℕ × ℕ × ℕ) : Set (Run Ω S J) :=
  {x | (∀ j ∈ populations,
      (((prefixesOf j Mm.2.1 x).filter (fun p => ¬ decided O b fpr accFnr
          (clusterAt O populations x b Mm) p (nz x))).card : ℝ)
        ≤ indecisionLimit * (prefixesOf j Mm.2.1 x).card)
    ∧ ∀ j ∈ populations, admitted O b fpr accFnr α
        ((clusterAt O populations x b Mm).erase 1) (certOf j Mm.2.1 x) (nz x)}

def FailB (O : Oracle μ S) (populations : Finset J) (D : J → Measure S)
    (fpr accFnr εcov : ℝ) (b : ℝ) (Mm : ℕ × ℕ × ℕ) : Set (Run Ω S J) :=
  {x | ¬ ∀ j ∈ populations, 1 - εcov
        ≤ (D j).real {p | cutCorrect O b fpr accFnr
            (clusterAt O populations x b Mm) p (nz x)}}

open scoped Classical in
/-- **The loop's return test**: the FNR gate (PR #257: held per population, not over their
union) at the state's own boundary and margin, **and** the accept-preserving gate.

Both gates read the *distinct* prefixes, as the code does — `fnr_from_decision` runs on
`compute_decision(vs, table.representative)`, one entry per interned prefix.  Counting
draws instead would weight each prefix by how often it came up, which is a different
quantity and not the one the loop tests.  A
family that fails either is not returned — `judge_family` sets its FNR to 1 and the loop
samples more. -/
noncomputable def ret (O : Oracle μ S) (populations : Finset J)
    (fpr accFnr indecisionLimit α : ℝ) (hM : Hist × (ℕ × ℕ × ℕ)) : Set (Run Ω S J) :=
  {x | x ∈ retB O populations fpr accFnr indecisionLimit α
      (boundaryAfter O populations fpr accFnr x hM.1) hM.2}

/-- The family at a reachable state is **invalid**: on some population its cut is wrong on
more than an `εcov` fraction. -/
def FailAt (O : Oracle μ S) (populations : Finset J) (D : J → Measure S)
    (fpr accFnr εcov : ℝ) (hM : Hist × (ℕ × ℕ × ℕ)) : Set (Run Ω S J) :=
  {x | x ∈ FailB O populations D fpr accFnr εcov
      (boundaryAfter O populations fpr accFnr x hM.1) hM.2}

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
def admittedCount (O : Oracle μ S) (b fpr accFnr : ℝ) (F P : Finset S) (ω : Ω) : Prop :=
  ((splitAcc O b fpr accFnr F P ω).2 : ℝ) * cfgAcc O fpr accFnr b
      ≤ (splitAcc O b fpr accFnr F P ω).1
    ∧ ((splitRej O b fpr accFnr F P ω).1 : ℝ)
      ≤ (splitRej O b fpr accFnr F P ω).2 * cfgRej O fpr accFnr b

lemma admittedCount_of_admitted (O : Oracle μ S) (b fpr accFnr α : ℝ) (F P : Finset S)
    (ω : Ω) (hα : α < 1 / 2)
    (hacc0 : 0 ≤ cfgAcc O fpr accFnr b) (hacc1 : cfgAcc O fpr accFnr b ≤ 1)
    (hrej0 : 0 ≤ cfgRej O fpr accFnr b) (hrej1 : cfgRej O fpr accFnr b ≤ 1)
    (h : admitted O b fpr accFnr α F P ω) : admittedCount O b fpr accFnr F P ω :=
  ⟨le_of_lt (lt_of_binomSfGe_le _ _ _ hacc0 hacc1 hα h.1),
    le_of_lt (lt_of_binomCdf_le _ _ _ hrej0 hrej1 hα h.2)⟩

/-! ### From states to budgets

The union over states is over `Hist × (ℕ × ℕ × ℕ)`, and infinitely many histories reach any
one budget while the gate's error depends only on the budget — so that union diverges.
But a state enters its failure event *only* through its boundary, and `boundaryAfter` is
clamped, so the whole thing is subsumed by a union over budgets alone with the boundary
quantified uniformly over `[s, 1−s]`. -/

lemma mem_ret {O : Oracle μ S} {populations : Finset J} {fpr accFnr indecisionLimit α : ℝ}
    {hM : Hist × (ℕ × ℕ × ℕ)} {x : Run Ω S J} :
    x ∈ ret O populations fpr accFnr indecisionLimit α hM ↔
      x ∈ retB O populations fpr accFnr indecisionLimit α
        (boundaryAfter O populations fpr accFnr x hM.1) hM.2 := by
  simp only [ret, Set.mem_setOf_eq]

lemma mem_FailAt {O : Oracle μ S} {populations : Finset J} {D : J → Measure S}
    {fpr accFnr εcov : ℝ} {hM : Hist × (ℕ × ℕ × ℕ)} {x : Run Ω S J} :
    x ∈ FailAt O populations D fpr accFnr εcov hM ↔
      x ∈ FailB O populations D fpr accFnr εcov
        (boundaryAfter O populations fpr accFnr x hM.1) hM.2 := by
  simp only [FailAt, Set.mem_setOf_eq]

/-- **Every reachable state is covered by its budget**, at some clamped boundary. -/
lemma state_subset_budget (O : Oracle μ S) (populations : Finset J) (D : J → Measure S)
    (fpr accFnr indecisionLimit α εcov : ℝ) :
    (⋃ t : Hist × (ℕ × ℕ × ℕ), ret O populations fpr accFnr indecisionLimit α t
        ∩ FailAt O populations D fpr accFnr εcov t)
      ⊆ ⋃ Mm : ℕ × ℕ × ℕ, ⋃ b ∈ Set.Icc (1 / 2 - O.η) (1 / 2 + O.η),
          retB O populations fpr accFnr indecisionLimit α b Mm
            ∩ FailB O populations D fpr accFnr εcov b Mm := by
  refine Set.iUnion_subset (fun t x hmem => ?_)
  refine Set.mem_iUnion.2 ⟨t.2, Set.mem_iUnion₂.2
    ⟨boundaryAfter O populations fpr accFnr x t.1,
      boundaryAfter_mem_Icc O populations fpr accFnr x t.1, ?_, ?_⟩⟩
  · exact (mem_ret (O := O)).mp hmem.1
  · exact (mem_FailAt (O := O)).mp hmem.2

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

/-- **Part 1, reduced to one state.**  Reachable states are countable, so the whole of
Part 1 is a per-state bound at any summable weight.  What remains is the per-state
obligation: at a *single* history and budget, a family that passes both gates is valid on
every population except with probability `w`. -/
theorem validity_of_per_state (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (fpr accFnr indecisionLimit α εcov δ : ℝ)
    (w : Hist × (ℕ × ℕ × ℕ) → ℝ) (hw0 : ∀ t, 0 ≤ w t) (hsum : Summable w)
    (hle : ∑' t, w t ≤ δ / 2)
    (hper : ∀ t, (runLaw μ D Dsf).real
      (ret O populations fpr accFnr indecisionLimit α t
        ∩ FailAt O populations D fpr accFnr εcov t) ≤ w t) :
    (runLaw μ D Dsf).real (⋃ t, ret O populations fpr accFnr indecisionLimit α t
        ∩ FailAt O populations D fpr accFnr εcov t) ≤ δ / 2 :=
  le_trans (measureReal_iUnion_le_tsum _ w hw0 hper hsum) hle

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

/-- Two independent draws coincide with probability exactly the collision mass. -/
lemma prod_diagonal_eq_collisionMass (Dj : Measure S) [IsProbabilityMeasure Dj] :
    (Dj.prod Dj).real {q : S × S | q.1 = q.2} = collisionMass Dj := by
  classical
  have hdiag : {q : S × S | q.1 = q.2} = ⋃ a : S, {((a, a) : S × S)} := by
    ext q; simp [Prod.ext_iff, eq_comm]
  have hdisj : Pairwise (Function.onFun Disjoint (fun a : S => ({(a, a)} : Set (S × S)))) := by
    intro a b hab
    simp only [Function.onFun, Set.disjoint_singleton]
    exact fun h => hab (congrArg Prod.fst h)
  rw [measureReal_def, hdiag, measure_iUnion hdisj (fun a => measurableSet_singleton _),
    collisionMass]
  rw [ENNReal.tsum_toReal_eq (fun a => by
    simp only [← Set.singleton_prod_singleton, Measure.prod_prod]
    exact ENNReal.mul_ne_top (measure_ne_top _ _) (measure_ne_top _ _))]
  refine tsum_congr (fun a => ?_)
  rw [← Set.singleton_prod_singleton, Measure.prod_prod, ENNReal.toReal_mul, sq,
    measureReal_def]

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

/-- The bad event at one budget: the gates pass and the cut is wrong, at *some* boundary
the loop could have reached. -/
noncomputable def BadB (O : Oracle μ S) (populations : Finset J) (D : J → Measure S)
    (fpr accFnr indecisionLimit α εcov : ℝ) (Mm : ℕ × ℕ × ℕ) : Set (Run Ω S J) :=
  ⋃ b ∈ Set.Icc (1 / 2 - O.η) (1 / 2 + O.η),
    retB O populations fpr accFnr indecisionLimit α b Mm
      ∩ FailB O populations D fpr accFnr εcov b Mm

/-- **Part 1, reduced to one budget.**  Budgets are `ℕ × ℕ × ℕ`, so this union does converge —
unlike the union over states, which the boundary clamp is what lets us avoid.

All that is left of Part 1 is `hper`: at one budget, uniformly over the boundary, the
gates pass on a wrong cut only with probability `w`. -/
theorem validity_of_budget (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (fpr accFnr indecisionLimit α εcov δ : ℝ)
    (w : ℕ × ℕ × ℕ → ℝ) (hw0 : ∀ Mm, 0 ≤ w Mm) (hsum : Summable w) (hle : ∑' Mm, w Mm ≤ δ / 2)
    (hper : ∀ Mm, (runLaw μ D Dsf).real
      (BadB O populations D fpr accFnr indecisionLimit α εcov Mm) ≤ w Mm) :
    (runLaw μ D Dsf).real (⋃ t, ret O populations fpr accFnr indecisionLimit α t
        ∩ FailAt O populations D fpr accFnr εcov t) ≤ δ / 2 :=
  le_trans (measureReal_mono (state_subset_budget O populations D fpr accFnr
      indecisionLimit α εcov))
    (le_trans (measureReal_iUnion_le_tsum _ w hw0 hper hsum) hle)

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
    (fpr accFnr indecisionLimit α : ℝ) (hfpr : 0 < fpr) (haccFnr : 0 < accFnr)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (pAP : ℝ) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (ρ : ℝ) (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ)
    (εcov : ℝ) (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) (hα : α < 1 / 2)
    (hρsmall : ρ ≤ εcov ^ 2 * δ) :
    ∃ w : ℕ × ℕ × ℕ → ℝ, (∀ Mm, 0 ≤ w Mm) ∧ Summable w ∧ (∑' Mm, w Mm ≤ δ / 2) ∧
      ∀ Mm, (runLaw μ D Dsf).real
        (BadB O populations D fpr accFnr indecisionLimit α εcov Mm) ≤ w Mm :=
  sorry

theorem validity_of_returned (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (fpr accFnr indecisionLimit α : ℝ) (hfpr : 0 < fpr) (haccFnr : 0 < accFnr)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (pAP : ℝ) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (ρ : ℝ) (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ)
    (εcov : ℝ) (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) (hα : α < 1 / 2)
    (hρsmall : ρ ≤ εcov ^ 2 * δ) :
    (runLaw μ D Dsf).real (⋃ hM, ret O populations fpr accFnr indecisionLimit α hM
        ∩ FailAt O populations D fpr accFnr εcov hM) ≤ δ / 2 := by
  obtain ⟨w, hw0, hsum, hle, hper⟩ := exists_budget_weight O populations D Dsf fpr accFnr
    indecisionLimit α hfpr haccFnr hsig hpop pAP hpAPPositive hpAPBound ρ hρ εcov hεcov δ hδ
    hα hρsmall
  exact validity_of_budget O populations D Dsf fpr accFnr indecisionLimit α εcov δ
    w hw0 hsum hle hper

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
    (fpr accFnr indecisionLimit α : ℝ) (hfpr : 0 < fpr) (haccFnr : 0 < accFnr)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (pAP : ℝ) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (δ : ℝ) (hδ : 0 < δ) (hindLim : 0 < indecisionLimit)
    (hslack : accFnr < indecisionLimit) :
    (runLaw μ D Dsf).real {x | ∀ hM, x ∉ ret O populations fpr accFnr indecisionLimit α hM} ≤ δ / 2 :=
  sorry

/-- **The E-L\* clustering algorithm is PAC-correct.**

With probability `≥ 1 − δ` the adaptive loop **terminates**, and the family it returns —
at whatever state it chooses to stop — preserves acceptance on `≥ 1 − εcov` of **each**
prefix population.

Nothing is fixed or idealised.  The growth schedule is not a parameter: the guarantee is
uniform over every history the algorithm might follow and every budget it might reach, so
it may optimize its own schedule.  The stopping time is likewise arbitrary.  The decision
boundary, evidence margin, thresholds and family size are recomputed at each state from
the oracle's signal, as `build_pst` computes them; the cluster is the Lloyd fixed point
against its own thresholded mean; and the run space is the concrete `runLaw`, not an
abstract space assumed to exist. -/
theorem clustering_correct (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (fpr accFnr indecisionLimit α : ℝ) (hfpr : 0 < fpr) (haccFnr : 0 < accFnr)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (pAP : ℝ) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (ρ : ℝ) (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ)
    (εcov : ℝ) (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) (hindLim : 0 < indecisionLimit)
    (hslack : accFnr < indecisionLimit) (hα : α < 1 / 2) (hρsmall : ρ ≤ εcov ^ 2 * δ) :
    1 - δ ≤ (runLaw μ D Dsf).real
      {x | (∃ hM, x ∈ ret O populations fpr accFnr indecisionLimit α hM) ∧
        ∀ hM : Hist × (ℕ × ℕ × ℕ), x ∈ ret O populations fpr accFnr indecisionLimit α hM →
          ∀ j ∈ populations, 1 - εcov
            ≤ (D j).real {p | cutCorrect O (boundaryAfter O populations fpr accFnr x hM.1)
                fpr accFnr (famAt O populations fpr accFnr x hM.1 hM.2) p (nz x)}} := by
  have h := sound_and_terminating (runLaw μ D Dsf)
    (fun hM => ret O populations fpr accFnr indecisionLimit α hM
      ∩ FailAt O populations D fpr accFnr εcov hM)
    (ret O populations fpr accFnr indecisionLimit α) δ
    (validity_of_returned O populations D Dsf fpr accFnr indecisionLimit α hfpr haccFnr hsig hpop
      pAP hpAPPositive hpAPBound ρ hρ εcov hεcov δ hδ hα hρsmall)
    (loop_terminates O populations D Dsf fpr accFnr indecisionLimit α hfpr haccFnr hsig hpop
      pAP hpAPPositive hpAPBound δ hδ hindLim hslack)
  refine le_trans h (le_of_eq ?_)
  congr 1
  ext x
  simp only [Set.mem_setOf_eq, Set.mem_inter_iff, FailAt, FailB, famAt, not_and, not_not]

#print axioms validity_of_returned
#print axioms loop_terminates
#print axioms clustering_correct

end Loop

end OrthoDFA
