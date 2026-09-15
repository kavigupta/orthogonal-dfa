import OrthoDFA.Liveness
import OrthoDFA.Complexity
import OrthoDFA.Termination

/-!
# The distributional clustering guarantee (PR #257) — target statement + proof

The clustering algorithm's own guarantee, distributional and **per population**, as in
PR #257 ("Hold every prefix population to the FNR limit"):

* prefixes come from a **collection** of distributions `D : J → Measure S` (the uniform
  pool, the boundary set, one per state) held individually;
* candidate suffixes are drawn from a suffix distribution `Dsf`, and **findability** is a
  single number `pAP` — the probability a drawn suffix is accept-preserving on the true
  noiseless oracle (`∀ p, ℓ(p·v) = ℓ(p)`);
* the seed is `ε = 1` (so accept-preserving = the seed's Nerode class);
* the guarantee: w.p. `≥ 1 − δ`, the returned family preserves acceptance on `≥ 1 − εcov`
  of **each** population `D j`.

This file builds the proof from reusable pieces.  `coverage` is the first: if every
family member flips at most `β` of a population's mass, the family preserves `≥ 1 − #F·β`
of it — a clean union bound, the step that turns per-suffix flip control into the
per-population fraction.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [MeasurableSpace S] [Monoid S]

/-- **Coverage.**  If every suffix in the family `F` flips at most `β` of population
`Dj`'s prefix mass, then the family preserves acceptance on at least `1 − #F·β` of `Dj`:
a union bound over the family's flip sets. -/
theorem coverage (O : Oracle μ S)
    (Dj : Measure S) [IsProbabilityMeasure Dj]
    (F : Finset S) (β : ℝ)
    (hmeas : ∀ v ∈ F, MeasurableSet {p | O.label (p * v) ≠ O.label p})
    (hF : ∀ v ∈ F, Dj.real {p | O.label (p * v) ≠ O.label p} ≤ β) :
    1 - (F.card : ℝ) * β ≤ Dj.real {p | ∀ v ∈ F, O.label (p * v) = O.label p} := by
  classical
  -- the "some family member flips p" set is the finite union of the per-suffix flip sets
  have hunion : {p | ¬ ∀ v ∈ F, O.label (p * v) = O.label p}
      = ⋃ v ∈ F, {p | O.label (p * v) ≠ O.label p} := by
    ext p; simp only [Set.mem_setOf_eq, Set.mem_iUnion, exists_prop]
    push_neg; rfl
  have hbad : Dj.real {p | ¬ ∀ v ∈ F, O.label (p * v) = O.label p} ≤ (F.card : ℝ) * β := by
    rw [hunion]
    calc Dj.real (⋃ v ∈ F, {p | O.label (p * v) ≠ O.label p})
        ≤ ∑ v ∈ F, Dj.real {p | O.label (p * v) ≠ O.label p} := measureReal_biUnion_le _ _
      _ ≤ ∑ _v ∈ F, β := Finset.sum_le_sum hF
      _ = (F.card : ℝ) * β := by rw [Finset.sum_const, nsmul_eq_mul]
  -- the preservation set is the complement of the flip set
  have hcompl : Dj.real {p | ∀ v ∈ F, O.label (p * v) = O.label p}
      = 1 - Dj.real {p | ¬ ∀ v ∈ F, O.label (p * v) = O.label p} := by
    have hmeasBad : MeasurableSet {p | ¬ ∀ v ∈ F, O.label (p * v) = O.label p} := by
      rw [hunion]; exact F.measurableSet_biUnion (fun v hv => hmeas v hv)
    have huniv : Dj.real Set.univ = 1 := by simp [measureReal_def, measure_univ]
    have hset : {p | ∀ v ∈ F, O.label (p * v) = O.label p}
        = {p | ¬ ∀ v ∈ F, O.label (p * v) = O.label p}ᶜ := by
      ext p; simp
    rw [hset, measureReal_compl hmeasBad, huniv]
  rw [hcompl]; linarith [hbad]

#print axioms coverage

/-- **Findability.**  Over `M` i.i.d. suffix draws from `Dsf`, each accept-preserving with
probability `≥ pAP`, the probability that *none* is accept-preserving is `≤ (1−pAP)^M`.
A direct instance of the product-measure `geometric_miss`; `geom_le` then drives it below
any budget once `M ≥ log(1/·)/pAP`. -/
theorem findAP (Dsf : Measure S) [IsProbabilityMeasure Dsf]
    (M : ℕ) (pAP : ℝ) (AP : Set S) (hAPmeas : MeasurableSet AP)
    (hpAP1 : pAP ≤ 1) (hfind : pAP ≤ Dsf.real AP) :
    (Measure.pi (fun _ : Fin M => Dsf)).real {x | ∀ i, x i ∉ AP} ≤ (1 - pAP) ^ M :=
  geometric_miss (fun _ => Dsf) (fun _ => AP) (fun _ => hAPmeas) pAP hpAP1 (fun _ => hfind)

#print axioms findAP

section Draws
variable [MeasurableMul S]

/-- The read at a **drawn** prefix: `flip ⊕ noise` on the concatenated query string,
as a function of one `(prefix, noise)` draw.  Modelling a draw as an independent
`(prefix, noise)` pair is what makes the reads independent *derivably*. -/
noncomputable def rdAt (O : Oracle μ S) (v : S) : S × Ω → ℝ :=
  fun z => O.flip v z.1 + (1 - 2 * O.flip v z.1) * O.noise (z.1 * v) z.2

/-- The `Dj`-flip-mass of a suffix: the probability that `v` flips a `Dj`-drawn prefix.
This is the *distributional* quantity `good`/`bad` are defined by. -/
noncomputable def flipMass (O : Oracle μ S) (Dj : Measure S) (v : S) : ℝ :=
  ∫ p, O.flip v p ∂Dj

lemma flip_meas (O : Oracle μ S) (v : S) : Measurable (fun p => O.flip v p) := by
  have h1 : Measurable (fun p : S => O.label (p * v)) :=
    O.label_meas.comp (measurable_mul_const v)
  show Measurable (fun p => O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p)
  exact (h1.add O.label_meas).sub ((measurable_const.mul h1).mul O.label_meas)

lemma flip_icc (O : Oracle μ S) (v p : S) : O.flip v p ∈ Set.Icc (0 : ℝ) 1 := by
  rcases O.flip_bit v p with h | h <;> rw [Set.mem_Icc, h] <;> constructor <;> norm_num

lemma rdAt_meas (O : Oracle μ S) (v : S) : Measurable (rdAt O v) := by
  have hf : Measurable (fun z : S × Ω => O.flip v z.1) := (flip_meas O v).comp measurable_fst
  have hn : Measurable (fun z : S × Ω => O.noise (z.1 * v) z.2) :=
    O.noise_meas.comp (((measurable_mul_const v).comp measurable_fst).prodMk measurable_snd)
  show Measurable (fun z : S × Ω =>
    O.flip v z.1 + (1 - 2 * O.flip v z.1) * O.noise (z.1 * v) z.2)
  exact hf.add ((measurable_const.sub (measurable_const.mul hf)).mul hn)

lemma rdAt_icc (O : Oracle μ S) (Dj : Measure S) [SFinite Dj] (v : S) :
    ∀ᵐ z ∂(Dj.prod μ), rdAt O v z ∈ Set.Icc (0 : ℝ) 1 := by
  rw [Measure.ae_prod_iff_ae_ae ((rdAt_meas O v) measurableSet_Icc)]
  filter_upwards with p
  filter_upwards [O.noise_icc (p * v)] with ω hω
  rw [Set.mem_Icc] at hω
  show O.flip v p + (1 - 2 * O.flip v p) * O.noise (p * v) ω ∈ Set.Icc (0 : ℝ) 1
  rw [Set.mem_Icc]
  rcases O.flip_bit v p with h | h <;> rw [h] <;> constructor <;> nlinarith [hω.1, hω.2]

/-- **The drawn read's mean is the distributional flip-mass.**  Averaging over both the
prefix draw and the noise, `E[rdAt] = η + (1−2η)·flipMass`.  This is the one place the
prefix distribution enters: the empirical loss estimates the `Dj`-mass directly, so a
single Hoeffding level suffices (no separate sample→distribution step). -/
lemma rdAt_mean (O : Oracle μ S) (Dj : Measure S) [IsProbabilityMeasure Dj] (v : S) :
    ∫ z, rdAt O v z ∂(Dj.prod μ) = O.η + (1 - 2 * O.η) * flipMass O Dj v := by
  have hint : Integrable (rdAt O v) (Dj.prod μ) :=
    MeasureTheory.Integrable.of_mem_Icc 0 1 (rdAt_meas O v).aemeasurable (rdAt_icc O Dj v)
  rw [integral_prod _ hint]
  have hinner : ∀ p, ∫ ω, rdAt O v (p, ω) ∂μ = O.η + O.flip v p * (1 - 2 * O.η) := fun p =>
    read_disagreement_mean μ O.η (O.flip v p) (O.noise (p * v)) (O.noise_int _) (O.noise_mean _)
  rw [integral_congr_ae (Filter.Eventually.of_forall hinner)]
  have hflipint : Integrable (fun p => O.flip v p) Dj :=
    MeasureTheory.Integrable.of_mem_Icc 0 1 (flip_meas O v).aemeasurable
      (Filter.Eventually.of_forall (flip_icc O v))
  rw [integral_add (integrable_const _) (hflipint.mul_const _), integral_const,
    integral_mul_const]
  simp only [flipMass, measureReal_def, measure_univ, ENNReal.toReal_one, smul_eq_mul, one_mul]
  ring

/-- **Selection (distributional, per-coordinate populations).**  Each draw `z : ι` is an
independent `(prefix, noise)` pair whose prefix comes from that coordinate's population
`Dfam z` (for the multi-population setting, `ι = J × Fin m` with coordinate `(j,i)` drawn
from `D j`).  The greedy's least-loss `k`-subset then avoids every `bad` suffix, where
`good`/`bad` are stated by the **summed distributional flip-mass** `∑ z, flipMass (Dfam z)`.

Separation is definitional; independence of the reads is derived from the product
(`iIndepFun_pi`), and each read's mean is its coordinate's flip-mass (`rdAt_mean`). -/
theorem selection [DecidableEq S] {ι : Type*} [Fintype ι] (O : Oracle μ S)
    (Dfam : ι → Measure S) [∀ z, IsProbabilityMeasure (Dfam z)]
    (good bad : S → Prop) [DecidablePred good] [DecidablePred bad]
    (hdisj : ∀ v, bad v → ¬ good v)
    (cands : Finset S) (k : ℕ) (σ : ℝ) (hσ0 : 0 ≤ σ)
    (hgoodmass : ∀ v ∈ cands, good v → ∑ z, flipMass O (Dfam z) v = 0)
    (hbadmass : ∀ v ∈ cands, bad v → σ ≤ ∑ z, flipMass O (Dfam z) v)
    (goodCount : k ≤ (cands.filter good).card)
    (chosen : ((z : ι) → S × Ω) → Finset S)
    (hsub : ∀ x, chosen x ⊆ cands) (hcard : ∀ x, (chosen x).card = k)
    (hleast : ∀ x, ∀ v ∈ chosen x, ∀ w ∈ cands, w ∉ chosen x →
        (∑ z, rdAt O v (x z)) ≤ ∑ z, rdAt O w (x z)) :
    (Measure.pi (fun z : ι => (Dfam z).prod μ)).real {x | ¬ ∀ w ∈ chosen x, ¬ bad w}
      ≤ (cands.card : ℝ)
          * Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ)
              * ((1 / 2 - O.η) * σ / ((Finset.univ : Finset ι).card : ℝ)) ^ 2) := by
  set νs : ι → Measure (S × Ω) := fun z => (Dfam z).prod μ with hνs
  set X : S → ι → ((z : ι) → S × Ω) → ℝ := fun v z x => rdAt O v (x z) with hX
  set N : ℝ := ((Finset.univ : Finset ι).card : ℝ) with hN
  have hmarg : ∀ (v : S) (z : ι),
      (Measure.pi νs)[X v z] = O.η + (1 - 2 * O.η) * flipMass O (Dfam z) v := by
    intro v z
    have hmap : Measure.map (fun x : (z : ι) → S × Ω => x z) (Measure.pi νs) = (Dfam z).prod μ :=
      (measurePreserving_eval νs z).map_eq
    have h : (Measure.pi νs)[X v z] = ∫ y, rdAt O v y ∂((Dfam z).prod μ) := by
      rw [← hmap, integral_map (measurable_pi_apply z).aemeasurable
        (rdAt_meas O v).aestronglyMeasurable]
    rw [h, rdAt_mean O (Dfam z) v]
  have hmeas : ∀ v z, AEMeasurable (X v z) (Measure.pi νs) := fun v z =>
    ((rdAt_meas O v).comp (measurable_pi_apply z)).aemeasurable
  have hindep : ∀ v, iIndepFun (X v) (Measure.pi νs) := fun v =>
    iIndepFun_pi (fun _ => (rdAt_meas O v).aemeasurable)
  have hIcc : ∀ v z, ∀ᵐ x ∂(Measure.pi νs), X v z x ∈ Set.Icc (0 : ℝ) 1 := fun v z =>
    (measurePreserving_eval νs z).quasiMeasurePreserving.ae (rdAt_icc O (Dfam z) v)
  have hsum : ∀ v, ∑ z ∈ (Finset.univ : Finset ι), (Measure.pi νs)[X v z]
      = N * O.η + (1 - 2 * O.η) * ∑ z, flipMass O (Dfam z) v := by
    intro v
    rw [Finset.sum_congr rfl (fun z _ => hmarg v z), Finset.sum_add_distrib, Finset.sum_const,
      nsmul_eq_mul, ← Finset.mul_sum, hN]
  have hgm : ∀ v ∈ cands, good v →
      ∑ z ∈ (Finset.univ : Finset ι), (Measure.pi νs)[X v z] ≤ N * O.η := by
    intro v hv hg; rw [hsum v, hgoodmass v hv hg]; simp
  have hbm : ∀ v ∈ cands, bad v →
      N * (O.η + (1 - 2 * O.η) * σ / N)
        ≤ ∑ z ∈ (Finset.univ : Finset ι), (Measure.pi νs)[X v z] := by
    intro v hv hb
    rw [hsum v]
    have hmass := hbadmass v hv hb
    have h1 : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith [O.hη]
    have h2 : (1 - 2 * O.η) * σ ≤ (1 - 2 * O.η) * ∑ z, flipMass O (Dfam z) v :=
      mul_le_mul_of_nonneg_left hmass h1
    have hA0 : (0 : ℝ) ≤ (1 - 2 * O.η) * σ := mul_nonneg h1 hσ0
    have hkey : N * ((1 - 2 * O.η) * σ / N) ≤ (1 - 2 * O.η) * σ := by
      rcases eq_or_lt_of_le (show (0:ℝ) ≤ N from Nat.cast_nonneg _) with hN0 | hNpos
      · rw [← hN0]; simpa using hA0
      · rw [mul_div_cancel₀ _ (ne_of_gt hNpos)]
    have hexp : N * (O.η + (1 - 2 * O.η) * σ / N)
        = N * O.η + N * ((1 - 2 * O.η) * σ / N) := by ring
    rw [hexp]
    linarith [hkey, h2]
  have hgap : O.η + (1 / 2 - O.η) * σ / N
      ≤ (O.η + (1 - 2 * O.η) * σ / N) - (1 / 2 - O.η) * σ / N := by
    have : (1 - 2 * O.η) * σ / N = 2 * ((1 / 2 - O.η) * σ / N) := by ring
    rw [this]; linarith
  have hγ : (0 : ℝ) ≤ (1 / 2 - O.η) * σ / N :=
    div_nonneg (mul_nonneg (by linarith [O.hη]) hσ0) (Nat.cast_nonneg _)
  exact chosen_avoids_bad_whp good bad hdisj cands k (Finset.univ : Finset ι)
    O.η (O.η + (1 - 2 * O.η) * σ / N) ((1 / 2 - O.η) * σ / N) X hmeas hindep hIcc hgm hbm
    hgap hγ goodCount chosen hsub hcard hleast

#print axioms rdAt_mean
#print axioms selection

/-- `flip` is the indicator of "the labels of `p·v` and `p` differ". -/
lemma flip_eq_one_iff (O : Oracle μ S) (v p : S) :
    O.flip v p = 1 ↔ O.label (p * v) ≠ O.label p := by
  rcases O.label_bit (p * v) with h1 | h1 <;> rcases O.label_bit p with h2 | h2 <;>
    simp only [Oracle.flip, h1, h2] <;> norm_num

lemma flipSet_meas (O : Oracle μ S) (v : S) :
    MeasurableSet {p | O.label (p * v) ≠ O.label p} := by
  have : {p | O.label (p * v) ≠ O.label p} = (fun p => O.flip v p) ⁻¹' {1} := by
    ext p; simpa using (flip_eq_one_iff O v p).symm
  rw [this]; exact (flip_meas O v) (measurableSet_singleton 1)

/-- **The flip-mass is the measure of the flip set**: `∫ flip ∂Dj = Dj{p | ℓ(p·v) ≠ ℓ(p)}`.
This is what lets the selection guarantee (stated in `flipMass`) feed `coverage`
(stated as a measure). -/
lemma flipMass_eq (O : Oracle μ S) (Dj : Measure S) [IsProbabilityMeasure Dj] (v : S) :
    flipMass O Dj v = Dj.real {p | O.label (p * v) ≠ O.label p} := by
  have hind : (fun p => O.flip v p)
      = Set.indicator {p | O.label (p * v) ≠ O.label p} (fun _ => (1 : ℝ)) := by
    funext p
    by_cases h : O.label (p * v) = O.label p
    · rw [Set.indicator_of_notMem (by simpa using h)]
      rcases O.label_bit p with h2 | h2 <;>
        simp only [Oracle.flip, h, h2] <;> norm_num
    · rw [Set.indicator_of_mem (by simpa using h)]
      exact (flip_eq_one_iff O v p).mpr h
  rw [flipMass, hind, integral_indicator (flipSet_meas O v), setIntegral_const,
    smul_eq_mul, mul_one, measureReal_def]

lemma flipMass_nonneg (O : Oracle μ S) (Dj : Measure S) [IsProbabilityMeasure Dj] (v : S) :
    0 ≤ flipMass O Dj v := by
  rw [flipMass_eq]; exact measureReal_nonneg

/-- **Selection ⇒ per-population coverage.**  If every family member's *summed*
flip-mass over the populations is below `εfam`, then on **each** population the family
preserves acceptance on at least a `1 − #F·εfam` fraction.  (Summed control gives
per-population control because flip-masses are nonnegative.) -/
theorem coverage_of_summed_flip [DecidableEq S] {J : Type*} (O : Oracle μ S)
    (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    (populations : Finset J) (F : Finset S) (εfam : ℝ)
    (hF : ∀ w ∈ F, ∑ j ∈ populations, flipMass O (D j) w ≤ εfam) :
    ∀ j ∈ populations,
      1 - (F.card : ℝ) * εfam ≤ (D j).real {p | ∀ v ∈ F, O.label (p * v) = O.label p} := by
  intro j hj
  refine coverage O (D j) F εfam (fun v _ => flipSet_meas O v) (fun v hv => ?_)
  rw [← flipMass_eq]
  refine le_trans ?_ (hF v hv)
  exact Finset.single_le_sum (f := fun j' => flipMass O (D j') v)
    (fun j' _ => flipMass_nonneg O (D j') v) hj

#print axioms flipMass_eq
#print axioms coverage_of_summed_flip

end Draws

end OrthoDFA
