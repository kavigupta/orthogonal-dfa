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

/-- **Uniform slice bound transfers to the product.**  If every slice of a measurable
product event has probability `≤ ε`, so does the event.  This is the plumbing that lets a
bound proved for each *fixed* candidate pool be used when the pool is itself drawn. -/
theorem prod_le_of_slice {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (ma : Measure α) [IsProbabilityMeasure ma] (mb : Measure β) [IsProbabilityMeasure mb]
    (E : Set (α × β)) (hE : MeasurableSet E) (ε : ℝ) (hε : 0 ≤ ε)
    (hslice : ∀ a, mb.real (Prod.mk a ⁻¹' E) ≤ ε) :
    (ma.prod mb).real E ≤ ε := by
  have hle : ∀ a, mb (Prod.mk a ⁻¹' E) ≤ ENNReal.ofReal ε := by
    intro a
    rw [← ENNReal.ofReal_toReal (measure_ne_top mb (Prod.mk a ⁻¹' E))]
    exact ENNReal.ofReal_le_ofReal (hslice a)
  have hmain : (ma.prod mb) E ≤ ENNReal.ofReal ε := by
    rw [Measure.prod_apply hE]
    calc ∫⁻ a, mb (Prod.mk a ⁻¹' E) ∂ma ≤ ∫⁻ _a, ENNReal.ofReal ε ∂ma := lintegral_mono hle
      _ = ENNReal.ofReal ε := by simp
  calc (ma.prod mb).real E = ((ma.prod mb) E).toReal := rfl
    _ ≤ (ENNReal.ofReal ε).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hmain
    _ = ε := ENNReal.toReal_ofReal hε

#print axioms prod_le_of_slice

/-- First-marginal probability of a product event depending only on the first coordinate. -/
theorem prod_fst_real {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (ma : Measure α) [IsProbabilityMeasure ma] (ma' : Measure β) [IsProbabilityMeasure ma']
    (B : Set α) : (ma.prod ma').real {x : α × β | x.1 ∈ B} = ma.real B := by
  have hset : {x : α × β | x.1 ∈ B} = B ×ˢ (Set.univ : Set β) := by ext x; simp
  rw [measureReal_def, hset, Measure.prod_prod, measure_univ, mul_one, ← measureReal_def]


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

section Persistent
variable {ι : Type*} [Fintype ι] [IsCancelMul S]

/-- Run space for the **persistent** oracle: `ι` prefix draws together with **one shared**
noise sample.  Re-reading the same query string returns the same bit — this is the honest
RCN model, unlike a fresh-noise-per-query product. -/
noncomputable def runMeasure (Dfam : ι → Measure S) : Measure ((ι → S) × Ω) :=
  (Measure.pi Dfam).prod μ

instance (Dfam : ι → Measure S) [∀ z, IsProbabilityMeasure (Dfam z)] :
    IsProbabilityMeasure (runMeasure (μ := μ) Dfam) := by
  unfold runMeasure; infer_instance

/-- The loss of suffix `v`: its reads at the drawn prefixes, against the shared noise. -/
noncomputable def ploss (O : Oracle μ S) (v : S) (x : (ι → S) × Ω) : ℝ :=
  ∑ z, O.read x.1 v z x.2

/-- The flip count of `v` on the drawn prefixes (a function of the draw alone). -/
noncomputable def pflip (O : Oracle μ S) (v : S) (x : (ι → S) × Ω) : ℝ :=
  ∑ z, O.flip v (x.1 z)

variable [Countable S] [MeasurableSingletonClass S]

lemma pread_meas (O : Oracle μ S) (v : S) (z : ι) :
    Measurable (fun x : (ι → S) × Ω => O.read x.1 v z x.2) := by
  have hp : Measurable (fun x : (ι → S) × Ω => x.1 z) :=
    (measurable_pi_apply z).comp measurable_fst
  have hf : Measurable (fun x : (ι → S) × Ω => O.flip v (x.1 z)) :=
    (flip_meas O v).comp hp
  have hn : Measurable (fun x : (ι → S) × Ω => O.noise (x.1 z * v) x.2) :=
    O.noise_meas.comp (((measurable_mul_const v).comp hp).prodMk measurable_snd)
  show Measurable (fun x : (ι → S) × Ω =>
    O.flip v (x.1 z) + (1 - 2 * O.flip v (x.1 z)) * O.noise (x.1 z * v) x.2)
  exact hf.add ((measurable_const.sub (measurable_const.mul hf)).mul hn)

lemma ploss_meas (O : Oracle μ S) (v : S) : Measurable (ploss (μ := μ) (ι := ι) O v) :=
  Finset.measurable_sum _ (fun z _ => pread_meas O v z)

lemma pflip_meas (O : Oracle μ S) (v : S) : Measurable (pflip (μ := μ) (ι := ι) O v) :=
  Finset.measurable_sum _ (fun z _ =>
    (flip_meas O v).comp ((measurable_pi_apply z).comp measurable_fst))

/-- The draws are distinct — the event on which the persistent oracle is never read
twice at the same query string. -/
lemma injective_meas : MeasurableSet {x : (ι → S) × Ω | Function.Injective x.1} := by
  classical
  have hset : {x : (ι → S) × Ω | Function.Injective x.1}
      = ⋂ z : ι, ⋂ z' : ι, {x | z = z' ∨ x.1 z ≠ x.1 z'} := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iInter]
    constructor
    · intro h z z'
      by_cases hzz : z = z'
      · exact Or.inl hzz
      · exact Or.inr (fun he => hzz (h he))
    · intro h a b hab
      rcases h a b with hz | hne
      · exact hz
      · exact absurd hab hne
  rw [hset]
  refine MeasurableSet.iInter (fun z => MeasurableSet.iInter (fun z' => ?_))
  by_cases hzz : z = z'
  · simp [hzz]
  · have : {x : (ι → S) × Ω | z = z' ∨ x.1 z ≠ x.1 z'}
        = {x : (ι → S) × Ω | x.1 z ≠ x.1 z'} := by
      ext x; simp [hzz]
    rw [this]
    have hm : Measurable (fun x : (ι → S) × Ω => (x.1 z, x.1 z')) :=
      (((measurable_pi_apply z).comp measurable_fst)).prodMk
        ((measurable_pi_apply z').comp measurable_fst)
    have hdiag : MeasurableSet {q : S × S | q.1 ≠ q.2} := by
      have : {q : S × S | q.1 ≠ q.2} = {q : S × S | q.1 = q.2}ᶜ := by ext q; simp
      rw [this]
      exact (measurableSet_eq_fun measurable_fst measurable_snd).compl
    exact hm hdiag

/-- **Level 1 (noise), conditional on distinct draws.**  Given that the draws are
distinct — so the persistent oracle is never read twice at the same query string, and its
bits really are independent (`read_indep`, via right-cancellation) — the loss concentrates
around its conditional mean `N·η + (1−2η)·(flip count)`. -/
lemma ploss_cond_upper (O : Oracle μ S) (Dfam : ι → Measure S)
    [∀ z, IsProbabilityMeasure (Dfam z)] (v : S) (g : ℝ) (hg : 0 ≤ g)
    (hNpos : 0 < ((Finset.univ : Finset ι).card : ℝ)) :
    (runMeasure (μ := μ) Dfam).real
        {x : (ι → S) × Ω | Function.Injective x.1 ∧
          ((Finset.univ : Finset ι).card : ℝ) * O.η + (1 - 2 * O.η) * pflip (μ := μ) O v x
            + ((Finset.univ : Finset ι).card : ℝ) * g ≤ ploss (μ := μ) O v x}
      ≤ Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g ^ 2) := by
  classical
  set N : ℝ := ((Finset.univ : Finset ι).card : ℝ) with hN
  refine prod_le_of_slice _ _ _ ?_ _ (Real.exp_pos _).le (fun p => ?_)
  · -- measurable
    exact (injective_meas (ι := ι) (S := S) (Ω := Ω)).inter
      (measurableSet_le (((pflip_meas O v).const_mul _).const_add _ |>.add measurable_const)
        (ploss_meas O v))
  · -- slice
    by_cases hinj : Function.Injective p
    · have hmeanEq : ∑ z, μ[O.read p v z] = N * (O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) / N) := by
        rw [Finset.sum_congr rfl (fun z _ => O.read_mean p v z), Finset.sum_add_distrib,
          Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum, ← hN]
        field_simp
      have h := sumUpper_le (O.read p v) (Finset.univ : Finset ι)
        (O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) / N) g
        (fun z => (O.read_meas p v z).aemeasurable) (O.read_indep p hinj v)
        (O.read_icc p v) (le_of_eq hmeanEq) hg
      refine le_trans (measureReal_mono ?_) h
      intro ω hω
      obtain ⟨-, hle⟩ := hω
      show N * ((O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) / N) + g) ≤ ∑ z, O.read p v z ω
      have hNne : N ≠ 0 := ne_of_gt hNpos
      have : N * ((O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) / N) + g)
          = N * O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) + N * g := by
        field_simp
      rw [this]
      exact hle
    · have : (Prod.mk p ⁻¹' {x : (ι → S) × Ω | Function.Injective x.1 ∧
          N * O.η + (1 - 2 * O.η) * pflip (μ := μ) O v x + N * g ≤ ploss (μ := μ) O v x})
          = ∅ := by
        ext ω; simp only [Set.mem_preimage, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
        rintro ⟨h, -⟩; exact hinj h
      rw [this]; simpa using (Real.exp_pos _).le

/-- Level 1, lower tail (symmetric). -/
lemma ploss_cond_lower (O : Oracle μ S) (Dfam : ι → Measure S)
    [∀ z, IsProbabilityMeasure (Dfam z)] (v : S) (g : ℝ) (hg : 0 ≤ g)
    (hNpos : 0 < ((Finset.univ : Finset ι).card : ℝ)) :
    (runMeasure (μ := μ) Dfam).real
        {x : (ι → S) × Ω | Function.Injective x.1 ∧
          ploss (μ := μ) O v x ≤ ((Finset.univ : Finset ι).card : ℝ) * O.η
            + (1 - 2 * O.η) * pflip (μ := μ) O v x
            - ((Finset.univ : Finset ι).card : ℝ) * g}
      ≤ Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g ^ 2) := by
  classical
  set N : ℝ := ((Finset.univ : Finset ι).card : ℝ) with hN
  refine prod_le_of_slice _ _ _ ?_ _ (Real.exp_pos _).le (fun p => ?_)
  · exact (injective_meas (ι := ι) (S := S) (Ω := Ω)).inter
      (measurableSet_le (ploss_meas O v)
        ((((pflip_meas O v).const_mul _).const_add _).sub measurable_const))
  · by_cases hinj : Function.Injective p
    · have hmeanEq : ∑ z, μ[O.read p v z]
          = N * (O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) / N) := by
        rw [Finset.sum_congr rfl (fun z _ => O.read_mean p v z), Finset.sum_add_distrib,
          Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum, ← hN]
        field_simp
      have h := sumLower_le (O.read p v) (Finset.univ : Finset ι)
        (O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) / N) g
        (fun z => (O.read_meas p v z).aemeasurable) (O.read_indep p hinj v)
        (O.read_icc p v) (ge_of_eq hmeanEq) hg
      refine le_trans (measureReal_mono ?_) h
      intro ω hω
      obtain ⟨-, hle⟩ := hω
      show ∑ z, O.read p v z ω
        ≤ N * ((O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) / N) - g)
      have hNne : N ≠ 0 := ne_of_gt hNpos
      have hrw : N * ((O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) / N) - g)
          = N * O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) - N * g := by field_simp
      rw [hrw]; exact hle
    · have : (Prod.mk p ⁻¹' {x : (ι → S) × Ω | Function.Injective x.1 ∧
          ploss (μ := μ) O v x ≤ N * O.η + (1 - 2 * O.η) * pflip (μ := μ) O v x - N * g})
          = ∅ := by
        ext ω; simp only [Set.mem_preimage, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
        rintro ⟨h, -⟩; exact hinj h
      rw [this]; simpa using (Real.exp_pos _).le

/-- The flip count's mean on the prefix marginal is the summed flip-mass. -/
lemma pflip_meanSum (O : Oracle μ S) (Dfam : ι → Measure S)
    [∀ z, IsProbabilityMeasure (Dfam z)] (v : S) :
    ∑ z, (Measure.pi Dfam)[fun p : ι → S => O.flip v (p z)]
      = ∑ z, flipMass O (Dfam z) v := by
  refine Finset.sum_congr rfl (fun z _ => ?_)
  have hmap : Measure.map (fun p : ι → S => p z) (Measure.pi Dfam) = Dfam z :=
    (measurePreserving_eval Dfam z).map_eq
  calc (Measure.pi Dfam)[fun p : ι → S => O.flip v (p z)]
      = ∫ w, O.flip v w ∂(Measure.map (fun p : ι → S => p z) (Measure.pi Dfam)) := by
        rw [integral_map (measurable_pi_apply z).aemeasurable
          (flip_meas O v).aestronglyMeasurable]
    _ = ∫ w, O.flip v w ∂(Dfam z) := by rw [hmap]
    _ = flipMass O (Dfam z) v := rfl

/-- **Level 2 (sampling), upper tail.**  The flip count on the drawn prefixes concentrates
above its summed distributional flip-mass. -/
lemma pflip_upper (O : Oracle μ S) (Dfam : ι → Measure S)
    [∀ z, IsProbabilityMeasure (Dfam z)] (v : S) (g : ℝ) (hg : 0 ≤ g)
    (hNpos : 0 < ((Finset.univ : Finset ι).card : ℝ)) :
    (runMeasure (μ := μ) Dfam).real
        {x : (ι → S) × Ω | (∑ z, flipMass O (Dfam z) v)
          + ((Finset.univ : Finset ι).card : ℝ) * g ≤ pflip (μ := μ) O v x}
      ≤ Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g ^ 2) := by
  set N : ℝ := ((Finset.univ : Finset ι).card : ℝ) with hN
  have hNne : N ≠ 0 := ne_of_gt hNpos
  have heq : (runMeasure (μ := μ) Dfam).real
      {x : (ι → S) × Ω | (∑ z, flipMass O (Dfam z) v) + N * g ≤ pflip (μ := μ) O v x}
      = (Measure.pi Dfam).real
        {p : ι → S | (∑ z, flipMass O (Dfam z) v) + N * g ≤ ∑ z, O.flip v (p z)} := by
    unfold runMeasure
    exact prod_fst_real (Measure.pi Dfam) μ
      {p : ι → S | (∑ z, flipMass O (Dfam z) v) + N * g ≤ ∑ z, O.flip v (p z)}
  rw [heq]
  have hmean : ∑ z, (Measure.pi Dfam)[fun p : ι → S => O.flip v (p z)]
      ≤ N * ((∑ z, flipMass O (Dfam z) v) / N) := by
    have hc : N * ((∑ z, flipMass O (Dfam z) v) / N) = ∑ z, flipMass O (Dfam z) v := by
      field_simp
    rw [hc]; exact le_of_eq (pflip_meanSum O Dfam v)
  have h := sumUpper_le (fun (z : ι) (p : ι → S) => O.flip v (p z)) (Finset.univ : Finset ι)
    ((∑ z, flipMass O (Dfam z) v) / N) g
    (fun z => ((flip_meas O v).comp (measurable_pi_apply z)).aemeasurable)
    (iIndepFun_pi (fun _ => (flip_meas O v).aemeasurable))
    (fun z => Filter.Eventually.of_forall (fun p => flip_icc O v (p z))) hmean hg
  refine le_trans (measureReal_mono ?_) h
  intro q hq
  show N * ((∑ z, flipMass O (Dfam z) v) / N + g) ≤ ∑ z, O.flip v (q z)
  have hrw : N * ((∑ z, flipMass O (Dfam z) v) / N + g)
      = (∑ z, flipMass O (Dfam z) v) + N * g := by field_simp
  rw [hrw]; exact hq

/-- **Level 2 (sampling), lower tail.** -/
lemma pflip_lower (O : Oracle μ S) (Dfam : ι → Measure S)
    [∀ z, IsProbabilityMeasure (Dfam z)] (v : S) (g : ℝ) (hg : 0 ≤ g)
    (hNpos : 0 < ((Finset.univ : Finset ι).card : ℝ)) :
    (runMeasure (μ := μ) Dfam).real
        {x : (ι → S) × Ω | pflip (μ := μ) O v x
          ≤ (∑ z, flipMass O (Dfam z) v) - ((Finset.univ : Finset ι).card : ℝ) * g}
      ≤ Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g ^ 2) := by
  set N : ℝ := ((Finset.univ : Finset ι).card : ℝ) with hN
  have hNne : N ≠ 0 := ne_of_gt hNpos
  have heq : (runMeasure (μ := μ) Dfam).real
      {x : (ι → S) × Ω | pflip (μ := μ) O v x ≤ (∑ z, flipMass O (Dfam z) v) - N * g}
      = (Measure.pi Dfam).real
        {p : ι → S | ∑ z, O.flip v (p z) ≤ (∑ z, flipMass O (Dfam z) v) - N * g} := by
    unfold runMeasure
    exact prod_fst_real (Measure.pi Dfam) μ
      {p : ι → S | ∑ z, O.flip v (p z) ≤ (∑ z, flipMass O (Dfam z) v) - N * g}
  rw [heq]
  have hmean : N * ((∑ z, flipMass O (Dfam z) v) / N)
      ≤ ∑ z, (Measure.pi Dfam)[fun p : ι → S => O.flip v (p z)] := by
    have hc : N * ((∑ z, flipMass O (Dfam z) v) / N) = ∑ z, flipMass O (Dfam z) v := by
      field_simp
    rw [hc]; exact le_of_eq (pflip_meanSum O Dfam v).symm
  have h := sumLower_le (fun (z : ι) (p : ι → S) => O.flip v (p z)) (Finset.univ : Finset ι)
    ((∑ z, flipMass O (Dfam z) v) / N) g
    (fun z => ((flip_meas O v).comp (measurable_pi_apply z)).aemeasurable)
    (iIndepFun_pi (fun _ => (flip_meas O v).aemeasurable))
    (fun z => Filter.Eventually.of_forall (fun p => flip_icc O v (p z))) hmean hg
  refine le_trans (measureReal_mono ?_) h
  intro q hq
  show ∑ z, O.flip v (q z) ≤ N * ((∑ z, flipMass O (Dfam z) v) / N - g)
  have hrw : N * ((∑ z, flipMass O (Dfam z) v) / N - g)
      = (∑ z, flipMass O (Dfam z) v) - N * g := by field_simp
  rw [hrw]; exact hq

/-- **Per-suffix upper tail, persistent oracle.**  A suffix with zero summed flip-mass
keeps its loss below the band, except for three sources: a **collision** (the persistent
oracle read twice at the same query string — which is why a spread-out prefix
distribution is genuinely necessary), the sampling tail, and the noise tail. -/
lemma ploss_good_upper (O : Oracle μ S) (Dfam : ι → Measure S)
    [∀ z, IsProbabilityMeasure (Dfam z)] (v : S) (g₁ g₂ κ : ℝ) (hg₁ : 0 ≤ g₁) (hg₂ : 0 ≤ g₂)
    (hNpos : 0 < ((Finset.univ : Finset ι).card : ℝ))
    (hcoll : (runMeasure (μ := μ) Dfam).real
      {x : (ι → S) × Ω | ¬ Function.Injective x.1} ≤ κ)
    (hv : ∑ z, flipMass O (Dfam z) v = 0) :
    (runMeasure (μ := μ) Dfam).real
        {x : (ι → S) × Ω | ((Finset.univ : Finset ι).card : ℝ)
            * (O.η + (1 - 2 * O.η) * g₂ + g₁) ≤ ploss (μ := μ) O v x}
      ≤ κ + Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g₂ ^ 2)
          + Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g₁ ^ 2) := by
  have h2η : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith [O.hη]
  have hincl : {x : (ι → S) × Ω | ((Finset.univ : Finset ι).card : ℝ)
        * (O.η + (1 - 2 * O.η) * g₂ + g₁) ≤ ploss (μ := μ) O v x}
      ⊆ {x : (ι → S) × Ω | ¬ Function.Injective x.1}
        ∪ ({x : (ι → S) × Ω | (∑ z, flipMass O (Dfam z) v)
              + ((Finset.univ : Finset ι).card : ℝ) * g₂ ≤ pflip (μ := μ) O v x}
          ∪ {x : (ι → S) × Ω | Function.Injective x.1 ∧
              ((Finset.univ : Finset ι).card : ℝ) * O.η
                + (1 - 2 * O.η) * pflip (μ := μ) O v x
                + ((Finset.univ : Finset ι).card : ℝ) * g₁ ≤ ploss (μ := μ) O v x}) := by
    intro x hx
    by_cases hinj : Function.Injective x.1
    · right
      by_cases hfl : (∑ z, flipMass O (Dfam z) v)
          + ((Finset.univ : Finset ι).card : ℝ) * g₂ ≤ pflip (μ := μ) O v x
      · exact Or.inl hfl
      · refine Or.inr ⟨hinj, ?_⟩
        push_neg at hfl
        rw [hv, zero_add] at hfl
        have hkey : (1 - 2 * O.η) * pflip (μ := μ) O v x
            ≤ (1 - 2 * O.η) * (((Finset.univ : Finset ι).card : ℝ) * g₂) :=
          mul_le_mul_of_nonneg_left hfl.le h2η
        simp only [Set.mem_setOf_eq] at hx ⊢
        linarith [hx, hkey]
    · exact Or.inl hinj
  have hu1 := measureReal_mono (μ := runMeasure (μ := μ) Dfam) hincl
  have hu2 := measureReal_union_le (μ := runMeasure (μ := μ) Dfam)
    {x : (ι → S) × Ω | ¬ Function.Injective x.1}
    ({x : (ι → S) × Ω | (∑ z, flipMass O (Dfam z) v)
        + ((Finset.univ : Finset ι).card : ℝ) * g₂ ≤ pflip (μ := μ) O v x}
      ∪ {x : (ι → S) × Ω | Function.Injective x.1 ∧
          ((Finset.univ : Finset ι).card : ℝ) * O.η + (1 - 2 * O.η) * pflip (μ := μ) O v x
            + ((Finset.univ : Finset ι).card : ℝ) * g₁ ≤ ploss (μ := μ) O v x})
  have hu3 := measureReal_union_le (μ := runMeasure (μ := μ) Dfam)
    {x : (ι → S) × Ω | (∑ z, flipMass O (Dfam z) v)
        + ((Finset.univ : Finset ι).card : ℝ) * g₂ ≤ pflip (μ := μ) O v x}
    {x : (ι → S) × Ω | Function.Injective x.1 ∧
      ((Finset.univ : Finset ι).card : ℝ) * O.η + (1 - 2 * O.η) * pflip (μ := μ) O v x
        + ((Finset.univ : Finset ι).card : ℝ) * g₁ ≤ ploss (μ := μ) O v x}
  linarith [hu1, hu2, hu3, hcoll, pflip_upper O Dfam v g₂ hg₂ hNpos,
    ploss_cond_upper O Dfam v g₁ hg₁ hNpos]

/-- **Per-suffix lower tail, persistent oracle.**  A suffix whose summed flip-mass is at
least `σ` keeps its loss *above* the band, provided the band leaves room (`hband`). -/
lemma ploss_bad_lower (O : Oracle μ S) (Dfam : ι → Measure S)
    [∀ z, IsProbabilityMeasure (Dfam z)] (v : S) (g₁ g₂ κ σ : ℝ)
    (hg₁ : 0 ≤ g₁) (hg₂ : 0 ≤ g₂)
    (hNpos : 0 < ((Finset.univ : Finset ι).card : ℝ))
    (hcoll : (runMeasure (μ := μ) Dfam).real
      {x : (ι → S) × Ω | ¬ Function.Injective x.1} ≤ κ)
    (hband : ((Finset.univ : Finset ι).card : ℝ) * (O.η + (1 - 2 * O.η) * g₂ + g₁)
      ≤ ((Finset.univ : Finset ι).card : ℝ) * O.η + (1 - 2 * O.η)
          * (σ - ((Finset.univ : Finset ι).card : ℝ) * g₂)
        - ((Finset.univ : Finset ι).card : ℝ) * g₁)
    (hv : σ ≤ ∑ z, flipMass O (Dfam z) v) :
    (runMeasure (μ := μ) Dfam).real
        {x : (ι → S) × Ω | ploss (μ := μ) O v x ≤ ((Finset.univ : Finset ι).card : ℝ)
            * (O.η + (1 - 2 * O.η) * g₂ + g₁)}
      ≤ κ + Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g₂ ^ 2)
          + Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g₁ ^ 2) := by
  have h2η : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith [O.hη]
  have hincl : {x : (ι → S) × Ω | ploss (μ := μ) O v x
        ≤ ((Finset.univ : Finset ι).card : ℝ) * (O.η + (1 - 2 * O.η) * g₂ + g₁)}
      ⊆ {x : (ι → S) × Ω | ¬ Function.Injective x.1}
        ∪ ({x : (ι → S) × Ω | pflip (μ := μ) O v x
              ≤ (∑ z, flipMass O (Dfam z) v) - ((Finset.univ : Finset ι).card : ℝ) * g₂}
          ∪ {x : (ι → S) × Ω | Function.Injective x.1 ∧
              ploss (μ := μ) O v x ≤ ((Finset.univ : Finset ι).card : ℝ) * O.η
                + (1 - 2 * O.η) * pflip (μ := μ) O v x
                - ((Finset.univ : Finset ι).card : ℝ) * g₁}) := by
    intro x hx
    by_cases hinj : Function.Injective x.1
    · right
      by_cases hfl : pflip (μ := μ) O v x
          ≤ (∑ z, flipMass O (Dfam z) v) - ((Finset.univ : Finset ι).card : ℝ) * g₂
      · exact Or.inl hfl
      · refine Or.inr ⟨hinj, ?_⟩
        push_neg at hfl
        have hstep : σ - ((Finset.univ : Finset ι).card : ℝ) * g₂ ≤ pflip (μ := μ) O v x := by
          linarith [hfl, hv]
        have hkey : (1 - 2 * O.η) * (σ - ((Finset.univ : Finset ι).card : ℝ) * g₂)
            ≤ (1 - 2 * O.η) * pflip (μ := μ) O v x :=
          mul_le_mul_of_nonneg_left hstep h2η
        simp only [Set.mem_setOf_eq] at hx ⊢
        linarith [hx, hband, hkey]
    · exact Or.inl hinj
  have hu1 := measureReal_mono (μ := runMeasure (μ := μ) Dfam) hincl
  have hu2 := measureReal_union_le (μ := runMeasure (μ := μ) Dfam)
    {x : (ι → S) × Ω | ¬ Function.Injective x.1}
    ({x : (ι → S) × Ω | pflip (μ := μ) O v x
        ≤ (∑ z, flipMass O (Dfam z) v) - ((Finset.univ : Finset ι).card : ℝ) * g₂}
      ∪ {x : (ι → S) × Ω | Function.Injective x.1 ∧
          ploss (μ := μ) O v x ≤ ((Finset.univ : Finset ι).card : ℝ) * O.η
            + (1 - 2 * O.η) * pflip (μ := μ) O v x
            - ((Finset.univ : Finset ι).card : ℝ) * g₁})
  have hu3 := measureReal_union_le (μ := runMeasure (μ := μ) Dfam)
    {x : (ι → S) × Ω | pflip (μ := μ) O v x
        ≤ (∑ z, flipMass O (Dfam z) v) - ((Finset.univ : Finset ι).card : ℝ) * g₂}
    {x : (ι → S) × Ω | Function.Injective x.1 ∧
      ploss (μ := μ) O v x ≤ ((Finset.univ : Finset ι).card : ℝ) * O.η
        + (1 - 2 * O.η) * pflip (μ := μ) O v x
        - ((Finset.univ : Finset ι).card : ℝ) * g₁}
  linarith [hu1, hu2, hu3, hcoll, pflip_lower O Dfam v g₂ hg₂ hNpos,
    ploss_cond_lower O Dfam v g₁ hg₁ hNpos]

end Persistent

section PerSuffix
variable {ι : Type*} [Fintype ι]

/-- The run measure for draws indexed by `ι`, coordinate `z` drawing a prefix from
population `Dfam z` together with a fresh noise sample. -/
noncomputable def drawMeasure (Dfam : ι → Measure S) : Measure ((_ : ι) → S × Ω) :=
  Measure.pi (fun z => (Dfam z).prod μ)

instance (Dfam : ι → Measure S) [∀ z, IsProbabilityMeasure (Dfam z)] :
    IsProbabilityMeasure (drawMeasure (μ := μ) Dfam) := by
  unfold drawMeasure; infer_instance

variable (O : Oracle μ S) (Dfam : ι → Measure S) [∀ z, IsProbabilityMeasure (Dfam z)]

lemma lossRead_meas (v : S) (z : ι) :
    AEMeasurable (fun x : (_ : ι) → S × Ω => rdAt O v (x z)) (drawMeasure (μ := μ) Dfam) :=
  ((rdAt_meas O v).comp (measurable_pi_apply z)).aemeasurable

lemma lossRead_indep (v : S) :
    iIndepFun (fun z (x : (_ : ι) → S × Ω) => rdAt O v (x z)) (drawMeasure (μ := μ) Dfam) :=
  iIndepFun_pi (fun _ => (rdAt_meas O v).aemeasurable)

lemma lossRead_icc (v : S) (z : ι) :
    ∀ᵐ x ∂(drawMeasure (μ := μ) Dfam), rdAt O v (x z) ∈ Set.Icc (0 : ℝ) 1 :=
  (measurePreserving_eval (fun z => (Dfam z).prod μ) z).quasiMeasurePreserving.ae
    (rdAt_icc O (Dfam z) v)

/-- The summed mean of a fixed suffix's reads is its summed distributional flip-mass. -/
lemma lossRead_meanSum (v : S) :
    ∑ z, (drawMeasure (μ := μ) Dfam)[fun x : (_ : ι) → S × Ω => rdAt O v (x z)]
      = ((Finset.univ : Finset ι).card : ℝ) * O.η
        + (1 - 2 * O.η) * ∑ z, flipMass O (Dfam z) v := by
  have hmarg : ∀ z : ι, (drawMeasure (μ := μ) Dfam)[fun x : (_ : ι) → S × Ω => rdAt O v (x z)]
      = O.η + (1 - 2 * O.η) * flipMass O (Dfam z) v := by
    intro z
    have hmap : Measure.map (fun x : (_ : ι) → S × Ω => x z) (drawMeasure (μ := μ) Dfam)
        = (Dfam z).prod μ := (measurePreserving_eval (fun z => (Dfam z).prod μ) z).map_eq
    have h : (drawMeasure (μ := μ) Dfam)[fun x : (_ : ι) → S × Ω => rdAt O v (x z)]
        = ∫ y, rdAt O v y ∂((Dfam z).prod μ) := by
      rw [← hmap, integral_map (measurable_pi_apply z).aemeasurable
        (rdAt_meas O v).aestronglyMeasurable]
    rw [h, rdAt_mean O (Dfam z) v]
  rw [Finset.sum_congr rfl (fun z _ => hmarg z), Finset.sum_add_distrib, Finset.sum_const,
    nsmul_eq_mul, ← Finset.mul_sum]

/-- The separating loss threshold `N·η + (½−η)σ` — both selection bands collapse to it. -/
noncomputable def lossThresh (O : Oracle μ S) (σ N : ℝ) : ℝ := N * O.η + (1 / 2 - O.η) * σ

/-- **Per-suffix upper tail.**  A suffix with zero summed flip-mass (accept-preserving on
every population) has loss below the threshold except w.p. `exp(-2Nγ²)`. -/
lemma loss_good_upper (v : S) (σ : ℝ) (hσ0 : 0 ≤ σ)
    (hNpos : 0 < ((Finset.univ : Finset ι).card : ℝ))
    (hv : ∑ z, flipMass O (Dfam z) v = 0) :
    (drawMeasure (μ := μ) Dfam).real
        {x | lossThresh O σ ((Finset.univ : Finset ι).card : ℝ) ≤ ∑ z, rdAt O v (x z)}
      ≤ Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ)
          * ((1 / 2 - O.η) * σ / ((Finset.univ : Finset ι).card : ℝ)) ^ 2) := by
  have hγ : (0 : ℝ) ≤ (1 / 2 - O.η) * σ / ((Finset.univ : Finset ι).card : ℝ) :=
    div_nonneg (mul_nonneg (by linarith [O.hη]) hσ0) hNpos.le
  have hmean : ∑ z, (drawMeasure (μ := μ) Dfam)[fun x : (_ : ι) → S × Ω => rdAt O v (x z)]
      ≤ ((Finset.univ : Finset ι).card : ℝ) * O.η := by
    rw [lossRead_meanSum O Dfam v, hv]; simp
  have h := sumUpper_le (fun z (x : (_ : ι) → S × Ω) => rdAt O v (x z)) Finset.univ O.η
    ((1 / 2 - O.η) * σ / ((Finset.univ : Finset ι).card : ℝ))
    (lossRead_meas O Dfam v) (lossRead_indep O Dfam v) (lossRead_icc O Dfam v) hmean hγ
  have hthr : ((Finset.univ : Finset ι).card : ℝ)
      * (O.η + (1 / 2 - O.η) * σ / ((Finset.univ : Finset ι).card : ℝ))
      = lossThresh O σ ((Finset.univ : Finset ι).card : ℝ) := by
    unfold lossThresh; field_simp
  rwa [hthr] at h

/-- **Per-suffix lower tail.**  A suffix whose summed flip-mass is at least `σ` has loss
*above* the threshold except w.p. `exp(-2Nγ²)`. -/
lemma loss_bad_lower (v : S) (σ : ℝ) (hσ0 : 0 ≤ σ)
    (hNpos : 0 < ((Finset.univ : Finset ι).card : ℝ))
    (hv : σ ≤ ∑ z, flipMass O (Dfam z) v) :
    (drawMeasure (μ := μ) Dfam).real
        {x | ∑ z, rdAt O v (x z) ≤ lossThresh O σ ((Finset.univ : Finset ι).card : ℝ)}
      ≤ Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ)
          * ((1 / 2 - O.η) * σ / ((Finset.univ : Finset ι).card : ℝ)) ^ 2) := by
  set N := ((Finset.univ : Finset ι).card : ℝ) with hN
  have hγ : (0 : ℝ) ≤ (1 / 2 - O.η) * σ / N :=
    div_nonneg (mul_nonneg (by linarith [O.hη]) hσ0) hNpos.le
  have h1 : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith [O.hη]
  have hmean : N * (O.η + (1 - 2 * O.η) * σ / N)
      ≤ ∑ z, (drawMeasure (μ := μ) Dfam)[fun x : (_ : ι) → S × Ω => rdAt O v (x z)] := by
    rw [lossRead_meanSum O Dfam v]
    have h2 : (1 - 2 * O.η) * σ ≤ (1 - 2 * O.η) * ∑ z, flipMass O (Dfam z) v :=
      mul_le_mul_of_nonneg_left hv h1
    have hexp : N * (O.η + (1 - 2 * O.η) * σ / N) = N * O.η + (1 - 2 * O.η) * σ := by
      field_simp
    rw [hexp]; linarith [h2]
  have h := sumLower_le (fun z (x : (_ : ι) → S × Ω) => rdAt O v (x z)) Finset.univ
    (O.η + (1 - 2 * O.η) * σ / N) ((1 / 2 - O.η) * σ / N)
    (lossRead_meas O Dfam v) (lossRead_indep O Dfam v) (lossRead_icc O Dfam v) hmean hγ
  have hthr : N * ((O.η + (1 - 2 * O.η) * σ / N) - (1 / 2 - O.η) * σ / N)
      = lossThresh O σ N := by
    unfold lossThresh; field_simp; ring
  rwa [hthr] at h

end PerSuffix

/-- **Selection (distributional, per-coordinate populations).**  Each draw `z : ι` is an
independent `(prefix, noise)` pair whose prefix comes from that coordinate's population
`Dfam z` (for the multi-population setting, `ι = J × Fin m` with coordinate `(j,i)` drawn
from `D j`).  Candidates live in a type `C` carrying a suffix assignment `sfx : C → S` —
taking `C = Fin M` (the draw indices) keeps the candidate set fixed when the suffixes are
themselves drawn.  The greedy's least-loss `k`-subset then avoids every `bad` candidate,
where `good`/`bad` are stated by the **summed distributional flip-mass**.

Separation is definitional; independence of the reads is derived from the product
(`iIndepFun_pi`), and each read's mean is its coordinate's flip-mass (`rdAt_mean`). -/
theorem selection {C ι : Type*} [DecidableEq C] [Fintype ι] (O : Oracle μ S) (sfx : C → S)
    (Dfam : ι → Measure S) [∀ z, IsProbabilityMeasure (Dfam z)]
    (good bad : C → Prop) [DecidablePred good] [DecidablePred bad]
    (hdisj : ∀ c, bad c → ¬ good c)
    (cands : Finset C) (k : ℕ) (σ : ℝ) (hσ0 : 0 ≤ σ)
    (hgoodmass : ∀ c ∈ cands, good c → ∑ z, flipMass O (Dfam z) (sfx c) = 0)
    (hbadmass : ∀ c ∈ cands, bad c → σ ≤ ∑ z, flipMass O (Dfam z) (sfx c))
    (goodCount : k ≤ (cands.filter good).card)
    (chosen : ((z : ι) → S × Ω) → Finset C)
    (hsub : ∀ x, chosen x ⊆ cands) (hcard : ∀ x, (chosen x).card = k)
    (hleast : ∀ x, ∀ c ∈ chosen x, ∀ c' ∈ cands, c' ∉ chosen x →
        (∑ z, rdAt O (sfx c) (x z)) ≤ ∑ z, rdAt O (sfx c') (x z)) :
    (Measure.pi (fun z : ι => (Dfam z).prod μ)).real {x | ¬ ∀ c ∈ chosen x, ¬ bad c}
      ≤ (cands.card : ℝ)
          * Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ)
              * ((1 / 2 - O.η) * σ / ((Finset.univ : Finset ι).card : ℝ)) ^ 2) := by
  set νs : ι → Measure (S × Ω) := fun z => (Dfam z).prod μ with hνs
  set X : C → ι → ((z : ι) → S × Ω) → ℝ := fun c z x => rdAt O (sfx c) (x z) with hX
  set N : ℝ := ((Finset.univ : Finset ι).card : ℝ) with hN
  have hmarg : ∀ (c : C) (z : ι),
      (Measure.pi νs)[X c z] = O.η + (1 - 2 * O.η) * flipMass O (Dfam z) (sfx c) := by
    intro c z
    have hmap : Measure.map (fun x : (z : ι) → S × Ω => x z) (Measure.pi νs) = (Dfam z).prod μ :=
      (measurePreserving_eval νs z).map_eq
    have h : (Measure.pi νs)[X c z] = ∫ y, rdAt O (sfx c) y ∂((Dfam z).prod μ) := by
      rw [← hmap, integral_map (measurable_pi_apply z).aemeasurable
        (rdAt_meas O (sfx c)).aestronglyMeasurable]
    rw [h, rdAt_mean O (Dfam z) (sfx c)]
  have hmeas : ∀ c z, AEMeasurable (X c z) (Measure.pi νs) := fun c z =>
    ((rdAt_meas O (sfx c)).comp (measurable_pi_apply z)).aemeasurable
  have hindep : ∀ c, iIndepFun (X c) (Measure.pi νs) := fun c =>
    iIndepFun_pi (fun _ => (rdAt_meas O (sfx c)).aemeasurable)
  have hIcc : ∀ c z, ∀ᵐ x ∂(Measure.pi νs), X c z x ∈ Set.Icc (0 : ℝ) 1 := fun c z =>
    (measurePreserving_eval νs z).quasiMeasurePreserving.ae (rdAt_icc O (Dfam z) (sfx c))
  have hsum : ∀ c, ∑ z ∈ (Finset.univ : Finset ι), (Measure.pi νs)[X c z]
      = N * O.η + (1 - 2 * O.η) * ∑ z, flipMass O (Dfam z) (sfx c) := by
    intro c
    rw [Finset.sum_congr rfl (fun z _ => hmarg c z), Finset.sum_add_distrib, Finset.sum_const,
      nsmul_eq_mul, ← Finset.mul_sum, hN]
  have hgm : ∀ c ∈ cands, good c →
      ∑ z ∈ (Finset.univ : Finset ι), (Measure.pi νs)[X c z] ≤ N * O.η := by
    intro c hc hg; rw [hsum c, hgoodmass c hc hg]; simp
  have hbm : ∀ c ∈ cands, bad c →
      N * (O.η + (1 - 2 * O.η) * σ / N)
        ≤ ∑ z ∈ (Finset.univ : Finset ι), (Measure.pi νs)[X c z] := by
    intro c hc hb
    rw [hsum c]
    have hmass := hbadmass c hc hb
    have h1 : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith [O.hη]
    have h2 : (1 - 2 * O.η) * σ ≤ (1 - 2 * O.η) * ∑ z, flipMass O (Dfam z) (sfx c) :=
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


section Assembly
variable [Countable S] [MeasurableSingletonClass S] [DecidableEq S]

/-- With `S` countable and discrete (strings over a finite alphabet), a set of pairs is
measurable as soon as each of its `S`-slices is: it is a countable union of rectangles. -/
lemma measurableSet_of_countable_slices {Y : Type*} [MeasurableSpace Y] (P : S → Set Y)
    (hP : ∀ v, MeasurableSet (P v)) : MeasurableSet {q : S × Y | q.2 ∈ P q.1} := by
  have hset : {q : S × Y | q.2 ∈ P q.1} = ⋃ v : S, {v} ×ˢ P v := by
    ext q
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_prod, Set.mem_singleton_iff]
    exact ⟨fun h => ⟨q.1, rfl, h⟩, fun ⟨v, hv, h⟩ => by subst hv; exact h⟩
  rw [hset]
  exact MeasurableSet.iUnion (fun v => (measurableSet_singleton v).prod (hP v))

/-- **The per-index separation-failure event is small.**  For draw index `c`, the event
that the drawn suffix `x.1 c` is good yet reads above the threshold, or bad yet reads
below it, has probability at most the Hoeffding tail — uniformly in the draw, by
`prod_le_of_slice` applied to the per-suffix tails. -/
theorem index_event_le {ι : Type*} [Fintype ι] (O : Oracle μ S) (Dfam : ι → Measure S)
    [∀ z, IsProbabilityMeasure (Dfam z)] (Dsf : Measure S) [IsProbabilityMeasure Dsf]
    (M : ℕ) (c : Fin M) (σ : ℝ) (hσpos : 0 < σ)
    (hNpos : 0 < ((Finset.univ : Finset ι).card : ℝ)) :
    ((Measure.pi (fun _ : Fin M => Dsf)).prod (drawMeasure (μ := μ) Dfam)).real
        {x | ((∑ z, flipMass O (Dfam z) (x.1 c) = 0)
                ∧ lossThresh O σ ((Finset.univ : Finset ι).card : ℝ)
                    ≤ ∑ z, rdAt O (x.1 c) (x.2 z))
             ∨ ((σ ≤ ∑ z, flipMass O (Dfam z) (x.1 c))
                ∧ ∑ z, rdAt O (x.1 c) (x.2 z)
                    ≤ lossThresh O σ ((Finset.univ : Finset ι).card : ℝ))}
      ≤ Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ)
          * ((1 / 2 - O.η) * σ / ((Finset.univ : Finset ι).card : ℝ)) ^ 2) := by
  classical
  set N := ((Finset.univ : Finset ι).card : ℝ) with hN
  -- the slice at a fixed suffix assignment
  set P : S → Set ((_ : ι) → S × Ω) := fun v =>
    {y | ((∑ z, flipMass O (Dfam z) v = 0) ∧ lossThresh O σ N ≤ ∑ z, rdAt O v (y z))
         ∨ ((σ ≤ ∑ z, flipMass O (Dfam z) v) ∧ ∑ z, rdAt O v (y z) ≤ lossThresh O σ N)}
    with hP
  have hlossmeas : ∀ v : S, Measurable (fun y : (_ : ι) → S × Ω => ∑ z, rdAt O v (y z)) :=
    fun v => Finset.measurable_sum _ (fun z _ => (rdAt_meas O v).comp (measurable_pi_apply z))
  have hPmeas : ∀ v, MeasurableSet (P v) := by
    intro v
    have h1 : MeasurableSet
        {y : (_ : ι) → S × Ω | lossThresh O σ N ≤ ∑ z, rdAt O v (y z)} :=
      measurableSet_le measurable_const (hlossmeas v)
    have h2 : MeasurableSet
        {y : (_ : ι) → S × Ω | ∑ z, rdAt O v (y z) ≤ lossThresh O σ N} :=
      measurableSet_le (hlossmeas v) measurable_const
    have hA : MeasurableSet {y : (_ : ι) → S × Ω |
        (∑ z, flipMass O (Dfam z) v = 0) ∧ lossThresh O σ N ≤ ∑ z, rdAt O v (y z)} := by
      by_cases hg : ∑ z, flipMass O (Dfam z) v = 0
      · convert h1 using 1; ext y; simp [hg]
      · convert MeasurableSet.empty; ext y; simp [hg]
    have hB : MeasurableSet {y : (_ : ι) → S × Ω |
        (σ ≤ ∑ z, flipMass O (Dfam z) v) ∧ ∑ z, rdAt O v (y z) ≤ lossThresh O σ N} := by
      by_cases hb : σ ≤ ∑ z, flipMass O (Dfam z) v
      · convert h2 using 1; ext y; simp [hb]
      · convert MeasurableSet.empty; ext y; simp [hb]
    exact hA.union hB
  -- each slice is bounded by the per-suffix tails
  have hslice : ∀ s : Fin M → S,
      (drawMeasure (μ := μ) Dfam).real (P (s c))
        ≤ Real.exp (-2 * N * ((1 / 2 - O.η) * σ / N) ^ 2) := by
    intro s
    by_cases hg : ∑ z, flipMass O (Dfam z) (s c) = 0
    · refine le_trans (measureReal_mono ?_) (loss_good_upper O Dfam (s c) σ hσpos.le hNpos hg)
      intro y hy
      rcases hy with ⟨_, h⟩ | ⟨hb, h⟩
      · exact h
      · exfalso; rw [hg] at hb; linarith
    · by_cases hb : σ ≤ ∑ z, flipMass O (Dfam z) (s c)
      · refine le_trans (measureReal_mono ?_) (loss_bad_lower O Dfam (s c) σ hσpos.le hNpos hb)
        intro y hy
        rcases hy with ⟨hg', _⟩ | ⟨_, h⟩
        · exact absurd hg' hg
        · exact h
      · have : P (s c) = ∅ := by
          ext y; simp only [hP, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
          rintro (⟨h, _⟩ | ⟨h, _⟩)
          · exact hg h
          · exact hb h
        rw [this]; simpa using (Real.exp_pos _).le
  -- transfer the uniform slice bound to the product
  refine prod_le_of_slice _ _ _ ?_ _ (Real.exp_pos _).le (fun s => hslice s)
  have hfm : Measurable
      (fun x : (Fin M → S) × ((_ : ι) → S × Ω) => (x.1 c, x.2)) :=
    ((measurable_pi_apply c).comp measurable_fst).prodMk measurable_snd
  show MeasurableSet ((fun x : (Fin M → S) × ((_ : ι) → S × Ω) => (x.1 c, x.2)) ⁻¹'
    {q : S × ((_ : ι) → S × Ω) | q.2 ∈ P q.1})
  exact hfm (measurableSet_of_countable_slices P hPmeas)

/-- **Findability, quantitative.**  Over `M` i.i.d. suffix draws, the count of draws
landing in `G` (probability `≥ pAP` each) falls to `M(pAP−γ)` only w.p. `exp(-2Mγ²)`. -/
theorem goodCount_le (Dsf : Measure S) [IsProbabilityMeasure Dsf]
    (M : ℕ) (pAP γ : ℝ) (hγ : 0 ≤ γ) (G : Set S) (hGm : MeasurableSet G)
    (hG : pAP ≤ Dsf.real G) :
    (Measure.pi (fun _ : Fin M => Dsf)).real
        {s : Fin M → S | ∑ c, Set.indicator G (fun _ => (1 : ℝ)) (s c)
            ≤ (M : ℝ) * (pAP - γ)}
      ≤ Real.exp (-2 * (M : ℝ) * γ ^ 2) := by
  classical
  set I : S → ℝ := Set.indicator G (fun _ => (1 : ℝ)) with hI
  have hImeas : Measurable I := (measurable_one.indicator hGm)
  have hIicc : ∀ v, I v ∈ Set.Icc (0 : ℝ) 1 := by
    intro v
    rw [Set.mem_Icc]
    by_cases h : v ∈ G
    · simp only [hI, Set.indicator_of_mem h]; norm_num
    · simp only [hI, Set.indicator_of_notMem h]; norm_num
  have hmean : ∫ v, I v ∂Dsf = Dsf.real G := by
    rw [hI, integral_indicator hGm, setIntegral_const, smul_eq_mul, mul_one, measureReal_def]
  have hXmeas : ∀ c : Fin M, AEMeasurable (fun s : Fin M → S => I (s c))
      (Measure.pi (fun _ : Fin M => Dsf)) :=
    fun c => (hImeas.comp (measurable_pi_apply c)).aemeasurable
  have hXindep : iIndepFun (fun (c : Fin M) (s : Fin M → S) => I (s c))
      (Measure.pi (fun _ : Fin M => Dsf)) := iIndepFun_pi (fun _ => hImeas.aemeasurable)
  have hXicc : ∀ c : Fin M, ∀ᵐ s ∂(Measure.pi (fun _ : Fin M => Dsf)),
      I (s c) ∈ Set.Icc (0 : ℝ) 1 := fun c => Filter.Eventually.of_forall (fun s => hIicc (s c))
  have hXmarg : ∀ c : Fin M,
      (Measure.pi (fun _ : Fin M => Dsf))[fun s : Fin M → S => I (s c)] = Dsf.real G := by
    intro c
    have hmap : Measure.map (fun s : Fin M → S => s c) (Measure.pi (fun _ : Fin M => Dsf)) = Dsf :=
      (measurePreserving_eval (fun _ : Fin M => Dsf) c).map_eq
    calc (Measure.pi (fun _ : Fin M => Dsf))[fun s : Fin M → S => I (s c)]
        = ∫ v, I v ∂(Measure.map (fun s : Fin M → S => s c)
            (Measure.pi (fun _ : Fin M => Dsf))) := by
          rw [integral_map (measurable_pi_apply c).aemeasurable hImeas.aestronglyMeasurable]
      _ = ∫ v, I v ∂Dsf := by rw [hmap]
      _ = Dsf.real G := hmean
  have hsum : (M : ℝ) * pAP
      ≤ ∑ c, (Measure.pi (fun _ : Fin M => Dsf))[fun s : Fin M → S => I (s c)] := by
    rw [Finset.sum_congr rfl (fun c _ => hXmarg c), Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul]
    exact mul_le_mul_of_nonneg_left hG (Nat.cast_nonneg M)
  have h := sumLower_le (fun (c : Fin M) (s : Fin M → S) => I (s c)) Finset.univ pAP γ
    hXmeas hXindep hXicc (by simpa using hsum) hγ
  simpa using h

#print axioms goodCount_le
#print axioms index_event_le

/-- A failure bound gives the complementary success bound, with **no** measurability
needed: outer measure is subadditive and `s ∪ sᶜ = univ`. -/
theorem one_sub_le_compl_real {α : Type*} [MeasurableSpace α] (ν : Measure α)
    [IsProbabilityMeasure ν] (s : Set α) (d : ℝ) (h : ν.real s ≤ d) : 1 - d ≤ ν.real sᶜ := by
  have hsub : (1 : ℝ≥0∞) ≤ ν s + ν sᶜ := by
    calc (1 : ℝ≥0∞) = ν Set.univ := measure_univ.symm
      _ = ν (s ∪ sᶜ) := by rw [Set.union_compl_self]
      _ ≤ ν s + ν sᶜ := measure_union_le _ _
  have hreal : (1 : ℝ) ≤ ν.real s + ν.real sᶜ := by
    have := ENNReal.toReal_mono (by finiteness) hsub
    rwa [ENNReal.toReal_add (measure_ne_top _ _) (measure_ne_top _ _), ENNReal.toReal_one] at this
  linarith

#print axioms one_sub_le_compl_real

/-- **The distributional clustering theorem (PR #257).**

Prefixes come from a *collection* of populations `D : J → Measure S`; candidate suffixes
are drawn from `Dsf`, and findability is the single probability `pAP` that a drawn suffix
is accept-preserving on the true noiseless oracle.  The seed is `ε = 1`.

With probability `≥ 1 − δ` (stated as: the failure event has probability `≤ δ`) the
returned family preserves acceptance on at least a `1 − k·εpop` fraction of **each**
population.  `good`/`bad` are the distributional flip-mass conditions, so the separation
is definitional; the greedy is the defined `leastLossSubset`; the two error terms are the
selection tail (`M` draws × the Hoeffding tail) and the findability tail. -/
theorem clustering_budget {J : Type*} [Fintype J]
    (O : Oracle μ S)
    /- Family of prefix distributions that we must classify -/
    (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    /- Distribution of suffixes from which we draw -/
    (Dsf : Measure S) [IsProbabilityMeasure Dsf]
    (M m k : ℕ) (hkM : k ≤ M)
    (εpop : ℝ) (hεpop : 0 < εpop) (hmpos : 0 < m) (hJ : 0 < Fintype.card J)
    (pAP γsuf δ : ℝ) (hγsuf : 0 ≤ γsuf)
    (hfind : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hcount : (k : ℝ) ≤ (M : ℝ) * (pAP - γsuf))
    (hbudget : Real.exp (-2 * (M : ℝ) * γsuf ^ 2)
        + (M : ℝ) * Real.exp (-2 * ((Finset.univ : Finset (J × Fin m)).card : ℝ)
            * ((1 / 2 - O.η) * ((m : ℝ) * εpop)
                / ((Finset.univ : Finset (J × Fin m)).card : ℝ)) ^ 2) ≤ δ) :
    ((Measure.pi (fun _ : Fin M => Dsf)).prod
        (drawMeasure (μ := μ) (fun z : J × Fin m => D z.1))).real
      {x | ¬ ∀ j : J, 1 - (k : ℝ) * εpop
            ≤ (D j).real {p | ∀ v ∈ (leastLossSubset
                  (fun c => ∑ z, rdAt O (x.1 c) (x.2 z)) (Finset.univ : Finset (Fin M)) k).image x.1,
                O.label (p * v) = O.label p}}
      ≤ δ := by
  classical
  set ι := J × Fin m
  set Dfam : ι → Measure S := fun z => D z.1 with hDfam
  set N : ℝ := ((Finset.univ : Finset ι).card : ℝ) with hN
  set σ : ℝ := (m : ℝ) * εpop with hσ
  set G : Set S := {v | ∀ p, O.label (p * v) = O.label p} with hG
  set I : S → ℝ := Set.indicator G (fun _ => (1 : ℝ)) with hI
  set thresh : ℝ := lossThresh O σ N with hthresh
  have hσpos : 0 < σ := by
    rw [hσ]; exact mul_pos (by exact_mod_cast hmpos) hεpop
  have hNpos : 0 < N := by
    rw [hN, Finset.card_univ]
    have : 0 < Fintype.card ι := by
      rw [show Fintype.card ι = Fintype.card J * m from Fintype.card_prod J (Fin m) ▸ by simp]
      exact Nat.mul_pos hJ hmpos
    exact_mod_cast this
  -- accept-preserving ⇒ zero summed flip-mass
  have hGmeas : MeasurableSet G := Set.Countable.measurableSet (Set.to_countable G)
  have hGgood : ∀ v ∈ G, ∑ z, flipMass O (Dfam z) v = 0 := by
    intro v hv
    refine Finset.sum_eq_zero (fun z _ => ?_)
    rw [flipMass_eq]
    have : {p | O.label (p * v) ≠ O.label p} = (∅ : Set S) := by
      ext p; simpa using hv p
    rw [this]; simp
  -- summed flip-mass splits as m × the per-population sum
  have hsplit : ∀ v, ∑ z : ι, flipMass O (Dfam z) v
      = (m : ℝ) * ∑ j : J, flipMass O (D j) v := by
    intro v
    rw [Fintype.sum_prod_type]
    simp [hDfam, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
      Finset.mul_sum]
  -- the failure event is contained in (count failure) ∪ (per-index separation failures)
  set A : Set ((Fin M → S) × ((_ : ι) → S × Ω)) :=
    {x | ∑ c, I (x.1 c) ≤ (M : ℝ) * (pAP - γsuf)} with hA
  set E : Fin M → Set ((Fin M → S) × ((_ : ι) → S × Ω)) := fun c =>
    {x | ((∑ z, flipMass O (Dfam z) (x.1 c) = 0) ∧ thresh ≤ ∑ z, rdAt O (x.1 c) (x.2 z))
         ∨ ((σ ≤ ∑ z, flipMass O (Dfam z) (x.1 c)) ∧ ∑ z, rdAt O (x.1 c) (x.2 z) ≤ thresh)}
    with hE
  have hincl : {x : (Fin M → S) × ((_ : ι) → S × Ω) | ¬ ∀ j : J, 1 - (k : ℝ) * εpop
      ≤ (D j).real {p | ∀ v ∈ (leastLossSubset
            (fun c => ∑ z, rdAt O (x.1 c) (x.2 z)) (Finset.univ : Finset (Fin M)) k).image x.1,
          O.label (p * v) = O.label p}}
      ⊆ A ∪ ⋃ c, E c := by
    intro x hx
    by_contra hnot
    rw [Set.mem_union, not_or] at hnot
    obtain ⟨hnA, hnE⟩ := hnot
    simp only [Set.mem_iUnion, not_exists] at hnE
    -- enough good candidates among the draws
    have hcountgt : (M : ℝ) * (pAP - γsuf) < ∑ c, I (x.1 c) := by
      by_contra hle; exact hnA (by rw [hA]; exact not_lt.mp hle)
    have hIsum : ∑ c, I (x.1 c)
        = (((Finset.univ : Finset (Fin M)).filter (fun c => x.1 c ∈ G)).card : ℝ) := by
      have hpt : ∀ c : Fin M, I (x.1 c) = if x.1 c ∈ G then (1 : ℝ) else 0 := by
        intro c; rw [hI, Set.indicator_apply]
      rw [Finset.sum_congr rfl (fun c _ => hpt c), Finset.sum_boole]
    have hgoodCount : k ≤ ((Finset.univ : Finset (Fin M)).filter
        (fun c => ∑ z, flipMass O (Dfam z) (x.1 c) = 0)).card := by
      have h1 : ((Finset.univ : Finset (Fin M)).filter (fun c => x.1 c ∈ G)).card
          ≤ ((Finset.univ : Finset (Fin M)).filter
              (fun c => ∑ z, flipMass O (Dfam z) (x.1 c) = 0)).card := by
        refine Finset.card_le_card ?_
        intro c hc
        simp only [Finset.mem_filter] at hc ⊢
        exact ⟨hc.1, hGgood _ hc.2⟩
      rw [hIsum] at hcountgt
      have hlt : (k : ℝ)
          < (((Finset.univ : Finset (Fin M)).filter (fun c => x.1 c ∈ G)).card : ℝ) :=
        lt_of_le_of_lt hcount hcountgt
      have hk1 : k ≤ ((Finset.univ : Finset (Fin M)).filter (fun c => x.1 c ∈ G)).card := by
        exact_mod_cast hlt.le
      exact le_trans hk1 h1
    -- the reads separate the classes
    have hsep : ∀ c ∈ (Finset.univ : Finset (Fin M)), ∀ c' ∈ (Finset.univ : Finset (Fin M)),
        (∑ z, flipMass O (Dfam z) (x.1 c) = 0) → (σ ≤ ∑ z, flipMass O (Dfam z) (x.1 c')) →
        (∑ z, rdAt O (x.1 c) (x.2 z)) < ∑ z, rdAt O (x.1 c') (x.2 z) := by
      intro c _ c' _ hgc hbc'
      have h1 : ∑ z, rdAt O (x.1 c) (x.2 z) < thresh := by
        by_contra hge
        exact hnE c (Or.inl ⟨hgc, not_lt.mp hge⟩)
      have h2 : thresh < ∑ z, rdAt O (x.1 c') (x.2 z) := by
        by_contra hle
        exact hnE c' (Or.inr ⟨hbc', not_lt.mp hle⟩)
      linarith
    -- the greedy therefore avoids the bad candidates
    have hkcard : k ≤ (Finset.univ : Finset (Fin M)).card := by
      rw [Finset.card_univ, Fintype.card_fin]; exact hkM
    have havoid := chosen_avoids_bad
      (fun c => ∑ z, rdAt O (x.1 c) (x.2 z))
      (fun c => ∑ z, flipMass O (Dfam z) (x.1 c) = 0)
      (fun c => σ ≤ ∑ z, flipMass O (Dfam z) (x.1 c))
      (fun c hb hg => by rw [hg] at hb; linarith)
      (Finset.univ : Finset (Fin M))
      (leastLossSubset (fun c => ∑ z, rdAt O (x.1 c) (x.2 z)) (Finset.univ : Finset (Fin M)) k)
      k
      (leastLossSubset_subset _ _ _ hkcard) (leastLossSubset_card _ _ _ hkcard)
      (leastLossSubset_least _ _ _ hkcard) hgoodCount hsep
    -- hence every selected suffix has small per-population flip-mass
    have hfam : ∀ v ∈ (leastLossSubset (fun c => ∑ z, rdAt O (x.1 c) (x.2 z))
        (Finset.univ : Finset (Fin M)) k).image x.1,
        ∑ j : J, flipMass O (D j) v ≤ εpop := by
      intro v hv
      obtain ⟨c, hc, rfl⟩ := Finset.mem_image.mp hv
      have := havoid c hc
      rw [not_le] at this
      have hlt : (m : ℝ) * ∑ j : J, flipMass O (D j) (x.1 c) < (m : ℝ) * εpop := by
        rw [← hsplit]; exact this
      have hm0 : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hmpos
      exact le_of_lt (lt_of_mul_lt_mul_left hlt hm0.le)
    -- so each population is covered
    apply hx
    intro j
    have hcov := coverage_of_summed_flip O D (Finset.univ : Finset J)
      ((leastLossSubset (fun c => ∑ z, rdAt O (x.1 c) (x.2 z))
        (Finset.univ : Finset (Fin M)) k).image x.1) εpop hfam j (Finset.mem_univ j)
    refine le_trans ?_ hcov
    have hcard : (((leastLossSubset (fun c => ∑ z, rdAt O (x.1 c) (x.2 z))
        (Finset.univ : Finset (Fin M)) k).image x.1).card : ℝ) ≤ (k : ℝ) := by
      have := Finset.card_image_le (s := leastLossSubset
        (fun c => ∑ z, rdAt O (x.1 c) (x.2 z)) (Finset.univ : Finset (Fin M)) k) (f := x.1)
      rw [leastLossSubset_card _ _ _ hkcard] at this
      exact_mod_cast this
    nlinarith [hcard, hεpop.le]
  -- bound the two error sources
  have hAbound : ((Measure.pi (fun _ : Fin M => Dsf)).prod
      (drawMeasure (μ := μ) Dfam)).real A ≤ Real.exp (-2 * (M : ℝ) * γsuf ^ 2) := by
    have heq : ((Measure.pi (fun _ : Fin M => Dsf)).prod (drawMeasure (μ := μ) Dfam)).real A
        = (Measure.pi (fun _ : Fin M => Dsf)).real
          {s : Fin M → S | ∑ c, I (s c) ≤ (M : ℝ) * (pAP - γsuf)} :=
      prod_fst_real (Measure.pi (fun _ : Fin M => Dsf)) (drawMeasure (μ := μ) Dfam)
        {s : Fin M → S | ∑ c, I (s c) ≤ (M : ℝ) * (pAP - γsuf)}
    rw [heq]
    exact goodCount_le Dsf M pAP γsuf hγsuf G hGmeas hfind
  have hEbound : ∀ c, ((Measure.pi (fun _ : Fin M => Dsf)).prod
      (drawMeasure (μ := μ) Dfam)).real (E c)
      ≤ Real.exp (-2 * N * ((1 / 2 - O.η) * σ / N) ^ 2) :=
    fun c => index_event_le O Dfam Dsf M c σ hσpos hNpos
  have hUnion : ((Measure.pi (fun _ : Fin M => Dsf)).prod
      (drawMeasure (μ := μ) Dfam)).real (⋃ c, E c)
      ≤ (M : ℝ) * Real.exp (-2 * N * ((1 / 2 - O.η) * σ / N) ^ 2) := by
    have hset : (⋃ c, E c) = ⋃ c ∈ (Finset.univ : Finset (Fin M)), E c := by
      ext x; simp
    rw [hset]
    calc ((Measure.pi (fun _ : Fin M => Dsf)).prod (drawMeasure (μ := μ) Dfam)).real
          (⋃ c ∈ (Finset.univ : Finset (Fin M)), E c)
        ≤ ∑ c, ((Measure.pi (fun _ : Fin M => Dsf)).prod
            (drawMeasure (μ := μ) Dfam)).real (E c) := measureReal_biUnion_le _ _
      _ ≤ ∑ _c : Fin M, Real.exp (-2 * N * ((1 / 2 - O.η) * σ / N) ^ 2) :=
          Finset.sum_le_sum (fun c _ => hEbound c)
      _ = (M : ℝ) * Real.exp (-2 * N * ((1 / 2 - O.η) * σ / N) ^ 2) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  calc ((Measure.pi (fun _ : Fin M => Dsf)).prod (drawMeasure (μ := μ) Dfam)).real _
      ≤ ((Measure.pi (fun _ : Fin M => Dsf)).prod
          (drawMeasure (μ := μ) Dfam)).real (A ∪ ⋃ c, E c) := measureReal_mono hincl
    _ ≤ ((Measure.pi (fun _ : Fin M => Dsf)).prod (drawMeasure (μ := μ) Dfam)).real A
        + ((Measure.pi (fun _ : Fin M => Dsf)).prod
            (drawMeasure (μ := μ) Dfam)).real (⋃ c, E c) := measureReal_union_le _ _
    _ ≤ Real.exp (-2 * (M : ℝ) * γsuf ^ 2)
        + (M : ℝ) * Real.exp (-2 * N * ((1 / 2 - O.η) * σ / N) ^ 2) := add_le_add hAbound hUnion
    _ ≤ δ := hbudget

#print axioms clustering_budget

/-- **The distributional clustering theorem, in the promised form.**  With probability
`≥ 1 − δ`, the returned family preserves acceptance on `≥ 1 − εcov` of **each**
population.  Derived from `clustering_budget` by instantiating the population index with
the given `Finset`, taking the per-suffix tolerance `εcov/k`, fixing the findability slack
at `pAP/2`, and turning the failure bound into its complement. -/
theorem clustering_pac (O : Oracle μ S) (hsig : O.η < 1 / 2)
    {J : Type*} (populations : Finset J) (hpop : populations.Nonempty)
    (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    (Dsf : Measure S) [IsProbabilityMeasure Dsf]
    (pAP : ℝ) (hpAP : 0 < pAP)
    (hfind : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (εcov : ℝ) (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ)
    (m M k : ℕ) (hk : 0 < k) (hkM : k ≤ M) (hmpos : 0 < m)
    (hM1 : (2 * k : ℝ) / pAP ≤ (M : ℝ))
    (hM2 : 2 * Real.log (2 / δ) / pAP ^ 2 ≤ (M : ℝ))
    (hm : (populations.card : ℝ) * Real.log (2 * M / δ)
            / (2 * ((1 / 2 - O.η) * (εcov / k)) ^ 2) ≤ (m : ℝ)) :
    1 - δ ≤ ((Measure.pi (fun _ : Fin M => Dsf)).prod
        (drawMeasure (μ := μ) (fun z : {j // j ∈ populations} × Fin m => D z.1.val))).real
      {x | ∀ j ∈ populations, 1 - εcov
            ≤ (D j).real {p | ∀ v ∈ (leastLossSubset
                  (fun c => ∑ z, rdAt O (x.1 c) (x.2 z))
                  (Finset.univ : Finset (Fin M)) k).image x.1,
                O.label (p * v) = O.label p}} := by
  classical
  have hkR : (0 : ℝ) < (k : ℝ) := by exact_mod_cast hk
  have hMR : (0 : ℝ) < (M : ℝ) := by
    have : 0 < M := lt_of_lt_of_le hk hkM
    exact_mod_cast this
  have hPcard : Fintype.card {j // j ∈ populations} = populations.card := Fintype.card_coe _
  have hPpos : 0 < Fintype.card {j // j ∈ populations} := by
    rw [hPcard]; exact Finset.card_pos.mpr hpop
  set εpop : ℝ := εcov / k with hεpop_def
  have hεpop : 0 < εpop := div_pos hεcov hkR
  have hkε : (k : ℝ) * εpop = εcov := by rw [hεpop_def]; field_simp
  set N : ℝ := ((Finset.univ : Finset ({j // j ∈ populations} × Fin m)).card : ℝ) with hN
  have hNval : N = (populations.card : ℝ) * (m : ℝ) := by
    rw [hN, Finset.card_univ, Fintype.card_prod, hPcard, Fintype.card_fin]; push_cast; ring
  have hNpos : 0 < N := by
    rw [hNval]; exact mul_pos (by exact_mod_cast Finset.card_pos.mpr hpop) (by exact_mod_cast hmpos)
  -- findability slack fixed at pAP/2
  have hcount : (k : ℝ) ≤ (M : ℝ) * (pAP - pAP / 2) := by
    rw [show pAP - pAP / 2 = pAP / 2 by ring]
    rw [div_le_iff₀ hpAP] at hM1
    nlinarith [hM1, hpAP]
  -- the two error terms are each ≤ δ/2
  have hsuffix : Real.exp (-2 * (M : ℝ) * (pAP / 2) ^ 2) ≤ δ / 2 := by
    have hthr : Real.log (1 / (δ / 2)) / (2 * (pAP / 2) ^ 2) ≤ (M : ℝ) := by
      rw [show (1 : ℝ) / (δ / 2) = 2 / δ by rw [one_div, inv_div]]
      rw [div_le_iff₀ (by positivity)] at hM2 ⊢
      nlinarith [hM2, hpAP]
    have h := tail_le (t := pAP / 2) (c := 1) (ε := δ / 2) (k := (M : ℝ))
      (by linarith) one_pos (by linarith) hthr
    simpa using h
  have hexpeq : -2 * N * ((1 / 2 - O.η) * ((m : ℝ) * εpop) / N) ^ 2
      = -2 * (m : ℝ) * (((1 / 2 - O.η) * εpop) ^ 2 / (populations.card : ℝ)) := by
    rw [hNval]
    have hPne : ((populations.card : ℝ)) ≠ 0 := by
      have := Finset.card_pos.mpr hpop; positivity
    have hmne : ((m : ℝ)) ≠ 0 := by positivity
    field_simp
  have hprefix : (M : ℝ) * Real.exp (-2 * N
      * ((1 / 2 - O.η) * ((m : ℝ) * εpop) / N) ^ 2) ≤ δ / 2 := by
    rw [hexpeq]
    have hPpos' : (0 : ℝ) < (populations.card : ℝ) := by
      exact_mod_cast Finset.card_pos.mpr hpop
    have ht : (0 : ℝ) < (1 / 2 - O.η) * εpop / Real.sqrt (populations.card : ℝ) :=
      div_pos (mul_pos (by linarith) hεpop) (Real.sqrt_pos.mpr hPpos')
    have hsq : ((1 / 2 - O.η) * εpop / Real.sqrt (populations.card : ℝ)) ^ 2
        = ((1 / 2 - O.η) * εpop) ^ 2 / (populations.card : ℝ) := by
      rw [div_pow, Real.sq_sqrt hPpos'.le]
    rw [← hsq]
    refine tail_le ht hMR (by linarith) ?_
    rw [hsq, show (M : ℝ) / (δ / 2) = 2 * M / δ by rw [div_div_eq_mul_div]; ring]
    have hconv : Real.log (2 * M / δ)
          / (2 * (((1 / 2 - O.η) * εpop) ^ 2 / (populations.card : ℝ)))
        = (populations.card : ℝ) * Real.log (2 * M / δ)
          / (2 * ((1 / 2 - O.η) * εpop) ^ 2) := by
      have hA : ((1 / 2 - O.η) * εpop) ≠ 0 :=
        ne_of_gt (mul_pos (by linarith) hεpop)
      field_simp
    rw [hconv]
    exact hm
  -- apply the proved core and complement it
  have hcore := clustering_budget O (fun j : {j // j ∈ populations} => D j.val) Dsf M m k hkM
    εpop hεpop hmpos hPpos pAP (pAP / 2) δ (by linarith) hfind hcount
    (by rw [← hN]; linarith [hsuffix, hprefix])
  have hcompl := one_sub_le_compl_real _ _ δ hcore
  refine le_trans hcompl (le_of_eq ?_)
  congr 1
  ext x
  simp only [Set.mem_compl_iff, Set.mem_setOf_eq, not_not, Subtype.forall]
  constructor
  · intro h j hj; rw [← hkε]; exact h j hj
  · intro h j hj; rw [hkε]; exact h j hj

#print axioms clustering_pac

end Assembly

end Draws

end OrthoDFA
