import OrthoDFA.Proofs.LoopRounds
import OrthoDFA.Proofs.Replay
import Mathlib.Probability.ProductMeasure
import Mathlib.MeasureTheory.Constructions.UnitInterval
import Mathlib.MeasureTheory.Function.Floor

/-!
# A round's clustering run, at given harvests, is drawn as the clustering's theorem draws it

A uniform `u` picks the string at `⌊u n⌋` of a list of `n`, each with chance `1/n`, so a pick is
drawn from `harvestMeasure` (`map_pickFrom`). The round's draws, run on given harvests, are then a
run drawn by `runMeasure` at the harvests' populations (`measurePreserving_clusterRun`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

lemma volume_floor_eq {n : ℕ} (hn : 0 < n) {i : ℕ} (hi : i < n) :
    (volume : Measure unitInterval) {u | ⌊(u : ℝ) * n⌋₊ = i} = ((n : ℝ≥0∞))⁻¹ := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hset : Subtype.val '' {u : unitInterval | ⌊(u : ℝ) * n⌋₊ = i}
      = Set.Ico ((i : ℝ) / n) ((i + 1) / n) := by
    ext t
    simp only [Set.mem_image, Set.mem_ofPred_eq, Set.mem_Ico]
    constructor
    · rintro ⟨u, hu, rfl⟩
      have h0 : 0 ≤ (u : ℝ) * n := mul_nonneg u.2.1 hn'.le
      rw [Nat.floor_eq_iff h0] at hu
      exact ⟨by rw [div_le_iff₀ hn']; exact hu.1, by rw [lt_div_iff₀ hn']; exact hu.2⟩
    · rintro ⟨h1, h2⟩
      have ht0 : 0 ≤ t := le_trans (by positivity) h1
      have ht1 : t ≤ 1 := by
        have : ((i : ℝ) + 1) / n ≤ 1 := by
          rw [div_le_one hn']; exact_mod_cast hi
        linarith
      refine ⟨⟨t, ht0, ht1⟩, ?_, rfl⟩
      show ⌊t * n⌋₊ = i
      rw [Nat.floor_eq_iff (by positivity)]
      exact ⟨by rwa [div_le_iff₀ hn'] at h1, by rwa [lt_div_iff₀ hn'] at h2⟩
  rw [unitInterval.volume_apply, hset, Real.volume_Ico]
  rw [show ((i : ℝ) + 1) / n - i / n = (n : ℝ)⁻¹ by field_simp; ring,
    ENNReal.ofReal_inv_of_pos hn', ENNReal.ofReal_natCast]

lemma volume_floor_ge {n : ℕ} (hn : 0 < n) {i : ℕ} (hi : n ≤ i) :
    (volume : Measure unitInterval) {u | ⌊(u : ℝ) * n⌋₊ = i} = 0 := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  refine measure_mono_null (t := {1}) (fun u hu => ?_) (measure_singleton 1)
  have h0 : 0 ≤ (u : ℝ) * n := mul_nonneg u.2.1 hn'.le
  have hu' := (Nat.floor_eq_iff h0).1 hu
  have : (n : ℝ) ≤ i := by exact_mod_cast hi
  have h1 : (1 : ℝ) ≤ u := by
    by_contra h
    push Not at h
    nlinarith [hu'.1]
  exact Set.mem_singleton_iff.2 (Subtype.ext (le_antisymm u.2.2 h1))

lemma measurable_pickFrom (zs : List (FreeMonoid α)) : Measurable (pickFrom zs) :=
  (measurable_from_nat (f := fun i => zs.getD i 1)).comp
    ((continuous_subtype_val.measurable.mul_const _).nat_floor)

lemma list_sum_dirac (zs : List (FreeMonoid α)) :
    (zs.map Measure.dirac).sum = ∑ i ∈ Finset.range zs.length, Measure.dirac (zs.getD i 1) := by
  induction zs with
  | nil => simp
  | cons z zs ih =>
    rw [List.map_cons, List.sum_cons, ih, List.length_cons, Finset.sum_range_succ']
    simp only [List.getD_cons_succ, List.getD_cons_zero]
    rw [add_comm]

/-- A uniform pick from a list is drawn from its `harvestMeasure`. -/
lemma map_pickFrom (zs : List (FreeMonoid α)) :
    (volume : Measure unitInterval).map (pickFrom zs) = harvestMeasure zs := by
  classical
  by_cases hzs : zs = []
  · subst hzs
    have : pickFrom ([] : List (FreeMonoid α)) = fun _ => 1 := by
      funext u; simp [pickFrom]
    rw [this, Measure.map_const, measure_univ, one_smul, harvestMeasure, if_pos rfl]
  have hn : 0 < zs.length := List.length_pos_of_ne_nil hzs
  set n := zs.length
  have hF : Measurable fun u : unitInterval => ⌊(u : ℝ) * n⌋₊ :=
    (continuous_subtype_val.measurable.mul_const _).nat_floor
  have hlaw : (volume : Measure unitInterval).map (fun u : unitInterval => ⌊(u : ℝ) * n⌋₊)
      = ((n : ℝ≥0∞))⁻¹ • ∑ i ∈ Finset.range n, Measure.dirac i := by
    refine Measure.ext_of_singleton fun j => ?_
    rw [Measure.map_apply hF (measurableSet_singleton j), Measure.smul_apply,
      Measure.coe_finset_sum, Finset.sum_apply]
    simp only [Set.preimage, Set.mem_singleton_iff]
    rcases lt_or_ge j n with hj | hj
    · rw [volume_floor_eq hn hj, Finset.sum_eq_single j]
      · simp
      · intro b _ hb
        simp [Measure.dirac_apply' _ (measurableSet_singleton j), Set.indicator, hb]
      · intro h; exact absurd (Finset.mem_range.2 hj) h
    · rw [volume_floor_ge hn hj, Finset.sum_eq_zero]
      · simp
      · intro b hb
        have : b ≠ j := by simp at hb; omega
        simp [Measure.dirac_apply' _ (measurableSet_singleton j), Set.indicator, this]
  have hpick : pickFrom zs = (fun i => zs.getD i 1) ∘ fun u : unitInterval => ⌊(u : ℝ) * n⌋₊ :=
    rfl
  rw [hpick, ← Measure.map_map (measurable_from_nat) hF, hlaw, harvestMeasure, if_neg hzs,
    list_sum_dirac, Measure.map_smul]
  congr 1
  refine Measure.ext_of_singleton fun a => ?_
  rw [Measure.map_apply measurable_from_nat (measurableSet_singleton a), Measure.coe_finset_sum,
    Finset.sum_apply, Measure.coe_finset_sum, Finset.sum_apply]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Measure.dirac_apply' _ (measurableSet_of_countable _)]
  simp [Set.indicator, Set.preimage]

instance (zs : List (FreeMonoid α)) : IsProbabilityMeasure (harvestMeasure zs) := by
  rw [← map_pickFrom]
  exact Measure.isProbabilityMeasure_map (measurable_pickFrom zs).aemeasurable

instance {K : ℕ} (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (harvests : List (List (FreeMonoid α))) (j : Option (Fin K)) :
    IsProbabilityMeasure (loopDistributions D harvests j) := by
  cases j <;> simp only [loopDistributions] <;> infer_instance

section Run

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

/-- The populations' prefix streams: the probes' own, and each harvest's picks. -/
def popStreams (harvests : List (List (FreeMonoid α)))
    (w : (ℕ → FreeMonoid α) × (Fin K → ℕ → unitInterval)) : Option (Fin K) → ℕ → FreeMonoid α
  | none => w.1
  | some i => fun n => pickFrom (harvests.getD i []) (w.2 i n)

lemma measurePreserving_popStreams (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (harvests : List (List (FreeMonoid α))) :
    MeasurePreserving (popStreams (K := K) harvests)
      ((Measure.infinitePi fun _ : ℕ => D).prod
        (Measure.pi fun _ : Fin K => Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)))
      (Measure.pi fun j : Option (Fin K) =>
        Measure.infinitePi fun _ : ℕ => loopDistributions D harvests j) := by
  have hpick : MeasurePreserving
      (fun U : Fin K → ℕ → unitInterval => fun (i : Fin K) n => pickFrom (harvests.getD i []) (U i n))
      (Measure.pi fun _ : Fin K => Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval))
      (Measure.pi fun i : Fin K =>
        Measure.infinitePi fun _ : ℕ => loopDistributions D harvests (some i)) := by
    have hf : ∀ i : Fin K, Measurable fun u : ℕ → unitInterval =>
        fun n => pickFrom (harvests.getD i []) (u n) := fun i =>
      measurable_pi_lambda _ fun n => (measurable_pickFrom _).comp (measurable_pi_apply n)
    refine ⟨measurable_pi_lambda _ fun i => (hf i).comp (measurable_pi_apply i), ?_⟩
    rw [Measure.pi_map_pi (fun i => (hf i).aemeasurable)]
    congr 1
    funext i
    rw [Measure.infinitePi_map_pi (μ := fun _ : ℕ => (volume : Measure unitInterval))
      (hf := fun _ => measurable_pickFrom _), map_pickFrom]
    rfl
  have he : MeasurePreserving
      (MeasurableEquiv.piOptionEquivProd fun _ : Option (Fin K) => ℕ → FreeMonoid α).symm
      ((Measure.pi fun i : Fin K =>
        Measure.infinitePi fun _ : ℕ => loopDistributions D harvests (some i)).prod
        (Measure.infinitePi fun _ : ℕ => loopDistributions D harvests none))
      (Measure.pi fun j : Option (Fin K) =>
        Measure.infinitePi fun _ : ℕ => loopDistributions D harvests j) :=
    ⟨(MeasurableEquiv.piOptionEquivProd _).symm.measurable, Measure.pi_map_piOptionEquivProd
      (fun j : Option (Fin K) => Measure.infinitePi fun _ : ℕ => loopDistributions D harvests j)⟩
  have hc := he.comp ((hpick.prod (MeasurePreserving.id _)).comp Measure.measurePreserving_swap)
  convert hc using 1
  all_goals first
    | rfl
    | (funext w j; cases j <;> rfl)

/-- At given harvests, a round's clustering draws give a run drawn by `runMeasure` at the
harvests' populations. -/
lemma measurePreserving_clusterRun (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    [IsProbabilityMeasure Dsf] (harvests : List (List (FreeMonoid α))) :
    MeasurePreserving (clusterRun (Ω := Ω) (K := K) harvests) (clusterLaw μ D Dsf K)
      (runMeasure μ (loopDistributions D harvests) Dsf) := by
  set Dp := Measure.infinitePi fun _ : ℕ => D
  set PU := Measure.pi fun _ : Fin K => Measure.infinitePi fun _ : ℕ =>
    (volume : Measure unitInterval)
  set PL := Measure.pi fun j : Option (Fin K) =>
    Measure.infinitePi fun _ : ℕ => loopDistributions D harvests j
  have hP := measurePreserving_popStreams (K := K) D harvests
  have f1 := measurePreserving_prodAssoc Dp Dp (PU.prod PU)
  have f2 := (MeasurePreserving.id Dp).prod (measurePreserving_prodAssoc Dp PU PU).symm
  have f3 := (MeasurePreserving.id Dp).prod
    ((Measure.measurePreserving_swap (μ := Dp) (ν := PU)).prod (MeasurePreserving.id PU))
  have f4 := (MeasurePreserving.id Dp).prod (measurePreserving_prodAssoc PU Dp PU)
  have f5 := (measurePreserving_prodAssoc Dp PU (Dp.prod PU)).symm
  have hR := f5.comp (f4.comp (f3.comp (f2.comp f1)))
  have hin := (MeasurePreserving.id (Measure.infinitePi fun _ : ℕ => Dsf)).prod
    ((hP.prod hP).comp hR)
  have hassoc := (measurePreserving_prodAssoc (Measure.infinitePi fun _ : ℕ => Dsf) PL PL).symm
  have h := (MeasurePreserving.id μ).prod (hassoc.comp hin)
  exact h

end Run

end OrthoDFA
