import OrthoDFA.Proofs.LoopRounds
import OrthoDFA.Proofs.Replay
import Mathlib.Probability.ProductMeasure
import Mathlib.MeasureTheory.Constructions.UnitInterval
import Mathlib.MeasureTheory.Function.Floor
import Mathlib.Data.List.GetD

/-!
# A round's clustering run, at given populations, is drawn as the clustering's theorem draws it

A uniform `u` lands where `ν`'s running mass over the strings' numbering first passes it, at a
string with chance its mass, so a draw is from `ν` (`map_drawFrom`). The round's draws, run at
given populations, are then a run drawn by `runMeasure` at those populations
(`measurePreserving_clusterRun`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The uniform lands in `[a, b)` with chance `b - a`. -/
lemma volume_Ico_unit {a b : ℝ} (ha : 0 ≤ a) (hab : a ≤ b) (hb : b ≤ 1) :
    (volume : Measure unitInterval) {u | a ≤ (u : ℝ) ∧ (u : ℝ) < b} = ENNReal.ofReal (b - a) := by
  have hset : Subtype.val '' {u : unitInterval | a ≤ (u : ℝ) ∧ (u : ℝ) < b} = Set.Ico a b := by
    ext t
    simp only [Set.mem_image, Set.mem_setOf_eq, Set.mem_Ico]
    constructor
    · rintro ⟨u, hu, rfl⟩
      exact hu
    · rintro ⟨h1, h2⟩
      exact ⟨⟨t, by linarith, by linarith⟩, ⟨h1, h2⟩, rfl⟩
  rw [unitInterval.volume_apply, hset, Real.volume_Ico]

lemma strCode_injective : Function.Injective (strCode (α := α)) :=
  Classical.choose_spec (exists_injective_nat (FreeMonoid α))

lemma strAt_strCode (z : FreeMonoid α) : strAt (strCode z) = some z := by
  have h : ∃ z', strCode z' = strCode z := ⟨z, rfl⟩
  simp only [strAt, dif_pos h, Option.some.injEq]
  exact strCode_injective h.choose_spec

lemma strCode_of_strAt {n : ℕ} {z : FreeMonoid α} (h : strAt n = some z) : strCode z = n := by
  unfold strAt at h
  split_ifs at h with h'
  simp only [Option.some.injEq] at h
  rw [← h]
  exact h'.choose_spec

lemma massBelow_succ (ν : Measure (FreeMonoid α)) (n : ℕ) :
    massBelow ν (n + 1) = massBelow ν n + ν.real {z | strCode z = n} := by
  simp only [massBelow, Finset.sum_range_succ]

lemma massBelow_mono (ν : Measure (FreeMonoid α)) : Monotone (massBelow ν) := by
  refine monotone_nat_of_le_succ fun n => ?_
  rw [massBelow_succ]
  have : 0 ≤ ν.real {z | strCode z = n} := measureReal_nonneg
  linarith

lemma massBelow_eq (ν : Measure (FreeMonoid α)) [IsFiniteMeasure ν] (n : ℕ) :
    massBelow ν n = ν.real {z | strCode z < n} := by
  have hset : {z : FreeMonoid α | strCode z < n} = ⋃ i ∈ Finset.range n, {z | strCode z = i} := by
    ext z; simp
  rw [hset, measureReal_biUnion_finset (fun i _ j _ hij => Set.disjoint_left.2 fun z h1 h2 =>
    hij (h1.symm.trans h2)) fun i _ => measurableSet_of_countable _]
  rfl

/-- Below any `u < ν univ`, the running mass passes `u`. -/
lemma exists_lt_massBelow (ν : Measure (FreeMonoid α)) [IsFiniteMeasure ν] {u : ℝ}
    (hu : u < ν.real Set.univ) : ∃ n, u < massBelow ν (n + 1) := by
  have hlim : Filter.Tendsto (fun n => ν {z : FreeMonoid α | strCode z < n + 1}) Filter.atTop
      (nhds (ν (⋃ n : ℕ, {z : FreeMonoid α | strCode z < n + 1}))) :=
    tendsto_measure_iUnion_atTop fun m n hmn z (hz : strCode z < m + 1) =>
      (show strCode z < n + 1 by omega)
  have hU : (⋃ n : ℕ, {z : FreeMonoid α | strCode z < n + 1}) = Set.univ := by
    ext z
    simp only [Set.mem_iUnion, Set.mem_setOf_eq, Set.mem_univ, iff_true]
    exact ⟨strCode z, by omega⟩
  rw [hU] at hlim
  have hlim' := (ENNReal.tendsto_toReal (measure_ne_top ν _)).comp hlim
  obtain ⟨n, hn⟩ := (hlim'.eventually (lt_mem_nhds hu)).exists
  exact ⟨n, by rw [massBelow_eq]; exact hn⟩

/-- Where the draw lands: in its string's slot, or, past all of `ν`'s mass, at `1`. -/
lemma drawFrom_eq_iff (ν : Measure (FreeMonoid α)) [IsFiniteMeasure ν] (u : unitInterval)
    (z : FreeMonoid α) :
    drawFrom ν u = z ↔ (massBelow ν (strCode z) ≤ u ∧ (u : ℝ) < massBelow ν (strCode z + 1))
      ∨ (z = 1 ∧ ∀ n, massBelow ν (n + 1) ≤ u) := by
  classical
  unfold drawFrom
  by_cases h : ∃ n, (u : ℝ) < massBelow ν (n + 1)
  · rw [dif_pos h]
    have hspec := Nat.find_spec h
    have hlow : massBelow ν (Nat.find h) ≤ u := by
      rcases hm : Nat.find h with _ | m
      · simp [massBelow, u.2.1]
      · have := Nat.find_min h (show m < Nat.find h by omega)
        push Not at this
        exact this
    constructor
    · intro hz
      left
      rcases hs : strAt (α := α) (Nat.find h) with _ | z'
      · exfalso
        have hempty : {z : FreeMonoid α | strCode z = Nat.find h} = ∅ := by
          refine Set.eq_empty_of_forall_notMem fun z hz => ?_
          have := strAt_strCode z
          rw [Set.mem_setOf_eq.mp hz, hs] at this
          cases this
        have := massBelow_succ ν (Nat.find h)
        rw [hempty, measureReal_empty, add_zero] at this
        linarith
      · rw [hs, Option.getD_some] at hz
        subst hz
        rw [strCode_of_strAt hs]
        exact ⟨hlow, hspec⟩
    · rintro (⟨h1, h2⟩ | ⟨-, h2⟩)
      · have hfind : Nat.find h = strCode z := by
          refine (Nat.find_eq_iff h).2 ⟨h2, fun m hm hlt => ?_⟩
          have := massBelow_mono ν (show m + 1 ≤ strCode z by omega)
          linarith
        rw [hfind, strAt_strCode, Option.getD_some]
      · exfalso
        obtain ⟨n, hn⟩ := h
        linarith [h2 n]
  · rw [dif_neg h]
    push Not at h
    constructor
    · intro hz
      exact .inr ⟨hz.symm, h⟩
    · rintro (⟨-, h2⟩ | ⟨hz, -⟩)
      · linarith [h (strCode z)]
      · exact hz.symm

lemma measurable_drawFrom (ν : Measure (FreeMonoid α)) [IsFiniteMeasure ν] :
    Measurable (drawFrom ν) := by
  refine measurable_to_countable' fun z => ?_
  have hset : drawFrom ν ⁻¹' {z} = ({u : unitInterval | massBelow ν (strCode z) ≤ u}
      ∩ {u | (u : ℝ) < massBelow ν (strCode z + 1)})
      ∪ ({_u : unitInterval | z = 1} ∩ ⋂ n, {u | massBelow ν (n + 1) ≤ (u : ℝ)}) := by
    ext u
    simp only [Set.mem_preimage, Set.mem_singleton_iff, drawFrom_eq_iff, Set.mem_union,
      Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_iInter]
  rw [hset]
  have hc : Measurable fun u : unitInterval => (u : ℝ) := continuous_subtype_val.measurable
  refine ((measurableSet_le measurable_const hc).inter (measurableSet_lt hc measurable_const)).union
    ((MeasurableSet.const _).inter (MeasurableSet.iInter fun n =>
      measurableSet_le measurable_const hc))

/-- A draw is from `ν`. -/
lemma map_drawFrom (ν : Measure (FreeMonoid α)) [IsProbabilityMeasure ν] :
    (volume : Measure unitInterval).map (drawFrom ν) = ν := by
  refine Measure.ext_of_singleton fun z => ?_
  rw [Measure.map_apply (measurable_drawFrom ν) (measurableSet_singleton z)]
  have hset : drawFrom ν ⁻¹' {z} ⊆ {u : unitInterval | massBelow ν (strCode z) ≤ u
      ∧ (u : ℝ) < massBelow ν (strCode z + 1)} ∪ {1} := by
    intro u hu
    rcases (drawFrom_eq_iff ν u z).1 hu with h | ⟨-, h⟩
    · exact .inl h
    · right
      by_contra hne
      have hu1 : (u : ℝ) < ν.real Set.univ := by
        rw [probReal_univ]
        exact lt_of_le_of_ne u.2.2 fun h' => hne (Subtype.ext h')
      obtain ⟨n, hn⟩ := exists_lt_massBelow ν hu1
      linarith [h n]
  have hsub : {u : unitInterval | massBelow ν (strCode z) ≤ u
      ∧ (u : ℝ) < massBelow ν (strCode z + 1)} ⊆ drawFrom ν ⁻¹' {z} := fun u hu =>
    (drawFrom_eq_iff ν u z).2 (.inl hu)
  have h0 : 0 ≤ massBelow ν (strCode z) := by
    rw [massBelow_eq]; exact measureReal_nonneg
  have h1 : massBelow ν (strCode z + 1) ≤ 1 := by
    rw [massBelow_eq]; exact measureReal_le_one
  have hle := massBelow_mono ν (Nat.le_succ (strCode z))
  have hslot : (volume : Measure unitInterval) {u : unitInterval | massBelow ν (strCode z) ≤ u
      ∧ (u : ℝ) < massBelow ν (strCode z + 1)} = ν {z} := by
    rw [volume_Ico_unit h0 hle h1, massBelow_succ, add_sub_cancel_left]
    have : {z' : FreeMonoid α | strCode z' = strCode z} = {z} := by
      ext z'; simp [strCode_injective.eq_iff]
    rw [this, ofReal_measureReal (measure_ne_top _ _)]
  refine le_antisymm ?_ ?_
  · calc _ ≤ (volume : Measure unitInterval) ({u : unitInterval | massBelow ν (strCode z) ≤ u
          ∧ (u : ℝ) < massBelow ν (strCode z + 1)} ∪ {1}) := measure_mono hset
      _ ≤ _ := measure_union_le _ _
      _ = ν {z} := by
          have h1 : (volume : Measure unitInterval) {1} = 0 := by
            rw [unitInterval.volume_apply, Set.image_singleton, Real.volume_singleton]
          rw [hslot, h1, add_zero]
  · rw [← hslot]
    exact measure_mono hsub

instance (ν : Measure (FreeMonoid α)) : IsProbabilityMeasure (popLaw ν) := by
  unfold popLaw
  split_ifs with h
  · infer_instance
  · push Not at h
    constructor
    rw [Measure.smul_apply, smul_eq_mul, ENNReal.inv_mul_cancel h.1 h.2]

lemma popLaw_real {ν : Measure (FreeMonoid α)} (h0 : ν Set.univ ≠ 0) (htop : ν Set.univ ≠ ∞)
    (A : Set (FreeMonoid α)) : (popLaw ν).real A = ν.real A / ν.real Set.univ := by
  unfold popLaw
  rw [if_neg (by tauto), measureReal_def, Measure.smul_apply, smul_eq_mul, ENNReal.toReal_mul,
    ENNReal.toReal_inv, div_eq_inv_mul]
  rfl

lemma isProbabilityMeasure_loopDistributions {K : ℕ} (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (pops : List (Measure (FreeMonoid α)))
    (hpops : ∀ ν ∈ pops, IsProbabilityMeasure ν) (j : Option (Fin K)) :
    IsProbabilityMeasure (loopDistributions D pops j) := by
  cases j with
  | none => simp only [loopDistributions]; infer_instance
  | some i =>
    simp only [loopDistributions]
    by_cases hi : i.val < pops.length
    · rw [List.getD_eq_getElem (hn := hi)]
      exact hpops _ (List.getElem_mem _)
    · rw [List.getD_eq_default _ _ (not_lt.1 hi)]
      infer_instance

section Run

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

/-- The populations' prefix streams: the probes' own, and each harvest population's draws. -/
def popStreams (pops : List (Measure (FreeMonoid α)))
    (w : (ℕ → FreeMonoid α) × (Fin K → ℕ → unitInterval)) : Option (Fin K) → ℕ → FreeMonoid α
  | none => w.1
  | some i => fun n => drawFrom (pops.getD i (Measure.dirac 1)) (w.2 i n)

lemma measurePreserving_popStreams (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (pops : List (Measure (FreeMonoid α))) (hpops : ∀ ν ∈ pops, IsProbabilityMeasure ν) :
    MeasurePreserving (popStreams (K := K) pops)
      ((Measure.infinitePi fun _ : ℕ => D).prod
        (Measure.pi fun _ : Fin K =>
          Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)))
      (Measure.pi fun j : Option (Fin K) =>
        Measure.infinitePi fun _ : ℕ => loopDistributions D pops j) := by
  have hP : ∀ i : Fin K, IsProbabilityMeasure (pops.getD i (Measure.dirac 1)) := fun i =>
    isProbabilityMeasure_loopDistributions D pops hpops (some i)
  have hdraw : MeasurePreserving
      (fun U : Fin K → ℕ → unitInterval => fun (i : Fin K) n =>
        drawFrom (pops.getD i (Measure.dirac 1)) (U i n))
      (Measure.pi fun _ : Fin K => Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval))
      (Measure.pi fun i : Fin K =>
        Measure.infinitePi fun _ : ℕ => loopDistributions D pops (some i)) := by
    have hf : ∀ i : Fin K, Measurable fun u : ℕ → unitInterval =>
        fun n => drawFrom (pops.getD i (Measure.dirac 1)) (u n) := fun i =>
      measurable_pi_lambda _ fun n => (measurable_drawFrom _).comp (measurable_pi_apply n)
    refine ⟨measurable_pi_lambda _ fun i => (hf i).comp (measurable_pi_apply i), ?_⟩
    rw [Measure.pi_map_pi (fun i => (hf i).aemeasurable)]
    congr 1
    funext i
    rw [Measure.infinitePi_map_pi (μ := fun _ : ℕ => (volume : Measure unitInterval))
      (hf := fun _ => measurable_drawFrom _), map_drawFrom]
    rfl
  have : ∀ j, IsProbabilityMeasure (loopDistributions (K := K) D pops j) :=
    isProbabilityMeasure_loopDistributions D pops hpops
  have he : MeasurePreserving
      (MeasurableEquiv.piOptionEquivProd fun _ : Option (Fin K) => ℕ → FreeMonoid α).symm
      ((Measure.pi fun i : Fin K =>
        Measure.infinitePi fun _ : ℕ => loopDistributions D pops (some i)).prod
        (Measure.infinitePi fun _ : ℕ => loopDistributions D pops none))
      (Measure.pi fun j : Option (Fin K) =>
        Measure.infinitePi fun _ : ℕ => loopDistributions D pops j) :=
    ⟨(MeasurableEquiv.piOptionEquivProd _).symm.measurable, Measure.pi_map_piOptionEquivProd
      (fun j : Option (Fin K) => Measure.infinitePi fun _ : ℕ => loopDistributions D pops j)⟩
  have hc := he.comp ((hdraw.prod (MeasurePreserving.id _)).comp Measure.measurePreserving_swap)
  convert hc using 1
  all_goals first
    | rfl
    | (funext w j; cases j <;> rfl)

/-- At given populations, a round's clustering draws give a run drawn by `runMeasure` at
them. -/
lemma measurePreserving_clusterRun (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    [IsProbabilityMeasure Dsf] (pops : List (Measure (FreeMonoid α)))
    (hpops : ∀ ν ∈ pops, IsProbabilityMeasure ν) :
    MeasurePreserving (clusterRun (Ω := Ω) (K := K) pops) (clusterLaw μ D Dsf K)
      (runMeasure μ (loopDistributions D pops) Dsf) := by
  have : ∀ j, IsProbabilityMeasure (loopDistributions (K := K) D pops j) :=
    isProbabilityMeasure_loopDistributions D pops hpops
  set Dp := Measure.infinitePi fun _ : ℕ => D
  set PU := Measure.pi fun _ : Fin K => Measure.infinitePi fun _ : ℕ =>
    (volume : Measure unitInterval)
  set PL := Measure.pi fun j : Option (Fin K) =>
    Measure.infinitePi fun _ : ℕ => loopDistributions D pops j
  have hP := measurePreserving_popStreams (K := K) D pops hpops
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
