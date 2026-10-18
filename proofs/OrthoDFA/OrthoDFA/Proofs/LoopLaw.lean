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

/-- A stream of uniforms, each drawing from `ν`, is a stream of draws from `ν`. -/
lemma measurePreserving_drawStream (ν : Measure (FreeMonoid α)) [IsProbabilityMeasure ν] :
    MeasurePreserving (fun u : ℕ → unitInterval => fun n => drawFrom ν (u n))
      (Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval))
      (Measure.infinitePi fun _ : ℕ => ν) :=
  ⟨measurable_pi_lambda _ fun n => (measurable_drawFrom _).comp (measurable_pi_apply n), by
    rw [Measure.infinitePi_map_pi (μ := fun _ : ℕ => (volume : Measure unitInterval))
      (hf := fun _ => measurable_drawFrom _), map_drawFrom]⟩

/-- Each population's stream of uniforms drawing from its law. -/
lemma measurePreserving_popDraws (L : Option (Fin K) → Measure (FreeMonoid α))
    [∀ j, IsProbabilityMeasure (L j)] :
    MeasurePreserving (fun U : Option (Fin K) → ℕ → unitInterval => fun j n => drawFrom (L j) (U j n))
      (Measure.pi fun _ : Option (Fin K) =>
        Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval))
      (Measure.pi fun j : Option (Fin K) => Measure.infinitePi fun _ : ℕ => L j) := by
  refine ⟨measurable_pi_lambda _ fun j => (measurePreserving_drawStream (L j)).measurable.comp
    (measurable_pi_apply j), ?_⟩
  rw [Measure.pi_map_pi (fun j => (measurePreserving_drawStream (L j)).measurable.aemeasurable)]
  congr 1
  funext j
  exact (measurePreserving_drawStream (L j)).map_eq

instance (ν : Measure (FreeMonoid α)) (T : Finset (FreeMonoid α)) :
    IsProbabilityMeasure (offLaw ν T) := by
  unfold offLaw; infer_instance

/-- After the history `h`, a round's clustering draws give a run drawn by `runMeasure` at the
populations and suffixes conditioned off what the earlier rounds read. -/
lemma measurePreserving_clusterRun (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α) :
    MeasurePreserving (clusterRun (Ω := Ω) (K := K) D Dsf h) (clusterLaw μ α K)
      (runMeasure μ (fun j => offLaw (loopDistributions D h.pops j) h.prefs)
        (offLaw Dsf h.suffs)) := by
  set L : Option (Fin K) → Measure (FreeMonoid α) :=
    fun j => offLaw (loopDistributions D h.pops j) h.prefs
  have hS := measurePreserving_drawStream (offLaw Dsf h.suffs)
  have hP := measurePreserving_popDraws L
  set US := Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)
  set UP := Measure.pi fun _ : Option (Fin K) =>
    Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)
  have hassoc := (measurePreserving_prodAssoc US UP UP).symm
  have hin := ((hS.prod hP).prod hP).comp hassoc
  exact (MeasurePreserving.id μ).prod hin

omit [IsProbabilityMeasure μ] in
lemma measurable_clusterRun (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α) :
    Measurable (clusterRun (Ω := Ω) (K := K) D Dsf h) := by
  unfold clusterRun
  refine measurable_fst.prodMk (Measurable.prodMk (Measurable.prodMk ?_ ?_) ?_)
  · exact measurable_pi_lambda _ fun n => (measurable_drawFrom _).comp
      ((measurable_pi_apply n).comp (measurable_fst.comp measurable_snd))
  · exact measurable_pi_lambda _ fun j => measurable_pi_lambda _ fun n =>
      (measurable_drawFrom _).comp ((measurable_pi_apply n).comp ((measurable_pi_apply j).comp
        (measurable_fst.comp (measurable_snd.comp measurable_snd))))
  · exact measurable_pi_lambda _ fun j => measurable_pi_lambda _ fun n =>
      (measurable_drawFrom _).comp ((measurable_pi_apply n).comp ((measurable_pi_apply j).comp
        (measurable_snd.comp (measurable_snd.comp measurable_snd))))

end Run

end OrthoDFA
