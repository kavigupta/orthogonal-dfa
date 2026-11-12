import RLStar.Proofs.LoopE2Round
import RLStar.Proofs.LoopGood
import RLStar.LowLevelClaims.Trie

/-!
# A round's probes and batches are long

Where `D` almost surely draws strings at least `k` long, so do level 3's probes, drawn from `D` off
the clustering's keys (`pi_not_l3Good_le`), and a harvested population's batches of draws from `D`
(`batches_bad_le`).
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory

open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

section Pairs

variable {ι : Type*} [Fintype ι] [DecidableEq ι] (ν : Measure (FreeMonoid α))
  [IsProbabilityMeasure ν]

omit [DecidableEq α] in
lemma pi_coord_eq' (i : ι) (A : Set (FreeMonoid α)) :
    (Measure.pi fun _ : ι => ν) {d | d i ∈ A} = ν A :=
  (measurePreserving_eval (fun _ : ι => ν) i).measure_preimage
    (Set.to_countable A).measurableSet.nullMeasurableSet

omit [DecidableEq α] in
lemma map_pi_drawFrom :
    (Measure.pi fun _ : ι => (volume : Measure unitInterval)).map (fun u i => drawFrom ν (u i))
      = Measure.pi fun _ : ι => ν := by
  rw [Measure.pi_map_pi (fun _ => (measurable_drawFrom ν).aemeasurable)]
  simp only [map_drawFrom]

end Pairs

lemma avoidLaw_ae (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (K₀ : Finset (FreeMonoid α)) (hm : D.real (avoiding k ↑K₀)ᶜ < 1) {Pr : FreeMonoid α → Prop}
    (h : ∀ᵐ x ∂D, Pr x) : ∀ᵐ x ∂(avoidLaw D k K₀), Pr x := by
  set A := avoiding (α := α) k ↑K₀
  have hA : D.real A = 1 - D.real Aᶜ := by
    rw [probReal_compl_eq_one_sub (measurableSet_of_countable A)]; ring
  have hApos : 0 < D.real A := by rw [hA]; linarith
  rw [avoidLaw_eq]
  unfold popLaw
  split_ifs with h0
  · exfalso
    rcases h0 with h0 | h0
    · rw [Measure.restrict_apply_univ] at h0
      rw [measureReal_def, h0, ENNReal.toReal_zero] at hApos
      exact lt_irrefl _ hApos
    · exact measure_ne_top _ _ h0
  · exact Measure.ae_smul_measure (ae_restrict_of_ae h) _

/-! ## Level 3's probes -/

/-- `P` probes from `avoidLaw D k K₀` fail `L3Good` with chance `0`. -/
theorem pi_not_l3Good_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (K₀ : Finset (FreeMonoid α)) {P : ℕ} {p₀ : ℝ}
    (hpre : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (hk : ∀ᵐ x ∂D, k ≤ x.toList.length)
    (hK : K₀.card * p₀ ≤ 1 / 2) :
    (Measure.pi fun _ : Fin P => (volume : Measure unitInterval)) {u | ¬ L3Good k D K₀ u} = 0 := by
  classical
  have hpm : prefixMax D k ≤ p₀ := ciSup_le hpre
  have hm : D.real (avoiding k ↑K₀)ᶜ ≤ 1 / 2 :=
    (real_compl_avoiding_le D k K₀).trans
      ((mul_le_mul_of_nonneg_left hpm (Nat.cast_nonneg _)).trans hK)
  set ν := avoidLaw D k K₀
  have : IsProbabilityMeasure ν := by unfold ν avoidLaw; infer_instance
  have hkA : ∀ᵐ x ∂ν, k ≤ x.toList.length := avoidLaw_ae D k K₀ (by linarith) hk
  set Bad : Set (Fin P → FreeMonoid α) := ⋃ i, {d | d i ∈ {x : FreeMonoid α | x.toList.length < k}}
  have hsub : {u : Fin P → unitInterval | ¬ L3Good k D K₀ u}
      ⊆ (fun u i => drawFrom ν (u i)) ⁻¹' Bad := by
    intro u hu
    simp only [L3Good, not_forall, not_le] at hu
    simp only [Set.mem_preimage, Bad, Set.mem_iUnion, Set.mem_setOf_eq]
    exact hu
  have hmeas : Measurable fun (u : Fin P → unitInterval) (i : Fin P) => drawFrom ν (u i) :=
    measurable_pi_lambda _ fun i => (measurable_drawFrom ν).comp (measurable_pi_apply i)
  refine measure_mono_null hsub ?_
  rw [← Measure.map_apply hmeas (Set.to_countable Bad).measurableSet, map_pi_drawFrom]
  refine measure_iUnion_null fun i => ?_
  rw [pi_coord_eq' ν i]
  exact measure_mono_null (fun (x : FreeMonoid α) (hx : x.toList.length < k) =>
    show ¬ k ≤ x.toList.length from not_le.2 hx) (ae_iff.1 hkA)

/-! ## A harvested population's batches -/

/-- The batches' uniforms, the `t`-th of the `n`-th batch at `(n, t)`. -/
def batchUniforms (N M : ℕ) (u : ℕ → unitInterval × (Fin N → unitInterval)) :
    Fin M × Fin N → unitInterval :=
  fun j => (u j.1).2 j.2

lemma measurable_batchUniforms (N M : ℕ) : Measurable (batchUniforms N M) :=
  measurable_pi_lambda _ fun j => (measurable_pi_apply j.2).comp
    (measurable_snd.comp (measurable_pi_apply (j.1 : ℕ)))

lemma map_batchUniforms (N M : ℕ) :
    (Measure.infinitePi fun _ : ℕ => drawLaw N).map (batchUniforms N M)
      = Measure.pi fun _ : Fin M × Fin N => (volume : Measure unitInterval) := by
  classical
  refine (Measure.pi_eq fun s hs => ?_).symm
  rw [Measure.map_apply (measurable_batchUniforms N M) (MeasurableSet.univ_pi hs)]
  set t : ℕ → Set (unitInterval × (Fin N → unitInterval)) := fun n =>
    if h : n < M then Set.univ ×ˢ Set.univ.pi fun i => s (⟨n, h⟩, i) else Set.univ
  have e : batchUniforms N M ⁻¹' Set.univ.pi s
      = Set.pi ((Finset.univ : Finset (Fin M)).map Fin.valEmbedding : Finset ℕ) t := by
    ext u
    simp only [Set.mem_preimage, Set.mem_pi, Set.mem_univ, true_implies, Finset.coe_map,
      Finset.coe_univ, Set.image_univ, Set.mem_range, forall_exists_index, batchUniforms]
    constructor
    · rintro h _ m rfl
      simp only [t, Fin.valEmbedding_apply, Fin.is_lt, dif_pos, Set.mem_prod, Set.mem_univ,
        Set.mem_pi, true_implies, true_and]
      exact fun i => h (m, i)
    · intro h j
      have := h j.1 j.1 rfl
      simp only [t, Fin.valEmbedding_apply, Fin.is_lt, dif_pos, Set.mem_prod, Set.mem_univ,
        Set.mem_pi, true_implies, true_and] at this
      exact this j.2
  rw [e, Measure.infinitePi_pi]
  · rw [Finset.prod_map, ← Finset.univ_product_univ, Finset.prod_product]
    refine Finset.prod_congr rfl fun m _ => ?_
    simp only [t, Fin.valEmbedding_apply, Fin.is_lt, dif_pos, Fin.eta, drawLaw,
      Measure.prod_prod, Measure.pi_pi, measure_univ, one_mul]
  · intro n _
    simp only [t]
    split_ifs
    · exact MeasurableSet.univ.prod (MeasurableSet.univ_pi fun i => hs _)
    · exact MeasurableSet.univ

/-- `M` batches of `N` draws from `D` fail `BatchesGood` with chance `0`. -/
theorem batches_bad_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k N M : ℕ)
    (hk : ∀ᵐ x ∂D, k ≤ x.toList.length) :
    (Measure.infinitePi fun _ : ℕ => drawLaw N) {u | ¬ BatchesGood k D N M u} = 0 := by
  classical
  set Bad : Set (Fin M × Fin N → FreeMonoid α) :=
    ⋃ j, {d | d j ∈ {x : FreeMonoid α | x.toList.length < k}}
  set Φ := fun u : ℕ → unitInterval × (Fin N → unitInterval) =>
    fun j => drawFrom D (batchUniforms N M u j)
  have hΦ : Measurable Φ := measurable_pi_lambda _ fun j =>
    (measurable_drawFrom D).comp ((measurable_pi_apply j).comp (measurable_batchUniforms N M))
  have hsub : {u | ¬ BatchesGood k D N M u} ⊆ Φ ⁻¹' Bad := by
    intro u hu
    simp only [BatchesGood, not_forall, not_le] at hu
    obtain ⟨n, hn, t, h⟩ := hu
    simp only [Set.mem_preimage, Bad, Set.mem_iUnion, Set.mem_setOf_eq, Φ, batchUniforms]
    exact ⟨(⟨n, hn⟩, t), h⟩
  refine measure_mono_null hsub ?_
  rw [← Measure.map_apply hΦ (Set.to_countable Bad).measurableSet]
  have hlaw : (Measure.infinitePi fun _ : ℕ => drawLaw N).map Φ
      = Measure.pi fun _ : Fin M × Fin N => D := by
    have hg : Measurable fun (v : Fin M × Fin N → unitInterval) (j : Fin M × Fin N) =>
        drawFrom D (v j) :=
      measurable_pi_lambda _ fun j => (measurable_drawFrom D).comp (measurable_pi_apply j)
    calc (Measure.infinitePi fun _ : ℕ => drawLaw N).map Φ
        = ((Measure.infinitePi fun _ : ℕ => drawLaw N).map (batchUniforms N M)).map
          fun v j => drawFrom D (v j) := (Measure.map_map hg (measurable_batchUniforms N M)).symm
      _ = Measure.pi fun _ : Fin M × Fin N => D := by
        rw [map_batchUniforms, map_pi_drawFrom]
  rw [hlaw]
  refine measure_iUnion_null fun j => ?_
  rw [pi_coord_eq' D j]
  exact measure_mono_null (fun (x : FreeMonoid α) (hx : x.toList.length < k) =>
    show ¬ k ≤ x.toList.length from not_le.2 hx) (ae_iff.1 hk)

lemma measurableSet_l3Good (k : ℕ) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (K₀ : Finset (FreeMonoid α)) {P : ℕ} :
    MeasurableSet {u : Fin P → unitInterval | L3Good k D K₀ u} := by
  have : IsProbabilityMeasure (avoidLaw D k K₀) := by unfold avoidLaw; infer_instance
  have hmeas : Measurable fun (u : Fin P → unitInterval) (i : Fin P) =>
      drawFrom (avoidLaw D k K₀) (u i) :=
    measurable_pi_lambda _ fun i => (measurable_drawFrom _).comp (measurable_pi_apply i)
  exact hmeas (Set.to_countable {d : Fin P → FreeMonoid α |
    ∀ i, k ≤ (d i).toList.length}).measurableSet

lemma measurableSet_batchesGood (k : ℕ) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (N M : ℕ) :
    MeasurableSet {u | BatchesGood k D N M u} := by
  have hΦ : Measurable fun u : ℕ → unitInterval × (Fin N → unitInterval) =>
      fun j : Fin M × Fin N => drawFrom D (batchUniforms N M u j) :=
    measurable_pi_lambda _ fun j =>
      (measurable_drawFrom D).comp ((measurable_pi_apply j).comp (measurable_batchUniforms N M))
  have e : {u | BatchesGood k D N M u} = (fun u : ℕ → unitInterval × (Fin N → unitInterval) =>
      fun j : Fin M × Fin N => drawFrom D (batchUniforms N M u j)) ⁻¹'
      {d | ∀ j : Fin M × Fin N, k ≤ (d j).toList.length} := by
    ext u
    simp only [Set.mem_setOf_eq, Set.mem_preimage, BatchesGood, batchUniforms]
    exact ⟨fun h j => h j.1 j.1.2 j.2, fun h n hn t => h (⟨n, hn⟩, t)⟩
  rw [e]
  exact hΦ (Set.to_countable _).measurableSet

end OrthoDFA

end
