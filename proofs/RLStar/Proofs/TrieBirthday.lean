import RLStar.Proofs.LoopE2Round
import RLStar.Proofs.LoopGood
import RLStar.LowLevelClaims.Trie

/-!
# A round's probes and batches have distinct first letters

Level 3's probes, drawn from `D` off the clustering's keys, are at least `k` long with distinct
first `k` letters but for chance `P² p₀` (`pi_not_l3Good_le`), and a harvested population's `M`
batches of `N` draws from `D` miss the earlier rounds' prefixes and each other's first `k` letters
but for chance `MN (|T| + MN) p₀` (`batches_bad_le`), where every `k`-letter prefix has mass at
most `p₀`.
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
lemma pi_pair_eq {i j : ι} (hij : i ≠ j) (A : Set (FreeMonoid α)) :
    (Measure.pi fun _ : ι => ν) {d | d i ∈ A ∧ d j ∈ A} = ν A * ν A := by
  have : {d : ι → FreeMonoid α | d i ∈ A ∧ d j ∈ A}
      = Set.univ.pi fun l => if l = i ∨ l = j then A else Set.univ := by
    ext d
    simp only [Set.mem_setOf_eq, Set.mem_pi, Set.mem_univ, true_implies]
    constructor
    · rintro ⟨h1, h2⟩ l
      split_ifs with h
      · rcases h with rfl | rfl
        · exact h1
        · exact h2
      · trivial
    · intro h
      exact ⟨by simpa using h i, by simpa using h j⟩
  rw [this, Measure.pi_pi, ← Finset.prod_erase_mul _ _ (Finset.mem_univ j),
    ← Finset.prod_erase_mul _ _ (Finset.mem_erase.2 ⟨hij, Finset.mem_univ i⟩),
    Finset.prod_eq_one]
  · simp
  · intro l hl
    have h1 := (Finset.mem_erase.1 hl).1
    have h2 := (Finset.mem_erase.1 (Finset.mem_erase.1 hl).2).1
    simp [h1, h2]

omit [DecidableEq α] in
lemma ofReal_prefix_le (k : ℕ) {p₀ : ℝ} (hpre : ∀ u, ν.real {x | prefixOf x k = u} ≤ p₀)
    (u : FreeMonoid α) : ν {x | prefixOf x k = u} ≤ ENNReal.ofReal p₀ := by
  rw [← ENNReal.ofReal_toReal (measure_ne_top ν _)]
  exact ENNReal.ofReal_le_ofReal (hpre u)

omit [DecidableEq α] in
/-- Two coordinates share their first `k` letters with chance at most `p₀`. -/
lemma pi_collide_le (k : ℕ) {p₀ : ℝ} (hpre : ∀ u, ν.real {x | prefixOf x k = u} ≤ p₀) {i j : ι}
    (hij : i ≠ j) :
    (Measure.pi fun _ : ι => ν) {d | prefixOf (d i) k = prefixOf (d j) k}
      ≤ ENNReal.ofReal p₀ := by
  set S : FreeMonoid α → Set (FreeMonoid α) := fun u => {x | prefixOf x k = u}
  have hsub : {d : ι → FreeMonoid α | prefixOf (d i) k = prefixOf (d j) k}
      ⊆ ⋃ u, {d | d i ∈ S u ∧ d j ∈ S u} := fun d hd =>
    Set.mem_iUnion.2 ⟨prefixOf (d i) k, rfl, hd.symm⟩
  have hdisj : Pairwise (Function.onFun Disjoint S) := fun u v huv =>
    Set.disjoint_left.2 fun x hu hv => huv (hu.symm.trans hv)
  calc _ ≤ ∑' u, (Measure.pi fun _ : ι => ν) {d | d i ∈ S u ∧ d j ∈ S u} :=
        (measure_mono hsub).trans (measure_iUnion_le _)
    _ ≤ ∑' u, ENNReal.ofReal p₀ * ν (S u) := ENNReal.tsum_le_tsum fun u => by
        rw [pi_pair_eq ν hij]
        gcongr
        exact ofReal_prefix_le ν k hpre u
    _ = ENNReal.ofReal p₀ * ν (⋃ u, S u) := by
        rw [ENNReal.tsum_mul_left, measure_iUnion hdisj fun u => (Set.to_countable _).measurableSet]
    _ ≤ ENNReal.ofReal p₀ * 1 := by gcongr; exact prob_le_one
    _ = _ := mul_one _

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

/-- `P` probes from `avoidLaw D k K₀` fail `L3Good` with chance at most `P² p₀`. -/
theorem pi_not_l3Good_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (K₀ : Finset (FreeMonoid α)) {P : ℕ} {p₀ : ℝ}
    (hpre : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (hk : ∀ᵐ x ∂D, k ≤ x.toList.length)
    (hK : K₀.card * p₀ ≤ 1 / 2) :
    (Measure.pi fun _ : Fin P => (volume : Measure unitInterval)) {u | ¬ L3Good k D K₀ u}
      ≤ ENNReal.ofReal (P ^ 2 * p₀) := by
  classical
  have hpm : prefixMax D k ≤ p₀ := ciSup_le hpre
  have hm : D.real (avoiding k ↑K₀)ᶜ ≤ 1 / 2 :=
    (real_compl_avoiding_le D k K₀).trans
      ((mul_le_mul_of_nonneg_left hpm (Nat.cast_nonneg _)).trans hK)
  set ν := avoidLaw D k K₀
  have : IsProbabilityMeasure ν := by unfold ν avoidLaw; infer_instance
  have hpreA : ∀ u, ν.real {x | prefixOf x k = u} ≤ 2 * p₀ := fun u => by
    refine (le_prefixMax ν k u).trans ((prefixMax_avoidLaw_le D k K₀ (by linarith)).trans ?_)
    rw [div_le_iff₀ (by linarith)]
    have h0 : 0 ≤ prefixMax D k := (prefixMax_pos D k).le
    nlinarith
  have hkA : ∀ᵐ x ∂ν, k ≤ x.toList.length := avoidLaw_ae D k K₀ (by linarith) hk
  set pairs := (Finset.univ : Finset (Fin P × Fin P)).filter fun q => q.1 < q.2
  set Bad : Set (Fin P → FreeMonoid α) :=
    (⋃ i, {d | d i ∈ {x : FreeMonoid α | x.toList.length < k}})
      ∪ ⋃ q ∈ pairs, {d | prefixOf (d q.1) k = prefixOf (d q.2) k}
  have hsub : {u : Fin P → unitInterval | ¬ L3Good k D K₀ u}
      ⊆ (fun u i => drawFrom ν (u i)) ⁻¹' Bad := by
    intro u hu
    simp only [L3Good, not_and_or, not_forall, not_le] at hu
    simp only [Set.mem_preimage, Bad, Set.mem_union, Set.mem_iUnion, Set.mem_setOf_eq]
    rcases hu with ⟨i, hi⟩ | hinj
    · exact Or.inl ⟨i, hi⟩
    · right
      simp only [Function.Injective, not_forall] at hinj
      obtain ⟨i, j, hij, hne⟩ := hinj
      rcases lt_or_gt_of_ne hne with h | h
      · exact ⟨(i, j), by simp [pairs, h], hij⟩
      · exact ⟨(j, i), by simp [pairs, h], hij.symm⟩
  have hmeas : Measurable fun (u : Fin P → unitInterval) (i : Fin P) => drawFrom ν (u i) :=
    measurable_pi_lambda _ fun i => (measurable_drawFrom ν).comp (measurable_pi_apply i)
  refine (measure_mono hsub).trans ?_
  rw [← Measure.map_apply hmeas (Set.to_countable Bad).measurableSet, map_pi_drawFrom]
  calc (Measure.pi fun _ : Fin P => ν) Bad
      ≤ ∑ i, (Measure.pi fun _ : Fin P => ν) {d | d i ∈ {x : FreeMonoid α | x.toList.length < k}}
        + ∑ q ∈ pairs, (Measure.pi fun _ : Fin P => ν)
          {d | prefixOf (d q.1) k = prefixOf (d q.2) k} :=
        (measure_union_le _ _).trans (add_le_add (measure_iUnion_fintype_le _ _)
          (measure_biUnion_finset_le _ _))
    _ ≤ ∑ _i : Fin P, 0 + ∑ _q ∈ pairs, ENNReal.ofReal (2 * p₀) := by
        gcongr with i _ q hq
        · rw [pi_coord_eq' ν i]
          exact (measure_mono_null (fun (x : FreeMonoid α) (hx : x.toList.length < k) =>
            show ¬ k ≤ x.toList.length from not_le.2 hx) (ae_iff.1 hkA)).le
        · exact pi_collide_le ν k hpreA (ne_of_lt (Finset.mem_filter.1 hq).2)
    _ = ENNReal.ofReal (pairs.card * (2 * p₀)) := by
        rw [Finset.sum_const_zero, zero_add, Finset.sum_const, nsmul_eq_mul,
          ENNReal.ofReal_mul (Nat.cast_nonneg _), ENNReal.ofReal_natCast]
    _ ≤ ENNReal.ofReal (P ^ 2 * p₀) := by
        refine ENNReal.ofReal_le_ofReal ?_
        rw [Random.Spur.card_lt_pairs P]
        have h0 : 0 ≤ p₀ := measureReal_nonneg.trans (hpre 1)
        have : (P : ℝ) * (P - 1) / 2 * (2 * p₀) = (P ^ 2 - P) * p₀ := by ring
        rw [this]
        nlinarith [(Nat.cast_nonneg P : (0 : ℝ) ≤ P)]

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

/-- `M` batches of `N` draws from `D` fail `BatchesGood` with chance at most `MN (|T| + MN) p₀`. -/
theorem batches_bad_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k N M : ℕ)
    (T : Finset (FreeMonoid α)) {p₀ : ℝ} (hpre : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀)
    (hk : ∀ᵐ x ∂D, k ≤ x.toList.length) :
    (Measure.infinitePi fun _ : ℕ => drawLaw N) {u | ¬ BatchesGood k D N M T u}
      ≤ ENNReal.ofReal (M * N * (T.card * p₀) + (M * N) ^ 2 * p₀) := by
  classical
  set Bad : Set (Fin M × Fin N → FreeMonoid α) :=
    (⋃ j, {d | d j ∈ {x : FreeMonoid α | x.toList.length < k}})
      ∪ (⋃ j, ⋃ z ∈ T, {d | d j ∈ {x : FreeMonoid α | prefixOf x k = prefixOf z k}})
      ∪ ⋃ j, ⋃ j' : Fin M × Fin N, {d | j' ≠ j ∧ prefixOf (d j') k = prefixOf (d j) k}
  set Φ := fun u : ℕ → unitInterval × (Fin N → unitInterval) =>
    fun j => drawFrom D (batchUniforms N M u j)
  have hΦ : Measurable Φ := measurable_pi_lambda _ fun j =>
    (measurable_drawFrom D).comp ((measurable_pi_apply j).comp (measurable_batchUniforms N M))
  have hsub : {u | ¬ BatchesGood k D N M T u} ⊆ Φ ⁻¹' Bad := by
    intro u hu
    simp only [BatchesGood, not_forall, not_and_or, not_le, not_not] at hu
    obtain ⟨n, hn, t, h⟩ := hu
    simp only [Set.mem_preimage, Bad, Set.mem_union, Set.mem_iUnion, Set.mem_setOf_eq, Φ,
      batchUniforms]
    rcases h with h | ⟨z, hz, h⟩ | ⟨m, hm, t', h⟩
    · exact Or.inl (Or.inl ⟨(⟨n, hn⟩, t), h⟩)
    · exact Or.inl (Or.inr ⟨(⟨n, hn⟩, t), z, hz, h.symm⟩)
    · refine Or.inr ⟨(⟨n, hn⟩, t), (⟨m, by omega⟩, t'), fun he => ?_, h⟩
      have := congrArg (fun j : Fin M × Fin N => (j.1 : ℕ)) he
      simp only at this
      omega
  refine (measure_mono hsub).trans ?_
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
  set π := Measure.pi fun _ : Fin M × Fin N => D
  have hcard : (Fintype.card (Fin M × Fin N) : ℝ) = M * N := by simp
  calc π Bad
      ≤ ∑ j, π {d | d j ∈ {x : FreeMonoid α | x.toList.length < k}}
        + ∑ j, ∑ z ∈ T, π {d | d j ∈ {x : FreeMonoid α | prefixOf x k = prefixOf z k}}
        + ∑ j, ∑ j' : Fin M × Fin N, π {d | j' ≠ j ∧ prefixOf (d j') k = prefixOf (d j) k} := by
        refine (measure_union_le _ _).trans (add_le_add ?_ ?_)
        · refine (measure_union_le _ _).trans (add_le_add (measure_iUnion_fintype_le _ _) ?_)
          exact (measure_iUnion_fintype_le _ _).trans
            (Finset.sum_le_sum fun j _ => measure_biUnion_finset_le _ _)
        · exact (measure_iUnion_fintype_le _ _).trans
            (Finset.sum_le_sum fun j _ => measure_iUnion_fintype_le _ _)
    _ ≤ ∑ _j : Fin M × Fin N, 0 + ∑ _j : Fin M × Fin N, ∑ _z ∈ T, ENNReal.ofReal p₀
        + ∑ _j : Fin M × Fin N, ∑ _j' : Fin M × Fin N, ENNReal.ofReal p₀ := by
        refine add_le_add (add_le_add (Finset.sum_le_sum fun j _ => ?_)
          (Finset.sum_le_sum fun j _ => Finset.sum_le_sum fun z _ => ?_))
          (Finset.sum_le_sum fun j _ => Finset.sum_le_sum fun j' _ => ?_)
        · rw [pi_coord_eq' D j]
          exact (measure_mono_null (fun (x : FreeMonoid α) (hx : x.toList.length < k) =>
            show ¬ k ≤ x.toList.length from not_le.2 hx) (ae_iff.1 hk)).le
        · rw [pi_coord_eq' D j]
          exact ofReal_prefix_le D k hpre _
        · by_cases hj : j' = j
          · simp [hj]
          · exact (measure_mono fun d hd => hd.2).trans (pi_collide_le D k hpre hj)
    _ = ENNReal.ofReal (M * N * (T.card * p₀) + (M * N) ^ 2 * p₀) := by
        have h0 : 0 ≤ p₀ := measureReal_nonneg.trans (hpre 1)
        have e1 : (M : ℝ) * N * (T.card * p₀) = ((M * N * T.card : ℕ) : ℝ) * p₀ := by
          push_cast; ring
        have e2 : ((M : ℝ) * N) ^ 2 * p₀ = ((M * N * (M * N) : ℕ) : ℝ) * p₀ := by
          push_cast; ring
        rw [e1, e2, ENNReal.ofReal_add (by positivity) (by positivity),
          ENNReal.ofReal_mul (Nat.cast_nonneg _), ENNReal.ofReal_mul (Nat.cast_nonneg _),
          ENNReal.ofReal_natCast, ENNReal.ofReal_natCast]
        simp only [Finset.sum_const_zero, zero_add, Finset.sum_const, nsmul_eq_mul,
          Finset.card_univ, Fintype.card_prod, Fintype.card_fin]
        push_cast
        ring

lemma measurableSet_l3Good (k : ℕ) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (K₀ : Finset (FreeMonoid α)) {P : ℕ} :
    MeasurableSet {u : Fin P → unitInterval | L3Good k D K₀ u} := by
  have : IsProbabilityMeasure (avoidLaw D k K₀) := by unfold avoidLaw; infer_instance
  have hmeas : Measurable fun (u : Fin P → unitInterval) (i : Fin P) =>
      drawFrom (avoidLaw D k K₀) (u i) :=
    measurable_pi_lambda _ fun i => (measurable_drawFrom _).comp (measurable_pi_apply i)
  exact hmeas (Set.to_countable {d : Fin P → FreeMonoid α | (∀ i, k ≤ (d i).toList.length)
    ∧ Function.Injective fun i => prefixOf (d i) k}).measurableSet

lemma measurableSet_batchesGood (k : ℕ) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (N M : ℕ) (T : Finset (FreeMonoid α)) :
    MeasurableSet {u | BatchesGood k D N M T u} := by
  have hΦ : Measurable fun u : ℕ → unitInterval × (Fin N → unitInterval) =>
      fun j : Fin M × Fin N => drawFrom D (batchUniforms N M u j) :=
    measurable_pi_lambda _ fun j =>
      (measurable_drawFrom D).comp ((measurable_pi_apply j).comp (measurable_batchUniforms N M))
  have e : {u | BatchesGood k D N M T u} = (fun u : ℕ → unitInterval × (Fin N → unitInterval) =>
      fun j : Fin M × Fin N => drawFrom D (batchUniforms N M u j)) ⁻¹'
      {d | ∀ j : Fin M × Fin N, k ≤ (d j).toList.length
        ∧ (∀ z ∈ T, prefixOf z k ≠ prefixOf (d j) k)
        ∧ ∀ j' : Fin M × Fin N, j'.1 < j.1 → prefixOf (d j') k ≠ prefixOf (d j) k} := by
    ext u
    simp only [Set.mem_setOf_eq, Set.mem_preimage, BatchesGood, batchUniforms]
    constructor
    · intro h j
      obtain ⟨h1, h2, h3⟩ := h j.1 j.1.2 j.2
      exact ⟨h1, h2, fun j' hj' => h3 j'.1 hj' j'.2⟩
    · intro h n hn t
      obtain ⟨h1, h2, h3⟩ := h (⟨n, hn⟩, t)
      exact ⟨h1, h2, fun m hm t' => h3 (⟨m, by omega⟩, t') hm⟩
  rw [e]
  exact hΦ (Set.to_countable _).measurableSet

end OrthoDFA

end
