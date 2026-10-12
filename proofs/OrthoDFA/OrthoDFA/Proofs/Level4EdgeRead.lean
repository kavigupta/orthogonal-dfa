import OrthoDFA.Proofs.Level4Firsts
import OrthoDFA.Proofs.Level4EdgeNoise

/-!
# The noise event's edge field, per read

A probe's first reads are fresh, so its undecided first reads at an edge at good read-states
average at most `1.5θ` of its first reads there (`edge_first_mean`). Over the probes' length-`k`
prefixes the field's tail is then `block_tail`'s, at twice that rate.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

attribute [local instance] ARU.fintype

section Mean

variable {σ : Type*} (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] {read : FreeMonoid α → Ω → ARU}

open scoped Classical in
theorem undecAt_eq (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : EdgeTable α)
    (d : List Bool × α) (x : FreeMonoid α) :
    G.undecAt rd k T edges d G.Good x = (((probeQuery T edges k x).firsts rd ∅).filter fun q =>
      q.1.2 = some d ∧ rd q.2 = .undecided ∧ G.Good (G.M.eval q.2.toList)).length := by
  unfold ReadModel.undecAt edgeHarvestBy firstsBy
  rw [ofCut_rd, List.filter_map, List.length_map, List.filter_filter]
  congr 1
  refine List.filter_congr fun q _ => ?_
  have hc : ((rd q.2).cut = none) ↔ rd q.2 = .undecided := by
    cases rd q.2 <;> simp [ARU.cut]
  by_cases h1 : q.1.2 = some d <;> by_cases h2 : rd q.2 = .undecided
  · simp [h1, h2, ARU.cut]
  · have h2' : ¬ (rd q.2).cut = none := fun h => h2 (hc.1 h)
    simp [h1, h2, h2']
  · simp [h1]
  · simp [h1]

theorem edgeReadsBy_eq (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : EdgeTable α)
    (d : List Bool × α) (x : FreeMonoid α) :
    edgeReadsBy (fun z => (rd z).cut) T edges k x d
      = (((probeQuery T edges k x).firsts rd ∅).filter fun q => q.1.2 = some d).length := by
  unfold edgeReadsBy firstsBy
  rw [ofCut_rd]

open scoped Classical in
/-- The per-read edge field in mean at one probe, over real expectations. -/
theorem undecAt_mean_real (hmeas : ∀ w, Measurable (read w)) (hind : iIndepFun read μ)
    (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.readDist (G.M.eval w.toList) r) (hθ : 0 ≤ G.θ)
    (T : DTree α) (edges : EdgeTable α) (k : ℕ) (d : List Bool × α) (x : FreeMonoid α) :
    ∫ ω, (G.undecAt (read · ω) k T edges d G.Good x : ℝ) ∂μ
      ≤ 3 / 2 * G.θ * ∫ ω, (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ) ∂μ := by
  have h := edge_first_mean G hmeas hind hlaw G.Good (3 / 2 * G.θ) (fun r h => h) d
    (probeQuery T edges k x)
  have hmU : Measurable fun ω => (G.undecAt (read · ω) k T edges d G.Good x : ℝ) :=
    measurable_of_strings hmeas (prefStrings T x)
      (fun rd => (G.undecAt rd k T edges d G.Good x : ℝ)) fun rd rd' h => by
        simp only [ReadModel.undecAt, edgeHarvestBy_congr_pref (edges := edges) (k := k) (T := T)
          (x := x) (cut := fun z => (rd z).cut) (cut' := fun z => (rd' z).cut) fun z hz => by
            rw [h z hz]]
  have hmN : Measurable fun ω => (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ) :=
    measurable_of_strings hmeas (prefStrings T x)
      (fun rd => (edgeReadsBy (fun z => (rd z).cut) T edges k x d : ℝ)) fun rd rd' h => by
        simp only [edgeReadsBy_congr_pref (edges := edges) (k := k) (T := T) (x := x)
          (cut := fun z => (rd z).cut) (cut' := fun z => (rd' z).cut) fun z hz => by
            rw [h z hz]]
  have hbN : ∀ ω, (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ)
      ≤ (x.toList.length + 1) * T.midfixes.card := fun ω => by
    have := edgeReadsBy_le (fun z => (read z ω).cut) T edges k x d
    exact_mod_cast this
  have hbU : ∀ ω, (G.undecAt (read · ω) k T edges d G.Good x : ℝ)
      ≤ (x.toList.length + 1) * T.midfixes.card := fun ω => by
    have h1 : G.undecAt (read · ω) k T edges d G.Good x
        ≤ (edgeHarvestBy (fun z => (read z ω).cut) T edges k x d).length :=
      List.length_filter_le _ _
    have h2 := edgeHarvestBy_le_edgeReadsBy (fun z => (read z ω).cut) T edges k x d
    have h3 := edgeReadsBy_le (fun z => (read z ω).cut) T edges k x d
    have := h1.trans (h2.trans h3)
    exact_mod_cast this
  have hiU : Integrable (fun ω => (G.undecAt (read · ω) k T edges d G.Good x : ℝ)) μ :=
    Integrable.of_bound hmU.aestronglyMeasurable _ (Filter.Eventually.of_forall fun ω => by
      rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]; exact hbU ω)
  have hiN : Integrable (fun ω => (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ)) μ :=
    Integrable.of_bound hmN.aestronglyMeasurable _ (Filter.Eventually.of_forall fun ω => by
      rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]; exact hbN ω)
  have e1 : ∫⁻ ω, (((((probeQuery T edges k x).firsts (fun y => read y ω) ∅).filter fun q =>
      q.1.2 = some d ∧ read q.2 ω = .undecided ∧ G.Good (G.M.eval q.2.toList)).length : ℕ)
        : ENNReal) ∂μ
      = ENNReal.ofReal (∫ ω, (G.undecAt (read · ω) k T edges d G.Good x : ℝ) ∂μ) := by
    rw [ofReal_integral_eq_lintegral_ofReal hiU
      (Filter.Eventually.of_forall fun ω => Nat.cast_nonneg _)]
    congr 1; funext ω
    rw [undecAt_eq G (read · ω), ENNReal.ofReal_natCast]
  have e2 : ∫⁻ ω, (((((probeQuery T edges k x).firsts (fun y => read y ω) ∅).filter fun q =>
      q.1.2 = some d).length : ℕ) : ENNReal) ∂μ
      = ENNReal.ofReal (∫ ω, (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ) ∂μ) := by
    rw [ofReal_integral_eq_lintegral_ofReal hiN
      (Filter.Eventually.of_forall fun ω => Nat.cast_nonneg _)]
    congr 1; funext ω
    rw [edgeReadsBy_eq (read · ω), ENNReal.ofReal_natCast]
  rw [e1, e2, ← ENNReal.ofReal_mul (by positivity)] at h
  have hI : 0 ≤ ∫ ω, (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ) ∂μ :=
    integral_nonneg fun ω => Nat.cast_nonneg _
  exact (ENNReal.ofReal_le_ofReal_iff (by positivity)).1 h

end Mean

section One

variable {σ : Type*} (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)

/-- At one tree, edge map and edge, the good read-states' undecided first reads exceed `3θ` of the
first reads by `η` with chance at most this. -/
theorem goodEdge_one (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.readDist (G.M.eval z.toList) r) (hθ : 0 ≤ G.θ)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {k L : ℕ} {p₀ η : ℝ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hp₀ : 0 < p₀)
    (hpmax : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (hη : 0 ≤ η) (T : DTree α)
    (edges : EdgeTable α) (d : List Bool × α) :
    μ {ω | 2 * (3 / 2 * G.θ) * ∫ x, (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ) ∂D + η
        < ∫ x, (G.undecAt (read · ω) k T edges d G.Good x : ℝ) ∂D}
      ≤ ENNReal.ofReal (Real.exp (-(η / (2 * p₀ * (L + 1)
          * (1 + 4 * (3 / 2 * G.θ) * T.midfixes.card))))) := by
  set uf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    if k ≤ x.toList.length then (G.undecAt rd k T edges d G.Good x : ℝ) else 0
  set tf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    if k ≤ x.toList.length then (edgeReadsBy (fun z => (rd z).cut) T edges k x d : ℝ) else 0
  have hagr : ∀ x, k ≤ x.toList.length → ∀ rd rd' : FreeMonoid α → ARU,
      (∀ z ∈ probeStrings T k x, rd z = rd' z) →
      ∀ z ∈ probeStrings T k x, (rd z).cut = (rd' z).cut := fun x _ rd rd' h z hz => by
    rw [h z hz]
  have huf : ∀ x rd rd', (∀ z ∈ probeStrings T k x, rd z = rd' z) → uf rd x = uf rd' x := by
    intro x rd rd' h
    by_cases hkx : k ≤ x.toList.length
    · simp only [uf, if_pos hkx, ReadModel.undecAt,
        edgeHarvestBy_congr_strings (edges := edges) hkx (hagr x hkx rd rd' h)]
    · simp only [uf, if_neg hkx]
  have htf : ∀ x rd rd', (∀ z ∈ probeStrings T k x, rd z = rd' z) → tf rd x = tf rd' x := by
    intro x rd rd' h
    by_cases hkx : k ≤ x.toList.length
    · simp only [tf, if_pos hkx, edgeReadsBy_congr_strings (edges := edges) hkx
        (hagr x hkx rd rd' h)]
    · simp only [tf, if_neg hkx]
  have hub : ∀ rd, ∀ x ∈ lenRange 0 L, 0 ≤ uf rd x ∧ uf rd x ≤ L + 1 := by
    intro rd x hx
    obtain ⟨-, hxL⟩ := mem_lenRange.1 hx
    by_cases hkx : k ≤ x.toList.length
    swap
    · simp only [uf, if_neg hkx]; constructor <;> positivity
    simp only [uf, if_pos hkx]
    refine ⟨Nat.cast_nonneg _, ?_⟩
    have h1 : G.undecAt rd k T edges d G.Good x
        ≤ (edgeHarvestBy (fun z => (rd z).cut) T edges k x d).length := List.length_filter_le _ _
    have h2 := edgeHarvestBy_length_le (fun z => (rd z).cut) T edges k x d
    have : G.undecAt rd k T edges d G.Good x ≤ L + 1 := by omega
    exact_mod_cast this
  have htb : ∀ rd, ∀ x ∈ lenRange 0 L, 0 ≤ tf rd x
      ∧ tf rd x ≤ (L + 1) * (T.midfixes.card : ℝ) := by
    intro rd x hx
    obtain ⟨-, hxL⟩ := mem_lenRange.1 hx
    by_cases hkx : k ≤ x.toList.length
    swap
    · simp only [tf, if_neg hkx]; constructor <;> positivity
    simp only [tf, if_pos hkx]
    refine ⟨Nat.cast_nonneg _, ?_⟩
    have h1 := edgeReadsBy_le (fun z => (rd z).cut) T edges k x d
    have : edgeReadsBy (fun z => (rd z).cut) T edges k x d ≤ (L + 1) * T.midfixes.card :=
      h1.trans (Nat.mul_le_mul_right _ (by omega))
    exact_mod_cast this
  have hmean : ∀ x ∈ lenRange 0 L,
      ∫ ω, uf (read · ω) x ∂μ ≤ 3 / 2 * G.θ * ∫ ω, tf (read · ω) x ∂μ := by
    intro x hx
    by_cases hkx : k ≤ x.toList.length
    swap
    · simp only [uf, tf, if_neg hkx, integral_zero, mul_zero, le_refl]
    simp only [uf, tf, if_pos hkx]
    exact undecAt_mean_real G hmeas hind hlaw hθ T edges k d x
  have hTB := block_tail (μ := μ) (read := read) hmeas hind D (L := L) hp₀ hpmax hη
    (by positivity : (0 : ℝ) ≤ 3 / 2 * G.θ) (by positivity : (0 : ℝ) < L + 1)
    (Nat.cast_nonneg T.midfixes.card) (probeStrings T k ·)
    (fun x z hz => probeStrings_prefix hz) uf tf huf htf hub htb hmean
  have hM : ∀ (f : FreeMonoid α → ℝ), (∀ x, x.toList.length ≤ L → 0 ≤ f x ∧
      f x ≤ (L + 1) * (T.midfixes.card + 1)) →
      ∀ x, x.toList.length ≤ L → |f x| ≤ (L + 1) * (T.midfixes.card + 1) := by
    intro f hf x hx
    rw [abs_of_nonneg (hf x hx).1]; exact (hf x hx).2
  have hUs : ∀ ω, ∫ x, (G.undecAt (read · ω) k T edges d G.Good x : ℝ) ∂D
      = ∑ x ∈ lenRange 0 L, D.real {x} * uf (read · ω) x := by
    intro ω
    rw [integral_eq_sum_len D hlen _ ((L + 1) * (T.midfixes.card + 1)) (hM _ fun x hx => by
      refine ⟨Nat.cast_nonneg _, ?_⟩
      have h1 : G.undecAt (read · ω) k T edges d G.Good x
          ≤ (edgeHarvestBy (fun z => (read z ω).cut) T edges k x d).length :=
        List.length_filter_le _ _
      have h2 := edgeHarvestBy_length_le (fun z => (read z ω).cut) T edges k x d
      have : (G.undecAt (read · ω) k T edges d G.Good x : ℝ) ≤ L + 1 := by
        have : G.undecAt (read · ω) k T edges d G.Good x ≤ L + 1 := by omega
        exact_mod_cast this
      nlinarith [(Nat.cast_nonneg T.midfixes.card : (0 : ℝ) ≤ _)])]
    refine Finset.sum_congr rfl fun x _ => ?_
    by_cases hkx : k ≤ x.toList.length
    · simp only [uf, if_pos hkx]
    · have hs' : x.toList.length < k := by omega
      simp [uf, hkx, ReadModel.undecAt, edgeHarvestBy_short _ T edges hs']
  have hNs : ∀ ω, ∫ x, (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ) ∂D
      = ∑ x ∈ lenRange 0 L, D.real {x} * tf (read · ω) x := by
    intro ω
    rw [integral_eq_sum_len D hlen _ ((L + 1) * (T.midfixes.card + 1)) (hM _ fun x hx => by
      refine ⟨Nat.cast_nonneg _, ?_⟩
      have h1 := edgeReadsBy_le (fun z => (read z ω).cut) T edges k x d
      have : (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ)
          ≤ (L + 1) * T.midfixes.card := by
        have : edgeReadsBy (fun z => (read z ω).cut) T edges k x d ≤ (L + 1) * T.midfixes.card :=
          h1.trans (Nat.mul_le_mul_right _ (by omega))
        exact_mod_cast this
      nlinarith)]
    refine Finset.sum_congr rfl fun x _ => ?_
    by_cases hkx : k ≤ x.toList.length
    · simp only [tf, if_pos hkx]
    · simp [tf, hkx, edgeReadsBy_short _ T edges (show x.toList.length < k by omega)]
  calc μ _ ≤ μ {ω | 2 * (3 / 2 * G.θ) * ∑ x ∈ lenRange 0 L, D.real {x} * tf (read · ω) x + η
        < ∑ x ∈ lenRange 0 L, D.real {x} * uf (read · ω) x} := by
        refine measure_mono fun ω hω => ?_
        simp only [Set.mem_setOf_eq] at hω ⊢
        rw [← hUs, ← hNs]
        exact hω
    _ ≤ _ := hTB
    _ = _ := by congr 4

end One

section Union

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)

/-- Over the class and its edge maps, the good read-states' undecided first reads at an edge exceed
`θg ≥ 3θ` of its first reads by `η` with chance at most the count of trees, edge maps and edges
times `exp(−η / (2 p₀ (L + 1)(1 + 6θ (|Q| + S + 1))))`. -/
theorem goodEdge_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.readDist (G.M.eval z.toList) r) (hθ : 0 ≤ G.θ)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k L S : ℕ) {p₀ η θg : ℝ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hp₀ : 0 < p₀) (hpmax : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (hη : 0 ≤ η)
    (hθg : 2 * (3 / 2 * G.θ) ≤ θg) :
    μ {ω | ¬ ∀ T, G.InClass S T → ∀ edges, EdgesInto T edges → ∀ d,
        ∫ x, (G.undecAt (read · ω) k T edges d G.Good x : ℝ) ∂D
          ≤ θg * ∫ x, (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ) ∂D + η}
      ≤ ENNReal.ofReal ((classSet (Fintype.card σ + S) : Finset (DTree α)).card
        * ((Fintype.card σ + S + 3) ^ ((Fintype.card σ + S + 2) * Fintype.card α)
          * ((Fintype.card σ + S + 2) * Fintype.card α))
        * Real.exp (-(η / (2 * p₀ * (L + 1)
          * (1 + 4 * (3 / 2 * G.θ) * (Fintype.card σ + S + 1)))))) := by
  set n := Fintype.card σ + S
  set ε : ℝ := Real.exp (-(η / (2 * p₀ * (L + 1) * (1 + 4 * (3 / 2 * G.θ) * (n + 1)))))
  set B : DTree α → EdgeTable α → List Bool × α → Set Ω := fun T e d => {ω |
    2 * (3 / 2 * G.θ) * ∫ x, (edgeReadsBy (fun z => (read z ω).cut) T e k x d : ℝ) ∂D + η
      < ∫ x, (G.undecAt (read · ω) k T e d G.Good x : ℝ) ∂D}
  have hsub : {ω | ¬ ∀ T, G.InClass S T → ∀ edges, EdgesInto T edges → ∀ d,
      ∫ x, (G.undecAt (read · ω) k T edges d G.Good x : ℝ) ∂D
        ≤ θg * ∫ x, (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ) ∂D + η}
      ⊆ ⋃ T ∈ (classSet n : Finset (DTree α)), ⋃ e ∈ edgeMaps T,
          ⋃ d ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α), B T e d := by
    intro ω hω
    simp only [Set.mem_setOf_eq, not_forall, not_le] at hω
    obtain ⟨T, hT, edges, he, d, hlt⟩ := hω
    obtain ⟨e', he'm, hagr, -⟩ := exists_edgeMaps he
    have hU : ∀ x, G.undecAt (read · ω) k T edges d G.Good x
        = G.undecAt (read · ω) k T e' d G.Good x := by
      intro x
      simp only [ReadModel.undecAt]
      rw [edgeHarvestBy_econgr (k := k) (x := x) he hagr (fun z => (read z ω).cut)]
    have hN : ∀ x, edgeReadsBy (fun z => (read z ω).cut) T edges k x d
        = edgeReadsBy (fun z => (read z ω).cut) T e' k x d := by
      intro x
      rw [edgeReadsBy_econgr (k := k) (x := x) he hagr (fun z => (read z ω).cut)]
    have hd : d.1 ∈ T.leaves := by
      by_contra hd
      have h0 : ∀ x, G.undecAt (read · ω) k T edges d G.Good x = 0 := by
        intro x
        simp only [ReadModel.undecAt]
        have : edgeHarvestBy (fun z => (read z ω).cut) T edges k x d = [] :=
          by_contra fun hne => hd (edgeHarvestBy_mem _ T k x he hne)
        simp [this]
      simp only [h0, Nat.cast_zero, integral_zero] at hlt
      have : 0 ≤ ∫ x, (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ) ∂D :=
        integral_nonneg fun x => Nat.cast_nonneg _
      have : 0 ≤ θg := le_trans (by positivity) hθg
      nlinarith
    refine Set.mem_biUnion (inClass_mem G hT) (Set.mem_biUnion he'm (Set.mem_biUnion
      (Finset.mem_product.2 ⟨List.mem_toFinset.2 hd, Finset.mem_univ _⟩) ?_))
    simp only [B, Set.mem_setOf_eq]
    simp only [hU, hN] at hlt
    have : 0 ≤ ∫ x, (edgeReadsBy (fun z => (read z ω).cut) T e' k x d : ℝ) ∂D :=
      integral_nonneg fun x => Nat.cast_nonneg _
    nlinarith [mul_le_mul_of_nonneg_right hθg this]
  have hone : ∀ T ∈ (classSet n : Finset (DTree α)), ∀ e d, μ (B T e d) ≤ ENNReal.ofReal ε := by
    intro T hT e d
    have hm : (T.midfixes.card : ℝ) ≤ n + 1 := by
      have := midfixes_card T
      have := classSet_paths n T hT
      exact_mod_cast (by omega : T.midfixes.card ≤ n + 1)
    refine (goodEdge_one G read hmeas hind hlaw hθ D hlen hp₀ hpmax hη T e d).trans
      (ENNReal.ofReal_le_ofReal ?_)
    simp only [ε]
    gcongr
  calc μ _ ≤ μ (⋃ T ∈ (classSet n : Finset (DTree α)), ⋃ e ∈ edgeMaps T,
        ⋃ d ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α), B T e d) := measure_mono hsub
    _ ≤ ∑ T ∈ (classSet n : Finset (DTree α)), ∑ e ∈ edgeMaps T,
          ∑ d ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α), μ (B T e d) := by
        refine (measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun T _ => ?_)
        refine (measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun e _ => ?_)
        exact measure_biUnion_finset_le _ _
    _ ≤ ∑ T ∈ (classSet n : Finset (DTree α)),
          (((n + 3) ^ ((n + 2) * Fintype.card α) * ((n + 2) * Fintype.card α) : ℕ) : ENNReal)
            * ENNReal.ofReal ε := by
        refine Finset.sum_le_sum fun T hT => ?_
        have hp := classSet_paths n T hT
        calc ∑ e ∈ edgeMaps T, ∑ d ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α), μ (B T e d)
            ≤ ∑ e ∈ edgeMaps T, ∑ d ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α),
                ENNReal.ofReal ε := Finset.sum_le_sum fun e _ =>
              Finset.sum_le_sum fun d _ => hone T hT e d
          _ = ((edgeMaps T).card * (T.leaves.toFinset ×ˢ (Finset.univ : Finset α)).card : ℕ)
                * ENNReal.ofReal ε := by
              simp only [Finset.sum_const, nsmul_eq_mul]; push_cast; ring
          _ ≤ _ := by
              refine mul_le_mul' ?_ le_rfl
              have h1 : (edgeMaps T).card ≤ (n + 3) ^ ((n + 2) * Fintype.card α) :=
                calc (edgeMaps T).card
                    ≤ (T.leaves.length + 1) ^ (T.leaves.length * Fintype.card α) := edgeMaps_card T
                  _ ≤ (n + 3) ^ (T.leaves.length * Fintype.card α) :=
                    Nat.pow_le_pow_left (by omega) _
                  _ ≤ (n + 3) ^ ((n + 2) * Fintype.card α) :=
                    Nat.pow_le_pow_right (by omega) (Nat.mul_le_mul_right _ hp)
              have h2 : (T.leaves.toFinset ×ˢ (Finset.univ : Finset α)).card
                  ≤ (n + 2) * Fintype.card α := by
                simp only [Finset.card_product, Finset.card_univ]
                exact Nat.mul_le_mul_right _ ((List.toFinset_card_le _).trans hp)
              exact_mod_cast Nat.mul_le_mul h1 h2
    _ = _ := by
        rw [Finset.sum_const, nsmul_eq_mul, ← ENNReal.ofReal_natCast,
          ← ENNReal.ofReal_natCast ((n + 3) ^ ((n + 2) * Fintype.card α)
            * ((n + 2) * Fintype.card α)), ← ENNReal.ofReal_mul (by positivity),
          ← ENNReal.ofReal_mul (by positivity)]
        congr 1
        simp only [n, ε]
        push_cast
        ring

end Union

end OrthoDFA
