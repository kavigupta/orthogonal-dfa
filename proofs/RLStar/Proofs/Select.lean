import RLStar.LowLevelClaims.LoopDFA
import RLStar.Proofs.LoopPersist

/-!
# A stage of the run, and fresh draws read elsewhere

`select_le`: where a stage of the run is fixed by the answers at `W`, and a bad event of fresh
draws `b` by the answers at `Z`, the run lands at a stage whose bad event holds, or at one whose
`W` and `Z` meet, with chance at most the bad event's bound plus the meeting's.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ]

section Select

theorem select_le (O : Oracle μ (FreeMonoid α)) {X B Y : Type*} [MeasurableSpace X]
    [MeasurableSpace B] [Countable Y] (ν : Measure X) [IsProbabilityMeasure ν] (β : Measure B)
    [IsProbabilityMeasure β] (stage : Ω → X → Y) (G : Y → Prop)
    (hmeas : ∀ y, G y → MeasurableSet {p : Ω × X | stage p.1 p.2 = y})
    (W : Y → X → Finset (FreeMonoid α))
    (hW : ∀ y x ω ω', G y → stage ω x = y → (∀ w ∈ W y x, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
      stage ω' x = y)
    (Ok : Y → Set X) (hmOk : ∀ y, MeasurableSet (Ok y))
    (Bad : Y → Set (Ω × B)) (hmBad : ∀ y, MeasurableSet (Bad y))
    (Z : Y → B → Finset (FreeMonoid α))
    (hZ : ∀ y b ω ω', (∀ w ∈ Z y b, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) → (ω, b) ∈ Bad y →
      (ω', b) ∈ Bad y)
    (Dj : Y → Set (X × B)) (hmDj : ∀ y, MeasurableSet (Dj y))
    (hDj : ∀ y x b, G y → x ∈ Ok y → (x, b) ∈ Dj y → Disjoint (W y x) (Z y b))
    (δ₁ δ₂ : ℝ≥0∞) (hBad : ∀ y, G y → (μ.prod β) (Bad y) ≤ δ₁)
    (hcoll : ∀ y, G y → ∀ x ∈ Ok y, β {b | (x, b) ∉ Dj y} ≤ δ₂) :
    ((μ.prod ν).prod β) {q | G (stage q.1.1 q.1.2) ∧ q.1.2 ∈ Ok (stage q.1.1 q.1.2)
      ∧ ((q.1.1, q.2) ∈ Bad (stage q.1.1 q.1.2) ∨ (q.1.2, q.2) ∉ Dj (stage q.1.1 q.1.2))}
      ≤ δ₁ + δ₂ := by
  classical
  set U : Y → Set (Ω × X) := fun y =>
    if G y then {p | stage p.1 p.2 = y} ∩ Prod.snd ⁻¹' Ok y else ∅ with hUd
  have hmU : ∀ y, MeasurableSet (U y) := fun y => by
    simp only [hUd]
    split_ifs with h
    · exact (hmeas y h).inter (measurable_snd (hmOk y))
    · exact MeasurableSet.empty
  have hmemU : ∀ y p, p ∈ U y ↔ G y ∧ stage p.1 p.2 = y ∧ p.2 ∈ Ok y := fun y p => by
    simp only [hUd]
    split_ifs with h
    · simp [h]
    · simp [h]
  set T₁ : Y → Set ((Ω × X) × B) := fun y =>
    Prod.fst ⁻¹' U y ∩ {q | (q.1.1, q.2) ∈ Bad y} ∩ {q | (q.1.2, q.2) ∈ Dj y} with hT₁d
  set T₂ : Y → Set ((Ω × X) × B) := fun y =>
    Prod.fst ⁻¹' U y ∩ {q | (q.1.2, q.2) ∉ Dj y} with hT₂d
  have hm1 : ∀ y, MeasurableSet (T₁ y) := fun y =>
    ((measurable_fst (hmU y)).inter
      (((measurable_fst.comp measurable_fst).prodMk measurable_snd) (hmBad y))).inter
      (((measurable_snd.comp measurable_fst).prodMk measurable_snd) (hmDj y))
  have hm2 : ∀ y, MeasurableSet (T₂ y) := fun y =>
    (measurable_fst (hmU y)).inter
      (((measurable_snd.comp measurable_fst).prodMk measurable_snd) (hmDj y)).compl
  have hsub : {q : (Ω × X) × B | G (stage q.1.1 q.1.2) ∧ q.1.2 ∈ Ok (stage q.1.1 q.1.2)
      ∧ ((q.1.1, q.2) ∈ Bad (stage q.1.1 q.1.2) ∨ (q.1.2, q.2) ∉ Dj (stage q.1.1 q.1.2))}
      ⊆ (⋃ y, T₁ y) ∪ ⋃ y, T₂ y := by
    rintro q ⟨hG, hOk, h⟩
    have hU : q.1 ∈ U (stage q.1.1 q.1.2) := (hmemU _ _).2 ⟨hG, rfl, hOk⟩
    by_cases hd : (q.1.2, q.2) ∈ Dj (stage q.1.1 q.1.2)
    · rcases h with h | h
      · exact Or.inl (Set.mem_iUnion.2 ⟨_, ⟨hU, h⟩, hd⟩)
      · exact absurd hd h
    · exact Or.inr (Set.mem_iUnion.2 ⟨_, hU, hd⟩)
  -- the run's stage, fixed by the answers at `W`, and the bad event, by those at `Z`
  have hb1 : ∀ y, ((μ.prod ν).prod β) (T₁ y) ≤ δ₁ * (μ.prod ν) (U y) := by
    intro y
    by_cases hG : ¬ G y
    · have : T₁ y = ∅ := by
        ext q
        simp only [hT₁d, Set.mem_inter_iff, Set.mem_preimage, hmemU, Set.mem_empty_iff_false,
          iff_false]
        tauto
      rw [this, measure_empty]
      exact zero_le
    push Not at hG
    rw [Measure.prod_apply (hm1 y), lintegral_prod_symm _
      (measurable_measure_prodMk_left (hm1 y)).aemeasurable,
      Measure.prod_apply_symm (hmU y), ← lintegral_const_mul _
        (measurable_measure_prodMk_right (hmU y))]
    refine lintegral_mono fun x => ?_
    have hsec : MeasurableSet ((fun z : Ω × B => ((z.1, x), z.2)) ⁻¹' T₁ y) :=
      (measurable_fst.prodMk measurable_const).prodMk measurable_snd (hm1 y)
    have e1 : ∫⁻ ω, β (Prod.mk (ω, x) ⁻¹' T₁ y) ∂μ
        = ∫⁻ b, μ {ω | ((ω, x), b) ∈ T₁ y} ∂β := by
      exact (Measure.prod_apply (μ := μ) (ν := β) hsec).symm.trans
        (Measure.prod_apply_symm (μ := μ) (ν := β) hsec)
    rw [e1]
    have hb : ∀ b, μ {ω | ((ω, x), b) ∈ T₁ y}
        ≤ μ {ω | (ω, x) ∈ U y} * μ {ω | (ω, b) ∈ Bad y} := by
      intro b
      by_cases hc : x ∈ Ok y ∧ (x, b) ∈ Dj y
      · have hset : {ω | ((ω, x), b) ∈ T₁ y}
            = {ω | stage ω x = y} ∩ {ω | (ω, b) ∈ Bad y} := by
          ext ω
          simp only [hT₁d, Set.mem_inter_iff, Set.mem_preimage, Set.mem_ofPred_eq, hmemU]
          tauto
        have hset' : {ω | (ω, x) ∈ U y} = {ω | stage ω x = y} := by
          ext ω
          simp only [Set.mem_ofPred_eq, hmemU]
          tauto
        rw [hset, hset']
        exact (measure_inter_of_reads O (hDj y x b hG hc.1 hc.2) {ω | stage ω x = y}
          {ω | (ω, b) ∈ Bad y} (fun ω ω' h hω => hW y x ω ω' hG hω h)
          (fun ω ω' h hω => hZ y b ω ω' h hω)).le
      · have : {ω | ((ω, x), b) ∈ T₁ y} = ∅ := by
          ext ω
          simp only [hT₁d, Set.mem_inter_iff, Set.mem_preimage, Set.mem_ofPred_eq, hmemU,
            Set.mem_empty_iff_false, iff_false]
          tauto
        rw [this, measure_empty]
        exact zero_le
    calc ∫⁻ b, μ {ω | ((ω, x), b) ∈ T₁ y} ∂β
        ≤ ∫⁻ b, μ {ω | (ω, x) ∈ U y} * μ {ω | (ω, b) ∈ Bad y} ∂β := lintegral_mono hb
      _ = μ {ω | (ω, x) ∈ U y} * ∫⁻ b, μ {ω | (ω, b) ∈ Bad y} ∂β :=
          lintegral_const_mul _ (measurable_measure_prodMk_right (hmBad y))
      _ = μ {ω | (ω, x) ∈ U y} * (μ.prod β) (Bad y) := by
          rw [Measure.prod_apply_symm (hmBad y)]
          rfl
      _ ≤ μ {ω | (ω, x) ∈ U y} * δ₁ := by gcongr; exact hBad y hG
      _ = δ₁ * μ ((fun ω => (ω, x)) ⁻¹' U y) := mul_comm _ _
  have hb2 : ∀ y, ((μ.prod ν).prod β) (T₂ y) ≤ δ₂ * (μ.prod ν) (U y) := by
    intro y
    rw [Measure.prod_apply (hm2 y), ← lintegral_indicator_const (hmU y)]
    refine lintegral_mono fun p => ?_
    by_cases hp : p ∈ U y
    · rw [Set.indicator_of_mem hp]
      obtain ⟨hG, -, hOk⟩ := (hmemU y p).1 hp
      refine (measure_mono fun b hb => ?_).trans (hcoll y hG p.2 hOk)
      exact hb.2
    · rw [Set.indicator_of_notMem hp]
      have : Prod.mk p ⁻¹' T₂ y = ∅ := by
        ext b
        simp only [hT₂d, Set.mem_preimage, Set.mem_inter_iff, Set.mem_empty_iff_false,
          iff_false]
        exact fun h => hp h.1
      rw [this, measure_empty]
  have hdisj : Pairwise (Function.onFun Disjoint U) := by
    intro y y' hyy'
    refine Set.disjoint_left.2 fun p hp hp' => hyy' ?_
    exact ((hmemU y p).1 hp).2.1.symm.trans ((hmemU y' p).1 hp').2.1
  have hsumU : ∑' y, (μ.prod ν) (U y) ≤ 1 := by
    rw [← measure_iUnion hdisj hmU]
    exact prob_le_one
  calc ((μ.prod ν).prod β) _ ≤ ((μ.prod ν).prod β) ((⋃ y, T₁ y) ∪ ⋃ y, T₂ y) := measure_mono hsub
    _ ≤ ((μ.prod ν).prod β) (⋃ y, T₁ y) + ((μ.prod ν).prod β) (⋃ y, T₂ y) := measure_union_le _ _
    _ ≤ ∑' y, ((μ.prod ν).prod β) (T₁ y) + ∑' y, ((μ.prod ν).prod β) (T₂ y) :=
        add_le_add (measure_iUnion_le _) (measure_iUnion_le _)
    _ ≤ ∑' y, δ₁ * (μ.prod ν) (U y) + ∑' y, δ₂ * (μ.prod ν) (U y) :=
        add_le_add (ENNReal.tsum_le_tsum hb1) (ENNReal.tsum_le_tsum hb2)
    _ = δ₁ * ∑' y, (μ.prod ν) (U y) + δ₂ * ∑' y, (μ.prod ν) (U y) := by
        rw [ENNReal.tsum_mul_left, ENNReal.tsum_mul_left]
    _ ≤ δ₁ * 1 + δ₂ * 1 := by gcongr
    _ = δ₁ + δ₂ := by rw [mul_one, mul_one]

/-- `select_le` over stages indexed by `r`, at most one of which lands where its `G` holds. -/
theorem select_family_le (O : Oracle μ (FreeMonoid α)) {X B R : Type*} {Y : R → Type*}
    [MeasurableSpace X] [MeasurableSpace B] [Countable R] [∀ r, Countable (Y r)]
    (ν : Measure X) [IsProbabilityMeasure ν] (β : Measure B) [IsProbabilityMeasure β]
    (stage : ∀ r, Ω → X → Y r) (G : ∀ r, Y r → Prop)
    (huniq : ∀ r r' ω x, G r (stage r ω x) → G r' (stage r' ω x) → r = r')
    (hmeas : ∀ r y, G r y → MeasurableSet {p : Ω × X | stage r p.1 p.2 = y})
    (W : ∀ r, Y r → X → Finset (FreeMonoid α))
    (hW : ∀ r y x ω ω', G r y → stage r ω x = y →
      (∀ w ∈ W r y x, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) → stage r ω' x = y)
    (Ok : ∀ r, Y r → Set X) (hmOk : ∀ r y, MeasurableSet (Ok r y))
    (Bad : ∀ r, Y r → Set (Ω × B)) (hmBad : ∀ r y, MeasurableSet (Bad r y))
    (Z : ∀ r, Y r → B → Finset (FreeMonoid α))
    (hZ : ∀ r y b ω ω', (∀ w ∈ Z r y b, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) → (ω, b) ∈ Bad r y →
      (ω', b) ∈ Bad r y)
    (Dj : ∀ r, Y r → Set (X × B)) (hmDj : ∀ r y, MeasurableSet (Dj r y))
    (hDj : ∀ r y x b, G r y → x ∈ Ok r y → (x, b) ∈ Dj r y → Disjoint (W r y x) (Z r y b))
    (δ₁ δ₂ : ℝ≥0∞) (hBad : ∀ r y, G r y → (μ.prod β) (Bad r y) ≤ δ₁)
    (hcoll : ∀ r y, G r y → ∀ x ∈ Ok r y, β {b | (x, b) ∉ Dj r y} ≤ δ₂) :
    ((μ.prod ν).prod β) (⋃ r, {q | G r (stage r q.1.1 q.1.2)
      ∧ q.1.2 ∈ Ok r (stage r q.1.1 q.1.2) ∧ ((q.1.1, q.2) ∈ Bad r (stage r q.1.1 q.1.2)
        ∨ (q.1.2, q.2) ∉ Dj r (stage r q.1.1 q.1.2))})
      ≤ δ₁ + δ₂ := by
  classical
  obtain ⟨st, hst, hsome⟩ : ∃ st : Ω → X → Option (Σ r, Y r),
      (∀ r ω x, G r (stage r ω x) → st ω x = some ⟨r, stage r ω x⟩)
        ∧ ∀ r y ω x, st ω x = some ⟨r, y⟩ → stage r ω x = y := by
    refine ⟨fun ω x => if h : ∃ r, G r (stage r ω x) then some ⟨h.choose, stage h.choose ω x⟩
      else none, fun r ω x hG => ?_, fun r y ω x he => ?_⟩
    · have h : ∃ r, G r (stage r ω x) := ⟨r, hG⟩
      have hc : h.choose = r := huniq _ _ ω x h.choose_spec hG
      simp only [dif_pos h]
      rw [hc]
    · simp only at he
      split_ifs at he with h
      · generalize h.choose = c at he
        obtain ⟨rfl, he⟩ := Sigma.mk.inj_iff.1 (Option.some.inj he)
        exact eq_of_heq he
  let G' : Option (Σ r, Y r) → Prop
    | some ⟨r, y⟩ => G r y
    | none => False
  have hmain := select_le O ν β st G'
    (fun y' hy => by
      rcases y' with _ | ⟨r, y⟩
      · exact hy.elim
      have hset : {p : Ω × X | st p.1 p.2 = some ⟨r, y⟩} = {p | stage r p.1 p.2 = y} := by
        ext p
        refine ⟨hsome r y p.1 p.2, fun he => ?_⟩
        have hG : G r (stage r p.1 p.2) := he ▸ hy
        change st p.1 p.2 = _
        rw [hst r p.1 p.2 hG, he]
      rw [hset]
      exact hmeas r y hy)
    (fun | some ⟨r, y⟩ => W r y | none => fun _ => ∅)
    (fun y' x ω ω' hy hs h => by
      rcases y' with _ | ⟨r, y⟩
      · exact hy.elim
      have h' := hW r y x ω ω' hy (hsome r y ω x hs) h
      have hG : G r (stage r ω' x) := h' ▸ hy
      rw [hst r ω' x hG, h'])
    (fun | some ⟨r, y⟩ => Ok r y | none => Set.univ)
    (fun | some ⟨r, y⟩ => hmOk r y | none => MeasurableSet.univ)
    (fun | some ⟨r, y⟩ => Bad r y | none => ∅)
    (fun | some ⟨r, y⟩ => hmBad r y | none => MeasurableSet.empty)
    (fun | some ⟨r, y⟩ => Z r y | none => fun _ => ∅)
    (fun | some ⟨r, y⟩ => hZ r y | none => fun _ _ _ _ h => h)
    (fun | some ⟨r, y⟩ => Dj r y | none => Set.univ)
    (fun | some ⟨r, y⟩ => hmDj r y | none => MeasurableSet.univ)
    (fun | some ⟨r, y⟩ => hDj r y | none => fun _ _ h => h.elim)
    δ₁ δ₂
    (fun | some ⟨r, y⟩ => hBad r y | none => fun h => h.elim)
    (fun | some ⟨r, y⟩ => hcoll r y | none => fun h => h.elim)
  refine (measure_mono fun q hq => ?_).trans hmain
  obtain ⟨r, hq⟩ := Set.mem_iUnion.1 hq
  change G' (st q.1.1 q.1.2) ∧ _
  rw [hst r q.1.1 q.1.2 hq.1]
  exact hq

end Select

end OrthoDFA

end
