import OrthoDFA.Proofs.GateFlip

/-! # An event of fresh bits, at a choice the other bits make -/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- A choice `sel` decided by the oracle's bits at `E`, a set those bits decide too: an event of
the bits at strings `S` the choice keeps off `E` has, at the choice, the chance it has at any
fixed one. -/
theorem fresh_select_le [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) {σ : Type*}
    (E : Ω → Finset (FreeMonoid α)) (sel : Ω → σ)
    (hdet : ∀ ω ω', (∀ x ∈ E ω, O.noise x ω = O.noise x ω') → E ω' = E ω ∧ sel ω' = sel ω)
    (S : σ → Finset (FreeMonoid α)) (hS : ∀ ω, Disjoint (S (sel ω)) (E ω))
    (Bad : σ → Set Ω) (hBad : ∀ s, MeasurableSet[noiseAlg O ↑(S s)] (Bad s)) {φ : ℝ≥0∞}
    (hφ : ∀ s, μ (Bad s) ≤ φ) :
    μ {ω | ω ∈ Bad (sel ω)} ≤ φ := by
  classical
  set PC : Finset (FreeMonoid α) × Finset (FreeMonoid α) → Set Ω := fun c =>
    {ω | noisePattern O c.1 ω = c.2} ∩ noiseClean O c.1 with hPC
  set C : Finset (FreeMonoid α) × Finset (FreeMonoid α) → Set Ω := fun c =>
    PC c ∩ cleanAll O ∩ {ω | E ω = c.1} with hCdef
  have hagree : ∀ c ω ω', ω ∈ C c → ω' ∈ PC c → ∀ x ∈ E ω, O.noise x ω = O.noise x ω' := by
    intro c ω ω' hω hω' x hx
    obtain ⟨⟨⟨hp, hc⟩, -⟩, hT⟩ := hω
    rw [show E ω = c.1 from hT] at hx
    exact noise_eq_of_pattern O hc hω'.2 (hp.trans hω'.1.symm) x hx
  have hC_eq : ∀ c, (C c).Nonempty → C c = PC c ∩ cleanAll O := by
    intro c ⟨ω₀, h₀⟩
    refine Set.Subset.antisymm (fun ω hω => hω.1) fun ω hω => ⟨hω, ?_⟩
    exact (hdet ω₀ ω (hagree c ω₀ ω h₀ hω.1)).1.trans h₀.2
  have hsel : ∀ c ω, (h : (C c).Nonempty) → ω ∈ C c → sel ω = sel h.some := by
    intro c ω hne hω
    exact (hdet _ ω (hagree c _ ω hne.some_mem hω.1.1)).2
  have hmeasPC : ∀ c, MeasurableSet (PC c) := fun c =>
    noiseAlg_le O _ _ ((measurableSet_noisePattern O _ c.2).inter (measurableSet_noiseClean O _))
  have hmeasC : ∀ c, MeasurableSet (C c) := by
    intro c
    by_cases hne : (C c).Nonempty
    · rw [hC_eq c hne]; exact (hmeasPC c).inter (measurableSet_cleanAll O)
    · rw [Set.not_nonempty_iff_eq_empty.1 hne]; exact MeasurableSet.empty
  have hdisj : Pairwise (Function.onFun Disjoint C) := by
    intro c c' hcc'
    rw [Function.onFun, Set.disjoint_left]
    intro ω hω hω'
    apply hcc'
    have h1 : c.1 = c'.1 := (show E ω = c.1 from hω.2).symm.trans (show E ω = c'.1 from hω'.2)
    have h2 : c.2 = c'.2 := by
      have a : noisePattern O c.1 ω = c.2 := hω.1.1.1
      have b : noisePattern O c'.1 ω = c'.2 := hω'.1.1.1
      rw [h1] at a
      exact a.symm.trans b
    exact Prod.ext h1 h2
  have hcover : {ω | ω ∈ Bad (sel ω)} ⊆ (cleanAll O)ᶜ ∪ ⋃ c, {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)} := by
    intro ω hω
    by_cases hcl : ω ∈ cleanAll O
    · right
      simp only [Set.mem_iUnion]
      exact ⟨(E ω, noisePattern O (E ω) ω), ⟨⟨⟨rfl, fun x _ => hcl x⟩, hcl⟩, rfl⟩, hω⟩
    · exact .inl hcl
  have hcell : ∀ c, μ {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)} ≤ φ * μ (C c) := by
    intro c
    by_cases hne : (C c).Nonempty
    · have hsub : {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)} ⊆ PC c ∩ Bad (sel hne.some) := by
        rintro ω ⟨hω, hb⟩
        exact ⟨hω.1.1, hsel c ω hne hω ▸ hb⟩
      have hdisj' : Disjoint (↑c.1 : Set (FreeMonoid α)) ↑(S (sel hne.some)) := by
        have := hS hne.some
        rw [hne.some_mem.2] at this
        exact Finset.disjoint_coe.2 this.symm
      have hind := indep_noiseAlg O hdisj'
      have hPCm : MeasurableSet[noiseAlg O ↑c.1] (PC c) :=
        (measurableSet_noisePattern O _ c.2).inter (measurableSet_noiseClean O _)
      calc μ {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)} ≤ μ (PC c ∩ Bad (sel hne.some)) := measure_mono hsub
        _ = μ (PC c) * μ (Bad (sel hne.some)) := (Indep_iff _ _ μ).1 hind _ _ hPCm (hBad _)
        _ ≤ μ (PC c) * φ := by gcongr; exact hφ _
        _ = φ * μ (C c) := by
            rw [mul_comm, hC_eq c hne, measure_inter_conull (measure_cleanAll_compl O)]
    · have : {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)} = ∅ := by
        ext ω
        simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_and]
        exact fun hω => absurd ⟨ω, hω⟩ hne
      rw [this, measure_empty]
      exact zero_le
  calc μ {ω | ω ∈ Bad (sel ω)}
      ≤ μ ((cleanAll O)ᶜ ∪ ⋃ c, {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)}) := measure_mono hcover
    _ ≤ μ (cleanAll O)ᶜ + μ (⋃ c, {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)}) := measure_union_le _ _
    _ = μ (⋃ c, {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)}) := by rw [measure_cleanAll_compl O, zero_add]
    _ ≤ ∑' c, μ {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)} := measure_iUnion_le _
    _ ≤ ∑' c, φ * μ (C c) := ENNReal.tsum_le_tsum hcell
    _ = φ * ∑' c, μ (C c) := ENNReal.tsum_mul_left
    _ = φ * μ (⋃ c, C c) := by rw [measure_iUnion hdisj hmeasC]
    _ ≤ φ * 1 := by gcongr; exact prob_le_one
    _ = φ := mul_one _

end OrthoDFA

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- `fresh_select_le` at the choices `Good` holds of: the event's chance is at most `φ` times the
chance the choice is good. -/
theorem fresh_select_le' [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) {σ : Type*}
    (E : Ω → Finset (FreeMonoid α)) (sel : Ω → σ)
    (hdet : ∀ ω ω', (∀ x ∈ E ω, O.noise x ω = O.noise x ω') → E ω' = E ω ∧ sel ω' = sel ω)
    (S : σ → Finset (FreeMonoid α)) (hS : ∀ ω, Disjoint (S (sel ω)) (E ω))
    (Good : σ → Prop) (Bad : σ → Set Ω) (hBad : ∀ s, MeasurableSet[noiseAlg O ↑(S s)] (Bad s))
    (hnot : ∀ s, ¬ Good s → Bad s = ∅) {φ : ℝ≥0∞} (hφ : ∀ s, Good s → μ (Bad s) ≤ φ) :
    μ {ω | ω ∈ Bad (sel ω)} ≤ φ * μ {ω | Good (sel ω)} := by
  classical
  set PC : Finset (FreeMonoid α) × Finset (FreeMonoid α) → Set Ω := fun c =>
    {ω | noisePattern O c.1 ω = c.2} ∩ noiseClean O c.1 with hPC
  set C : Finset (FreeMonoid α) × Finset (FreeMonoid α) → Set Ω := fun c =>
    PC c ∩ cleanAll O ∩ {ω | E ω = c.1} with hCdef
  have hagree : ∀ c ω ω', ω ∈ C c → ω' ∈ PC c → ∀ x ∈ E ω, O.noise x ω = O.noise x ω' := by
    intro c ω ω' hω hω' x hx
    obtain ⟨⟨⟨hp, hc⟩, -⟩, hT⟩ := hω
    rw [show E ω = c.1 from hT] at hx
    exact noise_eq_of_pattern O hc hω'.2 (hp.trans hω'.1.symm) x hx
  have hC_eq : ∀ c, (C c).Nonempty → C c = PC c ∩ cleanAll O := by
    intro c ⟨ω₀, h₀⟩
    refine Set.Subset.antisymm (fun ω hω => hω.1) fun ω hω => ⟨hω, ?_⟩
    exact (hdet ω₀ ω (hagree c ω₀ ω h₀ hω.1)).1.trans h₀.2
  have hsel : ∀ c ω, (h : (C c).Nonempty) → ω ∈ C c → sel ω = sel h.some := by
    intro c ω hne hω
    exact (hdet _ ω (hagree c _ ω hne.some_mem hω.1.1)).2
  have hmeasPC : ∀ c, MeasurableSet (PC c) := fun c =>
    noiseAlg_le O _ _ ((measurableSet_noisePattern O _ c.2).inter (measurableSet_noiseClean O _))
  have hmeasC : ∀ c, MeasurableSet (C c) := by
    intro c
    by_cases hne : (C c).Nonempty
    · rw [hC_eq c hne]; exact (hmeasPC c).inter (measurableSet_cleanAll O)
    · rw [Set.not_nonempty_iff_eq_empty.1 hne]; exact MeasurableSet.empty
  set G : Finset (FreeMonoid α) × Finset (FreeMonoid α) → Set Ω := fun c =>
    if ∃ ω ∈ C c, Good (sel ω) then C c else ∅ with hG
  have hGsub : ∀ c, G c ⊆ C c := by
    intro c; simp only [hG]; split_ifs <;> simp
  have hmeasG : ∀ c, MeasurableSet (G c) := by
    intro c; simp only [hG]; split_ifs
    · exact hmeasC c
    · exact MeasurableSet.empty
  have hdisj : Pairwise (Function.onFun Disjoint G) := by
    intro c c' hcc'
    rw [Function.onFun, Set.disjoint_left]
    intro ω hω hω'
    apply hcc'
    have hω := hGsub c hω
    have hω' := hGsub c' hω'
    have h1 : c.1 = c'.1 := (show E ω = c.1 from hω.2).symm.trans (show E ω = c'.1 from hω'.2)
    have h2 : c.2 = c'.2 := by
      have a : noisePattern O c.1 ω = c.2 := hω.1.1.1
      have b : noisePattern O c'.1 ω = c'.2 := hω'.1.1.1
      rw [h1] at a
      exact a.symm.trans b
    exact Prod.ext h1 h2
  have hGgood : ⋃ c, G c ⊆ {ω | Good (sel ω)} := by
    intro ω hω
    simp only [Set.mem_iUnion] at hω
    obtain ⟨c, hc⟩ := hω
    simp only [hG] at hc
    split_ifs at hc with h
    · obtain ⟨ω₀, h₀, hg⟩ := h
      have hne : (C c).Nonempty := ⟨ω, hc⟩
      change Good (sel ω)
      rw [hsel c ω hne hc]
      rw [hsel c ω₀ hne h₀] at hg
      exact hg
    · exact absurd hc (Set.notMem_empty _)
  have hcover : {ω | ω ∈ Bad (sel ω)} ⊆ (cleanAll O)ᶜ ∪ ⋃ c, {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)} := by
    intro ω hω
    by_cases hcl : ω ∈ cleanAll O
    · right
      simp only [Set.mem_iUnion]
      exact ⟨(E ω, noisePattern O (E ω) ω), ⟨⟨⟨rfl, fun x _ => hcl x⟩, hcl⟩, rfl⟩, hω⟩
    · exact .inl hcl
  have hcell : ∀ c, μ {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)} ≤ φ * μ (G c) := by
    intro c
    by_cases hne : (C c).Nonempty
    · have hsub : {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)} ⊆ PC c ∩ Bad (sel hne.some) := by
        rintro ω ⟨hω, hb⟩
        exact ⟨hω.1.1, hsel c ω hne hω ▸ hb⟩
      by_cases hg : Good (sel hne.some)
      · have hdisj' : Disjoint (↑c.1 : Set (FreeMonoid α)) ↑(S (sel hne.some)) := by
          have := hS hne.some
          rw [hne.some_mem.2] at this
          exact Finset.disjoint_coe.2 this.symm
        have hind := indep_noiseAlg O hdisj'
        have hPCm : MeasurableSet[noiseAlg O ↑c.1] (PC c) :=
          (measurableSet_noisePattern O _ c.2).inter (measurableSet_noiseClean O _)
        have hGc : G c = C c := by
          simp only [hG]; rw [if_pos ⟨hne.some, hne.some_mem, hg⟩]
        calc μ {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)} ≤ μ (PC c ∩ Bad (sel hne.some)) :=
              measure_mono hsub
          _ = μ (PC c) * μ (Bad (sel hne.some)) := (Indep_iff _ _ μ).1 hind _ _ hPCm (hBad _)
          _ ≤ μ (PC c) * φ := by gcongr; exact hφ _ hg
          _ = φ * μ (G c) := by
              rw [mul_comm, hGc, hC_eq c hne, measure_inter_conull (measure_cleanAll_compl O)]
      · have : PC c ∩ Bad (sel hne.some) = ∅ := by rw [hnot _ hg, Set.inter_empty]
        rw [this] at hsub
        rw [Set.subset_empty_iff.1 hsub, measure_empty]
        exact zero_le
    · have : {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)} = ∅ := by
        ext ω
        simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_and]
        exact fun hω => absurd ⟨ω, hω⟩ hne
      rw [this, measure_empty]
      exact zero_le
  calc μ {ω | ω ∈ Bad (sel ω)}
      ≤ μ ((cleanAll O)ᶜ ∪ ⋃ c, {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)}) := measure_mono hcover
    _ ≤ μ (cleanAll O)ᶜ + μ (⋃ c, {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)}) := measure_union_le _ _
    _ = μ (⋃ c, {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)}) := by rw [measure_cleanAll_compl O, zero_add]
    _ ≤ ∑' c, μ {ω | ω ∈ C c ∧ ω ∈ Bad (sel ω)} := measure_iUnion_le _
    _ ≤ ∑' c, φ * μ (G c) := ENNReal.tsum_le_tsum hcell
    _ = φ * ∑' c, μ (G c) := ENNReal.tsum_mul_left
    _ = φ * μ (⋃ c, G c) := by rw [measure_iUnion hdisj hmeasG]
    _ ≤ φ * μ {ω | Good (sel ω)} := by gcongr

end OrthoDFA
