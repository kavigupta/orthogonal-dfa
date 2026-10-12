import OrthoDFA.Proofs.RandomStretch
import OrthoDFA.Proofs.RandomNoise
import OrthoDFA.Proofs.IdealRound
import OrthoDFA.Proofs.Yield

/-!
# The random round: the proofs

A run with no bad step ends well within `stretches · Ns` probes: each stretch is shorter than
`Ns` probes, and each change of hypothesis lowers `Ψr`. The chance of a bad step is at most what
the current stretch risks plus `Ψr` times what a fresh one does (`anyB_le`), the chance part of
#417 with its runs of clean probes replaced by stretches. Over the reads, the wrong ones are
null and the read field is good but for `noiseRisk`.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory
open OrthoDFA.Ideal (DTree Disagrees pi_succ_apply)

section Compose

variable {S X E : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
  (step : S → X → S ⊕ E) (B : S → X → Prop) (same : S → S → Prop)
  (D : Measure X) [IsProbabilityMeasure D] (I F : S → Prop) (Ψ : S → ℕ) (b : ENNReal)

/-- The chance of a bad step is at most the current segment's, plus `Ψ` times a fresh one's. -/
theorem anyB_le
    (hsame : ∀ s x s', I s → step s x = .inl s' → same s s' → ¬ B s x → I s' ∧ Ψ s' = Ψ s)
    (hchg : ∀ s x s', I s → step s x = .inl s' → ¬ same s s' → ¬ B s x →
      I s' ∧ F s' ∧ Ψ s' < Ψ s)
    (hseg : ∀ s, I s → F s → ∀ N, (Measure.pi fun _ : Fin N => D)
      {xs | SegB step B same s (List.ofFn xs)} ≤ b) :
    ∀ N s, I s → (Measure.pi fun _ : Fin N => D) {xs | AnyB step B s (List.ofFn xs)}
      ≤ (Measure.pi fun _ : Fin N => D) {xs | SegB step B same s (List.ofFn xs)} + Ψ s * b := by
  intro N
  induction N with
  | zero =>
    intro s _
    have : {xs : Fin 0 → X | AnyB step B s (List.ofFn xs)} = ∅ := by ext xs; simp [AnyB]
    rw [this, measure_empty]
    exact bot_le
  | succ N ih =>
    intro s hs
    rw [pi_succ_apply, pi_succ_apply]
    calc ∫⁻ x, (Measure.pi fun _ : Fin N => D)
          {xs | Fin.cons x xs ∈ {xs : Fin (N + 1) → X | AnyB step B s (List.ofFn xs)}} ∂D
        ≤ ∫⁻ x, ((Measure.pi fun _ : Fin N => D)
          {xs | Fin.cons x xs ∈ {xs : Fin (N + 1) → X | SegB step B same s (List.ofFn xs)}}
            + Ψ s * b) ∂D := lintegral_mono fun x => ?_
      _ = _ := by
          rw [lintegral_add_right _ measurable_const, lintegral_const, measure_univ, mul_one]
    simp only [Set.mem_ofPred_eq, List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, AnyB, SegB]
    by_cases hB : B s x
    · simp only [hB, true_or, Set.ofPred_true, measure_univ]
      exact le_add_right le_rfl
    · simp only [hB, false_or]
      rcases hst : step s x with s' | e
      · simp only [Sum.inl.injEq, exists_eq_left']
        by_cases hsm : same s s'
        · obtain ⟨hs', hΨ⟩ := hsame s x s' hs hst hsm hB
          simp only [hsm, true_and]
          rw [← hΨ]
          exact ih s' hs'
        · obtain ⟨hs', hF, hΨ⟩ := hchg s x s' hs hst hsm hB
          simp only [hsm, false_and, Set.ofPred_false, measure_empty, zero_add]
          calc _ ≤ _ := ih s' hs'
            _ ≤ b + Ψ s' * b := add_le_add (hseg s' hs' hF N) le_rfl
            _ = ((Ψ s' + 1 : ℕ) : ENNReal) * b := by push_cast; ring
            _ ≤ Ψ s * b := by gcongr; exact_mod_cast hΨ
      · simp only [reduceCtorEq, false_and, exists_false, Set.ofPred_false, measure_empty]
        exact bot_le

end Compose

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Cfg)
  {σ : Type*} [Fintype σ] (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) (D : Measure (FreeMonoid α))
  (ε : ℝ) (Ns : ℕ)

/-- A run with no bad step ends well, given enough probes. -/
theorem ends_of_not_anyB (X : RState α → FreeMonoid α → Prop) (hm : 1 ≤ C.m) :
    ∀ (xs : List (FreeMonoid α)) (s : RState α), Inv read C s → s.n < Ns →
      Ψr C s * Ns + (Ns - s.n) ≤ xs.length →
      ¬ AnyB (step read C) (BadStep read C M U θ D ε Ns X) s xs →
      EndsWell M (BadAt U θ) D ε
        {y | Disagrees read (round read C s xs).1.tree (round read C s xs).1.edges C.k y}
        0 0 (fun _ _ => []) 0 (toRoundEnd (round read C s xs).2)
  | [], s, _, hn, hlen, _ => by simp at hlen; omega
  | x :: xs, s, hs, hn, hlen, hb => by
    simp only [AnyB, not_or, not_exists, not_and] at hb
    obtain ⟨hb1, hb2⟩ := hb
    rcases hst : step read C s x with s' | e
    · simp only [round, hst]
      by_cases hsm : Same s s'
      · obtain ⟨hs', hΨ, hn'⟩ := step_same read C hs hst hsm.1 hsm.2
        have hlt : s'.n < Ns := by
          have : s'.n ≠ Ns := fun h => hb1 (Or.inr ⟨s', hst, hsm, h⟩)
          omega
        refine ends_of_not_anyB X hm xs s' hs' hlt ?_ (hb2 s' hst)
        rw [hΨ, hn']
        simp only [List.length_cons] at hlen
        omega
      · obtain ⟨hs', hf, hΨ⟩ := step_change read C hm hs hst hsm
        have h0 : s'.n = 0 := by rw [hf]; rfl
        refine ends_of_not_anyB X hm xs s' hs' (by omega) ?_ (hb2 s' hst)
        rw [h0, Nat.sub_zero]
        simp only [List.length_cons] at hlen
        have : (Ψr C s' + 1) * Ns ≤ Ψr C s * Ns := Nat.mul_le_mul_right _ hΨ
        rw [Nat.add_mul, one_mul] at this
        omega
    · simp only [round, hst]
      by_contra hne
      exact hb1 (Or.inl ⟨hn, Or.inl ⟨e, hst, hne⟩⟩)

theorem probe_level {side : σ → Bool} (hW : NoWrong M side read) [IsProbabilityMeasure D]
    (L P N₁ : ℕ) (G θr εd' θpt' : ℝ) (hL : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hN : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
      D.real (PotGood M U θ C.k read T) ≤ G)
    (hcap : Fintype.card σ + 2 ≤ C.Lmax) (hm : 1 ≤ C.m) (hn₀ : 1 ≤ C.n₀) (hN₁ : C.n₀ ≤ N₁)
    (hN₁s : N₁ ≤ Ns) (hθ : 0 ≤ θ) (hG0 : 0 ≤ G) (hG1 : G ≤ 1) (ha : 0 ≤ C.a)
    (hθs0 : 0 ≤ C.θs) (hθs1 : C.θs ≤ 1) (hθe : 0 ≤ C.θe) (hθpt0 : 0 ≤ C.θpt)
    (hθpt1 : C.θpt ≤ 1) (hεd0 : 0 ≤ C.εd) (hεd1 : C.εd ≤ 1) (hεd : C.εd ≤ ε) (hθr0 : 0 ≤ θr)
    (hθr1 : θr ≤ 1) (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1) (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1)
    (hsep : (C.Lmax : ℝ) ^ 2 * Fintype.card α * θr ≤ (1 - θpt') * εd')
    (hP : stretches C (Fintype.card α) * Ns ≤ P) :
    (Measure.pi fun _ : Fin P => D)
        {xs | let r := round read C start (List.ofFn xs)
          ¬ EndsWell M (BadAt U θ) D ε {x | Disagrees read r.1.tree r.1.edges C.k x}
            0 0 (fun _ _ => []) 0 (toRoundEnd r.2)}
      ≤ ENNReal.ofReal (stretches C (Fintype.card α) * stretchRisk C L Ns N₁ G θr εd' θpt') := by
  have h2 : 2 ≤ C.Lmax := by omega
  have hNs : 0 < Ns := by omega
  have hst := inv_start read C hm h2 (α := α)
  have hΨ := Ψr_start C h2 (α := α)
  set b := ENNReal.ofReal (stretchRisk C L Ns N₁ G θr εd' θpt')
  set X : RState α → FreeMonoid α → Prop := fun _ _ => False
  have hseg : ∀ s, Inv read C s → s = fresh s.tree s.edges s.moved → ∀ N,
      (Measure.pi fun _ : Fin N => D)
        {xs | SegB (step read C) (BadStep read C M U θ D ε Ns X) Same s (List.ofFn xs)} ≤ b := by
    intro s hs hf N
    have := seg_le read C M U θ D ε Ns X (fun _ => False) 0 L N₁ G θr εd' θpt' hL hs hf
      (cls_mem read C hW hs) hN (fun _ _ _ _ _ h => h)
      (fun _ s' x hs' _ _ _ => step_ne_tooBig read C hW hcap hs' x)
      (fun N => by simp) hm hn₀ hN₁ hN₁s hG0 hG1 ha hθs0 hθs1 hθe hθpt0 hθpt1 hεd0 hεd1 hεd
      hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep N
    rwa [add_zero] at this
  have hany := anyB_le (step read C) (BadStep read C M U θ D ε Ns X) Same D (Inv read C)
    (fun s => s = fresh s.tree s.edges s.moved) (Ψr C) b
    (fun s x s' hs h hsm _ => by
      obtain ⟨h1, h2, -⟩ := step_same read C hs h hsm.1 hsm.2
      exact ⟨h1, h2⟩)
    (fun s x s' hs h hsm _ => step_change read C hm hs h hsm) hseg P start hst
  calc _ ≤ (Measure.pi fun _ : Fin P => D)
        {xs | AnyB (step read C) (BadStep read C M U θ D ε Ns X) start (List.ofFn xs)} := by
        refine measure_mono fun xs hxs => ?_
        by_contra hb
        have hn0 : (start : RState α).n = 0 := rfl
        refine hxs (ends_of_not_anyB read C M U θ D ε Ns X hm _ start hst
          (by rw [hn0]; exact hNs) ?_ hb)
        rw [hn0, Nat.sub_zero, List.length_ofFn]
        have : (Ψr C (start : RState α) + 1) * Ns ≤ stretches C (Fintype.card α) * Ns :=
          Nat.mul_le_mul_right _ hΨ
        rw [Nat.add_mul, one_mul] at this
        omega
    _ ≤ b + Ψr C (start : RState α) * b := hany.trans (add_le_add (hseg _ hst rfl P) le_rfl)
    _ = ((Ψr C (start : RState α) + 1 : ℕ) : ENNReal) * b := by push_cast; ring
    _ ≤ (stretches C (Fintype.card α) : ENNReal) * b := by gcongr
    _ = _ := by
        rw [ENNReal.ofReal_mul (by positivity), ENNReal.ofReal_natCast]

theorem le_prefixMax [Fintype α] (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (u : FreeMonoid α) : D.real {x | Ideal.pre x k = u} ≤ prefixMax D k :=
  le_ciSup (f := fun u => D.real {x | Ideal.pre x k = u}) ⟨1, by
    rintro _ ⟨v, rfl⟩; exact measureReal_le_one⟩ u

theorem prefixMax_pos [Fintype α] (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ) :
    0 < prefixMax D k := by
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  by_contra h
  push_neg at h
  have h0 : ∀ u : FreeMonoid α, D {x | Ideal.pre x k = u} = 0 := by
    intro u
    have h1 := (le_prefixMax D k u).trans h
    have h2 : D.real {x | Ideal.pre x k = u} = 0 := le_antisymm h1 measureReal_nonneg
    rwa [measureReal_eq_zero_iff] at h2
  have : (Set.univ : Set (FreeMonoid α)) = ⋃ u, {x | Ideal.pre x k = u} := by
    ext x; simp
  have hu := measure_iUnion_null (μ := D) h0
  rw [← this, measure_univ] at hu
  exact one_ne_zero hu

section Root

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Cfg)

theorem finish_tree {s s' : RState α} (h : finish C s = .inl s') : s'.tree = s.tree := by
  unfold finish at h
  split at h
  · cases h
  · injection h with h; rw [← h]

theorem fix_root {s s' : RState α} {p : List Bool} {c : α} {u : FreeMonoid α} {t : List Bool}
    (h : fix read C s p c u t = .inl s') (hs : Ideal.RootOne s.tree) : Ideal.RootOne s'.tree := by
  unfold fix split at h
  simp only at h
  repeat' (first | split_ifs at h | split at h)
  all_goals first
    | (injection h with h; rw [← h]; exact Ideal.splitAt_root hs _ _)
    | (injection h with h; rw [← h]; exact hs)

theorem step_root {s s' : RState α} {x : FreeMonoid α} (h : step read C s x = .inl s')
    (hs : Ideal.RootOne s.tree) : Ideal.RootOne s'.tree := by
  unfold step at h
  split at h
  · injection h with h; rw [← h]; exact hs
  · split_ifs at h
    · exact fix_root read C h hs
    · exact finish_tree C h ▸ hs
  · exact finish_tree C h ▸ hs

theorem round_root : ∀ (xs : List (FreeMonoid α)) (s : RState α), Ideal.RootOne s.tree →
    Ideal.RootOne (round read C s xs).1.tree
  | [], s, hs => hs
  | x :: xs, s, hs => by
    simp only [round]
    split
    · rename_i s' h; exact round_root xs s' (step_root read C h hs)
    · exact hs

end Root

theorem randomRoundCorrect_holds : RandomRoundCorrect := by
  intro α _ _ σ _ M side U θ Ω _ μ _ R P C ε
  obtain ⟨D, _, L, st, hst, hL, q, hq, hpreq⟩ := P
  obtain ⟨read, hmeas, hind, hU, hwrong⟩ := R
  simp only [roundRisk, ENNReal.iInf_add]
  refine le_iInf₂ fun kn hA => ?_
  obtain ⟨Ns, N₁, G, θr, εd', θpt'⟩ := kn
  obtain ⟨hcap, hm, hn₀, hN₁, hN₁s, hθ, hG0, hG1, ha, hθs0, hθs1, hθe, hθpt0, hθpt1, hεd0, hεd1,
    hεd, hθr0, hθr1, hεd'0, hεd'1, hθpt'0, hθpt'1, hsep, hP⟩ := hA
  simp only at hN₁ hN₁s hG0 hG1 hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep hP
  unfold roundBound
  simp only
  set P := C.budget
  set p₀ := prefixMax D C.k
  have hp₀ : 0 < p₀ := prefixMax_pos D C.k
  have hpre : ∀ u : FreeMonoid α, u.toList.length = C.k → D.real {x | Ideal.pre x C.k = u} ≤ p₀ :=
    fun u _ => le_prefixMax D C.k u
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  set W := {ω | ¬ NoWrong M side (read · ω)}
  set Nn := {ω | ¬ ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
    D.real (PotGood M U θ C.k (read · ω) T) ≤ G}
  set m₀ := 3 / 2 * θ * coverage D st L M (BadAt U θ) / (L + 1 - st : ℕ)
  set Ny := {ω | (D.prod (anchorLaw st L)).real {r | r.2 ≤ r.1.toList.length ∧
      read (FreeMonoid.ofList (r.1.toList.take r.2)) ω = .undecided} < m₀ / 2}
  have hm₀ : m₀ ≤ 3 / 2 * θ * (D.prod (anchorLaw st L)).real {r | r.2 ≤ r.1.toList.length ∧
      BadAt U θ (M.eval (FreeMonoid.ofList (r.1.toList.take r.2)).toList)} := by
    have hcov : coverage D st L M (BadAt U θ) / (L + 1 - st : ℕ)
        ≤ (D.prod (anchorLaw st L)).real {r | r.2 ≤ r.1.toList.length ∧
          BadAt U θ (M.eval (FreeMonoid.ofList (r.1.toList.take r.2)).toList)} :=
      coverage_le_anchor D st L M (BadAt U θ)
    simp only [m₀, mul_div_assoc]
    exact mul_le_mul_of_nonneg_left hcov (by linarith)
  have hW : μ W = 0 := by
    have : W ⊆ ⋃ z, {ω | read z ω = if side (M.eval z.toList) then .reject else .accept} := by
      intro ω hω
      simp only [W, NoWrong, Set.mem_ofPred_eq, not_forall, not_not] at hω
      obtain ⟨z, hz⟩ := hω
      exact Set.mem_iUnion.2 ⟨z, hz⟩
    exact measure_mono_null this (measure_iUnion_null hwrong)
  have hNn := noise_le M U θ μ read D C.k L p₀ G hmeas hind hU hL hp₀ hpre hθ
  have hNy : μ Ny ≤ ENNReal.ofReal
      (Real.exp (-(3 / 2 * θ * coverage D st L M (BadAt U θ)) / (16 * q))) := by
    rw [← ENNReal.ofReal_toReal (measure_ne_top μ Ny)]
    refine ENNReal.ofReal_le_ofReal ((anchor_reads read hmeas hind D hst
      (fun u => BadAt U θ (M.eval u.toList)) (3 / 2 * θ) (fun u hu => (hU u).symm ▸ hu.le) m₀ q
      hm₀ hq hpreq).trans_eq ?_)
    have hn : ((L + 1 - st : ℕ) : ℝ) ≠ 0 := by exact_mod_cast (anchorCount_pos hst).ne'
    simp only [m₀]
    congr 1
    field_simp
  set c := ENNReal.ofReal (stretches C (Fintype.card α) * stretchRisk C L Ns N₁ G θr εd' θpt')
  set B := toMeasurable μ (W ∪ Nn ∪ Ny)
  have hpt : ∀ ω, (Measure.pi fun _ : Fin P => D)
      {xs | let r := round (read · ω) C start (List.ofFn xs)
        ¬ EndsWell M (BadAt U θ) D ε {x | Disagrees (read · ω) r.1.tree r.1.edges C.k x}
          st L (Ideal.replay (read · ω) r.1.tree r.1.edges)
          (3 / 4 * θ * coverage D st L M (BadAt U θ) / (L + 1 - st : ℕ))
          (toRoundEnd r.2)}
        ≤ c + B.indicator 1 ω := by
    intro ω
    by_cases hω : ω ∈ B
    · rw [Set.indicator_of_mem hω, Pi.one_apply]
      exact prob_le_one.trans le_add_self
    · have hω' : ω ∉ W ∪ Nn ∪ Ny := fun h => hω (subset_toMeasurable _ _ h)
      rw [Set.indicator_of_notMem hω, add_zero]
      simp only [Set.mem_union, not_or] at hω'
      obtain ⟨⟨hωW, hωN⟩, hωY⟩ := hω'
      simp only [W, Nn, Set.mem_ofPred_eq, not_not] at hωW hωN
      simp only [Ny, Set.mem_ofPred_eq, not_lt] at hωY
      refine le_trans (measure_mono ?_) (probe_level (read · ω) C M U θ D ε Ns hωW L P N₁ G
        θr εd' θpt' hL hωN hcap hm hn₀ hN₁ hN₁s hθ hG0 hG1 ha hθs0 hθs1 hθe hθpt0 hθpt1 hεd0
        hεd1 hεd hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep hP)
      intro xs hxs h
      refine hxs (h.respread (Ideal.replay_spread (read · ω) _ _ D st L) ?_)
      have hroot := round_root (read · ω) C (List.ofFn xs) start ⟨_, _, rfl⟩
      refine le_trans (le_of_eq ?_) (hωY.trans (measureReal_mono ?_ (measure_ne_top _ _)))
      · simp only [m₀]; ring
      · rintro ⟨w, e⟩ ⟨he, hu⟩
        exact Ideal.replay_root (read · ω) hroot _ he hu
  have hnn : 0 ≤ stretchRisk C L Ns N₁ G θr εd' θpt' :=
    stretchRisk_nonneg C L Ns N₁ G θr εd' θpt' hG0 hG1 ha hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1
  calc _ ≤ ∫⁻ ω, (c + B.indicator 1 ω) ∂μ := lintegral_mono hpt
    _ = c + μ B := by
        rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
          lintegral_indicator_one (measurableSet_toMeasurable _ _)]
    _ ≤ c + (μ W + μ Nn + μ Ny) := by
        gcongr
        rw [measure_toMeasurable]
        exact (measure_union_le _ _).trans (by gcongr; exact measure_union_le _ _)
    _ ≤ c + (ENNReal.ofReal (noiseRisk (Fintype.card σ) (Fintype.card α) L (3 / 2 * θ) p₀ G)
          + ENNReal.ofReal (Real.exp (-(3 / 2 * θ * coverage D st L M (BadAt U θ)) / (16 * q)))) := by
        rw [hW, zero_add]
        gcongr
    _ = _ := by
        rw [← add_assoc, add_comm c, ← ENNReal.ofReal_add (by unfold noiseRisk; positivity)
          (by positivity)]

end Random

end OrthoDFA
