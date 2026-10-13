import OrthoDFA.Proofs.IdealSteps
import OrthoDFA.Proofs.IdealHarvest
import Mathlib.Probability.ProductMeasure

/-!
# The idealized round: the proofs

Every step that does not end the round lowers `Θ`, and the start's `Θ` is below the budget, so the
round ends on its probes, and `step_ok` says how.

The chance part: the hypothesis is fixed while a run of clean probes grows, and the probes are
fresh, so a hypothesis whose probes disagree more than `ε` of the time completes a run of `n` with
chance at most `(1 − ε)^n`. Each probe that does not continue a run starts at most one new one.

A harvest's last probe is charged a string at its site, and a probe is charged one at a site whose
sampler yields at most `y` with chance at most `y`; a probe is charged at one of at most
`2 + (|Q| + 2)|Σ|` sites.
-/

namespace OrthoDFA

namespace Ideal

open MeasureTheory Finset

variable {α : Type*} [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level2Settings)

/-! ## The round ends -/

section Ends

variable [Fintype α] {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}

theorem round_ok (hI : IdealReads read M side) (hcap : Fintype.card σ + 2 ≤ C.maxLeaves)
    (hm : 1 ≤ C.recordsToMove) (hh : 1 ≤ C.harvestSize) (hn : 1 ≤ C.cleanProbesToEnd) :
    ∀ (xs : List (FreeMonoid α)) (s : Level2State α), Inv read C s → Reached read s.tree →
      Θ C s < xs.length → EndOK (runProbes read C s xs).2 ∧
        Inv read C (runProbes read C s xs).1 ∧ Reached read (runProbes read C s xs).1.tree
  | [], s, _, _, h => by simp at h
  | x :: xs, s, hs, hr, h => by
    have hst := step_ok read C hI hcap hm hh hn hs hr x
    rcases hstep : applyProbe read C s x with s' | e <;> rw [hstep] at hst <;>
      simp only [runProbes, hstep]
    · obtain ⟨hs', hr', hlt⟩ := hst
      exact round_ok hI hcap hm hh hn xs s' hs' hr' (by simp only [List.length_cons] at h; omega)
    · exact ⟨hst, hs, hr⟩

omit [DecidableEq α] [Fintype α] in
theorem leaves_start : (initialState : Level2State α).tree.leaves = [[false], [true]] := by
  simp [initialState, newHypothesis, DTree.leaves]

omit [Fintype α] in
theorem inv_start (hm : 1 ≤ C.recordsToMove) (hh : 1 ≤ C.harvestSize) (hn : 1 ≤ C.cleanProbesToEnd)
    (h2 : 2 ≤ C.maxLeaves) :
    Inv read C (initialState : Level2State α) :=
  inv_fresh read C hm hh hn (fun _ _ _ _ _ h => by simp at h) (by simp [DTree.leaves]; omega)

omit [DecidableEq α] [Fintype α] in
theorem reached_start : Reached read (initialState : Level2State α).tree := by
  intro p hp
  rw [leaves_start] at hp
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl
  · exact Or.inl rfl
  · exact Or.inr (Or.inl rfl)

theorem Θ_start (hn : 1 ≤ C.cleanProbesToEnd) (h2 : 2 ≤ C.maxLeaves) :
    Θ C (initialState : Level2State α) < probeBudget C (Fintype.card α) := by
  have hΦ := Φ_le (initialState : Level2State α)
  rw [leaves_start] at hΦ
  have hΨ : Ψ C (initialState : Level2State α) ≤ C.maxLeaves * X (α := α) C := by
    unfold Ψ
    rw [leaves_start]
    have hL : C.maxLeaves * X (α := α) C = (C.maxLeaves - 2) * X (α := α) C + 2 * X (α := α) C := by
      rw [← Nat.add_mul, Nat.sub_add_cancel h2]
    have hX : 4 * Fintype.card α ≤ 2 * X (α := α) C := by
      unfold X
      have := Nat.mul_le_mul_right (Fintype.card α) h2
      nlinarith
    simp only [List.length_cons, List.length_nil, zero_add, Nat.reduceAdd] at hΦ ⊢
    omega
  have hρ : ρ C (initialState : Level2State α) + 1 ≤ B (α := α) C := by
    unfold ρ B Dc
    simp only [initialState, newHypothesis, List.length_nil, sum_const_zero, add_zero, Nat.sub_zero]
    rw [Nat.add_mul, one_mul]
    omega
  have hb : probeBudget C (Fintype.card α) = (C.maxLeaves * X (α := α) C + 1) * B (α := α) C := by
    unfold probeBudget X B R; ring
  rw [hb]
  unfold Θ
  have := Nat.mul_le_mul_right (B (α := α) C) hΨ
  nlinarith

end Ends

/-! ## The chance of a false consistent end -/

section Chance

/-- What a step does to the clean run: a clean probe extends it or ends the round, any other
resets it. -/
def CleanOK (s : Level2State α) (cl : Bool) : Level2State α ⊕ RoundEnd (UndecidedSite α) → Prop
  | .inl s' => if cl then s'.tree = s.tree ∧ s'.edges = s.edges ∧ s'.cleanRun = s.cleanRun + 1
      else s'.cleanRun = 0
  | .inr e => e = .consistent → cl = true ∧ C.cleanProbesToEnd ≤ s.cleanRun + 1

theorem charge_clean (s : Level2State α) (k : UndecidedSite α) (zs : List (FreeMonoid α))
    (cl : Bool) :
    CleanOK C s cl (addUndecided C s k zs cl) := by
  unfold addUndecided extendCleanRun
  simp only
  split_ifs with h1 h2 h3 <;> simp_all [CleanOK]

theorem step_clean (s : Level2State α) (x : FreeMonoid α) :
    CleanOK C s (probeOrAgree read s.tree s.edges C.walkStart x).clean (applyProbe read C s x) := by
  unfold applyProbe
  generalize probeOrAgree read s.tree s.edges C.walkStart x = o
  cases o with
  | agree =>
    simp only [Outcome.clean]
    unfold extendCleanRun
    split_ifs <;> simp_all [CleanOK]
  | undecided k zs cl => exact charge_clean C s _ _ _
  | unlearnedEdge p c u t => simp [CleanOK, Outcome.clean, newHypothesis]
  | wrongEdge p c u t =>
    simp only [Outcome.clean]
    unfold recordDisagreement splitLeaf
    simp only
    split_ifs <;> (try split) <;> (try split_ifs) <;> simp_all [CleanOK, newHypothesis]

variable {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]

omit [Countable X] [MeasurableSingletonClass X] in
theorem piFinSuccAbove_zero {n : ℕ} (d : Fin (n + 1) → X) :
    MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => X) 0 d = (d 0, Fin.tail d) := by
  ext j
  · rfl
  · simp [MeasurableEquiv.piFinSuccAbove, Fin.tail]

/-- The chance of `E` over `T + 1` draws, as the first draw's average of its section. -/
theorem pi_succ_apply (D : Measure X) [IsProbabilityMeasure D] (T : ℕ)
    (E : Set (Fin (T + 1) → X)) :
    (Measure.pi fun _ : Fin (T + 1) => D) E
      = ∫⁻ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈ E} ∂D := by
  have hmp := measurePreserving_piFinSuccAbove (fun _ : Fin (T + 1) => D) 0
  set e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (T + 1) => X) 0
  have hE : E = e ⁻¹' {p | Fin.cons p.1 p.2 ∈ E} := by
    ext d
    simp only [Set.mem_preimage, Set.mem_ofPred_eq, e, piFinSuccAbove_zero, Fin.cons_self_tail]
  rw [hE, hmp.measure_preimage (Set.to_countable _).measurableSet.nullMeasurableSet,
    Measure.prod_apply (Set.to_countable _).measurableSet]
  rfl

variable [Fintype α] (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (ε : ℝ)

/-- The hypothesis of `s` disagrees on more than `ε` of the probes. -/
def Bad (s : Level2State α) : Prop :=
  ε < D.real {x | (probeOrAgree read s.tree s.edges C.walkStart x).clean = false}

open scoped Classical in
theorem bad_round (hε : ε ≤ 1) : ∀ (N : ℕ) (s : Level2State α),
    (Measure.pi fun _ : Fin N => D)
        {xs | (runProbes read C s (List.ofFn xs)).2 = .consistent
          ∧ Bad read C D ε (runProbes read C s (List.ofFn xs)).1}
      ≤ (if Bad read C D ε s then ENNReal.ofReal ((1 - ε) ^ (C.cleanProbesToEnd - s.cleanRun))
          else 0)
        + N * ENNReal.ofReal ((1 - ε) ^ C.cleanProbesToEnd) := by
  intro N
  induction N with
  | zero =>
    intro s
    have : {xs : Fin 0 → FreeMonoid α | (runProbes read C s (List.ofFn xs)).2 = .consistent
        ∧ Bad read C D ε (runProbes read C s (List.ofFn xs)).1} = ∅ := by
      ext xs; simp [runProbes]
    rw [this, measure_empty]
    exact bot_le
  | succ N ih =>
    intro s
    set δ := ENNReal.ofReal ((1 - ε) ^ C.cleanProbesToEnd)
    set A := if Bad read C D ε s
      then ENNReal.ofReal ((1 - ε) ^ (C.cleanProbesToEnd - s.cleanRun - 1)) else 0
    set S := {x : FreeMonoid α | (probeOrAgree read s.tree s.edges C.walkStart x).clean = true}
    have hpt : ∀ x, (Measure.pi fun _ : Fin N => D)
        {ys | Fin.cons x ys ∈ {xs : Fin (N + 1) → FreeMonoid α |
          (runProbes read C s (List.ofFn xs)).2 = .consistent
            ∧ Bad read C D ε (runProbes read C s (List.ofFn xs)).1}}
        ≤ S.indicator (fun _ => A) x + Sᶜ.indicator (fun _ => δ) x + N * δ := by
      intro x
      have hcl := step_clean read C s x
      simp only [Set.mem_ofPred_eq, List.ofFn_succ, Fin.cons_zero, Fin.cons_succ]
      rcases hst : applyProbe read C s x with s' | e <;> rw [hst] at hcl <;>
        simp only [runProbes, hst]
      · refine (ih s').trans ?_
        by_cases hx : (probeOrAgree read s.tree s.edges C.walkStart x).clean = true
        · rw [hx] at hcl
          obtain ⟨ht, he, hc⟩ := hcl
          have hbad : Bad read C D ε s' ↔ Bad read C D ε s := by unfold Bad; rw [ht, he]
          have hxS : x ∈ S := hx
          simp only [Set.indicator_of_mem hxS, Set.indicator_of_notMem (Set.notMem_compl_iff.2 hxS),
            add_zero, hc, A]
          rw [show C.cleanProbesToEnd - (s.cleanRun + 1) = C.cleanProbesToEnd - s.cleanRun - 1 by
            omega]
          simp only [hbad, le_refl]
        · have hx' : (probeOrAgree read s.tree s.edges C.walkStart x).clean = false := by
            simpa using hx
          rw [hx'] at hcl
          simp only [CleanOK, Bool.false_eq_true, if_false] at hcl
          have hxS : x ∈ Sᶜ := fun h => hx h
          simp only [Set.indicator_of_mem hxS, Set.indicator_of_notMem (fun h => hxS h), zero_add,
            hcl, Nat.sub_zero]
          gcongr
          split_ifs <;> simp [δ]
      · by_cases hb : e = .consistent ∧ Bad read C D ε s
        · obtain ⟨rfl, hb⟩ := hb
          obtain ⟨hx, hc⟩ := hcl rfl
          have hxS : x ∈ S := hx
          simp only [Set.indicator_of_mem hxS, Set.indicator_of_notMem (Set.notMem_compl_iff.2 hxS),
            add_zero, A, if_pos hb, show C.cleanProbesToEnd - s.cleanRun - 1 = 0 by omega, pow_zero,
            ENNReal.ofReal_one]
          calc _ ≤ (1 : ENNReal) := prob_le_one
            _ ≤ 1 + N * δ := le_self_add
        · have : {ys : Fin N → FreeMonoid α | (s, e).2 = RoundEnd.consistent
              ∧ Bad read C D ε (s, e).1} = ∅ := by
            ext ys
            simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
            exact hb
          rw [this, measure_empty]
          exact bot_le
    rw [pi_succ_apply]
    have hS : MeasurableSet S := (Set.to_countable _).measurableSet
    calc _ ≤ ∫⁻ x, S.indicator (fun _ => A) x + Sᶜ.indicator (fun _ => δ) x + N * δ ∂D :=
          lintegral_mono hpt
      _ = A * D S + δ * D Sᶜ + N * δ := by
          rw [lintegral_add_right _ measurable_const,
            lintegral_add_left (measurable_const.indicator hS), lintegral_indicator_const hS,
            lintegral_indicator_const hS.compl, lintegral_const, measure_univ, mul_one]
      _ ≤ (if Bad read C D ε s then ENNReal.ofReal ((1 - ε) ^ (C.cleanProbesToEnd - s.cleanRun))
            else 0)
          + δ + N * δ := by
          gcongr
          · unfold A
            split_ifs with hb
            · have hDS : D S ≤ ENNReal.ofReal (1 - ε) := by
                have hc : D.real S = 1 - D.real Sᶜ := by
                  rw [measureReal_compl hS, probReal_univ]; ring
                have hSc :
                    Sᶜ = {x | (probeOrAgree read s.tree s.edges C.walkStart x).clean = false} := by
                  ext x; simp [S]
                rw [← ENNReal.ofReal_toReal (measure_ne_top D S)]
                apply ENNReal.ofReal_le_ofReal
                change D.real S ≤ 1 - ε
                rw [hc, hSc]
                unfold Bad at hb
                linarith
              by_cases hcn : s.cleanRun < C.cleanProbesToEnd
              · calc ENNReal.ofReal ((1 - ε) ^ (C.cleanProbesToEnd - s.cleanRun - 1)) * D S
                    ≤ ENNReal.ofReal ((1 - ε) ^ (C.cleanProbesToEnd - s.cleanRun - 1))
                      * ENNReal.ofReal (1 - ε) := by
                      gcongr
                  _ = ENNReal.ofReal ((1 - ε) ^ (C.cleanProbesToEnd - s.cleanRun)) := by
                      rw [← ENNReal.ofReal_mul (pow_nonneg (by linarith) _), ← pow_succ]
                      congr 2
                      omega
              · rw [show C.cleanProbesToEnd - s.cleanRun - 1 = 0 by omega,
                  show C.cleanProbesToEnd - s.cleanRun = 0 by omega]
                calc _ ≤ ENNReal.ofReal ((1 - ε) ^ 0) * 1 := by gcongr; exact prob_le_one
                  _ = _ := mul_one _
            · simp
          · calc δ * D Sᶜ ≤ δ * 1 := by gcongr; exact prob_le_one
              _ = δ := mul_one _
      _ = _ := by push_cast; ring

end Chance

/-! ## The chance of a thin harvest -/

section Thin

variable [Fintype α] {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}
  (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (y : ℝ)

/-- The sites a probe can be charged at under a tree. -/
def sites (T : DTree α) : Finset (UndecidedSite α) :=
  {.start, .stop} ∪ (T.leaves.toFinset ×ˢ (Finset.univ : Finset α)).image fun e => .edge e.1 e.2

theorem mem_sites {T : DTree α} {k : UndecidedSite α} (h : SiteOK T k) :
    k ∈ sites T := by
  cases k with
  | start => simp [sites]
  | stop => simp [sites]
  | edge p c =>
    simp only [sites, mem_union, mem_image, mem_product, List.mem_toFinset, mem_univ, and_true]
    exact Or.inr ⟨(p, c), h, rfl⟩

theorem sites_card (T : DTree α) : (sites T).card ≤ 2 + T.leaves.length * Fintype.card α := by
  calc (sites T).card
      ≤ ({.start, .stop} : Finset (UndecidedSite α)).card
        + ((T.leaves.toFinset ×ˢ (univ : Finset α)).image
          fun e => (UndecidedSite.edge e.1 e.2 : UndecidedSite α)).card := card_union_le _ _
    _ ≤ 2 + (T.leaves.toFinset ×ˢ (univ : Finset α)).card := add_le_add card_le_two card_image_le
    _ ≤ 2 + T.leaves.length * Fintype.card α := by
        rw [card_product, card_univ]
        exact Nat.add_le_add_left (Nat.mul_le_mul_right _ (List.toFinset_card_le _)) _

omit [Fintype α] in
/-- A step from `s` that ends the round in a harvest is charged a string at the harvested site. -/
theorem harvest_charged [Countable α] {s : Level2State α} (hs : Inv read C s) {x : FreeMonoid α}
    {site : UndecidedSite α} (h : applyProbe read C s x = .inr (.harvest site)) :
    SiteOK s.tree site ∧ siteHarvest read ⟨s.tree, s.edges⟩ C.walkStart site x ≠ [] := by
  have hOK := probeOrAgree_ok read s.tree s.edges C.walkStart hs.edges x
  unfold applyProbe at h
  unfold siteHarvest
  generalize probeOrAgree read s.tree s.edges C.walkStart x = o at hOK h ⊢
  cases o with
  | agree =>
    unfold extendCleanRun at h
    split_ifs at h
    simp at h
  | undecided k zs cl =>
    simp only at h ⊢
    unfold addUndecided at h
    simp only [Function.update_self] at h
    split_ifs at h with hh
    · simp only [Sum.inr.injEq, RoundEnd.harvest.injEq] at h
      subst h
      refine ⟨hOK.1, ?_⟩
      rw [if_pos rfl]
      intro h0
      rw [List.dedup_eq_nil] at h0
      have := hs.und_lt k
      simp only [h0, List.append_nil] at hh
      omega
    · unfold extendCleanRun at h
      split_ifs at h
      simp at h
  | wrongEdge p c u t =>
    simp only at h
    unfold recordDisagreement splitLeaf at h
    simp only at h
    repeat' (first | split_ifs at h | split at h)
    all_goals simp at h
  | unlearnedEdge p c u t => simp at h

/-- A step from `s` ends the round in a harvest whose sampler yields at most `y`. -/
def ThinStep (s : Level2State α) : Set (FreeMonoid α) :=
  {x | ∃ site, applyProbe read C s x = .inr (.harvest site)
    ∧ (harvestLaw read D C.walkStart ⟨s.tree, s.edges⟩ site).real Set.univ ≤ y}

theorem thinStep_le (hy : 0 ≤ y) {s : Level2State α} (hs : Inv read C s) :
    D (ThinStep read C D y s)
      ≤ ENNReal.ofReal ((2 + s.tree.leaves.length * Fintype.card α) * y) := by
  classical
  set H : Hypothesis α := ⟨s.tree, s.edges⟩
  set F := (sites s.tree).filter
    fun site => (harvestLaw read D C.walkStart H site).real Set.univ ≤ y
  have hsub : ThinStep read C D y s
      ⊆ ⋃ site ∈ F, {x | siteHarvest read H C.walkStart site x ≠ []} := by
    intro x hx
    obtain ⟨site, h, hy'⟩ := Set.mem_ofPred_eq ▸ hx
    obtain ⟨hk, hne⟩ := harvest_charged read C hs h
    exact Set.mem_biUnion (mem_filter.2 ⟨mem_sites hk, hy'⟩) hne
  have hone : ∀ site ∈ F, D {x | siteHarvest read H C.walkStart site x ≠ []}
      ≤ ENNReal.ofReal y := by
    intro site hsite
    have hfin : harvestLaw read D C.walkStart H site Set.univ ≠ ⊤ :=
      ne_top_of_le_ne_top (by simp) (harvestLaw_univ_le read D H C.walkStart site)
    exact (charged_le_harvestLaw read D H C.walkStart site).trans
      ((ENNReal.le_ofReal_iff_toReal_le hfin hy).2 (mem_filter.1 hsite).2)
  have hF : (F.card : ℝ) ≤ 2 + s.tree.leaves.length * Fintype.card α := by
    exact_mod_cast (card_filter_le _ _).trans (sites_card s.tree)
  calc D (ThinStep read C D y s)
      ≤ D (⋃ site ∈ F, {x | siteHarvest read H C.walkStart site x ≠ []}) := measure_mono hsub
    _ ≤ ∑ site ∈ F, D {x | siteHarvest read H C.walkStart site x ≠ []} :=
        measure_biUnion_finset_le _ _
    _ ≤ ∑ _site ∈ F, ENNReal.ofReal y := sum_le_sum hone
    _ = ENNReal.ofReal (F.card * y) := by
        rw [sum_const, nsmul_eq_mul, ENNReal.ofReal_mul (Nat.cast_nonneg _), ENNReal.ofReal_natCast]
    _ ≤ _ := ENNReal.ofReal_le_ofReal (mul_le_mul_of_nonneg_right hF hy)

/-- The run ends in a harvest whose sampler yields at most `y`. -/
def Thin (r : Level2State α × RoundEnd (UndecidedSite α)) : Prop :=
  ∃ site, r.2 = .harvest site
    ∧ (harvestLaw read D C.walkStart ⟨r.1.tree, r.1.edges⟩ site).real Set.univ ≤ y

/-- Each step ends the round in a thin harvest with chance at most `(2 + (|Q| + 2)|Σ|) y`. -/
theorem thin_round (hI : IdealReads read M side) (hcap : Fintype.card σ + 2 ≤ C.maxLeaves)
    (hm : 1 ≤ C.recordsToMove) (hh : 1 ≤ C.harvestSize) (hn : 1 ≤ C.cleanProbesToEnd)
    (hy : 0 ≤ y) : ∀ (N : ℕ) (s : Level2State α), Inv read C s → Reached read s.tree →
      (Measure.pi fun _ : Fin N => D)
          {xs | Thin read C D y (runProbes read C s (List.ofFn xs))}
        ≤ N * ENNReal.ofReal ((2 + (Fintype.card σ + 2) * Fintype.card α) * y) := by
  intro N
  induction N with
  | zero =>
    intro s _ _
    have : {xs : Fin 0 → FreeMonoid α | Thin read C D y (runProbes read C s (List.ofFn xs))}
        = ∅ := by
      ext xs; simp [runProbes, Thin]
    rw [this, measure_empty]
    exact bot_le
  | succ N ih =>
    intro s hs hr
    set c := ENNReal.ofReal ((2 + (Fintype.card σ + 2) * Fintype.card α) * y)
    have hpt : ∀ x, (Measure.pi fun _ : Fin N => D)
        {ys | Fin.cons x ys ∈ {xs : Fin (N + 1) → FreeMonoid α |
          Thin read C D y (runProbes read C s (List.ofFn xs))}}
        ≤ (ThinStep read C D y s).indicator (fun _ => 1) x + N * c := by
      intro x
      simp only [Set.mem_ofPred_eq, List.ofFn_succ, Fin.cons_zero, Fin.cons_succ]
      have hst := step_ok read C hI hcap hm hh hn hs hr x
      rcases hstep : applyProbe read C s x with s' | e <;> rw [hstep] at hst <;>
        simp only [runProbes, hstep]
      · exact (ih s' hst.1 hst.2.1).trans le_add_self
      · by_cases hx : x ∈ ThinStep read C D y s
        · rw [Set.indicator_of_mem hx]
          exact prob_le_one.trans le_self_add
        · have : {ys : Fin N → FreeMonoid α | Thin read C D y (s, e)} = ∅ := by
            ext ys
            simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
            rintro ⟨site, hse, hle⟩
            exact hx ⟨site, hstep.trans (congrArg _ hse), hle⟩
          rw [this, measure_empty]
          exact bot_le
    have hS : MeasurableSet (ThinStep read C D y s) := (Set.to_countable _).measurableSet
    have hleaves : (s.tree.leaves.length : ℝ) ≤ Fintype.card σ + 2 := by
      exact_mod_cast leaves_le_of_reached read hI hr
    rw [pi_succ_apply]
    calc _ ≤ ∫⁻ x, (ThinStep read C D y s).indicator (fun _ => 1) x + N * c ∂D :=
          lintegral_mono hpt
      _ = D (ThinStep read C D y s) + N * c := by
          rw [lintegral_add_right _ measurable_const, lintegral_indicator_const hS, one_mul,
            lintegral_const, measure_univ, mul_one]
      _ ≤ c + N * c := by
          gcongr
          refine (thinStep_le read C D y hy hs).trans (ENNReal.ofReal_le_ofReal ?_)
          gcongr
      _ = (N + 1 : ℕ) * c := by push_cast; ring

end Thin

/-! ## The theorem -/

theorem idealRoundCorrect_holds : IdealRoundCorrect := by
  intro α _ _ σ _ M read side C P hI hm hh hn hcap hP
  have h2 : 2 ≤ C.maxLeaves := by omega
  have hΘ : Θ C (initialState : Level2State α) < P := (Θ_start C hn h2).trans_le hP
  intro D _ ε y hε hy
  have hsub : {draws : Fin P → FreeMonoid α | let result := indecisiveLStarLevel2 C read draws
        ¬ EndsWell M (AllUndecided read M) D ε
          {x | Disagrees read result.hypothesis C.walkStart x}
          C.walkStart (harvestLaw read D C.walkStart result.hypothesis) y
          result.ending}
      ⊆ {xs | (runProbes read C initialState (List.ofFn xs)).2 = .consistent
          ∧ Bad read C D ε (runProbes read C initialState (List.ofFn xs)).1}
        ∪ {xs | Thin read C D y (runProbes read C initialState (List.ofFn xs))} := by
    intro xs hxs
    simp only [Set.mem_ofPred_eq, indecisiveLStarLevel2] at hxs
    obtain ⟨hok, hinv, -⟩ := round_ok read C hI hcap hm hh hn (List.ofFn xs) initialState
      (inv_start read C hm hh hn h2) (reached_start read) (by simpa using hΘ)
    simp only [Set.mem_union, Set.mem_ofPred_eq]
    generalize runProbes read C initialState (List.ofFn xs) = r at hok hinv hxs ⊢
    rcases r with ⟨s, _ | site | _⟩
    · refine Or.inl ⟨rfl, lt_of_lt_of_le (not_le.1 hxs) (measureReal_mono fun x hx => ?_)⟩
      exact probeOrAgree_disagrees read _ _ _ hx
    · refine Or.inr ⟨site, rfl, not_lt.1 fun hlt => hxs ?_⟩
      exact endsWell_harvest read D ⟨s.tree, s.edges⟩ C.walkStart site hI hinv.edges ε _ hy hlt
    · exact hok.elim
  have hb := bad_round read C D ε hε P initialState
  have ht := thin_round read C D y hI hcap hm hh hn hy P initialState
    (inv_start read C hm hh hn h2) (reached_start read)
  have h1ε : 0 ≤ 1 - ε := by linarith
  have hB : 0 ≤ (P : ℝ) * (2 + (Fintype.card σ + 2) * Fintype.card α) * y := by positivity
  refine (measureReal_mono hsub).trans (ENNReal.toReal_le_of_le_ofReal
    (add_nonneg (mul_nonneg (by positivity) (pow_nonneg h1ε _)) hB)
    ((measure_union_le _ _).trans ?_))
  rw [ENNReal.ofReal_add (mul_nonneg (by positivity) (pow_nonneg h1ε _)) hB]
  gcongr
  · refine hb.trans ?_
    rw [ENNReal.ofReal_mul (by positivity)]
    have : ENNReal.ofReal ((P : ℝ) + 1) = (P : ENNReal) + 1 := by
      rw [ENNReal.ofReal_add (by positivity) zero_le_one, ENNReal.ofReal_natCast,
        ENNReal.ofReal_one]
    rw [this, add_mul, one_mul, add_comm]
    gcongr
    split_ifs
    · simp [initialState, newHypothesis]
    · exact bot_le
  · refine ht.trans (le_of_eq ?_)
    rw [show (P : ℝ) * (2 + (Fintype.card σ + 2) * Fintype.card α) * y
        = P * ((2 + (Fintype.card σ + 2) * Fintype.card α) * y) by ring,
      ENNReal.ofReal_mul (Nat.cast_nonneg P), ENNReal.ofReal_natCast]

end Ideal

end OrthoDFA
