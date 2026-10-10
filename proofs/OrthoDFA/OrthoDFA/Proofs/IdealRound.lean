import OrthoDFA.Proofs.IdealSteps
import Mathlib.Probability.ProductMeasure

/-!
# The idealized round: the proofs

Every step that does not end the round lowers `Θ`, and the start's `Θ` is below the budget, so the
round ends on its probes, and `step_ok` says how.

The chance part: the hypothesis is fixed while a run of clean probes grows, and the probes are
fresh, so a hypothesis whose probes disagree more than `ε` of the time completes a run of `n` with
chance at most `(1 − ε)^n`. Each probe that does not continue a run starts at most one new one.
-/

namespace OrthoDFA

namespace Ideal

open MeasureTheory Finset

variable {α : Type*} [DecidableEq α] (read : FreeMonoid α → ARU) (C : RoundCfg)

/-! ## The round ends -/

section Ends

variable [Fintype α] {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}

theorem round_ok (hI : IdealReads read M side) (hcap : Fintype.card σ + 2 ≤ C.Lmax)
    (hm : 1 ≤ C.m) (hh : 1 ≤ C.h) (hn : 1 ≤ C.n) :
    ∀ (xs : List (FreeMonoid α)) (s : RState α), Inv read C s → Reached read s.tree →
      Θ C s < xs.length → ∃ e, (round read C s xs).2 = some e ∧ EndOK read e ∧
        Inv read C (round read C s xs).1 ∧ Reached read (round read C s xs).1.tree
  | [], s, _, _, h => by simp at h
  | x :: xs, s, hs, hr, h => by
    have hst := step_ok read C hI hcap hm hh hn hs hr x
    rcases hstep : step read C s x with s' | e <;> rw [hstep] at hst <;> simp only [round, hstep]
    · obtain ⟨hs', hr', hlt⟩ := hst
      exact round_ok hI hcap hm hh hn xs s' hs' hr' (by simp only [List.length_cons] at h; omega)
    · exact ⟨e, rfl, hst, hs, hr⟩

omit [DecidableEq α] [Fintype α] in
theorem leaves_start : (start : RState α).tree.leaves = [[false], [true]] := by
  simp [start, fresh, DTree.leaves]

omit [Fintype α] in
theorem inv_start (hm : 1 ≤ C.m) (hh : 1 ≤ C.h) (hn : 1 ≤ C.n) (h2 : 2 ≤ C.Lmax) :
    Inv read C (start : RState α) :=
  inv_fresh read C hm hh hn (fun _ _ _ _ _ h => by simp at h) (by simp [DTree.leaves]; omega)

omit [DecidableEq α] [Fintype α] in
theorem reached_start : Reached read (start : RState α).tree := by
  intro p hp
  rw [leaves_start] at hp
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl
  · exact Or.inl rfl
  · exact Or.inr (Or.inl rfl)

theorem Θ_start (hn : 1 ≤ C.n) (h2 : 2 ≤ C.Lmax) :
    Θ C (start : RState α) < budget C (Fintype.card α) := by
  have hΦ := Φ_le (start : RState α)
  rw [leaves_start] at hΦ
  have hΨ : Ψ C (start : RState α) ≤ C.Lmax * X (α := α) C := by
    unfold Ψ
    rw [leaves_start]
    have hL : C.Lmax * X (α := α) C = (C.Lmax - 2) * X (α := α) C + 2 * X (α := α) C := by
      rw [← Nat.add_mul, Nat.sub_add_cancel h2]
    have hX : 4 * Fintype.card α ≤ 2 * X (α := α) C := by
      unfold X
      have := Nat.mul_le_mul_right (Fintype.card α) h2
      nlinarith
    simp only [List.length_cons, List.length_nil, zero_add, Nat.reduceAdd] at hΦ ⊢
    omega
  have hρ : ρ C (start : RState α) + 1 ≤ B (α := α) C := by
    unfold ρ B Dc
    simp only [start, fresh, List.length_nil, sum_const_zero, add_zero, Nat.sub_zero]
    rw [Nat.add_mul, one_mul]
    omega
  have hb : budget C (Fintype.card α) = (C.Lmax * X (α := α) C + 1) * B (α := α) C := by
    unfold budget X B R; ring
  rw [hb]
  unfold Θ
  have := Nat.mul_le_mul_right (B (α := α) C) hΨ
  nlinarith

/-- The round ends consistent or with a nonempty harvest of undecided states, and every leaf
past the first two is reached by some string. -/
def Sound {σ : Type*} [Fintype σ] (M : DFA α σ) (r : RState α × Option (REnd α)) : Prop :=
  (r.2 = some .consistent ∨ ∃ zs, r.2 = some (.harvest zs) ∧ zs ≠ [] ∧
      ∀ z ∈ zs, ∀ w : FreeMonoid α, M.eval w.toList = M.eval z.toList → read w = .undecided)
  ∧ (∀ p ∈ r.1.tree.leaves, p = [false] ∨ p = [true] ∨ ∃ w, r.1.tree.sift read w = .inl p)
  ∧ r.1.tree.leaves.length ≤ Fintype.card σ + 2

omit [DecidableEq α] [Fintype α] in
theorem sound_of (hI : IdealReads read M side) {r : RState α × Option (REnd α)} {e : REnd α}
    (he : r.2 = some e) (hok : EndOK read e) (hr : Reached read r.1.tree) : Sound read M r := by
  refine ⟨?_, hr, leaves_le_of_reached read hI hr⟩
  cases e with
  | consistent => exact Or.inl he
  | harvest zs =>
    obtain ⟨hne, hu⟩ := hok
    refine Or.inr ⟨zs, he, hne, fun z hz w hw => ?_⟩
    rcases hI (M.eval z.toList) with h | h
    · exact h w hw
    · have := h z rfl
      rw [hu z hz] at this
      split_ifs at this
  | tooBig => exact hok.elim

end Ends

/-! ## The chance of a false consistent end -/

section Chance

/-- What a step does to the clean run: a clean probe extends it or ends the round, any other
resets it. -/
def CleanOK (s : RState α) (cl : Bool) : RState α ⊕ REnd α → Prop
  | .inl s' => if cl then s'.tree = s.tree ∧ s'.edges = s.edges ∧ s'.clean = s.clean + 1
      else s'.clean = 0
  | .inr e => e = .consistent → cl = true ∧ C.n ≤ s.clean + 1

theorem charge_clean (s : RState α) (k : Cls α) (zs : List (FreeMonoid α)) (cl : Bool) :
    CleanOK C s cl (charge C s k zs cl) := by
  unfold charge tick
  simp only
  split_ifs with h1 h2 h3 <;> simp_all [CleanOK]

theorem step_clean (s : RState α) (x : FreeMonoid α) :
    CleanOK C s (probe read s.tree s.edges C.k x).clean (step read C s x) := by
  unfold step
  generalize probe read s.tree s.edges C.k x = o
  cases o with
  | agree =>
    simp only [Outcome.clean]
    unfold tick
    split_ifs <;> simp_all [CleanOK]
  | startU z => exact charge_clean C s _ _ _
  | endU z => exact charge_clean C s _ _ _
  | triple k zs => exact charge_clean C s _ _ _
  | pair k zs => exact charge_clean C s _ _ _
  | member p c u t => simp [CleanOK, Outcome.clean, fresh]
  | edge p c u t =>
    simp only [Outcome.clean]
    unfold record split
    simp only
    split_ifs <;> (try split) <;> (try split_ifs) <;> simp_all [CleanOK, fresh]

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
def Bad (s : RState α) : Prop :=
  ε < D.real {x | (probe read s.tree s.edges C.k x).clean = false}

open scoped Classical in
theorem bad_round (hε : ε ≤ 1) : ∀ (N : ℕ) (s : RState α),
    (Measure.pi fun _ : Fin N => D)
        {xs | (round read C s (List.ofFn xs)).2 = some .consistent
          ∧ Bad read C D ε (round read C s (List.ofFn xs)).1}
      ≤ (if Bad read C D ε s then ENNReal.ofReal ((1 - ε) ^ (C.n - s.clean)) else 0)
        + N * ENNReal.ofReal ((1 - ε) ^ C.n) := by
  intro N
  induction N with
  | zero =>
    intro s
    have : {xs : Fin 0 → FreeMonoid α | (round read C s (List.ofFn xs)).2 = some .consistent
        ∧ Bad read C D ε (round read C s (List.ofFn xs)).1} = ∅ := by
      ext xs; simp [round]
    rw [this, measure_empty]
    exact bot_le
  | succ N ih =>
    intro s
    set δ := ENNReal.ofReal ((1 - ε) ^ C.n)
    set A := if Bad read C D ε s then ENNReal.ofReal ((1 - ε) ^ (C.n - s.clean - 1)) else 0
    set S := {x : FreeMonoid α | (probe read s.tree s.edges C.k x).clean = true}
    have hpt : ∀ x, (Measure.pi fun _ : Fin N => D)
        {ys | Fin.cons x ys ∈ {xs : Fin (N + 1) → FreeMonoid α |
          (round read C s (List.ofFn xs)).2 = some .consistent
            ∧ Bad read C D ε (round read C s (List.ofFn xs)).1}}
        ≤ S.indicator (fun _ => A) x + Sᶜ.indicator (fun _ => δ) x + N * δ := by
      intro x
      have hcl := step_clean read C s x
      simp only [Set.mem_ofPred_eq, List.ofFn_succ, Fin.cons_zero, Fin.cons_succ]
      rcases hst : step read C s x with s' | e <;> rw [hst] at hcl <;> simp only [round, hst]
      · refine (ih s').trans ?_
        by_cases hx : (probe read s.tree s.edges C.k x).clean = true
        · rw [hx] at hcl
          obtain ⟨ht, he, hc⟩ := hcl
          have hbad : Bad read C D ε s' ↔ Bad read C D ε s := by unfold Bad; rw [ht, he]
          have hxS : x ∈ S := hx
          simp only [Set.indicator_of_mem hxS, Set.indicator_of_notMem (Set.notMem_compl_iff.2 hxS),
            add_zero, hc, A]
          rw [show C.n - (s.clean + 1) = C.n - s.clean - 1 by omega]
          simp only [hbad, le_refl]
        · have hx' : (probe read s.tree s.edges C.k x).clean = false := by simpa using hx
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
            add_zero, A, if_pos hb, show C.n - s.clean - 1 = 0 by omega, pow_zero,
            ENNReal.ofReal_one]
          calc _ ≤ (1 : ENNReal) := prob_le_one
            _ ≤ 1 + N * δ := le_self_add
        · have : {ys : Fin N → FreeMonoid α | (s, some e).2 = some REnd.consistent
              ∧ Bad read C D ε (s, some e).1} = ∅ := by
            ext ys
            simp only [Set.mem_ofPred_eq, Option.some.injEq, Set.mem_empty_iff_false, iff_false]
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
      _ ≤ (if Bad read C D ε s then ENNReal.ofReal ((1 - ε) ^ (C.n - s.clean)) else 0)
          + δ + N * δ := by
          gcongr
          · unfold A
            split_ifs with hb
            · have hDS : D S ≤ ENNReal.ofReal (1 - ε) := by
                have hc : D.real S = 1 - D.real Sᶜ := by
                  rw [measureReal_compl hS, probReal_univ]; ring
                have hSc : Sᶜ = {x | (probe read s.tree s.edges C.k x).clean = false} := by
                  ext x; simp [S]
                rw [← ENNReal.ofReal_toReal (measure_ne_top D S)]
                apply ENNReal.ofReal_le_ofReal
                change D.real S ≤ 1 - ε
                rw [hc, hSc]
                unfold Bad at hb
                linarith
              by_cases hcn : s.clean < C.n
              · calc ENNReal.ofReal ((1 - ε) ^ (C.n - s.clean - 1)) * D S
                    ≤ ENNReal.ofReal ((1 - ε) ^ (C.n - s.clean - 1)) * ENNReal.ofReal (1 - ε) := by
                      gcongr
                  _ = ENNReal.ofReal ((1 - ε) ^ (C.n - s.clean)) := by
                      rw [← ENNReal.ofReal_mul (pow_nonneg (by linarith) _), ← pow_succ]
                      congr 2
                      omega
              · rw [show C.n - s.clean - 1 = 0 by omega, show C.n - s.clean = 0 by omega]
                calc _ ≤ ENNReal.ofReal ((1 - ε) ^ 0) * 1 := by gcongr; exact prob_le_one
                  _ = _ := mul_one _
            · simp
          · calc δ * D Sᶜ ≤ δ * 1 := by gcongr; exact prob_le_one
              _ = δ := mul_one _
      _ = _ := by push_cast; ring

end Chance

/-! ## The theorem -/

theorem idealRoundCorrect_holds : IdealRoundCorrect := by
  intro α _ _ σ _ M read side C P hI hm hh hn hcap hP
  have h2 : 2 ≤ C.Lmax := by omega
  have hΘ : Θ C (start : RState α) < P := (Θ_start C hn h2).trans_le hP
  intro D _ ε hε
  have hsub : {xs : Fin P → FreeMonoid α | let r := round read C start (List.ofFn xs)
        ¬ EndsWell M (AllUndecided read M) D ε {x | Disagrees read r.1.tree r.1.edges C.k x}
          (REnd.toRoundEnd r.2)}
      ⊆ {xs | (round read C start (List.ofFn xs)).2 = some .consistent
          ∧ Bad read C D ε (round read C start (List.ofFn xs)).1} := by
    intro xs hxs
    simp only [Set.mem_setOf_eq] at hxs
    obtain ⟨e, he, hok, -, hr⟩ := round_ok read C hI hcap hm hh hn (List.ofFn xs) start
      (inv_start read C hm hh hn h2) (reached_start read) (by simpa using hΘ)
    rcases (sound_of read hI he hok hr).1 with hc | hv
    · rw [hc] at hxs
      refine ⟨hc, lt_of_lt_of_le (not_le.1 hxs) (measureReal_mono fun x hx => ?_)⟩
      exact probe_disagrees read _ _ _ hx
    · obtain ⟨zs, hz, hne, hu⟩ := hv
      rw [hz] at hxs
      refine absurd ?_ hxs
      classical
      show zs.length < 2 * (zs.filter fun z => AllUndecided read M (M.eval z.toList)).length
      have hall : ∀ z ∈ zs, decide (AllUndecided read M (M.eval (FreeMonoid.toList z))) = true :=
        fun z hz' => decide_eq_true (hu z hz')
      rw [List.filter_eq_self.2 hall]
      have := List.length_pos_of_ne_nil hne
      omega
  have hb := bad_round read C D ε hε P start
  have h1ε : 0 ≤ 1 - ε := by linarith
  refine (measureReal_mono hsub).trans (ENNReal.toReal_le_of_le_ofReal
    (mul_nonneg (by positivity) (pow_nonneg h1ε _)) (hb.trans ?_))
  rw [ENNReal.ofReal_mul (by positivity)]
  have : ENNReal.ofReal ((P : ℝ) + 1) = (P : ENNReal) + 1 := by
    rw [ENNReal.ofReal_add (by positivity) zero_le_one, ENNReal.ofReal_natCast,
      ENNReal.ofReal_one]
  rw [this, add_mul, one_mul, add_comm]
  gcongr
  split_ifs
  · simp [start, fresh]
  · exact bot_le

end Ideal

end OrthoDFA
