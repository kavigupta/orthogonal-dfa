import OrthoDFA.Level3Check
import OrthoDFA.Proofs.EpsRound

/-!
# The checked round's steps

A fix is level 3's change for a probe walked from `0` (`fixOf_apply`), so where its probe and its
split read nothing wrong it keeps level 3's invariants and lowers `Ψr` as a probe's change does
(`fix_keeps`). A step of the checked round is level 3's step on its probe, except where that would
end the round consistent, where it is the check: a pass, a fix, or a failure (`applyCheck_inl`).
-/

namespace OrthoDFA

namespace Random

/-- Level 3's settings walking from `0`. -/
def Level3Settings.atZero (C : Level3Settings) : Level3Settings := { C with walkStart := 0 }

variable {α : Type*} [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)
  (K : CheckSettings)

theorem fixOf_apply {s s' : Level3State α} {x : FreeMonoid α} (h : fixOf read s x = some s') :
    applyProbe read C.atZero s x = .inl s' := by
  have hp : probeOrAgree read s.tree s.edges C.atZero.walkStart x = probe read s.tree s.edges 0 x :=
    by simp [probeOrAgree, Level3Settings.atZero]
  unfold fixOf at h
  unfold applyProbe
  rw [hp]
  generalize probe read s.tree s.edges 0 x = o at h
  cases o with
  | unlearnedEdge p c u t =>
    simp only [Option.some.injEq] at h
    simp only [h]
  | wrongEdge p c u t =>
    simp only at h ⊢
    rcases hE : s.edges (p, c) with _ | ⟨t₀, w₀⟩ <;> rw [hE] at h
    · simp at h
    · simp only [Option.some.injEq] at h
      simp only [h]
  | _ => simp at h

theorem firstFix_mem {s s' : Level3State α} :
    ∀ {xs : List (FreeMonoid α)}, firstFix read s xs = some s' → ∃ x ∈ xs, fixOf read s x = some s'
  | [], h => by simp [firstFix] at h
  | x :: xs, h => by
    simp only [firstFix] at h
    rcases hf : fixOf read s x with _ | t
    · rw [hf] at h
      obtain ⟨y, hy, h'⟩ := firstFix_mem h
      exact ⟨y, List.mem_cons_of_mem _ hy, h'⟩
    · rw [hf] at h
      simp only [Option.elim, Option.some.injEq] at h
      subst h
      exact ⟨x, List.mem_cons_self, hf⟩

theorem checkStep_inl_fix {s s' : Level3State α} {b : Fin K.size → FreeMonoid α}
    (h : checkStep read K s b = .inl s') : ∃ i, fixOf read s (b i) = some s' := by
  classical
  unfold checkStep at h
  split_ifs at h
  rcases hf : firstFix read s (List.ofFn b) with _ | t <;> rw [hf] at h
  · simp at h
  · simp only [Sum.inl.injEq] at h
    subst h
    obtain ⟨x, hx, h⟩ := firstFix_mem read hf
    obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
    exact ⟨i, h⟩

theorem checkStep_inr {s : Level3State α} {b : Fin K.size → FreeMonoid α} {e : RoundEnd Level3Site}
    (h : checkStep read K s b = .inr e) :
    (e = .consistent ∧ checkPasses read K s.hypotheses s.tree s.edges (List.ofFn b))
      ∨ (e = .failed ∧ firstFix read s (List.ofFn b) = none) := by
  classical
  unfold checkStep at h
  split_ifs at h with hp
  · simp only [Sum.inr.injEq] at h
    exact Or.inl ⟨h.symm, hp⟩
  · rcases hf : firstFix read s (List.ofFn b) with _ | t <;> rw [hf] at h
    · simp only [Sum.inr.injEq] at h
      exact Or.inr ⟨h.symm, rfl⟩
    · simp at h

/-- A step of the checked round: level 3's step, or the check where level 3 would end it
consistent. -/
theorem applyCheck_cases (s : Level3State α) (d : CheckDraw α K) :
    (applyCheck read C K s d = applyProbe read C s d.1
      ∧ applyProbe read C s d.1 ≠ .inr .consistent)
    ∨ (applyProbe read C s d.1 = .inr .consistent
      ∧ applyCheck read C K s d = checkStep read K s d.2) := by
  unfold applyCheck
  rcases ha : applyProbe read C s d.1 with t | e
  · exact Or.inl ⟨rfl, by simp⟩
  · rcases e with _ | e | _
    · exact Or.inr ⟨rfl, rfl⟩
    · exact Or.inl ⟨rfl, by simp⟩
    · exact Or.inl ⟨rfl, by simp⟩

variable [Fintype α]

include C in
theorem fixOf_not_same {s s' : Level3State α} (hs : Inv read s) {x : FreeMonoid α}
    (h : fixOf read s x = some s') : ¬ Same s s' := by
  rintro ⟨ht, he⟩
  have ha := fixOf_apply read C h
  rcases step_cases read C.atZero hs ha with
    ⟨p, c, u, t, -, hE, -, -, -, rfl⟩ | ⟨p, c, u, t, t₀, w₀, -, hE, hne, hp, -, -, rfl⟩ |
    ⟨rfl, -, hch⟩
  · have := congrFun he (p, c)
    simp only [newHypothesis, Function.update_self, hE] at this
    exact absurd this (by simp)
  · have := congrArg List.length (congrArg DTree.leaves ht)
    simp only [splitLeaf, newHypothesis] at this
    rw [DTree.length_leaves_splitAt _ _ _ hp] at this
    omega
  · unfold fixOf at h
    have hp : probeOrAgree read s.tree s.edges C.atZero.walkStart x = probe read s.tree s.edges 0 x
      := by simp [probeOrAgree, Level3Settings.atZero]
    rw [hp] at hch
    generalize probe read s.tree s.edges 0 x = o at h hch
    cases o <;> simp_all [changes]

/-- A step that keeps the hypothesis is level 3's. -/
theorem applyCheck_same {s s' : Level3State α} (hs : Inv read s) {d : CheckDraw α K}
    (h : applyCheck read C K s d = .inl s') (hsm : Same s s') :
    applyProbe read C s d.1 = .inl s' := by
  rcases applyCheck_cases read C K s d with ⟨he, -⟩ | ⟨-, he⟩
  · rw [← he]; exact h
  · rw [he] at h
    obtain ⟨i, hi⟩ := checkStep_inl_fix read K h
    exact absurd hsm (fixOf_not_same read C hs hi)

variable {σ : Type*} [Fintype σ] (M : DFA α σ) (side : σ → Bool)

/-- A fix reading nothing wrong keeps the invariants and lowers `Ψr`. -/
theorem fix_keeps {s s' : Level3State α} (hs : Inv read s) (hg : Eps.WitInv M side s)
    {x : FreeMonoid α} (hx : ¬ Eps.WrongStep M side read C.atZero s x)
    (h : fixOf read s x = some s') :
    Inv read s' ∧ Eps.WitInv M side s' ∧ s' = newHypothesis s'.tree s'.edges s'.hypotheses
      ∧ s'.hypotheses = s.hypotheses + 1 ∧ Ψr (Fintype.card σ) s' < Ψr (Fintype.card σ) s :=
  Eps.eps_step_change M side read C.atZero hs hg hx (fixOf_apply read C h)
    (fixOf_not_same read C hs h)

/-- The check reads a string wrong: one some check string's probe from `0` reads, or its split
re-sifts. -/
def CheckWrong (s : Level3State α) (b : Fin K.size → FreeMonoid α) : Prop :=
  ∃ i, Eps.WrongStep M side read C.atZero s (b i)

/-- What the checked round's step must avoid: a wrong read of its probe's step, or, where the
check runs, a wrong read of the check or its failing. -/
def CheckHazard (W : Level3State α → FreeMonoid α → Prop) (s : Level3State α) (d : CheckDraw α K) :
    Prop :=
  W s d.1 ∨ (applyProbe read C s d.1 = .inr .consistent
    ∧ (CheckWrong read C K M side s d.2 ∨ checkStep read K s d.2 = .inr .failed))

/-- A step changing the hypothesis that avoids the hazard keeps the invariants. -/
theorem check_keeps (W : Level3State α → FreeMonoid α → Prop) (hW : Eps.KeepsInv read C M side W)
    {s s' : Level3State α} {d : CheckDraw α K} (hs : Inv read s) (hg : Eps.WitInv M side s)
    (hx : ¬ CheckHazard read C K M side W s d) (h : applyCheck read C K s d = .inl s')
    (hne : ¬ (s'.tree = s.tree ∧ s'.edges = s.edges)) :
    Inv read s' ∧ Eps.WitInv M side s' ∧ s' = newHypothesis s'.tree s'.edges s'.hypotheses
      ∧ s'.hypotheses = s.hypotheses + 1 ∧ Ψr (Fintype.card σ) s' < Ψr (Fintype.card σ) s := by
  rcases applyCheck_cases read C K s d with ⟨he, -⟩ | ⟨hf, he⟩
  · rw [he] at h
    exact hW hs hg (fun hw => hx (Or.inl hw)) h hne
  · rw [he] at h
    obtain ⟨i, hi⟩ := checkStep_inl_fix read K h
    exact fix_keeps read C M side hs hg (fun hw => hx (Or.inr ⟨hf, Or.inl ⟨i, hw⟩⟩)) hi

end Random

end OrthoDFA
