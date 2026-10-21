import Mathlib.Computability.DFA
import Mathlib.Data.Fintype.Card

/-!
# States agreeing on short words agree on all

Two states whose runs accept the same words shorter than `|Q|²` accept the same words: along a
shortest word they part on, no pair of states repeats, so it is shorter than `|Q|²`
(`evalFrom_agree_of_short`).
-/

namespace OrthoDFA

variable {α σ : Type*} [Fintype σ] (M : DFA α σ)

theorem evalFrom_agree_of_short {p q : σ}
    (h : ∀ w : List α, w.length < Fintype.card σ * Fintype.card σ →
      (M.evalFrom p w ∈ M.accept ↔ M.evalFrom q w ∈ M.accept)) :
    ∀ w : List α, M.evalFrom p w ∈ M.accept ↔ M.evalFrom q w ∈ M.accept := by
  classical
  intro w
  induction hn : w.length using Nat.strong_induction_on generalizing w with
  | _ n ih =>
    by_cases hw : w.length < Fintype.card σ * Fintype.card σ
    · exact h w hw
    push Not at hw
    have hcard : Fintype.card (σ × σ) < Fintype.card (Fin (w.length + 1)) := by
      rw [Fintype.card_prod, Fintype.card_fin]; omega
    obtain ⟨i, j, hij, heq⟩ := Fintype.exists_ne_map_eq_of_card_lt
      (fun i : Fin (w.length + 1) => (M.evalFrom p (w.take i), M.evalFrom q (w.take i))) hcard
    simp only [Prod.mk.injEq] at heq
    obtain ⟨hp, hq⟩ := heq
    wlog hlt : (i : ℕ) < j generalizing i j
    · exact this j i hij.symm hp.symm hq.symm (by
        have : (i : ℕ) ≠ j := fun h' => hij (Fin.ext h')
        omega)
    set w' := w.take i ++ w.drop j
    have hj : (j : ℕ) ≤ w.length := Nat.lt_succ_iff.1 j.2
    have hlen : w'.length < n := by
      simp only [w', List.length_append, List.length_take, List.length_drop]
      omega
    have key : ∀ s : σ, M.evalFrom s (w.take i) = M.evalFrom s (w.take j) →
        M.evalFrom s w = M.evalFrom s w' := by
      intro s hs
      calc M.evalFrom s w = M.evalFrom s (w.take j ++ w.drop j) := by rw [List.take_append_drop]
        _ = M.evalFrom (M.evalFrom s (w.take j)) (w.drop j) := DFA.evalFrom_of_append _ _ _ _
        _ = M.evalFrom (M.evalFrom s (w.take i)) (w.drop j) := by rw [hs]
        _ = M.evalFrom s w' := (DFA.evalFrom_of_append _ _ _ _).symm
    rw [key p hp, key q hq]
    exact ih w'.length (hn ▸ hlen) w' rfl

/-- A state reachable from `s` is reachable in fewer than `|Q|` letters. -/
theorem evalFrom_short (s : σ) :
    ∀ w : List α, ∃ w' : List α, w'.length < Fintype.card σ ∧ M.evalFrom s w' = M.evalFrom s w := by
  classical
  intro w
  induction hn : w.length using Nat.strong_induction_on generalizing w with
  | _ n ih =>
    by_cases hw : w.length < Fintype.card σ
    · exact ⟨w, hw, rfl⟩
    push Not at hw
    have hcard : Fintype.card σ < Fintype.card (Fin (w.length + 1)) := by
      rw [Fintype.card_fin]; omega
    obtain ⟨i, j, hij, heq⟩ := Fintype.exists_ne_map_eq_of_card_lt
      (fun i : Fin (w.length + 1) => M.evalFrom s (w.take i)) hcard
    wlog hlt : (i : ℕ) < j generalizing i j
    · exact this j i hij.symm heq.symm (by
        have : (i : ℕ) ≠ j := fun h' => hij (Fin.ext h')
        omega)
    set w' := w.take i ++ w.drop j
    have hj : (j : ℕ) ≤ w.length := Nat.lt_succ_iff.1 j.2
    have hlen : w'.length < n := by
      simp only [w', List.length_append, List.length_take, List.length_drop]
      omega
    have key : M.evalFrom s w = M.evalFrom s w' := by
      calc M.evalFrom s w = M.evalFrom s (w.take j ++ w.drop j) := by rw [List.take_append_drop]
        _ = M.evalFrom (M.evalFrom s (w.take j)) (w.drop j) := DFA.evalFrom_of_append _ _ _ _
        _ = M.evalFrom (M.evalFrom s (w.take i)) (w.drop j) := by rw [heq]
        _ = M.evalFrom s w' := (DFA.evalFrom_of_append _ _ _ _).symm
    obtain ⟨w'', h1, h2⟩ := ih w'.length (hn ▸ hlen) w' rfl
    exact ⟨w'', h1, h2.trans key.symm⟩

end OrthoDFA
