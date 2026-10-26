import OrthoDFA.LowLevelClaims.CoveredStart
import OrthoDFA.Proofs.UniformStrings

/-!
# From a covered start

Where every state of `S` has its edges true and is read on its true side, the hypothesis from the
start's true leaf follows the re-rooted target along every run that stays in `S`
(`classifies_of_matches`), so it misjudges at most what re-rooting does (`classify_start_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

section Start

open scoped Classical

variable {σ : Type*} (M : _root_.DFA α σ) (side : σ → Bool)

/-- Where every state of `S` has its edges true and is read on its true side, the hypothesis from
`q₀`'s true leaf classifies every string whose run from `q₀` stays in `S` as the re-rooted target
does. -/
theorem classifies_of_matches (H : HypothesisDFA α) (hroot : ∃ r a, H.tree = .node 1 r a)
    (S : Set σ)
    (hE : ∀ q ∈ S, ∀ a, ∃ w, H.edges (Random.leafOf M side H.tree q, a)
      = some (Random.leafOf M side H.tree (M.step q a), w))
    (hside : ∀ q ∈ S, side q = decide (q ∈ M.accept)) (q₀ : σ) (x : FreeMonoid α)
    (hstay : ∀ i ≤ x.toList.length, M.evalFrom q₀ (prefixOf x i).toList ∈ S)
    (hagree : M.evalFrom q₀ x.toList ∈ M.accept ↔ M.eval x.toList ∈ M.accept) :
    H.Classifies (Random.leafOf M side H.tree q₀) x (M.eval x.toList ∈ M.accept) := by
  have track : ∀ (l : List α) (q : σ), (∀ i ≤ l.length, M.evalFrom q (l.take i) ∈ S) →
      walkEnd H.edges (Random.leafOf M side H.tree q) l = some (Random.leafOf M side H.tree (M.evalFrom q l)) := by
    intro l
    induction l with
    | nil => intro q _; simp [walkEnd]
    | cons c l ih =>
      intro q hs
      have hq : q ∈ S := by simpa using hs 0 (by simp)
      obtain ⟨w, hw⟩ := hE q hq c
      simp only [walkEnd, hw, Option.bind_some]
      rw [ih (M.step q c) fun i hi => by simpa using hs (i + 1) (by simp; omega)]
      rfl
  have hs : ∀ i ≤ x.toList.length, M.evalFrom q₀ (x.toList.take i) ∈ S := fun i hi => by
    simpa [prefixOf] using hstay i hi
  refine ⟨_, track x.toList q₀ hs, ?_⟩
  have hend := hs x.toList.length le_rfl
  rw [List.take_length] at hend
  obtain ⟨r, a, hT⟩ := hroot
  have hhead : (Random.leafOf M side H.tree (M.evalFrom q₀ x.toList)).head?
      = some (side (M.evalFrom q₀ x.toList)) := by
    rw [hT]
    simp only [Random.leafOf, FreeMonoid.toList_one, DFA.evalFrom_nil]
    split_ifs with h <;> simp [h]
  rw [hhead, ← hagree, hside _ hend]
  simp

variable [Nonempty α]

/-- The hypothesis from `q₀`'s true leaf misjudges at most what re-rooting at `q₀` does. -/
theorem classify_start_le (H : HypothesisDFA α) (hroot : ∃ r a, H.tree = .node 1 r a) (S : Set σ)
    (hE : ∀ q ∈ S, ∀ a, ∃ w, H.edges (Random.leafOf M side H.tree q, a)
      = some (Random.leafOf M side H.tree (M.step q a), w))
    (hside : ∀ q ∈ S, side q = decide (q ∈ M.accept)) (L : ℕ) (q₀ : σ) {γ : ℝ}
    (hacc : 1 - γ ≤ rerootAccuracy M L S q₀) :
    (uniformStrings L).real {x | ¬ H.Classifies (Random.leafOf M side H.tree q₀) x
      (M.eval x.toList ∈ M.accept)} ≤ γ := by
  set good := {x : FreeMonoid α | (∀ i ≤ L, M.evalFrom q₀ (prefixOf x i).toList ∈ S)
    ∧ (M.evalFrom q₀ x.toList ∈ M.accept ↔ M.eval x.toList ∈ M.accept)}
  have hsub : {x | ¬ H.Classifies (Random.leafOf M side H.tree q₀) x (M.eval x.toList ∈ M.accept)}
      ⊆ goodᶜ ∪ {x | x.toList.length ≠ L} := by
    intro x hx
    by_contra hc
    simp only [Set.mem_union, Set.mem_compl_iff, Set.mem_setOf_eq, not_or, not_not] at hc hx
    obtain ⟨⟨hst, hag⟩, hl⟩ := hc
    exact hx (classifies_of_matches M side H hroot S hE hside q₀ x (fun i hi => hst i (hl ▸ hi)) hag)
  have hnull : (uniformStrings (α := α) L).real {x | x.toList.length ≠ L} = 0 := by
    rw [measureReal_def, uniformStrings_length, ENNReal.toReal_zero]
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans ((measureReal_union_le _ _).trans ?_)
  rw [hnull, add_zero, measureReal_compl MeasurableSpace.measurableSet_top, probReal_univ]
  have : 1 - γ ≤ (uniformStrings L).real good := hacc
  linarith

end Start

theorem coveredStartCorrect_holds : CoveredStartCorrect := by
  intro α _ _ _ σ M side H hroot S hE hside L q₀ γ hacc
  exact classify_start_le M side H hroot S hE hside L q₀ hacc

end OrthoDFA

end
