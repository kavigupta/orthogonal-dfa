import RLStar.LowLevelClaims.LazyRound
import RLStar.Proofs.Level3NoWrong
import RLStar.Proofs.RandomHarvest
import Mathlib.Data.List.ReduceOption

/-!
# `EpochCorrect`: the proof

An epoch is `level3NoWrongCorrect_holds` over the alphabet with a pad letter: the target loops on
pads, so reading `read` with the pads dropped is never wrong, and the probes are drawn from `D`
padded.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

namespace Random

variable {α : Type*} [Fintype α] [DecidableEq α]

omit [Fintype α] [DecidableEq α] in
private theorem padded_evalFrom {σ : Type*} (M : DFA α σ) (l : List (Option α)) :
    ∀ q, M.padded.evalFrom q l = M.evalFrom q l.reduceOption := by
  induction l with
  | nil => intro q; rfl
  | cons c l ih =>
    intro q
    cases c with
    | none => rw [List.reduceOption_cons_of_none, DFA.evalFrom_cons]; exact ih q
    | some a =>
      rw [List.reduceOption_cons_of_some, DFA.evalFrom_cons, DFA.evalFrom_cons]
      exact ih _

omit [Fintype α] [DecidableEq α] in
private theorem readsNoWrong_padded {σ : Type*} {M : DFA α σ} {side : σ → Bool}
    {read : FreeMonoid α → ARU} (hW : ReadsNoWrong M side read) :
    ReadsNoWrong M.padded side (padRead read) := fun z => by
  change read (unpad z) ≠ if side (M.padded.evalFrom M.start z.toList) then _ else _
  rw [padded_evalFrom]
  exact hW (unpad z)

private theorem epochQ_eval (read : FreeMonoid α → ARU) (C : Level3Settings) (K : ℕ)
    (X : Finset (FreeMonoid α)) {P : ℕ} (draws : Fin P → FreeMonoid α) :
    (epochQ C K X draws).eval read
      = indecisiveLStarLevel3 { C with walkStart := K } (padRead read)
          fun i => padIn K X (draws i) := by
  rw [epochQ, Query.eval_subst]
  rfl

omit [DecidableEq α] in
private theorem measurable_padIn (K : ℕ) (X : Finset (FreeMonoid α)) :
    Measurable (padIn K X) :=
  measurable_of_countable _

omit [DecidableEq α] in
private theorem harvestLaw_padded (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (K : ℕ) (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α))
    (s : Level3Site) :
    (harvestLaw (padRead read) (D.map (padIn K X)) K H s).real Set.univ
      = harvestMass read D K X H s := by
  rw [harvestLaw_real_univ, integral_map (measurable_padIn K X).aemeasurable
    (measurable_of_countable _).aestronglyMeasurable, harvestMass,
    ← integral_indicator_one (Set.to_countable _).measurableSet]
  congr 1
  funext x
  have hl := siteHarvest_length (padRead read) K H s (padIn K X x)
  simp only [Set.indicator_apply, Set.mem_ofPred_eq, Pi.one_apply]
  generalize siteHarvest (padRead read) K H s (padIn K X x) = l at hl ⊢
  rcases l with _ | ⟨a, _ | ⟨b, l⟩⟩
  · simp
  · simp
  · simp at hl

end Random

open Random

theorem epochCorrect_holds : EpochCorrect := by
  intro α _ _ σ _ M side read C D _ K X
  dsimp only
  intro hW hδ hδ1 hε0 hε1 hP
  have hπ := measurable_padIn K X
  have : IsProbabilityMeasure (D.map (padIn K X)) :=
    Measure.isProbabilityMeasure_map hπ.aemeasurable
  have key := level3NoWrongCorrect_holds M.padded side (padRead read) { C with walkStart := K }
    (D.map (padIn K X))
  dsimp only at key
  simp only [Fintype.card_option] at key
  have h := key (readsNoWrong_padded hW) hδ hδ1 hε0 hε1 hP
  have hF : Measurable fun (d : Fin C.probeBudget → FreeMonoid α) i => padIn K X (d i) :=
    measurable_pi_lambda _ fun i => hπ.comp (measurable_pi_apply i)
  rw [← Measure.pi_map_pi fun _ => hπ.aemeasurable,
    map_measureReal_apply hF (Set.to_countable _).measurableSet] at h
  refine le_trans (measureReal_mono ?_) h
  intro d hd
  simp only [Set.mem_ofPred_eq, Set.mem_preimage, epochQ_eval] at hd ⊢
  generalize indecisiveLStarLevel3 { C with walkStart := K } (padRead read)
    (fun i => padIn K X (d i)) = r at hd ⊢
  rcases hd with hd | hd
  · exact Or.inl hd
  · right
    rcases r with ⟨H, _ | s | _⟩ <;> dsimp only at hd ⊢
    · rw [map_measureReal_apply hπ (Set.to_countable _).measurableSet]
      exact hd
    · rw [harvestLaw_padded]
      exact hd

end OrthoDFA

end
