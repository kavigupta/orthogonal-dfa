import OrthoDFA.SuffixFamily
import OrthoDFA.Proofs.FamilyRead

/-!
# A suffix family's read distributions

The reads are independent across strings and every string reads with its state's distribution,
the law of a vote whose count of accepting members the state fixes; so it is nonnegative and on
its rarer decided side at most `κ` times as often as undecided.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {σ : Type*} (fam : SuffixFamily μ α σ)

namespace SuffixFamily

theorem readProb_eq_readDist (w : FreeMonoid α) :
    readProb fam.O fam.F fam.kl fam.kh w = fam.readDist (fam.M.eval w.toList) := by
  classical
  have h : ∃ w' : FreeMonoid α, fam.M.eval w'.toList = fam.M.eval w.toList := ⟨w, rfl⟩
  simp only [readDist, dif_pos h]
  funext rd
  rw [readProb_eq_readLaw, readProb_eq_readLaw, acceptCount_eq_state fam.O fam.M fam.lang,
    acceptCount_eq_state fam.O fam.M fam.lang, h.choose_spec]

theorem readDist_nonneg (q : σ) (r : ARU) : 0 ≤ fam.readDist q r := by
  classical
  simp only [readDist]
  split_ifs
  · exact measureReal_nonneg
  · exact le_rfl

theorem readDist_rare (q : σ) :
    min (fam.readDist q .accept) (fam.readDist q .reject)
        ≤ fam.κ * fam.readDist q .undecided := by
  classical
  by_cases h : ∃ w : FreeMonoid α, fam.M.eval w.toList = q
  · simp only [readDist, dif_pos h]
    rw [readProb_eq_readLaw, readProb_eq_readLaw, readProb_eq_readLaw]
    refine fam.band _ ?_
    rw [acceptCount_eq_state fam.O fam.M fam.lang]
    exact Finset.card_filter_le _ _
  · simp [readDist, dif_neg h]

theorem read_measurable (z : FreeMonoid α) : Measurable fun ω => fam.read ω z :=
  measurable_from_nat.comp (measurable_voteCount fam.O fam.F z)

theorem read_indep : iIndepFun (fun z ω => fam.read ω z) μ :=
  read_iIndep fam.O fam.suffixFree fam.kl fam.kh

theorem reads_by_state (z : FreeMonoid α) (r : ARU) :
    μ.real {ω | fam.read ω z = r} = fam.readDist (fam.M.eval z.toList) r :=
  congrFun (fam.readProb_eq_readDist z) r

end SuffixFamily

end OrthoDFA
