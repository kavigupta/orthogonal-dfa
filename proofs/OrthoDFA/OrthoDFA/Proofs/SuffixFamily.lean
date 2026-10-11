import OrthoDFA.SuffixFamily
import OrthoDFA.Proofs.FamilyRead

/-!
# A suffix family's laws

By `FamilyReadBound`, the reads are independent across strings and every string reads with its
state's law, which is nonnegative and meets the band's minority bound.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {σ : Type*} (fam : SuffixFamily μ α σ)

namespace SuffixFamily

theorem readProb_eq_law (w : FreeMonoid α) :
    readProb fam.O fam.F fam.kl fam.kh w = fam.law (fam.M.eval w.toList) := by
  classical
  obtain ⟨-, hq⟩ := family_read_bound_holds fam.O fam.M fam.F fam.kl fam.kh fam.ε fam.ε₂ fam.κ
    fam.lang fam.suffixFree fam.band
  choose d hd _ _ using hq
  have h : ∃ w' : FreeMonoid α, fam.M.eval w'.toList = fam.M.eval w.toList := ⟨w, rfl⟩
  simp only [law, dif_pos h]
  exact (hd _ w rfl).trans (hd _ _ h.choose_spec).symm

theorem law_nonneg (q : σ) (r : ARU) : 0 ≤ fam.law q r := by
  classical
  simp only [law]
  split_ifs
  · exact measureReal_nonneg
  · exact le_rfl

theorem law_rare (q : σ) :
    min (fam.law q .accept) (fam.law q .reject) ≤ max fam.ε (fam.κ * fam.law q .undecided) := by
  classical
  obtain ⟨-, hq⟩ := family_read_bound_holds fam.O fam.M fam.F fam.kl fam.kh fam.ε fam.ε₂ fam.κ
    fam.lang fam.suffixFree fam.band
  choose d hd _ hrare using hq
  by_cases h : ∃ w : FreeMonoid α, fam.M.eval w.toList = q
  · have e : fam.law q = d q := by
      simp only [law, dif_pos h]; exact hd _ _ h.choose_spec
    rw [e]; exact hrare q
  · simp [law, dif_neg h]

theorem read_measurable (z : FreeMonoid α) : Measurable fun ω => fam.read ω z :=
  measurable_from_nat.comp (measurable_voteCount fam.O fam.F z)

theorem read_indep : iIndepFun (fun z ω => fam.read ω z) μ :=
  read_iIndep fam.O fam.suffixFree fam.kl fam.kh

theorem read_law (z : FreeMonoid α) (r : ARU) :
    μ.real {ω | fam.read ω z = r} = fam.law (fam.M.eval z.toList) r :=
  congrFun (fam.readProb_eq_law z) r

end SuffixFamily

end OrthoDFA
