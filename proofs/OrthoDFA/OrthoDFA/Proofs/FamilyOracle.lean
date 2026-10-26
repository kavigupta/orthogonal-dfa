import OrthoDFA.LowLevelClaims.FamilyOracle
import OrthoDFA.Proofs.SuffixFamily

/-!
# A suffix family's reads are level 5's oracle

The family reads every string once, independently across strings, each with its target state's
distribution, and on the side its state reads less often at most `κ U` of the time, `U` the
state's chance of reading undecided. So the reads are a `KappaReadOracle`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace SuffixFamily

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] {σ : Type*}
  (fam : SuffixFamily μ α σ)

theorem readDist_wrong (q : σ) :
    fam.readDist q (if readSide (fam.readDist q) then .reject else .accept)
      ≤ fam.κ * fam.readDist q .undecided := by
  have hr := fam.readDist_rare q
  by_cases h : fam.readDist q .reject < fam.readDist q .accept
  · have hs : readSide (fam.readDist q) = true := decide_eq_true h
    rw [if_pos hs]
    exact (le_of_eq (min_eq_right h.le).symm).trans hr
  · have hs : readSide (fam.readDist q) = false := decide_eq_false h
    rw [if_neg (by simp [hs])]
    push Not at h
    exact (le_of_eq (min_eq_left h).symm).trans hr

/-- The family's reads, as level 5's oracle. -/
def kappaOracle :
    Random.Kappa.KappaReadOracle μ fam.M (fun q => readSide (fam.readDist q))
      (fun q => fam.readDist q .undecided) fam.κ where
  read := fun z ω => fam.read ω z
  measurable := fam.read_measurable
  indep := fam.read_indep
  undecided := fun z => fam.reads_by_state z .undecided
  wrong := fun z => by
    rw [fam.reads_by_state]
    exact fam.readDist_wrong _

end SuffixFamily

theorem familyKappaOracle_holds : FamilyKappaOracle := by
  intro α _ σ Ω _ μ _ fam
  classical
  exact ⟨fam.kappaOracle, rfl⟩

end OrthoDFA

end
