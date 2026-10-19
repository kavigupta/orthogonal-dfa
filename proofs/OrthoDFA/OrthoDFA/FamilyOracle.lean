import OrthoDFA.Level5Claim
import OrthoDFA.Proofs.SuffixFamily

/-!
# Level 5's oracle, read by a suffix family

The clustering's family reads every string once, independently across strings, each with its
target state's distribution (`readDist`), and on the side its state reads less often at most
`max ε (κ U)` of the time, `U` the state's chance of reading undecided (`FamilyReadBound`). So its
reads are a `KappaReadOracle` (`SuffixFamily.kappaOracle`) with each state's side the one it reads
more often, its undecided chance `U`, `εw = ε` and the band's `κ`, since
`max ε (κ U) ≤ ε + κ U`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The side a read distribution reads on more often. -/
def readSide (d : ARU → ℝ) : Bool := decide (d .reject < d .accept)

namespace SuffixFamily

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] {σ : Type*}
  (fam : SuffixFamily μ α σ)

theorem readDist_wrong (hε : 0 ≤ fam.ε) (hκ : 0 ≤ fam.κ) (q : σ) :
    fam.readDist q (if readSide (fam.readDist q) then .reject else .accept)
      ≤ fam.ε + fam.κ * fam.readDist q .undecided := by
  have hr := fam.readDist_rare q
  have hU : 0 ≤ fam.κ * fam.readDist q .undecided := mul_nonneg hκ (fam.readDist_nonneg q _)
  have hmax : max fam.ε (fam.κ * fam.readDist q .undecided)
      ≤ fam.ε + fam.κ * fam.readDist q .undecided := max_le (by linarith) (by linarith)
  by_cases h : fam.readDist q .reject < fam.readDist q .accept
  · have hs : readSide (fam.readDist q) = true := decide_eq_true h
    rw [if_pos hs]
    exact (le_of_eq (min_eq_right h.le).symm).trans (hr.trans hmax)
  · have hs : readSide (fam.readDist q) = false := decide_eq_false h
    rw [if_neg (by simp [hs])]
    push Not at h
    exact (le_of_eq (min_eq_left h).symm).trans (hr.trans hmax)

/-- The family's reads, as level 5's oracle. -/
def kappaOracle (hε : 0 ≤ fam.ε) (hκ : 0 ≤ fam.κ) :
    Random.Kappa.KappaReadOracle μ fam.M (fun q => readSide (fam.readDist q))
      (fun q => fam.readDist q .undecided) fam.ε fam.κ where
  read := fun z ω => fam.read ω z
  measurable := fam.read_measurable
  indep := fam.read_indep
  undecided := fun z => fam.reads_by_state z .undecided
  wrong := fun z => by
    rw [fam.reads_by_state]
    exact fam.readDist_wrong hε hκ _

end SuffixFamily

end OrthoDFA

end
