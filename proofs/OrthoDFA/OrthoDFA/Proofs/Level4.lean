import OrthoDFA.LowLevelClaims.Level4Claim
import OrthoDFA.Proofs.Level5

/-!
# Level 4: the proof

A read wrong with chance at most `εw` is a `KappaReadOracle` at `κ = 0`, so level 5 applies.
-/

namespace OrthoDFA

open MeasureTheory Random Random.Eps

theorem level4RoundCorrect_holds : Level4RoundCorrect := by
  intro α _ _ σ _ M side U εw Ω _ μ _ R P C p₀ f t t'
  dsimp only
  intro hpre hδ hδ1 hy0 hyρ hρ1 hP ht ht' htc hfin
  have hεw : 0 ≤ εw := measureReal_nonneg.trans (R.wrong default)
  exact level5RoundCorrect_holds M side U εw 0 μ
    ⟨R.read, R.measurable, R.indep, R.undecided, fun z => by simpa using R.wrong z⟩ P C p₀ f t t'
    hpre hδ hδ1 hy0 hyρ hρ1 hP hεw le_rfl ht ht' (by simpa using htc) (by simpa using hfin)

end OrthoDFA
