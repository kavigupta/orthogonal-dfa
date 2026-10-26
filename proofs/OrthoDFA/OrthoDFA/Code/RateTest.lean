import OrthoDFA.Code.Clustering

/-! The binomial test of a count of hits against a rate, and what it promises. -/

namespace OrthoDFA

open MeasureTheory

/-- `statistics.rate_side`: once `n₀` draws are in, above `θ` when `Bin(n, θ)` reaches `h`
hits with chance below `a`, below when it stays at or under `h` with chance below `a`. -/
noncomputable def rateSide (θ a : ℝ) (n₀ n h : ℕ) : Option Bool :=
  if n₀ ≤ n then
    if binomSfGe n θ h < a then some true
    else if 1 - binomSfGe n θ (h + 1) < a then some false
    else none
  else none

end OrthoDFA
