import OrthoDFA.LearnerLoop

/-! # Definitions the loop's proofs use -/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω}

def clusterLaw (μ : Measure Ω) (α : Type*) (K : ℕ) : Measure (ClusterDraws Ω α K) :=
  μ.prod (restLaw α K)

open scoped Classical in
/-- The chance `F`'s read of `p`, cut at `kl` and `kh`, is decided against `p`'s membership. -/
def misreadProb (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    (p : FreeMonoid α) : ℝ :=
  μ.real {ω | familyRead O.mq F kl kh p ω = if p ∈ O.L then .reject else .accept}

/-- The round ended consistent, and its read decides a draw from `D` against the target at most
`εr` of the time and leaves it undecided at most `εu`. -/
def LoopRound.Agrees {K : ℕ} (O : Oracle μ (FreeMonoid α)) (D : Measure (FreeMonoid α))
    (εr εu : ℝ) (o : LoopRound Ω α K) : Prop :=
  o.result.ending = .consistent
    ∧ ∫ p, misreadProb O o.family o.state.lo (o.state.hi + 1) p ∂D ≤ εr
    ∧ ∫ p, readProb O o.family o.state.lo (o.state.hi + 1) p .undecided ∂D ≤ εu

end OrthoDFA

end
