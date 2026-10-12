import OrthoDFA.ReturnAccuracy

/-!
# The learner stops

Each round the clustering picks a family from the populations so far, the L\* stage builds a
hypothesis with it, and the merge check reads every state carrying at least `ε/|R|` of the
sampler.  A round in which some such state fails adds that state's own population,
`reaching H h`, and goes on.  That population puts enough on some target state that no earlier
population did, so at most `2·|Q|` rounds fail.

The clustering, the stage and the check are each held to a spec, and nothing else is assumed of
them.  The clustering's is what `ClusteringQualityGuarantee` proves of one round.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S] {Q R : Type*} [Fintype Q] [Fintype R]

/-- The share of the sampler the hypothesis labels wrongly although the family cuts it well. -/
noncomputable def mislabelledWellCut (A : DFA S Q) (O : Oracle μ S) (tolerance : ℝ) (B : State)
    (F : Finset S) (H : DFA S R) (Dsamp : Measure S) : ℝ :=
  Dsamp.real {p | (A.state p ∈ A.accept ↔ H.state p ∉ H.accept)
    ∧ miscutProb O B.lo B.hi F p ≤ tolerance ∧ undecidedProb O B.lo B.hi F p ≤ tolerance}

/-- Round `r`'s populations: the first ones, and every state that failed the check before. -/
def poolsAt {Θ J : Type*} (populations : Finset J) (D : J → Measure S) (Dsamp : Measure S)
    (hyp : ℕ → Θ → DFA S R) (fails : ℕ → Θ → Finset R) (r : ℕ) (θ : Θ) : Set (Measure S) :=
  {D' | (∃ j ∈ populations, D' = D j)
    ∨ ∃ i < r, ∃ h ∈ fails i θ, D' = reaching (hyp i θ) Dsamp h}

/-- Round `r` settles on the cut `family r` and builds `hyp r`, and the merge check fails at
`fails r`.  Suppose each round, but for probabilities `δc`, `δs` and `δa`:

* the clustering cuts well every state the populations so far force it to, as
  `ClusteringQualityGuarantee` has it;
* the stage labels all but `ζ` of the sampler the family cuts well the way the target does;
* the check fails only states with minority share at least `wₛ`, and only states carrying
  `ε/|R|`.

Then all of the first `2·|Q| + 1` rounds fail with probability at most
`(2·|Q| + 1)·(δc + δs + δa)`, provided the minority a failed state must hold outweighs, spread
over the target states, both of `qualityBound`'s thresholds. -/
def Termination : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {Q R : Type*} [Fintype Q] [Fintype R] {J Θ : Type*}
    [MeasurableSpace Θ] (P : Measure Θ) [IsProbabilityMeasure P]
    (A : DFA S Q) (O : Oracle μ S) (Dsamp : Measure S) (populations : Finset J)
    (D : J → Measure S) (family : ℕ → Θ → State × Finset S) (hyp : ℕ → Θ → DFA S R)
    (fails : ℕ → Θ → Finset R) (tolerance εcov indecisionLimit ε ζ wₛ δc δs δa : ℝ),
  IsProbabilityMeasure Dsamp →
  0 < ε →
  0 ≤ ζ →
  0 < tolerance →
  0 ≤ εcov →
  2 * εcov / tolerance < (wₛ - ζ * Fintype.card R / ε) / Fintype.card Q →
  4 * indecisionLimit / tolerance < (wₛ - ζ * Fintype.card R / ε) / Fintype.card Q →
  (∀ r, P.real {θ | ∃ q,
      ((∃ D' ∈ poolsAt populations D Dsamp hyp fails r θ,
          2 * εcov / tolerance < stateMass A D' q)
        ∧ ∃ p, A.state p = q
          ∧ tolerance < miscutProb O (family r θ).1.lo (family r θ).1.hi (family r θ).2 p)
      ∨ ((∃ D' ∈ poolsAt populations D Dsamp hyp fails r θ,
          4 * indecisionLimit / tolerance < stateMass A D' q)
        ∧ ∃ p, A.state p = q
          ∧ tolerance < undecidedProb O (family r θ).1.lo (family r θ).1.hi (family r θ).2 p)}
    ≤ δc) →
  (∀ r, P.real {θ | ζ < mislabelledWellCut A O tolerance (family r θ).1 (family r θ).2
      (hyp r θ) Dsamp} ≤ δs) →
  (∀ r, P.real {θ | ∃ h ∈ fails r θ, minorityShare A (hyp r θ) Dsamp h < wₛ} ≤ δa) →
  (∀ r θ, ∀ h ∈ fails r θ, ε / Fintype.card R ≤ Dsamp.real {v | (hyp r θ).state v = h}) →
  P.real {θ | ∀ r < 2 * Fintype.card Q + 1, (fails r θ).Nonempty}
    ≤ (2 * Fintype.card Q + 1) * (δc + δs + δa)

end OrthoDFA
