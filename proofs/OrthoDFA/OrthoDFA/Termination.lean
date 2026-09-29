import OrthoDFA.ReturnAccuracy

/-!
# The learner stops

Each round the clustering picks a family from the populations so far, and the L\* stage builds a
hypothesis with it.  A gate compares the hypothesis with the family's cut; past the gate, the
merge check reads every state carrying at least `ε/|R|` of the sampler.  Every such state of
every round's hypothesis becomes a population, and so does what a round harvests, when it keeps
it.

A round that does not return makes one of those populations put enough on some target state
that no earlier population did.  If the gate refused, either the family cut much of the sampler
badly, and some state of the hypothesis holds enough of it, or the round harvested enough of a
state the family leaves undecided.  If the check failed a state, the minority it holds is mostly
strings the family cuts badly.  So at most `2·|Q|` rounds do not return.

The clustering, the stage, the gate and the check are each held to a spec, and nothing else is
assumed of them.  The clustering's is what `ClusteringQualityGuarantee` proves of one round.
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

/-- The share of the sampler the family cuts badly. -/
noncomputable def badlyCut (O : Oracle μ S) (tolerance : ℝ) (B : State) (F : Finset S)
    (Dsamp : Measure S) : ℝ :=
  Dsamp.real {p | tolerance < miscutProb O B.lo B.hi F p
    ∨ tolerance < undecidedProb O B.lo B.hi F p}

/-- Round `r`'s populations: the first ones, every state carrying `ε/|R|` of the sampler in
every earlier round's hypothesis, and every earlier harvest that was kept. -/
def poolsAt {Θ J : Type*} (populations : Finset J) (D : J → Measure S) (Dsamp : Measure S)
    (ε : ℝ) (hyp : ℕ → Θ → DFA S R) (harvest : ℕ → Θ → Measure S) (kept : ℕ → Θ → Prop)
    (r : ℕ) (θ : Θ) : Set (Measure S) :=
  {D' | (∃ j ∈ populations, D' = D j)
    ∨ (∃ i < r, ∃ h, ε / Fintype.card R ≤ Dsamp.real {v | (hyp i θ).state v = h}
      ∧ D' = reaching (hyp i θ) Dsamp h)
    ∨ ∃ i < r, kept i θ ∧ D' = harvest i θ}

/-- Round `r` settles on the cut `family r` and builds `hyp r`; `gate r` says whether it gets
past the gate, and the merge check fails at `fails r`.  It harvests `harvest r`, which becomes a
population when `kept r`.  A round returns when it gets past the
gate and no state fails.  Suppose each round, but for probabilities `δc`, `δs`, `δg` and `δa`:

* the clustering cuts well every state the populations so far force it to, as
  `ClusteringQualityGuarantee` has it;
* past the gate, the hypothesis labels all but `ζ` of what the family cuts well the way the
  target does;
* the gate refuses only when the family cuts at least `x₀` of the sampler badly, or the round
  keeps a harvest putting more than `θh` on a state the family leaves undecided;
* the check fails only states with minority share at least `wₛ`, and only states carrying
  `ε/|R|`.

Then none of the first `2·|Q| + 1` rounds returns with probability at most
`(2·|Q| + 1)·(δc + δs + δg + δa)`.  This needs the minority a failed state must hold, spread
over the target states, and the badly cut share a refusal implies, spread over the states of both
automata, each to outweigh both of `qualityBound`'s thresholds. -/
def Termination : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {Q R : Type*} [Fintype Q] [Fintype R] {J Θ : Type*}
    [MeasurableSpace Θ] (P : Measure Θ) [IsProbabilityMeasure P]
    (A : DFA S Q) (O : Oracle μ S) (Dsamp : Measure S) (populations : Finset J)
    (D : J → Measure S) (family : ℕ → Θ → State × Finset S) (hyp : ℕ → Θ → DFA S R)
    (gate : ℕ → Θ → Prop) (fails : ℕ → Θ → Finset R) (harvest : ℕ → Θ → Measure S)
    (kept : ℕ → Θ → Prop) (tolerance εcov indecisionLimit ε ζ x₀ θh wₛ δc δs δg δa : ℝ),
  IsProbabilityMeasure Dsamp →
  0 < ε →
  0 ≤ ζ →
  0 < tolerance →
  0 ≤ εcov →
  2 * εcov / tolerance < (wₛ - ζ * Fintype.card R / ε) / Fintype.card Q →
  4 * indecisionLimit / tolerance < (wₛ - ζ * Fintype.card R / ε) / Fintype.card Q →
  2 * εcov / tolerance < (x₀ - ε) / (Fintype.card R * Fintype.card Q) →
  4 * indecisionLimit / tolerance < (x₀ - ε) / (Fintype.card R * Fintype.card Q) →
  2 * εcov / tolerance ≤ θh →
  4 * indecisionLimit / tolerance ≤ θh →
  (∀ r, P.real {θ | ∃ q,
      ((∃ D' ∈ poolsAt populations D Dsamp ε hyp harvest kept r θ,
          2 * εcov / tolerance < stateMass A D' q)
        ∧ ∃ p, A.state p = q
          ∧ tolerance < miscutProb O (family r θ).1.lo (family r θ).1.hi (family r θ).2 p)
      ∨ ((∃ D' ∈ poolsAt populations D Dsamp ε hyp harvest kept r θ,
          4 * indecisionLimit / tolerance < stateMass A D' q)
        ∧ ∃ p, A.state p = q
          ∧ tolerance < undecidedProb O (family r θ).1.lo (family r θ).1.hi (family r θ).2 p)}
    ≤ δc) →
  (∀ r, P.real {θ | gate r θ ∧ ζ < mislabelledWellCut A O tolerance (family r θ).1
      (family r θ).2 (hyp r θ) Dsamp} ≤ δs) →
  (∀ r, P.real {θ | ¬ gate r θ
      ∧ badlyCut O tolerance (family r θ).1 (family r θ).2 Dsamp < x₀
      ∧ ¬ (kept r θ ∧ ∃ q, (∃ p, A.state p = q
          ∧ tolerance < undecidedProb O (family r θ).1.lo (family r θ).1.hi (family r θ).2 p)
        ∧ θh < stateMass A (harvest r θ) q)} ≤ δg) →
  (∀ r, P.real {θ | ∃ h ∈ fails r θ, minorityShare A (hyp r θ) Dsamp h < wₛ} ≤ δa) →
  (∀ r θ, ∀ h ∈ fails r θ, ε / Fintype.card R ≤ Dsamp.real {v | (hyp r θ).state v = h}) →
  P.real {θ | ∀ r < 2 * Fintype.card Q + 1, ¬ gate r θ ∨ (fails r θ).Nonempty}
    ≤ (2 * Fintype.card Q + 1) * (δc + δs + δg + δa)

end OrthoDFA
