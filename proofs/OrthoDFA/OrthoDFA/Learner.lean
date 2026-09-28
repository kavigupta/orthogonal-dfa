import OrthoDFA.Clustering

/-!
# The target as a DFA, and the quality of a suffix family

With the target a DFA, the state a prefix reaches decides how any suffix family's vote on it
is distributed, so how well a family cuts is a matter of which states it cuts well.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

/-- A deterministic automaton over the string monoid, acting on the right. -/
structure DFA (S : Type*) [Monoid S] (Q : Type*) where
  step : Q → S → Q
  step_one : ∀ q, step q 1 = q
  step_mul : ∀ q a b, step q (a * b) = step (step q a) b
  start : Q
  accept : Set Q

variable {Q : Type*} [Fintype Q]

/-- The state a prefix reaches. -/
def DFA.state (A : DFA S Q) (p : S) : Q := A.step A.start p

/-- The chance, over the oracle's noise, that `G` decides `p` on the wrong side. -/
noncomputable def miscutAt (O : Oracle μ S) (lo hi : ℕ) (G : Finset S) (p : S) : ℝ :=
  μ.real {ω | ¬ cutCorrect O lo hi G p ω}

/-- The chance that `G`, read as the gate reads it, leaves `p` undecided. -/
noncomputable def undecidedAt (O : Oracle μ S) (lo hi : ℕ) (G : Finset S) (p : S) : ℝ :=
  μ.real {ω | ¬ decided O.mq lo (hi - 1) (G.erase 1) p ω}

open scoped Classical in
/-- `(states G cuts wrongly, states G leaves undecided)`, ordered componentwise.  A state counts
when its prefixes suffer that more than `θ` of the time; every prefix of a state suffers it
equally often, since the vote's distribution depends only on the state. -/
noncomputable def quality (A : DFA S Q) (O : Oracle μ S) (θ : ℝ) (B : State) (G : Finset S) :
    ℕ × ℕ :=
  ((Finset.univ.filter fun q => ∃ p, A.state p = q ∧ θ < miscutAt O B.lo B.hi G p).card,
    (Finset.univ.filter fun q => ∃ p, A.state p = q ∧ θ < undecidedAt O B.lo B.hi G p).card)

/-- The mass a population puts on a state. -/
noncomputable def stateMass (A : DFA S Q) (D : Measure S) (q : Q) : ℝ :=
  D.real {p | A.state p = q}

open scoped Classical in
/-- The quality the populations force.  A state is safe from being cut wrongly once some
population puts more than `2·εcov/θ` on it, and from being left undecided once some population
puts more than `4·indecisionLimit/θ` on it; every other state may go either way. -/
noncomputable def qualityBound (A : DFA S Q) {J : Type*} (populations : Finset J)
    (D : J → Measure S) (θ εcov indecisionLimit : ℝ) : ℕ × ℕ :=
  (Fintype.card Q - (Finset.univ.filter fun q =>
      ∃ j ∈ populations, 2 * εcov / θ < stateMass A (D j) q).card,
    Fintype.card Q - (Finset.univ.filter fun q =>
      ∃ j ∈ populations, 4 * indecisionLimit / θ < stateMass A (D j) q).card)

/-- With probability `≥ 1 − δ` the loop stops at one of `states`, and the family it returns
there is at least as good as its populations force. -/
def QualityGuarantee : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {J : Type*} [Fintype J] {Q : Type*} [Fintype Q]
    (A : DFA S Q) (O : Oracle μ S) (populations : Finset J) (Pre : Set S)
    (η₀ indecisionLimit εcov α δ pAP θ : ℝ),
  O.L = {w | A.state w ∈ A.accept} →
  O.η ≤ η₀ →
  η₀ < 1 / 2 →
  populations.Nonempty →
  Flat Pre →
  0 < pAP →
  0 < indecisionLimit →
  indecisionLimit ≤ 1 / 2 →
  0 < α →
  α < 1 / 2 →
  0 < εcov →
  εcov ≤ 1 →
  0 < δ →
  δ ≤ 1 →
  0 < θ →
  ∃ cap : ℝ,
    0 < cap ∧
    ∀ (D : J → Measure S) (Dsf : Measure S),
      (∀ j, IsProbabilityMeasure (D j)) → IsProbabilityMeasure Dsf →
      (∀ j ∈ populations, D j Preᶜ = 0) →
      pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
      ∀ ρ : ℝ,
      (∀ j ∈ populations, collisionMass (D j) ≤ ρ) →
      ρ ≤ cap →
      collisionMass Dsf ≤ cap →
      ∃ states : Finset State,
        1 - δ ≤ (runMeasure μ D Dsf).real
          {x | (∃ B : {B : State // B ∈ states},
                x ∈ ret O.mq populations indecisionLimit α B.val)
            ∧ ∀ B : {B : State // B ∈ states},
              x ∈ ret O.mq populations indecisionLimit α B.val →
              quality A O θ B.val (clusterAt O.mq populations x B.val)
                ≤ qualityBound A populations D θ εcov indecisionLimit}

end OrthoDFA
