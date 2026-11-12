import RLStar.Code.Clustering

namespace OrthoDFA

open MeasureTheory

open scoped Classical in
/-- The draws of `y` landing in the set `A` fixes from the draws before them. -/
noncomputable def adaptiveHits {S : Type*} (A : List S → Set S) {n : ℕ} (y : Fin n → S) : ℕ :=
  (Finset.univ.filter fun i : Fin n => y i ∈ A ((List.ofFn y).take i)).card

/-- Over `n` independent draws from `D`, with every set `A` fixes holding at least `q` of `D`, the
hits stay at or below `c` with chance at most `P(Bin(n, q) ≤ c)`, and

    P(Bin(n, q) ≤ c) ≤ exp(−n q / 8)    for c ≤ n q / 2. -/
def AdaptiveHitsLower : Prop :=
  ∀ {S : Type*} [MeasurableSpace S] [Countable S] [MeasurableSingletonClass S]
    (D : Measure S) [IsProbabilityMeasure D] (n c : ℕ) (q : ℝ) (A : List S → Set S),
    0 ≤ q → q ≤ 1 →
    (Measure.pi fun _ : Fin n => D).real
        {y | (∀ i : Fin n, q ≤ D.real (A ((List.ofFn y).take i))) ∧ adaptiveHits A y ≤ c}
      ≤ 1 - binomSfGe n q (c + 1)
    ∧ ((c : ℝ) ≤ n * q / 2 → 1 - binomSfGe n q (c + 1) ≤ Real.exp (-(n * q / 8)))

end OrthoDFA
