import OrthoDFA.ReturnAccuracy

/-!
# The merge check

The check reads each state's members against fresh suffixes.  When every member has the same
label, a member's own read is independent of its read after any suffix, whatever target states
the members come from, so a state with a pure label rarely fails.  When the labels are mixed,
an accept-preserving suffix makes the two reads agree, so a state with a large minority rarely
passes.

Known modelling gap.  `state_split` orders the suffixes and tests hypergeometric tails of the
members' reads; here each fresh suffix gets a sign test over pairs of members.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S] {Q R : Type*} [Fintype Q] [Fintype R]

open scoped Classical in
/-- The draws reaching `h`, in order. -/
noncomputable def membersOf (H : DFA S R) (h : R) {N : ℕ} (u : Fin N → S) : List S :=
  (List.ofFn u).filter fun p => H.state p = h

/-- Summed over the pairs `(2k, 2k+1)` for `k < n`: `+1` when both reads after `v` and both
own reads differ within the pair in the same direction, `-1` when in opposite ones. -/
noncomputable def leaning (O : Oracle μ S) (n : ℕ) (members : List S) (v : S) (ω : Ω) : ℝ :=
  ∑ k ∈ Finset.range n,
    (O.mq (members.getD (2 * k) 1 * v) ω - O.mq (members.getD (2 * k + 1) 1 * v) ω)
      * (O.mq (members.getD (2 * k) 1) ω - O.mq (members.getD (2 * k + 1) 1) ω)

/-- `h` fails when it has `2n` members and some fresh suffix other than `ε` leans by `2t`. -/
def checkFails (O : Oracle μ S) (H : DFA S R) (n : ℕ) (t : ℝ) {N m : ℕ} (h : R)
    (x : (Fin N → S) × (Fin m → S)) (ω : Ω) : Prop :=
  2 * n ≤ (membersOf H h x.1).length
    ∧ ∃ j, x.2 j ≠ 1 ∧ 2 * t ≤ leaning O n (membersOf H h x.1) (x.2 j) ω

/-- The check at a state `h` carrying `ε/|R|` of the sampler, drawing `N` strings from the
sampler and `m` suffixes from `Dsf`, with the noise already read at `U` pinned to anything.

It fails `h` with probability at most `m·e^{−2t²/n}` beyond the chance that one of its `2n`
members is in the minority.  It passes `h` with probability at most the sum of:
* the chance that no suffix drawn is accept-preserving;
* the chance that `h` gets fewer than `2n` members;
* a Hoeffding tail at the margin a minority share `w` gives.

Both bounds also allow for strings read twice. -/
def CheckGuarantee : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {Q R : Type*} [Fintype Q] [Fintype R]
    (A : DFA S Q) (O : Oracle μ S) (Dsamp Dsf : Measure S) (Pre Suf : Set S) (H : DFA S R) (h : R)
    (U : Finset S) (b : S → ℝ) (N m n T : ℕ) (η₀ ε κ t pAP w : ℝ),
  IsProbabilityMeasure Dsamp → IsProbabilityMeasure Dsf →
  O.L = {v | A.state v ∈ A.accept} →
  Flat Pre Suf →
  Dsamp Preᶜ = 0 →
  Dsf Sufᶜ = 0 →
  (∀ a, Dsamp.real {a} ≤ κ) →
  0 < ε →
  ε / Fintype.card R ≤ Dsamp.real {v | H.state v = h} →
  U.card ≤ T →
  0 ≤ t →
  (((μ[|pinned O U b]).prod ((Measure.pi fun _ : Fin N => Dsamp).prod
        (Measure.pi fun _ : Fin m => Dsf))).real {z | checkFails O H n t h z.2 z.1}
      ≤ m * Real.exp (-2 * t ^ 2 / n) + 2 * n * minorityShare A H Dsamp h
        + κ * Fintype.card R / ε * (2 * n ^ 2 + 2 * n * (m + 1) * T))
  ∧ (O.η ≤ η₀ → η₀ < 1 / 2 →
      pAP ≤ Dsf.real {v | v ≠ 1 ∧ ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
      w ≤ minorityShare A H Dsamp h →
      (2 * n : ℝ) ≤ N * ε / Fintype.card R →
      2 * t ≤ n * w * (1 - 2 * η₀) ^ 2 →
      ((μ[|pinned O U b]).prod ((Measure.pi fun _ : Fin N => Dsamp).prod
          (Measure.pi fun _ : Fin m => Dsf))).real {z | ¬ checkFails O H n t h z.2 z.1}
        ≤ (1 - pAP) ^ m + Real.exp (-2 * (N * ε / Fintype.card R - 2 * n) ^ 2 / N)
          + Real.exp (-(n * w * (1 - 2 * η₀) ^ 2 - 2 * t) ^ 2 / (2 * n))
          + κ * Fintype.card R / ε * (2 * n ^ 2 + 2 * n * (m + 1) * T))

end OrthoDFA
