import OrthoDFA.LoopDFA
import OrthoDFA.Proofs.LoopPersist
import OrthoDFA.Proofs.Hoeffding
import OrthoDFA.Proofs.UniformStrings

/-!
# Picking a leaf on held-out strings

`select_le`: where a stage of the run is fixed by the answers at `W`, and a bad event of fresh
draws `b` by the answers at `Z`, the run lands at a stage whose bad event holds, or at one whose
`W` and `Z` meet, with chance at most the bad event's bound plus the meeting's.

A leaf's score sums `agreeAt` over the held-out strings. At distinct strings the family's reads are
independent, so the score is within `n t/2` of its mean given the strings, and that mean within
`n t/2` of `n · agreeMean` (`scoreBad_le`). Where every leaf's score is within `n t` of its mean,
the leaf of highest score misjudges at most `2t` more than the best leaf, and the mean is within
the family's misread and undecided shares of the leaf's accuracy (`pickLeaf_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ]

section Select

theorem select_le (O : Oracle μ (FreeMonoid α)) {X B Y : Type*} [MeasurableSpace X]
    [MeasurableSpace B] [Countable Y] (ν : Measure X) [IsProbabilityMeasure ν] (β : Measure B)
    [IsProbabilityMeasure β] (stage : Ω → X → Y) (G : Y → Prop)
    (hmeas : ∀ y, G y → MeasurableSet {p : Ω × X | stage p.1 p.2 = y})
    (W : Y → X → Finset (FreeMonoid α))
    (hW : ∀ y x ω ω', G y → stage ω x = y → (∀ w ∈ W y x, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
      stage ω' x = y)
    (Ok : Y → Set X) (hmOk : ∀ y, MeasurableSet (Ok y))
    (Bad : Y → Set (Ω × B)) (hmBad : ∀ y, MeasurableSet (Bad y))
    (Z : Y → B → Finset (FreeMonoid α))
    (hZ : ∀ y b ω ω', (∀ w ∈ Z y b, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) → (ω, b) ∈ Bad y →
      (ω', b) ∈ Bad y)
    (Dj : Y → Set (X × B)) (hmDj : ∀ y, MeasurableSet (Dj y))
    (hDj : ∀ y x b, G y → x ∈ Ok y → (x, b) ∈ Dj y → Disjoint (W y x) (Z y b))
    (δ₁ δ₂ : ℝ≥0∞) (hBad : ∀ y, G y → (μ.prod β) (Bad y) ≤ δ₁)
    (hcoll : ∀ y, G y → ∀ x ∈ Ok y, β {b | (x, b) ∉ Dj y} ≤ δ₂) :
    ((μ.prod ν).prod β) {q | G (stage q.1.1 q.1.2) ∧ q.1.2 ∈ Ok (stage q.1.1 q.1.2)
      ∧ ((q.1.1, q.2) ∈ Bad (stage q.1.1 q.1.2) ∨ (q.1.2, q.2) ∉ Dj (stage q.1.1 q.1.2))}
      ≤ δ₁ + δ₂ := by
  sorry

end Select

section Score

variable (O : Oracle μ (FreeMonoid α)) (H : HypothesisDFA α) (F : Finset (FreeMonoid α))
  (kl kh : ℕ)

open scoped Classical in
/-- `1` where the read of `x` is decided and the run from `s` classifies `x` as it decides. -/
def agreeAt (s : List Bool) (x : FreeMonoid α) (ω : Ω) : ℝ :=
  if familyRead O.mq F kl kh x ω ≠ .undecided
    ∧ H.Classifies s x (familyRead O.mq F kl kh x ω = .accept) then 1 else 0

/-- `agreeAt`'s mean over the noise. -/
def agreeProb (s : List Bool) (x : FreeMonoid α) : ℝ := ∫ ω, agreeAt O H F kl kh s x ω ∂μ

/-- `agreeProb`'s mean over the strings of length `L`. -/
def agreeMean (L : ℕ) (s : List Bool) : ℝ :=
  ∫ x, agreeProb O H F kl kh s x ∂(uniformStrings L)

/-- Distinct held-out strings at which some leaf's score strays `n t` from its mean. -/
def scoreBad (L n : ℕ) (t : ℝ) : Set (Ω × (Fin n → FreeMonoid α)) :=
  {p | Function.Injective p.2 ∧ ∃ s ∈ H.tree.leaves,
    n * (agreeMean O H F kl kh L s + t) ≤ ∑ i, agreeAt O H F kl kh s (p.2 i) p.1
      ∨ ∑ i, agreeAt O H F kl kh s (p.2 i) p.1 ≤ n * (agreeMean O H F kl kh L s - t)}

theorem leafScore_eq (ω : Ω) {n : ℕ} (xs : Fin n → FreeMonoid α) (s : List Bool) :
    (leafScore H (fun z => familyRead O.mq F kl kh z ω) xs s : ℝ)
      = ∑ i, agreeAt O H F kl kh s (xs i) ω := by
  sorry

theorem measurableSet_scoreBad (L n : ℕ) (t : ℝ) :
    MeasurableSet (scoreBad O H F kl kh L n t) := by
  sorry

theorem scoreBad_le [Nonempty α] (hF : SuffixFree F) (L n : ℕ) (t : ℝ) (ht : 0 ≤ t) :
    (μ.prod (Measure.pi fun _ : Fin n => uniformStrings L)) (scoreBad O H F kl kh L n t)
      ≤ ENNReal.ofReal (H.tree.leaves.length * (4 * Real.exp (-(n * t ^ 2 / 2)))) := by
  sorry

end Score

section Pick

variable [Nonempty α] {σ : Type*} (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
  (H : HypothesisDFA α) (F : Finset (FreeMonoid α)) (kl kh : ℕ)

/-- Off `scoreBad`, the leaf of highest score misjudges at most `2t`, the family's undecided
share `u` and twice its misread share `m` more than any leaf. -/
theorem pickLeaf_le (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) (L n : ℕ) (hn : 0 < n)
    (t u m η : ℝ) (ω : Ω) (xs : Fin n → FreeMonoid α) (hxs : Function.Injective xs)
    (hgood : (ω, xs) ∉ scoreBad O H F kl kh L n t)
    (hu : ∫ x, readProb O F kl kh x .undecided ∂(uniformStrings L) ≤ u)
    (hm : ∫ x, misreadProb O F kl kh x ∂(uniformStrings L) ≤ m)
    (hs : ∃ s ∈ H.tree.leaves, (uniformStrings L).real
      {x | ¬ H.Classifies s x (M.eval x.toList ∈ M.accept)} ≤ η) :
    (uniformStrings L).real {x | ¬ H.Classifies
      (pickLeaf H (fun z => familyRead O.mq F kl kh z ω) xs) x (M.eval x.toList ∈ M.accept)}
      ≤ η + 2 * t + u + 2 * m := by
  sorry

end Pick

section Collide

variable [Nonempty α]

/-- Held-out strings uniform of length `L` repeat, or one's first `k` letters fall in `S`, with
chance at most `n |S| / |Σ|^k + n² / |Σ|^L`. -/
theorem batch_collide_le (L k n : ℕ) (hk : k ≤ L) (S : Finset (FreeMonoid α)) :
    (Measure.pi fun _ : Fin n => uniformStrings (α := α) L)
      {b | ¬ Function.Injective b ∨ ∃ i, prefixOf (b i) k ∈ S}
      ≤ ENNReal.ofReal (n * S.card / (Fintype.card α : ℝ) ^ k
        + n ^ 2 / (Fintype.card α : ℝ) ^ L) := by
  sorry

end Collide

end OrthoDFA

end
