import OrthoDFA.Code.CoveredStart
import Mathlib.Computability.DFA

/-!
# A hypothesis DFA from a covered start

`CoveredStartCorrect`: where every state of `S` has its edges learned and true and is read on its
true side, the hypothesis from a state's true leaf misjudges at most what re-rooting the target at
that state does.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The law of a string's first `n` letters, the string drawn from `D`. -/
def prefixLaw (D : Measure (FreeMonoid α)) (n : ℕ) : Measure (FreeMonoid α) :=
  D.map fun x => prefixOf x n

/-- Uniform over the strings of length `n`. -/
def uniformStrings (n : ℕ) : Measure (FreeMonoid α) :=
  ((Fintype.card α : ℝ≥0∞) ^ n)⁻¹
    • ∑ v : List.Vector α n, Measure.dirac (FreeMonoid.ofList v.toList)

/-- The share of `T`'s strings on which the target re-rooted at `q` stays in `S` for `L` letters and
classifies as the target does. -/
def rerootAccuracyOn {σ : Type*} (T : Measure (FreeMonoid α)) (M : _root_.DFA α σ) (L : ℕ)
    (S : Set σ) (q : σ) : ℝ :=
  T.real {x : FreeMonoid α | (∀ i ≤ L, M.evalFrom q (prefixOf x i).toList ∈ S)
    ∧ (M.evalFrom q x.toList ∈ M.accept ↔ M.eval x.toList ∈ M.accept)}

/-- `rerootAccuracyOn` the strings of length `L`. -/
def rerootAccuracy {σ : Type*} (M : _root_.DFA α σ) (L : ℕ) (S : Set σ) (q : σ) : ℝ :=
  rerootAccuracyOn (uniformStrings L) M L S q

namespace Random

variable {σ : Type*} (M : _root_.DFA α σ) (side : σ → Bool)

/-- The leaf a state's strings sift to where no read on the way is undecided or wrong. -/
def leafOf : DTree α → σ → List Bool
  | .leaf, _ => []
  | .node m r a, q =>
    if side (M.evalFrom q m.toList) then true :: leafOf a q else false :: leafOf r q

end Random

open scoped Classical in
def CoveredStartCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} (M : _root_.DFA α σ)
    (side : σ → Bool) (H : HypothesisDFA α), (∃ r a, H.tree = .node 1 r a) →
    ∀ (S : Set σ),
    (∀ q ∈ S, ∀ a, ∃ w, H.edges (Random.leafOf M side H.tree q, a)
      = some (Random.leafOf M side H.tree (M.step q a), w)) →
    (∀ q ∈ S, side q = decide (q ∈ M.accept)) → ∀ (L : ℕ) (q₀ : σ) {γ : ℝ},
    1 - γ ≤ rerootAccuracy M L S q₀ →
    (uniformStrings L).real {x | ¬ H.Classifies (Random.leafOf M side H.tree q₀) x
      (M.eval x.toList ∈ M.accept)} ≤ γ

end OrthoDFA

end
