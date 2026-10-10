import OrthoDFA.Proofs.RandomTree
import OrthoDFA.Proofs.RandomCounts

/-!
# One stretch

A step is bad where, within the first `Ns` probes of its stretch, it ends the round badly, or
where its stretch reaches `Ns` probes. With the reads fixed, the hypothesis fixed over a stretch
and the probes fresh, a stretch's counts are binomial: `seg_le` bounds the chance that a stretch
has a bad step by `stretchRisk`.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory
open OrthoDFA.Ideal (DTree Disagrees)

section Segments

variable {S X E : Type*} (step : S → X → S ⊕ E) (B : S → X → Prop) (same : S → S → Prop)

/-- Some step before the run leaves the segment is bad. -/
def SegB : S → List X → Prop
  | _, [] => False
  | s, x :: xs => B s x ∨ ∃ s', step s x = .inl s' ∧ same s s' ∧ SegB s' xs

/-- Some step of the run is bad. -/
def AnyB : S → List X → Prop
  | _, [] => False
  | s, x :: xs => B s x ∨ ∃ s', step s x = .inl s' ∧ AnyB s' xs

end Segments

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Cfg)
  {σ : Type*} [Fintype σ] (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) (D : Measure (FreeMonoid α))
  (ε : ℝ) (Ns : ℕ)

def Same (s s' : RState α) : Prop := s'.tree = s.tree ∧ s'.edges = s.edges

/-- Within the first `Ns` probes of its stretch the step ends the round badly, or its stretch
reaches `Ns` probes. -/
def BadStep (s : RState α) (x : FreeMonoid α) : Prop :=
  (s.n < Ns ∧ ∃ e, step read C s x = .inr e ∧ ¬ EndsWell M (BadAt U θ) D ε
      {y | Disagrees read s.tree s.edges C.k y} (toRoundEnd (some e)))
  ∨ ∃ s', step read C s x = .inl s' ∧ Same s s' ∧ s'.n = Ns

theorem seg_le {side : σ → Bool} (hW : NoWrong M side read) [IsProbabilityMeasure D]
    (L N₁ : ℕ) (G θr εd' θpt' : ℝ) (hL : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hN : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
      D.real (PotGood M U θ C.k read T) ≤ G)
    (hcap : Fintype.card σ + 2 ≤ C.Lmax) (hm : 1 ≤ C.m) (hn₀ : 1 ≤ C.n₀) (hN₁ : C.n₀ ≤ N₁)
    (hN₁s : N₁ ≤ Ns) (hθ : 0 ≤ θ) (hG0 : 0 ≤ G) (hG1 : G ≤ 1) (ha : 0 ≤ C.a)
    (hθs0 : 0 ≤ C.θs) (hθs1 : C.θs ≤ 1) (hθe : 0 ≤ C.θe) (hθpt0 : 0 ≤ C.θpt)
    (hθpt1 : C.θpt ≤ 1) (hεd0 : 0 ≤ C.εd) (hεd1 : C.εd ≤ 1) (hεd : C.εd ≤ ε) (hθr0 : 0 ≤ θr)
    (hθr1 : θr ≤ 1) (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1) (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1)
    (hsep : (C.Lmax : ℝ) ^ 2 * Fintype.card α * θr ≤ (1 - θpt') * εd')
    {s : RState α} (hs : Inv read C s) (hf : s = fresh s.tree s.edges s.moved) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D)
        {xs | SegB (step read C) (BadStep read C M U θ D ε Ns) Same s (List.ofFn xs)}
      ≤ ENNReal.ofReal (stretchRisk C L Ns N₁ G θr εd' θpt') := by
  sorry

end Random

end OrthoDFA
