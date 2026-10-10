import OrthoDFA.Proofs.TallyPTNoise

/-!
# The noise event's good-read fields

`TallyE` fails only where its records field does or one of the edge, start and middles fields'
tails is exceeded, at the slacks the gates leave.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (G : ReadModel α σ)
  {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  (read : FreeMonoid α → Ω → ARU)

theorem tallyE_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.dist (G.M.eval z.toList) r) (hθ : 0 ≤ G.θ)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (C : TallyCfg) (S L : ℕ)
    {p₀ ρ θg θgs θgpt l : ℝ} (hL : 1 ≤ L) (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hk : ∀ᵐ x ∂D, C.k ≤ x.toList.length) (hp₀ : 0 < p₀)
    (hpmax : ∀ u, D.real {x | prefixOf x C.k = u} ≤ p₀) (hl : 0 ≤ l) (hφe : 0 ≤ C.φe)
    (hφpt : 0 ≤ C.φpt) (hθe : 4 * θg ≤ C.θe)
    (hθg : 2 * ((Fintype.card σ + S + 1) * (3 / 2 * G.θ)) ≤ θg)
    (hθgpt : 2 * ((L + 1) * (Fintype.card σ + S + 1) * (3 / 2 * G.θ)) ≤ θgpt) :
    μ {ω | ¬ TallyE G D C S ρ θg θgs θgpt (read · ω)}
      ≤ μ {ω | ¬ ∀ T edges, G.InClass S T → EdgesInto T edges → ∀ pct,
          D.real {x | ∃ sp, recordBy (fun z => (read z ω).cut) C.k (T, edges) x = some (pct, sp)
            ∧ ¬ G.TrueRec T pct sp} ≤ ρ}
        + ENNReal.ofReal ((classSet (Fintype.card σ + S) : Finset (DTree α)).card
          * ((Fintype.card σ + S + 3) ^ ((Fintype.card σ + S + 2) * Fintype.card α)
            * ((Fintype.card σ + S + 2) * Fintype.card α))
          * Real.exp (-(C.φe * (C.θe / 2 - 2 * θg) / 2
            / (2 * L * p₀ * (1 + 4 * ((Fintype.card σ + S + 1) * (3 / 2 * G.θ)) * 2)))))
        + ENNReal.ofReal ((classSet (Fintype.card σ + S) : Finset (DTree α)).card
          * Real.exp ((Fintype.card σ + S + 1) * (3 / 2 * G.θ) * (Real.exp (l * p₀) - 1) / p₀
            - l * θgs))
        + ENNReal.ofReal ((classSet (Fintype.card σ + S) : Finset (DTree α)).card
          * (Fintype.card σ + S + 3) ^ ((Fintype.card σ + S + 2) * Fintype.card α)
          * Real.exp (-(θgpt * C.φpt / 2 / (2 * p₀ * (1 + 4 * ((L + 1) * (Fintype.card σ + S + 1)
            * (3 / 2 * G.θ)) * 1))))) := by
  have hθg0 : 0 ≤ θg := le_trans (by positivity) hθg
  have hθgpt0 : 0 ≤ θgpt := le_trans (by positivity) hθgpt
  have hE := goodEdge_le G read hmeas hind hlaw hθ D C.k L S hL hlen hk hp₀ hpmax
    (η := C.φe * (C.θe / 2 - 2 * θg) / 2)
    (mul_nonneg (mul_nonneg hφe (by linarith)) (by norm_num)) hθg
  have hS := goodStart_le G read hmeas hind hlaw D C.k (Fintype.card σ + S) (θgs := θgs) hθ hk
    hp₀ hpmax hl
  rw [Nat.cast_add] at hS
  have hP := goodPT_le G read hmeas hind hlaw hθ D C.k L S hlen hk hp₀ hpmax
    (η := θgpt * C.φpt / 2) (by positivity) hθgpt
  refine le_trans (measure_mono ?_) (le_trans (measure_union_le _ _) (add_le_add
    (le_trans (measure_union_le _ _) (add_le_add
      (le_trans (measure_union_le _ _) (add_le_add le_rfl hE)) hS)) hP))
  intro ω hω
  simp only [Set.mem_setOf_eq] at hω
  by_contra hno
  simp only [Set.mem_union, Set.mem_setOf_eq, not_or, not_not] at hno
  obtain ⟨⟨⟨hsp, he⟩, hs⟩, hp⟩ := hno
  exact hω ⟨hsp, fun T edges hT hed e => he T hT edges hed e,
    fun T hT => hs T (inClass_mem G hT), hp⟩

end OrthoDFA
