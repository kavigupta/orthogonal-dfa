import OrthoDFA.StartAtK
import OrthoDFA.Proofs.Hoeffding
import OrthoDFA.Proofs.PassReads

/-!
# `RoundAtK`, `WalkYield` and `SourceSpread`

A decision on a batch against a fixed hypothesis is a Hoeffding tail over the batch.  A refusal's
probes reach the split test or stop at a string the cut cannot place, because the pass keeps its
edges learned: every edge's witness sits at its leaf and its extension at its target.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Batch

/-- A predicate holding on at most `θ − δ` of draws holds on more than `θ` of a batch of `n`
with chance at most `exp(−2nδ²)`. -/
theorem share_gt_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) (n : ℕ) {θ δ : ℝ} (hδ : 0 ≤ δ)
    (h : D.real {x | P x} ≤ θ - δ) :
    (Measure.pi fun _ : Fin n => D).real {b | θ < share b P} ≤ Real.exp (-2 * n * δ ^ 2) := by
  sorry

/-- One holding on at least `ε + δ` of draws holds on at most `ε` of a batch with chance at most
`exp(−2nδ²)`. -/
theorem share_le_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) (n : ℕ) {ε δ : ℝ} (hδ : 0 ≤ δ)
    (h : ε + δ ≤ D.real {x | P x}) :
    (Measure.pi fun _ : Fin n => D).real {b | share b P ≤ ε} ≤ Real.exp (-2 * n * δ ^ 2) := by
  sorry

end Batch

section Learned

variable (K : StageKnobs α) (R : CutReads α)

/-- Every edge's witness sits at the edge's leaf, and its extension by the letter at the edge's
target. -/
def Learned (t : DTree α) (edges : Edges α) : Prop :=
  ∀ p c q y, edges p c = some (q, y) →
    t.sift R.cut y = .inl p ∧ t.sift R.cut (y * FreeMonoid.of c) = .inl q

theorem runPassK_learned (k : ℕ) (seed probes : List (FreeMonoid α)) :
    Learned R (runPassK K R k (initialK K R seed) probes).tree
      (runPassK K R k (initialK K R seed) probes).edges := by
  sorry

/-- A probe whose walk from `k` reaches the end along learned edges and whose sift disagrees
splits a leaf, adds a member, or stops at a string the cut cannot place. -/
theorem seedStep_ne_dropped {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    (hl : Learned R t edges) {k : ℕ} {x : FreeMonoid α} {ps : List (List Bool)}
    (h : kCheck R t edges k x = .disagree ps) :
    seedStep K R t pool edges k x ps ≠ .dropped := by
  sorry

end Learned

theorem prod_real_rect {β γ : Type*} [MeasurableSpace β] [MeasurableSpace γ]
    (μ : Measure β) (ν : Measure γ) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (A : Set β) (B : Set γ) : (μ.prod ν).real (A ×ˢ B) = μ.real A * ν.real B := by
  simp only [measureReal_def, Measure.prod_prod, ENNReal.toReal_mul]

theorem round_at_k_holds : RoundAtK := by
  intro α _ _ K R D _ k nw ng seed probes θw θc ε δ hδ
  simp only []
  set s₀ := initialK K R seed
  set s := runPassK K R k s₀ probes
  have hl : Learned R s.tree s.edges := runPassK_learned K R k seed probes
  set Pw := fun x => (kWalk R s₀.tree s₀.edges k x).isBlocked
  set Pc := fun x => (kCheck R s.tree s.edges k x).isBlocked
  set Pd := fun x => (kCheck R s.tree s.edges k x).isDisagreement
  set νw := Measure.pi fun _ : Fin nw => D
  set νg := Measure.pi fun _ : Fin ng => D
  set A := {bw : Fin nw → FreeMonoid α | θw < share bw Pw ∧ D.real {x | Pw x} < θw - δ}
  set B := {bg : Fin ng → FreeMonoid α | θc < share bg Pc ∧ D.real {x | Pc x} < θc - δ}
  set C := {bg : Fin ng → FreeMonoid α | share bg Pd ≤ ε ∧ ε + δ < D.real {x | Pd x}}
  have hsub : {b : (Fin nw → FreeMonoid α) × (Fin ng → FreeMonoid α) |
      ¬ RoundAtKHolds K R s₀.tree s₀.edges s D k θw θc ε δ b.1 b.2}
      ⊆ (A ×ˢ Set.univ ∪ Set.univ ×ˢ B) ∪ Set.univ ×ˢ C := by
    rintro ⟨bw, bg⟩ hb
    simp only [Set.mem_setOf_eq, RoundAtKHolds] at hb
    split_ifs at hb with h1 h2 h3
    · exact .inl (.inl ⟨⟨h1, not_le.1 hb⟩, trivial⟩)
    · exact .inl (.inr ⟨trivial, h2, not_le.1 hb⟩)
    · exact .inr ⟨trivial, h3, not_le.1 hb⟩
    · push_neg at hb
      obtain ⟨i, ps, hc, hd⟩ := hb
      exact absurd hd (seedStep_ne_dropped K R hl hc)
  have hA : νw.real A ≤ Real.exp (-2 * nw * δ ^ 2) := by
    by_cases h : D.real {x | Pw x} < θw - δ
    · refine le_trans (measureReal_mono (fun b hb => hb.1)) (share_gt_le D Pw nw hδ h.le)
    · rw [show A = ∅ from Set.eq_empty_of_forall_notMem fun b hb => h hb.2]
      simp [Real.exp_nonneg]
  have hB : νg.real B ≤ Real.exp (-2 * ng * δ ^ 2) := by
    by_cases h : D.real {x | Pc x} < θc - δ
    · refine le_trans (measureReal_mono (fun b hb => hb.1)) (share_gt_le D Pc ng hδ h.le)
    · rw [show B = ∅ from Set.eq_empty_of_forall_notMem fun b hb => h hb.2]
      simp [Real.exp_nonneg]
  have hC : νg.real C ≤ Real.exp (-2 * ng * δ ^ 2) := by
    by_cases h : ε + δ < D.real {x | Pd x}
    · refine le_trans (measureReal_mono (fun b hb => hb.1)) (share_le_le D Pd ng hδ h.le)
    · rw [show C = ∅ from Set.eq_empty_of_forall_notMem fun b hb => h hb.2]
      simp [Real.exp_nonneg]
  have h1 := measureReal_union_le (μ := νw.prod νg) (A ×ˢ Set.univ ∪ Set.univ ×ˢ B)
    (Set.univ ×ˢ C)
  have h2 := measureReal_union_le (μ := νw.prod νg) (A ×ˢ Set.univ) (Set.univ ×ˢ B)
  rw [prod_real_rect, prod_real_rect] at h2
  rw [prod_real_rect] at h1
  simp only [probReal_univ, mul_one, one_mul] at h1 h2
  refine (measureReal_mono hsub).trans ?_
  linarith

theorem walk_yield_holds : WalkYield := by
  sorry

theorem source_spread_holds : SourceSpread := by
  sorry

end OrthoDFA
