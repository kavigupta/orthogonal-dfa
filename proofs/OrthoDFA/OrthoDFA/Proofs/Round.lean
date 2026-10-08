import OrthoDFA.Round
import OrthoDFA.Proofs.Replay

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
variable {μ : Measure Ω} {Q : Type*}

/-- Believed true: `RoundOutcome`'s yield gives `d ≤ L · P(harvest)`, its harvest spread gives
`P(t harvested) ≤ ∑ min(i+1,L)/L · D(first i letters are t's)`, and for `i ≤ |t|` that last chance
is `L · ν(t's first i letters) ≤ L · κ · V(its state)`. -/
theorem harvestSpread_of (A : DFA (FreeMonoid α) Q) (R : CutReads α) (H : Hypothesis α)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (L : ℕ) (κ : ℝ)
    (hb : R.B.lo ≤ R.B.hi)
    (hκ : ∀ y, prefixWeight D L y ≤ κ * stateWeight A D L (A.state y)) :
    HarvestSpread A R H D L κ := by
  sorry

variable (R : CutReads α)

omit [Fintype α] [DecidableEq α] in
theorem le_length_of_suffixDisagree (H : Hypothesis α) {x : FreeMonoid α} {e : ℕ}
    (h : SuffixDisagree R H x e) : e ≤ x.toList.length := by
  by_contra hlt
  apply h
  have hle : x.toList.length ≤ e := by omega
  have hp : prefixOf x e = x := by
    simp [prefixOf, List.take_of_length_le hle]
  rw [List.drop_of_length_le hle, hp]
  rfl

/-- A replay anchored no earlier than `e` harvests when the draw, walked from where the
middle-of-band reading puts its first `e` letters, disagrees with the tree. -/
theorem replay_harvests_of_suffix (H : Hypothesis α) (hb : R.B.lo ≤ R.B.hi) {x : FreeMonoid α}
    {e : ℕ} (hd : SuffixDisagree R H x e) : (replay R H x e).2 ≠ [] := by
  have he := le_length_of_suffixDisagree R H hd
  obtain ⟨rest, hrest⟩ := filter_range_ge he
  rcases hε : H.tree.sift R.cut (prefixOf x e) with p | b
  · have hp : midPath R H (prefixOf x e) = p := midPath_of_sift R H hb hε
    have hs := anchorSearch_cons_of_sift R H.tree (is := rest) hε
    have hend : ((x.toList.drop e).scanl H.step p).getD (x.toList.length - e) []
        = (x.toList.drop e).foldl H.step p := by
      rw [← List.length_drop, scanl_getD_length]
    rcases hx : H.tree.sift R.cut x with actual | b
    · have ha : actual ≠ (x.toList.drop e).foldl H.step p := by
        intro h
        apply hd
        rw [midPath_of_sift R H hb hx, hp, h]
      simp only [replay, hrest, hs, hx, hend, ha, if_false]
      split <;> simp
    · simp [replay, hrest, hs, hx]
  · obtain ⟨l, hl⟩ := replay_snd R H x e
    rw [hl, hrest]
    simp [anchorSearch, hε]

/-- An anchor at which a disagreeing draw is still in sync walks on as the draw's walk from `ε`
does, so it ends in the same disagreement. -/
theorem suffixDisagree_of_inSync (H : Hypothesis α) {x : FreeMonoid α} {e : ℕ}
    (hs : InSync R H x e) (hd : DFAandDTDisagree R H x) : SuffixDisagree R H x e := by
  intro h
  apply hd
  rw [← List.take_append_drop e x.toList, List.foldl_append, hs]
  exact h

theorem prod_sum_dirac (D : Measure (FreeMonoid α)) [SFinite D] {S : Set (FreeMonoid α × ℕ)}
    (hS : MeasurableSet S) (n : ℕ) :
    (D.prod (∑ k ∈ Finset.range n, Measure.dirac k)) S
      = ∑ k ∈ Finset.range n, D ((fun x => (x, k)) ⁻¹' S) := by
  induction n with
  | zero => simp
  | succ n ih =>
    have hm : Measurable fun x : FreeMonoid α => (x, n) := measurable_id.prodMk measurable_const
    rw [Finset.sum_range_succ, Measure.prod_add, Measure.add_apply, ih, Finset.sum_range_succ,
      Measure.prod_dirac, Measure.map_apply hm hS]

theorem anchored_yield_holds : AnchoredYield := by
  intro α _ _ R H D _ L hb
  set S := {q : FreeMonoid α × ℕ | (replay R H q.1 q.2).2 ≠ []}
  have hS : MeasurableSet S := S.to_countable.measurableSet
  have hsum : (D.prod (anchorLaw L)) S
      = (L : ℝ≥0∞)⁻¹ * ∑ k ∈ Finset.range L, D ((fun x => (x, k)) ⁻¹' S) := by
    rw [anchorLaw, Measure.prod_smul_right, Measure.smul_apply, prod_sum_dirac D hS, smul_eq_mul]
  have hle : ∀ k, D {x | SuffixDisagree R H x k} ≤ D ((fun x => (x, k)) ⁻¹' S) := fun k =>
    measure_mono fun x hx => replay_harvests_of_suffix R H hb hx
  have hsumle : ∑ e ∈ Finset.range L, D {x | SuffixDisagree R H x e}
      ≤ ∑ k ∈ Finset.range L, D ((fun x => (x, k)) ⁻¹' S) :=
    Finset.sum_le_sum fun k _ => hle k
  have htop : ∀ k, D {x | SuffixDisagree R H x k} ≠ ⊤ := fun _ => measure_ne_top _ _
  rw [measureReal_def, hsum, div_eq_inv_mul, ENNReal.toReal_mul, ENNReal.toReal_inv,
    ENNReal.toReal_natCast]
  simp only [measureReal_def]
  rw [← ENNReal.toReal_sum fun k _ => htop k]
  gcongr
  · exact ENNReal.sum_ne_top.2 fun _ _ => measure_ne_top _ _

theorem inSync_yield_holds : InSyncYield := by
  intro α _ _ R H D _ L hb
  refine le_trans ?_ (anchored_yield_holds R H D L hb)
  apply div_le_div_of_nonneg_right _ (Nat.cast_nonneg L)
  exact Finset.sum_le_sum fun e _ =>
    measureReal_mono (fun x hx => suffixDisagree_of_inSync R H hx.1 hx.2)

end OrthoDFA
