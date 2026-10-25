import OrthoDFA.LoopDFATSR
import OrthoDFA.Proofs.DFABridge

/-!
# Typical-start replication: the re-rooted run rarely leaves the covered states

`window_mul_hit_le`: the first visit to `q` from `⌈L/2⌉` to `L` depends only on the prefix up to
it, and the next `L` letters are uniform and independent of it, so a run from `q` that hits `B` is
a run from `M`'s start that hits `B` from `⌈L/2⌉` to `2L`:

    w(q) · Pr_{|x|=L}[the run from q hits B]
      ≤ Pr_{|x|=2L}[the run from the start hits B in [⌈L/2⌉, 2L]].

At `B` the states below `c`, the re-rooted run leaves `coveredStates M L c` on at most `Φ(c)/w(q)`
of the strings (`reroot_accuracy_of_tsr`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α]

omit [DecidableEq α] in
theorem uavg_mono (ℓ : ℕ) {g h : FreeMonoid α → ℝ} (hgh : ∀ x, g x ≤ h x) :
    uavg ℓ g ≤ uavg ℓ h :=
  div_le_div_of_nonneg_right (Finset.sum_le_sum fun _ _ => hgh _) (by positivity)

/-- A uniform string's prefix of length `j` and the rest are independent. -/
theorem uavg_prefix_mul_drop (j t : ℕ) (f g : FreeMonoid α → ℝ) :
    uavg (j + t) (fun x => f (prefixOf x j) * g (FreeMonoid.ofList (x.toList.drop j)))
      = uavg j f * uavg t g := by
  rw [uavg_split]
  have e1 : ∀ (w : List.Vector α j) (z : List.Vector α t),
      prefixOf (FreeMonoid.ofList (w.toList ++ z.toList)) j = FreeMonoid.ofList w.toList := by
    intro w z
    simp [prefixOf]
  have e2 : ∀ (w : List.Vector α j) (z : List.Vector α t),
      FreeMonoid.ofList ((FreeMonoid.ofList (w.toList ++ z.toList)).toList.drop j)
        = FreeMonoid.ofList z.toList := by
    intro w z
    simp
  simp only [e1, e2]
  unfold uavg
  rw [pow_add, ← Finset.sum_mul_sum, mul_div_mul_comm]

omit [Fintype α] [DecidableEq α] [Nonempty α] in
theorem prefixOf_prefixOf (x : FreeMonoid α) {i j : ℕ} (h : i ≤ j) :
    prefixOf (prefixOf x j) i = prefixOf x i := by
  simp [prefixOf, List.take_take, min_eq_left h]

theorem window_mul_hit_le {σ : Type*} (M : _root_.DFA α σ) (L : ℕ) (q : σ) (B : Set σ) :
    windowVisitRate M L q
        * (uniformStrings L).real {x | ∃ i ≤ L, M.evalFrom q (prefixOf x i).toList ∈ B}
      ≤ (uniformStrings (2 * L)).real
        {x | ∃ j, (L + 1) / 2 ≤ j ∧ j ≤ 2 * L ∧ M.eval (prefixOf x j).toList ∈ B} := by
  classical
  set k := (L + 1) / 2
  -- `F j y`: `y`'s run is at `q` at `j`, and not from `k` to before `j`.
  let F : ℕ → FreeMonoid α → Prop := fun j y =>
    M.eval y.toList = q ∧ ∀ i, k ≤ i → i < j → M.eval (prefixOf y i).toList ≠ q
  let f : ℕ → FreeMonoid α → ℝ := fun j => {y | F j y}.indicator 1
  let H : Set (FreeMonoid α) := {z | ∃ i ≤ L, M.evalFrom q (prefixOf z i).toList ∈ B}
  let E : Set (FreeMonoid α) :=
    {x | ∃ j, k ≤ j ∧ j ≤ 2 * L ∧ M.eval (prefixOf x j).toList ∈ B}
  have hf0 : ∀ j y, 0 ≤ f j y := fun j y => Set.indicator_nonneg (fun _ _ => zero_le_one) _
  have hf1 : ∀ j y, f j y ≤ 1 := by
    intro j y
    simp only [f, Set.indicator_apply]
    split_ifs <;> norm_num
  have hH1 : ∀ z, H.indicator (1 : FreeMonoid α → ℝ) z ≤ 1 := by
    intro z
    simp only [Set.indicator_apply]
    split_ifs <;> norm_num
  have huniq : ∀ x {j j'}, k ≤ j → k ≤ j' → F j (prefixOf x j) → F j' (prefixOf x j') → j = j' := by
    intro x j j' hj hj' h h'
    by_contra hne
    rcases Nat.lt_or_gt_of_ne hne with hlt | hlt
    · exact h'.2 j hj hlt (by rw [prefixOf_prefixOf x hlt.le]; exact h.1)
    · exact h.2 j' hj' hlt (by rw [prefixOf_prefixOf x hlt.le]; exact h'.1)
  have hw : windowVisitRate M L q ≤ ∑ j ∈ Finset.Icc k L, uavg j (f j) := by
    rw [windowVisitRate, uniformStrings_real_uavg]
    rw [Finset.sum_congr rfl fun j hj => (uavg_prefix (Finset.mem_Icc.1 hj).2 (f j)).symm,
      ← uavg_sum]
    let W : Set (FreeMonoid α) := {x | ∃ j, k ≤ j ∧ j ≤ L ∧ M.eval (prefixOf x j).toList = q}
    change uavg L (W.indicator 1) ≤ _
    refine uavg_mono L fun x => ?_
    by_cases hW : x ∈ W
    · rw [Set.indicator_of_mem hW, Pi.one_apply]
      have hx : ∃ j, k ≤ j ∧ j ≤ L ∧ M.eval (prefixOf x j).toList = q := hW
      have hs := Nat.find_spec hx
      have hF : F (Nat.find hx) (prefixOf x (Nat.find hx)) := by
        refine ⟨hs.2.2, fun i hi hlt => ?_⟩
        rw [prefixOf_prefixOf x hlt.le]
        intro heq
        exact Nat.find_min hx hlt ⟨hi, by omega, heq⟩
      have h1 : f (Nat.find hx) (prefixOf x (Nat.find hx)) = 1 := by
        simp only [f, Set.indicator_of_mem (show prefixOf x (Nat.find hx) ∈ {y | F _ y} from hF),
          Pi.one_apply]
      rw [← h1]
      exact Finset.single_le_sum (f := fun j => f j (prefixOf x j)) (fun j _ => hf0 _ _)
        (Finset.mem_Icc.2 ⟨hs.1, hs.2.1⟩)
    · rw [Set.indicator_of_notMem hW]
      exact Finset.sum_nonneg fun j _ => hf0 _ _
  have hHL : ∀ z, prefixOf z L ∈ H ↔ z ∈ H := by
    intro z
    constructor
    · rintro ⟨i, hi, h⟩
      exact ⟨i, hi, by rwa [prefixOf_prefixOf z hi] at h⟩
    · rintro ⟨i, hi, h⟩
      exact ⟨i, hi, by rwa [prefixOf_prefixOf z hi]⟩
  have hprod : ∀ j ∈ Finset.Icc k L, uavg j (f j) * uavg L (H.indicator 1)
      = uavg (2 * L) (fun x => f j (prefixOf x j)
          * H.indicator 1 (FreeMonoid.ofList (x.toList.drop j))) := by
    intro j hj
    have hjL := (Finset.mem_Icc.1 hj).2
    rw [show 2 * L = j + (2 * L - j) by omega, uavg_prefix_mul_drop,
      ← uavg_prefix (show L ≤ 2 * L - j by omega) (H.indicator 1)]
    congr 2
    funext z
    simp only [Set.indicator_apply, hHL, Pi.one_apply]
  have hpt : ∀ x, ∑ j ∈ Finset.Icc k L, f j (prefixOf x j)
      * H.indicator 1 (FreeMonoid.ofList (x.toList.drop j)) ≤ E.indicator 1 x := by
    intro x
    by_cases hx : ∃ j ∈ Finset.Icc k L, F j (prefixOf x j)
        ∧ FreeMonoid.ofList (x.toList.drop j) ∈ H
    · obtain ⟨j₀, hj₀, hF, i, hi, hB⟩ := hx
      have hjk := (Finset.mem_Icc.1 hj₀).1
      have hjL := (Finset.mem_Icc.1 hj₀).2
      have hE : x ∈ E := by
        refine ⟨j₀ + i, by omega, by omega, ?_⟩
        have : (prefixOf x (j₀ + i)).toList = (prefixOf x j₀).toList
            ++ (prefixOf (FreeMonoid.ofList (x.toList.drop j₀)) i).toList := by
          simp [prefixOf, List.take_add]
        rw [this, DFA.eval, DFA.evalFrom_of_append, ← DFA.eval, hF.1]
        exact hB
      rw [Set.indicator_of_mem hE, Pi.one_apply,
        Finset.sum_eq_single_of_mem j₀ hj₀ fun j hj hne => ?_]
      · exact mul_le_one₀ (hf1 _ _) (Set.indicator_nonneg (fun _ _ => zero_le_one) _) (hH1 _)
      · have : ¬ F j (prefixOf x j) := fun h =>
          hne (huniq x (Finset.mem_Icc.1 hj).1 hjk h hF)
        simp only [f, Set.indicator_of_notMem (show prefixOf x j ∉ {y | F j y} from this),
          zero_mul]
    · push Not at hx
      refine (Finset.sum_nonpos fun j hj => ?_).trans
        (Set.indicator_nonneg (fun _ _ => zero_le_one) _)
      by_cases hF : F j (prefixOf x j)
      · rw [Set.indicator_of_notMem (hx j hj hF), mul_zero]
      · simp only [f, Set.indicator_of_notMem (show prefixOf x j ∉ {y | F j y} from hF),
          zero_mul, le_refl]
  have hh0 : 0 ≤ (uniformStrings L).real H := measureReal_nonneg
  calc windowVisitRate M L q * (uniformStrings L).real H
      ≤ (∑ j ∈ Finset.Icc k L, uavg j (f j)) * (uniformStrings L).real H :=
        mul_le_mul_of_nonneg_right hw hh0
    _ = ∑ j ∈ Finset.Icc k L, uavg (2 * L) (fun x => f j (prefixOf x j)
          * H.indicator 1 (FreeMonoid.ofList (x.toList.drop j))) := by
        rw [Finset.sum_mul, uniformStrings_real_uavg]
        exact Finset.sum_congr rfl hprod
    _ = uavg (2 * L) (fun x => ∑ j ∈ Finset.Icc k L, f j (prefixOf x j)
          * H.indicator 1 (FreeMonoid.ofList (x.toList.drop j))) := (uavg_sum _ _ _).symm
    _ ≤ uavg (2 * L) (E.indicator 1) := uavg_mono _ hpt
    _ = (uniformStrings (2 * L)).real E := (uniformStrings_real_uavg _ _).symm

/-- Corollary A: the re-rooted run errs `e(q)` and leaves the covered states `Φ(c)/w(q)` of the
time. -/
theorem reroot_accuracy_of_tsr {σ : Type*} (M : _root_.DFA α σ) (L : ℕ) {c : ℝ} (hc : 0 < c)
    {q : σ} (hq : c ≤ windowVisitRate M L q) :
    1 - (rerootError M L q + thinVisitRate M L c / windowVisitRate M L q)
      ≤ rerootAccuracy M L (coveredStates M L c) q := by
  set w := windowVisitRate M L q
  have hw : 0 < w := lt_of_lt_of_le hc hq
  let μL : Measure (FreeMonoid α) := uniformStrings L
  let G : Set (FreeMonoid α) :=
    {x | (∀ i ≤ L, M.evalFrom q (prefixOf x i).toList ∈ coveredStates M L c)
      ∧ (M.evalFrom q x.toList ∈ M.accept ↔ M.eval x.toList ∈ M.accept)}
  let D : Set (FreeMonoid α) :=
    {x | ¬ (M.evalFrom q x.toList ∈ M.accept ↔ M.eval x.toList ∈ M.accept)}
  let T : Set (FreeMonoid α) :=
    {x | ∃ i ≤ L, M.evalFrom q (prefixOf x i).toList ∈ {q' | windowVisitRate M L q' < c}}
  have hT : μL.real T ≤ thinVisitRate M L c / w := by
    rw [le_div_iff₀ hw, mul_comm]
    exact window_mul_hit_le M L q {q' | windowVisitRate M L q' < c}
  have hcov : Set.univ ⊆ G ∪ D ∪ T := by
    intro x _
    by_cases hD : x ∈ D
    · exact Or.inl (Or.inr hD)
    by_cases hT' : x ∈ T
    · exact Or.inr hT'
    refine Or.inl (Or.inl ⟨fun i hi => ?_, not_not.1 hD⟩)
    change c ≤ windowVisitRate M L _
    by_contra hlt
    exact hT' ⟨i, hi, not_le.1 hlt⟩
  have h1 : (1 : ℝ) ≤ μL.real (G ∪ D ∪ T) := by
    rw [← probReal_univ (μ := μL)]
    exact measureReal_mono hcov
  have h2 := measureReal_union_le (μ := μL) (G ∪ D) T
  have h3 := measureReal_union_le (μ := μL) G D
  have hG : μL.real G = rerootAccuracy M L (coveredStates M L c) q := rfl
  have hDe : μL.real D = rerootError M L q := rfl
  linarith

end OrthoDFA

end
