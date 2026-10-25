import OrthoDFA.LoopDFATSRB
import OrthoDFA.Proofs.LoopDFA

/-!
# Typical-start replication with probes to twice the length: the bridge

A string `y` of length `i` at a state `q` with a wrong edge at `a q` is disagreeing, has `y·a q`
disagreeing, or is misled (`edge_of_pair`); summed over the wrong-edge states this is at most
`U_i(Bad) + |Σ| U_{i+1}(Bad)` plus the misled ones, at most half of each state's plus `β'` each
(`wrongEdge_occupancy_le`). A string at a state read on its wrong side has
`2·misread + undecided ≥ 1` (`wrongSide_mass_le`).

The first visit to `q` from `⌈L/2⌉` to `L` depends only on the prefix up to it, and the next `L`
letters are uniform and independent of it (`window_mul_hit_le`):

    w(q) · Pr_{|x|=L}[the run from q hits B] ≤ Pr_{|x|=2L}[the run from the start hits B in [⌈L/2⌉, 2L]].

At `B` the wrong states, the right side is at most their occupancy of the lengths `⌈L/2⌉` to `2L`,
all inside the probes' window (`dfa_of_checks_tsrb`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Summed

theorem uavg_mono {ℓ : ℕ} {g h : FreeMonoid α → ℝ}
    (hgh : ∀ v : List.Vector α ℓ, g (FreeMonoid.ofList v.toList) ≤ h (FreeMonoid.ofList v.toList)) :
    uavg ℓ g ≤ uavg ℓ h :=
  div_le_div_of_nonneg_right (Finset.sum_le_sum fun v _ => hgh v) (by positivity)

theorem uavg_add (ℓ : ℕ) (g h : FreeMonoid α → ℝ) :
    uavg ℓ (fun x => g x + h x) = uavg ℓ g + uavg ℓ h := by
  unfold uavg
  rw [Finset.sum_add_distrib, add_div]

variable [Nonempty α] {σ : Type*} [Fintype σ] (M : _root_.DFA α σ) (dist : σ → ARU → ℝ)
  (rd : FreeMonoid α → ARU)

/-- The states of `W`, each with a wrong or unlearned edge at `a q`, occupy the window's lengths
`k` to `J - 1` at most `2N((|Σ|+1)β + |W|β')` in all. -/
theorem wrongEdge_occupancy_le (H : HypothesisDFA α) {k J L₁ : ℕ} (hkJ : k < J) (hJ : J ≤ L₁)
    (W : Finset σ) (a : σ → α)
    (hW : ∀ q ∈ W, ¬ ∃ w, H.edges (Random.leafOf M (fun q => readSide (dist q)) H.tree q, a q)
      = some (Random.leafOf M (fun q => readSide (dist q)) H.tree (M.step q (a q)), w))
    {β β' : ℝ} (hbad : (windowStrings k L₁).real (disagreeSet rd k H) ≤ β)
    (hmis : ∀ q, (windowStrings k L₁).real (shortAt M J q ∩ misledSet M dist rd k H.tree (a q))
      ≤ (windowStrings k L₁).real (shortAt M J q) / 2 + β') :
    ∑ q ∈ W, ∑ i ∈ Finset.Ico k J, (uniformStrings i).real {y | M.eval y.toList = q}
      ≤ 2 * (((Fintype.card α : ℝ) + 1) * ((L₁ : ℝ) + 1 - k) * β
        + W.card * (((L₁ : ℝ) + 1 - k) * β')) := by
  classical
  set Bad := disagreeSet rd k H
  set Mis : σ → Set (FreeMonoid α) := fun q => shortAt M J q ∩ misledSet M dist rd k H.tree (a q)
  set A : ℝ := (Fintype.card α : ℝ) with hA
  have hkL : k ≤ L₁ := by omega
  have hU0 : ∀ i (S : Set (FreeMonoid α)), 0 ≤ (uniformStrings (α := α) i).real S :=
    fun _ _ => measureReal_nonneg
  have hnull : ∀ i, (uniformStrings (α := α) i).real {y | y.toList.length ≠ i} = 0 := fun i => by
    rw [measureReal_def, uniformStrings_length, ENNReal.toReal_zero]
  -- at each length, pointwise
  have hi : ∀ i ∈ Finset.Ico k J, ∑ q ∈ W, (uniformStrings i).real {y | M.eval y.toList = q}
      ≤ (uniformStrings i).real Bad + A * (uniformStrings (i + 1)).real Bad
        + ∑ q ∈ W, (uniformStrings i).real (Mis q) := by
    intro i hi
    simp only [Finset.mem_Ico] at hi
    set g : FreeMonoid α → ℝ := fun y =>
      ∑ q ∈ W, {y : FreeMonoid α | M.eval y.toList = q}.indicator 1 y with hgd
    set h : FreeMonoid α → ℝ := fun y => Bad.indicator 1 y
      + ∑ z : List.Vector α 1, Bad.indicator (1 : FreeMonoid α → ℝ) (y * FreeMonoid.ofList z.toList)
      + ∑ q ∈ W, (Mis q).indicator 1 y with hhd
    have hpt : ∀ v : List.Vector α i, g (FreeMonoid.ofList v.toList)
        ≤ h (FreeMonoid.ofList v.toList) := by
      intro v
      simp only [hgd, hhd]
      set y := FreeMonoid.ofList v.toList
      have hy : y.toList.length = i := by simp [y]
      have hnn : ∀ (S : Set (FreeMonoid α)) (x : FreeMonoid α),
          0 ≤ S.indicator (1 : FreeMonoid α → ℝ) x :=
        fun S x => Set.indicator_nonneg (fun _ _ => zero_le_one) x
      have hz0 : 0 ≤ ∑ z : List.Vector α 1, Bad.indicator (1 : FreeMonoid α → ℝ)
          (y * FreeMonoid.ofList z.toList) := Finset.sum_nonneg fun z _ => hnn _ _
      have hm0 : 0 ≤ ∑ q ∈ W, (Mis q).indicator (1 : FreeMonoid α → ℝ) y :=
        Finset.sum_nonneg fun q _ => hnn _ _
      have hb0 := hnn Bad y
      have hlhs : ∑ q ∈ W,
          {y : FreeMonoid α | M.eval y.toList = q}.indicator (1 : FreeMonoid α → ℝ) y
          = if M.eval y.toList ∈ W then 1 else 0 := by
        simp only [Set.indicator_apply, Set.mem_setOf_eq, Pi.one_apply]
        exact Finset.sum_ite_eq W (M.eval y.toList) fun _ => (1 : ℝ)
      show ∑ q ∈ W, {y : FreeMonoid α | M.eval y.toList = q}.indicator (1 : FreeMonoid α → ℝ) y ≤ _
      rw [hlhs]
      split_ifs with hq
      swap
      · linarith
      set q := M.eval y.toList
      by_cases hyB : y ∈ Bad
      · rw [Set.indicator_of_mem hyB]; simp only [Pi.one_apply]; linarith
      by_cases hyaB : y * FreeMonoid.of (a q) ∈ Bad
      · have h1 : (1 : ℝ) ≤ ∑ z : List.Vector α 1, Bad.indicator (1 : FreeMonoid α → ℝ)
            (y * FreeMonoid.ofList z.toList) := by
          refine le_trans (le_of_eq ?_) (Finset.single_le_sum (f := fun z : List.Vector α 1 =>
            Bad.indicator (1 : FreeMonoid α → ℝ) (y * FreeMonoid.ofList z.toList))
            (fun z _ => hnn _ _) (Finset.mem_univ (a q ::ᵥ List.Vector.nil)))
          rw [Set.indicator_of_mem (by simpa using hyaB)]; rfl
        linarith
      by_cases hyM : y ∈ Mis q
      · have h1 : (1 : ℝ) ≤ ∑ q' ∈ W, (Mis q').indicator (1 : FreeMonoid α → ℝ) y := by
          refine le_trans (le_of_eq ?_) (Finset.single_le_sum (f := fun q' =>
            (Mis q').indicator (1 : FreeMonoid α → ℝ) y) (fun q' _ => hnn _ _) hq)
          rw [Set.indicator_of_mem hyM]; rfl
        linarith
      exfalso
      have hm : y ∉ misledSet M dist rd k H.tree (a q) := fun h =>
        hyM ⟨⟨by omega, rfl⟩, h⟩
      obtain ⟨w, hw⟩ := edge_of_pair M dist rd H (a q) (by omega) hyB hyaB hm
      rw [FreeMonoid.toList_mul, FreeMonoid.toList_of, DFA.eval_append_singleton] at hw
      exact hW q hq ⟨w, hw⟩
    have hu := uavg_mono hpt
    rw [hgd, hhd, uavg_sum, uavg_add, uavg_add, uavg_sum, uavg_sum, uavg_append i 1,
      pow_one] at hu
    simpa only [← uniformStrings_real_uavg] using hu
  have hsum := Finset.sum_le_sum hi
  rw [Finset.sum_comm, Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum,
    Finset.sum_comm (s := Finset.Ico k J)] at hsum
  have h1 : ∑ i ∈ Finset.Ico k J, (uniformStrings (α := α) i).real Bad
      ≤ ∑ i ∈ Finset.Icc k L₁, (uniformStrings (α := α) i).real Bad :=
    Finset.sum_le_sum_of_subset_of_nonneg (fun i hi => by
      simp only [Finset.mem_Ico, Finset.mem_Icc] at hi ⊢; omega) fun i _ _ => hU0 i Bad
  have h2 : ∑ i ∈ Finset.Ico k J, (uniformStrings (α := α) (i + 1)).real Bad
      ≤ ∑ i ∈ Finset.Icc k L₁, (uniformStrings (α := α) i).real Bad := by
    have e := Finset.sum_image (f := fun i => (uniformStrings (α := α) i).real Bad)
      (s := Finset.Ico k J) (g := (· + 1)) (fun x _ y _ h => by simpa using h)
    rw [← e]
    exact Finset.sum_le_sum_of_subset_of_nonneg (fun i hi => by
      simp only [Finset.mem_image, Finset.mem_Ico, Finset.mem_Icc] at hi ⊢; omega)
      fun i _ _ => hU0 i Bad
  -- the window holds a state's strings shorter than `J` only at the lengths `k` to `J - 1`
  have hshort : ∀ q, ∀ S ⊆ shortAt M J q, ∑ i ∈ Finset.Icc k L₁, (uniformStrings (α := α) i).real S
      = ∑ i ∈ Finset.Ico k J, (uniformStrings (α := α) i).real S := by
    intro q S hS
    refine (Finset.sum_subset (fun i hi => by
      simp only [Finset.mem_Ico, Finset.mem_Icc] at hi ⊢; omega) fun i hi hi' => ?_).symm
    simp only [Finset.mem_Ico, Finset.mem_Icc, not_and, not_lt] at hi hi'
    refine le_antisymm ((measureReal_mono (fun y hy => ?_) (measure_ne_top _ _)).trans
      (hnull i).le) (hU0 i S)
    have := (hS hy).1
    show y.toList.length ≠ i
    omega
  have hN : 0 < (L₁ : ℝ) + 1 - k := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL
    linarith
  have hw : ∀ S, (windowStrings (α := α) k L₁).real S * ((L₁ : ℝ) + 1 - k)
      = ∑ i ∈ Finset.Icc k L₁, (uniformStrings i).real S := fun S => by
    rw [windowStrings_real hkL, div_mul_cancel₀ _ hN.ne']
  have hMq : ∀ q, ∑ i ∈ Finset.Ico k J, (uniformStrings (α := α) i).real (Mis q)
      ≤ (∑ i ∈ Finset.Ico k J, (uniformStrings (α := α) i).real {y | M.eval y.toList = q}) / 2
        + ((L₁ : ℝ) + 1 - k) * β' := by
    intro q
    have hwM := hw (Mis q)
    have hwE := hw (shortAt M J q)
    rw [hshort q _ Set.inter_subset_left] at hwM
    rw [hshort q _ subset_rfl] at hwE
    have hEx : ∑ i ∈ Finset.Ico k J, (uniformStrings (α := α) i).real (shortAt M J q)
        ≤ ∑ i ∈ Finset.Ico k J, (uniformStrings (α := α) i).real {y | M.eval y.toList = q} :=
      Finset.sum_le_sum fun i _ => measureReal_mono (fun y hy => hy.2) (measure_ne_top _ _)
    have h5 := mul_le_mul_of_nonneg_right (hmis q) hN.le
    nlinarith
  have hM := Finset.sum_le_sum fun q (_ : q ∈ W) => hMq q
  rw [Finset.sum_add_distrib, ← Finset.sum_div, Finset.sum_const, nsmul_eq_mul] at hM
  have hwB := hw Bad
  have hA0 : 0 ≤ A := by positivity
  have h3 := mul_le_mul_of_nonneg_left hbad hN.le
  have h4 := mul_le_mul_of_nonneg_left h2 hA0
  nlinarith

open scoped Classical in
/-- The states of `W`, each read on its wrong side, hold at most `2εr + εu` of `D`. -/
theorem wrongSide_mass_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (h0 : ∀ r v, 0 ≤ dist r v) (hle1 : ∀ r v, dist r v ≤ 1)
    (hsum : ∀ w : FreeMonoid α, dist (M.eval w.toList) .accept + dist (M.eval w.toList) .reject
      + dist (M.eval w.toList) .undecided = 1)
    (W : Finset σ) (hW : ∀ q ∈ W, readSide (dist q) ≠ decide (q ∈ M.accept)) {εr εu : ℝ}
    (hmis : ∫ p, (if M.eval p.toList ∈ M.accept then dist (M.eval p.toList) .reject
      else dist (M.eval p.toList) .accept) ∂D ≤ εr)
    (hund : ∫ p, dist (M.eval p.toList) .undecided ∂D ≤ εu) :
    ∑ q ∈ W, D.real {p | M.eval p.toList = q} ≤ 2 * εr + εu := by
  have hint : ∀ g : FreeMonoid α → ℝ, (∀ p, 0 ≤ g p) → (∀ p, g p ≤ 1) → Integrable g D :=
    fun g g0 g1 => Integrable.of_bound (measurable_of_countable g).aestronglyMeasurable 1
      (ae_of_all _ fun p => by rw [Real.norm_eq_abs, abs_of_nonneg (g0 p)]; exact g1 p)
  set mis : FreeMonoid α → ℝ := fun p => if M.eval p.toList ∈ M.accept
    then dist (M.eval p.toList) .reject else dist (M.eval p.toList) .accept
  have hmi : Integrable mis D :=
    hint mis (fun p => by simp only [mis]; split_ifs <;> exact h0 _ _)
      (fun p => by simp only [mis]; split_ifs <;> exact hle1 _ _)
  have hui : Integrable (fun p : FreeMonoid α => dist (M.eval p.toList) .undecided) D :=
    hint _ (fun p => h0 _ _) (fun p => hle1 _ _)
  have hind : ∀ q : σ, Integrable ({p : FreeMonoid α | M.eval p.toList = q}.indicator 1) D :=
    fun q => hint _ (fun p => Set.indicator_nonneg (fun _ _ => zero_le_one) p)
      (fun p => by simp only [Set.indicator_apply]; split_ifs <;> simp)
  have e : ∑ q ∈ W, D.real {p | M.eval p.toList = q}
      = ∫ p, ∑ q ∈ W, {p : FreeMonoid α | M.eval p.toList = q}.indicator 1 p ∂D := by
    rw [integral_finsetSum _ fun q _ => hind q]
    exact Finset.sum_congr rfl fun q _ =>
      (integral_indicator_one MeasurableSpace.measurableSet_top).symm
  rw [e]
  have hpt : ∀ p,
      ∑ q ∈ W, {p : FreeMonoid α | M.eval p.toList = q}.indicator (1 : FreeMonoid α → ℝ) p
      ≤ 2 * mis p + dist (M.eval p.toList) .undecided := by
    intro p
    have hlhs : ∑ q ∈ W, {p : FreeMonoid α | M.eval p.toList = q}.indicator (1 : FreeMonoid α → ℝ) p
        = if M.eval p.toList ∈ W then 1 else 0 := by
      simp only [Set.indicator_apply, Set.mem_setOf_eq, Pi.one_apply]
      exact Finset.sum_ite_eq W (M.eval p.toList) fun _ => (1 : ℝ)
    have hm0 : 0 ≤ mis p := by simp only [mis]; split_ifs <;> exact h0 _ _
    have hu0 := h0 (M.eval p.toList) .undecided
    rw [hlhs]
    split_ifs with hq
    swap
    · linarith
    have hs := hsum p
    have hw := hW _ hq
    simp only [mis]
    by_cases ha : M.eval p.toList ∈ M.accept
    · rw [if_pos ha]
      simp only [readSide, ha, decide_true, ne_eq, decide_eq_true_eq, not_lt] at hw
      linarith
    · rw [if_neg ha]
      simp only [readSide, ha, decide_false, ne_eq, decide_eq_false_iff_not, not_not] at hw
      linarith
  calc ∫ p, ∑ q ∈ W, {p : FreeMonoid α | M.eval p.toList = q}.indicator 1 p ∂D
      ≤ ∫ p, (2 * mis p + dist (M.eval p.toList) .undecided) ∂D :=
        integral_mono (integrable_finsetSum _ fun q _ => hind q)
          ((hmi.const_mul 2).add hui) hpt
    _ = 2 * ∫ p, mis p ∂D + ∫ p, dist (M.eval p.toList) .undecided ∂D := by
        rw [integral_add (hmi.const_mul 2) hui, integral_const_mul]
    _ ≤ 2 * εr + εu := by linarith

end Summed

section Replication

variable [Nonempty α]

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
    refine uavg_mono fun v => ?_
    generalize FreeMonoid.ofList v.toList = x
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
    _ ≤ uavg (2 * L) (E.indicator 1) := uavg_mono fun v => hpt _
    _ = (uniformStrings (2 * L)).real E := (uniformStrings_real_uavg _ _).symm

end Replication

section Bridge

variable [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ]

/-- A last round over the probes to `2L + |Q| + 3`, at most `εB` of them disagreeing and the
misled ones at most half of each state's strings shorter than `2L + 1` plus `εm`, its reads wrong
at most `εr` and undecided at most `εu`, has a leaf from which its hypothesis misjudges at most
`γ + N(2((|Σ|+1)εB + |Q|εm) + 2εr + εu)/m`, where a state visited `m` of the time re-roots the
target with error at most `γ`. -/
theorem dfa_of_checks_tsrb (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) {K : ℕ} (o : LoopRound Ω α K)
    (L : ℕ) {m εr εu εB εm γ : ℝ}
    (hch : o.Checks M O (windowStrings ((L + 1) / 2) (2 * L + Fintype.card σ + 3)) ((L + 1) / 2)
      εr εu εB (shortAt M (2 * L + 1)) εm) (hεm : 0 ≤ εm) (hm : 0 < m)
    (hstart : ∃ q, m ≤ windowVisitRate M L q ∧ rerootError M L q ≤ γ) :
    ∃ s ∈ o.result.hypothesis.tree.leaves, (uniformStrings L).real
      {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)}
        ≤ γ + ((2 * L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
          * (2 * (((Fintype.card α : ℝ) + 1) * εB + Fintype.card σ * εm) + (2 * εr + εu)) / m := by
  classical
  set k := (L + 1) / 2 with hkd
  set L₁ := 2 * L + Fintype.card σ + 3 with hL₁d
  have hkL₁ : k ≤ L₁ := by omega
  set D : Measure (FreeMonoid α) := windowStrings k L₁ with hDd
  haveI : IsProbabilityMeasure D := windowStrings_isProbability hkL₁
  set H := o.result.hypothesis with hHd
  set dist := familyReadDist O M o.family o.state.lo (o.state.hi + 1) with hdistd
  set side : σ → Bool := fun q => readSide (dist q)
  have hd0 := familyReadDist_nonneg O M o.family o.state.lo (o.state.hi + 1)
  have hd1 := familyReadDist_le_one O M o.family o.state.lo (o.state.hi + 1)
  obtain ⟨⟨hcons, hmis, hund⟩, hchk, hcls, hfew⟩ := hch
  have hroot : ∃ r a, H.tree = .node 1 r a := rootOne_of_classSet _ _ hcls
  -- the family's reads over the window
  have hund' : ∫ p, dist (M.eval p.toList) .undecided ∂D ≤ εu := by
    refine le_trans (le_of_eq ?_) hund
    congr 1
    funext p
    rw [readProb_eq_familyReadDist O M hL]
  have hmis' : ∫ p, (if M.eval p.toList ∈ M.accept then dist (M.eval p.toList) .reject
      else dist (M.eval p.toList) .accept) ∂D ≤ εr := by
    refine le_trans (le_of_eq ?_) hmis
    congr 1
    funext p
    have e := congrFun (readProb_eq_familyReadDist O M hL o.family o.state.lo (o.state.hi + 1) p)
    unfold misreadProb
    by_cases hp : M.eval p.toList ∈ M.accept
    · simp only [if_pos hp, if_pos ((hL p).2 hp)]
      exact (e .reject).symm
    · simp only [if_neg hp, if_neg (fun h => hp ((hL p).1 h))]
      exact (e .accept).symm
  set N : ℝ := (L₁ : ℝ) + 1 - k with hNd
  have hN0 : 0 < N := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL₁
    linarith
  have hεB0 : 0 ≤ εB := measureReal_nonneg.trans hchk
  -- a state's occupancy of the lengths `k` to `2L`
  set occ : σ → ℝ := fun q => ∑ i ∈ Finset.Ico k (2 * L + 1),
    (uniformStrings i).real {y : FreeMonoid α | M.eval y.toList = q} with hoccd
  have hocc0 : ∀ q, 0 ≤ occ q := fun q => Finset.sum_nonneg fun _ _ => measureReal_nonneg
  -- the wrong states: an edge out of the true leaf wrong or unlearned, or the wrong side
  set WE : Finset σ := Finset.univ.filter fun q => ∃ a, ¬ ∃ w,
    H.edges (Random.leafOf M side H.tree q, a) = some (Random.leafOf M side H.tree (M.step q a), w)
  set WS : Finset σ := Finset.univ.filter fun q => side q ≠ decide (q ∈ M.accept)
  let a : σ → α := fun q => if h : ∃ a, ¬ ∃ w, H.edges (Random.leafOf M side H.tree q, a)
      = some (Random.leafOf M side H.tree (M.step q a), w) then h.choose
    else Classical.arbitrary α
  have haW : ∀ q ∈ WE, ¬ ∃ w, H.edges (Random.leafOf M side H.tree q, a q)
      = some (Random.leafOf M side H.tree (M.step q (a q)), w) := by
    intro q hq
    have h := (Finset.mem_filter.1 hq).2
    simp only [a, dif_pos h]
    exact h.choose_spec
  have hE : ∑ q ∈ WE, occ q ≤ 2 * (((Fintype.card α : ℝ) + 1) * N * εB + WE.card * (N * εm)) :=
    wrongEdge_occupancy_le M dist o.read H (by omega : k < 2 * L + 1)
      (by omega : 2 * L + 1 ≤ L₁) WE a haW hchk (fun q => hfew q (a q))
  have hEc : (WE.card : ℝ) ≤ Fintype.card σ := by exact_mod_cast Finset.card_le_univ WE
  have hSm := wrongSide_mass_le M dist D hd0 hd1
    (fun w => familyReadDist_sum O M _ _ _ _ ⟨w, rfl⟩) WS (fun q hq => (Finset.mem_filter.1 hq).2)
    hmis' hund'
  have hoccD : ∀ q, occ q ≤ N * D.real {p | M.eval p.toList = q} := by
    intro q
    rw [hDd, windowStrings_real hkL₁, mul_div_cancel₀ _ hN0.ne']
    exact Finset.sum_le_sum_of_subset_of_nonneg (fun i hi => by
      simp only [Finset.mem_Ico, Finset.mem_Icc] at hi ⊢; omega) fun _ _ _ => measureReal_nonneg
  have hWS : ∑ q ∈ WS, occ q ≤ N * (2 * εr + εu) := by
    calc ∑ q ∈ WS, occ q ≤ ∑ q ∈ WS, N * D.real {p | M.eval p.toList = q} :=
          Finset.sum_le_sum fun q _ => hoccD q
      _ = N * ∑ q ∈ WS, D.real {p | M.eval p.toList = q} := (Finset.mul_sum _ _ _).symm
      _ ≤ N * (2 * εr + εu) := mul_le_mul_of_nonneg_left hSm hN0.le
  set X : ℝ := N * (2 * (((Fintype.card α : ℝ) + 1) * εB + Fintype.card σ * εm) + (2 * εr + εu))
    with hXd
  have hWrong : ∑ q ∈ WE ∪ WS, occ q ≤ X := by
    have hu := Finset.sum_union_inter (s₁ := WE) (s₂ := WS) (f := occ)
    have hi0 : 0 ≤ ∑ q ∈ WE ∩ WS, occ q := Finset.sum_nonneg fun q _ => hocc0 q
    have h1 : (WE.card : ℝ) * (N * εm) ≤ Fintype.card σ * (N * εm) :=
      mul_le_mul_of_nonneg_right hEc (by positivity)
    rw [hXd]
    nlinarith
  -- the run from `M`'s start meets a wrong state from `k` to `2L` at most their occupancy
  set Bad : Set σ := ↑(WE ∪ WS) with hBadd
  have hΦ : (uniformStrings (2 * L)).real
      {x | ∃ j, k ≤ j ∧ j ≤ 2 * L ∧ M.eval (prefixOf x j).toList ∈ Bad}
      ≤ ∑ q ∈ WE ∪ WS, occ q := by
    have hsub : {x : FreeMonoid α | ∃ j, k ≤ j ∧ j ≤ 2 * L ∧ M.eval (prefixOf x j).toList ∈ Bad}
        ⊆ ⋃ q ∈ WE ∪ WS, ⋃ j ∈ Finset.Ico k (2 * L + 1),
          {x | prefixOf x j ∈ {y : FreeMonoid α | M.eval y.toList = q}} := by
      rintro x ⟨j, h1, h2, h3⟩
      exact Set.mem_biUnion (Finset.mem_coe.1 h3)
        (Set.mem_biUnion (Finset.mem_Ico.2 ⟨h1, by omega⟩) rfl)
    refine (measureReal_mono hsub (measure_ne_top _ _)).trans
      ((measureReal_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun q _ => ?_))
    refine (measureReal_biUnion_finset_le _ _).trans (le_of_eq (Finset.sum_congr rfl fun j hj => ?_))
    exact uniformStrings_prefix (by have := (Finset.mem_Ico.1 hj).2; omega) _
  obtain ⟨q₀, hw, he⟩ := hstart
  set Hit : Set (FreeMonoid α) :=
    {x | ∃ i ≤ L, M.evalFrom q₀ (prefixOf x i).toList ∈ Bad} with hHitd
  have hhit : (uniformStrings L).real Hit ≤ X / m := by
    rw [le_div_iff₀ hm]
    have h0 : 0 ≤ (uniformStrings L).real Hit := measureReal_nonneg
    calc (uniformStrings L).real Hit * m ≤ (uniformStrings L).real Hit * windowVisitRate M L q₀ :=
          mul_le_mul_of_nonneg_left hw h0
      _ = windowVisitRate M L q₀ * (uniformStrings L).real Hit := mul_comm _ _
      _ ≤ _ := window_mul_hit_le M L q₀ Bad
      _ ≤ _ := hΦ
      _ ≤ X := hWrong
  -- off the runs through a wrong state, the start's true leaf classifies as re-rooting does
  have hsub : {x | ¬ H.Classifies (Random.leafOf M side H.tree q₀) x (M.eval x.toList ∈ M.accept)}
      ⊆ ({x | ¬ (M.evalFrom q₀ x.toList ∈ M.accept ↔ M.eval x.toList ∈ M.accept)} ∪ Hit)
        ∪ {x | x.toList.length ≠ L} := by
    intro x hx
    by_contra hcon
    simp only [Set.mem_union, Set.mem_setOf_eq, not_or, not_not] at hcon
    obtain ⟨⟨hag, hH⟩, hl⟩ := hcon
    apply hx
    refine classifies_of_matches M side H hroot {q | q ∉ WE ∪ WS} (fun q hq b => ?_)
      (fun q hq => ?_) q₀ x (fun i hi hb => hH ⟨i, hl ▸ hi, Finset.mem_coe.2 hb⟩) hag
    · by_contra hb
      exact hq (Finset.mem_union_left _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, b, hb⟩))
    · by_contra hb
      exact hq (Finset.mem_union_right _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, hb⟩))
  have hnull : (uniformStrings (α := α) L).real {x | x.toList.length ≠ L} = 0 := by
    rw [measureReal_def, uniformStrings_length, ENNReal.toReal_zero]
  refine ⟨Random.leafOf M side H.tree q₀, Random.leafOf_mem M side _ _, ?_⟩
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans ((measureReal_union_le _ _).trans ?_)
  rw [hnull, add_zero]
  refine (measureReal_union_le _ _).trans ?_
  have : (uniformStrings L).real
      {x | ¬ (M.evalFrom q₀ x.toList ∈ M.accept ↔ M.eval x.toList ∈ M.accept)} ≤ γ := he
  linarith

end Bridge

end OrthoDFA

end
