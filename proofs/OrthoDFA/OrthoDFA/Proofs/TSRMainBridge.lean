import OrthoDFA.LoopDFATSRMain
import OrthoDFA.Proofs.LoopDFA

/-!
# Typical-start replication with `LearnerLoopDFA`'s probes: the bridge

The wrong states' occupancy of the lengths `⌈L/2⌉` to `L₁ - 1` is bounded as in #530
(`wrongEdge_occupancy_le`, `wrongSide_mass_le`). The run from the start `q` meets them at a
position `i ≤ L`, either at `i ≥ ⌈L/2⌉` or before (`hit_le_occupancy`). A shortest word `v` from a
state to `q` has `|v| < |Q|` (`short_word`), and the strings beginning with it are `|Σ|^-|v|` of
all (`hitSet_shift`). For `i ≥ ⌈L/2⌉`, the word from `M`'s start puts the meeting at
`|v| + i ≤ L + |Q| - 1`. For `i < ⌈L/2⌉`, a string visiting `q` from `⌈L/2⌉` to `L` is at a state
reaching `q` at `⌈L/2⌉` (`hitSet_split`), and that state's word puts the meeting before
`2⌈L/2⌉ + |Q| - 1 ≤ L + |Q|`.
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

section Paths

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

variable {σ : Type*} (M : _root_.DFA α σ) (B : Set σ)

/-- The strings whose run from `s` meets `B` at some position from `a` to `b`. -/
def hitSet (s : σ) (a b : ℕ) : Set (FreeMonoid α) :=
  {y | ∃ i, a ≤ i ∧ i ≤ b ∧ M.evalFrom s (prefixOf y i).toList ∈ B}

theorem hitSet_prefix (s : σ) (a : ℕ) {b n : ℕ} (h : b ≤ n) :
    (uniformStrings n).real (hitSet M B s a b) = (uniformStrings b).real (hitSet M B s a b) := by
  rw [← uniformStrings_prefix h]
  congr 1
  ext y
  simp only [hitSet, Set.mem_setOf_eq]
  constructor
  · rintro ⟨i, h1, h2, h3⟩
    exact ⟨i, h1, h2, by rwa [prefixOf_prefixOf y h2]⟩
  · rintro ⟨i, h1, h2, h3⟩
    exact ⟨i, h1, h2, by rwa [prefixOf_prefixOf y h2] at h3⟩

theorem hitSet_shift (s : σ) (u : List α) (a b t : ℕ) :
    ((Fintype.card α : ℝ) ^ u.length)⁻¹
        * (uniformStrings t).real (hitSet M B (M.evalFrom s u) a b)
      ≤ (uniformStrings (u.length + t)).real (hitSet M B s (u.length + a) (u.length + b)) := by
  classical
  have hf : uavg u.length ({w : FreeMonoid α | w = FreeMonoid.ofList u}.indicator 1)
      = ((Fintype.card α : ℝ) ^ u.length)⁻¹ := by
    unfold uavg
    obtain ⟨v₀, hv₀⟩ : ∃ v₀ : List.Vector α u.length, v₀.toList = u := ⟨⟨u, rfl⟩, rfl⟩
    rw [Finset.sum_eq_single_of_mem v₀ (Finset.mem_univ v₀)]
    · rw [Set.indicator_of_mem (show FreeMonoid.ofList v₀.toList ∈ {w | w = FreeMonoid.ofList u}
        by rw [hv₀]; rfl), Pi.one_apply, one_div]
    · intro v _ hv
      refine Set.indicator_of_notMem (fun h => hv ?_) _
      apply List.Vector.toList_injective
      rw [hv₀]
      exact FreeMonoid.ofList.injective h
  rw [uniformStrings_real_uavg, uniformStrings_real_uavg, ← hf, ← uavg_prefix_mul_drop]
  refine uavg_mono fun v => ?_
  generalize FreeMonoid.ofList v.toList = x
  have hnn : ∀ (S : Set (FreeMonoid α)) (z : FreeMonoid α),
      0 ≤ S.indicator (1 : FreeMonoid α → ℝ) z := fun S z =>
    Set.indicator_nonneg (fun _ _ => zero_le_one) z
  by_cases hp : prefixOf x u.length = FreeMonoid.ofList u
  · by_cases hd : FreeMonoid.ofList (x.toList.drop u.length) ∈ hitSet M B (M.evalFrom s u) a b
    · obtain ⟨i, h1, h2, h3⟩ := hd
      have hmem : x ∈ hitSet M B s (u.length + a) (u.length + b) := by
        refine ⟨u.length + i, by omega, by omega, ?_⟩
        have hp' : x.toList.take u.length = u := by
          simpa [prefixOf] using congrArg FreeMonoid.toList hp
        have e : (prefixOf x (u.length + i)).toList
            = u ++ (prefixOf (FreeMonoid.ofList (x.toList.drop u.length)) i).toList := by
          simp [prefixOf, List.take_add, hp']
        rw [e, DFA.evalFrom_of_append]
        exact h3
      rw [Set.indicator_of_mem (show prefixOf x u.length ∈ {w | w = FreeMonoid.ofList u} from hp),
        Set.indicator_of_mem (show FreeMonoid.ofList (x.toList.drop u.length)
          ∈ hitSet M B (M.evalFrom s u) a b from ⟨i, h1, h2, h3⟩),
        Set.indicator_of_mem hmem]
      simp
    · rw [Set.indicator_of_notMem hd, mul_zero]
      exact hnn _ _
  · rw [Set.indicator_of_notMem (show prefixOf x u.length ∉ {w | w = FreeMonoid.ofList u} from hp),
      zero_mul]
    exact hnn _ _

theorem hitSet_start_le (W : Finset σ) {a b n : ℕ} (h : b ≤ n) :
    (uniformStrings n).real (hitSet M ↑W M.start a b)
      ≤ ∑ q ∈ W, ∑ j ∈ Finset.Icc a b,
        (uniformStrings j).real {y : FreeMonoid α | M.eval y.toList = q} := by
  have hsub : hitSet M ↑W M.start a b ⊆ ⋃ q ∈ W, ⋃ j ∈ Finset.Icc a b,
      {x | prefixOf x j ∈ {y : FreeMonoid α | M.eval y.toList = q}} := by
    rintro x ⟨j, h1, h2, h3⟩
    exact Set.mem_biUnion (Finset.mem_coe.1 h3)
      (Set.mem_biUnion (Finset.mem_Icc.2 ⟨h1, h2⟩) rfl)
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans
    ((measureReal_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun q _ => ?_))
  refine (measureReal_biUnion_finset_le _ _).trans (le_of_eq (Finset.sum_congr rfl fun j hj => ?_))
  exact uniformStrings_prefix (by have := (Finset.mem_Icc.1 hj).2; omega) _

variable [Fintype σ]

theorem hitSet_split (k t : ℕ) :
    ∑ r : σ, (uniformStrings k).real {y : FreeMonoid α | M.eval y.toList = r}
        * (uniformStrings t).real (hitSet M B r 0 t)
      ≤ (uniformStrings (k + t)).real (hitSet M B M.start k (k + t)) := by
  classical
  have hnn : ∀ (S : Set (FreeMonoid α)) (z : FreeMonoid α),
      0 ≤ S.indicator (1 : FreeMonoid α → ℝ) z := fun S z =>
    Set.indicator_nonneg (fun _ _ => zero_le_one) z
  have hle1 : ∀ (S : Set (FreeMonoid α)) (z : FreeMonoid α),
      S.indicator (1 : FreeMonoid α → ℝ) z ≤ 1 := by
    intro S z
    simp only [Set.indicator_apply]
    split_ifs <;> norm_num
  simp only [uniformStrings_real_uavg]
  rw [Finset.sum_congr rfl fun r _ => (uavg_prefix_mul_drop k t _ _).symm, ← uavg_sum]
  refine uavg_mono fun v => ?_
  generalize FreeMonoid.ofList v.toList = x
  set r₀ := M.eval (prefixOf x k).toList
  rw [Finset.sum_eq_single_of_mem r₀ (Finset.mem_univ _) fun r _ hr => by
    rw [Set.indicator_of_notMem (show prefixOf x k ∉ {y : FreeMonoid α | M.eval y.toList = r}
      from fun h => hr h.symm), zero_mul]]
  rw [Set.indicator_of_mem (show prefixOf x k ∈ {y : FreeMonoid α | M.eval y.toList = r₀}
    from rfl), Pi.one_apply, one_mul]
  by_cases hd : FreeMonoid.ofList (x.toList.drop k) ∈ hitSet M B r₀ 0 t
  · obtain ⟨i, -, h2, h3⟩ := hd
    have hmem : x ∈ hitSet M B M.start k (k + t) := by
      refine ⟨k + i, by omega, by omega, ?_⟩
      have e : (prefixOf x (k + i)).toList
          = (prefixOf x k).toList ++ (prefixOf (FreeMonoid.ofList (x.toList.drop k)) i).toList := by
        simp [prefixOf, List.take_add]
      rw [e, DFA.evalFrom_of_append]
      exact h3
    rw [Set.indicator_of_mem hmem]
    exact hle1 _ _
  · rw [Set.indicator_of_notMem hd]
    exact hnn _ _

omit [Fintype α] [DecidableEq α] [Nonempty α] in
theorem short_word (s : σ) : ∀ (n : ℕ) (u : List α), u.length = n →
    ∃ v : List α, v.length + 1 ≤ Fintype.card σ ∧ M.evalFrom s v = M.evalFrom s u := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro u hn
    by_cases hlen : Fintype.card σ ≤ u.length
    · obtain ⟨q, a, b, c, hx, -, hb, hqa, -, hqc⟩ := M.evalFrom_split hlen rfl
      have hlt : (a ++ c).length < n := by
        have := List.length_pos_of_ne_nil hb
        rw [← hn, hx]
        simp only [List.length_append]
        omega
      obtain ⟨v, hv, hve⟩ := ih _ hlt (a ++ c) rfl
      refine ⟨v, hv, ?_⟩
      rw [hve, DFA.evalFrom_of_append, hqa, hqc]
    · exact ⟨u, by omega, rfl⟩

/-- The run from `q`, the window visiting `q` on `w > 0` of the strings, meets `W` within `L`
letters on at most `|Σ|^(|Q|-1)(1 + 1/w)` times `W`'s occupancy of the lengths `⌈L/2⌉` to
`L + |Q| + 2`. -/
theorem hit_le_occupancy (L : ℕ) (hL : 1 ≤ L) (q : σ) (W : Finset σ)
    (hw : 0 < windowVisitRate M L q) :
    (uniformStrings L).real {x | ∃ i ≤ L, M.evalFrom q (prefixOf x i).toList ∈ (W : Set σ)}
      ≤ (Fintype.card α : ℝ) ^ (Fintype.card σ - 1) * (1 + 1 / windowVisitRate M L q)
        * ∑ r ∈ W, ∑ j ∈ Finset.Ico ((L + 1) / 2) (L + Fintype.card σ + 3),
          (uniformStrings j).real {y : FreeMonoid α | M.eval y.toList = r} := by
  classical
  set k := (L + 1) / 2 with hkd
  set w := windowVisitRate M L q with hwd
  set A : ℝ := (Fintype.card α : ℝ) with hAd
  set G : ℝ := A ^ (Fintype.card σ - 1) with hGd
  set X : ℝ := ∑ r ∈ W, ∑ j ∈ Finset.Ico k (L + Fintype.card σ + 3),
    (uniformStrings j).real {y : FreeMonoid α | M.eval y.toList = r} with hXd
  set Bad : Set σ := ↑W
  have hk1 : 1 ≤ k := by omega
  have hA1 : (1 : ℝ) ≤ A := by rw [hAd]; exact_mod_cast Fintype.card_pos
  have hA0 : 0 < A := by linarith
  have hG1 : 1 ≤ G := one_le_pow₀ hA1
  have hG0 : 0 < G := by linarith
  have : Nonempty σ := ⟨M.start⟩
  have hσ : 1 ≤ Fintype.card σ := Fintype.card_pos
  have hX : ∀ a b n, b ≤ n → k ≤ a → b < L + Fintype.card σ + 3 →
      (uniformStrings n).real (hitSet M Bad M.start a b) ≤ X := by
    intro a b n hbn hka hb
    refine (hitSet_start_le M W hbn).trans (Finset.sum_le_sum fun r _ => ?_)
    exact Finset.sum_le_sum_of_subset_of_nonneg (fun j hj => by
      simp only [Finset.mem_Icc, Finset.mem_Ico] at hj ⊢; omega) fun _ _ _ => measureReal_nonneg
  have hpowG : ∀ v : List α, v.length + 1 ≤ Fintype.card σ → A ^ v.length ≤ G := fun v hv =>
    pow_le_pow_right₀ hA1 (by omega)
  -- the hits are at or after `k`, or before
  have hsplit : {x | ∃ i ≤ L, M.evalFrom q (prefixOf x i).toList ∈ Bad}
      ⊆ hitSet M Bad q 0 (k - 1) ∪ hitSet M Bad q k L := by
    rintro x ⟨i, hi, h⟩
    by_cases hik : i < k
    · exact Or.inl ⟨i, Nat.zero_le _, by omega, h⟩
    · exact Or.inr ⟨i, by omega, hi, h⟩
  -- `q` is reached from `M`'s start
  have hreach : ∃ u : List α, M.evalFrom M.start u = q := by
    by_contra hno
    push Not at hno
    have hempty : {x : FreeMonoid α | ∃ j, (L + 1) / 2 ≤ j ∧ j ≤ L
        ∧ M.eval (prefixOf x j).toList = q} = ∅ := by
      ext x
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
      exact fun j _ _ => hno _
    have h := hw
    simp only [hwd, windowVisitRate, hempty, measureReal_empty] at h
    exact lt_irrefl _ h
  -- the late hits, along a shortest word from `M`'s start
  have hlate : (uniformStrings L).real (hitSet M Bad q k L) ≤ G * X := by
    obtain ⟨u, hu⟩ := hreach
    obtain ⟨v, hv, hve⟩ := short_word M M.start _ u rfl
    rw [hu] at hve
    have h1 := hitSet_shift M Bad M.start v k L L
    rw [hve] at h1
    have h2 := hX (v.length + k) (v.length + L) (v.length + L) le_rfl (by omega) (by omega)
    have hpv : 0 < A ^ v.length := pow_pos hA0 _
    have h3 : (uniformStrings L).real (hitSet M Bad q k L) ≤ A ^ v.length * X := by
      have := h1.trans h2
      rw [inv_mul_le_iff₀ hpv] at this
      exact this
    have hX0 : 0 ≤ X := by
      refine Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => measureReal_nonneg
    exact h3.trans (mul_le_mul_of_nonneg_right (hpowG v hv) hX0)
  -- the early hits, after a visit to `q` within `|Q| - 1` letters of `k`
  set t := (Fintype.card σ - 1) + (k - 1) with htd
  set h₁ := (uniformStrings (k - 1)).real (hitSet M Bad q 0 (k - 1)) with hh₁d
  set R : Finset σ := Finset.univ.filter fun r => ∃ u : List α, M.evalFrom r u = q with hRd
  have hR : ∀ r ∈ R, G⁻¹ * h₁ ≤ (uniformStrings t).real (hitSet M Bad r 0 t) := by
    intro r hr
    obtain ⟨u, hu⟩ := (Finset.mem_filter.1 hr).2
    obtain ⟨v, hv, hve⟩ := short_word M r _ u rfl
    rw [hu] at hve
    have h1 := hitSet_shift M Bad r v 0 (k - 1) (k - 1)
    rw [hve, add_zero] at h1
    have h2 : (uniformStrings (v.length + (k - 1))).real
        (hitSet M Bad r v.length (v.length + (k - 1)))
        = (uniformStrings t).real (hitSet M Bad r v.length (v.length + (k - 1))) :=
      (hitSet_prefix M Bad r _ (by omega)).symm
    have h3 : (uniformStrings t).real (hitSet M Bad r v.length (v.length + (k - 1)))
        ≤ (uniformStrings t).real (hitSet M Bad r 0 t) :=
      measureReal_mono (fun y ⟨i, hi1, hi2, hi3⟩ => ⟨i, Nat.zero_le _, by omega, hi3⟩)
        (measure_ne_top _ _)
    have h4 : G⁻¹ ≤ (A ^ v.length)⁻¹ := inv_anti₀ (pow_pos hA0 _) (hpowG v hv)
    have h0 : 0 ≤ h₁ := measureReal_nonneg
    calc G⁻¹ * h₁ ≤ (A ^ v.length)⁻¹ * h₁ := mul_le_mul_of_nonneg_right h4 h0
      _ ≤ (uniformStrings (v.length + (k - 1))).real
          (hitSet M Bad r v.length (v.length + (k - 1))) := h1
      _ = _ := h2
      _ ≤ _ := h3
  have hwR : w ≤ ∑ r ∈ R, (uniformStrings k).real {y : FreeMonoid α | M.eval y.toList = r} := by
    show (uniformStrings L).real {x : FreeMonoid α | ∃ j, k ≤ j ∧ j ≤ L
      ∧ M.eval (prefixOf x j).toList = q} ≤ _
    have hsub : {x : FreeMonoid α | ∃ j, k ≤ j ∧ j ≤ L
        ∧ M.eval (prefixOf x j).toList = q}
        ⊆ ⋃ r ∈ R, {x | prefixOf x k ∈ {y : FreeMonoid α | M.eval y.toList = r}} := by
      rintro x ⟨j, h1, h2, h3⟩
      refine Set.mem_biUnion (x := M.eval (prefixOf x k).toList)
        (Finset.mem_filter.2 ⟨Finset.mem_univ _, ⟨(x.toList.take j).drop k, ?_⟩⟩)
        (show M.eval (prefixOf x k).toList = _ from rfl)
      have e : (prefixOf x j).toList = (prefixOf x k).toList ++ (x.toList.take j).drop k := by
        have := List.take_append_drop k (x.toList.take j)
        rw [List.take_take, min_eq_left h1] at this
        simpa [prefixOf] using this.symm
      rw [DFA.eval, ← DFA.evalFrom_of_append, ← e]
      exact h3
    refine (measureReal_mono hsub (measure_ne_top _ _)).trans
      ((measureReal_biUnion_finset_le _ _).trans (le_of_eq (Finset.sum_congr rfl fun r _ => ?_)))
    exact uniformStrings_prefix (by omega) _
  have hearly : G⁻¹ * h₁ * w ≤ X := by
    have hnn : ∀ r, 0 ≤ (uniformStrings k).real {y : FreeMonoid α | M.eval y.toList = r}
        * (uniformStrings t).real (hitSet M Bad r 0 t) := fun r =>
      mul_nonneg measureReal_nonneg measureReal_nonneg
    have hc : 0 ≤ G⁻¹ * h₁ := mul_nonneg (inv_nonneg.2 hG0.le) measureReal_nonneg
    calc G⁻¹ * h₁ * w
        ≤ G⁻¹ * h₁ * ∑ r ∈ R, (uniformStrings k).real {y : FreeMonoid α | M.eval y.toList = r} :=
          mul_le_mul_of_nonneg_left hwR hc
      _ = ∑ r ∈ R, (uniformStrings k).real {y : FreeMonoid α | M.eval y.toList = r}
            * (G⁻¹ * h₁) := by
          rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun _ _ => mul_comm _ _
      _ ≤ ∑ r ∈ R, (uniformStrings k).real {y : FreeMonoid α | M.eval y.toList = r}
            * (uniformStrings t).real (hitSet M Bad r 0 t) :=
          Finset.sum_le_sum fun r hr => mul_le_mul_of_nonneg_left (hR r hr) measureReal_nonneg
      _ ≤ ∑ r : σ, (uniformStrings k).real {y : FreeMonoid α | M.eval y.toList = r}
            * (uniformStrings t).real (hitSet M Bad r 0 t) :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) fun r _ _ => hnn r
      _ ≤ (uniformStrings (k + t)).real (hitSet M Bad M.start k (k + t)) := hitSet_split M Bad k t
      _ ≤ X := hX k (k + t) (k + t) le_rfl le_rfl (by omega)
  have hearly' : (uniformStrings L).real (hitSet M Bad q 0 (k - 1)) ≤ G * X / w := by
    rw [hitSet_prefix M Bad q 0 (by omega : k - 1 ≤ L), ← hh₁d, le_div_iff₀ hw]
    have : h₁ * w = G * (G⁻¹ * h₁ * w) := by
      rw [← mul_assoc, ← mul_assoc, mul_inv_cancel₀ hG0.ne', one_mul]
    rw [this]
    exact mul_le_mul_of_nonneg_left hearly hG0.le
  calc (uniformStrings L).real {x | ∃ i ≤ L, M.evalFrom q (prefixOf x i).toList ∈ Bad}
      ≤ (uniformStrings L).real (hitSet M Bad q 0 (k - 1) ∪ hitSet M Bad q k L) :=
        measureReal_mono hsplit (measure_ne_top _ _)
    _ ≤ G * X / w + G * X := (measureReal_union_le _ _).trans (add_le_add hearly' hlate)
    _ = G * (1 + 1 / w) * X := by
        have := hw.ne'
        field_simp
        ring

end Paths

section Bridge

variable [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ]

/-- A last round over `LearnerLoopDFA`'s probes, at most `εB` of them disagreeing and the misled
ones at most half of each state's strings shorter than `L + |Q| + 3` plus `εm`, its reads wrong at
most `εr` and undecided at most `εu`, has a leaf from which its hypothesis misjudges at most
`γ + 2|Σ|^(|Q|-1) N(2((|Σ|+1)εB + |Q|εm) + 2εr + εu)/m`, where a state visited `m` of the time
re-roots the target with error at most `γ`. -/
theorem dfa_of_checks_tsrm (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) {K : ℕ} (o : LoopRound Ω α K)
    (L : ℕ) (hL1 : 1 ≤ L) {m εr εu εB εm γ : ℝ}
    (hch : o.Checks M O (windowStrings ((L + 1) / 2) (L + Fintype.card σ + 3)) ((L + 1) / 2)
      εr εu εB (shortAt M (L + Fintype.card σ + 3)) εm) (hεm : 0 ≤ εm) (hm : 0 < m)
    (hstart : ∃ q, m ≤ windowVisitRate M L q ∧ rerootError M L q ≤ γ) :
    ∃ s ∈ o.result.hypothesis.tree.leaves, (uniformStrings L).real
      {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)}
        ≤ γ + 2 * (Fintype.card α : ℝ) ^ (Fintype.card σ - 1)
          * (((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
            * (2 * (((Fintype.card α : ℝ) + 1) * εB + Fintype.card σ * εm)
              + (2 * εr + εu))) / m := by
  classical
  set k := (L + 1) / 2 with hkd
  set L₁ := L + Fintype.card σ + 3 with hL₁d
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
  -- a state's occupancy of the lengths `k` to `L₁ - 1`
  set occ : σ → ℝ := fun q => ∑ i ∈ Finset.Ico k L₁,
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
    wrongEdge_occupancy_le M dist o.read H (by omega : k < L₁) le_rfl WE a haW hchk
      (fun q => hfew q (a q))
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
  -- the run from the start meets a wrong state on at most `2|Σ|^(|Q|-1) X / m` of the strings
  set Bad : Set σ := ↑(WE ∪ WS) with hBadd
  obtain ⟨q₀, hw, he⟩ := hstart
  set Hit : Set (FreeMonoid α) :=
    {x | ∃ i ≤ L, M.evalFrom q₀ (prefixOf x i).toList ∈ Bad} with hHitd
  set G : ℝ := (Fintype.card α : ℝ) ^ (Fintype.card σ - 1) with hGd
  have hG0 : 0 < G := by
    have : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
    positivity
  have hX0 : 0 ≤ X := (Finset.sum_nonneg fun q _ => hocc0 q).trans hWrong
  have hhit : (uniformStrings L).real Hit ≤ 2 * G * X / m := by
    have hw0 : 0 < windowVisitRate M L q₀ := hm.trans_le hw
    have hw1 : windowVisitRate M L q₀ ≤ 1 := measureReal_le_one
    have h1 := hit_le_occupancy M L hL1 q₀ (WE ∪ WS) hw0
    have h2 : 1 + 1 / windowVisitRate M L q₀ ≤ 2 / m := by
      have : 1 ≤ 1 / windowVisitRate M L q₀ := by rw [le_div_iff₀ hw0]; linarith
      have : 1 / windowVisitRate M L q₀ ≤ 1 / m := one_div_le_one_div_of_le hm hw
      have e : 2 / m = 1 / m + 1 / m := by ring
      linarith
    calc (uniformStrings L).real Hit
        ≤ G * (1 + 1 / windowVisitRate M L q₀) * ∑ q ∈ WE ∪ WS, occ q := h1
      _ ≤ G * (2 / m) * X :=
          mul_le_mul (mul_le_mul_of_nonneg_left h2 hG0.le) hWrong
            (Finset.sum_nonneg fun q _ => hocc0 q) (by positivity)
      _ = 2 * G * X / m := by ring
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
  have e : 2 * G * X / m = 2 * (Fintype.card α : ℝ) ^ (Fintype.card σ - 1)
      * (((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
        * (2 * (((Fintype.card α : ℝ) + 1) * εB + Fintype.card σ * εm) + (2 * εr + εu))) / m := rfl
  linarith

end Bridge
end OrthoDFA

end
