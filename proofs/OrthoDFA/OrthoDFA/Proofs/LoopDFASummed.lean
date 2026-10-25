import OrthoDFA.LoopDFASummed
import OrthoDFA.Proofs.LoopDFA

/-!
# The learner's DFA under the relative rule, by summed occupancy: the proof

The bridge (`dfa_of_checks_summed`) replaces `dfa_of_checks`'s per-state steps by sums over the
wrong states. A string `y` of length `i` at a state `q` with a wrong edge at `a q` is disagreeing,
has `y·a q` disagreeing, or is misled (`edge_of_pair`). Each `y` has one state, so summed over the
wrong-edge states this is at most `U_i(Bad) + |Σ| U_{i+1}(Bad)` plus the misled ones, at most half
of each state's plus `β'` each (`FewMisled`); the occupancy is at most `2N((|Σ|+1)β + |Q|β')`
(`wrongEdge_occupancy_le`). A string at a state read on its wrong side has
`2·misread + undecided ≥ 1`, so those states hold at most `2εr + εu` of the window
(`wrongSide_mass_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

open Random (prefixMax le_prefixMax prefixMax_pos)

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

section Bridge

variable [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ]

/-- A last round as in `dfa_of_checks`, whose wrong states together occupy the window's lengths
`⌈L/2⌉` to `L` less than `c`, has a leaf from which its hypothesis misjudges at most `γ + Cc`,
`C = R(L+1)/(L+1−⌈L/2⌉)`, where a start covered at `c` re-roots with accuracy `1 − γ` through
itself and the states it dominates. -/
theorem dfa_of_checks_summed (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) {K : ℕ} (o : LoopRound Ω α K)
    (L : ℕ) {c R εr εu εB εm γ : ℝ}
    (hch : o.Checks M O (windowStrings ((L + 1) / 2) (L + Fintype.card σ + 3)) ((L + 1) / 2)
      εr εu εB (shortAt M (L + 1)) εm) (hεm : 0 ≤ εm) (hR : 0 < R)
    (hbnd : ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
      * (2 * (((Fintype.card α : ℝ) + 1) * εB + Fintype.card σ * εm) + (2 * εr + εu)) < c)
    (hstart : ∃ q ∈ coveredStates M L c,
      1 - γ ≤ rerootAccuracy M L (insert q (dominatedStates M L R q)) q) :
    ∃ s ∈ o.result.hypothesis.tree.leaves, (uniformStrings L).real
      {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)}
        ≤ γ + R * (L + 1) / ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) * c := by
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
  set occ : σ → ℝ := fun q => ∑ i ∈ Finset.Icc k L,
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
  have hE := wrongEdge_occupancy_le M dist o.read H (by omega : k < L + 1)
    (by omega : L + 1 ≤ L₁) WE a haW hchk (fun q => hfew q (a q))
  rw [Finset.Ico_add_one_right_eq_Icc] at hE
  have hE' : ∑ q ∈ WE, occ q ≤ 2 * (((Fintype.card α : ℝ) + 1) * N * εB + WE.card * (N * εm)) :=
    hE
  have hEc : (WE.card : ℝ) ≤ Fintype.card σ := by exact_mod_cast Finset.card_le_univ WE
  have hSm := wrongSide_mass_le M dist D hd0 hd1
    (fun w => familyReadDist_sum O M _ _ _ _ ⟨w, rfl⟩) WS (fun q hq => (Finset.mem_filter.1 hq).2)
    hmis' hund'
  have hoccD : ∀ q, occ q ≤ N * D.real {p | M.eval p.toList = q} := by
    intro q
    rw [hDd, windowStrings_real hkL₁, mul_div_cancel₀ _ hN0.ne']
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.Icc_subset_Icc le_rfl (by omega))
      fun _ _ _ => measureReal_nonneg
  have hWS : ∑ q ∈ WS, occ q ≤ N * (2 * εr + εu) := by
    calc ∑ q ∈ WS, occ q ≤ ∑ q ∈ WS, N * D.real {p | M.eval p.toList = q} :=
          Finset.sum_le_sum fun q _ => hoccD q
      _ = N * ∑ q ∈ WS, D.real {p | M.eval p.toList = q} := (Finset.mul_sum _ _ _).symm
      _ ≤ N * (2 * εr + εu) := mul_le_mul_of_nonneg_left hSm hN0.le
  have hWrong : ∑ q ∈ WE ∪ WS, occ q < c := by
    have hu := Finset.sum_union_inter (s₁ := WE) (s₂ := WS) (f := occ)
    have hi0 : 0 ≤ ∑ q ∈ WE ∩ WS, occ q := Finset.sum_nonneg fun q _ => hocc0 q
    have h1 : (WE.card : ℝ) * (N * εm) ≤ Fintype.card σ * (N * εm) :=
      mul_le_mul_of_nonneg_right hEc (by positivity)
    have hb : N * (2 * (((Fintype.card α : ℝ) + 1) * εB + Fintype.card σ * εm) + (2 * εr + εu))
        < c := hbnd
    nlinarith
  -- every state is visited from `⌈L/2⌉` on at most its occupancy
  have hvis : ∀ q, windowVisitRate M L q ≤ occ q := fun q =>
    occupancy_of_covered M (c := windowVisitRate M L q)
      (show windowVisitRate M L q ≤ windowVisitRate M L q from le_rfl)
  obtain ⟨q₀, hq₀c, hacc⟩ := hstart
  have hq₀ : q₀ ∉ WE ∪ WS := by
    intro h
    have h1 := Finset.single_le_sum (fun q _ => hocc0 q) h
    have h2 : c ≤ occ q₀ := occupancy_of_covered M hq₀c
    linarith
  set C : ℝ := R * (L + 1) / ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) with hCd
  have hC0 : 0 ≤ C := by positivity
  set Dom := dominatedStates M L R q₀ with hDomd
  set T : Finset σ := (WE ∪ WS).filter (· ∈ Dom) with hTd
  set V : Set (FreeMonoid α) :=
    ⋃ q' ∈ T, {x | ∃ i ≤ L, M.evalFrom q₀ (prefixOf x i).toList = q'} with hVd
  set G : Set (FreeMonoid α) := {x | (∀ i ≤ L, M.evalFrom q₀ (prefixOf x i).toList ∈ insert q₀ Dom)
    ∧ (M.evalFrom q₀ x.toList ∈ M.accept ↔ M.eval x.toList ∈ M.accept)} with hGd
  -- off the runs through a wrong dominated state, a good string is classified as re-rooting does
  have hsub : {x | ¬ H.Classifies (Random.leafOf M side H.tree q₀) x (M.eval x.toList ∈ M.accept)}
      ⊆ (Gᶜ ∪ V) ∪ {x | x.toList.length ≠ L} := by
    intro x hx
    by_contra hcon
    simp only [Set.mem_union, Set.mem_compl_iff, Set.mem_setOf_eq, not_or, not_not] at hcon
    obtain ⟨⟨hG, hV⟩, hl⟩ := hcon
    apply hx
    refine classifies_of_matches M side H hroot {q | q ∉ WE ∪ WS} (fun q hq b => ?_)
      (fun q hq => ?_) q₀ x (fun i hi => ?_) hG.2
    · by_contra hb
      exact hq (Finset.mem_union_left _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, b, hb⟩))
    · by_contra hb
      exact hq (Finset.mem_union_right _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, hb⟩))
    · have hi' : i ≤ L := hl ▸ hi
      intro hw
      rcases Set.mem_insert_iff.1 (hG.1 i hi') with h0 | hd
      · exact hq₀ (h0 ▸ hw)
      · exact hV (Set.mem_iUnion₂.2 ⟨_, Finset.mem_filter.2 ⟨hw, hd⟩, i, hi', rfl⟩)
  have hnull : (uniformStrings (α := α) L).real {x | x.toList.length ≠ L} = 0 := by
    rw [measureReal_def, uniformStrings_length, ENNReal.toReal_zero]
  have hVm : (uniformStrings L).real V ≤ C * c := by
    refine (measureReal_biUnion_finset_le _ _).trans ?_
    calc ∑ q' ∈ T, (uniformStrings L).real
          {x : FreeMonoid α | ∃ i ≤ L, M.evalFrom q₀ (prefixOf x i).toList = q'}
        ≤ ∑ q' ∈ T, C * occ q' := Finset.sum_le_sum fun q' hq' => by
          have hd : rerootVisitRate M L q₀ q' ≤ C * windowVisitRate M L q' :=
            (Finset.mem_filter.1 hq').2
          exact hd.trans (mul_le_mul_of_nonneg_left (hvis q') hC0)
      _ ≤ ∑ q' ∈ WE ∪ WS, C * occ q' :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
            fun q _ _ => mul_nonneg hC0 (hocc0 q)
      _ = C * ∑ q' ∈ WE ∪ WS, occ q' := (Finset.mul_sum _ _ _).symm
      _ ≤ C * c := mul_le_mul_of_nonneg_left hWrong.le hC0
  refine ⟨Random.leafOf M side H.tree q₀, Random.leafOf_mem M side _ _, ?_⟩
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans ((measureReal_union_le _ _).trans ?_)
  rw [hnull, add_zero]
  refine (measureReal_union_le _ _).trans ?_
  rw [measureReal_compl MeasurableSpace.measurableSet_top, probReal_univ]
  have : 1 - γ ≤ (uniformStrings L).real G := hacc
  linarith

end Bridge

/-- The claim's rates, from the capped coverage `m`. -/
lemma summed_rates {Q A N ℓ m ξ ε εcov ι : ℝ} (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hN : 1 ≤ N)
    (hNℓ : ℓ ≤ 2 * N) (hm : 0 < m) (hm1 : m ≤ 1) (hξ : 0 < ξ)
    (he : ε = m / (3 * (A + Q + 1) * N)) (hεc : εcov = min ε (ξ / (32 * N)))
    (hι : ι = min (εcov / 8) (ε / (4608 * Q * (Q + 1)))) :
    0 < ε ∧ ε ≤ 1 / 2 ∧ 0 < εcov ∧ εcov ≤ ε ∧ 0 < ι ∧ ι ≤ εcov / 8
      ∧ N * (2 * ((A + 1) * (ε + ε / 96) + Q * (ε + ε / 96))
        + (2 * (1025 / 1024 * εcov) + 3 / 8 * εcov)) < m
      ∧ ℓ * ε < 3 / 8 := by
  have hX : 0 < 3 * (A + Q + 1) * N := by positivity
  have hq : ε * (3 * (A + Q + 1) * N) = m := by rw [he]; field_simp
  have he0 : 0 < ε := by rw [he]; positivity
  have hεc0 : 0 < εcov := by rw [hεc]; exact lt_min he0 (by positivity)
  have hεcε : εcov ≤ ε := by rw [hεc]; exact min_le_left _ _
  have hι0 : 0 < ι := by rw [hι]; exact lt_min (by positivity) (by positivity)
  have hι8 : ι ≤ εcov / 8 := by rw [hι]; exact min_le_left _ _
  have h12 : 12 ≤ 3 * (A + Q + 1) * N := by nlinarith
  refine ⟨he0, ?_, hεc0, hεcε, hι0, hι8, ?_, ?_⟩
  · have : ε * 12 ≤ 1 := by nlinarith
    linarith
  · have hX0 : 0 < ε * N := by positivity
    have hY : εcov * N ≤ ε * N := mul_le_mul_of_nonneg_right hεcε (by linarith)
    have h4 : 4 * (ε * N) ≤ (A + Q + 1) * (ε * N) :=
      mul_le_mul_of_nonneg_right (by linarith) hX0.le
    have e : N * (2 * ((A + 1) * (ε + ε / 96) + Q * (ε + ε / 96))
        + (2 * (1025 / 1024 * εcov) + 3 / 8 * εcov))
        = 97 / 48 * ((A + Q + 1) * (ε * N)) + 1217 / 512 * (εcov * N) := by ring
    have e2 : m = 3 * ((A + Q + 1) * (ε * N)) := by rw [← hq]; ring
    rw [e, e2]
    linarith
  · have h2 : ℓ * ε ≤ 2 * N * ε := mul_le_mul_of_nonneg_right hNℓ he0.le
    have h3 : 2 * N * ε * (3 * (A + Q + 1)) = 2 * m := by rw [← hq]; ring
    have h4 : 2 * N * ε * 12 ≤ 2 * m := by
      rw [← h3]
      exact mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    nlinarith

/-- The relative rule's cost: `C ≤ 2R`, so `C m ≤ 3ξ/4`. -/
lemma summed_harm_le {R ξ m : ℝ} (L : ℕ) (hR : 0 < R) (hm0 : 0 ≤ m)
    (hm : m ≤ 3 * ξ / (8 * R)) :
    R * (L + 1) / ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) * m ≤ 3 * ξ / 4 := by
  have hd : (0 : ℝ) < ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) := by
    exact_mod_cast (by omega : 0 < L + 1 - (L + 1) / 2)
  have hd2 : ((L : ℝ) + 1) ≤ 2 * ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) := by
    have : L + 1 ≤ 2 * (L + 1 - (L + 1) / 2) := by omega
    exact_mod_cast this
  have hC : R * (L + 1) / ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) ≤ 2 * R := by
    rw [div_le_iff₀ hd]; nlinarith
  calc R * (L + 1) / ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) * m ≤ 2 * R * m :=
        mul_le_mul_of_nonneg_right hC hm0
    _ ≤ 2 * R * (3 * ξ / (8 * R)) := mul_le_mul_of_nonneg_left hm (by positivity)
    _ = 3 * ξ / 4 := by field_simp; ring

set_option maxHeartbeats 8000000 in -- the claim's settings meet `loop_simple_checks`'s conditions
theorem learner_loop_dfa_summed_simple_holds : LearnerLoopDFASummedSimple := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ c R γ δ pAP ξ Q A k L₁ N m ε εcov δ' ι s τ J S B₃ Wm W PB
    RB cap Kc n hL hη hη₀ hc hR hδ hδ1 hpAP hξ hA2 hpAPu hceil hk1 hk2 hk3 hk4
  classical
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hA2' : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hQ0 : (0 : ℝ) ≤ Q := by linarith
  have hkL₁ : k ≤ L₁ := by show (L + 1) / 2 ≤ L + Fintype.card σ + 3; omega
  have hNe : N = ((L₁ : ℕ) : ℝ) + 1 - k := rfl
  have hN1 : (1 : ℝ) ≤ N := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL₁
    rw [hNe]; linarith
  obtain ⟨q₀, hq₀, hq₀acc⟩ := hceil
  have hc1 : c ≤ 1 := hq₀.trans measureReal_le_one
  have hm0 : 0 < m := lt_min hc (by positivity)
  have hmc : m ≤ c := min_le_left _ _
  have hm1 : m ≤ 1 := hmc.trans hc1
  have hwN : 2 + (Nat.clog 2 L₁ : ℝ) ≤ 2 * N := by
    have h1 : 1 ≤ L₁ := by change 1 ≤ L + Fintype.card σ + 3; omega
    have h2 : Nat.clog 2 L₁ ≤ L₁ - 1 :=
      Nat.clog_le_of_le_pow (by have := (L₁ - 1).lt_two_pow_self; omega)
    have : 2 + Nat.clog 2 L₁ ≤ L₁ + 1 := by omega
    have : 2 + (Nat.clog 2 L₁ : ℝ) ≤ L₁ + 1 := by exact_mod_cast this
    have : (2 * k : ℕ) ≤ L₁ + 1 := by show 2 * ((L + 1) / 2) ≤ L + Fintype.card σ + 3 + 1; omega
    have : (2 : ℝ) * k ≤ L₁ + 1 := by exact_mod_cast this
    rw [hNe]
    linarith
  obtain ⟨hε0, hε12, hεc0, hεcε, hι0, hι8, hbad, hιL⟩ :=
    summed_rates (ℓ := 2 + (Nat.clog 2 L₁ : ℝ)) hQ1 hA2' hN1 hwN hm0 hm1 hξ rfl rfl rfl
  have hεcov1 : εcov ≤ 1 := by linarith
  have hδ'0 : 0 < δ' := by show 0 < δ / (8 * (Q + 1)); positivity
  have hδ'1' : δ' < 1 := by
    show δ / (8 * (Q + 1)) < 1
    rw [div_lt_one (by positivity)]
    linarith
  have hδ'1 : δ' ≤ 1 := hδ'1'.le
  have hKc1 : (1 : ℝ) ≤ Kc := by
    show (1 : ℝ) ≤ (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ
      * (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α)
    have h1 : (1 : ℝ) ≤ (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ :=
      one_le_pow₀ (by nlinarith [pow_nonneg (show (0 : ℝ) ≤ Q + 1 by linarith) 3])
    have h2 : (1 : ℝ) ≤ (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α) :=
      one_le_pow₀ (by linarith)
    nlinarith
  have hlogK : 2 ≤ Real.log (16 * (Q + 1) * Kc / δ') := by
    rw [Real.le_log_iff_exp_le (by positivity)]
    have he := Real.exp_one_lt_d9
    have : Real.exp 2 ≤ 16 := by
      calc Real.exp 2 = Real.exp 1 * Real.exp 1 := by rw [← Real.exp_add]; norm_num
        _ ≤ 3 * 3 := by gcongr <;> linarith
        _ ≤ 16 := by norm_num
    refine this.trans ?_
    rw [le_div_iff₀ hδ'0]
    nlinarith only [hδ'1, hQ0, hKc1, mul_nonneg hQ0 (sub_nonneg.2 hKc1)]
  -- the walk starts after at least one letter
  have hk0 : 1 ≤ k := by
    by_contra h
    have hk : k = 0 := by omega
    rw [hk, pow_zero, inv_one, mul_one] at hk2
    linarith
  have hkL : k ≤ L := by
    have : (L + 1) / 2 ≤ L ∨ L = 0 := by omega
    rcases this with h | h
    · exact h
    · exfalso; have : k = 0 := by show (L + 1) / 2 = 0; omega
      omega
  let P : PrefixDistribution α := ⟨windowStrings k L₁, windowStrings_isProbability hkL₁, L₁,
    (windowStrings_length k L₁).mono fun x hx => hx.2⟩
  have hpm : prefixMax P.D k ≤ (A ^ k)⁻¹ := by
    have := prefixMax_windowStrings_le (α := α) (le_refl k) hkL₁
    rwa [one_div] at this
  have : IsProbabilityMeasure P.D := windowStrings_isProbability hkL₁
  have hpm0 : 0 ≤ prefixMax P.D k := (prefixMax_pos P.D k).le
  obtain ⟨settings, hwalk, hdis, hbud, hnp, hmain⟩ := loop_simple_checks M O P (shortAt M (L + 1))
    η₀ εcov ε δ' pAP k
    hL hη hη₀ hεc0 hεcov1 hε0 hε12 hδ'0 hδ'1' hpAP (by show 1 ≤ L + Fintype.card σ + 3; omega)
    hk0 hιL (hpm.trans hk1)
    ((mul_le_mul_of_nonneg_left hpm (mul_nonneg (by norm_num) (by linarith [hlogK]))).trans hk2)
    ((mul_le_mul_of_nonneg_left hpm (by positivity)).trans hk4)
  refine ⟨settings, hwalk, hdis, hbud, ?_⟩
  have hcap0 : 0 < cap := by
    have h1 : 0 < (A ^ k)⁻¹ := by positivity
    by_contra hc0
    push Not at hc0
    have : cap * ε / 1536 ≤ 0 := by
      have := mul_nonpos_of_nonpos_of_nonneg hc0 hε0.le
      linarith
    linarith
  have hatom : ∀ v : FreeMonoid α,
      (uniformStrings (L₁ + Fintype.card σ + 3)).real {v} ≤ cap := fun v => by
    refine (uniformStrings_singleton_le _ v).trans ?_
    have h1 : 1 / A ^ (L₁ + Fintype.card σ + 3) ≤ (A ^ k)⁻¹ := by
      rw [one_div]
      exact inv_anti₀ (by positivity) (pow_le_pow_right₀ (by linarith) (by omega))
    refine h1.trans (hk1.trans ?_)
    have : cap * ε ≤ cap * 1 := mul_le_mul_of_nonneg_left (by linarith) hcap0.le
    linarith
  set Dsf : Measure (FreeMonoid α) := uniformStrings (L₁ + Fintype.card σ + 3) with hDsfd
  have hD := hmain Dsf inferInstance (uniformStrings_length _) hpAPu hatom
  set T := settings.level3.probeBudget
  set μL := loopMeasure μ α (Fintype.card σ + 1) T
  set β := Measure.pi fun _ : Fin n => uniformStrings (α := α) L
  set Nn := P.L + T + 2
  set Ls := P.L + Fintype.card σ + 3
  set t : ℝ := ξ / 16 with htd
  set Bloop : Set (Ω × (Fin (Fintype.card σ + 1) → RoundDraws α (Fintype.card σ + 1) T)) :=
    {p | let rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p)
      ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
          o.Checks M O P.D k (1025 / 1024 * εcov) (3 / 8 * εcov) (ε + ε / 96) (shortAt M (L + 1))
            (ε + ε / 96))
        ∧ RoundsFresh settings O P.D Dsf Nn Ls p)} with hBloopd
  set Sel : ℕ → Set ((Ω × (Fin (Fintype.card σ + 1) → RoundDraws α (Fintype.card σ + 1) T))
      × (Fin n → FreeMonoid α)) := fun r =>
    if hr : r < Fintype.card σ + 1 then
      selFail settings O P.D Dsf Nn Ls L n (Fintype.card σ + 2) t r hr else ∅
    with hSeld
  -- the loop's last round, consistent and fresh, picks well unless its scores stray or a
  -- held-out string collides
  have hsub : {q : (Ω × (Fin (Fintype.card σ + 1) → RoundDraws α (Fintype.card σ + 1) T))
        × (Fin n → FreeMonoid α) |
      let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
        (uniformStrings (L₁ + Fintype.card σ + 3)) (persistDraws q.1)
      ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
          o.result.ending = .consistent ∧ (uniformStrings L).real {x | ¬
            o.result.hypothesis.Classifies (pickLeaf o.result.hypothesis o.read q.2) x
              (M.eval x.toList ∈ M.accept)} ≤ γ + ξ)}
      ⊆ (Prod.fst ⁻¹' Bloop) ∪ ⋃ r ∈ Finset.range (Fintype.card σ + 1), Sel r := by
    intro q hq
    by_cases hlp : q.1 ∈ Bloop
    · exact Or.inl hlp
    right
    have hlp' := not_not.1 hlp
    obtain ⟨⟨hlen, o, ho, hch⟩, hfr⟩ := hlp'
    have hcons : o.result.ending = .consistent := hch.1.1
    obtain ⟨r, hr, -, tt, hst, hG, hOk, hB, hF, hread, hhyp⟩ :=
      stageSel_last settings O P Dsf Ls q.1 hfr o ho hcons
    refine Set.mem_iUnion₂.2 ⟨r, Finset.mem_range.2 hr, ?_⟩
    simp only [hSeld, dif_pos hr]
    by_contra hnot
    apply hq
    refine ⟨hlen, o, ho, hcons, ?_⟩
    have hnot' : (q.1.1, q.2) ∉ SelBad O L n (Fintype.card σ + 2) t (some (tt, resCode o.result))
        ∧ (q.1.2, q.2) ∈ SelDk settings P.D L n r hr (some (tt, resCode o.result)) := by
      constructor
      · intro h
        apply hnot
        show SelG r (stageSel settings O P.D Dsf Nn r hr q.1.1 q.1.2) ∧ _
        rw [hst]
        exact ⟨hG, hOk, Or.inl h⟩
      · by_contra h
        apply hnot
        show SelG r (stageSel settings O P.D Dsf Nn r hr q.1.1 q.1.2) ∧ _
        rw [hst]
        exact ⟨hG, hOk, Or.inr h⟩
    obtain ⟨hnb, hdk⟩ := hnot'
    obtain ⟨hinj, -⟩ := hdk
    have hsf : SuffixFree o.family := by
      have hlenF : ∀ v ∈ o.family, v.toList.length = Ls := fun v hv =>
        hOk.1.1 ⟨r, hr⟩ le_rfl v (hOk.2.1 (hF ▸ hv))
      intro v hv v' hv' hsuf
      apply FreeMonoid.toList.injective
      exact hsuf.eq_of_length ((hlenF v hv).trans (hlenF v' hv').symm)
    have hleaves : o.result.hypothesis.tree.leaves.length ≤ Fintype.card σ + 2 :=
      (Random.classSet_props _ _ hch.2.2.1).2
    have hgood : (q.1.1, q.2) ∉ scoreBad O o.result.hypothesis o.family o.state.lo
        (o.state.hi + 1) L n t := by
      intro h
      apply hnb
      show (q.1.1, q.2) ∈ (if SuffixFree tt.2.2.1
          ∧ (resCode o.result).hyp.tree.leaves.length ≤ Fintype.card σ + 2
        then scoreBad O (resCode o.result).hyp tt.2.2.1 tt.2.1.lo (tt.2.1.hi + 1) L n t else ∅)
      rw [hF, hB, hhyp, if_pos ⟨hsf, hleaves⟩]
      exact h
    -- the best leaf, and the family's reads on the strings of length `L`
    have hs := dfa_of_checks_summed M O hL o L hch (by linarith) hR (c := m) ?_
      ⟨q₀, hmc.trans hq₀, hq₀acc⟩
    · have hn : 0 < n := by
        refine Nat.ceil_pos.2 (div_pos (mul_pos (by norm_num) (Real.log_pos ?_)) (by positivity))
        rw [one_lt_div hδ]
        nlinarith only [hQ0, hδ1, mul_nonneg hQ0 hQ0]
      have hεcov_ξ : εcov ≤ ξ / (32 * N) := min_le_right _ _
      have hNε : N * εcov ≤ ξ / 32 := by
        have h := mul_le_mul_of_nonneg_left hεcov_ξ (by linarith : (0 : ℝ) ≤ N)
        rw [show N * (ξ / (32 * N)) = ξ / 32 by field_simp] at h
        exact h
      have hu : ∫ x, readProb O o.family o.state.lo (o.state.hi + 1) x .undecided
          ∂(uniformStrings L) ≤ 3 / 8 * (N * εcov) := by
        refine (integral_uniform_le_window (L₁ := L₁) hkL (by omega) _
          fun x => measureReal_nonneg).trans ?_
        exact (mul_le_mul_of_nonneg_left hch.1.2.2 (zero_le_one.trans hN1)).trans
          (le_of_eq (by ring))
      have hm : ∫ x, misreadProb O o.family o.state.lo (o.state.hi + 1) x ∂(uniformStrings L)
          ≤ 1025 / 1024 * (N * εcov) := by
        refine (integral_uniform_le_window (L₁ := L₁) hkL (by omega) _
          fun x => measureReal_nonneg).trans ?_
        exact (mul_le_mul_of_nonneg_left hch.1.2.1 (zero_le_one.trans hN1)).trans
          (le_of_eq (by ring))
      have h := pickLeaf_le M O o.result.hypothesis o.family o.state.lo (o.state.hi + 1) hL L n hn
        t (3 / 8 * (N * εcov)) (1025 / 1024 * (N * εcov)) _ q.1.1 q.2 hinj hgood hu hm hs
      rw [hread]
      have hharm := summed_harm_le L hR hm0.le (min_le_right _ _)
      have : γ + R * (L + 1) / ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) * m + 2 * t
          + 3 / 8 * (N * εcov) + 2 * (1025 / 1024 * (N * εcov)) ≤ γ + ξ := by
        have h0 : 0 ≤ N * εcov := mul_nonneg (zero_le_one.trans hN1) hεc0.le
        rw [htd]; linarith
      exact h.trans this
    · exact hbad
  -- the loop's failure, and each round's selection
  have hBl : (μL.prod β).real (Prod.fst ⁻¹' Bloop) ≤ δ / 2 := by
    have e : (μL.prod β).real (Prod.fst ⁻¹' Bloop) = μL.real Bloop := by
      rw [← Set.prod_univ, measureReal_def, measureReal_def, Measure.prod_prod, measure_univ,
        mul_one]
    rw [e]
    refine hD.trans ?_
    show (Q + 1) * (5 * (δ / (8 * (Q + 1))) / 2) ≤ δ / 2
    have e : (Q + 1) * (5 * (δ / (8 * (Q + 1))) / 2) = 5 * δ / 16 := by field_simp; ring
    rw [e]; linarith
  have hsel : ∀ r ∈ Finset.range (Fintype.card σ + 1), (μL.prod β).real (Sel r)
      ≤ δ / (2 * (Q + 1)) := by
    intro r hr
    have hr' := Finset.mem_range.1 hr
    simp only [hSeld, dif_pos hr']
    have hs0 : 0 < s := by show 0 < 1 / 2 - η₀; linarith
    have hs1 : s ≤ 1 / 2 := by
      show 1 / 2 - η₀ ≤ 1 / 2
      linarith
    have hp1 : pAP ≤ 1 := hpAPu.trans measureReal_le_one
    have hτ0 : 0 < τ := lt_min (mul_pos (div_pos hεc0 (by norm_num))
      (pow_pos (gateA_pos hη0 hη₀) 2)) hι0
    have hτ1 : τ ≤ 1 := (min_le_right _ _).trans (by linarith)
    have hW1 : 1 ≤ W := le_max_right _ _
    have hRB0 : 0 ≤ RB := rb_nonneg hQ0 hs0 hs1 hpAP hp1 hδ'0 hδ'1 hτ0 hτ1
      (Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith))
    refine (loop_select_round_le settings O P Dsf Ls L n (Fintype.card σ + 2) ⌊RB⌋₊ t
      (by positivity) (hwalk ▸ hkL) hnp r hr').trans ?_
    rw [hwalk]
    have e1 : ((Fintype.card σ + 2 : ℕ) : ℝ) = Q + 2 := by push_cast; rfl
    have e2 : ((Fintype.card σ + 1 : ℕ) : ℝ) + 1 = Q + 2 := by push_cast; ring
    rw [e1, e2, htd]
    have hn' : 32 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / (ξ / 4) ^ 2 ≤ n := by
      rw [show 32 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / (ξ / 4) ^ 2
        = 512 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / ξ ^ 2 by ring]
      exact Nat.le_ceil _
    have h := dfa_select_arith hQ0 hA2' hδ (by positivity : 0 < ξ / 4) hkL hn' hbud
      (Nat.floor_le hRB0) hk3
    rwa [show ξ / 4 / 4 = ξ / 16 by ring] at h
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans ?_
  refine (measureReal_union_le _ _).trans ?_
  refine (add_le_add hBl ((measureReal_biUnion_finset_le _ _).trans
    (Finset.sum_le_sum hsel))).trans (le_of_eq ?_)
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  show δ / 2 + ((Fintype.card σ + 1 : ℕ) : ℝ) * (δ / (2 * (Q + 1))) = δ
  push_cast
  field_simp
  ring

/-- `dfa_rate_bounds` for any `ε` with `1/ε ≤ aε ℓ` and `1 ≤ aε`. -/
lemma summed_rate_bounds {Q ℓ N ξ η₀ ε εcov ι τ s aε Ce Θ : ℝ} (hQ : 1 ≤ Q)
    (hℓ : 1 ≤ ℓ) (hN1 : 1 ≤ N) (hNℓ : N ≤ ℓ) (hξ : 0 < ξ) (hη0 : 0 ≤ η₀) (hη : η₀ < 1 / 2)
    (hε0 : 0 < ε) (hinvε : 1 / ε ≤ aε * ℓ) (haε : 1 ≤ aε)
    (hεc : εcov = min ε (ξ / (8 * N))) (hι : ι = min (εcov / 8) (ε / (4608 * Q * (Q + 1))))
    (hs : s = 1 / 2 - η₀) (hτ : τ = min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι)
    (hCe : Ce = aε + 8 / ξ) (hΘ : Θ = Ce / (2 * s ^ 2) + 8 * Ce + 4608 * Q * (Q + 1) * aε) :
    0 < ι ∧ 1 / ι ≤ Θ * ℓ ∧ 0 < τ ∧ 1 / τ ≤ Θ * ℓ ∧ 1 ≤ Θ := by
  have hs0 : 0 < s := by rw [hs]; linarith
  have haε0 : 0 < aε := by linarith
  have hCe0 : 0 < Ce := by rw [hCe]; positivity
  have haCe : aε ≤ Ce := by
    have : 0 ≤ 8 / ξ := by positivity
    rw [hCe]; linarith
  have h8Ce : 8 / ξ ≤ Ce := by rw [hCe]; linarith
  have hεc0 : 0 < εcov := by rw [hεc]; exact lt_min hε0 (by positivity)
  have hinvεc : 1 / εcov ≤ Ce * ℓ := by
    rw [hεc]
    rcases min_choice ε (ξ / (8 * N)) with h | h <;> rw [h]
    · exact hinvε.trans (mul_le_mul_of_nonneg_right haCe (by linarith))
    · rw [one_div_div]
      calc 8 * N / ξ ≤ 8 * ℓ / ξ := by gcongr
        _ = 8 / ξ * ℓ := by ring
        _ ≤ Ce * ℓ := mul_le_mul_of_nonneg_right h8Ce (by linarith)
  have hQQ : 0 < 4608 * Q * (Q + 1) := by positivity
  have hι0 : 0 < ι := by rw [hι]; exact lt_min (by positivity) (by positivity)
  have hΘ1 : 8 * Ce + 4608 * Q * (Q + 1) * aε ≤ Θ := by
    rw [hΘ]; have : 0 ≤ Ce / (2 * s ^ 2) := by positivity
    linarith
  have hinvι : 1 / ι ≤ Θ * ℓ := by
    rw [hι]
    rcases min_choice (εcov / 8) (ε / (4608 * Q * (Q + 1))) with h | h <;> rw [h]
    · rw [one_div_div]
      have h1 : 8 / εcov = 8 * (1 / εcov) := by ring
      have h2 : 8 * (1 / εcov) ≤ 8 * (Ce * ℓ) := by linarith
      have h3 : 8 * (Ce * ℓ) ≤ Θ * ℓ := by
        have : 8 * Ce ≤ Θ := by
          have : 0 ≤ 4608 * Q * (Q + 1) * aε := by positivity
          linarith
        nlinarith
      linarith
    · rw [one_div_div]
      have h1 : 4608 * Q * (Q + 1) / ε = 4608 * Q * (Q + 1) * (1 / ε) := by ring
      have h2 : 4608 * Q * (Q + 1) * (1 / ε) ≤ 4608 * Q * (Q + 1) * (aε * ℓ) :=
        mul_le_mul_of_nonneg_left hinvε hQQ.le
      have h3 : 4608 * Q * (Q + 1) * (aε * ℓ) ≤ Θ * ℓ := by
        have : 4608 * Q * (Q + 1) * aε ≤ Θ := by
          have : 0 ≤ 8 * Ce := by positivity
          linarith
        have e : 4608 * Q * (Q + 1) * (aε * ℓ) = 4608 * Q * (Q + 1) * aε * ℓ := by ring
        rw [e]
        exact mul_le_mul_of_nonneg_right this (by linarith)
      linarith
  have ha : 2 * s ≤ (1 - 2 * η₀) / (1 - η₀) := by
    rw [le_div_iff₀ (by linarith), hs]
    have := mul_le_mul_of_nonneg_left (show 1 - η₀ ≤ 1 by linarith)
      (show (0 : ℝ) ≤ 1 - 2 * η₀ by linarith)
    linarith
  have hτ0 : 0 < τ := by
    rw [hτ]
    exact lt_min (mul_pos (by positivity) (pow_pos (div_pos (by linarith) (by linarith)) 2)) hι0
  have hΘ2 : Ce / (2 * s ^ 2) ≤ Θ := by
    rw [hΘ]; have : 0 ≤ 8 * Ce + 4608 * Q * (Q + 1) * aε := by positivity
    linarith
  have hinvτ : 1 / τ ≤ Θ * ℓ := by
    rw [hτ]
    rcases min_choice ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι with h | h <;> rw [h]
    · have hsq : (2 * s) ^ 2 ≤ ((1 - 2 * η₀) / (1 - η₀)) ^ 2 :=
        pow_le_pow_left₀ (by positivity) ha 2
      have hlow : 2 * s ^ 2 * εcov ≤ (εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2 := by
        have := mul_le_mul_of_nonneg_left hsq (show 0 ≤ εcov / 2 by positivity)
        nlinarith
      calc 1 / ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ≤ 1 / (2 * s ^ 2 * εcov) :=
            one_div_le_one_div_of_le (by positivity) hlow
        _ = 1 / (2 * s ^ 2) * (1 / εcov) := by field_simp
        _ ≤ 1 / (2 * s ^ 2) * (Ce * ℓ) :=
            mul_le_mul_of_nonneg_left hinvεc (by positivity)
        _ = Ce / (2 * s ^ 2) * ℓ := by field_simp
        _ ≤ Θ * ℓ := mul_le_mul_of_nonneg_right hΘ2 (by linarith)
    · exact hinvι
  have hΘge : 1 ≤ Θ := by
    have hCe1 : 1 ≤ Ce := haε.trans haCe
    calc (1 : ℝ) ≤ 8 * Ce := by linarith
      _ ≤ 8 * Ce + 4608 * Q * (Q + 1) * aε := le_add_of_nonneg_right (by positivity)
      _ ≤ Θ := hΘ1
  exact ⟨hι0, hinvι, hτ0, hinvτ, hΘge⟩

/-- `dfa_tail` for any `aε ≥ 1`. -/
lemma summed_tail {A ℓ ε cap aε E₁ E E₂ E₃ lg Q n J RB R₀ B₃ K₁ δ δ' s pAP Θ P₀ L T : ℝ}
    {k : ℕ} (hA : 2 ≤ A) (hQ : 0 ≤ Q) (hL : 0 ≤ L) (haε : 1 ≤ aε) (hs0 : 0 < s)
    (hs1 : s ≤ 1 / 2) (hpAP : 0 < pAP) (hδ'0 : 0 < δ') (hδ : 0 < δ) (hΘ1 : 1 ≤ Θ)
    (hR₀ : 0 ≤ R₀) (hP₀ : 0 ≤ P₀) (hlR₀ : 0 ≤ Real.logb 2 (R₀ + 1)) (hK₁0 : 0 ≤ K₁)
    (hn : 0 ≤ n) (hℓd : ℓ = L + Q + 4) (hℓ1 : 1 ≤ ℓ) (hε0 : 0 < ε) (hinvε : 1 / ε ≤ aε * ℓ)
    (hcap : 1 / (E₁ * ℓ ^ 5) ≤ cap) (hlg : 0 ≤ lg)
    (hE₁d : E₁ = (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) / s
      + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ' + 2 ^ 16 * (P₀ + 2) / (pAP * δ')))
    (hEd : E = 1536 * aε * E₁ + 1536 * lg * aε)
    (hE₂d : E₂ = 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ + n) / δ)
    (hE₃d : E₃ = 16 * K₁ ^ 2 / δ') (hJ : J = Q + 2) (hRB0 : 0 ≤ RB) (hRB : RB ≤ R₀ * ℓ ^ 3)
    (hB₃0 : 0 ≤ B₃) (hB₃ : B₃ ≤ K₁ * ℓ) (hk : L ≤ 2 * k)
    (hT : T = 2 * (Real.log (E + E₂ + E₃) + 12) / Real.log A + Q + 4)
    (hL₀ : 2 * (Real.log (E + E₂ + E₃) + 6 * Real.log T) / Real.log A
      + 144 * (Real.log T - 2) / (Real.log A * (T * Real.log A - 12)) ≤ L) :
    (A ^ k)⁻¹ ≤ cap * ε / 1536 ∧ 1536 * lg * (A ^ k)⁻¹ ≤ ε
      ∧ (Q + 1) * n * (6 * J * RB + B₃ + n) * (A ^ k)⁻¹ ≤ δ / 4
      ∧ 16 * B₃ ^ 2 * (A ^ k)⁻¹ ≤ δ' := by
  have haε0 : 0 < aε := by linarith only [haε]
  have hE₁ : 1 ≤ E₁ := by
    rw [hE₁d]
    have h1 : 1 ≤ 2 ^ 20 * Θ * (R₀ + 1) / s := by
      rw [le_div_iff₀ hs0, one_mul]
      have : 1 ≤ Θ * (R₀ + 1) := one_le_mul_of_one_le_of_one_le hΘ1 (by linarith only [hR₀])
      linarith only [this, hs1]
    have h2 : 0 ≤ 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ' := by positivity
    have h3 : 0 ≤ 2 ^ 16 * (P₀ + 2) / (pAP * δ') := by positivity
    have h4 : (1 : ℝ) ≤ (Q + 3) ^ 2 := one_le_pow₀ (by linarith only [hQ])
    exact one_le_mul_of_one_le_of_one_le h4 (by linarith only [h1, h2, h3])
  have hE₂0 : 0 ≤ E₂ := by rw [hE₂d]; positivity
  have hE₃0 : 0 ≤ E₃ := by rw [hE₃d]; positivity
  have hEE : 1 ≤ E + E₂ := by
    have h1 : 0 ≤ 1536 * lg * aε := by positivity
    have h2 : 1 ≤ 1536 * aε * E₁ :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by norm_num) haε) hE₁
    rw [hEd]
    linarith only [h1, h2, hE₂0]
  have hpow3 : ℓ ^ 6 * (E + E₂ + E₃) ≤ A ^ k :=
    dfa_pow hA hℓd hQ hL (by linarith only [hEE, hE₃0]) hk hT hL₀
  have hℓ6 : 0 ≤ ℓ ^ 6 := pow_nonneg (by linarith only [hℓ1]) 6
  have hpow : ℓ ^ 6 * (E + E₂) ≤ A ^ k :=
    (mul_le_mul_of_nonneg_left (by linarith only [hE₃0]) hℓ6).trans hpow3
  have hAk : 0 < A ^ k := pow_pos (by linarith only [hA]) k
  obtain ⟨h1, h2, h3⟩ := dfa_final (Ak := A ^ k) hAk hℓ1 hε0 hinvε haε0 hcap
    (by linarith only [hE₁]) hlg hEd hE₂d hQ hn hδ hJ hRB0 hRB hB₃0 hB₃ hpow
  exact ⟨h1, h2, h3, dfa_noise_tail hℓ1 hB₃0 hB₃ hE₃d hδ'0 hEE hpow3 hAk⟩

set_option maxRecDepth 4000 in
set_option maxHeartbeats 8000000 in -- unfolding the simple claim's settings
theorem learner_loop_dfa_summed_holds : LearnerLoopDFASummed := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ c R γ δ pAP ξ Q A k L₁ N m ε δ' s S B₃ n aε Ce Θ K₁ Wm W₀
    P₀ R₀ E₁ Kc E E₂ E₃ T L₀ hL hη hη₀ hc hR hδ hδ1 hpAP hξ hA2 hpAPu hceil hL₀
  classical
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hQ0 : (0 : ℝ) ≤ Q := by linarith
  have hA : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hp1 : pAP ≤ 1 := hpAPu.trans measureReal_le_one
  have hkL₁ : k ≤ L₁ := by show (L + 1) / 2 ≤ L + Fintype.card σ + 3; omega
  have hkr : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL₁
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg _
  have hN1 : 1 ≤ N := by show (1 : ℝ) ≤ (L₁ : ℝ) + 1 - k; linarith
  have hNℓ : N ≤ (L₁ : ℝ) + 1 := by show (L₁ : ℝ) + 1 - k ≤ _; linarith
  have hℓ1 : (1 : ℝ) ≤ (L₁ : ℝ) + 1 := by have : (0 : ℝ) ≤ L₁ := Nat.cast_nonneg _; linarith
  have hc1 : c ≤ 1 := by
    obtain ⟨q₀, hq₀, -⟩ := hceil
    exact hq₀.trans measureReal_le_one
  have hm0 : 0 < m := lt_min hc (by positivity)
  have hm1 : m ≤ 1 := (min_le_left _ _).trans hc1
  have hδ'0 : 0 < δ' := by show 0 < δ / (8 * (Q + 1)); positivity
  have hδ'1 : δ' ≤ 1 := by
    show δ / (8 * (Q + 1)) ≤ 1
    rw [div_le_one (by positivity)]
    linarith
  have hs0 : 0 < s := by show 0 < 1 / 2 - η₀; linarith
  have hs1 : s ≤ 1 / 2 := by show 1 / 2 - η₀ ≤ 1 / 2; linarith
  set ℓ : ℝ := (L₁ : ℝ) + 1 with hℓd
  set εcov : ℝ := min ε (ξ / (32 * N)) with hεcd
  set ι : ℝ := min (εcov / 8) (ε / (4608 * Q * (Q + 1))) with hιd
  set τ : ℝ := min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι with hτd
  set J : ℝ := Q + 2 with hJd
  set W : ℝ := max (Real.exp 1 * Real.sqrt δ' * Wm / 2) 1 with hWd
  set PB : ℝ := (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
    + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2 with hPBd
  set RB : ℝ := 161000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
    / (s ^ 4 * τ ^ 2) with hRBd
  set cap : ℝ := min (min (min (τ * s / (2 ^ 19 * (J + 3) * (RB + 1)))
        (δ' * min (εcov / 2) ι ^ 2 / (2 ^ 35 * J * (3 + 2 * Real.logb 2 (RB + 1)))))
      (min (δ' * (ι / 2) ^ 2 / (2 ^ 5 * (J + 1) * (3 + 2 * Real.logb 2 (RB + 1))))
        (ι / 2 / (8 * J * (RB + 1)))))
    (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
      (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1))))) with hcapd
  have hX : 0 < 3 * (A + Q + 1) := by positivity
  have hε0 : 0 < ε := by show 0 < m / (3 * (A + Q + 1) * N); positivity
  have hinvε : 1 / ε ≤ aε * ℓ := by
    show 1 / (m / (3 * (A + Q + 1) * N)) ≤ 3 * (A + Q + 1) / m * ℓ
    rw [one_div_div, div_mul_eq_mul_div]
    exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hNℓ hX.le) hm0.le
  have haε1 : 1 ≤ aε := by
    show 1 ≤ 3 * (A + Q + 1) / m
    rw [le_div_iff₀ hm0]
    linarith
  have hε1 : ε ≤ 1 := by
    show m / (3 * (A + Q + 1) * N) ≤ 1
    rw [div_le_one (by positivity)]
    nlinarith
  obtain ⟨hι0, hinvι, hτ0, hinvτ, hΘ1⟩ :=
    summed_rate_bounds (ℓ := ℓ) (ξ := ξ / 4) (εcov := εcov) (Ce := Ce) hQ1 hℓ1 hN1 hNℓ
      (by positivity) hη0 hη₀
      hε0 hinvε haε1 (by rw [show ξ / 4 / (8 * N) = ξ / (32 * N) by ring]) rfl rfl rfl
      (by show aε + 32 / ξ = aε + 8 / (ξ / 4); ring) rfl
  have haε0 : 0 < aε := by linarith
  have hτ1 : τ ≤ 1 := by
    have h1 : ι ≤ εcov / 8 := min_le_left _ _
    have h2 : εcov ≤ ε := min_le_left _ _
    exact (min_le_right _ _).trans (by linarith)
  have hS1 : 1 ≤ S := by
    show 1 ≤ Q * ((Q + 2) * A + 1) + 2 * A + 1
    have : 0 ≤ Q * ((Q + 2) * A + 1) := by positivity
    linarith
  have hlS : 0 ≤ Real.log (16 * S * (S + 1) / δ') := log_budget_nonneg hS1 hδ'0 hδ'1
  have hB₃ : B₃ ≤ K₁ * ℓ := budget_le (by linarith) hε0 hinvε hℓ1 hlS
  have hB₃0 : 0 ≤ B₃ := by
    show 0 ≤ 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
    positivity
  have hK₁0 : 0 ≤ K₁ := by
    show 0 ≤ 36 * S * aε * Real.log (16 * S * (S + 1) / δ') + S + 1
    positivity
  have hWW : W ≤ W₀ * ℓ ^ 2 := by
    have hWm0 : 0 ≤ Wm := by
      have : 0 ≤ Real.log (8 * S * (S + 1) / δ') :=
        Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith)
      show 0 ≤ 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
      positivity
    exact w_le hℓ1 hWm0
  have hW1 : 1 ≤ W := le_max_right _ _
  have hW₀1 : 1 ≤ W₀ := by
    have : 0 ≤ Real.exp 1 * Real.sqrt δ' * Wm / 2 := by
      have : 0 ≤ Real.log (8 * S * (S + 1) / δ') :=
        Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith)
      have : 0 ≤ Wm := by
        show 0 ≤ 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
        positivity
      positivity
    show 1 ≤ Real.exp 1 * Real.sqrt δ' * Wm / 2 + 1
    linarith
  obtain ⟨hPB1, hPB⟩ := pb_le (Q := Q) (W := W) (W₀ := W₀) hQ0 hτ0 hτ1 hinvτ hΘ1 hℓ1 hW1 hW₀1
    hWW hδ'0 hδ'1 hs0 hs1 hpAP hp1
  have hP₀1 : 1 ≤ P₀ := pool_ge_one hQ0 hs0 hs1 hpAP hp1 hΘ1 hW₀1 hδ'0 hδ'1
  have hPBJ : PB = (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
      + 64 * Real.log ((Q + 2 + 2) / (δ' / 2)) / pAP ^ 2 := rfl
  obtain ⟨hRB0, hRB⟩ := rb_le (Q := Q) (PB := PB) (P₀ := P₀) hQ0 hτ0 hτ1 hinvτ hΘ1 hℓ1
    (by rw [hPBJ]; exact hPB1) (by rw [hPBJ]; exact hPB) hP₀1 hδ'0 hδ'1 hs0 hpAP hp1
  have hR₀ : 0 ≤ R₀ := by
    have := log_count_nonneg hQ0 hP₀1 hΘ1 hδ'0 hδ'1 hpAP hp1
    show 0 ≤ 161000 * (Q + 2) ^ 2 * Θ ^ 2
      * (Real.log (16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * pAP)) + 2) / s ^ 4
    positivity
  have hτm : τ ≤ min (εcov / 2) ι := by
    refine le_min ?_ (min_le_right _ _)
    have ha := gateA_le_one hη0 hη₀
    have ha0 := (gateA_pos hη0 hη₀).le
    have hsq : ((1 - 2 * η₀) / (1 - η₀)) ^ 2 ≤ 1 := by
      have : (1 - 2 * η₀) / (1 - η₀) = gateA η₀ := rfl
      rw [this]; exact pow_le_one₀ ha0 ha
    have hεc0 : 0 ≤ εcov / 2 := by
      have : 0 < εcov := lt_min hε0 (by positivity)
      positivity
    exact (min_le_left _ _).trans (mul_le_of_le_one_right hεc0 hsq)
  have hlR₀ : 0 ≤ Real.logb 2 (R₀ + 1) := Real.logb_nonneg (by norm_num) (by linarith)
  have hcap := cap_ge (Q := Q) (RB := RB) (R₀ := R₀) (PB := PB) (P₀ := P₀) (E₁ := E₁) hQ0 hτ0
    hinvτ hι0 hinvι hΘ1 hℓ1 hRB0 (by rw [hRBd]; exact hRB) (by linarith) (by rw [hPBJ]; exact hPB)
    (by linarith) hR₀ hδ'0 hδ'1 hpAP hp1 hs0 hs1 hτm rfl
  have hKc1 : (1 : ℝ) ≤ Kc := kc_ge_one hQ0 (by linarith)
  have hlg : 0 ≤ Real.log (16 * (Q + 1) * Kc / δ') :=
    Real.log_nonneg (by
      rw [le_div_iff₀ hδ'0]
      nlinarith only [hδ'1, hQ0, hKc1, mul_nonneg hQ0 (sub_nonneg.2 hKc1)])
  obtain ⟨h1, h2, h3, h4⟩ := summed_tail (k := k) hA hQ0 (Nat.cast_nonneg L) haε1 hs0 hs1 hpAP
    hδ'0 hδ hΘ1 hR₀ (by linarith only [hP₀1]) hlR₀ hK₁0 (Nat.cast_nonneg n)
    (by show ((L + Fintype.card σ + 3 : ℕ) : ℝ) + 1 = _; push_cast; ring) hℓ1 hε0 hinvε hcap hlg
    rfl rfl rfl rfl rfl hRB0 hRB hB₃0 hB₃
    (by exact_mod_cast (show L ≤ 2 * ((L + 1) / 2) by omega)) rfl hL₀
  exact learner_loop_dfa_summed_simple_holds M O L η₀ c R γ δ pAP ξ hL hη hη₀ hc hR hδ hδ1 hpAP
    hξ hA2 hpAPu hceil h1 h2 h3 h4

end OrthoDFA

end
