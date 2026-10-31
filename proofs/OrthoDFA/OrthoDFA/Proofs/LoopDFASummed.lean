import OrthoDFA.Proofs.LoopDFA

/-!
# The learner's DFA under the relative rule: the bridge, by summed occupancy

A target state is wrong when an edge out of its true leaf is wrong or unlearned, or it is read on
its wrong side. A string `y` of `i` letters at a state `q` with a wrong edge at `a q` is
disagreeing, has `y·a q` disagreeing, or is misled (`edge_of_pair`). Each `y` has one state, so
summed over the wrong-edge states this is at most `P_i(Bad) + Ae P_{i+1}(Bad)` plus the misled
ones, at most half of each state's plus `β'` each (`FewMisled`); the occupancy is at most
`2N((Ae+1)β + |Q|β')` (`wrongEdge_occupancy_le`). A string at a state read on its wrong side has
`2·misread + undecided ≥ 1`, so those states hold at most `2εr + εu` of the window
(`wrongSide_mass_le`). Off the runs from the start through a wrong non-start-zone state, a string
the start re-roots well is classified as re-rooting does, and those runs hold at most `C` times the
wrong states' occupancy (`dfa_of_checks_summed`). No state need be covered: a wrong start is in its
own zone, so its runs count among them.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Summed

variable [Nonempty α] {σ : Type*} [Fintype σ] (M : _root_.DFA α σ) (dist : σ → ARU → ℝ)
  (rd : FreeMonoid α → ARU)

/-- The states of `W`, each with a wrong or unlearned edge at `a q`, occupy the window's lengths
`k` to `J - 1` at most `2N((Ae+1)β + |W|β')` in all. -/
theorem wrongEdge_occupancy_le (H : HypothesisDFA α) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] {k J L₁ : ℕ} (hkJ : k < J) (hJ : J ≤ L₁)
    (hlong : ∀ᵐ x ∂D, L₁ ≤ x.toList.length) (W : Finset σ) (a : σ → α) {β β' Ae : ℝ}
    (hAe : 0 ≤ Ae)
    (hfl : ∀ i, k ≤ i → i < J → ∀ y b, D.real {w | prefixOf w i = y}
      ≤ Ae * D.real {w | prefixOf w (i + 1) = y * FreeMonoid.of b})
    (hW : ∀ q ∈ W, ¬ ∃ w, H.edges (Random.leafOf M (fun q => readSide (dist q)) H.tree q, a q)
      = some (Random.leafOf M (fun q => readSide (dist q)) H.tree (M.step q (a q)), w))
    (hbad : (windowLaw D k L₁).real (disagreeSet rd k H) ≤ β)
    (hmis : ∀ q, (windowLaw D k L₁).real (shortAt M J q ∩ misledSet M dist rd k H.tree (a q))
      ≤ (windowLaw D k L₁).real (shortAt M J q) / 2 + β') :
    ∑ q ∈ W, ∑ i ∈ Finset.Ico k J, (prefixLaw D i).real {y | M.eval y.toList = q}
      ≤ 2 * ((Ae + 1) * ((L₁ : ℝ) + 1 - k) * β + W.card * (((L₁ : ℝ) + 1 - k) * β')) := by
  classical
  set Bad := disagreeSet rd k H
  set Mis : σ → Set (FreeMonoid α) := fun q => shortAt M J q ∩ misledSet M dist rd k H.tree (a q)
  have hkL : k ≤ L₁ := by omega
  have hU0 : ∀ i (S : Set (FreeMonoid α)), 0 ≤ (prefixLaw D i).real S :=
    fun _ _ => measureReal_nonneg
  have hnull : ∀ i ≤ L₁, (prefixLaw D i).real {y | y.toList.length ≠ i} = 0 := fun i hi => by
    rw [measureReal_def, prefixLaw_length D (hlong.mono fun x hx => hi.trans hx),
      ENNReal.toReal_zero]
  -- at each length, each string has one state
  have hi : ∀ i ∈ Finset.Ico k J, ∑ q ∈ W, (prefixLaw D i).real {y | M.eval y.toList = q}
      ≤ (prefixLaw D i).real Bad + Ae * (prefixLaw D (i + 1)).real Bad
        + ∑ q ∈ W, (prefixLaw D i).real (Mis q) := by
    intro i hi
    simp only [Finset.mem_Ico] at hi
    rw [← measureReal_biUnion_finset (fun q _ q' _ hqq' => Set.disjoint_left.2 fun y h h' =>
      hqq' (h.symm.trans h')) fun _ _ => MeasurableSpace.measurableSet_top]
    have hsub : (⋃ q ∈ W, {y : FreeMonoid α | M.eval y.toList = q})
        ⊆ ((Bad ∪ {y | y * FreeMonoid.of (a (M.eval y.toList)) ∈ Bad}) ∪ ⋃ q ∈ W, Mis q)
          ∪ {y | y.toList.length ≠ i} := by
      intro y hy
      obtain ⟨q, hq, hyq⟩ := Set.mem_iUnion₂.1 hy
      have hyq : M.eval y.toList = q := hyq
      by_contra hc
      have hl : y.toList.length = i := by_contra fun h => hc (Or.inr h)
      have hyB : y ∉ Bad := fun h => hc (Or.inl (Or.inl (Or.inl h)))
      have hyaB : y * FreeMonoid.of (a q) ∉ Bad := fun h => hc (Or.inl (Or.inl (Or.inr (by
        show y * FreeMonoid.of (a (M.eval y.toList)) ∈ Bad
        rw [hyq]; exact h))))
      have hm : y ∉ misledSet M dist rd k H.tree (a q) := fun h =>
        hc (Or.inl (Or.inr (Set.mem_biUnion hq ⟨⟨by omega, hyq⟩, h⟩)))
      obtain ⟨w, hw⟩ := edge_of_pair M dist rd H (a q) (by omega) hyB hyaB hm
      rw [FreeMonoid.toList_mul, FreeMonoid.toList_of, DFA.eval_append_singleton, hyq] at hw
      exact hW q hq ⟨w, hw⟩
    refine (measureReal_mono hsub (measure_ne_top _ _)).trans ?_
    refine (measureReal_union_le _ _).trans ?_
    rw [hnull i (by omega), add_zero]
    refine (measureReal_union_le _ _).trans (add_le_add ?_ (measureReal_biUnion_finset_le _ _))
    refine (measureReal_union_le _ _).trans (add_le_add le_rfl ?_)
    exact prefixLaw_snoc_le D (fun y => a (M.eval y.toList)) hAe
      (fun y => hfl i hi.1 hi.2 y _) Bad
  have hsum := Finset.sum_le_sum hi
  rw [Finset.sum_comm, Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum,
    Finset.sum_comm (s := Finset.Ico k J)] at hsum
  have h1 : ∑ i ∈ Finset.Ico k J, (prefixLaw D i).real Bad
      ≤ ∑ i ∈ Finset.Icc k L₁, (prefixLaw D i).real Bad :=
    Finset.sum_le_sum_of_subset_of_nonneg (fun i hi => by
      simp only [Finset.mem_Ico, Finset.mem_Icc] at hi ⊢; omega) fun i _ _ => hU0 i Bad
  have h2 : ∑ i ∈ Finset.Ico k J, (prefixLaw D (i + 1)).real Bad
      ≤ ∑ i ∈ Finset.Icc k L₁, (prefixLaw D i).real Bad := by
    have e := Finset.sum_image (f := fun i => (prefixLaw D i).real Bad)
      (s := Finset.Ico k J) (g := (· + 1)) (fun x _ y _ h => by simpa using h)
    rw [← e]
    exact Finset.sum_le_sum_of_subset_of_nonneg (fun i hi => by
      simp only [Finset.mem_image, Finset.mem_Ico, Finset.mem_Icc] at hi ⊢; omega)
      fun i _ _ => hU0 i Bad
  -- the window holds a state's strings shorter than `J` only at the lengths `k` to `J - 1`
  have hshort : ∀ q, ∀ S ⊆ shortAt M J q, ∑ i ∈ Finset.Icc k L₁, (prefixLaw D i).real S
      = ∑ i ∈ Finset.Ico k J, (prefixLaw D i).real S := by
    intro q S hS
    refine (Finset.sum_subset (fun i hi => by
      simp only [Finset.mem_Ico, Finset.mem_Icc] at hi ⊢; omega) fun i hi hi' => ?_).symm
    simp only [Finset.mem_Ico, Finset.mem_Icc, not_and, not_lt] at hi hi'
    refine le_antisymm ((measureReal_mono (fun y hy => ?_) (measure_ne_top _ _)).trans
      (hnull i hi.2).le) (hU0 i S)
    have := (hS hy).1
    show y.toList.length ≠ i
    omega
  have hN : 0 < (L₁ : ℝ) + 1 - k := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL
    linarith
  have hw : ∀ S, (windowLaw D k L₁).real S * ((L₁ : ℝ) + 1 - k)
      = ∑ i ∈ Finset.Icc k L₁, (prefixLaw D i).real S := fun S => by
    rw [windowLaw_real D hkL, div_mul_cancel₀ _ hN.ne']
  have hMq : ∀ q, ∑ i ∈ Finset.Ico k J, (prefixLaw D i).real (Mis q)
      ≤ (∑ i ∈ Finset.Ico k J, (prefixLaw D i).real {y | M.eval y.toList = q}) / 2
        + ((L₁ : ℝ) + 1 - k) * β' := by
    intro q
    have hwM := hw (Mis q)
    have hwE := hw (shortAt M J q)
    rw [hshort q _ Set.inter_subset_left] at hwM
    rw [hshort q _ subset_rfl] at hwE
    have hEx : ∑ i ∈ Finset.Ico k J, (prefixLaw D i).real (shortAt M J q)
        ≤ ∑ i ∈ Finset.Ico k J, (prefixLaw D i).real {y | M.eval y.toList = q} :=
      Finset.sum_le_sum fun i _ => measureReal_mono (fun y hy => hy.2) (measure_ne_top _ _)
    have h5 := mul_le_mul_of_nonneg_right (hmis q) hN.le
    nlinarith
  have hM := Finset.sum_le_sum fun q (_ : q ∈ W) => hMq q
  rw [Finset.sum_add_distrib, ← Finset.sum_div, Finset.sum_const, nsmul_eq_mul] at hM
  have hwB := hw Bad
  have h3 := mul_le_mul_of_nonneg_left hbad hN.le
  have h4 := mul_le_mul_of_nonneg_left h2 hAe
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
    have hlhs : ∑ q ∈ W,
        {p : FreeMonoid α | M.eval p.toList = q}.indicator (1 : FreeMonoid α → ℝ) p
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

/-- A last round as in `dfa_of_checks`, its misled probes counted among each state's strings
shorter than `J`, has a set of wrong states, off which every edge out of a state's true leaf is
learned and every state is read on its true side, that occupies the window's lengths `⌈L/2⌉` to
`J - 1` at most `N(2((Ae+1)εB + |Q|εm) + 2εr + εu)`. -/
theorem wrong_occupancy_le (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) {K : ℕ} (o : LoopRound Ω α K)
    (L : ℕ) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {J : ℕ}
    (hkJ : (L + 1) / 2 < J) (hJ : J ≤ L + Fintype.card σ + 3)
    (hlong : ∀ᵐ x ∂D, L + Fintype.card σ + 3 ≤ x.toList.length) {Ae εr εu εB εm : ℝ}
    (hAe : 0 ≤ Ae)
    (hfl : ∀ i, (L + 1) / 2 ≤ i → i < J → ∀ y a, D.real {w | prefixOf w i = y}
      ≤ Ae * D.real {w | prefixOf w (i + 1) = y * FreeMonoid.of a})
    (hch : o.Checks M O (windowLaw D ((L + 1) / 2) (L + Fintype.card σ + 3)) ((L + 1) / 2)
      εr εu εB (shortAt M J) εm) (hεm : 0 ≤ εm) :
    ∃ B : Finset σ,
      (∀ q ∉ B, ∀ a, ∃ w, o.result.hypothesis.edges (Random.leafOf M
          (fun q => readSide (familyReadDist O M o.family o.state.lo (o.state.hi + 1) q))
          o.result.hypothesis.tree q, a)
        = some (Random.leafOf M
          (fun q => readSide (familyReadDist O M o.family o.state.lo (o.state.hi + 1) q))
          o.result.hypothesis.tree (M.step q a), w)) ∧
      (∀ q ∉ B, readSide (familyReadDist O M o.family o.state.lo (o.state.hi + 1) q)
        = decide (q ∈ M.accept)) ∧
      ∑ q ∈ B, ∑ i ∈ Finset.Ico ((L + 1) / 2) J, (prefixLaw D i).real {y | M.eval y.toList = q}
        ≤ ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
          * (2 * ((Ae + 1) * εB + Fintype.card σ * εm) + (2 * εr + εu)) := by
  classical
  set k := (L + 1) / 2 with hkd
  set L₁ := L + Fintype.card σ + 3 with hL₁d
  have hkL₁ : k ≤ L₁ := by omega
  set P : Measure (FreeMonoid α) := windowLaw D k L₁ with hPd
  haveI : IsProbabilityMeasure P := windowLaw_isProbability D hkL₁
  set H := o.result.hypothesis with hHd
  set dist := familyReadDist O M o.family o.state.lo (o.state.hi + 1) with hdistd
  set side : σ → Bool := fun q => readSide (dist q)
  have hd0 := familyReadDist_nonneg O M o.family o.state.lo (o.state.hi + 1)
  have hd1 := familyReadDist_le_one O M o.family o.state.lo (o.state.hi + 1)
  obtain ⟨⟨hcons, hmis, hund⟩, hchk, hcls, hfew⟩ := hch
  -- the family's reads over the window
  have hund' : ∫ p, dist (M.eval p.toList) .undecided ∂P ≤ εu := by
    refine le_trans (le_of_eq ?_) hund
    congr 1
    funext p
    rw [readProb_eq_familyReadDist O M hL]
  have hmis' : ∫ p, (if M.eval p.toList ∈ M.accept then dist (M.eval p.toList) .reject
      else dist (M.eval p.toList) .accept) ∂P ≤ εr := by
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
  set occ : σ → ℝ := fun q => ∑ i ∈ Finset.Ico k J,
    (prefixLaw D i).real {y : FreeMonoid α | M.eval y.toList = q} with hoccd
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
  have hE := wrongEdge_occupancy_le M dist o.read H D hkJ hJ hlong WE a hAe hfl haW hchk
    (fun q => hfew q (a q))
  have hE' : ∑ q ∈ WE, occ q ≤ 2 * ((Ae + 1) * N * εB + WE.card * (N * εm)) := hE
  have hEc : (WE.card : ℝ) ≤ Fintype.card σ := by exact_mod_cast Finset.card_le_univ WE
  have hSm := wrongSide_mass_le M dist P hd0 hd1
    (fun w => familyReadDist_sum O M _ _ _ _ ⟨w, rfl⟩) WS (fun q hq => (Finset.mem_filter.1 hq).2)
    hmis' hund'
  have hoccP : ∀ q, occ q ≤ N * P.real {p | M.eval p.toList = q} := by
    intro q
    rw [hPd, windowLaw_real D hkL₁, mul_div_cancel₀ _ hN0.ne']
    exact Finset.sum_le_sum_of_subset_of_nonneg (fun i hi => by
      simp only [Finset.mem_Ico, Finset.mem_Icc] at hi ⊢; omega) fun _ _ _ => measureReal_nonneg
  have hWS : ∑ q ∈ WS, occ q ≤ N * (2 * εr + εu) := by
    calc ∑ q ∈ WS, occ q ≤ ∑ q ∈ WS, N * P.real {p | M.eval p.toList = q} :=
          Finset.sum_le_sum fun q _ => hoccP q
      _ = N * ∑ q ∈ WS, P.real {p | M.eval p.toList = q} := (Finset.mul_sum _ _ _).symm
      _ ≤ N * (2 * εr + εu) := mul_le_mul_of_nonneg_left hSm hN0.le
  refine ⟨WE ∪ WS, fun q hq b => ?_, fun q hq => ?_, ?_⟩
  · by_contra hb
    exact hq (Finset.mem_union_left _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, b, hb⟩))
  · by_contra hb
    exact hq (Finset.mem_union_right _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, hb⟩))
  have hu := Finset.sum_union_inter (s₁ := WE) (s₂ := WS) (f := occ)
  have hi0 : 0 ≤ ∑ q ∈ WE ∩ WS, occ q := Finset.sum_nonneg fun q _ => hocc0 q
  have h1 : (WE.card : ℝ) * (N * εm) ≤ Fintype.card σ * (N * εm) :=
    mul_le_mul_of_nonneg_right hEc (by positivity)
  show ∑ q ∈ WE ∪ WS, occ q ≤ N * _
  nlinarith

/-- A last round as in `dfa_of_checks`, whose wrong states together occupy the window's lengths
`⌈L/2⌉` to `L` less than `c`, has a leaf from which its hypothesis misjudges at most `γ + Cc`,
`C = R(L+1)/(L+1−⌈L/2⌉)`, where `q₀` re-roots with accuracy `1 − γ` through its non-start zone. -/
theorem dfa_of_checks_summed (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) {K : ℕ} (o : LoopRound Ω α K)
    (L : ℕ) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (hlong : ∀ᵐ x ∂D, L + Fintype.card σ + 3 ≤ x.toList.length) {Ae c R εr εu εB εm γ : ℝ}
    (hAe : 0 ≤ Ae)
    (hfl : ∀ i, (L + 1) / 2 ≤ i → i ≤ L → ∀ y a, D.real {w | prefixOf w i = y}
      ≤ Ae * D.real {w | prefixOf w (i + 1) = y * FreeMonoid.of a})
    (hch : o.Checks M O (windowLaw D ((L + 1) / 2) (L + Fintype.card σ + 3)) ((L + 1) / 2)
      εr εu εB (shortAt M (L + 1)) εm) (hεm : 0 ≤ εm) (hR : 0 ≤ R)
    (hbnd : ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
      * (2 * ((Ae + 1) * εB + Fintype.card σ * εm) + (2 * εr + εu)) < c) {q₀ : σ}
    (hacc : 1 - γ ≤ rerootAccuracyOn (prefixLaw D L) M L
      (nonStartZoneStatesOn (prefixLaw D L) M L R q₀) q₀) :
    ∃ s ∈ o.result.hypothesis.tree.leaves, (prefixLaw D L).real
      {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)}
        ≤ γ + R * (L + 1) / ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) * c := by
  classical
  set T : Measure (FreeMonoid α) := prefixLaw D L with hTd
  set H := o.result.hypothesis with hHd
  set side : σ → Bool :=
    fun q => readSide (familyReadDist O M o.family o.state.lo (o.state.hi + 1) q)
  have hroot : ∃ r a, H.tree = .node 1 r a := rootOne_of_classSet _ _ hch.2.2.1
  obtain ⟨B, hBE, hBS, hB⟩ := wrong_occupancy_le M O hL o L D (by omega) (by omega) hlong hAe
    (fun i hi hi' y a => hfl i hi (by omega) y a) hch hεm
  rw [Finset.Ico_add_one_right_eq_Icc] at hB
  set occ : σ → ℝ := fun q => ∑ i ∈ Finset.Icc ((L + 1) / 2) L,
    (prefixLaw D i).real {y : FreeMonoid α | M.eval y.toList = q} with hoccd
  have hocc0 : ∀ q, 0 ≤ occ q := fun q => Finset.sum_nonneg fun _ _ => measureReal_nonneg
  have hWrong : ∑ q ∈ B, occ q < c := hB.trans_lt hbnd
  -- every state is visited from `⌈L/2⌉` on at most its occupancy
  have hvis : ∀ q, windowVisitRateOn T M L q ≤ occ q := fun q =>
    occupancy_of_covered D M (c := windowVisitRateOn T M L q)
      (show windowVisitRateOn T M L q ≤ windowVisitRateOn T M L q from le_rfl)
  set C : ℝ := R * (L + 1) / ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) with hCd
  have hC0 : 0 ≤ C := by positivity
  set Z := nonStartZoneStatesOn T M L R q₀ with hZd
  set U : Finset σ := B.filter (· ∈ Z) with hUd
  set V : Set (FreeMonoid α) :=
    ⋃ q' ∈ U, {x | ∃ i ≤ L, M.evalFrom q₀ (prefixOf x i).toList = q'} with hVd
  set G : Set (FreeMonoid α) := {x | (∀ i ≤ L, M.evalFrom q₀ (prefixOf x i).toList ∈ Z)
    ∧ (M.evalFrom q₀ x.toList ∈ M.accept ↔ M.eval x.toList ∈ M.accept)} with hGd
  -- off the runs through a wrong non-start-zone state, a good string is classified as re-rooting
  -- does
  have hsub : {x | ¬ H.Classifies (Random.leafOf M side H.tree q₀) x (M.eval x.toList ∈ M.accept)}
      ⊆ (Gᶜ ∪ V) ∪ {x | x.toList.length ≠ L} := by
    intro x hx
    by_contra hcon
    simp only [Set.mem_union, Set.mem_compl_iff, Set.mem_setOf_eq, not_or, not_not] at hcon
    obtain ⟨⟨hG, hV⟩, hl⟩ := hcon
    apply hx
    refine classifies_of_matches M side H hroot {q | q ∉ B} hBE hBS q₀ x (fun i hi => ?_) hG.2
    have hi' : i ≤ L := hl ▸ hi
    intro hw
    exact hV (Set.mem_iUnion₂.2 ⟨_, Finset.mem_filter.2 ⟨hw, hG.1 i hi'⟩, i, hi', rfl⟩)
  have hnull : T.real {x | x.toList.length ≠ L} = 0 := by
    rw [measureReal_def, hTd, prefixLaw_length D (hlong.mono fun x hx => by omega),
      ENNReal.toReal_zero]
  have hVm : T.real V ≤ C * c := by
    refine (measureReal_biUnion_finset_le _ _).trans ?_
    calc ∑ q' ∈ U, T.real {x : FreeMonoid α | ∃ i ≤ L, M.evalFrom q₀ (prefixOf x i).toList = q'}
        ≤ ∑ q' ∈ U, C * occ q' := Finset.sum_le_sum fun q' hq' => by
          have hd : rerootVisitRateOn T M L q₀ q' ≤ C * windowVisitRateOn T M L q' :=
            (Finset.mem_filter.1 hq').2
          exact hd.trans (mul_le_mul_of_nonneg_left (hvis q') hC0)
      _ ≤ ∑ q' ∈ B, C * occ q' :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
            fun q _ _ => mul_nonneg hC0 (hocc0 q)
      _ = C * ∑ q' ∈ B, occ q' := (Finset.mul_sum _ _ _).symm
      _ ≤ C * c := mul_le_mul_of_nonneg_left hWrong.le hC0
  refine ⟨Random.leafOf M side H.tree q₀, Random.leafOf_mem M side _ _, ?_⟩
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans ((measureReal_union_le _ _).trans ?_)
  rw [hnull, add_zero]
  refine (measureReal_union_le _ _).trans ?_
  rw [measureReal_compl MeasurableSpace.measurableSet_top, probReal_univ]
  have : 1 - γ ≤ T.real G := hacc
  linarith

end Bridge

/-- The wrong states' occupancy at the summed rate `ε = c/(4(A+|Q|+1)N)`, against `3c/4`. -/
lemma summed_bnd {Q A N c ε εcov : ℝ} (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hN : 0 < N) (hc : 0 < c)
    (he : ε = c / (4 * (A + Q + 1) * N)) (hεc : εcov ≤ ε) :
    N * (2 * ((A + 1) * (ε + ε / 96) + Q * (ε + ε / 96))
      + (2 * (1025 / 1024 * εcov) + 3 / 8 * εcov)) < 3 * c / 4 := by
  have hq : ε * N * (4 * (A + Q + 1)) = c := by rw [he]; field_simp
  have hX0 : 0 < ε * N := by
    rw [he]; positivity
  have hY : εcov * N ≤ ε * N := mul_le_mul_of_nonneg_right hεc hN.le
  have h4 : 4 * (ε * N) ≤ (A + Q + 1) * (ε * N) :=
    mul_le_mul_of_nonneg_right (by linarith) hX0.le
  have e : N * (2 * ((A + 1) * (ε + ε / 96) + Q * (ε + ε / 96))
      + (2 * (1025 / 1024 * εcov) + 3 / 8 * εcov))
      = 97 / 48 * ((A + Q + 1) * (ε * N)) + 1217 / 512 * (εcov * N) := by ring
  rw [e, ← hq]
  nlinarith

lemma visitC_le {R : ℝ} (L : ℕ) (hR : 0 ≤ R) :
    R * (L + 1) / ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) ≤ 2 * R := by
  have hd : (0 : ℝ) < ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) := by
    exact_mod_cast (by omega : 0 < L + 1 - (L + 1) / 2)
  have hd2 : ((L : ℝ) + 1) ≤ 2 * ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) := by
    have : L + 1 ≤ 2 * (L + 1 - (L + 1) / 2) := by omega
    exact_mod_cast this
  rw [div_le_iff₀ hd]; nlinarith

/-- The relative rule's cost: `C ≤ 2R`, so `C m ≤ 3ξ/4` at `m = 3ξ/(8R)`. -/
lemma summed_harm_le {R ξ : ℝ} (L : ℕ) (hR : 0 < R) (hξ : 0 ≤ ξ) :
    R * (L + 1) / ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) * (3 * (ξ / (2 * R)) / 4) ≤ 3 * ξ / 4 := by
  calc R * (L + 1) / ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) * (3 * (ξ / (2 * R)) / 4)
      ≤ 2 * R * (3 * (ξ / (2 * R)) / 4) :=
        mul_le_mul_of_nonneg_right (visitC_le L hR.le) (by positivity)
    _ = 3 * ξ / 4 := by field_simp

omit [DecidableEq α] in
/-- A start in its own non-start zone is visited from `⌈L/2⌉` on by at least `1/C ≥ 1/(2R)` of
`T`'s strings. -/
theorem half_le_of_self_nonStartZone {σ : Type*} (T : Measure (FreeMonoid α))
    [IsProbabilityMeasure T] (M : _root_.DFA α σ) (L : ℕ) {R : ℝ} (hR : 0 < R) {q : σ}
    (hq : q ∈ nonStartZoneStatesOn T M L R q) : 1 / 2 ≤ R := by
  have h1 : rerootVisitRateOn T M L q q = 1 := by
    have hu : {x : FreeMonoid α | ∃ i ≤ L, M.evalFrom q (prefixOf x i).toList = q} = Set.univ :=
      Set.eq_univ_of_forall fun x => ⟨0, Nat.zero_le _, by simp [prefixOf]⟩
    rw [rerootVisitRateOn, hu, probReal_univ]
  have hw : 1 ≤ R * (L + 1) / ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) * windowVisitRateOn T M L q := by
    have h := hq
    change rerootVisitRateOn T M L q q ≤ _ at h
    rwa [h1] at h
  have hw1 : windowVisitRateOn T M L q ≤ 1 := measureReal_le_one
  have hC := visitC_le L hR.le
  have hC0 : 0 ≤ R * (L + 1) / ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) := by positivity
  nlinarith

/-- `dfa_of_checks_summed` at the rates `ξ/(2R)` sets with edges crossed at `Ae + |Q|`: a start
that re-roots with accuracy `1 − γ` through its non-start zone costs `γ + 3ξ/4`. -/
theorem summed_bridge {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*}
    [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) (L : ℕ) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] {K : ℕ} {R γ ξ Ae : ℝ} (hR : 0 < R) (hξ : 0 < ξ) (hAe : 2 ≤ Ae)
    (hlong : ∀ᵐ x ∂D, L + Fintype.card σ + 3 ≤ x.toList.length)
    (hfl : ∀ i, (L + 1) / 2 ≤ i → i ≤ L → ∀ y a, D.real {w | prefixOf w i = y}
      ≤ Ae * D.real {w | prefixOf w (i + 1) = y * FreeMonoid.of a})
    (hdom : ∃ q, 1 - γ ≤ rerootAccuracyOn (prefixLaw D L) M L
      (nonStartZoneStatesOn (prefixLaw D L) M L R q) q) :
    let k : ℕ := (L + 1) / 2
    let L₁ : ℕ := L + Fintype.card σ + 3
    let N : ℝ := L₁ + 1 - k
    let ε : ℝ := ξ / (2 * R) / (4 * (Ae + Fintype.card σ + 1) * N)
    let εcov : ℝ := min ε (ξ / 4 / (8 * N))
    ∀ o : LoopRound Ω α K,
      o.Checks M O (windowLaw D k L₁) k (1025 / 1024 * εcov) (3 / 8 * εcov) (ε + ε / 96)
        (shortAt M (L + 1)) (ε + ε / 96) →
      ∃ s ∈ o.result.hypothesis.tree.leaves, (prefixLaw D L).real
        {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)}
          ≤ γ + 3 * ξ / 4 := by
  intro k L₁ N ε εcov o hch
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Fintype.card σ := by exact_mod_cast Fintype.card_pos
  have hN0 : 0 < N := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast (show k ≤ L₁ by omega)
    show (0 : ℝ) < (L₁ : ℝ) + 1 - k
    linarith
  have hε0 : 0 < ε := by
    show 0 < ξ / (2 * R) / (4 * (Ae + Fintype.card σ + 1) * N)
    have : 0 < Ae + Fintype.card σ + 1 := by linarith
    positivity
  obtain ⟨q, hq⟩ := hdom
  obtain ⟨s, hs, h⟩ := dfa_of_checks_summed M O hL o L D hlong (by linarith) hfl hch
    (by positivity) hR.le
    (summed_bnd hQ1 hAe hN0 (by positivity) rfl (min_le_left _ _)) hq
  exact ⟨s, hs, h.trans (by linarith [summed_harm_le (ξ := ξ) L hR hξ.le])⟩

end OrthoDFA

end
