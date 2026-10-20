import OrthoDFA.Proofs.LoopDFA

/-!
# From a checked round walked from before the window to the learner's DFA

`edge_of_occupied` and `dfa_of_checks` for probes walked from `kw`, at most the window's start: a
probe and its extension by a letter share their walk's start, wherever it is.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Edges

variable [Nonempty α] {σ : Type*} (M : _root_.DFA α σ) (dist : σ → ARU → ℝ)
  (rd : FreeMonoid α → ARU)

/-- `edge_of_occupied` for probes walked from `kw`, at most the window's start: a state visited `c`
of the time over the window's lengths below `J` has every edge out of its
true leaf learned and true, unless the disagreeing probes hold `c/((|Σ|+1)N)` of the draws. -/
theorem edge_of_occupied_at (H : HypothesisDFA α) {kw k J L₁ : ℕ} (hkw : kw ≤ k) (hkJ : k < J)
    (hJ : J ≤ L₁) (q : σ) (a : α) {c β : ℝ}
    (hbad : (windowStrings k L₁).real (disagreeSet M dist rd kw H) ≤ β)
    (hocc : c ≤ ∑ i ∈ Finset.Ico k J, (uniformStrings i).real {y | M.eval y.toList = q})
    (hβ : ((Fintype.card α : ℝ) + 1) * ((L₁ : ℝ) + 1 - k) * β < c) :
    ∃ w, H.edges (Random.leafOf M (fun q => readSide (dist q)) H.tree q, a)
      = some (Random.leafOf M (fun q => readSide (dist q)) H.tree (M.step q a), w) := by
  set Bad := disagreeSet M dist rd kw H
  set A : ℝ := (Fintype.card α : ℝ) with hA
  by_contra hno
  have hkL : k ≤ L₁ := by omega
  have hU0 : ∀ i, 0 ≤ (uniformStrings (α := α) i).real Bad := fun _ => measureReal_nonneg
  have hi : ∀ i ∈ Finset.Ico k J, (uniformStrings i).real {y | M.eval y.toList = q}
      ≤ (uniformStrings i).real Bad + A * (uniformStrings (i + 1)).real Bad := by
    intro i hi
    simp only [Finset.mem_Ico] at hi
    have hsub : {y : FreeMonoid α | M.eval y.toList = q}
        ⊆ (Bad ∪ {y | y * FreeMonoid.of a ∈ Bad}) ∪ {y | y.toList.length ≠ i} := by
      intro y hy
      by_contra hc
      simp only [Set.mem_union, Set.mem_setOf_eq, not_or, not_not] at hc hy
      apply hno
      obtain ⟨w, hw⟩ := edge_of_pair M dist rd H a (k := kw) (by omega) hc.1.1 hc.1.2
      rw [FreeMonoid.toList_mul, FreeMonoid.toList_of, DFA.eval_append_singleton, hy] at hw
      exact ⟨w, hw⟩
    have hnull : (uniformStrings (α := α) i).real {y | y.toList.length ≠ i} = 0 := by
      rw [measureReal_def, uniformStrings_length, ENNReal.toReal_zero]
    refine (measureReal_mono hsub (measure_ne_top _ _)).trans ?_
    refine (measureReal_union_le _ _).trans ?_
    rw [hnull, add_zero]
    refine (measureReal_union_le _ _).trans (add_le_add le_rfl ?_)
    exact uniformStrings_snoc_le i a Bad
  have hsum := Finset.sum_le_sum hi
  rw [Finset.sum_add_distrib, ← Finset.mul_sum] at hsum
  have h1 : ∑ i ∈ Finset.Ico k J, (uniformStrings (α := α) i).real Bad
      ≤ ∑ i ∈ Finset.Icc k L₁, (uniformStrings (α := α) i).real Bad :=
    Finset.sum_le_sum_of_subset_of_nonneg (fun i hi => by
      simp only [Finset.mem_Ico, Finset.mem_Icc] at hi ⊢; omega) fun i _ _ => hU0 i
  have h2 : ∑ i ∈ Finset.Ico k J, (uniformStrings (α := α) (i + 1)).real Bad
      ≤ ∑ i ∈ Finset.Icc k L₁, (uniformStrings (α := α) i).real Bad := by
    have e := Finset.sum_image (f := fun i => (uniformStrings (α := α) i).real Bad)
      (s := Finset.Ico k J) (g := (· + 1)) (fun x _ y _ h => by simpa using h)
    rw [← e]
    exact Finset.sum_le_sum_of_subset_of_nonneg (fun i hi => by
      simp only [Finset.mem_image, Finset.mem_Ico, Finset.mem_Icc] at hi ⊢; omega)
      fun i _ _ => hU0 i
  have hw := windowStrings_real (α := α) hkL Bad
  have hN : 0 < (L₁ : ℝ) + 1 - k := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL
    linarith
  rw [eq_div_iff hN.ne'] at hw
  have hA0 : 0 ≤ A := by positivity
  have h3 := mul_le_mul_of_nonneg_left hbad hN.le
  have h4 := mul_le_mul_of_nonneg_left h2 hA0
  nlinarith

end Edges

section Bridge

variable [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ]

/-- `dfa_of_checks` for probes walked from `kw`, at most the window's start: a last round whose
disagreeing probes over the window are at most `εB`, small enough against
the coverage `c`, has a leaf from which its hypothesis misjudges at most what re-rooting at a
covered start does. -/
theorem dfa_of_checks_at (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) {K : ℕ} (o : LoopRound Ω α K)
    (L kw : ℕ) (hkw : kw ≤ (L + 1) / 2) {c εr εu εB γ : ℝ}
    (hch : o.Checks M O (windowStrings ((L + 1) / 2) (L + Fintype.card σ + 3)) kw
      εr εu εB)
    (hεB : ((Fintype.card α : ℝ) + 1) * ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
      * εB < c)
    (hcov : (2 * εr + εu) * ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ)) < c)
    (hceil : ∃ q ∈ coveredStates M L c, 1 - γ ≤ rerootAccuracy M L (coveredStates M L c) q) :
    ∃ s ∈ o.result.hypothesis.tree.leaves, (uniformStrings L).real
      {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)} ≤ γ := by
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
  obtain ⟨⟨hcons, hmis, hund⟩, hchk, hcls⟩ := hch
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
  have hN0 : (0 : ℝ) < (L₁ : ℝ) + 1 - k := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL₁
    linarith
  have hεB0 : 0 ≤ εB := measureReal_nonneg.trans hchk
  -- the covered states
  set Sc := coveredStates M L c with hSd
  have hreach : ∀ q ∈ Sc, ∃ w : FreeMonoid α, M.eval w.toList = q := by
    intro q hq
    by_contra hno
    push Not at hno
    have hempty : {x : FreeMonoid α | ∃ j, (L + 1) / 2 ≤ j ∧ j ≤ L
        ∧ M.eval (prefixOf x j).toList = q} = ∅ := by
      ext x
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
      exact fun j _ _ => hno _
    have h := hq
    simp only [hSd, coveredStates, Set.mem_setOf_eq, hempty, measureReal_empty] at h
    have hc0 : 0 < c := by
      have : 0 ≤ ((Fintype.card α : ℝ) + 1) * ((L₁ : ℝ) + 1 - k) * εB := by positivity
      have e : ((Fintype.card α : ℝ) + 1)
          * ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ)) * εB
          = ((Fintype.card α : ℝ) + 1) * ((L₁ : ℝ) + 1 - k) * εB := rfl
      linarith
    linarith
  have hocc : ∀ q ∈ Sc, c ≤ ∑ i ∈ Finset.Ico k (L + 1),
      (uniformStrings i).real {y : FreeMonoid α | M.eval y.toList = q} := by
    intro q hq
    have := occupancy_of_covered M hq
    rwa [Finset.Ico_add_one_right_eq_Icc]
  have hE : ∀ q ∈ Sc, ∀ a, ∃ w, H.edges (Random.leafOf M side H.tree q, a)
      = some (Random.leafOf M side H.tree (M.step q a), w) := fun q hq a =>
    edge_of_occupied_at M dist o.read H hkw (by omega : k < L + 1) (by omega : L + 1 ≤ L₁) q a
      hchk (hocc q hq) hεB
  have hside : ∀ q ∈ Sc, side q = decide (q ∈ M.accept) := by
    intro q hq
    refine sideOf_eq_of_mass M D dist hd0 hd1 q
      (familyReadDist_sum O M _ _ _ q (hreach q hq)) hmis' hund' (b := c / ((L₁ : ℝ) + 1 - k))
      ?_ ?_
    · rw [windowStrings_real hkL₁, div_le_div_iff_of_pos_right hN0]
      refine (hocc q hq).trans ?_
      rw [Finset.Ico_add_one_right_eq_Icc]
      exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.Icc_subset_Icc le_rfl (by omega))
        fun _ _ _ => measureReal_nonneg
    · rw [lt_div_iff₀ hN0]
      have e : (2 * εr + εu) * ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
          = (2 * εr + εu) * ((L₁ : ℝ) + 1 - k) := rfl
      linarith
  obtain ⟨q₀, -, hacc⟩ := hceil
  exact ⟨Random.leafOf M side H.tree q₀, Random.leafOf_mem M side _ _,
    classify_start_le M side H hroot Sc hE hside L q₀ hacc⟩

end Bridge

end OrthoDFA

end
