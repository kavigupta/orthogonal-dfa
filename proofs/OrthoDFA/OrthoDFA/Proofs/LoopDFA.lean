import OrthoDFA.LoopDFA
import OrthoDFA.Proofs.LoopUniform
import OrthoDFA.Proofs.DFABridge
import OrthoDFA.Proofs.LoopDFAArith
import OrthoDFA.Proofs.LoopSelect

/-!
# The learner's DFA: the proof

`loop_simple_checks` at the window's probes leaves a last round whose checked probes are at most
`εd`, with `ū` the window's bound on `undecidedMass` at the clustering's accuracy. From there the
bridge is deterministic (`dfa_of_checks`): the tree's root reads the string itself and its
midfixes are short, so a covered state's edges are true (`edge_of_occupied`), it is read on its
true side (`sideOf_eq_of_mass`), and the covered start's true leaf misjudges at most what
re-rooting does (`start_le`).

The family reads a string of length `L` wrongly, or leaves it undecided, at most `N εcov ≤ ξ/8` of
the time, so off a stray score the picked leaf misjudges at most `ξ` more (`pickLeaf_le`). Each
round that can be last strays or collides with chance at most `δ/(2(|Q|+1))`
(`loop_select_round_le`), so the selection costs half of `δ` and the loop the other half.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Root

variable (C : Level5Settings) (cut : FreeMonoid α → Option Bool)

/-- The tree's root reads the string itself. -/
def RootOne (T : DTree α) : Prop := ∃ r a, T = .node 1 r a

theorem rootOne_splitAt {T : DTree α} (h : RootOne T) (d : FreeMonoid α) (p : List Bool) :
    RootOne (T.splitAt d p) := by
  obtain ⟨r, a, rfl⟩ := h
  rcases p with _ | ⟨_ | _, p⟩ <;> exact ⟨_, _, rfl⟩

theorem rootOne_recordProbe {s : Level5State α} (h : RootOne s.tree) (x : FreeMonoid α) :
    RootOne (recordProbe C cut s x).tree := by
  rcases recordProbe_spec C cut s x with h1 | ⟨u, c, p, t, -, -, -, -, -, h1⟩ <;> rw [h1]
  · exact h
  · simpa [Level5State.setEdge, freshState, Level5State.countProbe] using h

theorem rootOne_splitState {s : Level5State α} (h : RootOne s.tree) (p : List Bool) (c : α)
    (t : List Bool) : RootOne (splitState cut s p c t).tree := by
  unfold splitState
  split
  · exact rootOne_splitAt h _ _
  · exact h

theorem rootOne_runProbes :
    ∀ (s : Level5State α) (xs : List (FreeMonoid α)), RootOne s.tree →
      RootOne (runProbes C cut s xs).1.tree
  | s, [], h => h
  | s, x :: xs, h => by
    simp only [runProbes]
    rcases ha : applyProbe C cut s x with s' | ⟨e, s'⟩
    · simp only []
      refine rootOne_runProbes s' xs ?_
      rcases (applyProbe_inl C cut ha).2 with ⟨-, rfl⟩ | ⟨p, c, t, sp, -, rfl, -⟩
      · exact rootOne_recordProbe C cut h x
      · exact rootOne_splitState cut (rootOne_recordProbe C cut h x) p c t
    · simp only []
      rcases applyProbe_inr C cut ha with ⟨-, rfl⟩ | ⟨-, p, c, t, sp, -, rfl, -⟩
      · exact rootOne_recordProbe C cut h x
      · exact rootOne_splitState cut (rootOne_recordProbe C cut h x) p c t

theorem rootOne_level5 (read : FreeMonoid α → ARU) {T : ℕ} (xs : Fin T → FreeMonoid α) :
    RootOne (indecisiveLStarLevel5 C read xs).hypothesis.tree :=
  rootOne_runProbes C _ initialState _ ⟨.leaf, .leaf, rfl⟩

end Root

section Reads

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

theorem readProb_sum (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    (w : FreeMonoid α) :
    readProb O F kl kh w .accept + readProb O F kl kh w .reject
      + readProb O F kl kh w .undecided = 1 := by
  have hm : ∀ v : ARU, MeasurableSet {ω | familyRead O.mq F kl kh w ω = v} := fun v =>
    (measurable_voteCount O F w) ((Set.to_countable {j | readOf kl kh j = v}).measurableSet)
  unfold readProb
  rw [← measureReal_union (Set.disjoint_left.2 fun ω h1 h2 => by
      simp only [Set.mem_setOf_eq] at h1 h2; rw [h1] at h2; cases h2) (hm _),
    ← measureReal_union (Set.disjoint_left.2 fun ω h1 h2 => by
      simp only [Set.mem_union, Set.mem_setOf_eq] at h1 h2
      rcases h1 with h1 | h1 <;> rw [h1] at h2 <;> cases h2) (hm _)]
  have : {ω | familyRead O.mq F kl kh w ω = .accept} ∪ {ω | familyRead O.mq F kl kh w ω = .reject}
      ∪ {ω | familyRead O.mq F kl kh w ω = .undecided} = Set.univ := by
    ext ω
    simp only [Set.mem_union, Set.mem_setOf_eq, Set.mem_univ, iff_true]
    rcases familyRead O.mq F kl kh w ω <;> simp
  rw [this, probReal_univ]

variable {σ : Type*} (O : Oracle μ (FreeMonoid α)) (M : _root_.DFA α σ)
  (F : Finset (FreeMonoid α)) (kl kh : ℕ)

theorem familyReadDist_nonneg (q : σ) (v : ARU) : 0 ≤ familyReadDist O M F kl kh q v := by
  unfold familyReadDist
  split_ifs
  · exact measureReal_nonneg
  · exact le_rfl

theorem familyReadDist_le_one (q : σ) (v : ARU) : familyReadDist O M F kl kh q v ≤ 1 := by
  unfold familyReadDist
  split_ifs
  · exact measureReal_le_one
  · norm_num

theorem familyReadDist_sum (q : σ) (hq : ∃ w : FreeMonoid α, M.eval w.toList = q) :
    familyReadDist O M F kl kh q .accept + familyReadDist O M F kl kh q .reject
      + familyReadDist O M F kl kh q .undecided = 1 := by
  unfold familyReadDist
  rw [dif_pos hq]
  exact readProb_sum O F kl kh _

end Reads

section Bridge

variable [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ]

/-- A consistent last round whose checked probes over the window are at most `εd`, small enough
against the coverage `c`, has a leaf from which its hypothesis misjudges at most what re-rooting
at a covered start does. -/
theorem dfa_of_checks (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) {K : ℕ} (C : Level5Settings)
    (o : LoopRound Ω α K)
    (hres : ∃ T, ∃ xs : Fin T → FreeMonoid α, o.result = indecisiveLStarLevel5 C o.read xs)
    (hmaxL : C.maxLeaves = Fintype.card σ + 2) (L : ℕ) {c εcov εd ū η : ℝ}
    (hch : o.Checks M O (windowStrings ((L + 1) / 2) (L + Fintype.card σ + 3)) ((L + 1) / 2)
      (L + 1) ū εcov εcov εd)
    (hεd : ((Fintype.card α : ℝ) + 1) * ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
      * εd < c)
    (hcov : 3 * εcov * ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ)) < c)
    (hū : ((L : ℝ) + 2 - ((L + 1) / 2 : ℕ)) * (Fintype.card σ + 3)
      * (Fintype.card α : ℝ) ^ (Fintype.card σ + 2)
      * ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ)) * εcov ≤ ū)
    (hceil : ∃ q ∈ coveredStates M L c, 1 - η ≤ rerootAccuracy M L (coveredStates M L c) q) :
    ∃ s ∈ o.result.hypothesis.tree.leaves, (uniformStrings L).real
      {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)} ≤ η := by
  classical
  set k := (L + 1) / 2 with hkd
  set L₁ := L + Fintype.card σ + 3 with hL₁d
  have hkL₁ : k ≤ L₁ := by omega
  set D : Measure (FreeMonoid α) := windowStrings k L₁ with hDd
  haveI : IsProbabilityMeasure D := windowStrings_isProbability hkL₁
  set H := o.result.hypothesis with hHd
  set dist := familyReadDist O M o.family o.state.lo (o.state.hi + 1) with hdistd
  have hd0 := familyReadDist_nonneg O M o.family o.state.lo (o.state.hi + 1)
  have hd1 := familyReadDist_le_one O M o.family o.state.lo (o.state.hi + 1)
  let G : ReadModel α σ := ⟨M, dist, 0, 0, 1, hd0, fun r =>
    (min_le_left _ _).trans ((hd1 r .accept).trans (le_max_left _ _))⟩
  obtain ⟨⟨hcons, hmis, hund, -⟩, hchk⟩ := hch
  -- the tree reads the string at its root, through short midfixes
  obtain ⟨T, xs, hxs⟩ := hres
  have hroot : ∃ r a, H.tree = .node 1 r a := by
    rw [hHd, hxs]; exact rootOne_level5 C o.read xs
  have hmid : ∀ m ∈ H.tree.midfixes, m ∈ shortWords (Fintype.card σ + 2) := by
    have hC : 2 ≤ C.maxLeaves := by rw [hmaxL]; omega
    have hle := level5_tree_midLe C hC o.read xs (by rw [← hxs, hcons]; simp)
    intro m hm
    rw [hHd, hxs] at hm
    exact mem_shortWords (by have := hle m hm; omega)
  -- the family's reads over the window
  have hund' : ∫ p, dist (M.eval p.toList) .undecided ∂D ≤ εcov := by
    refine le_trans (le_of_eq ?_) hund
    congr 1
    funext p
    rw [readProb_eq_familyReadDist O M hL]
  have hmis' : ∫ p, (if M.eval p.toList ∈ M.accept then dist (M.eval p.toList) .reject
      else dist (M.eval p.toList) .accept) ∂D ≤ εcov := by
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
  have hmass : undecidedMass M dist D k (Fintype.card σ + 2) (L + 1) ≤ ū := by
    refine (undecidedMass_window_le M dist hd0 (by omega : k ≤ L + 1)
      (by omega : L + 1 + (Fintype.card σ + 2) ≤ L₁)).trans (le_trans ?_ hū)
    have hk : (k : ℝ) ≤ L + 1 := by exact_mod_cast (by omega : k ≤ L + 1)
    have hcoef : 0 ≤ ((L : ℝ) + 2 - k) * (Fintype.card σ + 3)
        * (Fintype.card α : ℝ) ^ (Fintype.card σ + 2) * ((L₁ : ℝ) + 1 - k) := by
      have : (0 : ℝ) ≤ (L : ℝ) + 2 - k := by linarith
      positivity
    have e : (((L + 1 : ℕ) : ℝ) + 1 - k) * ((Fintype.card σ + 2 : ℕ) + 1)
        * (Fintype.card α : ℝ) ^ (Fintype.card σ + 2) * ((L₁ : ℝ) + 1 - k)
        = ((L : ℝ) + 2 - k) * (Fintype.card σ + 3)
          * (Fintype.card α : ℝ) ^ (Fintype.card σ + 2) * ((L₁ : ℝ) + 1 - k) := by
      push_cast; ring
    rw [e]
    exact mul_le_mul_of_nonneg_left hund' hcoef
  have hbad : D.real (searchingProbes o.read k H ∪ rarerProbes G.M G.readDist o.read k
      (Fintype.card σ + 2) ∪ undecidedProbes o.read k (Fintype.card σ + 2) (L + 1)) ≤ εd := by
    refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) hchk
    rintro x ((h | h) | h)
    · exact .inl (.inl h)
    · exact .inl (.inr h)
    · exact .inr ⟨hmass, h⟩
  have hεd0 : 0 ≤ εd := measureReal_nonneg.trans hbad
  have hc0 : 0 < c := by
    have : 0 ≤ ((Fintype.card α : ℝ) + 1) * ((L₁ : ℝ) + 1 - k) * εd := by positivity
    have e : ((Fintype.card α : ℝ) + 1) * ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
        * εd = ((Fintype.card α : ℝ) + 1) * ((L₁ : ℝ) + 1 - k) * εd := rfl
    linarith
  -- the covered states
  set S := coveredStates M L c with hSd
  have hreach : ∀ q ∈ S, ∃ w : FreeMonoid α, M.eval w.toList = q := by
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
    linarith
  have hocc : ∀ q ∈ S, c ≤ ∑ i ∈ Finset.Ico k (L + 1),
      (uniformStrings i).real {y : FreeMonoid α | M.eval y.toList = q} := by
    intro q hq
    have := occupancy_of_covered M hq
    rwa [Finset.Ico_add_one_right_eq_Icc]
  have hE : ∀ q ∈ S, ∀ a, ∃ w, H.edges (G.leafOf H.tree q, a)
      = some (G.leafOf H.tree (G.M.step q a), w) := fun q hq a =>
    edge_of_occupied G o.read H hmid (by omega : k < L + 1) (by omega : L + 1 ≤ L₁) q a hbad
      (hocc q hq) hεd
  have hside : ∀ q ∈ S, G.side q = decide (q ∈ G.M.accept) := by
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
      have e : 3 * εcov * ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
          = 3 * εcov * ((L₁ : ℝ) + 1 - k) := rfl
      linarith
  obtain ⟨q₀, -, hacc⟩ := hceil
  exact ⟨G.leafOf H.tree q₀, G.leafOf_mem_paths _ _,
    classify_start_le G H hroot S hE hside L q₀ hacc⟩

end Bridge

section Select

variable [Nonempty α]

theorem integral_uniform_le_window {k L L₁ : ℕ} (hk : k ≤ L) (hL : L ≤ L₁)
    (g : FreeMonoid α → ℝ) (hg : ∀ x, 0 ≤ g x) :
    ∫ x, g x ∂(uniformStrings L) ≤ ((L₁ : ℝ) + 1 - k) * ∫ x, g x ∂(windowStrings k L₁) := by
  have hN : (0 : ℝ) < (L₁ : ℝ) + 1 - k := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hk.trans hL
    linarith
  rw [integral_windowStrings (hk.trans hL), integral_uniformStrings, mul_div_cancel₀ _ hN.ne']
  have hu : ∀ n, 0 ≤ uavg n g := fun n =>
    div_nonneg (Finset.sum_nonneg fun v _ => hg _) (by positivity)
  exact Finset.single_le_sum (f := fun n => uavg n g) (fun n _ => hu n)
    (Finset.mem_Icc.2 ⟨hk, hL⟩)

omit [Nonempty α] in
theorem level5_leaves_le (C : Level5Settings) (hC : 2 ≤ C.maxLeaves) (read : FreeMonoid α → ARU)
    {T : ℕ} (xs : Fin T → FreeMonoid α) (h : (indecisiveLStarLevel5 C read xs).ending ≠ .failed) :
    (indecisiveLStarLevel5 C read xs).hypothesis.tree.leaves.length ≤ C.maxLeaves := by
  have hinv := runProbes_readInv C (fun z => (read z).cut) (Lx := probesLen xs) initialState
    (List.ofFn xs) ⟨by simp [initialState, freshState, DTree.leaves],
      by simpa [initialState, freshState, DTree.leaves] using hC,
      by simp [initialState, freshState, DTree.leaves, DTree.midfixes],
      by simp [initialState, freshState]⟩
    (fun x hx => by obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx; exact le_probesLen xs i) h
  exact hinv.2.1

lemma rb_nonneg {Q s pAP δ' τ cL : ℝ} (hQ : 0 ≤ Q) (hs0 : 0 < s) (hs1 : s ≤ 1 / 2)
    (hp : 0 < pAP) (hp1 : pAP ≤ 1) (hδ'0 : 0 < δ') (hδ'1 : δ' ≤ 1) (hτ0 : 0 < τ) (hτ1 : τ ≤ 1)
    (hcL : 0 ≤ cL) :
    0 ≤ 6000 * (Q + 2) ^ 2 * Real.log ((Q + 2 + 2) * (256 * (Real.log (2 / τ) + cL) / (s ^ 2 * pAP)
        + 64 * Real.log ((Q + 2 + 2) / (δ' / 2)) / pAP ^ 2)
        / (δ' / 2 * (1 / 4) * (pAP / 2) * τ)) / (s ^ 4 * τ ^ 3) := by
  have hl2 := Real.log_two_gt_d9
  have hlτ : Real.log 2 ≤ Real.log (2 / τ) :=
    Real.log_le_log (by norm_num) (by rw [le_div_iff₀ hτ0]; linarith)
  have hsp : s ^ 2 * pAP ≤ 1 / 4 := by
    have : s ^ 2 ≤ 1 / 4 := by nlinarith
    nlinarith [sq_nonneg s]
  have h1 : 1 ≤ 256 * (Real.log (2 / τ) + cL) / (s ^ 2 * pAP) := by
    rw [le_div_iff₀ (by positivity)]
    nlinarith
  have h2 : 0 ≤ 64 * Real.log ((Q + 2 + 2) / (δ' / 2)) / pAP ^ 2 := by
    refine div_nonneg (mul_nonneg (by norm_num) (Real.log_nonneg ?_)) (by positivity)
    rw [le_div_iff₀ (by positivity)]
    linarith
  refine div_nonneg (mul_nonneg (by positivity) (Real.log_nonneg ?_)) (by positivity)
  have hD1 : δ' / 2 * (1 / 4) * (pAP / 2) * τ ≤ 1 := by
    have : δ' / 2 * (1 / 4) * (pAP / 2) ≤ 1 := by nlinarith
    nlinarith
  rw [le_div_iff₀ (by positivity), one_mul]
  nlinarith

lemma dfa_select_arith {Q A δ ξ RB KpG T NPf : ℝ} {n k L : ℕ} (hQ : 0 ≤ Q) (hA : 2 ≤ A)
    (hδ : 0 < δ) (hξ : 0 < ξ) (hkL : k ≤ L)
    (hn : 32 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / ξ ^ 2 ≤ n) (hT : T ≤ KpG)
    (hNP : NPf ≤ RB)
    (h3 : (Q + 1) * n * (6 * (Q + 2) * RB + KpG + n) * (A ^ k)⁻¹ ≤ δ / 4) :
    (Q + 2) * (4 * Real.exp (-(n * (ξ / 4) ^ 2 / 2)))
      + (n * (6 * (Q + 2) * NPf + T) / A ^ k + n ^ 2 / A ^ L) ≤ δ / (2 * (Q + 1)) := by
  set X : ℝ := 16 * (Q + 1) * (Q + 2) / δ with hXd
  have hX0 : 0 < X := by positivity
  have hnX : Real.log X ≤ n * (ξ / 4) ^ 2 / 2 := by
    rw [div_le_iff₀ (by positivity)] at hn
    have e : (n : ℝ) * (ξ / 4) ^ 2 / 2 = (n : ℝ) * ξ ^ 2 / 32 := by ring
    rw [e]
    linarith
  have hexp : Real.exp (-(n * (ξ / 4) ^ 2 / 2)) ≤ δ / (16 * (Q + 1) * (Q + 2)) := by
    calc Real.exp (-(n * (ξ / 4) ^ 2 / 2)) ≤ Real.exp (-Real.log X) :=
          Real.exp_le_exp.2 (by linarith)
      _ = δ / (16 * (Q + 1) * (Q + 2)) := by rw [Real.exp_neg, Real.exp_log hX0, hXd, inv_div]
  have h1 : (Q + 2) * (4 * Real.exp (-(n * (ξ / 4) ^ 2 / 2))) ≤ δ / (4 * (Q + 1)) := by
    calc (Q + 2) * (4 * Real.exp (-(n * (ξ / 4) ^ 2 / 2)))
        ≤ (Q + 2) * (4 * (δ / (16 * (Q + 1) * (Q + 2)))) := by gcongr
      _ = δ / (4 * (Q + 1)) := by field_simp; ring
  have hAk : (0 : ℝ) < A ^ k := by positivity
  have hAkL : A ^ k ≤ A ^ L := pow_le_pow_right₀ (by linarith) hkL
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg _
  have hb1 : (n : ℝ) ^ 2 / A ^ L ≤ (n : ℝ) ^ 2 / A ^ k :=
    div_le_div_of_nonneg_left (by positivity) hAk hAkL
  have hb2 : (n : ℝ) * (6 * (Q + 2) * NPf + T) ≤ n * (6 * (Q + 2) * RB + KpG) := by
    gcongr
  have hQ1 : (0 : ℝ) < Q + 1 := by linarith
  have h3' : (n : ℝ) * (6 * (Q + 2) * RB + KpG + n) * (A ^ k)⁻¹ ≤ δ / (4 * (Q + 1)) := by
    rw [le_div_iff₀ (by positivity)]
    nlinarith
  have h2 : (n : ℝ) * (6 * (Q + 2) * NPf + T) / A ^ k + (n : ℝ) ^ 2 / A ^ L
      ≤ δ / (4 * (Q + 1)) := by
    calc (n : ℝ) * (6 * (Q + 2) * NPf + T) / A ^ k + (n : ℝ) ^ 2 / A ^ L
        ≤ n * (6 * (Q + 2) * RB + KpG) / A ^ k + (n : ℝ) ^ 2 / A ^ k :=
          add_le_add (div_le_div_of_nonneg_right hb2 hAk.le) hb1
      _ = (n : ℝ) * (6 * (Q + 2) * RB + KpG + n) * (A ^ k)⁻¹ := by field_simp
      _ ≤ δ / (4 * (Q + 1)) := h3'
  calc (Q + 2) * (4 * Real.exp (-(n * (ξ / 4) ^ 2 / 2)))
        + ((n : ℝ) * (6 * (Q + 2) * NPf + T) / A ^ k + (n : ℝ) ^ 2 / A ^ L)
      ≤ δ / (4 * (Q + 1)) + δ / (4 * (Q + 1)) := add_le_add h1 h2
    _ = δ / (2 * (Q + 1)) := by field_simp; ring

end Select

set_option maxHeartbeats 4000000 in -- the claim's settings meet `loop_simple_checks`'s conditions
theorem learner_loop_dfa_simple_holds : LearnerLoopDFASimple := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ c η δ pAP ξ Q A k L₁ N εd εcov δ' ι ρ s τ J V Kp G PB RB cap
    n hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu hceil hk1 hk2 hk3
  classical
  have hA2' : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have hQ0 : (0 : ℝ) ≤ Q := Nat.cast_nonneg _
  have hkL₁ : k ≤ L₁ := by show (L + 1) / 2 ≤ L + Fintype.card σ + 3; omega
  have hNe : N = ((L₁ : ℕ) : ℝ) + 1 - k := rfl
  have hN1 : (1 : ℝ) ≤ N := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL₁
    rw [hNe]; linarith
  obtain ⟨q₀, hq₀, hq₀acc⟩ := hceil
  have hc1 : c ≤ 1 := hq₀.trans measureReal_le_one
  have hεd0 : 0 < εd := by show 0 < c / (2 * (A + 1) * N); positivity
  have hεd1 : εd ≤ 1 := by
    show c / (2 * (A + 1) * N) ≤ 1
    rw [div_le_one (by positivity)]
    nlinarith
  have hAp : (1 : ℝ) ≤ A ^ (Fintype.card σ + 2) := one_le_pow₀ (by linarith)
  set εc : ℝ := δ * c / (1024 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2) * N ^ 3)
    with hεcd
  have hεcov_c : εcov ≤ εc := min_le_left _ _
  have hεcov_ξ : εcov ≤ ξ / (8 * N) := min_le_right _ _
  have hεcov0 : 0 < εcov := lt_min (by positivity) (by positivity)
  have hεc1 : εc ≤ 1 := by
    rw [hεcd, div_le_one (by positivity)]
    have h1 : (1 : ℝ) ≤ N ^ 3 := one_le_pow₀ hN1
    have h3 : (1 : ℝ) ≤ 1024 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2) := by
      have h2 : (1 : ℝ) ≤ (Q + 1) * (Q + 3) * (A + 1) :=
        one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by linarith)
          (by linarith)) (by linarith)
      have e : 1024 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2)
          = 1024 * ((Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2)) := by ring
      rw [e]; nlinarith
    have h4 : δ * c ≤ 1 := by nlinarith
    have := one_le_mul_of_one_le_of_one_le h3 h1
    linarith
  have hεcov1 : εcov ≤ 1 := hεcov_c.trans hεc1
  have hδ'0 : 0 < δ' := by show 0 < δ / (8 * (Q + 1)); positivity
  have hδ'1 : δ' ≤ 1 := by
    show δ / (8 * (Q + 1)) ≤ 1
    rw [div_le_one (by positivity)]
    linarith
  have hlog : 0 ≤ Real.log ((Q + 3) * A / δ') := by
    refine Real.log_nonneg ?_
    rw [le_div_iff₀ hδ'0, one_mul]
    nlinarith
  -- the walk starts after at least one letter
  have hk0 : 1 ≤ k := by
    by_contra h
    have hk : k = 0 := by omega
    rw [hk, pow_zero, inv_one, mul_one] at hk2
    have hlog1 : 1 ≤ Real.log ((Q + 3) * A / δ') := by
      rw [Real.le_log_iff_exp_le (by positivity)]
      have he : Real.exp 1 ≤ 3 := by
        have := Real.exp_one_lt_d9; linarith
      refine he.trans ?_
      rw [le_div_iff₀ hδ'0]
      nlinarith
    have hL₁ : (1 : ℝ) ≤ (L₁ : ℝ) + 1 := by have : (0 : ℝ) ≤ L₁ := Nat.cast_nonneg _; linarith
    have hQ2 : (2 : ℝ) ≤ Q + 2 := by linarith
    have hbig : (16384 : ℝ) * 1 * 2 * 2 * 1
        ≤ 16384 * ((L₁ : ℝ) + 1) * (Q + 2) * A * Real.log ((Q + 3) * A / δ') := by gcongr
    linarith
  have hkL : k ≤ L := by
    have : (L + 1) / 2 ≤ L ∨ L = 0 := by omega
    rcases this with h | h
    · exact h
    · exfalso; have : k = 0 := by show (L + 1) / 2 = 0; omega
      omega
  let P : PrefixDistribution α := ⟨windowStrings k L₁, windowStrings_isProbability hkL₁, L₁,
    (windowStrings_length k L₁).mono fun x hx => hx.2⟩
  set ū : ℝ := δ * εd / (512 * (Q + 1)) with hūd
  have hū0 : 0 ≤ ū := by positivity
  have hpm : prefixMax P.D k ≤ (A ^ k)⁻¹ := by
    have := prefixMax_windowStrings_le (α := α) (le_refl k) hkL₁
    rwa [one_div] at this
  obtain ⟨settings, hwalk, hdis, hbud, hmaxL, hnp, hmain⟩ := loop_simple_checks M O P η₀ εcov εd δ'
    pAP k (L + 1) ū hL hη hη₀ hεcov0 hεcov1 hεd0 hεd1 hδ'0 hδ'1 hpAP
    (by show 1 ≤ L + Fintype.card σ + 3; omega) hk0 (hpm.trans hk1)
    (le_trans (mul_le_mul_of_nonneg_left hpm (by positivity)) hk2) hū0
  refine ⟨settings, hwalk, hdis, hbud, ?_⟩
  have hρ : ρ ≤ 4 := by show 3 * εd / 16 ≤ 4; linarith
  have hcap0 : 0 < cap := by
    have h1 : 0 < (A ^ k)⁻¹ := by positivity
    have h2 : 0 < ρ := by show 0 < 3 * εd / 16; positivity
    by_contra hc0
    push Not at hc0
    have : cap * ρ / 4 ≤ 0 := by
      have := mul_nonpos_of_nonpos_of_nonneg hc0 h2.le
      linarith
    linarith
  have hatom : ∀ v : FreeMonoid α,
      (uniformStrings (L₁ + Fintype.card σ + 3)).real {v} ≤ cap := fun v => by
    refine (uniformStrings_singleton_le _ v).trans ?_
    have h1 : 1 / A ^ (L₁ + Fintype.card σ + 3) ≤ (A ^ k)⁻¹ := by
      rw [one_div]
      exact inv_anti₀ (by positivity) (pow_le_pow_right₀ (by linarith) (by omega))
    refine h1.trans (hk1.trans ?_)
    have : cap * ρ ≤ cap * 4 := mul_le_mul_of_nonneg_left hρ hcap0.le
    linarith
  set Dsf : Measure (FreeMonoid α) := uniformStrings (L₁ + Fintype.card σ + 3) with hDsfd
  have hD := hmain Dsf inferInstance (uniformStrings_length _) hpAPu hatom
  set T := settings.level5.probeBudget
  set μL := loopMeasure μ α (Fintype.card σ + 1) T
  set β := Measure.pi fun _ : Fin n => uniformStrings (α := α) L
  set Nn := P.L + settings.level5.maxLeaves
  set Ls := P.L + Fintype.card σ + 3
  set t : ℝ := ξ / 4 with htd
  have hC : 2 ≤ settings.level5.maxLeaves := by rw [hmaxL]; omega
  set Bloop : Set (Ω × (Fin (Fintype.card σ + 1) → RoundDraws α (Fintype.card σ + 1) T)) :=
    {p | let rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p)
      ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
          o.Checks M O P.D k (L + 1) ū εcov εcov εd)
        ∧ RoundsFresh settings O P.D Dsf Nn Ls p)} with hBloopd
  set Sel : ℕ → Set ((Ω × (Fin (Fintype.card σ + 1) → RoundDraws α (Fintype.card σ + 1) T))
      × (Fin n → FreeMonoid α)) := fun r =>
    if hr : r < Fintype.card σ + 1 then selFail settings O P.D Dsf Nn Ls L n t r hr else ∅
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
              (M.eval x.toList ∈ M.accept)} ≤ η + ξ)}
      ⊆ (Prod.fst ⁻¹' Bloop) ∪ ⋃ r ∈ Finset.range (Fintype.card σ + 1), Sel r := by
    intro q hq
    by_cases hlp : q.1 ∈ Bloop
    · exact Or.inl hlp
    right
    have hlp' := not_not.1 hlp
    obtain ⟨⟨hlen, o, ho, hch⟩, hfr⟩ := hlp'
    have hcons : o.result.ending = .consistent := hch.1.1
    obtain ⟨r, hr, -, tt, hst, hG, hOk, hB, hF, hread, hhyp⟩ :=
      stageSel_last settings O P Dsf hC Ls q.1 hfr o ho hcons
    refine Set.mem_iUnion₂.2 ⟨r, Finset.mem_range.2 hr, ?_⟩
    simp only [hSeld, dif_pos hr]
    by_contra hnot
    apply hq
    refine ⟨hlen, o, ho, hcons, ?_⟩
    have hnot' : (q.1.1, q.2) ∉ SelBad settings O L n t (some (tt, resCode o.result))
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
    obtain ⟨T', xs, hres⟩ := loopRounds_result settings O P.D Dsf (persistDraws q.1) _ o
      (List.mem_of_getLast? ho)
    have hsf : SuffixFree o.family := by
      have hlenF : ∀ v ∈ o.family, v.toList.length = Ls := fun v hv =>
        hOk.1.1 ⟨r, hr⟩ le_rfl v (hOk.2.1 (hF ▸ hv))
      intro v hv v' hv' hsuf
      apply FreeMonoid.toList.injective
      exact hsuf.eq_of_length ((hlenF v hv).trans (hlenF v' hv').symm)
    have hleaves : o.result.hypothesis.tree.leaves.length ≤ settings.level5.maxLeaves := by
      rw [hres]
      exact level5_leaves_le settings.level5 hC o.read xs (by rw [← hres, hcons]; simp)
    have hgood : (q.1.1, q.2) ∉ scoreBad O o.result.hypothesis o.family o.state.lo
        (o.state.hi + 1) L n t := by
      intro h
      apply hnb
      show (q.1.1, q.2) ∈ (if SuffixFree tt.2.2.1
          ∧ (resCode o.result).hyp.tree.leaves.length ≤ settings.level5.maxLeaves
        then scoreBad O (resCode o.result).hyp tt.2.2.1 tt.2.1.lo (tt.2.1.hi + 1) L n t else ∅)
      rw [hF, hB, hhyp, if_pos ⟨hsf, hleaves⟩]
      exact h
    -- the best leaf, and the family's reads on the strings of length `L`
    have hs := dfa_of_checks M O hL settings.level5 o ⟨T', xs, hres⟩ hmaxL L hch ?_ ?_ ?_
      ⟨q₀, hq₀, hq₀acc⟩
    · have hn : 0 < n := by
        refine Nat.ceil_pos.2 (div_pos (mul_pos (by norm_num) (Real.log_pos ?_)) (by positivity))
        rw [one_lt_div hδ]
        nlinarith
      have hNε : N * εcov ≤ ξ / 8 := by
        have h := mul_le_mul_of_nonneg_left hεcov_ξ (by linarith : (0 : ℝ) ≤ N)
        rw [show N * (ξ / (8 * N)) = ξ / 8 by field_simp] at h
        exact h
      have hu : ∫ x, readProb O o.family o.state.lo (o.state.hi + 1) x .undecided
          ∂(uniformStrings L) ≤ N * εcov := by
        refine (integral_uniform_le_window (L₁ := L₁) hkL (by omega) _
          fun x => measureReal_nonneg).trans ?_
        exact mul_le_mul_of_nonneg_left hch.1.2.2.1 (zero_le_one.trans hN1)
      have hm : ∫ x, misreadProb O o.family o.state.lo (o.state.hi + 1) x ∂(uniformStrings L)
          ≤ N * εcov := by
        refine (integral_uniform_le_window (L₁ := L₁) hkL (by omega) _
          fun x => measureReal_nonneg).trans ?_
        exact mul_le_mul_of_nonneg_left hch.1.2.1 (zero_le_one.trans hN1)
      have h := pickLeaf_le M O o.result.hypothesis o.family o.state.lo (o.state.hi + 1) hL L n hn
        t (N * εcov) (N * εcov) η q.1.1 q.2 hinj hgood hu hm hs
      rw [hread]
      have : η + 2 * t + N * εcov + 2 * (N * εcov) ≤ η + ξ := by
        rw [htd]; linarith
      exact h.trans this
    · show (A + 1) * N * εd < c
      have e : (A + 1) * N * εd = c / 2 := by
        show (A + 1) * N * (c / (2 * (A + 1) * N)) = c / 2
        field_simp
      linarith
    · show 3 * εcov * N < c
      have e : 3 * εc * N
          = 3 * (δ * c) / (1024 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2)
            * N ^ 2) := by
        rw [hεcd]
        field_simp
      have h0 : 3 * εcov * N ≤ 3 * εc * N := by gcongr
      refine lt_of_le_of_lt h0 ?_
      rw [e, div_lt_iff₀ (by positivity)]
      have h1 : (1 : ℝ) ≤ N ^ 2 := one_le_pow₀ hN1
      have h2 : (1024 : ℝ) ≤ 1024 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2) := by
        have e : 1024 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2)
            = 1024 * ((Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2)) := by ring
        rw [e]
        have : (1 : ℝ) ≤ (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2) :=
          one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
            (one_le_mul_of_one_le_of_one_le (by linarith) (by linarith)) (by linarith)) hAp
        linarith
      have h3 := mul_le_mul h2 h1 (by norm_num) (by positivity)
      have h4 : δ * c ≤ c := by nlinarith
      nlinarith
    · show ((L : ℝ) + 2 - k) * (Q + 3) * A ^ (Fintype.card σ + 2) * N * εcov ≤ ū
      have hLk : (L : ℝ) + 2 - k ≤ N := by
        rw [hNe]
        push_cast [L₁]
        linarith
      have hLk0 : (0 : ℝ) ≤ (L : ℝ) + 2 - k := by
        have : (k : ℝ) ≤ L + 1 := by exact_mod_cast (show (L + 1) / 2 ≤ L + 1 by omega)
        linarith
      have h1 : ((L : ℝ) + 2 - k) * (Q + 3) * A ^ (Fintype.card σ + 2) * N * εcov
          ≤ N * (Q + 3) * A ^ (Fintype.card σ + 2) * N * εc := by gcongr
      refine h1.trans (le_of_eq ?_)
      rw [hεcd, hūd]
      show N * (Q + 3) * A ^ (Fintype.card σ + 2) * N
          * (δ * c / (1024 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2) * N ^ 3))
        = δ * (c / (2 * (A + 1) * N)) / (512 * (Q + 1))
      field_simp
      ring
  -- the loop's failure, and each round's selection
  have hBl : (μL.prod β).real (Prod.fst ⁻¹' Bloop) ≤ δ / 2 := by
    have e : (μL.prod β).real (Prod.fst ⁻¹' Bloop) = μL.real Bloop := by
      rw [← Set.prod_univ, measureReal_def, measureReal_def, Measure.prod_prod, measure_univ,
        mul_one]
    rw [e]
    refine hD.trans (le_of_eq ?_)
    rw [hūd]
    show (Q + 1) * (2 * (δ / (8 * (Q + 1))) + 128 * (δ * εd / (512 * (Q + 1))) / εd) = δ / 2
    field_simp
    ring
  have hsel : ∀ r ∈ Finset.range (Fintype.card σ + 1), (μL.prod β).real (Sel r)
      ≤ δ / (2 * (Q + 1)) := by
    intro r hr
    have hr' := Finset.mem_range.1 hr
    simp only [hSeld, dif_pos hr']
    have hs0 : 0 < s := by show 0 < 1 / 2 - η₀; linarith
    have hs1 : s ≤ 1 / 2 := by
      show 1 / 2 - η₀ ≤ 1 / 2
      linarith [le_trans (eta_nonneg O) hη]
    have hp1 : pAP ≤ 1 := hpAPu.trans measureReal_le_one
    have : Nonempty σ := ⟨M.start⟩
    have hQ1 : (1 : ℝ) ≤ Q := by
      show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
    have hι0 : 0 < ι := lt_min (div_pos hεcov0 (by norm_num))
      (one_div_pos.2 (mul_pos (by positivity) (by linarith)))
    have hτ0 : 0 < τ := lt_min (div_pos hεcov0 (by norm_num)) (lt_min hs0 hι0)
    have hτ1 : τ ≤ 1 := (min_le_left _ _).trans (by linarith)
    have hV0 : 0 ≤ V := by positivity
    have hG0 : 0 ≤ G := by
      refine Real.log_nonneg ?_
      rw [le_div_iff₀ hδ'0, one_mul]
      have hV1 : 1 ≤ V := by
        show 1 ≤ (Q + 2) * ((Q + 2) * A + 1)
        have : 0 ≤ (Q + 2) * ((Q + 2) * A) := mul_nonneg (by linarith) (by nlinarith)
        nlinarith
      have : (4 : ℝ) ≤ 2 * (Q + 2) * A + 4 := by nlinarith
      nlinarith
    have hKp0 : 0 ≤ Kp := by show 0 ≤ V * (224 / εd + 1); positivity
    have hRB0 : 0 ≤ RB := rb_nonneg hQ0 hs0 hs1 hpAP hp1 hδ'0 hδ'1 hτ0 hτ1
      (by positivity : 0 ≤ 8 * Kp * G * ((L₁ : ℝ) + 1) * (Q + 3) * A ^ (Fintype.card σ + 2) / δ')
    refine (loop_select_round_le settings O P Dsf Ls L n ⌊RB⌋₊ t (by positivity) hC (hwalk ▸ hkL)
      hnp r hr').trans ?_
    rw [hmaxL, hwalk]
    have e1 : ((Fintype.card σ + 2 : ℕ) : ℝ) = Q + 2 := by push_cast; rfl
    have e2 : ((Fintype.card σ + 1 : ℕ) : ℝ) + 1 = Q + 2 := by push_cast; ring
    rw [e1, e2, htd]
    exact dfa_select_arith hQ0 hA2' hδ hξ hkL (Nat.le_ceil _) hbud (Nat.floor_le hRB0) hk3
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans ?_
  refine (measureReal_union_le _ _).trans ?_
  refine (add_le_add hBl ((measureReal_biUnion_finset_le _ _).trans
    (Finset.sum_le_sum hsel))).trans (le_of_eq ?_)
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  show δ / 2 + ((Fintype.card σ + 1 : ℕ) : ℝ) * (δ / (2 * (Q + 1))) = δ
  push_cast
  field_simp
  ring

set_option maxHeartbeats 1000000 in -- unfolding the simple claim's settings
theorem learner_loop_dfa_holds : LearnerLoopDFA := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ c η δ pAP ξ Q A k L₁ N εd δ' s V Kp G n Ce Θ K₁ B₁ P₀ R₀
    E₁ E E₂ L₀ hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu hceil hL₀
  classical
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
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
  have hδ'0 : 0 < δ' := by show 0 < δ / (8 * (Q + 1)); positivity
  have hδ'1 : δ' ≤ 1 := by
    show δ / (8 * (Q + 1)) ≤ 1
    rw [div_le_one (by positivity)]
    linarith
  have hεd : 0 < εd := by show 0 < c / (2 * (A + 1) * N); positivity
  have hV0 : 0 ≤ V := by positivity
  have hK₁0 : 0 ≤ K₁ := by show 0 ≤ V * (448 * (A + 1) / c + 1); positivity
  -- `E` and `E₂` are positive, so `L₀` holds both
  have hpow : ((L₁ : ℝ) + 1) ^ 17 * (E + E₂) ≤ A ^ k :=
    dfa_pow (L := (L : ℝ)) (Q := Q) hA
      (by show ((L + Fintype.card σ + 3 : ℕ) : ℝ) + 1 = _; push_cast; ring)
      (by linarith) (by exact_mod_cast (show L ≤ 2 * ((L + 1) / 2) by omega)) hL₀
  set εcov : ℝ := min (δ * c / (1024 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2)
      * N ^ 3)) (ξ / (8 * N)) with hεcovd
  have hCe0 : 0 < Ce := by
    show 0 < 1024 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2) / (δ * c) + 8 / ξ
    positivity
  have hε0 : 0 < εcov := lt_min (by positivity) (by positivity)
  have hεl : 1 / (Ce * ((L₁ : ℝ) + 1) ^ 3) ≤ εcov := by
    have hN3 : N ^ 3 ≤ ((L₁ : ℝ) + 1) ^ 3 := pow_le_pow_left₀ (by linarith) hNℓ 3
    have hℓ3 : (L₁ : ℝ) + 1 ≤ ((L₁ : ℝ) + 1) ^ 3 := le_self_pow₀ hℓ1 (by norm_num)
    set K : ℝ := 1024 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2) with hKd
    have hK0 : 0 < K := by positivity
    have hCe : Ce = K / (δ * c) + 8 / ξ := rfl
    refine le_min ?_ ?_
    · rw [div_le_div_iff₀ (by positivity) (by positivity), one_mul]
      have h1 : K / (δ * c) * (δ * c) = K := by field_simp
      have h2 : K * N ^ 3 ≤ (K / (δ * c) + 8 / ξ) * (δ * c) * ((L₁ : ℝ) + 1) ^ 3 := by
        have : K / (δ * c) * (δ * c) ≤ (K / (δ * c) + 8 / ξ) * (δ * c) :=
          mul_le_mul_of_nonneg_right (le_add_of_nonneg_right (by positivity)) (by positivity)
        rw [h1] at this
        calc K * N ^ 3 ≤ K * ((L₁ : ℝ) + 1) ^ 3 := mul_le_mul_of_nonneg_left hN3 hK0.le
          _ ≤ _ := mul_le_mul_of_nonneg_right this (by positivity)
      rw [hCe]
      nlinarith
    · rw [div_le_div_iff₀ (by positivity) (by positivity), one_mul]
      have h1 : 8 / ξ * ξ = 8 := by field_simp
      have h2 : 8 * N ≤ (K / (δ * c) + 8 / ξ) * ξ * ((L₁ : ℝ) + 1) ^ 3 := by
        have : 8 / ξ * ξ ≤ (K / (δ * c) + 8 / ξ) * ξ :=
          mul_le_mul_of_nonneg_right (le_add_of_nonneg_left (by positivity)) hξ.le
        rw [h1] at this
        calc 8 * N ≤ 8 * ((L₁ : ℝ) + 1) ^ 3 := by nlinarith
          _ ≤ _ := mul_le_mul_of_nonneg_right this (by positivity)
      rw [hCe]
      nlinarith
  have hb := dfa_cap_bound (ℓ := (L₁ : ℝ) + 1) (Nn := N) (εcov := εcov)
    (ι := min (εcov / 8) (1 / (3072 * ((L₁ : ℝ) + 1) * (Q + 1) * Q)))
    (ρ := 3 * εd / 16)
    (τ := min (εcov / 2) (min s (min (εcov / 8) (1 / (3072 * ((L₁ : ℝ) + 1) * (Q + 1) * Q)))))
    (m := Fintype.card σ + 2) (n := k) hQ1 hA hℓ1 hN1 hNℓ hη0 hη₀ hc hpAP hp1 rfl hε0 hCe0 hεl
    hδ'0 hδ'1 rfl rfl rfl rfl rfl rfl rfl rfl rfl rfl rfl rfl rfl rfl
  dsimp only at hb
  obtain ⟨hRB, hKp, hR₀, hG1, hE0, hcap⟩ := hb
  have hℓ17 : 0 ≤ ((L₁ : ℝ) + 1) ^ 17 := by positivity
  have hE₂0 : 0 ≤ E₂ := by
    show 0 ≤ 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ * G + n) / δ
    have : 0 ≤ K₁ * G := mul_nonneg hK₁0 (zero_le_one.trans hG1)
    positivity
  obtain ⟨hb1, hb2⟩ := hcap
    (le_trans (mul_le_mul_of_nonneg_left (le_add_of_nonneg_right hE₂0) hℓ17) hpow)
  exact learner_loop_dfa_simple_holds M O L η₀ c η δ pAP ξ hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu
    hceil hb1 hb2 (dfa_collide_arith (by positivity) (by positivity) hℓ1 hδ hR₀ hK₁0
      (zero_le_one.trans hG1) hRB hKp rfl
      (le_trans (mul_le_mul_of_nonneg_left (le_add_of_nonneg_left hE0) hℓ17) hpow))

end OrthoDFA

end
