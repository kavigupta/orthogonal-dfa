import OrthoDFA.LoopDFA
import OrthoDFA.Proofs.LoopUniform
import OrthoDFA.Proofs.DFABridge

/-!
# The learner's DFA: the proof

`loop_simple_checks` at the window's probes leaves a last round whose checked probes are at most
`εd`, with `ū` the window's bound on `undecidedMass` at the clustering's accuracy. From there the
bridge is deterministic (`dfa_of_checks`): the tree's root reads the string itself and its
midfixes are short, so a covered state's edges are true (`edge_of_occupied`), it is read on its
true side (`sideOf_eq_of_mass`), and the covered start's true leaf misjudges at most what
re-rooting does (`start_le`).
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

set_option maxHeartbeats 1000000 in -- the claim's settings meet `loop_simple_checks`'s conditions
theorem learner_loop_dfa_simple_holds : LearnerLoopDFASimple := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ c η δ pAP Q A k L₁ N εd εcov δ' ι ρ s τ J V Kp G PB RB cap
    hL hη hη₀ hc hδ hδ1 hpAP hA2 hpAPu hceil hk1 hk2
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
  have hQA : (1 : ℝ) ≤ (Q + 1) * (Q + 3) * (A + 1) :=
    one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by linarith) (by linarith))
      (by linarith)
  have hQA' : (1 : ℝ) ≤ (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2) :=
    one_le_mul_of_one_le_of_one_le hQA hAp
  have hεcov0 : 0 < εcov := by
    show 0 < δ * c / (512 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2) * N ^ 3)
    positivity
  have hεcov1 : εcov ≤ 1 := by
    show δ * c / (512 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2) * N ^ 3) ≤ 1
    rw [div_le_one (by positivity)]
    have h1 : (1 : ℝ) ≤ N ^ 3 := one_le_pow₀ hN1
    have h3 : (1 : ℝ) ≤ 512 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2) := by
      have e : 512 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2)
          = 512 * ((Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2)) := by ring
      rw [e]; linarith
    have h4 : δ * c ≤ 1 := by nlinarith
    have := one_le_mul_of_one_le_of_one_le h3 h1
    linarith
  have hδ'0 : 0 < δ' := by show 0 < δ / (4 * (Q + 1)); positivity
  have hδ'1 : δ' ≤ 1 := by
    show δ / (4 * (Q + 1)) ≤ 1
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
  let P : PrefixDistribution α := ⟨windowStrings k L₁, windowStrings_isProbability hkL₁, L₁,
    (windowStrings_length k L₁).mono fun x hx => hx.2⟩
  set ū : ℝ := δ * εd / (256 * (Q + 1)) with hūd
  have hū0 : 0 ≤ ū := by positivity
  have hpm : prefixMax P.D k ≤ (A ^ k)⁻¹ := by
    have := prefixMax_windowStrings_le (α := α) (le_refl k) hkL₁
    rwa [one_div] at this
  obtain ⟨settings, hwalk, hdis, hbud, hmaxL, hmain⟩ := loop_simple_checks M O P η₀ εcov εd δ' pAP
    k (L + 1) ū hL hη hη₀ hεcov0 hεcov1 hεd0 hεd1 hδ'0 hδ'1 hpAP
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
  have hD := hmain (uniformStrings (L₁ + Fintype.card σ + 3)) inferInstance
    (uniformStrings_length _) hpAPu hatom
  refine (measureReal_mono ?_ (measure_ne_top _ _)).trans (hD.trans (le_of_eq ?_))
  · intro p hp hgood
    apply hp
    obtain ⟨hlen, o, ho, hch⟩ := hgood
    refine ⟨hlen, o, ho, hch.1.1, ?_⟩
    have hres := loopRounds_result settings O P.D (uniformStrings (L₁ + Fintype.card σ + 3))
      (persistDraws p) _ o (List.mem_of_getLast? ho)
    refine dfa_of_checks M O hL settings.level5 o hres hmaxL L hch ?_ ?_ ?_ ⟨q₀, hq₀, ?_⟩
    · show (A + 1) * N * εd < c
      have e : (A + 1) * N * εd = c / 2 := by
        show (A + 1) * N * (c / (2 * (A + 1) * N)) = c / 2
        field_simp
      linarith
    · show 3 * εcov * N < c
      have e : 3 * εcov * N
          = 3 * (δ * c) / (512 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2) * N ^ 2) := by
        show 3 * (δ * c / (512 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2) * N ^ 3))
          * N = _
        field_simp
      rw [e, div_lt_iff₀ (by positivity)]
      have h1 : (1 : ℝ) ≤ N ^ 2 := one_le_pow₀ hN1
      have h2 : (512 : ℝ) ≤ 512 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2) := by
        have e : 512 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2)
            = 512 * ((Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2)) := by ring
        rw [e]; linarith
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
          ≤ N * (Q + 3) * A ^ (Fintype.card σ + 2) * N * εcov := by gcongr
      refine h1.trans (le_of_eq ?_)
      show N * (Q + 3) * A ^ (Fintype.card σ + 2) * N
          * (δ * c / (512 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2) * N ^ 3))
        = δ * (c / (2 * (A + 1) * N)) / (256 * (Q + 1))
      field_simp
      ring
    · exact hq₀acc
  · show (Q + 1) * (2 * (δ / (4 * (Q + 1))) + 128 * (δ * εd / (256 * (Q + 1))) / εd) = δ
    field_simp
    ring

end OrthoDFA

end
