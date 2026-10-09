import OrthoDFA.Proofs.Quality

/-!
# The trichotomy's batch claims

The gate's tests settle on the wrong side of `acc` for some start with chance at most `a` per
look; a test left unsettled at the batch's end leaves its start within `δc` of `acc` but for a
binomial tail. A refusal then has every start disagreeing on more than `1 − acc`, so a covering
start leaves the classes and live edges more than `need`, and below `τ₀` every class fires on its
first hit: halving needs the refusal sample to miss that mass.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Combinatorics

omit [Fintype α] [DecidableEq α] in
theorem foldl_best (score : List Bool → ℕ) :
    ∀ (qs : List (List Bool)) (b₀ : List Bool),
      (qs.foldl (fun b q => if score b < score q then q else b) b₀ = b₀
        ∨ qs.foldl (fun b q => if score b < score q then q else b) b₀ ∈ qs)
      ∧ score b₀ ≤ score (qs.foldl (fun b q => if score b < score q then q else b) b₀)
      ∧ ∀ q ∈ qs, score q ≤ score (qs.foldl (fun b q => if score b < score q then q else b) b₀)
  | [], b₀ => by simp
  | q :: qs, b₀ => by
    simp only [List.foldl_cons]
    set b₁ := if score b₀ < score q then q else b₀
    obtain ⟨h1, h2, h3⟩ := foldl_best score qs b₁
    have hb₁ : (b₁ = q ∨ b₁ = b₀) ∧ score b₀ ≤ score b₁ ∧ score q ≤ score b₁ := by
      simp only [b₁]; split_ifs with h <;> simp <;> omega
    refine ⟨?_, hb₁.2.1.trans h2, ?_⟩
    · rcases h1 with h1 | h1
      · rw [h1]; rcases hb₁.1 with h | h
        · exact .inr (by simp [h])
        · exact .inl h
      · exact .inr (List.mem_cons_of_mem _ h1)
    · intro q' hq'
      rcases List.mem_cons.1 hq' with rfl | hq'
      · exact hb₁.2.2.trans h2
      · exact h3 q' hq'

omit [Fintype α] [DecidableEq α] in
theorem bestOf_max (qs : List (List Bool)) (score : List Bool → ℕ) {q : List Bool}
    (hq : q ∈ qs) : score q ≤ score (bestOf qs score) :=
  (foldl_best score qs _).2.2 q hq

omit [Fintype α] [DecidableEq α] in
theorem bestOf_mem {qs : List (List Bool)} (hqs : qs ≠ []) (score : List Bool → ℕ) :
    bestOf qs score ∈ qs := by
  rcases (foldl_best score qs (qs.headD [])).1 with h | h
  · unfold bestOf; rw [h]
    obtain ⟨q, qs', rfl⟩ := List.exists_cons_of_ne_nil hqs
    simp
  · exact h

omit [Fintype α] [DecidableEq α] in
theorem paths_ne_nil : ∀ t : DTree α, t.paths ≠ []
  | .leaf => by simp [DTree.paths]
  | .node _ r _ => by
    simp only [DTree.paths, ne_eq, List.append_eq_nil_iff, List.map_eq_nil_iff, not_and]
    exact fun h => absurd h (paths_ne_nil r)

theorem one_sub_sf_ge {m h : ℕ} {θ : ℝ} (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) :
    1 - m * θ ≤ 1 - binomSfGe m θ (h + 1) := by
  rw [← sum_binomTerm_range]
  have h0 : binomTerm m θ 0 = (1 - θ) ^ m := by simp [binomTerm]
  have hle : binomTerm m θ 0 ≤ ∑ i ∈ Finset.range (h + 1), binomTerm m θ i :=
    Finset.single_le_sum (fun i _ => binomTerm_nonneg hθ0 hθ1 i) (by simp)
  have hb := one_add_mul_le_pow (a := -θ) (by linarith) m
  rw [h0] at hle
  have : 1 + (m : ℝ) * -θ = 1 - m * θ := by ring
  rw [this, ← sub_eq_add_neg] at hb
  linarith

/-- A test whose rate times its trials is short of `1 − a` fires on its first hit. -/
theorem testFires_of_hit {θ a : ℝ} {m h : ℕ} (ha : 0 ≤ a) (hθm : θ * m < 1 - a) (h1 : 1 ≤ h)
    (hhm : h ≤ m) : testFires θ a m h := by
  have hm1 : (1 : ℝ) ≤ m := by exact_mod_cast h1.trans hhm
  unfold testFires
  split_ifs with h0 h1'
  · exact h1
  · nlinarith
  · push Not at h0 h1'
    rcases hr : rateSide θ a 0 m h with _ | s
    · simp only []
      have : θ * m < h := by
        have : (1 : ℝ) ≤ h := by exact_mod_cast h1
        linarith
      exact this
    · simp only []
      unfold rateSide at hr
      rw [if_pos (Nat.zero_le _)] at hr
      split_ifs at hr with ha1 ha2
      · simp at hr; exact hr
      · exfalso
        have := one_sub_sf_ge (m := m) (h := h) h0.le h1'.le
        have : (m : ℝ) * θ = θ * m := mul_comm _ _
        linarith

theorem rateSide_false_mono {θ a : ℝ} {n h h' : ℕ} (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1)
    (hr : rateSide θ a 0 n h = some false) (hh : h' ≤ h) :
    1 - binomSfGe n θ (h' + 1) < a := by
  unfold rateSide at hr
  rw [if_pos (Nat.zero_le _)] at hr
  split_ifs at hr with h1 h2
  · simp at hr
  · have := binomSfGe_antitone' (n := n) hθ0 hθ1 (Nat.succ_le_succ hh)
    linarith

end Combinatorics

section Outcomes

variable (R : CutReads α)

theorem walkCheck_of_not_search {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {o : Outcome α} (h : probeOutcome R t edges k x = o) (ho : ¬ o.IsSearch) :
    walkCheck R t edges k x = .inl o := by
  unfold probeOutcome at h
  rcases hc : walkCheck R t edges k x with o' | d <;> rw [hc] at h
  · simp only [Sum.elim_inl, id] at h; rw [h]
  · have := bracketAt_isSearch (α := α) (agreesAt R t x fun j => d.1.getD (j - k) []) d.1
      (d.2 - k) k d.2
    simp only [Sum.elim_inr] at h
    rw [h] at this
    exact absurd this ho

theorem probeOutcome_startUndecided {t : DTree α} {edges : Edges α} {k : ℕ} {x w : FreeMonoid α}
    (h : probeOutcome R t edges k x = .startUndecided w) : w = prefixOf x k := by
  have hw := walkCheck_of_not_search R h id
  unfold walkCheck at hw
  split at hw
  · simpa using hw.symm
  · split at hw
    · simp at hw
    · split at hw
      · simp at hw
      · split_ifs at hw <;> simp at hw
  · split at hw
    · simp at hw
    · split_ifs at hw <;> simp at hw

theorem probeOutcome_endUndecided {t : DTree α} {edges : Edges α} {k : ℕ} {x w : FreeMonoid α}
    (h : probeOutcome R t edges k x = .endUndecided w) :
    (∃ s c j, kWalk R t edges k x = .edge s c j) ∨ w = x := by
  have hw := walkCheck_of_not_search R h id
  unfold walkCheck at hw
  split at hw
  · simp at hw
  · rename_i s c j hk
    exact .inl ⟨s, c, j, hk⟩
  · split at hw
    · simp only [Sum.inl.injEq, Outcome.endUndecided.injEq] at hw
      exact .inr hw.symm
    · split_ifs at hw <;> simp at hw

theorem edgeAt_of_edge {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {ps : List (List Bool)} {fd : ℕ} (h : probeOutcome R t edges k x = .edge ps fd) :
    ∃ e, edgeAt R t edges k x = some e := by
  obtain ⟨ps', hi, hw, hbr⟩ := probeOutcome_search R h trivial
  obtain ⟨-, -, -, hkhi, hhi, -⟩ := walkCheck_inr R hw
  have := bracketAt_edge_range (α := α) _ ps' _ _ _ _ _ hkhi hbr
  have hlt : fd - 1 < x.toList.length := by omega
  unfold edgeAt
  rw [h]
  simp [List.getElem?_eq_getElem hlt]

theorem naOff_cases {t : DTree α} {edges : Edges α} {k : ℕ} {gu : List Bool × α → Prop}
    {x : FreeMonoid α} (h : NAOff R t edges k gu x) :
    LiveEdge R t edges k gu x ∨ StartDeep R t k x ∨ EndDeep R t x ∨ IsTriple R t edges k x
      ∨ IsPair R t edges k x ∨ IsBlocked R t edges k x ∨ IsMember R t edges k x := by
  obtain ⟨hna, hroot, hdead⟩ := h
  rcases ho : probeOutcome R t edges k x with _ | w | w | j | ⟨ps, fd⟩ | j | u
  · exact absurd ho hna
  · right; left
    have hw := probeOutcome_startUndecided R ho
    subst hw
    by_contra hd
    exact hroot (.inl ⟨_, ho, hd⟩)
  · rcases probeOutcome_endUndecided R ho with hB | rfl
    · right; right; right; right; right; left; exact ⟨hB, w, ho⟩
    · right; right; left
      by_contra hd
      exact hroot (.inr ⟨ho, hd⟩)
  · right; right; right; right; left; exact ⟨j, ho⟩
  · obtain ⟨e, he⟩ := edgeAt_of_edge R ho
    by_cases hg : gu e
    · exact absurd ⟨e, he, hg⟩ hdead
    · left; exact ⟨e, he, hg⟩
  · right; right; right; left; exact ⟨j, ho⟩
  · right; right; right; right; right; right; exact ⟨u, ho⟩

end Outcomes

section Need

variable (R : CutReads α) {Q : Type*}

open scoped Classical in
/-- A covering start disagreeing on more than `1 − acc` leaves the classes and live edges more
than `need`. -/
theorem need_lt (A : DFA (FreeMonoid α) Q) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (S : Set Q) (h : Q → List Bool) (t : DTree α) (edges : Edges α) (k : ℕ) (q₀ : Q)
    (gu : List Bool × α → Prop) {acc η : ℝ} (hcov : 1 - η ≤ D.real (CoverGood A S q₀))
    (hh : ∀ q ∈ S, leafAccepts (h q) = decide (q ∈ A.accept))
    (herr : 1 - acc < D.real {x | StartDis R edges (h q₀) x}) :
    need R A D S h t edges k q₀ gu acc η < D.real {x | NAOff R t edges k gu x} := by
  set CG := CoverGood A S q₀
  set LE := {x : FreeMonoid α | R.mid x ≠ decide (A.state x ∈ A.accept)}
  set MK := {x | x ∈ CG ∧ probeOutcome R t edges k x = .agree
    ∧ ∃ i ≤ x.toList.length, tRun edges (h q₀) (prefixOf x i) ≠ h (A.step q₀ (prefixOf x i))}
  set RT := {x | RootUndecided R t edges k x}
  set DE := {x | DeadEdge R t edges k gu x}
  set NA := {x | NAOff R t edges k gu x}
  have hsub : {x | StartDis R edges (h q₀) x} ⊆ CGᶜ ∪ (LE ∪ (MK ∪ (NA ∪ (RT ∪ DE)))) := by
    intro x hx
    by_cases hcg : x ∈ CG
    swap
    · exact .inl hcg
    by_cases hl : R.mid x = decide (A.state x ∈ A.accept)
    swap
    · exact .inr (.inl hl)
    have hS : A.step q₀ x ∈ S := by
      have := hcg.1 x.toList.length le_rfl
      rwa [prefixOf_length] at this
    have hne : tRun edges (h q₀) x ≠ h (A.step q₀ x) := by
      intro heq
      apply hx
      rw [heq, hh _ hS, hl]
      exact decide_eq_decide.2 hcg.2
    by_cases hag : probeOutcome R t edges k x = .agree
    · exact .inr (.inr (.inl ⟨hcg, hag, x.toList.length, le_rfl, by
        rw [prefixOf_length]; exact hne⟩))
    by_cases hr : RootUndecided R t edges k x
    · exact .inr (.inr (.inr (.inr (.inl hr))))
    by_cases hd : DeadEdge R t edges k gu x
    · exact .inr (.inr (.inr (.inr (.inr hd))))
    exact .inr (.inr (.inr (.inl ⟨hag, hr, hd⟩)))
  have hcg : D.real CGᶜ ≤ η := by
    rw [measureReal_compl (Set.to_countable _).measurableSet, probReal_univ]
    linarith
  have h1 := measureReal_mono (μ := D) hsub (measure_ne_top _ _)
  have h2 := measureReal_union_le (μ := D) CGᶜ (LE ∪ (MK ∪ (NA ∪ (RT ∪ DE))))
  have h3 := measureReal_union_le (μ := D) LE (MK ∪ (NA ∪ (RT ∪ DE)))
  have h4 := measureReal_union_le (μ := D) MK (NA ∪ (RT ∪ DE))
  have h5 := measureReal_union_le (μ := D) NA (RT ∪ DE)
  have h6 := measureReal_union_le (μ := D) RT DE
  unfold need labelErr maskMass
  linarith

end Need

section Stops

omit [Fintype α] [DecidableEq α] in
theorem gateStop_spec (R : CutReads α) (t : DTree α) (edges : Edges α) {N : ℕ}
    (bg : Fin N → FreeMonoid α) (acc a : ℝ) :
    gateStop R t edges bg acc a ∈ lookSet 30 N
      ∧ ((gateSide R t edges bg acc a (gateStop R t edges bg acc a)).isSome
        ∨ (gateSide R t edges bg acc a (gateStop R t edges bg acc a) = none
          ∧ gateStop R t edges bg acc a = N)) := by
  have hN : N ∈ lookSet 30 N := Finset.mem_insert_self _ _
  unfold gateStop
  rcases hf : ((lookSet 30 N).sort (· ≤ ·)).find?
      (fun n => (gateSide R t edges bg acc a n).isSome) with _ | n
  · simp only [Option.getD_none]
    have := List.find?_eq_none.1 hf N ((Finset.mem_sort _).2 hN)
    exact ⟨hN, .inr ⟨by simpa using this, trivial⟩⟩
  · simp only [Option.getD_some]
    exact ⟨(Finset.mem_sort _).1 (List.mem_of_find?_eq_some hf),
      .inl (by have := List.find?_some hf; simpa using this)⟩

open scoped Classical in
omit [Fintype α] [DecidableEq α] in
theorem refusalStop_spec {N : ℕ} (b : Fin N → FreeMonoid α) (a : ℝ)
    (tests : List (ClassTest α)) (live : FreeMonoid α → Prop) :
    refusalStop b a tests live = N
      ∨ (∃ T ∈ tests, T.fires b a (refusalStop b a tests live))
      ∨ ∃ i : Fin N, (i : ℕ) < refusalStop b a tests live ∧ live (b i) := by
  unfold refusalStop
  rcases hf : ((lookSet 30 N).sort (· ≤ ·)).find? (fun n =>
      decide ((∃ T ∈ tests, T.fires b a n) ∨ ∃ i : Fin N, (i : ℕ) < n ∧ live (b i))) with _ | n
  · exact .inl rfl
  · simp only [Option.getD_some]
    have := List.find?_some hf
    exact .inr (by simpa using this)

end Stops

section Rates

theorem logb_steps_nonneg (L k : ℕ) : 0 ≤ searchSteps L k := by
  unfold searchSteps
  rcases Nat.eq_zero_or_pos (L - k) with h | h
  · rw [h]; simp
  · have : 0 ≤ Real.logb 2 ((L - k : ℕ) : ℝ) :=
      Real.logb_nonneg (by norm_num) (by exact_mod_cast h)
    linarith

omit [Fintype α] [DecidableEq α] in
theorem tau_rate {t : DTree α} {k L nr : ℕ} {c a f coef : ℝ} (hf : 0 ≤ f)
    (htau : f < tauZero t k L nr c a) (hcoef : coef ≤ max ((t.depth - 1 : ℕ) : ℝ)
      (max (2 * c * t.depth) (c * t.depth * searchSteps L k))) :
    f * coef * nr < 1 - a := by
  set M := max ((t.depth - 1 : ℕ) : ℝ) (max (2 * c * t.depth) (c * t.depth * searchSteps L k))
  have hM : 0 ≤ M := (Nat.cast_nonneg _).trans (le_max_left _ _)
  have hnM : 0 < (nr : ℝ) * M := by
    rcases (mul_nonneg (Nat.cast_nonneg nr) hM).lt_or_eq with h | h
    · exact h
    · exfalso
      unfold tauZero at htau
      rw [← h, div_zero] at htau
      linarith
  have := (lt_div_iff₀ hnM).1 htau
  have h2 : f * coef * nr ≤ f * M * nr :=
    mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hcoef hf) (Nat.cast_nonneg _)
  linarith

/-- Below `τ₀`, with members firing on their first hit, every harvest class does. -/
theorem harvest_fires (R : CutReads α) (t : DTree α) (edges : Edges α) (k L : ℕ) {f c θM a : ℝ}
    (hf : 0 ≤ f) (ha : 0 ≤ a) {nr : ℕ} (htau : f < tauZero t k L nr c a)
    (hM : θM * nr < 1 - a) (br : Fin nr → FreeMonoid α) :
    ∀ T ∈ harvestTests R t edges k L f c θM, (∀ x, T.hits x → T.trials x) →
      1 ≤ hitsIn br T.hits nr → T.fires br a nr := by
  classical
  have hS := logb_steps_nonneg L k
  have hd : (0 : ℝ) ≤ t.depth := Nat.cast_nonneg _
  intro T hT hht h1
  have hm : hitsIn br T.trials nr ≤ nr := (Finset.card_filter_le _ _).trans (by simp)
  have hhm : hitsIn br T.hits nr ≤ hitsIn br T.trials nr :=
    Finset.card_le_card fun i hi => by
      simp only [Finset.mem_filter] at hi ⊢; exact ⟨hi.1, hi.2.1, hht _ hi.2.2⟩
  have hmr : (hitsIn br T.trials nr : ℝ) ≤ nr := by exact_mod_cast hm
  unfold ClassTest.fires
  by_cases hθ : T.θ ≤ 0
  · unfold testFires; rw [if_pos hθ]; omega
  push Not at hθ
  refine testFires_of_hit ha ?_ h1 hhm
  refine lt_of_le_of_lt (mul_le_mul_of_nonneg_left hmr hθ.le) ?_
  simp only [harvestTests, List.mem_cons, List.not_mem_nil, or_false] at hT
  rcases hT with rfl | rfl | rfl | rfl | rfl | rfl
  · have := tau_rate hf htau (le_max_left _ _)
    simp only []; linarith [mul_comm ((t.depth - 1 : ℕ) : ℝ) f]
  · have := tau_rate hf htau (le_max_left _ _)
    simp only []; linarith [mul_comm ((t.depth - 1 : ℕ) : ℝ) f]
  · have := tau_rate hf htau ((le_max_right _ _).trans (le_max_right _ _))
    simp only []; linarith [show c * f * t.depth * searchSteps L k
      = f * (c * t.depth * searchSteps L k) by ring]
  · have := tau_rate hf htau ((le_max_right _ _).trans (le_max_right _ _))
    simp only []; linarith [show c * f * t.depth * searchSteps L k
      = f * (c * t.depth * searchSteps L k) by ring]
  · have := tau_rate hf htau ((le_max_left _ _).trans (le_max_right _ _))
    simp only []; linarith [show 2 * c * f * t.depth = f * (2 * c * t.depth) by ring]
  · simp only []; exact hM

end Rates

section Batch

/-- A sample missing every draw of `C`, when `C` carries more than `ν`. -/
theorem miss_all_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (C : FreeMonoid α → Prop) (n : ℕ) {ν : ℝ} (hν : ν ≤ 1) :
    (Measure.pi fun _ : Fin n => D).real {b | ν < D.real {x | C x} ∧ ∀ i, ¬ C (b i)}
      ≤ (1 - ν) ^ n := by
  by_cases h : ν < D.real {x | C x}
  · have hset : {b : Fin n → FreeMonoid α | ν < D.real {x | C x} ∧ ∀ i, ¬ C (b i)}
        = Set.univ.pi fun _ : Fin n => {x | ¬ C x} := by
      ext b
      simp [h]
    rw [hset, measureReal_def, Measure.pi_pi]
    have hc : D {x | ¬ C x} = ENNReal.ofReal (1 - D.real {x | C x}) := by
      rw [show {x | ¬ C x} = {x | C x}ᶜ from rfl, ← ENNReal.ofReal_toReal (measure_ne_top D _),
        ← measureReal_def, measureReal_compl (Set.to_countable _).measurableSet, probReal_univ]
    have hp1 : D.real {x | C x} ≤ 1 := measureReal_le_one
    simp only [hc, Finset.prod_const, Finset.card_univ, Fintype.card_fin, ENNReal.toReal_pow,
      ENNReal.toReal_ofReal (by linarith : (0 : ℝ) ≤ 1 - D.real {x | C x})]
    exact pow_le_pow_left₀ (by linarith) (by linarith) _
  · rw [show {b : Fin n → FreeMonoid α | ν < D.real {x | C x} ∧ ∀ i, ¬ C (b i)} = ∅ from
      Set.eq_empty_of_forall_notMem fun b hb => h hb.1]
    simp only [measureReal_empty]
    exact pow_nonneg (by linarith) _

variable (R : CutReads α) {Q : Type*}

theorem gate_look_above (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) {ng n : ℕ} (hn : n ≤ ng) {acc a : ℝ} (hacc1 : acc ≤ 1)
    (hp : D.real {x | P x} ≤ acc) (ha : 0 ≤ a) :
    (Measure.pi fun _ : Fin ng => D).real {bg | binomSfGe n acc (hitsIn bg P n) < a} ≤ a := by
  classical
  refine le_of_eq_of_le ?_ (look_above_le D P (Finset.univ.filter fun i : Fin ng => (i : ℕ) < n)
    hacc1 hp ha)
  congr 1
  ext bg
  simp only [Set.mem_ofPred_eq, (look_set hn bg P).1, (look_set hn bg P).2]

theorem gate_look_below (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) {ng n : ℕ} (hn : n ≤ ng) {acc a : ℝ} (hacc0 : 0 ≤ acc)
    (hp : acc ≤ D.real {x | P x}) (ha : 0 ≤ a) :
    (Measure.pi fun _ : Fin ng => D).real {bg | 1 - binomSfGe n acc (hitsIn bg P n + 1) < a}
      ≤ a := by
  classical
  refine le_of_eq_of_le ?_ (look_below_le D P (Finset.univ.filter fun i : Fin ng => (i : ℕ) < n)
    hacc0 hp ha)
  congr 1
  ext bg
  simp only [Set.mem_ofPred_eq, (look_set hn bg P).1, (look_set hn bg P).2]

theorem gate_tail (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) (ng j : ℕ) {θ : ℝ} (hθ1 : θ ≤ 1) (hp : D.real {x | P x} ≤ θ) :
    (Measure.pi fun _ : Fin ng => D).real {bg | j ≤ hitsIn bg P ng} ≤ binomSfGe ng θ j := by
  classical
  have h := pi_count_ge D P (Finset.univ.filter fun i : Fin ng => (i : ℕ) < ng) j
  rw [(look_set le_rfl (fun _ => (1 : FreeMonoid α)) P).2] at h
  refine le_of_eq_of_le ?_ ((le_of_eq h).trans
    (binomSfGe_mono measureReal_nonneg hθ1 hp _ _))
  congr 1
  ext bg
  simp only [Set.mem_ofPred_eq, (look_set le_rfl bg P).1]

open scoped Classical in
/-- The gate's claims: off a set of batches of the stated chance, a pass leaves its start within
`δc` of `acc`, and a refusal leaves every start short of `acc`. -/
theorem gate_claims (s : KState α) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (ng : ℕ) {acc a δc : ℝ} (hacc0 : 0 ≤ acc) (hacc1 : acc ≤ 1) (ha : 0 ≤ a) (hδc0 : 0 ≤ δc)
    (hδc : δc ≤ acc) :
    ∃ Bad : Set (Fin ng → FreeMonoid α),
      (Measure.pi fun _ : Fin ng => D).real Bad
        ≤ 2 * (Nat.log 2 (ng / 30) + 2) * a
          + s.tree.paths.length
            * binomSfGe ng (acc - δc) (gateCut ng acc (a / s.tree.paths.length))
      ∧ ∀ bg ∉ Bad,
        (GatePasses R s.tree s.edges bg acc a
          → D.real {x | StartDis R s.edges (gateStart R s.tree s.edges bg acc a) x} ≤ 1 - acc + δc)
        ∧ (¬ GatePasses R s.tree s.edges bg acc a
          → ∀ q ∈ s.tree.paths, D.real {x | ¬ StartDis R s.edges q x} < acc) := by
  set t := s.tree with ht
  set e := s.edges with he
  set P := t.paths with hP
  set a' := a / P.length with ha'
  have hP0 : 0 < P.length := List.length_pos_of_ne_nil (paths_ne_nil t)
  have ha'0 : 0 ≤ a' := div_nonneg ha (Nat.cast_nonneg _)
  set pq : List Bool → ℝ := fun q => D.real {x | ¬ StartDis R e q x} with hpq
  have hpq_compl : ∀ q, pq q = 1 - D.real {x | StartDis R e q x} := fun q => by
    simp only [hpq]
    rw [show {x | ¬ StartDis R e q x} = {x | StartDis R e q x}ᶜ from rfl,
      measureReal_compl (Set.to_countable _).measurableSet, probReal_univ]
  set looks := lookSet 30 ng
  set Gab := ⋃ q ∈ P.toFinset.filter (fun q => pq q < acc), ⋃ n ∈ looks,
    {bg : Fin ng → FreeMonoid α | binomSfGe n acc (hitsIn bg (fun x => ¬ StartDis R e q x) n) < a'}
  set Gbe := ⋃ q ∈ P.toFinset.filter (fun q => acc ≤ pq q), ⋃ n ∈ looks,
    {bg : Fin ng → FreeMonoid α |
      1 - binomSfGe n acc (hitsIn bg (fun x => ¬ StartDis R e q x) n + 1) < a'}
  set Gun := ⋃ q ∈ P.toFinset.filter (fun q => pq q < acc - δc),
    {bg : Fin ng → FreeMonoid α | gateCut ng acc a' ≤ hitsIn bg (fun x => ¬ StartDis R e q x) ng}
  refine ⟨Gab ∪ Gbe ∪ Gun, ?_, fun bg hbad => ?_⟩
  · set ν₁ := Measure.pi fun _ : Fin ng => D
    have hfil : ∀ p : List Bool → Prop, ((P.toFinset.filter p).card : ℝ) * a' ≤ a := by
      intro p
      have h1 : ((P.toFinset.filter p).card : ℝ) ≤ P.length := by
        exact_mod_cast (Finset.card_filter_le _ _).trans (List.toFinset_card_le _)
      rw [ha', mul_div_assoc']
      rw [div_le_iff₀ (by exact_mod_cast hP0)]
      nlinarith
    have hlooks : (looks.card : ℝ) ≤ Nat.log 2 (ng / 30) + 2 := by
      exact_mod_cast lookSet_card 30 ng
    have hGab : ν₁.real Gab ≤ (Nat.log 2 (ng / 30) + 2) * a := by
      refine (measureReal_biUnion_finset_le _ _).trans ?_
      calc ∑ q ∈ P.toFinset.filter (fun q => pq q < acc), ν₁.real (⋃ n ∈ looks,
            {bg : Fin ng → FreeMonoid α |
              binomSfGe n acc (hitsIn bg (fun x => ¬ StartDis R e q x) n) < a'})
          ≤ ∑ _q ∈ P.toFinset.filter (fun q => pq q < acc), (looks.card : ℝ) * a' := by
            refine Finset.sum_le_sum fun q hq => (measureReal_biUnion_finset_le _ _).trans ?_
            rw [← nsmul_eq_mul, ← Finset.sum_const]
            exact Finset.sum_le_sum fun n hn => gate_look_above D _ (mem_lookSet hn).1 hacc1
              (Finset.mem_filter.1 hq).2.le ha'0
        _ = (looks.card : ℝ) * ((P.toFinset.filter (fun q => pq q < acc)).card * a') := by
            rw [Finset.sum_const, nsmul_eq_mul]; ring
        _ ≤ (Nat.log 2 (ng / 30) + 2) * a :=
            mul_le_mul hlooks (hfil _) (by positivity) (by positivity)
    have hGbe : ν₁.real Gbe ≤ (Nat.log 2 (ng / 30) + 2) * a := by
      refine (measureReal_biUnion_finset_le _ _).trans ?_
      calc ∑ q ∈ P.toFinset.filter (fun q => acc ≤ pq q), ν₁.real (⋃ n ∈ looks,
            {bg : Fin ng → FreeMonoid α |
              1 - binomSfGe n acc (hitsIn bg (fun x => ¬ StartDis R e q x) n + 1) < a'})
          ≤ ∑ _q ∈ P.toFinset.filter (fun q => acc ≤ pq q), (looks.card : ℝ) * a' := by
            refine Finset.sum_le_sum fun q hq => (measureReal_biUnion_finset_le _ _).trans ?_
            rw [← nsmul_eq_mul, ← Finset.sum_const]
            exact Finset.sum_le_sum fun n hn => gate_look_below D _ (mem_lookSet hn).1 hacc0
              (Finset.mem_filter.1 hq).2 ha'0
        _ = (looks.card : ℝ) * ((P.toFinset.filter (fun q => acc ≤ pq q)).card * a') := by
            rw [Finset.sum_const, nsmul_eq_mul]; ring
        _ ≤ (Nat.log 2 (ng / 30) + 2) * a :=
            mul_le_mul hlooks (hfil _) (by positivity) (by positivity)
    have htail0 : 0 ≤ binomSfGe ng (acc - δc) (gateCut ng acc a') :=
      binomSfGe_nonneg (by linarith) (by linarith) _
    have hGun : ν₁.real Gun ≤ P.length * binomSfGe ng (acc - δc) (gateCut ng acc a') := by
      refine (measureReal_biUnion_finset_le _ _).trans ?_
      calc ∑ q ∈ P.toFinset.filter (fun q => pq q < acc - δc), ν₁.real
            {bg : Fin ng → FreeMonoid α |
              gateCut ng acc a' ≤ hitsIn bg (fun x => ¬ StartDis R e q x) ng}
          ≤ ∑ _q ∈ P.toFinset.filter (fun q => pq q < acc - δc),
              binomSfGe ng (acc - δc) (gateCut ng acc a') :=
            Finset.sum_le_sum fun q hq => gate_tail D _ ng _ (by linarith)
              (Finset.mem_filter.1 hq).2.le
        _ = (P.toFinset.filter (fun q => pq q < acc - δc)).card
              * binomSfGe ng (acc - δc) (gateCut ng acc a') := by
            rw [Finset.sum_const, nsmul_eq_mul]
        _ ≤ P.length * binomSfGe ng (acc - δc) (gateCut ng acc a') := by
            refine mul_le_mul_of_nonneg_right ?_ htail0
            exact_mod_cast (Finset.card_filter_le _ _).trans (List.toFinset_card_le _)
    refine (measureReal_union_le _ _).trans ?_
    refine (add_le_add (measureReal_union_le _ _) hGun).trans ?_
    linarith
  simp only [Set.mem_union, not_or] at hbad
  obtain ⟨⟨hab, hbe⟩, hun⟩ := hbad
  obtain ⟨hstop, hside⟩ := gateStop_spec R t e bg acc a
  set stop := gateStop R t e bg acc a
  set qh := gateStart R t e bg acc a
  have hqh : qh ∈ P := bestOf_mem (paths_ne_nil t) _
  have hgs : gateSide R t e bg acc a stop
      = rateSide acc a' 0 stop (hitsIn bg (fun x => ¬ StartDis R e qh x) stop) := rfl
  refine ⟨fun hpass => ?_, fun hpass => ?_⟩
  · by_contra hA
    have hlt : pq qh < acc - δc := by
      rw [hpq_compl]; push Not at hA; linarith
    rcases hside with hs | ⟨hs, hN⟩
    · have hv : gateSide R t e bg acc a stop = some true := by
        rcases hv : gateSide R t e bg acc a stop with _ | _ | _
        · rw [hv] at hs; exact absurd hs (by simp)
        · exact absurd hv hpass
        · rfl
      rw [hgs] at hv
      unfold rateSide at hv
      rw [if_pos (Nat.zero_le _)] at hv
      split_ifs at hv with h1 h2
      · exact hab (Set.mem_biUnion (Finset.mem_filter.2 ⟨List.mem_toFinset.2 hqh, by linarith⟩)
          (Set.mem_biUnion hstop h1))
      · simp at hv
    · rw [hgs] at hs
      unfold rateSide at hs
      rw [if_pos (Nat.zero_le _)] at hs
      split_ifs at hs with h1 h2
      refine hun (Set.mem_biUnion (Finset.mem_filter.2 ⟨List.mem_toFinset.2 hqh, hlt⟩) ?_)
      simp only [Set.mem_ofPred_eq]
      have : gateCut stop acc a' ≤ hitsIn bg (fun x => ¬ StartDis R e qh x) stop :=
        Nat.sInf_le (show hitsIn bg (fun x => ¬ StartDis R e qh x) stop
          ∈ {h | ¬ 1 - binomSfGe stop acc (h + 1) < a'} from h2)
      rwa [hN] at this
  · have hfalse : gateSide R t e bg acc a stop = some false := by
      unfold GatePasses at hpass; push Not at hpass; exact hpass
    intro q hq
    by_contra hge
    push Not at hge
    have hle : hitsIn bg (fun x => ¬ StartDis R e q x) stop
        ≤ hitsIn bg (fun x => ¬ StartDis R e qh x) stop :=
      bestOf_max P (fun q => hitsIn bg (fun x => ¬ StartDis R e q x) stop) hq
    have := rateSide_false_mono hacc0 hacc1 (hgs.symm.trans hfalse) hle
    exact hbe (Set.mem_biUnion (Finset.mem_filter.2 ⟨List.mem_toFinset.2 hq, hge⟩)
      (Set.mem_biUnion hstop this))

open scoped Classical in
/-- The gate's claims where its test settles: off a set of batches of chance at most the looks'
failure chances, settling above leaves its start disagreeing on at most `1 − acc`, and settling
below leaves every start short of `acc`. -/
theorem gate_settled (s : KState α) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (ng : ℕ) {acc a : ℝ} (hacc0 : 0 ≤ acc) (hacc1 : acc ≤ 1) (ha : 0 ≤ a) :
    ∃ Bad : Set (Fin ng → FreeMonoid α),
      (Measure.pi fun _ : Fin ng => D).real Bad ≤ 2 * (Nat.log 2 (ng / 30) + 2) * a
      ∧ ∀ bg ∉ Bad,
        (gateSide R s.tree s.edges bg acc a (gateStop R s.tree s.edges bg acc a) = some true
          → D.real {x | StartDis R s.edges (gateStart R s.tree s.edges bg acc a) x} ≤ 1 - acc)
        ∧ (gateSide R s.tree s.edges bg acc a (gateStop R s.tree s.edges bg acc a) = some false
          → ∀ q ∈ s.tree.paths, D.real {x | ¬ StartDis R s.edges q x} < acc) := by
  set t := s.tree with ht
  set e := s.edges with he
  set P := t.paths with hP
  set a' := a / P.length with ha'
  have hP0 : 0 < P.length := List.length_pos_of_ne_nil (paths_ne_nil t)
  have ha'0 : 0 ≤ a' := div_nonneg ha (Nat.cast_nonneg _)
  set pq : List Bool → ℝ := fun q => D.real {x | ¬ StartDis R e q x} with hpq
  have hpq_compl : ∀ q, pq q = 1 - D.real {x | StartDis R e q x} := fun q => by
    simp only [hpq]
    rw [show {x | ¬ StartDis R e q x} = {x | StartDis R e q x}ᶜ from rfl,
      measureReal_compl (Set.to_countable _).measurableSet, probReal_univ]
  set looks := lookSet 30 ng
  set Gab := ⋃ q ∈ P.toFinset.filter (fun q => pq q < acc), ⋃ n ∈ looks,
    {bg : Fin ng → FreeMonoid α | binomSfGe n acc (hitsIn bg (fun x => ¬ StartDis R e q x) n) < a'}
  set Gbe := ⋃ q ∈ P.toFinset.filter (fun q => acc ≤ pq q), ⋃ n ∈ looks,
    {bg : Fin ng → FreeMonoid α |
      1 - binomSfGe n acc (hitsIn bg (fun x => ¬ StartDis R e q x) n + 1) < a'}
  refine ⟨Gab ∪ Gbe, ?_, fun bg hbad => ?_⟩
  · set ν₁ := Measure.pi fun _ : Fin ng => D
    have hfil : ∀ p : List Bool → Prop, ((P.toFinset.filter p).card : ℝ) * a' ≤ a := by
      intro p
      have h1 : ((P.toFinset.filter p).card : ℝ) ≤ P.length := by
        exact_mod_cast (Finset.card_filter_le _ _).trans (List.toFinset_card_le _)
      rw [ha', mul_div_assoc']
      rw [div_le_iff₀ (by exact_mod_cast hP0)]
      nlinarith
    have hlooks : (looks.card : ℝ) ≤ Nat.log 2 (ng / 30) + 2 := by
      exact_mod_cast lookSet_card 30 ng
    have hGab : ν₁.real Gab ≤ (Nat.log 2 (ng / 30) + 2) * a := by
      refine (measureReal_biUnion_finset_le _ _).trans ?_
      calc ∑ q ∈ P.toFinset.filter (fun q => pq q < acc), ν₁.real (⋃ n ∈ looks,
            {bg : Fin ng → FreeMonoid α |
              binomSfGe n acc (hitsIn bg (fun x => ¬ StartDis R e q x) n) < a'})
          ≤ ∑ _q ∈ P.toFinset.filter (fun q => pq q < acc), (looks.card : ℝ) * a' := by
            refine Finset.sum_le_sum fun q hq => (measureReal_biUnion_finset_le _ _).trans ?_
            rw [← nsmul_eq_mul, ← Finset.sum_const]
            exact Finset.sum_le_sum fun n hn => gate_look_above D _ (mem_lookSet hn).1 hacc1
              (Finset.mem_filter.1 hq).2.le ha'0
        _ = (looks.card : ℝ) * ((P.toFinset.filter (fun q => pq q < acc)).card * a') := by
            rw [Finset.sum_const, nsmul_eq_mul]; ring
        _ ≤ (Nat.log 2 (ng / 30) + 2) * a :=
            mul_le_mul hlooks (hfil _) (by positivity) (by positivity)
    have hGbe : ν₁.real Gbe ≤ (Nat.log 2 (ng / 30) + 2) * a := by
      refine (measureReal_biUnion_finset_le _ _).trans ?_
      calc ∑ q ∈ P.toFinset.filter (fun q => acc ≤ pq q), ν₁.real (⋃ n ∈ looks,
            {bg : Fin ng → FreeMonoid α |
              1 - binomSfGe n acc (hitsIn bg (fun x => ¬ StartDis R e q x) n + 1) < a'})
          ≤ ∑ _q ∈ P.toFinset.filter (fun q => acc ≤ pq q), (looks.card : ℝ) * a' := by
            refine Finset.sum_le_sum fun q hq => (measureReal_biUnion_finset_le _ _).trans ?_
            rw [← nsmul_eq_mul, ← Finset.sum_const]
            exact Finset.sum_le_sum fun n hn => gate_look_below D _ (mem_lookSet hn).1 hacc0
              (Finset.mem_filter.1 hq).2 ha'0
        _ = (looks.card : ℝ) * ((P.toFinset.filter (fun q => acc ≤ pq q)).card * a') := by
            rw [Finset.sum_const, nsmul_eq_mul]; ring
        _ ≤ (Nat.log 2 (ng / 30) + 2) * a :=
            mul_le_mul hlooks (hfil _) (by positivity) (by positivity)
    refine (measureReal_union_le _ _).trans ?_
    linarith
  simp only [Set.mem_union, not_or] at hbad
  obtain ⟨hab, hbe⟩ := hbad
  obtain ⟨hstop, -⟩ := gateStop_spec R t e bg acc a
  set stop := gateStop R t e bg acc a
  set qh := gateStart R t e bg acc a
  have hqh : qh ∈ P := bestOf_mem (paths_ne_nil t) _
  have hgs : gateSide R t e bg acc a stop
      = rateSide acc a' 0 stop (hitsIn bg (fun x => ¬ StartDis R e qh x) stop) := rfl
  refine ⟨fun hv => ?_, fun hfalse => ?_⟩
  · by_contra hA
    have hlt : pq qh < acc := by
      rw [hpq_compl]; push Not at hA; linarith
    rw [hgs] at hv
    unfold rateSide at hv
    rw [if_pos (Nat.zero_le _)] at hv
    split_ifs at hv with h1 h2
    · exact hab (Set.mem_biUnion (Finset.mem_filter.2 ⟨List.mem_toFinset.2 hqh, hlt⟩)
        (Set.mem_biUnion hstop h1))
    · simp at hv
  · intro q hq
    by_contra hge
    push Not at hge
    have hle : hitsIn bg (fun x => ¬ StartDis R e q x) stop
        ≤ hitsIn bg (fun x => ¬ StartDis R e qh x) stop :=
      bestOf_max P (fun q => hitsIn bg (fun x => ¬ StartDis R e q x) stop) hq
    have := rateSide_false_mono hacc0 hacc1 (hgs.symm.trans hfalse) hle
    exact hbe (Set.mem_biUnion (Finset.mem_filter.2 ⟨List.mem_toFinset.2 hq, hge⟩)
      (Set.mem_biUnion hstop this))

open scoped Classical in
/-- A refusal sample that reads to its end with no class firing and no live edge, below `τ₀`,
holds no draw the walk leaves non-agreeing off the root and the given-up edges. -/
theorem refusal_clear (t : DTree α) (edges : Edges α) (k L : ℕ) {f c θM a : ℝ} (hf : 0 ≤ f)
    (ha : 0 ≤ a) (gu : List Bool × α → Prop) {nr : ℕ} (br : Fin nr → FreeMonoid α)
    (htau : f < tauZero t k L nr c a) (hM : θM * nr < 1 - a)
    (hB : ¬ ∃ i : Fin nr, (i : ℕ) < refusalStop br a (harvestTests R t edges k L f c θM)
      (LiveEdge R t edges k gu) ∧ LiveEdge R t edges k gu (br i))
    (hC : ¬ ∃ T ∈ harvestTests R t edges k L f c θM, T.fires br a
      (refusalStop br a (harvestTests R t edges k L f c θM) (LiveEdge R t edges k gu))) :
    ∀ i, ¬ NAOff R t edges k gu (br i) := by
  intro i hi
  set tests := harvestTests R t edges k L f c θM
  set Tr := refusalStop br a tests (LiveEdge R t edges k gu)
  have hTr : Tr = nr := by
    rcases refusalStop_spec br a tests (LiveEdge R t edges k gu) with h1 | h1 | h1
    · exact h1
    · exact absurd h1 hC
    · exact absurd h1 hB
  have hfire : ∀ T ∈ tests, T.hits (br i) → (∀ x, T.hits x → T.trials x) → False := by
    intro T hT hx hht
    have h1 : 1 ≤ hitsIn br T.hits nr :=
      Finset.card_pos.2 ⟨i, Finset.mem_filter.2 ⟨Finset.mem_univ _, i.2, hx⟩⟩
    exact hC ⟨T, hT, by rw [hTr]; exact harvest_fires R t edges k L hf ha htau hM br T hT hht h1⟩
  have hsearch : ∀ x, (∃ j, probeOutcome R t edges k x = .triple j)
      ∨ (∃ j, probeOutcome R t edges k x = .pair j) → Searched R t edges k x := by
    rintro x (⟨j, hj⟩ | ⟨j, hj⟩)
    · obtain ⟨ps, hi', hw, -⟩ := probeOutcome_search R hj trivial
      simp [Searched, hw]
    · obtain ⟨ps, hi', hw, -⟩ := probeOutcome_search R hj trivial
      simp [Searched, hw]
  rcases naOff_cases R hi with hl | hc' | hc' | hc' | hc' | hc' | hc'
  · exact hB ⟨i, by rw [hTr]; exact i.2, hl⟩
  · exact hfire ⟨StartDeep R t k, fun _ => True, (t.depth - 1 : ℕ) * f⟩
      (by simp [tests, harvestTests]) hc' fun _ _ => trivial
  · exact hfire ⟨EndDeep R t, fun _ => True, (t.depth - 1 : ℕ) * f⟩
      (by simp [tests, harvestTests]) hc' fun _ _ => trivial
  · exact hfire ⟨IsTriple R t edges k, Searched R t edges k, c * f * t.depth * searchSteps L k⟩
      (by simp [tests, harvestTests]) hc' fun x hx => hsearch x (.inl hx)
  · exact hfire ⟨IsPair R t edges k, Searched R t edges k, c * f * t.depth * searchSteps L k⟩
      (by simp [tests, harvestTests]) hc' fun x hx => hsearch x (.inr hx)
  · exact hfire ⟨IsBlocked R t edges k, fun _ => True, 2 * c * f * t.depth⟩
      (by simp [tests, harvestTests]) hc' fun _ _ => trivial
  · exact hfire ⟨IsMember R t edges k, fun _ => True, θM⟩
      (by simp [tests, harvestTests]) hc' fun _ _ => trivial

open scoped Classical in
theorem trichotomy_batch (A : DFA (FreeMonoid α) Q) (s : KState α) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (k L ng nr : ℕ) {f c θM acc a δc η minCov ν : ℝ}
    (gu : List Bool × α → Prop) (hacc0 : 0 ≤ acc) (hacc1 : acc ≤ 1) (hf : 0 ≤ f) (ha : 0 ≤ a)
    (hδc0 : 0 ≤ δc) (hδc : δc ≤ acc) (hν : ν ≤ 1) :
    ((Measure.pi fun _ : Fin ng => D).prod (Measure.pi fun _ : Fin nr => D)).real
        {b | ¬ TrichotomyHolds R A s D k L f c θM acc a δc η minCov ν gu b.1 b.2}
      ≤ 2 * (Nat.log 2 (ng / 30) + 2) * a
        + max 1 s.tree.paths.length
          * binomSfGe ng (acc - δc) (gateCut ng acc (a / s.tree.paths.length))
        + (1 - ν) ^ nr := by
  obtain ⟨Bad, hBad, hgood⟩ := gate_claims R s D ng hacc0 hacc1 ha hδc0 hδc
  set Miss := {br : Fin nr → FreeMonoid α | ν < D.real {x | NAOff R s.tree s.edges k gu x}
    ∧ ∀ i, ¬ NAOff R s.tree s.edges k gu (br i)}
  have hsub : {b : (Fin ng → FreeMonoid α) × (Fin nr → FreeMonoid α) |
      ¬ TrichotomyHolds R A s D k L f c θM acc a δc η minCov ν gu b.1 b.2}
      ⊆ Bad ×ˢ Set.univ ∪ Set.univ ×ˢ Miss := by
    rintro ⟨bg, br⟩ hb
    simp only [Set.mem_ofPred_eq, TrichotomyHolds, not_or] at hb
    obtain ⟨hA, hB, hC, hD⟩ := hb
    by_cases hbad : bg ∈ Bad
    · exact .inl ⟨hbad, trivial⟩
    refine .inr ⟨trivial, ?_⟩
    obtain ⟨hpass1, hrefuse⟩ := hgood bg hbad
    by_cases hpass : GatePasses R s.tree s.edges bg acc a
    · exact absurd ⟨hpass, hpass1 hpass⟩ hA
    have hall := hrefuse hpass
    have hclaim := (not_and.1 hD) hpass
    simp only [not_or, not_forall, not_le] at hclaim
    obtain ⟨htau, hM, q₀, h, hq₀, hhq, hcov, hh, hneed⟩ := hclaim
    have herr : 1 - acc < D.real {x | StartDis R s.edges (h q₀) x} := by
      have := hall _ hhq
      rw [show {x | ¬ StartDis R s.edges (h q₀) x} = {x | StartDis R s.edges (h q₀) x}ᶜ from rfl,
        measureReal_compl (Set.to_countable _).measurableSet, probReal_univ] at this
      linarith
    have hNA := need_lt R A D _ h s.tree s.edges k q₀ gu hcov hh herr
    exact ⟨by linarith, refusal_clear R s.tree s.edges k L hf ha gu br htau hM
      ((not_and.1 hB) hpass) ((not_and.1 hC) hpass)⟩
  set ν₁ := Measure.pi fun _ : Fin ng => D
  set ν₂ := Measure.pi fun _ : Fin nr => D
  have hMiss := miss_all_le D (NAOff R s.tree s.edges k gu) nr hν
  have htail0 : 0 ≤ binomSfGe ng (acc - δc) (gateCut ng acc (a / s.tree.paths.length)) :=
    binomSfGe_nonneg (by linarith) (by linarith) _
  have hmax : (s.tree.paths.length : ℝ) ≤ ((max 1 s.tree.paths.length : ℕ) : ℝ) := by
    exact_mod_cast le_max_right _ _
  calc (ν₁.prod ν₂).real {b | ¬ TrichotomyHolds R A s D k L f c θM acc a δc η minCov ν gu b.1 b.2}
      ≤ (ν₁.prod ν₂).real (Bad ×ˢ Set.univ ∪ Set.univ ×ˢ Miss) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ (ν₁.prod ν₂).real (Bad ×ˢ Set.univ) + (ν₁.prod ν₂).real (Set.univ ×ˢ Miss) :=
        measureReal_union_le _ _
    _ = ν₁.real Bad + ν₂.real Miss := by
        rw [measureReal_prod_prod, measureReal_prod_prod, probReal_univ, probReal_univ, mul_one,
          one_mul]
    _ ≤ _ := by nlinarith

end Batch

end OrthoDFA
