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

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_edge_range (agrees : ℕ → Option Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi ps' j, lo < hi → bracketAt (α := α) agrees ps fuel lo hi = .edge ps' j →
      lo < j ∧ j ≤ hi
  | 0, lo, hi, ps', j, hlt, h => by
    simp only [bracketAt, Outcome.edge.injEq] at h
    omega
  | fuel + 1, lo, hi, ps', j, hlt, h => by
    simp only [bracketAt] at h
    by_cases hlh : lo + 1 < hi
    swap
    · rw [if_neg hlh] at h
      simp only [Outcome.edge.injEq] at h
      omega
    rw [if_pos hlh] at h
    have hl0 : (lo + hi) / 2 - 1 = lo → (if (lo + hi) / 2 - 1 = lo then some true
        else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1))
          = some true := fun e => if_pos e
    have hr0 : (lo + hi) / 2 + 1 = hi → (if (lo + hi) / 2 + 1 = lo then some true
        else if (lo + hi) / 2 + 1 = hi then some false else agrees ((lo + hi) / 2 + 1))
          = some false := fun e => by rw [if_neg (by omega), if_pos e]
    generalize (if (lo + hi) / 2 = lo then some true else if (lo + hi) / 2 = hi then some false
      else agrees ((lo + hi) / 2)) = v at h
    generalize (if (lo + hi) / 2 - 1 = lo then some true
      else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1)) = l at h hl0
    generalize (if (lo + hi) / 2 + 1 = lo then some true
      else if (lo + hi) / 2 + 1 = hi then some false else agrees ((lo + hi) / 2 + 1)) = r at h hr0
    rcases v with _ | _ | _
    · rcases l with _ | _ | _ <;> rcases r with _ | _ | _ <;> simp only [reduceCtorEq] at h
      · -- false, some false
        have hne : (lo + hi) / 2 - 1 ≠ lo := fun e => by simpa using hl0 e
        have := bracketAt_edge_range agrees ps fuel lo _ ps' j (by omega) h
        omega
      · have hne : (lo + hi) / 2 - 1 ≠ lo := fun e => by simpa using hl0 e
        have := bracketAt_edge_range agrees ps fuel lo _ ps' j (by omega) h
        omega
      · -- true, some true
        have hne : (lo + hi) / 2 + 1 ≠ hi := fun e => by simpa using hr0 e
        have := bracketAt_edge_range agrees ps fuel _ hi ps' j (by omega) h
        omega
    · have := bracketAt_edge_range agrees ps fuel lo _ ps' j (by omega) h
      omega
    · have := bracketAt_edge_range agrees ps fuel _ hi ps' j (by omega) h
      omega

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

/-- The draws whose walk from `k` does not agree, off the root's and the given-up edges'. -/
def NAOff (t : DTree α) (edges : Edges α) (k : ℕ) (gu : List Bool × α → Prop)
    (x : FreeMonoid α) : Prop :=
  probeOutcome R t edges k x ≠ .agree ∧ ¬ RootUndecided R t edges k x ∧ ¬ DeadEdge R t edges k gu x

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

theorem tau_rate {t : DTree α} {k L nr : ℕ} {c a f coef : ℝ} (hf : 0 ≤ f) (hc : 0 ≤ c)
    (htau : f < tauZero t k L nr c a) (hcoef : coef ≤ max ((t.depth - 1 : ℕ) : ℝ)
      (max (2 * c * t.depth) (c * t.depth * searchSteps L k))) (hcoef0 : 0 ≤ coef) :
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
    (hf : 0 ≤ f) (hc : 0 ≤ c) (ha : 0 ≤ a) {nr : ℕ} (htau : f < tauZero t k L nr c a)
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
  · have := tau_rate hf hc htau (le_max_left _ _) (Nat.cast_nonneg _)
    simp only []; linarith [mul_comm ((t.depth - 1 : ℕ) : ℝ) f]
  · have := tau_rate hf hc htau (le_max_left _ _) (Nat.cast_nonneg _)
    simp only []; linarith [mul_comm ((t.depth - 1 : ℕ) : ℝ) f]
  · have := tau_rate hf hc htau ((le_max_right _ _).trans (le_max_right _ _))
      (by positivity)
    simp only []; linarith [show c * f * t.depth * searchSteps L k
      = f * (c * t.depth * searchSteps L k) by ring]
  · have := tau_rate hf hc htau ((le_max_right _ _).trans (le_max_right _ _))
      (by positivity)
    simp only []; linarith [show c * f * t.depth * searchSteps L k
      = f * (c * t.depth * searchSteps L k) by ring]
  · have := tau_rate hf hc htau ((le_max_left _ _).trans (le_max_right _ _)) (by positivity)
    simp only []; linarith [show 2 * c * f * t.depth = f * (2 * c * t.depth) by ring]
  · simp only []; exact hM

end Rates

end OrthoDFA
