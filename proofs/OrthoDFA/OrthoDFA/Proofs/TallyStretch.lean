import OrthoDFA.Proofs.TallyCount
import OrthoDFA.Proofs.TallyProb

/-!
# A stretch's counts

While the hypothesis stays, each probe adds its own counts to the stretch's: one probe, a start
left undecided or not, and at each edge the reads it charges there and how many were undecided.
So a harvest fired within a stretch from a fresh state is a test on a sum of fresh draws' counts
against the fixed hypothesis.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Counts

variable (cut : FreeMonoid α → Option Bool)

/-- The probes whose start the cut leaves undecided. -/
def startSet (T : DTree α) (k : ℕ) : Set (FreeMonoid α) :=
  {x | ∃ b, T.sift cut (prefixOf x k) = .inr b}

theorem bracketAt_ne_start (agrees : ℕ → Option Bool) (ps : List (List Bool)) (fuel lo hi : ℕ)
    (w : FreeMonoid α) : bracketAt agrees ps fuel lo hi ≠ .startUndecided w := by
  intro h
  have := bracketAt_isSearch (α := α) agrees ps fuel lo hi
  rw [h] at this
  exact this

theorem walkCheckBy_start {T : DTree α} {edges : Edges α} {k : ℕ} {x w : FreeMonoid α}
    (h : walkCheckBy cut T edges k x = .inl (.startUndecided w)) :
    kWalkBy cut T edges k x = .anchor := by
  unfold walkCheckBy at h
  split at h
  · assumption
  · split at h
    · simp at h
    · split at h
      · simp at h
      · split_ifs at h <;> simp at h
  · split at h
    · simp at h
    · split_ifs at h <;> simp at h

theorem probeBy_start {T : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α} :
    (∃ w, probeBy cut T edges k x = .startUndecided w) ↔ x ∈ startSet cut T k := by
  constructor
  · rintro ⟨w, h⟩
    unfold probeBy at h
    rcases hw : walkCheckBy cut T edges k x with o | d <;> rw [hw] at h
    · simp only [Sum.elim_inl, id] at h
      subst h
      have := walkCheckBy_start cut hw
      unfold kWalkBy at this
      rcases hk : T.sift cut (prefixOf x k) with p | b <;> rw [hk] at this
      · simp only [] at this
        split at this <;> simp at this
      · exact ⟨b, hk⟩
    · exact absurd h (bracketAt_ne_start _ _ _ _ _ w)
  · rintro ⟨b, hb⟩
    refine ⟨prefixOf x k, ?_⟩
    have : kWalkBy cut T edges k x = .anchor := by unfold kWalkBy; rw [hb]
    simp [probeBy, walkCheckBy, this]

open scoped Classical in
/-- With the version unchanged, a probe adds one probe, its start if undecided, and the reads it
charges each edge. -/
theorem tallyPre_counts (C : TallyCfg) (s : TState α) (x : FreeMonoid α)
    (hv : (tallyPre C cut s x).version = s.version) :
    (tallyPre C cut s x).n = s.n + 1
      ∧ (tallyPre C cut s x).starts = s.starts + (if x ∈ startSet cut s.tree C.k then 1 else 0)
      ∧ (∀ p c, (tallyPre C cut s x).reads p c
          = s.reads p c + edgeReadsBy cut s.tree s.edges C.k x (p, c))
      ∧ ∀ p c, (tallyPre C cut s x).und p c
          = s.und p c + edgeUndecBy cut s.tree s.edges C.k x (p, c) fun _ => True := by
  classical
  have hns : ¬ (∃ w, probeBy cut s.tree s.edges C.k x = .startUndecided w) →
      x ∉ startSet cut s.tree C.k := fun h => (probeBy_start cut).not.1 h
  unfold tallyPre at hv ⊢
  split
  · rename_i u ho
    have hx := hns (by rw [ho]; simp)
    rw [ho] at hv
    simp only [] at hv
    split at hv
    · split at hv
      · simp [TState.setEdge] at hv
      · simp_all [TState.charge]
    · simp_all [TState.charge]
  · rename_i ps fd ho
    have hx := hns (by rw [ho]; simp)
    split <;> simp [hx, TState.charge, TState.addRec]
  · rename_i j ho
    have hx := hns (by rw [ho]; simp)
    simp [hx, TState.charge]
  · rename_i j ho
    have hx := hns (by rw [ho]; simp)
    simp [hx, TState.charge]
  · rename_i w ho
    have hx : x ∈ startSet cut s.tree C.k := (probeBy_start cut).1 ⟨w, ho⟩
    simp [hx, TState.charge]
  · rename_i _ _ _ _ _ hst
    have hx := hns (fun ⟨w, h⟩ => hst w h)
    simp [hx, TState.charge]

end Counts

section Stretch

variable (C : TallyCfg) (cut : FreeMonoid α → Option Bool) (D : Measure (FreeMonoid α))

theorem settle_inr {m Lmax : ℕ} : ∀ (fuel : ℕ) (s s' : TState α) (e : TEnd α),
    settleEdges cut m Lmax fuel s = .inr (e, s') → e = .tooBig ∨ e = .stuck
  | 0, s, s', e, h => by
    unfold settleEdges at h
    split_ifs at h
    simp only [Sum.inr.injEq, Prod.mk.injEq] at h
    exact .inr h.1.symm
  | fuel + 1, s, s', e, h => by
    unfold settleEdges at h
    split_ifs at h
    · simp only [Sum.inr.injEq, Prod.mk.injEq] at h
      exact .inl h.1.symm
    · exact settle_inr fuel _ s' e h

/-- A step that keeps the version is the probe's outcome counted, the edges already settled. -/
theorem tallyStep_inl_same {s s' : TState α} {x : FreeMonoid α}
    (h : tallyStep C cut s x = .inl s') (hv : s'.version = s.version) :
    s' = tallyPre C cut s x ∧ (tallyPre C cut s x).version = s.version := by
  unfold tallyStep at h
  simp only [] at h
  split at h
  · cases h
  obtain ⟨hle, heq⟩ := settle_version cut C.m C.Lmax C.fuel _ s' h
  have hge := tallyPre_version_ge C cut s x
  have hs' : s' = tallyPre C cut s x := heq (by omega)
  exact ⟨hs', by rw [← hs', hv]⟩

/-- A harvest or a success is the stretch's tests firing on the probe's outcome counted. -/
theorem tallyStep_inr_look {s s₁ : TState α} {x : FreeMonoid α} {e : TEnd α}
    (h : tallyStep C cut s x = .inr (e, s₁)) (he : e ≠ .tooBig ∧ e ≠ .stuck) :
    s₁ = tallyPre C cut s x ∧ tallyLook C s₁ = some e := by
  unfold tallyStep at h
  simp only [] at h
  split at h
  · rename_i e' hl
    simp only [Sum.inr.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨rfl, hl⟩
  · rcases settle_inr cut _ _ _ _ h with rfl | rfl
    · exact absurd rfl he.1
    · exact absurd rfl he.2

theorem tallyLook_start {s : TState α} (h : tallyLook C s = some .harvestStart) :
    C.n₀ ≤ s.n ∧ binomSfGe s.n C.θs s.starts < C.a := by
  unfold tallyLook at h
  by_cases h1 : rateSide C.θs C.a C.n₀ s.n s.starts = some true
  · unfold rateSide at h1
    split_ifs at h1 with h2 h3 <;> simp_all
  · rw [if_neg h1] at h
    split_ifs at h <;> simp at h

theorem tallyLook_harvest {s : TState α} {e : List Bool × α}
    (h : tallyLook C s = some (.harvest e)) :
    e.1 ∈ s.tree.paths ∧ C.n₀ ≤ s.n ∧ C.exc s.n ≤ (s.und e.1 e.2 : ℝ) - C.θe * s.reads e.1 e.2 := by
  unfold tallyLook at h
  by_cases h1 : rateSide C.θs C.a C.n₀ s.n s.starts = some true
  · rw [if_pos h1] at h; simp at h
  · rw [if_neg h1] at h
    by_cases h2 : ∃ e : List Bool × α, e.1 ∈ s.tree.paths ∧ C.n₀ ≤ s.n
        ∧ C.exc s.n ≤ (s.und e.1 e.2 : ℝ) - C.θe * s.reads e.1 e.2
    · rw [dif_pos h2] at h
      simp only [Option.some.injEq, TEnd.harvest.injEq] at h
      rw [← h]
      exact h2.choose_spec
    · rw [dif_neg h2] at h
      split_ifs at h <;> simp at h

open scoped Classical in
/-- From `s`, the stretch keeps its hypothesis until a harvest fires that it does not call for:
the start left undecided at most `θs` of the time, or the edge's undecided reads at most `θe` of
its reads on average. -/
noncomputable def StretchBad : TState α → (T : ℕ) → (Fin T → FreeMonoid α) → Prop
  | _, 0, _ => False
  | s, _ + 1, xs =>
    match tallyStep C cut s (xs 0) with
    | .inl s' => tallyKey s' = tallyKey s ∧ StretchBad s' _ (Fin.tail xs)
    | .inr (.harvestStart, s₁) =>
      tallyKey s₁ = tallyKey s ∧ D.real (startSet cut s.tree C.k) ≤ C.θs
    | .inr (.harvest e, s₁) => tallyKey s₁ = tallyKey s
      ∧ ∫ x, (edgeUndecBy cut s.tree s.edges C.k x e (fun _ => True) : ℝ) ∂D
        ≤ C.θe * ∫ x, (edgeReadsBy cut s.tree s.edges C.k x e : ℝ) ∂D
    | .inr _ => False

open scoped Classical in
/-- A stretch's harvest that its hypothesis does not call for is a test firing on the fresh
draws' counts, added to `s`'s. -/
theorem stretchBad_event : ∀ (T : ℕ) (s : TState α) (xs : Fin T → FreeMonoid α),
    StretchBad C cut D s T xs → ∃ j < T,
      (D.real (startSet cut s.tree C.k) ≤ C.θs ∧ C.n₀ ≤ s.n + j + 1
        ∧ binomSfGe (s.n + j + 1) C.θs (s.starts + ∑ i : Fin T,
            if (i : ℕ) ≤ j ∧ xs i ∈ startSet cut s.tree C.k then 1 else 0) < C.a)
      ∨ ∃ e : List Bool × α, e.1 ∈ s.tree.paths
        ∧ ∫ x, (edgeUndecBy cut s.tree s.edges C.k x e (fun _ => True) : ℝ) ∂D
          ≤ C.θe * ∫ x, (edgeReadsBy cut s.tree s.edges C.k x e : ℝ) ∂D
        ∧ C.n₀ ≤ s.n + j + 1
        ∧ C.exc (s.n + j + 1) ≤ ((s.und e.1 e.2 : ℝ) + ∑ i : Fin T, if (i : ℕ) ≤ j then
            (edgeUndecBy cut s.tree s.edges C.k (xs i) e (fun _ => True) : ℝ) else 0)
          - C.θe * ((s.reads e.1 e.2 : ℝ) + ∑ i : Fin T, if (i : ℕ) ≤ j then
            (edgeReadsBy cut s.tree s.edges C.k (xs i) e : ℝ) else 0)
  | 0, _, _, h => h.elim
  | T + 1, s, xs, h => by
    unfold StretchBad at h
    rcases hs : tallyStep C cut s (xs 0) with s' | ⟨e, s₁⟩ <;> rw [hs] at h
    · obtain ⟨hk, h⟩ := h
      simp only [tallyKey, Prod.mk.injEq] at hk
      obtain ⟨ht, hed, hv⟩ := hk
      obtain ⟨rfl, hv'⟩ := tallyStep_inl_same C cut hs hv
      obtain ⟨hn, hst, hrd, hud⟩ := tallyPre_counts cut C s (xs 0) hv'
      obtain ⟨j, hj, hev⟩ := stretchBad_event T _ _ h
      refine ⟨j + 1, by omega, ?_⟩
      rw [ht, hed, hn, hst] at hev
      rcases hev with ⟨h1, h2, h3⟩ | ⟨e, h1, h2, h3, h4⟩
      · refine .inl ⟨h1, by omega, ?_⟩
        have hsum : (∑ i : Fin (T + 1),
            if (i : ℕ) ≤ j + 1 ∧ xs i ∈ startSet cut s.tree C.k then 1 else 0)
            = (if xs 0 ∈ startSet cut s.tree C.k then 1 else 0) + ∑ i : Fin T,
              if (i : ℕ) ≤ j ∧ Fin.tail xs i ∈ startSet cut s.tree C.k then 1 else 0 := by
          rw [Fin.sum_univ_succ]
          congr 1
          · exact if_congr (by simp) rfl rfl
          · exact Finset.sum_congr rfl fun i _ => if_congr (by simp [Fin.tail]) rfl rfl
        rw [hsum, show s.n + (j + 1) + 1 = s.n + 1 + j + 1 by omega, ← add_assoc]
        exact h3
      · refine .inr ⟨e, h1, h2, by omega, ?_⟩
        have hsU : (∑ i : Fin (T + 1), if (i : ℕ) ≤ j + 1 then
            (edgeUndecBy cut s.tree s.edges C.k (xs i) e (fun _ => True) : ℝ) else 0)
            = (edgeUndecBy cut s.tree s.edges C.k (xs 0) e (fun _ => True) : ℝ)
              + ∑ i : Fin T, if (i : ℕ) ≤ j then
                (edgeUndecBy cut s.tree s.edges C.k (Fin.tail xs i) e (fun _ => True) : ℝ)
                else 0 := by
          rw [Fin.sum_univ_succ]
          congr 1
          exact Finset.sum_congr rfl fun i _ => if_congr (by simp) rfl rfl
        have hsN : (∑ i : Fin (T + 1), if (i : ℕ) ≤ j + 1 then
            (edgeReadsBy cut s.tree s.edges C.k (xs i) e : ℝ) else 0)
            = (edgeReadsBy cut s.tree s.edges C.k (xs 0) e : ℝ)
              + ∑ i : Fin T, if (i : ℕ) ≤ j then
                (edgeReadsBy cut s.tree s.edges C.k (Fin.tail xs i) e : ℝ) else 0 := by
          rw [Fin.sum_univ_succ]
          congr 1
          exact Finset.sum_congr rfl fun i _ => if_congr (by simp) rfl rfl
        rw [hsU, hsN, show s.n + (j + 1) + 1 = s.n + 1 + j + 1 by omega]
        rw [hrd, hud] at h4
        push_cast at h4
        linarith
    · rcases e with _ | _ | e | _ | _
      · exact h.elim
      · obtain ⟨hk, hrate⟩ := h
        obtain ⟨rfl, hl⟩ := tallyStep_inr_look C cut hs (by simp)
        simp only [tallyKey, Prod.mk.injEq] at hk
        obtain ⟨ht, hed, hv⟩ := hk
        obtain ⟨hn, hst, -, -⟩ := tallyPre_counts cut C s (xs 0) hv
        obtain ⟨h1, h2⟩ := tallyLook_start C hl
        refine ⟨0, by omega, .inl ⟨hrate, by omega, ?_⟩⟩
        rw [hn, hst] at h2
        convert h2 using 2
        rw [Fin.sum_univ_succ]
        simp [Fin.succ_ne_zero]
      · obtain ⟨hk, hun⟩ := h
        obtain ⟨rfl, hl⟩ := tallyStep_inr_look C cut hs (by simp)
        simp only [tallyKey, Prod.mk.injEq] at hk
        obtain ⟨ht, hed, hv⟩ := hk
        obtain ⟨hn, -, hrd, hud⟩ := tallyPre_counts cut C s (xs 0) hv
        obtain ⟨h1, h2, h3⟩ := tallyLook_harvest C hl
        refine ⟨0, by omega, .inr ⟨e, ht ▸ h1, hun, by omega, ?_⟩⟩
        rw [hn, hrd, hud] at h3
        rw [Fin.sum_univ_succ, Fin.sum_univ_succ]
        simp only [Fin.val_zero, le_refl, if_true, Fin.val_succ, Nat.add_one_le_iff,
          Nat.not_lt_zero, if_false, Finset.sum_const_zero, add_zero]
        push_cast at h3
        simpa using h3
      · exact h.elim
      · exact h.elim

variable [IsProbabilityMeasure D]

open scoped Classical in
/-- From a fresh state within `Lmax` leaves, a stretch's harvest that its hypothesis does not call
for has chance at most `T (Lmax |Σ| + 1) a`: at each of the `T` looks, the start's test and each
edge's test fire with chance at most `a`. -/
theorem stretchBad_le (L : ℕ) (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hθs : C.θs ≤ 1)
    (hθe : 0 < C.θe) (hLmax : 1 ≤ C.Lmax) (T : ℕ)
    (hexc : ∀ j, 1 ≤ j → j ≤ T → 0 ≤ C.exc j ∧ edgeLevel C ((L + 1) * C.Lmax) j ≤ C.a)
    (s : TState α) (hs : s.Fresh) (hL : s.tree.paths.length ≤ C.Lmax) :
    (Measure.pi fun _ : Fin T => D) {xs | StretchBad C cut D s T xs}
      ≤ T * (C.Lmax * Fintype.card α + 1) * ENNReal.ofReal C.a := by
  obtain ⟨hn, hst, -, hrd, hud⟩ := hs
  set Sst : Set (FreeMonoid α) := startSet cut s.tree C.k
  set U : List Bool × α → FreeMonoid α → ℕ :=
    fun e x => edgeUndecBy cut s.tree s.edges C.k x e (fun _ => True)
  set N : List Bool × α → FreeMonoid α → ℕ := fun e x => edgeReadsBy cut s.tree s.edges C.k x e
  set Est : ℕ → Set (Fin T → FreeMonoid α) := fun j =>
    if D.real Sst ≤ C.θs then
      {xs | binomSfGe (j + 1) C.θs (∑ i : Fin T, if (i : ℕ) ≤ j ∧ xs i ∈ Sst then 1 else 0) < C.a}
    else ∅
  set Eed : List Bool × α → ℕ → Set (Fin T → FreeMonoid α) := fun e j =>
    if ∫ x, (U e x : ℝ) ∂D ≤ C.θe * ∫ x, (N e x : ℝ) ∂D then
      {xs | C.exc (j + 1) ≤ (∑ i : Fin T, if (i : ℕ) ≤ j then (U e (xs i) : ℝ) else 0)
        - C.θe * ∑ i : Fin T, if (i : ℕ) ≤ j then (N e (xs i) : ℝ) else 0}
    else ∅
  have hsub : {xs | StretchBad C cut D s T xs}
      ⊆ ⋃ j ∈ Finset.range T, (Est j ∪ ⋃ e ∈ s.keys, Eed e j) := by
    intro xs hxs
    obtain ⟨j, hj, hev⟩ := stretchBad_event C cut D T s xs hxs
    simp only [Set.mem_iUnion, Finset.mem_range, Set.mem_union]
    refine ⟨j, hj, ?_⟩
    rcases hev with ⟨h1, -, h3⟩ | ⟨e, h1, h2, -, h4⟩
    · left
      show xs ∈ (if D.real Sst ≤ C.θs then _ else ∅)
      rw [if_pos h1]
      rw [hn, hst] at h3
      simpa using h3
    · right
      refine ⟨e, by simp [TState.keys, h1], ?_⟩
      show xs ∈ (if ∫ x, (U e x : ℝ) ∂D ≤ C.θe * ∫ x, (N e x : ℝ) ∂D then _ else ∅)
      rw [if_pos h2]
      rw [hn, hrd, hud] at h4
      simpa using h4
  have hR : (0 : ℝ) < ((L + 1) * C.Lmax : ℕ) := by
    have : 0 < (L + 1) * C.Lmax := Nat.mul_pos (Nat.succ_pos L) hLmax
    exact_mod_cast this
  have hbd : ∀ e, ∀ᵐ x ∂D, (U e x : ℝ) ≤ ((L + 1) * C.Lmax : ℕ)
      ∧ (N e x : ℝ) ≤ ((L + 1) * C.Lmax : ℕ) := by
    intro e
    filter_upwards [hlen] with x hx
    have hU := edgeUndecBy_le cut (t := s.tree) (edges := s.edges) (k := C.k) (x := x) e
      (fun _ => True)
    have hN := edgeReadsBy_le cut (t := s.tree) (edges := s.edges) (k := C.k) (x := x) e
    constructor
    · have : U e x ≤ (L + 1) * C.Lmax := by
        simp only [U]; nlinarith
      exact_mod_cast this
    · have : N e x ≤ (L + 1) * C.Lmax := by
        simp only [N]
        calc _ ≤ x.toList.length * s.tree.paths.length := hN
          _ ≤ (L + 1) * C.Lmax := Nat.mul_le_mul (by omega) hL
      exact_mod_cast this
  have hEst : ∀ j ∈ Finset.range T, (Measure.pi fun _ : Fin T => D) (Est j)
      ≤ ENNReal.ofReal C.a := by
    intro j hj
    simp only [Est]
    split_ifs with h
    · exact binom_test_le D Sst hθs h (Finset.mem_range.1 hj)
    · simp
  have hEed : ∀ j ∈ Finset.range T, ∀ e, (Measure.pi fun _ : Fin T => D) (Eed e j)
      ≤ ENNReal.ofReal C.a := by
    intro j hj e
    simp only [Eed]
    split_ifs with h
    · have hj' := Finset.mem_range.1 hj
      obtain ⟨hc, hlev⟩ := hexc (j + 1) (by omega) (by omega)
      refine (excess_test_le D (U e) (N e) hθe hR hc (hbd e) h hj').trans
        (ENNReal.ofReal_le_ofReal ?_)
      refine le_trans (le_of_eq ?_) hlev
      simp only [edgeLevel]
    · simp
  refine (measure_mono hsub).trans ((measure_biUnion_finset_le _ _).trans ?_)
  have hk : s.keys.card ≤ C.Lmax * Fintype.card α :=
    (keys_card_le s).trans (Nat.mul_le_mul_right _ hL)
  calc ∑ j ∈ Finset.range T, (Measure.pi fun _ : Fin T => D) (Est j ∪ ⋃ e ∈ s.keys, Eed e j)
      ≤ ∑ j ∈ Finset.range T, (ENNReal.ofReal C.a + s.keys.card * ENNReal.ofReal C.a) := by
        refine Finset.sum_le_sum fun j hj => (measure_union_le _ _).trans (add_le_add (hEst j hj)
          ((measure_biUnion_finset_le _ _).trans ?_))
        calc ∑ e ∈ s.keys, (Measure.pi fun _ : Fin T => D) (Eed e j)
            ≤ ∑ _e ∈ s.keys, ENNReal.ofReal C.a := Finset.sum_le_sum fun e _ => hEed j hj e
          _ = s.keys.card * ENNReal.ofReal C.a := by rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ T * (C.Lmax * Fintype.card α + 1) * ENNReal.ofReal C.a := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        rw [mul_assoc]
        gcongr
        calc ENNReal.ofReal C.a + s.keys.card * ENNReal.ofReal C.a
            = (s.keys.card + 1) * ENNReal.ofReal C.a := by ring
          _ ≤ (C.Lmax * Fintype.card α + 1) * ENNReal.ofReal C.a := by
            gcongr; exact_mod_cast hk

end Stretch

end OrthoDFA
