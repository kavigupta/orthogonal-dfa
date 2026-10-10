import OrthoDFA.Proofs.TallyInv

/-!
# Heavy places harvest in time

A record that is not true reads at a light place, which `TallyE.spurious` bounds, or at a heavy
one, which arms its hypothesis. A stretch at an armed hypothesis outlasts `J` probes only where
the heavy place's test misses its drift: Hoeffding's lower tail at an edge, the binomial at the
start. So the probes at armed hypotheses within `J` of their stretch's start are a budget of
`J` per version, and the heavy records are made within it.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Budget

variable (cut : FreeMonoid α → Option Bool) (m Lmax : ℕ)

/-- Settling within `Lmax` leaves lowers `pot` by the versions it adds, at any state. -/
theorem settle_pot : ∀ (fuel : ℕ) (s s' : TState α),
    settleEdges cut m Lmax fuel s = .inl s' →
    s'.pot m Lmax + (s'.version - s.version) ≤ s.pot m Lmax
  | 0, s, s', h => by
    unfold settleEdges at h
    split_ifs at h
    simp only [Sum.inl.injEq] at h
    subst h; simp
  | fuel + 1, s, s', h => by
    unfold settleEdges at h
    split_ifs at h with hv hL
    · have h1 := settle_pot fuel _ s' h
      have h2 := pot_fixEdge m Lmax cut s hv.choose_spec.choose_spec.choose_spec (not_lt.1 hL)
      have h3 := fixEdge_version cut m s hv.choose hv.choose_spec.choose
        hv.choose_spec.choose_spec.choose
      have h4 := (settle_version cut m Lmax fuel _ s' h).1
      omega
    · simp only [Sum.inl.injEq] at h
      subst h; simp

variable (C : TallyCfg)

/-- A step lowers `pot` by the versions it adds, at any state. -/
theorem step_pot {s s' : TState α} {x : FreeMonoid α} (h : tallyStep C cut s x = .inl s') :
    s'.pot C.m C.Lmax + (s'.version - s.version) ≤ s.pot C.m C.Lmax ∧ s.version ≤ s'.version := by
  unfold tallyStep at h
  simp only [] at h
  split at h
  · cases h
  · have h1 := settle_pot cut C.m C.Lmax C.fuel _ s' h
    have h2 := (settle_version cut C.m C.Lmax C.fuel _ s' h).1
    obtain ⟨-, h3, h4, -⟩ := tallyPre_spec C.Lmax cut C s x
    omega

variable (D : Measure (FreeMonoid α)) (η ηs : ℝ) (J : ℕ)

open scoped Classical in
/-- `J` per version to come, and what is left of `J` in the stretch where it is armed. -/
noncomputable def heavyBudget (s : TState α) : ℕ :=
  s.pot C.m C.Lmax * J + if Armed C D cut η ηs s.tree s.edges ∧ s.n < J then J - s.n else 0

open scoped Classical in
/-- A step from an armed state within `J` of its stretch's start uses up a unit of the budget,
and no step raises it. -/
theorem budget_step {s s' : TState α} {x : FreeMonoid α} (h : tallyStep' C cut s x = some s') :
    heavyBudget cut C D η ηs J s'
        + (if Armed C D cut η ηs s.tree s.edges ∧ s.n < J then 1 else 0)
      ≤ heavyBudget cut C D η ηs J s := by
  have hst : tallyStep C cut s x = .inl s' := by
    simpa only [tallyStep', Sum.getLeft?_eq_some_iff] using h
  obtain ⟨hp, hle⟩ := step_pot cut C hst
  unfold heavyBudget
  by_cases hver : s'.version = s.version
  · obtain ⟨rfl, hv'⟩ := tallyStep_inl_same C cut hst hver
    obtain ⟨hn, -, -, -⟩ := tallyPre_counts cut C s x hv'
    obtain ⟨ht, he, -⟩ := tallyPre_same C cut s x hv'
    rw [hn, ht, he]
    have hpm := Nat.mul_le_mul_right J (show (tallyPre C cut s x).pot C.m C.Lmax
      ≤ s.pot C.m C.Lmax by omega)
    by_cases hA : Armed C D cut η ηs s.tree s.edges
    · simp only [hA, true_and]
      split_ifs <;> omega
    · simp only [hA, false_and, if_false]
      omega
  · have hpm := Nat.mul_le_mul_right J (show s'.pot C.m C.Lmax + 1 ≤ s.pot C.m C.Lmax by omega)
    rw [add_mul, one_mul] at hpm
    split_ifs <;> omega

end Budget

section Looks

variable {X S H : Type*} (step : S → X → Option S) (key : S → H)

theorem stays_le_T : ∀ (n T : ℕ) (s : S) (xs : Fin T → X), Stays step key n s T xs → n ≤ T
  | 0, _, _, _, _ => Nat.zero_le _
  | _ + 1, 0, _, _, h => h.elim
  | n + 1, T + 1, _, xs, h => by
    obtain ⟨s', -, -, h⟩ := h
    have := stays_le_T n T s' (Fin.tail xs) h
    omega

variable (C : TallyCfg) (cut : FreeMonoid α → Option Bool)

theorem tallyStep_inl_look {s s' : TState α} {x : FreeMonoid α}
    (h : tallyStep C cut s x = .inl s') : tallyLook C (tallyPre C cut s x) = none := by
  unfold tallyStep at h
  simp only [] at h
  rcases hl : tallyLook C (tallyPre C cut s x) with _ | e
  · rfl
  · rw [hl] at h; cases h

theorem tallyLook_none {s : TState α} (h : tallyLook C s = none) :
    rateSide C.θs C.a C.n₀ s.n s.starts ≠ some true
      ∧ ∀ e : List Bool × α, e.1 ∈ s.tree.paths → C.n₀ ≤ s.n →
        (s.und e.1 e.2 : ℝ) - C.θe * s.reads e.1 e.2 < C.exc s.n := by
  unfold tallyLook at h
  split_ifs at h with h1 h2
  refine ⟨h1, fun e he hn => ?_⟩
  by_contra hc
  exact h2 ⟨e, he, hn, not_lt.1 hc⟩

open scoped Classical in
/-- A stretch that keeps its hypothesis over `j + 1` probes saw none of its tests fire at the
last of them, on the probes' counts added to `s`'s. -/
theorem stays_look : ∀ (j T : ℕ) (s : TState α) (xs : Fin T → FreeMonoid α),
    Stays (tallyStep' C cut) tallyKey (j + 1) s T xs →
    rateSide C.θs C.a C.n₀ (s.n + j + 1) (s.starts + ∑ i : Fin T,
        if (i : ℕ) ≤ j ∧ xs i ∈ startSet cut s.tree C.k then 1 else 0) ≠ some true
      ∧ ∀ e : List Bool × α, e.1 ∈ s.tree.paths → C.n₀ ≤ s.n + j + 1 →
        ((s.und e.1 e.2 : ℝ) + ∑ i : Fin T, if (i : ℕ) ≤ j then
            (edgeUndecBy cut s.tree s.edges C.k (xs i) e (fun _ => True) : ℝ) else 0)
          - C.θe * ((s.reads e.1 e.2 : ℝ) + ∑ i : Fin T, if (i : ℕ) ≤ j then
            (edgeReadsBy cut s.tree s.edges C.k (xs i) e : ℝ) else 0)
          < C.exc (s.n + j + 1)
  | _, 0, _, _, h => h.elim
  | j, T + 1, s, xs, h => by
    obtain ⟨s', hs', hk, h'⟩ := h
    have hst : tallyStep C cut s (xs 0) = .inl s' := by
      simpa only [tallyStep', Sum.getLeft?_eq_some_iff] using hs'
    simp only [tallyKey, Prod.mk.injEq] at hk
    obtain ⟨ht, hed, hv⟩ := hk
    obtain ⟨rfl, hv'⟩ := tallyStep_inl_same C cut hst hv
    obtain ⟨hn, hst', hrd, hud⟩ := tallyPre_counts cut C s (xs 0) hv'
    have hsS : ∀ j', (∑ i : Fin (T + 1),
        if (i : ℕ) ≤ j' + 1 ∧ xs i ∈ startSet cut s.tree C.k then 1 else 0)
        = (if xs 0 ∈ startSet cut s.tree C.k then 1 else 0) + ∑ i : Fin T,
          if (i : ℕ) ≤ j' ∧ Fin.tail xs i ∈ startSet cut s.tree C.k then 1 else 0 := by
      intro j'
      rw [Fin.sum_univ_succ]
      congr 1
      · exact if_congr (by simp) rfl rfl
      · exact Finset.sum_congr rfl fun i _ => if_congr (by simp [Fin.tail]) rfl rfl
    have hsF : ∀ (f : FreeMonoid α → ℝ) (j' : ℕ), (∑ i : Fin (T + 1),
        if (i : ℕ) ≤ j' + 1 then f (xs i) else 0)
        = f (xs 0) + ∑ i : Fin T, if (i : ℕ) ≤ j' then f (Fin.tail xs i) else 0 := by
      intro f j'
      rw [Fin.sum_univ_succ]
      congr 1
      exact Finset.sum_congr rfl fun i _ => if_congr (by simp) rfl rfl
    rcases j with _ | j
    · have hl := tallyLook_none C (tallyStep_inl_look C cut hst)
      rw [hn, hst'] at hl
      obtain ⟨h1, h2⟩ := hl
      have hs0 : (∑ i : Fin (T + 1),
          if (i : ℕ) ≤ 0 ∧ xs i ∈ startSet cut s.tree C.k then 1 else 0)
          = if xs 0 ∈ startSet cut s.tree C.k then 1 else 0 := by
        rw [Fin.sum_univ_succ]
        simp
      have hf0 : ∀ f : FreeMonoid α → ℝ,
          (∑ i : Fin (T + 1), if (i : ℕ) ≤ 0 then f (xs i) else 0) = f (xs 0) := by
        intro f
        rw [Fin.sum_univ_succ]
        simp
      refine ⟨by rw [hs0]; exact h1, fun e he hn₀ => ?_⟩
      rw [hf0 fun x => (edgeUndecBy cut s.tree s.edges C.k x e (fun _ => True) : ℝ),
        hf0 fun x => (edgeReadsBy cut s.tree s.edges C.k x e : ℝ)]
      have := h2 e (by rw [tallyPre_same C cut s (xs 0) hv' |>.1]; exact he) hn₀
      rw [hrd, hud] at this
      push_cast at this
      linarith
    · obtain ⟨h1, h2⟩ := stays_look j T _ (Fin.tail xs) h'
      rw [ht, hn, hst'] at h1
      rw [ht, hed, hn] at h2
      refine ⟨?_, fun e he hn₀ => ?_⟩
      · rw [hsS, show s.n + (j + 1) + 1 = s.n + 1 + j + 1 by omega, ← add_assoc]
        exact h1
      · rw [hsF (fun x => (edgeUndecBy cut s.tree s.edges C.k x e (fun _ => True) : ℝ)),
          hsF (fun x => (edgeReadsBy cut s.tree s.edges C.k x e : ℝ)),
          show s.n + (j + 1) + 1 = s.n + 1 + j + 1 by omega]
        have := h2 e he (by omega)
        rw [hrd, hud] at this
        push_cast at this
        linarith

variable (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (η ηs : ℝ)

open scoped Classical in
/-- From a fresh state within `Lmax` leaves, an armed hypothesis outlasts `J` probes with chance at
most `heavyLevel` at a heavy edge, or the start's `P(Bin(J, θs + ηs) < hS)`. -/
theorem stretchLong_le (L : ℕ) (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hθe : 0 < C.θe)
    (hLmax : 1 ≤ C.Lmax) {J hS : ℕ} (hJ : C.n₀ ≤ J) (hn₀ : 1 ≤ C.n₀)
    (hexcJ : C.exc J ≤ J * η) (hηs : 0 ≤ ηs) (hθs0 : 0 ≤ C.θs) (hθs1 : C.θs + ηs ≤ 1)
    (hSa : binomSfGe J C.θs hS < C.a) (T : ℕ) (s : TState α) (hs : s.Fresh)
    (hL : s.tree.paths.length ≤ C.Lmax) :
    (Measure.pi fun _ : Fin T => D)
        {xs | Armed C D cut η ηs s.tree s.edges ∧ Stays (tallyStep' C cut) tallyKey J s T xs}
      ≤ ENNReal.ofReal (heavyLevel C ((L + 1) * C.Lmax) J η
          + (1 - binomSfGe J (C.θs + ηs) hS)) := by
  obtain ⟨n0, st0, -, hrd0, hud0⟩ := hs
  obtain ⟨j, rfl⟩ : ∃ j, J = j + 1 := ⟨J - 1, by omega⟩
  have hp0 : 0 ≤ C.θs + ηs := by linarith
  have hlev : 0 ≤ heavyLevel C ((L + 1) * C.Lmax) (j + 1) η := (Real.exp_pos _).le
  have hsf : 0 ≤ 1 - binomSfGe (j + 1) (C.θs + ηs) hS := by
    have := binomSfGe_antitone' hp0 hθs1 (n := j + 1) (Nat.zero_le hS)
    rw [binomSfGe_zero_right] at this
    linarith
  by_cases hT : j < T
  swap
  · have : {xs : Fin T → FreeMonoid α | Armed C D cut η ηs s.tree s.edges
        ∧ Stays (tallyStep' C cut) tallyKey (j + 1) s T xs} = ∅ := by
      ext xs
      simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_and]
      intro _ h
      have := stays_le_T _ _ (j + 1) T s xs h
      omega
    rw [this, measure_empty]; exact zero_le
  set S := Finset.univ.filter fun i : Fin T => (i : ℕ) ≤ j
  have hScard : S.card = j + 1 := by
    rw [show S = (Finset.range (j + 1)).attachFin (fun i hi => by
      simp only [Finset.mem_range] at hi; omega) from by
        ext i; simp [S, Finset.mem_attachFin]]
    simp
  by_cases harm : Armed C D cut η ηs s.tree s.edges
  swap
  · have : {xs : Fin T → FreeMonoid α | Armed C D cut η ηs s.tree s.edges
        ∧ Stays (tallyStep' C cut) tallyKey (j + 1) s T xs} = ∅ := by
      ext xs; simp [harm]
    rw [this, measure_empty]; exact zero_le
  rcases harm with hstart | ⟨e, hpe, hmean⟩
  · set Sst := startSet cut s.tree C.k
    have hsub : {xs : Fin T → FreeMonoid α | Armed C D cut η ηs s.tree s.edges
        ∧ Stays (tallyStep' C cut) tallyKey (j + 1) s T xs}
        ⊆ {xs | hS ≤ (S.filter fun i => xs i ∈ Sst).card}ᶜ := by
      rintro xs ⟨-, h⟩
      obtain ⟨h1, -⟩ := stays_look C cut j T s xs h
      rw [n0, st0, zero_add, zero_add] at h1
      have hcnt : (∑ i : Fin T, if (i : ℕ) ≤ j ∧ xs i ∈ Sst then 1 else 0)
          = (S.filter fun i => xs i ∈ Sst).card := by
        rw [Finset.card_eq_sum_ones, Finset.sum_filter, Finset.sum_filter]
        exact Finset.sum_congr rfl fun i _ => by split_ifs <;> simp_all
      rw [hcnt] at h1
      simp only [Set.mem_compl_iff, Set.mem_ofPred_eq, not_le]
      by_contra hge
      push_neg at hge
      apply h1
      unfold rateSide
      rw [if_pos hJ, if_pos ((binomSfGe_antitone' hθs0 (by linarith) hge).trans_lt hSa)]
    refine (measure_mono hsub).trans ?_
    rw [← ofReal_measureReal, probReal_compl_eq_one_sub (Set.to_countable _).measurableSet]
    have hreal := pi_count_ge D (fun x => x ∈ Sst) S hS
    rw [hScard] at hreal
    rw [hreal]
    refine ENNReal.ofReal_le_ofReal ?_
    have : binomSfGe (j + 1) (C.θs + ηs) hS ≤ binomSfGe (j + 1) (D.real {x | x ∈ Sst}) hS :=
      binomSfGe_mono hp0 measureReal_le_one hstart _ _
    linarith
  · have hR : (0 : ℝ) < ((L + 1) * C.Lmax : ℕ) := by
      have : 0 < (L + 1) * C.Lmax := Nat.mul_pos (Nat.succ_pos L) hLmax
      exact_mod_cast this
    have hbd : ∀ᵐ x ∂D, (edgeUndecBy cut s.tree s.edges C.k x e (fun _ => True) : ℝ)
        ≤ ((L + 1) * C.Lmax : ℕ) ∧ (edgeReadsBy cut s.tree s.edges C.k x e : ℝ)
        ≤ ((L + 1) * C.Lmax : ℕ) := by
      filter_upwards [hlen] with x hx
      have hU := edgeUndecBy_le cut (t := s.tree) (edges := s.edges) (k := C.k) (x := x) e
        (fun _ => True)
      have hN := edgeReadsBy_le cut (t := s.tree) (edges := s.edges) (k := C.k) (x := x) e
      constructor
      · have : edgeUndecBy cut s.tree s.edges C.k x e (fun _ => True) ≤ (L + 1) * C.Lmax := by
          nlinarith
        exact_mod_cast this
      · have : edgeReadsBy cut s.tree s.edges C.k x e ≤ (L + 1) * C.Lmax :=
          hN.trans (Nat.mul_le_mul (by omega) hL)
        exact_mod_cast this
    have hsub : {xs : Fin T → FreeMonoid α | Armed C D cut η ηs s.tree s.edges
        ∧ Stays (tallyStep' C cut) tallyKey (j + 1) s T xs}
        ⊆ {xs | (∑ i : Fin T, if (i : ℕ) ≤ j then
            (edgeUndecBy cut s.tree s.edges C.k (xs i) e (fun _ => True) : ℝ) else 0)
          - C.θe * (∑ i : Fin T, if (i : ℕ) ≤ j then
            (edgeReadsBy cut s.tree s.edges C.k (xs i) e : ℝ) else 0) < C.exc (j + 1)} := by
      rintro xs ⟨-, h⟩
      obtain ⟨-, h2⟩ := stays_look C cut j T s xs h
      have := h2 e hpe (by omega)
      rw [n0, hrd0, hud0] at this
      simp only [Nat.cast_zero, zero_add] at this
      exact this
    refine (measure_mono hsub).trans ((lower_test_le D
      (fun x => edgeUndecBy cut s.tree s.edges C.k x e (fun _ => True))
      (fun x => edgeReadsBy cut s.tree s.edges C.k x e) hθe.le hR hbd hmean hT
      (by exact_mod_cast hexcJ)).trans (ENNReal.ofReal_le_ofReal ?_))
    refine le_trans (le_of_eq ?_) (le_add_of_nonneg_right hsf)
    simp only [heavyLevel]

end Looks

section Split

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : TallyCfg) (D : Measure (FreeMonoid α))
  (rd : FreeMonoid α → ARU) (η ηs : ℝ) (J : ℕ)

open scoped Classical in
/-- The probes whose record is not true, with neither prefix read at a heavy place, at a genuine
hypothesis with edges into its leaves. -/
def lightSet (s : TState α) : Set (FreeMonoid α) :=
  if G.Genuine s.tree ∧ EdgesInto s.tree s.edges then
    {x | ∃ pct sp, recordBy (rdCut rd) C.k (s.tree, s.edges) x = some (pct, sp)
      ∧ ¬ G.TrueRec s.tree pct sp ∧ ¬ HeavyRec C D (rdCut rd) η ηs s.tree s.edges x sp}
  else ∅

open scoped Classical in
/-- The probes whose record is not true, at an armed genuine hypothesis with edges into its leaves
within `J` probes of its stretch's start. -/
def heavySet (s : TState α) : Set (FreeMonoid α) :=
  if G.Genuine s.tree ∧ EdgesInto s.tree s.edges ∧ Armed C D (rdCut rd) η ηs s.tree s.edges
      ∧ s.n < J then
    {x | ∃ pct sp, recordBy (rdCut rd) C.k (s.tree, s.edges) x = some (pct, sp)
      ∧ ¬ G.TrueRec s.tree pct sp}
  else ∅

theorem heavyAt_armed {cut : FreeMonoid α → Option Bool} {T : DTree α} {edges : Edges α}
    {x : FreeMonoid α} {i : ℕ} (h : HeavyAt C D cut η ηs T edges x i) :
    Armed C D cut η ηs T edges := by
  unfold HeavyAt at h
  split_ifs at h
  · exact .inl h
  · obtain ⟨e, -, he⟩ := h
    exact .inr ⟨e, he⟩

open scoped Classical in
/-- A record that is not true reads at a light place, or at a heavy one, which arms the
hypothesis. -/
theorem spur_split {s : TState α} (x : FreeMonoid α)
    (hA : Armed C D (rdCut rd) η ηs s.tree s.edges → s.n < J) :
    (if x ∈ spurSet G C rd s then 1 else 0)
      ≤ (if x ∈ lightSet G C D rd η ηs s then 1 else 0)
        + (if x ∈ heavySet G C D rd η ηs J s then 1 else 0) := by
  by_cases hx : x ∈ spurSet G C rd s
  swap
  · rw [if_neg hx]; exact Nat.zero_le _
  rw [if_pos hx]
  unfold spurSet at hx
  split_ifs at hx with hg
  · obtain ⟨pct, sp, hr, hn⟩ := hx
    by_cases hh : HeavyRec C D (rdCut rd) η ηs s.tree s.edges x sp
    · have harm : Armed C D (rdCut rd) η ηs s.tree s.edges :=
        hh.elim (heavyAt_armed C D η ηs) (heavyAt_armed C D η ηs)
      have : x ∈ heavySet G C D rd η ηs J s := by
        rw [heavySet, if_pos ⟨hg.1, hg.2, harm, hA harm⟩]
        exact ⟨pct, sp, hr, hn⟩
      rw [if_pos this]; omega
    · have : x ∈ lightSet G C D rd η ηs s := by
        rw [lightSet, if_pos hg]
        exact ⟨pct, sp, hr, hn, hh⟩
      rw [if_pos this]; omega
  · exact absurd hx (Set.notMem_empty x)

variable {q₀ c₀ ρ ρH θg θgs : ℝ}

theorem lightSet_le (hρ : 0 ≤ ρ) (hE : TallyE G D C q₀ c₀ ρ ρH θg θgs η ηs rd) (s : TState α) :
    D.real (lightSet G C D rd η ηs s) ≤ ρ := by
  unfold lightSet
  split_ifs with h
  · exact hE.spurious s.tree s.edges h.1 h.2
  · simpa using hρ

theorem heavySet_le (hρH : 0 ≤ ρH) (hE : TallyE G D C q₀ c₀ ρ ρH θg θgs η ηs rd)
    (s : TState α) : D.real (heavySet G C D rd η ηs J s) ≤ ρH := by
  unfold heavySet
  split_ifs with h
  · exact hE.spuriousAll s.tree s.edges h.1 h.2.1
  · simpa using hρH

end Split

end OrthoDFA
