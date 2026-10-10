import OrthoDFA.Proofs.TallyHarvest

/-!
# Sub-rounds end

Within a sub-round the tree is fixed, so its records that are not true accumulate at one edge
and target at most binomially, and a fake split needs `m` of them (`fake_bin`). A hypothesis
recording less than `θr` of the time and keeping over `nEnd` probes has its probes' outcomes
independent: its middle-stopping searches are rarer than `θpt'` of its searches, or its
disagreements, rarer than `εd'`, are too many for success (`keep_le`). No edge is ever left with
`m` records at a target it does not point at (`settled_step`), so every record raises its edge and
target towards `m` (`phi_step`) and a sub-round holds at most `Lmax² |Σ| m` records. Each change of
hypothesis learns or redirects an edge, lowering a potential of at most `2 Lmax |Σ|`, so a
sub-round is unfinished after `subT` probes only if some hypothesis recording less than `θr` kept
over `nEnd`, or its hypotheses recording more made too few records over `nRec` probes
(`open_le`).
-/
namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Steps

variable (cut : FreeMonoid α → Option Bool) (C : TallyCfg)

open scoped Classical in
theorem tallyPre_cases (s : TState α) (x : FreeMonoid α) :
    (tallyPre C cut s x).tree = s.tree
      ∧ (((tallyPre C cut s x).version = s.version ∧ (tallyPre C cut s x).edges = s.edges
        ∧ (tallyPre C cut s x).n = s.n + 1
        ∧ (tallyPre C cut s x).pt = s.pt ++ ptHarvBy cut s.tree s.edges C.k x
        ∧ (tallyPre C cut s x).dis
          = s.dis + if (probeBy cut s.tree s.edges C.k x).IsSearch then 1 else 0)
      ∨ ((tallyPre C cut s x).version = s.version + 1 ∧ (tallyPre C cut s x).n = 0
        ∧ (tallyPre C cut s x).dis = 0 ∧ ∃ p c e, s.edges p c = none ∧ p ∈ s.tree.paths
        ∧ ∀ q c', (tallyPre C cut s x).edges q c' = if (q, c') = (p, c) then some e
          else s.edges q c')) := by
  unfold tallyPre
  rcases ho : probeBy cut s.tree s.edges C.k x with _ | w | w | j | ⟨ps, fd⟩ | j | u <;>
    try dsimp only
  · simp [TState.charge, Outcome.IsSearch]
  · simp [TState.charge, Outcome.IsSearch]
  · simp [TState.charge, Outcome.IsSearch]
  · simp [TState.charge, Outcome.IsSearch]
  · rcases recordBy cut C.k (s.tree, s.edges) x with _ | ⟨⟨p, c, t⟩, sp⟩ <;>
      simp [TState.charge, TState.addRec, Outcome.IsSearch]
  · simp [TState.charge, Outcome.IsSearch]
  · rcases hc : x.toList[u.toList.length]? with _ | c <;> try dsimp only
    · simp [TState.charge, Outcome.IsSearch]
    rcases hp : s.tree.sift cut u with p | b <;> try dsimp only
    swap
    · simp [TState.charge, Outcome.IsSearch]
    rcases ht : s.tree.sift cut (u * FreeMonoid.of c) with t | b <;> try dsimp only
    swap
    · simp [TState.charge, Outcome.IsSearch]
    rcases he : s.edges p c with _ | e <;> try dsimp only
    · refine ⟨rfl, .inr ⟨rfl, rfl, rfl, p, c, (t, u), he, DTree.sift_mem_paths _ _ _ hp, ?_⟩⟩
      intro q c'
      exact setEdge_eq (s.charge cut C.k x) p c (t, u) q c'
    · simp [TState.charge, Outcome.IsSearch]

theorem tallyPre_recs (s : TState α) (x : FreeMonoid α) :
    (recordBy cut C.k (s.tree, s.edges) x = none ∧ (tallyPre C cut s x).recs = s.recs)
      ∨ ∃ p c t sp, recordBy cut C.k (s.tree, s.edges) x = some ((p, c, t), sp)
        ∧ (tallyPre C cut s x).recs = (s.addRec p c (sp, t)).recs := by
  have hn : ∀ o : Outcome α, probeBy cut s.tree s.edges C.k x = o → (∀ ps fd, o ≠ .edge ps fd) →
      recordBy cut C.k (s.tree, s.edges) x = none := by
    intro o ho hne
    unfold recordBy
    rw [ho]
    rcases o with _ | _ | _ | _ | ⟨ps, fd⟩ | _ | _
    all_goals first | rfl | exact absurd rfl (hne _ _)
  unfold tallyPre
  rcases ho : probeBy cut s.tree s.edges C.k x with _ | w | w | j | ⟨ps, fd⟩ | j | u <;>
    try dsimp only
  all_goals first
    | (left; exact ⟨hn _ ho (by simp), rfl⟩)
    | skip
  · rcases hr : recordBy cut C.k (s.tree, s.edges) x with _ | ⟨⟨p, c, t⟩, sp⟩ <;> try dsimp only
    · left; exact ⟨rfl, rfl⟩
    · right; exact ⟨p, c, t, sp, rfl, by simp [TState.charge, TState.addRec]⟩
  · have hr := hn _ ho (by simp)
    rcases x.toList[u.toList.length]? with _ | c <;> try dsimp only
    · left; exact ⟨hr, rfl⟩
    rcases s.tree.sift cut u with p | b <;> try dsimp only
    swap
    · left; exact ⟨hr, rfl⟩
    rcases s.tree.sift cut (u * FreeMonoid.of c) with t | b <;> rcases s.edges p c with _ | e <;>
      dsimp only
    all_goals left; exact ⟨hr, rfl⟩

/-- A fix is a new version: a redirect of the edge at the same tree, or a split. -/
theorem fixEdge_cases {s : TState α} {p : List Bool} {c : α} {t : List Bool}
    (hv : Violates C.m s p c t) :
    (fixEdge cut C.m s p c t).version = s.version + 1
      ∧ (((fixEdge cut C.m s p c t).tree = s.tree ∧ (fixEdge cut C.m s p c t).recs = s.recs
          ∧ (fixEdge cut C.m s p c t).n = 0 ∧ (fixEdge cut C.m s p c t).dis = 0
          ∧ (∀ t₀ w₀, s.edges p c = some (t₀, w₀) → s.tally p c t₀ < C.m)
          ∧ ∃ w, ∀ q c', (fixEdge cut C.m s p c t).edges q c'
            = if (q, c') = (p, c) then some (t, w) else s.edges q c')
        ∨ ∃ t₀ w₀, s.edges p c = some (t₀, w₀) ∧ C.m ≤ s.tally p c t₀
          ∧ (fixEdge cut C.m s p c t).tree
            = s.tree.splitAt (FreeMonoid.of c * s.tree.midAt (lcp t t₀)) p) := by
  unfold fixEdge
  simp only []
  rcases he0 : s.edges p c with _ | ⟨t₀, w₀⟩
  · exact ⟨rfl, .inl ⟨rfl, rfl, rfl, rfl, fun _ _ h => by simp at h, _,
      fun q c' => setEdge_eq s p c _ q c'⟩⟩
  · simp only []
    split_ifs with hm
    · exact ⟨rfl, .inr ⟨t₀, w₀, rfl, hm, rfl⟩⟩
    · refine ⟨rfl, .inl ⟨rfl, rfl, rfl, rfl, fun t' w' h => ?_, _,
        fun q c' => setEdge_eq s p c _ q c'⟩⟩
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      omega

open scoped Classical in
theorem settleOne_cases {s₁ : TState α} {r : TState α ⊕ (TEnd α × TState α)}
    (h : settleOne cut C.m C.Lmax s₁ = r) :
    (r = .inl s₁ ∧ ∀ p c t, ¬ Violates C.m s₁ p c t)
      ∨ ∃ p c t, Violates C.m s₁ p c t ∧ (r = .inl (fixEdge cut C.m s₁ p c t)
        ∨ (r = .inr (.tooBig, fixEdge cut C.m s₁ p c t)
          ∧ C.Lmax < (fixEdge cut C.m s₁ p c t).tree.paths.length)) := by
  subst h
  unfold settleOne
  split_ifs with hv hL
  · exact .inr ⟨_, _, _, hv.choose_spec.choose_spec.choose_spec, .inr ⟨rfl, hL⟩⟩
  · exact .inr ⟨_, _, _, hv.choose_spec.choose_spec.choose_spec, .inl rfl⟩
  · push_neg at hv
    exact .inl ⟨rfl, hv⟩

theorem tallyStep_cases {s : TState α} {x : FreeMonoid α} {r : TState α ⊕ (TEnd α × TState α)}
    (h : tallyStep C cut s x = r) :
    (∃ e, tallyLook C (tallyPre C cut s x) = some e ∧ r = .inr (e, tallyPre C cut s x))
      ∨ (tallyLook C (tallyPre C cut s x) = none ∧ settleOne cut C.m C.Lmax (tallyPre C cut s x)
        = r) := by
  subst h
  unfold tallyStep
  simp only []
  rcases hl : tallyLook C (tallyPre C cut s x) with _ | e
  · exact .inr ⟨rfl, rfl⟩
  · exact .inl ⟨e, rfl, rfl⟩

/-- A step that keeps the version keeps the hypothesis: it is the probe's charge, its tests do not
fire and no edge is violated. -/
theorem tallyStep_same {s s' : TState α} {x : FreeMonoid α} (h : tallyStep C cut s x = .inl s')
    (hv : s'.version = s.version) :
    s' = tallyPre C cut s x ∧ tallyLook C (tallyPre C cut s x) = none
      ∧ (∀ p c t, ¬ Violates C.m (tallyPre C cut s x) p c t)
      ∧ (tallyPre C cut s x).version = s.version := by
  have hver : s.version ≤ (tallyPre C cut s x).version := by
    rcases (tallyPre_cases cut C s x).2 with h' | h' <;> omega
  rcases tallyStep_cases cut C h with ⟨e, -, he⟩ | ⟨hl, hs⟩
  · cases he
  rcases settleOne_cases cut C hs with ⟨he, hnv⟩ | ⟨p, c, t, hvi, he | ⟨he, -⟩⟩
  · cases he
    exact ⟨rfl, hl, hnv, hv⟩
  · cases he
    have := (fixEdge_cases cut C hvi).1
    omega
  · cases he

/-- A step past `Lmax` leaves splits a leaf. -/
theorem tallyStep_tooBig {s s' : TState α} {x : FreeMonoid α}
    (h : tallyStep C cut s x = .inr (.tooBig, s')) :
    C.Lmax < s'.tree.paths.length ∧ s'.tree.paths.length ≤ s.tree.paths.length + 1 := by
  have htr := (tallyPre_cases cut C s x).1
  rcases tallyStep_cases cut C h with ⟨e, hl, he⟩ | ⟨hl, hs⟩
  · simp only [Sum.inr.injEq, Prod.mk.injEq] at he
    obtain ⟨rfl, rfl⟩ := he
    unfold tallyLook at hl
    split_ifs at hl <;> simp at hl
  rcases settleOne_cases cut C hs with ⟨he, -⟩ | ⟨p, c, t, hvi, he | ⟨he, hL⟩⟩
  · cases he
  · cases he
  · simp only [Sum.inr.injEq, Prod.mk.injEq] at he
    obtain ⟨-, rfl⟩ := he
    refine ⟨hL, ?_⟩
    rcases (fixEdge_cases cut C hvi).2 with ⟨ht, -⟩ | ⟨t₀, w₀, -, -, ht⟩
    · rw [ht, htr]; omega
    · rw [ht, DTree.splitAt_paths_length _ _ _ hvi.1, htr]

end Steps

section Fake

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : TallyCfg)
  (cut : FreeMonoid α → Option Bool) (S : ℕ)

/-- The probes whose record at `pct` is not true. -/
def untrueAt (s : TState α) (pct : List Bool × α × List Bool) : Set (FreeMonoid α) :=
  {x | ∃ sp, recordBy cut C.k (s.tree, s.edges) x = some (pct, sp) ∧ ¬ G.TrueRec s.tree pct sp}

open scoped Classical in
/-- The records at `pct` that are not true. -/
noncomputable def badRecs (s : TState α) (pct : List Bool × α × List Bool) : ℕ :=
  ((s.recs pct.1 pct.2.1).filter fun r => r.2 = pct.2.2 ∧ ¬ G.TrueRec s.tree pct r.1).length

open scoped Classical in
/-- Within `j` probes from `s`, at trees of the class, the probes whose record at `pct` is not
true reach `k` before the tree changes. -/
def EvF (pct : List Bool × α × List Bool) :
    TState α → ℕ → ℕ → (T : ℕ) → (Fin T → FreeMonoid α) → Prop
  | _, _, 0, _, _ => True
  | _, 0, _ + 1, _, _ => False
  | _, _ + 1, _ + 1, 0, _ => False
  | s, j + 1, k + 1, T + 1, xs => G.InClass S s.tree ∧ EdgesInto s.tree s.edges ∧
    match tallyStep C cut s (xs 0) with
    | .inl s' => (if xs 0 ∈ untrueAt G C cut s pct then k else k + 1) = 0
        ∨ (s'.tree = s.tree
          ∧ EvF pct s' j (if xs 0 ∈ untrueAt G C cut s pct then k else k + 1) T (Fin.tail xs))
    | .inr _ => False

theorem evF_mono (pct : List Bool × α × List Bool) :
    ∀ (j : ℕ) (s : TState α) (k k' T : ℕ) (xs : Fin T → FreeMonoid α), k ≤ k' →
      EvF G C cut S pct s j k' T xs → EvF G C cut S pct s j k T xs := by
  intro j
  induction j with
  | zero =>
    intro s k k' T xs hk h
    rcases k with _ | k
    · simp [EvF]
    rcases k' with _ | k'
    · omega
    simp [EvF] at h
  | succ j ih =>
    intro s k k' T xs hk h
    rcases k with _ | k
    · simp [EvF]
    rcases k' with _ | k'
    · omega
    rcases T with _ | T
    · simp [EvF] at h
    simp only [EvF] at h ⊢
    obtain ⟨hc, he, h⟩ := h
    refine ⟨hc, he, ?_⟩
    rcases hst : tallyStep C cut s (xs 0) with s' | _ <;> rw [hst] at h
    · simp only []
      rcases h with h | ⟨htr, h⟩
      · left; split_ifs at h ⊢ <;> omega
      · by_cases h0 : (if xs 0 ∈ untrueAt G C cut s pct then k else k + 1) = 0
        · exact .inl h0
        refine .inr ⟨htr, ih s' _ _ T _ ?_ h⟩
        split_ifs <;> omega
    · exact h

open scoped Classical in
theorem fake_bin (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (pct : List Bool × α × List Bool) {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1)
    (hU : ∀ s : TState α, G.InClass S s.tree → EdgesInto s.tree s.edges →
      D.real (untrueAt G C cut s pct) ≤ ρ) :
    ∀ (T j : ℕ) (s : TState α) (k : ℕ),
      (Measure.pi fun _ : Fin T => D) {xs | EvF G C cut S pct s j k T xs}
        ≤ ENNReal.ofReal (binomSfGe j ρ k) := by
  intro T
  induction T with
  | zero =>
    intro j s k
    rcases k with _ | k
    · rw [binomSfGe_zero_right, ENNReal.ofReal_one]; exact prob_le_one
    rcases j with _ | j <;> simp [EvF]
  | succ T ih =>
    intro j s k
    rcases k with _ | k
    · rw [binomSfGe_zero_right, ENNReal.ofReal_one]; exact prob_le_one
    rcases j with _ | j
    · simp [EvF]
    by_cases hc : G.InClass S s.tree ∧ EdgesInto s.tree s.edges
    swap
    · have : {xs : Fin (T + 1) → FreeMonoid α | EvF G C cut S pct s (j + 1) (k + 1) (T + 1) xs}
          = ∅ := by
        ext xs
        simp only [EvF, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_and]
        exact fun h1 h2 => absurd ⟨h1, h2⟩ hc
      rw [this, measure_empty]; exact zero_le
    set A := untrueAt G C cut s pct
    have hA : D.real A ≤ ρ := hU s hc.1 hc.2
    have hA0 : 0 ≤ D.real A := measureReal_nonneg
    set c₁ := binomSfGe j ρ k
    set c₀ := binomSfGe j ρ (k + 1)
    have hc₀ : 0 ≤ c₀ := binomSfGe_nonneg hρ0 hρ1 _
    have hc₀₁ : c₀ ≤ c₁ := binomSfGe_antitone hρ0 hρ1 k
    have hc₁ : 0 ≤ c₁ := hc₀.trans hc₀₁
    rw [pi_succ_apply]
    have hsec : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
        {xs : Fin (T + 1) → FreeMonoid α | EvF G C cut S pct s (j + 1) (k + 1) (T + 1) xs}}
        ≤ A.indicator (fun _ => ENNReal.ofReal c₁) x
          + Aᶜ.indicator (fun _ => ENNReal.ofReal c₀) x := by
      intro x
      have hset : {xs | Fin.cons x xs ∈
          {xs : Fin (T + 1) → FreeMonoid α | EvF G C cut S pct s (j + 1) (k + 1) (T + 1) xs}}
          = {xs | match tallyStep C cut s x with
            | .inl s' => (if x ∈ A then k else k + 1) = 0
                ∨ (s'.tree = s.tree ∧ EvF G C cut S pct s' j (if x ∈ A then k else k + 1) T xs)
            | .inr _ => False} := by
        ext xs
        simp only [Set.mem_ofPred_eq, EvF, Fin.cons_zero, Fin.tail_cons, hc, true_and, A]
      rw [hset]
      rcases hst : tallyStep C cut s x with s' | _
      · simp only []
        by_cases hx : x ∈ A
        · rw [Set.indicator_of_mem hx, Set.indicator_of_notMem (by simpa using hx), add_zero]
          simp only [if_pos hx]
          rcases k with _ | k
          · rw [show c₁ = 1 from binomSfGe_zero_right _ _, ENNReal.ofReal_one]
            exact prob_le_one
          · refine le_trans (measure_mono fun xs hxs => ?_) (ih j s' (k + 1))
            rcases hxs with h | h
            · omega
            · exact h.2
        · rw [Set.indicator_of_notMem hx, Set.indicator_of_mem (by simpa using hx), zero_add]
          simp only [if_neg hx]
          refine le_trans (measure_mono fun xs hxs => ?_) (ih j s' (k + 1))
          rcases hxs with h | h
          · omega
          · exact h.2
      · simp only [Set.setOf_false, measure_empty]
        exact zero_le
    refine (lintegral_mono hsec).trans ?_
    rw [lintegral_add_left (measurable_of_countable _), lintegral_indicator
        (Set.to_countable _).measurableSet, lintegral_indicator
        (Set.to_countable _).measurableSet,
      setLIntegral_const, setLIntegral_const, ← ofReal_measureReal, ← ofReal_measureReal,
      measureReal_compl (Set.to_countable _).measurableSet, probReal_univ,
      ← ENNReal.ofReal_mul hc₁, ← ENNReal.ofReal_mul hc₀,
      ← ENNReal.ofReal_add (mul_nonneg hc₁ hA0) (mul_nonneg hc₀ (by linarith))]
    refine ENNReal.ofReal_le_ofReal ?_
    rw [binomSfGe_succ]
    change c₁ * D.real A + c₀ * (1 - D.real A) ≤ ρ * c₁ + (1 - ρ) * c₀
    nlinarith

open scoped Classical in
theorem badRecs_pre (s : TState α) (x : FreeMonoid α) (pct : List Bool × α × List Bool) :
    badRecs G (tallyPre C cut s x) pct
      ≤ badRecs G s pct + if x ∈ untrueAt G C cut s pct then 1 else 0 := by
  have htr := (tallyPre_cases cut C s x).1
  unfold badRecs
  rw [htr]
  rcases tallyPre_recs cut C s x with ⟨-, h⟩ | ⟨p, c, t, sp, hr, h⟩
  · rw [h]; omega
  rw [h]
  obtain ⟨p', c', t'⟩ := pct
  simp only [TState.addRec]
  by_cases hp : p' = p
  · subst hp
    by_cases hc : c' = c
    · subst hc
      simp only [Function.update_self, List.filter_append, List.length_append]
      by_cases hb : t = t' ∧ ¬ G.TrueRec s.tree (p', c', t') sp
      · have : x ∈ untrueAt G C cut s (p', c', t') := ⟨sp, by rw [hr, hb.1], hb.2⟩
        simp only [if_pos this]
        have := List.length_filter_le (fun r : FreeMonoid α × List Bool =>
          decide (r.2 = t' ∧ ¬ G.TrueRec s.tree (p', c', t') r.1)) [(sp, t)]
        simp only [List.length_singleton] at this
        omega
      · simp [hb]
    · simp [Function.update_of_ne hc]
  · simp [Function.update_of_ne hp]

/-- A step at the same tree keeps the charge's records. -/
theorem tallyStep_recs {s s' : TState α} {x : FreeMonoid α} (h : tallyStep C cut s x = .inl s')
    (htr : s'.tree = s.tree) : s'.recs = (tallyPre C cut s x).recs := by
  have htr₁ := (tallyPre_cases cut C s x).1
  rcases tallyStep_cases cut C h with ⟨e, -, he⟩ | ⟨-, hs⟩
  · cases he
  rcases settleOne_cases cut C hs with ⟨he, -⟩ | ⟨p, c, t, hvi, he | ⟨he, -⟩⟩
  · cases he; rfl
  · cases he
    rcases (fixEdge_cases cut C hvi).2 with ⟨-, hr, -⟩ | ⟨t₀, w₀, -, -, ht⟩
    · exact hr
    · rw [ht, htr₁] at htr
      exact absurd htr (DTree.splitAt_ne (htr₁ ▸ hvi.1))
  · cases he

/-- A step that changes the tree splits a leaf where two targets both have `m` records. -/
theorem tallyStep_split (hm : 0 < C.m) {s s' : TState α} {x : FreeMonoid α}
    (he : EdgesInto s.tree s.edges)
    (hr : RecsInto s) (h : tallyStep C cut s x = .inl s') (htr : s'.tree ≠ s.tree) :
    ∃ p c t t₀, p ∈ s.tree.paths ∧ t ∈ s.tree.paths ∧ t₀ ∈ s.tree.paths ∧ t ≠ t₀
      ∧ C.m ≤ (tallyPre C cut s x).tally p c t ∧ C.m ≤ (tallyPre C cut s x).tally p c t₀
      ∧ s'.tree = s.tree.splitAt (FreeMonoid.of c * s.tree.midAt (lcp t t₀)) p := by
  obtain ⟨htr₁, he₁, hr₁⟩ := tallyPre_into cut C x he hr
  rcases tallyStep_cases cut C h with ⟨e, -, he'⟩ | ⟨-, hs⟩
  · cases he'
  rcases settleOne_cases cut C hs with ⟨he', -⟩ | ⟨p, c, t, hvi, he' | ⟨he', -⟩⟩
  · cases he'; exact absurd htr₁ htr
  · cases he'
    rcases (fixEdge_cases cut C hvi).2 with ⟨ht, -⟩ | ⟨t₀, w₀, he0, hm, ht⟩
    · exact absurd (ht.trans htr₁) htr
    obtain ⟨hp, hne, htal⟩ := hvi
    have htp : t ∈ s.tree.paths := by
      have : 0 < (((tallyPre C cut s x).recs p c).filter (·.2 = t)).length := by
        unfold TState.tally at htal; omega
      obtain ⟨r, hr'⟩ := List.exists_mem_of_length_pos this
      obtain ⟨h1, h2⟩ := List.mem_filter.1 hr'
      simp only [decide_eq_true_eq] at h2
      exact htr₁ ▸ h2 ▸ hr₁ _ _ _ h1
    refine ⟨p, c, t, t₀, htr₁ ▸ hp, htp, htr₁ ▸ he₁ _ _ _ _ he0, fun h' => ?_, htal, hm,
      by rw [ht, htr₁]⟩
    subst h'
    rw [he0] at hne
    exact hne rfl
  · cases he'

open scoped Classical in
/-- A split that is not genuine has all of one target's records not true. -/
theorem badRecs_of_fake {s₁ : TState α} {p : List Bool} {c : α} {t t₀ : List Bool}
    (hne : t ≠ t₀) (hm₁ : C.m ≤ s₁.tally p c t) (hm₂ : C.m ≤ s₁.tally p c t₀)
    (hng : ¬ G.GenuineSplit s₁.tree p (FreeMonoid.of c * s₁.tree.midAt (lcp t t₀))) :
    C.m ≤ badRecs G s₁ (p, c, t) ∨ C.m ≤ badRecs G s₁ (p, c, t₀) := by
  have hall : ∀ t', (¬ ∃ r ∈ s₁.recs p c, r.2 = t' ∧ G.TrueRec s₁.tree (p, c, t') r.1) →
      badRecs G s₁ (p, c, t') = s₁.tally p c t' := by
    intro t' hn
    unfold badRecs TState.tally
    congr 1
    refine List.filter_congr fun r hr => ?_
    simp only [decide_eq_decide]
    exact ⟨fun h => h.1, fun h => ⟨h, fun ht => hn ⟨r, hr, h, ht⟩⟩⟩
  by_cases h₁ : ∃ r ∈ s₁.recs p c, r.2 = t ∧ G.TrueRec s₁.tree (p, c, t) r.1
  · by_cases h₂ : ∃ r ∈ s₁.recs p c, r.2 = t₀ ∧ G.TrueRec s₁.tree (p, c, t₀) r.1
    · obtain ⟨r₁, -, -, hr₁⟩ := h₁
      obtain ⟨r₂, -, -, hr₂⟩ := h₂
      exact absurd (G.genuineSplit_of_recs hr₁ hr₂ hne) hng
    · exact .inr (hall t₀ h₂ ▸ hm₂)
  · exact .inl (hall t h₁ ▸ hm₁)

/-- The edges and targets of a tree. -/
def keysF (T : DTree α) : Finset (List Bool × α × List Bool) :=
  T.paths.toFinset ×ˢ (Finset.univ : Finset α) ×ˢ T.paths.toFinset

open scoped Classical in
theorem fake_incl (hm : 0 < C.m) :
    ∀ (j : ℕ) (s : TState α) (T : ℕ) (xs : Fin T → FreeMonoid α), G.InClass S s.tree →
      EdgesInto s.tree s.edges → RecsInto s → subEnd G C cut s j T xs = .fake →
      ∃ pct ∈ keysF s.tree, EvF G C cut S pct s j (C.m - badRecs G s pct) T xs := by
  intro j
  induction j with
  | zero => intro s T xs _ _ _ h; simp [subEnd] at h
  | succ j ih =>
    intro s T xs hc he hr h
    rcases T with _ | T
    · simp [subEnd] at h
    have hu : ∀ pct, ∀ s' : TState α, tallyStep C cut s (xs 0) = .inl s' →
        (s'.tree = s.tree ∧ EvF G C cut S pct s' j
          (if xs 0 ∈ untrueAt G C cut s pct then C.m - badRecs G s pct - 1
            else C.m - badRecs G s pct) T (Fin.tail xs))
          ∨ (if xs 0 ∈ untrueAt G C cut s pct then C.m - badRecs G s pct - 1
            else C.m - badRecs G s pct) = 0 →
        EvF G C cut S pct s (j + 1) (C.m - badRecs G s pct) (T + 1) xs := by
      intro pct s' hst hev
      rcases hk : C.m - badRecs G s pct with _ | k
      · simp [EvF]
      simp only [EvF]
      refine ⟨hc, he, ?_⟩
      rw [hst]
      simp only []
      rw [hk] at hev
      rcases hev with ⟨htr, hev⟩ | hev
      · by_cases h0 : (if xs 0 ∈ untrueAt G C cut s pct then k else k + 1) = 0
        · exact .inl h0
        refine .inr ⟨htr, ?_⟩
        split_ifs at hev ⊢ <;> simpa using hev
      · left; split_ifs at hev ⊢ <;> omega
    rcases hst : tallyStep C cut s (xs 0) with s' | ⟨e, s'⟩
    · have hspec := tallyStep_spec cut C hm (x := xs 0) he hr
      by_cases htr : s'.tree = s.tree
      · have hseg : subSeg G C cut s (xs 0) = .inl s' := by simp [subSeg, hst, htr]
        rw [subEnd_succ, hseg] at h
        simp only [Sum.elim_inl] at h
        rcases hspec.1 _ hst with ⟨-, he', hr'⟩ | ⟨p₁, c₁, t₁, t₁', -, -, hp, hT', -⟩
        swap
        · exact absurd htr (by rw [hT']; exact DTree.splitAt_ne hp)
        obtain ⟨pct, hpct, hev⟩ := ih s' T (Fin.tail xs) (htr ▸ hc) he' hr' h
        refine ⟨pct, htr ▸ hpct, hu pct s' hst (.inl ⟨htr, evF_mono G C cut S pct j s' _ _ T _ ?_ hev⟩)⟩
        have h1 := badRecs_pre G C cut s (xs 0) pct
        have h2 : badRecs G s' pct = badRecs G (tallyPre C cut s (xs 0)) pct := by
          unfold badRecs
          rw [tallyStep_recs C cut hst htr, htr, (tallyPre_cases cut C s (xs 0)).1]
        split_ifs at h1 ⊢ <;> omega
      · obtain ⟨p, c, t, t₀, hp, ht, ht₀, hne, hm₁, hm₂, hT⟩ :=
          tallyStep_split C cut hm he hr hst htr
        have htr₁ := (tallyPre_cases cut C s (xs 0)).1
        by_cases hgen : ∃ p d, s'.tree = s.tree.splitAt d p ∧ G.GenuineSplit s.tree p d
        · have hseg : subSeg G C cut s (xs 0) = .inr (.inr true, s') := by
            simp [subSeg, hst, htr, hgen]
          rw [subEnd_succ, hseg] at h
          simp [subKind] at h
        have hng : ¬ G.GenuineSplit (tallyPre C cut s (xs 0)).tree p
            (FreeMonoid.of c * (tallyPre C cut s (xs 0)).tree.midAt (lcp t t₀)) := by
          rw [htr₁]; exact fun hg => hgen ⟨p, _, hT, hg⟩
        have hpct : ∀ t', t' ∈ s.tree.paths →
            C.m ≤ badRecs G (tallyPre C cut s (xs 0)) (p, c, t') →
            ∃ pct ∈ keysF s.tree, EvF G C cut S pct s (j + 1) (C.m - badRecs G s pct) (T + 1) xs := by
          intro t' ht' hb
          refine ⟨(p, c, t'), by simp [keysF, hp, ht'], hu _ s' hst (.inr ?_)⟩
          have := badRecs_pre G C cut s (xs 0) (p, c, t')
          split_ifs at this ⊢ <;> omega
        rcases badRecs_of_fake G C hne hm₁ hm₂ hng with hb | hb
        · exact hpct t ht hb
        · exact hpct t₀ ht₀ hb
    · have hseg : subSeg G C cut s (xs 0) = .inr (.inl e, s') := by simp [subSeg, hst]
      rw [subEnd_succ, hseg] at h
      rcases e with _ | _ | _ | _ | _ <;> simp [subKind] at h

end Fake

section Keep

variable (C : TallyCfg) (cut : FreeMonoid α → Option Bool)

/-- The next `j` probes from `s` keep its version. -/
def KeepV : TState α → ℕ → (T : ℕ) → (Fin T → FreeMonoid α) → Prop
  | _, 0, _, _ => True
  | _, _ + 1, 0, _ => False
  | s, j + 1, T + 1, xs => ∃ s', tallyStep C cut s (xs 0) = .inl s' ∧ s'.version = s.version
      ∧ KeepV s' j T (Fin.tail xs)

open scoped Classical in
/-- How many of the first `j` draws satisfy `P`. -/
noncomputable def cnt {X : Type*} (j : ℕ) (P : X → Prop) {T : ℕ} (xs : Fin T → X) : ℕ :=
  ∑ i : Fin T, if (i : ℕ) < j ∧ P (xs i) then 1 else 0

open scoped Classical in
theorem cnt_succ {X : Type*} (j : ℕ) (P : X → Prop) {T : ℕ} (xs : Fin (T + 1) → X) :
    cnt (j + 1) P xs = (if P (xs 0) then 1 else 0) + cnt j P (Fin.tail xs) := by
  unfold cnt
  rw [Fin.sum_univ_succ]
  simp only [Fin.val_zero, Nat.zero_lt_succ, true_and, Fin.val_succ, Nat.add_lt_add_iff_right]
  rfl

theorem cnt_zero {X : Type*} (P : X → Prop) {T : ℕ} (xs : Fin T → X) : cnt 0 P xs = 0 := by
  simp [cnt]

theorem keepV_le : ∀ (j : ℕ) (s : TState α) (T : ℕ) (xs : Fin T → FreeMonoid α),
    KeepV C cut s j T xs → j ≤ T
  | 0, _, _, _, _ => Nat.zero_le _
  | _ + 1, _, 0, _, h => h.elim
  | j + 1, s, T + 1, xs, ⟨s', _, _, h⟩ => by have := keepV_le j s' T _ h; omega

theorem keepV_mono : ∀ (j j' : ℕ) (s : TState α) (T : ℕ) (xs : Fin T → FreeMonoid α),
    j' ≤ j → KeepV C cut s j T xs → KeepV C cut s j' T xs
  | _, 0, _, _, _, _, _ => trivial
  | 0, _ + 1, _, _, _, h, _ => absurd h (by omega)
  | _ + 1, _ + 1, _, 0, _, _, h => h.elim
  | j + 1, j' + 1, s, T + 1, xs, hj, ⟨s', h1, h2, h3⟩ =>
    ⟨s', h1, h2, keepV_mono j j' s' T _ (by omega) h3⟩

open scoped Classical in
theorem tally_pre (s : TState α) (x : FreeMonoid α) (p : List Bool) (c : α) (t : List Bool) :
    s.tally p c t + (if ∃ sp, recordBy cut C.k (s.tree, s.edges) x = some ((p, c, t), sp)
      then 1 else 0) ≤ (tallyPre C cut s x).tally p c t := by
  unfold TState.tally
  rcases tallyPre_recs cut C s x with ⟨hr, h⟩ | ⟨p', c', t', sp, hr, h⟩
  · rw [h, hr]; simp
  rw [h]
  by_cases hpc : p = p' ∧ c = c'
  · obtain ⟨hp, hc⟩ := hpc
    subst hp hc
    simp only [TState.addRec, Function.update_self, List.filter_append, List.length_append]
    by_cases ht : t = t'
    · subst ht
      rw [if_pos ⟨sp, hr⟩]
      simp
    · have : ¬ ∃ sp', recordBy cut C.k (s.tree, s.edges) x = some ((p, c, t), sp') := by
        rintro ⟨sp', h'⟩
        rw [hr] at h'
        simp only [Option.some.injEq, Prod.mk.injEq] at h'
        exact ht h'.1.2.2.symm
      rw [if_neg this]
      omega
  · have : ¬ ∃ sp', recordBy cut C.k (s.tree, s.edges) x = some ((p, c, t), sp') := by
      rintro ⟨sp', h'⟩
      rw [hr] at h'
      simp only [Option.some.injEq, Prod.mk.injEq] at h'
      exact hpc ⟨h'.1.1.symm, h'.1.2.1.symm⟩
    rw [if_neg this, add_zero]
    simp only [TState.addRec]
    by_cases hp : p = p'
    · subst hp
      have hc : c ≠ c' := fun hc => hpc ⟨rfl, hc⟩
      simp [Function.update_of_ne hc]
    · simp [Function.update_of_ne hp]

open scoped Classical in
/-- Over a run that keeps its version, the counts grow by the outcomes of the fixed hypothesis,
and the last charge's tests do not fire and no edge is violated. -/
theorem keep_counts (T0 : DTree α) (E0 : Edges α) :
    ∀ (j : ℕ) (s : TState α) (T : ℕ) (xs : Fin T → FreeMonoid α), s.tree = T0 → s.edges = E0 →
      KeepV C cut s (j + 1) T xs → ∃ s₁ : TState α, s₁.tree = T0 ∧ s₁.edges = E0
        ∧ s₁.n = s.n + (j + 1)
        ∧ s₁.pt.length = s.pt.length + cnt (j + 1) (fun x => ptHarvBy cut T0 E0 C.k x ≠ []) xs
        ∧ s₁.dis = s.dis + cnt (j + 1) (fun x => (probeBy cut T0 E0 C.k x).IsSearch) xs
        ∧ (∀ p c t, s.tally p c t + cnt (j + 1)
            (fun x => ∃ sp, recordBy cut C.k (T0, E0) x = some ((p, c, t), sp)) xs
              ≤ s₁.tally p c t)
        ∧ tallyLook C s₁ = none ∧ ∀ p c t, ¬ Violates C.m s₁ p c t := by
  intro j
  induction j with
  | zero =>
    intro s T xs htr hed h
    rcases T with _ | T
    · exact h.elim
    obtain ⟨s', hst, hv, -⟩ := h
    obtain ⟨rfl, hl, hnv, hv'⟩ := tallyStep_same cut C hst hv
    obtain ⟨htr', hc⟩ := tallyPre_cases cut C s (xs 0)
    rcases hc with ⟨-, hed', hn, hpt, hdis⟩ | ⟨hv'', -⟩
    swap
    · omega
    subst htr hed
    refine ⟨_, htr', hed', hn, ?_, ?_, fun p c t => ?_, hl, hnv⟩
    · rw [hpt, List.length_append, cnt_succ, cnt_zero]
      rcases ptHarvBy_cases cut s.tree s.edges C.k (xs 0) with h0 | ⟨b, hb⟩
      · simp [h0]
      · simp [hb]
    · rw [hdis, cnt_succ, cnt_zero, add_zero]
    · rw [cnt_succ, cnt_zero, add_zero]
      exact tally_pre C cut s (xs 0) p c t
  | succ j ih =>
    intro s T xs htr hed h
    rcases T with _ | T
    · exact h.elim
    obtain ⟨s', hst, hv, hk⟩ := h
    obtain ⟨rfl, hl, hnv, hv'⟩ := tallyStep_same cut C hst hv
    obtain ⟨htr', hc⟩ := tallyPre_cases cut C s (xs 0)
    rcases hc with ⟨-, hed', hn, hpt, hdis⟩ | ⟨hv'', -⟩
    swap
    · omega
    obtain ⟨s₁, h1, h2, h3, h4, h5, h6, h7, h8⟩ :=
      ih _ T (Fin.tail xs) (htr'.trans htr) (hed'.trans hed) hk
    subst htr hed
    refine ⟨s₁, h1, h2, by rw [h3, hn]; ring, ?_, ?_, fun p c t => ?_, h7, h8⟩
    · rw [h4, hpt, List.length_append, cnt_succ (j + 1)]
      rcases ptHarvBy_cases cut s.tree s.edges C.k (xs 0) with h0 | ⟨b, hb⟩
      · simp [h0]
      · simp [hb]; ring
    · rw [h5, hdis, cnt_succ (j + 1)]; ring
    · have := h6 p c t
      have := tally_pre C cut s (xs 0) p c t
      rw [cnt_succ (j + 1)]
      omega

end Keep

section Outcomes

variable (cut : FreeMonoid α → Option Bool)

omit [Fintype α] [DecidableEq α] in
/-- A search stopping at a pair or a triple stops at an undecided read. -/
theorem bracketAt_none (agrees : ℕ → Option Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi j, (bracketAt (α := α) agrees ps fuel lo hi = .pair j
      ∨ bracketAt (α := α) agrees ps fuel lo hi = .triple j) → agrees j = none
  | 0, _, _, _, h => by simp [bracketAt] at h
  | fuel + 1, lo, hi, j, h => by
    have key : ∀ q, (if q = lo then some true else if q = hi then some false else agrees q)
        = none → agrees q = none := fun q hq => by split_ifs at hq; exact hq
    simp only [bracketAt] at h
    by_cases hlh : lo + 1 < hi
    swap
    · simp only [if_neg hlh] at h; simp at h
    simp only [if_pos hlh] at h
    have k1 := key ((lo + hi) / 2)
    have k2 := key ((lo + hi) / 2 - 1)
    generalize (if (lo + hi) / 2 = lo then some true else if (lo + hi) / 2 = hi then some false
      else agrees ((lo + hi) / 2)) = v at h k1
    generalize (if (lo + hi) / 2 - 1 = lo then some true
      else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1)) = l at h k2
    generalize (if (lo + hi) / 2 + 1 = lo then some true
      else if (lo + hi) / 2 + 1 = hi then some false else agrees ((lo + hi) / 2 + 1)) = r at h
    rcases v with _ | _ | _
    · rcases l with _ | _ | _ <;> rcases r with _ | _ | _ <;>
        simp only [reduceCtorEq, or_false, false_or, Outcome.pair.injEq,
          Outcome.triple.injEq] at h
      all_goals first
        | exact bracketAt_none agrees ps fuel _ _ j h
        | (subst h; first | exact k2 rfl | exact k1 rfl)
    · exact bracketAt_none agrees ps fuel _ _ j h
    · exact bracketAt_none agrees ps fuel _ _ j h

/-- A search ends at an undecided middle, or records. -/
theorem search_cases {T : DTree α} {E : Edges α} {k : ℕ} {x : FreeMonoid α}
    (h : (probeBy cut T E k x).IsSearch) :
    ptHarvBy cut T E k x ≠ [] ∨ ∃ pct sp, recordBy cut k (T, E) x = some (pct, sp) := by
  obtain ⟨ps₀, hi, hw, hb⟩ := probeBy_search cut rfl h
  obtain ⟨p₀, hk, hf, hkh, hhx, hag⟩ := walkCheckBy_inr cut hw
  rcases ho : probeBy cut T E k x with _ | _ | _ | j | ⟨ps, fd⟩ | j | _ <;> rw [ho] at h hb <;>
    simp only [Outcome.IsSearch] at h
  · left
    have := bracketAt_none _ ps₀ _ _ _ j (.inl hb)
    unfold ptHarvBy
    rw [ho]
    simp only [agreesAtBy] at this
    rcases hs : T.sift cut (prefixOf x j) with _ | b <;> rw [hs] at this
    · simp at this
    · simp [hs]
  · right
    obtain ⟨hlen, hhead, -⟩ := follow_inl _ _ _ hf
    have hlo : agreesAtBy cut T x (fun j => ps₀.getD (j - k) []) k = some true := by
      simp [agreesAtBy, hk, ← hhead, List.getElem?_eq_getElem]
    obtain ⟨-, hfd1, hfd2, -, hfd⟩ :=
      bracketAt_edge _ ps₀ (hi - k) k hi ps fd hkh le_rfl hlo hag hb
    unfold recordBy
    simp only [ho]
    have hc : ∃ c, x.toList[fd - 1]? = some c :=
      ⟨_, List.getElem?_eq_getElem (by omega)⟩
    obtain ⟨c, hc⟩ := hc
    rw [hc]
    simp only [agreesAtBy] at hfd
    rcases ht : T.sift cut (prefixOf x fd) with t | b <;> rw [ht] at hfd
    · exact ⟨_, _, rfl⟩
    · simp at hfd
  · left
    have := bracketAt_none _ ps₀ _ _ _ j (.inr hb)
    unfold ptHarvBy
    rw [ho]
    simp only [agreesAtBy] at this
    rcases hs : T.sift cut (prefixOf x j) with _ | b <;> rw [hs] at this
    · simp at this
    · simp [hs]

/-- A record is at a leaf, out along an edge not already pointing at its target, a leaf. -/
theorem rec_keys {T : DTree α} {E : Edges α} {k : ℕ} {x : FreeMonoid α}
    {pct : List Bool × α × List Bool} {sp : FreeMonoid α} (he : EdgesInto T E)
    (h : recordBy cut k (T, E) x = some (pct, sp)) :
    pct ∈ keysF T ∧ (E pct.1 pct.2.1).map Prod.fst ≠ some pct.2.2 := by
  obtain ⟨p, c, t⟩ := pct
  obtain ⟨p₀, ps, hk, hf, hp, ht, hne⟩ := recordBy_spec cut h
  refine ⟨?_, hne⟩
  simp only [keysF, Finset.mem_product, List.mem_toFinset, Finset.mem_univ, true_and]
  exact ⟨follow_mem_paths he _ p₀ ps (DTree.sift_mem_paths _ _ _ hk) hf p hp,
    DTree.sift_mem_paths _ _ _ ht⟩

end Outcomes

section KeepBound

variable (C : TallyCfg) (cut : FreeMonoid α → Option Bool)

theorem look_none_pt {s : TState α} {hP : ℕ} (hl : tallyLook C s = none) (hn₀ : C.n₀ ≤ s.dis)
    (hθ0 : 0 ≤ C.θpt) (hθ1 : C.θpt ≤ 1) (hhP : binomSfGe s.dis C.θpt hP < C.a) :
    s.pt.length < hP := by
  by_contra hge
  push_neg at hge
  have hr : rateSide C.θpt C.a C.n₀ s.dis s.pt.length = some true := by
    unfold rateSide
    rw [if_pos hn₀, if_pos ((binomSfGe_antitone' hθ0 hθ1 hge).trans_lt hhP)]
  unfold tallyLook at hl
  split_ifs at hl <;> contradiction

theorem look_none_dis {s : TState α} {hS : ℕ} (hl : tallyLook C s = none) (hn₀ : C.n₀ ≤ s.n)
    (hε0 : 0 ≤ C.εd) (hε1 : C.εd ≤ 1) (hhS : 1 - binomSfGe s.n C.εd (hS + 1) < C.a)
    (hhS' : C.a ≤ binomSfGe s.n C.εd hS) : hS + 1 ≤ s.dis := by
  by_contra hlt
  push_neg at hlt
  have h1 : ¬ binomSfGe s.n C.εd s.dis < C.a :=
    not_lt.2 (hhS'.trans (binomSfGe_antitone' hε0 hε1 (by omega)))
  have h3 := binomSfGe_antitone' (n := s.n) hε0 hε1 (show s.dis + 1 ≤ hS + 1 by omega)
  have h2 : 1 - binomSfGe s.n C.εd (s.dis + 1) < C.a := by linarith
  have hr : rateSide C.εd C.a C.n₀ s.n s.dis = some false := by
    unfold rateSide
    rw [if_pos hn₀, if_neg h1, if_pos h2]
  unfold tallyLook at hl
  split_ifs at hl <;> contradiction

open scoped Classical in
theorem pi_cnt (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (P : FreeMonoid α → Prop)
    {T j : ℕ} (hj : j ≤ T) (h : ℕ) :
    (Measure.pi fun _ : Fin T => D).real {xs | h ≤ cnt j P xs}
      = binomSfGe j (D.real {x | P x}) h := by
  set S := Finset.univ.filter fun i : Fin T => (i : ℕ) < j
  have hS : S.card = j := by
    rw [show S = (Finset.range j).attachFin (fun i hi => by
      simp only [Finset.mem_range] at hi; omega) from by
        ext i; simp [S, Finset.mem_attachFin]]
    simp
  have hcnt : ∀ xs : Fin T → FreeMonoid α, cnt j P xs = (S.filter fun i => P (xs i)).card := by
    intro xs
    unfold cnt
    rw [Finset.card_eq_sum_ones, Finset.sum_filter, Finset.sum_filter]
    exact Finset.sum_congr rfl fun i _ => by split_ifs <;> simp_all
  simp only [hcnt]
  rw [pi_count_ge D P S h, hS]

end KeepBound

section Trials

variable {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]

theorem cnt_succ_le (P : X → Prop) : ∀ (T j : ℕ) (xs : Fin T → X),
    cnt j P xs ≤ cnt (j + 1) P xs ∧ cnt (j + 1) P xs ≤ cnt j P xs + 1
  | 0, _, _ => by simp [cnt]
  | T + 1, 0, xs => by rw [cnt_succ, cnt_zero, cnt_zero]; split_ifs <;> omega
  | T + 1, j + 1, xs => by
    have := cnt_succ_le P T j (Fin.tail xs)
    rw [cnt_succ (j + 1), cnt_succ j]
    omega

theorem cnt_mono (P : X → Prop) {T : ℕ} (xs : Fin T → X) {j j' : ℕ} (h : j ≤ j') :
    cnt j P xs ≤ cnt j' P xs := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => exact ih.trans (cnt_succ_le P T _ xs).1

open scoped Classical in
/-- The draws reach `t` in `A` with fewer than `h` of them in `B`. -/
def RR (A B : Set X) : (T : ℕ) → ℕ → ℕ → (Fin T → X) → Prop
  | _, 0, h, _ => 0 < h
  | 0, _ + 1, _, _ => False
  | T + 1, t + 1, h, xs => if xs 0 ∈ B then 0 < h ∧ RR A B T t (h - 1) (Fin.tail xs)
      else if xs 0 ∈ A then RR A B T t h (Fin.tail xs) else RR A B T (t + 1) h (Fin.tail xs)

theorem rr_zero (A B : Set X) : ∀ (T t : ℕ) (xs : Fin T → X), ¬ RR A B T t 0 xs
  | _, 0, _ => by simp [RR]
  | 0, _ + 1, _ => by simp [RR]
  | T + 1, t + 1, xs => by
    simp only [RR]
    split_ifs
    · simp
    · exact rr_zero A B T t _
    · exact rr_zero A B T (t + 1) _

open scoped Classical in
theorem rr_incl {A B : Set X} (hBA : B ⊆ A) : ∀ (T j t h : ℕ) (xs : Fin T → X),
    cnt j (· ∈ A) xs = t → cnt j (· ∈ B) xs < h → RR A B T t h xs
  | T, j, 0, h, xs, _, hB => by simp only [RR]; omega
  | 0, j, t + 1, h, xs, hA, _ => by simp [cnt] at hA
  | T + 1, 0, t + 1, h, xs, hA, _ => by rw [cnt_zero] at hA; omega
  | T + 1, j + 1, t + 1, h, xs, hA, hB => by
    rw [cnt_succ] at hA hB
    simp only [RR]
    by_cases hx : xs 0 ∈ B
    · rw [if_pos hx] at hB
      rw [if_pos (hBA hx)] at hA
      rw [if_pos hx]
      exact ⟨by omega, rr_incl hBA T j t (h - 1) _ (by omega) (by omega)⟩
    · rw [if_neg hx] at hB
      rw [if_neg hx]
      by_cases hxA : xs 0 ∈ A
      · rw [if_pos hxA] at hA
        rw [if_pos hxA]
        exact rr_incl hBA T j t h _ (by omega) (by omega)
      · rw [if_neg hxA] at hA
        rw [if_neg hxA]
        exact rr_incl hBA T j (t + 1) h _ (by omega) (by omega)

open scoped Classical in
/-- Where at least `π` of the draws in `A` are in `B`, the draws reach `t` in `A` with fewer than
`h` in `B` with chance at most `P(Bin(t, π) < h)`. -/
theorem rr_le (D : Measure X) [IsProbabilityMeasure D] {A B : Set X} (hBA : B ⊆ A) {π : ℝ}
    (hπ0 : 0 ≤ π) (hπ1 : π ≤ 1) (hπ : π * D.real A ≤ D.real B) :
    ∀ (T t h : ℕ), (Measure.pi fun _ : Fin T => D) {xs | RR A B T t h xs}
      ≤ ENNReal.ofReal (1 - binomSfGe t π h) := by
  have hle1 : ∀ E : Set X, D.real E ≤ 1 := fun E => by
    have := measureReal_mono (μ := D) (Set.subset_univ E)
    rwa [probReal_univ] at this
  intro T
  induction T with
  | zero =>
    intro t h
    rcases t with _ | t
    · rcases h with _ | h
      · simp [RR]
      · rw [binomSfGe_zero_left, sub_zero, ENNReal.ofReal_one]; exact prob_le_one
    · simp [RR]
  | succ T ih =>
    intro t h
    rcases h with _ | h
    · have : {xs : Fin (T + 1) → X | RR A B (T + 1) t 0 xs} = ∅ := by
        ext xs; simpa using rr_zero A B (T + 1) t xs
      rw [this, measure_empty]; exact zero_le
    rcases t with _ | t
    · rw [binomSfGe_zero_left, sub_zero, ENNReal.ofReal_one]; exact prob_le_one
    set g₁ := 1 - binomSfGe t π h
    set g₀ := 1 - binomSfGe t π (h + 1)
    set g₂ := 1 - binomSfGe (t + 1) π (h + 1)
    have hg₁ : 0 ≤ g₁ := sub_nonneg.2 (binomSfGe_le_one hπ0 hπ1 _)
    have hg₁₀ : g₁ ≤ g₀ := by
      have := binomSfGe_antitone hπ0 hπ1 (n := t) h
      simp only [g₁, g₀]; linarith
    have hg₀ : 0 ≤ g₀ := hg₁.trans hg₁₀
    have hg₂ : 0 ≤ g₂ := sub_nonneg.2 (binomSfGe_le_one hπ0 hπ1 _)
    have hsec : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
        {xs : Fin (T + 1) → X | RR A B (T + 1) (t + 1) (h + 1) xs}}
        ≤ B.indicator (fun _ => ENNReal.ofReal g₁) x
          + (A \ B).indicator (fun _ => ENNReal.ofReal g₀) x
          + Aᶜ.indicator (fun _ => ENNReal.ofReal g₂) x := by
      intro x
      by_cases hxB : x ∈ B
      · rw [Set.indicator_of_mem hxB, Set.indicator_of_notMem (fun h => h.2 hxB),
          Set.indicator_of_notMem (by simp [hBA hxB]), add_zero, add_zero]
        refine le_trans (measure_mono fun xs hxs => ?_) (ih t h)
        simp only [Set.mem_ofPred_eq, RR, Fin.cons_zero, Fin.tail_cons, if_pos hxB] at hxs
        simpa using hxs.2
      by_cases hxA : x ∈ A
      · rw [Set.indicator_of_notMem hxB, Set.indicator_of_mem (Set.mem_sdiff_of_mem hxA hxB),
          Set.indicator_of_notMem (by simp [hxA]), zero_add, add_zero]
        refine le_trans (measure_mono fun xs hxs => ?_) (ih t (h + 1))
        simp only [Set.mem_ofPred_eq, RR, Fin.cons_zero, Fin.tail_cons, if_neg hxB,
          if_pos hxA] at hxs
        exact hxs
      · rw [Set.indicator_of_notMem hxB, Set.indicator_of_notMem (fun h => hxA h.1),
          Set.indicator_of_mem hxA, zero_add, zero_add]
        refine le_trans (measure_mono fun xs hxs => ?_) (ih (t + 1) (h + 1))
        simp only [Set.mem_ofPred_eq, RR, Fin.cons_zero, Fin.tail_cons, if_neg hxB,
          if_neg hxA] at hxs
        exact hxs
    rw [pi_succ_apply]
    refine (lintegral_mono hsec).trans ?_
    have hmeas : ∀ E : Set X, MeasurableSet E := fun E => (Set.to_countable _).measurableSet
    have hB0 : 0 ≤ D.real B := measureReal_nonneg
    have hBA' : D.real B ≤ D.real A := measureReal_mono hBA
    rw [lintegral_add_left (measurable_of_countable _), lintegral_add_left
        (measurable_of_countable _), lintegral_indicator (hmeas _), lintegral_indicator (hmeas _),
      lintegral_indicator (hmeas _), setLIntegral_const, setLIntegral_const, setLIntegral_const,
      ← ofReal_measureReal, ← ofReal_measureReal, ← ofReal_measureReal,
      measureReal_compl (hmeas _), probReal_univ, measureReal_sdiff hBA (hmeas _),
      ← ENNReal.ofReal_mul hg₁, ← ENNReal.ofReal_mul hg₀, ← ENNReal.ofReal_mul hg₂,
      ← ENNReal.ofReal_add (mul_nonneg hg₁ hB0) (mul_nonneg hg₀ (by linarith)),
      ← ENNReal.ofReal_add (by positivity) (mul_nonneg hg₂ (by linarith [hle1 A]))]
    refine ENNReal.ofReal_le_ofReal ?_
    have e₂ : g₂ = π * g₁ + (1 - π) * g₀ := by
      simp only [g₂, g₁, g₀, binomSfGe_succ]; ring
    rw [e₂]
    nlinarith [mul_le_mul_of_nonneg_right hπ (sub_nonneg.2 hg₁₀)]

end Trials

section KeepLe

variable (C : TallyCfg) (cut : FreeMonoid α → Option Bool)

/-- The probes that record against the hypothesis of `s`. -/
def recAt (s : TState α) : Set (FreeMonoid α) :=
  {x | ∃ pct sp, recordBy cut C.k (s.tree, s.edges) x = some (pct, sp)}

theorem ptHarvBy_search {T : DTree α} {E : Edges α} {k : ℕ} {x : FreeMonoid α}
    (h : ptHarvBy cut T E k x ≠ []) : (probeBy cut T E k x).IsSearch := by
  unfold ptHarvBy at h
  rcases ho : probeBy cut T E k x with _ | _ | _ | _ | _ | _ | _ <;> rw [ho] at h <;>
    simp_all [Outcome.IsSearch]

open scoped Classical in
/-- A fresh hypothesis recording less than `θr` of the time keeps over `nEnd` probes with chance at
most `termLevel`. -/
theorem keep_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {nEnd hP hS : ℕ}
    {θpt' θr εd' : ℝ} (hn₀ : C.n₀ ≤ nEnd) (hn₀' : C.n₀ ≤ hS + 1)
    (hθpt0 : 0 ≤ C.θpt) (hθpt1 : C.θpt ≤ 1) (hεd0 : 0 ≤ C.εd) (hεd1 : C.εd ≤ 1)
    (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1) (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1)
    (hcond : θr ≤ (1 - θpt') * εd')
    (hhP : binomSfGe (hS + 1) C.θpt hP < C.a) (hhS : 1 - binomSfGe nEnd C.εd (hS + 1) < C.a)
    (hhS' : C.a ≤ binomSfGe nEnd C.εd hS)
    {s : TState α} (hrec : D.real (recAt C cut s) < θr) (hn : s.n = 0) (hd : s.dis = 0) (T : ℕ) :
    (Measure.pi fun _ : Fin T => D) {xs | KeepV C cut s nEnd T xs}
      ≤ ENNReal.ofReal (termLevel nEnd hP hS θpt' εd') := by
  have t1 : 0 ≤ 1 - binomSfGe (hS + 1) θpt' hP :=
    sub_nonneg.2 (binomSfGe_le_one hθpt'0 hθpt'1 _)
  have t3 : 0 ≤ binomSfGe nEnd εd' (hS + 1) := binomSfGe_nonneg hεd'0 hεd'1 _
  have hτ : termLevel nEnd hP hS θpt' εd'
      = (1 - binomSfGe (hS + 1) θpt' hP) + binomSfGe nEnd εd' (hS + 1) := rfl
  by_cases hT : nEnd ≤ T
  swap
  · have : {xs : Fin T → FreeMonoid α | KeepV C cut s nEnd T xs} = ∅ := by
      ext xs
      simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
      exact fun h => hT (keepV_le C cut _ _ _ _ h)
    rw [this, measure_empty]; exact zero_le
  rcases nEnd with _ | N
  · exfalso
    rw [binomSfGe_zero_left] at hhS
    have := binomSfGe_le_one (n := 0) hεd0 hεd1 hS
    linarith
  set T0 := s.tree
  set E0 := s.edges
  set ptO : FreeMonoid α → Prop := fun x => ptHarvBy cut T0 E0 C.k x ≠ []
  set disO : FreeMonoid α → Prop := fun x => (probeBy cut T0 E0 C.k x).IsSearch
  have hBA : {x | ptO x} ⊆ {x | disO x} := fun x hx => ptHarvBy_search cut hx
  have hincl : {xs : Fin T → FreeMonoid α | KeepV C cut s (N + 1) T xs}
      ⊆ {xs | RR {x | disO x} {x | ptO x} T (hS + 1) hP xs}
        ∩ {xs | hS + 1 ≤ cnt (N + 1) disO xs} := by
    intro xs hk
    obtain ⟨s₁, h1, h2, h3, h4, h5, h6, h7, h8⟩ := keep_counts C cut T0 E0 N s T xs rfl rfl hk
    rw [hn, zero_add] at h3
    rw [hd, zero_add] at h5
    have hdis : hS + 1 ≤ cnt (N + 1) disO xs := by
      have := look_none_dis C h7 (by rw [h3]; exact hn₀) hεd0 hεd1 (by rw [h3]; exact hhS)
        (by rw [h3]; exact hhS')
      simp only [disO]
      omega
    refine ⟨?_, hdis⟩
    have hex : ∃ j, hS + 1 ≤ cnt j disO xs := ⟨N + 1, hdis⟩
    set j₀ := Nat.find hex
    have hj₀ : hS + 1 ≤ cnt j₀ disO xs := Nat.find_spec hex
    have hj₀N : j₀ ≤ N + 1 := Nat.find_min' hex hdis
    obtain ⟨j', hj'⟩ : ∃ j', j₀ = j' + 1 := by
      rcases h : j₀ with _ | j'
      · rw [h, cnt_zero] at hj₀; omega
      · exact ⟨j', rfl⟩
    have hlt : cnt j' disO xs < hS + 1 := by
      have := Nat.find_min hex (show j' < j₀ by omega)
      omega
    have heq : cnt j₀ disO xs = hS + 1 := by
      have := (cnt_succ_le disO T j' xs).2
      rw [hj'] at hj₀ ⊢
      omega
    have hk' := keepV_mono C cut (N + 1) (j' + 1) s T xs (by omega) hk
    obtain ⟨s₂, -, -, -, g4, g5, -, g7, -⟩ := keep_counts C cut T0 E0 j' s T xs rfl rfl hk'
    rw [hd, zero_add] at g5
    rw [← hj'] at g4 g5
    have hpt := look_none_pt C g7 (by rw [g5, heq]; exact hn₀') hθpt0 hθpt1
      (by rw [g5, heq]; exact hhP)
    refine rr_incl hBA T j₀ (hS + 1) hP xs heq ?_
    show cnt j₀ ptO xs < hP
    simp only [ptO]
    omega
  by_cases ha : θpt' * D.real {x | disO x} ≤ D.real {x | ptO x}
  · calc _ ≤ (Measure.pi fun _ : Fin T => D) {xs | RR {x | disO x} {x | ptO x} T (hS + 1) hP xs} :=
          measure_mono fun xs h => (hincl h).1
      _ ≤ _ := rr_le D hBA hθpt'0 hθpt'1 ha T (hS + 1) hP
      _ ≤ _ := by
        refine ENNReal.ofReal_le_ofReal ?_
        rw [hτ]; linarith
  push_neg at ha
  have hdis : D.real {x | disO x} ≤ εd' := by
    have hsub : {x | disO x} ⊆ {x | ptO x} ∪ recAt C cut s := by
      intro x hx
      rcases search_cases cut hx with h | ⟨pct, sp, h⟩
      · exact .inl h
      · exact .inr ⟨pct, sp, h⟩
    have hsum : D.real {x | disO x} ≤ D.real {x | ptO x} + D.real (recAt C cut s) :=
      (measureReal_mono hsub).trans (measureReal_union_le _ _)
    by_contra hcon
    push_neg at hcon
    nlinarith [mul_nonneg (sub_nonneg.2 hθpt'1) (sub_nonneg.2 hcon.le)]
  calc _ ≤ (Measure.pi fun _ : Fin T => D) {xs | hS + 1 ≤ cnt (N + 1) disO xs} :=
        measure_mono fun xs h => (hincl h).2
    _ = ENNReal.ofReal (binomSfGe (N + 1) (D.real {x | disO x}) (hS + 1)) := by
        rw [← ofReal_measureReal, pi_cnt D disO hT]
    _ ≤ _ := by
        refine ENNReal.ofReal_le_ofReal ?_
        have := binomSfGe_mono measureReal_nonneg hεd'1 hdis (N + 1) (hS + 1)
        rw [hτ]; linarith

end KeepLe

section Pot

variable (C : TallyCfg) (cut : FreeMonoid α → Option Bool)

open scoped Classical in
/-- An edge's share of the potential: unlearned, learned at a target with fewer than `m` records,
or neither. -/
noncomputable def psi (s : TState α) (p : List Bool) (c : α) : ℕ :=
  match s.edges p c with
  | none => 2
  | some (t₀, _) => if C.m ≤ s.tally p c t₀ then 0 else 1

noncomputable def pot (s : TState α) : ℕ :=
  ∑ pc ∈ s.tree.paths.toFinset ×ˢ (Finset.univ : Finset α), psi C s pc.1 pc.2

theorem psi_le_two (s : TState α) (p : List Bool) (c : α) : psi C s p c ≤ 2 := by
  unfold psi
  rcases s.edges p c with _ | ⟨t₀, w⟩
  · exact le_rfl
  · simp only []; split_ifs <;> omega

theorem pot_le (s : TState α) : pot C s ≤ 2 * s.tree.paths.length * Fintype.card α := by
  unfold pot
  refine (Finset.sum_le_card_nsmul _ _ 2 fun pc _ => psi_le_two C s pc.1 pc.2).trans ?_
  simp only [Finset.card_product, Finset.card_univ, smul_eq_mul]
  have := List.toFinset_card_le s.tree.paths
  calc _ ≤ s.tree.paths.length * Fintype.card α * 2 := by gcongr
    _ = _ := by ring

/-- With the same edges and no fewer records, each share is no larger. -/
theorem psi_mono {s s' : TState α} {p : List Bool} {c : α} (he : s'.edges p c = s.edges p c)
    (ht : ∀ t, s.tally p c t ≤ s'.tally p c t) : psi C s' p c ≤ psi C s p c := by
  unfold psi
  rw [he]
  rcases s.edges p c with _ | ⟨t₀, w⟩
  · exact le_rfl
  · simp only []
    have := ht t₀
    split_ifs <;> omega

theorem pot_mono {s s' : TState α} (htr : s'.tree = s.tree)
    (hle : ∀ p c, psi C s' p c ≤ psi C s p c) : pot C s' ≤ pot C s := by
  unfold pot
  rw [htr]
  exact Finset.sum_le_sum fun pc _ => hle pc.1 pc.2

theorem pot_lt {s s' : TState α} (htr : s'.tree = s.tree)
    (hle : ∀ p c, psi C s' p c ≤ psi C s p c)
    (hlt : ∃ p ∈ s.tree.paths, ∃ c, psi C s' p c < psi C s p c) : pot C s' < pot C s := by
  unfold pot
  rw [htr]
  obtain ⟨p, hp, c, h⟩ := hlt
  exact Finset.sum_lt_sum (fun pc _ => hle pc.1 pc.2)
    ⟨(p, c), by simp [hp], h⟩

theorem tally_mono_pre (s : TState α) (x : FreeMonoid α) (p : List Bool) (c : α)
    (t : List Bool) : s.tally p c t ≤ (tallyPre C cut s x).tally p c t := by
  have := tally_pre C cut s x p c t
  omega

theorem psi_pre (s : TState α) (x : FreeMonoid α) :
    (∀ p c, psi C (tallyPre C cut s x) p c ≤ psi C s p c)
      ∧ ((tallyPre C cut s x).version ≠ s.version →
        ∃ p ∈ s.tree.paths, ∃ c, psi C (tallyPre C cut s x) p c < psi C s p c) := by
  rcases (tallyPre_cases cut C s x).2 with ⟨hv, hed, -⟩ | ⟨hv, -, -, p, c, e, hnone, hp, hed⟩
  · exact ⟨fun p c => psi_mono C (by rw [hed]) (tally_mono_pre C cut s x p c),
      fun h => absurd hv h⟩
  have hother : ∀ q c', (q, c') ≠ (p, c) → psi C (tallyPre C cut s x) q c' ≤ psi C s q c' :=
    fun q c' hne => psi_mono C (by rw [hed, if_neg hne]) (tally_mono_pre C cut s x q c')
  have hat : psi C (tallyPre C cut s x) p c < psi C s p c := by
    have h2 : psi C s p c = 2 := by unfold psi; rw [hnone]
    have h1 : psi C (tallyPre C cut s x) p c ≤ 1 := by
      unfold psi
      rw [hed, if_pos rfl]
      obtain ⟨t₀, w⟩ := e
      simp only []
      split_ifs <;> omega
    omega
  refine ⟨fun q c' => ?_, fun _ => ⟨p, hp, c, hat⟩⟩
  by_cases hq : (q, c') = (p, c)
  · simp only [Prod.mk.injEq] at hq
    obtain ⟨rfl, rfl⟩ := hq
    exact hat.le
  · exact hother q c' hq

theorem psi_fix {s₁ : TState α} {p : List Bool} {c : α} {t : List Bool}
    (hvi : Violates C.m s₁ p c t)
    (hre : (fixEdge cut C.m s₁ p c t).tree = s₁.tree) :
    (∀ q c', psi C (fixEdge cut C.m s₁ p c t) q c' ≤ psi C s₁ q c')
      ∧ psi C (fixEdge cut C.m s₁ p c t) p c < psi C s₁ p c := by
  rcases (fixEdge_cases cut C hvi).2 with ⟨-, hr, -, -, hlt, w, hed⟩ | ⟨t₀, w₀, -, -, ht⟩
  swap
  · rw [ht] at hre; exact absurd hre (DTree.splitAt_ne hvi.1)
  have htal : ∀ q c' t', (fixEdge cut C.m s₁ p c t).tally q c' t' = s₁.tally q c' t' := by
    intro q c' t'; unfold TState.tally; rw [hr]
  have hat : psi C (fixEdge cut C.m s₁ p c t) p c < psi C s₁ p c := by
    have h0 : psi C (fixEdge cut C.m s₁ p c t) p c = 0 := by
      unfold psi
      rw [hed, if_pos rfl]
      simp only []
      rw [if_pos (by rw [htal]; exact hvi.2.2)]
    have h1 : 1 ≤ psi C s₁ p c := by
      unfold psi
      rcases he : s₁.edges p c with _ | ⟨t₀, w₀⟩
      · simp
      · simp only []
        rw [if_neg (by have := hlt t₀ w₀ he; omega)]
    omega
  refine ⟨fun q c' => ?_, hat⟩
  by_cases hq : (q, c') = (p, c)
  · simp only [Prod.mk.injEq] at hq
    obtain ⟨rfl, rfl⟩ := hq
    exact hat.le
  · exact psi_mono C (by rw [hed, if_neg hq]) fun t' => (htal q c' t').ge

/-- A step at the same tree lowers no share, and a new version lowers the potential. -/
theorem pot_step {s s' : TState α} {x : FreeMonoid α} (h : tallyStep C cut s x = .inl s')
    (htr : s'.tree = s.tree) :
    pot C s' ≤ pot C s ∧ (s'.version ≠ s.version → pot C s' < pot C s
      ∧ s'.n = 0 ∧ s'.dis = 0) := by
  have htr₁ := (tallyPre_cases cut C s x).1
  obtain ⟨hpre, hpre'⟩ := psi_pre C cut s x
  rcases tallyStep_cases cut C h with ⟨e, -, he⟩ | ⟨-, hs⟩
  · cases he
  rcases settleOne_cases cut C hs with ⟨he, -⟩ | ⟨p, c, t, hvi, he | ⟨he, -⟩⟩
  · cases he
    refine ⟨pot_mono C htr₁ hpre, fun hv => ⟨pot_lt C htr₁ hpre (hpre' hv), ?_⟩⟩
    rcases (tallyPre_cases cut C s x).2 with ⟨hv', -⟩ | ⟨-, hn, hd, -⟩
    · exact absurd hv' hv
    · exact ⟨hn, hd⟩
  · cases he
    have hre : (fixEdge cut C.m (tallyPre C cut s x) p c t).tree = (tallyPre C cut s x).tree :=
      htr.trans htr₁.symm
    obtain ⟨hf, hfl⟩ := psi_fix C cut hvi hre
    have hlt : pot C (fixEdge cut C.m (tallyPre C cut s x) p c t) < pot C s := by
      refine lt_of_lt_of_le (pot_lt C hre hf ⟨p, hvi.1, c, hfl⟩) (pot_mono C htr₁ hpre)
    rcases (fixEdge_cases cut C hvi).2 with ⟨-, -, hn, hd, -⟩ | ⟨t₀, w₀, -, -, ht⟩
    · exact ⟨hlt.le, fun _ => ⟨hlt, hn, hd⟩⟩
    · rw [ht, htr₁] at htr; exact absurd htr (DTree.splitAt_ne (htr₁ ▸ hvi.1))
  · cases he

end Pot

section Records

variable (C : TallyCfg) (cut : FreeMonoid α → Option Bool)

open scoped Classical in
theorem tally_pre_le (s : TState α) (x : FreeMonoid α) (p : List Bool) (c : α) (t : List Bool) :
    (tallyPre C cut s x).tally p c t ≤ s.tally p c t
      + (if ∃ sp, recordBy cut C.k (s.tree, s.edges) x = some ((p, c, t), sp) then 1 else 0) := by
  unfold TState.tally
  rcases tallyPre_recs cut C s x with ⟨-, h⟩ | ⟨p', c', t', sp, hr, h⟩
  · rw [h]; omega
  rw [h]
  simp only [TState.addRec]
  by_cases hpc : p = p' ∧ c = c'
  · obtain ⟨rfl, rfl⟩ := hpc
    simp only [Function.update_self, List.filter_append, List.length_append]
    by_cases ht : t = t'
    · subst ht
      rw [if_pos ⟨sp, hr⟩]
      have := List.length_filter_le (fun r : FreeMonoid α × List Bool => decide (r.2 = t)) [(sp, t)]
      simp only [List.length_singleton] at this
      omega
    · have : ([(sp, t')].filter fun r : FreeMonoid α × List Bool => decide (r.2 = t)) = [] := by
        simp [Ne.symm ht]
      rw [this]
      simp
  · by_cases hp : p = p'
    · subst hp
      have hc : c ≠ c' := fun hc => hpc ⟨rfl, hc⟩
      simp [Function.update_of_ne hc]
    · simp [Function.update_of_ne hp]

/-- With no edge violated, a probe's charge violates only its record's edge and target. -/
theorem pre_viol {s : TState α} (hs : Settled C.m s) {x : FreeMonoid α} {p : List Bool} {c : α}
    {t : List Bool} (hv : Violates C.m (tallyPre C cut s x) p c t) :
    ∃ sp, recordBy cut C.k (s.tree, s.edges) x = some ((p, c, t), sp) := by
  classical
  by_contra hno
  have htal := tally_pre_le C cut s x p c t
  rw [if_neg hno, add_zero] at htal
  obtain ⟨hp, hne, hle⟩ := hv
  obtain ⟨htr, hc⟩ := tallyPre_cases cut C s x
  refine hs p c t ⟨htr ▸ hp, ?_, hle.trans htal⟩
  rcases hc with ⟨-, hed, -⟩ | ⟨-, -, -, p₁, c₁, e, hnone, -, hed⟩
  · rwa [hed] at hne
  · by_cases hq : (p, c) = (p₁, c₁)
    · simp only [Prod.mk.injEq] at hq
      obtain ⟨rfl, rfl⟩ := hq
      rw [hnone]
      simp
    · rwa [hed, if_neg hq] at hne

/-- A step at the same tree keeps every edge unviolated. -/
theorem settled_step {s s' : TState α} {x : FreeMonoid α} (hs : Settled C.m s)
    (h : tallyStep C cut s x = .inl s') (htr : s'.tree = s.tree) : Settled C.m s' := by
  have htr₁ := (tallyPre_cases cut C s x).1
  rcases tallyStep_cases cut C h with ⟨e, -, he⟩ | ⟨-, hst⟩
  · cases he
  rcases settleOne_cases cut C hst with ⟨he, hnv⟩ | ⟨p, c, t, hvi, he | ⟨he, -⟩⟩
  · cases he; exact hnv
  · cases he
    rcases (fixEdge_cases cut C hvi).2 with ⟨hft, hr, -, -, hlt, w, hed⟩ | ⟨t₀, w₀, -, -, ht⟩
    swap
    · rw [ht, htr₁] at htr; exact absurd htr (DTree.splitAt_ne (htr₁ ▸ hvi.1))
    obtain ⟨sp, hsp⟩ := pre_viol C cut hs hvi
    have hone : ∀ q c' t', Violates C.m (tallyPre C cut s x) q c' t' → (q, c', t') = (p, c, t) := by
      intro q c' t' hv'
      obtain ⟨sp', hsp'⟩ := pre_viol C cut hs hv'
      rw [hsp] at hsp'
      simp only [Option.some.injEq, Prod.mk.injEq] at hsp'
      obtain ⟨⟨rfl, rfl, rfl⟩, -⟩ := hsp'
      rfl
    intro q c' t' hv'
    obtain ⟨hq, hne, hle⟩ := hv'
    have htal : (fixEdge cut C.m (tallyPre C cut s x) p c t).tally q c' t'
        = (tallyPre C cut s x).tally q c' t' := by
      unfold TState.tally; rw [hr]
    rw [htal] at hle
    rw [hft] at hq
    by_cases hqc : (q, c') = (p, c)
    · simp only [Prod.mk.injEq] at hqc
      obtain ⟨rfl, rfl⟩ := hqc
      rw [hed, if_pos rfl] at hne
      have htt : t' ≠ t := fun h => hne (by simp [h])
      rcases he₀ : (tallyPre C cut s x).edges q c' with _ | ⟨t₀, w₀⟩
      · have := hone q c' t' ⟨hq, by rw [he₀]; simp, hle⟩
        simp only [Prod.mk.injEq] at this
        exact htt this.2.2
      · by_cases ht0 : t' = t₀
        · subst ht0
          have := hlt t' w₀ he₀
          omega
        · have := hone q c' t' ⟨hq, by rw [he₀]; simpa using Ne.symm ht0, hle⟩
          simp only [Prod.mk.injEq] at this
          exact htt this.2.2
    · rw [hed, if_neg hqc] at hne
      have := hone q c' t' ⟨hq, hne, hle⟩
      simp only [Prod.mk.injEq] at this hqc
      exact hqc ⟨this.1, this.2.1⟩
  · cases he

/-- The probes that record against the hypothesis of `s` at an edge and target of `K`. -/
def recIn (K : DTree α → Finset (List Bool × α × List Bool)) (s : TState α) :
    Set (FreeMonoid α) :=
  {x | ∃ pct sp, recordBy cut C.k (s.tree, s.edges) x = some (pct, sp) ∧ pct ∈ K s.tree}

/-- The records the edges and targets of `K` hold, each counted up to `m`. -/
noncomputable def phi (K : DTree α → Finset (List Bool × α × List Bool)) (s : TState α) : ℕ :=
  ∑ pct ∈ K s.tree, min (s.tally pct.1 pct.2.1 pct.2.2) C.m

theorem phi_le (K : DTree α → Finset (List Bool × α × List Bool)) (s : TState α) :
    phi C K s ≤ (K s.tree).card * C.m := by
  unfold phi
  refine (Finset.sum_le_card_nsmul _ _ C.m fun _ _ => min_le_right _ _).trans ?_
  simp

open scoped Classical in
/-- A step at the same tree lowers no count, and its record at an edge and target of `K` raises
one. -/
theorem phi_step (K : DTree α → Finset (List Bool × α × List Bool)) {s s' : TState α}
    {x : FreeMonoid α} (hs : Settled C.m s) (he : EdgesInto s.tree s.edges)
    (h : tallyStep C cut s x = .inl s') (htr : s'.tree = s.tree) :
    phi C K s + (if x ∈ recIn C cut K s then 1 else 0) ≤ phi C K s' := by
  have htal : ∀ p c t, s'.tally p c t = (tallyPre C cut s x).tally p c t := by
    intro p c t; unfold TState.tally; rw [tallyStep_recs C cut h htr]
  have hmono : ∀ pct : List Bool × α × List Bool, min (s.tally pct.1 pct.2.1 pct.2.2) C.m
      ≤ min (s'.tally pct.1 pct.2.1 pct.2.2) C.m := fun pct =>
    min_le_min_right _ (by rw [htal]; exact tally_mono_pre C cut s x _ _ _)
  unfold phi
  rw [htr]
  split_ifs with hx
  · obtain ⟨⟨p, c, t⟩, sp, hsp, hk⟩ := hx
    obtain ⟨hk', hne⟩ := rec_keys cut he hsp
    have hp : p ∈ s.tree.paths := by
      simp only [keysF, Finset.mem_product, List.mem_toFinset] at hk'
      exact hk'.1
    have hlt : s.tally p c t < C.m := by
      by_contra hge
      exact hs p c t ⟨hp, hne, not_lt.1 hge⟩
    have hinc : s.tally p c t + 1 ≤ s'.tally p c t := by
      have := tally_pre C cut s x p c t
      rw [if_pos ⟨sp, hsp⟩] at this
      rw [htal]; exact this
    rw [← Finset.add_sum_erase _ _ hk, ← Finset.add_sum_erase _ _ hk]
    have := Finset.sum_le_sum fun pct (_ : pct ∈ (K s.tree).erase (p, c, t)) => hmono pct
    have : min (s.tally p c t) C.m + 1 ≤ min (s'.tally p c t) C.m := by
      simp only [min_eq_left hlt.le]; omega
    simp only at this ⊢
    omega
  · rw [add_zero]
    exact Finset.sum_le_sum fun pct _ => hmono pct

/-- A step at the same tree keeps edges into its leaves, records at them and every edge
unviolated, lowers no share of the potential, keeps the edges with the version, and otherwise
lowers the potential with a fresh stretch. -/
theorem same_step (hm : 0 < C.m) {s s' : TState α} {x : FreeMonoid α}
    (he : EdgesInto s.tree s.edges) (hr : RecsInto s) (hs : Settled C.m s)
    (h : tallyStep C cut s x = .inl s') (htr : s'.tree = s.tree) :
    EdgesInto s'.tree s'.edges ∧ RecsInto s' ∧ Settled C.m s' ∧ pot C s' ≤ pot C s
      ∧ (s'.version = s.version → s'.edges = s.edges)
      ∧ (s'.version ≠ s.version → pot C s' < pot C s ∧ s'.n = 0 ∧ s'.dis = 0) := by
  obtain ⟨hpl, hpv⟩ := pot_step C cut h htr
  rcases (tallyStep_spec cut C hm he hr).1 _ h with ⟨-, he', hr'⟩ |
    ⟨p₁, c₁, t₁, t₁', -, -, hp, hT', -⟩
  swap
  · exact absurd htr (by rw [hT']; exact DTree.splitAt_ne hp)
  refine ⟨he', hr', settled_step C cut hs h htr, hpl, fun hv => ?_, hpv⟩
  obtain ⟨rfl, -, -, -⟩ := tallyStep_same cut C h hv
  rcases (tallyPre_cases cut C s x).2 with ⟨-, hed, -⟩ | ⟨hv', -⟩
  · exact hed
  · obtain ⟨-, -, -, hv''⟩ := tallyStep_same cut C h hv
    omega

end Records

section Open

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : TallyCfg)
  (cut : FreeMonoid α → Option Bool) (S : ℕ) (D : Measure (FreeMonoid α)) (θr : ℝ)

open scoped Classical in
/-- A probe of a stretch with `b` probes left, of which `r` are to be at hypotheses recording at
least `θr` of the time: it continues at the same version, or the stretch ends at a new version of
the same tree (`some true`), with the sub-round (`some false`), or out of probes (`none`). -/
noncomputable def segR : TState α × ℕ × ℕ → FreeMonoid α →
    (TState α × ℕ × ℕ) ⊕ (Option Bool × (TState α × ℕ × ℕ))
  | (s, 0, r), _ => .inr (none, (s, 0, r))
  | (s, b + 1, r), x =>
    match tallyStep C cut s x with
    | .inr (_, s') => .inr (some false, (s', b, r))
    | .inl s' => if s'.tree = s.tree then
        if s'.version = s.version then
          .inl (s', b, if θr ≤ D.real (recAt C cut s) then r - 1 else r)
        else .inr (some true, (s', b, if θr ≤ D.real (recAt C cut s) then r - 1 else r))
      else .inr (some false, (s', b, r))

theorem segVal_succ' {K S' : Type*} (seg : S' → FreeMonoid α → S' ⊕ (K × S'))
    (w : K → S' → ℕ → ENNReal) (s : S') (j T : ℕ) (xs : Fin (T + 1) → FreeMonoid α) :
    segVal seg w s (j + 1) (T + 1) xs = (seg s (xs 0)).elim
      (fun s' => segVal seg w s' j T (Fin.tail xs)) fun p => w p.1 p.2 T := by
  simp only [segVal]
  rcases seg s (xs 0) with _ | ⟨_, _⟩ <;> rfl

theorem segVal_le {K S' : Type*} (seg : S' → FreeMonoid α → S' ⊕ (K × S'))
    (w : K → S' → ℕ → ENNReal) {M : ENNReal} (h1 : 1 ≤ M) (hw : ∀ k s T, w k s T ≤ M) :
    ∀ j s T xs, segVal seg w s j T xs ≤ M
  | 0, _, _, _ => by simpa [segVal] using h1
  | _ + 1, _, 0, _ => by simpa [segVal] using h1
  | j + 1, s, T + 1, xs => by
    rw [segVal_succ']
    rcases seg s (xs 0) with s' | ⟨k, s'⟩
    · exact segVal_le seg w h1 hw j s' T _
    · exact hw k s' T

theorem termLevel_nonneg {nEnd hP hS : ℕ} {θpt' εd' : ℝ} (hθpt'0 : 0 ≤ θpt')
    (hθpt'1 : θpt' ≤ 1) (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1) :
    0 ≤ termLevel nEnd hP hS θpt' εd' := by
  unfold termLevel
  have t1 := binomSfGe_le_one (n := hS + 1) hθpt'0 hθpt'1 hP
  have t3 := binomSfGe_nonneg (n := nEnd) hεd'0 hεd'1 (hS + 1)
  linarith

open scoped Classical in
theorem no_bad (hm : 0 < C.m) (hLmax : Fintype.card σ + S + 3 ≤ C.Lmax) :
    ∀ (j : ℕ) (s : TState α) (T : ℕ) (xs : Fin T → FreeMonoid α), G.InClass S s.tree →
      EdgesInto s.tree s.edges → RecsInto s → subEnd G C cut s j T xs ≠ .bad := by
  intro j
  induction j with
  | zero => intro s T xs _ _ _; simp [subEnd]
  | succ j ih =>
    intro s T xs hc he hr
    rcases T with _ | T
    · simp [subEnd]
    rw [subEnd_succ]
    rcases hst : tallyStep C cut s (xs 0) with s' | ⟨e, s'⟩
    · by_cases htr : s'.tree = s.tree
      · have hseg : subSeg G C cut s (xs 0) = .inl s' := by simp [subSeg, hst, htr]
        rw [hseg]
        simp only [Sum.elim_inl]
        rcases (tallyStep_spec cut C hm he hr).1 _ hst with ⟨-, he', hr'⟩ |
          ⟨p₁, c₁, t₁, t₁', -, -, hp, hT', -⟩
        · exact ih s' T _ (htr ▸ hc) he' hr'
        · exact absurd htr (by rw [hT']; exact DTree.splitAt_ne hp)
      · have hseg : subSeg G C cut s (xs 0) = .inr (.inr (decide (∃ p d, s'.tree
            = s.tree.splitAt d p ∧ G.GenuineSplit s.tree p d)), s') := by
          simp [subSeg, hst, htr]
        rw [hseg]
        simp only [Sum.elim_inr]
        cases decide (∃ p d, s'.tree = s.tree.splitAt d p ∧ G.GenuineSplit s.tree p d) <;>
          simp [subKind]
    · have hseg : subSeg G C cut s (xs 0) = .inr (.inl e, s') := by simp [subSeg, hst]
      rw [hseg]
      simp only [Sum.elim_inr]
      rcases e with _ | _ | _ | _ | _ <;> simp only [subKind, ne_eq, reduceCtorEq,
        not_false_eq_true]
      obtain ⟨hL, hle⟩ := tallyStep_tooBig cut C hst
      obtain ⟨f, hf, hg⟩ := hc
      have := G.grown_paths hg
      omega

open scoped Classical in
/-- A sub-round from a fresh hypothesis of potential at most `v`, with the edges and targets of `K`
holding `phi` records, is unfinished after `(v + 1) nEnd + r` probes with chance at most `v + 1`
times `termLevel`, and that of fewer than the `|K| m - phi` more records `K` can hold over `r`
probes at rate `p`, given each hypothesis recording at least `θr` of the time records at `K` with
chance at least `p`. -/
theorem open_le [IsProbabilityMeasure D] {nEnd hP hS : ℕ} {θpt' εd' p : ℝ}
    (K : DTree α → Finset (List Bool × α × List Bool)) (hm : 0 < C.m) (hn₀ : C.n₀ ≤ nEnd)
    (hn₀' : C.n₀ ≤ hS + 1) (hθpt0 : 0 ≤ C.θpt) (hθpt1 : C.θpt ≤ 1) (hεd0 : 0 ≤ C.εd)
    (hεd1 : C.εd ≤ 1) (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1) (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1)
    (hcond : θr ≤ (1 - θpt') * εd') (hp0 : 0 ≤ p) (hp1 : p ≤ 1)
    (hhP : binomSfGe (hS + 1) C.θpt hP < C.a) (hhS : 1 - binomSfGe nEnd C.εd (hS + 1) < C.a)
    (hhS' : C.a ≤ binomSfGe nEnd C.εd hS)
    (hK : ∀ s : TState α, G.InClass S s.tree → EdgesInto s.tree s.edges →
      θr ≤ D.real (recAt C cut s) → p ≤ D.real (recIn C cut K s)) :
    ∀ (v : ℕ) (s : TState α) (r B T : ℕ), G.InClass S s.tree → EdgesInto s.tree s.edges →
      RecsInto s → Settled C.m s → s.n = 0 → s.dis = 0 → pot C s ≤ v →
      (v + 1) * nEnd + r ≤ B → B ≤ T →
      (Measure.pi fun _ : Fin T => D) {xs | subEnd G C cut s B T xs = .unfinished}
        ≤ ENNReal.ofReal ((v + 1) * termLevel nEnd hP hS θpt' εd'
          + (1 - binomSfGe r p ((K s.tree).card * C.m - phi C K s + 1))) := by
  set τ := termLevel nEnd hP hS θpt' εd'
  have hτ0 : 0 ≤ τ := termLevel_nonneg hθpt'0 hθpt'1 hεd'0 hεd'1
  set gR : ℕ → ℕ → ℝ := fun r c => 1 - binomSfGe r p (c + 1) with hgR
  have hg0 : ∀ r c, 0 ≤ gR r c := fun r c => sub_nonneg.2 (binomSfGe_le_one hp0 hp1 _)
  have hg1 : ∀ r c, gR r c ≤ 1 := fun r c => by
    have := binomSfGe_nonneg (n := r) hp0 hp1 (c + 1); simp only [gR]; linarith
  have hgc : ∀ r c c', c ≤ c' → gR r c ≤ gR r c' := fun r c c' h => by
    have := binomSfGe_antitone' (n := r) hp0 hp1 (show c + 1 ≤ c' + 1 by omega)
    simp only [gR]; linarith
  have hg00 : ∀ c, gR 0 c = 1 := fun c => by simp [gR, binomSfGe_zero_left]
  set cap : TState α → ℕ := fun s' => (K s'.tree).card * C.m with hcap
  intro v
  induction v using Nat.strong_induction_on with
  | _ v ih =>
  intro s r B T hc he hr hset hn hd hpot hB hBT
  set Inv : TState α → Prop := fun s' => G.InClass S s'.tree ∧ EdgesInto s'.tree s'.edges
    ∧ RecsInto s' ∧ Settled C.m s' ∧ s'.n = 0 ∧ s'.dis = 0
  set w : Option Bool → TState α × ℕ × ℕ → ℕ → ENNReal := fun k sb T' => match k with
    | none => 1
    | some false => 0
    | some true => if Inv sb.1 ∧ pot C sb.1 < v ∧ v * nEnd + sb.2.2 ≤ sb.2.1 ∧ sb.2.1 ≤ T' then
        ENNReal.ofReal (v * τ + gR sb.2.2 (cap sb.1 - phi C K sb.1)) else 1
  set F : TState α × ℕ × ℕ → (T : ℕ) → (Fin T → FreeMonoid α) → Prop :=
    fun sb T xs => subEnd G C cut sb.1 sb.2.1 T xs = .unfinished
  set V : Option Bool → TState α × ℕ × ℕ → (T : ℕ) → (Fin T → FreeMonoid α) → Prop :=
    fun k sb T xs => match k with
    | none => True
    | some false => False
    | some true => F sb T xs
  have hF : ∀ sb T x xs, F sb (T + 1) (Fin.cons x xs)
      ↔ (segR C cut D θr sb x).elim (fun sb' => F sb' T xs) fun q => V q.1 q.2 T xs := by
    rintro ⟨s', b, r'⟩ T' x xs
    rcases b with _ | b
    · simp [F, V, segR, subEnd]
    simp only [F, V]
    rw [subEnd_succ]
    simp only [Fin.cons_zero, Fin.tail_cons, subSeg, segR]
    rcases hst : tallyStep C cut s' x with s'' | ⟨e, s''⟩
    · simp only []
      by_cases htr : s''.tree = s'.tree
      · by_cases hv : s''.version = s'.version
        · simp [htr, hv]
        · simp [htr, hv]
      · simp only [htr, if_false, Sum.elim_inr]
        cases decide (∃ p d, s''.tree = s'.tree.splitAt d p ∧ G.GenuineSplit s'.tree p d) <;>
          simp [subKind]
    · simp only [Sum.elim_inr]
      rcases e with _ | _ | _ | _ | _ <;> simp [subKind]
  have hV : ∀ k sb T', (Measure.pi fun _ : Fin T' => D) {xs | V k sb T' xs} ≤ w k sb T' := by
    rintro (_ | _ | _) ⟨s', b, r'⟩ T'
    · simp only [V, w, Set.setOf_true]; exact prob_le_one
    · simp [V, w]
    · simp only [V, w]
      split_ifs with hcond'
      · obtain ⟨⟨hc', he', hr', hs', hn', hd'⟩, hlt, hb, hbT⟩ := hcond'
        have := ih (v - 1) (by omega) s' r' b T' hc' he' hr' hs' hn' hd' (by omega)
          (by rw [show v - 1 + 1 = v by omega]; exact hb) hbT
        rwa [show ((v - 1 : ℕ) : ℝ) + 1 = v by
          rw [Nat.cast_sub (by omega)]; push_cast; ring] at this
      · exact prob_le_one
  have hseg := fun j => seg_le D (segR C cut D θr) F V w hF hV T (s, B, r) j
  have hrecAt : ∀ s' : TState α, s'.tree = s.tree → s'.edges = s.edges →
      recAt C cut s' = recAt C cut s := by
    intro s' htr hed; simp only [recAt, htr, hed]
  have hvn : (v + 1) * nEnd = v * nEnd + nEnd := by ring
  by_cases htype : θr ≤ D.real (recAt C cut s)
  swap
  · have hpt : ∀ j (s' : TState α) (b' T' : ℕ) (xs : Fin T' → FreeMonoid α), s'.tree = s.tree →
        s'.edges = s.edges → EdgesInto s'.tree s'.edges → RecsInto s' → Settled C.m s' →
        pot C s' ≤ v → phi C K s ≤ phi C K s' → v * nEnd + r + j ≤ b' → b' ≤ T' →
        segVal (segR C cut D θr) w (s', b', r) j T' xs
          ≤ (if KeepV C cut s' j T' xs then 1 else 0)
            + ENNReal.ofReal (v * τ + gR r (cap s - phi C K s)) := by
      intro j
      induction j with
      | zero => intro s' b' T' xs _ _ _ _ _ _ _ _ _; simp [segVal, KeepV]
      | succ j ihj =>
        intro s' b' T' xs htr hed he' hr' hs' hpot' hphi hb hbT
        rcases b' with _ | b'
        · omega
        rcases T' with _ | T'
        · omega
        rw [segVal_succ']
        have hty : ¬ θr ≤ D.real (recAt C cut s') := by rwa [hrecAt s' htr hed]
        rcases hst : tallyStep C cut s' (xs 0) with s'' | ⟨e, s''⟩
        · by_cases htr' : s''.tree = s'.tree
          · obtain ⟨he'', hr'', hs'', hpl, hved, hpv⟩ := same_step C cut hm he' hr' hs' hst htr'
            have hph := phi_step C cut K hs' he' hst htr'
            by_cases hv : s''.version = s'.version
            · have hsv : segR C cut D θr (s', b' + 1, r) (xs 0) = .inl (s'', b', r) := by
                simp [segR, hst, htr', hv, hty]
              rw [hsv]
              simp only [Sum.elim_inl]
              refine (ihj s'' b' T' (Fin.tail xs) (htr'.trans htr) ((hved hv).trans hed) he''
                hr'' hs'' (hpl.trans hpot') (hphi.trans (by omega)) (by omega)
                (by omega)).trans (add_le_add ?_ le_rfl)
              split_ifs with h1 h2
              · exact le_rfl
              · exact absurd ⟨s'', hst, hv, h1⟩ h2
              · exact zero_le
              · exact le_rfl
            · have hsv : segR C cut D θr (s', b' + 1, r) (xs 0)
                  = .inr (some true, (s'', b', r)) := by
                simp [segR, hst, htr', hv, hty]
              rw [hsv]
              simp only [Sum.elim_inr, w]
              obtain ⟨hlt, hn'', hd''⟩ := hpv hv
              rw [if_pos ⟨⟨(htr'.trans htr) ▸ hc, he'', hr'', hs'', hn'', hd''⟩, by omega,
                by omega, by omega⟩]
              refine le_trans ?_ le_add_self
              refine ENNReal.ofReal_le_ofReal (add_le_add le_rfl (hgc r _ _ ?_))
              have : cap s'' = cap s := by simp only [cap, htr', htr]
              omega
          · have hsv : segR C cut D θr (s', b' + 1, r) (xs 0)
                = .inr (some false, (s'', b', r)) := by
              simp [segR, hst, htr']
            rw [hsv]
            simp [w]
        · have hsv : segR C cut D θr (s', b' + 1, r) (xs 0)
              = .inr (some false, (s'', b', r)) := by
            simp [segR, hst]
          rw [hsv]
          simp [w]
    have hkeep := keep_le C cut D hn₀ hn₀' hθpt0 hθpt1 hεd0 hεd1 hθpt'0 hθpt'1 hεd'0 hεd'1 hcond
      hhP hhS hhS' (not_le.1 htype) hn hd T
    have hB' : v * nEnd + r + nEnd ≤ B := by
      have : (v + 1) * nEnd = v * nEnd + nEnd := by ring
      omega
    have hg := hg0 r (cap s - phi C K s)
    calc (Measure.pi fun _ : Fin T => D) {xs | subEnd G C cut s B T xs = .unfinished}
        = (Measure.pi fun _ : Fin T => D) {xs | F (s, B, r) T xs} := rfl
      _ ≤ _ := hseg nEnd
      _ ≤ ∫⁻ xs, ((if KeepV C cut s nEnd T xs then 1 else 0)
            + ENNReal.ofReal (v * τ + gR r (cap s - phi C K s)))
            ∂(Measure.pi fun _ : Fin T => D) :=
          lintegral_mono fun xs => hpt nEnd s B T xs rfl rfl he hr hset hpot le_rfl hB' hBT
      _ = (Measure.pi fun _ : Fin T => D) {xs | KeepV C cut s nEnd T xs}
            + ENNReal.ofReal (v * τ + gR r (cap s - phi C K s)) := by
          rw [lintegral_add_left (measurable_of_countable _), lintegral_ite_one, lintegral_const,
            measure_univ, mul_one]
      _ ≤ ENNReal.ofReal τ + ENNReal.ofReal (v * τ + gR r (cap s - phi C K s)) :=
          add_le_add hkeep le_rfl
      _ = _ := by
          rw [← ENNReal.ofReal_add hτ0 (by positivity)]
          congr 1
          simp only [gR, cap]
          ring
  · have hwM : ∀ k sb T', w k sb T' ≤ ENNReal.ofReal (v * τ + 1) := by
      have hvt : 0 ≤ (v : ℝ) * τ := by positivity
      rintro (_ | _ | _) sb T'
      · simp only [w]; rw [← ENNReal.ofReal_one]; exact ENNReal.ofReal_le_ofReal (by linarith)
      · simp [w]
      · simp only [w]
        split_ifs
        · exact ENNReal.ofReal_le_ofReal (by linarith [hg1 sb.2.2 (cap sb.1 - phi C K sb.1)])
        · rw [← ENNReal.ofReal_one]; exact ENNReal.ofReal_le_ofReal (by linarith)
    have h1M : (1 : ENNReal) ≤ ENNReal.ofReal (v * τ + 1) := by
      have hvt : 0 ≤ (v : ℝ) * τ := by positivity
      rw [← ENNReal.ofReal_one]; exact ENNReal.ofReal_le_ofReal (by linarith)
    have key : ∀ j (s' : TState α) (b' r' T' : ℕ), s'.tree = s.tree → s'.edges = s.edges →
        EdgesInto s'.tree s'.edges → RecsInto s' → Settled C.m s' → pot C s' ≤ v → b' < j →
        (v + 1) * nEnd + r' ≤ b' → b' ≤ T' →
        ∫⁻ xs, segVal (segR C cut D θr) w (s', b', r') j T' xs ∂(Measure.pi fun _ : Fin T' => D)
          ≤ ENNReal.ofReal (v * τ + gR r' (cap s' - phi C K s')) := by
      intro j
      induction j with
      | zero => intro s' b' r' T' _ _ _ _ _ _ hbj; omega
      | succ j ihj =>
        intro s' b' r' T' htr hed he' hr' hs' hpot' hbj hb hbT
        rcases r' with _ | r''
        · rw [hg00]
          refine (lintegral_mono fun xs => segVal_le (segR C cut D θr) w h1M hwM _ _ _ xs).trans ?_
          rw [lintegral_const, measure_univ, mul_one]
        obtain ⟨b'', rfl⟩ : ∃ b'', b' = b'' + 1 := ⟨b' - 1, by omega⟩
        obtain ⟨T'', rfl⟩ : ∃ T'', T' = T'' + 1 := ⟨T' - 1, by omega⟩
        have hty : θr ≤ D.real (recAt C cut s') := by rwa [hrecAt s' htr hed]
        have hrec : p ≤ D.real (recIn C cut K s') := hK s' (htr ▸ hc) he' hty
        set c := cap s' - phi C K s' with hc_def
        set A := 1 - binomSfGe r'' p c with hA_def
        set Bv := gR r'' c with hBv_def
        have hA0 : 0 ≤ A := sub_nonneg.2 (binomSfGe_le_one hp0 hp1 _)
        have hAB : A ≤ Bv := by
          have := binomSfGe_antitone (n := r'') hp0 hp1 c
          simp only [A, Bv, gR]; linarith
        set E := recIn C cut K s' with hE_def
        have hcaseval : ∀ x (s'' : TState α), tallyStep C cut s' x = .inl s'' →
            s''.tree = s'.tree →
            ENNReal.ofReal (v * τ + gR r'' (cap s'' - phi C K s''))
              ≤ E.indicator (fun _ => ENNReal.ofReal (v * τ + A)) x
                + Eᶜ.indicator (fun _ => ENNReal.ofReal (v * τ + Bv)) x := by
          intro x s'' hst htr'
          have hph := phi_step C cut K hs' he' hst htr'
          have hcap' : cap s'' = cap s' := by simp only [cap, htr']
          have hle := phi_le C K s''
          by_cases hx : x ∈ E
          · rw [Set.indicator_of_mem hx, Set.indicator_of_notMem (by simpa using hx), add_zero]
            refine ENNReal.ofReal_le_ofReal (add_le_add le_rfl ?_)
            rw [if_pos hx] at hph
            have h1 : cap s'' - phi C K s'' + 1 ≤ c := by
              have hle' : phi C K s'' ≤ cap s'' := hle
              rw [hc_def, ← hcap']; omega
            have := binomSfGe_antitone' (n := r'') hp0 hp1 h1
            simp only [gR, A]; linarith
          · rw [Set.indicator_of_notMem hx, Set.indicator_of_mem (by simpa using hx), zero_add]
            refine ENNReal.ofReal_le_ofReal (add_le_add le_rfl (hgc r'' _ _ ?_))
            rw [if_neg hx] at hph
            rw [hc_def, ← hcap']; omega
        have hstep : ∀ x, ∫⁻ xs, segVal (segR C cut D θr) w (s', b'' + 1, r'' + 1) (j + 1)
            (T'' + 1) (Fin.cons x xs) ∂(Measure.pi fun _ : Fin T'' => D)
            ≤ E.indicator (fun _ => ENNReal.ofReal (v * τ + A)) x
              + Eᶜ.indicator (fun _ => ENNReal.ofReal (v * τ + Bv)) x := by
          intro x
          simp only [segVal_succ', Fin.cons_zero, Fin.tail_cons]
          rcases hst : tallyStep C cut s' x with s'' | ⟨e, s''⟩
          · by_cases htr' : s''.tree = s'.tree
            · obtain ⟨he'', hr'', hs'', hpl, hved, hpv⟩ := same_step C cut hm he' hr' hs' hst htr'
              by_cases hv : s''.version = s'.version
              · have hsv : segR C cut D θr (s', b'' + 1, r'' + 1) x = .inl (s'', b'', r'') := by
                  simp [segR, hst, htr', hv, hty]
                rw [hsv]
                simp only [Sum.elim_inl]
                exact (ihj s'' b'' r'' T'' (htr'.trans htr) ((hved hv).trans hed) he'' hr'' hs''
                  (hpl.trans hpot') (by omega) (by omega) (by omega)).trans
                  (hcaseval x s'' hst htr')
              · have hsv : segR C cut D θr (s', b'' + 1, r'' + 1) x
                    = .inr (some true, (s'', b'', r'')) := by
                  simp [segR, hst, htr', hv, hty]
                rw [hsv]
                simp only [Sum.elim_inr, w]
                obtain ⟨hlt, hn'', hd''⟩ := hpv hv
                rw [if_pos ⟨⟨(htr'.trans htr) ▸ hc, he'', hr'', hs'', hn'', hd''⟩, by omega,
                  by omega, by omega⟩]
                rw [lintegral_const, measure_univ, mul_one]
                exact hcaseval x s'' hst htr'
            · have hsv : segR C cut D θr (s', b'' + 1, r'' + 1) x
                  = .inr (some false, (s'', b'', r'' + 1)) := by
                simp [segR, hst, htr']
              rw [hsv]
              simp [w]
          · have hsv : segR C cut D θr (s', b'' + 1, r'' + 1) x
                = .inr (some false, (s'', b'', r'' + 1)) := by
              simp [segR, hst]
            rw [hsv]
            simp [w]
        rw [lintegral_pi_succ]
        refine (lintegral_mono hstep).trans ?_
        have hE0 : 0 ≤ D.real E := measureReal_nonneg
        have hE1 : D.real E ≤ 1 := by
          have := measureReal_mono (μ := D) (Set.subset_univ E); rwa [probReal_univ] at this
        have hvt : 0 ≤ (v : ℝ) * τ := by positivity
        rw [lintegral_add_left (measurable_of_countable _), lintegral_indicator
            (Set.to_countable _).measurableSet, lintegral_indicator
            (Set.to_countable _).measurableSet,
          setLIntegral_const, setLIntegral_const, ← ofReal_measureReal, ← ofReal_measureReal,
          measureReal_compl (Set.to_countable _).measurableSet, probReal_univ,
          ← ENNReal.ofReal_mul (by linarith), ← ENNReal.ofReal_mul (by linarith),
          ← ENNReal.ofReal_add (mul_nonneg (by linarith) hE0)
            (mul_nonneg (by linarith) (by linarith))]
        refine ENNReal.ofReal_le_ofReal ?_
        have hrecur : gR (r'' + 1) c = p * A + (1 - p) * Bv := by
          simp only [gR, A, Bv, binomSfGe_succ]; ring
        rw [hrecur]
        nlinarith [mul_le_mul_of_nonneg_right hrec (sub_nonneg.2 hAB)]
    calc (Measure.pi fun _ : Fin T => D) {xs | subEnd G C cut s B T xs = .unfinished}
        = (Measure.pi fun _ : Fin T => D) {xs | F (s, B, r) T xs} := rfl
      _ ≤ _ := hseg (B + 1)
      _ ≤ ENNReal.ofReal (v * τ + gR r (cap s - phi C K s)) :=
          key (B + 1) s B r T rfl rfl he hr hset hpot (by omega) hB hBT
      _ ≤ _ := by
          refine ENNReal.ofReal_le_ofReal ?_
          simp only [gR, cap]
          nlinarith [hτ0]

end Open

open scoped Classical in
theorem sub_round_holds : SubRound := by
  intro α _ _ σ _ G rd D _ C S nEnd nRec hP hS ρ θg θgs θgpt θpt' θr εd' hρ0 hρ1 hm hLmax hn₀
    hn₀' hθpt0 hθpt1 hεd0 hεd1 hθpt'0 hθpt'1 hθr0 hθr1 hεd'0 hεd'1 hcond hhP hhS hhS' hE
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
  set Tsub := subT C (Fintype.card α) nEnd nRec
  intro s hs T hT
  obtain ⟨hc, he, hrecs, hn, hd⟩ := hs
  have hr : RecsInto s := fun q c r h => by rw [hrecs q c] at h; cases h
  have hpaths : s.tree.paths.length ≤ C.Lmax := by
    obtain ⟨f, hf, hg⟩ := hc
    have := G.grown_paths hg
    omega
  have hcard : (keysF s.tree).card ≤ C.Lmax ^ 2 * Fintype.card α := by
    simp only [keysF, Finset.card_product, Finset.card_univ]
    have := (List.toFinset_card_le s.tree.paths).trans hpaths
    calc _ ≤ C.Lmax * (Fintype.card α * C.Lmax) := by gcongr
      _ = C.Lmax ^ 2 * Fintype.card α := by ring
  constructor
  · rw [ENNReal.ofReal_zero, zero_mul, zero_add]
    have hsub : {xs : Fin T → FreeMonoid α | subEnd G C cut s Tsub T xs = .fake}
        ⊆ ⋃ pct ∈ keysF s.tree, {xs | EvF G C cut S pct s Tsub C.m T xs} := by
      intro xs hx
      obtain ⟨pct, hpct, hev⟩ := fake_incl G C cut S hm Tsub s T xs hc he hr hx
      have h0 : badRecs G s pct = 0 := by simp [badRecs, hrecs]
      rw [h0, Nat.sub_zero] at hev
      exact Set.mem_biUnion hpct hev
    refine (measure_mono hsub).trans ((measure_biUnion_finset_le _ _).trans ?_)
    have hb : ∀ pct ∈ keysF s.tree, (Measure.pi fun _ : Fin T => D)
        {xs | EvF G C cut S pct s Tsub C.m T xs} ≤ ENNReal.ofReal (binomSfGe Tsub ρ C.m) :=
      fun pct _ => fake_bin G C cut S D pct hρ0 hρ1
        (fun s' hc' he' => hE.spurious s'.tree s'.edges hc' he' pct) T Tsub s C.m
    have hsf := binomSfGe_nonneg (n := Tsub) hρ0 hρ1 C.m
    calc _ ≤ ∑ pct ∈ keysF s.tree, ENNReal.ofReal (binomSfGe Tsub ρ C.m) := Finset.sum_le_sum hb
      _ = ENNReal.ofReal ((keysF s.tree).card * binomSfGe Tsub ρ C.m) := by
          rw [Finset.sum_const, nsmul_eq_mul, ENNReal.ofReal_mul (Nat.cast_nonneg _),
            ENNReal.ofReal_natCast]
      _ ≤ _ := by
          refine ENNReal.ofReal_le_ofReal ?_
          unfold subFake
          have : ((keysF s.tree).card : ℝ) ≤ C.Lmax ^ 2 * Fintype.card α := by exact_mod_cast hcard
          nlinarith
  · have hsub : {xs : Fin T → FreeMonoid α | subEnd G C cut s Tsub T xs = .bad
        ∨ subEnd G C cut s Tsub T xs = .unfinished}
        ⊆ {xs | subEnd G C cut s Tsub T xs = .unfinished} :=
      fun xs h => h.resolve_left (no_bad G C cut S hm hLmax Tsub s T xs hc he hr)
    refine (measure_mono hsub).trans ?_
    have hpot : pot C s ≤ subVersions C (Fintype.card α) := by
      refine (pot_le C s).trans ?_
      unfold subVersions
      gcongr
    have hset : Settled C.m s := fun p c t hv => by
      have := hv.2.2
      simp [TState.tally, hrecs] at this
      omega
    have hphi : phi C keysF s = 0 := by simp [phi, TState.tally, hrecs]
    have hK : ∀ s' : TState α, G.InClass S s'.tree → EdgesInto s'.tree s'.edges →
        θr ≤ D.real (recAt C cut s') → θr ≤ D.real (recIn C cut keysF s') := by
      intro s' _ he' h
      refine h.trans (measureReal_mono fun x hx => ?_)
      obtain ⟨pct, sp, h'⟩ := hx
      exact ⟨pct, sp, h', (rec_keys cut he' h').1⟩
    refine (open_le G C cut S D θr keysF hm hn₀ hn₀' hθpt0 hθpt1 hεd0 hεd1 hθpt'0 hθpt'1 hεd'0
      hεd'1 hcond hθr0 hθr1 hhP hhS hhS' hK _ s nRec Tsub T hc he hr hset hn hd hpot le_rfl
      hT).trans ?_
    refine ENNReal.ofReal_le_ofReal ?_
    rw [hphi, Nat.sub_zero]
    unfold subOpen
    have : binomSfGe nRec θr (C.Lmax ^ 2 * Fintype.card α * C.m + 1)
        ≤ binomSfGe nRec θr ((keysF s.tree).card * C.m + 1) :=
      binomSfGe_antitone' hθr0 hθr1 (by have := Nat.mul_le_mul_right C.m hcard; omega)
    push_cast
    linarith

end OrthoDFA
