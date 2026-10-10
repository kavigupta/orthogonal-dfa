import OrthoDFA.Proofs.TallyStretch

/-!
# The tally round's invariant

While fewer than `m` records that are not true have been made, the tree stays genuine: a split
needs `m` records at each of two targets, so a true one at each, and true records at two
targets make the split genuine. The edges and the records point at leaves, the records at each
edge that are not true number at most the probes that made them, and `pot` stays within
`versionCap`.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ]

namespace DTree

theorem mem_paths_splitAt (d : FreeMonoid α) :
    ∀ (T : DTree α) (p q : List Bool), q ∈ T.paths → q ≠ p → q ∈ (T.splitAt d p).paths
  | .leaf, [], q, hq, hne => by simp [paths] at hq; exact absurd hq hne
  | .leaf, _ :: _, q, hq, _ => by simpa [splitAt] using hq
  | .node _ _ _, [], q, hq, _ => by simpa [splitAt] using hq
  | .node n r a, false :: p, q, hq, hne => by
    simp only [splitAt, paths, List.mem_append, List.mem_map] at hq ⊢
    rcases hq with ⟨q', hq', rfl⟩ | ⟨q', hq', rfl⟩
    · exact .inl ⟨q', mem_paths_splitAt d r p q' hq' (fun h => hne (by rw [h])), rfl⟩
    · exact .inr ⟨q', hq', rfl⟩
  | .node n r a, true :: p, q, hq, hne => by
    simp only [splitAt, paths, List.mem_append, List.mem_map] at hq ⊢
    rcases hq with ⟨q', hq', rfl⟩ | ⟨q', hq', rfl⟩
    · exact .inl ⟨q', hq', rfl⟩
    · exact .inr ⟨q', mem_paths_splitAt d a p q' hq' (fun h => hne (by rw [h])), rfl⟩

end DTree

variable (G : ReadModel α σ) (C : TallyCfg) (rd : FreeMonoid α → ARU)

/-- The reads' cut. -/
abbrev rdCut (rd : FreeMonoid α → ARU) : FreeMonoid α → Option Bool := fun z => (rd z).cut

open scoped Classical in
/-- The invariant, with at most `K` records made that are not true. -/
def TallyInv (K : ℕ) (s : TState α) : Prop :=
  G.Genuine s.tree ∧ EdgesInto s.tree s.edges ∧ (∀ q c r, r ∈ s.recs q c → r.2 ∈ s.tree.paths)
    ∧ (∀ q c, ((s.recs q c).filter fun r => ¬ G.TrueRec s.tree (q, c, r.2) r.1).length ≤ K)
    ∧ s.pot C.m C.Lmax ≤ versionCap C (Fintype.card α)

open scoped Classical in
/-- The probes whose record is not true, at a genuine hypothesis with edges into its leaves. -/
def spurSet (s : TState α) : Set (FreeMonoid α) :=
  if G.Genuine s.tree ∧ EdgesInto s.tree s.edges then
    {x | ∃ pct sp, recordBy (rdCut rd) C.k (s.tree, s.edges) x = some (pct, sp)
      ∧ ¬ G.TrueRec s.tree pct sp}
  else ∅

theorem tallyStart_inv (hL : 2 ≤ C.Lmax) : TallyInv G C 0 (tallyStart : TState α) := by
  refine ⟨.start, fun _ _ _ _ h => by simp [tallyStart] at h, fun _ _ _ h => by
    simp [tallyStart] at h, fun _ _ => by simp [tallyStart], ?_⟩
  have h1 := unl_le (tallyStart : TState α)
  have h2 := uns_le C.m (tallyStart : TState α)
  have hp : (tallyStart : TState α).tree.paths.length = 2 := by simp [tallyStart, DTree.paths]
  rw [hp] at h1 h2
  unfold TState.pot versionCap
  rw [hp]
  obtain ⟨L', hL'⟩ : ∃ L', C.Lmax = L' + 2 := ⟨C.Lmax - 2, by omega⟩
  rw [hL', show L' + 2 - 2 = L' by omega]
  nlinarith

theorem TrueRec.splitAt {T : DTree α} {p : List Bool} (hp : p ∈ T.paths) (d : FreeMonoid α)
    {q : List Bool} {c : α} {t : List Bool} {sp : FreeMonoid α} (h : G.TrueRec T (q, c, t) sp)
    (hq : q ≠ p) (ht : t ≠ p) : G.TrueRec (T.splitAt d p) (q, c, t) sp := by
  obtain ⟨l₁, l₂⟩ := h
  have h1 := (G.leafOf_splitAt d T p hp _).1 (by rw [l₁]; exact hq)
  have h2 := (G.leafOf_splitAt d T p hp _).1 (by rw [l₂]; exact ht)
  exact ⟨h1.trans l₁, h2.trans l₂⟩

open scoped Classical in
theorem exists_trueRec {T : DTree α} {q : List Bool} {c : α} {t : List Bool}
    {l : List (FreeMonoid α × List Bool)} {m K : ℕ} (hm : m ≤ (l.filter (·.2 = t)).length)
    (hK : (l.filter fun r => ¬ G.TrueRec T (q, c, r.2) r.1).length ≤ K) (hKm : K < m) :
    ∃ sp, G.TrueRec T (q, c, t) sp := by
  classical
  by_contra hno
  push_neg at hno
  have : (l.filter (·.2 = t)).length
      ≤ (l.filter fun r => ¬ G.TrueRec T (q, c, r.2) r.1).length := by
    rw [← List.countP_eq_length_filter, ← List.countP_eq_length_filter]
    refine List.countP_mono_left fun r _ hr => ?_
    simp only [decide_eq_true_eq] at hr ⊢
    rw [hr]; exact hno r.1
  omega

open scoped Classical in
/-- A fix keeps the invariant while fewer than `m` records that are not true were made: a split
is genuine, and only drops records; a redirect points at a target some record has. -/
theorem fix_inv (cut : FreeMonoid α → Option Bool) {K : ℕ} (hK : K < C.m)
    (hLmax : Fintype.card σ + 2 ≤ C.Lmax) {s : TState α} (hI : TallyInv G C K s)
    {p : List Bool} {c : α} {t : List Bool} (hv : Violates C.m s p c t) :
    TallyInv G C K (fixEdge cut C.m s p c t)
      ∧ (fixEdge cut C.m s p c t).tree.paths.length ≤ C.Lmax := by
  obtain ⟨hg, he, hr, hc, hpot⟩ := hI
  have hv' := hv
  obtain ⟨hp, hne, ht⟩ := hv
  rcases fixEdge_cases C.m cut s p c t with ⟨t₀, w₀, he0, ht0, hfix⟩ | ⟨hn, hfix⟩
  · have htt0 : t ≠ t₀ := fun h => hne (by rw [he0, h]; rfl)
    obtain ⟨sp₁, h₁⟩ := exists_trueRec G ht (hc p c) hK
    obtain ⟨sp₂, h₂⟩ := exists_trueRec G ht0 (hc p c) hK
    have hgs := G.genuineSplit_of_recs h₁ h₂ htt0
    set T' := s.tree.splitAt (FreeMonoid.of c * s.tree.midAt (lcp t t₀)) p
    have hg' : G.Genuine T' := ReadModel.Genuine.split hg hgs
    have hL' : T'.paths.length ≤ C.Lmax := (G.genuine_paths hg').trans hLmax
    have hpot' := pot_fixEdge C.m C.Lmax cut s hv' (by rw [hfix]; exact hL')
    rw [hfix] at hpot' ⊢
    refine ⟨⟨hg', ?_, ?_, ?_, by omega⟩, hL'⟩
    · intro q e t' w h
      simp only [] at h
      split_ifs at h
      unfold retargetBy at h
      rcases hq : s.edges q e with _ | ⟨s', w'⟩ <;> rw [hq] at h
      · simp at h
      · simp only [] at h
        split_ifs at h with hs'
        · rcases hsift : T'.sift cut (w' * FreeMonoid.of e) with s'' | b <;> rw [hsift] at h
          · simp only [Sum.elim_inl, Option.some.injEq, Prod.mk.injEq] at h
            obtain ⟨rfl, -⟩ := h
            exact DTree.sift_mem_paths _ _ _ hsift
          · simp at h
        · simp only [Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, -⟩ := h
          exact DTree.mem_paths_splitAt _ _ _ _ (he _ _ _ _ hq) hs'
    · intro q e r h
      simp only [] at h
      split_ifs at h
      · simp at h
      · obtain ⟨h1, h2⟩ := List.mem_filter.1 h
        simp only [decide_eq_true_eq] at h2
        exact DTree.mem_paths_splitAt _ _ _ _ (hr _ _ _ h1) h2
    · intro q e
      simp only []
      split_ifs with hpq
      · simp
      · refine le_trans ?_ (hc q e)
        rw [List.filter_filter, ← List.countP_eq_length_filter, ← List.countP_eq_length_filter]
        refine List.countP_mono_left fun r _ h => ?_
        simp only [Bool.and_eq_true, decide_eq_true_eq] at h ⊢
        obtain ⟨h1, h2⟩ := h
        exact fun htr => h1 (TrueRec.splitAt G hp _ htr (fun h => hpq (h ▸ List.prefix_refl _)) h2)
  · rw [hfix]
    set w := (((s.recs p c).find? (·.2 = t)).map Prod.fst).getD 1
    have hpot' := pot_redirect C.m C.Lmax s (w := w) hp hn ht
    have htp : t ∈ s.tree.paths := by
      have : 0 < ((s.recs p c).filter (·.2 = t)).length := by
        unfold TState.tally at ht; omega
      obtain ⟨r, hr'⟩ := List.exists_mem_of_length_pos this
      obtain ⟨h1, h2⟩ := List.mem_filter.1 hr'
      simp only [decide_eq_true_eq] at h2
      exact h2 ▸ hr _ _ _ h1
    refine ⟨⟨hg, ?_, hr, hc, by omega⟩, (G.genuine_paths hg).trans hLmax⟩
    intro q e t' w' h
    rw [setEdge_eq] at h
    split_ifs at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, -⟩ := h
      exact htp
    · exact he _ _ _ _ h

open scoped Classical in
/-- A probe keeps the invariant, counting itself where its record is not true. -/
theorem tallyPre_inv {K : ℕ} {s : TState α} (hI : TallyInv G C K s) (x : FreeMonoid α) :
    TallyInv G C (K + if x ∈ spurSet G C rd s then 1 else 0) (tallyPre C (rdCut rd) s x) := by
  obtain ⟨hg, he, hr, hc, hpot⟩ := hI
  have hpot' := (tallyPre_spec C.Lmax (rdCut rd) C s x).2.2.1
  rcases tallyPre_cases (rdCut rd) C s x with ⟨p, c, t, u, hp, htp, hen, h⟩ |
    ⟨s₂, ht, hed, hrs, hv, h | ⟨p, c, r, hrec, h⟩⟩
  · rw [h] at hpot' ⊢
    refine ⟨hg, ?_, hr, fun q e => (hc q e).trans (by omega), by omega⟩
    intro q e t' w' h'
    rw [setEdge_eq] at h'
    split_ifs at h'
    · simp only [Option.some.injEq, Prod.mk.injEq] at h'
      obtain ⟨rfl, -⟩ := h'
      exact htp
    · exact he _ _ _ _ h'
  · rw [h] at hpot' ⊢
    refine ⟨ht ▸ hg, ht ▸ hed ▸ he, fun q e r' h' => ht ▸ hr q e r' (hrs ▸ h'),
      fun q e => ?_, by omega⟩
    rw [hrs, ht]
    exact (hc q e).trans (by omega)
  · rw [h] at hpot' ⊢
    have hrec' := hrec
    obtain ⟨-, -, -, -, -, hsift, -⟩ := recordBy_spec (rdCut rd) hrec'
    have htp : r.2 ∈ s.tree.paths := DTree.sift_mem_paths _ _ _ hsift
    refine ⟨ht ▸ hg, ht ▸ hed ▸ he, ?_, ?_, by
      simp only [show (s₂.addRec p c r).version = s₂.version from rfl] at hpot'; omega⟩
    · intro q e r' h'
      simp only [TState.addRec, show (s₂.addRec p c r).tree = s₂.tree from rfl] at h' ⊢
      rw [ht]
      by_cases hq : q = p
      · subst hq
        by_cases hce : e = c
        · subst hce
          simp only [Function.update_self, List.mem_append, List.mem_singleton] at h'
          rcases h' with h' | rfl
          · exact hr _ _ _ (hrs ▸ h')
          · exact htp
        · simp only [Function.update_self, Function.update_of_ne hce] at h'
          exact hr _ _ _ (hrs ▸ h')
      · simp only [Function.update_of_ne hq] at h'
        exact hr _ _ _ (hrs ▸ h')
    · intro q e
      simp only [TState.addRec, show (s₂.addRec p c r).tree = s₂.tree from rfl, ht]
      by_cases hqe : q = p ∧ e = c
      · obtain ⟨rfl, rfl⟩ := hqe
        simp only [Function.update_self, List.filter_append, List.length_append, hrs]
        have h1 := hc q e
        by_cases htr : G.TrueRec s.tree (q, e, r.2) r.1
        · have : ([r].filter fun r' => decide ¬ G.TrueRec s.tree (q, e, r'.2) r'.1) = [] := by
            simp [htr]
          rw [this, List.length_nil, add_zero]
          omega
        · have hx : x ∈ spurSet G C rd s := by
            simp only [spurSet, if_pos (And.intro hg he)]
            exact ⟨_, _, hrec, htr⟩
          have : ([r].filter fun r' => decide ¬ G.TrueRec s.tree (q, e, r'.2) r'.1) = [r] := by
            simp [htr]
          rw [this, List.length_singleton, if_pos hx]
          omega
      · have : (Function.update s₂.recs p (Function.update (s₂.recs p) c (s₂.recs p c ++ [r])))
            q e = s.recs q e := by
          by_cases hq : q = p
          · subst hq
            have hce : e ≠ c := fun h => hqe ⟨rfl, h⟩
            simp [Function.update_of_ne hce, hrs]
          · simp [Function.update_of_ne hq, hrs]
        rw [this]
        exact (hc q e).trans (by omega)

theorem tallyLook_cases {s : TState α} {e : TEnd α} (h : tallyLook C s = some e) :
    e = .success ∨ e = .harvestStart ∨ ∃ e', e = .harvest e' := by
  unfold tallyLook at h
  split_ifs at h <;> simp only [Option.some.injEq] at h <;> subst h <;> simp

open scoped Classical in
/-- A step keeps the invariant, counting the probe where its record is not true: it continues,
`pot` falling by the versions it adds, the hypothesis kept or a fresh stretch begun; or it ends
in success or a harvest, at the hypothesis it had. -/
theorem step_spec {K : ℕ} {s : TState α} (x : FreeMonoid α)
    (hK : K + (if x ∈ spurSet G C rd s then 1 else 0) < C.m)
    (hLmax : Fintype.card σ + 2 ≤ C.Lmax) (hfuel : versionCap C (Fintype.card α) ≤ C.fuel)
    (hn₀ : 1 ≤ C.n₀) (hI : TallyInv G C K s) :
    (∃ s', tallyStep C (rdCut rd) s x = .inl s'
      ∧ TallyInv G C (K + if x ∈ spurSet G C rd s then 1 else 0) s' ∧ s.version ≤ s'.version
      ∧ s'.pot C.m C.Lmax + (s'.version - s.version) ≤ s.pot C.m C.Lmax
      ∧ (s'.version = s.version → tallyKey s' = tallyKey s)
      ∧ (s'.version ≠ s.version → s'.Fresh))
    ∨ ∃ e s₁, tallyStep C (rdCut rd) s x = .inr (e, s₁)
      ∧ (e = .success ∨ e = .harvestStart ∨ ∃ e', e = .harvest e') ∧ tallyKey s₁ = tallyKey s := by
  obtain ⟨htree, hvle, hpot₁, hfr₁⟩ := tallyPre_spec C.Lmax (rdCut rd) C s x
  have hI₁ := tallyPre_inv G C rd hI x
  set K' := K + if x ∈ spurSet G C rd s then 1 else 0
  set s₁ := tallyPre C (rdCut rd) s x
  have hkey₁ : s₁.version = s.version → tallyKey s₁ = tallyKey s := fun hv => by
    obtain ⟨ht, he, -⟩ := tallyPre_same C (rdCut rd) s x hv
    simp only [tallyKey, ht, he, hv, s₁]
  rcases hl : tallyLook C s₁ with _ | e
  · left
    have hst : tallyStep C (rdCut rd) s x = settleEdges (rdCut rd) C.m C.Lmax C.fuel s₁ := by
      unfold tallyStep; simp only [s₁] at hl ⊢; rw [hl]
    obtain ⟨s', hs', hI', hle, hpot', heq, hfr⟩ := settle_spec C.m C.Lmax (rdCut rd)
      (TallyInv G C K') (fun s p c t hIs hv => fix_inv G C (rdCut rd) hK hLmax hIs hv)
      C.fuel s₁ hI₁ (hI₁.2.2.2.2.trans hfuel)
    refine ⟨s', hst.trans hs', hI', by omega, by omega, fun hv => ?_, fun hv => ?_⟩
    · have hv₁ : s₁.version = s.version := by omega
      rw [heq (by omega)]; exact hkey₁ hv₁
    · by_cases h' : s'.version = s₁.version
      · rw [heq h']; exact hfr₁ (by omega)
      · exact hfr h'
  · right
    have hst : tallyStep C (rdCut rd) s x = .inr (e, s₁) := by
      unfold tallyStep; simp only [s₁] at hl ⊢; rw [hl]
    refine ⟨e, s₁, hst, tallyLook_cases C hl, hkey₁ ?_⟩
    by_contra hv
    rw [tallyLook_fresh C hn₀ (hfr₁ hv)] at hl
    cases hl

end OrthoDFA
