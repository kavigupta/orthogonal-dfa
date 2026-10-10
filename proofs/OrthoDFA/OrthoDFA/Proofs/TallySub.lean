import OrthoDFA.Proofs.TallyTrue
import OrthoDFA.Proofs.SubCompose

/-!
# The round from its sub-rounds

A probe keeps the edges and records pointing at leaves; a fix redirects an edge at the same tree,
or splits a leaf on a letter followed by the midfix where two leaves part, dropping every record
and starting a fresh stretch. So a sub-round of the class ends at a tree of the class with one
more split, genuine or not, and the next sub-round starts there. Over the round, at most `|Q|`
sub-rounds end in a genuine split, since each raises the count of true leaves, and the round
fails once more than `S` end in one that is not: `round_of_sub` composes `SubRoundBound` with
`round_step`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace DTree

omit [Fintype α] [DecidableEq α] in
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

section Steps

variable (cut : FreeMonoid α → Option Bool)

/-- Every record's target is a leaf. -/
def RecsInto (s : TState α) : Prop := ∀ q c r, r ∈ s.recs q c → r.2 ∈ s.tree.paths

theorem setEdge_eq (s : TState α) (p : List Bool) (c : α) (e) (q : List Bool) (c' : α) :
    (s.setEdge p c e).edges q c' = if (q, c') = (p, c) then some e else s.edges q c' := by
  unfold TState.setEdge
  by_cases hq : q = p
  · subst hq
    by_cases hc : c' = c
    · subst hc; simp
    · simp [hc]
  · simp [hq]

theorem setEdge_into {s : TState α} (he : EdgesInto s.tree s.edges) {p : List Bool} {c : α}
    {t : List Bool} {w : FreeMonoid α} (ht : t ∈ s.tree.paths) :
    EdgesInto (s.setEdge p c (t, w)).tree (s.setEdge p c (t, w)).edges := by
  intro q e t' w' h
  rw [setEdge_eq] at h
  split_ifs at h
  · simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, -⟩ := h
    exact ht
  · exact he _ _ _ _ h

theorem addRec_into {s : TState α} (hr : RecsInto s) {p : List Bool} {c : α}
    {r : FreeMonoid α × List Bool} (ht : r.2 ∈ s.tree.paths) : RecsInto (s.addRec p c r) := by
  intro q e r' h
  simp only [TState.addRec] at h ⊢
  by_cases hq : q = p
  · subst hq
    by_cases hce : e = c
    · subst hce
      simp only [Function.update_self, List.mem_append, List.mem_singleton] at h
      rcases h with h | rfl
      · exact hr _ _ _ h
      · exact ht
    · simp only [Function.update_self, Function.update_of_ne hce] at h
      exact hr _ _ _ h
  · simp only [Function.update_of_ne hq] at h
    exact hr _ _ _ h

/-- A probe's outcome keeps the tree, and the edges and records pointing at leaves. -/
theorem tallyPre_into (C : TallyCfg) {s : TState α} (x : FreeMonoid α)
    (he : EdgesInto s.tree s.edges) (hr : RecsInto s) :
    (tallyPre C cut s x).tree = s.tree ∧ EdgesInto (tallyPre C cut s x).tree
      (tallyPre C cut s x).edges ∧ RecsInto (tallyPre C cut s x) := by
  unfold tallyPre
  split
  · split
    · split
      · rename_i u _ _ _ c p _ hp _ _ t ht he'
        exact ⟨rfl, setEdge_into (s := s.charge cut C.k x) he (DTree.sift_mem_paths _ _ _ ht),
          hr⟩
      · exact ⟨rfl, he, hr⟩
    · exact ⟨rfl, he, hr⟩
  · split
    · rename_i p c t sp hrec
      obtain ⟨-, -, -, -, -, ht, -⟩ := recordBy_spec cut hrec
      exact ⟨rfl, he, addRec_into (s := { s.charge cut C.k x with dis := _ }) hr
        (DTree.sift_mem_paths _ _ _ ht)⟩
    · exact ⟨rfl, he, hr⟩
  all_goals exact ⟨rfl, he, hr⟩

/-- A fix: a redirect keeps the tree, the edges and records pointing at leaves; a split is at a
leaf, on a letter followed by the midfix where two leaves part, with edges into the new tree's
leaves, no records and a fresh stretch. -/
theorem fixEdge_spec {m : ℕ} (hm : 0 < m) {s : TState α} (he : EdgesInto s.tree s.edges)
    (hr : RecsInto s)
    {p : List Bool} {c : α} {t : List Bool} (hv : Violates m s p c t) :
    let s' := fixEdge cut m s p c t
    (s'.tree = s.tree ∧ EdgesInto s'.tree s'.edges ∧ RecsInto s')
    ∨ (∃ t₀, t ∈ s.tree.paths ∧ t₀ ∈ s.tree.paths ∧ p ∈ s.tree.paths
      ∧ s'.tree = s.tree.splitAt (FreeMonoid.of c * s.tree.midAt (lcp t t₀)) p
      ∧ EdgesInto s'.tree s'.edges ∧ (∀ q e, s'.recs q e = []) ∧ s'.n = 0 ∧ s'.dis = 0) := by
  obtain ⟨hp, -, ht⟩ := hv
  have htp : t ∈ s.tree.paths := by
    have : 0 < ((s.recs p c).filter (·.2 = t)).length := by
      unfold TState.tally at ht; omega
    obtain ⟨r, hr'⟩ := List.exists_mem_of_length_pos this
    obtain ⟨h1, h2⟩ := List.mem_filter.1 hr'
    simp only [decide_eq_true_eq] at h2
    exact h2 ▸ hr _ _ _ h1
  simp only []
  unfold fixEdge
  simp only []
  rcases he0 : s.edges p c with _ | ⟨t₀, w₀⟩
  · exact .inl ⟨rfl, setEdge_into he htp, hr⟩
  · simp only []
    split_ifs with hm
    · right
      set T' := s.tree.splitAt (FreeMonoid.of c * s.tree.midAt (lcp t t₀)) p
      refine ⟨t₀, htp, he _ _ _ _ he0, hp, rfl, ?_, fun _ _ => rfl, rfl, rfl⟩
      intro q e t' w h
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
    · exact .inl ⟨rfl, setEdge_into he htp, hr⟩

variable (C : TallyCfg)

/-- A step that continues keeps the tree, the edges and records pointing at leaves, or splits as
`fixEdge_spec` says; one that ends past `Lmax` leaves splits, and any other keeps the tree. -/
theorem tallyStep_spec (hm : 0 < C.m) {s : TState α} {x : FreeMonoid α}
    (he : EdgesInto s.tree s.edges)
    (hr : RecsInto s) :
    (∀ s', tallyStep C cut s x = .inl s' →
      (s'.tree = s.tree ∧ EdgesInto s'.tree s'.edges ∧ RecsInto s')
      ∨ (∃ p c t t₀, t ∈ s.tree.paths ∧ t₀ ∈ s.tree.paths ∧ p ∈ s.tree.paths
        ∧ s'.tree = s.tree.splitAt (FreeMonoid.of c * s.tree.midAt (lcp t t₀)) p
        ∧ EdgesInto s'.tree s'.edges ∧ (∀ q e, s'.recs q e = []) ∧ s'.n = 0 ∧ s'.dis = 0))
    ∧ (∀ e s', tallyStep C cut s x = .inr (e, s') → e = .tooBig ∨ s'.tree = s.tree) := by
  obtain ⟨htr, he₁, hr₁⟩ := tallyPre_into cut C x he hr
  set s₁ := tallyPre C cut s x
  constructor
  · intro s' h
    unfold tallyStep at h
    simp only [] at h
    split at h
    · cases h
    · unfold settleOne at h
      split_ifs at h with hv hL
      · simp only [Sum.inl.injEq] at h
        subst h
        rcases fixEdge_spec cut hm he₁ hr₁ hv.choose_spec.choose_spec.choose_spec with
          ⟨ht, he', hr'⟩ | ⟨t₀, ht, ht₀, hp, hT, he', hr', hn, hd⟩
        · exact .inl ⟨ht.trans htr, he', hr'⟩
        · rw [htr] at ht ht₀ hp hT
          exact .inr ⟨_, _, _, t₀, ht, ht₀, hp, hT, he', hr', hn, hd⟩
      · simp only [Sum.inl.injEq] at h
        subst h
        exact .inl ⟨htr, he₁, hr₁⟩
  · intro e s' h
    unfold tallyStep at h
    simp only [] at h
    split at h
    · simp only [Sum.inr.injEq, Prod.mk.injEq] at h
      obtain ⟨-, rfl⟩ := h
      exact .inr htr
    · unfold settleOne at h
      split_ifs at h
      simp only [Sum.inr.injEq, Prod.mk.injEq] at h
      exact .inl h.1.symm

end Steps

section Compose

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (D : Measure (FreeMonoid α))
  [IsProbabilityMeasure D] (C : TallyCfg) (cut : FreeMonoid α → Option Bool) (S : ℕ)

/-- An ending that is not past `Lmax` leaves, at a tree of the class. -/
def OkEnd (e : TEnd α) (s : TState α) : Prop := e ≠ .tooBig ∧ G.InClass S s.tree

/-- A sub-round may start at `s` with `k` more splits that are not genuine allowed, at most `g`
genuine ones to come, and `T` draws enough for `k + g + 1` sub-rounds. -/
def Adm (Tsub : ℕ) (s : TState α) (k g T : ℕ) : Prop :=
  SubStart G S s ∧ (∃ f, G.Grown s.tree f ∧ f + k ≤ S)
    ∧ Fintype.card σ ≤ (G.trueLeaves s.tree).card + g ∧ (k + g + 1) * Tsub ≤ T

omit [Fintype σ] [IsProbabilityMeasure D] in
theorem lintegral_ite_one {β : Type*} [MeasurableSpace β] [Countable β]
    [MeasurableSingletonClass β] (μ : Measure β) (P : β → Prop) [DecidablePred P] :
    ∫⁻ x, (if P x then (1 : ENNReal) else 0) ∂μ = μ {x | P x} := by
  rw [← lintegral_indicator_one (Set.to_countable _).measurableSet]
  congr 1
  ext x
  simp [Set.indicator_apply]

theorem segVal_succ (w : (TEnd α ⊕ Bool) → TState α → ℕ → ENNReal) (s : TState α) (j T : ℕ)
    (xs : Fin (T + 1) → FreeMonoid α) :
    segVal (subSeg G C cut) w s (j + 1) (T + 1) xs = (subSeg G C cut s (xs 0)).elim
      (fun s' => segVal (subSeg G C cut) w s' j T (Fin.tail xs)) fun p => w p.1 p.2 T := by
  simp only [segVal]
  rcases subSeg G C cut s (xs 0) with _ | ⟨_, _⟩ <;> rfl

theorem subEnd_succ (s : TState α) (j T : ℕ) (xs : Fin (T + 1) → FreeMonoid α) :
    subEnd G C cut s (j + 1) (T + 1) xs = (subSeg G C cut s (xs 0)).elim
      (fun s' => subEnd G C cut s' j T (Fin.tail xs)) fun p => subKind p.1 := by
  simp only [subEnd]
  rcases subSeg G C cut s (xs 0) with _ | ⟨_, _⟩ <;> rfl

/-- A split from a sub-round's state lands at a start of the class, genuine or not. -/
theorem split_lands {s s' : TState α} {f : ℕ} (hf : G.Grown s.tree f) {p : List Bool} {c : α}
    {t t₀ : List Bool} (ht : t ∈ s.tree.paths) (ht₀ : t₀ ∈ s.tree.paths)
    (hp : p ∈ s.tree.paths)
    (hT : s'.tree = s.tree.splitAt (FreeMonoid.of c * s.tree.midAt (lcp t t₀)) p) :
    (∃ p d, s'.tree = s.tree.splitAt d p ∧ G.GenuineSplit s.tree p d) →
      G.Grown s'.tree f ∧ (G.trueLeaves s.tree).card + 1 ≤ (G.trueLeaves s'.tree).card := by
  rintro ⟨p', d', hT', hg⟩
  have hp' : p' ∈ s.tree.paths := by
    obtain ⟨q₁, -, hl, -⟩ := hg
    exact hl ▸ G.leafOf_mem_paths _ _
  obtain ⟨rfl, rfl⟩ := DTree.splitAt_inj hp' hp (hT'.symm.trans hT)
  refine ⟨hT ▸ ReadModel.Grown.real hf ht ht₀ hg, ?_⟩
  rw [hT]
  exact (G.trueLeaves_splitAt hp _).2 hg

open scoped Classical in
/-- The round from a sub-round's start fails to end well with chance at most `roundW`. -/
theorem round_of_sub (hm : 0 < C.m) {c δ δ' : ℝ} {Tsub : ℕ} (hc : 0 ≤ c) (hδ : 0 ≤ δ)
    (hδ' : 0 ≤ δ') (hsub : SubRoundBound G D C cut S c δ δ' Tsub) :
    ∀ N k g (s : TState α) (T : ℕ), k + g = N → Adm G S Tsub s k g T →
      (Measure.pi fun _ : Fin T => D)
          {xs | ¬ RunEnds (tallyStep C cut) (OkEnd G S) s (List.ofFn xs)}
        ≤ ENNReal.ofReal (roundW c δ δ' k g) := by
  intro N
  induction N using Nat.strong_induction_on with
  | _ N ih =>
  intro k g s T hN hadm
  obtain ⟨hs, ⟨f, hf, hfk⟩, hg, hT⟩ := hadm
  set w : (TEnd α ⊕ Bool) → TState α → ℕ → ENNReal := fun kd s' T' => match kd with
    | .inl e => if OkEnd G S e s' then 0 else 1
    | .inr _ => min 1 (⨅ (k' : ℕ) (g' : ℕ) (_ : k' + g' < N ∧ Adm G S Tsub s' k' g' T'),
        ENNReal.ofReal (roundW c δ δ' k' g'))
  set F : TState α → (T : ℕ) → (Fin T → FreeMonoid α) → Prop := fun s' T xs =>
    ¬ RunEnds (tallyStep C cut) (OkEnd G S) s' (List.ofFn xs)
  set V : (TEnd α ⊕ Bool) → TState α → (T : ℕ) → (Fin T → FreeMonoid α) → Prop :=
    fun kd s' T xs => match kd with
    | .inl e => ¬ OkEnd G S e s'
    | .inr _ => F s' T xs
  have hF : ∀ s' T x xs, F s' (T + 1) (Fin.cons x xs)
      ↔ (subSeg G C cut s' x).elim (fun s'' => F s'' T xs) fun p => V p.1 p.2 T xs := by
    intro s' T x xs
    simp only [F, List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, RunEnds, subSeg]
    rcases h : tallyStep C cut s' x with s'' | ⟨e, s''⟩
    · simp only []
      split_ifs <;> rfl
    · rfl
  have hV : ∀ kd s' T, (Measure.pi fun _ : Fin T => D) {xs | V kd s' T xs} ≤ w kd s' T := by
    intro kd s' T
    rcases kd with e | b
    · simp only [V, w]
      by_cases hok : OkEnd G S e s'
      · simp [hok]
      · simp only [hok, not_false_eq_true, Set.setOf_true, if_false]
        exact prob_le_one
    · simp only [V, w]
      refine le_min prob_le_one (le_iInf fun k' => le_iInf fun g' => le_iInf fun h => ?_)
      exact ih (k' + g') (hN ▸ h.1) k' g' s' T rfl h.2
  have hseg := seg_le D (subSeg G C cut) F V w hF hV T s Tsub
  set A : ℝ := if k = 0 then 1 else roundW c δ δ' (k - 1) g
  set B : ℝ := if g = 0 then 0 else roundW c δ δ' k (g - 1)
  have hnn := roundW_nonneg hc hδ hδ'
  have hA0 : 0 ≤ A := by simp only [A]; split_ifs <;> simp [hnn]
  have hB0 : 0 ≤ B := by simp only [B]; split_ifs <;> simp [hnn]
  have hw1 : ∀ kd s' T', w kd s' T' ≤ 1 := by
    intro kd s' T'
    rcases kd with e | b
    · simp only [w]; split_ifs <;> simp
    · exact min_le_left _ _
  have hpt : ∀ j T' (s' : TState α) (xs : Fin T' → FreeMonoid α), s'.tree = s.tree →
      EdgesInto s'.tree s'.edges → RecsInto s' → (k + g) * Tsub + j ≤ T' →
      segVal (subSeg G C cut) w s' j T' xs
        ≤ ENNReal.ofReal A * (if subEnd G C cut s' j T' xs = .fake then 1 else 0)
          + ENNReal.ofReal B * (if subEnd G C cut s' j T' xs = .real then 1 else 0)
          + (if subEnd G C cut s' j T' xs = .bad ∨ subEnd G C cut s' j T' xs = .unfinished
            then 1 else 0) := by
    intro j
    induction j with
    | zero => intro T' s' xs _ _ _ _; simp [segVal, subEnd]
    | succ j ihj =>
      intro T' s' xs htr he hr hTj
      rcases T' with _ | T'
      · simp [segVal, subEnd]
      have hspec := tallyStep_spec cut C hm (x := xs 0) he hr
      rw [segVal_succ, subEnd_succ]
      have hf' : G.Grown s'.tree f := htr ▸ hf
      have hg' : Fintype.card σ ≤ (G.trueLeaves s'.tree).card + g := htr ▸ hg
      have hT0 : (k + g) * Tsub ≤ T' := by
        have := hTj; generalize (k + g) * Tsub = X at this ⊢; omega
      rcases hst : tallyStep C cut s' (xs 0) with s₂ | ⟨e, s₂⟩
      · by_cases htr' : s₂.tree = s'.tree
        · have hseg : subSeg G C cut s' (xs 0) = .inl s₂ := by simp [subSeg, hst, htr']
          simp only [hseg, Sum.elim_inl]
          rcases hspec.1 _ hst with ⟨-, he', hr'⟩ | ⟨p, c, t, t₀, ht, ht₀, hp, hT', -⟩
          · exact ihj T' s₂ (Fin.tail xs) (htr'.trans htr) he' hr' (by omega)
          · exact absurd (htr'.symm.trans hT') (DTree.splitAt_ne hp).symm
        rcases hspec.1 _ hst with ⟨htr'', -, -⟩ |
          ⟨p, c, t, t₀, ht, ht₀, hp, hT', he', hr', hn, hd⟩
        · exact absurd htr'' htr'
        have hstart : ∀ f', G.Grown s₂.tree f' → f' ≤ S → SubStart G S s₂ :=
          fun f' h hle => ⟨⟨f', hle, h⟩, he', hr', hn, hd⟩
        by_cases hgen : ∃ p d, s₂.tree = s'.tree.splitAt d p ∧ G.GenuineSplit s'.tree p d
        · have hseg : subSeg G C cut s' (xs 0) = .inr (.inr true, s₂) := by
            simp [subSeg, hst, htr', hgen]
          simp only [hseg, Sum.elim_inr]
          obtain ⟨hg₂, hc₂⟩ := split_lands G hf' ht ht₀ hp hT' hgen
          have hle := G.trueLeaves_card_le s₂.tree
          have hg1 : 1 ≤ g := by omega
          have hadm : k + (g - 1) < N ∧ Adm G S Tsub s₂ k (g - 1) T' := by
            refine ⟨by omega, hstart f hg₂ (by omega), ⟨f, hg₂, hfk⟩, by omega, ?_⟩
            rw [show k + (g - 1) + 1 = k + g by omega]
            exact hT0
          have hw : w (.inr true) s₂ T' ≤ ENNReal.ofReal B := by
            refine (min_le_right _ _).trans ((iInf_le _ k).trans ((iInf_le _ (g - 1)).trans
              ((iInf_le _ hadm).trans (le_of_eq ?_))))
            simp only [B, if_neg (by omega : g ≠ 0)]
          simpa [subKind] using hw
        · have hseg : subSeg G C cut s' (xs 0) = .inr (.inr false, s₂) := by
            simp [subSeg, hst, htr', hgen]
          simp only [hseg, Sum.elim_inr]
          have hng : ¬ G.GenuineSplit s'.tree p (FreeMonoid.of c * s'.tree.midAt (lcp t t₀)) :=
            fun h => hgen ⟨_, _, hT', h⟩
          have hw : w (.inr false) s₂ T' ≤ ENNReal.ofReal A := by
            rcases Nat.eq_zero_or_pos k with rfl | hk
            · simpa [A] using hw1 (.inr false) s₂ T'
            have hg₂ : G.Grown s₂.tree (f + 1) := hT' ▸ ReadModel.Grown.fake hf' hp ht ht₀ hng
            have hc₂ := (G.trueLeaves_splitAt hp (FreeMonoid.of c * s'.tree.midAt (lcp t t₀))).1
            rw [← hT'] at hc₂
            have hadm : (k - 1) + g < N ∧ Adm G S Tsub s₂ (k - 1) g T' := by
              refine ⟨by omega, hstart (f + 1) hg₂ (by omega), ⟨f + 1, hg₂, by omega⟩, by omega,
                ?_⟩
              rw [show k - 1 + g + 1 = k + g by omega]
              exact hT0
            refine (min_le_right _ _).trans ((iInf_le _ (k - 1)).trans ((iInf_le _ g).trans
              ((iInf_le _ hadm).trans (le_of_eq ?_))))
            simp only [A, if_neg hk.ne']
          simpa [subKind] using hw
      · have hseg : subSeg G C cut s' (xs 0) = .inr (.inl e, s₂) := by simp [subSeg, hst]
        simp only [hseg, Sum.elim_inr]
        have hok : e ≠ .tooBig → OkEnd G S e s₂ := fun he => by
          refine ⟨he, ?_⟩
          rcases hspec.2 _ _ hst with h | h
          · exact absurd h he
          · rw [h, htr]; exact hs.1
        rcases e with _ | _ | e | _ | _
        · simp [w, subKind, hok (by simp)]
        · simp [w, subKind, hok (by simp)]
        · simp [w, subKind, hok (by simp)]
        · simp [w, subKind, hok (by simp)]
        · simpa [subKind] using hw1 (.inl .tooBig) s₂ T'
  have hT1 : (k + g) * Tsub + Tsub ≤ T := by
    have : (k + g + 1) * Tsub = (k + g) * Tsub + Tsub := by ring
    omega
  set P := Measure.pi fun _ : Fin T => D
  set E := fun xs => subEnd G C cut s Tsub T xs
  have hbound := lintegral_mono (μ := P) fun xs =>
    hpt Tsub T s xs rfl hs.2.1 (fun q c r h => by rw [hs.2.2.1 q c] at h; cases h) hT1
  rw [lintegral_add_left (measurable_of_countable _),
    lintegral_add_left (measurable_of_countable _),
    lintegral_const_mul _ (measurable_of_countable _),
    lintegral_const_mul _ (measurable_of_countable _),
    lintegral_ite_one, lintegral_ite_one, lintegral_ite_one] at hbound
  obtain ⟨hfake, hbo⟩ := hsub s hs T (by nlinarith)
  have hgr : P {xs | E xs = .good ∨ E xs = .real}
      = P {xs | E xs = .good} + P {xs | E xs = .real} := by
    rw [← lintegral_ite_one, ← lintegral_ite_one, ← lintegral_ite_one,
      ← lintegral_add_left (measurable_of_countable _)]
    congr 1; ext xs
    rcases E xs <;> simp
  have hsum : P {xs | E xs = .fake} + P {xs | E xs = .real} + P {xs | E xs = .good}
      + P {xs | E xs = .bad ∨ E xs = .unfinished} ≤ 1 := by
    rw [← lintegral_ite_one, ← lintegral_ite_one, ← lintegral_ite_one, ← lintegral_ite_one,
      ← lintegral_add_left (measurable_of_countable _),
      ← lintegral_add_left (measurable_of_countable _),
      ← lintegral_add_left (measurable_of_countable _)]
    calc _ ≤ ∫⁻ _, (1 : ENNReal) ∂P := lintegral_mono fun xs => by rcases E xs <;> simp
      _ = 1 := by rw [lintegral_const, measure_univ, mul_one]
  set pf := (P {xs | E xs = .fake}).toReal
  set pr := (P {xs | E xs = .real}).toReal
  set pg := (P {xs | E xs = .good}).toReal
  set pb := (P {xs | E xs = .bad ∨ E xs = .unfinished}).toReal
  have hPf : P {xs | E xs = .fake} = ENNReal.ofReal pf :=
    (ENNReal.ofReal_toReal (measure_ne_top _ _)).symm
  have hPr : P {xs | E xs = .real} = ENNReal.ofReal pr :=
    (ENNReal.ofReal_toReal (measure_ne_top _ _)).symm
  have hPg : P {xs | E xs = .good} = ENNReal.ofReal pg :=
    (ENNReal.ofReal_toReal (measure_ne_top _ _)).symm
  have hPb : P {xs | E xs = .bad ∨ E xs = .unfinished} = ENNReal.ofReal pb :=
    (ENNReal.ofReal_toReal (measure_ne_top _ _)).symm
  have h0 : 0 ≤ pf := ENNReal.toReal_nonneg
  have hr0 : 0 ≤ pr := ENNReal.toReal_nonneg
  have hg0 : 0 ≤ pg := ENNReal.toReal_nonneg
  have hb0 : 0 ≤ pb := ENNReal.toReal_nonneg
  rw [hgr, hPf, hPr, hPg, ← ENNReal.ofReal_add hg0 hr0, ← ENNReal.ofReal_mul hc,
    ← ENNReal.ofReal_add (by positivity) hδ, ENNReal.ofReal_le_ofReal_iff (by positivity)] at hfake
  rw [hPb, ENNReal.ofReal_le_ofReal_iff hδ'] at hbo
  rw [hPf, hPr, hPg, hPb, ← ENNReal.ofReal_add h0 hr0, ← ENNReal.ofReal_add (by positivity) hg0,
    ← ENNReal.ofReal_add (by positivity) hb0, ← ENNReal.ofReal_one,
    ENNReal.ofReal_le_ofReal_iff zero_le_one] at hsum
  rw [hPf, hPr, hPb, ← ENNReal.ofReal_mul hA0, ← ENNReal.ofReal_mul hB0,
    ← ENNReal.ofReal_add (by positivity) (by positivity),
    ← ENNReal.ofReal_add (by positivity) hb0] at hbound
  refine hseg.trans (hbound.trans (ENNReal.ofReal_le_ofReal ?_))
  have := round_step hc hδ hδ' k g hfake hbo h0 hr0 hg0 (by linarith) hb0
  simpa [A, B] using this

theorem runEnds_mono {S X E : Type*} {step : S → X → S ⊕ (E × S)} {P Q : E → S → Prop}
    (h : ∀ e s, P e s → Q e s) : ∀ (s : S) (l : List X), RunEnds step P s l → RunEnds step Q s l
  | _, [] => id
  | s, x :: xs => by
    simp only [RunEnds]
    rcases step s x with s' | ⟨e, s'⟩
    · exact runEnds_mono h s' xs
    · exact h e s'

theorem runEnds_or {S X E : Type*} {step : S → X → S ⊕ (E × S)} {P Q : E → S → Prop} :
    ∀ (s : S) (l : List X), RunEnds step P s l →
      RunEnds step Q s l ∨ RunEnds step (fun e s' => P e s' ∧ ¬ Q e s') s l
  | _, [] => False.elim
  | s, x :: xs => by
    simp only [RunEnds]
    rcases step s x with s' | ⟨e, s'⟩
    · exact runEnds_or s' xs
    · intro hp
      by_cases hq : Q e s'
      · exact .inl hq
      · exact .inr ⟨hp, hq⟩

theorem harvest_endsWell {σ : Type*} (G : ReadModel α σ) (D : Measure (FreeMonoid α)) (ε : ℝ)
    (dis : Set (FreeMonoid α)) (l : List (FreeMonoid α)) :
    EndsWell G.M (fun q => ¬ G.Good q) D ε dis (.harvest l) ↔ l.length < 2 * G.badCount l := by
  simp only [EndsWell, ReadModel.badCount]
  constructor <;> intro h <;> convert h using 5 <;> exact decide_eq_decide.2 Iff.rfl

universe u v w in
/-- `TallyRound` from `SubRound` and `HarvestGood`. -/
theorem tally_round_of (hsub : SubRound.{u, v}) (hharv : HarvestGood.{u, v})
    (hsucc : SuccessSound.{u}) : TallyRound.{u, v, w} := by
  intro α _ _ σ _ Ω _ μ _ G read D _ C S nEnd nRec hP hS L T ρ θg θgs θgpt θpt' θr εd' hlen hρ0
    hρ1 hm hLmax hn₀ hn₀' hθpt0 hθpt1 hεd0 hεd1 hθpt'0 hθpt'1 hθr0 hθr1 hεd'0 hεd'1 hcond hhP hhS
    hhS' hL1 ha ha1 hθg0 hθg hexc hθgs0 hθgs1 hstart hθgpt0 hθgpt1 hstartpt hT
  set Tsub := subT C (Fintype.card α) nEnd nRec
  set δ := subFake C (Fintype.card α) ρ Tsub
  set δ' := subOpen C (Fintype.card α) nRec θr (termLevel nEnd hP hS θpt' εd')
  set c := ENNReal.ofReal (roundW 0 δ δ' S (Fintype.card σ))
    + ENNReal.ofReal ((2 ^ (C.Lmax + 1) * Fintype.card α + T + 2 * (T * T)) * C.a)
  set M := toMeasurable μ {ω | ¬ TallyE G D C S ρ θg θgs θgpt (read · ω)}
  have hδ0 : 0 ≤ δ := by
    have := binomSfGe_nonneg (n := Tsub) hρ0 hρ1 C.m
    simp only [δ, subFake]; positivity
  have hδ'0 : 0 ≤ δ' := by
    have h1 := binomSfGe_le_one (n := hS + 1) hθpt'0 hθpt'1 hP
    have h2 := binomSfGe_le_one (n := nRec) hθr0 hθr1 (C.Lmax ^ 2 * Fintype.card α * C.m + 1)
    have h3 := binomSfGe_nonneg (n := nEnd) hεd'0 hεd'1 (hS + 1)
    simp only [δ', subOpen, termLevel]
    have : 0 ≤ 1 - binomSfGe (hS + 1) θpt' hP + binomSfGe nEnd εd' (hS + 1) := by linarith
    have : 0 ≤ 1 - binomSfGe nRec θr (C.Lmax ^ 2 * Fintype.card α * C.m + 1) := by linarith
    positivity
  have hpt : ∀ ω, (Measure.pi fun _ : Fin T => D)
      {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut) (TallyEndsWell G D C (read · ω))
        tallyStart (List.ofFn xs)} ≤ M.indicator 1 ω + c := by
    intro ω
    by_cases hω : TallyE G D C S ρ θg θgs θgpt (read · ω)
    · refine le_trans ?_ le_add_self
      set cut : FreeMonoid α → Option Bool := fun z => (read z ω).cut
      have hsb := hsub G (read · ω) D C S nEnd nRec hP hS ρ θg θgs θgpt θpt' θr εd' hρ0 hρ1 hm
        hLmax hn₀ hn₀' hθpt0 hθpt1 hεd0 hεd1 hθpt'0 hθpt'1 hθr0 hθr1 hεd'0 hεd'1 hcond hhP hhS hhS'
        hω
      have hstart₀ : Adm G S Tsub (tallyStart : TState α) S (Fintype.card σ) T :=
        ⟨⟨⟨0, Nat.zero_le _, .start⟩, fun _ _ _ _ h => by simp [tallyStart] at h,
          fun _ _ => rfl, rfl, rfl⟩, ⟨0, .start, by omega⟩, by omega, hT⟩
      have h1 := round_of_sub G D C cut S hm le_rfl hδ0 hδ'0 hsb _ S (Fintype.card σ)
        tallyStart T rfl hstart₀
      have h2 := hharv G (read · ω) D C S L T ρ θg θgs θgpt hlen hL1 hm ha ha1 hθg0 hθg
        (by omega) hexc hθgs0 hθgs1 hstart hθgpt0 hθgpt1 hstartpt hω
      have hsub' : {xs : Fin T → FreeMonoid α | ¬ RunEnds (tallyStep C cut) (GoodEnd G)
          tallyStart (List.ofFn xs)}
          ⊆ {xs | ¬ RunEnds (tallyStep C cut) (OkEnd G S) tallyStart (List.ofFn xs)}
            ∪ {xs | RunEnds (tallyStep C cut)
              (fun e s' => G.InClass S s'.tree ∧ e ≠ .tooBig ∧ ¬ GoodEnd G e s') tallyStart
              (List.ofFn xs)} := by
        intro xs hxs
        by_cases hok : RunEnds (tallyStep C cut) (OkEnd G S) tallyStart (List.ofFn xs)
        · rcases runEnds_or (Q := GoodEnd G) _ _ hok with h | h
          · exact absurd h hxs
          · exact .inr (runEnds_mono (fun e s' h => ⟨h.1.2, h.1.1, h.2⟩) _ _ h)
        · exact .inl hok
      have h3 := hsucc (read · ω) D C T hεd0 hεd1 ha1
      have hsnd : {xs : Fin T → FreeMonoid α | ¬ RunEnds (tallyStep C cut)
          (TallyEndsWell G D C (read · ω)) tallyStart (List.ofFn xs)}
          ⊆ {xs | ¬ RunEnds (tallyStep C cut) (GoodEnd G) tallyStart (List.ofFn xs)}
            ∪ {xs | RunEnds (tallyStep C cut) (fun e s' => e = .success
              ∧ C.εd ≤ D.real (ReadModel.searchAt (read · ω) C.k s'.tree s'.edges)) tallyStart
              (List.ofFn xs)} := by
        intro xs hxs
        by_cases hg : RunEnds (tallyStep C cut) (GoodEnd G) tallyStart (List.ofFn xs)
        · rcases runEnds_or (Q := TallyEndsWell G D C (read · ω)) _ _ hg with h | h
          · exact absurd h hxs
          · refine .inr (runEnds_mono (fun e s' h => ?_) _ _ h)
            obtain ⟨hge, hns⟩ := h
            rcases e with _ | _ | e | _ | _
            · simp only [TallyEndsWell, tallyEnd, EndsWell, not_le] at hns
              exact ⟨rfl, hns.le⟩
            · exact absurd ((harvest_endsWell G D _ _ _).2 hge) hns
            · exact absurd ((harvest_endsWell G D _ _ _).2 hge) hns
            · exact absurd ((harvest_endsWell G D _ _ _).2 hge) hns
            · exact hge.elim
        · exact .inl hg
      have hsplit : ENNReal.ofReal ((2 ^ (C.Lmax + 1) * Fintype.card α + T + T * T) * C.a)
          + ENNReal.ofReal (T * T * C.a)
          = ENNReal.ofReal ((2 ^ (C.Lmax + 1) * Fintype.card α + T + 2 * (T * T)) * C.a) := by
        rw [← ENNReal.ofReal_add (by positivity) (by positivity)]
        congr 1
        ring
      refine (measure_mono hsnd).trans ((measure_union_le _ _).trans ((add_le_add
        ((measure_mono hsub').trans ((measure_union_le _ _).trans (add_le_add h1 h2))) h3).trans
        (le_of_eq ?_)))
      simp only [c]
      rw [add_assoc, hsplit]
    · have hM : ω ∈ M := subset_toMeasurable _ _ hω
      rw [Set.indicator_of_mem hM, Pi.one_apply]
      exact prob_le_one.trans le_self_add
  calc ∫⁻ ω, (Measure.pi fun _ : Fin T => D)
        {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut) (TallyEndsWell G D C (read · ω))
          tallyStart (List.ofFn xs)} ∂μ
      ≤ ∫⁻ ω, (M.indicator 1 ω + c) ∂μ := lintegral_mono hpt
    _ = μ {ω | ¬ TallyE G D C S ρ θg θgs θgpt (read · ω)} + c := by
      rw [lintegral_add_right _ measurable_const, lintegral_indicator_one
        (measurableSet_toMeasurable _ _), measure_toMeasurable, lintegral_const, measure_univ,
        mul_one]
    _ = _ := by simp only [c]; ring

end Compose

end OrthoDFA
