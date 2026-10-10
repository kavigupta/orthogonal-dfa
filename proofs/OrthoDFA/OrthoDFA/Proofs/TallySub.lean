import OrthoDFA.Proofs.TallyTrue
import OrthoDFA.Proofs.SubCompose

/-!
# Sub-rounds' steps

A probe keeps the edges and records pointing at leaves; a fix redirects an edge at the same tree,
or splits a leaf on a letter followed by the midfix where two leaves part, dropping every record
and starting a fresh stretch. So a sub-round of the class ends at a tree of the class with one
more split, genuine or not, and the next sub-round starts there; a genuine split raises the count
of true leaves (`split_lands`).
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

omit [Fintype σ] [IsProbabilityMeasure D] in
theorem lintegral_ite_one {β : Type*} [MeasurableSpace β] [Countable β]
    [MeasurableSingletonClass β] (μ : Measure β) (P : β → Prop) [DecidablePred P] :
    ∫⁻ x, (if P x then (1 : ENNReal) else 0) ∂μ = μ {x | P x} := by
  rw [← lintegral_indicator_one (Set.to_countable _).measurableSet]
  congr 1
  ext x
  simp [Set.indicator_apply]

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

end Compose

end OrthoDFA
