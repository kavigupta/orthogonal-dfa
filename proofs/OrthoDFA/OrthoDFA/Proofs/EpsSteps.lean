import OrthoDFA.Level4Claim
import OrthoDFA.Proofs.RandomCorrect
import OrthoDFA.Proofs.EpsTrue

/-!
# Level 4's steps

A step reads its probe's prefixes, each followed by a midfix of the tree (`pot 0`), and, where it
splits, re-sifts every learned edge's witness's successor through the new tree (`splitReads`).
Where none of those reads is wrong (`WrongStep`), every edge it learns has a witness at its leaf
and a successor at its target by their states (`WitInv`), so its split is genuine. `WitInv` is
about states, not reads, so it holds whatever is read later.

Reads that agree on what a step reads give the same step (`applyProbe_congr`).
-/

namespace OrthoDFA

namespace Random

namespace Eps

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
  (side : σ → Bool) (read : FreeMonoid α → ARU) (C : Level3Settings)

/-- The tree has a true leaf for each leaf past the first, and every learned edge's witness is at
its leaf, and its successor at its target, by their states. -/
structure WitInv (s : Level3State α) : Prop where
  leaves : s.tree.leaves.length ≤ (img M side s.tree).card + 1
  wit : ∀ p c t w, s.edges (p, c) = some (t, w) →
    leafOf M side s.tree (M.eval w.toList) = p
      ∧ leafOf M side s.tree (M.eval (w * FreeMonoid.of c).toList) = t

open scoped Classical in
/-- Each learned edge's witness's successor followed by a midfix of `T'`. -/
noncomputable def resifts (s : Level3State α) (T' : DTree α) : Finset (FreeMonoid α) :=
  (s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α)).biUnion fun e =>
    match s.edges e with
    | some (_, w) => (mids T').image ((w * FreeMonoid.of e.2) * ·)
    | none => ∅

/-- What a step's split re-sifts. -/
noncomputable def splitReads (s : Level3State α) (x : FreeMonoid α) : Finset (FreeMonoid α) :=
  match probeOrAgree read s.tree s.edges C.walkStart x with
  | .wrongEdge p c _ t =>
    match s.edges (p, c) with
    | some (t₀, _) =>
      resifts s (s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p)
    | none => ∅
  | _ => ∅

/-- The step reads a string, a prefix of the probe followed by a midfix or one its split
re-sifts, on its state's wrong side. -/
def WrongStep (s : Level3State α) (x : FreeMonoid α) : Prop :=
  ∃ z ∈ pot 0 x s.tree ∪ splitReads read C s x,
    read z = if side (M.eval z.toList) then .reject else .accept

/-! ## The invariant -/

omit [DecidableEq α] in
theorem witInv_start : WitInv M side (initialState : Level3State α) where
  leaves := by
    have : 0 < (img M side (initialState : Level3State α).tree).card :=
      Finset.card_pos.2 ⟨_, Finset.mem_image_of_mem _ (Finset.mem_univ M.start)⟩
    simp only [initialState, newHypothesis, DTree.leaves] at this ⊢
    simp only [List.map_cons, List.map_nil, List.cons_append, List.nil_append, List.length_cons,
      List.length_nil]
    omega
  wit := fun _ _ _ _ h => by simp [initialState, newHypothesis] at h

omit [DecidableEq α] in
theorem witInv_congr {s s' : Level3State α} (hg : WitInv M side s) (ht : s'.tree = s.tree)
    (he : s'.edges = s.edges) : WitInv M side s' :=
  ⟨ht ▸ hg.leaves, by rw [ht, he]; exact hg.wit⟩

omit [DecidableEq α] in
/-- Learned edges leave leaves. -/
theorem edge_leaf {s : Level3State α} (hg : WitInv M side s) {e : List Bool × α}
    {t : List Bool} {w : FreeMonoid α} (h : s.edges e = some (t, w)) : e.1 ∈ s.tree.leaves := by
  rw [← (hg.wit e.1 e.2 t w h).1]
  exact leafOf_mem M side _ _

theorem cls_mem' {s : Level3State α} (hs : Inv read s) (hg : WitInv M side s) :
    s.tree ∈ classSet (Fintype.card σ) :=
  classSet_mono' (by have := hg.leaves; have := img_card_le M side s.tree; omega) hs.cls

omit [DecidableEq α] [Fintype σ] in
/-- A sift reading nothing wrong ends at the true leaf. -/
theorem sift_true' {T : DTree α} {z : FreeMonoid α} {p : List Bool} (h : T.sift read z = .inl p)
    (hw : ∀ m ∈ mids T,
      read (z * m) ≠ if side (M.eval (z * m).toList) then .reject else .accept) :
    leafOf M side T (M.eval z.toList) = p :=
  (sift_true M side read T z p h hw).symm

omit [DecidableEq α] [Fintype σ] in
/-- The prefix and successor of an edge a probe learns or disagrees at sift, where the step reads
nothing wrong, to their true leaves. -/
theorem true_of_probe {s : Level3State α} {x : FreeMonoid α} (hx : ¬ WrongStep M side read C s x)
    {p : List Bool} {c : α} {u : FreeMonoid α} {t : List Bool}
    (h : probeOrAgree read s.tree s.edges C.walkStart x = .wrongEdge p c u t
      ∨ probeOrAgree read s.tree s.edges C.walkStart x = .unlearnedEdge p c u t)
    (hu : s.tree.sift read u = .inl p) (huc : s.tree.sift read (u * FreeMonoid.of c) = .inl t) :
    leafOf M side s.tree (M.eval u.toList) = p
      ∧ leafOf M side s.tree (M.eval (u * FreeMonoid.of c).toList) = t := by
  obtain ⟨i, -, h2, rfl, hc⟩ := probe_prefix read s.tree s.edges C.walkStart x h
  have hno : ∀ j ≤ x.toList.length, ∀ m ∈ mids s.tree, read (prefixOf x j * m)
      ≠ if side (M.eval (prefixOf x j * m).toList) then .reject else .accept :=
    fun j hj m hm hw => hx ⟨_, Finset.mem_union_left _ (mem_pot (Nat.zero_le j) hj hm), hw⟩
  rw [← prefixOf_succ hc] at huc ⊢
  exact ⟨sift_true' M side read hu (hno i h2.le), sift_true' M side read huc (hno (i + 1) h2)⟩

omit [DecidableEq α] in
/-- A split's retargeted edges keep their witnesses at their leaves and successors at their
targets, where nothing it re-sifts reads wrong. -/
theorem wit_retarget {s : Level3State α} (hg : WitInv M side s) {p : List Bool}
    (hp : p ∈ s.tree.leaves) (d : FreeMonoid α)
    (hR : ∀ z ∈ resifts s (s.tree.splitAt d p),
      read z ≠ if side (M.eval z.toList) then .reject else .accept)
    (p' : List Bool) (c' : α) (t' : List Bool) (w : FreeMonoid α)
    (h : retarget read (s.tree.splitAt d p) p s.edges (p', c') = some (t', w)) :
    leafOf M side (s.tree.splitAt d p) (M.eval w.toList) = p'
      ∧ leafOf M side (s.tree.splitAt d p) (M.eval (w * FreeMonoid.of c').toList) = t' := by
  classical
  unfold retarget at h
  split_ifs at h with hpre
  rcases hE : s.edges (p', c') with _ | ⟨t₁, w₁⟩ <;> rw [hE] at h
  · simp at h
  obtain ⟨h1, h2⟩ := hg.wit p' c' t₁ w₁ hE
  have hp' : p' ≠ p := by
    rintro rfl
    simp at hpre
  have hsrc : leafOf M side (s.tree.splitAt d p) (M.eval w₁.toList) = p' := by
    rw [leafOf_splitAt M side d s.tree p hp, h1, if_neg hp']
  simp only at h
  split_ifs at h with ht
  · rcases hs : (s.tree.splitAt d p).sift read (w₁ * FreeMonoid.of c') with a | z <;>
      rw [hs] at h <;> simp only [Sum.elim_inl, Sum.elim_inr, Option.some.injEq, Prod.mk.injEq,
        reduceCtorEq] at h
    obtain ⟨rfl, rfl⟩ := h
    refine ⟨hsrc, sift_true' M side read hs fun m hm => hR _ ?_⟩
    have hl : p' ∈ s.tree.leaves := h1 ▸ leafOf_mem M side _ _
    refine Finset.mem_biUnion.2 ⟨(p', c'), by simp [hl], ?_⟩
    simp only [hE]
    exact Finset.mem_image.2 ⟨m, hm, rfl⟩
  · simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    refine ⟨hsrc, ?_⟩
    rw [leafOf_splitAt M side d s.tree p hp, h2, if_neg ht]

/-- A step that changes the hypothesis, its edge's two strings at their true leaves and its
split re-sifting nothing wrong, starts a fresh stretch, keeps the invariants and lowers `Ψr`. -/
theorem eps_step_change_of {s s' : Level3State α} (hs : Inv read s) (hg : WitInv M side s)
    {x : FreeMonoid α}
    (htrue : ∀ {p c u t}, (probeOrAgree read s.tree s.edges C.walkStart x = .wrongEdge p c u t
        ∨ probeOrAgree read s.tree s.edges C.walkStart x = .unlearnedEdge p c u t) →
      s.tree.sift read u = .inl p → s.tree.sift read (u * FreeMonoid.of c) = .inl t →
      leafOf M side s.tree (M.eval u.toList) = p
        ∧ leafOf M side s.tree (M.eval (u * FreeMonoid.of c).toList) = t)
    (hR : ∀ z ∈ splitReads read C s x, read z ≠ if side (M.eval z.toList) then .reject else .accept)
    (h : applyProbe read C s x = .inl s') (hne : ¬ (s'.tree = s.tree ∧ s'.edges = s.edges)) :
    Inv read s' ∧ WitInv M side s' ∧ s' = newHypothesis s'.tree s'.edges s'.hypotheses
      ∧ s'.hypotheses = s.hypotheses + 1 ∧ Ψr (Fintype.card σ) s' < Ψr (Fintype.card σ) s := by
  rcases step_cases read C hs h with
    ⟨p, c, u, t, ho, hE, hu, huc, hp, rfl⟩ |
    ⟨p, c, u, t, t₀, w₀, ho, hE, htt, hp, hu, huc, rfl⟩ | ⟨rfl, -, -⟩
  · obtain ⟨hcu, hcuc⟩ := htrue (Or.inr ho) hu huc
    refine ⟨⟨edgesOK_upd read hs.edges hu huc, hs.cls, hs.two⟩,
      ⟨hg.leaves, fun p' c' t' w h => ?_⟩, rfl, rfl,
      Level2.Ψ_lt_learn _ (s := Level2.newHypothesis s.tree s.edges 0) hp hE _ 0⟩
    simp only [newHypothesis] at h ⊢
    by_cases hpc : (p', c') = (p, c)
    · simp only [Prod.mk.injEq] at hpc
      obtain ⟨rfl, rfl⟩ := hpc
      simp only [Function.update_self, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨hcu, hcuc⟩
    · rw [Function.update_of_ne hpc] at h
      exact hg.wit p' c' t' w h
  · obtain ⟨hcu, hcuc⟩ := htrue (Or.inl ho) hu huc
    have hw₀ := hg.wit p c t₀ w₀ hE
    have ht : RealT M side s.tree p c t := ⟨M.eval u.toList, hcu, by rw [← hcuc, eval_mul]; rfl⟩
    have ht₀ : RealT M side s.tree p c t₀ :=
      ⟨M.eval w₀.toList, hw₀.1, by rw [← hw₀.2, eval_mul]; rfl⟩
    have hgen := genuine_of_real M side ht ht₀ (Ne.symm htt)
    have himg := img_card_splitAt M side hp hgen
    have hlen := DTree.length_leaves_splitAt
      (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) s.tree p hp
    have ht' := DTree.sift_inl_mem read _ _ t huc
    have ht₀' := DTree.sift_inl_mem read _ _ t₀ (hs.edges p hp c t₀ w₀ hE).2
    have hleaves : (splitLeaf read s p c t t₀).tree.leaves.length
        ≤ (img M side (splitLeaf read s p c t t₀).tree).card + 1 := by
      simp only [splitLeaf, newHypothesis]
      have := hg.leaves
      omega
    have hR : ∀ z ∈ resifts s
        (s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p),
        read z ≠ if side (M.eval z.toList) then .reject else .accept := by
      intro z hz
      refine hR z ?_
      unfold splitReads
      rw [ho]
      simp only [hE]
      exact hz
    refine ⟨⟨edgesOK_retarget read hs.edges hp _, ?_,
      by simp only [splitLeaf, newHypothesis]; rw [hlen]; have := hs.two; omega⟩,
      ⟨hleaves, fun p' c' t' w h => ?_⟩, rfl, rfl, ?_⟩
    · simp only [splitLeaf, newHypothesis]
      rw [hlen, show s.tree.leaves.length + 1 - 2 = s.tree.leaves.length - 2 + 1 by
        have := hs.two; omega]
      exact classSet_split hs.cls hp ht' ht₀'
    · simp only [splitLeaf, newHypothesis] at h ⊢
      exact wit_retarget M side read hg hp _ hR p' c' t' w h
    · exact Level2.Ψ_lt_split _ (s := Level2.newHypothesis s.tree s.edges 0) hp
        (by
          have := img_card_le M side (splitLeaf read s p c t t₀).tree
          simp only [splitLeaf, newHypothesis] at this hleaves
          simp only [Level2.newHypothesis]
          omega) _ 0
  · exact absurd ⟨rfl, rfl⟩ hne

/-- A step that changes the hypothesis and reads nothing wrong starts a fresh stretch, keeps the
invariants and lowers `Ψr`. -/
theorem eps_step_change {s s' : Level3State α} (hs : Inv read s) (hg : WitInv M side s)
    {x : FreeMonoid α} (hx : ¬ WrongStep M side read C s x)
    (h : applyProbe read C s x = .inl s') (hne : ¬ (s'.tree = s.tree ∧ s'.edges = s.edges)) :
    Inv read s' ∧ WitInv M side s' ∧ s' = newHypothesis s'.tree s'.edges s'.hypotheses
      ∧ s'.hypotheses = s.hypotheses + 1 ∧ Ψr (Fintype.card σ) s' < Ψr (Fintype.card σ) s :=
  eps_step_change_of M side read C hs hg (fun ho hu huc => true_of_probe M side read C hx ho hu huc)
    (fun z hz hw => hx ⟨z, Finset.mem_union_right _ hz, hw⟩) h hne

/-! ## Reads that agree on what a step reads -/

omit [DecidableEq α] in
theorem sift_congr {r r' : FreeMonoid α → ARU} :
    ∀ (T : DTree α) (z : FreeMonoid α), (∀ m ∈ mids T, r (z * m) = r' (z * m)) →
      T.sift r z = T.sift r' z
  | .leaf, _, _ => rfl
  | .node m a b, z, h => by
    classical
    have hm := h m (by simp [mids])
    have ha := sift_congr a z fun m' hm' => h m' (by simp [mids, hm'])
    have hb := sift_congr b z fun m' hm' => h m' (by simp [mids, hm'])
    simp only [DTree.sift, hm, ha, hb]

omit [DecidableEq α] in
/-- Reads agreeing on the probe's prefixes followed by midfixes sift every prefix alike. -/
theorem prefix_congr {r r' : FreeMonoid α → ARU} {T : DTree α} {x : FreeMonoid α}
    (h : ∀ z ∈ pot 0 x T, r z = r' z) (i : ℕ) :
    T.sift r (prefixOf x i) = T.sift r' (prefixOf x i) := by
  have hi : prefixOf x i = prefixOf x (min i x.toList.length) := by
    simp only [prefixOf]
    rcases le_total i x.toList.length with hl | hl
    · rw [min_eq_left hl]
    · rw [min_eq_right hl, List.take_of_length_le hl, List.take_length]
  rw [hi]
  exact sift_congr T _ fun m hm => h _ (mem_pot (Nat.zero_le _) (min_le_right _ _) hm)

omit [DecidableEq α] in
theorem probeOrAgree_congr {r r' : FreeMonoid α → ARU} {T : DTree α} {E : Edges α} {k : ℕ}
    {x : FreeMonoid α} (h : ∀ z ∈ pot 0 x T, r z = r' z) :
    probeOrAgree r T E k x = probeOrAgree r' T E k x := by
  have hs : ∀ i, T.sift r (prefixOf x i) = T.sift r' (prefixOf x i) := prefix_congr h
  have hA : ∀ st, siftAgrees r T x st = siftAgrees r' T x st := fun st => by
    funext p
    simp only [siftAgrees, hs]
  have hO : ∀ st un hi res, searchOutcome r T x st un hi res = searchOutcome r' T x st un hi res :=
    fun st un hi res => by cases res <;> simp only [searchOutcome, undecidedAt, hs]
  simp only [probeOrAgree, probe, hs, hA, hO]

omit [DecidableEq α] in
theorem countProbe_congr {r r' : FreeMonoid α → ARU} {s : Level3State α} {x : FreeMonoid α}
    (h : ∀ z ∈ pot 0 x s.tree, r z = r' z) : countProbe r C s x = countProbe r' C s x := by
  simp only [countProbe, probeOrAgree_congr h]

omit [DecidableEq α] in
theorem splitReads_congr {r r' : FreeMonoid α → ARU} {s : Level3State α} {x : FreeMonoid α}
    (h : ∀ z ∈ pot 0 x s.tree, r z = r' z) : splitReads r C s x = splitReads r' C s x := by
  simp only [splitReads, probeOrAgree_congr h]

/-- Reads giving the same probe and agreeing on what its split re-sifts give the same step, its
learned edges leaving leaves. -/
theorem applyProbe_congr_of {r r' : FreeMonoid α → ARU} {s : Level3State α} {x : FreeMonoid α}
    (hlv : ∀ e t w, s.edges e = some (t, w) → e.1 ∈ s.tree.leaves)
    (ho : probeOrAgree r s.tree s.edges C.walkStart x = probeOrAgree r' s.tree s.edges C.walkStart x)
    (h : ∀ z ∈ splitReads r C s x, r z = r' z) :
    applyProbe r C s x = applyProbe r' C s x := by
  have hc : countProbe r C s x = countProbe r' C s x := by simp only [countProbe, ho]
  unfold applyProbe
  rw [← ho, hc]
  rcases hpo : probeOrAgree r s.tree s.edges C.walkStart x with _ | ⟨p, c, u, t⟩ | _ | _ | _ <;>
    simp only
  rcases hE : s.edges (p, c) with _ | ⟨t₀, w₀⟩
  · rfl
  simp only [splitLeaf]
  congr 2
  funext e
  unfold retarget
  split_ifs with hpre
  · rfl
  rcases hEe : s.edges e with _ | ⟨t₁, w₁⟩
  · rfl
  simp only
  split_ifs with ht
  · have hmem : ∀ m ∈ mids (s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt
        (commonPrefix t t₀)) p),
        w₁ * FreeMonoid.of e.2 * m ∈ splitReads r C s x := by
      intro m hm
      unfold splitReads
      rw [hpo]
      simp only [hE]
      refine Finset.mem_biUnion.2 ⟨e, ?_, ?_⟩
      · simp [hlv e t₁ w₁ hEe]
      · simp only [hEe]
        exact Finset.mem_image.2 ⟨m, hm, rfl⟩
    rw [sift_congr _ _ fun m hm => h _ (hmem m hm)]
  · rfl

/-- Reads agreeing on what a step reads give the same step, its learned edges leaving leaves. -/
theorem applyProbe_congr {r r' : FreeMonoid α → ARU} {s : Level3State α} {x : FreeMonoid α}
    (hlv : ∀ e t w, s.edges e = some (t, w) → e.1 ∈ s.tree.leaves)
    (h : ∀ z ∈ pot 0 x s.tree ∪ splitReads r C s x, r z = r' z) :
    applyProbe r C s x = applyProbe r' C s x :=
  applyProbe_congr_of C hlv (probeOrAgree_congr fun z hz => h z (Finset.mem_union_left _ hz))
    fun z hz => h z (Finset.mem_union_right _ hz)

end Eps

end Random

end OrthoDFA
