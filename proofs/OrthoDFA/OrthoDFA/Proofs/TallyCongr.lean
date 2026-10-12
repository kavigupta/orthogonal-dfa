import OrthoDFA.Proofs.TallyNoise
import OrthoDFA.Proofs.HarvestClasses
import OrthoDFA.Proofs.TripleBound

/-!
# What a probe depends on

A probe of `x` reads the cut only through the sifts of `x`'s prefixes, and the edges only through
their targets at the tree's leaves. So two cuts placing every prefix of `x` alike, or two edge
maps with the same targets out of every leaf, give the same probe.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

section EdgeTable

/-- The two edge maps point every leaf's edges at the same targets. -/
def TgtAgree (T : DTree α) (e e' : EdgeTable α) : Prop :=
  ∀ p ∈ T.leaves, ∀ c, (e p c).map Prod.fst = (e' p c).map Prod.fst

theorem follow_congr {T : DTree α} {e e' : EdgeTable α} (he : EdgesInto T e) (h : TgtAgree T e e') :
    ∀ (cs : List α) (p : List Bool), p ∈ T.leaves → follow e p cs = follow e' p cs
  | [], _, _ => rfl
  | c :: cs, p, hp => by
    have hc := h p hp c
    simp only [follow]
    rcases h1 : e p c with _ | ⟨q, w⟩ <;> rcases h2 : e' p c with _ | ⟨q', w'⟩ <;>
      rw [h1, h2] at hc <;> simp only [Option.map_none, Option.map_some, reduceCtorEq,
        Option.some.injEq] at hc
    all_goals first
      | rfl
      | (subst hc; simp only []; rw [follow_congr he h cs q (he _ _ _ _ h1)])

variable {cut : FreeMonoid α → Option Bool} {T : DTree α} {e e' : EdgeTable α} {k : ℕ}
  {x : FreeMonoid α}

theorem kWalkBy_econgr (he : EdgesInto T e) (h : TgtAgree T e e') :
    kWalkBy cut T e k x = kWalkBy cut T e' k x := by
  simp only [kWalkBy]
  rcases hs : T.siftCut cut (prefixOf x k) with p | b
  · simp only []
    rw [follow_congr he h _ p (DTree.siftCut_mem_paths _ _ _ hs)]
  · rfl

theorem walkToBy_econgr (he : EdgesInto T e) (h : TgtAgree T e e') :
    walkToBy cut T e k x = walkToBy cut T e' k x := by
  funext j
  simp only [walkToBy]
  rcases hs : T.siftCut cut (prefixOf x k) with p | b
  · simp only []
    rw [follow_congr he h _ p (DTree.siftCut_mem_paths _ _ _ hs)]
  · rfl

theorem walkCheckBy_econgr (he : EdgesInto T e) (h : TgtAgree T e e') :
    walkCheckBy cut T e k x = walkCheckBy cut T e' k x := by
  simp only [walkCheckBy, kWalkBy_econgr (x := x) (k := k) (cut := cut) he h,
    walkToBy_econgr (x := x) (k := k) (cut := cut) he h]

theorem probeBy_econgr (he : EdgesInto T e) (h : TgtAgree T e e') :
    probeBy cut T e k x = probeBy cut T e' k x := by
  simp only [probeBy, walkCheckBy_econgr (x := x) (k := k) (cut := cut) he h]

theorem ptHarvBy_econgr (he : EdgesInto T e) (h : TgtAgree T e e') :
    ptHarvBy cut T e k x = ptHarvBy cut T e' k x := by
  simp only [ptHarvBy, probeBy_econgr (x := x) (k := k) (cut := cut) he h]

open scoped Classical in
/-- Edge maps out of the leaves into the leaves, each witness `1`. -/
noncomputable def edgeMaps (T : DTree α) : Finset (EdgeTable α) :=
  (Finset.univ : Finset (↥(T.leaves.toFinset ×ˢ (Finset.univ : Finset α))
    → Option ↥T.leaves.toFinset)).image fun f p c =>
      if h : (p, c) ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α) then
        (f ⟨(p, c), h⟩).map fun t => (t.1, 1)
      else none

theorem edgeMaps_card (T : DTree α) :
    (edgeMaps T).card ≤ (T.leaves.length + 1) ^ (T.leaves.length * Fintype.card α) := by
  classical
  refine Finset.card_image_le.trans ?_
  simp only [Finset.card_univ, Fintype.card_fun, Fintype.card_option, Fintype.card_coe,
    Finset.card_product, Finset.card_univ]
  have h := List.toFinset_card_le T.leaves
  calc _ ≤ (T.leaves.length + 1) ^ (T.leaves.toFinset.card * Fintype.card α) := by gcongr
    _ ≤ _ := by
      apply Nat.pow_le_pow_right (by omega)
      gcongr

theorem exists_edgeMaps {T : DTree α} {e : EdgeTable α} (he : EdgesInto T e) :
    ∃ e' ∈ edgeMaps T, TgtAgree T e e' ∧ EdgesInto T e' := by
  classical
  refine ⟨fun p c => if h : (p, c) ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α) then
      (e p c).map fun q => (q.1, 1) else none, ?_, ?_, ?_⟩
  · refine Finset.mem_image.2 ⟨fun pc => (e pc.1.1 pc.1.2).bind fun q =>
      if hq : q.1 ∈ T.leaves.toFinset then some ⟨q.1, hq⟩ else none, Finset.mem_univ _, ?_⟩
    funext p c
    split_ifs with h
    · rcases hpc : e p c with _ | ⟨q, w⟩
      · simp [hpc]
      · have hq : q ∈ T.leaves := he _ _ _ _ hpc
        simp [hpc, hq]
    · rfl
  · intro p hp c
    have h : (p, c) ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α) := by simp [hp]
    simp only [dif_pos h]
    rcases e p c with _ | ⟨q, w⟩ <;> simp
  · intro p c q w hpc
    simp only at hpc
    split_ifs at hpc with h
    · rcases he' : e p c with _ | ⟨q', w'⟩ <;> rw [he'] at hpc
      · simp at hpc
      · simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at hpc
        obtain ⟨rfl, -⟩ := hpc
        exact he _ _ _ _ he'

end EdgeTable

section CutK

/-- The sifts of `x`'s prefixes from position `k` on agree. -/
def PrefAgreeK (cut cut' : FreeMonoid α → Option Bool) (t : DTree α) (x : FreeMonoid α) (k : ℕ) :
    Prop :=
  ∀ j, k ≤ j → t.siftCut cut (prefixOf x j) = t.siftCut cut' (prefixOf x j)

variable {cut cut' : FreeMonoid α → Option Bool} {t : DTree α} {edges : EdgeTable α} {k : ℕ}
  {x : FreeMonoid α}

theorem PrefAgreeK.whole (h : PrefAgreeK cut cut' t x k) : t.siftCut cut x = t.siftCut cut' x := by
  by_cases hk : k ≤ x.toList.length
  · have := h x.toList.length hk
    rwa [show prefixOf x x.toList.length = x by simp [prefixOf]] at this
  · have := h k le_rfl
    rwa [show prefixOf x k = x by
      simp only [prefixOf]; rw [List.take_of_length_le (by omega)]; simp] at this

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_congr (ps : List (List Bool)) :
    ∀ (fuel lo hi : ℕ) (ag ag' : ℕ → Option Bool), (∀ i, lo ≤ i → ag i = ag' i) →
      bracketAt (α := α) ag ps fuel lo hi = bracketAt ag' ps fuel lo hi
  | 0, _, _, _, _, _ => rfl
  | fuel + 1, lo, hi, ag, ag', h => by
    simp only [bracketAt]
    by_cases hlh : lo + 1 < hi
    · rw [if_pos hlh, if_pos hlh, h ((lo + hi) / 2) (by omega), h ((lo + hi) / 2 - 1) (by omega),
        h ((lo + hi) / 2 + 1) (by omega),
        bracketAt_congr ps fuel ((lo + hi) / 2) hi ag ag' (fun i hi' => h i (by omega)),
        bracketAt_congr ps fuel lo ((lo + hi) / 2) ag ag' h,
        bracketAt_congr ps fuel ((lo + hi) / 2 + 1) hi ag ag' (fun i hi' => h i (by omega)),
        bracketAt_congr ps fuel lo ((lo + hi) / 2 - 1) ag ag' h]
    · rw [if_neg hlh, if_neg hlh]

theorem kWalkBy_congrK (h : PrefAgreeK cut cut' t x k) :
    kWalkBy cut t edges k x = kWalkBy cut' t edges k x := by
  simp only [kWalkBy, h k le_rfl]

theorem walkToBy_congrK (h : PrefAgreeK cut cut' t x k) :
    walkToBy cut t edges k x = walkToBy cut' t edges k x := by
  funext j
  simp only [walkToBy, h k le_rfl]

theorem agreesAtBy_congrK (h : PrefAgreeK cut cut' t x k) (w : ℕ → List Bool) :
    ∀ i, k ≤ i → agreesAtBy cut t x w i = agreesAtBy cut' t x w i := by
  intro i hi
  simp only [agreesAtBy, h i hi]

theorem kWalkBy_edge_ge {s : List Bool} {c : α} {j : ℕ} (hk : kWalkBy cut t edges k x = .edge s c j) :
    k ≤ j := by
  unfold kWalkBy at hk
  split at hk
  · cases hk
  · split at hk
    · cases hk
    · simp only [KWalk.edge.injEq] at hk; omega

theorem walkCheckBy_congrK (h : PrefAgreeK cut cut' t x k) :
    walkCheckBy cut t edges k x = walkCheckBy cut' t edges k x := by
  unfold walkCheckBy
  rw [← kWalkBy_congrK (edges := edges) h, walkToBy_congrK (edges := edges) h, h.whole]
  rcases hk : kWalkBy cut t edges k x with _ | ⟨s, c, j⟩ | ps
  · rfl
  · have hj := kWalkBy_edge_ge hk
    simp only [h (j + 1) (by omega), h j hj]
  · rfl

theorem probeBy_congrK (h : PrefAgreeK cut cut' t x k) :
    probeBy cut t edges k x = probeBy cut' t edges k x := by
  unfold probeBy
  split_ifs
  · rfl
  rw [walkCheckBy_congrK (edges := edges) h]
  rcases walkCheckBy cut' t edges k x with o | ⟨ps, hi⟩
  · rfl
  · simp only [Sum.elim_inr]
    exact bracketAt_congr ps _ _ _ _ _ (agreesAtBy_congrK h _)

theorem ptHarvBy_congrK (h : PrefAgreeK cut cut' t x k) :
    ptHarvBy cut t edges k x = ptHarvBy cut' t edges k x := by
  unfold ptHarvBy
  rw [probeBy_congrK (edges := edges) h]
  rcases ho : probeBy cut' t edges k x with _ | _ | _ | j | _ | j | _
  · rfl
  · rfl
  · rfl
  · obtain ⟨ps, hi, -, hb⟩ := probeBy_search cut' ho trivial
    have := bracketAt_pair_gt _ _ _ _ _ _ hb
    simp only [h j this.le]
  · rfl
  · obtain ⟨ps, hi, -, hb⟩ := probeBy_search cut' ho trivial
    have := bracketAt_triple_gt _ _ _ _ _ _ hb
    simp only [h j this.le]
  · rfl

end CutK

end OrthoDFA
