import OrthoDFA.Proofs.TallyNoise

/-!
# What a probe depends on

A probe of `x` reads the cut only through the sifts of `x`'s prefixes, and the edges only through
their targets at the tree's leaves. So two cuts placing every prefix of `x` alike, or two edge
maps with the same targets out of every leaf, give the same probe.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

section Cut

variable {cut cut' : FreeMonoid α → Option Bool} {t : DTree α} {edges : Edges α} {k : ℕ}
  {x : FreeMonoid α}

/-- The sifts of `x`'s prefixes agree. -/
def PrefAgree (cut cut' : FreeMonoid α → Option Bool) (t : DTree α) (x : FreeMonoid α) : Prop :=
  ∀ j, t.sift cut (prefixOf x j) = t.sift cut' (prefixOf x j)

theorem PrefAgree.whole (h : PrefAgree cut cut' t x) : t.sift cut x = t.sift cut' x := by
  have := h x.toList.length
  rwa [show prefixOf x x.toList.length = x by simp [prefixOf]] at this

theorem kWalkBy_congr (h : PrefAgree cut cut' t x) :
    kWalkBy cut t edges k x = kWalkBy cut' t edges k x := by
  simp only [kWalkBy, h k]

theorem walkToBy_congr (h : PrefAgree cut cut' t x) :
    walkToBy cut t edges k x = walkToBy cut' t edges k x := by
  funext j
  simp only [walkToBy, h k]

theorem agreesAtBy_congr (h : PrefAgree cut cut' t x) :
    agreesAtBy cut t x = agreesAtBy cut' t x := by
  funext w i
  simp only [agreesAtBy, h i]

theorem walkCheckBy_congr (h : PrefAgree cut cut' t x) :
    walkCheckBy cut t edges k x = walkCheckBy cut' t edges k x := by
  have h' : ∀ j, t.sift cut (prefixOf x j) = t.sift cut' (prefixOf x j) := h
  simp only [walkCheckBy, kWalkBy_congr (edges := edges) (k := k) h, h', h.whole,
    walkToBy_congr (edges := edges) (k := k) h]

theorem probeBy_congr (h : PrefAgree cut cut' t x) :
    probeBy cut t edges k x = probeBy cut' t edges k x := by
  simp only [probeBy, walkCheckBy_congr (edges := edges) (k := k) h, agreesAtBy_congr h]

theorem edgeAtBy_congr (h : PrefAgree cut cut' t x) :
    edgeAtBy cut t edges k x = edgeAtBy cut' t edges k x := by
  funext i
  simp only [edgeAtBy, walkToBy_congr (edges := edges) (k := k) h]

theorem posEdgeBy_congr (h : PrefAgree cut cut' t x) :
    posEdgeBy cut t edges k x = posEdgeBy cut' t edges k x := by
  funext i
  simp only [posEdgeBy, edgeAtBy_congr (edges := edges) (k := k) h]

theorem travBy_congr (h : PrefAgree cut cut' t x) :
    travBy cut t edges k x = travBy cut' t edges k x := by
  funext e
  simp only [travBy, posEdgeBy_congr (edges := edges) (k := k) h]

theorem siftsBy_congr (h : PrefAgree cut cut' t x) :
    siftsBy cut t edges k x = siftsBy cut' t edges k x := by
  have h' : ∀ j, t.sift cut (prefixOf x j) = t.sift cut' (prefixOf x j) := h
  simp only [siftsBy, kWalkBy_congr (edges := edges) (k := k) h, h', h.whole,
    walkToBy_congr (edges := edges) (k := k) h, agreesAtBy_congr h]

theorem edgeHarvBy_congr (h : PrefAgree cut cut' t x) :
    edgeHarvBy cut t edges k x = edgeHarvBy cut' t edges k x := by
  funext e
  have h' : ∀ j, t.sift cut (prefixOf x j) = t.sift cut' (prefixOf x j) := h
  simp only [edgeHarvBy, siftsBy_congr (edges := edges) (k := k) h,
    posEdgeBy_congr (edges := edges) (k := k) h, h']

theorem ptHarvBy_congr (h : PrefAgree cut cut' t x) :
    ptHarvBy cut t edges k x = ptHarvBy cut' t edges k x := by
  have h' : ∀ j, t.sift cut (prefixOf x j) = t.sift cut' (prefixOf x j) := h
  simp only [ptHarvBy, probeBy_congr (edges := edges) (k := k) h, h']

end Cut

section Edges

/-- The two edge maps point every leaf's edges at the same targets. -/
def TgtAgree (T : DTree α) (e e' : Edges α) : Prop :=
  ∀ p ∈ T.paths, ∀ c, (e p c).map Prod.fst = (e' p c).map Prod.fst

theorem follow_congr {T : DTree α} {e e' : Edges α} (he : EdgesInto T e) (h : TgtAgree T e e') :
    ∀ (cs : List α) (p : List Bool), p ∈ T.paths → follow e p cs = follow e' p cs
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

variable {cut : FreeMonoid α → Option Bool} {T : DTree α} {e e' : Edges α} {k : ℕ}
  {x : FreeMonoid α}

theorem kWalkBy_econgr (he : EdgesInto T e) (h : TgtAgree T e e') :
    kWalkBy cut T e k x = kWalkBy cut T e' k x := by
  simp only [kWalkBy]
  rcases hs : T.sift cut (prefixOf x k) with p | b
  · simp only []
    rw [follow_congr he h _ p (DTree.sift_mem_paths _ _ _ hs)]
  · rfl

theorem walkToBy_econgr (he : EdgesInto T e) (h : TgtAgree T e e') :
    walkToBy cut T e k x = walkToBy cut T e' k x := by
  funext j
  simp only [walkToBy]
  rcases hs : T.sift cut (prefixOf x k) with p | b
  · simp only []
    rw [follow_congr he h _ p (DTree.sift_mem_paths _ _ _ hs)]
  · rfl

theorem walkCheckBy_econgr (he : EdgesInto T e) (h : TgtAgree T e e') :
    walkCheckBy cut T e k x = walkCheckBy cut T e' k x := by
  simp only [walkCheckBy, kWalkBy_econgr (x := x) (k := k) (cut := cut) he h,
    walkToBy_econgr (x := x) (k := k) (cut := cut) he h]

theorem probeBy_econgr (he : EdgesInto T e) (h : TgtAgree T e e') :
    probeBy cut T e k x = probeBy cut T e' k x := by
  simp only [probeBy, walkCheckBy_econgr (x := x) (k := k) (cut := cut) he h]

theorem posEdgeBy_econgr (he : EdgesInto T e) (h : TgtAgree T e e') :
    posEdgeBy cut T e k x = posEdgeBy cut T e' k x := by
  funext i
  simp only [posEdgeBy, edgeAtBy, walkToBy_econgr (x := x) (k := k) (cut := cut) he h]

theorem travBy_econgr (he : EdgesInto T e) (h : TgtAgree T e e') :
    travBy cut T e k x = travBy cut T e' k x := by
  funext d
  simp only [travBy, posEdgeBy_econgr (x := x) (k := k) (cut := cut) he h]

theorem siftsBy_econgr (he : EdgesInto T e) (h : TgtAgree T e e') :
    siftsBy cut T e k x = siftsBy cut T e' k x := by
  simp only [siftsBy, kWalkBy_econgr (x := x) (k := k) (cut := cut) he h,
    walkToBy_econgr (x := x) (k := k) (cut := cut) he h]

theorem edgeHarvBy_econgr (he : EdgesInto T e) (h : TgtAgree T e e') :
    edgeHarvBy cut T e k x = edgeHarvBy cut T e' k x := by
  funext d
  simp only [edgeHarvBy, siftsBy_econgr (x := x) (k := k) (cut := cut) he h,
    posEdgeBy_econgr (x := x) (k := k) (cut := cut) he h]

theorem ptHarvBy_econgr (he : EdgesInto T e) (h : TgtAgree T e e') :
    ptHarvBy cut T e k x = ptHarvBy cut T e' k x := by
  simp only [ptHarvBy, probeBy_econgr (x := x) (k := k) (cut := cut) he h]

open scoped Classical in
/-- Edge maps out of the leaves into the leaves, each witness `1`. -/
noncomputable def edgeMaps (T : DTree α) : Finset (Edges α) :=
  (Finset.univ : Finset (↥(T.paths.toFinset ×ˢ (Finset.univ : Finset α))
    → Option ↥T.paths.toFinset)).image fun f p c =>
      if h : (p, c) ∈ T.paths.toFinset ×ˢ (Finset.univ : Finset α) then
        (f ⟨(p, c), h⟩).map fun t => (t.1, 1)
      else none

theorem edgeMaps_card (T : DTree α) :
    (edgeMaps T).card ≤ (T.paths.length + 1) ^ (T.paths.length * Fintype.card α) := by
  classical
  refine Finset.card_image_le.trans ?_
  simp only [Finset.card_univ, Fintype.card_fun, Fintype.card_option, Fintype.card_coe,
    Finset.card_product, Finset.card_univ]
  have h := List.toFinset_card_le T.paths
  calc _ ≤ (T.paths.length + 1) ^ (T.paths.toFinset.card * Fintype.card α) := by gcongr
    _ ≤ _ := by
      apply Nat.pow_le_pow_right (by omega)
      gcongr

theorem exists_edgeMaps {T : DTree α} {e : Edges α} (he : EdgesInto T e) :
    ∃ e' ∈ edgeMaps T, TgtAgree T e e' ∧ EdgesInto T e' := by
  classical
  refine ⟨fun p c => if h : (p, c) ∈ T.paths.toFinset ×ˢ (Finset.univ : Finset α) then
      (e p c).map fun q => (q.1, 1) else none, ?_, ?_, ?_⟩
  · refine Finset.mem_image.2 ⟨fun pc => (e pc.1.1 pc.1.2).bind fun q =>
      if hq : q.1 ∈ T.paths.toFinset then some ⟨q.1, hq⟩ else none, Finset.mem_univ _, ?_⟩
    funext p c
    split_ifs with h
    · rcases hpc : e p c with _ | ⟨q, w⟩
      · simp [hpc]
      · have hq : q ∈ T.paths := he _ _ _ _ hpc
        simp [hpc, hq]
    · rfl
  · intro p hp c
    have h : (p, c) ∈ T.paths.toFinset ×ˢ (Finset.univ : Finset α) := by simp [hp]
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

end Edges

end OrthoDFA
