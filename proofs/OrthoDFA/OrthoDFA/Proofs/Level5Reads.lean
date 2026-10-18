import OrthoDFA.Proofs.Level5Short

/-!
# Level 5 reads only short strings

Over probes at most `Lx` long, a run reads the cut only at a probe's prefix, or one followed by a
letter, then a midfix, and an edge's witness is a probe's prefix. The run keeps its midfixes
shorter than `maxLeaves`, so two cuts agreeing on the strings at most `Lx + 1 + maxLeaves` long
give the same run (`runProbes_congr_up`).
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- Two cuts agreeing on every string at most `N` long. -/
def AgreeUpTo (N : ℕ) (cut cut' : FreeMonoid α → Option Bool) : Prop :=
  ∀ z : FreeMonoid α, z.toList.length ≤ N → cut z = cut' z

/-- The midfixes of `T` are at most `M` long. -/
def MidLe (M : ℕ) (T : DTree α) : Prop := ∀ m ∈ T.midfixes, m.toList.length ≤ M

section Congr

variable {N M : ℕ} {cut cut' : FreeMonoid α → Option Bool}

theorem sift_congr_up (h : AgreeUpTo N cut cut') {T : DTree α} (hT : MidLe M T)
    {u : FreeMonoid α} (hu : u.toList.length + M ≤ N) :
    T.sift (ofCut cut) u = T.sift (ofCut cut') u :=
  sift_congr_mid T u fun m hm => h _ (by
    simp only [FreeMonoid.toList_mul, List.length_append]; have := hT m hm; omega)

theorem prefAgreeK_up (h : AgreeUpTo N cut cut') {T : DTree α} (hT : MidLe M T)
    {x : FreeMonoid α} (hx : x.toList.length + M ≤ N) (k : ℕ) : PrefAgreeK cut cut' T x k :=
  fun j _ => sift_congr_up h hT (by have := length_prefixOf_le x j; omega)

theorem prefStrings_agree_up (h : AgreeUpTo N cut cut') {T : DTree α} (hT : MidLe M T)
    {x : FreeMonoid α} (hx : x.toList.length + M ≤ N) :
    ∀ z ∈ prefStrings T x, cut z = cut' z := by
  intro z hz
  obtain ⟨j, -, hz⟩ := Finset.mem_biUnion.1 hz
  obtain ⟨m, hm, rfl⟩ := Finset.mem_image.1 hz
  refine h _ ?_
  simp only [FreeMonoid.toList_mul, List.length_append]
  have := length_prefixOf_le x j
  have := hT m hm
  omega

theorem probeOrAgree_congr_up (h : AgreeUpTo N cut cut') {T : DTree α} (hT : MidLe M T)
    {x : FreeMonoid α} (hx : x.toList.length + M ≤ N) (E : Edges α) (k : ℕ) :
    probeOrAgree (ofCut cut) T E k x = probeOrAgree (ofCut cut') T E k x :=
  probeOrAgree_congrK (prefAgreeK_up h hT hx k)

theorem probeOrAgree_congr_up_rd {rd rd' : FreeMonoid α → ARU}
    (h : AgreeUpTo N (fun z => (rd z).cut) (fun z => (rd' z).cut)) {T : DTree α} (hT : MidLe M T)
    {x : FreeMonoid α} (hx : x.toList.length + M ≤ N) (E : Edges α) (k : ℕ) :
    probeOrAgree rd T E k x = probeOrAgree rd' T E k x := by
  have := probeOrAgree_congr_up h hT hx E k
  rwa [ofCut_rd, ofCut_rd] at this

theorem recordBy_congr_up (h : AgreeUpTo N cut cut') {T : DTree α} (hT : MidLe M T)
    {x : FreeMonoid α} (hx : x.toList.length + M ≤ N) (E : Edges α) (k : ℕ) :
    recordBy cut k (T, E) x = recordBy cut' k (T, E) x := by
  unfold recordBy
  rw [probeOrAgree_congr_up h hT hx E k]

theorem startHarvestBy_congr_up (h : AgreeUpTo N cut cut') {T : DTree α} (hT : MidLe M T)
    {x : FreeMonoid α} (hx : x.toList.length + M ≤ N) (k : ℕ) :
    startHarvestBy cut T k x = startHarvestBy cut' T k x := by
  unfold startHarvestBy
  rw [sift_congr_up h hT (u := prefixOf x k) (by have := length_prefixOf_le x k; omega)]

theorem countProbe_congr_up (h : AgreeUpTo N cut cut') (s : Level5State α) (hT : MidLe M s.tree)
    {x : FreeMonoid α} (hx : x.toList.length + M ≤ N) (k : ℕ) :
    s.countProbe cut k x = s.countProbe cut' k x := by
  unfold Level5State.countProbe
  rw [middleHarvestBy_congrK (prefAgreeK_up h hT hx k), startHarvestBy_congr_up h hT hx k,
    edgeReadsBy_congr_pref (prefStrings_agree_up h hT hx),
    edgeHarvestBy_congr_pref (prefStrings_agree_up h hT hx)]

theorem recordProbe_congr_up (h : AgreeUpTo N cut cut') (C : Level5Settings) (s : Level5State α)
    (hT : MidLe M s.tree) {x : FreeMonoid α} (hx : x.toList.length + M ≤ N) :
    recordProbe C cut s x = recordProbe C cut' s x := by
  unfold recordProbe
  rw [probeOrAgree_congr_up h hT hx s.edges C.walkStart, countProbe_congr_up h s hT hx C.walkStart]

theorem retarget_congr_up (h : AgreeUpTo N cut cut') {t' : DTree α} (ht : MidLe M t')
    (p : List Bool) {E : Edges α}
    (hE : ∀ q c t w, E (q, c) = some (t, w) → w.toList.length + 1 + M ≤ N) :
    retarget (ofCut cut) t' p E = retarget (ofCut cut') t' p E := by
  funext ⟨q, c⟩
  unfold retarget
  split_ifs
  · rfl
  rcases he : E (q, c) with _ | ⟨t, w⟩
  · rfl
  · dsimp only
    split_ifs
    · rw [sift_congr_up h ht (u := w * FreeMonoid.of c)
        (by simp; have := hE q c t w he; omega)]
    · rfl

theorem midLe_splitAt {T : DTree α} (hT : ∀ m ∈ T.midfixes, m.toList.length + 1 ≤ M)
    (hM : 1 ≤ M) (c : α) (t : List Bool) (p : List Bool) :
    MidLe M (T.splitAt (FreeMonoid.of c * T.midfixAt t) p) := by
  intro m hm
  rcases Finset.mem_insert.1 (midfixes_splitAt _ T p hm) with rfl | h'
  · simp only [FreeMonoid.toList_mul, FreeMonoid.toList_of, List.length_append,
      List.length_singleton]
    rcases midfixAt_mem T t with h' | h'
    · rw [h']; simp; omega
    · have := hT _ h'; omega
  · have := hT m h'; omega

theorem splitState_congr_up (h : AgreeUpTo N cut cut') (s : Level5State α)
    (hT : ∀ m ∈ s.tree.midfixes, m.toList.length + 1 ≤ M) (hM : 1 ≤ M)
    (hE : ∀ q c t w, s.edges (q, c) = some (t, w) → w.toList.length + 1 + M ≤ N)
    (p : List Bool) (c : α) (t : List Bool) :
    splitState cut s p c t = splitState cut' s p c t := by
  unfold splitState
  rcases he : s.edges (p, c) with _ | ⟨t₀, w₀⟩
  · rfl
  · simp only [HypothesisDFA.split]
    rw [retarget_congr_up h (midLe_splitAt hT hM c _ p) p hE]

end Congr

/-- What a run keeps short: the tree's leaves within `maxLeaves` and its midfixes two shorter, and
the edges' witnesses at most `Lx` long. -/
def ReadInv (C : Level5Settings) (Lx : ℕ) (s : Level5State α) : Prop :=
  2 ≤ s.tree.leaves.length ∧ s.tree.leaves.length ≤ C.maxLeaves
    ∧ (∀ m ∈ s.tree.midfixes, m.toList.length + 2 ≤ s.tree.leaves.length)
    ∧ ∀ q c t w, s.edges (q, c) = some (t, w) → w.toList.length ≤ Lx

variable (C : Level5Settings) (cut : FreeMonoid α → Option Bool)

theorem recordProbe_edges (s : Level5State α) (x : FreeMonoid α) (q : List Bool) (c : α)
    (t : List Bool) (w : FreeMonoid α) :
    (recordProbe C cut s x).edges (q, c) = some (t, w) →
      s.edges (q, c) = some (t, w) ∨ w.toList.length < x.toList.length := by
  rcases recordProbe_spec C cut s x with h | ⟨u, c', p, t', -, -, hc, -, -, h⟩ <;> rw [h]
  · exact .inl
  · intro he
    rw [setEdge_eq] at he
    split_ifs at he
    · simp only [Option.some.injEq, Prod.mk.injEq] at he
      right; rw [← he.2]; exact (List.getElem?_eq_some_iff.1 hc).1
    · exact .inl he

theorem recordProbe_readInv {Lx : ℕ} {s : Level5State α} {x : FreeMonoid α}
    (h : ReadInv C Lx s) (hx : x.toList.length ≤ Lx) : ReadInv C Lx (recordProbe C cut s x) := by
  obtain ⟨h2, hL, hm, hE⟩ := h
  have htree := recordProbe_tree C cut s x
  refine ⟨htree ▸ h2, htree ▸ hL, htree ▸ hm, ?_⟩
  intro q c t w hqc
  rcases recordProbe_edges C cut s x q c t w hqc with h' | h'
  · exact hE q c t w h'
  · omega

theorem retarget_witness {t' : DTree α} {p : List Bool} {E : Edges α} {q : List Bool}
    {c : α} {t : List Bool} {w : FreeMonoid α}
    (h : retarget (ofCut cut) t' p E (q, c) = some (t, w)) : ∃ t₀, E (q, c) = some (t₀, w) := by
  unfold retarget at h
  split_ifs at h
  rcases he : E (q, c) with _ | ⟨t₀, w₀⟩ <;> rw [he] at h
  · cases h
  · dsimp only at h
    split_ifs at h
    · rcases hs : t'.sift (ofCut cut) (w₀ * FreeMonoid.of c) with a | b <;> rw [hs] at h <;>
        simp only [Sum.elim_inl, Sum.elim_inr, Option.some.injEq, Prod.mk.injEq,
          reduceCtorEq] at h
      exact ⟨t₀, by rw [h.2]⟩
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      exact ⟨t₀, by rw [h.2]⟩

theorem splitState_witness {s : Level5State α} {p : List Bool} {c : α} {t : List Bool}
    {q : List Bool} {c' : α} {t' : List Bool} {w : FreeMonoid α}
    (h : (splitState cut s p c t).edges (q, c') = some (t', w)) :
    ∃ t₁, s.edges (q, c') = some (t₁, w) := by
  unfold splitState at h
  split at h
  · simp only [freshState, HypothesisDFA.split] at h
    exact retarget_witness cut h
  · exact ⟨t', h⟩

theorem applyProbe_readInv {Lx : ℕ} {s s' : Level5State α} {x : FreeMonoid α}
    (h : ReadInv C Lx (recordProbe C cut s x)) (ha : applyProbe C cut s x = .inl s') :
    ReadInv C Lx s' := by
  have hts : TreeShort C s'.tree := treeShort_step C cut
    (by rw [← recordProbe_tree C cut s x]; exact ⟨h.1, h.2.1, h.2.2.1⟩) ha
  refine ⟨hts.1, hts.2.1, hts.2.2, ?_⟩
  rcases (applyProbe_inl C cut ha).2 with ⟨-, rfl⟩ | ⟨p, c, t, sp, -, rfl, -⟩
  · exact h.2.2.2
  · intro q c' t' w hqc
    obtain ⟨t₁, h₁⟩ := splitState_witness cut hqc
    exact h.2.2.2 q c' t₁ w h₁

theorem applyProbe_congr_up {Lx : ℕ} {cut cut' : FreeMonoid α → Option Bool}
    (h : AgreeUpTo (Lx + 1 + C.maxLeaves) cut cut') {s : Level5State α} (hs : ReadInv C Lx s)
    {x : FreeMonoid α} (hx : x.toList.length ≤ Lx) :
    applyProbe C cut s x = applyProbe C cut' s x := by
  have hs' := recordProbe_readInv C cut' hs hx
  obtain ⟨h2, hL, hm, -⟩ := hs
  have hT : MidLe C.maxLeaves s.tree := fun m hm' => by have := hm m hm'; omega
  have hrec := recordProbe_congr_up h C s hT (x := x) (by omega)
  have hrb := recordBy_congr_up h hT (x := x) (by omega) s.edges C.walkStart
  obtain ⟨-, hL', hm', hE'⟩ := hs'
  unfold applyProbe
  rw [hrec, hrb]
  dsimp only
  split
  · rfl
  · split
    · rw [splitState_congr_up (M := C.maxLeaves) h _
        (fun m' hm'' => by have := hm' m' hm''; omega) (by omega)
        (fun q c t w hqc => by have := hE' q c t w hqc; omega)]
    · rfl

theorem runProbes_congr_up {Lx : ℕ} {cut cut' : FreeMonoid α → Option Bool}
    (h : AgreeUpTo (Lx + 1 + C.maxLeaves) cut cut') :
    ∀ (s : Level5State α) (xs : List (FreeMonoid α)), ReadInv C Lx s →
      (∀ x ∈ xs, x.toList.length ≤ Lx) → runProbes C cut s xs = runProbes C cut' s xs
  | s, [], _, _ => rfl
  | s, x :: xs, hs, hxs => by
    have hx := hxs x List.mem_cons_self
    simp only [runProbes]
    rw [applyProbe_congr_up C h hs hx]
    rcases ha : applyProbe C cut' s x with s' | ⟨e, s'⟩
    · have hinv : ReadInv C Lx s' :=
        applyProbe_readInv C cut' (recordProbe_readInv C cut' hs hx) ha
      exact runProbes_congr_up h s' xs hinv (fun y hy => hxs y (List.mem_cons_of_mem _ hy))
    · rfl

/-- Level 5 over probes at most `Lx` long reads only strings at most `Lx + 1 + maxLeaves`
long. -/
theorem level5_congr_up (hC : 2 ≤ C.maxLeaves) {T : ℕ} (draws : Fin T → FreeMonoid α) {Lx : ℕ}
    (hd : ∀ i, (draws i).toList.length ≤ Lx) {read read' : FreeMonoid α → ARU}
    (h : ∀ z : FreeMonoid α, z.toList.length ≤ Lx + 1 + C.maxLeaves → read z = read' z) :
    indecisiveLStarLevel5 C read draws = indecisiveLStarLevel5 C read' draws := by
  unfold indecisiveLStarLevel5
  rw [runProbes_congr_up C (cut := fun z => (read z).cut) (cut' := fun z => (read' z).cut)
    (fun z hz => by simp only [h z hz]) initialState (List.ofFn draws)
    ⟨by simp [initialState, freshState, DTree.leaves],
      by simpa [initialState, freshState, DTree.leaves] using hC,
      by simp [initialState, freshState, DTree.leaves, DTree.midfixes],
      by simp [initialState, freshState]⟩
    (fun x hx => by obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx; exact hd i)]

end OrthoDFA
