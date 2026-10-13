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

theorem siftCut_congr_up (h : AgreeUpTo N cut cut') {T : DTree α} (hT : MidLe M T)
    {u : FreeMonoid α} (hu : u.toList.length + M ≤ N) : T.siftCut cut u = T.siftCut cut' u :=
  siftCut_congr_mid T u fun m hm => h _ (by
    simp only [FreeMonoid.toList_mul, List.length_append]; have := hT m hm; omega)

theorem prefAgreeK_up (h : AgreeUpTo N cut cut') {T : DTree α} (hT : MidLe M T)
    {x : FreeMonoid α} (hx : x.toList.length + M ≤ N) (k : ℕ) : PrefAgreeK cut cut' T x k :=
  fun j _ => siftCut_congr_up h hT (by have := length_prefixOf_le x j; omega)

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

theorem probeBy_congr_up (h : AgreeUpTo N cut cut') {T : DTree α} (hT : MidLe M T)
    {x : FreeMonoid α} (hx : x.toList.length + M ≤ N) (E : EdgeTable α) (k : ℕ) :
    probeBy cut T E k x = probeBy cut' T E k x :=
  probeBy_congrK (prefAgreeK_up h hT hx k)

theorem recordBy_congr_up (h : AgreeUpTo N cut cut') {T : DTree α} (hT : MidLe M T)
    {x : FreeMonoid α} (hx : x.toList.length + M ≤ N) (E : EdgeTable α) (k : ℕ) :
    recordBy cut k (T, E) x = recordBy cut' k (T, E) x := by
  have hs : ∀ j, T.siftCut cut (prefixOf x j) = T.siftCut cut' (prefixOf x j) := fun j =>
    siftCut_congr_up h hT (by have := length_prefixOf_le x j; omega)
  unfold recordBy
  rw [probeBy_congr_up h hT hx E k]
  simp only [hs]

theorem startHarvestBy_congr_up (h : AgreeUpTo N cut cut') {T : DTree α} (hT : MidLe M T)
    {x : FreeMonoid α} (hx : x.toList.length + M ≤ N) (k : ℕ) :
    startHarvestBy cut T k x = startHarvestBy cut' T k x := by
  unfold startHarvestBy
  rw [siftCut_congr_up h hT (u := prefixOf x k) (by have := length_prefixOf_le x k; omega)]

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
  have hpb := probeBy_congr_up h hT hx s.edges C.walkStart
  have hrb := recordBy_congr_up h hT hx s.edges C.walkStart
  have hcp := countProbe_congr_up h s hT hx C.walkStart
  unfold recordProbe
  rw [hpb, hrb, hcp]
  rcases probeBy cut' s.tree s.edges C.walkStart x with _ | w | w | j | ⟨ps, fd⟩ | u <;>
    try rfl
  try dsimp only
  rcases hc : x.toList[u.toList.length]? with _ | c
  · rfl
  try dsimp only
  have hlen : u.toList.length < x.toList.length := (List.getElem?_eq_some_iff.1 hc).1
  have hs : ∀ c' : α, x.toList[u.toList.length]? = some c' →
      s.tree.siftCut cut (u * FreeMonoid.of c') = s.tree.siftCut cut' (u * FreeMonoid.of c') :=
    fun c' _ => siftCut_congr_up h hT (by simp; omega)
  rw [siftCut_congr_up h hT (u := u) (by omega)]
  rcases s.tree.siftCut cut' u with p | b
  · dsimp only
    rw [hs c hc]
  · rfl

theorem retargetBy_congr_up (h : AgreeUpTo N cut cut') {t' : DTree α} (ht : MidLe M t')
    (p : List Bool) {E : EdgeTable α}
    (hE : ∀ q c t w, E q c = some (t, w) → w.toList.length + 1 + M ≤ N) :
    retargetBy cut t' p E = retargetBy cut' t' p E := by
  funext q c
  unfold retargetBy
  rcases he : E q c with _ | ⟨t, w⟩
  · rfl
  · dsimp only
    split_ifs
    · rw [siftCut_congr_up h ht (u := w * FreeMonoid.of c)
        (by simp; have := hE q c t w he; omega)]
    · rfl

theorem midLe_splitAt {T : DTree α} (hT : ∀ m ∈ T.midfixes, m.toList.length + 1 ≤ M)
    (hM : 1 ≤ M) (c : α) (t : List Bool) (p : List Bool) :
    MidLe M (T.splitAt (FreeMonoid.of c * T.midfixAt t) p) := by
  intro m hm
  rcases midfixes_splitAt _ _ _ m hm with h' | rfl
  · have := hT m h'; omega
  · simp only [FreeMonoid.toList_mul, FreeMonoid.toList_of, List.length_append,
      List.length_singleton]
    rcases midfixAt_mem_or_one T t with h' | h'
    · have := hT _ h'; omega
    · rw [h']; simp; omega

theorem fixEdge_congr_up (h : AgreeUpTo N cut cut') (m : ℕ) (s : Level5State α)
    (hT : ∀ m' ∈ s.tree.midfixes, m'.toList.length + 1 ≤ M) (hM : 1 ≤ M)
    (hE : ∀ q c t w, s.edges q c = some (t, w) → w.toList.length + 1 + M ≤ N)
    (p : List Bool) (c : α) (t : List Bool) :
    fixEdge cut m s p c t = fixEdge cut' m s p c t := by
  unfold fixEdge
  rcases he : s.edges p c with _ | ⟨t₀, w₀⟩
  · rfl
  · dsimp only
    split_ifs
    · rw [retargetBy_congr_up h (midLe_splitAt hT hM c _ p) p hE]
    · rfl

theorem settleOne_congr_up (h : AgreeUpTo N cut cut') (m L : ℕ) (s : Level5State α)
    (hT : ∀ m' ∈ s.tree.midfixes, m'.toList.length + 1 ≤ M) (hM : 1 ≤ M)
    (hE : ∀ q c t w, s.edges q c = some (t, w) → w.toList.length + 1 + M ≤ N) :
    settleOne cut m L s = settleOne cut' m L s := by
  have hf : ∀ p c t, fixEdge cut m s p c t = fixEdge cut' m s p c t :=
    fixEdge_congr_up h m s hT hM hE
  unfold settleOne
  simp only [hf]

end Congr

/-- What a run keeps short: the tree's leaves within `maxLeaves` and its midfixes two shorter, and
the edges' witnesses and the records' prefixes at most `Lx` long. -/
def ReadInv (C : Level5Settings) (Lx : ℕ) (s : Level5State α) : Prop :=
  2 ≤ s.tree.leaves.length ∧ s.tree.leaves.length ≤ C.maxLeaves
    ∧ (∀ m ∈ s.tree.midfixes, m.toList.length + 2 ≤ s.tree.leaves.length)
    ∧ (∀ q c t w, s.edges q c = some (t, w) → w.toList.length ≤ Lx)
    ∧ (∀ q c, ∀ r ∈ s.records q c, r.1.toList.length ≤ Lx)

variable (C : Level5Settings) (cut : FreeMonoid α → Option Bool)

omit [Fintype α] [DecidableEq α] in
theorem recordBy_prefix_len {k : ℕ} {T : DTree α} {E : EdgeTable α} {x : FreeMonoid α}
    {pct : List Bool × α × List Bool} {sp : FreeMonoid α}
    (h : recordBy cut k (T, E) x = some (pct, sp)) : sp.toList.length ≤ x.toList.length := by
  unfold recordBy at h
  split at h
  · split at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      rw [← h.2]
      exact length_prefixOf_le x _
    · cases h
  · cases h

theorem recordProbe_edges (s : Level5State α) (x : FreeMonoid α) (q : List Bool) (c : α)
    (t : List Bool) (w : FreeMonoid α) :
    (recordProbe C cut s x).edges q c = some (t, w) →
      s.edges q c = some (t, w) ∨ w.toList.length < x.toList.length := by
  unfold recordProbe
  rcases ho : probeBy cut s.tree s.edges C.walkStart x with _ | w' | w' | j | ⟨ps, fd⟩ | u <;>
    try dsimp only
  all_goals try (intro h; left; simpa [Level5State.countProbe] using h)
  · rcases recordBy cut C.walkStart (s.tree, s.edges) x with _ | ⟨⟨p, c', t'⟩, sp⟩ <;>
      (intro h; left; simpa [Level5State.countProbe, Level5State.addRecord] using h)
  · rcases hc : x.toList[u.toList.length]? with _ | c' <;> try dsimp only
    · intro h; left; simpa [Level5State.countProbe] using h
    have hlen : u.toList.length < x.toList.length := (List.getElem?_eq_some_iff.1 hc).1
    rcases hp : s.tree.siftCut cut u with p | b <;> try dsimp only
    swap
    · intro h; left; simpa [Level5State.countProbe] using h
    rcases ht : s.tree.siftCut cut (u * FreeMonoid.of c') with t'' | b <;> try dsimp only
    swap
    · intro h; left; simpa [Level5State.countProbe] using h
    rcases he : s.edges p c' with _ | e <;> try dsimp only
    · intro h
      rw [setEdge_eq] at h
      split_ifs at h with hqc
      · simp only [Option.some.injEq, Prod.mk.injEq] at h
        right; rw [← h.2]; exact hlen
      · left; simpa [Level5State.countProbe] using h
    · intro h; left; simpa [Level5State.countProbe] using h

theorem recordProbe_readInv {Lx : ℕ} {s : Level5State α} {x : FreeMonoid α}
    (h : ReadInv C Lx s) (hx : x.toList.length ≤ Lx) : ReadInv C Lx (recordProbe C cut s x) := by
  obtain ⟨h2, hL, hm, hE, hR⟩ := h
  have htree := (recordProbe_cases cut C s x).1
  refine ⟨htree ▸ h2, htree ▸ hL, htree ▸ hm, ?_, ?_⟩
  · intro q c t w hqc
    rcases recordProbe_edges C cut s x q c t w hqc with h' | h'
    · exact hE q c t w h'
    · omega
  · intro q c r hr
    rcases recordProbe_recs cut C s x with ⟨-, hrec⟩ | ⟨p, c', t, sp, hrb, hrec⟩
    · rw [hrec] at hr; exact hR q c r hr
    · rw [hrec] at hr
      simp only [Level5State.addRecord] at hr
      by_cases hq : q = p
      · subst hq
        simp only [Function.update_self] at hr
        by_cases hc : c = c'
        · subst hc
          simp only [Function.update_self, List.mem_append, List.mem_singleton] at hr
          rcases hr with hr | rfl
          · exact hR q c r hr
          · exact (recordBy_prefix_len cut hrb).trans hx
        · rw [Function.update_of_ne hc] at hr; exact hR q c r hr
      · rw [Function.update_of_ne hq] at hr; exact hR q c r hr

theorem getD_find_le {Lx : ℕ} (l : List (FreeMonoid α × List Bool))
    (p : FreeMonoid α × List Bool → Bool) (h : ∀ r ∈ l, r.1.toList.length ≤ Lx) :
    (((l.find? p).map Prod.fst).getD 1).toList.length ≤ Lx := by
  rcases hf : l.find? p with _ | r
  · simp
  · simpa using h r (List.mem_of_find?_eq_some hf)

theorem retargetBy_witness {t' : DTree α} {p : List Bool} {E : EdgeTable α} {q : List Bool}
    {c : α} {t : List Bool} {w : FreeMonoid α} (h : retargetBy cut t' p E q c = some (t, w)) :
    ∃ t₀, E q c = some (t₀, w) := by
  unfold retargetBy at h
  rcases he : E q c with _ | ⟨t₀, w₀⟩ <;> rw [he] at h
  · cases h
  · dsimp only at h
    split_ifs at h
    · rcases hs : t'.siftCut cut (w₀ * FreeMonoid.of c) with a | b <;> rw [hs] at h <;>
        simp only [Sum.elim_inl, Sum.elim_inr, Option.some.injEq, Prod.mk.injEq,
          reduceCtorEq] at h
      exact ⟨t₀, by rw [h.2]⟩
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      exact ⟨t₀, by rw [h.2]⟩

theorem applyProbe_readInv {Lx : ℕ} {s s' : Level5State α} {x : FreeMonoid α}
    (h : ReadInv C Lx (recordProbe C cut s x)) (ha : applyProbe C cut s x = .inl s') :
    ReadInv C Lx s' := by
  obtain ⟨h2, hL, hm, hE, hR⟩ := h
  unfold applyProbe at ha
  simp only [] at ha
  split at ha
  · cases ha
  unfold settleOne at ha
  split_ifs at ha with hv hbig
  · simp only [Sum.inl.injEq] at ha
    subst ha
    set s₁ := recordProbe C cut s x
    have hvi := hv.choose_spec.choose_spec.choose_spec
    refine ⟨?_, not_lt.1 hbig, ?_, ?_, ?_⟩
    · rcases (fixEdge_cases cut C hvi).2 with ⟨ht, -⟩ | ⟨t₀, w₀, -, -, ht⟩
      · rw [ht]; exact h2
      · rw [ht, DTree.length_leaves_splitAt _ _ _ hvi.1]; omega
    · rcases (fixEdge_cases cut C hvi).2 with ⟨ht, -⟩ | ⟨t₀, w₀, -, -, ht⟩
      · rw [ht]; exact hm
      · rw [ht, DTree.length_leaves_splitAt _ _ _ hvi.1]
        intro m hmm
        rcases midfixes_splitAt _ _ _ m hmm with hm' | rfl
        · have := hm m hm'; omega
        · simp only [FreeMonoid.toList_mul, FreeMonoid.toList_of, List.length_append,
            List.length_singleton]
          rcases midfixAt_mem_or_one s₁.tree (commonPrefix hv.choose_spec.choose_spec.choose t₀)
            with h' | h'
          · have := hm _ h'; omega
          · rw [h']; simp; omega
    · intro q c t w hqc
      unfold fixEdge at hqc
      dsimp only at hqc
      split at hqc
      · split_ifs at hqc
        · have hqc' : (if hv.choose <+: q then none
              else retargetBy cut _ hv.choose (recordProbe C cut s x).edges q c) = some (t, w) :=
            hqc
          split_ifs at hqc'
          obtain ⟨t₁, h₁⟩ := retargetBy_witness cut hqc'
          exact hE q c t₁ w h₁
        · rw [setEdge_eq] at hqc
          split_ifs at hqc
          · simp only [Option.some.injEq, Prod.mk.injEq] at hqc
            rw [← hqc.2]; exact getD_find_le _ _ (hR _ _)
          · exact hE q c t w hqc
      · rw [setEdge_eq] at hqc
        split_ifs at hqc
        · simp only [Option.some.injEq, Prod.mk.injEq] at hqc
          rw [← hqc.2]; exact getD_find_le _ _ (hR _ _)
        · exact hE q c t w hqc
    · intro q c r hr
      unfold fixEdge at hr
      dsimp only at hr
      split at hr
      · split_ifs at hr
        · simp at hr
        · exact hR q c r hr
      · exact hR q c r hr
  · simp only [Sum.inl.injEq] at ha
    subst ha
    exact ⟨h2, hL, hm, hE, hR⟩

theorem applyProbe_congr_up {Lx : ℕ} {cut cut' : FreeMonoid α → Option Bool}
    (h : AgreeUpTo (Lx + 1 + C.maxLeaves) cut cut') {s : Level5State α} (hs : ReadInv C Lx s)
    {x : FreeMonoid α} (hx : x.toList.length ≤ Lx) :
    applyProbe C cut s x = applyProbe C cut' s x := by
  have hs' := recordProbe_readInv C cut' hs hx
  obtain ⟨h2, hL, hm, -, -⟩ := hs
  have hT : MidLe C.maxLeaves s.tree := fun m hm' => by have := hm m hm'; omega
  have hrec := recordProbe_congr_up h C s hT (x := x) (by omega)
  obtain ⟨-, hL', hm', hE', -⟩ := hs'
  unfold applyProbe
  rw [hrec]
  dsimp only
  split
  · rfl
  · exact settleOne_congr_up (M := C.maxLeaves) h _ _ _
      (fun m' hm'' => by have := hm' m' hm''; omega) (by omega)
      (fun q c t w hqc => by have := hE' q c t w hqc; omega)

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
    ⟨by simp [initialState, DTree.leaves], by simpa [initialState, DTree.leaves] using hC,
      by simp [initialState, DTree.leaves, DTree.midfixes], by simp [initialState],
      by simp [initialState]⟩
    (fun x hx => by obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx; exact hd i)]

end OrthoDFA
