import OrthoDFA.Proofs.Level4Reads

/-!
# Level 4 reads only through its probes' first `k` letters

A probe reads the cut only at strings that begin with its own first `k` letters and are at least
`k` long, and an edge's witness followed by its letter is such a prefix of the probe that made it.
So over probes whose first `k` letters avoid a set `K₀`, two cuts agreeing on every string at
least `k` long whose first `k` letters avoid `K₀` give the same run (`level4_congr_off`), and the
same search and harvest at every such probe.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- `z` is at least `k` long and its first `k` letters are not in `K₀`. -/
def KeyOff (k : ℕ) (K₀ : Set (FreeMonoid α)) (z : FreeMonoid α) : Prop :=
  k ≤ z.toList.length ∧ prefixOf z k ∉ K₀

/-- A probe level 4 may draw: shorter than `k`, or its first `k` letters off `K₀`. -/
def Avoids (k : ℕ) (K₀ : Set (FreeMonoid α)) (x : FreeMonoid α) : Prop :=
  k ≤ x.toList.length → prefixOf x k ∉ K₀

/-- Two cuts agreeing wherever `KeyOff` holds. -/
def AgreeOff (k : ℕ) (K₀ : Set (FreeMonoid α)) (cut cut' : FreeMonoid α → Option Bool) : Prop :=
  ∀ z, KeyOff k K₀ z → cut z = cut' z

lemma length_prefixOf (x : FreeMonoid α) (j : ℕ) :
    (prefixOf x j).toList.length = min j x.toList.length := by
  simp [prefixOf]

lemma prefixOf_mul_of_le {u : FreeMonoid α} {k : ℕ} (h : k ≤ u.toList.length) (m : FreeMonoid α) :
    prefixOf (u * m) k = prefixOf u k := by
  apply FreeMonoid.toList.injective
  simp only [prefixOf, FreeMonoid.toList_ofList, FreeMonoid.toList_mul]
  rw [List.take_append_of_le_length h]

lemma prefixOf_prefixOf_of_le (x : FreeMonoid α) {j k : ℕ} (h : k ≤ j) :
    prefixOf (prefixOf x j) k = prefixOf x k := by
  apply FreeMonoid.toList.injective
  simp only [prefixOf, FreeMonoid.toList_ofList, List.take_take, min_eq_left h]

lemma prefixOf_mul_of {x : FreeMonoid α} {j : ℕ} {c : α} (hc : x.toList[j]? = some c) :
    prefixOf x j * FreeMonoid.of c = prefixOf x (j + 1) := by
  apply FreeMonoid.toList.injective
  simp only [prefixOf, FreeMonoid.toList_ofList, FreeMonoid.toList_mul, FreeMonoid.toList_of]
  rw [List.take_succ, hc]
  rfl

variable {k : ℕ} {K₀ : Set (FreeMonoid α)}

lemma keyOff_mul {u : FreeMonoid α} (h : KeyOff k K₀ u) (m : FreeMonoid α) :
    KeyOff k K₀ (u * m) :=
  ⟨by simp only [FreeMonoid.toList_mul, List.length_append]; have := h.1; omega,
    by rw [prefixOf_mul_of_le h.1]; exact h.2⟩

lemma keyOff_prefixOf {x : FreeMonoid α} (h : KeyOff k K₀ x) {j : ℕ} (hj : k ≤ j) :
    KeyOff k K₀ (prefixOf x j) :=
  ⟨by rw [length_prefixOf]; exact le_min hj h.1, by rw [prefixOf_prefixOf_of_le x hj]; exact h.2⟩

lemma keyOff_of_avoids {x : FreeMonoid α} (h : Avoids k K₀ x) (hk : k ≤ x.toList.length) :
    KeyOff k K₀ x :=
  ⟨hk, h hk⟩

section Congr

variable {cut cut' : FreeMonoid α → Option Bool}

theorem siftCut_congr_off (h : AgreeOff k K₀ cut cut') (T : DTree α) {u : FreeMonoid α}
    (hu : KeyOff k K₀ u) : T.siftCut cut u = T.siftCut cut' u :=
  siftCut_congr_mid T u fun m _ => h _ (keyOff_mul hu m)

theorem probeStrings_agree_off (h : AgreeOff k K₀ cut cut') (T : DTree α) {x : FreeMonoid α}
    (hx : KeyOff k K₀ x) : ∀ z ∈ probeStrings T k x, cut z = cut' z := by
  intro z hz
  obtain ⟨j, hj, hz⟩ := Finset.mem_biUnion.1 hz
  obtain ⟨m, -, rfl⟩ := Finset.mem_image.1 hz
  exact h _ (keyOff_mul (keyOff_prefixOf hx (Finset.mem_Icc.1 hj).1) m)

theorem prefAgreeK_off (h : AgreeOff k K₀ cut cut') (T : DTree α) {x : FreeMonoid α}
    (hx : KeyOff k K₀ x) : PrefAgreeK cut cut' T x k :=
  fun _ hj => siftCut_congr_off h T (keyOff_prefixOf hx hj)

theorem probeBy_congr_off (h : AgreeOff k K₀ cut cut') (T : DTree α) (E : EdgeTable α)
    {x : FreeMonoid α} (hx : Avoids k K₀ x) : probeBy cut T E k x = probeBy cut' T E k x := by
  by_cases hk : k ≤ x.toList.length
  · exact probeBy_congrK (prefAgreeK_off h T (keyOff_of_avoids hx hk))
  · rw [probeBy_short cut T E (by omega), probeBy_short cut' T E (by omega)]

theorem middleHarvestBy_congr_off (h : AgreeOff k K₀ cut cut') (T : DTree α) (E : EdgeTable α)
    {x : FreeMonoid α} (hx : Avoids k K₀ x) :
    middleHarvestBy cut T E k x = middleHarvestBy cut' T E k x := by
  by_cases hk : k ≤ x.toList.length
  · exact middleHarvestBy_congrK (prefAgreeK_off h T (keyOff_of_avoids hx hk))
  · rw [middleHarvestBy_short cut T E (by omega), middleHarvestBy_short cut' T E (by omega)]

theorem startHarvestBy_congr_off (h : AgreeOff k K₀ cut cut') (T : DTree α)
    {x : FreeMonoid α} (hx : Avoids k K₀ x) : startHarvestBy cut T k x = startHarvestBy cut' T k x := by
  by_cases hk : k ≤ x.toList.length
  · exact startHarvestBy_congr_strings hk (probeStrings_agree_off h T (keyOff_of_avoids hx hk))
  · simp only [startHarvestBy, if_pos (show x.toList.length < k by omega)]

theorem edgeReadsBy_congr_off (h : AgreeOff k K₀ cut cut') (T : DTree α) (E : EdgeTable α)
    {x : FreeMonoid α} (hx : Avoids k K₀ x) :
    edgeReadsBy cut T E k x = edgeReadsBy cut' T E k x := by
  by_cases hk : k ≤ x.toList.length
  · exact edgeReadsBy_congr_strings hk (probeStrings_agree_off h T (keyOff_of_avoids hx hk))
  · funext e
    rw [edgeReadsBy_short cut T E (by omega) e, edgeReadsBy_short cut' T E (by omega) e]

theorem edgeHarvestBy_congr_off (h : AgreeOff k K₀ cut cut') (T : DTree α) (E : EdgeTable α)
    {x : FreeMonoid α} (hx : Avoids k K₀ x) :
    edgeHarvestBy cut T E k x = edgeHarvestBy cut' T E k x := by
  by_cases hk : k ≤ x.toList.length
  · exact edgeHarvestBy_congr_strings hk (probeStrings_agree_off h T (keyOff_of_avoids hx hk))
  · funext e
    rw [edgeHarvestBy_short cut T E (by omega) e, edgeHarvestBy_short cut' T E (by omega) e]

theorem siteHarvest_congr_off (h : AgreeOff k K₀ cut cut') (site : Level4Site α)
    (v : DTree α × EdgeTable α) {x : FreeMonoid α} (hx : Avoids k K₀ x) :
    siteHarvest cut k site v x = siteHarvest cut' k site v x := by
  cases site with
  | start => exact startHarvestBy_congr_off h v.1 hx
  | edge e => exact congrFun (edgeHarvestBy_congr_off h v.1 v.2 hx) e
  | middles => exact middleHarvestBy_congr_off h v.1 v.2 hx

theorem countProbe_congr_off (h : AgreeOff k K₀ cut cut') (s : Level4State α)
    {x : FreeMonoid α} (hx : Avoids k K₀ x) :
    s.countProbe cut k x = s.countProbe cut' k x := by
  unfold Level4State.countProbe
  rw [middleHarvestBy_congr_off h s.tree s.edges hx, startHarvestBy_congr_off h s.tree hx,
    edgeReadsBy_congr_off h s.tree s.edges hx, edgeHarvestBy_congr_off h s.tree s.edges hx]

/-- A clean disagreement is past the walk's start. -/
theorem probeBy_edge_gt {T : DTree α} {E : EdgeTable α} {x : FreeMonoid α}
    {ps : List (List Bool)} {fd : ℕ} (ho : probeBy cut T E k x = .edge ps fd) :
    k < fd ∧ fd ≤ x.toList.length := by
  obtain ⟨ps₀, hi, hw, hb⟩ := probeBy_search cut ho trivial
  obtain ⟨p₀, hk, hf, hkh, hhx, hag⟩ := walkCheckBy_inr cut hw
  obtain ⟨-, hhead, -⟩ := follow_inl _ _ _ hf
  have hlo : agreesAtBy cut T x (fun j => ps₀.getD (j - k) []) k = some true := by
    simp [agreesAtBy, hk, ← hhead, List.getD_eq_getElem?_getD]
  obtain ⟨-, hfd1, hfd2, -, -⟩ := bracketAt_edge _ ps₀ (hi - k) k hi ps fd hkh le_rfl hlo hag hb
  exact ⟨hfd1, hfd2.trans hhx⟩

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_ne_member (agrees : ℕ → Option Bool) (ps : List (List Bool)) (u : FreeMonoid α) :
    ∀ fuel lo hi, bracketAt (α := α) agrees ps fuel lo hi ≠ .member u
  | 0, _, _ => by simp [bracketAt]
  | fuel + 1, lo, hi => by
    unfold bracketAt
    split_ifs
    · split
      · exact bracketAt_ne_member agrees ps u fuel _ _
      · exact bracketAt_ne_member agrees ps u fuel _ _
      · simp
    · simp

/-- A member outcome is a prefix from `k` on. -/
theorem probeBy_member {T : DTree α} {E : EdgeTable α} {x u : FreeMonoid α}
    (ho : probeBy cut T E k x = .member u) : ∃ j, k ≤ j ∧ u = prefixOf x j := by
  rw [probeBy_eq] at ho
  split_ifs at ho with hs
  rcases hw : walkCheckBy cut T E k x with o | ⟨ps, hi⟩ <;> rw [hw] at ho
  · simp only [Sum.elim_inl, id] at ho
    subst ho
    unfold walkCheckBy at hw
    rcases hk : kWalkBy cut T E k x with _ | ⟨s, c, j⟩ | ps' <;> rw [hk] at hw <;>
      dsimp only at hw
    · simp at hw
    · have hj := kWalkBy_edge_ge hk
      split at hw <;> (try split at hw) <;> (try split_ifs at hw) <;>
        first
        | (simp only [Sum.inl.injEq, Outcome.member.injEq] at hw; exact ⟨j, hj, hw.symm⟩)
        | simp at hw
    · split at hw <;> (try split_ifs at hw) <;> simp at hw
  · simp only [Sum.elim_inr] at ho
    exact absurd ho (bracketAt_ne_member _ _ u _ _ _)

theorem recordBy_congr_off (h : AgreeOff k K₀ cut cut') (T : DTree α) (E : EdgeTable α)
    {x : FreeMonoid α} (hx : Avoids k K₀ x) :
    recordBy cut k (T, E) x = recordBy cut' k (T, E) x := by
  unfold recordBy
  rw [probeBy_congr_off h T E hx]
  rcases ho : probeBy cut' T E k x with _ | _ | _ | _ | ⟨ps, fd⟩ | _ | _ <;> try rfl
  obtain ⟨hfd, hfdx⟩ := probeBy_edge_gt ho
  simp only [siftCut_congr_off h T (keyOff_prefixOf (keyOff_of_avoids hx (by omega)) hfd.le)]

/-- What the run keeps that it reads through: every edge's witness followed by its letter, and
every record's prefix followed by its edge's letter, is at least `k` long and off `K₀`. -/
def OffInv (k : ℕ) (K₀ : Set (FreeMonoid α)) (s : Level4State α) : Prop :=
  (∀ q c t w, s.edges q c = some (t, w) → KeyOff k K₀ (w * FreeMonoid.of c))
    ∧ ∀ q c, ∀ r ∈ s.records q c, KeyOff k K₀ (r.1 * FreeMonoid.of c)

variable (C : Level4Settings)

theorem recordProbe_congr_off (h : AgreeOff C.walkStart K₀ cut cut') (s : Level4State α)
    {x : FreeMonoid α} (hx : Avoids C.walkStart K₀ x) :
    recordProbe C cut s x = recordProbe C cut' s x := by
  unfold recordProbe
  rw [probeBy_congr_off h s.tree s.edges hx, recordBy_congr_off h s.tree s.edges hx,
    countProbe_congr_off h s hx]
  rcases ho : probeBy cut' s.tree s.edges C.walkStart x with _ | w | w | j | ⟨ps, fd⟩ | u <;>
    try rfl
  try dsimp only
  obtain ⟨j, hj, rfl⟩ := probeBy_member ho
  rcases hc : x.toList[(prefixOf x j).toList.length]? with _ | c
  · rfl
  try dsimp only
  have hlen := (List.getElem?_eq_some_iff.1 hc).1
  have hkx : C.walkStart ≤ x.toList.length := by rw [length_prefixOf] at hlen; omega
  have hkey := keyOff_prefixOf (keyOff_of_avoids hx hkx) hj
  rw [siftCut_congr_off h s.tree hkey]
  rcases s.tree.siftCut cut' (prefixOf x j) with p | b
  · simp only [siftCut_congr_off h s.tree (keyOff_mul hkey (FreeMonoid.of c))]
  · rfl

theorem retargetBy_congr_off (h : AgreeOff k K₀ cut cut') (t' : DTree α) (p : List Bool)
    {E : EdgeTable α} (hE : ∀ q c t w, E q c = some (t, w) → KeyOff k K₀ (w * FreeMonoid.of c)) :
    retargetBy cut t' p E = retargetBy cut' t' p E := by
  funext q c
  unfold retargetBy
  rcases he : E q c with _ | ⟨t, w⟩
  · rfl
  · dsimp only
    split_ifs
    · rw [siftCut_congr_off h t' (hE q c t w he)]
    · rfl

theorem fixEdge_congr_off (h : AgreeOff k K₀ cut cut') (m : ℕ) (s : Level4State α)
    (hs : OffInv k K₀ s) (p : List Bool) (c : α) (t : List Bool) :
    fixEdge cut m s p c t = fixEdge cut' m s p c t := by
  unfold fixEdge
  rcases he : s.edges p c with _ | ⟨t₀, w₀⟩
  · rfl
  · dsimp only
    split_ifs
    · rw [retargetBy_congr_off h _ p hs.1]
    · rfl

theorem settleOne_congr_off (h : AgreeOff k K₀ cut cut') (m L : ℕ) (s : Level4State α)
    (hs : OffInv k K₀ s) : settleOne cut m L s = settleOne cut' m L s := by
  have hf : ∀ p c t, fixEdge cut m s p c t = fixEdge cut' m s p c t :=
    fixEdge_congr_off h m s hs
  unfold settleOne
  simp only [hf]

theorem recordProbe_offInv {s : Level4State α} (hs : OffInv C.walkStart K₀ s)
    {x : FreeMonoid α} (hx : Avoids C.walkStart K₀ x) :
    OffInv C.walkStart K₀ (recordProbe C cut s x) := by
  obtain ⟨hE, hR⟩ := hs
  refine ⟨?_, ?_⟩
  · intro q c t w hqc
    unfold recordProbe at hqc
    rcases ho : probeBy cut s.tree s.edges C.walkStart x with _ | w' | w' | j | ⟨ps, fd⟩ | u <;>
      rw [ho] at hqc <;> try dsimp only at hqc
    all_goals try (exact hE q c t w (by simpa [Level4State.countProbe] using hqc))
    · rcases hrb : recordBy cut C.walkStart (s.tree, s.edges) x with _ | ⟨⟨p, c', t'⟩, sp⟩ <;>
        rw [hrb] at hqc <;> (try dsimp only at hqc) <;>
        exact hE q c t w (by simpa [Level4State.countProbe, Level4State.addRecord] using hqc)
    · obtain ⟨j, hj, rfl⟩ := probeBy_member ho
      rcases hc : x.toList[(prefixOf x j).toList.length]? with _ | c' <;> rw [hc] at hqc <;>
        (try dsimp only at hqc)
      · exact hE q c t w (by simpa [Level4State.countProbe] using hqc)
      have hlen := (List.getElem?_eq_some_iff.1 hc).1
      have hkx : C.walkStart ≤ x.toList.length := by rw [length_prefixOf] at hlen; omega
      rcases hp : s.tree.siftCut cut (prefixOf x j) with p | b <;> rw [hp] at hqc <;>
        (try dsimp only at hqc)
      swap
      · exact hE q c t w (by simpa [Level4State.countProbe] using hqc)
      rcases ht : s.tree.siftCut cut (prefixOf x j * FreeMonoid.of c') with t'' | b <;>
        rw [ht] at hqc <;> (try dsimp only at hqc)
      swap
      · exact hE q c t w (by simpa [Level4State.countProbe] using hqc)
      rcases he : s.edges p c' with _ | e <;> rw [he] at hqc <;> (try dsimp only at hqc)
      · rw [setEdge_eq] at hqc
        split_ifs at hqc with hqc'
        · simp only [Prod.mk.injEq] at hqc'
          obtain ⟨rfl, rfl⟩ := hqc'
          simp only [Option.some.injEq, Prod.mk.injEq] at hqc
          rw [← hqc.2]
          have hjx : (prefixOf x j).toList.length = j := by
            rw [length_prefixOf] at hlen ⊢; omega
          rw [hjx] at hc
          rw [prefixOf_mul_of hc]
          exact keyOff_prefixOf (keyOff_of_avoids hx hkx) (by omega)
        · exact hE q c t w (by simpa [Level4State.countProbe] using hqc)
      · exact hE q c t w (by simpa [Level4State.countProbe] using hqc)
  · intro q c r hr
    rcases recordProbe_recs cut C s x with ⟨-, hrec⟩ | ⟨p, c', t, sp, hrb, hrec⟩
    · rw [hrec] at hr; exact hR q c r hr
    · rw [hrec] at hr
      simp only [Level4State.addRecord] at hr
      by_cases hq : q = p
      · subst hq
        simp only [Function.update_self] at hr
        by_cases hc : c = c'
        · subst hc
          simp only [Function.update_self, List.mem_append, List.mem_singleton] at hr
          rcases hr with hr | rfl
          · exact hR q c r hr
          · -- the record's prefix and letter are the probe's prefix at the disagreement
            unfold recordBy at hrb
            rcases ho : probeBy cut s.tree s.edges C.walkStart x with
              _ | _ | _ | _ | ⟨ps, fd⟩ | _ | _ <;> rw [ho] at hrb <;>
              simp only [reduceCtorEq] at hrb
            obtain ⟨hfd, hfdx⟩ := probeBy_edge_gt ho
            rcases hcx : x.toList[fd - 1]? with _ | c₁ <;> rw [hcx] at hrb
            · simp at hrb
            rcases hts : s.tree.siftCut cut (prefixOf x fd) with t₁ | b <;> rw [hts] at hrb
            swap
            · simp at hrb
            simp only [Option.some.injEq, Prod.mk.injEq] at hrb
            obtain ⟨⟨-, rfl, -⟩, rfl⟩ := hrb
            dsimp only
            rw [prefixOf_mul_of hcx, show fd - 1 + 1 = fd by omega]
            exact keyOff_prefixOf (keyOff_of_avoids hx (by omega)) hfd.le
        · rw [Function.update_of_ne hc] at hr; exact hR q c r hr
      · rw [Function.update_of_ne hq] at hr; exact hR q c r hr

theorem applyProbe_offInv (hm : 1 ≤ C.recordsToMove) {s s' : Level4State α} {x : FreeMonoid α}
    (h : OffInv C.walkStart K₀ (recordProbe C cut s x)) (ha : applyProbe C cut s x = .inl s') :
    OffInv C.walkStart K₀ s' := by
  obtain ⟨hE, hR⟩ := h
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
    set p := hv.choose
    set c := hv.choose_spec.choose
    set t := hv.choose_spec.choose_spec.choose
    -- the violating target has a record, so the fix's witness is a record's prefix
    have hfind : ∃ r ∈ s₁.records p c, r.2 = t := by
      have h1 : 1 ≤ s₁.recordCount p c t := hm.trans hvi.2.2
      unfold Level4State.recordCount at h1
      obtain ⟨r, hr⟩ := List.exists_mem_of_length_pos (by omega : 0 < ((s₁.records p c).filter
        (·.2 = t)).length)
      obtain ⟨hr1, hr2⟩ := List.mem_filter.1 hr
      exact ⟨r, hr1, by simpa using hr2⟩
    have hwit : KeyOff C.walkStart K₀
        ((((s₁.records p c).find? (·.2 = t)).map Prod.fst).getD 1 * FreeMonoid.of c) := by
      obtain ⟨r, hr, hrt⟩ := hfind
      have hsome : ((s₁.records p c).find? (·.2 = t)).isSome :=
        List.find?_isSome.2 ⟨r, hr, by simp [hrt]⟩
      obtain ⟨r', hf⟩ := Option.isSome_iff_exists.1 hsome
      rw [hf]
      simp only [Option.map_some, Option.getD_some]
      exact hR p c r' (List.mem_of_find?_eq_some hf)
    refine ⟨?_, ?_⟩
    · intro q c' t' w hqc
      unfold fixEdge at hqc
      dsimp only at hqc
      split at hqc
      · split_ifs at hqc
        · have hqc' : (if p <+: q then none
              else retargetBy cut _ p s₁.edges q c') = some (t', w) := hqc
          split_ifs at hqc'
          obtain ⟨t₁, h₁⟩ := retargetBy_witness cut hqc'
          exact hE q c' t₁ w h₁
        · rw [setEdge_eq] at hqc
          split_ifs at hqc with hpq
          · simp only [Prod.mk.injEq] at hpq
            obtain ⟨rfl, rfl⟩ := hpq
            simp only [Option.some.injEq, Prod.mk.injEq] at hqc
            rw [← hqc.2]; exact hwit
          · exact hE q c' t' w hqc
      · rw [setEdge_eq] at hqc
        split_ifs at hqc with hpq
        · simp only [Prod.mk.injEq] at hpq
          obtain ⟨rfl, rfl⟩ := hpq
          simp only [Option.some.injEq, Prod.mk.injEq] at hqc
          rw [← hqc.2]; exact hwit
        · exact hE q c' t' w hqc
    · intro q c' r hr
      unfold fixEdge at hr
      dsimp only at hr
      split at hr
      · split_ifs at hr
        · simp at hr
        · exact hR q c' r hr
      · exact hR q c' r hr
  · simp only [Sum.inl.injEq] at ha
    subst ha
    exact ⟨hE, hR⟩

theorem applyProbe_congr_off (h : AgreeOff C.walkStart K₀ cut cut') {s : Level4State α}
    (hs : OffInv C.walkStart K₀ s) {x : FreeMonoid α} (hx : Avoids C.walkStart K₀ x) :
    applyProbe C cut s x = applyProbe C cut' s x := by
  have hs' := recordProbe_offInv C (cut := cut') hs hx
  have hrec := recordProbe_congr_off C h s hx
  unfold applyProbe
  rw [hrec]
  dsimp only
  split
  · rfl
  · exact settleOne_congr_off h _ _ _ hs'

theorem runProbes_congr_off (hm : 1 ≤ C.recordsToMove) (h : AgreeOff C.walkStart K₀ cut cut') :
    ∀ (s : Level4State α) (xs : List (FreeMonoid α)), OffInv C.walkStart K₀ s →
      (∀ x ∈ xs, Avoids C.walkStart K₀ x) → runProbes C cut s xs = runProbes C cut' s xs
  | s, [], _, _ => rfl
  | s, x :: xs, hs, hxs => by
    have hx := hxs x List.mem_cons_self
    simp only [runProbes]
    rw [applyProbe_congr_off C h hs hx]
    rcases ha : applyProbe C cut' s x with s' | ⟨e, s'⟩
    · have hinv : OffInv C.walkStart K₀ s' :=
        applyProbe_offInv C hm (recordProbe_offInv C hs hx) ha
      exact runProbes_congr_off hm h s' xs hinv (fun y hy => hxs y (List.mem_cons_of_mem _ hy))
    · rfl

end Congr

/-- Level 4 over probes avoiding `K₀` reads only strings `KeyOff` holds of. -/
theorem level4_congr_off (C : Level4Settings) (hm : 1 ≤ C.recordsToMove) {T : ℕ}
    (draws : Fin T → FreeMonoid α) (hd : ∀ i, Avoids C.walkStart K₀ (draws i))
    {read read' : FreeMonoid α → ARU}
    (h : ∀ z, KeyOff C.walkStart K₀ z → read z = read' z) :
    indecisiveLStarLevel4 C read draws = indecisiveLStarLevel4 C read' draws := by
  unfold indecisiveLStarLevel4
  rw [runProbes_congr_off C (cut := fun z => (read z).cut) (cut' := fun z => (read' z).cut) hm
    (fun z hz => by simp only [h z hz]) initialState (List.ofFn draws)
    ⟨by simp [initialState], by simp [initialState]⟩
    (fun x hx => by obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx; exact hd i)]

end OrthoDFA
