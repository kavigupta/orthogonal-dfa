import OrthoDFA.Proofs.Level5Steps

/-!
# Reads on the rarer side

A split is genuine unless a read on its state's rarer decided side made the contradiction's
record, or the edge it contradicts, untrue. A step reads its probe's prefixes, each followed by a
midfix of the tree (`potMid`), and, where it splits, every learned edge's witness's successor
followed by a midfix of the new tree (`splitReads`). Where none of those is read on the rarer side
(`WrongStep`), the hypothesis stays true: its tree grown by genuine splits alone, so within
`|Q| + 2` leaves, and its edges true records, judged by their states (`truthInv_step`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]


namespace ReadModel

variable {σ : Type*} (G : ReadModel α σ)

/-- A read on its state's rarer decided side. -/
def WrongRead (rd : FreeMonoid α → ARU) (z : FreeMonoid α) : Prop :=
  (rd z).cut = some (!G.side (G.M.eval z.toList))

end ReadModel

section Chance

attribute [local instance] ARU.fintype

variable {σ : Type*} (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] {read : FreeMonoid α → Ω → ARU}

/-- A string is read on its state's rarer side with chance at most `ε + κ`. -/
theorem wrongRead_le (hmeas : ∀ w, Measurable (read w))
    (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.readDist (G.M.eval w.toList) r) (hκ : 0 ≤ G.κ)
    (hε : 0 ≤ G.ε) (z : FreeMonoid α) :
    μ {ω | G.WrongRead (read · ω) z} ≤ ENNReal.ofReal (G.ε + G.κ) := by
  have hset : {ω | G.WrongRead (read · ω) z} = {ω | read z ω = G.minVal (G.M.eval z.toList)} := by
    ext ω
    simp only [ReadModel.WrongRead, Set.mem_setOf_eq]
    exact cut_minVal G _ _
  rw [hset, measure_read G hlaw, dist_minVal]
  refine ENNReal.ofReal_le_ofReal ((G.rare _).trans (max_le (by linarith) ?_))
  have hu : G.readDist (G.M.eval z.toList) .undecided ≤ 1 := by
    rw [← hlaw z .undecided]; exact measureReal_le_one
  nlinarith [G.nonneg (G.M.eval z.toList) .undecided]

end Chance

section Mids

theorem midfixes_splitAt (d : FreeMonoid α) :
    ∀ (T : DTree α) (p : List Bool), (T.splitAt d p).midfixes ⊆ insert d T.midfixes
  | .leaf, [] => by simp [DTree.splitAt, DTree.midfixes]
  | .leaf, _ :: _ => by simp [DTree.splitAt, DTree.midfixes]
  | .node m r a, [] => by simp [DTree.splitAt]
  | .node m r a, false :: p => by
    intro z hz
    simp only [DTree.splitAt, DTree.midfixes, Finset.mem_insert, Finset.mem_union] at hz ⊢
    rcases hz with rfl | hz | hz
    · exact .inr (.inl rfl)
    · rcases Finset.mem_insert.1 (midfixes_splitAt d r p hz) with rfl | h
      · exact .inl rfl
      · exact .inr (.inr (.inl h))
    · exact .inr (.inr (.inr hz))
  | .node m r a, true :: p => by
    intro z hz
    simp only [DTree.splitAt, DTree.midfixes, Finset.mem_insert, Finset.mem_union] at hz ⊢
    rcases hz with rfl | hz | hz
    · exact .inr (.inl rfl)
    · exact .inr (.inr (.inl hz))
    · rcases Finset.mem_insert.1 (midfixes_splitAt d a p hz) with rfl | h
      · exact .inl rfl
      · exact .inr (.inr (.inr h))

end Mids

section Reads

variable {σ : Type*} (G : ReadModel α σ) (rd : FreeMonoid α → ARU) (C : Level5Settings)

/-- Every prefix of `x` followed by a midfix of `T`. -/
noncomputable def potMid (x : FreeMonoid α) (T : DTree α) : Finset (FreeMonoid α) :=
  (Finset.range (x.toList.length + 1)).biUnion fun i => T.midfixes.image (prefixOf x i * ·)

theorem mem_potMid {x : FreeMonoid α} {T : DTree α} {i : ℕ} (hi : i ≤ x.toList.length)
    {m : FreeMonoid α} (hm : m ∈ T.midfixes) : prefixOf x i * m ∈ potMid x T :=
  Finset.mem_biUnion.2 ⟨i, Finset.mem_range.2 (by omega), Finset.mem_image.2 ⟨m, hm, rfl⟩⟩

open scoped Classical in
/-- Each learned edge's witness's successor followed by a midfix of `T'`. -/
noncomputable def resifts (s : Level5State α) (T' : DTree α) : Finset (FreeMonoid α) :=
  (s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α)).biUnion fun e =>
    match s.edges e.1 e.2 with
    | some (_, w) => T'.midfixes.image ((w * FreeMonoid.of e.2) * ·)
    | none => ∅

/-- What a step's split re-sifts. -/
noncomputable def splitReads (cut : FreeMonoid α → Option Bool) (s : Level5State α)
    (x : FreeMonoid α) : Finset (FreeMonoid α) :=
  match recordBy cut C.walkStart (s.tree, s.edges) x with
  | some ((p, c, t), _) =>
    match s.edges p c with
    | some (t₀, _) =>
      resifts s (s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p)
    | none => ∅
  | none => ∅

/-- The step reads a string, a prefix of the probe followed by a midfix or one its split
re-sifts, on its state's rarer side. -/
def WrongStep (s : Level5State α) (x : FreeMonoid α) : Prop :=
  ∃ z ∈ potMid x s.tree ∪ splitReads C (fun z => (rd z).cut) s x, G.WrongRead rd z

end Reads

section Truth

variable {σ : Type*} (G : ReadModel α σ) (rd : FreeMonoid α → ARU)

/-- A sift through midfixes none of whose reads is on the rarer side lands at the true leaf. -/
theorem leafOf_of_sift {T : DTree α} {z : FreeMonoid α} {p : List Bool}
    (hc : ∀ m ∈ T.midfixes, ¬ G.WrongRead rd (z * m))
    (h : T.siftCut (fun z => (rd z).cut) z = .inl p) : G.leafOf T (G.M.eval z.toList) = p := by
  by_contra hne
  obtain ⟨w, hw, hcw⟩ := minority_of_sift G T z p h hne
  obtain ⟨m, hm, rfl⟩ := route_mid _ T z w hw
  exact hc m hm hcw

/-- Every learned edge is a true record. -/
def TrueEdges (s : Level5State α) : Prop :=
  ∀ p c t w, s.edges p c = some (t, w) → G.TrueRec s.tree (p, c, t) w

/-- A state of a round with no read on the rarer side: its tree grown by genuine splits alone,
its edges into its leaves and true, its leaves at most `n`. -/
def TruthInv (n : ℕ) (s : Level5State α) : Prop :=
  G.Grown s.tree 0 ∧ EdgesInto s.tree s.edges ∧ TrueEdges G s ∧ s.tree.leaves.length ≤ n

theorem truthInv_start {n : ℕ} (hn : 2 ≤ n) : TruthInv G n (initialState : Level5State α) :=
  ⟨.start, fun _ _ _ _ h => by simp [initialState, freshState] at h,
    fun _ _ _ _ h => by simp [initialState, freshState] at h,
    by simp [initialState, freshState, DTree.leaves]; omega⟩

/-- Learned edges leave leaves. -/
theorem edge_leaf {s : Level5State α} (ht : TrueEdges G s) {p : List Bool} {c : α}
    {t : List Bool} {w : FreeMonoid α} (h : s.edges p c = some (t, w)) : p ∈ s.tree.leaves := by
  have h1 : G.leafOf s.tree (G.M.eval w.toList) = p := (ht p c t w h).1
  rw [← h1]
  exact G.leafOf_mem_paths _ _

variable (cut : FreeMonoid α → Option Bool)

theorem probeBy_member {T : DTree α} {edges : EdgeTable α} {k : ℕ} {x u : FreeMonoid α}
    (h : probeBy cut T edges k x = .member u) :
    ∃ j, k ≤ j ∧ j < x.toList.length ∧ u = prefixOf x j := by
  rw [probeBy_eq] at h
  split_ifs at h with hs
  rcases hw : walkCheckBy cut T edges k x with o | ⟨ps, hi⟩ <;> rw [hw] at h
  swap
  · simp only [Sum.elim_inr] at h
    have := bracketAt_isSearch (α := α) (agreesAtBy cut T x fun j => ps.getD (j - k) []) ps
      (hi - k) k hi
    rw [h] at this
    exact this.elim
  simp only [Sum.elim_inl, id] at h
  subst h
  unfold walkCheckBy at hw
  rcases hk : kWalkBy cut T edges k x with _ | ⟨s, c, j⟩ | ps' <;> rw [hk] at hw
  · simp at hw
  · simp only [] at hw
    split at hw
    · simp at hw
    · split at hw
      · simp at hw
      · split_ifs at hw
        · simp only [Sum.inl.injEq, WalkOutcome.member.injEq] at hw
          subst hw
          refine ⟨j, kWalkBy_edge_ge hk, ?_, rfl⟩
          unfold kWalkBy at hk
          split at hk
          · simp at hk
          · split at hk
            · simp at hk
            · rename_i p₀ _ s' c' i hf
              simp only [KWalk.edge.injEq] at hk
              obtain ⟨rfl, rfl, rfl⟩ := hk
              obtain ⟨hi, -⟩ := follow_inr _ _ _ _ _ hf
              simp only [List.length_drop] at hi
              omega
  · simp only [] at hw
    split at hw
    · simp at hw
    · split_ifs at hw <;> simp at hw

/-- A record's parts: its probe searched to the edge it is at, its prefix the probe's before the
edge, which its walk agrees with, and its target where the next prefix sifts. -/
theorem recordBy_parts {T : DTree α} {edges : EdgeTable α} {k : ℕ} {x : FreeMonoid α}
    {p : List Bool} {c : α} {t : List Bool} {sp : FreeMonoid α}
    (h : recordBy cut k (T, edges) x = some ((p, c, t), sp)) :
    ∃ fd, k < fd ∧ fd ≤ x.toList.length ∧ sp = prefixOf x (fd - 1)
      ∧ x.toList[fd - 1]? = some c ∧ T.siftCut cut sp = .inl p
      ∧ T.siftCut cut (prefixOf x fd) = .inl t := by
  obtain ⟨ps, fd, ho⟩ := probeBy_of_record cut T edges k x h
  have h' := h
  unfold recordBy at h'
  rw [ho] at h'
  simp only [] at h'
  rcases hc : x.toList[fd - 1]? with _ | c' <;> rw [hc] at h'
  · simp at h'
  rcases ht : T.siftCut cut (prefixOf x fd) with t' | b <;> rw [ht] at h'
  · simp only [Option.some.injEq, Prod.mk.injEq] at h'
    obtain ⟨⟨rfl, rfl, rfl⟩, rfl⟩ := h'
    have hlen : fd - 1 < x.toList.length := by
      rw [List.getElem?_eq_some_iff] at hc; exact hc.1
    have ho' : probeBy (fun z => (ofCut cut z).cut) T edges k x = .edge ps fd := by
      rw [cut_ofCut]; exact ho
    obtain ⟨hkf, hs1, -⟩ := probeBy_edge T edges k x (ofCut cut) ho'
    rw [cut_ofCut] at hs1
    exact ⟨fd, hkf, by omega, rfl, hc, hs1, ht⟩
  · simp at h'

variable (C : Level5Settings)

theorem truthInv_congr {n : ℕ} {s s' : Level5State α} (ht : s'.tree = s.tree)
    (he : s'.edges = s.edges) (hs : TruthInv G n s) : TruthInv G n s' := by
  obtain ⟨h1, h2, h3, h5⟩ := hs
  refine ⟨ht ▸ h1, ht ▸ he ▸ h2, fun p c t w h => ?_, ht ▸ h5⟩
  rw [he] at h
  rw [ht]
  exact h3 p c t w h

/-- A step reading nothing wrong reads its probe's prefixes followed by midfixes rightly. -/
theorem clean_of_step {s : Level5State α} {x : FreeMonoid α} (hx : ¬ WrongStep G rd C s x)
    {i : ℕ} (hi : i ≤ x.toList.length) {m : FreeMonoid α} (hm : m ∈ s.tree.midfixes) :
    ¬ G.WrongRead rd (prefixOf x i * m) :=
  fun hw => hx ⟨_, Finset.mem_union_left _ (mem_potMid hi hm), hw⟩

/-- A probe reading nothing wrong keeps the hypothesis true: an edge it learns is a true
record. -/
theorem truthInv_record {s : Level5State α} (hs : TruthInv G C.maxLeaves s) {x : FreeMonoid α}
    (hx : ¬ WrongStep G rd C s x) :
    TruthInv G C.maxLeaves (recordProbe C (fun z => (rd z).cut) s x) := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
  rcases recordProbe_spec C cut s x with h | ⟨u, c, p, t, hm, hp, hc, ht, hn, h⟩ <;> rw [h]
  · exact truthInv_congr G rfl rfl hs
  obtain ⟨hg, he, htr, hL⟩ := hs
  obtain ⟨j, hkj, hjx, rfl⟩ := probeBy_member cut hm
  have hlen : (prefixOf x j).toList.length = j := by
    simp [prefixOf]; omega
  rw [hlen] at hc
  rw [← prefixOf_succ hc] at ht
  have hj1 : j + 1 ≤ x.toList.length := hjx
  refine ⟨hg, setEdge_into (s := s.countProbe cut C.walkStart x) he
    (DTree.siftCut_mem_paths _ _ _ ht), fun q c' t' w' hq => ?_, hL⟩
  rw [setEdge_eq] at hq
  split_ifs at hq with hqc
  · simp only [Prod.mk.injEq] at hqc
    obtain ⟨rfl, rfl⟩ := hqc
    simp only [Option.some.injEq, Prod.mk.injEq] at hq
    obtain ⟨rfl, rfl⟩ := hq
    refine ⟨leafOf_of_sift G rd (fun m hm => clean_of_step G rd C hx (by omega) hm) hp, ?_⟩
    rw [← prefixOf_succ hc]
    exact leafOf_of_sift G rd (fun m hm => clean_of_step G rd C hx hj1 hm) ht
  · exact htr q c' t' w' hq

/-- A contradiction found reading nothing wrong splits genuinely, and the re-sifted edges stay
true. -/
theorem truthInv_split [Fintype σ] (hσ : Fintype.card σ + 2 ≤ C.maxLeaves) {s : Level5State α}
    (hs : TruthInv G C.maxLeaves s) {x : FreeMonoid α}
    (hx : ¬ WrongStep G rd C s x) {p : List Bool} {c : α} {t : List Bool}
    {sp : FreeMonoid α}
    (hrec : recordBy (fun z => (rd z).cut) C.walkStart (s.tree, s.edges) x = some ((p, c, t), sp)) :
    TruthInv G C.maxLeaves
      (splitState (fun z => (rd z).cut) (recordProbe C (fun z => (rd z).cut) s x) p c t) := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
  obtain ⟨hg, he, htr, hL⟩ := hs
  obtain ⟨-, -, -, -, -, -, t₀, w, hpe, hne⟩ := recordBy_spec cut hrec
  obtain ⟨fd, hkf, hfx, rfl, hcfd, hsp, ht⟩ := recordBy_parts cut hrec
  have hrec₁ : G.TrueRec s.tree (p, c, t) (prefixOf x (fd - 1)) := by
    refine ⟨leafOf_of_sift G rd (fun m hm => clean_of_step G rd C hx (by omega) hm) hsp, ?_⟩
    simp only []
    rw [← prefixOf_succ hcfd, show fd - 1 + 1 = fd by omega]
    exact leafOf_of_sift G rd (fun m hm => clean_of_step G rd C hx hfx hm) ht
  have hrec₀ := htr p c t₀ w hpe
  have hgen := G.genuineSplit_of_recs hrec₁ hrec₀ hne.symm
  have hp : p ∈ s.tree.leaves := by
    obtain ⟨q₁, -, hl, -⟩ := hgen
    exact hl ▸ G.leafOf_mem_paths _ _
  have htl : t ∈ s.tree.leaves := DTree.siftCut_mem_paths _ _ _ ht
  have ht₀l : t₀ ∈ s.tree.leaves := he _ _ _ _ hpe
  have hG' := ReadModel.Grown.real hg htl ht₀l hgen
  have hrp := recordProbe_of_record C cut s x hrec
  have hpe' : (recordProbe C cut s x).edges p c = some (t₀, w) := by rw [hrp]; exact hpe
  rw [splitState_of_some cut _ hpe', hrp]
  simp only [Level5State.countProbe]
  set d := FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)
  set T' := s.tree.splitAt d p
  have hL' : T'.leaves.length ≤ C.maxLeaves := by
    have := G.grown_paths hG'; omega
  have hR : ∀ z ∈ resifts s T', ¬ G.WrongRead rd z := by
    intro z hz hw
    refine hx ⟨z, Finset.mem_union_right _ ?_, hw⟩
    unfold splitReads
    rw [hrec]
    simp only [hpe]
    exact hz
  refine ⟨hG', ?_, ?_, hL'⟩
  · have := splitState_into cut he p c t
    rw [splitState_of_some cut _ hpe] at this
    exact this
  intro q e t' w' hq
  simp only [freshState] at hq ⊢
  split_ifs at hq with hpq
  unfold retargetBy at hq
  rcases hqe : s.edges q e with _ | ⟨s', w''⟩ <;> rw [hqe] at hq
  · simp at hq
  obtain ⟨hsrc, htgt⟩ := htr q e s' w'' hqe
  have hqp : q ≠ p := fun h => hpq (h ▸ List.prefix_refl q)
  have hsrc' : G.leafOf T' (G.M.eval w''.toList) = q := by
    rw [(G.leafOf_splitAt d s.tree p hp _).1 (by rw [hsrc]; exact hqp), hsrc]
  simp only [] at hq
  split_ifs at hq with hs'
  · rcases hsift : T'.siftCut cut (w'' * FreeMonoid.of e) with s'' | b <;> rw [hsift] at hq
    · simp only [Sum.elim_inl, Option.some.injEq, Prod.mk.injEq] at hq
      obtain ⟨rfl, rfl⟩ := hq
      refine ⟨hsrc', leafOf_of_sift G rd (fun m hm => hR _ ?_) hsift⟩
      have hql : q ∈ s.tree.leaves := edge_leaf G (fun a b c d h => htr a b c d h) hqe
      refine Finset.mem_biUnion.2 ⟨(q, e), by simp [hql], ?_⟩
      simp only [hqe]
      exact Finset.mem_image.2 ⟨m, hm, rfl⟩
    · simp at hq
  · simp only [Option.some.injEq, Prod.mk.injEq] at hq
    obtain ⟨rfl, rfl⟩ := hq
    refine ⟨hsrc', ?_⟩
    simp only []
    rw [(G.leafOf_splitAt d s.tree p hp _).1 (by rw [htgt]; exact hs'), htgt]

/-- A step reading nothing wrong keeps the hypothesis true, and does not end past `maxLeaves`
leaves. -/
theorem truthInv_step [Fintype σ] (hσ : Fintype.card σ + 2 ≤ C.maxLeaves) {s : Level5State α}
    (hs : TruthInv G C.maxLeaves s) {x : FreeMonoid α} (hx : ¬ WrongStep G rd C s x) :
    (∀ s', applyProbe C (fun z => (rd z).cut) s x = .inl s' → TruthInv G C.maxLeaves s')
      ∧ ∀ e s', applyProbe C (fun z => (rd z).cut) s x = .inr (e, s') →
        e ≠ .tooBig ∧ TruthInv G C.maxLeaves s' := by
  constructor
  · intro s' h
    rcases (applyProbe_inl C _ h).2 with ⟨-, rfl⟩ | ⟨p, c, t, sp, hrec, rfl, -⟩
    · exact truthInv_record G rd C hs hx
    · exact truthInv_split G rd C hσ hs hx hrec
  · intro e s' h
    rcases applyProbe_inr C _ h with ⟨hr, rfl⟩ | ⟨rfl, p, c, t, sp, hrec, rfl, hL⟩
    · refine ⟨fun he => runTests_ne_tooBig C _ (he ▸ hr), truthInv_record G rd C hs hx⟩
    · exact absurd (truthInv_split G rd C hσ hs hx hrec).2.2.2 (not_le.2 hL)

end Truth

end OrthoDFA
