import OrthoDFA.Proofs.Level5Steps

/-!
# Reads on the rarer side

A split is genuine unless a read on its state's rarer decided side made the contradiction's
record, or the edge it contradicts, untrue. Every string a probe of `x` reads, and every string a
later re-sift of one of `x`'s prefixes reads, is a prefix of `x` from `k` on followed by a word of
at most `maxLeaves` letters. Each such read is on the rarer side with chance at most `ε + κ`, and
the probes are drawn independently of the reads, so some probe of the round has such a string read
on the rarer side with chance at most the probes times `(L + 1)(maxLeaves + 1) |Σ|^maxLeaves (ε + κ)`
(`dirty_le`). On a round with none, every hypothesis is true: its tree grown by genuine splits
alone, so within `|Q| + 2` leaves, and its edges true records (`truthInv_step`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Words

/-- The words of at most `n` letters. -/
noncomputable def shortWords (n : ℕ) : Finset (FreeMonoid α) :=
  Finset.univ.image fun v : (Σ i : Fin (n + 1), Fin i → α) => FreeMonoid.ofList (List.ofFn v.2)

theorem mem_shortWords {n : ℕ} {m : FreeMonoid α} (h : m.toList.length ≤ n) :
    m ∈ shortWords n := by
  refine Finset.mem_image.2 ⟨⟨⟨m.toList.length, by omega⟩, fun j => m.toList[j]⟩,
    Finset.mem_univ _, ?_⟩
  apply FreeMonoid.toList.injective
  simp

theorem shortWords_card [Nonempty α] (n : ℕ) :
    (shortWords (α := α) n).card ≤ (n + 1) * Fintype.card α ^ n := by
  refine Finset.card_image_le.trans ?_
  rw [Finset.card_univ, Fintype.card_sigma]
  have hA : 1 ≤ Fintype.card α := Fintype.card_pos
  calc ∑ i : Fin (n + 1), Fintype.card (Fin i → α) ≤ ∑ _i : Fin (n + 1), Fintype.card α ^ n := by
        refine Finset.sum_le_sum fun i _ => ?_
        rw [Fintype.card_fun, Fintype.card_fin]
        exact Nat.pow_le_pow_right hA (by omega)
    _ = (n + 1) * Fintype.card α ^ n := by simp

end Words

namespace ReadModel

variable {σ : Type*} (G : ReadModel α σ)

/-- A read on its state's rarer decided side. -/
def WrongRead (rd : FreeMonoid α → ARU) (z : FreeMonoid α) : Prop :=
  (rd z).cut = some (!G.side (G.M.eval z.toList))

/-- Some prefix of `x` from `k` on, followed by a word of at most `n` letters, is read on its
state's rarer side. -/
def Dirty (rd : FreeMonoid α → ARU) (k n : ℕ) (x : FreeMonoid α) : Prop :=
  ∃ j, k ≤ j ∧ j ≤ x.toList.length ∧ ∃ m ∈ shortWords n, G.WrongRead rd (prefixOf x j * m)

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

theorem dirty_set_eq (k n : ℕ) (x : FreeMonoid α) :
    {ω | G.Dirty (read · ω) k n x} = ⋃ j ∈ (Finset.range (x.toList.length + 1)).filter (k ≤ ·),
      ⋃ m ∈ shortWords n, {ω | G.WrongRead (read · ω) (prefixOf x j * m)} := by
  ext ω
  simp only [ReadModel.Dirty, Set.mem_setOf_eq, Set.mem_iUnion, Finset.mem_filter,
    Finset.mem_range, exists_prop]
  constructor
  · rintro ⟨j, hkj, hjx, m, hm, hw⟩; exact ⟨j, ⟨by omega, hkj⟩, m, hm, hw⟩
  · rintro ⟨j, ⟨hjx, hkj⟩, m, hm, hw⟩; exact ⟨j, hkj, by omega, m, hm, hw⟩

theorem dirty_measurable (hmeas : ∀ w, Measurable (read w)) (k n : ℕ) (x : FreeMonoid α) :
    MeasurableSet {ω | G.Dirty (read · ω) k n x} := by
  rw [dirty_set_eq]
  refine Finset.measurableSet_biUnion _ fun j _ => Finset.measurableSet_biUnion _ fun m _ => ?_
  have : {ω | G.WrongRead (read · ω) (prefixOf x j * m)}
      = {ω | read (prefixOf x j * m) ω = G.minVal (G.M.eval (prefixOf x j * m).toList)} := by
    ext ω
    simp only [ReadModel.WrongRead, Set.mem_setOf_eq]
    exact cut_minVal G _ _
  rw [this]
  exact hmeas _ (measurableSet_singleton _)

theorem dirty_point_le (hmeas : ∀ w, Measurable (read w))
    (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.readDist (G.M.eval w.toList) r) (hκ : 0 ≤ G.κ)
    (hε : 0 ≤ G.ε) (k n : ℕ) (x : FreeMonoid α) :
    μ {ω | G.Dirty (read · ω) k n x}
      ≤ ENNReal.ofReal ((x.toList.length + 1) * (shortWords (α := α) n).card * (G.ε + G.κ)) := by
  rw [dirty_set_eq]
  refine (measure_biUnion_finset_le _ _).trans ?_
  refine (Finset.sum_le_sum fun j _ => (measure_biUnion_finset_le _ _).trans
    (Finset.sum_le_sum fun m _ => wrongRead_le G hmeas hlaw hκ hε _)).trans ?_
  simp only [Finset.sum_const, nsmul_eq_mul]
  have h1 : ((Finset.range (x.toList.length + 1)).filter (k ≤ ·)).card ≤ x.toList.length + 1 :=
    (Finset.card_filter_le _ _).trans (by simp)
  rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul (by positivity),
    ← ENNReal.ofReal_mul (by positivity)]
  refine ENNReal.ofReal_le_ofReal ?_
  have h1' : (((Finset.range (x.toList.length + 1)).filter (k ≤ ·)).card : ℝ)
      ≤ x.toList.length + 1 := by exact_mod_cast h1
  have := add_nonneg hε hκ
  have h0 : (0 : ℝ) ≤ (shortWords (α := α) n).card := Nat.cast_nonneg _
  nlinarith [mul_le_mul_of_nonneg_right h1' (mul_nonneg h0 this)]

/-- Some probe drawn from `D` reading a prefix from `k` on, followed by a word of at most `n`
letters, on the rarer side: at most `(L + 1)` prefixes times the words, each at `ε + κ`. -/
theorem dirty_le (hmeas : ∀ w, Measurable (read w))
    (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.readDist (G.M.eval w.toList) r) (hκ : 0 ≤ G.κ)
    (hε : 0 ≤ G.ε) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {L : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (k n : ℕ) :
    ∫⁻ ω, D {x | G.Dirty (read · ω) k n x} ∂μ
      ≤ ENNReal.ofReal ((L + 1) * (shortWords (α := α) n).card * (G.ε + G.κ)) := by
  classical
  have hD : ∀ ω, D {x | G.Dirty (read · ω) k n x}
      = ∫⁻ x, {x | G.Dirty (read · ω) k n x}.indicator 1 x ∂D := fun ω =>
    (lintegral_indicator_one (measurableSet_of_countable _)).symm
  simp_rw [hD]
  rw [lintegral_lintegral_swap]
  · calc ∫⁻ x, ∫⁻ ω, {x | G.Dirty (read · ω) k n x}.indicator 1 x ∂μ ∂D
        = ∫⁻ x, μ {ω | G.Dirty (read · ω) k n x} ∂D := by
          congr 1; funext x
          rw [← lintegral_indicator_one (dirty_measurable G hmeas k n x)]
          rfl
      _ ≤ ∫⁻ _x, ENNReal.ofReal ((L + 1) * (shortWords (α := α) n).card * (G.ε + G.κ)) ∂D := by
          refine lintegral_mono_ae ?_
          filter_upwards [hlen] with x hx
          refine (dirty_point_le G hmeas hlaw hκ hε k n x).trans (ENNReal.ofReal_le_ofReal ?_)
          have hxL : ((x.toList.length : ℝ) + 1) ≤ L + 1 := by exact_mod_cast (by omega : _ ≤ L + 1)
          have h0 : (0 : ℝ) ≤ (shortWords (α := α) n).card := Nat.cast_nonneg _
          nlinarith [mul_le_mul_of_nonneg_right hxL (mul_nonneg h0 (add_nonneg hε hκ))]
      _ = _ := by rw [lintegral_const, measure_univ, mul_one]
  · refine (measurable_from_prod_countable_left fun x => ?_).aemeasurable
    exact measurable_one.indicator (dirty_measurable G hmeas k n x)

end Chance

section Mids

/-- A node's midfix is at most the leaves less two letters long. -/
def ShortMids (T : DTree α) : Prop := ∀ m ∈ T.midfixes, m.toList.length + 2 ≤ T.leaves.length

theorem midfixAt_mem : ∀ (T : DTree α) (p : List Bool), T.midfixAt p = 1 ∨ T.midfixAt p ∈ T.midfixes
  | .leaf, _ => .inl rfl
  | .node m r a, [] => .inr (by simp [DTree.midfixAt, DTree.midfixes])
  | .node m r a, false :: p => by
    rcases midfixAt_mem r p with h | h
    · exact .inl h
    · exact .inr (by simp [DTree.midfixAt, DTree.midfixes, h])
  | .node m r a, true :: p => by
    rcases midfixAt_mem a p with h | h
    · exact .inl h
    · exact .inr (by simp [DTree.midfixAt, DTree.midfixes, h])

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

theorem shortMids_start : ShortMids (.node 1 .leaf .leaf : DTree α) := by
  intro m hm
  simp [DTree.midfixes] at hm
  subst hm
  simp [DTree.leaves]

theorem shortMids_split {T : DTree α} (hT : ShortMids T) (h2 : 2 ≤ T.leaves.length)
    {p : List Bool} (hp : p ∈ T.leaves) (c : α) (q : List Bool) :
    ShortMids (T.splitAt (FreeMonoid.of c * T.midfixAt q) p) := by
  intro m hm
  rw [DTree.length_leaves_splitAt _ _ _ hp]
  rcases Finset.mem_insert.1 (midfixes_splitAt _ T p hm) with rfl | hm
  · simp only [FreeMonoid.toList_mul, FreeMonoid.toList_of, List.length_append,
      List.length_singleton]
    rcases midfixAt_mem T q with h | h
    · rw [h]; simp; omega
    · have := hT _ h; omega
  · have := hT m hm; omega

theorem mem_shortWords_of_mids {T : DTree α} (hT : ShortMids T) {n : ℕ} (hn : T.leaves.length ≤ n)
    {m : FreeMonoid α} (hm : m ∈ T.midfixes) : m ∈ shortWords n :=
  mem_shortWords (by have := hT m hm; omega)

end Mids

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

/-- The reads of `z` followed by any word of at most `n` letters are none on the rarer side. -/
def CleanStr (n : ℕ) (z : FreeMonoid α) : Prop := ∀ m ∈ shortWords n, ¬ G.WrongRead rd (z * m)

/-- Every learned edge is a true record, its witness's successor clean. -/
def TrueEdges (n : ℕ) (s : Level5State α) : Prop :=
  ∀ p c t w, s.edges p c = some (t, w) → G.TrueRec s.tree (p, c, t) w
    ∧ CleanStr G rd n (w * FreeMonoid.of c)

/-- A state of a round with no read on the rarer side: its tree grown by genuine splits alone,
its edges into its leaves and true, its midfixes short, its leaves at most `n`. -/
def TruthInv (n : ℕ) (s : Level5State α) : Prop :=
  G.Grown s.tree 0 ∧ EdgesInto s.tree s.edges ∧ TrueEdges G rd n s ∧ ShortMids s.tree
    ∧ s.tree.leaves.length ≤ n

theorem truthInv_start {n : ℕ} (hn : 2 ≤ n) : TruthInv G rd n (initialState : Level5State α) :=
  ⟨.start, fun _ _ _ _ h => by simp [initialState, freshState] at h,
    fun _ _ _ _ h => by simp [initialState, freshState] at h, shortMids_start,
    by simp [initialState, freshState, DTree.leaves]; omega⟩

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
        · simp only [Sum.inl.injEq, Outcome.member.injEq] at hw
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

/-- A probe with no read on the rarer side reads every prefix from `k` on cleanly. -/
theorem clean_of_not_dirty {n : ℕ} {x : FreeMonoid α} (hx : ¬ G.Dirty rd C.walkStart n x) {j : ℕ}
    (hkj : C.walkStart ≤ j) (hjx : j ≤ x.toList.length) : CleanStr G rd n (prefixOf x j) :=
  fun m hm hw => hx ⟨j, hkj, hjx, m, hm, hw⟩

theorem truthInv_congr {n : ℕ} {s s' : Level5State α} (ht : s'.tree = s.tree)
    (he : s'.edges = s.edges) (hs : TruthInv G rd n s) : TruthInv G rd n s' := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := hs
  refine ⟨ht ▸ h1, ht ▸ he ▸ h2, fun p c t w h => ?_, ht ▸ h4, ht ▸ h5⟩
  rw [he] at h
  rw [ht]
  exact h3 p c t w h

/-- A probe with no read on the rarer side keeps the hypothesis true: an edge it learns is a true
record. -/
theorem truthInv_record {s : Level5State α} (hs : TruthInv G rd C.maxLeaves s) {x : FreeMonoid α}
    (hx : ¬ G.Dirty rd C.walkStart C.maxLeaves x) :
    TruthInv G rd C.maxLeaves (recordProbe C (fun z => (rd z).cut) s x) := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
  rcases recordProbe_spec C cut s x with h | ⟨u, c, p, t, hm, hp, hc, ht, hn, h⟩ <;> rw [h]
  · exact truthInv_congr G rd rfl rfl hs
  obtain ⟨hg, he, htr, hsm, hL⟩ := hs
  obtain ⟨j, hkj, hjx, rfl⟩ := probeBy_member cut hm
  have hlen : (prefixOf x j).toList.length = j := by
    simp [prefixOf]; omega
  rw [hlen] at hc
  rw [← prefixOf_succ hc] at ht
  have hcl : ∀ i, C.walkStart ≤ i → i ≤ x.toList.length → ∀ m ∈ s.tree.midfixes,
      ¬ G.WrongRead rd (prefixOf x i * m) := fun i h1 h2 m hm =>
    clean_of_not_dirty G rd C hx h1 h2 m (mem_shortWords_of_mids hsm hL hm)
  have hj1 : j + 1 ≤ x.toList.length := hjx
  refine ⟨hg, setEdge_into (s := s.countProbe cut C.walkStart x) he
    (DTree.siftCut_mem_paths _ _ _ ht), fun q c' t' w' hq => ?_, hsm, hL⟩
  rw [setEdge_eq] at hq
  split_ifs at hq with hqc
  · simp only [Prod.mk.injEq] at hqc
    obtain ⟨rfl, rfl⟩ := hqc
    simp only [Option.some.injEq, Prod.mk.injEq] at hq
    obtain ⟨rfl, rfl⟩ := hq
    refine ⟨⟨leafOf_of_sift G rd (hcl j hkj (by omega)) hp, ?_⟩, ?_⟩
    · rw [← prefixOf_succ hc]
      exact leafOf_of_sift G rd (hcl (j + 1) (by omega) hj1) ht
    · rw [← prefixOf_succ hc]
      exact clean_of_not_dirty G rd C hx (by omega) hj1
  · exact htr q c' t' w' hq

/-- A contradiction found with no read on the rarer side splits genuinely, and the re-sifted
edges stay true. -/
theorem truthInv_split [Fintype σ] (hσ : Fintype.card σ + 2 ≤ C.maxLeaves) {s : Level5State α}
    (hs : TruthInv G rd C.maxLeaves s) {x : FreeMonoid α}
    (hx : ¬ G.Dirty rd C.walkStart C.maxLeaves x) {p : List Bool} {c : α} {t : List Bool}
    {sp : FreeMonoid α}
    (hrec : recordBy (fun z => (rd z).cut) C.walkStart (s.tree, s.edges) x = some ((p, c, t), sp)) :
    TruthInv G rd C.maxLeaves
      (splitState (fun z => (rd z).cut) (recordProbe C (fun z => (rd z).cut) s x) p c t) := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
  obtain ⟨hg, he, htr, hsm, hL⟩ := hs
  obtain ⟨-, -, -, -, -, -, t₀, w, hpe, hne⟩ := recordBy_spec cut hrec
  obtain ⟨fd, hkf, hfx, rfl, hcfd, hsp, ht⟩ := recordBy_parts cut hrec
  have hcl : ∀ i, C.walkStart ≤ i → i ≤ x.toList.length → ∀ m ∈ s.tree.midfixes,
      ¬ G.WrongRead rd (prefixOf x i * m) := fun i h1 h2 m hm =>
    clean_of_not_dirty G rd C hx h1 h2 m (mem_shortWords_of_mids hsm hL hm)
  have hrec₁ : G.TrueRec s.tree (p, c, t) (prefixOf x (fd - 1)) := by
    refine ⟨leafOf_of_sift G rd (hcl (fd - 1) (by omega) (by omega)) hsp, ?_⟩
    simp only []
    rw [← prefixOf_succ hcfd, show fd - 1 + 1 = fd by omega]
    exact leafOf_of_sift G rd (hcl fd (by omega) hfx) ht
  obtain ⟨hrec₀, hcw⟩ := htr p c t₀ w hpe
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
  have hsm' : ShortMids T' := shortMids_split hsm (grown_two G hg) hp c _
  refine ⟨hG', ?_, ?_, hsm', hL'⟩
  · have := splitState_into cut he p c t
    rw [splitState_of_some cut _ hpe] at this
    exact this
  intro q e t' w' hq
  simp only [freshState] at hq ⊢
  split_ifs at hq with hpq
  unfold retargetBy at hq
  rcases hqe : s.edges q e with _ | ⟨s', w''⟩ <;> rw [hqe] at hq
  · simp at hq
  obtain ⟨⟨hsrc, htgt⟩, hcl'⟩ := htr q e s' w'' hqe
  have hqp : q ≠ p := fun h => hpq (h ▸ List.prefix_refl q)
  have hsrc' : G.leafOf T' (G.M.eval w''.toList) = q := by
    rw [(G.leafOf_splitAt d s.tree p hp _).1 (by rw [hsrc]; exact hqp), hsrc]
  simp only [] at hq
  split_ifs at hq with hs'
  · rcases hsift : T'.siftCut cut (w'' * FreeMonoid.of e) with s'' | b <;> rw [hsift] at hq
    · simp only [Sum.elim_inl, Option.some.injEq, Prod.mk.injEq] at hq
      obtain ⟨rfl, rfl⟩ := hq
      refine ⟨⟨hsrc', leafOf_of_sift G rd (fun m hm => hcl' m
        (mem_shortWords_of_mids hsm' hL' hm)) hsift⟩, hcl'⟩
    · simp at hq
  · simp only [Option.some.injEq, Prod.mk.injEq] at hq
    obtain ⟨rfl, rfl⟩ := hq
    refine ⟨⟨hsrc', ?_⟩, hcl'⟩
    simp only []
    rw [(G.leafOf_splitAt d s.tree p hp _).1 (by rw [htgt]; exact hs'), htgt]

/-- A step with no read on the rarer side keeps the hypothesis true, and does not end past
`maxLeaves` leaves. -/
theorem truthInv_step [Fintype σ] (hσ : Fintype.card σ + 2 ≤ C.maxLeaves) {s : Level5State α}
    (hs : TruthInv G rd C.maxLeaves s) {x : FreeMonoid α}
    (hx : ¬ G.Dirty rd C.walkStart C.maxLeaves x) :
    (∀ s', applyProbe C (fun z => (rd z).cut) s x = .inl s' → TruthInv G rd C.maxLeaves s')
      ∧ ∀ e s', applyProbe C (fun z => (rd z).cut) s x = .inr (e, s') →
        e ≠ .tooBig ∧ TruthInv G rd C.maxLeaves s' := by
  constructor
  · intro s' h
    rcases (applyProbe_inl C _ h).2 with ⟨-, rfl⟩ | ⟨p, c, t, sp, hrec, rfl, -⟩
    · exact truthInv_record G rd C hs hx
    · exact truthInv_split G rd C hσ hs hx hrec
  · intro e s' h
    rcases applyProbe_inr C _ h with ⟨hr, rfl⟩ | ⟨rfl, p, c, t, sp, hrec, rfl, hL⟩
    · refine ⟨fun he => runTests_ne_tooBig C _ (he ▸ hr), truthInv_record G rd C hs hx⟩
    · exact absurd (truthInv_split G rd C hσ hs hx hrec).2.2.2.2 (not_le.2 hL)

end Truth

end OrthoDFA
