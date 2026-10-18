import OrthoDFA.Proofs.Level5Reads

/-!
# Level 5 reads only through its probes' first `k` letters

A probe reads the cut only at strings that begin with its own first `k` letters and are at least
`k` long, and an edge's witness followed by its letter is such a prefix of the probe that made it.
So over probes whose first `k` letters avoid a set `K₀`, two cuts agreeing on every string at
least `k` long whose first `k` letters avoid `K₀` give the same run (`level5_congr_off`), and the
same search and harvest at every such probe.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- `z` is at least `k` long and its first `k` letters are not in `K₀`. -/
def KeyOff (k : ℕ) (K₀ : Set (FreeMonoid α)) (z : FreeMonoid α) : Prop :=
  k ≤ z.toList.length ∧ prefixOf z k ∉ K₀

/-- A probe level 5 may draw: shorter than `k`, or its first `k` letters off `K₀`. -/
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

theorem sift_congr_off (h : AgreeOff k K₀ cut cut') (T : DTree α) {u : FreeMonoid α}
    (hu : KeyOff k K₀ u) : T.sift (ofCut cut) u = T.sift (ofCut cut') u :=
  sift_congr_mid T u fun m _ => h _ (keyOff_mul hu m)

theorem probeStrings_agree_off (h : AgreeOff k K₀ cut cut') (T : DTree α) {x : FreeMonoid α}
    (hx : KeyOff k K₀ x) : ∀ z ∈ probeStrings T k x, cut z = cut' z := by
  intro z hz
  obtain ⟨j, hj, hz⟩ := Finset.mem_biUnion.1 hz
  obtain ⟨m, -, rfl⟩ := Finset.mem_image.1 hz
  exact h _ (keyOff_mul (keyOff_prefixOf hx (Finset.mem_Icc.1 hj).1) m)

theorem prefAgreeK_off (h : AgreeOff k K₀ cut cut') (T : DTree α) {x : FreeMonoid α}
    (hx : KeyOff k K₀ x) : PrefAgreeK cut cut' T x k :=
  fun _ hj => sift_congr_off h T (keyOff_prefixOf hx hj)

theorem probeOrAgree_congr_off (h : AgreeOff k K₀ cut cut') (T : DTree α) (E : Edges α)
    {x : FreeMonoid α} (hx : Avoids k K₀ x) :
    probeOrAgree (ofCut cut) T E k x = probeOrAgree (ofCut cut') T E k x := by
  by_cases hk : k ≤ x.toList.length
  · exact probeOrAgree_congrK (prefAgreeK_off h T (keyOff_of_avoids hx hk))
  · rw [probeOrAgree_short cut T E (by omega), probeOrAgree_short cut' T E (by omega)]

theorem middleHarvestBy_congr_off (h : AgreeOff k K₀ cut cut') (T : DTree α) (E : Edges α)
    {x : FreeMonoid α} (hx : Avoids k K₀ x) :
    middleHarvestBy cut T E k x = middleHarvestBy cut' T E k x := by
  by_cases hk : k ≤ x.toList.length
  · exact middleHarvestBy_congrK (prefAgreeK_off h T (keyOff_of_avoids hx hk))
  · rw [middleHarvestBy_short cut T E (by omega), middleHarvestBy_short cut' T E (by omega)]

theorem startHarvestBy_congr_strings {T : DTree α} {x : FreeMonoid α}
    (hkx : k ≤ x.toList.length) (h : ∀ z ∈ probeStrings T k x, cut z = cut' z) :
    startHarvestBy cut T k x = startHarvestBy cut' T k x := by
  unfold startHarvestBy
  rw [sift_congr_mid T _ fun m hm => h _ (Finset.mem_biUnion.2 ⟨k,
    Finset.mem_Icc.2 ⟨le_rfl, hkx⟩, Finset.mem_image_of_mem _ hm⟩)]

theorem startHarvestBy_congr_off (h : AgreeOff k K₀ cut cut') (T : DTree α)
    {x : FreeMonoid α} (hx : Avoids k K₀ x) : startHarvestBy cut T k x = startHarvestBy cut' T k x := by
  by_cases hk : k ≤ x.toList.length
  · exact startHarvestBy_congr_strings hk (probeStrings_agree_off h T (keyOff_of_avoids hx hk))
  · simp only [startHarvestBy, if_pos (show x.toList.length < k by omega)]

theorem edgeReadsBy_congr_off (h : AgreeOff k K₀ cut cut') (T : DTree α) (E : Edges α)
    {x : FreeMonoid α} (hx : Avoids k K₀ x) :
    edgeReadsBy cut T E k x = edgeReadsBy cut' T E k x := by
  by_cases hk : k ≤ x.toList.length
  · exact edgeReadsBy_congr_strings hk (probeStrings_agree_off h T (keyOff_of_avoids hx hk))
  · funext e
    rw [edgeReadsBy_short cut T E (by omega) e, edgeReadsBy_short cut' T E (by omega) e]

theorem edgeHarvestBy_congr_off (h : AgreeOff k K₀ cut cut') (T : DTree α) (E : Edges α)
    {x : FreeMonoid α} (hx : Avoids k K₀ x) :
    edgeHarvestBy cut T E k x = edgeHarvestBy cut' T E k x := by
  by_cases hk : k ≤ x.toList.length
  · exact edgeHarvestBy_congr_strings hk (probeStrings_agree_off h T (keyOff_of_avoids hx hk))
  · funext e
    rw [edgeHarvestBy_short cut T E (by omega) e, edgeHarvestBy_short cut' T E (by omega) e]

theorem siteHarvest_congr_off (h : AgreeOff k K₀ cut cut') (site : Level5Site α)
    (v : DTree α × Edges α) {x : FreeMonoid α} (hx : Avoids k K₀ x) :
    siteHarvest cut k site v x = siteHarvest cut' k site v x := by
  cases site with
  | start => exact startHarvestBy_congr_off h v.1 hx
  | edge e => exact congrFun (edgeHarvestBy_congr_off h v.1 v.2 hx) e
  | middles => exact middleHarvestBy_congr_off h v.1 v.2 hx

theorem countProbe_congr_off (h : AgreeOff k K₀ cut cut') (s : Level5State α)
    {x : FreeMonoid α} (hx : Avoids k K₀ x) :
    s.countProbe cut k x = s.countProbe cut' k x := by
  unfold Level5State.countProbe
  rw [middleHarvestBy_congr_off h s.tree s.edges hx, startHarvestBy_congr_off h s.tree hx,
    edgeReadsBy_congr_off h s.tree s.edges hx, edgeHarvestBy_congr_off h s.tree s.edges hx]

theorem recordBy_congr_off (h : AgreeOff k K₀ cut cut') (T : DTree α) (E : Edges α)
    {x : FreeMonoid α} (hx : Avoids k K₀ x) :
    recordBy cut k (T, E) x = recordBy cut' k (T, E) x := by
  unfold recordBy
  rw [probeOrAgree_congr_off h T E hx]

/-- What the run keeps that it reads through: every edge's witness followed by its letter is at
least `k` long and off `K₀`. -/
def OffInv (k : ℕ) (K₀ : Set (FreeMonoid α)) (s : Level5State α) : Prop :=
  ∀ q c t w, s.edges (q, c) = some (t, w) → KeyOff k K₀ (w * FreeMonoid.of c)

variable (C : Level5Settings)

theorem recordProbe_congr_off (h : AgreeOff C.walkStart K₀ cut cut') (s : Level5State α)
    {x : FreeMonoid α} (hx : Avoids C.walkStart K₀ x) :
    recordProbe C cut s x = recordProbe C cut' s x := by
  unfold recordProbe
  rw [probeOrAgree_congr_off h s.tree s.edges hx, countProbe_congr_off h s hx]

theorem retarget_congr_off (h : AgreeOff k K₀ cut cut') (t' : DTree α) (p : List Bool)
    {E : Edges α}
    (hE : ∀ q c t w, E (q, c) = some (t, w) → KeyOff k K₀ (w * FreeMonoid.of c)) :
    retarget (ofCut cut) t' p E = retarget (ofCut cut') t' p E := by
  funext ⟨q, c⟩
  unfold retarget
  split_ifs
  · rfl
  rcases he : E (q, c) with _ | ⟨t, w⟩
  · rfl
  · dsimp only
    split_ifs
    · rw [sift_congr_off h t' (hE q c t w he)]
    · rfl

theorem splitState_congr_off (h : AgreeOff k K₀ cut cut') (s : Level5State α)
    (hs : OffInv k K₀ s) (p : List Bool) (c : α) (t : List Bool) :
    splitState cut s p c t = splitState cut' s p c t := by
  unfold splitState
  rcases he : s.edges (p, c) with _ | ⟨t₀, w₀⟩
  · rfl
  · simp only [HypothesisDFA.split]
    rw [retarget_congr_off h _ p hs]

theorem recordProbe_offInv {s : Level5State α} (hs : OffInv C.walkStart K₀ s)
    {x : FreeMonoid α} (hx : Avoids C.walkStart K₀ x) :
    OffInv C.walkStart K₀ (recordProbe C cut s x) := by
  rcases recordProbe_spec C cut s x with h | ⟨u, c', p, t', ho, -, hc, -, -, h⟩ <;> rw [h]
  · exact hs
  intro q c t w hqc
  rw [setEdge_eq] at hqc
  split_ifs at hqc with hpq
  · simp only [Prod.mk.injEq] at hpq
    obtain ⟨rfl, rfl⟩ := hpq
    simp only [Option.some.injEq, Prod.mk.injEq] at hqc
    rw [← hqc.2]
    obtain ⟨-, j, -, hj, hjx, rfl, -⟩ := probe_unlearnedEdge _ _ _ _ _ ho
    have hjl : (prefixOf x j).toList.length = j := by rw [length_prefixOf]; omega
    rw [hjl] at hc
    rw [prefixOf_mul_of hc]
    exact keyOff_prefixOf (keyOff_of_avoids hx (by omega)) (by omega)
  · exact hs q c t w hqc

theorem applyProbe_offInv {s s' : Level5State α} {x : FreeMonoid α}
    (h : OffInv C.walkStart K₀ (recordProbe C cut s x)) (ha : applyProbe C cut s x = .inl s') :
    OffInv C.walkStart K₀ s' := by
  rcases (applyProbe_inl C cut ha).2 with ⟨-, rfl⟩ | ⟨p, c, t, sp, -, rfl, -⟩
  · exact h
  · intro q c' t' w hqc
    obtain ⟨t₁, h₁⟩ := splitState_witness cut hqc
    exact h q c' t₁ w h₁

theorem applyProbe_congr_off (h : AgreeOff C.walkStart K₀ cut cut') {s : Level5State α}
    (hs : OffInv C.walkStart K₀ s) {x : FreeMonoid α} (hx : Avoids C.walkStart K₀ x) :
    applyProbe C cut s x = applyProbe C cut' s x := by
  have hs' := recordProbe_offInv C (cut := cut') hs hx
  have hrec := recordProbe_congr_off C h s hx
  have hrb := recordBy_congr_off h s.tree s.edges hx
  unfold applyProbe
  rw [hrec, hrb]
  dsimp only
  split
  · rfl
  · split
    · rw [splitState_congr_off h _ hs']
    · rfl

theorem runProbes_congr_off (h : AgreeOff C.walkStart K₀ cut cut') :
    ∀ (s : Level5State α) (xs : List (FreeMonoid α)), OffInv C.walkStart K₀ s →
      (∀ x ∈ xs, Avoids C.walkStart K₀ x) → runProbes C cut s xs = runProbes C cut' s xs
  | s, [], _, _ => rfl
  | s, x :: xs, hs, hxs => by
    have hx := hxs x List.mem_cons_self
    simp only [runProbes]
    rw [applyProbe_congr_off C h hs hx]
    rcases ha : applyProbe C cut' s x with s' | ⟨e, s'⟩
    · have hinv : OffInv C.walkStart K₀ s' :=
        applyProbe_offInv C (recordProbe_offInv C hs hx) ha
      exact runProbes_congr_off h s' xs hinv (fun y hy => hxs y (List.mem_cons_of_mem _ hy))
    · rfl

end Congr

/-- Level 5 over probes avoiding `K₀` reads only strings `KeyOff` holds of. -/
theorem level5_congr_off (C : Level5Settings) {T : ℕ}
    (draws : Fin T → FreeMonoid α) (hd : ∀ i, Avoids C.walkStart K₀ (draws i))
    {read read' : FreeMonoid α → ARU}
    (h : ∀ z, KeyOff C.walkStart K₀ z → read z = read' z) :
    indecisiveLStarLevel5 C read draws = indecisiveLStarLevel5 C read' draws := by
  unfold indecisiveLStarLevel5
  rw [runProbes_congr_off C (cut := fun z => (read z).cut) (cut' := fun z => (read' z).cut)
    (fun z hz => by simp only [h z hz]) initialState (List.ofFn draws)
    (fun _ _ _ _ h => by simp [initialState, freshState] at h)
    (fun x hx => by obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx; exact hd i)]

end OrthoDFA
