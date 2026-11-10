import RLStar.Proofs.Trie
import RLStar.Proofs.RandomProbe

/-!
# Local walk starts: the cap does not bind

A skipped probe and the probe that first read its string both read it as a prefix of their own
followed by a midfix of the tree. So a walk start of length at least `K` among them makes one
draw's first `K` letters those of another draw's prefix followed by a midfix of the tree the
first was probed under (`TrieCollides`). That set is fixed by the other draws and has at most
`P(K+1)P` elements.
-/

noncomputable section

namespace OrthoDFA

namespace Random

open MeasureTheory Query

theorem pi_hit_le {β : Type*} [MeasurableSpace β] [Countable β] [MeasurableSingletonClass β]
    (D : Measure β) [IsProbabilityMeasure D] {N : ℕ} (a : Fin N)
    (F : (Fin N → β) → Set β) (hF : ∀ d e : Fin N → β, (∀ l, l ≠ a → d l = e l) → F d = F e)
    (q : ENNReal) (hq : ∀ d, D (F d) ≤ q) :
    (Measure.pi fun _ : Fin N => D) {d | d a ∈ F d} ≤ q := by
  obtain ⟨n, rfl⟩ : ∃ n, N = n + 1 := ⟨N - 1, by have := a.pos; omega⟩
  have hmp := measurePreserving_piFinSuccAbove (fun _ : Fin (n + 1) => D) a
  rw [← hmp.symm.measure_preimage_equiv, Measure.prod_apply_symm (Set.to_countable _).measurableSet]
  calc _ ≤ ∫⁻ _, q ∂(Measure.pi fun _ : Fin n => D) := lintegral_mono fun v => ?_
    _ = q := by simp
  set G := (fun x => (x, v)) ⁻¹' ((MeasurableEquiv.piFinSuccAbove (fun _ => β) a).symm ⁻¹'
    {d | d a ∈ F d})
  rcases G.eq_empty_or_nonempty with h | ⟨x₀, -⟩
  · rw [h, measure_empty]; exact bot_le
  refine (measure_mono fun x hx => ?_).trans (hq (Fin.insertNth (α := fun _ => β) a x₀ v))
  have hx' : Fin.insertNth (α := fun _ => β) a x v a ∈ F (Fin.insertNth (α := fun _ => β) a x v) :=
    hx
  rw [Fin.insertNth_apply_same] at hx'
  rwa [hF (Fin.insertNth (α := fun _ => β) a x v) (Fin.insertNth (α := fun _ => β) a x₀ v)
    fun l hl => by
    obtain ⟨j, rfl⟩ := Fin.exists_succAbove_eq hl
    simp only [Fin.insertNth_apply_succAbove]] at hx'

variable {α : Type*}

theorem prefixOf_mul_of_le {w : FreeMonoid α} (m : FreeMonoid α) {p K : ℕ} (hK : K ≤ p)
    (hp : p ≤ w.toList.length) : prefixOf (prefixOf w p * m) K = prefixOf w K := by
  simp only [prefixOf, FreeMonoid.toList_mul, FreeMonoid.toList_ofList]
  rw [List.take_append_of_le_length (by simp; omega), List.take_take, min_eq_left hK]

theorem prefixOf_mul_min (w m : FreeMonoid α) {p : ℕ} (K : ℕ) (hp : p ≤ w.toList.length) :
    prefixOf (prefixOf w p * m) K = prefixOf (prefixOf w (min p K) * m) K := by
  rcases le_total p K with h | h
  · rw [min_eq_left h]
  · rw [min_eq_right h, prefixOf_mul_of_le m h hp, prefixOf_mul_of_le m le_rfl (h.trans hp)]

theorem length_prefixOf {x : FreeMonoid α} {k : ℕ} (hk : k ≤ x.toList.length) :
    (prefixOf x k).toList.length = k := by
  simp [prefixOf, hk]

theorem mem_pot_iff {k : ℕ} {x : FreeMonoid α} {T : DTree α} {z : FreeMonoid α} :
    z ∈ pot k x T → ∃ p, k ≤ p ∧ p ≤ x.toList.length ∧ ∃ m ∈ mids T, z = prefixOf x p * m := by
  classical
  intro h
  unfold pot at h
  obtain ⟨p, hp, hz⟩ := Finset.mem_biUnion.1 h
  obtain ⟨m, hm, rfl⟩ := Finset.mem_image.1 hz
  exact ⟨p, (Finset.mem_Icc.1 hp).1, (Finset.mem_Icc.1 hp).2, m, hm, rfl⟩

variable [Fintype α] [DecidableEq α] (C : Level3Settings) (y : ℝ) (read : FreeMonoid α → ARU)

omit [Fintype α] [DecidableEq α] in
theorem harvest_pot {T : DTree α} {E : Edges α} {k : ℕ} {x : FreeMonoid α} :
    ∀ z ∈ harvestString (probeOrAgree read T E k x), z ∈ pot k x T := by
  intro z hz
  rcases List.mem_append.1 hz with h | h
  · exact (probeOrAgree_ends read T x E k z h).1
  · exact (probeOrAgree_pt read T x E k z h).1

theorem mids_card_applyTrieProbe (cap : ℕ) {s s' : TrieState α} {x : FreeMonoid α}
    (h : (applyTrieProbeQ C y cap s x).eval read = .inl s') :
    (mids s'.tree).card ≤ (mids s.tree).card + 1 := by
  classical
  unfold applyTrieProbeQ at h
  rw [eval_bind] at h
  generalize (probeOrAgreeQ s.tree s.edges (walkStartIn s.expanded x) x).eval read = o at h
  by_cases hr : ∃ z ∈ harvestString o, z ∈ s.seen.map Prod.fst
  · rw [if_pos hr, eval_pure, Sum.inl.injEq] at h
    subst h; simp [TrieState.skipRepeat]
  rw [if_neg hr] at h
  have hc : ∀ o', applyTrieTests C y (countTrieOutcome s o' (prefixOf x (walkStartIn s.expanded x)))
      = .inl s' → (mids s'.tree).card ≤ (mids s.tree).card + 1 := by
    intro o' h'
    rw [applyTrieTests_inl C y h']
    simp [countTrieOutcome]
  cases o with
  | unlearnedEdge p c u t =>
    simp only [eval_pure, Sum.inl.injEq] at h; subst h
    simp [newTrieHypothesis]
  | wrongEdge p c u t =>
    simp only at h
    split at h
    · rw [eval_map, splitTrieLeafQ, eval_map, Sum.inl.injEq] at h
      subst h
      change (mids (HypothesisDFA.split read ⟨s.tree, s.edges⟩ _ p).tree).card ≤ _
      rw [HypothesisDFA.split_tree]
      exact (Finset.card_le_card (mids_splitAt _ _ _)).trans (Finset.card_insert_le _ _)
    · rw [eval_pure] at h; exact hc _ h
  | _ => rw [eval_pure] at h; exact hc _ h

theorem mids_card_runTrieProbes (cap : ℕ) : ∀ (s : TrieState α) (xs : List (FreeMonoid α)),
    (mids ((runTrieProbesQ C y cap s xs).eval read).1.tree).card ≤ (mids s.tree).card + xs.length
  | s, [] => by simp [runTrieProbesQ]
  | s, x :: xs => by
    unfold runTrieProbesQ
    rw [eval_bind]
    rcases ho : (applyTrieProbeQ C y cap s x).eval read with s' | e
    · have := mids_card_runTrieProbes cap s' xs
      have := mids_card_applyTrieProbe C y read cap ho
      simp only [List.length_cons]
      exact (by assumption : (mids ((runTrieProbesQ C y cap s' xs).eval read).1.tree).card ≤ _).trans
        (by omega)
    · simp

/-- Some draw's first `K` letters are those of another draw's prefix of length at most `K`
followed by a midfix of the tree the first was probed under. -/
def TrieCollides (cap K : ℕ) (xs : List (FreeMonoid α)) : Prop :=
  ∃ pre x post, xs = pre ++ x :: post ∧ ∃ w ∈ pre ++ post, ∃ p ≤ K,
    ∃ m ∈ mids ((runTrieProbesQ C y cap trieInitialState pre).eval read).1.tree,
      prefixOf x K = prefixOf (prefixOf w p * m) K

/-- Expanded prefixes are shorter than `K`, and each string seen was read by an earlier draw of
`pre`, under the current tree, as a prefix past the walk start it records followed by a midfix. -/
def CapInv (cap K : ℕ) (pre : List (FreeMonoid α)) (s : TrieState α) : Prop :=
  (∀ u ∈ s.expanded, u.toList.length < K) ∧
  ∀ q ∈ s.seen, ∃ pre₁ x' mid, pre = pre₁ ++ x' :: mid ∧
    ((runTrieProbesQ C y cap trieInitialState pre₁).eval read).1.tree = s.tree ∧
    ∃ k', k' ≤ x'.toList.length ∧ q.2 = prefixOf x' k' ∧ q.1 ∈ pot k' x' s.tree

theorem collides_of_seen {cap K : ℕ} {pre : List (FreeMonoid α)} {s : TrieState α}
    (hinv : CapInv C y read cap K pre s)
    (htree : ((runTrieProbesQ C y cap trieInitialState pre).eval read).1.tree = s.tree)
    {x : FreeMonoid α} {k : ℕ} (rest : List (FreeMonoid α)) {q : FreeMonoid α × FreeMonoid α}
    (hq : q ∈ s.seen) (hz : q.1 ∈ pot k x s.tree) :
    (K ≤ k → TrieCollides C y read cap K (pre ++ x :: rest))
      ∧ (K ≤ q.2.toList.length → TrieCollides C y read cap K (pre ++ x :: rest)) := by
  obtain ⟨pre₁, x', mid, hpre, htree₁, k', hk', hq2, hq1⟩ := hinv.2 q hq
  obtain ⟨p, hkp, hpx, m, hm, hzx⟩ := mem_pot_iff hz
  obtain ⟨p', hkp', hpx', m', hm', hzx'⟩ := mem_pot_iff hq1
  constructor
  · intro hK
    refine ⟨pre, x, rest, rfl, x', by rw [hpre]; simp, min p' K, min_le_right _ _, m',
      by rw [htree]; exact hm', ?_⟩
    rw [← prefixOf_mul_min _ _ _ hpx', ← hzx', hzx, prefixOf_mul_of_le m (hK.trans hkp) hpx]
  · intro hK
    rw [hq2, length_prefixOf hk'] at hK
    refine ⟨pre₁, x', mid ++ x :: rest, by rw [hpre]; simp, x, by simp, min p K,
      min_le_right _ _, m, by rw [htree₁]; exact hm, ?_⟩
    rw [← prefixOf_mul_min _ _ _ hpx, ← hzx, hzx', prefixOf_mul_of_le m' (hK.trans hkp') hpx']

theorem capInv_extend {cap K : ℕ} {pre : List (FreeMonoid α)} {s : TrieState α}
    (hinv : CapInv C y read cap K pre s) (x : FreeMonoid α) (q : FreeMonoid α × FreeMonoid α)
    (hq : q ∈ s.seen) : ∃ pre₁ x' mid, pre ++ [x] = pre₁ ++ x' :: mid ∧
      ((runTrieProbesQ C y cap trieInitialState pre₁).eval read).1.tree = s.tree ∧
      ∃ k', k' ≤ x'.toList.length ∧ q.2 = prefixOf x' k' ∧ q.1 ∈ pot k' x' s.tree := by
  obtain ⟨pre₁, x', mid, hpre, h⟩ := hinv.2 q hq
  exact ⟨pre₁, x', mid ++ [x], by rw [hpre]; simp, h⟩

theorem capStep {cap cap' K : ℕ} (hK : K ≤ cap) (hK' : K ≤ cap') {pre : List (FreeMonoid α)}
    {s : TrieState α} (hinv : CapInv C y read cap K pre s)
    (htree : ((runTrieProbesQ C y cap trieInitialState pre).eval read).1.tree = s.tree)
    (x : FreeMonoid α) (rest : List (FreeMonoid α)) :
    TrieCollides C y read cap K (pre ++ x :: rest)
      ∨ ((applyTrieProbeQ C y cap s x).eval read = (applyTrieProbeQ C y cap' s x).eval read
        ∧ ∀ s', (applyTrieProbeQ C y cap s x).eval read = .inl s' →
          CapInv C y read cap K (pre ++ [x]) s') := by
  classical
  unfold applyTrieProbeQ
  rw [eval_bind, eval_bind]
  set k := walkStartIn s.expanded x with hk
  have hkx : k ≤ x.toList.length := walkStartIn_le _ _
  generalize ho' : (probeOrAgreeQ s.tree s.edges k x).eval read = o
  have ho : probeOrAgree read s.tree s.edges k x = o := ho'
  have hpot : ∀ z ∈ harvestString o, z ∈ pot k x s.tree := fun z hz =>
    harvest_pot read z (ho ▸ hz)
  by_cases hr : ∃ z ∈ harvestString o, z ∈ s.seen.map Prod.fst
  · simp only [if_pos hr, eval_pure]
    set cands := prefixOf x k :: (s.seen.filter fun p => p.1 ∈ harvestString o).map Prod.snd
    by_cases hall : ∀ v ∈ cands, v.toList.length < K
    · right
      have hcap : ∀ c, K ≤ c → s.skipExpands c o (prefixOf x k) = s.expanded ∪ cands.toFinset := by
        intro c hc
        unfold TrieState.skipExpands
        rw [List.filter_eq_self.2 fun v hv => by
          simpa using (hall v hv).trans_le hc]
      refine ⟨by simp only [TrieState.skipRepeat, hcap cap hK, hcap cap' hK'], ?_⟩
      intro s' hs'
      simp only [Sum.inl.injEq] at hs'
      subst hs'
      refine ⟨?_, ?_⟩
      · intro u hu
        simp only [TrieState.skipRepeat, hcap cap hK, Finset.mem_union, List.mem_toFinset] at hu
        rcases hu with hu | hu
        · exact hinv.1 u hu
        · exact hall u hu
      · intro q hq
        exact capInv_extend C y read hinv x q hq
    · left
      push Not at hall
      obtain ⟨v, hv, hlen⟩ := hall
      rcases List.mem_cons.1 hv with rfl | hv
      · obtain ⟨z, hz, hzs⟩ := hr
        obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hzs
        exact (collides_of_seen C y read hinv htree rest hq (hpot _ hz)).1
          (by rwa [length_prefixOf hkx] at hlen)
      · obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hv
        obtain ⟨hq, hqh⟩ := List.mem_filter.1 hq
        exact (collides_of_seen C y read hinv htree rest hq
          (hpot _ (by simpa using hqh))).2 hlen
  · rw [if_neg hr, if_neg hr]
    right
    refine ⟨rfl, ?_⟩
    intro s' h
    have hc : ∀ o', (∀ z ∈ harvestString o', z ∈ pot k x s.tree) →
        applyTrieTests C y (countTrieOutcome s o' (prefixOf x k)) = .inl s' →
        CapInv C y read cap K (pre ++ [x]) s' := by
      intro o' hpot' h'
      rw [applyTrieTests_inl C y h']
      refine ⟨hinv.1, ?_⟩
      intro q hq
      simp only [countTrieOutcome, List.mem_append, List.mem_map] at hq
      rcases hq with hq | ⟨z, hz, rfl⟩
      · exact capInv_extend C y read hinv x q hq
      · exact ⟨pre, x, [], rfl, htree, k, hkx, rfl, hpot' z hz⟩
    cases o with
    | unlearnedEdge p c u t =>
      simp only [eval_pure, Sum.inl.injEq] at h; subst h
      exact ⟨hinv.1, by simp [newTrieHypothesis]⟩
    | wrongEdge p c u t =>
      simp only at h
      split at h
      · rw [eval_map, splitTrieLeafQ, eval_map, Sum.inl.injEq] at h
        subst h
        exact ⟨hinv.1, by simp [newTrieHypothesis]⟩
      · rw [eval_pure] at h; exact hc _ hpot h
    | _ => rw [eval_pure] at h; exact hc _ hpot h

theorem capRun {cap cap' K : ℕ} (hK : K ≤ cap) (hK' : K ≤ cap') :
    ∀ (rest pre : List (FreeMonoid α)) (s : TrieState α), CapInv C y read cap K pre s →
      (∀ ys, (runTrieProbesQ C y cap trieInitialState (pre ++ ys)).eval read
        = (runTrieProbesQ C y cap s ys).eval read) →
      TrieCollides C y read cap K (pre ++ rest)
        ∨ ((runTrieProbesQ C y cap s rest).eval read = (runTrieProbesQ C y cap' s rest).eval read
          ∧ ∀ u ∈ ((runTrieProbesQ C y cap s rest).eval read).1.expanded, u.toList.length < K)
  | [], pre, s, hinv, _ => Or.inr ⟨by simp [runTrieProbesQ], by simpa [runTrieProbesQ] using hinv.1⟩
  | x :: rest, pre, s, hinv, hrun => by
    have htree : ((runTrieProbesQ C y cap trieInitialState pre).eval read).1.tree = s.tree := by
      have := hrun []
      rw [List.append_nil] at this
      rw [this]; simp [runTrieProbesQ]
    rcases capStep C y read hK hK' hinv htree x rest with h | ⟨heq, hinv'⟩
    · exact Or.inl h
    rcases ho : (applyTrieProbeQ C y cap s x).eval read with s' | e
    · have ho' : (applyTrieProbeQ C y cap' s x).eval read = .inl s' := heq ▸ ho
      have hrun' : ∀ ys, (runTrieProbesQ C y cap trieInitialState ((pre ++ [x]) ++ ys)).eval read
          = (runTrieProbesQ C y cap s' ys).eval read := by
        intro ys
        rw [List.append_assoc, List.singleton_append, hrun]
        show ((applyTrieProbeQ C y cap s x).bind _).eval read = _
        rw [eval_bind, ho]
      rcases capRun hK hK' rest (pre ++ [x]) s' (hinv' s' ho) hrun' with h | h
      · left; simpa using h
      · right
        unfold runTrieProbesQ
        rw [eval_bind, eval_bind, ho, ho']
        exact h
    · right
      have ho' : (applyTrieProbeQ C y cap' s x).eval read = .inr e := heq ▸ ho
      unfold runTrieProbesQ
      rw [eval_bind, eval_bind, ho, ho']
      exact ⟨rfl, hinv.1⟩

open scoped Classical in
/-- The `K`-prefixes of another draw's prefix of length at most `K` followed by a midfix of the
tree draw `a` is probed under. -/
def capSet (cap K : ℕ) {P : ℕ} (d : Fin P → FreeMonoid α) (a : Fin P) : Finset (FreeMonoid α) :=
  ((((Finset.univ.erase a).image d) ×ˢ Finset.range (K + 1)) ×ˢ
    mids ((runTrieProbesQ C y cap trieInitialState ((List.ofFn d).take a)).eval read).1.tree).image
    fun q => prefixOf (prefixOf q.1.1 q.1.2 * q.2) K

omit [Fintype α] [DecidableEq α] in
theorem take_ofFn_congr {P : ℕ} {d e : Fin P → FreeMonoid α} {a : Fin P}
    (h : ∀ l, l ≠ a → d l = e l) : (List.ofFn d).take a = (List.ofFn e).take a := by
  apply List.ext_getElem (by simp)
  intro i h1 h2
  simp only [List.getElem_take, List.getElem_ofFn]
  apply h
  intro hia
  have : i < (a : ℕ) := by simp at h1; omega
  rw [← hia] at this
  simp at this

theorem capSet_congr (cap K : ℕ) {P : ℕ} {d e : Fin P → FreeMonoid α} (a : Fin P)
    (h : ∀ l, l ≠ a → d l = e l) : capSet C y read cap K d a = capSet C y read cap K e a := by
  classical
  unfold capSet
  rw [take_ofFn_congr h, Finset.image_congr fun l hl => h l (Finset.ne_of_mem_erase hl)]

theorem capSet_card (cap K : ℕ) {P : ℕ} (d : Fin P → FreeMonoid α) (a : Fin P) :
    (capSet C y read cap K d a).card ≤ P * (K + 1) * P := by
  classical
  unfold capSet
  refine Finset.card_image_le.trans ?_
  rw [Finset.card_product, Finset.card_product, Finset.card_range]
  have h1 : ((Finset.univ.erase a).image d).card ≤ P :=
    Finset.card_image_le.trans ((Finset.card_erase_le).trans (by simp))
  have h2 := mids_card_runTrieProbes C y read cap trieInitialState ((List.ofFn d).take a)
  have h3 : (mids (trieInitialState : TrieState α).tree).card = 1 := by
    simp [trieInitialState, newTrieHypothesis, mids]
  rw [h3, List.length_take, List.length_ofFn] at h2
  have h4 : 1 + min (a : ℕ) P ≤ P := by have := a.isLt; omega
  exact Nat.mul_le_mul (Nat.mul_le_mul_right _ h1) (h2.trans h4)

theorem capSet_of_collides (cap K : ℕ) {P : ℕ} (d : Fin P → FreeMonoid α)
    (h : TrieCollides C y read cap K (List.ofFn d)) :
    ∃ a : Fin P, prefixOf (d a) K ∈ capSet C y read cap K d a := by
  classical
  obtain ⟨pre, x, post, hxs, w, hw, p, hp, m, hm, heq⟩ := h
  have hP : pre.length + post.length + 1 = P := by
    have := congrArg List.length hxs; simp at this; omega
  have hlen : pre.length < P := by omega
  have hget : ∀ (i : ℕ) (hi : i < P), d ⟨i, hi⟩
      = (pre ++ x :: post)[i]'(by simp; omega) := by
    intro i hi
    rw [← List.getElem_ofFn (f := d) (h := by simpa using hi)]
    exact List.getElem_of_eq hxs _
  refine ⟨⟨pre.length, hlen⟩, ?_⟩
  have hx : d ⟨pre.length, hlen⟩ = x := by rw [hget]; simp
  have htake : (List.ofFn d).take pre.length = pre := by rw [hxs]; simp
  have hwb : ∃ b : Fin P, b ≠ ⟨pre.length, hlen⟩ ∧ d b = w := by
    rcases List.mem_append.1 hw with hw | hw
    · obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hw
      refine ⟨⟨i, by omega⟩, fun h => by simp [Fin.ext_iff] at h; omega, ?_⟩
      rw [hget, List.getElem_append_left hi]
    · obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hw
      have : pre.length + 1 + j < P := by omega
      refine ⟨⟨pre.length + 1 + j, this⟩, fun h => by simp [Fin.ext_iff] at h; omega, ?_⟩
      rw [hget, List.getElem_append_right (by omega)]
      simp only [show pre.length + 1 + j - pre.length = j + 1 by omega, List.getElem_cons_succ]
  obtain ⟨b, hb, rfl⟩ := hwb
  unfold capSet
  rw [hx, heq]
  refine Finset.mem_image.2 ⟨((d b, p), m), ?_, rfl⟩
  simp only [Finset.mem_product, Finset.mem_image, Finset.mem_erase, Finset.mem_univ, and_true,
    Finset.mem_range]
  refine ⟨⟨⟨b, hb, rfl⟩, by omega⟩, ?_⟩
  simpa [htake] using hm

theorem level3TrieCapFree : Level3TrieCapFree := by
  intro α _ _ C y read D _ K cap cap' P p₀ hpre hK hK'
  classical
  have hp₀ : 0 ≤ p₀ := measureReal_nonneg.trans (hpre 1)
  have hsub : {draws : Fin P → FreeMonoid α | indecisiveLStarLevel3Trie C y cap read draws
        ≠ indecisiveLStarLevel3Trie C y cap' read draws
      ∨ ∃ u ∈ (indecisiveLStarLevel3Trie C y cap read draws).expanded, K ≤ u.toList.length}
      ⊆ ⋃ a : Fin P, {d | d a ∈ {x | prefixOf x K ∈ capSet C y read cap K d a}} := by
    intro d hd
    rcases capRun C y read hK hK' (List.ofFn d) [] trieInitialState
        ⟨by simp [trieInitialState, newTrieHypothesis], by simp [trieInitialState, newTrieHypothesis]⟩
        (fun ys => rfl) with h | ⟨heq, hlt⟩
    · obtain ⟨a, ha⟩ := capSet_of_collides C y read cap K d (by simpa using h)
      exact Set.mem_iUnion.2 ⟨a, ha⟩
    · exfalso
      unfold indecisiveLStarLevel3Trie level3TrieQ at hd
      simp only [Set.mem_ofPred_eq, eval_map] at hd
      rw [heq] at hd
      rcases hd with hne | ⟨u, hu, hle⟩
      · exact hne rfl
      · rw [← heq] at hu
        exact absurd (hlt u hu) (by omega)
  have hq : ∀ (a : Fin P) (d : Fin P → FreeMonoid α),
      D {x | prefixOf x K ∈ capSet C y read cap K d a} ≤ ENNReal.ofReal (P * (K + 1) * P * p₀) := by
    intro a d
    have hS : {x | prefixOf x K ∈ capSet C y read cap K d a}
        = ⋃ u ∈ capSet C y read cap K d a, {x | prefixOf x K = u} := by
      ext x; simp
    have hu : ∀ u, D {x | prefixOf x K = u} ≤ ENNReal.ofReal p₀ := fun u =>
      (ENNReal.ofReal_toReal (measure_ne_top _ _)).symm.le.trans (ENNReal.ofReal_le_ofReal (hpre u))
    calc _ ≤ ∑ u ∈ capSet C y read cap K d a, D {x | prefixOf x K = u} :=
          hS ▸ measure_biUnion_finset_le _ _
      _ ≤ ∑ _u ∈ capSet C y read cap K d a, ENNReal.ofReal p₀ := Finset.sum_le_sum fun u _ => hu u
      _ = ENNReal.ofReal ((capSet C y read cap K d a).card * p₀) := by
          rw [Finset.sum_const, nsmul_eq_mul, ENNReal.ofReal_mul (by positivity),
            ENNReal.ofReal_natCast]
      _ ≤ _ := ENNReal.ofReal_le_ofReal (mul_le_mul_of_nonneg_right
          (by exact_mod_cast capSet_card C y read cap K d a) hp₀)
  rw [measureReal_def]
  refine ENNReal.toReal_le_of_le_ofReal (by positivity) ?_
  calc _ ≤ _ := measure_mono hsub
    _ ≤ ∑ a : Fin P, (Measure.pi fun _ : Fin P => D)
          {d | d a ∈ {x | prefixOf x K ∈ capSet C y read cap K d a}} := measure_iUnion_fintype_le _ _
    _ ≤ ∑ _a : Fin P, ENNReal.ofReal (P * (K + 1) * P * p₀) := Finset.sum_le_sum fun a _ =>
        pi_hit_le D a (fun d => {x | prefixOf x K ∈ capSet C y read cap K d a})
          (fun d e h => by simp only [capSet_congr C y read cap K a h]) _ (hq a)
    _ = _ := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
          ← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul (by positivity)]
        congr 1; ring

end Random

end OrthoDFA
