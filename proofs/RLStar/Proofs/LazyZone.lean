import RLStar.Proofs.LoopMeasurable
import RLStar.Proofs.QueryLevel3
import RLStar.LowLevelClaims.LazyRound
import RLStar.Proofs.LazyShape
import Mathlib.Data.List.ReduceOption

/-!
# Where a lazy round reads

A lazy round started from the trie of the clustering's prefixes `CP`, over probes whose first `k`
letters are no prefix's of `CP`, walks each probe from a prefix of it no prefix of `CP` shares
(`LazyKeyOff`), and reads only such strings, at most `probes + 1 + T + 2` long
(`reads_lazyRoundQ_keyOff`). With every epoch's tree at most `n + 2` leaves it asks at most
`lazyReadCap` strings (`card_reads_lazyRoundQ_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- Some prefix of `z`, at most `k` letters, is no prefix of a string of `CP`. -/
def LazyKeyOff (k : ℕ) (CP : Finset (FreeMonoid α)) (z : FreeMonoid α) : Prop :=
  ∃ i ≤ k, i ≤ z.toList.length ∧ ∀ p ∈ CP, i ≤ p.toList.length → prefixOf p i ≠ prefixOf z i

open scoped Classical in
/-- The strings at most `N'` long, `LazyKeyOff` but for the family's suffix. -/
def lazyZone (N' k : ℕ) (CP F : Finset (FreeMonoid α)) : Finset (FreeMonoid α) :=
  (readStrings (shortStrings N') F).filter fun w =>
    ∃ z, LazyKeyOff k CP z ∧ z.toList.length ≤ N' ∧ ∃ v ∈ F, w = z * v

/-- Every probe and block draw of `d` satisfies `Φ`. -/
def LazyDrawsAll {E P J b : ℕ} (Φ : FreeMonoid α → Prop) (d : Fin E → Random.EpochDraws α P J b) :
    Prop :=
  ∀ e, (∀ i, Φ ((d e).1 i)) ∧ ∀ j t, Φ ((d e).2 j t)

/-- What a lazy round asks with every epoch's tree at most `nQ + 2` leaves, its probes and block
draws at most `Lp` long: per epoch, level 3 and the top-up's sifts, and level 3's refinements. -/
def lazyReadCap (C : Random.Level3Settings) (E J b Lp nQ nα : ℕ) : ℕ :=
  E * ((C.probeBudget + J * b) * (Random.Kappa.probeSifts (Lp + C.walkStart) * (nQ + 3))
    + (nQ + 2) * ((nQ + 2) * (nα + 1) * (nQ + 2)))

theorem trieOK_prefixTrie (k : ℕ) (CP : Finset (FreeMonoid α)) :
    Random.TrieOK k (Random.prefixTrie k CP) := by
  refine ⟨fun u hu => ?_, fun u hu i => ?_⟩ <;>
    obtain ⟨p, hp, hu⟩ := Finset.mem_biUnion.1 hu <;>
    obtain ⟨j, hj, rfl⟩ := Finset.mem_image.1 hu <;>
    rw [Finset.mem_range, lt_min_iff] at hj
  · rw [length_prefixOf]
    omega
  · have he : prefixOf (prefixOf p j) i = prefixOf p (min i j) := by
      simp [prefixOf, List.take_take]
    rw [he]
    exact Finset.mem_biUnion.2 ⟨p, hp, Finset.mem_image.2
      ⟨min i j, Finset.mem_range.2 (lt_min_iff.2 ⟨by omega, by omega⟩), rfl⟩⟩

section Zone

open Random Query

/-- An answer other than `a`: a read changed by `rot` off a set agrees with the old one exactly on
the set, so `step_reads_agree` puts every read in it. -/
private def rot : ARU → ARU
  | .accept => .reject
  | .reject => .undecided
  | .undecided => .accept

private theorem rot_ne (a : ARU) : rot a ≠ a := by cases a <;> simp [rot]

/-- A level-3 run reads a probe's prefix at least `walkStart` long followed by at most `B`
letters, where the steps left keep the midfixes within `B`. -/
private theorem reads_runProbesQ_zone {β : Type*} [Fintype β] [DecidableEq β]
    (read : FreeMonoid β → ARU) (C : Level3Settings) (X : List (FreeMonoid β)) (B : ℕ) :
    ∀ (xs : List (FreeMonoid β)) (s : Level3State β), (∀ x ∈ xs, x ∈ X) → RunInv read C X s →
      s.tree.leaves.length - 2 + xs.length ≤ B →
      ∀ z ∈ (runProbesQ C s xs).reads read, ∃ x ∈ X, ∃ j, C.walkStart ≤ j
        ∧ j ≤ x.toList.length ∧ ∃ m : FreeMonoid β, m.toList.length ≤ B ∧ z = prefixOf x j * m
  | [], _, _, _, _ => by simp [runProbesQ]
  | x :: xs, s, hX, hs, hB => by
    classical
    intro z hz
    simp only [List.length_cons] at hB
    simp only [runProbesQ, reads_bind, Finset.mem_union] at hz
    rcases hz with hz | hz
    · by_contra hn
      have := step_reads_agree C (r := read)
        (r' := fun z => if ∃ x ∈ X, ∃ j, C.walkStart ≤ j ∧ j ≤ x.toList.length
          ∧ ∃ m : FreeMonoid β, m.toList.length ≤ B ∧ z = prefixOf x j * m then read z
          else rot (read z))
        (fun x' hx' j hj hjx m hm => by rw [if_pos ⟨x', hx', j, hj, hjx, m, hm, rfl⟩])
        (hX x (by simp)) hs (by omega) z (reads_applyProbeQ_sub read C s x hz)
      rw [if_neg hn] at this
      exact rot_ne _ this.symm
    · rcases hap : (applyProbeQ C s x).eval read with s' | e <;> rw [hap] at hz
      · obtain ⟨hs', hl⟩ := runInv_step read C (hX x (by simp)) hs hap
        exact reads_runProbesQ_zone read C X B xs s' (fun y hy => hX y (by simp [hy])) hs'
          (by omega) z hz
      · simp at hz

private theorem reads_level3Q_zone {β : Type*} [Fintype β] [DecidableEq β]
    (read : FreeMonoid β → ARU) (C : Level3Settings) {T : ℕ} (xs : Fin T → FreeMonoid β) :
    ∀ z ∈ (level3Q C xs).reads read, ∃ i j, C.walkStart ≤ j ∧ j ≤ (xs i).toList.length
      ∧ ∃ m : FreeMonoid β, m.toList.length ≤ T ∧ z = prefixOf (xs i) j * m := by
  intro z hz
  rw [level3Q, reads_map] at hz
  obtain ⟨x, hx, j, h1, h2, m, hm, rfl⟩ := reads_runProbesQ_zone read C (List.ofFn xs) T
    (List.ofFn xs) initialState (fun _ h => h) (runInv_start read C _)
    (by simp [initialState, newHypothesis, DTree.leaves]) z hz
  obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
  exact ⟨i, j, h1, h2, m, hm, rfl⟩

omit [Fintype α] [DecidableEq α] in
/-- `z` is a draw satisfying `Φ`, padded from a trie holding `X₀`, cut at least `K` letters in and
followed by at most `P` letters, its pads dropped. -/
private def PadZone (K P : ℕ) (X₀ : Finset (FreeMonoid α)) (Φ : FreeMonoid α → Prop)
    (z : FreeMonoid α) : Prop :=
  ∃ x, Φ x ∧ ∃ X, X₀ ⊆ X ∧ ∃ j, K ≤ j ∧ j ≤ (padIn K X x).toList.length
    ∧ ∃ m : FreeMonoid (Option α), m.toList.length ≤ P ∧ z = unpad (prefixOf (padIn K X x) j * m)

private theorem reads_epochQ_zone (read : FreeMonoid α → ARU) (C : Level3Settings) (K : ℕ)
    {X₀ X : Finset (FreeMonoid α)} (hX : X₀ ⊆ X) {P : ℕ} (xs : Fin P → FreeMonoid α)
    {Φ : FreeMonoid α → Prop} (hΦ : ∀ i, Φ (xs i)) :
    ∀ z ∈ (epochQ C K X xs).reads read, PadZone K P X₀ Φ z := by
  intro z hz
  rw [epochQ, reads_subst] at hz
  obtain ⟨q, hq, hzq⟩ := Finset.mem_biUnion.1 hz
  simp only [unpadQ, reads_query, Finset.mem_singleton] at hzq
  obtain ⟨i, j, h1, h2, m, hm, rfl⟩ := reads_level3Q_zone _ _ _ q hq
  exact ⟨xs i, hΦ i, X, hX, j, h1, h2, m, hm, hzq⟩

private theorem midsLe_epochQ (read : FreeMonoid α → ARU) (C : Level3Settings) (K : ℕ)
    (X : Finset (FreeMonoid α)) {P : ℕ} (xs : Fin P → FreeMonoid α) :
    MidsLe P ((epochQ C K X xs).eval read).hypothesis.tree := by
  have h := (runProbes_tree (padRead read) { C with walkStart := K }
    (List.ofFn fun i => padIn K X (xs i))).2.2
  rw [List.length_ofFn] at h
  rw [epochQ, eval_subst, level3Q, eval_map]
  exact h

omit [DecidableEq α] in
private theorem reads_harvest_zone (read : FreeMonoid α → ARU) (K P : ℕ)
    (H : HypothesisDFA (Option α)) (hH : MidsLe P H.tree) (s : Level3Site)
    {X₀ X : Finset (FreeMonoid α)} (hX : X₀ ⊆ X) {Φ : FreeMonoid α → Prop} {x : FreeMonoid α}
    (hx : Φ x) :
    ∀ z ∈ ((siteHarvestQ K H s (padIn K X x)).subst unpadQ).reads read, PadZone K P X₀ Φ z := by
  intro z hz
  rw [reads_subst] at hz
  obtain ⟨q, hq, hzq⟩ := Finset.mem_biUnion.1 hz
  simp only [unpadQ, reads_query, Finset.mem_singleton] at hzq
  have := Kappa.probeReads_pot H.edges K (reads_siteHarvestQ_sub _ K H s _ hq)
  simp only [pot, Finset.mem_biUnion, Finset.mem_image, Finset.mem_Icc] at this
  obtain ⟨j, ⟨h1, h2⟩, m, hm, rfl⟩ := this
  exact ⟨x, hx, X, hX, j, h1, h2, m, hH m hm, hzq⟩

omit [DecidableEq α] in
private theorem reads_blockQ_zone (read : FreeMonoid α → ARU) (K P : ℕ)
    (H : HypothesisDFA (Option α)) (hH : MidsLe P H.tree) (s : Level3Site)
    {X₀ X : Finset (FreeMonoid α)} (hX : X₀ ⊆ X) {Φ : FreeMonoid α → Prop} :
    ∀ xs : List (FreeMonoid α), (∀ x ∈ xs, Φ x) →
      ∀ z ∈ (blockQ K H s X xs).reads read, PadZone K P X₀ Φ z
  | [], _ => by simp [blockQ]
  | x :: xs, hxs => by
    intro z hz
    simp only [blockQ, reads_bind, reads_map, Finset.mem_union] at hz
    rcases hz with hz | hz
    · exact reads_harvest_zone read K P H hH s hX (hxs x (by simp)) z hz
    · exact reads_blockQ_zone read K P H hH s hX xs (fun y hy => hxs y (by simp [hy])) z hz

omit [DecidableEq α] in
private theorem topUpQ_zone (read : FreeMonoid α → ARU) (K nh P : ℕ)
    (H : HypothesisDFA (Option α)) (hH : MidsLe P H.tree) (s : Level3Site)
    {X₀ : Finset (FreeMonoid α)} {Φ : FreeMonoid α → Prop} {b : ℕ} :
    ∀ (Bs : List (Fin b → FreeMonoid α)) (X : Finset (FreeMonoid α)) (ℓ : List (FreeMonoid α)),
      X₀ ⊆ X → (∀ B ∈ Bs, ∀ t, Φ (B t)) →
      (∀ z ∈ (topUpQ K nh H s Bs X ℓ).reads read, PadZone K P X₀ Φ z)
        ∧ ∀ ℓ' X' rest, (topUpQ K nh H s Bs X ℓ).eval read = some (.retract ℓ' X', rest) →
          X₀ ⊆ X'
  | [], _, _, _, _ => by simp [topUpQ]
  | B :: Bs, X, ℓ, hX, hB => by
    have hblk := reads_blockQ_zone read K P H hH s hX (List.ofFn B) fun x hx => by
      obtain ⟨t, rfl⟩ := List.mem_ofFn.1 hx
      exact hB B (by simp) t
    simp only [topUpQ, eval_bind, reads_bind]
    generalize (blockQ K H s X (List.ofFn B)).eval read = ws
    split_ifs with hfull hret
    · refine ⟨fun z hz => ?_, fun _ _ _ h => by simp at h⟩
      rcases Finset.mem_union.1 hz with hz | hz
      · exact hblk z hz
      · simp at hz
    · refine ⟨fun z hz => ?_, fun ℓ' X' rest h => ?_⟩
      · rcases Finset.mem_union.1 hz with hz | hz
        · exact hblk z hz
        · simp at hz
      · simp only [eval_pure, Option.some.injEq, Prod.mk.injEq, TopUpEnd.retract.injEq] at h
        rw [← h.1.2]
        exact hX
    · obtain ⟨ih1, ih2⟩ := topUpQ_zone read K nh P H hH s Bs (X ∪ (blockUpdate K X ℓ ws).2)
        (blockUpdate K X ℓ ws).1 (hX.trans Finset.subset_union_left)
        fun B' hB' => hB B' (by simp [hB'])
      exact ⟨fun z hz => (Finset.mem_union.1 hz).elim (hblk z) (ih1 z), ih2⟩

private theorem lazyEpochsQ_zone (read : FreeMonoid α → ARU) (C : Level3Settings) (K nh : ℕ)
    {J b : ℕ} {X₀ : Finset (FreeMonoid α)} {Φ : FreeMonoid α → Prop} :
    ∀ (e pk : ℕ) (ds : List (EpochDraws α C.probeBudget J b)) (X : Finset (FreeMonoid α))
      (ls : LazyLists α), X₀ ⊆ X → (∀ d ∈ ds, (∀ i, Φ (d.1 i)) ∧ ∀ j t, Φ (d.2 j t)) →
      ∀ z ∈ (lazyEpochsQ C K nh e pk ds X ls).reads read, PadZone K C.probeBudget X₀ Φ z
  | _, _, [], X, ls, _, _ => by simp [lazyEpochsQ]
  | e, pk, d :: ds, X, ls, hX, hd => by
    obtain ⟨hd1, hd2⟩ := hd d (by simp)
    have hE := reads_epochQ_zone read (epochSettings C e) K hX d.1 hd1
    have hH := midsLe_epochQ read (epochSettings C e) K X d.1
    intro z hz
    simp only [lazyEpochsQ, reads_bind, Finset.mem_union] at hz
    rcases hz with hz | hz
    · exact hE z hz
    generalize (epochQ (epochSettings C e) K X d.1).eval read = res at hz hH
    rcases res with ⟨hyp, _ | s | _⟩
    · simp at hz
    · simp only [reads_bind, Finset.mem_union] at hz
      have hT := topUpQ_zone read K nh C.probeBudget hyp hH s (List.ofFn d.2) X (ls s) hX
        fun B hB t => by
          obtain ⟨j, rfl⟩ := List.mem_ofFn.1 hB
          exact hd2 j t
      rcases hz with hz | hz
      · exact hT.1 z hz
      · generalize ho : (topUpQ K nh hyp s (List.ofFn d.2) X (ls s)).eval read = o at hz
        rcases o with _ | ⟨_ | ⟨ℓ, X'⟩, rest⟩
        · simp at hz
        · simp at hz
        · exact lazyEpochsQ_zone read C K nh (e + 1) _ ds X' _ (hT.2 ℓ X' rest ho)
            (fun d' hd' => hd d' (by simp [hd'])) z hz
    · simp at hz

omit [Fintype α] [DecidableEq α] in
private theorem toList_unpad_prefixOf_padIn (K : ℕ) (X : Finset (FreeMonoid α))
    (x : FreeMonoid α) (p : ℕ) :
    (unpad (prefixOf (padIn K X x) p)).toList = x.toList.take (p - (K - walkStartIn X x)) := by
  simp only [unpad, prefixOf, padIn, FreeMonoid.toList_ofList, List.take_append,
    List.length_replicate, List.take_replicate, List.reduceOption_append,
    List.reduceOption_replicate_none, List.nil_append, ← List.map_take]
  simp [List.reduceOption]

omit [DecidableEq α] in
private theorem toList_padZone (K : ℕ) (X : Finset (FreeMonoid α)) (x : FreeMonoid α) (j : ℕ)
    (m : FreeMonoid (Option α)) :
    (unpad (prefixOf (padIn K X x) j * m)).toList
      = x.toList.take (j - (K - walkStartIn X x)) ++ m.toList.reduceOption := by
  rw [← toList_unpad_prefixOf_padIn]
  simp [unpad, List.reduceOption_append]

omit [DecidableEq α] in
private theorem length_of_padZone {K P N : ℕ} {X₀ : Finset (FreeMonoid α)}
    {Φ : FreeMonoid α → Prop} (hΦ : ∀ x, Φ x → x.toList.length ≤ N) {z : FreeMonoid α}
    (h : PadZone K P X₀ Φ z) : z.toList.length ≤ N + P := by
  obtain ⟨x, hx, X, -, j, -, -, m, hm, rfl⟩ := h
  rw [toList_padZone, List.length_append, List.length_take]
  have := List.reduceOption_length_le m.toList
  have := hΦ x hx
  omega

omit [Fintype α] [DecidableEq α] in
private theorem walkStartIn_mono {X Y : Finset (FreeMonoid α)} (h : X ⊆ Y) (x : FreeMonoid α) :
    walkStartIn X x ≤ walkStartIn Y x := by
  by_contra hlt
  obtain ⟨hX1, hX2, -⟩ := walkStartIn_spec X x
  exact (walkStartIn_spec Y x).2.2 (by omega) (h (hX2 _ (by omega)))

private theorem keyOff_of_padZone {k P : ℕ} {CP : Finset (FreeMonoid α)}
    {Φ : FreeMonoid α → Prop}
    (hΦ : ∀ x, Φ x → k ≤ x.toList.length ∧ Avoids k ↑(CP.image fun p => prefixOf p k) x)
    {z : FreeMonoid α} (h : PadZone k P (prefixTrie k CP) Φ z) : LazyKeyOff k CP z := by
  obtain ⟨x, hx, X, hX, j, hj, -, m, -, rfl⟩ := h
  obtain ⟨hkx, hav⟩ := hΦ x hx
  obtain ⟨-, hin, hout⟩ := walkStartIn_spec (prefixTrie k CP) x
  have hik : walkStartIn (prefixTrie k CP) x ≤ k := by
    by_contra hlt
    have := (trieOK_prefixTrie k CP).1 _ (hin k (by omega))
    rw [length_prefixOf] at this
    omega
  have hmono := walkStartIn_mono hX x
  set i := walkStartIn (prefixTrie k CP) x
  have hij : i ≤ j - (k - walkStartIn X x) := by omega
  have hpre : prefixOf (unpad (prefixOf (padIn k X x) j * m)) i = prefixOf x i := by
    apply FreeMonoid.toList.injective
    change List.take i (unpad (prefixOf (padIn k X x) j * m)).toList = List.take i x.toList
    rw [toList_padZone, List.take_append_of_le_length (by rw [List.length_take]; omega),
      List.take_take, min_eq_left hij]
  refine ⟨i, hik, ?_, fun p hp hip => ?_⟩
  · rw [toList_padZone, List.length_append, List.length_take]
    omega
  rw [hpre]
  intro he
  rcases lt_or_eq_of_le hik with hlt | heq
  · refine hout (by omega) ?_
    rw [← he]
    exact Finset.mem_biUnion.2 ⟨p, hp, Finset.mem_image.2
      ⟨i, Finset.mem_range.2 (lt_min_iff.2 ⟨hlt, by omega⟩), rfl⟩⟩
  · rw [heq] at he
    exact hav hkx (Finset.mem_coe.2 (Finset.mem_image.2 ⟨p, hp, he⟩))

omit [Fintype α] [DecidableEq α] in
private theorem lazyDrawsAll_len {E P J b : ℕ} (d : Fin E → EpochDraws α P J b) :
    LazyDrawsAll (fun x => x.toList.length ≤ lazyProbesLen d) d := by
  intro e
  have he : max (probesLen (d e).1) (Finset.univ.sup fun j => probesLen ((d e).2 j))
      ≤ lazyProbesLen d :=
    Finset.le_sup (f := fun e => max (probesLen (d e).1)
      (Finset.univ.sup fun j => probesLen ((d e).2 j))) (Finset.mem_univ e)
  refine ⟨fun i => (le_probesLen _ i).trans ((le_max_left _ _).trans he), fun j t => ?_⟩
  exact (le_probesLen _ t).trans ((Finset.le_sup (f := fun j => probesLen ((d e).2 j))
    (Finset.mem_univ j)).trans ((le_max_right _ _).trans he))

private theorem lazyRoundQ_zone (read : FreeMonoid α → ARU) (C : Level3Settings) (K nh : ℕ)
    {E J b : ℕ} (X₀ : Finset (FreeMonoid α)) (d : Fin E → EpochDraws α C.probeBudget J b)
    {Φ : FreeMonoid α → Prop} (hd : LazyDrawsAll Φ d) :
    ∀ z ∈ (lazyRoundQ C K nh X₀ d).reads read, PadZone K C.probeBudget X₀ Φ z :=
  lazyEpochsQ_zone read C K nh 0 0 (List.ofFn d) X₀ _ subset_rfl fun d' hd' => by
    obtain ⟨e, rfl⟩ := List.mem_ofFn.1 hd'
    exact hd e

omit [Fintype α] [DecidableEq α] in
private theorem length_padIn_le (K : ℕ) (X : Finset (FreeMonoid α)) (x : FreeMonoid α) :
    (padIn K X x).toList.length ≤ x.toList.length + K := by
  simp only [padIn, FreeMonoid.toList_ofList, List.length_append, List.length_replicate,
    List.length_map]
  omega

private theorem card_reads_epochQ_le (read : FreeMonoid α → ARU) (C : Level3Settings) (K : ℕ)
    (X : Finset (FreeMonoid α)) {P : ℕ} (xs : Fin P → FreeMonoid α) (nQ Lp : ℕ)
    (hL : ∀ i, (xs i).toList.length ≤ Lp)
    (hn : ((epochQ C K X xs).eval read).hypothesis.tree.leaves.length ≤ nQ + 2) :
    ((epochQ C K X xs).reads read).card ≤ P * (Kappa.probeSifts (Lp + K) * (nQ + 1))
      + nQ * ((nQ + 2) * (Fintype.card α + 1) * (nQ + 2)) := by
  rw [epochQ, eval_subst] at hn
  rw [epochQ]
  refine (card_reads_subst_le _ _ _ (n := 1) fun q _ => by simp [unpadQ]).trans ?_
  rw [mul_one]
  have h := card_reads_level3Q_le (padRead read) { C with walkStart := K } nQ
    (fun i => padIn K X (xs i)) (L := Lp + K)
    (fun i => (length_padIn_le K X (xs i)).trans (by have := hL i; omega)) hn
  rw [Fintype.card_option] at h
  exact h

omit [DecidableEq α] in
private theorem card_reads_harvest_le (read : FreeMonoid α → ARU) (K Lp nQ : ℕ)
    (H : HypothesisDFA (Option α)) (hH : (mids H.tree).card ≤ nQ + 1) (s : Level3Site)
    (X : Finset (FreeMonoid α)) {x : FreeMonoid α} (hx : x.toList.length ≤ Lp) :
    (((siteHarvestQ K H s (padIn K X x)).subst unpadQ).reads read).card
      ≤ Kappa.probeSifts (Lp + K) * (nQ + 1) := by
  refine (card_reads_subst_le _ _ _ (n := 1) fun q _ => by simp [unpadQ]).trans ?_
  rw [mul_one]
  refine (Finset.card_le_card (reads_siteHarvestQ_sub _ K H s _)).trans ?_
  refine (Spur.probeReads_card_mids _ (L := Lp + K)
    ((length_padIn_le K X x).trans (by omega))).trans ?_
  exact Nat.mul_le_mul_left _ hH

omit [DecidableEq α] in
private theorem card_reads_blockQ_le (read : FreeMonoid α → ARU) (K Lp nQ : ℕ)
    (H : HypothesisDFA (Option α)) (hH : (mids H.tree).card ≤ nQ + 1) (s : Level3Site)
    (X : Finset (FreeMonoid α)) :
    ∀ xs : List (FreeMonoid α), (∀ x ∈ xs, x.toList.length ≤ Lp) →
      ((blockQ K H s X xs).reads read).card ≤ xs.length * (Kappa.probeSifts (Lp + K) * (nQ + 1))
  | [], _ => by simp [blockQ]
  | x :: xs, hxs => by
    simp only [blockQ, reads_bind, reads_map, List.length_cons]
    refine (Finset.card_union_le _ _).trans ?_
    have h1 := card_reads_harvest_le read K Lp nQ H hH s X (hxs x (by simp))
    have h2 := card_reads_blockQ_le read K Lp nQ H hH s X xs fun y hy => hxs y (by simp [hy])
    generalize Kappa.probeSifts (Lp + K) * (nQ + 1) = A at h1 h2 ⊢
    calc _ ≤ A + xs.length * A := Nat.add_le_add h1 h2
      _ = (xs.length + 1) * A := by ring

omit [DecidableEq α] in
private theorem card_reads_topUpQ_le (read : FreeMonoid α → ARU) (K nh Lp nQ : ℕ)
    (H : HypothesisDFA (Option α)) (hH : (mids H.tree).card ≤ nQ + 1) (s : Level3Site) {b : ℕ} :
    ∀ (Bs : List (Fin b → FreeMonoid α)) (X : Finset (FreeMonoid α)) (ℓ : List (FreeMonoid α)),
      (∀ B ∈ Bs, ∀ t, (B t).toList.length ≤ Lp) →
      ((topUpQ K nh H s Bs X ℓ).reads read).card
        ≤ Bs.length * (b * (Kappa.probeSifts (Lp + K) * (nQ + 1)))
  | [], _, _, _ => by simp [topUpQ]
  | B :: Bs, X, ℓ, hB => by
    have hblk := card_reads_blockQ_le read K Lp nQ H hH s X (List.ofFn B) fun x hx => by
      obtain ⟨t, rfl⟩ := List.mem_ofFn.1 hx
      exact hB B (by simp) t
    rw [List.length_ofFn] at hblk
    simp only [topUpQ, reads_bind, List.length_cons]
    refine (Finset.card_union_le _ _).trans ?_
    generalize (blockQ K H s X (List.ofFn B)).eval read = ws
    have hrest : ∀ X' ℓ', ((topUpQ K nh H s Bs X' ℓ').reads read).card
        ≤ Bs.length * (b * (Kappa.probeSifts (Lp + K) * (nQ + 1))) := fun X' ℓ' =>
      card_reads_topUpQ_le read K nh Lp nQ H hH s Bs X' ℓ' fun B' hB' => hB B' (by simp [hB'])
    generalize b * (Kappa.probeSifts (Lp + K) * (nQ + 1)) = A at hblk hrest ⊢
    have hle : ∀ n, n ≤ Bs.length * A → ((blockQ K H s X (List.ofFn B)).reads read).card + n
        ≤ (Bs.length + 1) * A := fun n hn => by
      calc _ ≤ A + Bs.length * A := Nat.add_le_add hblk hn
        _ = (Bs.length + 1) * A := by ring
    split_ifs
    · exact hle 0 (Nat.zero_le _)
    · exact hle 0 (Nat.zero_le _)
    · exact hle _ (hrest _ _)

private theorem epoch_arith (P J b PS nQ q l3 tu : ℕ) (h3 : l3 ≤ P * (PS * (nQ + 1)) + nQ * q)
    (ht : tu ≤ J * (b * (PS * (nQ + 1)))) :
    l3 + tu ≤ (P + J * b) * (PS * (nQ + 3)) + (nQ + 2) * q := by
  nlinarith

private theorem lazyEpochsQ_card (read : FreeMonoid α → ARU) (C : Level3Settings) (K nh : ℕ)
    {J b : ℕ} (Lp nQ : ℕ) :
    ∀ (e pk : ℕ) (ds : List (EpochDraws α C.probeBudget J b)) (X : Finset (FreeMonoid α))
      (ls : LazyLists α), (∀ d ∈ ds, (∀ i, (d.1 i).toList.length ≤ Lp)
        ∧ ∀ j t, (d.2 j t).toList.length ≤ Lp) →
      pk ≤ ((lazyEpochsQ C K nh e pk ds X ls).eval read).peak
        ∧ (((lazyEpochsQ C K nh e pk ds X ls).eval read).peak ≤ nQ + 2 →
          ((lazyEpochsQ C K nh e pk ds X ls).reads read).card
            ≤ ds.length * ((C.probeBudget + J * b) * (Kappa.probeSifts (Lp + K) * (nQ + 3))
              + (nQ + 2) * ((nQ + 2) * (Fintype.card α + 1) * (nQ + 2))))
  | _, _, [], X, ls, _ => by simp [lazyEpochsQ]
  | e, pk, d :: ds, X, ls, hd => by
    obtain ⟨hd1, hd2⟩ := hd d (by simp)
    have hl3 := card_reads_epochQ_le read (epochSettings C e) K X d.1 nQ Lp hd1
    simp only [lazyEpochsQ, eval_bind, reads_bind, List.length_cons]
    generalize (epochQ (epochSettings C e) K X d.1).reads read = RE at hl3 ⊢
    generalize (epochQ (epochSettings C e) K X d.1).eval read = res at hl3 ⊢
    have key : ∀ (hyp : HypothesisDFA (Option α)) (tu n : ℕ),
        (hyp.tree.leaves.length ≤ nQ + 2 → RE.card ≤ C.probeBudget
          * (Kappa.probeSifts (Lp + K) * (nQ + 1))
          + nQ * ((nQ + 2) * (Fintype.card α + 1) * (nQ + 2))) →
        hyp.tree.leaves.length ≤ nQ + 2 →
        tu ≤ J * (b * (Kappa.probeSifts (Lp + K) * (nQ + 1))) →
        n ≤ ds.length * ((C.probeBudget + J * b) * (Kappa.probeSifts (Lp + K) * (nQ + 3))
          + (nQ + 2) * ((nQ + 2) * (Fintype.card α + 1) * (nQ + 2))) →
        RE.card + tu + n ≤ (ds.length + 1)
          * ((C.probeBudget + J * b) * (Kappa.probeSifts (Lp + K) * (nQ + 3))
            + (nQ + 2) * ((nQ + 2) * (Fintype.card α + 1) * (nQ + 2))) := by
      intro hyp tu n h3 hlv ht hn
      have := epoch_arith _ _ _ _ _ _ _ _ (h3 hlv) ht
      rw [add_one_mul]
      omega
    have hstop : ∀ (hyp : HypothesisDFA (Option α)) (R : Finset (FreeMonoid α)) (tu : ℕ),
        (hyp.tree.leaves.length ≤ nQ + 2 → RE.card ≤ C.probeBudget
          * (Kappa.probeSifts (Lp + K) * (nQ + 1))
          + nQ * ((nQ + 2) * (Fintype.card α + 1) * (nQ + 2))) →
        (hyp.tree.leaves.length ≤ nQ + 2 → R.card ≤ tu) →
        (hyp.tree.leaves.length ≤ nQ + 2 →
          tu ≤ J * (b * (Kappa.probeSifts (Lp + K) * (nQ + 1)))) →
        max pk hyp.tree.leaves.length ≤ nQ + 2 →
        (RE ∪ R).card ≤ (ds.length + 1)
          * ((C.probeBudget + J * b) * (Kappa.probeSifts (Lp + K) * (nQ + 3))
            + (nQ + 2) * ((nQ + 2) * (Fintype.card α + 1) * (nQ + 2))) := by
      intro hyp R tu h3 hR ht hp
      have hlv := (le_max_right _ _).trans hp
      have := key hyp tu 0 h3 hlv (ht hlv) (Nat.zero_le _)
      have := Finset.card_union_le RE R
      have := hR hlv
      omega
    rcases res with ⟨hyp, _ | s | _⟩
    · dsimp only at hl3 ⊢
      simp only [eval_pure, reads_pure, Finset.union_empty]
      exact ⟨le_max_left _ _, fun hp => by
        simpa using hstop hyp ∅ 0 hl3 (fun _ => Finset.card_empty.le) (fun _ => Nat.zero_le _) hp⟩
    · dsimp only at hl3 ⊢
      simp only [eval_bind, reads_bind]
      have hT : hyp.tree.leaves.length ≤ nQ + 2 →
          ((topUpQ K nh hyp s (List.ofFn d.2) X (ls s)).reads read).card
            ≤ J * (b * (Kappa.probeSifts (Lp + K) * (nQ + 1))) := fun hlv => by
        have := card_reads_topUpQ_le read K nh Lp nQ hyp
          (by have := mids_card hyp.tree; omega) s (List.ofFn d.2) X (ls s) fun B hB t => by
            obtain ⟨j, rfl⟩ := List.mem_ofFn.1 hB
            exact hd2 j t
        rwa [List.length_ofFn] at this
      generalize (topUpQ K nh hyp s (List.ofFn d.2) X (ls s)).reads read = RT at hT ⊢
      generalize (topUpQ K nh hyp s (List.ofFn d.2) X (ls s)).eval read = o
      rcases o with _ | ⟨_ | ⟨ℓ, X'⟩, rest⟩
      · dsimp only
        simp only [eval_pure, reads_pure, Finset.union_empty]
        exact ⟨le_max_left _ _, hstop hyp RT _ hl3 (fun h => le_refl _) hT⟩
      · dsimp only
        simp only [eval_pure, reads_pure, Finset.union_empty]
        exact ⟨le_max_left _ _, hstop hyp RT _ hl3 (fun h => le_refl _) hT⟩
      · dsimp only
        have hds := fun d' hd' => hd d' (List.mem_cons_of_mem d hd')
        refine ⟨(le_max_left _ _).trans
          (lazyEpochsQ_card read C K nh Lp nQ (e + 1) _ ds X' _ hds).1, fun hp => ?_⟩
        have ih1 := (lazyEpochsQ_card read C K nh Lp nQ (e + 1) _ ds X' _ hds).1.trans hp
        have ih2 := (lazyEpochsQ_card read C K nh Lp nQ (e + 1) _ ds X' _ hds).2 hp
        have := key hyp _ _ hl3 ((le_max_right _ _).trans ih1)
          (hT ((le_max_right _ _).trans ih1)) ih2
        refine (Finset.card_union_le _ _).trans
          ((Nat.add_le_add_left (Finset.card_union_le _ _) _).trans ?_)
        omega
    · dsimp only at hl3 ⊢
      simp only [eval_pure, reads_pure, Finset.union_empty]
      exact ⟨le_max_left _ _, fun hp => by
        simpa using hstop hyp ∅ 0 hl3 (fun _ => Finset.card_empty.le) (fun _ => Nat.zero_le _) hp⟩

end Zone

/-- Over probes at least `k` long whose first `k` letters avoid `CP`'s, the round from `CP`'s trie
reads only `LazyKeyOff` strings at most `probes + 1 + T + 2` long. -/
theorem reads_lazyRoundQ_keyOff (C : Random.Level3Settings) (k nh : ℕ) {E J b : ℕ}
    (CP : Finset (FreeMonoid α)) (d : Fin E → Random.EpochDraws α C.probeBudget J b)
    (read : FreeMonoid α → ARU) (hk : C.walkStart = k)
    (hd : LazyDrawsAll (fun x => k ≤ x.toList.length
      ∧ Avoids k ↑(CP.image fun p => prefixOf p k) x) d) :
    ∀ z ∈ (Random.lazyRoundQ C k nh (Random.prefixTrie k CP) d).reads read,
      LazyKeyOff k CP z ∧ z.toList.length ≤ lazyProbesLen d + 1 + C.probeBudget + 2 := by
  subst hk
  have hlen := lazyDrawsAll_len d
  intro z hz
  have h := lazyRoundQ_zone read C _ nh _ d (Φ := fun x => (C.walkStart ≤ x.toList.length
    ∧ Avoids C.walkStart ↑(CP.image fun p => prefixOf p C.walkStart) x)
    ∧ x.toList.length ≤ lazyProbesLen d)
    (fun e => ⟨fun i => ⟨(hd e).1 i, (hlen e).1 i⟩, fun j t => ⟨(hd e).2 j t, (hlen e).2 j t⟩⟩)
    z hz
  have h2 := length_of_padZone (fun x hx => hx.2) h
  exact ⟨keyOff_of_padZone (fun x hx => hx.1) h, by omega⟩

/-- With every epoch's tree at most `nQ + 2` leaves, a round over probes at most `Lp` long asks at
most `lazyReadCap` strings. -/
theorem card_reads_lazyRoundQ_le (C : Random.Level3Settings) (k nh : ℕ) {E J b : ℕ}
    (X₀ : Finset (FreeMonoid α)) (d : Fin E → Random.EpochDraws α C.probeBudget J b)
    (read : FreeMonoid α → ARU) (hk : C.walkStart = k) (nQ Lp : ℕ)
    (hL : LazyDrawsAll (fun x => x.toList.length ≤ Lp) d)
    (hpk : ((Random.lazyRoundQ C k nh X₀ d).eval read).peak ≤ nQ + 2) :
    ((Random.lazyRoundQ C k nh X₀ d).reads read).card
      ≤ lazyReadCap C E J b Lp nQ (Fintype.card α) := by
  subst hk
  have h := (lazyEpochsQ_card read C C.walkStart nh Lp nQ 0 0 (List.ofFn d) X₀ (fun _ => [])
    fun d' hd' => by
      obtain ⟨e, rfl⟩ := List.mem_ofFn.1 hd'
      exact hL e).2 hpk
  rw [List.length_ofFn] at h
  exact h

/-- Unless it fails, a round's listed strings are at most `probes + 1 + T + 2` long. -/
theorem length_lazy_list_le (C : Random.Level3Settings) (k nh : ℕ) {E J b : ℕ}
    (X₀ : Finset (FreeMonoid α)) (d : Fin E → Random.EpochDraws α C.probeBudget J b)
    (read : FreeMonoid α → ARU) (hnh : 0 < nh) (hX₀ : Random.TrieOK k X₀) :
    ∀ s, ∀ w ∈ ((Random.lazyRoundQ C k nh X₀ d).eval read).lists s,
      w.toList.length ≤ lazyProbesLen d + 1 + C.probeBudget + 2 := by
  intro s w hw
  obtain ⟨-, -, -, -, -, -, -, h⟩ := lazyRoundShape_holds read C k nh E J b X₀ d hnh hX₀
  have hz := lazyRoundQ_zone read C k nh X₀ d (lazyDrawsAll_len d) w (h s w hw).1
  have := length_of_padZone (fun x hx => hx) hz
  omega

end OrthoDFA

end
