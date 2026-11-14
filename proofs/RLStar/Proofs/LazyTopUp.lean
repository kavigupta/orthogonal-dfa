import RLStar.LowLevelClaims.LazyRound
import RLStar.Proofs.BlockMiss
import RLStar.Proofs.RandomHarvest
import RLStar.Proofs.QueryLevel3
import RLStar.Proofs.TrieMass
import RLStar.Proofs.TrieAgree
import Mathlib.Data.List.ReduceOption

/-!
# `TopUpCorrect`: the proof

A probe's harvest string, pads dropped, has the probe's walk start, which is at most `K`. So a
probe that harvests but grows neither the list nor the trie walks from `K` through the `K`-letter
prefix of a listed string, and where the harvest mass is at least `y/2` the probes that grow one
hold at least `y/4` of `D`. A block retracts only if all its probes miss them (`BlockMiss`, at each
block index). The blocks that grow one number at most `(K + 1)(nh − 1) < J`, so the blocks never
run out.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

namespace Random

variable {α : Type*} [Fintype α]

/-- The strings `x` walked from `X` harvests at the site, pads dropped. -/
private def harv (read : FreeMonoid α → ARU) (K : ℕ) (H : HypothesisDFA (Option α))
    (s : Level3Site) (X : Finset (FreeMonoid α)) (x : FreeMonoid α) : List (FreeMonoid α) :=
  (siteHarvest (padRead read) K H s (padIn K X x)).map unpad

open scoped Classical in
/-- The trie and list after the block `B`. -/
private def upd (read : FreeMonoid α → ARU) (K : ℕ) (H : HypothesisDFA (Option α))
    (s : Level3Site) {b : ℕ} (st : Finset (FreeMonoid α) × List (FreeMonoid α))
    (B : Fin b → FreeMonoid α) : Finset (FreeMonoid α) × List (FreeMonoid α) :=
  (st.1 ∪ (blockUpdate K st.1 st.2 ((List.ofFn B).flatMap (harv read K H s st.1))).2,
    (blockUpdate K st.1 st.2 ((List.ofFn B).flatMap (harv read K H s st.1))).1)

/-- The probes whose harvest grows the list or the trie. -/
private def prog (read : FreeMonoid α → ARU) (K : ℕ) (H : HypothesisDFA (Option α))
    (s : Level3Site) (X : Finset (FreeMonoid α)) (ℓ : List (FreeMonoid α)) :
    Set (FreeMonoid α) :=
  {x | ∃ w ∈ harv read K H s X x,
    w ∉ ℓ ∨ walkStartIn X w < K ∧ prefixOf w (walkStartIn X w) ∉ X}

open scoped Classical in
/-- The prefixes shorter than `K` of the strings of `ℓ`. -/
private def shortPrefixes (K : ℕ) (ℓ : List (FreeMonoid α)) : Finset (FreeMonoid α) :=
  ℓ.toFinset.biUnion fun w => (Finset.range K).image (prefixOf w)

omit [Fintype α] in
private theorem eval_blockQ (read : FreeMonoid α → ARU) (K : ℕ) (H : HypothesisDFA (Option α))
    (s : Level3Site) (X : Finset (FreeMonoid α)) :
    ∀ xs : List (FreeMonoid α), (blockQ K H s X xs).eval read = xs.flatMap (harv read K H s X)
  | [] => rfl
  | x :: xs => by
    simp only [blockQ, Query.eval_bind, Query.eval_map, Query.eval_subst, eval_siteHarvestQ,
      eval_blockQ read K H s X xs, List.flatMap_cons]
    rfl

open scoped Classical in
private theorem eval_topUpQ_cons (read : FreeMonoid α → ARU) (K nh : ℕ)
    (H : HypothesisDFA (Option α)) (s : Level3Site) {b : ℕ} (B : Fin b → FreeMonoid α)
    (Bs : List (Fin b → FreeMonoid α)) (X : Finset (FreeMonoid α)) (ℓ : List (FreeMonoid α)) :
    (topUpQ K nh H s (B :: Bs) X ℓ).eval read =
      if nh ≤ (blockUpdate K X ℓ ((List.ofFn B).flatMap (harv read K H s X))).1.length then
        some (.full (blockUpdate K X ℓ ((List.ofFn B).flatMap (harv read K H s X))).1, Bs)
      else if (blockUpdate K X ℓ ((List.ofFn B).flatMap (harv read K H s X))).1.length = ℓ.length
          ∧ (blockUpdate K X ℓ ((List.ofFn B).flatMap (harv read K H s X))).2 ⊆ X then
        some (.retract (blockUpdate K X ℓ ((List.ofFn B).flatMap (harv read K H s X))).1 X, Bs)
      else (topUpQ K nh H s Bs (upd read K H s (X, ℓ) B).1 (upd read K H s (X, ℓ) B).2).eval
        read := by
  rw [topUpQ, Query.eval_bind, eval_blockQ]
  dsimp only [upd]
  split_ifs <;> rfl

open scoped Classical in
private theorem blockUpdate_spec (K : ℕ) (X : Finset (FreeMonoid α)) :
    ∀ (ws ℓ : List (FreeMonoid α)),
      (∃ t, (blockUpdate K X ℓ ws).1 = ℓ ++ t)
      ∧ (∀ w ∈ ws, w ∈ (blockUpdate K X ℓ ws).1)
      ∧ (∀ u ∈ (blockUpdate K X ℓ ws).2, ∃ w ∈ ws,
          walkStartIn X w < K ∧ u = prefixOf w (walkStartIn X w))
      ∧ ∀ w ∈ ws, w ∈ ℓ → walkStartIn X w < K →
          prefixOf w (walkStartIn X w) ∈ (blockUpdate K X ℓ ws).2
  | [], ℓ => by simp [blockUpdate]
  | w :: ws, ℓ => by
    by_cases hw : w ∈ ℓ
    · obtain ⟨⟨t, ht⟩, hmem, hS, hin⟩ := blockUpdate_spec K X ws ℓ
      simp only [blockUpdate, if_pos hw]
      refine ⟨⟨t, ht⟩, fun v hv => ?_, fun u hu => ?_, fun v hv hvℓ hvK => ?_⟩
      · rcases List.mem_cons.1 hv with rfl | hv
        · rw [ht]
          exact List.mem_append_left _ hw
        · exact hmem v hv
      · split_ifs at hu with hK
        · rcases Finset.mem_insert.1 hu with rfl | hu
          · exact ⟨w, List.mem_cons_self, hK, rfl⟩
          · obtain ⟨v, hv, hvK, rfl⟩ := hS u hu
            exact ⟨v, List.mem_cons_of_mem _ hv, hvK, rfl⟩
        · obtain ⟨v, hv, hvK, rfl⟩ := hS u hu
          exact ⟨v, List.mem_cons_of_mem _ hv, hvK, rfl⟩
      · rcases List.mem_cons.1 hv with rfl | hv
        · rw [if_pos hvK]
          exact Finset.mem_insert_self _ _
        · split_ifs
          · exact Finset.mem_insert_of_mem (hin v hv hvℓ hvK)
          · exact hin v hv hvℓ hvK
    · obtain ⟨⟨t, ht⟩, hmem, hS, hin⟩ := blockUpdate_spec K X ws (ℓ ++ [w])
      simp only [blockUpdate, if_neg hw]
      refine ⟨⟨w :: t, by rw [ht]; simp⟩, fun v hv => ?_, fun u hu => ?_, fun v hv hvℓ hvK => ?_⟩
      · rcases List.mem_cons.1 hv with rfl | hv
        · rw [ht]
          simp
        · exact hmem v hv
      · obtain ⟨v, hv, hvK, rfl⟩ := hS u hu
        exact ⟨v, List.mem_cons_of_mem _ hv, hvK, rfl⟩
      · rcases List.mem_cons.1 hv with rfl | hv
        · exact absurd hvℓ hw
        · exact hin v hv (List.mem_append_left _ hvℓ) hvK

private theorem mem_shortPrefixes {K i : ℕ} {ℓ : List (FreeMonoid α)} {w : FreeMonoid α}
    (hw : w ∈ ℓ) (hi : i < K) : prefixOf w i ∈ shortPrefixes K ℓ :=
  Finset.mem_biUnion.2
    ⟨w, List.mem_toFinset.2 hw, Finset.mem_image.2 ⟨i, Finset.mem_range.2 hi, rfl⟩⟩

private theorem card_shortPrefixes (K : ℕ) (ℓ : List (FreeMonoid α)) :
    (shortPrefixes K ℓ).card ≤ ℓ.length * K := by
  classical
  exact (Finset.card_biUnion_le_card_mul _ _ K fun _ _ =>
    Finset.card_image_le.trans (Finset.card_range K).le).trans
      (Nat.mul_le_mul_right K (List.toFinset_card_le ℓ))

/-! ## The harvest string of a probe -/

omit [Fintype α] in
private theorem unpad_prefixOf_padIn (K : ℕ) (X : Finset (FreeMonoid α)) (x : FreeMonoid α)
    (p : ℕ) :
    (unpad (prefixOf (padIn K X x) p)).toList = x.toList.take (p - (K - walkStartIn X x)) := by
  simp only [unpad, prefixOf, padIn, FreeMonoid.toList_ofList, List.take_append,
    List.length_replicate, List.take_replicate, List.reduceOption_append,
    List.reduceOption_replicate_none, List.nil_append, ← List.map_take]
  simp [List.reduceOption]

omit [Fintype α] in
/-- A harvest string, pads dropped, walks from the probe's walk start, at most `K`. -/
private theorem harv_shape [Finite α] (read : FreeMonoid α → ARU) (K : ℕ)
    (H : HypothesisDFA (Option α)) (s : Level3Site) {X : Finset (FreeMonoid α)}
    (hX : ∀ u ∈ X, u.toList.length < K) {x w : FreeMonoid α} (hx : K ≤ x.toList.length)
    (hw : w ∈ harv read K H s X x) :
    walkStartIn X x ≤ K ∧ walkStartIn X w = walkStartIn X x
      ∧ prefixOf w (walkStartIn X x) = prefixOf x (walkStartIn X x)
      ∧ prefixOf x (walkStartIn X x) ∉ X := by
  have := Fintype.ofFinite α
  obtain ⟨hle, hin, hout⟩ := walkStartIn_spec X x
  have hkK : walkStartIn X x ≤ K := by
    by_contra h
    have := hX _ (hin K (by omega))
    rw [length_prefixOf'] at this
    omega
  have hnot : prefixOf x (walkStartIn X x) ∉ X := by
    rcases lt_or_eq_of_le hkK with h | h
    · exact hout (by omega)
    · intro hm
      have := hX _ hm
      rw [length_prefixOf'] at this
      omega
  obtain ⟨z, hz, rfl⟩ := List.mem_map.1 hw
  have hpot := (siteHarvest_pot (padRead read) K H s (padIn K X x) z hz).1
  unfold pot at hpot
  obtain ⟨p, hp, hzp⟩ := Finset.mem_biUnion.1 hpot
  obtain ⟨m, -, rfl⟩ := Finset.mem_image.1 hzp
  have hpK : K ≤ p := (Finset.mem_Icc.1 hp).1
  have hwl : (unpad (prefixOf (padIn K X x) p * m)).toList
      = x.toList.take (p - (K - walkStartIn X x)) ++ (unpad m).toList := by
    rw [← unpad_prefixOf_padIn K X x p]
    simp [unpad, List.reduceOption_append]
  have hpre : ∀ j ≤ walkStartIn X x,
      prefixOf (unpad (prefixOf (padIn K X x) p * m)) j = prefixOf x j := by
    intro j hj
    change FreeMonoid.ofList (List.take j (unpad (prefixOf (padIn K X x) p * m)).toList)
      = FreeMonoid.ofList (x.toList.take j)
    rw [hwl, List.take_append_of_le_length (by rw [List.length_take]; omega), List.take_take]
    congr 2
    omega
  have hwlen : walkStartIn X x ≤ (unpad (prefixOf (padIn K X x) p * m)).toList.length := by
    rw [hwl, List.length_append, List.length_take]
    omega
  refine ⟨hkK, walkStartIn_eq hwlen (fun j hj => ?_) (fun _ => ?_), hpre _ le_rfl, hnot⟩
  · rw [hpre j hj.le]
    exact hin j hj
  · rw [hpre _ le_rfl]
    exact hnot

/-- Where the harvest mass is at least `y/2`, the probes that grow the list or the trie hold at
least `y/4`. -/
private theorem mass_prog (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (K nh : ℕ) (H : HypothesisDFA (Option α)) (s : Level3Site)
    {y p₁ : ℝ} {X : Finset (FreeMonoid α)} {ℓ : List (FreeMonoid α)}
    (hX : ∀ u ∈ X, u.toList.length < K) (hℓ : ℓ.length < nh)
    (hlong : ∀ᵐ x ∂D, K ≤ x.toList.length) (hp : ∀ u, D.real {x | prefixOf x K = u} ≤ p₁)
    (hy : 8 * nh * p₁ ≤ y) (hm : y / 2 ≤ harvestMass read D K X H s) :
    y / 4 ≤ D.real (prog read K H s X ℓ) := by
  classical
  have hp0 : 0 ≤ p₁ := le_trans measureReal_nonneg (hp 1)
  have hsub : {x | siteHarvest (padRead read) K H s (padIn K X x) ≠ []} ⊆
      prog read K H s X ℓ ∪ {x | ¬ K ≤ x.toList.length} ∪
        ⋃ w ∈ ℓ.toFinset, {x | prefixOf x K = prefixOf w K} := by
    intro x hx
    by_cases hxK : K ≤ x.toList.length
    swap
    · exact Or.inl (Or.inr hxK)
    by_cases hxA : x ∈ prog read K H s X ℓ
    · exact Or.inl (Or.inl hxA)
    right
    obtain ⟨z, hz⟩ := List.exists_mem_of_ne_nil _ hx
    have hw : unpad z ∈ harv read K H s X x := List.mem_map_of_mem hz
    obtain ⟨hkK, hws, hpre, hnot⟩ := harv_shape read K H s hX hxK hw
    have hwℓ : unpad z ∈ ℓ := by
      by_contra h
      exact hxA ⟨unpad z, hw, Or.inl h⟩
    rcases lt_or_eq_of_le hkK with hlt | heq
    · exact absurd ⟨unpad z, hw, Or.inr ⟨by rw [hws]; exact hlt, by rw [hws, hpre]; exact hnot⟩⟩
        hxA
    · simp only [Set.mem_iUnion, Set.mem_ofPred_eq, List.mem_toFinset]
      exact ⟨unpad z, hwℓ, by rw [← heq, hpre]⟩
  have hnull : D.real {x | ¬ K ≤ x.toList.length} = 0 := by
    rw [measureReal_def, ae_iff.1 hlong, ENNReal.toReal_zero]
  have hkeys : D.real (⋃ w ∈ ℓ.toFinset, {x | prefixOf x K = prefixOf w K}) ≤ nh * p₁ := by
    refine (measureReal_biUnion_finset_le _ _).trans ?_
    refine (Finset.sum_le_sum fun w _ => hp (prefixOf w K)).trans ?_
    rw [Finset.sum_const, nsmul_eq_mul]
    exact mul_le_mul_of_nonneg_right
      (by exact_mod_cast (List.toFinset_card_le ℓ).trans hℓ.le) hp0
  have h1 := measureReal_mono (μ := D) hsub
  have h2 := measureReal_union_le (μ := D) (prog read K H s X ℓ ∪ {x | ¬ K ≤ x.toList.length})
    (⋃ w ∈ ℓ.toFinset, {x | prefixOf x K = prefixOf w K})
  have h3 := measureReal_union_le (μ := D) (prog read K H s X ℓ) {x | ¬ K ≤ x.toList.length}
  unfold harvestMass at hm
  have hy0 : 0 ≤ y := le_trans (by positivity) hy
  linarith

/-! ## The blocks -/

private theorem upd_short (read : FreeMonoid α → ARU) (K : ℕ) (H : HypothesisDFA (Option α))
    (s : Level3Site) {b : ℕ} :
    ∀ (Bs : List (Fin b → FreeMonoid α)) (st : Finset (FreeMonoid α) × List (FreeMonoid α)),
      (∀ u ∈ st.1, u.toList.length < K) →
      ∀ u ∈ (Bs.foldl (upd read K H s) st).1, u.toList.length < K
  | [], _, h => h
  | B :: Bs, st, h => by
    refine upd_short read K H s Bs _ fun u hu => ?_
    rcases Finset.mem_union.1 hu with hu | hu
    · exact h u hu
    · obtain ⟨w, -, hwK, rfl⟩ := (blockUpdate_spec K st.1 _ st.2).2.2.1 u hu
      rw [length_prefixOf']
      omega

private theorem retract_block (read : FreeMonoid α → ARU) (K nh : ℕ)
    (H : HypothesisDFA (Option α)) (s : Level3Site) {b : ℕ} :
    ∀ (Bs : List (Fin b → FreeMonoid α)) (st : Finset (FreeMonoid α) × List (FreeMonoid α))
      (ℓ₁ : List (FreeMonoid α)) (X₁ : Finset (FreeMonoid α))
      (Bs₁ : List (Fin b → FreeMonoid α)),
      st.2.length < nh → (topUpQ K nh H s Bs st.1 st.2).eval read = some (.retract ℓ₁ X₁, Bs₁) →
      ∃ j, ∃ hj : j < Bs.length, ((Bs.take j).foldl (upd read K H s) st).1 = X₁
        ∧ ((Bs.take j).foldl (upd read K H s) st).2.length < nh
        ∧ ∀ i, Bs[j] i ∉ prog read K H s ((Bs.take j).foldl (upd read K H s) st).1
            ((Bs.take j).foldl (upd read K H s) st).2
  | [], _, _, _, _, _, h => by simp [topUpQ] at h
  | B :: Bs, (X, ℓ), ℓ₁, X₁, Bs₁, hℓ, h => by
    rw [eval_topUpQ_cons] at h
    dsimp only at h
    split_ifs at h with hfull hret
    · simp at h
    · simp only [Option.some.injEq, Prod.mk.injEq, TopUpEnd.retract.injEq] at h
      refine ⟨0, by simp, h.1.2, hℓ, fun i hi => ?_⟩
      simp only [List.take_zero, List.foldl_nil, List.getElem_cons_zero] at hi
      obtain ⟨w, hw, hprog⟩ := hi
      obtain ⟨⟨t, ht⟩, hmem, -, hin⟩ :=
        blockUpdate_spec K X ((List.ofFn B).flatMap (harv read K H s X)) ℓ
      have hwws : w ∈ (List.ofFn B).flatMap (harv read K H s X) :=
        List.mem_flatMap.2 ⟨B i, List.mem_ofFn.2 ⟨i, rfl⟩, hw⟩
      have ht0 : t = [] := by
        have h1 := congrArg List.length ht
        have h2 := hret.1
        rw [List.length_append] at h1
        exact List.eq_nil_of_length_eq_zero (by omega)
      have hwℓ : w ∈ ℓ := by
        have := hmem w hwws
        rwa [ht, ht0, List.append_nil] at this
      rcases hprog with h1 | ⟨h2, h3⟩
      · exact h1 hwℓ
      · exact h3 (hret.2 (hin w hwws hwℓ h2))
    · obtain ⟨j, hj, h1, h2, h3⟩ :=
        retract_block read K nh H s Bs (upd read K H s (X, ℓ) B) ℓ₁ X₁ Bs₁ (not_le.1 hfull) h
      refine ⟨j + 1, by simpa using hj, ?_⟩
      simp only [List.take_succ_cons, List.foldl_cons, List.getElem_cons_succ]
      exact ⟨h1, h2, h3⟩

private theorem none_bound (read : FreeMonoid α → ARU) (K nh : ℕ)
    (H : HypothesisDFA (Option α)) (s : Level3Site) {b : ℕ} :
    ∀ (Bs : List (Fin b → FreeMonoid α)) (X : Finset (FreeMonoid α)) (ℓ : List (FreeMonoid α)),
      ℓ.length < nh → (topUpQ K nh H s Bs X ℓ).eval read = none →
      ∃ (X' : Finset (FreeMonoid α)) (ℓ' : List (FreeMonoid α)), ℓ'.length < nh
        ∧ (∀ w ∈ ℓ, w ∈ ℓ') ∧ X' ⊆ X ∪ shortPrefixes K ℓ'
        ∧ Bs.length + X.card + ℓ.length ≤ X'.card + ℓ'.length
  | [], X, ℓ, hℓ, _ => ⟨X, ℓ, hℓ, fun _ h => h, Finset.subset_union_left, by simp⟩
  | B :: Bs, X, ℓ, hℓ, h => by
    rw [eval_topUpQ_cons] at h
    split_ifs at h with hfull hret
    obtain ⟨X', ℓ', hlt, hsub, hX', hc⟩ := none_bound read K nh H s Bs _ _ (not_le.1 hfull) h
    obtain ⟨⟨t, ht⟩, hmem, hS, -⟩ :=
      blockUpdate_spec K X ((List.ofFn B).flatMap (harv read K H s X)) ℓ
    simp only [upd] at hsub hX' hc
    set r := blockUpdate K X ℓ ((List.ofFn B).flatMap (harv read K H s X))
    refine ⟨X', ℓ', hlt, fun w hw => hsub w (by rw [ht]; exact List.mem_append_left _ hw), ?_, ?_⟩
    · intro u hu
      rcases Finset.mem_union.1 (hX' hu) with hu | hu
      · rcases Finset.mem_union.1 hu with hu | hu
        · exact Finset.mem_union_left _ hu
        · obtain ⟨w, hw, hwK, rfl⟩ := hS u hu
          exact Finset.mem_union_right _ (mem_shortPrefixes (hsub w (hmem w hw)) hwK)
      · exact Finset.mem_union_right _ hu
    · have hlen : r.1.length = ℓ.length + t.length := by rw [ht, List.length_append]
      have hcard : X.card ≤ (X ∪ r.2).card := Finset.card_le_card Finset.subset_union_left
      have hgrow : X.card + ℓ.length + 1 ≤ (X ∪ r.2).card + r.1.length := by
        rcases Nat.eq_zero_or_pos t.length with h0 | h0
        · have hns : ¬ r.2 ⊆ X := fun h' => hret ⟨by omega, h'⟩
          have : X.card < (X ∪ r.2).card := Finset.card_lt_card
            (Finset.ssubset_iff_subset_ne.2 ⟨Finset.subset_union_left,
              fun he => hns (he ▸ Finset.subset_union_right)⟩)
          omega
        · omega
      simp only [List.length_cons]
      omega

private theorem eval_topUpQ_ne_none (read : FreeMonoid α → ARU) (K nh : ℕ)
    (H : HypothesisDFA (Option α)) (s : Level3Site) {b : ℕ} {Bs : List (Fin b → FreeMonoid α)}
    {X : Finset (FreeMonoid α)} {ℓ : List (FreeMonoid α)} (hℓ : ℓ.length < nh)
    (hJ : nh * (K + 1) + 1 ≤ Bs.length) : (topUpQ K nh H s Bs X ℓ).eval read ≠ none := by
  intro h
  obtain ⟨X', ℓ', hlt, -, hX', hc⟩ := none_bound read K nh H s Bs X ℓ hℓ h
  have h1 := (Finset.card_le_card hX').trans (Finset.card_union_le _ _)
  have h2 := card_shortPrefixes K ℓ'
  have h3 := Nat.mul_le_mul_right (K + 1) hlt
  nlinarith

end Random

open Random

theorem topUpCorrect_holds : TopUpCorrect := by
  intro α _ _ read H s D _ K nh J b y p₁ X ℓ hX _ hℓ hlong hp hy hy1 hJ
  obtain ⟨n, rfl⟩ : ∃ n, J = n + 1 := ⟨J - 1, by omega⟩
  set μ : Measure (Fin b → FreeMonoid α) := Measure.pi fun _ : Fin b => D
  let st : List (Fin b → FreeMonoid α) → Finset (FreeMonoid α) × List (FreeMonoid α) :=
    fun Bs => Bs.foldl (upd read K H s) (X, ℓ)
  let E : Fin (n + 1) → Set (Fin (n + 1) → Fin b → FreeMonoid α) := fun j =>
    {B | y / 2 ≤ harvestMass read D K (st ((List.ofFn B).take j)).1 H s
      ∧ (st ((List.ofFn B).take j)).2.length < nh
      ∧ ∀ i, B j i ∉ prog read K H s (st ((List.ofFn B).take j)).1
          (st ((List.ofFn B).take j)).2}
  have hE : ∀ j, (Measure.pi fun _ : Fin (n + 1) => μ).real (E j) ≤ (1 - y / 4) ^ b := by
    intro j
    let past : (Fin n → Fin b → FreeMonoid α) → List (Fin b → FreeMonoid α) :=
      fun R => List.ofFn fun i : Fin j => R ⟨i, by omega⟩
    let A : (Fin n → Fin b → FreeMonoid α) → Set (FreeMonoid α) :=
      fun R => prog read K H s (st (past R)).1 (st (past R)).2
    let F : Set ((Fin b → FreeMonoid α) × (Fin n → Fin b → FreeMonoid α)) :=
      {wR | y / 2 ≤ harvestMass read D K (st (past wR.2)).1 H s
        ∧ (st (past wR.2)).2.length < nh ∧ ∀ i, wR.1 i ∉ A wR.2}
    have hpast : ∀ B : Fin (n + 1) → Fin b → FreeMonoid α,
        (List.ofFn B).take j = past (Fin.removeNth j B) := by
      intro B
      apply List.ext_getElem
      · simp [past]
      · intro i h1 h2
        simp only [List.getElem_take, List.getElem_ofFn, past, Fin.removeNth]
        congr 1
        rw [Fin.succAbove_of_castSucc_lt]
        · rfl
        · simp only [List.length_take, List.length_ofFn] at h1
          exact Fin.lt_def.2 (by simp; omega)
    have hEF : E j = MeasurableEquiv.piFinSuccAbove (fun _ => Fin b → FreeMonoid α) j ⁻¹' F := by
      ext B
      simp only [E, F, A, Set.mem_preimage, MeasurableEquiv.piFinSuccAbove_apply,
        Fin.insertNthEquiv_symm_apply, Set.mem_ofPred_eq, hpast]
    have he : MeasurePreserving (MeasurableEquiv.piFinSuccAbove (fun _ => Fin b → FreeMonoid α) j)
        (Measure.pi fun _ => μ) (μ.prod (Measure.pi fun _ : Fin n => μ)) :=
      measurePreserving_piFinSuccAbove (fun _ => μ) j
    have hF : MeasurableSet F := (Set.to_countable F).measurableSet
    rw [hEF, he.measureReal_preimage hF.nullMeasurableSet,
      ← (Measure.measurePreserving_swap (μ := Measure.pi fun _ : Fin n => μ)
        (ν := μ)).measureReal_preimage hF.nullMeasurableSet]
    have hsub : Prod.swap ⁻¹' F ⊆ {pw : (Fin n → Fin b → FreeMonoid α) × (Fin b → FreeMonoid α) |
        y / 4 ≤ D.real (A pw.1) ∧ ∀ i, pw.2 i ∉ A pw.1} := fun pw hpw =>
      ⟨mass_prog read D K nh H s (upd_short read K H s _ (X, ℓ) hX.1) hpw.2.1 hlong hp hy hpw.1,
        hpw.2.2⟩
    exact (measureReal_mono hsub).trans (blockMiss_holds (Measure.pi fun _ : Fin n => μ) D b
      (y / 4) A (fun _ => (Set.to_countable _).measurableSet) (by linarith))
  refine (measureReal_mono (s₂ := ⋃ j, E j) fun B hB => ?_).trans
    ((measureReal_iUnion_fintype_le _).trans ((Finset.sum_le_sum fun j _ => hE j).trans ?_))
  · rcases hres : (topUpQ K nh H s (List.ofFn B) X ℓ).eval read with _ | ⟨_ | ⟨ℓ₁, X₁⟩, Bs₁⟩
    · exact absurd hres (eval_topUpQ_ne_none read K nh H s hℓ (by simpa using hJ))
    · simp only [Set.mem_ofPred_eq, hres] at hB
    · simp only [Set.mem_ofPred_eq, hres] at hB
      obtain ⟨j, hj, h1, h2, h3⟩ :=
        retract_block read K nh H s (List.ofFn B) (X, ℓ) ℓ₁ X₁ Bs₁ hℓ hres
      simp only [List.length_ofFn] at hj
      refine Set.mem_iUnion.2 ⟨⟨j, hj⟩, ?_, h2, fun i => ?_⟩
      · exact h1 ▸ hB
      · simpa using h3 i
  · rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end OrthoDFA
