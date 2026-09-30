import OrthoDFA.Proofs.Clusterer

/-!
# `identify_cluster_around` is a `Clusterer`

`identifyCluster` meets `Clusterer`'s conditions, so the proof, which is carried out for any
`Clusterer`, covers the clustering the claim is stated for.
-/

namespace OrthoDFA

section nearest

variable {S : Type*}

open scoped Classical

variable (ℓ : S → ℝ) (ord : S → ℕ) (C : Finset S)

/-- How many candidates sort before `v`. -/
noncomputable def sortRank (v : S) : ℕ :=
  (C.filter (fun u => toLex (ℓ u, ord u) < toLex (ℓ v, ord v))).card

lemma nearest_eq (k : ℕ) : nearest ℓ ord C k = C.filter (fun v => sortRank ℓ ord C v < k) := rfl

lemma nearest_subset (k : ℕ) : nearest ℓ ord C k ⊆ C := Finset.filter_subset _ _

variable {ℓ ord C}

lemma key_injOn (hord : Set.InjOn ord ↑C) :
    Set.InjOn (fun v => toLex (ℓ v, ord v)) ↑C := by
  intro u hu v hv h
  have h' : (ℓ u, ord u) = (ℓ v, ord v) := toLex.injective h
  exact hord hu hv (Prod.ext_iff.1 h').2

lemma sortRank_lt_card {v : S} (hv : v ∈ C) : sortRank ℓ ord C v < C.card := by
  refine Finset.card_lt_card (Finset.ssubset_iff_subset_ne.2 ⟨Finset.filter_subset _ _, ?_⟩)
  intro h
  have : v ∈ C.filter (fun u => toLex (ℓ u, ord u) < toLex (ℓ v, ord v)) := by rw [h]; exact hv
  exact lt_irrefl _ (Finset.mem_filter.1 this).2

lemma sortRank_strictMono {u v : S} (hu : u ∈ C)
    (h : toLex (ℓ u, ord u) < toLex (ℓ v, ord v)) : sortRank ℓ ord C u < sortRank ℓ ord C v := by
  refine Finset.card_lt_card (Finset.ssubset_iff_subset_ne.2 ⟨fun w hw => ?_, fun heq => ?_⟩)
  · obtain ⟨hwC, hw⟩ := Finset.mem_filter.1 hw
    exact Finset.mem_filter.2 ⟨hwC, hw.trans h⟩
  · have : u ∈ C.filter (fun w => toLex (ℓ w, ord w) < toLex (ℓ u, ord u)) :=
      heq ▸ Finset.mem_filter.2 ⟨hu, h⟩
    exact lt_irrefl _ (Finset.mem_filter.1 this).2

lemma sortRank_injOn (hord : Set.InjOn ord ↑C) : Set.InjOn (sortRank ℓ ord C) ↑C := by
  intro u hu v hv h
  by_contra hne
  have hk : toLex (ℓ u, ord u) ≠ toLex (ℓ v, ord v) := fun he => hne (key_injOn hord hu hv he)
  rcases lt_or_gt_of_ne hk with hlt | hlt
  · exact absurd h (ne_of_lt (sortRank_strictMono hu hlt))
  · exact absurd h (ne_of_gt (sortRank_strictMono hv hlt))

lemma image_sortRank (hord : Set.InjOn ord ↑C) :
    C.image (sortRank ℓ ord C) = Finset.range C.card := by
  refine Finset.eq_of_subset_of_card_le (fun r hr => ?_) ?_
  · obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hr
    exact Finset.mem_range.2 (sortRank_lt_card hv)
  · rw [Finset.card_range, Finset.card_image_of_injOn (sortRank_injOn hord)]

/-- With the pool order injective, the stable top `k` has exactly `min k #C` members. -/
lemma card_nearest (hord : Set.InjOn ord ↑C) (k : ℕ) :
    (nearest ℓ ord C k).card = min k C.card := by
  rw [nearest_eq]
  have hinj : Set.InjOn (sortRank ℓ ord C) ↑(C.filter (fun v => sortRank ℓ ord C v < k)) :=
    (sortRank_injOn hord).mono (fun v hv => (Finset.mem_filter.1 hv).1)
  rw [← Finset.card_image_of_injOn hinj]
  have himg : (C.filter (fun v => sortRank ℓ ord C v < k)).image (sortRank ℓ ord C)
      = (Finset.range C.card).filter (fun r => r < k) := by
    rw [← image_sortRank hord, Finset.filter_image]
  rw [himg]
  have : (Finset.range C.card).filter (fun r => r < k) = Finset.range (min k C.card) := by
    ext r; simp only [Finset.mem_filter, Finset.mem_range, lt_min_iff]; tauto
  rw [this, Finset.card_range]

end nearest

variable {S : Type*} [Stringlike S]

open scoped Classical in
lemma lastOf_mem {ℓ : S → ℝ} {ord : S → ℕ} {N : Finset S} (hN : N.Nonempty) :
    lastOf ℓ ord N ∈ N := by
  rw [lastOf, dif_pos hN]
  exact (N.exists_max_image (fun v => toLex (ℓ v, ord v)) hN).choose_spec.1

section loop

open scoped Classical

variable (reads : S → Prop) (w : S → ℝ) (ord : S → ℕ) (bnd : ℝ) (P C : Finset S) (k : ℕ)

/-- The loss `identifyStep` ranks by, at cluster `F`. -/
noncomputable def stepLoss (F : Finset S) (v : S) : ℝ :=
  if v ∈ C then weightedLoss reads w bnd P F v else 0

/-- The cluster `identifyStep` moves to from `F`, if it moves. -/
noncomputable def stepNext (F : Finset S) : Finset S :=
  if (1 : S) ∈ nearest (stepLoss reads w bnd P C F) ord C k then
    nearest (stepLoss reads w bnd P C F) ord C k
  else insert 1 ((nearest (stepLoss reads w bnd P C F) ord C k).erase
    (lastOf (stepLoss reads w bnd P C F) ord (nearest (stepLoss reads w bnd P C F) ord C k)))

lemma identifyStep_eq (st : Finset S × WithTop ℝ × Bool) :
    identifyStep reads w ord bnd P C k st =
      if st.2.2 then st
      else if stepLoss reads w bnd P C st.1 (lastOf (stepLoss reads w bnd P C st.1) ord
          (nearest (stepLoss reads w bnd P C st.1) ord C k))
          < stepLoss reads w bnd P C st.1 1 then (st.1, st.2.1, true)
      else if st.2.1 ≤ ((∑ v ∈ stepNext reads w ord bnd P C k st.1,
          stepLoss reads w bnd P C st.1 v : ℝ) : WithTop ℝ) then (st.1, st.2.1, true)
      else (stepNext reads w ord bnd P C k st.1, ((∑ v ∈ stepNext reads w ord bnd P C k st.1,
          stepLoss reads w bnd P C st.1 v : ℝ) : WithTop ℝ), false) := rfl

/-- A step either keeps the cluster or moves to `stepNext`. -/
lemma identifyStep_fst (st : Finset S × WithTop ℝ × Bool) :
    (identifyStep reads w ord bnd P C k st).1 = st.1
      ∨ (identifyStep reads w ord bnd P C k st).1 = stepNext reads w ord bnd P C k st.1 := by
  rw [identifyStep_eq]
  split_ifs <;> simp

variable {reads w ord bnd P C k}

lemma stepNext_seed_subset (hone : (1 : S) ∈ C) (F : Finset S) :
    (1 : S) ∈ stepNext reads w ord bnd P C k F ∧ stepNext reads w ord bnd P C k F ⊆ C := by
  unfold stepNext
  split_ifs with h
  · exact ⟨h, nearest_subset _ _ _ _⟩
  · refine ⟨Finset.mem_insert_self _ _, Finset.insert_subset hone ?_⟩
    exact (Finset.erase_subset _ _).trans (nearest_subset _ _ _ _)

lemma stepNext_card_le (hord : Set.InjOn ord ↑C) (hk : 0 < k) (F : Finset S) :
    (stepNext reads w ord bnd P C k F).card ≤ k := by
  have hN := card_nearest (ℓ := stepLoss reads w bnd P C F) hord k
  unfold stepNext
  split_ifs with h
  · rw [hN]; exact min_le_left _ _
  · set N := nearest (stepLoss reads w bnd P C F) ord C k with hNdef
    rcases N.eq_empty_or_nonempty with he | hne
    · rw [he]; simp; omega
    · have hl := lastOf_mem (ℓ := stepLoss reads w bnd P C F) (ord := ord) hne
      calc (insert 1 (N.erase _)).card ≤ (N.erase _).card + 1 := Finset.card_insert_le _ _
        _ = N.card := Finset.card_erase_add_one hl
        _ ≤ k := by rw [hN]; exact min_le_left _ _

lemma stepNext_card (hord : Set.InjOn ord ↑C) (hone : (1 : S) ∈ C) (hk : 0 < k) (F : Finset S) :
    (stepNext reads w ord bnd P C k F).card = min k C.card := by
  have hN := card_nearest (ℓ := stepLoss reads w bnd P C F) hord k
  unfold stepNext
  split_ifs with h
  · exact hN
  · set N := nearest (stepLoss reads w bnd P C F) ord C k with hNdef
    have hne : N.Nonempty := by
      rw [← Finset.card_pos, hN]
      exact lt_min hk (Finset.card_pos.2 ⟨1, hone⟩)
    have hl := lastOf_mem (ℓ := stepLoss reads w bnd P C F) (ord := ord) hne
    have h1 : (1 : S) ∉ N.erase (lastOf (stepLoss reads w bnd P C F) ord N) :=
      fun h' => h (Finset.mem_of_mem_erase h')
    rw [Finset.card_insert_of_notMem h1, Finset.card_erase_add_one hl, hN]

/-- The seed's own loss against the seed alone is nothing: its column is the centre. -/
lemma stepLoss_seed_init (hone : (1 : S) ∈ C) (hb0 : 0 ≤ bnd) (hb1 : bnd < 1) :
    stepLoss reads w bnd P C {1} 1 = 0 := by
  rw [stepLoss, if_pos hone, weightedLoss]
  refine Finset.sum_eq_zero (fun p hp => ?_)
  exfalso
  refine (Finset.mem_filter.1 hp).2 ?_
  rw [mul_one, Finset.card_singleton, Nat.cast_one, mul_one]
  by_cases hr : reads p
  · have : ({(1 : S)} : Finset S).filter (fun u => reads (p * u)) = {1} := by
      ext u; simp only [Finset.mem_filter, Finset.mem_singleton]
      constructor
      · exact fun h => h.1
      · rintro rfl; exact ⟨rfl, by rwa [mul_one]⟩
    rw [this, Finset.card_singleton]
    exact ⟨fun _ => by exact_mod_cast hb1, fun _ => hr⟩
  · have : ({(1 : S)} : Finset S).filter (fun u => reads (p * u)) = ∅ := by
      ext u; simp only [Finset.mem_filter, Finset.mem_singleton, Finset.notMem_empty, iff_false]
      rintro ⟨rfl, h⟩; exact hr (by rwa [mul_one] at h)
    rw [this, Finset.card_empty, Nat.cast_zero]
    exact ⟨fun h => absurd h hr, fun h => absurd h (not_lt.2 hb0)⟩

lemma stepLoss_nonneg (hw : ∀ p, 0 ≤ w p) (F : Finset S) (v : S) :
    0 ≤ stepLoss reads w bnd P C F v := by
  unfold stepLoss
  split_ifs
  · exact Finset.sum_nonneg (fun p _ => hw p)
  · exact le_rfl

end loop

section iterate

open scoped Classical

variable {reads : S → Prop} {w : S → ℝ} {ord : S → ℕ} {bnd : ℝ} {P C : Finset S} {k : ℕ}

lemma iterate_seed_subset (hone : (1 : S) ∈ C) (n : ℕ) :
    (1 : S) ∈ ((identifyStep reads w ord bnd P C k)^[n] ({1}, ⊤, false)).1
      ∧ ((identifyStep reads w ord bnd P C k)^[n] ({1}, ⊤, false)).1 ⊆ C := by
  induction n with
  | zero => exact ⟨Finset.mem_singleton_self _, Finset.singleton_subset_iff.2 hone⟩
  | succ n ih =>
    rw [Function.iterate_succ_apply']
    rcases identifyStep_fst reads w ord bnd P C k _ with h | h <;> rw [h]
    · exact ih
    · exact stepNext_seed_subset hone _

lemma iterate_card_le (hord : Set.InjOn ord ↑C) (hk : 0 < k) (n : ℕ) :
    ((identifyStep reads w ord bnd P C k)^[n] ({1}, ⊤, false)).1.card ≤ k := by
  induction n with
  | zero => show ({1} : Finset S).card ≤ k; rw [Finset.card_singleton]; exact hk
  | succ n ih =>
    rw [Function.iterate_succ_apply']
    rcases identifyStep_fst reads w ord bnd P C k _ with h | h <;> rw [h]
    · exact ih
    · exact stepNext_card_le hord hk _

/-- The first pass always moves: the seed's loss against itself is nothing, and the loss it is
compared against starts infinite. -/
lemma identifyStep_init (hw : ∀ p, 0 ≤ w p) (hone : (1 : S) ∈ C) (hb0 : 0 ≤ bnd) (hb1 : bnd < 1) :
    (identifyStep reads w ord bnd P C k ({1}, ⊤, false)).1
      = stepNext reads w ord bnd P C k {1} := by
  rw [identifyStep_eq]
  have h0 := stepLoss_seed_init (reads := reads) (w := w) (P := P) hone hb0 hb1
  simp only [Bool.false_eq_true, if_false, h0]
  rw [if_neg (not_lt.2 (stepLoss_nonneg hw _ _)), if_neg (by simp)]

lemma iterate_card_eq (hw : ∀ p, 0 ≤ w p) (hord : Set.InjOn ord ↑C) (hone : (1 : S) ∈ C)
    (hb0 : 0 ≤ bnd) (hb1 : bnd < 1) (hk : 0 < k) (n : ℕ) :
    ((identifyStep reads w ord bnd P C k)^[n + 1] ({1}, ⊤, false)).1.card = min k C.card := by
  rw [Function.iterate_succ_apply]
  have hinit : (identifyStep reads w ord bnd P C k ({1}, ⊤, false)).1.card = min k C.card := by
    rw [identifyStep_init hw hone hb0 hb1]; exact stepNext_card hord hone hk _
  generalize identifyStep reads w ord bnd P C k ({1}, ⊤, false) = st at hinit ⊢
  induction n generalizing st with
  | zero => exact hinit
  | succ n ih =>
    rw [Function.iterate_succ_apply]
    refine ih _ ?_
    rcases identifyStep_fst reads w ord bnd P C k st with h | h <;> rw [h]
    · exact hinit
    · exact stepNext_card hord hone hk _

lemma weightedLoss_congr {reads' : S → Prop} {F : Finset S} {v : S} (hF : F ⊆ C) (hv : v ∈ C)
    (h : ∀ p ∈ P, ∀ u ∈ C, (reads (p * u) ↔ reads' (p * u))) :
    weightedLoss reads w bnd P F v = weightedLoss reads' w bnd P F v := by
  unfold weightedLoss
  refine Finset.sum_congr (Finset.filter_congr (fun p hp => ?_)) (fun _ _ => rfl)
  have hc : F.filter (fun u => reads (p * u)) = F.filter (fun u => reads' (p * u)) :=
    Finset.filter_congr (fun u hu => h p hp u (hF hu))
  rw [hc, h p hp v hv]

lemma identifyStep_congr {reads' : S → Prop} (h : ∀ p ∈ P, ∀ u ∈ C, (reads (p * u) ↔ reads' (p *
    u)))
    (st : Finset S × WithTop ℝ × Bool) (hst : st.1 ⊆ C) :
    identifyStep reads w ord bnd P C k st = identifyStep reads' w ord bnd P C k st := by
  have hL : stepLoss reads w bnd P C st.1 = stepLoss reads' w bnd P C st.1 := by
    funext v
    unfold stepLoss
    split_ifs with hv
    · exact weightedLoss_congr hst hv h
    · rfl
  have hN : stepNext reads w ord bnd P C k st.1 = stepNext reads' w ord bnd P C k st.1 := by
    unfold stepNext; rw [hL]
  rw [identifyStep_eq, identifyStep_eq, hL, hN]

lemma iterate_congr {reads' : S → Prop} (hone : (1 : S) ∈ C)
    (h : ∀ p ∈ P, ∀ u ∈ C, (reads (p * u) ↔ reads' (p * u))) (n : ℕ) :
    (identifyStep reads w ord bnd P C k)^[n] ({1}, ⊤, false)
      = (identifyStep reads' w ord bnd P C k)^[n] ({1}, ⊤, false) := by
  induction n with
  | zero => rw [Function.iterate_zero_apply, Function.iterate_zero_apply]
  | succ n ih =>
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply', ← ih]
    exact identifyStep_congr h _ (iterate_seed_subset hone n).2

lemma identifyCluster_congr {reads' : S → Prop} (hone : (1 : S) ∈ C)
    (h : ∀ p ∈ P, ∀ u ∈ C, (reads (p * u) ↔ reads' (p * u))) :
    identifyCluster reads w ord bnd P C k = identifyCluster reads' w ord bnd P C k := by
  unfold identifyCluster
  rw [iterate_congr hone h]

end iterate

/-- `identify_cluster_around` at a boundary in `[0, 1)`. -/
noncomputable def identifyClusterer (bnd : ℝ) (hb0 : 0 ≤ bnd) (hb1 : bnd < 1) : Clusterer S where
  pick := fun reads w ord P C k => identifyCluster reads w ord bnd P C k
  seed_mem := fun _ _ _ _ _ _ hone => (iterate_seed_subset hone _).1
  subset := fun _ _ _ _ _ _ hone => (iterate_seed_subset hone _).2
  card_le := fun _ _ _ _ _ _ hord hk => iterate_card_le hord hk _
  card_eq := fun reads w ord P C k hw hord hone hkC hk => by
    show ((identifyStep reads w ord bnd P C k)^[(2 ^ C.card + 1) ^ 2] ({1}, ⊤, false)).1.card = k
    obtain ⟨n, hn⟩ : ∃ n, (2 ^ C.card + 1) ^ 2 = n + 1 :=
      ⟨_, (Nat.succ_pred_eq_of_pos (by positivity)).symm⟩
    rw [hn, iterate_card_eq hw hord hone hb0 hb1 hk n, min_eq_left hkC]
  congr := fun _ _ _ _ _ _ _ hone h => identifyCluster_congr hone h

variable {Ω : Type*} [MeasurableSpace Ω] {J : Type*} [Fintype J]

omit [MeasurableSpace Ω] [Fintype J] in
lemma clusterAt_eq (bnd : ℝ) (hb0 : 0 ≤ bnd) (hb1 : bnd < 1) (mq : S → Ω → ℝ)
    (populations : Finset J) (x : Run Ω S J) (B : State) :
    clusterAt mq populations bnd x B = clusterBy (identifyClusterer bnd hb0 hb1) mq populations x B
        :=
  rfl

omit [MeasurableSpace Ω] [Fintype J] in
lemma familyAt_eq (bnd : ℝ) (hb0 : 0 ≤ bnd) (hb1 : bnd < 1) (mq : S → Ω → ℝ)
    (populations : Finset J) (x : Run Ω S J) (B : State) :
    familyAt mq populations bnd x B = familyBy (identifyClusterer bnd hb0 hb1) mq populations x B :=
  rfl

lemma ret_eq (bnd : ℝ) (hb0 : 0 ≤ bnd) (hb1 : bnd < 1) (mq : S → Ω → ℝ) (populations : Finset J)
    (uni : J) (indecisionLimit α : ℝ) (v n : ℕ) (B : State) :
    ret mq populations uni bnd indecisionLimit α v n B
      = retBy (identifyClusterer bnd hb0 hb1) mq populations uni indecisionLimit α v n B :=
  rfl

end OrthoDFA
