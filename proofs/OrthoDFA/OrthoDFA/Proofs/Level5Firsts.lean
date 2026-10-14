import OrthoDFA.Proofs.Level5Query
import OrthoDFA.Proofs.Level5Congr

/-!
# A probe's first reads

The edge tests count, per probe, the strings first read at positions charged to an edge, and those
of them read undecided. A probe's first reads are distinct strings on the routes of its positions'
sifts, so they number at most the positions times the midfixes, and its undecided ones at most one
a position, each where that position's sift stops.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

attribute [local instance] ARU.fintype

namespace QTree

variable {Λ ι V β γ : Type*} [DecidableEq ι]

theorem firsts_sub_asks (rd : ι → V) :
    ∀ (t : QTree Λ ι V β) (P : Finset ι) (q : Λ × ι), q ∈ t.firsts rd P → q ∈ t.asks rd
  | done _, _, _, h => by simp [firsts] at h
  | ask ℓ w next, P, q, h => by
    simp only [firsts] at h
    simp only [asks, List.mem_cons]
    split_ifs at h
    · exact .inr (firsts_sub_asks rd _ _ q h)
    · rcases List.mem_cons.1 h with h | h
      · exact .inl h
      · exact .inr (firsts_sub_asks rd _ _ q h)

theorem firsts_congr_asks {rd rd' : ι → V} :
    ∀ (t : QTree Λ ι V β) (P : Finset ι), (∀ q ∈ t.asks rd, rd q.2 = rd' q.2) →
      t.firsts rd P = t.firsts rd' P
  | done _, _, _ => rfl
  | ask ℓ w next, P, h => by
    have hw : rd w = rd' w := h (ℓ, w) (by simp [asks])
    have hn : ∀ q ∈ (next (rd w)).asks rd, rd q.2 = rd' q.2 := fun q hq =>
      h q (by simp only [asks, List.mem_cons]; exact .inr hq)
    simp only [firsts, ← hw]
    rw [firsts_congr_asks _ P hn, firsts_congr_asks _ (insert w P) hn]

theorem firsts_bind (rd : ι → V) (f : β → QTree Λ ι V γ) :
    ∀ (t : QTree Λ ι V β) (P : Finset ι), (t.bind f).firsts rd P
      = t.firsts rd P ++ (f (t.run rd)).firsts rd (P ∪ ((t.asks rd).map Prod.snd).toFinset)
  | done _, P => by simp [bind, firsts, asks, run]
  | ask ℓ w next, P => by
    simp only [bind, firsts, asks, run, List.map_cons, List.toFinset_cons]
    by_cases hw : w ∈ P
    · rw [if_pos hw, if_pos hw, firsts_bind rd f (next (rd w)) P]
      congr 2
      ext y; simp only [Finset.mem_union, Finset.mem_insert]
      constructor
      · rintro (h | h)
        · exact .inl h
        · exact .inr (.inr h)
      · rintro (h | rfl | h)
        · exact .inl h
        · exact .inl hw
        · exact .inr h
    · rw [if_neg hw, if_neg hw, firsts_bind rd f (next (rd w)) (insert w P), List.cons_append]
      congr 3
      ext y; simp only [Finset.mem_union, Finset.mem_insert]; tauto

end QTree

section Cut

omit [Fintype α] [DecidableEq α] in
theorem ofCut_cut (cut : FreeMonoid α → Option Bool) (z : FreeMonoid α) :
    (ofCut cut z).cut = cut z := by
  unfold ofCut; rcases cut z with _ | _ | _ <;> rfl

omit [Fintype α] [DecidableEq α] in
theorem cut_ofCut (cut : FreeMonoid α → Option Bool) : (fun z => (ofCut cut z).cut) = cut :=
  funext (ofCut_cut cut)

omit [Fintype α] [DecidableEq α] in
theorem ofCut_rd (rd : FreeMonoid α → ARU) : ofCut (fun z => (rd z).cut) = rd := by
  funext z; unfold ofCut; rcases h : rd z <;> simp [h, ARU.cut]

end Cut

section Label

variable (T : DTree α) (edges : EdgeTable α) (k : ℕ) (x : FreeMonoid α) (rd : FreeMonoid α → ARU)

/-- A read of the probe is labelled by the start, or by its position and that position's edge. -/
theorem asks_probeQuery_label (q : ReadLabel α × FreeMonoid α)
    (hq : q ∈ (probeQuery T edges k x).asks rd) :
    q.1 = (k, none) ∨ q.1.2 = posEdgeBy (fun z => (rd z).cut) T edges k x q.1.1 := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut with hcut
  unfold probeQuery at hq
  split_ifs at hq with hshort
  · simp [QTree.asks] at hq
  rw [QTree.asks_bind, List.mem_append, run_siftQuery] at hq
  rcases hq with hq | hq
  · exact .inl (mem_asks_siftQuery T rd hq).1
  rcases hk : T.siftCut cut (prefixOf x k) with p₀ | b <;> rw [hk] at hq
  swap
  · simp [QTree.asks] at hq
  right
  suffices h : ∃ i, q.1 = readLabel edges k x p₀ i by
    obtain ⟨i, hi⟩ := h
    rw [hi]
    simp only [readLabel]
    rw [posEdgeBy_of_inl hk]
  have hbr : ∀ (ps : List (List Bool)) (walkAt : ℕ → List Bool) (fuel lo hi : ℕ),
      q ∈ (bracketQuery (agreementQuery T edges k x p₀ walkAt) ps fuel lo hi).asks rd →
      ∃ i, q.1 = readLabel edges k x p₀ i := by
    intro ps walkAt fuel lo hi h
    obtain ⟨i, -, -, -, h'⟩ := asks_bracketQuery _ ps rd _ (run_agreementQuery T edges k x rd p₀
        walkAt) _ _ _ q h
    rw [agreementQuery, QTree.asks_map] at h'
    exact ⟨i, (mem_asks_siftQuery T rd h').1⟩
  simp only [] at hq
  rcases hf : follow edges p₀ (x.toList.drop k) with ps | ⟨s, c, i₀⟩ <;> rw [hf] at hq <;>
    simp only [] at hq
  · rw [QTree.asks_bind, List.mem_append, run_siftQuery] at hq
    rcases hq with hq | hq
    · exact ⟨_, (mem_asks_siftQuery T rd hq).1⟩
    rcases hx : T.siftCut cut (prefixOf x x.toList.length) with a' | b' <;> rw [hx] at hq <;>
      simp only [] at hq
    · split_ifs at hq
      · simp [QTree.asks] at hq
      · exact hbr _ _ _ _ _ hq
    · simp [QTree.asks] at hq
  · rw [QTree.asks_bind, List.mem_append, run_siftQuery] at hq
    rcases hq with hq | hq
    · exact ⟨_, (mem_asks_siftQuery T rd hq).1⟩
    rcases h1 : T.siftCut cut (prefixOf x (k + i₀ + 1)) with q₁ | b₁ <;> rw [h1] at hq <;>
      simp only [] at hq
    swap
    · simp [QTree.asks] at hq
    rw [QTree.asks_bind, List.mem_append, run_siftQuery] at hq
    rcases hq with hq | hq
    · exact ⟨_, (mem_asks_siftQuery T rd hq).1⟩
    rcases h2 : T.siftCut cut (prefixOf x (k + i₀)) with q₂ | b₂ <;> rw [h2] at hq <;>
      simp only [] at hq
    swap
    · simp [QTree.asks] at hq
    split_ifs at hq
    · simp [QTree.asks] at hq
    · exact hbr _ _ _ _ _ hq

end Label

section Count

theorem route_mid (cut : FreeMonoid α → Option Bool) :
    ∀ (t : DTree α) (u q : FreeMonoid α), q ∈ (t.route cut u).1 → ∃ m ∈ t.midfixes, q = u * m
  | .leaf, _, _, h => by simp [DTree.route] at h
  | .node m r a, u, q, h => by
    classical
    simp only [DTree.route] at h
    split at h
    · simp only [List.mem_singleton] at h
      exact ⟨m, by simp [DTree.midfixes], h⟩
    · simp only [List.mem_cons] at h
      rcases h with h | h
      · exact ⟨m, by simp [DTree.midfixes], h⟩
      · obtain ⟨m', hm', rfl⟩ := route_mid cut a u q h
        exact ⟨m', by simp [DTree.midfixes, hm'], rfl⟩
    · simp only [List.mem_cons] at h
      rcases h with h | h
      · exact ⟨m, by simp [DTree.midfixes], h⟩
      · obtain ⟨m', hm', rfl⟩ := route_mid cut r u q h
        exact ⟨m', by simp [DTree.midfixes, hm'], rfl⟩

/-- The strings any sift of a prefix of `x` from `k` on reads. -/
noncomputable def probeStrings (T : DTree α) (k : ℕ) (x : FreeMonoid α) : Finset (FreeMonoid α) :=
  (Finset.Icc k x.toList.length).biUnion fun j => T.midfixes.image (prefixOf x j * ·)

theorem probeStrings_prefix {T : DTree α} {k : ℕ} {x z : FreeMonoid α}
    (hz : z ∈ probeStrings T k x) : prefixOf z k = prefixOf x k := by
  obtain ⟨j, hj, hz⟩ := Finset.mem_biUnion.1 hz
  obtain ⟨m, -, rfl⟩ := Finset.mem_image.1 hz
  have hjk := (Finset.mem_Icc.1 hj).1
  have hjx := (Finset.mem_Icc.1 hj).2
  apply FreeMonoid.toList.injective
  simp only [prefixOf, FreeMonoid.toList_ofList, FreeMonoid.toList_mul]
  rw [List.take_append_of_le_length (by simp; omega), List.take_take, min_eq_left hjk]

theorem route_mem_probeStrings {cut : FreeMonoid α → Option Bool} {T : DTree α} {k : ℕ}
    {x : FreeMonoid α} (hkx : k ≤ x.toList.length) {i : ℕ} (hi : k ≤ i) {q : FreeMonoid α}
    (h : q ∈ (T.route cut (prefixOf x i)).1) : q ∈ probeStrings T k x := by
  obtain ⟨m, hm, rfl⟩ := route_mid cut T _ q h
  refine Finset.mem_biUnion.2 ⟨min i x.toList.length, Finset.mem_Icc.2 ⟨by omega, by omega⟩,
    Finset.mem_image.2 ⟨m, hm, ?_⟩⟩
  congr 1
  apply FreeMonoid.toList.injective
  simp only [prefixOf, FreeMonoid.toList_ofList]
  rcases le_total i x.toList.length with h | h
  · rw [min_eq_left h]
  · rw [min_eq_right h, List.take_of_length_le h, List.take_of_length_le le_rfl]

/-- The strings any sift of a prefix of `x` reads. -/
noncomputable def prefStrings (T : DTree α) (x : FreeMonoid α) : Finset (FreeMonoid α) :=
  (Finset.range (x.toList.length + 1)).biUnion fun j => T.midfixes.image (prefixOf x j * ·)

theorem prefStrings_card (T : DTree α) (x : FreeMonoid α) :
    (prefStrings T x).card ≤ (x.toList.length + 1) * T.midfixes.card := by
  unfold prefStrings
  refine Finset.card_biUnion_le.trans ?_
  calc ∑ j ∈ Finset.range (x.toList.length + 1), (T.midfixes.image (prefixOf x j * ·)).card
      ≤ ∑ _j ∈ Finset.range (x.toList.length + 1), T.midfixes.card :=
        Finset.sum_le_sum fun j _ => Finset.card_image_le
    _ = _ := by simp

theorem route_mem_prefStrings {cut : FreeMonoid α → Option Bool} {T : DTree α} {x : FreeMonoid α}
    {i : ℕ} {q : FreeMonoid α} (h : q ∈ (T.route cut (prefixOf x i)).1) :
    q ∈ prefStrings T x := by
  obtain ⟨m, hm, rfl⟩ := route_mid cut T _ q h
  refine Finset.mem_biUnion.2 ⟨min i x.toList.length, Finset.mem_range.2 (by omega),
    Finset.mem_image.2 ⟨m, hm, ?_⟩⟩
  congr 1
  apply FreeMonoid.toList.injective
  simp only [prefixOf, FreeMonoid.toList_ofList]
  rcases le_total i x.toList.length with h | h
  · rw [min_eq_left h]
  · rw [min_eq_right h, List.take_of_length_le h, List.take_of_length_le le_rfl]

variable (cut : FreeMonoid α → Option Bool) (T : DTree α) (edges : EdgeTable α) (k : ℕ)
  (x : FreeMonoid α)

theorem mem_firstsBy_asks {q : ReadLabel α × FreeMonoid α} (hq : q ∈ firstsBy cut T edges k x) :
    q ∈ (probeQuery T edges k x).asks (ofCut cut) :=
  QTree.firsts_sub_asks _ _ _ q hq

theorem firstsBy_nodup : ((firstsBy cut T edges k x).map Prod.snd).Nodup :=
  (QTree.firsts_keys_eq _ _ _).1

/-- A probe's first read is on its position's route, at the start or at a position it sifts. -/
theorem firstsBy_pos {q : ReadLabel α × FreeMonoid α} (hq : q ∈ firstsBy cut T edges k x) :
    q.2 ∈ (T.route cut (prefixOf x q.1.1)).1
      ∧ (prefixOf x q.1.1 = prefixOf x k
        ∨ (q.1.1 ∈ siftsBy cut T edges k x
          ∧ ∃ e, posEdgeBy cut T edges k x q.1.1 = some e ∧ q.1.2 = some e)) := by
  have := asks_probeQuery T edges k x (ofCut cut) q (mem_firstsBy_asks cut T edges k x hq)
  simpa only [cut_ofCut] using this

theorem firstsBy_label {q : ReadLabel α × FreeMonoid α} (hq : q ∈ firstsBy cut T edges k x) :
    q.1 = (k, none) ∨ q.1.2 = posEdgeBy cut T edges k x q.1.1 := by
  have := asks_probeQuery_label T edges k x (ofCut cut) q (mem_firstsBy_asks cut T edges k x hq)
  simpa only [cut_ofCut] using this

theorem firstsBy_mem_probeStrings (hkx : k ≤ x.toList.length) {q : ReadLabel α × FreeMonoid α}
    (hq : q ∈ firstsBy cut T edges k x) : q.2 ∈ probeStrings T k x := by
  obtain ⟨hr, hp⟩ := firstsBy_pos cut T edges k x hq
  rcases hp with hp | ⟨hs, -⟩
  · rw [hp] at hr; exact route_mem_probeStrings hkx le_rfl hr
  · exact route_mem_probeStrings hkx (siftsBy_range cut hs).1.le hr

theorem firstsBy_length_le :
    (firstsBy cut T edges k x).length ≤ (x.toList.length + 1) * T.midfixes.card := by
  rw [← List.length_map (f := Prod.snd), ← List.toFinset_card_of_nodup
    (firstsBy_nodup cut T edges k x)]
  refine le_trans (Finset.card_le_card fun w hw => ?_) (prefStrings_card T x)
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 (List.mem_toFinset.1 hw)
  exact route_mem_prefStrings (firstsBy_pos cut T edges k x hq).1

/-- The strings the sifts of `x`'s prefixes stop at undecided. -/
noncomputable def stopStrings : Finset (FreeMonoid α) :=
  (Finset.range (x.toList.length + 1)).biUnion fun i =>
    (T.siftCut cut (prefixOf x i)).elim (fun _ => ∅) fun b => {b}

theorem stopStrings_card : (stopStrings cut T x).card ≤ x.toList.length + 1 := by
  unfold stopStrings
  refine Finset.card_biUnion_le.trans ?_
  calc ∑ i ∈ Finset.range (x.toList.length + 1),
        ((T.siftCut cut (prefixOf x i)).elim (fun _ => ∅) fun b => {b} : Finset (FreeMonoid α)).card
      ≤ ∑ _i ∈ Finset.range (x.toList.length + 1), 1 := by
        refine Finset.sum_le_sum fun i _ => ?_
        cases T.siftCut cut (prefixOf x i) <;> simp
    _ = _ := by simp

theorem und_mem_stopStrings {q : ReadLabel α × FreeMonoid α} (hq : q ∈ firstsBy cut T edges k x)
    (hu : cut q.2 = none) : q.2 ∈ stopStrings cut T x := by
  have hr := (firstsBy_pos cut T edges k x hq).1
  have hs := siftCut_of_route_und T _ _ hr hu
  refine Finset.mem_biUnion.2 ⟨min q.1.1 x.toList.length, Finset.mem_range.2 (by omega), ?_⟩
  have : prefixOf x (min q.1.1 x.toList.length) = prefixOf x q.1.1 := by
    apply FreeMonoid.toList.injective
    simp only [prefixOf, FreeMonoid.toList_ofList]
    rcases le_total q.1.1 x.toList.length with h | h
    · rw [min_eq_left h]
    · rw [min_eq_right h, List.take_of_length_le h, List.take_of_length_le le_rfl]
  rw [this, hs]
  simp

/-- A probe's undecided first reads number at most its positions. -/
theorem undFirsts_length_le :
    ((firstsBy cut T edges k x).filter fun q => cut q.2 = none).length ≤ x.toList.length + 1 := by
  have hnd : (((firstsBy cut T edges k x).filter fun q => cut q.2 = none).map Prod.snd).Nodup :=
    (firstsBy_nodup cut T edges k x).sublist (List.filter_sublist.map _)
  rw [← List.length_map (f := Prod.snd), ← List.toFinset_card_of_nodup hnd]
  refine le_trans (Finset.card_le_card fun w hw => ?_) (stopStrings_card cut T x)
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 (List.mem_toFinset.1 hw)
  rw [List.mem_filter, decide_eq_true_eq] at hq
  exact und_mem_stopStrings cut T edges k x hq.1 hq.2

omit [Fintype α] [DecidableEq α] in
theorem sum_filter_some_le {X E : Type*} [DecidableEq E] (g : X → Option E) (K : Finset E) :
    ∀ l : List X, ∑ e ∈ K, (l.filter fun q => g q = some e).length ≤ l.length
  | [] => by simp
  | a :: l => by
    have ih := sum_filter_some_le g K l
    have h1 : ∑ e ∈ K, (if g a = some e then 1 else 0) ≤ 1 := by
      rcases hg : g a with _ | v
      · simp
      · simp only [Option.some.injEq]
        rw [Finset.sum_ite_eq]; split_ifs <;> simp
    have : ∀ e, ((a :: l).filter fun q => g q = some e).length
        = (if g a = some e then 1 else 0) + (l.filter fun q => g q = some e).length := by
      intro e
      by_cases h : g a = some e <;> simp [List.filter_cons, h, add_comm]
    simp only [this, Finset.sum_add_distrib, List.length_cons]
    omega

theorem edgeReadsBy_le (e : List Bool × α) :
    edgeReadsBy cut T edges k x e ≤ (x.toList.length + 1) * T.midfixes.card :=
  (List.length_filter_le _ _).trans (firstsBy_length_le cut T edges k x)

theorem edgeHarvestBy_eq (e : List Bool × α) : edgeHarvestBy cut T edges k x e
    = ((((firstsBy cut T edges k x).filter fun q => cut q.2 = none).filter
        fun q => q.1.2 = some e)).map Prod.snd := by
  unfold edgeHarvestBy
  congr 1
  rw [List.filter_filter]
  congr 1
  funext q
  by_cases h1 : q.1.2 = some e <;> by_cases h2 : cut q.2 = none <;> simp [h1, h2]

theorem sum_edgeHarvestBy_le (K : Finset (List Bool × α)) :
    ∑ e ∈ K, (edgeHarvestBy cut T edges k x e).length ≤ x.toList.length + 1 := by
  simp only [edgeHarvestBy_eq, List.length_map]
  exact (sum_filter_some_le (fun q : ReadLabel α × FreeMonoid α => q.1.2) K _).trans
    (undFirsts_length_le cut T edges k x)

theorem edgeHarvestBy_length_le (e : List Bool × α) :
    (edgeHarvestBy cut T edges k x e).length ≤ x.toList.length + 1 := by
  have := sum_edgeHarvestBy_le cut T edges k x {e}
  simpa using this

theorem edgeHarvestBy_le_edgeReadsBy (e : List Bool × α) :
    (edgeHarvestBy cut T edges k x e).length ≤ edgeReadsBy cut T edges k x e := by
  unfold edgeHarvestBy edgeReadsBy
  rw [List.length_map]
  exact (List.monotone_filter_right _ fun q h => by
    simp only [decide_eq_true_eq] at h ⊢; exact h.1).length_le

theorem firstsBy_short (h : x.toList.length < k) : firstsBy cut T edges k x = [] := by
  simp [firstsBy, probeQuery, h, QTree.firsts]

theorem edgeReadsBy_mem {edges : EdgeTable α} (he : EdgesInto T edges) {e : List Bool × α}
    (h : edgeReadsBy cut T edges k x e ≠ 0) : e.1 ∈ T.leaves := by
  unfold edgeReadsBy at h
  obtain ⟨q, hq⟩ := List.exists_mem_of_length_pos (Nat.pos_of_ne_zero h)
  rw [List.mem_filter, decide_eq_true_eq] at hq
  rcases firstsBy_label cut T edges k x hq.1 with hl | hl
  · rw [hl] at hq; simp at hq
  · rw [hq.2] at hl
    exact posEdgeBy_mem_paths he hl.symm

theorem edgeHarvestBy_mem {edges : EdgeTable α} (he : EdgesInto T edges) {e : List Bool × α}
    (h : edgeHarvestBy cut T edges k x e ≠ []) : e.1 ∈ T.leaves := by
  refine edgeReadsBy_mem cut T k x he fun h0 => h ?_
  have := edgeHarvestBy_le_edgeReadsBy cut T edges k x e
  rw [h0] at this
  exact List.eq_nil_of_length_eq_zero (by omega)

end Count

section Short

variable (cut : FreeMonoid α → Option Bool) (T : DTree α) (edges : EdgeTable α) {k : ℕ}
  {x : FreeMonoid α}

theorem probeBy_short (h : x.toList.length < k) : probeBy cut T edges k x = .agree := by
  simp [probeBy, h]

theorem edgeReadsBy_short (h : x.toList.length < k) (e : List Bool × α) :
    edgeReadsBy cut T edges k x e = 0 := by
  simp [edgeReadsBy, firstsBy_short cut T edges k x h]

theorem edgeHarvestBy_short (h : x.toList.length < k) (e : List Bool × α) :
    edgeHarvestBy cut T edges k x e = [] := by
  simp [edgeHarvestBy, firstsBy_short cut T edges k x h]

theorem middleHarvestBy_short (h : x.toList.length < k) : middleHarvestBy cut T edges k x = [] := by
  simp [middleHarvestBy, probeBy_short cut T edges h]

end Short

section Congr

variable {cut cut' : FreeMonoid α → Option Bool} {T : DTree α} {edges : EdgeTable α} {k : ℕ}
  {x : FreeMonoid α}

theorem firstsBy_congr_pref (h : ∀ z ∈ prefStrings T x, cut z = cut' z) :
    firstsBy cut T edges k x = firstsBy cut' T edges k x := by
  unfold firstsBy
  refine QTree.firsts_congr_asks _ _ fun q hq => ?_
  have hp := (asks_probeQuery T edges k x (ofCut cut) q hq).1
  simp only [cut_ofCut] at hp
  simp only [ofCut, h q.2 (route_mem_prefStrings hp)]

theorem edgeReadsBy_congr_pref (h : ∀ z ∈ prefStrings T x, cut z = cut' z) :
    edgeReadsBy cut T edges k x = edgeReadsBy cut' T edges k x := by
  funext e
  simp only [edgeReadsBy, firstsBy_congr_pref (edges := edges) (k := k) h]

theorem edgeHarvestBy_congr_pref (h : ∀ z ∈ prefStrings T x, cut z = cut' z) :
    edgeHarvestBy cut T edges k x = edgeHarvestBy cut' T edges k x := by
  funext e
  unfold edgeHarvestBy
  rw [firstsBy_congr_pref (edges := edges) (k := k) h]
  congr 1
  refine List.filter_congr fun q hq => ?_
  rw [h q.2 (route_mem_prefStrings (firstsBy_pos cut' T edges k x hq).1)]

theorem firstsBy_congr_strings (hkx : k ≤ x.toList.length)
    (h : ∀ z ∈ probeStrings T k x, cut z = cut' z) :
    firstsBy cut T edges k x = firstsBy cut' T edges k x := by
  unfold firstsBy
  refine QTree.firsts_congr_asks _ _ fun q hq => ?_
  have hp := asks_probeQuery T edges k x (ofCut cut) q hq
  simp only [cut_ofCut] at hp
  have hz : q.2 ∈ probeStrings T k x := by
    obtain ⟨hr, hp⟩ := hp
    rcases hp with hp | ⟨hs, -⟩
    · rw [hp] at hr; exact route_mem_probeStrings hkx le_rfl hr
    · exact route_mem_probeStrings hkx (siftsBy_range cut hs).1.le hr
  simp only [ofCut, h q.2 hz]

theorem edgeReadsBy_congr_strings (hkx : k ≤ x.toList.length)
    (h : ∀ z ∈ probeStrings T k x, cut z = cut' z) :
    edgeReadsBy cut T edges k x = edgeReadsBy cut' T edges k x := by
  funext e
  simp only [edgeReadsBy, firstsBy_congr_strings (edges := edges) hkx h]

theorem edgeHarvestBy_congr_strings (hkx : k ≤ x.toList.length)
    (h : ∀ z ∈ probeStrings T k x, cut z = cut' z) :
    edgeHarvestBy cut T edges k x = edgeHarvestBy cut' T edges k x := by
  funext e
  unfold edgeHarvestBy
  rw [firstsBy_congr_strings (edges := edges) hkx h]
  congr 1
  refine List.filter_congr fun q hq => ?_
  rw [h q.2 (firstsBy_mem_probeStrings cut' T edges k x hkx hq)]

theorem firstsBy_econgr {e e' : EdgeTable α} (he : EdgesInto T e) (hg : TgtAgree T e e')
    (cut : FreeMonoid α → Option Bool) : firstsBy cut T e k x = firstsBy cut T e' k x := by
  unfold firstsBy probeQuery
  split_ifs
  · rfl
  rw [QTree.firsts_bind, QTree.firsts_bind]
  congr 1
  rw [run_siftQuery]
  rcases hs : T.siftCut (fun z => (ofCut cut z).cut) (prefixOf x k) with p₀ | b
  · have hp₀ : p₀ ∈ T.leaves := DTree.siftCut_mem_paths _ _ _ hs
    have hf : ∀ cs, follow e p₀ cs = follow e' p₀ cs := fun cs => follow_congr he hg cs p₀ hp₀
    have hw : walkFrom e k x p₀ = walkFrom e' k x p₀ := by
      funext j; simp only [walkFrom, hf]
    have hpos : posEdgeFrom e k x p₀ = posEdgeFrom e' k x p₀ := by
      funext i; simp only [posEdgeFrom, hw]
    have hpl : readLabel e k x p₀ = readLabel e' k x p₀ := by
      funext i; simp only [readLabel, hpos]
    have hag : agreementQuery T e k x p₀ = agreementQuery T e' k x p₀ := by
      funext wa i; simp only [agreementQuery, hpl]
    simp only [hf, hw, hpl, hag]
  · rfl

theorem edgeReadsBy_econgr {e e' : EdgeTable α} (he : EdgesInto T e) (hg : TgtAgree T e e')
    (cut : FreeMonoid α → Option Bool) : edgeReadsBy cut T e k x = edgeReadsBy cut T e' k x := by
  funext d
  simp only [edgeReadsBy, firstsBy_econgr he hg cut]

theorem edgeHarvestBy_econgr {e e' : EdgeTable α} (he : EdgesInto T e) (hg : TgtAgree T e e')
    (cut : FreeMonoid α → Option Bool) : edgeHarvestBy cut T e k x
        = edgeHarvestBy cut T e' k x := by
  funext d
  simp only [edgeHarvestBy, firstsBy_econgr he hg cut]

end Congr

end OrthoDFA
