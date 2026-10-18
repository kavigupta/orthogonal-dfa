import OrthoDFA.Proofs.LoopRenoise

/-!
# Level 5's claim is fixed by the short reads

Over probes at most `L` long, level 5's claim reads the family only at strings at most
`max (L + maxLeaves) (probes + 1 + maxLeaves)` long (`endsWell_congr_reads`). The clustering
reads at most three prefixes per population per prefix the schedule draws, and at most one suffix
more than the pool.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {K : ℕ}

section Reads

variable (C : Level5Settings) (cut : FreeMonoid α → Option Bool)

theorem runProbes_readInv {Lx : ℕ} :
    ∀ (s : Level5State α) (xs : List (FreeMonoid α)), ReadInv C Lx s →
      (∀ x ∈ xs, x.toList.length ≤ Lx) → (runProbes C cut s xs).2 ≠ .failed →
      ReadInv C Lx (runProbes C cut s xs).1
  | s, [], _, _, h => absurd rfl h
  | s, x :: xs, hs, hxs, h => by
    have hx := hxs x List.mem_cons_self
    simp only [runProbes] at h ⊢
    rcases ha : applyProbe C cut s x with s' | ⟨e, s'⟩ <;> simp only [ha] at h ⊢
    · exact runProbes_readInv s' xs (applyProbe_readInv C cut (recordProbe_readInv C cut hs hx) ha)
        (fun y hy => hxs y (List.mem_cons_of_mem _ hy)) h
    · rcases applyProbe_inr C cut ha with ⟨-, rfl⟩ | ⟨rfl, -⟩
      · exact recordProbe_readInv C cut hs hx
      · simp [toRoundEnd] at h

theorem level5_tree_midLe (hC : 2 ≤ C.maxLeaves) (read : FreeMonoid α → ARU) {T : ℕ}
    (xs : Fin T → FreeMonoid α) (h : (indecisiveLStarLevel5 C read xs).ending ≠ .failed) :
    MidLe C.maxLeaves (indecisiveLStarLevel5 C read xs).hypothesis.tree := by
  have hinv := runProbes_readInv C (fun z => (read z).cut) (Lx := probesLen xs) initialState
    (List.ofFn xs) ⟨by simp [initialState, freshState, DTree.leaves],
      by simpa [initialState, freshState, DTree.leaves] using hC,
      by simp [initialState, freshState, DTree.leaves, DTree.midfixes],
      by simp [initialState, freshState]⟩
    (fun x hx => by obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx; exact le_probesLen xs i) h
  obtain ⟨-, hL, hm, -⟩ := hinv
  intro m hmm
  have := hm m hmm
  show m.toList.length ≤ C.maxLeaves
  omega

/-- Over probes at most `L` long, level 5's claim reads only the short strings. -/
theorem endsWell_congr_reads (hC : 2 ≤ C.maxLeaves) {σ : Type*} (M : _root_.DFA α σ)
    (bad : σ → Prop) (dist : σ → ARU → ℝ) (D' : Measure (FreeMonoid α)) {L : ℕ}
    (hlen : ∀ᵐ x ∂D', x.toList.length ≤ L) (ε y : ℝ) {n : ℕ} (hn : n ≤ C.maxLeaves) (J : ℕ)
    (ū : ℝ) (sift : Bool) {T : ℕ} (xs : Fin T → FreeMonoid α) {read read' : FreeMonoid α → ARU}
    (h : ∀ z, z.toList.length ≤ max (L + C.maxLeaves) (probesLen xs + 1 + C.maxLeaves) →
      read z = read' z) :
    EndsWell M bad D' ε (checkedProbes M dist D' read C.walkStart n J ū
        (indecisiveLStarLevel5 C read xs).hypothesis
        ∪ siftedProbes sift read C.walkStart (indecisiveLStarLevel5 C read xs).hypothesis)
        C.walkStart
        (harvestLaw D' (fun z => (read z).cut) C.walkStart
          (indecisiveLStarLevel5 C read xs).hypothesis) y
        (indecisiveLStarLevel5 C read xs).ending
      ↔ EndsWell M bad D' ε (checkedProbes M dist D' read' C.walkStart n J ū
        (indecisiveLStarLevel5 C read' xs).hypothesis
        ∪ siftedProbes sift read' C.walkStart (indecisiveLStarLevel5 C read' xs).hypothesis)
        C.walkStart
        (harvestLaw D' (fun z => (read' z).cut) C.walkStart
          (indecisiveLStarLevel5 C read' xs).hypothesis) y
        (indecisiveLStarLevel5 C read' xs).ending := by
  have hres : indecisiveLStarLevel5 C read xs = indecisiveLStarLevel5 C read' xs :=
    level5_congr_up C hC xs (le_probesLen xs) fun z hz => h z (le_max_of_le_right hz)
  rw [← hres]
  set res := indecisiveLStarLevel5 C read xs
  have hcut : AgreeUpTo (L + C.maxLeaves) (fun z => (read z).cut) (fun z => (read' z).cut) :=
    fun z hz => by dsimp only; rw [h z (le_max_of_le_left hz)]
  rcases he : res.ending with _ | hv | _
  · -- consistent: the checked probes, at every short probe
    have hmid := level5_tree_midLe C hC read xs (by rw [he]; simp)
    have hae : (checkedProbes M dist D' read C.walkStart n J ū res.hypothesis
        ∪ siftedProbes sift read C.walkStart res.hypothesis : Set (FreeMonoid α))
        =ᵐ[D'] (checkedProbes M dist D' read' C.walkStart n J ū res.hypothesis
          ∪ siftedProbes sift read' C.walkStart res.hypothesis : Set (FreeMonoid α)) := by
      filter_upwards [hlen] with x hx
      have hpb := probeBy_congr_up hcut hmid (x := x) (by omega)
        (Function.curry res.hypothesis.edges) C.walkStart
      have hpre : ∀ j ≤ x.toList.length, ∀ m ∈ shortWords n,
          read (prefixOf x j * m) = read' (prefixOf x j * m) := fun j hj m hm => by
        refine h _ (le_max_of_le_left ?_)
        have := length_le_of_mem_shortWords hm
        simp only [FreeMonoid.toList_mul, List.length_append, prefixOf, FreeMonoid.toList_ofList,
          List.length_take]
        omega
      apply propext
      change x ∈ checkedProbes M dist D' read C.walkStart n J ū res.hypothesis
          ∪ siftedProbes sift read C.walkStart res.hypothesis
        ↔ x ∈ checkedProbes M dist D' read' C.walkStart n J ū res.hypothesis
          ∪ siftedProbes sift read' C.walkStart res.hypothesis
      have hsf : x ∈ siftedProbes sift read C.walkStart res.hypothesis
          ↔ x ∈ siftedProbes sift read' C.walkStart res.hypothesis := by
        simp only [siftedProbes, undSiftSet, Set.mem_setOf_eq]
        rw [hpb]
      rw [Set.mem_union, Set.mem_union, hsf]
      refine or_congr ?_ Iff.rfl
      simp only [checkedProbes, rarerProbes, undecidedProbes, Set.mem_union, Set.mem_setOf_eq]
      have hs : x ∈ searchingProbes read C.walkStart res.hypothesis
          ↔ x ∈ searchingProbes read' C.walkStart res.hypothesis := by
        show (match probeBy (fun z => (read z).cut) res.hypothesis.tree
            (Function.curry res.hypothesis.edges) C.walkStart x with
          | .undecidedMid _ | .edge _ _ | .member _ => True
          | _ => False) ↔ (match probeBy (fun z => (read' z).cut) res.hypothesis.tree
            (Function.curry res.hypothesis.edges) C.walkStart x with
          | .undecidedMid _ | .edge _ _ | .member _ => True
          | _ => False)
        rw [hpb]
      rw [hs]
      refine or_congr (or_congr Iff.rfl ?_) (and_congr Iff.rfl (and_congr Iff.rfl ?_))
      · refine exists_congr fun j => and_congr_right fun _ => and_congr_right fun hj =>
          exists_congr fun m => and_congr_right fun hm => ?_
        rw [hpre j hj m hm]
      · refine exists_congr fun j => and_congr_right fun _ => and_congr_right fun hj =>
          exists_congr fun m => and_congr_right fun hm => ?_
        rw [hpre j hj m hm]
    simp only [EndsWell]
    rw [measureReal_congr hae]
  · have hshort := harvest_short C hC read xs he
    have hlaw := harvestLaw_congr_up (fun z => (read z).cut) (L := L) (M := C.maxLeaves) hlen le_rfl
      hcut C.walkStart (H := res.hypothesis) (fun m hm => by
        obtain ⟨-, hL, hmid⟩ := hshort
        have := hmid m hm
        show m.toList.length ≤ C.maxLeaves
        omega) hv
    simp only [EndsWell]
    rw [hlaw]
  · simp [EndsWell]

end Reads

section Prefs

lemma freshDraws_mono {S : Type*} [Stringlike S] (T : Finset S) (y : ℕ → S) {m m' : ℕ}
    (h : m ≤ m') (h' : ∃ n, m' ≤ (acceptedAt T y n).card) :
    freshDraws T y m ⊆ freshDraws T y m' := by
  have h'' : ∃ n, m ≤ (acceptedAt T y n).card := by
    obtain ⟨n, hn⟩ := h'; exact ⟨n, h.trans hn⟩
  rw [freshDraws_eq T y m h'', freshDraws_eq T y m' h']
  exact acceptedAt_mono T y (Nat.find_mono fun n hn => h.trans hn)

/-- Draws at the counts `n` to `2n` hold at most `2n` strings: the larger counts' draws extend the
smaller's, and a count the stream cannot fill draws nothing. -/
lemma card_biUnion_freshDraws_le {S : Type*} [Stringlike S] (T : Finset S) (y : ℕ → S) (n : ℕ) :
    ((Finset.range (n + 1)).biUnion fun e => freshDraws T y (n + e)).card ≤ 2 * n := by
  classical
  set P : ℕ → Prop := fun e => ∃ k, n + e ≤ (acceptedAt T y k).card
  set E := Nat.findGreatest P n
  by_cases hE : P E
  · have hsub : ((Finset.range (n + 1)).biUnion fun e => freshDraws T y (n + e))
        ⊆ freshDraws T y (n + E) := by
      intro v hv
      obtain ⟨e, he, hv⟩ := Finset.mem_biUnion.1 hv
      by_cases hPe : P e
      · have hle : e ≤ E := Nat.le_findGreatest (Nat.lt_succ_iff.1 (Finset.mem_range.1 he)) hPe
        exact freshDraws_mono T y (by omega) hE hv
      · rw [freshDraws_of_not T y _ hPe] at hv
        simp at hv
    refine (Finset.card_le_card hsub).trans ((card_freshDraws_le T y _).trans ?_)
    have := Nat.findGreatest_le (P := P) n
    omega
  · have hnone : ∀ e ≤ n, ¬ P e := fun e he hPe => hE (Nat.findGreatest_spec he hPe)
    have : ((Finset.range (n + 1)).biUnion fun e => freshDraws T y (n + e)) = ∅ := by
      refine Finset.eq_empty_of_forall_notMem fun v hv => ?_
      obtain ⟨e, he, hv⟩ := Finset.mem_biUnion.1 hv
      rw [freshDraws_of_not T y _ (hnone e (Nat.lt_succ_iff.1 (Finset.mem_range.1 he)))] at hv
      simp at hv
    rw [this]
    simp

variable {Ω : Type*} [MeasurableSpace Ω]

/-- The prefixes a round's clustering reads number at most three per population per prefix the
schedule draws. -/
lemma card_roundPrefs_le (sched : Finset State) (pops : Finset (Option (Fin K)))
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) :
    (roundPrefs sched pops x).card ≤ 3 * pops.card * ∑ B ∈ sched, B.npref := by
  classical
  unfold roundPrefs
  refine Finset.card_biUnion_le.trans ?_
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun B _ => (Finset.card_union_le _ _).trans ?_
  have h1 : (prefixesAt pops B.npref x).card ≤ pops.card * B.npref := by
    unfold prefixesAt
    refine Finset.card_biUnion_le.trans ?_
    calc ∑ j ∈ pops, (prefixesOf j B.npref x).card ≤ ∑ _j ∈ pops, B.npref :=
          Finset.sum_le_sum fun j _ => card_freshDraws_le _ _ _
      _ = _ := by rw [Finset.sum_const, smul_eq_mul]
  have h2 : (pops.biUnion fun j => (Finset.range (B.npref + 1)).biUnion fun e =>
      certOf (prefixesAt pops B.npref x) j (B.npref + e) x).card ≤ pops.card * (2 * B.npref) := by
    refine Finset.card_biUnion_le.trans ?_
    calc ∑ j ∈ pops, ((Finset.range (B.npref + 1)).biUnion fun e =>
          certOf (prefixesAt pops B.npref x) j (B.npref + e) x).card
        ≤ ∑ _j ∈ pops, 2 * B.npref :=
          Finset.sum_le_sum fun j _ => card_biUnion_freshDraws_le _ _ _
      _ = _ := by rw [Finset.sum_const, smul_eq_mul]
  nlinarith

/-- Where every state of the schedule draws `n` suffixes, the clustering reads at most `n + 1`. -/
lemma card_roundSuffs_le (sched : Finset State) {n : ℕ} (hn : ∀ B ∈ sched, B.nsuff = n)
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) : (roundSuffs sched x).card ≤ n + 1 := by
  classical
  have hsub : roundSuffs sched x ⊆ poolAt n x := by
    intro v hv
    obtain ⟨B, hB, hv⟩ := Finset.mem_biUnion.1 hv
    rwa [hn B hB] at hv
  refine (Finset.card_le_card hsub).trans ((Finset.card_insert_le _ _).trans ?_)
  exact Nat.succ_le_succ (card_freshDraws_le _ _ _)

end Prefs

end OrthoDFA
