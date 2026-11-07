import OrthoDFA.Proofs.Query
import OrthoDFA.Proofs.KappaReads
import OrthoDFA.Proofs.EpsReads
import OrthoDFA.Proofs.LoopRuns
import OrthoDFA.Proofs.RunReads
import OrthoDFA.Code.Level3DrawMany

/-!
# Level 3 and the harvest, as programs

Each program asks the read at no more strings than the function's reads, so that with probes at most `L` long a level-3
run over `T` probes whose final tree has at most `n + 2` leaves asks at most
`T probeSifts L (n + 1) + n (n + 2)|Σ|(n + 2)` strings, a split's re-sifts charged to the leaf
it adds (`card_reads_level3Q_le`), and a harvest
draw over `N` probes at most `N (3 + ⌈log₂ L⌉)|mids|` (`card_reads_harvestFirstQ_le`).
-/

noncomputable section

namespace OrthoDFA

open Query

variable {α : Type*}

/-! ## The family read -/

section FamilyRead

variable {S : Type*} [Stringlike S]

theorem card_reads_familyReadQ_le [DecidableEq S] (F : Finset S) (kl kh : ℕ) (w : S)
    (o : S → ℝ) : ((familyReadQ F kl kh w).reads o).card ≤ F.card := by
  unfold familyReadQ
  rw [reads_map, reads_table]
  refine (List.toFinset_card_le _).trans ?_
  simp

end FamilyRead

/-! ## The sift -/

variable [inst : DecidableEq (FreeMonoid α)]

theorem DTree.reads_siftQ_sub (read : FreeMonoid α → ARU) :
    ∀ (T : DTree α) (z : FreeMonoid α),
      (T.siftQ z).reads read ⊆ Random.Kappa.siftReads read T z
  | .leaf, _ => by simp [DTree.siftQ]
  | .node m r a, z => by
    intro w hw
    simp only [DTree.siftQ, reads_ask, Finset.mem_insert] at hw
    simp only [Random.Kappa.siftReads, Finset.mem_insert]
    rcases hw with rfl | hw
    · exact Or.inl rfl
    right
    rcases h : read (z * m) with _ | _ | _ <;> rw [h] at hw <;> simp only [reads_map] at hw
    · exact DTree.reads_siftQ_sub read a z hw
    · exact DTree.reads_siftQ_sub read r z hw
    · simp at hw

/-! ## The probe -/

section Probe

open Random.Kappa

variable (read : FreeMonoid α → ARU) (T : DTree α) (x : FreeMonoid α) (st : ℕ → List Bool)

theorem reads_siftAgreesQ_sub (p : ℕ) :
    (siftAgreesQ T x st p).reads read ⊆ siftReads read T (prefixOf x p) := by
  simpa [siftAgreesQ] using DTree.reads_siftQ_sub read T (prefixOf x p)

theorem reads_searchOutcomeQ_sub (un : Option (α × List Bool)) (hi : ℕ) (res : SearchResult) :
    (searchOutcomeQ T x st un hi res).reads read ⊆ posReads read T x (resultPos res hi) := by
  intro w hw
  refine mem_posReads.2 ?_
  cases res with
  | edge p =>
    simp only [searchOutcomeQ] at hw
    split at hw
    · simp only [reads_map] at hw
      exact ⟨p, by simp [resultPos], DTree.reads_siftQ_sub read T _ hw⟩
    · simp at hw
  | undecidedHi =>
    simp only [searchOutcomeQ, undecidedAtQ, reads_map] at hw
    exact ⟨hi, by simp [resultPos], DTree.reads_siftQ_sub read T _ hw⟩
  | undecidedMid p =>
    simp only [searchOutcomeQ, undecidedAtQ, reads_map] at hw
    exact ⟨p, by simp [resultPos], DTree.reads_siftQ_sub read T _ hw⟩
  | _ => simp [searchOutcomeQ] at hw

theorem reads_probeOrAgreeQ_sub (E : Edges α) (k : ℕ) :
    (probeOrAgreeQ T E k x).reads read ⊆ probeReads read T E k x := by
  unfold probeOrAgreeQ probeReads
  split_ifs with hk
  · simp
  intro w hw
  unfold probeQ at hw
  simp only [reads_bind, Finset.mem_union] at hw
  simp only [Finset.mem_union]
  rcases hw with hw | hw
  · exact Or.inl (DTree.reads_siftQ_sub read T _ hw)
  refine Or.inr ?_
  rw [← DTree.sift_def] at hw
  rcases hs : T.sift read (prefixOf x k) with a | z
  · rw [hs] at hw
    simp only at hw ⊢
    set st : ℕ → List Bool := fun p => (walk E a (x.toList.drop k)).getD (p - k) []
    set hi := k + (walk E a (x.toList.drop k)).length - 1
    have hsr : ∀ un, ∀ v ∈ ((tripartiteQ false k hi).subst (siftAgreesQ T x st)).reads read
        ∪ (searchOutcomeQ T x st un hi
          (((tripartiteQ false k hi).subst (siftAgreesQ T x st)).eval read)).reads read,
        v ∈ posReads read T x (searchPos (siftAgrees read T x st) k hi) := by
      intro un v hv
      rcases Finset.mem_union.1 hv with hv | hv
      · rw [reads_subst] at hv
        obtain ⟨p, hp, hvp⟩ := Finset.mem_biUnion.1 hv
        exact mem_posReads.2 ⟨p, Finset.mem_union_left _ hp,
          reads_siftAgreesQ_sub read T x st p hvp⟩
      · rw [eval_subst] at hv
        obtain ⟨p, hp, hvp⟩ := mem_posReads.1 (reads_searchOutcomeQ_sub read T x st un hi _ hv)
        exact mem_posReads.2 ⟨p, Finset.mem_union_right _ hp, hvp⟩
    rcases hc : x.toList[hi]? with _ | c
    · rw [hc] at hw
      simp only [reads_bind] at hw
      exact hsr none w hw
    · rw [hc] at hw
      simp only [reads_bind, Finset.mem_union] at hw
      simp only [Finset.mem_union]
      rcases hw with hw | hw
      · exact Or.inl (DTree.reads_siftQ_sub read T _ hw)
      · refine Or.inr ?_
        rw [← DTree.sift_def] at hw
        rcases hs' : T.sift read (prefixOf x (hi + 1)) with t | z'
        · rw [hs'] at hw
          simp only [reads_bind] at hw
          exact hsr _ w hw
        · rw [hs'] at hw
          simp at hw
  · rw [hs] at hw
    simp at hw

end Probe

/-! ## Level 3 -/

section Level3

open Random Random.Kappa Random.Eps

variable (read : FreeMonoid α → ARU)

theorem reads_retargetAtQ_sub (T' : DTree α) (p : List Bool) (E : Edges α) (e : List Bool × α)
    {v : FreeMonoid α} (hv : v ∈ (retargetAtQ T' p E e).reads read) :
    ∃ t w, E e = some (t, w) ∧ ∃ m ∈ mids T', v = (w * FreeMonoid.of e.2) * m := by
  unfold retargetAtQ at hv
  split_ifs at hv
  · simp at hv
  rcases hE : E e with _ | ⟨t, w⟩ <;> rw [hE] at hv
  · simp at hv
  · simp only at hv
    split_ifs at hv
    · rw [reads_map] at hv
      obtain ⟨m, hm, rfl⟩ := siftReads_mids T' _ v (DTree.reads_siftQ_sub read T' _ hv)
      exact ⟨t, w, rfl, m, hm, rfl⟩
    · simp at hv

variable [Fintype α] [DecidableEq α] (C : Level3Settings)

theorem reads_splitLeafQ_sub (s : Level3State α) (p : List Bool) (c : α) (t t₀ : List Bool) :
    (splitLeafQ s p c t t₀).reads read
      ⊆ resifts s (s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p) := by
  unfold splitLeafQ
  rw [reads_map]
  generalize s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p = T'
  have : ∀ es : List (List Bool × α), (∀ e ∈ es, e ∈ s.tree.leaves.toFinset ×ˢ Finset.univ) →
      (retargetQ T' p s.edges es).reads read ⊆ resifts s T' := by
    intro es
    induction es with
    | nil => simp [retargetQ]
    | cons e es ih =>
      intro hes v hv
      simp only [retargetQ, reads_bind, reads_map, Finset.mem_union] at hv
      rcases hv with hv | hv
      · obtain ⟨t, w, hE, m, hm, rfl⟩ := reads_retargetAtQ_sub read T' p _ e hv
        simp only [resifts, Finset.mem_biUnion]
        refine ⟨e, hes e (by simp), ?_⟩
        rw [hE]
        simp only [Finset.mem_image]
        exact ⟨m, hm, rfl⟩
      · exact ih (fun e' he' => hes e' (by simp [he'])) hv
  exact this _ fun e he => Finset.mem_toList.1 he

theorem reads_applyProbeQ_sub (s : Level3State α) (x : FreeMonoid α) :
    (applyProbeQ C s x).reads read ⊆ probeReads read s.tree s.edges C.walkStart x
      ∪ splitReads read C s x := by
  unfold applyProbeQ splitReads
  rw [reads_bind, eval_probeOrAgreeQ]
  refine Finset.union_subset_union (reads_probeOrAgreeQ_sub read _ x _ _) ?_
  rcases probeOrAgree read s.tree s.edges C.walkStart x with _ | ⟨p, c, u, t⟩ | _ | _ | _ <;>
    try simp
  rcases s.edges (p, c) with _ | ⟨t₀, w⟩
  · simp
  · simp only [reads_map]
    exact reads_splitLeafQ_sub read s p c t t₀

end Level3

/-! ## The level-3 run's reads -/

section Level3Card

open Random Random.Kappa Random.Eps Random.Spur

variable [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)

omit inst in
/-- A step that re-sifts adds a leaf. -/
theorem splitReads_eq_empty_or {X : List (FreeMonoid α)} {s : Level3State α}
    (hs : RunInv read C X s) (x : FreeMonoid α) :
    splitReads read C s x = ∅ ∨ ∃ s', applyProbe read C s x = .inl s'
      ∧ s'.tree.leaves.length = s.tree.leaves.length + 1 := by
  have hOK := probeOrAgree_ok read s.tree s.edges C.walkStart hs.inv.edges x
  unfold splitReads applyProbe
  generalize probeOrAgree read s.tree s.edges C.walkStart x = o at hOK ⊢
  cases o with
  | wrongEdge p c u t =>
    obtain ⟨hp, -, -, t₀, w₀, hE, -⟩ := hOK
    simp only [hE]
    exact Or.inr ⟨_, rfl, DTree.length_leaves_splitAt _ _ _ hp⟩
  | _ => exact Or.inl rfl

/-- A run from `s` over probes at most `L` long, its final tree at most `n + 2` leaves, asks at most
`probeSifts L (n + 1)` strings in each probe and `(n + 2)|Σ|(n + 2)` in each split, of which there
are at most `n + 2 - |leaves s|`: no step loses a leaf. -/
theorem card_reads_runProbesQ_le {L : ℕ} (n : ℕ) (X : List (FreeMonoid α)) :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α), (∀ y ∈ xs, y ∈ X) →
      (∀ y ∈ xs, y.toList.length ≤ L) → RunInv read C X s →
      (runProbes read C s xs).1.tree.leaves.length ≤ n + 2 →
      ((runProbesQ C s xs).reads read).card
        ≤ xs.length * (probeSifts L * (n + 1))
          + (n + 2 - s.tree.leaves.length) * ((n + 2) * Fintype.card α * (n + 2))
  | [], _, _, _, _, _ => by simp [runProbesQ]
  | x :: xs, s, hX, hL, hs, hfin => by
    have hc : s.tree.leaves.length ≤ n + 2 :=
      (leaves_le_runProbes read C (x :: xs) s hX hs).trans hfin
    have hxL : x.toList.length ≤ L := hL x (by simp)
    have hmid : (mids s.tree).card ≤ n + 1 := by have := mids_card s.tree; omega
    have hprobe : (probeReads read s.tree s.edges C.walkStart x).card ≤ probeSifts L * (n + 1) :=
      (probeReads_card_mids read hxL).trans (Nat.mul_le_mul_left _ hmid)
    have hsplit : (splitReads read C s x).card ≤ (n + 2) * Fintype.card α * (n + 2) :=
      (splitReads_card_le C read s x).trans
        (Nat.mul_le_mul (Nat.mul_le_mul_right _ (by omega)) (by omega))
    have hhead : ((applyProbeQ C s x).reads read).card
        ≤ (probeReads read s.tree s.edges C.walkStart x).card + (splitReads read C s x).card :=
      (Finset.card_le_card (reads_applyProbeQ_sub read C s x)).trans (Finset.card_union_le _ _)
    have hX' : ∀ y ∈ xs, y ∈ X := fun y hy => hX y (by simp [hy])
    have hL' : ∀ y ∈ xs, y.toList.length ≤ L := fun y hy => hL y (by simp [hy])
    have hfin' : ∀ s', applyProbe read C s x = .inl s' →
        (runProbes read C s' xs).1.tree.leaves.length ≤ n + 2 := fun s' hap => by
      have e : runProbes read C s (x :: xs) = runProbes read C s' xs := by simp [runProbes, hap]
      rwa [e] at hfin
    simp only [runProbesQ, reads_bind, eval_applyProbeQ read C s x, List.length_cons]
    refine (Finset.card_union_le _ _).trans ?_
    rcases splitReads_eq_empty_or read C hs x with h0 | ⟨s', hap, hlen⟩
    · rw [h0, Finset.card_empty, add_zero] at hhead
      rcases hap : applyProbe read C s x with s' | e
      · have hrec := card_reads_runProbesQ_le n X xs s' hX' hL'
          (runInv_step read C (hX x (by simp)) hs hap).1 (hfin' s' hap)
        have hle := leaves_le_step read C hs hap
        simp only
        generalize probeSifts L * (n + 1) = P at hprobe hrec ⊢
        generalize (n + 2) * Fintype.card α * (n + 2) = R at hrec ⊢
        have hR : (n + 2 - s'.tree.leaves.length) * R
            ≤ (n + 2 - s.tree.leaves.length) * R := Nat.mul_le_mul_right _ (by omega)
        calc _ ≤ P + (xs.length * P + (n + 2 - s'.tree.leaves.length) * R) :=
              Nat.add_le_add (hhead.trans hprobe) hrec
          _ ≤ P + (xs.length * P + (n + 2 - s.tree.leaves.length) * R) := by omega
          _ = (xs.length + 1) * P + (n + 2 - s.tree.leaves.length) * R := by ring
      · simp only [reads_pure, Finset.card_empty, add_zero]
        calc _ ≤ probeSifts L * (n + 1) := hhead.trans hprobe
          _ ≤ (xs.length + 1) * (probeSifts L * (n + 1)) :=
            Nat.le_mul_of_pos_left _ (Nat.succ_pos _)
          _ ≤ _ := Nat.le_add_right _ _
    · rw [hap]
      have hs' := (runInv_step read C (hX x (by simp)) hs hap).1
      have hrec := card_reads_runProbesQ_le n X xs s' hX' hL' hs' (hfin' s' hap)
      have hs'le := (leaves_le_runProbes read C xs s' hX' hs').trans (hfin' s' hap)
      simp only
      obtain ⟨m, hm⟩ : ∃ m, n + 2 - s.tree.leaves.length = m + 1 :=
        ⟨n + 1 - s.tree.leaves.length, by omega⟩
      rw [hlen, show n + 2 - (s.tree.leaves.length + 1) = m by omega] at hrec
      rw [hm]
      generalize probeSifts L * (n + 1) = P at hprobe hrec ⊢
      generalize (n + 2) * Fintype.card α * (n + 2) = R at hsplit hrec ⊢
      calc _ ≤ (P + R) + (xs.length * P + m * R) :=
            Nat.add_le_add (hhead.trans (Nat.add_le_add hprobe hsplit)) hrec
        _ = (xs.length + 1) * P + (m + 1) * R := by ring

/-- Level 3's run over `T` probes at most `L` long, its final tree at most `n + 2` leaves, asks at
most `probeSifts L (n + 1)` strings in each probe and `(n + 2)|Σ|(n + 2)` in each of at most `n`
splits. -/
theorem card_reads_level3Q_le {L T : ℕ} (n : ℕ) (xs : Fin T → FreeMonoid α)
    (hL : ∀ i, (xs i).toList.length ≤ L)
    (hn : (indecisiveLStarLevel3 C read xs).hypothesis.tree.leaves.length ≤ n + 2) :
    ((level3Q C xs).reads read).card
      ≤ T * (probeSifts L * (n + 1)) + n * ((n + 2) * Fintype.card α * (n + 2)) := by
  unfold level3Q
  rw [reads_map]
  have h := card_reads_runProbesQ_le read C n (List.ofFn xs) (List.ofFn xs) initialState
    (fun _ h => h) (fun y hy => by
      obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hy
      exact hL i) (runInv_start read C _) hn
  have h0 : (initialState : Level3State α).tree.leaves.length = 2 := by
    simp [initialState, newHypothesis, DTree.leaves]
  rwa [List.length_ofFn, h0, show n + 2 - 2 = n by omega] at h

end Level3Card

/-! ## The harvest -/

section Harvest

open Random Random.Kappa

variable (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α) (site : Level3Site)
  (T : Set (FreeMonoid α))

omit inst in
theorem eval_siteHarvestQ (x : FreeMonoid α) :
    (siteHarvestQ k H site x).eval read = siteHarvest read k H site x := by
  cases site <;> simp [siteHarvestQ, siteHarvest, eval_probeOrAgreeQ]

theorem reads_siteHarvestQ_sub (x : FreeMonoid α) :
    (siteHarvestQ k H site x).reads read ⊆ probeReads read H.tree H.edges k x := by
  cases site <;> simp only [siteHarvestQ, reads_map] <;>
    exact reads_probeOrAgreeQ_sub read H.tree x H.edges k

omit inst in
theorem eval_harvestFirstQ (xs : List (FreeMonoid α)) :
    (harvestFirstQ k H site T xs).eval read = harvestFirst read k H site T xs := by
  unfold harvestFirstQ harvestFirst
  rw [eval_findSome]
  congr 1
  funext x
  rw [eval_map, eval_siteHarvestQ]

theorem card_reads_harvestFirstQ_le {L : ℕ} (xs : List (FreeMonoid α))
    (hL : ∀ x ∈ xs, x.toList.length ≤ L) :
    ((harvestFirstQ k H site T xs).reads read).card
      ≤ xs.length * ((3 + Nat.clog 2 L) * (mids H.tree).card) :=
  card_reads_findSome_le _ _ _ fun x hx => by
    rw [reads_map]
    exact (Finset.card_le_card (reads_siteHarvestQ_sub read k H site x)).trans
      (probeReads_card_clog read H.tree H.edges k (hL x hx))

omit inst in
theorem eval_drawManyFirstQ {m N' : ℕ} (yss : Fin m → (Fin N' → FreeMonoid α)) :
    (drawManyFirstQ k H site T yss).eval read = drawManyFirst read k H site T yss := by
  unfold drawManyFirstQ drawManyFirst
  simp only [eval_map, eval_pi, eval_harvestFirstQ]

theorem card_reads_drawManyFirstQ_le {L m N' : ℕ} (yss : Fin m → (Fin N' → FreeMonoid α))
    (hL : ∀ i t, (yss i t).toList.length ≤ L) :
    ((drawManyFirstQ k H site T yss).reads read).card
      ≤ m * (N' * ((3 + Nat.clog 2 L) * (mids H.tree).card)) := by
  unfold drawManyFirstQ
  rw [reads_map, reads_pi]
  refine (Finset.card_biUnion_le_card_mul _ _
    (N' * ((3 + Nat.clog 2 L) * (mids H.tree).card)) fun i _ => ?_).trans (by simp)
  have := card_reads_harvestFirstQ_le read k H site T (List.ofFn (yss i)) (L := L) fun y hy => by
    obtain ⟨t, rfl⟩ := List.mem_ofFn.1 hy
    exact hL i t
  rwa [List.length_ofFn] at this

end Harvest

end OrthoDFA

end
