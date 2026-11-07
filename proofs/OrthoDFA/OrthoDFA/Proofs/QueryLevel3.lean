import OrthoDFA.Proofs.Query
import OrthoDFA.Proofs.KappaReads
import OrthoDFA.Proofs.EpsReads
import OrthoDFA.Proofs.LoopRuns
import OrthoDFA.Proofs.RunReads
import OrthoDFA.Code.Level3DrawMany

/-!
# Level 3 and the harvest, as programs

Each program returns what its function returns (`eval_siftQ`, `eval_probeOrAgreeQ`,
`eval_harvestFirstQ`, `eval_drawManyFirstQ`, `eval_familyReadQ`), level 3's where its final tree
has at most `leafCap` leaves and `none` otherwise (`eval_level3Q`), and asks the read at no more
strings than the function's reads, so that with probes at most `L` long and `leafCap ≤ n + 2` a
level-3 run over `T` probes asks at most `T probeSifts L (n + 1) + (n + 1)(n + 2)|Σ|(n + 2)`
strings, a split's re-sifts charged to the leaf it adds (`card_reads_level3Q_le`), and a harvest
draw over `N` probes at most `N (3 + ⌈log₂ L⌉)|mids|` (`card_reads_harvestFirstQ_le`). A split asks only at the tree's leaves' edges, where a run keeps
every learned edge, so the program's run is the function's only along a run (`RunInv`).
-/

noncomputable section

namespace OrthoDFA

open Query

variable {α : Type*}

/-! ## The family read -/

section FamilyRead

variable {S : Type*} [Stringlike S]

theorem eval_familyReadQ (F : Finset S) (kl kh : ℕ) (w : S) (o : S → ℝ) :
    (familyReadQ F kl kh w).eval o = readOf kl kh (voteCount (fun z (_ : Unit) => o z) F w ()) := by
  unfold familyReadQ
  rw [eval_map, eval_table]
  simp only [voteCount]
  refine congrArg (fun n => readOf kl kh n) (congrArg Finset.card (Finset.filter_congr
    fun v hv => ?_))
  rw [if_pos (List.mem_map.2 ⟨v, Finset.mem_toList.2 hv, rfl⟩)]

theorem eval_familyReadQ_mq {Ω : Type*} [MeasurableSpace Ω] (mq : S → Ω → ℝ) (ω : Ω)
    (F : Finset S) (kl kh : ℕ) (w : S) :
    (familyReadQ F kl kh w).eval (fun z => mq z ω) = familyRead mq F kl kh w ω :=
  eval_familyReadQ F kl kh w _

theorem card_reads_familyReadQ_le [DecidableEq S] (F : Finset S) (kl kh : ℕ) (w : S)
    (o : S → ℝ) : ((familyReadQ F kl kh w).reads o).card ≤ F.card := by
  unfold familyReadQ
  rw [reads_map, reads_table]
  refine (List.toFinset_card_le _).trans ?_
  simp

end FamilyRead

/-! ## The sift and the search -/

theorem DTree.eval_siftQ (read : FreeMonoid α → ARU) :
    ∀ (T : DTree α) (z : FreeMonoid α), (T.siftQ z).eval read = T.sift read z
  | .leaf, _ => rfl
  | .node m r a, z => by
    rcases h : read (z * m) with _ | _ | _ <;>
      simp [DTree.siftQ, DTree.sift, h, DTree.eval_siftQ read a z, DTree.eval_siftQ read r z]

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

section Search

variable {Q R : Type*} [DecidableEq Q] (o : Q → R) (recover : Bool)
  (agrees : ℕ → Query Q R (Option Bool))

/-- What `p` asks is among the questions `agrees` asks at the positions `ps`. -/
def Covered (W : Finset Q) (ps : List ℕ) : Prop := ∀ q ∈ W, ∃ p ∈ ps, q ∈ (agrees p).reads o

theorem bisectQ_spec : ∀ (n lo hi : ℕ), hi - lo = n →
    (bisectQ recover agrees lo hi).eval o
        = (bisectBoundary recover (fun p => (agrees p).eval o) lo hi).1
      ∧ Covered o agrees ((bisectQ recover agrees lo hi).reads o)
        (bisectBoundary recover (fun p => (agrees p).eval o) lo hi).2 := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn
  rw [bisectQ, bisectBoundary]
  by_cases h : hi ≤ lo + 1
  · rw [dif_pos h, dif_pos h]
    exact ⟨rfl, fun q hq => by simp at hq⟩
  rw [dif_neg h, dif_neg h]
  have hlt : ∀ lo' hi', lo ≤ lo' → hi' ≤ hi → (lo' ≠ lo ∨ hi' ≠ hi) → hi' - lo' < n :=
    fun lo' hi' h1 h2 h3 => by omega
  set mid := (lo + hi) / 2 with hmid
  have hm1 : lo < mid := by omega
  have hm2 : mid < hi := by omega
  simp only [eval_bind, reads_bind]
  have cov_cons : ∀ {W : Finset Q} {ps : List ℕ} (p : ℕ), Covered o agrees W ps →
      Covered o agrees ((agrees p).reads o ∪ W) (p :: ps) := fun p hW q hq => by
    rcases Finset.mem_union.1 hq with hq | hq
    · exact ⟨p, List.mem_cons_self, hq⟩
    · obtain ⟨p', hp', hq'⟩ := hW q hq
      exact ⟨p', List.mem_cons_of_mem _ hp', hq'⟩
  rcases hA : (agrees mid).eval o with _ | _ | _
  · -- undecided at the midpoint
    simp only [hA]
    cases recover
    · refine ⟨rfl, fun q hq => ?_⟩
      simp only [Bool.false_eq_true, ↓reduceIte, reads_bind, reads_pure,
        Finset.union_empty] at hq
      exact ⟨mid, by simp, hq⟩
    simp only [↓reduceIte, eval_bind, reads_bind]
    rcases hB : (agrees (mid - 1)).eval o with _ | left
    · refine ⟨rfl, fun q hq => ?_⟩
      simp only [hB, reads_pure, Finset.union_empty, Finset.mem_union] at hq
      rcases hq with hq | hq
      · exact ⟨mid, by simp, hq⟩
      · exact ⟨mid - 1, by simp, hq⟩
    simp only [hB, eval_bind, reads_bind]
    rcases hC : (agrees (mid + 1)).eval o with _ | right
    · refine ⟨rfl, fun q hq => ?_⟩
      simp only [hC, reads_pure, Finset.union_empty, Finset.mem_union] at hq
      rcases hq with hq | hq | hq
      · exact ⟨mid, by simp, hq⟩
      · exact ⟨mid - 1, by simp, hq⟩
      · exact ⟨mid + 1, by simp, hq⟩
    have key : ∀ {W : Finset Q} {ps : List ℕ}, Covered o agrees W ps →
        Covered o agrees ((agrees mid).reads o ∪ ((agrees (mid - 1)).reads o
          ∪ ((agrees (mid + 1)).reads o ∪ W))) ([mid, mid - 1, mid + 1] ++ ps) :=
      fun hW q hq => by
        simp only [Finset.mem_union] at hq
        rcases hq with hq | hq | hq | hq
        · exact ⟨mid, by simp, hq⟩
        · exact ⟨mid - 1, by simp, hq⟩
        · exact ⟨mid + 1, by simp, hq⟩
        · obtain ⟨p', hp', hq'⟩ := hW q hq
          exact ⟨p', List.mem_append_right _ hp', hq'⟩
    rcases right with _ | _ <;> rcases left with _ | _
    · obtain ⟨h1, h2⟩ := ih _ (hlt lo (mid - 1) le_rfl (by omega) (by omega)) lo (mid - 1) rfl
      exact ⟨h1, key h2⟩
    · exact ⟨rfl, key fun q hq => by simp at hq⟩
    · obtain ⟨h1, h2⟩ := ih _ (hlt lo (mid - 1) le_rfl (by omega) (by omega)) lo (mid - 1) rfl
      exact ⟨h1, key h2⟩
    · obtain ⟨h1, h2⟩ := ih _ (hlt (mid + 1) hi (by omega) le_rfl (by omega)) (mid + 1) hi rfl
      exact ⟨h1, key h2⟩
  · try simp only [hA]
    obtain ⟨h1, h2⟩ := ih _ (hlt lo mid le_rfl (by omega) (by omega)) lo mid rfl
    exact ⟨h1, cov_cons mid h2⟩
  · try simp only [hA]
    obtain ⟨h1, h2⟩ := ih _ (hlt mid hi (by omega) le_rfl (by omega)) mid hi rfl
    exact ⟨h1, cov_cons mid h2⟩

theorem tripartiteQ_spec (lo hi : ℕ) :
    (tripartiteQ recover agrees lo hi).eval o
        = (tripartiteSearch recover (fun p => (agrees p).eval o) lo hi).1
      ∧ Covered o agrees ((tripartiteQ recover agrees lo hi).reads o)
        (tripartiteSearch recover (fun p => (agrees p).eval o) lo hi).2 := by
  unfold tripartiteQ tripartiteSearch
  simp only [eval_bind, reads_bind]
  rcases hA : (agrees lo).eval o with _ | _ | _
  · exact ⟨rfl, fun q hq => ⟨lo, by simp, by simpa using hq⟩⟩
  · exact ⟨rfl, fun q hq => ⟨lo, by simp, by simpa using hq⟩⟩
  simp only [eval_bind, reads_bind]
  rcases hB : (agrees hi).eval o with _ | _ | _
  · refine ⟨rfl, fun q hq => ?_⟩
    simp only [reads_pure, Finset.union_empty, Finset.mem_union] at hq
    rcases hq with hq | hq
    · exact ⟨lo, by simp, hq⟩
    · exact ⟨hi, by simp, hq⟩
  · obtain ⟨h1, h2⟩ := bisectQ_spec o recover agrees _ lo hi rfl
    refine ⟨h1, fun q hq => ?_⟩
    simp only [Finset.mem_union] at hq
    rcases hq with hq | hq | hq
    · exact ⟨lo, by simp, hq⟩
    · exact ⟨hi, by simp, hq⟩
    · obtain ⟨p', hp', hq'⟩ := h2 q hq
      exact ⟨p', by simp [hp'], hq'⟩
  · refine ⟨rfl, fun q hq => ?_⟩
    simp only [reads_pure, Finset.union_empty, Finset.mem_union] at hq
    rcases hq with hq | hq
    · exact ⟨lo, by simp, hq⟩
    · exact ⟨hi, by simp, hq⟩

end Search

/-! ## The probe -/

section Probe

open Random.Kappa

variable (read : FreeMonoid α → ARU) (T : DTree α) (x : FreeMonoid α) (st : ℕ → List Bool)

omit inst in
theorem eval_siftAgreesQ (p : ℕ) : (siftAgreesQ T x st p).eval read = siftAgrees read T x st p := by
  simp [siftAgreesQ, siftAgrees, DTree.eval_siftQ]

omit inst in
theorem eval_undecidedAtQ (p : ℕ) : (undecidedAtQ T x p).eval read = undecidedAt read T x p := by
  simp [undecidedAtQ, undecidedAt, DTree.eval_siftQ]

omit inst in
theorem eval_searchOutcomeQ (un : Option (α × List Bool)) (hi : ℕ) (res : SearchResult) :
    (searchOutcomeQ T x st un hi res).eval read = searchOutcome read T x st un hi res := by
  cases res with
  | edge p =>
    simp only [searchOutcomeQ, searchOutcome]
    rcases x.toList[p - 1]? with _ | c
    · simp
    · simp only [eval_map, DTree.eval_siftQ]
      rcases T.sift read (prefixOf x p) with t | z <;> rfl
  | _ => simp [searchOutcomeQ, searchOutcome, eval_undecidedAtQ]

theorem reads_siftAgreesQ_sub (p : ℕ) :
    (siftAgreesQ T x st p).reads read ⊆ siftReads read T (prefixOf x p) := by
  simpa [siftAgreesQ] using DTree.reads_siftQ_sub read T (prefixOf x p)

theorem reads_searchOutcomeQ_sub (un : Option (α × List Bool)) (hi : ℕ) (res : SearchResult) :
    (searchOutcomeQ T x st un hi res).reads read ⊆ posReads read T x (searchPos (res, []) hi) := by
  intro w hw
  refine mem_posReads.2 ?_
  cases res with
  | edge p =>
    simp only [searchOutcomeQ] at hw
    split at hw
    · simp only [reads_map] at hw
      exact ⟨p, by simp [searchPos], DTree.reads_siftQ_sub read T _ hw⟩
    · simp at hw
  | undecidedHi =>
    simp only [searchOutcomeQ, undecidedAtQ, reads_map] at hw
    exact ⟨hi, by simp [searchPos], DTree.reads_siftQ_sub read T _ hw⟩
  | undecidedMid p =>
    simp only [searchOutcomeQ, undecidedAtQ, reads_map] at hw
    exact ⟨p, by simp [searchPos], DTree.reads_siftQ_sub read T _ hw⟩
  | _ => simp [searchOutcomeQ] at hw

omit inst in
theorem eval_probeQ (E : Edges α) (k : ℕ) : (probeQ T E k x).eval read = probe read T E k x := by
  classical
  unfold probeQ probe
  simp only [eval_bind, DTree.eval_siftQ]
  rcases T.sift read (prefixOf x k) with a | z
  · simp only
    have hag : (fun p => (siftAgreesQ T x (fun p => (walk E a (x.toList.drop k)).getD (p - k) [])
        p).eval read) = siftAgrees read T x fun p => (walk E a (x.toList.drop k)).getD (p - k) [] :=
      funext fun p => eval_siftAgreesQ read T x _ p
    have ht := (tripartiteQ_spec (o := read) false (siftAgreesQ T x
      fun p => (walk E a (x.toList.drop k)).getD (p - k) []) k
      (k + (walk E a (x.toList.drop k)).length - 1)).1
    rw [hag] at ht
    rcases x.toList[k + (walk E a (x.toList.drop k)).length - 1]? with _ | c
    · simp only [eval_bind, ht, eval_searchOutcomeQ]
    · simp only [eval_bind, DTree.eval_siftQ]
      rcases T.sift read (prefixOf x (k + (walk E a (x.toList.drop k)).length - 1 + 1)) with t | z
      · simp only [eval_bind, ht, eval_searchOutcomeQ]
      · rfl
  · rfl

omit inst in
theorem eval_probeOrAgreeQ (E : Edges α) (k : ℕ) :
    (probeOrAgreeQ T E k x).eval read = probeOrAgree read T E k x := by
  unfold probeOrAgreeQ probeOrAgree
  split_ifs
  · rfl
  · exact eval_probeQ read T x E k

theorem reads_probeOrAgreeQ_sub (E : Edges α) (k : ℕ) :
    (probeOrAgreeQ T E k x).reads read ⊆ probeReads read T E k x := by
  unfold probeOrAgreeQ probeReads
  split_ifs with hk
  · simp
  intro w hw
  unfold probeQ at hw
  simp only [reads_bind, Finset.mem_union, DTree.eval_siftQ] at hw
  simp only [Finset.mem_union]
  rcases hw with hw | hw
  · exact Or.inl (DTree.reads_siftQ_sub read T _ hw)
  refine Or.inr ?_
  rcases hs : T.sift read (prefixOf x k) with a | z
  · rw [hs] at hw
    simp only at hw ⊢
    set st : ℕ → List Bool := fun p => (walk E a (x.toList.drop k)).getD (p - k) []
    set hi := k + (walk E a (x.toList.drop k)).length - 1
    have hag : (fun p => (siftAgreesQ T x st p).eval read) = siftAgrees read T x st :=
      funext fun p => eval_siftAgreesQ read T x st p
    obtain ⟨ht, hcov⟩ := tripartiteQ_spec (o := read) false (siftAgreesQ T x st) k hi
    rw [hag] at ht hcov
    have hsr : ∀ un, ∀ v ∈ (tripartiteQ false (siftAgreesQ T x st) k hi).reads read
        ∪ (searchOutcomeQ T x st un hi
          ((tripartiteQ false (siftAgreesQ T x st) k hi).eval read)).reads read,
        v ∈ posReads read T x (searchPos (tripartiteSearch false (siftAgrees read T x st) k hi)
          hi) := by
      intro un v hv
      rcases Finset.mem_union.1 hv with hv | hv
      · obtain ⟨p, hp, hvp⟩ := hcov v hv
        exact mem_posReads.2 ⟨p, List.mem_append_left _ hp,
          reads_siftAgreesQ_sub read T x st p hvp⟩
      · rw [ht] at hv
        obtain ⟨p, hp, hvp⟩ := mem_posReads.1 (reads_searchOutcomeQ_sub read T x st un hi _ hv)
        refine mem_posReads.2 ⟨p, ?_, hvp⟩
        simp only [searchPos, List.nil_append] at hp
        exact List.mem_append_right _ hp
    rcases hc : x.toList[hi]? with _ | c
    · rw [hc] at hw
      simp only [reads_bind] at hw
      exact hsr none w hw
    · rw [hc] at hw
      simp only [reads_bind, Finset.mem_union, DTree.eval_siftQ] at hw
      simp only [Finset.mem_union]
      rcases hw with hw | hw
      · exact Or.inl (DTree.reads_siftQ_sub read T _ hw)
      · refine Or.inr ?_
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

omit inst in
theorem eval_retargetAtQ (T' : DTree α) (p : List Bool) (E : Edges α) (e : List Bool × α) :
    (retargetAtQ T' p E e).eval read = retarget read T' p E e := by
  unfold retargetAtQ retarget
  split_ifs
  · rfl
  rcases E e with _ | ⟨t, w⟩
  · rfl
  · simp only
    split_ifs
    · simp [DTree.eval_siftQ]
    · rfl

omit inst in
theorem eval_retargetQ [DecidableEq α] (T' : DTree α) (p : List Bool) (E : Edges α) :
    ∀ es : List (List Bool × α), (retargetQ T' p E es).eval read
      = fun e => if e ∈ es then retarget read T' p E e else none
  | [] => by simp [retargetQ]
  | e :: es => by
    funext e'
    simp only [retargetQ, eval_bind, eval_map, eval_retargetQ T' p E es, eval_retargetAtQ,
      List.mem_cons]
    by_cases h : e' = e
    · subst h; simp
    · simp [Function.update_of_ne h, h]

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

/-- Every learned edge is at a leaf. -/
def EdgesAtLeaves (s : Level3State α) : Prop :=
  ∀ e t w, s.edges e = some (t, w) → e.1 ∈ s.tree.leaves

omit inst in
theorem eval_splitLeafQ {s : Level3State α} (hs : EdgesAtLeaves s) (p : List Bool) (c : α)
    (t t₀ : List Bool) : (splitLeafQ s p c t t₀).eval read = splitLeaf read s p c t t₀ := by
  unfold splitLeafQ splitLeaf
  rw [eval_map, eval_retargetQ]
  congr 1
  funext e
  split_ifs with he
  · rfl
  · unfold retarget
    split_ifs
    · rfl
    rcases hE : s.edges e with _ | ⟨t', w⟩
    · rfl
    · exact absurd (Finset.mem_toList.2 (Finset.mem_product.2
        ⟨List.mem_toFinset.2 (hs e t' w hE), Finset.mem_univ _⟩)) he

omit inst in
theorem eval_applyProbeQ {s : Level3State α} (hs : EdgesAtLeaves s) (x : FreeMonoid α) :
    (applyProbeQ C s x).eval read = applyProbe read C s x := by
  unfold applyProbeQ applyProbe countProbe
  rw [eval_bind, eval_probeOrAgreeQ]
  rcases probeOrAgree read s.tree s.edges C.walkStart x with _ | ⟨p, c, u, t⟩ | _ | _ | _ <;>
    try rfl
  simp only
  rcases s.edges (p, c) with _ | ⟨t₀, w⟩
  · rfl
  · simp [eval_splitLeafQ read hs]

theorem reads_applyProbeQ_sub {s : Level3State α} (hs : EdgesAtLeaves s) (x : FreeMonoid α) :
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

theorem eval_runProbesQ (X : List (FreeMonoid α)) :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α), (∀ y ∈ xs, y ∈ X) → RunInv read C X s →
      (runProbesQ C s xs).eval read
        = if (runProbes read C s xs).1.tree.leaves.length ≤ C.leafCap
          then some (runProbes read C s xs) else none
  | [], _, _, _ => by simp only [runProbesQ, runProbes, eval_pure]; exact if_congr Iff.rfl rfl rfl
  | x :: xs, s, hX, hs => by
    have hk : EdgesAtLeaves s := fun e t w h => hs.keys e t w h
    by_cases hc : s.tree.leaves.length ≤ C.leafCap
    · rcases hap : applyProbe read C s x with s' | e
      · have e1 : runProbes read C s (x :: xs) = runProbes read C s' xs := by
          simp [runProbes, hap]
        have e2 : (runProbesQ C s (x :: xs)).eval read = (runProbesQ C s' xs).eval read := by
          simp [runProbesQ, hc, eval_applyProbeQ read C hk x, hap]
        rw [e1, e2]
        exact eval_runProbesQ X xs s' (fun y hy => hX y (by simp [hy]))
          (runInv_step read C (hX x (by simp)) hs hap).1
      · have e1 : runProbes read C s (x :: xs) = (s, e) := by simp [runProbes, hap]
        have e2 : (runProbesQ C s (x :: xs)).eval read = some (s, e) := by
          simp [runProbesQ, hc, eval_applyProbeQ read C hk x, hap]
        rw [e1, e2, if_pos hc]
    · have hle := leaves_le_runProbes read C (x :: xs) s hX hs
      simp only [runProbesQ, if_neg hc, eval_pure]
      rw [if_neg (by omega)]

omit inst in
theorem eval_level3Q {T : ℕ} (xs : Fin T → FreeMonoid α) :
    (level3Q C xs).eval read
      = if (indecisiveLStarLevel3 C read xs).hypothesis.tree.leaves.length ≤ C.leafCap
        then some (indecisiveLStarLevel3 C read xs) else none := by
  classical
  unfold level3Q indecisiveLStarLevel3
  rw [eval_map, eval_runProbesQ read C (List.ofFn xs) (List.ofFn xs) initialState (fun _ h => h)
    (runInv_start read C _)]
  split_ifs <;> rfl

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

/-- With `leafCap ≤ n + 2`, a run from `s` over probes at most `L` long asks at most
`probeSifts L (n + 1)` strings in each probe and `(n + 2)|Σ|(n + 2)` in each split, of which there
are at most `leafCap + 1 - |leaves s|`. -/
theorem card_reads_runProbesQ_le {L : ℕ} (n : ℕ) (hn : C.leafCap ≤ n + 2)
    (X : List (FreeMonoid α)) :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α), (∀ y ∈ xs, y ∈ X) →
      (∀ y ∈ xs, y.toList.length ≤ L) → RunInv read C X s →
      ((runProbesQ C s xs).reads read).card
        ≤ xs.length * (probeSifts L * (n + 1))
          + (C.leafCap + 1 - s.tree.leaves.length) * ((n + 2) * Fintype.card α * (n + 2))
  | [], _, _, _, _ => by simp [runProbesQ]
  | x :: xs, s, hX, hL, hs => by
    by_cases hc : s.tree.leaves.length ≤ C.leafCap
    swap
    · simp [runProbesQ, if_neg hc]
    have hk : EdgesAtLeaves s := fun e t w h => hs.keys e t w h
    have hxL : x.toList.length ≤ L := hL x (by simp)
    have hmid : (mids s.tree).card ≤ n + 1 := by have := mids_card s.tree; omega
    have hprobe : (probeReads read s.tree s.edges C.walkStart x).card ≤ probeSifts L * (n + 1) :=
      (probeReads_card_mids read hxL).trans (Nat.mul_le_mul_left _ hmid)
    have hsplit : (splitReads read C s x).card ≤ (n + 2) * Fintype.card α * (n + 2) :=
      (splitReads_card_le C read s x).trans
        (Nat.mul_le_mul (Nat.mul_le_mul_right _ (by omega)) (by omega))
    have hhead : ((applyProbeQ C s x).reads read).card
        ≤ (probeReads read s.tree s.edges C.walkStart x).card + (splitReads read C s x).card :=
      (Finset.card_le_card (reads_applyProbeQ_sub read C hk x)).trans (Finset.card_union_le _ _)
    have hX' : ∀ y ∈ xs, y ∈ X := fun y hy => hX y (by simp [hy])
    have hL' : ∀ y ∈ xs, y.toList.length ≤ L := fun y hy => hL y (by simp [hy])
    simp only [runProbesQ, if_pos hc, reads_bind, eval_applyProbeQ read C hk x, List.length_cons]
    refine (Finset.card_union_le _ _).trans ?_
    rcases splitReads_eq_empty_or read C hs x with h0 | ⟨s', hap, hlen⟩
    · rw [h0, Finset.card_empty, add_zero] at hhead
      rcases hap : applyProbe read C s x with s' | e
      · have hrec := card_reads_runProbesQ_le n hn X xs s' hX' hL'
          (runInv_step read C (hX x (by simp)) hs hap).1
        have hle := leaves_le_step read C hs hap
        simp only
        generalize probeSifts L * (n + 1) = P at hprobe hrec ⊢
        generalize (n + 2) * Fintype.card α * (n + 2) = R at hrec ⊢
        have hR : (C.leafCap + 1 - s'.tree.leaves.length) * R
            ≤ (C.leafCap + 1 - s.tree.leaves.length) * R := Nat.mul_le_mul_right _ (by omega)
        calc _ ≤ P + (xs.length * P + (C.leafCap + 1 - s'.tree.leaves.length) * R) :=
              Nat.add_le_add (hhead.trans hprobe) hrec
          _ ≤ P + (xs.length * P + (C.leafCap + 1 - s.tree.leaves.length) * R) := by omega
          _ = (xs.length + 1) * P + (C.leafCap + 1 - s.tree.leaves.length) * R := by ring
      · simp only [reads_pure, Finset.card_empty, add_zero]
        calc _ ≤ probeSifts L * (n + 1) := hhead.trans hprobe
          _ ≤ (xs.length + 1) * (probeSifts L * (n + 1)) :=
            Nat.le_mul_of_pos_left _ (Nat.succ_pos _)
          _ ≤ _ := Nat.le_add_right _ _
    · rw [hap]
      have hrec := card_reads_runProbesQ_le n hn X xs s' hX' hL'
        (runInv_step read C (hX x (by simp)) hs hap).1
      simp only
      obtain ⟨m, hm⟩ : ∃ m, C.leafCap + 1 - s.tree.leaves.length = m + 1 :=
        ⟨C.leafCap - s.tree.leaves.length, by omega⟩
      rw [hlen, show C.leafCap + 1 - (s.tree.leaves.length + 1) = m by omega] at hrec
      rw [hm]
      generalize probeSifts L * (n + 1) = P at hprobe hrec ⊢
      generalize (n + 2) * Fintype.card α * (n + 2) = R at hsplit hrec ⊢
      calc _ ≤ (P + R) + (xs.length * P + m * R) :=
            Nat.add_le_add (hhead.trans (Nat.add_le_add hprobe hsplit)) hrec
        _ = (xs.length + 1) * P + (m + 1) * R := by ring

/-- With `leafCap ≤ n + 2`, level 3's run over `T` probes at most `L` long asks at most
`probeSifts L (n + 1)` strings in each probe and `(n + 2)|Σ|(n + 2)` in each of at most `n + 1`
splits. -/
theorem card_reads_level3Q_le {L T : ℕ} (n : ℕ) (hn : C.leafCap ≤ n + 2)
    (xs : Fin T → FreeMonoid α) (hL : ∀ i, (xs i).toList.length ≤ L) :
    ((level3Q C xs).reads read).card
      ≤ T * (probeSifts L * (n + 1)) + (n + 1) * ((n + 2) * Fintype.card α * (n + 2)) := by
  unfold level3Q
  rw [reads_map]
  have h := card_reads_runProbesQ_le read C n hn (List.ofFn xs) (List.ofFn xs) initialState
    (fun _ h => h) (fun y hy => by
      obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hy
      exact hL i) (runInv_start read C _)
  have h0 : (initialState : Level3State α).tree.leaves.length = 2 := by
    simp [initialState, newHypothesis, DTree.leaves]
  rw [List.length_ofFn, h0] at h
  exact h.trans (Nat.add_le_add_left (Nat.mul_le_mul_right _ (by omega)) _)

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
