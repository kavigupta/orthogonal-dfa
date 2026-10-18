import OrthoDFA.Proofs.Level5Law
import OrthoDFA.Proofs.SeqRate

/-!
# A round's steps

A probe either keeps the hypothesis, adding to its counts, or learns an edge, and a step that goes
on then splits at a contradiction or keeps what the probe recorded. A learned edge or a split is a
fresh hypothesis with the next version, and every edge stays pointed at a leaf. A contradiction is
a search at a learned edge, so within a version every search but the last stops at an undecided
middle.

Each learned edge lowers the unlearned edges out of the leaves, and each split adds a leaf, so a
round goes through fewer than `versionBound` hypotheses (`version_lt`). At knobs where success fires
at `nEnd` probes with at most `hS` searches and the middles test at `hS + 1` middles, every
hypothesis ends the round or changes within `nEnd` probes, so a round over
`versionBound · nEnd` probes ends (`runEnds_of_budget`).
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Probe

variable (cut : FreeMonoid α → Option Bool) (T : DTree α) (edges : Edges α) (k : ℕ)
  (x : FreeMonoid α)

/-- The middles a probe adds: one where it stops at an undecided middle, none otherwise. -/
theorem middleHarvestBy_length :
    (middleHarvestBy cut T edges k x).length
      = match probeOrAgree (ofCut cut) T edges k x with
        | .undecided _ false => 1
        | _ => 0 := by
  unfold middleHarvestBy
  rcases ho : probeOrAgree (ofCut cut) T edges k x with _ | _ | _ | _ | ⟨zs, _ | _⟩ <;> try rfl
  obtain ⟨-, b, -, -, -, rfl⟩ := probe_middle _ T edges k x ho
  rfl

end Probe

section Steps

variable (C : Level5Settings) (cut : FreeMonoid α → Option Bool)

theorem setEdge_eq (s : Level5State α) (p : List Bool) (c : α) (e) (q : List Bool) (c' : α) :
    (s.setEdge p c e).edges (q, c') = if (q, c') = (p, c) then some e else s.edges (q, c') := by
  unfold Level5State.setEdge freshState
  simp only [Function.update_apply]

open scoped Classical in
/-- A probe keeps the hypothesis, adding to its counts, or learns an unlearned edge out of the
leaf its prefix sifts to. -/
theorem recordProbe_spec (s : Level5State α) (x : FreeMonoid α) :
    recordProbe C cut s x = { s.countProbe cut C.walkStart x with
        searches := s.searches
          + if (probeOrAgree (ofCut cut) s.tree s.edges C.walkStart x).IsSearch then 1 else 0 }
    ∨ ∃ u c p t, probeOrAgree (ofCut cut) s.tree s.edges C.walkStart x = .unlearnedEdge p c u t
      ∧ s.tree.sift (ofCut cut) u = .inl p ∧ x.toList[u.toList.length]? = some c
      ∧ s.tree.sift (ofCut cut) (u * FreeMonoid.of c) = .inl t ∧ s.edges (p, c) = none
      ∧ recordProbe C cut s x = (s.countProbe cut C.walkStart x).setEdge p c (t, u) := by
  unfold recordProbe
  rcases ho : probeOrAgree (ofCut cut) s.tree s.edges C.walkStart x with
    _ | _ | ⟨p, c, u, t⟩ | _ | ⟨zs, _ | _⟩ <;> simp only []
  · left; simp [Outcome.IsSearch]; rfl
  · left; simp [Outcome.IsSearch]; rfl
  · right
    obtain ⟨-, j, -, -, hjx, rfl, hc, -, -, hp, ht, hE⟩ := probe_unlearnedEdge _ _ _ _ _ ho
    have hlen : (prefixOf x j).toList.length = j := by simp [prefixOf]; omega
    refine ⟨prefixOf x j, c, p, t, rfl, hp, by rw [hlen]; exact hc, ?_, hE, rfl⟩
    rw [← prefixOf_succ hc]; exact ht
  · left; simp [Outcome.IsSearch]; rfl
  · left; simp [Outcome.IsSearch]; rfl
  · left; simp [Outcome.IsSearch]; rfl

theorem countProbe_hyp (s : Level5State α) (k : ℕ) (x : FreeMonoid α) :
    (s.countProbe cut k x).tree = s.tree ∧ (s.countProbe cut k x).edges = s.edges
      ∧ (s.countProbe cut k x).version = s.version := ⟨rfl, rfl, rfl⟩

theorem recordProbe_tree (s : Level5State α) (x : FreeMonoid α) :
    (recordProbe C cut s x).tree = s.tree := by
  rcases recordProbe_spec C cut s x with h | ⟨u, c, p, t, -, -, -, -, -, h⟩ <;> rw [h] <;> rfl

/-- A probe's record leaves the hypothesis and the counts but the searches as the probe's. -/
theorem recordProbe_of_record (s : Level5State α) (x : FreeMonoid α) {p : List Bool} {c : α}
    {t : List Bool} {sp : FreeMonoid α}
    (h : recordBy cut C.walkStart (s.tree, s.edges) x = some ((p, c, t), sp)) :
    recordProbe C cut s x
      = { s.countProbe cut C.walkStart x with searches := s.searches + 1 } := by
  unfold recordProbe
  rw [(recordBy_eq_some cut).1 h]
  rfl

/-- A split at a learned edge is a fresh hypothesis, the leaf split. -/
theorem splitState_of_some (s : Level5State α) {p : List Bool} {c : α} {t t₀ : List Bool}
    {w : FreeMonoid α} (he : s.edges (p, c) = some (t₀, w)) :
    splitState cut s p c t = freshState
      (s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p)
      (retarget (ofCut cut)
        (s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p) p s.edges)
      (s.version + 1) := by
  unfold splitState
  rw [he]
  rfl

/-- A step that goes on keeps what the probe recorded, or splits at its contradiction. -/
theorem applyProbe_inl {s s' : Level5State α} {x : FreeMonoid α}
    (h : applyProbe C cut s x = .inl s') :
    runTests C (recordProbe C cut s x) = none
      ∧ ((recordBy cut C.walkStart (s.tree, s.edges) x = none ∧ s' = recordProbe C cut s x)
        ∨ ∃ p c t sp, recordBy cut C.walkStart (s.tree, s.edges) x = some ((p, c, t), sp)
          ∧ s' = splitState cut (recordProbe C cut s x) p c t
          ∧ s'.tree.leaves.length ≤ C.maxLeaves) := by
  unfold applyProbe at h
  simp only [] at h
  rcases hr : runTests C (recordProbe C cut s x) with _ | e <;> rw [hr] at h
  · refine ⟨rfl, ?_⟩
    simp only [] at h
    rcases hrec : recordBy cut C.walkStart (s.tree, s.edges) x with _ | ⟨⟨p, c, t⟩, sp⟩ <;>
      rw [hrec] at h
    · simp only [Sum.inl.injEq] at h
      exact .inl ⟨rfl, h.symm⟩
    · simp only [] at h
      split_ifs at h with hL
      simp only [Sum.inl.injEq] at h
      subst h
      exact .inr ⟨p, c, t, sp, rfl, rfl, not_lt.1 hL⟩
  · simp at h

/-- A step that ends: a test fired at what the probe recorded, or a split past `maxLeaves`
leaves. -/
theorem applyProbe_inr {s s' : Level5State α} {x : FreeMonoid α} {e : Level5Ending α}
    (h : applyProbe C cut s x = .inr (e, s')) :
    (runTests C (recordProbe C cut s x) = some e ∧ s' = recordProbe C cut s x)
      ∨ (e = .tooBig ∧ ∃ p c t sp, recordBy cut C.walkStart (s.tree, s.edges) x
          = some ((p, c, t), sp) ∧ s' = splitState cut (recordProbe C cut s x) p c t
          ∧ C.maxLeaves < s'.tree.leaves.length) := by
  unfold applyProbe at h
  simp only [] at h
  rcases hr : runTests C (recordProbe C cut s x) with _ | e' <;> rw [hr] at h
  · simp only [] at h
    rcases hrec : recordBy cut C.walkStart (s.tree, s.edges) x with _ | ⟨⟨p, c, t⟩, sp⟩ <;>
      rw [hrec] at h
    · simp at h
    · simp only [] at h
      split_ifs at h with hL
      simp only [Sum.inr.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact .inr ⟨rfl, p, c, t, sp, rfl, rfl, hL⟩
  · simp only [Sum.inr.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact .inl ⟨rfl, rfl⟩

theorem recordProbe_version (s : Level5State α) (x : FreeMonoid α) :
    s.version ≤ (recordProbe C cut s x).version := by
  rcases recordProbe_spec C cut s x with h | ⟨u, c, p, t, -, -, -, -, -, h⟩ <;> rw [h]
  · exact le_rfl
  · exact Nat.le_succ _

theorem splitState_version (s : Level5State α) (p : List Bool) (c : α) (t : List Bool) :
    s.version ≤ (splitState cut s p c t).version := by
  unfold splitState
  split
  · exact Nat.le_succ _
  · exact le_rfl

/-- A step that goes on keeps or raises the version. -/
theorem applyProbe_version {s s' : Level5State α} {x : FreeMonoid α}
    (h : applyProbe C cut s x = .inl s') : s.version ≤ s'.version := by
  rcases (applyProbe_inl C cut h).2 with ⟨-, rfl⟩ | ⟨p, c, t, sp, -, rfl, -⟩
  · exact recordProbe_version C cut s x
  · exact (recordProbe_version C cut s x).trans (splitState_version cut _ p c t)

end Steps

section Into

variable (C : Level5Settings) (cut : FreeMonoid α → Option Bool)

theorem setEdge_into {s : Level5State α} (he : EdgesInto s.tree s.edges) {p : List Bool} {c : α}
    {t : List Bool} {w : FreeMonoid α} (ht : t ∈ s.tree.leaves) :
    EdgesInto (s.setEdge p c (t, w)).tree (s.setEdge p c (t, w)).edges := by
  intro q e t' w' h
  rw [setEdge_eq] at h
  split_ifs at h
  · simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, -⟩ := h
    exact ht
  · exact he _ _ _ _ h

theorem recordProbe_into {s : Level5State α} (x : FreeMonoid α) (he : EdgesInto s.tree s.edges) :
    EdgesInto (recordProbe C cut s x).tree (recordProbe C cut s x).edges := by
  rcases recordProbe_spec C cut s x with h | ⟨u, c, p, t, -, -, -, ht, -, h⟩ <;> rw [h]
  · exact he
  · exact setEdge_into (s := s.countProbe cut C.walkStart x) he
      (DTree.sift_inl_mem _ _ _ _ ht)

theorem splitState_into {s : Level5State α} (he : EdgesInto s.tree s.edges) (p : List Bool) (c : α)
    (t : List Bool) : EdgesInto (splitState cut s p c t).tree (splitState cut s p c t).edges := by
  rcases he0 : s.edges (p, c) with _ | ⟨t₀, w₀⟩
  · unfold splitState; rw [he0]; exact he
  rw [splitState_of_some cut s he0]
  set T' := s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p
  intro q e t' w h
  simp only [freshState, retarget] at h
  split_ifs at h
  rcases hq : s.edges (q, e) with _ | ⟨s', w'⟩ <;> rw [hq] at h
  · simp at h
  · simp only [] at h
    split_ifs at h with hs'
    · rcases hsift : T'.sift (ofCut cut) (w' * FreeMonoid.of e) with s'' | b <;> rw [hsift] at h
      · simp only [Sum.elim_inl, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, -⟩ := h
        exact DTree.sift_inl_mem _ _ _ _ hsift
      · simp at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, -⟩ := h
      exact DTree.mem_leaves_splitAt_of_ne _ _ _ _ (he _ _ _ _ hq) hs'

/-- A step that goes on keeps every edge pointed at a leaf. -/
theorem applyProbe_into {s s' : Level5State α} {x : FreeMonoid α} (he : EdgesInto s.tree s.edges)
    (h : applyProbe C cut s x = .inl s') : EdgesInto s'.tree s'.edges := by
  rcases (applyProbe_inl C cut h).2 with ⟨-, rfl⟩ | ⟨p, c, t, sp, -, rfl, -⟩
  · exact recordProbe_into C cut x he
  · exact splitState_into cut (recordProbe_into C cut x he) p c t

end Into

section Versions

variable (C : Level5Settings) (cut : FreeMonoid α → Option Bool)

open scoped Classical in
/-- The unlearned edges out of the leaves. -/
noncomputable def unlearned (s : Level5State α) : ℕ :=
  ((s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α)).filter
    fun e => s.edges e = none).card

/-- The potential every change of hypothesis lowers. -/
noncomputable def verPot (s : Level5State α) : ℕ :=
  (C.maxLeaves - s.tree.leaves.length) * (C.maxLeaves * Fintype.card α + 1) + unlearned s

theorem unlearned_le (s : Level5State α) : unlearned s ≤ s.tree.leaves.length * Fintype.card α := by
  classical
  unfold unlearned
  refine (Finset.card_filter_le _ _).trans ?_
  rw [Finset.card_product, Finset.card_univ]
  exact Nat.mul_le_mul_right _ (List.toFinset_card_le _)

theorem unlearned_setEdge (s : Level5State α) {p : List Bool} {c : α} {e : List Bool × FreeMonoid α}
    (hp : p ∈ s.tree.leaves) (hn : s.edges (p, c) = none) :
    unlearned (s.setEdge p c e) + 1 = unlearned s := by
  classical
  unfold unlearned
  have hset : ((s.setEdge p c e).tree.leaves.toFinset ×ˢ (Finset.univ : Finset α)).filter
      (fun e' => (s.setEdge p c e).edges e' = none)
      = (((s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α)).filter
        fun e' => s.edges e' = none)).erase (p, c) := by
    ext ⟨q, c'⟩
    simp only [Finset.mem_filter, Finset.mem_product, List.mem_toFinset, Finset.mem_univ,
      and_true, Finset.mem_erase, ne_eq, setEdge_eq]
    rw [show (s.setEdge p c e).tree = s.tree from rfl]
    by_cases h : (q, c') = (p, c)
    · simp [h]
    · simp [h]
  rw [hset, Finset.card_erase_of_mem (by simp [hp, hn])]
  have : 0 < ((s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α)).filter
      fun e' => s.edges e' = none).card :=
    Finset.card_pos.2 ⟨(p, c), by simp [hp, hn]⟩
  omega

/-- What a reached state keeps: edges into its leaves, at most `maxLeaves` leaves, and its version
and potential at most the first state's potential. -/
def VerInv (s : Level5State α) : Prop :=
  EdgesInto s.tree s.edges ∧ s.tree.leaves.length ≤ C.maxLeaves
    ∧ s.version + verPot C s ≤ verPot C (initialState : Level5State α)

theorem verInv_start (h : 2 ≤ C.maxLeaves) : VerInv C (initialState : Level5State α) := by
  refine ⟨fun _ _ _ _ h => by simp [initialState, freshState] at h, ?_, ?_⟩
  · simp [initialState, freshState, DTree.leaves]; omega
  · simp [initialState, freshState]

theorem verInv_step {s s' : Level5State α} {x : FreeMonoid α} (hs : VerInv C s)
    (h : applyProbe C cut s x = .inl s') : VerInv C s' := by
  obtain ⟨he, hL, hv⟩ := hs
  refine ⟨applyProbe_into C cut he h, ?_⟩
  rcases (applyProbe_inl C cut h).2 with ⟨-, rfl⟩ | ⟨p, c, t, sp, hrec, rfl, hL'⟩
  · rcases recordProbe_spec C cut s x with h1 | ⟨u, c, p, t, -, hp, -, -, hn, h1⟩ <;> rw [h1]
    · exact ⟨hL, hv⟩
    · have hp' := DTree.sift_inl_mem _ _ _ _ hp
      have := unlearned_setEdge (s.countProbe cut C.walkStart x) (e := (t, u)) hp' hn
      refine ⟨hL, ?_⟩
      unfold verPot at hv ⊢
      simp only [Level5State.setEdge, freshState] at this ⊢
      change s.version + 1 + ((C.maxLeaves - s.tree.leaves.length)
        * (C.maxLeaves * Fintype.card α + 1) + _) ≤ _
      have e1 : unlearned (s.countProbe cut C.walkStart x) = unlearned s := rfl
      omega
  · refine ⟨hL', ?_⟩
    obtain ⟨p₀, hk, hpm, -, t₀, w, hpe, -⟩ := recordBy_spec cut hrec
    have hp : p ∈ s.tree.leaves := walk_mem_leaves he (DTree.sift_inl_mem _ _ _ _ hk) _ p hpm
    have hrec' := recordProbe_of_record C cut s x hrec
    have hpe' : (recordProbe C cut s x).edges (p, c) = some (t₀, w) := by rw [hrec']; exact hpe
    rw [splitState_of_some cut _ hpe'] at hL' ⊢
    rw [hrec'] at hL' ⊢
    simp only [Level5State.countProbe] at hL' ⊢
    set T' := s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p
    set E' : Edges α := retarget (ofCut cut) T' p s.edges
    have hlen : T'.leaves.length = s.tree.leaves.length + 1 :=
      DTree.length_leaves_splitAt _ s.tree p hp
    have hu := unlearned_le (freshState T' E' (s.version + 1))
    simp only [freshState] at hL' hu ⊢
    rw [hlen] at hL' hu
    unfold verPot at hv ⊢
    simp only [freshState]
    rw [hlen]
    have hA : (s.tree.leaves.length + 1) * Fintype.card α ≤ C.maxLeaves * Fintype.card α :=
      Nat.mul_le_mul_right _ hL'
    obtain ⟨m, hm⟩ : ∃ m, C.maxLeaves - s.tree.leaves.length = m + 1 :=
      ⟨C.maxLeaves - s.tree.leaves.length - 1, by omega⟩
    have hm' : C.maxLeaves - (s.tree.leaves.length + 1) = m := by omega
    rw [hm] at hv
    rw [hm']
    have e1 : (m + 1) * (C.maxLeaves * Fintype.card α + 1)
        = m * (C.maxLeaves * Fintype.card α + 1) + (C.maxLeaves * Fintype.card α + 1) := by ring
    generalize m * (C.maxLeaves * Fintype.card α + 1) = mK at e1 hv ⊢
    generalize (s.tree.leaves.length + 1) * Fintype.card α = b at hA hu
    generalize C.maxLeaves * Fintype.card α = LA at hA e1 hv ⊢
    omega

theorem version_lt {s : Level5State α} (hs : VerInv C s) (h : 2 ≤ C.maxLeaves) :
    s.version < versionBound C (Fintype.card α) := by
  obtain ⟨-, -, hv⟩ := hs
  have hu := unlearned_le (initialState : Level5State α)
  have h2 : (initialState : Level5State α).tree.leaves.length = 2 := by
    simp [initialState, freshState, DTree.leaves]
  unfold verPot at hv
  rw [h2] at hv hu
  unfold versionBound
  set K := C.maxLeaves * Fintype.card α + 1
  obtain ⟨m, hm⟩ : ∃ m, C.maxLeaves = m + 2 := ⟨C.maxLeaves - 2, by omega⟩
  have hK : 2 * Fintype.card α < 2 * K := by simp only [K]; nlinarith
  rw [hm, show m + 2 - 2 = m by omega] at hv
  rw [hm]
  have : (m + 2) * K = m * K + 2 * K := by ring
  omega

end Versions

section Stretch

variable (C : Level5Settings) (cut : FreeMonoid α → Option Bool)

open scoped Classical in
theorem runTests_ne_tooBig (s : Level5State α) : runTests C s ≠ some .tooBig := by
  unfold runTests
  split_ifs <;> simp

open scoped Classical in
theorem runTests_none {s : Level5State α} (h : runTests C s = none) :
    ¬ (C.searchShare * s.probes ≤ s.searches ∧ C.firstLook ≤ s.searches
        ∧ seqRate C.middleUndecidedRate C.middleAltRate C.testLevel s.searches s.middles.length)
      ∧ ¬ seqRate C.disagreementRate C.disagreementAltRate C.testLevel s.probes s.searches := by
  unfold runTests at h
  split_ifs at h with h1 h2 h3 h4
  exact ⟨h3, h4⟩

/-- Within a version, fewer than `nEnd` probes and at most `hS` searches, every one stopping at an
undecided middle. -/
def StretchOK (nEnd hS : ℕ) (s : Level5State α) : Prop :=
  s.probes < nEnd ∧ s.searches ≤ hS ∧ s.middles.length = s.searches

theorem stretchOK_fresh {nEnd hS : ℕ} (hn : 1 ≤ nEnd) (T : DTree α) (E : Edges α) (v : ℕ) :
    StretchOK nEnd hS (freshState T E v) :=
  ⟨by show 0 < nEnd; omega, Nat.zero_le _, rfl⟩

theorem stretchOK_start {nEnd hS : ℕ} (hn : 1 ≤ nEnd) :
    StretchOK nEnd hS (initialState : Level5State α) := stretchOK_fresh hn _ _ _

open scoped Classical in
/-- A step that goes on changes the hypothesis, its counts afresh, or adds a probe, keeping the
version within `nEnd` probes and `hS` searches: past those, success or the middles test fires. -/
theorem stretchOK_step {nEnd hS : ℕ} (hSq : C.SeqOK) (hn : 1 ≤ nEnd) (hSn : hS ≤ nEnd)
    (hφ0 : 0 ≤ C.searchShare) (hφ : C.searchShare * nEnd ≤ hS + 1) (hn₀ : C.firstLook ≤ hS + 1)
    (hpt : seqRate C.middleUndecidedRate C.middleAltRate C.testLevel (hS + 1) (hS + 1))
    (hsucc : seqRate C.disagreementRate C.disagreementAltRate C.testLevel nEnd hS)
    {s s' : Level5State α} {x : FreeMonoid α} (hs : StretchOK nEnd hS s) (he : EdgesInto s.tree s.edges)
    (h : applyProbe C cut s x = .inl s') :
    StretchOK nEnd hS s' ∧ ((s'.version = s.version + 1 ∧ s'.probes = 0)
      ∨ (s'.version = s.version ∧ s'.probes = s.probes + 1)) := by
  obtain ⟨hpr, hse, hmi⟩ := hs
  obtain ⟨hnone, hcases⟩ := applyProbe_inl C cut h
  rcases hcases with ⟨hrec, rfl⟩ | ⟨p, c, t, sp, hrec, rfl, -⟩
  · rcases recordProbe_spec C cut s x with h1 | ⟨u, c, p, t, -, -, -, -, -, h1⟩
    · have hne : ∀ p c u t, probeOrAgree (ofCut cut) s.tree s.edges C.walkStart x
          ≠ .wrongEdge p c u t := by
        intro p c u t ho
        rw [← recordBy_eq_some cut, hrec] at ho
        simp at ho
      have hmid := middleHarvestBy_length cut s.tree s.edges C.walkStart x
      set d := (middleHarvestBy cut s.tree s.edges C.walkStart x).length with hd
      have hsearch : (if (probeOrAgree (ofCut cut) s.tree s.edges C.walkStart x).IsSearch then 1 else 0) = d := by
        rw [hmid]
        rcases ho : probeOrAgree (ofCut cut) s.tree s.edges C.walkStart x with
          _ | ⟨p, c, u, t⟩ | _ | _ | ⟨zs, _ | _⟩
        · simp [Outcome.IsSearch]
        · exact absurd ho (hne p c u t)
        · simp [Outcome.IsSearch]
        · simp [Outcome.IsSearch]
        · simp [Outcome.IsSearch]
        · simp [Outcome.IsSearch]
      have hd1 : d ≤ 1 := by
        rw [hmid]; split <;> omega
      have hP : (recordProbe C cut s x).probes = s.probes + 1 := by rw [h1]; rfl
      have hSe : (recordProbe C cut s x).searches = s.searches + d := by rw [h1, ← hsearch]
      have hM : (recordProbe C cut s x).middles.length = s.middles.length + d := by
        rw [h1]; simp [Level5State.countProbe, hd]
      have hV : (recordProbe C cut s x).version = s.version := by rw [h1]; rfl
      obtain ⟨hmn, hsn⟩ := runTests_none C hnone
      rw [hP, hSe, hM] at hmn
      rw [hP, hSe] at hsn
      have hprobes : s.probes + 1 ≤ nEnd := hpr
      have hS' : s.searches + d ≤ hS := by
        by_contra hc
        have heq : s.searches + d = hS + 1 := by omega
        apply hmn
        refine ⟨?_, by rw [heq]; exact hn₀, ?_⟩
        · rw [heq]
          have : (((s.probes + 1 : ℕ)) : ℝ) ≤ nEnd := by exact_mod_cast hprobes
          calc C.searchShare * ((s.probes + 1 : ℕ) : ℝ) ≤ C.searchShare * nEnd :=
                mul_le_mul_of_nonneg_left this hφ0
            _ ≤ _ := by push_cast; linarith
        · rw [heq, show s.middles.length + d = hS + 1 by omega]
          exact hpt
      have hpl : s.probes + 1 < nEnd := by
        by_contra hc
        have heq : s.probes + 1 = nEnd := by omega
        apply hsn
        rw [heq]
        exact seqRate_anti_down hSq.εd_pos hSq.εd_lt hSq.εd_alt_pos.le hSq.εd_alt.le hsucc hS' hSn
      refine ⟨⟨?_, ?_, ?_⟩, .inr ⟨hV, hP⟩⟩
      · rw [hP]; exact hpl
      · rw [hSe]; exact hS'
      · rw [hM, hSe]; omega
    · rw [h1]
      exact ⟨stretchOK_fresh hn _ _ _, .inl ⟨rfl, rfl⟩⟩
  · obtain ⟨-, -, -, -, t₀, w, hpe, -⟩ := recordBy_spec cut hrec
    have hrec' := recordProbe_of_record C cut s x hrec
    have hpe' : (recordProbe C cut s x).edges (p, c) = some (t₀, w) := by rw [hrec']; exact hpe
    rw [splitState_of_some cut _ hpe']
    refine ⟨stretchOK_fresh hn _ _ _, .inl ⟨?_, rfl⟩⟩
    simp only [freshState, hrec']
    rfl

/-- A round from a state reached at the budget's pace ends within its draws. -/
theorem runEnds_of_budget {nEnd hS : ℕ} (hSq : C.SeqOK) (hn : 1 ≤ nEnd) (hSn : hS ≤ nEnd)
    (hφ0 : 0 ≤ C.searchShare) (hφ : C.searchShare * nEnd ≤ hS + 1) (hn₀ : C.firstLook ≤ hS + 1)
    (hpt : seqRate C.middleUndecidedRate C.middleAltRate C.testLevel (hS + 1) (hS + 1))
    (hsucc : seqRate C.disagreementRate C.disagreementAltRate C.testLevel nEnd hS)
    (hL : 2 ≤ C.maxLeaves) :
    ∀ (l : List (FreeMonoid α)) (s : Level5State α), VerInv C s → StretchOK nEnd hS s →
      versionBound C (Fintype.card α) * nEnd ≤ s.version * nEnd + s.probes + l.length →
      RunEnds (applyProbe C cut) (fun _ _ => True) s l
  | [], s, hv, hs, hb => by
    have h1 := version_lt C hv hL
    have h2 : (s.version + 1) * nEnd ≤ versionBound C (Fintype.card α) * nEnd :=
      Nat.mul_le_mul_right _ h1
    have := hs.1
    simp only [List.length_nil, add_zero] at hb
    nlinarith
  | x :: xs, s, hv, hs, hb => by
    simp only [RunEnds]
    rcases hst : applyProbe C cut s x with s' | ⟨e, s'⟩
    · obtain ⟨hs', hc⟩ := stretchOK_step C cut hSq hn hSn hφ0 hφ hn₀ hpt hsucc hs hv.1 hst
      refine runEnds_of_budget hSq hn hSn hφ0 hφ hn₀ hpt hsucc hL xs s'
        (verInv_step C cut hv hst) hs' ?_
      simp only [List.length_cons] at hb
      have := hs.1
      rcases hc with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> rw [h1, h2] <;> nlinarith
    · trivial

end Stretch

end OrthoDFA
