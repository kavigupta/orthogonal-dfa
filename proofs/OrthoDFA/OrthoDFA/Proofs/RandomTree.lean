import OrthoDFA.RandomRound
import OrthoDFA.Proofs.IdealSteps

/-!
# The tree and the steps under reads that are never wrong

A decided read is on its state's side, so two strings of one state that both sift decidedly
reach the same leaf, and the leaves past the first two are at most `|Q|`. Every split is on a
letter and the midfix where two leaves part, so the tree is in `classSet |Q|`.

A step that keeps the hypothesis counts one more probe into the stretch; one that changes it
starts a fresh stretch and lowers `Ψ`, #417's count of the splits, learnings and redirects left.
-/

namespace OrthoDFA

namespace Random

open OrthoDFA.Ideal (DTree Edges walk pre search agrees probe Outcome upd2 lcp retarget Disagrees
  EdgesOK Reached OutcomeOK)

variable {α : Type*}

/-- No read is on its state's wrong side. -/
def NoWrong {σ : Type*} (M : DFA α σ) (side : σ → Bool) (read : FreeMonoid α → ARU) : Prop :=
  ∀ z, read z ≠ if side (M.eval z.toList) then .reject else .accept

open scoped Classical in
/-- The midfixes of the tree's nodes. -/
noncomputable def mids : DTree α → Finset (FreeMonoid α)
  | .leaf => ∅
  | .node m r a => insert m (mids r ∪ mids a)

open scoped Classical in
/-- The strings a probe of `x` can read: a prefix of length at least `k` followed by a midfix. -/
noncomputable def pot (k : ℕ) (x : FreeMonoid α) (T : DTree α) : Finset (FreeMonoid α) :=
  (Finset.Icc k x.toList.length).biUnion fun p => (mids T).image (pre x p * ·)

/-- The probes that can read a string at a good read-state undecided. -/
def PotGood {σ : Type*} (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) (k : ℕ) (read : FreeMonoid α → ARU)
    (T : DTree α) : Set (FreeMonoid α) :=
  {x | ∃ z ∈ pot k x T, ¬ BadAt U θ (M.eval z.toList) ∧ read z = .undecided}

open scoped Classical in
/-- The trees at most `n` splits reach, each at a leaf on a letter and the midfix where two
leaves part. -/
noncomputable def classSet [Fintype α] : ℕ → Finset (DTree α)
  | 0 => {.node 1 .leaf .leaf}
  | n + 1 => classSet n ∪ (classSet n).biUnion fun T =>
      (T.leaves.toFinset ×ˢ (Finset.univ : Finset α) ×ˢ T.leaves.toFinset ×ˢ
        T.leaves.toFinset).image
        fun q => T.splitAt (FreeMonoid.of q.2.1 * T.midAt (lcp q.2.2.1 q.2.2.2)) q.1

variable [DecidableEq α] (read : FreeMonoid α → ARU) (C : Cfg)

/-- #417's settings with this round's cap, for its potential. -/
def idealCfg (C : Cfg) : Ideal.RoundCfg := ⟨C.k, C.m, 1, 1, C.Lmax⟩

/-- The splits, learnings and redirects left, as #417 counts them. -/
def Ψr [Fintype α] (s : RState α) : ℕ := Ideal.Ψ (idealCfg C) (Ideal.fresh s.tree s.edges s.moved)

structure Inv [Fintype α] (s : RState α) : Prop where
  edges : EdgesOK read s.tree s.edges
  recs_lt : ∀ r, s.recs.count r < C.m
  size : s.tree.leaves.length ≤ C.Lmax
  reached : Reached read s.tree
  cls : s.tree ∈ classSet (s.tree.leaves.length - 2)
  two : 2 ≤ s.tree.leaves.length

section Steps

variable [Fintype α] {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}

theorem inv_start (hm : 1 ≤ C.m) (h2 : 2 ≤ C.Lmax) : Inv read C (start : RState α) := by
  sorry

theorem Ψr_start : Ψr C (start : RState α) + 1 ≤ stretches C (Fintype.card α) := by
  sorry

/-- A step that keeps the hypothesis counts one more probe. -/
theorem step_same (hW : NoWrong M side read) {s s' : RState α} (hs : Inv read C s)
    {x : FreeMonoid α} (h : step read C s x = .inl s') (ht : s'.tree = s.tree)
    (he : s'.edges = s.edges) : Inv read C s' ∧ Ψr C s' = Ψr C s ∧ s'.n = s.n + 1 := by
  sorry

/-- A step that changes the hypothesis starts a fresh stretch and lowers `Ψr`. -/
theorem step_change (hW : NoWrong M side read) (hcap : Fintype.card σ + 2 ≤ C.Lmax)
    (hm : 1 ≤ C.m) {s s' : RState α} (hs : Inv read C s) {x : FreeMonoid α}
    (h : step read C s x = .inl s') (hne : ¬ (s'.tree = s.tree ∧ s'.edges = s.edges)) :
    Inv read C s' ∧ s' = fresh s'.tree s'.edges s'.moved ∧ Ψr C s' < Ψr C s := by
  sorry

theorem step_ne_tooBig (hW : NoWrong M side read) (hcap : Fintype.card σ + 2 ≤ C.Lmax)
    {s : RState α} (hs : Inv read C s) (x : FreeMonoid α) : step read C s x ≠ .inr .tooBig := by
  sorry

theorem cls_mem (hW : NoWrong M side read) (hcap : Fintype.card σ + 2 ≤ C.Lmax) {s : RState α}
    (hs : Inv read C s) : s.tree ∈ classSet (Fintype.card σ) := by
  sorry

end Steps

end Random

end OrthoDFA
