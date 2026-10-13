import OrthoDFA.Level3

/-!
# Level 3.5 of indecisive L*: rare wrong reads

Level 3's probes, counts and tests, run where a read can fall on its state's wrong side. A wrong
read can make a clean disagreement untrue, so one is not acted on at once: it records its edge and
target, `recordsToMove` records redirect the edge or, once it has been redirected, split its leaf,
and a split past `maxLeaves` leaves fails the round. A probe at an unlearned edge learns it.
-/

namespace OrthoDFA

namespace Random

namespace Eps

open OrthoDFA.Ideal (probeOrAgree retarget)

variable {α : Type*} [DecidableEq α]

structure Level3_5Settings where
  /-- Each probe is walked along the edges from its prefix this long. -/
  walkStart : ℕ
  /-- Records of one edge and target that redirect the edge, or split its leaf once redirected. -/
  recordsToMove : ℕ
  maxLeaves : ℕ
  /-- The count of trials before `rateSide` decides. -/
  firstLook : ℕ
  /-- Each test's chance of firing falsely at one look. -/
  testLevel : ℝ
  startRate : ℝ
  edgeRate : ℝ
  edgeReadShare : ℝ
  edgeExcess : ℝ
  middleRate : ℝ
  searchShare : ℝ
  disagreementRate : ℝ
  probeBudget : ℕ

/-- `redirected` and the counts are since the edge, or the hypothesis, last changed. -/
structure Level3_5State (α : Type*) where
  tree : DTree α
  edges : Edges α
  redirected : List Bool × α → Bool
  /-- Each clean disagreement's edge and the leaf the next prefix sifts to. -/
  records : List (List Bool × α × List Bool)
  probeCount : ℕ
  searchCount : ℕ
  startUndecided : List (FreeMonoid α)
  middleUndecided : List (FreeMonoid α)
  edgeReads : List Bool → α → ℕ
  edgeUndecided : List Bool → α → List (FreeMonoid α)

def newHypothesis (T : DTree α) (E : Edges α) (redirected : List Bool × α → Bool) :
    Level3_5State α :=
  ⟨T, E, redirected, [], 0, 0, [], [], fun _ _ => 0, fun _ _ => []⟩

/-- The root reads at `ε`; no edge is learned. -/
def initialState : Level3_5State α :=
  newHypothesis (.node 1 .leaf .leaf) (fun _ => none) fun _ => false

section

variable (read : FreeMonoid α → ARU) (C : Level3_5Settings)

/-- A probe counted into the stretch. -/
noncomputable def countProbe [Fintype α] (s : Level3_5State α) (x : FreeMonoid α) :
    Level3_5State α :=
  let o := probeOrAgree read s.tree s.edges C.walkStart x
  let ch := positionReads read s.tree s.edges C.walkStart x
  { s with
    probeCount := s.probeCount + 1
    searchCount := s.searchCount + if searches o then 1 else 0
    startUndecided := s.startUndecided ++ startString o
    middleUndecided := s.middleUndecided ++ middleString o
    edgeReads := fun p c => s.edgeReads p c + readsAtEdge ch p c
    edgeUndecided := fun p c => s.edgeUndecided p c ++ undecidedAtEdge ch p c }

open scoped Classical in
/-- The tests, in turn: the start's undecided rate, an edge's undecided reads, the undecided
middles among the searches, and the disagreement rate. -/
noncomputable def runTests [Fintype α] (s : Level3_5State α) : Option (RoundEnd (Level3Site α)) :=
  if rateSide C.startRate C.testLevel C.firstLook s.probeCount s.startUndecided.length = some true
  then some (.harvest .start)
  else if h : ∃ e ∈ s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α),
      C.edgeReadShare * s.probeCount ≤ s.edgeReads e.1 e.2
        ∧ C.edgeExcess + C.edgeRate * s.edgeReads e.1 e.2 ≤ (s.edgeUndecided e.1 e.2).length then
    some (.harvest (.edge h.choose.1 h.choose.2))
  else if C.searchShare * s.probeCount ≤ s.searchCount
      ∧ rateSide C.middleRate C.testLevel C.firstLook s.searchCount s.middleUndecided.length
        = some true then
    some (.harvest .middles)
  else if rateSide C.disagreementRate C.testLevel C.firstLook s.probeCount s.searchCount
      = some false then some .consistent
  else none

noncomputable def applyTests [Fintype α] (s : Level3_5State α) :
    Level3_5State α ⊕ RoundEnd (Level3Site α) :=
  match runTests C s with
  | some e => .inr e
  | none => .inl s

def splitLeaf (s : Level3_5State α) (p : List Bool) (d : FreeMonoid α) :
    Level3_5State α ⊕ RoundEnd (Level3Site α) :=
  let T' := s.tree.splitAt d p
  if C.maxLeaves < T'.leaves.length then .inr .failed
  else .inl (newHypothesis T' (retarget read T' p s.edges) fun _ => false)

/-- The edge `(p, c)` has `recordsToMove` records at `t`, with `u` at `p`: split where it has
been redirected, else redirect it. -/
def redirectOrSplit (s : Level3_5State α) (p : List Bool) (c : α) (u : FreeMonoid α)
    (t : List Bool) : Level3_5State α ⊕ RoundEnd (Level3Site α) :=
  match s.edges (p, c) with
  | some (t₀, _) =>
    if s.redirected (p, c) then
      splitLeaf read C s p (FreeMonoid.of c * s.tree.midAt (commonPrefix t t₀))
    else .inl (newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u)))
      (Function.update s.redirected (p, c) true))
  | none => .inl s

noncomputable def applyProbe [Fintype α] (s : Level3_5State α) (x : FreeMonoid α) :
    Level3_5State α ⊕ RoundEnd (Level3Site α) :=
  match probeOrAgree read s.tree s.edges C.walkStart x with
  | .unlearnedEdge p c u t =>
    .inl (newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u)))
      (Function.update s.redirected (p, c) false))
  | .wrongEdge p c u t =>
    if C.recordsToMove ≤ (s.records ++ [(p, c, t)]).count (p, c, t) then
      redirectOrSplit read C s p c u t
    else applyTests C (countProbe read C { s with records := s.records ++ [(p, c, t)] } x)
  | _ => applyTests C (countProbe read C s x)

/-- The state the probes end in, and how; `failed` where they ran out. -/
noncomputable def runProbes [Fintype α] :
    Level3_5State α → List (FreeMonoid α) → Level3_5State α × RoundEnd (Level3Site α)
  | s, [] => (s, .failed)
  | s, x :: xs =>
    match applyProbe read C s x with
    | .inl s' => runProbes s' xs
    | .inr e => (s, e)

end

noncomputable def indecisiveLStarLevel3_5 [Fintype α] (settings : Level3_5Settings)
    (read : FreeMonoid α → ARU) {P : ℕ} (draws : Fin P → FreeMonoid α) : Level3Result α :=
  let r := runProbes read settings initialState (List.ofFn draws)
  ⟨⟨r.1.tree, r.1.edges⟩, r.2⟩

end Eps

end Random

end OrthoDFA
