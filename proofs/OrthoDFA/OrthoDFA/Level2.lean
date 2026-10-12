import OrthoDFA.Level2Probe
import OrthoDFA.RoundEnd

/-! Level 2 of indecisive L*: probes redirect the hypothesis's edges and split its leaves. -/

namespace OrthoDFA

namespace Ideal

variable {α : Type*}

structure Level2Settings where
  /-- Each probe is walked along the edges from its prefix this long. -/
  walkStart : ℕ
  /-- Records of one edge and target that redirect the edge, or split its leaf once redirected. -/
  recordsToMove : ℕ
  /-- Undecided strings charged to one site that end the round as its harvest. -/
  harvestSize : ℕ
  /-- Probes in a row with no disagreement that end the round consistent. -/
  cleanProbesToEnd : ℕ
  maxLeaves : ℕ

/-- `redirected` and the counts are since the edge, or the hypothesis, last changed. -/
structure Level2State (α : Type*) where
  tree : DTree α
  edges : Edges α
  redirected : List Bool × α → Bool
  /-- Each clean disagreement's edge and the leaf the next prefix sifts to. -/
  records : List (List Bool × α × List Bool)
  undecided : UndecidedSite α → List (FreeMonoid α)
  /-- Probes since the last disagreement. -/
  cleanRun : ℕ

variable [DecidableEq α]

def newHypothesis (T : DTree α) (E : Edges α) (redirected : List Bool × α → Bool) :
    Level2State α :=
  ⟨T, E, redirected, [], fun _ => [], 0⟩

/-- The root reads at `ε`; no edge is learned. -/
def initialState : Level2State α :=
  newHypothesis (.node 1 .leaf .leaf) (fun _ => none) fun _ => false

/-- The edges after the leaf `p` splits into `T'`: those into `p` sift their witness's successor
again, those out of it are dropped. -/
def retarget (read : FreeMonoid α → ARU) (T' : DTree α) (p : List Bool) (E : Edges α) :
    Edges α := fun e =>
  if p.isPrefixOf e.1 then none
  else
    match E e with
    | some (t, w) =>
      if t = p then
        (T'.sift read (w * FreeMonoid.of e.2)).elim (fun t' => some (t', w)) fun _ => none
      else some (t, w)
    | none => none

variable (read : FreeMonoid α → ARU) (C : Level2Settings)

def extendCleanRun (s : Level2State α) : Level2State α ⊕ RoundEnd α :=
  if C.cleanProbesToEnd ≤ s.cleanRun + 1 then .inr .consistent
  else .inl { s with cleanRun := s.cleanRun + 1 }

/-- Undecided strings `zs` charged to `k`, from a probe that disagreed unless `clean`. -/
def addUndecided (s : Level2State α) (k : UndecidedSite α) (zs : List (FreeMonoid α))
    (clean : Bool) : Level2State α ⊕ RoundEnd α :=
  let s' := { s with undecided := Function.update s.undecided k (s.undecided k ++ zs) }
  if C.harvestSize ≤ (s'.undecided k).length then .inr (.harvest (s'.undecided k))
  else if clean then extendCleanRun C s' else .inl { s' with cleanRun := 0 }

def splitLeaf (s : Level2State α) (p : List Bool) (d : FreeMonoid α) :
    Level2State α ⊕ RoundEnd α :=
  let T' := s.tree.splitAt d p
  if C.maxLeaves < T'.leaves.length then .inr .failed
  else .inl (newHypothesis T' (retarget read T' p s.edges) fun _ => false)

/-- A clean disagreement at the edge `(p, c)`, with `u` at `p` and `u·c` at `t`. -/
def recordDisagreement (s : Level2State α) (p : List Bool) (c : α) (u : FreeMonoid α)
    (t : List Bool) : Level2State α ⊕ RoundEnd α :=
  let s' := { s with records := s.records ++ [(p, c, t)], cleanRun := 0 }
  if C.recordsToMove ≤ s'.records.count (p, c, t) then
    match s.edges (p, c) with
    | some (t₀, _) =>
      if s.redirected (p, c) then
        splitLeaf read C s p (FreeMonoid.of c * s.tree.midAt (commonPrefix t t₀))
      else .inl (newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u)))
        (Function.update s.redirected (p, c) true))
    | none => .inl s'
  else .inl s'

def applyProbe (s : Level2State α) (x : FreeMonoid α) : Level2State α ⊕ RoundEnd α :=
  match probe read s.tree s.edges C.walkStart x with
  | .agree => extendCleanRun C s
  | .undecided k zs clean => addUndecided C s k zs clean
  | .wrongEdge p c u t => recordDisagreement read C s p c u t
  | .unlearnedEdge p c u t =>
    .inl (newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u)))
      (Function.update s.redirected (p, c) false))

/-- The state the probes end in, and how; `failed` where they ran out. -/
def runProbes : Level2State α → List (FreeMonoid α) → Level2State α × RoundEnd α
  | s, [] => (s, .failed)
  | s, x :: xs =>
    match applyProbe read C s x with
    | .inl s' => runProbes s' xs
    | .inr e => (s, e)

structure Level2Result (α : Type*) where
  hypothesis : Hypothesis α
  ending : RoundEnd α

def indecisiveLStarLevel2 {P : ℕ} (draws : Fin P → FreeMonoid α) : Level2Result α :=
  let r := runProbes read C initialState (List.ofFn draws)
  ⟨⟨r.1.tree, r.1.edges⟩, r.2⟩

end Ideal

end OrthoDFA
