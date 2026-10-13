import OrthoDFA.Level2Probe
import OrthoDFA.RateTest
import OrthoDFA.RoundEnd

/-!
Level 2 of indecisive L*. Each probe is walked from its prefix `walkStart` long along the learned
edges, and a bisection with no recovery finds where the walk and the sift part, stopping at the
first undecided read. A probe shorter than `walkStart` counts as agreeing.
* A probe that the edges and the tree part on at decided reads splits the leaf the edge leaves,
  on the edge's letter and the midfix where the two leaves its strings' successors sift to part.
* A probe that reaches an unlearned edge learns it.
* The undecided reads probes stop at, and the disagreements, since the hypothesis last changed
  are counted, and after each probe that keeps the hypothesis the binomial rate test (`rateSide`)
  at level `testLevel` checks them: the undecided reads against `harvestRate`, and once it calls
  them above the round ends in a harvest; else the disagreements against `disagreementRate`, and
  once it calls them below the round ends consistent. A disagreement at decided reads changes the
  hypothesis, so the disagreements counted are the searches that stopped at an undecided midpoint,
  as at level 3.
-/

namespace OrthoDFA

namespace Ideal

variable {α : Type*}

structure Level2Settings where
  /-- Each probe is walked along the edges from its prefix this long. -/
  walkStart : ℕ
  /-- The rate the undecided reads are tested against. -/
  harvestRate : ℝ
  /-- The rate the disagreements are tested against. -/
  disagreementRate : ℝ
  /-- Each rate test's level. -/
  testLevel : ℝ

/-- The counts are since the hypothesis last changed. -/
structure Level2State (α : Type*) where
  tree : DTree α
  edges : Edges α
  undecided : ℕ
  disagreements : ℕ
  probes : ℕ

def newHypothesis (T : DTree α) (E : Edges α) : Level2State α := ⟨T, E, 0, 0, 0⟩

/-- The root reads at `ε`; no edge is learned. -/
def initialState : Level2State α := newHypothesis (.node 1 .leaf .leaf) fun _ => none

variable [DecidableEq α]

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

section

variable (read : FreeMonoid α → ARU) (C : Level2Settings)

/-- A probe that keeps the hypothesis, its `n` undecided reads and `d` disagreements counted. -/
noncomputable def keep (s : Level2State α) (n d : ℕ) : Level2State α ⊕ RoundEnd Unit :=
  let s' : Level2State α :=
    ⟨s.tree, s.edges, s.undecided + n, s.disagreements + d, s.probes + 1⟩
  if rateSide C.harvestRate C.testLevel 0 s'.probes s'.undecided = some true then
    .inr (.harvest ())
  else if rateSide C.disagreementRate C.testLevel 0 s'.probes s'.disagreements = some false then
    .inr .consistent
  else .inl s'

/-- The leaf `p` split on `c` and the midfix where the leaves `t` and `t₀` part. -/
def splitLeaf (s : Level2State α) (p : List Bool) (c : α) (t t₀ : List Bool) : Level2State α :=
  let T' := s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p
  newHypothesis T' (retarget read T' p s.edges)

/-- At `wrongEdge p c u t`, `u·c` sifts to `t` while the edge `(p, c)` points at `t₀`, where its
witness's successor sifts, so the two strings at `p` part at `c` and the midfix where `t` and `t₀`
part. -/
noncomputable def applyProbe (s : Level2State α) (x : FreeMonoid α) :
    Level2State α ⊕ RoundEnd Unit :=
  match probeOrAgree false read s.tree s.edges C.walkStart x with
  | .wrongEdge p c _ t =>
    match s.edges (p, c) with
    | some (t₀, _) => .inl (splitLeaf read s p c t t₀)
    | none => keep C s 0 1
  | .unlearnedEdge p c u t =>
    .inl (newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u))))
  | .undecided zs false => keep C s zs.length 1
  | o => keep C s o.undecidedStrings.length 0

/-- The state the probes end in, and how; `failed` where they ran out. -/
noncomputable def runProbes :
    Level2State α → List (FreeMonoid α) → Level2State α × RoundEnd Unit
  | s, [] => (s, .failed)
  | s, x :: xs =>
    match applyProbe read C s x with
    | .inl s' => runProbes s' xs
    | .inr e => (s, e)

structure Level2Result (α : Type*) where
  hypothesis : Hypothesis α
  ending : RoundEnd Unit

end

noncomputable def indecisiveLStarLevel2 (settings : Level2Settings) (read : FreeMonoid α → ARU)
    {P : ℕ} (draws : Fin P → FreeMonoid α) : Level2Result α :=
  let r := runProbes read settings initialState (List.ofFn draws)
  ⟨⟨r.1.tree, r.1.edges⟩, r.2⟩

end Ideal

end OrthoDFA
