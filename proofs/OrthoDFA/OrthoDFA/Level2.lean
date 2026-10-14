import OrthoDFA.Probe
import OrthoDFA.SeqTest
import OrthoDFA.RoundEnd

/-!
Level 2 of indecisive L*. Each probe is walked from its prefix `walkStart` long along the learned
edges, and the search finds where the walk and the sift part, stopping at the first undecided
read. A probe shorter than `walkStart` counts as agreeing.
* A probe that the edges and the tree part on at decided reads splits the leaf the edge leaves,
  on the edge's letter and the midfix where the two leaves its strings' successors sift to part.
* A probe that reaches an unlearned edge learns it.
* The undecided reads probes stop at, and the searches that stopped at an undecided midpoint (the
  disagreements), are counted since the hypothesis last changed. After each probe that keeps the
  hypothesis a sequential likelihood-ratio test (`seqRate`) checks each count against
  `alternativeRate`: once the undecided reads' test of `harvestRate` fires the round ends in a
  harvest, and once the disagreements' test of `disagreementRate` fires it ends consistent. Under
  the `j`-th hypothesis both tests run at level `δ / (2 (j + 1) (j + 2))`, so the levels over all
  hypotheses sum to at most `δ / 2` for each test.
-/

namespace OrthoDFA

namespace Level2

variable {α : Type*}

structure Settings where
  /-- Each probe is walked along the edges from its prefix this long. -/
  walkStart : ℕ
  /-- The rate the undecided reads are tested against. -/
  harvestRate : ℝ
  /-- The rate the disagreements are tested against. -/
  disagreementRate : ℝ
  /-- The rate each test weighs its rate against, between the two. -/
  alternativeRate : ℝ
  /-- The chance the round may fail to end well. -/
  δ : ℝ

/-- The tests' level under the `j`-th hypothesis. -/
noncomputable def Settings.level (C : Settings) (j : ℕ) : ℝ := C.δ / (2 * (j + 1) * (j + 2))

/-- The counts are since the hypothesis last changed; `hypotheses` counts the changes before. -/
structure State (α : Type*) where
  tree : DTree α
  edges : Edges α
  undecided : ℕ
  disagreements : ℕ
  probes : ℕ
  hypotheses : ℕ

def newHypothesis (T : DTree α) (E : Edges α) (j : ℕ) : State α := ⟨T, E, 0, 0, 0, j⟩

/-- The root reads at `ε`; no edge is learned. -/
def initialState : State α := newHypothesis (.node 1 .leaf .leaf) (fun _ => none) 0

variable [DecidableEq α]

section

variable (read : FreeMonoid α → ARU) (C : Settings)

open scoped Classical in
/-- A probe that keeps the hypothesis, its `n` undecided reads and `d` disagreements counted. -/
noncomputable def keep (s : State α) (n d : ℕ) : State α ⊕ RoundEnd Unit :=
  let s' : State α :=
    ⟨s.tree, s.edges, s.undecided + n, s.disagreements + d, s.probes + 1, s.hypotheses⟩
  if seqRate C.harvestRate C.alternativeRate (C.level s.hypotheses) s'.probes s'.undecided then
    .inr (.harvest ())
  else if seqRate C.disagreementRate C.alternativeRate (C.level s.hypotheses) s'.probes
      s'.disagreements then
    .inr .consistent
  else .inl s'

/-- The leaf `p` split on `c` and the midfix where the leaves `t` and `t₀` part. -/
def splitLeaf (s : State α) (p : List Bool) (c : α) (t t₀ : List Bool) : State α :=
  let H := (HypothesisDFA.mk s.tree s.edges).split read
    (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p
  newHypothesis H.tree H.edges (s.hypotheses + 1)

/-- At `wrongEdge p c u t`, `u·c` sifts to `t` while the edge `(p, c)` points at `t₀`, where its
witness's successor sifts, so the two strings at `p` part at `c` and the midfix where `t` and `t₀`
part. -/
noncomputable def applyProbe (s : State α) (x : FreeMonoid α) : State α ⊕ RoundEnd Unit :=
  match probeOrAgree read s.tree s.edges C.walkStart x with
  | .wrongEdge p c _ t =>
    match s.edges (p, c) with
    | some (t₀, _) => .inl (splitLeaf read s p c t t₀)
    | none => keep C s 0 1
  | .unlearnedEdge p c u t =>
    .inl (newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u))) (s.hypotheses + 1))
  | .undecided zs false => keep C s zs.length 1
  | o => keep C s o.undecidedStrings.length 0

/-- The state the probes end in, and how; `failed` where they ran out. -/
noncomputable def runProbes : State α → List (FreeMonoid α) → State α × RoundEnd Unit
  | s, [] => (s, .failed)
  | s, x :: xs =>
    match applyProbe read C s x with
    | .inl s' => runProbes s' xs
    | .inr e => (s, e)

structure Result (α : Type*) where
  hypothesis : HypothesisDFA α
  ending : RoundEnd Unit

end

noncomputable def indecisiveLStarLevel2 (settings : Settings) (read : FreeMonoid α → ARU)
    {P : ℕ} (draws : Fin P → FreeMonoid α) : Result α :=
  let r := runProbes read settings initialState (List.ofFn draws)
  ⟨⟨r.1.tree, r.1.edges⟩, r.2⟩

end Level2

end OrthoDFA
