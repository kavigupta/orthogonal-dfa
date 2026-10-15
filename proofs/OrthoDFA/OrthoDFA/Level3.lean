import OrthoDFA.Level2Claim

/-!
# Level 3 of indecisive L*: random reads

Level 2's probes and walk, with reads drawn at random. Each probe is walked from `walkStart` along
the learned edges and sifted whole, and `tripartiteSearch`, with no recovery, finds where the two
part or stops at an undecided midpoint. A decided read is never on its state's wrong side, so as at
level 2 a clean disagreement splits the edge's leaf at once, and a probe at an unlearned edge learns
it. A probe shorter than `walkStart` has no walk and is counted as agreeing. Its counts, kept over
the stretch since the hypothesis last changed, feed the tests of level 5, checked after every probe,
each rate tested by `seqRate` against an alternative, twice the rate for the start and the middles
and half of it for success. Under the `j`-th hypothesis each test runs at level
`δ / (8 (j + 1) (j + 2))`, the edges' split evenly among the edges:
* the start's undecided rate above `startRate` harvests the start;
* an edge read at least `edgeReadShare` times a probe whose undecided reads exceed `edgeRate` of
  its reads by `edgeExcess` harvests the edge, a probe's undecided strings at an edge counted once;
* with searches at least `searchShare` of the probes, the searches stopping at an undecided middle
  above `middleRate` of them harvest the middles;
* the disagreement rate settling below `disagreementRate` ends the round consistent.
-/

namespace OrthoDFA

namespace Random


variable {α : Type*}

/-- How many reads the sift of `z` makes: one per node down to a leaf or an undecided read. -/
def siftReads (read : FreeMonoid α → ARU) : DTree α → FreeMonoid α → ℕ
  | .leaf, _ => 0
  | .node m r a, z =>
    match read (z * m) with
    | .accept => siftReads read a z + 1
    | .reject => siftReads read r z + 1
    | .undecided => 1

/-- The walk's edge out of position `i`, or where the probe ends there, its last edge. -/
def edgeAt (x : FreeMonoid α) (st : ℕ → List Bool) (i : ℕ) : Option (List Bool × α) :=
  match x.toList[i]? with
  | some c => some (st i, c)
  | none => (x.toList[i - 1]?).map fun c => (st (i - 1), c)

section Probe

variable (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α)

/-- A probe's walk, and the positions it sifts past its start, as `probe` reads them: the stop's
successor, where the walk stops at an unlearned edge, and the search's. -/
def probePositions : (ℕ → List Bool) × List ℕ :=
  match T.sift read (prefixOf x k) with
  | .inr _ => (fun _ => [], [])
  | .inl a =>
    let ss := walk E a (x.toList.drop k)
    let st := fun p => ss.getD (p - k) []
    let hi := k + ss.length - 1
    let searched := (tripartiteSearch false (siftAgrees read T x st) k hi).2
    match x.toList[hi]? with
    | some _ =>
      match T.sift read (prefixOf x (hi + 1)) with
      | .inr _ => (st, [hi + 1])
      | .inl _ => (st, (hi + 1) :: searched)
    | none => (st, searched)

/-- What each position the probe sifts past its start adds to its edge's counts: the reads, and
the string read undecided, if any. -/
def positionReads : List (List Bool × α × ℕ × Option (FreeMonoid α)) :=
  if x.toList.length < k then []
  else
    (((probePositions read T E k x).2.filter fun i => k < i ∧ i ≤ x.toList.length).dedup
      ).filterMap
      fun i => (edgeAt x (probePositions read T E k x).1 i).map fun e =>
        (e.1, e.2, siftReads read T (prefixOf x i),
          (T.sift read (prefixOf x i)).elim (fun _ => none) some)

end Probe

variable [DecidableEq α]

/-- Where a harvest's strings were read: the start, an edge, or the middles searches stop at. -/
inductive Level3Site (α : Type*)
  | start
  | edge (p : List Bool) (c : α)
  | middles

def readsAtEdge (ch : List (List Bool × α × ℕ × Option (FreeMonoid α))) (p : List Bool) (c : α) :
    ℕ :=
  ((ch.filter fun e => e.1 = p ∧ e.2.1 = c).map fun e => e.2.2.1).sum

/-- The strings read undecided at the edge `(p, c)`, each once. -/
noncomputable def undecidedAtEdge [Fintype α]
    (ch : List (List Bool × α × ℕ × Option (FreeMonoid α))) (p : List Bool) (c : α) :
    List (FreeMonoid α) :=
  ((ch.filter fun e => e.1 = p ∧ e.2.1 = c).filterMap fun e => e.2.2.2).dedup

/-- The probe searched: its walk and its sift parted at decided reads. -/
def searches : Outcome α → Bool
  | .wrongEdge .. => true
  | .undecided _ clean => !clean
  | _ => false

def startString : Outcome α → List (FreeMonoid α)
  | .undecidedStart z => [z]
  | _ => []

/-- The undecided middle the search stopped at. -/
def middleString : Outcome α → List (FreeMonoid α)
  | .undecided zs false => zs.take 1
  | _ => []

structure Level3Settings where
  /-- Each probe is walked along the edges from its prefix this long. -/
  walkStart : ℕ
  /-- The chance the round may fail to end well. -/
  δ : ℝ
  startRate : ℝ
  edgeRate : ℝ
  edgeReadShare : ℝ
  middleRate : ℝ
  searchShare : ℝ
  disagreementRate : ℝ
  probeBudget : ℕ

/-- The tests' level under the `j`-th hypothesis. -/
noncomputable def Level3Settings.level (C : Level3Settings) (j : ℕ) : ℝ :=
  C.δ / (8 * (j + 1) * (j + 2))

noncomputable def Level3Settings.startAlternative (C : Level3Settings) : ℝ := 2 * C.startRate

noncomputable def Level3Settings.middleAlternative (C : Level3Settings) : ℝ := 2 * C.middleRate

noncomputable def Level3Settings.disagreementAlternative (C : Level3Settings) : ℝ :=
  C.disagreementRate / 2

/-- How far an edge's undecided reads must pass `edgeRate` of its reads, with `n` edges, under the
`j`-th hypothesis: each edge's test at level `level j / n`. -/
noncomputable def Level3Settings.edgeExcess (C : Level3Settings) (n j : ℕ) : ℝ :=
  4 / 5 * Real.log (n / C.level j)

/-- The counts are since the hypothesis last changed; `hypotheses` counts the changes before. -/
structure Level3State (α : Type*) where
  tree : DTree α
  edges : Edges α
  hypotheses : ℕ
  probeCount : ℕ
  searchCount : ℕ
  startUndecided : List (FreeMonoid α)
  middleUndecided : List (FreeMonoid α)
  edgeReads : List Bool → α → ℕ
  edgeUndecided : List Bool → α → List (FreeMonoid α)

def newHypothesis (T : DTree α) (E : Edges α) (j : ℕ) : Level3State α :=
  ⟨T, E, j, 0, 0, [], [], fun _ _ => 0, fun _ _ => []⟩

/-- The root reads at `ε`; no edge is learned. -/
def initialState : Level3State α := newHypothesis (.node 1 .leaf .leaf) (fun _ => none) 0

section

variable (read : FreeMonoid α → ARU) (C : Level3Settings)

/-- A probe counted into the stretch. -/
noncomputable def countProbe [Fintype α] (s : Level3State α) (x : FreeMonoid α) : Level3State α :=
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
noncomputable def runTests [Fintype α] (s : Level3State α) : Option (RoundEnd (Level3Site α)) :=
  if seqRate C.startRate C.startAlternative (C.level s.hypotheses) s.probeCount
      s.startUndecided.length then
    some (.harvest .start)
  else if h : ∃ e ∈ s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α),
      C.edgeReadShare * s.probeCount ≤ s.edgeReads e.1 e.2
        ∧ C.edgeExcess (s.tree.leaves.length * Fintype.card α) s.hypotheses
            + C.edgeRate * s.edgeReads e.1 e.2 ≤ (s.edgeUndecided e.1 e.2).length then
    some (.harvest (.edge h.choose.1 h.choose.2))
  else if C.searchShare * s.probeCount ≤ s.searchCount
      ∧ seqRate C.middleRate C.middleAlternative (C.level s.hypotheses) s.searchCount
        s.middleUndecided.length then
    some (.harvest .middles)
  else if seqRate C.disagreementRate C.disagreementAlternative (C.level s.hypotheses)
      s.probeCount s.searchCount then some .consistent
  else none

noncomputable def applyTests [Fintype α] (s : Level3State α) :
    Level3State α ⊕ RoundEnd (Level3Site α) :=
  match runTests C s with
  | some e => .inr e
  | none => .inl s

/-- The leaf `p` split on `c` and the midfix where the leaves `t` and `t₀` part. -/
def splitLeaf (s : Level3State α) (p : List Bool) (c : α) (t t₀ : List Bool) :
    Level3State α :=
  let T' := s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p
  newHypothesis T' (retarget read T' p s.edges) (s.hypotheses + 1)

noncomputable def applyProbe [Fintype α] (s : Level3State α) (x : FreeMonoid α) :
    Level3State α ⊕ RoundEnd (Level3Site α) :=
  match probeOrAgree read s.tree s.edges C.walkStart x with
  | .unlearnedEdge p c u t =>
    .inl (newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u))) (s.hypotheses + 1))
  | .wrongEdge p c _ t =>
    match s.edges (p, c) with
    | some (t₀, _) => .inl (splitLeaf read s p c t t₀)
    | none => applyTests C (countProbe read C s x)
  | _ => applyTests C (countProbe read C s x)

/-- The state the probes end in, and how; `failed` where they ran out. -/
noncomputable def runProbes [Fintype α] :
    Level3State α → List (FreeMonoid α) → Level3State α × RoundEnd (Level3Site α)
  | s, [] => (s, .failed)
  | s, x :: xs =>
    match applyProbe read C s x with
    | .inl s' => runProbes s' xs
    | .inr e => (s, e)

structure Level3Result (α : Type*) where
  hypothesis : HypothesisDFA α
  ending : RoundEnd (Level3Site α)

end

noncomputable def indecisiveLStarLevel3 [Fintype α] (settings : Level3Settings)
    (read : FreeMonoid α → ARU) {P : ℕ} (draws : Fin P → FreeMonoid α) : Level3Result α :=
  let r := runProbes read settings initialState (List.ofFn draws)
  ⟨⟨r.1.tree, r.1.edges⟩, r.2⟩

end Random

end OrthoDFA
