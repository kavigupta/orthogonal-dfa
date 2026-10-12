import OrthoDFA.Level2Claim

/-!
# Level 3 of indecisive L*: random reads

Level 2's probes, walk and search, with reads drawn at random. Each probe is walked from
`walkStart` along the learned edges and sifted whole, a binary search finds where the two part, a
clean disagreement records its edge and target, `recordsToMove` records redirect the edge or,
once it has been redirected, split its leaf, and a probe at an unlearned edge learns it. A probe
shorter than `walkStart` has no walk and is counted as agreeing. Its counts, kept over the stretch
since the hypothesis last changed, feed the tests of level 4, checked after every probe:
* the start's undecided rate above `startRate` (`rateSide`) harvests the start's undecided
  strings;
* an edge read at least `edgeReadShare` times a probe whose undecided reads exceed `edgeRate` of
  its reads by `edgeExcess` harvests them;
* with searches at least `searchShare` of the probes, the searches stopping at an undecided middle
  above `middleRate` of them harvest those middles;
* the disagreement rate settling below `disagreementRate` ends the round consistent.
-/

namespace OrthoDFA

namespace Random

open OrthoDFA.Ideal (siftAgrees probe Outcome retarget)

variable {α : Type*}

/-- `binomial_side_of_boundary` once `n₀` draws are in: above `θ` when `Bin(n, θ)` reaches `h`
hits with chance below `a`, below when it stays at or under `h` with chance below `a`. -/
noncomputable def rateSide (θ a : ℝ) (n₀ n h : ℕ) : Option Bool :=
  if n₀ ≤ n then
    if binomSfGe n θ h < a then some true
    else if 1 - binomSfGe n θ (h + 1) < a then some false
    else none
  else none

/-- The positions `bisect` sifts: each middle, and its neighbours where it is undecided. -/
def bisectPositions (ag : ℕ → Option Bool) (lo hi : ℕ) : List ℕ :=
  if _h : hi ≤ lo + 1 then []
  else
    match ag ((lo + hi) / 2) with
    | some true => (lo + hi) / 2 :: bisectPositions ag ((lo + hi) / 2) hi
    | some false => (lo + hi) / 2 :: bisectPositions ag lo ((lo + hi) / 2)
    | none =>
      match ag ((lo + hi) / 2 - 1), ag ((lo + hi) / 2 + 1) with
      | none, _ => [(lo + hi) / 2, (lo + hi) / 2 - 1]
      | _, none => [(lo + hi) / 2, (lo + hi) / 2 - 1, (lo + hi) / 2 + 1]
      | some true, some false => [(lo + hi) / 2, (lo + hi) / 2 - 1, (lo + hi) / 2 + 1]
      | some false, _ =>
        (lo + hi) / 2 :: ((lo + hi) / 2 - 1) :: ((lo + hi) / 2 + 1)
          :: bisectPositions ag lo ((lo + hi) / 2 - 1)
      | some true, some true =>
        (lo + hi) / 2 :: ((lo + hi) / 2 - 1) :: ((lo + hi) / 2 + 1)
          :: bisectPositions ag ((lo + hi) / 2 + 1) hi
termination_by hi - lo
decreasing_by all_goals omega

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

/-- A probe's walk, and the positions it sifts past its start, as `probe` reads them. -/
def probePositions : (ℕ → List Bool) × List ℕ :=
  match T.sift read (prefixOf x k) with
  | .inr _ => (fun _ => [], [])
  | .inl a =>
    let ss := walk E a (x.toList.drop k)
    let st := fun p => ss.getD (p - k) []
    let j := k + ss.length - 1
    match x.toList[j]? with
    | some _ =>
      match T.sift read (prefixOf x (j + 1)), T.sift read (prefixOf x j) with
      | .inr _, _ => (st, [j + 1])
      | _, .inr _ => (st, [j + 1, j])
      | .inl _, .inl s =>
        (st, if s = st j then [j + 1, j]
          else (j + 1) :: j :: bisectPositions (siftAgrees read T x st) k j)
    | none =>
      match T.sift read x with
      | .inr _ => (st, [j])
      | .inl e =>
        (st, if e = st j then [j] else j :: bisectPositions (siftAgrees read T x st) k j)

/-- A probe, which counts as agreeing where it is shorter than `k`. -/
def probeOrAgree : Outcome α := if x.toList.length < k then .agree else probe read T E k x

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

def readsAtEdge (ch : List (List Bool × α × ℕ × Option (FreeMonoid α))) (p : List Bool) (c : α) :
    ℕ :=
  ((ch.filter fun e => e.1 = p ∧ e.2.1 = c).map fun e => e.2.2.1).sum

def undecidedAtEdge (ch : List (List Bool × α × ℕ × Option (FreeMonoid α))) (p : List Bool)
    (c : α) : List (FreeMonoid α) :=
  (ch.filter fun e => e.1 = p ∧ e.2.1 = c).filterMap fun e => e.2.2.2

/-- The probe searched: its walk and its sift parted at decided reads. -/
def searches : Outcome α → Bool
  | .wrongEdge .. => true
  | .undecided _ _ clean => !clean
  | _ => false

def startString : Outcome α → List (FreeMonoid α)
  | .undecided .start [z] _ => [z]
  | _ => []

/-- The undecided middle the search stopped at. -/
def middleString : Outcome α → List (FreeMonoid α)
  | .undecided _ zs false => zs.take 1
  | _ => []

structure Level3Settings where
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
structure Level3State (α : Type*) where
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
    Level3State α :=
  ⟨T, E, redirected, [], 0, 0, [], [], fun _ _ => 0, fun _ _ => []⟩

/-- The root reads at `ε`; no edge is learned. -/
def initialState : Level3State α :=
  newHypothesis (.node 1 .leaf .leaf) (fun _ => none) fun _ => false

variable (read : FreeMonoid α → ARU) (C : Level3Settings)

/-- A probe counted into the stretch. -/
def countProbe (s : Level3State α) (x : FreeMonoid α) : Level3State α :=
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
noncomputable def runTests [Fintype α] (s : Level3State α) : Option (RoundEnd α) :=
  if rateSide C.startRate C.testLevel C.firstLook s.probeCount s.startUndecided.length = some true
  then some (.harvest s.startUndecided)
  else if h : ∃ e ∈ s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α),
      C.edgeReadShare * s.probeCount ≤ s.edgeReads e.1 e.2
        ∧ C.edgeExcess + C.edgeRate * s.edgeReads e.1 e.2 ≤ (s.edgeUndecided e.1 e.2).length then
    some (.harvest (s.edgeUndecided h.choose.1 h.choose.2))
  else if C.searchShare * s.probeCount ≤ s.searchCount
      ∧ rateSide C.middleRate C.testLevel C.firstLook s.searchCount s.middleUndecided.length
        = some true then
    some (.harvest s.middleUndecided)
  else if rateSide C.disagreementRate C.testLevel C.firstLook s.probeCount s.searchCount
      = some false then some .consistent
  else none

noncomputable def applyTests [Fintype α] (s : Level3State α) : Level3State α ⊕ RoundEnd α :=
  match runTests C s with
  | some e => .inr e
  | none => .inl s

def splitLeaf (s : Level3State α) (p : List Bool) (d : FreeMonoid α) : Level3State α ⊕ RoundEnd α :=
  let T' := s.tree.splitAt d p
  if C.maxLeaves < T'.leaves.length then .inr .failed
  else .inl (newHypothesis T' (retarget read T' p s.edges) fun _ => false)

/-- The edge `(p, c)` has `recordsToMove` records at `t`, with `u` at `p`: split where it has
been redirected, else redirect it. -/
def redirectOrSplit (s : Level3State α) (p : List Bool) (c : α) (u : FreeMonoid α)
    (t : List Bool) : Level3State α ⊕ RoundEnd α :=
  match s.edges (p, c) with
  | some (t₀, _) =>
    if s.redirected (p, c) then
      splitLeaf read C s p (FreeMonoid.of c * s.tree.midAt (commonPrefix t t₀))
    else .inl (newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u)))
      (Function.update s.redirected (p, c) true))
  | none => .inl s

noncomputable def applyProbe [Fintype α] (s : Level3State α) (x : FreeMonoid α) :
    Level3State α ⊕ RoundEnd α :=
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
    Level3State α → List (FreeMonoid α) → Level3State α × RoundEnd α
  | s, [] => (s, .failed)
  | s, x :: xs =>
    match applyProbe read C s x with
    | .inl s' => runProbes s' xs
    | .inr e => (s, e)

structure Level3Result (α : Type*) where
  hypothesis : Hypothesis α
  ending : RoundEnd α

noncomputable def indecisiveLStarLevel3 [Fintype α] {P : ℕ} (draws : Fin P → FreeMonoid α) :
    Level3Result α :=
  let r := runProbes read C initialState (List.ofFn draws)
  ⟨⟨r.1.tree, r.1.edges⟩, r.2⟩

end Random

end OrthoDFA
