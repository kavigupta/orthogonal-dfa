import OrthoDFA.Level2Claim

/-!
# Level 3 of indecisive L*: random reads

Level 2's probes and walk, with reads drawn at random. Each probe is walked from `walkStart` along
the learned edges and sifted whole, and `tripartiteSearch`, with no recovery, finds where the two
part or stops at an undecided midpoint. A decided read is never on its state's wrong side, so as at
level 2 a clean disagreement splits the edge's leaf at once, and a probe at an unlearned edge learns
it. A probe shorter than `walkStart` has no walk and is counted as agreeing. Its counts, kept over
the stretch since the hypothesis last changed, feed the tests of level 5, checked after every probe,
each rate tested by `seqRate` against an alternative, twice the rate for the ends and the middles
and half of it for success. Under the `j`-th hypothesis each test runs at level
`δ / (8 (j + 1) (j + 2))`:
* the probes whose start or end sift stops undecided above `endsRate = y` harvest the ends;
* with searches at least `searchShare = ε/4` of the probes, the searches stopping at an undecided
  middle above `middleRate = 1/4` of them harvest the middles;
* the searches settling below `ε` of the probes end the round consistent.

Here `y = harvestRate` and `ε = disagreementRate`.
-/

namespace OrthoDFA

namespace Random


variable {α : Type*}

variable [DecidableEq α]

/-- Where a harvest's strings were read: the start or end sifts, or the middles searches stop
at. -/
inductive Level3Site
  | ends
  | middles

/-- The probe searched: its walk and its sift parted at decided reads. -/
def searches : Outcome α → Bool
  | .wrongEdge .. => true
  | .undecided _ clean => !clean
  | _ => false

/-- The undecided string the probe's start or end sift stopped at. -/
def endsString : Outcome α → List (FreeMonoid α)
  | .undecidedStart z => [z]
  | .undecided zs true => zs.take 1
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
  /-- The yield a harvest's sampler must have. -/
  harvestRate : ℝ
  /-- The rate of disagreements a consistent ending may have. -/
  disagreementRate : ℝ
  probeBudget : ℕ

/-- The tests' level under the `j`-th hypothesis. -/
noncomputable def Level3Settings.level (C : Level3Settings) (j : ℕ) : ℝ :=
  C.δ / (8 * (j + 1) * (j + 2))

noncomputable def Level3Settings.endsRate (C : Level3Settings) : ℝ := C.harvestRate

noncomputable def Level3Settings.middleRate (_C : Level3Settings) : ℝ := 1 / 4

noncomputable def Level3Settings.searchShare (C : Level3Settings) : ℝ := C.disagreementRate / 4

noncomputable def Level3Settings.endsAlternative (C : Level3Settings) : ℝ := 2 * C.endsRate

noncomputable def Level3Settings.middleAlternative (C : Level3Settings) : ℝ := 2 * C.middleRate

noncomputable def Level3Settings.disagreementAlternative (C : Level3Settings) : ℝ :=
  C.disagreementRate / 2

/-- The counts are since the hypothesis last changed; `hypotheses` counts the changes before. -/
structure Level3State (α : Type*) where
  tree : DTree α
  edges : Edges α
  hypotheses : ℕ
  probeCount : ℕ
  searchCount : ℕ
  endsUndecided : List (FreeMonoid α)
  middleUndecided : List (FreeMonoid α)

def newHypothesis (T : DTree α) (E : Edges α) (j : ℕ) : Level3State α :=
  ⟨T, E, j, 0, 0, [], []⟩

/-- The root reads at `ε`; no edge is learned. -/
def initialState : Level3State α := newHypothesis (.node 1 .leaf .leaf) (fun _ => none) 0

section

variable (read : FreeMonoid α → ARU) (C : Level3Settings)

/-- A probe counted into the stretch. -/
def countProbe (s : Level3State α) (x : FreeMonoid α) : Level3State α :=
  let o := probeOrAgree read s.tree s.edges C.walkStart x
  { s with
    probeCount := s.probeCount + 1
    searchCount := s.searchCount + if searches o then 1 else 0
    endsUndecided := s.endsUndecided ++ endsString o
    middleUndecided := s.middleUndecided ++ middleString o }

open scoped Classical in
/-- The tests, in turn: the probes stopping undecided at their start or end sift, the undecided
middles among the searches, and the disagreement rate. -/
noncomputable def runTests (s : Level3State α) : Option (RoundEnd Level3Site) :=
  if seqRate C.endsRate C.endsAlternative (C.level s.hypotheses) s.probeCount
      s.endsUndecided.length then
    some (.harvest .ends)
  else if C.searchShare * s.probeCount ≤ s.searchCount
      ∧ seqRate C.middleRate C.middleAlternative (C.level s.hypotheses) s.searchCount
        s.middleUndecided.length then
    some (.harvest .middles)
  else if seqRate C.disagreementRate C.disagreementAlternative (C.level s.hypotheses)
      s.probeCount s.searchCount then some .consistent
  else none

noncomputable def applyTests (s : Level3State α) :
    Level3State α ⊕ RoundEnd Level3Site :=
  match runTests C s with
  | some e => .inr e
  | none => .inl s

/-- The leaf `p` split on `c` and the midfix where the leaves `t` and `t₀` part. -/
def splitLeaf (s : Level3State α) (p : List Bool) (c : α) (t t₀ : List Bool) :
    Level3State α :=
  let T' := s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p
  newHypothesis T' (retarget read T' p s.edges) (s.hypotheses + 1)

noncomputable def applyProbe (s : Level3State α) (x : FreeMonoid α) :
    Level3State α ⊕ RoundEnd Level3Site :=
  match probeOrAgree read s.tree s.edges C.walkStart x with
  | .unlearnedEdge p c u t =>
    .inl (newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u))) (s.hypotheses + 1))
  | .wrongEdge p c _ t =>
    match s.edges (p, c) with
    | some (t₀, _) => .inl (splitLeaf read s p c t t₀)
    | none => applyTests C (countProbe read C s x)
  | _ => applyTests C (countProbe read C s x)

/-- The state the probes end in, and how; `failed` where they ran out. -/
noncomputable def runProbes :
    Level3State α → List (FreeMonoid α) → Level3State α × RoundEnd Level3Site
  | s, [] => (s, .failed)
  | s, x :: xs =>
    match applyProbe read C s x with
    | .inl s' => runProbes s' xs
    | .inr e => (s, e)

structure Level3Result (α : Type*) where
  hypothesis : HypothesisDFA α
  ending : RoundEnd Level3Site

end

noncomputable def indecisiveLStarLevel3 (settings : Level3Settings)
    (read : FreeMonoid α → ARU) {P : ℕ} (draws : Fin P → FreeMonoid α) : Level3Result α :=
  let r := runProbes read settings initialState (List.ofFn draws)
  ⟨⟨r.1.tree, r.1.edges⟩, r.2⟩

end Random

end OrthoDFA
