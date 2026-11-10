import RLStar.Code.Level3HarvestCode

/-!
# Level 3 with local walk starts

Level 3 (`indecisiveLStarLevel3`) with each probe walked from its own start: past every prefix of
it the round has expanded (`walkStartIn`), none at first.
* Before level 3's tests, the starts test fires on the probes stopping undecided at their start
  sift. It and the ends test run at the harvest rate `y`, against `2y`.
* A probe whose harvest string (`harvestString`) was read before since the hypothesis last changed
  is skipped: no test runs on it, and its walk start and that of the probe that first read the
  string are expanded, those shorter than `cap`. The harvest tests count neither it nor that first
  probe.
* The success test counts every probe since the hypothesis or the expanded prefixes last changed,
  a skipped one as disagreeing, at the level of that stretch: the `j`-th stretch of the round tests
  at `δ / (8 (j + 1) (j + 2))`.

Known modelling gap. The starts and ends tests run at `y`, where the Python's run at its stuck rate
and at its ends rate.

Known modelling gap. The Python's success test counts neither a skipped probe nor the probe that
first read its string (`SiteCounts.drop`), where here both count, the skipped one as disagreeing.

Known modelling gap. The Python's success test keeps counting across expansions, at the level of the
hypothesis, where here an expansion that changes the expanded prefixes restarts it.
-/

namespace OrthoDFA

namespace Random

variable {α : Type*}

open scoped Classical in
/-- The least `k` with `x`'s first `k` letters outside `X`, or `|x|`. -/
noncomputable def walkStartIn (X : Finset (FreeMonoid α)) (x : FreeMonoid α) : ℕ :=
  ((List.range x.toList.length).find? fun k => prefixOf x k ∉ X).getD x.toList.length

/-- Where a harvest's strings were read: the start sifts, the start or end sifts, or the middles
searches stop at. -/
inductive TrieSite
  | starts
  | ends
  | middles

/-- The undecided string the probe's start sift stopped at. -/
def startString : Outcome α → List (FreeMonoid α)
  | .undecidedStart z => [z]
  | _ => []

def harvestString (o : Outcome α) : List (FreeMonoid α) := endsString o ++ middleString o

/-- The harvest counts are since the hypothesis last changed; `hypotheses` counts the changes
before. `counted` holds the outcomes of the probes since and `seen` each harvest string read with
the walk start of the probe that first read it. The success counts are since the hypothesis or the
expanded prefixes last changed; `stretches` counts the changes before. -/
structure TrieState (α : Type*) where
  tree : DTree α
  edges : Edges α
  expanded : Finset (FreeMonoid α)
  hypotheses : ℕ
  counted : List (Outcome α)
  seen : List (FreeMonoid α × FreeMonoid α)
  stretches : ℕ
  successProbes : ℕ
  successDisagree : ℕ

def newTrieHypothesis (T : DTree α) (E : Edges α) (X : Finset (FreeMonoid α)) (j m : ℕ) :
    TrieState α :=
  ⟨T, E, X, j, [], [], m, 0, 0⟩

/-- The root reads at `ε`; no edge is learned and no prefix expanded. -/
def trieInitialState : TrieState α :=
  newTrieHypothesis (.node 1 .leaf .leaf) (fun _ => none) ∅ 0 0

/-- What the success test counts as disagreeing. -/
def successDisagrees (o : Outcome α) : Bool := searches o || !(endsString o).isEmpty

def TrieState.probeCount (s : TrieState α) : ℕ := s.counted.length

def TrieState.searchCount (s : TrieState α) : ℕ := s.counted.countP searches

/-- The strings a harvest at the site takes. -/
def TrieState.siteStrings (s : TrieState α) : TrieSite → List (FreeMonoid α)
  | .starts => s.counted.flatMap startString
  | .ends => s.counted.flatMap endsString
  | .middles => s.counted.flatMap middleString

section

variable (C : Level3Settings) (y : ℝ)

/-- The probe with outcome `o`, walked from `u`, counted into the stretch. -/
def countTrieOutcome (s : TrieState α) (o : Outcome α) (u : FreeMonoid α) : TrieState α :=
  { s with
    counted := s.counted ++ [o]
    seen := s.seen ++ (harvestString o).map fun z => (z, u)
    successProbes := s.successProbes + 1
    successDisagree := s.successDisagree + if successDisagrees o then 1 else 0 }

open scoped Classical in
/-- The prefixes a skip of a probe with outcome `o`, walked from `u`, expands. -/
noncomputable def TrieState.skipExpands (cap : ℕ) (s : TrieState α) (o : Outcome α)
    (u : FreeMonoid α) : Finset (FreeMonoid α) :=
  s.expanded ∪ ((u :: (s.seen.filter fun p => p.1 ∈ harvestString o).map Prod.snd).filter
    fun v => v.toList.length < cap).toFinset

open scoped Classical in
/-- The probe with outcome `o`, walked from `u`, skipped: uncounted, with the probe that first read
its harvest string, and their walk starts expanded, those shorter than `cap`. -/
noncomputable def TrieState.skipRepeat (cap : ℕ) (s : TrieState α) (o : Outcome α)
    (u : FreeMonoid α) : TrieState α :=
  let X := s.skipExpands cap o u
  { s with
    counted := s.counted.filter fun o' => ∀ z ∈ harvestString o', z ∉ harvestString o
    expanded := X
    stretches := if X = s.expanded then s.stretches else s.stretches + 1
    successProbes := if X = s.expanded then s.successProbes + 1 else 0
    successDisagree := if X = s.expanded then s.successDisagree + 1 else 0 }

open scoped Classical in
/-- The tests, in turn: the probes stopping undecided at their start sift, then level 3's. -/
noncomputable def runTrieTests (s : TrieState α) : Option (RoundEnd TrieSite) :=
  if seqRate y (2 * y) (C.level s.hypotheses) s.probeCount
      (s.siteStrings .starts).length then
    some (.harvest .starts)
  else if seqRate y (2 * y) (C.level s.hypotheses) s.probeCount
      (s.siteStrings .ends).length then
    some (.harvest .ends)
  else if C.searchShare * s.probeCount ≤ s.searchCount
      ∧ seqRate C.middleRate C.middleAlternative (C.level s.hypotheses) s.searchCount
        (s.siteStrings .middles).length then
    some (.harvest .middles)
  else if seqRate C.disagreementRate C.disagreementAlternative (C.level s.stretches)
      s.successProbes s.successDisagree then some .consistent
  else none

noncomputable def applyTrieTests (s : TrieState α) : TrieState α ⊕ RoundEnd TrieSite :=
  match runTrieTests C y s with
  | some e => .inr e
  | none => .inl s

structure TrieResult (α : Type*) where
  hypothesis : HypothesisDFA α
  expanded : Finset (FreeMonoid α)
  ending : RoundEnd TrieSite

end

section Query

variable [DecidableEq α] [Fintype α] (C : Level3Settings) (y : ℝ) (cap : ℕ)

/-- The leaf `p` split on `c` and the midfix where the leaves `t` and `t₀` part. -/
noncomputable def splitTrieLeafQ (s : TrieState α) (p : List Bool) (c : α) (t t₀ : List Bool) :
    Query (FreeMonoid α) ARU (TrieState α) :=
  ((HypothesisDFA.mk s.tree s.edges).splitQ
    (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p).map
    fun H => newTrieHypothesis H.tree H.edges s.expanded (s.hypotheses + 1) (s.stretches + 1)

open scoped Classical in
noncomputable def applyTrieProbeQ (s : TrieState α) (x : FreeMonoid α) :
    Query (FreeMonoid α) ARU (TrieState α ⊕ RoundEnd TrieSite) :=
  let k := walkStartIn s.expanded x
  (probeOrAgreeQ s.tree s.edges k x).bind fun o =>
    if ∃ z ∈ harvestString o, z ∈ s.seen.map Prod.fst then
      .pure (.inl (s.skipRepeat cap o (prefixOf x k)))
    else
    match o with
    | .unlearnedEdge p c u t =>
      .pure (.inl (newTrieHypothesis s.tree (Function.update s.edges (p, c) (some (t, u)))
        s.expanded (s.hypotheses + 1) (s.stretches + 1)))
    | .wrongEdge p c _ t =>
      match s.edges (p, c) with
      | some (t₀, _) => (splitTrieLeafQ s p c t t₀).map .inl
      | none => .pure (applyTrieTests C y (countTrieOutcome s o (prefixOf x k)))
    | _ => .pure (applyTrieTests C y (countTrieOutcome s o (prefixOf x k)))

/-- The state the probes end in, and how; `failed` where they ran out. -/
noncomputable def runTrieProbesQ :
    TrieState α → List (FreeMonoid α) → Query (FreeMonoid α) ARU (TrieState α × RoundEnd TrieSite)
  | s, [] => .pure (s, .failed)
  | s, x :: xs =>
    (applyTrieProbeQ C y cap s x).bind fun
    | .inl s' => runTrieProbesQ s' xs
    | .inr e => .pure (s, e)

noncomputable def level3TrieQ {P : ℕ} (draws : Fin P → FreeMonoid α) :
    Query (FreeMonoid α) ARU (TrieResult α) :=
  (runTrieProbesQ C y cap trieInitialState (List.ofFn draws)).map fun r =>
    ⟨⟨r.1.tree, r.1.edges⟩, r.1.expanded, r.2⟩

noncomputable def indecisiveLStarLevel3Trie (read : FreeMonoid α → ARU) {P : ℕ}
    (draws : Fin P → FreeMonoid α) : TrieResult α :=
  (level3TrieQ C y cap draws).eval read

end Query

/-! ## The harvest's sampler -/

section Sampler

variable [DecidableEq α]

/-- The strings a probe of `x`, walked past `X` under `H`, reads undecided at the site, as the
tests count them. -/
noncomputable def trieSiteHarvestQ (X : Finset (FreeMonoid α)) (H : HypothesisDFA α)
    (site : TrieSite) (x : FreeMonoid α) : Query (FreeMonoid α) ARU (List (FreeMonoid α)) :=
  (probeOrAgreeQ H.tree H.edges (walkStartIn X x) x).map fun o =>
    match site with
    | .starts => startString o
    | .ends => endsString o
    | .middles => middleString o

noncomputable def trieSiteHarvest (read : FreeMonoid α → ARU) (X : Finset (FreeMonoid α))
    (H : HypothesisDFA α) (site : TrieSite) (x : FreeMonoid α) : List (FreeMonoid α) :=
  (trieSiteHarvestQ X H site x).eval read

open scoped Classical in
/-- The probes in turn, from the expanded prefixes `X`, up to the first that harvests a string
outside `T`; a probe harvesting one inside `T` has its walk start, if shorter than `cap`,
expanded. The string, and the expanded prefixes after. -/
noncomputable def trieHarvestFirstQ (cap : ℕ) (H : HypothesisDFA α) (site : TrieSite)
    (T : Set (FreeMonoid α)) :
    Finset (FreeMonoid α) → List (FreeMonoid α) →
      Query (FreeMonoid α) ARU (Option (FreeMonoid α) × Finset (FreeMonoid α))
  | X, [] => .pure (none, X)
  | X, x :: xs => (trieSiteHarvestQ X H site x).bind fun l =>
    match l.head? with
    | none => trieHarvestFirstQ cap H site T X xs
    | some z =>
      if z ∈ T then
        let u := prefixOf x (walkStartIn X x)
        trieHarvestFirstQ cap H site T (if u.toList.length < cap then insert u X else X) xs
      else .pure (some z, X)

open MeasureTheory in
/-- The law of what the harvest's sampler finds, each string as often as a probe drawn from `D`
and walked past `X` under `H` finds it at the site. -/
noncomputable def trieHarvestLaw [Countable α] (read : FreeMonoid α → ARU)
    (D : Measure (FreeMonoid α)) (X : Finset (FreeMonoid α)) (H : HypothesisDFA α)
    (site : TrieSite) : Measure (FreeMonoid α) :=
  D.bind fun x => ((trieSiteHarvest read X H site x).map Measure.dirac).sum

end Sampler

end Random

end OrthoDFA
