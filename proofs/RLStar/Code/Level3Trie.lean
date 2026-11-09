import RLStar.Code.Level3HarvestCode

/-!
# Level 3 with local walk starts

Level 3 (`indecisiveLStarLevel3`) with each probe walked from its own start: past every prefix of
it the round has expanded (`walkStartIn`), none at first.
* Before level 3's tests, the starts test fires on the probes stopping undecided at their start
  sift, at the ends' rate.
* Where a harvest test fires, each string it would harvest that it counted twice or more, and
  that makes up at least a quarter of its strings, has the walk starts of the probes that read it
  expanded in place of the harvest; the harvest stands only where there is no such string.
Every expansion, like every split and every edge learned, is a new hypothesis: the counts restart.

Known modelling gap. The starts test runs at the ends' rate and alternative, where the Python's runs
at its stuck rate.
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

/-- The counts are since the hypothesis last changed; `hypotheses` counts the changes before.
`origins` pairs each string counted at the ends or the middles with the walk start of the probe
that read it. -/
structure TrieState (α : Type*) where
  tree : DTree α
  edges : Edges α
  expanded : Finset (FreeMonoid α)
  hypotheses : ℕ
  probeCount : ℕ
  searchCount : ℕ
  endsUndecided : List (FreeMonoid α)
  middleUndecided : List (FreeMonoid α)
  startsUndecided : List (FreeMonoid α)
  origins : List (FreeMonoid α × FreeMonoid α)

def newTrieHypothesis (T : DTree α) (E : Edges α) (X : Finset (FreeMonoid α)) (j : ℕ) :
    TrieState α :=
  ⟨T, E, X, j, 0, 0, [], [], [], []⟩

/-- The root reads at `ε`; no edge is learned and no prefix expanded. -/
def trieInitialState : TrieState α := newTrieHypothesis (.node 1 .leaf .leaf) (fun _ => none) ∅ 0

/-- The strings a harvest at the site takes. -/
def TrieState.siteStrings (s : TrieState α) : TrieSite → List (FreeMonoid α)
  | .starts => s.startsUndecided
  | .ends => s.endsUndecided
  | .middles => s.middleUndecided

section

variable (C : Level3Settings)

/-- The probe with outcome `o`, walked from `u`, counted into the stretch. -/
def countTrieOutcome (s : TrieState α) (o : Outcome α) (u : FreeMonoid α) : TrieState α :=
  { s with
    probeCount := s.probeCount + 1
    searchCount := s.searchCount + if searches o then 1 else 0
    endsUndecided := s.endsUndecided ++ endsString o
    middleUndecided := s.middleUndecided ++ middleString o
    startsUndecided := s.startsUndecided ++ startString o
    origins := s.origins ++ (endsString o ++ middleString o).map fun z => (z, u) }

open scoped Classical in
/-- The tests, in turn: the probes stopping undecided at their start sift, then level 3's. -/
noncomputable def runTrieTests (s : TrieState α) : Option (RoundEnd TrieSite) :=
  if seqRate C.endsRate C.endsAlternative (C.level s.hypotheses) s.probeCount
      s.startsUndecided.length then
    some (.harvest .starts)
  else if seqRate C.endsRate C.endsAlternative (C.level s.hypotheses) s.probeCount
      s.endsUndecided.length then
    some (.harvest .ends)
  else if C.searchShare * s.probeCount ≤ s.searchCount
      ∧ seqRate C.middleRate C.middleAlternative (C.level s.hypotheses) s.searchCount
        s.middleUndecided.length then
    some (.harvest .middles)
  else if seqRate C.disagreementRate C.disagreementAlternative (C.level s.hypotheses)
      s.probeCount (s.searchCount + s.endsUndecided.length) then some .consistent
  else none

open scoped Classical in
/-- The walk starts of the probes that read a string the site counted at least twice and as at
least a quarter of its strings. -/
noncomputable def harvestExpands (s : TrieState α) (site : TrieSite) : Finset (FreeMonoid α) :=
  ((s.origins.filter fun p => 2 ≤ (s.siteStrings site).count p.1
      ∧ (s.siteStrings site).length ≤ 4 * (s.siteStrings site).count p.1).map Prod.snd).toFinset

open scoped Classical in
/-- `X` expanded: a new hypothesis. -/
noncomputable def TrieState.expand (s : TrieState α) (X : Finset (FreeMonoid α)) : TrieState α :=
  newTrieHypothesis s.tree s.edges (s.expanded ∪ X) (s.hypotheses + 1)

open scoped Classical in
/-- The tests' verdict, a harvest giving way to the walk starts it expands. -/
noncomputable def applyTrieTests (s : TrieState α) : TrieState α ⊕ RoundEnd TrieSite :=
  match runTrieTests C s with
  | some (.harvest site) =>
    if (harvestExpands s site).Nonempty then .inl (s.expand (harvestExpands s site))
    else .inr (.harvest site)
  | some e => .inr e
  | none => .inl s

structure TrieResult (α : Type*) where
  hypothesis : HypothesisDFA α
  expanded : Finset (FreeMonoid α)
  ending : RoundEnd TrieSite

end

section Query

variable [DecidableEq α] [Fintype α] (C : Level3Settings)

/-- The leaf `p` split on `c` and the midfix where the leaves `t` and `t₀` part. -/
noncomputable def splitTrieLeafQ (s : TrieState α) (p : List Bool) (c : α) (t t₀ : List Bool) :
    Query (FreeMonoid α) ARU (TrieState α) :=
  ((HypothesisDFA.mk s.tree s.edges).splitQ
    (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p).map
    fun H => newTrieHypothesis H.tree H.edges s.expanded (s.hypotheses + 1)

open scoped Classical in
noncomputable def applyTrieProbeQ (s : TrieState α) (x : FreeMonoid α) :
    Query (FreeMonoid α) ARU (TrieState α ⊕ RoundEnd TrieSite) :=
  let k := walkStartIn s.expanded x
  (probeOrAgreeQ s.tree s.edges k x).bind fun o =>
    match o with
    | .unlearnedEdge p c u t =>
      .pure (.inl (newTrieHypothesis s.tree (Function.update s.edges (p, c) (some (t, u)))
        s.expanded (s.hypotheses + 1)))
    | .wrongEdge p c _ t =>
      match s.edges (p, c) with
      | some (t₀, _) => (splitTrieLeafQ s p c t t₀).map .inl
      | none => .pure (applyTrieTests C (countTrieOutcome s o (prefixOf x k)))
    | _ => .pure (applyTrieTests C (countTrieOutcome s o (prefixOf x k)))

/-- The state the probes end in, and how; `failed` where they ran out. -/
noncomputable def runTrieProbesQ :
    TrieState α → List (FreeMonoid α) → Query (FreeMonoid α) ARU (TrieState α × RoundEnd TrieSite)
  | s, [] => .pure (s, .failed)
  | s, x :: xs =>
    (applyTrieProbeQ C s x).bind fun
    | .inl s' => runTrieProbesQ s' xs
    | .inr e => .pure (s, e)

noncomputable def level3TrieQ {P : ℕ} (draws : Fin P → FreeMonoid α) :
    Query (FreeMonoid α) ARU (TrieResult α) :=
  (runTrieProbesQ C trieInitialState (List.ofFn draws)).map fun r =>
    ⟨⟨r.1.tree, r.1.edges⟩, r.1.expanded, r.2⟩

noncomputable def indecisiveLStarLevel3Trie (read : FreeMonoid α → ARU) {P : ℕ}
    (draws : Fin P → FreeMonoid α) : TrieResult α :=
  (level3TrieQ C draws).eval read

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
/-- The probes in turn, up to the first that harvests a string outside `T`. -/
noncomputable def trieHarvestFirstQ (X : Finset (FreeMonoid α)) (H : HypothesisDFA α)
    (site : TrieSite) (T : Set (FreeMonoid α)) (xs : List (FreeMonoid α)) :
    Query (FreeMonoid α) ARU (Option (FreeMonoid α)) :=
  Query.findSome
    (fun x => (trieSiteHarvestQ X H site x).map fun l => l.head?.filter fun z => z ∉ T) xs

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
