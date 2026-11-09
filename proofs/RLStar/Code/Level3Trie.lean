import RLStar.Code.Level3HarvestCode

/-!
# Level 3 with local walk starts

Level 3 (`indecisiveLStarLevel3`) with each probe walked from its own start: past every prefix of
it the round has expanded (`walkStartIn`), none at first.
* Before level 3's tests, the starts test fires on the probes stopping undecided at their start
  sift, at the ends' rate.
* A probe whose harvest string (`harvestString`) was read before since the hypothesis last changed
  is skipped, and the probe that first read it is uncounted; the walk starts of both are expanded,
  those shorter than `cap`. The other counts stand, and no test runs on a skip.

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

def harvestString (o : Outcome α) : List (FreeMonoid α) := endsString o ++ middleString o

/-- The counts are since the hypothesis last changed; `hypotheses` counts the changes before.
`counted` holds the outcomes of the probes counted, and `seen` each harvest string read with the
walk start of the probe that first read it. -/
structure TrieState (α : Type*) where
  tree : DTree α
  edges : Edges α
  expanded : Finset (FreeMonoid α)
  hypotheses : ℕ
  counted : List (Outcome α)
  seen : List (FreeMonoid α × FreeMonoid α)

def newTrieHypothesis (T : DTree α) (E : Edges α) (X : Finset (FreeMonoid α)) (j : ℕ) :
    TrieState α :=
  ⟨T, E, X, j, [], []⟩

/-- The root reads at `ε`; no edge is learned and no prefix expanded. -/
def trieInitialState : TrieState α := newTrieHypothesis (.node 1 .leaf .leaf) (fun _ => none) ∅ 0

def TrieState.probeCount (s : TrieState α) : ℕ := s.counted.length

def TrieState.searchCount (s : TrieState α) : ℕ := s.counted.countP searches

/-- The strings a harvest at the site takes. -/
def TrieState.siteStrings (s : TrieState α) : TrieSite → List (FreeMonoid α)
  | .starts => s.counted.flatMap startString
  | .ends => s.counted.flatMap endsString
  | .middles => s.counted.flatMap middleString

section

variable (C : Level3Settings)

/-- The probe with outcome `o`, walked from `u`, counted into the stretch. -/
def countTrieOutcome (s : TrieState α) (o : Outcome α) (u : FreeMonoid α) : TrieState α :=
  { s with
    counted := s.counted ++ [o]
    seen := s.seen ++ (harvestString o).map fun z => (z, u) }

open scoped Classical in
/-- The probe with outcome `o`, walked from `u`, skipped: the probe that first read its harvest
string uncounted, and both walk starts shorter than `cap` expanded. -/
noncomputable def TrieState.skipRepeat (cap : ℕ) (s : TrieState α) (o : Outcome α)
    (u : FreeMonoid α) : TrieState α :=
  let R := harvestString o
  { s with
    counted := s.counted.filter fun o' => ∀ z ∈ harvestString o', z ∉ R
    expanded := s.expanded ∪ ((u :: (s.seen.filter fun p => p.1 ∈ R).map Prod.snd).filter
      fun v => v.toList.length < cap).toFinset }

open scoped Classical in
/-- The tests, in turn: the probes stopping undecided at their start sift, then level 3's. -/
noncomputable def runTrieTests (s : TrieState α) : Option (RoundEnd TrieSite) :=
  if seqRate C.endsRate C.endsAlternative (C.level s.hypotheses) s.probeCount
      (s.siteStrings .starts).length then
    some (.harvest .starts)
  else if seqRate C.endsRate C.endsAlternative (C.level s.hypotheses) s.probeCount
      (s.siteStrings .ends).length then
    some (.harvest .ends)
  else if C.searchShare * s.probeCount ≤ s.searchCount
      ∧ seqRate C.middleRate C.middleAlternative (C.level s.hypotheses) s.searchCount
        (s.siteStrings .middles).length then
    some (.harvest .middles)
  else if seqRate C.disagreementRate C.disagreementAlternative (C.level s.hypotheses)
      s.probeCount (s.searchCount + (s.siteStrings .ends).length) then some .consistent
  else none

noncomputable def applyTrieTests (s : TrieState α) : TrieState α ⊕ RoundEnd TrieSite :=
  match runTrieTests C s with
  | some e => .inr e
  | none => .inl s

structure TrieResult (α : Type*) where
  hypothesis : HypothesisDFA α
  expanded : Finset (FreeMonoid α)
  ending : RoundEnd TrieSite

end

section Query

variable [DecidableEq α] [Fintype α] (C : Level3Settings) (cap : ℕ)

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
    if ∃ z ∈ harvestString o, z ∈ s.seen.map Prod.fst then
      .pure (.inl (s.skipRepeat cap o (prefixOf x k)))
    else
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
    (applyTrieProbeQ C cap s x).bind fun
    | .inl s' => runTrieProbesQ s' xs
    | .inr e => .pure (s, e)

noncomputable def level3TrieQ {P : ℕ} (draws : Fin P → FreeMonoid α) :
    Query (FreeMonoid α) ARU (TrieResult α) :=
  (runTrieProbesQ C cap trieInitialState (List.ofFn draws)).map fun r =>
    ⟨⟨r.1.tree, r.1.edges⟩, r.1.expanded, r.2⟩

noncomputable def indecisiveLStarLevel3Trie (read : FreeMonoid α → ARU) {P : ℕ}
    (draws : Fin P → FreeMonoid α) : TrieResult α :=
  (level3TrieQ C cap draws).eval read

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
