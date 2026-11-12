import RLStar.Code.Level3HarvestCode

/-!
# Level 3 with local walk starts

Level 3 (`indecisiveLStarLevel3`) with each probe walked from its own start: past every prefix of
it the round has expanded (`walkStartIn`). The round starts with every string shorter than
`walkStart` expanded (`completeTrie`). Its counts and tests are level 3's.

Known modelling gap. The Python skips a probe whose harvest string a counted probe read since the
hypothesis last changed: it is not counted, the probe that first read the string is uncounted, and
both walk starts are expanded. Here such a probe is counted, as level 3 counts it.

Known modelling gap. The Python expands lazily, only on repeats, from no expanded prefix; here
every string shorter than `walkStart` is expanded before the first probe.

Known modelling gap. The Python also runs its starts and stuck tests, which are not modelled here.
-/

namespace OrthoDFA

namespace Random

variable {α : Type*}

open scoped Classical in
/-- The least `k` with `x`'s first `k` letters outside `X`, or `|x|`. -/
noncomputable def walkStartIn (X : Finset (FreeMonoid α)) (x : FreeMonoid α) : ℕ :=
  ((List.range x.toList.length).find? fun k => prefixOf x k ∉ X).getD x.toList.length

open scoped Classical in
/-- The strings shorter than `K`. -/
noncomputable def completeTrie [Fintype α] (K : ℕ) : Finset (FreeMonoid α) :=
  (Finset.range K).biUnion fun j =>
    (Finset.univ : Finset (Fin j → α)).image fun v => FreeMonoid.ofList (List.ofFn v)

structure TrieState (α : Type*) where
  base : Level3State α
  expanded : Finset (FreeMonoid α)

structure TrieResult (α : Type*) where
  hypothesis : HypothesisDFA α
  expanded : Finset (FreeMonoid α)
  ending : RoundEnd Level3Site

section Query

variable [DecidableEq α] [Fintype α] (C : Level3Settings)

/-- Level 3's step on a probe with outcome `o`: `applyProbeQ` after the probe. -/
noncomputable def level3StepQ (s : Level3State α) (o : Outcome α) :
    Query (FreeMonoid α) ARU (Level3State α ⊕ RoundEnd Level3Site) :=
  match o with
  | .unlearnedEdge p c u t =>
    .pure (.inl (newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u)))
      (s.hypotheses + 1)))
  | .wrongEdge p c _ t =>
    match s.edges (p, c) with
    | some (t₀, _) => (splitLeafQ s p c t t₀).map .inl
    | none => .pure (applyTests C (countOutcome s o))
  | _ => .pure (applyTests C (countOutcome s o))

noncomputable def applyTrieProbeQ (s : TrieState α) (x : FreeMonoid α) :
    Query (FreeMonoid α) ARU (TrieState α ⊕ RoundEnd Level3Site) :=
  (probeOrAgreeQ s.base.tree s.base.edges (walkStartIn s.expanded x) x).bind fun o =>
    (level3StepQ C s.base o).map (Sum.map (fun b => ⟨b, s.expanded⟩) id)

/-- The state the probes end in, and how; `failed` where they ran out. -/
noncomputable def runTrieProbesQ :
    TrieState α → List (FreeMonoid α) →
      Query (FreeMonoid α) ARU (TrieState α × RoundEnd Level3Site)
  | s, [] => .pure (s, .failed)
  | s, x :: xs =>
    (applyTrieProbeQ C s x).bind fun
    | .inl s' => runTrieProbesQ s' xs
    | .inr e => .pure (s, e)

/-- The root reads at `ε`, no edge is learned, and every string shorter than `walkStart` is
expanded. -/
noncomputable def trieInitialState : TrieState α :=
  ⟨initialState, completeTrie C.walkStart⟩

noncomputable def level3TrieQ {P : ℕ} (draws : Fin P → FreeMonoid α) :
    Query (FreeMonoid α) ARU (TrieResult α) :=
  (runTrieProbesQ C (trieInitialState C) (List.ofFn draws)).map fun r =>
    ⟨⟨r.1.base.tree, r.1.base.edges⟩, r.1.expanded, r.2⟩

noncomputable def indecisiveLStarLevel3Trie (read : FreeMonoid α → ARU) {P : ℕ}
    (draws : Fin P → FreeMonoid α) : TrieResult α :=
  (level3TrieQ C draws).eval read

end Query

/-! ## The harvest's sampler -/

section Sampler

variable [DecidableEq α]

open scoped Classical in
/-- The probes in turn, from the expanded prefixes `X`, up to the first that harvests a string
outside `T`; a probe harvesting one inside `T` has its walk start, if shorter than `cap`,
expanded. The string, and the expanded prefixes after. -/
noncomputable def trieHarvestFirstQ (cap : ℕ) (H : HypothesisDFA α) (site : Level3Site)
    (T : Set (FreeMonoid α)) :
    Finset (FreeMonoid α) → List (FreeMonoid α) →
      Query (FreeMonoid α) ARU (Option (FreeMonoid α) × Finset (FreeMonoid α))
  | X, [] => .pure (none, X)
  | X, x :: xs => (siteHarvestQ (walkStartIn X x) H site x).bind fun l =>
    match l.head? with
    | none => trieHarvestFirstQ cap H site T X xs
    | some z =>
      if z ∈ T then
        let u := prefixOf x (walkStartIn X x)
        trieHarvestFirstQ cap H site T (if u.toList.length < cap then insert u X else X) xs
      else .pure (some z, X)

end Sampler

end Random

end OrthoDFA
