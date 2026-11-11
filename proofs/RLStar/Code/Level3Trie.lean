import RLStar.Code.Level3HarvestCode

/-!
# Level 3 with local walk starts

Level 3 (`indecisiveLStarLevel3`) with each probe walked from its own start: past every prefix of
it the round has expanded (`walkStartIn`). The round starts with every string shorter than
`walkStart` expanded (`completeTrie`). Its counts and tests are level 3's, but:
* a probe whose harvest string (`harvestString`) a counted probe read since the hypothesis last
  changed is skipped: it is not counted, the probe that first read the string is uncounted, and
  both walk starts shorter than `cap` are expanded.

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

/-- `reads` holds each harvest string a counted probe read since the hypothesis last changed, with
that probe's walk start and outcome, and `repeated` those read again. -/
structure TrieState (α : Type*) where
  base : Level3State α
  expanded : Finset (FreeMonoid α)
  reads : List (FreeMonoid α × FreeMonoid α × Outcome α)
  repeated : List (FreeMonoid α)

section

open scoped Classical

/-- `z` marked repeated, its first reader uncounted unless it already was. -/
noncomputable def TrieState.drop (s : TrieState α) (z : FreeMonoid α) : TrieState α :=
  if z ∈ s.repeated then s else
  match s.reads.find? (fun r => r.1 = z) with
  | some r => { s with base := uncountOutcome s.base r.2.2, repeated := z :: s.repeated }
  | none => { s with repeated := z :: s.repeated }

def TrieState.Repeats (s : TrieState α) (o : Outcome α) : Prop :=
  ∃ z ∈ harvestString o, z ∈ s.reads.map Prod.fst

/-- The probe with outcome `o`, walked from `u`, skipped: the strings it repeats dropped, and its
walk start and those of their first readers expanded, those shorter than `cap`. -/
noncomputable def TrieState.skipRepeat (cap : ℕ) (s : TrieState α) (o : Outcome α)
    (u : FreeMonoid α) : TrieState α :=
  let rep := (harvestString o).filter fun z => z ∈ s.reads.map Prod.fst
  let starts := u :: (s.reads.filter fun r => r.1 ∈ rep).map fun r => r.2.1
  let s' := rep.foldl TrieState.drop s
  { s' with expanded := s.expanded ∪ (starts.filter fun v => v.toList.length < cap).toFinset }

/-- The probe with outcome `o`, walked from `u`, after level 3's step took the counts to `b`:
recorded if the hypothesis stands, else the reads restart. -/
def TrieState.record (s : TrieState α) (o : Outcome α) (u : FreeMonoid α) (b : Level3State α) :
    TrieState α :=
  if b.hypotheses = s.base.hypotheses then
    ⟨b, s.expanded, s.reads ++ (harvestString o).map fun z => (z, u, o), s.repeated⟩
  else ⟨b, s.expanded, [], []⟩

end

structure TrieResult (α : Type*) where
  hypothesis : HypothesisDFA α
  expanded : Finset (FreeMonoid α)
  ending : RoundEnd Level3Site

section Query

variable [DecidableEq α] [Fintype α] (C : Level3Settings) (cap : ℕ)

open scoped Classical in
noncomputable def applyTrieProbeQ (s : TrieState α) (x : FreeMonoid α) :
    Query (FreeMonoid α) ARU (TrieState α ⊕ RoundEnd Level3Site) :=
  let k := walkStartIn s.expanded x
  (probeOrAgreeQ s.base.tree s.base.edges k x).bind fun o =>
    if s.Repeats o then .pure (.inl (s.skipRepeat cap o (prefixOf x k)))
    else (level3StepQ C s.base o).map (Sum.map (s.record o (prefixOf x k)) id)

/-- The state the probes end in, and how; `failed` where they ran out. -/
noncomputable def runTrieProbesQ :
    TrieState α → List (FreeMonoid α) →
      Query (FreeMonoid α) ARU (TrieState α × RoundEnd Level3Site)
  | s, [] => .pure (s, .failed)
  | s, x :: xs =>
    (applyTrieProbeQ C cap s x).bind fun
    | .inl s' => runTrieProbesQ s' xs
    | .inr e => .pure (s, e)

/-- The root reads at `ε`, no edge is learned, and every string shorter than `walkStart` is
expanded. -/
noncomputable def trieInitialState : TrieState α :=
  ⟨initialState, completeTrie C.walkStart, [], []⟩

noncomputable def level3TrieQ {P : ℕ} (draws : Fin P → FreeMonoid α) :
    Query (FreeMonoid α) ARU (TrieResult α) :=
  (runTrieProbesQ C cap (trieInitialState C) (List.ofFn draws)).map fun r =>
    ⟨⟨r.1.base.tree, r.1.base.edges⟩, r.1.expanded, r.2⟩

noncomputable def indecisiveLStarLevel3Trie (read : FreeMonoid α → ARU) {P : ℕ}
    (draws : Fin P → FreeMonoid α) : TrieResult α :=
  (level3TrieQ C cap draws).eval read

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
