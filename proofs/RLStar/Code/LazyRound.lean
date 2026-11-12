import RLStar.Code.TrieTable
import RLStar.Code.Level3Harvest

/-!
# A round with a lazy trie

The round keeps a trie `X` of expanded prefixes, every one shorter than `K`, and walks each probe
from its first prefix outside `X`. Level 3 runs in epochs, each with `X` frozen. An epoch is
level 3 from its first hypothesis, walked `K` letters into the probe padded by `K − ws_X(x)` pad
letters, each question read with its pads dropped. So it reads exactly the strings the walk from
`X` does. An epoch that harvests at a site starts a top-up. The top-up runs the epoch's blocks of
fresh probes, appends the site's new strings to the round's list for that site, and expands the walk
start of every repeated string whose walk start is shorter than `K`. The round harvests once a
list reaches `nh` strings. A block with neither appends nor expansions retracts: the next epoch
runs with the grown trie.

Known modelling gap. A retraction restarts level 3 from its first hypothesis, and each epoch
tests at its own level. The Python keeps its hypothesis and its counts.

Known modelling gap. The lists are kept per site for the whole round, and the round's population
is uniform over the first `nh` strings of the list that fills. The Python's population is the
strings its harvest draws.

Known modelling gap. The top-up runs blocks of a fixed size, expands at each block's end, and
retracts at the first block without progress. The Python's top-up budget differs.

Known modelling gap. Level 3 runs over the alphabet with a pad letter. It reads the same strings
as the walk from `X`, and the hypotheses it learns have no pad edges.
-/

noncomputable section

namespace OrthoDFA

namespace Random

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- `x` behind `K − ws_X(x)` pads. -/
def padIn (K : ℕ) (X : Finset (FreeMonoid α)) (x : FreeMonoid α) : FreeMonoid (Option α) :=
  FreeMonoid.ofList (List.replicate (K - walkStartIn X x) none ++ x.toList.map some)

/-- `z` without its pads. -/
def unpad (z : FreeMonoid (Option α)) : FreeMonoid α := FreeMonoid.ofList z.toList.reduceOption

def unpadQ (z : FreeMonoid (Option α)) : Query (FreeMonoid α) ARU ARU := Query.query (unpad z)

/-- Level 3 with `X` frozen. -/
def epochQ (C : Level3Settings) (K : ℕ) (X : Finset (FreeMonoid α)) {P : ℕ}
    (draws : Fin P → FreeMonoid α) : Query (FreeMonoid α) ARU (Level3Result (Option α)) :=
  (level3Q { C with walkStart := K } fun i => padIn K X (draws i)).subst unpadQ

/-- The strings the probes `xs`, walked from `X` under `H`, harvest at the site, in turn. -/
def blockQ (K : ℕ) (H : HypothesisDFA (Option α)) (s : Level3Site) (X : Finset (FreeMonoid α)) :
    List (FreeMonoid α) → Query (FreeMonoid α) ARU (List (FreeMonoid α))
  | [] => .pure []
  | x :: xs => ((siteHarvestQ K H s (padIn K X x)).subst unpadQ).bind fun l =>
    (blockQ K H s X xs).map fun ws => l.map unpad ++ ws

open scoped Classical in
/-- The list with each new string of `ws` appended, and the walk starts in `X`, shorter than `K`,
of the strings `ws` repeats. -/
def blockUpdate (K : ℕ) (X : Finset (FreeMonoid α)) :
    List (FreeMonoid α) → List (FreeMonoid α) → List (FreeMonoid α) × Finset (FreeMonoid α)
  | ℓ, [] => (ℓ, ∅)
  | ℓ, w :: ws =>
    if w ∈ ℓ then
      let r := blockUpdate K X ℓ ws
      (r.1, if walkStartIn X w < K then insert (prefixOf w (walkStartIn X w)) r.2 else r.2)
    else blockUpdate K X (ℓ ++ [w]) ws

inductive TopUpEnd (α : Type*)
  | full (ℓ : List (FreeMonoid α))
  | retract (ℓ : List (FreeMonoid α)) (X : Finset (FreeMonoid α))

open scoped Classical in
/-- The blocks in turn until the list reaches `nh` strings, or a block grows neither the list nor
`X`; with the blocks left, `none` where they run out. -/
def topUpQ (K nh : ℕ) (H : HypothesisDFA (Option α)) (s : Level3Site) {b : ℕ} :
    List (Fin b → FreeMonoid α) → Finset (FreeMonoid α) → List (FreeMonoid α) →
      Query (FreeMonoid α) ARU (Option (TopUpEnd α × List (Fin b → FreeMonoid α)))
  | [], _, _ => .pure none
  | B :: Bs, X, ℓ => (blockQ K H s X (List.ofFn B)).bind fun ws =>
    let r := blockUpdate K X ℓ ws
    if nh ≤ r.1.length then .pure (some (.full r.1, Bs))
    else if r.1.length = ℓ.length ∧ r.2 ⊆ X then .pure (some (.retract r.1 X, Bs))
    else topUpQ K nh H s Bs (X ∪ r.2) r.1

/-- The round's list at each site. -/
def LazyLists (α : Type*) := Level3Site → List (FreeMonoid α)

inductive LazyEnd (α : Type*)
  | consistent
  | harvest (s : Level3Site) (ℓ : List (FreeMonoid α))
  | failed

structure LazyResult (α : Type*) where
  hypothesis : HypothesisDFA (Option α)
  expanded : Finset (FreeMonoid α)
  lists : LazyLists α
  ending : LazyEnd α

/-- Epoch `e` tests at `δ / ((e + 1)(e + 2))`. -/
def epochSettings (C : Level3Settings) (e : ℕ) : Level3Settings :=
  { C with δ := C.δ / ((e + 1) * (e + 2)) }

/-- An epoch's probes, and the blocks its top-up may draw. -/
abbrev EpochDraws (α : Type*) (P J b : ℕ) := (Fin P → FreeMonoid α) × (Fin J → Fin b → FreeMonoid α)

open scoped Classical in
/-- The epochs in turn, from epoch `e`, the trie `X` and the lists `ls`; failed where the epochs
or an epoch's blocks run out. -/
def lazyEpochsQ (C : Level3Settings) (K nh : ℕ) {J b : ℕ} :
    ℕ → List (EpochDraws α C.probeBudget J b) → Finset (FreeMonoid α) → LazyLists α →
      Query (FreeMonoid α) ARU (LazyResult α)
  | _, [], X, ls => .pure ⟨⟨.leaf, fun _ => none⟩, X, ls, .failed⟩
  | e, d :: ds, X, ls => (epochQ (epochSettings C e) K X d.1).bind fun res =>
    match res.ending with
    | .consistent => .pure ⟨res.hypothesis, X, ls, .consistent⟩
    | .failed => .pure ⟨res.hypothesis, X, ls, .failed⟩
    | .harvest s => (topUpQ K nh res.hypothesis s (List.ofFn d.2) X (ls s)).bind fun
      | none => .pure ⟨res.hypothesis, X, ls, .failed⟩
      | some (.full ℓ, _) => .pure ⟨res.hypothesis, X, Function.update ls s ℓ, .harvest s ℓ⟩
      | some (.retract ℓ X', _) => lazyEpochsQ C K nh (e + 1) ds X' (Function.update ls s ℓ)

/-- A round from the trie `X₀`, every list empty. -/
def lazyRoundQ (C : Level3Settings) (K nh : ℕ) {E J b : ℕ} (X₀ : Finset (FreeMonoid α))
    (draws : Fin E → EpochDraws α C.probeBudget J b) : Query (FreeMonoid α) ARU (LazyResult α) :=
  lazyEpochsQ C K nh 0 (List.ofFn draws) X₀ fun _ => []

/-- Every prefix shorter than `K` of the strings `P`. -/
def prefixTrie (K : ℕ) (P : Finset (FreeMonoid α)) : Finset (FreeMonoid α) :=
  P.biUnion fun p => (Finset.range (min K (p.toList.length + 1))).image fun i => prefixOf p i

/-- The tree's sift of every frontier string of `X`, padded. -/
def lazyTableQ (K : ℕ) (T : DTree (Option α)) (X : Finset (FreeMonoid α)) :
    Query (FreeMonoid α) ARU (FreeMonoid α → List Bool ⊕ FreeMonoid (Option α)) :=
  (Query.piList (fun u => T.siftQ (padIn K X u)) (trieFrontier X).toList).subst unpadQ

end Random

/-- The walk start's sift read off the table `tab`, then `H`'s edges along the rest of `x`. -/
def HypothesisDFA.LazyClassifies (H : HypothesisDFA (Option α))
    (tab : FreeMonoid α → List Bool ⊕ FreeMonoid (Option α)) (X : Finset (FreeMonoid α))
    (x : FreeMonoid α) (b : Prop) : Prop :=
  ∃ a, tab (prefixOf x (Random.walkStartIn X x)) = .inl a
    ∧ H.Classifies a (FreeMonoid.ofList ((x.toList.drop (Random.walkStartIn X x)).map some)) b

end OrthoDFA

end
