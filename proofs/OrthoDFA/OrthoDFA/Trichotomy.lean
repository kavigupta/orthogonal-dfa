import OrthoDFA.StartAtK
import OrthoDFA.Pass
import Mathlib.Analysis.SpecialFunctions.Log.Base

/-!
# The round's trichotomy

The gate runs the hypothesis's DFA from every start on each draw and scores it against the draw's
root read; it passes where its test of the best start's agreement settles above `acc`, and a
gate whose test never settles refuses. On a refusal a fresh sample is read, every draw to one
outcome, until a harvest class's test against the rate incidental indecision alone would give it
fires or an edge still live turns up, else to its end: a class that fires is held, live edges rerun the pass, and
otherwise the limit halves.

`RoundTrichotomy`: outside a set of the oracle's noise of measure at most `β₁`, the gate's batch
and the refusal sample leave, but for chance `β₂`, the reading consistent, or holding a class
that fired, whose bad share is at least one less its incidental rate over its rate, or with live
edges left to rerun, or halving, which below `τ₀` needs every covering start to leave the classes
at most `ν`.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The totalised step: a learned edge's target, else stay. -/
def tStep (edges : Edges α) (p : List Bool) (c : α) : List Bool :=
  match edges p c with
  | some (q, _) => q
  | none => p

/-- The totalised DFA's leaf from `q` after `x`. -/
def tRun (edges : Edges α) (q : List Bool) (x : FreeMonoid α) : List Bool :=
  x.toList.foldl (tStep edges) q

/-- A leaf accepts: it lies on the root's accepting side. -/
def leafAccepts (p : List Bool) : Bool := p.head? == some true

variable (R : CutReads α)

/-- The DFA from `q` disagrees with the middle side of the draw's root read. -/
def StartDis (edges : Edges α) (q : List Bool) (x : FreeMonoid α) : Prop :=
  leafAccepts (tRun edges q x) ≠ R.mid x

/-- `_best`: the start with the most agreements, the first among ties. -/
def bestOf (qs : List (List Bool)) (score : List Bool → ℕ) : List Bool :=
  qs.foldl (fun b q => if score b < score q then q else b) (qs.headD [])

section Gate

variable (t : DTree α) (edges : Edges α) {N : ℕ} (bg : Fin N → FreeMonoid α)

/-- The best start after the first `n` draws. -/
noncomputable def gateBest (n : ℕ) : List Bool :=
  bestOf t.paths fun q => hitsIn bg (fun x => ¬ StartDis R edges q x) n

/-- The best start's test against `acc`, at failure chance `a` over the starts, after `n` draws. -/
noncomputable def gateSide (acc a : ℝ) (n : ℕ) : Option Bool :=
  rateSide acc (a / t.paths.length) 0 n
    (hitsIn bg (fun x => ¬ StartDis R edges (gateBest R t edges bg n) x) n)

/-- Where the gate stops: the first look at which its test settles, else the batch's end. -/
noncomputable def gateStop (acc a : ℝ) : ℕ :=
  (((lookSet 30 N).sort (· ≤ ·)).find? fun n => (gateSide R t edges bg acc a n).isSome).getD N

/-- The start the gate chooses. -/
noncomputable def gateStart (acc a : ℝ) : List Bool :=
  gateBest R t edges bg (gateStop R t edges bg acc a)

end Gate

section Classes

variable (t : DTree α) (edges : Edges α) (k : ℕ)

/-- The draw is searched: its read ends at a pair, an edge or a triple. -/
def Searched (x : FreeMonoid α) : Prop := (walkCheck R t edges k x).isRight

/-- Its read ends at a pair. -/
def IsPair (x : FreeMonoid α) : Prop := ∃ j, probeOutcome R t edges k x = .pair j

/-- Its read ends at a triple. -/
def IsTriple (x : FreeMonoid α) : Prop := ∃ j, probeOutcome R t edges k x = .triple j

/-- Its read ends at a member of a leaf whose edge is unlearned. -/
def IsMember (x : FreeMonoid α) : Prop := ∃ u, probeOutcome R t edges k x = .member u

/-- Its walk reaches an unlearned edge, and the cut cannot place a proper prefix either side of
it. -/
def IsBlocked (x : FreeMonoid α) : Prop :=
  (∃ s c j, kWalk R t edges k x = .edge s c j)
    ∧ ∃ w, probeOutcome R t edges k x = .endUndecided w ∧ w.toList.length < x.toList.length

/-- The start is undecided below the root. -/
def StartDeep (x : FreeMonoid α) : Prop := DeepUndecided R t (prefixOf x k)

/-- The whole draw is undecided below the root. -/
def EndDeep (x : FreeMonoid α) : Prop := DeepUndecided R t x

/-- The edge a searched draw's search ends at, as the leaf it leaves and the letter. -/
noncomputable def edgeAt (x : FreeMonoid α) : Option (List Bool × α) :=
  match probeOutcome R t edges k x with
  | .edge ps fd => (x.toList[fd - 1]?).map fun c => (ps.getD (fd - 1 - k) [], c)
  | _ => none

/-- Its read ends at an edge not given up. -/
def LiveEdge (gu : List Bool × α → Prop) (x : FreeMonoid α) : Prop :=
  ∃ e, edgeAt R t edges k x = some e ∧ ¬ gu e

end Classes

/-- `_fires`: `binomial_side_of_boundary` of `h` hits in `m` trials against `θ`, then, at the
`final` look only, its count where it does not settle; at a rate of zero any hit fires, and at one
none does. -/
noncomputable def testFires (θ a : ℝ) (final : Prop) (m h : ℕ) : Prop :=
  if θ ≤ 0 then 0 < h
  else if 1 ≤ θ then False
  else match rateSide θ a 0 m h with
    | some s => s = true
    | none => final ∧ θ * m < h

/-- A harvest class's test on the refusal sample: hits `H` among trials `Tr`, against `θ`. -/
structure ClassTest (α : Type*) where
  hits : FreeMonoid α → Prop
  trials : FreeMonoid α → Prop
  θ : ℝ

/-- The test fires, read after the first `n` draws. -/
noncomputable def ClassTest.fires {N : ℕ} (b : Fin N → FreeMonoid α) (a : ℝ) (T : ClassTest α)
    (n : ℕ) : Prop :=
  testFires T.θ a (n = N) (hitsIn b T.trials n) (hitsIn b T.hits n)

open scoped Classical in
/-- Where the refusal sample stops: the first look at which some test fires or a live edge has
turned up, else its end. -/
noncomputable def refusalStop {N : ℕ} (b : Fin N → FreeMonoid α) (a : ℝ)
    (tests : List (ClassTest α)) (live : FreeMonoid α → Prop) : ℕ :=
  (((lookSet 30 N).sort (· ≤ ·)).find? fun n =>
    decide ((∃ T ∈ tests, T.fires b a n) ∨ ∃ i : Fin N, (i : ℕ) < n ∧ live (b i))).getD N

section Round

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}

/-- The prefixes `log₂` bounds a search by. -/
noncomputable def searchSteps (L k : ℕ) : ℝ := Real.logb 2 (L - k : ℕ) + 1

/-- The held harvest classes, with the rates incidental indecision gives them: the ends at
`(depth − 1)·f` of draws, triples and pairs at `c·f·depth·searchSteps` of searched draws, the
reads undecided at an unlearned edge at `2·c·f·depth` of draws, and members at `θM`. -/
noncomputable def heldTests (t : DTree α) (edges : Edges α) (k L : ℕ) (f c θM : ℝ) :
    List (ClassTest α) :=
  [⟨StartDeep R t k, fun _ => True, (t.depth - 1 : ℕ) * f⟩,
   ⟨EndDeep R t, fun _ => True, (t.depth - 1 : ℕ) * f⟩,
   ⟨IsTriple R t edges k, Searched R t edges k, c * f * t.depth * searchSteps L k⟩,
   ⟨IsPair R t edges k, Searched R t edges k, c * f * t.depth * searchSteps L k⟩,
   ⟨IsBlocked R t edges k, fun _ => True, 2 * c * f * t.depth⟩,
   ⟨IsMember R t edges k, fun _ => True, θM⟩]

/-- `PAIR_TRIP`: pairs over half the searched draws. It holds nothing, and halves the limit. -/
noncomputable def pairTrip (t : DTree α) (edges : Edges α) (k : ℕ) : ClassTest α :=
  ⟨IsPair R t edges k, Searched R t edges k, 1 / 2⟩

/-- The harvest classes. -/
noncomputable def harvestTests (t : DTree α) (edges : Edges α) (k L : ℕ) (f c θM : ℝ) :
    List (ClassTest α) :=
  heldTests R t edges k L f c θM ++ [pairTrip R t edges k]

end Round

end OrthoDFA
