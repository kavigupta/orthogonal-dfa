import OrthoDFA.StartAtK
import OrthoDFA.StartState
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

/-- The gate's test settles on the side `b`: above `acc` where `b` is true. -/
def GateSettles (acc a : ℝ) (b : Bool) : Prop :=
  gateSide R t edges bg acc a (gateStop R t edges bg acc a) = some b

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

/-- Its walk reaches an unlearned edge whose prefixes the cut cannot place. -/
def IsBlocked (x : FreeMonoid α) : Prop :=
  (∃ s c j, kWalk R t edges k x = .edge s c j) ∧ ∃ w, probeOutcome R t edges k x = .endUndecided w

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

/-- Its read ends at an edge given up. -/
def DeadEdge (gu : List Bool × α → Prop) (x : FreeMonoid α) : Prop :=
  ∃ e, edgeAt R t edges k x = some e ∧ gu e

/-- Its start or whole is undecided at the root. -/
def RootUndecided (x : FreeMonoid α) : Prop :=
  (∃ w, probeOutcome R t edges k x = .startUndecided w ∧ ¬ DeepUndecided R t w)
    ∨ (probeOutcome R t edges k x = .endUndecided x ∧ ¬ DeepUndecided R t x)

/-- The draws whose walk from `k` does not agree, off the root's and the given-up edges'. -/
def NAOff (gu : List Bool × α → Prop) (x : FreeMonoid α) : Prop :=
  probeOutcome R t edges k x ≠ .agree ∧ ¬ RootUndecided R t edges k x ∧ ¬ DeadEdge R t edges k gu x

end Classes

/-- `_fires`: `binomial_side_of_boundary` of `h` hits in `m` trials against `θ`, then its count
where it does not settle; at a rate of zero any hit fires, and at one none does. -/
noncomputable def testFires (θ a : ℝ) (m h : ℕ) : Prop :=
  if θ ≤ 0 then 0 < h
  else if 1 ≤ θ then False
  else match rateSide θ a 0 m h with
    | some s => s = true
    | none => θ * m < h

/-- A harvest class's test on the refusal sample: hits `H` among trials `Tr`, against `θ`. -/
structure ClassTest (α : Type*) where
  hits : FreeMonoid α → Prop
  trials : FreeMonoid α → Prop
  θ : ℝ

/-- The test fires, read after the first `n` draws. -/
noncomputable def ClassTest.fires {N : ℕ} (b : Fin N → FreeMonoid α) (a : ℝ) (T : ClassTest α)
    (n : ℕ) : Prop :=
  testFires T.θ a (hitsIn b T.trials n) (hitsIn b T.hits n)

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

/-- The harvest classes, with the rates incidental indecision gives them: the ends at
`(depth − 1)·f` of draws, triples and pairs at `c·f·depth·searchSteps` of searched draws, the
reads undecided at an unlearned edge at `2·c·f·depth` of draws, and members at `θM`. -/
noncomputable def harvestTests (t : DTree α) (edges : Edges α) (k L : ℕ) (f c θM : ℝ) :
    List (ClassTest α) :=
  [⟨StartDeep R t k, fun _ => True, (t.depth - 1 : ℕ) * f⟩,
   ⟨EndDeep R t, fun _ => True, (t.depth - 1 : ℕ) * f⟩,
   ⟨IsTriple R t edges k, Searched R t edges k, c * f * t.depth * searchSteps L k⟩,
   ⟨IsPair R t edges k, Searched R t edges k, c * f * t.depth * searchSteps L k⟩,
   ⟨IsBlocked R t edges k, fun _ => True, 2 * c * f * t.depth⟩,
   ⟨IsMember R t edges k, fun _ => True, θM⟩]

/-- The target's states some position from `k` to `L` of at least `minCov` of the draws visits:
the covered states. -/
def Covered (A : DFA (FreeMonoid α) Q) (D : Measure (FreeMonoid α)) (k L : ℕ) (minCov : ℝ) :
    Set Q :=
  {q | minCov ≤ D.real {x | ∃ i, k ≤ i ∧ i ≤ L ∧ A.state (prefixOf x i) = q}}

/-- The draws the target re-rooted at `q₀` judges as the target does, staying covered. -/
def CoverGood (A : DFA (FreeMonoid α) Q) (S : Set Q) (q₀ : Q) : Set (FreeMonoid α) :=
  {x | StaysIn A S q₀ x ∧ (A.step q₀ x ∈ A.accept ↔ A.state x ∈ A.accept)}

/-- The masking residue: draws the target re-rooted at `q₀` judges well, on which the DFA from
`h q₀` leaves `h` of the target's run while the walk from `k` agrees. -/
def maskMass (A : DFA (FreeMonoid α) Q) (D : Measure (FreeMonoid α)) (S : Set Q)
    (h : Q → List Bool) (t : DTree α) (edges : Edges α) (k : ℕ) (q₀ : Q) : ℝ :=
  D.real {x | x ∈ CoverGood A S q₀ ∧ probeOutcome R t edges k x = .agree
    ∧ ∃ i ≤ x.toList.length, tRun edges (h q₀) (prefixOf x i) ≠ h (A.step q₀ (prefixOf x i))}

open scoped Classical in
/-- How often the middle side of the root read misjudges the target. -/
noncomputable def labelErr (A : DFA (FreeMonoid α) Q) (D : Measure (FreeMonoid α)) : ℝ :=
  D.real {x | R.mid x ≠ decide (A.state x ∈ A.accept)}

/-- What a refusal leaves to the harvest classes and live edges. -/
noncomputable def need (A : DFA (FreeMonoid α) Q) (D : Measure (FreeMonoid α)) (S : Set Q)
    (h : Q → List Bool) (t : DTree α) (edges : Edges α) (k : ℕ) (q₀ : Q)
    (gu : List Bool × α → Prop) (acc η : ℝ) : ℝ :=
  1 - acc - η - labelErr R A D - maskMass R A D S h t edges k q₀
    - D.real {x | RootUndecided R t edges k x} - D.real {x | DeadEdge R t edges k gu x}

/-- The limit below which every class fires on its first hit in `nr` draws. -/
noncomputable def tauZero (t : DTree α) (k L nr : ℕ) (c a : ℝ) : ℝ :=
  (1 - a) / (nr * max ((t.depth - 1 : ℕ) : ℝ)
    (max (2 * c * t.depth) (c * t.depth * searchSteps L k)))

open scoped Classical in
/-- What a reading of the round against the hypothesis `s` claims: the gate passes and its
chosen start disagrees on at most `1 − acc` of draws; or it refuses and the refusal sample turns
up live edges to rerun; or some harvest class fires; or the limit halves, and then, below `τ₀`
with members firing on their first hit, the walk's non-agreeing mass is at most `ν` and, where the
gate settled below, every covering start leaves at most `ν`. -/
def TrichotomyHolds (A : DFA (FreeMonoid α) Q) (s : KState α) (D : Measure (FreeMonoid α))
    (k L : ℕ) (f c θM acc a η minCov ν : ℝ) (gu : List Bool × α → Prop) {ng nr : ℕ}
    (bg : Fin ng → FreeMonoid α) (br : Fin nr → FreeMonoid α) : Prop :=
  let t := s.tree
  let e := s.edges
  let tests := harvestTests R t e k L f c θM
  let Tr := refusalStop br a tests (LiveEdge R t e k gu)
  let S := Covered A D k L minCov
  (GateSettles R t e bg acc a true
      ∧ D.real {x | StartDis R e (gateStart R t e bg acc a) x} ≤ 1 - acc)
    ∨ (¬ GateSettles R t e bg acc a true
      ∧ ∃ i : Fin nr, (i : ℕ) < Tr ∧ LiveEdge R t e k gu (br i))
    ∨ (¬ GateSettles R t e bg acc a true ∧ ∃ T ∈ tests, T.fires br a Tr)
    ∨ (¬ GateSettles R t e bg acc a true
      ∧ (tauZero t k L nr c a ≤ f ∨ 1 - a ≤ θM * nr
        ∨ (D.real {x | NAOff R t e k gu x} ≤ ν
          ∧ (GateSettles R t e bg acc a false → ∀ (q₀ : Q) (h : Q → List Bool), q₀ ∈ S →
            h q₀ ∈ t.paths → 1 - η ≤ D.real (CoverGood A S q₀) →
            (∀ q ∈ S, leafAccepts (h q) = decide (q ∈ A.accept)) →
            need R A D S h t e k q₀ gu acc η ≤ ν))))

/-- The draws whose class's harvested read is at a well-read state: a state the family leaves
undecided less than `c·f` of the time. -/
def WellRead (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (cf : ℝ) (b : FreeMonoid α) : Prop :=
  stateIndecision A O B F (A.state b) < cf

/-- What the harvest classes claim of the hypothesis `s` that reads `R`: in each, the draws whose
harvested read is at a well-read state are at most the incidental rate over the class's trials,
but for the draws sharing their first `k` letters with a string the pass may have read and the
fluctuation `ε`, in units of the most tagged reads a draw makes. So a class with rate `r` has a
bad share of at least `1 − (incidental + slack)/r`. -/
def QualityHolds (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (s : KState α) (D : Measure (FreeMonoid α)) (k L : ℕ)
    (seed probes : List (FreeMonoid α)) (f c ε : ℝ) : Prop :=
  let t := s.tree
  let e := s.edges
  let slack := fun M : ℝ =>
    (kPrefixes k (passReadSet k seed probes t)).card * prefixMax D k + ε * (1 + c * f * M)
  let wr := WellRead A O B F (c * f)
  D.real {x | ∃ j b, probeOutcome R t e k x = .triple j ∧ tripleRead R t x j = some b ∧ wr b}
      ≤ c * f * t.depth * searchSteps L k * D.real {x | Searched R t e k x} + slack (t.depth * L)
    ∧ D.real {x | ∃ j b b', probeOutcome R t e k x = .pair j ∧ tripleRead R t x j = some b
        ∧ tripleRead R t x (j + 1) = some b' ∧ wr b ∧ wr b'}
      ≤ c * f * t.depth * searchSteps L k * D.real {x | Searched R t e k x} + slack (t.depth * L)
    ∧ D.real {x | ∃ b, t.sift R.cut (prefixOf x k) = .inr b ∧ StartDeep R t k x ∧ wr b}
      ≤ c * f * (t.depth - 1 : ℕ) + slack (t.depth - 1 : ℕ)
    ∧ D.real {x | ∃ b, t.sift R.cut x = .inr b ∧ EndDeep R t x ∧ wr b}
      ≤ c * f * (t.depth - 1 : ℕ) + slack (t.depth - 1 : ℕ)
    ∧ D.real {x | ∃ w b, IsBlocked R t e k x ∧ probeOutcome R t e k x = .endUndecided w
        ∧ t.sift R.cut w = .inr b ∧ wr b}
      ≤ 2 * c * f * t.depth + slack (2 * t.depth)

/-- `RoundTrichotomy`: for any probes, outside a set of the oracle's noise of measure at most
`5·exp(−2ε²/prefixMax D k)`, where the classes' quality may fail, the gate's batch and the refusal
sample break the trichotomy with chance at most the gate tests' failure chances at each look and
the chance the refusal sample misses mass `ν`. -/
def RoundTrichotomy : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {Q : Type*} (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α))
    (B : State) (F : Finset (FreeMonoid α)) (K : StageKnobs α) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (L ng nr : ℕ) (seed probes : List (FreeMonoid α))
    (f c θM acc a η minCov ν ε : ℝ) (gu : List Bool × α → Prop),
    0 ≤ acc → acc ≤ 1 → 0 ≤ f → 0 ≤ c → 0 ≤ a → ν ≤ 1 → 0 < ε →
    (∀ᵐ x ∂D, x.toList.length = L) → SuffixFree (F ∪ K.train F) →
    let k := (L + 1) / 2
    ∃ E : Set Ω, μ.real E ≤ 5 * Real.exp (-2 * ε ^ 2 / prefixMax D k) ∧ ∀ ω ∉ E,
      let R := readsAt O B F ω
      let s := runPassK K R k (initialK K R seed) probes
      QualityHolds R A O B F s D k L seed probes f c ε
        ∧ ((Measure.pi fun _ : Fin ng => D).prod (Measure.pi fun _ : Fin nr => D)).real
            {b | ¬ TrichotomyHolds R A s D k L f c θM acc a η minCov ν gu b.1 b.2}
          ≤ 2 * (Nat.log 2 (ng / 30) + 2) * a + (1 - ν) ^ nr

end Round

end OrthoDFA
