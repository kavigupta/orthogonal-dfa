import OrthoDFA.Round

/-!
# The walk's ends

The walk from `k` sifts two strings: a draw's first `k` letters and the whole draw. The FNR gate
reads their root reads on the uniform population and on the sampler's draws cut to length `k`.
For the reads below the root, the round adds, for each midfix `m` of its tree, the population of
those strings followed by `m`, whose read is the node-`m` read of the string.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The sampler's draws cut to their first `k` letters. -/
noncomputable def startPopulation (D : Measure (FreeMonoid α)) (k : ℕ) : Measure (FreeMonoid α) :=
  D.map (prefixOf · k)

/-- `StartRootCovered`: where the family leaves the start population undecided at most `b` of
the time, as the clustering's gate certifies of each population it reads, the cut leaves the
root read of a draw's first `k` letters undecided at most `b` of the time. -/
def StartRootCovered : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ) (b : ℝ),
    ∫ p, undecidedProb O B.lo (B.hi + 1) F p ∂(startPopulation D k) ≤ b →
    ∫ x, μ.real {ω | (readsAt O B F ω).cut (prefixOf x k * 1) = none} ∂D ≤ b

namespace DTree

open scoped Classical in
noncomputable def midfixes : DTree α → Finset (FreeMonoid α)
  | .leaf => ∅
  | .node m r a => insert m (r.midfixes ∪ a.midfixes)

open scoped Classical in
/-- The midfixes of the nodes below the root. -/
noncomputable def belowRoot : DTree α → Finset (FreeMonoid α)
  | .leaf => ∅
  | .node _ r a => r.midfixes ∪ a.midfixes

end DTree

/-- `x`'s sift is left undecided at a node below the root. -/
def DeepUndecided (R : CutReads α) (t : DTree α) (x : FreeMonoid α) : Prop :=
  (t.sift R.cut x).isRight ∧ 2 ≤ (t.route R.cut x).1.length

/-- The population of draws from `X` followed by `m`. -/
noncomputable def endPopulation (X : Measure (FreeMonoid α)) (m : FreeMonoid α) :
    Measure (FreeMonoid α) :=
  X.map (· * m)

/-- `EndsCovered`: for any reads and tree, the draws from `X` whose sift is left undecided below
the root are no more than the populations `X·m`, over the midfixes `m` below the root, that the
reads leave undecided. -/
def EndsCovered : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (R : CutReads α) (t : DTree α)
    (X : Measure (FreeMonoid α)) [IsFiniteMeasure X],
    X.real {x | DeepUndecided R t x}
      ≤ ∑ m ∈ t.belowRoot, (endPopulation X m).real {p | R.cut p = none}

/-- `BadShare`: a population the family leaves undecided `ā` of the time, over the oracle's noise,
has at least `(ā − f)/(umax − f)` of its mass at states read undecided more than `f` of the time,
`umax` bounding every state's indecision. -/
def BadShare : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {Q : Type*} (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α))
    (B : State) (F : Finset (FreeMonoid α)) (P : Measure (FreeMonoid α)) [IsProbabilityMeasure P]
    (f umax : ℝ),
    f ≤ umax → (∀ q, stateIndecision A O B F q ≤ umax) →
    ∫ p, undecidedProb O B.lo B.hi F p ∂P
      ≤ f + (umax - f) * P.real {p | f < stateIndecision A O B F (A.state p)}

end OrthoDFA
