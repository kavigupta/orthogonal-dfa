import OrthoDFA.RoundStrong

/-!
# What a halving leaves to given-up edges

`RoundStrongNoStop`: no attempt in the round stops, so it holds no stopped strings.

`RoundStrongDeadEdge`: off a noise set of measure `δ`, the draws the round's hypothesis leaves at
given-up edges are at most those at edges the reads' likelier sides get wrong, plus, for edges they
get right, `ρ` times the reads of the walk, search and edge at strings read undecided at least `u`
of the time, the chance a string read undecided less than `u` of the time is decided on its less
likely side times every read, and the classes' usual slack.
-/

namespace OrthoDFA

open MeasureTheory

/-- `RoundStrongNoStop`: the round holds no string that stopped an attempt's guards. -/
def RoundStrongNoStop : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (C : StrongCfg α) (R : CutReads α)
    (seed : List (FreeMonoid α)) (Rmax : ℕ) (d : Fin Rmax → C.Draws),
    (strongRun C R seed Rmax d).2.1.stopped = []

section Reads

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
variable (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))

/-- The chance the cut leaves `w` undecided. -/
noncomputable def undecProb (w : FreeMonoid α) : ℝ :=
  μ.real {ω | (readsAt O B F ω).cut w = none}

/-- The chance the cut decides `w` on its less likely side. -/
noncomputable def minorProb (w : FreeMonoid α) : ℝ :=
  μ.real {ω | (readsAt O B F ω).cut w = some (!majSide O B F w)}

/-- `ρ`: the most any read is decided on its less likely side. -/
noncomputable def rhoMax : ℝ := ⨆ w, minorProb O B F w

/-- The most a read left undecided less than `u` of the time is decided on its less likely
side. -/
noncomputable def crossWell (u : ℝ) : ℝ :=
  ⨆ w, if undecProb O B F w < u then minorProb O B F w else 0

open scoped Classical in
/-- How many of the reads a draw's walk, search and edge can make, by prefix length and node, are
at strings read undecided at least `u` of the time. -/
noncomputable def badReads (u : ℝ) (t : DTree α) (k : ℕ) (x : FreeMonoid α) : ℕ :=
  ((Finset.Icc k (max k x.toList.length) ×ˢ t.midfixes).filter fun p =>
    u ≤ undecProb O B F (prefixOf x p.1 * p.2)).card

end Reads

variable {α : Type*}

/-- The edge from `s1` by `c` leads to `s2` for every string the sides `side` place at `s1`. -/
def EdgeCorrect [Fintype α] [DecidableEq α] (side : FreeMonoid α → Bool) (t : DTree α)
    (s1 : List Bool) (c : α) (s2 : List Bool) : Prop :=
  ∀ u, sidePath side t u = s1 → sidePath side t (u * FreeMonoid.of c) = s2

/-- The draw's walk ends at a learned edge the sides `side` get wrong. -/
def WrongEdgeAt [Fintype α] [DecidableEq α] (side : FreeMonoid α → Bool) (R : CutReads α)
    (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) : Prop :=
  ∃ s1 c s2 y, edgeAt R t edges k x = some (s1, c) ∧ edges s1 c = some (s2, y)
    ∧ ¬ EdgeCorrect side t s1 c s2

/-- `RoundStrongDeadEdge`: off a noise set of measure `δ`, the draws the round's hypothesis leaves
at given-up edges are at most those at edges the likelier sides get wrong, plus `ρ` times the reads
at strings read undecided at least `u` of the time, `crossWell u` times every read, and the
classes' slack at the realised readings' fluctuation. -/
def RoundStrongDeadEdge : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] (C : StrongCfg α) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (seed : List (FreeMonoid α)) (δ u : ℝ) (Rmax : ℕ) (d : Fin Rmax → C.Draws),
    0 < δ → δ ≤ 1 → C.k ≤ C.L → (∀ᵐ x ∂D, x.toList.length = C.L) →
    SuffixFree (F ∪ C.K.train F) →
    ∃ E : Set Ω, μ.real E ≤ δ ∧ ∀ ω ∉ E,
      let R := readsAt O B F ω
      let r := strongRun C R seed Rmax d
      let s := r.2.1.s
      let bnd : ℝ := (C.L + 1) * s.tree.midfixes.card
      D.real {x | DeadEdge R s.tree s.edges C.k (givenUp C r.2.1) x}
        ≤ rhoMax O B F * ∫ x, (badReads O B F u s.tree C.k x : ℝ) ∂D
          + crossWell O B F u * bnd
          + 2 * ((kPrefixes C.k (passReadSet C.k seed
              (strongProbes C R Rmax 0 (startAcc C R seed) [] d) s.tree)).card * prefixMax D C.k)
          + strongEps C D δ r.2.2.length * (2 + (rhoMax O B F + crossWell O B F u) * bnd)
          + D.real {x | DeadEdge R s.tree s.edges C.k (givenUp C r.2.1) x
              ∧ WrongEdgeAt (majSide O B F) R s.tree s.edges C.k x}

end OrthoDFA
