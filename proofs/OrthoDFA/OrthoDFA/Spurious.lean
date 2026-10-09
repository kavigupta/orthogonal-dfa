import OrthoDFA.RoundStrong

/-!
# Where a draw's search ends at an edge

`EdgeCause`: a draw whose search ends at an edge has a read, down the route a reference placement
gives one of its prefixes' states, that the cut decides on the other side; or the edge it ends at
leads off the leaf its next state is placed at.

`SpuriousDraw`: whatever chose the hypothesis, a draw drawn apart from the noise has such a read
with chance at most `spurRate`, the misread chances of at most `ψ₀` summed over its prefixes and
the midfixes up to length `n`, and the chance it lies in `BadRoute`, the draws a prefix of which the
hypothesis routes through a midfix longer than `n` or one misread more than `ψ₀`.

`RoundStrongSpurious`: so for each reading of the round, each draw of its refusal sample and each
fresh probe of its pass.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The midfixes down `z`'s route by `side`. -/
def routeMids (side : FreeMonoid α → Bool) (z : FreeMonoid α) : DTree α → List (FreeMonoid α)
  | .leaf => []
  | .node m r a => m :: if side (z * m) then routeMids side z a else routeMids side z r

/-- The strings of length at most `n`. -/
noncomputable def wordsUpTo (n : ℕ) : Finset (FreeMonoid α) :=
  (Finset.range (n + 1)).biUnion fun l => Finset.univ.image fun f : Fin l → α =>
    FreeMonoid.ofList (List.ofFn f)

section Defs

variable {Q : Type*}

/-- Some prefix of `x` from `k` on has a read down the route `side` gives its state's
representative that the cut decides on the other side. -/
def SpuriousAt (R : CutReads α) (A : DFA (FreeMonoid α) Q) (side : FreeMonoid α → Bool)
    (rep : Q → FreeMonoid α) (t : DTree α) (k : ℕ) (x : FreeMonoid α) : Prop :=
  ∃ i, k ≤ i ∧ i ≤ x.toList.length ∧ ∃ m ∈ routeMids side (rep (A.state (prefixOf x i))) t,
    R.cut (prefixOf x i * m) = some (!side (rep (A.state (prefixOf x i)) * m))

/-- The edge out of `y`'s leaf on `c` does not lead to `y·c`'s, leaves placed by `side`. -/
def WrongEdgeAt (A : DFA (FreeMonoid α) Q) (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α)
    (t : DTree α) (edges : Edges α) (y : FreeMonoid α) (c : α) : Prop :=
  ∃ s₂ w, edges (sidePath side t (rep (A.state y))) c = some (s₂, w)
    ∧ s₂ ≠ sidePath side t (rep (A.state (y * FreeMonoid.of c)))

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- `ψ(q, m)`: the chance the cut decides a read `s·m` of a string `s` reaching `q` off the side
`side` gives `rep q · m`. -/
noncomputable def wrongRate (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α) (q : Q)
    (m : FreeMonoid α) : ℝ :=
  sSup ((fun s => μ.real {ω | (readsAt O B F ω).cut (s * m) = some (!side (rep q * m))}) ''
    {s | A.state s = q})

/-- Over a draw's prefixes from `k` to `L`, the misread chances of at most `ψ₀` at the midfixes
up to length `n`. -/
noncomputable def spurRate (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α)
    (D : Measure (FreeMonoid α)) (k L n : ℕ) (ψ₀ : ℝ) : ℝ :=
  ∫ x, ∑ i ∈ Finset.Icc k L, ∑ m ∈ wordsUpTo n,
    (if wrongRate A O B F side rep (A.state (prefixOf x i)) m ≤ ψ₀
      then wrongRate A O B F side rep (A.state (prefixOf x i)) m else 0) ∂D

/-- The draws a prefix of which from `k` on `t` routes through a midfix longer than `n` or one its
state is misread at more than `ψ₀` of the time: a state near the band there, badly read, which is
what the harvest classes and the FNR gate go after across rounds. -/
def BadRoute (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α) (k n : ℕ)
    (ψ₀ : ℝ) (t : DTree α) : Set (FreeMonoid α) :=
  {x | ∃ i, k ≤ i ∧ i ≤ x.toList.length ∧ ∃ m ∈ routeMids side (rep (A.state (prefixOf x i))) t,
    n < m.toList.length ∨ ψ₀ < wrongRate A O B F side rep (A.state (prefixOf x i)) m}

/-- The residual: how much of `D` lies in `BadRoute`. -/
noncomputable def badRouteMass (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α))
    (B : State) (F : Finset (FreeMonoid α)) (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α)
    (D : Measure (FreeMonoid α)) (k n : ℕ) (ψ₀ : ℝ) (t : DTree α) : ℝ :=
  D.real (BadRoute A O B F side rep k n ψ₀ t)

end Defs

/-- `EdgeCause`: a draw whose search ends at the edge into `fd` has a read off its route, or the
edge out of its first `fd − 1` letters' leaf on its next letter is wrong. -/
def EdgeCause : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Q : Type*} (R : CutReads α)
    (A : DFA (FreeMonoid α) Q) (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α) (t : DTree α)
    (edges : Edges α) (k : ℕ) (x : FreeMonoid α) (ps : List (List Bool)) (fd : ℕ),
    probeOutcome R t edges k x = .edge ps fd →
      SpuriousAt R A side rep t k x
        ∨ ∃ c, x.toList[fd - 1]? = some c ∧ WrongEdgeAt A side rep t edges (prefixOf x (fd - 1)) c

/-- `SpuriousDraw`: a draw `x`, with the noise `ω` drawn as `μ × D` by `g`, has a read off its
route in the tree `T` picks, however `T` picks it, with chance at most `spurRate` and the chance it
lies in `BadRoute` of that tree. When `T` is chosen apart from `x`, the latter is the expected
`badRouteMass`. -/
def SpuriousDraw : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {Q : Type*} (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α))
    (B : State) (F : Finset (FreeMonoid α)) (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k L n : ℕ) (ψ₀ : ℝ)
    {Z : Type*} [MeasurableSpace Z] (P : Measure Z) (g : Z → Ω × FreeMonoid α) (T : Z → DTree α),
    (∀ᵐ x ∂D, x.toList.length = L) → Measurable g → P.map g = μ.prod D →
    P.real {z | SpuriousAt (readsAt O B F (g z).1) A side rep (T z) k (g z).2}
      ≤ spurRate A O B F side rep D k L n ψ₀
        + P.real {z | (g z).2 ∈ BadRoute A O B F side rep k n ψ₀ (T z)}

/-- What each reading's pass starts from: the round's accumulator and the probes it reruns. -/
noncomputable def strongEntries (C : StrongCfg α) (R : CutReads α) :
    (n : ℕ) → ℕ → RoundAcc α → List (FreeMonoid α) → (Fin n → C.Draws)
      → List (RoundAcc α × List (FreeMonoid α))
  | 0, _, _, _, _ => []
  | n + 1, j, A, first, d =>
    (A, first) :: match strongReading C R j A first (d 0) with
      | (_, .inl _) => []
      | (A', .inr lv) => strongEntries C R n (j + 1) A' lv (Fin.tail d)

/-- `RoundStrongSpurious`: at reading `j`, the refusal sample's `i`-th draw has a read off its
route in the reading's tree, and the pass's `i`-th fresh probe one in the tree before it, each with
chance at most `spurRate` and the chance it lies in that tree's `BadRoute`. -/
def RoundStrongSpurious : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {Q : Type*} (C : StrongCfg α) (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (seed : List (FreeMonoid α)) (n : ℕ) (ψ₀ : ℝ) (Rmax : ℕ)
    (j : Fin Rmax),
    (∀ᵐ x ∂D, x.toList.length = C.L) →
    let P := μ.prod (Measure.pi fun _ : Fin Rmax => C.drawMeasure D)
    let E := fun p : Ω × (Fin Rmax → C.Draws) =>
      (strongEntries C (readsAt O B F p.1) Rmax 0 (startAcc C (readsAt O B F p.1) seed) [] p.2)[j]?
    let bad := BadRoute A O B F side rep C.k n ψ₀
    (∀ i : Fin C.nr,
      P.real {p | ∃ e ∈ E p, SpuriousAt (readsAt O B F p.1) A side rep
          (strongPass C (readsAt O B F p.1) e.1 (e.2 ++ List.ofFn (p.2 j).1)).s.tree C.k
          ((p.2 j).2.2.1 i)}
        ≤ spurRate A O B F side rep D C.k C.L n ψ₀
          + P.real {p | ∃ e ∈ E p, (p.2 j).2.2.1 i ∈
              bad (strongPass C (readsAt O B F p.1) e.1 (e.2 ++ List.ofFn (p.2 j).1)).s.tree})
    ∧ ∀ i : Fin C.np,
      P.real {p | ∃ e ∈ E p, SpuriousAt (readsAt O B F p.1) A side rep
          (strongPass C (readsAt O B F p.1) e.1 (e.2 ++ (List.ofFn (p.2 j).1).take i)).s.tree
          C.k ((p.2 j).1 i)}
        ≤ spurRate A O B F side rep D C.k C.L n ψ₀
          + P.real {p | ∃ e ∈ E p, (p.2 j).1 i ∈
              bad (strongPass C (readsAt O B F p.1) e.1
                (e.2 ++ (List.ofFn (p.2 j).1).take i)).s.tree}

end OrthoDFA
