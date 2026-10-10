import OrthoDFA.TallyLoop

/-!
# One round of the tally loop

The family is the oracle: every string's read is accept, reject or undecided, independently
across strings, with a law that depends only on the string's state in the target DFA
(`FamilyReadTrichotomy`). `ReadModel` holds the target and those laws; `rd` is one draw of all
the reads. A read-state is good where it is undecided at most `1.5θ` of the time and reads on its
minority side at most `ε`. A state's true leaf is where its reads' majority sides lead, and it is
path-good where every read on the way is good.

A tree is genuine where it grows from the root's cut by splits that each separate two path-good
states at the split leaf, with good reads of the new midfix on different sides. The noise event
`TallyE` is a hypothesis on `rd`, quantified over genuine trees and edges into their leaves:
* `clean`: a path-good transition `τ = (q, c)` whose boundary (where the probe's search stops,
  read on majority sides, at `τ`) has mass at least `q₀` gets clean records, at `τ`'s true leaves,
  at least `c₀` times that mass;
* `spurious`: at each edge and target, probes whose record there is not true (its prefix or its
  successor not path-good, or off their true leaves) have mass at most `ρ`;
* `goodEdge`: at each edge, the undecided reads at good read-states charged to it average at most
  `θg` times the reads charged to it;
* `goodStart`: start-undecided outcomes at good read-states have mass at most `θgs`.

`TallyRound`: over `T` probes, the round reaches a state where no path-good transition's boundary
has mass `q₀`, or ends in success, or ends in a harvest whose bad read-states' undecided reads
exceed the threshold less the good ones' bound, but for chance at most that of not `TallyE`,
`T · P(Bin(n, c₀q₀) < m)`, `P(Bin(T, (|Q| + 2)² |Σ| ρ) ≥ m)`, and the harvest tests' levels over
the round.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The target, each of its states' read law, and what makes a read-state good. -/
structure ReadModel (α σ : Type*) where
  M : _root_.DFA α σ
  dist : σ → ARU → ℝ
  θ : ℝ
  ε : ℝ

namespace ReadModel

variable {σ : Type*} (G : ReadModel α σ)

/-- The side a read-state reads on more often. -/
def side (r : σ) : Bool := decide (G.dist r .reject < G.dist r .accept)

/-- How often a read-state reads on its other side. -/
def wrong (r : σ) : ℝ := if G.side r then G.dist r .reject else G.dist r .accept

def Good (r : σ) : Prop := G.dist r .undecided ≤ 3 / 2 * G.θ ∧ G.wrong r ≤ G.ε

/-- The state a string in state `q` reads `w` at. -/
def at' (q : σ) (w : FreeMonoid α) : σ := G.M.evalFrom q w.toList

/-- The leaf the state `q`'s reads lead to, each node read on its side. -/
def leafOf : DTree α → σ → List Bool
  | .leaf, _ => []
  | .node m r a, q => if G.side (G.at' q m) then true :: leafOf a q else false :: leafOf r q

/-- Every read on the way to `q`'s leaf is good. -/
def PathGood : DTree α → σ → Prop
  | .leaf, _ => True
  | .node m r a, q => G.Good (G.at' q m) ∧ if G.side (G.at' q m) then PathGood a q else PathGood r q

/-- The cut that reads every string on its state's side. -/
def trueCut (z : FreeMonoid α) : Option Bool := some (G.side (G.M.eval z.toList))

/-- Splitting the leaf `p` on `d` separates two path-good states there, with good reads of `d` on
different sides. -/
def GenuineSplit (T : DTree α) (p : List Bool) (d : FreeMonoid α) : Prop :=
  ∃ q₁ q₂, G.PathGood T q₁ ∧ G.PathGood T q₂ ∧ G.leafOf T q₁ = p ∧ G.leafOf T q₂ = p
    ∧ G.Good (G.at' q₁ d) ∧ G.Good (G.at' q₂ d) ∧ G.side (G.at' q₁ d) ≠ G.side (G.at' q₂ d)

/-- A tree grown from the root's cut by genuine splits. -/
inductive Genuine : DTree α → Prop
  | start : Genuine (.node 1 .leaf .leaf)
  | split {T : DTree α} {p : List Bool} {d : FreeMonoid α} :
      Genuine T → G.GenuineSplit T p d → Genuine (T.splitAt d p)

/-- A record of the prefix `sp` at the edge `(p, c)` with target `t` is true: `sp` and its
successor are path-good, at the leaves `p` and `t`. -/
def TrueRec (T : DTree α) (pct : List Bool × α × List Bool) (sp : FreeMonoid α) : Prop :=
  G.PathGood T (G.M.eval sp.toList) ∧ G.leafOf T (G.M.eval sp.toList) = pct.1
    ∧ G.PathGood T (G.M.eval (sp * FreeMonoid.of pct.2.1).toList)
    ∧ G.leafOf T (G.M.eval (sp * FreeMonoid.of pct.2.1).toList) = pct.2.2

/-- `τ = (q, c)`'s boundary: read on majority sides, the probe's search stops at the edge out of
`q`'s leaf by `c`, from a prefix in state `q`. -/
def tauBoundary (k : ℕ) (T : DTree α) (edges : Edges α) (q : σ) (c : α) :
    Set (FreeMonoid α) :=
  {x | ∃ sp, recordBy G.trueCut k (T, edges) x
      = some ((G.leafOf T q, c, G.leafOf T (G.M.step q c)), sp) ∧ G.M.eval sp.toList = q}

open scoped Classical in
/-- The undecided reads a probe charges to `e` at read-states satisfying `P`. -/
noncomputable def undecAt (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : Edges α)
    (e : List Bool × α) (P : σ → Prop) (x : FreeMonoid α) : ℕ :=
  edgeUndecBy (fun z => (rd z).cut) T edges k x e fun b => P (G.M.eval b.toList)

/-- The probes whose start is undecided, at a read-state satisfying `P`. -/
def startAt (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (P : σ → Prop) :
    Set (FreeMonoid α) :=
  {x | ∃ b, T.sift (fun z => (rd z).cut) (prefixOf x k) = .inr b ∧ P (G.M.eval b.toList)}

end ReadModel

/-- Every learned edge points at a leaf. -/
def EdgesInto (T : DTree α) (edges : Edges α) : Prop :=
  ∀ q c t w, edges q c = some (t, w) → t ∈ T.paths

/-- The noise event: over every genuine tree and edges into its leaves, path-good transitions
with a heavy boundary get clean records, records that are not true are rare at each edge and
target, and good read-states' undecided reads are rare at each edge and at the start. -/
structure TallyE {σ : Type*} (G : ReadModel α σ) (D : Measure (FreeMonoid α)) (k : ℕ)
    (q₀ c₀ ρ θg θgs : ℝ) (rd : FreeMonoid α → ARU) : Prop where
  clean : ∀ T edges, G.Genuine T → EdgesInto T edges → ∀ q c, G.PathGood T q →
    G.PathGood T (G.M.step q c) → q₀ ≤ D.real (G.tauBoundary k T edges q c) →
    c₀ * D.real (G.tauBoundary k T edges q c)
      ≤ D.real {x | (recordBy (fun z => (rd z).cut) k (T, edges) x).map Prod.fst
          = some (G.leafOf T q, c, G.leafOf T (G.M.step q c))}
  spurious : ∀ T edges, G.Genuine T → EdgesInto T edges → ∀ pct,
    D.real {x | ∃ sp, recordBy (fun z => (rd z).cut) k (T, edges) x = some (pct, sp)
      ∧ ¬ G.TrueRec T pct sp} ≤ ρ
  goodEdge : ∀ T edges, G.Genuine T → EdgesInto T edges → ∀ e,
    ∫ x, (G.undecAt rd k T edges e G.Good x : ℝ) ∂D
      ≤ θg * ∫ x, (edgeReadsBy (fun z => (rd z).cut) T edges k x e : ℝ) ∂D
  goodStart : ∀ T, G.Genuine T → D.real (G.startAt rd k T G.Good) ≤ θgs

/-- The run from `s` over the draws reaches a state satisfying `P`, or ends with an ending
satisfying `Q`. -/
def RunReaches {S X E : Type*} (step : S → X → S ⊕ (E × S)) (P : S → Prop)
    (Q : E → S → Prop) : S → List X → Prop
  | s, [] => P s
  | s, x :: xs => P s ∨ match step s x with
    | .inl s' => RunReaches step P Q s' xs
    | .inr (e, s') => Q e s'

/-- No path-good transition's boundary has mass `q₀`. -/
def AllFixed {σ : Type*} (G : ReadModel α σ) (D : Measure (FreeMonoid α)) (k : ℕ) (q₀ : ℝ)
    (s : TState α) : Prop :=
  ∀ q c, G.PathGood s.tree q → G.PathGood s.tree (G.M.step q c) →
    D.real (G.tauBoundary k s.tree s.edges q c) < q₀

/-- The endings the round claims: success, or a harvest whose bad read-states' undecided outcomes
there exceed the threshold less the good read-states' bound. -/
def GoodEnd {σ : Type*} (G : ReadModel α σ) (D : Measure (FreeMonoid α)) (C : TallyCfg)
    (rd : FreeMonoid α → ARU) (θg θgs : ℝ) : TEnd α → TState α → Prop
  | .success, _ => True
  | .harvestStart, s => C.θs - θgs < D.real (G.startAt rd C.k s.tree fun r => ¬ G.Good r)
  | .harvest e, s =>
    (C.θe - θg) * ∫ x, (edgeReadsBy (fun z => (rd z).cut) s.tree s.edges C.k x e : ℝ) ∂D
      < ∫ x, (G.undecAt rd C.k s.tree s.edges e (fun r => ¬ G.Good r) x : ℝ) ∂D
  | _, _ => False

/-- The most versions the round's hypothesis can take within `Lmax` leaves. -/
def versionCap (C : TallyCfg) (nα : ℕ) : ℕ := C.Lmax * (2 * C.Lmax * nα + 1)

/-- Bennett's bound on the chance that `j` probes' undecided reads at an edge, reading at most `R`
each, exceed `θe` of their reads by `C.exc j`, where they average at most `θe` of them. -/
noncomputable def edgeLevel (C : TallyCfg) (R j : ℕ) : ℝ :=
  let A := j * C.θe * R
  let s := C.exc j + A
  Real.exp (-(s * Real.log (s / A) - s + A) / ((1 + C.θe) * R))

/-- `TallyRound`: with draws of length at most `L`, the caps above the genuine trees' leaves and
the hypothesis's versions, `T` probes enough for every version to last `n`, and each edge test's
excess making Bennett's bound at most `a`, the round reaches a state where no path-good
transition's boundary has mass `q₀`, or ends in success or in a harvest its bad read-states
trigger, but for chance at most that of not `TallyE`, `T · P(Bin(n, c₀q₀) < m)`,
`P(Bin(T, (|Q| + 2)² |Σ| ρ) ≥ m)`, and `T² (Lmax |Σ| + 1) a` for the harvest tests. -/
def TallyRound : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (G : ReadModel α σ)
    (read : FreeMonoid α → Ω → ARU) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (C : TallyCfg) (q₀ c₀ ρ θg θgs : ℝ) (L n T : ℕ),
    (∀ᵐ x ∂D, x.toList.length ≤ L) →
    0 < c₀ * q₀ → c₀ * q₀ ≤ 1 → 0 ≤ ρ → (Fintype.card σ + 2) ^ 2 * Fintype.card α * ρ ≤ 1 →
    0 ≤ C.θs → C.θs ≤ 1 → 0 < C.θe → 0 < C.m → 1 ≤ C.n₀ → Fintype.card σ + 2 ≤ C.Lmax →
    versionCap C (Fintype.card α) ≤ C.fuel → (versionCap C (Fintype.card α) + 1) * n ≤ T →
    (∀ j, 1 ≤ j → j ≤ T → 0 ≤ C.exc j ∧ edgeLevel C ((L + 1) * C.Lmax) j ≤ C.a) →
    ∫⁻ ω, (Measure.pi fun _ : Fin T => D)
        {xs | ¬ RunReaches (tallyStep C fun z => (read z ω).cut) (AllFixed G D C.k q₀)
          (GoodEnd G D C (read · ω) θg θgs) tallyStart (List.ofFn xs)} ∂μ
      ≤ μ {ω | ¬ TallyE G D C.k q₀ c₀ ρ θg θgs (read · ω)}
        + T * ENNReal.ofReal (1 - binomSfGe n (c₀ * q₀) C.m)
        + ENNReal.ofReal
            (binomSfGe T ((Fintype.card σ + 2) ^ 2 * Fintype.card α * ρ) C.m)
        + T * T * (C.Lmax * Fintype.card α + 1) * ENNReal.ofReal C.a

end OrthoDFA

end
