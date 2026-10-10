import OrthoDFA.TallyLoop

/-!
# One round of the tally loop

The family is the oracle: every string's read is accept, reject or undecided, independently
across strings, with a law that depends only on the string's state in the target DFA
(`FamilyReadTrichotomy`). `ReadModel` holds the target and those laws; `rd` is one draw of all
the reads. A read-state is good where it is undecided at most `1.5θ` of the time and reads on its
minority side at most `ε`. A state's true leaf is where its reads' majority sides lead, and it is
path-good where every read on the way is good.

A tree is genuine where it grows from the root's cut by splits that each separate two states whose
true leaf is the split leaf, their reads of the new midfix on different sides. The noise event
`TallyE` is a hypothesis on `rd`, quantified over genuine trees and edges into their leaves:
* `clean`: a path-good transition `τ = (q, c)` whose boundary (where the probe, read on majority
  sides, stops at `τ`'s edge: its search, or its walk where the edge is unlearned) has mass at
  least `q₀` gets clean records at `τ`'s true leaves, or learns its edge, at least `c₀` times that
  mass;
* `spurious`: probes whose record is not true (its prefix or its successor off its true leaf)
  have mass at most `ρH`, and those of them whose two prefixes' reads are counted at places that
  are not heavy at most `ρ`. A place is the start, for the prefix at `k`, or the edge a later
  prefix's reads are charged to, where an undecided read there would be counted; the start is
  heavy where it is left undecided `θs + ηs` of the time, an edge where its undecided reads exceed
  `θe` of its reads by `η` a probe;
* `goodEdge`: at each edge, the undecided reads at good read-states charged to it average at most
  `θg` times the reads charged to it;
* `goodStart`: start-undecided outcomes at good read-states have mass at most `θgs`.

`TallyRound`: over `T` probes, the round reaches a state where no path-good transition's boundary
has mass `q₀`, or ends in success, or ends in a harvest whose bad read-states' undecided reads
exceed the threshold less the good ones' bound, but for chance at most that of not `TallyE`,
`T · P(Bin(n, c₀q₀) < m)`, the tails of the records that are not true, and the harvest tests'
levels over the round. A record that is not true reads at a light place, at rate `ρ` over all
`T` probes, or at a heavy one. A heavy place arms its hypothesis, whose stretch then harvests
within `J` probes but for `heavyLevel` or the start's binomial tail; so the heavy records are made
over at most `(versionCap + 1) J` probes, at rate `ρH`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The target, each of its states' read law, and what makes a read-state good. Every law is
accept at most `ε` of the time, or reject at most `ε`, or undecided at least a third with accept
or reject at most `ε₂` and the rarer of them at most `κ` times the undecided. -/
structure ReadModel (α σ : Type*) where
  M : _root_.DFA α σ
  dist : σ → ARU → ℝ
  θ : ℝ
  ε : ℝ
  ε₂ : ℝ
  κ : ℝ
  trich : ∀ r, dist r .accept ≤ ε ∨ dist r .reject ≤ ε
    ∨ ((dist r .accept ≤ ε₂ ∨ dist r .reject ≤ ε₂) ∧ 1 / 3 ≤ dist r .undecided
      ∧ min (dist r .accept) (dist r .reject) ≤ κ * dist r .undecided)

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

/-- Splitting the leaf `p` on `d` separates two states whose true leaf is `p`, their reads of `d`
on different sides. -/
def GenuineSplit (T : DTree α) (p : List Bool) (d : FreeMonoid α) : Prop :=
  ∃ q₁ q₂, G.leafOf T q₁ = p ∧ G.leafOf T q₂ = p ∧ G.side (G.at' q₁ d) ≠ G.side (G.at' q₂ d)

/-- A tree grown from the root's cut by genuine splits. -/
inductive Genuine : DTree α → Prop
  | start : Genuine (.node 1 .leaf .leaf)
  | split {T : DTree α} {p : List Bool} {d : FreeMonoid α} :
      Genuine T → G.GenuineSplit T p d → Genuine (T.splitAt d p)

/-- A record of the prefix `sp` at the edge `(p, c)` with target `t` is true: the true leaves of
`sp` and its successor are `p` and `t`. -/
def TrueRec (T : DTree α) (pct : List Bool × α × List Bool) (sp : FreeMonoid α) : Prop :=
  G.leafOf T (G.M.eval sp.toList) = pct.1
    ∧ G.leafOf T (G.M.eval (sp * FreeMonoid.of pct.2.1).toList) = pct.2.2

/-- `τ = (q, c)`'s boundary: read on majority sides, the probe's search stops at the edge out of
`q`'s leaf by `c`, from a prefix in state `q`, or its walk stops there, the edge unlearned. -/
def tauBoundary (k : ℕ) (T : DTree α) (edges : Edges α) (q : σ) (c : α) :
    Set (FreeMonoid α) :=
  {x | (∃ sp, recordBy G.trueCut k (T, edges) x
      = some ((G.leafOf T q, c, G.leafOf T (G.M.step q c)), sp) ∧ G.M.eval sp.toList = q)
    ∨ ∃ u, probeBy G.trueCut T edges k x = .member u ∧ x.toList[u.toList.length]? = some c
      ∧ G.M.eval u.toList = q}

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

section Heavy

variable (C : TallyCfg) (D : Measure (FreeMonoid α)) (cut : FreeMonoid α → Option Bool) (η ηs : ℝ)
  (T : DTree α) (edges : Edges α)

/-- The edge `e`, out of a leaf, has undecided reads exceeding `θe` of its reads by `η` a probe on
average. -/
def HeavyEdge (e : List Bool × α) : Prop :=
  e.1 ∈ T.paths ∧ C.θe * ∫ x, (edgeReadsBy cut T edges C.k x e : ℝ) ∂D + η
    ≤ ∫ x, (edgeUndecBy cut T edges C.k x e (fun _ => True) : ℝ) ∂D

/-- The start is left undecided at least `θs + ηs` of the time. -/
def HeavyStart : Prop := C.θs + ηs ≤ D.real {x | ∃ b, T.sift cut (prefixOf x C.k) = .inr b}

/-- Some edge, or the start, is heavy. -/
def Armed : Prop := HeavyStart C D cut ηs T ∨ ∃ e, HeavyEdge C D cut η T edges e

/-- Where an undecided read at position `i` of `x` is counted is heavy: the start, at `k`, or the
edge it is charged to. -/
def HeavyAt (x : FreeMonoid α) (i : ℕ) : Prop :=
  if i = C.k then HeavyStart C D cut ηs T
  else ∃ e, posEdgeBy cut T edges C.k x i = some e ∧ HeavyEdge C D cut η T edges e

/-- A record of `x` at the prefix `sp` reads at a heavy place: `sp`'s or the next prefix's. -/
def HeavyRec (x sp : FreeMonoid α) : Prop :=
  HeavyAt C D cut η ηs T edges x sp.toList.length
    ∨ HeavyAt C D cut η ηs T edges x (sp.toList.length + 1)

end Heavy

/-- The noise event: over every genuine tree and edges into its leaves, path-good transitions
with a heavy boundary get clean records or learn their edge, records that are not true are rare,
and rarer still where their reads are at places that are not heavy, and good read-states'
undecided reads are rare at each edge and at the start. -/
structure TallyE {σ : Type*} (G : ReadModel α σ) (D : Measure (FreeMonoid α)) (C : TallyCfg)
    (q₀ c₀ ρ ρH θg θgs η ηs : ℝ) (rd : FreeMonoid α → ARU) : Prop where
  clean : ∀ T edges, G.Genuine T → EdgesInto T edges → ∀ q c, G.PathGood T q →
    G.PathGood T (G.M.step q c) → q₀ ≤ D.real (G.tauBoundary C.k T edges q c) →
    c₀ * D.real (G.tauBoundary C.k T edges q c)
      ≤ D.real {x | (recordBy (fun z => (rd z).cut) C.k (T, edges) x).map Prod.fst
          = some (G.leafOf T q, c, G.leafOf T (G.M.step q c))
        ∨ LearnsBy (fun z => (rd z).cut) C.k (T, edges) (G.leafOf T q) c x}
  spurious : ∀ T edges, G.Genuine T → EdgesInto T edges →
    D.real {x | ∃ pct sp, recordBy (fun z => (rd z).cut) C.k (T, edges) x = some (pct, sp)
      ∧ ¬ G.TrueRec T pct sp ∧ ¬ HeavyRec C D (fun z => (rd z).cut) η ηs T edges x sp} ≤ ρ
  spuriousAll : ∀ T edges, G.Genuine T → EdgesInto T edges →
    D.real {x | ∃ pct sp, recordBy (fun z => (rd z).cut) C.k (T, edges) x = some (pct, sp)
      ∧ ¬ G.TrueRec T pct sp} ≤ ρH
  goodEdge : ∀ T edges, G.Genuine T → EdgesInto T edges → ∀ e,
    ∫ x, (G.undecAt rd C.k T edges e G.Good x : ℝ) ∂D
      ≤ θg * ∫ x, (edgeReadsBy (fun z => (rd z).cut) T edges C.k x e : ℝ) ∂D
  goodStart : ∀ T, G.Genuine T → D.real (G.startAt rd C.k T G.Good) ≤ θgs

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

/-- Hoeffding's bound on the chance that `J` probes' undecided reads at an edge, reading at most
`R` each, exceed `θe` of their reads by less than `C.exc J`, where they exceed it by `η` a probe on
average. -/
noncomputable def heavyLevel (C : TallyCfg) (R J : ℕ) (η : ℝ) : ℝ :=
  Real.exp (-2 * (J * η - C.exc J) ^ 2 / (J * ((1 + C.θe) * R) ^ 2))

/-- `TallyRound`: with draws of length at most `L`, the caps above the genuine trees' leaves and
the hypothesis's versions, `T` probes enough for every version to last `n`, and each edge test's
excess making Bennett's bound at most `a`, the round reaches a state where no path-good
transition's boundary has mass `q₀`, or ends in success or in a harvest its bad read-states
trigger, but for chance at most that of not `TallyE`, `T · P(Bin(n, c₀q₀) < m)`, the records that
are not true reaching `m`, and `T² (Lmax |Σ| + 1) a` for the harvest tests. Those records reach
`m` only where `m₁` of them read at light places, `P(Bin(T, ρ) ≥ m₁)`, or `m₂` at heavy ones within
the first `J` probes of their stretch, `P(Bin((versionCap + 1) J, ρH) ≥ m₂)`, or some stretch at a
hypothesis with a heavy place outlasts `J` probes, at most `T` times `heavyLevel` and the start's
`P(Bin(J, θs + ηs) < hS)`, where `hS` start-undecided outcomes in `J` fire the start's test. -/
def TallyRound : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (G : ReadModel α σ)
    (read : FreeMonoid α → Ω → ARU) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (C : TallyCfg) (q₀ c₀ ρ ρH θg θgs η ηs : ℝ) (L n T m₁ m₂ J hS : ℕ),
    (∀ᵐ x ∂D, x.toList.length ≤ L) →
    0 < q₀ → 0 < c₀ * q₀ → c₀ * q₀ ≤ 1 → 0 ≤ ρ → ρ ≤ 1 → 0 ≤ ρH → ρH ≤ 1 →
    m₁ + m₂ ≤ C.m + 1 →
    0 ≤ C.θs → C.θs ≤ 1 → 0 < C.θe → 0 < C.m → 1 ≤ C.n₀ → Fintype.card σ + 2 ≤ C.Lmax →
    versionCap C (Fintype.card α) ≤ C.fuel → 1 ≤ n →
    (versionCap C (Fintype.card α) + 1) * n ≤ T →
    (∀ j, 1 ≤ j → j ≤ T → 0 ≤ C.exc j ∧ edgeLevel C ((L + 1) * C.Lmax) j ≤ C.a) →
    C.n₀ ≤ J → C.exc J ≤ J * η → 0 ≤ ηs → C.θs + ηs ≤ 1 → binomSfGe J C.θs hS < C.a →
    ∫⁻ ω, (Measure.pi fun _ : Fin T => D)
        {xs | ¬ RunReaches (tallyStep C fun z => (read z ω).cut) (AllFixed G D C.k q₀)
          (GoodEnd G D C (read · ω) θg θgs) tallyStart (List.ofFn xs)} ∂μ
      ≤ μ {ω | ¬ TallyE G D C q₀ c₀ ρ ρH θg θgs η ηs (read · ω)}
        + T * ENNReal.ofReal (1 - binomSfGe n (c₀ * q₀) C.m)
        + ENNReal.ofReal (binomSfGe T ρ m₁)
        + ENNReal.ofReal (binomSfGe ((versionCap C (Fintype.card α) + 1) * J) ρH m₂)
        + T * ENNReal.ofReal (heavyLevel C ((L + 1) * C.Lmax) J η
          + (1 - binomSfGe J (C.θs + ηs) hS))
        + T * T * (C.Lmax * Fintype.card α + 1) * ENNReal.ofReal C.a

end OrthoDFA

end
