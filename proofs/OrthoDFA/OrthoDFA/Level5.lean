import OrthoDFA.StartAtK
import OrthoDFA.Pass
import Mathlib.Analysis.SpecialFunctions.Log.Base
import OrthoDFA.Probe
import OrthoDFA.FamilyRead
import OrthoDFA.RoundEnd
import OrthoDFA.SeqTest

/-!
# Level 5 of indecisive L*: the counterexample search

The family is the oracle: a read of a string is accept, reject or undecided, and a node of the
tree reads `w·m` as one read. `read` holds every string's read; its cut is the read's side, or
`none` where undecided.

Probes are fresh draws, each main's probe (`probeOrAgree`) from `walkStart`: walked along the
learned edges and searched for its first disagreement without recovery, so stopping at the first
undecided midpoint. A probe at an unlearned edge learns it.

Every count is kept since the hypothesis last changed: the probes, the start's undecided reads'
strings, each edge's strings first read in a probe at positions charged to it and those of them
read undecided, the searches, and the undecided middles they stop at. Every position past the
start is charged to an edge: the walk's edge out of that position, or, where there is none, the
walk's last edge; every position a probe sifts reads at its edge. The rates are checked by the
anytime-valid sequential test (`seqRate`) at level `testLevel`, each against an alternative on
its far side. After every probe (`applyProbe`):
* the start's undecided reads' test of `startUndecidedRate` against `startAltRate` firing; an edge
  charged at least `edgeTraffic` of the probes, its undecided strings exceeding
  `edgeUndecidedRate` of its reads by `edgeExcess` of them; or searches on at least `searchShare`
  of the probes, past `firstLook` searches, their middles' test of `middleUndecidedRate` against
  `middleAltRate` firing, ends the round with a harvest there;
* the searches' test of `disagreementRate` against the lower `disagreementAltRate` firing ends it
  in success;
* otherwise a clean disagreement at a learned edge (a contradiction) splits the edge's leaf at
  once (Kearns–Vazirani), on the edge's letter and the midfix where the probe's target and the
  edge's part; the leaf's edges are cleared and the witnesses of those into it re-sifted.

A learned edge or a split is a new hypothesis, and every count starts afresh. A tree past
`maxLeaves` leaves ends the round failed, and so does running out of probes.
`indecisiveLStarLevel5` runs the probes and returns the hypothesis and the ending.

A harvest is its site. Its sampler (`harvestLaw`) draws a fresh probe, walks it under the final
hypothesis and finds the probe's undecided strings at the site, as the tests counted them.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- `cut` as reads. -/
def ofCut (cut : FreeMonoid α → Option Bool) (z : FreeMonoid α) : ARU :=
  match cut z with
  | some true => .accept
  | some false => .reject
  | none => .undecided

section Probe

variable (cut : FreeMonoid α → Option Bool)

/-- The record a probe of `x` against the hypothesis `h` makes at a wrong edge: the edge, the leaf
its next prefix sifts to, and its prefix at the edge. -/
def recordBy (k : ℕ) (h : DTree α × Edges α) (x : FreeMonoid α) :
    Option ((List Bool × α × List Bool) × FreeMonoid α) :=
  match probeOrAgree (ofCut cut) h.1 h.2 k x with
  | .wrongEdge s c u t => some ((s, c, t), u)
  | _ => none

/-- The string its search read undecided at the midpoint it stopped at, if any. -/
def middleHarvestBy (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α) :
    List (FreeMonoid α) :=
  match probeOrAgree (ofCut cut) T E k x with
  | .undecided zs false => zs
  | _ => []

/-- The string it read undecided at its start, if any. -/
def startHarvestBy (T : DTree α) (k : ℕ) (x : FreeMonoid α) : List (FreeMonoid α) :=
  if x.toList.length < k then []
  else (T.sift (ofCut cut) (prefixOf x k)).elim (fun _ => []) fun b => [b]

end Probe

/-! ## A probe's reads, in order

A query tree reads keys one at a time, each chosen by the values read before it. A probe is one,
each read labelled by the position it sifts and the edge that position is charged to. -/

/-- An adaptive computation: done with a value, or read a key under a label and go on by its
value. -/
inductive QTree (Λ ι V β : Type*) where
  | done (b : β)
  | ask (ℓ : Λ) (w : ι) (next : V → QTree Λ ι V β)

namespace QTree

variable {Λ ι V β γ : Type*}

/-- The run's value under the reads `rd`. -/
def run (rd : ι → V) : QTree Λ ι V β → β
  | done b => b
  | ask _ w next => run rd (next (rd w))

/-- The run's reads, in order, with their labels. -/
def asks (rd : ι → V) : QTree Λ ι V β → List (Λ × ι)
  | done _ => []
  | ask ℓ w next => (ℓ, w) :: asks rd (next (rd w))

/-- `t`, and then `f` of its value. -/
def bind : QTree Λ ι V β → (β → QTree Λ ι V γ) → QTree Λ ι V γ
  | done b, f => f b
  | ask ℓ w next, f => ask ℓ w fun v => bind (next v) f

/-- `t` with its value mapped. -/
def map (f : β → γ) (t : QTree Λ ι V β) : QTree Λ ι V γ := t.bind fun b => done (f b)

variable [DecidableEq ι]

/-- The first read of each key not in `P`, in order, with its label. -/
def firsts (rd : ι → V) : Finset ι → QTree Λ ι V β → List (Λ × ι)
  | _, done _ => []
  | P, ask ℓ w next => if w ∈ P then firsts rd P (next (rd w))
      else (ℓ, w) :: firsts rd (insert w P) (next (rd w))

end QTree

section ProbeQ

/-- A read's label: the position sifted, and the edge it is charged to. -/
abbrev ReadLabel (α : Type*) := ℕ × Option (List Bool × α)

/-- A probe's query trees. -/
abbrev ProbeQuery (α β : Type*) := QTree (ReadLabel α) (FreeMonoid α) ARU β

/-- The sift of `u`, each node's string read under the label `ℓ`. -/
def siftQuery (ℓ : ReadLabel α) : DTree α → FreeMonoid α → ProbeQuery α (List Bool ⊕ FreeMonoid α)
  | .leaf, _ => .done (.inl [])
  | .node m r a, u => .ask ℓ (u * m) fun v => match v.cut with
    | none => .done (.inr (u * m))
    | some true => (siftQuery ℓ a u).map fun s => s.map (true :: ·) id
    | some false => (siftQuery ℓ r u).map fun s => s.map (false :: ·) id

variable (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α)

/-- The edge position `i` is charged to, the walk starting at the leaf `p`: the walk's edge out of
`i`, else out of `i - 1`. -/
def posEdgeFrom (p : List Bool) (i : ℕ) : Option (List Bool × α) :=
  let at' := fun i => match (walk E p (x.toList.drop k))[i - k]?, x.toList[i]? with
    | some q, some c => some (q, c)
    | _, _ => none
  (at' i).orElse fun _ => at' (i - 1)

/-- The label of position `i`'s reads, the start at the leaf `p₀`. -/
def readLabel (p₀ : List Bool) (i : ℕ) : ReadLabel α := (i, posEdgeFrom E k x p₀ i)

/-- The search's outcome at an edge into `p`, whose prefix sifts to `t`. -/
def edgeOutcome (st : ℕ → List Bool) (p : ℕ) (t : List Bool) : Outcome α :=
  match x.toList[p - 1]? with
  | some c => .wrongEdge (st (p - 1)) c (prefixOf x (p - 1)) t
  | none => .undecided [] false

/-- `bisectBoundary false` by fuel, between `lo`, which agrees with the walk `st`, and `hi`, whose
prefix sifts to `t` off it, each position's sift read under its label. -/
def bracketQuery (lab : ℕ → ReadLabel α) (st : ℕ → List Bool) :
    ℕ → ℕ → ℕ → List Bool → ProbeQuery α (Outcome α)
  | 0, _, hi, t => .done (edgeOutcome x st hi t)
  | fuel + 1, lo, hi, t =>
    if lo + 1 < hi then
      (siftQuery (lab ((lo + hi) / 2)) T (prefixOf x ((lo + hi) / 2))).bind fun r => match r with
      | .inl q => if q = st ((lo + hi) / 2) then bracketQuery lab st fuel ((lo + hi) / 2) hi t
          else bracketQuery lab st fuel lo ((lo + hi) / 2) q
      | .inr z => .done (.undecided [z] false)
    else .done (edgeOutcome x st hi t)

/-- `tripartiteSearch false` from `k`, which agrees with the walk `st`, to `hi`; `un` is the
unlearned edge the walk stopped at, if any. -/
def searchQuery (lab : ℕ → ReadLabel α) (st : ℕ → List Bool) (un : Option (α × List Bool))
    (hi : ℕ) : ProbeQuery α (Outcome α) :=
  (siftQuery (lab hi) T (prefixOf x hi)).bind fun r => match r with
  | .inr z => .done (.undecided [z] true)
  | .inl a =>
    if a = st hi then
      .done (match un with
        | some (c, t) => .unlearnedEdge (st hi) c (prefixOf x hi) t
        | none => .agree)
    else bracketQuery T x lab st (hi - k) k hi a

/-- One probe of `x` from `k`, as a query tree. -/
def probeQuery : ProbeQuery α (Outcome α) :=
  if x.toList.length < k then .done .agree else
  (siftQuery (k, none) T (prefixOf x k)).bind fun a => match a with
  | .inr z => .done (.undecidedStart z)
  | .inl p₀ =>
    let ss := walk E p₀ (x.toList.drop k)
    let st := fun j => ss.getD (j - k) []
    let hi := k + ss.length - 1
    match x.toList[hi]? with
    | some c =>
      (siftQuery (readLabel E k x p₀ (hi + 1)) T (prefixOf x (hi + 1))).bind fun r => match r with
      | .inr z => .done (.undecided [z] true)
      | .inl t => searchQuery T k x (readLabel E k x p₀) st (some (c, t)) hi
    | none => searchQuery T k x (readLabel E k x p₀) st none hi

end ProbeQ

section FirstReads

variable (cut : FreeMonoid α → Option Bool) (T : DTree α) (E : Edges α) (k : ℕ)
  (x : FreeMonoid α)

/-- A probe's first read of each string, in order, with its label. -/
def firstsBy : List (ReadLabel α × FreeMonoid α) := (probeQuery T E k x).firsts (ofCut cut) ∅

/-- How many strings a probe first reads at positions charged to the edge `e`. -/
def edgeReadsBy (e : List Bool × α) : ℕ :=
  ((firstsBy cut T E k x).filter fun q => q.1.2 = some e).length

/-- The strings among them read undecided. -/
def edgeHarvestBy (e : List Bool × α) : List (FreeMonoid α) :=
  ((firstsBy cut T E k x).filter fun q => q.1.2 = some e ∧ cut q.2 = none).map Prod.snd

end FirstReads

/-- What the loop carries: the hypothesis, how many times it has changed, and, since it last
changed, the probes, the strings read undecided at the start, for each edge how many strings
probes first read at it and those of them read undecided, the searches, and the undecided middles
they stopped at. -/
structure Level5State (α : Type*) where
  tree : DTree α
  edges : Edges α
  version : ℕ
  probes : ℕ
  startUndecided : List (FreeMonoid α)
  edgeReads : List Bool → α → ℕ
  edgeUndecided : List Bool → α → List (FreeMonoid α)
  searches : ℕ
  middles : List (FreeMonoid α)

/-- The hypothesis `(T, E)` as the `v`-th, with no counts. -/
def freshState (T : DTree α) (E : Edges α) (v : ℕ) : Level5State α :=
  ⟨T, E, v, 0, [], fun _ _ => 0, fun _ _ => [], 0, []⟩

/-- `s` with the edge `(p, c)` set to `e`, as a new hypothesis. -/
def Level5State.setEdge (s : Level5State α) (p : List Bool) (c : α)
    (e : List Bool × FreeMonoid α) : Level5State α :=
  freshState s.tree (Function.update s.edges (p, c) (some e)) (s.version + 1)

/-- `s` with a probe counted and its reads charged to their places. -/
def Level5State.countProbe (cut : FreeMonoid α → Option Bool) (k : ℕ) (s : Level5State α)
    (x : FreeMonoid α) : Level5State α :=
  { s with
    probes := s.probes + 1
    startUndecided := s.startUndecided ++ startHarvestBy cut s.tree k x
    edgeReads := fun p c => s.edgeReads p c + edgeReadsBy cut s.tree s.edges k x (p, c)
    edgeUndecided := fun p c =>
      s.edgeUndecided p c ++ edgeHarvestBy cut s.tree s.edges k x (p, c)
    middles := s.middles ++ middleHarvestBy cut s.tree s.edges k x }

/-- The first state: the root reads at `ε`, no edge learned. -/
def initialState : Level5State α := freshState (.node 1 .leaf .leaf) (fun _ => none) 0

/-- Split the leaf `p` at a contradiction of its edge `(p, c)` by the target `t`, on `c` and the
midfix where `t` and the edge's target part (Kearns–Vazirani), by main's split: the leaf's edges are
cleared and the witnesses of those into it re-sifted, as a new hypothesis. -/
def splitState (cut : FreeMonoid α → Option Bool) (s : Level5State α) (p : List Bool) (c : α)
    (t : List Bool) : Level5State α :=
  match s.edges (p, c) with
  | some (t₀, _) =>
    let H := HypothesisDFA.split (ofCut cut) ⟨s.tree, s.edges⟩
      (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p
    freshState H.tree H.edges (s.version + 1)
  | none => s

/-- How the round ends: in success, a harvest at the start, at an edge or of the undecided
middles, or the tree past `maxLeaves` leaves. -/
inductive Level5Ending (α : Type*)
  | success
  | harvestStart
  | harvest (e : List Bool × α)
  | harvestPT
  | tooBig

/-- Level 5's settings: the walk's start, the most leaves, the tests' rates and their
alternatives, the tests' level, the searches a middles test waits for, the excess
`edgeExcess r` an edge's undecided strings need after `r` reads, the share of the probes an edge
test and the share of searches a middles test need, and the probes a round may draw. -/
structure Level5Settings where
  walkStart : ℕ
  maxLeaves : ℕ
  startUndecidedRate : ℝ
  edgeUndecidedRate : ℝ
  middleUndecidedRate : ℝ
  disagreementRate : ℝ
  startAltRate : ℝ
  middleAltRate : ℝ
  disagreementAltRate : ℝ
  testLevel : ℝ
  firstLook : ℕ
  edgeExcess : ℕ → ℝ
  edgeTraffic : ℝ
  searchShare : ℝ
  probeBudget : ℕ

/-- A probe's reads charged, its search counted, and an unlearned edge it stops at learned. -/
def recordProbe (C : Level5Settings) (cut : FreeMonoid α → Option Bool)
    (s : Level5State α) (x : FreeMonoid α) : Level5State α :=
  let s₁ := s.countProbe cut C.walkStart x
  match probeOrAgree (ofCut cut) s.tree s.edges C.walkStart x with
  | .unlearnedEdge p c u t => s₁.setEdge p c (t, u)
  | .wrongEdge _ _ _ _ | .undecided _ false => { s₁ with searches := s₁.searches + 1 }
  | _ => s₁

open scoped Classical in
/-- The tests, on the counts since the hypothesis last changed: the start's undecided reads,
tested above `startUndecidedRate`; at an edge charged at least `edgeTraffic` of the probes, its
undecided strings exceeding `edgeUndecidedRate` of its reads by `edgeExcess`; with searches on at
least `searchShare` of the probes and past `firstLook` searches, the searches stopping at an
undecided middle, tested above `middleUndecidedRate`; or the searches, tested below
`disagreementRate`. Each rate is tested by `seqRate` against its alternative. -/
noncomputable def runTests (C : Level5Settings) (s : Level5State α) : Option (Level5Ending α) :=
  if seqRate C.startUndecidedRate C.startAltRate C.testLevel s.probes s.startUndecided.length
      then some .harvestStart
  else if h : ∃ e : List Bool × α, e.1 ∈ s.tree.leaves
      ∧ C.edgeTraffic * s.probes ≤ s.edgeReads e.1 e.2
      ∧ C.edgeExcess (s.edgeReads e.1 e.2)
        ≤ ((s.edgeUndecided e.1 e.2).length : ℝ) - C.edgeUndecidedRate * s.edgeReads e.1 e.2 then
    some (.harvest h.choose)
  else if C.searchShare * s.probes ≤ s.searches ∧ C.firstLook ≤ s.searches
      ∧ seqRate C.middleUndecidedRate C.middleAltRate C.testLevel s.searches s.middles.length then
    some .harvestPT
  else if seqRate C.disagreementRate C.disagreementAltRate C.testLevel s.probes s.searches then
    some .success
  else none

/-- One probe: its reads are charged, its search counted and an unlearned edge it stops at
learned; the tests may end the round; otherwise a contradiction splits its edge's leaf, ending the
round past `maxLeaves` leaves. -/
noncomputable def applyProbe (C : Level5Settings) (cut : FreeMonoid α → Option Bool)
    (s : Level5State α) (x : FreeMonoid α) : Level5State α ⊕ (Level5Ending α × Level5State α) :=
  let s₁ := recordProbe C cut s x
  match runTests C s₁ with
  | some e => .inr (e, s₁)
  | none =>
    match recordBy cut C.walkStart (s.tree, s.edges) x with
    | some ((p, c, t), _) =>
      let s₂ := splitState cut s₁ p c t
      if C.maxLeaves < s₂.tree.leaves.length then .inr (.tooBig, s₂) else .inl s₂
    | none => .inl s₁

/-- Where a harvest's strings were read: the start, an edge, or the middles searches stop at. -/
inductive Level5Site (α : Type*)
  | start
  | edge (e : List Bool × α)
  | middles

/-- The strings a probe of `x` walked under the hypothesis `v` reads undecided at the site. -/
def siteHarvest (cut : FreeMonoid α → Option Bool) (k : ℕ) :
    Level5Site α → DTree α × Edges α → FreeMonoid α → List (FreeMonoid α)
  | .start, v, x => startHarvestBy cut v.1 k x
  | .edge e, v, x => edgeHarvestBy cut v.1 v.2 k x e
  | .middles, v, x => middleHarvestBy cut v.1 v.2 k x

open MeasureTheory in
/-- The law of what the harvest's sampler finds: each string as often as a probe drawn from `D`
and walked under the hypothesis `H` finds it at the site. -/
noncomputable def harvestLaw (D : Measure (FreeMonoid α)) (cut : FreeMonoid α → Option Bool)
    (k : ℕ) (H : HypothesisDFA α) (site : Level5Site α) : Measure (FreeMonoid α) :=
  D.bind fun x => ((siteHarvest cut k site (H.tree, H.edges) x).map Measure.dirac).sum

/-- The ending, as every level states it: success is consistent, a harvest is its site. -/
def toRoundEnd : Level5Ending α → RoundEnd (Level5Site α)
  | .success => .consistent
  | .harvestStart => .harvest .start
  | .harvest e => .harvest (.edge e)
  | .harvestPT => .harvest .middles
  | .tooBig => .failed

/-- The state the probes end in, and how; `failed` where they ran out. -/
noncomputable def runProbes (C : Level5Settings) (cut : FreeMonoid α → Option Bool) :
    Level5State α → List (FreeMonoid α) → Level5State α × RoundEnd (Level5Site α)
  | s, [] => (s, .failed)
  | s, x :: xs =>
    match applyProbe C cut s x with
    | .inl s' => runProbes C cut s' xs
    | .inr (e, s') => (s', toRoundEnd e)

structure Level5Result (α : Type*) where
  hypothesis : HypothesisDFA α
  ending : RoundEnd (Level5Site α)

noncomputable def indecisiveLStarLevel5 (settings : Level5Settings) (read : FreeMonoid α → ARU)
    {P : ℕ} (draws : Fin P → FreeMonoid α) : Level5Result α :=
  let r := runProbes settings (fun z => (read z).cut) initialState (List.ofFn draws)
  ⟨⟨r.1.tree, r.1.edges⟩, r.2⟩

end OrthoDFA
