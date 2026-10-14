import OrthoDFA.Loop
import OrthoDFA.FamilyRead
import OrthoDFA.RoundEnd
import OrthoDFA.SeqTest

/-!
# Level 5 of indecisive L*: the counterexample search

The family is the oracle: a read of a string is accept, reject or undecided, and a node of the
tree reads `w·m` as one read. `read` holds every string's read; its cut is the read's side, or
`none` where undecided.

Probes are fresh draws, each walked from `walkStart` along the learned edges and searched for its
first disagreement by main's bisection (`probeBy`, through `bisectBoundary false`), which stops
at the first undecided midpoint. A probe at an unlearned edge learns it.

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

section Probe

variable (cut : FreeMonoid α → Option Bool)

/-- `kWalk` through `cut`. -/
def kWalkBy (t : DTree α) (edges : EdgeTable α) (k : ℕ) (x : FreeMonoid α) : KWalk α :=
  match t.siftCut cut (prefixOf x k) with
  | .inr _ => .anchor
  | .inl p =>
    match follow edges p (x.toList.drop k) with
    | .inl ps => .reached ps
    | .inr (s, c, i) => .edge s c (k + i)

/-- `walkTo` through `cut`. -/
def walkToBy (t : DTree α) (edges : EdgeTable α) (k : ℕ) (x : FreeMonoid α) (j : ℕ) :
    List (List Bool) :=
  match t.siftCut cut (prefixOf x k) with
  | .inr _ => []
  | .inl p => (follow edges p ((x.toList.drop k).take (j - k))).elim id fun _ => []

/-- `walkCheck` through `cut`. -/
def walkCheckBy (t : DTree α) (edges : EdgeTable α) (k : ℕ) (x : FreeMonoid α) :
    WalkOutcome α ⊕ (List (List Bool) × ℕ) :=
  match kWalkBy cut t edges k x with
  | .anchor => .inl (.startUndecided (prefixOf x k))
  | .edge s _ j =>
    match t.siftCut cut (prefixOf x (j + 1)) with
    | .inr _ => .inl (.endUndecided (prefixOf x (j + 1)))
    | .inl _ =>
      match t.siftCut cut (prefixOf x j) with
      | .inr _ => .inl (.endUndecided (prefixOf x j))
      | .inl p => if p = s then .inl (.member (prefixOf x j))
        else .inr (walkToBy cut t edges k x j, j)
  | .reached ps =>
    match t.siftCut cut x with
    | .inr _ => .inl (.endUndecided x)
    | .inl a => if some a = ps.getLast? then .inl .agree else .inr (ps, x.toList.length)

/-- `agreesAt` through `cut`. -/
def agreesAtBy (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool) (i : ℕ) :
    Option Bool :=
  (t.siftCut cut (prefixOf x i)).elim (fun p => some (decide (p = walkAt i))) fun _ => none

/-- What one probe comes to, read through `cut`; a probe shorter than `k` agrees and reads
nothing. -/
def probeBy (t : DTree α) (edges : EdgeTable α) (k : ℕ) (x : FreeMonoid α) : WalkOutcome α :=
  if x.toList.length < k then .agree
  else (walkCheckBy cut t edges k x).elim id fun d =>
    bisectOutcome d.1
      (bisectBoundary false (agreesAtBy cut t x fun j => d.1.getD (j - k) []) k d.2).1

/-- The record a probe of `x` makes against the hypothesis `h`: at a clean disagreement, the edge
it crosses, the leaf its next prefix sifts to, and its prefix at the edge. -/
def recordBy (k : ℕ) (h : DTree α × EdgeTable α) (x : FreeMonoid α) :
    Option ((List Bool × α × List Bool) × FreeMonoid α) :=
  match probeBy cut h.1 h.2 k x with
  | .edge ps fd =>
    match x.toList[fd - 1]?, h.1.siftCut cut (prefixOf x fd) with
    | some c, .inl t => some ((ps.getD (fd - 1 - k) [], c, t), prefixOf x (fd - 1))
    | _, _ => none
  | _ => none

/-- The walk's edge out of position `i`: its leaf there and the probe's next letter. -/
def edgeAtBy (t : DTree α) (edges : EdgeTable α) (k : ℕ) (x : FreeMonoid α) (i : ℕ) :
    Option (List Bool × α) :=
  match (walkToBy cut t edges k x i).getLast?, x.toList[i]? with
  | some p, some c => some (p, c)
  | _, _ => none

/-- The positions `bracketAt` sifts: each middle. -/
def bracketSifts (agrees : ℕ → Option Bool) : ℕ → ℕ → ℕ → List ℕ
  | 0, _, _ => []
  | fuel + 1, lo, hi =>
    if lo + 1 < hi then
      match agrees ((lo + hi) / 2) with
      | some true => (lo + hi) / 2 :: bracketSifts agrees fuel ((lo + hi) / 2) hi
      | some false => (lo + hi) / 2 :: bracketSifts agrees fuel lo ((lo + hi) / 2)
      | none => [(lo + hi) / 2]
    else []

/-- The positions a probe sifts past its start, as `probeBy` reads them. -/
def siftsBy (t : DTree α) (edges : EdgeTable α) (k : ℕ) (x : FreeMonoid α) : List ℕ :=
  let found := fun (ps : List (List Bool)) (hi : ℕ) =>
    hi :: bracketSifts (agreesAtBy cut t x fun j => ps.getD (j - k) []) (hi - k) k hi
  let all := match kWalkBy cut t edges k x with
    | .anchor => []
    | .edge s _ j =>
      match t.siftCut cut (prefixOf x (j + 1)) with
      | .inr _ => [j + 1]
      | .inl _ =>
        match t.siftCut cut (prefixOf x j) with
        | .inr _ => [j + 1, j]
        | .inl p => if p = s then [j + 1, j] else (j + 1) :: found (walkToBy cut t edges k x j) j
    | .reached ps =>
      match t.siftCut cut x with
      | .inl a => if some a = ps.getLast? then [x.toList.length] else found ps x.toList.length
      | .inr _ => [x.toList.length]
  (all.filter (k < ·)).dedup

/-- The edge a position's reads are charged to: the walk's edge out of it, else the walk's last
edge. -/
def posEdgeBy (t : DTree α) (edges : EdgeTable α) (k : ℕ) (x : FreeMonoid α) (i : ℕ) :
    Option (List Bool × α) :=
  (edgeAtBy cut t edges k x i).orElse fun _ => edgeAtBy cut t edges k x (i - 1)

/-- The string its search read undecided at the midpoint it stopped at, if any. -/
def middleHarvestBy (t : DTree α) (edges : EdgeTable α) (k : ℕ)
    (x : FreeMonoid α) : List (FreeMonoid α) :=
  match probeBy cut t edges k x with
  | .undecidedMid j => (t.siftCut cut (prefixOf x j)).elim (fun _ => []) fun b => [b]
  | _ => []

/-- The string it read undecided at its start, if any. -/
def startHarvestBy (t : DTree α) (k : ℕ) (x : FreeMonoid α) : List (FreeMonoid α) :=
  if x.toList.length < k then [] else (t.siftCut cut (prefixOf x k)).elim (fun _ => []) fun b => [b]

/-- `retarget` through `cut`. -/
def retargetBy (t' : DTree α) (p : List Bool) (edges : EdgeTable α) : EdgeTable α := fun q c =>
  match edges q c with
  | some (s, w) =>
    if s = p then (t'.siftCut cut (w * FreeMonoid.of c)).elim (fun s' => some (s', w)) fun _ => none
    else some (s, w)
  | none => none

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

/-- The walk from the leaf `p` at the start, to position `j`. -/
def walkFrom (edges : EdgeTable α) (k : ℕ) (x : FreeMonoid α) (p : List Bool) (j : ℕ) :
    List (List Bool) :=
  (follow edges p ((x.toList.drop k).take (j - k))).elim id fun _ => []

/-- The edge position `i` is charged to, the walk starting at the leaf `p`. -/
def posEdgeFrom (edges : EdgeTable α) (k : ℕ) (x : FreeMonoid α) (p : List Bool) (i : ℕ) :
    Option (List Bool × α) :=
  let at' := fun i => match (walkFrom edges k x p i).getLast?, x.toList[i]? with
    | some q, some c => some (q, c)
    | _, _ => none
  (at' i).orElse fun _ => at' (i - 1)

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

/-- `bracketAt`, each position's agreement read through `agreementQuery`. -/
def bracketQuery (agreementQuery : ℕ → ProbeQuery α (Option Bool)) (ps : List (List Bool)) :
    ℕ → ℕ → ℕ → ProbeQuery α (WalkOutcome α)
  | 0, _, hi => .done (.edge ps hi)
  | fuel + 1, lo, hi =>
    if lo + 1 < hi then
      (agreementQuery ((lo + hi) / 2)).bind fun am => match am with
      | some true => bracketQuery agreementQuery ps fuel ((lo + hi) / 2) hi
      | some false => bracketQuery agreementQuery ps fuel lo ((lo + hi) / 2)
      | none => .done (.undecidedMid ((lo + hi) / 2))
    else .done (.edge ps hi)

variable (T : DTree α) (edges : EdgeTable α) (k : ℕ) (x : FreeMonoid α)

/-- The label of position `i`'s reads, the start at the leaf `p₀`. -/
def readLabel (p₀ : List Bool) (i : ℕ) : ReadLabel α := (i, posEdgeFrom edges k x p₀ i)

/-- The agreement of position `i`'s sift with the walk. -/
def agreementQuery (p₀ : List Bool) (walkAt : ℕ → List Bool) (i : ℕ) : ProbeQuery α (Option Bool) :=
  (siftQuery (readLabel edges k x p₀ i) T (prefixOf x i)).map fun s =>
    s.elim (fun q => some (decide (q = walkAt i))) fun _ => none

/-- One probe of `x` from `k`, as a query tree. -/
def probeQuery : ProbeQuery α (WalkOutcome α) :=
  if x.toList.length < k then .done .agree else
  (siftQuery (k, none) T (prefixOf x k)).bind fun a => match a with
  | .inr _ => .done (.startUndecided (prefixOf x k))
  | .inl p₀ => match follow edges p₀ (x.toList.drop k) with
    | .inl ps =>
      (siftQuery (readLabel edges k x p₀ x.toList.length) T
          (prefixOf x x.toList.length)).bind fun r =>
        match r with
        | .inr _ => .done (.endUndecided x)
        | .inl a' => if some a' = ps.getLast? then .done .agree
            else bracketQuery (agreementQuery T edges k x p₀ fun j => ps.getD (j - k) []) ps
              (x.toList.length - k) k x.toList.length
    | .inr (s, _, i) =>
      (siftQuery (readLabel edges k x p₀ (k + i + 1)) T (prefixOf x (k + i + 1))).bind fun r1 =>
        match r1 with
        | .inr _ => .done (.endUndecided (prefixOf x (k + i + 1)))
        | .inl _ => (siftQuery (readLabel edges k x p₀ (k + i)) T (prefixOf x (k
            + i))).bind fun r2 =>
          match r2 with
          | .inr _ => .done (.endUndecided (prefixOf x (k + i)))
          | .inl p => if p = s then .done (.member (prefixOf x (k + i)))
              else bracketQuery
                (agreementQuery T edges k x p₀ fun j =>
                  (walkFrom edges k x p₀ (k + i)).getD (j - k) [])
                (walkFrom edges k x p₀ (k + i)) (k + i - k) k (k + i)

end ProbeQ

/-- `cut` as reads. -/
def ofCut (cut : FreeMonoid α → Option Bool) (z : FreeMonoid α) : ARU :=
  match cut z with
  | some true => .accept
  | some false => .reject
  | none => .undecided

section FirstReads

variable (cut : FreeMonoid α → Option Bool) (t : DTree α) (edges : EdgeTable α) (k : ℕ)
  (x : FreeMonoid α)

/-- A probe's first read of each string, in order, with its label. -/
def firstsBy : List (ReadLabel α × FreeMonoid α) := (probeQuery t edges k x).firsts (ofCut cut) ∅

/-- How many strings a probe first reads at positions charged to the edge `e`. -/
def edgeReadsBy (e : List Bool × α) : ℕ :=
  ((firstsBy cut t edges k x).filter fun q => q.1.2 = some e).length

/-- The strings among them read undecided. -/
def edgeHarvestBy (e : List Bool × α) : List (FreeMonoid α) :=
  ((firstsBy cut t edges k x).filter fun q => q.1.2 = some e ∧ cut q.2 = none).map Prod.snd

end FirstReads

/-- What the loop carries: the hypothesis, how many times it has changed, and, since it last
changed, the probes, the strings read undecided at the start, for each edge how many strings
probes first read at it and those of them read undecided, the searches, and the undecided middles
they stopped at. -/
structure Level5State (α : Type*) where
  tree : DTree α
  edges : EdgeTable α
  version : ℕ
  probes : ℕ
  startUndecided : List (FreeMonoid α)
  edgeReads : List Bool → α → ℕ
  edgeUndecided : List Bool → α → List (FreeMonoid α)
  searches : ℕ
  middles : List (FreeMonoid α)

/-- The hypothesis `(T, E)` as the `v`-th, with no counts. -/
def freshState (T : DTree α) (E : EdgeTable α) (v : ℕ) : Level5State α :=
  ⟨T, E, v, 0, [], fun _ _ => 0, fun _ _ => [], 0, []⟩

/-- `s` with the edge `(p, c)` set to `e`, as a new hypothesis. -/
def Level5State.setEdge (s : Level5State α) (p : List Bool) (c : α)
    (e : List Bool × FreeMonoid α) : Level5State α :=
  freshState s.tree (Function.update s.edges p (Function.update (s.edges p) c (some e)))
    (s.version + 1)

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
def initialState : Level5State α := freshState (.node 1 .leaf .leaf) (fun _ _ => none) 0

/-- Split the leaf `p` at a contradiction of its edge `(p, c)` by the target `t`, on `c` and the
midfix where `t` and the edge's target part (Kearns–Vazirani): the leaf's edges are cleared and
the witnesses of those into it re-sifted, as a new hypothesis. -/
def splitState (cut : FreeMonoid α → Option Bool) (s : Level5State α) (p : List Bool) (c : α)
    (t : List Bool) : Level5State α :=
  match s.edges p c with
  | some (t₀, _) =>
    let t' := s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p
    freshState t' (fun q e => if p <+: q then none else retargetBy cut t' p s.edges q e)
      (s.version + 1)
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
  match probeBy cut s.tree s.edges C.walkStart x with
  | .member u =>
    match x.toList[u.toList.length]?, s.tree.siftCut cut u with
    | some c, .inl p =>
      match s.tree.siftCut cut (u * FreeMonoid.of c), s.edges p c with
      | .inl t, none => s₁.setEdge p c (t, u)
      | _, _ => s₁
    | _, _ => s₁
  | .edge _ _ | .undecidedMid _ => { s₁ with searches := s₁.searches + 1 }
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
    Level5Site α → DTree α × EdgeTable α → FreeMonoid α → List (FreeMonoid α)
  | .start, v, x => startHarvestBy cut v.1 k x
  | .edge e, v, x => edgeHarvestBy cut v.1 v.2 k x e
  | .middles, v, x => middleHarvestBy cut v.1 v.2 k x

open MeasureTheory in
/-- The law of what the harvest's sampler finds: each string as often as a probe drawn from `D`
and walked under the hypothesis `H` finds it at the site. -/
noncomputable def harvestLaw (D : Measure (FreeMonoid α)) (cut : FreeMonoid α → Option Bool)
    (k : ℕ) (H : HypothesisDFA α) (site : Level5Site α) : Measure (FreeMonoid α) :=
  D.bind fun x => ((siteHarvest cut k site (H.tree, Function.curry H.edges) x).map
    Measure.dirac).sum

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
  ⟨⟨r.1.tree, Function.uncurry r.1.edges⟩, r.2⟩

end OrthoDFA
