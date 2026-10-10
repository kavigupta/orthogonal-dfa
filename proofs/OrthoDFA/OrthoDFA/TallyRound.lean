import OrthoDFA.TallyLoop
import OrthoDFA.RoundEnd

/-!
# One round of the tally loop, as sub-rounds

The family is the oracle: every string's read is accept, reject or undecided, independently
across strings, with a law that depends only on the string's state in the target DFA
(`FamilyReadTrichotomy`). `ReadModel` holds the target and those laws; `rd` is one draw of all
the reads, each law reading its rarer decided side at most `κ` times as often as it is undecided
or at most `ε`.
A read-state is good where it is undecided at most `1.5θ` of the time. A state's true leaf is
where its reads' majority sides lead.

A tree in the class grows from the root's cut by splits each at a leaf, on a letter followed by
the midfix where two leaves part: genuine ones, separating two states whose true leaf is the split
leaf, their reads of the new midfix on different sides, and at most `S` that are not. The noise
event `TallyE` is a hypothesis on `rd`, quantified over the class and edges into its leaves:
* `spurious`: at each edge and target, probes whose record there is not true (its prefix or its
  successor off its true leaf) have mass at most `ρ`;
* `goodEdge`, `goodStart`, `goodPT`: good read-states' undecided reads average at most `θg` of an
  edge's reads, and `θgs` a probe at the start; at most `θgpt` of the searches stop at an
  undecided middle read at a good read-state.

A sub-round starts at a tree of the class with no records and a fresh stretch, and lasts until the
tree changes or the round ends: in a split that is genuine (real) or not (fake), in success or a
harvest (good), or with the tree past `Lmax` leaves (bad). Each hypothesis ends the round or
changes within `nEnd` probes but for `termLevel`, unless it records at least `θr` of the time:
more than `θpt'` of searches stopping at undecided middles fire their test by the `hS + 1`-th
search, and otherwise the disagreement rate is below `εd'`, where `θr ≤ (1 - θpt') εd'`, and success
fires. Within a sub-round each edge is learned, and redirected, at most once, and every record
raises its edge and target towards the `m` at which it is fixed, so a sub-round holds at most
`Lmax² |Σ| m` records. `SubRound`: a sub-round is fake with chance at most `subFake`, `m` records
that are not true at one edge and target over its `subT` probes, and unfinished with chance at
most `subOpen`. `HarvestGood`: a harvest the round
ends in at a tree of the class has most of its strings at read-states that are not good, but for
chance `(2^(Lmax+1) |Σ| + T + T²) a`. `TallyRound`: over `T ≥ (S + |Q| + 1) · subT` probes the
round ends in success or such a harvest but for chance at most that of not `TallyE`, the fake
sub-rounds outnumbering `S` before the real ones (at most `|Q|`) run out,
`P(Bin(S + |Q| + 1, subFake) ≥ S + 1)`, `(S + |Q| + 1) · subOpen`, and the harvests' chance.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The target, each of its states' read law, and what makes a read-state good. Every law reads
its rarer decided side at most `κ` times as often as it is undecided, or at most `ε`. -/
structure ReadModel (α σ : Type*) where
  M : _root_.DFA α σ
  dist : σ → ARU → ℝ
  θ : ℝ
  κ : ℝ
  ε : ℝ
  rare : ∀ r, min (dist r .accept) (dist r .reject) ≤ max ε (κ * dist r .undecided)

namespace ReadModel

variable {σ : Type*} (G : ReadModel α σ)

/-- The side a read-state reads on more often. -/
def side (r : σ) : Bool := decide (G.dist r .reject < G.dist r .accept)

def Good (r : σ) : Prop := G.dist r .undecided ≤ 3 / 2 * G.θ

/-- The state a string in state `q` reads `w` at. -/
def at' (q : σ) (w : FreeMonoid α) : σ := G.M.evalFrom q w.toList

/-- The leaf the state `q`'s reads lead to, each node read on its side. -/
def leafOf : DTree α → σ → List Bool
  | .leaf, _ => []
  | .node m r a, q => if G.side (G.at' q m) then true :: leafOf a q else false :: leafOf r q

/-- Splitting the leaf `p` on `d` separates two states whose true leaf is `p`, their reads of `d`
on different sides. -/
def GenuineSplit (T : DTree α) (p : List Bool) (d : FreeMonoid α) : Prop :=
  ∃ q₁ q₂, G.leafOf T q₁ = p ∧ G.leafOf T q₂ = p ∧ G.side (G.at' q₁ d) ≠ G.side (G.at' q₂ d)

/-- A tree grown from the root's cut by `f` splits that are not genuine and any number that are,
each at a leaf, on a letter followed by the midfix where two leaves part. -/
inductive Grown : DTree α → ℕ → Prop
  | start : Grown (.node 1 .leaf .leaf) 0
  | real {T : DTree α} {f : ℕ} {p t t₀ : List Bool} {c : α} : Grown T f → t ∈ T.paths →
      t₀ ∈ T.paths → G.GenuineSplit T p (FreeMonoid.of c * T.midAt (lcp t t₀)) →
      Grown (T.splitAt (FreeMonoid.of c * T.midAt (lcp t t₀)) p) f
  | fake {T : DTree α} {f : ℕ} {p t t₀ : List Bool} {c : α} : Grown T f → p ∈ T.paths →
      t ∈ T.paths → t₀ ∈ T.paths → ¬ G.GenuineSplit T p (FreeMonoid.of c * T.midAt (lcp t t₀)) →
      Grown (T.splitAt (FreeMonoid.of c * T.midAt (lcp t t₀)) p) (f + 1)

/-- Grown with at most `S` splits that are not genuine. -/
def InClass (S : ℕ) (T : DTree α) : Prop := ∃ f ≤ S, G.Grown T f

/-- A record of the prefix `sp` at the edge `(p, c)` with target `t` is true: the true leaves of
`sp` and its successor are `p` and `t`. -/
def TrueRec (T : DTree α) (pct : List Bool × α × List Bool) (sp : FreeMonoid α) : Prop :=
  G.leafOf T (G.M.eval sp.toList) = pct.1
    ∧ G.leafOf T (G.M.eval (sp * FreeMonoid.of pct.2.1).toList) = pct.2.2

open scoped Classical in
/-- The undecided reads a probe charges to `e` at read-states satisfying `P`. -/
noncomputable def undecAt (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : Edges α)
    (e : List Bool × α) (P : σ → Prop) (x : FreeMonoid α) : ℕ :=
  ((edgeHarvBy (fun z => (rd z).cut) T edges k x e).filter fun b => P (G.M.eval b.toList)).length

open scoped Classical in
/-- How many of the strings were read at a read-state that is not good. -/
noncomputable def badCount (l : List (FreeMonoid α)) : ℕ :=
  (l.filter fun b => ¬ G.Good (G.M.eval b.toList)).length

/-- The probes whose start is undecided, at a read-state satisfying `P`. -/
def startAt (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (P : σ → Prop) :
    Set (FreeMonoid α) :=
  {x | ∃ b, T.sift (fun z => (rd z).cut) (prefixOf x k) = .inr b ∧ P (G.M.eval b.toList)}

/-- The probes that search: their walk and the cut disagree at a decided read. -/
def searchAt (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : Edges α) :
    Set (FreeMonoid α) :=
  {x | match probeBy (fun z => (rd z).cut) T edges k x with
    | .pair _ | .edge _ _ | .triple _ => True
    | _ => False}

/-- The probes whose search stops at an undecided middle, read at a read-state satisfying `P`. -/
def ptAt (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : Edges α) (P : σ → Prop) :
    Set (FreeMonoid α) :=
  {x | ∃ b, ptHarvBy (fun z => (rd z).cut) T edges k x = [b] ∧ P (G.M.eval b.toList)}

end ReadModel

/-- Every learned edge points at a leaf. -/
def EdgesInto (T : DTree α) (edges : Edges α) : Prop :=
  ∀ q c t w, edges q c = some (t, w) → t ∈ T.paths

/-- The noise event: over the class and edges into its leaves, records that are not true are rare
at each edge and target, and good read-states' undecided reads are rare at each edge, at the start
and at the middles searches stop at. -/
structure TallyE {σ : Type*} (G : ReadModel α σ) (D : Measure (FreeMonoid α)) (C : TallyCfg)
    (S : ℕ) (ρ θg θgs θgpt : ℝ) (rd : FreeMonoid α → ARU) : Prop where
  spurious : ∀ T edges, G.InClass S T → EdgesInto T edges → ∀ pct,
    D.real {x | ∃ sp, recordBy (fun z => (rd z).cut) C.k (T, edges) x = some (pct, sp)
      ∧ ¬ G.TrueRec T pct sp} ≤ ρ
  goodEdge : ∀ T edges, G.InClass S T → EdgesInto T edges → ∀ e,
    ∫ x, (G.undecAt rd C.k T edges e G.Good x : ℝ) ∂D
      ≤ θg * ∫ x, (edgeReadsBy (fun z => (rd z).cut) T edges C.k x e : ℝ) ∂D
  goodStart : ∀ T, G.InClass S T → D.real (G.startAt rd C.k T G.Good) ≤ θgs
  goodPT : ∀ T edges, G.InClass S T → EdgesInto T edges →
    D.real (G.ptAt rd C.k T edges G.Good) ≤ θgpt * D.real (ReadModel.searchAt rd C.k T edges)

/-- The run from `s` over the draws ends with an ending satisfying `Q`. -/
def RunEnds {S X E : Type*} (step : S → X → S ⊕ (E × S)) (Q : E → S → Prop) :
    S → List X → Prop
  | _, [] => False
  | s, x :: xs => match step s x with
    | .inl s' => RunEnds step Q s' xs
    | .inr (e, s') => Q e s'

/-- The run from `s` keeps its hypothesis over the draws. -/
def Keeps (C : TallyCfg) (cut : FreeMonoid α → Option Bool) : TState α → List (FreeMonoid α) →
    Prop
  | _, [] => True
  | s, x :: xs => ∃ s', tallyStep C cut s x = .inl s' ∧ s'.tree = s.tree ∧ s'.edges = s.edges
    ∧ Keeps C cut s' xs

/-- The endings the round claims: success, or a harvest most of whose strings are at read-states
that are not good. -/
def GoodEnd {σ : Type*} (G : ReadModel α σ) : TEnd α → TState α → Prop
  | .success, _ => True
  | .harvestStart, s => s.startH.length < 2 * G.badCount s.startH
  | .harvest e, s => (s.harv e.1 e.2).length < 2 * G.badCount (s.harv e.1 e.2)
  | .harvestPT, s => s.pt.length < 2 * G.badCount s.pt
  | .tooBig, _ => False

/-- The round's ending, as every level of the loop states it: success is consistent, a harvest
is its strings. -/
def tallyEnd : TEnd α → TState α → RoundEnd α
  | .success, _ => .consistent
  | .harvestStart, s => .harvest s.startH
  | .harvest e, s => .harvest (s.harv e.1 e.2)
  | .harvestPT, s => .harvest s.pt
  | .tooBig, _ => .failed

/-- The round ends well: consistent with the hypothesis disagreeing on at most `εd` of the probes,
or a harvest more than half of whose strings are at read-states that are not good. -/
def TallyEndsWell {σ : Type*} (G : ReadModel α σ) (D : Measure (FreeMonoid α)) (C : TallyCfg)
    (rd : FreeMonoid α → ARU) (e : TEnd α) (s : TState α) : Prop :=
  EndsWell G.M (fun q => ¬ G.Good q) D C.εd (ReadModel.searchAt rd C.k s.tree s.edges)
    (tallyEnd e s)

/-- How a sub-round ends. -/
inductive SubEnd
  | good
  | bad
  | real
  | fake
  | unfinished

open scoped Classical in
/-- One probe of a sub-round: it continues at the same tree, or ends the round, or the tree
changes by a split that is genuine (`true`) or not. -/
noncomputable def subSeg {σ : Type*} (G : ReadModel α σ) (C : TallyCfg)
    (cut : FreeMonoid α → Option Bool) (s : TState α) (x : FreeMonoid α) :
    TState α ⊕ ((TEnd α ⊕ Bool) × TState α) :=
  match tallyStep C cut s x with
  | .inr (e, s') => .inr (.inl e, s')
  | .inl s' => if s'.tree = s.tree then .inl s' else
      .inr (.inr (decide (∃ p d, s'.tree = s.tree.splitAt d p ∧ G.GenuineSplit s.tree p d)), s')

/-- The kind of a sub-round's last probe. -/
def subKind : (TEnd α ⊕ Bool) → SubEnd
  | .inl .tooBig => .bad
  | .inl _ => .good
  | .inr true => .real
  | .inr false => .fake

/-- How the sub-round from `s` ends within its first `j` probes, or is unfinished. -/
noncomputable def subEnd {σ : Type*} (G : ReadModel α σ) (C : TallyCfg)
    (cut : FreeMonoid α → Option Bool) : TState α → ℕ → (T : ℕ) → (Fin T → FreeMonoid α) → SubEnd
  | _, 0, _, _ => .unfinished
  | _, _ + 1, 0, _ => .unfinished
  | s, j + 1, T + 1, xs => match subSeg G C cut s (xs 0) with
    | .inl s' => subEnd G C cut s' j T (Fin.tail xs)
    | .inr (k, _) => subKind k

/-- Where a sub-round starts: a tree of the class, edges into its leaves, no records, a fresh
stretch. -/
def SubStart {σ : Type*} (G : ReadModel α σ) (S : ℕ) (s : TState α) : Prop :=
  G.InClass S s.tree ∧ EdgesInto s.tree s.edges ∧ (∀ q c, s.recs q c = []) ∧ s.n = 0 ∧ s.dis = 0

/-- From every start, over any `T ≥ Tsub` fresh draws, the sub-round's first `Tsub` probes end it
fake with chance at most `c` times its ending real or good plus `δ`, and bad or unfinished with
chance at most `δ'`. -/
def SubRoundBound {σ : Type*} (G : ReadModel α σ) (D : Measure (FreeMonoid α)) (C : TallyCfg)
    (cut : FreeMonoid α → Option Bool) (S : ℕ) (c δ δ' : ℝ) (Tsub : ℕ) : Prop :=
  ∀ s, SubStart G S s → ∀ T, Tsub ≤ T →
    (Measure.pi fun _ : Fin T => D) {xs | subEnd G C cut s Tsub T xs = .fake}
        ≤ ENNReal.ofReal c * (Measure.pi fun _ : Fin T => D)
          {xs | subEnd G C cut s Tsub T xs = .good ∨ subEnd G C cut s Tsub T xs = .real}
        + ENNReal.ofReal δ
      ∧ (Measure.pi fun _ : Fin T => D)
          {xs | subEnd G C cut s Tsub T xs = .bad ∨ subEnd G C cut s Tsub T xs = .unfinished}
        ≤ ENNReal.ofReal δ'

/-- A hypothesis recording less than `θr` of the time keeps over `nEnd` probes with chance at most
this: `θpt'` of its searches stopping at undecided middles but fewer than `hP` of the first
`hS + 1`, or the disagreement rate at most `εd'` with more than `hS` of `nEnd`, too many for
success. -/
noncomputable def termLevel (nEnd hP hS : ℕ) (θpt' εd' : ℝ) : ℝ :=
  (1 - binomSfGe (hS + 1) θpt' hP) + binomSfGe nEnd εd' (hS + 1)

/-- The most versions within a sub-round: each edge learned, and redirected, at most once. -/
def subVersions (C : TallyCfg) (nα : ℕ) : ℕ := 2 * C.Lmax * nα

/-- A sub-round's probes: `nEnd` for each version, and `nRec` at hypotheses that record. -/
def subT (C : TallyCfg) (nα nEnd nRec : ℕ) : ℕ := (subVersions C nα + 1) * nEnd + nRec

/-- A fake sub-round's chance: `m` records that are not true at one edge and target. -/
noncomputable def subFake (C : TallyCfg) (nα : ℕ) (ρ : ℝ) (Tsub : ℕ) : ℝ :=
  C.Lmax ^ 2 * nα * binomSfGe Tsub ρ C.m

/-- An unfinished sub-round's chance: some version keeping over `nEnd` probes, or `nRec` probes,
each recording with chance `θr`, making no more than the `Lmax² |Σ| m` records a sub-round holds. -/
noncomputable def subOpen (C : TallyCfg) (nα nRec : ℕ) (θr βt : ℝ) : ℝ :=
  (subVersions C nα + 1) * βt + (1 - binomSfGe nRec θr (C.Lmax ^ 2 * nα * C.m + 1))

/-- The chance that fake sub-rounds outnumber `k` before `g` real ones and an ending, each fake
with chance at most `(c + δ)/(1 + c)`, with `δ'` for each bad or unfinished one. -/
noncomputable def roundW (c δ δ' : ℝ) (k g : ℕ) : ℝ :=
  binomSfGe (k + g + 1) (min 1 ((c + δ) / (1 + c))) (k + 1) + (k + g + 1) * δ'

/-- `SubRound`: given `TallyE`, with the middles test's threshold `hP` at `hS + 1` searches, the
success test's `hS` at `nEnd` probes and `θr ≤ (1 - θpt') εd'`, sub-rounds are fake with chance at
most `subFake` and bad or unfinished within `subT` with chance at most `subOpen` of `termLevel`. -/
def SubRound : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (G : ReadModel α σ)
    (rd : FreeMonoid α → ARU) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (C : TallyCfg) (S nEnd nRec hP hS : ℕ) (ρ θg θgs θgpt θpt' θr εd' : ℝ),
    0 ≤ ρ → ρ ≤ 1 → 0 < C.m → Fintype.card σ + S + 3 ≤ C.Lmax → C.n₀ ≤ nEnd →
    C.n₀ ≤ hS + 1 → 0 ≤ C.θpt → C.θpt ≤ 1 → 0 ≤ C.εd → C.εd ≤ 1 → 0 ≤ θpt' → θpt' ≤ 1 →
    0 ≤ θr → θr ≤ 1 → 0 ≤ εd' → εd' ≤ 1 → θr ≤ (1 - θpt') * εd' →
    binomSfGe (hS + 1) C.θpt hP < C.a → 1 - binomSfGe nEnd C.εd (hS + 1) < C.a →
    C.a ≤ binomSfGe nEnd C.εd hS →
    TallyE G D C S ρ θg θgs θgpt rd →
    SubRoundBound G D C (fun z => (rd z).cut) S 0
      (subFake C (Fintype.card α) ρ (subT C (Fintype.card α) nEnd nRec))
      (subOpen C (Fintype.card α) nRec θr (termLevel nEnd hP hS θpt' εd'))
      (subT C (Fintype.card α) nEnd nRec)

/-- `HarvestGood`: with draws of length at most `L`, `θe ≥ 4θg`, every edge excess above
`4 L (1 + 4 θg Lmax) log(1/a)`, and the start's and the middles' tests firing only where good
read-states' undecided reads reach half their count with chance at most `a`, the round ends in a
harvest at a tree of the class most of whose strings are at good read-states with chance at most
`(2^(Lmax+1) |Σ| + T + T²) a`. -/
def HarvestGood : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (G : ReadModel α σ)
    (rd : FreeMonoid α → ARU) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (C : TallyCfg) (S L T : ℕ) (ρ θg θgs θgpt : ℝ),
    (∀ᵐ x ∂D, x.toList.length ≤ L) → 1 ≤ L → 0 < C.m → 0 < C.a → C.a ≤ 1 → 0 ≤ θg →
    4 * θg ≤ C.θe → Fintype.card σ + S + 2 ≤ C.Lmax →
    (∀ j, 4 * L * (1 + 4 * θg * C.Lmax) * Real.log (1 / C.a) < C.exc j) → 0 ≤ θgs → θgs ≤ 1 →
    (∀ t h, C.n₀ ≤ t → binomSfGe t C.θs h < C.a → binomSfGe t θgs ((h + 1) / 2) ≤ C.a) →
    0 ≤ θgpt → θgpt ≤ 1 →
    (∀ t h, C.n₀ ≤ t → binomSfGe t C.θpt h < C.a → binomSfGe t θgpt ((h + 1) / 2) ≤ C.a) →
    TallyE G D C S ρ θg θgs θgpt rd →
    (Measure.pi fun _ : Fin T => D)
        {xs | RunEnds (tallyStep C fun z => (rd z).cut)
          (fun e s' => G.InClass S s'.tree ∧ e ≠ .tooBig ∧ ¬ GoodEnd G e s') tallyStart
          (List.ofFn xs)}
      ≤ ENNReal.ofReal ((2 ^ (C.Lmax + 1) * Fintype.card α + T + T * T) * C.a)

/-- `SuccessSound`: a round ends in success at a hypothesis disagreeing on at least `εd` of the
probes with chance at most `T² a`. -/
def SuccessSound : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (rd : FreeMonoid α → ARU)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (C : TallyCfg) (T : ℕ),
    0 ≤ C.εd → C.εd ≤ 1 → C.a ≤ 1 →
    (Measure.pi fun _ : Fin T => D)
        {xs | RunEnds (tallyStep C fun z => (rd z).cut)
          (fun e s' => e = .success
            ∧ C.εd ≤ D.real (ReadModel.searchAt rd C.k s'.tree s'.edges)) tallyStart
          (List.ofFn xs)}
      ≤ ENNReal.ofReal (T * T * C.a)

/-- `TallyRound`: under `SubRound`'s and `HarvestGood`'s conditions, over
`T ≥ (S + |Q| + 1) · subT` probes the round ends in success at a hypothesis disagreeing on fewer
than `εd` of the probes, or in a harvest most of whose strings are at read-states that are not
good, but for chance at most that of not `TallyE`, `roundW 0 subFake subOpen S |Q|`,
`HarvestGood`'s and `SuccessSound`'s. -/
def TallyRound : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (G : ReadModel α σ)
    (read : FreeMonoid α → Ω → ARU) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (C : TallyCfg) (S nEnd nRec hP hS L T : ℕ) (ρ θg θgs θgpt θpt' θr εd' : ℝ),
    (∀ᵐ x ∂D, x.toList.length ≤ L) →
    0 ≤ ρ → ρ ≤ 1 → 0 < C.m → Fintype.card σ + S + 3 ≤ C.Lmax → C.n₀ ≤ nEnd →
    C.n₀ ≤ hS + 1 → 0 ≤ C.θpt → C.θpt ≤ 1 → 0 ≤ C.εd → C.εd ≤ 1 → 0 ≤ θpt' → θpt' ≤ 1 →
    0 ≤ θr → θr ≤ 1 → 0 ≤ εd' → εd' ≤ 1 → θr ≤ (1 - θpt') * εd' →
    binomSfGe (hS + 1) C.θpt hP < C.a → 1 - binomSfGe nEnd C.εd (hS + 1) < C.a →
    C.a ≤ binomSfGe nEnd C.εd hS →
    1 ≤ L → 0 < C.a → C.a ≤ 1 → 0 ≤ θg → 4 * θg ≤ C.θe →
    (∀ j, 4 * L * (1 + 4 * θg * C.Lmax) * Real.log (1 / C.a) < C.exc j) → 0 ≤ θgs → θgs ≤ 1 →
    (∀ t h, C.n₀ ≤ t → binomSfGe t C.θs h < C.a → binomSfGe t θgs ((h + 1) / 2) ≤ C.a) →
    0 ≤ θgpt → θgpt ≤ 1 →
    (∀ t h, C.n₀ ≤ t → binomSfGe t C.θpt h < C.a → binomSfGe t θgpt ((h + 1) / 2) ≤ C.a) →
    (S + Fintype.card σ + 1) * subT C (Fintype.card α) nEnd nRec ≤ T →
    ∫⁻ ω, (Measure.pi fun _ : Fin T => D)
        {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut) (TallyEndsWell G D C (read · ω))
          tallyStart (List.ofFn xs)} ∂μ
      ≤ μ {ω | ¬ TallyE G D C S ρ θg θgs θgpt (read · ω)}
        + ENNReal.ofReal (roundW 0
          (subFake C (Fintype.card α) ρ (subT C (Fintype.card α) nEnd nRec))
          (subOpen C (Fintype.card α) nRec θr (termLevel nEnd hP hS θpt' εd')) S (Fintype.card σ))
        + ENNReal.ofReal ((2 ^ (C.Lmax + 1) * Fintype.card α + T + 2 * (T * T)) * C.a)

end OrthoDFA

end
