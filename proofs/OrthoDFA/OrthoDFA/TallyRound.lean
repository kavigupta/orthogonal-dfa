import OrthoDFA.TallyLoop
import OrthoDFA.RoundEnd

/-!
# One round of the tally loop, as sub-rounds

The family is the oracle: every string's read is accept, reject or undecided, independently
across strings, with a law that depends only on the string's state in the target DFA
(`FamilyReadBound`). `ReadModel` holds the target and those laws; `rd` is one draw of all
the reads, each law reading its rarer decided side at most `κ` times as often as it is undecided
or at most `ε`.
A read-state is good where it is undecided at most `1.5θ` of the time. A state's true leaf is
where its reads' majority sides lead.

A tree in the class grows from the root's cut by splits each at a leaf, on a letter followed by
the midfix where two leaves part: genuine ones, separating two states whose true leaf is the split
leaf, their reads of the new midfix on different sides, and at most `S` that are not. The noise
event `TallyE` is a hypothesis on `rd`, quantified over the class and edges into its leaves:
* `spurious`: probes whose record is not true (its prefix or its successor off its true leaf)
  have mass at most `κ` times a probe's undecided strings, at the start and charged to the edges
  out of the leaves (`twinsBy`), plus `ρ`;
* `goodEdge`, `goodStart`, `goodPT`: good read-states' undecided reads average at most `θg` of an
  edge's positions, up to a slack the edge test's traffic gate covers, and `θgs` a probe at the
  start; at most `θgpt` of the searches stop at an undecided middle read at a good read-state,
  up to a slack the middles test's search-rate gate covers.

A sub-round starts at a tree of the class with no records and a fresh stretch, and lasts until the
tree changes or the round ends. Each hypothesis ends the round or changes within `nEnd` probes but
for `termLevel`, unless it records at least `θr` of the time: more than `θpt'` of searches
stopping at undecided middles fire their test by the `hS + 1`-th search, and otherwise the
disagreement rate is below `εd'`, where `θr ≤ (1 - θpt') εd'`, and success fires. Within a
sub-round each edge is learned, and redirected, at most once, and every record raises its edge and
target towards the `m` at which it is fixed; a true record is at one of the `|Q| |Σ|` edges and
targets a state and a letter lead to. `SubRound`: a sub-round keeps its tree over its `subT`
probes with fewer than `B` records that are not true with chance at most `subOpen`.
`FakeRace`: the round's records that are not true reach `(S + 1) m` within `W` probes with chance
at most `fakeRun`, as the tests hold its undecided strings. `HarvestGood`: a harvest the round
ends in at a tree of the class has most of its strings at read-states that are not good, but for
chance `(2^(Lmax+1) |Σ| + T + T²) a`. `TallyRound`: over `T ≥ (S + |Q| + 1) · subT` probes the
round ends in success or such a harvest but for chance at most that of not `TallyE`, `fakeRun`
over its first `(S + |Q| + 1) · subT` probes (a split that is not genuine needs `m` records that are
not true, so below `(S + 1) m` its trees stay in the class and at most `S + |Q|` sub-rounds end in
a split), `(S + |Q| + 1) · subOpen`, and the harvests' and success's chances.
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

/-- The probes whose record is not true. -/
def untrueAt (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : Edges α) :
    Set (FreeMonoid α) :=
  {x | ∃ pct sp, recordBy (fun z => (rd z).cut) k (T, edges) x = some (pct, sp)
    ∧ ¬ G.TrueRec T pct sp}

end ReadModel

/-- The undecided strings a probe reads at the start and charges to the edges out of the leaves. -/
noncomputable def twinsBy (cut : FreeMonoid α → Option Bool) (T : DTree α) (edges : Edges α)
    (k : ℕ) (x : FreeMonoid α) : ℕ :=
  (startHarvBy cut T k x).length
    + ∑ e ∈ T.paths.toFinset ×ˢ (Finset.univ : Finset α), (edgeHarvBy cut T edges k x e).length

/-- Every learned edge points at a leaf. -/
def EdgesInto (T : DTree α) (edges : Edges α) : Prop :=
  ∀ q c t w, edges q c = some (t, w) → t ∈ T.paths

/-- The noise event: over the class and edges into its leaves, records that are not true are rare
at each edge and target, and good read-states' undecided reads are rare at each edge, at the start
and at the middles searches stop at. -/
structure TallyE {σ : Type*} (G : ReadModel α σ) (D : Measure (FreeMonoid α)) (C : TallyCfg)
    (S : ℕ) (ρ θg θgs θgpt : ℝ) (rd : FreeMonoid α → ARU) : Prop where
  spurious : ∀ T edges, G.InClass S T → EdgesInto T edges →
    D.real (G.untrueAt rd C.k T edges)
      ≤ G.κ * ∫ x, (twinsBy (fun z => (rd z).cut) T edges C.k x : ℝ) ∂D + ρ
  goodEdge : ∀ T edges, G.InClass S T → EdgesInto T edges → ∀ e,
    ∫ x, (G.undecAt rd C.k T edges e G.Good x : ℝ) ∂D
      ≤ θg * ∫ x, (travBy (fun z => (rd z).cut) T edges C.k x e : ℝ) ∂D
        + C.φe * (C.θe / 2 - 2 * θg) / 2
  goodStart : ∀ T, G.InClass S T → D.real (G.startAt rd C.k T G.Good) ≤ θgs
  goodPT : ∀ T edges, G.InClass S T → EdgesInto T edges →
    D.real (G.ptAt rd C.k T edges G.Good)
      ≤ θgpt * D.real (ReadModel.searchAt rd C.k T edges) + θgpt * C.φpt / 2

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

/-- Where a sub-round starts: a tree of the class, edges into its leaves, no records, a fresh
stretch. -/
def SubStart {σ : Type*} (G : ReadModel α σ) (S : ℕ) (s : TState α) : Prop :=
  G.InClass S s.tree ∧ EdgesInto s.tree s.edges ∧ (∀ q c, s.recs q c = []) ∧ s.n = 0 ∧ s.dis = 0

open scoped Classical in
/-- The sub-round from `s` keeps its tree over its first `j` probes, making fewer than `B` records
that are not true. -/
def SubOpen {σ : Type*} (G : ReadModel α σ) (C : TallyCfg) (rd : FreeMonoid α → ARU) :
    TState α → ℕ → ℕ → (T : ℕ) → (Fin T → FreeMonoid α) → Prop
  | _, B, 0, _, _ => 0 < B
  | _, B, _ + 1, 0, _ => 0 < B
  | s, B, j + 1, T + 1, xs => ∃ s', tallyStep C (fun z => (rd z).cut) s (xs 0) = .inl s'
      ∧ s'.tree = s.tree ∧ SubOpen G C rd s'
        (if xs 0 ∈ G.untrueAt rd C.k s.tree s.edges then B - 1 else B) j T (Fin.tail xs)

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

/-- A sub-round's chance of keeping its tree over its `subT` probes with fewer than `B` records
that are not true: some version keeping over `nEnd` probes, or `nRec` probes, each recording with
chance `θr`, making no more than the `|Q| |Σ| m` records true edges and targets hold and the
`B - 1` that are not true. -/
noncomputable def subOpen (C : TallyCfg) (nα nσ nRec B : ℕ) (θr βt : ℝ) : ℝ :=
  (subVersions C nα + 1) * βt + (1 - binomSfGe nRec θr (nσ * nα * C.m + B))

/-- `SubRound`: with the middles test's threshold `hP` at `hS + 1` searches, the success test's
`hS` at `nEnd` probes and `θr ≤ (1 - θpt') εd'`, a sub-round keeps its tree over its `subT`
probes with fewer than `B` records that are not true with chance at most `subOpen`. -/
def SubRound : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (G : ReadModel α σ)
    (rd : FreeMonoid α → ARU) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (C : TallyCfg) (S nEnd nRec hP hS B : ℕ) (θpt' θr εd' : ℝ),
    0 < C.m → Fintype.card σ + S + 3 ≤ C.Lmax → C.n₀ ≤ nEnd →
    C.n₀ ≤ hS + 1 → 0 ≤ C.θpt → C.θpt ≤ 1 → 0 ≤ C.εd → C.εd ≤ 1 → 0 ≤ θpt' → θpt' ≤ 1 →
    0 ≤ θr → θr ≤ 1 → 0 ≤ εd' → εd' ≤ 1 → θr ≤ (1 - θpt') * εd' →
    binomSfGe (hS + 1) C.θpt hP < C.a → 1 - binomSfGe nEnd C.εd (hS + 1) < C.a →
    C.a ≤ binomSfGe nEnd C.εd hS → C.φpt * nEnd ≤ hS + 1 →
    ∀ s, SubStart G S s → ∀ T, subT C (Fintype.card α) nEnd nRec ≤ T →
      (Measure.pi fun _ : Fin T => D)
          {xs | SubOpen G C rd s B (subT C (Fintype.card α) nEnd nRec) T xs}
        ≤ ENNReal.ofReal (subOpen C (Fintype.card α) (Fintype.card σ) nRec B θr
          (termLevel nEnd hP hS θpt' εd'))

open scoped Classical in
/-- Over its first `W` probes the run from `s` makes `u` more records that are not true, each at a
probe after which it goes on. -/
def UntrueHit {σ : Type*} (G : ReadModel α σ) (C : TallyCfg) (rd : FreeMonoid α → ARU) :
    TState α → ℕ → ℕ → (T : ℕ) → (Fin T → FreeMonoid α) → Prop
  | _, 0, _, _, _ => True
  | _, _ + 1, 0, _, _ => False
  | _, _ + 1, _ + 1, 0, _ => False
  | s, u + 1, W + 1, T + 1, xs => ∃ s', tallyStep C (fun z => (rd z).cut) s (xs 0) = .inl s'
      ∧ UntrueHit G C rd s'
        (if xs 0 ∈ G.untrueAt rd C.k s.tree s.edges then u else u + 1) W T (Fin.tail xs)

/-- The chance that the round's records that are not true reach `(S + 1) m` within `W` probes:
a probe's are at most `κ` times its undecided strings plus `ρ`, which the start's and the edges'
tests hold to `θs'` a probe past `Xs`, and at each of at most `2 Lmax |Σ|` edges to `θe` of its
positions past `Xe` once charged `φe` of the probes, with at most `L + 1` undecided strings and
`L + 1` positions a probe. -/
noncomputable def fakeRun (C : TallyCfg) (nα S L W : ℕ) (ρ Xe θs' Xs η ν : ℝ) : ℝ :=
  Real.exp (-η * ((S + 1) * C.m) + ν * (2 * C.Lmax * nα * Xe + Xs)
    + W * (ν * (C.θe * (L + 1) + θs' + 2 * C.Lmax * nα * C.φe) + (Real.exp η - 1) * ρ))

/-- `FakeRace`: with draws of length at most `L`, every edge excess at most `Xe`, the start's test
firing past `θs'` of `n` probes and `Xs`, and `(e^η - 1) κ (L + 1) ≤ 1 - e^{-ν (L + 1)}`, the
round's records that are not true reach `(S + 1) m` within `W` probes with chance at most
`fakeRun`; an edge charged less than `φe` of the probes holds its undecided strings to that. -/
def FakeRace : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (G : ReadModel α σ)
    (rd : FreeMonoid α → ARU) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (C : TallyCfg) (S L W : ℕ) (ρ θg θgs θgpt Xe θs' Xs η ν : ℝ),
    (∀ᵐ x ∂D, x.toList.length ≤ L) → 0 < C.m → Fintype.card σ + S + 3 ≤ C.Lmax → 0 ≤ C.θe →
    0 ≤ C.φe → 0 ≤ ρ → 0 ≤ Xe → (∀ j, C.exc j ≤ Xe) → 0 ≤ θs' → C.n₀ ≤ Xs →
    (∀ n h : ℕ, C.n₀ ≤ n → θs' * n + Xs ≤ h → binomSfGe n C.θs h < C.a) →
    0 ≤ η → 0 ≤ ν → (Real.exp η - 1) * G.κ * (L + 1) ≤ 1 - Real.exp (-(ν * (L + 1))) →
    TallyE G D C S ρ θg θgs θgpt rd →
    ∀ T, (Measure.pi fun _ : Fin T => D)
        {xs | UntrueHit G C rd tallyStart ((S + 1) * C.m) W T xs}
      ≤ ENNReal.ofReal (fakeRun C (Fintype.card α) S L W ρ Xe θs' Xs η ν)

/-- `HarvestGood`: with draws of length at most `L`, `θe ≥ 4θg`, every edge excess above
`4 L (1 + 4 θg Lmax) log(1/a)`, the start's and the middles' tests firing only where good
read-states' undecided reads (at `2θgpt` of searches for the middles) reach half their count with
chance at most `a`, and a stretch searching on at most `φpt/2` of its probes reaching `φpt` of
`n ≥ n₀` with chance at most `a`, the round ends in a harvest at a tree of the class most of whose
strings are at good read-states with chance at most `(2^(Lmax+1) |Σ| + T + 2T²) a`. -/
def HarvestGood : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (G : ReadModel α σ)
    (rd : FreeMonoid α → ARU) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (C : TallyCfg) (S L T : ℕ) (ρ θg θgs θgpt : ℝ),
    (∀ᵐ x ∂D, x.toList.length ≤ L) → 1 ≤ L → 0 < C.m → 0 < C.a → C.a ≤ 1 → 0 ≤ θg →
    4 * θg ≤ C.θe → 0 ≤ C.φe → Fintype.card σ + S + 2 ≤ C.Lmax →
    (∀ j, 4 * L * (1 + 4 * θg * C.Lmax) * Real.log (1 / C.a) < C.exc j) → 0 ≤ θgs → θgs ≤ 1 →
    (∀ t h, C.n₀ ≤ t → binomSfGe t C.θs h < C.a → binomSfGe t θgs ((h + 1) / 2) ≤ C.a) →
    0 ≤ θgpt → 2 * θgpt ≤ 1 →
    (∀ t h, C.n₀ ≤ t → binomSfGe t C.θpt h < C.a → binomSfGe t (2 * θgpt) ((h + 1) / 2) ≤ C.a) →
    0 ≤ C.φpt → C.φpt ≤ 1 →
    (∀ n, C.n₀ ≤ n → binomSfGe n (C.φpt / 2) ⌈C.φpt * n⌉₊ ≤ C.a) →
    TallyE G D C S ρ θg θgs θgpt rd →
    (Measure.pi fun _ : Fin T => D)
        {xs | RunEnds (tallyStep C fun z => (rd z).cut)
          (fun e s' => G.InClass S s'.tree ∧ e ≠ .tooBig ∧ ¬ GoodEnd G e s') tallyStart
          (List.ofFn xs)}
      ≤ ENNReal.ofReal ((2 ^ (C.Lmax + 1) * Fintype.card α + T + 2 * (T * T)) * C.a)

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

/-- `TallyRound`: under `SubRound`'s, `FakeRace`'s and `HarvestGood`'s conditions, over
`T ≥ (S + |Q| + 1) · subT` probes the round ends in success at a hypothesis disagreeing on fewer
than `εd` of the probes, or in a harvest most of whose strings are at read-states that are not
good, but for chance at most that of not `TallyE`, `fakeRun` over `(S + |Q| + 1) · subT` probes,
`(S + |Q| + 1) · subOpen` with an allowance of `(S + 1) m`, `HarvestGood`'s and
`SuccessSound`'s. -/
def TallyRound : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (G : ReadModel α σ)
    (read : FreeMonoid α → Ω → ARU) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (C : TallyCfg) (S nEnd nRec hP hS L T : ℕ) (ρ θg θgs θgpt θpt' θr εd' Xe θs' Xs η ν : ℝ),
    (∀ᵐ x ∂D, x.toList.length ≤ L) →
    0 ≤ ρ → 0 < C.m → Fintype.card σ + S + 3 ≤ C.Lmax → C.n₀ ≤ nEnd →
    C.n₀ ≤ hS + 1 → 0 ≤ C.θpt → C.θpt ≤ 1 → 0 ≤ C.εd → C.εd ≤ 1 → 0 ≤ θpt' → θpt' ≤ 1 →
    0 ≤ θr → θr ≤ 1 → 0 ≤ εd' → εd' ≤ 1 → θr ≤ (1 - θpt') * εd' →
    binomSfGe (hS + 1) C.θpt hP < C.a → 1 - binomSfGe nEnd C.εd (hS + 1) < C.a →
    C.a ≤ binomSfGe nEnd C.εd hS → C.φpt * nEnd ≤ hS + 1 →
    1 ≤ L → 0 < C.a → C.a ≤ 1 → 0 ≤ θg → 4 * θg ≤ C.θe → 0 ≤ C.φe →
    (∀ j, 4 * L * (1 + 4 * θg * C.Lmax) * Real.log (1 / C.a) < C.exc j) → 0 ≤ θgs → θgs ≤ 1 →
    (∀ t h, C.n₀ ≤ t → binomSfGe t C.θs h < C.a → binomSfGe t θgs ((h + 1) / 2) ≤ C.a) →
    0 ≤ θgpt → 2 * θgpt ≤ 1 →
    (∀ t h, C.n₀ ≤ t → binomSfGe t C.θpt h < C.a → binomSfGe t (2 * θgpt) ((h + 1) / 2) ≤ C.a) →
    0 ≤ C.φpt → C.φpt ≤ 1 →
    (∀ n, C.n₀ ≤ n → binomSfGe n (C.φpt / 2) ⌈C.φpt * n⌉₊ ≤ C.a) →
    0 ≤ C.θe → 0 ≤ Xe → (∀ j, C.exc j ≤ Xe) → 0 ≤ θs' → C.n₀ ≤ Xs →
    (∀ n h : ℕ, C.n₀ ≤ n → θs' * n + Xs ≤ h → binomSfGe n C.θs h < C.a) →
    0 ≤ η → 0 ≤ ν → (Real.exp η - 1) * G.κ * (L + 1) ≤ 1 - Real.exp (-(ν * (L + 1))) →
    (S + Fintype.card σ + 1) * subT C (Fintype.card α) nEnd nRec ≤ T →
    ∫⁻ ω, (Measure.pi fun _ : Fin T => D)
        {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut) (TallyEndsWell G D C (read · ω))
          tallyStart (List.ofFn xs)} ∂μ
      ≤ μ {ω | ¬ TallyE G D C S ρ θg θgs θgpt (read · ω)}
        + ENNReal.ofReal (fakeRun C (Fintype.card α) S L
          ((S + Fintype.card σ + 1) * subT C (Fintype.card α) nEnd nRec) ρ Xe θs' Xs η ν)
        + ENNReal.ofReal ((S + Fintype.card σ + 1) * subOpen C (Fintype.card α) (Fintype.card σ)
          nRec ((S + 1) * C.m) θr (termLevel nEnd hP hS θpt' εd'))
        + ENNReal.ofReal ((2 ^ (C.Lmax + 1) * Fintype.card α + T + 3 * (T * T)) * C.a)

end OrthoDFA

end
