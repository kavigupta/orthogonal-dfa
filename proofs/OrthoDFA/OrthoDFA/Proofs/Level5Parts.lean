import OrthoDFA.Proofs.Level5RiskDefs

/-!
# Level 5's parts

The family is the oracle: every string's read is accept, reject or undecided, independently
across strings, with a law that depends only on the string's state in the target DFA
(`FamilyReadBound`). `ReadModel` holds the target and those laws, as `TriOracle` does with the
goodness level beside it; `rd` is one draw of all the reads, each law reading its rarer decided
side at most `κ` times as often as it is undecided or at most `ε`. A read-state is good where it
is undecided at most `1.5θ` of the time. A state's true leaf is where its reads' majority sides
lead.

A tree in the class grows from the root's cut by splits each at a leaf, on a letter followed by
the midfix where two leaves part: genuine ones, separating two states whose true leaf is the split
leaf, their reads of the new midfix on different sides, and at most `S` that are not. The noise
event `NoiseEvent` is a hypothesis on `rd`, quantified over the class and edges into its leaves:
good read-states' undecided first reads average at most `θg` of an edge's first reads, up to a
slack the edge test's traffic gate covers, and `θgs` a probe at the start; at most `θgpt` of the
searches stop at an undecided middle read at a good read-state, up to a slack the middles test's
search-rate gate covers.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The proofs' view of the oracle: the target, each of its states' read law, the goodness level
`θ`, and the laws' `κ` and `ε`. -/
structure ReadModel (α σ : Type*) where
  M : _root_.DFA α σ
  readDist : σ → ARU → ℝ
  θ : ℝ
  κ : ℝ
  ε : ℝ
  nonneg : ∀ r v, 0 ≤ readDist r v
  rare : ∀ r, min (readDist r .accept) (readDist r .reject) ≤ max ε (κ * readDist r .undecided)

namespace ReadModel

/-- A read-state is good where it reads undecided at most `1.5θ` of the time. -/
def Good {σ : Type*} (G : ReadModel α σ) (r : σ) : Prop := G.readDist r .undecided ≤ 3 / 2 * G.θ

variable {σ : Type*} (G : ReadModel α σ)

/-- The side a read-state reads on more often. -/
def side (r : σ) : Bool := decide (G.readDist r .reject < G.readDist r .accept)

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
  | real {T : DTree α} {f : ℕ} {p t t₀ : List Bool} {c : α} : Grown T f → t ∈ T.leaves →
      t₀ ∈ T.leaves → G.GenuineSplit T p (FreeMonoid.of c * T.midfixAt (commonPrefix t t₀)) →
      Grown (T.splitAt (FreeMonoid.of c * T.midfixAt (commonPrefix t t₀)) p) f
  | fake {T : DTree α} {f : ℕ} {p t t₀ : List Bool} {c : α} : Grown T f → p ∈ T.leaves →
      t ∈ T.leaves → t₀ ∈ T.leaves → ¬ G.GenuineSplit T p (FreeMonoid.of c
          * T.midfixAt (commonPrefix t t₀)) →
      Grown (T.splitAt (FreeMonoid.of c * T.midfixAt (commonPrefix t t₀)) p) (f + 1)

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
  ((edgeHarvestBy (fun z => (rd z).cut) T edges k x e).filter fun b => P (G.M.eval b.toList)).length

/-- The probes whose start is undecided, at a read-state satisfying `P`. -/
def startAt (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (P : σ → Prop) :
    Set (FreeMonoid α) :=
  {x | k ≤ x.toList.length ∧ ∃ b, T.sift rd (prefixOf x k) = .inr b
    ∧ P (G.M.eval b.toList)}

/-- The probes whose search stops at an undecided middle, read at a read-state satisfying `P`. -/
def ptAt (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : Edges α) (P : σ → Prop) :
    Set (FreeMonoid α) :=
  {x | ∃ b, middleHarvestBy (fun z => (rd z).cut) T edges k x = [b] ∧ P (G.M.eval b.toList)}

end ReadModel

/-- Every learned edge points at a leaf. -/
def EdgesInto (T : DTree α) (edges : Edges α) : Prop :=
  ∀ q c t w, edges (q, c) = some (t, w) → t ∈ T.leaves

/-- The noise event: over the class and edges into its leaves, good read-states' undecided reads
are rare at each edge, at the start and at the middles searches stop at. -/
structure NoiseEvent {σ : Type*} (G : ReadModel α σ) (D : Measure (FreeMonoid α))
    (C : Level5Settings) (S : ℕ) (θg θgs θgpt : ℝ) (rd : FreeMonoid α → ARU) : Prop where
  goodEdge : ∀ T edges, G.InClass S T → EdgesInto T edges → ∀ e,
    ∫ x, (G.undecAt rd C.walkStart T edges e G.Good x : ℝ) ∂D
      ≤ θg * ∫ x, (edgeReadsBy (fun z => (rd z).cut) T edges C.walkStart x e : ℝ) ∂D
        + C.edgeTraffic * (C.edgeUndecidedRate - 8 * θg) / 4
  goodStart : ∀ T, G.InClass S T → D.real (G.startAt rd C.walkStart T G.Good) ≤ θgs
  goodPT : ∀ T edges, G.InClass S T → EdgesInto T edges →
    D.real (G.ptAt rd C.walkStart T edges G.Good)
      ≤ θgpt * D.real (searchAt rd C.walkStart T edges) + θgpt * C.searchShare / 2

open scoped Classical in
/-- The mean count of the strings a probe walked from `k` under `v` reads undecided at the site,
at read-states satisfying `P`. -/
noncomputable def siteMean {σ : Type*} (G : ReadModel α σ) (D : Measure (FreeMonoid α))
    (rd : FreeMonoid α → ARU) (k : ℕ) (site : Level5Site α) (P : σ → Prop)
    (v : DTree α × Edges α) : ℝ :=
  ∫ x, (((siteHarvest (fun z => (rd z).cut) k site v x).filter
    fun b => P (G.M.eval b.toList)).length : ℝ) ∂D

/-- The mean count of the strings a probe walked from `k` under `v` reads undecided at the site. -/
noncomputable def siteTotal (D : Measure (FreeMonoid α)) (rd : FreeMonoid α → ARU) (k : ℕ)
    (site : Level5Site α) (v : DTree α × Edges α) : ℝ :=
  ∫ x, ((siteHarvest (fun z => (rd z).cut) k site v x).length : ℝ) ∂D

/-- The harvest's sampler at the hypothesis `v` finds more strings at read-states that are not
good than at good ones, and at least `y` in all. -/
def SiteGood {σ : Type*} (G : ReadModel α σ) (D : Measure (FreeMonoid α)) (rd : FreeMonoid α → ARU)
    (k : ℕ) (y : ℝ) (site : Level5Site α) (v : DTree α × Edges α) : Prop :=
  2 * siteMean G D rd k site G.Good v < siteTotal D rd k site v ∧ y ≤ siteTotal D rd k site v

end OrthoDFA

end
