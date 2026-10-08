import OrthoDFA.Round

/-!
# The round, walked from position `k`

A probe is walked from where the cut places its first `k` letters, along learned edges only, and
checked against where the cut places the whole probe. `probeOutcome` says what one probe comes
to; a decided disagreement is searched, narrowing on decided reads only, down to two adjacent
undecided reads, a triple, or an edge.

The counterexample pass acts on edges and members. The gate then reads fresh draws against the
frozen hypothesis until its agreement's test and its ends test both settle, or to the batch's
end. Its share agreeing decides between passing and refusing, a refusal's decided disagreements
becoming the next pass's first probes. The limit halves where the searches end at pairs too
often, or where a refusal has nothing decided to carry.

`RoundAtK` makes one claim per outcome: the gate's agreement, the ends test, the pair test and a
refusal with nothing to carry, edges reaching the split test, members placed at the leaf whose
edge is missing, and triples harvested from badly read states but for their share among the
searches.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- Learned edges: each leaf's edge by a letter, its target and the member that voted for it. -/
abbrev Edges (α : Type*) := List Bool → α → Option (List Bool × FreeMonoid α)

/-- Following only learned edges from `p` along `cs`: the leaves visited, `p` first, or the leaf,
the letter and the step at which an edge is not learned. -/
def follow (edges : Edges α) : List Bool → List α → List (List Bool) ⊕ (List Bool × α × ℕ)
  | p, [] => .inl [p]
  | p, c :: cs =>
    match edges p c with
    | some (q, _) =>
      match follow edges q cs with
      | .inl ps => .inl (p :: ps)
      | .inr (s, c', i) => .inr (s, c', i + 1)
    | none => .inr (p, c, 0)

/-- How a walk from position `k` ends: the cut cannot place the first `k` letters, an edge out of
leaf `s` by the letter at position `j` is not learned, or the walk reaches the end, the leaves it
visits listed from position `k`. -/
inductive KWalk (α : Type*)
  | anchor
  | edge (s : List Bool) (c : α) (j : ℕ)
  | reached (ps : List (List Bool))

/-- What one probe comes to: the walk ends where the cut places the probe; the cut cannot place
the first `k` letters, `w`; it cannot place the probe, or, at an unlearned edge at `j`, its first
`j + 1` or `j` letters, `w`; the search of a decided disagreement ends at undecided reads at `j`
and `j + 1`, at the edge into `fd` of the walk `ps`, or at an undecided read at `j` between an
agreeing and a disagreeing one; or an unlearned edge's source is where the cut places `u`. -/
inductive Outcome (α : Type*)
  | agree
  | startUndecided (w : FreeMonoid α)
  | endUndecided (w : FreeMonoid α)
  | pair (j : ℕ)
  | edge (ps : List (List Bool)) (fd : ℕ)
  | triple (j : ℕ)
  | member (u : FreeMonoid α)

/-- What the pass does with a disagreeing probe: split a leaf, add `sprime` as a member, stop at
a string the cut cannot place, or drop it. -/
inductive SeedResult (α : Type*)
  | split (d : FreeMonoid α) (s1 : List Bool) (y sprime : FreeMonoid α)
  | member (s1 : List Bool) (sprime : FreeMonoid α)
  | stopped (b : FreeMonoid α)
  | dropped

namespace DTree

/-- `first_disagreement`, saying why it finds none: the midfix where `x·pre` and `y·pre` part,
the first string on the way the cut cannot place, or `none` where they reach a leaf together. -/
def parting (cut : FreeMonoid α → Option Bool) (x y pre : FreeMonoid α) :
    DTree α → Option (FreeMonoid α ⊕ FreeMonoid α)
  | .leaf => none
  | .node m r a =>
    match cut (x * (pre * m)), cut (y * (pre * m)) with
    | some true, some true => a.parting cut x y pre
    | some false, some false => r.parting cut x y pre
    | some _, some _ => some (.inl (pre * m))
    | none, _ => some (.inr (x * (pre * m)))
    | _, none => some (.inr (y * (pre * m)))

end DTree

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

/-- `bracket` from `lo`, agreeing, to `hi`, disagreeing, over the walk `ps`, `agrees i` being
`none` where the read of the first `i` letters is undecided: a decided middle narrows, an
undecided one reads its neighbours. -/
def bracketAt (agrees : ℕ → Option Bool) (ps : List (List Bool)) : ℕ → ℕ → ℕ → Outcome α
  | 0, _, hi => .edge ps hi
  | fuel + 1, lo, hi =>
    if lo + 1 < hi then
      let ag := fun p => if p = lo then some true else if p = hi then some false else agrees p
      let mid := (lo + hi) / 2
      match ag mid with
      | some true => bracketAt agrees ps fuel mid hi
      | some false => bracketAt agrees ps fuel lo mid
      | none =>
        match ag (mid - 1), ag (mid + 1) with
        | none, _ => .pair (mid - 1)
        | some _, none => .pair mid
        | some true, some false => .triple mid
        | some true, some true => bracketAt agrees ps fuel (mid + 1) hi
        | some false, some _ => bracketAt agrees ps fuel lo (mid - 1)
    else .edge ps hi

/-- How many middles `bracketAt` visits. -/
def visited (agrees : ℕ → Option Bool) : ℕ → ℕ → ℕ → ℕ
  | 0, _, _ => 0
  | fuel + 1, lo, hi =>
    if lo + 1 < hi then
      let ag := fun p => if p = lo then some true else if p = hi then some false else agrees p
      let mid := (lo + hi) / 2
      match ag mid with
      | some true => 1 + visited agrees fuel mid hi
      | some false => 1 + visited agrees fuel lo mid
      | none =>
        match ag (mid - 1), ag (mid + 1) with
        | some true, some true => 1 + visited agrees fuel (mid + 1) hi
        | some false, some _ => 1 + visited agrees fuel lo (mid - 1)
        | _, _ => 1
    else 0

variable (K : StageKnobs α) (R : CutReads α)

/-- The walk of `x` from position `k`, along `edges`. -/
noncomputable def kWalk (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) : KWalk α :=
  match t.sift R.cut (prefixOf x k) with
  | .inr _ => .anchor
  | .inl p =>
    match follow edges p (x.toList.drop k) with
    | .inl ps => .reached ps
    | .inr (s, c, i) => .edge s c (k + i)

/-- The leaves the walk of `x` from `k` visits up to position `j`. -/
noncomputable def walkTo (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) (j : ℕ) :
    List (List Bool) :=
  match t.sift R.cut (prefixOf x k) with
  | .inr _ => []
  | .inl p => (follow edges p ((x.toList.drop k).take (j - k))).elim id fun _ => []

/-- `walk` and `_edge_block`: the probe's outcome where it is not a decided disagreement, else the
walk's leaves and the length `hi` whose prefix the cut places off the walk. At an unlearned edge
out of `s` at position `j`, the prefix before it is a member where the cut places it at `s`, and
a decided disagreement where it places it elsewhere. -/
noncomputable def walkCheck (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    Outcome α ⊕ (List (List Bool) × ℕ) :=
  match kWalk R t edges k x with
  | .anchor => .inl (.startUndecided (prefixOf x k))
  | .edge s _ j =>
    match t.sift R.cut (prefixOf x (j + 1)) with
    | .inr _ => .inl (.endUndecided (prefixOf x (j + 1)))
    | .inl _ =>
      match t.sift R.cut (prefixOf x j) with
      | .inr _ => .inl (.endUndecided (prefixOf x j))
      | .inl p => if p = s then .inl (.member (prefixOf x j)) else .inr (walkTo R t edges k x j, j)
  | .reached ps =>
    match t.sift R.cut x with
    | .inr _ => .inl (.endUndecided x)
    | .inl a => if some a = ps.getLast? then .inl .agree else .inr (ps, x.toList.length)

/-- Where the bisection places `x`'s first `i` letters: where the cut does, else where the gate's
reading does. -/
noncomputable def place (t : DTree α) (x : FreeMonoid α) (i : ℕ) : List Bool :=
  (t.sift R.cut (prefixOf x i)).elim id fun _ => (t.halfway R.cut R.mid (prefixOf x i)).1

/-- Whether the cut places `x`'s first `i` letters where the walk does, or `none` where it cannot
place them. -/
noncomputable def agreesAt (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool) (i : ℕ) :
    Option Bool :=
  (t.sift R.cut (prefixOf x i)).elim (fun p => some (decide (p = walkAt i))) fun _ => none

/-- What one probe comes to. -/
noncomputable def probeOutcome (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    Outcome α :=
  (walkCheck R t edges k x).elim id fun d =>
    bracketAt (agreesAt R t x fun j => d.1.getD (j - k) []) d.1 (d.2 - k) k d.2

/-- How many middles the search of a decided disagreement visits. -/
noncomputable def visits (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) : ℕ :=
  (walkCheck R t edges k x).elim (fun _ => 0) fun d =>
    visited (agreesAt R t x fun j => d.1.getD (j - k) []) (d.2 - k) k d.2

/-- `_act_on_disagreement` at the edge into `fd` of a probe whose walk from `k` visits `ps`: the
guards, then the split test, a no-split answered as undecided is (#412). -/
noncomputable def seedStep (t : DTree α) (pool : List (FreeMonoid α)) (edges : Edges α) (k : ℕ)
    (x : FreeMonoid α) (ps : List (List Bool)) (fd : ℕ) : SeedResult α :=
  let walkAt := fun j => ps.getD (j - k) []
  match x.toList[fd - 1]? with
  | none => .dropped
  | some c =>
    let s1 := walkAt (fd - 1)
    let sprime := prefixOf x (fd - 1)
    match edges s1 c with
    | none => .dropped
    | some (s2, y) =>
      if s2 ≠ walkAt fd then .dropped
      else
      match t.sift R.cut sprime with
      | .inr b => .stopped b
      | .inl p =>
        if p ≠ s1 ∨ t.sift R.cut y ≠ .inl s1 then .dropped
        else
        match t.parting R.cut y sprime (FreeMonoid.of c) with
        | none => .dropped
        | some (.inr b) => .stopped b
        | some (.inl d) =>
          match verdict K R t pool s1 d (t.paths.length * Fintype.card α) with
          | .split => .split d s1 y sprime
          | _ => .member s1 sprime

/-- What the pass carries: the tree, the population, the learned edges, and the probes since the
last split or evidence weighed. -/
structure KState (α : Type*) where
  tree : DTree α
  pool : List (FreeMonoid α)
  edges : Edges α
  streak : ℕ

/-- Every edge re-voted, as after every probe. -/
noncomputable def closeK (t : DTree α) (pool : List (FreeMonoid α)) (edges : Edges α)
    (streak : ℕ) : KState α :=
  ⟨t, pool, closeEdges K R t pool edges, streak⟩

/-- One probe of the counterexample pass. -/
noncomputable def probeStepK (k : ℕ) (s : KState α) (x : FreeMonoid α) : KState α :=
  let quiet := closeK K R s.tree s.pool s.edges (s.streak + 1)
  match probeOutcome R s.tree s.edges k x with
  | .edge ps fd =>
      match seedStep K R s.tree s.pool s.edges k x ps fd with
      | .split d s1 y sprime =>
        let cleared : Edges α := fun p c' =>
          match s.edges p c' with
          | some (q, w) => if p = s1 ∨ q = s1 then none else some (q, w)
          | none => none
        closeK K R (s.tree.splitAt d s1) (s.pool ++ ([y, sprime].filter (· ∉ s.pool))) cleared 0
      | .member _ sprime => closeK K R s.tree (sprime :: s.pool.filter (· ≠ sprime)) s.edges 0
      | _ => quiet
  | .member u =>
    closeK K R s.tree (if u ∈ s.pool then s.pool else s.pool ++ [u]) s.edges (s.streak + 1)
  | _ => quiet

/-- The pass: probes in order until `patience` in a row are quiet. -/
noncomputable def runPassK (k : ℕ) (s : KState α) (probes : List (FreeMonoid α)) : KState α :=
  probes.foldl (fun s x => if K.patience ≤ s.streak then s else probeStepK K R k s x) s

/-- The first state: the root reads at `ε`, the population is the table's prefixes, and the
edges are voted once. -/
noncomputable def initialK (seed : List (FreeMonoid α)) : KState α :=
  closeK K R (.node 1 .leaf .leaf) seed (fun _ _ => none) 0

open scoped Classical in
/-- How many of a batch's first `n` draws satisfy `P`. -/
noncomputable def hitsIn {N : ℕ} (b : Fin N → FreeMonoid α) (P : FreeMonoid α → Prop) (n : ℕ) :
    ℕ :=
  (Finset.univ.filter fun i : Fin N => (i : ℕ) < n ∧ P (b i)).card

/-- `binomial_side_of_boundary` once `n₀` draws are in: above `θ` when `Bin(n, θ)` reaches `h`
hits with chance below `a`, below when it stays at or under `h` with chance below `a`. -/
noncomputable def rateSide (θ a : ℝ) (n₀ n h : ℕ) : Option Bool :=
  if n₀ ≤ n then
    if binomSfGe n θ h < a then some true
    else if 1 - binomSfGe n θ (h + 1) < a then some false
    else none
  else none

/-- The looks at which `read_fresh` tests: `n₀` doubling while within the batch of `N`, and `N`. -/
def lookSet (n₀ N : ℕ) : Finset ℕ :=
  insert N (((Finset.range (Nat.log 2 (N / n₀) + 1)).image (n₀ * 2 ^ ·)).filter (· ≤ N))

/-- The look at which `read_fresh` stops: the first at which the tests of `P` against `θ` and of
`P'` against `θ'` have both settled, else the batch's end. -/
noncomputable def stopLook (θ θ' a : ℝ) (n₀ : ℕ) {N : ℕ} (b : Fin N → FreeMonoid α)
    (P P' : FreeMonoid α → Prop) : ℕ :=
  (((lookSet n₀ N).sort (· ≤ ·)).find? fun n => (rateSide θ a n₀ n (hitsIn b P n)).isSome
    && (rateSide θ' a n₀ n (hitsIn b P' n)).isSome).getD N

/-- `SequentialRate.above` read at look `n`: the side its test has settled on, or, where it has
not, whether the rate so far exceeds `θ`. -/
noncomputable def sideAt (θ a : ℝ) (n₀ : ℕ) {N : ℕ} (b : Fin N → FreeMonoid α)
    (P : FreeMonoid α → Prop) (n : ℕ) : Prop :=
  match rateSide θ a n₀ n (hitsIn b P n) with
  | some s => s = true
  | none => θ * n < hitsIn b P n

/-- The gate's reading of a draw disagrees with the walk: from where the gate places its first `k`
letters, the walk along learned edges meets an unlearned edge, or ends somewhere other than where
the gate places the whole draw. -/
def gateDisagrees (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) : Prop :=
  match follow edges (place R t x k) (x.toList.drop k) with
  | .inl ps => ps.getLast? ≠ some (place R t x x.toList.length)
  | .inr _ => True

/-- The search's harvest from a triple: the read of the middle the cut leaves undecided. -/
noncomputable def tripleRead (t : DTree α) (x : FreeMonoid α) (j : ℕ) : Option (FreeMonoid α) :=
  (t.sift R.cut (prefixOf x j)).getRight?

/-- The gate searches the draw: it is a decided disagreement. -/
def Bisected (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) : Prop :=
  (walkCheck R t edges k x).isRight

/-- `x`'s sift is left undecided at a node below the root. -/
def DeepUndecided (t : DTree α) (x : FreeMonoid α) : Prop :=
  (t.sift R.cut x).isRight ∧ 2 ≤ (t.route R.cut x).1.length

/-- The probe's start or end is left undecided below the root. -/
def EndsDeep (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) : Prop :=
  ∃ w, (probeOutcome R t edges k x = .startUndecided w
    ∨ probeOutcome R t edges k x = .endUndecided w) ∧ DeepUndecided R t w

open scoped Classical in
/-- What the gate's reading of batch `bg` against the frozen hypothesis `s` claims, read up to
`T`, the first look at which the agreement's test against `acc` and the ends test against
`min(2·(depth − 1)·f, 1)`, two sifts per draw, have both settled:
* the agreement: passing, its share agreeing at least `acc`, the gate's reading disagrees on at
  most `1 − acc + δ` of draws; refusing, on at least `1 − acc − δ`;
* the ends: the test reading above, the draws whose start or end is undecided below the root are
  at least the threshold less `δ`; reading below, at most the threshold and `δ`;
* the pairs: the pair test over the searched draws tripping, which halves the limit, more than
  `θp` of searched draws end at a pair; a refusal with none of its draws searched, which halves
  it too, has at most `δ` of draws searched;
* an edge: run again, the probe splits a leaf, adds a member, or stops at a string the cut cannot
  place;
* a member: the cut places it at a leaf whose edge by some letter is unlearned, and places it
  followed by that letter. -/
def RoundAtKHolds (s : KState α) (D : Measure (FreeMonoid α)) (k : ℕ) (acc θp f a δ : ℝ)
    (n₀ : ℕ) {ng : ℕ} (bg : Fin ng → FreeMonoid α) : Prop :=
  let t := s.tree
  let e := s.edges
  let agree := fun x => ¬ gateDisagrees R t e k x
  let θe := min (2 * ((t.depth - 1 : ℕ) : ℝ) * f) 1
  let T := stopLook acc θe a n₀ bg agree (EndsDeep R t e k)
  ((acc * T ≤ hitsIn bg agree T → D.real {x | gateDisagrees R t e k x} ≤ 1 - acc + δ)
    ∧ ((hitsIn bg agree T : ℝ) < acc * T → 1 - acc - δ ≤ D.real {x | gateDisagrees R t e k x}))
  ∧ ((sideAt θe a n₀ bg (EndsDeep R t e k) T → θe - δ ≤ D.real {x | EndsDeep R t e k x})
    ∧ (¬ sideAt θe a n₀ bg (EndsDeep R t e k) T → D.real {x | EndsDeep R t e k x} ≤ θe + δ))
  ∧ ((binomSfGe (hitsIn bg (Bisected R t e k) T) θp
        (hitsIn bg (fun x => ∃ j, probeOutcome R t e k x = .pair j) T) < a →
      θp * D.real {x | Bisected R t e k x}
        < D.real {x | ∃ j, probeOutcome R t e k x = .pair j})
    ∧ ((hitsIn bg agree T : ℝ) < acc * T → (∀ i : Fin ng, (i : ℕ) < T → ¬ Bisected R t e k (bg i)) →
      D.real {x | Bisected R t e k x} ≤ δ))
  ∧ (∀ i ps fd, probeOutcome R t e k (bg i) = .edge ps fd →
      seedStep K R t s.pool e k (bg i) ps fd ≠ .dropped)
  ∧ (∀ i u, probeOutcome R t e k (bg i) = .member u →
      ∃ p c, e p c = none ∧ t.sift R.cut u = .inl p ∧ (t.sift R.cut (u * FreeMonoid.of c)).isLeft)

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}

/-- Strings the pass can read: a seed string or a probe's prefix, then at most a letter, then `1`
or a midfix of `t`. -/
noncomputable def passReadSet (seed probes : List (FreeMonoid α)) (t : DTree α) :
    Finset (FreeMonoid α) :=
  (((seed ++ probes.flatMap fun p => (List.range (p.toList.length + 1)).map (prefixOf p)).toFinset
    ×ˢ insert 1 (Finset.univ.image FreeMonoid.of)) ×ˢ insert 1 t.midfixes).image
    fun z => z.1.1 * z.1.2 * z.2

/-- The most of `D` on draws beginning with any one string of length `k`. -/
noncomputable def prefixMax (D : Measure (FreeMonoid α)) (k : ℕ) : ℝ :=
  ⨆ p : FreeMonoid α, if p.toList.length = k then D.real {x | p.toList <+: x.toList} else 0

/-- What the triples' harvest claims of the hypothesis `s` that reads `R`: the draws whose triple
harvests a read at a state read undecided less than `uGood` of the time are at most `uGood` times
the depth times the middles the searches visit, but for the draws whose middles the pass
may have read and a fluctuation `ε` in units of the largest a draw can add. -/
def TripleHolds (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (s : KState α) (D : Measure (FreeMonoid α)) (k L : ℕ)
    (seed probes : List (FreeMonoid α)) (uGood ε : ℝ) : Prop :=
  D.real {x | ∃ j b, probeOutcome R s.tree s.edges k x = .triple j
      ∧ tripleRead R s.tree x j = some b ∧ stateIndecision A O B F (A.state b) < uGood}
    ≤ uGood * s.tree.depth * ∫ x, (visits R s.tree s.edges k x : ℝ) ∂D
      + (passReadSet seed probes s.tree).card * prefixMax D (k + 1)
      + ε * (1 + uGood * s.tree.depth * L)

/-- `RoundAtK`: for any reads, the gate's batch makes the round's claims but for the tests'
failure chances at each of the `log₂(ng/n₀) + 2` looks, Hoeffding's tails and a refusal's chance
of missing every searched draw; and over the oracle's noise the triples' claim holds but for
Chebyshev's bound, the draws spread at most `prefixMax D k` over any first `k` letters. -/
def RoundAtK : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {Q : Type*} (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α))
    (B : State) (F : Finset (FreeMonoid α)) (K : StageKnobs α) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (k L ng n₀ : ℕ) (seed probes : List (FreeMonoid α))
    (acc θp f a δ uGood ε : ℝ),
    0 ≤ acc → acc ≤ 1 → 0 ≤ θp → θp ≤ 1 → 0 ≤ f → 0 ≤ a → 0 ≤ δ → 0 ≤ uGood → 0 < ε →
    (∀ᵐ x ∂D, x.toList.length = L) → SuffixFree (F ∪ K.train F) →
    (∀ R : CutReads α,
      (Measure.pi fun _ : Fin ng => D).real
          {bg | ¬ RoundAtKHolds K R (runPassK K R k (initialK K R seed) probes) D k acc θp f a δ
            n₀ bg}
        ≤ (3 * (Nat.log 2 (ng / n₀) + 2) + 1) * a + 2 * Real.exp (-2 * ng * δ ^ 2)
          + Real.exp (-(min n₀ ng : ℕ) * δ))
    ∧ μ.real {ω | ¬ TripleHolds (readsAt O B F ω) A O B F
        (runPassK K (readsAt O B F ω) k (initialK K (readsAt O B F ω) seed) probes) D k L seed
        probes uGood ε}
      ≤ prefixMax D k / ε ^ 2

end OrthoDFA
