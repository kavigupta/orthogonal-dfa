import OrthoDFA.IdealRound

/-!
# A round with random reads, none of them wrong

Every string's read is drawn once, independently across strings, from a law fixed by the
string's target state: undecided with chance `U q`, otherwise on the state's side, never on the
other. A read-state is bad where it is undecided more than `1.5θ` of the time.

The round keeps the shape of `OrthoDFA.IdealRound`: each probe is walked from `k` along the
learned edges and sifted whole, a binary search finds where the two part, a clean disagreement
records its edge and target, `m` records redirect the edge or, once it has been redirected, split
its leaf, and a probe at an unlearned edge learns it. A probe shorter than `k` has no walk and is
counted as agreeing. Its counts, kept over the stretch since the hypothesis last changed, feed the
tests of `OrthoDFA.TallyLoop`, checked after every probe:
* the start's undecided rate above `θs` (`rateSide`) harvests the start's undecided strings;
* an edge read at least `τe` times a probe whose undecided reads exceed `θe` of its reads by
  `exc` harvests them;
* with searches at least `qg` of the probes, the searches stopping at an undecided middle above
  `θpt` of them harvest those middles;
* the disagreement rate settling below `εd` ends the round consistent.

`RandomRoundCorrect`: with at least `stretches · Ns` probes, the round fails to end well
(`OrthoDFA.RoundEnd`) with chance at most `noiseRisk`, the chance that the probes that can read a
good read-state undecided reach `G` of the probes for some tree the round can grow, plus
`stretches` times `stretchRisk`, what one stretch risks over its first `Ns` probes.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory
open OrthoDFA.Ideal (DTree Edges walk pre search agrees probe Outcome upd2 lcp retarget Disagrees)

variable {α : Type*}

/-- `binomial_side_of_boundary` once `n₀` draws are in: above `θ` when `Bin(n, θ)` reaches `h`
hits with chance below `a`, below when it stays at or under `h` with chance below `a`. -/
noncomputable def rateSide (θ a : ℝ) (n₀ n h : ℕ) : Option Bool :=
  if n₀ ≤ n then
    if binomSfGe n θ h < a then some true
    else if 1 - binomSfGe n θ (h + 1) < a then some false
    else none
  else none

/-- The positions `search` sifts: each middle, and its neighbours where it is undecided. -/
def searchPos (ag : ℕ → Option Bool) (lo hi : ℕ) : List ℕ :=
  if _h : hi ≤ lo + 1 then []
  else
    match ag ((lo + hi) / 2) with
    | some true => (lo + hi) / 2 :: searchPos ag ((lo + hi) / 2) hi
    | some false => (lo + hi) / 2 :: searchPos ag lo ((lo + hi) / 2)
    | none =>
      match ag ((lo + hi) / 2 - 1), ag ((lo + hi) / 2 + 1) with
      | none, _ => [(lo + hi) / 2, (lo + hi) / 2 - 1]
      | _, none => [(lo + hi) / 2, (lo + hi) / 2 - 1, (lo + hi) / 2 + 1]
      | some true, some false => [(lo + hi) / 2, (lo + hi) / 2 - 1, (lo + hi) / 2 + 1]
      | some false, _ =>
        (lo + hi) / 2 :: ((lo + hi) / 2 - 1) :: ((lo + hi) / 2 + 1)
          :: searchPos ag lo ((lo + hi) / 2 - 1)
      | some true, some true =>
        (lo + hi) / 2 :: ((lo + hi) / 2 - 1) :: ((lo + hi) / 2 + 1)
          :: searchPos ag ((lo + hi) / 2 + 1) hi
termination_by hi - lo
decreasing_by all_goals omega

/-- How many reads the sift of `z` makes: one per node down to a leaf or an undecided read. -/
def siftReads (read : FreeMonoid α → ARU) : DTree α → FreeMonoid α → ℕ
  | .leaf, _ => 0
  | .node m r a, z =>
    match read (z * m) with
    | .accept => siftReads read a z + 1
    | .reject => siftReads read r z + 1
    | .undecided => 1

/-- The walk's edge out of position `i`, or where the probe ends there, its last edge. -/
def edgeOf (x : FreeMonoid α) (st : ℕ → List Bool) (i : ℕ) : Option (List Bool × α) :=
  match x.toList[i]? with
  | some c => some (st i, c)
  | none => (x.toList[i - 1]?).map fun c => (st (i - 1), c)

section Probe

variable (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α)

/-- A probe's walk, and the positions it sifts past its start, as `probe` reads them. -/
def probePos : (ℕ → List Bool) × List ℕ :=
  match T.sift read (pre x k) with
  | .inr _ => (fun _ => [], [])
  | .inl a =>
    let ss := walk E a (x.toList.drop k)
    let st := fun p => ss.getD (p - k) []
    let j := k + ss.length - 1
    match x.toList[j]? with
    | some _ =>
      match T.sift read (pre x (j + 1)), T.sift read (pre x j) with
      | .inr _, _ => (st, [j + 1])
      | _, .inr _ => (st, [j + 1, j])
      | .inl _, .inl s =>
        (st, if s = st j then [j + 1, j] else (j + 1) :: j :: searchPos (agrees read T x st) k j)
    | none =>
      match T.sift read x with
      | .inr _ => (st, [j])
      | .inl e => (st, if e = st j then [j] else j :: searchPos (agrees read T x st) k j)

/-- A probe, which counts as agreeing where it is shorter than `k`. -/
def probeR : Outcome α := if x.toList.length < k then .agree else probe read T E k x

/-- What each position the probe sifts past its start charges to its edge: the reads, and the
string read undecided, if any. -/
def charges : List (List Bool × α × ℕ × Option (FreeMonoid α)) :=
  if x.toList.length < k then []
  else
    (((probePos read T E k x).2.filter fun i => k < i ∧ i ≤ x.toList.length).dedup).filterMap
      fun i => (edgeOf x (probePos read T E k x).1 i).map fun e =>
        (e.1, e.2, siftReads read T (pre x i), (T.sift read (pre x i)).elim (fun _ => none) some)

end Probe

variable [DecidableEq α]

def readsAt (ch : List (List Bool × α × ℕ × Option (FreeMonoid α))) (p : List Bool) (c : α) : ℕ :=
  ((ch.filter fun e => e.1 = p ∧ e.2.1 = c).map fun e => e.2.2.1).sum

def undAt (ch : List (List Bool × α × ℕ × Option (FreeMonoid α))) (p : List Bool) (c : α) :
    List (FreeMonoid α) :=
  (ch.filter fun e => e.1 = p ∧ e.2.1 = c).filterMap fun e => e.2.2.2

/-- The probe searched: its walk and its sift parted at decided reads. -/
def searches : Outcome α → Bool
  | .edge .. | .pair .. | .triple .. => true
  | _ => false

def startStr : Outcome α → List (FreeMonoid α)
  | .startU z => [z]
  | _ => []

/-- The undecided middle the search stopped at. -/
def ptStr : Outcome α → List (FreeMonoid α)
  | .pair _ zs | .triple _ zs => zs.take 1
  | _ => []

/-- The round's settings: the walk's start, the records that move an edge, the most leaves, and
the tests' first look, level and thresholds. -/
structure Cfg where
  k : ℕ
  m : ℕ
  Lmax : ℕ
  n₀ : ℕ
  a : ℝ
  θs : ℝ
  θe : ℝ
  τe : ℝ
  exc : ℝ
  θpt : ℝ
  qg : ℝ
  εd : ℝ

/-- The hypothesis, whether each edge has been redirected since it was learned, and over the
stretch since the hypothesis last changed: the records by edge and target, the probes, the
searches, the start's undecided strings, the undecided middles, and each edge's reads and
undecided strings. -/
structure RState (α : Type*) where
  tree : DTree α
  edges : Edges α
  moved : List Bool → α → Bool
  recs : List (List Bool × α × List Bool)
  n : ℕ
  dis : ℕ
  startH : List (FreeMonoid α)
  pt : List (FreeMonoid α)
  reads : List Bool → α → ℕ
  und : List Bool → α → List (FreeMonoid α)

/-- How the round ends: consistent, a harvest, or the tree too big. -/
inductive REnd (α : Type*)
  | consistent
  | harvest (zs : List (FreeMonoid α))
  | tooBig

def fresh (T : DTree α) (E : Edges α) (moved : List Bool → α → Bool) : RState α :=
  ⟨T, E, moved, [], 0, 0, [], [], fun _ _ => 0, fun _ _ => []⟩

/-- The root reads at `ε`; no edge is learned. -/
def start : RState α := fresh (.node 1 .leaf .leaf) (fun _ _ => none) fun _ _ => false

variable (read : FreeMonoid α → ARU) (C : Cfg)

/-- A probe counted into the stretch. -/
def charge (s : RState α) (x : FreeMonoid α) : RState α :=
  { s with
    n := s.n + 1
    dis := s.dis + if searches (probeR read s.tree s.edges C.k x) then 1 else 0
    startH := s.startH ++ startStr (probeR read s.tree s.edges C.k x)
    pt := s.pt ++ ptStr (probeR read s.tree s.edges C.k x)
    reads := fun p c => s.reads p c + readsAt (charges read s.tree s.edges C.k x) p c
    und := fun p c => s.und p c ++ undAt (charges read s.tree s.edges C.k x) p c }

open scoped Classical in
/-- The tests, in turn: the start's undecided rate, an edge's undecided reads, the undecided
middles among the searches, and the disagreement rate. -/
noncomputable def look [Fintype α] (s : RState α) : Option (REnd α) :=
  if rateSide C.θs C.a C.n₀ s.n s.startH.length = some true then some (.harvest s.startH)
  else if h : ∃ e ∈ s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α),
      C.τe * s.n ≤ s.reads e.1 e.2 ∧ C.exc + C.θe * s.reads e.1 e.2 ≤ (s.und e.1 e.2).length then
    some (.harvest (s.und h.choose.1 h.choose.2))
  else if C.qg * s.n ≤ s.dis ∧ rateSide C.θpt C.a C.n₀ s.dis s.pt.length = some true then
    some (.harvest s.pt)
  else if rateSide C.εd C.a C.n₀ s.n s.dis = some false then some .consistent
  else none

noncomputable def finish [Fintype α] (s : RState α) : RState α ⊕ REnd α :=
  match look C s with
  | some e => .inr e
  | none => .inl s

def split (s : RState α) (p : List Bool) (d : FreeMonoid α) : RState α ⊕ REnd α :=
  let T' := s.tree.splitAt d p
  if C.Lmax < T'.leaves.length then .inr .tooBig
  else .inl (fresh T' (retarget read T' p s.edges) fun _ _ => false)

/-- The edge `(p, c)` has `m` records at `t`, with `u` at `p`: split where it has been
redirected, else redirect it. -/
def fix (s : RState α) (p : List Bool) (c : α) (u : FreeMonoid α) (t : List Bool) :
    RState α ⊕ REnd α :=
  match s.edges p c with
  | some (t₀, _) =>
    if s.moved p c then split read C s p (FreeMonoid.of c * s.tree.midAt (lcp t t₀))
    else .inl (fresh s.tree (upd2 s.edges p c (some (t, u))) (upd2 s.moved p c true))
  | none => .inl s

noncomputable def step [Fintype α] (s : RState α) (x : FreeMonoid α) : RState α ⊕ REnd α :=
  match probeR read s.tree s.edges C.k x with
  | .member p c u t =>
    .inl (fresh s.tree (upd2 s.edges p c (some (t, u))) (upd2 s.moved p c false))
  | .edge p c u t =>
    if C.m ≤ (s.recs ++ [(p, c, t)]).count (p, c, t) then fix read C s p c u t
    else finish C (charge read C { s with recs := s.recs ++ [(p, c, t)] } x)
  | _ => finish C (charge read C s x)

/-- The round over the probes; the state it ended in, with `none` where the probes ran out. -/
noncomputable def round [Fintype α] : RState α → List (FreeMonoid α) → RState α × Option (REnd α)
  | s, [] => (s, none)
  | s, x :: xs =>
    match step read C s x with
    | .inl s' => round s' xs
    | .inr e => (s, some e)

def toRoundEnd : Option (REnd α) → RoundEnd α
  | some .consistent => .consistent
  | some (.harvest zs) => .harvest zs
  | _ => .failed

/-! ## The claim -/

/-- A read-state is bad where it reads undecided more than `1.5θ` of the time. -/
def BadAt {σ : Type*} (U : σ → ℝ) (θ : ℝ) (q : σ) : Prop := 3 / 2 * θ < U q

/-- `E[f(Bin(n, p))]`. -/
noncomputable def binE (n : ℕ) (p : ℝ) (f : ℕ → ℝ) : ℝ :=
  ∑ j ∈ Finset.range (n + 1), (n.choose j : ℝ) * p ^ j * (1 - p) ^ (n - j) * f j

/-- The least count at which `rateSide θ` calls a rate above `θ` after `n` trials. -/
noncomputable def hFire (θ a : ℝ) (n : ℕ) : ℕ := sInf {h | binomSfGe n θ h < a}

/-- Each split, and each edge's learning and redirect between splits, changes the hypothesis
once; the stretches are one more than those changes. -/
def stretches (C : Cfg) (nα : ℕ) : ℕ := C.Lmax * (2 * C.Lmax * nα + 1) + 1

/-- The chance that the probes that can read a string with a read of chance at most `β` reach
`G` of the probes, for some tree of at most `nQ + 2` leaves grown by splits on a letter and a
midfix where two leaves part, with probes of length at most `L` and no length-`k` prefix above
`p₀`. -/
noncomputable def noiseRisk (nQ nα L : ℕ) (β p₀ G : ℝ) : ℝ :=
  ((1 + (nQ + 1) ^ 3 * nα) ^ nQ : ℕ) * Real.exp (-(G - 2 * β * (L + 1) * (nQ + 1)) / p₀)

/-- What one stretch risks over its first `Ns` probes, with good read-states' undecided reads
at most `G` of the probes: a test harvesting good strings at one of its looks, or a false
success; and the stretch outliving `Ns` probes, though a record recurs at least `θr` of the
time, or the searches are at most `εd'` of the probes yet success has not fired at look `N₁`, or
they are more and the undecided middles at least `θpt'` of them yet their test has not fired. -/
noncomputable def stretchRisk (C : Cfg) (L Ns N₁ : ℕ) (G θr εd' θpt' : ℝ) : ℝ :=
  ∑ n ∈ Finset.Icc C.n₀ Ns, (binomSfGe n G ((hFire C.θs C.a n + 1) / 2) + C.a)
    + ∑ n ∈ Finset.Icc 1 Ns,
      (binomSfGe n G ⌈(C.exc + C.θe * C.τe * n) / (2 * (L + 1))⌉₊
        + binomSfGe n G ((hFire C.θpt C.a (max C.n₀ ⌈C.qg * n⌉₊) + 1) / 2))
    + (1 - binomSfGe Ns θr C.m)
    + binE N₁ εd' (fun s =>
        if C.a ≤ 1 - binomSfGe N₁ C.εd (s + 1) ∨ binomSfGe N₁ C.εd s < C.a then 1 else 0)
    + (1 - binomSfGe Ns εd' (max C.n₀ ⌈C.qg * Ns⌉₊))
    + ∑ s ∈ Finset.Icc (max C.n₀ ⌈C.qg * Ns⌉₊) Ns,
        binE s θpt' (fun j => if C.a ≤ binomSfGe s C.θpt j then 1 else 0)

/-- The reads are independent across strings, each undecided with its state's chance `U` and
never on its state's wrong side; the probes are `P` draws from `D`, of length at most `L`, with
no length-`k` prefix above `p₀`. With at least `stretches · Ns` probes, the round fails to end
well, a state being bad where it is undecided more than `1.5θ` of the time, with chance at most
`noiseRisk + stretches · stretchRisk`. -/
def RandomRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (θ : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (C : Cfg) (L P Ns N₁ : ℕ) (p₀ ε G θr εd' θpt' : ℝ),
    (∀ z, Measurable (read z)) → ProbabilityTheory.iIndepFun read μ →
    (∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)) →
    (∀ z, μ {ω | read z ω = if side (M.eval z.toList) then .reject else .accept} = 0) →
    (∀ᵐ x ∂D, x.toList.length ≤ L) → 0 < p₀ →
    (∀ u : FreeMonoid α, u.toList.length = C.k → D.real {x | pre x C.k = u} ≤ p₀) →
    Fintype.card σ + 2 ≤ C.Lmax → 1 ≤ C.m → 1 ≤ C.n₀ → C.n₀ ≤ N₁ → N₁ ≤ Ns →
    0 ≤ θ → 0 ≤ G → G ≤ 1 → 0 ≤ C.a → 0 ≤ C.θs → C.θs ≤ 1 → 0 ≤ C.θe → 0 ≤ C.θpt → C.θpt ≤ 1 →
    0 ≤ C.εd → C.εd ≤ 1 → C.εd ≤ ε → 0 ≤ θr → θr ≤ 1 → 0 ≤ εd' → εd' ≤ 1 → 0 ≤ θpt' →
    θpt' ≤ 1 → (C.Lmax : ℝ) ^ 2 * Fintype.card α * θr ≤ (1 - θpt') * εd' →
    stretches C (Fintype.card α) * Ns ≤ P →
    ∫⁻ ω, (Measure.pi fun _ : Fin P => D)
        {xs | let r := round (read · ω) C start (List.ofFn xs)
          ¬ EndsWell M (BadAt U θ) D ε {x | Disagrees (read · ω) r.1.tree r.1.edges C.k x}
            (toRoundEnd r.2)} ∂μ
      ≤ ENNReal.ofReal (noiseRisk (Fintype.card σ) (Fintype.card α) L (3 / 2 * θ) p₀ G
        + stretches C (Fintype.card α) * stretchRisk C L Ns N₁ G θr εd' θpt')

end Random

end OrthoDFA
