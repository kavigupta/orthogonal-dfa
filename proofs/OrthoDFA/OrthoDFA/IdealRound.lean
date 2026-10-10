import OrthoDFA.FamilyRead

/-!
# A round with ideal reads

Every read of a string is fixed by its target state: undecided for all of the state's strings, or
decided on the state's side for all of them.

The round keeps the real loop's shape. Each probe is walked from `k` along the learned edges and
sifted whole, and where the two part a binary search finds the boundary. A clean disagreement
records its edge and the leaf the next prefix sifts to; `m` records at one target redirect the
edge, or, once it has been redirected, split its leaf on the letter and the midfix where the two
targets part. A probe at an unlearned edge learns it. Undecided reads are charged to the start,
the end, or an edge, and `h` strings of one class end the round with them as its harvest; `n`
probes in a row with no disagreement end it consistent. Every change of the hypothesis starts the
counts afresh.

`IdealRoundCorrect`: the round never stops short; a harvest is all undecided states; every leaf
past the first two is reached, so there are at most `|Q| + 2`; and it ends consistent with the
hypothesis disagreeing on more than `ε` of the probes with chance at most `(P + 1)(1 − ε)^n`.
-/

namespace OrthoDFA

namespace Ideal

open MeasureTheory

variable {α : Type*}

/-- A tree whose nodes read the string followed by their midfix. -/
inductive DTree (α : Type*)
  | leaf
  | node (m : FreeMonoid α) (rej acc : DTree α)

namespace DTree

/-- The path to the leaf `z` sifts to, or the string whose read was undecided. -/
def sift (read : FreeMonoid α → ARU) : DTree α → FreeMonoid α → List Bool ⊕ FreeMonoid α
  | leaf, _ => .inl []
  | node m r a, z =>
    match read (z * m) with
    | .accept => (a.sift read z).map (true :: ·) id
    | .reject => (r.sift read z).map (false :: ·) id
    | .undecided => .inr (z * m)

def leaves : DTree α → List (List Bool)
  | leaf => [[]]
  | node _ r a => r.leaves.map (false :: ·) ++ a.leaves.map (true :: ·)

/-- The leaf at `p` replaced by a node reading at `d`. -/
def splitAt (d : FreeMonoid α) : DTree α → List Bool → DTree α
  | leaf, [] => node d leaf leaf
  | node m r a, false :: p => node m (r.splitAt d p) a
  | node m r a, true :: p => node m r (a.splitAt d p)
  | t, _ => t

def midAt : DTree α → List Bool → FreeMonoid α
  | leaf, _ => 1
  | node m _ _, [] => m
  | node _ r _, false :: p => r.midAt p
  | node _ _ a, true :: p => a.midAt p

end DTree

def lcp : List Bool → List Bool → List Bool
  | a :: as, b :: bs => if a = b then a :: lcp as bs else []
  | _, _ => []

/-- Each learned edge's target, with a string at its leaf whose successor sifts there. -/
abbrev Edges (α : Type*) := List Bool → α → Option (List Bool × FreeMonoid α)

/-- The leaves the edges visit from `s` along `cs`, `s` first; it stops at an unlearned edge. -/
def walk (E : Edges α) : List Bool → List α → List (List Bool)
  | s, [] => [s]
  | s, c :: cs => s :: match E s c with
    | some (t, _) => walk E t cs
    | none => []

def run (E : Edges α) : List Bool → List α → Option (List Bool)
  | s, [] => some s
  | s, c :: cs => (E s c).bind fun e => run E e.1 cs

def pre (x : FreeMonoid α) (i : ℕ) : FreeMonoid α := FreeMonoid.ofList (x.toList.take i)

inductive Found
  | edge (p : ℕ)
  | triple (p : ℕ)
  | pair (p : ℕ)

/-- Bisection between `lo`, where the sift agrees with the walk, and `hi`, where it disagrees,
on the decided reads of `ag`: an edge `p` agrees at `p - 1` and disagrees at `p`, a triple is
undecided at `p` between the two, a pair undecided at `p` and `p + 1`. -/
def search (ag : ℕ → Option Bool) (lo hi : ℕ) : Found :=
  if _h : hi ≤ lo + 1 then .edge hi
  else
    match ag ((lo + hi) / 2) with
    | some true => search ag ((lo + hi) / 2) hi
    | some false => search ag lo ((lo + hi) / 2)
    | none =>
      match ag ((lo + hi) / 2 - 1), ag ((lo + hi) / 2 + 1) with
      | none, _ => .pair ((lo + hi) / 2 - 1)
      | _, none => .pair ((lo + hi) / 2)
      | some true, some false => .triple ((lo + hi) / 2)
      | some false, _ => search ag lo ((lo + hi) / 2 - 1)
      | some true, some true => search ag ((lo + hi) / 2 + 1) hi
termination_by hi - lo
decreasing_by all_goals omega

/-- What undecided reads are charged to. -/
inductive Cls (α : Type*)
  | start
  | stop
  | cell (s : List Bool) (c : α)
  deriving DecidableEq

/-- How a probe ends. `edge s c u t`: `u` sifts to `s` and `u·c` to `t`, which the edge `(s, c)`
does not point at; `member` is the same at an unlearned edge. -/
inductive Outcome (α : Type*)
  | agree
  | startU (z : FreeMonoid α)
  | endU (z : FreeMonoid α)
  | edge (s : List Bool) (c : α) (u : FreeMonoid α) (t : List Bool)
  | triple (k : Cls α) (zs : List (FreeMonoid α))
  | pair (k : Cls α) (zs : List (FreeMonoid α))
  | member (s : List Bool) (c : α) (u : FreeMonoid α) (t : List Bool)

section Probe

variable (read : FreeMonoid α → ARU) (T : DTree α) (x : FreeMonoid α) (st : ℕ → List Bool)

def agrees (p : ℕ) : Option Bool :=
  (T.sift read (pre x p)).elim (fun l => some (decide (l = st p))) fun _ => none

def stuckAt (p : ℕ) : List (FreeMonoid α) :=
  (T.sift read (pre x p)).elim (fun _ => []) fun z => [z]

def cellAt (p : ℕ) : Cls α :=
  match x.toList[p]? with
  | some c => .cell (st p) c
  | none => .stop

/-- The search between `lo` and `hi`, against the walk's leaves `st`. -/
def located (lo hi : ℕ) : Outcome α :=
  match search (agrees read T x st) lo hi with
  | .edge p =>
    match x.toList[p - 1]?, T.sift read (pre x p) with
    | some c, .inl t => .edge (st (p - 1)) c (pre x (p - 1)) t
    | _, _ => .pair .stop []
  | .triple p => .triple (cellAt x st p) (stuckAt read T x p)
  | .pair p => .pair (cellAt x st p) (stuckAt read T x p ++ stuckAt read T x (p + 1))

end Probe

/-- A probe of `x`: its first `k` letters sifted, then walked along the edges and sifted whole. -/
def probe (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α) :
    Outcome α :=
  match T.sift read (pre x k) with
  | .inr z => .startU z
  | .inl a =>
    let ss := walk E a (x.toList.drop k)
    let st := fun p => ss.getD (p - k) []
    let j := k + ss.length - 1
    match x.toList[j]? with
    | some c =>
      match T.sift read (pre x (j + 1)), T.sift read (pre x j) with
      | .inr z, _ => .endU z
      | _, .inr z => .endU z
      | .inl t, .inl s => if s = st j then .member s c (pre x j) t else located read T x st k j
    | none =>
      match T.sift read x with
      | .inr z => .endU z
      | .inl e => if e = st j then .agree else located read T x st k j

/-- The round's settings: the walk's start, the records that move an edge, the undecided strings
that harvest a class, the clean probes that end it consistent, and the most leaves. -/
structure RoundCfg where
  k : ℕ
  m : ℕ
  h : ℕ
  n : ℕ
  Lmax : ℕ

/-- The hypothesis, whether each edge has been redirected since it was learned, and the counts
since the hypothesis last changed: records by edge and target, undecided strings by class, and
probes since the last disagreement. -/
structure RState (α : Type*) where
  tree : DTree α
  edges : Edges α
  moved : List Bool → α → Bool
  recs : List (List Bool × α × List Bool)
  und : Cls α → List (FreeMonoid α)
  clean : ℕ

/-- How the round ends: (a) consistent, (b) a harvest, or (c) flagged, the tree too big. Running
out of probes is (c) too. -/
inductive REnd (α : Type*)
  | consistent
  | harvest (zs : List (FreeMonoid α))
  | tooBig

variable [DecidableEq α]

def upd2 {β γ δ : Type*} [DecidableEq β] [DecidableEq γ] (f : β → γ → δ) (b : β) (c : γ)
    (v : δ) : β → γ → δ :=
  Function.update f b (Function.update (f b) c v)

def fresh (T : DTree α) (E : Edges α) (moved : List Bool → α → Bool) : RState α :=
  ⟨T, E, moved, [], fun _ => [], 0⟩

/-- The root reads at `ε`; no edge is learned. -/
def start : RState α := fresh (.node 1 .leaf .leaf) (fun _ _ => none) fun _ _ => false

/-- The edges after the leaf `p` splits into `T'`: those into `p` sift their witness's successor
again, those out of it are dropped. -/
def retarget (read : FreeMonoid α → ARU) (T' : DTree α) (p : List Bool) (E : Edges α) :
    Edges α := fun q c =>
  if p.isPrefixOf q then none
  else
    match E q c with
    | some (t, w) =>
      if t = p then (T'.sift read (w * FreeMonoid.of c)).elim (fun t' => some (t', w)) fun _ => none
      else some (t, w)
    | none => none

variable (read : FreeMonoid α → ARU) (C : RoundCfg)

def tick (s : RState α) : RState α ⊕ REnd α :=
  if C.n ≤ s.clean + 1 then .inr .consistent else .inl { s with clean := s.clean + 1 }

/-- Undecided strings `zs` charged to `k`, from a probe that disagreed unless `clean`. -/
def charge (s : RState α) (k : Cls α) (zs : List (FreeMonoid α)) (clean : Bool) :
    RState α ⊕ REnd α :=
  let s' := { s with und := Function.update s.und k (s.und k ++ zs) }
  if C.h ≤ (s'.und k).length then .inr (.harvest (s'.und k))
  else if clean then tick C s' else .inl { s' with clean := 0 }

def split (s : RState α) (p : List Bool) (d : FreeMonoid α) : RState α ⊕ REnd α :=
  let T' := s.tree.splitAt d p
  if C.Lmax < T'.leaves.length then .inr .tooBig
  else .inl (fresh T' (retarget read T' p s.edges) fun _ _ => false)

/-- A clean disagreement at the edge `(p, c)`, with `u` at `p` and `u·c` at `t`. -/
def record (s : RState α) (p : List Bool) (c : α) (u : FreeMonoid α) (t : List Bool) :
    RState α ⊕ REnd α :=
  let s' := { s with recs := s.recs ++ [(p, c, t)], clean := 0 }
  if C.m ≤ s'.recs.count (p, c, t) then
    match s.edges p c with
    | some (t₀, _) =>
      if s.moved p c then split read C s p (FreeMonoid.of c * s.tree.midAt (lcp t t₀))
      else .inl (fresh s.tree (upd2 s.edges p c (some (t, u))) (upd2 s.moved p c true))
    | none => .inl s'
  else .inl s'

def step (s : RState α) (x : FreeMonoid α) : RState α ⊕ REnd α :=
  match probe read s.tree s.edges C.k x with
  | .agree => tick C s
  | .startU z => charge C s .start [z] true
  | .endU z => charge C s .stop [z] true
  | .triple k zs => charge C s k zs false
  | .pair k zs => charge C s k zs false
  | .edge p c u t => record read C s p c u t
  | .member p c u t =>
    .inl (fresh s.tree (upd2 s.edges p c (some (t, u))) (upd2 s.moved p c false))

/-- The round over the probes; the state it ended in, with `none` where the probes ran out. -/
def round : RState α → List (FreeMonoid α) → RState α × Option (REnd α)
  | s, [] => (s, none)
  | s, x :: xs =>
    match step read C s x with
    | .inl s' => round s' xs
    | .inr e => (s, some e)

/-! ## The claim -/

/-- Every target state's strings all read undecided, or all read its side. -/
def IdealReads {σ : Type*} (M : DFA α σ) (side : σ → Bool) : Prop :=
  ∀ q, (∀ w : FreeMonoid α, M.eval w.toList = q → read w = .undecided) ∨
    ∀ w : FreeMonoid α, M.eval w.toList = q → read w = if side q then .accept else .reject

/-- The edges, run from the leaf `x`'s first `k` letters sift to, end at a leaf other than the
one `x` sifts to. -/
def Disagrees (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α) : Prop :=
  ∃ a q e, T.sift read (pre x k) = .inl a ∧ run E a (x.toList.drop k) = some q ∧
    T.sift read x = .inl e ∧ q ≠ e

/-- Enough probes for the round to end. -/
def budget (C : RoundCfg) (nα : ℕ) : ℕ :=
  (C.Lmax * (2 * C.Lmax * nα + 1) + 1)
    * ((C.m - 1) * C.Lmax ^ 2 * nα + (C.h - 1) * (C.Lmax * nα + 2) + 1) * C.n

/-- The round ends consistent or with a nonempty harvest of undecided states, and every leaf
past the first two is reached by some string. -/
def Sound {σ : Type*} [Fintype σ] (M : DFA α σ) (r : RState α × Option (REnd α)) : Prop :=
  (r.2 = some .consistent ∨ ∃ zs, r.2 = some (.harvest zs) ∧ zs ≠ [] ∧
      ∀ z ∈ zs, ∀ w : FreeMonoid α, M.eval w.toList = M.eval z.toList → read w = .undecided)
  ∧ (∀ p ∈ r.1.tree.leaves, p = [false] ∨ p = [true] ∨ ∃ w, r.1.tree.sift read w = .inl p)
  ∧ r.1.tree.leaves.length ≤ Fintype.card σ + 2

/-- With ideal reads and enough probes, every run of the round is sound, and it ends consistent
with the hypothesis disagreeing on more than `ε` of the draws with chance at most
`(P + 1)(1 − ε)^n`. -/
def IdealRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (read : FreeMonoid α → ARU) (side : σ → Bool) (C : RoundCfg) (P : ℕ),
    IdealReads read M side → 1 ≤ C.m → 1 ≤ C.h → 1 ≤ C.n → Fintype.card σ + 2 ≤ C.Lmax →
    budget C (Fintype.card α) ≤ P →
    (∀ xs : Fin P → FreeMonoid α, Sound read M (round read C start (List.ofFn xs)))
    ∧ ∀ (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (ε : ℝ), ε ≤ 1 →
      (Measure.pi fun _ : Fin P => D).real
          {xs | (round read C start (List.ofFn xs)).2 = some .consistent
            ∧ ε < D.real {x | Disagrees read (round read C start (List.ofFn xs)).1.tree
                (round read C start (List.ofFn xs)).1.edges C.k x}}
        ≤ (P + 1) * (1 - ε) ^ C.n

end Ideal

end OrthoDFA
