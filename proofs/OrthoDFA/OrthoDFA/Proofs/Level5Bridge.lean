import OrthoDFA.Proofs.Level5Firsts
import OrthoDFA.Proofs.Level5Correct
import OrthoDFA.Proofs.Probe

/-!
# Level 5 on main's probe

Level 5 written over main's probe (`probeOrAgree`, `walk`, `Edges`, `retarget`, `Outcome`), and
the proof that it is level 5: the same hypothesis and ending on every draw
(`indecisiveLStarLevel5_new`).
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace New

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

section ProbeQ

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

def freshState (T : DTree α) (E : Edges α) (v : ℕ) : Level5State α :=
  ⟨T, E, v, 0, [], fun _ _ => 0, fun _ _ => [], 0, []⟩

def Level5State.setEdge (s : Level5State α) (p : List Bool) (c : α)
    (e : List Bool × FreeMonoid α) : Level5State α :=
  freshState s.tree (Function.update s.edges (p, c) (some e)) (s.version + 1)

def Level5State.countProbe (cut : FreeMonoid α → Option Bool) (k : ℕ) (s : Level5State α)
    (x : FreeMonoid α) : Level5State α :=
  { s with
    probes := s.probes + 1
    startUndecided := s.startUndecided ++ startHarvestBy cut s.tree k x
    edgeReads := fun p c => s.edgeReads p c + edgeReadsBy cut s.tree s.edges k x (p, c)
    edgeUndecided := fun p c =>
      s.edgeUndecided p c ++ edgeHarvestBy cut s.tree s.edges k x (p, c)
    middles := s.middles ++ middleHarvestBy cut s.tree s.edges k x }

def initialState : Level5State α := freshState (.node 1 .leaf .leaf) (fun _ => none) 0

def splitState (cut : FreeMonoid α → Option Bool) (s : Level5State α) (p : List Bool) (c : α)
    (t : List Bool) : Level5State α :=
  match s.edges (p, c) with
  | some (t₀, _) =>
    let H := HypothesisDFA.split (ofCut cut) ⟨s.tree, s.edges⟩
      (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p
    freshState H.tree H.edges (s.version + 1)
  | none => s

def recordProbe (C : Level5Settings) (cut : FreeMonoid α → Option Bool)
    (s : Level5State α) (x : FreeMonoid α) : Level5State α :=
  let s₁ := s.countProbe cut C.walkStart x
  match probeOrAgree (ofCut cut) s.tree s.edges C.walkStart x with
  | .unlearnedEdge p c u t => s₁.setEdge p c (t, u)
  | .wrongEdge _ _ _ _ | .undecided _ false => { s₁ with searches := s₁.searches + 1 }
  | _ => s₁

open scoped Classical in
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

def siteHarvest (cut : FreeMonoid α → Option Bool) (k : ℕ) :
    Level5Site α → DTree α × Edges α → FreeMonoid α → List (FreeMonoid α)
  | .start, v, x => startHarvestBy cut v.1 k x
  | .edge e, v, x => edgeHarvestBy cut v.1 v.2 k x e
  | .middles, v, x => middleHarvestBy cut v.1 v.2 k x

open MeasureTheory in
noncomputable def harvestLaw (D : Measure (FreeMonoid α)) (cut : FreeMonoid α → Option Bool)
    (k : ℕ) (H : HypothesisDFA α) (site : Level5Site α) : Measure (FreeMonoid α) :=
  D.bind fun x => ((siteHarvest cut k site (H.tree, H.edges) x).map Measure.dirac).sum

noncomputable def runProbes (C : Level5Settings) (cut : FreeMonoid α → Option Bool) :
    Level5State α → List (FreeMonoid α) → Level5State α × RoundEnd (Level5Site α)
  | s, [] => (s, .failed)
  | s, x :: xs =>
    match applyProbe C cut s x with
    | .inl s' => runProbes C cut s' xs
    | .inr (e, s') => (s', toRoundEnd e)

noncomputable def indecisiveLStarLevel5 (settings : Level5Settings) (read : FreeMonoid α → ARU)
    {P : ℕ} (draws : Fin P → FreeMonoid α) : Level5Result α :=
  let r := runProbes settings (fun z => (read z).cut) initialState (List.ofFn draws)
  ⟨⟨r.1.tree, r.1.edges⟩, r.2⟩

/-- The probes that search. -/
def searchingProbes (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α) :
    Set (FreeMonoid α) :=
  {x | match probeOrAgree read H.tree H.edges k x with
    | .wrongEdge _ _ _ _ | .undecided _ false => True
    | _ => False}

end New

/-! ## The query tree's run is main's probe -/

section Run

theorem run_siftQuery_sift (ℓ : ReadLabel α) (rd : FreeMonoid α → ARU) :
    ∀ (T : DTree α) (u : FreeMonoid α), (siftQuery ℓ T u).run rd = T.sift rd u
  | .leaf, _ => by simp [siftQuery, QTree.run, DTree.sift]
  | .node m r a, u => by
    simp only [siftQuery, QTree.run, DTree.sift]
    rcases h : rd (u * m) with _ | _ | _ <;> simp only [ARU.cut]
    · rw [QTree.run_map, run_siftQuery_sift ℓ rd a u]
    · rw [QTree.run_map, run_siftQuery_sift ℓ rd r u]
    · rfl

variable (rd : FreeMonoid α → ARU) (T : DTree α) (x : FreeMonoid α) (k : ℕ)
  (lab : ℕ → ReadLabel α) (st : ℕ → List Bool)

theorem run_bracketQuery_new (un : Option (α × List Bool)) (hi₀ : ℕ) :
    ∀ fuel lo hi t, hi - lo ≤ fuel → T.sift rd (prefixOf x hi) = .inl t →
      (New.bracketQuery T x lab st fuel lo hi t).run rd
        = searchOutcome rd T x st un hi₀ (bisectBoundary false (siftAgrees rd T x st) lo hi).1
  | 0, lo, hi, t, hf, ht => by
    rw [bisectBoundary, dif_pos (by omega)]
    simp only [New.bracketQuery, QTree.run, searchOutcome, New.edgeOutcome, ht]
    rcases x.toList[hi - 1]? with _ | c <;> rfl
  | fuel + 1, lo, hi, t, hf, ht => by
    simp only [New.bracketQuery]
    by_cases h1 : lo + 1 < hi
    · rw [if_pos h1, bisectBoundary, dif_neg (by omega), QTree.run_bind, run_siftQuery_sift]
      rcases hm : T.sift rd (prefixOf x ((lo + hi) / 2)) with q | z
      · rw [agrees_inl rd T x st hm]
        simp only
        by_cases hq : q = st ((lo + hi) / 2)
        · rw [if_pos hq, decide_eq_true hq]
          exact run_bracketQuery_new un hi₀ fuel _ hi t (by omega) ht
        · rw [if_neg hq, decide_eq_false hq]
          exact run_bracketQuery_new un hi₀ fuel lo _ q (by omega) hm
      · have : siftAgrees rd T x st ((lo + hi) / 2) = none := by simp [siftAgrees, hm]
        simp only [this, Bool.false_eq_true, ↓reduceIte, QTree.run, searchOutcome, undecidedAt,
          hm, Sum.elim_inr]
    · rw [if_neg h1, bisectBoundary, dif_pos (by omega)]
      simp only [QTree.run, searchOutcome, New.edgeOutcome, ht]
      rcases x.toList[hi - 1]? with _ | c <;> rfl

theorem run_searchQuery_new (un : Option (α × List Bool)) (hi : ℕ) (hk : k ≤ hi)
    (hst : T.sift rd (prefixOf x k) = .inl (st k)) :
    (New.searchQuery T k x lab st un hi).run rd
      = searchOutcome rd T x st un hi (tripartiteSearch false (siftAgrees rd T x st) k hi).1 := by
  have hlo : siftAgrees rd T x st k = some true := by
    rw [agrees_inl rd T x st hst]; simp
  unfold New.searchQuery
  rw [QTree.run_bind, run_siftQuery_sift]
  simp only [tripartiteSearch, hlo]
  rcases ha : T.sift rd (prefixOf x hi) with a | z
  · rw [agrees_inl rd T x st ha]
    simp only
    by_cases hq : a = st hi
    · rw [if_pos hq, decide_eq_true hq]
      simp only [QTree.run, searchOutcome]
      rcases un with _ | ⟨c, t⟩ <;> rfl
    · rw [if_neg hq, decide_eq_false hq]
      exact run_bracketQuery_new rd T x lab st un hi _ k hi a le_rfl ha
  · have : siftAgrees rd T x st hi = none := by simp [siftAgrees, ha]
    simp only [this, QTree.run, searchOutcome, undecidedAt, ha]
    rfl

theorem run_probeQuery_new (E : Edges α) :
    (New.probeQuery T E k x).run rd = probeOrAgree rd T E k x := by
  unfold New.probeQuery probeOrAgree
  by_cases hshort : x.toList.length < k
  · simp only [if_pos hshort, QTree.run]
  rw [if_neg hshort, if_neg hshort, QTree.run_bind, run_siftQuery_sift]
  unfold probe
  rcases hk : T.sift rd (prefixOf x k) with p₀ | z
  · simp only
    have hpos := length_walk_pos E p₀ (x.toList.drop k)
    have hst : T.sift rd (prefixOf x k)
        = .inl ((walk E p₀ (x.toList.drop k)).getD (k - k) []) := by
      rw [Nat.sub_self, walk_getD_zero, hk]
    rcases hc : x.toList[k + (walk E p₀ (x.toList.drop k)).length - 1]? with _ | c
    · exact run_searchQuery_new rd T x k _ _ _ _ (by omega) hst
    · simp only
      rw [QTree.run_bind, run_siftQuery_sift]
      rcases T.sift rd (prefixOf x (k + (walk E p₀ (x.toList.drop k)).length - 1 + 1)) with t | z
      · exact run_searchQuery_new rd T x k _ _ _ _ (by omega) hst
      · rfl
  · rfl

end Run

/-! ## Level 5's probe is main's -/

section Old

omit [Fintype α] [DecidableEq α] in
theorem sift_ofCut (cut : FreeMonoid α → Option Bool) :
    ∀ (T : DTree α) (z : FreeMonoid α), T.sift (ofCut cut) z = T.siftCut cut z
  | .leaf, _ => rfl
  | .node m r a, z => by
    simp only [DTree.sift, DTree.siftCut, DTree.route, ofCut]
    rcases h : cut (z * m) with _ | _ | _
    · rfl
    · simp only [sift_ofCut cut r z]; rfl
    · simp only [sift_ofCut cut a z]; rfl

omit [Fintype α] [DecidableEq α] in
theorem follow_walk_inl {E : EdgeTable α} : ∀ (cs : List α) (p : List Bool)
    (ps : List (List Bool)), follow E p cs = .inl ps → walk (Function.uncurry E) p cs = ps
  | [], p, ps, h => by
    simp only [follow, Sum.inl.injEq] at h
    subst h; rfl
  | c :: cs, p, ps, h => by
    simp only [follow] at h
    rcases he : E p c with _ | ⟨q, y⟩ <;> rw [he] at h
    · simp at h
    · simp only at h
      rcases hf : follow E q cs with ps' | r <;> rw [hf] at h
      · simp only [Sum.inl.injEq] at h
        subst h
        simp [walk, Function.uncurry_apply_pair, he, follow_walk_inl cs q ps' hf]
      · simp at h

omit [Fintype α] [DecidableEq α] in
theorem follow_walk_inr {E : EdgeTable α} : ∀ (cs : List α) (p s : List Bool) (c : α) (i : ℕ),
    follow E p cs = .inr (s, c, i) →
      follow E p (cs.take i) = .inl (walk (Function.uncurry E) p cs)
        ∧ (walk (Function.uncurry E) p cs).length = i + 1
        ∧ (walk (Function.uncurry E) p cs).getD i [] = s
  | [], p, s, c, i, h => by simp [follow] at h
  | c' :: cs, p, s, c, i, h => by
    simp only [follow] at h
    rcases he : E p c' with _ | ⟨q, y⟩ <;> rw [he] at h
    · simp only [Sum.inr.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      simp [follow, walk, Function.uncurry_apply_pair, he]
    · simp only at h
      rcases hf : follow E q cs with ps' | ⟨s', c'', i'⟩ <;> rw [hf] at h
      · simp at h
      · simp only [Sum.inr.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl, rfl⟩ := h
        obtain ⟨h1, h2, h3⟩ := follow_walk_inr cs q s' c'' i' hf
        have hw : walk (Function.uncurry E) p (c' :: cs) = p :: walk (Function.uncurry E) q cs := by
          simp [walk, Function.uncurry_apply_pair, he]
        rw [hw]
        refine ⟨?_, by simp [h2], by simpa using h3⟩
        simp [follow, he, h1]

omit [Fintype α] [DecidableEq α] in
theorem follow_take_last {E : EdgeTable α} : ∀ (cs : List α) (p : List Bool) (n : ℕ),
    n ≤ cs.length → ((follow E p (cs.take n)).elim id fun _ => []).getLast?
      = (walk (Function.uncurry E) p cs)[n]?
  | [], p, n, hn => by
    obtain rfl : n = 0 := by simpa using hn
    simp [follow, walk]
  | c :: cs, p, 0, _ => by
    obtain ⟨l, hl⟩ := walk_cons (Function.uncurry E) p (c :: cs)
    simp [follow, hl]
  | c :: cs, p, n + 1, hn => by
    simp only [List.take_succ_cons, follow]
    rcases he : E p c with _ | ⟨q, y⟩
    · simp [walk, Function.uncurry_apply_pair, he]
    · have hw : walk (Function.uncurry E) p (c :: cs) = p :: walk (Function.uncurry E) q cs := by
        simp [walk, Function.uncurry_apply_pair, he]
      rw [hw, List.getElem?_cons_succ, ← follow_take_last cs q n (by simpa using hn)]
      simp only
      rcases hf : follow E q (cs.take n) with ps | r
      · obtain ⟨hl, -⟩ := follow_inl _ _ _ hf
        have hne : ps ≠ [] := by intro h; subst h; simp at hl
        simp [List.getLast?_cons, hne, List.getLast?_eq_getLast hne]
      · simp

theorem posEdgeFrom_new (E : EdgeTable α) (k : ℕ) (x : FreeMonoid α) (p : List Bool) (i : ℕ) :
    New.posEdgeFrom (Function.uncurry E) k x p i = posEdgeFrom E k x p i := by
  have key : ∀ j, x.toList[j]? = none ∨ (walkFrom E k x p j).getLast?
      = (walk (Function.uncurry E) p (x.toList.drop k))[j - k]? := by
    intro j
    rcases hj : x.toList[j]? with _ | c
    · exact .inl rfl
    right
    have hlt : j < x.toList.length := (List.getElem?_eq_some_iff.1 hj).1
    simp only [walkFrom]
    rw [follow_take_last _ _ _ (by simp; omega)]
  simp only [New.posEdgeFrom, posEdgeFrom]
  rcases key i with h | h <;> rcases key (i - 1) with h' | h' <;> simp only [h, h']
  all_goals
    generalize (walkFrom E k x p i).getLast? = a
    generalize (walkFrom E k x p (i - 1)).getLast? = b
    generalize (walk (Function.uncurry E) p (x.toList.drop k))[i - k]? = a'
    generalize (walk (Function.uncurry E) p (x.toList.drop k))[i - 1 - k]? = b'
    generalize x.toList[i]? = ci
    generalize x.toList[i - 1]? = cj
    rcases a with _ | a <;> rcases b with _ | b <;> rcases a' with _ | a' <;>
      rcases b' with _ | b' <;> rcases ci with _ | ci <;> rcases cj with _ | cj <;> rfl

theorem readLabel_new (E : EdgeTable α) (k : ℕ) (x : FreeMonoid α) (p : List Bool) :
    New.readLabel (Function.uncurry E) k x p = readLabel E k x p := by
  funext i; simp only [New.readLabel, readLabel, posEdgeFrom_new]

omit [Fintype α] [DecidableEq α] in
theorem append_and {γ δ : Type*} {l a b : List γ} {r r' : δ} (h : a = b ∧ r = r') :
    l ++ a = l ++ b ∧ r = r' := ⟨by rw [h.1], h.2⟩

/-- Level 5's outcome as main's. -/
def conv (read : FreeMonoid α → ARU) (T : DTree α) (E : EdgeTable α) (k : ℕ) (x : FreeMonoid α) :
    WalkOutcome α → Outcome α
  | .agree => .agree
  | .startUndecided w => (T.sift read w).elim (fun _ => .agree) .undecidedStart
  | .endUndecided w => (T.sift read w).elim (fun _ => .agree) fun z => .undecided [z] true
  | .undecidedMid j => .undecided (undecidedAt read T x j) false
  | .edge ps fd =>
    match x.toList[fd - 1]?, T.sift read (prefixOf x fd) with
    | some c, .inl t => .wrongEdge (ps.getD (fd - 1 - k) []) c (prefixOf x (fd - 1)) t
    | _, _ => .undecided [] false
  | .member u =>
    match x.toList[u.toList.length]?, T.sift read u with
    | some c, .inl p =>
      match T.sift read (u * FreeMonoid.of c), E p c with
      | .inl t, none => .unlearnedEdge p c u t
      | _, _ => .agree
    | _, _ => .agree

variable (rd : FreeMonoid α → ARU) (T : DTree α) (E : EdgeTable α) (k : ℕ) (x : FreeMonoid α)

theorem bracketQuery_new (p₀ : List Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi t, T.sift rd (prefixOf x hi) = .inl t →
      (New.bracketQuery T x (readLabel E k x p₀)
          (fun j => ps.getD (j - k) []) fuel lo hi t).asks rd
        = (bracketQuery (agreementQuery T E k x p₀ fun j => ps.getD (j - k) []) ps fuel lo hi).asks rd
      ∧ (New.bracketQuery T x (readLabel E k x p₀)
          (fun j => ps.getD (j - k) []) fuel lo hi t).run rd
        = conv rd T E k x
          ((bracketQuery (agreementQuery T E k x p₀ fun j => ps.getD (j - k) []) ps fuel lo hi).run rd)
  | 0, lo, hi, t, ht => by
    simp only [New.bracketQuery, bracketQuery, QTree.asks, QTree.run, conv, New.edgeOutcome, ht,
      true_and]
    rcases x.toList[hi - 1]? with _ | c <;> rfl
  | fuel + 1, lo, hi, t, ht => by
    simp only [New.bracketQuery, bracketQuery]
    by_cases h1 : lo + 1 < hi
    · rw [if_pos h1, if_pos h1, QTree.asks_bind, QTree.asks_bind, QTree.run_bind, QTree.run_bind,
        agreementQuery, QTree.asks_map, QTree.run_map, run_siftQuery_sift]
      rcases hm : T.sift rd (prefixOf x ((lo + hi) / 2)) with q | z
      · simp only [Sum.elim_inl]
        by_cases hq : q = (ps.getD ((lo + hi) / 2 - k) [])
        · rw [if_pos hq, decide_eq_true hq]
          obtain ⟨h1, h2⟩ := bracketQuery_new p₀ ps fuel _ hi t ht
          simp only
          exact ⟨by rw [h1], h2⟩
        · rw [if_neg hq, decide_eq_false hq]
          obtain ⟨h1, h2⟩ := bracketQuery_new p₀ ps fuel lo _ q hm
          simp only
          exact ⟨by rw [h1], h2⟩
      · simp only [Sum.elim_inr, QTree.asks, QTree.run, conv, undecidedAt, hm]
        exact ⟨by first | trivial | rfl, by first | trivial | rfl⟩
    · rw [if_neg h1, if_neg h1]
      simp only [QTree.asks, QTree.run, conv, New.edgeOutcome, ht, true_and]
      rcases x.toList[hi - 1]? with _ | c <;> rfl

theorem probeQuery_new :
    (New.probeQuery T (Function.uncurry E) k x).asks rd = (probeQuery T E k x).asks rd
      ∧ (New.probeQuery T (Function.uncurry E) k x).run rd
        = conv rd T E k x ((probeQuery T E k x).run rd) := by
  unfold New.probeQuery probeQuery
  by_cases hshort : x.toList.length < k
  · simp only [if_pos hshort, QTree.asks, QTree.run, conv]; exact ⟨by first | trivial | rfl, by first | trivial | rfl⟩
  rw [if_neg hshort, if_neg hshort, QTree.asks_bind, QTree.asks_bind, QTree.run_bind,
    QTree.run_bind, run_siftQuery_sift]
  rcases hk : T.sift rd (prefixOf x k) with p₀ | z
  swap
  · simp only [QTree.asks, QTree.run, conv]
    rw [hk]
    exact ⟨by first | trivial | rfl, by first | trivial | rfl⟩
  simp only
  refine append_and ?_
  rcases hf : follow E p₀ (x.toList.drop k) with ps | ⟨s, c, i⟩
  · -- the walk reaches the end
    have hw := follow_walk_inl _ _ _ hf
    obtain ⟨hl, -⟩ := follow_inl _ _ _ hf
    simp only [List.length_drop] at hl
    rw [hw, hl]
    have hhi : k + (x.toList.length - k + 1) - 1 = x.toList.length := by omega
    simp only [hhi, List.getElem?_eq_none le_rfl]
    unfold New.searchQuery
    try rw [readLabel_new]
    rw [QTree.asks_bind, QTree.asks_bind, QTree.run_bind, QTree.run_bind, run_siftQuery_sift]
    refine append_and ?_
    rcases hx : T.sift rd (prefixOf x x.toList.length) with a | z
    · simp only
      have hlast : ps.getLast? = some (ps.getD (x.toList.length - k) []) := by
        rw [List.getLast?_eq_getElem?, show ps.length - 1 = x.toList.length - k by omega,
          List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]
        rfl
      by_cases ha : a = ps.getD (x.toList.length - k) []
      · rw [if_pos ha, if_pos (by rw [hlast, ha])]
        exact ⟨by first | trivial | rfl, by first | trivial | rfl⟩
      · rw [if_neg ha, if_neg (by rw [hlast]; exact fun h => ha (Option.some.inj h))]
        exact bracketQuery_new rd T E k x p₀ ps _ _ _ a hx
    · simp only [QTree.asks, QTree.run, conv]
      rw [prefixOf_length] at hx
      rw [hx]
      exact ⟨by first | trivial | rfl, by first | trivial | rfl⟩
  · -- the walk stops at an unlearned edge
    obtain ⟨hi, hc, hn, -⟩ := follow_inr _ _ _ _ _ hf
    obtain ⟨hft, hlen, hget⟩ := follow_walk_inr _ _ _ _ _ hf
    simp only [List.length_drop] at hi
    rw [hlen, show k + (i + 1) - 1 = k + i by omega]
    have hxc : x.toList[k + i]? = some c := by
      rw [← hc, List.getElem_drop]; exact List.getElem?_eq_getElem (by omega)
    simp only [hxc]
    try rw [readLabel_new]
    rw [QTree.asks_bind, QTree.asks_bind, QTree.run_bind, QTree.run_bind, run_siftQuery_sift]
    refine append_and ?_
    rcases h1 : T.sift rd (prefixOf x (k + i + 1)) with t | z
    swap
    · simp only [QTree.asks, QTree.run, conv]
      rw [h1]
      exact ⟨by first | trivial | rfl, by first | trivial | rfl⟩
    simp only
    unfold New.searchQuery
    try rw [readLabel_new]
    rw [QTree.asks_bind, QTree.asks_bind, QTree.run_bind, QTree.run_bind, run_siftQuery_sift]
    refine append_and ?_
    rcases h2 : T.sift rd (prefixOf x (k + i)) with p | z
    swap
    · simp only [QTree.asks, QTree.run, conv]
      rw [h2]
      exact ⟨by first | trivial | rfl, by first | trivial | rfl⟩
    simp only
    have hst : (walk (Function.uncurry E) p₀ (x.toList.drop k)).getD (k + i - k) [] = s := by
      rw [show k + i - k = i by omega, hget]
    rw [hst]
    by_cases hps : p = s
    · rw [if_pos hps, if_pos hps]
      subst hps
      have hlenu : (prefixOf x (k + i)).toList.length = k + i := by simp [prefixOf]; omega
      simp only [QTree.asks, QTree.run, conv, hlenu, hxc, h2, ← prefixOf_succ hxc, h1, hn]
      exact ⟨by first | trivial | rfl, by first | trivial | rfl⟩
    · rw [if_neg hps, if_neg hps]
      have hwf : walkFrom E k x p₀ (k + i) = walk (Function.uncurry E) p₀ (x.toList.drop k) := by
        simp only [walkFrom, show k + i - k = i by omega, hft, Sum.elim_inl, id]
      rw [hwf]
      exact bracketQuery_new rd T E k x p₀ _ _ _ _ p h2

/-- Level 5's probe is main's. -/
theorem probeBy_new (cut : FreeMonoid α → Option Bool) :
    probeOrAgree (ofCut cut) T (Function.uncurry E) k x
      = conv (ofCut cut) T E k x (probeBy cut T E k x) := by
  rw [← run_probeQuery_new, (probeQuery_new (ofCut cut) T E k x).2, run_probeQuery, cut_ofCut]

end Old

/-! ## The algorithm -/

section Algo

namespace QTree

variable {Λ ι V β γ : Type*} [DecidableEq ι]

/-- The first read of each key not in `P`, in a list of reads. -/
def firstsList : List (Λ × ι) → Finset ι → List (Λ × ι)
  | [], _ => []
  | q :: l, P => if q.2 ∈ P then firstsList l P else q :: firstsList l (insert q.2 P)

theorem firsts_eq_list (rd : ι → V) :
    ∀ (t : QTree Λ ι V β) (P : Finset ι), t.firsts rd P = firstsList (t.asks rd) P
  | .done _, _ => rfl
  | .ask ℓ w next, P => by
    simp only [firsts, asks, firstsList]
    split_ifs <;> simp [firsts_eq_list rd (next (rd w))]

theorem firsts_of_asks {rd : ι → V} {t : QTree Λ ι V β} {t' : QTree Λ ι V γ} (P : Finset ι)
    (h : t.asks rd = t'.asks rd) : t.firsts rd P = t'.firsts rd P := by
  rw [firsts_eq_list, firsts_eq_list, h]

end QTree

variable (cut : FreeMonoid α → Option Bool) (T : DTree α) (E : EdgeTable α) (k : ℕ)
  (x : FreeMonoid α)

theorem firstsBy_new : New.firstsBy cut T (Function.uncurry E) k x = firstsBy cut T E k x :=
  QTree.firsts_of_asks _ (probeQuery_new (ofCut cut) T E k x).1

theorem edgeReadsBy_new : New.edgeReadsBy cut T (Function.uncurry E) k x = edgeReadsBy cut T E k x := by
  funext e; simp only [New.edgeReadsBy, edgeReadsBy, firstsBy_new]

theorem edgeHarvestBy_new :
    New.edgeHarvestBy cut T (Function.uncurry E) k x = edgeHarvestBy cut T E k x := by
  funext e; simp only [New.edgeHarvestBy, edgeHarvestBy, firstsBy_new]

theorem startHarvestBy_new : New.startHarvestBy cut T k x = startHarvestBy cut T k x := by
  simp only [New.startHarvestBy, startHarvestBy, sift_ofCut]

theorem middleHarvestBy_new :
    New.middleHarvestBy cut T (Function.uncurry E) k x = middleHarvestBy cut T E k x := by
  unfold New.middleHarvestBy middleHarvestBy
  rw [probeBy_new]
  rcases probeBy cut T E k x with _ | w | w | j | ⟨ps, fd⟩ | u <;> simp only [conv]
  · rcases T.sift (ofCut cut) w <;> rfl
  · rcases T.sift (ofCut cut) w <;> rfl
  · simp only [undecidedAt, sift_ofCut]
  · rcases x.toList[fd - 1]? with _ | c <;> rcases T.sift (ofCut cut) (prefixOf x fd) <;> rfl
  · rcases x.toList[u.toList.length]? with _ | c <;> rcases T.sift (ofCut cut) u with p | z <;>
      try rfl
    simp only
    rcases T.sift (ofCut cut) (u * FreeMonoid.of c) <;> rcases E p c <;> rfl

theorem recordBy_new :
    New.recordBy cut k (T, Function.uncurry E) x = recordBy cut k (T, E) x := by
  unfold New.recordBy recordBy
  simp only
  rw [probeBy_new]
  rcases probeBy cut T E k x with _ | w | w | j | ⟨ps, fd⟩ | u <;> simp only [conv]
  · rcases T.sift (ofCut cut) w <;> rfl
  · rcases T.sift (ofCut cut) w <;> rfl
  · rw [sift_ofCut]
    rcases x.toList[fd - 1]? with _ | c <;> rcases T.siftCut cut (prefixOf x fd) <;> rfl
  · rcases x.toList[u.toList.length]? with _ | c <;> rcases T.sift (ofCut cut) u with p | z <;>
      try rfl
    simp only
    rcases T.sift (ofCut cut) (u * FreeMonoid.of c) <;> rcases E p c <;> rfl

/-- Level 5's state over main's edges. -/
def lift (s : Level5State α) : New.Level5State α :=
  ⟨s.tree, Function.uncurry s.edges, s.version, s.probes, s.startUndecided, s.edgeReads,
    s.edgeUndecided, s.searches, s.middles⟩

omit [Fintype α] [DecidableEq α] in
@[simp] theorem lift_tree (s : Level5State α) : (lift s).tree = s.tree := rfl

theorem countProbe_new (s : Level5State α) :
    (lift s).countProbe cut k x = lift (s.countProbe cut k x) := by
  simp only [New.Level5State.countProbe, Level5State.countProbe, lift, startHarvestBy_new,
    edgeReadsBy_new, edgeHarvestBy_new, middleHarvestBy_new]

theorem setEdge_new (s : Level5State α) (p : List Bool) (c : α) (e : List Bool × FreeMonoid α) :
    (lift s).setEdge p c e = lift (s.setEdge p c e) := by
  simp only [New.Level5State.setEdge, Level5State.setEdge, New.freshState, freshState, lift]
  congr 1
  funext ⟨q, c'⟩
  simp only [Function.uncurry_apply_pair, Function.update_apply, Prod.mk.injEq]
  by_cases hq : q = p
  · subst hq; by_cases hc : c' = c <;> simp [hc]
  · simp [hq]

variable (C : Level5Settings)

theorem recordProbe_new (s : Level5State α) :
    New.recordProbe C cut (lift s) x = lift (recordProbe C cut s x) := by
  unfold New.recordProbe recordProbe
  simp only
  rw [countProbe_new, show (lift s).tree = s.tree from rfl,
    show (lift s).edges = Function.uncurry s.edges from rfl, probeBy_new]
  rcases probeBy cut s.tree s.edges C.walkStart x with _ | w | w | j | ⟨ps, fd⟩ | u <;>
    simp only [conv]
  · rcases s.tree.sift (ofCut cut) w <;> rfl
  · rcases s.tree.sift (ofCut cut) w <;> rfl
  · rfl
  · rcases x.toList[fd - 1]? with _ | c <;> rcases s.tree.sift (ofCut cut) (prefixOf x fd) <;> rfl
  · simp only [sift_ofCut]
    rcases x.toList[u.toList.length]? with _ | c
    · rfl
    rcases s.tree.siftCut cut u with p | b
    · simp only
      rcases s.tree.siftCut cut (u * FreeMonoid.of c) with t | b <;> rcases s.edges p c with _ | e
      · exact setEdge_new _ _ _ _
      all_goals rfl
    · rfl

theorem splitState_new (s : Level5State α) (p : List Bool) (c : α) (t : List Bool) :
    New.splitState cut (lift s) p c t = lift (splitState cut s p c t) := by
  unfold New.splitState splitState
  simp only [show (lift s).edges = Function.uncurry s.edges from rfl, Function.uncurry_apply_pair]
  rcases s.edges p c with _ | ⟨t₀, w₀⟩
  · rfl
  simp only [HypothesisDFA.split, New.freshState, freshState, lift]
  congr 1
  funext ⟨q, e⟩
  simp only [retarget, retargetBy, Function.uncurry_apply_pair, List.isPrefixOf_iff_prefix]
  split_ifs
  · rfl
  rcases s.edges q e with _ | ⟨s', w⟩
  · rfl
  simp only [sift_ofCut]

theorem runTests_new (s : Level5State α) : New.runTests C (lift s) = runTests C s := rfl

theorem applyProbe_new (s : Level5State α) :
    New.applyProbe C cut (lift s) x = (applyProbe C cut s x).map lift (Prod.map id lift) := by
  unfold New.applyProbe applyProbe
  simp only
  rw [recordProbe_new, runTests_new, show (lift s).tree = s.tree from rfl,
    show (lift s).edges = Function.uncurry s.edges from rfl, recordBy_new]
  rcases runTests C (recordProbe C cut s x) with _ | e
  · simp only
    rcases recordBy cut C.walkStart (s.tree, s.edges) x with _ | ⟨⟨p, c, t⟩, sp⟩
    · rfl
    · simp only [splitState_new, lift_tree]
      split_ifs <;> rfl
  · rfl

theorem runProbes_new : ∀ (xs : List (FreeMonoid α)) (s : Level5State α),
    New.runProbes C cut (lift s) xs = Prod.map lift id (runProbes C cut s xs)
  | [], s => rfl
  | x :: xs, s => by
    simp only [New.runProbes, runProbes, applyProbe_new]
    rcases applyProbe C cut s x with s' | ⟨e, s'⟩
    · exact runProbes_new xs s'
    · rfl

/-- Level 5 over main's probe is level 5: the same hypothesis and ending on every draw. -/
theorem indecisiveLStarLevel5_new (settings : Level5Settings) (read : FreeMonoid α → ARU) {P : ℕ}
    (draws : Fin P → FreeMonoid α) :
    New.indecisiveLStarLevel5 settings read draws = indecisiveLStarLevel5 settings read draws := by
  have h0 : lift (initialState : Level5State α) = New.initialState := by
    simp only [lift, initialState, New.initialState, freshState, New.freshState]
    congr 1
  unfold New.indecisiveLStarLevel5 indecisiveLStarLevel5
  simp only [← h0, runProbes_new]
  rfl

theorem siteHarvest_new (site : Level5Site α) :
    New.siteHarvest cut k site (T, Function.uncurry E) x = siteHarvest cut k site (T, E) x := by
  cases site <;> simp only [New.siteHarvest, siteHarvest, startHarvestBy_new, edgeHarvestBy_new,
    middleHarvestBy_new]

open MeasureTheory in
theorem harvestLaw_new (D : Measure (FreeMonoid α)) (H : HypothesisDFA α) (site : Level5Site α) :
    New.harvestLaw D cut k H site = harvestLaw D cut k H site := by
  unfold New.harvestLaw harvestLaw
  congr 1
  funext x
  rw [← siteHarvest_new, Function.uncurry_curry]

theorem searchingProbes_new (read : FreeMonoid α → ARU) (H : HypothesisDFA α) :
    New.searchingProbes read k H = searchingProbes read k H := by
  ext x
  simp only [New.searchingProbes, searchingProbes, Set.mem_setOf_eq]
  have h := probeBy_new H.tree (Function.curry H.edges) k x (fun z => (read z).cut)
  rw [ofCut_rd, Function.uncurry_curry] at h
  rw [h]
  rcases probeBy (fun z => (read z).cut) H.tree (Function.curry H.edges) k x
    with _ | w | w | j | ⟨ps, fd⟩ | u <;> simp only [conv]
  · rcases H.tree.sift read w <;> simp
  · rcases H.tree.sift read w <;> simp
  · rcases x.toList[fd - 1]? with _ | c <;> rcases H.tree.sift read (prefixOf x fd) <;> simp
  · rcases x.toList[u.toList.length]? with _ | c <;> rcases H.tree.sift read u with p | z
    · simp
    · simp
    · simp only
      rcases H.tree.sift read (u * FreeMonoid.of c) <;> rcases Function.curry H.edges p c <;> simp
    · simp

end Algo

#print axioms indecisiveLStarLevel5_new
#print axioms harvestLaw_new
#print axioms searchingProbes_new
#print axioms run_probeQuery_new

end OrthoDFA
