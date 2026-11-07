import OrthoDFA.Code.Level3
import OrthoDFA.Proofs.Query
import Mathlib.Tactic.Eqns

/-!
# The functions' equations

Each function of `Code/` that reads the oracle is its program's run. Here each gets back the
equations it would have as a function of the read, which `rw` and `simp` use in its place
(`eqns`). A split keeps only the edges at the old tree's leaves (`splitLeaf_eq`), every learned
edge where they all are (`splitLeaf_eq_of`).
-/

namespace OrthoDFA

open Query

variable {α : Type*}

/-! ## The family read -/

section FamilyRead

variable {S : Type*} [Stringlike S]

theorem eval_familyReadQ (F : Finset S) (kl kh : ℕ) (w : S) (o : S → ℝ) :
    (familyReadQ F kl kh w).eval o = readOf kl kh (voteCount (fun z (_ : Unit) => o z) F w ()) := by
  unfold familyReadQ
  rw [eval_map, eval_table]
  simp only [voteCount]
  refine congrArg (fun n => readOf kl kh n) (congrArg Finset.card (Finset.filter_congr
    fun v hv => ?_))
  rw [if_pos (List.mem_map.2 ⟨v, Finset.mem_toList.2 hv, rfl⟩)]

theorem familyRead_eq {Ω : Type*} [MeasurableSpace Ω] (mq : S → Ω → ℝ) (F : Finset S)
    (kl kh : ℕ) (w : S) (ω : Ω) :
    familyRead mq F kl kh w ω = readOf kl kh (voteCount mq F w ω) :=
  eval_familyReadQ F kl kh w _

attribute [eqns familyRead_eq] familyRead

theorem eval_familyReadQ_mq {Ω : Type*} [MeasurableSpace Ω] (mq : S → Ω → ℝ) (ω : Ω)
    (F : Finset S) (kl kh : ℕ) (w : S) :
    (familyReadQ F kl kh w).eval (fun z => mq z ω) = familyRead mq F kl kh w ω :=
  rfl

end FamilyRead

/-! ## The sift -/

namespace DTree

variable (read : FreeMonoid α → ARU)

theorem sift_def (T : DTree α) (z : FreeMonoid α) : T.sift read z = (T.siftQ z).eval read := rfl

theorem sift_leaf (z : FreeMonoid α) : (leaf : DTree α).sift read z = .inl [] := rfl

theorem sift_node (m : FreeMonoid α) (r a : DTree α) (z : FreeMonoid α) :
    (node m r a).sift read z =
      match read (z * m) with
      | .accept => (a.sift read z).map (true :: ·) id
      | .reject => (r.sift read z).map (false :: ·) id
      | .undecided => .inr (z * m) := by
  change ((node m r a).siftQ z).eval read = _
  rw [siftQ, eval_ask]
  rcases read (z * m) with _ | _ | _ <;> simp only [eval_map, eval_pure] <;> rfl

attribute [eqns sift_leaf sift_node] sift

theorem eval_siftQ (T : DTree α) (z : FreeMonoid α) : (T.siftQ z).eval read = T.sift read z :=
  rfl

end DTree

/-! ## The retarget -/

section Retarget

variable (read : FreeMonoid α → ARU)

theorem retarget_eq (T' : DTree α) (p : List Bool) (E : Edges α) (e : List Bool × α) :
    retarget read T' p E e =
      if p.isPrefixOf e.1 then none
      else
        match E e with
        | some (t, w) =>
          if t = p then
            (T'.sift read (w * FreeMonoid.of e.2)).elim (fun t' => some (t', w)) fun _ => none
          else some (t, w)
        | none => none := by
  change (retargetAtQ T' p E e).eval read = _
  unfold retargetAtQ
  split_ifs
  · rfl
  rcases E e with _ | ⟨t, w⟩
  · rfl
  · simp only
    split_ifs
    · simp only [eval_map]; rfl
    · rfl

attribute [eqns retarget_eq] retarget

theorem eval_retargetAtQ (T' : DTree α) (p : List Bool) (E : Edges α) (e : List Bool × α) :
    (retargetAtQ T' p E e).eval read = retarget read T' p E e := rfl

theorem eval_retargetQ [DecidableEq α] (T' : DTree α) (p : List Bool) (E : Edges α) :
    ∀ es : List (List Bool × α), (retargetQ T' p E es).eval read
      = fun e => if e ∈ es then retarget read T' p E e else none
  | [] => by simp [retargetQ]
  | e :: es => by
    funext e'
    simp only [retargetQ, eval_bind, eval_map, eval_retargetQ T' p E es, eval_retargetAtQ,
      List.mem_cons]
    by_cases h : e' = e
    · subst h; simp
    · simp [h]

end Retarget

/-! ## The probe -/

section Probe

variable (read : FreeMonoid α → ARU) (T : DTree α) (x : FreeMonoid α) (st : ℕ → List Bool)

theorem siftAgrees_eq (p : ℕ) :
    siftAgrees read T x st p
      = (T.sift read (prefixOf x p)).elim (fun l => some (decide (l = st p))) fun _ => none := by
  change (siftAgreesQ T x st p).eval read = _
  simp only [siftAgreesQ, eval_map]
  rfl

theorem undecidedAt_eq (p : ℕ) :
    undecidedAt read T x p = (T.sift read (prefixOf x p)).elim (fun _ => []) fun z => [z] := by
  change (undecidedAtQ T x p).eval read = _
  simp only [undecidedAtQ, eval_map]
  rfl

attribute [eqns siftAgrees_eq] siftAgrees
attribute [eqns undecidedAt_eq] undecidedAt

variable (un : Option (α × List Bool)) (hi : ℕ)

theorem searchOutcome_undecidedHi :
    searchOutcome read T x st un hi .undecidedHi = .undecided (undecidedAt read T x hi) true := by
  change (searchOutcomeQ T x st un hi .undecidedHi).eval read = _
  simp only [searchOutcomeQ, eval_map]
  rfl

theorem searchOutcome_allAgree :
    searchOutcome read T x st un hi .allAgree =
      match un with
      | some (c, t) => .unlearnedEdge (st hi) c (prefixOf x hi) t
      | none => .agree := rfl

theorem searchOutcome_edge (p : ℕ) :
    searchOutcome read T x st un hi (.edge p) =
      match x.toList[p - 1]?, T.sift read (prefixOf x p) with
      | some c, .inl t => .wrongEdge (st (p - 1)) c (prefixOf x (p - 1)) t
      | _, _ => .undecided [] false := by
  change (searchOutcomeQ T x st un hi (.edge p)).eval read = _
  simp only [searchOutcomeQ]
  rcases x.toList[p - 1]? with _ | c
  · rfl
  · simp only [eval_map]
    rw [← DTree.sift_def]
    rcases T.sift read (prefixOf x p) with t | z <;> rfl

theorem searchOutcome_undecidedMid (p : ℕ) :
    searchOutcome read T x st un hi (.undecidedMid p)
      = .undecided (undecidedAt read T x p) false := by
  change (searchOutcomeQ T x st un hi (.undecidedMid p)).eval read = _
  simp only [searchOutcomeQ, eval_map]
  rfl

theorem searchOutcome_undecidedLo :
    searchOutcome read T x st un hi .undecidedLo = .undecided [] true := rfl

theorem searchOutcome_disagreeLo :
    searchOutcome read T x st un hi .disagreeLo = .undecided [] true := rfl

theorem searchOutcome_pair (p : ℕ) :
    searchOutcome read T x st un hi (.pair p) = .undecided [] true := rfl

theorem searchOutcome_triple (p : ℕ) :
    searchOutcome read T x st un hi (.triple p) = .undecided [] true := rfl

attribute [eqns searchOutcome_undecidedHi searchOutcome_allAgree searchOutcome_edge
  searchOutcome_undecidedMid searchOutcome_undecidedLo searchOutcome_disagreeLo searchOutcome_pair
  searchOutcome_triple] searchOutcome

variable (E : Edges α) (k : ℕ)

theorem probe_eq : probe read T E k x =
    match T.sift read (prefixOf x k) with
    | .inr z => .undecidedStart z
    | .inl a =>
      let ss := walk E a (x.toList.drop k)
      let st := fun p => ss.getD (p - k) []
      let hi := k + ss.length - 1
      let search := tripartiteSearch false (siftAgrees read T x st) k hi
      match x.toList[hi]? with
      | some c =>
        match T.sift read (prefixOf x (hi + 1)) with
        | .inr z => .undecided [z] true
        | .inl t => searchOutcome read T x st (some (c, t)) hi search
      | none => searchOutcome read T x st none hi search := by
  change (probeQ T E k x).eval read = _
  simp only [probeQ, eval_bind]
  rw [← DTree.sift_def]
  rcases T.sift read (prefixOf x k) with a | z
  · simp only
    rcases x.toList[k + (walk E a (x.toList.drop k)).length - 1]? with _ | c
    · simp only [eval_bind, eval_subst]
      rfl
    · simp only [eval_bind]
      rw [← DTree.sift_def]
      rcases T.sift read (prefixOf x (k + (walk E a (x.toList.drop k)).length - 1 + 1)) with t | z
      · simp only [eval_bind, eval_subst]
        rfl
      · rfl
  · rfl

attribute [eqns probe_eq] probe

theorem probeOrAgree_eq :
    probeOrAgree read T E k x = if x.toList.length < k then .agree else probe read T E k x := by
  change (probeOrAgreeQ T E k x).eval read = _
  unfold probeOrAgreeQ
  split_ifs <;> rfl

attribute [eqns probeOrAgree_eq] probeOrAgree

theorem eval_probeOrAgreeQ : (probeOrAgreeQ T E k x).eval read = probeOrAgree read T E k x := rfl

end Probe

/-! ## Level 3 -/

namespace Random

variable [DecidableEq α] (read : FreeMonoid α → ARU)

/-- Every learned edge is at a leaf. -/
def EdgesAtLeaves (s : Level3State α) : Prop :=
  ∀ e t w, s.edges e = some (t, w) → e.1 ∈ s.tree.leaves

variable [Fintype α] (C : Level3Settings)

theorem splitLeaf_eq (s : Level3State α) (p : List Bool) (c : α) (t t₀ : List Bool) :
    splitLeaf read s p c t t₀ =
      let T' := s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p
      newHypothesis T'
        (fun e => if e.1 ∈ s.tree.leaves then retarget read T' p s.edges e else none)
        (s.hypotheses + 1) := by
  change (splitLeafQ s p c t t₀).eval read = _
  simp only [splitLeafQ, eval_map, eval_retargetQ, Finset.mem_toList, Finset.mem_product,
    List.mem_toFinset, Finset.mem_univ, and_true]

attribute [eqns splitLeaf_eq] splitLeaf

theorem splitLeaf_eq_of {s : Level3State α} (hs : EdgesAtLeaves s) (p : List Bool) (c : α)
    (t t₀ : List Bool) :
    splitLeaf read s p c t t₀ =
      let T' := s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p
      newHypothesis T' (retarget read T' p s.edges) (s.hypotheses + 1) := by
  rw [splitLeaf_eq]
  simp only
  congr 1
  funext e
  split_ifs with he
  · rfl
  · rw [retarget_eq]
    split_ifs
    · rfl
    rcases hE : s.edges e with _ | ⟨t', w⟩
    · rfl
    · exact absurd (hs e t' w hE) he

theorem applyProbe_eq (s : Level3State α) (x : FreeMonoid α) :
    applyProbe read C s x =
      match probeOrAgree read s.tree s.edges C.walkStart x with
      | .unlearnedEdge p c u t =>
        .inl (newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u)))
          (s.hypotheses + 1))
      | .wrongEdge p c _ t =>
        match s.edges (p, c) with
        | some (t₀, _) => .inl (splitLeaf read s p c t t₀)
        | none => applyTests C (countProbe read C s x)
      | _ => applyTests C (countProbe read C s x) := by
  change (applyProbeQ C s x).eval read = _
  simp only [applyProbeQ, eval_bind, countProbe]
  rw [← eval_probeOrAgreeQ]
  rcases (probeOrAgreeQ s.tree s.edges C.walkStart x).eval read with _ | ⟨p, c, u, t⟩ | _ | _ | _
    <;> try rfl
  simp only
  rcases s.edges (p, c) with _ | ⟨t₀, w⟩
  · rfl
  · simp only [eval_map]
    rfl

attribute [eqns applyProbe_eq] applyProbe

theorem runProbes_nil (s : Level3State α) : runProbes read C s [] = (s, .failed) := rfl

theorem runProbes_cons (s : Level3State α) (x : FreeMonoid α) (xs : List (FreeMonoid α)) :
    runProbes read C s (x :: xs) =
      match applyProbe read C s x with
      | .inl s' => runProbes read C s' xs
      | .inr e => (s, e) := by
  change (runProbesQ C s (x :: xs)).eval read = _
  rw [runProbesQ, eval_bind]
  show _ = match (applyProbeQ C s x).eval read with
    | .inl s' => (runProbesQ C s' xs).eval read
    | .inr e => (s, e)
  rcases (applyProbeQ C s x).eval read with s' | e <;> rfl

attribute [eqns runProbes_nil runProbes_cons] runProbes

theorem indecisiveLStarLevel3_eq (settings : Level3Settings) {P : ℕ}
    (draws : Fin P → FreeMonoid α) :
    indecisiveLStarLevel3 settings read draws =
      let r := runProbes read settings initialState (List.ofFn draws)
      ⟨⟨r.1.tree, r.1.edges⟩, r.2⟩ := by
  change (level3Q settings draws).eval read = _
  simp only [level3Q, eval_map]
  rfl

attribute [eqns indecisiveLStarLevel3_eq] indecisiveLStarLevel3

theorem eval_applyProbeQ (s : Level3State α) (x : FreeMonoid α) :
    (applyProbeQ C s x).eval read = applyProbe read C s x := rfl

theorem eval_runProbesQ (s : Level3State α) (xs : List (FreeMonoid α)) :
    (runProbesQ C s xs).eval read = runProbes read C s xs := rfl

theorem eval_level3Q {T : ℕ} (xs : Fin T → FreeMonoid α) :
    (level3Q C xs).eval read = indecisiveLStarLevel3 C read xs := rfl

end Random

end OrthoDFA
