import OrthoDFA.StartAtK

/-!
# A probe's reads, in order

`Qry` is a computation that asks the cut about one string at a time, each ask tagged. Its trace
is what it asks, in order, and what it asks next depends only on the answers so far
(`trace_congr`). `qProbe` is `probeOutcome` written this way, with the reads of the search's
middles tagged.
-/

namespace OrthoDFA

/-- A computation that asks the cut about strings one at a time, each ask tagged. -/
inductive Qry (α : Type*) (β : Type*)
  | pure (b : β)
  | ask (w : FreeMonoid α) (tag : Bool) (k : Option Bool → Qry α β)

namespace Qry

variable {α β γ : Type*}

def run (cut : FreeMonoid α → Option Bool) : Qry α β → β
  | pure b => b
  | ask w _ k => (k (cut w)).run cut

def trace (cut : FreeMonoid α → Option Bool) : Qry α β → List (FreeMonoid α × Bool)
  | pure _ => []
  | ask w g k => (w, g) :: (k (cut w)).trace cut

def bind : Qry α β → (β → Qry α γ) → Qry α γ
  | pure b, f => f b
  | ask w g k, f => ask w g fun o => (k o).bind f

def map (f : β → γ) (q : Qry α β) : Qry α γ := q.bind fun b => pure (f b)

theorem run_bind (cut : FreeMonoid α → Option Bool) (f : β → Qry α γ) :
    ∀ q : Qry α β, (q.bind f).run cut = (f (q.run cut)).run cut
  | pure _ => rfl
  | ask w g k => run_bind cut f (k (cut w))

theorem trace_bind (cut : FreeMonoid α → Option Bool) (f : β → Qry α γ) :
    ∀ q : Qry α β, (q.bind f).trace cut = q.trace cut ++ (f (q.run cut)).trace cut
  | pure _ => rfl
  | ask w g k => by
    simp only [bind, trace, run, List.cons_append, trace_bind cut f (k (cut w))]

theorem run_map (cut : FreeMonoid α → Option Bool) (f : β → γ) (q : Qry α β) :
    (q.map f).run cut = f (q.run cut) := run_bind cut _ q

theorem trace_map (cut : FreeMonoid α → Option Bool) (f : β → γ) (q : Qry α β) :
    (q.map f).trace cut = q.trace cut := by
  simp [map, trace_bind, trace]

/-- What the computation asks next depends only on the answers to what it asked before. -/
theorem trace_congr {c₁ c₂ : FreeMonoid α → Option Bool} :
    ∀ (q : Qry α β) (r : ℕ), (∀ i (h : i < (q.trace c₁).length), i < r →
        c₂ ((q.trace c₁)[i]).1 = c₁ ((q.trace c₁)[i]).1) →
      (q.trace c₂).take (r + 1) = (q.trace c₁).take (r + 1)
  | pure _, _, _ => rfl
  | ask w g k, 0, _ => by simp [trace]
  | ask w g k, r + 1, h => by
    have hw : c₂ w = c₁ w := h 0 (by simp [trace]) (by omega)
    simp only [trace, hw, List.take_succ_cons]
    congr 1
    exact trace_congr (k (c₁ w)) r fun i hi hir => by
      have := h (i + 1) (by simp [trace]; omega) (by omega)
      simpa [trace] using this

/-- And where the answers to all it asks agree, so does what it returns. -/
theorem run_congr {c₁ c₂ : FreeMonoid α → Option Bool} :
    ∀ q : Qry α β, (∀ e ∈ q.trace c₁, c₂ e.1 = c₁ e.1) → q.run c₂ = q.run c₁
  | pure _, _ => rfl
  | ask w g k, h => by
    have hw : c₂ w = c₁ w := h (w, g) (by simp [trace])
    simp only [run, hw]
    exact run_congr (k (c₁ w)) fun e he => h e (by simp [trace, hw, he])

end Qry

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- `sift`, its reads tagged `g`. -/
def qSift (w : FreeMonoid α) (g : Bool) : DTree α → Qry α (List Bool ⊕ FreeMonoid α)
  | .leaf => .pure (.inl [])
  | .node m r a => .ask (w * m) g fun o =>
    match o with
    | none => .pure (.inr (w * m))
    | some true => (qSift w g a).map (Sum.map (true :: ·) id)
    | some false => (qSift w g r).map (Sum.map (false :: ·) id)

theorem qSift_run (cut : FreeMonoid α → Option Bool) (w : FreeMonoid α) (g : Bool) :
    ∀ t : DTree α, (qSift w g t).run cut = t.sift cut w
  | .leaf => rfl
  | .node m r a => by
    simp only [qSift, Qry.run, DTree.sift, DTree.route]
    rcases cut (w * m) with _ | _ | _
    · rfl
    · simp only [Qry.run_map, qSift_run cut w g r]; rfl
    · simp only [Qry.run_map, qSift_run cut w g a]; rfl

theorem qSift_trace (cut : FreeMonoid α → Option Bool) (w : FreeMonoid α) (g : Bool) :
    ∀ t : DTree α, (qSift w g t).trace cut = (t.route cut w).1.map (·, g)
  | .leaf => rfl
  | .node m r a => by
    simp only [qSift, Qry.trace, DTree.route]
    rcases cut (w * m) with _ | _ | _
    · rfl
    · simp only [Qry.trace_map, qSift_trace cut w g r, List.map_cons]
    · simp only [Qry.trace_map, qSift_trace cut w g a, List.map_cons]

/-- Whether the cut places `x`'s first `i` letters where the walk does, its reads tagged `g`. -/
def qAgrees (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool) (i : ℕ) (g : Bool) :
    Qry α (Option Bool) :=
  (qSift (prefixOf x i) g t).map fun s => s.elim (fun p => some (decide (p = walkAt i))) fun _ => none

theorem qAgrees_run (R : CutReads α) (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool)
    (i : ℕ) (g : Bool) : (qAgrees t x walkAt i g).run R.cut = agreesAt R t x walkAt i := by
  simp [qAgrees, Qry.run_map, qSift_run, agreesAt]

/-- `bracketAt`, the middles' reads tagged and their neighbours' not. -/
def qBracket (agq : ℕ → Bool → Qry α (Option Bool)) (ps : List (List Bool)) :
    ℕ → ℕ → ℕ → Qry α (Outcome α)
  | 0, _, hi => .pure (.edge ps hi)
  | fuel + 1, lo, hi =>
    if lo + 1 < hi then
      let ag := fun p g =>
        if p = lo then .pure (some true) else if p = hi then .pure (some false) else agq p g
      let mid := (lo + hi) / 2
      (ag mid true).bind fun o =>
        match o with
        | some true => qBracket agq ps fuel mid hi
        | some false => qBracket agq ps fuel lo mid
        | none => (ag (mid - 1) false).bind fun l =>
          match l with
          | none => .pure (.pair (mid - 1))
          | some lv => (ag (mid + 1) false).bind fun r =>
            match lv, r with
            | _, none => .pure (.pair mid)
            | true, some false => .pure (.triple mid)
            | true, some true => qBracket agq ps fuel (mid + 1) hi
            | false, some _ => qBracket agq ps fuel lo (mid - 1)
    else .pure (.edge ps hi)

theorem qBracket_run (cut : FreeMonoid α → Option Bool) (agq : ℕ → Bool → Qry α (Option Bool))
    (agrees : ℕ → Option Bool) (hag : ∀ p g, (agq p g).run cut = agrees p)
    (ps : List (List Bool)) :
    ∀ fuel lo hi, (qBracket agq ps fuel lo hi).run cut = bracketAt agrees ps fuel lo hi
  | 0, _, _ => rfl
  | fuel + 1, lo, hi => by
    have e : ∀ p g, (if p = lo then Qry.pure (some true) else if p = hi then Qry.pure (some false)
        else agq p g).run cut
        = if p = lo then some true else if p = hi then some false else agrees p := by
      intro p g; split_ifs <;> simp [Qry.run, hag]
    by_cases h : lo + 1 < hi
    · simp only [qBracket, bracketAt, if_pos h]
      rw [Qry.run_bind, e]
      generalize (if (lo + hi) / 2 = lo then some true else if (lo + hi) / 2 = hi then some false
        else agrees ((lo + hi) / 2)) = v
      rcases v with _ | _ | _
      · simp only []
        rw [Qry.run_bind, e]
        generalize (if (lo + hi) / 2 - 1 = lo then some true
          else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1)) = l
        rcases l with _ | _ | _
        · rfl
        all_goals
          simp only []
          rw [Qry.run_bind, e]
          generalize (if (lo + hi) / 2 + 1 = lo then some true
            else if (lo + hi) / 2 + 1 = hi then some false else agrees ((lo + hi) / 2 + 1)) = r
          rcases r with _ | _ | _ <;>
            simp only [Qry.run, qBracket_run cut agq agrees hag ps fuel]
      · exact qBracket_run cut agq agrees hag ps fuel _ _
      · exact qBracket_run cut agq agrees hag ps fuel _ _
    · simp only [qBracket, bracketAt, if_neg h]
      rfl

/-- `walkCheck`, its reads untagged. -/
def qWalk (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    Qry α (Outcome α ⊕ (List (List Bool) × ℕ)) :=
  (qSift (prefixOf x k) false t).bind fun s0 =>
    match s0 with
    | .inr _ => .pure (.inl (.startUndecided (prefixOf x k)))
    | .inl p =>
      match follow edges p (x.toList.drop k) with
      | .inl ps => (qSift x false t).bind fun s =>
        match s with
        | .inr _ => .pure (.inl (.endUndecided x))
        | .inl a => .pure (if some a = ps.getLast? then .inl .agree else .inr (ps, x.toList.length))
      | .inr (s, _, i) => (qSift (prefixOf x (k + i + 1)) false t).bind fun s1 =>
        match s1 with
        | .inr _ => .pure (.inl (.endUndecided (prefixOf x (k + i + 1))))
        | .inl _ => (qSift (prefixOf x (k + i)) false t).bind fun s2 =>
          match s2 with
          | .inr _ => .pure (.inl (.endUndecided (prefixOf x (k + i))))
          | .inl p' => .pure (if p' = s then .inl (.member (prefixOf x (k + i)))
              else .inr ((follow edges p ((x.toList.drop k).take i)).elim id (fun _ => []), k + i))

theorem qWalk_run (R : CutReads α) (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    (qWalk t edges k x).run R.cut = walkCheck R t edges k x := by
  unfold qWalk walkCheck kWalk
  rw [Qry.run_bind, qSift_run]
  rcases hk : t.sift R.cut (prefixOf x k) with p | b
  · simp only []
    rcases hf : follow edges p (x.toList.drop k) with ps | ⟨s, c, i⟩
    · simp only [Qry.run_bind, qSift_run]
      rcases t.sift R.cut x with a | b <;> simp [Qry.run]
    · simp only [Qry.run_bind, qSift_run]
      rcases t.sift R.cut (prefixOf x (k + i + 1)) with _ | _
      · simp only [Qry.run_bind, qSift_run]
        rcases t.sift R.cut (prefixOf x (k + i)) with p' | _
        · simp only [Qry.run]
          split_ifs
          · rfl
          · simp [walkTo, hk]
        · rfl
      · rfl
  · rfl

/-- `probeOutcome`, the reads of the search's middles tagged. -/
def qProbe (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) : Qry α (Outcome α) :=
  (qWalk t edges k x).bind fun d => d.elim .pure fun d =>
    qBracket (qAgrees t x fun j => d.1.getD (j - k) []) d.1 (d.2 - k) k d.2

theorem qProbe_run (R : CutReads α) (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    (qProbe t edges k x).run R.cut = probeOutcome R t edges k x := by
  unfold qProbe probeOutcome
  rw [Qry.run_bind, qWalk_run]
  rcases walkCheck R t edges k x with o | d
  · rfl
  · exact qBracket_run R.cut _ _ (qAgrees_run R t x _) d.1 _ _ _

end OrthoDFA
