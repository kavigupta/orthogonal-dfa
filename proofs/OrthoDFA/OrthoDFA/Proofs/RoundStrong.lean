import OrthoDFA.RoundStrong
import OrthoDFA.Proofs.RoundLevel

/-!
# The strengthened round's deterministic claims
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace DTree

omit [Fintype α] [DecidableEq α] in
theorem sift_mem_paths {cut : FreeMonoid α → Option Bool} :
    ∀ (t : DTree α) (x : FreeMonoid α) (p : List Bool), t.sift cut x = .inl p → p ∈ t.paths
  | .leaf, x, p, h => by
    simp only [sift, route, Sum.inl.injEq] at h
    simp [paths, ← h]
  | .node m r a, x, p, h => by
    simp only [sift, route] at h
    rcases hc : cut (x * m) with _ | _ | _ <;> simp only [hc] at h
    · simp at h
    · rcases hr : (r.route cut x).2 with q | b <;> rw [hr] at h
      · simp only [Sum.map_inl, Sum.inl.injEq] at h
        subst h
        simp only [paths, List.mem_append, List.mem_map]
        exact .inl ⟨q, sift_mem_paths r x q hr, rfl⟩
      · simp at h
    · rcases hr : (a.route cut x).2 with q | b <;> rw [hr] at h
      · simp only [Sum.map_inl, Sum.inl.injEq] at h
        subst h
        simp only [paths, List.mem_append, List.mem_map]
        exact .inr ⟨q, sift_mem_paths a x q hr, rfl⟩
      · simp at h

omit [Fintype α] [DecidableEq α] in
theorem paths_ne_nil_of_node {m : FreeMonoid α} {r a : DTree α} :
    [] ∉ (DTree.node m r a).paths := by
  simp [paths]

omit [Fintype α] [DecidableEq α] in
theorem splitAt_paths_length (d : FreeMonoid α) :
    ∀ (t : DTree α) (p : List Bool), p ∈ t.paths →
      (t.splitAt d p).paths.length = t.paths.length + 1
  | .leaf, [], _ => by simp [splitAt, paths]
  | .leaf, _ :: _, h => by simp [paths] at h
  | .node _ _ _, [], h => absurd h paths_ne_nil_of_node
  | .node n r a, false :: p, h => by
    have : p ∈ r.paths := by simpa [paths] using h
    simp only [splitAt, paths, List.length_append, List.length_map,
      splitAt_paths_length d r p this]
    omega
  | .node n r a, true :: p, h => by
    have : p ∈ a.paths := by simpa [paths] using h
    simp only [splitAt, paths, List.length_append, List.length_map,
      splitAt_paths_length d a p this]
    omega

omit [Fintype α] [DecidableEq α] in
theorem append_not_mem_paths :
    ∀ (t : DTree α) (p : List Bool) (b : Bool), p ∈ t.paths → p ++ [b] ∉ t.paths
  | .leaf, p, b, h => by simp_all [paths]
  | .node _ r a, [], b, h => absurd h paths_ne_nil_of_node
  | .node _ r a, c :: p, b, h => by
    simp only [paths, List.mem_append, List.mem_map, List.cons_append] at h ⊢
    rintro (⟨q, hq, he⟩ | ⟨q, hq, he⟩) <;> rcases h with ⟨q', hq', he'⟩ | ⟨q', hq', he'⟩ <;>
      simp only [List.cons.injEq] at he he' <;> obtain ⟨rfl, rfl⟩ := he'
    · obtain ⟨-, rfl⟩ := he; exact append_not_mem_paths r q' b hq' hq
    · simp at he
    · simp at he
    · obtain ⟨-, rfl⟩ := he; exact append_not_mem_paths a q' b hq' hq

end DTree

section MajPath

theorem sidePath_mem_paths (side : FreeMonoid α → Bool) :
    ∀ (t : DTree α) (x : FreeMonoid α), sidePath side t x ∈ t.paths
  | .leaf, x => by simp [sidePath, DTree.paths]
  | .node m r a, x => by
    simp only [sidePath, DTree.paths, List.mem_append, List.mem_map]
    split
    · exact .inr ⟨_, sidePath_mem_paths side a x, rfl⟩
    · exact .inl ⟨_, sidePath_mem_paths side r x, rfl⟩

theorem sidePath_splitAt (side : FreeMonoid α → Bool) (d : FreeMonoid α) :
    ∀ (t : DTree α) (p : List Bool), p ∈ t.paths → ∀ x : FreeMonoid α,
      sidePath side (t.splitAt d p) x
        = if sidePath side t x = p then p ++ [side (x * d)] else sidePath side t x
  | .leaf, [], _, x => by
    simp only [DTree.splitAt, sidePath, if_true, List.nil_append]
    split <;> simp_all
  | .leaf, _ :: _, h, _ => by simp [DTree.paths] at h
  | .node _ _ _, [], h, _ => absurd h DTree.paths_ne_nil_of_node
  | .node n r a, false :: p, h, x => by
    have hp : p ∈ r.paths := by simpa [DTree.paths] using h
    simp only [DTree.splitAt, sidePath]
    split
    · simp
    · rw [sidePath_splitAt side d r p hp x]
      split <;> simp_all
  | .node n r a, true :: p, h, x => by
    have hp : p ∈ a.paths := by simpa [DTree.paths] using h
    simp only [DTree.splitAt, sidePath]
    split
    · rw [sidePath_splitAt side d a p hp x]
      split <;> simp_all
    · simp

variable {Q : Type*} [Fintype Q]

end MajPath

section SplitFacts

/-- What a recorded split read: its two strings sift to the leaf, and part at the distinguisher,
both decided. -/
def SplitOK (R : CutReads α) (r : SplitRec α) : Prop :=
  r.tree.sift R.cut r.y = .inl r.leaf ∧ r.tree.sift R.cut r.sprime = .inl r.leaf
    ∧ ∃ b, R.cut (r.y * r.d) = some b ∧ R.cut (r.sprime * r.d) = some (!b)

theorem parting_inl {cut : FreeMonoid α → Option Bool} {x y pre d : FreeMonoid α} :
    ∀ t : DTree α, t.parting cut x y pre = some (.inl d) →
      ∃ b, cut (x * d) = some b ∧ cut (y * d) = some (!b)
  | .leaf, h => by simp [DTree.parting] at h
  | .node m r a, h => by
    simp only [DTree.parting] at h
    rcases hx : cut (x * (pre * m)) with _ | _ | _ <;>
      rcases hy : cut (y * (pre * m)) with _ | _ | _ <;>
      simp only [hx, hy, reduceCtorEq, Option.some.injEq, Sum.inl.injEq] at h
    · exact parting_inl r h
    · subst h; exact ⟨false, hx, hy⟩
    · subst h; exact ⟨true, hx, hy⟩
    · exact parting_inl a h

theorem seedStep_split_facts (K : StageKnobs α) (R : CutReads α) {t : DTree α}
    {pool : List (FreeMonoid α)} {edges : Edges α} {skip : TestKey α → FreeMonoid α → Prop}
    {forced : Set (TestKey α)} {k : ℕ} {x : FreeMonoid α}
    {ps : List (List Bool)} {fd : ℕ} {d : FreeMonoid α} {s1 : List Bool} {y sprime : FreeMonoid α}
    (h : seedStep K R t pool edges skip forced k x ps fd = .split d s1 y sprime) :
    SplitOK R ⟨t, s1, d, y, sprime⟩ := by
  unfold seedStep at h
  simp only [] at h
  split at h
  · simp at h
  split at h
  · simp at h
  split_ifs at h
  rename_i hsp
  split at h
  · simp at h
  rename_i p hp
  split_ifs at h with hg
  push Not at hg
  split at h
  · simp at h
  · simp at h
  rename_i d' hpt
  split_ifs at h
  all_goals try (simp at h; done)
  split at h
  · simp only [SeedResult.split.injEq] at h
    obtain ⟨rfl, rfl, rfl, rfl⟩ := h
    exact ⟨hg.2, hg.1 ▸ hp, parting_inl t hpt⟩
  · simp at h

theorem faithfulRoute_sift {R : CutReads α} {side : FreeMonoid α → Bool} {z : FreeMonoid α} :
    ∀ (t : DTree α) (x : FreeMonoid α), FaithfulRoute R side z t x →
      t.sift R.cut x = .inl (sidePath side t z)
  | .leaf, _, _ => rfl
  | .node m r a, x, ⟨hc, hrest⟩ => by
    unfold DTree.sift DTree.route
    simp only [hc, sidePath]
    cases hs : side (z * m) <;> simp only [hs, if_true] at hrest ⊢
    · have := faithfulRoute_sift r x hrest
      unfold DTree.sift at this
      simp [this]
    · have := faithfulRoute_sift a x hrest
      unfold DTree.sift at this
      simp [this]

variable {Q : Type*} (R : CutReads α) (A : DFA (FreeMonoid α) Q) (side : FreeMonoid α → Bool)
  (rep : Q → FreeMonoid α)

end SplitFacts

section Step

variable (C : StrongCfg α) (R : CutReads α)

theorem split_leaf_mem {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    {skip : TestKey α → FreeMonoid α → Prop}
    {forced : Set (TestKey α)} {k : ℕ}
    (hl : Learned R t edges) {x : FreeMonoid α} {ps : List (List Bool)} {fd : ℕ}
    {d y sp : FreeMonoid α} {s1 : List Bool}
    (h : seedStep C.K R t pool edges skip forced k x ps fd = .split d s1 y sp) : s1 ∈ t.paths := by
  obtain ⟨⟨c, s2, he⟩, -⟩ := seedStep_split_spec C.K R h
  exact DTree.sift_mem_paths t _ _ (hl _ _ _ _ he).1

theorem strongStep_state (A : RoundAcc α) (x : FreeMonoid α) :
    (strongStep C R A x).s = probeStepK C.K R C.k A.s x
      ∧ (strongStep C R A x).used = A.used + 1 ∧ (strongStep C R A x).certs = A.certs := by
  unfold strongStep
  simp only []
  repeat' split
  all_goals simp

theorem probeStepK_streak (A : RoundAcc α) (x : FreeMonoid α) :
    (probeStepK C.K R C.k A.s x).streak = A.s.streak + 1
      ∨ (probeStepK C.K R C.k A.s x).streak = 0 := by
  unfold probeStepK
  simp only []
  repeat' split
  all_goals simp [closeK]

theorem strongStep_cases (A : RoundAcc α) (x : FreeMonoid α)
    (hl : Learned R A.s.tree A.s.edges) :
    ((strongStep C R A x).splits = A.splits ∧ (strongStep C R A x).s.tree = A.s.tree)
    ∨ ∃ r : SplitRec α, (strongStep C R A x).splits = A.splits ++ [r] ∧ r.tree = A.s.tree
      ∧ r.leaf ∈ A.s.tree.paths ∧ (strongStep C R A x).s.tree = A.s.tree.splitAt r.d r.leaf
      ∧ SplitOK R r := by
  by_cases hE : ∃ ps fd, probeOutcome R A.s.tree A.s.edges C.k x = .edge ps fd
  swap
  · left
    have hst : strongStep C R A x = { A with s := probeStepK C.K R C.k A.s x, used := A.used + 1 }
        := by
      unfold strongStep
      simp only []
      split
      · exact absurd ⟨_, _, by assumption⟩ hE
      · rfl
    have hp : (probeStepK C.K R C.k A.s x).tree = A.s.tree := by
      unfold probeStepK
      simp only []
      split
      · exact absurd ⟨_, _, by assumption⟩ hE
      all_goals simp [closeK]
    rw [hst]
    exact ⟨rfl, hp⟩
  obtain ⟨ps, fd, ho⟩ := hE
  rcases hs : seedStep C.K R A.s.tree A.s.pool A.s.edges (stepSkip C.K R C.k A.s x) C.K.forced
      C.k x ps fd with
    ⟨d, s1, y, sp⟩ | ⟨s1, sp⟩ | b | _
  · right
    refine ⟨⟨A.s.tree, s1, d, y, sp⟩, ?_, rfl, split_leaf_mem C R hl hs, ?_,
      seedStep_split_facts C.K R hs⟩ <;>
      simp [strongStep, probeStepK, ho, hs, closeK]
  all_goals
    left
    exact ⟨by simp [strongStep, ho, hs], by simp [strongStep, probeStepK, ho, hs, closeK]⟩

end Step

section Count

variable (C : StrongCfg α) (R : CutReads α)

/-- What every state the round reaches satisfies. -/
def StrongInv (A : RoundAcc α) : Prop :=
  Learned R A.s.tree A.s.edges ∧ A.splits.length + 2 = A.s.tree.paths.length

/-- One probe of the pass. -/
noncomputable def passBody (A : RoundAcc α) (x : FreeMonoid α) : RoundAcc α :=
  if C.K.patience ≤ A.s.streak ∨ C.budget A.s.tree.paths.length ≤ A.used then A
  else strongStep C R A x

theorem strongPass_eq (A : RoundAcc α) (probes : List (FreeMonoid α)) :
    strongPass C R A probes
      = probes.foldl (passBody C R)
        { A with s := { A.s with streak := 0, log := A.s.log ∪ A.reads } } :=
  rfl

theorem strongReading_fst (j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (y : C.Draws) :
    (strongReading C R j A first y).1.s = (strongPass C R A (first ++ List.ofFn y.1)).s
      ∧ (strongReading C R j A first y).1.splits
        = (strongPass C R A (first ++ List.ofFn y.1)).splits
      ∧ (strongReading C R j A first y).1.used
        = (strongPass C R A (first ++ List.ofFn y.1)).used := by
  unfold strongReading
  simp only []
  split_ifs <;> simp

theorem strongReading_rerun_budget {j : ℕ} {A : RoundAcc α} {first lv : List (FreeMonoid α)}
    {y : C.Draws} (h : (strongReading C R j A first y).2 = .inr lv) :
    (strongPass C R A (first ++ List.ofFn y.1)).used
      < C.budget (strongPass C R A (first ++ List.ofFn y.1)).s.tree.paths.length := by
  unfold strongReading at h
  simp only [] at h
  split_ifs at h with h1 h2 h3 h4 <;> first | omega | simp at h

theorem strongStep_inv (A : RoundAcc α) (x : FreeMonoid α) (hI : StrongInv R A) :
    StrongInv R (strongStep C R A x) := by
  obtain ⟨hl, hlen⟩ := hI
  obtain ⟨hs, -⟩ := strongStep_state C R A x
  refine ⟨by rw [hs]; exact probeStepK_learned C.K R hl, ?_⟩
  rcases strongStep_cases C R A x hl with ⟨hsp, ht⟩ | ⟨r, hsp, -, hrl, ht, -⟩
  · rw [hsp, ht]; exact hlen
  · rw [hsp, ht, DTree.splitAt_paths_length _ _ _ hrl, List.length_append]
    simp only [List.length_singleton]
    omega

theorem strongStep_leaves (A : RoundAcc α) (x : FreeMonoid α) (hI : StrongInv R A) :
    A.s.tree.paths.length ≤ (strongStep C R A x).s.tree.paths.length := by
  have h1 := hI.2
  have h2 := (strongStep_inv C R A x hI).2
  rcases strongStep_cases C R A x hI.1 with ⟨hsp, -⟩ | ⟨r, hsp, -⟩ <;> rw [hsp] at h2
  · omega
  · simp only [List.length_append, List.length_singleton] at h2
    omega

/-- Through a pass, the state stays reachable, the leaves only grow, and the probes spent stay
within the budget at the leaves reached. -/
theorem fold_inv (hb : Monotone C.budget) :
    ∀ (probes : List (FreeMonoid α)) (X : RoundAcc α), StrongInv R X →
      X.used ≤ C.budget X.s.tree.paths.length →
      StrongInv R (probes.foldl (passBody C R) X)
        ∧ X.s.tree.paths.length ≤ (probes.foldl (passBody C R) X).s.tree.paths.length
        ∧ (probes.foldl (passBody C R) X).used
          ≤ C.budget (probes.foldl (passBody C R) X).s.tree.paths.length
        ∧ (probes.foldl (passBody C R) X).certs = X.certs := by
  intro probes
  induction probes with
  | nil => exact fun X hI hu => ⟨hI, le_rfl, hu, rfl⟩
  | cons x xs ih =>
    intro X hI hu
    simp only [List.foldl_cons]
    by_cases hg : C.K.patience ≤ X.s.streak ∨ C.budget X.s.tree.paths.length ≤ X.used
    · have h1 : passBody C R X x = X := by unfold passBody; rw [if_pos hg]
      rw [h1]
      exact ih X hI hu
    · have h1 : passBody C R X x = strongStep C R X x := by unfold passBody; rw [if_neg hg]
      rw [h1]
      push Not at hg
      obtain ⟨-, hus, hce⟩ := strongStep_state C R X x
      have hl := strongStep_leaves C R X x hI
      obtain ⟨i1, i2, i3, i4⟩ := ih _ (strongStep_inv C R X x hI)
        (by rw [hus]; exact (Nat.succ_le_of_lt hg.2).trans (hb hl))
      exact ⟨i1, hl.trans i2, i3, i4.trans hce⟩

/-- A pass spends at least `patience` less its starting streak, or all it is given, unless it
stops at the budget. -/
theorem fold_gain :
    ∀ (probes : List (FreeMonoid α)) (X : RoundAcc α),
      C.budget (probes.foldl (passBody C R) X).s.tree.paths.length
          ≤ (probes.foldl (passBody C R) X).used
        ∨ X.used + min probes.length (C.K.patience - X.s.streak)
          ≤ (probes.foldl (passBody C R) X).used := by
  have hstuck : ∀ (l : List (FreeMonoid α)) (X : RoundAcc α),
      (C.K.patience ≤ X.s.streak ∨ C.budget X.s.tree.paths.length ≤ X.used) →
        l.foldl (passBody C R) X = X := by
    intro l
    induction l with
    | nil => exact fun _ _ => rfl
    | cons x xs ih =>
      intro X hg
      simp only [List.foldl_cons]
      have h1 : passBody C R X x = X := by unfold passBody; rw [if_pos hg]
      rw [h1]
      exact ih X hg
  intro probes
  induction probes with
  | nil => exact fun X => .inr (by simp)
  | cons x xs ih =>
    intro X
    by_cases hg : C.K.patience ≤ X.s.streak ∨ C.budget X.s.tree.paths.length ≤ X.used
    · rw [hstuck _ X hg]
      rcases hg with hg | hg
      · exact .inr (by rw [Nat.sub_eq_zero_of_le hg]; simp)
      · exact .inl hg
    · simp only [List.foldl_cons]
      have h1 : passBody C R X x = strongStep C R X x := by unfold passBody; rw [if_neg hg]
      rw [h1]
      push Not at hg
      obtain ⟨hs, hus, -⟩ := strongStep_state C R X x
      rcases ih (strongStep C R X x) with h | h
      · exact .inl h
      · refine .inr (le_trans ?_ h)
        rw [hus, hs]
        simp only [List.length_cons]
        rcases probeStepK_streak C R X x with e | e <;> rw [e] <;> omega

end Count

end OrthoDFA

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

section Round

variable (C : StrongCfg α) (R : CutReads α)

/-- What a reading starts from, `j` readings into the round: the probes spent are at least
`patience` per reading so far and within the budget. -/
def Entry (A : RoundAcc α) (j : ℕ) : Prop :=
  StrongInv R A ∧ j * C.K.patience ≤ A.used ∧ A.used ≤ C.budget A.s.tree.paths.length

theorem strongReading_ne_cap (j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α))
    (y : C.Draws) : (strongReading C R j A first y).2 ≠ .inl .cap := by
  unfold strongReading
  simp only []
  split_ifs <;> simp

/-- A reading from an entry keeps the probes within the budget, and a rerun spends at least
`patience` more. -/
theorem reading_entry (hp : C.K.patience ≤ C.np) (hb : Monotone C.budget) {A : RoundAcc α}
    {j : ℕ} (hE : Entry C R A j) (first : List (FreeMonoid α)) (y : C.Draws) :
    StrongInv R (strongReading C R j A first y).1
      ∧ (strongReading C R j A first y).1.used
        ≤ C.budget (strongReading C R j A first y).1.s.tree.paths.length
      ∧ j * C.K.patience ≤ (strongReading C R j A first y).1.used
      ∧ ∀ lv, (strongReading C R j A first y).2 = .inr lv →
        Entry C R (strongReading C R j A first y).1 (j + 1) := by
  obtain ⟨hI, hj, hu⟩ := hE
  obtain ⟨hs, hsp, hus⟩ := strongReading_fst C R j A first y
  have hrr := fun lv (h : (strongReading C R j A first y).2 = .inr lv) =>
    strongReading_rerun_budget C R h
  rw [strongPass_eq] at hs hsp hus hrr
  set X : RoundAcc α := { A with s := { A.s with streak := 0, log := A.s.log ∪ A.reads } }
  have hIX : StrongInv R X := hI
  obtain ⟨i1, i2, i3, -⟩ := fold_inv C R hb (first ++ List.ofFn y.1) X hIX hu
  have hgain := fold_gain C R (first ++ List.ofFn y.1) X
  set Y := (first ++ List.ofFn y.1).foldl (passBody C R) X
  have hIR : StrongInv R (strongReading C R j A first y).1 := by
    unfold StrongInv; rw [hs, hsp]; exact i1
  have hle : X.used ≤ Y.used := by
    rcases hgain with h | h
    · exact hu.trans ((hb i2).trans h)
    · exact le_trans (Nat.le_add_right _ _) h
  refine ⟨hIR, by rw [hs, hus]; exact i3, by rw [hus]; exact hj.trans hle, fun lv h => ?_⟩
  refine ⟨hIR, ?_, by rw [hs, hus]; exact i3⟩
  have hlt := hrr lv h
  rcases hgain with h' | h'
  · omega
  · rw [hus]
    have hlen : C.K.patience ≤ (first ++ List.ofFn y.1).length := by
      simp only [List.length_append, List.length_ofFn]; omega
    have : X.s.streak = 0 := rfl
    rw [this, Nat.sub_zero, min_eq_right hlen] at h'
    have hXu : X.used = A.used := rfl
    rw [Nat.succ_mul]
    omega

/-- Over the round from an entry: the probes stay within the budget, the readings times
`patience` are at most the probes spent and one more `patience`, and running out of readings
means `patience` for each. -/
theorem round_used (hp : C.K.patience ≤ C.np) (hb : Monotone C.budget) :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      Entry C R A j →
      StrongInv R (strongRound C R n j A first d).2.1
        ∧ (strongRound C R n j A first d).2.1.used
          ≤ C.budget (strongRound C R n j A first d).2.1.s.tree.paths.length
        ∧ (j + (strongRound C R n j A first d).2.2.length) * C.K.patience
          ≤ (strongRound C R n j A first d).2.1.used + C.K.patience
        ∧ ((strongRound C R n j A first d).1 = .cap →
          (j + n) * C.K.patience ≤ (strongRound C R n j A first d).2.1.used)
  | 0, j, A, first, d, hE => by
    simp only [strongRound, List.length_nil, Nat.add_zero]
    exact ⟨hE.1, hE.2.2, by have := hE.2.1; omega, fun _ => hE.2.1⟩
  | n + 1, j, A, first, d, hE => by
    obtain ⟨g1, g2, g3, g4⟩ := reading_entry C R hp hb hE first (d 0)
    have hnc := strongReading_ne_cap C R j A first (d 0)
    rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩
    all_goals
      rw [hR] at g1 g2 g3 g4 hnc
      dsimp only at g1 g2 g3 g4 hnc
      simp only [strongRound, hR]
    · refine ⟨g1, g2, by simp only [List.length_singleton]; rw [Nat.add_mul]; omega,
        fun he => absurd (by rw [he]) hnc⟩
    · obtain ⟨i1, i2, i3, i4⟩ := round_used hp hb n (j + 1) A'' lv (Fin.tail d) (g4 lv rfl)
      refine ⟨i1, i2, by simp only [List.length_cons]; convert i3 using 2; omega, fun hc => ?_⟩
      have := i4 hc
      rw [show j + (n + 1) = j + 1 + n by omega]
      exact this

end Round

end OrthoDFA

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Q : Type*}

section StrongReading

variable (C : StrongCfg α) (R : CutReads α)

variable (A : DFA (FreeMonoid α) Q) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
  (CertGood : DTree α → Edges α → Prop)

instance StrongCfg.drawMeasure_prob (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] :
    IsProbabilityMeasure (C.drawMeasure D) := by
  unfold StrongCfg.drawMeasure; infer_instance

end StrongReading

end OrthoDFA

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Segments

variable (C : StrongCfg α)

/-- A reading's accumulator once its gate, refusal sample and certificate have read the draws
`y`. -/
noncomputable def afterReads (R : CutReads α) (A : RoundAcc α) (y : C.Draws) : RoundAcc α :=
  { A with reads := A.reads ∪ readingReads C R.F A.s.tree y }

/-- The round's passes over its readings' probes, one segment each, with each reading's draws. -/
noncomputable def segRun (R : CutReads α) :
    RoundAcc α → List (List (FreeMonoid α) × C.Draws) → RoundAcc α
  | A, [] => A
  | A, seg :: rest => segRun R (afterReads C R (strongPass C R A seg.1) seg.2) rest

theorem strongStep_certs (R : CutReads α) (A : RoundAcc α) (c : ℕ) (x : FreeMonoid α) :
    strongStep C R { A with certs := c } x = { strongStep C R A x with certs := c } := by
  unfold strongStep
  simp only []
  split
  · split <;> rfl
  · rfl

theorem passBody_certs (R : CutReads α) (A : RoundAcc α) (c : ℕ) (x : FreeMonoid α) :
    passBody C R { A with certs := c } x = { passBody C R A x with certs := c } := by
  unfold passBody
  split_ifs <;> first | rfl | exact strongStep_certs C R A c x

theorem fold_certs (R : CutReads α) :
    ∀ (probes : List (FreeMonoid α)) (A : RoundAcc α) (c : ℕ),
      probes.foldl (passBody C R) { A with certs := c }
        = { probes.foldl (passBody C R) A with certs := c }
  | [], _, _ => rfl
  | x :: xs, A, c => by
    simp only [List.foldl_cons, passBody_certs]
    exact fold_certs R xs _ c

theorem afterReads_certs (R : CutReads α) (A : RoundAcc α) (c : ℕ) (y : C.Draws) :
    afterReads C R { A with certs := c } y = { afterReads C R A y with certs := c } := rfl

theorem segRun_certs (R : CutReads α) :
    ∀ (segs : List (List (FreeMonoid α) × C.Draws)) (A : RoundAcc α) (c : ℕ),
      segRun C R { A with certs := c } segs = { segRun C R A segs with certs := c }
  | [], _, _ => rfl
  | seg :: rest, A, c => by
    simp only [segRun, strongPass_eq]
    rw [show ({ ({ A with certs := c } : RoundAcc α) with
        s := { A.s with streak := 0, log := A.s.log ∪ A.reads } } : RoundAcc α)
        = { ({ A with s := { A.s with streak := 0, log := A.s.log ∪ A.reads } } : RoundAcc α) with
          certs := c } from rfl, fold_certs, afterReads_certs]
    exact segRun_certs R rest _ c

theorem strongReading_acc (R : CutReads α) (j : ℕ) (A : RoundAcc α)
    (first : List (FreeMonoid α)) (y : C.Draws) :
    (strongReading C R j A first y).1
      = { afterReads C R (strongPass C R A (first ++ List.ofFn y.1)) y
      with certs := (strongReading C R j A first y).1.certs } := by
  unfold strongReading
  simp only []
  split_ifs <;> rfl

open scoped Classical in
/-- The rerun lists a refusal sample `br` can give: its draws picked by a set of positions. -/
noncomputable def sublistsOfS (br : Fin C.nr → FreeMonoid α) : Finset (List (FreeMonoid α)) :=
  (Finset.univ : Finset (Finset (Fin C.nr))).image fun S =>
    ((List.finRange C.nr).filter fun i => decide (i ∈ S)).map br

theorem card_sublistsOfS (br : Fin C.nr → FreeMonoid α) :
    (sublistsOfS C br).card ≤ 2 ^ C.nr := by
  unfold sublistsOfS
  refine Finset.card_image_le.trans (le_of_eq ?_)
  rw [Finset.card_univ, Fintype.card_finset, Fintype.card_fin]

open scoped Classical in
theorem mem_sublistsOfS (br : Fin C.nr → FreeMonoid α) (q : Fin C.nr → Bool) :
    ((List.finRange C.nr).filter q).map br ∈ sublistsOfS C br :=
  Finset.mem_image.2 ⟨Finset.univ.filter fun i => q i = true, Finset.mem_univ _, by
    congr 1
    exact List.filter_congr fun i _ => by simp⟩

theorem strongReading_rerun_mem (R : CutReads α) {j : ℕ} {A : RoundAcc α}
    {first lv : List (FreeMonoid α)} {y : C.Draws}
    (h : (strongReading C R j A first y).2 = .inr lv) : lv ∈ sublistsOfS C y.2.2.1 := by
  classical
  unfold strongReading at h
  simp only [] at h
  split_ifs at h <;> simp only [Sum.inr.injEq] at h
  all_goals
    subst h
    exact mem_sublistsOfS C _ _

open scoped Classical in
/-- The segment lists a round over the draws `d` can take in `r` readings, from first probes
among `firsts`. -/
noncomputable def candS : (n : ℕ) → (Fin n → C.Draws) → ℕ → Finset (List (FreeMonoid α))
    → Finset (List (List (FreeMonoid α) × C.Draws))
  | _, _, 0, _ => {[]}
  | 0, _, _ + 1, _ => ∅
  | n + 1, d, r + 1, firsts => firsts.biUnion fun f =>
    insert [(f ++ List.ofFn (d 0).1, d 0)]
      ((candS n (Fin.tail d) r (sublistsOfS C (d 0).2.2.1)).image
        ((f ++ List.ofFn (d 0).1, d 0) :: ·))

open scoped Classical in
theorem card_candS :
    ∀ (n : ℕ) (d : Fin n → C.Draws) (r : ℕ) (firsts : Finset (List (FreeMonoid α))),
      (candS C n d r firsts).card ≤ max 1 firsts.card * 2 ^ ((C.nr + 1) * r)
  | _, _, 0, firsts => by simp [candS]
  | 0, _, _ + 1, firsts => by simp [candS]
  | n + 1, d, r + 1, firsts => by
    have ih := card_candS n (Fin.tail d) r (sublistsOfS C (d 0).2.2.1)
    have hS : max 1 (sublistsOfS C (d 0).2.2.1).card ≤ 2 ^ C.nr :=
      max_le (Nat.one_le_two_pow) (card_sublistsOfS C _)
    have hone : ∀ f ∈ firsts, (insert [(f ++ List.ofFn (d 0).1, d 0)]
        ((candS C n (Fin.tail d) r (sublistsOfS C (d 0).2.2.1)).image
          ((f ++ List.ofFn (d 0).1, d 0) :: ·))).card ≤ 2 ^ ((C.nr + 1) * (r + 1)) := by
      intro f _
      refine (Finset.card_insert_le _ _).trans ?_
      have h1 := (Finset.card_image_le (s := candS C n (Fin.tail d) r (sublistsOfS C (d 0).2.2.1))
        (f := ((f ++ List.ofFn (d 0).1, d 0) :: ·))).trans (ih.trans (Nat.mul_le_mul_right _ hS))
      have h2 : 2 ^ ((C.nr + 1) * (r + 1)) = 2 * (2 ^ C.nr * 2 ^ ((C.nr + 1) * r)) := by ring
      have h3 : 1 ≤ 2 ^ C.nr * 2 ^ ((C.nr + 1) * r) := Nat.one_le_iff_ne_zero.2 (by positivity)
      omega
    simp only [candS]
    refine (Finset.card_biUnion_le).trans ((Finset.sum_le_sum hone).trans ?_)
    rw [Finset.sum_const, smul_eq_mul]
    exact Nat.mul_le_mul_right _ (le_max_right _ _)

open scoped Classical in
/-- The round's probes are its readings' segments in order, and it ends where the segments' passes
do. -/
theorem strongRound_segs (R : CutReads α) :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws)
      (firsts : Finset (List (FreeMonoid α))), first ∈ firsts →
      ∃ segs ∈ candS C n d (strongRound C R n j A first d).2.2.length firsts,
        strongProbes C R n j A first d = (segs.map Prod.fst).flatten
          ∧ (strongRound C R n j A first d).2.1.s = (segRun C R A segs).s
  | 0, _, _, _, _, _, _ => ⟨[], by simp [strongRound, candS], by simp [strongProbes],
      by simp [strongRound, segRun]⟩
  | n + 1, j, A, first, d, firsts, hf => by
    have hacc := strongReading_acc C R j A first (d 0)
    have hmem := fun lv (h : (strongReading C R j A first (d 0)).2 = .inr lv) =>
      strongReading_rerun_mem C R h
    rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩ <;>
      rw [hR] at hacc hmem <;> dsimp only at hacc hmem <;>
      simp only [strongRound, strongProbes, hR]
    · refine ⟨[(first ++ List.ofFn (d 0).1, d 0)], ?_, by simp, ?_⟩
      · simp only [List.length_singleton, candS]
        exact Finset.mem_biUnion.2 ⟨first, hf, Finset.mem_insert_self _ _⟩
      · rw [hacc]; rfl
    · obtain ⟨segs, hs, hp, hst⟩ := strongRound_segs R n (j + 1) A'' lv (Fin.tail d)
        (sublistsOfS C (d 0).2.2.1) (hmem lv rfl)
      refine ⟨(first ++ List.ofFn (d 0).1, d 0) :: segs, ?_, ?_, ?_⟩
      · simp only [List.length_cons, candS]
        exact Finset.mem_biUnion.2 ⟨first, hf, Finset.mem_insert_of_mem
          (Finset.mem_image.2 ⟨segs, hs, rfl⟩)⟩
      · rw [hp, List.map_cons, List.flatten_cons]
      · rw [hst, hacc, segRun_certs]
        rfl

theorem strongStep_mids (R : CutReads α) (A : RoundAcc α) (x : FreeMonoid α) :
    ∀ m ∈ A.s.tree.mids, m ∈ (strongStep C R A x).s.tree.mids := fun m hm => by
  rw [(strongStep_state C R A x).1]
  rcases probeStepK_tree_cases C.K R C.k A.s x with h | ⟨d, p, h⟩
  · rw [h]; exact hm
  · rw [h]; exact DTree.mem_mids_splitAt hm

theorem fold_mids (R : CutReads α) :
    ∀ (probes : List (FreeMonoid α)) (A : RoundAcc α),
      ∀ m ∈ A.s.tree.mids, m ∈ (probes.foldl (passBody C R) A).s.tree.mids
  | [], _, _, hm => hm
  | x :: xs, A, m, hm => by
    simp only [List.foldl_cons]
    refine fold_mids R xs _ m ?_
    unfold passBody
    split_ifs
    · exact hm
    · exact strongStep_mids C R A x m hm

theorem segRun_mids (R : CutReads α) :
    ∀ (segs : List (List (FreeMonoid α) × C.Draws)) (A : RoundAcc α),
      ∀ m ∈ A.s.tree.mids, m ∈ (segRun C R A segs).s.tree.mids
  | [], _, _, hm => hm
  | seg :: rest, A, m, hm => by
    simp only [segRun]
    refine segRun_mids R rest _ m ?_
    rw [strongPass_eq]
    exact fold_mids C R seg.1 _ m hm

section Congr

variable {B : State} {F : Finset (FreeMonoid α)} {f₁ f₂ : FreeMonoid α → ℝ}

theorem strongStep_congr {A : RoundAcc α} {x : FreeMonoid α} (Tf : DTree α)
    (hT : ∀ m ∈ A.s.tree.mids, m ∈ Tf.mids)
    (hT' : ∀ m ∈ (strongStep C (rd B F f₁) A x).s.tree.mids, m ∈ Tf.mids)
    (hpool : ∀ b ∈ A.s.pool, AgreeOne C.K F f₁ f₂ Tf b)
    (hwit : ∀ p c q y, A.s.edges p c = some (q, y) → AgreeOne C.K F f₁ f₂ Tf y)
    (hw : ∀ i, C.k ≤ i → AgreeOne C.K F f₁ f₂ Tf (prefixOf x i))
    (htest : StepTestsAgree C.K B F f₁ f₂ C.k A.s x) :
    strongStep C (rd B F f₁) A x = strongStep C (rd B F f₂) A x := by
  have hT'p : ∀ m ∈ (probeStepK C.K (rd B F f₁) C.k A.s x).tree.mids, m ∈ Tf.mids := by
    rw [← (strongStep_state C _ A x).1]; exact hT'
  have hps := probeStepK_congr Tf hT hT'p hpool hwit hw htest
  have hpool' : ∀ b ∈ A.s.pool, AgreeOne C.K F f₁ f₂ A.s.tree b :=
    fun b hb => (hpool b hb).mono' hT
  have hwit' : ∀ p c q y, A.s.edges p c = some (q, y) → AgreeOne C.K F f₁ f₂ A.s.tree y :=
    fun p c q y h => (hwit p c q y h).mono' hT
  have hw' : ∀ i, C.k ≤ i → AgreeOne C.K F f₁ f₂ A.s.tree (prefixOf x i) :=
    fun i hi => (hw i hi).mono' hT
  have hpo : probeOutcome (rd B F f₁) A.s.tree A.s.edges C.k x
      = probeOutcome (rd B F f₂) A.s.tree A.s.edges C.k x :=
    probeOutcome_congr fun i hi m hm => (hw' i hi).tree m hm
  have hsk : stepSkip C.K (rd B F f₁) C.k A.s x = stepSkip C.K (rd B F f₂) C.k A.s x := rfl
  unfold strongStep
  simp only []
  rw [hps, hpo, hsk]
  split
  · rename_i ps fd heq
    have hfd : C.k ≤ fd - 1 := by have := probeOutcome_edge_gt _ heq; omega
    rw [seedStep_congr hpool' hwit' (hw' _ hfd) fun κ hk hf => htest ps fd κ heq hk hf]
  · rfl

theorem strongStep_poolIn {Bs : Set (FreeMonoid α)} (R : CutReads α) {A : RoundAcc α}
    {x : FreeMonoid α} (hs : KPoolIn Bs A.s) (hw : ∀ i, C.k ≤ i → prefixOf x i ∈ Bs) :
    KPoolIn Bs (strongStep C R A x).s := by
  rw [(strongStep_state C R A x).1]
  exact probeStepK_poolIn C.K R hs hw

theorem fold_congr {Bs : Set (FreeMonoid α)} (Tf : DTree α)
    (hB : ∀ b ∈ Bs, AgreeOne C.K F f₁ f₂ Tf b) (hBblk : ∀ b ∈ Bs, AgreeBlk C.K F f₁ f₂ Tf b) :
    ∀ (probes : List (FreeMonoid α)) (A : RoundAcc α), KPoolIn Bs A.s →
      (∀ x ∈ probes, ∀ i, C.k ≤ i → prefixOf x i ∈ Bs) →
      (∀ m ∈ (probes.foldl (passBody C (rd B F f₁)) A).s.tree.mids, m ∈ Tf.mids) →
      probes.foldl (passBody C (rd B F f₁)) A = probes.foldl (passBody C (rd B F f₂)) A := by
  intro probes
  induction probes with
  | nil => intro A _ _ _; simp only [List.foldl_nil]
  | cons x xs ih =>
    intro A hA hws hT
    simp only [List.foldl_cons] at hT ⊢
    have hx := hws x (List.mem_cons_self ..)
    by_cases hg : C.K.patience ≤ A.s.streak ∨ C.budget A.s.tree.paths.length ≤ A.used
    · have h1 : passBody C (rd B F f₁) A x = A := by unfold passBody; rw [if_pos hg]
      have h2 : passBody C (rd B F f₂) A x = A := by unfold passBody; rw [if_pos hg]
      rw [h1] at hT ⊢
      rw [h2]
      exact ih A hA (fun y hy => hws y (List.mem_cons_of_mem _ hy)) hT
    · have h1 : passBody C (rd B F f₁) A x = strongStep C (rd B F f₁) A x := by
        unfold passBody; rw [if_neg hg]
      have h2 : passBody C (rd B F f₂) A x = strongStep C (rd B F f₂) A x := by
        unfold passBody; rw [if_neg hg]
      rw [h1] at hT ⊢
      have hTs : ∀ m ∈ (strongStep C (rd B F f₁) A x).s.tree.mids, m ∈ Tf.mids :=
        fun m hm => hT m (fold_mids C _ xs _ m hm)
      have hstep := strongStep_congr C Tf (fun m hm => hTs m (strongStep_mids C _ A x m hm)) hTs
        (fun b hb => hB b (hA.1 b hb)) (fun p c q y h => hB y (hA.2 p c q y h))
        (fun i hi => hB _ (hx i hi))
        (stepTests_of_agreeTests (agreeTests_of_blk
          (fun m hm => hTs m (strongStep_mids C _ A x m hm)) fun b hb => hBblk b (hA.1 b hb)))
      rw [h2, ← hstep]
      exact ih _ (strongStep_poolIn C _ hA hx) (fun y hy => hws y (List.mem_cons_of_mem _ hy)) hT

theorem segRun_congr {Bs : Set (FreeMonoid α)} (Tf : DTree α)
    (hB : ∀ b ∈ Bs, AgreeOne C.K F f₁ f₂ Tf b) (hBblk : ∀ b ∈ Bs, AgreeBlk C.K F f₁ f₂ Tf b) :
    ∀ (segs : List (List (FreeMonoid α) × C.Draws)) (A : RoundAcc α), KPoolIn Bs A.s →
      (∀ x ∈ (segs.map Prod.fst).flatten, ∀ i, C.k ≤ i → prefixOf x i ∈ Bs) →
      (∀ m ∈ (segRun C (rd B F f₁) A segs).s.tree.mids, m ∈ Tf.mids) →
      segRun C (rd B F f₁) A segs = segRun C (rd B F f₂) A segs
  | [], _, _, _, _ => by simp only [segRun]
  | seg :: rest, A, hA, hws, hT => by
    simp only [segRun] at hT ⊢
    have hseg : ∀ x ∈ seg.1, ∀ i, C.k ≤ i → prefixOf x i ∈ Bs := fun x hx =>
      hws x (List.mem_flatten.2 ⟨seg.1, List.mem_cons_self .., hx⟩)
    have hpass : strongPass C (rd B F f₁) A seg.1 = strongPass C (rd B F f₂) A seg.1 := by
      rw [strongPass_eq, strongPass_eq]
      exact fold_congr C Tf hB hBblk seg.1 _ hA hseg fun m hm =>
        hT m (segRun_mids C _ rest _ m (by rw [strongPass_eq]; exact hm))
    have hA' : KPoolIn Bs (strongPass C (rd B F f₁) A seg.1).s := by
      rw [strongPass_eq]
      have : ∀ (l : List (FreeMonoid α)) (X : RoundAcc α), KPoolIn Bs X.s →
          (∀ x ∈ l, ∀ i, C.k ≤ i → prefixOf x i ∈ Bs) →
          KPoolIn Bs (l.foldl (passBody C (rd B F f₁)) X).s := by
        intro l
        induction l with
        | nil => exact fun X h _ => h
        | cons x xs ih =>
          intro X h hl
          simp only [List.foldl_cons]
          refine ih _ ?_ fun y hy => hl y (List.mem_cons_of_mem _ hy)
          unfold passBody
          split_ifs
          · exact h
          · exact strongStep_poolIn C _ h (hl x (List.mem_cons_self ..))
      exact this seg.1 _ hA hseg
    rw [← hpass]
    exact segRun_congr Tf hB hBblk rest _ hA'
      (fun x hx => hws x (List.mem_flatten.2 (by
        obtain ⟨l, hl, hxl⟩ := List.mem_flatten.1 hx
        exact ⟨l, List.mem_cons_of_mem _ hl, hxl⟩))) hT

end Congr

end Segments

end OrthoDFA
