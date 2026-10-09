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

open scoped Classical in
/-- How many leaves the reference's states reach. -/
noncomputable def classCount (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α)
    (t : DTree α) : ℕ :=
  (Finset.univ.image fun q => sidePath side t (rep q)).card

theorem classCount_le (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α) (t : DTree α) :
    classCount side rep t ≤ Fintype.card Q := by
  classical
  exact Finset.card_image_le.trans (by simp)

open scoped Classical in
/-- A split at a leaf raises the count by one where it separates two states, else keeps it. -/
theorem classCount_splitAt (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α)
    (r : SplitRec α) (hp : r.leaf ∈ r.tree.paths) :
    classCount side rep r.tree + (if r.Separates side rep then 1 else 0)
      ≤ classCount side rep (r.tree.splitAt r.d r.leaf) := by
  classical
  set p := r.leaf
  set g : Q → List Bool := fun q => sidePath side r.tree (rep q)
  set g' : Q → List Bool := fun q => sidePath side (r.tree.splitAt r.d p) (rep q)
  set π : List Bool → List Bool := fun z => if z = p ++ [true] ∨ z = p ++ [false] then p else z
  have hg' : ∀ q, g' q = if g q = p then p ++ [side (rep q * r.d)] else g q := fun q =>
    sidePath_splitAt side r.d r.tree p hp (rep q)
  have hπ : ∀ q, π (g' q) = g q := fun q => by
    rw [hg']
    split_ifs with h
    · simp only [π]
      rw [if_pos (by cases side (rep q * r.d) <;> simp)]
      exact h.symm
    · have hm : g q ∈ r.tree.paths := sidePath_mem_paths side r.tree (rep q)
      simp only [π]
      rw [if_neg]
      rintro (he | he) <;> rw [he] at hm
      · exact DTree.append_not_mem_paths r.tree p true hp hm
      · exact DTree.append_not_mem_paths r.tree p false hp hm
  have himg : Finset.univ.image g = (Finset.univ.image g').image π := by
    rw [Finset.image_image]
    congr 1
    funext q
    exact (hπ q).symm
  unfold classCount
  change (Finset.univ.image g).card + _ ≤ (Finset.univ.image g').card
  rw [himg]
  split_ifs with hs
  · obtain ⟨q, q', hq, hq', hne⟩ := hs
    have hlt : ((Finset.univ.image g').image π).card < (Finset.univ.image g').card := by
      refine lt_of_le_of_ne Finset.card_image_le fun he => ?_
      have hinj := Finset.card_image_iff.1 he
      have h1 : g' q ∈ Finset.univ.image g' := Finset.mem_image_of_mem _ (Finset.mem_univ _)
      have h2 : g' q' ∈ Finset.univ.image g' := Finset.mem_image_of_mem _ (Finset.mem_univ _)
      have := hinj h1 h2 (by rw [hπ, hπ]; exact hq.trans hq'.symm)
      rw [hg', hg', if_pos hq, if_pos hq'] at this
      exact hne (by simpa using this)
    omega
  · simpa using Finset.card_image_le

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

theorem noisy_of_not_separates {r : SplitRec α} (hok : SplitOK R r)
    (hs : ¬ r.Separates side rep) : r.Noisy R A side rep := by
  by_contra hn
  simp only [SplitRec.Noisy, not_or, not_not] at hn
  obtain ⟨h1, h2, h3, h4⟩ := hn
  obtain ⟨hy, hsp, b, hby, hbs⟩ := hok
  refine hs ⟨A.state r.y, A.state r.sprime, ?_, ?_, fun he => ?_⟩
  · have := faithfulRoute_sift r.tree r.y h1
    rw [hy] at this
    exact (Sum.inl.inj this).symm
  · have := faithfulRoute_sift r.tree r.sprime h2
    rw [hsp] at this
    exact (Sum.inl.inj this).symm
  · unfold Faithful at h3 h4
    rw [hby] at h3
    rw [hbs] at h4
    rw [← Option.some.inj h3, ← Option.some.inj h4] at he
    cases b <;> simp at he

theorem same_state_unfaithful {r : SplitRec α} (hok : SplitOK R r)
    (hq : A.state r.y = A.state r.sprime) :
    ¬ Faithful R A side rep r.y r.d ∨ ¬ Faithful R A side rep r.sprime r.d := by
  by_contra hn
  simp only [not_or, not_not] at hn
  obtain ⟨h3, h4⟩ := hn
  obtain ⟨-, -, b, hby, hbs⟩ := hok
  unfold Faithful at h3 h4
  rw [hby] at h3
  rw [hbs, ← hq] at h4
  have := (Option.some.inj h3).trans (Option.some.inj h4).symm
  cases b <;> simp at this

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
  rcases hs : seedStep C.K R A.s.tree A.s.pool A.s.edges (stepSkip C.K R C.k A.s x) A.s.forced
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

theorem startAcc_inv (seed : List (FreeMonoid α)) : StrongInv R (startAcc C R seed) :=
  ⟨closeEdges_learned C.K R fun _ _ _ _ he => by simp at he,
    by simp [startAcc, initialK, closeK, DTree.paths]⟩

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

/-- A property of the tree, edges and splits that every step keeps holds of the round's end. -/
theorem strongRound_preserves (P : RoundAcc α → Prop)
    (hcongr : ∀ A B : RoundAcc α, A.s.tree = B.s.tree → A.s.edges = B.s.edges →
      A.splits = B.splits → P A → P B)
    (hstep : ∀ A x, StrongInv R A → P A → P (strongStep C R A x)) :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      StrongInv R A → P A →
        StrongInv R (strongRound C R n j A first d).2.1
          ∧ P (strongRound C R n j A first d).2.1 := by
  have hfold : ∀ (probes : List (FreeMonoid α)) (A : RoundAcc α), StrongInv R A → P A →
      StrongInv R (probes.foldl (passBody C R) A) ∧ P (probes.foldl (passBody C R) A) := by
    intro probes
    induction probes with
    | nil => exact fun A h1 h2 => ⟨h1, h2⟩
    | cons x xs ih =>
      intro A h1 h2
      simp only [List.foldl_cons]
      unfold passBody
      split_ifs
      · exact ih A h1 h2
      · exact ih _ (strongStep_inv C R A x h1) (hstep A x h1 h2)
  have hread : ∀ j A first (y : C.Draws), StrongInv R A → P A →
      StrongInv R (strongReading C R j A first y).1 ∧ P (strongReading C R j A first y).1 := by
    intro j A first y h1 h2
    obtain ⟨hs, hsp, -⟩ := strongReading_fst C R j A first y
    rw [strongPass_eq] at hs hsp
    obtain ⟨g1, g2⟩ := hfold (first ++ List.ofFn y.1)
      { A with s := { A.s with streak := 0, log := A.s.log ∪ A.reads } } h1
      (hcongr A _ rfl rfl rfl h2)
    refine ⟨⟨?_, ?_⟩, hcongr _ _ (by rw [hs]) (by rw [hs]) (by rw [hsp]) g2⟩
    · rw [hs]; exact g1.1
    · rw [hs, hsp]; exact g1.2
  intro n
  induction n with
  | zero => exact fun _ A _ _ h1 h2 => ⟨h1, h2⟩
  | succ n ih =>
    intro j A first d h1 h2
    obtain ⟨g1, g2⟩ := hread j A first (d 0) h1 h2
    rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩ <;>
      rw [hR] at g1 g2 <;> simp only [strongRound, hR]
    · exact ⟨g1, g2⟩
    · exact ih _ _ _ _ g1 g2

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

theorem entry_start (seed : List (FreeMonoid α)) : Entry C R (startAcc C R seed) 0 :=
  ⟨startAcc_inv C R seed, by simp [startAcc], by simp [startAcc]⟩

theorem round_strong_readings : RoundStrongReadings := by
  intro α _ _ C R seed Rmax d hp hb
  obtain ⟨-, h2, h3, -⟩ := round_used C R hp hb Rmax 0 (startAcc C R seed) [] d
    (entry_start C R seed)
  unfold strongRun
  simp only [Nat.zero_add] at h3
  omega

theorem round_strong_leaf_paths : RoundStrongLeafPaths := by
  intro α _ _ C R seed Rmax d
  have hl := (strongRound_preserves C R (fun _ => True) (fun _ _ _ _ _ _ => trivial)
    (fun _ _ _ _ => trivial) Rmax 0 (startAcc C R seed) [] d (startAcc_inv C R seed) trivial).1.1
  intro p c q w he
  obtain ⟨h1, h2⟩ := hl p c q w he
  exact ⟨DTree.sift_mem_paths _ _ _ h1, DTree.sift_mem_paths _ _ _ h2⟩

theorem round_strong_same_state : RoundStrongSameState := by
  intro α _ _ Q C R A side rep seed Rmax d
  have h := (strongRound_preserves C R (fun B => ∀ r ∈ B.splits, SplitOK R r)
    (fun A B _ _ hsp hA => hsp ▸ hA) (fun A x hI hA => by
      rcases strongStep_cases C R A x hI.1 with ⟨hsp, -⟩ | ⟨r, hsp, -, -, -, hok⟩
      · rw [hsp]; exact hA
      · rw [hsp]
        intro r' hr'
        rcases List.mem_append.1 hr' with h | h
        · exact hA r' h
        · rw [List.mem_singleton.1 h]; exact hok)
    Rmax 0 (startAcc C R seed) [] d (startAcc_inv C R seed) (by simp [startAcc])).2
  exact fun r hr hq => same_state_unfaithful R A side rep (h r hr) hq

theorem round_strong_leaves : RoundStrongLeaves := by
  classical
  intro α _ _ Q _ C R A side rep seed Rmax d
  have h := (strongRound_preserves C R
    (fun B => (∀ r ∈ B.splits, SplitOK R r) ∧ B.s.tree.paths.length
      ≤ classCount side rep B.s.tree + 2 + (B.splits.filter fun r => ¬ r.Separates side rep).length)
    (fun A B ht _ hsp hA => by rw [← ht, ← hsp]; exact hA) (fun A x hI hA => by
      obtain ⟨hok, hc⟩ := hA
      rcases strongStep_cases C R A x hI.1 with ⟨hsp, ht⟩ |
        ⟨r, hsp, hrt, hrl, ht, hokr⟩
      · rw [hsp, ht]; exact ⟨hok, hc⟩
      · rw [hsp, ht]
        refine ⟨fun r' hr' => ?_, ?_⟩
        · rcases List.mem_append.1 hr' with h | h
          · exact hok r' h
          · rw [List.mem_singleton.1 h]; exact hokr
        · have hcs := classCount_splitAt side rep r (hrt ▸ hrl)
          rw [hrt] at hcs
          rw [DTree.splitAt_paths_length _ _ _ hrl, List.filter_append, List.length_append]
          by_cases hs : r.Separates side rep
          · rw [if_pos hs] at hcs
            simp only [hs, not_true_eq_false, decide_false, List.filter_cons_of_neg,
              List.filter_nil, List.length_nil, Bool.false_eq_true, not_false_eq_true]
            omega
          · rw [if_neg hs] at hcs
            simp only [hs, not_false_eq_true, decide_true, List.filter_singleton, cond_true,
              List.length_singleton]
            omega)
    Rmax 0 (startAcc C R seed) [] d (startAcc_inv C R seed)
    ⟨by simp [startAcc], by simp [startAcc, initialK, closeK, DTree.paths]⟩).2
  obtain ⟨hok, hc⟩ := h
  have hcq := classCount_le side rep (strongRun C R seed Rmax d).2.1.s.tree
  have hle : ((strongRun C R seed Rmax d).2.1.splits.filter fun r => ¬ r.Separates side rep).length
      ≤ noisySplits R A side rep (strongRun C R seed Rmax d).2.1.splits := by
    unfold noisySplits
    rw [← List.countP_eq_length_filter, ← List.countP_eq_length_filter]
    refine List.countP_mono_left fun r hr h => ?_
    simp only [decide_eq_true_eq] at h ⊢
    exact noisy_of_not_separates R A side rep (hok r hr) h
  unfold strongRun at hle hcq ⊢
  omega

end Round

end OrthoDFA

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Q : Type*}

section StrongReading

variable (C : StrongCfg α) (R : CutReads α)

theorem strongPass_certs (Ac : RoundAcc α) (probes : List (FreeMonoid α)) :
    (strongPass C R Ac probes).certs = Ac.certs := by
  rw [strongPass_eq]
  have : ∀ (l : List (FreeMonoid α)) (X : RoundAcc α),
      (l.foldl (passBody C R) X).certs = X.certs := by
    intro l
    induction l with
    | nil => exact fun _ => rfl
    | cons x xs ih =>
      intro X
      simp only [List.foldl_cons]
      rw [ih]
      unfold passBody
      split_ifs
      · rfl
      · exact (strongStep_state C R X x).2.2
  exact this _ _

/-- Whether reading `j`'s gate settles above, so that it calls the certificate. -/
def CallsCert (j : ℕ) (Ac : RoundAcc α) (first : List (FreeMonoid α))
    (pr : Fin C.np → FreeMonoid α) (bg : Fin C.ng → FreeMonoid α) : Prop :=
  gateSide R (strongPass C R Ac (first ++ List.ofFn pr)).s.tree
    (strongPass C R Ac (first ++ List.ofFn pr)).s.edges bg C.acc (C.a / 2 ^ j)
    (gateStop R (strongPass C R Ac (first ++ List.ofFn pr)).s.tree
      (strongPass C R Ac (first ++ List.ofFn pr)).s.edges bg C.acc (C.a / 2 ^ j)) = some true

open scoped Classical in
theorem strongReading_certs (j : ℕ) (Ac : RoundAcc α) (first : List (FreeMonoid α))
    (y : C.Draws) :
    (strongReading C R j Ac first y).1.certs
      = Ac.certs + if CallsCert C R j Ac first y.1 y.2.1 then 1 else 0 := by
  unfold strongReading CallsCert
  simp only []
  split_ifs <;> simp_all [strongPass_certs]

variable (A : DFA (FreeMonoid α) Q) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
  (CertGood : DTree α → Edges α → Prop)

/-- A reading's failure chance but for the certificate's. -/
noncomputable def strongBound (ν : ℝ) (j : ℕ) : ℝ :=
  2 * (Nat.log 2 (C.ng / 30) + 2) * (C.a / 2 ^ j) + (1 - ν) ^ C.nr

open scoped Classical in
/-- Given its probes, a reading from an entry ends the round badly with chance at most
`strongBound`, and the certificate's failure chance where its gate settles above. -/
theorem strong_section_le {αs : ℕ → ℝ} {η minCov ν : ℝ} (hacc0 : 0 ≤ C.acc) (hacc1 : C.acc ≤ 1)
    (hf : 0 ≤ C.f) (ha : 0 ≤ C.a) (hν : ν ≤ 1)
    (hcert : ∀ i t e (g : FreeMonoid α → ℝ), (Measure.pi fun _ : Fin C.nc => D).real
      {cs | C.cert i t e cs (fun j => g (cs j)) = true ∧ ¬ CertGood t e} ≤ αs i)
    {Ac : RoundAcc α} {first : List (FreeMonoid α)}
    (j Rmax : ℕ) (pr : Fin C.np → FreeMonoid α) :
    ((Measure.pi fun _ : Fin C.ng => D).prod ((Measure.pi fun _ : Fin C.nr => D).prod
        (Measure.pi fun _ : Fin C.nc => D))).real
      {z | ∃ e, (strongReading C R j Ac first (pr, z)).2 = .inl e
        ∧ ¬ StrongEndHolds C R A D CertGood η minCov ν Rmax (strongReading C R j Ac first (pr, z)).1
          e}
      ≤ strongBound C ν j
        + (Measure.pi fun _ : Fin C.ng => D).real {bg | CallsCert C R j Ac first pr bg}
          * αs Ac.certs := by
  set A' := strongPass C R Ac (first ++ List.ofFn pr) with hA'
  set s := A'.s with hs
  set aj := C.a / 2 ^ j
  have haj : 0 ≤ aj := by positivity
  have hcerts : A'.certs = Ac.certs := strongPass_certs C R Ac _
  obtain ⟨Bad, hBad, hgood⟩ := gate_settled R s D C.ng hacc0 hacc1 haj
  set gu : List Bool × α → Prop := fun _ => False
  set Miss := {br : Fin C.nr → FreeMonoid α | ν < D.real {x | NAOff R s.tree s.edges C.k gu x}
    ∧ ∀ i, ¬ NAOff R s.tree s.edges C.k gu (br i)}
  set Called := {bg | CallsCert C R j Ac first pr bg}
  set CB := {cs : Fin C.nc → FreeMonoid α |
    C.cert Ac.certs s.tree s.edges cs (fun i => R.f (cs i)) = true ∧ ¬ CertGood s.tree s.edges}
  have hsub : {z | ∃ e, (strongReading C R j Ac first (pr, z)).2 = .inl e
      ∧ ¬ StrongEndHolds C R A D CertGood η minCov ν Rmax (strongReading C R j Ac first (pr, z)).1
        e}
      ⊆ Bad ×ˢ Set.univ ∪ Set.univ ×ˢ (Miss ×ˢ Set.univ) ∪ Called ×ˢ (Set.univ ×ˢ CB) := by
    rintro ⟨bg, br, cs⟩ ⟨e, he, hne⟩
    by_cases hbad : bg ∈ Bad
    · exact .inl (.inl ⟨hbad, trivial⟩)
    obtain ⟨hpass, hrefuse⟩ := hgood bg hbad
    unfold strongReading at he hne
    simp only [← hA', ← hs] at he hne
    split_ifs at he hne with h1 h2 h3 h4 h5 h6 <;>
      simp only [Sum.inl.injEq] at he <;> subst he
    all_goals dsimp only at hne
    all_goals first
      | exact absurd h1.1 h2
      | exact absurd trivial hne
      | (simp only [StrongEndHolds, not_and_or] at hne
         rcases hne with hne | hne
         · exact absurd (hpass h1.1) hne
         · exact .inr ⟨h1.1, trivial, by rw [hcerts] at h1; exact h1.2, hne⟩)
      | (simp only [StrongEndHolds, not_or, not_and_or, not_le] at hne
         obtain ⟨htau, hM, hne⟩ := hne
         have hB : ¬ ∃ i : Fin C.nr, (i : ℕ) < refusalStop br C.a
             (harvestTests R s.tree s.edges C.k C.L C.f C.c C.θM)
             (LiveEdge R s.tree s.edges C.k gu) ∧ LiveEdge R s.tree s.edges C.k gu (br i) := by
           rintro ⟨i, hi, hl⟩
           apply h5
           rw [Ne, List.map_eq_nil_iff, List.filter_eq_nil_iff]
           push Not
           refine ⟨i, List.mem_finRange _, ?_⟩
           first | exact ⟨hi, hl⟩ | exact decide_eq_true ⟨hi, hl⟩
         have hclear := refusal_clear R s.tree s.edges C.k C.L hf ha gu br htau hM hB h3
         refine .inl (.inr ⟨trivial, ?_, trivial⟩)
         refine ⟨?_, hclear⟩
         rcases hne with hne | hne
         · exact hne
         · simp only [Classical.not_imp, not_forall, decide_eq_true_eq] at hne
           obtain ⟨hgr, q₀, h, hq₀, hhq, hcov, hh, hneed⟩ := hne
           have hall := hrefuse hgr
           have herr : 1 - C.acc < D.real {x | StartDis R s.edges (h q₀) x} := by
             have := hall _ hhq
             rw [show {x | ¬ StartDis R s.edges (h q₀) x} = {x | StartDis R s.edges (h q₀) x}ᶜ
               from rfl, measureReal_compl (Set.to_countable _).measurableSet,
               probReal_univ] at this
             linarith
           have hNA := need_lt R A D _ h s.tree s.edges C.k q₀ gu hcov hh herr
           have hneed' : ν < need R A D (Covered A D C.k C.L minCov) h s.tree s.edges C.k q₀ gu
               C.acc η := lt_of_not_ge hneed
           linarith)
  set ν₁ := Measure.pi fun _ : Fin C.ng => D
  set ν₂ := Measure.pi fun _ : Fin C.nr => D
  set ν₃ := Measure.pi fun _ : Fin C.nc => D
  have hMiss := miss_all_le D (NAOff R s.tree s.edges C.k gu) C.nr hν
  have hCB : ν₃.real CB ≤ αs Ac.certs := hcert Ac.certs s.tree s.edges R.f
  calc (ν₁.prod (ν₂.prod ν₃)).real {z | ∃ e, (strongReading C R j Ac first (pr, z)).2 = .inl e
        ∧ ¬ StrongEndHolds C R A D CertGood η minCov ν Rmax
          (strongReading C R j Ac first (pr, z)).1 e}
      ≤ (ν₁.prod (ν₂.prod ν₃)).real
          (Bad ×ˢ Set.univ ∪ Set.univ ×ˢ (Miss ×ˢ Set.univ) ∪ Called ×ˢ (Set.univ ×ˢ CB)) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ (ν₁.prod (ν₂.prod ν₃)).real (Bad ×ˢ Set.univ)
        + (ν₁.prod (ν₂.prod ν₃)).real (Set.univ ×ˢ (Miss ×ˢ Set.univ))
        + (ν₁.prod (ν₂.prod ν₃)).real (Called ×ˢ (Set.univ ×ˢ CB)) :=
        (measureReal_union_le _ _).trans (add_le_add (measureReal_union_le _ _) le_rfl)
    _ = ν₁.real Bad + ν₂.real Miss + ν₁.real Called * ν₃.real CB := by
        simp only [measureReal_prod_prod, probReal_univ, mul_one, one_mul]
    _ ≤ strongBound C ν j + ν₁.real Called * αs Ac.certs := by
        unfold strongBound
        have := mul_le_mul_of_nonneg_left hCB (measureReal_nonneg (μ := ν₁) (s := Called))
        linarith

theorem strongRound_len_certs :
    ∀ (n j : ℕ) (Ac : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      (strongRound C R n j Ac first d).2.2.length ≤ n
        ∧ Ac.certs ≤ (strongRound C R n j Ac first d).2.1.certs
        ∧ (strongRound C R n j Ac first d).2.1.certs ≤ Ac.certs + n := by
  intro n
  induction n with
  | zero => intro j Ac first d; simp [strongRound]
  | succ n ih =>
    intro j Ac first d
    have hc := strongReading_certs C R j Ac first (d 0)
    have hc1 : (strongReading C R j Ac first (d 0)).1.certs ≤ Ac.certs + 1 := by
      rw [hc]; split_ifs <;> omega
    rcases hR : strongReading C R j Ac first (d 0) with ⟨A'', e | lv⟩ <;>
      rw [hR] at hc hc1 <;> simp only [strongRound, hR] <;> dsimp only at hc hc1
    · refine ⟨by simp, by omega, by omega⟩
    · obtain ⟨h1, h2, h3⟩ := ih (j + 1) A'' lv (Fin.tail d)
      refine ⟨by simp only [List.length_cons]; omega, by omega, by omega⟩

instance StrongCfg.drawMeasure_prob (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] :
    IsProbabilityMeasure (C.drawMeasure D) := by
  unfold StrongCfg.drawMeasure; infer_instance

open scoped Classical in
/-- The certificate's failure chance a reading spends: on the calls it makes. -/
noncomputable def certCost (αs : ℕ → ℝ) (c₀ c₁ : ℕ) : ℝ≥0∞ :=
  ENNReal.ofReal (∑ i ∈ Finset.Ico c₀ c₁, αs i)

theorem certCost_split {αs : ℕ → ℝ} (hα : ∀ i, 0 ≤ αs i) {c₀ c₁ c₂ : ℕ} (h₁ : c₀ ≤ c₁)
    (h₂ : c₁ ≤ c₂) : certCost αs c₀ c₂ = certCost αs c₀ c₁ + certCost αs c₁ c₂ := by
  unfold certCost
  rw [← Finset.sum_Ico_consecutive _ h₁ h₂, ENNReal.ofReal_add (Finset.sum_nonneg fun i _ => hα i)
    (Finset.sum_nonneg fun i _ => hα i)]

open scoped Classical in
/-- A reading from an entry ends the round badly with chance at most `strongBound`, and the
certificate's failure chance on the calls it makes. -/
theorem strong_reading_bad_le {αs : ℕ → ℝ} {η minCov ν : ℝ} (hacc0 : 0 ≤ C.acc)
    (hacc1 : C.acc ≤ 1) (hf : 0 ≤ C.f) (ha : 0 ≤ C.a) (hν : ν ≤ 1)
    (hcert : ∀ i t e (g : FreeMonoid α → ℝ), (Measure.pi fun _ : Fin C.nc => D).real
      {cs | C.cert i t e cs (fun j => g (cs j)) = true ∧ ¬ CertGood t e} ≤ αs i)
    {Ac : RoundAcc α} {first : List (FreeMonoid α)} (j Rmax : ℕ) :
    C.drawMeasure D {y | ∃ e, (strongReading C R j Ac first y).2 = .inl e
        ∧ ¬ StrongEndHolds C R A D CertGood η minCov ν Rmax (strongReading C R j Ac first y).1 e}
      ≤ ENNReal.ofReal (strongBound C ν j)
        + ∫⁻ y, certCost αs Ac.certs (strongReading C R j Ac first y).1.certs
          ∂C.drawMeasure D := by
  have hα : 0 ≤ αs Ac.certs := measureReal_nonneg.trans (hcert Ac.certs Ac.s.tree Ac.s.edges R.f)
  have hβ : 0 ≤ strongBound C ν j := by
    unfold strongBound
    have := pow_nonneg (by linarith : (0 : ℝ) ≤ 1 - ν) C.nr
    positivity
  have hK : ∀ pr, ∫⁻ z, certCost αs Ac.certs (strongReading C R j Ac first (pr, z)).1.certs
      ∂((Measure.pi fun _ : Fin C.ng => D).prod ((Measure.pi fun _ : Fin C.nr => D).prod
        (Measure.pi fun _ : Fin C.nc => D)))
      = ENNReal.ofReal ((Measure.pi fun _ : Fin C.ng => D).real
          {bg | CallsCert C R j Ac first pr bg} * αs Ac.certs) := by
    intro pr
    have hpt : ∀ z, certCost αs Ac.certs (strongReading C R j Ac first (pr, z)).1.certs
        = ({bg | CallsCert C R j Ac first pr bg} ×ˢ (Set.univ : Set _)).indicator
          (fun _ => ENNReal.ofReal (αs Ac.certs)) z := by
      intro z
      rw [strongReading_certs]
      unfold certCost
      by_cases hc : CallsCert C R j Ac first pr z.1
      · rw [if_pos hc, Set.indicator_of_mem (by exact ⟨hc, trivial⟩)]
        simp
      · rw [if_neg hc, Set.indicator_of_notMem (by exact fun h => hc h.1)]
        simp
    simp only [hpt]
    rw [lintegral_indicator_const (Set.to_countable _).measurableSet, Measure.prod_prod,
      measure_univ, mul_one, ENNReal.ofReal_mul measureReal_nonneg, ofReal_measureReal, mul_comm]
  unfold StrongCfg.drawMeasure
  rw [Measure.prod_apply (Set.to_countable _).measurableSet,
    lintegral_prod _ (measurable_of_countable _).aemeasurable]
  simp only [hK]
  rw [← (show ∫⁻ _pr : Fin C.np → FreeMonoid α, ENNReal.ofReal (strongBound C ν j)
      ∂(Measure.pi fun _ : Fin C.np => D) = ENNReal.ofReal (strongBound C ν j) by simp),
    ← lintegral_add_left measurable_const]
  refine lintegral_mono fun pr => ?_
  have := strong_section_le C R A D CertGood (η := η) (minCov := minCov) hacc0 hacc1 hf ha hν hcert
    (Ac := Ac) (first := first) j Rmax pr
  rw [← ENNReal.ofReal_toReal (measure_ne_top _ _), ← ENNReal.ofReal_add hβ
    (mul_nonneg measureReal_nonneg hα)]
  exact ENNReal.ofReal_le_ofReal this

open scoped Classical in
/-- Over `n` readings from an entry at reading `j`, the round ends badly with chance at most the
expected sum of `strongBound` over the readings it makes, and the certificate's failure chance on
the calls it makes. -/
theorem strong_round_bad {αs : ℕ → ℝ} {η minCov ν : ℝ} (hacc0 : 0 ≤ C.acc) (hacc1 : C.acc ≤ 1)
    (hf : 0 ≤ C.f) (ha : 0 ≤ C.a) (hν : ν ≤ 1)
    (hcert : ∀ i t e (g : FreeMonoid α → ℝ), (Measure.pi fun _ : Fin C.nc => D).real
      {cs | C.cert i t e cs (fun j => g (cs j)) = true ∧ ¬ CertGood t e} ≤ αs i)
    (hp : C.K.patience ≤ C.np) (hb : Monotone C.budget) (Rmax : ℕ) :
    ∀ (n j : ℕ) (Ac : RoundAcc α) (first : List (FreeMonoid α)), Entry C R Ac j →
      j + n = Rmax →
      (Measure.pi fun _ : Fin n => C.drawMeasure D)
          {d | ¬ StrongEndHolds C R A D CertGood η minCov ν Rmax
            (strongRound C R n j Ac first d).2.1 (strongRound C R n j Ac first d).1}
        ≤ ∫⁻ d, (∑ i ∈ Finset.range (strongRound C R n j Ac first d).2.2.length,
              ENNReal.ofReal (strongBound C ν (j + i)))
            + certCost αs Ac.certs (strongRound C R n j Ac first d).2.1.certs
          ∂(Measure.pi fun _ : Fin n => C.drawMeasure D) := by
  have hα : ∀ i, 0 ≤ αs i := fun i =>
    measureReal_nonneg.trans (hcert i .leaf (fun _ _ => none) R.f)
  intro n
  induction n with
  | zero =>
    intro j Ac first hE hjn
    rw [show {d : Fin 0 → C.Draws | ¬ StrongEndHolds C R A D CertGood η minCov ν Rmax
        (strongRound C R 0 j Ac first d).2.1 (strongRound C R 0 j Ac first d).1} = ∅ from
      Set.eq_empty_of_forall_notMem fun d hd => hd ?_]
    · simp
    · simp only [strongRound, StrongEndHolds]
      rw [← hjn, Nat.add_zero]
      exact hE.2.1.trans hE.2.2
  | succ n ih =>
    intro j Ac first hE hjn
    set ρm := C.drawMeasure D
    set μn := Measure.pi fun _ : Fin n => ρm
    set β : ℕ → ℝ≥0∞ := fun i => ENNReal.ofReal (strongBound C ν i)
    have hmp := measurePreserving_piFinSuccAbove (fun _ : Fin (n + 1) => ρm) 0
    set e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => C.Draws) 0
    set F : C.Draws × (Fin n → C.Draws) → StrongEnd × RoundAcc α × List (ℕ × ℕ) := fun p =>
      match strongReading C R j Ac first p.1 with
      | (A', .inl e) => (e, A', [(A'.s.tree.paths.length, A'.used)])
      | (A', .inr lv) =>
        ((strongRound C R n (j + 1) A' lv p.2).1, (strongRound C R n (j + 1) A' lv p.2).2.1,
          (A'.s.tree.paths.length, A'.used) :: (strongRound C R n (j + 1) A' lv p.2).2.2)
    have hF : ∀ d, strongRound C R (n + 1) j Ac first d = F (e d) := by
      intro d
      rw [piFinSuccAbove_zero]
      rfl
    set G : C.Draws → ℝ≥0∞ := fun y => match strongReading C R j Ac first y with
      | (_, .inl _) => 0
      | (A', .inr lv) => ∫⁻ d', (∑ i ∈ Finset.range (strongRound C R n (j + 1) A' lv d').2.2.length,
          β (j + 1 + i)) + certCost αs A'.certs (strongRound C R n (j + 1) A' lv d').2.1.certs ∂μn
    set K : C.Draws → ℝ≥0∞ := fun y =>
      certCost αs Ac.certs (strongReading C R j Ac first y).1.certs
    set Bad := {y | ∃ e, (strongReading C R j Ac first y).2 = .inl e
      ∧ ¬ StrongEndHolds C R A D CertGood η minCov ν Rmax (strongReading C R j Ac first y).1 e}
    have hcount : ∀ (T : Set (C.Draws × (Fin n → C.Draws))), MeasurableSet T :=
      fun T => (Set.to_countable T).measurableSet
    have hL : (Measure.pi fun _ : Fin (n + 1) => ρm)
        {d | ¬ StrongEndHolds C R A D CertGood η minCov ν Rmax
          (strongRound C R (n + 1) j Ac first d).2.1 (strongRound C R (n + 1) j Ac first d).1}
        ≤ ∫⁻ y, Bad.indicator 1 y + G y ∂ρm := by
      have hpre : {d | ¬ StrongEndHolds C R A D CertGood η minCov ν Rmax
          (strongRound C R (n + 1) j Ac first d).2.1 (strongRound C R (n + 1) j Ac first d).1}
          = e ⁻¹' {p | ¬ StrongEndHolds C R A D CertGood η minCov ν Rmax (F p).2.1 (F p).1} := by
        ext d; simp only [Set.mem_ofPred_eq, Set.mem_preimage, hF]
      rw [hpre, hmp.measure_preimage (hcount _).nullMeasurableSet,
        Measure.prod_apply (hcount _)]
      refine lintegral_mono fun y => ?_
      have hnext := (reading_entry C R hp hb hE first y).2.2.2
      simp only [F, G, Set.preimage, Set.mem_ofPred_eq]
      rcases hstep : strongReading C R j Ac first y with ⟨A', e' | lv⟩
      · simp only []
        by_cases hb : StrongEndHolds C R A D CertGood η minCov ν Rmax A' e'
        · simp [hb]
        · have hy : y ∈ Bad := ⟨e', by rw [hstep], by rw [hstep]; exact hb⟩
          simp only [hb, not_false_eq_true, Set.setOf_true, measure_univ, add_zero,
            Set.indicator_of_mem hy, Pi.one_apply, le_refl]
      · simp only []
        rw [hstep] at hnext
        refine le_trans ?_ le_add_self
        exact ih (j + 1) A' lv (hnext lv rfl) (by omega)
    set Hf : C.Draws × (Fin n → C.Draws) → ℝ≥0∞ := fun p =>
      (∑ i ∈ Finset.range (F p).2.2.length, β (j + i)) + certCost αs Ac.certs (F p).2.1.certs
    have hR : ∫⁻ d, (∑ i ∈ Finset.range (strongRound C R (n + 1) j Ac first d).2.2.length,
            β (j + i)) + certCost αs Ac.certs (strongRound C R (n + 1) j Ac first d).2.1.certs
          ∂(Measure.pi fun _ : Fin (n + 1) => ρm)
        = ∫⁻ y, β j + (K y + G y) ∂ρm := by
      calc _ = ∫⁻ d, Hf (e d) ∂(Measure.pi fun _ : Fin (n + 1) => ρm) :=
            lintegral_congr fun d => by simp only [Hf, hF]
        _ = ∫⁻ p, Hf p ∂(ρm.prod μn) := hmp.lintegral_comp (measurable_of_countable Hf)
        _ = ∫⁻ y, ∫⁻ d', Hf (y, d') ∂μn ∂ρm :=
            lintegral_prod _ (measurable_of_countable Hf).aemeasurable
        _ = ∫⁻ y, β j + (K y + G y) ∂ρm := by
          refine lintegral_congr fun y => ?_
          have hc := strongReading_certs C R j Ac first y
          simp only [Hf, F, G, K]
          rcases hstep : strongReading C R j Ac first y with ⟨A', e' | lv⟩
          · simp
          · rw [hstep] at hc
            dsimp only at hc
            simp only []
            have hca : ∀ (c : ℝ≥0∞) (g : (Fin n → C.Draws) → ℝ≥0∞),
                c + ∫⁻ d', g d' ∂μn = ∫⁻ d', c + g d' ∂μn := fun c g => by
              rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one]
            rw [hca, hca]
            refine lintegral_congr fun d' => ?_
            obtain ⟨-, h2, -⟩ := strongRound_len_certs C R n (j + 1) A' lv d'
            rw [List.length_cons, Finset.sum_range_succ', add_zero,
              certCost_split hα (show Ac.certs ≤ A'.certs by rw [hc]; omega) h2]
            simp only [add_assoc, add_comm, add_left_comm]
    have hbad := strong_reading_bad_le C R A D CertGood (αs := αs) (η := η) (minCov := minCov)
      hacc0 hacc1 hf ha hν hcert (Ac := Ac) (first := first) j Rmax
    calc _ ≤ ∫⁻ y, Bad.indicator 1 y + G y ∂ρm := hL
      _ = ρm Bad + ∫⁻ y, G y ∂ρm := by
          rw [lintegral_add_left (measurable_of_countable _), lintegral_indicator_one
            (Set.to_countable _).measurableSet]
      _ ≤ (β j + ∫⁻ y, K y ∂ρm) + ∫⁻ y, G y ∂ρm := add_le_add hbad le_rfl
      _ = ∫⁻ y, β j + (K y + G y) ∂ρm := by
          rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
            lintegral_add_left (measurable_of_countable _), add_assoc]
      _ = _ := hR.symm

end StrongReading




theorem round_strong_trichotomy : RoundStrongTrichotomy := by
  intro α _ _ Q C A D _ seed CertGood αs η minCov ν Rmax hacc0 hacc1 hf ha hν hp hb hcert R
  have hα : ∀ i, 0 ≤ αs i := fun i =>
    measureReal_nonneg.trans (hcert i .leaf (fun _ _ => none) R.f)
  set P := Measure.pi fun _ : Fin Rmax => C.drawMeasure D
  set N : (Fin Rmax → C.Draws) → ℕ := fun d => (strongRun C R seed Rmax d).2.2.length
  set M : (Fin Rmax → C.Draws) → ℕ := fun d => (strongRun C R seed Rmax d).2.1.certs
  set S : (Fin Rmax → C.Draws) → ℝ := fun d => ∑ i ∈ Finset.range (M d), αs i
  set γ : ℝ := (1 - ν) ^ C.nr
  set c₀ : ℝ := 2 * (Nat.log 2 (C.ng / 30) + 2)
  have hc₀ : 0 ≤ c₀ := by positivity
  have hγ : 0 ≤ γ := pow_nonneg (by linarith) _
  have hlc : ∀ d, N d ≤ Rmax ∧ M d ≤ Rmax := fun d => by
    obtain ⟨h1, -, h3⟩ := strongRound_len_certs C R Rmax 0 (startAcc C R seed) [] d
    have h0 : (startAcc C R seed).certs = 0 := rfl
    exact ⟨h1, by simp only [M, strongRun]; omega⟩
  have hS0 : ∀ d, 0 ≤ S d := fun d => Finset.sum_nonneg fun i _ => hα i
  have hSle : ∀ d, S d ≤ ∑ i ∈ Finset.range Rmax, αs i := fun d =>
    Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.2 (hlc d).2)
      fun i _ _ => hα i
  have hNint : Integrable (fun d => (N d : ℝ)) P :=
    Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable Rmax
      (ae_of_all _ fun d => by
        rw [Real.norm_of_nonneg (Nat.cast_nonneg _)]; exact_mod_cast (hlc d).1)
  have hSint : Integrable S P :=
    Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable _
      (ae_of_all _ fun d => by rw [Real.norm_of_nonneg (hS0 d)]; exact hSle d)
  have h := strong_round_bad C R A D CertGood (αs := αs) (η := η) (minCov := minCov) hacc0 hacc1
    hf ha hν hcert hp hb Rmax Rmax 0 (startAcc C R seed) [] (entry_start C R seed)
    (by omega)
  have hsb : ∀ i, strongBound C ν i = c₀ * (C.a / 2 ^ i) + γ := fun i => rfl
  have hpt : ∀ d, (∑ i ∈ Finset.range (N d), ENNReal.ofReal (strongBound C ν (0 + i)))
      + certCost αs (startAcc C R seed).certs (M d)
      ≤ ENNReal.ofReal (2 * c₀ * C.a + N d * γ + S d) := by
    intro d
    have hnn : ∀ i, 0 ≤ strongBound C ν (0 + i) := fun i => by rw [hsb]; positivity
    rw [← ENNReal.ofReal_sum_of_nonneg (fun i _ => hnn i)]
    unfold certCost
    rw [show (startAcc C R seed).certs = 0 from rfl, ← Finset.range_eq_Ico,
      ← ENNReal.ofReal_add (Finset.sum_nonneg fun i _ => hnn i) (hS0 d)]
    refine ENNReal.ofReal_le_ofReal ?_
    simp only [hsb, zero_add, Finset.sum_add_distrib, Finset.sum_const, Finset.card_range,
      nsmul_eq_mul]
    have hgeo : ∑ i ∈ Finset.range (N d), c₀ * (C.a / 2 ^ i)
        = c₀ * C.a * ∑ i ∈ Finset.range (N d), (1 / 2 : ℝ) ^ i := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [one_div_pow]; ring
    have := sum_geometric_two_le (N d)
    rw [hgeo]
    nlinarith [mul_nonneg hc₀ ha]
  have hi1 : Integrable (fun d => 2 * c₀ * C.a + (N d : ℝ) * γ) P :=
    (integrable_const _).add (hNint.mul_const _)
  have hint : Integrable (fun d => 2 * c₀ * C.a + (N d : ℝ) * γ + S d) P := hi1.add hSint
  have hlin : ∫⁻ d, ENNReal.ofReal (2 * c₀ * C.a + N d * γ + S d) ∂P
      = ENNReal.ofReal (2 * c₀ * C.a + (∫ d, (N d : ℝ) ∂P) * γ + ∫ d, S d ∂P) := by
    rw [← ofReal_integral_eq_lintegral_ofReal hint (ae_of_all _ fun d => by
        have := hS0 d; positivity),
      integral_add hi1 hSint,
      integral_add (integrable_const _) (hNint.mul_const _), integral_const, integral_mul_const]
    simp
  have hrhs0 : 0 ≤ 2 * c₀ * C.a + (∫ d, (N d : ℝ) ∂P) * γ + ∫ d, S d ∂P := by
    have := integral_nonneg (μ := P) (f := fun d => (N d : ℝ)) fun d => Nat.cast_nonneg _
    have := integral_nonneg (μ := P) (f := S) hS0
    positivity
  have hle : P {d | ¬ StrongEndHolds C R A D CertGood η minCov ν Rmax
      (strongRun C R seed Rmax d).2.1 (strongRun C R seed Rmax d).1}
      ≤ ENNReal.ofReal (2 * c₀ * C.a + (∫ d, (N d : ℝ) ∂P) * γ + ∫ d, S d ∂P) :=
    h.trans ((lintegral_mono hpt).trans hlin.le)
  rw [show 4 * ((Nat.log 2 (C.ng / 30) : ℝ) + 2) * C.a = 2 * c₀ * C.a by simp only [c₀]; ring]
  rw [measureReal_def]
  exact ENNReal.toReal_le_of_le_ofReal hrhs0 hle

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

/-- The round's passes over the segments `segs` are decided by the oracle's bits at what they
can read against the tree they end with. -/
theorem segRun_determined {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (seed : List (FreeMonoid α)) (segs : List (List (FreeMonoid α) × C.Draws)) :
    PassDetermined O B F C.K C.k seed (segs.map Prod.fst).flatten
      fun R => (segRun C R (startAcc C R seed) segs).s := by
  intro ω ω' h
  set probes := (segs.map Prod.fst).flatten
  set Bs : Set (FreeMonoid α) := {b | b ∈ seed ++ probes.flatMap fun p =>
    (List.range (p.toList.length + 1)).map fun i => prefixOf p (max C.k i)}
  set Tf := (segRun C (readsAt O B F ω) (startAcc C (readsAt O B F ω) seed) segs).s.tree
  have hseed : ∀ b ∈ seed, b ∈ Bs := fun b hb => List.mem_append_left _ hb
  have hws : ∀ w ∈ probes, ∀ i, C.k ≤ i → prefixOf w i ∈ Bs := fun w hw i hi =>
    prefixOf_mem_bases hw hi
  have hAll : ∀ b ∈ Bs, ∀ e, ∀ m ∈ Tf.mids,
      AgreeAll C.K F (fun w => O.mq w ω) (fun w => O.mq w ω') (b * ext e * m) := by
    intro b hb e m hm
    refine agree_of_noise C.K O F fun v hv => h _ ?_
    simp only [readsOf, passOf, vBits, Finset.mem_biUnion, Finset.mem_image]
    refine ⟨b * ext e * m, ?_, v, hv, rfl⟩
    simp only [passReadSet, Finset.mem_image, Finset.mem_product, Prod.exists]
    refine ⟨b, ext e, m, ⟨⟨List.mem_toFinset.2 hb, ?_⟩, ?_⟩, rfl⟩
    · rcases e with _ | c
      · exact Finset.mem_insert_self _ _
      · exact Finset.mem_insert_of_mem (Finset.mem_image.2 ⟨c, Finset.mem_univ _, rfl⟩)
    · exact Finset.mem_insert_of_mem ((DTree.mem_midfixes_iff _).2 hm)
  have hB : ∀ b ∈ Bs, AgreeOne C.K F (fun w => O.mq w ω) (fun w => O.mq w ω') Tf b :=
    fun b hb e m hm => (hAll b hb e m hm).agree
  have hBblk : ∀ b ∈ Bs, AgreeBlk C.K F (fun w => O.mq w ω) (fun w => O.mq w ω') Tf b := by
    intro b hb c m hm v hv
    have := hAll b hb (some c) m hm v (C.K.block_sub F hv)
    simpa [ext, mul_assoc] using this
  have h0mids : ∀ m ∈ (startAcc C (readsAt O B F ω) seed).s.tree.mids, m ∈ Tf.mids :=
    fun m hm => segRun_mids C _ segs _ m hm
  have h0 : startAcc C (readsAt O B F ω) seed = startAcc C (readsAt O B F ω') seed := by
    simp only [startAcc, initialK, closeK] at h0mids ⊢
    rw [show readsAt O B F ω = rd B F (fun w => O.mq w ω) from rfl,
      show readsAt O B F ω' = rd B F (fun w => O.mq w ω') from rfl,
      closeEdges_congr fun b hb => (hB b (hseed b hb)).mono' h0mids]
  have hpool0 : KPoolIn Bs (startAcc C (readsAt O B F ω) seed).s :=
    closeK_poolIn C.K _ hseed fun _ _ _ _ h => by simp at h
  have := segRun_congr C (B := B) (F := F) (f₁ := fun w => O.mq w ω) (f₂ := fun w => O.mq w ω')
    Tf hB hBblk segs _ hpool0 hws fun m hm => hm
  change (segRun C (readsAt O B F ω') (startAcc C (readsAt O B F ω') seed) segs).s
    = (segRun C (readsAt O B F ω) (startAcc C (readsAt O B F ω) seed) segs).s
  rw [← h0]
  exact (congrArg RoundAcc.s this).symm

end Segments

theorem round_strong_quality : RoundStrongQuality := by
  intro α _ _ Ω _ μ _ Q C A O B F D _ seed δ Rmax d hf hc hδ hδ1 hkL hlen hV
  have hpm := prefixMax_pos D hkL hlen
  have hlog : 0 < Real.log (5 / δ) := Real.log_pos (by rw [lt_div_iff₀ hδ]; linarith)
  have heps : ∀ r, 0 < strongEps C D δ r := fun r => by
    unfold strongEps
    refine Real.sqrt_pos.2 (div_pos (mul_pos hpm ?_) two_pos)
    have : 0 ≤ (((C.nr + 1) * r + r + 1 : ℕ) : ℝ) * Real.log 2 :=
      mul_nonneg (Nat.cast_nonneg _) (Real.log_nonneg (by norm_num))
    linarith
  have hq := fun (r : ℕ) (segs : List (List (FreeMonoid α) × C.Draws)) => quality_holds_of
    (μ := μ) A O B F C.K D hkL seed (segs.map Prod.fst).flatten _
    (segRun_determined C O B F seed segs) hf hc (heps r) hlen hV
  choose Ef hEf hgood using hq
  set E : Set Ω := ⋃ r ∈ Finset.range (Rmax + 1), ⋃ p ∈ candS C Rmax d r {[]}, Ef r p
  refine ⟨E, ?_, fun ω hω => ?_⟩
  · have hterm : ∀ r, ∑ p ∈ candS C Rmax d r {[]}, μ.real (Ef r p) ≤ δ / 2 ^ (r + 1) := by
      intro r
      have hcard : ((candS C Rmax d r {[]}).card : ℝ) ≤ 2 ^ ((C.nr + 1) * r) := by
        have := card_candS C Rmax d r {[]}
        simp only [Finset.card_singleton, max_self, one_mul] at this
        exact_mod_cast this
      have hexp : 5 * Real.exp (-2 * strongEps C D δ r ^ 2 / prefixMax D C.k)
          = δ / 2 ^ ((C.nr + 1) * r + r + 1) := by
        unfold strongEps
        rw [Real.sq_sqrt (by
          have : 0 ≤ (((C.nr + 1) * r + r + 1 : ℕ) : ℝ) * Real.log 2 :=
            mul_nonneg (Nat.cast_nonneg _) (Real.log_nonneg (by norm_num))
          positivity)]
        rw [show -2 * (prefixMax D C.k * ((((C.nr + 1) * r + r + 1 : ℕ) : ℝ) * Real.log 2
            + Real.log (5 / δ)) / 2) / prefixMax D C.k
            = -((((C.nr + 1) * r + r + 1 : ℕ) : ℝ) * Real.log 2) - Real.log (5 / δ) by
          field_simp; ring]
        rw [Real.exp_sub, Real.exp_neg, ← Real.log_rpow two_pos, Real.exp_log (by positivity),
          Real.exp_log (by positivity), Real.rpow_natCast]
        field_simp
      calc ∑ p ∈ candS C Rmax d r {[]}, μ.real (Ef r p)
          ≤ ∑ _p ∈ candS C Rmax d r {[]}, δ / 2 ^ ((C.nr + 1) * r + r + 1) :=
            Finset.sum_le_sum fun p _ => (hEf r p).trans (le_of_eq hexp)
        _ = (candS C Rmax d r {[]}).card * (δ / 2 ^ ((C.nr + 1) * r + r + 1)) := by
            rw [Finset.sum_const, nsmul_eq_mul]
        _ ≤ 2 ^ ((C.nr + 1) * r) * (δ / 2 ^ ((C.nr + 1) * r + r + 1)) :=
            mul_le_mul_of_nonneg_right hcard (by positivity)
        _ = δ / 2 ^ (r + 1) := by
            rw [pow_add, pow_add (2 : ℝ) ((C.nr + 1) * r)]
            field_simp
            ring
    calc μ.real E ≤ ∑ r ∈ Finset.range (Rmax + 1), μ.real (⋃ p ∈ candS C Rmax d r {[]}, Ef r p) :=
          measureReal_biUnion_finset_le _ _
      _ ≤ ∑ r ∈ Finset.range (Rmax + 1), δ / 2 ^ (r + 1) :=
          Finset.sum_le_sum fun r _ => (measureReal_biUnion_finset_le _ _).trans (hterm r)
      _ ≤ δ := by
          have h := sum_geometric_two_le (Rmax + 1)
          have : ∑ r ∈ Finset.range (Rmax + 1), δ / 2 ^ (r + 1)
              = δ / 2 * ∑ r ∈ Finset.range (Rmax + 1), (1 / 2 : ℝ) ^ r := by
            rw [Finset.mul_sum]
            refine Finset.sum_congr rfl fun r _ => ?_
            rw [one_div_pow, pow_succ]; ring
          rw [this]
          nlinarith
  · simp only []
    set R := readsAt O B F ω
    obtain ⟨segs, hs, hp, hst⟩ := strongRound_segs C R Rmax 0 (startAcc C R seed) [] d {[]}
      (Finset.mem_singleton_self _)
    have hr : (strongRun C R seed Rmax d).2.2.length ∈ Finset.range (Rmax + 1) :=
      Finset.mem_range.2 (Nat.lt_succ_of_le
        (strongRound_len_certs C R Rmax 0 (startAcc C R seed) [] d).1)
    simp only [E, Set.mem_iUnion, not_exists] at hω
    have := hgood _ _ ω (hω _ hr _ hs)
    unfold strongRun
    rw [hst, hp]
    exact this

end OrthoDFA
