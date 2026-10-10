import OrthoDFA.Proofs.RandomProbe
import OrthoDFA.Proofs.RandomCounts

/-!
# One stretch

A step is bad where, within the first `Ns` probes of its stretch, it ends the round badly, or
where its stretch reaches `Ns` probes. Over a stretch the hypothesis is fixed, so each probe's
outcome is a function of the probe alone, and the stretch's counts are counts over fresh draws
(`StInv`). A bad step then shows in the counts (`seg_reduce`): a test that harvested mostly good
strings has at least half its count in probes that can read a good read-state undecided, a false
success has few disagreements, and a stretch that reaches `Ns` probes has fewer than `m` records
at every edge and target, no success at look `N₁`, and no harvest of undecided middles at look
`Ns`. `seg_le` bounds each by its binomial tail.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory
open OrthoDFA.Ideal (DTree Edges Disagrees Outcome EdgesOK OutcomeOK)

section Segments

variable {S X E : Type*} (step : S → X → S ⊕ E) (B : S → X → Prop) (same : S → S → Prop)

/-- Some step before the run leaves the segment is bad. -/
def SegB : S → List X → Prop
  | _, [] => False
  | s, x :: xs => B s x ∨ ∃ s', step s x = .inl s' ∧ same s s' ∧ SegB s' xs

/-- Some step of the run is bad. -/
def AnyB : S → List X → Prop
  | _, [] => False
  | s, x :: xs => B s x ∨ ∃ s', step s x = .inl s' ∧ AnyB s' xs

end Segments

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Cfg)
  {σ : Type*} [Fintype σ] (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) (D : Measure (FreeMonoid α))
  (ε : ℝ) (Ns : ℕ)

def Same (s s' : RState α) : Prop := s'.tree = s.tree ∧ s'.edges = s.edges

/-- Within the first `Ns` probes of its stretch the step ends the round badly, or its stretch
reaches `Ns` probes. -/
def BadStep (s : RState α) (x : FreeMonoid α) : Prop :=
  (s.n < Ns ∧ ∃ e, step read C s x = .inr e ∧ ¬ EndsWell M (BadAt U θ) D ε
      {y | Disagrees read s.tree s.edges C.k y} (toRoundEnd (some e)))
  ∨ ∃ s', step read C s x = .inl s' ∧ Same s s' ∧ s'.n = Ns

/-! ## A step within a stretch -/

/-- The state a probe leaves before the tests look at it. -/
def chg (s : RState α) (x : FreeMonoid α) : RState α :=
  charge read C { s with recs := s.recs ++ (recOf (probeR read s.tree s.edges C.k x)).toList } x

section StepFacts

variable {σ' : Type*} [Fintype σ'] {M' : DFA α σ'} {side : σ' → Bool}

theorem finish_inr {t : RState α} {e : REnd α} (h : finish C t = .inr e) : look C t = some e := by
  unfold finish at h
  split at h
  · rename_i e' he'; simp only [Sum.inr.injEq] at h; rw [he', h]
  · simp at h

theorem mk_eta (s : RState α) :
    (⟨s.tree, s.edges, s.moved, s.recs, s.n, s.dis, s.startH, s.pt, s.reads, s.und⟩ : RState α)
      = s := rfl

theorem chg_of_none {s : RState α} {x : FreeMonoid α}
    (h : recOf (probeR read s.tree s.edges C.k x) = none) : chg read C s x = charge read C s x := by
  unfold chg
  rw [h]
  simp only [Option.toList_none, List.append_nil]

theorem step_inr (hW : NoWrong M' side read) (hcap : Fintype.card σ' + 2 ≤ C.Lmax)
    (hm : 1 ≤ C.m) {s : RState α} (hs : Inv read C s) {x : FreeMonoid α} {e : REnd α}
    (h : step read C s x = .inr e) : look C (chg read C s x) = some e := by
  have hne := step_ne_tooBig read C hW hcap hm hs x
  have h' := h
  unfold step at h
  rcases ho : probeR read s.tree s.edges C.k x with _ | z | z | ⟨p, c, u, t⟩ | ⟨k, zs⟩ | ⟨k, zs⟩ |
    ⟨p, c, u, t⟩
  all_goals rw [ho] at h
  all_goals try (rw [chg_of_none read C (by rw [ho]; rfl)]; exact finish_inr C h)
  swap
  · simp at h
  · unfold chg
    rw [ho]
    simp only [recOf, Option.toList_some]
    simp only at h
    split_ifs at h with hc
    · exfalso
      have : e = .tooBig := by
        unfold fix at h
        split at h
        · split_ifs at h
          · unfold split at h
            simp only at h
            split_ifs at h
            simp only [Sum.inr.injEq] at h
            exact h.symm
        · simp at h
      exact hne (this ▸ h')
    · exact finish_inr C h

theorem step_same_chg (hW : NoWrong M' side read) {s s' : RState α} (hs : Inv read C s)
    {x : FreeMonoid α} (h : step read C s x = .inl s') (hsm : Same s s') :
    s' = chg read C s x ∧ look C s' = none ∧ Inv read C s' := by
  have hinv := (step_same read C hW hs h hsm.1 hsm.2).1
  rcases step_cases read C hs h with
    ⟨p, c, u, t, -, hE, -, -, -, rfl⟩ | ⟨p, c, u, t, t₀, w₀, -, hE, hne, hp, -, -, -, hfix⟩ |
    ⟨h1, h2, -⟩
  · exfalso
    have := congrFun (congrFun hsm.2 p) c
    simp only [fresh, Ideal.upd2_same, hE] at this
    exact absurd this (by simp)
  · exfalso
    unfold fix at hfix
    rw [hE] at hfix
    simp only at hfix
    split_ifs at hfix with hmv
    · unfold split at hfix
      simp only at hfix
      split_ifs at hfix
      simp only [Sum.inl.injEq] at hfix
      subst hfix
      have := congrArg List.length (congrArg DTree.leaves hsm.1)
      simp only [fresh] at this
      rw [DTree.length_leaves_splitAt _ _ _ hp] at this
      omega
    · simp only [Sum.inl.injEq] at hfix
      subst hfix
      have := congrFun (congrFun hsm.2 p) c
      simp only [fresh, Ideal.upd2_same, hE, Option.some.injEq, Prod.mk.injEq] at this
      exact hne this.1.symm
  · exact ⟨h1, h2, hinv⟩

theorem look_none {t : RState α} (h : look C t = none) :
    rateSide C.εd C.a C.n₀ t.n t.dis ≠ some false
      ∧ ¬ (C.qg * t.n ≤ t.dis ∧ rateSide C.θpt C.a C.n₀ t.dis t.pt.length = some true) := by
  unfold look at h
  split_ifs at h with h1 h2 h3 h4
  exact ⟨h4, h3⟩

theorem look_some {t : RState α} {e : REnd α} (h : look C t = some e) :
    (rateSide C.θs C.a C.n₀ t.n t.startH.length = some true ∧ e = .harvest t.startH)
    ∨ (∃ p c, C.τe * t.n ≤ t.reads p c ∧ C.exc + C.θe * t.reads p c ≤ (t.und p c).length
        ∧ e = .harvest (t.und p c))
    ∨ (C.qg * t.n ≤ t.dis ∧ rateSide C.θpt C.a C.n₀ t.dis t.pt.length = some true
        ∧ e = .harvest t.pt)
    ∨ (rateSide C.εd C.a C.n₀ t.n t.dis = some false ∧ e = .consistent) := by
  unfold look at h
  split_ifs at h with h1 h2 h3 h4
  · exact Or.inl ⟨h1, (Option.some.inj h).symm⟩
  · obtain ⟨-, h5⟩ := h2.choose_spec
    exact Or.inr (Or.inl ⟨_, _, h5.1, h5.2, (Option.some.inj h).symm⟩)
  · exact Or.inr (Or.inr (Or.inl ⟨h3.1, h3.2, (Option.some.inj h).symm⟩))
  · exact Or.inr (Or.inr (Or.inr ⟨h4, (Option.some.inj h).symm⟩))

end StepFacts

/-! ## Good strings -/

open scoped Classical in
/-- How many of the strings are at good read-states. -/
noncomputable def goodCount (zs : List (FreeMonoid α)) : ℕ :=
  (zs.filter fun z => ¬ BadAt U θ (M.eval z.toList)).length

theorem len_le_two_good {zs : List (FreeMonoid α)} {dis : Set (FreeMonoid α)}
    (h : ¬ EndsWell M (BadAt U θ) D ε dis (.harvest zs)) : zs.length ≤ 2 * goodCount M U θ zs := by
  classical
  simp only [EndsWell, not_lt] at h
  unfold goodCount
  have := List.length_eq_length_filter_add (l := zs)
    (f := fun z => decide (BadAt U θ (M.eval z.toList)))
  have e : (zs.filter fun z => !decide (BadAt U θ (M.eval z.toList)))
      = zs.filter fun z => decide (¬ BadAt U θ (M.eval z.toList)) := by
    congr 1; ext z; simp
  rw [e] at this
  omega

theorem goodCount_append (l l' : List (FreeMonoid α)) :
    goodCount M U θ (l ++ l') = goodCount M U θ l + goodCount M U θ l' := by
  simp [goodCount, List.filter_append]

theorem goodCount_le (l : List (FreeMonoid α)) : goodCount M U θ l ≤ l.length :=
  List.length_filter_le _ _

open scoped Classical in
/-- Strings a probe can read undecided count as good only where the probe can read a good
read-state undecided. -/
theorem goodCount_le_of {k : ℕ} {x : FreeMonoid α} {T : DTree α} {l : List (FreeMonoid α)}
    (hl : ∀ z ∈ l, z ∈ pot k x T ∧ read z = .undecided) :
    goodCount M U θ l ≤ l.length * (if x ∈ PotGood M U θ k read T then 1 else 0) := by
  classical
  split_ifs with hx
  · rw [mul_one]; exact goodCount_le M U θ l
  · rw [mul_zero, Nat.le_zero]
    unfold goodCount
    rw [List.length_eq_zero_iff, List.filter_eq_nil_iff]
    intro z hz hg
    exact hx ⟨z, (hl z hz).1, by simpa using hg, (hl z hz).2⟩

/-! ## The stretch's counts -/

section Counts

variable (T : DTree α) (E : Edges α)

def fS (x : FreeMonoid α) : Bool := searches (probeR read T E C.k x)

def fP (x : FreeMonoid α) : Bool := !(ptStr (probeR read T E C.k x)).isEmpty

open scoped Classical in
noncomputable def fG (x : FreeMonoid α) : Bool := decide (x ∈ PotGood M U θ C.k read T)

open scoped Classical in
noncomputable def fDis (x : FreeMonoid α) : Bool := decide (Disagrees read T E C.k x)

def fR (τ : List Bool × α × List Bool) (x : FreeMonoid α) : Bool :=
  decide (recOf (probeR read T E C.k x) = some τ)

variable (L N₁ : ℕ)

/-- After the probes `ys` of a stretch at `(T, E)`, the counts are counts over `ys`. -/
structure StInv (ys : List (FreeMonoid α)) (s : RState α) : Prop where
  tree : s.tree = T
  edges : s.edges = E
  n : s.n = ys.length
  dis : s.dis = ys.countP (fS read C T E)
  pt : s.pt.length = ys.countP (fP read C T E)
  ptg : goodCount M U θ s.pt ≤ ys.countP (fG read C M U θ T)
  stg : goodCount M U θ s.startH ≤ ys.countP (fG read C M U θ T)
  undg : (∀ y ∈ ys, y.toList.length ≤ L) →
    ∀ p c, goodCount M U θ (s.und p c) ≤ (L + 1) * ys.countP (fG read C M U θ T)
  recs : ∀ τ, s.recs.count τ = ys.countP (fR read C T E τ)

/-- The stretch's success test did not fire at look `N₁`. -/
def Look₁ (ys : List (FreeMonoid α)) : Prop :=
  N₁ ≤ ys.length → rateSide C.εd C.a C.n₀ N₁ ((ys.take N₁).countP (fS read C T E)) ≠ some false

theorem stInv_fresh (mv : List Bool → α → Bool) :
    StInv read C M U θ T E L [] (fresh T E mv) where
  tree := rfl
  edges := rfl
  n := rfl
  dis := rfl
  pt := rfl
  ptg := by simp [goodCount, fresh]
  stg := by simp [goodCount, fresh]
  undg := fun _ p c => by simp [goodCount, fresh]
  recs := fun τ => by simp [fresh]

theorem countP_snoc (f : FreeMonoid α → Bool) (ys : List (FreeMonoid α)) (x : FreeMonoid α) :
    (ys ++ [x]).countP f = ys.countP f + if f x then 1 else 0 := by
  simp [List.countP_append]

theorem length_le_one_eq {X : Type*} (l : List X) (h : l.length ≤ 1) :
    l.length = if (!l.isEmpty) = true then 1 else 0 := by
  rcases l with _ | ⟨z, _ | ⟨w, l⟩⟩ <;> simp_all

theorem ite_bool_le (b : Bool) (P : Prop) [Decidable P] (h : P → b = true) :
    (if P then 1 else 0) ≤ if b then 1 else 0 := by
  split_ifs with h1 h2 <;> simp_all

/-- A probe counted into a stretch whose counts are its probes' keeps them so. -/
theorem stInv_chg {ys : List (FreeMonoid α)} {s : RState α} (hs : Inv read C s)
    (hst : StInv read C M U θ T E L ys s) (x : FreeMonoid α) :
    StInv read C M U θ T E L (ys ++ [x]) (chg read C s x) := by
  classical
  obtain ⟨htree, hedges, hn, hdis, hpt, hptg, hstg, hundg, hrecs⟩ := hst
  subst htree hedges
  have hOK := probeR_ok read hs.edges C.k x
  have hG : (if x ∈ PotGood M U θ C.k read s.tree then 1 else 0)
      = if fG read C M U θ s.tree x then 1 else 0 := by
    simp [fG]
  refine ⟨rfl, rfl, by simp [chg, charge, hn], ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [chg, charge, hdis, countP_snoc]
    cases hb : searches (probeR read s.tree s.edges C.k x) <;> simp [fS, hb]
  · simp only [chg, charge, List.length_append, hpt, countP_snoc]
    rw [length_le_one_eq _ (ptStr_length _)]
    cases hb : (ptStr (probeR read s.tree s.edges C.k x)).isEmpty <;> simp [fP, hb]
  · simp only [chg, charge, goodCount_append, countP_snoc]
    have h1 := goodCount_le_of read M U θ (probeR_pt read s.tree x s.edges C.k)
    have h2 := ptStr_length (probeR read s.tree s.edges C.k x)
    rw [hG] at h1
    have : (ptStr (probeR read s.tree s.edges C.k x)).length
        * (if fG read C M U θ s.tree x then 1 else 0) ≤ if fG read C M U θ s.tree x then 1 else 0 := by
      split_ifs <;> omega
    omega
  · simp only [chg, charge, goodCount_append, countP_snoc]
    have h1 := goodCount_le_of read M U θ (probeR_start read s.tree x s.edges C.k)
    have h2 := startStr_length (probeR read s.tree s.edges C.k x)
    rw [hG] at h1
    have : (startStr (probeR read s.tree s.edges C.k x)).length
        * (if fG read C M U θ s.tree x then 1 else 0) ≤ if fG read C M U θ s.tree x then 1 else 0 := by
      split_ifs <;> omega
    omega
  · intro hL p c
    have hL' : ∀ y ∈ ys, y.toList.length ≤ L := fun y hy => hL y (List.mem_append_left _ hy)
    have hx : x.toList.length ≤ L := hL x (by simp)
    simp only [chg, charge, goodCount_append, countP_snoc]
    have h0 := hundg hL' p c
    have hsub : ∀ z ∈ undAt (charges read s.tree s.edges C.k x) p c,
        z ∈ pot C.k x s.tree ∧ read z = .undecided := by
      intro z hz
      obtain ⟨e, he, hez⟩ := undAt_sub _ p c z hz
      exact charges_und read s.tree x s.edges C.k e he z hez
    have h1 := goodCount_le_of read M U θ hsub
    have h2 := undAt_length (charges read s.tree s.edges C.k x) p c
    have h3 := charges_length read s.tree x s.edges C.k
    rw [hG] at h1
    have : (undAt (charges read s.tree s.edges C.k x) p c).length
        * (if fG read C M U θ s.tree x then 1 else 0)
        ≤ (L + 1) * if fG read C M U θ s.tree x then 1 else 0 := by
      split_ifs <;> omega
    rw [Nat.mul_add]
    omega
  · intro τ
    simp only [chg, charge, List.count_append, hrecs, countP_snoc]
    cases hr : recOf (probeR read s.tree s.edges C.k x) with
    | none => simp [fR, hr]
    | some τ' => by_cases h : τ' = τ <;> simp [fR, hr, h, List.count_singleton]

theorem look₁_snoc {ys : List (FreeMonoid α)} {s : RState α} {x : FreeMonoid α}
    (hst : StInv read C M U θ T E L (ys ++ [x]) s) (h₀ : Look₁ read C T E N₁ ys)
    (h₁ : N₁ = ys.length + 1 → rateSide C.εd C.a C.n₀ N₁ s.dis ≠ some false) :
    Look₁ read C T E N₁ (ys ++ [x]) := by
  intro hN
  by_cases hlt : N₁ ≤ ys.length
  · rw [List.take_append_of_le_length hlt]
    exact h₀ hlt
  · have hN' : N₁ = ys.length + 1 := by simp at hN; omega
    rw [List.take_of_length_le (by simp; omega), ← hst.dis]
    exact h₁ hN'

end Counts

/-! ## A bad step shows in the counts -/

section Reduce

variable (T : DTree α) (E : Edges α) (L N₁ : ℕ)

/-- A test harvested mostly good strings at look `n`, or success fired falsely. -/
def FalseFire (zs : List (FreeMonoid α)) : Prop :=
  (C.n₀ ≤ zs.length ∧
      binomSfGe zs.length C.θs (2 * zs.countP (fG read C M U θ T)) < C.a)
  ∨ (∃ y ∈ zs, L < y.toList.length)
  ∨ C.exc + C.θe * C.τe * zs.length ≤ 2 * (L + 1) * zs.countP (fG read C M U θ T)
  ∨ binomSfGe (max C.n₀ ⌈C.qg * zs.length⌉₊) C.θpt (2 * zs.countP (fG read C M U θ T)) < C.a
  ∨ (ε < D.real {x | Disagrees read T E C.k x} ∧ C.n₀ ≤ zs.length
      ∧ 1 - binomSfGe zs.length C.εd (zs.countP (fDis read C T E) + 1) < C.a)

/-- The stretch reached `zs.length` probes: fewer than `m` records at every edge and target,
no success at look `N₁`, and no harvest of undecided middles at the last look. -/
def Survives (zs : List (FreeMonoid α)) : Prop :=
  (∀ τ, zs.countP (fR read C T E τ) < C.m)
  ∧ rateSide C.εd C.a C.n₀ N₁ ((zs.take N₁).countP (fS read C T E)) ≠ some false
  ∧ ¬ (C.qg * zs.length ≤ zs.countP (fS read C T E)
      ∧ rateSide C.θpt C.a C.n₀ (zs.countP (fS read C T E)) (zs.countP (fP read C T E))
        = some true)

theorem rateSide_true {θ' a : ℝ} {n₀ n h : ℕ} (hr : rateSide θ' a n₀ n h = some true) :
    n₀ ≤ n ∧ binomSfGe n θ' h < a := by
  unfold rateSide at hr
  split_ifs at hr with h1 h2 h3 <;> simp_all

theorem rateSide_true_of {θ' a : ℝ} {n₀ n h : ℕ} (hn : n₀ ≤ n) (hb : binomSfGe n θ' h < a) :
    rateSide θ' a n₀ n h = some true := by
  unfold rateSide
  rw [if_pos hn, if_pos hb]

theorem rateSide_false {θ' a : ℝ} {n₀ n h : ℕ} (hr : rateSide θ' a n₀ n h = some false) :
    n₀ ≤ n ∧ 1 - binomSfGe n θ' (h + 1) < a := by
  unfold rateSide at hr
  split_ifs at hr with h1 h2 h3 <;> simp_all

variable {σ' : Type*} [Fintype σ'] {M' : DFA α σ'} {side : σ' → Bool}

/-- A step that ends the round badly fires a test falsely. -/
theorem falseFire_of_end (hW : NoWrong M' side read) (hcap : Fintype.card σ' + 2 ≤ C.Lmax)
    (hm : 1 ≤ C.m) (hθs0 : 0 ≤ C.θs) (hθs1 : C.θs ≤ 1) (hθe : 0 ≤ C.θe) (hθpt0 : 0 ≤ C.θpt)
    (hθpt1 : C.θpt ≤ 1) (hεd0 : 0 ≤ C.εd) (hεd1 : C.εd ≤ 1) {ys : List (FreeMonoid α)}
    {s : RState α} (hs : Inv read C s) (hst : StInv read C M U θ T E L ys s)
    {x : FreeMonoid α} {e : REnd α} (he : step read C s x = .inr e)
    (hbad : ¬ EndsWell M (BadAt U θ) D ε {y | Disagrees read s.tree s.edges C.k y}
      (toRoundEnd (some e))) :
    FalseFire read C M U θ D ε T E L (ys ++ [x]) := by
  classical
  have hlook := step_inr read C hW hcap hm hs he
  have hst' := stInv_chg read C M U θ T E L hs hst x
  rw [hst.tree, hst.edges] at hbad
  have hlen : (chg read C s x).n = (ys ++ [x]).length := hst'.n
  rcases look_some C hlook with ⟨hr, rfl⟩ | ⟨p, c, h1, h2, rfl⟩ | ⟨hq, hr, rfl⟩ | ⟨hr, rfl⟩
  · obtain ⟨hn₀, hsf⟩ := rateSide_true hr
    have hl := len_le_two_good M U θ D ε hbad
    have hg := hst'.stg
    rw [hlen] at hn₀ hsf
    refine Or.inl ⟨hn₀, lt_of_le_of_lt (binomSfGe_anti hθs0 hθs1 (by omega)) hsf⟩
  · by_cases hlong : ∃ y ∈ ys ++ [x], L < y.toList.length
    · exact Or.inr (Or.inl hlong)
    · push Not at hlong
      have hg := hst'.undg hlong p c
      have hl := len_le_two_good M U θ D ε hbad
      refine Or.inr (Or.inr (Or.inl ?_))
      rw [hlen] at h1
      have h3 : C.θe * C.τe * ((ys ++ [x]).length : ℝ) ≤ C.θe * (chg read C s x).reads p c := by
        rw [mul_assoc]; exact mul_le_mul_of_nonneg_left h1 hθe
      have h4 : ((chg read C s x).und p c).length ≤ 2 * (L + 1) * (ys ++ [x]).countP
          (fG read C M U θ T) := by
        have := Nat.mul_le_mul_left 2 hg
        rw [← mul_assoc] at this
        omega
      have h5 : (((chg read C s x).und p c).length : ℝ) ≤ 2 * (L + 1) * ((ys ++ [x]).countP
          (fG read C M U θ T) : ℝ) := by exact_mod_cast h4
      linarith
  · obtain ⟨hn₀, hsf⟩ := rateSide_true hr
    have hl := len_le_two_good M U θ D ε hbad
    have hg := hst'.ptg
    refine Or.inr (Or.inr (Or.inr (Or.inl ?_)))
    have hceil : ⌈C.qg * ((ys ++ [x]).length : ℝ)⌉₊ ≤ (chg read C s x).dis := by
      rw [Nat.ceil_le, ← hlen]; exact hq
    calc _ ≤ binomSfGe (chg read C s x).dis C.θpt (2 * (ys ++ [x]).countP (fG read C M U θ T)) :=
          binomSfGe_mono_n hθpt0 hθpt1 (max_le hn₀ hceil) _
      _ ≤ binomSfGe (chg read C s x).dis C.θpt (chg read C s x).pt.length :=
          binomSfGe_anti hθpt0 hθpt1 (by omega)
      _ < C.a := hsf
  · obtain ⟨hn₀, hsf⟩ := rateSide_false hr
    have hD : ε < D.real {y | Disagrees read T E C.k y} := by
      simpa [toRoundEnd, EndsWell] using hbad
    rw [hlen] at hn₀ hsf
    refine Or.inr (Or.inr (Or.inr (Or.inr ⟨hD, hn₀, ?_⟩)))
    have hle : (ys ++ [x]).countP (fDis read C T E) ≤ (ys ++ [x]).countP (fS read C T E) :=
      List.countP_mono_left fun y _ hy => by
        simp only [fDis, decide_eq_true_eq] at hy
        exact (disagrees_searches read T y E C.k hy).2
    rw [← hst'.dis] at hle
    have := binomSfGe_anti hεd0 hεd1 (n := (ys ++ [x]).length) (Nat.succ_le_succ hle)
    linarith

/-- A bad step in a stretch shows in its counts. -/
theorem seg_reduce (hW : NoWrong M' side read) (hcap : Fintype.card σ' + 2 ≤ C.Lmax)
    (hm : 1 ≤ C.m) (hN₁s : N₁ ≤ Ns) (hθs0 : 0 ≤ C.θs) (hθs1 : C.θs ≤ 1) (hθe : 0 ≤ C.θe)
    (hθpt0 : 0 ≤ C.θpt) (hθpt1 : C.θpt ≤ 1) (hεd0 : 0 ≤ C.εd) (hεd1 : C.εd ≤ 1) :
    ∀ (xs ys : List (FreeMonoid α)) (s : RState α), Inv read C s →
      StInv read C M U θ T E L ys s → Look₁ read C T E N₁ ys → s.n < Ns →
      SegB (step read C) (BadStep read C M U θ D ε Ns) Same s xs →
      ∃ i, i < xs.length ∧ ys.length + i + 1 ≤ Ns ∧
        (FalseFire read C M U θ D ε T E L (ys ++ xs.take (i + 1))
          ∨ (ys.length + i + 1 = Ns ∧ Survives read C T E N₁ (ys ++ xs.take (i + 1))))
  | [], _, _, _, _, _, _, h => by simp [SegB] at h
  | x :: xs, ys, s, hs, hst, h₁, hn, h => by
    have hn' : s.n = ys.length := hst.n
    have hsame : ∀ s', step read C s x = .inl s' → Same s s' →
        Inv read C s' ∧ StInv read C M U θ T E L (ys ++ [x]) s' ∧ Look₁ read C T E N₁ (ys ++ [x])
          ∧ look C s' = none ∧ s'.n = ys.length + 1 := by
      intro s' h' hsm
      obtain ⟨rfl, hl, hinv⟩ := step_same_chg read C hW hs h' hsm
      have hst' := stInv_chg read C M U θ T E L hs hst x
      refine ⟨hinv, hst', look₁_snoc read C M U θ T E L N₁ hst' h₁ fun hN => ?_, hl, ?_⟩
      · have := (look_none C hl).1
        rwa [show (chg read C s x).n = N₁ by rw [hst'.n]; simp; omega] at this
      · rw [hst'.n]; simp
    by_cases hb : BadStep read C M U θ D ε Ns s x
    · refine ⟨0, by simp, by omega, ?_⟩
      simp only [zero_add, List.take_succ_cons, List.take_zero]
      rcases hb with ⟨-, e, he, hbad⟩ | ⟨s', h', hsm, hNs⟩
      · exact Or.inl (falseFire_of_end read C M U θ D ε T E L hW hcap hm hθs0 hθs1 hθe hθpt0
          hθpt1 hεd0 hεd1 hs hst he hbad)
      · obtain ⟨hinv, hst', h₁', hl, hn''⟩ := hsame s' h' hsm
        refine Or.inr ⟨by omega, ?_, ?_, ?_⟩
        · intro τ
          rw [← hst'.recs]
          exact hinv.recs_lt τ
        · exact h₁' (by simp; omega)
        · have := (look_none C hl).2
          rwa [hst'.n, hst'.dis, hst'.pt] at this
    · rcases h with h | ⟨s', h', hsm, hseg⟩
      · exact absurd h hb
      obtain ⟨hinv, hst', h₁', -, hn''⟩ := hsame s' h' hsm
      have hlt : s'.n < Ns := by
        have : s'.n ≠ Ns := fun hc => hb (Or.inr ⟨s', h', hsm, hc⟩)
        omega
      obtain ⟨i, hi, hiN, hev⟩ := seg_reduce hW hcap hm hN₁s hθs0 hθs1 hθe hθpt0 hθpt1 hεd0 hεd1
        xs (ys ++ [x]) s' hinv hst' h₁' hlt hseg
      refine ⟨i + 1, by simp; omega, by simp at hiN; omega, ?_⟩
      have heq : ys ++ (x :: xs).take (i + 1 + 1) = ys ++ [x] ++ xs.take (i + 1) := by simp
      rw [heq]
      simp only [List.length_append, List.length_singleton] at hev
      rcases hev with hev | ⟨hev1, hev2⟩
      · exact Or.inl hev
      · exact Or.inr ⟨by omega, hev2⟩

end Reduce

/-! ## The chance of a bad step -/

section Prob

variable [IsProbabilityMeasure D] (T : DTree α) (E : Edges α) (L N₁ : ℕ)

theorem long_null (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) (N n : ℕ) :
    (Measure.pi fun _ : Fin N => D).real
      {xs | ∃ y ∈ (List.ofFn xs).take n, L < y.toList.length} = 0 := by
  have hD : D {y | L < y.toList.length} = 0 := by
    rw [ae_iff] at hL; simpa [not_le] using hL
  have hsub : {xs : Fin N → FreeMonoid α | ∃ y ∈ (List.ofFn xs).take n, L < y.toList.length}
      ⊆ ⋃ i, Function.eval i ⁻¹' {y | L < y.toList.length} := by
    rintro xs ⟨y, hy, hL'⟩
    obtain ⟨i, rfl⟩ := List.mem_ofFn.1 (List.mem_of_mem_take hy)
    exact Set.mem_iUnion.2 ⟨i, hL'⟩
  have h0 := measure_mono_null hsub
    (measure_iUnion_null fun i => Measure.pi_eval_preimage_null (fun _ : Fin N => D) hD)
  simp [measureReal_def, h0]

theorem fire_le (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) {G : ℝ}
    (hG : D.real {x | fG read C M U θ T x} ≤ G) (hG0 : 0 ≤ G) (hG1 : G ≤ 1) (ha : 0 ≤ C.a)
    (hθs0 : 0 ≤ C.θs) (hθs1 : C.θs ≤ 1) (hθpt0 : 0 ≤ C.θpt) (hθpt1 : C.θpt ≤ 1)
    (hεd0 : 0 ≤ C.εd) (hεd1 : C.εd ≤ 1) (hεd : C.εd ≤ ε) (N n : ℕ) :
    (Measure.pi fun _ : Fin N => D).real
        {xs | n ≤ N ∧ FalseFire read C M U θ D ε T E L ((List.ofFn xs).take n)}
      ≤ (if C.n₀ ≤ n then binomSfGe n G ((hFire C.θs C.a n + 1) / 2) + C.a else 0)
        + (binomSfGe n G ⌈(C.exc + C.θe * C.τe * n) / (2 * (L + 1))⌉₊
          + binomSfGe n G ((hFire C.θpt C.a (max C.n₀ ⌈C.qg * n⌉₊) + 1) / 2)) := by
  classical
  have hRHS : 0 ≤ (if C.n₀ ≤ n then binomSfGe n G ((hFire C.θs C.a n + 1) / 2) + C.a else 0)
      + (binomSfGe n G ⌈(C.exc + C.θe * C.τe * n) / (2 * (L + 1))⌉₊
        + binomSfGe n G ((hFire C.θpt C.a (max C.n₀ ⌈C.qg * n⌉₊) + 1) / 2)) := by
    have := binomSfGe_nonneg hG0 hG1 n ((hFire C.θs C.a n + 1) / 2)
    have := binomSfGe_nonneg hG0 hG1 n ⌈(C.exc + C.θe * C.τe * n) / (2 * (L + 1))⌉₊
    have := binomSfGe_nonneg hG0 hG1 n ((hFire C.θpt C.a (max C.n₀ ⌈C.qg * n⌉₊) + 1) / 2)
    split_ifs <;> linarith
  by_cases hnN : n ≤ N
  swap
  · have : {xs : Fin N → FreeMonoid α | n ≤ N ∧
        FalseFire read C M U θ D ε T E L ((List.ofFn xs).take n)} = ∅ := by
      ext xs; simp [hnN]
    rw [this, measureReal_empty]
    exact hRHS
  have hlen : ∀ xs : Fin N → FreeMonoid α, ((List.ofFn xs).take n).length = n := by
    intro xs; simp [hnN]
  set fg := fG read C M U θ T
  set A1 := {xs : Fin N → FreeMonoid α |
    C.n₀ ≤ n ∧ binomSfGe n C.θs (2 * cnt fg n (List.ofFn xs)) < C.a}
  set A2 := {xs : Fin N → FreeMonoid α | ∃ y ∈ (List.ofFn xs).take n, L < y.toList.length}
  set A3 := {xs : Fin N → FreeMonoid α |
    C.exc + C.θe * C.τe * n ≤ 2 * (L + 1) * (cnt fg n (List.ofFn xs) : ℝ)}
  set A4 := {xs : Fin N → FreeMonoid α |
    binomSfGe (max C.n₀ ⌈C.qg * n⌉₊) C.θpt (2 * cnt fg n (List.ofFn xs)) < C.a}
  set A5 := {xs : Fin N → FreeMonoid α | ε < D.real {x | Disagrees read T E C.k x} ∧ C.n₀ ≤ n ∧
    1 - binomSfGe n C.εd (cnt (fDis read C T E) n (List.ofFn xs) + 1) < C.a}
  have hsub : {xs : Fin N → FreeMonoid α | n ≤ N ∧
      FalseFire read C M U θ D ε T E L ((List.ofFn xs).take n)} ⊆ A1 ∪ A2 ∪ A3 ∪ A4 ∪ A5 := by
    rintro xs ⟨-, h⟩
    unfold FalseFire at h
    rw [hlen xs] at h
    rcases h with h | h | h | h | h
    · exact Or.inl (Or.inl (Or.inl (Or.inl h)))
    · exact Or.inl (Or.inl (Or.inl (Or.inr h)))
    · exact Or.inl (Or.inl (Or.inr (by simpa [A3, cnt] using h)))
    · exact Or.inl (Or.inr h)
    · exact Or.inr h
  have h1 : (Measure.pi fun _ : Fin N => D).real A1
      ≤ if C.n₀ ≤ n then binomSfGe n G ((hFire C.θs C.a n + 1) / 2) else 0 := by
    split_ifs with hn₀
    · have : A1 = {xs | binomSfGe n C.θs (2 * cnt fg n (List.ofFn xs)) < C.a} := by
        ext xs; simp [A1, hn₀]
      rw [this]
      refine pi_cnt_up D fg hnN hG hG1 _ (fun j j' h hj => lt_of_le_of_lt
        (binomSfGe_anti hθs0 hθs1 (by omega)) hj) _ fun j hj => ?_
      have := Nat.sInf_le (s := {h | binomSfGe n C.θs h < C.a}) hj
      unfold hFire; omega
    · have : A1 = ∅ := by ext xs; simp [A1, hn₀]
      rw [this, measureReal_empty]
  have h3 : (Measure.pi fun _ : Fin N => D).real A3
      ≤ binomSfGe n G ⌈(C.exc + C.θe * C.τe * n) / (2 * (L + 1))⌉₊ := by
    refine pi_cnt_up D fg hnN hG hG1
      (fun j : ℕ => C.exc + C.θe * C.τe * n ≤ 2 * (L + 1) * (j : ℝ))
      (fun j j' h hj => le_trans hj ?_) _ fun j hj => ?_
    · have : (j : ℝ) ≤ j' := by exact_mod_cast h
      have : (0 : ℝ) ≤ 2 * (L + 1) := by positivity
      nlinarith
    · rw [Nat.ceil_le, div_le_iff₀ (by positivity)]
      linarith
  have h4 : (Measure.pi fun _ : Fin N => D).real A4
      ≤ binomSfGe n G ((hFire C.θpt C.a (max C.n₀ ⌈C.qg * n⌉₊) + 1) / 2) := by
    refine pi_cnt_up D fg hnN hG hG1 _ (fun j j' h hj => lt_of_le_of_lt
      (binomSfGe_anti hθpt0 hθpt1 (by omega)) hj) _ fun j hj => ?_
    have := Nat.sInf_le (s := {h | binomSfGe (max C.n₀ ⌈C.qg * n⌉₊) C.θpt h < C.a}) hj
    unfold hFire; omega
  have h5 : (Measure.pi fun _ : Fin N => D).real A5 ≤ if C.n₀ ≤ n then C.a else 0 := by
    by_cases hc : ε < D.real {x | Disagrees read T E C.k x} ∧ C.n₀ ≤ n
    · rw [if_pos hc.2]
      have : A5 = {xs | 1 - binomSfGe n C.εd (cnt (fDis read C T E) n (List.ofFn xs) + 1) < C.a} := by
        ext xs; simp [A5, hc.1, hc.2]
      rw [this]
      refine (le_of_eq (pi_cnt D (fDis read C T E) N n hnN
        (fun j => 1 - binomSfGe n C.εd (j + 1) < C.a))).trans ?_
      have hp : D.real {x | fDis read C T E x} = D.real {x | Disagrees read T E C.k x} := by
        congr 1; ext x; simp [fDis]
      rw [hp]
      calc _ ≤ binE n C.εd (fun j => if 1 - binomSfGe n C.εd (j + 1) < C.a then 1 else 0) :=
            binE_le_of_le_anti hεd0 (by linarith [hc.1]) (by
              have := measureReal_le_one (μ := D) (s := {x | Disagrees read T E C.k x})
              exact this) n _ fun i j h => by
              have := binomSfGe_anti hεd0 hεd1 (n := n) (Nat.succ_le_succ h)
              by_cases hj : 1 - binomSfGe n C.εd (j + 1) < C.a
              · rw [if_pos hj, if_pos (by linarith)]
              · rw [if_neg hj]; split_ifs <;> norm_num
        _ ≤ C.a := binE_pvalue_le hεd0 hεd1 ha n
    · have : A5 = ∅ := by
        ext xs
        simp only [A5, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_and]
        exact fun h1 h2 => absurd ⟨h1, h2⟩ hc
      rw [this, measureReal_empty]
      split_ifs <;> linarith
  have h2 := long_null D L hL N n
  calc _ ≤ (Measure.pi fun _ : Fin N => D).real (A1 ∪ A2 ∪ A3 ∪ A4 ∪ A5) := measureReal_mono hsub
    _ ≤ (Measure.pi fun _ : Fin N => D).real A1 + (Measure.pi fun _ : Fin N => D).real A2
        + (Measure.pi fun _ : Fin N => D).real A3 + (Measure.pi fun _ : Fin N => D).real A4
        + (Measure.pi fun _ : Fin N => D).real A5 := by
      refine (measureReal_union_le _ _).trans (add_le_add_left ?_ _)
      refine (measureReal_union_le _ _).trans (add_le_add_left ?_ _)
      refine (measureReal_union_le _ _).trans (add_le_add_left ?_ _)
      exact measureReal_union_le _ _
    _ ≤ _ := by
      rw [h2]
      split_ifs at h1 h5 ⊢ <;> linarith

theorem surv_le {s : RState α} (hs : Inv read C s) (hT : s.tree = T) (hE : s.edges = E)
    (hm : 1 ≤ C.m) (hn₀ : 1 ≤ C.n₀) (hN₁ : C.n₀ ≤ N₁) (hN₁s : N₁ ≤ Ns) (hθpt0 : 0 ≤ C.θpt)
    (hθpt1 : C.θpt ≤ 1) (hεd0 : 0 ≤ C.εd) (hεd1 : C.εd ≤ 1) {θr εd' θpt' : ℝ} (hθr0 : 0 ≤ θr)
    (hθr1 : θr ≤ 1) (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1) (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1)
    (hsep : (C.Lmax : ℝ) ^ 2 * Fintype.card α * θr ≤ (1 - θpt') * εd') (N : ℕ) :
    (Measure.pi fun _ : Fin N => D).real
        {xs | Ns ≤ N ∧ Survives read C T E N₁ ((List.ofFn xs).take Ns)}
      ≤ (1 - binomSfGe Ns θr C.m)
        + binE N₁ εd' (fun s =>
            if C.a ≤ 1 - binomSfGe N₁ C.εd (s + 1) ∨ binomSfGe N₁ C.εd s < C.a then 1 else 0)
        + (1 - binomSfGe Ns εd' (max C.n₀ ⌈C.qg * Ns⌉₊))
        + ∑ s ∈ Finset.Icc (max C.n₀ ⌈C.qg * Ns⌉₊) Ns,
            binE s θpt' (fun j => if C.a ≤ binomSfGe s C.θpt j then 1 else 0) := by
  classical
  set sm := max C.n₀ ⌈C.qg * Ns⌉₊ with hsm
  have hsm1 : 1 ≤ sm := le_trans hn₀ (le_max_left _ _)
  have h01 : ∀ (P : Prop) [Decidable P], (0 : ℝ) ≤ if P then 1 else 0 := fun P _ => by
    split_ifs <;> norm_num
  have hB0 : 0 ≤ 1 - binomSfGe Ns θr C.m := by linarith [binomSfGe_le_one hθr0 hθr1 Ns C.m]
  have hC0 : 0 ≤ binE N₁ εd' (fun s =>
      if C.a ≤ 1 - binomSfGe N₁ C.εd (s + 1) ∨ binomSfGe N₁ C.εd s < C.a then 1 else 0) :=
    binE_nonneg hεd'0 hεd'1 fun j => h01 _
  have hD10 : 0 ≤ 1 - binomSfGe Ns εd' sm := by linarith [binomSfGe_le_one hεd'0 hεd'1 Ns sm]
  have hD20 : 0 ≤ ∑ s ∈ Finset.Icc sm Ns,
      binE s θpt' (fun j => if C.a ≤ binomSfGe s C.θpt j then 1 else 0) :=
    Finset.sum_nonneg fun s _ => binE_nonneg hθpt'0 hθpt'1 fun j => h01 _
  by_cases hNs : Ns ≤ N
  swap
  · have : {xs : Fin N → FreeMonoid α | Ns ≤ N ∧
        Survives read C T E N₁ ((List.ofFn xs).take Ns)} = ∅ := by ext xs; simp [hNs]
    rw [this, measureReal_empty]
    linarith
  have hlen : ∀ xs : Fin N → FreeMonoid α, ((List.ofFn xs).take Ns).length = Ns := by
    intro xs; simp [hNs]
  have htake : ∀ xs : Fin N → FreeMonoid α,
      ((List.ofFn xs).take Ns).take N₁ = (List.ofFn xs).take N₁ := by
    intro xs; rw [List.take_take, min_eq_left hN₁s]
  set Rset := T.leaves.toFinset ×ˢ ((Finset.univ : Finset α) ×ˢ T.leaves.toFinset)
  by_cases hB : ∃ τ ∈ Rset, θr ≤ D.real {x | fR read C T E τ x}
  · obtain ⟨τ, -, hτ⟩ := hB
    have hsub : {xs : Fin N → FreeMonoid α | Ns ≤ N ∧
        Survives read C T E N₁ ((List.ofFn xs).take Ns)}
        ⊆ {xs | cnt (fR read C T E τ) Ns (List.ofFn xs) < C.m} := by
      rintro xs ⟨-, h1, -, -⟩; exact h1 τ
    calc _ ≤ _ := measureReal_mono hsub
      _ = binE Ns (D.real {x | fR read C T E τ x}) (fun j => if j < C.m then 1 else 0) :=
          pi_cnt D _ N Ns hNs (fun j => j < C.m)
      _ ≤ binE Ns θr (fun j => if j < C.m then 1 else 0) :=
          binE_le_of_le_anti hθr0 hτ measureReal_le_one Ns _ fun i j h => by
            by_cases hj : j < C.m
            · rw [if_pos hj, if_pos (by omega)]
            · rw [if_neg hj]; split_ifs <;> norm_num
      _ = 1 - binomSfGe Ns θr C.m := (one_sub_binomSfGe' hθr0 hθr1 hm).symm
      _ ≤ _ := by linarith
  push Not at hB
  set q := D.real {x | fS read C T E x}
  set indC := fun s : ℕ =>
    if C.a ≤ 1 - binomSfGe N₁ C.εd (s + 1) ∨ binomSfGe N₁ C.εd s < C.a then (1 : ℝ) else 0
  by_cases hC : q ≤ εd'
  · have hsub : {xs : Fin N → FreeMonoid α | Ns ≤ N ∧
        Survives read C T E N₁ ((List.ofFn xs).take Ns)}
        ⊆ {xs | C.a ≤ 1 - binomSfGe N₁ C.εd (cnt (fS read C T E) N₁ (List.ofFn xs) + 1)
            ∨ binomSfGe N₁ C.εd (cnt (fS read C T E) N₁ (List.ofFn xs)) < C.a} := by
      rintro xs ⟨-, -, h2, -⟩
      rw [htake] at h2
      simp only [Set.mem_ofPred_eq]
      unfold rateSide at h2
      rw [if_pos hN₁] at h2
      split_ifs at h2 with h3 h4
      · exact Or.inr h3
      · exact absurd rfl h2
      · exact Or.inl (not_lt.1 h4)
    calc _ ≤ _ := measureReal_mono hsub
      _ = binE N₁ q indC := pi_cnt D _ N N₁ (hN₁s.trans hNs)
          (fun j => C.a ≤ 1 - binomSfGe N₁ C.εd (j + 1) ∨ binomSfGe N₁ C.εd j < C.a)
      _ ≤ binE N₁ εd' indC := binE_le_of_le measureReal_nonneg hC hεd'1 N₁ _ fun i j h => by
          simp only [indC]
          have h1 := binomSfGe_anti hεd0 hεd1 (n := N₁) (Nat.succ_le_succ h)
          have h2 := binomSfGe_anti hεd0 hεd1 (n := N₁) h
          by_cases hi : C.a ≤ 1 - binomSfGe N₁ C.εd (i + 1) ∨ binomSfGe N₁ C.εd i < C.a
          · rw [if_pos hi, if_pos]
            rcases hi with hi | hi
            · exact Or.inl (by linarith)
            · exact Or.inr (by linarith)
          · rw [if_neg hi]; split_ifs <;> norm_num
      _ ≤ _ := by linarith
  push Not at hC
  set pP := D.real {x | fP read C T E x}
  have hEOK : EdgesOK read T E := hT ▸ hE ▸ hs.edges
  have hcov : {x | fS read C T E x = true}
      ⊆ {x | fP read C T E x = true} ∪ ⋃ τ ∈ Rset, {x | fR read C T E τ x = true} := by
    intro x hx
    simp only [Set.mem_ofPred_eq, fS] at hx
    have hOK := probeR_ok read hEOK C.k x
    generalize hg : probeR read T E C.k x = o at hx hOK
    cases o with
    | edge p c u t =>
      obtain ⟨hp, -, huc, -⟩ := hOK
      refine Or.inr (Set.mem_biUnion (x := (p, c, t)) ?_ ?_)
      · simp [Rset, hp, DTree.sift_inl_mem read T _ t huc]
      · simp [fR, hg, recOf]
    | triple k zs =>
      obtain ⟨-, hzs, -⟩ := hOK
      refine Or.inl ?_
      simp only [Set.mem_ofPred_eq, fP, hg, ptStr]
      cases zs with
      | nil => exact absurd rfl hzs
      | cons z zs => simp
    | pair k zs =>
      obtain ⟨-, hzs, -⟩ := hOK
      refine Or.inl ?_
      simp only [Set.mem_ofPred_eq, fP, hg, ptStr]
      cases zs with
      | nil => exact absurd rfl hzs
      | cons z zs => simp
    | agree => simp [searches] at hx
    | startU z => simp [searches] at hx
    | endU z => simp [searches] at hx
    | member p c u t => simp [searches] at hx
  have hRcard : (Rset.card : ℝ) ≤ (C.Lmax : ℝ) ^ 2 * Fintype.card α := by
    have h1 : T.leaves.toFinset.card ≤ C.Lmax :=
      (List.toFinset_card_le _).trans (hT ▸ hs.size)
    have : Rset.card ≤ C.Lmax ^ 2 * Fintype.card α := by
      simp only [Rset, Finset.card_product, Finset.card_univ]
      calc _ ≤ C.Lmax * (Fintype.card α * C.Lmax) := by gcongr
        _ = _ := by ring
    exact_mod_cast this
  have hqle : q ≤ pP + Rset.card * θr := by
    calc q ≤ D.real ({x | fP read C T E x = true} ∪ ⋃ τ ∈ Rset, {x | fR read C T E τ x = true}) :=
          measureReal_mono hcov
      _ ≤ pP + ∑ τ ∈ Rset, D.real {x | fR read C T E τ x = true} :=
          (measureReal_union_le _ _).trans (add_le_add le_rfl (measureReal_biUnion_finset_le _ _))
      _ ≤ pP + ∑ _τ ∈ Rset, θr := by gcongr with τ hτ; exact (hB τ hτ).le
      _ = pP + Rset.card * θr := by rw [Finset.sum_const, nsmul_eq_mul]
  have hq0 : 0 < q := lt_of_le_of_lt hεd'0 hC
  have hPq : θpt' * q ≤ pP := by
    have : (Rset.card : ℝ) * θr ≤ (1 - θpt') * q := by
      calc _ ≤ (C.Lmax : ℝ) ^ 2 * Fintype.card α * θr := by gcongr
        _ ≤ (1 - θpt') * εd' := hsep
        _ ≤ _ := by gcongr
    nlinarith
  have hPle : pP ≤ q := measureReal_mono fun x hx => by
    simp only [Set.mem_ofPred_eq, fP, fS] at hx ⊢
    exact searches_of_ptStr (by simpa using hx)
  have hπ0 : θpt' ≤ pP / q := by rw [le_div_iff₀ hq0]; linarith
  have hπ1 : pP / q ≤ 1 := by rw [div_le_one hq0]; exact hPle
  have hsub : {xs : Fin N → FreeMonoid α | Ns ≤ N ∧
      Survives read C T E N₁ ((List.ofFn xs).take Ns)}
      ⊆ {xs | cnt (fS read C T E) Ns (List.ofFn xs) < sm}
        ∪ {xs | sm ≤ cnt (fS read C T E) Ns (List.ofFn xs) ∧ C.a ≤ binomSfGe
            (cnt (fS read C T E) Ns (List.ofFn xs)) C.θpt (cnt (fP read C T E) Ns (List.ofFn xs))} := by
    rintro xs ⟨-, -, -, h3⟩
    rw [hlen] at h3
    by_cases hS : cnt (fS read C T E) Ns (List.ofFn xs) < sm
    · exact Or.inl hS
    · refine Or.inr ⟨not_lt.1 hS, ?_⟩
      push Not at hS
      have hq' : C.qg * Ns ≤ cnt (fS read C T E) Ns (List.ofFn xs) :=
        Nat.ceil_le.1 (le_trans (le_max_right _ _) hS)
      have hn₀' : C.n₀ ≤ cnt (fS read C T E) Ns (List.ofFn xs) := le_trans (le_max_left _ _) hS
      by_contra hlt
      push Not at hlt
      apply h3
      refine ⟨hq', ?_⟩
      exact rateSide_true_of hn₀' hlt
  calc _ ≤ _ := measureReal_mono hsub
    _ ≤ _ := measureReal_union_le _ _
    _ ≤ (1 - binomSfGe Ns εd' sm) + ∑ s ∈ Finset.Icc sm Ns,
          binE s θpt' (fun j => if C.a ≤ binomSfGe s C.θpt j then 1 else 0) := by
      gcongr
      · calc _ = binE Ns q (fun j => if j < sm then 1 else 0) :=
            pi_cnt D _ N Ns hNs (fun j => j < sm)
          _ ≤ binE Ns εd' (fun j => if j < sm then 1 else 0) :=
              binE_le_of_le_anti hεd'0 hC.le measureReal_le_one Ns _ fun i j h => by
                by_cases hj : j < sm
                · rw [if_pos hj, if_pos (by omega)]
                · rw [if_neg hj]; split_ifs <;> norm_num
          _ = _ := (one_sub_binomSfGe' hεd'0 hεd'1 hsm1).symm
      · rw [pi_cnt2 D (fS read C T E) (fP read C T E) (fun x hx => searches_of_ptStr (by
            simp only [fP] at hx; simpa using hx)) N Ns hNs
          (fun s j => sm ≤ s ∧ C.a ≤ binomSfGe s C.θpt j)]
        calc _ ≤ binE Ns q (fun _ => ∑ s ∈ Finset.Icc sm Ns,
              binE s θpt' (fun j => if C.a ≤ binomSfGe s C.θpt j then 1 else 0)) := by
              refine binE_mono_le measureReal_nonneg measureReal_le_one fun s hs => ?_
              by_cases hss : sm ≤ s
              · calc _ = binE s (pP / q) (fun j => if C.a ≤ binomSfGe s C.θpt j then 1 else 0) :=
                      binE_congr fun j _ => by simp [hss]
                  _ ≤ binE s θpt' (fun j => if C.a ≤ binomSfGe s C.θpt j then 1 else 0) :=
                      binE_le_of_le_anti hθpt'0 hπ0 hπ1 s _ fun i j h => by
                        have := binomSfGe_anti hθpt0 hθpt1 (n := s) h
                        by_cases hj : C.a ≤ binomSfGe s C.θpt j
                        · rw [if_pos hj, if_pos (by linarith)]
                        · rw [if_neg hj]; split_ifs <;> norm_num
                  _ ≤ _ := Finset.single_le_sum (f := fun s =>
                        binE s θpt' (fun j => if C.a ≤ binomSfGe s C.θpt j then 1 else 0))
                      (fun s _ => binE_nonneg hθpt'0 hθpt'1 fun j => h01 _)
                      (Finset.mem_Icc.2 ⟨hss, hs⟩)
              · calc _ = binE s (pP / q) (fun _ => 0) := binE_congr fun j _ => by simp [hss]
                  _ = 0 := binE_const _ _ _
                  _ ≤ _ := hD20
          _ = _ := binE_const _ _ _
    _ ≤ _ := by linarith

end Prob

theorem seg_le {side : σ → Bool} (hW : NoWrong M side read) [IsProbabilityMeasure D]
    (L N₁ : ℕ) (G θr εd' θpt' : ℝ) (hL : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hN : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
      D.real (PotGood M U θ C.k read T) ≤ G)
    (hcap : Fintype.card σ + 2 ≤ C.Lmax) (hm : 1 ≤ C.m) (hn₀ : 1 ≤ C.n₀) (hN₁ : C.n₀ ≤ N₁)
    (hN₁s : N₁ ≤ Ns) (hθ : 0 ≤ θ) (hG0 : 0 ≤ G) (hG1 : G ≤ 1) (ha : 0 ≤ C.a)
    (hθs0 : 0 ≤ C.θs) (hθs1 : C.θs ≤ 1) (hθe : 0 ≤ C.θe) (hθpt0 : 0 ≤ C.θpt)
    (hθpt1 : C.θpt ≤ 1) (hεd0 : 0 ≤ C.εd) (hεd1 : C.εd ≤ 1) (hεd : C.εd ≤ ε) (hθr0 : 0 ≤ θr)
    (hθr1 : θr ≤ 1) (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1) (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1)
    (hsep : (C.Lmax : ℝ) ^ 2 * Fintype.card α * θr ≤ (1 - θpt') * εd')
    {s : RState α} (hs : Inv read C s) (hf : s = fresh s.tree s.edges s.moved) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D)
        {xs | SegB (step read C) (BadStep read C M U θ D ε Ns) Same s (List.ofFn xs)}
      ≤ ENNReal.ofReal (stretchRisk C L Ns N₁ G θr εd' θpt') := by
  classical
  have hTcls := cls_mem read C hW hs
  have hG : D.real {x | fG read C M U θ s.tree x} ≤ G := by
    have : {x | fG read C M U θ s.tree x = true} = PotGood M U θ C.k read s.tree := by
      ext x; simp [fG]
    rw [this]; exact hN _ hTcls
  have hsub : {xs : Fin N → FreeMonoid α |
      SegB (step read C) (BadStep read C M U θ D ε Ns) Same s (List.ofFn xs)}
      ⊆ (⋃ n ∈ Finset.Icc 1 Ns, {xs | n ≤ N ∧
          FalseFire read C M U θ D ε s.tree s.edges L ((List.ofFn xs).take n)})
        ∪ {xs | Ns ≤ N ∧ Survives read C s.tree s.edges N₁ ((List.ofFn xs).take Ns)} := by
    intro xs hxs
    have hst0 : StInv read C M U θ s.tree s.edges L [] s := by
      have := stInv_fresh read C M U θ s.tree s.edges L s.moved
      rwa [← hf] at this
    have h₁0 : Look₁ read C s.tree s.edges N₁ [] := fun h => by simp at h; omega
    have hn0 : s.n < Ns := by rw [hf]; simp only [fresh]; omega
    obtain ⟨i, hi, hiN, h⟩ := seg_reduce read C M U θ D ε Ns s.tree s.edges L N₁ hW hcap hm hN₁s
      hθs0 hθs1 hθe hθpt0 hθpt1 hεd0 hεd1 (List.ofFn xs) [] s hs hst0 h₁0 hn0 hxs
    simp only [List.nil_append, List.length_nil, zero_add, List.length_ofFn] at hi hiN h
    rcases h with h | ⟨hNs', h⟩
    · exact Or.inl (Set.mem_biUnion (Finset.mem_coe.2 (Finset.mem_Icc.2 ⟨by omega, hiN⟩))
        ⟨by omega, h⟩)
    · refine Or.inr ⟨by omega, ?_⟩
      rwa [← hNs']
  rw [← ENNReal.ofReal_toReal (measure_ne_top _ _), ← measureReal_def]
  apply ENNReal.ofReal_le_ofReal
  have hsum : ∀ f : ℕ → ℝ, ∑ n ∈ Finset.Icc 1 Ns, (if C.n₀ ≤ n then f n else 0)
      = ∑ n ∈ Finset.Icc C.n₀ Ns, f n := by
    intro f
    rw [← Finset.sum_filter]
    congr 1
    ext n
    simp only [Finset.mem_filter, Finset.mem_Icc]
    omega
  calc _ ≤ _ := measureReal_mono hsub
    _ ≤ _ := (measureReal_union_le _ _).trans
        (add_le_add (measureReal_biUnion_finset_le _ _) le_rfl)
    _ ≤ ∑ n ∈ Finset.Icc 1 Ns,
          ((if C.n₀ ≤ n then binomSfGe n G ((hFire C.θs C.a n + 1) / 2) + C.a else 0)
            + (binomSfGe n G ⌈(C.exc + C.θe * C.τe * n) / (2 * (L + 1))⌉₊
              + binomSfGe n G ((hFire C.θpt C.a (max C.n₀ ⌈C.qg * n⌉₊) + 1) / 2)))
        + ((1 - binomSfGe Ns θr C.m)
          + binE N₁ εd' (fun s =>
              if C.a ≤ 1 - binomSfGe N₁ C.εd (s + 1) ∨ binomSfGe N₁ C.εd s < C.a then 1 else 0)
          + (1 - binomSfGe Ns εd' (max C.n₀ ⌈C.qg * Ns⌉₊))
          + ∑ s ∈ Finset.Icc (max C.n₀ ⌈C.qg * Ns⌉₊) Ns,
              binE s θpt' (fun j => if C.a ≤ binomSfGe s C.θpt j then 1 else 0)) := by
      gcongr with n hn
      · exact fire_le read C M U θ D ε s.tree s.edges L hL hG hG0 hG1 ha hθs0 hθs1 hθpt0 hθpt1
          hεd0 hεd1 hεd N n
      · exact surv_le read C D Ns s.tree s.edges N₁ hs rfl rfl hm hn₀ hN₁ hN₁s hθpt0 hθpt1 hεd0
          hεd1 hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep N
    _ = _ := by
      unfold stretchRisk
      rw [Finset.sum_add_distrib, hsum]
      ring

end Random

end OrthoDFA
