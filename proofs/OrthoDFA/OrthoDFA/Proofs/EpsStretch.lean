import OrthoDFA.Proofs.EpsSteps
import OrthoDFA.Proofs.RandomStretch

/-!
# One stretch of level 4

Level 3's stretch, with records: a stretch that reaches `Ns` probes has fewer than
`recordsToMove` records of its own at every edge and target, no success at look `N₁`, and no
harvest of undecided middles at look `Ns`. A stretch here is cut short by a step `Z` (the fake
fixes), which the round accounts for apart, over the probes since the last split. As at level 3,
each sequential test's false firing and each edge's threshold is paid once per stretch.
-/

namespace OrthoDFA

namespace Random

namespace Eps

open MeasureTheory
open OrthoDFA.Ideal (Outcome EdgesOK OutcomeOK probeOrAgree)

section Segments

variable {S X E : Type*} (step : S → X → S ⊕ E) (B Z : S → X → Prop) (same : S → S → Prop)

/-- Some step of the segment is bad, before any step `Z`. -/
def SegBZ : S → List X → Prop
  | _, [] => False
  | s, x :: xs => ¬ Z s x ∧ (B s x ∨ ∃ s', step s x = .inl s' ∧ same s s' ∧ SegBZ s' xs)

end Segments

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level4Settings)
  {σ : Type*} [Fintype σ] (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) (D : Measure (FreeMonoid α))
  (ε : ℝ) (Ns : ℕ) (y : ℝ)

def Same (s s' : Level4State α) : Prop := s'.tree = s.tree ∧ s'.edges = s.edges

/-- Within the first `Ns` probes of its stretch the step ends the round badly, a harvest's sampler
yielding `y`, or is one of the steps `X`, or its stretch reaches `Ns` probes. -/
def BadStep (X : Level4State α → FreeMonoid α → Prop) (s : Level4State α) (x : FreeMonoid α) :
    Prop :=
  (s.probeCount < Ns ∧ ((∃ e, applyProbe read C s x = .inr e ∧ ¬ EndsWell M (BadAt U θ) D ε
      {w | Disagrees read ⟨s.tree, s.edges⟩ C.walkStart w} C.walkStart
      (harvestLaw read D C.walkStart ⟨s.tree, s.edges⟩) y e)
      ∨ X s x))
  ∨ ∃ s', applyProbe read C s x = .inl s' ∧ Same s s' ∧ s'.probeCount = Ns

/-! ## A step within a stretch -/

/-- The state a probe leaves before the tests look at it. -/
noncomputable def chg (s : Level4State α) (x : FreeMonoid α) : Level4State α :=
  countProbe read C { s with records := s.records ++
      (recOf (probeOrAgree false read s.tree s.edges C.walkStart x)).toList } x

section StepFacts

theorem finish_inr {t : Level4State α} {e : RoundEnd (Level3Site α)}
    (h : applyTests C t = .inr e) :
    runTests C t = some e := by
  unfold applyTests at h
  split at h
  · rename_i e' he'; simp only [Sum.inr.injEq] at h; rw [he', h]
  · simp at h

theorem mk_eta (s : Level4State α) :
    (⟨s.tree, s.edges, s.records, s.probeCount, s.searchCount, s.startUndecided,
        s.middleUndecided, s.edgeReads, s.edgeUndecided⟩ : Level4State α)
      = s := rfl

theorem chg_of_none {s : Level4State α} {x : FreeMonoid α}
    (h : recOf (probeOrAgree false read s.tree s.edges C.walkStart x) = none) :
    chg read C s x = countProbe read C s x := by
  unfold chg
  rw [h]
  simp only [Option.toList_none, List.append_nil]

theorem step_inr {s : Level4State α} {x : FreeMonoid α}
    (hne : applyProbe read C s x ≠ .inr .failed)
    {e : RoundEnd (Level3Site α)} (h : applyProbe read C s x = .inr e) :
    runTests C (chg read C s x) = some e := by
  have h' := h
  unfold applyProbe at h
  rcases ho : probeOrAgree false read s.tree s.edges C.walkStart x with
    _ | ⟨p, c, u, t⟩ | ⟨p, c, u, t⟩
      | ⟨k, zs, cl⟩
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
      have : e = .failed := by
        unfold redirectOrSplit at h
        split at h
        · split_ifs at h
          · unfold splitLeaf at h
            simp only at h
            split_ifs at h
            simp only [Sum.inr.injEq] at h
            exact h.symm
        · simp at h
      exact hne (this ▸ h')
    · exact finish_inr C h

theorem step_same_chg {s s' : Level4State α} (hs : Inv read C s) {x : FreeMonoid α}
    (h : applyProbe read C s x = .inl s') (hsm : Same s s') :
    s' = chg read C s x ∧ runTests C s' = none ∧ Inv read C s' := by
  have hinv := (step_same read C hs h hsm.1 hsm.2).1
  rcases step_cases read C hs h with
    ⟨p, c, u, t, -, hE, -, -, -, rfl⟩ | ⟨p, c, u, t, t₀, w₀, -, hE, hne, hp, -, -, -, hfix⟩ |
    ⟨h1, h2, -⟩
  · exfalso
    have := congrFun hsm.2 (p, c)
    simp only [newHypothesis, Function.update_self, hE] at this
    exact absurd this (by simp)
  · exfalso
    unfold redirectOrSplit at hfix
    rw [hE] at hfix
    simp only at hfix
    split_ifs at hfix with hmv
    · unfold splitLeaf at hfix
      simp only at hfix
      split_ifs at hfix
      simp only [Sum.inl.injEq] at hfix
      subst hfix
      have := congrArg List.length (congrArg DTree.leaves hsm.1)
      simp only [newHypothesis] at this
      rw [DTree.length_leaves_splitAt _ _ _ hp] at this
      omega
    · simp only [Sum.inl.injEq] at hfix
      subst hfix
      have := congrFun hsm.2 (p, c)
      simp only [newHypothesis, Function.update_self, hE, Option.some.injEq, Prod.mk.injEq] at this
      exact hne this.1.symm
  · exact ⟨h1, h2, hinv⟩

theorem look_none {t : Level4State α} (h : runTests C t = none) :
    ¬ seqRate C.disagreementRate C.disagreementAlternative C.testLevel t.probeCount t.searchCount
      ∧ ¬ (C.searchShare * t.probeCount ≤ t.searchCount
          ∧ seqRate C.middleRate C.middleAlternative C.testLevel t.searchCount
            t.middleUndecided.length) := by
  unfold runTests at h
  split_ifs at h with h1 h2 h3 h4
  exact ⟨h4, h3⟩

theorem look_some {t : Level4State α} {e : RoundEnd (Level3Site α)} (h : runTests C t = some e) :
    (seqRate C.startRate C.startAlternative C.testLevel t.probeCount t.startUndecided.length
        ∧ e = .harvest .start)
    ∨ (∃ p c, (p, c) ∈ t.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α)
        ∧ C.edgeReadShare * t.probeCount ≤ t.edgeReads p c
        ∧ C.edgeExcess + C.edgeRate * t.edgeReads p c ≤ (t.edgeUndecided p c).length
        ∧ e = .harvest (.edge p c))
    ∨ (C.searchShare * t.probeCount ≤ t.searchCount
        ∧ seqRate C.middleRate C.middleAlternative C.testLevel t.searchCount
          t.middleUndecided.length
        ∧ e = .harvest .middles)
    ∨ (seqRate C.disagreementRate C.disagreementAlternative C.testLevel t.probeCount t.searchCount
        ∧ e = .consistent) := by
  unfold runTests at h
  split_ifs at h with h1 h2 h3 h4
  · exact Or.inl ⟨h1, (Option.some.inj h).symm⟩
  · obtain ⟨h6, h5⟩ := h2.choose_spec
    exact Or.inr (Or.inl ⟨_, _, h6, h5.1, h5.2, (Option.some.inj h).symm⟩)
  · exact Or.inr (Or.inr (Or.inl ⟨h3.1, h3.2, (Option.some.inj h).symm⟩))
  · exact Or.inr (Or.inr (Or.inr ⟨h4, (Option.some.inj h).symm⟩))

end StepFacts

/-! ## The stretch's counts -/

section Counts

variable (T : DTree α) (E : Edges α)

def fS (x : FreeMonoid α) : Bool := searches (probeOrAgree false read T E C.walkStart x)

def fSt (x : FreeMonoid α) : Bool :=
  !(startString (probeOrAgree false read T E C.walkStart x)).isEmpty

def fP (x : FreeMonoid α) : Bool :=
  !(middleString (probeOrAgree false read T E C.walkStart x)).isEmpty

/-- The strings a probe charges to the edge `e`. -/
noncomputable def edgeCount (e : List Bool × α) (x : FreeMonoid α) : ℕ :=
  (undecidedAtEdge (positionReads read T E C.walkStart x) e.1 e.2).length

open scoped Classical in
noncomputable def fDis (x : FreeMonoid α) : Bool := decide (Disagrees read ⟨T, E⟩ C.walkStart x)

def fR (τ : List Bool × α × List Bool) (x : FreeMonoid α) : Bool :=
  decide (recOf (probeOrAgree false read T E C.walkStart x) = some τ)

/-- No probe records an edge at the target it points at. -/
theorem fR_target (hE : EdgesOK read T E) {p : List Bool} {c : α} {t : List Bool}
    (ht : (E (p, c)).map Prod.fst = some t) (x : FreeMonoid α) :
    fR read C T E (p, c, t) x = false := by
  have hOK := Ideal.probeOrAgree_ok false read T E C.walkStart hE x
  unfold fR
  generalize probeOrAgree false read T E C.walkStart x = o at hOK ⊢
  cases o with
  | wrongEdge p' c' u t' =>
    obtain ⟨-, -, -, t₀, w₀, hE', hne⟩ := hOK
    simp only [recOf, Option.some.injEq, Prod.mk.injEq, decide_eq_false_iff_not, not_and]
    rintro rfl rfl rfl
    simp [hE'] at ht
    exact hne ht
  | _ => simp [recOf]

variable (N₁ : ℕ)

/-- After the probes `ys` of a stretch at `(T, E)` that started with the records `R`, the counts
are counts over `ys`. -/
structure StInv (R : List (List Bool × α × List Bool)) (ys : List (FreeMonoid α))
    (s : Level4State α) : Prop where
  tree : s.tree = T
  edges : s.edges = E
  probeCount : s.probeCount = ys.length
  searchCount : s.searchCount = ys.countP (fS read C T E)
  startUndecided : s.startUndecided.length = ys.countP (fSt read C T E)
  middleUndecided : s.middleUndecided.length = ys.countP (fP read C T E)
  edgeUndecided : ∀ p c, (s.edgeUndecided p c).length = (ys.map (edgeCount read C T E (p, c))).sum
  records : ∀ τ, s.records.count τ = R.count τ + ys.countP (fR read C T E τ)

/-- The stretch's success test did not fire at look `N₁`. -/
def Look₁ (ys : List (FreeMonoid α)) : Prop :=
  N₁ ≤ ys.length → ¬ seqRate C.disagreementRate C.disagreementAlternative C.testLevel N₁
      ((ys.take N₁).countP (fS read C T E))

theorem stInv_fresh (R : List (List Bool × α × List Bool)) :
    StInv read C T E R [] (newHypothesis T E R) where
  tree := rfl
  edges := rfl
  probeCount := rfl
  searchCount := rfl
  startUndecided := rfl
  middleUndecided := rfl
  edgeUndecided := fun p c => by simp [newHypothesis]
  records := fun τ => by simp [newHypothesis]

/-- A probe counted into a stretch whose counts are its probes' keeps them so. -/
theorem stInv_chg {R : List (List Bool × α × List Bool)} {ys : List (FreeMonoid α)}
    {s : Level4State α} (hst : StInv read C T E R ys s) (x : FreeMonoid α) :
    StInv read C T E R (ys ++ [x]) (chg read C s x) := by
  classical
  obtain ⟨htree, hedges, hn, hdis, hst', hpt, hund, hrecs⟩ := hst
  subst htree hedges
  refine ⟨rfl, rfl, by simp [chg, countProbe, hn], ?_, ?_, ?_, ?_, ?_⟩
  · simp only [chg, countProbe, hdis, countP_snoc]
    cases hb : searches (probeOrAgree false read s.tree s.edges C.walkStart x) <;> simp [fS, hb]
  · simp only [chg, countProbe, List.length_append, hst', countP_snoc]
    rw [length_le_one_eq _ (startString_length _)]
    cases hb : (startString (probeOrAgree false read s.tree s.edges C.walkStart x)).isEmpty <;>
      simp [fSt, hb]
  · simp only [chg, countProbe, List.length_append, hpt, countP_snoc]
    rw [length_le_one_eq _ (middleString_length _)]
    cases hb : (middleString (probeOrAgree false read s.tree s.edges C.walkStart x)).isEmpty <;>
      simp [fP, hb]
  · intro p c
    simp only [chg, countProbe, List.length_append, hund, List.map_append, List.sum_append,
      List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
    rfl
  · intro τ
    simp only [chg, countProbe, List.count_append, hrecs, countP_snoc]
    cases hr : recOf (probeOrAgree false read s.tree s.edges C.walkStart x) with
    | none => simp [fR, hr]
    | some τ' => by_cases h : τ' = τ <;> simp [fR, hr, h, List.count_singleton, add_assoc]

theorem look₁_snoc {R : List (List Bool × α × List Bool)} {ys : List (FreeMonoid α)}
    {s : Level4State α} {x : FreeMonoid α} (hst : StInv read C T E R (ys ++ [x]) s)
    (h₀ : Look₁ read C T E N₁ ys)
    (h₁ : N₁ = ys.length + 1 →
      ¬ seqRate C.disagreementRate C.disagreementAlternative C.testLevel N₁ s.searchCount) :
    Look₁ read C T E N₁ (ys ++ [x]) := by
  intro hN
  by_cases hlt : N₁ ≤ ys.length
  · rw [List.take_append_of_le_length hlt]
    exact h₀ hlt
  · have hN' : N₁ = ys.length + 1 := by simp at hN; omega
    rw [List.take_of_length_le (by simp; omega), ← hst.searchCount]
    exact h₁ hN'

end Counts

/-! ## A bad step shows in the counts -/

section Reduce

variable (T : DTree α) (E : Edges α) (L N₁ : ℕ)

/-- A test fired at look `n` at the start or the middles, whose sampler finds at most `p`, or at an
edge whose sampler finds at most `q`, or success fired falsely, or a probe is longer than `L`. -/
def FalseFire (p q : ℝ) (zs : List (FreeMonoid α)) : Prop :=
  (D.real {x | fSt read C T E x} ≤ p
      ∧ seqRate C.startRate C.startAlternative C.testLevel zs.length (zs.countP (fSt read C T E)))
  ∨ (∃ w ∈ zs, L < w.toList.length)
  ∨ (∃ e ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α),
      ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ q
      ∧ C.edgeExcess + C.edgeRate * C.edgeReadShare * zs.length
        ≤ (zs.map fun x => (edgeCount read C T E e x : ℝ)).sum)
  ∨ (D.real {x | fP read C T E x} ≤ p ∧ C.searchShare * zs.length ≤ zs.countP (fS read C T E)
      ∧ seqRate C.middleRate C.middleAlternative C.testLevel (zs.countP (fS read C T E))
        (zs.countP (fP read C T E)))
  ∨ (ε < D.real {x | Disagrees read ⟨T, E⟩ C.walkStart x}
      ∧ seqRate C.disagreementRate C.disagreementAlternative C.testLevel zs.length
        (zs.countP (fS read C T E)))

/-- The stretch reached `zs.length` probes: fewer than `recordsToMove` records at every edge and
target, no success at look `N₁`, and no harvest of undecided middles at the last look. -/
def Survives (zs : List (FreeMonoid α)) : Prop :=
  (∀ τ, zs.countP (fR read C T E τ) < C.recordsToMove)
  ∧ ¬ seqRate C.disagreementRate C.disagreementAlternative C.testLevel N₁
      ((zs.take N₁).countP (fS read C T E))
  ∧ ¬ (C.searchShare * zs.length ≤ zs.countP (fS read C T E)
      ∧ seqRate C.middleRate C.middleAlternative C.testLevel (zs.countP (fS read C T E))
          (zs.countP (fP read C T E)))

/-- A step that ends the round badly, not too big, fires a test falsely. -/
theorem falseFire_of_end [IsProbabilityMeasure D] (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) {G : ℝ}
    (hG : D.real (PotGood M U θ C.walkStart read T) ≤ G) (hθe : 0 ≤ C.edgeRate)
    {ys : List (FreeMonoid α)} {s : Level4State α} {R : List (List Bool × α × List Bool)}
    (hst : StInv read C T E R ys s) {x : FreeMonoid α}
    (hnt : applyProbe read C s x ≠ .inr .failed) {e : RoundEnd (Level3Site α)}
    (he : applyProbe read C s x = .inr e)
    (hbad : ¬ EndsWell M (BadAt U θ) D ε {w | Disagrees read ⟨s.tree, s.edges⟩ C.walkStart w}
      C.walkStart (harvestLaw read D C.walkStart ⟨s.tree, s.edges⟩) y e) :
    FalseFire read C D ε T E L (max (2 * G) y) (max (2 * (L + 1) * G) y) (ys ++ [x]) := by
  classical
  have hlook := step_inr read C hnt he
  have hst' := stInv_chg read C T E hst x
  rw [hst.tree, hst.edges] at hbad
  have hlen : (chg read C s x).probeCount = (ys ++ [x]).length := hst'.probeCount
  have htot := fun site w hw h => total_le_of_not_endsWell read D C.walkStart ⟨T, E⟩ hL M U θ
    (y := y) hG ε {w | Disagrees read ⟨T, E⟩ C.walkStart w} site w hw h
  have hone : ∀ site, (∀ x, (siteHarvest read C.walkStart ⟨T, E⟩ site x).length ≤ 1) →
      ¬ EndsWell M (BadAt U θ) D ε {w | Disagrees read ⟨T, E⟩ C.walkStart w} C.walkStart
        (harvestLaw read D C.walkStart ⟨T, E⟩) y (.harvest site) →
      (harvestLaw read D C.walkStart ⟨T, E⟩ site).real Set.univ ≤ max (2 * G) y := by
    intro site h1 h
    have := htot site 1 (ae_of_all _ h1) h
    rwa [Nat.cast_one, mul_one] at this
  rcases look_some C hlook with ⟨hr, rfl⟩ | ⟨p, c, hpc, h1, h2, rfl⟩ | ⟨hq, hr, rfl⟩ | ⟨hr, rfl⟩
  · have ht := hone .start (fun x => startString_length _) hbad
    rw [harvestLaw_real_univ read D C.walkStart ⟨T, E⟩ hL, integral_length_le_one D
      (siteHarvest read C.walkStart ⟨T, E⟩ .start) fun x => startString_length _] at ht
    refine Or.inl ⟨ht, ?_⟩
    rwa [hlen, hst'.startUndecided] at hr
  · have ht := htot (.edge p c) (L + 1) (by
      filter_upwards [hL] with x hx
      exact siteHarvest_length read C.walkStart ⟨T, E⟩ (.edge p c) hx) hbad
    rw [harvestLaw_real_univ read D C.walkStart ⟨T, E⟩ hL] at ht
    push_cast at ht
    have hT : (chg read C s x).tree = T := hst.tree
    refine Or.inr (Or.inr (Or.inl ⟨(p, c), by rwa [hT] at hpc, ht, ?_⟩))
    rw [hlen] at h1
    have h3 : C.edgeRate * C.edgeReadShare * ((ys ++ [x]).length : ℝ)
        ≤ C.edgeRate * (chg read C s x).edgeReads p c := by
      rw [mul_assoc]; exact mul_le_mul_of_nonneg_left h1 hθe
    have h4 : (((chg read C s x).edgeUndecided p c).length : ℝ)
        = ((ys ++ [x]).map fun x => (edgeCount read C T E (p, c) x : ℝ)).sum := by
      rw [hst'.edgeUndecided p c, Nat.cast_list_sum, List.map_map]
      rfl
    linarith
  · have ht := hone .middles (fun x => middleString_length _) hbad
    rw [harvestLaw_real_univ read D C.walkStart ⟨T, E⟩ hL, integral_length_le_one D
      (siteHarvest read C.walkStart ⟨T, E⟩ .middles) fun x => middleString_length _] at ht
    rw [hlen, hst'.searchCount] at hq
    rw [hst'.searchCount, hst'.middleUndecided] at hr
    exact Or.inr (Or.inr (Or.inr (Or.inl ⟨ht, hq, hr⟩)))
  · have hD : ε < D.real {w | Disagrees read ⟨T, E⟩ C.walkStart w} := by
      simpa [EndsWell] using hbad
    rw [hlen, hst'.searchCount] at hr
    exact Or.inr (Or.inr (Or.inr (Or.inr ⟨hD, hr⟩)))

/-- A bad step in a stretch shows in its counts. -/
theorem seg_reduce [IsProbabilityMeasure D] (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) {G : ℝ}
    (hG : D.real (PotGood M U θ C.walkStart read T) ≤ G)
    (X Z : Level4State α → FreeMonoid α → Prop) (XEv : List (FreeMonoid α) → Prop)
    (R : List (List Bool × α × List Bool)) (hm : 1 ≤ C.recordsToMove)
    (hX : ∀ ys s x, Inv read C s → StInv read C T E R ys s → X s x → XEv (ys ++ [x]))
    (hnt : ∀ ys s x, Inv read C s → StInv read C T E R ys s → ¬ X s x → ¬ Z s x
      → applyProbe read C s x ≠ .inr .failed) (hN₁s : N₁ ≤ Ns) (hθe : 0 ≤ C.edgeRate) :
    ∀ (xs ys : List (FreeMonoid α)) (s : Level4State α), Inv read C s →
      StInv read C T E R ys s → Look₁ read C T E N₁ ys → s.probeCount < Ns →
      SegBZ (applyProbe read C) (BadStep read C M U θ D ε Ns y X) Z Same s xs →
      ∃ i, i < xs.length ∧ ys.length + i + 1 ≤ Ns ∧
        (FalseFire read C D ε T E L (max (2 * G) y) (max (2 * (L + 1) * G) y)
            (ys ++ xs.take (i + 1))
          ∨ (ys.length + i + 1 = Ns ∧ Survives read C T E N₁ (ys ++ xs.take (i + 1)))
          ∨ XEv (ys ++ xs.take (i + 1)))
  | [], _, _, _, _, _, _, h => by simp [SegBZ] at h
  | x :: xs, ys, s, hs, hst, h₁, hn, ⟨hz, h⟩ => by
    have hn' : s.probeCount = ys.length := hst.probeCount
    have hsame : ∀ s', applyProbe read C s x = .inl s' → Same s s' →
        Inv read C s' ∧ StInv read C T E R (ys ++ [x]) s' ∧ Look₁ read C T E N₁ (ys ++ [x])
          ∧ runTests C s' = none ∧ s'.probeCount = ys.length + 1 := by
      intro s' h' hsm
      obtain ⟨rfl, hl, hinv⟩ := step_same_chg read C hs h' hsm
      have hst' := stInv_chg read C T E hst x
      refine ⟨hinv, hst', look₁_snoc read C T E N₁ hst' h₁ fun hN => ?_, hl, ?_⟩
      · have := (look_none C hl).1
        rwa [show (chg read C s x).probeCount = N₁ by rw [hst'.probeCount]; simp; omega] at this
      · rw [hst'.probeCount]; simp
    by_cases hb : BadStep read C M U θ D ε Ns y X s x
    · refine ⟨0, by simp, by omega, ?_⟩
      simp only [zero_add, List.take_succ_cons, List.take_zero]
      rcases hb with ⟨-, ⟨e, he, hbad⟩ | hx⟩ | ⟨s', h', hsm, hNs⟩
      · by_cases hx : X s x
        · exact Or.inr (Or.inr (hX ys s x hs hst hx))
        · exact Or.inl (falseFire_of_end read C M U θ D ε y T E L hL hG hθe hst
            (hnt ys s x hs hst hx hz) he hbad)
      · exact Or.inr (Or.inr (hX ys s x hs hst hx))
      · obtain ⟨hinv, hst', h₁', hl, hn''⟩ := hsame s' h' hsm
        refine Or.inr (Or.inl ⟨by omega, ?_, ?_, ?_⟩)
        · rintro ⟨p, c, t⟩
          by_cases ht : (E (p, c)).map Prod.fst = some t
          · rw [List.countP_eq_zero.2 fun z _ => by
              rw [fR_target read C T E (hst'.edges ▸ hst'.tree ▸ hinv.edges) ht z]; simp]
            omega
          · have := hinv.recs_lt p c t (by rwa [hst'.edges])
            rw [hst'.records] at this
            omega
        · exact h₁' (by simp; omega)
        · have := (look_none C hl).2
          rwa [hst'.probeCount, hst'.searchCount, hst'.middleUndecided] at this
    · rcases h with h | ⟨s', h', hsm, hseg⟩
      · exact absurd h hb
      obtain ⟨hinv, hst', h₁', -, hn''⟩ := hsame s' h' hsm
      have hlt : s'.probeCount < Ns := by
        have : s'.probeCount ≠ Ns := fun hc => hb (Or.inr ⟨s', h', hsm, hc⟩)
        omega
      obtain ⟨i, hi, hiN, hev⟩ := seg_reduce hL hG X Z XEv R hm hX hnt hN₁s hθe xs (ys ++ [x]) s'
        hinv hst' h₁' hlt hseg
      refine ⟨i + 1, by simp; omega, by simp at hiN; omega, ?_⟩
      have heq : ys ++ (x :: xs).take (i + 1 + 1) = ys ++ [x] ++ xs.take (i + 1) := by simp
      rw [heq]
      simp only [List.length_append, List.length_singleton] at hev
      rcases hev with hev | ⟨hev1, hev2⟩ | hev
      · exact Or.inl hev
      · exact Or.inr (Or.inl ⟨by omega, hev2⟩)
      · exact Or.inr (Or.inr hev)

end Reduce

/-! ## The chance of a bad step -/

section Prob

variable [IsProbabilityMeasure D] (T : DTree α) (E : Edges α) (L N₁ : ℕ)

/-- At an edge whose sampler finds at most `q`, with `2q` at most the edge test's least rate per
probe, the test's count passes `edgeExcess` beyond twice its mean at some look with chance at most
`exp(−edgeExcess / (2 (L + 1)))`. -/
theorem edge_any_le (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) (e : List Bool × α) {q : ℝ}
    (hmean : ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ q)
    (hqt : 2 * q ≤ C.edgeRate * C.edgeReadShare) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D).real {xs | ∃ n ≤ N,
        C.edgeExcess + C.edgeRate * C.edgeReadShare * n
          ≤ (((List.ofFn xs).take n).map fun x => (edgeCount read C T E e x : ℝ)).sum}
      ≤ Real.exp (-C.edgeExcess / (2 * (L + 1))) := by
  set m := ∫ x, (edgeCount read C T E e x : ℝ) ∂D
  have hm0 : 0 ≤ m := integral_nonneg fun _ => Nat.cast_nonneg _
  have hb : ∀ᵐ x ∂D, (edgeCount read C T E e x : ℝ) ≤ L + 1 := by
    filter_upwards [hL] with x hx
    have := (undecidedAtEdge_length _ e.1 e.2).trans (positionReads_length read T x E C.walkStart)
    have : edgeCount read C T E e x ≤ L + 1 := this.trans (by omega)
    exact_mod_cast this
  have hany := pi_sum_any D (fun x => (edgeCount read C T E e x : ℝ) - 2 * m)
    (K := 2 * (L + 1)) (by positivity)
    (mgf_count D (edgeCount read C T E e) (B := L + 1) (by positivity) hb le_rfl) N C.edgeExcess
  have hsub : {xs : Fin N → FreeMonoid α | ∃ n ≤ N,
      C.edgeExcess + C.edgeRate * C.edgeReadShare * n
        ≤ (((List.ofFn xs).take n).map fun x => (edgeCount read C T E e x : ℝ)).sum}
      ⊆ {xs | ∃ n ≤ N, C.edgeExcess
          ≤ sumFirst (fun x => (edgeCount read C T E e x : ℝ) - 2 * m) n (List.ofFn xs)} := by
    rintro xs ⟨n, hn, hxs⟩
    refine ⟨n, hn, ?_⟩
    simp only [sumFirst]
    rw [sum_map_sub_const, List.length_take, List.length_ofFn, min_eq_left hn]
    have : 2 * m * n ≤ C.edgeRate * C.edgeReadShare * n := by
      have : (0 : ℝ) ≤ n := Nat.cast_nonneg n
      nlinarith
    linarith
  exact (measureReal_mono hsub).trans (ENNReal.toReal_le_of_le_ofReal (Real.exp_pos _).le hany)

/-- Over a stretch's first `Ns` probes, a test fires falsely at some look with chance at most a
level per sequential test and `exp(−edgeExcess / (2 (L + 1)))` per edge. -/
theorem fire_any_le (hL : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hT : T.leaves.length ≤ C.maxLeaves) {p q : ℝ} (hp0 : 0 ≤ p)
    (hqt : 2 * q ≤ C.edgeRate * C.edgeReadShare) (ha : 0 < C.testLevel)
    (hθs0 : 0 < C.startRate) (hθs : C.startRate < C.startAlternative)
    (hθs1 : C.startAlternative < 1) (hps : p ≤ C.startRate) (hθpt0 : 0 < C.middleRate)
    (hθpt : C.middleRate < C.middleAlternative) (hθpt1 : C.middleAlternative < 1)
    (hfall : p * (C.middleAlternative / C.middleRate)
        * ((1 - C.middleAlternative) / (1 - C.middleRate)) ^ (C.searchShare - 1)
      + (1 - p) * ((1 - C.middleAlternative) / (1 - C.middleRate)) ^ C.searchShare ≤ 1)
    (hεd0 : 0 < C.disagreementAlternative) (hεd : C.disagreementAlternative < C.disagreementRate)
    (hεd1 : C.disagreementRate < 1) (hε : C.disagreementRate ≤ ε) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D).real
        {xs | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N
          ∧ FalseFire read C D ε T E L p q ((List.ofFn xs).take n)}
      ≤ 3 * C.testLevel + (C.maxLeaves * Fintype.card α : ℕ)
          * Real.exp (-C.edgeExcess / (2 * (L + 1))) := by
  classical
  set ex := Real.exp (-C.edgeExcess / (2 * (L + 1)))
  set F := T.leaves.toFinset ×ˢ (Finset.univ : Finset α)
  set A1 := {xs : Fin N → FreeMonoid α | D.real {x | fSt read C T E x} ≤ p ∧ ∃ n ≤ N,
    seqRate C.startRate C.startAlternative C.testLevel n (cnt (fSt read C T E) n (List.ofFn xs))}
  set A2 := {xs : Fin N → FreeMonoid α | ∃ w ∈ (List.ofFn xs).take N, L < w.toList.length}
  set A3 := ⋃ e ∈ F.filter (fun e => ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ q),
    {xs : Fin N → FreeMonoid α | ∃ n ≤ N, C.edgeExcess + C.edgeRate * C.edgeReadShare * n
      ≤ (((List.ofFn xs).take n).map fun x => (edgeCount read C T E e x : ℝ)).sum}
  set A4 := {xs : Fin N → FreeMonoid α | D.real {x | fP read C T E x} ≤ p ∧ ∃ n ≤ N,
    C.searchShare * n ≤ cnt (fS read C T E) n (List.ofFn xs)
    ∧ seqRate C.middleRate C.middleAlternative C.testLevel (cnt (fS read C T E) n (List.ofFn xs))
      (cnt (fP read C T E) n (List.ofFn xs))}
  set A5 := {xs : Fin N → FreeMonoid α | ε < D.real {x | Disagrees read ⟨T, E⟩ C.walkStart x}
    ∧ ∃ n ≤ N, seqRate C.disagreementRate C.disagreementAlternative C.testLevel n
      (cnt (fS read C T E) n (List.ofFn xs))}
  have hsub : {xs : Fin N → FreeMonoid α | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N
      ∧ FalseFire read C D ε T E L p q ((List.ofFn xs).take n)} ⊆ A1 ∪ A2 ∪ A3 ∪ A4 ∪ A5 := by
    rintro xs ⟨n, -, hnN, h⟩
    have hlen : ((List.ofFn xs).take n).length = n := by simp [hnN]
    unfold FalseFire at h
    rw [hlen] at h
    rcases h with ⟨h1, h2⟩ | ⟨w, hw, hw'⟩ | ⟨e, he, hm, hc⟩ | ⟨h1, h2, h3⟩ | ⟨h1, h2⟩
    · exact Or.inl (Or.inl (Or.inl (Or.inl ⟨h1, n, hnN, h2⟩)))
    · refine Or.inl (Or.inl (Or.inl (Or.inr ⟨w, ?_, hw'⟩)))
      rw [List.take_of_length_le (by simp)]
      exact List.mem_of_mem_take hw
    · exact Or.inl (Or.inl (Or.inr (Set.mem_biUnion (Finset.mem_coe.2
        (Finset.mem_filter.2 ⟨he, hm⟩)) ⟨n, hnN, hc⟩)))
    · exact Or.inl (Or.inr ⟨h1, n, hnN, h2, h3⟩)
    · exact Or.inr ⟨h1, n, hnN, h2⟩
  have h1 : (Measure.pi fun _ : Fin N => D).real A1 ≤ C.testLevel := by
    by_cases hc : D.real {x | fSt read C T E x} ≤ p
    · exact real_le_of_sub D ha.le (fun xs h => h.2) (seq_any_le D (fSt read C T E) hθs0
        (by linarith) (by linarith) hθs1 ha (by nlinarith) N)
    · exact real_le_of_not D ha.le hc fun xs h => h.1
  have h2 := long_null D L hL N N
  have h3 : (Measure.pi fun _ : Fin N => D).real A3
      ≤ (C.maxLeaves * Fintype.card α : ℕ) * ex := by
    have hF : ((F.filter fun e => ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ q).card : ℝ)
        ≤ (C.maxLeaves * Fintype.card α : ℕ) := by
      have : (F.filter fun e => ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ q).card
          ≤ C.maxLeaves * Fintype.card α := by
        refine (Finset.card_filter_le _ _).trans ?_
        simp only [F, Finset.card_product, Finset.card_univ]
        exact Nat.mul_le_mul_right _ ((List.toFinset_card_le _).trans hT)
      exact_mod_cast this
    calc _ ≤ ∑ e ∈ F.filter (fun e => ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ q),
          (Measure.pi fun _ : Fin N => D).real {xs : Fin N → FreeMonoid α | ∃ n ≤ N,
            C.edgeExcess + C.edgeRate * C.edgeReadShare * n
              ≤ (((List.ofFn xs).take n).map fun x => (edgeCount read C T E e x : ℝ)).sum} :=
          measureReal_biUnion_finset_le _ _
      _ ≤ ∑ _e ∈ F.filter (fun e => ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ q), ex :=
          Finset.sum_le_sum fun e he =>
            edge_any_le read C D T E L hL e (Finset.mem_filter.1 he).2 hqt N
      _ = _ := by rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ _ := by gcongr
  have h4 : (Measure.pi fun _ : Fin N => D).real A4 ≤ C.testLevel := by
    by_cases hc : D.real {x | fP read C T E x} ≤ p
    · exact real_le_of_sub D ha.le (fun xs h => h.2) (mid_any_le D (fS read C T E)
        (fP read C T E) (fun x hx => searches_of_ptStr (by simp only [fP] at hx; simpa using hx))
        hθpt0 hθpt hθpt1 ha hp0 hc hfall N)
    · exact real_le_of_not D ha.le hc fun xs h => h.1
  have h5 : (Measure.pi fun _ : Fin N => D).real A5 ≤ C.testLevel := by
    by_cases hc : ε < D.real {x | Disagrees read ⟨T, E⟩ C.walkStart x}
    · have hS : D.real {x | Disagrees read ⟨T, E⟩ C.walkStart x} ≤ D.real {x | fS read C T E x} :=
        measureReal_mono fun w hw =>
          show fS read C T E w = true from (disagrees_searches read T w E C.walkStart hw).2
      exact real_le_of_sub D ha.le (fun xs h => h.2) (seq_any_le D (fS read C T E)
        (by linarith) hεd1 hεd0 (by linarith) ha (by nlinarith) N)
    · exact real_le_of_not D ha.le hc fun xs h => h.1
  calc _ ≤ (Measure.pi fun _ : Fin N => D).real (A1 ∪ A2 ∪ A3 ∪ A4 ∪ A5) := measureReal_mono hsub
    _ ≤ (Measure.pi fun _ : Fin N => D).real A1 + (Measure.pi fun _ : Fin N => D).real A2
        + (Measure.pi fun _ : Fin N => D).real A3 + (Measure.pi fun _ : Fin N => D).real A4
        + (Measure.pi fun _ : Fin N => D).real A5 := by
      refine (measureReal_union_le _ _).trans (add_le_add_left ?_ _)
      refine (measureReal_union_le _ _).trans (add_le_add_left ?_ _)
      refine (measureReal_union_le _ _).trans (add_le_add_left ?_ _)
      exact measureReal_union_le _ _
    _ ≤ _ := by rw [h2]; linarith

open scoped Classical in
theorem surv_le {s : Level4State α} (hs : Inv read C s) (hT : s.tree = T) (hE : s.edges = E)
    (hm : 1 ≤ C.recordsToMove) (hN₁s : N₁ ≤ Ns) (hθpt0 : 0 < C.middleRate)
    (hθpt : C.middleRate < C.middleAlternative)
    (hθpt1 : C.middleAlternative < 1) (hεd0 : 0 < C.disagreementAlternative)
    (hεd : C.disagreementAlternative < C.disagreementRate) (hεd1 : C.disagreementRate < 1)
    {θr εd' θpt' : ℝ} (hθr0 : 0 ≤ θr) (hθr1 : θr ≤ 1)
    (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1) (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1)
    (hsep : (C.maxLeaves : ℝ) ^ 2 * Fintype.card α * θr ≤ (1 - θpt') * εd') (N : ℕ) :
    (Measure.pi fun _ : Fin N => D).real
        {xs | Ns ≤ N ∧ Survives read C T E N₁ ((List.ofFn xs).take Ns)}
      ≤ (1 - binomSfGe Ns θr C.recordsToMove)
        + binE N₁ εd' (fun s =>
            if seqRate C.disagreementRate C.disagreementAlternative C.testLevel N₁ s then 0 else 1)
        + (1 - binomSfGe Ns εd' ⌈C.searchShare * Ns⌉₊)
        + ∑ s ∈ Finset.Icc ⌈C.searchShare * Ns⌉₊ Ns,
            binE s θpt' (fun j => if seqRate C.middleRate C.middleAlternative C.testLevel s j then 0
              else 1) := by
  classical
  set sm := ⌈C.searchShare * Ns⌉₊ with hsm
  set silD := fun s : ℕ =>
    if seqRate C.disagreementRate C.disagreementAlternative C.testLevel N₁ s then (0 : ℝ) else 1
  set silM := fun s j : ℕ =>
    if seqRate C.middleRate C.middleAlternative C.testLevel s j then (0 : ℝ) else 1
  have h01 : ∀ (P : Prop) [Decidable P], (0 : ℝ) ≤ if P then 0 else 1 := fun P _ => by
    split_ifs <;> norm_num
  have hB0 : 0 ≤ 1 - binomSfGe Ns θr C.recordsToMove := by
    linarith [binomSfGe_le_one hθr0 hθr1 Ns C.recordsToMove]
  have hC0 : 0 ≤ binE N₁ εd' silD := binE_nonneg hεd'0 hεd'1 fun j => h01 _
  have hD10 : 0 ≤ 1 - binomSfGe Ns εd' sm := by linarith [binomSfGe_le_one hεd'0 hεd'1 Ns sm]
  have hD20 : 0 ≤ ∑ s ∈ Finset.Icc sm Ns, binE s θpt' (silM s) :=
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
        ⊆ {xs | cnt (fR read C T E τ) Ns (List.ofFn xs) < C.recordsToMove} := by
      rintro xs ⟨-, h1, -, -⟩; exact h1 τ
    calc _ ≤ _ := measureReal_mono hsub
      _ = binE Ns (D.real {x | fR read C T E τ x})
          (fun j => if j < C.recordsToMove then 1 else 0) :=
          pi_cnt D _ N Ns hNs (fun j => j < C.recordsToMove)
      _ ≤ binE Ns θr (fun j => if j < C.recordsToMove then 1 else 0) :=
          binE_le_of_le_anti hθr0 hτ measureReal_le_one Ns _ fun i j h => by
            by_cases hj : j < C.recordsToMove
            · rw [if_pos hj, if_pos (by omega)]
            · rw [if_neg hj]; split_ifs <;> norm_num
      _ = 1 - binomSfGe Ns θr C.recordsToMove := (one_sub_binomSfGe' hθr0 hθr1 hm).symm
      _ ≤ _ := by linarith
  push Not at hB
  set q := D.real {x | fS read C T E x}
  by_cases hC : q ≤ εd'
  · have hsub : {xs : Fin N → FreeMonoid α | Ns ≤ N ∧
        Survives read C T E N₁ ((List.ofFn xs).take Ns)}
        ⊆ {xs | ¬ seqRate C.disagreementRate C.disagreementAlternative C.testLevel N₁
            (cnt (fS read C T E) N₁ (List.ofFn xs))} := by
      rintro xs ⟨-, -, h2, -⟩
      rw [htake] at h2
      exact h2
    calc _ ≤ _ := measureReal_mono hsub
      _ = binE N₁ q (fun j => if ¬ seqRate C.disagreementRate C.disagreementAlternative
            C.testLevel N₁ j then 1 else 0) := pi_cnt D _ N N₁ (hN₁s.trans hNs)
          (fun j => ¬ seqRate C.disagreementRate C.disagreementAlternative C.testLevel N₁ j)
      _ = binE N₁ q silD := binE_congr fun j _ => by
          by_cases h : seqRate C.disagreementRate C.disagreementAlternative C.testLevel N₁ j <;>
            simp [silD, h]
      _ ≤ binE N₁ εd' silD := binE_le_of_le_on measureReal_nonneg hC hεd'1 N₁ _
          fun i j hij hj => by
            have := likelihoodRatio_anti (by linarith) hεd0.le hεd.le hεd1 hij hj
            by_cases hfj : seqRate C.disagreementRate C.disagreementAlternative C.testLevel N₁ j
            · simp only [silD]
              have hfi : seqRate C.disagreementRate C.disagreementAlternative C.testLevel N₁ i :=
                le_trans hfj this
              rw [if_pos hfj, if_pos hfi]
            · simp only [silD]
              rw [if_neg hfj]; split_ifs <;> norm_num
      _ ≤ _ := by linarith
  push Not at hC
  set pP := D.real {x | fP read C T E x}
  have hEOK : EdgesOK read T E := hT ▸ hE ▸ hs.edges
  have hcov : {x | fS read C T E x = true}
      ⊆ {x | fP read C T E x = true} ∪ ⋃ τ ∈ Rset, {x | fR read C T E τ x = true} := by
    intro x hx
    simp only [Set.mem_ofPred_eq, fS] at hx
    have hOK := Ideal.probeOrAgree_ok false read T E C.walkStart hEOK x
    generalize hg : probeOrAgree false read T E C.walkStart x = o at hx hOK
    cases o with
    | wrongEdge p c u t =>
      obtain ⟨hp, -, huc, -⟩ := hOK
      refine Or.inr (Set.mem_biUnion (x := (p, c, t)) ?_ ?_)
      · simp [Rset, hp, DTree.sift_inl_mem read T _ t huc]
      · simp [fR, hg, recOf]
    | undecided zs cl =>
      cases cl with
      | true => simp [searches] at hx
      | false =>
        obtain ⟨hzs, -⟩ := hOK
        refine Or.inl ?_
        simp only [Set.mem_ofPred_eq, fP, hg, middleString]
        cases zs with
        | nil => exact absurd rfl (hzs rfl)
        | cons z zs => simp
    | agree | undecidedStart _ | unlearnedEdge _ _ _ _ => simp [searches] at hx
  have hRcard : (Rset.card : ℝ) ≤ (C.maxLeaves : ℝ) ^ 2 * Fintype.card α := by
    have h1 : T.leaves.toFinset.card ≤ C.maxLeaves :=
      (List.toFinset_card_le _).trans (hT ▸ hs.size)
    have : Rset.card ≤ C.maxLeaves ^ 2 * Fintype.card α := by
      simp only [Rset, Finset.card_product, Finset.card_univ]
      calc _ ≤ C.maxLeaves * (Fintype.card α * C.maxLeaves) := by gcongr
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
      calc _ ≤ (C.maxLeaves : ℝ) ^ 2 * Fintype.card α * θr := by gcongr
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
        ∪ {xs | sm ≤ cnt (fS read C T E) Ns (List.ofFn xs)
          ∧ ¬ seqRate C.middleRate C.middleAlternative C.testLevel
            (cnt (fS read C T E) Ns (List.ofFn xs)) (cnt (fP read C T E) Ns (List.ofFn xs))} := by
    rintro xs ⟨-, -, -, h3⟩
    rw [hlen] at h3
    by_cases hS : cnt (fS read C T E) Ns (List.ofFn xs) < sm
    · exact Or.inl hS
    · refine Or.inr ⟨not_lt.1 hS, fun hf => h3 ⟨?_, hf⟩⟩
      exact Nat.ceil_le.1 (not_lt.1 hS)
  calc _ ≤ _ := measureReal_mono hsub
    _ ≤ _ := measureReal_union_le _ _
    _ ≤ (1 - binomSfGe Ns εd' sm) + ∑ s ∈ Finset.Icc sm Ns, binE s θpt' (silM s) := by
      gcongr
      · calc _ = binE Ns q (fun j => if j < sm then 1 else 0) :=
            pi_cnt D _ N Ns hNs (fun j => j < sm)
          _ ≤ binE Ns εd' (fun j => if j < sm then 1 else 0) :=
              binE_le_of_le_anti hεd'0 hC.le measureReal_le_one Ns _ fun i j h => by
                by_cases hj : j < sm
                · rw [if_pos hj, if_pos (by omega)]
                · rw [if_neg hj]; split_ifs <;> norm_num
          _ = _ := (one_sub_binomSfGe_lt hεd'0 hεd'1 Ns sm).symm
      · rw [pi_cnt2 D (fS read C T E) (fP read C T E) (fun x hx => searches_of_ptStr (by
            simp only [fP] at hx; simpa using hx)) N Ns hNs
          (fun s j => sm ≤ s ∧ ¬ seqRate C.middleRate C.middleAlternative C.testLevel s j)]
        calc _ ≤ binE Ns q (fun _ => ∑ s ∈ Finset.Icc sm Ns, binE s θpt' (silM s)) := by
              refine binE_mono_le measureReal_nonneg measureReal_le_one fun s hs => ?_
              by_cases hss : sm ≤ s
              · calc _ = binE s (pP / q) (silM s) := binE_congr fun j _ => by
                      by_cases h : seqRate C.middleRate C.middleAlternative C.testLevel s j <;>
                        simp [silM, hss, h]
                  _ ≤ binE s θpt' (silM s) :=
                      binE_le_of_le_anti_on hθpt'0 hπ0 hπ1 s _ fun i j hij hj => by
                        have := likelihoodRatio_mono hθpt0 (by linarith) hθpt.le hθpt1.le hij hj
                        by_cases hfi : seqRate C.middleRate C.middleAlternative C.testLevel s i
                        · simp only [silM]
                          have hfj : seqRate C.middleRate C.middleAlternative C.testLevel s j :=
                            le_trans hfi this
                          rw [if_pos hfi, if_pos hfj]
                        · simp only [silM]
                          rw [if_neg hfi]; split_ifs <;> norm_num
                  _ ≤ _ := Finset.single_le_sum (f := fun s => binE s θpt' (silM s))
                      (fun s _ => binE_nonneg hθpt'0 hθpt'1 fun j => h01 _)
                      (Finset.mem_Icc.2 ⟨hss, hs⟩)
              · calc _ = binE s (pP / q) (fun _ => 0) := binE_congr fun j _ => by simp [hss]
                  _ = 0 := binE_const _ _ _
                  _ ≤ _ := hD20
          _ = _ := binE_const _ _ _
    _ ≤ _ := by linarith

end Prob

/-- What a stretch risks: a bad step of the round's own kinds, at most `stretchRisk`, or one of
the steps `X`, which show in the counts as `XEv` with chance at most `bX`. -/
theorem seg_le (X : Level4State α → FreeMonoid α → Prop) (XEv : List (FreeMonoid α) → Prop)
    (bX : ℝ)
    [IsProbabilityMeasure D] (L N₁ : ℕ) (G θr εd' θpt' : ℝ) (hL : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (Z : Level4State α → FreeMonoid α → Prop)
    {s : Level4State α} (hs : Inv read C s) (hf : s = newHypothesis s.tree s.edges s.records)
    (hT : s.tree ∈ (classSet (Fintype.card σ) : Finset (DTree α)))
    (hN : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
      D.real (PotGood M U θ C.walkStart read T) ≤ G)
    (hX : ∀ ys s' x, Inv read C s' → StInv read C s.tree s.edges s.records ys s' → X s' x
      → XEv (ys ++ [x]))
    (hnt : ∀ ys s' x, Inv read C s' → StInv read C s.tree s.edges s.records ys s'
      → ¬ X s' x → ¬ Z s' x → applyProbe read C s' x ≠ .inr .failed)
    (hbX : ∀ N, (Measure.pi fun _ : Fin N => D).real
      {xs | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N ∧ XEv ((List.ofFn xs).take n)} ≤ bX)
    (hm : 1 ≤ C.recordsToMove) (hN₁ : 1 ≤ N₁) (hN₁s : N₁ ≤ Ns) (hG0 : 0 ≤ G)
    (hGt : 4 * (L + 1) * G ≤ C.edgeRate * C.edgeReadShare)
    (hyt : 2 * y ≤ C.edgeRate * C.edgeReadShare) (ha : 0 < C.testLevel)
    (hθs0 : 0 < C.startRate) (hθs : C.startRate < C.startAlternative)
    (hθs1 : C.startAlternative < 1) (hps : max (2 * G) y ≤ C.startRate) (hθe : 0 ≤ C.edgeRate)
    (hθpt0 : 0 < C.middleRate) (hθpt : C.middleRate < C.middleAlternative)
    (hθpt1 : C.middleAlternative < 1)
    (hfall : max (2 * G) y * (C.middleAlternative / C.middleRate)
        * ((1 - C.middleAlternative) / (1 - C.middleRate)) ^ (C.searchShare - 1)
      + (1 - max (2 * G) y) * ((1 - C.middleAlternative) / (1 - C.middleRate)) ^ C.searchShare
      ≤ 1)
    (hεd0 : 0 < C.disagreementAlternative) (hεd : C.disagreementAlternative < C.disagreementRate)
    (hεd1 : C.disagreementRate < 1) (hε : C.disagreementRate ≤ ε) (hθr0 : 0 ≤ θr) (hθr1 : θr ≤ 1)
    (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1) (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1)
    (hsep : (C.maxLeaves : ℝ) ^ 2 * Fintype.card α * θr ≤ (1 - θpt') * εd') (N : ℕ) :
    (Measure.pi fun _ : Fin N => D)
        {xs | SegBZ (applyProbe read C) (BadStep read C M U θ D ε Ns y X) Z Same s (List.ofFn xs)}
      ≤ ENNReal.ofReal
          (stretchRisk C (Fintype.card α) L Ns N₁ θr εd' θpt' + bX) := by
  classical
  have hp0 : 0 ≤ max (2 * G) y := le_max_of_le_left (by positivity)
  have hqt : 2 * max (2 * (L + 1) * G) y ≤ C.edgeRate * C.edgeReadShare := by
    rcases max_cases (2 * (L + 1) * G) y with ⟨h, -⟩ | ⟨h, -⟩ <;> rw [h] <;> linarith
  have hsub : {xs : Fin N → FreeMonoid α |
      SegBZ (applyProbe read C) (BadStep read C M U θ D ε Ns y X) Z Same s (List.ofFn xs)}
      ⊆ ({xs | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N ∧
          FalseFire read C D ε s.tree s.edges L (max (2 * G) y) (max (2 * (L + 1) * G) y)
            ((List.ofFn xs).take n)}
        ∪ {xs | Ns ≤ N ∧ Survives read C s.tree s.edges N₁ ((List.ofFn xs).take Ns)})
        ∪ {xs | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N ∧ XEv ((List.ofFn xs).take n)} := by
    intro xs hxs
    have hst0 : StInv read C s.tree s.edges s.records [] s := by
      have := stInv_fresh read C s.tree s.edges s.records
      rwa [← hf] at this
    have h₁0 : Look₁ read C s.tree s.edges N₁ [] := fun h => by simp at h; omega
    have hn0 : s.probeCount < Ns := by rw [hf]; simp only [newHypothesis]; omega
    obtain ⟨i, hi, hiN, h⟩ := seg_reduce read C M U θ D ε Ns y s.tree s.edges L N₁ hL (hN _ hT)
      X Z XEv s.records hm (fun ys s' x hs' hst' hx => hX ys s' x hs' hst' hx) hnt hN₁s hθe
      (List.ofFn xs) [] s hs hst0 h₁0 hn0 hxs
    simp only [List.nil_append, List.length_nil, zero_add, List.length_ofFn] at hi hiN h
    rcases h with h | ⟨hNs', h⟩ | h
    · exact Or.inl (Or.inl ⟨i + 1, Finset.mem_Icc.2 ⟨by omega, hiN⟩, by omega, h⟩)
    · refine Or.inl (Or.inr ⟨by omega, ?_⟩)
      rwa [← hNs']
    · exact Or.inr ⟨i + 1, Finset.mem_Icc.2 ⟨by omega, hiN⟩, by omega, h⟩
  rw [← ENNReal.ofReal_toReal (measure_ne_top _ _), ← measureReal_def]
  apply ENNReal.ofReal_le_ofReal
  calc _ ≤ _ := measureReal_mono hsub
    _ ≤ _ := measureReal_union_le _ _
    _ ≤ _ := add_le_add (measureReal_union_le _ _) (hbX N)
    _ ≤ (3 * C.testLevel + (C.maxLeaves * Fintype.card α : ℕ)
            * Real.exp (-C.edgeExcess / (2 * (L + 1))))
        + ((1 - binomSfGe Ns θr C.recordsToMove)
          + binE N₁ εd' (fun s =>
              if seqRate C.disagreementRate C.disagreementAlternative C.testLevel N₁ s then 0
              else 1)
          + (1 - binomSfGe Ns εd' ⌈C.searchShare * Ns⌉₊)
          + ∑ s ∈ Finset.Icc ⌈C.searchShare * Ns⌉₊ Ns,
              binE s θpt' (fun j => if seqRate C.middleRate C.middleAlternative C.testLevel s j
                then 0 else 1))
        + bX := by
      gcongr
      · exact fire_any_le read C D ε Ns s.tree s.edges L hL hs.size hp0 hqt ha hθs0 hθs hθs1 hps
          hθpt0 hθpt hθpt1 hfall hεd0 hεd hεd1 hε N
      · exact surv_le read C D Ns s.tree s.edges N₁ hs rfl rfl hm hN₁s hθpt0 hθpt hθpt1 hεd0 hεd
          hεd1 hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep N
    _ = _ := by
      unfold stretchRisk
      ring

end Eps

end Random

end OrthoDFA
