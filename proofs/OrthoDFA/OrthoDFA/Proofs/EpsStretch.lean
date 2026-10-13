import OrthoDFA.Proofs.EpsSteps
import OrthoDFA.Proofs.RandomStretch

/-!
# One stretch of level 3.5

Level 3's stretch, with records: a stretch that reaches `Ns` probes has fewer than
`recordsToMove` records at every edge and target, no success at look `N₁`, and no harvest of
undecided middles at look `Ns`.
-/

namespace OrthoDFA

namespace Random

namespace Eps

open MeasureTheory
open OrthoDFA.Ideal (Outcome EdgesOK OutcomeOK probeOrAgree)

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3_5Settings)
  {σ : Type*} [Fintype σ] (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) (D : Measure (FreeMonoid α))
  (ε : ℝ) (Ns : ℕ) (y : ℝ)

def Same (s s' : Level3_5State α) : Prop := s'.tree = s.tree ∧ s'.edges = s.edges

/-- Within the first `Ns` probes of its stretch the step ends the round badly, a harvest's sampler
yielding `y`, or is one of the steps `X`, or its stretch reaches `Ns` probes. -/
def BadStep (X : Level3_5State α → FreeMonoid α → Prop) (s : Level3_5State α) (x : FreeMonoid α) :
    Prop :=
  (s.probeCount < Ns ∧ ((∃ e, applyProbe read C s x = .inr e ∧ ¬ EndsWell M (BadAt U θ) D ε
      {w | Disagrees read ⟨s.tree, s.edges⟩ C.walkStart w} C.walkStart
      (harvestLaw read D C.walkStart ⟨s.tree, s.edges⟩) y e)
      ∨ X s x))
  ∨ ∃ s', applyProbe read C s x = .inl s' ∧ Same s s' ∧ s'.probeCount = Ns

/-! ## A step within a stretch -/

/-- The state a probe leaves before the tests look at it. -/
noncomputable def chg (s : Level3_5State α) (x : FreeMonoid α) : Level3_5State α :=
  countProbe read C { s with records := s.records ++
      (recOf (probeOrAgree read s.tree s.edges C.walkStart x)).toList } x

section StepFacts

theorem finish_inr {t : Level3_5State α} {e : RoundEnd (Level3Site α)}
    (h : applyTests C t = .inr e) :
    runTests C t = some e := by
  unfold applyTests at h
  split at h
  · rename_i e' he'; simp only [Sum.inr.injEq] at h; rw [he', h]
  · simp at h

theorem mk_eta (s : Level3_5State α) :
    (⟨s.tree, s.edges, s.redirected, s.records, s.probeCount, s.searchCount, s.startUndecided,
        s.middleUndecided, s.edgeReads, s.edgeUndecided⟩ : Level3_5State α)
      = s := rfl

theorem chg_of_none {s : Level3_5State α} {x : FreeMonoid α}
    (h : recOf (probeOrAgree read s.tree s.edges C.walkStart x) = none) :
    chg read C s x = countProbe read C s x := by
  unfold chg
  rw [h]
  simp only [Option.toList_none, List.append_nil]

theorem step_inr {s : Level3_5State α} {x : FreeMonoid α}
    (hne : applyProbe read C s x ≠ .inr .failed)
    {e : RoundEnd (Level3Site α)} (h : applyProbe read C s x = .inr e) :
    runTests C (chg read C s x) = some e := by
  have h' := h
  unfold applyProbe at h
  rcases ho : probeOrAgree read s.tree s.edges C.walkStart x with _ | ⟨p, c, u, t⟩ | ⟨p, c, u, t⟩
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

theorem step_same_chg {s s' : Level3_5State α} (hs : Inv read C s) {x : FreeMonoid α}
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

theorem look_none {t : Level3_5State α} (h : runTests C t = none) :
    rateSide C.disagreementRate C.testLevel C.firstLook t.probeCount t.searchCount ≠ some false
      ∧ ¬ (C.searchShare * t.probeCount ≤ t.searchCount
          ∧ rateSide C.middleRate C.testLevel C.firstLook t.searchCount t.middleUndecided.length
          = some true) := by
  unfold runTests at h
  split_ifs at h with h1 h2 h3 h4
  exact ⟨h4, h3⟩

theorem look_some {t : Level3_5State α} {e : RoundEnd (Level3Site α)} (h : runTests C t = some e) :
    (rateSide C.startRate C.testLevel C.firstLook t.probeCount t.startUndecided.length = some true
        ∧ e = .harvest .start)
    ∨ (∃ p c, (p, c) ∈ t.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α)
        ∧ C.edgeReadShare * t.probeCount ≤ t.edgeReads p c
        ∧ C.edgeExcess + C.edgeRate * t.edgeReads p c ≤ (t.edgeUndecided p c).length
        ∧ e = .harvest (.edge p c))
    ∨ (C.searchShare * t.probeCount ≤ t.searchCount
        ∧ rateSide C.middleRate C.testLevel C.firstLook t.searchCount t.middleUndecided.length
        = some true
        ∧ e = .harvest .middles)
    ∨ (rateSide C.disagreementRate C.testLevel C.firstLook t.probeCount t.searchCount = some false
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

def fS (x : FreeMonoid α) : Bool := searches (probeOrAgree read T E C.walkStart x)

def fSt (x : FreeMonoid α) : Bool := !(startString (probeOrAgree read T E C.walkStart x)).isEmpty

def fP (x : FreeMonoid α) : Bool := !(middleString (probeOrAgree read T E C.walkStart x)).isEmpty

/-- The strings a probe charges to the edge `e`. -/
noncomputable def edgeCount (e : List Bool × α) (x : FreeMonoid α) : ℕ :=
  (undecidedAtEdge (positionReads read T E C.walkStart x) e.1 e.2).length

open scoped Classical in
noncomputable def fDis (x : FreeMonoid α) : Bool := decide (Disagrees read ⟨T, E⟩ C.walkStart x)

def fR (τ : List Bool × α × List Bool) (x : FreeMonoid α) : Bool :=
  decide (recOf (probeOrAgree read T E C.walkStart x) = some τ)

variable (N₁ : ℕ)

/-- After the probes `ys` of a stretch at `(T, E)`, the counts are counts over `ys`. -/
structure StInv (ys : List (FreeMonoid α)) (s : Level3_5State α) : Prop where
  tree : s.tree = T
  edges : s.edges = E
  probeCount : s.probeCount = ys.length
  searchCount : s.searchCount = ys.countP (fS read C T E)
  startUndecided : s.startUndecided.length = ys.countP (fSt read C T E)
  middleUndecided : s.middleUndecided.length = ys.countP (fP read C T E)
  edgeUndecided : ∀ p c, (s.edgeUndecided p c).length = (ys.map (edgeCount read C T E (p, c))).sum
  records : ∀ τ, s.records.count τ = ys.countP (fR read C T E τ)

/-- The stretch's success test did not fire at look `N₁`. -/
def Look₁ (ys : List (FreeMonoid α)) : Prop :=
  N₁ ≤ ys.length → rateSide C.disagreementRate C.testLevel C.firstLook N₁
      ((ys.take N₁).countP (fS read C T E)) ≠ some false

theorem stInv_fresh (mv : List Bool × α → Bool) :
    StInv read C T E [] (newHypothesis T E mv) where
  tree := rfl
  edges := rfl
  probeCount := rfl
  searchCount := rfl
  startUndecided := rfl
  middleUndecided := rfl
  edgeUndecided := fun p c => by simp [newHypothesis]
  records := fun τ => by simp [newHypothesis]

/-- A probe counted into a stretch whose counts are its probes' keeps them so. -/
theorem stInv_chg {ys : List (FreeMonoid α)} {s : Level3_5State α}
    (hst : StInv read C T E ys s) (x : FreeMonoid α) :
    StInv read C T E (ys ++ [x]) (chg read C s x) := by
  classical
  obtain ⟨htree, hedges, hn, hdis, hst', hpt, hund, hrecs⟩ := hst
  subst htree hedges
  refine ⟨rfl, rfl, by simp [chg, countProbe, hn], ?_, ?_, ?_, ?_, ?_⟩
  · simp only [chg, countProbe, hdis, countP_snoc]
    cases hb : searches (probeOrAgree read s.tree s.edges C.walkStart x) <;> simp [fS, hb]
  · simp only [chg, countProbe, List.length_append, hst', countP_snoc]
    rw [length_le_one_eq _ (startString_length _)]
    cases hb : (startString (probeOrAgree read s.tree s.edges C.walkStart x)).isEmpty <;>
      simp [fSt, hb]
  · simp only [chg, countProbe, List.length_append, hpt, countP_snoc]
    rw [length_le_one_eq _ (middleString_length _)]
    cases hb : (middleString (probeOrAgree read s.tree s.edges C.walkStart x)).isEmpty <;>
      simp [fP, hb]
  · intro p c
    simp only [chg, countProbe, List.length_append, hund, List.map_append, List.sum_append,
      List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
    rfl
  · intro τ
    simp only [chg, countProbe, List.count_append, hrecs, countP_snoc]
    cases hr : recOf (probeOrAgree read s.tree s.edges C.walkStart x) with
    | none => simp [fR, hr]
    | some τ' => by_cases h : τ' = τ <;> simp [fR, hr, h, List.count_singleton]

theorem look₁_snoc {ys : List (FreeMonoid α)} {s : Level3_5State α} {x : FreeMonoid α}
    (hst : StInv read C T E (ys ++ [x]) s) (h₀ : Look₁ read C T E N₁ ys)
    (h₁ : N₁ = ys.length + 1 → rateSide C.disagreementRate C.testLevel C.firstLook N₁ s.searchCount
      ≠ some false) :
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
  (C.firstLook ≤ zs.length ∧ D.real {x | fSt read C T E x} ≤ p
      ∧ binomSfGe zs.length C.startRate (zs.countP (fSt read C T E)) < C.testLevel)
  ∨ (∃ w ∈ zs, L < w.toList.length)
  ∨ (∃ e ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α),
      ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ q
      ∧ C.edgeExcess + C.edgeRate * C.edgeReadShare * zs.length
        ≤ (zs.map fun x => (edgeCount read C T E e x : ℝ)).sum)
  ∨ (D.real {x | fP read C T E x} ≤ p
      ∧ binomSfGe (max C.firstLook ⌈C.searchShare * zs.length⌉₊) C.middleRate
        (zs.countP (fP read C T E)) < C.testLevel)
  ∨ (ε < D.real {x | Disagrees read ⟨T, E⟩ C.walkStart x} ∧ C.firstLook ≤ zs.length
      ∧ 1 - binomSfGe zs.length C.disagreementRate (zs.countP (fDis read C T E) + 1) < C.testLevel)

/-- The stretch reached `zs.length` probes: fewer than `recordsToMove` records at every edge and
target, no success at look `N₁`, and no harvest of undecided middles at the last look. -/
def Survives (zs : List (FreeMonoid α)) : Prop :=
  (∀ τ, zs.countP (fR read C T E τ) < C.recordsToMove)
  ∧ rateSide C.disagreementRate C.testLevel C.firstLook N₁ ((zs.take N₁).countP (fS read C T E))
      ≠ some false
  ∧ ¬ (C.searchShare * zs.length ≤ zs.countP (fS read C T E)
      ∧ rateSide C.middleRate C.testLevel C.firstLook (zs.countP (fS read C T E))
          (zs.countP (fP read C T E))
        = some true)

/-- A step that ends the round badly, not too big, fires a test falsely. -/
theorem falseFire_of_end [IsProbabilityMeasure D] (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) {G : ℝ}
    (hG : D.real (PotGood M U θ C.walkStart read T) ≤ G)
    (hθs0 : 0 ≤ C.startRate) (hθs1 : C.startRate ≤ 1) (hθe : 0 ≤ C.edgeRate)
    (hθpt0 : 0 ≤ C.middleRate) (hθpt1 : C.middleRate ≤ 1) (hεd0 : 0 ≤ C.disagreementRate)
    (hεd1 : C.disagreementRate ≤ 1) {ys : List (FreeMonoid α)} {s : Level3_5State α}
    (hst : StInv read C T E ys s) {x : FreeMonoid α}
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
  · obtain ⟨hn₀, hsf⟩ := rateSide_true hr
    rw [hlen] at hn₀ hsf
    have ht := hone .start (fun x => startString_length _) hbad
    rw [harvestLaw_real_univ read D C.walkStart ⟨T, E⟩ hL, integral_length_le_one D
      (siteHarvest read C.walkStart ⟨T, E⟩ .start) fun x => startString_length _] at ht
    refine Or.inl ⟨hn₀, ht, ?_⟩
    rwa [hst'.startUndecided] at hsf
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
  · obtain ⟨hn₀, hsf⟩ := rateSide_true hr
    have ht := hone .middles (fun x => middleString_length _) hbad
    rw [harvestLaw_real_univ read D C.walkStart ⟨T, E⟩ hL, integral_length_le_one D
      (siteHarvest read C.walkStart ⟨T, E⟩ .middles) fun x => middleString_length _] at ht
    refine Or.inr (Or.inr (Or.inr (Or.inl ⟨ht, ?_⟩)))
    have hceil : ⌈C.searchShare * ((ys ++ [x]).length : ℝ)⌉₊ ≤ (chg read C s x).searchCount := by
      rw [Nat.ceil_le, ← hlen]; exact hq
    calc _ ≤ binomSfGe (chg read C s x).searchCount C.middleRate
            ((ys ++ [x]).countP (fP read C T E)) :=
          binomSfGe_mono_n hθpt0 hθpt1 (max_le hn₀ hceil) _
      _ < C.testLevel := by rwa [hst'.middleUndecided] at hsf
  · obtain ⟨hn₀, hsf⟩ := rateSide_false hr
    have hD : ε < D.real {w | Disagrees read ⟨T, E⟩ C.walkStart w} := by
      simpa [EndsWell] using hbad
    rw [hlen] at hn₀ hsf
    refine Or.inr (Or.inr (Or.inr (Or.inr ⟨hD, hn₀, ?_⟩)))
    have hle : (ys ++ [x]).countP (fDis read C T E) ≤ (ys ++ [x]).countP (fS read C T E) :=
      List.countP_mono_left fun w _ hw => by
        simp only [fDis, decide_eq_true_eq] at hw
        exact (disagrees_searches read T w E C.walkStart hw).2
    rw [← hst'.searchCount] at hle
    have := binomSfGe_anti hεd0 hεd1 (n := (ys ++ [x]).length) (Nat.succ_le_succ hle)
    linarith

/-- A bad step in a stretch shows in its counts. -/
theorem seg_reduce [IsProbabilityMeasure D] (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) {G : ℝ}
    (hG : D.real (PotGood M U θ C.walkStart read T) ≤ G)
    (X : Level3_5State α → FreeMonoid α → Prop) (XEv : List (FreeMonoid α) → Prop)
    (mv : List Bool × α → Bool)
    (hX : ∀ ys s x, Inv read C s → StInv read C T E ys s → X s x → XEv (ys ++ [x]))
    (hnt : ∀ ys s x, Inv read C s → StInv read C T E ys s → s.redirected = mv → ¬ X s x
      → applyProbe read C s x ≠ .inr .failed) (hN₁s : N₁ ≤ Ns) (hθs0 : 0 ≤ C.startRate)
    (hθs1 : C.startRate ≤ 1) (hθe : 0 ≤ C.edgeRate) (hθpt0 : 0 ≤ C.middleRate)
    (hθpt1 : C.middleRate ≤ 1) (hεd0 : 0 ≤ C.disagreementRate) (hεd1 : C.disagreementRate ≤ 1) :
    ∀ (xs ys : List (FreeMonoid α)) (s : Level3_5State α), Inv read C s →
      StInv read C T E ys s → Look₁ read C T E N₁ ys → s.redirected = mv
          → s.probeCount < Ns →
      SegB (applyProbe read C) (BadStep read C M U θ D ε Ns y X) Same s xs →
      ∃ i, i < xs.length ∧ ys.length + i + 1 ≤ Ns ∧
        (FalseFire read C D ε T E L (max (2 * G) y) (max (2 * (L + 1) * G) y)
            (ys ++ xs.take (i + 1))
          ∨ (ys.length + i + 1 = Ns ∧ Survives read C T E N₁ (ys ++ xs.take (i + 1)))
          ∨ XEv (ys ++ xs.take (i + 1)))
  | [], _, _, _, _, _, _, _, h => by simp [SegB] at h
  | x :: xs, ys, s, hs, hst, h₁, hmv, hn, h => by
    have hn' : s.probeCount = ys.length := hst.probeCount
    have hsame : ∀ s', applyProbe read C s x = .inl s' → Same s s' →
        Inv read C s' ∧ StInv read C T E (ys ++ [x]) s' ∧ Look₁ read C T E N₁ (ys ++ [x])
          ∧ runTests C s' = none ∧ s'.probeCount = ys.length + 1 ∧ s'.redirected = mv := by
      intro s' h' hsm
      obtain ⟨rfl, hl, hinv⟩ := step_same_chg read C hs h' hsm
      have hst' := stInv_chg read C T E hst x
      refine ⟨hinv, hst', look₁_snoc read C T E N₁ hst' h₁ fun hN => ?_, hl, ?_, hmv⟩
      · have := (look_none C hl).1
        rwa [show (chg read C s x).probeCount = N₁ by rw [hst'.probeCount]; simp; omega] at this
      · rw [hst'.probeCount]; simp
    by_cases hb : BadStep read C M U θ D ε Ns y X s x
    · refine ⟨0, by simp, by omega, ?_⟩
      simp only [zero_add, List.take_succ_cons, List.take_zero]
      rcases hb with ⟨-, ⟨e, he, hbad⟩ | hx⟩ | ⟨s', h', hsm, hNs⟩
      · by_cases hx : X s x
        · exact Or.inr (Or.inr (hX ys s x hs hst hx))
        · exact Or.inl (falseFire_of_end read C M U θ D ε y T E L hL hG hθs0 hθs1 hθe hθpt0
            hθpt1 hεd0 hεd1 hst (hnt ys s x hs hst hmv hx) he hbad)
      · exact Or.inr (Or.inr (hX ys s x hs hst hx))
      · obtain ⟨hinv, hst', h₁', hl, hn'', -⟩ := hsame s' h' hsm
        refine Or.inr (Or.inl ⟨by omega, ?_, ?_, ?_⟩)
        · intro τ
          rw [← hst'.records]
          exact hinv.recs_lt τ
        · exact h₁' (by simp; omega)
        · have := (look_none C hl).2
          rwa [hst'.probeCount, hst'.searchCount, hst'.middleUndecided] at this
    · rcases h with h | ⟨s', h', hsm, hseg⟩
      · exact absurd h hb
      obtain ⟨hinv, hst', h₁', -, hn'', hmv'⟩ := hsame s' h' hsm
      have hlt : s'.probeCount < Ns := by
        have : s'.probeCount ≠ Ns := fun hc => hb (Or.inr ⟨s', h', hsm, hc⟩)
        omega
      obtain ⟨i, hi, hiN, hev⟩ := seg_reduce hL hG X XEv mv hX hnt hN₁s hθs0 hθs1 hθe hθpt0
        hθpt1 hεd0 hεd1 xs (ys ++ [x]) s' hinv hst' h₁' hmv' hlt hseg
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
probe, the test's count passes `edgeExcess` beyond twice its mean with chance at most
`exp(−edgeExcess / (2 (L + 1)))`. -/
theorem edge_fire_le (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) (e : List Bool × α) {q : ℝ}
    (hmean : ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ q)
    (hqt : 2 * q ≤ C.edgeRate * C.edgeReadShare) {N n : ℕ} (hn : n ≤ N) :
    (Measure.pi fun _ : Fin N => D).real {xs |
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
  have htail := pi_sum_tail D (fun x => (edgeCount read C T E e x : ℝ) - 2 * m)
    (K := 2 * (L + 1)) (by positivity)
    (mgf_count D (edgeCount read C T E e) (B := L + 1) (by positivity) hb le_rfl) N n
    C.edgeExcess
  have hsub : {xs : Fin N → FreeMonoid α |
      C.edgeExcess + C.edgeRate * C.edgeReadShare * n
        ≤ (((List.ofFn xs).take n).map fun x => (edgeCount read C T E e x : ℝ)).sum}
      ⊆ {xs | C.edgeExcess ≤ sumFirst (fun x => (edgeCount read C T E e x : ℝ) - 2 * m) n
          (List.ofFn xs)} := by
    intro xs hxs
    simp only [Set.mem_ofPred_eq, sumFirst] at hxs ⊢
    rw [sum_map_sub_const, List.length_take, List.length_ofFn, min_eq_left hn]
    have : 2 * m * n ≤ C.edgeRate * C.edgeReadShare * n := by
      have : (0 : ℝ) ≤ n := Nat.cast_nonneg n
      nlinarith
    linarith
  refine (measureReal_mono hsub).trans (ENNReal.toReal_le_of_le_ofReal (Real.exp_pos _).le ?_)
  refine htail.trans (le_of_eq ?_)
  congr 2

theorem fire_le (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) (hT : T.leaves.length ≤ C.maxLeaves)
    {p q : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hqt : 2 * q ≤ C.edgeRate * C.edgeReadShare)
    (ha : 0 ≤ C.testLevel) (hθs0 : 0 ≤ C.startRate) (hθs1 : C.startRate ≤ 1)
    (hθpt0 : 0 ≤ C.middleRate) (hθpt1 : C.middleRate ≤ 1) (hεd0 : 0 ≤ C.disagreementRate)
    (hεd1 : C.disagreementRate ≤ 1) (hεd : C.disagreementRate ≤ ε) (N n : ℕ) :
    (Measure.pi fun _ : Fin N => D).real
        {xs | n ≤ N ∧ FalseFire read C D ε T E L p q ((List.ofFn xs).take n)}
      ≤ (if C.firstLook ≤ n then binomSfGe n p (hFire C.startRate C.testLevel n)
          + C.testLevel else 0)
        + ((C.maxLeaves * Fintype.card α : ℕ) * Real.exp (-C.edgeExcess / (2 * (L + 1)))
          + binomSfGe n p (hFire C.middleRate C.testLevel
              (max C.firstLook ⌈C.searchShare * n⌉₊))) := by
  classical
  set ex := Real.exp (-C.edgeExcess / (2 * (L + 1)))
  have hex : 0 ≤ ex := (Real.exp_pos _).le
  have hRHS : 0 ≤ (if C.firstLook ≤ n then binomSfGe n p (hFire C.startRate C.testLevel n)
      + C.testLevel else 0)
      + ((C.maxLeaves * Fintype.card α : ℕ) * ex
        + binomSfGe n p
          (hFire C.middleRate C.testLevel (max C.firstLook ⌈C.searchShare * n⌉₊))) := by
    have := binomSfGe_nonneg hp0 hp1 n (hFire C.startRate C.testLevel n)
    have := binomSfGe_nonneg hp0 hp1 n
      (hFire C.middleRate C.testLevel (max C.firstLook ⌈C.searchShare * n⌉₊))
    have : (0 : ℝ) ≤ (C.maxLeaves * Fintype.card α : ℕ) * ex := by positivity
    split_ifs <;> linarith
  by_cases hnN : n ≤ N
  swap
  · have : {xs : Fin N → FreeMonoid α | n ≤ N ∧
        FalseFire read C D ε T E L p q ((List.ofFn xs).take n)} = ∅ := by
      ext xs; simp [hnN]
    rw [this, measureReal_empty]
    exact hRHS
  have hlen : ∀ xs : Fin N → FreeMonoid α, ((List.ofFn xs).take n).length = n := by
    intro xs; simp [hnN]
  set F := T.leaves.toFinset ×ˢ (Finset.univ : Finset α)
  set A1 := {xs : Fin N → FreeMonoid α | C.firstLook ≤ n ∧ D.real {x | fSt read C T E x} ≤ p
    ∧ binomSfGe n C.startRate (cnt (fSt read C T E) n (List.ofFn xs)) < C.testLevel}
  set A2 := {xs : Fin N → FreeMonoid α | ∃ w ∈ (List.ofFn xs).take n, L < w.toList.length}
  set A3 := ⋃ e ∈ F.filter (fun e => ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ q),
    {xs : Fin N → FreeMonoid α | C.edgeExcess + C.edgeRate * C.edgeReadShare * n
      ≤ (((List.ofFn xs).take n).map fun x => (edgeCount read C T E e x : ℝ)).sum}
  set A4 := {xs : Fin N → FreeMonoid α | D.real {x | fP read C T E x} ≤ p ∧
    binomSfGe (max C.firstLook ⌈C.searchShare * n⌉₊) C.middleRate (cnt (fP read C T E) n
      (List.ofFn xs)) < C.testLevel}
  set A5 := {xs : Fin N → FreeMonoid α | ε < D.real {x | Disagrees read ⟨T, E⟩ C.walkStart x}
      ∧ C.firstLook ≤ n ∧
    1 - binomSfGe n C.disagreementRate (cnt (fDis read C T E) n (List.ofFn xs) + 1) < C.testLevel}
  have hsub : {xs : Fin N → FreeMonoid α | n ≤ N ∧
      FalseFire read C D ε T E L p q ((List.ofFn xs).take n)} ⊆ A1 ∪ A2 ∪ A3 ∪ A4 ∪ A5 := by
    rintro xs ⟨-, h⟩
    unfold FalseFire at h
    rw [hlen xs] at h
    rcases h with h | h | ⟨e, he, hm, hc⟩ | h | h
    · exact Or.inl (Or.inl (Or.inl (Or.inl h)))
    · exact Or.inl (Or.inl (Or.inl (Or.inr h)))
    · exact Or.inl (Or.inl (Or.inr (Set.mem_biUnion (Finset.mem_coe.2
        (Finset.mem_filter.2 ⟨he, hm⟩)) hc)))
    · exact Or.inl (Or.inr h)
    · exact Or.inr h
  have h1 : (Measure.pi fun _ : Fin N => D).real A1
      ≤ if C.firstLook ≤ n then binomSfGe n p (hFire C.startRate C.testLevel n) else 0 := by
    by_cases hc : C.firstLook ≤ n ∧ D.real {x | fSt read C T E x} ≤ p
    · rw [if_pos hc.1]
      have : A1 = {xs | binomSfGe n C.startRate (cnt (fSt read C T E) n (List.ofFn xs))
          < C.testLevel} := by
        ext xs; simp [A1, hc.1, hc.2]
      rw [this]
      refine pi_cnt_up D (fSt read C T E) hnN hc.2 hp1
        (fun j => binomSfGe n C.startRate j < C.testLevel) (fun j j' h hj => lt_of_le_of_lt
        (binomSfGe_anti hθs0 hθs1 (by omega)) hj) _ fun j hj => ?_
      have := Nat.sInf_le (s := {h | binomSfGe n C.startRate h < C.testLevel}) hj
      unfold hFire; omega
    · have : A1 = ∅ := by
        ext xs
        simp only [A1, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_and]
        exact fun h1 h2 => absurd ⟨h1, h2⟩ hc
      rw [this, measureReal_empty]
      split_ifs <;> simp [binomSfGe_nonneg hp0 hp1]
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
          (Measure.pi fun _ : Fin N => D).real {xs : Fin N → FreeMonoid α |
            C.edgeExcess + C.edgeRate * C.edgeReadShare * n
              ≤ (((List.ofFn xs).take n).map fun x => (edgeCount read C T E e x : ℝ)).sum} :=
          measureReal_biUnion_finset_le _ _
      _ ≤ ∑ _e ∈ F.filter (fun e => ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ q), ex :=
          Finset.sum_le_sum fun e he =>
            edge_fire_le read C D T E L hL e (Finset.mem_filter.1 he).2 hqt hnN
      _ = _ := by rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ _ := by gcongr
  have h4 : (Measure.pi fun _ : Fin N => D).real A4
      ≤ binomSfGe n p (hFire C.middleRate C.testLevel (max C.firstLook ⌈C.searchShare * n⌉₊)) := by
    by_cases hc : D.real {x | fP read C T E x} ≤ p
    · have : A4 = {xs | binomSfGe (max C.firstLook ⌈C.searchShare * n⌉₊) C.middleRate
          (cnt (fP read C T E) n (List.ofFn xs)) < C.testLevel} := by
        ext xs; simp [A4, hc]
      rw [this]
      refine pi_cnt_up D (fP read C T E) hnN hc hp1
        (fun j => binomSfGe (max C.firstLook ⌈C.searchShare * n⌉₊) C.middleRate j < C.testLevel)
        (fun j j' h hj => lt_of_le_of_lt
        (binomSfGe_anti hθpt0 hθpt1 (by omega)) hj) _ fun j hj => ?_
      have := Nat.sInf_le
        (s := {h | binomSfGe (max C.firstLook ⌈C.searchShare * n⌉₊) C.middleRate h < C.testLevel})
        hj
      unfold hFire; omega
    · have : A4 = ∅ := by
        ext xs
        simp only [A4, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_and]
        exact fun h => absurd h hc
      rw [this, measureReal_empty]
      exact binomSfGe_nonneg hp0 hp1 _ _
  have h5 : (Measure.pi fun _ : Fin N => D).real A5
      ≤ if C.firstLook ≤ n then C.testLevel else 0 := by
    by_cases hc : ε < D.real {x | Disagrees read ⟨T, E⟩ C.walkStart x} ∧ C.firstLook ≤ n
    · rw [if_pos hc.2]
      have : A5 = {xs | 1 - binomSfGe n C.disagreementRate
          (cnt (fDis read C T E) n (List.ofFn xs) + 1)
          < C.testLevel} := by
        ext xs; simp [A5, hc.1, hc.2]
      rw [this]
      refine (le_of_eq (pi_cnt D (fDis read C T E) N n hnN
        (fun j => 1 - binomSfGe n C.disagreementRate (j + 1) < C.testLevel))).trans ?_
      have hp : D.real {x | fDis read C T E x} = D.real
          {x | Disagrees read ⟨T, E⟩ C.walkStart x} := by
        congr 1; ext x; simp [fDis]
      rw [hp]
      calc _ ≤ binE n C.disagreementRate (fun j => if 1 - binomSfGe n C.disagreementRate (j + 1)
          < C.testLevel then 1 else 0) :=
            binE_le_of_le_anti hεd0 (by linarith [hc.1]) (by
              have := measureReal_le_one (μ := D) (s := {x | Disagrees read ⟨T, E⟩ C.walkStart x})
              exact this) n _ fun i j h => by
              have := binomSfGe_anti hεd0 hεd1 (n := n) (Nat.succ_le_succ h)
              by_cases hj : 1 - binomSfGe n C.disagreementRate (j + 1) < C.testLevel
              · rw [if_pos hj, if_pos (by linarith)]
              · rw [if_neg hj]; split_ifs <;> norm_num
        _ ≤ C.testLevel := binE_pvalue_le hεd0 hεd1 ha n
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

theorem surv_le {s : Level3_5State α} (hs : Inv read C s) (hT : s.tree = T) (hE : s.edges = E)
    (hm : 1 ≤ C.recordsToMove) (hn₀ : 1 ≤ C.firstLook) (hN₁ : C.firstLook ≤ N₁) (hN₁s : N₁ ≤ Ns)
    (hθpt0 : 0 ≤ C.middleRate) (hθpt1 : C.middleRate ≤ 1) (hεd0 : 0 ≤ C.disagreementRate)
    (hεd1 : C.disagreementRate ≤ 1) {θr εd' θpt' : ℝ} (hθr0 : 0 ≤ θr) (hθr1 : θr ≤ 1)
    (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1) (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1)
    (hsep : (C.maxLeaves : ℝ) ^ 2 * Fintype.card α * θr ≤ (1 - θpt') * εd') (N : ℕ) :
    (Measure.pi fun _ : Fin N => D).real
        {xs | Ns ≤ N ∧ Survives read C T E N₁ ((List.ofFn xs).take Ns)}
      ≤ (1 - binomSfGe Ns θr C.recordsToMove)
        + binE N₁ εd' (fun s =>
            if C.testLevel ≤ 1 - binomSfGe N₁ C.disagreementRate (s + 1)
                ∨ binomSfGe N₁ C.disagreementRate s < C.testLevel then 1 else 0)
        + (1 - binomSfGe Ns εd' (max C.firstLook ⌈C.searchShare * Ns⌉₊))
        + ∑ s ∈ Finset.Icc (max C.firstLook ⌈C.searchShare * Ns⌉₊) Ns,
            binE s θpt' (fun j => if C.testLevel ≤ binomSfGe s C.middleRate j then 1 else 0) := by
  classical
  set sm := max C.firstLook ⌈C.searchShare * Ns⌉₊ with hsm
  have hsm1 : 1 ≤ sm := le_trans hn₀ (le_max_left _ _)
  have h01 : ∀ (P : Prop) [Decidable P], (0 : ℝ) ≤ if P then 1 else 0 := fun P _ => by
    split_ifs <;> norm_num
  have hB0 : 0 ≤ 1 - binomSfGe Ns θr C.recordsToMove := by
    linarith [binomSfGe_le_one hθr0 hθr1 Ns C.recordsToMove]
  have hC0 : 0 ≤ binE N₁ εd' (fun s =>
      if C.testLevel ≤ 1 - binomSfGe N₁ C.disagreementRate (s + 1)
          ∨ binomSfGe N₁ C.disagreementRate s < C.testLevel then 1 else 0) :=
    binE_nonneg hεd'0 hεd'1 fun j => h01 _
  have hD10 : 0 ≤ 1 - binomSfGe Ns εd' sm := by linarith [binomSfGe_le_one hεd'0 hεd'1 Ns sm]
  have hD20 : 0 ≤ ∑ s ∈ Finset.Icc sm Ns,
      binE s θpt' (fun j => if C.testLevel ≤ binomSfGe s C.middleRate j then 1 else 0) :=
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
  set indC := fun s : ℕ =>
    if C.testLevel ≤ 1 - binomSfGe N₁ C.disagreementRate (s + 1)
        ∨ binomSfGe N₁ C.disagreementRate s < C.testLevel then (1 : ℝ) else 0
  by_cases hC : q ≤ εd'
  · have hsub : {xs : Fin N → FreeMonoid α | Ns ≤ N ∧
        Survives read C T E N₁ ((List.ofFn xs).take Ns)}
        ⊆ {xs | C.testLevel ≤ 1 - binomSfGe N₁ C.disagreementRate
            (cnt (fS read C T E) N₁ (List.ofFn xs) + 1)
            ∨ binomSfGe N₁ C.disagreementRate (cnt (fS read C T E) N₁ (List.ofFn xs))
                < C.testLevel} := by
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
          (fun j => C.testLevel ≤ 1 - binomSfGe N₁ C.disagreementRate (j + 1)
              ∨ binomSfGe N₁ C.disagreementRate j < C.testLevel)
      _ ≤ binE N₁ εd' indC := binE_le_of_le measureReal_nonneg hC hεd'1 N₁ _ fun i j h => by
          simp only [indC]
          have h1 := binomSfGe_anti hεd0 hεd1 (n := N₁) (Nat.succ_le_succ h)
          have h2 := binomSfGe_anti hεd0 hεd1 (n := N₁) h
          by_cases hi : C.testLevel ≤ 1 - binomSfGe N₁ C.disagreementRate (i + 1)
              ∨ binomSfGe N₁ C.disagreementRate i < C.testLevel
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
    have hOK := Ideal.probeOrAgree_ok read T E C.walkStart hEOK x
    generalize hg : probeOrAgree read T E C.walkStart x = o at hx hOK
    cases o with
    | wrongEdge p c u t =>
      obtain ⟨hp, -, huc, -⟩ := hOK
      refine Or.inr (Set.mem_biUnion (x := (p, c, t)) ?_ ?_)
      · simp [Rset, hp, DTree.sift_inl_mem read T _ t huc]
      · simp [fR, hg, recOf]
    | undecided k zs cl =>
      cases cl with
      | true => simp [searches] at hx
      | false =>
        obtain ⟨-, hzs, -⟩ := hOK
        refine Or.inl ?_
        simp only [Set.mem_ofPred_eq, fP, hg, middleString]
        cases zs with
        | nil => exact absurd rfl (hzs rfl)
        | cons z zs => simp
    | agree => simp [searches] at hx
    | unlearnedEdge p c u t => simp [searches] at hx
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
        ∪ {xs | sm ≤ cnt (fS read C T E) Ns (List.ofFn xs) ∧ C.testLevel ≤ binomSfGe
            (cnt (fS read C T E) Ns (List.ofFn xs)) C.middleRate
            (cnt (fP read C T E) Ns (List.ofFn xs))} := by
    rintro xs ⟨-, -, -, h3⟩
    rw [hlen] at h3
    by_cases hS : cnt (fS read C T E) Ns (List.ofFn xs) < sm
    · exact Or.inl hS
    · refine Or.inr ⟨not_lt.1 hS, ?_⟩
      push Not at hS
      have hq' : C.searchShare * Ns ≤ cnt (fS read C T E) Ns (List.ofFn xs) :=
        Nat.ceil_le.1 (le_trans (le_max_right _ _) hS)
      have hn₀' : C.firstLook ≤ cnt (fS read C T E) Ns (List.ofFn xs) :=
          le_trans (le_max_left _ _) hS
      by_contra hlt
      push Not at hlt
      apply h3
      refine ⟨hq', ?_⟩
      exact rateSide_true_of hn₀' hlt
  calc _ ≤ _ := measureReal_mono hsub
    _ ≤ _ := measureReal_union_le _ _
    _ ≤ (1 - binomSfGe Ns εd' sm) + ∑ s ∈ Finset.Icc sm Ns,
          binE s θpt' (fun j => if C.testLevel ≤ binomSfGe s C.middleRate j then 1 else 0) := by
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
          (fun s j => sm ≤ s ∧ C.testLevel ≤ binomSfGe s C.middleRate j)]
        calc _ ≤ binE Ns q (fun _ => ∑ s ∈ Finset.Icc sm Ns,
              binE s θpt' (fun j =>
                if C.testLevel ≤ binomSfGe s C.middleRate j then 1 else 0)) := by
              refine binE_mono_le measureReal_nonneg measureReal_le_one fun s hs => ?_
              by_cases hss : sm ≤ s
              · calc _ = binE s (pP / q) (fun j =>
                      if C.testLevel ≤ binomSfGe s C.middleRate j then 1 else 0) :=
                      binE_congr fun j _ => by simp [hss]
                  _ ≤ binE s θpt' (fun j =>
                      if C.testLevel ≤ binomSfGe s C.middleRate j then 1 else 0) :=
                      binE_le_of_le_anti hθpt'0 hπ0 hπ1 s _ fun i j h => by
                        have := binomSfGe_anti hθpt0 hθpt1 (n := s) h
                        by_cases hj : C.testLevel ≤ binomSfGe s C.middleRate j
                        · rw [if_pos hj, if_pos (by linarith)]
                        · rw [if_neg hj]; split_ifs <;> norm_num
                  _ ≤ _ := Finset.single_le_sum (f := fun s =>
                        binE s θpt' (fun j =>
                          if C.testLevel ≤ binomSfGe s C.middleRate j then 1 else 0))
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
theorem seg_le (X : Level3_5State α → FreeMonoid α → Prop) (XEv : List (FreeMonoid α) → Prop)
    (bX : ℝ)
    [IsProbabilityMeasure D] (L N₁ : ℕ) (G θr εd' θpt' : ℝ) (hL : ∀ᵐ x ∂D, x.toList.length ≤ L)
    {s : Level3_5State α} (hs : Inv read C s) (hf : s = newHypothesis s.tree s.edges s.redirected)
    (hT : s.tree ∈ (classSet (Fintype.card σ) : Finset (DTree α)))
    (hN : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
      D.real (PotGood M U θ C.walkStart read T) ≤ G)
    (hX : ∀ ys s' x, Inv read C s' → StInv read C s.tree s.edges ys s' → X s' x
      → XEv (ys ++ [x]))
    (hnt : ∀ ys s' x, Inv read C s' → StInv read C s.tree s.edges ys s'
      → s'.redirected = s.redirected → ¬ X s' x → applyProbe read C s' x ≠ .inr .failed)
    (hbX : ∀ N, (Measure.pi fun _ : Fin N => D).real
      {xs | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N ∧ XEv ((List.ofFn xs).take n)} ≤ bX)
    (hm : 1 ≤ C.recordsToMove) (hn₀ : 1 ≤ C.firstLook) (hN₁ : C.firstLook ≤ N₁) (hN₁s : N₁ ≤ Ns)
    (hG0 : 0 ≤ G) (hG1 : 2 * G ≤ 1) (hy1 : y ≤ 1)
    (hGt : 4 * (L + 1) * G ≤ C.edgeRate * C.edgeReadShare)
    (hyt : 2 * y ≤ C.edgeRate * C.edgeReadShare) (ha : 0 ≤ C.testLevel) (hθs0 : 0 ≤ C.startRate)
    (hθs1 : C.startRate ≤ 1) (hθe : 0 ≤ C.edgeRate) (hθpt0 : 0 ≤ C.middleRate)
    (hθpt1 : C.middleRate ≤ 1) (hεd0 : 0 ≤ C.disagreementRate) (hεd1 : C.disagreementRate ≤ 1)
    (hεd : C.disagreementRate ≤ ε) (hθr0 : 0 ≤ θr) (hθr1 : θr ≤ 1) (hεd'0 : 0 ≤ εd')
    (hεd'1 : εd' ≤ 1) (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1)
    (hsep : (C.maxLeaves : ℝ) ^ 2 * Fintype.card α * θr ≤ (1 - θpt') * εd') (N : ℕ) :
    (Measure.pi fun _ : Fin N => D)
        {xs | SegB (applyProbe read C) (BadStep read C M U θ D ε Ns y X) Same s (List.ofFn xs)}
      ≤ ENNReal.ofReal
          (stretchRisk C (Fintype.card α) L Ns N₁ (max (2 * G) y) θr εd' θpt' + bX) := by
  classical
  have hp0 : 0 ≤ max (2 * G) y := le_max_of_le_left (by positivity)
  have hp1 : max (2 * G) y ≤ 1 := max_le hG1 hy1
  have hqt : 2 * max (2 * (L + 1) * G) y ≤ C.edgeRate * C.edgeReadShare := by
    rcases max_cases (2 * (L + 1) * G) y with ⟨h, -⟩ | ⟨h, -⟩ <;> rw [h] <;> linarith
  have hsub : {xs : Fin N → FreeMonoid α |
      SegB (applyProbe read C) (BadStep read C M U θ D ε Ns y X) Same s (List.ofFn xs)}
      ⊆ ((⋃ n ∈ Finset.Icc 1 Ns, {xs | n ≤ N ∧
          FalseFire read C D ε s.tree s.edges L (max (2 * G) y) (max (2 * (L + 1) * G) y)
            ((List.ofFn xs).take n)})
        ∪ {xs | Ns ≤ N ∧ Survives read C s.tree s.edges N₁ ((List.ofFn xs).take Ns)})
        ∪ {xs | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N ∧ XEv ((List.ofFn xs).take n)} := by
    intro xs hxs
    have hst0 : StInv read C s.tree s.edges [] s := by
      have := stInv_fresh read C s.tree s.edges s.redirected
      rwa [← hf] at this
    have h₁0 : Look₁ read C s.tree s.edges N₁ [] := fun h => by simp at h; omega
    have hn0 : s.probeCount < Ns := by rw [hf]; simp only [newHypothesis]; omega
    obtain ⟨i, hi, hiN, h⟩ := seg_reduce read C M U θ D ε Ns y s.tree s.edges L N₁ hL (hN _ hT)
      X XEv s.redirected (fun ys s' x hs' hst' hx => hX ys s' x hs' hst' hx) hnt hN₁s hθs0 hθs1
      hθe hθpt0 hθpt1 hεd0 hεd1 (List.ofFn xs) [] s hs hst0 h₁0 rfl hn0 hxs
    simp only [List.nil_append, List.length_nil, zero_add, List.length_ofFn] at hi hiN h
    rcases h with h | ⟨hNs', h⟩ | h
    · exact Or.inl (Or.inl (Set.mem_biUnion (Finset.mem_coe.2 (Finset.mem_Icc.2 ⟨by omega, hiN⟩))
        ⟨by omega, h⟩))
    · refine Or.inl (Or.inr ⟨by omega, ?_⟩)
      rwa [← hNs']
    · exact Or.inr ⟨i + 1, Finset.mem_Icc.2 ⟨by omega, hiN⟩, by omega, h⟩
  rw [← ENNReal.ofReal_toReal (measure_ne_top _ _), ← measureReal_def]
  apply ENNReal.ofReal_le_ofReal
  have hsum : ∀ f : ℕ → ℝ, ∑ n ∈ Finset.Icc 1 Ns, (if C.firstLook ≤ n then f n else 0)
      = ∑ n ∈ Finset.Icc C.firstLook Ns, f n := by
    intro f
    rw [← Finset.sum_filter]
    congr 1
    ext n
    simp only [Finset.mem_filter, Finset.mem_Icc]
    omega
  calc _ ≤ _ := measureReal_mono hsub
    _ ≤ _ := measureReal_union_le _ _
    _ ≤ _ := add_le_add ((measureReal_union_le _ _).trans
        (add_le_add (measureReal_biUnion_finset_le _ _) le_rfl)) (hbX N)
    _ ≤ (∑ n ∈ Finset.Icc 1 Ns,
          ((if C.firstLook ≤ n then binomSfGe n (max (2 * G) y) (hFire C.startRate C.testLevel n)
              + C.testLevel else 0)
            + ((C.maxLeaves * Fintype.card α : ℕ) * Real.exp (-C.edgeExcess / (2 * (L + 1)))
              + binomSfGe n (max (2 * G) y) (hFire C.middleRate C.testLevel
                  (max C.firstLook ⌈C.searchShare * n⌉₊))))
        + ((1 - binomSfGe Ns θr C.recordsToMove)
          + binE N₁ εd' (fun s =>
              if C.testLevel ≤ 1 - binomSfGe N₁ C.disagreementRate (s + 1)
                  ∨ binomSfGe N₁ C.disagreementRate s < C.testLevel then 1 else 0)
          + (1 - binomSfGe Ns εd' (max C.firstLook ⌈C.searchShare * Ns⌉₊))
          + ∑ s ∈ Finset.Icc (max C.firstLook ⌈C.searchShare * Ns⌉₊) Ns,
              binE s θpt' (fun j => if C.testLevel ≤ binomSfGe s C.middleRate j then 1 else 0)))
                  + bX := by
      gcongr with n hn
      · exact fire_le read C D ε s.tree s.edges L hL (hs.size) hp0 hp1 hqt ha hθs0 hθs1 hθpt0
          hθpt1 hεd0 hεd1 hεd N n
      · exact surv_le read C D Ns s.tree s.edges N₁ hs rfl rfl hm hn₀ hN₁ hN₁s hθpt0 hθpt1 hεd0
          hεd1 hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep N
    _ = _ := by
      unfold stretchRisk
      rw [Finset.sum_add_distrib, hsum]
      ring

end Eps

end Random

end OrthoDFA
