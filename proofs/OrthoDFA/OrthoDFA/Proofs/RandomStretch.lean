import OrthoDFA.Proofs.RandomHarvest
import OrthoDFA.Proofs.RandomCounts

/-!
# One stretch

A step is bad where, within the first `Ns` probes of its stretch, it ends the round badly, or
where its stretch reaches `Ns` probes. Over a stretch the hypothesis is fixed, so each probe's
outcome is a function of the probe alone, and the stretch's counts are counts over fresh draws
(`StInv`). A bad step then shows in the counts (`seg_reduce`): a test that harvested a site whose
sampler finds at most `y` fired on a count whose mean per probe is at most `y`, a false success has
many disagreements, and a stretch that reaches `Ns` probes has no clean disagreement, no success at
look `N₁`, and no harvest of undecided middles at look `Ns`.
`seg_le` bounds a false firing at any look once per stretch, each sequential test by its level and
an edge, whose count per probe is at most `1`, by the count passing twice its mean by
`edgeExcess`; and the stretch reaching `Ns` probes by binomial expectations.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory

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

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)
  {σ : Type*} [Fintype σ] (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) (D : Measure (FreeMonoid α))
  (ε : ℝ) (Ns : ℕ) (y : ℝ)

def Same (s s' : Level3State α) : Prop := s'.tree = s.tree ∧ s'.edges = s.edges

/-- Within the first `Ns` probes of its stretch the step ends the round badly, a harvest's sampler
yielding `y`, or is one of the steps `X`, or its stretch reaches `Ns` probes. -/
def BadStep (X : Level3State α → FreeMonoid α → Prop) (s : Level3State α) (x : FreeMonoid α) :
    Prop :=
  (s.probeCount < Ns ∧ ((∃ e, applyProbe read C s x = .inr e ∧ ¬ EndsWell M (BadAt U θ) D ε
      {w | Disagrees read ⟨s.tree, s.edges⟩ C.walkStart w} C.walkStart
      (harvestLaw read D C.walkStart ⟨s.tree, s.edges⟩) y e)
      ∨ X s x))
  ∨ ∃ s', applyProbe read C s x = .inl s' ∧ Same s s' ∧ s'.probeCount = Ns

/-! ## A step within a stretch -/

section StepFacts

theorem finish_inr {t : Level3State α} {e : RoundEnd (Level3Site α)} (h : applyTests C t = .inr e) :
    runTests C t = some e := by
  unfold applyTests at h
  split at h
  · rename_i e' he'; simp only [Sum.inr.injEq] at h; rw [he', h]
  · simp at h

theorem step_inr {s : Level3State α} {x : FreeMonoid α} {e : RoundEnd (Level3Site α)}
    (h : applyProbe read C s x = .inr e) : runTests C (countProbe read C s x) = some e := by
  unfold applyProbe at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · exact finish_inr C h
  · exact finish_inr C h

/-- A step that keeps the hypothesis counts a probe that does not disagree at decided reads. -/
theorem step_same_counts {s s' : Level3State α} (hs : Inv read s) {x : FreeMonoid α}
    (h : applyProbe read C s x = .inl s') (hsm : Same s s') :
    s' = countProbe read C s x ∧ runTests C s' = none ∧ Inv read s'
      ∧ recOf (probeOrAgree read s.tree s.edges C.walkStart x) = none := by
  have hinv := (step_same read C hs h hsm.1 hsm.2).1
  rcases step_cases read C hs h with
    ⟨p, c, u, t, -, hE, -, -, -, rfl⟩ | ⟨p, c, u, t, t₀, w₀, -, hE, hne, hp, -, -, rfl⟩ |
    ⟨h1, h2, h3⟩
  · exfalso
    have := congrFun hsm.2 (p, c)
    simp only [newHypothesis, Function.update_self, hE] at this
    exact absurd this (by simp)
  · exfalso
    have := congrArg List.length (congrArg DTree.leaves hsm.1)
    simp only [splitLeaf, newHypothesis] at this
    rw [DTree.length_leaves_splitAt _ _ _ hp] at this
    omega
  · exact ⟨h1, h2, hinv, h3⟩

theorem look_none {t : Level3State α} (h : runTests C t = none) :
    ¬ seqRate C.disagreementRate C.disagreementAlternative (C.level t.hypotheses) t.probeCount
        t.searchCount
      ∧ ¬ (C.searchShare * t.probeCount ≤ t.searchCount
          ∧ seqRate C.middleRate C.middleAlternative (C.level t.hypotheses) t.searchCount
            t.middleUndecided.length) := by
  unfold runTests at h
  split_ifs at h with h1 h2 h3 h4
  exact ⟨h4, h3⟩

theorem look_some {t : Level3State α} {e : RoundEnd (Level3Site α)} (h : runTests C t = some e) :
    (seqRate C.startRate C.startAlternative (C.level t.hypotheses) t.probeCount
        t.startUndecided.length
        ∧ e = .harvest .start)
    ∨ (∃ p c, (p, c) ∈ t.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α)
        ∧ C.edgeReadShare * t.probeCount ≤ t.edgeReads p c
        ∧ C.edgeExcess (t.tree.leaves.length * Fintype.card α) t.hypotheses
          + C.edgeRate * t.edgeReads p c ≤ (t.edgeUndecided p c).length
        ∧ e = .harvest (.edge p c))
    ∨ (C.searchShare * t.probeCount ≤ t.searchCount
        ∧ seqRate C.middleRate C.middleAlternative (C.level t.hypotheses) t.searchCount
          t.middleUndecided.length
        ∧ e = .harvest .middles)
    ∨ (seqRate C.disagreementRate C.disagreementAlternative (C.level t.hypotheses) t.probeCount
          t.searchCount
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

variable (T : DTree α) (E : Edges α) (j : ℕ)

def fS (x : FreeMonoid α) : Bool := searches (probeOrAgree read T E C.walkStart x)

def fSt (x : FreeMonoid α) : Bool :=
  !(startString (probeOrAgree read T E C.walkStart x)).isEmpty

def fP (x : FreeMonoid α) : Bool :=
  !(middleString (probeOrAgree read T E C.walkStart x)).isEmpty

/-- The strings a probe charges to the edge `e`. -/
noncomputable def edgeCount (e : List Bool × α) (x : FreeMonoid α) : ℕ :=
  (undecidedAtEdge (positionReads read T E C.walkStart x) e.1 e.2).length

open scoped Classical in
noncomputable def fDis (x : FreeMonoid α) : Bool := decide (Disagrees read ⟨T, E⟩ C.walkStart x)

def fW (x : FreeMonoid α) : Bool := (recOf (probeOrAgree read T E C.walkStart x)).isSome

variable (N₁ : ℕ)

/-- After the probes `ys` of a stretch at `(T, E)`, the counts are counts over `ys`. -/
structure StInv (ys : List (FreeMonoid α)) (s : Level3State α) : Prop where
  tree : s.tree = T
  edges : s.edges = E
  hypotheses : s.hypotheses = j
  probeCount : s.probeCount = ys.length
  searchCount : s.searchCount = ys.countP (fS read C T E)
  startUndecided : s.startUndecided.length = ys.countP (fSt read C T E)
  middleUndecided : s.middleUndecided.length = ys.countP (fP read C T E)
  edgeUndecided : ∀ p c, (s.edgeUndecided p c).length = (ys.map (edgeCount read C T E (p, c))).sum

/-- The stretch's success test did not fire at look `N₁`. -/
def Look₁ (ys : List (FreeMonoid α)) : Prop :=
  N₁ ≤ ys.length → ¬ seqRate C.disagreementRate C.disagreementAlternative (C.level j) N₁
      ((ys.take N₁).countP (fS read C T E))

theorem stInv_fresh : StInv read C T E j [] (newHypothesis T E j) where
  tree := rfl
  edges := rfl
  hypotheses := rfl
  probeCount := rfl
  searchCount := rfl
  startUndecided := rfl
  middleUndecided := rfl
  edgeUndecided := fun p c => by simp [newHypothesis]

theorem countP_snoc (f : FreeMonoid α → Bool) (ys : List (FreeMonoid α)) (x : FreeMonoid α) :
    (ys ++ [x]).countP f = ys.countP f + if f x then 1 else 0 := by
  simp [List.countP_append]

theorem length_le_one_eq {X : Type*} (l : List X) (h : l.length ≤ 1) :
    l.length = if (!l.isEmpty) = true then 1 else 0 := by
  rcases l with _ | ⟨z, _ | ⟨w, l⟩⟩ <;> simp_all

/-- A probe counted into a stretch whose counts are its probes' keeps them so. -/
theorem stInv_countProbe {ys : List (FreeMonoid α)} {s : Level3State α}
    (hst : StInv read C T E j ys s) (x : FreeMonoid α) :
    StInv read C T E j (ys ++ [x]) (countProbe read C s x) := by
  classical
  obtain ⟨htree, hedges, hhyp, hn, hdis, hst', hpt, hund⟩ := hst
  subst htree hedges
  refine ⟨rfl, rfl, hhyp, by simp [countProbe, hn], ?_, ?_, ?_, ?_⟩
  · simp only [countProbe, hdis, countP_snoc]
    cases hb : searches (probeOrAgree read s.tree s.edges C.walkStart x) <;> simp [fS, hb]
  · simp only [countProbe, List.length_append, hst', countP_snoc]
    rw [length_le_one_eq _ (startString_length _)]
    cases hb : (startString (probeOrAgree read s.tree s.edges C.walkStart x)).isEmpty <;>
      simp [fSt, hb]
  · simp only [countProbe, List.length_append, hpt, countP_snoc]
    rw [length_le_one_eq _ (middleString_length _)]
    cases hb : (middleString (probeOrAgree read s.tree s.edges C.walkStart x)).isEmpty <;>
      simp [fP, hb]
  · intro p c
    simp only [countProbe, List.length_append, hund, List.map_append, List.sum_append,
      List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
    rfl

theorem look₁_snoc {ys : List (FreeMonoid α)} {s : Level3State α} {x : FreeMonoid α}
    (hst : StInv read C T E j (ys ++ [x]) s) (h₀ : Look₁ read C T E j N₁ ys)
    (h₁ : N₁ = ys.length + 1 →
      ¬ seqRate C.disagreementRate C.disagreementAlternative (C.level j) N₁ s.searchCount) :
    Look₁ read C T E j N₁ (ys ++ [x]) := by
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

variable (T : DTree α) (E : Edges α) (j : ℕ) (N₁ : ℕ)

/-- A test fired at look `n` at a site whose sampler finds at most `p`, or success fired
falsely. -/
def FalseFire (p : ℝ) (zs : List (FreeMonoid α)) : Prop :=
  (D.real {x | fSt read C T E x} ≤ p
      ∧ seqRate C.startRate C.startAlternative (C.level j) zs.length (zs.countP (fSt read C T E)))
  ∨ (∃ e ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α),
      ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ p
      ∧ C.edgeExcess (T.leaves.length * Fintype.card α) j + C.edgeRate * C.edgeReadShare * zs.length
        ≤ (zs.map fun x => (edgeCount read C T E e x : ℝ)).sum)
  ∨ (D.real {x | fP read C T E x} ≤ p ∧ C.searchShare * zs.length ≤ zs.countP (fS read C T E)
      ∧ seqRate C.middleRate C.middleAlternative (C.level j) (zs.countP (fS read C T E))
        (zs.countP (fP read C T E)))
  ∨ (ε < D.real {x | Disagrees read ⟨T, E⟩ C.walkStart x}
      ∧ seqRate C.disagreementRate C.disagreementAlternative (C.level j) zs.length
        (zs.countP (fS read C T E)))

/-- The stretch reached `zs.length` probes: no clean disagreement, no success at look `N₁`, and no
harvest of undecided middles at the last look. -/
def Survives (zs : List (FreeMonoid α)) : Prop :=
  zs.countP (fW read C T E) = 0
  ∧ ¬ seqRate C.disagreementRate C.disagreementAlternative (C.level j) N₁
      ((zs.take N₁).countP (fS read C T E))
  ∧ ¬ (C.searchShare * zs.length ≤ zs.countP (fS read C T E)
      ∧ seqRate C.middleRate C.middleAlternative (C.level j) (zs.countP (fS read C T E))
          (zs.countP (fP read C T E)))

omit [DecidableEq α] in
theorem integral_length_le_one {β : Type*} [IsProbabilityMeasure D]
    (f : FreeMonoid α → List β) (hf : ∀ x, (f x).length ≤ 1) :
    ∫ x, ((f x).length : ℝ) ∂D = D.real {x | (!(f x).isEmpty) = true} := by
  rw [← integral_indicator_one (Set.to_countable _).measurableSet]
  congr 1
  funext x
  rw [length_le_one_eq _ (hf x)]
  by_cases h : f x = []
  · simp [h, Set.indicator]
  · simp [h, Set.indicator]

/-- A step that ends the round badly fires a test falsely. -/
theorem falseFire_of_end [IsProbabilityMeasure D] {G : ℝ}
    (hG : D.real (PotGood M U θ C.walkStart read T) ≤ G) (hθe : 0 ≤ C.edgeRate)
    {ys : List (FreeMonoid α)} {s : Level3State α}
    (hst : StInv read C T E j ys s) {x : FreeMonoid α} {e : RoundEnd (Level3Site α)}
    (he : applyProbe read C s x = .inr e)
    (hbad : ¬ EndsWell M (BadAt U θ) D ε {w | Disagrees read ⟨s.tree, s.edges⟩ C.walkStart w}
      C.walkStart (harvestLaw read D C.walkStart ⟨s.tree, s.edges⟩) y e) :
    FalseFire read C D ε T E j (max (2 * G) y) (ys ++ [x]) := by
  classical
  have hlook := step_inr read C he
  have hst' := stInv_countProbe read C T E j hst x
  rw [hst.tree, hst.edges] at hbad
  have hlen : (countProbe read C s x).probeCount = (ys ++ [x]).length := hst'.probeCount
  have htot := fun site h => total_le_of_not_endsWell read D C.walkStart ⟨T, E⟩ M U θ
    (y := y) hG ε {w | Disagrees read ⟨T, E⟩ C.walkStart w} site h
  have hj : (countProbe read C s x).hypotheses = j := hst'.hypotheses
  rcases look_some C hlook with ⟨hr, rfl⟩ | ⟨p, c, hpc, h1, h2, rfl⟩ | ⟨hq, hr, rfl⟩ | ⟨hr, rfl⟩
  · rw [hj] at hr
    have ht := htot .start hbad
    rw [harvestLaw_real_univ read D C.walkStart ⟨T, E⟩, integral_length_le_one D
      (siteHarvest read C.walkStart ⟨T, E⟩ .start) fun x => startString_length _] at ht
    refine Or.inl ⟨ht, ?_⟩
    rwa [hlen, hst'.startUndecided] at hr
  · rw [hj, hst'.tree] at h2
    have ht := htot (.edge p c) hbad
    rw [harvestLaw_real_univ read D C.walkStart ⟨T, E⟩] at ht
    have hT : (countProbe read C s x).tree = T := hst.tree
    refine Or.inr (Or.inl ⟨(p, c), by rwa [hT] at hpc, ht, ?_⟩)
    rw [hlen] at h1
    have h3 : C.edgeRate * C.edgeReadShare * ((ys ++ [x]).length : ℝ)
        ≤ C.edgeRate * (countProbe read C s x).edgeReads p c := by
      rw [mul_assoc]; exact mul_le_mul_of_nonneg_left h1 hθe
    have h4 : (((countProbe read C s x).edgeUndecided p c).length : ℝ)
        = ((ys ++ [x]).map fun x => (edgeCount read C T E (p, c) x : ℝ)).sum := by
      rw [hst'.edgeUndecided p c, Nat.cast_list_sum, List.map_map]
      rfl
    linarith
  · rw [hj] at hr
    have ht := htot .middles hbad
    rw [harvestLaw_real_univ read D C.walkStart ⟨T, E⟩, integral_length_le_one D
      (siteHarvest read C.walkStart ⟨T, E⟩ .middles) fun x => middleString_length _] at ht
    rw [hlen, hst'.searchCount] at hq
    rw [hst'.searchCount, hst'.middleUndecided] at hr
    exact Or.inr (Or.inr (Or.inl ⟨ht, hq, hr⟩))
  · rw [hj] at hr
    have hD : ε < D.real {w | Disagrees read ⟨T, E⟩ C.walkStart w} := by
      simpa [EndsWell] using hbad
    rw [hlen, hst'.searchCount] at hr
    exact Or.inr (Or.inr (Or.inr ⟨hD, hr⟩))

/-- A bad step in a stretch shows in its counts. -/
theorem seg_reduce [IsProbabilityMeasure D] {G : ℝ}
    (hG : D.real (PotGood M U θ C.walkStart read T) ≤ G)
    (X : Level3State α → FreeMonoid α → Prop) (XEv : List (FreeMonoid α) → Prop)
    (hX : ∀ ys s x, Inv read s → StInv read C T E j ys s → X s x → XEv (ys ++ [x]))
    (hN₁s : N₁ ≤ Ns) (hθe : 0 ≤ C.edgeRate) :
    ∀ (xs ys : List (FreeMonoid α)) (s : Level3State α), Inv read s →
      StInv read C T E j ys s → Look₁ read C T E j N₁ ys → ys.countP (fW read C T E) = 0
          → s.probeCount < Ns →
      SegB (applyProbe read C) (BadStep read C M U θ D ε Ns y X) Same s xs →
      ∃ i, i < xs.length ∧ ys.length + i + 1 ≤ Ns ∧
        (FalseFire read C D ε T E j (max (2 * G) y) (ys ++ xs.take (i + 1))
          ∨ (ys.length + i + 1 = Ns ∧ Survives read C T E j N₁ (ys ++ xs.take (i + 1)))
          ∨ XEv (ys ++ xs.take (i + 1)))
  | [], _, _, _, _, _, _, _, h => by simp [SegB] at h
  | x :: xs, ys, s, hs, hst, h₁, hcl, hn, h => by
    have hn' : s.probeCount = ys.length := hst.probeCount
    have hsame : ∀ s', applyProbe read C s x = .inl s' → Same s s' →
        Inv read s' ∧ StInv read C T E j (ys ++ [x]) s' ∧ Look₁ read C T E j N₁ (ys ++ [x])
          ∧ runTests C s' = none ∧ s'.probeCount = ys.length + 1
          ∧ (ys ++ [x]).countP (fW read C T E) = 0 := by
      intro s' h' hsm
      obtain ⟨rfl, hl, hinv, hr⟩ := step_same_counts read C hs h' hsm
      have hst' := stInv_countProbe read C T E j hst x
      refine ⟨hinv, hst', look₁_snoc read C T E j N₁ hst' h₁ fun hN => ?_, hl, ?_, ?_⟩
      · have := (look_none C hl).1
        rwa [show (countProbe read C s x).probeCount = N₁ by
          rw [hst'.probeCount]; simp; omega, hst'.hypotheses] at this
      · rw [hst'.probeCount]; simp
      · rw [countP_snoc, hcl]
        simp [fW, ← hst.tree, ← hst.edges, hr]
    by_cases hb : BadStep read C M U θ D ε Ns y X s x
    · refine ⟨0, by simp, by omega, ?_⟩
      simp only [zero_add, List.take_succ_cons, List.take_zero]
      rcases hb with ⟨-, ⟨e, he, hbad⟩ | hx⟩ | ⟨s', h', hsm, hNs⟩
      · by_cases hx : X s x
        · exact Or.inr (Or.inr (hX ys s x hs hst hx))
        · exact Or.inl (falseFire_of_end read C M U θ D ε y T E j hG hθe hst he hbad)
      · exact Or.inr (Or.inr (hX ys s x hs hst hx))
      · obtain ⟨-, hst', h₁', hl, hn'', hcl'⟩ := hsame s' h' hsm
        refine Or.inr (Or.inl ⟨by omega, hcl', ?_, ?_⟩)
        · exact h₁' (by simp; omega)
        · have := (look_none C hl).2
          rwa [hst'.probeCount, hst'.searchCount, hst'.middleUndecided, hst'.hypotheses] at this
    · rcases h with h | ⟨s', h', hsm, hseg⟩
      · exact absurd h hb
      obtain ⟨hinv, hst', h₁', -, hn'', hcl'⟩ := hsame s' h' hsm
      have hlt : s'.probeCount < Ns := by
        have : s'.probeCount ≠ Ns := fun hc => hb (Or.inr ⟨s', h', hsm, hc⟩)
        omega
      obtain ⟨i, hi, hiN, hev⟩ := seg_reduce hG X XEv hX hN₁s hθe xs (ys ++ [x]) s' hinv hst'
        h₁' hcl' hlt hseg
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

variable [IsProbabilityMeasure D] (T : DTree α) (E : Edges α) (j : ℕ) (N₁ : ℕ)

theorem sum_map_sub_const (l : List (FreeMonoid α)) (f : FreeMonoid α → ℝ) (c : ℝ) :
    (l.map fun x => f x - c).sum = (l.map f).sum - l.length * c := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons, ih, List.length_cons]
    push_cast
    ring

/-- At an edge whose sampler finds at most `q`, with `2q` at most the edge test's least rate per
probe, the test's count passes `c` beyond twice its mean at some look with chance at most
`exp(−5c / 4)`. -/
theorem edge_any_le (e : List Bool × α) {q : ℝ}
    (hmean : ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ q)
    (hqt : 2 * q ≤ C.edgeRate * C.edgeReadShare) (c : ℝ) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D).real {xs | ∃ n ≤ N,
        c + C.edgeRate * C.edgeReadShare * n
          ≤ (((List.ofFn xs).take n).map fun x => (edgeCount read C T E e x : ℝ)).sum}
      ≤ Real.exp (-5 * c / 4) := by
  set m := ∫ x, (edgeCount read C T E e x : ℝ) ∂D
  have hm0 : 0 ≤ m := integral_nonneg fun _ => Nat.cast_nonneg _
  have hany := pi_sum_any D (fun x => (edgeCount read C T E e x : ℝ) - 2 * m) (K := 4 / 5)
    (by norm_num) (mgf_bit D (edgeCount read C T E e)
      (fun x => undecidedAtEdge_positionReads read T x E C.walkStart e.1 e.2) le_rfl) N c
  rw [show -c / (4 / 5) = -5 * c / 4 by ring] at hany
  have hsub : {xs : Fin N → FreeMonoid α | ∃ n ≤ N,
      c + C.edgeRate * C.edgeReadShare * n
        ≤ (((List.ofFn xs).take n).map fun x => (edgeCount read C T E e x : ℝ)).sum}
      ⊆ {xs | ∃ n ≤ N, c
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

theorem real_le_of_sub {N : ℕ} {S S' : Set (Fin N → FreeMonoid α)} {b : ℝ} (hb : 0 ≤ b)
    (h : S ⊆ S') (h' : (Measure.pi fun _ : Fin N => D) S' ≤ ENNReal.ofReal b) :
    (Measure.pi fun _ : Fin N => D).real S ≤ b :=
  (measureReal_mono h).trans (ENNReal.toReal_le_of_le_ofReal hb h')

theorem real_le_of_not {N : ℕ} {P : Prop} {S : Set (Fin N → FreeMonoid α)} {b : ℝ} (hb : 0 ≤ b)
    (hP : ¬ P) (h : ∀ xs ∈ S, P) : (Measure.pi fun _ : Fin N => D).real S ≤ b := by
  have : S = ∅ := Set.eq_empty_of_forall_notMem fun xs hxs => hP (h xs hxs)
  rw [this, measureReal_empty]
  exact hb

/-- Over a stretch's first `Ns` probes, a test fires falsely at some look with chance at most its
level for each sequential test, and for the edges together. -/
theorem fire_any_le {p : ℝ} (hp0 : 0 ≤ p)
    (hpt : 2 * p ≤ C.edgeRate * C.edgeReadShare) (ha : 0 < (C.level j))
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
          ∧ FalseFire read C D ε T E j p ((List.ofFn xs).take n)}
      ≤ 4 * C.level j := by
  classical
  set n := T.leaves.length * Fintype.card α
  set ex := Real.exp (-5 * C.edgeExcess n j / 4)
  set F := T.leaves.toFinset ×ˢ (Finset.univ : Finset α)
  set A1 := {xs : Fin N → FreeMonoid α | D.real {x | fSt read C T E x} ≤ p ∧ ∃ n ≤ N,
    seqRate C.startRate C.startAlternative (C.level j) n (cnt (fSt read C T E) n (List.ofFn xs))}
  set A2 := ⋃ e ∈ F.filter (fun e => ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ p),
    {xs : Fin N → FreeMonoid α | ∃ k ≤ N, C.edgeExcess n j + C.edgeRate * C.edgeReadShare * k
      ≤ (((List.ofFn xs).take k).map fun x => (edgeCount read C T E e x : ℝ)).sum}
  set A3 := {xs : Fin N → FreeMonoid α | D.real {x | fP read C T E x} ≤ p ∧ ∃ n ≤ N,
    C.searchShare * n ≤ cnt (fS read C T E) n (List.ofFn xs)
    ∧ seqRate C.middleRate C.middleAlternative (C.level j) (cnt (fS read C T E) n (List.ofFn xs))
      (cnt (fP read C T E) n (List.ofFn xs))}
  set A4 := {xs : Fin N → FreeMonoid α | ε < D.real {x | Disagrees read ⟨T, E⟩ C.walkStart x}
    ∧ ∃ n ≤ N, seqRate C.disagreementRate C.disagreementAlternative (C.level j) n
      (cnt (fS read C T E) n (List.ofFn xs))}
  have hsub : {xs : Fin N → FreeMonoid α | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N
      ∧ FalseFire read C D ε T E j p ((List.ofFn xs).take n)} ⊆ A1 ∪ A2 ∪ A3 ∪ A4 := by
    rintro xs ⟨k, -, hkN, h⟩
    have hlen : ((List.ofFn xs).take k).length = k := by simp [hkN]
    unfold FalseFire at h
    rw [hlen] at h
    rcases h with ⟨h1, h2⟩ | ⟨e, he, hm, hc⟩ | ⟨h1, h2, h3⟩ | ⟨h1, h2⟩
    · exact Or.inl (Or.inl (Or.inl ⟨h1, k, hkN, h2⟩))
    · exact Or.inl (Or.inl (Or.inr (Set.mem_biUnion (Finset.mem_coe.2
        (Finset.mem_filter.2 ⟨he, hm⟩)) ⟨k, hkN, hc⟩)))
    · exact Or.inl (Or.inr ⟨h1, k, hkN, h2, h3⟩)
    · exact Or.inr ⟨h1, k, hkN, h2⟩
  have h1 : (Measure.pi fun _ : Fin N => D).real A1 ≤ (C.level j) := by
    by_cases hc : D.real {x | fSt read C T E x} ≤ p
    · exact real_le_of_sub D ha.le (fun xs h => h.2) (seq_any_le D (fSt read C T E) hθs0
        (by linarith) (by linarith) hθs1 ha (by nlinarith) N)
    · exact real_le_of_not D ha.le hc fun xs h => h.1
  have h2 : (Measure.pi fun _ : Fin N => D).real A2 ≤ C.level j := by
    have hF : (F.filter fun e => ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ p).card ≤ n := by
      refine (Finset.card_filter_le _ _).trans ?_
      simp only [F, Finset.card_product, Finset.card_univ, n]
      exact Nat.mul_le_mul_right _ (List.toFinset_card_le _)
    calc _ ≤ ∑ e ∈ F.filter (fun e => ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ p),
          (Measure.pi fun _ : Fin N => D).real {xs : Fin N → FreeMonoid α | ∃ k ≤ N,
            C.edgeExcess n j + C.edgeRate * C.edgeReadShare * k
              ≤ (((List.ofFn xs).take k).map fun x => (edgeCount read C T E e x : ℝ)).sum} :=
          measureReal_biUnion_finset_le _ _
      _ ≤ ∑ _e ∈ F.filter (fun e => ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ p), ex :=
          Finset.sum_le_sum fun e he =>
            edge_any_le read C D T E e (Finset.mem_filter.1 he).2 hpt _ N
      _ = _ := by rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ C.level j := by
          rcases Nat.eq_zero_or_pos n with hn | hn
          · have : (F.filter fun e => ∫ x, (edgeCount read C T E e x : ℝ) ∂D ≤ p).card = 0 := by
              omega
            rw [this, Nat.cast_zero, zero_mul]
            exact ha.le
          · have hn' : (0 : ℝ) < n := by exact_mod_cast hn
            have hex : ex = C.level j / n := by
              simp only [ex, Level3Settings.edgeExcess]
              rw [show -5 * (4 / 5 * Real.log (n / C.level j)) / 4 = -Real.log (n / C.level j) by
                ring, Real.exp_neg, Real.exp_log (div_pos hn' ha), inv_div]
            rw [hex]
            calc _ ≤ (n : ℝ) * (C.level j / n) := by
                  gcongr
              _ = C.level j := by field_simp
  have h3 : (Measure.pi fun _ : Fin N => D).real A3 ≤ (C.level j) := by
    by_cases hc : D.real {x | fP read C T E x} ≤ p
    · exact real_le_of_sub D ha.le (fun xs h => h.2) (mid_any_le D (fS read C T E)
        (fP read C T E) (fun x hx => searches_of_ptStr (by simp only [fP] at hx; simpa using hx))
        hθpt0 hθpt hθpt1 ha hp0 hc hfall N)
    · exact real_le_of_not D ha.le hc fun xs h => h.1
  have h4 : (Measure.pi fun _ : Fin N => D).real A4 ≤ (C.level j) := by
    by_cases hc : ε < D.real {x | Disagrees read ⟨T, E⟩ C.walkStart x}
    · have hS : D.real {x | Disagrees read ⟨T, E⟩ C.walkStart x} ≤ D.real {x | fS read C T E x} :=
        measureReal_mono fun w hw =>
          show fS read C T E w = true from (disagrees_searches read T w E C.walkStart hw).2
      exact real_le_of_sub D ha.le (fun xs h => h.2) (seq_any_le D (fS read C T E)
        (by linarith) hεd1 hεd0 (by linarith) ha (by nlinarith) N)
    · exact real_le_of_not D ha.le hc fun xs h => h.1
  calc _ ≤ (Measure.pi fun _ : Fin N => D).real (A1 ∪ A2 ∪ A3 ∪ A4) := measureReal_mono hsub
    _ ≤ (Measure.pi fun _ : Fin N => D).real A1 + (Measure.pi fun _ : Fin N => D).real A2
        + (Measure.pi fun _ : Fin N => D).real A3 + (Measure.pi fun _ : Fin N => D).real A4 := by
      refine (measureReal_union_le _ _).trans (add_le_add_left ?_ _)
      refine (measureReal_union_le _ _).trans (add_le_add_left ?_ _)
      exact measureReal_union_le _ _
    _ ≤ _ := by linarith

open scoped Classical in
/-- The stretch outlives `Ns` probes with chance at most: `(1 − θr)^Ns` where clean disagreements
are at least `θr`; the success test's silence at look `N₁` where the searches are at most `εd'`;
fewer than `searchShare · Ns` searches where they are more; and the middles' test's silence where
the undecided middles are at least `θpt'` of the searches. -/
theorem surv_le {s : Level3State α} (hs : Inv read s) (hT : s.tree = T) (hE : s.edges = E)
    (hN₁s : N₁ ≤ Ns) (ha : 0 < C.level j) (hθpt0 : 0 < C.middleRate)
    (hθpt : C.middleRate < C.middleAlternative)
    (hθpt1 : C.middleAlternative < 1) (hεd0 : 0 < C.disagreementAlternative)
    (hεd : C.disagreementAlternative < C.disagreementRate) (hεd1 : C.disagreementRate < 1)
    {θr εd' θpt' : ℝ} (hθr0 : 0 ≤ θr) (hθr1 : θr ≤ 1)
    (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1) (hθpt'0 : C.middleAlternative ≤ θpt') (hθpt'1 : θpt' ≤ 1)
    (hsep : θr ≤ (1 - θpt') * εd') (N : ℕ) :
    (Measure.pi fun _ : Fin N => D).real
        {xs | Ns ≤ N ∧ Survives read C T E j N₁ ((List.ofFn xs).take Ns)}
      ≤ (1 - θr) ^ Ns
        + (εd' * (C.disagreementRate / C.disagreementAlternative)
            + (1 - εd') * ((1 - C.disagreementRate) / (1 - C.disagreementAlternative))) ^ N₁
          / C.level j
        + 2 ^ (⌈C.searchShare * Ns⌉₊ - 1) * (1 - εd' / 2) ^ Ns
        + (θpt' * (C.middleRate / C.middleAlternative)
            + (1 - θpt') * ((1 - C.middleRate) / (1 - C.middleAlternative)))
              ^ ⌈C.searchShare * Ns⌉₊ / C.level j := by
  classical
  set a := C.level j with ha_def
  set ρ := C.disagreementRate
  set d₁ := C.disagreementAlternative
  set μ := C.middleRate
  set m := C.middleAlternative
  set sm := ⌈C.searchShare * Ns⌉₊ with hsm
  set BD := εd' * (ρ / d₁) + (1 - εd') * ((1 - ρ) / (1 - d₁))
  set BM := θpt' * (μ / m) + (1 - θpt') * ((1 - μ) / (1 - m))
  set silD := fun h : ℕ => if seqRate ρ d₁ a N₁ h then (0 : ℝ) else 1
  set silM := fun n h : ℕ => if seqRate μ m a n h then (0 : ℝ) else 1
  have hθpt'0' : 0 ≤ θpt' := by linarith
  have hBD0 : 0 ≤ BD := by
    have : 0 ≤ ρ / d₁ := div_nonneg (by linarith) hεd0.le
    have : 0 ≤ (1 - ρ) / (1 - d₁) := div_nonneg (by linarith) (by linarith)
    have : 0 ≤ 1 - εd' := by linarith
    positivity
  have hu0 : 0 ≤ μ / m := div_nonneg hθpt0.le (by linarith)
  have hv0 : 0 ≤ (1 - μ) / (1 - m) := div_nonneg (by linarith) (by linarith)
  have hBM0 : 0 ≤ BM := by
    have : 0 ≤ 1 - θpt' := by linarith
    positivity
  have hBM1 : BM ≤ 1 := by
    have hm0 : 0 < m := by linarith
    have hm1 : 0 < 1 - m := by linarith
    have hsum : m * (μ / m) + (1 - m) * ((1 - μ) / (1 - m)) = 1 := by field_simp; ring
    have huv : μ / m ≤ (1 - μ) / (1 - m) := by
      rw [div_le_div_iff₀ hm0 hm1]; nlinarith
    have : BM = (1 - μ) / (1 - m) - θpt' * ((1 - μ) / (1 - m) - μ / m) := by simp only [BM]; ring
    rw [this]
    nlinarith
  have hA0 : 0 ≤ (1 - θr) ^ Ns := pow_nonneg (by linarith) _
  have hC0 : 0 ≤ BD ^ N₁ / a := by positivity
  have hD10 : 0 ≤ 2 ^ (sm - 1) * (1 - εd' / 2) ^ Ns := by
    have : 0 ≤ 1 - εd' / 2 := by linarith
    positivity
  have hD20 : 0 ≤ BM ^ sm / a := by positivity
  by_cases hNs : Ns ≤ N
  swap
  · have : {xs : Fin N → FreeMonoid α | Ns ≤ N ∧
        Survives read C T E j N₁ ((List.ofFn xs).take Ns)} = ∅ := by ext xs; simp [hNs]
    rw [this, measureReal_empty]
    linarith
  have hlen : ∀ xs : Fin N → FreeMonoid α, ((List.ofFn xs).take Ns).length = Ns := by
    intro xs; simp [hNs]
  have htake : ∀ xs : Fin N → FreeMonoid α,
      ((List.ofFn xs).take Ns).take N₁ = (List.ofFn xs).take N₁ := by
    intro xs; rw [List.take_take, min_eq_left hN₁s]
  by_cases hB : θr ≤ D.real {x | fW read C T E x}
  · have hsub : {xs : Fin N → FreeMonoid α | Ns ≤ N ∧
        Survives read C T E j N₁ ((List.ofFn xs).take Ns)}
        ⊆ {xs | cnt (fW read C T E) Ns (List.ofFn xs) < 1} := by
      rintro xs ⟨-, h1, -, -⟩
      simp only [Set.mem_ofPred_eq, cnt, h1, zero_lt_one]
    calc _ ≤ _ := measureReal_mono hsub
      _ = binE Ns (D.real {x | fW read C T E x}) (fun i => if i < 1 then 1 else 0) :=
          pi_cnt D _ N Ns hNs (fun i => i < 1)
      _ ≤ binE Ns θr (fun i => if i < 1 then 1 else 0) :=
          binE_le_of_le_anti hθr0 hB measureReal_le_one Ns _ fun i i' h => by
            by_cases hi' : i' < 1
            · rw [if_pos hi', if_pos (by omega)]
            · rw [if_neg hi']; split_ifs <;> norm_num
      _ = (1 - θr) ^ Ns := binE_lt_one Ns θr
      _ ≤ _ := by linarith
  push Not at hB
  set q := D.real {x | fS read C T E x}
  by_cases hC : q ≤ εd'
  · have hsub : {xs : Fin N → FreeMonoid α | Ns ≤ N ∧
        Survives read C T E j N₁ ((List.ofFn xs).take Ns)}
        ⊆ {xs | ¬ seqRate ρ d₁ a N₁ (cnt (fS read C T E) N₁ (List.ofFn xs))} := by
      rintro xs ⟨-, -, h2, -⟩
      rw [htake] at h2
      exact h2
    calc _ ≤ _ := measureReal_mono hsub
      _ = binE N₁ q (fun h => if ¬ seqRate ρ d₁ a N₁ h then 1 else 0) :=
          pi_cnt D _ N N₁ (hN₁s.trans hNs) (fun h => ¬ seqRate ρ d₁ a N₁ h)
      _ = binE N₁ q silD := binE_congr fun h _ => by
          by_cases hf : seqRate ρ d₁ a N₁ h <;> simp [silD, hf]
      _ ≤ binE N₁ εd' silD := binE_le_of_le_on measureReal_nonneg hC hεd'1 N₁ _
          fun i i' hii' hi' => by
            have := likelihoodRatio_anti (by linarith) hεd0.le hεd.le hεd1 hii' hi'
            by_cases hfi' : seqRate ρ d₁ a N₁ i'
            · simp only [silD]
              have hfi : seqRate ρ d₁ a N₁ i := le_trans hfi' this
              rw [if_pos hfi', if_pos hfi]
            · simp only [silD]
              rw [if_neg hfi']; split_ifs <;> norm_num
      _ ≤ BD ^ N₁ / a :=
          binE_silent_le (by linarith) hεd1 hεd0 (by linarith) ha hεd'0 hεd'1 N₁
      _ ≤ _ := by linarith
  push Not at hC
  set pP := D.real {x | fP read C T E x}
  have hEOK : EdgesOK read T E := hT ▸ hE ▸ hs.edges
  have hcov : {x | fS read C T E x = true}
      ⊆ {x | fP read C T E x = true} ∪ {x | fW read C T E x = true} := by
    intro x hx
    simp only [Set.mem_ofPred_eq, fS] at hx
    have hOK := probeOrAgree_ok read T E C.walkStart hEOK x
    generalize hg : probeOrAgree read T E C.walkStart x = o at hx hOK
    cases o with
    | wrongEdge p c u t => exact Or.inr (by simp [fW, hg, recOf])
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
  have hqle : q ≤ pP + θr :=
    (measureReal_mono hcov).trans ((measureReal_union_le _ _).trans (add_le_add le_rfl hB.le))
  have hq0 : 0 < q := lt_of_le_of_lt hεd'0 hC
  have hPq : θpt' * q ≤ pP := by
    have : θr ≤ (1 - θpt') * q :=
      hsep.trans (mul_le_mul_of_nonneg_left hC.le (by linarith))
    nlinarith
  have hPle : pP ≤ q := measureReal_mono fun x hx => by
    simp only [Set.mem_ofPred_eq, fP, fS] at hx ⊢
    exact searches_of_ptStr (by simpa using hx)
  have hπ0 : θpt' ≤ pP / q := by rw [le_div_iff₀ hq0]; linarith
  have hπ1 : pP / q ≤ 1 := by rw [div_le_one hq0]; exact hPle
  have hsub : {xs : Fin N → FreeMonoid α | Ns ≤ N ∧
      Survives read C T E j N₁ ((List.ofFn xs).take Ns)}
      ⊆ {xs | cnt (fS read C T E) Ns (List.ofFn xs) < sm}
        ∪ {xs | sm ≤ cnt (fS read C T E) Ns (List.ofFn xs)
          ∧ ¬ seqRate μ m a
            (cnt (fS read C T E) Ns (List.ofFn xs)) (cnt (fP read C T E) Ns (List.ofFn xs))} := by
    rintro xs ⟨-, -, -, h3⟩
    rw [hlen] at h3
    by_cases hS : cnt (fS read C T E) Ns (List.ofFn xs) < sm
    · exact Or.inl hS
    · refine Or.inr ⟨not_lt.1 hS, fun hf => h3 ⟨?_, hf⟩⟩
      exact Nat.ceil_le.1 (not_lt.1 hS)
  calc _ ≤ _ := measureReal_mono hsub
    _ ≤ _ := measureReal_union_le _ _
    _ ≤ 2 ^ (sm - 1) * (1 - εd' / 2) ^ Ns + BM ^ sm / a := by
      gcongr
      · calc _ = binE Ns q (fun i => if i < sm then 1 else 0) :=
            pi_cnt D _ N Ns hNs (fun i => i < sm)
          _ ≤ binE Ns εd' (fun i => if i < sm then 1 else 0) :=
              binE_le_of_le_anti hεd'0 hC.le measureReal_le_one Ns _ fun i i' h => by
                by_cases hi' : i' < sm
                · rw [if_pos hi', if_pos (by omega)]
                · rw [if_neg hi']; split_ifs <;> norm_num
          _ ≤ _ := binE_lt_le hεd'0 hεd'1 Ns sm
      · rw [pi_cnt2 D (fS read C T E) (fP read C T E) (fun x hx => searches_of_ptStr (by
            simp only [fP] at hx; simpa using hx)) N Ns hNs
          (fun n h => sm ≤ n ∧ ¬ seqRate μ m a n h)]
        calc _ ≤ binE Ns q (fun _ => BM ^ sm / a) := by
              refine binE_mono_le measureReal_nonneg measureReal_le_one fun n hn => ?_
              by_cases hss : sm ≤ n
              · calc _ = binE n (pP / q) (silM n) := binE_congr fun h _ => by
                      by_cases hf : seqRate μ m a n h <;> simp [silM, hss, hf]
                  _ ≤ binE n θpt' (silM n) :=
                      binE_le_of_le_anti_on hθpt'0' hπ0 hπ1 n _ fun i i' hii' hi' => by
                        have := likelihoodRatio_mono hθpt0 (by linarith) hθpt.le hθpt1.le hii' hi'
                        by_cases hfi : seqRate μ m a n i
                        · simp only [silM]
                          have hfi' : seqRate μ m a n i' := le_trans hfi this
                          rw [if_pos hfi, if_pos hfi']
                        · simp only [silM]
                          rw [if_neg hfi]; split_ifs <;> norm_num
                  _ ≤ BM ^ n / a :=
                      binE_silent_le hθpt0 (by linarith) (by linarith) hθpt1 ha hθpt'0' hθpt'1 n
                  _ ≤ BM ^ sm / a :=
                      div_le_div_of_nonneg_right (pow_le_pow_of_le_one hBM0 hBM1 hss) ha.le
              · calc _ = binE n (pP / q) (fun _ => 0) := binE_congr fun h _ => by simp [hss]
                  _ = 0 := binE_const _ _ _
                  _ ≤ _ := hD20
          _ = _ := binE_const _ _ _
    _ ≤ _ := by linarith

end Prob

/-- What a stretch risks: a bad step of the round's own kinds, at most `stretchRisk`, or one of
the steps `X`, which show in the counts as `XEv` with chance at most `bX`. -/
theorem seg_le (X : Level3State α → FreeMonoid α → Prop) (XEv : List (FreeMonoid α) → Prop)
    (bX : ℝ)
    [IsProbabilityMeasure D] (N₁ : ℕ) (G θr εd' θpt' : ℝ)
    {s : Level3State α} (hs : Inv read s) (hf : s = newHypothesis s.tree s.edges s.hypotheses)
    (hT : s.tree ∈ (classSet (Fintype.card σ) : Finset (DTree α)))
    (hN : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
      D.real (PotGood M U θ C.walkStart read T) ≤ G)
    (hX : ∀ ys s' x, Inv read s' → StInv read C s.tree s.edges s.hypotheses ys s' → X s' x
      → XEv (ys ++ [x]))
    (hbX : ∀ N, (Measure.pi fun _ : Fin N => D).real
      {xs | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N ∧ XEv ((List.ofFn xs).take n)} ≤ bX)
    (hN₁ : 1 ≤ N₁) (hN₁s : N₁ ≤ Ns) (hG0 : 0 ≤ G)
    (hGt : 4 * G ≤ C.edgeRate * C.edgeReadShare)
    (hyt : 2 * y ≤ C.edgeRate * C.edgeReadShare) (ha : 0 < C.level s.hypotheses)
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
    (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1) (hθpt'0 : C.middleAlternative ≤ θpt') (hθpt'1 : θpt' ≤ 1)
    (hsep : θr ≤ (1 - θpt') * εd') (N : ℕ) :
    (Measure.pi fun _ : Fin N => D)
        {xs | SegB (applyProbe read C) (BadStep read C M U θ D ε Ns y X) Same s (List.ofFn xs)}
      ≤ ENNReal.ofReal
          (stretchRisk C (C.level s.hypotheses) Ns N₁ θr εd' θpt' + bX) := by
  classical
  have hp0 : 0 ≤ max (2 * G) y := le_max_of_le_left (by positivity)
  have hpt : 2 * max (2 * G) y ≤ C.edgeRate * C.edgeReadShare := by
    rcases max_cases (2 * G) y with ⟨h, -⟩ | ⟨h, -⟩ <;> rw [h] <;> linarith
  have hsub : {xs : Fin N → FreeMonoid α |
      SegB (applyProbe read C) (BadStep read C M U θ D ε Ns y X) Same s (List.ofFn xs)}
      ⊆ ({xs | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N ∧
          FalseFire read C D ε s.tree s.edges s.hypotheses (max (2 * G) y) ((List.ofFn xs).take n)}
        ∪ {xs | Ns ≤ N ∧ Survives read C s.tree s.edges s.hypotheses N₁ ((List.ofFn xs).take Ns)})
        ∪ {xs | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N ∧ XEv ((List.ofFn xs).take n)} := by
    intro xs hxs
    have hst0 : StInv read C s.tree s.edges s.hypotheses [] s := by
      have := stInv_fresh read C s.tree s.edges s.hypotheses
      rwa [← hf] at this
    have h₁0 : Look₁ read C s.tree s.edges s.hypotheses N₁ [] := fun h => by simp at h; omega
    have hn0 : s.probeCount < Ns := by rw [hf]; simp only [newHypothesis]; omega
    obtain ⟨i, hi, hiN, h⟩ := seg_reduce read C M U θ D ε Ns y s.tree s.edges s.hypotheses N₁
      (hN _ hT) X XEv (fun ys s' x hs' hst' hx => hX ys s' x hs' hst' hx) hN₁s hθe (List.ofFn xs) []
      s hs hst0 h₁0 (by simp) hn0 hxs
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
    _ ≤ 4 * C.level s.hypotheses
        + ((1 - θr) ^ Ns
          + (εd' * (C.disagreementRate / C.disagreementAlternative)
              + (1 - εd') * ((1 - C.disagreementRate) / (1 - C.disagreementAlternative))) ^ N₁
            / C.level s.hypotheses
          + 2 ^ (⌈C.searchShare * Ns⌉₊ - 1) * (1 - εd' / 2) ^ Ns
          + (θpt' * (C.middleRate / C.middleAlternative)
              + (1 - θpt') * ((1 - C.middleRate) / (1 - C.middleAlternative)))
                ^ ⌈C.searchShare * Ns⌉₊ / C.level s.hypotheses)
        + bX := by
      gcongr
      · exact fire_any_le read C D ε Ns s.tree s.edges s.hypotheses hp0 hpt ha hθs0 hθs hθs1 hps
          hθpt0 hθpt hθpt1 hfall hεd0 hεd hεd1 hε N
      · exact surv_le read C D Ns s.tree s.edges s.hypotheses N₁ hs rfl rfl hN₁s ha hθpt0 hθpt
          hθpt1 hεd0 hεd hεd1 hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep N
    _ = _ := by
      unfold stretchRisk
      ring

end Random

end OrthoDFA
