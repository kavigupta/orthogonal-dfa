import OrthoDFA.Proofs.RandomHarvest
import OrthoDFA.Proofs.RandomCounts

/-!
# One stretch

A step is bad where, within the first `Ns` probes of its stretch, it ends the round badly, or
where its stretch reaches `Ns` probes. Over a stretch the hypothesis is fixed, so each probe's
outcome is a function of the probe alone, and the stretch's counts are counts over fresh draws
(`StInv`). A bad step then shows in the counts (`seg_reduce`): a test that harvested a site whose
sampler would not end well fired on a count whose mean per probe is small, or, at the middles,
at most `middleRate` of the searches', a false success has
many searches and probes stopping undecided at their ends, and a stretch that reaches `Ns` probes
has no change of hypothesis, no success at look `N₁`, and no harvest at look `Ns`.
`seg_le` bounds a false firing at any look once per stretch, each sequential test by its level,
and the stretch reaching `Ns` probes by binomial expectations.
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
      (searchSet read C.walkStart ⟨s.tree, s.edges⟩ ∪ endsSet read C.walkStart ⟨s.tree, s.edges⟩)
      C.walkStart (harvestLaw read D C.walkStart ⟨s.tree, s.edges⟩) y e)
      ∨ X s x))
  ∨ ∃ s', applyProbe read C s x = .inl s' ∧ Same s s' ∧ s'.probeCount = Ns

/-! ## A step within a stretch -/

section StepFacts

theorem finish_inr {t : Level3State α} {e : RoundEnd Level3Site} (h : applyTests C t = .inr e) :
    runTests C t = some e := by
  unfold applyTests at h
  split at h
  · rename_i e' he'; simp only [Sum.inr.injEq] at h; rw [he', h]
  · simp at h

theorem step_inr {s : Level3State α} {x : FreeMonoid α} {e : RoundEnd Level3Site}
    (h : applyProbe read C s x = .inr e) : runTests C (countProbe read C s x) = some e := by
  unfold applyProbe at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · exact finish_inr C h
  · exact finish_inr C h

/-- A step that keeps the hypothesis counts a probe that does not change it. -/
theorem step_same_counts {s s' : Level3State α} (hs : Inv read s) {x : FreeMonoid α}
    (h : applyProbe read C s x = .inl s') (hsm : Same s s') :
    s' = countProbe read C s x ∧ runTests C s' = none ∧ Inv read s'
      ∧ changes (probeOrAgree read s.tree s.edges C.walkStart x) = false := by
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
    ¬ seqRate C.endsRate C.endsAlternative (C.level t.hypotheses) t.probeCount
        t.endsUndecided.length
      ∧ ¬ (C.searchShare * t.probeCount ≤ t.searchCount
          ∧ seqRate C.middleRate C.middleAlternative (C.level t.hypotheses) t.searchCount
            t.middleUndecided.length)
      ∧ ¬ seqRate C.disagreementRate C.disagreementAlternative (C.level t.hypotheses) t.probeCount
        (t.searchCount + t.endsUndecided.length) := by
  unfold runTests at h
  split_ifs at h with h1 h2 h3
  exact ⟨h1, h2, h3⟩

theorem look_some {t : Level3State α} {e : RoundEnd Level3Site} (h : runTests C t = some e) :
    (seqRate C.endsRate C.endsAlternative (C.level t.hypotheses) t.probeCount
        t.endsUndecided.length
        ∧ e = .harvest .ends)
    ∨ (C.searchShare * t.probeCount ≤ t.searchCount
        ∧ seqRate C.middleRate C.middleAlternative (C.level t.hypotheses) t.searchCount
          t.middleUndecided.length
        ∧ e = .harvest .middles)
    ∨ (seqRate C.disagreementRate C.disagreementAlternative (C.level t.hypotheses) t.probeCount
          (t.searchCount + t.endsUndecided.length)
        ∧ e = .consistent) := by
  unfold runTests at h
  split_ifs at h with h1 h2 h3
  · exact Or.inl ⟨h1, (Option.some.inj h).symm⟩
  · exact Or.inr (Or.inl ⟨h2.1, h2.2, (Option.some.inj h).symm⟩)
  · exact Or.inr (Or.inr ⟨h3, (Option.some.inj h).symm⟩)

end StepFacts

/-! ## The stretch's counts -/

section Counts

variable (T : DTree α) (E : Edges α) (j : ℕ)

def fS (x : FreeMonoid α) : Bool := searches (probeOrAgree read T E C.walkStart x)

def fE (x : FreeMonoid α) : Bool :=
  !(endsString (probeOrAgree read T E C.walkStart x)).isEmpty

def fP (x : FreeMonoid α) : Bool :=
  !(middleString (probeOrAgree read T E C.walkStart x)).isEmpty

def fW (x : FreeMonoid α) : Bool := changes (probeOrAgree read T E C.walkStart x)

/-- What success counts: the searches and the probes stopping undecided at their ends. -/
def fSE (x : FreeMonoid α) : Bool := fS read C T E x || fE read C T E x

theorem fS_fE (x : FreeMonoid α) (h : fS read C T E x = true) : fE read C T E x = false := by
  simp only [fS, fE] at h ⊢
  cases hg : probeOrAgree read T E C.walkStart x with
  | undecided zs cl => cases cl <;> simp_all [searches, endsString]
  | _ => simp_all [searches, endsString]

theorem countP_fSE (l : List (FreeMonoid α)) :
    l.countP (fSE read C T E) = l.countP (fS read C T E) + l.countP (fE read C T E) := by
  induction l with
  | nil => simp
  | cons x l ih =>
    have h := fS_fE read C T E x
    simp only [List.countP_cons, ih, fSE]
    rcases Bool.eq_false_or_eq_true (fS read C T E x) with hs | hs
    · simp [hs, h hs]
      omega
    · rcases Bool.eq_false_or_eq_true (fE read C T E x) with he | he <;> simp [hs, he] <;> omega

theorem disagreeSet_eq :
    searchSet read C.walkStart ⟨T, E⟩ ∪ endsSet read C.walkStart ⟨T, E⟩
      = {x | fSE read C T E x || fW read C T E x} := by
  ext x
  simp only [searchSet, endsSet, fSE, fS, fE, fW, Set.mem_union, Set.mem_setOf_eq]
  cases probeOrAgree read T E C.walkStart x with
  | undecided zs cl => cases cl <;> cases zs <;> simp [searches, changes, endsString]
  | _ => simp [searches, changes, endsString]

variable (N₁ : ℕ)

/-- After the probes `ys` of a stretch at `(T, E)`, the counts are counts over `ys`. -/
structure StInv (ys : List (FreeMonoid α)) (s : Level3State α) : Prop where
  tree : s.tree = T
  edges : s.edges = E
  hypotheses : s.hypotheses = j
  probeCount : s.probeCount = ys.length
  searchCount : s.searchCount = ys.countP (fS read C T E)
  endsUndecided : s.endsUndecided.length = ys.countP (fE read C T E)
  middleUndecided : s.middleUndecided.length = ys.countP (fP read C T E)

/-- The stretch's success test did not fire at look `N₁`. -/
def Look₁ (ys : List (FreeMonoid α)) : Prop :=
  N₁ ≤ ys.length → ¬ seqRate C.disagreementRate C.disagreementAlternative (C.level j) N₁
      ((ys.take N₁).countP (fSE read C T E))

theorem stInv_fresh : StInv read C T E j [] (newHypothesis T E j) where
  tree := rfl
  edges := rfl
  hypotheses := rfl
  probeCount := rfl
  searchCount := rfl
  endsUndecided := rfl
  middleUndecided := rfl

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
  obtain ⟨htree, hedges, hhyp, hn, hdis, hst', hpt⟩ := hst
  subst htree hedges
  refine ⟨rfl, rfl, hhyp, by simp [countProbe, hn], ?_, ?_, ?_⟩
  · simp only [countProbe, hdis, countP_snoc]
    cases hb : searches (probeOrAgree read s.tree s.edges C.walkStart x) <;> simp [fS, hb]
  · simp only [countProbe, List.length_append, hst', countP_snoc]
    rw [length_le_one_eq _ (endsString_length _)]
    cases hb : (endsString (probeOrAgree read s.tree s.edges C.walkStart x)).isEmpty <;>
      simp [fE, hb]
  · simp only [countProbe, List.length_append, hpt, countP_snoc]
    rw [length_le_one_eq _ (middleString_length _)]
    cases hb : (middleString (probeOrAgree read s.tree s.edges C.walkStart x)).isEmpty <;>
      simp [fP, hb]

theorem look₁_snoc {ys : List (FreeMonoid α)} {s : Level3State α} {x : FreeMonoid α}
    (hst : StInv read C T E j (ys ++ [x]) s) (h₀ : Look₁ read C T E j N₁ ys)
    (h₁ : N₁ = ys.length + 1 →
      ¬ seqRate C.disagreementRate C.disagreementAlternative (C.level j) N₁
        (s.searchCount + s.endsUndecided.length)) :
    Look₁ read C T E j N₁ (ys ++ [x]) := by
  intro hN
  by_cases hlt : N₁ ≤ ys.length
  · rw [List.take_append_of_le_length hlt]
    exact h₀ hlt
  · have hN' : N₁ = ys.length + 1 := by simp at hN; omega
    rw [List.take_of_length_le (by simp; omega), countP_fSE, ← hst.searchCount,
      ← hst.endsUndecided]
    exact h₁ hN'

end Counts

/-! ## A bad step shows in the counts -/

section Reduce

variable (T : DTree α) (E : Edges α) (j : ℕ) (N₁ : ℕ)

/-- A test fired at look `n` at a site whose sampler finds at most `pE` at the ends, or at most
`pM`, or at most `middleRate` of the searches, at the middles, or success fired falsely. -/
def FalseFire (pE pM : ℝ) (zs : List (FreeMonoid α)) : Prop :=
  (D.real {x | fE read C T E x} ≤ pE
      ∧ seqRate C.endsRate C.endsAlternative (C.level j) zs.length (zs.countP (fE read C T E)))
  ∨ ((D.real {x | fP read C T E x} ≤ pM
        ∨ D.real {x | fP read C T E x} ≤ C.middleRate * D.real {x | fS read C T E x})
      ∧ C.searchShare * zs.length ≤ zs.countP (fS read C T E)
      ∧ seqRate C.middleRate C.middleAlternative (C.level j) (zs.countP (fS read C T E))
        (zs.countP (fP read C T E)))
  ∨ (ε < D.real {x | fSE read C T E x || fW read C T E x}
      ∧ seqRate C.disagreementRate C.disagreementAlternative (C.level j) zs.length
        (zs.countP fun x => fSE read C T E x || fW read C T E x))

/-- The stretch reached `zs.length` probes: no change of hypothesis, no success at look `N₁`, and
no harvest at the last look. -/
def Survives (zs : List (FreeMonoid α)) : Prop :=
  zs.countP (fW read C T E) = 0
  ∧ ¬ seqRate C.disagreementRate C.disagreementAlternative (C.level j) N₁
      ((zs.take N₁).countP (fSE read C T E))
  ∧ ¬ seqRate C.endsRate C.endsAlternative (C.level j) zs.length (zs.countP (fE read C T E))
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

omit [Fintype α] [DecidableEq α] in
theorem countP_or_of_zero {X : Type*} (f g : X → Bool) {l : List X} (h : l.countP g = 0) :
    l.countP (fun x => f x || g x) = l.countP f := by
  rw [List.countP_eq_zero] at h
  exact List.countP_congr fun x hx => by simp [h x hx]

/-- A step that ends the round badly fires a test falsely. -/
theorem falseFire_of_end [IsProbabilityMeasure D] {Ge θgpt ηm pM : ℝ}
    (hGe : D.real (goodAt read C.walkStart ⟨T, E⟩ M U θ .ends) ≤ Ge)
    (hGm : D.real (goodAt read C.walkStart ⟨T, E⟩ M U θ .middles)
      ≤ θgpt * D.real {x | fS read C T E x} + ηm)
    (hyM : y ≤ pM) (hηm : 4 * ηm ≤ pM) (hθg : 4 * θgpt ≤ C.middleRate)
    {ys : List (FreeMonoid α)} {s : Level3State α} (hs : Inv read s)
    (hst : StInv read C T E j ys s) (hcl : ys.countP (fW read C T E) = 0) {x : FreeMonoid α}
    {e : RoundEnd Level3Site} (he : applyProbe read C s x = .inr e)
    (hbad : ¬ EndsWell M (BadAt U θ) D ε
      (searchSet read C.walkStart ⟨s.tree, s.edges⟩ ∪ endsSet read C.walkStart ⟨s.tree, s.edges⟩)
      C.walkStart (harvestLaw read D C.walkStart ⟨s.tree, s.edges⟩) y e) :
    FalseFire read C D ε T E j (max (2 * Ge) y) pM (ys ++ [x]) := by
  classical
  have hlook := step_inr read C he
  have hst' := stInv_countProbe read C T E j hst x
  have hx : fW read C T E x = false := by
    rw [fW, ← hst.tree, ← hst.edges]
    exact changes_of_inr read C hs he
  rw [hst.tree, hst.edges] at hbad
  have hlen : (countProbe read C s x).probeCount = (ys ++ [x]).length := hst'.probeCount
  have hj : (countProbe read C s x).hypotheses = j := hst'.hypotheses
  rcases look_some C hlook with ⟨hr, rfl⟩ | ⟨hq, hr, rfl⟩ | ⟨hr, rfl⟩
  · rw [hj] at hr
    have ht := total_le_of_not_endsWell read D C.walkStart ⟨T, E⟩ M U θ (y := y) .ends hGe ε _ hbad
    rw [harvestLaw_real_univ read D C.walkStart ⟨T, E⟩, integral_length_le_one D
      (siteHarvest read C.walkStart ⟨T, E⟩ .ends) fun x => endsString_length _] at ht
    refine Or.inl ⟨ht, ?_⟩
    rwa [hlen, hst'.endsUndecided] at hr
  · rw [hj] at hr
    have ht := total_le_of_not_endsWell read D C.walkStart ⟨T, E⟩ M U θ (y := y) .middles hGm ε _
      hbad
    rw [harvestLaw_real_univ read D C.walkStart ⟨T, E⟩, integral_length_le_one D
      (siteHarvest read C.walkStart ⟨T, E⟩ .middles) fun x => middleString_length _] at ht
    rw [hlen, hst'.searchCount] at hq
    rw [hst'.searchCount, hst'.middleUndecided] at hr
    refine Or.inr (Or.inl ⟨?_, hq, hr⟩)
    change D.real {x | fP read C T E x} ≤ _ at ht
    by_cases hpM : D.real {x | fP read C T E x} ≤ pM
    · exact Or.inl hpM
    · push Not at hpM
      have hS : 0 ≤ D.real {x | fS read C T E x} := measureReal_nonneg
      have h2 : D.real {x | fP read C T E x} ≤ 2 * (θgpt * D.real {x | fS read C T E x} + ηm) := by
        rcases le_max_iff.1 ht with h | h
        · exact h
        · linarith
      right
      nlinarith
  · rw [hj] at hr
    rw [hlen, hst'.searchCount, hst'.endsUndecided, ← countP_fSE] at hr
    have hD : ε < D.real {w | fSE read C T E w || fW read C T E w} := by
      rw [← disagreeSet_eq]
      simpa [EndsWell] using hbad
    have hc : (ys ++ [x]).countP (fW read C T E) = 0 := by
      rw [countP_snoc, hcl, hx]; rfl
    rw [← countP_or_of_zero _ _ hc] at hr
    exact Or.inr (Or.inr ⟨hD, hr⟩)

/-- A bad step in a stretch shows in its counts. -/
theorem seg_reduce [IsProbabilityMeasure D] {Ge θgpt ηm pM : ℝ}
    (hGe : D.real (goodAt read C.walkStart ⟨T, E⟩ M U θ .ends) ≤ Ge)
    (hGm : D.real (goodAt read C.walkStart ⟨T, E⟩ M U θ .middles)
      ≤ θgpt * D.real {x | fS read C T E x} + ηm)
    (hyM : y ≤ pM) (hηm : 4 * ηm ≤ pM) (hθg : 4 * θgpt ≤ C.middleRate)
    (X : Level3State α → FreeMonoid α → Prop) (XEv : List (FreeMonoid α) → Prop)
    (hX : ∀ ys s x, Inv read s → StInv read C T E j ys s → X s x → XEv (ys ++ [x]))
    (hN₁s : N₁ ≤ Ns) :
    ∀ (xs ys : List (FreeMonoid α)) (s : Level3State α), Inv read s →
      StInv read C T E j ys s → Look₁ read C T E j N₁ ys → ys.countP (fW read C T E) = 0
          → s.probeCount < Ns →
      SegB (applyProbe read C) (BadStep read C M U θ D ε Ns y X) Same s xs →
      ∃ i, i < xs.length ∧ ys.length + i + 1 ≤ Ns ∧
        (FalseFire read C D ε T E j (max (2 * Ge) y) pM (ys ++ xs.take (i + 1))
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
      · have := (look_none C hl).2.2
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
        · exact Or.inl (falseFire_of_end read C M U θ D ε y T E j hGe hGm hyM hηm hθg hs hst hcl
            he hbad)
      · exact Or.inr (Or.inr (hX ys s x hs hst hx))
      · obtain ⟨-, hst', h₁', hl, hn'', hcl'⟩ := hsame s' h' hsm
        refine Or.inr (Or.inl ⟨by omega, hcl', ?_, ?_, ?_⟩)
        · exact h₁' (by simp; omega)
        · have := (look_none C hl).1
          rwa [hst'.probeCount, hst'.endsUndecided, hst'.hypotheses] at this
        · have := (look_none C hl).2.1
          rwa [hst'.probeCount, hst'.searchCount, hst'.middleUndecided, hst'.hypotheses] at this
    · rcases h with h | ⟨s', h', hsm, hseg⟩
      · exact absurd h hb
      obtain ⟨hinv, hst', h₁', -, hn'', hcl'⟩ := hsame s' h' hsm
      have hlt : s'.probeCount < Ns := by
        have : s'.probeCount ≠ Ns := fun hc => hb (Or.inr ⟨s', h', hsm, hc⟩)
        omega
      obtain ⟨i, hi, hiN, hev⟩ := seg_reduce hGe hGm hyM hηm hθg X XEv hX hN₁s xs (ys ++ [x]) s'
        hinv hst' h₁' hcl' hlt hseg
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
level for each sequential test, the middles' test once against `pM` and once against the
searches. -/
theorem fire_any_le {pE pM : ℝ} (hp0 : 0 ≤ pM) (ha : 0 < (C.level j))
    (hθs0 : 0 < C.endsRate) (hθs : C.endsRate < C.endsAlternative)
    (hθs1 : C.endsAlternative < 1) (hps : pE ≤ C.endsRate) (hθpt0 : 0 < C.middleRate)
    (hθpt : C.middleRate < C.middleAlternative) (hθpt1 : C.middleAlternative < 1)
    (hfall : pM * (C.middleAlternative / C.middleRate)
        * ((1 - C.middleAlternative) / (1 - C.middleRate)) ^ (C.searchShare - 1)
      + (1 - pM) * ((1 - C.middleAlternative) / (1 - C.middleRate)) ^ C.searchShare ≤ 1)
    (hεd0 : 0 < C.disagreementAlternative) (hεd : C.disagreementAlternative < C.disagreementRate)
    (hεd1 : C.disagreementRate < 1) (hε : C.disagreementRate ≤ ε) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D).real
        {xs | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N
          ∧ FalseFire read C D ε T E j pE pM ((List.ofFn xs).take n)}
      ≤ 4 * C.level j := by
  classical
  set fSW : FreeMonoid α → Bool := fun x => fSE read C T E x || fW read C T E x
  set A1 := {xs : Fin N → FreeMonoid α | D.real {x | fE read C T E x} ≤ pE ∧ ∃ n ≤ N,
    seqRate C.endsRate C.endsAlternative (C.level j) n (cnt (fE read C T E) n (List.ofFn xs))}
  set A3 := {xs : Fin N → FreeMonoid α | D.real {x | fP read C T E x} ≤ pM ∧ ∃ n ≤ N,
    C.searchShare * n ≤ cnt (fS read C T E) n (List.ofFn xs)
    ∧ seqRate C.middleRate C.middleAlternative (C.level j) (cnt (fS read C T E) n (List.ofFn xs))
      (cnt (fP read C T E) n (List.ofFn xs))}
  set A3' := {xs : Fin N → FreeMonoid α |
    D.real {x | fP read C T E x} ≤ C.middleRate * D.real {x | fS read C T E x} ∧ ∃ n ≤ N,
    seqRate C.middleRate C.middleAlternative (C.level j) (cnt (fS read C T E) n (List.ofFn xs))
      (cnt (fP read C T E) n (List.ofFn xs))}
  set A4 := {xs : Fin N → FreeMonoid α | ε < D.real {x | fSW x}
    ∧ ∃ n ≤ N, seqRate C.disagreementRate C.disagreementAlternative (C.level j) n
      (cnt fSW n (List.ofFn xs))}
  have hsub : {xs : Fin N → FreeMonoid α | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N
      ∧ FalseFire read C D ε T E j pE pM ((List.ofFn xs).take n)} ⊆ A1 ∪ A3 ∪ A3' ∪ A4 := by
    rintro xs ⟨k, -, hkN, h⟩
    have hlen : ((List.ofFn xs).take k).length = k := by simp [hkN]
    unfold FalseFire at h
    rw [hlen] at h
    rcases h with ⟨h1, h2⟩ | ⟨h1 | h1, h2, h3⟩ | ⟨h1, h2⟩
    · exact Or.inl (Or.inl (Or.inl ⟨h1, k, hkN, h2⟩))
    · exact Or.inl (Or.inl (Or.inr ⟨h1, k, hkN, h2, h3⟩))
    · exact Or.inl (Or.inr ⟨h1, k, hkN, h3⟩)
    · exact Or.inr ⟨h1, k, hkN, h2⟩
  have h1 : (Measure.pi fun _ : Fin N => D).real A1 ≤ (C.level j) := by
    by_cases hc : D.real {x | fE read C T E x} ≤ pE
    · exact real_le_of_sub D ha.le (fun xs h => h.2) (seq_any_le D (fE read C T E) hθs0
        (by linarith) (by linarith) hθs1 ha (by nlinarith) N)
    · exact real_le_of_not D ha.le hc fun xs h => h.1
  have hPS : ∀ x, fP read C T E x = true → fS read C T E x = true :=
    fun x hx => searches_of_ptStr (by simp only [fP] at hx; simpa using hx)
  have h3 : (Measure.pi fun _ : Fin N => D).real A3 ≤ (C.level j) := by
    by_cases hc : D.real {x | fP read C T E x} ≤ pM
    · exact real_le_of_sub D ha.le (fun xs h => h.2) (mid_any_le D (fS read C T E)
        (fP read C T E) hPS hθpt0 hθpt hθpt1 ha hp0 hc hfall N)
    · exact real_le_of_not D ha.le hc fun xs h => h.1
  have h3' : (Measure.pi fun _ : Fin N => D).real A3' ≤ (C.level j) := by
    by_cases hc : D.real {x | fP read C T E x} ≤ C.middleRate * D.real {x | fS read C T E x}
    · exact real_le_of_sub D ha.le (fun xs h => h.2) (mid_rel_le D (fS read C T E)
        (fP read C T E) hPS hθpt0 hθpt hθpt1 ha hc N)
    · exact real_le_of_not D ha.le hc fun xs h => h.1
  have h4 : (Measure.pi fun _ : Fin N => D).real A4 ≤ (C.level j) := by
    by_cases hc : ε < D.real {x | fSW x}
    · exact real_le_of_sub D ha.le (fun xs h => h.2) (seq_any_le D fSW
        (by linarith) hεd1 hεd0 (by linarith) ha (by nlinarith) N)
    · exact real_le_of_not D ha.le hc fun xs h => h.1
  calc _ ≤ (Measure.pi fun _ : Fin N => D).real (A1 ∪ A3 ∪ A3' ∪ A4) := measureReal_mono hsub
    _ ≤ (Measure.pi fun _ : Fin N => D).real A1 + (Measure.pi fun _ : Fin N => D).real A3
        + (Measure.pi fun _ : Fin N => D).real A3' + (Measure.pi fun _ : Fin N => D).real A4 := by
      refine (measureReal_union_le _ _).trans (add_le_add_left ?_ _)
      refine (measureReal_union_le _ _).trans (add_le_add_left ?_ _)
      exact measureReal_union_le _ _
    _ ≤ _ := by linarith

open scoped Classical in
/-- The stretch outlives `Ns` probes with chance at most: `(1 − θr)^Ns` where the probes changing
the hypothesis are at least `θr`; the success test's silence at look `N₁` where the searches and
the probes stopping undecided at their ends are at most `εd'`; the ends' test's silence where the
latter are at least `θe'`; fewer than `searchShare · Ns` searches where they are more than
`εd' − θe'`; and the middles' test's silence where the undecided middles are at least `θpt'` of
the searches. -/
theorem surv_le {s : Level3State α} (hs : Inv read s) (hT : s.tree = T) (hE : s.edges = E)
    (hN₁s : N₁ ≤ Ns) (ha : 0 < C.level j) (hθpt0 : 0 < C.middleRate)
    (hθpt : C.middleRate < C.middleAlternative)
    (hθpt1 : C.middleAlternative < 1) (hεd0 : 0 < C.disagreementAlternative)
    (hεd : C.disagreementAlternative < C.disagreementRate) (hεd1 : C.disagreementRate < 1)
    (hθs0 : 0 < C.endsRate) (hθs : C.endsRate < C.endsAlternative)
    (hθs1 : C.endsAlternative < 1)
    {θr εd' θe' θpt' : ℝ} (hθr0 : 0 ≤ θr) (hθr1 : θr ≤ 1)
    (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1) (hθe'0 : 0 ≤ θe') (hθe'1 : θe' ≤ εd')
    (hθpt'0 : C.middleAlternative ≤ θpt') (hθpt'1 : θpt' ≤ 1)
    (hsep : θr ≤ (1 - θpt') * (εd' - θe')) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D).real
        {xs | Ns ≤ N ∧ Survives read C T E j N₁ ((List.ofFn xs).take Ns)}
      ≤ (1 - θr) ^ Ns
        + (εd' * (C.disagreementRate / C.disagreementAlternative)
            + (1 - εd') * ((1 - C.disagreementRate) / (1 - C.disagreementAlternative))) ^ N₁
          / C.level j
        + (θe' * (C.endsRate / C.endsAlternative)
            + (1 - θe') * ((1 - C.endsRate) / (1 - C.endsAlternative))) ^ Ns / C.level j
        + 2 ^ (⌈C.searchShare * Ns⌉₊ - 1) * (1 - (εd' - θe') / 2) ^ Ns
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
  set te := C.endsRate
  set me := C.endsAlternative
  set BE := θe' * (te / me) + (1 - θe') * ((1 - te) / (1 - me))
  set εs' := εd' - θe' with hεs'
  have hεs'0 : 0 ≤ εs' := by linarith
  have hεs'1 : εs' ≤ 1 := by linarith
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
  have hD10 : 0 ≤ 2 ^ (sm - 1) * (1 - εs' / 2) ^ Ns := by
    have : 0 ≤ 1 - εs' / 2 := by linarith
    positivity
  have hBE0 : 0 ≤ BE ^ Ns / a := by
    have : 0 ≤ te / me := div_nonneg hθs0.le (by linarith)
    have : 0 ≤ (1 - te) / (1 - me) := div_nonneg (by linarith) (by linarith)
    have : 0 ≤ 1 - θe' := by linarith
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
      rintro xs ⟨-, h1, -, -, -⟩
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
  set qd := D.real {x | fSE read C T E x}
  by_cases hC : qd ≤ εd'
  · have hsub : {xs : Fin N → FreeMonoid α | Ns ≤ N ∧
        Survives read C T E j N₁ ((List.ofFn xs).take Ns)}
        ⊆ {xs | ¬ seqRate ρ d₁ a N₁ (cnt (fSE read C T E) N₁ (List.ofFn xs))} := by
      rintro xs ⟨-, -, h2, -, -⟩
      rw [htake] at h2
      exact h2
    calc _ ≤ _ := measureReal_mono hsub
      _ = binE N₁ qd (fun h => if ¬ seqRate ρ d₁ a N₁ h then 1 else 0) :=
          pi_cnt D _ N N₁ (hN₁s.trans hNs) (fun h => ¬ seqRate ρ d₁ a N₁ h)
      _ = binE N₁ qd silD := binE_congr fun h _ => by
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
  set qE := D.real {x | fE read C T E x}
  by_cases hEe : θe' ≤ qE
  · have hsub : {xs : Fin N → FreeMonoid α | Ns ≤ N ∧
        Survives read C T E j N₁ ((List.ofFn xs).take Ns)}
        ⊆ {xs | ¬ seqRate te me a Ns (cnt (fE read C T E) Ns (List.ofFn xs))} := by
      rintro xs ⟨-, -, -, h3, -⟩
      rw [hlen] at h3
      exact h3
    set silE := fun h : ℕ => if seqRate te me a Ns h then (0 : ℝ) else 1
    calc _ ≤ _ := measureReal_mono hsub
      _ = binE Ns qE (fun h => if ¬ seqRate te me a Ns h then 1 else 0) :=
          pi_cnt D _ N Ns hNs (fun h => ¬ seqRate te me a Ns h)
      _ = binE Ns qE silE := binE_congr fun h _ => by
          by_cases hf : seqRate te me a Ns h <;> simp [silE, hf]
      _ ≤ binE Ns θe' silE :=
          binE_le_of_le_anti_on hθe'0 hEe measureReal_le_one Ns _ fun i i' hii' hi' => by
            have := likelihoodRatio_mono hθs0 (by linarith) hθs.le hθs1.le hii' hi'
            by_cases hfi : seqRate te me a Ns i
            · simp only [silE]
              have hfi' : seqRate te me a Ns i' := le_trans hfi this
              rw [if_pos hfi, if_pos hfi']
            · simp only [silE]
              rw [if_neg hfi]; split_ifs <;> norm_num
      _ ≤ BE ^ Ns / a :=
          binE_silent_le hθs0 (by linarith) (by linarith) hθs1 ha hθe'0 (by linarith) Ns
      _ ≤ _ := by linarith
  push Not at hEe
  set q := D.real {x | fS read C T E x}
  have hSE : qd ≤ q + qE := by
    refine (measureReal_mono fun x hx => ?_).trans (measureReal_union_le _ _)
    simp only [Set.mem_ofPred_eq, fSE, Bool.or_eq_true] at hx
    exact hx
  have hCS : εs' < q := by linarith
  set pP := D.real {x | fP read C T E x}
  have hEOK : EdgesOK read T E := hT ▸ hE ▸ hs.edges
  have hcov : {x | fS read C T E x = true}
      ⊆ {x | fP read C T E x = true} ∪ {x | fW read C T E x = true} := by
    intro x hx
    simp only [Set.mem_ofPred_eq, fS] at hx
    have hOK := probeOrAgree_ok read T E C.walkStart hEOK x
    generalize hg : probeOrAgree read T E C.walkStart x = o at hx hOK
    cases o with
    | wrongEdge p c u t => exact Or.inr (by simp [fW, hg, changes])
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
  have hq0 : 0 < q := lt_of_le_of_lt hεs'0 hCS
  have hPq : θpt' * q ≤ pP := by
    have : θr ≤ (1 - θpt') * q :=
      hsep.trans (mul_le_mul_of_nonneg_left hCS.le (by linarith))
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
    rintro xs ⟨-, -, -, -, h3⟩
    rw [hlen] at h3
    by_cases hS : cnt (fS read C T E) Ns (List.ofFn xs) < sm
    · exact Or.inl hS
    · refine Or.inr ⟨not_lt.1 hS, fun hf => h3 ⟨?_, hf⟩⟩
      exact Nat.ceil_le.1 (not_lt.1 hS)
  calc _ ≤ _ := measureReal_mono hsub
    _ ≤ _ := measureReal_union_le _ _
    _ ≤ 2 ^ (sm - 1) * (1 - εs' / 2) ^ Ns + BM ^ sm / a := by
      gcongr
      · calc _ = binE Ns q (fun i => if i < sm then 1 else 0) :=
            pi_cnt D _ N Ns hNs (fun i => i < sm)
          _ ≤ binE Ns εs' (fun i => if i < sm then 1 else 0) :=
              binE_le_of_le_anti hεs'0 hCS.le measureReal_le_one Ns _ fun i i' h => by
                by_cases hi' : i' < sm
                · rw [if_pos hi', if_pos (by omega)]
                · rw [if_neg hi']; split_ifs <;> norm_num
          _ ≤ _ := binE_lt_le hεs'0 hεs'1 Ns sm
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
    [IsProbabilityMeasure D] (N₁ : ℕ) (Ge θgpt ηm pM θr εd' θe' θpt' : ℝ)
    {s : Level3State α} (hs : Inv read s) (hf : s = newHypothesis s.tree s.edges s.hypotheses)
    (hT : s.tree ∈ (classSet (Fintype.card σ) : Finset (DTree α)))
    (hNe : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)), ∀ E, EdgesInto T E →
      D.real (goodAt read C.walkStart ⟨T, E⟩ M U θ .ends) ≤ Ge)
    (hNm : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)), ∀ E, EdgesInto T E →
      D.real (goodAt read C.walkStart ⟨T, E⟩ M U θ .middles)
        ≤ θgpt * D.real {x | searches (probeOrAgree read T E C.walkStart x) = true} + ηm)
    (hX : ∀ ys s' x, Inv read s' → StInv read C s.tree s.edges s.hypotheses ys s' → X s' x
      → XEv (ys ++ [x]))
    (hbX : ∀ N, (Measure.pi fun _ : Fin N => D).real
      {xs | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N ∧ XEv ((List.ofFn xs).take n)} ≤ bX)
    (hN₁ : 1 ≤ N₁) (hN₁s : N₁ ≤ Ns) (ha : 0 < C.level s.hypotheses)
    (hθs0 : 0 < C.endsRate) (hθs : C.endsRate < C.endsAlternative)
    (hθs1 : C.endsAlternative < 1) (hps : max (2 * Ge) y ≤ C.endsRate)
    (hθpt0 : 0 < C.middleRate) (hθpt : C.middleRate < C.middleAlternative)
    (hθpt1 : C.middleAlternative < 1) (hθg : 4 * θgpt ≤ C.middleRate) (hpM0 : 0 ≤ pM)
    (hyM : y ≤ pM) (hηm : 4 * ηm ≤ pM)
    (hfall : pM * (C.middleAlternative / C.middleRate)
        * ((1 - C.middleAlternative) / (1 - C.middleRate)) ^ (C.searchShare - 1)
      + (1 - pM) * ((1 - C.middleAlternative) / (1 - C.middleRate)) ^ C.searchShare ≤ 1)
    (hεd0 : 0 < C.disagreementAlternative) (hεd : C.disagreementAlternative < C.disagreementRate)
    (hεd1 : C.disagreementRate < 1) (hε : C.disagreementRate ≤ ε) (hθr0 : 0 ≤ θr) (hθr1 : θr ≤ 1)
    (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1) (hθe'0 : 0 ≤ θe') (hθe'1 : θe' ≤ εd')
    (hθpt'0 : C.middleAlternative ≤ θpt') (hθpt'1 : θpt' ≤ 1)
    (hsep : θr ≤ (1 - θpt') * (εd' - θe')) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D)
        {xs | SegB (applyProbe read C) (BadStep read C M U θ D ε Ns y X) Same s (List.ofFn xs)}
      ≤ ENNReal.ofReal
          (stretchRisk C (C.level s.hypotheses) Ns N₁ θr εd' θe' θpt' + bX) := by
  classical
  have hsub : {xs : Fin N → FreeMonoid α |
      SegB (applyProbe read C) (BadStep read C M U θ D ε Ns y X) Same s (List.ofFn xs)}
      ⊆ ({xs | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N ∧
          FalseFire read C D ε s.tree s.edges s.hypotheses (max (2 * Ge) y) pM
            ((List.ofFn xs).take n)}
        ∪ {xs | Ns ≤ N ∧ Survives read C s.tree s.edges s.hypotheses N₁ ((List.ofFn xs).take Ns)})
        ∪ {xs | ∃ n ∈ Finset.Icc 1 Ns, n ≤ N ∧ XEv ((List.ofFn xs).take n)} := by
    intro xs hxs
    have hst0 : StInv read C s.tree s.edges s.hypotheses [] s := by
      have := stInv_fresh read C s.tree s.edges s.hypotheses
      rwa [← hf] at this
    have h₁0 : Look₁ read C s.tree s.edges s.hypotheses N₁ [] := fun h => by simp at h; omega
    have hn0 : s.probeCount < Ns := by rw [hf]; simp only [newHypothesis]; omega
    have hE := edgesInto_of_ok hs.edges
    obtain ⟨i, hi, hiN, h⟩ := seg_reduce read C M U θ D ε Ns y s.tree s.edges s.hypotheses N₁
      (hNe _ hT _ hE) (hNm _ hT _ hE) hyM hηm hθg X XEv
      (fun ys s' x hs' hst' hx => hX ys s' x hs' hst' hx) hN₁s (List.ofFn xs) []
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
          + (θe' * (C.endsRate / C.endsAlternative)
              + (1 - θe') * ((1 - C.endsRate) / (1 - C.endsAlternative))) ^ Ns
            / C.level s.hypotheses
          + 2 ^ (⌈C.searchShare * Ns⌉₊ - 1) * (1 - (εd' - θe') / 2) ^ Ns
          + (θpt' * (C.middleRate / C.middleAlternative)
              + (1 - θpt') * ((1 - C.middleRate) / (1 - C.middleAlternative)))
                ^ ⌈C.searchShare * Ns⌉₊ / C.level s.hypotheses)
        + bX := by
      gcongr
      · exact fire_any_le read C D ε Ns s.tree s.edges s.hypotheses hpM0 ha hθs0 hθs hθs1 hps
          hθpt0 hθpt hθpt1 hfall hεd0 hεd hεd1 hε N
      · exact surv_le read C D Ns s.tree s.edges s.hypotheses N₁ hs rfl rfl hN₁s ha hθpt0 hθpt
          hθpt1 hεd0 hεd hεd1 hθs0 hθs hθs1 hθr0 hθr1 hεd'0 hεd'1 hθe'0 hθe'1 hθpt'0 hθpt'1 hsep
          N
    _ = _ := by
      unfold stretchRisk
      ring

end Random

end OrthoDFA
