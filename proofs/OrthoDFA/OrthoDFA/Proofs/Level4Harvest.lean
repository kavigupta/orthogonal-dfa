import OrthoDFA.Proofs.Level4Sub
import OrthoDFA.Proofs.Level4Law

/-!
# A harvest's sampler finds mostly strings at read-states that are not good

Each place's counts only grow, by the probe's own charge, and the round keeps the hypothesis each
probe ran under. The sampler draws a fresh probe under one of them. Under every hypothesis of the
class, whatever came before, `NoiseEvent` bounds the good read-states' undecided strings by `θgs` a
probe at the start and by `θg` of an edge's reads there. What the tests certify is the mixture's
own total: a count less twice the running sum of its means, exponentiated, is a supermartingale,
and so is an edge's reads' means' half-sum less its reads, so each passes `log(1/a)` times its
scale only with chance `a`. A start or edge harvest whose sampler fails `MixGood` has one of them
past that. A stretch keeps one hypothesis: where its sampler fails `MixGood` while it searches on
at least `φpt/2` of the probes, a search stops at an undecided middle at most `θpt` of the time,
and the middles test fires with chance at most `a`. The class's trees only gain splits that are
not genuine, so a run that ends in the class was in it throughout, at every hypothesis it kept.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Counts

variable (C : Level4Settings) (cut : FreeMonoid α → Option Bool)

/-- A probe's outcome adds its own charge to each place's counts. -/
theorem recordProbe_countProbe (s : Level4State α) (x : FreeMonoid α) :
    (recordProbe C cut s x).edgeUndecided =
      (fun p c => s.edgeUndecided p c ++ edgeHarvestBy cut s.tree s.edges C.walkStart x (p, c))
      ∧ (recordProbe C cut s x).edgeReads
        = (fun p c => s.edgeReads p c + edgeReadsBy cut s.tree s.edges C.walkStart x (p, c))
      ∧ (recordProbe C cut s x).startUndecided = s.startUndecided
          ++ startHarvestBy cut s.tree C.walkStart x
      ∧ (recordProbe C cut s x).probes = s.probes + 1 := by
  unfold recordProbe
  split
  · split
    · split <;> simp [Level4State.countProbe, Level4State.setEdge, Level4State.newStretch]
    · simp [Level4State.countProbe]
  · split <;> simp [Level4State.countProbe, Level4State.addRecord]
  all_goals simp [Level4State.countProbe]

theorem settleOne_counts {m Lmax : ℕ} {s s' : Level4State α}
    (h : settleOne cut m Lmax s = .inl s') :
    s'.edgeUndecided = s.edgeUndecided ∧ s'.edgeReads = s.edgeReads
        ∧ s'.startUndecided = s.startUndecided ∧ s'.probes = s.probes := by
  unfold settleOne at h
  split_ifs at h with hv
  · simp only [Sum.inl.injEq] at h
    subst h
    unfold fixEdge
    simp only []
    split
    · split_ifs <;> simp [Level4State.setEdge, Level4State.newStretch]
    · simp [Level4State.setEdge, Level4State.newStretch]
  · simp only [Sum.inl.injEq] at h
    subst h
    exact ⟨rfl, rfl, rfl, rfl⟩

theorem applyProbe_counts {s s' : Level4State α} {x : FreeMonoid α}
    (h : applyProbe C cut s x = .inl s') :
    s'.edgeUndecided = (recordProbe C cut s x).edgeUndecided
        ∧ s'.edgeReads = (recordProbe C cut s x).edgeReads
      ∧ s'.startUndecided = (recordProbe C cut s x).startUndecided
          ∧ s'.probes = (recordProbe C cut s x).probes := by
  unfold applyProbe at h
  simp only [] at h
  split at h
  · cases h
  · exact settleOne_counts cut h

/-- A step that continues starts a fresh stretch, or is the probe's outcome at the same hypothesis,
one probe further into the stretch. -/
theorem applyProbe_newStretch {s s' : Level4State α} {x : FreeMonoid α}
    (h : applyProbe C cut s x = .inl s') :
    (s'.stretchProbes = 0 ∧ s'.stretchDisagreements = 0 ∧ s'.stretchMiddles = [])
      ∨ (s' = recordProbe C cut s x ∧ s'.stretchProbes = s.stretchProbes + 1 ∧ s'.tree = s.tree
          ∧ s'.edges = s.edges) := by
  unfold applyProbe at h
  simp only [] at h
  split at h
  · cases h
  unfold settleOne at h
  split_ifs at h with hv
  · simp only [Sum.inl.injEq] at h
    subst h
    left
    unfold fixEdge
    simp only []
    split
    · split_ifs <;> simp [Level4State.setEdge, Level4State.newStretch]
    · simp [Level4State.setEdge, Level4State.newStretch]
  · simp only [Sum.inl.injEq] at h
    subst h
    unfold recordProbe
    split
    · split
      · split
        · left; simp [Level4State.setEdge, Level4State.newStretch]
        · right; simp [Level4State.countProbe]
      · right; simp [Level4State.countProbe]
    · right; split <;> simp [Level4State.countProbe, Level4State.addRecord]
    all_goals right; simp [Level4State.countProbe]

theorem applyProbe_inr_pre {s s' : Level4State α} {x : FreeMonoid α} {e : Level4Ending α}
    (h : applyProbe C cut s x = .inr (e, s')) (he : e ≠ .tooBig) :
    s' = recordProbe C cut s x ∧ runTests C (recordProbe C cut s x) = some e := by
  unfold applyProbe at h
  simp only [] at h
  split at h
  · rename_i e' hl
    simp only [Sum.inr.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨rfl, hl⟩
  · unfold settleOne at h
    split_ifs at h
    simp only [Sum.inr.injEq, Prod.mk.injEq] at h
    exact absurd h.1.symm he


theorem recordProbe_versions (s : Level4State α) (x : FreeMonoid α) :
    (recordProbe C cut s x).versions = s.versions ++ [(s.tree, s.edges)] := by
  unfold recordProbe
  split
  · split
    · split <;> simp [Level4State.countProbe, Level4State.setEdge, Level4State.newStretch]
    · simp [Level4State.countProbe]
  · split <;> simp [Level4State.countProbe, Level4State.addRecord]
  all_goals simp [Level4State.countProbe]

theorem settleOne_versions {m Lmax : ℕ} {s s' : Level4State α}
    (h : settleOne cut m Lmax s = .inl s') : s'.versions = s.versions := by
  unfold settleOne at h
  split_ifs at h with hv
  · simp only [Sum.inl.injEq] at h
    subst h
    unfold fixEdge
    simp only []
    split
    · split_ifs <;> simp [Level4State.setEdge, Level4State.newStretch]
    · simp [Level4State.setEdge, Level4State.newStretch]
  · simp only [Sum.inl.injEq] at h
    subst h
    rfl

theorem applyProbe_versions {s s' : Level4State α} {x : FreeMonoid α}
    (h : applyProbe C cut s x = .inl s') : s'.versions = (recordProbe C cut s x).versions := by
  unfold applyProbe at h
  simp only [] at h
  split at h
  · cases h
  · exact settleOne_versions cut h

/-- A probe that leaves its stretch going keeps the hypothesis. -/
theorem recordProbe_keeps (s : Level4State α) (x : FreeMonoid α)
    (h : (recordProbe C cut s x).stretchProbes ≠ 0) :
    (recordProbe C cut s x).tree = s.tree ∧ (recordProbe C cut s x).edges = s.edges := by
  revert h
  unfold recordProbe
  split
  · split
    · split
      · simp [Level4State.setEdge, Level4State.newStretch]
      · simp [Level4State.countProbe]
    · simp [Level4State.countProbe]
  · split <;> simp [Level4State.countProbe, Level4State.addRecord]
  all_goals simp [Level4State.countProbe]

end Counts

section Reach

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : Level4Settings)
  (cut : FreeMonoid α → Option Bool) (S : ℕ)

/-- Reached by the loop: a tree of the class's shape, edges and records pointing at leaves, and a
hypothesis kept for each probe, with edges into its leaves and at most the current tree's splits
that are not genuine. -/
def Reach (s : Level4State α) : Prop :=
  (∃ f, G.Grown s.tree f) ∧ EdgesInto s.tree s.edges ∧ RecsInto s
      ∧ (s.stretchProbes = 0 → s.stretchMiddles = []) ∧ s.versions.length = s.probes
      ∧ ∀ v ∈ s.versions, EdgesInto v.1 v.2 ∧ ∀ f, G.Grown s.tree f → ∃ f' ≤ f, G.Grown v.1 f'

theorem reach_start : Reach G (initialState : Level4State α) :=
  ⟨⟨0, .start⟩, fun _ _ _ _ h => by simp [initialState] at h,
    fun _ _ _ h => by simp [initialState] at h, fun _ => rfl, rfl,
    fun _ h => by simp [initialState] at h⟩

theorem reach_step (hm : 0 < C.recordsToMove) {s s' : Level4State α} {x : FreeMonoid α}
    (hs : Reach G s) (h : applyProbe C cut s x = .inl s') :
    Reach G s' ∧ ∀ f f', G.Grown s.tree f → G.Grown s'.tree f' → f ≤ f' := by
  obtain ⟨⟨f, hf⟩, he, hr, -, hvl, hv⟩ := hs
  have hpt : s'.stretchProbes = 0 → s'.stretchMiddles = [] := fun h0 => by
    rcases applyProbe_newStretch C cut h with ⟨-, -, h'⟩ | ⟨-, h', -⟩
    · exact h'
    · omega
  have hvers : s'.versions = s.versions ++ [(s.tree, s.edges)] := by
    rw [applyProbe_versions C cut h, recordProbe_versions]
  have hvl' : s'.versions.length = s'.probes := by
    rw [hvers, (applyProbe_counts C cut h).2.2.2, (recordProbe_countProbe C cut s x).2.2.2,
      List.length_append, hvl]
    rfl
  have core : (∃ f', G.Grown s'.tree f') ∧ EdgesInto s'.tree s'.edges ∧ RecsInto s'
      ∧ ∀ f₁ f₂, G.Grown s.tree f₁ → G.Grown s'.tree f₂ → f₁ ≤ f₂ := by
    rcases (applyProbe_spec cut C hm he hr).1 s' h with ⟨ht, he', hr'⟩ |
      ⟨p, c, t, t₀, ht, ht₀, hp, hT, he', hr', -, -⟩
    · refine ⟨⟨f, ht ▸ hf⟩, he', hr', fun f₁ f₂ h₁ h₂ => ?_⟩
      rw [ht] at h₂
      exact (G.grown_unique h₁ h₂).le
    · have hg : G.Grown s'.tree f ∨ G.Grown s'.tree (f + 1) := by
        by_cases hgen : G.GenuineSplit s.tree p (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀))
        · exact .inl (hT ▸ .real hf ht ht₀ hgen)
        · exact .inr (hT ▸ .fake hf hp ht ht₀ hgen)
      refine ⟨hg.elim (⟨_, ·⟩) (⟨_, ·⟩), he', ?_, fun f₁ f₂ h₁ h₂ => ?_⟩
      · intro q e r hmem
        rw [hr' q e] at hmem
        cases hmem
      · have := G.grown_unique hf h₁
        rcases hg with hg | hg <;> have := G.grown_unique hg h₂ <;> omega
  obtain ⟨hg', he', hr', hmono⟩ := core
  refine ⟨⟨hg', he', hr', hpt, hvl', fun v hvm => ?_⟩, hmono⟩
  rw [hvers] at hvm
  rcases List.mem_append.1 hvm with hvm | hvm
  · obtain ⟨hve, hvf⟩ := hv v hvm
    refine ⟨hve, fun f₂ h₂ => ?_⟩
    obtain ⟨f', hf'le, hf'⟩ := hvf f hf
    exact ⟨f', hf'le.trans (hmono f f₂ hf h₂), hf'⟩
  · rw [List.mem_singleton] at hvm
    subst hvm
    exact ⟨he, fun f₂ h₂ => ⟨f, hmono f f₂ hf h₂, hf⟩⟩

theorem notClass_step (hm : 0 < C.recordsToMove) {s s' : Level4State α} {x : FreeMonoid α}
    (hs : Reach G s) (hn : ¬ G.InClass S s.tree) (h : applyProbe C cut s x = .inl s') :
    ¬ G.InClass S s'.tree := by
  obtain ⟨f, hf⟩ := hs.1
  have hfS : S < f := by
    by_contra hle
    exact hn ⟨f, by omega, hf⟩
  rintro ⟨f', hf'S, hf'⟩
  have := (reach_step G C cut hm hs h).2 f f' hf hf'
  omega

theorem class_paths {T : DTree α} (hT : G.InClass S T) (hL : Fintype.card σ + S + 2 ≤ C.maxLeaves) :
    T.leaves.length ≤ C.maxLeaves := by
  obtain ⟨f, hf, hg⟩ := hT
  have := G.grown_paths hg
  omega

/-- At a reached state of the class, every hypothesis it kept is of the class, with edges into its
leaves. -/
theorem reach_versions {s : Level4State α} (hs : Reach G s) (hc : G.InClass S s.tree) :
    ∀ v ∈ s.versions, G.InClass S v.1 ∧ EdgesInto v.1 v.2 := by
  intro v hv
  obtain ⟨f, hfS, hf⟩ := hc
  obtain ⟨hve, hvf⟩ := hs.2.2.2.2.2 v hv
  obtain ⟨f', hf'le, hf'⟩ := hvf f hf
  exact ⟨⟨f', hf'le.trans hfS, hf'⟩, hve⟩

end Reach

section Potential

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : Level4Settings)
  (cut : FreeMonoid α → Option Bool) (S : ℕ)

/-- One probe's exponentiated increment averages at most `1`. -/
theorem mgf_step {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
    (D : Measure X) [IsProbabilityMeasure D] (U N : X → ℕ) {θ L R : ℝ} (hθ : 0 ≤ θ) (hL : 0 < L)
    (hR : 0 ≤ R) (hb : ∀ᵐ x ∂D, (U x : ℝ) ≤ L ∧ (N x : ℝ) ≤ L * R) {η : ℝ} (hη : 0 ≤ η)
    (hmean : ∫ x, (U x : ℝ) ∂D ≤ θ * ∫ x, (N x : ℝ) ∂D + η) :
    ∫⁻ x, ENNReal.ofReal (Real.exp ((U x - 2 * θ * N x - 2 * η) / (2 * L * (1 + 4 * θ * R)))) ∂D
      ≤ 1 := by
  have hK : 0 < 2 * L * (1 + 4 * θ * R) := by positivity
  have hsplit : ∀ x, ENNReal.ofReal (Real.exp ((U x - 2 * θ * N x - 2 * η)
      / (2 * L * (1 + 4 * θ * R)))) = ENNReal.ofReal (Real.exp ((U x - 2 * θ * N x)
        / (2 * L * (1 + 4 * θ * R)))) * ENNReal.ofReal (Real.exp (-(2 * η)
          / (2 * L * (1 + 4 * θ * R)))) := by
    intro x
    rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
    congr 2
    ring
  simp_rw [hsplit]
  rw [lintegral_mul_const _ (measurable_of_countable _)]
  set l := 1 / (2 * L * (1 + 4 * θ * R))
  have hl : 0 < l := by positivity
  have hlL : l * L * (1 + 4 * θ * R) = 1 / 2 := by
    simp only [l]; field_simp
  set A := l + l ^ 2 * L
  set B := 2 * θ * l - 4 * θ ^ 2 * l ^ 2 * L * R
  set g : X → ℝ := fun x => 1 + A * U x - B * N x
  have hUi : Integrable (fun x => (U x : ℝ)) D :=
    Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable L (by
      filter_upwards [hb] with x hx
      rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]; exact hx.1)
  have hNi : Integrable (fun x => (N x : ℝ)) D :=
    Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable (L * R) (by
      filter_upwards [hb] with x hx
      rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]; exact hx.2)
  have hgi : Integrable g D := ((integrable_const 1).add (hUi.const_mul A)).sub (hNi.const_mul B)
  have hpt : ∀ᵐ x ∂D, Real.exp ((U x - 2 * θ * N x) / (2 * L * (1 + 4 * θ * R))) ≤ g x := by
    filter_upwards [hb] with x ⟨hU, hN⟩
    have hU0 : (0 : ℝ) ≤ U x := Nat.cast_nonneg _
    have hN0 : (0 : ℝ) ≤ N x := Nat.cast_nonneg _
    set y := (U x - 2 * θ * N x) / (2 * L * (1 + 4 * θ * R))
    have hy : y = l * (U x - 2 * θ * N x) := by simp only [y, l]; ring
    have hy1 : |y| ≤ 1 := by
      rw [hy, abs_le]
      constructor
      · have : l * (2 * θ * N x) ≤ 1 := by
          calc l * (2 * θ * N x) ≤ l * (2 * θ * (L * R)) := by gcongr
            _ ≤ l * L * (1 + 4 * θ * R) := by
              nlinarith [mul_nonneg (mul_nonneg hl.le hL.le) (mul_nonneg hθ hR)]
            _ ≤ 1 := by rw [hlL]; norm_num
        nlinarith
      · have : l * U x ≤ 1 := by
          calc l * U x ≤ l * L := by gcongr
            _ ≤ l * L * (1 + 4 * θ * R) := by
              nlinarith [mul_nonneg (mul_nonneg hl.le hL.le) (mul_nonneg hθ hR)]
            _ ≤ 1 := by rw [hlL]; norm_num
        nlinarith [mul_nonneg hl.le (mul_nonneg hθ hN0)]
    have hexp := Real.abs_exp_sub_one_sub_id_le hy1
    have hsq : y ^ 2 ≤ l ^ 2 * (L * U x + 4 * θ ^ 2 * (L * R) * N x) := by
      rw [hy, mul_pow]
      gcongr
      nlinarith [mul_le_mul_of_nonneg_left hU hU0, mul_le_mul_of_nonneg_left hN hN0,
        mul_nonneg hθ (mul_nonneg hU0 hN0)]
    have h1 : Real.exp y ≤ 1 + y + y ^ 2 := by
      have := (abs_le.1 hexp).2
      linarith
    have h2 : 1 + y + l ^ 2 * (L * U x + 4 * θ ^ 2 * (L * R) * N x) = g x := by
      simp only [g, A, B]; rw [hy]; ring
    linarith
  calc (∫⁻ x, ENNReal.ofReal (Real.exp ((U x - 2 * θ * N x) / (2 * L * (1 + 4 * θ * R)))) ∂D)
        * ENNReal.ofReal (Real.exp (-(2 * η) / (2 * L * (1 + 4 * θ * R))))
      ≤ (∫⁻ x, ENNReal.ofReal (g x) ∂D)
        * ENNReal.ofReal (Real.exp (-(2 * η) / (2 * L * (1 + 4 * θ * R)))) := by
        exact mul_le_mul' (lintegral_mono_ae (by
          filter_upwards [hpt] with x hx using ENNReal.ofReal_le_ofReal hx)) le_rfl
    _ = ENNReal.ofReal (∫ x, g x ∂D)
        * ENNReal.ofReal (Real.exp (-(2 * η) / (2 * L * (1 + 4 * θ * R)))) := by
        rw [ofReal_integral_eq_lintegral_ofReal hgi]
        filter_upwards [hpt] with x hx using (Real.exp_pos _).le.trans hx
    _ ≤ 1 := by
        rw [← ENNReal.ofReal_mul' (Real.exp_pos _).le, ← ENNReal.ofReal_one]
        refine ENNReal.ofReal_le_ofReal ?_
        suffices h : ∫ x, g x ∂D ≤ 1 + 2 * η / (2 * L * (1 + 4 * θ * R)) by
          have h1 := Real.add_one_le_exp (2 * η / (2 * L * (1 + 4 * θ * R)))
          have h2 : Real.exp (2 * η / (2 * L * (1 + 4 * θ * R)))
              * Real.exp (-(2 * η) / (2 * L * (1 + 4 * θ * R))) = 1 := by
            rw [← Real.exp_add, show 2 * η / (2 * L * (1 + 4 * θ * R))
              + -(2 * η) / (2 * L * (1 + 4 * θ * R)) = 0 by ring, Real.exp_zero]
          have h3 := (Real.exp_pos (-(2 * η) / (2 * L * (1 + 4 * θ * R)))).le
          nlinarith
        have hint : ∫ x, g x ∂D = 1 + A * ∫ x, (U x : ℝ) ∂D - B * ∫ x, (N x : ℝ) ∂D := by
          have e1 := integral_sub (μ := D) (f := fun x => 1 + A * (U x : ℝ))
            (g := fun x => B * (N x : ℝ)) ((integrable_const 1).add (hUi.const_mul A))
            (hNi.const_mul B)
          have e2 := integral_add (μ := D) (f := fun _ => (1 : ℝ)) (g := fun x => A * (U x : ℝ))
            (integrable_const 1) (hUi.const_mul A)
          simp only [g]
          rw [e1, e2, integral_const_mul, integral_const_mul, integral_const]
          simp
        have hEN : 0 ≤ ∫ x, (N x : ℝ) ∂D := integral_nonneg fun _ => Nat.cast_nonneg _
        have hA : 0 ≤ A := by positivity
        rw [hint]
        have h1 := mul_le_mul_of_nonneg_left hmean hA
        have hkey : A * θ - B = -(θ * l) / 2 := by
          simp only [A, B]
          have : l * L = 1 / 2 / (1 + 4 * θ * R) := by
            rw [← hlL]; field_simp
          field_simp
          nlinarith [hlL]
        have hAl : A ≤ 2 * l := by
          have : l * L ≤ 1 / 2 := by
            rw [← hlL]
            nlinarith [mul_nonneg (mul_nonneg hl.le hL.le) (mul_nonneg hθ hR)]
          simp only [A]; nlinarith
        have hl' : 2 * η / (2 * L * (1 + 4 * θ * R)) = 2 * η * l := by simp only [l]; ring
        rw [hl']
        nlinarith [mul_nonneg (mul_nonneg hθ hl.le) hEN, mul_le_mul_of_nonneg_left hAl hη]

/-- A count's halved mean exceeds it by much only rarely. -/
theorem mgf_lower {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
    (D : Measure X) [IsProbabilityMeasure D] (N : X → ℕ) {B : ℝ} (hB : 0 < B)
    (hb : ∀ᵐ x ∂D, (N x : ℝ) ≤ B) :
    ∫⁻ x, ENNReal.ofReal (Real.exp (((∫ y, (N y : ℝ) ∂D) / 2 - N x) / (2 * B))) ∂D ≤ 1 := by
  set r := ∫ y, (N y : ℝ) ∂D
  have hNi : Integrable (fun x => (N x : ℝ)) D :=
    Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable B (by
      filter_upwards [hb] with x hx
      rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]; exact hx)
  have hpt : ∀ᵐ x ∂D, Real.exp (-(N x : ℝ) / (2 * B)) ≤ 1 - (N x : ℝ) / (4 * B) := by
    filter_upwards [hb] with x hx
    have hN0 : (0 : ℝ) ≤ N x := Nat.cast_nonneg _
    set y := (N x : ℝ) / (2 * B) with hy
    have hy0 : 0 ≤ y := by positivity
    have hy1 : y ≤ 1 / 2 := by rw [hy, div_le_iff₀ (by positivity)]; linarith
    have habs : |-y| ≤ 1 := by rw [abs_neg, abs_of_nonneg hy0]; linarith
    have h1 := (abs_le.1 (Real.abs_exp_sub_one_sub_id_le habs)).2
    have h2 : -(N x : ℝ) / (2 * B) = -y := by rw [hy]; ring
    have h3 : (N x : ℝ) / (4 * B) = y / 2 := by rw [hy]; field_simp; ring
    rw [h2, h3]
    nlinarith
  have hsplit : ∀ x, ENNReal.ofReal (Real.exp ((r / 2 - N x) / (2 * B)))
      = ENNReal.ofReal (Real.exp (r / (4 * B)))
        * ENNReal.ofReal (Real.exp (-(N x : ℝ) / (2 * B))) := by
    intro x
    rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
    congr 2
    field_simp
    ring
  simp_rw [hsplit]
  rw [lintegral_const_mul _ (measurable_of_countable _)]
  have hgi : Integrable (fun x => 1 - (N x : ℝ) / (4 * B)) D :=
    (integrable_const 1).sub (hNi.div_const _)
  have hg0 : ∀ᵐ x ∂D, 0 ≤ 1 - (N x : ℝ) / (4 * B) := by
    filter_upwards [hb] with x hx
    have : (N x : ℝ) / (4 * B) ≤ 1 / 4 := by rw [div_le_iff₀ (by positivity)]; linarith
    linarith
  have hint : ∫ x, (1 - (N x : ℝ) / (4 * B)) ∂D = 1 - r / (4 * B) := by
    rw [integral_sub (integrable_const 1) (hNi.div_const _), integral_const, integral_div]
    simp [r]
  calc ENNReal.ofReal (Real.exp (r / (4 * B)))
        * ∫⁻ x, ENNReal.ofReal (Real.exp (-(N x : ℝ) / (2 * B))) ∂D
      ≤ ENNReal.ofReal (Real.exp (r / (4 * B)))
        * ∫⁻ x, ENNReal.ofReal (1 - (N x : ℝ) / (4 * B)) ∂D := by
        exact mul_le_mul' le_rfl (lintegral_mono_ae (by
          filter_upwards [hpt] with x hx using ENNReal.ofReal_le_ofReal hx))
    _ = ENNReal.ofReal (Real.exp (r / (4 * B)) * (1 - r / (4 * B))) := by
        rw [← ofReal_integral_eq_lintegral_ofReal hgi hg0, hint,
          ENNReal.ofReal_mul (Real.exp_pos _).le]
    _ ≤ 1 := by
        rw [← ENNReal.ofReal_one]
        refine ENNReal.ofReal_le_ofReal ?_
        have h1 := Real.add_one_le_exp (-(r / (4 * B)))
        have h2 : Real.exp (r / (4 * B)) * Real.exp (-(r / (4 * B))) = 1 := by
          rw [← Real.exp_add]; simp
        have h3 := (Real.exp_pos (r / (4 * B))).le
        nlinarith

/-- At some probe from a state of the class, the potential after the probe's outcome reaches
`thr`. -/
def EvP (Φ : Level4State α → ℝ) (thr : ℝ) : Level4State α → List (FreeMonoid α) → Prop
  | _, [] => False
  | s, x :: xs => (G.InClass S s.tree ∧ thr ≤ Φ (recordProbe C cut s x))
    ∨ match applyProbe C cut s x with
      | .inl s' => EvP Φ thr s' xs
      | .inr _ => False

theorem evP_notClass (hm : 0 < C.recordsToMove) (Φ : Level4State α → ℝ) (thr : ℝ) :
    ∀ (l : List (FreeMonoid α)) (s : Level4State α), Reach G s → ¬ G.InClass S s.tree →
      ¬ EvP G C cut S Φ thr s l
  | [], _, _, _ => id
  | x :: xs, s, hs, hn => by
    simp only [EvP, hn, false_and, false_or]
    rcases h : applyProbe C cut s x with s' | _
    · exact evP_notClass hm Φ thr xs s' (reach_step G C cut hm hs h).1
        (notClass_step G C cut S hm hs hn h)
    · exact id

/-- A potential whose exponentiated step averages at most one at every reached state of the class
reaches `thr` with chance at most `exp ((Φ s - thr) / K)`. -/
theorem pot_super (hm : 0 < C.recordsToMove) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (Φ : Level4State α → ℝ) {K : ℝ} (hK : 0 < K)
    (hkeep : ∀ s s' x, applyProbe C cut s x = .inl s' → Φ s' = Φ (recordProbe C cut s x))
    (hstep : ∀ s, Reach G s → G.InClass S s.tree →
      ∫⁻ x, ENNReal.ofReal (Real.exp ((Φ (recordProbe C cut s x) - Φ s) / K)) ∂D ≤ 1)
    (thr : ℝ) :
    ∀ (T : ℕ) (s : Level4State α), Reach G s →
      (Measure.pi fun _ : Fin T => D) {xs | EvP G C cut S Φ thr s (List.ofFn xs)}
        ≤ ENNReal.ofReal (Real.exp ((Φ s - thr) / K)) := by
  intro T
  induction T with
  | zero => intro s _; simp [EvP]
  | succ T ih =>
    intro s hs
    rw [pi_succ_apply]
    by_cases hc : G.InClass S s.tree
    swap
    · have : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
          {xs : Fin (T + 1) → FreeMonoid α | EvP G C cut S Φ thr s (List.ofFn xs)}} = 0 := by
        intro x
        have hempty : {xs | Fin.cons x xs ∈
            {xs : Fin (T + 1) → FreeMonoid α | EvP G C cut S Φ thr s (List.ofFn xs)}} = ∅ := by
          ext xs
          simp only [Set.mem_ofPred_eq, List.ofFn_cons, Set.mem_empty_iff_false, iff_false]
          exact evP_notClass G C cut S hm Φ thr _ s hs hc
        rw [hempty, measure_empty]
      simp only [this, lintegral_zero]
      exact zero_le
    have hsec : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
        {xs : Fin (T + 1) → FreeMonoid α | EvP G C cut S Φ thr s (List.ofFn xs)}}
        ≤ ENNReal.ofReal (Real.exp ((Φ (recordProbe C cut s x) - thr) / K)) := by
      intro x
      by_cases hz : thr ≤ Φ (recordProbe C cut s x)
      · refine prob_le_one.trans ?_
        rw [← ENNReal.ofReal_one]
        exact ENNReal.ofReal_le_ofReal (Real.one_le_exp (div_nonneg (by linarith) hK.le))
      rcases hst : applyProbe C cut s x with s' | ⟨e', s'⟩
      · have : {xs | Fin.cons x xs ∈
            {xs : Fin (T + 1) → FreeMonoid α | EvP G C cut S Φ thr s (List.ofFn xs)}}
            = {xs | EvP G C cut S Φ thr s' (List.ofFn xs)} := by
          ext xs
          simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvP, hz, and_false, false_or, hst]
        rw [this, ← hkeep s s' x hst]
        exact ih s' (reach_step G C cut hm hs hst).1
      · have : {xs | Fin.cons x xs ∈
            {xs : Fin (T + 1) → FreeMonoid α | EvP G C cut S Φ thr s (List.ofFn xs)}} = ∅ := by
          ext xs
          simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvP, hz, and_false, false_or, hst,
            Set.mem_empty_iff_false]
        rw [this, measure_empty]
        exact zero_le
    refine (lintegral_mono hsec).trans ?_
    have hfac : ∀ x, ENNReal.ofReal (Real.exp ((Φ (recordProbe C cut s x) - thr) / K))
        = ENNReal.ofReal (Real.exp ((Φ s - thr) / K))
          * ENNReal.ofReal (Real.exp ((Φ (recordProbe C cut s x) - Φ s) / K)) := by
      intro x
      rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
      congr 2
      ring
    simp only [hfac]
    rw [lintegral_const_mul _ (measurable_of_countable _)]
    exact (mul_le_mul_right (hstep s hs hc) _).trans (le_of_eq (mul_one _))

end Potential

section Sites

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : Level4Settings)
  (D : Measure (FreeMonoid α)) (rd : FreeMonoid α → ARU)

/-- The start's undecided strings less twice the running sum of their means. -/
noncomputable def startPot (s : Level4State α) : ℝ :=
  s.startUndecided.length - 2 * (s.versions.map (siteTotal D rd C.walkStart .start)).sum

/-- An edge's undecided strings less twice the running sum of their means. -/
noncomputable def edgePot (e : List Bool × α) (s : Level4State α) : ℝ :=
  (s.edgeUndecided e.1 e.2).length
    - 2 * (s.versions.map (siteTotal D rd C.walkStart (.edge e))).sum

/-- The mean count of the strings a probe walked from `k` under `v` first reads at the edge
`e`. -/
noncomputable def readMean (k : ℕ) (e : List Bool × α) (v : DTree α × EdgeTable α) : ℝ :=
  ∫ x, (edgeReadsBy (fun z => (rd z).cut) v.1 v.2 k x e : ℝ) ∂D

/-- Half the running sum of an edge's reads' means less its reads. -/
noncomputable def readPot (e : List Bool × α) (s : Level4State α) : ℝ :=
  (s.versions.map (readMean D rd C.walkStart e)).sum / 2 - s.edgeReads e.1 e.2

theorem startPot_pre (s : Level4State α) (x : FreeMonoid α) :
    startPot C D rd (recordProbe C (fun z => (rd z).cut) s x) - startPot C D rd s
      = (startHarvestBy (fun z => (rd z).cut) s.tree C.walkStart x).length
        - 2 * siteTotal D rd C.walkStart .start (s.tree, s.edges) := by
  obtain ⟨-, -, h3, -⟩ := recordProbe_countProbe C (fun z => (rd z).cut) s x
  simp only [startPot, h3, recordProbe_versions, List.length_append, List.map_append,
    List.sum_append, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  push_cast
  ring

theorem edgePot_pre (e : List Bool × α) (s : Level4State α) (x : FreeMonoid α) :
    edgePot C D rd e (recordProbe C (fun z => (rd z).cut) s x) - edgePot C D rd e s
      = (edgeHarvestBy (fun z => (rd z).cut) s.tree s.edges C.walkStart x e).length
        - 2 * siteTotal D rd C.walkStart (.edge e) (s.tree, s.edges) := by
  obtain ⟨h1, -, -, -⟩ := recordProbe_countProbe C (fun z => (rd z).cut) s x
  simp only [edgePot, h1, recordProbe_versions, List.length_append, List.map_append,
    List.sum_append, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Prod.mk.eta]
  push_cast
  ring

theorem readPot_pre (e : List Bool × α) (s : Level4State α) (x : FreeMonoid α) :
    readPot C D rd e (recordProbe C (fun z => (rd z).cut) s x) - readPot C D rd e s
      = readMean D rd C.walkStart e (s.tree, s.edges) / 2
        - edgeReadsBy (fun z => (rd z).cut) s.tree s.edges C.walkStart x e := by
  obtain ⟨-, h2, -, -⟩ := recordProbe_countProbe C (fun z => (rd z).cut) s x
  simp only [readPot, h2, recordProbe_versions, List.map_append, List.sum_append, List.map_cons,
    List.map_nil, List.sum_cons, List.sum_nil, Prod.mk.eta]
  push_cast
  ring

theorem startPot_keep {s s' : Level4State α} {x : FreeMonoid α}
    (h : applyProbe C (fun z => (rd z).cut) s x = .inl s') :
    startPot C D rd s' = startPot C D rd (recordProbe C (fun z => (rd z).cut) s x) := by
  simp only [startPot, (applyProbe_counts C _ h).2.2.1, applyProbe_versions C _ h]

theorem edgePot_keep (e : List Bool × α) {s s' : Level4State α} {x : FreeMonoid α}
    (h : applyProbe C (fun z => (rd z).cut) s x = .inl s') :
    edgePot C D rd e s' = edgePot C D rd e (recordProbe C (fun z => (rd z).cut) s x) := by
  simp only [edgePot, (applyProbe_counts C _ h).1, applyProbe_versions C _ h]

theorem readPot_keep (e : List Bool × α) {s s' : Level4State α} {x : FreeMonoid α}
    (h : applyProbe C (fun z => (rd z).cut) s x = .inl s') :
    readPot C D rd e s' = readPot C D rd e (recordProbe C (fun z => (rd z).cut) s x) := by
  simp only [readPot, (applyProbe_counts C _ h).2.1, applyProbe_versions C _ h]

theorem startPot_step [IsProbabilityMeasure D] (s : Level4State α) :
    ∫⁻ x, ENNReal.ofReal (Real.exp ((startPot C D rd (recordProbe C (fun z => (rd z).cut) s x)
      - startPot C D rd s) / 2)) ∂D ≤ 1 := by
  have hU : ∀ x, (startHarvestBy (fun z => (rd z).cut) s.tree C.walkStart x).length ≤ 1 := by
    intro x
    unfold startHarvestBy
    split_ifs
    · simp
    · rcases s.tree.siftCut _ (prefixOf x C.walkStart) with _ | _ <;> simp
  have h0 : 0 ≤ siteTotal D rd C.walkStart .start (s.tree, s.edges) :=
    integral_nonneg fun _ => Nat.cast_nonneg _
  have := mgf_step D (fun x => (startHarvestBy (fun z => (rd z).cut) s.tree C.walkStart x).length)
    (fun _ => 0) (θ := 0) (L := 1) (R := 0) le_rfl one_pos le_rfl
    (ae_of_all _ fun x => ⟨by exact_mod_cast hU x, by simp⟩) h0
    (by simp [siteTotal, siteHarvest])
  refine le_trans (le_of_eq ?_) this
  congr 1
  funext x
  rw [startPot_pre]
  congr 2
  push_cast
  ring

theorem edgePot_step [IsProbabilityMeasure D] {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (e : List Bool × α) (s : Level4State α) :
    ∫⁻ x, ENNReal.ofReal (Real.exp ((edgePot C D rd e (recordProbe C (fun z => (rd z).cut) s x)
      - edgePot C D rd e s) / (2 * (L + 1)))) ∂D ≤ 1 := by
  have h0 : 0 ≤ siteTotal D rd C.walkStart (.edge e) (s.tree, s.edges) :=
    integral_nonneg fun _ => Nat.cast_nonneg _
  have := mgf_step D
    (fun x => (edgeHarvestBy (fun z => (rd z).cut) s.tree s.edges C.walkStart x e).length)
    (fun _ => 0) (θ := 0) (L := (L : ℝ) + 1) (R := 0) le_rfl (by positivity) le_rfl
    (by
      filter_upwards [hlen] with x hx
      refine ⟨?_, by simp⟩
      have := edgeHarvestBy_length_le (fun z => (rd z).cut) s.tree s.edges C.walkStart x e
      exact_mod_cast (by omega :
        (edgeHarvestBy (fun z => (rd z).cut) s.tree s.edges C.walkStart x e).length ≤ L + 1))
    h0 (by simp [siteTotal, siteHarvest])
  refine le_trans (le_of_eq ?_) this
  congr 1
  funext x
  rw [edgePot_pre]
  congr 2
  push_cast
  ring

theorem readPot_step [IsProbabilityMeasure D] {L S : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hLmax : Fintype.card σ + S + 2 ≤ C.maxLeaves) (e : List Bool × α) (s : Level4State α)
    (hc : G.InClass S s.tree) :
    ∫⁻ x, ENNReal.ofReal (Real.exp ((readPot C D rd e (recordProbe C (fun z => (rd z).cut) s x)
      - readPot C D rd e s) / (2 * ((L + 1) * C.maxLeaves)))) ∂D ≤ 1 := by
  have hpaths := class_paths G C S hc hLmax
  have hB : (0 : ℝ) < (L + 1) * C.maxLeaves := by
    have : (0 : ℝ) < C.maxLeaves := by exact_mod_cast (by omega : 0 < C.maxLeaves)
    positivity
  have := mgf_lower D (fun x => edgeReadsBy (fun z => (rd z).cut) s.tree s.edges C.walkStart x e)
    hB (by
      filter_upwards [hlen] with x hx
      have h1 := edgeReadsBy_le (fun z => (rd z).cut) s.tree s.edges C.walkStart x e
      have hm := midfixes_card s.tree
      have : edgeReadsBy (fun z => (rd z).cut) s.tree s.edges C.walkStart x e
          ≤ (L + 1) * C.maxLeaves := h1.trans (Nat.mul_le_mul (by omega) (by omega))
      exact_mod_cast this)
  refine le_trans (le_of_eq ?_) this
  congr 1
  funext x
  rw [readPot_pre]
  rfl

open scoped Classical in
theorem startHarvest_indicator (k : ℕ) (T : DTree α) (P : σ → Prop) (x : FreeMonoid α) :
    ((((startHarvestBy (fun z => (rd z).cut) T k x).filter
        fun b => decide (P (G.M.eval b.toList))).length : ℕ) : ℝ)
      = (G.startAt rd k T P).indicator 1 x := by
  classical
  unfold startHarvestBy ReadModel.startAt
  by_cases hk : x.toList.length < k
  · simp [hk, Set.indicator, not_le.2 hk]
  · rcases hs : T.siftCut (fun z => (rd z).cut) (prefixOf x k) with p | b
    · simp [hk, hs, Set.indicator]
    · by_cases hg : P (G.M.eval b.toList) <;> simp [hk, hs, hg, Set.indicator, not_lt.1 hk]

open scoped Classical in
theorem siteMean_start (k : ℕ) (v : DTree α × EdgeTable α) (P : σ → Prop) :
    siteMean G D rd k .start P v = D.real (G.startAt rd k v.1 P) := by
  unfold siteMean
  rw [← integral_indicator_one (measurableSet_of_countable _)]
  exact integral_congr_ae (ae_of_all _ fun x => startHarvest_indicator G rd k v.1 P x)

open scoped Classical in
theorem middleHarvest_indicator (k : ℕ) (T : DTree α) (E : EdgeTable α) (P : σ → Prop)
    (x : FreeMonoid α) :
    ((((middleHarvestBy (fun z => (rd z).cut) T E k x).filter
        fun b => decide (P (G.M.eval b.toList))).length : ℕ) : ℝ)
      = (G.ptAt rd k T E P).indicator 1 x := by
  classical
  rcases middleHarvestBy_cases (fun z => (rd z).cut) T E k x with h | ⟨b, h⟩
  · have hx : x ∉ G.ptAt rd k T E P := by
      rintro ⟨b, hb, -⟩
      rw [h] at hb
      cases hb
    rw [Set.indicator_of_notMem hx, h]
    simp
  · by_cases hg : P (G.M.eval b.toList)
    · have hx : x ∈ G.ptAt rd k T E P := ⟨b, h, hg⟩
      rw [Set.indicator_of_mem hx, h]
      simp [hg]
    · have hx : x ∉ G.ptAt rd k T E P := by
        rintro ⟨b', hb', hg'⟩
        rw [h, List.cons.injEq] at hb'
        exact hg (hb'.1 ▸ hg')
      rw [Set.indicator_of_notMem hx, h]
      simp [hg]

open scoped Classical in
theorem siteMean_middles (k : ℕ) (v : DTree α × EdgeTable α) (P : σ → Prop) :
    siteMean G D rd k .middles P v = D.real (G.ptAt rd k v.1 v.2 P) := by
  unfold siteMean
  rw [← integral_indicator_one (measurableSet_of_countable _)]
  exact integral_congr_ae (ae_of_all _ fun x => middleHarvest_indicator G rd k v.1 v.2 P x)

open scoped Classical in
theorem siteTotal_middles (k : ℕ) (v : DTree α × EdgeTable α) :
    siteTotal D rd k .middles v = D.real (G.ptAt rd k v.1 v.2 fun _ => True) := by
  rw [← siteMean_middles G D rd k v]
  unfold siteTotal siteMean
  congr 1
  funext x
  congr 2
  exact (List.filter_eq_self.2 (by simp)).symm

end Sites

theorem binomSfGe_gt {n j : ℕ} (p : ℝ) (h : n < j) : binomSfGe n p j = 0 := by
  rw [binomSfGe_eq_range]
  exact Finset.sum_eq_zero fun i hi => by
    simp only [Finset.mem_range] at hi; rw [if_neg (by omega)]

section PT

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : Level4Settings)
  (cut : FreeMonoid α → Option Bool) (S : ℕ)

/-- After `i` probes, the run is at a state from which `Q` holds of the remaining draws. -/
def Skip (Q : Level4State α → List (FreeMonoid α) → Prop) : ℕ → Level4State α → List (FreeMonoid α)
    → Prop
  | 0, s, l => Q s l
  | _ + 1, _, [] => False
  | i + 1, s, x :: xs => match applyProbe C cut s x with
    | .inl s' => Skip Q i s' xs
    | .inr _ => False

open scoped Classical in
theorem recordProbe_stretch (s : Level4State α) (x : FreeMonoid α) :
    ((recordProbe C cut s x).stretchProbes = s.stretchProbes + 1
      ∧ (recordProbe C cut s x).stretchMiddles = s.stretchMiddles
          ++ middleHarvestBy cut s.tree s.edges C.walkStart x
      ∧ (recordProbe C cut s x).stretchDisagreements
        = s.stretchDisagreements
          + if (probeBy cut s.tree s.edges C.walkStart x).IsSearch then 1 else 0)
    ∨ ((recordProbe C cut s x).stretchProbes = 0 ∧ (recordProbe C cut s x).stretchMiddles = []
        ∧ (recordProbe C cut s x).stretchDisagreements = 0
      ∧ ¬ (probeBy cut s.tree s.edges C.walkStart x).IsSearch) := by
  unfold recordProbe
  rcases ho : probeBy cut s.tree s.edges C.walkStart x with _ | w | w | j | ⟨ps, fd⟩ | j | u <;>
    try dsimp only
  · left; simp [Level4State.countProbe, Outcome.IsSearch]
  · left; simp [Level4State.countProbe, Outcome.IsSearch]
  · left; simp [Level4State.countProbe, Outcome.IsSearch]
  · left; simp [Level4State.countProbe, Outcome.IsSearch]
  · left
    rcases recordBy cut C.walkStart (s.tree, s.edges) x with _ | ⟨⟨p, c, t⟩, sp⟩ <;>
      simp [Level4State.countProbe, Level4State.addRecord, Outcome.IsSearch]
  · left; simp [Level4State.countProbe, Outcome.IsSearch]
  · rcases x.toList[u.toList.length]? with _ | c <;> try dsimp only
    · left; simp [Level4State.countProbe, Outcome.IsSearch]
    rcases s.tree.siftCut cut u with p | b <;> try dsimp only
    swap
    · left; simp [Level4State.countProbe, Outcome.IsSearch]
    rcases s.tree.siftCut cut (u * FreeMonoid.of c) with t | b <;> rcases s.edges p c with _ | e <;>
      try dsimp only
    · right; simp [Level4State.setEdge, Level4State.newStretch, Outcome.IsSearch]
    all_goals left; simp [Level4State.countProbe, Outcome.IsSearch]

open scoped Classical in
theorem length_ptPre (rd : FreeMonoid α → ARU) (s : Level4State α) (x : FreeMonoid α) :
    (recordProbe C (fun z => (rd z).cut) s x).stretchMiddles.length
      ≤ s.stretchMiddles.length
        + if x ∈ G.ptAt rd C.walkStart s.tree s.edges (fun _ => True) then 1 else 0 := by
  have h : (recordProbe C (fun z => (rd z).cut) s x).stretchMiddles
      = s.stretchMiddles ++ middleHarvestBy (fun z => (rd z).cut)
      s.tree s.edges C.walkStart x ∨ (recordProbe C
          (fun z => (rd z).cut) s x).stretchMiddles = [] :=
    (recordProbe_stretch C _ s x).imp (·.2.1) (·.2.1)
  rcases h with h | h
  · rw [h, List.length_append]
    rcases middleHarvestBy_cases (fun z => (rd z).cut) s.tree s.edges C.walkStart x with h0
        | ⟨b, hb⟩
    · rw [h0]; simp
    · have : x ∈ G.ptAt rd C.walkStart s.tree s.edges (fun _ => True) := ⟨b, hb, trivial⟩
      rw [hb, if_pos this]
      simp
  · rw [h]; simp

theorem mem_searchAt {rd : FreeMonoid α → ARU} {k : ℕ} {T : DTree α} {edges : EdgeTable α}
    {x : FreeMonoid α} :
    x ∈ searchAt rd k T edges ↔ (probeBy (fun z => (rd z).cut) T edges k x).IsSearch := by
  simp only [searchAt_eq, Set.mem_setOf_eq]
  rcases probeBy (fun z => (rd z).cut) T edges k x with _ | _ | _ | _ | _ | _ | _ <;>
    simp [Outcome.IsSearch]

theorem ptAt_sub_searchAt (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : EdgeTable α)
    (P : σ → Prop) : G.ptAt rd k T edges P ⊆ searchAt rd k T edges := by
  rintro x ⟨b, hb, -⟩
  rw [mem_searchAt]
  unfold middleHarvestBy at hb
  rcases ho : probeBy (fun z => (rd z).cut) T edges k x with _ | _ | _ | _ | _ | _ | _ <;>
    rw [ho] at hb <;> simp_all [Outcome.IsSearch]

/-- From a state of the class, within the stretch, the undecided middles reach `k` more at the
probe that brings the stretch's searches to `t` more. -/
def EvR (ok : Level4State α → Prop) : ℕ → ℕ → Level4State α → List (FreeMonoid α) → Prop
  | _, _, _, [] => False
  | t, k, s, x :: xs => (G.InClass S s.tree ∧ ok s)
      ∧ (recordProbe C cut s x).stretchDisagreements - s.stretchDisagreements ≤ t ∧
    (((recordProbe C cut s x).stretchDisagreements - s.stretchDisagreements = t
        ∧ k ≤ (recordProbe C cut s x).stretchMiddles.length - s.stretchMiddles.length)
      ∨ match applyProbe C cut s x with
        | .inl s' => s'.stretchProbes ≠ 0 ∧ EvR ok (t
            - ((recordProbe C cut s x).stretchDisagreements - s.stretchDisagreements))
            (k - ((recordProbe C cut s x).stretchMiddles.length
                - s.stretchMiddles.length)) s' xs
        | .inr _ => False)

theorem evR_ne_nil {ok : Level4State α → Prop} {t k : ℕ} {s : Level4State α}
    {l : List (FreeMonoid α)} (h : EvR G C cut S ok t k s l) : l ≠ [] := by
  rintro rfl; exact h

theorem evR_mono (ok : Level4State α → Prop) :
    ∀ (l : List (FreeMonoid α)) (t : ℕ) (s : Level4State α) {k k' : ℕ}, k ≤ k' →
      EvR G C cut S ok t k' s l → EvR G C cut S ok t k s l
  | [], _, _, _, _, _, h => h
  | x :: xs, t, s, k, k', hk, h => by
    simp only [EvR] at h ⊢
    obtain ⟨hc, hdt, h⟩ := h
    refine ⟨hc, hdt, ?_⟩
    rcases h with ⟨h1, h2⟩ | h
    · exact .inl ⟨h1, hk.trans h2⟩
    · right
      rcases hst : applyProbe C cut s x with s' | _ <;> rw [hst] at h
      · exact ⟨h.1, evR_mono ok xs _ s' (by omega) h.2⟩
      · exact h

theorem evR_notClass (ok : Level4State α → Prop) :
    ∀ (l : List (FreeMonoid α)) (t k : ℕ) (s : Level4State α),
      ¬ (G.InClass S s.tree ∧ ok s) → ¬ EvR G C cut S ok t k s l
  | [], _, _, _, _ => id
  | x :: xs, t, k, s, hn => fun h => hn h.1

open scoped Classical in
theorem recordProbe_dis_sub (rd : FreeMonoid α → ARU) (s : Level4State α) (x : FreeMonoid α) :
    (recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements - s.stretchDisagreements
      = if x ∈ searchAt rd C.walkStart s.tree s.edges then 1 else 0 := by
  rcases recordProbe_stretch C (fun z => (rd z).cut) s x with ⟨-, -, h⟩ | ⟨-, -, h, hn⟩
  · rw [h]
    by_cases hx : x ∈ searchAt rd C.walkStart s.tree s.edges
    · rw [if_pos hx, if_pos (mem_searchAt.1 hx)]; omega
    · rw [if_neg hx, if_neg (fun h' => hx (mem_searchAt.2 h'))]; omega
  · rw [h, if_neg (fun h' => hn (mem_searchAt.1 h'))]; omega

/-- At a hypothesis searching on at least `φpt/2` of the probes, at most `θpt` of whose searches
stop at an undecided middle. -/
def PTLow (D : Measure (FreeMonoid α)) (rd : FreeMonoid α → ARU) (s : Level4State α) : Prop :=
  C.searchShare / 2 ≤ D.real (searchAt rd C.walkStart s.tree s.edges)
    ∧ D.real (G.ptAt rd C.walkStart s.tree s.edges fun _ => True)
      ≤ C.middleUndecidedRate * D.real (searchAt rd C.walkStart s.tree s.edges)

open scoped Classical in
/-- At such hypotheses, the undecided middles among the stretch's next `t` searches are at most
binomial. -/
theorem ptR_bin (hm : 0 < C.recordsToMove) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (rd : FreeMonoid α → ARU) (hθ0 : 0 ≤ C.middleUndecidedRate)
    (hθ1 : C.middleUndecidedRate ≤ 1) :
    ∀ (T t k : ℕ) (s : Level4State α), Reach G s →
      (Measure.pi fun _ : Fin T => D)
          {xs | EvR G C (fun z => (rd z).cut) S
            (PTLow G C D rd) t k s
            (List.ofFn xs)}
        ≤ ENNReal.ofReal (binomSfGe t (C.middleUndecidedRate) k) := by
  intro T
  induction T with
  | zero => intro t k s _; simp [EvR]
  | succ T ih =>
    intro t k s hs
    rcases k with _ | k
    · rw [binomSfGe_zero_right, ENNReal.ofReal_one]; exact prob_le_one
    by_cases hc : G.InClass S s.tree ∧ PTLow G C D rd s
    swap
    · have : {xs : Fin (T + 1) → FreeMonoid α |
          EvR G C (fun z => (rd z).cut) S
            (PTLow G C D rd) t
                (k + 1) s
            (List.ofFn xs)} = ∅ := by
        ext xs
        simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
        exact evR_notClass G C _ S _ _ t _ s hc
      rw [this, measure_empty]; exact zero_le
    set A := G.ptAt rd C.walkStart s.tree s.edges fun _ => True
    set B := searchAt rd C.walkStart s.tree s.edges
    have hAB : A ⊆ B := ptAt_sub_searchAt G rd C.walkStart s.tree s.edges _
    have hA : D.real A ≤ C.middleUndecidedRate * D.real B := hc.2.2
    have hA0 : 0 ≤ D.real A := measureReal_nonneg
    have hAB' : D.real A ≤ D.real B := measureReal_mono hAB
    have hB1 : D.real B ≤ 1 := by
      have := measureReal_mono (μ := D) (Set.subset_univ B)
      rwa [probReal_univ] at this
    have hdt := fun x => recordProbe_dis_sub C rd s x
    have hdg : ∀ x, (recordProbe C (fun z => (rd z).cut) s x).stretchMiddles.length
        - s.stretchMiddles.length
        ≤ if x ∈ A then 1 else 0 := fun x => by
      have := length_ptPre G C rd s x
      split_ifs at this ⊢ <;> omega
    -- the section at `x` of the event
    have hsecE : ∀ x, {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
        EvR G C (fun z => (rd z).cut) S
            (PTLow G C D rd)
            t (k + 1) s (List.ofFn xs)}}
        ⊆ {xs | (recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements
            - s.stretchDisagreements ≤ t ∧
          (((recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements
              - s.stretchDisagreements = t
            ∧ k + 1 ≤ (recordProbe C (fun z => (rd z).cut) s x).stretchMiddles.length
              - s.stretchMiddles.length)
          ∨ ∃ s', applyProbe C (fun z => (rd z).cut) s x = .inl s' ∧ Reach G s' ∧
            EvR G C (fun z => (rd z).cut) S
              (PTLow G C D rd)
                  (t - ((recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements
              - s.stretchDisagreements))
              (k + 1 - ((recordProbe C (fun z => (rd z).cut) s x).stretchMiddles.length
                - s.stretchMiddles.length)) s' (List.ofFn xs))} := by
      intro x xs hxs
      simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvR] at hxs
      obtain ⟨-, hd, h⟩ := hxs
      refine ⟨hd, ?_⟩
      rcases h with h | h
      · exact .inl h
      · right
        rcases hst : applyProbe C (fun z => (rd z).cut) s x with s' | _ <;> rw [hst] at h
        · exact ⟨s', rfl, (reach_step G C _ hm hs hst).1, h.2⟩
        · exact h.elim
    -- the run continues from the state after `x`
    have hcont : ∀ x t' k', (Measure.pi fun _ : Fin T => D) {xs : Fin T → FreeMonoid α |
        ∃ s', applyProbe C (fun z => (rd z).cut) s x = .inl s' ∧ Reach G s' ∧
          EvR G C (fun z => (rd z).cut) S
            (PTLow G C D rd) t' k' s'
                (List.ofFn xs)}
        ≤ ENNReal.ofReal (binomSfGe t' (C.middleUndecidedRate) k') := by
      intro x t' k'
      rcases hst : applyProbe C (fun z => (rd z).cut) s x with s' | _
      · refine le_trans (measure_mono fun xs hxs => ?_) (ih t' k' s' (reach_step G C _ hm hs hst).1)
        obtain ⟨s'', h1, -, h2⟩ := hxs
        cases h1; exact h2
      · simp
    set c₁ : ℝ := if t = 0 then 0 else binomSfGe (t - 1) (C.middleUndecidedRate) k
    set c₀ : ℝ := if t = 0 then 0 else binomSfGe (t - 1) (C.middleUndecidedRate) (k + 1)
    set c₂ : ℝ := binomSfGe t (C.middleUndecidedRate) (k + 1)
    have hc₀ : 0 ≤ c₀ := by
      simp only [c₀]; split_ifs; exact le_rfl; exact binomSfGe_nonneg hθ0 hθ1 _
    have hc₁ : 0 ≤ c₁ := by
      simp only [c₁]; split_ifs; exact le_rfl; exact binomSfGe_nonneg hθ0 hθ1 _
    have hc₂ : 0 ≤ c₂ := binomSfGe_nonneg hθ0 hθ1 _
    have hsec : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
        {xs : Fin (T + 1) → FreeMonoid α |
          EvR G C (fun z => (rd z).cut) S
            (PTLow G C D rd) t (k + 1)
                s (List.ofFn xs)}}
        ≤ A.indicator (fun _ => ENNReal.ofReal c₁) x
          + (B \ A).indicator (fun _ => ENNReal.ofReal c₀) x
          + Bᶜ.indicator (fun _ => ENNReal.ofReal c₂) x := by
      intro x
      refine (measure_mono (hsecE x)).trans ?_
      have h1 := hdt x
      have h2 := hdg x
      by_cases hxA : x ∈ A
      · have hxB := hAB hxA
        rw [Set.indicator_of_mem hxA, Set.indicator_of_notMem (fun h => h.2 hxA),
          Set.indicator_of_notMem (by simp [hxB]), add_zero, add_zero]
        rw [if_pos hxB] at h1
        rw [if_pos hxA] at h2
        rcases t with _ | t
        · refine (measure_mono (t := ∅) fun xs hxs => ?_).trans (by simp)
          exact absurd hxs.1 (by omega)
        rcases k with _ | k
        · simp only [c₁, if_neg (Nat.succ_ne_zero t), Nat.add_sub_cancel, binomSfGe_zero_right,
            ENNReal.ofReal_one]
          exact prob_le_one
        refine (measure_mono fun xs hxs => ?_).trans ((hcont x t (k + 1)).trans
          (le_of_eq (by simp [c₁])))
        obtain ⟨-, h | ⟨s', h3, h4, h5⟩⟩ := hxs
        · omega
        · refine ⟨s', h3, h4, ?_⟩
          rw [show t + 1 - ((recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements
              - s.stretchDisagreements) = t by omega]
            at h5
          exact evR_mono G C _ S _ _ t s' (by omega) h5
      by_cases hxB : x ∈ B
      · rw [Set.indicator_of_notMem hxA, Set.indicator_of_mem (Set.mem_sdiff_of_mem hxB hxA),
          Set.indicator_of_notMem (by simp [hxB]), zero_add, add_zero]
        rw [if_pos hxB] at h1
        rw [if_neg hxA] at h2
        rcases t with _ | t
        · refine (measure_mono (t := ∅) fun xs hxs => ?_).trans (by simp)
          exact absurd hxs.1 (by omega)
        refine (measure_mono fun xs hxs => ?_).trans ((hcont x t (k + 1)).trans
          (le_of_eq (by simp [c₀])))
        obtain ⟨-, h | ⟨s', h3, h4, h5⟩⟩ := hxs
        · omega
        · refine ⟨s', h3, h4, ?_⟩
          rwa [show t + 1 - ((recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements
              - s.stretchDisagreements) = t by omega,
            show k + 1 - ((recordProbe C (fun z => (rd z).cut) s x).stretchMiddles.length
              - s.stretchMiddles.length) = k + 1 by omega] at h5
      · rw [Set.indicator_of_notMem hxA, Set.indicator_of_notMem (fun h => hxB h.1),
          Set.indicator_of_mem hxB, zero_add, zero_add]
        rw [if_neg hxB] at h1
        rw [if_neg hxA] at h2
        refine (measure_mono fun xs hxs => ?_).trans (hcont x t (k + 1))
        obtain ⟨-, h | ⟨s', h3, h4, h5⟩⟩ := hxs
        · omega
        · refine ⟨s', h3, h4, ?_⟩
          rwa [show t - ((recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements
              - s.stretchDisagreements) = t by omega,
            show k + 1 - ((recordProbe C (fun z => (rd z).cut) s x).stretchMiddles.length
              - s.stretchMiddles.length) = k + 1 by omega] at h5
    rw [pi_succ_apply]
    refine (lintegral_mono hsec).trans ?_
    have hmeas : ∀ E : Set (FreeMonoid α), MeasurableSet E := fun E =>
      (Set.to_countable _).measurableSet
    rw [lintegral_add_left (measurable_of_countable _), lintegral_add_left
        (measurable_of_countable _), lintegral_indicator (hmeas _), lintegral_indicator (hmeas _),
      lintegral_indicator (hmeas _), setLIntegral_const, setLIntegral_const, setLIntegral_const,
      ← ofReal_measureReal, ← ofReal_measureReal, ← ofReal_measureReal,
      measureReal_compl (hmeas _), probReal_univ, measureReal_sdiff hAB (hmeas _),
      ← ENNReal.ofReal_mul hc₁, ← ENNReal.ofReal_mul hc₀, ← ENNReal.ofReal_mul hc₂,
      ← ENNReal.ofReal_add (mul_nonneg hc₁ hA0) (mul_nonneg hc₀ (by linarith)),
      ← ENNReal.ofReal_add (by positivity) (mul_nonneg hc₂ (by linarith))]
    refine ENNReal.ofReal_le_ofReal ?_
    rcases t with _ | t
    · simp only [c₁, c₀, if_pos rfl, zero_mul, zero_add]
      nlinarith
    · have e₁ : c₁ = binomSfGe t (C.middleUndecidedRate) k := by simp [c₁]
      have e₀ : c₀ = binomSfGe t (C.middleUndecidedRate) (k + 1) := by simp [c₀]
      have e₂ : c₂ = C.middleUndecidedRate * c₁ + (1 - C.middleUndecidedRate) * c₀ := by
        simp only [c₂, e₁, e₀]; exact binomSfGe_succ t (C.middleUndecidedRate) k
      have hmono : c₀ ≤ c₁ := by rw [e₁, e₀]; exact binomSfGe_antitone hθ0 hθ1 k
      rw [e₂]
      nlinarith [mul_le_mul_of_nonneg_right hA (sub_nonneg.2 hmono)]

theorem skip_le (hm : 0 < C.recordsToMove) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (Q : Level4State α → List (FreeMonoid α) → Prop) {c : ENNReal}
    (hQ : ∀ (T : ℕ) (s : Level4State α), Reach G s
      → (Measure.pi fun _ : Fin T => D) {xs | Q s (List.ofFn xs)} ≤ c) :
    ∀ (T i : ℕ) (s : Level4State α), Reach G s →
      (Measure.pi fun _ : Fin T => D) {xs | Skip C cut Q i s (List.ofFn xs)} ≤ c := by
  intro T
  induction T with
  | zero =>
    intro i s hs
    rcases i with _ | i
    · exact hQ 0 s hs
    · simp [Skip]
  | succ T ih =>
    intro i s hs
    rcases i with _ | i
    · exact hQ _ s hs
    rw [pi_succ_apply]
    calc ∫⁻ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
          {xs : Fin (T + 1) → FreeMonoid α | Skip C cut Q (i + 1) s (List.ofFn xs)}} ∂D
        ≤ ∫⁻ _, c ∂D := lintegral_mono fun x => by
          rcases hst : applyProbe C cut s x with s' | _
          · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
                Skip C cut Q (i + 1) s (List.ofFn xs)}}
                    = {xs | Skip C cut Q i s' (List.ofFn xs)} := by
              ext xs; simp only [Set.mem_ofPred_eq, List.ofFn_cons, Skip, hst]
            rw [this]; exact ih i s' (reach_step G C cut hm hs hst).1
          · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
                Skip C cut Q (i + 1) s (List.ofFn xs)}} = ∅ := by
              ext xs
              simp only [Set.mem_ofPred_eq, List.ofFn_cons, Skip, hst, Set.mem_empty_iff_false]
            rw [this, measure_empty]; exact zero_le
      _ = c := by rw [lintegral_const, measure_univ, mul_one]

end PT

section Low

variable (C : Level4Settings) (D : Measure (FreeMonoid α)) (rd : FreeMonoid α → ARU)

/-- The hypothesis's disagreement rate: the probes that search. -/
noncomputable def disRate (s : Level4State α) : ℝ := D.real (searchAt rd C.walkStart s.tree s.edges)

/-- The searches a probe adds to the stretch's count. -/
noncomputable def dt (s : Level4State α) (x : FreeMonoid α) : ℕ :=
  (recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements - s.stretchDisagreements

/-- From `s`, at a hypothesis searching on at most `φpt/2` of the probes, the next `N + 1` probes
stay in the stretch and search at least `h` times. -/
def EvH : ℕ → ℕ → Level4State α → List (FreeMonoid α) → Prop
  | _, _, _, [] => False
  | 0, h, s, x :: _ => disRate C D rd s ≤ C.searchShare / 2 ∧ h ≤ dt C rd s x
  | N + 1, h, s, x :: xs => disRate C D rd s ≤ C.searchShare / 2 ∧
    match applyProbe C (fun z => (rd z).cut) s x with
    | .inl s' => s'.stretchProbes ≠ 0 ∧ EvH N (h - dt C rd s x) s' xs
    | .inr _ => False

open scoped Classical in
/-- A stretch at a hypothesis searching on at most `φpt/2` of the probes searches at least `h`
times in its next `N + 1` probes with chance at most `P(Bin(N + 1, φpt/2) ≥ h)`. -/
theorem evH_le [IsProbabilityMeasure D] (hφ0 : 0 ≤ C.searchShare) (hφ1 : C.searchShare ≤ 1) :
    ∀ (T N h : ℕ) (s : Level4State α),
      (Measure.pi fun _ : Fin T => D) {xs | EvH C D rd N h s (List.ofFn xs)}
        ≤ ENNReal.ofReal (binomSfGe (N + 1) (C.searchShare / 2) h) := by
  have hp0 : 0 ≤ C.searchShare / 2 := by linarith
  have hp1 : C.searchShare / 2 ≤ 1 := by linarith
  intro T
  induction T with
  | zero => intro N h s; rcases N with _ | N <;> simp [EvH]
  | succ T ih =>
    intro N h s
    rcases h with _ | h
    · rw [binomSfGe_zero_right, ENNReal.ofReal_one]; exact prob_le_one
    by_cases hd : disRate C D rd s ≤ C.searchShare / 2
    swap
    · refine (measure_mono (t := ∅) fun xs hxs => ?_).trans (by simp)
      simp only [Set.mem_ofPred_eq, List.ofFn_succ] at hxs
      rcases N with _ | N
      · exact hd hxs.1
      · exact hd hxs.1
    set B := searchAt rd C.walkStart s.tree s.edges
    set A : ℝ := binomSfGe N (C.searchShare / 2) h
    set Cc : ℝ := binomSfGe N (C.searchShare / 2) (h + 1)
    have hC0 : 0 ≤ Cc := binomSfGe_nonneg hp0 hp1 _
    have hCA : Cc ≤ A := binomSfGe_antitone hp0 hp1 h
    have hA0 : 0 ≤ A := hC0.trans hCA
    have hsec : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
        {xs : Fin (T + 1) → FreeMonoid α | EvH C D rd N (h + 1) s (List.ofFn xs)}}
        ≤ B.indicator (fun _ => ENNReal.ofReal A) x
          + Bᶜ.indicator (fun _ => ENNReal.ofReal Cc) x := by
      intro x
      have hdt : dt C rd s x = if x ∈ B then 1 else 0 := recordProbe_dis_sub C rd s x
      by_cases hx : x ∈ B
      · rw [Set.indicator_of_mem hx, Set.indicator_of_notMem (Set.notMem_compl_iff.2 hx),
          add_zero]
        rw [if_pos hx] at hdt
        rcases N with _ | N
        · rcases h with _ | h
          · simp only [A, binomSfGe_zero_right, ENNReal.ofReal_one]; exact prob_le_one
          · refine (measure_mono (t := ∅) fun xs hxs => ?_).trans (by simp)
            simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvH] at hxs
            omega
        · rcases hst : applyProbe C (fun z => (rd z).cut) s x with s' | _
          · refine le_trans (measure_mono fun xs hxs => ?_) (ih N h s')
            simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvH, hst] at hxs
            rw [hdt, Nat.add_sub_cancel] at hxs
            exact hxs.2.2
          · refine (measure_mono (t := ∅) fun xs hxs => ?_).trans (by simp)
            simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvH, hst] at hxs
            exact hxs.2
      · rw [Set.indicator_of_notMem hx, Set.indicator_of_mem (Set.mem_compl hx), zero_add]
        rw [if_neg hx] at hdt
        rcases N with _ | N
        · refine (measure_mono (t := ∅) fun xs hxs => ?_).trans (by simp)
          simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvH] at hxs
          omega
        · rcases hst : applyProbe C (fun z => (rd z).cut) s x with s' | _
          · refine le_trans (measure_mono fun xs hxs => ?_) (ih N (h + 1) s')
            simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvH, hst] at hxs
            rw [hdt, Nat.sub_zero] at hxs
            exact hxs.2.2
          · refine (measure_mono (t := ∅) fun xs hxs => ?_).trans (by simp)
            simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvH, hst] at hxs
            exact hxs.2
    rw [pi_succ_apply]
    refine (lintegral_mono hsec).trans ?_
    have hB0 : 0 ≤ D.real B := measureReal_nonneg
    rw [lintegral_add_left (measurable_of_countable _), lintegral_indicator
        (Set.to_countable _).measurableSet, lintegral_indicator
        (Set.to_countable _).measurableSet,
      setLIntegral_const, setLIntegral_const, ← ofReal_measureReal, ← ofReal_measureReal,
      measureReal_compl (Set.to_countable _).measurableSet, probReal_univ,
      ← ENNReal.ofReal_mul hA0, ← ENNReal.ofReal_mul hC0,
      ← ENNReal.ofReal_add (mul_nonneg hA0 hB0) (mul_nonneg hC0 (by
        have := measureReal_mono (μ := D) (Set.subset_univ B)
        rw [probReal_univ] at this; linarith))]
    refine ENNReal.ofReal_le_ofReal ?_
    have hrec : binomSfGe (N + 1) (C.searchShare / 2) (h + 1)
        = C.searchShare / 2 * A + (1 - C.searchShare / 2)
            * Cc := binomSfGe_succ N (C.searchShare / 2) h
    rw [hrec]
    have hdB : D.real B ≤ C.searchShare / 2 := hd
    nlinarith

end Low

section Assemble

theorem runTests_start' (C : Level4Settings) {s : Level4State α}
    (h : runTests C s = some .harvestStart) :
    C.firstLook ≤ s.probes ∧ binomSfGe s.probes C.startUndecidedRate s.startUndecided.length
        < C.testLevel := by
  unfold runTests at h
  by_cases h1 : rateSide C.startUndecidedRate C.testLevel C.firstLook s.probes
      s.startUndecided.length = some true
  · unfold rateSide at h1
    split_ifs at h1 with h2 h3 <;> simp_all
  · rw [if_neg h1] at h
    split_ifs at h <;> simp at h

theorem runTests_harvest' (C : Level4Settings) {s : Level4State α} {e : List Bool × α}
    (h : runTests C s = some (.harvest e)) :
    e.1 ∈ s.tree.leaves ∧ C.edgeTraffic * s.probes ≤ s.edgeReads e.1 e.2
        ∧ C.edgeExcess (s.edgeReads e.1 e.2)
      ≤ ((s.edgeUndecided e.1 e.2).length : ℝ) - C.edgeUndecidedRate * s.edgeReads e.1 e.2 := by
  unfold runTests at h
  by_cases h1 : rateSide C.startUndecidedRate C.testLevel C.firstLook s.probes
      s.startUndecided.length = some true
  · rw [if_pos h1] at h; simp at h
  · rw [if_neg h1] at h
    by_cases h2 : ∃ e : List Bool × α, e.1 ∈ s.tree.leaves
        ∧ C.edgeTraffic * s.probes ≤ s.edgeReads e.1 e.2
        ∧ C.edgeExcess (s.edgeReads e.1 e.2) ≤ ((s.edgeUndecided e.1 e.2).length : ℝ)
            - C.edgeUndecidedRate * s.edgeReads e.1 e.2
    · rw [dif_pos h2] at h
      simp only [Option.some.injEq, Level4Ending.harvest.injEq] at h
      rw [← h]
      exact h2.choose_spec
    · rw [dif_neg h2] at h
      split_ifs at h <;> simp at h

theorem runTests_pt' (C : Level4Settings) {s : Level4State α} (h : runTests C s = some .harvestPT) :
    C.searchShare * s.stretchProbes ≤ s.stretchDisagreements ∧ C.firstLook ≤ s.stretchDisagreements
        ∧ binomSfGe s.stretchDisagreements C.middleUndecidedRate s.stretchMiddles.length
        < C.testLevel := by
  unfold runTests at h
  by_cases h1 : rateSide C.startUndecidedRate C.testLevel C.firstLook s.probes
      s.startUndecided.length = some true
  · rw [if_pos h1] at h; simp at h
  rw [if_neg h1] at h
  by_cases h2 : ∃ e : List Bool × α, e.1 ∈ s.tree.leaves
      ∧ C.edgeTraffic * s.probes ≤ s.edgeReads e.1 e.2
      ∧ C.edgeExcess (s.edgeReads e.1 e.2) ≤ ((s.edgeUndecided e.1 e.2).length : ℝ)
          - C.edgeUndecidedRate * s.edgeReads e.1 e.2
  · rw [dif_pos h2] at h; simp at h
  rw [dif_neg h2] at h
  by_cases h3 : C.searchShare * s.stretchProbes ≤ s.stretchDisagreements
      ∧ rateSide C.middleUndecidedRate C.testLevel C.firstLook s.stretchDisagreements
          s.stretchMiddles.length = some true
  · obtain ⟨hg, h3⟩ := h3
    unfold rateSide at h3
    split_ifs at h3 with h4 h5 <;> first | exact ⟨hg, h4, h5⟩ | simp at h3
  · rw [if_neg h3] at h
    split_ifs at h <;> simp at h

theorem paths_length_pos : ∀ T : DTree α, 0 < T.leaves.length
  | .leaf => by simp [DTree.leaves]
  | .node _ r a => by
    simp only [DTree.leaves, List.length_append, List.length_map]
    have := paths_length_pos r
    omega

theorem path_length_lt : ∀ (T : DTree α) (p : List Bool), p ∈ T.leaves → p.length < T.leaves.length
  | .leaf, p, h => by simp [DTree.leaves] at h; simp [h, DTree.leaves]
  | .node _ r a, p, h => by
    simp only [DTree.leaves, List.mem_append, List.mem_map, List.length_append,
      List.length_map] at h ⊢
    have hr := paths_length_pos r
    have ha := paths_length_pos a
    rcases h with ⟨q, hq, rfl⟩ | ⟨q, hq, rfl⟩
    · have := path_length_lt r q hq; simp; omega
    · have := path_length_lt a q hq; simp; omega

/-- The leaves' paths of length at most `n`. -/
def shortPaths (n : ℕ) : Finset (List Bool) :=
  (Finset.range (n + 1)).biUnion fun i => Finset.univ.image fun v : Fin i → Bool => List.ofFn v

theorem mem_shortPaths {n : ℕ} {p : List Bool} (h : p.length ≤ n) : p ∈ shortPaths n := by
  simp only [shortPaths, Finset.mem_biUnion, Finset.mem_range, Finset.mem_image, Finset.mem_univ,
    true_and]
  exact ⟨p.length, by omega, p.get, List.ofFn_get p⟩

theorem shortPaths_card (n : ℕ) : (shortPaths n).card ≤ 2 ^ (n + 1) := by
  refine Finset.card_biUnion_le.trans ?_
  have : ∀ i, (Finset.univ.image fun v : Fin i → Bool => List.ofFn v).card ≤ 2 ^ i := fun i =>
    Finset.card_image_le.trans (by simp)
  refine (Finset.sum_le_sum fun i _ => this i).trans ?_
  induction n with
  | zero => simp
  | succ n ih => rw [Finset.sum_range_succ, pow_succ]; omega

omit [Fintype α] [DecidableEq α] in
theorem sum_map_le_affine {β : Type*} (l : List β) (f g : β → ℝ) (a c : ℝ)
    (h : ∀ v ∈ l, f v ≤ a * g v + c) : (l.map f).sum ≤ a * (l.map g).sum + l.length * c := by
  induction l with
  | nil => simp
  | cons v l ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    have h1 := h v (by simp)
    have h2 := ih fun w hw => h w (by simp [hw])
    push_cast
    linarith

set_option maxHeartbeats 1000000 in -- the seven events and their sampler facts
open scoped Classical in
theorem harvest_good_holds : HarvestGood := by
  intro α _ _ σ _ G rd D _ C S L T ρ θg θgs θgpt hlen hL1 hm ha ha1 hθg0 hθg hφe0 hLmax hexc
    hθgs0 hθgs hroom hhalf hθgpt0 hθgpt hθpt1 hφ0 hφ1 hlow hE
  set ℓ := Real.log (1 / C.testLevel) with hℓ
  have hℓ0 : 0 ≤ ℓ := Real.log_nonneg (by rw [le_div_iff₀ ha]; linarith)
  have hθpt0 : 0 ≤ C.middleUndecidedRate := by linarith
  have hθs0 : 0 ≤ C.startUndecidedRate := by linarith
  have hθe0 : 0 ≤ C.edgeUndecidedRate := by linarith
  have hLm : (0 : ℝ) < C.maxLeaves := by exact_mod_cast (by omega : 0 < C.maxLeaves)
  have hK₁ : (0 : ℝ) < 2 * (L + 1) := by positivity
  have hK₂ : (0 : ℝ) < 2 * ((L + 1) * C.maxLeaves) := by positivity
  set keys := shortPaths C.maxLeaves ×ˢ (Finset.univ : Finset α)
  set kp : ℕ → ℕ := fun t => if h : ∃ h', C.firstLook ≤ t
      ∧ binomSfGe t C.middleUndecidedRate h' < C.testLevel then Nat.find h else t + 1
  set hf : ℕ → ℕ := fun n => max ⌈C.searchShare * n⌉₊ C.firstLook
  set ok : Level4State α → Prop := PTLow G C D rd
  set Qp : ℕ → Level4State α → List (FreeMonoid α) → Prop := fun dd s l =>
    EvR G C (fun z => (rd z).cut) S ok dd (kp dd) s l
  set Qh : ℕ → Level4State α → List (FreeMonoid α) → Prop := fun N s l =>
    EvH C D rd N (hf (N + 1)) s l
  set P : Level4Ending α → Level4State α → Prop := fun e s' => G.InClass S s'.tree ∧ e ≠ .tooBig
    ∧ ¬ GoodEnd G D rd C.walkStart C.harvestYield e s'
  have hok : ∀ s s' : Level4State α, s'.tree = s.tree → s'.edges = s.edges → (ok s' ↔ ok s) := by
    intro s s' h1 h2; simp only [ok, PTLow, h1, h2]
  have hdr : ∀ s s' : Level4State α, s'.tree = s.tree → s'.edges = s.edges →
      disRate C D rd s' = disRate C D rd s := by
    intro s s' h1 h2; simp only [disRate, h1, h2]
  -- the hypotheses a probe from a state of the class leaves kept
  have hvers : ∀ s x, Reach G s → G.InClass S s.tree →
      (∀ v ∈ (recordProbe C (fun z => (rd z).cut) s x).versions,
        G.InClass S v.1 ∧ EdgesInto v.1 v.2)
      ∧ (((recordProbe C (fun z => (rd z).cut) s x).versions.length : ℕ) : ℝ)
        = (recordProbe C (fun z => (rd z).cut) s x).probes := by
    intro s x hs hc
    rw [recordProbe_versions, (recordProbe_countProbe C _ s x).2.2.2]
    refine ⟨fun v hv => ?_, ?_⟩
    · rcases List.mem_append.1 hv with hv | hv
      · exact reach_versions G S hs hc v hv
      · rw [List.mem_singleton] at hv
        subst hv
        exact ⟨hc, hs.2.1⟩
    · rw [List.length_append, hs.2.2.2.2.1]
      simp only [List.length_singleton]
  -- a start harvest's sampler, its potential short of `2ℓ`
  have hstartFin : ∀ s x, Reach G s → G.InClass S s.tree →
      runTests C (recordProbe C (fun z => (rd z).cut) s x) = some .harvestStart →
      startPot C D rd (recordProbe C (fun z => (rd z).cut) s x) < 2 * ℓ →
      GoodEnd G D rd C.walkStart C.harvestYield .harvestStart
        (recordProbe C (fun z => (rd z).cut) s x) := by
    intro s x hs hc hrt hpot
    obtain ⟨hcv, hN⟩ := hvers s x hs hc
    obtain ⟨hn₀, hlt⟩ := runTests_start' C hrt
    have hh := hhalf _ _ hn₀ hlt
    show MixGood G D rd C.walkStart C.harvestYield
      ⟨.start, (recordProbe C (fun z => (rd z).cut) s x).versions⟩
    simp only [MixGood]
    set N : ℝ := (((recordProbe C (fun z => (rd z).cut) s x).versions.length : ℕ) : ℝ)
    set Tt := ((recordProbe C (fun z => (rd z).cut) s x).versions.map
      (siteTotal D rd C.walkStart .start)).sum
    have hG : ((recordProbe C (fun z => (rd z).cut) s x).versions.map
        (siteMean G D rd C.walkStart .start G.Good)).sum ≤ 0 * 0 + N * θgs := by
      have := sum_map_le_affine _ (siteMean G D rd C.walkStart .start G.Good) (fun _ => 0) 0 θgs
        fun v hv => by
          rw [siteMean_start]
          have := hE.goodStart v.1 (hcv v hv).1
          linarith
      simpa using this
    have hpot' : ((recordProbe C (fun z => (rd z).cut) s x).startUndecided.length : ℝ) - 2 * Tt
        < 2 * ℓ := hpot
    have hNn : (C.firstLook : ℝ) ≤ N := by rw [hN]; exact_mod_cast hn₀
    have hroom' : 8 * ℓ ≤ N * C.startUndecidedRate :=
      le_trans hroom (mul_le_mul_of_nonneg_right hNn hθs0)
    have hh' : N * C.startUndecidedRate / 2
        < ((recordProbe C (fun z => (rd z).cut) s x).startUndecided.length : ℝ) := by
      rw [hN]; exact hh
    have hT : N * C.startUndecidedRate / 8 < Tt := by linarith
    have hy : C.harvestYield ≤ C.startUndecidedRate / 8 := min_le_left _ _
    have hN0 : 0 ≤ N := Nat.cast_nonneg _
    have h16 := mul_le_mul_of_nonneg_left hθgs hN0
    refine ⟨by linarith, ?_⟩
    nlinarith [mul_le_mul_of_nonneg_right hy hN0]
  -- an edge harvest's sampler, both its potentials short
  have hedgeFin : ∀ s x e, Reach G s → G.InClass S s.tree →
      runTests C (recordProbe C (fun z => (rd z).cut) s x) = some (.harvest e) →
      edgePot C D rd e (recordProbe C (fun z => (rd z).cut) s x) < 2 * (L + 1) * ℓ →
      readPot C D rd e (recordProbe C (fun z => (rd z).cut) s x)
        < 2 * ((L + 1) * C.maxLeaves) * ℓ →
      GoodEnd G D rd C.walkStart C.harvestYield (.harvest e)
        (recordProbe C (fun z => (rd z).cut) s x) := by
    intro s x e hs hc hrt hp1 hp2
    obtain ⟨hcv, hN⟩ := hvers s x hs hc
    obtain ⟨-, hgate, hexc'⟩ := runTests_harvest' C hrt
    show MixGood G D rd C.walkStart C.harvestYield
      ⟨.edge e, (recordProbe C (fun z => (rd z).cut) s x).versions⟩
    simp only [MixGood]
    set N : ℝ := (((recordProbe C (fun z => (rd z).cut) s x).versions.length : ℕ) : ℝ)
    set Tt := ((recordProbe C (fun z => (rd z).cut) s x).versions.map
      (siteTotal D rd C.walkStart (.edge e))).sum
    set Rm := ((recordProbe C (fun z => (rd z).cut) s x).versions.map
      (readMean D rd C.walkStart e)).sum
    set U : ℝ := (((recordProbe C (fun z => (rd z).cut) s x).edgeUndecided e.1 e.2).length : ℝ)
    set R : ℝ := ((recordProbe C (fun z => (rd z).cut) s x).edgeReads e.1 e.2 : ℝ)
    set η : ℝ := C.edgeTraffic * (C.edgeUndecidedRate - 8 * θg) / 4
    have hG : ((recordProbe C (fun z => (rd z).cut) s x).versions.map
        (siteMean G D rd C.walkStart (.edge e) G.Good)).sum ≤ θg * Rm + N * η :=
      sum_map_le_affine _ _ _ _ _ fun v hv => hE.goodEdge v.1 v.2 (hcv v hv).1 (hcv v hv).2 e
    have h1 : U - 2 * Tt < 2 * (L + 1) * ℓ := hp1
    have h2 : Rm / 2 - R < 2 * ((L + 1) * C.maxLeaves) * ℓ := hp2
    have hx := hexc ((recordProbe C (fun z => (rd z).cut) s x).edgeReads e.1 e.2)
    have hgate' : C.edgeTraffic * N ≤ R := by rw [hN]; exact hgate
    have hN0 : 0 ≤ N := Nat.cast_nonneg _
    have hgap : 0 ≤ C.edgeUndecidedRate - 8 * θg := by linarith
    have hη : 4 * (N * η) ≤ R * (C.edgeUndecidedRate - 8 * θg) := by
      have := mul_le_mul_of_nonneg_right hgate' hgap
      simp only [η]
      nlinarith
    have hRm : θg * Rm ≤ θg * (2 * R + 4 * ((L + 1) * C.maxLeaves) * ℓ) :=
      mul_le_mul_of_nonneg_left (by linarith) hθg0
    have hy : C.harvestYield ≤ C.edgeTraffic * C.edgeUndecidedRate / 2 :=
      (min_le_right _ _).trans (min_le_left _ _)
    have hTlow : C.edgeUndecidedRate * R / 2 + (L + 1) * ℓ
        + 8 * θg * ((L + 1) * C.maxLeaves) * ℓ < Tt := by
      have : 4 * (L + 1) * (1 + 4 * θg * C.maxLeaves) * ℓ
          = 4 * (L + 1) * ℓ + 16 * θg * ((L + 1) * C.maxLeaves) * ℓ := by ring
      linarith
    have hL0 : 0 ≤ (L + 1) * ℓ := mul_nonneg (by positivity) hℓ0
    have hg0 : 0 ≤ 8 * θg * ((L + 1) * C.maxLeaves) * ℓ := by positivity
    refine ⟨?_, ?_⟩
    · linarith
    · have h3 := mul_le_mul_of_nonneg_right hy hN0
      have h4 := mul_le_mul_of_nonneg_left hgate' hθe0
      linarith
  -- a middles harvest's sampler, at a hypothesis searching enough, not `PTLow`
  have hptFin : ∀ s x, Reach G s → G.InClass S s.tree →
      (recordProbe C (fun z => (rd z).cut) s x).stretchProbes ≠ 0 → ¬ ok s →
      C.searchShare / 2 ≤ D.real (searchAt rd C.walkStart s.tree s.edges) →
      GoodEnd G D rd C.walkStart C.harvestYield .harvestPT
        (recordProbe C (fun z => (rd z).cut) s x) := by
    intro s x hs hc hn hk hq
    obtain ⟨htr, hed⟩ := recordProbe_keeps C _ s x hn
    show MixGood G D rd C.walkStart C.harvestYield ⟨.middles,
      [((recordProbe C (fun z => (rd z).cut) s x).tree,
        (recordProbe C (fun z => (rd z).cut) s x).edges)]⟩
    rw [htr, hed]
    simp only [MixGood, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero,
      List.length_singleton, Nat.cast_one, mul_one]
    rw [siteMean_middles, siteTotal_middles G]
    have hp : C.middleUndecidedRate * D.real (searchAt rd C.walkStart s.tree s.edges)
        < D.real (G.ptAt rd C.walkStart s.tree s.edges fun _ => True) := by
      by_contra hle
      push Not at hle
      exact hk ⟨hq, hle⟩
    have hg := hE.goodPT s.tree s.edges hc hs.2.1
    have hy : C.harvestYield ≤ C.searchShare * C.middleUndecidedRate / 2 :=
      (min_le_right _ _).trans (min_le_right _ _)
    have hq0 : 0 ≤ D.real (searchAt rd C.walkStart s.tree s.edges) := measureReal_nonneg
    refine ⟨?_, ?_⟩
    · nlinarith [mul_le_mul_of_nonneg_left hq hθgpt0,
        mul_le_mul_of_nonneg_right hθgpt hq0]
    · nlinarith [mul_le_mul_of_nonneg_left hq hθpt0]
  have hincl : ∀ (l : List (FreeMonoid α)) (s : Level4State α), Reach G s →
      RunEnds (applyProbe C fun z => (rd z).cut) P s l →
      (∃ e ∈ keys, EvP G C (fun z => (rd z).cut) S (edgePot C D rd e) (2 * (L + 1) * ℓ) s l)
        ∨ (∃ e ∈ keys, EvP G C (fun z => (rd z).cut) S (readPot C D rd e)
            (2 * ((L + 1) * C.maxLeaves) * ℓ) s l)
        ∨ EvP G C (fun z => (rd z).cut) S (startPot C D rd) (2 * ℓ) s l
        ∨ (∃ i dd, i < l.length ∧ dd ≤ l.length
          ∧ Skip C (fun z => (rd z).cut) (Qp dd) i s l)
        ∨ (∃ dd, s.stretchDisagreements ≤ dd ∧ dd - s.stretchDisagreements ≤ l.length
          ∧ EvR G C (fun z => (rd z).cut) S ok (dd - s.stretchDisagreements)
              (kp dd - s.stretchMiddles.length) s l)
        ∨ (∃ i N, i < l.length ∧ N < l.length ∧ Skip C (fun z => (rd z).cut) (Qh N) i s l)
        ∨ ∃ N, N < l.length ∧ EvH C D rd N (hf (s.stretchProbes + N + 1)
            - s.stretchDisagreements) s l := by
    intro l
    induction l with
    | nil => intro s _ h; exact h.elim
    | cons x xs ih =>
      intro s hs h
      simp only [RunEnds] at h
      rcases hst : applyProbe C (fun z => (rd z).cut) s x with s' | ⟨e, s''⟩ <;> rw [hst] at h
      · rcases ih s' (reach_step G C _ hm hs hst).1 h with ⟨e, he, hE'⟩ | ⟨e, he, hE'⟩ | hS |
          ⟨i, dd, hil, hD, hk⟩ | ⟨dd, hD1, hD2, hP⟩ | ⟨i, N, hi, hN, hk⟩ | ⟨N, hN, hH⟩
        · exact .inl ⟨e, he, .inr (by simp only [hst]; exact hE')⟩
        · exact .inr (.inl ⟨e, he, .inr (by simp only [hst]; exact hE')⟩)
        · exact .inr (.inr (.inl (.inr (by simp only [hst]; exact hS))))
        · refine .inr (.inr (.inr (.inl ⟨i + 1, dd, by simp; omega, by simp; omega, ?_⟩)))
          simp only [Skip, hst]
          exact hk
        · have hne := evR_ne_nil G C _ S hP
          have hlen1 : 1 ≤ xs.length := by
            rcases xs with _ | _
            · exact absurd rfl hne
            · simp
          rcases applyProbe_newStretch C _ hst with ⟨hn0, hd0, hpt0⟩ | ⟨hpre, hn, htr', hed'⟩
          · refine .inr (.inr (.inr (.inl ⟨1, dd, by simp; omega, by simp; omega, ?_⟩)))
            simp only [Skip, hst, Qp]
            rw [hd0, hpt0] at hP
            simpa using hP
          · subst hpre
            rcases recordProbe_stretch C _ s x with ⟨-, hpt, hdis⟩ | ⟨hn', -, -, -⟩
            swap
            · exact absurd (hn'.symm.trans hn) (by omega)
            have hgc : s.stretchMiddles.length ≤
                (recordProbe C (fun z => (rd z).cut) s x).stretchMiddles.length := by
              rw [hpt, List.length_append]; omega
            have hcl : G.InClass S s.tree ∧ ok s := by
              rcases xs with _ | ⟨y, ys⟩
              · exact hP.elim
              · obtain ⟨h1, h2⟩ := hP.1
                exact ⟨htr' ▸ h1, (hok s _ htr' hed').1 h2⟩
            have hdt : (recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements
                - s.stretchDisagreements ≤ 1 := by
              rw [hdis]; split_ifs <;> omega
            refine .inr (.inr (.inr (.inr (.inl ⟨dd, by rw [hdis] at hD1; omega,
              by simp; omega, ?_⟩))))
            simp only [EvR]
            refine ⟨hcl, by omega, .inr ?_⟩
            rw [hst]
            refine ⟨by omega, ?_⟩
            rwa [show dd - s.stretchDisagreements -
                ((recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements
                    - s.stretchDisagreements)
                = dd - (recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements by omega,
              show kp dd - s.stretchMiddles.length
                - ((recordProbe C (fun z => (rd z).cut) s x).stretchMiddles.length
                    - s.stretchMiddles.length)
                = kp dd - (recordProbe C (fun z => (rd z).cut) s x).stretchMiddles.length by
                  omega]
        · refine .inr (.inr (.inr (.inr (.inr (.inl ⟨i + 1, N, by simp; omega, by simp; omega,
            ?_⟩)))))
          simp only [Skip, hst]
          exact hk
        · have hne : xs ≠ [] := by
            rintro rfl; rcases N with _ | N <;> exact hH.elim
          have hlen1 : 1 ≤ xs.length := by
            rcases xs with _ | _
            · exact absurd rfl hne
            · simp
          rcases applyProbe_newStretch C _ hst with ⟨hn0, hd0, -⟩ | ⟨hpre, hn, htr', hed'⟩
          · refine .inr (.inr (.inr (.inr (.inr (.inl ⟨1, N, by simp; omega, by simp; omega,
              ?_⟩)))))
            simp only [Skip, hst, Qh]
            rw [hn0, hd0, zero_add, Nat.sub_zero] at hH
            exact hH
          · subst hpre
            have hdR := hdr s _ htr' hed'
            have hlo : disRate C D rd s ≤ C.searchShare / 2 := by
              rcases xs with _ | ⟨y, ys⟩
              · exact absurd rfl hne
              · rcases N with _ | N
                · exact hdR ▸ hH.1
                · exact hdR ▸ hH.1
            have hdtv : (recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements
                = s.stretchDisagreements + dt C rd s x := by
              rcases recordProbe_stretch C _ s x with ⟨-, -, hdis⟩ | ⟨hn', -, -, -⟩
              · simp only [dt]; rw [hdis]; split_ifs <;> omega
              · exact absurd (hn'.symm.trans hn) (by omega)
            refine .inr (.inr (.inr (.inr (.inr (.inr ⟨N + 1, by simp; omega, ?_⟩)))))
            simp only [EvH]
            refine ⟨hlo, ?_⟩
            rw [hst]
            refine ⟨by omega, ?_⟩
            rwa [show s.stretchProbes + (N + 1) + 1
                = (recordProbe C (fun z => (rd z).cut) s x).stretchProbes + N + 1 by
              omega, show hf ((recordProbe C (fun z => (rd z).cut) s x).stretchProbes + N + 1)
                  - s.stretchDisagreements
                - dt C rd s x = hf ((recordProbe C (fun z => (rd z).cut) s x).stretchProbes + N + 1)
                  - (recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements by omega]
      · obtain ⟨hcl, hne, hng⟩ := h
        obtain ⟨rfl, hl⟩ := applyProbe_inr_pre C _ hst hne
        have htr : (recordProbe C (fun z => (rd z).cut) s x).tree = s.tree :=
          (recordProbe_into _ C x hs.2.1 hs.2.2.1).1
        rw [htr] at hcl
        rcases e with _ | _ | e | _ | _
        · exact absurd trivial hng
        · by_cases hpot : 2 * ℓ ≤ startPot C D rd (recordProbe C (fun z => (rd z).cut) s x)
          · exact .inr (.inr (.inl (.inl ⟨hcl, hpot⟩)))
          · exact absurd (hstartFin s x hs hcl hl (not_le.1 hpot)) hng
        · obtain ⟨hp, -, -⟩ := runTests_harvest' C hl
          rw [htr] at hp
          have he : e ∈ keys := by
            refine Finset.mem_product.2 ⟨mem_shortPaths ?_, Finset.mem_univ _⟩
            have := path_length_lt _ _ hp
            have := class_paths G C S hcl hLmax
            omega
          by_cases h1 : 2 * (L + 1) * ℓ ≤ edgePot C D rd e
              (recordProbe C (fun z => (rd z).cut) s x)
          · exact .inl ⟨e, he, .inl ⟨hcl, h1⟩⟩
          by_cases h2 : 2 * ((L + 1) * C.maxLeaves) * ℓ ≤ readPot C D rd e
              (recordProbe C (fun z => (rd z).cut) s x)
          · exact .inr (.inl ⟨e, he, .inl ⟨hcl, h2⟩⟩)
          exact absurd (hedgeFin s x e hs hcl hl (not_le.1 h1) (not_le.1 h2)) hng
        · obtain ⟨hgate, hn₀, hlt⟩ := runTests_pt' C hl
          rcases recordProbe_stretch C _ s x with ⟨hn, hpt, hdis⟩ | ⟨-, hpt0, -, -⟩
          · by_cases hk : ok s
            · set dd := (recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements
              have hex : ∃ h', C.firstLook ≤ dd ∧ binomSfGe dd C.middleUndecidedRate h'
                  < C.testLevel := ⟨_, hn₀, hlt⟩
              have hfind : Nat.find hex ≤ (recordProbe C
                  (fun z => (rd z).cut) s x).stretchMiddles.length :=
                Nat.find_min' hex ⟨hn₀, hlt⟩
              have hkp : kp dd ≤ (recordProbe C
                  (fun z => (rd z).cut) s x).stretchMiddles.length := by
                simp only [kp, dif_pos hex]; omega
              have hdt : dd - s.stretchDisagreements ≤ 1 := by rw [hdis]; split_ifs <;> omega
              refine .inr (.inr (.inr (.inr (.inl ⟨dd, by rw [hdis]; omega, by simp; omega,
                ?_⟩))))
              simp only [EvR]
              exact ⟨⟨hcl, hk⟩, le_rfl, .inl ⟨rfl, by omega⟩⟩
            · by_cases hq : C.searchShare / 2 ≤ D.real (searchAt rd C.walkStart s.tree s.edges)
              · exact absurd (hptFin s x hs hcl (by omega) hk hq) hng
              have hceil : ⌈C.searchShare * ((recordProbe C
                  (fun z => (rd z).cut) s x).stretchProbes : ℝ)⌉₊
                  ≤ (recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements :=
                      Nat.ceil_le.2 hgate
              refine .inr (.inr (.inr (.inr (.inr (.inr ⟨0, by simp, ?_⟩)))))
              simp only [EvH]
              refine ⟨le_of_lt (not_le.1 hq), ?_⟩
              have hH : hf (s.stretchProbes + 0 + 1)
                  ≤ (recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements := by
                simp only [hf]
                rw [show s.stretchProbes + 0 + 1 =
                    (recordProbe C (fun z => (rd z).cut) s x).stretchProbes by omega]
                exact max_le hceil hn₀
              have hdtv : (recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements
                  = s.stretchDisagreements + dt C rd s x := by
                simp only [dt]; rw [hdis]; split_ifs <;> omega
              omega
          · rw [hpt0, List.length_nil, binomSfGe_zero_right] at hlt
            exact absurd hlt (not_lt.2 ha1)
        · exact absurd rfl hne
  have hsub : {xs : Fin T → FreeMonoid α | RunEnds (applyProbe C fun z => (rd z).cut) P initialState
      (List.ofFn xs)}
      ⊆ ((((⋃ e ∈ keys, {xs | EvP G C (fun z => (rd z).cut) S (edgePot C D rd e)
            (2 * (L + 1) * ℓ) initialState (List.ofFn xs)})
          ∪ ⋃ e ∈ keys, {xs | EvP G C (fun z => (rd z).cut) S (readPot C D rd e)
            (2 * ((L + 1) * C.maxLeaves) * ℓ) initialState (List.ofFn xs)})
          ∪ {xs | EvP G C (fun z => (rd z).cut) S (startPot C D rd) (2 * ℓ) initialState
            (List.ofFn xs)})
        ∪ ⋃ q ∈ Finset.range T ×ˢ Finset.range (T + 1),
          {xs | Skip C (fun z => (rd z).cut) (Qp q.2) q.1 initialState (List.ofFn xs)})
        ∪ ⋃ q ∈ Finset.range T ×ˢ Finset.range T,
          {xs | Skip C (fun z => (rd z).cut) (Qh q.2) q.1 initialState (List.ofFn xs)} := by
    intro xs hxs
    rcases hincl _ _ (reach_start G) hxs with ⟨e, he, h⟩ | ⟨e, he, h⟩ | h |
      ⟨i, dd, hi, hD, h⟩ | ⟨dd, -, hD, h⟩ | ⟨i, N, hi, hN, h⟩ | ⟨N, hN, h⟩
    · exact .inl (.inl (.inl (.inl (Set.mem_biUnion he h))))
    · exact .inl (.inl (.inl (.inr (Set.mem_biUnion he h))))
    · exact .inl (.inl (.inr h))
    · simp only [List.length_ofFn] at hi hD
      exact .inl (.inr (Set.mem_biUnion (x := (i, dd))
        (Finset.mem_product.2 ⟨Finset.mem_range.2 hi, Finset.mem_range.2 (by omega)⟩) h))
    · simp only [List.length_ofFn] at hD
      have hT0 : 0 < T := by
        rcases T with _ | T
        · exact absurd (evR_ne_nil G C _ S h) (by simp)
        · omega
      refine .inl (.inr (Set.mem_biUnion (x := (0, dd))
        (Finset.mem_product.2 ⟨Finset.mem_range.2 hT0, Finset.mem_range.2 (by
          simp [initialState] at hD; omega)⟩) ?_))
      simp only [Skip, Qp]
      simpa [initialState] using h
    · simp only [List.length_ofFn] at hi hN
      exact .inr (Set.mem_biUnion (x := (i, N))
        (Finset.mem_product.2 ⟨Finset.mem_range.2 hi, Finset.mem_range.2 hN⟩) h)
    · simp only [List.length_ofFn] at hN
      refine .inr (Set.mem_biUnion (x := (0, N))
        (Finset.mem_product.2 ⟨Finset.mem_range.2 (by omega), Finset.mem_range.2 hN⟩) ?_)
      simp only [Skip, Qh]
      simpa [initialState] using h
  have hpot0 : ∀ (Φ : Level4State α → ℝ) (K : ℝ), 0 < K → Φ initialState = 0 →
      ENNReal.ofReal (Real.exp ((Φ initialState - K * ℓ) / K)) = ENNReal.ofReal C.testLevel := by
    intro Φ K hK h0
    have : (0 - K * ℓ) / K = -ℓ := by field_simp; ring
    rw [h0, this, hℓ, one_div, Real.log_inv, neg_neg, Real.exp_log ha]
  have h1 : ∀ e ∈ keys, (Measure.pi fun _ : Fin T => D)
      {xs | EvP G C (fun z => (rd z).cut) S (edgePot C D rd e) (2 * (L + 1) * ℓ) initialState
        (List.ofFn xs)} ≤ ENNReal.ofReal C.testLevel := by
    intro e _
    refine (pot_super G C _ S hm D (edgePot C D rd e) hK₁ (fun s s' x h => edgePot_keep C D rd e h)
      (fun s _ _ => edgePot_step C D rd hlen e s) _ T initialState (reach_start G)).trans
        (le_of_eq (hpot0 _ _ hK₁ (by simp [edgePot, initialState])))
  have h2 : ∀ e ∈ keys, (Measure.pi fun _ : Fin T => D)
      {xs | EvP G C (fun z => (rd z).cut) S (readPot C D rd e)
        (2 * ((L + 1) * C.maxLeaves) * ℓ) initialState (List.ofFn xs)}
        ≤ ENNReal.ofReal C.testLevel := by
    intro e _
    refine (pot_super G C _ S hm D (readPot C D rd e) hK₂ (fun s s' x h => readPot_keep C D rd e h)
      (fun s _ hc => readPot_step G C D rd hlen hLmax e s hc) _ T initialState
        (reach_start G)).trans
        (le_of_eq (hpot0 _ _ hK₂ (by simp [readPot, initialState])))
  have h2' : (Measure.pi fun _ : Fin T => D)
      {xs | EvP G C (fun z => (rd z).cut) S (startPot C D rd) (2 * ℓ) initialState
        (List.ofFn xs)} ≤ ENNReal.ofReal C.testLevel :=
    (pot_super G C _ S hm D (startPot C D rd) two_pos (fun s s' x h => startPot_keep C D rd h)
      (fun s _ _ => startPot_step C D rd s) _ T initialState (reach_start G)).trans
        (le_of_eq (hpot0 _ _ two_pos (by simp [startPot, initialState])))
  have h3 : ∀ q ∈ Finset.range T ×ˢ Finset.range (T + 1), (Measure.pi fun _ : Fin T => D)
      {xs | Skip C (fun z => (rd z).cut) (Qp q.2) q.1 initialState (List.ofFn xs)}
      ≤ if q.2 = 0 then 0 else ENNReal.ofReal C.testLevel := by
    intro q _
    refine skip_le G C _ hm D (Qp q.2) (fun T' s hs => ?_) T q.1 initialState (reach_start G)
    refine (ptR_bin G C S hm D rd hθpt0 hθpt1 T' q.2 _ s hs).trans ?_
    have hkp0 : 1 ≤ kp 0 := by
      simp only [kp]
      split_ifs with hex
      · have := Nat.find_spec hex
        have h0 : Nat.find hex ≠ 0 := fun h0 => by
          rw [h0, binomSfGe_zero_right] at this; linarith [this.2]
        omega
      · omega
    split_ifs with hq
    · rw [hq, binomSfGe_gt _ (by omega), ENNReal.ofReal_zero]
    · refine ENNReal.ofReal_le_ofReal ?_
      simp only [kp]
      split_ifs with hex
      · exact (Nat.find_spec hex).2.le
      · rw [binomSfGe_gt _ (by omega)]; exact ha.le
  have h4 : ∀ q ∈ Finset.range T ×ˢ Finset.range T, (Measure.pi fun _ : Fin T => D)
      {xs | Skip C (fun z => (rd z).cut) (Qh q.2) q.1 initialState (List.ofFn xs)}
      ≤ ENNReal.ofReal C.testLevel := by
    intro q _
    refine skip_le G C _ hm D (Qh q.2) (fun T' s _ => ?_) T q.1 initialState (reach_start G)
    refine (evH_le C D rd hφ0 hφ1 T' q.2 _ s).trans (ENNReal.ofReal_le_ofReal ?_)
    have hp0 : 0 ≤ C.searchShare / 2 := by linarith
    have hp1 : C.searchShare / 2 ≤ 1 := by linarith
    by_cases hn : C.firstLook ≤ q.2 + 1
    · refine (binomSfGe_antitone' hp0 hp1 (le_max_left _ _)).trans ?_
      have := hlow (q.2 + 1) hn
      push_cast at this ⊢
      exact this
    · rw [binomSfGe_gt _ (lt_of_lt_of_le (not_le.1 hn) (le_max_right _ _))]
      exact ha.le
  have hsum3 : ∑ q ∈ Finset.range T ×ˢ Finset.range (T + 1),
      (if q.2 = 0 then 0 else ENNReal.ofReal C.testLevel) = T * T * ENNReal.ofReal C.testLevel := by
    have h : ∑ y ∈ Finset.range (T + 1), (if y = 0 then (0 : ENNReal) else ENNReal.ofReal
        C.testLevel)
        = T * ENNReal.ofReal C.testLevel := by
      rw [Finset.sum_range_succ']
      simp
    rw [Finset.sum_product (f := fun q : ℕ × ℕ => if q.2 = 0 then (0 : ENNReal)
      else ENNReal.ofReal C.testLevel)]
    simp only [h, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    ring
  have hU : ∀ A B E F H : Set (Fin T → FreeMonoid α),
      (Measure.pi fun _ : Fin T => D) ((((A ∪ B) ∪ E) ∪ F) ∪ H)
        ≤ (Measure.pi fun _ : Fin T => D) A + (Measure.pi fun _ : Fin T => D) B
          + (Measure.pi fun _ : Fin T => D) E + (Measure.pi fun _ : Fin T => D) F
          + (Measure.pi fun _ : Fin T => D) H := by
    intro A B E F H
    refine (measure_union_le _ _).trans (add_le_add ((measure_union_le _ _).trans
      (add_le_add ((measure_union_le _ _).trans (add_le_add (measure_union_le _ _) le_rfl))
        le_rfl)) le_rfl)
  refine (measure_mono hsub).trans ((hU _ _ _ _ _).trans ?_)
  calc _ ≤ ∑ e ∈ keys, ENNReal.ofReal C.testLevel + ∑ e ∈ keys, ENNReal.ofReal C.testLevel
        + ENNReal.ofReal C.testLevel
        + ∑ q ∈ Finset.range T ×ˢ Finset.range (T + 1),
          (if q.2 = 0 then 0 else ENNReal.ofReal C.testLevel)
        + ∑ _q ∈ Finset.range T ×ˢ Finset.range T, ENNReal.ofReal C.testLevel :=
        add_le_add (add_le_add (add_le_add (add_le_add
          ((measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum h1))
          ((measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum h2))) h2')
          ((measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum h3)))
          ((measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum h4))
    _ = (2 * keys.card + 1 + 2 * (T * T)) * ENNReal.ofReal C.testLevel := by
        rw [hsum3]
        simp only [Finset.sum_const, Finset.card_range, Finset.card_product, nsmul_eq_mul]
        push_cast
        ring
    _ ≤ ENNReal.ofReal ((2 ^ (C.maxLeaves + 2) * Fintype.card α + 1 + 2 * (T * T))
        * C.testLevel) := by
        rw [ENNReal.ofReal_mul (by positivity)]
        gcongr
        rw [show ((2 : ℝ) ^ (C.maxLeaves + 2) * Fintype.card α + 1 + 2 * (T * T))
            = ((2 ^ (C.maxLeaves + 2)
          * Fintype.card α + 1 + 2 * (T * T) : ℕ) : ℝ) by push_cast; ring,
          ENNReal.ofReal_natCast]
        have : keys.card ≤ 2 ^ (C.maxLeaves + 1) * Fintype.card α := by
          simp only [keys, Finset.card_product, Finset.card_univ]
          exact Nat.mul_le_mul_right _ (shortPaths_card _)
        have h2 : 2 ^ (C.maxLeaves + 2) = 2 * 2 ^ (C.maxLeaves + 1) := by ring
        exact_mod_cast (by rw [h2]; nlinarith : 2 * keys.card + 1 + 2 * (T * T)
          ≤ 2 ^ (C.maxLeaves + 2) * Fintype.card α + 1 + 2 * (T * T))

end Assemble

end OrthoDFA
