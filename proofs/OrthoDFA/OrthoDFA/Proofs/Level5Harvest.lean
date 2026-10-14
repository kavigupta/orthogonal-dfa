import OrthoDFA.Proofs.Level5Sub
import OrthoDFA.Proofs.Level5Law
import OrthoDFA.Proofs.DTree
import OrthoDFA.Proofs.HypothesisDFA
import OrthoDFA.Proofs.SeqRate

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
and the middles test, a likelihood ratio that is a supermartingale over the stretch, fires in it
with chance at most `a`; the search gate's own test of `φpt/2` against `φpt`, which its pass and
`firstLook ≥ 6 log(1/a)` make fire (`seqRate_share`), fires in a stretch searching on less than
`φpt/2` with chance at most `a`. The class's trees only gain splits that are
not genuine, so a run that ends in the class was in it throughout, at every hypothesis it kept.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Counts

variable (C : Level5Settings) (cut : FreeMonoid α → Option Bool)

/-- A probe's outcome adds its own charge to each place's counts. -/
theorem recordProbe_countProbe (s : Level5State α) (x : FreeMonoid α) :
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
    · split <;> simp [Level5State.countProbe, Level5State.setEdge, Level5State.newStretch]
    · simp [Level5State.countProbe]
  · split <;> simp [Level5State.countProbe, Level5State.addRecord]
  all_goals simp [Level5State.countProbe]

theorem settleOne_counts {m Lmax : ℕ} {s s' : Level5State α}
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
    · split_ifs <;> simp [Level5State.setEdge, Level5State.newStretch]
    · simp [Level5State.setEdge, Level5State.newStretch]
  · simp only [Sum.inl.injEq] at h
    subst h
    exact ⟨rfl, rfl, rfl, rfl⟩

theorem applyProbe_counts {s s' : Level5State α} {x : FreeMonoid α}
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
theorem applyProbe_newStretch {s s' : Level5State α} {x : FreeMonoid α}
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
    · split_ifs <;> simp [Level5State.setEdge, Level5State.newStretch]
    · simp [Level5State.setEdge, Level5State.newStretch]
  · simp only [Sum.inl.injEq] at h
    subst h
    unfold recordProbe
    split
    · split
      · split
        · left; simp [Level5State.setEdge, Level5State.newStretch]
        · right; simp [Level5State.countProbe]
      · right; simp [Level5State.countProbe]
    · right; split <;> simp [Level5State.countProbe, Level5State.addRecord]
    all_goals right; simp [Level5State.countProbe]

theorem applyProbe_inr_pre {s s' : Level5State α} {x : FreeMonoid α} {e : Level5Ending α}
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

theorem recordProbe_versions (s : Level5State α) (x : FreeMonoid α) :
    (recordProbe C cut s x).versions = s.versions ++ [(s.tree, s.edges)] := by
  unfold recordProbe
  split
  · split
    · split <;> simp [Level5State.countProbe, Level5State.setEdge, Level5State.newStretch]
    · simp [Level5State.countProbe]
  · split <;> simp [Level5State.countProbe, Level5State.addRecord]
  all_goals simp [Level5State.countProbe]

theorem settleOne_versions {m Lmax : ℕ} {s s' : Level5State α}
    (h : settleOne cut m Lmax s = .inl s') : s'.versions = s.versions := by
  unfold settleOne at h
  split_ifs at h with hv
  · simp only [Sum.inl.injEq] at h
    subst h
    unfold fixEdge
    simp only []
    split
    · split_ifs <;> simp [Level5State.setEdge, Level5State.newStretch]
    · simp [Level5State.setEdge, Level5State.newStretch]
  · simp only [Sum.inl.injEq] at h
    subst h
    rfl

theorem applyProbe_versions {s s' : Level5State α} {x : FreeMonoid α}
    (h : applyProbe C cut s x = .inl s') : s'.versions = (recordProbe C cut s x).versions := by
  unfold applyProbe at h
  simp only [] at h
  split at h
  · cases h
  · exact settleOne_versions cut h

/-- A probe that leaves its stretch going keeps the hypothesis. -/
theorem recordProbe_keeps (s : Level5State α) (x : FreeMonoid α)
    (h : (recordProbe C cut s x).stretchProbes ≠ 0) :
    (recordProbe C cut s x).tree = s.tree ∧ (recordProbe C cut s x).edges = s.edges := by
  revert h
  unfold recordProbe
  split
  · split
    · split
      · simp [Level5State.setEdge, Level5State.newStretch]
      · simp [Level5State.countProbe]
    · simp [Level5State.countProbe]
  · split <;> simp [Level5State.countProbe, Level5State.addRecord]
  all_goals simp [Level5State.countProbe]

theorem middleHarvestBy_search {T : DTree α} {E : EdgeTable α} {k : ℕ} {x : FreeMonoid α}
    (h : middleHarvestBy cut T E k x ≠ []) : (probeBy cut T E k x).IsSearch := by
  unfold middleHarvestBy at h
  rcases ho : probeBy cut T E k x with _ | _ | _ | _ | _ | _ <;> rw [ho] at h <;>
    simp_all [Outcome.IsSearch]

theorem startHarvestBy_length_le (T : DTree α) (k : ℕ) (x : FreeMonoid α) :
    (startHarvestBy cut T k x).length ≤ 1 := by
  unfold startHarvestBy
  split_ifs <;> [simp; (rcases T.siftCut cut (prefixOf x k) with _ | b <;> simp)]

end Counts

section Reach

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : Level5Settings)
  (cut : FreeMonoid α → Option Bool) (S : ℕ)

/-- Reached by the loop: a tree of the class's shape, edges and records pointing at leaves, and a
hypothesis kept for each probe, with edges into its leaves and at most the current tree's splits
that are not genuine. -/
def Reach (s : Level5State α) : Prop :=
  (∃ f, G.Grown s.tree f) ∧ EdgesInto s.tree s.edges ∧ RecsInto s
      ∧ (s.stretchProbes = 0 → s.stretchMiddles = []) ∧ s.versions.length = s.probes
      ∧ ∀ v ∈ s.versions, EdgesInto v.1 v.2 ∧ ∀ f, G.Grown s.tree f → ∃ f' ≤ f, G.Grown v.1 f'

theorem reach_start : Reach G (initialState : Level5State α) :=
  ⟨⟨0, .start⟩, fun _ _ _ _ h => by simp [initialState] at h,
    fun _ _ _ h => by simp [initialState] at h, fun _ => rfl, rfl,
    fun _ h => by simp [initialState] at h⟩

theorem reach_step (hm : 0 < C.recordsToMove) {s s' : Level5State α} {x : FreeMonoid α}
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

theorem notClass_step (hm : 0 < C.recordsToMove) {s s' : Level5State α} {x : FreeMonoid α}
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
theorem reach_versions {s : Level5State α} (hs : Reach G s) (hc : G.InClass S s.tree) :
    ∀ v ∈ s.versions, G.InClass S v.1 ∧ EdgesInto v.1 v.2 := by
  intro v hv
  obtain ⟨f, hfS, hf⟩ := hc
  obtain ⟨hve, hvf⟩ := hs.2.2.2.2.2 v hv
  obtain ⟨f', hf'le, hf'⟩ := hvf f hf
  exact ⟨⟨f', hf'le.trans hfS, hf'⟩, hve⟩

end Reach

section Potential

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : Level5Settings)
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
def EvP (Φ : Level5State α → ℝ) (thr : ℝ) : Level5State α → List (FreeMonoid α) → Prop
  | _, [] => False
  | s, x :: xs => (G.InClass S s.tree ∧ thr ≤ Φ (recordProbe C cut s x))
    ∨ match applyProbe C cut s x with
      | .inl s' => EvP Φ thr s' xs
      | .inr _ => False

theorem evP_notClass (hm : 0 < C.recordsToMove) (Φ : Level5State α → ℝ) (thr : ℝ) :
    ∀ (l : List (FreeMonoid α)) (s : Level5State α), Reach G s → ¬ G.InClass S s.tree →
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
    (Φ : Level5State α → ℝ) {K : ℝ} (hK : 0 < K)
    (hkeep : ∀ s s' x, applyProbe C cut s x = .inl s' → Φ s' = Φ (recordProbe C cut s x))
    (hstep : ∀ s, Reach G s → G.InClass S s.tree →
      ∫⁻ x, ENNReal.ofReal (Real.exp ((Φ (recordProbe C cut s x) - Φ s) / K)) ∂D ≤ 1)
    (thr : ℝ) :
    ∀ (T : ℕ) (s : Level5State α), Reach G s →
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

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : Level5Settings)
  (D : Measure (FreeMonoid α)) (rd : FreeMonoid α → ARU)

/-- The start's undecided strings less twice the running sum of their means. -/
noncomputable def startPot (s : Level5State α) : ℝ :=
  s.startUndecided.length - 2 * (s.versions.map (siteTotal D rd C.walkStart .start)).sum

/-- An edge's undecided strings less twice the running sum of their means. -/
noncomputable def edgePot (e : List Bool × α) (s : Level5State α) : ℝ :=
  (s.edgeUndecided e.1 e.2).length
    - 2 * (s.versions.map (siteTotal D rd C.walkStart (.edge e))).sum

/-- The mean count of the strings a probe walked from `k` under `v` first reads at the edge
`e`. -/
noncomputable def readMean (k : ℕ) (e : List Bool × α) (v : DTree α × EdgeTable α) : ℝ :=
  ∫ x, (edgeReadsBy (fun z => (rd z).cut) v.1 v.2 k x e : ℝ) ∂D

/-- Half the running sum of an edge's reads' means less its reads. -/
noncomputable def readPot (e : List Bool × α) (s : Level5State α) : ℝ :=
  (s.versions.map (readMean D rd C.walkStart e)).sum / 2 - s.edgeReads e.1 e.2

theorem startPot_pre (s : Level5State α) (x : FreeMonoid α) :
    startPot C D rd (recordProbe C (fun z => (rd z).cut) s x) - startPot C D rd s
      = (startHarvestBy (fun z => (rd z).cut) s.tree C.walkStart x).length
        - 2 * siteTotal D rd C.walkStart .start (s.tree, s.edges) := by
  obtain ⟨-, -, h3, -⟩ := recordProbe_countProbe C (fun z => (rd z).cut) s x
  simp only [startPot, h3, recordProbe_versions, List.length_append, List.map_append,
    List.sum_append, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  push_cast
  ring

theorem edgePot_pre (e : List Bool × α) (s : Level5State α) (x : FreeMonoid α) :
    edgePot C D rd e (recordProbe C (fun z => (rd z).cut) s x) - edgePot C D rd e s
      = (edgeHarvestBy (fun z => (rd z).cut) s.tree s.edges C.walkStart x e).length
        - 2 * siteTotal D rd C.walkStart (.edge e) (s.tree, s.edges) := by
  obtain ⟨h1, -, -, -⟩ := recordProbe_countProbe C (fun z => (rd z).cut) s x
  simp only [edgePot, h1, recordProbe_versions, List.length_append, List.map_append,
    List.sum_append, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Prod.mk.eta]
  push_cast
  ring

theorem readPot_pre (e : List Bool × α) (s : Level5State α) (x : FreeMonoid α) :
    readPot C D rd e (recordProbe C (fun z => (rd z).cut) s x) - readPot C D rd e s
      = readMean D rd C.walkStart e (s.tree, s.edges) / 2
        - edgeReadsBy (fun z => (rd z).cut) s.tree s.edges C.walkStart x e := by
  obtain ⟨-, h2, -, -⟩ := recordProbe_countProbe C (fun z => (rd z).cut) s x
  simp only [readPot, h2, recordProbe_versions, List.map_append, List.sum_append, List.map_cons,
    List.map_nil, List.sum_cons, List.sum_nil, Prod.mk.eta]
  push_cast
  ring

theorem startPot_keep {s s' : Level5State α} {x : FreeMonoid α}
    (h : applyProbe C (fun z => (rd z).cut) s x = .inl s') :
    startPot C D rd s' = startPot C D rd (recordProbe C (fun z => (rd z).cut) s x) := by
  simp only [startPot, (applyProbe_counts C _ h).2.2.1, applyProbe_versions C _ h]

theorem edgePot_keep (e : List Bool × α) {s s' : Level5State α} {x : FreeMonoid α}
    (h : applyProbe C (fun z => (rd z).cut) s x = .inl s') :
    edgePot C D rd e s' = edgePot C D rd e (recordProbe C (fun z => (rd z).cut) s x) := by
  simp only [edgePot, (applyProbe_counts C _ h).1, applyProbe_versions C _ h]

theorem readPot_keep (e : List Bool × α) {s s' : Level5State α} {x : FreeMonoid α}
    (h : applyProbe C (fun z => (rd z).cut) s x = .inl s') :
    readPot C D rd e s' = readPot C D rd e (recordProbe C (fun z => (rd z).cut) s x) := by
  simp only [readPot, (applyProbe_counts C _ h).2.1, applyProbe_versions C _ h]

theorem startPot_step [IsProbabilityMeasure D] (s : Level5State α) :
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
    (e : List Bool × α) (s : Level5State α) :
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
    (hLmax : Fintype.card σ + S + 2 ≤ C.maxLeaves) (e : List Bool × α) (s : Level5State α)
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

section PT

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : Level5Settings)
  (cut : FreeMonoid α → Option Bool) (S : ℕ)

/-- After `i` probes, the run is at a state from which `Q` holds of the remaining draws. -/
def Skip (Q : Level5State α → List (FreeMonoid α) → Prop) : ℕ → Level5State α → List (FreeMonoid α)
    → Prop
  | 0, s, l => Q s l
  | _ + 1, _, [] => False
  | i + 1, s, x :: xs => match applyProbe C cut s x with
    | .inl s' => Skip Q i s' xs
    | .inr _ => False

open scoped Classical in
theorem recordProbe_stretch (s : Level5State α) (x : FreeMonoid α) :
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
  rcases ho : probeBy cut s.tree s.edges C.walkStart x with _ | w | w | j | ⟨ps, fd⟩ | u <;>
    try dsimp only
  · left; simp [Level5State.countProbe, Outcome.IsSearch]
  · left; simp [Level5State.countProbe, Outcome.IsSearch]
  · left; simp [Level5State.countProbe, Outcome.IsSearch]
  · left; simp [Level5State.countProbe, Outcome.IsSearch]
  · left
    rcases recordBy cut C.walkStart (s.tree, s.edges) x with _ | ⟨⟨p, c, t⟩, sp⟩ <;>
      simp [Level5State.countProbe, Level5State.addRecord, Outcome.IsSearch]
  · rcases x.toList[u.toList.length]? with _ | c <;> try dsimp only
    · left; simp [Level5State.countProbe, Outcome.IsSearch]
    rcases s.tree.siftCut cut u with p | b <;> try dsimp only
    swap
    · left; simp [Level5State.countProbe, Outcome.IsSearch]
    rcases s.tree.siftCut cut (u * FreeMonoid.of c) with t | b <;> rcases s.edges p c with _ | e <;>
      try dsimp only
    · right; simp [Level5State.setEdge, Level5State.newStretch, Outcome.IsSearch]
    all_goals left; simp [Level5State.countProbe, Outcome.IsSearch]

theorem mem_searchAt {rd : FreeMonoid α → ARU} {k : ℕ} {T : DTree α} {edges : EdgeTable α}
    {x : FreeMonoid α} :
    x ∈ searchAt rd k T edges ↔ (probeBy (fun z => (rd z).cut) T edges k x).IsSearch := by
  simp only [searchAt_eq, Set.mem_setOf_eq]
  rcases probeBy (fun z => (rd z).cut) T edges k x with _ | _ | _ | _ | _ | _ <;>
    simp [Outcome.IsSearch]

theorem ptAt_sub_searchAt (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : EdgeTable α)
    (P : σ → Prop) : G.ptAt rd k T edges P ⊆ searchAt rd k T edges := by
  rintro x ⟨b, hb, -⟩
  rw [mem_searchAt]
  unfold middleHarvestBy at hb
  rcases ho : probeBy (fun z => (rd z).cut) T edges k x with _ | _ | _ | _ | _ | _ <;>
    rw [ho] at hb <;> simp_all [Outcome.IsSearch]

/-- At a hypothesis searching on at least `φpt/2` of the probes, at most `θpt` of whose searches
stop at an undecided middle. -/
def PTLow (D : Measure (FreeMonoid α)) (rd : FreeMonoid α → ARU) (s : Level5State α) : Prop :=
  C.searchShare / 2 ≤ D.real (searchAt rd C.walkStart s.tree s.edges)
    ∧ D.real (G.ptAt rd C.walkStart s.tree s.edges fun _ => True)
      ≤ C.middleUndecidedRate * D.real (searchAt rd C.walkStart s.tree s.edges)

/-- The counts every reached state keeps in order: the start's undecided strings at most the
probes, and a stretch's middles at most its searches, at most its probes. -/
def CountInv (s : Level5State α) : Prop :=
  s.startUndecided.length ≤ s.probes ∧ s.stretchMiddles.length ≤ s.stretchDisagreements
    ∧ s.stretchDisagreements ≤ s.stretchProbes

theorem countInv_start : CountInv (initialState : Level5State α) := by
  simp [CountInv, initialState]

open scoped Classical in
theorem countInv_record {s : Level5State α} (hs : CountInv s) (x : FreeMonoid α) :
    CountInv (recordProbe C cut s x) := by
  obtain ⟨h1, h2, h3⟩ := hs
  obtain ⟨-, -, p3, p4⟩ := recordProbe_countProbe C cut s x
  refine ⟨?_, ?_⟩
  · rw [p3, p4, List.length_append]
    have := startHarvestBy_length_le cut s.tree C.walkStart x
    omega
  rcases recordProbe_stretch C cut s x with ⟨hn, hm, hd⟩ | ⟨hn, hm, hd, -⟩
  · rw [hn, hm, hd, List.length_append]
    rcases middleHarvestBy_cases cut s.tree s.edges C.walkStart x with h0 | ⟨b, hb⟩
    · rw [h0]
      split_ifs <;> simp <;> omega
    · have hsr := middleHarvestBy_search cut (by rw [hb]; simp)
      rw [hb, if_pos hsr]
      simp
      omega
  · rw [hn, hm, hd]
    simp

theorem countInv_step {s s' : Level5State α} {x : FreeMonoid α} (hs : CountInv s)
    (h : applyProbe C cut s x = .inl s') : CountInv s' := by
  have h1 := countInv_record C cut hs x
  obtain ⟨-, -, c3, c4⟩ := applyProbe_counts C cut h
  rcases applyProbe_newStretch C cut h with ⟨hn0, hd0, hm0⟩ | ⟨rfl, -, -, -⟩
  · exact ⟨by rw [c3, c4]; exact h1.1, by rw [hm0, hd0]; simp, by rw [hd0, hn0]⟩
  · exact h1

/-- From `s`, within its stretch, the weight `r` times its probes' factors reaches `1`, every state
on the way satisfying `ok`. -/
def EvL (ok : Level5State α → Prop) (f : Level5State α → FreeMonoid α → ℝ) :
    ℝ → Level5State α → List (FreeMonoid α) → Prop
  | _, _, [] => False
  | r, s, x :: xs => ok s ∧ (1 ≤ r * f s x ∨ match applyProbe C cut s x with
      | .inl s' => s'.stretchProbes ≠ 0 ∧ EvL ok f (r * f s x) s' xs
      | .inr _ => False)

/-- Ville's inequality along a stretch: where every probe's factor averages at most `1`, the
weight `r` times the factors reaches `1` with chance at most `r`. -/
theorem evL_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (ok : Level5State α → Prop) (f : Level5State α → FreeMonoid α → ℝ) (hf0 : ∀ s x, 0 ≤ f s x)
    (hf : ∀ s, ok s → ∫⁻ x, ENNReal.ofReal (f s x) ∂D ≤ 1) :
    ∀ (T : ℕ) (r : ℝ) (s : Level5State α), 0 ≤ r →
      (Measure.pi fun _ : Fin T => D) {xs | EvL C cut ok f r s (List.ofFn xs)}
        ≤ ENNReal.ofReal r := by
  intro T
  induction T with
  | zero => intro r s _; simp [EvL]
  | succ T ih =>
    intro r s hr
    by_cases hok : ok s
    swap
    · refine (measure_mono (t := ∅) fun xs hxs => ?_).trans (by simp)
      simp only [Set.mem_ofPred_eq, List.ofFn_succ, EvL] at hxs
      exact hok hxs.1
    rw [pi_succ_apply]
    have hsec : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
        {xs : Fin (T + 1) → FreeMonoid α | EvL C cut ok f r s (List.ofFn xs)}}
        ≤ ENNReal.ofReal (r * f s x) := by
      intro x
      by_cases h1 : 1 ≤ r * f s x
      · exact prob_le_one.trans (by rw [← ENNReal.ofReal_one]; exact ENNReal.ofReal_le_ofReal h1)
      rcases hst : applyProbe C cut s x with s' | _
      · refine le_trans (measure_mono fun xs hxs => ?_)
          (ih (r * f s x) s' (mul_nonneg hr (hf0 s x)))
        simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvL, hst] at hxs
        rcases hxs.2 with h | h
        · exact absurd h h1
        · exact h.2
      · refine (measure_mono (t := ∅) fun xs hxs => ?_).trans (by simp)
        simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvL, hst] at hxs
        rcases hxs.2 with h | h
        · exact h1 h
        · exact h
    refine (lintegral_mono hsec).trans ?_
    have : ∀ x, ENNReal.ofReal (r * f s x) = ENNReal.ofReal r * ENNReal.ofReal (f s x) :=
      fun x => ENNReal.ofReal_mul hr
    simp_rw [this]
    rw [lintegral_const_mul _ (measurable_of_countable _)]
    calc ENNReal.ofReal r * ∫⁻ x, ENNReal.ofReal (f s x) ∂D ≤ ENNReal.ofReal r * 1 := by
          gcongr; exact hf s hok
      _ = _ := mul_one _

open scoped Classical in
/-- A probe's factor in its stretch's middles test: a search stopping at an undecided middle
`p₁/p₀`, any other search `(1 - p₁)/(1 - p₀)`, and a probe that does not search `1`. -/
noncomputable def ptFactor (rd : FreeMonoid α → ARU) (s : Level5State α) (x : FreeMonoid α) : ℝ :=
  if x ∈ searchAt rd C.walkStart s.tree s.edges then
    ratioFactor C.middleUndecidedRate C.middleAltRate
      (decide (x ∈ G.ptAt rd C.walkStart s.tree s.edges fun _ => True))
  else 1

theorem ptFactor_nonneg (rd : FreeMonoid α → ARU) (hS : C.SeqOK) (s : Level5State α)
    (x : FreeMonoid α) : 0 ≤ ptFactor G C rd s x := by
  have := hS.θpt_pos
  have := hS.θpt_alt
  have := hS.θpt_alt_lt
  unfold ptFactor ratioFactor
  split_ifs
  · exact div_nonneg (by linarith) (by linarith)
  · exact div_nonneg (by linarith) (by linarith)
  · norm_num

open scoped Classical in
/-- Where at most `θpt` of the searches stop at an undecided middle, the middles test's factor
averages at most `1`. -/
theorem lintegral_ptFactor_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (rd : FreeMonoid α → ARU) (hS : C.SeqOK) (s : Level5State α)
    (hs : D.real (G.ptAt rd C.walkStart s.tree s.edges fun _ => True)
      ≤ C.middleUndecidedRate * D.real (searchAt rd C.walkStart s.tree s.edges)) :
    ∫⁻ x, ENNReal.ofReal (ptFactor G C rd s x) ∂D ≤ 1 := by
  set p₀ := C.middleUndecidedRate
  set p₁ := C.middleAltRate
  have hp₀ : 0 < p₀ := hS.θpt_pos
  have hp₀₁ : p₀ < p₁ := hS.θpt_alt
  have hp₁1 : p₁ < 1 := hS.θpt_alt_lt
  set A := G.ptAt rd C.walkStart s.tree s.edges fun _ => True
  set B := searchAt rd C.walkStart s.tree s.edges
  have hAB : A ⊆ B := ptAt_sub_searchAt G rd C.walkStart s.tree s.edges _
  set a₁ : ℝ := p₁ / p₀
  set a₂ : ℝ := (1 - p₁) / (1 - p₀)
  have ha₁ : 0 ≤ a₁ := div_nonneg (by linarith) hp₀.le
  have ha₂ : 0 ≤ a₂ := div_nonneg (by linarith) (by linarith)
  have hpt : ∀ x, ENNReal.ofReal (ptFactor G C rd s x)
      = A.indicator (fun _ => ENNReal.ofReal a₁) x + (B \ A).indicator (fun _ => ENNReal.ofReal a₂) x
        + Bᶜ.indicator (fun _ => 1) x := by
    intro x
    unfold ptFactor ratioFactor
    by_cases hxA : x ∈ A
    · have hxB := hAB hxA
      rw [if_pos hxB, if_pos (by simpa [A] using hxA), Set.indicator_of_mem hxA,
        Set.indicator_of_notMem (fun h => h.2 hxA), Set.indicator_of_notMem (by simp [hxB])]
      simp [a₁] <;> rfl
    by_cases hxB : x ∈ B
    · rw [if_pos hxB, if_neg (by simpa [A] using hxA), Set.indicator_of_notMem hxA,
        Set.indicator_of_mem (Set.mem_sdiff_of_mem hxB hxA), Set.indicator_of_notMem (by simp [hxB])]
      simp [a₂] <;> rfl
    · rw [if_neg hxB, Set.indicator_of_notMem hxA, Set.indicator_of_notMem (fun h => hxB h.1),
        Set.indicator_of_mem hxB]
      simp
  simp_rw [hpt]
  have hmeas : ∀ E : Set (FreeMonoid α), MeasurableSet E := fun E =>
    (Set.to_countable _).measurableSet
  rw [lintegral_add_left (measurable_of_countable _), lintegral_add_left
      (measurable_of_countable _), lintegral_indicator (hmeas _), lintegral_indicator (hmeas _),
    lintegral_indicator (hmeas _), setLIntegral_const, setLIntegral_const, setLIntegral_const,
    ← ofReal_measureReal, ← ofReal_measureReal, ← ofReal_measureReal,
    measureReal_compl (hmeas _), probReal_univ, measureReal_sdiff hAB (hmeas _),
    ← ENNReal.ofReal_mul ha₁, ← ENNReal.ofReal_mul ha₂, one_mul]
  have hA0 : 0 ≤ D.real A := measureReal_nonneg
  have hAB' : D.real A ≤ D.real B := measureReal_mono hAB
  have hB1 : D.real B ≤ 1 := measureReal_le_one
  rw [← ENNReal.ofReal_add (by positivity) (mul_nonneg ha₂ (by linarith)),
    ← ENNReal.ofReal_add (by positivity) (by linarith), ← ENNReal.ofReal_one]
  refine ENNReal.ofReal_le_ofReal ?_
  have e1 : p₀ * a₁ = p₁ := by simp only [a₁]; field_simp
  have e2 : (1 - p₀) * a₂ = 1 - p₁ := by
    simp only [a₂]; field_simp [show (1 - p₀) ≠ 0 by linarith]
  have hgap : a₂ ≤ a₁ := by
    simp only [a₁, a₂]
    rw [div_le_div_iff₀ (by linarith) hp₀]
    nlinarith
  -- linear in `D A`, increasing, so at most its value at `p₀ D B`
  have h1 : D.real A * a₁ + (D.real B - D.real A) * a₂
      ≤ p₀ * D.real B * a₁ + (D.real B - p₀ * D.real B) * a₂ := by
    nlinarith [mul_le_mul_of_nonneg_right hs (sub_nonneg.2 hgap)]
  have h2 : p₀ * D.real B * a₁ + (D.real B - p₀ * D.real B) * a₂ = D.real B := by
    have : p₀ * D.real B * a₁ + (D.real B - p₀ * D.real B) * a₂
        = D.real B * (p₀ * a₁) + D.real B * ((1 - p₀) * a₂) := by ring
    rw [this, e1, e2]; ring
  linarith

theorem skip_le (hm : 0 < C.recordsToMove) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (Q : Level5State α → List (FreeMonoid α) → Prop) {c : ENNReal}
    (hQ : ∀ (T : ℕ) (s : Level5State α), Reach G s
      → (Measure.pi fun _ : Fin T => D) {xs | Q s (List.ofFn xs)} ≤ c) :
    ∀ (T i : ℕ) (s : Level5State α), Reach G s →
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

variable (C : Level5Settings) (D : Measure (FreeMonoid α)) (rd : FreeMonoid α → ARU)

/-- The hypothesis's disagreement rate: the probes that search. -/
noncomputable def disRate (s : Level5State α) : ℝ := D.real (searchAt rd C.walkStart s.tree s.edges)

open scoped Classical in
/-- A probe's factor in a test of its stretch's searches: `p₁/p₀` where it searches, else
`(1 - p₁)/(1 - p₀)`. -/
noncomputable def searchFactor (p₀ p₁ : ℝ) (s : Level5State α) (x : FreeMonoid α) : ℝ :=
  ratioFactor p₀ p₁ (decide (x ∈ searchAt rd C.walkStart s.tree s.edges))

theorem searchFactor_nonneg {p₀ p₁ : ℝ} (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : 0 ≤ p₁)
    (hp₁1 : p₁ ≤ 1) (s : Level5State α) (x : FreeMonoid α) : 0 ≤ searchFactor C rd p₀ p₁ s x := by
  unfold searchFactor ratioFactor
  split_ifs
  · positivity
  · exact div_nonneg (by linarith) (by linarith)

open scoped Classical in
/-- On the far side of `p₀` from `p₁`, the factor averages at most `1`. -/
theorem lintegral_searchFactor_le [IsProbabilityMeasure D] {p₀ p₁ : ℝ} (hp₀ : 0 < p₀)
    (hp₀1 : p₀ < 1) (hp₁ : 0 ≤ p₁) (hp₁1 : p₁ ≤ 1) {s : Level5State α}
    (hq : (disRate C D rd s - p₀) * (p₁ - p₀) ≤ 0) :
    ∫⁻ x, ENNReal.ofReal (searchFactor C rd p₀ p₁ s x) ∂D ≤ 1 :=
  lintegral_ratioFactor_le D (fun x => x ∈ searchAt rd C.walkStart s.tree s.edges) hp₀ hp₀1 hp₁
    hp₁1 hq

open scoped Classical in
/-- A probe that carries the stretch on multiplies a test of its searches' ratio by its factor. -/
theorem lr_search_step (p₀ p₁ : ℝ) {s : Level5State α} {x : FreeMonoid α} (hci : CountInv s)
    (hn : (recordProbe C (fun z => (rd z).cut) s x).stretchProbes = s.stretchProbes + 1) :
    likelihoodRatio p₀ p₁ (recordProbe C (fun z => (rd z).cut) s x).stretchProbes
        (recordProbe C (fun z => (rd z).cut) s x).stretchDisagreements
      = searchFactor C rd p₀ p₁ s x * likelihoodRatio p₀ p₁ s.stretchProbes
        s.stretchDisagreements := by
  rcases recordProbe_stretch C (fun z => (rd z).cut) s x with ⟨hn₁, -, hd⟩ | ⟨hn', -, -, -⟩
  swap
  · omega
  unfold searchFactor
  rw [hn₁, hd]
  have e : (if (probeBy (fun z => (rd z).cut) s.tree s.edges C.walkStart x).IsSearch then 1 else 0)
      = if decide (x ∈ searchAt rd C.walkStart s.tree s.edges) then 1 else 0 := by
    by_cases hx : x ∈ searchAt rd C.walkStart s.tree s.edges
    · rw [if_pos (mem_searchAt.1 hx)]; simp [hx]
    · rw [if_neg (fun h => hx (mem_searchAt.2 h))]; simp [hx]
  rw [e]
  exact likelihoodRatio_succ hci.2.2 _

end Low

section Assemble

open scoped Classical in
theorem runTests_start' (C : Level5Settings) {s : Level5State α}
    (h : runTests C s = some .harvestStart) :
    seqRate C.startUndecidedRate C.startAltRate C.testLevel s.probes s.startUndecided.length := by
  unfold runTests at h
  by_contra h1
  rw [if_neg h1] at h
  split_ifs at h <;> simp at h

open scoped Classical in
theorem runTests_harvest' (C : Level5Settings) {s : Level5State α} {e : List Bool × α}
    (h : runTests C s = some (.harvest e)) :
    e.1 ∈ s.tree.leaves ∧ C.edgeTraffic * s.probes ≤ s.edgeReads e.1 e.2
        ∧ C.edgeExcess (s.edgeReads e.1 e.2)
      ≤ ((s.edgeUndecided e.1 e.2).length : ℝ) - C.edgeUndecidedRate * s.edgeReads e.1 e.2 := by
  unfold runTests at h
  by_cases h1 : seqRate C.startUndecidedRate C.startAltRate C.testLevel s.probes
      s.startUndecided.length
  · rw [if_pos h1] at h; simp at h
  · rw [if_neg h1] at h
    by_cases h2 : ∃ e : List Bool × α, e.1 ∈ s.tree.leaves
        ∧ C.edgeTraffic * s.probes ≤ s.edgeReads e.1 e.2
        ∧ C.edgeExcess (s.edgeReads e.1 e.2) ≤ ((s.edgeUndecided e.1 e.2).length : ℝ)
            - C.edgeUndecidedRate * s.edgeReads e.1 e.2
    · rw [dif_pos h2] at h
      simp only [Option.some.injEq, Level5Ending.harvest.injEq] at h
      rw [← h]
      exact h2.choose_spec
    · rw [dif_neg h2] at h
      split_ifs at h <;> simp at h

open scoped Classical in
theorem runTests_pt' (C : Level5Settings) {s : Level5State α} (h : runTests C s = some .harvestPT) :
    C.searchShare * s.stretchProbes ≤ s.stretchDisagreements ∧ C.firstLook ≤ s.stretchDisagreements
        ∧ seqRate C.middleUndecidedRate C.middleAltRate C.testLevel s.stretchDisagreements
          s.stretchMiddles.length := by
  unfold runTests at h
  by_cases h1 : seqRate C.startUndecidedRate C.startAltRate C.testLevel s.probes
      s.startUndecided.length
  · rw [if_pos h1] at h; simp at h
  rw [if_neg h1] at h
  by_cases h2 : ∃ e : List Bool × α, e.1 ∈ s.tree.leaves
      ∧ C.edgeTraffic * s.probes ≤ s.edgeReads e.1 e.2
      ∧ C.edgeExcess (s.edgeReads e.1 e.2) ≤ ((s.edgeUndecided e.1 e.2).length : ℝ)
          - C.edgeUndecidedRate * s.edgeReads e.1 e.2
  · rw [dif_pos h2] at h; simp at h
  rw [dif_neg h2] at h
  by_cases h3 : C.searchShare * s.stretchProbes ≤ s.stretchDisagreements
      ∧ C.firstLook ≤ s.stretchDisagreements ∧ seqRate C.middleUndecidedRate C.middleAltRate
        C.testLevel s.stretchDisagreements s.stretchMiddles.length
  · exact h3
  · rw [if_neg h3] at h
    split_ifs at h <;> simp at h

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
  intro α _ _ σ _ G rd D _ C S L T ρ θg θgs θgpt hlen hL1 hm hS hθg0 hθg hφe0 hLmax hexc
    hθgs0 hθgs hhalf hθgpt0 hθgpt hgate hE
  classical
  have ha := hS.a_pos
  have ha1 := hS.a_lt
  set ℓ := Real.log (1 / C.testLevel) with hℓ
  have hℓ0 : 0 ≤ ℓ := Real.log_nonneg (by rw [le_div_iff₀ ha]; linarith)
  have hθpt0 : 0 ≤ C.middleUndecidedRate := hS.θpt_pos.le
  have hθs0 : 0 ≤ C.startUndecidedRate := hS.θs_pos.le
  have hθe0 : 0 ≤ C.edgeUndecidedRate := by linarith
  have hφ0 := hS.φ_pos
  have hφ1 := hS.φ_lt
  have hLm : (0 : ℝ) < C.maxLeaves := by exact_mod_cast (by omega : 0 < C.maxLeaves)
  have hK₁ : (0 : ℝ) < 2 * (L + 1) := by positivity
  have hK₂ : (0 : ℝ) < 2 * ((L + 1) * C.maxLeaves) := by positivity
  set keys := shortPaths C.maxLeaves ×ˢ (Finset.univ : Finset α)
  set okP : Level5State α → Prop := PTLow G C D rd
  set fP : Level5State α → FreeMonoid α → ℝ := ptFactor G C rd
  set LP : Level5State α → ℝ := fun s => likelihoodRatio C.middleUndecidedRate C.middleAltRate
    s.stretchDisagreements s.stretchMiddles.length
  set okH : Level5State α → Prop := fun s => disRate C D rd s ≤ C.searchShare / 2
  set fH : Level5State α → FreeMonoid α → ℝ := searchFactor C rd (C.searchShare / 2) C.searchShare
  set LH : Level5State α → ℝ := fun s => likelihoodRatio (C.searchShare / 2) C.searchShare
    s.stretchProbes s.stretchDisagreements
  set Qp : Level5State α → List (FreeMonoid α) → Prop := fun s l =>
    s.stretchDisagreements = 0 ∧ s.stretchMiddles = [] ∧ EvL C (fun z => (rd z).cut) okP fP C.testLevel s l
  set Qh : Level5State α → List (FreeMonoid α) → Prop := fun s l =>
    s.stretchProbes = 0 ∧ s.stretchDisagreements = 0 ∧ EvL C (fun z => (rd z).cut) okH fH C.testLevel s l
  set P : Level5Ending α → Level5State α → Prop := fun e s' => G.InClass S s'.tree ∧ e ≠ .tooBig
    ∧ ¬ GoodEnd G D rd C.walkStart C.harvestYield e s'
  have hokP : ∀ s s' : Level5State α, s'.tree = s.tree → s'.edges = s.edges →
      (okP s' ↔ okP s) := by
    intro s s' h1 h2; simp only [okP, PTLow, h1, h2]
  have hdr : ∀ s s' : Level5State α, s'.tree = s.tree → s'.edges = s.edges →
      disRate C D rd s' = disRate C D rd s := by
    intro s s' h1 h2; simp only [disRate, h1, h2]
  -- a probe that carries the stretch on multiplies each test's ratio by its factor
  have hstepP : ∀ s x, CountInv s →
      (recordProbe C (fun z => (rd z).cut) s x).stretchProbes = s.stretchProbes + 1 →
      LP (recordProbe C (fun z => (rd z).cut) s x) = fP s x * LP s := by
    intro s x hci hn
    rcases recordProbe_stretch C (fun z => (rd z).cut) s x with ⟨-, hm, hd⟩ | ⟨hn', -, -, -⟩
    swap
    · omega
    simp only [LP, fP, ptFactor]
    rw [hm, hd, List.length_append]
    by_cases hx : x ∈ searchAt rd C.walkStart s.tree s.edges
    · have hxs := mem_searchAt.1 hx
      rw [if_pos hxs, if_pos hx]
      have hlen : (middleHarvestBy (fun z => (rd z).cut) s.tree s.edges C.walkStart x).length
          = if decide (x ∈ G.ptAt rd C.walkStart s.tree s.edges fun _ => True) then 1 else 0 := by
        rcases middleHarvestBy_cases (fun z => (rd z).cut) s.tree s.edges C.walkStart x with h0 | ⟨b, hb⟩
        · have : x ∉ G.ptAt rd C.walkStart s.tree s.edges fun _ => True := by
            rintro ⟨b, hb, -⟩
            rw [h0] at hb
            simp at hb
          simp [h0, this]
        · have : x ∈ G.ptAt rd C.walkStart s.tree s.edges fun _ => True := ⟨b, hb, trivial⟩
          simp [hb, this]
      rw [hlen]
      exact likelihoodRatio_succ hci.2.1 _
    · have hxs : ¬ (probeBy (fun z => (rd z).cut) s.tree s.edges C.walkStart x).IsSearch :=
        fun h => hx (mem_searchAt.2 h)
      rw [if_neg hxs, if_neg hx, one_mul]
      have h0 : middleHarvestBy (fun z => (rd z).cut) s.tree s.edges C.walkStart x = [] := by
        by_contra hne
        exact hxs (middleHarvestBy_search (fun z => (rd z).cut) hne)
      simp [h0]
  have hstepH : ∀ s x, CountInv s →
      (recordProbe C (fun z => (rd z).cut) s x).stretchProbes = s.stretchProbes + 1 →
      LH (recordProbe C (fun z => (rd z).cut) s x) = fH s x * LH s := by
    intro s x hci hn
    rcases recordProbe_stretch C (fun z => (rd z).cut) s x with ⟨hn₁, -, hd⟩ | ⟨hn', -, -, -⟩
    swap
    · omega
    simp only [LH, fH, searchFactor]
    rw [hn₁, hd]
    have e : (if (probeBy (fun z => (rd z).cut) s.tree s.edges C.walkStart x).IsSearch then 1 else 0)
        = if decide (x ∈ searchAt rd C.walkStart s.tree s.edges) then 1 else 0 := by
      by_cases hx : x ∈ searchAt rd C.walkStart s.tree s.edges
      · rw [if_pos (mem_searchAt.1 hx)]; simp [hx]
      · rw [if_neg (fun h => hx (mem_searchAt.2 h))]; simp [hx]
    rw [e]
    exact likelihoodRatio_succ hci.2.2 _
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
  have hstartFin : ∀ s x, Reach G s → CountInv s → G.InClass S s.tree →
      runTests C (recordProbe C (fun z => (rd z).cut) s x) = some .harvestStart →
      startPot C D rd (recordProbe C (fun z => (rd z).cut) s x) < 2 * ℓ →
      GoodEnd G D rd C.walkStart C.harvestYield .harvestStart
        (recordProbe C (fun z => (rd z).cut) s x) := by
    intro s x hs hci hc hrt hpot
    obtain ⟨hcv, hN⟩ := hvers s x hs hc
    have hlt := runTests_start' C hrt
    have hle := (countInv_record C (fun z => (rd z).cut) hci x).1
    have hh := hhalf _ _ hle hlt
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
    have hh' : N * C.startUndecidedRate / 4 + 2 * ℓ
        ≤ ((recordProbe C (fun z => (rd z).cut) s x).startUndecided.length : ℝ) := by
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
      (recordProbe C (fun z => (rd z).cut) s x).stretchProbes ≠ 0 → ¬ okP s →
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
  have hincl : ∀ (l : List (FreeMonoid α)) (s : Level5State α), Reach G s → CountInv s →
      RunEnds (applyProbe C fun z => (rd z).cut) P s l →
      (∃ e ∈ keys, EvP G C (fun z => (rd z).cut) S (edgePot C D rd e) (2 * (L + 1) * ℓ) s l)
        ∨ (∃ e ∈ keys, EvP G C (fun z => (rd z).cut) S (readPot C D rd e)
            (2 * ((L + 1) * C.maxLeaves) * ℓ) s l)
        ∨ EvP G C (fun z => (rd z).cut) S (startPot C D rd) (2 * ℓ) s l
        ∨ (∃ i, i < l.length ∧ Skip C (fun z => (rd z).cut) Qp i s l)
        ∨ EvL C (fun z => (rd z).cut) okP fP (C.testLevel * LP s) s l
        ∨ (∃ i, i < l.length ∧ Skip C (fun z => (rd z).cut) Qh i s l)
        ∨ EvL C (fun z => (rd z).cut) okH fH (C.testLevel * LH s) s l := by
    intro l
    induction l with
    | nil => intro s _ _ h; exact h.elim
    | cons x xs ih =>
      intro s hs hci h
      simp only [RunEnds] at h
      rcases hst : applyProbe C (fun z => (rd z).cut) s x with s' | ⟨e, s''⟩ <;> rw [hst] at h
      · rcases ih s' (reach_step G C _ hm hs hst).1 (countInv_step C _ hci hst) h with
          ⟨e, he, hE'⟩ | ⟨e, he, hE'⟩ | hS' | ⟨i, hi, hk⟩ | hP | ⟨i, hi, hk⟩ | hH
        · exact .inl ⟨e, he, .inr (by simp only [hst]; exact hE')⟩
        · exact .inr (.inl ⟨e, he, .inr (by simp only [hst]; exact hE')⟩)
        · exact .inr (.inr (.inl (.inr (by simp only [hst]; exact hS'))))
        · refine .inr (.inr (.inr (.inl ⟨i + 1, by simp; omega, ?_⟩)))
          simp only [Skip, hst]
          exact hk
        · have hne : xs ≠ [] := by rintro rfl; exact hP
          rcases applyProbe_newStretch C _ hst with ⟨hn0, hd0, hpt0⟩ | ⟨hpre, hn, htr', hed'⟩
          · refine .inr (.inr (.inr (.inl ⟨1, by
              rcases xs with _ | _
              · exact absurd rfl hne
              · simp, ?_⟩)))
            simp only [Skip, hst, Qp]
            refine ⟨hd0, hpt0, ?_⟩
            simpa [LP, hd0, hpt0, likelihoodRatio_zero] using hP
          · subst hpre
            have hok : okP s := by
              rcases xs with _ | ⟨y, ys⟩
              · exact hP.elim
              · exact (hokP s _ htr' hed').1 hP.1
            refine .inr (.inr (.inr (.inr (.inl ?_))))
            simp only [EvL]
            refine ⟨hok, .inr ?_⟩
            rw [hst]
            refine ⟨by omega, ?_⟩
            rw [show C.testLevel * LP s * fP s x = C.testLevel * LP (recordProbe C (fun z => (rd z).cut) s x) by
              rw [hstepP s x hci hn]; ring]
            exact hP
        · refine .inr (.inr (.inr (.inr (.inr (.inl ⟨i + 1, by simp; omega, ?_⟩)))))
          simp only [Skip, hst]
          exact hk
        · have hne : xs ≠ [] := by rintro rfl; exact hH
          rcases applyProbe_newStretch C _ hst with ⟨hn0, hd0, -⟩ | ⟨hpre, hn, htr', hed'⟩
          · refine .inr (.inr (.inr (.inr (.inr (.inl ⟨1, by
              rcases xs with _ | _
              · exact absurd rfl hne
              · simp, ?_⟩)))))
            simp only [Skip, hst, Qh]
            refine ⟨hn0, hd0, ?_⟩
            simpa [LH, hn0, hd0, likelihoodRatio_zero] using hH
          · subst hpre
            have hok : okH s := by
              rcases xs with _ | ⟨y, ys⟩
              · exact hH.elim
              · have := hH.1
                simp only [okH] at this ⊢
                rwa [hdr s _ htr' hed'] at this
            refine .inr (.inr (.inr (.inr (.inr (.inr ?_)))))
            simp only [EvL]
            refine ⟨hok, .inr ?_⟩
            rw [hst]
            refine ⟨by omega, ?_⟩
            rw [show C.testLevel * LH s * fH s x = C.testLevel * LH (recordProbe C (fun z => (rd z).cut) s x) by
              rw [hstepH s x hci hn]; ring]
            exact hH
      · obtain ⟨hcl, hne, hng⟩ := h
        obtain ⟨rfl, hl⟩ := applyProbe_inr_pre C _ hst hne
        have htr : (recordProbe C (fun z => (rd z).cut) s x).tree = s.tree :=
          (recordProbe_into _ C x hs.2.1 hs.2.2.1).1
        rw [htr] at hcl
        rcases e with _ | _ | e | _ | _
        · exact absurd trivial hng
        · by_cases hpot : 2 * ℓ ≤ startPot C D rd (recordProbe C (fun z => (rd z).cut) s x)
          · exact .inr (.inr (.inl (.inl ⟨hcl, hpot⟩)))
          · exact absurd (hstartFin s x hs hci hcl hl (not_le.1 hpot)) hng
        · obtain ⟨hp, -, -⟩ := runTests_harvest' C hl
          rw [htr] at hp
          have he : e ∈ keys := by
            refine Finset.mem_product.2 ⟨mem_shortPaths ?_, Finset.mem_univ _⟩
            have := DTree.length_lt_of_mem_leaves _ _ hp
            have := class_paths G C S hcl hLmax
            omega
          by_cases h1 : 2 * (L + 1) * ℓ ≤ edgePot C D rd e
              (recordProbe C (fun z => (rd z).cut) s x)
          · exact .inl ⟨e, he, .inl ⟨hcl, h1⟩⟩
          by_cases h2 : 2 * ((L + 1) * C.maxLeaves) * ℓ ≤ readPot C D rd e
              (recordProbe C (fun z => (rd z).cut) s x)
          · exact .inr (.inl ⟨e, he, .inl ⟨hcl, h2⟩⟩)
          exact absurd (hedgeFin s x e hs hcl hl (not_le.1 h1) (not_le.1 h2)) hng
        · obtain ⟨hgate', hn₀, hlt⟩ := runTests_pt' C hl
          rcases recordProbe_stretch C _ s x with ⟨hn, hpt, hdis⟩ | ⟨hn, hpt0, hd0, -⟩
          · by_cases hk : okP s
            · refine .inr (.inr (.inr (.inr (.inl ?_))))
              simp only [EvL]
              refine ⟨hk, .inl ?_⟩
              have := hstepP s x hci hn
              have hlt' : 1 / C.testLevel ≤ LP (recordProbe C (fun z => (rd z).cut) s x) := hlt
              rw [this, div_le_iff₀ ha] at hlt'
              linarith
            · by_cases hq : C.searchShare / 2 ≤ D.real (searchAt rd C.walkStart s.tree s.edges)
              · exact absurd (hptFin s x hs hcl (by omega) hk hq) hng
              refine .inr (.inr (.inr (.inr (.inr (.inr ?_)))))
              simp only [EvL]
              refine ⟨le_of_lt (not_le.1 hq), .inl ?_⟩
              have hci₁ := countInv_record C (fun z => (rd z).cut) hci x
              have hfire := seqRate_share (a := C.testLevel) hφ0 hφ1 ha hgate hn₀ hci₁.2.2 hgate'
              have := hstepH s x hci hn
              have hlt' : 1 / C.testLevel ≤ LH (recordProbe C (fun z => (rd z).cut) s x) := hfire
              rw [this, div_le_iff₀ ha] at hlt'
              linarith
          · rw [hd0, hpt0, List.length_nil] at hlt
            exact absurd hlt (not_seqRate_zero ha ha1)
        · exact absurd rfl hne
  have hsub : {xs : Fin T → FreeMonoid α | RunEnds (applyProbe C fun z => (rd z).cut) P initialState
      (List.ofFn xs)}
      ⊆ ((((⋃ e ∈ keys, {xs | EvP G C (fun z => (rd z).cut) S (edgePot C D rd e)
            (2 * (L + 1) * ℓ) initialState (List.ofFn xs)})
          ∪ ⋃ e ∈ keys, {xs | EvP G C (fun z => (rd z).cut) S (readPot C D rd e)
            (2 * ((L + 1) * C.maxLeaves) * ℓ) initialState (List.ofFn xs)})
          ∪ {xs | EvP G C (fun z => (rd z).cut) S (startPot C D rd) (2 * ℓ) initialState
            (List.ofFn xs)})
        ∪ ⋃ i ∈ Finset.range T, {xs | Skip C (fun z => (rd z).cut) Qp i initialState (List.ofFn xs)})
        ∪ ⋃ i ∈ Finset.range T, {xs | Skip C (fun z => (rd z).cut) Qh i initialState (List.ofFn xs)} := by
    intro xs hxs
    have hT0 : ∀ l : List (FreeMonoid α), l = List.ofFn xs → l ≠ [] → 0 < T := by
      intro l hl hne
      rcases T with _ | T
      · simp at hl; exact absurd hl hne
      · omega
    rcases hincl _ _ (reach_start G) countInv_start hxs with ⟨e, he, h⟩ | ⟨e, he, h⟩ | h |
      ⟨i, hi, h⟩ | h | ⟨i, hi, h⟩ | h
    · exact .inl (.inl (.inl (.inl (Set.mem_biUnion he h))))
    · exact .inl (.inl (.inl (.inr (Set.mem_biUnion he h))))
    · exact .inl (.inl (.inr h))
    · simp only [List.length_ofFn] at hi
      exact .inl (.inr (Set.mem_biUnion (Finset.mem_range.2 hi) h))
    · have hne : List.ofFn xs ≠ [] := by intro h0; rw [h0] at h; exact h
      refine .inl (.inr (Set.mem_biUnion (x := 0) (Finset.mem_range.2 (hT0 _ rfl hne)) ?_))
      simp only [Set.mem_ofPred_eq, Skip, Qp]
      refine ⟨rfl, rfl, ?_⟩
      simpa [LP, initialState, likelihoodRatio_zero] using h
    · simp only [List.length_ofFn] at hi
      exact .inr (Set.mem_biUnion (Finset.mem_range.2 hi) h)
    · have hne : List.ofFn xs ≠ [] := by intro h0; rw [h0] at h; exact h
      refine .inr (Set.mem_biUnion (x := 0) (Finset.mem_range.2 (hT0 _ rfl hne)) ?_)
      simp only [Set.mem_ofPred_eq, Skip, Qh]
      refine ⟨rfl, rfl, ?_⟩
      simpa [LH, initialState, likelihoodRatio_zero] using h
  have hpot0 : ∀ (Φ : Level5State α → ℝ) (K : ℝ), 0 < K → Φ initialState = 0 →
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
  have hfP0 : ∀ s x, 0 ≤ fP s x := fun s x => ptFactor_nonneg G C rd hS s x
  have hfH0 : ∀ s x, 0 ≤ fH s x := fun s x =>
    searchFactor_nonneg C rd (by linarith) (by linarith) hφ0.le hφ1.le s x
  have h3 : ∀ i ∈ Finset.range T, (Measure.pi fun _ : Fin T => D)
      {xs | Skip C (fun z => (rd z).cut) Qp i initialState (List.ofFn xs)} ≤ ENNReal.ofReal C.testLevel := by
    intro i _
    refine skip_le G C _ hm D Qp (fun T' s _ => ?_) T i initialState (reach_start G)
    refine (measure_mono fun xs hxs => hxs.2.2).trans ?_
    exact evL_le C (fun z => (rd z).cut) D okP fP hfP0 (fun s hs => lintegral_ptFactor_le G C D rd hS s hs.2) T'
      C.testLevel s ha.le
  have h4 : ∀ i ∈ Finset.range T, (Measure.pi fun _ : Fin T => D)
      {xs | Skip C (fun z => (rd z).cut) Qh i initialState (List.ofFn xs)} ≤ ENNReal.ofReal C.testLevel := by
    intro i _
    refine skip_le G C _ hm D Qh (fun T' s _ => ?_) T i initialState (reach_start G)
    refine (measure_mono fun xs hxs => hxs.2.2).trans ?_
    exact evL_le C (fun z => (rd z).cut) D okH fH hfH0 (fun s hs => lintegral_searchFactor_le C D rd (by linarith)
      (by linarith) hφ0.le hφ1.le (by
        have : 0 ≤ C.searchShare - C.searchShare / 2 := by linarith
        nlinarith [hs])) T' C.testLevel s ha.le
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
        + ∑ _i ∈ Finset.range T, ENNReal.ofReal C.testLevel
        + ∑ _i ∈ Finset.range T, ENNReal.ofReal C.testLevel :=
        add_le_add (add_le_add (add_le_add (add_le_add
          ((measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum h1))
          ((measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum h2))) h2')
          ((measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum h3)))
          ((measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum h4))
    _ = (2 * keys.card + 1 + 2 * T) * ENNReal.ofReal C.testLevel := by
        simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        ring
    _ ≤ ENNReal.ofReal ((2 ^ (C.maxLeaves + 2) * Fintype.card α + 1 + 2 * T)
        * C.testLevel) := by
        rw [ENNReal.ofReal_mul (by positivity)]
        gcongr
        rw [show ((2 : ℝ) ^ (C.maxLeaves + 2) * Fintype.card α + 1 + 2 * T)
            = ((2 ^ (C.maxLeaves + 2)
          * Fintype.card α + 1 + 2 * T : ℕ) : ℝ) by push_cast; ring,
          ENNReal.ofReal_natCast]
        have : keys.card ≤ 2 ^ (C.maxLeaves + 1) * Fintype.card α := by
          simp only [keys, Finset.card_product, Finset.card_univ]
          exact Nat.mul_le_mul_right _ (shortPaths_card _)
        have h2 : 2 ^ (C.maxLeaves + 2) = 2 * 2 ^ (C.maxLeaves + 1) := by ring
        exact_mod_cast (by rw [h2]; nlinarith : 2 * keys.card + 1 + 2 * T
          ≤ 2 ^ (C.maxLeaves + 2) * Fintype.card α + 1 + 2 * T)

end Assemble

end OrthoDFA
