import OrthoDFA.Proofs.TallySub

/-!
# A harvest's strings are mostly at read-states that are not good

Each place's counts only grow, by the probe's own charge. At an edge, the good read-states'
undecided reads less `2θg` of its reads, scaled by `λ = 1 / (2L (1 + 4θg Lmax))`, exponentiate to
a supermartingale while the tree is in the class, since `goodEdge` bounds their mean by `θg` of
the reads; so they pass `log(1/a)/λ` with chance at most `a`. A harvest that is not mostly bad
has them past half its excess. At the start, the good read-states' undecided reads by the `t`-th
probe are at most binomial, and a firing start test that is not mostly bad has them at half its
count. The class's trees only gain splits that are not genuine, so a run that ends in the class
was in it throughout.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Counts

variable (C : TallyCfg) (cut : FreeMonoid α → Option Bool)

/-- A probe's outcome adds its own charge to each place's counts. -/
theorem tallyPre_charge (s : TState α) (x : FreeMonoid α) :
    (tallyPre C cut s x).harv =
      (fun p c => s.harv p c ++ edgeHarvBy cut s.tree s.edges C.k x (p, c))
      ∧ (tallyPre C cut s x).trav
        = (fun p c => s.trav p c + travBy cut s.tree s.edges C.k x (p, c))
      ∧ (tallyPre C cut s x).startH = s.startH ++ startHarvBy cut s.tree C.k x
      ∧ (tallyPre C cut s x).probes = s.probes + 1 := by
  unfold tallyPre
  split
  · split
    · split <;> simp [TState.charge, TState.setEdge, TState.fresh]
    · simp [TState.charge]
  · split <;> simp [TState.charge, TState.addRec]
  all_goals simp [TState.charge]

theorem settleOne_counts {m Lmax : ℕ} {s s' : TState α} (h : settleOne cut m Lmax s = .inl s') :
    s'.harv = s.harv ∧ s'.trav = s.trav ∧ s'.startH = s.startH ∧ s'.probes = s.probes := by
  unfold settleOne at h
  split_ifs at h with hv
  · simp only [Sum.inl.injEq] at h
    subst h
    unfold fixEdge
    simp only []
    split
    · split_ifs <;> simp [TState.setEdge, TState.fresh]
    · simp [TState.setEdge, TState.fresh]
  · simp only [Sum.inl.injEq] at h
    subst h
    exact ⟨rfl, rfl, rfl, rfl⟩

theorem tallyStep_counts {s s' : TState α} {x : FreeMonoid α} (h : tallyStep C cut s x = .inl s') :
    s'.harv = (tallyPre C cut s x).harv ∧ s'.trav = (tallyPre C cut s x).trav
      ∧ s'.startH = (tallyPre C cut s x).startH ∧ s'.probes = (tallyPre C cut s x).probes := by
  unfold tallyStep at h
  simp only [] at h
  split at h
  · cases h
  · exact settleOne_counts cut h

/-- A step that continues starts a fresh stretch, or is the probe's outcome at the same hypothesis,
one probe further into the stretch. -/
theorem tallyStep_fresh {s s' : TState α} {x : FreeMonoid α} (h : tallyStep C cut s x = .inl s') :
    (s'.n = 0 ∧ s'.dis = 0 ∧ s'.pt = [])
      ∨ (s' = tallyPre C cut s x ∧ s'.n = s.n + 1 ∧ s'.tree = s.tree ∧ s'.edges = s.edges) := by
  unfold tallyStep at h
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
    · split_ifs <;> simp [TState.setEdge, TState.fresh]
    · simp [TState.setEdge, TState.fresh]
  · simp only [Sum.inl.injEq] at h
    subst h
    unfold tallyPre
    split
    · split
      · split
        · left; simp [TState.setEdge, TState.fresh]
        · right; simp [TState.charge]
      · right; simp [TState.charge]
    · right; split <;> simp [TState.charge, TState.addRec]
    all_goals right; simp [TState.charge]

theorem tallyStep_inr_pre {s s' : TState α} {x : FreeMonoid α} {e : TEnd α}
    (h : tallyStep C cut s x = .inr (e, s')) (he : e ≠ .tooBig) :
    s' = tallyPre C cut s x ∧ tallyLook C (tallyPre C cut s x) = some e := by
  unfold tallyStep at h
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

end Counts

section Reach

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : TallyCfg)
  (cut : FreeMonoid α → Option Bool) (S : ℕ)

/-- Reached by the loop: a tree of the class's shape, edges and records pointing at leaves. -/
def Reach (s : TState α) : Prop :=
  (∃ f, G.Grown s.tree f) ∧ EdgesInto s.tree s.edges ∧ RecsInto s ∧ (s.n = 0 → s.pt = [])

theorem reach_start : Reach G (tallyStart : TState α) :=
  ⟨⟨0, .start⟩, fun _ _ _ _ h => by simp [tallyStart] at h,
    fun _ _ _ h => by simp [tallyStart] at h, fun _ => rfl⟩

theorem reach_step (hm : 0 < C.m) {s s' : TState α} {x : FreeMonoid α} (hs : Reach G s)
    (h : tallyStep C cut s x = .inl s') :
    Reach G s' ∧ ∀ f f', G.Grown s.tree f → G.Grown s'.tree f' → f ≤ f' := by
  obtain ⟨⟨f, hf⟩, he, hr, -⟩ := hs
  have hpt : s'.n = 0 → s'.pt = [] := fun h0 => by
    rcases tallyStep_fresh C cut h with ⟨-, -, h'⟩ | ⟨-, h', -⟩
    · exact h'
    · omega
  rcases (tallyStep_spec cut C hm he hr).1 s' h with ⟨ht, he', hr'⟩ |
    ⟨p, c, t, t₀, ht, ht₀, hp, hT, he', hr', -, -⟩
  · refine ⟨⟨⟨f, ht ▸ hf⟩, he', hr', hpt⟩, fun f₁ f₂ h₁ h₂ => ?_⟩
    rw [ht] at h₂
    exact (G.grown_unique h₁ h₂).le
  · have hg : G.Grown s'.tree f ∨ G.Grown s'.tree (f + 1) := by
      by_cases hgen : G.GenuineSplit s.tree p (FreeMonoid.of c * s.tree.midAt (lcp t t₀))
      · exact .inl (hT ▸ .real hf ht ht₀ hgen)
      · exact .inr (hT ▸ .fake hf hp ht ht₀ hgen)
    refine ⟨⟨hg.elim (⟨_, ·⟩) (⟨_, ·⟩), he', ?_, hpt⟩, fun f₁ f₂ h₁ h₂ => ?_⟩
    · intro q e r hmem
      rw [hr' q e] at hmem
      cases hmem
    · have := G.grown_unique hf h₁
      rcases hg with hg | hg <;> have := G.grown_unique hg h₂ <;> omega

theorem notClass_step (hm : 0 < C.m) {s s' : TState α} {x : FreeMonoid α} (hs : Reach G s)
    (hn : ¬ G.InClass S s.tree) (h : tallyStep C cut s x = .inl s') : ¬ G.InClass S s'.tree := by
  obtain ⟨f, hf⟩ := hs.1
  have hfS : S < f := by
    by_contra hle
    exact hn ⟨f, by omega, hf⟩
  rintro ⟨f', hf'S, hf'⟩
  have := (reach_step G C cut hm hs h).2 f f' hf hf'
  omega

theorem class_paths {T : DTree α} (hT : G.InClass S T) (hL : Fintype.card σ + S + 2 ≤ C.Lmax) :
    T.paths.length ≤ C.Lmax := by
  obtain ⟨f, hf, hg⟩ := hT
  have := G.grown_paths hg
  omega

end Reach

theorem travBy_le (cut : FreeMonoid α → Option Bool) {t : DTree α} {edges : Edges α} {k : ℕ}
    {x : FreeMonoid α} (e : List Bool × α) : travBy cut t edges k x e ≤ x.toList.length + 1 := by
  unfold travBy
  exact (List.length_filter_le _ _).trans (by simp)

section Edge

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : TallyCfg)
  (cut : FreeMonoid α → Option Bool) (S : ℕ)

open scoped Classical in
/-- How many of the strings were read at a good read-state. -/
noncomputable def goodCount (l : List (FreeMonoid α)) : ℕ :=
  (l.filter fun b => G.Good (G.M.eval b.toList)).length

open scoped Classical in
theorem goodCount_add_badCount (l : List (FreeMonoid α)) :
    goodCount G l + G.badCount l = l.length := by
  unfold goodCount ReadModel.badCount
  rw [List.length_eq_length_filter_add (l := l) (fun b => decide (G.Good (G.M.eval b.toList)))]
  simp [decide_not]

theorem goodCount_append (l l' : List (FreeMonoid α)) :
    goodCount G (l ++ l') = goodCount G l + goodCount G l' := by
  simp [goodCount, List.filter_append]

/-- At the edge `e`: good read-states' undecided reads less `2θg` of its positions and `2η` of the
probes. -/
noncomputable def edgeZ (θg η : ℝ) (e : List Bool × α) (s : TState α) : ℝ :=
  goodCount G (s.harv e.1 e.2) - 2 * θg * s.trav e.1 e.2 - 2 * η * s.probes

/-- At some probe from a state of the class, the edge's count reaches `thr`. -/
def EvE (θg η : ℝ) (e : List Bool × α) (thr : ℝ) : TState α → List (FreeMonoid α) → Prop
  | _, [] => False
  | s, x :: xs => (G.InClass S s.tree ∧ thr ≤ edgeZ G θg η e (tallyPre C cut s x))
    ∨ match tallyStep C cut s x with
      | .inl s' => EvE θg η e thr s' xs
      | .inr _ => False

theorem evE_notClass (hm : 0 < C.m) (θg η : ℝ) (e : List Bool × α) (thr : ℝ) :
    ∀ (l : List (FreeMonoid α)) (s : TState α), Reach G s → ¬ G.InClass S s.tree →
      ¬ EvE G C cut S θg η e thr s l
  | [], _, _, _ => id
  | x :: xs, s, hs, hn => by
    simp only [EvE, hn, false_and, false_or]
    rcases h : tallyStep C cut s x with s' | _
    · exact evE_notClass hm θg η e thr xs s' (reach_step G C cut hm hs h).1
        (notClass_step G C cut S hm hs hn h)
    · exact id

/-- One probe's exponentiated increment averages at most `1`. -/
theorem mgf_step {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
    (D : Measure X) [IsProbabilityMeasure D] (U N : X → ℕ) {θ L R : ℝ} (hθ : 0 ≤ θ)
    (hL : 0 < L) (hR : 0 ≤ R) (hb : ∀ᵐ x ∂D, (U x : ℝ) ≤ L ∧ (N x : ℝ) ≤ L * R)
    {η : ℝ} (hη : 0 ≤ η) (hmean : ∫ x, (U x : ℝ) ∂D ≤ θ * ∫ x, (N x : ℝ) ∂D + η) :
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

theorem edgeZ_pre (θg η : ℝ) (e : List Bool × α) (s : TState α) (x : FreeMonoid α) :
    edgeZ G θg η e (tallyPre C cut s x) = edgeZ G θg η e s
      + (goodCount G (edgeHarvBy cut s.tree s.edges C.k x e)
        - 2 * θg * travBy cut s.tree s.edges C.k x e - 2 * η) := by
  obtain ⟨h1, h2, -, h4⟩ := tallyPre_charge C cut s x
  simp only [edgeZ, h1, h2, h4, goodCount_append]
  push_cast; ring

theorem edgeZ_step {θg η : ℝ} {e : List Bool × α} {s s' : TState α} {x : FreeMonoid α}
    (h : tallyStep C cut s x = .inl s') :
    edgeZ G θg η e s' = edgeZ G θg η e (tallyPre C cut s x) := by
  obtain ⟨h1, h2, -, h4⟩ := tallyStep_counts C cut h
  simp only [edgeZ, h1, h2, h4]

/-- The edge's exponentiated count is a supermartingale while the tree is in the class. -/
theorem edge_super (hm : 0 < C.m) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (rd : FreeMonoid α → ARU) {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hL : 1 ≤ L)
    {ρ θg θgs θgpt : ℝ} (hθg : 0 ≤ θg) (hLmax : Fintype.card σ + S + 2 ≤ C.Lmax)
    (hη : 0 ≤ C.φe * (C.θe / 2 - 2 * θg) / 2)
    (hE : TallyE G D C S ρ θg θgs θgpt rd) (e : List Bool × α) (thr : ℝ) :
    ∀ (T : ℕ) (s : TState α), Reach G s →
      (Measure.pi fun _ : Fin T => D)
          {xs | EvE G C (fun z => (rd z).cut) S θg (C.φe * (C.θe / 2 - 2 * θg) / 2) e thr s
            (List.ofFn xs)}
        ≤ ENNReal.ofReal (Real.exp ((edgeZ G θg (C.φe * (C.θe / 2 - 2 * θg) / 2) e s - thr)
          / (2 * L * (1 + 4 * θg * C.Lmax)))) := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
  set η : ℝ := C.φe * (C.θe / 2 - 2 * θg) / 2
  set K : ℝ := 2 * L * (1 + 4 * θg * C.Lmax)
  have hK : 0 < K := by
    have : (1 : ℝ) ≤ L := by exact_mod_cast hL
    positivity
  intro T
  induction T with
  | zero => intro s _; simp [EvE]
  | succ T ih =>
    intro s hs
    rw [pi_succ_apply]
    by_cases hc : G.InClass S s.tree
    swap
    · have : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
          {xs : Fin (T + 1) → FreeMonoid α | EvE G C cut S θg η e thr s (List.ofFn xs)}} = 0 := by
        intro x
        have hempty : {xs | Fin.cons x xs ∈
            {xs : Fin (T + 1) → FreeMonoid α | EvE G C cut S θg η e thr s (List.ofFn xs)}} = ∅ := by
          ext xs
          simp only [Set.mem_ofPred_eq, List.ofFn_cons, Set.mem_empty_iff_false, iff_false]
          exact evE_notClass G C cut S hm θg η e thr _ s hs hc
        rw [hempty, measure_empty]
      simp only [this, lintegral_zero]
      exact zero_le
    have hsec : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
        {xs : Fin (T + 1) → FreeMonoid α | EvE G C cut S θg η e thr s (List.ofFn xs)}}
        ≤ ENNReal.ofReal (Real.exp ((edgeZ G θg η e (tallyPre C cut s x) - thr) / K)) := by
      intro x
      by_cases hz : thr ≤ edgeZ G θg η e (tallyPre C cut s x)
      · refine prob_le_one.trans ?_
        rw [← ENNReal.ofReal_one]
        exact ENNReal.ofReal_le_ofReal (Real.one_le_exp (div_nonneg (by linarith) hK.le))
      rcases hst : tallyStep C cut s x with s' | ⟨e', s'⟩
      · have : {xs | Fin.cons x xs ∈
            {xs : Fin (T + 1) → FreeMonoid α | EvE G C cut S θg η e thr s (List.ofFn xs)}}
            = {xs | EvE G C cut S θg η e thr s' (List.ofFn xs)} := by
          ext xs
          simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvE, hz, and_false, false_or, hst]
        rw [this, ← edgeZ_step G C cut hst]
        exact ih s' (reach_step G C cut hm hs hst).1
      · have : {xs | Fin.cons x xs ∈
            {xs : Fin (T + 1) → FreeMonoid α | EvE G C cut S θg η e thr s (List.ofFn xs)}} = ∅ := by
          ext xs
          simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvE, hz, and_false, false_or, hst,
            Set.mem_empty_iff_false]
        rw [this, measure_empty]
        exact zero_le
    refine (lintegral_mono hsec).trans ?_
    have hfac : ∀ x, ENNReal.ofReal (Real.exp ((edgeZ G θg η e (tallyPre C cut s x) - thr) / K))
        = ENNReal.ofReal (Real.exp ((edgeZ G θg η e s - thr) / K))
          * ENNReal.ofReal (Real.exp ((goodCount G (edgeHarvBy cut s.tree s.edges C.k x e)
            - 2 * θg * travBy cut s.tree s.edges C.k x e - 2 * η) / K)) := by
      intro x
      rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add, edgeZ_pre]
      congr 2
      ring
    simp only [hfac]
    rw [lintegral_const_mul _ (measurable_of_countable _)]
    refine (mul_le_mul_right (mgf_step D
      (fun x => goodCount G (edgeHarvBy cut s.tree s.edges C.k x e))
      (fun x => travBy cut s.tree s.edges C.k x e) hθg (by positivity) (by positivity) ?_
      hη ?_) _).trans (le_of_eq (mul_one _))
    · have hpaths := class_paths G C S hc hLmax
      filter_upwards [hlen] with x hx
      constructor
      · have h1 : goodCount G (edgeHarvBy cut s.tree s.edges C.k x e)
            ≤ (edgeHarvBy cut s.tree s.edges C.k x e).length := List.length_filter_le _ _
        have h2 := edgeHarvBy_length_le cut (t := s.tree) (edges := s.edges) (k := C.k) (x := x) e
        exact_mod_cast h1.trans (h2.trans hx)
      · have h1 := travBy_le cut (t := s.tree) (edges := s.edges) (k := C.k) (x := x) e
        have : travBy cut s.tree s.edges C.k x e ≤ L * C.Lmax := by
          have : 2 ≤ C.Lmax := by omega
          calc _ ≤ x.toList.length + 1 := h1
            _ ≤ L + L := by omega
            _ ≤ L * C.Lmax := by nlinarith
        exact_mod_cast this
    · exact hE.goodEdge s.tree s.edges hc hs.2.1 e

end Edge

section Start

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : TallyCfg)
  (cut : FreeMonoid α → Option Bool) (S : ℕ)

/-- At the `j`-th probe after this one, from a state of the class, the start's good read-states'
undecided reads reach `k`. -/
def EvS : ℕ → TState α → ℕ → List (FreeMonoid α) → Prop
  | _, _, _, [] => False
  | 0, s, k, x :: _ => G.InClass S s.tree ∧ k ≤ goodCount G (tallyPre C cut s x).startH
  | j + 1, s, k, x :: xs => match tallyStep C cut s x with
    | .inl s' => EvS j s' k xs
    | .inr _ => False

theorem evS_notClass (hm : 0 < C.m) :
    ∀ (l : List (FreeMonoid α)) (j : ℕ) (s : TState α) (k : ℕ), Reach G s →
      ¬ G.InClass S s.tree → ¬ EvS G C cut S j s k l
  | [], j, _, _, _, _ => by cases j <;> simp [EvS]
  | x :: xs, 0, s, k, _, hn => fun h => hn h.1
  | x :: xs, j + 1, s, k, hs, hn => by
    simp only [EvS]
    rcases h : tallyStep C cut s x with s' | _
    · exact evS_notClass hm xs j s' k (reach_step G C cut hm hs h).1
        (notClass_step G C cut S hm hs hn h)
    · exact id

open scoped Classical in
theorem goodCount_startPre (rd : FreeMonoid α → ARU) (s : TState α) (x : FreeMonoid α) :
    goodCount G (tallyPre C (fun z => (rd z).cut) s x).startH = goodCount G s.startH
      + if x ∈ G.startAt rd C.k s.tree G.Good then 1 else 0 := by
  obtain ⟨-, -, h, -⟩ := tallyPre_charge C (fun z => (rd z).cut) s x
  rw [h, goodCount_append]
  congr 1
  simp only [startHarvBy, ReadModel.startAt, Set.mem_ofPred_eq]
  obtain ⟨p, hp⟩ | ⟨b, hb⟩ : (∃ p, s.tree.sift (fun z => (rd z).cut) (prefixOf x C.k) = .inl p)
      ∨ ∃ b, s.tree.sift (fun z => (rd z).cut) (prefixOf x C.k) = .inr b := by
    rcases s.tree.sift (fun z => (rd z).cut) (prefixOf x C.k) with p | b
    · exact .inl ⟨p, rfl⟩
    · exact .inr ⟨b, rfl⟩
  · simp [hp, goodCount]
  · by_cases hg : G.Good (G.M.eval b.toList) <;> simp [hb, goodCount, hg]

theorem binomSfGe_gt {n j : ℕ} (p : ℝ) (h : n < j) : binomSfGe n p j = 0 := by
  rw [binomSfGe_eq_range]
  exact Finset.sum_eq_zero fun i hi => by
    simp only [Finset.mem_range] at hi; rw [if_neg (by omega)]

/-- The start's good read-states' undecided reads at the `j + 1`-th probe are at most binomial. -/
theorem start_bin (hm : 0 < C.m) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (rd : FreeMonoid α → ARU) {ρ θg θgs θgpt : ℝ} (hθ0 : 0 ≤ θgs) (hθ1 : θgs ≤ 1)
    (hE : TallyE G D C S ρ θg θgs θgpt rd) :
    ∀ (T j : ℕ) (s : TState α) (k : ℕ), Reach G s →
      (Measure.pi fun _ : Fin T => D)
          {xs | EvS G C (fun z => (rd z).cut) S j s (goodCount G s.startH + k) (List.ofFn xs)}
        ≤ ENNReal.ofReal (binomSfGe (j + 1) θgs k) := by
  classical
  intro T
  induction T with
  | zero => intro j s k _; cases j <;> simp [EvS]
  | succ T ih =>
    intro j s k hs
    rcases k with _ | k
    · rw [binomSfGe_zero_right, ENNReal.ofReal_one]; exact prob_le_one
    by_cases hc : G.InClass S s.tree
    swap
    · have : {xs : Fin (T + 1) → FreeMonoid α |
          EvS G C (fun z => (rd z).cut) S j s
            (goodCount G s.startH + (k + 1)) (List.ofFn xs)} = ∅ := by
        ext xs
        simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
        exact evS_notClass G C (fun z => (rd z).cut) S hm _ j s _ hs hc
      rw [this, measure_empty]; exact zero_le
    set A := G.startAt rd C.k s.tree G.Good
    have hA : D.real A ≤ θgs := hE.goodStart s.tree hc
    have hA0 : 0 ≤ D.real A := measureReal_nonneg
    rw [pi_succ_apply]
    rcases j with _ | j
    · have hsec : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
          {xs : Fin (T + 1) → FreeMonoid α |
            EvS G C (fun z => (rd z).cut) S 0 s (goodCount G s.startH + (k + 1)) (List.ofFn xs)}}
          ≤ A.indicator (fun _ => if k = 0 then 1 else 0) x := by
        intro x
        by_cases hx : x ∈ A
        · rw [Set.indicator_of_mem hx]
          rcases k with _ | k
          · exact prob_le_one
          · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
                EvS G C (fun z => (rd z).cut) S 0 s
                  (goodCount G s.startH + (k + 1 + 1)) (List.ofFn xs)}} = ∅ := by
              ext xs
              simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvS, goodCount_startPre,
                if_pos (show x ∈ G.startAt rd C.k s.tree G.Good from hx),
                Set.mem_empty_iff_false, iff_false, not_and]
              intro _; omega
            rw [this, measure_empty]; simp
        · rw [Set.indicator_of_notMem hx]
          have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
              EvS G C (fun z => (rd z).cut) S 0 s
                (goodCount G s.startH + (k + 1)) (List.ofFn xs)}} = ∅ := by
            ext xs
            simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvS, goodCount_startPre,
              if_neg (show x ∉ G.startAt rd C.k s.tree G.Good from hx),
              Set.mem_empty_iff_false, iff_false, not_and]
            intro _; omega
          rw [this, measure_empty]
      refine (lintegral_mono hsec).trans ?_
      rw [lintegral_indicator (Set.to_countable _).measurableSet, setLIntegral_const,
        ← ofReal_measureReal]
      rcases k with _ | k
      · rw [if_pos rfl, one_mul, show (0 + 1 : ℕ) = 0 + 1 from rfl, binomSfGe_succ,
          binomSfGe_zero_right, binomSfGe_zero_left]
        exact ENNReal.ofReal_le_ofReal (by linarith)
      · simp
    · set c₁ := binomSfGe (j + 1) θgs k
      set c₀ := binomSfGe (j + 1) θgs (k + 1)
      have hc₀ : 0 ≤ c₀ := binomSfGe_nonneg hθ0 hθ1 _
      have hc₀₁ : c₀ ≤ c₁ := binomSfGe_antitone hθ0 hθ1 k
      have hsec : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
          {xs : Fin (T + 1) → FreeMonoid α |
            EvS G C (fun z => (rd z).cut) S (j + 1) s
              (goodCount G s.startH + (k + 1)) (List.ofFn xs)}}
          ≤ A.indicator (fun _ => ENNReal.ofReal c₁) x
            + Aᶜ.indicator (fun _ => ENNReal.ofReal c₀) x := by
        intro x
        rcases hst : tallyStep C (fun z => (rd z).cut) s x with s' | ⟨e, s'⟩
        · have hgc : goodCount G s'.startH = goodCount G s.startH
              + if x ∈ A then 1 else 0 := by
            rw [(tallyStep_counts C (fun z => (rd z).cut) hst).2.2.1]
            exact goodCount_startPre G C rd s x
          have hs' := (reach_step G C (fun z => (rd z).cut) hm hs hst).1
          by_cases hx : x ∈ A
          · rw [Set.indicator_of_mem hx, Set.indicator_of_notMem (by simpa using hx), add_zero]
            have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
                EvS G C (fun z => (rd z).cut) S (j + 1) s
                  (goodCount G s.startH + (k + 1)) (List.ofFn xs)}}
                = {xs | EvS G C (fun z => (rd z).cut) S j s'
                  (goodCount G s'.startH + k) (List.ofFn xs)} := by
              ext xs
              simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvS, hst, hgc, if_pos hx]
              rw [show goodCount G s.startH + (k + 1) = goodCount G s.startH + 1 + k by omega]
            rw [this]; exact ih j s' k hs'
          · rw [Set.indicator_of_notMem hx, Set.indicator_of_mem (by simpa using hx), zero_add]
            have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
                EvS G C (fun z => (rd z).cut) S (j + 1) s
                  (goodCount G s.startH + (k + 1)) (List.ofFn xs)}}
                = {xs | EvS G C (fun z => (rd z).cut) S j s'
                  (goodCount G s'.startH + (k + 1)) (List.ofFn xs)} := by
              ext xs
              simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvS, hst, hgc, if_neg hx, add_zero]
            rw [this]; exact ih j s' (k + 1) hs'
        · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
              EvS G C (fun z => (rd z).cut) S (j + 1) s
                (goodCount G s.startH + (k + 1)) (List.ofFn xs)}} = ∅ := by
            ext xs
            simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvS, hst, Set.mem_empty_iff_false]
          rw [this, measure_empty]; exact zero_le
      refine (lintegral_mono hsec).trans ?_
      have hc₁ : 0 ≤ c₁ := hc₀.trans hc₀₁
      rw [lintegral_add_left (measurable_of_countable _), lintegral_indicator
          (Set.to_countable _).measurableSet,
            lintegral_indicator (Set.to_countable _).measurableSet,
        setLIntegral_const, setLIntegral_const, ← ofReal_measureReal, ← ofReal_measureReal,
        measureReal_compl (Set.to_countable _).measurableSet, probReal_univ,
        ← ENNReal.ofReal_mul hc₁, ← ENNReal.ofReal_mul hc₀,
        ← ENNReal.ofReal_add (mul_nonneg hc₁ hA0) (mul_nonneg hc₀ (by linarith))]
      refine ENNReal.ofReal_le_ofReal ?_
      rw [show j + 1 + 1 = (j + 1) + 1 from rfl, binomSfGe_succ]
      change c₁ * D.real A + c₀ * (1 - D.real A) ≤ θgs * c₁ + (1 - θgs) * c₀
      nlinarith

end Start

section PT

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) (C : TallyCfg)
  (cut : FreeMonoid α → Option Bool) (S : ℕ)

/-- After `i` probes, the run is at a state from which `Q` holds of the remaining draws. -/
def Skip (Q : TState α → List (FreeMonoid α) → Prop) : ℕ → TState α → List (FreeMonoid α) → Prop
  | 0, s, l => Q s l
  | _ + 1, _, [] => False
  | i + 1, s, x :: xs => match tallyStep C cut s x with
    | .inl s' => Skip Q i s' xs
    | .inr _ => False

theorem ptHarvBy_cases (T : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    ptHarvBy cut T edges k x = [] ∨ ∃ b, ptHarvBy cut T edges k x = [b] := by
  unfold ptHarvBy
  split
  · rcases T.sift cut _ with _ | b
    · exact .inl rfl
    · exact .inr ⟨b, rfl⟩
  · rcases T.sift cut _ with _ | b
    · exact .inl rfl
    · exact .inr ⟨b, rfl⟩
  · exact .inl rfl

open scoped Classical in
theorem tallyPre_stretch (s : TState α) (x : FreeMonoid α) :
    ((tallyPre C cut s x).n = s.n + 1
      ∧ (tallyPre C cut s x).pt = s.pt ++ ptHarvBy cut s.tree s.edges C.k x
      ∧ (tallyPre C cut s x).dis
        = s.dis + if (probeBy cut s.tree s.edges C.k x).IsSearch then 1 else 0)
    ∨ ((tallyPre C cut s x).n = 0 ∧ (tallyPre C cut s x).pt = [] ∧ (tallyPre C cut s x).dis = 0
      ∧ ¬ (probeBy cut s.tree s.edges C.k x).IsSearch) := by
  unfold tallyPre
  rcases ho : probeBy cut s.tree s.edges C.k x with _ | w | w | j | ⟨ps, fd⟩ | j | u <;>
    try dsimp only
  · left; simp [TState.charge, Outcome.IsSearch]
  · left; simp [TState.charge, Outcome.IsSearch]
  · left; simp [TState.charge, Outcome.IsSearch]
  · left; simp [TState.charge, Outcome.IsSearch]
  · left
    rcases recordBy cut C.k (s.tree, s.edges) x with _ | ⟨⟨p, c, t⟩, sp⟩ <;>
      simp [TState.charge, TState.addRec, Outcome.IsSearch]
  · left; simp [TState.charge, Outcome.IsSearch]
  · rcases x.toList[u.toList.length]? with _ | c <;> try dsimp only
    · left; simp [TState.charge, Outcome.IsSearch]
    rcases s.tree.sift cut u with p | b <;> try dsimp only
    swap
    · left; simp [TState.charge, Outcome.IsSearch]
    rcases s.tree.sift cut (u * FreeMonoid.of c) with t | b <;> rcases s.edges p c with _ | e <;>
      try dsimp only
    · right; simp [TState.setEdge, TState.fresh, Outcome.IsSearch]
    all_goals left; simp [TState.charge, Outcome.IsSearch]

open scoped Classical in
theorem goodCount_ptPre (rd : FreeMonoid α → ARU) (s : TState α) (x : FreeMonoid α) :
    goodCount G (tallyPre C (fun z => (rd z).cut) s x).pt ≤ goodCount G s.pt
      + if x ∈ G.ptAt rd C.k s.tree s.edges G.Good then 1 else 0 := by
  have h : (tallyPre C (fun z => (rd z).cut) s x).pt = s.pt ++ ptHarvBy (fun z => (rd z).cut)
      s.tree s.edges C.k x ∨ (tallyPre C (fun z => (rd z).cut) s x).pt = [] :=
    (tallyPre_stretch C _ s x).imp (·.2.1) (·.2.1)
  rcases h with h | h
  · rw [h, goodCount_append]
    rcases ptHarvBy_cases (fun z => (rd z).cut) s.tree s.edges C.k x with h0 | ⟨b, hb⟩
    · rw [h0]; simp [goodCount]
    · rw [hb]
      by_cases hg : G.Good (G.M.eval b.toList)
      · have : x ∈ G.ptAt rd C.k s.tree s.edges G.Good := ⟨b, hb, hg⟩
        simp [goodCount, hg, this]
      · simp [goodCount, hg]
  · rw [h]; simp [goodCount]

theorem mem_searchAt {rd : FreeMonoid α → ARU} {k : ℕ} {T : DTree α} {edges : Edges α}
    {x : FreeMonoid α} :
    x ∈ ReadModel.searchAt rd k T edges ↔ (probeBy (fun z => (rd z).cut) T edges k x).IsSearch := by
  simp only [ReadModel.searchAt, Set.mem_setOf_eq]
  rcases probeBy (fun z => (rd z).cut) T edges k x with _ | _ | _ | _ | _ | _ | _ <;>
    simp [Outcome.IsSearch]

theorem ptAt_sub_searchAt (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : Edges α)
    (P : σ → Prop) : G.ptAt rd k T edges P ⊆ ReadModel.searchAt rd k T edges := by
  rintro x ⟨b, hb, -⟩
  rw [mem_searchAt]
  unfold ptHarvBy at hb
  rcases ho : probeBy (fun z => (rd z).cut) T edges k x with _ | _ | _ | _ | _ | _ | _ <;>
    rw [ho] at hb <;> simp_all [Outcome.IsSearch]

open scoped Classical in
/-- From a state of the class, within the stretch, the good read-states' undecided middles reach
`k` more at the probe that brings the stretch's searches to `t` more. -/
def EvR (ok : TState α → Prop) : ℕ → ℕ → TState α → List (FreeMonoid α) → Prop
  | _, _, _, [] => False
  | t, k, s, x :: xs => (G.InClass S s.tree ∧ ok s) ∧ (tallyPre C cut s x).dis - s.dis ≤ t ∧
    (((tallyPre C cut s x).dis - s.dis = t
        ∧ k ≤ goodCount G (tallyPre C cut s x).pt - goodCount G s.pt)
      ∨ match tallyStep C cut s x with
        | .inl s' => s'.n ≠ 0 ∧ EvR ok (t - ((tallyPre C cut s x).dis - s.dis))
            (k - (goodCount G (tallyPre C cut s x).pt - goodCount G s.pt)) s' xs
        | .inr _ => False)

theorem evR_ne_nil {ok : TState α → Prop} {t k : ℕ} {s : TState α} {l : List (FreeMonoid α)}
    (h : EvR G C cut S ok t k s l) : l ≠ [] := by
  rintro rfl; exact h

theorem evR_mono (ok : TState α → Prop) :
    ∀ (l : List (FreeMonoid α)) (t : ℕ) (s : TState α) {k k' : ℕ}, k ≤ k' →
      EvR G C cut S ok t k' s l → EvR G C cut S ok t k s l
  | [], _, _, _, _, _, h => h
  | x :: xs, t, s, k, k', hk, h => by
    simp only [EvR] at h ⊢
    obtain ⟨hc, hdt, h⟩ := h
    refine ⟨hc, hdt, ?_⟩
    rcases h with ⟨h1, h2⟩ | h
    · exact .inl ⟨h1, hk.trans h2⟩
    · right
      rcases hst : tallyStep C cut s x with s' | _ <;> rw [hst] at h
      · exact ⟨h.1, evR_mono ok xs _ s' (by omega) h.2⟩
      · exact h

theorem evR_notClass (ok : TState α → Prop) :
    ∀ (l : List (FreeMonoid α)) (t k : ℕ) (s : TState α),
      ¬ (G.InClass S s.tree ∧ ok s) → ¬ EvR G C cut S ok t k s l
  | [], _, _, _, _ => id
  | x :: xs, t, k, s, hn => fun h => hn h.1

open scoped Classical in
theorem tallyPre_dis_sub (rd : FreeMonoid α → ARU) (s : TState α) (x : FreeMonoid α) :
    (tallyPre C (fun z => (rd z).cut) s x).dis - s.dis
      = if x ∈ ReadModel.searchAt rd C.k s.tree s.edges then 1 else 0 := by
  rcases tallyPre_stretch C (fun z => (rd z).cut) s x with ⟨-, -, h⟩ | ⟨-, -, h, hn⟩
  · rw [h]
    by_cases hx : x ∈ ReadModel.searchAt rd C.k s.tree s.edges
    · rw [if_pos hx, if_pos (mem_searchAt.1 hx)]; omega
    · rw [if_neg hx, if_neg (fun h' => hx (mem_searchAt.2 h'))]; omega
  · rw [h, if_neg (fun h' => hn (mem_searchAt.1 h'))]; omega

open scoped Classical in
/-- The good read-states' undecided middles among the stretch's next `t` searches are at most
binomial. -/
theorem ptR_bin (hm : 0 < C.m) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (rd : FreeMonoid α → ARU) {ρ θg θgs θgpt : ℝ} (hθ0 : 0 ≤ θgpt) (hθ1 : 2 * θgpt ≤ 1)
    (hE : TallyE G D C S ρ θg θgs θgpt rd) :
    ∀ (T t k : ℕ) (s : TState α), Reach G s →
      (Measure.pi fun _ : Fin T => D)
          {xs | EvR G C (fun z => (rd z).cut) S
            (fun s => C.φpt / 2 ≤ D.real (ReadModel.searchAt rd C.k s.tree s.edges)) t k s
            (List.ofFn xs)}
        ≤ ENNReal.ofReal (binomSfGe t (2 * θgpt) k) := by
  have h20 : 0 ≤ 2 * θgpt := by linarith
  intro T
  induction T with
  | zero => intro t k s _; simp [EvR]
  | succ T ih =>
    intro t k s hs
    rcases k with _ | k
    · rw [binomSfGe_zero_right, ENNReal.ofReal_one]; exact prob_le_one
    by_cases hc : G.InClass S s.tree
        ∧ C.φpt / 2 ≤ D.real (ReadModel.searchAt rd C.k s.tree s.edges)
    swap
    · have : {xs : Fin (T + 1) → FreeMonoid α |
          EvR G C (fun z => (rd z).cut) S
            (fun s => C.φpt / 2 ≤ D.real (ReadModel.searchAt rd C.k s.tree s.edges)) t (k + 1) s
            (List.ofFn xs)} = ∅ := by
        ext xs
        simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
        exact evR_notClass G C _ S _ _ t _ s hc
      rw [this, measure_empty]; exact zero_le
    set A := G.ptAt rd C.k s.tree s.edges G.Good
    set B := ReadModel.searchAt rd C.k s.tree s.edges
    have hAB : A ⊆ B := ptAt_sub_searchAt G rd C.k s.tree s.edges G.Good
    have hA : D.real A ≤ 2 * θgpt * D.real B := by
      have := hE.goodPT s.tree s.edges hc.1 hs.2.1
      have := hc.2
      nlinarith
    have hA0 : 0 ≤ D.real A := measureReal_nonneg
    have hAB' : D.real A ≤ D.real B := measureReal_mono hAB
    have hB1 : D.real B ≤ 1 := by
      have := measureReal_mono (μ := D) (Set.subset_univ B)
      rwa [probReal_univ] at this
    have hdt := fun x => tallyPre_dis_sub C rd s x
    have hdg : ∀ x, goodCount G (tallyPre C (fun z => (rd z).cut) s x).pt - goodCount G s.pt
        ≤ if x ∈ A then 1 else 0 := fun x => by
      have := goodCount_ptPre G C rd s x
      split_ifs at this ⊢ <;> omega
    -- the section at `x` of the event
    have hsecE : ∀ x, {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
        EvR G C (fun z => (rd z).cut) S
            (fun s => C.φpt / 2 ≤ D.real (ReadModel.searchAt rd C.k s.tree s.edges)) t (k + 1) s (List.ofFn xs)}}
        ⊆ {xs | (tallyPre C (fun z => (rd z).cut) s x).dis - s.dis ≤ t ∧
          (((tallyPre C (fun z => (rd z).cut) s x).dis - s.dis = t ∧ k + 1
              ≤ goodCount G (tallyPre C (fun z => (rd z).cut) s x).pt - goodCount G s.pt)
          ∨ ∃ s', tallyStep C (fun z => (rd z).cut) s x = .inl s' ∧ Reach G s' ∧
            EvR G C (fun z => (rd z).cut) S
              (fun s => C.φpt / 2 ≤ D.real (ReadModel.searchAt rd C.k s.tree s.edges)) (t - ((tallyPre C (fun z => (rd z).cut) s x).dis
              - s.dis)) (k + 1 - (goodCount G (tallyPre C (fun z => (rd z).cut) s x).pt
                - goodCount G s.pt)) s' (List.ofFn xs))} := by
      intro x xs hxs
      simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvR] at hxs
      obtain ⟨-, hd, h⟩ := hxs
      refine ⟨hd, ?_⟩
      rcases h with h | h
      · exact .inl h
      · right
        rcases hst : tallyStep C (fun z => (rd z).cut) s x with s' | _ <;> rw [hst] at h
        · exact ⟨s', rfl, (reach_step G C _ hm hs hst).1, h.2⟩
        · exact h.elim
    -- the run continues from the state after `x`
    have hcont : ∀ x t' k', (Measure.pi fun _ : Fin T => D) {xs : Fin T → FreeMonoid α |
        ∃ s', tallyStep C (fun z => (rd z).cut) s x = .inl s' ∧ Reach G s' ∧
          EvR G C (fun z => (rd z).cut) S
            (fun s => C.φpt / 2 ≤ D.real (ReadModel.searchAt rd C.k s.tree s.edges)) t' k' s' (List.ofFn xs)}
        ≤ ENNReal.ofReal (binomSfGe t' (2 * θgpt) k') := by
      intro x t' k'
      rcases hst : tallyStep C (fun z => (rd z).cut) s x with s' | _
      · refine le_trans (measure_mono fun xs hxs => ?_) (ih t' k' s' (reach_step G C _ hm hs hst).1)
        obtain ⟨s'', h1, -, h2⟩ := hxs
        cases h1; exact h2
      · simp
    set c₁ : ℝ := if t = 0 then 0 else binomSfGe (t - 1) (2 * θgpt) k
    set c₀ : ℝ := if t = 0 then 0 else binomSfGe (t - 1) (2 * θgpt) (k + 1)
    set c₂ : ℝ := binomSfGe t (2 * θgpt) (k + 1)
    have hc₀ : 0 ≤ c₀ := by simp only [c₀]; split_ifs; exact le_rfl; exact binomSfGe_nonneg h20 hθ1 _
    have hc₁ : 0 ≤ c₁ := by simp only [c₁]; split_ifs; exact le_rfl; exact binomSfGe_nonneg h20 hθ1 _
    have hc₂ : 0 ≤ c₂ := binomSfGe_nonneg h20 hθ1 _
    have hsec : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
        {xs : Fin (T + 1) → FreeMonoid α |
          EvR G C (fun z => (rd z).cut) S
            (fun s => C.φpt / 2 ≤ D.real (ReadModel.searchAt rd C.k s.tree s.edges)) t (k + 1) s (List.ofFn xs)}}
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
          rw [show t + 1 - ((tallyPre C (fun z => (rd z).cut) s x).dis - s.dis) = t by omega]
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
          rwa [show t + 1 - ((tallyPre C (fun z => (rd z).cut) s x).dis - s.dis) = t by omega,
            show k + 1 - (goodCount G (tallyPre C (fun z => (rd z).cut) s x).pt
              - goodCount G s.pt) = k + 1 by omega] at h5
      · rw [Set.indicator_of_notMem hxA, Set.indicator_of_notMem (fun h => hxB h.1),
          Set.indicator_of_mem hxB, zero_add, zero_add]
        rw [if_neg hxB] at h1
        rw [if_neg hxA] at h2
        refine (measure_mono fun xs hxs => ?_).trans (hcont x t (k + 1))
        obtain ⟨-, h | ⟨s', h3, h4, h5⟩⟩ := hxs
        · omega
        · refine ⟨s', h3, h4, ?_⟩
          rwa [show t - ((tallyPre C (fun z => (rd z).cut) s x).dis - s.dis) = t by omega,
            show k + 1 - (goodCount G (tallyPre C (fun z => (rd z).cut) s x).pt
              - goodCount G s.pt) = k + 1 by omega] at h5
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
    · have e₁ : c₁ = binomSfGe t (2 * θgpt) k := by simp [c₁]
      have e₀ : c₀ = binomSfGe t (2 * θgpt) (k + 1) := by simp [c₀]
      have e₂ : c₂ = 2 * θgpt * c₁ + (1 - 2 * θgpt) * c₀ := by
        simp only [c₂, e₁, e₀]; exact binomSfGe_succ t (2 * θgpt) k
      have hmono : c₀ ≤ c₁ := by rw [e₁, e₀]; exact binomSfGe_antitone h20 hθ1 k
      rw [e₂]
      nlinarith [mul_le_mul_of_nonneg_right hA (sub_nonneg.2 hmono)]

theorem skip_le (hm : 0 < C.m) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (Q : TState α → List (FreeMonoid α) → Prop) {c : ENNReal}
    (hQ : ∀ (T : ℕ) (s : TState α), Reach G s →
      (Measure.pi fun _ : Fin T => D) {xs | Q s (List.ofFn xs)} ≤ c) :
    ∀ (T i : ℕ) (s : TState α), Reach G s →
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
          rcases hst : tallyStep C cut s x with s' | _
          · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
                Skip C cut Q (i + 1) s (List.ofFn xs)}} = {xs | Skip C cut Q i s' (List.ofFn xs)} := by
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

variable (C : TallyCfg) (D : Measure (FreeMonoid α)) (rd : FreeMonoid α → ARU)

/-- The hypothesis's disagreement rate: the probes that search. -/
noncomputable def disRate (s : TState α) : ℝ := D.real (ReadModel.searchAt rd C.k s.tree s.edges)

/-- The searches a probe adds to the stretch's count. -/
noncomputable def dt (s : TState α) (x : FreeMonoid α) : ℕ :=
  (tallyPre C (fun z => (rd z).cut) s x).dis - s.dis

/-- From `s`, at a hypothesis searching on at most `φpt/2` of the probes, the next `N + 1` probes
stay in the stretch and search at least `h` times. -/
def EvH : ℕ → ℕ → TState α → List (FreeMonoid α) → Prop
  | _, _, _, [] => False
  | 0, h, s, x :: _ => disRate C D rd s ≤ C.φpt / 2 ∧ h ≤ dt C rd s x
  | N + 1, h, s, x :: xs => disRate C D rd s ≤ C.φpt / 2 ∧
    match tallyStep C (fun z => (rd z).cut) s x with
    | .inl s' => s'.n ≠ 0 ∧ EvH N (h - dt C rd s x) s' xs
    | .inr _ => False

open scoped Classical in
/-- A stretch at a hypothesis searching on at most `φpt/2` of the probes searches at least `h`
times in its next `N + 1` probes with chance at most `P(Bin(N + 1, φpt/2) ≥ h)`. -/
theorem evH_le [IsProbabilityMeasure D] (hφ0 : 0 ≤ C.φpt) (hφ1 : C.φpt ≤ 1) :
    ∀ (T N h : ℕ) (s : TState α),
      (Measure.pi fun _ : Fin T => D) {xs | EvH C D rd N h s (List.ofFn xs)}
        ≤ ENNReal.ofReal (binomSfGe (N + 1) (C.φpt / 2) h) := by
  have hp0 : 0 ≤ C.φpt / 2 := by linarith
  have hp1 : C.φpt / 2 ≤ 1 := by linarith
  intro T
  induction T with
  | zero => intro N h s; rcases N with _ | N <;> simp [EvH]
  | succ T ih =>
    intro N h s
    rcases h with _ | h
    · rw [binomSfGe_zero_right, ENNReal.ofReal_one]; exact prob_le_one
    by_cases hd : disRate C D rd s ≤ C.φpt / 2
    swap
    · refine (measure_mono (t := ∅) fun xs hxs => ?_).trans (by simp)
      simp only [Set.mem_ofPred_eq, List.ofFn_succ] at hxs
      rcases N with _ | N
      · exact hd hxs.1
      · exact hd hxs.1
    set B := ReadModel.searchAt rd C.k s.tree s.edges
    set A : ℝ := binomSfGe N (C.φpt / 2) h
    set Cc : ℝ := binomSfGe N (C.φpt / 2) (h + 1)
    have hC0 : 0 ≤ Cc := binomSfGe_nonneg hp0 hp1 _
    have hCA : Cc ≤ A := binomSfGe_antitone hp0 hp1 h
    have hA0 : 0 ≤ A := hC0.trans hCA
    have hsec : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
        {xs : Fin (T + 1) → FreeMonoid α | EvH C D rd N (h + 1) s (List.ofFn xs)}}
        ≤ B.indicator (fun _ => ENNReal.ofReal A) x
          + Bᶜ.indicator (fun _ => ENNReal.ofReal Cc) x := by
      intro x
      have hdt : dt C rd s x = if x ∈ B then 1 else 0 := tallyPre_dis_sub C rd s x
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
        · rcases hst : tallyStep C (fun z => (rd z).cut) s x with s' | _
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
        · rcases hst : tallyStep C (fun z => (rd z).cut) s x with s' | _
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
    have hrec : binomSfGe (N + 1) (C.φpt / 2) (h + 1)
        = C.φpt / 2 * A + (1 - C.φpt / 2) * Cc := binomSfGe_succ N (C.φpt / 2) h
    rw [hrec]
    have hdB : D.real B ≤ C.φpt / 2 := hd
    nlinarith

end Low

section Assemble

theorem tallyLook_start' (C : TallyCfg) {s : TState α} (h : tallyLook C s = some .harvestStart) :
    C.n₀ ≤ s.probes ∧ binomSfGe s.probes C.θs s.startH.length < C.a := by
  unfold tallyLook at h
  by_cases h1 : rateSide C.θs C.a C.n₀ s.probes s.startH.length = some true
  · unfold rateSide at h1
    split_ifs at h1 with h2 h3 <;> simp_all
  · rw [if_neg h1] at h
    split_ifs at h <;> simp at h

theorem tallyLook_harvest' (C : TallyCfg) {s : TState α} {e : List Bool × α}
    (h : tallyLook C s = some (.harvest e)) :
    e.1 ∈ s.tree.paths ∧ C.φe * s.probes ≤ s.trav e.1 e.2 ∧ C.exc (s.trav e.1 e.2)
      ≤ ((s.harv e.1 e.2).length : ℝ) - C.θe * s.trav e.1 e.2 := by
  unfold tallyLook at h
  by_cases h1 : rateSide C.θs C.a C.n₀ s.probes s.startH.length = some true
  · rw [if_pos h1] at h; simp at h
  · rw [if_neg h1] at h
    by_cases h2 : ∃ e : List Bool × α, e.1 ∈ s.tree.paths ∧ C.φe * s.probes ≤ s.trav e.1 e.2
        ∧ C.exc (s.trav e.1 e.2) ≤ ((s.harv e.1 e.2).length : ℝ) - C.θe * s.trav e.1 e.2
    · rw [dif_pos h2] at h
      simp only [Option.some.injEq, TEnd.harvest.injEq] at h
      rw [← h]
      exact h2.choose_spec
    · rw [dif_neg h2] at h
      split_ifs at h <;> simp at h

theorem tallyLook_pt' (C : TallyCfg) {s : TState α} (h : tallyLook C s = some .harvestPT) :
    C.φpt * s.n ≤ s.dis ∧ C.n₀ ≤ s.dis ∧ binomSfGe s.dis C.θpt s.pt.length < C.a := by
  unfold tallyLook at h
  by_cases h1 : rateSide C.θs C.a C.n₀ s.probes s.startH.length = some true
  · rw [if_pos h1] at h; simp at h
  rw [if_neg h1] at h
  by_cases h2 : ∃ e : List Bool × α, e.1 ∈ s.tree.paths ∧ C.φe * s.probes ≤ s.trav e.1 e.2
      ∧ C.exc (s.trav e.1 e.2) ≤ ((s.harv e.1 e.2).length : ℝ) - C.θe * s.trav e.1 e.2
  · rw [dif_pos h2] at h; simp at h
  rw [dif_neg h2] at h
  by_cases h3 : C.φpt * s.n ≤ s.dis ∧ rateSide C.θpt C.a C.n₀ s.dis s.pt.length = some true
  · obtain ⟨hg, h3⟩ := h3
    unfold rateSide at h3
    split_ifs at h3 with h4 h5 <;> first | exact ⟨hg, h4, h5⟩ | simp at h3
  · rw [if_neg h3] at h
    split_ifs at h <;> simp at h

theorem paths_length_pos : ∀ T : DTree α, 0 < T.paths.length
  | .leaf => by simp [DTree.paths]
  | .node _ r a => by
    simp only [DTree.paths, List.length_append, List.length_map]
    have := paths_length_pos r
    omega

theorem path_length_lt : ∀ (T : DTree α) (p : List Bool), p ∈ T.paths → p.length < T.paths.length
  | .leaf, p, h => by simp [DTree.paths] at h; simp [h, DTree.paths]
  | .node _ r a, p, h => by
    simp only [DTree.paths, List.mem_append, List.mem_map, List.length_append,
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

open scoped Classical in
theorem harvest_good_holds : HarvestGood := by
  intro α _ _ σ _ G rd D _ C S L T ρ θg θgs θgpt hlen hL1 hm ha ha1 hθg0 hθg hφe0 hLmax hexc
    hθgs0 hθgs1 hstart hθgpt0 hθgpt1 hstartpt hφ0 hφ1 hlow hE
  set K : ℝ := 2 * L * (1 + 4 * θg * C.Lmax)
  have hK : 0 < K := by
    have : (1 : ℝ) ≤ L := by exact_mod_cast hL1
    positivity
  set η : ℝ := C.φe * (C.θe / 2 - 2 * θg) / 2
  have hη : 0 ≤ η := by
    have : 0 ≤ C.θe / 2 - 2 * θg := by linarith
    positivity
  set thr : ℝ := Real.log (1 / C.a) * K
  set keys := shortPaths C.Lmax ×ˢ (Finset.univ : Finset α)
  set kf : ℕ → ℕ := fun t => if h : ∃ h', C.n₀ ≤ t ∧ binomSfGe t C.θs h' < C.a then
    (Nat.find h + 1) / 2 else t + 1
  set kp : ℕ → ℕ := fun t => if h : ∃ h', C.n₀ ≤ t ∧ binomSfGe t C.θpt h' < C.a then
    (Nat.find h + 1) / 2 else t + 1
  set hf : ℕ → ℕ := fun n => max ⌈C.φpt * n⌉₊ C.n₀
  set ok : TState α → Prop := fun s => C.φpt / 2 ≤ D.real (ReadModel.searchAt rd C.k s.tree s.edges)
  set Qp : ℕ → TState α → List (FreeMonoid α) → Prop := fun dd s l =>
    EvR G C (fun z => (rd z).cut) S ok dd (kp dd) s l
  set Qh : ℕ → TState α → List (FreeMonoid α) → Prop := fun N s l =>
    EvH C D rd N (hf (N + 1)) s l
  set P : TEnd α → TState α → Prop := fun e s' => G.InClass S s'.tree ∧ e ≠ .tooBig
    ∧ ¬ GoodEnd G e s'
  have hok : ∀ s s' : TState α, s'.tree = s.tree → s'.edges = s.edges → (ok s' ↔ ok s) := by
    intro s s' h1 h2; simp only [ok, h1, h2]
  have hdr : ∀ s s' : TState α, s'.tree = s.tree → s'.edges = s.edges →
      disRate C D rd s' = disRate C D rd s := by
    intro s s' h1 h2; simp only [disRate, h1, h2]
  have hincl : ∀ (l : List (FreeMonoid α)) (s : TState α), Reach G s →
      RunEnds (tallyStep C fun z => (rd z).cut) P s l →
      (∃ e ∈ keys, EvE G C (fun z => (rd z).cut) S θg η e thr s l)
        ∨ (∃ j < l.length, EvS G C (fun z => (rd z).cut) S j s (kf (s.probes + j + 1)) l)
        ∨ (∃ i dd, i < l.length ∧ dd ≤ l.length
          ∧ Skip C (fun z => (rd z).cut) (Qp dd) i s l)
        ∨ (∃ dd, s.dis ≤ dd ∧ dd - s.dis ≤ l.length
          ∧ EvR G C (fun z => (rd z).cut) S ok (dd - s.dis) (kp dd - goodCount G s.pt) s l)
        ∨ (∃ i N, i < l.length ∧ N < l.length ∧ Skip C (fun z => (rd z).cut) (Qh N) i s l)
        ∨ ∃ N, N < l.length ∧ EvH C D rd N (hf (s.n + N + 1) - s.dis) s l := by
    intro l
    induction l with
    | nil => intro s _ h; exact h.elim
    | cons x xs ih =>
      intro s hs h
      simp only [RunEnds] at h
      rcases hst : tallyStep C (fun z => (rd z).cut) s x with s' | ⟨e, s''⟩ <;> rw [hst] at h
      · rcases ih s' (reach_step G C _ hm hs hst).1 h with ⟨e, he, hE'⟩ | ⟨j, hj, hS⟩ |
          ⟨i, dd, hil, hD, hk⟩ | ⟨dd, hD1, hD2, hP⟩ | ⟨i, N, hi, hN, hk⟩ | ⟨N, hN, hH⟩
        · exact .inl ⟨e, he, .inr (by simp only [hst]; exact hE')⟩
        · refine .inr (.inl ⟨j + 1, by simp; omega, ?_⟩)
          simp only [EvS, hst]
          have hp := (tallyStep_counts C _ hst).2.2.2
          rw [(tallyPre_charge C _ s x).2.2.2] at hp
          rw [hp] at hS
          rw [show s.probes + (j + 1) + 1 = s.probes + 1 + j + 1 by omega]
          exact hS
        · refine .inr (.inr (.inl ⟨i + 1, dd, by simp; omega, by simp; omega, ?_⟩))
          simp only [Skip, hst]
          exact hk
        · have hne := evR_ne_nil G C _ S hP
          have hlen1 : 1 ≤ xs.length := by
            rcases xs with _ | _
            · exact absurd rfl hne
            · simp
          rcases tallyStep_fresh C _ hst with ⟨hn0, hd0, hpt0⟩ | ⟨hpre, hn, htr', hed'⟩
          · refine .inr (.inr (.inl ⟨1, dd, by simp; omega, by simp; omega, ?_⟩))
            simp only [Skip, hst, Qp]
            rw [hd0, hpt0] at hP
            simpa [goodCount] using hP
          · subst hpre
            rcases tallyPre_stretch C _ s x with ⟨-, hpt, hdis⟩ | ⟨hn', -, -, -⟩
            swap
            · exact absurd (hn'.symm.trans hn) (by omega)
            have hgc : goodCount G s.pt ≤ goodCount G
                (tallyPre C (fun z => (rd z).cut) s x).pt := by
              rw [hpt, goodCount_append]; omega
            have hcl : G.InClass S s.tree ∧ ok s := by
              rcases xs with _ | ⟨y, ys⟩
              · exact hP.elim
              · obtain ⟨h1, h2⟩ := hP.1
                exact ⟨htr' ▸ h1, (hok s _ htr' hed').1 h2⟩
            have hdt : (tallyPre C (fun z => (rd z).cut) s x).dis - s.dis ≤ 1 := by
              rw [hdis]; split_ifs <;> omega
            refine .inr (.inr (.inr (.inl ⟨dd, by rw [hdis] at hD1; omega, by simp; omega, ?_⟩)))
            simp only [EvR]
            refine ⟨hcl, by omega, .inr ?_⟩
            rw [hst]
            refine ⟨by omega, ?_⟩
            rwa [show dd - s.dis - ((tallyPre C (fun z => (rd z).cut) s x).dis - s.dis)
                = dd - (tallyPre C (fun z => (rd z).cut) s x).dis by omega,
              show kp dd - goodCount G s.pt - (goodCount G
                (tallyPre C (fun z => (rd z).cut) s x).pt - goodCount G s.pt)
                = kp dd - goodCount G (tallyPre C (fun z => (rd z).cut) s x).pt by omega]
        · refine .inr (.inr (.inr (.inr (.inl ⟨i + 1, N, by simp; omega, by simp; omega, ?_⟩))))
          simp only [Skip, hst]
          exact hk
        · have hne : xs ≠ [] := by
            rintro rfl; rcases N with _ | N <;> exact hH.elim
          have hlen1 : 1 ≤ xs.length := by
            rcases xs with _ | _
            · exact absurd rfl hne
            · simp
          rcases tallyStep_fresh C _ hst with ⟨hn0, hd0, -⟩ | ⟨hpre, hn, htr', hed'⟩
          · refine .inr (.inr (.inr (.inr (.inl ⟨1, N, by simp; omega, by simp; omega, ?_⟩))))
            simp only [Skip, hst, Qh]
            rw [hn0, hd0, zero_add, Nat.sub_zero] at hH
            exact hH
          · subst hpre
            have hdR := hdr s _ htr' hed'
            have hlo : disRate C D rd s ≤ C.φpt / 2 := by
              rcases xs with _ | ⟨y, ys⟩
              · exact absurd rfl hne
              · rcases N with _ | N
                · exact hdR ▸ hH.1
                · exact hdR ▸ hH.1
            have hdtv : (tallyPre C (fun z => (rd z).cut) s x).dis = s.dis + dt C rd s x := by
              rcases tallyPre_stretch C _ s x with ⟨-, -, hdis⟩ | ⟨hn', -, -, -⟩
              · simp only [dt]; rw [hdis]; split_ifs <;> omega
              · exact absurd (hn'.symm.trans hn) (by omega)
            refine .inr (.inr (.inr (.inr (.inr ⟨N + 1, by simp; omega, ?_⟩))))
            simp only [EvH]
            refine ⟨hlo, ?_⟩
            rw [hst]
            refine ⟨by omega, ?_⟩
            rwa [show s.n + (N + 1) + 1 = (tallyPre C (fun z => (rd z).cut) s x).n + N + 1 by
              omega, show hf ((tallyPre C (fun z => (rd z).cut) s x).n + N + 1) - s.dis
                - dt C rd s x = hf ((tallyPre C (fun z => (rd z).cut) s x).n + N + 1)
                  - (tallyPre C (fun z => (rd z).cut) s x).dis by omega]
      · obtain ⟨hcl, hne, hng⟩ := h
        obtain ⟨rfl, hl⟩ := tallyStep_inr_pre C _ hst hne
        have htr : (tallyPre C (fun z => (rd z).cut) s x).tree = s.tree :=
          (tallyPre_into _ C x hs.2.1 hs.2.2.1).1
        rw [htr] at hcl
        obtain ⟨-, -, hstH, hprob⟩ := tallyPre_charge C (fun z => (rd z).cut) s x
        rcases e with _ | _ | e | _ | _
        · exact absurd trivial hng
        · refine .inr (.inl ⟨0, by simp, hcl, ?_⟩)
          obtain ⟨hn₀, hlt⟩ := tallyLook_start' C hl
          rw [hprob] at hn₀ hlt
          have hn₀' : C.n₀ ≤ s.probes + 0 + 1 := by simpa using hn₀
          have hlt' : binomSfGe (s.probes + 0 + 1) C.θs
              (tallyPre C (fun z => (rd z).cut) s x).startH.length < C.a := by simpa using hlt
          have hex : ∃ h', C.n₀ ≤ s.probes + 0 + 1 ∧ binomSfGe (s.probes + 0 + 1) C.θs h' < C.a :=
            ⟨_, hn₀', hlt'⟩
          simp only [kf, dif_pos hex]
          have hfind : Nat.find hex ≤ (tallyPre C (fun z => (rd z).cut) s x).startH.length :=
            Nat.find_min' hex ⟨hn₀', hlt'⟩
          have hsum := goodCount_add_badCount G (tallyPre C (fun z => (rd z).cut) s x).startH
          simp only [GoodEnd, not_lt] at hng
          omega
        · left
          obtain ⟨hp, hgate, hexc'⟩ := tallyLook_harvest' C hl
          rw [htr] at hp
          refine ⟨e, Finset.mem_product.2 ⟨mem_shortPaths ?_, Finset.mem_univ _⟩, .inl ⟨hcl, ?_⟩⟩
          · have := path_length_lt _ _ hp
            have := class_paths G C S hcl hLmax
            omega
          · simp only [GoodEnd, not_lt] at hng
            set H := (tallyPre C (fun z => (rd z).cut) s x).harv e.1 e.2
            set Tr : ℝ := ((tallyPre C (fun z => (rd z).cut) s x).trav e.1 e.2 : ℝ)
            set Pr : ℝ := ((tallyPre C (fun z => (rd z).cut) s x).probes : ℝ)
            have hsum := goodCount_add_badCount G H
            have hb : (G.badCount H : ℝ) * 2 ≤ H.length := by
              exact_mod_cast (by omega : _ * 2 ≤ _)
            have hs' : (goodCount G H : ℝ) + G.badCount H = H.length := by
              exact_mod_cast hsum
            have hx := hexc ((tallyPre C (fun z => (rd z).cut) s x).trav e.1 e.2)
            have hr0 : (0 : ℝ) ≤ Tr := Nat.cast_nonneg _
            have hP0 : (0 : ℝ) ≤ Pr := Nat.cast_nonneg _
            have hgap : 0 ≤ C.θe / 2 - 2 * θg := by linarith
            have hgt : 2 * η * Pr ≤ (C.θe / 2 - 2 * θg) * Tr := by
              simp only [η]
              nlinarith [mul_le_mul_of_nonneg_left hgate hgap]
            simp only [edgeZ, thr, K]
            have : Real.log (1 / C.a) * (2 * L * (1 + 4 * θg * C.Lmax))
                = 2 * L * (1 + 4 * θg * C.Lmax) * Real.log (1 / C.a) := by ring
            rw [this]
            nlinarith
        · obtain ⟨hgate, hn₀, hlt⟩ := tallyLook_pt' C hl
          rcases tallyPre_stretch C _ s x with ⟨hn, hpt, hdis⟩ | ⟨-, hpt0, -, -⟩
          · by_cases hk : ok s
            · set dd := (tallyPre C (fun z => (rd z).cut) s x).dis
              have hex : ∃ h', C.n₀ ≤ dd ∧ binomSfGe dd C.θpt h' < C.a := ⟨_, hn₀, hlt⟩
              have hfind : Nat.find hex ≤ (tallyPre C (fun z => (rd z).cut) s x).pt.length :=
                Nat.find_min' hex ⟨hn₀, hlt⟩
              have hsum := goodCount_add_badCount G (tallyPre C (fun z => (rd z).cut) s x).pt
              simp only [GoodEnd, not_lt] at hng
              have hkp : kp dd ≤ goodCount G (tallyPre C (fun z => (rd z).cut) s x).pt := by
                simp only [kp, dif_pos hex]; omega
              have hdt : dd - s.dis ≤ 1 := by rw [hdis]; split_ifs <;> omega
              refine .inr (.inr (.inr (.inl ⟨dd, by rw [hdis]; omega, by simp; omega, ?_⟩)))
              simp only [EvR]
              exact ⟨⟨hcl, hk⟩, le_rfl, .inl ⟨rfl, by omega⟩⟩
            · have hdtv : (tallyPre C (fun z => (rd z).cut) s x).dis = s.dis + dt C rd s x := by
                simp only [dt]; rw [hdis]; split_ifs <;> omega
              have hceil : ⌈C.φpt * ((tallyPre C (fun z => (rd z).cut) s x).n : ℝ)⌉₊
                  ≤ (tallyPre C (fun z => (rd z).cut) s x).dis := Nat.ceil_le.2 hgate
              refine .inr (.inr (.inr (.inr (.inr ⟨0, by simp, ?_⟩))))
              simp only [EvH]
              refine ⟨le_of_lt (not_le.1 hk), ?_⟩
              have hH : hf (s.n + 0 + 1) ≤ (tallyPre C (fun z => (rd z).cut) s x).dis := by
                simp only [hf]
                rw [show s.n + 0 + 1 = (tallyPre C (fun z => (rd z).cut) s x).n by omega]
                exact max_le hceil hn₀
              omega
          · rw [hpt0, List.length_nil, binomSfGe_zero_right] at hlt
            exact absurd hlt (not_lt.2 ha1)
        · exact absurd rfl hne
  have hsub : {xs : Fin T → FreeMonoid α | RunEnds (tallyStep C fun z => (rd z).cut) P tallyStart
      (List.ofFn xs)}
      ⊆ (((⋃ e ∈ keys, {xs | EvE G C (fun z => (rd z).cut) S θg η e thr tallyStart
          (List.ofFn xs)})
        ∪ ⋃ j ∈ Finset.range T, {xs | EvS G C (fun z => (rd z).cut) S j tallyStart
          (goodCount G (tallyStart : TState α).startH + kf (j + 1)) (List.ofFn xs)})
        ∪ ⋃ q ∈ Finset.range T ×ˢ Finset.range (T + 1),
          {xs | Skip C (fun z => (rd z).cut) (Qp q.2) q.1 tallyStart (List.ofFn xs)})
        ∪ ⋃ q ∈ Finset.range T ×ˢ Finset.range T,
          {xs | Skip C (fun z => (rd z).cut) (Qh q.2) q.1 tallyStart (List.ofFn xs)} := by
    intro xs hxs
    rcases hincl _ _ (reach_start G) hxs with ⟨e, he, h⟩ | ⟨j, hj, h⟩ | ⟨i, dd, hi, hD, h⟩ |
      ⟨dd, -, hD, h⟩ | ⟨i, N, hi, hN, h⟩ | ⟨N, hN, h⟩
    · exact .inl (.inl (.inl (Set.mem_biUnion he h)))
    · refine .inl (.inl (.inr (Set.mem_biUnion (Finset.mem_range.2 (by simpa using hj)) ?_)))
      simpa [tallyStart, goodCount] using h
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
          simp [tallyStart] at hD; omega)⟩) ?_))
      simp only [Skip, Qp]
      simpa [tallyStart, goodCount] using h
    · simp only [List.length_ofFn] at hi hN
      exact .inr (Set.mem_biUnion (x := (i, N))
        (Finset.mem_product.2 ⟨Finset.mem_range.2 hi, Finset.mem_range.2 hN⟩) h)
    · simp only [List.length_ofFn] at hN
      refine .inr (Set.mem_biUnion (x := (0, N))
        (Finset.mem_product.2 ⟨Finset.mem_range.2 (by omega), Finset.mem_range.2 hN⟩) ?_)
      simp only [Skip, Qh]
      simpa [tallyStart] using h
  refine (measure_mono hsub).trans ((measure_union_le _ _).trans ?_)
  refine (add_le_add ((measure_union_le _ _).trans (add_le_add (measure_union_le _ _) le_rfl))
    le_rfl).trans ?_
  have h1 : ∀ e ∈ keys, (Measure.pi fun _ : Fin T => D)
      {xs | EvE G C (fun z => (rd z).cut) S θg η e thr tallyStart (List.ofFn xs)}
        ≤ ENNReal.ofReal C.a := by
    intro e _
    refine (edge_super G C S hm D rd hlen hL1 hθg0 hLmax hη hE e thr T tallyStart
      (reach_start G)).trans (le_of_eq ?_)
    have hz : edgeZ G θg η e (tallyStart : TState α) = 0 := by
      simp [edgeZ, tallyStart, goodCount]
    rw [hz, show (0 - thr) / K = Real.log C.a by
      simp only [thr]; field_simp; rw [one_div, Real.log_inv]; ring, Real.exp_log ha]
  have h2 : ∀ j ∈ Finset.range T, (Measure.pi fun _ : Fin T => D)
      {xs | EvS G C (fun z => (rd z).cut) S j tallyStart
        (goodCount G (tallyStart : TState α).startH + kf (j + 1)) (List.ofFn xs)}
      ≤ ENNReal.ofReal C.a := by
    intro j _
    refine (start_bin G C S hm D rd hθgs0 hθgs1 hE T j tallyStart _ (reach_start G)).trans
      (ENNReal.ofReal_le_ofReal ?_)
    simp only [kf]
    split_ifs with hex
    · exact hstart (j + 1) _ (Nat.find_spec hex).1 (Nat.find_spec hex).2
    · rw [binomSfGe_gt _ (by omega)]; exact ha.le
  have h3 : ∀ q ∈ Finset.range T ×ˢ Finset.range (T + 1), (Measure.pi fun _ : Fin T => D)
      {xs | Skip C (fun z => (rd z).cut) (Qp q.2) q.1 tallyStart (List.ofFn xs)}
      ≤ if q.2 = 0 then 0 else ENNReal.ofReal C.a := by
    intro q _
    refine skip_le G C _ hm D (Qp q.2) (fun T' s hs => ?_) T q.1 tallyStart (reach_start G)
    refine (ptR_bin G C S hm D rd hθgpt0 hθgpt1 hE T' q.2 _ s hs).trans ?_
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
      · exact hstartpt q.2 _ (Nat.find_spec hex).1 (Nat.find_spec hex).2
      · rw [binomSfGe_gt _ (by omega)]; exact ha.le
  have h4 : ∀ q ∈ Finset.range T ×ˢ Finset.range T, (Measure.pi fun _ : Fin T => D)
      {xs | Skip C (fun z => (rd z).cut) (Qh q.2) q.1 tallyStart (List.ofFn xs)}
      ≤ ENNReal.ofReal C.a := by
    intro q _
    refine skip_le G C _ hm D (Qh q.2) (fun T' s _ => ?_) T q.1 tallyStart (reach_start G)
    refine (evH_le C D rd hφ0 hφ1 T' q.2 _ s).trans (ENNReal.ofReal_le_ofReal ?_)
    have hp0 : 0 ≤ C.φpt / 2 := by linarith
    have hp1 : C.φpt / 2 ≤ 1 := by linarith
    by_cases hn : C.n₀ ≤ q.2 + 1
    · refine (binomSfGe_antitone' hp0 hp1 (le_max_left _ _)).trans ?_
      have := hlow (q.2 + 1) hn
      push_cast at this ⊢
      exact this
    · rw [binomSfGe_gt _ (lt_of_lt_of_le (not_le.1 hn) (le_max_right _ _))]
      exact ha.le
  have hsum3 : ∑ q ∈ Finset.range T ×ˢ Finset.range (T + 1),
      (if q.2 = 0 then 0 else ENNReal.ofReal C.a) = T * T * ENNReal.ofReal C.a := by
    have h : ∑ y ∈ Finset.range (T + 1), (if y = 0 then (0 : ENNReal) else ENNReal.ofReal C.a)
        = T * ENNReal.ofReal C.a := by
      rw [Finset.sum_range_succ']
      simp
    rw [Finset.sum_product (f := fun q : ℕ × ℕ => if q.2 = 0 then (0 : ENNReal)
      else ENNReal.ofReal C.a)]
    simp only [h, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    ring
  calc _ ≤ ∑ e ∈ keys, ENNReal.ofReal C.a + ∑ j ∈ Finset.range T, ENNReal.ofReal C.a
        + ∑ q ∈ Finset.range T ×ˢ Finset.range (T + 1),
          (if q.2 = 0 then 0 else ENNReal.ofReal C.a)
        + ∑ _q ∈ Finset.range T ×ˢ Finset.range T, ENNReal.ofReal C.a :=
        add_le_add (add_le_add (add_le_add
          ((measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum h1))
          ((measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum h2)))
          ((measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum h3)))
          ((measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum h4))
    _ = (keys.card + T + 2 * (T * T)) * ENNReal.ofReal C.a := by
        rw [hsum3]
        simp only [Finset.sum_const, Finset.card_range, Finset.card_product, nsmul_eq_mul]
        push_cast
        ring
    _ ≤ ENNReal.ofReal ((2 ^ (C.Lmax + 1) * Fintype.card α + T + 2 * (T * T)) * C.a) := by
        rw [ENNReal.ofReal_mul (by positivity)]
        gcongr
        rw [show ((2 : ℝ) ^ (C.Lmax + 1) * Fintype.card α + T + 2 * (T * T)) = ((2 ^ (C.Lmax + 1)
          * Fintype.card α + T + 2 * (T * T) : ℕ) : ℝ) by push_cast; ring,
          ENNReal.ofReal_natCast]
        have : keys.card ≤ 2 ^ (C.Lmax + 1) * Fintype.card α := by
          simp only [keys, Finset.card_product, Finset.card_univ]
          exact Nat.mul_le_mul_right _ (shortPaths_card _)
        exact_mod_cast (by omega : keys.card + T + 2 * (T * T)
          ≤ 2 ^ (C.Lmax + 1) * Fintype.card α + T + 2 * (T * T))

end Assemble

end OrthoDFA
