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
      ∧ (tallyPre C cut s x).reads
        = (fun p c => s.reads p c + edgeReadsBy cut s.tree s.edges C.k x (p, c))
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
    s'.harv = s.harv ∧ s'.reads = s.reads ∧ s'.startH = s.startH ∧ s'.probes = s.probes := by
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
    s'.harv = (tallyPre C cut s x).harv ∧ s'.reads = (tallyPre C cut s x).reads
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

/-- At the edge `e`: good read-states' undecided reads less `2θg` of its reads. -/
noncomputable def edgeZ (θg : ℝ) (e : List Bool × α) (s : TState α) : ℝ :=
  goodCount G (s.harv e.1 e.2) - 2 * θg * s.reads e.1 e.2

/-- At some probe from a state of the class, the edge's count reaches `thr`. -/
def EvE (θg : ℝ) (e : List Bool × α) (thr : ℝ) : TState α → List (FreeMonoid α) → Prop
  | _, [] => False
  | s, x :: xs => (G.InClass S s.tree ∧ thr ≤ edgeZ G θg e (tallyPre C cut s x))
    ∨ match tallyStep C cut s x with
      | .inl s' => EvE θg e thr s' xs
      | .inr _ => False

theorem evE_notClass (hm : 0 < C.m) (θg : ℝ) (e : List Bool × α) (thr : ℝ) :
    ∀ (l : List (FreeMonoid α)) (s : TState α), Reach G s → ¬ G.InClass S s.tree →
      ¬ EvE G C cut S θg e thr s l
  | [], _, _, _ => id
  | x :: xs, s, hs, hn => by
    simp only [EvE, hn, false_and, false_or]
    rcases h : tallyStep C cut s x with s' | _
    · exact evE_notClass hm θg e thr xs s' (reach_step G C cut hm hs h).1
        (notClass_step G C cut S hm hs hn h)
    · exact id

/-- One probe's exponentiated increment averages at most `1`. -/
theorem mgf_step {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
    (D : Measure X) [IsProbabilityMeasure D] (U N : X → ℕ) {θ L R : ℝ} (hθ : 0 ≤ θ)
    (hL : 0 < L) (hR : 0 ≤ R) (hb : ∀ᵐ x ∂D, (U x : ℝ) ≤ L ∧ (N x : ℝ) ≤ L * R)
    (hmean : ∫ x, (U x : ℝ) ∂D ≤ θ * ∫ x, (N x : ℝ) ∂D) :
    ∫⁻ x, ENNReal.ofReal (Real.exp ((U x - 2 * θ * N x) / (2 * L * (1 + 4 * θ * R)))) ∂D
      ≤ 1 := by
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
  calc ∫⁻ x, ENNReal.ofReal (Real.exp ((U x - 2 * θ * N x) / (2 * L * (1 + 4 * θ * R)))) ∂D
      ≤ ∫⁻ x, ENNReal.ofReal (g x) ∂D := lintegral_mono_ae (by
        filter_upwards [hpt] with x hx using ENNReal.ofReal_le_ofReal hx)
    _ = ENNReal.ofReal (∫ x, g x ∂D) := by
        rw [ofReal_integral_eq_lintegral_ofReal hgi]
        filter_upwards [hpt] with x hx using (Real.exp_pos _).le.trans hx
    _ ≤ 1 := by
        rw [← ENNReal.ofReal_one]
        refine ENNReal.ofReal_le_ofReal ?_
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
        nlinarith [mul_nonneg (mul_nonneg hθ hl.le) hEN]

theorem edgeZ_pre (θg : ℝ) (e : List Bool × α) (s : TState α) (x : FreeMonoid α) :
    edgeZ G θg e (tallyPre C cut s x) = edgeZ G θg e s
      + (goodCount G (edgeHarvBy cut s.tree s.edges C.k x e)
        - 2 * θg * edgeReadsBy cut s.tree s.edges C.k x e) := by
  obtain ⟨h1, h2, -, -⟩ := tallyPre_charge C cut s x
  simp only [edgeZ, h1, h2, goodCount_append]
  push_cast; ring

theorem edgeZ_step {θg : ℝ} {e : List Bool × α} {s s' : TState α} {x : FreeMonoid α}
    (h : tallyStep C cut s x = .inl s') :
    edgeZ G θg e s' = edgeZ G θg e (tallyPre C cut s x) := by
  obtain ⟨h1, h2, -, -⟩ := tallyStep_counts C cut h
  simp only [edgeZ, h1, h2]

/-- The edge's exponentiated count is a supermartingale while the tree is in the class. -/
theorem edge_super (hm : 0 < C.m) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (rd : FreeMonoid α → ARU) {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hL : 1 ≤ L)
    {ρ θg θgs θgpt : ℝ} (hθg : 0 ≤ θg) (hLmax : Fintype.card σ + S + 2 ≤ C.Lmax)
    (hE : TallyE G D C S ρ θg θgs θgpt rd) (e : List Bool × α) (thr : ℝ) :
    ∀ (T : ℕ) (s : TState α), Reach G s →
      (Measure.pi fun _ : Fin T => D)
          {xs | EvE G C (fun z => (rd z).cut) S θg e thr s (List.ofFn xs)}
        ≤ ENNReal.ofReal (Real.exp ((edgeZ G θg e s - thr)
          / (2 * L * (1 + 4 * θg * C.Lmax)))) := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
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
          {xs : Fin (T + 1) → FreeMonoid α | EvE G C cut S θg e thr s (List.ofFn xs)}} = 0 := by
        intro x
        have hempty : {xs | Fin.cons x xs ∈
            {xs : Fin (T + 1) → FreeMonoid α | EvE G C cut S θg e thr s (List.ofFn xs)}} = ∅ := by
          ext xs
          simp only [Set.mem_ofPred_eq, List.ofFn_cons, Set.mem_empty_iff_false, iff_false]
          exact evE_notClass G C cut S hm θg e thr _ s hs hc
        rw [hempty, measure_empty]
      simp only [this, lintegral_zero]
      exact zero_le
    have hsec : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
        {xs : Fin (T + 1) → FreeMonoid α | EvE G C cut S θg e thr s (List.ofFn xs)}}
        ≤ ENNReal.ofReal (Real.exp ((edgeZ G θg e (tallyPre C cut s x) - thr) / K)) := by
      intro x
      by_cases hz : thr ≤ edgeZ G θg e (tallyPre C cut s x)
      · refine prob_le_one.trans ?_
        rw [← ENNReal.ofReal_one]
        exact ENNReal.ofReal_le_ofReal (Real.one_le_exp (div_nonneg (by linarith) hK.le))
      rcases hst : tallyStep C cut s x with s' | ⟨e', s'⟩
      · have : {xs | Fin.cons x xs ∈
            {xs : Fin (T + 1) → FreeMonoid α | EvE G C cut S θg e thr s (List.ofFn xs)}}
            = {xs | EvE G C cut S θg e thr s' (List.ofFn xs)} := by
          ext xs
          simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvE, hz, and_false, false_or, hst]
        rw [this, ← edgeZ_step G C cut hst]
        exact ih s' (reach_step G C cut hm hs hst).1
      · have : {xs | Fin.cons x xs ∈
            {xs : Fin (T + 1) → FreeMonoid α | EvE G C cut S θg e thr s (List.ofFn xs)}} = ∅ := by
          ext xs
          simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvE, hz, and_false, false_or, hst,
            Set.mem_empty_iff_false]
        rw [this, measure_empty]
        exact zero_le
    refine (lintegral_mono hsec).trans ?_
    have hfac : ∀ x, ENNReal.ofReal (Real.exp ((edgeZ G θg e (tallyPre C cut s x) - thr) / K))
        = ENNReal.ofReal (Real.exp ((edgeZ G θg e s - thr) / K))
          * ENNReal.ofReal (Real.exp ((goodCount G (edgeHarvBy cut s.tree s.edges C.k x e)
            - 2 * θg * edgeReadsBy cut s.tree s.edges C.k x e) / K)) := by
      intro x
      rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add, edgeZ_pre]
      congr 2
      ring
    simp only [hfac]
    rw [lintegral_const_mul _ (measurable_of_countable _)]
    refine (mul_le_mul_right (mgf_step D
      (fun x => goodCount G (edgeHarvBy cut s.tree s.edges C.k x e))
      (fun x => edgeReadsBy cut s.tree s.edges C.k x e) hθg (by positivity) (by positivity) ?_
      ?_) _).trans (le_of_eq (mul_one _))
    · have hpaths := class_paths G C S hc hLmax
      filter_upwards [hlen] with x hx
      constructor
      · have h1 : goodCount G (edgeHarvBy cut s.tree s.edges C.k x e)
            ≤ (edgeHarvBy cut s.tree s.edges C.k x e).length := List.length_filter_le _ _
        have h2 := edgeHarvBy_length_le cut (t := s.tree) (edges := s.edges) (k := C.k) (x := x) e
        exact_mod_cast h1.trans (h2.trans hx)
      · have h1 := edgeReadsBy_le cut (t := s.tree) (edges := s.edges) (k := C.k) (x := x) e
        have : edgeReadsBy cut s.tree s.edges C.k x e ≤ L * C.Lmax :=
          h1.trans (Nat.mul_le_mul hx hpaths)
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

/-- At the `len`-th probe after this one, within the stretch, from a state of the class, the good
read-states' undecided middles searches stopped at reach `k`. -/
def EvP : ℕ → TState α → ℕ → List (FreeMonoid α) → Prop
  | _, _, _, [] => False
  | 0, s, k, x :: _ => G.InClass S s.tree ∧ k ≤ goodCount G (tallyPre C cut s x).pt
  | len + 1, s, k, x :: xs => match tallyStep C cut s x with
    | .inl s' => s'.n ≠ 0 ∧ EvP len s' k xs
    | .inr _ => False

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

theorem tallyPre_stretch (s : TState α) (x : FreeMonoid α) :
    ((tallyPre C cut s x).n = s.n + 1
      ∧ (tallyPre C cut s x).pt = s.pt ++ ptHarvBy cut s.tree s.edges C.k x)
    ∨ ((tallyPre C cut s x).n = 0 ∧ (tallyPre C cut s x).pt = []) := by
  unfold tallyPre
  split
  · split
    · split
      · right; simp [TState.setEdge, TState.fresh]
      · left; simp [TState.charge]
    · left; simp [TState.charge]
  · left; split <;> simp [TState.charge, TState.addRec]
  all_goals left; simp [TState.charge]

open scoped Classical in
theorem goodCount_ptPre (rd : FreeMonoid α → ARU) (s : TState α) (x : FreeMonoid α) :
    goodCount G (tallyPre C (fun z => (rd z).cut) s x).pt ≤ goodCount G s.pt
      + if x ∈ G.ptAt rd C.k s.tree s.edges G.Good then 1 else 0 := by
  have h : (tallyPre C (fun z => (rd z).cut) s x).pt = s.pt ++ ptHarvBy (fun z => (rd z).cut)
      s.tree s.edges C.k x ∨ (tallyPre C (fun z => (rd z).cut) s x).pt = [] :=
    (tallyPre_stretch C _ s x).imp (·.2) (·.2)
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

theorem evP_mono : ∀ (l : List (FreeMonoid α)) (len : ℕ) (s : TState α) {k k' : ℕ}, k ≤ k' →
    EvP G C cut S len s k' l → EvP G C cut S len s k l
  | [], len, _, _, _, _, h => by cases len <;> simp [EvP] at h
  | x :: xs, 0, s, _, _, hk, h => ⟨h.1, hk.trans h.2⟩
  | x :: xs, len + 1, s, _, _, hk, h => by
    simp only [EvP] at h ⊢
    rcases hst : tallyStep C cut s x with s' | _ <;> rw [hst] at h
    · exact ⟨h.1, evP_mono xs len s' hk h.2⟩
    · exact h

theorem evP_notClass (hm : 0 < C.m) :
    ∀ (l : List (FreeMonoid α)) (len : ℕ) (s : TState α) (k : ℕ), Reach G s →
      ¬ G.InClass S s.tree → ¬ EvP G C cut S len s k l
  | [], len, _, _, _, _ => by cases len <;> simp [EvP]
  | x :: xs, 0, s, k, _, hn => fun h => hn h.1
  | x :: xs, len + 1, s, k, hs, hn => by
    simp only [EvP]
    rcases h : tallyStep C cut s x with s' | _
    · exact fun h' => evP_notClass hm xs len s' k (reach_step G C cut hm hs h).1
        (notClass_step G C cut S hm hs hn h) h'.2
    · exact id

/-- The good read-states' undecided middles within a stretch, `len + 1` probes on, are at most
binomial. -/
theorem pt_bin (hm : 0 < C.m) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (rd : FreeMonoid α → ARU) {ρ θg θgs θgpt : ℝ} (hθ0 : 0 ≤ θgpt) (hθ1 : θgpt ≤ 1)
    (hE : TallyE G D C S ρ θg θgs θgpt rd) :
    ∀ (T len : ℕ) (s : TState α) (k : ℕ), Reach G s →
      (Measure.pi fun _ : Fin T => D)
          {xs | EvP G C (fun z => (rd z).cut) S len s (goodCount G s.pt + k) (List.ofFn xs)}
        ≤ ENNReal.ofReal (binomSfGe (len + 1) θgpt k) := by
  classical
  intro T
  induction T with
  | zero => intro len s k _; cases len <;> simp [EvP]
  | succ T ih =>
    intro len s k hs
    rcases k with _ | k
    · rw [binomSfGe_zero_right, ENNReal.ofReal_one]; exact prob_le_one
    by_cases hc : G.InClass S s.tree
    swap
    · have : {xs : Fin (T + 1) → FreeMonoid α |
          EvP G C (fun z => (rd z).cut) S len s (goodCount G s.pt + (k + 1)) (List.ofFn xs)}
          = ∅ := by
        ext xs
        simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
        exact evP_notClass G C _ S hm _ len s _ hs hc
      rw [this, measure_empty]; exact zero_le
    set A := G.ptAt rd C.k s.tree s.edges G.Good
    have hA : D.real A ≤ θgpt := hE.goodPT s.tree s.edges hc hs.2.1
    have hA0 : 0 ≤ D.real A := measureReal_nonneg
    have hgc := fun x => goodCount_ptPre G C rd s x
    rw [pi_succ_apply]
    rcases len with _ | len
    · have hsec : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
          {xs : Fin (T + 1) → FreeMonoid α |
            EvP G C (fun z => (rd z).cut) S 0 s (goodCount G s.pt + (k + 1)) (List.ofFn xs)}}
          ≤ A.indicator (fun _ => if k = 0 then 1 else 0) x := by
        intro x
        have h1 := hgc x
        by_cases hx : x ∈ A
        · rw [Set.indicator_of_mem hx]
          rcases k with _ | k
          · exact prob_le_one
          · rw [if_pos (show x ∈ G.ptAt rd C.k s.tree s.edges G.Good from hx)] at h1
            have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
                EvP G C (fun z => (rd z).cut) S 0 s (goodCount G s.pt + (k + 1 + 1))
                  (List.ofFn xs)}} = ∅ := by
              ext xs
              simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvP, Set.mem_empty_iff_false,
                iff_false, not_and]
              intro _; omega
            rw [this, measure_empty]; simp
        · rw [Set.indicator_of_notMem hx]
          rw [if_neg (show x ∉ G.ptAt rd C.k s.tree s.edges G.Good from hx)] at h1
          have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
              EvP G C (fun z => (rd z).cut) S 0 s (goodCount G s.pt + (k + 1))
                (List.ofFn xs)}} = ∅ := by
            ext xs
            simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvP, Set.mem_empty_iff_false,
              iff_false, not_and]
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
    · set c₁ := binomSfGe (len + 1) θgpt k
      set c₀ := binomSfGe (len + 1) θgpt (k + 1)
      have hc₀ : 0 ≤ c₀ := binomSfGe_nonneg hθ0 hθ1 _
      have hc₀₁ : c₀ ≤ c₁ := binomSfGe_antitone hθ0 hθ1 k
      have hsec : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
          {xs : Fin (T + 1) → FreeMonoid α |
            EvP G C (fun z => (rd z).cut) S (len + 1) s (goodCount G s.pt + (k + 1))
              (List.ofFn xs)}}
          ≤ A.indicator (fun _ => ENNReal.ofReal c₁) x
            + Aᶜ.indicator (fun _ => ENNReal.ofReal c₀) x := by
        intro x
        rcases hst : tallyStep C (fun z => (rd z).cut) s x with s' | ⟨e, s'⟩
        · rcases tallyStep_fresh C _ hst with ⟨h0, -, -⟩ | ⟨rfl, -, -, -⟩
          · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
                EvP G C (fun z => (rd z).cut) S (len + 1) s (goodCount G s.pt + (k + 1))
                  (List.ofFn xs)}} = ∅ := by
              ext xs
              simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvP, hst, h0, ne_eq,
                not_true_eq_false, false_and, Set.mem_empty_iff_false]
            rw [this, measure_empty]; exact zero_le
          have hs' := (reach_step G C _ hm hs hst).1
          have h1 := hgc x
          by_cases hx : x ∈ A
          · rw [Set.indicator_of_mem hx, Set.indicator_of_notMem (by simpa using hx), add_zero]
            rw [if_pos (show x ∈ G.ptAt rd C.k s.tree s.edges G.Good from hx)] at h1
            refine le_trans (measure_mono fun xs hxs => ?_) (ih len _ k hs')
            simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvP, hst] at hxs ⊢
            exact evP_mono G C _ S _ len _ (by omega) hxs.2
          · rw [Set.indicator_of_notMem hx, Set.indicator_of_mem (by simpa using hx), zero_add]
            rw [if_neg (show x ∉ G.ptAt rd C.k s.tree s.edges G.Good from hx)] at h1
            refine le_trans (measure_mono fun xs hxs => ?_) (ih len _ (k + 1) hs')
            simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvP, hst] at hxs ⊢
            exact evP_mono G C _ S _ len _ (by omega) hxs.2
        · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
              EvP G C (fun z => (rd z).cut) S (len + 1) s (goodCount G s.pt + (k + 1))
                (List.ofFn xs)}} = ∅ := by
            ext xs
            simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvP, hst, Set.mem_empty_iff_false]
          rw [this, measure_empty]; exact zero_le
      refine (lintegral_mono hsec).trans ?_
      have hc₁ : 0 ≤ c₁ := hc₀.trans hc₀₁
      rw [lintegral_add_left (measurable_of_countable _), lintegral_indicator
          (Set.to_countable _).measurableSet, lintegral_indicator
          (Set.to_countable _).measurableSet,
        setLIntegral_const, setLIntegral_const, ← ofReal_measureReal, ← ofReal_measureReal,
        measureReal_compl (Set.to_countable _).measurableSet, probReal_univ,
        ← ENNReal.ofReal_mul hc₁, ← ENNReal.ofReal_mul hc₀,
        ← ENNReal.ofReal_add (mul_nonneg hc₁ hA0) (mul_nonneg hc₀ (by linarith))]
      refine ENNReal.ofReal_le_ofReal ?_
      rw [show len + 1 + 1 = (len + 1) + 1 from rfl, binomSfGe_succ]
      change c₁ * D.real A + c₀ * (1 - D.real A) ≤ θgpt * c₁ + (1 - θgpt) * c₀
      nlinarith

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
    e.1 ∈ s.tree.paths ∧ C.exc (s.reads e.1 e.2)
      ≤ ((s.harv e.1 e.2).length : ℝ) - C.θe * s.reads e.1 e.2 := by
  unfold tallyLook at h
  by_cases h1 : rateSide C.θs C.a C.n₀ s.probes s.startH.length = some true
  · rw [if_pos h1] at h; simp at h
  · rw [if_neg h1] at h
    by_cases h2 : ∃ e : List Bool × α, e.1 ∈ s.tree.paths
        ∧ C.exc (s.reads e.1 e.2) ≤ ((s.harv e.1 e.2).length : ℝ) - C.θe * s.reads e.1 e.2
    · rw [dif_pos h2] at h
      simp only [Option.some.injEq, TEnd.harvest.injEq] at h
      rw [← h]
      exact h2.choose_spec
    · rw [dif_neg h2] at h
      split_ifs at h <;> simp at h

theorem tallyLook_pt' (C : TallyCfg) {s : TState α} (h : tallyLook C s = some .harvestPT) :
    C.n₀ ≤ s.n ∧ binomSfGe s.n C.θpt s.pt.length < C.a := by
  unfold tallyLook at h
  by_cases h1 : rateSide C.θs C.a C.n₀ s.probes s.startH.length = some true
  · rw [if_pos h1] at h; simp at h
  rw [if_neg h1] at h
  by_cases h2 : ∃ e : List Bool × α, e.1 ∈ s.tree.paths
      ∧ C.exc (s.reads e.1 e.2) ≤ ((s.harv e.1 e.2).length : ℝ) - C.θe * s.reads e.1 e.2
  · rw [dif_pos h2] at h; simp at h
  rw [dif_neg h2] at h
  by_cases h3 : rateSide C.θpt C.a C.n₀ s.n s.pt.length = some true
  · unfold rateSide at h3
    split_ifs at h3 with h4 h5 <;> first | exact ⟨h4, h5⟩ | simp at h3
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
  intro α _ _ σ _ G rd D _ C S L T ρ θg θgs θgpt hlen hL1 hm ha ha1 hθg0 hθg hLmax hexc hθgs0
    hθgs1 hstart hθgpt0 hθgpt1 hstartpt hE
  set K : ℝ := 2 * L * (1 + 4 * θg * C.Lmax)
  have hK : 0 < K := by
    have : (1 : ℝ) ≤ L := by exact_mod_cast hL1
    positivity
  set thr : ℝ := Real.log (1 / C.a) * K
  set keys := shortPaths C.Lmax ×ˢ (Finset.univ : Finset α)
  set kf : ℕ → ℕ := fun t => if h : ∃ h', C.n₀ ≤ t ∧ binomSfGe t C.θs h' < C.a then
    (Nat.find h + 1) / 2 else t + 1
  set kp : ℕ → ℕ := fun t => if h : ∃ h', C.n₀ ≤ t ∧ binomSfGe t C.θpt h' < C.a then
    (Nat.find h + 1) / 2 else t + 1
  set cr : FreeMonoid α → Option Bool := fun z => (rd z).cut
  set Qp : ℕ → TState α → List (FreeMonoid α) → Prop := fun len s l =>
    EvP G C cr S len s (goodCount G s.pt + kp (len + 1)) l
  set P : TEnd α → TState α → Prop := fun e s' => G.InClass S s'.tree ∧ e ≠ .tooBig
    ∧ ¬ GoodEnd G e s'
  have hincl : ∀ (l : List (FreeMonoid α)) (s : TState α), Reach G s →
      RunEnds (tallyStep C cr) P s l →
      (∃ e ∈ keys, EvE G C cr S θg e thr s l)
        ∨ (∃ j < l.length, EvS G C cr S j s (kf (s.probes + j + 1)) l)
        ∨ (∃ i len, i + len < l.length ∧ Skip C cr (Qp len) i s l)
        ∨ ∃ len < l.length, EvP G C cr S len s (kp (s.n + len + 1)) l := by
    intro l
    induction l with
    | nil => intro s _ h; exact h.elim
    | cons x xs ih =>
      intro s hs h
      simp only [RunEnds] at h
      rcases hst : tallyStep C cr s x with s' | ⟨e, s''⟩ <;> rw [hst] at h
      · rcases ih s' (reach_step G C _ hm hs hst).1 h with ⟨e, he, hE'⟩ | ⟨j, hj, hS⟩ |
          ⟨i, len, hil, hk⟩ | ⟨len, hlen', hP⟩
        · exact .inl ⟨e, he, .inr (by simp only [hst]; exact hE')⟩
        · refine .inr (.inl ⟨j + 1, by simp; omega, ?_⟩)
          simp only [EvS, hst]
          have hp := (tallyStep_counts C _ hst).2.2.2
          rw [(tallyPre_charge C _ s x).2.2.2] at hp
          rw [hp] at hS
          rw [show s.probes + (j + 1) + 1 = s.probes + 1 + j + 1 by omega]
          exact hS
        · refine .inr (.inr (.inl ⟨i + 1, len, by simp; omega, ?_⟩))
          simp only [Skip, hst]
          exact hk
        · rcases tallyStep_fresh C _ hst with ⟨hn0, -, hpt0⟩ | ⟨-, hn, -, -⟩
          · refine .inr (.inr (.inl ⟨1, len, by simp; omega, ?_⟩))
            simp only [Skip, hst, Qp]
            have : goodCount G s'.pt + kp (len + 1) = kp (s'.n + len + 1) := by
              rw [hpt0, hn0]; simp [goodCount]
            rw [this]
            exact hP
          · refine .inr (.inr (.inr ⟨len + 1, by simp; omega, ?_⟩))
            simp only [EvP, hst]
            refine ⟨by omega, ?_⟩
            rw [show s.n + (len + 1) + 1 = s'.n + len + 1 by omega]
            exact hP
      · obtain ⟨hcl, hne, hng⟩ := h
        obtain ⟨rfl, hl⟩ := tallyStep_inr_pre C _ hst hne
        have htr : (tallyPre C cr s x).tree = s.tree :=
          (tallyPre_into _ C x hs.2.1 hs.2.2.1).1
        rw [htr] at hcl
        obtain ⟨-, -, hstH, hprob⟩ := tallyPre_charge C cr s x
        rcases e with _ | _ | e | _ | _
        · exact absurd trivial hng
        · refine .inr (.inl ⟨0, by simp, hcl, ?_⟩)
          obtain ⟨hn₀, hlt⟩ := tallyLook_start' C hl
          rw [hprob] at hn₀ hlt
          have hn₀' : C.n₀ ≤ s.probes + 0 + 1 := by simpa using hn₀
          have hlt' : binomSfGe (s.probes + 0 + 1) C.θs
              (tallyPre C cr s x).startH.length < C.a := by simpa using hlt
          have hex : ∃ h', C.n₀ ≤ s.probes + 0 + 1 ∧ binomSfGe (s.probes + 0 + 1) C.θs h' < C.a :=
            ⟨_, hn₀', hlt'⟩
          simp only [kf, dif_pos hex]
          have hfind : Nat.find hex ≤ (tallyPre C cr s x).startH.length :=
            Nat.find_min' hex ⟨hn₀', hlt'⟩
          have hsum := goodCount_add_badCount G (tallyPre C cr s x).startH
          simp only [GoodEnd, not_lt] at hng
          omega
        · left
          obtain ⟨hp, hexc'⟩ := tallyLook_harvest' C hl
          rw [htr] at hp
          refine ⟨e, Finset.mem_product.2 ⟨mem_shortPaths ?_, Finset.mem_univ _⟩, .inl ⟨hcl, ?_⟩⟩
          · have := path_length_lt _ _ hp
            have := class_paths G C S hcl hLmax
            omega
          · have hsum := goodCount_add_badCount G ((tallyPre C cr s x).harv e.1 e.2)
            simp only [GoodEnd, not_lt] at hng
            have hb : (G.badCount ((tallyPre C cr s x).harv e.1 e.2) : ℝ) * 2
                ≤ ((tallyPre C cr s x).harv e.1 e.2).length := by
              exact_mod_cast (by omega : _ * 2 ≤ _)
            have hs' : (goodCount G ((tallyPre C cr s x).harv e.1 e.2) : ℝ)
                + G.badCount ((tallyPre C cr s x).harv e.1 e.2)
                = ((tallyPre C cr s x).harv e.1 e.2).length := by
              exact_mod_cast hsum
            have hx := hexc ((tallyPre C cr s x).reads e.1 e.2)
            have hr0 : (0 : ℝ) ≤ (tallyPre C cr s x).reads e.1 e.2 := Nat.cast_nonneg _
            simp only [edgeZ, thr, K]
            have : Real.log (1 / C.a) * (2 * L * (1 + 4 * θg * C.Lmax))
                = 2 * L * (1 + 4 * θg * C.Lmax) * Real.log (1 / C.a) := by ring
            rw [this]
            nlinarith
        · refine .inr (.inr (.inr ⟨0, by simp, hcl, ?_⟩))
          obtain ⟨hn₀, hlt⟩ := tallyLook_pt' C hl
          rcases tallyPre_stretch C cr s x with ⟨hn, -⟩ | ⟨-, hpt0⟩
          · rw [hn] at hn₀ hlt
            have hn₀' : C.n₀ ≤ s.n + 0 + 1 := by simpa using hn₀
            have hlt' : binomSfGe (s.n + 0 + 1) C.θpt (tallyPre C cr s x).pt.length < C.a := by
              simpa using hlt
            have hex : ∃ h', C.n₀ ≤ s.n + 0 + 1 ∧ binomSfGe (s.n + 0 + 1) C.θpt h' < C.a :=
              ⟨_, hn₀', hlt'⟩
            simp only [kp, dif_pos hex]
            have hfind : Nat.find hex ≤ (tallyPre C cr s x).pt.length :=
              Nat.find_min' hex ⟨hn₀', hlt'⟩
            have hsum := goodCount_add_badCount G (tallyPre C cr s x).pt
            simp only [GoodEnd, not_lt] at hng
            omega
          · rw [hpt0, List.length_nil, binomSfGe_zero_right] at hlt
            exact absurd hlt (not_lt.2 ha1)
        · exact absurd rfl hne
  have hsub : {xs : Fin T → FreeMonoid α | RunEnds (tallyStep C cr) P tallyStart (List.ofFn xs)}
      ⊆ ((⋃ e ∈ keys, {xs | EvE G C cr S θg e thr tallyStart (List.ofFn xs)})
        ∪ ⋃ j ∈ Finset.range T, {xs | EvS G C cr S j tallyStart
          (goodCount G (tallyStart : TState α).startH + kf (j + 1)) (List.ofFn xs)})
        ∪ ⋃ q ∈ Finset.range T ×ˢ Finset.range T,
          {xs | Skip C cr (Qp q.2) q.1 tallyStart (List.ofFn xs)} := by
    intro xs hxs
    rcases hincl _ _ (reach_start G) hxs with ⟨e, he, h⟩ | ⟨j, hj, h⟩ | ⟨i, len, hil, h⟩ |
      ⟨len, hlen', h⟩
    · exact .inl (.inl (Set.mem_biUnion he h))
    · refine .inl (.inr (Set.mem_biUnion (Finset.mem_range.2 (by simpa using hj)) ?_))
      simpa [tallyStart, goodCount] using h
    · simp only [List.length_ofFn] at hil
      exact .inr (Set.mem_biUnion (x := (i, len))
        (Finset.mem_product.2 ⟨Finset.mem_range.2 (by omega), Finset.mem_range.2 (by omega)⟩) h)
    · simp only [List.length_ofFn] at hlen'
      refine .inr (Set.mem_biUnion (x := (0, len))
        (Finset.mem_product.2 ⟨Finset.mem_range.2 (by omega), Finset.mem_range.2 hlen'⟩) ?_)
      simp only [Skip, Qp]
      simpa [tallyStart, goodCount] using h
  refine (measure_mono hsub).trans ((measure_union_le _ _).trans ?_)
  refine (add_le_add (measure_union_le _ _) le_rfl).trans ?_
  have h1 : ∀ e ∈ keys, (Measure.pi fun _ : Fin T => D)
      {xs | EvE G C cr S θg e thr tallyStart (List.ofFn xs)} ≤ ENNReal.ofReal C.a := by
    intro e _
    refine (edge_super G C S hm D rd hlen hL1 hθg0 hLmax hE e thr T tallyStart
      (reach_start G)).trans (le_of_eq ?_)
    have hz : edgeZ G θg e (tallyStart : TState α) = 0 := by simp [edgeZ, tallyStart, goodCount]
    rw [hz, show (0 - thr) / K = Real.log C.a by
      simp only [thr]; field_simp; rw [one_div, Real.log_inv]; ring, Real.exp_log ha]
  have h2 : ∀ j ∈ Finset.range T, (Measure.pi fun _ : Fin T => D)
      {xs | EvS G C cr S j tallyStart
        (goodCount G (tallyStart : TState α).startH + kf (j + 1)) (List.ofFn xs)}
      ≤ ENNReal.ofReal C.a := by
    intro j _
    refine (start_bin G C S hm D rd hθgs0 hθgs1 hE T j tallyStart _ (reach_start G)).trans
      (ENNReal.ofReal_le_ofReal ?_)
    simp only [kf]
    split_ifs with hex
    · exact hstart (j + 1) _ (Nat.find_spec hex).1 (Nat.find_spec hex).2
    · rw [binomSfGe_gt _ (by omega)]; exact ha.le
  have h3 : ∀ q ∈ Finset.range T ×ˢ Finset.range T, (Measure.pi fun _ : Fin T => D)
      {xs | Skip C cr (Qp q.2) q.1 tallyStart (List.ofFn xs)} ≤ ENNReal.ofReal C.a := by
    intro q _
    refine skip_le G C cr hm D (Qp q.2) (fun T' s hs => ?_) T q.1 tallyStart (reach_start G)
    refine (pt_bin G C S hm D rd hθgpt0 hθgpt1 hE T' q.2 s _ hs).trans
      (ENNReal.ofReal_le_ofReal ?_)
    simp only [kp]
    split_ifs with hex
    · exact hstartpt (q.2 + 1) _ (Nat.find_spec hex).1 (Nat.find_spec hex).2
    · rw [binomSfGe_gt _ (by omega)]; exact ha.le
  calc _ ≤ ∑ e ∈ keys, ENNReal.ofReal C.a + ∑ j ∈ Finset.range T, ENNReal.ofReal C.a
        + ∑ q ∈ Finset.range T ×ˢ Finset.range T, ENNReal.ofReal C.a :=
        add_le_add (add_le_add ((measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum h1))
          ((measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum h2)))
          ((measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum h3))
    _ = (keys.card + T + T * T) * ENNReal.ofReal C.a := by
        simp only [Finset.sum_const, Finset.card_product, Finset.card_range, nsmul_eq_mul]
        push_cast; ring
    _ ≤ ENNReal.ofReal ((2 ^ (C.Lmax + 1) * Fintype.card α + T + T * T) * C.a) := by
        rw [ENNReal.ofReal_mul (by positivity)]
        gcongr
        rw [show ((2 : ℝ) ^ (C.Lmax + 1) * Fintype.card α + T + T * T) = ((2 ^ (C.Lmax + 1)
          * Fintype.card α + T + T * T : ℕ) : ℝ) by push_cast; ring, ENNReal.ofReal_natCast]
        have : keys.card ≤ 2 ^ (C.Lmax + 1) * Fintype.card α := by
          simp only [keys, Finset.card_product, Finset.card_univ]
          exact Nat.mul_le_mul_right _ (shortPaths_card _)
        exact_mod_cast (by omega : keys.card + T + T * T ≤ 2 ^ (C.Lmax + 1) * Fintype.card α
          + T + T * T)

end Assemble

end OrthoDFA
