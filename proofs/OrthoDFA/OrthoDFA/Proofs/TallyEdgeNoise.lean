import OrthoDFA.Proofs.TallyCongr

/-!
# Concentration over the probes' length-`k` prefixes

Probes whose length-`k` prefixes differ read disjoint strings, so per-probe quantities read only
through strings sharing the probe's prefix are independent across prefixes. A bounded increment
whose mean is at most `θ₀` of a bounded count's then exceeds twice that by `η` with chance at most
`exp(−η / (2 p₀ B (1 + 4θ₀R)))` (`block_tail`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section PathCond

/-- The reads that place a string at the leaf `p`: each node's midfix on the way, and its side. -/
def pathCond : DTree α → List Bool → Option (List (FreeMonoid α × Bool))
  | .leaf, [] => some []
  | .node m r _, false :: p => (pathCond r p).map ((m, false) :: ·)
  | .node m _ a, true :: p => (pathCond a p).map ((m, true) :: ·)
  | _, _ => none

theorem sift_inl_iff {cut : FreeMonoid α → Option Bool} (u : FreeMonoid α) :
    ∀ (T : DTree α) (p : List Bool), T.sift cut u = .inl p ↔
      ∃ L, pathCond T p = some L ∧ ∀ mb ∈ L, cut (u * mb.1) = some mb.2
  | .leaf, p => by
    rcases p with _ | ⟨b, p⟩ <;> simp [DTree.sift, DTree.route, pathCond]
  | .node m r a, p => by
    have ihr := sift_inl_iff (cut := cut) u r
    have iha := sift_inl_iff (cut := cut) u a
    simp only [DTree.sift] at ihr iha ⊢
    simp only [DTree.route]
    rcases hc : cut (u * m) with _ | b
    · simp only [reduceCtorEq, false_iff, not_exists, not_and]
      intro L hL hall
      rcases p with _ | ⟨_ | _, p⟩ <;> simp only [pathCond, reduceCtorEq] at hL
      all_goals
        obtain ⟨L', -, rfl⟩ := Option.map_eq_some_iff.1 hL
        have := hall _ List.mem_cons_self
        rw [hc] at this; cases this
    · cases b
      · simp only [map_cons_eq_inl]
        constructor
        · rintro ⟨p', rfl, hp'⟩
          obtain ⟨L, hL, hall⟩ := (ihr p').1 hp'
          refine ⟨(m, false) :: L, by simp [pathCond, hL], ?_⟩
          intro mb hmb
          rcases List.mem_cons.1 hmb with rfl | h
          · exact hc
          · exact hall mb h
        · rintro ⟨L, hL, hall⟩
          rcases p with _ | ⟨_ | _, p⟩ <;> simp only [pathCond, reduceCtorEq] at hL
          · obtain ⟨L', hL', rfl⟩ := Option.map_eq_some_iff.1 hL
            exact ⟨p, rfl, (ihr p).2 ⟨L', hL', fun mb h => hall mb (List.mem_cons_of_mem _ h)⟩⟩
          · obtain ⟨L', -, rfl⟩ := Option.map_eq_some_iff.1 hL
            have := hall _ List.mem_cons_self
            rw [hc] at this; cases this
      · simp only [map_cons_eq_inl]
        constructor
        · rintro ⟨p', rfl, hp'⟩
          obtain ⟨L, hL, hall⟩ := (iha p').1 hp'
          refine ⟨(m, true) :: L, by simp [pathCond, hL], ?_⟩
          intro mb hmb
          rcases List.mem_cons.1 hmb with rfl | h
          · exact hc
          · exact hall mb h
        · rintro ⟨L, hL, hall⟩
          rcases p with _ | ⟨_ | _, p⟩ <;> simp only [pathCond, reduceCtorEq] at hL
          · obtain ⟨L', -, rfl⟩ := Option.map_eq_some_iff.1 hL
            have := hall _ List.mem_cons_self
            rw [hc] at this; cases this
          · obtain ⟨L', hL', rfl⟩ := Option.map_eq_some_iff.1 hL
            exact ⟨p, rfl, (iha p).2 ⟨L', hL', fun mb h => hall mb (List.mem_cons_of_mem _ h)⟩⟩

end PathCond

section Mgf

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- A bounded increment whose mean is at most `θ` of a bounded count's, less twice that, has
exponential moment at most `1` at any scale `c` with `c B (1 + 4θR) ≤ 1/2`. -/
theorem mgf_real (U N : Ω → ℝ) (hUm : Measurable U) (hNm : Measurable N) {θ B R c : ℝ}
    (hθ : 0 ≤ θ) (hB : 0 ≤ B) (hR : 0 ≤ R) (hc : 0 < c) (hcB : c * B * (1 + 4 * θ * R) ≤ 1 / 2)
    (hU : ∀ ω, 0 ≤ U ω ∧ U ω ≤ B) (hN : ∀ ω, 0 ≤ N ω ∧ N ω ≤ B * R)
    (hmean : ∫ ω, U ω ∂μ ≤ θ * ∫ ω, N ω ∂μ) :
    ∫⁻ ω, ENNReal.ofReal (Real.exp (c * (U ω - 2 * θ * N ω))) ∂μ ≤ 1 := by
  have hcBθR : 0 ≤ c * B * (θ * R) := mul_nonneg (mul_nonneg hc.le hB) (mul_nonneg hθ hR)
  set A := c + c ^ 2 * B
  set B' := 2 * θ * c - 4 * θ ^ 2 * c ^ 2 * B * R
  set g : Ω → ℝ := fun ω => 1 + A * U ω - B' * N ω
  have hUi : Integrable U μ := Integrable.of_bound hUm.aestronglyMeasurable B
    (Filter.Eventually.of_forall fun ω => by
      rw [Real.norm_eq_abs, abs_of_nonneg (hU ω).1]; exact (hU ω).2)
  have hNi : Integrable N μ := Integrable.of_bound hNm.aestronglyMeasurable (B * R)
    (Filter.Eventually.of_forall fun ω => by
      rw [Real.norm_eq_abs, abs_of_nonneg (hN ω).1]; exact (hN ω).2)
  have hgi : Integrable g μ := ((integrable_const 1).add (hUi.const_mul A)).sub (hNi.const_mul B')
  have hcBR : c * B ≤ 1 / 2 := by nlinarith
  have hpt : ∀ ω, Real.exp (c * (U ω - 2 * θ * N ω)) ≤ g ω := by
    intro ω
    obtain ⟨hU0, hU1⟩ := hU ω
    obtain ⟨hN0, hN1⟩ := hN ω
    set y := c * (U ω - 2 * θ * N ω)
    have hy1 : |y| ≤ 1 := by
      rw [abs_le]
      constructor
      · have : c * (2 * θ * N ω) ≤ 1 := by
          calc c * (2 * θ * N ω) ≤ c * (2 * θ * (B * R)) := by gcongr
            _ ≤ c * B * (1 + 4 * θ * R) := by nlinarith
            _ ≤ 1 := by linarith
        nlinarith [mul_nonneg hc.le hU0]
      · have : c * U ω ≤ 1 := by
          calc c * U ω ≤ c * B := by gcongr
            _ ≤ 1 := by linarith
        nlinarith [mul_nonneg hc.le (mul_nonneg hθ hN0)]
    have hexp := Real.abs_exp_sub_one_sub_id_le hy1
    have hsq : y ^ 2 ≤ c ^ 2 * (B * U ω + 4 * θ ^ 2 * (B * R) * N ω) := by
      simp only [y]
      rw [mul_pow]
      gcongr
      nlinarith [mul_le_mul_of_nonneg_left hU1 hU0, mul_le_mul_of_nonneg_left hN1 hN0,
        mul_nonneg hθ (mul_nonneg hU0 hN0)]
    have h1 : Real.exp y ≤ 1 + y + y ^ 2 := by
      have := (abs_le.1 hexp).2
      linarith
    have h2 : 1 + y + c ^ 2 * (B * U ω + 4 * θ ^ 2 * (B * R) * N ω) = g ω := by
      simp only [g, A, B', y]; ring
    linarith
  calc ∫⁻ ω, ENNReal.ofReal (Real.exp (c * (U ω - 2 * θ * N ω))) ∂μ
      ≤ ∫⁻ ω, ENNReal.ofReal (g ω) ∂μ := lintegral_mono fun ω => ENNReal.ofReal_le_ofReal (hpt ω)
    _ = ENNReal.ofReal (∫ ω, g ω ∂μ) := by
        rw [ofReal_integral_eq_lintegral_ofReal hgi]
        exact Filter.Eventually.of_forall fun ω => (Real.exp_pos _).le.trans (hpt ω)
    _ ≤ 1 := by
        rw [← ENNReal.ofReal_one]
        refine ENNReal.ofReal_le_ofReal ?_
        have hint : ∫ ω, g ω ∂μ = 1 + A * ∫ ω, U ω ∂μ - B' * ∫ ω, N ω ∂μ := by
          have e1 := integral_sub (μ := μ) (f := fun ω => 1 + A * U ω)
            (g := fun ω => B' * N ω) ((integrable_const 1).add (hUi.const_mul A))
            (hNi.const_mul B')
          have e2 := integral_add (μ := μ) (f := fun _ => (1 : ℝ)) (g := fun ω => A * U ω)
            (integrable_const 1) (hUi.const_mul A)
          simp only [g]
          rw [e1, e2, integral_const_mul, integral_const_mul, integral_const]
          simp
        have hEN : 0 ≤ ∫ ω, N ω ∂μ := integral_nonneg fun ω => (hN ω).1
        have hA : 0 ≤ A := by positivity
        rw [hint]
        have h1 := mul_le_mul_of_nonneg_left hmean hA
        have hkey : A * θ - B' ≤ 0 := by
          simp only [A, B']
          nlinarith [mul_nonneg (mul_nonneg hc.le hc.le) hθ, mul_nonneg hθ hc.le]
        nlinarith

end Mgf

section One

variable {σ : Type*} (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)

/-- The probes of length from `k` to `L`. -/
noncomputable def lenRange (k L : ℕ) : Finset (FreeMonoid α) :=
  (Finset.Icc k L).biUnion lenK

theorem mem_lenRange {k L : ℕ} {x : FreeMonoid α} :
    x ∈ lenRange k L ↔ k ≤ x.toList.length ∧ x.toList.length ≤ L := by
  simp [lenRange, mem_lenK]

/-- The read of `z` from the reads of the block `S`, undecided off it. -/
def rdOf (S : Finset (FreeMonoid α)) (v : S → ARU) (z : FreeMonoid α) : ARU :=
  if h : z ∈ S then v ⟨z, h⟩ else .undecided

theorem integral_eq_sum_lenRange' (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    {k L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hk : ∀ᵐ x ∂D, k ≤ x.toList.length)
    (f : FreeMonoid α → ℝ) (M : ℝ) (hf : ∀ x, x.toList.length ≤ L → |f x| ≤ M) :
    ∫ x, f x ∂D = ∑ x ∈ lenRange k L, D.real {x} * f x := by
  have hae : ∀ᵐ x ∂D, x ∈ ((lenRange k L : Finset (FreeMonoid α)) : Set (FreeMonoid α)) := by
    filter_upwards [hlen, hk] with x h1 h2
    exact mem_lenRange.2 ⟨h2, h1⟩
  have hi : Integrable f D := Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable
    M (by filter_upwards [hlen] with x hx; rw [Real.norm_eq_abs]; exact hf x hx)
  calc ∫ x, f x ∂D = ∫ x in ((lenRange k L : Finset (FreeMonoid α)) : Set (FreeMonoid α)),
        f x ∂D := by rw [Measure.restrict_eq_self_of_ae_mem hae]
    _ = _ := by rw [setIntegral_finset _ hi.integrableOn]; simp [smul_eq_mul]

theorem integral_eq_sum_lenRange (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    {k L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hk : ∀ᵐ x ∂D, k ≤ x.toList.length)
    (f : FreeMonoid α → ℝ) (hf : ∀ x, x.toList.length ≤ L → |f x| ≤ L + 1) :
    ∫ x, f x ∂D = ∑ x ∈ lenRange k L, D.real {x} * f x :=
  integral_eq_sum_lenRange' D hlen hk f (L + 1) hf

end One

section Blocks

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {read : FreeMonoid α → Ω → ARU}

theorem measurable_of_strings {β : Type*} [MeasurableSpace β] (hmeas : ∀ z, Measurable (read z))
    (S : Finset (FreeMonoid α)) (f : (FreeMonoid α → ARU) → β)
    (hf : ∀ rd rd', (∀ z ∈ S, rd z = rd' z) → f rd = f rd') :
    Measurable fun ω => f (read · ω) := by
  have : (fun ω => f (read · ω))
      = (fun v : S → ARU => f (rdOf S v)) ∘ fun ω (i : S) => read i ω := by
    funext ω
    exact hf _ _ fun z hz => by simp only [rdOf, dif_pos hz]
  rw [this]
  exact (measurable_of_countable _).comp (measurable_pi_lambda _ fun i => hmeas i)

/-- Probe quantities each read through strings sharing the probe's length-`k` prefix, the first's
mean at most `θ₀` of the second's: their `D`-weighted sums exceed twice `θ₀` by `η` with chance at
most `exp(−η / (2 p₀ B (1 + 4θ₀R)))`. -/
theorem block_tail (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {k L : ℕ} {p₀ η θ₀ B R : ℝ}
    (hp₀ : 0 < p₀)
    (hpmax : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (hη : 0 ≤ η) (hθ₀ : 0 ≤ θ₀) (hB : 0 < B)
    (hR : 0 ≤ R) (Sx : FreeMonoid α → Finset (FreeMonoid α))
    (hSx : ∀ x z, z ∈ Sx x → k ≤ x.toList.length → prefixOf z k = prefixOf x k)
    (uf tf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ)
    (huf : ∀ x rd rd', (∀ z ∈ Sx x, rd z = rd' z) → uf rd x = uf rd' x)
    (htf : ∀ x rd rd', (∀ z ∈ Sx x, rd z = rd' z) → tf rd x = tf rd' x)
    (hub : ∀ rd, ∀ x ∈ lenRange k L, 0 ≤ uf rd x ∧ uf rd x ≤ B)
    (htb : ∀ rd, ∀ x ∈ lenRange k L, 0 ≤ tf rd x ∧ tf rd x ≤ B * R)
    (hmean : ∀ x ∈ lenRange k L,
      ∫ ω, uf (read · ω) x ∂μ ≤ θ₀ * ∫ ω, tf (read · ω) x ∂μ) :
    μ {ω | 2 * θ₀ * ∑ x ∈ lenRange k L, D.real {x} * tf (read · ω) x + η
        < ∑ x ∈ lenRange k L, D.real {x} * uf (read · ω) x}
      ≤ ENNReal.ofReal (Real.exp (-(η / (2 * p₀ * B * (1 + 4 * θ₀ * R))))) := by
  set c : ℝ := 1 / (2 * p₀ * B * (1 + 4 * θ₀ * R))
  have hc : 0 < c := by positivity
  set X : Finset (FreeMonoid α) := lenRange k L
  set U : Finset (FreeMonoid α) := lenK k
  set Xu : FreeMonoid α → Finset (FreeMonoid α) := fun u => X.filter fun x => prefixOf x k = u
  set w : FreeMonoid α → ℝ := fun x => D.real {x}
  set Sb : FreeMonoid α → Finset (FreeMonoid α) := fun u => (Xu u).biUnion Sx
  have hXk : ∀ x ∈ X, k ≤ x.toList.length := fun x hx => (mem_lenRange.1 hx).1
  have hmapsto : ∀ x ∈ X, prefixOf x k ∈ U := by
    intro x hx
    refine mem_lenK.2 ?_
    simp [prefixOf, hXk x hx]
  have hsum : ∀ f : FreeMonoid α → ℝ, ∑ x ∈ X, w x * f x = ∑ u ∈ U, ∑ x ∈ Xu u, w x * f x :=
    fun f => (Finset.sum_fiberwise_of_maps_to hmapsto _).symm
  have hblk : ∀ u, ∀ x ∈ Xu u, ∀ rd rd' : FreeMonoid α → ARU, (∀ z ∈ Sb u, rd z = rd' z) →
      ∀ z ∈ Sx x, rd z = rd' z := fun u x hx rd rd' h z hz =>
    h z (Finset.mem_biUnion.2 ⟨x, hx, hz⟩)
  have hrd : ∀ u ω, ∀ z ∈ Sb u, (read z ω) = rdOf (Sb u) (fun i => read i ω) z := by
    intro u ω z hz
    simp only [rdOf, dif_pos hz]
  have hmf : ∀ (f : (FreeMonoid α → ARU) → FreeMonoid α → ℝ),
      (∀ x rd rd', (∀ z ∈ Sx x, rd z = rd' z) → f rd x = f rd' x) →
      ∀ x, Measurable fun ω => f (read · ω) x := fun f hf x =>
    measurable_of_strings hmeas (Sx x) (fun rd => f rd x) fun rd rd' h => hf x rd rd' h
  have hw0 : ∀ x, 0 ≤ w x := fun x => measureReal_nonneg
  have hwsum : ∀ u, ∑ x ∈ Xu u, w x ≤ p₀ := by
    intro u
    refine le_trans ?_ (hpmax u)
    rw [← measureReal_biUnion_finset (fun x _ x' _ h => Set.disjoint_singleton.2 h)
      (fun x _ => measurableSet_singleton x)]
    refine measureReal_mono ?_
    intro y hy
    simp only [Set.mem_iUnion, Set.mem_singleton_iff, exists_prop] at hy
    obtain ⟨x, hx, rfl⟩ := hy
    exact (Finset.mem_filter.1 hx).2
  set Y : FreeMonoid α → Ω → ℝ := fun u ω =>
    ∑ x ∈ Xu u, w x * (uf (read · ω) x - 2 * θ₀ * tf (read · ω) x)
  have hev : {ω | 2 * θ₀ * ∑ x ∈ X, w x * tf (read · ω) x + η < ∑ x ∈ X, w x * uf (read · ω) x}
      ⊆ {ω | η < ∑ u ∈ U, Y u ω} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    rw [hsum, hsum] at hω
    have : ∑ u ∈ U, Y u ω = ∑ u ∈ U, ∑ x ∈ Xu u, w x * uf (read · ω) x
        - 2 * θ₀ * ∑ u ∈ U, ∑ x ∈ Xu u, w x * tf (read · ω) x := by
      simp only [Y, Finset.mul_sum, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun u _ => Finset.sum_congr rfl fun x _ => ?_
      ring
    rw [this]
    linarith
  have hXuX : ∀ u, ∀ x ∈ Xu u, x ∈ X := fun u x hx => (Finset.mem_filter.1 hx).1
  have hint : ∀ (u : FreeMonoid α) (f : (FreeMonoid α → ARU) → FreeMonoid α → ℝ) (M : ℝ),
      (∀ x rd rd', (∀ z ∈ Sx x, rd z = rd' z) → f rd x = f rd' x) →
      (∀ rd, ∀ x ∈ X, 0 ≤ f rd x ∧ f rd x ≤ M) →
      ∫ ω, ∑ x ∈ Xu u, w x * f (read · ω) x ∂μ = ∑ x ∈ Xu u, w x * ∫ ω, f (read · ω) x ∂μ := by
    intro u f M hf hfb
    rw [integral_finset_sum _ fun x hx => (Integrable.of_bound (hmf f hf x).aestronglyMeasurable
      M (Filter.Eventually.of_forall fun ω => by
        rw [Real.norm_eq_abs, abs_of_nonneg (hfb _ x (hXuX u x hx)).1]
        exact (hfb _ x (hXuX u x hx)).2)).const_mul _]
    exact Finset.sum_congr rfl fun x _ => integral_const_mul _ _
  have hblock : ∀ u, ∫⁻ ω, ENNReal.ofReal (Real.exp (c * Y u ω)) ∂μ ≤ 1 := by
    intro u
    set Uf : Ω → ℝ := fun ω => ∑ x ∈ Xu u, w x * uf (read · ω) x
    set Nf : Ω → ℝ := fun ω => ∑ x ∈ Xu u, w x * tf (read · ω) x
    have hY : ∀ ω, c * Y u ω = c * (Uf ω - 2 * θ₀ * Nf ω) := by
      intro ω
      have : Y u ω = Uf ω - 2 * θ₀ * Nf ω := by
        simp only [Y, Uf, Nf, Finset.mul_sum, ← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun x _ => by ring
      rw [this]
    simp_rw [hY]
    have hUm : Measurable Uf := Finset.measurable_sum _ fun x _ =>
      measurable_const.mul (hmf uf huf x)
    have hNm : Measurable Nf := Finset.measurable_sum _ fun x _ =>
      measurable_const.mul (hmf tf htf x)
    refine mgf_real Uf Nf hUm hNm hθ₀ (by positivity : 0 ≤ p₀ * B) hR hc ?_ ?_ ?_ ?_
    · simp only [c]; field_simp; linarith
    · intro ω
      constructor
      · exact Finset.sum_nonneg fun x hx => mul_nonneg (hw0 x) (hub _ x (hXuX u x hx)).1
      · calc Uf ω ≤ ∑ x ∈ Xu u, w x * B := Finset.sum_le_sum fun x hx =>
              mul_le_mul_of_nonneg_left (hub _ x (hXuX u x hx)).2 (hw0 x)
          _ = (∑ x ∈ Xu u, w x) * B := by rw [Finset.sum_mul]
          _ ≤ p₀ * B := mul_le_mul_of_nonneg_right (hwsum u) hB.le
    · intro ω
      constructor
      · exact Finset.sum_nonneg fun x hx => mul_nonneg (hw0 x) (htb _ x (hXuX u x hx)).1
      · calc Nf ω ≤ ∑ x ∈ Xu u, w x * (B * R) := Finset.sum_le_sum fun x hx =>
              mul_le_mul_of_nonneg_left (htb _ x (hXuX u x hx)).2 (hw0 x)
          _ = (∑ x ∈ Xu u, w x) * (B * R) := by rw [Finset.sum_mul]
          _ ≤ p₀ * (B * R) := mul_le_mul_of_nonneg_right (hwsum u) (by positivity)
          _ = p₀ * B * R := by ring
    · rw [hint u uf B huf hub, hint u tf (B * R) htf htb, Finset.mul_sum]
      exact Finset.sum_le_sum fun x hx => by
        have := hmean x (hXuX u x hx)
        nlinarith [hw0 x]
  have hdisj : ∀ u ∈ U, ∀ u' ∈ U, u ≠ u' → Disjoint (Sb u) (Sb u') := by
    intro u _ u' _ hne
    rw [Finset.disjoint_left]
    intro z h1 h2
    apply hne
    have key : ∀ v, z ∈ Sb v → prefixOf z k = v := by
      intro v hz
      obtain ⟨x, hx, hz⟩ := Finset.mem_biUnion.1 hz
      obtain ⟨hxX, hxv⟩ := Finset.mem_filter.1 hx
      rw [← hxv]
      exact hSx x z hz (hXk x hxX)
    rw [← key u h1, key u' h2]
  set gb : ∀ u, (Sb u → ARU) → ENNReal := fun u v => ENNReal.ofReal (Real.exp (c *
    ∑ x ∈ Xu u, w x * (uf (rdOf (Sb u) v) x - 2 * θ₀ * tf (rdOf (Sb u) v) x)))
  have hgb : ∀ u, ∀ ω, gb u (fun i => read i ω) = ENNReal.ofReal (Real.exp (c * Y u ω)) := by
    intro u ω
    simp only [gb, Y]
    congr 3
    refine Finset.sum_congr rfl fun x hx => ?_
    rw [huf x _ _ (hblk u x hx _ _ fun z hz => hrd u ω z hz),
      htf x _ _ (hblk u x hx _ _ fun z hz => hrd u ω z hz)]
  have hprod : ∫⁻ ω, ENNReal.ofReal (Real.exp (c * ∑ u ∈ U, Y u ω)) ∂μ ≤ 1 := by
    have he : ∀ ω, ENNReal.ofReal (Real.exp (c * ∑ u ∈ U, Y u ω))
        = ∏ u ∈ U, gb u (fun i => read i ω) := by
      intro ω
      rw [Finset.mul_sum, Real.exp_sum, ENNReal.ofReal_prod_of_nonneg fun u _ =>
        (Real.exp_pos _).le]
      exact Finset.prod_congr rfl fun u _ => (hgb u ω).symm
    simp_rw [he]
    rw [lintegral_prod_blocks read hmeas hind Sb gb U hdisj]
    calc ∏ u ∈ U, ∫⁻ ω, gb u (fun i => read i ω) ∂μ ≤ ∏ _u ∈ U, (1 : ENNReal) := by
          refine Finset.prod_le_prod' fun u _ => ?_
          simp_rw [hgb]
          exact hblock u
      _ = 1 := Finset.prod_const_one
  have hYm : Measurable fun ω => ENNReal.ofReal (Real.exp (c * ∑ u ∈ U, Y u ω)) := by
    refine ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp (measurable_const.mul
      (Finset.measurable_sum _ fun u _ => Finset.measurable_sum _ fun x _ =>
        measurable_const.mul ((hmf uf huf x).sub (measurable_const.mul (hmf tf htf x))))))
  calc μ _ ≤ μ {ω | η < ∑ u ∈ U, Y u ω} := measure_mono hev
    _ ≤ μ {ω | ENNReal.ofReal (Real.exp (c * η))
          ≤ ENNReal.ofReal (Real.exp (c * ∑ u ∈ U, Y u ω))} := by
        refine measure_mono fun ω hω => ?_
        simp only [Set.mem_setOf_eq] at hω ⊢
        exact ENNReal.ofReal_le_ofReal (Real.exp_le_exp.2
          (mul_le_mul_of_nonneg_left hω.le hc.le))
    _ ≤ (∫⁻ ω, ENNReal.ofReal (Real.exp (c * ∑ u ∈ U, Y u ω)) ∂μ)
          / ENNReal.ofReal (Real.exp (c * η)) :=
        meas_ge_le_lintegral_div hYm.aemeasurable (by simp [Real.exp_pos]) ENNReal.ofReal_ne_top
    _ ≤ 1 / ENNReal.ofReal (Real.exp (c * η)) := by gcongr
    _ = _ := by
        rw [← ENNReal.ofReal_one, ← ENNReal.ofReal_div_of_pos (Real.exp_pos _), one_div,
          ← Real.exp_neg]
        congr 2
        simp only [c]
        field_simp

end Blocks

end OrthoDFA
