import OrthoDFA.Proofs.RandomSearch
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# Concentration over the probes' prefixes

The reads are independent across strings. A probe reads only strings sharing its length-`k`
prefix (`pot_pre`), so per-probe quantities read through `pot` are independent across prefixes,
and a bounded increment whose mean is at most `θ₀` of a bounded count's exceeds twice that by `η`
with chance at most `exp(−η / (2 p₀ B (1 + 4θ₀R)))` (`block_tail`). An event of finitely many
reads that making `z` undecided can only falsify is no likelier with `z` undecided (`forced_le`).
`classSet` counts the trees level 3 can grow.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory ProbabilityTheory

variable {α : Type*}

/-! ## The class -/

theorem mids_card : ∀ T : DTree α, (mids T).card + 1 ≤ T.leaves.length
  | .leaf => by simp [mids, DTree.leaves]
  | .node m r a => by
    classical
    have hr := mids_card r
    have ha := mids_card a
    simp only [mids, DTree.leaves, List.length_append, List.length_map]
    have := Finset.card_insert_le m (mids r ∪ mids a)
    have := Finset.card_union_le (mids r) (mids a)
    omega

variable [Fintype α]

theorem classSet_leaves : ∀ (n : ℕ) (T : DTree α), T ∈ classSet n → T.leaves.length ≤ n + 2
  | 0, T, h => by
    simp only [classSet, Finset.mem_singleton] at h
    subst h
    simp [DTree.leaves]
  | n + 1, T, h => by
    classical
    simp only [classSet, Finset.mem_union, Finset.mem_biUnion, Finset.mem_image] at h
    rcases h with h | ⟨T', hT', ⟨p, c, t, t₀⟩, hq, rfl⟩
    · have := classSet_leaves n T h; omega
    · simp only [Finset.mem_product, List.mem_toFinset] at hq
      rw [DTree.length_leaves_splitAt _ _ _ hq.1]
      have := classSet_leaves n T' hT'
      omega

theorem classSet_card : ∀ n : ℕ,
    (classSet n : Finset (DTree α)).card ≤ (1 + (n + 1) ^ 3 * Fintype.card α) ^ n
  | 0 => by simp [classSet]
  | n + 1 => by
    classical
    simp only [classSet]
    refine (Finset.card_union_le _ _).trans ?_
    have hstep : ∀ T ∈ (classSet n : Finset (DTree α)),
        ((T.leaves.toFinset ×ˢ (Finset.univ : Finset α) ×ˢ T.leaves.toFinset ×ˢ
          T.leaves.toFinset).image fun q => T.splitAt (FreeMonoid.of q.2.1 *
            T.midfixAt (commonPrefix q.2.2.1 q.2.2.2)) q.1).card
          ≤ (n + 2) ^ 3 * Fintype.card α := by
      intro T hT
      refine Finset.card_image_le.trans ?_
      have h := (List.toFinset_card_le T.leaves).trans (classSet_leaves n T hT)
      simp only [Finset.card_product, Finset.card_univ]
      calc _ ≤ (n + 2) * (Fintype.card α * ((n + 2) * (n + 2))) := by gcongr
        _ = _ := by ring
    have hb := Finset.card_biUnion_le_card_mul (classSet n) _ _ hstep
    have ih := classSet_card n
    have h2 : (1 + (n + 1) ^ 3 * Fintype.card α) ^ n
        ≤ (1 + (n + 1 + 1) ^ 3 * Fintype.card α) ^ n := by
      gcongr <;> omega
    calc _ ≤ (classSet n).card + (classSet n).card * ((n + 2) ^ 3 * Fintype.card α) :=
          Nat.add_le_add_left hb _
      _ = (classSet n).card * (1 + (n + 2) ^ 3 * Fintype.card α) := by ring
      _ ≤ (1 + (n + 1 + 1) ^ 3 * Fintype.card α) ^ n * (1 + (n + 2) ^ 3 * Fintype.card α) :=
          Nat.mul_le_mul_right _ (ih.trans h2)
      _ = _ := by ring

/-! ## Strings -/

/-- The strings of length `k`. -/
noncomputable def lenK (k : ℕ) : Finset (FreeMonoid α) :=
  (Finset.univ : Finset (Fin k → α)).image fun v => FreeMonoid.ofList (List.ofFn v)

theorem mem_lenK {k : ℕ} {u : FreeMonoid α} : u ∈ lenK k ↔ u.toList.length = k := by
  constructor
  · intro h
    obtain ⟨v, -, rfl⟩ := Finset.mem_image.1 h
    simp
  · intro h
    refine Finset.mem_image.2 ⟨fun i => u.toList.get ⟨i, by omega⟩, Finset.mem_univ _, ?_⟩
    apply FreeMonoid.toList.injective
    simp only [FreeMonoid.toList_ofList]
    apply List.ext_get (by simp [h])
    intro i h1 h2
    simp

/-- The strings of length at most `L`. -/
noncomputable def upTo (L : ℕ) : Finset (FreeMonoid α) := (Finset.range (L + 1)).biUnion lenK

theorem mem_upTo {L : ℕ} {u : FreeMonoid α} : u ∈ upTo L ↔ u.toList.length ≤ L := by
  classical
  simp only [upTo, Finset.mem_biUnion, Finset.mem_range, mem_lenK]
  constructor
  · rintro ⟨j, hj, rfl⟩; omega
  · intro h; exact ⟨_, by omega, rfl⟩

/-- A string a probe of `x` can read has `x`'s length-`k` prefix. -/
theorem pot_pre {k : ℕ} {x z : FreeMonoid α} {T : DTree α} (h : z ∈ pot k x T) :
    k ≤ x.toList.length ∧ prefixOf z k = prefixOf x k := by
  classical
  unfold pot at h
  obtain ⟨p, hp, hz⟩ := Finset.mem_biUnion.1 h
  obtain ⟨hkp, hpx⟩ := Finset.mem_Icc.1 hp
  obtain ⟨m, -, rfl⟩ := Finset.mem_image.1 hz
  refine ⟨hkp.trans hpx, ?_⟩
  simp only [prefixOf, FreeMonoid.toList_mul, FreeMonoid.toList_ofList]
  rw [List.take_append_of_le_length (by simp; omega), List.take_take, min_eq_left hkp]

theorem card_pot (k : ℕ) (x : FreeMonoid α) (T : DTree α) :
    (pot k x T).card ≤ (x.toList.length + 1) * (mids T).card := by
  classical
  unfold pot
  refine Finset.card_biUnion_le.trans ?_
  calc _ ≤ ∑ _p ∈ Finset.Icc k x.toList.length, (mids T).card :=
        Finset.sum_le_sum fun p _ => Finset.card_image_le
    _ = (Finset.Icc k x.toList.length).card * (mids T).card := by
        rw [Finset.sum_const, smul_eq_mul]
    _ ≤ _ := by
        gcongr
        simp only [Nat.card_Icc]
        omega

/-! ## Independent reads -/

/-- An index for each read. -/
def aruFin3 : ARU → Fin 3
  | .accept => 0
  | .reject => 1
  | .undecided => 2

instance : Finite ARU :=
  Finite.of_injective aruFin3 fun a b h => by cases a <;> cases b <;> simp_all [aruFin3]

section Blocks

variable [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)

open scoped Classical in
/-- Functions of reads of pairwise disjoint finite sets of strings: the moment of their product
is the product of their moments. -/
theorem lintegral_prod_blocks (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    {ι : Type*} (S : ι → Finset (FreeMonoid α)) (g : ∀ u, (S u → ARU) → ENNReal) :
    ∀ V : Finset ι, (∀ u ∈ V, ∀ u' ∈ V, u ≠ u' → Disjoint (S u) (S u')) →
      ∫⁻ ω, ∏ u ∈ V, g u (fun z => read z ω) ∂μ
        = ∏ u ∈ V, ∫⁻ ω, g u (fun z => read z ω) ∂μ := by
  intro V
  induction V using Finset.induction_on with
  | empty => intro _; simp
  | insert a V ha ih =>
    intro hdisj
    rw [Finset.prod_insert ha]
    simp_rw [Finset.prod_insert ha]
    rw [← ih fun u hu u' hu' h => hdisj u (Finset.mem_insert_of_mem hu) u'
        (Finset.mem_insert_of_mem hu') h]
    set B := V.biUnion S
    have hdB : Disjoint (S a) B := by
      rw [Finset.disjoint_biUnion_right]
      intro u hu
      exact hdisj a (Finset.mem_insert_self a V) u (Finset.mem_insert_of_mem hu)
        (fun h => ha (h ▸ hu))
    have hI := hind.indepFun_finset (S a) B hdB hmeas
    set H : (B → ARU) → ENNReal := fun v =>
      ∏ u ∈ V.attach, g u.1 (fun z => v ⟨z.1, Finset.mem_biUnion.2 ⟨u.1, u.2, z.2⟩⟩)
    have hH : ∀ ω, ∏ u ∈ V, g u (fun z => read z ω) = H (fun i => read i ω) := by
      intro ω
      rw [← Finset.prod_attach V]
    have hmA : Measurable fun ω (i : S a) => read i ω := measurable_pi_lambda _ fun i => hmeas i
    have hmB : Measurable fun ω (i : B) => read i ω := measurable_pi_lambda _ fun i => hmeas i
    have hI' := hI.comp (φ := g a) (ψ := H) (measurable_of_countable _) (measurable_of_countable _)
    simp_rw [hH]
    exact lintegral_mul_eq_lintegral_mul_lintegral_of_indepFun
      (f := fun ω => g a (fun z => read z ω)) (g := fun ω => H (fun i => read i ω))
      ((measurable_of_countable (g a)).comp hmA) ((measurable_of_countable H).comp hmB) hI'

end Blocks

section Mgf

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- A bounded increment whose mean is at most `θ` of a bounded count's, less twice that, has
exponential moment at most `1` at any scale `c` with `c B (1 + 4θR) ≤ 1/2`. -/
theorem mgf_real (U N : Ω → ℝ) (hUm : Measurable U) (hNm : Measurable N) {θ B R c : ℝ} (hθ : 0 ≤ θ)
    (hB : 0 ≤ B) (hR : 0 ≤ R) (hc : 0 < c) (hcB : c * B * (1 + 4 * θ * R) ≤ 1 / 2)
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

open scoped Classical in
/-- The read of `z` from the reads of the block `S`, undecided off it. -/
noncomputable def rdOf (S : Finset (FreeMonoid α)) (v : S → ARU) (z : FreeMonoid α) : ARU :=
  if h : z ∈ S then v ⟨z, h⟩ else .undecided


section Tail

variable [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {read : FreeMonoid α → Ω → ARU}

omit [DecidableEq α] in
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
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {k L : ℕ} {p₀ η θ₀ B R : ℝ} (hp₀ : 0 < p₀)
    (hpmax : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (hθ₀ : 0 ≤ θ₀) (hB : 0 < B)
    (hR : 0 ≤ R) (Sx : FreeMonoid α → Finset (FreeMonoid α))
    (hSx : ∀ x z, z ∈ Sx x → prefixOf z k = prefixOf x k)
    (uf tf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ)
    (huf : ∀ x rd rd', (∀ z ∈ Sx x, rd z = rd' z) → uf rd x = uf rd' x)
    (htf : ∀ x rd rd', (∀ z ∈ Sx x, rd z = rd' z) → tf rd x = tf rd' x)
    (hub : ∀ rd, ∀ x ∈ upTo L, 0 ≤ uf rd x ∧ uf rd x ≤ B)
    (htb : ∀ rd, ∀ x ∈ upTo L, 0 ≤ tf rd x ∧ tf rd x ≤ B * R)
    (hmean : ∀ x ∈ upTo L, ∫ ω, uf (read · ω) x ∂μ ≤ θ₀ * ∫ ω, tf (read · ω) x ∂μ) :
    μ {ω | 2 * θ₀ * ∑ x ∈ upTo L, D.real {x} * tf (read · ω) x + η
        < ∑ x ∈ upTo L, D.real {x} * uf (read · ω) x}
      ≤ ENNReal.ofReal (Real.exp (-(η / (2 * p₀ * B * (1 + 4 * θ₀ * R))))) := by
  set c : ℝ := 1 / (2 * p₀ * B * (1 + 4 * θ₀ * R))
  have hc : 0 < c := by positivity
  set X : Finset (FreeMonoid α) := upTo L
  set U : Finset (FreeMonoid α) := X.image (prefixOf · k)
  set Xu : FreeMonoid α → Finset (FreeMonoid α) := fun u => X.filter fun x => prefixOf x k = u
  set w : FreeMonoid α → ℝ := fun x => D.real {x}
  set Sb : FreeMonoid α → Finset (FreeMonoid α) := fun u => (Xu u).biUnion Sx
  have hmapsto : ∀ x ∈ X, prefixOf x k ∈ U := fun x hx => Finset.mem_image_of_mem _ hx
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
    simp only [Set.mem_ofPred_eq] at hω ⊢
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
    rw [integral_finsetSum _ fun x hx => (Integrable.of_bound (hmf f hf x).aestronglyMeasurable
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
      exact hSx x z hz
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
        simp only [Set.mem_ofPred_eq] at hω ⊢
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

omit [DecidableEq α] in
theorem measureReal_eq_sum_upTo (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {L : ℕ}
    (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) (A : Set (FreeMonoid α)) [DecidablePred (· ∈ A)] :
    D.real A = ∑ x ∈ upTo L, D.real {x} * if x ∈ A then 1 else 0 := by
  have hae : A =ᵐ[D] (A ∩ {x | x.toList.length ≤ L} : Set _) := by
    filter_upwards [hL] with x hx
    change (x ∈ A) = (x ∈ A ∧ _)
    simp [hx]
  rw [measureReal_congr hae]
  have : A ∩ {x | x.toList.length ≤ L} = ↑((upTo L).filter fun x => x ∈ A) := by
    ext x; simp [mem_upTo, and_comm]
  classical
  rw [this, ← sum_measureReal_singleton, Finset.sum_filter]
  exact Finset.sum_congr rfl fun x _ => by split_ifs <;> simp

omit [DecidableEq α] in
open scoped Classical in
/-- An event of finitely many reads that making `z` undecided can only falsify: it and `z` read
undecided have chance at most the product of theirs. -/
theorem forced_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (S : Finset (FreeMonoid α)) (Ev : (FreeMonoid α → ARU) → Prop)
    (hEv : ∀ rd rd', (∀ w ∈ S, rd w = rd' w) → (Ev rd ↔ Ev rd')) (z : FreeMonoid α)
    (hmono : ∀ rd, Ev (Function.update rd z .undecided) → Ev rd) :
    μ.real {ω | Ev (read · ω) ∧ read z ω = .undecided}
      ≤ μ.real {ω | Ev (read · ω)} * μ.real {ω | read z ω = .undecided} := by
  set R := S.erase z
  set A : Set Ω := {ω | Ev (Function.update (read · ω) z .undecided)}
  have hA : A = (fun ω (i : R) => read i ω) ⁻¹'
      {v | Ev (Function.update (rdOf R v) z .undecided)} := by
    ext ω
    simp only [A, Set.mem_preimage, Set.mem_ofPred_eq]
    refine hEv _ _ fun w hw => ?_
    by_cases hwz : w = z
    · subst hwz; simp
    · rw [Function.update_of_ne hwz, Function.update_of_ne hwz]
      simp only [rdOf]
      rw [dif_pos (Finset.mem_erase.2 ⟨hwz, hw⟩)]
  have hB : {ω | read z ω = .undecided}
      = (fun ω (i : ({z} : Finset (FreeMonoid α))) => read i ω) ⁻¹'
        {v | v ⟨z, Finset.mem_singleton_self z⟩ = .undecided} := by
    ext ω; simp
  have hI := hind.indepFun_finset R {z} (Finset.disjoint_singleton_right.2 (by simp [R])) hmeas
  have heq := hI.measure_inter_preimage_eq_mul
    {v : R → ARU | Ev (Function.update (rdOf R v) z .undecided)}
    {v : ({z} : Finset (FreeMonoid α)) → ARU | v ⟨z, Finset.mem_singleton_self z⟩ = .undecided}
    (Set.to_countable _).measurableSet (Set.to_countable _).measurableSet
  rw [← hA, ← hB] at heq
  have hset : {ω | Ev (read · ω) ∧ read z ω = .undecided} = A ∩ {ω | read z ω = .undecided} := by
    ext ω
    simp only [A, Set.mem_ofPred_eq, Set.mem_inter_iff]
    constructor <;> rintro ⟨h1, h2⟩ <;> refine ⟨?_, h2⟩
    · have : Function.update (read · ω) z ARU.undecided = (read · ω) := by
        rw [← h2]; exact Function.update_eq_self z _
      rwa [this]
    · have : Function.update (read · ω) z ARU.undecided = (read · ω) := by
        rw [← h2]; exact Function.update_eq_self z _
      rwa [this] at h1
  rw [hset, measureReal_def, heq, ENNReal.toReal_mul, ← measureReal_def, ← measureReal_def]
  exact mul_le_mul_of_nonneg_right (measureReal_mono fun ω hω => hmono _ hω) measureReal_nonneg

end Tail

end Random

end OrthoDFA
