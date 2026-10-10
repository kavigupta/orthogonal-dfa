import OrthoDFA.Proofs.RandomProbe
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# The read field

The reads are independent across strings, each undecided with its state's chance. For a fixed
tree, the mass of probes that can read a good read-state undecided is at most a weighted sum of
independent indicators, one per string `z` at a good read-state, of `z` reading undecided. A
probe can read `z` only where it shares `z`'s length-`k` prefix, so `z`'s weight, the mass of the
probes that can read it, is at most `p₀`; the weights sum to at most `(L + 1)(|Q| + 1)`, the most
strings one probe can read. The exponential moment bounds the sum's upper tail, and `noise_le`
takes the union over `classSet |Q|`.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory ProbabilityTheory
open OrthoDFA.Ideal (DTree pre lcp)

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
            T.midAt (lcp q.2.2.1 q.2.2.2)) q.1).card ≤ (n + 2) ^ 3 * Fintype.card α := by
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
    k ≤ x.toList.length ∧ pre z k = pre x k := by
  classical
  unfold pot at h
  obtain ⟨p, hp, hz⟩ := Finset.mem_biUnion.1 h
  obtain ⟨hkp, hpx⟩ := Finset.mem_Icc.1 hp
  obtain ⟨m, -, rfl⟩ := Finset.mem_image.1 hz
  refine ⟨hkp.trans hpx, ?_⟩
  simp only [pre, FreeMonoid.toList_mul, FreeMonoid.toList_ofList]
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

theorem exp_sub_one_le {w p₀ l : ℝ} (hp₀ : 0 < p₀) (hw0 : 0 ≤ w) (hw : w ≤ p₀) (hl : 0 ≤ l) :
    Real.exp (l * w) - 1 ≤ w / p₀ * (Real.exp (l * p₀) - 1) := by
  have ht0 : 0 ≤ w / p₀ := div_nonneg hw0 hp₀.le
  have ht1 : w / p₀ ≤ 1 := (div_le_one hp₀).2 hw
  have h := convexOn_exp.2 (Set.mem_univ (l * p₀)) (Set.mem_univ 0) ht0 (sub_nonneg.2 ht1)
    (by ring)
  simp only [smul_eq_mul, mul_zero, add_zero, Real.exp_zero, mul_one] at h
  have : w / p₀ * (l * p₀) = l * w := by field_simp
  rw [this] at h
  linarith

/-! ## One tree -/

variable [DecidableEq α]

theorem noise_one {σ : Type*} (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) {Ω : Type*}
    [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k L : ℕ) (p₀ G : ℝ)
    (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hU : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList))
    (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) (hp₀ : 0 < p₀)
    (hpre : ∀ u : FreeMonoid α, u.toList.length = k → D.real {x | pre x k = u} ≤ p₀)
    (hθ : 0 ≤ θ) (T : DTree α) (n : ℕ) (hT : (mids T).card ≤ n + 1) :
    μ {ω | G < D.real (PotGood M U θ k (read · ω) T)}
      ≤ ENNReal.ofReal (Real.exp (-(G - 3 * θ * (L + 1) * (n + 1)) / p₀)) := by
  classical
  set Zs := (upTo L).biUnion fun x => pot k x T
  set good : FreeMonoid α → Prop := fun z => ¬ BadAt U θ (M.eval z.toList)
  set Zg := Zs.filter good
  set c : FreeMonoid α → ℝ := fun z =>
    ∑ x ∈ (upTo L).filter (fun x => z ∈ pot k x T), D.real {x}
  set ind : FreeMonoid α → Ω → ℝ := fun z ω => if read z ω = .undecided then 1 else 0
  set W : Ω → ℝ := fun ω => ∑ z ∈ Zg, c z * ind z ω
  have hind01 : ∀ z ω, 0 ≤ ind z ω ∧ ind z ω ≤ 1 := fun z ω => by
    simp only [ind]; split_ifs <;> norm_num
  have hc0 : ∀ z, 0 ≤ c z := fun z => Finset.sum_nonneg fun _ _ => measureReal_nonneg
  have hcp : ∀ z ∈ Zs, c z ≤ p₀ := by
    intro z hz
    obtain ⟨x, -, hzx⟩ := Finset.mem_biUnion.1 hz
    obtain ⟨hkx, hpz⟩ := pot_pre hzx
    have hlen : (pre z k).toList.length = k := by
      rw [hpz]; simp [pre, hkx]
    calc c z = D.real ↑((upTo L).filter fun x => z ∈ pot k x T) := sum_measureReal_singleton _
      _ ≤ D.real {y | pre y k = pre z k} := measureReal_mono fun y hy => by
          simp only [Finset.coe_filter, Set.mem_ofPred_eq] at hy
          exact ((pot_pre hy.2).2).symm
      _ ≤ p₀ := hpre _ hlen
  have hswap : ∀ f : FreeMonoid α → FreeMonoid α → ℝ,
      ∑ x ∈ upTo L, ∑ z ∈ pot k x T, f x z
        = ∑ z ∈ Zs, ∑ x ∈ (upTo L).filter (fun x => z ∈ pot k x T), f x z := by
    intro f
    refine Finset.sum_comm' fun x z => ?_
    simp only [Zs, Finset.mem_filter, Finset.mem_biUnion]
    constructor
    · rintro ⟨hx, hz⟩; exact ⟨⟨hx, hz⟩, x, hx, hz⟩
    · rintro ⟨⟨hx, hz⟩, -⟩; exact ⟨hx, hz⟩
  have hcsum : ∑ z ∈ Zg, c z ≤ (L + 1) * (n + 1) := by
    have h1 : ∑ z ∈ Zg, c z ≤ ∑ z ∈ Zs, c z :=
      Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) fun z _ _ => hc0 z
    have h2 : ∑ z ∈ Zs, c z = ∑ x ∈ upTo L, ∑ _z ∈ pot k x T, D.real {x} := (hswap _).symm
    refine h1.trans (h2 ▸ ?_)
    calc ∑ x ∈ upTo L, ∑ _z ∈ pot k x T, D.real {x}
        = ∑ x ∈ upTo L, ((pot k x T).card : ℝ) * D.real {x} := by
          simp [Finset.sum_const, nsmul_eq_mul]
      _ ≤ ∑ x ∈ upTo L, ((L + 1) * (n + 1) : ℝ) * D.real {x} := by
          gcongr with x hx
          have h1 := card_pot k x T
          have h2 := mem_upTo.1 hx
          have : (pot k x T).card ≤ (L + 1) * (n + 1) :=
            h1.trans (Nat.mul_le_mul (by omega) hT)
          exact_mod_cast this
      _ = (L + 1) * (n + 1) * D.real ((upTo L : Finset (FreeMonoid α)) : Set (FreeMonoid α)) := by
          rw [← Finset.mul_sum, sum_measureReal_singleton]
      _ ≤ (L + 1) * (n + 1) * 1 := by gcongr; exact measureReal_le_one
      _ = _ := mul_one _
  -- the mass is below the weighted sum
  have hW : ∀ ω, D.real (PotGood M U θ k (read · ω) T) ≤ W ω := by
    intro ω
    have hae : PotGood M U θ k (read · ω) T
        =ᵐ[D] (PotGood M U θ k (read · ω) T ∩ {x | x.toList.length ≤ L} : Set _) := by
      filter_upwards [hL] with x hx
      change (x ∈ PotGood M U θ k (read · ω) T) = (x ∈ PotGood M U θ k (read · ω) T ∧ _)
      simp [hx]
    rw [measureReal_congr hae]
    calc _ ≤ D.real ↑((upTo L).filter fun x => x ∈ PotGood M U θ k (read · ω) T) :=
          measureReal_mono fun x hx => Finset.mem_coe.2 (Finset.mem_filter.2 ⟨mem_upTo.2 hx.2, hx.1⟩)
      _ = ∑ x ∈ (upTo L).filter (fun x => x ∈ PotGood M U θ k (read · ω) T), D.real {x} :=
          (sum_measureReal_singleton _).symm
      _ ≤ ∑ x ∈ upTo L, ∑ z ∈ pot k x T,
            D.real {x} * (if good z then ind z ω else 0) := by
          rw [Finset.sum_filter]
          gcongr with x hx
          have hnn : ∀ z ∈ pot k x T, 0 ≤ D.real {x} * (if good z then ind z ω else 0) :=
            fun z _ => mul_nonneg measureReal_nonneg (by split_ifs; exact (hind01 z ω).1; rfl)
          split_ifs with hxg
          · obtain ⟨z, hz, hg, hu⟩ := hxg
            calc D.real {x} = D.real {x} * (if good z then ind z ω else 0) := by
                  simp only [good, ind] at hg ⊢
                  rw [if_pos hg, if_pos hu, mul_one]
              _ ≤ _ := Finset.single_le_sum hnn hz
          · exact Finset.sum_nonneg hnn
      _ = ∑ z ∈ Zs, ∑ x ∈ (upTo L).filter (fun x => z ∈ pot k x T),
            D.real {x} * (if good z then ind z ω else 0) := hswap _
      _ = W ω := by
          simp only [W, Zg]
          rw [Finset.sum_filter]
          refine Finset.sum_congr rfl fun z _ => ?_
          rw [← Finset.sum_mul]
          split_ifs <;> simp [c]
  -- the exponential moment
  set l := 1 / p₀
  have hl : 0 ≤ l := by positivity
  have hlp : l * p₀ = 1 := one_div_mul_cancel hp₀.ne'
  set gz : ∀ z : FreeMonoid α, (({z} : Finset (FreeMonoid α)) → ARU) → ENNReal := fun z v =>
    ENNReal.ofReal (Real.exp (l * (c z * if v ⟨z, Finset.mem_singleton_self z⟩ = .undecided
      then 1 else 0)))
  have hexpW : ∀ ω, ENNReal.ofReal (Real.exp (l * W ω)) = ∏ z ∈ Zg, gz z fun y => read y ω := by
    intro ω
    rw [show l * W ω = ∑ z ∈ Zg, l * (c z * ind z ω) by simp only [W, Finset.mul_sum],
      Real.exp_sum, ENNReal.ofReal_prod_of_nonneg fun z _ => (Real.exp_pos _).le]
  have hfac : ∀ z ∈ Zg, ∫⁻ ω, gz z (fun y => read y ω) ∂μ
      ≤ ENNReal.ofReal (Real.exp (3 / 2 * θ * (c z / p₀ * (Real.exp 1 - 1)))) := by
    intro z hz
    have hzs : z ∈ Zs := (Finset.mem_filter.1 hz).1
    have hgz : (Finset.mem_filter.1 hz).2 = (Finset.mem_filter.1 hz).2 := rfl
    have hg : U (M.eval z.toList) ≤ 3 / 2 * θ := by
      have := (Finset.mem_filter.1 hz).2
      simpa [good, BadAt] using this
    have hU0 : 0 ≤ U (M.eval z.toList) := by rw [← hU z]; exact measureReal_nonneg
    set A := {ω | read z ω = .undecided}
    have hA : MeasurableSet A := hmeas z (measurableSet_singleton _)
    have he1 : 1 ≤ Real.exp (l * c z) := Real.one_le_exp (mul_nonneg hl (hc0 z))
    have hpt : ∀ ω, gz z (fun y => read y ω)
        = 1 + A.indicator (fun _ => ENNReal.ofReal (Real.exp (l * c z) - 1)) ω := by
      intro ω
      simp only [gz]
      by_cases h : read z ω = .undecided
      · rw [if_pos h, Set.indicator_of_mem (show ω ∈ A from h), mul_one,
          ENNReal.ofReal_sub _ zero_le_one, ENNReal.ofReal_one,
          add_tsub_cancel_of_le (by simpa using he1)]
      · rw [if_neg h, Set.indicator_of_notMem (show ω ∉ A from h), mul_zero, mul_zero,
          Real.exp_zero, ENNReal.ofReal_one, add_zero]
    simp_rw [hpt]
    rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
      lintegral_indicator hA, setLIntegral_const, ← ofReal_measureReal, hU z,
      ← ENNReal.ofReal_mul (by linarith), ← ENNReal.ofReal_one,
      ← ENNReal.ofReal_add zero_le_one (mul_nonneg (by linarith) hU0)]
    refine ENNReal.ofReal_le_ofReal ?_
    have h1 := exp_sub_one_le hp₀ (hc0 z) (hcp z hzs) hl
    rw [hlp] at h1
    have h2 := Real.add_one_le_exp (U (M.eval z.toList) * (Real.exp (l * c z) - 1))
    have h3 : U (M.eval z.toList) * (Real.exp (l * c z) - 1)
        ≤ 3 / 2 * θ * (c z / p₀ * (Real.exp 1 - 1)) := by
      have : 0 ≤ Real.exp (l * c z) - 1 := by linarith
      calc _ ≤ 3 / 2 * θ * (Real.exp (l * c z) - 1) := mul_le_mul_of_nonneg_right hg this
        _ ≤ _ := mul_le_mul_of_nonneg_left h1 (by positivity)
    calc 1 + (Real.exp (l * c z) - 1) * U (M.eval z.toList)
        = U (M.eval z.toList) * (Real.exp (l * c z) - 1) + 1 := by ring
      _ ≤ _ := h2
      _ ≤ _ := Real.exp_le_exp.2 h3
  have hmom : ∫⁻ ω, ENNReal.ofReal (Real.exp (l * W ω)) ∂μ
      ≤ ENNReal.ofReal (Real.exp (3 * θ * (L + 1) * (n + 1) / p₀)) := by
    simp_rw [hexpW]
    rw [lintegral_prod_blocks read hmeas hind (fun z => {z}) gz Zg fun u _ u' _ h =>
      Finset.disjoint_singleton.2 h]
    have he : Real.exp 1 - 1 ≤ 2 := by
      have := Real.exp_one_lt_d9; linarith
    have he0 : 0 ≤ Real.exp 1 - 1 := by linarith [Real.add_one_le_exp 1]
    calc _ ≤ ∏ z ∈ Zg, ENNReal.ofReal (Real.exp (3 / 2 * θ * (c z / p₀ * (Real.exp 1 - 1)))) :=
          Finset.prod_le_prod' hfac
      _ = ENNReal.ofReal (Real.exp (∑ z ∈ Zg, 3 / 2 * θ * (c z / p₀ * (Real.exp 1 - 1)))) := by
          rw [Real.exp_sum, ENNReal.ofReal_prod_of_nonneg fun z _ => (Real.exp_pos _).le]
      _ ≤ _ := by
          refine ENNReal.ofReal_le_ofReal (Real.exp_le_exp.2 ?_)
          rw [show ∑ z ∈ Zg, 3 / 2 * θ * (c z / p₀ * (Real.exp 1 - 1))
              = 3 / 2 * θ * (Real.exp 1 - 1) / p₀ * ∑ z ∈ Zg, c z by
            rw [Finset.mul_sum]; refine Finset.sum_congr rfl fun z _ => ?_; ring]
          rw [div_mul_eq_mul_div, div_le_div_iff_of_pos_right hp₀]
          calc 3 / 2 * θ * (Real.exp 1 - 1) * ∑ z ∈ Zg, c z
              ≤ 3 / 2 * θ * 2 * ((L + 1) * (n + 1)) := by
                gcongr
            _ = _ := by ring
  have hWm : Measurable fun ω => ENNReal.ofReal (Real.exp (l * W ω)) := by
    have : ∀ z, Measurable fun ω => ind z ω := fun z =>
      Measurable.ite (hmeas z (measurableSet_singleton _)) measurable_const measurable_const
    exact ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp (measurable_const.mul
      (Finset.measurable_sum _ fun z _ => measurable_const.mul (this z))))
  calc μ {ω | G < D.real (PotGood M U θ k (read · ω) T)}
      ≤ μ {ω | ENNReal.ofReal (Real.exp (l * G)) ≤ ENNReal.ofReal (Real.exp (l * W ω))} := by
        refine measure_mono fun ω hω => ?_
        simp only [Set.mem_ofPred_eq] at hω ⊢
        exact ENNReal.ofReal_le_ofReal (Real.exp_le_exp.2
          (mul_le_mul_of_nonneg_left (hω.le.trans (hW ω)) hl))
    _ ≤ (∫⁻ ω, ENNReal.ofReal (Real.exp (l * W ω)) ∂μ) / ENNReal.ofReal (Real.exp (l * G)) :=
        meas_ge_le_lintegral_div hWm.aemeasurable (by simp [Real.exp_pos])
          ENNReal.ofReal_ne_top
    _ ≤ ENNReal.ofReal (Real.exp (3 * θ * (L + 1) * (n + 1) / p₀))
          / ENNReal.ofReal (Real.exp (l * G)) := by gcongr
    _ = _ := by
        rw [← ENNReal.ofReal_div_of_pos (Real.exp_pos _), ← Real.exp_sub]
        congr 2
        simp only [l]
        field_simp
        ring

theorem noise_le {σ : Type*} [Fintype σ] (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) {Ω : Type*}
    [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k L : ℕ) (p₀ G : ℝ)
    (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hU : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList))
    (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) (hp₀ : 0 < p₀)
    (hpre : ∀ u : FreeMonoid α, u.toList.length = k → D.real {x | pre x k = u} ≤ p₀)
    (hθ : 0 ≤ θ) :
    μ {ω | ¬ ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
        D.real (PotGood M U θ k (read · ω) T) ≤ G}
      ≤ ENNReal.ofReal (noiseRisk (Fintype.card σ) (Fintype.card α) L θ p₀ G) := by
  set n := Fintype.card σ
  set e := Real.exp (-(G - 3 * θ * (L + 1) * (n + 1)) / p₀)
  calc μ {ω | ¬ ∀ T ∈ (classSet n : Finset (DTree α)), D.real (PotGood M U θ k (read · ω) T) ≤ G}
      ≤ μ (⋃ T ∈ (classSet n : Finset (DTree α)),
          {ω | G < D.real (PotGood M U θ k (read · ω) T)}) := by
        refine measure_mono fun ω hω => ?_
        simp only [Set.mem_ofPred_eq, not_forall, not_le] at hω
        obtain ⟨T, hT, h⟩ := hω
        exact Set.mem_biUnion hT h
    _ ≤ ∑ T ∈ (classSet n : Finset (DTree α)),
          μ {ω | G < D.real (PotGood M U θ k (read · ω) T)} := measure_biUnion_finset_le _ _
    _ ≤ ∑ _T ∈ (classSet n : Finset (DTree α)), ENNReal.ofReal e :=
        Finset.sum_le_sum fun T hT => noise_one M U θ μ read D k L p₀ G hmeas hind hU hL hp₀ hpre
          hθ T n (by have := mids_card T; have := classSet_leaves n T hT; omega)
    _ = (classSet n : Finset (DTree α)).card * ENNReal.ofReal e := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ ((1 + (n + 1) ^ 3 * Fintype.card α) ^ n : ℕ) * ENNReal.ofReal e := by
        gcongr
        exact_mod_cast classSet_card n
    _ = _ := by
        rw [noiseRisk, ENNReal.ofReal_mul (Nat.cast_nonneg _), ENNReal.ofReal_natCast]

end Random

end OrthoDFA
