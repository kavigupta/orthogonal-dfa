import OrthoDFA.LowLevelClaims.LoopDFA
import OrthoDFA.Proofs.LoopTermination
import OrthoDFA.Proofs.CoveredStart
import OrthoDFA.Proofs.UniformStrings

/-!
# From a checked round to the learner's DFA

The probes are `D`'s strings cut at a length uniform over `k` to `L₁` (`windowLaw`). A probe of
length `i` and its extension by a letter that both agree, read nothing on a rarer side past a
midfix and stop undecided at neither their start's sift nor their own, walk to the true leaves of
their states, so the edge between those leaves is learned and true (`edge_of_pair`). Where `D`
extends each prefix by each letter at least `1/Ae` of the time, a prefix's mass is at most `Ae`
times its extension's (`prefixLaw_snoc_le`; `Ae = |Σ|` for uniform strings, `uniformStrings_floor`).
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

section Window

variable [Nonempty α]

/-- The average of `g` over the strings of length `ℓ`. -/
def uavg (ℓ : ℕ) (g : FreeMonoid α → ℝ) : ℝ :=
  (∑ v : List.Vector α ℓ, g (FreeMonoid.ofList v.toList)) / (Fintype.card α : ℝ) ^ ℓ

theorem integral_uniformStrings (ℓ : ℕ) (g : FreeMonoid α → ℝ) :
    ∫ x, g x ∂(uniformStrings ℓ) = uavg ℓ g := by
  unfold uniformStrings uavg
  rw [integral_smul_measure, integral_finsetSum_measure fun _ _ => integrable_dirac (by simp)]
  simp only [integral_dirac, ENNReal.toReal_inv, ENNReal.toReal_pow, ENNReal.toReal_natCast,
    smul_eq_mul]
  rw [div_eq_inv_mul]

theorem uniformStrings_real_uavg (ℓ : ℕ) (S : Set (FreeMonoid α)) :
    (uniformStrings ℓ).real S = uavg ℓ (S.indicator 1) := by
  rw [← integral_uniformStrings, integral_indicator_one (MeasurableSpace.measurableSet_top)]

theorem uavg_nonneg (ℓ : ℕ) {g : FreeMonoid α → ℝ} (hg : ∀ x, 0 ≤ g x) : 0 ≤ uavg ℓ g :=
  div_nonneg (Finset.sum_nonneg fun _ _ => hg _) (by positivity)

theorem uavg_sum {ι : Type*} (ℓ : ℕ) (s : Finset ι) (g : ι → FreeMonoid α → ℝ) :
    uavg ℓ (fun x => ∑ i ∈ s, g i x) = ∑ i ∈ s, uavg ℓ (g i) := by
  unfold uavg
  rw [Finset.sum_comm, Finset.sum_div]

/-- The strings of length `j + t`, split after `j` letters. -/
def vecSplit (j t : ℕ) : List.Vector α (j + t) ≃ List.Vector α j × List.Vector α t where
  toFun v := (⟨v.toList.take j, by simp⟩, ⟨v.toList.drop j, by simp⟩)
  invFun w := ⟨w.1.toList ++ w.2.toList, by simp⟩
  left_inv v := by apply List.Vector.toList_injective; simp
  right_inv w := by
    obtain ⟨⟨w₁, h₁⟩, ⟨w₂, h₂⟩⟩ := w
    refine Prod.ext (List.Vector.toList_injective ?_) (List.Vector.toList_injective ?_)
    · simp [List.Vector.toList, ← h₁]
    · simp [List.Vector.toList, ← h₁]

theorem uavg_split (j t : ℕ) (g : FreeMonoid α → ℝ) :
    uavg (j + t) g = (∑ w : List.Vector α j, ∑ z : List.Vector α t,
      g (FreeMonoid.ofList (w.toList ++ z.toList))) / (Fintype.card α : ℝ) ^ (j + t) := by
  unfold uavg
  congr 1
  rw [← Fintype.sum_prod_type', ← (vecSplit j t).sum_comp]
  refine Fintype.sum_congr _ _ fun v => ?_
  rw [show ((vecSplit j t v).1.toList ++ (vecSplit j t v).2.toList) = v.toList from
    congrArg List.Vector.toList ((vecSplit j t).symm_apply_apply v)]

/-- A prefix of a uniform string is uniform, in average. -/
theorem uavg_prefix {j ℓ : ℕ} (h : j ≤ ℓ) (g : FreeMonoid α → ℝ) :
    uavg ℓ (fun x => g (prefixOf x j)) = uavg j g := by
  obtain ⟨t, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [uavg_split]
  have e : ∀ (w : List.Vector α j) (z : List.Vector α t),
      prefixOf (FreeMonoid.ofList (w.toList ++ z.toList)) j = FreeMonoid.ofList w.toList := by
    intro w z
    simp [prefixOf]
  simp only [e, Finset.sum_const, Finset.card_univ, card_vector, nsmul_eq_mul]
  unfold uavg
  rw [← Finset.mul_sum, pow_add]
  have : (0 : ℝ) < Fintype.card α ^ t := by
    have : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
    positivity
  push_cast
  field_simp

theorem uavg_append (j t : ℕ) (h : FreeMonoid α → ℝ) :
    ∑ z : List.Vector α t, uavg j (fun y => h (y * FreeMonoid.ofList z.toList))
      = (Fintype.card α : ℝ) ^ t * uavg (j + t) h := by
  rw [uavg_split]
  unfold uavg
  rw [← Finset.sum_div, Finset.sum_comm, pow_add]
  have : (0 : ℝ) < Fintype.card α ^ t := by
    have : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
    positivity
  simp only [← FreeMonoid.ofList_append]
  field_simp

/-- A prefix of a uniform string is uniform. -/
theorem uniformStrings_prefix {j n : ℕ} (h : j ≤ n) (S : Set (FreeMonoid α)) :
    (uniformStrings n).real {x | prefixOf x j ∈ S} = (uniformStrings j).real S := by
  rw [uniformStrings_real_uavg, uniformStrings_real_uavg, ← uavg_prefix h]
  rfl

theorem uniformStrings_snoc_le (i : ℕ) (a : α) (S : Set (FreeMonoid α)) :
    (uniformStrings i).real {y | y * FreeMonoid.of a ∈ S}
      ≤ Fintype.card α * (uniformStrings (i + 1)).real S := by
  rw [uniformStrings_real_uavg, uniformStrings_real_uavg]
  have h := uavg_append (α := α) i 1 (S.indicator 1)
  rw [pow_one] at h
  rw [← h]
  have e : uavg i ({y | y * FreeMonoid.of a ∈ S}.indicator 1)
      = uavg i (fun y => S.indicator 1 (y * FreeMonoid.ofList
        (a ::ᵥ List.Vector.nil).toList)) := by
    unfold uavg
    congr 1
  rw [e]
  exact Finset.single_le_sum (f := fun z : List.Vector α 1 =>
    uavg i (fun y => S.indicator 1 (y * FreeMonoid.ofList z.toList)))
    (fun z _ => uavg_nonneg i fun y => Set.indicator_nonneg (fun _ _ => zero_le_one) _)
    (Finset.mem_univ (a ::ᵥ List.Vector.nil))

end Window

section Laws

variable (D : Measure (FreeMonoid α))

theorem prefixLaw_apply (n : ℕ) (S : Set (FreeMonoid α)) :
    prefixLaw D n S = D {x | prefixOf x n ∈ S} :=
  Measure.map_apply measurable_from_top MeasurableSpace.measurableSet_top

theorem prefixLaw_real (n : ℕ) (S : Set (FreeMonoid α)) :
    (prefixLaw D n).real S = D.real {x | prefixOf x n ∈ S} := by
  rw [measureReal_def, prefixLaw_apply]
  rfl

instance [IsProbabilityMeasure D] (n : ℕ) : IsProbabilityMeasure (prefixLaw D n) :=
  Measure.isProbabilityMeasure_map measurable_from_top.aemeasurable

omit [Fintype α] [DecidableEq α] in
theorem prefixOf_prefixOf (x : FreeMonoid α) {j n : ℕ} (h : j ≤ n) :
    prefixOf (prefixOf x n) j = prefixOf x j := by
  simp [prefixOf, List.take_take, min_eq_left h]

theorem prefixLaw_length {n : ℕ} (h : ∀ᵐ x ∂D, n ≤ x.toList.length) :
    prefixLaw D n {x | x.toList.length ≠ n} = 0 := by
  rw [prefixLaw_apply]
  refine measure_mono_null (fun x hx => ?_) (ae_iff.1 h)
  simp only [Set.mem_setOf_eq, length_prefixOf] at hx ⊢
  omega

theorem integrable_of_mem_Icc {ν : Measure (FreeMonoid α)} [IsFiniteMeasure ν]
    {g : FreeMonoid α → ℝ} (hg : ∀ x, g x ∈ Set.Icc (0 : ℝ) 1) : Integrable g ν :=
  Integrable.of_bound measurable_from_top.aestronglyMeasurable 1
    (ae_of_all _ fun x => by
      rw [Real.norm_eq_abs, abs_le]; exact ⟨by linarith [(hg x).1], (hg x).2⟩)

variable [IsProbabilityMeasure D]

theorem windowLaw_isProbability {lo hi : ℕ} (h : lo ≤ hi) :
    IsProbabilityMeasure (windowLaw D lo hi) := by
  constructor
  unfold windowLaw
  rw [Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply]
  simp only [measure_univ, Finset.sum_const, Nat.card_Icc, nsmul_eq_mul, mul_one, smul_eq_mul]
  exact ENNReal.inv_mul_cancel (Nat.cast_ne_zero.2 (by omega)) (ENNReal.natCast_ne_top _)

theorem windowLaw_real {lo hi : ℕ} (h : lo ≤ hi) (S : Set (FreeMonoid α)) :
    (windowLaw D lo hi).real S
      = (∑ n ∈ Finset.Icc lo hi, (prefixLaw D n).real S) / ((hi : ℝ) + 1 - lo) := by
  have hN : ((hi + 1 - lo : ℕ) : ℝ) = (hi : ℝ) + 1 - lo := by
    rw [Nat.cast_sub (by omega)]; push_cast; ring
  rw [measureReal_def]
  unfold windowLaw
  rw [Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply, smul_eq_mul, ENNReal.toReal_mul,
    ENNReal.toReal_inv, ENNReal.toReal_natCast, ENNReal.toReal_sum fun n _ => measure_ne_top _ _,
    hN, div_eq_inv_mul]
  rfl

theorem integral_windowLaw {lo hi : ℕ} (h : lo ≤ hi) {g : FreeMonoid α → ℝ}
    (hg : ∀ x, g x ∈ Set.Icc (0 : ℝ) 1) :
    ∫ x, g x ∂(windowLaw D lo hi)
      = (∑ n ∈ Finset.Icc lo hi, ∫ x, g x ∂(prefixLaw D n)) / ((hi : ℝ) + 1 - lo) := by
  have hN : ((hi + 1 - lo : ℕ) : ℝ) = (hi : ℝ) + 1 - lo := by
    rw [Nat.cast_sub (by omega)]; push_cast; ring
  unfold windowLaw
  rw [integral_smul_measure, integral_finsetSum_measure fun _ _ => integrable_of_mem_Icc hg]
  simp only [ENNReal.toReal_inv, ENNReal.toReal_natCast, smul_eq_mul]
  rw [hN, div_eq_inv_mul]

theorem windowLaw_length {lo hi : ℕ} :
    ∀ᵐ x ∂(windowLaw D lo hi), x.toList.length ≤ hi := by
  rw [ae_iff]
  unfold windowLaw
  rw [Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply]
  have : ∀ n ∈ Finset.Icc lo hi, prefixLaw D n {x | ¬ x.toList.length ≤ hi} = 0 := by
    intro n hn
    simp only [Finset.mem_Icc] at hn
    rw [prefixLaw_apply]
    convert measure_empty (μ := D)
    ext x
    simp only [Set.mem_setOf_eq, length_prefixOf, Set.mem_empty_iff_false, iff_false, not_not]
    omega
  rw [Finset.sum_eq_zero this, smul_zero]

/-- A string's first `k` letters are a window string's, cut at any length from `k` on. -/
theorem windowLaw_prefix {k lo hi : ℕ} (hk : k ≤ lo) (h : lo ≤ hi) (u : FreeMonoid α) :
    (windowLaw D lo hi).real {x | prefixOf x k = u} = D.real {x | prefixOf x k = u} := by
  rw [windowLaw_real D h]
  have hN : (0 : ℝ) < (hi : ℝ) + 1 - lo := by
    have : (lo : ℝ) ≤ hi := by exact_mod_cast h
    linarith
  have e : ∀ n ∈ Finset.Icc lo hi, (prefixLaw D n).real {x | prefixOf x k = u}
      = D.real {x | prefixOf x k = u} := by
    intro n hn
    simp only [Finset.mem_Icc] at hn
    rw [prefixLaw_real]
    simp only [Set.mem_setOf_eq, prefixOf_prefixOf _ (show k ≤ n by omega)]
  rw [Finset.sum_congr rfl e, Finset.sum_const, Nat.card_Icc, nsmul_eq_mul,
    Nat.cast_sub (by omega), div_eq_iff hN.ne']
  push_cast
  ring

/-- `D`'s strings cut at `n ≥ k` letters weigh at most `D`'s heaviest first `k` letters. -/
theorem prefixLaw_singleton_le {k n : ℕ} (hk : k ≤ n) {p₀ : ℝ}
    (hp : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (t : FreeMonoid α) :
    (prefixLaw D n).real {t} ≤ p₀ := by
  rw [prefixLaw_real]
  refine (measureReal_mono (fun x hx => ?_) (measure_ne_top _ _)).trans (hp (prefixOf t k))
  simp only [Set.mem_setOf_eq, Set.mem_singleton_iff] at hx ⊢
  rw [← hx, prefixOf_prefixOf _ hk]

theorem prefixLaw_prefix_le {k n : ℕ} (hk : k ≤ n) {p₀ : ℝ}
    (hp : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (u : FreeMonoid α) :
    (prefixLaw D n).real {x | prefixOf x k = u} ≤ p₀ := by
  rw [prefixLaw_real]
  simp only [Set.mem_setOf_eq, prefixOf_prefixOf _ hk]
  exact hp u

/-- Where `D` extends every string's first `i` letters `y` by `f y` at least `1/Ae` of the time, a
string of `i` letters extended by its letter lands in `S` at most `Ae` times as often as one of
`i + 1`. -/
theorem prefixLaw_snoc_le {i : ℕ} (f : FreeMonoid α → α) {Ae : ℝ} (hAe : 0 ≤ Ae)
    (hfl : ∀ y, D.real {w | prefixOf w i = y}
      ≤ Ae * D.real {w | prefixOf w (i + 1) = y * FreeMonoid.of (f y)})
    (S : Set (FreeMonoid α)) :
    (prefixLaw D i).real {y | y * FreeMonoid.of (f y) ∈ S}
      ≤ Ae * (prefixLaw D (i + 1)).real S := by
  set T : Set (FreeMonoid α) := {y | y * FreeMonoid.of (f y) ∈ S}
  have hcyl : ∀ y : FreeMonoid α, D {w | prefixOf w i = y}
      ≤ ENNReal.ofReal Ae * D {w | prefixOf w (i + 1) = y * FreeMonoid.of (f y)} := by
    intro y
    rw [← ofReal_measureReal, ← ofReal_measureReal, ← ENNReal.ofReal_mul hAe]
    exact ENNReal.ofReal_le_ofReal (hfl y)
  have hdis : Pairwise (Function.onFun Disjoint fun y : T =>
      {w : FreeMonoid α | prefixOf w (i + 1) = (y : FreeMonoid α) * FreeMonoid.of (f y)}) := by
    intro y y' hyy'
    refine Set.disjoint_left.2 fun w h h' => hyy' (Subtype.ext ?_)
    simp only [Set.mem_setOf_eq] at h h'
    have e := congrArg FreeMonoid.toList (h.symm.trans h')
    simp only [FreeMonoid.toList_mul, FreeMonoid.toList_of] at e
    exact FreeMonoid.toList.injective (List.append_inj_left' e rfl)
  have hle : prefixLaw D i T ≤ ENNReal.ofReal Ae * prefixLaw D (i + 1) S := by
    rw [prefixLaw_apply, prefixLaw_apply]
    calc D {x | prefixOf x i ∈ T}
        = ∑' y : T, D ((fun x => prefixOf x i) ⁻¹' {(y : FreeMonoid α)}) :=
          (tsum_measure_preimage_singleton (Set.to_countable T)
            fun _ _ => MeasurableSpace.measurableSet_top).symm
      _ ≤ ∑' y : T, ENNReal.ofReal Ae
            * D {w | prefixOf w (i + 1) = (y : FreeMonoid α) * FreeMonoid.of (f y)} :=
          ENNReal.tsum_le_tsum fun y => hcyl y
      _ = ENNReal.ofReal Ae * D (⋃ y : T,
            {w | prefixOf w (i + 1) = (y : FreeMonoid α) * FreeMonoid.of (f y)}) := by
          rw [ENNReal.tsum_mul_left,
            measure_iUnion hdis fun _ => MeasurableSpace.measurableSet_top]
      _ ≤ ENNReal.ofReal Ae * D {x | prefixOf x (i + 1) ∈ S} := by
          gcongr
          intro w hw
          obtain ⟨y, hy⟩ := Set.mem_iUnion.1 hw
          simp only [Set.mem_setOf_eq] at hy ⊢
          rw [hy]
          exact y.2
  rw [measureReal_def, measureReal_def]
  refine (ENNReal.toReal_mono (ENNReal.mul_ne_top ENNReal.ofReal_ne_top (measure_ne_top _ _))
    hle).trans (le_of_eq ?_)
  rw [ENNReal.toReal_mul, ENNReal.toReal_ofReal hAe]

/-- A covered state is visited `c` of the time, summed over the window. -/
theorem occupancy_of_covered {σ : Type*} (M : _root_.DFA α σ) {L : ℕ} {c : ℝ} {q : σ}
    (hq : q ∈ coveredStatesOn (prefixLaw D L) M L c) :
    c ≤ ∑ i ∈ Finset.Icc ((L + 1) / 2) L, (prefixLaw D i).real {y | M.eval y.toList = q} := by
  refine hq.trans ?_
  change (prefixLaw D L).real _ ≤ _
  rw [prefixLaw_real]
  have hsub : {x : FreeMonoid α | prefixOf x L ∈ {x : FreeMonoid α | ∃ j, (L + 1) / 2 ≤ j ∧ j ≤ L
        ∧ M.eval (prefixOf x j).toList = q}}
      ⊆ ⋃ j ∈ Finset.Icc ((L + 1) / 2) L, {x | prefixOf x j ∈ {y | M.eval y.toList = q}} := by
    rintro x ⟨j, h1, h2, h3⟩
    refine Set.mem_biUnion (Finset.mem_Icc.2 ⟨h1, h2⟩) ?_
    simp only [Set.mem_setOf_eq]
    rwa [prefixOf_prefixOf _ h2] at h3
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans
    ((measureReal_biUnion_finset_le _ _).trans (le_of_eq (Finset.sum_congr rfl fun j _ => ?_)))
  rw [prefixLaw_real]

end Laws

section Uniform

variable [Nonempty α]

theorem prefixLaw_uniformStrings {n m : ℕ} (h : n ≤ m) :
    prefixLaw (uniformStrings (α := α) m) n = uniformStrings n := by
  ext S _
  rw [prefixLaw_apply]
  have e := uniformStrings_prefix (α := α) h S
  rw [measureReal_def, measureReal_def] at e
  exact (ENNReal.toReal_eq_toReal_iff' (measure_ne_top _ _) (measure_ne_top _ _)).1 e

theorem windowLaw_uniformStrings {lo hi m : ℕ} (h : hi ≤ m) :
    windowLaw (uniformStrings (α := α) m) lo hi = windowStrings lo hi := by
  unfold windowLaw windowStrings
  congr 1
  exact Finset.sum_congr rfl fun n hn => prefixLaw_uniformStrings (by simp at hn; omega)

theorem uniformStrings_long (m : ℕ) : ∀ᵐ x ∂(uniformStrings (α := α) m), m ≤ x.toList.length := by
  refine ae_iff.2 (measure_mono_null (fun x hx => ?_) (uniformStrings_length m))
  simp only [Set.mem_setOf_eq] at hx ⊢
  omega

theorem uniformStrings_prefix_le {k m : ℕ} (h : k ≤ m) (u : FreeMonoid α) :
    (uniformStrings m).real {x | prefixOf x k = u} ≤ ((Fintype.card α : ℝ) ^ k)⁻¹ := by
  have e := uniformStrings_prefix (α := α) h {u}
  simp only [Set.mem_singleton_iff] at e
  rw [e, ← one_div]
  exact uniformStrings_singleton_le k u

/-- Uniform strings extend every prefix by each letter `1/|Σ|` of the time. -/
theorem uniformStrings_floor {i m : ℕ} (h : i + 1 ≤ m) (y : FreeMonoid α) (a : α) :
    (uniformStrings m).real {w | prefixOf w i = y}
      ≤ Fintype.card α
        * (uniformStrings m).real {w | prefixOf w (i + 1) = y * FreeMonoid.of a} := by
  have e1 := uniformStrings_prefix (α := α) (show i ≤ m by omega) {y}
  have e2 := uniformStrings_prefix (α := α) h {y * FreeMonoid.of a}
  simp only [Set.mem_singleton_iff] at e1 e2
  rw [e1, e2]
  have hs : ({y} : Set (FreeMonoid α))
      = {y' | y' * FreeMonoid.of a ∈ ({y * FreeMonoid.of a} : Set _)} := by
    ext y'
    simp only [Set.mem_singleton_iff, Set.mem_setOf_eq]
    exact ⟨fun h => h ▸ rfl, fun h => mul_right_cancel h⟩
  rw [hs]
  exact uniformStrings_snoc_le i a _

end Uniform


section Edges

variable {σ : Type*} (M : _root_.DFA α σ) (dist : σ → ARU → ℝ) (rd : FreeMonoid α → ARU)

omit [Fintype α] [DecidableEq α] in
theorem walk_snoc (E : Edges α) : ∀ (l : List α) (c : α) (p : List Bool),
    (walk E p (l ++ [c])).length = l.length + 2 →
      ∃ w, E ((walk E p l).getLast?.getD [], c) = some ((walk E p (l ++ [c])).getLast?.getD [], w)
  | [], c, p, h => by
    simp only [List.nil_append, walk] at h ⊢
    rcases he : E (p, c) with _ | ⟨q, y⟩ <;> simp only [he] at h ⊢
    · simp at h
    · exact ⟨y, he⟩
  | d :: l, c, p, h => by
    simp only [List.cons_append, walk] at h ⊢
    rcases he : E (p, d) with _ | ⟨q, y⟩ <;> simp only [he] at h ⊢
    · simp at h
    · obtain ⟨w, hw⟩ := walk_snoc E l c q (by simpa using h)
      obtain ⟨l₁, hl₁⟩ := walk_cons E q l
      obtain ⟨l₂, hl₂⟩ := walk_cons E q (l ++ [c])
      refine ⟨w, ?_⟩
      rw [hl₁, hl₂] at hw ⊢
      simpa [List.getLast?_cons] using hw

theorem cut_none_iff (v : ARU) : v.cut = none ↔ v = .undecided := by
  cases v <;> simp [ARU.cut]

theorem walk_facts_full (E : Edges α) (k : ℕ) (x : FreeMonoid α) {p₀ : List Bool}
    (hk : k ≤ x.toList.length) :
    let ss := walk E p₀ (x.toList.drop k)
    0 < ss.length ∧ k + ss.length - 1 ≤ x.toList.length ∧ ss.getD 0 [] = p₀
      ∧ (∀ c, x.toList[k + ss.length - 1]? = some c → E (ss.getD (ss.length - 1) [], c) = none)
      ∧ (x.toList[k + ss.length - 1]? = none → k + ss.length - 1 = x.toList.length) := by
  intro ss
  have hpos : 0 < ss.length := length_walk_pos E p₀ _
  have hle : ss.length ≤ (x.toList.drop k).length + 1 := length_walk_le E p₀ _
  simp only [List.length_drop] at hle
  refine ⟨hpos, by omega, walk_getD_zero E p₀ _, fun c hc => ?_, fun hc => ?_⟩
  · have hstop : ss.length ≤ (x.toList.drop k).length := by
      by_contra hcon
      simp only [List.length_drop] at hcon
      rw [List.getElem?_eq_none (by omega)] at hc
      cases hc
    obtain ⟨c', hc', hE⟩ := walk_stop E p₀ _ hstop
    rw [List.getElem?_drop, show k + (ss.length - 1) = k + ss.length - 1 by omega, hc] at hc'
    cases hc'
    exact hE
  · by_contra hne
    rw [List.getElem?_eq_getElem (by omega)] at hc
    cases hc

/-- A probe neither searching nor stopping undecided at its start or end sift walks, learned all
the way, from its start's sift to its own. -/
theorem agree_of_unchecked (H : HypothesisDFA α) {k : ℕ} {x : FreeMonoid α}
    (hk : k ≤ x.toList.length) (hx : x ∉ Random.searchSet rd k H)
    (hu : x ∉ Random.endsSet rd k H) :
    ∃ p e, H.tree.sift rd (prefixOf x k) = .inl p
      ∧ (walk H.edges p (x.toList.drop k)).length = (x.toList.drop k).length + 1
      ∧ H.tree.sift rd x = .inl e ∧ (walk H.edges p (x.toList.drop k)).getLast? = some e := by
  simp only [Random.searchSet, Random.endsSet, Set.mem_ofPred_eq] at hx hu
  push Not at hu
  rcases hp : H.tree.sift rd (prefixOf x k) with p | z
  swap
  · exfalso
    have : probeOrAgree rd H.tree H.edges k x = .undecidedStart z := by
      simp [probeOrAgree, probe, hp, show ¬ x.toList.length < k by omega]
    rw [this] at hu
    simp [Random.endsString] at hu
  obtain ⟨hpos, hhi, hst0, -, hnone⟩ := walk_facts_full H.edges k x (p₀ := p) hk
  have hlo : H.tree.sift rd (prefixOf x k)
      = .inl ((walk H.edges p (x.toList.drop k)).getD (k - k) []) := by
    rw [Nat.sub_self, hst0, hp]
  rcases Random.probe_long_cases rd H.tree H.edges k x hk hp rfl with
    ⟨c, z, -, -, h⟩ | ⟨un, -, hu2, h'⟩
  · rw [h] at hu; simp [Random.endsString] at hu
  rcases Random.searchOutcome_cases rd H.tree k x _ un (by omega) hhi hlo h' with
    ⟨z, -, h''⟩ | ⟨hsh, h''⟩ | ⟨-, ⟨p', c', t', -, -, -, -, -, -, h''⟩ | ⟨p', z, -, -, -, h''⟩⟩
  · rw [h''] at hu; simp [Random.endsString] at hu
  swap
  · rw [h''] at hx; simp at hx
  swap
  · rw [h''] at hx; simp at hx
  rcases un with _ | ⟨c, t⟩
  swap
  · simp only [searchOutcome] at h''
    rw [h''] at hx; simp at hx
  have hend := hnone (hu2 rfl)
  have hlen : (walk H.edges p (x.toList.drop k)).length = (x.toList.drop k).length + 1 := by
    simp only [List.length_drop]; omega
  refine ⟨p, (walk H.edges p (x.toList.drop k)).getD
    (k + (walk H.edges p (x.toList.drop k)).length - 1 - k) [], rfl, hlen, ?_, ?_⟩
  · rwa [show prefixOf x (k + (walk H.edges p (x.toList.drop k)).length - 1) = x by
      rw [hend, prefixOf_length]] at hsh
  · rw [List.getLast?_eq_getElem?, List.getD_eq_getElem?_getD,
      show k + (walk H.edges p (x.toList.drop k)).length - 1 - k
        = (walk H.edges p (x.toList.drop k)).length - 1 by omega,
      List.getElem?_eq_getElem (by omega)]
    rfl

/-- A probe and its extension by a letter, neither disagreeing nor misled, cross a learned edge
between their states' true leaves. -/
theorem edge_of_pair (H : HypothesisDFA α) {k : ℕ} {y : FreeMonoid α} (a : α)
    (hk : k ≤ y.toList.length) (hy : y ∉ disagreeSet rd k H)
    (hya : y * FreeMonoid.of a ∉ disagreeSet rd k H) (hm : y ∉ misledSet M dist rd k H.tree a) :
    ∃ w, H.edges (Random.leafOf M (fun q => readSide (dist q)) H.tree (M.eval y.toList), a)
      = some (Random.leafOf M (fun q => readSide (dist q)) H.tree
        (M.eval (y * FreeMonoid.of a).toList), w) := by
  have hlen : (y * FreeMonoid.of a).toList.length = y.toList.length + 1 := by simp
  simp only [disagreeSet, Set.mem_union, not_or] at hy hya
  have hr : y ∉ rarerSet M dist rd k H.tree ∧ y * FreeMonoid.of a ∉ rarerSet M dist rd k H.tree :=
    ⟨fun h => hm ⟨hk, .inl h⟩, fun h => hm ⟨hk, .inr h⟩⟩
  obtain ⟨p, e, hp, hf, he, hl⟩ := agree_of_unchecked rd H hk hy.1 hy.2
  obtain ⟨p', e', hp', hf', he', hl'⟩ :=
    agree_of_unchecked rd H (by omega) hya.1 hya.2
  have hpre : prefixOf (y * FreeMonoid.of a) k = prefixOf y k := by
    simp only [prefixOf, FreeMonoid.toList_mul, FreeMonoid.toList_of]
    rw [List.take_append_of_le_length hk]
  rw [hpre, hp] at hp'
  cases hp'
  have hdrop : (y * FreeMonoid.of a).toList.drop k = y.toList.drop k ++ [a] := by
    simp only [FreeMonoid.toList_mul, FreeMonoid.toList_of]
    rw [List.drop_append_of_le_length hk]
  rw [hdrop] at hf' hl'
  obtain ⟨w, hw⟩ := walk_snoc H.edges _ a p (by rw [hf']; simp)
  have hclean : ∀ z : FreeMonoid α, z ∉ rarerSet M dist rd k H.tree →
      k ≤ z.toList.length → ∀ m ∈ Random.mids H.tree,
        rd (z * m) ≠ if readSide (dist (M.eval (z * m).toList)) then .reject else .accept :=
    fun z hz hkz m hm hwr => hz ⟨hkz, m, hm, hwr⟩
  have h1 := Random.Eps.sift_true' M (fun q => readSide (dist q)) rd he (hclean y hr.1 hk)
  have h2 := Random.Eps.sift_true' M (fun q => readSide (dist q)) rd he'
    (hclean _ hr.2 (by omega))
  rw [hl, hl'] at hw
  rw [h1, h2]
  exact ⟨w, hw⟩

/-- The strings shorter than `J` in state `q`. -/
def shortAt (J : ℕ) (q : σ) : Set (FreeMonoid α) := {y | y.toList.length < J ∧ M.eval y.toList = q}

end Edges

end OrthoDFA

end
