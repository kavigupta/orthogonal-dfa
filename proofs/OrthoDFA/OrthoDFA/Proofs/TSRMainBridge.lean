import OrthoDFA.LowLevelClaims.LoopDFATSRMain
import OrthoDFA.Proofs.LoopDFASummed

/-!
# Typical-start replication with `LearnerLoopDFA`'s probes: the bridge

The wrong states' occupancy of the lengths `⌈L/2⌉` to `L₁ - 1` is `wrong_occupancy_le`'s. The run
from the start `q` meets them at a position `i ≤ L`, either at `i ≥ ⌈L/2⌉` or before
(`hit_le_occupancy`). A shortest word `v` from a state to `q` has `|v| < |Q|` (`short_word`), and
the strings beginning with it are `|Σ|^-|v|` of all (`hitSet_shift`). For `i ≥ ⌈L/2⌉`, the word
from `M`'s start puts the meeting at `|v| + i ≤ L + |Q| - 1`. For `i < ⌈L/2⌉`, a string visiting
`q` from `⌈L/2⌉` to `L` is at a state reaching `q` at `⌈L/2⌉` (`hitSet_split`), and that state's
word puts the meeting before `2⌈L/2⌉ + |Q| - 1 ≤ L + |Q|`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Paths

variable [Nonempty α]

theorem uavg_mono {ℓ : ℕ} {g h : FreeMonoid α → ℝ}
    (hgh : ∀ v : List.Vector α ℓ, g (FreeMonoid.ofList v.toList) ≤ h (FreeMonoid.ofList v.toList)) :
    uavg ℓ g ≤ uavg ℓ h :=
  div_le_div_of_nonneg_right (Finset.sum_le_sum fun v _ => hgh v) (by positivity)



/-- A uniform string's prefix of length `j` and the rest are independent. -/
theorem uavg_prefix_mul_drop (j t : ℕ) (f g : FreeMonoid α → ℝ) :
    uavg (j + t) (fun x => f (prefixOf x j) * g (FreeMonoid.ofList (x.toList.drop j)))
      = uavg j f * uavg t g := by
  rw [uavg_split]
  have e1 : ∀ (w : List.Vector α j) (z : List.Vector α t),
      prefixOf (FreeMonoid.ofList (w.toList ++ z.toList)) j = FreeMonoid.ofList w.toList := by
    intro w z
    simp [prefixOf]
  have e2 : ∀ (w : List.Vector α j) (z : List.Vector α t),
      FreeMonoid.ofList ((FreeMonoid.ofList (w.toList ++ z.toList)).toList.drop j)
        = FreeMonoid.ofList z.toList := by
    intro w z
    simp
  simp only [e1, e2]
  unfold uavg
  rw [pow_add, ← Finset.sum_mul_sum, mul_div_mul_comm]


variable {σ : Type*} (M : _root_.DFA α σ) (B : Set σ)

/-- The strings whose run from `s` meets `B` at some position from `a` to `b`. -/
def hitSet (s : σ) (a b : ℕ) : Set (FreeMonoid α) :=
  {y | ∃ i, a ≤ i ∧ i ≤ b ∧ M.evalFrom s (prefixOf y i).toList ∈ B}

theorem hitSet_prefix (s : σ) (a : ℕ) {b n : ℕ} (h : b ≤ n) :
    (uniformStrings n).real (hitSet M B s a b) = (uniformStrings b).real (hitSet M B s a b) := by
  rw [← uniformStrings_prefix h]
  congr 1
  ext y
  simp only [hitSet, Set.mem_setOf_eq]
  constructor
  · rintro ⟨i, h1, h2, h3⟩
    exact ⟨i, h1, h2, by rwa [prefixOf_prefixOf y h2]⟩
  · rintro ⟨i, h1, h2, h3⟩
    exact ⟨i, h1, h2, by rwa [prefixOf_prefixOf y h2] at h3⟩

theorem hitSet_shift (s : σ) (u : List α) (a b t : ℕ) :
    ((Fintype.card α : ℝ) ^ u.length)⁻¹
        * (uniformStrings t).real (hitSet M B (M.evalFrom s u) a b)
      ≤ (uniformStrings (u.length + t)).real (hitSet M B s (u.length + a) (u.length + b)) := by
  classical
  have hf : uavg u.length ({w : FreeMonoid α | w = FreeMonoid.ofList u}.indicator 1)
      = ((Fintype.card α : ℝ) ^ u.length)⁻¹ := by
    unfold uavg
    obtain ⟨v₀, hv₀⟩ : ∃ v₀ : List.Vector α u.length, v₀.toList = u := ⟨⟨u, rfl⟩, rfl⟩
    rw [Finset.sum_eq_single_of_mem v₀ (Finset.mem_univ v₀)]
    · rw [Set.indicator_of_mem (show FreeMonoid.ofList v₀.toList ∈ {w | w = FreeMonoid.ofList u}
        by rw [hv₀]; rfl), Pi.one_apply, one_div]
    · intro v _ hv
      refine Set.indicator_of_notMem (fun h => hv ?_) _
      apply List.Vector.toList_injective
      rw [hv₀]
      exact FreeMonoid.ofList.injective h
  rw [uniformStrings_real_uavg, uniformStrings_real_uavg, ← hf, ← uavg_prefix_mul_drop]
  refine uavg_mono fun v => ?_
  generalize FreeMonoid.ofList v.toList = x
  have hnn : ∀ (S : Set (FreeMonoid α)) (z : FreeMonoid α),
      0 ≤ S.indicator (1 : FreeMonoid α → ℝ) z := fun S z =>
    Set.indicator_nonneg (fun _ _ => zero_le_one) z
  by_cases hp : prefixOf x u.length = FreeMonoid.ofList u
  · by_cases hd : FreeMonoid.ofList (x.toList.drop u.length) ∈ hitSet M B (M.evalFrom s u) a b
    · obtain ⟨i, h1, h2, h3⟩ := hd
      have hmem : x ∈ hitSet M B s (u.length + a) (u.length + b) := by
        refine ⟨u.length + i, by omega, by omega, ?_⟩
        have hp' : x.toList.take u.length = u := by
          simpa [prefixOf] using congrArg FreeMonoid.toList hp
        have e : (prefixOf x (u.length + i)).toList
            = u ++ (prefixOf (FreeMonoid.ofList (x.toList.drop u.length)) i).toList := by
          simp [prefixOf, List.take_add, hp']
        rw [e, DFA.evalFrom_of_append]
        exact h3
      rw [Set.indicator_of_mem (show prefixOf x u.length ∈ {w | w = FreeMonoid.ofList u} from hp),
        Set.indicator_of_mem (show FreeMonoid.ofList (x.toList.drop u.length)
          ∈ hitSet M B (M.evalFrom s u) a b from ⟨i, h1, h2, h3⟩),
        Set.indicator_of_mem hmem]
      simp
    · rw [Set.indicator_of_notMem hd, mul_zero]
      exact hnn _ _
  · rw [Set.indicator_of_notMem (show prefixOf x u.length ∉ {w | w = FreeMonoid.ofList u} from hp),
      zero_mul]
    exact hnn _ _

theorem hitSet_start_le (W : Finset σ) {a b n : ℕ} (h : b ≤ n) :
    (uniformStrings n).real (hitSet M ↑W M.start a b)
      ≤ ∑ q ∈ W, ∑ j ∈ Finset.Icc a b,
        (uniformStrings j).real {y : FreeMonoid α | M.eval y.toList = q} := by
  have hsub : hitSet M ↑W M.start a b ⊆ ⋃ q ∈ W, ⋃ j ∈ Finset.Icc a b,
      {x | prefixOf x j ∈ {y : FreeMonoid α | M.eval y.toList = q}} := by
    rintro x ⟨j, h1, h2, h3⟩
    exact Set.mem_biUnion (Finset.mem_coe.1 h3)
      (Set.mem_biUnion (Finset.mem_Icc.2 ⟨h1, h2⟩) rfl)
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans
    ((measureReal_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun q _ => ?_))
  refine (measureReal_biUnion_finset_le _ _).trans (le_of_eq (Finset.sum_congr rfl fun j hj => ?_))
  exact uniformStrings_prefix (by have := (Finset.mem_Icc.1 hj).2; omega) _

variable [Fintype σ]

theorem hitSet_split (k t : ℕ) :
    ∑ r : σ, (uniformStrings k).real {y : FreeMonoid α | M.eval y.toList = r}
        * (uniformStrings t).real (hitSet M B r 0 t)
      ≤ (uniformStrings (k + t)).real (hitSet M B M.start k (k + t)) := by
  classical
  have hnn : ∀ (S : Set (FreeMonoid α)) (z : FreeMonoid α),
      0 ≤ S.indicator (1 : FreeMonoid α → ℝ) z := fun S z =>
    Set.indicator_nonneg (fun _ _ => zero_le_one) z
  have hle1 : ∀ (S : Set (FreeMonoid α)) (z : FreeMonoid α),
      S.indicator (1 : FreeMonoid α → ℝ) z ≤ 1 := by
    intro S z
    simp only [Set.indicator_apply]
    split_ifs <;> norm_num
  simp only [uniformStrings_real_uavg]
  rw [Finset.sum_congr rfl fun r _ => (uavg_prefix_mul_drop k t _ _).symm, ← uavg_sum]
  refine uavg_mono fun v => ?_
  generalize FreeMonoid.ofList v.toList = x
  set r₀ := M.eval (prefixOf x k).toList
  rw [Finset.sum_eq_single_of_mem r₀ (Finset.mem_univ _) fun r _ hr => by
    rw [Set.indicator_of_notMem (show prefixOf x k ∉ {y : FreeMonoid α | M.eval y.toList = r}
      from fun h => hr h.symm), zero_mul]]
  rw [Set.indicator_of_mem (show prefixOf x k ∈ {y : FreeMonoid α | M.eval y.toList = r₀}
    from rfl), Pi.one_apply, one_mul]
  by_cases hd : FreeMonoid.ofList (x.toList.drop k) ∈ hitSet M B r₀ 0 t
  · obtain ⟨i, -, h2, h3⟩ := hd
    have hmem : x ∈ hitSet M B M.start k (k + t) := by
      refine ⟨k + i, by omega, by omega, ?_⟩
      have e : (prefixOf x (k + i)).toList
          = (prefixOf x k).toList ++ (prefixOf (FreeMonoid.ofList (x.toList.drop k)) i).toList := by
        simp [prefixOf, List.take_add]
      rw [e, DFA.evalFrom_of_append]
      exact h3
    rw [Set.indicator_of_mem hmem]
    exact hle1 _ _
  · rw [Set.indicator_of_notMem hd]
    exact hnn _ _

omit [Fintype α] [DecidableEq α] [Nonempty α] in
theorem short_word (s : σ) : ∀ (n : ℕ) (u : List α), u.length = n →
    ∃ v : List α, v.length + 1 ≤ Fintype.card σ ∧ M.evalFrom s v = M.evalFrom s u := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro u hn
    by_cases hlen : Fintype.card σ ≤ u.length
    · obtain ⟨q, a, b, c, hx, -, hb, hqa, -, hqc⟩ := M.evalFrom_split hlen rfl
      have hlt : (a ++ c).length < n := by
        have := List.length_pos_of_ne_nil hb
        rw [← hn, hx]
        simp only [List.length_append]
        omega
      obtain ⟨v, hv, hve⟩ := ih _ hlt (a ++ c) rfl
      refine ⟨v, hv, ?_⟩
      rw [hve, DFA.evalFrom_of_append, hqa, hqc]
    · exact ⟨u, by omega, rfl⟩

/-- The run from `q`, the window visiting `q` on `w > 0` of the strings, meets `W` within `L`
letters on at most `|Σ|^(|Q|-1)(1 + 1/w)` times `W`'s occupancy of the lengths `⌈L/2⌉` to
`L + |Q| + 2`. -/
theorem hit_le_occupancy (L : ℕ) (q : σ) (W : Finset σ)
    (hw : 0 < windowVisitRate M L q) :
    (uniformStrings L).real {x | ∃ i ≤ L, M.evalFrom q (prefixOf x i).toList ∈ (W : Set σ)}
      ≤ (Fintype.card α : ℝ) ^ (Fintype.card σ - 1) * (1 + 1 / windowVisitRate M L q)
        * ∑ r ∈ W, ∑ j ∈ Finset.Ico ((L + 1) / 2) (L + Fintype.card σ + 3),
          (uniformStrings j).real {y : FreeMonoid α | M.eval y.toList = r} := by
  classical
  set k := (L + 1) / 2 with hkd
  set w := windowVisitRate M L q with hwd
  set A : ℝ := (Fintype.card α : ℝ) with hAd
  set G : ℝ := A ^ (Fintype.card σ - 1) with hGd
  set X : ℝ := ∑ r ∈ W, ∑ j ∈ Finset.Ico k (L + Fintype.card σ + 3),
    (uniformStrings j).real {y : FreeMonoid α | M.eval y.toList = r} with hXd
  set Bad : Set σ := ↑W
  have hA1 : (1 : ℝ) ≤ A := by rw [hAd]; exact_mod_cast Fintype.card_pos
  have hA0 : 0 < A := by linarith
  have hG1 : 1 ≤ G := one_le_pow₀ hA1
  have hG0 : 0 < G := by linarith
  have : Nonempty σ := ⟨M.start⟩
  have hσ : 1 ≤ Fintype.card σ := Fintype.card_pos
  have hX : ∀ a b n, b ≤ n → k ≤ a → b < L + Fintype.card σ + 3 →
      (uniformStrings n).real (hitSet M Bad M.start a b) ≤ X := by
    intro a b n hbn hka hb
    refine (hitSet_start_le M W hbn).trans (Finset.sum_le_sum fun r _ => ?_)
    exact Finset.sum_le_sum_of_subset_of_nonneg (fun j hj => by
      simp only [Finset.mem_Icc, Finset.mem_Ico] at hj ⊢; omega) fun _ _ _ => measureReal_nonneg
  have hpowG : ∀ v : List α, v.length + 1 ≤ Fintype.card σ → A ^ v.length ≤ G := fun v hv =>
    pow_le_pow_right₀ hA1 (by omega)
  -- the hits are at or after `k`, or before
  have hsplit : {x | ∃ i ≤ L, M.evalFrom q (prefixOf x i).toList ∈ Bad}
      ⊆ hitSet M Bad q 0 (k - 1) ∪ hitSet M Bad q k L := by
    rintro x ⟨i, hi, h⟩
    by_cases hik : i < k
    · exact Or.inl ⟨i, Nat.zero_le _, by omega, h⟩
    · exact Or.inr ⟨i, by omega, hi, h⟩
  -- `q` is reached from `M`'s start
  have hreach : ∃ u : List α, M.evalFrom M.start u = q := by
    by_contra hno
    push Not at hno
    have hempty : {x : FreeMonoid α | ∃ j, (L + 1) / 2 ≤ j ∧ j ≤ L
        ∧ M.eval (prefixOf x j).toList = q} = ∅ := by
      ext x
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
      exact fun j _ _ => hno _
    have h := hw
    simp only [hwd, windowVisitRate, hempty, measureReal_empty] at h
    exact lt_irrefl _ h
  -- the late hits, along a shortest word from `M`'s start
  have hlate : (uniformStrings L).real (hitSet M Bad q k L) ≤ G * X := by
    obtain ⟨u, hu⟩ := hreach
    obtain ⟨v, hv, hve⟩ := short_word M M.start _ u rfl
    rw [hu] at hve
    have h1 := hitSet_shift M Bad M.start v k L L
    rw [hve] at h1
    have h2 := hX (v.length + k) (v.length + L) (v.length + L) le_rfl (by omega) (by omega)
    have hpv : 0 < A ^ v.length := pow_pos hA0 _
    have h3 : (uniformStrings L).real (hitSet M Bad q k L) ≤ A ^ v.length * X := by
      have := h1.trans h2
      rw [inv_mul_le_iff₀ hpv] at this
      exact this
    have hX0 : 0 ≤ X := by
      refine Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => measureReal_nonneg
    exact h3.trans (mul_le_mul_of_nonneg_right (hpowG v hv) hX0)
  -- the early hits, after a visit to `q` within `|Q| - 1` letters of `k`
  set t := (Fintype.card σ - 1) + (k - 1) with htd
  set h₁ := (uniformStrings (k - 1)).real (hitSet M Bad q 0 (k - 1)) with hh₁d
  set R : Finset σ := Finset.univ.filter fun r => ∃ u : List α, M.evalFrom r u = q with hRd
  have hR : ∀ r ∈ R, G⁻¹ * h₁ ≤ (uniformStrings t).real (hitSet M Bad r 0 t) := by
    intro r hr
    obtain ⟨u, hu⟩ := (Finset.mem_filter.1 hr).2
    obtain ⟨v, hv, hve⟩ := short_word M r _ u rfl
    rw [hu] at hve
    have h1 := hitSet_shift M Bad r v 0 (k - 1) (k - 1)
    rw [hve, add_zero] at h1
    have h2 : (uniformStrings (v.length + (k - 1))).real
        (hitSet M Bad r v.length (v.length + (k - 1)))
        = (uniformStrings t).real (hitSet M Bad r v.length (v.length + (k - 1))) :=
      (hitSet_prefix M Bad r _ (by omega)).symm
    have h3 : (uniformStrings t).real (hitSet M Bad r v.length (v.length + (k - 1)))
        ≤ (uniformStrings t).real (hitSet M Bad r 0 t) :=
      measureReal_mono (fun y ⟨i, hi1, hi2, hi3⟩ => ⟨i, Nat.zero_le _, by omega, hi3⟩)
        (measure_ne_top _ _)
    have h4 : G⁻¹ ≤ (A ^ v.length)⁻¹ := inv_anti₀ (pow_pos hA0 _) (hpowG v hv)
    have h0 : 0 ≤ h₁ := measureReal_nonneg
    calc G⁻¹ * h₁ ≤ (A ^ v.length)⁻¹ * h₁ := mul_le_mul_of_nonneg_right h4 h0
      _ ≤ (uniformStrings (v.length + (k - 1))).real
          (hitSet M Bad r v.length (v.length + (k - 1))) := h1
      _ = _ := h2
      _ ≤ _ := h3
  have hwR : w ≤ ∑ r ∈ R, (uniformStrings k).real {y : FreeMonoid α | M.eval y.toList = r} := by
    show (uniformStrings L).real {x : FreeMonoid α | ∃ j, k ≤ j ∧ j ≤ L
      ∧ M.eval (prefixOf x j).toList = q} ≤ _
    have hsub : {x : FreeMonoid α | ∃ j, k ≤ j ∧ j ≤ L
        ∧ M.eval (prefixOf x j).toList = q}
        ⊆ ⋃ r ∈ R, {x | prefixOf x k ∈ {y : FreeMonoid α | M.eval y.toList = r}} := by
      rintro x ⟨j, h1, h2, h3⟩
      refine Set.mem_biUnion (x := M.eval (prefixOf x k).toList)
        (Finset.mem_filter.2 ⟨Finset.mem_univ _, ⟨(x.toList.take j).drop k, ?_⟩⟩)
        (show M.eval (prefixOf x k).toList = _ from rfl)
      have e : (prefixOf x j).toList = (prefixOf x k).toList ++ (x.toList.take j).drop k := by
        have := List.take_append_drop k (x.toList.take j)
        rw [List.take_take, min_eq_left h1] at this
        simpa [prefixOf] using this.symm
      rw [DFA.eval, ← DFA.evalFrom_of_append, ← e]
      exact h3
    refine (measureReal_mono hsub (measure_ne_top _ _)).trans
      ((measureReal_biUnion_finset_le _ _).trans (le_of_eq (Finset.sum_congr rfl fun r _ => ?_)))
    exact uniformStrings_prefix (by omega) _
  have hearly : G⁻¹ * h₁ * w ≤ X := by
    have hnn : ∀ r, 0 ≤ (uniformStrings k).real {y : FreeMonoid α | M.eval y.toList = r}
        * (uniformStrings t).real (hitSet M Bad r 0 t) := fun r =>
      mul_nonneg measureReal_nonneg measureReal_nonneg
    have hc : 0 ≤ G⁻¹ * h₁ := mul_nonneg (inv_nonneg.2 hG0.le) measureReal_nonneg
    calc G⁻¹ * h₁ * w
        ≤ G⁻¹ * h₁ * ∑ r ∈ R, (uniformStrings k).real {y : FreeMonoid α | M.eval y.toList = r} :=
          mul_le_mul_of_nonneg_left hwR hc
      _ = ∑ r ∈ R, (uniformStrings k).real {y : FreeMonoid α | M.eval y.toList = r}
            * (G⁻¹ * h₁) := by
          rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun _ _ => mul_comm _ _
      _ ≤ ∑ r ∈ R, (uniformStrings k).real {y : FreeMonoid α | M.eval y.toList = r}
            * (uniformStrings t).real (hitSet M Bad r 0 t) :=
          Finset.sum_le_sum fun r hr => mul_le_mul_of_nonneg_left (hR r hr) measureReal_nonneg
      _ ≤ ∑ r : σ, (uniformStrings k).real {y : FreeMonoid α | M.eval y.toList = r}
            * (uniformStrings t).real (hitSet M Bad r 0 t) :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) fun r _ _ => hnn r
      _ ≤ (uniformStrings (k + t)).real (hitSet M Bad M.start k (k + t)) := hitSet_split M Bad k t
      _ ≤ X := hX k (k + t) (k + t) le_rfl le_rfl (by omega)
  have hearly' : (uniformStrings L).real (hitSet M Bad q 0 (k - 1)) ≤ G * X / w := by
    rw [hitSet_prefix M Bad q 0 (by omega : k - 1 ≤ L), ← hh₁d, le_div_iff₀ hw]
    have : h₁ * w = G * (G⁻¹ * h₁ * w) := by
      rw [← mul_assoc, ← mul_assoc, mul_inv_cancel₀ hG0.ne', one_mul]
    rw [this]
    exact mul_le_mul_of_nonneg_left hearly hG0.le
  calc (uniformStrings L).real {x | ∃ i ≤ L, M.evalFrom q (prefixOf x i).toList ∈ Bad}
      ≤ (uniformStrings L).real (hitSet M Bad q 0 (k - 1) ∪ hitSet M Bad q k L) :=
        measureReal_mono hsplit (measure_ne_top _ _)
    _ ≤ G * X / w + G * X := (measureReal_union_le _ _).trans (add_le_add hearly' hlate)
    _ = G * (1 + 1 / w) * X := by
        have := hw.ne'
        field_simp
        ring

end Paths

section Bridge

variable [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ]

/-- A last round over `LearnerLoopDFA`'s probes, its misled probes counted among each state's
strings shorter than `L + |Q| + 3`, has a leaf from which its hypothesis misjudges at most
`γ + 2|Σ|^(|Q|-1) N(2((|Σ|+1)εB + |Q|εm) + 2εr + εu)/m`, where a state visited `m` of the time
re-roots the target with error at most `γ`. -/
theorem dfa_of_checks_tsrm (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) {K : ℕ} (o : LoopRound Ω α K)
    (L : ℕ) {m εr εu εB εm γ : ℝ}
    (hch : o.Checks M O (windowLaw (uniformStrings (L + Fintype.card σ + 3 + Fintype.card σ + 3))
      ((L + 1) / 2) (L + Fintype.card σ + 3)) ((L + 1) / 2) εr εu εB
      (shortAt M (L + Fintype.card σ + 3)) εm) (hεm : 0 ≤ εm) (hm : 0 < m)
    (hstart : ∃ q, m ≤ windowVisitRate M L q ∧ 1 - γ ≤ rerootAccuracy M L Set.univ q) :
    ∃ s ∈ o.result.hypothesis.tree.leaves, (uniformStrings L).real
      {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)}
        ≤ γ + 2 * (Fintype.card α : ℝ) ^ (Fintype.card σ - 1)
          * (((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
            * (2 * (((Fintype.card α : ℝ) + 1) * εB + Fintype.card σ * εm)
              + (2 * εr + εu))) / m := by
  classical
  set k := (L + 1) / 2 with hkd
  set L₁ := L + Fintype.card σ + 3 with hL₁d
  set H := o.result.hypothesis with hHd
  set side : σ → Bool :=
    fun q => readSide (familyReadDist O M o.family o.state.lo (o.state.hi + 1) q)
  have hroot : ∃ r a, H.tree = .node 1 r a := rootOne_of_classSet _ _ hch.2.2.1
  obtain ⟨B, hBE, hBS, hB⟩ := wrong_occupancy_le M O hL o L
    (uniformStrings (L₁ + Fintype.card σ + 3)) (J := L₁) (by omega) le_rfl
    ((uniformStrings_long _).mono fun x hx => by omega) (Nat.cast_nonneg _)
    (fun i _ hi y a => uniformStrings_floor (by omega) y a) hch hεm
  set X : ℝ := ((L₁ : ℕ) + 1 - (k : ℕ))
    * (2 * (((Fintype.card α : ℝ) + 1) * εB + Fintype.card σ * εm) + (2 * εr + εu)) with hXd
  have hWrong : ∑ q ∈ B, ∑ i ∈ Finset.Ico k L₁,
      (uniformStrings i).real {y : FreeMonoid α | M.eval y.toList = q} ≤ X := by
    refine (Finset.sum_congr rfl fun q _ => Finset.sum_congr rfl fun i hi => ?_).trans_le hB
    rw [prefixLaw_uniformStrings (by simp only [Finset.mem_Ico] at hi; omega)]
  have hX0 : 0 ≤ X := (Finset.sum_nonneg fun q _ => Finset.sum_nonneg fun _ _ =>
    measureReal_nonneg).trans hWrong
  -- the run from the start meets a wrong state on at most `2|Σ|^(|Q|-1) X / m` of the strings
  obtain ⟨q₀, hw, he⟩ := hstart
  set Hit : Set (FreeMonoid α) :=
    {x | ∃ i ≤ L, M.evalFrom q₀ (prefixOf x i).toList ∈ (B : Set σ)} with hHitd
  set G : ℝ := (Fintype.card α : ℝ) ^ (Fintype.card σ - 1) with hGd
  have hG0 : 0 < G := by
    have : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
    positivity
  have hhit : (uniformStrings L).real Hit ≤ 2 * G * X / m := by
    have hw0 : 0 < windowVisitRate M L q₀ := hm.trans_le hw
    have hw1 : windowVisitRate M L q₀ ≤ 1 := measureReal_le_one
    have h1 := hit_le_occupancy M L q₀ B hw0
    have h2 : 1 + 1 / windowVisitRate M L q₀ ≤ 2 / m := by
      have : 1 ≤ 1 / windowVisitRate M L q₀ := by rw [le_div_iff₀ hw0]; linarith
      have : 1 / windowVisitRate M L q₀ ≤ 1 / m := one_div_le_one_div_of_le hm hw
      have e : 2 / m = 1 / m + 1 / m := by ring
      linarith
    calc (uniformStrings L).real Hit
        ≤ G * (1 + 1 / windowVisitRate M L q₀) * ∑ q ∈ B, ∑ i ∈ Finset.Ico k L₁,
            (uniformStrings i).real {y : FreeMonoid α | M.eval y.toList = q} := h1
      _ ≤ G * (2 / m) * X :=
          mul_le_mul (mul_le_mul_of_nonneg_left h2 hG0.le) hWrong
            (Finset.sum_nonneg fun q _ => Finset.sum_nonneg fun _ _ => measureReal_nonneg)
            (by positivity)
      _ = 2 * G * X / m := by ring
  -- off the runs through a wrong state, the start's true leaf classifies as re-rooting does
  set Gd : Set (FreeMonoid α) := {x | (∀ i ≤ L, M.evalFrom q₀ (prefixOf x i).toList ∈ Set.univ)
    ∧ (M.evalFrom q₀ x.toList ∈ M.accept ↔ M.eval x.toList ∈ M.accept)} with hGdd
  have hsub : {x | ¬ H.Classifies (Random.leafOf M side H.tree q₀) x (M.eval x.toList ∈ M.accept)}
      ⊆ (Gdᶜ ∪ Hit) ∪ {x | x.toList.length ≠ L} := by
    intro x hx
    by_contra hcon
    simp only [Set.mem_union, Set.mem_compl_iff, Set.mem_setOf_eq, not_or, not_not] at hcon
    obtain ⟨⟨hag, hH⟩, hl⟩ := hcon
    exact hx (classifies_of_matches M side H hroot {q | q ∉ B} hBE hBS q₀ x
      (fun i hi hb => hH ⟨i, hl ▸ hi, Finset.mem_coe.2 hb⟩) hag.2)
  have hnull : (uniformStrings (α := α) L).real {x | x.toList.length ≠ L} = 0 := by
    rw [measureReal_def, uniformStrings_length, ENNReal.toReal_zero]
  refine ⟨Random.leafOf M side H.tree q₀, Random.leafOf_mem M side _ _, ?_⟩
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans ((measureReal_union_le _ _).trans ?_)
  rw [hnull, add_zero]
  refine (measureReal_union_le _ _).trans ?_
  rw [measureReal_compl MeasurableSpace.measurableSet_top, probReal_univ]
  have : 1 - γ ≤ (uniformStrings L).real Gd := he
  have e : 2 * G * X / m = 2 * (Fintype.card α : ℝ) ^ (Fintype.card σ - 1)
      * (((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
        * (2 * (((Fintype.card α : ℝ) + 1) * εB + Fintype.card σ * εm) + (2 * εr + εu))) / m :=
    rfl
  linarith

/-- The wrong states' cost at `ε = c/(4(A+Q+3)N)`, `c = 2ξm/(5G)`, against `ξ/2`. -/
lemma tsrm_bnd {A Q G N m ξ ε εcov : ℝ} (hA : 0 ≤ A) (hQ : 0 ≤ Q) (hG : 0 < G) (hN : 0 < N)
    (hm : 0 < m) (he : ε = 2 * ξ * m / (5 * G) / (4 * (A + Q + 2 + 1) * N)) (hεc0 : 0 ≤ εcov)
    (hεc : εcov ≤ ε) :
    2 * G * (N * (2 * ((A + 1) * (ε + ε / 96) + Q * (ε + ε / 96))
      + (2 * (1025 / 1024 * εcov) + 3 / 8 * εcov))) / m ≤ ξ / 2 := by
  have hq : 10 * G * (A + Q + 3) * (ε * N) = ξ * m := by rw [he]; field_simp; ring
  have hY : εcov * N ≤ ε * N := mul_le_mul_of_nonneg_right hεc hN.le
  have hu0 : 0 ≤ ε * N := (mul_nonneg hεc0 hN.le).trans hY
  rw [div_le_iff₀ hm]
  have e : 2 * G * (N * (2 * ((A + 1) * (ε + ε / 96) + Q * (ε + ε / 96))
      + (2 * (1025 / 1024 * εcov) + 3 / 8 * εcov)))
      = 2 * G * (97 / 48 * (A + Q + 1) * (ε * N) + 1217 / 512 * (εcov * N)) := by ring
  rw [e]
  have h1 : G * (εcov * N) ≤ G * (ε * N) := mul_le_mul_of_nonneg_left hY hG.le
  have h2 : 0 ≤ G * (A + Q) * (ε * N) := by positivity
  have h3 : 0 ≤ G * (ε * N) := mul_nonneg hG.le hu0
  nlinarith

/-- `dfa_of_checks_tsrm` at the rates `LearnerLoopDFATSRMainSimple` sets, as
`learner_loop_dfa_dist_core`'s bridge: `c = 2ξm/(5|Σ|^(|Q|-1))`, edges crossed at `|Σ| + |Q| + 2`
and `ξ/2` for the pick. -/
theorem tsrm_bridge (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) (L : ℕ) {K : ℕ} {m γ ξ : ℝ} (hm : 0 < m)
    (hξ : 0 < ξ)
    (hstart : ∃ q, m ≤ windowVisitRate M L q ∧ 1 - γ ≤ rerootAccuracy M L Set.univ q) :
    let D : Measure (FreeMonoid α) := uniformStrings (L + Fintype.card σ + 3 + Fintype.card σ + 3)
    let k : ℕ := (L + 1) / 2
    let L₁ : ℕ := L + Fintype.card σ + 3
    let N : ℝ := L₁ + 1 - k
    let ε : ℝ := 2 * ξ * m / (5 * (Fintype.card α : ℝ) ^ (Fintype.card σ - 1))
      / (4 * ((Fintype.card α : ℝ) + Fintype.card σ + 2 + 1) * N)
    let εcov : ℝ := min ε (ξ / 2 / (8 * N))
    ∀ o : LoopRound Ω α K,
      o.Checks M O (windowLaw D k L₁) k (1025 / 1024 * εcov) (3 / 8 * εcov) (ε + ε / 96)
        (shortAt M L₁) (ε + ε / 96) →
      ∃ s ∈ o.result.hypothesis.tree.leaves, (prefixLaw D L).real
        {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)}
          ≤ γ + ξ / 2 := by
  intro D k L₁ N ε εcov o hch
  have : Nonempty σ := ⟨M.start⟩
  have hN0 : 0 < N := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast (show k ≤ L₁ by omega)
    show (0 : ℝ) < (L₁ : ℝ) + 1 - k
    linarith
  have hG : (0 : ℝ) < (Fintype.card α : ℝ) ^ (Fintype.card σ - 1) := by
    have : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
    positivity
  have hε0 : 0 ≤ ε := by simp only [ε]; positivity
  have hεc0 : 0 ≤ εcov := le_min hε0 (by positivity)
  have hpL : prefixLaw (uniformStrings (α := α) (L + Fintype.card σ + 3 + Fintype.card σ + 3)) L
      = uniformStrings L := prefixLaw_uniformStrings (by omega)
  obtain ⟨s, hs, h⟩ := dfa_of_checks_tsrm M O hL o L hch (by linarith) hm hstart
  refine ⟨s, hs, ?_⟩
  rw [show prefixLaw D L = uniformStrings L from hpL]
  exact h.trans (add_le_add le_rfl (tsrm_bnd (ξ := ξ) (ε := ε) (εcov := εcov)
    (Nat.cast_nonneg (Fintype.card α)) (Nat.cast_nonneg (Fintype.card σ)) hG hN0 hm rfl hεc0
    (min_le_left _ _)))

end Bridge

end OrthoDFA

end
