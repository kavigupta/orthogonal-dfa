import OrthoDFA.LoopDFA
import OrthoDFA.Proofs.Level5Clean
import OrthoDFA.Proofs.UniformStrings

/-!
# From a checked round to the learner's DFA

The probes are uniform over the lengths `k` to `L₁`. A probe of length `i` and its extension by a
letter that both agree, read nothing on a rarer side and nothing undecided, walk to the true
leaves of their states, so the edge between those leaves is learned and true (`edge_of_pair`).
Summed over the window, a state visited `c` of the time has every edge out of it crossed by such
pairs unless the checked probes hold `c/((|Σ|+1)N)` of the draws (`edge_of_occupied`). A state the
probes visit is read on its true side once the reads are rarely wrong or undecided
(`sideOf_eq_of_mass`). Where every covered state's edges are true and it is read on its true side,
the hypothesis from the covered start's leaf follows the re-rooted target along every run that
stays covered (`classifies_of_matches`), so it misjudges at most what re-rooting does
(`start_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

section Window

variable [Nonempty α]

/-- The average of `g` over the strings of length `ℓ`. -/
def uavg (ℓ : ℕ) (g : FreeMonoid α → ℝ) : ℝ :=
  (∑ v : List.Vector α ℓ, g (FreeMonoid.ofList v.toList)) / (Fintype.card α : ℝ) ^ ℓ

theorem integrable_uniformStrings (ℓ : ℕ) (g : FreeMonoid α → ℝ) :
    Integrable g (uniformStrings ℓ) :=
  (integrable_finsetSum_measure.2 fun _ _ => integrable_dirac (by simp)).smul_measure (by simp)

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

theorem integrable_windowStrings {lo hi : ℕ} (h : lo ≤ hi) (g : FreeMonoid α → ℝ) :
    Integrable g (windowStrings lo hi) :=
  (integrable_finsetSum_measure.2 fun _ _ => integrable_uniformStrings _ g).smul_measure
    (ENNReal.inv_ne_top.2 (Nat.cast_ne_zero.2 (by omega)))

theorem integral_windowStrings {lo hi : ℕ} (h : lo ≤ hi) (g : FreeMonoid α → ℝ) :
    ∫ x, g x ∂(windowStrings lo hi)
      = (∑ n ∈ Finset.Icc lo hi, uavg n g) / ((hi : ℝ) + 1 - lo) := by
  unfold windowStrings
  rw [integral_smul_measure, integral_finsetSum_measure fun _ _ => integrable_uniformStrings _ g]
  simp only [integral_uniformStrings, ENNReal.toReal_inv, ENNReal.toReal_natCast, smul_eq_mul]
  rw [div_eq_inv_mul, Nat.cast_sub (by omega)]
  push_cast
  rfl

theorem windowStrings_isProbability {lo hi : ℕ} (h : lo ≤ hi) :
    IsProbabilityMeasure (windowStrings (α := α) lo hi) := by
  constructor
  unfold windowStrings
  rw [Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply]
  simp only [measure_univ, Finset.sum_const, Nat.card_Icc, nsmul_eq_mul, mul_one, smul_eq_mul]
  exact ENNReal.inv_mul_cancel (Nat.cast_ne_zero.2 (by omega)) (ENNReal.natCast_ne_top _)

theorem windowStrings_real {lo hi : ℕ} (h : lo ≤ hi) (S : Set (FreeMonoid α)) :
    (windowStrings lo hi).real S
      = (∑ n ∈ Finset.Icc lo hi, (uniformStrings n).real S) / ((hi : ℝ) + 1 - lo) := by
  rw [← integral_indicator_one (MeasurableSpace.measurableSet_top), integral_windowStrings h]
  simp only [uniformStrings_real_uavg]

theorem windowStrings_length (lo hi : ℕ) :
    ∀ᵐ x ∂(windowStrings (α := α) lo hi), lo ≤ x.toList.length ∧ x.toList.length ≤ hi := by
  rw [ae_iff]
  unfold windowStrings
  rw [Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply]
  have : ∀ n ∈ Finset.Icc lo hi, uniformStrings (α := α) n
      {x | ¬ (lo ≤ x.toList.length ∧ x.toList.length ≤ hi)} = 0 := by
    intro n hn
    refine measure_mono_null (fun x hx => ?_) (uniformStrings_length n)
    simp only [Finset.mem_Icc] at hn
    simp only [Set.mem_setOf_eq] at hx ⊢
    omega
  rw [Finset.sum_eq_zero this, smul_zero]

/-- A prefix of a uniform string is uniform. -/
theorem uniformStrings_prefix {j n : ℕ} (h : j ≤ n) (S : Set (FreeMonoid α)) :
    (uniformStrings n).real {x | prefixOf x j ∈ S} = (uniformStrings j).real S := by
  rw [uniformStrings_real_uavg, uniformStrings_real_uavg, ← uavg_prefix h]
  rfl

theorem prefixMax_windowStrings_le {k lo hi : ℕ} (hk : k ≤ lo) (h : lo ≤ hi) :
    prefixMax (windowStrings (α := α) lo hi) k ≤ 1 / (Fintype.card α : ℝ) ^ k := by
  unfold prefixMax
  refine ciSup_le fun u => ?_
  have hlh : (lo : ℝ) ≤ hi := by exact_mod_cast h
  rw [windowStrings_real h, div_le_iff₀ (by linarith)]
  have hn : ∀ n ∈ Finset.Icc lo hi, (uniformStrings n).real {x : FreeMonoid α | prefixOf x k = u}
      ≤ 1 / (Fintype.card α : ℝ) ^ k := by
    intro n hn
    simp only [Finset.mem_Icc] at hn
    have := uniformStrings_prefix (α := α) (show k ≤ n by omega) {u}
    simp only [Set.mem_singleton_iff] at this
    rw [this]
    exact uniformStrings_singleton_le k u
  refine (Finset.sum_le_sum hn).trans (le_of_eq ?_)
  rw [Finset.sum_const, Nat.card_Icc, nsmul_eq_mul, Nat.cast_sub (by omega)]
  push_cast
  ring

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

/-- A covered state is visited `c` of the time, summed over the window. -/
theorem occupancy_of_covered {σ : Type*} (M : _root_.DFA α σ) {L : ℕ} {c : ℝ} {q : σ}
    (hq : q ∈ coveredStates M L c) :
    c ≤ ∑ i ∈ Finset.Icc ((L + 1) / 2) L, (uniformStrings i).real {y | M.eval y.toList = q} := by
  refine hq.trans ?_
  have hsub : {x : FreeMonoid α | ∃ j, (L + 1) / 2 ≤ j ∧ j ≤ L ∧ M.eval (prefixOf x j).toList = q}
      ⊆ ⋃ j ∈ Finset.Icc ((L + 1) / 2) L, {x | prefixOf x j ∈ {y | M.eval y.toList = q}} := by
    rintro x ⟨j, h1, h2, h3⟩
    exact Set.mem_biUnion (Finset.mem_Icc.2 ⟨h1, h2⟩) h3
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans
    ((measureReal_biUnion_finset_le _ _).trans (le_of_eq (Finset.sum_congr rfl fun j hj => ?_)))
  exact uniformStrings_prefix (Finset.mem_Icc.1 hj).2 _

/-- What a probe from the window reads past its start, `n` letters on, at the states' undecided
rates: at most the window's own undecided rate, times how many such reads there are and how much
more often one length is drawn than the window's average. -/
theorem undecidedMass_window_le {σ : Type*} (M : _root_.DFA α σ) (dist : σ → ARU → ℝ)
    (h0 : ∀ r v, 0 ≤ dist r v) {k n J L₁ : ℕ} (hkJ : k ≤ J)
    (hJ : J + n ≤ L₁) :
    undecidedMass M dist (windowStrings k L₁) k n J
      ≤ ((J : ℝ) + 1 - k) * (n + 1) * (Fintype.card α : ℝ) ^ n * ((L₁ : ℝ) + 1 - k)
        * ∫ p, dist (M.eval p.toList) .undecided ∂(windowStrings k L₁) := by
  set f : FreeMonoid α → ℝ := fun p => dist (M.eval p.toList) .undecided with hf
  set W := ∫ p, f p ∂(windowStrings k L₁)
  set N : ℝ := (L₁ : ℝ) + 1 - k
  set A : ℝ := (Fintype.card α : ℝ) with hA
  have hkL : k ≤ L₁ := by omega
  have hN : 0 < N := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL
    simp only [N]; linarith
  have hA1 : 1 ≤ A := by simp only [A]; exact_mod_cast Fintype.card_pos
  have hf0 : ∀ x, 0 ≤ f x := fun _ => h0 _ _
  have hW : W = (∑ ℓ ∈ Finset.Icc k L₁, uavg ℓ f) / N := integral_windowStrings hkL f
  have hW0 : 0 ≤ W := by
    rw [hW]; exact div_nonneg (Finset.sum_nonneg fun _ _ => uavg_nonneg _ hf0) hN.le
  have hone : ∀ ℓ ∈ Finset.Icc k L₁, uavg ℓ f ≤ N * W := by
    intro ℓ hℓ
    rw [hW, mul_div_cancel₀ _ hN.ne']
    exact Finset.single_le_sum (fun i _ => uavg_nonneg i hf0) hℓ
  have hjm : ∀ j, k ≤ j → j ≤ J → ∑ m ∈ shortWords n, uavg j (fun y => f (y * m))
      ≤ (n + 1) * A ^ n * (N * W) := by
    intro j hkj hjJ
    unfold shortWords
    refine (Finset.sum_image_le_of_nonneg fun _ _ => uavg_nonneg _ fun _ => hf0 _).trans ?_
    rw [Fintype.sum_sigma]
    calc ∑ i : Fin (n + 1), ∑ g : Fin i → α,
          uavg j (fun y => f (y * FreeMonoid.ofList (List.ofFn g)))
        = ∑ i : Fin (n + 1), A ^ (i : ℕ) * uavg (j + i) f := by
          refine Fintype.sum_congr _ _ fun i => ?_
          rw [← uavg_append, ← (Equiv.vectorEquivFin α i).symm.sum_comp]
          refine Fintype.sum_congr _ _ fun g => ?_
          simp [Equiv.vectorEquivFin]
      _ ≤ ∑ _i : Fin (n + 1), A ^ n * (N * W) := by
          refine Finset.sum_le_sum fun i _ => ?_
          refine mul_le_mul (pow_le_pow_right₀ hA1 (by omega)) (hone _ ?_)
            (uavg_nonneg _ hf0) (by positivity)
          simp only [Finset.mem_Icc]; omega
      _ = (n + 1) * A ^ n * (N * W) := by simp; ring
  have hl : ∀ ℓ ∈ Finset.Icc k L₁, uavg ℓ (fun x => ∑ j ∈ Finset.Icc k (min x.toList.length J),
      ∑ m ∈ shortWords n, f (prefixOf x j * m))
      ≤ ((J : ℝ) + 1 - k) * ((n + 1) * A ^ n * (N * W)) := by
    intro ℓ hℓ
    have e : uavg ℓ (fun x => ∑ j ∈ Finset.Icc k (min x.toList.length J),
        ∑ m ∈ shortWords n, f (prefixOf x j * m))
        = ∑ j ∈ Finset.Icc k (min ℓ J), ∑ m ∈ shortWords n, uavg j (fun y => f (y * m)) := by
      have e1 : uavg ℓ (fun x => ∑ j ∈ Finset.Icc k (min x.toList.length J),
          ∑ m ∈ shortWords n, f (prefixOf x j * m))
          = uavg ℓ (fun x => ∑ j ∈ Finset.Icc k (min ℓ J),
            ∑ m ∈ shortWords n, f (prefixOf x j * m)) := by
        unfold uavg
        congr 1
        refine Fintype.sum_congr _ _ fun v => ?_
        simp
      rw [e1, uavg_sum]
      refine Finset.sum_congr rfl fun j hj => ?_
      rw [uavg_sum]
      refine Finset.sum_congr rfl fun m _ => ?_
      exact uavg_prefix (g := fun y => f (y * m)) (by simp only [Finset.mem_Icc] at hj; omega)
    rw [e]
    refine (Finset.sum_le_sum fun j hj => hjm j (Finset.mem_Icc.1 hj).1
      (by have := (Finset.mem_Icc.1 hj).2; omega)).trans ?_
    rw [Finset.sum_const, nsmul_eq_mul]
    refine mul_le_mul_of_nonneg_right ?_ (by positivity)
    have hc : (Finset.Icc k (min ℓ J)).card ≤ J + 1 - k := by
      rw [Nat.card_Icc]; omega
    have : ((J + 1 - k : ℕ) : ℝ) = (J : ℝ) + 1 - k := by
      rw [Nat.cast_sub (by omega)]; push_cast; ring
    rw [← this]
    exact_mod_cast hc
  unfold undecidedMass
  rw [integral_windowStrings hkL, div_le_iff₀ hN]
  refine (Finset.sum_le_sum hl).trans (le_of_eq ?_)
  rw [Finset.sum_const, Nat.card_Icc, nsmul_eq_mul, Nat.cast_sub (by omega)]
  simp only [N]
  push_cast
  ring

end Window

section Edges

variable {σ : Type*} (G : ReadModel α σ) (rd : FreeMonoid α → ARU)

omit [Fintype α] [DecidableEq α] in
theorem follow_snoc {edges : EdgeTable α} :
    ∀ (l : List α) (c : α) (p : List Bool) (ps ps' : List (List Bool)),
      follow edges p l = .inl ps → follow edges p (l ++ [c]) = .inl ps' →
        ∃ w, edges (ps.getLast?.getD []) c = some (ps'.getLast?.getD [], w)
  | [], c, p, ps, ps', h, h' => by
    simp only [follow, Sum.inl.injEq] at h
    subst h
    simp only [List.nil_append, follow] at h'
    rcases he : edges p c with _ | ⟨q, y⟩ <;> rw [he] at h'
    · cases h'
    · simp only [Sum.inl.injEq] at h'
      subst h'
      exact ⟨y, by simp [he]⟩
  | d :: l, c, p, ps, ps', h, h' => by
    simp only [follow, List.cons_append] at h h'
    rcases he : edges p d with _ | ⟨q, y⟩ <;> rw [he] at h h'
    · cases h
    simp only [] at h h'
    rcases hf : follow edges q l with ps₁ | r <;> rw [hf] at h
    · rcases hf' : follow edges q (l ++ [c]) with ps₁' | r' <;> rw [hf'] at h'
      · simp only [Sum.inl.injEq] at h h'
        subst h h'
        obtain ⟨w, hw⟩ := follow_snoc l c q ps₁ ps₁' hf hf'
        have h1 : ps₁ ≠ [] := by
          intro h0; subst h0; obtain ⟨hl, -⟩ := follow_inl _ _ _ hf; simp at hl
        have h2 : ps₁' ≠ [] := by
          intro h0; subst h0; obtain ⟨hl, -⟩ := follow_inl _ _ _ hf'; simp at hl
        refine ⟨w, ?_⟩
        rw [List.getLast?_cons, List.getLast?_cons]
        obtain ⟨a₁, ha₁⟩ := List.getLast?_isSome.2 h1 |> Option.isSome_iff_exists.1
        obtain ⟨a₂, ha₂⟩ := List.getLast?_isSome.2 h2 |> Option.isSome_iff_exists.1
        rw [ha₁, ha₂] at hw ⊢
        simpa using hw
      · cases h'
    · cases h

theorem cut_none_iff (v : ARU) : v.cut = none ↔ v = .undecided := by
  cases v <;> simp [ARU.cut]

/-- A probe neither searching nor reading undecided walks, learned all the way, from its start's
sift to its own. -/
theorem agree_of_unchecked (H : HypothesisDFA α) {k n J : ℕ}
    (hmid : ∀ m ∈ H.tree.midfixes, m ∈ shortWords n) {x : FreeMonoid α}
    (hk : k ≤ x.toList.length) (hJ : x.toList.length ≤ J) (hx : x ∉ searchingProbes rd k H)
    (hu : x ∉ undecidedProbes rd k n J) :
    ∃ p ps e, H.tree.siftCut (fun z => (rd z).cut) (prefixOf x k) = .inl p
      ∧ follow (Function.curry H.edges) p (x.toList.drop k) = .inl ps
      ∧ H.tree.siftCut (fun z => (rd z).cut) x = .inl e ∧ ps.getLast? = some e := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
  have hdec : ∀ j, k ≤ j → j ≤ x.toList.length →
      ∃ p, H.tree.siftCut cut (prefixOf x j) = .inl p := by
    intro j hkj hjx
    rcases hs : H.tree.siftCut cut (prefixOf x j) with p | b
    · exact ⟨p, rfl⟩
    · obtain ⟨m, hm, -, hc⟩ := siftCut_inr H.tree _ b hs
      exact absurd ⟨hJ, j, hkj, hjx, m, hmid m hm, (cut_none_iff _).1 hc⟩ hu
  have hsrch : ∀ o, probeBy cut H.tree (Function.curry H.edges) k x = o → o.IsSearch ∨
      (∃ u, o = .member u) → False := by
    intro o ho h
    apply hx
    show match probeBy cut H.tree (Function.curry H.edges) k x with
      | .undecidedMid _ | .edge _ _ | .member _ => True
      | _ => False
    rw [ho]
    rcases h with h | ⟨u, rfl⟩
    · cases o <;> simp_all [WalkOutcome.IsSearch]
    · trivial
  have hpe := probeBy_eq cut H.tree (Function.curry H.edges) k x
  rw [if_neg (by omega)] at hpe
  obtain ⟨p, hp⟩ := hdec k le_rfl hk
  rcases hf : follow (Function.curry H.edges) p (x.toList.drop k) with ps | ⟨s, c, i⟩
  · obtain ⟨e, he⟩ := hdec x.toList.length hk le_rfl
    rw [prefixOf_length] at he
    refine ⟨p, ps, e, hp, hf, he, ?_⟩
    by_contra hne
    have hw : walkCheckBy cut H.tree (Function.curry H.edges) k x
        = .inr (ps, x.toList.length) := by
      simp only [walkCheckBy, kWalkBy, hp, hf, he]
      rw [if_neg (fun h => hne h.symm)]
    rw [hw] at hpe
    exact hsrch _ hpe (.inl (bracketAt_isSearch _ _ _ _ _))
  · obtain ⟨hi, -, -, -⟩ := follow_inr _ _ _ _ _ hf
    simp only [List.length_drop] at hi
    obtain ⟨p1, hp1⟩ := hdec (k + i + 1) (by omega) (by omega)
    obtain ⟨p2, hp2⟩ := hdec (k + i) (by omega) (by omega)
    by_cases hps : p2 = s
    · have hw : walkCheckBy cut H.tree (Function.curry H.edges) k x
          = .inl (.member (prefixOf x (k + i))) := by
        simp only [walkCheckBy, kWalkBy, hp, hf, hp1, hp2, if_pos hps]
      rw [hw] at hpe
      exact absurd (hsrch _ hpe (.inr ⟨_, rfl⟩)) id
    · have hw : walkCheckBy cut H.tree (Function.curry H.edges) k x
          = .inr (walkToBy cut H.tree (Function.curry H.edges) k x (k + i), k + i) := by
        simp only [walkCheckBy, kWalkBy, hp, hf, hp1, hp2, if_neg hps]
      rw [hw] at hpe
      exact absurd (hsrch _ hpe (.inl (bracketAt_isSearch _ _ _ _ _))) id

/-- A probe and its extension by a letter, both at most `J` long and neither checked, cross a
learned edge between their states' true leaves. -/
theorem edge_of_pair (H : HypothesisDFA α) {k n J : ℕ}
    (hmid : ∀ m ∈ H.tree.midfixes, m ∈ shortWords n) {y : FreeMonoid α} (a : α)
    (hk : k ≤ y.toList.length) (hJ : y.toList.length + 1 ≤ J)
    (hy : y ∉ searchingProbes rd k H ∪ rarerProbes G.M G.readDist rd k n
      ∪ undecidedProbes rd k n J)
    (hya : y * FreeMonoid.of a ∉ searchingProbes rd k H ∪ rarerProbes G.M G.readDist rd k n
      ∪ undecidedProbes rd k n J) :
    ∃ w, H.edges (G.leafOf H.tree (G.M.eval y.toList), a)
      = some (G.leafOf H.tree (G.M.eval (y * FreeMonoid.of a).toList), w) := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
  have hlen : (y * FreeMonoid.of a).toList.length = y.toList.length + 1 := by simp
  simp only [Set.mem_union, not_or] at hy hya
  obtain ⟨p, ps, e, hp, hf, he, hl⟩ := agree_of_unchecked rd H hmid hk (by omega) hy.1.1 hy.2
  obtain ⟨p', ps', e', hp', hf', he', hl'⟩ :=
    agree_of_unchecked rd H hmid (by omega) (by omega) hya.1.1 hya.2
  have hpre : prefixOf (y * FreeMonoid.of a) k = prefixOf y k := by
    simp only [prefixOf, FreeMonoid.toList_mul, FreeMonoid.toList_of]
    rw [List.take_append_of_le_length hk]
  rw [hpre, hp] at hp'
  cases hp'
  have hdrop : (y * FreeMonoid.of a).toList.drop k = y.toList.drop k ++ [a] := by
    simp only [FreeMonoid.toList_mul, FreeMonoid.toList_of]
    rw [List.drop_append_of_le_length hk]
  rw [hdrop] at hf'
  obtain ⟨w, hw⟩ := follow_snoc _ a p ps ps' hf hf'
  have hclean : ∀ z : FreeMonoid α, z ∉ rarerProbes G.M G.readDist rd k n →
      k ≤ z.toList.length → ∀ m ∈ H.tree.midfixes, ¬ G.WrongRead rd (z * m) := by
    intro z hz hkz m hm hwr
    refine hz ⟨z.toList.length, hkz, le_rfl, m, hmid m hm, ?_⟩
    rw [prefixOf_length]
    exact hwr
  have h1 := leafOf_of_sift G rd (hclean y hy.1.2 hk) he
  have h2 := leafOf_of_sift G rd (hclean _ hya.1.2 (by omega)) he'
  rw [hl, hl'] at hw
  rw [h1, h2]
  exact ⟨w, hw⟩

variable [Nonempty α]

/-- A state visited `c` of the time over the window's lengths below `J` has every edge out of its
true leaf learned and true, unless the checked probes hold `c/((|Σ|+1)N)` of the draws. -/
theorem edge_of_occupied (H : HypothesisDFA α) {k n J L₁ : ℕ}
    (hmid : ∀ m ∈ H.tree.midfixes, m ∈ shortWords n) (hkJ : k < J) (hJ : J ≤ L₁) (q : σ) (a : α)
    {c β : ℝ}
    (hbad : (windowStrings k L₁).real (searchingProbes rd k H ∪ rarerProbes G.M G.readDist rd k n
      ∪ undecidedProbes rd k n J) ≤ β)
    (hocc : c ≤ ∑ i ∈ Finset.Ico k J, (uniformStrings i).real {y | G.M.eval y.toList = q})
    (hβ : ((Fintype.card α : ℝ) + 1) * ((L₁ : ℝ) + 1 - k) * β < c) :
    ∃ w, H.edges (G.leafOf H.tree q, a) = some (G.leafOf H.tree (G.M.step q a), w) := by
  set Bad := searchingProbes rd k H ∪ rarerProbes G.M G.readDist rd k n
    ∪ undecidedProbes rd k n J
  set A : ℝ := (Fintype.card α : ℝ) with hA
  by_contra hno
  have hkL : k ≤ L₁ := by omega
  have hU0 : ∀ i, 0 ≤ (uniformStrings (α := α) i).real Bad := fun _ => measureReal_nonneg
  have hi : ∀ i ∈ Finset.Ico k J, (uniformStrings i).real {y | G.M.eval y.toList = q}
      ≤ (uniformStrings i).real Bad + A * (uniformStrings (i + 1)).real Bad := by
    intro i hi
    simp only [Finset.mem_Ico] at hi
    have hsub : {y : FreeMonoid α | G.M.eval y.toList = q}
        ⊆ (Bad ∪ {y | y * FreeMonoid.of a ∈ Bad}) ∪ {y | y.toList.length ≠ i} := by
      intro y hy
      by_contra hc
      simp only [Set.mem_union, Set.mem_setOf_eq, not_or, not_not] at hc hy
      apply hno
      obtain ⟨w, hw⟩ := edge_of_pair G rd H hmid a (by omega) (by omega) hc.1.1 hc.1.2
      rw [FreeMonoid.toList_mul, FreeMonoid.toList_of, DFA.eval_append_singleton, hy] at hw
      exact ⟨w, hw⟩
    have hnull : (uniformStrings (α := α) i).real {y | y.toList.length ≠ i} = 0 := by
      rw [measureReal_def, uniformStrings_length, ENNReal.toReal_zero]
    refine (measureReal_mono hsub (measure_ne_top _ _)).trans ?_
    refine (measureReal_union_le _ _).trans ?_
    rw [hnull, add_zero]
    refine (measureReal_union_le _ _).trans (add_le_add le_rfl ?_)
    exact uniformStrings_snoc_le i a Bad
  have hsum := Finset.sum_le_sum hi
  rw [Finset.sum_add_distrib, ← Finset.mul_sum] at hsum
  have h1 : ∑ i ∈ Finset.Ico k J, (uniformStrings (α := α) i).real Bad
      ≤ ∑ i ∈ Finset.Icc k L₁, (uniformStrings (α := α) i).real Bad :=
    Finset.sum_le_sum_of_subset_of_nonneg (fun i hi => by
      simp only [Finset.mem_Ico, Finset.mem_Icc] at hi ⊢; omega) fun i _ _ => hU0 i
  have h2 : ∑ i ∈ Finset.Ico k J, (uniformStrings (α := α) (i + 1)).real Bad
      ≤ ∑ i ∈ Finset.Icc k L₁, (uniformStrings (α := α) i).real Bad := by
    have e := Finset.sum_image (f := fun i => (uniformStrings (α := α) i).real Bad)
      (s := Finset.Ico k J) (g := (· + 1)) (fun x _ y _ h => by simpa using h)
    rw [← e]
    exact Finset.sum_le_sum_of_subset_of_nonneg (fun i hi => by
      simp only [Finset.mem_image, Finset.mem_Ico, Finset.mem_Icc] at hi ⊢; omega)
      fun i _ _ => hU0 i
  have hw := windowStrings_real (α := α) hkL Bad
  have hN : 0 < (L₁ : ℝ) + 1 - k := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL
    linarith
  rw [eq_div_iff hN.ne'] at hw
  have hA0 : 0 ≤ A := by positivity
  have h3 := mul_le_mul_of_nonneg_left hbad hN.le
  have h4 := mul_le_mul_of_nonneg_left h2 hA0
  nlinarith

end Edges

section Side

open scoped Classical

variable {σ : Type*}

/-- A state the probes visit `b` of the time is read more often on its true side than on the
other, once the reads are wrong at most `εr` and undecided at most `εu` on average and
`2εr + εu < b`. -/
theorem sideOf_eq_of_mass (M : _root_.DFA α σ) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (dist : σ → ARU → ℝ) (h0 : ∀ r v, 0 ≤ dist r v)
    (hsum : ∀ r, dist r .accept + dist r .reject + dist r .undecided = 1) (q : σ) {b εr εu : ℝ}
    (hmis : ∫ p, (if M.eval p.toList ∈ M.accept then dist (M.eval p.toList) .reject
      else dist (M.eval p.toList) .accept) ∂D ≤ εr)
    (hund : ∫ p, dist (M.eval p.toList) .undecided ∂D ≤ εu)
    (hocc : b ≤ D.real {p | M.eval p.toList = q}) (hb : 2 * εr + εu < b) :
    sideOf (dist q) = decide (q ∈ M.accept) := by
  have hle1 : ∀ r v, dist r v ≤ 1 := by
    intro r v
    have := hsum r
    have := h0 r .accept; have := h0 r .reject; have := h0 r .undecided
    cases v <;> linarith
  have hint : ∀ g : FreeMonoid α → ℝ, (∀ p, 0 ≤ g p) → (∀ p, g p ≤ 1) → Integrable g D :=
    fun g g0 g1 => Integrable.of_bound (measurable_of_countable g).aestronglyMeasurable 1
      (ae_of_all _ fun p => by rw [Real.norm_eq_abs, abs_of_nonneg (g0 p)]; exact g1 p)
  have hlow : ∀ g : FreeMonoid α → ℝ, (∀ p, 0 ≤ g p) → (∀ p, g p ≤ 1) → ∀ c : ℝ,
      (∀ p, M.eval p.toList = q → c ≤ g p) → 0 ≤ c →
        D.real {p | M.eval p.toList = q} * c ≤ ∫ p, g p ∂D := by
    intro g g0 g1 c hc c0
    have hS : MeasurableSet {p : FreeMonoid α | M.eval p.toList = q} :=
      MeasurableSpace.measurableSet_top
    rw [← smul_eq_mul, ← integral_indicator_const c hS]
    refine integral_mono (hint _ (fun p => Set.indicator_nonneg (fun _ _ => c0) p)
      (fun p => ?_)) (hint g g0 g1) fun p => ?_
    · by_cases hp : p ∈ {p : FreeMonoid α | M.eval p.toList = q}
      · rw [Set.indicator_of_mem hp]; exact (hc p hp).trans (g1 p)
      · rw [Set.indicator_of_notMem hp]; norm_num
    · by_cases hp : p ∈ {p : FreeMonoid α | M.eval p.toList = q}
      · rw [Set.indicator_of_mem hp]; exact hc p hp
      · rw [Set.indicator_of_notMem hp]; exact g0 p
  have hw0 : ∀ p : FreeMonoid α, 0 ≤ (if M.eval p.toList ∈ M.accept then dist (M.eval p.toList) .reject
      else dist (M.eval p.toList) .accept) := fun p => by split_ifs <;> exact h0 _ _
  have hw1 : ∀ p : FreeMonoid α, (if M.eval p.toList ∈ M.accept then dist (M.eval p.toList) .reject
      else dist (M.eval p.toList) .accept) ≤ 1 := fun p => by split_ifs <;> exact hle1 _ _
  have hD0 : 0 ≤ D.real {p : FreeMonoid α | M.eval p.toList = q} := measureReal_nonneg
  have hu := hlow (fun p => dist (M.eval p.toList) .undecided) (fun p => h0 _ _)
    (fun p => hle1 _ _) (dist q .undecided) (fun p hp => by rw [hp]) (h0 _ _)
  have hb0 : 0 < b := by
    have := (integral_nonneg hw0).trans hmis
    have := (integral_nonneg (fun p : FreeMonoid α => h0 (M.eval p.toList) .undecided)).trans hund
    linarith
  have hsq := hsum q
  have hacc := h0 q .accept
  have hrej := h0 q .reject
  have hund0 := h0 q .undecided
  have hbu := mul_le_mul_of_nonneg_right hocc hund0
  by_cases hq : q ∈ M.accept
  · have hm := hlow _ hw0 hw1 (dist q .reject) (fun p hp => by rw [if_pos (hp ▸ hq), hp]) hrej
    have hbr := mul_le_mul_of_nonneg_right hocc hrej
    simp only [sideOf, hq, decide_true, decide_eq_true_eq]
    by_contra hlt
    push Not at hlt
    nlinarith
  · have hm := hlow _ hw0 hw1 (dist q .accept) (fun p hp => by rw [if_neg (hp ▸ hq), hp]) hacc
    have hba := mul_le_mul_of_nonneg_right hocc hacc
    simp only [sideOf, hq, decide_false, decide_eq_false_iff_not]
    intro hlt
    nlinarith

end Side

section Start

open scoped Classical

variable {σ : Type*} (G : ReadModel α σ)

/-- Where every state of `S` has its edges true and is read on its true side, the hypothesis from
`q₀`'s true leaf classifies every string whose run from `q₀` stays in `S` as the re-rooted target
does. -/
theorem classifies_of_matches (H : HypothesisDFA α) (hroot : ∃ r a, H.tree = .node 1 r a)
    (S : Set σ)
    (hE : ∀ q ∈ S, ∀ a, ∃ w, H.edges (G.leafOf H.tree q, a)
      = some (G.leafOf H.tree (G.M.step q a), w))
    (hside : ∀ q ∈ S, G.side q = decide (q ∈ G.M.accept)) (q₀ : σ) (x : FreeMonoid α)
    (hstay : ∀ i ≤ x.toList.length, G.M.evalFrom q₀ (prefixOf x i).toList ∈ S)
    (hagree : G.M.evalFrom q₀ x.toList ∈ G.M.accept ↔ G.M.eval x.toList ∈ G.M.accept) :
    H.Classifies (G.leafOf H.tree q₀) x (G.M.eval x.toList ∈ G.M.accept) := by
  have track : ∀ (l : List α) (q : σ), (∀ i ≤ l.length, G.M.evalFrom q (l.take i) ∈ S) →
      walkEnd H.edges (G.leafOf H.tree q) l = some (G.leafOf H.tree (G.M.evalFrom q l)) := by
    intro l
    induction l with
    | nil => intro q _; simp [walkEnd]
    | cons c l ih =>
      intro q hs
      have hq : q ∈ S := by simpa using hs 0 (by simp)
      obtain ⟨w, hw⟩ := hE q hq c
      simp only [walkEnd, hw, Option.bind_some]
      rw [ih (G.M.step q c) fun i hi => by simpa using hs (i + 1) (by simp; omega)]
      rfl
  have hs : ∀ i ≤ x.toList.length, G.M.evalFrom q₀ (x.toList.take i) ∈ S := fun i hi => by
    simpa [prefixOf] using hstay i hi
  refine ⟨_, track x.toList q₀ hs, ?_⟩
  have hend := hs x.toList.length le_rfl
  rw [List.take_length] at hend
  obtain ⟨r, a, hT⟩ := hroot
  have hhead : (G.leafOf H.tree (G.M.evalFrom q₀ x.toList)).head?
      = some (G.side (G.M.evalFrom q₀ x.toList)) := by
    have hat : G.at' (G.M.evalFrom q₀ x.toList) 1 = G.M.evalFrom q₀ x.toList := by
      simp [ReadModel.at']
    rw [hT]
    simp only [ReadModel.leafOf, hat]
    cases G.side (G.M.evalFrom q₀ x.toList) <;> rfl
  rw [hhead, ← hagree, hside _ hend]
  simp

variable [Nonempty α]

/-- The hypothesis from `q₀`'s true leaf misjudges at most what re-rooting at `q₀` does. -/
theorem start_le (H : HypothesisDFA α) (hroot : ∃ r a, H.tree = .node 1 r a) (S : Set σ)
    (hE : ∀ q ∈ S, ∀ a, ∃ w, H.edges (G.leafOf H.tree q, a)
      = some (G.leafOf H.tree (G.M.step q a), w))
    (hside : ∀ q ∈ S, G.side q = decide (q ∈ G.M.accept)) (L : ℕ) (q₀ : σ) {η : ℝ}
    (hacc : 1 - η ≤ rerootAccuracy G.M L S q₀) :
    (uniformStrings L).real {x | ¬ H.Classifies (G.leafOf H.tree q₀) x
      (G.M.eval x.toList ∈ G.M.accept)} ≤ η := by
  set good := {x : FreeMonoid α | (∀ i ≤ L, G.M.evalFrom q₀ (prefixOf x i).toList ∈ S)
    ∧ (G.M.evalFrom q₀ x.toList ∈ G.M.accept ↔ G.M.eval x.toList ∈ G.M.accept)}
  have hsub : {x | ¬ H.Classifies (G.leafOf H.tree q₀) x (G.M.eval x.toList ∈ G.M.accept)}
      ⊆ goodᶜ ∪ {x | x.toList.length ≠ L} := by
    intro x hx
    by_contra hc
    simp only [Set.mem_union, Set.mem_compl_iff, Set.mem_setOf_eq, not_or, not_not] at hc hx
    obtain ⟨⟨hst, hag⟩, hl⟩ := hc
    exact hx (classifies_of_matches G H hroot S hE hside q₀ x (fun i hi => hst i (hl ▸ hi)) hag)
  have hnull : (uniformStrings (α := α) L).real {x | x.toList.length ≠ L} = 0 := by
    rw [measureReal_def, uniformStrings_length, ENNReal.toReal_zero]
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans ((measureReal_union_le _ _).trans ?_)
  rw [hnull, add_zero, measureReal_compl MeasurableSpace.measurableSet_top, probReal_univ]
  have : 1 - η ≤ (uniformStrings L).real good := hacc
  linarith

end Start

end OrthoDFA

end
