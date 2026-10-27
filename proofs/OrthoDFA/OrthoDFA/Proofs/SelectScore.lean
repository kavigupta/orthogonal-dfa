import OrthoDFA.LowLevelClaims.LoopDFA
import OrthoDFA.Proofs.Hoeffding
import OrthoDFA.Proofs.UniformStrings
import OrthoDFA.Proofs.DFABridge
import OrthoDFA.Proofs.LoopMeasurable

/-!
# Scoring leaves on held-out strings

A leaf's score sums `agreeAt` over the held-out strings. At distinct strings the family's reads are
independent, so the score is within `n t/2` of its mean given the strings, and that mean within
`n t/2` of `n · agreeMean` (`scoreBad_le`). Where every leaf's score is within `n t` of its mean,
the leaf of highest score misjudges at most `2t` more than the best leaf, and the mean is within
the family's misread and undecided shares of the leaf's accuracy (`pickLeaf_le`). Uniform
held-out strings repeat, or start with the first `k` letters of a given set, rarely
(`batch_collide_le`).
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ]

section Score

variable (O : Oracle μ (FreeMonoid α)) (H : HypothesisDFA α) (F : Finset (FreeMonoid α))
  (kl kh : ℕ)

open scoped Classical in
/-- `1` where the read of `x` is decided and the run from `s` classifies `x` as it decides. -/
def agreeAt (s : List Bool) (x : FreeMonoid α) (ω : Ω) : ℝ :=
  if familyRead O.mq F kl kh x ω ≠ .undecided
    ∧ H.Classifies s x (familyRead O.mq F kl kh x ω = .accept) then 1 else 0

/-- `agreeAt`'s mean over the noise. -/
def agreeProb (s : List Bool) (x : FreeMonoid α) : ℝ := ∫ ω, agreeAt O H F kl kh s x ω ∂μ

/-- `agreeProb`'s mean over `T`'s strings. -/
def agreeMean (T : Measure (FreeMonoid α)) (s : List Bool) : ℝ :=
  ∫ x, agreeProb O H F kl kh s x ∂T

/-- Distinct held-out strings at which some leaf's score strays `n t` from its mean. -/
def scoreBad (T : Measure (FreeMonoid α)) (n : ℕ) (t : ℝ) : Set (Ω × (Fin n → FreeMonoid α)) :=
  {p | Function.Injective p.2 ∧ ∃ s ∈ H.tree.leaves,
    n * (agreeMean O H F kl kh T s + t) ≤ ∑ i, agreeAt O H F kl kh s (p.2 i) p.1
      ∨ ∑ i, agreeAt O H F kl kh s (p.2 i) p.1 ≤ n * (agreeMean O H F kl kh T s - t)}

omit [IsProbabilityMeasure μ] in
theorem leafScore_eq (ω : Ω) {n : ℕ} (xs : Fin n → FreeMonoid α) (s : List Bool) :
    (leafScore H (fun z => familyRead O.mq F kl kh z ω) xs s : ℝ)
      = ∑ i, agreeAt O H F kl kh s (xs i) ω := by
  classical
  unfold leafScore agreeAt
  rw [Finset.sum_boole]

theorem measurable_agreeAt (s : List Bool) (x : FreeMonoid α) :
    Measurable (agreeAt O H F kl kh s x) := by
  classical
  have e : agreeAt O H F kl kh s x = (fun j : ℕ => if readOf kl kh j ≠ .undecided
      ∧ H.Classifies s x (readOf kl kh j = .accept) then (1 : ℝ) else 0)
        ∘ fun ω => voteCount O.mq F x ω := by
    funext ω
    rfl
  rw [e]
  exact measurable_from_nat.comp (measurable_voteCount O F x)

omit [IsProbabilityMeasure μ] in
theorem agreeAt_mem (s : List Bool) (x : FreeMonoid α) (ω : Ω) :
    agreeAt O H F kl kh s x ω ∈ Set.Icc (0 : ℝ) 1 := by
  unfold agreeAt
  split_ifs <;> simp

theorem measurableSet_scoreBad (T : Measure (FreeMonoid α)) (n : ℕ) (t : ℝ) :
    MeasurableSet (scoreBad O H F kl kh T n t) := by
  classical
  refine measurableSet_prod_countable_right _ fun b => ?_
  have hm : ∀ s, Measurable fun ω => ∑ i, agreeAt O H F kl kh s (b i) ω := fun s =>
    Finset.measurable_sum _ fun i _ => measurable_agreeAt O H F kl kh s (b i)
  have hset : {ω | (ω, b) ∈ scoreBad O H F kl kh T n t}
      = {_ω | Function.Injective b} ∩ ⋃ s ∈ H.tree.leaves.toFinset,
        ({ω | n * (agreeMean O H F kl kh T s + t) ≤ ∑ i, agreeAt O H F kl kh s (b i) ω}
          ∪ {ω | ∑ i, agreeAt O H F kl kh s (b i) ω ≤ n * (agreeMean O H F kl kh T s - t)}) := by
    ext ω
    simp [scoreBad]
  rw [hset]
  exact (MeasurableSet.const _).inter (Finset.measurableSet_biUnion _ fun s _ =>
    (measurableSet_le measurable_const (hm s)).union (measurableSet_le (hm s) measurable_const))

theorem agreeProb_mem (s : List Bool) (x : FreeMonoid α) :
    agreeProb O H F kl kh s x ∈ Set.Icc (0 : ℝ) 1 := by
  have hb : ∀ ω, agreeAt O H F kl kh s x ω ∈ Set.Icc (0 : ℝ) 1 := agreeAt_mem O H F kl kh s x
  have hi : Integrable (agreeAt O H F kl kh s x) μ :=
    Integrable.of_bound (measurable_agreeAt O H F kl kh s x).aestronglyMeasurable 1
      (Filter.Eventually.of_forall fun ω => by
        rw [Real.norm_eq_abs, abs_le]; exact ⟨by linarith [(hb ω).1], (hb ω).2⟩)
  refine ⟨integral_nonneg fun ω => (hb ω).1, ?_⟩
  calc agreeProb O H F kl kh s x ≤ ∫ _ω, (1 : ℝ) ∂μ :=
        integral_mono hi (integrable_const 1) fun ω => (hb ω).2
    _ = 1 := by simp

theorem scoreBad_le (hF : SuffixFree F) (T : Measure (FreeMonoid α)) [IsProbabilityMeasure T]
    (n : ℕ) (t : ℝ) (ht : 0 ≤ t) :
    (μ.prod (Measure.pi fun _ : Fin n => T)) (scoreBad O H F kl kh T n t)
      ≤ ENNReal.ofReal (H.tree.leaves.length * (4 * Real.exp (-(n * t ^ 2 / 2)))) := by
  classical
  set β := Measure.pi fun _ : Fin n => T with hβ
  set e := Real.exp (-(n * t ^ 2 / 2)) with he
  have he0 : 0 ≤ e := (Real.exp_pos _).le
  have hlen := List.length_pos_of_ne_nil (DTree.leaves_ne_nil H.tree)
  rcases Nat.eq_zero_or_pos n with hn | hn
  · refine prob_le_one.trans ?_
    rw [ENNReal.one_le_ofReal, he, hn]
    simp only [Nat.cast_zero, zero_mul, zero_div, neg_zero, Real.exp_zero, mul_one]
    have : (1 : ℝ) ≤ H.tree.leaves.length := by exact_mod_cast hlen
    linarith
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  set X : List Bool → (Fin n → FreeMonoid α) → Ω → ℝ := fun s b ω =>
    ∑ i, agreeAt O H F kl kh s (b i) ω with hXd
  set Y : List Bool → (Fin n → FreeMonoid α) → ℝ := fun s b =>
    ∑ i, agreeProb O H F kl kh s (b i) with hYd
  set mn : List Bool → ℝ := fun s => agreeMean O H F kl kh T s with hmnd
  have hXm : ∀ s b, Measurable (X s b) := fun s b =>
    Finset.measurable_sum _ fun i _ => measurable_agreeAt O H F kl kh s (b i)
  set A1 : List Bool → Set (Ω × (Fin n → FreeMonoid α)) := fun s =>
    {p | Function.Injective p.2 ∧ Y s p.2 + n * (t / 2) ≤ X s p.2 p.1}
  set A2 : List Bool → Set (Ω × (Fin n → FreeMonoid α)) := fun s =>
    {p | Function.Injective p.2 ∧ X s p.2 p.1 ≤ Y s p.2 - n * (t / 2)}
  set A3 : List Bool → Set (Fin n → FreeMonoid α) := fun s => {b | n * (mn s + t / 2) ≤ Y s b}
  set A4 : List Bool → Set (Fin n → FreeMonoid α) := fun s => {b | Y s b ≤ n * (mn s - t / 2)}
  have hsub : scoreBad O H F kl kh T n t ⊆ ⋃ s ∈ H.tree.leaves.toFinset,
      ((A1 s ∪ A2 s) ∪ (Prod.snd ⁻¹' (A3 s ∪ A4 s))) := by
    rintro ⟨ω, b⟩ ⟨hinj, s, hs, h⟩
    refine Set.mem_iUnion₂.2 ⟨s, List.mem_toFinset.2 hs, ?_⟩
    simp only [Set.mem_union, Set.mem_preimage, Set.mem_ofPred_eq, A1, A2, A3, A4]
    have e1 : (n : ℝ) * (mn s + t) = n * (mn s + t / 2) + n * (t / 2) := by ring
    have e2 : (n : ℝ) * (mn s - t) = n * (mn s - t / 2) - n * (t / 2) := by ring
    rcases h with h | h
    · by_cases h1 : Y s b + n * (t / 2) ≤ X s b ω
      · exact Or.inl (Or.inl ⟨hinj, h1⟩)
      · right; left
        show (n : ℝ) * (mn s + t / 2) ≤ Y s b
        have : (n : ℝ) * (mn s + t) ≤ X s b ω := h
        linarith
    · by_cases h1 : X s b ω ≤ Y s b - n * (t / 2)
      · exact Or.inl (Or.inr ⟨hinj, h1⟩)
      · right; right
        show Y s b ≤ (n : ℝ) * (mn s - t / 2)
        have : X s b ω ≤ (n : ℝ) * (mn s - t) := h
        linarith
  have hexp : Real.exp (-2 * ((Finset.univ : Finset (Fin n)).card : ℝ) * (t / 2) ^ 2) = e := by
    rw [he, Finset.card_univ, Fintype.card_fin]
    congr 1
    ring
  -- the noise's tails, given distinct strings
  have hind : ∀ s (b : Fin n → FreeMonoid α), Function.Injective b →
      iIndepFun (fun i ω => agreeAt O H F kl kh s (b i) ω) μ := by
    intro s b hb
    exact ((read_iIndep O hF kl kh).precomp hb).comp
      (fun i r => if r ≠ .undecided ∧ H.Classifies s (b i) (r = .accept) then (1 : ℝ) else 0)
      fun i => measurable_from_top
  have hIcc : ∀ s (b : Fin n → FreeMonoid α) (i : Fin n),
      ∀ᵐ ω ∂μ, agreeAt O H F kl kh s (b i) ω ∈ Set.Icc (0 : ℝ) 1 := fun s b i =>
    Filter.Eventually.of_forall (agreeAt_mem O H F kl kh s (b i))
  have hmean : ∀ s (b : Fin n → FreeMonoid α),
      ∑ i ∈ Finset.univ, μ[fun ω => agreeAt O H F kl kh s (b i) ω]
        = ((Finset.univ : Finset (Fin n)).card : ℝ) * (Y s b / n) := by
    intro s b
    rw [Finset.card_univ, Fintype.card_fin, mul_div_cancel₀ _ hn'.ne']
    rfl
  have hsec : ∀ (S : Set (Ω × (Fin n → FreeMonoid α))),
      (∀ b, MeasurableSet {ω | (ω, b) ∈ S}) →
      (∀ b, μ {ω | (ω, b) ∈ S} ≤ ENNReal.ofReal e) → (μ.prod β) S ≤ ENNReal.ofReal e := by
    intro S hm hb
    rw [Measure.prod_apply_symm (measurableSet_prod_countable_right S hm)]
    calc ∫⁻ b, μ ((fun x => (x, b)) ⁻¹' S) ∂β ≤ ∫⁻ _b, ENNReal.ofReal e ∂β :=
          lintegral_mono fun b => hb b
      _ = ENNReal.ofReal e := by rw [lintegral_const, measure_univ, mul_one]
  have h1 : ∀ s, (μ.prod β) (A1 s) ≤ ENNReal.ofReal e := by
    intro s
    refine hsec _ (fun b => ?_) fun b => ?_
    · show MeasurableSet ({_ω | Function.Injective b} ∩ {ω | Y s b + n * (t / 2) ≤ X s b ω})
      exact (MeasurableSet.const _).inter (measurableSet_le measurable_const (hXm s b))
    · by_cases hb : Function.Injective b
      · have h := sumUpper_le (fun i ω => agreeAt O H F kl kh s (b i) ω) Finset.univ
          (Y s b / n) (t / 2) (fun i => (measurable_agreeAt O H F kl kh s (b i)).aemeasurable)
          (hind s b hb) (hIcc s b) (le_of_eq (hmean s b)) (by linarith)
        rw [hexp] at h
        rw [← ofReal_measureReal]
        refine ENNReal.ofReal_le_ofReal (le_trans (measureReal_mono ?_) h)
        intro ω hω
        simp only [Set.mem_ofPred_eq] at hω ⊢
        rw [Finset.card_univ, Fintype.card_fin, mul_add, mul_div_cancel₀ _ hn'.ne']
        exact hω.2
      · have : {ω | (ω, b) ∈ A1 s} = ∅ := by
          ext ω; simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, A1]
          exact fun h => hb h.1
        rw [this, measure_empty]; exact bot_le
  have h2 : ∀ s, (μ.prod β) (A2 s) ≤ ENNReal.ofReal e := by
    intro s
    refine hsec _ (fun b => ?_) fun b => ?_
    · show MeasurableSet ({_ω | Function.Injective b} ∩ {ω | X s b ω ≤ Y s b - n * (t / 2)})
      exact (MeasurableSet.const _).inter (measurableSet_le (hXm s b) measurable_const)
    · by_cases hb : Function.Injective b
      · have h := sumLower_le (fun i ω => agreeAt O H F kl kh s (b i) ω) Finset.univ
          (Y s b / n) (t / 2) (fun i => (measurable_agreeAt O H F kl kh s (b i)).aemeasurable)
          (hind s b hb) (hIcc s b) (le_of_eq (hmean s b).symm) (by linarith)
        rw [hexp] at h
        rw [← ofReal_measureReal]
        refine ENNReal.ofReal_le_ofReal (le_trans (measureReal_mono ?_) h)
        intro ω hω
        simp only [Set.mem_ofPred_eq] at hω ⊢
        rw [Finset.card_univ, Fintype.card_fin, mul_sub, mul_div_cancel₀ _ hn'.ne']
        exact hω.2
      · have : {ω | (ω, b) ∈ A2 s} = ∅ := by
          ext ω; simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, A2]
          exact fun h => hb h.1
        rw [this, measure_empty]; exact bot_le
  -- the strings' tails
  have hpm : ∀ s, Measurable (agreeProb O H F kl kh s) := fun s => measurable_from_top
  have hindβ : ∀ s,
      iIndepFun (fun i (b : Fin n → FreeMonoid α) => agreeProb O H F kl kh s (b i)) β :=
    fun s => iIndepFun_pi (X := fun _ => agreeProb O H F kl kh s) fun _ => (hpm s).aemeasurable
  have hIccβ : ∀ s (i : Fin n), ∀ᵐ b ∂β, agreeProb O H F kl kh s (b i) ∈ Set.Icc (0 : ℝ) 1 :=
    fun s i => Filter.Eventually.of_forall fun b => agreeProb_mem O H F kl kh s (b i)
  have hmeanβ : ∀ s,
      ∑ i ∈ Finset.univ, β[fun b : Fin n → FreeMonoid α => agreeProb O H F kl kh s (b i)]
      = ((Finset.univ : Finset (Fin n)).card : ℝ) * mn s := by
    intro s
    have : ∀ i : Fin n, β[fun b : Fin n → FreeMonoid α => agreeProb O H F kl kh s (b i)] = mn s :=
      fun i => integral_comp_eval (μ := fun _ : Fin n => T)
        (hpm s).aestronglyMeasurable
    simp only [this, Finset.sum_const, nsmul_eq_mul]
  have hsnd : ∀ T : Set (Fin n → FreeMonoid α), (μ.prod β) (Prod.snd ⁻¹' T) = β T := by
    intro T
    rw [← Set.univ_prod, Measure.prod_prod, measure_univ, one_mul]
  have h3 : ∀ s, β (A3 s) ≤ ENNReal.ofReal e := by
    intro s
    have h := sumUpper_le (fun i (b : Fin n → FreeMonoid α) => agreeProb O H F kl kh s (b i))
      Finset.univ (mn s) (t / 2) (fun i => ((hpm s).comp (measurable_pi_apply i)).aemeasurable)
      (hindβ s) (hIccβ s) (le_of_eq (hmeanβ s)) (by linarith)
    rw [hexp] at h
    rw [← ofReal_measureReal]
    refine ENNReal.ofReal_le_ofReal (le_trans (measureReal_mono ?_) h)
    intro b hb
    simp only [Set.mem_ofPred_eq, A3] at hb ⊢
    rw [Finset.card_univ, Fintype.card_fin]
    exact hb
  have h4 : ∀ s, β (A4 s) ≤ ENNReal.ofReal e := by
    intro s
    have h := sumLower_le (fun i (b : Fin n → FreeMonoid α) => agreeProb O H F kl kh s (b i))
      Finset.univ (mn s) (t / 2) (fun i => ((hpm s).comp (measurable_pi_apply i)).aemeasurable)
      (hindβ s) (hIccβ s) (le_of_eq (hmeanβ s).symm) (by linarith)
    rw [hexp] at h
    rw [← ofReal_measureReal]
    refine ENNReal.ofReal_le_ofReal (le_trans (measureReal_mono ?_) h)
    intro b hb
    simp only [Set.mem_ofPred_eq, A4] at hb ⊢
    rw [Finset.card_univ, Fintype.card_fin]
    exact hb
  have hs : ∀ s, (μ.prod β) ((A1 s ∪ A2 s) ∪ (Prod.snd ⁻¹' (A3 s ∪ A4 s)))
      ≤ 4 * ENNReal.ofReal e := by
    intro s
    calc (μ.prod β) ((A1 s ∪ A2 s) ∪ (Prod.snd ⁻¹' (A3 s ∪ A4 s)))
        ≤ (μ.prod β) (A1 s) + (μ.prod β) (A2 s) + (β (A3 s) + β (A4 s)) := by
          refine (measure_union_le _ _).trans (add_le_add (measure_union_le _ _) ?_)
          rw [hsnd]
          exact measure_union_le _ _
      _ ≤ ENNReal.ofReal e + ENNReal.ofReal e + (ENNReal.ofReal e + ENNReal.ofReal e) :=
          add_le_add (add_le_add (h1 s) (h2 s)) (add_le_add (h3 s) (h4 s))
      _ = 4 * ENNReal.ofReal e := by ring
  calc (μ.prod β) (scoreBad O H F kl kh T n t)
      ≤ (μ.prod β) (⋃ s ∈ H.tree.leaves.toFinset,
          ((A1 s ∪ A2 s) ∪ (Prod.snd ⁻¹' (A3 s ∪ A4 s)))) := measure_mono hsub
    _ ≤ ∑ s ∈ H.tree.leaves.toFinset, (μ.prod β) ((A1 s ∪ A2 s) ∪ (Prod.snd ⁻¹' (A3 s ∪ A4 s))) :=
        measure_biUnion_finset_le _ _
    _ ≤ ∑ _s ∈ H.tree.leaves.toFinset, 4 * ENNReal.ofReal e := Finset.sum_le_sum fun s _ => hs s
    _ = H.tree.leaves.toFinset.card * (4 * ENNReal.ofReal e) := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ H.tree.leaves.length * (4 * ENNReal.ofReal e) := by
        gcongr
        exact_mod_cast List.toFinset_card_le _
    _ = ENNReal.ofReal (H.tree.leaves.length * (4 * e)) := by
        rw [ENNReal.ofReal_mul (by positivity), ENNReal.ofReal_mul (by norm_num),
          ENNReal.ofReal_natCast]
        norm_num

end Score

section Pick

variable {σ : Type*} (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
  (H : HypothesisDFA α) (F : Finset (FreeMonoid α)) (kl kh : ℕ)

/-- Off `scoreBad`, the leaf of highest score misjudges at most `2t`, the family's undecided
share `u` and twice its misread share `m` more than any leaf. -/
theorem pickLeaf_le (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)
    (T : Measure (FreeMonoid α)) [IsProbabilityMeasure T] (n : ℕ) (hn : 0 < n)
    (t u m γ : ℝ) (ω : Ω) (xs : Fin n → FreeMonoid α) (hxs : Function.Injective xs)
    (hgood : (ω, xs) ∉ scoreBad O H F kl kh T n t)
    (hu : ∫ x, readProb O F kl kh x .undecided ∂T ≤ u)
    (hm : ∫ x, misreadProb O F kl kh x ∂T ≤ m)
    (hs : ∃ s ∈ H.tree.leaves, T.real
      {x | ¬ H.Classifies s x (M.eval x.toList ∈ M.accept)} ≤ γ) :
    T.real {x | ¬ H.Classifies
      (pickLeaf H (fun z => familyRead O.mq F kl kh z ω) xs) x (M.eval x.toList ∈ M.accept)}
      ≤ γ + 2 * t + u + 2 * m := by
  classical
  set U := T with hUd
  set tgt : FreeMonoid α → Prop := fun x => M.eval x.toList ∈ M.accept
  set err : List Bool → ℝ := fun s => U.real {x | ¬ H.Classifies s x (tgt x)} with herrd
  set cl : List Bool → FreeMonoid α → ℝ := fun s x => if H.Classifies s x (tgt x) then 1 else 0
  set mis : FreeMonoid α → Set Ω := fun x =>
    {ω | familyRead O.mq F kl kh x ω = if x ∈ O.L then .reject else .accept}
  set und : FreeMonoid α → Set Ω := fun x => {ω | familyRead O.mq F kl kh x ω = .undecided}
  have hmeasR : ∀ x (S : Set ARU), MeasurableSet {ω | familyRead O.mq F kl kh x ω ∈ S} :=
    fun x S => (measurable_from_nat.comp (measurable_voteCount O F x))
      (MeasurableSpace.measurableSet_top (s := S))
  have hmis : ∀ x, MeasurableSet (mis x) := fun x => hmeasR x {_}
  have hund : ∀ x, MeasurableSet (und x) := fun x => hmeasR x {_}
  -- decided and not misread, the read classifies as the target does
  have hkey : ∀ s x ω, ω ∉ und x → ω ∉ mis x → agreeAt O H F kl kh s x ω = cl s x := by
    intro s x ω hu' hm'
    have hiff : (familyRead O.mq F kl kh x ω = .accept) ↔ tgt x := by
      show _ ↔ M.eval x.toList ∈ M.accept
      rw [← hL x]
      simp only [und, mis, Set.mem_ofPred_eq] at hu' hm'
      by_cases hx : x ∈ O.L
      · simp only [hx, if_true] at hm'
        rcases h : familyRead O.mq F kl kh x ω <;> simp_all
      · simp only [hx, if_false] at hm'
        rcases h : familyRead O.mq F kl kh x ω <;> simp_all
    have hd : familyRead O.mq F kl kh x ω ≠ .undecided := hu'
    unfold agreeAt
    simp only [cl, propext hiff, ne_eq, hd, not_false_eq_true, true_and]
  have hlo : ∀ s x ω, cl s x - (und x).indicator 1 ω - (mis x).indicator 1 ω
      ≤ agreeAt O H F kl kh s x ω := by
    intro s x ω
    have h0 := (agreeAt_mem O H F kl kh s x ω).1
    have hc : cl s x ≤ 1 := by simp only [cl]; split_ifs <;> norm_num
    by_cases hu' : ω ∈ und x
    · rw [Set.indicator_of_mem hu']
      have : 0 ≤ (mis x).indicator (1 : Ω → ℝ) ω := Set.indicator_nonneg (fun _ _ => zero_le_one) _
      simp only [Pi.one_apply]
      linarith
    by_cases hm' : ω ∈ mis x
    · rw [Set.indicator_of_mem hm', Set.indicator_of_notMem hu']
      simp only [Pi.one_apply]
      linarith
    rw [Set.indicator_of_notMem hu', Set.indicator_of_notMem hm', hkey s x ω hu' hm']
    ring_nf
    rfl
  have hhi : ∀ s x ω, agreeAt O H F kl kh s x ω ≤ cl s x + (mis x).indicator 1 ω := by
    intro s x ω
    have h1 := (agreeAt_mem O H F kl kh s x ω).2
    have hc : 0 ≤ cl s x := by simp only [cl]; split_ifs <;> norm_num
    by_cases hm' : ω ∈ mis x
    · rw [Set.indicator_of_mem hm']
      simp only [Pi.one_apply]
      linarith
    rw [Set.indicator_of_notMem hm']
    by_cases hu' : ω ∈ und x
    · have : agreeAt O H F kl kh s x ω = 0 := by
        unfold agreeAt
        rw [if_neg]
        rintro ⟨h, -⟩
        exact h hu'
      simp only [this, add_zero]
      exact hc
    rw [hkey s x ω hu' hm']
    simp
  have hint1 : ∀ S : Set Ω, MeasurableSet S → Integrable (S.indicator (1 : Ω → ℝ)) μ :=
    fun S hS => (integrable_const 1).indicator hS
  have hiA : ∀ s x, Integrable (agreeAt O H F kl kh s x) μ := fun s x =>
    Integrable.of_bound (measurable_agreeAt O H F kl kh s x).aestronglyMeasurable 1
      (Filter.Eventually.of_forall fun ω => by
        have hb := agreeAt_mem O H F kl kh s x ω
        rw [Real.norm_eq_abs, abs_le]; exact ⟨by linarith [hb.1], hb.2⟩)
  have hplo : ∀ s x, cl s x - readProb O F kl kh x .undecided - misreadProb O F kl kh x
      ≤ agreeProb O H F kl kh s x := by
    intro s x
    have hA : Integrable (fun ω => cl s x - (und x).indicator (1 : Ω → ℝ) ω) μ :=
      (integrable_const _).sub (hint1 _ (hund x))
    have h := integral_mono (f := fun ω => cl s x - (und x).indicator 1 ω
        - (mis x).indicator 1 ω) (hA.sub (hint1 _ (hmis x))) (hiA s x) (hlo s x)
    have e : ∫ ω, cl s x - (und x).indicator (1 : Ω → ℝ) ω - (mis x).indicator 1 ω ∂μ
        = cl s x - readProb O F kl kh x .undecided - misreadProb O F kl kh x := by
      rw [integral_sub hA (hint1 _ (hmis x)), integral_sub (integrable_const _) (hint1 _ (hund x)),
        integral_const, integral_indicator_one (hund x), integral_indicator_one (hmis x)]
      simp [readProb, misreadProb, und, mis]
    rw [e] at h
    exact h
  have hphi : ∀ s x, agreeProb O H F kl kh s x ≤ cl s x + misreadProb O F kl kh x := by
    intro s x
    have h := integral_mono (g := fun ω => cl s x + (mis x).indicator 1 ω) (hiA s x)
      ((integrable_const (cl s x)).add (hint1 _ (hmis x))) (hhi s x)
    have e : ∫ ω, cl s x + (mis x).indicator (1 : Ω → ℝ) ω ∂μ
        = cl s x + misreadProb O F kl kh x := by
      rw [integral_add (integrable_const _) (hint1 _ (hmis x)), integral_const,
        integral_indicator_one (hmis x)]
      simp [misreadProb, mis]
    rw [e] at h
    exact h
  -- over the strings of length `L`
  have hcl : ∀ s, ∫ x, cl s x ∂U = 1 - err s := by
    intro s
    have e : (fun x => cl s x) = {x | H.Classifies s x (tgt x)}.indicator 1 := by
      funext x
      simp only [cl, Set.indicator_apply, Set.mem_ofPred_eq, Pi.one_apply]
    rw [e, integral_indicator_one (MeasurableSpace.measurableSet_top), herrd]
    dsimp only
    rw [← probReal_compl_eq_one_sub₀ (MeasurableSpace.measurableSet_top).nullMeasurableSet]
    congr 1
    ext x
    simp
  have hcl01 : ∀ s x, cl s x ∈ Set.Icc (0 : ℝ) 1 := fun s x => by
    simp only [cl]; split_ifs <;> norm_num
  have hiC : ∀ s, Integrable (cl s) U := fun s => integrable_of_mem_Icc (hcl01 s)
  have hiR : Integrable (fun x => readProb O F kl kh x .undecided) U :=
    integrable_of_mem_Icc fun _ => ⟨measureReal_nonneg, measureReal_le_one⟩
  have hiM : Integrable (fun x => misreadProb O F kl kh x) U :=
    integrable_of_mem_Icc fun _ => ⟨measureReal_nonneg, measureReal_le_one⟩
  have hiP : ∀ s, Integrable (agreeProb O H F kl kh s) U := fun s =>
    integrable_of_mem_Icc (agreeProb_mem O H F kl kh s)
  have hmlo : ∀ s, 1 - err s - ∫ x, readProb O F kl kh x .undecided ∂U
      - ∫ x, misreadProb O F kl kh x ∂U ≤ agreeMean O H F kl kh T s := by
    intro s
    have hA : Integrable (fun x => cl s x - readProb O F kl kh x .undecided) U := (hiC s).sub hiR
    have h := integral_mono (f := fun x => cl s x - readProb O F kl kh x .undecided
      - misreadProb O F kl kh x) (hA.sub hiM) (hiP s) (hplo s)
    rw [integral_sub hA hiM, integral_sub (hiC s) hiR, hcl s] at h
    exact h
  have hmhi : ∀ s, agreeMean O H F kl kh T s ≤ 1 - err s + ∫ x, misreadProb O F kl kh x ∂U := by
    intro s
    have h := integral_mono (g := fun x => cl s x + misreadProb O F kl kh x) (hiP s)
      ((hiC s).add hiM) (hphi s)
    rw [integral_add (hiC s) hiM, hcl s] at h
    exact h
  -- the picked leaf
  set sc : FreeMonoid α → ARU := fun z => familyRead O.mq F kl kh z ω
  obtain ⟨sh, hsh⟩ : ∃ sh, H.tree.leaves.argmax (leafScore H sc xs) = some sh :=
    Option.ne_none_iff_exists'.1 (by
      rw [Ne, List.argmax_eq_none]; exact DTree.leaves_ne_nil _)
  have hpick : pickLeaf H sc xs = sh := by simp [pickLeaf, hsh]
  have hshm : sh ∈ H.tree.leaves := List.argmax_mem (Option.mem_def.2 hsh)
  obtain ⟨s₀, hs₀, hs₀η⟩ := hs
  have hle : (leafScore H sc xs s₀ : ℝ) ≤ leafScore H sc xs sh := by
    exact_mod_cast List.le_of_mem_argmax hs₀ (Option.mem_def.2 hsh)
  rw [leafScore_eq, leafScore_eq] at hle
  have hok : ∀ s ∈ H.tree.leaves,
      ∑ i, agreeAt O H F kl kh s (xs i) ω < n * (agreeMean O H F kl kh T s + t)
      ∧ n * (agreeMean O H F kl kh T s - t) < ∑ i, agreeAt O H F kl kh s (xs i) ω := by
    intro s hs
    by_contra h
    apply hgood
    refine ⟨hxs, s, hs, ?_⟩
    rcases not_and_or.1 h with h | h
    · exact Or.inl (not_lt.1 h)
    · exact Or.inr (not_lt.1 h)
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hmean : agreeMean O H F kl kh T s₀ - 2 * t < agreeMean O H F kl kh T sh := by
    have h1 := (hok sh hshm).1
    have h2 := (hok s₀ hs₀).2
    by_contra h
    have h3 : (n : ℝ) * (agreeMean O H F kl kh T sh + t)
        ≤ n * (agreeMean O H F kl kh T s₀ - t) :=
      mul_le_mul_of_nonneg_left (by linarith) hn'.le
    linarith
  rw [hpick]
  have a1 := hmlo s₀
  have a2 := hmhi sh
  show err sh ≤ γ + 2 * t + u + 2 * m
  linarith

end Pick

section Collide

/-- Held-out strings of length `L`, drawn from `T`, which holds at most `p₀` on any first `k`
letters, repeat, or one's first `k` letters fall in `S`, with chance at most `n |S| p₀ + n² p₀`. -/
theorem batch_collide_le (T : Measure (FreeMonoid α)) [IsProbabilityMeasure T] (L k n : ℕ)
    {p₀ : ℝ} (hlen : T {x | x.toList.length ≠ L} = 0)
    (hpre : ∀ u, T.real {x | prefixOf x k = u} ≤ p₀) (S : Finset (FreeMonoid α)) :
    (Measure.pi fun _ : Fin n => T)
      {b | ¬ Function.Injective b ∨ ∃ i, (b i).toList.length ≠ L ∨ prefixOf (b i) k ∈ S}
      ≤ ENNReal.ofReal (n * S.card * p₀ + n ^ 2 * p₀) := by
  classical
  set U := T with hUd
  set β := Measure.pi fun _ : Fin n => U with hβd
  have hp₀ : 0 ≤ p₀ := measureReal_nonneg.trans (hpre 1)
  have hev : ∀ (i : Fin n) (T : Set (FreeMonoid α)), β {b | b i ∈ T} = U T := fun i T =>
    (measurePreserving_eval (fun _ : Fin n => U) i).measure_preimage
      (MeasurableSpace.measurableSet_top).nullMeasurableSet
  have hatom : ∀ x : FreeMonoid α, U {x} ≤ ENNReal.ofReal p₀ :=
    fun x => by
      rw [← ofReal_measureReal]
      refine ENNReal.ofReal_le_ofReal ((measureReal_mono (fun y hy => ?_) (measure_ne_top _ _)).trans
        (hpre (prefixOf x k)))
      simp only [Set.mem_singleton_iff] at hy
      simp only [Set.mem_ofPred_eq, hy]
  -- a string of another length
  have h1 : ∀ i : Fin n, β {b | (b i).toList.length ≠ L} = 0 := fun i => by
    exact (hev i {x | x.toList.length ≠ L}).trans hlen
  -- a string starting in `S`
  have h2 : ∀ i : Fin n, β {b | prefixOf (b i) k ∈ S}
      ≤ ENNReal.ofReal (S.card * p₀) := fun i => by
    refine (hev i {x | prefixOf x k ∈ S}).le.trans ?_
    rw [← ofReal_measureReal]
    refine ENNReal.ofReal_le_ofReal ?_
    have hset : {x : FreeMonoid α | prefixOf x k ∈ S} = ⋃ u ∈ S, {x | prefixOf x k = u} := by
      ext x; simp
    rw [hset]
    refine (measureReal_biUnion_finset_le _ _).trans ?_
    calc ∑ u ∈ S, U.real {x | prefixOf x k = u} ≤ ∑ _u ∈ S, p₀ :=
          Finset.sum_le_sum fun u _ => hpre u
      _ = S.card * p₀ := by rw [Finset.sum_const, nsmul_eq_mul]
  -- two draws that coincide
  have h3 : ∀ i j : Fin n, β {b | i ≠ j ∧ b i = b j}
      ≤ ENNReal.ofReal p₀ := fun i j => by
    by_cases hij : i = j
    · have : {b : Fin n → FreeMonoid α | i ≠ j ∧ b i = b j} = ∅ := by
        ext b; simp [hij]
      rw [this, measure_empty]
      exact bot_le
    have hind : IndepFun (fun b : Fin n → FreeMonoid α => b i) (fun b => b j) β :=
      (iIndepFun_pi (X := fun _ (x : FreeMonoid α) => x) fun _ =>
        measurable_id.aemeasurable).indepFun hij
    have hset : {b : Fin n → FreeMonoid α | i ≠ j ∧ b i = b j}
        ⊆ ⋃ x : FreeMonoid α, ((fun b : Fin n → FreeMonoid α => b i) ⁻¹' {x}
          ∩ (fun b => b j) ⁻¹' {x}) := by
      rintro b ⟨-, h⟩
      exact Set.mem_iUnion.2 ⟨b i, rfl, h.symm⟩
    calc β {b | i ≠ j ∧ b i = b j}
        ≤ ∑' x : FreeMonoid α, β ((fun b : Fin n → FreeMonoid α => b i) ⁻¹' {x}
          ∩ (fun b => b j) ⁻¹' {x}) := (measure_mono hset).trans (measure_iUnion_le _)
      _ = ∑' x : FreeMonoid α, U {x} * U {x} := by
          refine tsum_congr fun x => ?_
          rw [hind.measure_inter_preimage_eq_mul _ _ (measurableSet_singleton x)
            (measurableSet_singleton x)]
          exact congrArg₂ (· * ·) (hev i {x}) (hev j {x})
      _ ≤ ∑' x : FreeMonoid α, U {x} * ENNReal.ofReal p₀ :=
          ENNReal.tsum_le_tsum fun x => by gcongr; exact hatom x
      _ = ENNReal.ofReal p₀ := by
          rw [ENNReal.tsum_mul_right]
          have : ∑' x : FreeMonoid α, U {x} = 1 := by
            rw [← measure_iUnion (fun x y hxy => Set.disjoint_singleton.2 hxy)
              (fun x => measurableSet_singleton x), Set.iUnion_singleton_eq_range, Set.range_id',
              measure_univ]
          rw [this, one_mul]
  have hsub : {b : Fin n → FreeMonoid α | ¬ Function.Injective b
      ∨ ∃ i, (b i).toList.length ≠ L ∨ prefixOf (b i) k ∈ S}
      ⊆ ((⋃ i, {b : Fin n → FreeMonoid α | (b i).toList.length ≠ L})
        ∪ ⋃ i, {b : Fin n → FreeMonoid α | prefixOf (b i) k ∈ S})
        ∪ ⋃ i, ⋃ j, {b : Fin n → FreeMonoid α | i ≠ j ∧ b i = b j} := by
    rintro b (h | ⟨i, h | h⟩)
    · obtain ⟨i, j, hb, hij⟩ : ∃ i j, b i = b j ∧ i ≠ j := by
        by_contra hc
        push Not at hc
        exact h fun i j hb => hc i j hb
      exact Or.inr (Set.mem_iUnion₂.2 ⟨i, j, hij, hb⟩)
    · exact Or.inl (Or.inl (Set.mem_iUnion.2 ⟨i, h⟩))
    · exact Or.inl (Or.inr (Set.mem_iUnion.2 ⟨i, h⟩))
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg _
  calc β _ ≤ β (((⋃ i, {b : Fin n → FreeMonoid α | (b i).toList.length ≠ L})
        ∪ ⋃ i, {b : Fin n → FreeMonoid α | prefixOf (b i) k ∈ S})
        ∪ ⋃ i, ⋃ j, {b : Fin n → FreeMonoid α | i ≠ j ∧ b i = b j}) := measure_mono hsub
    _ ≤ (∑ i : Fin n, β {b | (b i).toList.length ≠ L} + ∑ i : Fin n, β {b | prefixOf (b i) k ∈ S})
        + ∑ i : Fin n, ∑ j : Fin n, β {b | i ≠ j ∧ b i = b j} := by
        refine (measure_union_le _ _).trans (add_le_add ((measure_union_le _ _).trans
          (add_le_add (measure_iUnion_fintype_le _ _) (measure_iUnion_fintype_le _ _)))
          ((measure_iUnion_fintype_le _ _).trans (Finset.sum_le_sum fun i _ =>
            measure_iUnion_fintype_le _ _)))
    _ ≤ (∑ _i : Fin n, 0 + ∑ _i : Fin n, ENNReal.ofReal (S.card * p₀))
        + ∑ _i : Fin n, ∑ _j : Fin n, ENNReal.ofReal p₀ := by
        exact add_le_add (add_le_add (Finset.sum_le_sum fun i _ => (h1 i).le)
          (Finset.sum_le_sum fun i _ => h2 i))
          (Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => h3 i j)
    _ = ENNReal.ofReal (n * S.card * p₀ + n ^ 2 * p₀) := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_zero,
          zero_add]
        rw [show (n : ℝ) * S.card * p₀ + n ^ 2 * p₀ = n * (S.card * p₀) + n * (n * p₀) by ring,
          ENNReal.ofReal_add (mul_nonneg hn0 (mul_nonneg (Nat.cast_nonneg _) hp₀))
            (mul_nonneg hn0 (mul_nonneg hn0 hp₀)),
          ENNReal.ofReal_mul hn0, ENNReal.ofReal_mul hn0, ENNReal.ofReal_mul hn0,
          ENNReal.ofReal_natCast]

end Collide

end OrthoDFA

end
