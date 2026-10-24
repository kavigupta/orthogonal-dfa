import OrthoDFA.Utils.AnchorResolves
import OrthoDFA.Proofs.Maclaurin
import OrthoDFA.Proofs.Hoeffding
import OrthoDFA.Proofs.FamilyRead

/-!
# A family drawn from a set that votes with margin

Over a uniform `W`-subset `F` of `C`, with `X_F` the members reading `w` on side `β` and `ρ̄ ≥ r`
their mean chance over `C`, Chernoff at `−λ` and Maclaurin's inequality give

    E_F P(X_F ≤ W(r − g)) ≤ e^{λW(r − g)} (1 − ρ̄ + ρ̄e^{−λ})^W ≤ exp(−2Wg²)

at `λ = 4g`, by Hoeffding's lemma for one bit.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped NNReal

/-- Hoeffding's lemma for one bit at rate `ρ`. -/
lemma bern_mgf_le {ρ : ℝ} (h0 : 0 ≤ ρ) (h1 : ρ ≤ 1) (l : ℝ) :
    1 - ρ + ρ * Real.exp (-l) ≤ Real.exp (-(l * ρ) + l ^ 2 / 8) := by
  set D : Measure Bool := bernoulliMeasure true false ⟨ρ, h0, h1⟩ with hD
  set X : Bool → ℝ := fun b => if b then 1 else 0 with hX
  have hXI : ∀ᵐ b ∂D, X b ∈ Set.Icc (0 : ℝ) 1 :=
    ae_of_all _ (fun b => by cases b <;> simp [hX])
  have hsub := hasSubgaussianMGF_of_mem_Icc (X := X) (μ := D)
    (measurable_of_countable _).aemeasurable hXI
  have hc : ((‖(1 : ℝ) - 0‖₊ / 2) ^ 2 : ℝ≥0) = (1 / 4 : ℝ≥0) := by norm_num
  rw [hc] at hsub
  have hm := hsub.mgf_le (-l)
  have hDt : D.real {true} = ρ :=
    bernoulliMeasure_real_apply_of_mem_of_notMem _ (Set.to_countable _).measurableSet rfl
      (by simp)
  have hDf : D.real {false} = 1 - ρ := by
    have h := measureReal_compl (μ := D) (s := {true}) (Set.to_countable _).measurableSet
    rw [probReal_univ, hDt] at h
    rw [← h]
    congr 1
    ext b
    cases b <;> simp
  have hmean : D[X] = ρ := by
    rw [integral_fintype Integrable.of_finite]
    simp [hDt, hX]
  unfold mgf at hm
  rw [integral_fintype Integrable.of_finite] at hm
  simp only [hmean] at hm
  simp only [Fintype.sum_bool, hDt, hDf, hX, smul_eq_mul] at hm
  norm_num at hm
  have e1 : Real.exp (-(l * (1 - ρ))) = Real.exp (-l) * Real.exp (l * ρ) := by
    rw [← Real.exp_add]; ring_nf
  have e2 : Real.exp (l ^ 2 / 8) = Real.exp (-(l * ρ) + l ^ 2 / 8) * Real.exp (l * ρ) := by
    rw [← Real.exp_add]; ring_nf
  have hpos := Real.exp_pos (l * ρ)
  have hm' : ρ * Real.exp (-(l * (1 - ρ))) + (1 - ρ) * Real.exp (l * ρ)
      ≤ Real.exp (l ^ 2 / 8) := by
    convert hm using 2
    ring_nf
  rw [e1, e2] at hm'
  refine le_of_mul_le_mul_right ?_ hpos
  nlinarith [hm']

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

open scoped Classical in
/-- `1` where `w·u` reads `β`. -/
noncomputable def sideBit (O : Oracle μ S) (w : S) (β : Bool) (u : S) (ω : Ω) : ℝ :=
  if readsAs O.mq (w * u) β ω then 1 else 0

lemma measurableSet_readsAs (O : Oracle μ S) (s : S) (β : Bool) :
    MeasurableSet {ω | readsAs O.mq s β ω} := by
  have h1 : MeasurableSet {ω | O.mq s ω = 1} := measurableSet_eq_fun (mq_meas O s) measurable_const
  cases β
  · have : {ω | readsAs O.mq s false ω} = {ω | O.mq s ω = 1}ᶜ := by
      ext ω; simp [readsAs]
    rw [this]; exact h1.compl
  · have : {ω | readsAs O.mq s true ω} = {ω | O.mq s ω = 1} := by
      ext ω; simp [readsAs]
    rw [this]; exact h1

lemma sideBit_meas (O : Oracle μ S) (w : S) (β : Bool) (u : S) :
    Measurable (sideBit O w β u) := by
  classical
  exact Measurable.ite (measurableSet_readsAs O _ β) measurable_const measurable_const

lemma sideBit_iIndep (O : Oracle μ S) (w : S) (β : Bool) :
    iIndepFun (fun u => sideBit O w β u) μ := by
  classical
  have hinj : Function.Injective (fun u : S => w * u) := fun a b h => mul_left_cancel h
  have h := ((mq_indep O).precomp hinj).comp
    (fun _ (x : ℝ) => if (x = 1 ↔ β = true) then (1 : ℝ) else 0) (fun _ => by
      refine Measurable.ite ?_ measurable_const measurable_const
      cases β
      · have : {x : ℝ | x = 1 ↔ false = true} = {1}ᶜ := by ext x; simp
        rw [this]; exact (measurableSet_singleton 1).compl
      · have : {x : ℝ | x = 1 ↔ true = true} = {1} := by ext x; simp
        rw [this]; exact measurableSet_singleton 1)
  convert h using 1
  funext u ω
  exact if_congr Iff.rfl rfl rfl

lemma mgf_sideBit (O : Oracle μ S) (w : S) (β : Bool) (u : S) (l : ℝ) :
    mgf (sideBit O w β u) μ (-l)
      = 1 - μ.real {ω | readsAs O.mq (w * u) β ω}
        + μ.real {ω | readsAs O.mq (w * u) β ω} * Real.exp (-l) := by
  classical
  set H := {ω | readsAs O.mq (w * u) β ω} with hH
  have hHm : MeasurableSet H := measurableSet_readsAs O _ β
  have hpt : (fun ω => Real.exp (-l * sideBit O w β u ω))
      = fun ω => 1 + H.indicator (fun _ => Real.exp (-l) - 1) ω := by
    funext ω
    by_cases h : ω ∈ H
    · have h' : readsAs O.mq (w * u) β ω := h
      simp [sideBit, h', Set.indicator_of_mem h]
    · have h' : ¬ readsAs O.mq (w * u) β ω := h
      simp [sideBit, h', Set.indicator_of_notMem h]
  unfold mgf
  rw [hpt, integral_add (integrable_const _) ((integrable_const _).indicator hHm),
    integral_const, integral_indicator_const _ hHm]
  simp only [probReal_univ, smul_eq_mul, one_mul]
  ring

/-- The family's off-side chance, summed over the `W`-subsets of `C`. -/
theorem family_off_side_le (O : Oracle μ S) (C : Finset S) (w : S) (β : Bool) (r g : ℝ)
    (W kl kh : ℕ) (hr : r * C.card ≤ ∑ v ∈ C, μ.real {ω | readsAs O.mq (w * v) β ω})
    (hg : 0 ≤ g) (hk : kl < kh) (h1 : (kh : ℝ) ≤ W * (r - g))
    (h2 : W * (1 - r + g) ≤ kl) :
    ∑ F ∈ C.powersetCard W, μ.real {ω | familyRead O.mq F kl kh w ω ≠ sideRead β}
      ≤ C.card.choose W * Real.exp (-2 * W * g ^ 2) := by
  classical
  have hW : 0 < W := by
    rcases Nat.eq_zero_or_pos W with h | h
    · subst h
      have : (kh : ℝ) ≤ 0 := by simpa using h1
      have : kh = 0 := by exact_mod_cast le_antisymm this (Nat.cast_nonneg _)
      omega
    · exact h
  rcases Nat.eq_zero_or_pos C.card with hC | hC
  · rw [Finset.card_eq_zero.1 hC, Finset.powersetCard_eq_empty.2 (by simp; omega)]
    simp only [Finset.sum_empty]
    positivity
  set l : ℝ := 4 * g with hl
  have hl0 : 0 ≤ l := by positivity
  set ρ : S → ℝ := fun u => μ.real {ω | readsAs O.mq (w * u) β ω} with hρ
  set φ : S → ℝ := fun u => 1 - ρ u + ρ u * Real.exp (-l) with hφ
  have hρ01 : ∀ u, 0 ≤ ρ u ∧ ρ u ≤ 1 := fun u => ⟨measureReal_nonneg, measureReal_le_one⟩
  have hφ0 : ∀ u, 0 ≤ φ u := fun u => by
    have := hρ01 u
    have := Real.exp_pos (-l)
    simp only [hφ]; nlinarith
  set ε : ℝ := W * (r - g) with hε
  -- one family
  have hone : ∀ F ∈ C.powersetCard W,
      μ.real {ω | familyRead O.mq F kl kh w ω ≠ sideRead β}
        ≤ Real.exp (l * ε) * ∏ u ∈ F, φ u := by
    intro F hF
    have hFcard : F.card = W := (Finset.mem_powersetCard.1 hF).2
    have hsum : ∀ ω, ∑ u ∈ F, sideBit O w β u ω
        = ((F.filter fun u => readsAs O.mq (w * u) β ω).card : ℝ) := by
      intro ω
      simp only [sideBit]
      rw [Finset.sum_boole]
    have hsub : {ω | familyRead O.mq F kl kh w ω ≠ sideRead β}
        ⊆ {ω | (∑ u ∈ F, sideBit O w β u) ω ≤ ε} := by
      intro ω hω
      simp only [Set.mem_ofPred_eq, Finset.sum_apply] at hω ⊢
      rw [hsum]
      cases β
      · have hrej : ¬ voteCount O.mq F w ω ≤ kl := by
          intro h
          apply hω
          simp only [familyRead, sideRead]
          exact (readOf_reject hk).2 h
        have hsplit := Finset.card_filter_add_card_filter_not (s := F)
          (fun u => O.mq (w * u) ω = 1)
        have hfilt : (F.filter fun u => readsAs O.mq (w * u) false ω)
            = F.filter fun u => ¬ O.mq (w * u) ω = 1 := by
          refine Finset.filter_congr fun u _ => ?_
          simp [readsAs]
        rw [hfilt]
        have hv : voteCount O.mq F w ω = (F.filter fun u => O.mq (w * u) ω = 1).card := rfl
        have hlt : (kl : ℝ) < voteCount O.mq F w ω := by exact_mod_cast not_le.1 hrej
        have : ((F.filter fun u => ¬ O.mq (w * u) ω = 1).card : ℝ)
            = W - voteCount O.mq F w ω := by
          rw [hv, ← hFcard]
          have : (F.filter fun u => ¬ O.mq (w * u) ω = 1).card
              = F.card - (F.filter fun u => O.mq (w * u) ω = 1).card := by omega
          rw [this, Nat.cast_sub (by omega)]
        rw [this, hε]
        nlinarith
      · have hacc : ¬ kh ≤ voteCount O.mq F w ω := by
          intro h
          apply hω
          simp only [familyRead, sideRead]
          exact readOf_accept.2 h
        have hfilt : (F.filter fun u => readsAs O.mq (w * u) true ω)
            = F.filter fun u => O.mq (w * u) ω = 1 := by
          refine Finset.filter_congr fun u _ => ?_
          simp [readsAs]
        rw [hfilt]
        have hlt : (voteCount O.mq F w ω : ℝ) < kh := by exact_mod_cast not_le.1 hacc
        have hv : voteCount O.mq F w ω = (F.filter fun u => O.mq (w * u) ω = 1).card := rfl
        rw [← hv]
        linarith
    have hmeasS : Measurable (∑ u ∈ F, sideBit O w β u) := by
      rw [show (∑ u ∈ F, sideBit O w β u) = fun ω => ∑ u ∈ F, sideBit O w β u ω from by
        ext ω; simp]
      exact Finset.measurable_sum _ (fun u _ => sideBit_meas O w β u)
    have hint : Integrable (fun ω => Real.exp (-l * (∑ u ∈ F, sideBit O w β u) ω)) μ := by
      refine MeasureTheory.Integrable.of_mem_Icc 0 1 (by fun_prop) ?_
      refine ae_of_all _ fun ω => ⟨(Real.exp_pos _).le, Real.exp_le_one_iff.2 ?_⟩
      have : 0 ≤ (∑ u ∈ F, sideBit O w β u) ω := by
        rw [Finset.sum_apply]
        exact Finset.sum_nonneg fun u _ => by simp only [sideBit]; split_ifs <;> norm_num
      nlinarith
    have hch := measure_le_le_exp_mul_mgf (μ := μ) (X := ∑ u ∈ F, sideBit O w β u) ε
      (by linarith : -l ≤ 0) hint
    rw [(sideBit_iIndep O w β).mgf_sum (fun u => sideBit_meas O w β u) F] at hch
    refine (measureReal_mono hsub).trans (hch.trans (le_of_eq ?_))
    rw [neg_neg]
    congr 1
    exact Finset.prod_congr rfl fun u _ => mgf_sideBit O w β u l
  -- all of them
  have hN : (0 : ℝ) < C.card := by exact_mod_cast hC
  set ρbar : ℝ := (∑ v ∈ C, ρ v) / C.card with hρbar
  have hρbar01 : 0 ≤ ρbar ∧ ρbar ≤ 1 := by
    constructor
    · exact div_nonneg (Finset.sum_nonneg fun u _ => (hρ01 u).1) hN.le
    · rw [hρbar, div_le_one hN]
      calc ∑ v ∈ C, ρ v ≤ ∑ _v ∈ C, (1 : ℝ) := Finset.sum_le_sum fun u _ => (hρ01 u).2
        _ = C.card := by simp
  have hrbar : r ≤ ρbar := by rw [hρbar, le_div_iff₀ hN]; exact hr
  have hφbar : (∑ v ∈ C, φ v) / C.card = 1 - ρbar + ρbar * Real.exp (-l) := by
    simp only [hφ, hρbar, Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_const,
      nsmul_eq_mul, mul_one, ← Finset.sum_mul]
    field_simp
  have hbase : (∑ v ∈ C, φ v) / C.card ≤ Real.exp (-(l * r) + l ^ 2 / 8) := by
    rw [hφbar]
    refine (bern_mgf_le hρbar01.1 hρbar01.2 l).trans (Real.exp_le_exp.2 ?_)
    nlinarith [mul_le_mul_of_nonneg_left hrbar hl0]
  have hbase0 : 0 ≤ (∑ v ∈ C, φ v) / C.card :=
    div_nonneg (Finset.sum_nonneg fun u _ => hφ0 u) hN.le
  calc ∑ F ∈ C.powersetCard W, μ.real {ω | familyRead O.mq F kl kh w ω ≠ sideRead β}
      ≤ ∑ F ∈ C.powersetCard W, Real.exp (l * ε) * ∏ u ∈ F, φ u := Finset.sum_le_sum hone
    _ = Real.exp (l * ε) * ∑ F ∈ C.powersetCard W, ∏ u ∈ F, φ u := by rw [Finset.mul_sum]
    _ ≤ Real.exp (l * ε) * ((C.card.choose W : ℝ) * ((∑ v ∈ C, φ v) / C.card) ^ W) :=
        mul_le_mul_of_nonneg_left (sum_powersetCard_prod_le C φ (fun u _ => hφ0 u) W)
          (Real.exp_pos _).le
    _ ≤ Real.exp (l * ε) * ((C.card.choose W : ℝ) * Real.exp (-(l * r) + l ^ 2 / 8) ^ W) := by
        gcongr
    _ = C.card.choose W * Real.exp (-2 * W * g ^ 2) := by
        rw [← Real.exp_nat_mul, mul_left_comm, ← Real.exp_add, hε, hl]
        congr 2
        ring

end OrthoDFA
