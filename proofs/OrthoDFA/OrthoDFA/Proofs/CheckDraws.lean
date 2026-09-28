import OrthoDFA.Check
import OrthoDFA.Proofs.Hoeffding

/-!
# The members of a state are independent draws from `reaching`

Given that there are at least `k` of them, the first `k` members of a state are `k` independent
draws from the sampler conditioned on the state.  `listInt` integrates against those draws one
list element at a time, which is all the check's bounds need of their law.
-/

namespace OrthoDFA.CheckProof

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {S : Type*} [Stringlike S]

/-- `F` integrated over a list of `k` independent `ρ`-draws. -/
noncomputable def listInt (ρ : Measure S) : ℕ → (List S → ℝ≥0∞) → ℝ≥0∞
  | 0, F => F []
  | k + 1, F => ∫⁻ a, listInt ρ k (fun l => F (a :: l)) ∂ρ

section listInt
variable (ρ : Measure S)

lemma listInt_mono : ∀ (k : ℕ) {F G : List S → ℝ≥0∞},
    (∀ l : List S, l.length = k → F l ≤ G l) → listInt ρ k F ≤ listInt ρ k G
  | 0, _, _, h => h [] rfl
  | k + 1, _, _, h => lintegral_mono fun a =>
      listInt_mono k fun l hl => h (a :: l) (by simp [hl])

lemma listInt_add : ∀ (k : ℕ) (F G : List S → ℝ≥0∞),
    listInt ρ k (fun l => F l + G l) = listInt ρ k F + listInt ρ k G
  | 0, _, _ => rfl
  | k + 1, F, G => by
      simp only [listInt]
      rw [← lintegral_add_left Measurable.of_discrete]
      exact lintegral_congr fun a => listInt_add k _ _

lemma listInt_const_mul (c : ℝ≥0∞) : ∀ (k : ℕ) (F : List S → ℝ≥0∞),
    listInt ρ k (fun l => c * F l) = c * listInt ρ k F
  | 0, _ => rfl
  | k + 1, F => by
      simp only [listInt]
      rw [← lintegral_const_mul c Measurable.of_discrete]
      exact lintegral_congr fun a => listInt_const_mul c k _

lemma listInt_zero (k : ℕ) : listInt ρ k (fun _ => 0) = 0 := by
  simpa using listInt_const_mul ρ 0 k (fun _ => 0)

lemma listInt_const [IsProbabilityMeasure ρ] (c : ℝ≥0∞) : ∀ k : ℕ, listInt ρ k (fun _ => c) = c
  | 0 => rfl
  | k + 1 => by simp [listInt, listInt_const c k]

lemma listInt_sum {ι : Type*} (s : Finset ι) (k : ℕ) (F : ι → List S → ℝ≥0∞) :
    listInt ρ k (fun l => ∑ i ∈ s, F i l) = ∑ i ∈ s, listInt ρ k (F i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simpa using listInt_zero ρ k
  | insert i s hi ih =>
      simp only [Finset.sum_insert hi]
      rw [listInt_add, ih]

lemma listInt_le_one [IsProbabilityMeasure ρ] (k : ℕ) {F : List S → ℝ≥0∞} (hF : ∀ l, F l ≤ 1) :
    listInt ρ k F ≤ 1 :=
  (listInt_mono ρ k fun l _ => hF l).trans (listInt_const ρ 1 k).le

lemma listInt_exists_le [IsProbabilityMeasure ρ] (P : Set S) [DecidablePred (· ∈ P)] : ∀ k : ℕ,
    listInt ρ k (fun l => if ∃ x ∈ l, x ∈ P then 1 else 0) ≤ k * ρ P
  | 0 => by simp [listInt]
  | k + 1 => by
      simp only [listInt]
      calc ∫⁻ a, listInt ρ k (fun l => if ∃ x ∈ a :: l, x ∈ P then 1 else 0) ∂ρ
          ≤ ∫⁻ a, (P.indicator 1 a + k * ρ P) ∂ρ := lintegral_mono fun a => ?_
        _ = (k + 1 : ℕ) * ρ P := ?_
      · calc listInt ρ k (fun l => if ∃ x ∈ a :: l, x ∈ P then 1 else 0)
            ≤ listInt ρ k (fun l => P.indicator 1 a + if ∃ x ∈ l, x ∈ P then 1 else 0) :=
              listInt_mono ρ k fun l _ => by
                by_cases ha : a ∈ P
                · simp [ha]
                · simp [ha]
          _ = P.indicator 1 a + listInt ρ k (fun l => if ∃ x ∈ l, x ∈ P then 1 else 0) := by
              rw [listInt_add, listInt_const]
          _ ≤ _ := add_le_add_right (listInt_exists_le P k) _
      · rw [lintegral_add_right _ measurable_const,
          lintegral_indicator_one (MeasurableSet.of_discrete)]
        simp only [lintegral_const, measure_univ, mul_one]
        push_cast
        ring

open scoped Classical in
lemma listInt_not_nodup_le [IsProbabilityMeasure ρ] (κ : ℝ≥0∞) (hκ : ∀ a, ρ {a} ≤ κ) :
    ∀ k : ℕ, listInt ρ k (fun l => if l.Nodup then 0 else 1) ≤ k.choose 2 * κ
  | 0 => by simp [listInt]
  | k + 1 => by
      simp only [listInt]
      calc ∫⁻ a, listInt ρ k (fun l => if (a :: l).Nodup then 0 else 1) ∂ρ
          ≤ ∫⁻ _, (k * κ + k.choose 2 * κ) ∂ρ := lintegral_mono fun a => ?_
        _ = (k + 1).choose 2 * κ := ?_
      · calc listInt ρ k (fun l => if (a :: l).Nodup then 0 else 1)
            ≤ listInt ρ k (fun l => (if ∃ x ∈ l, x ∈ ({a} : Set S) then 1 else 0)
                + if l.Nodup then 0 else 1) :=
              listInt_mono ρ k fun l _ => by
                by_cases ha : a ∈ l
                · simp [ha]
                · by_cases hl : l.Nodup <;> simp [ha, hl]
          _ ≤ k * ρ {a} + k.choose 2 * κ := by
              rw [listInt_add]
              exact add_le_add (listInt_exists_le ρ _ k) (listInt_not_nodup_le κ hκ k)
          _ ≤ k * κ + k.choose 2 * κ := by gcongr; exact hκ a
      · rw [lintegral_const, measure_univ, mul_one, Nat.choose_succ_succ', Nat.choose_one_right]
        push_cast
        ring

/-- The product of `g` over consecutive pairs. -/
noncomputable def pairProd (g : S → S → ℝ≥0∞) : List S → ℝ≥0∞
  | a :: b :: l => g a b * pairProd g l
  | _ => 1

omit [Stringlike S] in
lemma pairProd_cons_cons (g : S → S → ℝ≥0∞) (a b : S) (l : List S) :
    pairProd g (a :: b :: l) = g a b * pairProd g l := by
  simp [pairProd]

omit [Stringlike S] in
lemma pairProd_mono {g g' : S → S → ℝ≥0∞} (h : ∀ a b, g a b ≤ g' a b) :
    ∀ l : List S, pairProd g l ≤ pairProd g' l
  | a :: b :: l => by
      rw [pairProd_cons_cons, pairProd_cons_cons]
      exact mul_le_mul' (h a b) (pairProd_mono h l)
  | [] => by simp [pairProd]
  | [_] => by simp [pairProd]

omit [Stringlike S] in
lemma exists_cons_cons {n : ℕ} {l : List S} (hl : l.length = 2 * (n + 1)) :
    ∃ a b l', l = a :: b :: l' ∧ l'.length = 2 * n := by
  match l, hl with
  | a :: b :: l', hl => exact ⟨a, b, l', rfl, by simp at hl; omega⟩

omit [Stringlike S] in
lemma pairProd_le (g : S → S → ℝ≥0∞) (C : ℝ≥0∞) : ∀ (n : ℕ) (l : List S), l.length = 2 * n →
    (∀ a ∈ l, ∀ b ∈ l, g a b ≤ C) → pairProd g l ≤ C ^ n
  | 0, l, hl, _ => by
      obtain rfl : l = [] := List.length_eq_zero_iff.1 (by simpa using hl)
      simp [pairProd]
  | n + 1, l, hl, hg => by
      obtain ⟨a, b, l', rfl, hl'⟩ := exists_cons_cons hl
      rw [pairProd_cons_cons, pow_succ']
      exact mul_le_mul' (hg a (by simp) b (by simp))
        (pairProd_le g C n l' hl' fun x hx y hy => hg x (by simp [hx]) y (by simp [hy]))

lemma listInt_pairProd [IsProbabilityMeasure ρ] (g : S → S → ℝ≥0∞) : ∀ n : ℕ,
    listInt ρ (2 * n) (pairProd g) = (∫⁻ a, ∫⁻ b, g a b ∂ρ ∂ρ) ^ n
  | 0 => by simp [listInt, pairProd]
  | n + 1 => by
      rw [show 2 * (n + 1) = 2 * n + 1 + 1 by ring]
      simp only [listInt, pairProd_cons_cons]
      simp_rw [listInt_const_mul]
      rw [listInt_pairProd g n, pow_succ']
      simp_rw [← lintegral_mul_const _ Measurable.of_discrete]

end listInt

section members
variable {R : Type*} (H : DFA S R) (h : R)

open scoped Classical in
lemma membersOf_cons {N : ℕ} (a : S) (u : Fin N → S) :
    membersOf H h (Fin.cons a u : Fin (N + 1) → S)
      = if H.state a = h then a :: membersOf H h u else membersOf H h u := by
  unfold membersOf
  rw [List.ofFn_succ]
  simp only [Fin.cons_zero, Fin.cons_succ, List.filter_cons]
  split_ifs <;> simp_all

lemma membersOf_zero (u : Fin 0 → S) : membersOf H h u = [] := by
  simp [membersOf]

lemma lintegral_pi_succ (D : Measure S) [IsProbabilityMeasure D] (N : ℕ)
    (G : (Fin (N + 1) → S) → ℝ≥0∞) :
    ∫⁻ u, G u ∂(Measure.pi fun _ : Fin (N + 1) => D)
      = ∫⁻ a, ∫⁻ u, G (Fin.cons a u) ∂(Measure.pi fun _ : Fin N => D) ∂D := by
  have e := (measurePreserving_piFinSuccAbove (fun _ : Fin (N + 1) => D) 0).symm
  rw [← e.lintegral_comp_emb (MeasurableEquiv.measurableEmbedding _),
    lintegral_prod _ Measurable.of_discrete.aemeasurable]
  simp [MeasurableEquiv.piFinSuccAbove_symm_apply]
  rfl

open scoped Classical in
/-- The first `k` members, when there are `k`, are `k` draws from `reaching`. -/
theorem lintegral_members_le (D : Measure S) [IsProbabilityMeasure D]
    (hD : D {w | H.state w = h} ≠ 0) : ∀ (N k : ℕ) (F : List S → ℝ≥0∞),
    ∫⁻ u, (if k ≤ (membersOf H h u).length then F ((membersOf H h u).take k) else 0)
        ∂(Measure.pi fun _ : Fin N => D)
      ≤ listInt (reaching H D h) k F
  | N, 0, F => by simp [listInt]
  | 0, k + 1, F => by simp [membersOf_zero]
  | N + 1, k + 1, F => by
      set Hh : Set S := {w | H.state w = h}
      set ρ := reaching H D h
      have hρ : ρ = D[|Hh] := by simp [ρ, reaching, Hh, hD]
      rw [lintegral_pi_succ]
      calc ∫⁻ a, ∫⁻ u, (if k + 1 ≤ (membersOf H h (Fin.cons a u : Fin (N + 1) → S)).length
              then F ((membersOf H h (Fin.cons a u : Fin (N + 1) → S)).take (k + 1)) else 0)
            ∂(Measure.pi fun _ : Fin N => D) ∂D
          ≤ ∫⁻ a, (Hh.indicator (fun a => listInt ρ k (fun l => F (a :: l))) a
              + Hhᶜ.indicator (fun _ => listInt ρ (k + 1) F) a) ∂D := by
            refine lintegral_mono fun a => ?_
            by_cases ha : H.state a = h
            · have ha' : a ∈ Hh := ha
              simp only [membersOf_cons, if_pos ha, Set.indicator_of_mem ha',
                Set.indicator_of_notMem (Set.notMem_compl_iff.2 ha'), add_zero,
                List.length_cons, add_le_add_iff_right, List.take_succ_cons]
              exact lintegral_members_le D hD N k (fun l => F (a :: l))
            · have ha' : a ∉ Hh := ha
              simp only [membersOf_cons, if_neg ha, Set.indicator_of_notMem ha',
                Set.indicator_of_mem (Set.mem_compl ha'), zero_add]
              exact lintegral_members_le D hD N (k + 1) F
        _ = listInt ρ (k + 1) F := by
            rw [lintegral_add_left Measurable.of_discrete,
              lintegral_indicator MeasurableSet.of_discrete,
              lintegral_indicator MeasurableSet.of_discrete, setLIntegral_const]
            have hcond : ∀ f : S → ℝ≥0∞, ∫⁻ a, f a ∂ρ = (D Hh)⁻¹ * ∫⁻ a in Hh, f a ∂D := by
              intro f
              rw [hρ, ProbabilityTheory.cond, lintegral_smul_measure, smul_eq_mul]
            have hint : ∫⁻ a in Hh, listInt ρ k (fun l => F (a :: l)) ∂D
                = D Hh * listInt ρ (k + 1) F := by
              rw [listInt, hcond, ← mul_assoc, ENNReal.mul_inv_cancel hD (measure_ne_top _ _),
                one_mul]
            rw [hint, mul_comm (listInt ρ (k + 1) F), ← add_mul,
              measure_add_measure_compl MeasurableSet.of_discrete, measure_univ, one_mul]

open scoped Classical in
lemma length_membersOf {N : ℕ} (u : Fin N → S) :
    ((membersOf H h u).length : ℝ) = ∑ i, if H.state (u i) = h then (1 : ℝ) else 0 := by
  induction N with
  | zero => simp [membersOf_zero]
  | succ N ih =>
      have hu : u = Fin.cons (u 0) (Fin.tail u) := (Fin.cons_self_tail u).symm
      rw [hu, membersOf_cons, Fin.sum_univ_succ]
      simp only [Fin.cons_zero, Fin.cons_succ]
      rw [← ih (Fin.tail u)]
      split_ifs <;> simp [add_comm]

open scoped Classical in
/-- Hoeffding for the number of members. -/
theorem members_short_le (D : Measure S) [IsProbabilityMeasure D] (N n : ℕ) (q : ℝ)
    (hq : q ≤ D.real {w | H.state w = h}) (hn : (2 * n : ℝ) ≤ N * q) :
    (Measure.pi fun _ : Fin N => D) {u | (membersOf H h u).length < 2 * n}
      ≤ ENNReal.ofReal (Real.exp (-2 * (N * q - 2 * n) ^ 2 / N)) := by
  set ν := Measure.pi fun _ : Fin N => D
  rcases Nat.eq_zero_or_pos N with rfl | hN
  · simp only [CharP.cast_eq_zero, div_zero, Real.exp_zero, ENNReal.ofReal_one]
    exact prob_le_one
  set X : Fin N → (Fin N → S) → ℝ := fun i u => if H.state (u i) = h then 1 else 0
  have hXm : ∀ i, Measurable (X i) := fun _ => Measurable.of_discrete
  have hindep : iIndepFun X ν :=
    iIndepFun_pi (X := fun _ (a : S) => if H.state a = h then (1 : ℝ) else 0)
      fun _ => Measurable.of_discrete.aemeasurable
  have hIcc : ∀ i, ∀ᵐ u ∂ν, X i u ∈ Set.Icc (0 : ℝ) 1 := fun i =>
    ae_of_all _ fun u => by by_cases hu : H.state (u i) = h <;> simp [X, hu]
  have hmean : ∀ i, ν[X i] = D.real {w | H.state w = h} := by
    intro i
    have hmap : ν.map (fun u => u i) = D :=
      (measurePreserving_eval (fun _ : Fin N => D) i).map_eq
    calc ν[X i] = ∫ a, (if H.state a = h then (1 : ℝ) else 0) ∂(ν.map fun u => u i) := by
          rw [integral_map (measurable_pi_apply i).aemeasurable
            Measurable.of_discrete.aestronglyMeasurable]
      _ = D.real {w | H.state w = h} := by
          rw [hmap, ← integral_indicator_one MeasurableSet.of_discrete]
          rfl
  have hNr : (0 : ℝ) < N := by exact_mod_cast hN
  set γ := q - 2 * n / N
  have hγ : 0 ≤ γ := by
    simp only [γ, sub_nonneg, div_le_iff₀ hNr]
    linarith
  have hmean' : ((Finset.univ : Finset (Fin N)).card : ℝ) * q ≤ ∑ i, ν[X i] := by
    simp only [hmean, Finset.card_univ, Fintype.card_fin, Finset.sum_const, nsmul_eq_mul]
    exact mul_le_mul_of_nonneg_left hq hNr.le
  have hmain := sumLower_le X Finset.univ q γ (fun i => (hXm i).aemeasurable) hindep hIcc
    hmean' hγ
  simp only [Finset.card_univ, Fintype.card_fin] at hmain
  have hsub : {u : Fin N → S | (membersOf H h u).length < 2 * n}
      ⊆ {u | ∑ i, X i u ≤ (N : ℝ) * (q - γ)} := by
    intro u hu
    simp only [Set.mem_ofPred_eq] at hu ⊢
    have h1 : (N : ℝ) * (q - γ) = 2 * n := by
      simp only [γ]; field_simp; ring
    rw [h1, ← length_membersOf]
    exact_mod_cast hu.le
  calc ν {u | (membersOf H h u).length < 2 * n}
      ≤ ν {u | ∑ i, X i u ≤ (N : ℝ) * (q - γ)} := measure_mono hsub
    _ = ENNReal.ofReal (ν.real {u | ∑ i, X i u ≤ (N : ℝ) * (q - γ)}) :=
        (ENNReal.ofReal_toReal (measure_ne_top _ _)).symm
    _ ≤ ENNReal.ofReal (Real.exp (-2 * N * γ ^ 2)) := ENNReal.ofReal_le_ofReal hmain
    _ = ENNReal.ofReal (Real.exp (-2 * (N * q - 2 * n) ^ 2 / N)) := by
        congr 2
        simp only [γ]
        field_simp

end members

end OrthoDFA.CheckProof
