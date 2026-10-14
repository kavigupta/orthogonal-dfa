import OrthoDFA.Level3Claim
import OrthoDFA.Proofs.IdealRound
import OrthoDFA.Proofs.SeqTest

/-!
# Counts over fresh draws

`binE n p f` is `E[f(Bin(n, p))]`; conditioning on one trial gives its recursion, and from it
its monotonicity in `p`. Over `N` fresh draws, how many of the first `n` fall in `B`, and how
many of those in `A ⊆ B`, are `Bin(n, D(B))` and, given the first, `Bin(·, D(A)/D(B))`. A sum over
the draws whose exponentiated step averages at most `1` reaches a level at any look with chance at
most its exponential (Ville), which bounds a sequential test's false firing over a whole stretch.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory Finset

/-! ## Binomial expectations -/

section Binomial

theorem binE_zero (p : ℝ) (f : ℕ → ℝ) : binE 0 p f = f 0 := by simp [binE]

theorem binE_succ (n : ℕ) (p : ℝ) (f : ℕ → ℝ) :
    binE (n + 1) p f = (1 - p) * binE n p f + p * binE n p (fun j => f (j + 1)) := by
  unfold binE
  rw [sum_range_succ' _ (n + 1)]
  simp only [Nat.choose_succ_succ, Nat.cast_add, add_mul, sum_add_distrib, Nat.choose_zero_right,
    pow_zero, Nat.cast_one, one_mul, Nat.sub_zero]
  have h1 : ∑ j ∈ range (n + 1), (n.choose (j + 1) : ℝ) * p ^ (j + 1) * (1 - p) ^ (n + 1 - (j + 1))
        * f (j + 1) + (1 - p) ^ (n + 1) * f 0
      = (1 - p) * ∑ j ∈ range (n + 1), (n.choose j : ℝ) * p ^ j * (1 - p) ^ (n - j) * f j := by
    have := sum_range_succ' (fun j => (n.choose j : ℝ) * p ^ j * (1 - p) ^ (n + 1 - j) * f j)
      (n + 1)
    simp only [Nat.choose_zero_right, pow_zero, Nat.cast_one, one_mul, Nat.sub_zero] at this
    rw [← this, sum_range_succ, Nat.choose_succ_self, Nat.cast_zero, zero_mul, zero_mul, zero_mul,
      add_zero, mul_sum]
    refine sum_congr rfl fun j hj => ?_
    rw [show n + 1 - j = n - j + 1 by have := mem_range.1 hj; omega, pow_succ]
    ring
  have h2 : ∑ j ∈ range (n + 1), (n.choose j : ℝ) * p ^ (j + 1) * (1 - p) ^ (n + 1 - (j + 1))
        * f (j + 1)
      = p * ∑ j ∈ range (n + 1), (n.choose j : ℝ) * p ^ j * (1 - p) ^ (n - j) * f (j + 1) := by
    rw [mul_sum]
    refine sum_congr rfl fun j _ => ?_
    rw [show n + 1 - (j + 1) = n - j by omega, pow_succ]
    ring
  rw [← h1, ← h2]
  ring

theorem binE_lin (n : ℕ) (p c₁ c₂ : ℝ) (f g : ℕ → ℝ) :
    binE n p (fun j => c₁ * f j + c₂ * g j) = c₁ * binE n p f + c₂ * binE n p g := by
  simp only [binE, mul_add, sum_add_distrib, mul_sum]
  congr 1 <;> refine sum_congr rfl fun j _ => ?_ <;> ring

theorem binE_const (n : ℕ) (p c : ℝ) : binE n p (fun _ => c) = c := by
  induction n with
  | zero => simp [binE]
  | succ n ih => rw [binE_succ, ih]; ring

theorem binE_congr {n : ℕ} {p : ℝ} {f g : ℕ → ℝ} (h : ∀ j ≤ n, f j = g j) :
    binE n p f = binE n p g :=
  sum_congr rfl fun j hj => by rw [h j (Nat.lt_succ_iff.1 (mem_range.1 hj))]

variable {p : ℝ}

theorem binE_mono (hp0 : 0 ≤ p) (hp1 : p ≤ 1) {n : ℕ} {f g : ℕ → ℝ} (h : ∀ j, f j ≤ g j) :
    binE n p f ≤ binE n p g := by
  refine sum_le_sum fun j _ => ?_
  have : 0 ≤ (n.choose j : ℝ) * p ^ j * (1 - p) ^ (n - j) := by
    have : 0 ≤ 1 - p := by linarith
    positivity
  exact mul_le_mul_of_nonneg_left (h j) this

theorem binE_mono_le (hp0 : 0 ≤ p) (hp1 : p ≤ 1) {n : ℕ} {f g : ℕ → ℝ} (h : ∀ j ≤ n, f j ≤ g j) :
    binE n p f ≤ binE n p g := by
  refine sum_le_sum fun j hj => ?_
  have : 0 ≤ (n.choose j : ℝ) * p ^ j * (1 - p) ^ (n - j) := by
    have : 0 ≤ 1 - p := by linarith
    positivity
  exact mul_le_mul_of_nonneg_left (h j (Nat.lt_succ_iff.1 (mem_range.1 hj))) this

theorem binE_nonneg (hp0 : 0 ≤ p) (hp1 : p ≤ 1) {n : ℕ} {f : ℕ → ℝ} (h : ∀ j, 0 ≤ f j) :
    0 ≤ binE n p f := by
  have := binE_mono hp0 hp1 (n := n) (f := fun _ => 0) h
  rwa [binE_const] at this

theorem binE_le_one (hp0 : 0 ≤ p) (hp1 : p ≤ 1) {n : ℕ} {f : ℕ → ℝ} (h : ∀ j, f j ≤ 1) :
    binE n p f ≤ 1 := by
  have := binE_mono hp0 hp1 (n := n) (g := fun _ => 1) h
  rwa [binE_const] at this

theorem binE_succ' (n : ℕ) (p : ℝ) (f : ℕ → ℝ) :
    binE (n + 1) p f = binE n p (fun j => (1 - p) * f j + p * f (j + 1)) := by
  rw [binE_succ, binE_lin]

/-- A monotone function of `Bin(n, p)` grows with `p`. -/
theorem binE_le_of_le {p' : ℝ} (hp0 : 0 ≤ p) (hpp : p ≤ p') (hp1 : p' ≤ 1) :
    ∀ (n : ℕ) (f : ℕ → ℝ), Monotone f → binE n p f ≤ binE n p' f := by
  intro n
  induction n with
  | zero => intro f _; simp [binE_zero]
  | succ n ih =>
    intro f hf
    rw [binE_succ', binE_succ']
    have hp'0 : 0 ≤ p' := hp0.trans hpp
    calc _ ≤ binE n p (fun j => (1 - p') * f j + p' * f (j + 1)) :=
          binE_mono hp0 (hpp.trans hp1) fun j => by
            have := hf (Nat.le_succ j)
            nlinarith
      _ ≤ _ := ih _ fun i j hij => by
          have h1 := hf hij
          have h2 := hf (Nat.succ_le_succ hij)
          have : 0 ≤ 1 - p' := by linarith
          nlinarith

/-- A function of `Bin(n, p)` monotone up to `n` grows with `p`. -/
theorem binE_le_of_le_on {p' : ℝ} (hp0 : 0 ≤ p) (hpp : p ≤ p') (hp1 : p' ≤ 1) (n : ℕ)
    (f : ℕ → ℝ) (hf : ∀ i j, i ≤ j → j ≤ n → f i ≤ f j) : binE n p f ≤ binE n p' f := by
  have e : ∀ q, binE n q f = binE n q (fun j => f (min j n)) :=
    fun q => binE_congr fun j hj => by rw [min_eq_left hj]
  rw [e, e]
  exact binE_le_of_le hp0 hpp hp1 n _ fun i j h => hf _ _ (min_le_min_right n h) (min_le_right _ _)

/-- An antitone function of `Bin(n, p)` falls with `p`. -/
theorem binE_le_of_le_anti {p' : ℝ} (hp0 : 0 ≤ p) (hpp : p ≤ p') (hp1 : p' ≤ 1) (n : ℕ) (f : ℕ → ℝ)
    (hf : Antitone f) : binE n p' f ≤ binE n p f := by
  have := binE_le_of_le hp0 hpp hp1 n (fun j => -f j) fun i j h => neg_le_neg (hf h)
  have e : ∀ q, binE n q (fun j => -f j) = -binE n q f := fun q => by
    have := binE_lin n q (-1) 0 f f
    simp only [neg_one_mul, zero_mul, add_zero] at this
    exact this
  rw [e, e] at this
  linarith

/-- A function of `Bin(n, p)` antitone up to `n` falls with `p`. -/
theorem binE_le_of_le_anti_on {p' : ℝ} (hp0 : 0 ≤ p) (hpp : p ≤ p') (hp1 : p' ≤ 1) (n : ℕ)
    (f : ℕ → ℝ) (hf : ∀ i j, i ≤ j → j ≤ n → f j ≤ f i) : binE n p' f ≤ binE n p f := by
  have e : ∀ q, binE n q f = binE n q (fun j => f (min j n)) :=
    fun q => binE_congr fun j hj => by rw [min_eq_left hj]
  rw [e, e]
  exact binE_le_of_le_anti hp0 hpp hp1 n _ fun i j h =>
    hf _ _ (min_le_min_right n h) (min_le_right _ _)

theorem binomSfGe_eq (n : ℕ) (p : ℝ) (j : ℕ) :
    binomSfGe n p j = binE n p (fun i => if j ≤ i then 1 else 0) := by
  unfold binomSfGe binE
  simp only [mul_ite, mul_one, mul_zero]
  rw [← sum_filter]
  congr 1
  ext i
  simp only [mem_Icc, mem_filter, mem_range]
  omega

variable (hp0 : 0 ≤ p) (hp1 : p ≤ 1)
include hp0 hp1

theorem binomSfGe_nonneg (n j : ℕ) : 0 ≤ binomSfGe n p j := by
  rw [binomSfGe_eq]
  exact binE_nonneg hp0 hp1 fun i => by split_ifs <;> norm_num

theorem binomSfGe_le_one (n j : ℕ) : binomSfGe n p j ≤ 1 := by
  rw [binomSfGe_eq]
  exact binE_le_one hp0 hp1 fun i => by split_ifs <;> norm_num

theorem binomSfGe_succ_n (n j : ℕ) : binomSfGe n p j ≤ binomSfGe (n + 1) p j := by
  rw [binomSfGe_eq, binomSfGe_eq, binE_succ']
  exact binE_mono hp0 hp1 fun i => by
    split_ifs <;> first | omega | nlinarith

omit hp0 hp1 in
theorem binomSfGe_mono_p {p' : ℝ} (hp0 : 0 ≤ p) (hpp : p ≤ p') (hp1 : p' ≤ 1) (n j : ℕ) :
    binomSfGe n p j ≤ binomSfGe n p' j := by
  rw [binomSfGe_eq, binomSfGe_eq]
  exact binE_le_of_le hp0 hpp hp1 n _ fun i i' h => by
    split_ifs <;> first | omega | norm_num

/-- `P(Bin(n, p) ≤ j)`. -/
theorem one_sub_binomSfGe (n j : ℕ) :
    1 - binomSfGe n p (j + 1) = binE n p (fun i => if i ≤ j then 1 else 0) := by
  rw [binomSfGe_eq]
  have := binE_lin n p 1 (-1) (fun _ => 1) (fun i => if j + 1 ≤ i then 1 else 0)
  rw [binE_const] at this
  rw [show (1 : ℝ) - _ = 1 * 1 + -1 * binE n p (fun i => if j + 1 ≤ i then 1 else 0) by ring,
    ← this]
  exact binE_congr fun i _ => by split_ifs <;> first | omega | norm_num

theorem one_sub_binomSfGe' {n c : ℕ} (hc : 1 ≤ c) :
    1 - binomSfGe n p c = binE n p (fun i => if i < c then 1 else 0) := by
  rw [show c = c - 1 + 1 by omega, one_sub_binomSfGe hp0 hp1]
  exact binE_congr fun i _ => by split_ifs <;> first | rfl | omega

theorem one_sub_binomSfGe_lt (n c : ℕ) :
    1 - binomSfGe n p c = binE n p (fun i => if i < c then 1 else 0) := by
  rcases Nat.eq_zero_or_pos c with rfl | hc
  · rw [binomSfGe_eq]
    simp only [zero_le, if_true, Nat.not_lt_zero, if_false, binE_const, sub_self]
  · exact one_sub_binomSfGe' hp0 hp1 hc

end Binomial

/-- More hits lower the ratio where `p₁ ≤ p₀`. -/
theorem likelihoodRatio_anti {p₀ p₁ : ℝ} (hp₀ : 0 < p₀) (hp₁ : 0 ≤ p₁) (hp₁0 : p₁ ≤ p₀)
    (hp₀1 : p₀ < 1) {n h h' : ℕ} (hh : h ≤ h') (hh' : h' ≤ n) :
    likelihoodRatio p₀ p₁ n h' ≤ likelihoodRatio p₀ p₁ n h := by
  induction hh with
  | refl => exact le_rfl
  | @step k hk ih =>
    refine le_trans ?_ (ih (by omega))
    unfold likelihoodRatio
    rw [show n - k = n - (k + 1) + 1 by omega, pow_succ, pow_succ]
    have hA0 : 0 ≤ p₁ / p₀ := by positivity
    have hA : p₁ / p₀ ≤ 1 := by rw [div_le_one hp₀]; linarith
    have hB : 1 ≤ (1 - p₁) / (1 - p₀) := by rw [le_div_iff₀ (by linarith)]; linarith
    have hX : 0 ≤ (p₁ / p₀) ^ k * ((1 - p₁) / (1 - p₀)) ^ (n - (k + 1)) := by
      have : 0 ≤ (1 - p₁) / (1 - p₀) := by linarith
      positivity
    nlinarith

theorem stretchRisk_nonneg (C : Level3Settings) (nσ nα L Ns N₁ : ℕ) (θr εd' θpt' : ℝ)
    (ha : 0 ≤ C.testLevel) (hθr0 : 0 ≤ θr) (hθr1 : θr ≤ 1) (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1)
    (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1) :
    0 ≤ stretchRisk C nσ nα L Ns N₁ θr εd' θpt' := by
  classical
  unfold stretchRisk
  have h01 : ∀ (j : ℕ) (P : Prop) [Decidable P], (0 : ℝ) ≤ if P then 0 else 1 := fun _ P _ => by
    split_ifs <;> norm_num
  have := binomSfGe_le_one hθr0 hθr1 Ns 1
  have := binomSfGe_le_one hεd'0 hεd'1 Ns ⌈C.searchShare * Ns⌉₊
  have := binE_nonneg hεd'0 hεd'1 (n := N₁)
    (f := fun s => if seqRate C.disagreementRate C.disagreementAlternative C.testLevel N₁ s then 0
      else 1) fun j => h01 j _
  have : 0 ≤ ∑ s ∈ Icc ⌈C.searchShare * Ns⌉₊ Ns,
      binE s θpt' (fun j => if seqRate C.middleRate C.middleAlternative C.testLevel s j then 0
        else 1) :=
    sum_nonneg fun s _ => binE_nonneg hθpt'0 hθpt'1 fun j => h01 j _
  have : (0 : ℝ) ≤ ((nσ + 2) * nα : ℕ) * Real.exp (-C.edgeExcess / (2 * (L + 1))) := by positivity
  linarith

/-! ## Counts over fresh draws -/

/-- How many of the first `n` draws satisfy `f`. -/
def cnt {X : Type*} (f : X → Bool) (n : ℕ) (xs : List X) : ℕ := (xs.take n).countP f

theorem cnt_zero {X : Type*} (f : X → Bool) (xs : List X) : cnt f 0 xs = 0 := by simp [cnt]

theorem cnt_cons {X : Type*} (f : X → Bool) (n : ℕ) (x : X) (xs : List X) :
    cnt f (n + 1) (x :: xs) = (if f x then 1 else 0) + cnt f n xs := by
  simp only [cnt, List.take_succ_cons, List.countP_cons]
  cases f x <;> simp [add_comm]

section Draws

variable {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
  (D : Measure X) [IsProbabilityMeasure D]

/-- The draws `x :: xs` with `x ∈ S` that fall in `E` are those whose rest falls in `F`. -/
theorem pi_cons_inter (N : ℕ) (E : Set (Fin (N + 1) → X)) (S : Set X) (F : Set (Fin N → X))
    (h : ∀ x ∈ S, ∀ ys, Fin.cons x ys ∈ E ↔ ys ∈ F) :
    (Measure.pi fun _ : Fin (N + 1) => D) (E ∩ {xs | xs 0 ∈ S})
      = D S * (Measure.pi fun _ : Fin N => D) F := by
  rw [Ideal.pi_succ_apply]
  have : ∀ x, (Measure.pi fun _ : Fin N => D) {ys | Fin.cons x ys ∈ E ∩ {xs | xs 0 ∈ S}}
      = S.indicator (fun _ => (Measure.pi fun _ : Fin N => D) F) x := by
    intro x
    by_cases hx : x ∈ S
    · rw [Set.indicator_of_mem hx]
      congr 1
      ext ys
      simp only [Set.mem_inter_iff, Set.mem_ofPred_eq, Fin.cons_zero, hx, and_true]
      exact h x hx ys
    · rw [Set.indicator_of_notMem hx]
      convert measure_empty (μ := Measure.pi fun _ : Fin N => D)
      ext ys
      simp [hx]
  simp_rw [this]
  rw [lintegral_indicator_const (Set.to_countable S).measurableSet, mul_comm]

/-- How many of the first `n` of `N` fresh draws fall in `B`, and how many of those in `A`. -/
theorem pi_cnt2 (fB fA : X → Bool) (hAB : ∀ x, fA x = true → fB x = true) :
    ∀ (N n : ℕ), n ≤ N → ∀ Q : ℕ → ℕ → Prop, ∀ [DecidableRel Q],
      (Measure.pi fun _ : Fin N => D).real
          {xs | Q (cnt fB n (List.ofFn xs)) (cnt fA n (List.ofFn xs))}
        = binE n (D.real {x | fB x}) (fun s => binE s (D.real {x | fA x} / D.real {x | fB x})
            (fun j => if Q s j then 1 else 0)) := by
  set q := D.real {x | fB x}
  set r := D.real {x | fA x}
  have hrq : r ≤ q := measureReal_mono fun x hx => hAB x hx
  have hr0 : 0 ≤ r := measureReal_nonneg
  have h0 : ∀ N (Q : ℕ → ℕ → Prop) [DecidableRel Q],
      (Measure.pi fun _ : Fin N => D).real
          {xs | Q (cnt fB 0 (List.ofFn xs)) (cnt fA 0 (List.ofFn xs))}
        = binE 0 q (fun s => binE s (r / q) (fun j => if Q s j then 1 else 0)) := by
    intro N Q _
    simp only [cnt_zero, binE_zero]
    by_cases hQ : Q 0 0
    · simp [hQ]
    · simp [hQ]
  intro N
  induction N with
  | zero =>
    intro n hn Q _
    obtain rfl : n = 0 := by omega
    exact h0 0 Q
  | succ N ih =>
    intro n hn Q _
    rcases n with _ | n
    · exact h0 _ Q
    have hS : ∀ (S : Set X) (b a : ℕ), (∀ x ∈ S, (if fB x then 1 else 0) = b ∧
        (if fA x then 1 else 0) = a) →
        (Measure.pi fun _ : Fin (N + 1) => D)
          ({xs | Q (cnt fB (n + 1) (List.ofFn xs)) (cnt fA (n + 1) (List.ofFn xs))}
            ∩ {xs | xs 0 ∈ S})
          = D S * (Measure.pi fun _ : Fin N => D)
            {ys | Q (b + cnt fB n (List.ofFn ys)) (a + cnt fA n (List.ofFn ys))} := by
      intro S b a hba
      refine pi_cons_inter D N _ S _ fun x hx ys => ?_
      simp only [Set.mem_ofPred_eq, List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, cnt_cons,
        (hba x hx).1, (hba x hx).2]
    set E := {xs : Fin (N + 1) → X |
      Q (cnt fB (n + 1) (List.ofFn xs)) (cnt fA (n + 1) (List.ofFn xs))}
    set SA := {x | fA x = true}
    set SBA := {x | fB x = true ∧ fA x = false}
    set SB' := {x | fB x = false}
    have hcov : E = (E ∩ {xs | xs 0 ∈ SA}) ∪ (E ∩ {xs | xs 0 ∈ SBA}) ∪ (E ∩ {xs | xs 0 ∈ SB'}) := by
      ext xs
      simp only [Set.mem_union, Set.mem_inter_iff, Set.mem_ofPred_eq, SA, SBA, SB']
      constructor
      · intro h
        cases hb : fB (xs 0) <;> cases ha : fA (xs 0) <;> simp_all
      · rintro ((⟨h, -⟩ | ⟨h, -⟩) | ⟨h, -⟩) <;> exact h
    have hm : ∀ T : Set (Fin (N + 1) → X), MeasurableSet T :=
      fun T => (Set.to_countable T).measurableSet
    have hd1 : Disjoint (E ∩ {xs | xs 0 ∈ SA}) (E ∩ {xs | xs 0 ∈ SBA}) :=
      Set.disjoint_left.2 fun xs h1 h2 => by
        simp only [Set.mem_inter_iff, Set.mem_ofPred_eq, SA, SBA] at h1 h2
        rw [h1.2] at h2; exact absurd h2.2.2 (by simp)
    have hd2 : Disjoint ((E ∩ {xs | xs 0 ∈ SA}) ∪ (E ∩ {xs | xs 0 ∈ SBA}))
        (E ∩ {xs | xs 0 ∈ SB'}) :=
      Set.disjoint_left.2 fun xs h1 h2 => by
        simp only [Set.mem_union, Set.mem_inter_iff, Set.mem_ofPred_eq, SA, SBA, SB'] at h1 h2
        rcases h1 with ⟨-, h1⟩ | ⟨-, h1, -⟩
        · rw [hAB _ h1] at h2; exact absurd h2.2 (by simp)
        · rw [h1] at h2; exact absurd h2.2 (by simp)
    have eA := hS SA 1 1 fun x hx => by
      simp only [SA, Set.mem_ofPred_eq] at hx; simp [hx, hAB x hx]
    have eBA := hS SBA 1 0 fun x hx => by
      simp only [SBA, Set.mem_ofPred_eq] at hx; simp [hx.1, hx.2]
    have eB' := hS SB' 0 0 fun x hx => by
      simp only [SB', Set.mem_ofPred_eq] at hx
      have : fA x = false := by
        cases h : fA x
        · rfl
        · rw [hAB x h] at hx; exact absurd hx (by simp)
      simp [hx, this]
    have ihA := ih n (by omega) (fun s j => Q (1 + s) (1 + j))
    have ihBA := ih n (by omega) (fun s j => Q (1 + s) (0 + j))
    have ihB' := ih n (by omega) (fun s j => Q (0 + s) (0 + j))
    have hDA : D.real SA = r := rfl
    have hDBA : D.real SBA = q - r := by
      have : {x | fB x = true} = SBA ∪ SA := by
        ext x; simp only [Set.mem_union, Set.mem_ofPred_eq, SBA, SA]
        constructor
        · intro h
          by_cases ha : fA x = true
          · exact Or.inr ha
          · exact Or.inl ⟨h, by simpa using ha⟩
        · rintro (h | h)
          · exact h.1
          · exact hAB x h
      have hq : q = D.real (SBA ∪ SA) := by rw [← this]
      rw [hq, measureReal_union (Set.disjoint_left.2 fun x h1 h2 => by
        simp only [SBA, SA, Set.mem_ofPred_eq] at h1 h2; rw [h1.2] at h2; exact absurd h2 (by simp))
        (Set.to_countable _).measurableSet]
      ring
    have hDB' : D.real SB' = 1 - q := by
      have : SB' = {x | fB x = true}ᶜ := by ext x; simp [SB']
      rw [this, measureReal_compl (Set.to_countable _).measurableSet, probReal_univ]
    have hreal : (Measure.pi fun _ : Fin (N + 1) => D).real E
        = r * binE n q (fun s => binE s (r / q) (fun j => if Q (1 + s) (1 + j) then 1 else 0))
          + (q - r) * binE n q (fun s => binE s (r / q)
              (fun j => if Q (1 + s) (0 + j) then 1 else 0))
          + (1 - q) * binE n q (fun s => binE s (r / q)
              (fun j => if Q (0 + s) (0 + j) then 1 else 0)) := by
      rw [hcov, measureReal_union hd2 (hm _), measureReal_union hd1 (hm _), measureReal_def,
        measureReal_def, measureReal_def, eA, eBA, eB', ENNReal.toReal_mul, ENNReal.toReal_mul,
        ENNReal.toReal_mul, ← measureReal_def, ← measureReal_def, ← measureReal_def,
        ← measureReal_def, ← measureReal_def, ← measureReal_def, hDA, hDBA, hDB', ihA, ihBA, ihB']
    rw [hreal, binE_succ]
    have hF : ∀ s, binE (s + 1) (r / q) (fun j => if Q (s + 1) j then 1 else 0) * q
        = (q - r) * binE s (r / q) (fun j => if Q (1 + s) (0 + j) then 1 else 0)
          + r * binE s (r / q) (fun j => if Q (1 + s) (1 + j) then 1 else 0) := by
      intro s
      rw [binE_succ]
      simp only [zero_add, add_comm 1 s, add_comm 1]
      by_cases hq0 : q = 0
      · have : r = 0 := le_antisymm (hq0 ▸ hrq) hr0
        simp [hq0, this]
      · field_simp
    have hlin : q * binE n q (fun s => binE (s + 1) (r / q) (fun j => if Q (s + 1) j then 1 else 0))
        = (q - r) * binE n q (fun s => binE s (r / q)
            (fun j => if Q (1 + s) (0 + j) then 1 else 0))
          + r * binE n q (fun s => binE s (r / q)
            (fun j => if Q (1 + s) (1 + j) then 1 else 0)) := by
      rw [← binE_lin]
      have := binE_lin n q q 0
        (fun s => binE (s + 1) (r / q) (fun j => if Q (s + 1) j then 1 else 0))
        (fun _ => 0)
      simp only [zero_mul, add_zero] at this
      rw [← this]
      exact binE_congr fun s _ => by rw [mul_comm, hF]
    simp only [zero_add] at hlin ⊢
    rw [show q * binE n q (fun j => binE (j + 1) (r / q) (fun j_1 => if Q (j + 1) j_1 then 1
        else 0)) = _ from hlin]
    simp only [add_comm 1]
    ring

/-- How many of the first `n` of `N` fresh draws fall in `B`. -/
theorem pi_cnt (fB : X → Bool) (N n : ℕ) (hn : n ≤ N) (Q : ℕ → Prop) [DecidablePred Q] :
    (Measure.pi fun _ : Fin N => D).real {xs | Q (cnt fB n (List.ofFn xs))}
      = binE n (D.real {x | fB x}) (fun s => if Q s then 1 else 0) := by
  have := pi_cnt2 D fB fB (fun _ h => h) N n hn (fun s _ => Q s)
  rw [this]
  exact binE_congr fun s _ => binE_const _ _ _

/-- The sum of `g` over the first `n` draws. -/
def sumFirst (g : X → ℝ) (n : ℕ) (xs : List X) : ℝ := ((xs.take n).map g).sum

/-- A count at most `B` whose mean is at most `η` passes twice `η`, at scale `2B`, with an
exponentiated step averaging at most `1`. -/
theorem mgf_count (U : X → ℕ) {B η : ℝ} (hB : 0 < B) (hb : ∀ᵐ x ∂D, (U x : ℝ) ≤ B)
    (hmean : ∫ x, (U x : ℝ) ∂D ≤ η) :
    ∫⁻ x, ENNReal.ofReal (Real.exp (((U x : ℝ) - 2 * η) / (2 * B))) ∂D ≤ 1 := by
  set r := ∫ x, (U x : ℝ) ∂D
  have hUi : Integrable (fun x => (U x : ℝ)) D :=
    Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable B (by
      filter_upwards [hb] with x hx
      rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]; exact hx)
  have hr0 : 0 ≤ r := integral_nonneg fun _ => Nat.cast_nonneg _
  have hpt : ∀ᵐ x ∂D, Real.exp ((U x : ℝ) / (2 * B)) ≤ 1 + 3 * (U x : ℝ) / (4 * B) := by
    filter_upwards [hb] with x hx
    have hU0 : (0 : ℝ) ≤ U x := Nat.cast_nonneg _
    set t := (U x : ℝ) / (2 * B) with ht
    have ht0 : 0 ≤ t := by positivity
    have ht1 : t ≤ 1 / 2 := by rw [ht, div_le_iff₀ (by positivity)]; linarith
    have habs : |t| ≤ 1 := by rw [abs_of_nonneg ht0]; linarith
    have h1 := (abs_le.1 (Real.abs_exp_sub_one_sub_id_le habs)).2
    have h3 : 3 * (U x : ℝ) / (4 * B) = 3 / 2 * t := by rw [ht]; field_simp; ring
    rw [h3]
    nlinarith
  have hsplit : ∀ x, ENNReal.ofReal (Real.exp (((U x : ℝ) - 2 * η) / (2 * B)))
      = ENNReal.ofReal (Real.exp (-η / B))
        * ENNReal.ofReal (Real.exp ((U x : ℝ) / (2 * B))) := by
    intro x
    rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
    congr 2
    field_simp
    ring
  simp_rw [hsplit]
  rw [lintegral_const_mul _ (measurable_of_countable _)]
  have hgi : Integrable (fun x => 1 + 3 * (U x : ℝ) / (4 * B)) D :=
    (integrable_const 1).add ((hUi.const_mul 3).div_const _)
  have hint : ∫ x, (1 + 3 * (U x : ℝ) / (4 * B)) ∂D = 1 + 3 * r / (4 * B) := by
    rw [integral_add (integrable_const 1) ((hUi.const_mul 3).div_const _), integral_const,
      integral_div, integral_const_mul]
    simp [r]
  calc ENNReal.ofReal (Real.exp (-η / B))
        * ∫⁻ x, ENNReal.ofReal (Real.exp ((U x : ℝ) / (2 * B))) ∂D
      ≤ ENNReal.ofReal (Real.exp (-η / B))
        * ∫⁻ x, ENNReal.ofReal (1 + 3 * (U x : ℝ) / (4 * B)) ∂D :=
        mul_le_mul' le_rfl (lintegral_mono_ae (by
          filter_upwards [hpt] with x hx using ENNReal.ofReal_le_ofReal hx))
    _ = ENNReal.ofReal (Real.exp (-η / B) * (1 + 3 * r / (4 * B))) := by
        rw [← ofReal_integral_eq_lintegral_ofReal hgi (ae_of_all _ fun x => by positivity), hint,
          ENNReal.ofReal_mul (Real.exp_pos _).le]
    _ ≤ 1 := by
        rw [← ENNReal.ofReal_one]
        refine ENNReal.ofReal_le_ofReal ?_
        have h1 := Real.add_one_le_exp (3 * r / (4 * B))
        have h2 : Real.exp (3 * r / (4 * B)) ≤ Real.exp (η / B) :=
          Real.exp_le_exp.2 (by rw [div_le_div_iff₀ (by positivity) hB]; nlinarith)
        have h3 : Real.exp (-η / B) * Real.exp (η / B) = 1 := by
          rw [← Real.exp_add]; simp [neg_div]
        have h4 := (Real.exp_pos (-η / B)).le
        nlinarith


/-- Ville's inequality: over fresh draws, a sum whose exponentiated step, at scale `K`, averages at
most `1` reaches `a` at some look with chance at most `exp(−a/K)`. -/
theorem pi_sum_any (g : X → ℝ) {K : ℝ} (hK : 0 < K)
    (hg : ∫⁻ x, ENNReal.ofReal (Real.exp (g x / K)) ∂D ≤ 1) :
    ∀ (N : ℕ) (a : ℝ), (Measure.pi fun _ : Fin N => D)
        {xs | ∃ n ≤ N, a ≤ sumFirst g n (List.ofFn xs)} ≤ ENNReal.ofReal (Real.exp (-a / K)) := by
  have hle : ∀ (N : ℕ) (a : ℝ) (E : Set (Fin N → X)), a ≤ 0 →
      (Measure.pi fun _ : Fin N => D) E ≤ ENNReal.ofReal (Real.exp (-a / K)) := by
    intro N a E ha
    refine prob_le_one.trans ?_
    rw [← ENNReal.ofReal_one]
    exact ENNReal.ofReal_le_ofReal (Real.one_le_exp (div_nonneg (by linarith) hK.le))
  intro N
  induction N with
  | zero =>
    intro a
    by_cases ha : a ≤ 0
    · exact hle 0 a _ ha
    · have : {xs : Fin 0 → X | ∃ n ≤ 0, a ≤ sumFirst g n (List.ofFn xs)} = ∅ := by
        ext xs
        simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
        intro n hn
        obtain rfl : n = 0 := by omega
        simp only [sumFirst, List.take_zero, List.map_nil, List.sum_nil]
        linarith
      rw [this, measure_empty]
      exact bot_le
  | succ N ih =>
    intro a
    by_cases ha : a ≤ 0
    · exact hle (N + 1) a _ ha
    rw [Ideal.pi_succ_apply]
    have hsec : ∀ x, (Measure.pi fun _ : Fin N => D) {ys | Fin.cons x ys ∈
        {xs : Fin (N + 1) → X | ∃ n ≤ N + 1, a ≤ sumFirst g n (List.ofFn xs)}}
        ≤ ENNReal.ofReal (Real.exp (-a / K)) * ENNReal.ofReal (Real.exp (g x / K)) := by
      intro x
      have hsub : {ys : Fin N → X | Fin.cons x ys ∈
          {xs : Fin (N + 1) → X | ∃ n ≤ N + 1, a ≤ sumFirst g n (List.ofFn xs)}}
          ⊆ {ys | ∃ m ≤ N, a - g x ≤ sumFirst g m (List.ofFn ys)} := by
        rintro ys ⟨n, hn, h⟩
        rcases n with _ | m
        · simp only [sumFirst, List.take_zero, List.map_nil, List.sum_nil] at h
          exact absurd h (by linarith)
        · refine ⟨m, by omega, ?_⟩
          simp only [List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, sumFirst, List.take_succ_cons,
            List.map_cons, List.sum_cons] at h
          simp only [sumFirst]
          linarith
      refine (measure_mono hsub).trans ((ih (a - g x)).trans (le_of_eq ?_))
      rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
      congr 2
      ring
    calc _ ≤ ∫⁻ x, ENNReal.ofReal (Real.exp (-a / K)) * ENNReal.ofReal (Real.exp (g x / K)) ∂D :=
          lintegral_mono hsec
      _ = ENNReal.ofReal (Real.exp (-a / K)) * ∫⁻ x, ENNReal.ofReal (Real.exp (g x / K)) ∂D :=
          lintegral_const_mul _ (measurable_of_countable _)
      _ ≤ ENNReal.ofReal (Real.exp (-a / K)) * 1 := by gcongr
      _ = _ := mul_one _

/-- A draw's value, `u` on `f` and `v` off it, averages `D(f) u + (1 − D(f)) v`. -/
theorem lintegral_ofReal_ite (f : X → Bool) {u v : ℝ} (hu : 0 ≤ u) (hv : 0 ≤ v) :
    ∫⁻ x, ENNReal.ofReal (if f x then u else v) ∂D
      = ENNReal.ofReal (D.real {x | f x} * u + (1 - D.real {x | f x}) * v) := by
  set S := {x | f x = true}
  have hS : MeasurableSet S := (Set.to_countable _).measurableSet
  have hq0 : 0 ≤ D.real S := measureReal_nonneg
  have hq1 : D.real S ≤ 1 := measureReal_le_one
  have hpt : ∀ x, ENNReal.ofReal (if f x then u else v)
      = S.indicator (fun _ => ENNReal.ofReal u) x + Sᶜ.indicator (fun _ => ENNReal.ofReal v) x := by
    intro x
    by_cases hx : f x = true
    · simp [hx, S]
    · simp [hx, S]
  have hDS : D S = ENNReal.ofReal (D.real S) := (ENNReal.ofReal_toReal (measure_ne_top _ _)).symm
  have hDSc : D Sᶜ = ENNReal.ofReal (1 - D.real S) := by
    rw [← ENNReal.ofReal_toReal (measure_ne_top D Sᶜ), ← measureReal_def, measureReal_compl hS,
      probReal_univ]
  simp_rw [hpt]
  rw [lintegral_add_left (measurable_const.indicator hS), lintegral_indicator_const hS,
    lintegral_indicator_const hS.compl, hDS, hDSc, ← ENNReal.ofReal_mul hu,
    ← ENNReal.ofReal_mul hv, ← ENNReal.ofReal_add (by positivity) (by nlinarith)]
  congr 1
  ring

theorem sum_map_ite (f : X → Bool) (A B : ℝ) (l : List X) :
    (l.map fun x => if f x then A else B).sum
      = l.countP f * A + ((l.length - l.countP f : ℕ) : ℝ) * B := by
  induction l with
  | nil => simp
  | cons x l ih =>
    have hc := List.countP_le_length (p := f) (l := l)
    simp only [List.map_cons, List.sum_cons, ih, List.countP_cons, List.length_cons]
    cases hf : f x
    · simp only [Bool.false_eq_true, if_false, add_zero]
      rw [show l.length + 1 - l.countP f = l.length - l.countP f + 1 by omega]
      push_cast
      ring
    · simp only [if_true]
      rw [show l.length + 1 - (l.countP f + 1) = l.length - l.countP f by omega]
      push_cast
      ring

/-- The sequential test, as a sum over the draws of each one's log factor. -/
theorem seqRate_iff_sum (f : X → Bool) {p₀ p₁ a : ℝ} (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1)
    (hp₁ : 0 < p₁) (hp₁1 : p₁ < 1) (ha : 0 < a) {n : ℕ} {xs : List X} (hn : n ≤ xs.length) :
    seqRate p₀ p₁ a n (cnt f n xs) ↔ Real.log (1 / a)
      ≤ sumFirst (fun x => if f x then Real.log (p₁ / p₀) else Real.log ((1 - p₁) / (1 - p₀)))
        n xs := by
  have hc : cnt f n xs ≤ n := List.countP_le_length.trans (by simp [hn])
  have hlr : 0 < likelihoodRatio p₀ p₁ n (cnt f n xs) := by
    unfold likelihoodRatio
    have : 0 < (1 - p₁) / (1 - p₀) := div_pos (by linarith) (by linarith)
    have : 0 < p₁ / p₀ := div_pos hp₁ hp₀
    positivity
  unfold seqRate
  rw [← Real.log_le_log_iff (by positivity) hlr, log_likelihoodRatio hp₀ hp₀1 hp₁ hp₁1 hc]
  simp only [sumFirst]
  rw [sum_map_ite, List.length_take, min_eq_left hn]
  have hc' : List.countP f (List.take n xs) ≤ n := hc
  rw [Nat.cast_sub hc']
  rfl

/-- The sequential test of a rate `p₀` against `p₁` fires at some look with chance at most `a`
where the draws' rate is on the far side of `p₀` from `p₁`. -/
theorem seq_any_le (f : X → Bool) {p₀ p₁ a : ℝ} (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : 0 < p₁)
    (hp₁1 : p₁ < 1) (ha : 0 < a) (hq : (D.real {x | f x} - p₀) * (p₁ - p₀) ≤ 0) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D)
        {xs | ∃ n ≤ N, seqRate p₀ p₁ a n (cnt f n (List.ofFn xs))} ≤ ENNReal.ofReal a := by
  set g : X → ℝ := fun x => if f x then Real.log (p₁ / p₀) else Real.log ((1 - p₁) / (1 - p₀))
  have hsub : {xs : Fin N → X | ∃ n ≤ N, seqRate p₀ p₁ a n (cnt f n (List.ofFn xs))}
      ⊆ {xs | ∃ n ≤ N, Real.log (1 / a) ≤ sumFirst g n (List.ofFn xs)} := by
    rintro xs ⟨n, hn, h⟩
    exact ⟨n, hn, (seqRate_iff_sum f hp₀ hp₀1 hp₁ hp₁1 ha (by simpa using hn)).1 h⟩
  have hg : ∫⁻ x, ENNReal.ofReal (Real.exp (g x / 1)) ∂D ≤ 1 := by
    have := lintegral_ratioFactor_le D (fun x => f x = true) hp₀ hp₀1 hp₁.le hp₁1.le
      (by simpa using hq)
    refine le_trans (le_of_eq (lintegral_congr fun x => ?_)) this
    simp only [g, div_one, ratioFactor]
    cases f x <;> simp [Real.exp_log (div_pos hp₁ hp₀),
      Real.exp_log (div_pos (by linarith : 0 < 1 - p₁) (by linarith : 0 < 1 - p₀))]
  refine (measure_mono hsub).trans ((pi_sum_any D g one_pos hg N _).trans (le_of_eq ?_))
  rw [div_one, Real.log_div one_ne_zero ha.ne', Real.log_one, zero_sub, neg_neg,
    Real.exp_log ha]


/-- The middles' sequential test, gated on the searches being at least `s` of the probes: under a
rate of undecided middles at most `p` among all the probes, it fires at some look with chance at
most `a`, where `p (m/μ) κ^(s−1) + (1 − p) κ^s ≤ 1`, `κ = (1 − m)/(1 − μ)`. -/
theorem mid_any_le (fS fP : X → Bool) (hPS : ∀ x, fP x = true → fS x = true) {μ m s a p : ℝ}
    (hμ : 0 < μ) (hμm : μ < m) (hm1 : m < 1) (ha : 0 < a) (hp0 : 0 ≤ p)
    (hp : D.real {x | fP x} ≤ p)
    (hsuper : p * (m / μ) * ((1 - m) / (1 - μ)) ^ (s - 1) + (1 - p) * ((1 - m) / (1 - μ)) ^ s ≤ 1)
    (N : ℕ) :
    (Measure.pi fun _ : Fin N => D) {xs | ∃ n ≤ N, s * n ≤ cnt fS n (List.ofFn xs)
        ∧ seqRate μ m a (cnt fS n (List.ofFn xs)) (cnt fP n (List.ofFn xs))}
      ≤ ENNReal.ofReal a := by
  set κ := (1 - m) / (1 - μ)
  have hκ0 : 0 < κ := div_pos (by linarith) (by linarith)
  have hκ1 : κ < 1 := by rw [div_lt_one (by linarith)]; linarith
  set lk := Real.log κ
  have hlk : lk < 0 := Real.log_neg hκ0 hκ1
  set lA := Real.log (m / μ)
  set g : X → ℝ := fun x => if fP x then lA + (s - 1) * lk else s * lk
  have hsub : {xs : Fin N → X | ∃ n ≤ N, s * n ≤ cnt fS n (List.ofFn xs)
        ∧ seqRate μ m a (cnt fS n (List.ofFn xs)) (cnt fP n (List.ofFn xs))}
      ⊆ {xs | ∃ n ≤ N, Real.log (1 / a) ≤ sumFirst g n (List.ofFn xs)} := by
    rintro xs ⟨n, hn, hgate, hfire⟩
    refine ⟨n, hn, ?_⟩
    have hnl : n ≤ (List.ofFn xs).length := by simpa using hn
    set sc := cnt fS n (List.ofFn xs)
    set mid := cnt fP n (List.ofFn xs)
    have hms : mid ≤ sc := List.countP_mono_left fun x _ h => hPS x h
    have hsn : sc ≤ n := List.countP_le_length.trans (by simp [hnl])
    have hlr : 0 < likelihoodRatio μ m sc mid := by
      unfold likelihoodRatio
      have : 0 < m / μ := div_pos (by linarith) hμ
      positivity
    unfold seqRate at hfire
    rw [← Real.log_le_log_iff (by positivity) hlr,
      log_likelihoodRatio hμ (by linarith) (by linarith) hm1 hms] at hfire
    have hsum : sumFirst g n (List.ofFn xs)
        = mid * (lA + (s - 1) * lk) + ((n - mid : ℕ) : ℝ) * (s * lk) := by
      simp only [sumFirst, g]
      rw [sum_map_ite, List.length_take, min_eq_left hnl]
      rfl
    rw [hsum, Nat.cast_sub (hms.trans hsn)]
    have hsc : (sc : ℝ) ≥ s * n := hgate
    have : ((sc : ℝ) - mid) * lk ≤ (s * n - mid) * lk := by nlinarith
    nlinarith
  have hg : ∫⁻ x, ENNReal.ofReal (Real.exp (g x / 1)) ∂D ≤ 1 := by
    have hu : Real.exp (lA + (s - 1) * lk) = m / μ * κ ^ (s - 1) := by
      rw [Real.exp_add, Real.exp_log (div_pos (by linarith) hμ), Real.rpow_def_of_pos hκ0,
        mul_comm (s - 1)]
    have hv : Real.exp (s * lk) = κ ^ s := by
      rw [Real.rpow_def_of_pos hκ0, mul_comm s]
    have hpt : ∀ x, ENNReal.ofReal (Real.exp (g x / 1))
        = ENNReal.ofReal (if fP x then m / μ * κ ^ (s - 1) else κ ^ s) := by
      intro x
      simp only [g, div_one]
      split_ifs
      · rw [hu]
      · rw [hv]
    have hu0 : 0 ≤ m / μ * κ ^ (s - 1) := by
      have : 0 < m / μ := div_pos (by linarith) hμ
      positivity
    have hv0 : 0 ≤ κ ^ s := by positivity
    simp_rw [hpt]
    rw [lintegral_ofReal_ite D fP hu0 hv0, ← ENNReal.ofReal_one]
    refine ENNReal.ofReal_le_ofReal ?_
    have huv : κ ^ s ≤ m / μ * κ ^ (s - 1) := by
      have hsplit : κ ^ s = κ ^ (s - 1) * κ := by
        rw [← Real.rpow_add_one hκ0.ne']; ring_nf
      rw [hsplit]
      have h1 : κ ≤ m / μ := by
        have : 1 ≤ m / μ := by rw [le_div_iff₀ hμ]; linarith
        linarith
      have : 0 ≤ κ ^ (s - 1) := by positivity
      nlinarith
    have hq0 : 0 ≤ D.real {x | fP x} := measureReal_nonneg
    nlinarith
  refine (measure_mono hsub).trans ((pi_sum_any D g one_pos hg N _).trans (le_of_eq ?_))
  rw [div_one, Real.log_div one_ne_zero ha.ne', Real.log_one, zero_sub, neg_neg, Real.exp_log ha]

/-- Some of the first `n` of `N` fresh draws falls in `S` with chance at most `n D(S)`. -/
theorem pi_mem_le (S : Set X) (N n : ℕ) :
    (Measure.pi fun _ : Fin N => D).real {xs | ∃ w ∈ (List.ofFn xs).take n, w ∈ S}
      ≤ n * D.real S := by
  classical
  have hsub : {xs : Fin N → X | ∃ w ∈ (List.ofFn xs).take n, w ∈ S}
      ⊆ ⋃ i ∈ (Finset.univ : Finset (Fin N)).filter (fun i => i.val < n),
        Function.eval i ⁻¹' S := by
    rintro xs ⟨w, hw, hwS⟩
    obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hw
    simp only [List.length_take, List.length_ofFn] at hj
    refine Set.mem_biUnion (x := ⟨j, by omega⟩) (by simp; omega) ?_
    simpa using hwS
  have hS : MeasurableSet S := (Set.to_countable S).measurableSet
  calc _ ≤ _ := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ i ∈ (Finset.univ : Finset (Fin N)).filter (fun i => i.val < n),
          (Measure.pi fun _ : Fin N => D).real (Function.eval i ⁻¹' S) :=
        measureReal_biUnion_finset_le _ _
    _ = ∑ _i ∈ (Finset.univ : Finset (Fin N)).filter (fun i => i.val < n), D.real S :=
        Finset.sum_congr rfl fun i _ => by
          rw [measureReal_def, measureReal_def,
            (measurePreserving_eval (fun _ : Fin N => D) i).measure_preimage hS.nullMeasurableSet]
    _ ≤ n * D.real S := by
        rw [Finset.sum_const, nsmul_eq_mul]
        gcongr
        calc ((Finset.univ : Finset (Fin N)).filter (fun i => i.val < n)).card
            ≤ (Finset.range n).card :=
              Finset.card_le_card_of_injOn (fun i => i.val)
                (fun i hi => by simpa using hi) (fun a _ b _ h => Fin.ext h)
          _ = n := Finset.card_range n

end Draws

end Random

end OrthoDFA
