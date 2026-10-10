import OrthoDFA.RandomRound
import OrthoDFA.Proofs.IdealRound

/-!
# Counts over fresh draws

`binE n p f` is `E[f(Bin(n, p))]`; conditioning on one trial gives its recursion, and from it
its monotonicity in `p`. Over `N` fresh draws, how many of the first `n` fall in `B`, and how
many of those in `A ⊆ B`, are `Bin(n, D(B))` and, given the first, `Bin(·, D(A)/D(B))`.
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

/-- An antitone function of `Bin(n, p)` falls with `p`. -/
theorem binE_le_of_le_anti {p' : ℝ} (hp0 : 0 ≤ p) (hpp : p ≤ p') (hp1 : p' ≤ 1) (n : ℕ)
    (f : ℕ → ℝ) (hf : Antitone f) : binE n p' f ≤ binE n p f := by
  have := binE_le_of_le hp0 hpp hp1 n (fun j => -f j) fun i j h => neg_le_neg (hf h)
  have e : ∀ q, binE n q (fun j => -f j) = -binE n q f := fun q => by
    have := binE_lin n q (-1) 0 f f
    simp only [neg_one_mul, zero_mul, add_zero] at this
    exact this
  rw [e, e] at this
  linarith

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

theorem binomSfGe_anti {n j j' : ℕ} (h : j ≤ j') : binomSfGe n p j' ≤ binomSfGe n p j := by
  rw [binomSfGe_eq, binomSfGe_eq]
  exact binE_mono hp0 hp1 fun i => by split_ifs <;> first | omega | norm_num

theorem binomSfGe_succ_n (n j : ℕ) : binomSfGe n p j ≤ binomSfGe (n + 1) p j := by
  rw [binomSfGe_eq, binomSfGe_eq, binE_succ']
  exact binE_mono hp0 hp1 fun i => by
    split_ifs <;> first | omega | nlinarith

theorem binomSfGe_mono_n {n n' : ℕ} (h : n ≤ n') (j : ℕ) : binomSfGe n p j ≤ binomSfGe n' p j := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => exact ih.trans (binomSfGe_succ_n hp0 hp1 _ j)

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

/-- The lower tail's p-value is below `a` with chance at most `a`. -/
theorem binE_pvalue_le {a : ℝ} (ha : 0 ≤ a) (n : ℕ) :
    binE n p (fun j => if 1 - binomSfGe n p (j + 1) < a then 1 else 0) ≤ a := by
  classical
  set F : ℕ → ℝ := fun j => 1 - binomSfGe n p (j + 1)
  have hF : Monotone F := fun i j h => by
    simp only [F]
    linarith [binomSfGe_anti hp0 hp1 (n := n) (Nat.succ_le_succ h)]
  set j₀ := Nat.findGreatest (fun j => F j < a) n
  by_cases h₀ : F j₀ < a
  · have heq : binE n p (fun j => if F j < a then 1 else 0)
        = binE n p (fun i => if i ≤ j₀ then 1 else 0) := by
      refine binE_congr fun j hj => ?_
      by_cases hj₀ : j ≤ j₀
      · rw [if_pos hj₀, if_pos (lt_of_le_of_lt (hF hj₀) h₀)]
      · rw [if_neg hj₀, if_neg]
        exact fun h => hj₀ (Nat.le_findGreatest (P := fun j => F j < a) hj h)
    change binE n p (fun j => if F j < a then 1 else 0) ≤ a
    rw [heq, ← one_sub_binomSfGe hp0 hp1]
    exact h₀.le
  · have heq : binE n p (fun j => if F j < a then 1 else 0) = binE n p (fun _ => 0) := by
      refine binE_congr fun j hj => ?_
      rw [if_neg]
      intro h
      exact h₀ (Nat.findGreatest_spec (P := fun j => F j < a) hj h)
    change binE n p (fun j => if F j < a then 1 else 0) ≤ a
    rw [heq, binE_const]
    exact ha

end Binomial

theorem stretchRisk_nonneg (C : Cfg) (L Ns N₁ : ℕ) (G θr εd' θpt' : ℝ) (hG0 : 0 ≤ G)
    (hG1 : G ≤ 1) (ha : 0 ≤ C.a) (hθr0 : 0 ≤ θr) (hθr1 : θr ≤ 1) (hεd'0 : 0 ≤ εd')
    (hεd'1 : εd' ≤ 1) (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1) :
    0 ≤ stretchRisk C L Ns N₁ G θr εd' θpt' := by
  unfold stretchRisk
  have h01 : ∀ (j : ℕ) (P : Prop) [Decidable P], (0 : ℝ) ≤ if P then 1 else 0 := fun _ P _ => by
    split_ifs <;> norm_num
  have := binomSfGe_le_one hθr0 hθr1 Ns C.m
  have := binomSfGe_le_one hεd'0 hεd'1 Ns (max C.n₀ ⌈C.qg * Ns⌉₊)
  have := binE_nonneg hεd'0 hεd'1 (n := N₁)
    (f := fun s => if C.a ≤ 1 - binomSfGe N₁ C.εd (s + 1) ∨ binomSfGe N₁ C.εd s < C.a then 1
      else 0) fun j => h01 j _
  have : 0 ≤ ∑ s ∈ Icc (max C.n₀ ⌈C.qg * Ns⌉₊) Ns,
      binE s θpt' (fun j => if C.a ≤ binomSfGe s C.θpt j then 1 else 0) :=
    sum_nonneg fun s _ => binE_nonneg hθpt'0 hθpt'1 fun j => h01 j _
  have : 0 ≤ ∑ n ∈ Icc C.n₀ Ns, (binomSfGe n G ((hFire C.θs C.a n + 1) / 2) + C.a) :=
    sum_nonneg fun n _ => add_nonneg (binomSfGe_nonneg hG0 hG1 _ _) ha
  have : 0 ≤ ∑ n ∈ Icc 1 Ns,
      (binomSfGe n G ⌈(C.exc + C.θe * C.τe * n) / (2 * (L + 1))⌉₊
        + binomSfGe n G ((hFire C.θpt C.a (max C.n₀ ⌈C.qg * n⌉₊) + 1) / 2)) :=
    sum_nonneg fun n _ => add_nonneg (binomSfGe_nonneg hG0 hG1 _ _)
      (binomSfGe_nonneg hG0 hG1 _ _)
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

end Draws

end Random

end OrthoDFA
