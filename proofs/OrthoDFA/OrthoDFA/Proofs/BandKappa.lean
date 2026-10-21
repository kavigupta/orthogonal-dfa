import OrthoDFA.Utils.BandKappa
import OrthoDFA.Proofs.BandWide

/-!
# A band's vote, against `Bin(n, 1/2)`

`g(j) = P(X = j) / C(n, j)` is log-concave with no zeros inside its support (Newton's
inequalities, proved one bit at a time), so from any count `c` it does not rise, or up to `c` it
does not fall. Where it does not rise, `P(X = j) C(n, i) ≤ C(n, j) P(X = i)` for `c ≤ i < kh ≤ j`,
and summing,

    P(X ≥ kh) · P(c ≤ Y < kh) ≤ P(Y ≥ kh) · P(c ≤ X < kh);

where it does not fall, the mirror at `kl`.
-/

namespace OrthoDFA

/-! ## The law over the binomial coefficients -/

/-- `pbStep` over the binomial coefficients, `m` bits before the step. -/
noncomputable def ulcStep (p : ℝ) (m : ℕ) (g : ℕ → ℝ) : ℕ → ℝ
  | 0 => (1 - p) * g 0
  | j + 1 => ((1 - p) * ((m - j : ℕ) : ℝ) * g (j + 1) + p * (j + 1) * g j) / (m + 1)

noncomputable def ulcLaw : List ℝ → ℕ → ℝ
  | [] => pbBase
  | p :: ps => ulcStep p ps.length (ulcLaw ps)

lemma pbLaw_cons (p : ℝ) (ps : List ℝ) : pbLaw (p :: ps) = pbStep p (pbLaw ps) := rfl

lemma pbLaw_eq_choose_mul : ∀ (ps : List ℝ) (j : ℕ),
    pbLaw ps j = (ps.length.choose j : ℝ) * ulcLaw ps j
  | [], j => by cases j <;> simp [pbLaw, ulcLaw, pbBase]
  | p :: ps, 0 => by
    rw [pbLaw_cons, pbStep, pbLaw_eq_choose_mul ps 0, ulcLaw, ulcStep]
    simp
  | p :: ps, j + 1 => by
    rw [pbLaw_cons, pbStep, pbLaw_eq_choose_mul ps (j + 1), pbLaw_eq_choose_mul ps j, ulcLaw,
      ulcStep, List.length_cons]
    have h1 : (ps.length.choose (j + 1) : ℝ) * ((ps.length : ℝ) + 1)
        = ((ps.length + 1).choose (j + 1) : ℝ) * ((ps.length - j : ℕ) : ℝ) := by
      have := Nat.choose_mul_succ_eq ps.length (j + 1)
      rw [show ps.length + 1 - (j + 1) = ps.length - j by omega] at this
      exact_mod_cast this
    have h2 : ((ps.length : ℝ) + 1) * (ps.length.choose j : ℝ)
        = ((ps.length + 1).choose (j + 1) : ℝ) * ((j : ℝ) + 1) := by
      exact_mod_cast Nat.add_one_mul_choose_eq ps.length j
    rw [mul_div_assoc', eq_div_iff (by positivity)]
    linear_combination (1 - p) * ulcLaw ps (j + 1) * h1 + p * ulcLaw ps j * h2

/-- Nonnegative, zero past `m`, log-concave, and positive on an interval. -/
structure ULC (m : ℕ) (g : ℕ → ℝ) : Prop where
  nonneg : ∀ j, 0 ≤ g j
  zero : ∀ j, m < j → g j = 0
  lc : ∀ j, g j * g (j + 2) ≤ g (j + 1) ^ 2
  conn : ∀ i j k, i < j → j < k → 0 < g i → 0 < g k → 0 < g j

lemma ULC.gap {m : ℕ} {g : ℕ → ℝ} (h : ULC m g) (i : ℕ) :
    g i * g (i + 3) ≤ g (i + 1) * g (i + 2) := by
  have h0 := h.nonneg i
  have h1 := h.nonneg (i + 1)
  have h2 := h.nonneg (i + 2)
  have h3 := h.nonneg (i + 3)
  by_cases hp : 0 < g (i + 1) ∧ 0 < g (i + 2)
  · have a := h.lc i
    have b : g (i + 1) * g (i + 3) ≤ g (i + 2) ^ 2 := h.lc (i + 1)
    refine le_of_mul_le_mul_right ?_ (mul_pos hp.1 hp.2)
    calc g i * g (i + 3) * (g (i + 1) * g (i + 2))
        = (g i * g (i + 2)) * (g (i + 1) * g (i + 3)) := by ring
      _ ≤ g (i + 1) ^ 2 * g (i + 2) ^ 2 :=
          mul_le_mul a b (mul_nonneg h1 h3) (sq_nonneg _)
      _ = g (i + 1) * g (i + 2) * (g (i + 1) * g (i + 2)) := by ring
  · have hz : g i = 0 ∨ g (i + 3) = 0 := by
      by_contra hne
      push Not at hne
      have p0 := lt_of_le_of_ne h0 (Ne.symm hne.1)
      have p3 := lt_of_le_of_ne h3 (Ne.symm hne.2)
      exact hp ⟨h.conn i (i + 1) (i + 3) (by omega) (by omega) p0 p3,
        h.conn i (i + 2) (i + 3) (by omega) (by omega) p0 p3⟩
    rcases hz with hz | hz <;> rw [hz] <;> simp only [zero_mul, mul_zero] <;>
      exact mul_nonneg h1 h2

/-- One step of Newton's inequalities. -/
lemma ulc_core {α β a s x0 x1 x2 x3 : ℝ} (hα : 0 ≤ α) (hβ : 0 ≤ β) (ha : 1 ≤ a) (hs : 0 ≤ s)
    (h1 : x1 * x3 ≤ x2 ^ 2) (h0 : s * (x0 * x2) ≤ s * x1 ^ 2)
    (hg : s * (x0 * x3) ≤ s * (x1 * x2)) :
    (α * (a + 1) * x1 + β * s * x0) * (α * (a - 1) * x3 + β * (s + 2) * x2)
      ≤ (α * a * x2 + β * (s + 1) * x1) ^ 2 := by
  have e1 : 0 ≤ α ^ 2 * (a ^ 2 - 1) * (x2 ^ 2 - x1 * x3) :=
    mul_nonneg (mul_nonneg (sq_nonneg _) (by nlinarith)) (by linarith)
  have e2 : 0 ≤ β ^ 2 * (s + 2) * (s * x1 ^ 2 - s * (x0 * x2)) :=
    mul_nonneg (mul_nonneg (sq_nonneg _) (by linarith)) (by linarith)
  have e3 : 0 ≤ α * β * (a - 1) * (s * (x1 * x2) - s * (x0 * x3)) :=
    mul_nonneg (mul_nonneg (mul_nonneg hα hβ) (by linarith)) (by linarith)
  nlinarith [sq_nonneg (α * x2 - β * x1)]

lemma ulcStep_mul (p : ℝ) (m : ℕ) (g : ℕ → ℝ) (J : ℕ) :
    ((m : ℝ) + 1) * ulcStep p m g J
      = (1 - p) * ((m + 1 - J : ℕ) : ℝ) * g J + p * J * g (J - 1) := by
  cases J with
  | zero => simp only [ulcStep, Nat.sub_zero, Nat.cast_zero]; push_cast; ring
  | succ j =>
    simp only [ulcStep]
    rw [show m + 1 - (j + 1) = m - j by omega, Nat.add_sub_cancel]
    field_simp
    push_cast
    ring

lemma pos_of_mul3 {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : 0 < x * y * z) :
    0 < x ∧ 0 < y ∧ 0 < z := by
  refine ⟨lt_of_le_of_ne hx ?_, lt_of_le_of_ne hy ?_, lt_of_le_of_ne hz ?_⟩ <;> rintro rfl <;>
    simp at h

lemma ulcStep_pos {p : ℝ} {m : ℕ} {g : ℕ → ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (h : ULC m g)
    (J : ℕ) : 0 < ulcStep p m g J ↔ (p < 1 ∧ 0 < g J) ∨ (0 < p ∧ 0 < J ∧ 0 < g (J - 1)) := by
  have hm : (0 : ℝ) < m + 1 := by positivity
  rw [← mul_pos_iff_of_pos_left hm, ulcStep_mul]
  have t1 : 0 ≤ (1 - p) * ((m + 1 - J : ℕ) : ℝ) * g J :=
    mul_nonneg (mul_nonneg (by linarith) (Nat.cast_nonneg _)) (h.nonneg J)
  have t2 : 0 ≤ p * J * g (J - 1) :=
    mul_nonneg (mul_nonneg hp0 (Nat.cast_nonneg _)) (h.nonneg _)
  constructor
  · intro hpos
    by_cases hA : 0 < (1 - p) * ((m + 1 - J : ℕ) : ℝ) * g J
    · obtain ⟨a1, -, a3⟩ := pos_of_mul3 (by linarith) (Nat.cast_nonneg _) (h.nonneg J) hA
      exact Or.inl ⟨by linarith, a3⟩
    · have hB : 0 < p * J * g (J - 1) := by linarith [le_antisymm (not_lt.1 hA) t1]
      obtain ⟨b1, b2, b3⟩ := pos_of_mul3 hp0 (Nat.cast_nonneg _) (h.nonneg _) hB
      exact Or.inr ⟨b1, Nat.cast_pos.1 b2, b3⟩
  · rintro (⟨hp, hg⟩ | ⟨hp, hJ, hg⟩)
    · have hJm : J ≤ m := by
        by_contra hc
        rw [h.zero J (by omega)] at hg
        exact lt_irrefl _ hg
      have : (0 : ℝ) < ((m + 1 - J : ℕ) : ℝ) := Nat.cast_pos.2 (by omega)
      have : 0 < (1 - p) * ((m + 1 - J : ℕ) : ℝ) * g J :=
        mul_pos (mul_pos (by linarith) this) hg
      linarith
    · have : 0 < p * J * g (J - 1) := mul_pos (mul_pos hp (Nat.cast_pos.2 hJ)) hg
      linarith

lemma ULC.step {p : ℝ} {m : ℕ} {g : ℕ → ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (h : ULC m g) :
    ULC (m + 1) (ulcStep p m g) where
  nonneg J := by
    have hm : (0 : ℝ) < m + 1 := by positivity
    refine (mul_nonneg_iff_of_pos_left hm).1 ?_
    rw [ulcStep_mul]
    exact add_nonneg (mul_nonneg (mul_nonneg (by linarith) (Nat.cast_nonneg _)) (h.nonneg J))
      (mul_nonneg (mul_nonneg hp0 (Nat.cast_nonneg _)) (h.nonneg _))
  zero J hJ := by
    have e := ulcStep_mul p m g J
    rw [h.zero J (by omega), h.zero (J - 1) (by omega), mul_zero, mul_zero, add_zero] at e
    exact (mul_eq_zero.1 e).resolve_left (by positivity)
  lc J := by
    have hm : (0 : ℝ) < m + 1 := by positivity
    suffices hs : ((m : ℝ) + 1) * ulcStep p m g J * (((m : ℝ) + 1) * ulcStep p m g (J + 2))
        ≤ (((m : ℝ) + 1) * ulcStep p m g (J + 1)) ^ 2 by
      have e : ((m : ℝ) + 1) * ulcStep p m g J * (((m : ℝ) + 1) * ulcStep p m g (J + 2))
          = ((m : ℝ) + 1) ^ 2 * (ulcStep p m g J * ulcStep p m g (J + 2)) := by ring
      have e' : (((m : ℝ) + 1) * ulcStep p m g (J + 1)) ^ 2
          = ((m : ℝ) + 1) ^ 2 * ulcStep p m g (J + 1) ^ 2 := by ring
      rw [e, e'] at hs
      exact le_of_mul_le_mul_left hs (by positivity)
    rw [ulcStep_mul, ulcStep_mul, ulcStep_mul, show J + 2 - 1 = J + 1 by omega,
      show J + 1 - 1 = J by omega]
    rcases Nat.lt_or_ge m (J + 1) with hJ | hJ
    · rw [h.zero (J + 1) hJ, h.zero (J + 2) (by omega)]
      simp only [mul_zero, zero_add, add_zero]
      positivity
    · obtain ⟨d, rfl⟩ : ∃ d, m = J + 1 + d := ⟨m - (J + 1), by omega⟩
      rw [show J + 1 + d + 1 - J = d + 2 by omega, show J + 1 + d + 1 - (J + 1) = d + 1 by omega,
        show J + 1 + d + 1 - (J + 2) = d by omega]
      have h0 : (J : ℝ) * (g (J - 1) * g (J + 1)) ≤ J * g J ^ 2 := by
        rcases J with _ | J
        · simp
        · refine mul_le_mul_of_nonneg_left ?_ (Nat.cast_nonneg _)
          have := h.lc J
          rwa [show J + 1 - 1 = J by omega]
      have hg : (J : ℝ) * (g (J - 1) * g (J + 2)) ≤ J * (g J * g (J + 1)) := by
        rcases J with _ | J
        · simp
        · refine mul_le_mul_of_nonneg_left ?_ (Nat.cast_nonneg _)
          have := h.gap J
          rwa [show J + 1 - 1 = J by omega]
      have key := ulc_core (α := 1 - p) (β := p) (a := (d : ℝ) + 1) (s := (J : ℝ))
        (x0 := g (J - 1)) (x1 := g J) (x2 := g (J + 1)) (x3 := g (J + 2)) (by linarith) hp0
        (by have := (Nat.cast_nonneg d : (0 : ℝ) ≤ d); linarith) (Nat.cast_nonneg _) (h.lc J)
        h0 hg
      push_cast
      linear_combination key
  conn i j k hij hjk hi hk := by
    rw [ulcStep_pos hp0 hp1 h] at hi hk ⊢
    rcases hi with ⟨hp, hi⟩ | ⟨hp, hi0, hi⟩ <;> rcases hk with ⟨hp', hk⟩ | ⟨hp', hk0, hk⟩
    · exact Or.inl ⟨hp, h.conn i j k hij hjk hi hk⟩
    · rcases Nat.lt_or_ge j (k - 1) with hj | hj
      · exact Or.inl ⟨hp, h.conn i j (k - 1) hij hj hi hk⟩
      · rw [show j = k - 1 by omega] at ⊢
        exact Or.inl ⟨hp, hk⟩
    · exact Or.inr ⟨hp, by omega, h.conn (i - 1) (j - 1) k (by omega) (by omega) hi hk⟩
    · exact Or.inr ⟨hp, by omega, h.conn (i - 1) (j - 1) (k - 1) (by omega) (by omega) hi hk⟩

lemma ulc_pbBase : ULC 0 pbBase where
  nonneg j := by unfold pbBase; split_ifs <;> norm_num
  zero j hj := by simp [pbBase]; omega
  lc j := by simp [pbBase]
  conn i j k hij hjk _ hk := by
    simp only [pbBase] at hk
    split_ifs at hk with h
    · omega
    · exact absurd hk (lt_irrefl _)

lemma ulcLaw_ulc : ∀ ps : List ℝ, (∀ x ∈ ps, 0 ≤ x ∧ x ≤ 1) → ULC ps.length (ulcLaw ps)
  | [], _ => ulc_pbBase
  | p :: ps, h => by
    have hp := h p List.mem_cons_self
    exact (ulcLaw_ulc ps fun x hx => h x (List.mem_cons_of_mem _ hx)).step hp.1 hp.2

/-- A log-concave sequence with no zeros inside its support does not rise from `c`, or does not
fall up to `c`. -/
lemma ULC.mono_or {m : ℕ} {g : ℕ → ℝ} (h : ULC m g) (c : ℕ) :
    (∀ i j, c ≤ i → i ≤ j → g j ≤ g i) ∨ (∀ i j, i ≤ j → j ≤ c → g i ≤ g j) := by
  by_cases hd : ∀ j, c ≤ j → g (j + 1) ≤ g j
  · left
    intro i j hi hij
    induction j, hij using Nat.le_induction with
    | base => exact le_rfl
    | succ j hij ih => exact (hd j (hi.trans hij)).trans ih
  · right
    push Not at hd
    obtain ⟨i₀, hci, hlt⟩ := hd
    have hpos : 0 < g (i₀ + 1) := lt_of_le_of_lt (h.nonneg i₀) hlt
    have hup : ∀ d, d ≤ i₀ → g (i₀ - d) ≤ g (i₀ - d + 1) := by
      intro d
      induction d with
      | zero => intro _; simpa using hlt.le
      | succ d ih =>
        intro hd
        obtain ⟨k, hk⟩ : ∃ k, i₀ - d = k + 1 := ⟨i₀ - (d + 1), by omega⟩
        have ih' : g (k + 1) ≤ g (k + 2) := by
          have := ih (by omega)
          rwa [hk] at this
        rw [show i₀ - (d + 1) = k by omega]
        by_cases hk1 : 0 < g (k + 1)
        · have hk2 : 0 < g (k + 2) := lt_of_lt_of_le hk1 ih'
          have : g k * g (k + 2) ≤ g (k + 1) * g (k + 2) :=
            (h.lc k).trans (by nlinarith [h.nonneg (k + 1)])
          exact le_of_mul_le_mul_right this hk2
        · have hz : g (k + 1) = 0 := le_antisymm (not_lt.1 hk1) (h.nonneg _)
          have hk0 : g k = 0 := by
            by_contra hne
            have := h.conn k (k + 1) (i₀ + 1) (by omega) (by omega)
              (lt_of_le_of_ne (h.nonneg k) (Ne.symm hne)) hpos
            linarith
          rw [hk0, hz]
    intro i j hij hjc
    induction j, hij using Nat.le_induction with
    | base => exact le_rfl
    | succ j hij ih =>
      have := hup (i₀ - j) (by omega)
      rw [show i₀ - (i₀ - j) = j by omega] at this
      exact (ih (by omega)).trans this

/-! ## Windows of the count -/

/-- `∑_{t ≤ j < t'} x j`, over `j ≤ n`. -/
noncomputable def win (n t t' : ℕ) (x : ℕ → ℝ) : ℝ :=
  ∑ j ∈ Finset.range (n + 1), if t ≤ j ∧ j < t' then x j else 0

lemma win_choose (n : ℕ) {t t' : ℕ} (h : t ≤ t') :
    win n t t' (fun j => (n.choose j : ℝ))
      = 2 ^ n * (binomSfGe n (1 / 2) t - binomSfGe n (1 / 2) t') := by
  rw [binomSfGe_eq_range, binomSfGe_eq_range, ← Finset.sum_sub_distrib, Finset.mul_sum, win]
  refine Finset.sum_congr rfl fun j hj => ?_
  have hjn : j ≤ n := Nat.lt_succ_iff.1 (Finset.mem_range.1 hj)
  have e : (n.choose j : ℝ) * (1 / 2) ^ j * (1 - 1 / 2) ^ (n - j) = n.choose j * (1 / 2) ^ n := by
    rw [show (1 : ℝ) - 1 / 2 = 1 / 2 by norm_num, mul_assoc, ← pow_add, Nat.add_sub_cancel' hjn]
  have e2 : (2 : ℝ) ^ n * (n.choose j * (1 / 2) ^ n) = n.choose j := by
    rw [mul_left_comm, ← mul_pow]; norm_num
  by_cases h1 : t ≤ j <;> by_cases h2 : t' ≤ j
  · rw [if_neg (by omega), if_pos h1, if_pos h2, sub_self, mul_zero]
  · rw [if_pos ⟨h1, by omega⟩, if_pos h1, if_neg h2, sub_zero, e, e2]
  · omega
  · rw [if_neg (by omega), if_neg h1, if_neg h2, sub_zero, mul_zero]

lemma win_nonneg (n t t' : ℕ) {x : ℕ → ℝ} (hx : ∀ j, 0 ≤ x j) : 0 ≤ win n t t' x :=
  Finset.sum_nonneg fun j _ => by split_ifs; exacts [hx j, le_rfl]

lemma win_mono (n : ℕ) {t₁ t₁' t₂ t₂' : ℕ} {x : ℕ → ℝ} (hx : ∀ j, 0 ≤ x j) (h : t₁ ≤ t₂)
    (h' : t₂' ≤ t₁') : win n t₂ t₂' x ≤ win n t₁ t₁' x :=
  Finset.sum_le_sum fun j _ => by
    by_cases h2 : t₂ ≤ j ∧ j < t₂'
    · rw [if_pos h2, if_pos ⟨by omega, by omega⟩]
    · rw [if_neg h2]; split_ifs; exacts [hx j, le_rfl]

lemma win_choose_pos {n t c t' : ℕ} (htc : t ≤ c) (hct : c < t') (hcn : c ≤ n) :
    0 < win n t t' (fun j => (n.choose j : ℝ)) := by
  refine lt_of_lt_of_le ?_ (Finset.single_le_sum (f := fun j => if t ≤ j ∧ j < t' then
    (n.choose j : ℝ) else 0) (fun j _ => by split_ifs <;> positivity)
    (Finset.mem_range.2 (Nat.lt_succ_of_le hcn)))
  rw [if_pos ⟨htc, hct⟩]
  exact_mod_cast Nat.choose_pos hcn

/-- Where `g` is no larger on the window `T` than on the window `S`, and `f = w g`, `T` holds at
most `w(T)/w(S)` of what `S` holds. -/
lemma win_cross {n s s' t t' : ℕ} {f g w : ℕ → ℝ} (hf : ∀ j, f j = w j * g j)
    (hw : ∀ j, 0 ≤ w j) (hST : ∀ i j, s ≤ i → i < s' → t ≤ j → j < t' → g j ≤ g i) :
    win n t t' f * win n s s' w ≤ win n t t' w * win n s s' f := by
  unfold win
  rw [Finset.sum_mul_sum, Finset.sum_mul_sum]
  refine Finset.sum_le_sum fun j _ => Finset.sum_le_sum fun i _ => ?_
  by_cases hj : t ≤ j ∧ j < t' <;> by_cases hi : s ≤ i ∧ i < s'
  · rw [if_pos hj, if_pos hi, if_pos hj, if_pos hi, hf, hf]
    have := hST i j hi.1 hi.2 hj.1 hj.2
    nlinarith [mul_nonneg (hw j) (hw i)]
  all_goals simp [hj, hi]

/-! ## The bound -/

lemma pbLaw_replicate_zero (d : ℕ) (h : ℕ → ℝ) :
    (List.replicate d (0 : ℝ)).foldr pbStep h = h := by
  induction d with
  | zero => rfl
  | succ d ih =>
    rw [List.replicate_succ, List.foldr_cons, ih]
    funext k
    cases k <;> simp [pbStep]

lemma readOf_accept {kl kh j : ℕ} : readOf kl kh j = .accept ↔ kh ≤ j := by
  unfold readOf; split_ifs <;> simp_all

lemma readOf_reject {kl kh j : ℕ} (h : kl < kh) : readOf kl kh j = .reject ↔ j ≤ kl := by
  unfold readOf; split_ifs <;> simp_all; omega

lemma readOf_undecided {kl kh j : ℕ} : readOf kl kh j = .undecided ↔ kl < j ∧ j < kh := by
  unfold readOf; split_ifs <;> simp_all

theorem bandKappaBound_holds : BandKappaBound := by
  intro N n a kl c kh p r haN hNn hklc hckh hcn hp0 hp1 hr0 hr1
  classical
  set ps := List.replicate (n - N) (0 : ℝ) ++ (List.replicate a p ++ List.replicate (N - a) r)
    with hps
  have hlen : ps.length = n := by simp only [ps, List.length_append, List.length_replicate]; omega
  have h01 : ∀ x ∈ ps, 0 ≤ x ∧ x ≤ 1 := by
    intro x hx
    simp only [ps, List.mem_append, List.mem_replicate] at hx
    rcases hx with ⟨-, rfl⟩ | ⟨-, rfl⟩ | ⟨-, rfl⟩
    · norm_num
    · exact ⟨hp0, hp1⟩
    · exact ⟨hr0, hr1⟩
  set g := ulcLaw ps with hgd
  have hg : ULC n g := hlen ▸ ulcLaw_ulc ps h01
  have hf : ∀ j, voteLaw N a p r j = (n.choose j : ℝ) * g j := by
    intro j
    have e : pbLaw ps = (List.replicate a p ++ List.replicate (N - a) r).foldr pbStep pbBase := by
      rw [pbLaw, hps, List.foldr_append, pbLaw_replicate_zero]
    rw [voteLaw_eq_foldr, ← e, pbLaw_eq_choose_mul, hlen]
  have hfN : ∀ j, N < j → voteLaw N a p r j = 0 := by
    intro j hj
    rw [voteLaw_eq_foldr, show (List.replicate a p ++ List.replicate (N - a) r).foldr pbStep pbBase
      = pbLaw (List.replicate a p ++ List.replicate (N - a) r) from rfl, pbLaw_eq_coeff]
    refine Polynomial.coeff_eq_zero_of_natDegree_lt ((natDegree_bitPoly_le _).trans_lt ?_)
    simp only [Multiset.coe_card, List.length_append, List.length_replicate]
    omega
  have hf0 : ∀ j, 0 ≤ voteLaw N a p r j := fun j => by
    rw [hf]; exact mul_nonneg (Nat.cast_nonneg _) (hg.nonneg j)
  -- the three reads as windows of the count
  have hdist : ∀ (rd : ARU) (t t' : ℕ), (∀ j, j ≤ n → (readOf kl kh j = rd ↔ t ≤ j ∧ j < t')) →
      (∑ j ∈ Finset.range (N + 1), if readOf kl kh j = rd then
        ∑ x ∈ Finset.antidiagonal j, (a.choose x.1 * p ^ x.1 * (1 - p) ^ (a - x.1))
          * ((N - a).choose x.2 * r ^ x.2 * (1 - r) ^ (N - a - x.2)) else 0)
        = win n t t' (voteLaw N a p r) := by
    intro rd t t' hrd
    rw [win]
    refine (Finset.sum_subset (Finset.range_subset_range.2 (by omega)) fun j _ hj => ?_).trans
      (Finset.sum_congr rfl fun j hj => ?_)
    · rw [Finset.mem_range, not_lt] at hj
      split_ifs
      · exact hfN j (by omega)
      · rfl
    · have hjn : j ≤ n := Nat.lt_succ_iff.1 (Finset.mem_range.1 hj)
      by_cases h : readOf kl kh j = rd
      · rw [if_pos h, if_pos ((hrd j hjn).1 h)]; rfl
      · rw [if_neg h, if_neg (fun h' => h ((hrd j hjn).2 h'))]
  dsimp only
  rw [hdist .accept kh (kh + n + 1) (fun j hj => by rw [readOf_accept]; omega),
    hdist .reject 0 (kl + 1) (fun j hj => by rw [readOf_reject (by omega)]; omega),
    hdist .undecided (kl + 1) kh (fun j hj => by rw [readOf_undecided]; omega)]
  set U := win n (kl + 1) kh (voteLaw N a p r)
  have hU0 : 0 ≤ U := win_nonneg _ _ _ hf0
  have h2n : (0 : ℝ) < 2 ^ n := by positivity
  rcases hg.mono_or c with hdown | hup
  · -- past `c` the law falls at least as fast as `Bin(n, 1/2)`'s
    have hx := win_cross (n := n) (s := c) (s' := kh) (t := kh) (t' := kh + n + 1) hf
      (fun j => Nat.cast_nonneg _) (fun i j hi _ _ _ => hdown i j hi (by omega))
    have hD := win_choose_pos (n := n) le_rfl hckh hcn
    rw [win_choose n hckh.le, win_choose n (by omega),
      binomSfGe_eq_zero n (1 / 2) (show n < kh + n + 1 by omega), sub_zero] at hx
    rw [win_choose n hckh.le] at hD
    have hY : 0 < binomSfGe n (1 / 2) c - binomSfGe n (1 / 2) kh := pos_of_mul_pos_right hD h2n.le
    have hx' : win n kh (kh + n + 1) (voteLaw N a p r)
        * (binomSfGe n (1 / 2) c - binomSfGe n (1 / 2) kh)
        ≤ binomSfGe n (1 / 2) kh * win n c kh (voteLaw N a p r) := by
      have := hx
      rw [show ∀ A B : ℝ, A * (2 ^ n * B) = 2 ^ n * (A * B) from fun _ _ => by ring,
        show ∀ C D : ℝ, 2 ^ n * C * D = 2 ^ n * (C * D) from fun _ _ => by ring] at this
      exact le_of_mul_le_mul_left this h2n
    have hUc : win n c kh (voteLaw N a p r) ≤ U := win_mono n hf0 (by omega) le_rfl
    have hA : win n kh (kh + n + 1) (voteLaw N a p r)
        ≤ binomSfGe n (1 / 2) kh / (binomSfGe n (1 / 2) c - binomSfGe n (1 / 2) kh) * U := by
      rw [div_mul_eq_mul_div, le_div_iff₀ hY]
      exact hx'.trans (mul_le_mul_of_nonneg_left hUc
        (binomSfGe_nonneg (by norm_num) (by norm_num) _))
    exact (min_le_left _ _).trans (hA.trans (mul_le_mul_of_nonneg_right (le_max_left _ _) hU0))
  · -- up to `c` the law rises at least as fast as `Bin(n, 1/2)`'s
    have hx := win_cross (n := n) (s := kl + 1) (s' := c + 1) (t := 0) (t' := kl + 1) hf
      (fun j => Nat.cast_nonneg _) (fun i j _ hi _ hj => hup j i (by omega) (by omega))
    have hD := win_choose_pos (n := n) (show kl + 1 ≤ c by omega) (lt_add_one c) hcn
    rw [win_choose n (show kl + 1 ≤ c + 1 by omega), win_choose n (Nat.zero_le _),
      binomSfGe_zero_right] at hx
    rw [win_choose n (show kl + 1 ≤ c + 1 by omega)] at hD
    have hY : 0 < binomSfGe n (1 / 2) (kl + 1) - binomSfGe n (1 / 2) (c + 1) :=
      pos_of_mul_pos_right hD h2n.le
    have hx' : win n 0 (kl + 1) (voteLaw N a p r)
        * (binomSfGe n (1 / 2) (kl + 1) - binomSfGe n (1 / 2) (c + 1))
        ≤ (1 - binomSfGe n (1 / 2) (kl + 1)) * win n (kl + 1) (c + 1) (voteLaw N a p r) := by
      have := hx
      rw [show ∀ A B : ℝ, A * (2 ^ n * B) = 2 ^ n * (A * B) from fun _ _ => by ring,
        show ∀ C D : ℝ, 2 ^ n * C * D = 2 ^ n * (C * D) from fun _ _ => by ring] at this
      exact le_of_mul_le_mul_left this h2n
    have hUc : win n (kl + 1) (c + 1) (voteLaw N a p r) ≤ U := win_mono n hf0 le_rfl (by omega)
    have hR : win n 0 (kl + 1) (voteLaw N a p r)
        ≤ (1 - binomSfGe n (1 / 2) (kl + 1))
          / (binomSfGe n (1 / 2) (kl + 1) - binomSfGe n (1 / 2) (c + 1)) * U := by
      rw [div_mul_eq_mul_div, le_div_iff₀ hY]
      have hb1 : binomSfGe n (1 / 2) (kl + 1) ≤ 1 := binomSfGe_le_one (by norm_num) (by norm_num) _
      exact hx'.trans (mul_le_mul_of_nonneg_left hUc (by linarith))
    exact (min_le_right _ _).trans (hR.trans (mul_le_mul_of_nonneg_right (le_max_right _ _) hU0))

/-- A band at `lo` and `hi + 1` centred over an even `n`, at `c = n/2`:

    bandKappa = h / (P(Y ≥ n/2) − h),   h = P(Y ≤ lo),   Y ~ Bin(n, 1/2) -/
lemma bandKappa_centre {n lo hi : ℕ} (hsum : lo + hi + 1 = n) (hlo : lo < n / 2)
    (hhi : n / 2 ≤ hi) (hev : Even n) :
    bandKappa n lo (n / 2) (hi + 1)
      = (1 - binomSfGe n (1 / 2) (lo + 1))
        / (binomSfGe n (1 / 2) (n / 2) - (1 - binomSfGe n (1 / 2) (lo + 1))) := by
  obtain ⟨m, rfl⟩ := hev
  rw [show (m + m) / 2 = m by omega] at hlo hhi ⊢
  have s1 : binomSfGe (m + m) (1 / 2) (hi + 1) = 1 - binomSfGe (m + m) (1 / 2) (lo + 1) := by
    rw [binomSfGe_symm _ _ (hi + 1) (by omega), show m + m + 1 - (hi + 1) = lo + 1 by omega,
      show (1 : ℝ) - 1 / 2 = 1 / 2 by norm_num]
  have s2 : binomSfGe (m + m) (1 / 2) (m + 1) = 1 - binomSfGe (m + m) (1 / 2) m := by
    rw [binomSfGe_symm _ _ (m + 1) (by omega), show m + m + 1 - (m + 1) = m by omega,
      show (1 : ℝ) - 1 / 2 = 1 / 2 by norm_num]
  rw [bandKappa, s1, s2]
  have e : binomSfGe (m + m) (1 / 2) (lo + 1) - (1 - binomSfGe (m + m) (1 / 2) m)
      = binomSfGe (m + m) (1 / 2) m - (1 - binomSfGe (m + m) (1 / 2) (lo + 1)) := by ring
  rw [e, max_self]

/-- `Bin(n, 1/2)` lands at or above `n/2` at least half the time, for `n` even. -/
lemma half_le_binomSfGe_half {n : ℕ} (hev : Even n) : 1 / 2 ≤ binomSfGe n (1 / 2) (n / 2) := by
  obtain ⟨m, rfl⟩ := hev
  rw [show (m + m) / 2 = m by omega]
  have s2 : binomSfGe (m + m) (1 / 2) (m + 1) = 1 - binomSfGe (m + m) (1 / 2) m := by
    rw [binomSfGe_symm _ _ (m + 1) (by omega), show m + m + 1 - (m + 1) = m by omega,
      show (1 : ℝ) - 1 / 2 = 1 / 2 by norm_num]
  have := binomSfGe_antitone (n := m + m) (p := 1 / 2) (by norm_num) (by norm_num) m
  linarith

end OrthoDFA
