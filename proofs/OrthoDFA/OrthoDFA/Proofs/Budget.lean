import OrthoDFA.Proofs.Adaptive

/-!
# What the budget costs

`prefCount` is the count the round's tails ask for, and `ClusteringCorrect` quantifies over
the states the schedule built from it.  Written out it is a sum of six ceilings, each a log
over the square of some margin, and nothing in the statement says how those margins depend on
the rates they resolve.  This file says it: every term is bounded by one polynomial in the
reciprocals of `sig η` and `budgetScale`, so the budget is `log` in the failure probability and
polynomial in everything else.

The two scales are not independent.  `screenMargin` carries `cutBudget · sig³`, and the tail
it sits in squares it, so `sig` enters at the sixth power — and `cutBudget` is itself a minimum
over `εcov`, `sig` and `indecisionLimit`, so a bound in `sig` alone cannot exist.  The scale
enters at the third power rather than the second because the coverage tails divide by `εcov`
once before squaring a margin that already carries `εcov`.

With that bound `ClusteringGuarantee` follows from `ClusteringCorrect`, at the end.
-/

namespace OrthoDFA

-- `Finset J` carries the instance the definitions were written with; these bounds read
-- only the card, so the linter is right that they do not use it themselves.
set_option linter.unusedFintypeInType false

variable {J : Type*} [Fintype J]

/-- Every ceiling is under its argument plus one, and under one when the argument is
negative -- which is what lets the six terms be bounded separately and summed. -/
private lemma cast_ceil_le (x : ℝ) : (⌈x⌉₊ : ℝ) ≤ max x 0 + 1 := by
  by_cases hx : x ≤ 0
  · have : ⌈x⌉₊ = 0 := Nat.ceil_eq_zero.2 hx
    rw [this]
    have : (0 : ℝ) ≤ max x 0 := le_max_right _ _
    simpa using by linarith
  · have hx' : 0 ≤ x := le_of_not_ge hx
    calc (⌈x⌉₊ : ℝ) ≤ x + 1 := (Nat.ceil_lt_add_one hx').le
      _ ≤ max x 0 + 1 := by gcongr; exact le_max_left _ _

/-- A ceiling under a nonnegative bound on its argument. -/
private lemma ceil_le_of_le {x M : ℝ} (hx : x ≤ M) (hM : 0 ≤ M) : (⌈x⌉₊ : ℝ) ≤ M + 1 :=
  le_trans (cast_ceil_le x) (by linarith [max_le hx hM])

/-- The six tails, each named, so the bound can be read term by term. -/
noncomputable def prefTerms (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ α pAP : ℝ) : List ℝ :=
  [ Real.log (128 * (populations.card : ℝ)
      * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 2) ^ 2 / δ)
      / (2 * (screenMargin η populations indecisionLimit εcov δ / 2) ^ 2),
    Real.log (128 * (populations.card : ℝ) / δ)
      / (2 * (cutBudget η indecisionLimit εcov / 4) ^ 2),
    64 * Real.log (1 / α) / (εcov * (sig η * εcov / 4) ^ 2),
    64 * Real.log (256 * (populations.card : ℝ) / δ) / (εcov * (sig η * εcov / 4) ^ 2),
    Real.log (128 * (populations.card : ℝ)
        * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 1) / δ)
      / (2 * ((populations.card : ℝ) * flipBudget η populations indecisionLimit εcov δ) ^ 2),
    64 / εcov ]

/-- Rounding up a list of reals costs one apiece. -/
private lemma cast_sum_ceil_le : ∀ l : List ℝ,
    (((l.map (fun x => ⌈x⌉₊)).sum : ℕ) : ℝ)
      ≤ (l.map (fun x => max x 0)).sum + l.length
  | [] => by simp
  | x :: xs => by
    have hx := cast_ceil_le x
    have hxs := cast_sum_ceil_le xs
    simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.cast_add, Nat.cast_one]
    linarith

/-- The count is exactly its six tails, each rounded up, and the one rung the ladder always
has. -/
theorem prefCount_eq_sum (populations : Finset J)
    (η indecisionLimit εcov δ α pAP : ℝ) :
    prefCount η populations indecisionLimit εcov δ α pAP
      = ((prefTerms η populations indecisionLimit εcov δ α pAP).map
          (fun x => ⌈x⌉₊)).sum + 1 := by
  simp only [prefCount, prefTerms, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  ring

/-- The mechanical half of the bound: the count is under its six tails once each rounding is
paid for.  Nothing analytic happens here -- what the tails are worth is `prefCount_le_poly`. -/
theorem prefCount_le_terms (populations : Finset J)
    (η indecisionLimit εcov δ α pAP : ℝ) :
    (prefCount η populations indecisionLimit εcov δ α pAP : ℝ)
      ≤ ((prefTerms η populations indecisionLimit εcov δ α pAP).map
          (fun x => max x 0)).sum + 7 := by
  have h := cast_sum_ceil_le (prefTerms η populations indecisionLimit εcov δ α pAP)
  rw [prefCount_eq_sum]
  have hlen : (prefTerms η populations indecisionLimit εcov δ α pAP).length = 6 := by
    simp [prefTerms]
  rw [hlen] at h
  push_cast at h ⊢
  linarith

/-! ## Arithmetic scaffolding

Each tail is a log over a squared margin.  The six lemmas below are pure real arithmetic:
they take the shape of one tail, a lower bound on its margin and an upper bound on its log,
and return a multiple of the common quotient `c² · L / (s⁶ · B³)`.  Nothing here knows what
the tails are; `prefCount_le_poly` supplies the margins.
-/

private lemma le_mul2 {a b x y : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hx : a ≤ x) (hy : b ≤ y) :
    a * b ≤ x * y :=
  mul_le_mul hx hy hb (ha.trans hx)

private lemma le_mul3 {a b c x y z : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
    (hx : a ≤ x) (hy : b ≤ y) (hz : c ≤ z) : a * b * c ≤ x * y * z :=
  le_mul2 (mul_nonneg ha hb) hc (le_mul2 ha hb hx hy) hz

/-- `log 6 > 1`, which is what makes the shared log factor at least one. -/
private lemma one_le_log_of_six_le {T : ℝ} (hT : 6 ≤ T) : 1 ≤ Real.log T := by
  have h2 : Real.log ((2 : ℝ)⁻¹) ≤ (2 : ℝ)⁻¹ - 1 := Real.log_le_sub_one_of_pos (by norm_num)
  have h3 : Real.log ((3 : ℝ)⁻¹) ≤ (3 : ℝ)⁻¹ - 1 := Real.log_le_sub_one_of_pos (by norm_num)
  rw [Real.log_inv] at h2 h3
  have h6 : Real.log 6 = Real.log 2 + Real.log 3 := by
    rw [show (6 : ℝ) = 2 * 3 by norm_num, Real.log_mul (by norm_num) (by norm_num)]
  have hmono : Real.log 6 ≤ Real.log T := Real.log_le_log (by norm_num) hT
  linarith

/-- A log is under `n` copies of `log T` once its argument is under `T ^ n`. -/
private lemma log_le_of_le_pow {T Z : ℝ} (hZ : 0 < Z) (n : ℕ) (h : Z ≤ T ^ n) :
    Real.log Z ≤ n * Real.log T :=
  le_trans (Real.log_le_log hZ h) (le_of_eq (Real.log_pow T n))

/-- The screen's tail.  Its margin carries `B · s³ / c`, and squaring that is where `s⁶`,
`B²` and `c²` all come from; the third power of `B` is slack paid for by `B ≤ 1/2`. -/
private lemma screen_tail_le {c s B L X M : ℝ}
    (hs : 0 < s) (hB : 0 < B) (hB2 : B ≤ 1 / 2) (hL : 1 ≤ L)
    (hX : 0 ≤ X) (hXL : X ≤ 16 * L) (hM : 0 < M) (hMlb : 7 * B * s ^ 3 ≤ 1024 * c * M) :
    X / (2 * (M / 2) ^ 2) ≤ 393216 * (c ^ 2 * L / (s ^ 6 * B ^ 3)) := by
  rw [show 393216 * (c ^ 2 * L / (s ^ 6 * B ^ 3)) = 393216 * c ^ 2 * L / (s ^ 6 * B ^ 3) from
      by ring,
    div_le_div_iff₀ (mul_pos (by norm_num) (pow_pos (by linarith) 2))
      (mul_pos (pow_pos hs 6) (pow_pos hB 3))]
  have hsq : 49 * B ^ 2 * s ^ 6 ≤ 1048576 * c ^ 2 * M ^ 2 := by
    nlinarith [mul_self_le_mul_self
      (mul_nonneg (by linarith : (0 : ℝ) ≤ 7 * B) (pow_nonneg hs.le 3)) hMlb]
  have hXB : X * B ≤ 8 * L := by
    nlinarith [mul_nonneg hX (by linarith : (0 : ℝ) ≤ 1 / 2 - B)]
  have hLnn : (0 : ℝ) ≤ L := by linarith
  nlinarith [mul_le_mul_of_nonneg_right hXB
      (mul_nonneg (pow_nonneg hs.le 6) (sq_nonneg B)),
    mul_le_mul_of_nonneg_left hsq hLnn,
    mul_nonneg (mul_nonneg (sq_nonneg c) hLnn) (sq_nonneg M)]

/-- The cut tail: a log over the squared cut budget, which is at least `B / 64`. -/
private lemma cut_tail_le {c s B L R cut : ℝ}
    (hc : 1 ≤ c) (hs : 0 < s) (hs2 : s ≤ 1 / 2) (hB : 0 < B) (hB2 : B ≤ 1 / 2) (hL : 1 ≤ L)
    (hR : 0 ≤ R) (hRL : R ≤ 4 * L) (hcut : 0 < cut) (hcutlb : B ≤ 64 * cut) :
    R / (2 * (cut / 4) ^ 2) ≤ 2048 * (c ^ 2 * L / (s ^ 6 * B ^ 3)) := by
  rw [show 2048 * (c ^ 2 * L / (s ^ 6 * B ^ 3)) = 2048 * c ^ 2 * L / (s ^ 6 * B ^ 3) from
      by ring,
    div_le_div_iff₀ (mul_pos (by norm_num) (pow_pos (by linarith) 2))
      (mul_pos (pow_pos hs 6) (pow_pos hB 3))]
  have hs6 : s ^ 6 ≤ 1 / 64 := by
    have h := pow_le_pow_left₀ hs.le hs2 6; norm_num at h; linarith
  have hB3 : B ^ 3 ≤ B ^ 2 / 2 := by
    nlinarith [mul_nonneg (sq_nonneg B) (by linarith : (0 : ℝ) ≤ 1 / 2 - B)]
  have hcut2 : B ^ 2 ≤ 4096 * cut ^ 2 := by
    nlinarith [mul_self_le_mul_self hB.le hcutlb]
  have hLnn : (0 : ℝ) ≤ L := by linarith
  have hc2 : (1 : ℝ) ≤ c ^ 2 := by nlinarith
  have h1 : R * s ^ 6 ≤ L / 16 := by
    nlinarith [mul_nonneg hR (by linarith : (0 : ℝ) ≤ 1 / 64 - s ^ 6)]
  have h2 : R * s ^ 6 * B ^ 3 ≤ L / 16 * B ^ 3 :=
    mul_le_mul_of_nonneg_right h1 (pow_nonneg hB.le 3)
  have h3 : L * B ^ 3 ≤ L * (B ^ 2 / 2) := mul_le_mul_of_nonneg_left hB3 hLnn
  have h4 : c ^ 2 * L * B ^ 2 ≤ c ^ 2 * L * (4096 * cut ^ 2) :=
    mul_le_mul_of_nonneg_left hcut2 (mul_nonneg (sq_nonneg c) hLnn)
  have h5 : 1 * (L * B ^ 2) ≤ c ^ 2 * (L * B ^ 2) :=
    mul_le_mul_of_nonneg_right hc2 (mul_nonneg hLnn (sq_nonneg B))
  nlinarith [mul_nonneg (mul_nonneg (sq_nonneg c) hLnn) (sq_nonneg cut)]

/-- The coverage tails.  The extra `εcov` the denominator carries beside the squared margin
is why the scale enters cubed: `B ≤ εcov` pays for all three copies at once. -/
private lemma coverage_tail_le {c s B L P ε k K : ℝ}
    (hc : 1 ≤ c) (hs : 0 < s) (hs2 : s ≤ 1 / 2) (hB : 0 < B) (hBε : B ≤ ε) (hL : 1 ≤ L)
    (hP : 0 ≤ P) (hk : 0 ≤ k) (hPL : P ≤ k * L) (hK : 64 * k ≤ K) :
    64 * P / (ε * (s * ε / 4) ^ 2) ≤ K * (c ^ 2 * L / (s ^ 6 * B ^ 3)) := by
  have hε : 0 < ε := lt_of_lt_of_le hB hBε
  have hLnn : (0 : ℝ) ≤ L := by linarith
  have hQ : (0 : ℝ) ≤ c ^ 2 * L / (s ^ 6 * B ^ 3) :=
    div_nonneg (mul_nonneg (sq_nonneg c) hLnn)
      (mul_pos (pow_pos hs 6) (pow_pos hB 3)).le
  refine le_trans ?_ (mul_le_mul_of_nonneg_right hK hQ)
  rw [show 64 * k * (c ^ 2 * L / (s ^ 6 * B ^ 3)) = 64 * k * c ^ 2 * L / (s ^ 6 * B ^ 3) from
      by ring,
    div_le_div_iff₀
      (mul_pos hε (pow_pos (div_pos (mul_pos hs hε) (by norm_num)) 2))
      (mul_pos (pow_pos hs 6) (pow_pos hB 3))]
  have hs4 : s ^ 4 ≤ 1 / 16 := by
    have h := pow_le_pow_left₀ hs.le hs2 4; norm_num at h; linarith
  have hc2 : (1 : ℝ) ≤ c ^ 2 := by nlinarith
  have h1 : P * s ^ 4 ≤ k * L / 16 := by
    nlinarith [mul_nonneg hP (by linarith : (0 : ℝ) ≤ 1 / 16 - s ^ 4)]
  have h2 : B ^ 3 ≤ ε ^ 3 := pow_le_pow_left₀ hB.le hBε 3
  have h3 : P * s ^ 4 * B ^ 3 ≤ k * L / 16 * ε ^ 3 :=
    le_mul2 (mul_nonneg hP (pow_nonneg hs.le 4)) (pow_nonneg hB.le 3) h1 h2
  have h4 : P * s ^ 4 * B ^ 3 * s ^ 2 ≤ k * L / 16 * ε ^ 3 * s ^ 2 :=
    mul_le_mul_of_nonneg_right h3 (sq_nonneg s)
  have h5 : 1 * (k * L * s ^ 2 * ε ^ 3) ≤ c ^ 2 * (k * L * s ^ 2 * ε ^ 3) :=
    mul_le_mul_of_nonneg_right hc2
      (mul_nonneg (mul_nonneg (mul_nonneg hk hLnn) (sq_nonneg s)) (pow_nonneg hε.le 3))
  linarith

/-- The flip tail: a log over the squared per-member flip budget, which carries `B · s`. -/
private lemma flip_tail_le {c s B L Y W : ℝ}
    (hc : 1 ≤ c) (hs : 0 < s) (hs2 : s ≤ 1 / 2) (hB : 0 < B) (hB2 : B ≤ 1 / 2) (hL : 1 ≤ L)
    (hY : 0 ≤ Y) (hYL : Y ≤ 10 * L) (hW : 0 < W) (hWlb : 7 * B * s ≤ 1920 * W) :
    Y / (2 * W ^ 2) ≤ 16384 * (c ^ 2 * L / (s ^ 6 * B ^ 3)) := by
  rw [show 16384 * (c ^ 2 * L / (s ^ 6 * B ^ 3)) = 16384 * c ^ 2 * L / (s ^ 6 * B ^ 3) from
      by ring,
    div_le_div_iff₀ (mul_pos (by norm_num) (pow_pos hW 2))
      (mul_pos (pow_pos hs 6) (pow_pos hB 3))]
  have hLnn : (0 : ℝ) ≤ L := by linarith
  have hc2 : (1 : ℝ) ≤ c ^ 2 := by nlinarith
  have hsq : 49 * B ^ 2 * s ^ 2 ≤ 3686400 * W ^ 2 := by
    nlinarith [mul_self_le_mul_self
      (mul_nonneg (by linarith : (0 : ℝ) ≤ 7 * B) hs.le) hWlb]
  have hs4 : s ^ 4 ≤ 1 / 16 := by
    have h := pow_le_pow_left₀ hs.le hs2 4; norm_num at h; linarith
  have hs6 : s ^ 6 ≤ s ^ 2 / 16 := by
    nlinarith [mul_le_mul_of_nonneg_right hs4 (sq_nonneg s)]
  have hB3 : B ^ 3 ≤ B ^ 2 / 2 := by
    nlinarith [mul_nonneg (sq_nonneg B) (by linarith : (0 : ℝ) ≤ 1 / 2 - B)]
  have h1 : s ^ 6 * B ^ 3 ≤ s ^ 2 / 16 * (B ^ 2 / 2) :=
    le_mul2 (pow_nonneg hs.le 6) (pow_nonneg hB.le 3) hs6 hB3
  have h2 : Y * (s ^ 6 * B ^ 3) ≤ 10 * L * (s ^ 2 / 16 * (B ^ 2 / 2)) :=
    le_mul2 hY (mul_nonneg (pow_nonneg hs.le 6) (pow_nonneg hB.le 3)) hYL h1
  have h3 : L * (49 * B ^ 2 * s ^ 2) ≤ L * (3686400 * W ^ 2) :=
    mul_le_mul_of_nonneg_left hsq hLnn
  have h4 : 1 * (L * W ^ 2) ≤ c ^ 2 * (L * W ^ 2) :=
    mul_le_mul_of_nonneg_right hc2 (mul_nonneg hLnn (sq_nonneg W))
  nlinarith [mul_nonneg (mul_nonneg (sq_nonneg c) hLnn) (sq_nonneg W)]

/-- The rate tail: the one term that is not a log at all. -/
private lemma rate_tail_le {c s B L ε : ℝ}
    (hc : 1 ≤ c) (hs : 0 < s) (hs2 : s ≤ 1 / 2) (hB : 0 < B) (hB2 : B ≤ 1 / 2) (hBε : B ≤ ε)
    (hL : 1 ≤ L) : 64 / ε ≤ c ^ 2 * L / (s ^ 6 * B ^ 3) := by
  have hε : 0 < ε := lt_of_lt_of_le hB hBε
  rw [div_le_div_iff₀ hε (mul_pos (pow_pos hs 6) (pow_pos hB 3))]
  have hLnn : (0 : ℝ) ≤ L := by linarith
  have hs6 : s ^ 6 ≤ 1 / 64 := by
    have h := pow_le_pow_left₀ hs.le hs2 6; norm_num at h; linarith
  have hBsq : B ^ 2 ≤ 1 / 4 := by
    have h := pow_le_pow_left₀ hB.le hB2 2; norm_num at h; linarith
  have hB3 : B ^ 3 ≤ B / 4 := by
    nlinarith [mul_le_mul_of_nonneg_right hBsq hB.le]
  have h1 : s ^ 6 * B ^ 3 ≤ 1 / 64 * (B / 4) :=
    le_mul2 (pow_nonneg hs.le 6) (pow_nonneg hB.le 3) hs6 hB3
  have hc2L : (1 : ℝ) ≤ c ^ 2 * L := by nlinarith
  nlinarith [mul_le_mul_of_nonneg_right hc2L hε.le]

/-- The budget is polynomial in the reciprocals of the rates it resolves.

`sig` enters at the sixth power and `budgetScale` at the third because the screen's tail is
the binding one: its margin is `cutBudget · sig³`, and a deviation bound
squares the margin it is given.  The population count enters squared for the same reason --
`flipBudget` divides by it, and `screenMargin` inherits that.  The scale is cubed rather than
squared because the coverage tails divide by `εcov` once before squaring a margin that already
carries `εcov`.

`0 ≤ η` is what bounds `sig η` above; without it the noise rate may be negative, `sig` may be
arbitrarily large, and the `sig ^ 6` in the denominator sends the bound below the count.

The `524288` is not tight -- the worst term reaches about a quarter of it.  It is a constant
that covers every term at once, which is what makes the statement one bound rather than six. -/
theorem prefCount_le_poly (populations : Finset J)
    (η indecisionLimit εcov δ α pAP : ℝ)
    (hsig : 0 < sig η) (hη : 0 ≤ η) (hpop : populations.Nonempty)
    (hind : 0 < indecisionLimit) (hεcov : 0 < εcov) (hεcov1 : εcov ≤ 1)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hα : 0 < α) (hα1 : α < 1)
    (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) :
    (prefCount η populations indecisionLimit εcov δ α pAP : ℝ)
      ≤ 524288 * (populations.card : ℝ) ^ 2
        * budgetLog populations η indecisionLimit εcov δ α pAP
        / (sig η ^ 6 * budgetScale η indecisionLimit εcov ^ 3) := by
  have hc1 : (1 : ℝ) ≤ (populations.card : ℝ) := Nat.one_le_cast.2 (Finset.card_pos.2 hpop)
  set c : ℝ := (populations.card : ℝ) with hcdef
  clear_value c
  have hcpos : (0 : ℝ) < c := by linarith only [hc1]
  have hcne : c ≠ 0 := ne_of_gt hcpos
  set s : ℝ := sig η with hsdef
  clear_value s
  have hsval : s = 1 / 2 - η := by rw [hsdef, sig]
  have hs2 : s ≤ 1 / 2 := by rw [hsval]; linarith only [hη]
  have hηlt : η < 1 / 2 := by rw [hsval] at hsig; linarith only [hsig]
  -- the scale, and the three rates it sits under
  set B : ℝ := budgetScale η indecisionLimit εcov with hBdef
  clear_value B
  have hBε : B ≤ εcov := by simp only [hBdef, budgetScale]; exact min_le_left _ _
  have hBs : B ≤ s := by
    simp only [hBdef, budgetScale, ← hsdef]
    exact le_trans (min_le_right _ _) (min_le_left _ _)
  have hBind : B ≤ indecisionLimit := by
    simp only [hBdef, budgetScale]; exact le_trans (min_le_right _ _) (min_le_right _ _)
  have hBpos : 0 < B := by
    simp only [hBdef, budgetScale, ← hsdef]
    exact lt_min hεcov (lt_min hsig hind)
  have hB2 : B ≤ 1 / 2 := le_trans hBs hs2
  have hB3pos : (0 : ℝ) < B ^ 3 := pow_pos hBpos 3
  have hB3le : B ^ 3 ≤ 1 / 8 := by
    have h := pow_le_pow_left₀ hBpos.le hB2 3; norm_num at h; linarith only [h]
  -- the cut budget is at least a sixty-fourth of the scale
  set cut : ℝ := cutBudget η indecisionLimit εcov with hcutdef
  clear_value cut
  have hcutlb : B ≤ 64 * cut := by
    have h : B / 64 ≤ cut := by
      simp only [hcutdef, cutBudget, ← hsdef]
      exact le_min (by linarith only [hBε, hεcov])
        (le_min (by linarith only [hBs]) (by linarith only [hBind, hind]))
    linarith only [h]
  have hcutpos : 0 < cut := by linarith only [hcutlb, hBpos]
  -- what the vote absorbs is at least seven tenths of the signal
  have hf : 7 * s ≤ 10 * flipFrac η := by
    have hne : (1 : ℝ) - η ≠ 0 := by intro h; linarith only [h, hηlt]
    have he : 10 * flipFrac η = 7 * s / (1 - η) := by
      simp only [flipFrac, ← hsdef]; field_simp
    rw [he, le_div_iff₀ (by linarith only [hηlt] : (0 : ℝ) < 1 - η)]
    linarith only [mul_nonneg (by linarith only [hsig] : (0 : ℝ) ≤ 7 * s) hη]
  have hcutf : 7 * B * s ≤ 640 * (cut * flipFrac η) := by
    have h : B * (7 * s) ≤ 64 * cut * (10 * flipFrac η) :=
      le_mul2 hBpos.le (by linarith only [hsig]) hcutlb hf
    linarith only [h]
  -- the screen's margin and the family's flip budget
  have hMeq : 1024 * c * screenMargin η populations indecisionLimit εcov δ
      = 640 * (cut * flipFrac η) * s ^ 2 := by
    simp only [screenMargin, flipBudget, ← hsdef, ← hcdef, ← hcutdef]
    field_simp
    ring
  have hMlb : 7 * B * s ^ 3
      ≤ 1024 * c * screenMargin η populations indecisionLimit εcov δ := by
    rw [hMeq]
    linarith only [mul_le_mul_of_nonneg_right hcutf (sq_nonneg s)]
  have hMpos : 0 < screenMargin η populations indecisionLimit εcov δ := by
    have h7 : (0 : ℝ) < 7 * B * s ^ 3 := by positivity
    nlinarith only [lt_of_lt_of_le h7 hMlb, hcpos]
  have hWeq : c * flipBudget η populations indecisionLimit εcov δ
      = cut * flipFrac η / 3 := by
    simp only [flipBudget, ← hcdef, ← hcutdef]; field_simp
  have hWlb : 7 * B * s ≤ 1920 * (c * flipBudget η populations indecisionLimit εcov δ) := by
    rw [hWeq]; linarith only [hcutf]
  have hWpos : 0 < c * flipBudget η populations indecisionLimit εcov δ := by
    have h7 : (0 : ℝ) < 7 * B * s := by positivity
    linarith only [h7, hWlb]
  -- the shared log factor
  set d : ℝ := δ * α * pAP * B with hddef
  clear_value d
  have hdpos : 0 < d := by
    rw [hddef]; exact mul_pos (mul_pos (mul_pos hδ hα) hpAP) hBpos
  have hδα : δ * α ≤ 1 := by linarith only [le_mul2 hδ.le hα.le hδ1 hα1.le]
  have hαp : α * pAP ≤ 1 := by linarith only [le_mul2 hα.le hpAP.le hα1.le hpAP1]
  have hδp : δ * pAP ≤ 1 := by linarith only [le_mul2 hδ.le hpAP.le hδ1 hpAP1]
  have hdap : δ * α * pAP ≤ 1 := by
    linarith only [le_mul2 (mul_pos hδ hα).le hpAP.le hδα hpAP1]
  have hapB : α * pAP * B ≤ 1 / 2 := by
    linarith only [le_mul2 (mul_pos hα hpAP).le hBpos.le hαp hB2]
  have hdpB : δ * pAP * B ≤ 1 / 2 := by
    linarith only [le_mul2 (mul_pos hδ hpAP).le hBpos.le hδp hB2]
  have hdaB : δ * α * B ≤ 1 / 2 := by
    linarith only [le_mul2 (mul_pos hδ hα).le hBpos.le hδα hB2]
  have hd2 : d ≤ 1 / 2 := by
    rw [hddef]
    linarith only [le_mul2 (mul_pos (mul_pos hδ hα) hpAP).le hBpos.le hdap hB2]
  set T : ℝ := (c + 2) / d with hTdef
  clear_value T
  have hT6 : 6 ≤ T := by
    rw [hTdef, le_div_iff₀ hdpos]; linarith only [hd2, hc1]
  have hTpos : 0 < T := by linarith only [hT6]
  have hTc : 2 * c + 4 ≤ T := by
    rw [hTdef, le_div_iff₀ hdpos]
    linarith only [mul_le_mul_of_nonneg_left hd2 (by linarith only [hc1] : (0 : ℝ) ≤ 2 * c + 4)]
  have hTδ : 6 ≤ T * δ := by
    rw [hTdef, div_mul_eq_mul_div, le_div_iff₀ hdpos, hddef]
    linarith only [mul_le_mul_of_nonneg_left hapB hδ.le,
      mul_le_mul_of_nonneg_right hc1 hδ.le]
  have hTα : 6 ≤ T * α := by
    rw [hTdef, div_mul_eq_mul_div, le_div_iff₀ hdpos, hddef]
    linarith only [mul_le_mul_of_nonneg_left hdpB hα.le,
      mul_le_mul_of_nonneg_right hc1 hα.le]
  have hTp : 6 ≤ T * pAP := by
    rw [hTdef, div_mul_eq_mul_div, le_div_iff₀ hdpos, hddef]
    linarith only [mul_le_mul_of_nonneg_left hdaB hpAP.le,
      mul_le_mul_of_nonneg_right hc1 hpAP.le]
  have hTB : 3 ≤ T * B := by
    rw [hTdef, div_mul_eq_mul_div, le_div_iff₀ hdpos, hddef]
    linarith only [mul_le_mul_of_nonneg_right hdap hBpos.le,
      mul_le_mul_of_nonneg_right (by linarith only [hc1] : (3 : ℝ) ≤ c + 2) hBpos.le]
  -- `T ^ 4` covers `256 c / δ`, and `T ^ 6` covers the pool count
  have hpow4 : 256 * c ≤ T ^ 4 * δ := by
    have e1 : (6 : ℝ) * 6 ≤ T * δ * T := le_mul2 (by norm_num) (by norm_num) hTδ hT6
    have e2 : (6 : ℝ) * (2 * c + 4) ≤ T * T :=
      le_mul2 (by norm_num) (by linarith only [hc1]) hT6 hTc
    linarith only [le_mul2 (by norm_num : (0 : ℝ) ≤ 6 * 6)
      (by linarith only [hc1] : (0 : ℝ) ≤ 6 * (2 * c + 4)) e1 e2, hc1]
  have hpow6B : 4278 ≤ T ^ 6 * (B ^ 3 * pAP) := by
    have e1 : (3 : ℝ) * 3 * 3 ≤ T * B * (T * B) * (T * B) :=
      le_mul3 (by norm_num) (by norm_num) (by norm_num) hTB hTB hTB
    have e2 : (6 : ℝ) * 6 * 6 ≤ T * pAP * T * T :=
      le_mul3 (by norm_num) (by norm_num) (by norm_num) hTp hT6 hT6
    linarith only [le_mul2 (by norm_num : (0 : ℝ) ≤ 3 * 3 * 3)
      (by norm_num : (0 : ℝ) ≤ 6 * 6 * 6) e1 e2]
  have hpow6C : 768 * c ≤ T ^ 6 * (δ * pAP ^ 2) := by
    have e1 : (6 : ℝ) * 6 * 6 ≤ T * δ * (T * pAP) * (T * pAP) :=
      le_mul3 (by norm_num) (by norm_num) (by norm_num) hTδ hTp hTp
    have e2 : (6 : ℝ) * 6 * (2 * c + 4) ≤ T * T * T :=
      le_mul3 (by norm_num) (by norm_num) (by linarith only [hc1]) hT6 hT6 hTc
    linarith only [le_mul2 (by norm_num : (0 : ℝ) ≤ 6 * 6 * 6)
      (by linarith only [hc1] : (0 : ℝ) ≤ 6 * 6 * (2 * c + 4)) e1 e2, hc1]
  have hpow6 : (46656 : ℝ) ≤ T ^ 6 := by
    have h := pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 6) hT6 6
    norm_num at h; linarith only [h]
  -- the family count, then the pool count
  have hlogcut : Real.log (2 / cut) * B ≤ 128 := by
    have h1 : Real.log (2 / cut) ≤ 2 / cut - 1 :=
      Real.log_le_sub_one_of_pos (div_pos (by norm_num) hcutpos)
    have h2 : 2 / cut ≤ 128 / B := by
      rw [div_le_div_iff₀ hcutpos hBpos]; linarith only [hcutlb]
    have h4 : Real.log (2 / cut) * B ≤ 128 / B * B :=
      mul_le_mul_of_nonneg_right (by linarith only [h1, h2]) hBpos.le
    rwa [div_mul_cancel₀ _ hBpos.ne'] at h4
  have hfcval : (famCount η populations indecisionLimit εcov δ : ℝ)
      = 2 * ((⌈Real.log (2 / cut) / (4 * voteSlack η ^ 2)⌉₊ : ℝ) + 1) := by
    rw [hcutdef, famCount]
    push_cast
    ring
  have hfc : (famCount η populations indecisionLimit εcov δ : ℝ)
      ≤ 2 * (3200 / (9 * B ^ 3)) + 4 := by
    have hvote : 4 * voteSlack η ^ 2 = 9 * s ^ 2 / 25 := by
      simp only [voteSlack, ← hsdef]; ring
    have hs2pos : (0 : ℝ) < s ^ 2 := pow_pos hsig 2
    have hz : Real.log (2 / cut) / (4 * voteSlack η ^ 2) ≤ 3200 / (9 * B ^ 3) := by
      rw [hvote, div_le_div_iff₀ (by linarith only [hs2pos]) (by linarith only [hB3pos])]
      linarith only [mul_le_mul_of_nonneg_right hlogcut
          (by positivity : (0 : ℝ) ≤ 9 * B ^ 2),
        pow_le_pow_left₀ hBpos.le hBs 2]
    have hceil := ceil_le_of_le hz (div_nonneg (by norm_num) (by linarith only [hB3pos]))
    rw [hfcval]
    linarith only [hceil]
  have hfcB : (famCount η populations indecisionLimit εcov δ : ℝ) * B ^ 3
      ≤ 6400 / 9 + 4 * B ^ 3 := by
    have h := mul_le_mul_of_nonneg_right hfc hB3pos.le
    have e : (2 * (3200 / (9 * B ^ 3)) + 4) * B ^ 3 = 6400 / 9 + 4 * B ^ 3 := by
      field_simp
      ring
    rw [e] at h
    exact h
  have hA : 2 * ((famCount η populations indecisionLimit εcov δ : ℝ) + 1) / pAP
      ≤ 1426 / (B ^ 3 * pAP) := by
    rw [div_le_div_iff₀ hpAP (mul_pos hB3pos hpAP)]
    linarith only [mul_le_mul_of_nonneg_left hfcB hpAP.le,
      mul_le_mul_of_nonneg_left hB3le hpAP.le, hpAP.le]
  have hClog : Real.log (128 * c / δ) * δ ≤ 128 * c := by
    have h1 : Real.log (128 * c / δ) ≤ 128 * c / δ - 1 :=
      Real.log_le_sub_one_of_pos (div_pos (by linarith only [hcpos]) hδ)
    have h2 : Real.log (128 * c / δ) * δ ≤ 128 * c / δ * δ :=
      mul_le_mul_of_nonneg_right (by linarith only [h1]) hδ.le
    rwa [div_mul_cancel₀ _ hδ.ne'] at h2
  have hClog0 : 0 ≤ Real.log (128 * c / δ) :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ]; linarith only [hδ1, hc1])
  have hC : Real.log (128 * c / δ) / (2 * (pAP / 2) ^ 2) ≤ 256 * c / (δ * pAP ^ 2) := by
    rw [div_le_div_iff₀ (mul_pos (by norm_num) (pow_pos (by linarith only [hpAP]) 2))
      (mul_pos hδ (pow_pos hpAP 2))]
    linarith only [mul_le_mul_of_nonneg_right hClog (sq_nonneg pAP)]
  have hpoolval : (poolCount η populations indecisionLimit εcov δ pAP : ℝ)
      = (⌈2 * ((famCount η populations indecisionLimit εcov δ : ℝ) + 1) / pAP⌉₊ : ℝ)
        + (⌈Real.log (128 * c / δ) / (2 * (pAP / 2) ^ 2)⌉₊ : ℝ) := by
    rw [hcdef, poolCount]
    push_cast
    ring
  have hpool : (poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 2
      ≤ 1426 / (B ^ 3 * pAP) + 256 * c / (δ * pAP ^ 2) + 4 := by
    have h1 := ceil_le_of_le hA (div_nonneg (by norm_num) (mul_pos hB3pos hpAP).le)
    have h2 := ceil_le_of_le hC
      (div_nonneg (by linarith only [hcpos]) (mul_pos hδ (pow_pos hpAP 2)).le)
    rw [hpoolval]
    linarith only [h1, h2]
  have hpc6 : (poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 2 ≤ T ^ 6 := by
    have g1 : 1426 / (B ^ 3 * pAP) ≤ T ^ 6 / 3 := by
      rw [div_le_div_iff₀ (mul_pos hB3pos hpAP) (by norm_num)]; linarith only [hpow6B]
    have g2 : 256 * c / (δ * pAP ^ 2) ≤ T ^ 6 / 3 := by
      rw [div_le_div_iff₀ (mul_pos hδ (pow_pos hpAP 2)) (by norm_num)]; linarith only [hpow6C]
    have g3 : (4 : ℝ) ≤ T ^ 6 / 3 := by linarith only [hpow6]
    linarith only [hpool, g1, g2, g3]
  have hpcnn : (0 : ℝ) ≤ (poolCount η populations indecisionLimit εcov δ pAP : ℝ) :=
    Nat.cast_nonneg _
  have hpcsq : (1 : ℝ) ≤ ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 2) ^ 2 := by
    nlinarith only [hpcnn]
  -- the logs that appear in the tails, each against the shared one
  set L : ℝ := Real.log T with hLdef
  clear_value L
  have hL1 : 1 ≤ L := by rw [hLdef]; exact one_le_log_of_six_le hT6
  have hLeq : budgetLog populations η indecisionLimit εcov δ α pAP = L := by
    rw [hLdef, hTdef, hddef, hBdef, hcdef, budgetLog]
  have hXpos : (0 : ℝ) < 128 * c
      * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 2) ^ 2 / δ := by
    positivity
  have hYpos : (0 : ℝ) < 128 * c
      * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 1) / δ := by
    positivity
  have h128 : 128 * c / δ ≤ T ^ 4 := by
    rw [div_le_iff₀ hδ]; linarith only [hpow4, hcpos]
  have hX : Real.log (128 * c
      * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 2) ^ 2 / δ) ≤ 16 * L := by
    have harg : 128 * c * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 2) ^ 2 / δ
        ≤ T ^ 16 := by
      have e : 128 * c * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 2) ^ 2 / δ
          = 128 * c / δ * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 2) ^ 2 := by
        ring
      have h2 : ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 2) ^ 2 ≤ (T ^ 6) ^ 2 :=
        pow_le_pow_left₀ (by linarith only [hpcnn]) hpc6 2
      rw [e]
      calc 128 * c / δ * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 2) ^ 2
          ≤ T ^ 4 * (T ^ 6) ^ 2 :=
            le_mul2 (div_nonneg (by linarith only [hcpos]) hδ.le) (sq_nonneg _) h128 h2
        _ = T ^ 16 := by ring
    have h := log_le_of_le_pow hXpos 16 harg
    rw [hLdef]
    push_cast at h
    linarith only [h]
  have hY : Real.log (128 * c
      * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 1) / δ) ≤ 10 * L := by
    have harg : 128 * c * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 1) / δ
        ≤ T ^ 10 := by
      have e : 128 * c * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 1) / δ
          = 128 * c / δ * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 1) := by
        ring
      have h2 : (poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 1 ≤ T ^ 6 := by
        linarith only [hpc6]
      rw [e]
      calc 128 * c / δ * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 1)
          ≤ T ^ 4 * T ^ 6 :=
            le_mul2 (div_nonneg (by linarith only [hcpos]) hδ.le)
              (by linarith only [hpcnn]) h128 h2
        _ = T ^ 10 := by ring
    have h := log_le_of_le_pow hYpos 10 harg
    rw [hLdef]
    push_cast at h
    linarith only [h]
  have hXnn : 0 ≤ Real.log (128 * c
      * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 2) ^ 2 / δ) := by
    refine Real.log_nonneg ?_
    rw [le_div_iff₀ hδ]
    linarith only [le_mul2 zero_le_one zero_le_one hc1 hpcsq, hδ1]
  have hYnn : 0 ≤ Real.log (128 * c
      * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 1) / δ) := by
    refine Real.log_nonneg ?_
    rw [le_div_iff₀ hδ]
    linarith only [le_mul2 zero_le_one zero_le_one hc1
      (by linarith only [hpcnn] :
        (1 : ℝ) ≤ (poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 1), hδ1]
  have hα0 : 0 ≤ Real.log (1 / α) :=
    Real.log_nonneg (by rw [le_div_iff₀ hα]; linarith only [hα1])
  have hαL : Real.log (1 / α) ≤ 1 * L := by
    have h : 1 / α ≤ T := by rw [div_le_iff₀ hα]; linarith only [hTα]
    have h' := log_le_of_le_pow (div_pos one_pos hα) 1 (by simpa using h)
    rw [hLdef]
    push_cast at h'
    linarith only [h']
  have h256nn : 0 ≤ Real.log (256 * c / δ) :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ]; linarith only [hδ1, hc1])
  have h256L : Real.log (256 * c / δ) ≤ 4 * L := by
    have h : 256 * c / δ ≤ T ^ 4 := by rw [div_le_iff₀ hδ]; linarith only [hpow4]
    have h' := log_le_of_le_pow (div_pos (by linarith only [hcpos]) hδ) 4 h
    rw [hLdef]
    push_cast at h'
    linarith only [h']
  have h128L : Real.log (128 * c / δ) ≤ 4 * L := by
    have h' := log_le_of_le_pow (div_pos (by linarith only [hcpos]) hδ) 4 h128
    rw [hLdef]
    push_cast at h'
    linarith only [h']
  -- the six tails, each against the common quotient
  have hQnn : (0 : ℝ) ≤ c ^ 2 * L / (s ^ 6 * B ^ 3) :=
    div_nonneg (mul_nonneg (sq_nonneg c) (by linarith only [hL1]))
      (mul_pos (pow_pos hsig 6) (pow_pos hBpos 3)).le
  have t1 := screen_tail_le (c := c) (s := s) (B := B) (L := L) hsig hBpos hB2 hL1 hXnn hX
    hMpos hMlb
  have t2 := cut_tail_le (c := c) (s := s) (B := B) (L := L) hc1 hsig hs2 hBpos hB2 hL1
    hClog0 h128L hcutpos hcutlb
  have t3 := coverage_tail_le (c := c) (s := s) (B := B) (L := L) (ε := εcov) (k := 1) (K := 64)
    hc1 hsig hs2 hBpos hBε hL1 hα0 (by norm_num) hαL (by norm_num)
  have t4 := coverage_tail_le (c := c) (s := s) (B := B) (L := L) (ε := εcov) (k := 4) (K := 256)
    hc1 hsig hs2 hBpos hBε hL1 h256nn (by norm_num) h256L (by norm_num)
  have t5 := flip_tail_le (c := c) (s := s) (B := B) (L := L) hc1 hsig hs2 hBpos hB2 hL1
    hYnn hY hWpos hWlb
  have t6 := rate_tail_le (c := c) (s := s) (B := B) (L := L) hc1 hsig hs2 hBpos hB2 hBε hL1
  have t7 : (7 : ℝ) ≤ c ^ 2 * L / (s ^ 6 * B ^ 3) := by
    rw [le_div_iff₀ (mul_pos (pow_pos hsig 6) (pow_pos hBpos 3))]
    have hs6 : s ^ 6 ≤ 1 / 64 := by
      have h := pow_le_pow_left₀ hsig.le hs2 6; norm_num at h; linarith only [h]
    have hc2 : (1 : ℝ) ≤ c ^ 2 := by nlinarith only [hc1]
    have hc2L : (1 : ℝ) ≤ c ^ 2 * L := by
      linarith only [le_mul2 zero_le_one zero_le_one hc2 hL1]
    linarith only [le_mul2 (pow_nonneg hsig.le 6) (pow_nonneg hBpos.le 3) hs6 hB3le, hc2L]
  -- and the sum
  have hbound := prefCount_le_terms populations η indecisionLimit εcov δ α pAP
  simp only [prefTerms, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    ← hcdef, ← hsdef, ← hcutdef] at hbound
  have m1 := max_le t1 (mul_nonneg (by norm_num) hQnn)
  have m2 := max_le t2 (mul_nonneg (by norm_num) hQnn)
  have m3 := max_le t3 (mul_nonneg (by norm_num) hQnn)
  have m4 := max_le t4 (mul_nonneg (by norm_num) hQnn)
  have m5 := max_le t5 (mul_nonneg (by norm_num) hQnn)
  have m6 := max_le t6 hQnn
  have hfin : 524288 * c ^ 2 * L / (s ^ 6 * B ^ 3)
      = 524288 * (c ^ 2 * L / (s ^ 6 * B ^ 3)) := by ring
  rw [hLeq, hfin]
  linarith only [hbound, m1, m2, m3, m4, m5, m6, t7, hQnn]

/-- The ladder is `log` of the budget, so any bound on the budget bounds the number of times
the loop runs the gate, logarithmically. -/
theorem ladderLen_le_of_prefCount_le (populations : Finset J)
    (η indecisionLimit εcov δ α pAP : ℝ) {N : ℕ}
    (h : prefCount η populations indecisionLimit εcov δ α pAP ≤ N) :
    ladderLen η populations indecisionLimit εcov δ α pAP ≤ Nat.log 2 N + 1 :=
  Nat.succ_le_succ (Nat.log_mono_right h)

open MeasureTheory ProbabilityTheory in
/-- From `clustering_correct`, with the schedule and collision cap it names as the witnesses,
and `prefCount_le_poly` for the cost. -/
theorem clustering_guarantee_of_correct : ClusteringGuarantee := by
  refine ⟨524288, ?_⟩
  intro Ω _ μ _ S _ J _ O populations Pre η₀ indecisionLimit εcov α δ pAP
    hηle hη₀ hpop hflat hpAPPositive hindLim hind1 hαpos hα hεcov hε1 hδ hδ1
  classical
  have hsig : 0 < sig η₀ := by simp only [sig]; linarith
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hηle
  refine ⟨collisionCap η₀ populations indecisionLimit εcov δ α pAP, ?_, ?_⟩
  · simp only [collisionCap]
    positivity
  intro D Dsf hD hDsf hsupp hpAPBound ρ hρ hρcap hρsf
  haveI := hD
  haveI := hDsf
  refine ⟨stoppable η₀ populations indecisionLimit εcov δ α pAP ρ (collisionMass Dsf), ?_, ?_⟩
  · -- Every rung's count is the top one halved, and the top one is what `budgetCap` bounds.
    intro B hB
    have hpAP1 : pAP ≤ 1 := le_trans hpAPBound measureReal_le_one
    have hsched : B ∈ schedule η₀ populations indecisionLimit εcov δ α pAP :=
      Finset.mem_of_mem_filter _ hB
    obtain ⟨i, _, rfl⟩ := Finset.mem_image.1 hsched
    refine le_trans ?_ (prefCount_le_poly populations η₀ indecisionLimit εcov δ α pAP
      hsig hη0 hpop hindLim hεcov hε1 hδ hδ1 hαpos (by linarith) hpAPPositive hpAP1)
    exact_mod_cast Nat.div_le_self _ _
  exact clustering_correct O populations D Dsf Pre η₀ indecisionLimit εcov α δ ρ pAP 524288
    hηle hη₀ hpop hflat hsupp hρ hpAPPositive hpAPBound hindLim hind1 hαpos hα hεcov hε1 hδ
    (prefCount_le_poly populations η₀ indecisionLimit εcov δ α pAP hsig hη0 hpop
      hindLim hεcov hε1 hδ hδ1 hαpos (by linarith) hpAPPositive
      (le_trans hpAPBound measureReal_le_one))
    hρcap hρsf

end OrthoDFA
