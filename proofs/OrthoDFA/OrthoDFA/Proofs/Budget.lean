import OrthoDFA.Proofs.Lloyd
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# What the budget costs

`prefCount` is the count the round's tails ask for, and `ClusteringCorrect` quantifies over
the states the schedule built from it.  Written out it is a sum of six ceilings, each a log
over the square of some margin, and nothing in the statement says how those margins depend on
the rates they resolve.  This file says it: every term is bounded by one polynomial in the
reciprocals of `sig η` and `budgetScale`, so the budget is `log` in the failure probability and
polynomial in everything else.

The two scales are not independent.  `screenMargin` carries `cutBudget · sig²`, and the tail
it sits in squares it, so `sig` enters at the fourth power — and `cutBudget` is itself a minimum
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
    (indecisionLimit εcov δ α pAP crossLimit : ℝ) : List ℝ :=
  [ Real.log (128 * (populations.card : ℝ)
      * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 2) ^ 2 / δ)
      / (2 * (screenMargin η populations indecisionLimit εcov δ / 2) ^ 2),
    Real.log (128 * (populations.card : ℝ) / δ)
      / (2 * (cutBudget η indecisionLimit εcov / 32) ^ 2),
    32 * Real.log (1 / α) / sig η ^ 2,
    32 * Real.log (256 * (populations.card : ℝ) / δ) / sig η ^ 2,
    Real.log (128 * (populations.card : ℝ)
        * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ)
      / (2 * (3 * cutBudget η indecisionLimit εcov * flipFrac η / 32) ^ 2),
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
    (η indecisionLimit εcov δ α pAP crossLimit : ℝ) :
    prefCount η populations indecisionLimit εcov δ α pAP crossLimit
      = ((prefTerms η populations indecisionLimit εcov δ α pAP crossLimit).map
          (fun x => ⌈x⌉₊)).sum + 1 := by
  simp only [prefCount, prefTerms, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  ring

/-- The mechanical half of the bound: the count is under its six tails once each rounding is
paid for.  Nothing analytic happens here -- what the tails are worth is `prefCount_le_poly`. -/
theorem prefCount_le_terms (populations : Finset J)
    (η indecisionLimit εcov δ α pAP crossLimit : ℝ) :
    (prefCount η populations indecisionLimit εcov δ α pAP crossLimit : ℝ)
      ≤ ((prefTerms η populations indecisionLimit εcov δ α pAP crossLimit).map
          (fun x => max x 0)).sum + 7 := by
  have h :=
    cast_sum_ceil_le (prefTerms η populations indecisionLimit εcov δ α pAP crossLimit)
  rw [prefCount_eq_sum]
  have hlen :
      (prefTerms η populations indecisionLimit εcov δ α pAP crossLimit).length = 6 := by
    simp [prefTerms]
  rw [hlen] at h
  push_cast at h ⊢
  linarith

/-! ## Arithmetic scaffolding

Each tail is a log over a squared margin.  `tail_le` turns one into a multiple of a common
quotient `N · L / D`, given how many copies of `L` the log is and how the margin compares with
`D`; the theorems below supply both.
-/

private lemma le_mul2 {a b x y : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hx : a ≤ x) (hy : b ≤ y) :
    a * b ≤ x * y :=
  mul_le_mul hx hy hb (ha.trans hx)

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

private lemma half_le_log_two : 1 / 2 ≤ Real.log 2 := by
  have h := Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 2⁻¹)
  rw [Real.log_inv] at h
  linarith only [h]

/-- A tail `V / G` whose log is `w` copies of `L`, against the quotient `N · L / D`. -/
private lemma tail_le {V L G N D w K : ℝ} (hV : V ≤ w * L) (hL : 0 ≤ L) (hG : 0 < G)
    (hD : 0 < D) (hGK : w * D ≤ K * N * G) : V / G ≤ K * (N * L / D) := by
  rw [div_le_iff₀ hG, show K * (N * L / D) * G = K * N * G * L / D by ring, le_div_iff₀ hD]
  nlinarith [mul_le_mul_of_nonneg_right hV hD.le, mul_le_mul_of_nonneg_right hGK hL]

/-- The rate tail: the one term that is not a log at all. -/
private lemma rate_tail_le {c s B L ε : ℝ}
    (hc : 1 ≤ c) (hs : 0 < s) (hs2 : s ≤ 1 / 2) (hB : 0 < B) (hB2 : B ≤ 1 / 2) (hBε : B ≤ ε)
    (hL : 1 ≤ L) : 64 / ε ≤ 4 * (c ^ 2 * L / (s ^ 4 * B ^ 2)) := by
  have hε : 0 < ε := lt_of_lt_of_le hB hBε
  have hs4 : s ^ 4 ≤ 1 / 16 := by
    have h := pow_le_pow_left₀ hs.le hs2 4; norm_num at h; linarith
  have hc2L : (1 : ℝ) ≤ c ^ 2 * L := by nlinarith
  have hD : 0 < s ^ 4 * B ^ 2 := by positivity
  have h1 : s ^ 4 * B ^ 2 ≤ 1 / 16 * B ^ 2 := mul_le_mul_of_nonneg_right hs4 (sq_nonneg B)
  have h2 : 64 / ε ≤ 64 / B := div_le_div_of_nonneg_left (by norm_num) hB hBε
  refine le_trans h2 ?_
  rw [mul_div_assoc', div_le_div_iff₀ hB hD]
  nlinarith [mul_le_mul_of_nonneg_right hc2L hB.le, mul_le_mul_of_nonneg_left h1 (by norm_num : (0:ℝ) ≤ 64)]

omit [Fintype J] in
/-- The pool never holds fewer than 256 suffixes, which is what lets its size stand in for the
constants inside the logs. -/
theorem poolCount_ge (populations : Finset J) (η indecisionLimit εcov δ pAP crossLimit : ℝ)
    (hsig : 0 < sig η) (hη : 0 ≤ η) (hind : 0 < indecisionLimit) (hεcov : 0 < εcov)
    (hεcov1 : εcov ≤ 1) (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) :
    (256 : ℝ) ≤ (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) := by
  set s : ℝ := sig η with hsdef
  have hs2 : s ≤ 1 / 2 := by rw [hsdef, sig]; linarith only [hη]
  have hcut8 : cutBudget η indecisionLimit εcov ≤ 1 / 8 :=
    le_trans (min_le_left _ _) (by linarith only [hεcov1])
  have hcutpos : 0 < cutBudget η indecisionLimit εcov := by
    rw [cutBudget]
    exact lt_min (by linarith only [hεcov]) (lt_min (by rw [← hsdef]; linarith only [hsig])
      (by linarith only [hind]))
  have hl2 := half_le_log_two
  have hlog : 2 ≤ Real.log (2 / cutBudget η indecisionLimit εcov) := by
    have h : Real.log 16 ≤ Real.log (2 / cutBudget η indecisionLimit εcov) :=
      Real.log_le_log (by norm_num) (by rw [le_div_iff₀ hcutpos]; linarith only [hcut8])
    rw [show (16 : ℝ) = 2 ^ 4 by norm_num, Real.log_pow, Nat.cast_ofNat] at h
    linarith only [h, hl2]
  have hv : voteSlack η ^ 2 ≤ 9 / 400 := by
    have h : voteSlack η ≤ 3 / 20 := by simp only [voteSlack, ← hsdef]; linarith only [hs2]
    have h0 : 0 ≤ voteSlack η := by simp only [voteSlack, ← hsdef]; linarith only [hsig]
    nlinarith only [h, h0]
  have hvpos : 0 < voteSlack η ^ 2 := by
    have : 0 < voteSlack η := by simp only [voteSlack, ← hsdef]; linarith only [hsig]
    positivity
  have ha : 800 / 9 ≤ Real.log (2 / cutBudget η indecisionLimit εcov) / voteSlack η ^ 2 := by
    rw [le_div_iff₀ hvpos]
    nlinarith only [hlog, hv, hvpos]
  have hfam : 2 * (800 / 9 + 1) ≤ (famCount η populations indecisionLimit εcov δ crossLimit : ℝ) := by
    have hval : (famCount η populations indecisionLimit εcov δ crossLimit : ℝ)
        = 2 * ((⌈Real.log (2 / cutBudget η indecisionLimit εcov) / voteSlack η ^ 2⌉₊ : ℝ)
          + (⌈Real.log (1 / crossLimit) / voteSlack η ^ 2⌉₊ : ℝ) + (⌈1 / voteSlack η⌉₊ : ℝ) + 1) := by
      rw [famCount]; push_cast; ring
    rw [hval]
    linarith only [Nat.le_ceil (Real.log (2 / cutBudget η indecisionLimit εcov) / voteSlack η ^ 2),
      ha, (Nat.cast_nonneg _ : (0 : ℝ) ≤ (⌈Real.log (1 / crossLimit) / voteSlack η ^ 2⌉₊ : ℝ)),
      (Nat.cast_nonneg _ : (0 : ℝ) ≤ (⌈1 / voteSlack η⌉₊ : ℝ))]
  have hpool : 2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1) / pAP
      ≤ (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) := by
    rw [poolCount]
    push_cast
    linarith only [Nat.le_ceil (2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ)
        + 1) / pAP),
      (Nat.cast_nonneg _ : (0 : ℝ)
        ≤ (⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (pAP / 2) ^ 2)⌉₊ : ℝ))]
  have hdiv : 2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1)
      ≤ 2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1) / pAP := by
    rw [le_div_iff₀ hpAP]
    nlinarith only [hpAP1, hfam]
  linarith only [hpool, hdiv, hfam]

set_option maxHeartbeats 1000000 in
/-- The budget is polynomial in the reciprocals of the rates it resolves.

`sig` enters at the fourth power and `budgetScale` at the third because the screen's tail is
the binding one: its margin is `cutBudget · sig²`, and a deviation bound
squares the margin it is given.  The population count enters squared for the same reason --
`flipBudget` divides by it, and `screenMargin` inherits that.  The scale is cubed rather than
squared because the coverage tails divide by `εcov` once before squaring a margin that already
carries `εcov`.

The pool's size is left inside the log rather than bounded in the rates: the screen's tail
union-bounds over its pairs and the flip tail over its members, and nothing else reads it.

`0 ≤ η` is what bounds `sig η` above; without it the noise rate may be negative, `sig` may be
arbitrarily large, and the `sig ^ 4` in the denominator sends the bound below the count.

The terms are bounded one at a time here, each at its worst with `sig η` and the scale at
`1/2`, and their bounds sum to under 12000. -/
theorem prefCount_le_poly (populations : Finset J)
    (η indecisionLimit εcov δ α pAP crossLimit : ℝ)
    (hsig : 0 < sig η) (hη : 0 ≤ η) (hpop : populations.Nonempty)
    (hind : 0 < indecisionLimit) (hεcov : 0 < εcov) (hεcov1 : εcov ≤ 1)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hα : 0 < α) (hα1 : α < 1)
    (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) :
    (prefCount η populations indecisionLimit εcov δ α pAP crossLimit : ℝ)
      ≤ 12000 * (populations.card : ℝ) ^ 2
        * budgetLog populations η indecisionLimit εcov δ α pAP crossLimit
        / (sig η ^ 4 * budgetScale η indecisionLimit εcov ^ 2) := by
  have hM256 := poolCount_ge populations η indecisionLimit εcov δ pAP crossLimit hsig hη hind
    hεcov hεcov1 hpAP hpAP1
  have hc1 : (1 : ℝ) ≤ (populations.card : ℝ) := Nat.one_le_cast.2 (Finset.card_pos.2 hpop)
  set c : ℝ := (populations.card : ℝ) with hcdef
  clear_value c
  have hcpos : (0 : ℝ) < c := by linarith only [hc1]
  have hc2 : (1 : ℝ) ≤ c ^ 2 := by nlinarith only [hc1]
  set s : ℝ := sig η with hsdef
  clear_value s
  have hsval : s = 1 / 2 - η := by rw [hsdef, sig]
  have hs2 : s ≤ 1 / 2 := by rw [hsval]; linarith only [hη]
  have hηlt : η < 1 / 2 := by rw [hsval] at hsig; linarith only [hsig]
  have hs4 : s ^ 4 ≤ 1 / 16 := by
    have h := pow_le_pow_left₀ hsig.le hs2 4; norm_num at h; linarith only [h]
  have hs2sq : s ^ 2 ≤ 1 / 4 := by
    have h := pow_le_pow_left₀ hsig.le hs2 2; norm_num at h; linarith only [h]
  have hs42 : s ^ 4 ≤ s ^ 2 / 4 := by
    nlinarith only [mul_le_mul_of_nonneg_left hs2sq (sq_nonneg s)]
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
  have hB3 : B ^ 3 ≤ B ^ 2 / 2 := by
    nlinarith only [mul_nonneg (sq_nonneg B) (by linarith only [hB2] : (0 : ℝ) ≤ 1 / 2 - B)]
  -- the cut budget is at least a tenth of the scale
  set cut : ℝ := cutBudget η indecisionLimit εcov with hcutdef
  clear_value cut
  have hcutlb : B ≤ 10 * cut := by
    have h : B / 10 ≤ cut := by
      simp only [hcutdef, cutBudget, ← hsdef]
      exact le_min (by linarith only [hBε, hBpos])
        (le_min (by linarith only [hBs]) (by linarith only [hBind, hBpos]))
    linarith only [h]
  have hcutpos : 0 < cut := by linarith only [hcutlb, hBpos]
  -- what the vote absorbs is at least seven tenths of the signal
  have hf : 7 * s ≤ 10 * flipFrac η := by
    have hne : (1 : ℝ) - η ≠ 0 := by intro h; linarith only [h, hηlt]
    have he : 10 * flipFrac η = 7 * s / (1 - η) := by
      simp only [flipFrac, ← hsdef]; field_simp
    rw [he, le_div_iff₀ (by linarith only [hηlt] : (0 : ℝ) < 1 - η)]
    linarith only [mul_nonneg (by linarith only [hsig] : (0 : ℝ) ≤ 7 * s) hη]
  have hcf : 7 * B * s ≤ 100 * (cut * flipFrac η) := by
    have h : B * (7 * s) ≤ 10 * cut * (10 * flipFrac η) :=
      le_mul2 hBpos.le (by linarith only [hsig]) hcutlb hf
    linarith only [h]
  -- the screen's margin pays `flipFrac`'s `1 − η` back, leaving `cutBudget · sig²`
  have hsm : 147 * B * s ^ 2 ≤ 2560 * (c * screenMargin η populations indecisionLimit εcov δ) := by
    have hne : (1 : ℝ) - η ≠ 0 := by intro h; linarith only [h, hηlt]
    have e : 2560 * (c * screenMargin η populations indecisionLimit εcov δ)
        = 1470 * cut * s ^ 2 := by
      simp only [screenMargin, flipBudget, flipFrac, ← hsdef, ← hcdef, ← hcutdef]
      field_simp
      ring
    rw [e]
    nlinarith only [mul_le_mul_of_nonneg_right hcutlb (sq_nonneg s)]
  have hsmpos : 0 < screenMargin η populations indecisionLimit εcov δ := by
    have h7 : (0 : ℝ) < 147 * B * s ^ 2 := by positivity
    nlinarith only [lt_of_lt_of_le h7 hsm, hcpos]
  -- the pool, which enters only through the log
  set M : ℝ := (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) with hMdef
  clear_value M
  have hM0 : 0 ≤ M := by linarith only [hM256]
  -- the shared log factor
  set d : ℝ := δ * α * pAP * B with hddef
  clear_value d
  have hdpos : 0 < d := by
    rw [hddef]; exact mul_pos (mul_pos (mul_pos hδ hα) hpAP) hBpos
  have hapB : α * pAP * B ≤ 1 / 2 := by
    have hαp : α * pAP ≤ 1 := by linarith only [le_mul2 hα.le hpAP.le hα1.le hpAP1]
    linarith only [le_mul2 (mul_pos hα hpAP).le hBpos.le hαp hB2]
  have hdpB : δ * pAP * B ≤ 1 / 2 := by
    have hδp : δ * pAP ≤ 1 := by linarith only [le_mul2 hδ.le hpAP.le hδ1 hpAP1]
    linarith only [le_mul2 (mul_pos hδ hpAP).le hBpos.le hδp hB2]
  set T : ℝ := (c + 2) * M / d with hTdef
  clear_value T
  have hcM0 : (0 : ℝ) ≤ (c + 2) * M := by positivity
  have hTδ : 2 * ((c + 2) * M) ≤ T * δ := by
    rw [hTdef, div_mul_eq_mul_div, le_div_iff₀ hdpos, hddef]
    nlinarith only [mul_le_mul_of_nonneg_left hapB (mul_nonneg hcM0 hδ.le)]
  have hTpos : 0 < T := by rw [hTdef]; positivity
  have hTge : T * δ ≤ T := by nlinarith only [hTpos, hδ1]
  have hT6 : 6 ≤ T := by nlinarith only [hTδ, hTge, hc1, hM256]
  have hTα : 1 ≤ T * α := by
    rw [hTdef, div_mul_eq_mul_div, le_div_iff₀ hdpos, hddef]
    have h1 : 1 ≤ (c + 2) * M := by nlinarith only [hc1, hM256]
    nlinarith only [mul_le_mul_of_nonneg_left hdpB hα.le, mul_le_mul_of_nonneg_right h1 hα.le, hα]
  set L : ℝ := Real.log T with hLdef
  clear_value L
  have hL1 : 1 ≤ L := by rw [hLdef]; exact one_le_log_of_six_le hT6
  have hL0 : 0 ≤ L := by linarith only [hL1]
  have hLeq : budgetLog populations η indecisionLimit εcov δ α pAP crossLimit = L := by
    rw [hLdef, hTdef, hddef, hBdef, hcdef, hMdef, budgetLog]
  -- each log in the tails, against `L`
  have hlog1 : ∀ Z : ℝ, 0 < Z → Z * δ ≤ T * δ → Real.log Z ≤ 1 * L := by
    intro Z hZ h
    have h' : Z ≤ T := le_of_mul_le_mul_right h hδ
    have := log_le_of_le_pow hZ 1 (by simpa using h')
    rw [hLdef]; push_cast at this; linarith only [this]
  have hX : Real.log (128 * c * (M + 2) ^ 2 / δ) ≤ 3 * L := by
    have h27 : 27 * c ≤ (c + 2) ^ 3 := by
      nlinarith only [mul_nonneg (sq_nonneg (c - 1)) (by linarith only [hc1] : (0 : ℝ) ≤ c + 8)]
    have hM3 : (M + 2) ^ 2 ≤ M ^ 3 := by nlinarith only [hM256]
    have hcube : (2 * ((c + 2) * M)) ^ 3 ≤ (T * δ) ^ 3 := pow_le_pow_left₀ (by positivity) hTδ 3
    have hδ3 : (T * δ) ^ 3 ≤ T ^ 3 * δ := by
      have : δ ^ 3 ≤ δ := by nlinarith only [hδ, hδ1, mul_pos hδ hδ]
      calc (T * δ) ^ 3 = T ^ 3 * δ ^ 3 := by ring
        _ ≤ T ^ 3 * δ := mul_le_mul_of_nonneg_left this (by positivity)
    have harg : 128 * c * (M + 2) ^ 2 / δ ≤ T ^ 3 := by
      rw [div_le_iff₀ hδ]
      have e : (2 * ((c + 2) * M)) ^ 3 = 8 * (c + 2) ^ 3 * M ^ 3 := by ring
      nlinarith only [hcube, hδ3, e, mul_le_mul_of_nonneg_right h27 (pow_nonneg hM0 3),
        mul_le_mul_of_nonneg_left hM3 (by linarith only [hcpos] : (0 : ℝ) ≤ 128 * c),
        mul_nonneg hcpos.le (pow_nonneg hM0 3)]
    have h := log_le_of_le_pow (by positivity) 3 harg
    rw [hLdef]; push_cast at h; linarith only [h]
  have hY : Real.log (128 * c * (M + 1) / δ) ≤ 2 * L := by
    have h8 : 8 * c ≤ (c + 2) ^ 2 := by nlinarith only [sq_nonneg (c - 2)]
    have hsq : (2 * ((c + 2) * M)) ^ 2 ≤ (T * δ) ^ 2 := pow_le_pow_left₀ (by positivity) hTδ 2
    have hδ2 : (T * δ) ^ 2 ≤ T ^ 2 * δ := by
      have : δ ^ 2 ≤ δ := by nlinarith only [hδ, hδ1]
      calc (T * δ) ^ 2 = T ^ 2 * δ ^ 2 := by ring
        _ ≤ T ^ 2 * δ := mul_le_mul_of_nonneg_left this (by positivity)
    have harg : 128 * c * (M + 1) / δ ≤ T ^ 2 := by
      rw [div_le_iff₀ hδ]
      have e : (2 * ((c + 2) * M)) ^ 2 = 4 * (c + 2) ^ 2 * M ^ 2 := by ring
      have hM2 : 4 * (M + 1) ≤ M ^ 2 := by nlinarith only [hM256]
      nlinarith only [hsq, hδ2, e, mul_le_mul_of_nonneg_right h8 (sq_nonneg M),
        mul_le_mul_of_nonneg_left hM2 (by linarith only [hcpos] : (0 : ℝ) ≤ 32 * c)]
    have h := log_le_of_le_pow (by positivity) 2 harg
    rw [hLdef]; push_cast at h; linarith only [h]
  have hR : Real.log (128 * c / δ) ≤ 1 * L :=
    hlog1 _ (by positivity) (by rw [div_mul_cancel₀ _ hδ.ne']; nlinarith only [hTδ, hc1, hM256])
  have hR256 : Real.log (256 * c / δ) ≤ 1 * L :=
    hlog1 _ (by positivity) (by rw [div_mul_cancel₀ _ hδ.ne']; nlinarith only [hTδ, hc1, hM256])
  have hAα : Real.log (1 / α) ≤ 1 * L := by
    have h : 1 / α ≤ T := by rw [div_le_iff₀ hα]; linarith only [hTα]
    have h' := log_le_of_le_pow (div_pos one_pos hα) 1 (by simpa using h)
    rw [hLdef]; push_cast at h'; linarith only [h']
  -- the six tails, each against the common quotient
  have hD : (0 : ℝ) < s ^ 4 * B ^ 2 := by positivity
  set Q : ℝ := c ^ 2 * L / (s ^ 4 * B ^ 2) with hQdef
  have hQnn : 0 ≤ Q := by positivity
  have hBsq : B ^ 2 ≤ 1 / 4 := by
    have h := pow_le_pow_left₀ hBpos.le hB2 2; norm_num at h; linarith only [h]
  have hs4B2 : s ^ 4 * B ^ 2 ≤ s ^ 2 / 4 * B ^ 2 :=
    mul_le_mul_of_nonneg_right hs42 (sq_nonneg B)
  have t1 : Real.log (128 * c * (M + 2) ^ 2 / δ)
      / (2 * (screenMargin η populations indecisionLimit εcov δ / 2) ^ 2) ≤ 2000 * Q := by
    refine tail_le hX hL0 (by positivity) hD ?_
    have h := mul_self_le_mul_self (by positivity) hsm
    nlinarith only [h]
  have t2 : Real.log (128 * c / δ) / (2 * (cut / 32) ^ 2) ≤ 3200 * Q := by
    refine tail_le hR hL0 (by positivity) hD ?_
    have h := mul_self_le_mul_self hBpos.le hcutlb
    have hBB : B ^ 2 ≤ 100 * cut ^ 2 := by nlinarith only [h]
    have h1 : s ^ 4 * B ^ 2 ≤ 1 / 16 * (100 * cut ^ 2) :=
      le_mul2 (pow_nonneg hsig.le 4) (sq_nonneg B) hs4 hBB
    have h2 : cut ^ 2 ≤ c ^ 2 * cut ^ 2 := le_mul_of_one_le_left (sq_nonneg cut) hc2
    nlinarith only [h1, h2]
  have hcov : 32 * (s ^ 4 * B ^ 2) ≤ 256 * c ^ 2 * s ^ 2 := by
    have h1 : s ^ 4 * B ^ 2 ≤ s ^ 2 / 4 * (1 / 4) :=
      le_trans hs4B2 (mul_le_mul_of_nonneg_left hBsq (by positivity))
    nlinarith only [h1, mul_le_mul_of_nonneg_right hc2 (sq_nonneg s), sq_nonneg s]
  have t3 : 32 * Real.log (1 / α) / s ^ 2 ≤ 256 * Q :=
    tail_le (w := 32) (by linarith only [hAα]) hL0 (by positivity) hD hcov
  have t4 : 32 * Real.log (256 * c / δ) / s ^ 2 ≤ 256 * Q :=
    tail_le (w := 32) (by linarith only [hR256]) hL0 (by positivity) hD hcov
  have t5 : Real.log (128 * c * (M + 1) / δ) / (2 * (3 * cut * flipFrac η / 32) ^ 2)
      ≤ 6000 * Q := by
    have hfpos : 0 < flipFrac η := by linarith only [hf, hsig]
    refine tail_le hY hL0
      (mul_pos two_pos (pow_pos (div_pos (mul_pos (mul_pos three_pos hcutpos) hfpos)
        (by norm_num)) 2)) hD ?_
    have h := mul_self_le_mul_self (by positivity) hcf
    have hcf2 : 0 ≤ (cut * flipFrac η) ^ 2 := sq_nonneg _
    have hs2B2 : 0 ≤ s ^ 2 * B ^ 2 := by positivity
    have hk : (cut * flipFrac η) ^ 2 ≤ c ^ 2 * (cut * flipFrac η) ^ 2 :=
      le_mul_of_one_le_left hcf2 hc2
    nlinarith only [h, hs4B2, hk, hcf2, hs2B2]
  have t6 : 64 / εcov ≤ 4 * Q := by rw [hQdef]; exact rate_tail_le hc1 hsig hs2 hBpos hB2 hBε hL1
  have t7 : (7 : ℝ) ≤ Q := by
    rw [hQdef, le_div_iff₀ hD]
    have hc2L : (1 : ℝ) ≤ c ^ 2 * L := by linarith only [le_mul2 zero_le_one zero_le_one hc2 hL1]
    linarith only [le_mul2 (pow_nonneg hsig.le 4) (sq_nonneg B) hs4 hBsq, hc2L]
  -- and the sum
  have hbound := prefCount_le_terms populations η indecisionLimit εcov δ α pAP
    crossLimit
  simp only [prefTerms, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    ← hcdef, ← hsdef, ← hcutdef, ← hMdef] at hbound
  have m1 := max_le t1 (by positivity)
  have m2 := max_le t2 (by positivity)
  have m3 := max_le t3 (by positivity)
  have m4 := max_le t4 (by positivity)
  have m5 := max_le t5 (by positivity)
  have m6 := max_le t6 (by positivity)
  rw [hLeq, show 12000 * c ^ 2 * L / (s ^ 4 * B ^ 2) = 12000 * Q by rw [hQdef]; ring]
  linarith only [hbound, m1, m2, m3, m4, m5, m6, t7, hQnn]

set_option maxHeartbeats 1000000 in
/-- `validCount` is polynomial in the rates validity resolves.  Next to `prefCount_le_poly`,
the scale is `min εcov indecisionLimit` rather than a minimum that also takes the signal, it
enters squared rather than cubed, and neither `α` nor `pAP` is in the log. -/
theorem validCount_le_poly (populations : Finset J)
    (η indecisionLimit εcov εval δ pAP crossLimit : ℝ)
    (hsig : 0 < sig η) (hη : 0 ≤ η) (hpop : populations.Nonempty)
    (hind : 0 < indecisionLimit) (hind1 : indecisionLimit ≤ 1 / 2) (hεcov : 0 < εcov)
    (hεcov1 : εcov ≤ 1) (hεval : 0 < εval) (hεval1 : εval ≤ 1) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) :
    (validCount η populations indecisionLimit εcov εval δ pAP crossLimit : ℝ)
      ≤ 52 * (populations.card : ℝ) ^ 2
        * Real.log (((populations.card : ℝ) + 2)
          * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 2) / δ)
        / (sig η ^ 4 * min εval indecisionLimit ^ 2) := by
  have hM256 := poolCount_ge populations η indecisionLimit εcov δ pAP crossLimit hsig hη hind
    hεcov hεcov1 hpAP hpAP1
  have hc1 : (1 : ℝ) ≤ (populations.card : ℝ) := Nat.one_le_cast.2 (Finset.card_pos.2 hpop)
  have hvm : validMargin η populations εval
      = 77 * εval * sig η ^ 2 / (128 * (populations.card : ℝ)) := by
    have hne : (1 : ℝ) - η ≠ 0 := by rw [sig] at hsig; intro h; linarith
    rw [validMargin, validFlip, validFrac]
    field_simp
    ring
  rw [validCount]
  push_cast
  rw [hvm]
  set c : ℝ := (populations.card : ℝ) with hcdef
  set M : ℝ := (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) with hMdef
  set s : ℝ := sig η with hsdef
  set μ : ℝ := min εval indecisionLimit with hμdef
  set X : ℝ := (c + 2) * (M + 2) / δ with hXdef
  have hM0 : 0 ≤ M := by linarith
  have hcpos : (0 : ℝ) < c := by linarith
  have hc2 : 1 ≤ c ^ 2 := by nlinarith
  have hsval : s = 1 / 2 - η := by rw [hsdef, sig]
  have hs2 : s ≤ 1 / 2 := by rw [hsval]; linarith
  have hηlt : η < 1 / 2 := by rw [hsval] at hsig; linarith
  have hμε : μ ≤ εval := min_le_left _ _
  have hμi : μ ≤ indecisionLimit := min_le_right _ _
  have hμ : 0 < μ := lt_min hεval hind
  have hf : s ≤ validFrac η := by
    rw [validFrac, ← hsdef, le_div_iff₀ (by linarith : (0 : ℝ) < 1 - η)]
    nlinarith [mul_nonneg hsig.le hη]
  have hfpos : 0 < validFrac η := by linarith
  have hs4 : s ^ 4 ≤ 1 / 16 := by
    have := pow_le_pow_left₀ hsig.le hs2 4; norm_num at this; linarith
  have hs2sq : s ^ 2 ≤ 1 / 4 := by
    have := pow_le_pow_left₀ hsig.le hs2 2; norm_num at this; linarith
  have hs42 : s ^ 4 ≤ s ^ 2 / 4 := by nlinarith [mul_le_mul_of_nonneg_left hs2sq (sq_nonneg s)]
  have hXpos : 0 < X := by positivity
  have hX : 6 ≤ X := by
    rw [hXdef, le_div_iff₀ hδ]
    nlinarith
  set L : ℝ := Real.log X with hLdef
  have hL : 1 ≤ L := one_le_log_of_six_le hX
  have hL0 : 0 ≤ L := by linarith
  have hD : (0 : ℝ) < s ^ 4 * μ ^ 2 := by positivity
  set Q : ℝ := c ^ 2 * L / (s ^ 4 * μ ^ 2) with hQdef
  have hQ1 : 64 ≤ Q := by
    rw [hQdef, le_div_iff₀ hD]
    have hμ2 : μ ^ 2 ≤ 1 / 4 := by
      have := pow_le_pow_left₀ hμ.le (le_trans hμi hind1) 2; norm_num at this; linarith
    nlinarith [mul_le_mul hs4 hμ2 (by positivity) (by norm_num), mul_le_mul hc2 hL (by norm_num)
      (by positivity)]
  -- the logs, against `L`
  have hZ1 : Real.log (128 * c * (M + 2) ^ 2 / δ) ≤ 5 / 2 * L := by
    have h8 : 8 * c ≤ (c + 2) ^ 2 := by nlinarith [sq_nonneg (c - 2)]
    have h192 : 192 * c ^ 2 ≤ (c + 2) ^ 5 := by
      have h4 : 64 * c ^ 2 ≤ (c + 2) ^ 4 := by
        nlinarith [mul_self_le_mul_self (by positivity) h8]
      nlinarith [mul_le_mul_of_nonneg_left (by linarith : (3 : ℝ) ≤ c + 2)
        (by positivity : (0 : ℝ) ≤ (c + 2) ^ 4)]
    have hδ3 : δ ^ 3 ≤ 1 := pow_le_one₀ hδ.le hδ1
    have harg : (128 * c * (M + 2) ^ 2 / δ) ^ 2 ≤ X ^ 5 := by
      rw [hXdef, div_pow, div_pow, mul_pow, div_le_div_iff₀ (by positivity) (by positivity)]
      have hδ5 : δ ^ 5 ≤ δ ^ 2 := by
        calc δ ^ 5 = δ ^ 2 * δ ^ 3 := by ring
          _ ≤ δ ^ 2 * 1 := mul_le_mul_of_nonneg_left hδ3 (by positivity)
          _ = δ ^ 2 := mul_one _
      have hbig : 16384 * c ^ 2 ≤ (c + 2) ^ 5 * (M + 2) := by
        nlinarith [mul_le_mul h192 (by linarith : (258 : ℝ) ≤ M + 2) (by norm_num)
          (by positivity), sq_nonneg c]
      calc (128 * c) ^ 2 * ((M + 2) ^ 2) ^ 2 * δ ^ 5
            = 16384 * c ^ 2 * (M + 2) ^ 4 * δ ^ 5 := by ring
        _ ≤ 16384 * c ^ 2 * (M + 2) ^ 4 * δ ^ 2 :=
            mul_le_mul_of_nonneg_left hδ5 (by positivity)
        _ = 16384 * c ^ 2 * ((M + 2) ^ 4 * δ ^ 2) := by ring
        _ ≤ (c + 2) ^ 5 * (M + 2) * ((M + 2) ^ 4 * δ ^ 2) :=
            mul_le_mul_of_nonneg_right hbig (by positivity)
        _ = ((c + 2) * (M + 2)) ^ 5 * δ ^ 2 := by ring
    have h := log_le_of_le_pow (by positivity) 5 harg
    rw [Real.log_pow] at h
    push_cast at h
    rw [hLdef]; linarith
  have hZ2 : Real.log (128 * c * (M + 1) / δ) ≤ 2 * L := by
    have h8 : 8 * c ≤ (c + 2) ^ 2 := by nlinarith [sq_nonneg (c - 2)]
    have harg : 128 * c * (M + 1) / δ ≤ X ^ 2 := by
      rw [hXdef, div_pow, mul_pow, div_le_div_iff₀ hδ (by positivity)]
      have hδ2 : δ ^ 2 ≤ δ := by nlinarith
      have hM : 16 * (M + 1) ≤ (M + 2) ^ 2 := by nlinarith
      calc 128 * c * (M + 1) * δ ^ 2 ≤ 128 * c * (M + 1) * δ :=
            mul_le_mul_of_nonneg_left hδ2 (by positivity)
        _ = 8 * c * (16 * (M + 1)) * δ := by ring
        _ ≤ (c + 2) ^ 2 * (M + 2) ^ 2 * δ := by gcongr
    have h := log_le_of_le_pow (by positivity) 2 harg
    push_cast at h
    rw [hLdef]; linarith
  have hZ3 : Real.log (128 * c / δ) ≤ 1 * L := by
    have harg : 128 * c / δ ≤ X := by
      rw [hXdef]
      exact div_le_div_of_nonneg_right (by nlinarith) hδ.le
    have := Real.log_le_log (by positivity) harg
    rw [hLdef]; linarith
  -- the four tails, each against the common quotient
  have t1 : Real.log (128 * c * (M + 2) ^ 2 / δ)
      / (2 * (77 * εval * s ^ 2 / (128 * c)) ^ 2) ≤ 7 / 2 * Q := by
    refine tail_le hZ1 hL0 (by positivity) hD ?_
    have h := mul_self_le_mul_self (by positivity) (mul_le_mul_of_nonneg_right hμε (sq_nonneg s))
    have e : 7 / 2 * c ^ 2 * (2 * (77 * εval * s ^ 2 / (128 * c)) ^ 2)
        = 41503 / 16384 * (εval * s ^ 2) ^ 2 := by
      field_simp
      ring
    rw [e]
    nlinarith [h]
  have t2 : Real.log (128 * c * (M + 1) / δ) / (2 * (εval * validFrac η / 8) ^ 2) ≤ 16 * Q := by
    refine tail_le hZ2 hL0 (by positivity) hD ?_
    have hg : μ * s ≤ εval * validFrac η := by
      nlinarith [mul_le_mul_of_nonneg_left hf hεval.le, mul_le_mul_of_nonneg_right hμε hsig.le]
    have h := mul_self_le_mul_self (by positivity) hg
    have e1 := mul_le_mul_of_nonneg_right hs42 (sq_nonneg μ)
    have e2 := mul_le_mul_of_nonneg_right hc2 (sq_nonneg (εval * validFrac η))
    nlinarith [h, e1, e2, sq_nonneg (εval * validFrac η)]
  have t3 : Real.log (128 * c / δ) / (2 * (εval / 32) ^ 2) ≤ 32 * Q := by
    refine tail_le hZ3 hL0 (by positivity) hD ?_
    have h := mul_self_le_mul_self hμ.le hμε
    nlinarith [mul_le_mul_of_nonneg_right hs4 (sq_nonneg μ),
      mul_le_mul_of_nonneg_right hc2 (sq_nonneg εval), sq_nonneg εval]
  have t4 : Real.log (128 * c / δ) / (2 * indecisionLimit ^ 2) ≤ 1 / 16 * Q := by
    refine tail_le hZ3 hL0 (by positivity) hD ?_
    have h := mul_self_le_mul_self hμ.le hμi
    nlinarith [mul_le_mul_of_nonneg_right hs4 (sq_nonneg μ),
      mul_le_mul_of_nonneg_right hc2 (sq_nonneg indecisionLimit), sq_nonneg indecisionLimit]
  have hQ0 : 0 ≤ Q := by linarith
  have c1 := ceil_le_of_le t1 (by positivity)
  have c2 := ceil_le_of_le t2 (by positivity)
  have c3 := ceil_le_of_le t3 (by positivity)
  have c4 := ceil_le_of_le t4 (by positivity)
  rw [show 52 * c ^ 2 * Real.log X / (s ^ 4 * μ ^ 2) = 52 * Q by rw [hQdef, hLdef]; ring]
  linarith

omit [Fintype J] in
/-- The family, seed included.  The cut budget is only `≥ B/10`, so its log carries a `log 10`,
which `log (2/B) ≥ log 4` pays for; the band's term is the rest of `log (2/(B·crossLimit))`. -/
theorem famCount_succ_le (populations : Finset J) (η indecisionLimit εcov δ crossLimit : ℝ)
    (hsig : 0 < sig η) (hη : 0 ≤ η) (hind : 0 < indecisionLimit) (hεcov : 0 < εcov)
    (hζ : 0 < crossLimit) (hζ1 : crossLimit ≤ 1) :
    (famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1
      ≤ 64 * Real.log (2 / (budgetScale η indecisionLimit εcov * crossLimit)) / sig η ^ 2 := by
  have hfcval : (famCount η populations indecisionLimit εcov δ crossLimit : ℝ)
      = 2 * ((⌈Real.log (2 / cutBudget η indecisionLimit εcov) / voteSlack η ^ 2⌉₊ : ℝ)
        + (⌈Real.log (1 / crossLimit) / voteSlack η ^ 2⌉₊ : ℝ) + (⌈1 / voteSlack η⌉₊ : ℝ) + 1) := by
    rw [famCount]
    push_cast
    ring
  set s : ℝ := sig η with hsdef
  have hs2 : s ≤ 1 / 2 := by rw [hsdef, sig]; linarith only [hη]
  set B : ℝ := budgetScale η indecisionLimit εcov with hBdef
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
  set cut : ℝ := cutBudget η indecisionLimit εcov with hcutdef
  have hcutlb : B ≤ 10 * cut := by
    have h : B / 10 ≤ cut := by
      simp only [hcutdef, cutBudget, ← hsdef]
      exact le_min (by linarith only [hBε, hBpos])
        (le_min (by linarith only [hBs]) (by linarith only [hBind, hBpos]))
    linarith only [h]
  have hcutpos : 0 < cut := by linarith only [hcutlb, hBpos]
  -- `G = log (2/B)` is at least `log 4`, and `1000 ≤ 4⁵` makes `log 10 ≤ (5/3)·log 4`
  set G : ℝ := Real.log (2 / B) with hGdef
  have hl2 := half_le_log_two
  have hG4 : 2 * Real.log 2 ≤ G := by
    have h : Real.log 4 ≤ G :=
      Real.log_le_log (by norm_num) (by rw [le_div_iff₀ hBpos]; linarith only [hB2])
    rwa [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.log_pow, Nat.cast_ofNat] at h
  have hcutG : Real.log (2 / cut) ≤ 8 / 3 * G := by
    have h1 : Real.log (2 / cut) ≤ Real.log (10 * (2 / B)) :=
      Real.log_le_log (div_pos (by norm_num) hcutpos)
        (by rw [← mul_div_assoc, div_le_div_iff₀ hcutpos hBpos]; linarith only [hcutlb])
    rw [Real.log_mul (by norm_num) (div_pos (by norm_num) hBpos).ne'] at h1
    have h10 : 3 * Real.log 10 ≤ 5 * (2 * Real.log 2) := by
      have h := Real.log_le_log (by norm_num : (0 : ℝ) < 10 ^ 3) (by norm_num : (10 : ℝ) ^ 3 ≤ 2 ^ 10)
      rw [Real.log_pow, Real.log_pow] at h
      push_cast at h
      linarith only [h]
    linarith only [h1, hG4, h10]
  set Z : ℝ := Real.log (1 / crossLimit) with hZdef
  have hZ0 : 0 ≤ Z := Real.log_nonneg (by rw [le_div_iff₀ hζ]; linarith only [hζ1])
  have hGZ : Real.log (2 / (B * crossLimit)) = G + Z := by
    rw [hGdef, hZdef, ← Real.log_mul (div_pos (by norm_num) hBpos).ne' (one_div_pos.2 hζ).ne']
    congr 1
    field_simp
  have hs2pos : (0 : ℝ) < s ^ 2 := pow_pos hsig 2
  have hGs : 0 ≤ G / s ^ 2 := div_nonneg (by linarith only [hG4, hl2]) hs2pos.le
  have hZs : 0 ≤ Z / s ^ 2 := div_nonneg hZ0 hs2pos.le
  set Q : ℝ := (G + Z) / s ^ 2 with hQdef
  have hsplit : G / s ^ 2 + Z / s ^ 2 = Q := by rw [hQdef, add_div]
  have hQ4 : 4 ≤ Q := by
    rw [hQdef, le_div_iff₀ hs2pos]
    have h := pow_le_pow_left₀ hsig.le hs2 2
    norm_num at h
    linarith only [h, hG4, hl2, hZ0]
  have hvote : voteSlack η ^ 2 = 9 * s ^ 2 / 100 := by
    simp only [voteSlack, ← hsdef]; ring
  have hzA : Real.log (2 / cut) / voteSlack η ^ 2 ≤ 800 / 27 * (G / s ^ 2) := by
    have e : 800 / 27 * (G / s ^ 2) * (9 * s ^ 2 / 100) = 8 / 3 * G := by
      field_simp; ring
    rw [hvote, div_le_iff₀ (by positivity), e]
    exact hcutG
  have hzC : Z / voteSlack η ^ 2 ≤ 100 / 9 * (Z / s ^ 2) := by
    have e : 100 / 9 * (Z / s ^ 2) * (9 * s ^ 2 / 100) = Z := by
      field_simp
    rw [hvote, div_le_iff₀ (by positivity), e]
  -- the seed's term: `1/voteSlack = 10/(3s) ≤ (5/3)/s²`, and `G ≥ 2·log 2 ≥ 4/3`
  have hzS : 1 / voteSlack η ≤ 5 / 4 * (G / s ^ 2) := by
    have e : 1 / voteSlack η = 10 / (3 * s) := by
      simp only [voteSlack, ← hsdef]; field_simp
    have hG43 : 4 / 3 ≤ G := by linarith [Real.log_two_gt_d9, hG4]
    rw [e, div_le_iff₀ (by positivity : (0 : ℝ) < 3 * s)]
    have : 5 / 4 * (G / s ^ 2) * (3 * s) = 15 / 4 * G / s := by field_simp; ring
    rw [this, le_div_iff₀ hsig]
    nlinarith [hG43, hs2, hsig]
  have hcA := ceil_le_of_le hzA (by positivity)
  have hcC := ceil_le_of_le hzC (by positivity)
  have hcS := ceil_le_of_le hzS (by positivity)
  have hGbig : 16 / 3 ≤ G / s ^ 2 := by
    have hG43 : 4 / 3 ≤ G := by linarith [Real.log_two_gt_d9, hG4]
    rw [le_div_iff₀ hs2pos]
    nlinarith [hs2, hsig]
  rw [hfcval, hGZ, mul_div_assoc, ← hQdef]
  linarith only [hcA, hcC, hcS, hsplit, hGs, hZs, hQ4, hGbig]

omit [Fintype J] in
/-- The suffix pool: the family over `pAP`, and the findability tail over `pAP²`. -/
theorem poolCount_le (populations : Finset J) (η indecisionLimit εcov δ pAP crossLimit : ℝ)
    (hsig : 0 < sig η) (hη : 0 ≤ η) (hind : 0 < indecisionLimit) (hεcov : 0 < εcov)
    (hpop : populations.Nonempty) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) (hζ : 0 < crossLimit) (hζ1 : crossLimit ≤ 1) :
    (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ)
      ≤ 128 * Real.log (2 / (budgetScale η indecisionLimit εcov * crossLimit))
          / (sig η ^ 2 * pAP)
        + 16 * Real.log (((populations.card : ℝ) + 2) / δ) / pAP ^ 2 := by
  have hc1 : (1 : ℝ) ≤ (populations.card : ℝ) := Nat.one_le_cast.2 (Finset.card_pos.2 hpop)
  set c : ℝ := (populations.card : ℝ) with hcdef
  have hcpos : (0 : ℝ) < c := by linarith only [hc1]
  have hf := famCount_succ_le populations η indecisionLimit εcov δ crossLimit hsig hη hind hεcov
    hζ hζ1
  set P : ℝ := 64 * Real.log (2 / (budgetScale η indecisionLimit εcov * crossLimit))
    / sig η ^ 2 with hPdef
  have hPnn : 0 ≤ P := le_trans (by positivity) hf
  have hA : 2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1) / pAP
      ≤ 2 * P / pAP := div_le_div_of_nonneg_right (by linarith only [hf]) hpAP.le
  -- `H = log ((c+2)/δ)` is at least `log 3`, so it covers `log 128 ≤ log 3⁵` five times over
  set H : ℝ := Real.log ((c + 2) / δ) with hHdef
  have hl2 := half_le_log_two
  have hH3 : Real.log 3 ≤ H :=
    Real.log_le_log (by norm_num) (by rw [le_div_iff₀ hδ]; linarith only [hc1, hδ1])
  have hH2 : Real.log 2 ≤ H := le_trans (Real.log_le_log (by norm_num) (by norm_num)) hH3
  have hlog128 : Real.log (128 * c / δ) ≤ 6 * H := by
    have h1 : Real.log (128 * c / δ) ≤ Real.log (3 ^ 5 * ((c + 2) / δ)) :=
      Real.log_le_log (div_pos (by linarith only [hcpos]) hδ) (by
        rw [mul_div_assoc]
        exact mul_le_mul (by norm_num)
          (div_le_div_of_nonneg_right (by linarith only : c ≤ c + 2) hδ.le)
          (div_nonneg hcpos.le hδ.le) (by norm_num))
    rw [Real.log_mul (by norm_num) (by positivity), Real.log_pow, Nat.cast_ofNat] at h1
    linarith only [h1, hH3]
  have hp2pos : (0 : ℝ) < pAP ^ 2 := pow_pos hpAP 2
  set R : ℝ := H / pAP ^ 2 with hRdef
  have hRhalf : 1 / 2 ≤ R := by
    rw [hRdef, le_div_iff₀ hp2pos]
    linarith only [pow_le_one₀ hpAP.le hpAP1 (n := 2), hH2, hl2]
  have hC : Real.log (128 * c / δ) / (2 * (pAP / 2) ^ 2) ≤ 12 * R := by
    rw [div_le_iff₀ (by positivity), hRdef]
    field_simp
    linarith only [hlog128]
  have h1 := ceil_le_of_le hA (div_nonneg (by linarith only [hPnn]) hpAP.le)
  have h2 := ceil_le_of_le hC (by linarith only [hRhalf])
  have hpoolval : (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ)
      = (⌈2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1) / pAP⌉₊ : ℝ)
        + (⌈Real.log (128 * c / δ) / (2 * (pAP / 2) ^ 2)⌉₊ : ℝ) := by
    rw [hcdef, poolCount]
    push_cast
    ring
  have hPe : 2 * P / pAP
      = 128 * Real.log (2 / (budgetScale η indecisionLimit εcov * crossLimit))
        / (sig η ^ 2 * pAP) := by
    rw [hPdef]; ring
  rw [hpoolval, ← hPe, mul_div_assoc 16]
  linarith only [h1, h2, hRhalf]

/-- The cut budget the proof runs at, `εcov` held under the signal and the veto's share of `δ`,
resolves to the statement's `cutScale`. -/
lemma cutScale_eq (populations : Finset J) (η₀ indecisionLimit εcov₀ : ℝ) (v : ℕ) :
    min (min εcov₀ (min (1 / 2 - η₀) (1 / ((populations.card : ℝ) * v))))
        (min (1 / 2 - η₀) indecisionLimit)
      = cutScale populations η₀ indecisionLimit εcov₀ v := by
  unfold cutScale
  apply le_antisymm <;> simp only [le_min_iff, min_le_iff, le_refl, true_or, or_true, and_self]

lemma cutScale_eq' (populations : Finset J) (η₀ indecisionLimit εcov₀ : ℝ) (v : ℕ) :
    min (min εcov₀ (min (1 / 2 - η₀) (1 / ((populations.card : ℝ) * v)))) indecisionLimit
      = cutScale populations η₀ indecisionLimit εcov₀ v := by
  unfold cutScale
  apply le_antisymm <;> simp only [le_min_iff, min_le_iff, le_refl, true_or, or_true, and_self]

open MeasureTheory ProbabilityTheory in
/-- `ClusteringGuarantee` with every population covered to `1 − εcov`, which is what the proof
shows. -/
def ClusteringGuaranteeAll : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {J : Type*} [Fintype J]
    (O : Oracle μ S) (populations : Finset J) (uni : J) (Pre Suf : Set S)
    (η₀ indecisionLimit εcov α δ pAP crossLimit : ℝ) (v : ℕ),
  O.η ≤ η₀ →
  η₀ < 1 / 2 →
  uni ∈ populations →
  Flat Pre Suf →
  0 < pAP →
  0 < indecisionLimit →
  indecisionLimit ≤ 1 / 2 →
  0 < α →
  α < 1 / 2 →
  0 < εcov →
  εcov ≤ 1 →
  0 < δ →
  δ ≤ 1 →
  0 < crossLimit →
  crossLimit ≤ 1 →
  0 < v →
  ∃ cap : ℝ,
    0 < cap ∧
    ∀ (D : J → Measure S) (Dsf : Measure S),
      (∀ j, IsProbabilityMeasure (D j)) → IsProbabilityMeasure Dsf →
      (∀ j ∈ populations, D j Preᶜ = 0) →
      Dsf Sufᶜ = 0 →
      pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
      ∀ ρ : ℝ,
      (∀ j ∈ populations, collisionMass (D j) ≤ ρ) →
      ρ ≤ cap →
      collisionMass Dsf ≤ cap →
      ∃ states : Finset State,
        (∀ B ∈ states, (B.npref : ℝ) ≤
          12000
          * (populations.card : ℝ) ^ 2
          * Real.log (
            ((populations.card : ℝ) + 2) * B.nsuff
            / (δ * α * pAP * cutScale populations η₀ indecisionLimit εcov v)
          )
          / ((1 / 2 - η₀) ^ 4 * cutScale populations η₀ indecisionLimit εcov v ^ 2)
        ) ∧
        (∃ B ∈ states, (B.npref : ℝ) ≤
          104
          * (populations.card : ℝ) ^ 2
          * Real.log (((populations.card : ℝ) + 2) * ((B.nsuff : ℝ) + 2) / δ)
          / ((1 / 2 - η₀) ^ 4 * min εcov indecisionLimit ^ 2)
        ) ∧
        (∀ B ∈ states,
          (B.k : ℝ) ≤ 64 * Real.log (2 / (cutScale populations η₀ indecisionLimit εcov v * crossLimit))
            / (1 / 2 - η₀) ^ 2
          ∧ (B.nsuff : ℝ) ≤ 128 * Real.log
                (2 / (cutScale populations η₀ indecisionLimit εcov v * crossLimit))
              / ((1 / 2 - η₀) ^ 2 * pAP)
            + 16 * Real.log (((populations.card : ℝ) + 2) / δ) / pAP ^ 2) ∧
        (∀ B ∈ states, ∀ F : Finset S, F.card + 1 ≤ B.k → ∀ p,
          (B.hi < meanVote O F p → μ.real {ω | voteCount O.mq F p ω ≤ B.lo} ≤ crossLimit)
          ∧ (meanVote O F p ≤ B.lo → μ.real {ω | B.hi < voteCount O.mq F p ω} ≤ crossLimit)) ∧
        1 - δ ≤ (runMeasure μ D Dsf).real
          {x | (∃ B : {B : State // B ∈ states},
                x ∈ ret O.mq populations uni v indecisionLimit α B.val)
            ∧ ∀ B : {B : State // B ∈ states},
              x ∈ ret O.mq populations uni v indecisionLimit α B.val →
              ∀ j ∈ populations, 1 - εcov
                ≤ (D j).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
                    (familyAt O.mq populations x B.val) p (oracleNoise x)}
                ∧ (D j).real {p | ¬ decided O.mq B.val.lo (B.val.hi + 1)
                    (familyAt O.mq populations x B.val) p (oracleNoise x)}
                  ≤ 2 * indecisionLimit}

open MeasureTheory ProbabilityTheory in
/-- From `clustering_correct`, with the schedule and collision cap it names as the witnesses,
`prefCount_le_poly` for the prefix count, and `famCount_succ_le` and `poolCount_le` for the
family and the pool. -/
theorem clustering_guarantee_all : ClusteringGuaranteeAll := by
  intro Ω _ μ _ S _ J _ O populations uni Pre Suf η₀ indecisionLimit εcov₀ α δ pAP crossLimit v
    hηle hη₀ huni hflat hpAPPositive hindLim hind1 hαpos hα hεcov₀ hε1₀ hδ hδ1 hζ hζ1 hv0
  classical
  have hpop : populations.Nonempty := ⟨uni, huni⟩
  simp only [ret_eq, familyAt_eq, clusterAt_eq]
  generalize (lloydClusterer : Clusterer S) = rule
  have hsig : 0 < sig η₀ := by simp only [sig]; linarith
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hηle
  have hs : 0 < 1 / 2 - η₀ := by linarith
  -- the proof runs at a cut budget small against the signal and one prefix in the veto's draws
  obtain ⟨εcov, hεdef⟩ : ∃ e : ℝ, e = min εcov₀ (min (1 / 2 - η₀)
      (1 / ((populations.card : ℝ) * v))) := ⟨_, rfl⟩
  have hcardR : (0 : ℝ) < (populations.card : ℝ) := by exact_mod_cast Finset.card_pos.2 hpop
  have hvR : (0 : ℝ) < (v : ℝ) := by exact_mod_cast hv0
  have hδv : 0 < 1 / ((populations.card : ℝ) * v) := by positivity
  have hεcov : 0 < εcov := by rw [hεdef]; exact lt_min hεcov₀ (lt_min hs hδv)
  have hε1 : εcov ≤ 1 := by rw [hεdef]; exact le_trans (min_le_left _ _) hε1₀
  have hεle : εcov ≤ εcov₀ := by rw [hεdef]; exact min_le_left _ _
  have hveto : (populations.card : ℝ) * v * εcov ≤ 1 := by
    have h : εcov ≤ 1 / ((populations.card : ℝ) * v) := by
      rw [hεdef]; exact min_le_of_right_le (min_le_right _ _)
    rw [le_div_iff₀ (by positivity)] at h
    linarith
  have hscale : min εcov (min (1 / 2 - η₀) indecisionLimit)
      = cutScale populations η₀ indecisionLimit εcov₀ v := by
    rw [hεdef]; exact cutScale_eq populations η₀ indecisionLimit εcov₀ v
  have hscale2 : min εcov indecisionLimit = cutScale populations η₀ indecisionLimit εcov₀ v := by
    rw [hεdef]; exact cutScale_eq' populations η₀ indecisionLimit εcov₀ v
  have hS : 0 < cutScale populations η₀ indecisionLimit εcov₀ v :=
    lt_min hεcov₀ (lt_min hs (lt_min hδv hindLim))
  refine ⟨collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit, ?_, ?_⟩
  · simp only [collisionCap]
    positivity
  intro D Dsf hD hDsf hsupp hsuppSf hpAPBound ρ hρ hρcap hρsf
  have := hD
  have := hDsf
  have hpAP1 : pAP ≤ 1 := le_trans hpAPBound measureReal_le_one
  have hpoly := prefCount_le_poly populations η₀ indecisionLimit εcov δ α pAP crossLimit
    hsig hη0 hpop hindLim hεcov hε1 hδ hδ1 hαpos (by linarith) hpAPPositive hpAP1
  refine ⟨stoppable η₀ populations indecisionLimit εcov εcov₀ δ α pAP crossLimit ρ,
    ?_, ?_, ?_, fun B hB => cross_of_mem_schedule O hη0 hη₀ hζ
      (Finset.mem_of_mem_filter _ hB), ?_⟩
  · -- Every rung's count is the top one halved, and the top one is what `budgetCap` bounds.
    intro B hB
    have hsched : B ∈ schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit :=
      Finset.mem_of_mem_filter _ hB
    obtain ⟨i, _, rfl⟩ := Finset.mem_image.1 hsched
    have hp := hpoly
    simp only [budgetCap, budgetLog, budgetScale, sig] at hp
    rw [hscale] at hp
    refine le_trans ?_ hp
    exact_mod_cast Nat.div_le_self _ _
  · -- The ladder reaches down to `validCount`, which `validCount_le_poly` bounds.
    have hρ0 : 0 ≤ ρ :=
      le_trans (tsum_nonneg (fun a => sq_nonneg _)) (hρ hpop.choose hpop.choose_spec)
    have hpAP' := hpAPBound
    rw [O.apSet_eq] at hpAP'
    obtain ⟨Bp, hBp, -⟩ := exists_passable O populations D Dsf indecisionLimit εcov εcov₀ α δ ρ
      (collisionMass Dsf) pAP v (lt_of_le_of_lt hηle hη₀) hpop hηle hη₀ hεcov
      hε1 hεle hδ hδ1 hveto hαpos hα
      hindLim hind1 hpAPPositive hpAP' hρ hρ0 le_rfl (tsum_nonneg (fun a => sq_nonneg _)) hρcap
      hρsf
    obtain ⟨B, hB, hBle⟩ := exists_small_stoppable η₀ populations hη0 hη₀ hεcov hεcov₀ hindLim hδ
      hδ1 hpAPPositive hcardR hρ0 hρcap ⟨Bp, hBp⟩
    refine ⟨B, hB, ?_⟩
    have hv := validCount_le_poly populations η₀ indecisionLimit εcov εcov₀ δ pAP crossLimit hsig
      hη0 hpop hindLim hind1 hεcov hε1 hεcov₀ hε1₀ hδ hδ1 hpAPPositive hpAP1
    rw [nsuff_of_mem_schedule (Finset.mem_of_mem_filter _ hB)]
    simp only [sig] at hv
    have hlog : 0 ≤ Real.log (((populations.card : ℝ) + 2)
        * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 2) / δ) := by
      refine Real.log_nonneg ?_
      rw [le_div_iff₀ hδ]
      nlinarith [(Nat.cast_nonneg _ : (0 : ℝ) ≤ (poolCount η₀ populations indecisionLimit εcov δ
        pAP crossLimit : ℝ)), (Nat.cast_nonneg _ : (0 : ℝ) ≤ (populations.card : ℝ))]
    have hQ : 0 ≤ (populations.card : ℝ) ^ 2 * Real.log (((populations.card : ℝ) + 2)
        * ((poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 2) / δ)
        / ((1 / 2 - η₀) ^ 4 * min εcov₀ indecisionLimit ^ 2) := by
      refine div_nonneg (mul_nonneg (sq_nonneg _) hlog) ?_
      have : 0 < min εcov₀ indecisionLimit := lt_min hεcov₀ hindLim
      positivity
    rw [mul_assoc, mul_div_assoc]
    rw [mul_assoc, mul_div_assoc] at hv
    linarith
  · intro B hB
    obtain ⟨i, _, rfl⟩ := Finset.mem_image.1 (Finset.mem_of_mem_filter _ hB)
    have hk := famCount_succ_le populations η₀ indecisionLimit εcov δ crossLimit hsig hη0
      hindLim hεcov hζ hζ1
    have hpool := poolCount_le populations η₀ indecisionLimit εcov δ pAP crossLimit hsig hη0
      hindLim hεcov hpop hδ hδ1 hpAPPositive hpAP1 hζ hζ1
    simp only [budgetScale, sig] at hk hpool
    rw [hscale] at hk hpool
    rw [mul_div_assoc] at hk
    have hG : 0 ≤ Real.log (2 / (cutScale populations η₀ indecisionLimit εcov₀ v * crossLimit))
        / (1 / 2 - η₀) ^ 2 := by
      linarith only [hk, (Nat.cast_nonneg _ : (0 : ℝ) ≤ famCount η₀ populations
        indecisionLimit εcov δ crossLimit)]
    have hGp : 0 ≤ Real.log (2 / (cutScale populations η₀ indecisionLimit εcov₀ v * crossLimit))
        / ((1 / 2 - η₀) ^ 2 * pAP) := by
      rw [← div_div]; exact div_nonneg hG hpAPPositive.le
    have hH : 0 ≤ Real.log (((populations.card : ℝ) + 2) / δ) / pAP ^ 2 :=
      div_nonneg (Real.log_nonneg (by
        rw [le_div_iff₀ hδ]; linarith [(Nat.cast_nonneg populations.card : (0 : ℝ) ≤ _)]))
        (sq_nonneg _)
    simp only [solvedStateAt]
    push_cast
    rw [mul_div_assoc, mul_div_assoc 128, mul_div_assoc 16]
    rw [mul_div_assoc 128, mul_div_assoc 16] at hpool
    constructor <;> linarith
  exact clustering_correct O rule populations uni D Dsf Pre Suf η₀ indecisionLimit εcov εcov₀
    α δ ρ pAP crossLimit 12000 v hηle hη₀ huni hflat hsupp hsuppSf hρ hpAPPositive hpAPBound
    hindLim hind1 hαpos hα hεcov hε1 hεle hδ hpoly hρcap hρsf hv0 hveto

open MeasureTheory ProbabilityTheory in
/-- From `clustering_guarantee_all` at `min εcov (1/2)`, which leaves every stated bound as it is:
each already takes the minimum with `1/2 − η₀` or with `indecisionLimit ≤ 1/2`. -/
theorem clustering_guarantee_of_correct : ClusteringGuarantee := by
  intro Ω _ μ _ S _ J _ O populations uni Pre Suf η₀ indecisionLimit εcov α δ pAP crossLimit v
    hηle hη₀ huni hflat hpAPPositive hindLim hind1 hαpos hα hεcov hε1 hδ hδ1 hζ hζ1 hv0
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hηle
  have e1 : ∀ x : ℝ, min (min εcov (1 / 2)) (min (1 / 2 - η₀) x) = min εcov (min (1 / 2 - η₀) x) := by
    intro x; rw [min_assoc]; congr 1; exact min_eq_right (le_trans (min_le_left _ _) (by linarith))
  have e2 : min (min εcov (1 / 2)) indecisionLimit = min εcov indecisionLimit := by
    rw [min_assoc]; congr 1; exact min_eq_right hind1
  obtain ⟨cap, hcap, h⟩ := clustering_guarantee_all O populations uni Pre Suf η₀ indecisionLimit
    (min εcov (1 / 2)) α δ pAP crossLimit v hηle hη₀ huni hflat hpAPPositive hindLim hind1
    hαpos hα (lt_min hεcov (by norm_num)) (le_trans (min_le_right _ _) (by norm_num)) hδ hδ1 hζ hζ1
    hv0
  refine ⟨cap, hcap, fun D Dsf hD hDsf hsupp hsuppSf hpAPBound ρ hρ hρcap hρsf => ?_⟩
  obtain ⟨states, h1, h2, h3, h4, h5⟩ :=
    h D Dsf hD hDsf hsupp hsuppSf hpAPBound ρ hρ hρcap hρsf
  simp only [cutScale, e1, e2] at h1 h2 h3 ⊢
  refine ⟨states, h1, h2, h3, h4, le_trans h5 (measureReal_mono ?_ (measure_ne_top _ _))⟩
  rintro x ⟨hex, hall⟩
  refine ⟨hex, fun B hB => ⟨?_, fun j hj => ⟨?_, (hall B hB j hj).2⟩⟩⟩
  · linarith [(hall B hB uni huni).1, min_le_left εcov (1 / 2)]
  · linarith [(hall B hB j hj).1, min_le_right εcov (1 / 2)]

end OrthoDFA
