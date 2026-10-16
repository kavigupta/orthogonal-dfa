import OrthoDFA.Proofs.Lloyd
import OrthoDFA.Proofs.GateLoop
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# What the budget costs

`prefCount` is the count the round's tails ask for, and `ClusteringCorrect` quantifies over
the states the schedule built from it.  Written out it is a sum of five ceilings, each a log
over the square of some margin, and nothing in the statement says how those margins depend on
the rates they resolve.  This file says it: every term is bounded by one polynomial in the
reciprocals of `sig η` and `budgetScale`, so the budget is `log` in the failure probability and
polynomial in everything else.

The two scales are not independent.  `screenMargin` carries `cutBudget · sig²`, and the tail
it sits in squares it, so `sig` enters at the fourth power — and `cutBudget` is itself a minimum
over `εcov`, `sig` and `indecisionLimit`, so a bound in `sig` alone cannot exist.

With that bound `ClusteringGuarantee` follows from `ClusteringCorrect`, at the end.
-/

namespace OrthoDFA

-- `Finset J` carries the instance the definitions were written with; these bounds read
-- only the card, so the linter is right that they do not use it themselves.
set_option linter.unusedFintypeInType false

variable {J : Type*} [Fintype J]

/-- Every ceiling is under its argument plus one, and under one when the argument is
negative -- which is what lets the five terms be bounded separately and summed. -/
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

/-- The five tails, each named, so the bound can be read term by term. -/
noncomputable def prefTerms (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ α pAP crossLimit : ℝ) : List ℝ :=
  [ Real.log (128 * (populations.card : ℝ)
      * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 2) ^ 2 / δ)
      / (2 * (screenMargin η populations indecisionLimit εcov δ / 2) ^ 2),
    Real.log (128 * (populations.card : ℝ) / δ)
      / (2 * (cutBudget η indecisionLimit εcov / 32) ^ 2),
    16 * Real.log (1664 * (populations.card : ℝ) ^ 2
        * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 3) / (α * δ))
      / (cutBudget η indecisionLimit εcov ^ 2 * sig η ^ 4),
    128 / (cutBudget η indecisionLimit εcov ^ 2 * sig η ^ 4)
      * Real.log (512 / (cutBudget η indecisionLimit εcov ^ 2 * sig η ^ 4)),
    Real.log (128 * (populations.card : ℝ)
        * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ)
      / (2 * (3 * cutBudget η indecisionLimit εcov * flipFrac η / 32) ^ 2) ]

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

/-- The count is exactly its five tails, each rounded up, and the one rung the ladder always
has. -/
theorem prefCount_eq_sum (populations : Finset J)
    (η indecisionLimit εcov δ α pAP crossLimit : ℝ) :
    prefCount η populations indecisionLimit εcov δ α pAP crossLimit
      = ((prefTerms η populations indecisionLimit εcov δ α pAP crossLimit).map
          (fun x => ⌈x⌉₊)).sum + 1 := by
  simp only [prefCount, prefTerms, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  ring

/-- The mechanical half of the bound: the count is under its five tails once each rounding is
paid for.  Nothing analytic happens here -- what the tails are worth is `prefCount_le_poly`. -/
theorem prefCount_le_terms (populations : Finset J)
    (η indecisionLimit εcov δ α pAP crossLimit : ℝ) :
    (prefCount η populations indecisionLimit εcov δ α pAP crossLimit : ℝ)
      ≤ ((prefTerms η populations indecisionLimit εcov δ α pAP crossLimit).map
          (fun x => max x 0)).sum + 6 := by
  have h :=
    cast_sum_ceil_le (prefTerms η populations indecisionLimit εcov δ α pAP crossLimit)
  rw [prefCount_eq_sum]
  have hlen :
      (prefTerms η populations indecisionLimit εcov δ α pAP crossLimit).length = 5 := by
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
        ≤ (⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (9 * pAP / 20) ^ 2)⌉₊ : ℝ))]
  have hdiv : 2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1)
      ≤ 2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1) / pAP := by
    rw [le_div_iff₀ hpAP]
    nlinarith only [hpAP1, hfam]
  linarith only [hpool, hdiv, hfam]

set_option maxHeartbeats 1000000 in
/-- The budget is polynomial in the reciprocals of the rates it resolves.

`sig` enters at the fourth power because the screen's tail and the gate's are the binding
ones: their margins are `cutBudget · sig²`, and a deviation bound squares the margin it is
given.  The population count enters squared for the same reason -- `flipBudget` divides by it,
and `screenMargin` inherits that.

The pool's size is left inside the log rather than bounded in the rates: the screen's tail
union-bounds over its pairs and the flip tail over its members, and nothing else reads it.

`0 ≤ η` is what bounds `sig η` above; without it the noise rate may be negative, `sig` may be
arbitrarily large, and the `sig ^ 4` in the denominator sends the bound below the count.

The terms are bounded one at a time here, each at its worst with `sig η` and the scale at
`1/2`, and their bounds sum to under 100000; the gate's log of its own count is most of it. -/
theorem prefCount_le_poly (populations : Finset J)
    (η indecisionLimit εcov δ α pAP crossLimit : ℝ)
    (hsig : 0 < sig η) (hη : 0 ≤ η) (hpop : populations.Nonempty)
    (hind : 0 < indecisionLimit) (hεcov : 0 < εcov) (hεcov1 : εcov ≤ 1)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hα : 0 < α) (hα1 : α < 1)
    (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) :
    (prefCount η populations indecisionLimit εcov δ α pAP crossLimit : ℝ)
      ≤ 100000 * (populations.card : ℝ) ^ 2
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
  have hsm : 1155 * B * s ^ 2
      ≤ 20480 * (c * screenMargin η populations indecisionLimit εcov δ) := by
    have hne : (1 : ℝ) - η ≠ 0 := by intro h; linarith only [h, hηlt]
    have e : 20480 * (c * screenMargin η populations indecisionLimit εcov δ)
        = 11550 * cut * s ^ 2 := by
      simp only [screenMargin, flipBudget, flipFrac, ← hsdef, ← hcdef, ← hcutdef]
      field_simp
      ring
    rw [e]
    nlinarith only [mul_le_mul_of_nonneg_right hcutlb (sq_nonneg s)]
  have hsmpos : 0 < screenMargin η populations indecisionLimit εcov δ := by
    have h7 : (0 : ℝ) < 1155 * B * s ^ 2 := by positivity
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
  -- the gate's two logs, against `L`
  have hT1 : (c + 2) * M ≤ T := by
    have hd1 : d ≤ 1 := by
      rw [hddef]
      have : δ * α * pAP ≤ 1 := by
        have hδα : δ * α ≤ 1 := by nlinarith only [le_mul2 hδ.le hα.le hδ1 hα1.le]
        nlinarith only [le_mul2 (mul_pos hδ hα).le hpAP.le hδα hpAP1]
      nlinarith only [le_mul2 (mul_pos (mul_pos hδ hα) hpAP).le hBpos.le this hB2]
    rw [hTdef, le_div_iff₀ hdpos]
    nlinarith only [hcM0, hd1, hdpos]
  have hG1 : Real.log (1664 * c ^ 2 * (M + 3) / (α * δ)) ≤ 4 * L := by
    have harg : 1664 * c ^ 2 * (M + 3) / (α * δ) ≤ T ^ 4 := by
      rw [div_le_iff₀ (mul_pos hα hδ)]
      have h1 : 2 * (c + 2) * M ≤ T * δ := by linarith only [hTδ]
      have hT2 : ((c + 2) * M) ^ 2 ≤ T ^ 2 := pow_le_pow_left₀ hcM0 hT1 2
      have hcM : 1664 * c ^ 2 * (M + 3) ≤ 2 * (c + 2) * M * ((c + 2) * M) ^ 2 := by
        have hc3 : c ^ 2 ≤ (c + 2) ^ 3 := by nlinarith only [hcpos, sq_nonneg c]
        have hM3 : 1664 * (M + 3) ≤ 2 * M ^ 3 := by
          have h1 : 256 * M ≤ M ^ 2 := by nlinarith only [hM256, hM0]
          have h2 : 256 * M ^ 2 ≤ M ^ 3 := by nlinarith only [hM256, hM0, sq_nonneg M]
          nlinarith only [h1, h2, hM256]
        calc 1664 * c ^ 2 * (M + 3) = c ^ 2 * (1664 * (M + 3)) := by ring
          _ ≤ c ^ 2 * (2 * M ^ 3) := mul_le_mul_of_nonneg_left hM3 (sq_nonneg c)
          _ ≤ (c + 2) ^ 3 * (2 * M ^ 3) := mul_le_mul_of_nonneg_right hc3 (by positivity)
          _ = 2 * (c + 2) * M * ((c + 2) * M) ^ 2 := by ring
      calc 1664 * c ^ 2 * (M + 3) ≤ 2 * (c + 2) * M * ((c + 2) * M) ^ 2 := hcM
        _ ≤ (T * δ) * T ^ 2 := mul_le_mul h1 hT2 (by positivity) (by positivity)
        _ ≤ (T * δ) * T ^ 2 * (T * α) := le_mul_of_one_le_right (by positivity) hTα
        _ = T ^ 4 * (α * δ) := by ring
    have h := log_le_of_le_pow (by positivity) 4 harg
    rw [hLdef]; push_cast at h; linarith only [h]
  set X : ℝ := cut ^ 2 * s ^ 4 with hXdef
  have hXpos : 0 < X := by positivity
  have hXB : B ^ 6 ≤ 100 * X := by
    have h1 : B ^ 2 ≤ 100 * cut ^ 2 := by nlinarith only [hcutlb, hBpos]
    have h2 : B ^ 4 ≤ s ^ 4 := pow_le_pow_left₀ hBpos.le hBs 4
    rw [hXdef]
    nlinarith only [mul_le_mul h1 h2 (by positivity) (by positivity)]
  have hG2 : Real.log (512 / X) ≤ 6 * L := by
    have hTB : 768 ≤ T * B := by
      rw [hTdef, div_mul_eq_mul_div, le_div_iff₀ hdpos, hddef]
      have hδαp : δ * α * pAP ≤ 1 := by
        have hδα : δ * α ≤ 1 := by nlinarith only [le_mul2 hδ.le hα.le hδ1 hα1.le]
        nlinarith only [le_mul2 (mul_pos hδ hα).le hpAP.le hδα hpAP1]
      have hcM768 : 768 ≤ (c + 2) * M := by nlinarith only [hc1, hM256]
      nlinarith only [hcM768, mul_le_mul_of_nonneg_right hδαp hBpos.le, hBpos]
    have harg : 512 / X ≤ T ^ 6 := by
      rw [div_le_iff₀ hXpos]
      have h6 : (768 : ℝ) ^ 6 ≤ (T * B) ^ 6 := pow_le_pow_left₀ (by norm_num) hTB 6
      have e : (T * B) ^ 6 = T ^ 6 * B ^ 6 := by ring
      nlinarith only [h6, e, hXB, pow_pos hTpos 6]
    have h := log_le_of_le_pow (by positivity) 6 harg
    rw [hLdef]; push_cast at h; linarith only [h]
  -- the five tails, each against the common quotient
  have hD : (0 : ℝ) < s ^ 4 * B ^ 2 := by positivity
  set Q : ℝ := c ^ 2 * L / (s ^ 4 * B ^ 2) with hQdef
  have hQnn : 0 ≤ Q := by positivity
  have hB2c : B ^ 2 ≤ 100 * cut ^ 2 := by
    have := mul_self_le_mul_self hBpos.le hcutlb
    nlinarith only [this]
  have t1 : Real.log (128 * c * (M + 2) ^ 2 / δ)
      / (2 * (screenMargin η populations indecisionLimit εcov δ / 2) ^ 2) ≤ 2000 * Q := by
    refine tail_le hX hL0 (by positivity) hD ?_
    have h := mul_self_le_mul_self (by positivity) hsm
    nlinarith only [h]
  have t2 : Real.log (128 * c / δ) / (2 * (cut / 32) ^ 2) ≤ 3200 * Q := by
    refine tail_le hR hL0 (by positivity) hD ?_
    nlinarith only [hB2c, mul_le_mul_of_nonneg_right hs4 (sq_nonneg B),
      mul_le_mul_of_nonneg_right hc2 (sq_nonneg cut)]
  have hgateK : ∀ w : ℝ, 0 ≤ w → w * (s ^ 4 * B ^ 2) ≤ 100 * w * c ^ 2 * X := by
    intro w hw
    have h2 : w * (s ^ 4 * B ^ 2) ≤ w * (s ^ 4 * (100 * cut ^ 2)) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hB2c (by positivity)) hw
    have h3 : w * (s ^ 4 * (100 * cut ^ 2)) ≤ 100 * w * c ^ 2 * X := by
      rw [hXdef]
      nlinarith only [mul_le_mul_of_nonneg_left hc2
        (by positivity : (0 : ℝ) ≤ 100 * w * (cut ^ 2 * s ^ 4))]
    linarith only [h2, h3]
  have t3 : 16 * Real.log (1664 * c ^ 2 * (M + 3) / (α * δ)) / X ≤ 6400 * Q := by
    refine tail_le (w := 64) (by linarith only [hG1]) hL0 hXpos hD ?_
    have := hgateK 64 (by norm_num)
    linarith only [this]
  have t4 : 128 / X * Real.log (512 / X) ≤ 76800 * Q := by
    rw [show 128 / X * Real.log (512 / X) = 128 * Real.log (512 / X) / X by ring]
    refine tail_le (w := 768) (by linarith only [hG2]) hL0 hXpos hD ?_
    have := hgateK 768 (by norm_num)
    linarith only [this]
  have t5 : Real.log (128 * c * (M + 1) / δ) / (2 * (3 * cut * flipFrac η / 32) ^ 2)
      ≤ 6000 * Q := by
    have hfpos : 0 < flipFrac η := by linarith only [hf, hsig]
    refine tail_le hY hL0
      (mul_pos two_pos (pow_pos (div_pos (mul_pos (mul_pos three_pos hcutpos) hfpos)
        (by norm_num)) 2)) hD ?_
    have h := mul_self_le_mul_self (by positivity) hcf
    have hcf2 : 0 ≤ (cut * flipFrac η) ^ 2 := sq_nonneg _
    nlinarith only [h, mul_le_mul_of_nonneg_right hs2sq (by positivity : (0 : ℝ) ≤ s ^ 2 * B ^ 2),
      mul_le_mul_of_nonneg_right hc2 hcf2, hcf2]
  have t7 : (6 : ℝ) ≤ Q := by
    rw [hQdef, le_div_iff₀ hD]
    have hc2L : (1 : ℝ) ≤ c ^ 2 * L := by linarith only [le_mul2 zero_le_one zero_le_one hc2 hL1]
    have hBsq : B ^ 2 ≤ 1 / 4 := by
      have h := pow_le_pow_left₀ hBpos.le hB2 2; norm_num at h; linarith only [h]
    linarith only [le_mul2 (pow_nonneg hsig.le 4) (sq_nonneg B) hs4 hBsq, hc2L]
  -- and the sum
  have hbound := prefCount_le_terms populations η indecisionLimit εcov δ α pAP
    crossLimit
  simp only [prefTerms, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    ← hcdef, ← hsdef, ← hcutdef, ← hMdef, ← hXdef] at hbound
  have m1 := max_le t1 (by positivity)
  have m2 := max_le t2 (by positivity)
  have m3 := max_le t3 (by positivity)
  have m4 := max_le t4 (by positivity)
  have m5 := max_le t5 (by positivity)
  rw [hLeq, show 100000 * c ^ 2 * L / (s ^ 4 * B ^ 2) = 100000 * Q by rw [hQdef]; ring]
  linarith only [hbound, m1, m2, m3, m4, m5, t7, hQnn]

/-- `validCount` is its one tail, rounded up, and the ladder's rung. -/
theorem validCount_le (populations : Finset J) (η indecisionLimit εcov δ pAP crossLimit : ℝ)
    (hind : 0 < indecisionLimit) (hpop : populations.Nonempty) (hδ : 0 < δ) (hδ1 : δ ≤ 1) :
    (validCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ)
      ≤ 2 * Real.log (128 * (populations.card : ℝ) / δ) / indecisionLimit ^ 2 + 2 := by
  have hc1 : (1 : ℝ) ≤ (populations.card : ℝ) := Nat.one_le_cast.2 (Finset.card_pos.2 hpop)
  have hlog : 0 ≤ Real.log (128 * (populations.card : ℝ) / δ) :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ]; nlinarith)
  have e : Real.log (128 * (populations.card : ℝ) / δ) / (2 * (indecisionLimit / 2) ^ 2)
      = 2 * Real.log (128 * (populations.card : ℝ) / δ) / indecisionLimit ^ 2 := by
    field_simp
  rw [validCount]
  push_cast
  have := ceil_le_of_le (le_of_eq e) (by positivity)
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
  -- `H = log ((c+2)/δ)` is at least `log 3`
  set H : ℝ := Real.log ((c + 2) / δ) with hHdef
  have hl2 := half_le_log_two
  have hH3 : Real.log 3 ≤ H :=
    Real.log_le_log (by norm_num) (by rw [le_div_iff₀ hδ]; linarith only [hc1, hδ1])
  have hH2 : Real.log 2 ≤ H := le_trans (Real.log_le_log (by norm_num) (by norm_num)) hH3
  have hp2pos : (0 : ℝ) < pAP ^ 2 := pow_pos hpAP 2
  set R : ℝ := H / pAP ^ 2 with hRdef
  have hRhalf : 1 / 2 ≤ R := by
    rw [hRdef, le_div_iff₀ hp2pos]
    linarith only [pow_le_one₀ hpAP.le hpAP1 (n := 2), hH2, hl2]
  -- the findability tail, read at `9pAP/20`: `log 128 = 7 log 2`, and `H ≥ log 3 ≥ (3/2) log 2`
  have hl2' := Real.log_two_gt_d9
  have hlog3 : 3 / 2 * Real.log 2 ≤ Real.log 3 := by
    have h : Real.log ((2 : ℝ) ^ 3) ≤ Real.log ((3 : ℝ) ^ 2) :=
      Real.log_le_log (by norm_num) (by norm_num)
    rw [Real.log_pow, Real.log_pow] at h
    push_cast at h
    linarith only [h]
  have hlog128' : Real.log (128 * c / δ) ≤ 7 * Real.log 2 + H := by
    have h1 : Real.log (128 * c / δ) ≤ Real.log (2 ^ 7 * ((c + 2) / δ)) :=
      Real.log_le_log (div_pos (by linarith only [hcpos]) hδ) (by
        rw [mul_div_assoc]
        exact mul_le_mul (by norm_num)
          (div_le_div_of_nonneg_right (by linarith only : c ≤ c + 2) hδ.le)
          (div_nonneg hcpos.le hδ.le) (by norm_num))
    rw [Real.log_mul (by norm_num) (by positivity), Real.log_pow, Nat.cast_ofNat] at h1
    linarith only [h1]
  have hC : Real.log (128 * c / δ) / (2 * (9 * pAP / 20) ^ 2) + 2 ≤ 16 * R := by
    have hpAP2 : pAP ^ 2 ≤ 1 := pow_le_one₀ hpAP.le hpAP1
    have hR2 : 2 ≤ 2 / pAP ^ 2 := by
      rw [le_div_iff₀ hp2pos]; linarith only [hpAP2]
    have e : Real.log (128 * c / δ) / (2 * (9 * pAP / 20) ^ 2)
        = 200 / 81 * Real.log (128 * c / δ) / pAP ^ 2 := by
      field_simp
      ring
    rw [e, hRdef]
    have key : 200 / 81 * Real.log (128 * c / δ) + 2 ≤ 16 * H := by
      linarith only [hlog128', hH3, hlog3, hl2']
    have key' : 200 / 81 * Real.log (128 * c / δ) / pAP ^ 2 + 2 / pAP ^ 2 ≤ 16 * H / pAP ^ 2 := by
      rw [← add_div]
      exact div_le_div_of_nonneg_right key hp2pos.le
    calc 200 / 81 * Real.log (128 * c / δ) / pAP ^ 2 + 2
        ≤ 200 / 81 * Real.log (128 * c / δ) / pAP ^ 2 + 2 / pAP ^ 2 := by linarith only [hR2]
      _ ≤ 16 * H / pAP ^ 2 := key'
      _ = 16 * (H / pAP ^ 2) := by ring
  have h1 := ceil_le_of_le hA (div_nonneg (by linarith only [hPnn]) hpAP.le)
  have h2 := Nat.ceil_lt_add_one (show (0 : ℝ) ≤ Real.log (128 * c / δ)
    / (2 * (9 * pAP / 20) ^ 2) from div_nonneg (Real.log_nonneg (by
      rw [le_div_iff₀ hδ]; nlinarith only [hc1, hδ1])) (by positivity))
  have hpoolval : (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ)
      = (⌈2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1) / pAP⌉₊ : ℝ)
        + (⌈Real.log (128 * c / δ) / (2 * (9 * pAP / 20) ^ 2)⌉₊ : ℝ) := by
    rw [hcdef, poolCount]
    push_cast
    ring
  have hPe : 2 * P / pAP
      = 128 * Real.log (2 / (budgetScale η indecisionLimit εcov * crossLimit))
        / (sig η ^ 2 * pAP) := by
    rw [hPdef]; ring
  rw [hpoolval, ← hPe, mul_div_assoc 16]
  linarith only [h1, h2, hC]

/-- The statement's hypotheses at the top rung are the collision caps. -/
lemma caps_of_top {J : Type*} [Fintype J] (populations : Finset J)
    (η₀ indecisionLimit εcov δ α pAP crossLimit ρ ρsf : ℝ)
    (hN : (0 : ℝ) < (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ))
    (hM : (0 : ℝ) < (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ))
    (h1 : 2 ^ 20 * ((populations.card : ℝ) + 1) ^ 2
        * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ) * ρ
      ≤ δ * min εcov (min (1 / 2 - η₀) indecisionLimit) ^ 2)
    (h2 : 2 ^ 12 * ((populations.card : ℝ) + 1)
        * (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) * ρsf ≤ δ * pAP) :
    ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit
      ∧ ρsf ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit := by
  have hJ : (0 : ℝ) < (populations.card : ℝ) + 1 := by positivity
  constructor
  · rw [collisionCap, budgetScale, sig, le_div_iff₀ (by positivity)]
    linarith
  · rw [poolCollisionCap, le_div_iff₀ (by positivity)]
    linarith

open MeasureTheory ProbabilityTheory in
/-- From `clustering_correct`, with the schedule at `cutLimit` and the collision caps it names as
the witnesses, `prefCount_le_poly` for the prefix count, `validCount_le` for the lowest rung, and
`famCount_succ_le` and `poolCount_le` for the family and the pool.  The states are the rungs that
carry their share at the collision cap, which the top rung always does; the statement's
hypotheses at the top rung are the caps.  `cutLimit` sits under `εcov`, so the scale's minimum
need not take `εcov` again. -/
theorem clustering_guarantee_of_correct : ClusteringGuarantee := by
  intro Ω _ μ _ S _ J _ O populations uni Pre Suf η₀ indecisionLimit εcov α δ pAP pBal crossLimit
    hηle hη₀ huni hflat hpAPPositive hpBal hindLim hind1 hαpos hα hεcov hε1 hδ hδ1 hζ hζ1
  classical
  have hpop : populations.Nonempty := ⟨uni, huni⟩
  simp only [ret_eq, familyAt_eq]
  generalize (lloydClusterer : Clusterer S) = rule
  have hsig : 0 < sig η₀ := by simp only [sig]; linarith
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hηle
  have hcard : (0 : ℝ) < (populations.card : ℝ) := by exact_mod_cast Finset.card_pos.2 hpop
  set iLc : ℝ := cutLimit η₀ indecisionLimit εcov pBal with hiLc
  have hs : 0 < 1 / 2 - η₀ := by linarith
  have hiLc0 : 0 < iLc := lt_min hindLim (lt_min (by positivity) (by positivity))
  have hiLc1 : iLc ≤ indecisionLimit := min_le_left _ _
  have hiLcε : iLc ≤ εcov / 64 := (min_le_right _ _).trans (min_le_right _ _)
  have hiLch : iLc ≤ 1 / 2 := le_trans hiLc1 hind1
  have hscaleEq : budgetScale η₀ iLc εcov = min (1 / 2 - η₀) iLc := by
    simp only [budgetScale, sig]
    exact min_eq_right (le_trans (min_le_right _ _) (by linarith))
  have hscale : 0 < min (1 / 2 - η₀) iLc := lt_min hs hiLc0
  have hscale1 : min (1 / 2 - η₀) iLc ≤ 1 / 2 := le_trans (min_le_right _ _) hiLch
  have hlog128 : 1 ≤ Real.log (128 * (populations.card : ℝ) / δ) := by
    have hc1 : (1 : ℝ) ≤ (populations.card : ℝ) := Nat.one_le_cast.2 (Finset.card_pos.2 hpop)
    have h : Real.exp 1 ≤ 128 * (populations.card : ℝ) / δ := by
      rw [le_div_iff₀ hδ]
      nlinarith [Real.exp_one_lt_d9]
    have := Real.log_le_log (Real.exp_pos 1) h
    rwa [Real.log_exp] at this
  by_cases hpAP1 : pAP ≤ 1
  swap
  · -- No measure puts more than all its mass anywhere, so no suffix measure meets `pAP`, and a
    -- single empty state carries the bounds.
    have hlogB : 0 ≤ Real.log (2 / (min (1 / 2 - η₀) iLc * crossLimit)) :=
      Real.log_nonneg (by
        rw [le_div_iff₀ (mul_pos hscale hζ)]
        nlinarith [mul_le_mul hscale1 hζ1 hζ.le (by norm_num : (0 : ℝ) ≤ 1 / 2)])
    have hlogJ : 0 ≤ Real.log (((populations.card : ℝ) + 2) / δ) :=
      Real.log_nonneg (by rw [le_div_iff₀ hδ]; linarith)
    refine ⟨{⟨0, 0, 0, 0, 0, 0, 0⟩}, fun B hB => ?_,
      ⟨⟨0, 0, 0, 0, 0, 0, 0⟩, Finset.mem_singleton_self _, ?_⟩, fun B hB => ?_,
      fun B hB F hF => ?_, ?_⟩
    · rw [Finset.mem_singleton.1 hB]
      simp
    · simp only [Nat.cast_zero]
      positivity
    · rw [Finset.mem_singleton.1 hB]
      simp only [Nat.cast_zero]
      constructor <;> positivity
    · rw [Finset.mem_singleton.1 hB] at hF
      simp at hF
    · intro D Dsf hD hDsf _ _ hpAPBound
      exact absurd (le_trans hpAPBound measureReal_le_one) hpAP1
  set ρ₀ : ℝ := collisionCap η₀ populations iLc εcov δ α pAP crossLimit with hρ₀
  have hN : (0 : ℝ) < (prefCount η₀ populations iLc εcov δ α pAP crossLimit : ℝ) := by
    exact_mod_cast prefCount_pos (indecisionLimit := iLc)
      (crossLimit := crossLimit) η₀ populations εcov δ α pAP
  have hρ₀0 : 0 ≤ ρ₀ := by
    rw [hρ₀, collisionCap]
    have : 0 < budgetScale η₀ iLc εcov := by rw [hscaleEq]; exact hscale
    positivity
  have htop := solvedState_mem_stoppable (α := α) (crossLimit := crossLimit) η₀ populations hη0
    hη₀ hεcov hiLc0 hδ hpAPPositive hcard hρ₀0 le_rfl
  have hpoly := prefCount_le_poly populations η₀ iLc εcov δ α pAP crossLimit
    hsig hη0 hpop hiLc0 hεcov hε1 hδ hδ1 hαpos (by linarith) hpAPPositive hpAP1
  have hpoly' := hpoly
  simp only [budgetLog, hscaleEq, sig] at hpoly'
  refine ⟨stoppable η₀ populations iLc εcov δ α pAP crossLimit ρ₀,
    ?_, ?_, ?_, fun B hB => cross_of_mem_schedule O hη0 hη₀ hζ
      (Finset.mem_of_mem_filter _ hB), ?_⟩
  · -- Every rung's count is the top one halved, and the top one is what `budgetCap` bounds.
    intro B hB
    have hsched : B ∈ schedule η₀ populations iLc εcov δ α pAP crossLimit :=
      Finset.mem_of_mem_filter _ hB
    obtain ⟨i, _, rfl⟩ := Finset.mem_image.1 hsched
    refine le_trans ?_ hpoly'
    exact_mod_cast Nat.div_le_self _ _
  · -- The ladder reaches down to `validCount`.
    obtain ⟨B, hB, hBle⟩ := exists_small_stoppable η₀ populations hη0 hη₀ hεcov hiLc0 hδ hδ1
      hpAPPositive hcard hρ₀0 le_rfl ⟨_, htop⟩
    refine ⟨B, hB, ?_⟩
    have hv := validCount_le populations η₀ iLc εcov δ pAP crossLimit hiLc0 hpop hδ hδ1
    have hA : 4 ≤ Real.log (128 * (populations.card : ℝ) / δ) / iLc ^ 2 := by
      rw [le_div_iff₀ (by positivity)]
      nlinarith
    have e : 2 * Real.log (128 * (populations.card : ℝ) / δ) / iLc ^ 2
        = 2 * (Real.log (128 * (populations.card : ℝ) / δ) / iLc ^ 2) := by ring
    have e5 : 5 * Real.log (128 * (populations.card : ℝ) / δ) / iLc ^ 2
        = 5 * (Real.log (128 * (populations.card : ℝ) / δ) / iLc ^ 2) := by ring
    rw [e5]
    rw [e] at hv
    linarith
  · intro B hB
    obtain ⟨i, _, rfl⟩ := Finset.mem_image.1 (Finset.mem_of_mem_filter _ hB)
    have hk := famCount_succ_le populations η₀ iLc εcov δ crossLimit hsig hη0
      hiLc0 hεcov hζ hζ1
    have hpool := poolCount_le populations η₀ iLc εcov δ pAP crossLimit hsig hη0
      hiLc0 hεcov hpop hδ hδ1 hpAPPositive hpAP1 hζ hζ1
    rw [hscaleEq] at hk hpool
    simp only [sig] at hk hpool
    rw [mul_div_assoc] at hk
    have hG : 0 ≤ Real.log (2 / (min (1 / 2 - η₀) iLc * crossLimit)) / (1 / 2 - η₀) ^ 2 := by
      linarith only [hk, (Nat.cast_nonneg _ : (0 : ℝ) ≤ famCount η₀ populations
        iLc εcov δ crossLimit)]
    simp only [solvedStateAt]
    push_cast
    rw [mul_div_assoc, mul_div_assoc 128, mul_div_assoc 16]
    rw [mul_div_assoc 128, mul_div_assoc 16] at hpool
    constructor <;> linarith
  -- the statement's hypotheses, read at the top rung, are the caps
  intro D Dsf hD hDsf hsupp hsuppSf hpAPBound hpIn hpOut hcollD hcollSf
  haveI := hD
  haveI := hDsf
  have hM : (0 : ℝ) < (poolCount η₀ populations iLc εcov δ pAP crossLimit : ℝ) := by
    have h := poolCount_ge populations η₀ iLc εcov δ pAP crossLimit hsig hη0 hiLc0
      hεcov hε1 hpAPPositive hpAP1
    linarith
  have hmin : min εcov (min (1 / 2 - η₀) iLc) = min (1 / 2 - η₀) iLc := by
    rw [show min εcov (min (1 / 2 - η₀) iLc) = budgetScale η₀ iLc εcov from rfl, hscaleEq]
  have hcaps : ∀ j ∈ populations, collisionMass (D j) ≤ ρ₀ := fun j hj =>
    (caps_of_top populations η₀ iLc εcov δ α pAP crossLimit (collisionMass (D j)) 0
      hN hM (by rw [hmin]; exact hcollD _ htop j hj) (by simp only [mul_zero]; positivity)).1
  have hcapsf : collisionMass Dsf ≤ poolCollisionCap η₀ populations iLc εcov δ pAP crossLimit :=
    (caps_of_top populations η₀ iLc εcov δ α pAP crossLimit 0 (collisionMass Dsf)
      hN hM (by simp only [mul_zero]; positivity) (hcollSf _ htop)).2
  refine le_trans (clustering_correct O rule populations uni D Dsf Pre Suf η₀ indecisionLimit εcov
    α δ ρ₀ pAP pBal crossLimit 100000 hηle hη₀ huni hflat hsupp hsuppSf hcaps hpAPPositive
    hpAPBound hpBal hpIn hpOut hindLim hind1 hαpos hα hεcov hε1 hδ hpoly le_rfl hcapsf)
    (measureReal_mono ?_ (measure_ne_top _ _))
  rintro x ⟨hex, hall⟩
  refine ⟨hex, fun B hB => ⟨?_, fun j hj => ⟨fun hne => ?_, (hall B hB j hj).2⟩⟩⟩
  · have h := (hall B hB uni huni).1
    simp only [gateLimit, if_pos rfl] at h
    exact h
  · have h := (hall B hB j hj).1
    simp only [gateLimit, if_neg hne] at h
    linarith

end OrthoDFA
