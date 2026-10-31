import OrthoDFA.Proofs.Nearest
import OrthoDFA.Proofs.Correct
import OrthoDFA.Proofs.BandKappa
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# What the budget costs

`prefCount` is the count the round's tails ask for, and `ClusteringCorrect` quantifies over
the states the schedule built from it.  Written out it is a sum of ceilings, each a log
over some margin, and one for the harvested strings, and nothing in the statement says how
those margins depend on the rates they resolve.  This file says it: every term but the redraws'
`εs` one is bounded by one polynomial in the reciprocals of `sig η` and `shiftScale`, so the
budget is `log` in the failure probability and polynomial in everything else.

`shiftBudget` is `sig² · shiftScale`, and the count rule's term squares it, so `sig` enters at
the fourth power — and `shiftScale` is itself a minimum over `εcov` and `indecisionLimit`, so a
bound in `sig` alone cannot exist.

With that bound `ClusteringGuarantee` follows from `ClusteringCorrect`, at the end.
-/

namespace OrthoDFA

-- `Finset J` carries the instance the definitions were written with; these bounds read
-- only the card, so the linter is right that they do not use it themselves.
set_option linter.unusedFintypeInType false

variable {J : Type*} [Fintype J]
variable {hsz : ℕ}

/-- Every ceiling is under its argument plus one, and under one when the argument is
negative -- which is what lets the terms be bounded separately and summed. -/
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

/-- The eight tails, each named, so the bound can be read term by term. -/
noncomputable def prefTerms (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov εs δ α pAP crossLimit : ℝ) : List ℝ :=
  [ (populations.card : ℝ) * (cleanLevel η populations indecisionLimit εcov δ pAP crossLimit
        + heavyLevel η populations indecisionLimit εcov δ pAP crossLimit) ^ 2
      / (15 / 16 * shiftBudget η indecisionLimit εcov) ^ 2,
    Real.log (128 * (populations.card : ℝ) / δ) * (1056 / shiftBudget η indecisionLimit εcov),
    50 * (gateDevT η + gateDevU η) ^ 2
      * Real.log (32 * (populations.card : ℝ)
        * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 3) / δ)
      / gateRoom η indecisionLimit εcov ^ 2,
    gateLogCount (200 / 9 * (gateDevD η + 1) ^ 2 / gateRoom η indecisionLimit εcov ^ 2)
      (4 * Real.sqrt (Real.sqrt (7 * (populations.card : ℝ) / α))),
    Real.log (128 * (populations.card : ℝ)
        * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ)
      / dirtyRate η indecisionLimit εcov εs,
    64 * Real.log (128 * (populations.card : ℝ)
        * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ)
      / shiftBudget η indecisionLimit εcov,
    65536 * Real.log (128 * (populations.card : ℝ) / δ)
      / (9 * cutBudget η indecisionLimit εcov ^ 2),
    6 * Real.log (256 * (populations.card : ℝ) / δ) / εs ]

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

/-- The count is exactly its tails, each rounded up, the harvested strings' term, and the
one rung the ladder always has. -/
theorem prefCount_eq_sum (populations : Finset J)
    (η indecisionLimit εcov εs δ α pAP crossLimit : ℝ) :
    prefCount η populations indecisionLimit εcov εs δ α pAP crossLimit hsz
      = ((prefTerms η populations indecisionLimit εcov εs δ α pAP crossLimit).map
          (fun x => ⌈x⌉₊)).sum + ⌈128 * (hsz : ℝ) / cutBudget η indecisionLimit εcov⌉₊ + 1 := by
  simp only [prefCount, prefTerms, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  ring

/-- The mechanical half of the bound: the count is under its tails once each rounding is
paid for.  Nothing analytic happens here -- what the tails are worth is `prefCount_le_split`. -/
theorem prefCount_le_terms (populations : Finset J)
    (η indecisionLimit εcov εs δ α pAP crossLimit : ℝ) :
    (prefCount η populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ)
      ≤ ((prefTerms η populations indecisionLimit εcov εs δ α pAP crossLimit).map
          (fun x => max x 0)).sum + 10
        + max (128 * (hsz : ℝ) / cutBudget η indecisionLimit εcov) 0 := by
  have h :=
    cast_sum_ceil_le (prefTerms η populations indecisionLimit εcov εs δ α pAP crossLimit)
  have hh := cast_ceil_le (128 * (hsz : ℝ) / cutBudget η indecisionLimit εcov)
  rw [prefCount_eq_sum]
  have hlen :
      (prefTerms η populations indecisionLimit εcov εs δ α pAP crossLimit).length = 8 := by
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
/-- The pool never holds fewer than 64 suffixes, which is what lets its size stand in for the
constants inside the logs. -/
theorem poolCount_ge (populations : Finset J) (η indecisionLimit εcov δ pAP crossLimit : ℝ)
    (hsig : 0 < sig η) (hη : 0 ≤ η) (hind : 0 < indecisionLimit) (hεcov : 0 < εcov)
    (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) :
    (64 : ℝ) ≤ (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) := by
  have hfam : (32 : ℝ) ≤ famCount η populations indecisionLimit εcov δ crossLimit :=
    famCount_ge η populations hη (by rw [sig] at hsig; linarith only [hsig]) hεcov hind
  have hpool : 2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1) / pAP
      ≤ (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) := by
    rw [poolCount]
    push_cast
    linarith only [Nat.le_ceil (2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ)
        + 1) / pAP),
      (Nat.cast_nonneg _ : (0 : ℝ)
        ≤ (⌈800 * Real.log (128 / δ) / (81 * pAP)⌉₊ : ℝ))]
  have hdiv : 2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1)
      ≤ 2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1) / pAP := by
    rw [le_div_iff₀ hpAP]
    nlinarith only [hpAP1, hfam]
  linarith only [hpool, hdiv, hfam]

set_option maxHeartbeats 4000000 in
/-- The budget is polynomial in the reciprocals of the rates it resolves.

`sig` enters at the fourth power because the count rule's budget is `sig² · shiftScale` and its
term squares it.  The gate's `1/sig²` deviations are paid for by the `a² ≤ 16 sig²` that
`budgetScale` carries, so the terms `|populations|` does not multiply are charged only that.

The pool's size is left inside the log rather than bounded in the rates: the count rule's levels
and the shift tail union-bound over its members, and nothing else reads it.  `|populations|`,
`α` and the scale stay out of the log of the terms `|populations|` multiplies.

`0 ≤ η` is what bounds `sig η` above; without it the noise rate may be negative, `sig` may be
arbitrarily large, and the `sig ^ 4` in the denominator sends the bound below the count.

The redraws' `εs` term is left as it is. -/
theorem prefCount_le_split (populations : Finset J)
    (η indecisionLimit εcov εs δ α pAP crossLimit : ℝ)
    (hsig : 0 < sig η) (hη : 0 ≤ η) (hpop : populations.Nonempty)
    (hind : 0 < indecisionLimit) (hεcov : 0 < εcov) (hεs : 0 < εs)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hα : 0 < α) (hα1 : α < 1)
    (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) :
    (prefCount η populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ)
      ≤ 2 ^ 20 * (populations.card : ℝ)
          * levelLog η populations indecisionLimit εcov δ pAP crossLimit
          / (sig η ^ 4 * shiftScale indecisionLimit εcov ^ 2)
        + 2 ^ 26 * budgetLog populations η indecisionLimit εcov δ α pAP crossLimit
          / budgetScale η indecisionLimit εcov ^ 2
        + 6 * Real.log (256 * (populations.card : ℝ) / δ) / εs
        + 128 * (hsz : ℝ) / cutBudget η indecisionLimit εcov := by
  simp only [levelLog]
  have hM64 := poolCount_ge populations η indecisionLimit εcov δ pAP crossLimit hsig hη hind
    hεcov hpAP hpAP1
  have hc1 : (1 : ℝ) ≤ (populations.card : ℝ) := Nat.one_le_cast.2 (Finset.card_pos.2 hpop)
  -- the count rule's two levels, against the pool
  have hηlt' : η < 1 / 2 := by rw [sig] at hsig; linarith only [hsig]
  have hκ4 : (4 : ℝ) ≤ (famCount η populations indecisionLimit εcov δ crossLimit : ℝ) := by
    exact_mod_cast famCount_ge_four (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
      η populations εcov δ hηlt'
  have hMκ : 2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1) / pAP
      ≤ (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) := by
    rw [poolCount]
    push_cast
    linarith [Nat.le_ceil (2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ)
        + 1) / pAP),
      Nat.cast_nonneg (α := ℝ)
        ⌈800 * Real.log (128 / δ) / (81 * pAP)⌉₊]
  have hcl2 : cleanLevel η populations indecisionLimit εcov δ pAP crossLimit ^ 2
      ≤ max 0 (Real.log (16
        * (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) / δ)) / 2 := by
    set M' : ℝ := (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ)
    set κ' : ℝ := (famCount η populations indecisionLimit εcov δ crossLimit : ℝ)
    set a : ℝ := apFloor pAP (poolCount η populations indecisionLimit εcov δ pAP crossLimit)
      with ha
    have ha' : a = M' * pAP / 2 - 1 := rfl
    have haκ : κ' ≤ a := by
      rw [ha']; rw [div_le_iff₀ hpAP] at hMκ; nlinarith
    have hM'0 : 0 ≤ M' := Nat.cast_nonneg _
    have haM : a ≤ M' := by rw [ha']; nlinarith [hM'0, hpAP1, hpAP]
    have ha0 : 0 < a := by linarith
    set X : ℝ := a / ((a - κ' + 1) * (δ / 16)) with hX
    have hX0 : 0 < X := by
      have : 0 < a - κ' + 1 := by linarith
      positivity
    have hXle : X ≤ 16 * M' / δ := by
      have hden : 0 < (a - κ' + 1) * (δ / 16) := by
        have : 0 < a - κ' + 1 := by linarith
        positivity
      rw [hX, div_le_div_iff₀ hden hδ]
      have h1 : a * δ ≤ M' * δ := mul_le_mul_of_nonneg_right haM hδ.le
      have h2 : 1 ≤ a - κ' + 1 := by linarith
      nlinarith [mul_le_mul_of_nonneg_left h2 (by positivity : (0 : ℝ) ≤ M' * δ)]
    have hlog : Real.log X ≤ Real.log (16 * M' / δ) := Real.log_le_log hX0 hXle
    have heq : cleanLevel η populations indecisionLimit εcov δ pAP crossLimit
        = Real.sqrt (Real.log X / 2) := rfl
    rw [heq]
    by_cases h0 : 0 ≤ Real.log X / 2
    · rw [Real.sq_sqrt h0]
      have := le_max_right 0 (Real.log (16 * M' / δ))
      linarith
    · rw [Real.sqrt_eq_zero'.2 (by linarith)]
      have := le_max_left 0 (Real.log (16 * M' / δ))
      norm_num; positivity
  have hhv2 : heavyLevel η populations indecisionLimit εcov δ pAP crossLimit ^ 2
      ≤ max 4 (5 / 2 * Real.log (120 * (populations.card : ℝ)
        * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ)) := by
    set M' : ℝ := (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ)
    set κ' : ℝ := (famCount η populations indecisionLimit εcov δ crossLimit : ℝ)
    set Y : ℝ := 15 * (M' + 1) / (κ' * (δ / 16)) with hY
    have hcp : (0 : ℝ) < (populations.card : ℝ) := by linarith
    have hM0 : 0 ≤ M' := Nat.cast_nonneg _
    have hY0 : 0 < Y := by rw [hY]; positivity
    have hYle : Y ≤ 120 * (populations.card : ℝ) * (M' + 1) / δ := by
      rw [hY, div_le_div_iff₀ (by positivity) hδ]
      have h1 : 15 * (M' + 1) * δ * 16 ≤ 120 * (M' + 1) * (κ' * δ) := by
        nlinarith [mul_le_mul_of_nonneg_left hκ4 (by positivity : (0 : ℝ) ≤ 120 * (M' + 1) * δ)]
      have h2 : 120 * (M' + 1) * (κ' * δ) ≤ 120 * (populations.card : ℝ) * (M' + 1) * (κ' * δ)
        := by
        have : (0 : ℝ) ≤ 120 * (M' + 1) * (κ' * δ) := by positivity
        nlinarith
      nlinarith
    have hlog : Real.log Y ≤ Real.log (120 * (populations.card : ℝ) * (M' + 1) / δ) :=
      Real.log_le_log hY0 hYle
    have heq : heavyLevel η populations indecisionLimit εcov δ pAP crossLimit
        = max 2 (Real.sqrt (5 / 2 * Real.log Y)) := rfl
    rw [heq]
    rcases le_total 2 (Real.sqrt (5 / 2 * Real.log Y)) with h | h
    · rw [max_eq_right h]
      have h0 : 0 ≤ 5 / 2 * Real.log Y := by
        by_contra hn; push_neg at hn
        rw [Real.sqrt_eq_zero'.2 hn.le] at h; linarith
      rw [Real.sq_sqrt h0]
      exact le_trans (by linarith) (le_max_right _ _)
    · rw [max_eq_left h]; norm_num
  have hhv3 : heavyLevel η populations indecisionLimit εcov δ pAP crossLimit ^ 2
      ≤ max 4 (5 / 2 * Real.log (64
        * ((poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) + 1) / δ)) := by
    set M' : ℝ := (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ)
    set κ' : ℝ := (famCount η populations indecisionLimit εcov δ crossLimit : ℝ)
    set Y : ℝ := 15 * (M' + 1) / (κ' * (δ / 16)) with hY
    have hM0 : 0 ≤ M' := Nat.cast_nonneg _
    have hY0 : 0 < Y := by rw [hY]; positivity
    have hYle : Y ≤ 64 * (M' + 1) / δ := by
      rw [hY, div_le_div_iff₀ (by positivity) hδ]
      nlinarith [mul_le_mul_of_nonneg_left hκ4 (by positivity : (0 : ℝ) ≤ 64 * (M' + 1) * δ)]
    have hlog : Real.log Y ≤ Real.log (64 * (M' + 1) / δ) := Real.log_le_log hY0 hYle
    have heq : heavyLevel η populations indecisionLimit εcov δ pAP crossLimit
        = max 2 (Real.sqrt (5 / 2 * Real.log Y)) := rfl
    rw [heq]
    rcases le_total 2 (Real.sqrt (5 / 2 * Real.log Y)) with h | h
    · rw [max_eq_right h]
      have h0 : 0 ≤ 5 / 2 * Real.log Y := by
        by_contra hn; push_neg at hn
        rw [Real.sqrt_eq_zero'.2 hn.le] at h; linarith
      rw [Real.sq_sqrt h0]
      exact le_trans (by linarith) (le_max_right _ _)
    · rw [max_eq_left h]; norm_num
  set c : ℝ := (populations.card : ℝ) with hcdef
  clear_value c
  have hcpos : (0 : ℝ) < c := by linarith only [hc1]
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
  -- the scales: the log's, with the gate's `a²`, and the counts', without it
  have hBmn : budgetScale η indecisionLimit εcov ≤ min εcov (1 / 2) := by
    have ha := gateA_pos hη hηlt
    have ha1 := gateA_le_one hη hηlt
    have h := mul_le_of_le_one_right (lt_min hεcov (by norm_num : (0 : ℝ) < 1 / 2)).le
      (by nlinarith : gateA η ^ 2 ≤ 1)
    exact (min_le_left _ _).trans h
  have hBB' : budgetScale η indecisionLimit εcov ≤ shiftScale indecisionLimit εcov :=
    le_min hBmn (min_le_right _ _)
  obtain ⟨hgR0, hgR1, hgRlb⟩ := gateRoom_facts η indecisionLimit εcov hεcov
  have hBpos := budgetScale_pos η indecisionLimit εcov hεcov hind hη hηlt
  have hBa : budgetScale η indecisionLimit εcov ≤ min εcov (1 / 2) * gateA η ^ 2 :=
    min_le_left _ _
  have hcutge := cutBudget_ge η indecisionLimit εcov hεcov hind hη hηlt
  have hB'pos := shiftScale_pos hεcov hind
  have hB'm : shiftScale indecisionLimit εcov ≤ min εcov (1 / 2) := min_le_left _ _
  have hB'u : shiftScale indecisionLimit εcov ≤ indecisionLimit := min_le_right _ _
  have hmn2 : min εcov (1 / 2) ≤ 1 / 2 := min_le_right _ _
  have hmnε : min εcov (1 / 2) ≤ εcov := min_le_left _ _
  -- the count rule's budget and the cut budget, against `sig²·B'`
  have hΔeq : shiftBudget η indecisionLimit εcov
      = s ^ 2 * shiftScale indecisionLimit εcov / 256 := by rw [shiftBudget, ← hsdef]
  have ha2 : 4 * s ^ 2 ≤ gateA η ^ 2 := by
    have := two_sig_le_gateA hη hηlt
    rw [← hsdef] at this
    nlinarith
  have hcutB' : 2 / 5 * (s ^ 2 * shiftScale indecisionLimit εcov)
      ≤ cutBudget η indecisionLimit εcov := by
    rw [cutBudget, ← hsdef]
    have hsB : s * shiftScale indecisionLimit εcov ≤ 1 / 4 := by nlinarith
    refine le_min (by nlinarith) (le_min (by nlinarith) (le_min (by nlinarith) ?_))
    nlinarith [mul_le_mul_of_nonneg_right ha2 (le_trans hB'pos.le hB'm)]
  have hrej : rejFrac η indecisionLimit εcov εs ≤ s ^ 2 * min εcov (1 / 2) / 1024 := by
    have := (rejFrac_le (εs := εs) (indecisionLimit := indecisionLimit) η (εcov := εcov)).1
    rwa [← hsdef] at this
  have hrej0 := rejFrac_pos η hηlt hεcov hεs hind
  set B : ℝ := budgetScale η indecisionLimit εcov with hBdef
  clear_value B
  have hBs : B ≤ s := by
    rw [hBdef, hsdef]; exact budgetScale_le_sig η indecisionLimit εcov hη hηlt
  have hB2 : B ≤ 1 / 2 := le_trans hBs hs2
  set B' : ℝ := shiftScale indecisionLimit εcov with hB'def
  clear_value B'
  have hB'2 : B' ≤ 1 / 2 := le_trans hB'm hmn2
  set mn : ℝ := min εcov (1 / 2) with hmndef
  clear_value mn
  -- the gate's room is a fifth of the coverage, so at least a fifth of the scale
  set gR : ℝ := gateRoom η indecisionLimit εcov with hgRdef
  clear_value gR
  have hgRB : B' ≤ 5 * gR := by linarith only [hgRlb, hB'm]
  set cut : ℝ := cutBudget η indecisionLimit εcov with hcutdef
  clear_value cut
  have hcutlb : B ≤ 10 * cut := by linarith only [hcutge]
  have hcutpos : 0 < cut := by
    have : 0 < 2 / 5 * (s ^ 2 * B') := by positivity
    linarith only [this, hcutB']
  set Δ : ℝ := shiftBudget η indecisionLimit εcov with hΔdef
  clear_value Δ
  have hΔpos : 0 < Δ := by rw [hΔeq]; positivity
  set cr : ℝ := rejFrac η indecisionLimit εcov εs with hcrdef
  clear_value cr
  -- the pool, which enters only through the log
  set M : ℝ := (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) with hMdef
  clear_value M
  have hM0 : 0 ≤ M := by linarith only [hM64]
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
  have hT6 : 6 ≤ T := by nlinarith only [hTδ, hTge, hc1, hM64]
  have hTα : 1 ≤ T * α := by
    rw [hTdef, div_mul_eq_mul_div, le_div_iff₀ hdpos, hddef]
    have h1 : 1 ≤ (c + 2) * M := by nlinarith only [hc1, hM64]
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
      have hM2 : 4 * (M + 1) ≤ M ^ 2 := by nlinarith only [hM64]
      nlinarith only [hsq, hδ2, e, mul_le_mul_of_nonneg_right h8 (sq_nonneg M),
        mul_le_mul_of_nonneg_left hM2 (by linarith only [hcpos] : (0 : ℝ) ≤ 32 * c)]
    have h := log_le_of_le_pow (by positivity) 2 harg
    rw [hLdef]; push_cast at h; linarith only [h]
  have hR : Real.log (128 * c / δ) ≤ 1 * L :=
    hlog1 _ (by positivity) (by rw [div_mul_cancel₀ _ hδ.ne']; nlinarith only [hTδ, hc1, hM64])
  -- the gate's logs, against `L`
  have hT1 : (c + 2) * M ≤ T := by
    have hd1 : d ≤ 1 := by
      rw [hddef]
      have : δ * α * pAP ≤ 1 := by
        have hδα : δ * α ≤ 1 := by nlinarith only [le_mul2 hδ.le hα.le hδ1 hα1.le]
        nlinarith only [le_mul2 (mul_pos hδ hα).le hpAP.le hδα hpAP1]
      nlinarith only [le_mul2 (mul_pos (mul_pos hδ hα) hpAP).le hBpos.le this hB2]
    rw [hTdef, le_div_iff₀ hdpos]
    nlinarith only [hcM0, hd1, hdpos]
  have hTB : 30 ≤ T * B := by
    rw [hTdef, div_mul_eq_mul_div, le_div_iff₀ hdpos, hddef]
    have hδαp : δ * α * pAP ≤ 1 := by
      have hδα : δ * α ≤ 1 := by nlinarith only [le_mul2 hδ.le hα.le hδ1 hα1.le]
      nlinarith only [le_mul2 (mul_pos hδ hα).le hpAP.le hδα hpAP1]
    have hcM30 : 30 ≤ (c + 2) * M := by nlinarith only [hc1, hM64]
    nlinarith only [hcM30, mul_le_mul_of_nonneg_right hδαp hBpos.le, hBpos]
  have hY3 : Real.log (128 * c * (M + 3) / δ) ≤ 2 * L := by
    have h8 : 8 * c ≤ (c + 2) ^ 2 := by nlinarith only [sq_nonneg (c - 2)]
    have hsq : (2 * ((c + 2) * M)) ^ 2 ≤ (T * δ) ^ 2 := pow_le_pow_left₀ (by positivity) hTδ 2
    have hδ2 : (T * δ) ^ 2 ≤ T ^ 2 * δ := by
      have : δ ^ 2 ≤ δ := by nlinarith only [hδ, hδ1]
      calc (T * δ) ^ 2 = T ^ 2 * δ ^ 2 := by ring
        _ ≤ T ^ 2 * δ := mul_le_mul_of_nonneg_left this (by positivity)
    have harg : 128 * c * (M + 3) / δ ≤ T ^ 2 := by
      rw [div_le_iff₀ hδ]
      have e : (2 * ((c + 2) * M)) ^ 2 = 4 * (c + 2) ^ 2 * M ^ 2 := by ring
      have hM2 : 4 * (M + 3) ≤ M ^ 2 := by nlinarith only [hM64]
      nlinarith only [hsq, hδ2, e, mul_le_mul_of_nonneg_right h8 (sq_nonneg M),
        mul_le_mul_of_nonneg_left hM2 (by linarith only [hcpos] : (0 : ℝ) ≤ 32 * c)]
    have h := log_le_of_le_pow (by positivity) 2 harg
    rw [hLdef]; push_cast at h; linarith only [h]
  have hY32 : Real.log (32 * c * (M + 3) / δ) ≤ 2 * L := by
    refine le_trans (Real.log_le_log (by positivity) ?_) hY3
    exact div_le_div_of_nonneg_right (by nlinarith only [hcpos, hM0]) hδ.le
  -- the gate's constants, against `1/sig`: its gap is at least `8 sig/7`
  set u : ℝ := 1 / s with hudef
  have hus : u * s = 1 := by rw [hudef]; field_simp
  have hu2 : 2 ≤ u := by rw [hudef, le_div_iff₀ hsig]; linarith only [hs2]
  have hgg := gateGap_ge hη hηlt
  set g : ℝ := gateGap η with hgdef
  have hg : 8 * s / 7 ≤ g := by rw [hsval]; linarith only [hgg]
  have hgpos : 0 < g := by linarith only [hg, hsig]
  have hig : 1 / g ≤ 7 / 8 * u := by
    rw [div_le_iff₀ hgpos, hudef, show 7 / 8 * (1 / s) * g = 7 * g / (8 * s) by ring,
      le_div_iff₀ (by positivity)]
    linarith only [hg]
  have hig2 : 1 / g ^ 2 ≤ 49 / 64 * u ^ 2 := by
    rw [← one_div_pow]
    calc (1 / g) ^ 2 ≤ (7 / 8 * u) ^ 2 := pow_le_pow_left₀ (by positivity) hig 2
      _ = 49 / 64 * u ^ 2 := by ring
  obtain ⟨hT0', hD0', hU0'⟩ := gateDev_pos hη hηlt
  set Wg : ℝ := gateDevT η + gateDevU η with hWgdef
  have hW : Wg ≤ 191 / 50 * u ^ 2 := by
    have eW : Wg = 16 / 5 * (1 / g) + 63 / 20 * (1 / g ^ 2) := by
      rw [hWgdef]; unfold gateDevT gateDevU; rw [← hgdef]; ring
    have huu : 14 / 5 * u ≤ 7 / 5 * u ^ 2 := by nlinarith only [hu2]
    rw [eW]
    have h1 := mul_le_mul_of_nonneg_left hig (by norm_num : (0 : ℝ) ≤ 16 / 5)
    have h2 := mul_le_mul_of_nonneg_left hig2 (by norm_num : (0 : ℝ) ≤ 63 / 20)
    nlinarith only [h1, h2, huu, sq_nonneg u]
  have hW2 : Wg ^ 2 ≤ 73 / 5 * u ^ 4 := by
    have := pow_le_pow_left₀ (by linarith only [hT0', hU0'] : 0 ≤ Wg) hW 2
    nlinarith only [this, pow_nonneg (by linarith only [hu2] : (0 : ℝ) ≤ u) 4]
  set Dg : ℝ := gateDevD η with hDgdef
  have hD1 : Dg + 1 ≤ 58 / 25 * u ^ 2 := by
    have eD : Dg = 27 / 10 * (1 / g ^ 2) := by
      rw [hDgdef]; unfold gateDevD; rw [← hgdef]; ring
    rw [eD]
    nlinarith only [hig2, hu2]
  have hD2 : (Dg + 1) ^ 2 ≤ 27 / 5 * u ^ 4 := by
    have := pow_le_pow_left₀ (by linarith only [hD0'] : 0 ≤ Dg + 1) hD1 2
    nlinarith only [this, pow_nonneg (by linarith only [hu2] : (0 : ℝ) ≤ u) 4]
  have hBgR : B' ^ 2 ≤ 25 * gR ^ 2 := by nlinarith only [hgRB, hB'pos, hgR0]
  have hus4 : u ^ 4 * s ^ 4 = 1 := by rw [← mul_pow, hus]; norm_num
  have huB : u * B ≤ 1 := by nlinarith only [hus, hBs, hu2]
  -- the tails, each against the common quotient
  have hD : (0 : ℝ) < s ^ 4 * B' ^ 2 := by positivity
  have hsB8 : s ^ 2 * B' ≤ 1 / 8 := by nlinarith only [hs2sq, hB'2, hB'pos]
  -- the levels' term: `n ≥ 4K (log(4Kk) + 2 log log(4Kk))`, with `K·B'² ≤ 3000/sig⁴`
  set K : ℝ := 200 / 9 * (Dg + 1) ^ 2 / gR ^ 2 with hKdef
  have hgR2 : 0 < gR ^ 2 := by positivity
  have hKB : K * B' ^ 2 ≤ 3000 * u ^ 4 := by
    have e : K * B' ^ 2 * gR ^ 2 = 200 / 9 * (Dg + 1) ^ 2 * B' ^ 2 := by
      rw [hKdef]; field_simp
    have h1 : 200 / 9 * (Dg + 1) ^ 2 * B' ^ 2 ≤ 3000 * u ^ 4 * gR ^ 2 := by
      nlinarith only [mul_le_mul hD2 hBgR (sq_nonneg B') (by positivity), sq_nonneg (Dg + 1)]
    exact le_of_mul_le_mul_right (by linarith only [e, h1]) hgR2
  have hKB0 : K * B ^ 2 ≤ 3000 * u ^ 4 := by
    have : B ^ 2 ≤ B' ^ 2 := pow_le_pow_left₀ hBpos.le hBB' 2
    nlinarith only [hKB, this, show 0 ≤ K by rw [hKdef]; positivity]
  have hK4 : 4 ≤ K := by
    rw [hKdef, le_div_iff₀ hgR2]
    have : 1 ≤ (Dg + 1) ^ 2 := by nlinarith only [hD0']
    nlinarith only [this, hgR1, hgR0]
  set q : ℝ := 7 * c / α with hqdef
  have hq1 : 1 ≤ q := by rw [hqdef, le_div_iff₀ hα]; linarith only [hc1, hα1]
  have hr1 : 1 ≤ Real.sqrt (Real.sqrt q) := Real.one_le_sqrt.2 (Real.one_le_sqrt.2 hq1)
  set y : ℝ := Real.log (4 * K * (4 * Real.sqrt (Real.sqrt q))) with hydef
  have hy0 : 0 < y := Real.log_pos (by nlinarith only [hK4, hr1])
  have hyL : y ≤ 27 / 4 * L := by
    have hl4 : Real.log (Real.sqrt (Real.sqrt q)) = Real.log q / 4 := by
      rw [Real.log_sqrt (Real.sqrt_nonneg _), Real.log_sqrt (by linarith only [hq1])]; ring
    have e : y = Real.log (16 * K) + Real.log q / 4 := by
      rw [hydef, show 4 * K * (4 * Real.sqrt (Real.sqrt q)) = 16 * K * Real.sqrt (Real.sqrt q)
        by ring, Real.log_mul (by positivity) (by positivity), hl4]
    have hT7 : 7 ≤ T := by nlinarith only [hTB, hB2, hBpos]
    have hcT : c ≤ T := by nlinarith only [hT1, hM64, hcpos]
    have h16 : 16 * K ≤ T ^ 6 := by
      have hB6 : 16 * K * B ^ 6 ≤ 48000 := by
        have h4 : (u * B) ^ 4 ≤ 1 := pow_le_one₀ (by positivity) huB
        have : 16 * K * B ^ 6 = 16 * (K * B ^ 2) * B ^ 4 := by ring
        rw [this]
        nlinarith only [mul_le_mul_of_nonneg_right hKB0 (pow_nonneg hBpos.le 4), h4,
          show (u * B) ^ 4 = u ^ 4 * B ^ 4 by ring]
      have hT6 : (30 : ℝ) ^ 6 ≤ T ^ 6 * B ^ 6 := by
        rw [← mul_pow]; exact pow_le_pow_left₀ (by norm_num) hTB 6
      exact le_of_mul_le_mul_right (by nlinarith only [hB6, hT6]) (pow_pos hBpos 6)
    have hq : q ≤ T ^ 3 := by
      rw [hqdef, div_le_iff₀ hα]
      nlinarith only [mul_le_mul hT7 hcT hcpos.le (by linarith only [hT7]), hTα,
        mul_nonneg (by linarith only [hT7] : (0 : ℝ) ≤ T) (by linarith only [hT7] : (0 : ℝ) ≤ T)]
    have l1 := log_le_of_le_pow (by positivity) 6 h16
    have l2 := log_le_of_le_pow (by positivity) 3 hq
    rw [← hLdef] at l1 l2
    push_cast at l1 l2
    rw [e]; linarith only [l1, l2]
  have hlogy : Real.log y ≤ 2 * L := by
    have h1 : Real.log y ≤ Real.log (27 / 4 * L) := Real.log_le_log hy0 hyL
    rw [Real.log_mul (by norm_num) (by linarith only [hL1])] at h1
    have h2 : Real.log (27 / 4 : ℝ) ≤ 2 := by
      rw [Real.log_le_iff_le_exp (by norm_num), show (2 : ℝ) = 1 + 1 by norm_num, Real.exp_add]
      nlinarith only [Real.exp_one_gt_d9]
    have h3 : Real.log L ≤ L - 1 := Real.log_le_sub_one_of_pos (by linarith only [hL1])
    linarith only [h1, h2, h3, hL1]
  -- the second bound's tails: the count rule's and the dirty tail's against `Q1`, the rest
  -- against `Q2`, whose scale carries the gate's `a² ≤ 16 sig²`
  set Λ : ℝ := Real.log (64 * (M + 1) / δ) with hΛdef
  have hΛ1 : 1 ≤ Λ := by
    rw [hΛdef]
    refine one_le_log_of_six_le ?_
    rw [le_div_iff₀ hδ]; nlinarith only [hM64, hδ1, hδ]
  set Q1 : ℝ := c * Λ / (s ^ 4 * B' ^ 2) with hQ1def
  set Q2 : ℝ := L / B ^ 2 with hQ2def
  have hB2pos : 0 < B ^ 2 := by positivity
  have hQ1eq : Q1 * (s ^ 4 * B' ^ 2) = c * Λ := by rw [hQ1def]; field_simp
  have hQ2eq : Q2 * B ^ 2 = L := by rw [hQ2def]; field_simp
  have hQ1nn : 0 ≤ Q1 := by positivity
  have hQ2nn : 0 ≤ Q2 := by positivity
  have hmn0 : 0 ≤ mn := by linarith only [hB'm, hB'pos]
  have hB16 : B ≤ 16 * s ^ 2 * mn := by
    have ha0 := gateA_pos hη hηlt
    have ha4 : gateA η ≤ 4 * s := by
      have hne : (0 : ℝ) < 1 - η := by linarith only [hηlt]
      rw [gateA, div_le_iff₀ hne, hsval]
      nlinarith only [sq_nonneg (1 - 2 * η)]
    have h := pow_le_pow_left₀ ha0.le ha4 2
    nlinarith only [hBa, h, hmn0]
  have hB8 : B ^ 2 ≤ 8 * s ^ 2 * B' := by
    have h := mul_le_mul hB16 hBB' hBpos.le (by positivity)
    have h2 := mul_le_mul_of_nonneg_left hmn2 (by positivity : (0 : ℝ) ≤ 16 * s ^ 2 * B')
    nlinarith only [h, h2]
  have hBgR2 : B ^ 2 ≤ 6400 * s ^ 4 * gR ^ 2 := by
    have h1 : B ≤ 80 * s ^ 2 * gR := by nlinarith only [hB16, hgRlb, sq_nonneg s]
    have h2 := pow_le_pow_left₀ hBpos.le h1 2
    nlinarith only [h2]
  -- `1/(sig² B') ≤ 8/B²`
  have hinv2 : ∀ k : ℝ, 0 ≤ k → k * L / (s ^ 2 * B') ≤ 8 * k * Q2 := by
    intro k hk
    have hsB0 : 0 < s ^ 2 * B' := by positivity
    rw [div_le_iff₀ hsB0, hQ2def]
    have e : 8 * k * (L / B ^ 2) * (s ^ 2 * B') = k * L * (8 * s ^ 2 * B') / B ^ 2 := by
      field_simp
    rw [e, le_div_iff₀ hB2pos]
    exact mul_le_mul_of_nonneg_left hB8 (mul_nonneg hk hL0)
  have u1 : c * (cleanLevel η populations indecisionLimit εcov δ pAP crossLimit
        + heavyLevel η populations indecisionLimit εcov δ pAP crossLimit) ^ 2
      / (15 / 16 * Δ) ^ 2 ≤ 1044000 * Q1 := by
    have hcl : cleanLevel η populations indecisionLimit εcov δ pAP crossLimit ^ 2 ≤ Λ / 2 := by
      refine le_trans hcl2 ?_
      have hl : Real.log (16 * M / δ) ≤ Λ := by
        rcases eq_or_lt_of_le hM0 with h0 | h0
        · rw [← h0]; simp only [mul_zero, zero_div, Real.log_zero]; linarith only [hΛ1]
        · exact Real.log_le_log (by positivity)
            (div_le_div_of_nonneg_right (by linarith only [hM0]) hδ.le)
      have : max 0 (Real.log (16 * M / δ)) ≤ Λ := max_le (by linarith only [hΛ1]) hl
      linarith only [this]
    have hhv : heavyLevel η populations indecisionLimit εcov δ pAP crossLimit ^ 2
        ≤ 13 / 2 * Λ :=
      le_trans hhv3 (max_le (by linarith only [hΛ1]) (by linarith only [hΛ1]))
    have hux : (cleanLevel η populations indecisionLimit εcov δ pAP crossLimit
        + heavyLevel η populations indecisionLimit εcov δ pAP crossLimit) ^ 2 ≤ 14 * Λ := by
      nlinarith only [hcl, hhv, sq_nonneg (cleanLevel η populations indecisionLimit εcov δ pAP
        crossLimit - heavyLevel η populations indecisionLimit εcov δ pAP crossLimit)]
    rw [div_le_iff₀ (by positivity)]
    have hstep : c * (cleanLevel η populations indecisionLimit εcov δ pAP crossLimit
        + heavyLevel η populations indecisionLimit εcov δ pAP crossLimit) ^ 2
        ≤ c * (14 * Λ) := mul_le_mul_of_nonneg_left hux hcpos.le
    have e : 1044000 * Q1 * (15 / 16 * Δ) ^ 2
        = 1044000 * (15 / 16) ^ 2 / 256 ^ 2 * (Q1 * (s ^ 4 * B' ^ 2)) := by
      rw [hΔeq]; ring
    rw [e, hQ1eq]
    nlinarith only [hstep, mul_nonneg hcpos.le (by linarith only [hΛ1] : (0 : ℝ) ≤ Λ)]
  have u5 : Real.log (128 * c * (M + 1) / δ) / dirtyRate η indecisionLimit εcov εs
      ≤ 480 * Q1 := by
    have hdr : dirtyRate η indecisionLimit εcov εs = Δ ^ 2 / (4 * (Δ + cr)) := by
      rw [dirtyRate, ← hΔdef, ← hcrdef]
    rw [hdr, div_div_eq_mul_div]
    have hcr : cr ≤ s ^ 2 * mn / 1024 := hrej
    have hlog3 : Real.log (128 * c * (M + 1) / δ) ≤ 3 * (c * Λ) := by
      have e : 128 * c * (M + 1) / δ = 2 * c * (64 * (M + 1) / δ) := by ring
      rw [e, Real.log_mul (by positivity) (by positivity), ← hΛdef]
      have h1 : Real.log (2 * c) ≤ 2 * c - 1 := Real.log_le_sub_one_of_pos (by positivity)
      nlinarith only [h1, hΛ1, hc1]
    have hY0 : 0 ≤ Real.log (128 * c * (M + 1) / δ) :=
      Real.log_nonneg (by rw [le_div_iff₀ hδ]; nlinarith only [hc1, hδ1, hM64])
    have hk : 4 * (Δ + cr) * (s ^ 4 * B' ^ 2) ≤ 160 * Δ ^ 2 := by
      rw [hΔeq]
      have h1 : 4 * (s ^ 2 * B' / 256) * (s ^ 4 * B' ^ 2) ≤ 128 * (s ^ 2 * B' / 256) ^ 2 := by
        have : 0 ≤ s ^ 2 * B' := by positivity
        nlinarith only [hsB8, this, sq_nonneg (s ^ 2 * B')]
      have h2 : 4 * cr * (s ^ 4 * B' ^ 2) ≤ 32 * (s ^ 2 * B' / 256) ^ 2 := by
        have hcr' : cr * (s ^ 4 * B' ^ 2) ≤ s ^ 2 * mn / 1024 * (s ^ 4 * B' ^ 2) :=
          mul_le_mul_of_nonneg_right hcr (by positivity)
        have hm : s ^ 2 * mn / 1024 * (s ^ 4 * B' ^ 2) ≤ 8 * (s ^ 2 * B' / 256) ^ 2 := by
          have h3 : s ^ 2 * mn ≤ 1 / 8 := by nlinarith only [hs2sq, hmn2, hmn0]
          have : 0 ≤ s ^ 4 * B' ^ 2 := by positivity
          nlinarith only [mul_le_mul_of_nonneg_right h3 this]
        nlinarith only [hcr', hm]
      nlinarith only [h1, h2]
    rw [div_le_iff₀ (by positivity), ← mul_le_mul_iff_of_pos_right hD]
    have hk' : Real.log (128 * c * (M + 1) / δ) * (4 * (Δ + cr)) * (s ^ 4 * B' ^ 2)
        ≤ 3 * (c * Λ) * (160 * Δ ^ 2) := by
      rw [mul_assoc]
      exact mul_le_mul hlog3 hk (by positivity) (by positivity)
    have e : 480 * Q1 * Δ ^ 2 * (s ^ 4 * B' ^ 2) = 480 * (Q1 * (s ^ 4 * B' ^ 2)) * Δ ^ 2 := by
      ring
    rw [e, hQ1eq]
    linarith only [hk']
  have u2 : Real.log (128 * c / δ) * (1056 / Δ) ≤ 2170000 * Q2 := by
    have e : Real.log (128 * c / δ) * (1056 / Δ)
        = 270336 * Real.log (128 * c / δ) / (s ^ 2 * B') := by
      rw [hΔeq]; field_simp; ring
    rw [e]
    refine le_trans (div_le_div_of_nonneg_right (by nlinarith only [hR]) (by positivity))
      (le_trans (hinv2 270336 (by norm_num)) (by nlinarith only [hQ2nn]))
  have u3 : 50 * Wg ^ 2 * Real.log (32 * c * (M + 3) / δ) / gR ^ 2 ≤ 9350000 * Q2 := by
    have hL₁0 : 0 ≤ Real.log (32 * c * (M + 3) / δ) :=
      Real.log_nonneg (by rw [le_div_iff₀ hδ]; nlinarith only [hc1, hδ1, hM64])
    have h := tail_le (w := 1460 * u ^ 4) (N := 1) (K := 9350000) (D := B ^ 2) (L := L)
      (V := 50 * Wg ^ 2 * Real.log (32 * c * (M + 3) / δ)) (G := gR ^ 2)
      (by nlinarith only [mul_le_mul hW2 hY32 hL₁0 (by positivity), sq_nonneg Wg]) hL0
      (by positivity) hB2pos (by
        have e : 1460 * u ^ 4 * B ^ 2 ≤ 1460 * u ^ 4 * (6400 * s ^ 4 * gR ^ 2) :=
          mul_le_mul_of_nonneg_left hBgR2 (by positivity)
        have e2 : 1460 * u ^ 4 * (6400 * s ^ 4 * gR ^ 2) = 9344000 * gR ^ 2 := by
          rw [show 1460 * u ^ 4 * (6400 * s ^ 4 * gR ^ 2) = 9344000 * gR ^ 2 * (u ^ 4 * s ^ 4) by
            ring, hus4, mul_one]
        nlinarith only [e, e2, sq_nonneg gR])
    rw [one_mul] at h
    exact h
  have u4 : gateLogCount K (4 * Real.sqrt (Real.sqrt q)) ≤ 33030000 * Q2 := by
    unfold gateLogCount
    rw [← hydef]
    have h1 : 4 * K * (y + 2 * Real.log y) ≤ 43 * K * L := by
      nlinarith only [hyL, hlogy, hK4]
    have hKB' : K * B ^ 2 ≤ 768000 := by
      have e : K * B ^ 2 * gR ^ 2 = 200 / 9 * (Dg + 1) ^ 2 * B ^ 2 := by
        rw [hKdef]; field_simp
      have h2 : 200 / 9 * (Dg + 1) ^ 2 * B ^ 2
          ≤ 200 / 9 * (27 / 5 * u ^ 4) * (6400 * s ^ 4 * gR ^ 2) :=
        mul_le_mul (mul_le_mul_of_nonneg_left hD2 (by norm_num)) hBgR2 (sq_nonneg B)
          (by positivity)
      have e2 : 200 / 9 * (27 / 5 * u ^ 4) * (6400 * s ^ 4 * gR ^ 2) = 768000 * gR ^ 2 := by
        rw [show 200 / 9 * (27 / 5 * u ^ 4) * (6400 * s ^ 4 * gR ^ 2)
          = 768000 * gR ^ 2 * (u ^ 4 * s ^ 4) by ring, hus4, mul_one]
      exact le_of_mul_le_mul_right (by linarith only [e, h2, e2]) hgR2
    refine le_trans h1 ?_
    rw [hQ2def, show 33030000 * (L / B ^ 2) = 33030000 * L / B ^ 2 by ring, le_div_iff₀ hB2pos]
    nlinarith only [mul_le_mul_of_nonneg_left hKB' (by linarith only [hL0] : (0 : ℝ) ≤ 43 * L),
      hL0]
  have u6 : 64 * Real.log (128 * c * (M + 1) / δ) / Δ ≤ 262200 * Q2 := by
    have e : 64 * Real.log (128 * c * (M + 1) / δ) / Δ
        = 16384 * Real.log (128 * c * (M + 1) / δ) / (s ^ 2 * B') := by
      rw [hΔeq]; field_simp; ring
    rw [e]
    refine le_trans (div_le_div_of_nonneg_right (by nlinarith only [hY]) (by positivity))
      (le_trans (hinv2 32768 (by norm_num)) (by nlinarith only [hQ2nn]))
  have u8 : 65536 * Real.log (128 * c / δ) / (9 * cut ^ 2) ≤ 728200 * Q2 := by
    have h := tail_le (V := 65536 * Real.log (128 * c / δ)) (L := L) (G := 9 * cut ^ 2)
      (w := 65536) (N := 1) (K := 728200) (D := B ^ 2) (by linarith only [hR]) hL0
      (by positivity) hB2pos (by
        have hc2 : B ^ 2 ≤ 100 * cut ^ 2 := by
          have := mul_self_le_mul_self hBpos.le hcutlb
          nlinarith only [this]
        nlinarith only [hc2, sq_nonneg cut])
    rw [one_mul] at h
    exact h
  have u7 : (10 : ℝ) ≤ 3 * Q2 := by
    have hBsq : B ^ 2 ≤ 1 / 4 := by
      have h := pow_le_pow_left₀ hBpos.le hB2 2; norm_num at h; linarith only [h]
    have : 4 * B ^ 2 ≤ Q2 * B ^ 2 := by rw [hQ2eq]; nlinarith only [hL1, hBsq]
    have := le_of_mul_le_mul_right this hB2pos
    linarith only [this]
  -- and the sum
  have hbound := prefCount_le_terms (hsz := hsz) populations η indecisionLimit εcov εs δ α pAP
    crossLimit
  simp only [prefTerms, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    ← hcdef, ← hcutdef, ← hMdef, ← hgRdef, ← hWgdef, ← hDgdef, ← hKdef,
    ← hqdef, ← hΔdef] at hbound
  have hh : max (128 * (hsz : ℝ) / cut) 0 = 128 * (hsz : ℝ) / cut :=
    max_eq_left (by positivity)
  have h256 : 0 ≤ Real.log (256 * c / δ) :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ]; nlinarith only [hc1, hδ1])
  have m9 : max (6 * Real.log (256 * c / δ) / εs) 0 = 6 * Real.log (256 * c / δ) / εs :=
    max_eq_left (div_nonneg (by linarith only [h256]) hεs.le)
  rw [hh, m9] at hbound
  rw [hLeq, show 2 ^ 20 * c * Λ / (s ^ 4 * B' ^ 2) = 2 ^ 20 * Q1 by rw [hQ1def]; ring,
    show 2 ^ 26 * L / B ^ 2 = 2 ^ 26 * Q2 by rw [hQ2def]; ring]
  have n1 := max_le u1 (by positivity : (0 : ℝ) ≤ 1044000 * Q1)
  have n2 := max_le u2 (by positivity : (0 : ℝ) ≤ 2170000 * Q2)
  have n3 := max_le u3 (by positivity : (0 : ℝ) ≤ 9350000 * Q2)
  have n4 := max_le u4 (by positivity : (0 : ℝ) ≤ 33030000 * Q2)
  have n5 := max_le u5 (by positivity : (0 : ℝ) ≤ 480 * Q1)
  have n6 := max_le u6 (by positivity : (0 : ℝ) ≤ 262200 * Q2)
  have n8 := max_le u8 (by positivity : (0 : ℝ) ≤ 728200 * Q2)
  linarith only [hbound, n1, n2, n3, n4, n5, n6, n8, u7, hQ2nn, hQ1nn]

set_option maxHeartbeats 400000 in -- many `nlinarith` steps over a large context
/-- At `εcov ≤ 512·εs` and `indecisionLimit ≤ εcov/4`, the scale the count rule reads is
`indecisionLimit` and the redraws' term fits under the second group's. -/
theorem prefCount_le_split_of_le (populations : Finset J)
    (η indecisionLimit εcov εs δ α pAP crossLimit : ℝ)
    (hsig : 0 < sig η) (hη : 0 ≤ η) (hpop : populations.Nonempty)
    (hind : 0 < indecisionLimit) (hind4 : indecisionLimit ≤ εcov / 4)
    (hind1 : indecisionLimit ≤ 1 / 2) (hεs : εcov ≤ 512 * εs)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hα : 0 < α) (hα1 : α < 1)
    (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) :
    (prefCount η populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ)
      ≤ 2 ^ 20 * (populations.card : ℝ)
          * levelLog η populations indecisionLimit εcov δ pAP crossLimit
          / (sig η ^ 4 * indecisionLimit ^ 2)
        + 2 ^ 27 * budgetLog populations η indecisionLimit εcov δ α pAP crossLimit
          / budgetScale η indecisionLimit εcov ^ 2
        + 128 * (hsz : ℝ) / cutBudget η indecisionLimit εcov := by
  have hεcov : 0 < εcov := by linarith
  have hεs0 : 0 < εs := by linarith
  have h := prefCount_le_split (hsz := hsz) populations η indecisionLimit εcov εs δ α pAP
    crossLimit hsig hη hpop hind hεcov hεs0 hδ hδ1 hα hα1 hpAP hpAP1
  have hηlt : η < 1 / 2 := by rw [sig] at hsig; linarith
  have hsh : shiftScale indecisionLimit εcov = indecisionLimit :=
    min_eq_right (le_min (by linarith) hind1)
  rw [hsh] at h
  have hc1 : (1 : ℝ) ≤ (populations.card : ℝ) := Nat.one_le_cast.2 (Finset.card_pos.2 hpop)
  have hM64 := poolCount_ge populations η indecisionLimit εcov δ pAP crossLimit hsig hη hind
    hεcov hpAP hpAP1
  have hB0 := budgetScale_pos η indecisionLimit εcov hεcov hind hη hηlt
  have hBi : budgetScale η indecisionLimit εcov ≤ indecisionLimit := min_le_right _ _
  set c : ℝ := (populations.card : ℝ) with hcdef
  set B : ℝ := budgetScale η indecisionLimit εcov with hBdef
  set M : ℝ := (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) with hMdef
  set L : ℝ := budgetLog populations η indecisionLimit εcov δ α pAP crossLimit with hLdef
  have hB2 : B ≤ 1 / 2 := by linarith
  have hapB : α * pAP * B ≤ 1 / 2 := by
    have : α * pAP ≤ 1 := by nlinarith
    nlinarith
  have hl0 : 0 ≤ Real.log (256 * c / δ) :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ]; nlinarith)
  have hlog : Real.log (256 * c / δ) ≤ 2 * L := by
    have hd0 : 0 < δ * α * pAP * B := by positivity
    have hd : δ * α * pAP * B ≤ δ / 2 := by nlinarith
    set T : ℝ := (c + 2) * M / (δ * α * pAP * B) with hT
    have hLT : L = Real.log T := by rw [hLdef, budgetLog, ← hcdef, ← hMdef, ← hBdef]
    have hT2 : 2 * (c + 2) * M / δ ≤ T := by
      rw [hT, div_le_div_iff₀ hδ hd0]
      nlinarith [mul_le_mul_of_nonneg_left hd (by positivity : (0 : ℝ) ≤ 2 * (c + 2) * M)]
    have h1 : 256 * c / δ ≤ 2 * (2 * (c + 2) * M / δ) := by
      rw [← mul_div_assoc, div_le_div_iff_of_pos_right hδ]; nlinarith
    have h2 : 2 ≤ 2 * (c + 2) * M / δ := by rw [le_div_iff₀ hδ]; nlinarith
    have hX0 : 0 ≤ 2 * (c + 2) * M / δ := by linarith
    have h3 : 256 * c / δ ≤ T ^ 2 :=
      calc 256 * c / δ ≤ 2 * (2 * (c + 2) * M / δ) := h1
        _ ≤ (2 * (c + 2) * M / δ) * (2 * (c + 2) * M / δ) := mul_le_mul_of_nonneg_right h2 hX0
        _ ≤ T * T := mul_le_mul hT2 hT2 hX0 (hX0.trans hT2)
        _ = T ^ 2 := by ring
    calc Real.log (256 * c / δ) ≤ Real.log (T ^ 2) := Real.log_le_log (by positivity) h3
      _ = 2 * L := by rw [Real.log_pow, hLT]; push_cast; ring
  have hL0 : 0 ≤ L := by linarith
  clear_value c B M L
  have key : 6 * Real.log (256 * c / δ) / εs ≤ 2 ^ 26 * L / B ^ 2 := by
    have hBsq : B ^ 2 ≤ εcov / 8 := by nlinarith
    rw [div_le_div_iff₀ hεs0 (by positivity)]
    have s1 : 6 * Real.log (256 * c / δ) * B ^ 2 ≤ 12 * L * (εcov / 8) :=
      mul_le_mul (by linarith) hBsq (sq_nonneg B) (by linarith)
    nlinarith [mul_nonneg hL0 hεs0.le]
  have e : 2 ^ 27 * L / B ^ 2 = 2 ^ 26 * L / B ^ 2 + 2 ^ 26 * L / B ^ 2 := by ring
  linarith

/-- `prefCount_le_split`'s two log terms are under `budgetCap` at `2²⁶`. -/
theorem budgetCap_split (populations : Finset J) (η indecisionLimit εcov δ α pAP crossLimit : ℝ)
    (hind : 0 < indecisionLimit) (hεcov : 0 < εcov) (hδ : 0 < δ) (hδ1 : δ ≤ 1) :
    2 ^ 20 * (populations.card : ℝ)
          * levelLog η populations indecisionLimit εcov δ pAP crossLimit
          / (sig η ^ 4 * shiftScale indecisionLimit εcov ^ 2)
        + 2 ^ 26 * budgetLog populations η indecisionLimit εcov δ α pAP crossLimit
          / budgetScale η indecisionLimit εcov ^ 2
      ≤ budgetCap populations η indecisionLimit εcov δ α pAP crossLimit (2 ^ 26) := by
  have hB' := shiftScale_pos hεcov hind
  have hΛ : 0 ≤ levelLog η populations indecisionLimit εcov δ pAP crossLimit := by
    have : (0 : ℝ) ≤ (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ) :=
      Nat.cast_nonneg _
    exact Real.log_nonneg (by rw [le_div_iff₀ hδ]; linarith)
  set X : ℝ := (populations.card : ℝ)
    * levelLog η populations indecisionLimit εcov δ pAP crossLimit
    / (sig η ^ 4 * shiftScale indecisionLimit εcov ^ 2) with hX
  have hX0 : 0 ≤ X := div_nonneg (mul_nonneg (Nat.cast_nonneg _) hΛ) (by positivity)
  have e1 : 2 ^ 20 * (populations.card : ℝ)
      * levelLog η populations indecisionLimit εcov δ pAP crossLimit
      / (sig η ^ 4 * shiftScale indecisionLimit εcov ^ 2) = 2 ^ 20 * X := by rw [hX]; ring
  have e2 : budgetCap populations η indecisionLimit εcov δ α pAP crossLimit (2 ^ 26)
      = 2 ^ 26 * X + 2 ^ 26 * budgetLog populations η indecisionLimit εcov δ α pAP crossLimit
        / budgetScale η indecisionLimit εcov ^ 2 := by
    rw [budgetCap, hX]; ring
  rw [e1, e2]
  linarith

/-- `validCount` is its one tail, rounded up, and the ladder's rung. -/
theorem validCount_le (populations : Finset J) (η indecisionLimit εcov δ pAP crossLimit : ℝ)
    (hind : 0 < indecisionLimit) (hpop : populations.Nonempty) (hδ : 0 < δ) (hδ1 : δ ≤ 1) :
    (validCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ)
      ≤ 27 * Real.log (128 * (populations.card : ℝ) / δ) / (2 * indecisionLimit) + 2 := by
  have hc1 : (1 : ℝ) ≤ (populations.card : ℝ) := Nat.one_le_cast.2 (Finset.card_pos.2 hpop)
  have hlog : 0 ≤ Real.log (128 * (populations.card : ℝ) / δ) :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ]; nlinarith)
  have e : Real.log (128 * (populations.card : ℝ) / δ) / undecRate indecisionLimit
      = 27 * Real.log (128 * (populations.card : ℝ) / δ) / (2 * indecisionLimit) := by
    rw [undecRate]; field_simp
  rw [validCount]
  push_cast
  have := ceil_le_of_le (le_of_eq e) (by positivity)
  linarith

/-- Twice `validCount` is under `rejCountMin_le`'s first arm: its rate is linear in
`indecisionLimit`, which the scale sits under. -/
theorem validCount_le_rej (populations : Finset J) (η indecisionLimit εcov δ pAP crossLimit : ℝ)
    (hη : 0 ≤ η) (hηlt : η < 1 / 2) (hεcov : 0 < εcov) (hind : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2) (hpop : populations.Nonempty) (hδ : 0 < δ) (hδ1 : δ ≤ 1) :
    2 * (validCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ)
      ≤ 2 ^ 13 * Real.log (32 * (populations.card : ℝ) / δ)
        / (sig η * min (εcov * sig η) indecisionLimit) := by
  have hv := validCount_le populations η indecisionLimit εcov δ pAP crossLimit hind hpop hδ hδ1
  have hc1 : (1 : ℝ) ≤ (populations.card : ℝ) := Nat.one_le_cast.2 (Finset.card_pos.2 hpop)
  set L32 : ℝ := Real.log (32 * (populations.card : ℝ) / δ) with hL32
  have hL3 : 3 ≤ L32 := by
    rw [hL32, Real.le_log_iff_exp_le (by positivity)]
    have h3 : Real.exp 3 = Real.exp 1 ^ 3 := by rw [← Real.exp_nat_mul]; norm_num
    rw [h3, le_div_iff₀ hδ]
    have : Real.exp 1 ^ 3 ≤ 21 := by
      have h0 : 0 < Real.exp 1 := Real.exp_pos 1
      have : Real.exp 1 ^ 3 ≤ (2.7182818286 : ℝ) ^ 3 :=
        pow_le_pow_left₀ h0.le Real.exp_one_lt_d9.le 3
      nlinarith only [this]
    nlinarith only [this, hc1, hδ1, mul_le_mul_of_nonneg_right this hδ.le]
  have h128 : Real.log (128 * (populations.card : ℝ) / δ) ≤ 2 * L32 := by
    have e : 128 * (populations.card : ℝ) / δ = 4 * (32 * (populations.card : ℝ) / δ) := by
      ring
    rw [e, Real.log_mul (by norm_num) (by positivity), ← hL32]
    have h4 : Real.log 4 ≤ 4 - 1 := Real.log_le_sub_one_of_pos (by norm_num)
    linarith only [h4, hL3]
  have hs := sig_pos η hηlt
  have hB0 : 0 < min (εcov * sig η) indecisionLimit := lt_min (mul_pos hεcov hs) hind
  have hRs : sig η * min (εcov * sig η) indecisionLimit ≤ indecisionLimit / 2 := by
    have h1 : min (εcov * sig η) indecisionLimit ≤ indecisionLimit := min_le_right _ _
    have h2 : sig η ≤ 1 / 2 := by rw [sig]; linarith only [hη]
    have := mul_le_mul h2 h1 hB0.le (by norm_num)
    linarith only [this]
  have hA : 2 ^ 14 * L32 / indecisionLimit
      ≤ 2 ^ 13 * L32 / (sig η * min (εcov * sig η) indecisionLimit) := by
    rw [div_le_div_iff₀ hind (by positivity)]
    nlinarith only [mul_le_mul_of_nonneg_left hRs (by linarith only [hL3] : (0 : ℝ) ≤ L32)]
  have hB : 27 * Real.log (128 * (populations.card : ℝ) / δ) / (2 * indecisionLimit)
      ≤ 27 * L32 / indecisionLimit := by
    rw [div_le_div_iff₀ (by positivity) hind]
    nlinarith only [mul_le_mul_of_nonneg_right h128 hind.le]
  have hC : 4 ≤ L32 / indecisionLimit := by
    rw [le_div_iff₀ hind]; nlinarith only [hL3, hind1]
  have e : 2 ^ 14 * L32 / indecisionLimit = 2 ^ 14 * (L32 / indecisionLimit) := by ring
  have e' : 27 * L32 / indecisionLimit = 27 * (L32 / indecisionLimit) := by ring
  linarith only [hv, hA, hB, hC, e, e']

omit [Fintype J] in
/-- The family, seed included.  The cut budget is only `≥ B/10`, which `20/B` pays for. -/
theorem famCount_succ_le (populations : Finset J) (η indecisionLimit εcov δ crossLimit : ℝ)
    (hsig : 0 < sig η) (hη : 0 ≤ η) (hind : 0 < indecisionLimit) (hεcov : 0 < εcov)
    (hζ : 0 < crossLimit) (hζ1 : crossLimit ≤ 1) :
    (famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1
      ≤ 25 * max (4 * Real.log (20 / budgetScale η indecisionLimit εcov))
          (Real.log (1 / crossLimit)) / (8 * sig η ^ 2) + 5 / sig η + 7 := by
  have hfcval := famHoeff_eq (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η populations εcov δ
  have hle : (famCount η populations indecisionLimit εcov δ crossLimit : ℝ)
      ≤ famHoeff η populations indecisionLimit εcov δ crossLimit := by
    exact_mod_cast famCount_le_famHoeff η populations εcov δ
  set s : ℝ := sig η with hsdef
  have hs2 : s ≤ 1 / 2 := by rw [hsdef, sig]; linarith only [hη]
  have hηlt : η < 1 / 2 := by rw [hsdef, sig] at hsig; linarith only [hsig]
  have hcutlb' := cutBudget_ge η indecisionLimit εcov hεcov hind hη hηlt
  have hBpos := budgetScale_pos η indecisionLimit εcov hεcov hind hη hηlt
  have hBil : budgetScale η indecisionLimit εcov ≤ indecisionLimit := min_le_right _ _
  set B : ℝ := budgetScale η indecisionLimit εcov with hBdef
  have hBs : B ≤ s := by
    rw [hBdef, hsdef]; exact budgetScale_le_sig η indecisionLimit εcov hη hηlt
  have hB2 : B ≤ 1 / 2 := le_trans hBs hs2
  set cut : ℝ := cutBudget η indecisionLimit εcov with hcutdef
  have hcutlb : B ≤ 10 * cut := by linarith only [hcutlb']
  have hcutpos : 0 < cut := by linarith only [hcutlb, hBpos]
  set G : ℝ := Real.log (20 / B) with hGdef
  have hG0 : 0 ≤ G :=
    Real.log_nonneg (by rw [le_div_iff₀ hBpos]; linarith only [hB2])
  have hcutG : Real.log (2 / cut) ≤ G :=
    Real.log_le_log (div_pos (by norm_num) hcutpos)
      (by rw [div_le_div_iff₀ hcutpos hBpos]; linarith only [hcutlb])
  set Z : ℝ := Real.log (1 / crossLimit) with hZdef
  have hZ0 : 0 ≤ Z := Real.log_nonneg (by rw [le_div_iff₀ hζ]; linarith only [hζ1])
  have hs2pos : (0 : ℝ) < s ^ 2 := pow_pos hsig 2
  set M : ℝ := max (4 * G) Z with hMdef
  have hM0 : 0 ≤ M := le_max_of_le_right hZ0
  have hMs : 0 ≤ 25 * M / (16 * s ^ 2) := by positivity
  have hvote : voteSlack η ^ 2 = 16 * s ^ 2 / 25 := by
    simp only [voteSlack, ← hsdef]; ring
  have hzA : Real.log (2 / cut) / voteSlack η ^ 2 ≤ 25 * M / (16 * s ^ 2) := by
    calc Real.log (2 / cut) / voteSlack η ^ 2 ≤ G / voteSlack η ^ 2 :=
          div_le_div_of_nonneg_right hcutG (by positivity)
      _ = 25 * G / (16 * s ^ 2) := by rw [hvote]; field_simp
      _ ≤ 25 * M / (16 * s ^ 2) := by
          gcongr; exact le_trans (by linarith only [hG0]) (le_max_left _ _)
  have hG2 : Real.log 2 ≤ G :=
    Real.log_le_log (by norm_num) (by rw [le_div_iff₀ hBpos]; linarith only [hB2])
  have hZ4 : Z + 4 ≤ 4 * M := by
    have h1 : 4 * G ≤ M := le_max_left _ _
    have h2 : Z ≤ M := le_max_right _ _
    linarith only [h1, h2, hG2, half_le_log_two]
  have hzC : (Z + 4) / (4 * voteSlack η ^ 2) ≤ 25 * M / (16 * s ^ 2) := by
    calc (Z + 4) / (4 * voteSlack η ^ 2) = 25 * (Z + 4) / (64 * s ^ 2) := by
          rw [hvote]; field_simp; ring
      _ ≤ 25 * (4 * M) / (64 * s ^ 2) := by gcongr
      _ = 25 * M / (16 * s ^ 2) := by field_simp; ring
  have hzS : 1 / voteSlack η ≤ 5 / (2 * s) := by
    simp only [voteSlack, ← hsdef]
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    nlinarith only [hsig]
  have hcA := ceil_le_of_le hzA hMs
  have hcC := ceil_le_of_le hzC hMs
  have hcS := ceil_le_of_le hzS (by positivity)
  have hmax := max_le hcA hcC
  have e1 : 25 * M / (8 * s ^ 2) = 2 * (25 * M / (16 * s ^ 2)) := by ring
  have e2 : 5 / s = 2 * (5 / (2 * s)) := by ring
  rw [hfcval] at hle
  linarith only [hle, hmax, hcS, e1, e2]

omit [Fintype J] in
/-- The suffix pool: the family and the findability tail, each over `pAP`. -/
theorem poolCount_le (populations : Finset J) (η indecisionLimit εcov δ pAP crossLimit : ℝ)
    (hsig : 0 < sig η) (hη : 0 ≤ η) (hind : 0 < indecisionLimit) (hεcov : 0 < εcov)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) (hζ : 0 < crossLimit) (hζ1 : crossLimit ≤ 1) :
    (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ)
      ≤ 2 * (25 * max (4 * Real.log (20 / budgetScale η indecisionLimit εcov))
            (Real.log (1 / crossLimit)) / (8 * sig η ^ 2) + 5 / sig η + 7) / pAP
        + 60 * Real.log (3 / δ) / pAP := by
  have hf := famCount_succ_le populations η indecisionLimit εcov δ crossLimit hsig hη hind hεcov
    hζ hζ1
  set P : ℝ := 25 * max (4 * Real.log (20 / budgetScale η indecisionLimit εcov))
    (Real.log (1 / crossLimit)) / (8 * sig η ^ 2) + 5 / sig η + 7 with hPdef
  have hPnn : 0 ≤ P := le_trans (by positivity) hf
  have hA : 2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1) / pAP
      ≤ 2 * P / pAP := div_le_div_of_nonneg_right (by linarith only [hf]) hpAP.le
  -- `H = log (3/δ)` is at least `log 3`
  set H : ℝ := Real.log (3 / δ) with hHdef
  have hH3 : Real.log 3 ≤ H :=
    Real.log_le_log (by norm_num) (by rw [le_div_iff₀ hδ]; linarith only [hδ1])
  set R : ℝ := H / pAP with hRdef
  -- the findability tail, read at `9pAP/20`: `log 128 = 7 log 2`, and `H ≥ log 3 ≥ (3/2) log 2`
  have hl2' := Real.log_two_gt_d9
  have hlog3 : 3 / 2 * Real.log 2 ≤ Real.log 3 := by
    have h : Real.log ((2 : ℝ) ^ 3) ≤ Real.log ((3 : ℝ) ^ 2) :=
      Real.log_le_log (by norm_num) (by norm_num)
    rw [Real.log_pow, Real.log_pow] at h
    push_cast at h
    linarith only [h]
  have hlog128' : Real.log (128 / δ) ≤ 7 * Real.log 2 + H := by
    have h1 : Real.log (128 / δ) ≤ Real.log (2 ^ 7 * (3 / δ)) :=
      Real.log_le_log (div_pos (by norm_num) hδ) (by
        rw [div_le_iff₀ hδ, mul_assoc, div_mul_cancel₀ _ hδ.ne']
        norm_num)
    rw [Real.log_mul (by norm_num) (by positivity), Real.log_pow, Nat.cast_ofNat] at h1
    linarith only [h1]
  have hC : 800 * Real.log (128 / δ) / (81 * pAP) + 2 ≤ 60 * R := by
    have hR2 : 2 ≤ 2 / pAP := by
      rw [le_div_iff₀ hpAP]; linarith only [hpAP1]
    have e : 800 * Real.log (128 / δ) / (81 * pAP)
        = 800 / 81 * Real.log (128 / δ) / pAP := by ring
    rw [e, hRdef]
    have key : 800 / 81 * Real.log (128 / δ) + 2 ≤ 60 * H := by
      linarith only [hlog128', hH3, hlog3, hl2']
    have key' : 800 / 81 * Real.log (128 / δ) / pAP + 2 / pAP ≤ 60 * H / pAP := by
      rw [← add_div]
      exact div_le_div_of_nonneg_right key hpAP.le
    calc 800 / 81 * Real.log (128 / δ) / pAP + 2
        ≤ 800 / 81 * Real.log (128 / δ) / pAP + 2 / pAP := by linarith only [hR2]
      _ ≤ 60 * H / pAP := key'
      _ = 60 * (H / pAP) := by ring
  have h1 := ceil_le_of_le hA (div_nonneg (by linarith only [hPnn]) hpAP.le)
  have h2 := Nat.ceil_lt_add_one (show (0 : ℝ) ≤ 800 * Real.log (128 / δ)
    / (81 * pAP) from div_nonneg (mul_nonneg (by norm_num) (Real.log_nonneg (by
      rw [le_div_iff₀ hδ]; linarith only [hδ1]))) (by positivity))
  have hpoolval : (poolCount η populations indecisionLimit εcov δ pAP crossLimit : ℝ)
      = (⌈2 * ((famCount η populations indecisionLimit εcov δ crossLimit : ℝ) + 1) / pAP⌉₊ : ℝ)
        + (⌈800 * Real.log (128 / δ) / (81 * pAP)⌉₊ : ℝ) := by
    rw [poolCount]
    push_cast
    ring
  rw [hpoolval, mul_div_assoc 60]
  linarith only [h1, h2, hC]

/-- The statement's three prefix hypotheses at the top rung are the collision cap. -/
lemma cap_of_top {J : Type*} [Fintype J] (populations : Finset J)
    (η₀ indecisionLimit εcov εs δ α pAP crossLimit ρ : ℝ) (hη0 : 0 ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (hε : 0 < εcov) (hε1 : εcov ≤ 1) (hind : 0 < indecisionLimit)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hρ0 : 0 ≤ ρ)
    (hN : (0 : ℝ) < (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ))
    (h1 : 2 ^ 17 * ((populations.card : ℝ) + 3)
        * (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ) * ρ
      ≤ sig η₀ * min (εcov * sig η₀) indecisionLimit)
    (h1' : 2 ^ 8 * ((populations.card : ℝ) + 3)
        * (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ) * ρ
      ≤ εs)
    (h2 : 512 * Real.log (64 * (populations.card : ℝ)
        * (3 + 2 * (Nat.log 2 (prefCount η₀ populations indecisionLimit εcov εs δ α pAP
          crossLimit hsz) : ℝ)) / δ) * ρ
      ≤ min εs indecisionLimit ^ 2) :
    ρ ≤ collisionCap η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz := by
  set N : ℝ := (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz : ℝ)
    with hNdef
  set P : ℝ := (populations.card : ℝ) with hP
  have hs := sig_pos η₀ hη₀
  have hsig1 : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
  set R : ℝ := sig η₀ * min (εcov * sig η₀) indecisionLimit with hRdef
  have hRil : R ≤ indecisionLimit / 2 := by
    have h := mul_le_mul hsig1 (min_le_right (εcov * sig η₀) indecisionLimit)
      (le_min (by positivity) hind.le) (by norm_num)
    rw [hRdef]; linarith
  have hc := rejFrac_ge_lin (εs := εs) η₀ indecisionLimit εcov hε hε1 hind hη₀
  rw [← hRdef] at hc
  have hPN : ρ * (128 * P * N) ≤ ρ * (256 * (P + 3) * N) :=
    mul_le_mul_of_nonneg_left (by nlinarith) hρ0
  rw [collisionCap, ← hNdef, ← hP]
  refine le_min (le_min ?_ ?_) ?_
  · rw [le_div_iff₀ (by positivity)]
    refine le_trans (le_min ?_ ?_) hc
    · linarith
    · linarith
  · rw [le_div_iff₀ (by positivity)]
    refine le_min (by linarith) ?_
    linarith
  · have hP1 : (1 : ℝ) ≤ P := by
      have hc := hcard
      rw [hP] at hc ⊢
      exact Nat.one_le_cast.2 (by exact_mod_cast hc)
    have hLg : 0 < Real.log (64 * P * (3 + 2 * (Nat.log 2 (prefCount η₀ populations indecisionLimit
        εcov εs δ α pAP crossLimit hsz) : ℝ)) / δ) := by
      refine Real.log_pos ?_
      rw [lt_div_iff₀ hδ]
      have : (0 : ℝ) ≤ (Nat.log 2 (prefCount η₀ populations indecisionLimit εcov εs δ α pAP
        crossLimit hsz) : ℝ) := Nat.cast_nonneg _
      nlinarith
    rw [le_div_iff₀ (by positivity)]
    linarith

/-- The statement's pool hypothesis at the top rung is the pool's collision cap. -/
lemma poolCap_of_top {J : Type*} [Fintype J] (populations : Finset J)
    (η₀ indecisionLimit εcov δ pAP crossLimit ρsf : ℝ)
    (hM : (0 : ℝ) < (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ))
    (h2 : 2 ^ 13 * (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) * ρsf
      ≤ δ * pAP) :
    ρsf ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit := by
  rw [poolCollisionCap, le_div_iff₀ (by positivity)]
  linarith

/-- The lowest rung whose rejections are rare enough draws at most the claim's second count. -/
lemma rejCountMin_le {J : Type*} [Fintype J] (populations : Finset J)
    (η₀ indecisionLimit εcov εs δ : ℝ) (hη0 : 0 ≤ η₀) (hη₀ : η₀ < 1 / 2) (hε : 0 < εcov)
    (hε1 : εcov ≤ 1)
    (hεs : 0 < εs) (hind : 0 < indecisionLimit) (hind1 : indecisionLimit ≤ 1 / 2) (hδ : 0 < δ)
    (hδ1 : δ ≤ 1) (hcard : (1 : ℝ) ≤ (populations.card : ℝ)) :
    2 * (rejCountMin η₀ populations indecisionLimit εcov εs δ : ℝ)
      ≤ max (2 ^ 13 * Real.log (32 * (populations.card : ℝ) / δ)
          / (sig η₀ * min (εcov * sig η₀) indecisionLimit))
        (12 * Real.log (32 * (populations.card : ℝ) / δ) / εs) := by
  have hs := sig_pos η₀ hη₀
  set R : ℝ := sig η₀ * min (εcov * sig η₀) indecisionLimit with hRdef
  set c : ℝ := rejFrac η₀ indecisionLimit εcov εs with hcdef
  set Lg : ℝ := Real.log (32 * (populations.card : ℝ) / δ) with hLg
  have hR0 : 0 < R := mul_pos hs (lt_min (mul_pos hε hs) hind)
  have hc := rejFrac_ge_lin (εs := εs) η₀ indecisionLimit εcov hε hε1 hind hη₀
  rw [← hRdef, ← hcdef] at hc
  have hc0 : 0 < c := rejFrac_pos η₀ hη₀ hε hεs hind
  have hl2 : 0.6931471803 < Real.log 2 := Real.log_two_gt_d9
  have hLg3 : 3 ≤ Lg := by
    rw [hLg, Real.le_log_iff_exp_le (by positivity)]
    have he : Real.exp 1 < 2.7182818286 := Real.exp_one_lt_d9
    have h3 : Real.exp 3 = Real.exp 1 ^ 3 := by rw [← Real.exp_nat_mul]; norm_num
    rw [h3, le_div_iff₀ hδ]
    have : Real.exp 1 ^ 3 ≤ 21 := by
      have h0 : 0 < Real.exp 1 := Real.exp_pos 1
      have : Real.exp 1 ^ 3 ≤ (2.7182818286 : ℝ) ^ 3 := pow_le_pow_left₀ h0.le he.le 3
      nlinarith
    nlinarith
  have hsig1 : sig η₀ ≤ 1 / 2 := by rw [sig]; linarith
  have hR4 : R ≤ 1 / 4 := by
    have := mul_le_mul hsig1 ((min_le_right (εcov * sig η₀) indecisionLimit).trans hind1)
      (le_min (by positivity) hind.le) (by norm_num)
    rw [hRdef]; linarith
  have hceil : (rejCountMin η₀ populations indecisionLimit εcov εs δ : ℝ)
      < Lg / (c * Real.log 2) + 1 := by
    rw [rejCountMin]
    exact Nat.ceil_lt_add_one (div_nonneg (by linarith) (by positivity))
  have h2 : 0.6931471803 * c ≤ c * Real.log 2 := by nlinarith
  rcases le_total (R / 2048) (εs / 4) with hm | hm
  swap
  · -- the redraws held at `εs/4`
    rw [min_eq_right hm] at hc
    refine le_trans ?_ (le_max_right _ _)
    have hdiv : Lg / (c * Real.log 2) ≤ 29 / 5 * Lg / εs := by
      rw [div_le_div_iff₀ (by positivity) hεs]
      have hL0 : 0 ≤ Lg := by linarith
      nlinarith [mul_le_mul_of_nonneg_left h2 hL0, mul_le_mul_of_nonneg_left hc hL0]
    have hfin : 2 * (29 / 5 * Lg / εs + 1) ≤ 12 * Lg / εs := by
      rw [show 12 * Lg / εs = 2 * (29 / 5 * Lg / εs) + 2 / 5 * Lg / εs by ring]
      have : 2 ≤ 2 / 5 * Lg / εs := by
        rw [le_div_iff₀ hεs]; linarith
      linarith
    linarith
  rw [min_eq_left hm] at hc
  refine le_trans ?_ (le_max_left _ _)
  have hdiv : Lg / (c * Real.log 2) ≤ 2955 * Lg / R := by
    rw [div_le_div_iff₀ (by positivity) hR0]
    have h3 : R ≤ 2955 * (c * Real.log 2) := by linarith
    have hL0 : 0 ≤ Lg := by linarith
    have := mul_le_mul_of_nonneg_left h3 hL0
    linarith
  have hfin : 2 * (2955 * Lg / R + 1) ≤ 2 ^ 13 * Lg / R := by
    rw [show 2 ^ 13 * Lg / R = 2 * (2955 * Lg / R) + 2282 * Lg / R by ring]
    have : 2 ≤ 2282 * Lg / R := by
      rw [le_div_iff₀ hR0]; linarith
    linarith
  linarith

/-- A vote over a family no larger than a rung's own reads its rarer decided side at most `κ`
times as often as undecided, whatever its two rates, for any `κ` with `crossLimit ≤ κ⁴`. -/
theorem kappa_of_mem_schedule {η₀ indecisionLimit εcov εs δ α pAP crossLimit κ p r : ℝ}
    {populations : Finset J} {B : State} (hη0 : 0 ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (hcross : 0 < crossLimit) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hB : B ∈ schedule η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hsz)
    (N : ℕ) (hN : N + 1 ≤ B.k) (hκ : 0 ≤ κ) (hcκ : crossLimit ≤ κ ^ 4)
    (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hr0 : 0 ≤ r) (hr1 : r ≤ 1) :
    RarerBounded N B.lo (B.hi + 1) p r κ := by
  obtain ⟨hBk, hBlo, hBhi⟩ := band_of_mem_schedule hB
  have hκ4 := famCount_kappa (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    (δ := δ) η₀ populations hη0 hη₀ hcross hε hind
  have hev := famCount_even (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ
  have h4 := famCount_ge_four (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hη₀
  set κc : ℕ := famCount η₀ populations indecisionLimit εcov δ crossLimit with hκc
  obtain ⟨hlo, hhi⟩ := loAt_hiAt η₀ hev h4 hη0 hη₀
  have hb0 : (0 : ℝ) ≤ bandAt η₀ κc := Nat.cast_nonneg _
  have hκ2 : ((κc / 2 : ℕ) : ℝ) = (κc : ℝ) / 2 := by
    obtain ⟨m, hm⟩ := hev
    rw [hm, show (m + m) / 2 = m by omega]; push_cast; ring
  have hlo2 : loAt η₀ κc < κc / 2 := by
    have : (loAt η₀ κc : ℝ) < ((κc / 2 : ℕ) : ℝ) := by rw [hκ2, hlo]; linarith
    exact_mod_cast this
  have hhi2 : κc / 2 < hiAt η₀ κc + 1 := by
    have : ((κc / 2 : ℕ) : ℝ) < ((hiAt η₀ κc + 1 : ℕ) : ℝ) := by push_cast; rw [hκ2, hhi]; linarith
    exact_mod_cast this
  have hκ1 : bandKappa κc (loAt η₀ κc) (κc / 2) (hiAt η₀ κc + 1) ≤ κ :=
    le_of_pow_le_pow_left₀ (by norm_num : (4 : ℕ) ≠ 0) hκ (hκ4.trans hcκ)
  rw [hBlo, hBhi]
  exact (bandKappaBound_holds N κc _ _ _ p r (by omega) hlo2 hhi2 (Nat.div_le_self _ _) hp0 hp1
    hr0 hr1).mono hκ1 hp0 hp1 hr0 hr1

open MeasureTheory ProbabilityTheory in
/-- From `clustering_correct`, with the schedule and the collision caps it names as the
witnesses, `prefCount_le_split` for the prefix count, `validCount_le` for the lowest rung, and
`famCount_succ_le` and `poolCount_le` for the family and the pool.  The states are the rungs that
carry their share at the collision cap, which the top rung always does; the statement's
hypotheses at the top rung are the caps. -/
theorem clustering_guarantee_of_correct : ClusteringGuarantee := by
  intro Ω _ μ _ S _ J _ O populations tame uni H Pre Suf η₀ indecisionLimit εcov εs α δ pAP
    crossLimit hηle hη₀ huni htame hflat hH hpAPPositive hindLim hind1 hαpos hα hεcov hε1 hεs hδ
    hδ1 hζ hζ1
  classical
  set hz : ℕ := (allHarvest H).card with hhz
  have hpop : populations.Nonempty := ⟨uni, htame huni⟩
  simp only [ret_eq, familyAt_eq]
  set rule : Clusterer S := nearestClusterer with hrule
  have hsig : 0 < sig η₀ := by simp only [sig]; linarith
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hηle
  have hcard : (0 : ℝ) < (populations.card : ℝ) := by exact_mod_cast Finset.card_pos.2 hpop
  have hscaleEq : budgetScale η₀ indecisionLimit εcov = rateScale η₀ indecisionLimit εcov := by
    unfold budgetScale rateScale gateA; rw [min_eq_left hε1]
  have hscale : 0 < rateScale η₀ indecisionLimit εcov := by
    rw [← hscaleEq]; exact budgetScale_pos η₀ indecisionLimit εcov hεcov hindLim hη0 hη₀
  have hscale1 : rateScale η₀ indecisionLimit εcov ≤ 1 / 2 := le_trans (min_le_right _ _) hind1
  have hlog128 : 1 ≤ Real.log (128 * (populations.card : ℝ) / δ) := by
    have hc1 : (1 : ℝ) ≤ (populations.card : ℝ) := Nat.one_le_cast.2 (Finset.card_pos.2 hpop)
    have h : Real.exp 1 ≤ 128 * (populations.card : ℝ) / δ := by
      rw [le_div_iff₀ hδ]
      nlinarith [Real.exp_one_lt_d9]
    have := Real.log_le_log (Real.exp_pos 1) h
    rwa [Real.log_exp] at this
  have h256 : 0 ≤ 6 * Real.log (256 * (populations.card : ℝ) / δ) / εs := by
    have : Real.log (128 * (populations.card : ℝ) / δ)
        ≤ Real.log (256 * (populations.card : ℝ) / δ) :=
      Real.log_le_log (by positivity) (by gcongr; norm_num)
    exact div_nonneg (by linarith) hεs.le
  by_cases hpAP1 : pAP ≤ 1
  swap
  · -- No measure puts more than all its mass anywhere, so no suffix measure meets `pAP`, and a
    -- single empty state carries the bounds.
    have hlogB : 0 ≤ Real.log (2 / rateScale η₀ indecisionLimit εcov) :=
      Real.log_nonneg (by rw [le_div_iff₀ hscale]; linarith)
    have hlogZ : 0 ≤ Real.log (1 / crossLimit) :=
      Real.log_nonneg (by rw [le_div_iff₀ hζ]; linarith)
    have hlogJ : 0 ≤ Real.log (3 / δ) :=
      Real.log_nonneg (by rw [le_div_iff₀ hδ]; linarith)
    have hs : 0 < 1 / 2 - η₀ := by linarith
    refine ⟨{⟨0, 0, 0, 0, 0, 0, 0⟩}, fun B hB => ?_,
      ⟨⟨0, 0, 0, 0, 0, 0, 0⟩, Finset.mem_singleton_self _, ?_⟩, fun B hB => ?_,
      fun B hB F hF => ?_, fun B hB N hN => ?_, ?_⟩
    · rw [Finset.mem_singleton.1 hB]
      simp only [Nat.cast_zero, mul_zero, zero_div, Real.log_zero, zero_add]
      have : (0 : ℝ) ≤ 2 ^ 11 * (hz : ℝ) / rateScale η₀ indecisionLimit εcov := by positivity
      have h64 : 0 ≤ Real.log (64 * 1 / δ) := Real.log_nonneg (by rw [le_div_iff₀ hδ]; linarith)
      have hm : 0 < min εcov indecisionLimit := lt_min hεcov hindLim
      have : 0 ≤ 2 ^ 20 * (populations.card : ℝ) * Real.log (64 * 1 / δ)
          / ((1 / 2 - η₀) ^ 4 * min εcov indecisionLimit ^ 2) := by positivity
      linarith
    · simp only [Nat.cast_zero]
      have h32 : 0 ≤ Real.log (32 * (populations.card : ℝ) / δ) :=
        Real.log_nonneg (by
          rw [le_div_iff₀ hδ]
          have : (1 : ℝ) ≤ (populations.card : ℝ) := Nat.one_le_cast.2 (Finset.card_pos.2 hpop)
          linarith)
      exact le_max_of_le_right (by positivity)
    · rw [Finset.mem_singleton.1 hB]
      simp only [Nat.cast_zero]
      constructor <;> positivity
    · rw [Finset.mem_singleton.1 hB] at hF
      simp at hF
    · rw [Finset.mem_singleton.1 hB] at hN
      simp at hN
    · intro D Dsf hD hDsf _ _ hpAPBound
      exact absurd (le_trans hpAPBound measureReal_le_one) hpAP1
  set ρ₀ : ℝ := collisionCap η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hz with hρ₀
  have hN : (0 : ℝ)
      < (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hz : ℝ) := by
    exact_mod_cast prefCount_pos (εs := εs) (indecisionLimit := indecisionLimit)
      (crossLimit := crossLimit) η₀ populations εcov δ α pAP
  have hρ₀0 : 0 ≤ ρ₀ := by
    rw [hρ₀, collisionCap]
    have hc0 := rejFrac_pos η₀ hη₀ hεcov hεs hindLim (indecisionLimit := indecisionLimit)
    have hlog : 0 ≤ Real.log (64 * (populations.card : ℝ)
        * (3 + 2 * (Nat.log 2 (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit
          hz) : ℝ)) / δ) := by
      refine Real.log_nonneg ?_
      rw [le_div_iff₀ hδ]
      have : (0 : ℝ) ≤ (Nat.log 2 (prefCount η₀ populations indecisionLimit εcov εs δ α pAP
        crossLimit hz) : ℝ) := Nat.cast_nonneg _
      have hc1 : (1 : ℝ) ≤ (populations.card : ℝ) := Nat.one_le_cast.2 (Finset.card_pos.2 hpop)
      nlinarith
    exact le_min (le_min (div_nonneg hc0.le (by positivity)) (by positivity))
      (div_nonneg (sq_nonneg _) (mul_nonneg (by norm_num) hlog))
  have htop := solvedState_mem_stoppable (α := α) (crossLimit := crossLimit) (hsz := hz) η₀
    populations hη0 hη₀ hεcov hεs hindLim hδ hδ1 hpAPPositive hcard hρ₀0 le_rfl
  have hpoly := prefCount_le_split (hsz := hz) populations η₀ indecisionLimit εcov εs δ α pAP
    crossLimit hsig hη0 hpop hindLim hεcov hεs hδ hδ1 hαpos (by linarith) hpAPPositive
    hpAP1
  have hcutge := cutBudget_ge η₀ indecisionLimit εcov hεcov hindLim hη0 hη₀
  have hharv : 128 * (hz : ℝ) / cutBudget η₀ indecisionLimit εcov
      ≤ 2 ^ 11 * (hz : ℝ) / rateScale η₀ indecisionLimit εcov := by
    rw [← hscaleEq] at hscale ⊢
    rw [div_le_div_iff₀ (by linarith) hscale]
    have : (0 : ℝ) ≤ hz := Nat.cast_nonneg _
    nlinarith
  have hshift : shiftScale indecisionLimit εcov = min εcov indecisionLimit := by
    unfold shiftScale; rw [min_eq_left hε1]
  have hpoly' := hpoly
  simp only [budgetLog, levelLog, hscaleEq, hshift, sig] at hpoly'
  refine ⟨stoppable η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hz ρ₀,
    ?_, ?_, ?_, fun B hB => cross_of_mem_schedule O hη0 hη₀ hζ
      (Finset.mem_of_mem_filter _ hB),
    fun B hB N hN κ p r hκ hcκ hp0 hp1 hr0 hr1 => kappa_of_mem_schedule hη0 hη₀ hζ hεcov hindLim
      (Finset.mem_of_mem_filter _ hB) N hN hκ hcκ hp0 hp1 hr0 hr1, ?_⟩
  · -- Every rung's count is the top one halved, and the top one is what `budgetCap` bounds.
    intro B hB
    have hsched : B ∈ schedule η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hz :=
      Finset.mem_of_mem_filter _ hB
    obtain ⟨i, _, rfl⟩ := Finset.mem_image.1 hsched
    refine le_trans ?_ (le_trans hpoly' ?_)
    · exact_mod_cast Nat.div_le_self _ _
    simp only [solvedStateAt]
    linarith [hharv]
  · -- The ladder reaches down to `validCount` and to `rejCountMin`.
    obtain ⟨B, hB, hBle⟩ := exists_small_stoppable (hsz := hz) η₀ populations hη0 hη₀ hεcov hεs
      hindLim hδ hδ1
      hpAPPositive hcard hρ₀0 le_rfl ⟨_, htop⟩
    refine ⟨B, hB, ?_⟩
    have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
      have h1 : 0 < populations.card := Finset.card_pos.2 hpop
      exact_mod_cast h1
    have hrm := rejCountMin_le populations η₀ indecisionLimit εcov εs δ hη0 hη₀ hεcov (by linarith)
      hεs hindLim hind1 hδ hδ1 hcard1
    simp only [sig] at hrm
    have hval := validCount_le_rej populations η₀ indecisionLimit εcov δ pAP crossLimit hη0 hη₀
      hεcov hindLim hind1 hpop hδ hδ1
    simp only [sig] at hval
    refine le_trans hBle ?_
    rw [mul_max_of_nonneg _ _ (by norm_num : (0 : ℝ) ≤ 2)]
    exact max_le (le_trans hval (le_max_left _ _)) hrm
  · intro B hB
    obtain ⟨i, _, rfl⟩ := Finset.mem_image.1 (Finset.mem_of_mem_filter _ hB)
    have hk := famCount_succ_le populations η₀ indecisionLimit εcov δ crossLimit hsig hη0
      hindLim hεcov hζ hζ1
    have hpool := poolCount_le populations η₀ indecisionLimit εcov δ pAP crossLimit hsig hη0
      hindLim hεcov hδ hδ1 hpAPPositive hpAP1 hζ hζ1
    rw [hscaleEq] at hk hpool
    simp only [sig] at hk hpool
    simp only [solvedStateAt]
    push_cast
    constructor <;> linarith
  -- the statement's hypotheses, read at the top rung, are the caps
  intro D Dsf hD hDsf hsupp hsuppSf hpAPBound hcollD hcollE hcollH hcollSf
  have := hD
  have := hDsf
  have hM : (0 : ℝ) < (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) := by
    have h := poolCount_ge populations η₀ indecisionLimit εcov δ pAP crossLimit hsig hη0 hindLim
      hεcov hpAPPositive hpAP1
    linarith
  have hcaps : ∀ j ∈ tame, ∀ t, (D j).real {t} ≤ ρ₀ := fun j hj t =>
    cap_of_top populations η₀ indecisionLimit εcov εs δ α pAP crossLimit ((D j).real {t}) hη0
      hη₀ hεcov (by linarith) hindLim hcard hδ hδ1 measureReal_nonneg hN
      (by rw [sig]; exact hcollD _ htop j hj t) (hcollE _ htop j hj t)
      (hcollH _ htop j hj t)
  have hcapsf : collisionMass Dsf
      ≤ poolCollisionCap η₀ populations indecisionLimit εcov δ pAP crossLimit :=
    poolCap_of_top populations η₀ indecisionLimit εcov δ pAP crossLimit (collisionMass Dsf) hM
      (hcollSf _ htop)
  have hbudget : (prefCount η₀ populations indecisionLimit εcov εs δ α pAP crossLimit hz : ℝ)
      ≤ budgetCap populations η₀ indecisionLimit εcov δ α pAP crossLimit (2 ^ 26)
        + 6 * Real.log (256 * (populations.card : ℝ) / δ) / εs
        + harvestCap η₀ indecisionLimit εcov hz :=
    hpoly.trans (by
      rw [harvestCap]
      have := budgetCap_split populations η₀ indecisionLimit εcov δ α pAP crossLimit hindLim hεcov
        hδ hδ1
      linarith)
  refine le_trans (clustering_correct O populations tame uni H D Dsf Pre Suf η₀
    indecisionLimit εcov εs α δ ρ₀ pAP crossLimit (2 ^ 26) hηle hη₀ huni htame hflat hH hsupp
    hsuppSf hcaps hpAPPositive hpAPBound hindLim hind1 hαpos hα hεcov (by linarith) hεs hδ hbudget
    le_rfl hcapsf)
    (measureReal_mono ?_ (measure_ne_top _ _))
  have hrej : 2 * rejFrac η₀ indecisionLimit εcov εs ≤ εs / 2 := by
    have h := (rejFrac_le (εs := εs) (indecisionLimit := indecisionLimit) η₀ (εcov := εcov)).2.2
    linarith
  rintro x ⟨hex, hall⟩
  have huni' : ∀ B : {B : State // B ∈ stoppable η₀ populations indecisionLimit εcov εs δ α pAP
      crossLimit hz ρ₀}, streams H uni x ∈ retBy rule O.mq populations uni η₀ indecisionLimit
        εcov α B.val →
      1 - εcov - εs ≤ (D uni).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
        (familyBy rule O.mq populations (streams H uni x) B.val) p (oracleNoise x.1)} :=
    fun B hB => by
      have h := (hall B hB uni huni).1
      simp only [gateLimit, if_pos rfl, if_true, sub_zero] at h
      linarith
  refine ⟨hex, fun B hB => ⟨huni' B hB, fun j hj => ⟨?_, (hall B hB j hj).2⟩⟩⟩
  by_cases hne : j = uni
  · subst hne; linarith [huni' B hB]
  · have h := (hall B hB j hj).1
    simp only [gateLimit, if_neg hne] at h
    linarith

end OrthoDFA
