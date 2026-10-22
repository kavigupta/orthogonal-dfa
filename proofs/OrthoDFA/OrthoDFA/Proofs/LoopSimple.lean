import Mathlib.Analysis.SpecialFunctions.Log.Base
import OrthoDFA.Proofs.LearnerLoop
import OrthoDFA.Proofs.Budget

/-!
# The loop's claim at explicit settings

`capPB` and `capRB` bound the suffix pool and the prefix count over at most `Jm` populations, by
`poolCount_le` and `prefCount_le_poly` (`counts_le`). `loop_simple_checks` takes `LoopCore` at
`K = |Q| + 1`, with the failure split evenly between the clustering's and level 3's `|Q| + 1`
rounds.
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory

/-- A bound on the suffix pool, the clustering's `crossLimit` at `exp (-cLog)`. -/
def capPB (η₀ ι εcov δ pAP cLog Jm : ℝ) : ℝ :=
  2 * (25 * max (4 * Real.log (20 / budgetScale η₀ ι εcov)) cLog / (8 * (1 / 2 - η₀) ^ 2)
      + 5 / (1 / 2 - η₀) + 7) / pAP
    + 16 * Real.log ((Jm + 2) / (δ / 2)) / pAP ^ 2

/-- A bound on the prefix count. -/
def capRB (η₀ ι εcov α δ pAP cLog Jm : ℝ) : ℝ :=
  161000 * Jm ^ 2 * Real.log ((Jm + 2) * capPB η₀ ι εcov δ pAP cLog Jm
    / (δ / 2 * α * pAP * budgetScale η₀ ι εcov))
    / ((1 / 2 - η₀) ^ 4 * budgetScale η₀ ι εcov ^ 2)

theorem counts_le {J : Type*} [Fintype J] (η₀ : ℝ) (pops : Finset J)
    (ι εcov α δ pAP cLog Jm : ℝ) (hη0 : 0 ≤ η₀) (hη : η₀ < 1 / 2)
    (hpop : pops.Nonempty) (hcard : (pops.card : ℝ) ≤ Jm) (hι : 0 < ι) (hε : 0 < εcov)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hα : 0 < α) (hα1 : α < 1) (hpAP : 0 < pAP)
    (hpAP1 : pAP ≤ 1) (hc : 0 ≤ cLog) :
    (poolCount η₀ pops ι εcov (δ / 2) pAP (Real.exp (-cLog)) : ℝ)
        ≤ capPB η₀ ι εcov δ pAP cLog Jm
      ∧ (prefCount η₀ pops ι εcov (δ / 2) α pAP (Real.exp (-cLog)) 0 : ℝ)
        ≤ capRB η₀ ι εcov α δ pAP cLog Jm := by
  have hsig : 0 < sig η₀ := by unfold sig; linarith
  have hδ2 : 0 < δ / 2 := by positivity
  have hδ21 : δ / 2 ≤ 1 := by linarith
  have hc1 : (1 : ℝ) ≤ pops.card := Nat.one_le_cast.2 (Finset.card_pos.2 hpop)
  set s := 1 / 2 - η₀ with hs
  have hs0 : 0 < s := by linarith
  have hsig' : sig η₀ = s := rfl
  set τ := budgetScale η₀ ι εcov with hτ
  have hτ' : budgetScale η₀ ι εcov = τ := rfl
  have hτ0 : 0 < τ := budgetScale_pos η₀ ι εcov hε hι hη0 hη
  have hτ1 : τ ≤ 1 := (budgetScale_le_sig η₀ ι εcov hη0 hη).trans (by unfold sig; linarith)
  have hcross : 0 < Real.exp (-cLog) := Real.exp_pos _
  have hcross1 : Real.exp (-cLog) ≤ 1 := Real.exp_le_one_iff.2 (by linarith)
  set PB := capPB η₀ ι εcov δ pAP cLog Jm with hPB
  set pool := (poolCount η₀ pops ι εcov (δ / 2) pAP (Real.exp (-cLog)) : ℝ) with hpool
  set pref := (prefCount η₀ pops ι εcov (δ / 2) α pAP (Real.exp (-cLog)) 0 : ℝ) with hpref
  -- the pool
  have hpoolB : pool ≤ PB := by
    have h := poolCount_le pops η₀ ι εcov (δ / 2) pAP (Real.exp (-cLog)) hsig hη0 hι hε hpop hδ2
      hδ21 hpAP hpAP1 hcross hcross1
    rw [hsig', hτ'] at h
    have e : Real.log (1 / Real.exp (-cLog)) = cLog := by
      rw [one_div, ← Real.exp_neg, neg_neg, Real.log_exp]
    rw [e] at h
    refine h.trans (add_le_add le_rfl ?_)
    refine div_le_div_of_nonneg_right ?_ (by positivity)
    refine mul_le_mul_of_nonneg_left (Real.log_le_log (by positivity) ?_) (by norm_num)
    exact div_le_div_of_nonneg_right (by linarith) hδ2.le
  -- the prefix count
  have hlogarg : 1 ≤ ((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ) := by
    have hpool256 : (64 : ℝ) ≤ pool :=
      poolCount_ge pops η₀ ι εcov (δ / 2) pAP _ hsig hη0 hι hε hpAP hpAP1
    rw [le_div_iff₀ (by positivity)]
    have h1 : δ / 2 * α * pAP * τ ≤ 1 := by
      have : δ / 2 * α ≤ 1 := by nlinarith
      have : δ / 2 * α * pAP ≤ 1 := by nlinarith
      nlinarith
    nlinarith
  refine ⟨hpoolB, ?_⟩
  have h := prefCount_le_poly (hsz := 0) pops η₀ ι εcov (δ / 2) α pAP (Real.exp (-cLog)) hsig
    hη0 hpop hι hε hδ2 hδ21 hα hα1 hpAP hpAP1
  rw [Nat.cast_zero, mul_zero, zero_div, add_zero] at h
  rw [hsig', hτ'] at h
  refine h.trans ?_
  unfold budgetLog capRB
  rw [hτ', ← hPB]
  refine div_le_div_of_nonneg_right ?_ (by positivity)
  have hl0 : 0 ≤ Real.log (((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ)) :=
    Real.log_nonneg hlogarg
  have hlog : Real.log (((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ))
      ≤ Real.log ((Jm + 2) * PB / (δ / 2 * α * pAP * τ)) := by
    refine Real.log_le_log (by linarith) (div_le_div_of_nonneg_right ?_ (by positivity))
    exact mul_le_mul (by linarith) hpoolB (by positivity) (by linarith)
  have hsq : (pops.card : ℝ) ^ 2 ≤ Jm ^ 2 := pow_le_pow_left₀ (by positivity) hcard 2
  calc 161000 * (pops.card : ℝ) ^ 2
        * Real.log (((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ))
      ≤ 161000 * Jm ^ 2 * Real.log (((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ)) := by
        gcongr
    _ ≤ 161000 * Jm ^ 2 * Real.log ((Jm + 2) * PB / (δ / 2 * α * pAP * τ)) := by
        gcongr

end OrthoDFA

namespace OrthoDFA

open MeasureTheory

open Random (prefixMax le_prefixMax prefixMax_pos)

/-- The race tail at `t = 1/(λ + 2c)` and `t' = 2t`:

    exp(t'λ)/(1 + t)² ≤ e² (λ + 2c)². -/
lemma race_tail_choice {lam c : ℝ} (hl : 0 ≤ lam) (hc : 0 ≤ c) (hpos : 0 < lam + 2 * c) :
    0 ≤ 1 / (lam + 2 * c) ∧ 0 ≤ 2 * (1 / (lam + 2 * c))
      ∧ 1 / (lam + 2 * c) * c ≤ 1 - Real.exp (-(2 * (1 / (lam + 2 * c)) * c))
      ∧ Real.exp (2 * (1 / (lam + 2 * c)) * lam) / (1 + 1 / (lam + 2 * c)) ^ 2
        ≤ Real.exp 2 * (lam + 2 * c) ^ 2 := by
  set t := 1 / (lam + 2 * c) with ht
  have ht0 : 0 < t := by positivity
  refine ⟨ht0.le, by positivity, ?_, ?_⟩
  · set x := t * c with hx
    have hx0 : 0 ≤ x := by positivity
    have hx2 : x ≤ 1 / 2 := by
      rw [hx, ht, div_mul_eq_mul_div, one_mul, div_le_iff₀ hpos]; linarith
    have h1 : 1 + 2 * x ≤ Real.exp (2 * x) := by linarith [Real.add_one_le_exp (2 * x)]
    have h2 : Real.exp (-(2 * x)) ≤ 1 / (1 + 2 * x) := by
      rw [Real.exp_neg, ← one_div]
      exact one_div_le_one_div_of_le (by positivity) h1
    have h3 : 1 / (1 + 2 * x) ≤ 1 - x := by
      rw [div_le_iff₀ (by positivity)]; nlinarith
    have e : 2 * t * c = 2 * x := by rw [hx]; ring
    rw [e]
    linarith
  · have h1 : 2 * t * lam ≤ 2 := by
      rw [ht, show 2 * (1 / (lam + 2 * c)) * lam = 2 * (lam / (lam + 2 * c)) by ring]
      have : lam / (lam + 2 * c) ≤ 1 := by rw [div_le_one hpos]; linarith
      linarith
    have h2 : t ^ 2 ≤ (1 + t) ^ 2 := pow_le_pow_left₀ ht0.le (by linarith) 2
    have h3 : t ^ 2 * (lam + 2 * c) ^ 2 = 1 := by
      rw [ht, ← mul_pow, one_div_mul_cancel hpos.ne', one_pow]
    rw [div_le_iff₀ (by positivity)]
    calc Real.exp (2 * t * lam) ≤ Real.exp 2 := Real.exp_le_exp.2 h1
      _ = Real.exp 2 * (lam + 2 * c) ^ 2 * t ^ 2 := by
          rw [mul_assoc, mul_comm _ (t ^ 2), h3, mul_one]
      _ ≤ Real.exp 2 * (lam + 2 * c) ^ 2 * (1 + t) ^ 2 := by gcongr

/-- Level 5's premises at the band's `κ = ζ`, once the round's undecided reads, weighted `ζ`,
stay under `√δ/(4e)`, and the pairs of probes under `δ/16`. -/
theorem roundPremises_of {σ α : Type*} [Fintype σ] [Fintype α] (C : Random.Level3Settings)
    (cross p₀ Wm : ℝ) :
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    0 < C.δ → C.δ < 1 → 0 < C.harvestRate → C.harvestRate ≤ C.disagreementRate / 48 →
    C.disagreementRate < 1 →
    36 * S / C.disagreementRate * Real.log (16 * S * (S + 1) / C.δ) + S ≤ C.probeBudget →
    0 < cross →
    C.probeBudget * (C.probeBudget - 1 : ℝ) / 2 * p₀ ≤ C.δ / 16 →
    16 / 5 * S * Real.log (8 * S * (S + 1) / C.δ) + 3 + (nQ + 2) * nα * (nQ + 1) ≤ Wm →
    cross * Wm ≤ Real.sqrt C.δ / (4 * Real.exp 1) →
    RoundPremises σ α C cross p₀ := by
  intro nQ nα S hδ0 hδ1 hy hyε hε hP hcr hpair hWm hcm
  unfold RoundPremises
  dsimp only
  refine ⟨hδ0, hδ1, hy, hyε, hε, hP, hcr.le, ?_⟩
  have hS1 : (1 : ℝ) ≤ S := by
    have : 1 ≤ S := by omega
    exact_mod_cast this
  have hlog : 0 ≤ Real.log (8 * S * (S + 1) / C.δ) :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ0]; nlinarith)
  set lam : ℝ := cross * (16 / 5 * S * Real.log (8 * S * (S + 1) / C.δ) + 1
    + (nQ + 2) * nα * (nQ + 1)) with hlamd
  have hlam0 : 0 ≤ lam := by positivity
  have hpos : 0 < lam + 2 * cross := by linarith
  obtain ⟨ht, ht', htc, htail⟩ := race_tail_choice hlam0 hcr.le hpos
  have hsum : lam + 2 * cross ≤ cross * Wm := by
    have e : lam + 2 * cross = cross * (16 / 5 * S * Real.log (8 * S * (S + 1) / C.δ) + 3
        + (nQ + 2) * nα * (nQ + 1)) := by rw [hlamd]; ring
    rw [e]
    exact mul_le_mul_of_nonneg_left hWm hcr.le
  set u := Real.sqrt C.δ with hud
  have hu2 : u * u = C.δ := Real.mul_self_sqrt hδ0.le
  have hsq : Real.exp 2 * (lam + 2 * cross) ^ 2 ≤ C.δ / 16 := by
    have e : Real.exp 2 * (u / (4 * Real.exp 1)) ^ 2 = C.δ / 16 := by
      rw [show (2 : ℝ) = 1 + 1 by norm_num, Real.exp_add, ← hu2]; field_simp; ring
    calc Real.exp 2 * (lam + 2 * cross) ^ 2 ≤ Real.exp 2 * (u / (4 * Real.exp 1)) ^ 2 := by
          gcongr; linarith
      _ = C.δ / 16 := e
  refine ⟨1 / (lam + 2 * cross), 2 * (1 / (lam + 2 * cross)), ht, ht', htc, ?_⟩
  have := htail.trans hsq
  linarith

lemma agrees_mono {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} {K : ℕ} {O : Oracle μ (FreeMonoid α)} {D : Measure (FreeMonoid α)}
    {εr εu εr' εu' : ℝ} (hr : εr ≤ εr') (hu : εu ≤ εu') {o : LoopRound Ω α K}
    (h : o.Agrees O D εr εu) : o.Agrees O D εr' εu' :=
  ⟨h.1, h.2.1.trans hr, h.2.2.trans hu⟩

lemma checks_mono {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ} {M : _root_.DFA α σ} {O : Oracle μ (FreeMonoid α)}
    {D : Measure (FreeMonoid α)} {k : ℕ} {εr εu εd εr' εu' : ℝ} (hr : εr ≤ εr')
    (hu : εu ≤ εu') {o : LoopRound Ω α K} (h : o.Checks M O D k εr εu εd) :
    o.Checks M O D k εr' εu' εd :=
  ⟨agrees_mono hr hu h.1, h.2⟩

set_option maxHeartbeats 8000000 in -- the settings and the cap meet `LoopCore`'s conditions
/-- The loop's claim with each round's failure `δ'` free and the last round's disagreeing probes
at most `ε + ε + ε/96`, level 3 running at disagreement rate `ε`; its family misreads at most
`(1025/1024) εcov` and leaves undecided at most `(3/8) εcov` of `P.D`. -/
theorem loop_simple_checks {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*}
    [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (η₀ εcov ε δ' pAP : ℝ) (k : ℕ) :
    let Q : ℝ := Fintype.card σ
    let A : ℝ := Fintype.card α
    let L : ℝ := P.L
    let ι : ℝ := min (εcov / 8) (ε / (4608 * Q * (Q + 1)))
    let s : ℝ := 1 / 2 - η₀
    let τ : ℝ := min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι
    let J : ℝ := Q + 2
    let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
    let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
    let Wm : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
    let W : ℝ := max (Real.exp 1 * Real.sqrt δ' * Wm / 2) ((Q + 1) / ε)
    let PB : ℝ := (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
      + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
    let RB : ℝ := 161000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
      / (s ^ 4 * τ ^ 2)
    let cap : ℝ := min (min (min (τ * s / (2 ^ 19 * (J + 3) * (RB + 1)))
          (δ' * min (εcov / 2) ι ^ 2 / (2 ^ 35 * J * (3 + 2 * Real.logb 2 (RB + 1)))))
        (min (δ' * (ι / 2) ^ 2 / (2 ^ 5 * (J + 1) * (3 + 2 * Real.logb 2 (RB + 1))))
          (ι / 2 / (8 * J * (RB + 1)))))
      (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
        (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1)))))
    let Kc : ℝ := (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ * (Q + 3) ^ ((Fintype.card σ + 2)
      * Fintype.card α)
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
    O.η ≤ η₀ → η₀ < 1 / 2 →
    0 < εcov → εcov ≤ 1 → 0 < ε → ε ≤ 1 / 2 → 0 < δ' → δ' < 1 → 0 < pAP → 1 ≤ P.L → 1 ≤ k →
    (2 + Nat.clog 2 P.L) * ε < 3 / 8 →
    prefixMax P.D k ≤ cap * ε / 1536 →
    1536 * Real.log (16 * (Q + 1) * Kc / δ') * prefixMax P.D k ≤ ε →
    16 * B₃ ^ 2 * prefixMax P.D k ≤ δ' →
    ∃ settings : LoopSettings (Fintype.card σ + 1),
      settings.level3.walkStart = k ∧ settings.level3.disagreementRate = ε ∧
      (settings.level3.probeBudget : ℝ) ≤ B₃ ∧
      (∀ pops, ∑ B ∈ settings.schedule pops, B.npref ≤ 2 * ⌊RB⌋₊) ∧
      ∀ Dsf : Measure (FreeMonoid α), IsProbabilityMeasure Dsf →
        Dsf {v | v.toList.length ≠ P.L + Fintype.card σ + 3} = 0 →
        pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
        (∀ v, Dsf.real {v} ≤ cap) →
        (loopMeasure μ α (Fintype.card σ + 1) settings.level3.probeBudget).real
            {p | let rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p)
              ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
                  o.Checks M O P.D k (1025 / 1024 * εcov) (3 / 8 * εcov) (ε + ε + ε / 96))
                ∧ RoundsFresh settings O P.D Dsf (P.L + settings.level3.probeBudget + 2)
                  (P.L + Fintype.card σ + 3) p)}
          ≤ (Q + 1) * (5 * δ' / 2) := by
  intro Q A L ι s τ J S B₃ Wm W PB RB cap Kc
    hL hη hη₀ hε hε1 hεd hεd12 hδ'0 hδ'1' hpAP hL1 hk0 hLε hk1 hk2 hk3
  have hδ'1 : δ' ≤ 1 := hδ'1'.le
  classical
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hA1 : (1 : ℝ) ≤ A := by
    show (1 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast Fintype.card_pos
  have hL1' : (1 : ℝ) ≤ L := by show (1 : ℝ) ≤ (P.L : ℝ); exact_mod_cast hL1
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hι0 : 0 < ι := by
    show 0 < min (εcov / 8) (ε / (4608 * Q * (Q + 1)))
    exact lt_min (by positivity) (by positivity)
  have hι8 : ι ≤ εcov / 8 := min_le_left _ _
  have hιε : ι ≤ ε / (4608 * Q * (Q + 1)) := min_le_right _ _
  -- level 3's settings
  set S' : ℕ := Fintype.card σ * ((Fintype.card σ + 2) * Fintype.card α + 1)
    + 2 * Fintype.card α + 1 with hS'
  have hSS : (S' : ℝ) = S := by
    show ((Fintype.card σ * ((Fintype.card σ + 2) * Fintype.card α + 1)
      + 2 * Fintype.card α + 1 : ℕ) : ℝ) = _
    push_cast; rfl
  have hS1 : (1 : ℝ) ≤ S := by
    show 1 ≤ Q * ((Q + 2) * A + 1) + 2 * A + 1
    have : 0 ≤ Q * ((Q + 2) * A + 1) := by positivity
    linarith
  have hlogS : 0 ≤ Real.log (16 * S * (S + 1) / δ') :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith)
  set budget : ℕ := ⌈36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S⌉₊ with hbudd
  have hbud : (budget : ℝ) ≤ B₃ := by
    have := Nat.ceil_lt_add_one (show 0 ≤ 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S by
      positivity)
    show (budget : ℝ) ≤ 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
    linarith
  have hbud0 : (0 : ℝ) ≤ budget := Nat.cast_nonneg _
  set C : Random.Level3Settings := ⟨k, δ', ε / 48, ε, budget⟩ with hCd
  -- the band's far side
  have hWr : (Q + 1) / ε ≤ W := le_max_right _ _
  have hW1 : 1 ≤ W := by
    refine le_trans ?_ hWr
    have h3 : Q + 1 ≤ (Q + 1) / ε := by
      rw [le_div_iff₀ hεd]; nlinarith
    linarith
  set cross : ℝ := δ' / (8 * W) with hcrossd
  have hcr : 0 < cross := by positivity
  have hcrW : cross * W = δ' / 8 := by rw [hcrossd]; field_simp
  set cLog : ℝ := 4 * Real.log (8 * W / δ') with hcLog
  have hcL0 : 0 ≤ cLog := mul_nonneg (by norm_num) (Real.log_nonneg (by
    rw [le_div_iff₀ hδ'0]; linarith))
  have hcexp : cross ^ 4 = Real.exp (-cLog) := by
    have e : -cLog = Real.log (cross ^ 4) := by
      rw [hcLog, Real.log_pow, hcrossd, Real.log_div hδ'0.ne' (by positivity), Real.log_div
        (by positivity) hδ'0.ne']
      push_cast; ring
    rw [e, Real.exp_log (by positivity)]
  by_cases hpAP1 : pAP ≤ 1
  swap
  · -- no suffix distribution finds that many absorbing suffixes
    refine ⟨⟨fun _ => ∅, η₀, ι, εcov / 2, δ' / 4, C⟩, rfl, rfl, hbud, fun _ => by simp, ?_⟩
    intro Dsf hDsf _ hpAPD _
    have := measureReal_le_one (μ := Dsf) (s := {v : FreeMonoid α | ∀ p, p * v ∈ O.L ↔ p ∈ O.L})
    linarith
  have hτb : budgetScale η₀ ι (εcov / 2) = τ := by
    show min (min (εcov / 2) (1 / 2) * gateA η₀ ^ 2) ι
      = min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι
    rw [min_eq_left (by linarith : εcov / 2 ≤ 1 / 2)]
    rfl
  have hPBe : capPB η₀ ι (εcov / 2) δ' (pAP / 2) cLog J = PB := by
    show _ = (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / (1 / 2 - η₀) ^ 2 + 20 / (1 / 2 - η₀) + 28) / pAP
      + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
    unfold capPB
    rw [hτb, hcLog, ← mul_max_of_nonneg _ _ (by norm_num : (0 : ℝ) ≤ 4)]
    field_simp
    ring
  have hRBe : capRB η₀ ι (εcov / 2) (δ' / 4) δ' (pAP / 2) cLog J = RB := by
    unfold capRB
    rw [hPBe, hτb]
  have hJ : (((Fintype.card σ + 1 + 1 : ℕ)) : ℝ) = J := by
    push_cast; show _ = (Fintype.card σ : ℝ) + 2; ring
  have hcounts : ∀ pops : Finset (Option (Fin (Fintype.card σ + 1))), none ∈ pops →
      (poolCount η₀ pops ι (εcov / 2) (δ' / 2) (pAP / 2) (Real.exp (-cLog)) : ℝ) ≤ PB
        ∧ (prefCount η₀ pops ι (εcov / 2) (δ' / 2) (δ' / 4) (pAP / 2) (Real.exp (-cLog)) 0
          : ℝ)
          ≤ RB := by
    intro pops hpops
    have hcard : (pops.card : ℝ) ≤ J := by
      have h1 := Finset.card_le_univ pops
      simp only [Fintype.card_option, Fintype.card_fin] at h1
      rw [← hJ]
      exact_mod_cast h1
    have h := counts_le η₀ pops ι (εcov / 2) (δ' / 4) δ' (pAP / 2) cLog J hη0 hη₀ ⟨none, hpops⟩
      hcard hι0 (by positivity) hδ'0 hδ'1 (by positivity) (by linarith)
      (by positivity) (by linarith) hcL0
    rw [hPBe, hRBe] at h
    exact h
  obtain ⟨hPB0', hRB0'⟩ := hcounts {none} (Finset.mem_singleton_self _)
  have hPB0 : 0 ≤ PB := (Nat.cast_nonneg _).trans hPB0'
  have hRB0 : 0 ≤ RB := (Nat.cast_nonneg _).trans hRB0'
  have hJ0 : 0 < J := by show 0 < Q + 2; linarith
  have hτ0 : 0 < τ := by
    have ha : 0 < (1 - 2 * η₀) / (1 - η₀) := gateA_pos hη0 hη₀
    exact lt_min (mul_pos (by positivity) (pow_pos ha 2)) hι0
  have hlgRB : 0 < 3 + 2 * Real.logb 2 (RB + 1) := by
    have : 0 ≤ Real.logb 2 (RB + 1) := Real.logb_nonneg (by norm_num) (by linarith)
    linarith
  have hmn0 : 0 < min (εcov / 2) ι := lt_min (by positivity) hι0
  have hcap0 : 0 < cap := by
    show 0 < min (min (min (τ * s / (2 ^ 19 * (J + 3) * (RB + 1)))
          (δ' * min (εcov / 2) ι ^ 2 / (2 ^ 35 * J * (3 + 2 * Real.logb 2 (RB + 1)))))
        (min (δ' * (ι / 2) ^ 2 / (2 ^ 5 * (J + 1) * (3 + 2 * Real.logb 2 (RB + 1))))
          (ι / 2 / (8 * J * (RB + 1)))))
      (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
        (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1)))))
    have hs0 : 0 < s := by show 0 < 1 / 2 - η₀; linarith
    refine lt_min (lt_min (lt_min (by positivity) (div_pos (by positivity) (by positivity)))
      (lt_min (div_pos (by positivity) (by positivity)) (by positivity))) (by positivity)
  -- the cap's terms
  have hmul : ∀ {x b a : ℝ}, 0 < b → x ≤ b → cap ≤ a / b → x * cap ≤ a := fun hb hx h => by
    rw [le_div_iff₀ hb] at h
    exact (mul_le_mul_of_nonneg_right hx hcap0.le).trans (by linarith)
  have hNSr : ((⌊PB⌋₊ : ℕ) : ℝ) ≤ PB := Nat.floor_le hPB0
  have hNPr : ((⌊RB⌋₊ : ℕ) : ℝ) ≤ RB := Nat.floor_le hRB0
  have hK2 : (((Fintype.card σ + 1 : ℕ) : ℝ)) + 2 = J + 1 := by
    push_cast; show _ = (Fintype.card σ : ℝ) + 2 + 1; ring
  have hK1 : (((Fintype.card σ + 1 : ℕ) : ℝ)) + 1 = J := by
    push_cast; show _ = (Fintype.card σ : ℝ) + 2; ring
  have hKJ : (((Fintype.card σ + 1 : ℕ) : ℝ)) ≤ J := by linarith
  have hK0 : (0 : ℝ) ≤ ((Fintype.card σ + 1 : ℕ) : ℝ) := Nat.cast_nonneg _
  -- `⌊log₂ ⌊RB⌋⌋ ≤ log₂ (RB + 1)`
  have hnlog : ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ) ≤ Real.logb 2 (RB + 1) := by
    rcases Nat.eq_zero_or_pos ⌊RB⌋₊ with h0 | hpos
    · rw [h0]; simp only [Nat.log_zero_right, Nat.cast_zero]
      exact Real.logb_nonneg (by norm_num) (by linarith)
    · have h1 := Real.natLog_le_logb ⌊RB⌋₊ 2
      push_cast at h1
      refine h1.trans (Real.logb_le_logb_of_le (by norm_num) (by exact_mod_cast hpos) ?_)
      linarith
  have hq1a : 2 ^ 19 * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 4) * ((⌊RB⌋₊ : ℕ) : ℝ) * cap
      ≤ rateScale η₀ ι (εcov / 2) * (1 / 2 - η₀) := by
    have hK4 : (((Fintype.card σ + 1 : ℕ) : ℝ)) + 4 = J + 3 := by
      push_cast; show _ = (Fintype.card σ : ℝ) + 2 + 3; ring
    rw [hK4]
    change _ ≤ τ * s
    exact hmul (b := 2 ^ 19 * (J + 3) * (RB + 1)) (by positivity) (by gcongr; linarith)
      (((min_le_left _ _).trans (min_le_left _ _)).trans (min_le_left _ _))
  have hq1b : 2 ^ 35 * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 1)
      * (3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ)) * cap ≤ δ' * min (εcov / 2) ι ^ 2 := by
    rw [hK1]
    exact hmul (b := 2 ^ 35 * J * (3 + 2 * Real.logb 2 (RB + 1))) (by positivity)
      (by gcongr)
      (((min_le_left _ _).trans (min_le_left _ _)).trans (min_le_right _ _))
  have hq2a : 2 ^ 5 * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 2)
      * (3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ)) * cap ≤ δ' * (ι - ι / 2) ^ 2 := by
    rw [hK2, show ι - ι / 2 = ι / 2 by ring]
    exact hmul (b := 2 ^ 5 * (J + 1) * (3 + 2 * Real.logb 2 (RB + 1))) (by positivity)
      (by gcongr)
      (((min_le_left _ _).trans (min_le_right _ _)).trans (min_le_left _ _))
  have hq2b : 8 * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 1) * ((⌊RB⌋₊ : ℕ) : ℝ) * cap
      ≤ ι - ι / 2 := by
    rw [hK1, show ι - ι / 2 = ι / 2 by ring]
    exact hmul (b := 8 * J * (RB + 1)) (by positivity) (by gcongr; linarith)
      (((min_le_left _ _).trans (min_le_right _ _)).trans (min_le_right _ _))
  have hq3 : 2 ^ 14 * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 2) * ((⌊PB⌋₊ : ℕ) : ℝ) * cap
      ≤ δ' * (pAP - pAP / 2) := by
    rw [hK2]
    have h := hmul (x := 2 ^ 15 * (J + 1) * ((⌊PB⌋₊ : ℕ) : ℝ)) (a := δ' * pAP)
      (b := 2 ^ 15 * (J + 1) * (PB + 1)) (by positivity) (by gcongr; linarith)
      (((min_le_right _ _).trans (min_le_left _ _)).trans (min_le_left _ _))
    linarith only [h]
  have hqm : 6 * (((Fintype.card σ + 1 : ℕ) : ℝ)) * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 1)
      * ((⌊RB⌋₊ : ℕ) : ℝ) * cap ≤ ι / 2 := by
    rw [hK1]
    have h := hmul (x := 12 * (((Fintype.card σ + 1 : ℕ) : ℝ)) * J * ((⌊RB⌋₊ : ℕ) : ℝ)) (a := ι)
      (b := 12 * J ^ 2 * (RB + 1)) (by positivity)
      (by
        calc 12 * (((Fintype.card σ + 1 : ℕ) : ℝ)) * J * ((⌊RB⌋₊ : ℕ) : ℝ)
            ≤ 12 * J * J * (RB + 1) := by gcongr; linarith
          _ = 12 * J ^ 2 * (RB + 1) := by ring)
      (((min_le_right _ _).trans (min_le_left _ _)).trans (min_le_right _ _))
    linarith only [h]
  have hqms : (((Fintype.card σ + 1 : ℕ) : ℝ)) * (((⌊PB⌋₊ : ℕ) : ℝ) + 1) * cap ≤ pAP / 2 := by
    have h := hmul (x := 2 * (((Fintype.card σ + 1 : ℕ) : ℝ)) * (((⌊PB⌋₊ : ℕ) : ℝ) + 1))
      (a := pAP) (b := 2 * J * (PB + 2)) (by positivity) (by gcongr <;> linarith)
      (((min_le_right _ _).trans (min_le_right _ _)).trans (min_le_left _ _))
    linarith only [h]
  have hcapP : 6 * J * (RB + 1) * cap ≤ 1 :=
    hmul (b := 6 * J * (RB + 1)) (by positivity) le_rfl
      (((min_le_right _ _).trans (min_le_right _ _)).trans (min_le_right _ _))
  -- level 3's premises
  have hpm0 : 0 ≤ prefixMax P.D k := (prefixMax_pos P.D k).le
  have hKc : (((1 + (Fintype.card σ + 1) ^ 3 * Fintype.card α) ^ Fintype.card σ
      * (Fintype.card σ + 3) ^ ((Fintype.card σ + 2) * Fintype.card α) : ℕ) : ℝ) = Kc := by
    push_cast; rfl
  -- level 5's race tail, at the band's `κ`
  have hcr1 : cross ≤ 1 := by
    rw [hcrossd, div_le_one (by positivity)]; linarith
  set u := Real.sqrt δ' with hud
  have hu0 : 0 < u := Real.sqrt_pos.2 hδ'0
  have hu2 : u * u = δ' := Real.mul_self_sqrt hδ'0.le
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have hcm : cross * Wm ≤ u / (4 * Real.exp 1) := by
    have hV : Real.exp 1 * u * Wm / 2 ≤ W := le_max_left _ _
    rw [hcrossd, div_mul_eq_mul_div, div_le_div_iff₀ (by positivity) (by positivity)]
    calc δ' * Wm * (4 * Real.exp 1) = 8 * u * (Real.exp 1 * u * Wm / 2) := by rw [← hu2]; ring
      _ ≤ 8 * u * W := by gcongr
      _ = u * (8 * W) := by ring
  have hprem : RoundPremises σ α C cross (2 * prefixMax P.D k) := by
    refine roundPremises_of C cross (2 * prefixMax P.D k) Wm hδ'0 hδ'1'
      (by show 0 < ε / 48; positivity) (le_refl _) (by show ε < 1; linarith) ?_ hcr ?_ ?_ hcm
    · show 36 * (S' : ℝ) / ε * Real.log (16 * S' * (S' + 1) / δ') + S' ≤ (budget : ℝ)
      rw [hSS]
      exact Nat.le_ceil _
    · show (budget : ℝ) * ((budget : ℝ) - 1) / 2 * (2 * prefixMax P.D k) ≤ δ' / 16
      have hB0 : 0 ≤ B₃ := hbud0.trans hbud
      have h1 : (budget : ℝ) * ((budget : ℝ) - 1) ≤ B₃ ^ 2 := by
        nlinarith only [hbud, hbud0, hB0]
      have h2 := mul_le_mul_of_nonneg_right h1 hpm0
      linarith only [h2, hk3]
    · show 16 / 5 * (S' : ℝ) * Real.log (8 * S' * (S' + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
        ≤ 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
      rw [hSS]
  -- the probes' avoided mass
  have hmr : 6 * (((Fintype.card σ + 1 : ℕ) : ℝ) + 1) * ((⌊RB⌋₊ : ℕ) : ℝ) * prefixMax P.D k
      ≤ ε / 1536 := by
    rw [hK1]
    calc 6 * J * ((⌊RB⌋₊ : ℕ) : ℝ) * prefixMax P.D k
        ≤ 6 * J * (RB + 1) * (cap * ε / 1536) :=
          mul_le_mul (mul_le_mul_of_nonneg_left (by linarith) (by positivity)) hk1 hpm0
            (by positivity)
      _ = 6 * J * (RB + 1) * cap * (ε / 1536) := by ring
      _ ≤ 1 * (ε / 1536) := mul_le_mul_of_nonneg_right hcapP (by positivity)
      _ = ε / 1536 := one_mul _
  -- the harvests' threshold `Tu`: `A·Tu = 7/16` and `|Q| · 3ι / Tu < 7/16`, the slack and the
  -- avoided mass at most `1/16` each
  set Aₛ := roundSlope σ P.L C with hAₛ
  have hAeq : Aₛ = (Q + 1) * (288 / ε + 16 * (2 + Nat.clog 2 P.L)) := by
    show ((Fintype.card σ : ℝ) + 1) * (6 / (ε / 48) + 16 * (2 + (Nat.clog 2 P.L : ℝ))) = _
    rw [show (6 : ℝ) / (ε / 48) = 288 / ε by field_simp; ring]
  have hAₛ0 : 0 < Aₛ := by rw [hAeq]; positivity
  set Tu : ℝ := 7 / (16 * Aₛ) with hTu
  have hTu0 : 0 < Tu := by positivity
  have hsl : roundSlack σ α C (2 * prefixMax P.D k) ≤ 1 / 16 := by
    have e : roundSlack σ α C (2 * prefixMax P.D k)
        = 96 * (Real.log (16 * (Q + 1) * Kc / δ') * prefixMax P.D k) / ε := by
      unfold roundSlack
      rw [hKc]
      show 2 * prefixMax P.D k * Real.log (16 * (Q + 1) * Kc / δ') / (ε / 48) = _
      field_simp
      ring
    rw [e, div_le_iff₀ hεd]
    linarith
  have hmrq : ε / 1536 / (C.harvestRate / 2) = 1 / 16 := by
    show ε / 1536 / (ε / 48 / 2) = 1 / 16
    field_simp
    ring
  have hAQι : Q * ι * Aₛ < 49 / 768 := by
    have hQQ : 0 < Q * (Q + 1) := by positivity
    calc Q * ι * Aₛ ≤ Q * (ε / (4608 * Q * (Q + 1))) * Aₛ := by gcongr
      _ = (288 + 16 * (2 + Nat.clog 2 P.L) * ε) / 4608 := by
          rw [hAeq]
          field_simp
      _ < 49 / 768 := by linarith
  have hcond : Aₛ * Tu + roundSlack σ α C (2 * prefixMax P.D k) + ε / 1536 / (C.harvestRate / 2)
      + Fintype.card σ * (2 * ι + ι) / Tu < 1 := by
    have e1 : Aₛ * Tu = 7 / 16 := by rw [hTu]; field_simp
    have e2 : (Fintype.card σ : ℝ) * (2 * ι + ι) / Tu = 48 * (Q * ι * Aₛ) / 7 := by
      rw [hTu]
      show Q * (2 * ι + ι) / (7 / (16 * Aₛ)) = _
      field_simp
      ring
    rw [e1, e2, hmrq]
    linarith
  -- the rarer reads past a midfix: probes sharing a probe's first `k` letters, and the rest
  have hshort : ((budget : ℝ) * (2 * prefixMax P.D k) + (Q + 1) * cross) / ε
      ≤ δ' / 4 := by
    have h1 : (Q + 1) * cross / ε ≤ δ' / 8 :=
      calc (Q + 1) * cross / ε = cross * ((Q + 1) / ε) := by ring
        _ ≤ cross * W := mul_le_mul_of_nonneg_left hWr hcr.le
        _ = δ' / 8 := hcrW
    have hB₃ε : 1 ≤ B₃ * ε := by
      set lg := Real.log (16 * S * (S + 1) / δ')
      have hx : (32 : ℝ) ≤ 16 * S * (S + 1) / δ' := by rw [le_div_iff₀ hδ'0]; nlinarith
      have hl := Real.one_sub_inv_le_log_of_pos (show 0 < 16 * S * (S + 1) / δ' by linarith)
      have hinv : (16 * S * (S + 1) / δ')⁻¹ ≤ 1 / 32 := by
        rw [inv_eq_one_div]; exact one_div_le_one_div_of_le (by norm_num) hx
      have e : B₃ * ε = 36 * S * lg + (S + 1) * ε := by
        show (36 * S / ε * lg + S + 1) * ε = _
        field_simp
        ring
      have hSl : lg ≤ S * lg := le_mul_of_one_le_left hlogS hS1
      nlinarith
    have h2 : (budget : ℝ) * (2 * prefixMax P.D k) / ε ≤ δ' / 8 := by
      rw [div_le_iff₀ hεd]
      have hB0 : 0 ≤ B₃ := hbud0.trans hbud
      have e1 : (budget : ℝ) * prefixMax P.D k ≤ B₃ * prefixMax P.D k :=
        mul_le_mul_of_nonneg_right hbud hpm0
      have e2 : B₃ * prefixMax P.D k ≤ B₃ * prefixMax P.D k * (B₃ * ε) :=
        le_mul_of_one_le_right (by positivity) hB₃ε
      have e3 : 16 * B₃ ^ 2 * prefixMax P.D k * ε ≤ δ' * ε :=
        mul_le_mul_of_nonneg_right hk3 hεd.le
      have e4 : B₃ * prefixMax P.D k * (B₃ * ε) = B₃ ^ 2 * prefixMax P.D k * ε := by ring
      linarith
    rw [add_div]
    linarith
  have hcore := loop_core_holds M O P (Fintype.card σ + 1) (P.L + Fintype.card σ + 3)
    ⌊PB⌋₊ ⌊RB⌋₊ η₀ ι (δ' / 4) (εcov / 2) δ' ι pAP cap (ι / 2) (pAP / 2) (ε / 1536) ε cross Tu C
  dsimp only at hcore
  rw [hcexp] at hcore
  obtain ⟨settings, -, -, -, -, hC3, hnp, hmain⟩ := hcore hL hη hη₀ hι0
    (by linarith) (by positivity) (by linarith) (by positivity) (by linarith) hδ'0 hδ'1 hι0 hpAP
    hL1 (by omega) (by omega) hcap0 (by linarith) (by linarith) (by linarith) (by linarith)
    (fun pops hpops => by
      rw [show pAP - pAP / 2 = pAP / 2 by ring]
      exact Nat.le_floor (hcounts pops hpops).1)
    (fun pops hpops => by
      rw [show pAP - pAP / 2 = pAP / 2 by ring]
      exact Nat.le_floor (hcounts pops hpops).2)
    hq1a hq1b hq2a hq2b hq3 hqm hqms hk0
    (by show prefixMax P.D k ≤ cap * (ε / 48) / 2; linarith [hk1])
    hεd (by positivity) hmr (by linarith) hcr hprem hTu0 hcond
  refine ⟨settings, by rw [hC3], by rw [hC3], by rw [hC3]; exact hbud, hnp, ?_⟩
  intro Dsf hDsf hlen hpAPD hatom
  have h := hmain Dsf hDsf (by simpa using hlen) hpAPD hatom
  refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) (h.trans ?_)
  · intro draws hd hgood
    apply hd
    obtain ⟨⟨h1, o, ho, hag⟩, hfr⟩ := hgood
    have hag' : o.Checks M O P.D k (1025 / 1024 * εcov) (3 / 8 * εcov) (ε + ε + ε / 1536) :=
      checks_mono (by linarith) (by linarith) hag
    exact ⟨⟨h1, o, ho, ⟨hag'.1, hag'.2.1.trans (by linarith), hag'.2.2⟩⟩, hfr⟩
  · have hδ8 := hshort
    have hQ0 : (0 : ℝ) ≤ (Fintype.card σ : ℝ) + 1 := by positivity
    show ((Fintype.card σ : ℝ) + 1) * (δ' + δ' / 4 + (δ'
      + ((budget : ℝ) * (2 * prefixMax P.D k) + (Q + 1) * cross) / ε))
        ≤ (Q + 1) * (5 * δ' / 2)
    have : δ' + δ' / 4 + (δ'
        + ((budget : ℝ) * (2 * prefixMax P.D k) + (Q + 1) * cross) / ε)
          ≤ 5 * δ' / 2 := by
      linarith
    exact mul_le_mul_of_nonneg_left this hQ0

end OrthoDFA
