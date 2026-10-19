import OrthoDFA.Proofs.LearnerLoop
import OrthoDFA.Proofs.Budget

/-!
# The loop's claim at explicit settings

`capPB` and `capRB` bound the suffix pool and the prefix count over at most `Jm` populations, by
`poolCount_le` and `prefCount_le_poly` (`counts_le`). `loop_simple_checks` takes `LoopCore` at
`K = |Q| + 1`, with the failure split evenly between the clustering's and level 5's `|Q| + 1`
rounds.
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory

/-- A bound on the suffix pool, the clustering's `crossLimit` at `exp (-cLog)`. -/
def capPB (η₀ ι εcov δ pAP cLog Jm : ℝ) : ℝ :=
  128 * (Real.log (2 / budgetScale η₀ ι εcov) + cLog) / ((1 / 2 - η₀) ^ 2 * pAP)
    + 16 * Real.log ((Jm + 2) / (δ / 2)) / pAP ^ 2

/-- A bound on the prefix count. -/
def capRB (η₀ ι εcov α δ pAP cLog Jm : ℝ) : ℝ :=
  60000 * Jm ^ 2 * Real.log ((Jm + 2) * capPB η₀ ι εcov δ pAP cLog Jm
    / (δ / 2 * α * pAP * budgetScale η₀ ι εcov))
    / ((1 / 2 - η₀) ^ 4 * budgetScale η₀ ι εcov ^ 2)

theorem counts_le {J : Type*} [Fintype J] (η₀ : ℝ) (pops : Finset J)
    (ι εcov α δ pAP cLog Jm : ℝ) (hη0 : 0 ≤ η₀) (hη : η₀ < 1 / 2)
    (hpop : pops.Nonempty) (hcard : (pops.card : ℝ) ≤ Jm) (hι : 0 < ι) (hε : 0 < εcov)
    (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hα : 0 < α) (hα1 : α < 1) (hpAP : 0 < pAP)
    (hpAP1 : pAP ≤ 1) (hc : 0 ≤ cLog) :
    (poolCount η₀ pops ι εcov (δ / 2) pAP (Real.exp (-cLog)) : ℝ)
        ≤ capPB η₀ ι εcov δ pAP cLog Jm
      ∧ (prefCount η₀ pops ι εcov (δ / 2) α pAP (Real.exp (-cLog)) : ℝ)
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
  set pref := (prefCount η₀ pops ι εcov (δ / 2) α pAP (Real.exp (-cLog)) : ℝ) with hpref
  -- the pool
  have hpoolB : pool ≤ PB := by
    have h := poolCount_le pops η₀ ι εcov (δ / 2) pAP (Real.exp (-cLog)) hsig hη0 hι hε hpop hδ2
      hδ21 hpAP hpAP1 hcross hcross1
    rw [hsig', hτ'] at h
    have e : Real.log (2 / (τ * Real.exp (-cLog))) = Real.log (2 / τ) + cLog := by
      rw [div_mul_eq_div_div, Real.log_div (by positivity) hcross.ne', Real.log_exp]; ring
    rw [e] at h
    refine h.trans (add_le_add le_rfl ?_)
    refine div_le_div_of_nonneg_right ?_ (by positivity)
    refine mul_le_mul_of_nonneg_left (Real.log_le_log (by positivity) ?_) (by norm_num)
    exact div_le_div_of_nonneg_right (by linarith) hδ2.le
  -- the prefix count
  have hlogarg : 1 ≤ ((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ) := by
    have hpool256 : (256 : ℝ) ≤ pool :=
      poolCount_ge pops η₀ ι εcov (δ / 2) pAP _ hsig hη0 hι hε hε1 hpAP hpAP1
    rw [le_div_iff₀ (by positivity)]
    have h1 : δ / 2 * α * pAP * τ ≤ 1 := by
      have : δ / 2 * α ≤ 1 := by nlinarith
      have : δ / 2 * α * pAP ≤ 1 := by nlinarith
      nlinarith
    nlinarith
  refine ⟨hpoolB, ?_⟩
  have h := prefCount_le_poly pops η₀ ι εcov (δ / 2) α pAP (Real.exp (-cLog)) hsig hη0 hpop hι
    hε hε1 hδ2 hδ21 hα hα1 hpAP hpAP1
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
  calc 60000 * (pops.card : ℝ) ^ 2
        * Real.log (((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ))
      ≤ 60000 * Jm ^ 2 * Real.log (((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ)) := by
        gcongr
    _ ≤ 60000 * Jm ^ 2 * Real.log ((Jm + 2) * PB / (δ / 2 * α * pAP * τ)) := by
        gcongr

end OrthoDFA

namespace OrthoDFA

open MeasureTheory

open Random (prefixMax le_prefixMax prefixMax_pos)

/-- The strings of at most `n` letters number at most `(n + 1) |Σ|^n`. -/
lemma card_shortStrings_le {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] (n : ℕ) :
    ((shortStrings (α := α) n).card : ℝ) ≤ (n + 1) * (Fintype.card α : ℝ) ^ n := by
  classical
  have hA : (1 : ℝ) ≤ Fintype.card α := by exact_mod_cast Fintype.card_pos
  have h1 : (shortStrings (α := α) n).card ≤ ∑ i ∈ Finset.range (n + 1), Fintype.card α ^ i := by
    unfold shortStrings
    refine Finset.card_biUnion_le.trans (Finset.sum_le_sum fun i _ => ?_)
    refine Finset.card_image_le.trans ?_
    simp
  calc ((shortStrings (α := α) n).card : ℝ)
      ≤ ((∑ i ∈ Finset.range (n + 1), Fintype.card α ^ i : ℕ) : ℝ) := by exact_mod_cast h1
    _ = ∑ i ∈ Finset.range (n + 1), (Fintype.card α : ℝ) ^ i := by push_cast; rfl
    _ ≤ ∑ _i ∈ Finset.range (n + 1), (Fintype.card α : ℝ) ^ n :=
        Finset.sum_le_sum fun i hi => pow_le_pow_right₀ hA (by
          have := Finset.mem_range.1 hi; omega)
    _ = (n + 1) * (Fintype.card α : ℝ) ^ n := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]; push_cast; ring

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

set_option maxHeartbeats 2000000 in -- the settings and the cap meet `LoopCore`'s conditions
/-- The loop's claim with each round's failure `δ'` free and the last round's disagreeing probes
at most `ε + ε + ε/96`, level 3 running at disagreement rate `ε`. -/
theorem loop_simple_checks {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*}
    [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (η₀ εcov ε δ' pAP : ℝ) (k : ℕ) :
    let Q : ℝ := Fintype.card σ
    let A : ℝ := Fintype.card α
    let L : ℝ := P.L
    let ι : ℝ := min (εcov / 8) (ε / (13824 * Q * (Q + 1)))
    let s : ℝ := 1 / 2 - η₀
    let τ : ℝ := min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι
    let J : ℝ := Q + 2
    let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
    let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
    let W : ℝ := B₃ * (L + 1) * Q + Q * (Q + 1) ^ 2 * A + (Q + 1) * A ^ Fintype.card σ / ε
    let PB : ℝ := 256 * (Real.log (2 / τ) + 4 * Real.log (8 * W / δ')) / (s ^ 2 * pAP)
      + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
    let RB : ℝ := 60000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
      / (s ^ 4 * τ ^ 2)
    let cap : ℝ := min (min (δ' * τ ^ 2 / (2 ^ 22 * (J + 1) ^ 2 * (RB + 1)))
        (δ' * ι ^ 2 / (2 ^ 11 * (J + 1) * (RB + 1))))
      (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
        (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1)))))
    let Kc : ℝ := (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ * (Q + 3) ^ ((Fintype.card σ + 2)
      * Fintype.card α)
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
    O.η ≤ η₀ → η₀ < 1 / 2 →
    0 < εcov → εcov ≤ 1 → 0 < ε → ε ≤ 1 / 2 → 0 < δ' → δ' ≤ 1 → 0 < pAP → 1 ≤ P.L → 1 ≤ k →
    384 * (L + 1) * (Q + 1) * Q * ι ≤ 1 →
    prefixMax P.D k ≤ cap * ε / 96 →
    768 * Real.log (16 * Kc / δ') * prefixMax P.D k ≤ ε →
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
                  o.Checks M O P.D k εcov εcov (ε + ε + ε / 96))
                ∧ RoundsFresh settings O P.D Dsf (P.L + settings.level3.probeBudget + 2)
                  (P.L + Fintype.card σ + 3) p)}
          ≤ (Q + 1) * (5 * δ' / 2) := by
  sorry

end OrthoDFA
