import OrthoDFA.Proofs.LearnerLoop
import OrthoDFA.Proofs.Budget

/-!
# The loop's claim at the theorem's own settings

`capPB` and `capRB` bound the suffix pool and the prefix count over at most `Jm` populations, by
`poolCount_le` and `prefCount_le_poly` (`counts_le`). `learner_loop_simple_holds` takes `LoopCore`
at `K = |Q| + 1`, suffixes of length `L + |Q| + 3`, the gate at `1/4`, half of `εcov` for
the clustering, `indecisionLimit = slack = ι`, half of `ι` and of `pAP` for what the earlier
rounds read, and the failure split evenly between the clustering's and level 5's `|Q| + 1`
rounds.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

/-- A bound on the suffix pool, the band's far side at `exp (-cLog)`. -/
def capPB (η₀ ι εcov δ pAP cLog Jm : ℝ) : ℝ :=
  128 * (Real.log (2 / min εcov (min (1 / 2 - η₀) ι)) + cLog) / ((1 / 2 - η₀) ^ 2 * pAP)
    + 16 * Real.log ((Jm + 2) / (δ / 2)) / pAP ^ 2

/-- A bound on the prefix count. -/
def capRB (η₀ ι εcov α δ pAP cLog Jm : ℝ) : ℝ :=
  6000 * Jm ^ 2 * Real.log ((Jm + 2) * capPB η₀ ι εcov δ pAP cLog Jm
    / (δ / 2 * α * pAP * min εcov (min (1 / 2 - η₀) ι)))
    / ((1 / 2 - η₀) ^ 4 * min εcov (min (1 / 2 - η₀) ι) ^ 3)

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
  set τ := min εcov (min s ι) with hτ
  have hτ' : budgetScale η₀ ι εcov = τ := rfl
  have hτ0 : 0 < τ := lt_min hε (lt_min hs0 hι)
  have hτ1 : τ ≤ 1 := (min_le_left _ _).trans hε1
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
  calc 6000 * (pops.card : ℝ) ^ 2
        * Real.log (((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ))
      ≤ 6000 * Jm ^ 2 * Real.log (((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ)) := by
        gcongr
    _ ≤ 6000 * Jm ^ 2 * Real.log ((Jm + 2) * PB / (δ / 2 * α * pAP * τ)) := by
        gcongr

end OrthoDFA

namespace OrthoDFA

open MeasureTheory

lemma agrees_mono {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} {K : ℕ} {O : Oracle μ (FreeMonoid α)} {D : Measure (FreeMonoid α)} {k : ℕ}
    {εr εu εd εr' εu' : ℝ} (hr : εr ≤ εr') (hu : εu ≤ εu') {o : LoopRound Ω α K}
    (h : o.Agrees O D k εr εu εd) : o.Agrees O D k εr' εu' εd :=
  ⟨h.1, h.2.1.trans hr, h.2.2.1.trans hu, h.2.2.2⟩

lemma checks_mono {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ} {M : _root_.DFA α σ} {O : Oracle μ (FreeMonoid α)}
    {D : Measure (FreeMonoid α)} {k J : ℕ} {ū εr εu εd εr' εu' : ℝ} (hr : εr ≤ εr') (hu : εu ≤ εu')
    {o : LoopRound Ω α K} (h : o.Checks M O D k J ū εr εu εd) : o.Checks M O D k J ū εr' εu' εd :=
  ⟨agrees_mono hr hu h.1, h.2⟩

set_option maxHeartbeats 2000000 in -- the settings and the cap meet `LoopCore`'s conditions
/-- `LearnerLoopSimple` with each round's failure `δ'` free and the last round's checked probes at
most `εd`, at `128ū/εd` more a round. -/
theorem loop_simple_checks {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*}
    [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (η₀ εcov εd δ' pAP : ℝ) (k Jl : ℕ) (ū : ℝ) :
    let Q : ℝ := Fintype.card σ
    let A : ℝ := Fintype.card α
    let L : ℝ := P.L
    let ι : ℝ := min (εcov / 8) (1 / (3072 * (L + 1) * (Q + 1) * Q))
    let ρ : ℝ := 3 * εd / 16
    let s : ℝ := 1 / 2 - η₀
    let τ : ℝ := min (εcov / 2) (min s ι)
    let J : ℝ := Q + 2
    let V : ℝ := (Q + 2) * ((Q + 2) * A + 1)
    let Kp : ℝ := V * (224 / εd + 1)
    let G : ℝ := Real.log (4 * V * (2 * (Q + 2) * A + 4) / δ')
    let PB : ℝ := 256 * (Real.log (2 / τ)
        + 8 * Kp * G * (L + 1) * (Q + 3) * A ^ (Fintype.card σ + 2) / δ') / (s ^ 2 * pAP)
      + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
    let RB : ℝ := 6000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ))
      / (s ^ 4 * τ ^ 3)
    let cap : ℝ := min (min (δ' * τ ^ 2 / (2 ^ 22 * (J + 1) ^ 2 * (RB + 1)))
        (δ' * ι ^ 2 / (2 ^ 11 * (J + 1) * (RB + 1))))
      (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
        (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1) * (L + 1)))))
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
    O.η ≤ η₀ → η₀ < 1 / 2 →
    0 < εcov → εcov ≤ 1 → 0 < εd → εd ≤ 1 → 0 < δ' → δ' ≤ 1 → 0 < pAP → 1 ≤ P.L → 1 ≤ k →
    prefixMax P.D k ≤ cap * ρ / 4 →
    16384 * (L + 1) * (Q + 2) * A * Real.log ((Q + 3) * A / δ') * prefixMax P.D k ≤ εd →
    0 ≤ ū →
    ∃ settings : LoopSettings (Fintype.card σ + 1),
      settings.level5.walkStart = k ∧ settings.level5.disagreementRate = εd / 2 ∧
      (settings.level5.probeBudget : ℝ) ≤ Kp * G ∧
      ∀ Dsf : Measure (FreeMonoid α), IsProbabilityMeasure Dsf →
        Dsf {v | v.toList.length ≠ P.L + Fintype.card σ + 3} = 0 →
        pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
        (∀ v, Dsf.real {v} ≤ cap) →
        (loopMeasure μ α (Fintype.card σ + 1) settings.level5.probeBudget).real
            {p | let rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p)
              ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
                  o.Checks M O P.D k Jl ū εcov εcov εd)}
          ≤ (Q + 1) * (2 * δ' + 128 * ū / εd) := by
  intro Q A L ι ρ s τ J V Kp G PB RB cap
    hL hη hη₀ hε hε1 hεd hεd1 hδ'0 hδ'1 hpAP hL1 hk0 hk1 hk2 hū
  classical
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hA1 : (1 : ℝ) ≤ A := by
    show (1 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast Fintype.card_pos
  have hL1' : (1 : ℝ) ≤ L := by show (1 : ℝ) ≤ (P.L : ℝ); exact_mod_cast hL1
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hι0 : 0 < ι := by
    show 0 < min (εcov / 8) (1 / (3072 * (L + 1) * (Q + 1) * Q))
    exact lt_min (by positivity) (by positivity)
  have hι8 : ι ≤ εcov / 8 := min_le_left _ _
  have hιL : ι ≤ 1 / (3072 * (L + 1) * (Q + 1) * Q) := min_le_right _ _
  have hρ0 : 0 < ρ := by change 0 < 3 * εd / 16; positivity
  have e112 : (112 / (εd / 2) : ℝ) = 224 / εd := by field_simp; ring
  by_cases hpAP1 : pAP ≤ 1
  swap
  · -- no suffix distribution finds that many absorbing suffixes
    obtain ⟨s4, hwalk, hdis, hbud, -, hl4⟩ :=
      level5_guarantee_all M (εd / 2) P.L δ' k hδ'0 hδ'1 (by positivity) (by linarith) hL1
    have := hl4 μ
    clear this hl4
    refine ⟨⟨fun _ => ∅, ι, 1 / 4, s4⟩, hwalk, hdis, ?_, ?_⟩
    · refine hbud.trans (le_of_eq ?_)
      rw [e112]
    · intro Dsf hDsf _ hpAPD _
      have := measureReal_le_one (μ := Dsf) (s := {v : FreeMonoid α | ∀ p, p * v ∈ O.L ↔ p ∈ O.L})
      linarith
  set cLog : ℝ := 8 * Kp * G * (L + 1) * (Q + 3) * A ^ (Fintype.card σ + 2) / δ' with hcLog
  have hV1 : 1 ≤ V := by
    change 1 ≤ (Q + 2) * ((Q + 2) * A + 1)
    have : 0 ≤ (Q + 2) * ((Q + 2) * A) := mul_nonneg (by linarith) (by nlinarith)
    nlinarith
  have hG0 : 0 ≤ G := by
    refine Real.log_nonneg ?_
    rw [le_div_iff₀ hδ'0, one_mul]
    have : (4 : ℝ) ≤ 2 * (Q + 2) * A + 4 := by nlinarith
    nlinarith
  have hcL0 : 0 ≤ cLog := by
    have hK0 : 0 ≤ Kp := by
      change 0 ≤ V * (224 / εd + 1)
      exact mul_nonneg (by linarith) (by positivity)
    have e : cLog = 8 * Kp * G * ((L + 1) * (Q + 3) * A ^ (Fintype.card σ + 2)) / δ' := by
      rw [hcLog]; ring
    rw [e]
    have hA0 : 0 ≤ A := by linarith
    have hQ0 : 0 ≤ Q := by linarith
    have hL0 : 0 ≤ L := by linarith
    exact div_nonneg (mul_nonneg (mul_nonneg (mul_nonneg (by norm_num) hK0) hG0) (by positivity))
      hδ'0.le
  have hPBe : capPB η₀ ι (εcov / 2) δ' (pAP / 2) cLog J = PB := by
    show _ = 256 * (Real.log (2 / min (εcov / 2) (min (1 / 2 - η₀) ι)) + cLog)
        / ((1 / 2 - η₀) ^ 2 * pAP) + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
    unfold capPB
    field_simp
    ring
  have hRBe : capRB η₀ ι (εcov / 2) (1 / 4) δ' (pAP / 2) cLog J = RB := by
    unfold capRB
    rw [hPBe]
  have hJ : (((Fintype.card σ + 1 + 1 : ℕ)) : ℝ) = J := by
    push_cast; show _ = (Fintype.card σ : ℝ) + 2; ring
  have hcounts : ∀ pops : Finset (Option (Fin (Fintype.card σ + 1))), none ∈ pops →
      (poolCount η₀ pops ι (εcov / 2) (δ' / 2) (pAP / 2) (Real.exp (-cLog)) : ℝ) ≤ PB
        ∧ (prefCount η₀ pops ι (εcov / 2) (δ' / 2) (1 / 4) (pAP / 2) (Real.exp (-cLog)) : ℝ)
          ≤ RB := by
    intro pops hpops
    have hcard : (pops.card : ℝ) ≤ J := by
      have h1 := Finset.card_le_univ pops
      simp only [Fintype.card_option, Fintype.card_fin] at h1
      rw [← hJ]
      exact_mod_cast h1
    have h := counts_le η₀ pops ι (εcov / 2) (1 / 4) δ' (pAP / 2) cLog J hη0 hη₀ ⟨none, hpops⟩
      hcard hι0 (by positivity) (by linarith) hδ'0 hδ'1 (by norm_num) (by norm_num)
      (by positivity) (by linarith) hcL0
    rw [hPBe, hRBe] at h
    exact h
  obtain ⟨hPB0', hRB0'⟩ := hcounts {none} (Finset.mem_singleton_self _)
  have hPB0 : 0 ≤ PB := (Nat.cast_nonneg _).trans hPB0'
  have hRB0 : 0 ≤ RB := (Nat.cast_nonneg _).trans hRB0'
  have hJ0 : 0 < J := by show 0 < Q + 2; linarith
  have hτ0 : 0 < τ := by
    show 0 < min (εcov / 2) (min (1 / 2 - η₀) ι)
    exact lt_min (by positivity) (lt_min (by linarith) hι0)
  have hcap0 : 0 < cap := by
    show 0 < min (min (δ' * τ ^ 2 / (2 ^ 22 * (J + 1) ^ 2 * (RB + 1)))
      (δ' * ι ^ 2 / (2 ^ 11 * (J + 1) * (RB + 1))))
      (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
        (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1) * (L + 1)))))
    positivity
  have hθ : 1024 * ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + 1)
      * Fintype.card σ * (2 * ι + ι) ≤ 1 := by
    have hpos : 0 < 3072 * (L + 1) * (Q + 1) * Q := by positivity
    have h := mul_le_mul_of_nonneg_left hιL hpos.le
    rw [mul_one_div_cancel hpos.ne'] at h
    have e : 1024 * ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + 1)
        * Fintype.card σ * (2 * ι + ι) = 3072 * (L + 1) * (Q + 1) * Q * ι := by
      change _ = 3072 * ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + 1) * (Fintype.card σ : ℝ) * ι
      ring
    rw [e]
    exact h
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
  have hq1 : 2 ^ 22 * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 2) ^ 2 * ((⌊RB⌋₊ : ℕ) : ℝ) * cap
      ≤ δ' * min (εcov / 2) (min (1 / 2 - η₀) ι) ^ 2 := by
    rw [hK2]
    exact hmul (b := 2 ^ 22 * (J + 1) ^ 2 * (RB + 1)) (by positivity) (by gcongr; linarith)
      ((min_le_left _ _).trans (min_le_left _ _))
  have hq2 : 2 ^ 9 * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 2) * ((⌊RB⌋₊ : ℕ) : ℝ) * cap
      ≤ δ' * (ι - ι / 2) ^ 2 := by
    rw [hK2]
    have h := hmul (x := 2 ^ 11 * (J + 1) * ((⌊RB⌋₊ : ℕ) : ℝ)) (a := δ' * ι ^ 2)
      (b := 2 ^ 11 * (J + 1) * (RB + 1)) (by positivity) (by gcongr; linarith)
      ((min_le_left _ _).trans (min_le_right _ _))
    have e : δ' * (ι - ι / 2) ^ 2 = δ' * ι ^ 2 / 4 := by ring
    rw [e]
    nlinarith
  have hq3 : 2 ^ 14 * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 2) * ((⌊PB⌋₊ : ℕ) : ℝ) * cap
      ≤ δ' * (pAP - pAP / 2) := by
    rw [hK2]
    have h := hmul (x := 2 ^ 15 * (J + 1) * ((⌊PB⌋₊ : ℕ) : ℝ)) (a := δ' * pAP)
      (b := 2 ^ 15 * (J + 1) * (PB + 1)) (by positivity) (by gcongr; linarith)
      (((min_le_right _ _).trans (min_le_left _ _)).trans (min_le_left _ _))
    nlinarith
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
    nlinarith
  have hqms : (((Fintype.card σ + 1 : ℕ) : ℝ)) * (((⌊PB⌋₊ : ℕ) : ℝ) + 1) * cap ≤ pAP / 2 := by
    have h := hmul (x := 2 * (((Fintype.card σ + 1 : ℕ) : ℝ)) * (((⌊PB⌋₊ : ℕ) : ℝ) + 1))
      (a := pAP) (b := 2 * J * (PB + 2)) (by positivity) (by gcongr <;> linarith)
      (((min_le_right _ _).trans (min_le_right _ _)).trans (min_le_left _ _))
    nlinarith
  have hcapP : 6 * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 1) * ((⌊RB⌋₊ : ℕ) : ℝ) * ((P.L : ℝ) + 1)
      * cap ≤ 1 := by
    rw [hK1]
    exact hmul (b := 6 * J * (RB + 1) * (L + 1)) (by positivity) (by gcongr; linarith)
      (((min_le_right _ _).trans (min_le_right _ _)).trans (min_le_right _ _))
  have hcore := loop_core_holds M O P (Fintype.card σ + 1) (P.L + Fintype.card σ + 3)
    ⌊PB⌋₊ ⌊RB⌋₊ η₀ ι (1 / 4) (εcov / 2) δ' ι pAP εd δ' cap (ι / 2) (pAP / 2) k Jl ū
  dsimp only at hcore
  obtain ⟨settings, -, -, hwalk, hdis, hbud, hmain⟩ := hcore hL hη hη₀ hι0 (by linarith)
    (by norm_num) (by norm_num) (by positivity) (by linarith) hδ'0 hδ'1 hι0 hpAP hεd hεd1 hδ'0
    hδ'1 hL1 (by omega) (by omega) hθ hcap0 (by linarith) (by linarith) (by linarith)
    (by linarith)
    (fun pops hpops => by
      rw [show pAP - pAP / 2 = pAP / 2 by ring]
      exact Nat.le_floor (hcounts pops hpops).1)
    (fun pops hpops => by
      rw [show pAP - pAP / 2 = pAP / 2 by ring]
      exact Nat.le_floor (hcounts pops hpops).2)
    hq1 hq2 hq3 hqm hqms hcapP
    hk0 hk1 (by
      have e2 : 4096 * ((P.L : ℝ) + 1) * unionLog (α := α) σ δ' * prefixMax P.D k
          = 16384 * (L + 1) * (Q + 2) * A * Real.log ((Q + 3) * A / δ') * prefixMax P.D k := by
        unfold unionLog
        change _ = 16384 * ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + 2) * (Fintype.card α : ℝ)
          * Real.log (((Fintype.card σ : ℝ) + 3) * (Fintype.card α : ℝ) / δ') * prefixMax P.D k
        ring
      rw [e2]
      exact hk2) hū
  refine ⟨settings, hwalk, hdis, ?_, ?_⟩
  · exact hbud
  · intro Dsf hDsf hlen hpAPD hatom
    have h := hmain Dsf hDsf (by simpa using hlen) hpAPD hatom
    refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) (h.trans (le_of_eq ?_))
    · intro draws hd hgood
      apply hd
      obtain ⟨h1, o, ho, hag⟩ := hgood
      refine ⟨h1, o, ho, checks_mono ?_ ?_ hag⟩
      · linarith
      · linarith
    · show ((Fintype.card σ : ℝ) + 1) * (δ' + δ' + 128 * ū / εd) = _
      ring

set_option maxHeartbeats 400000 in -- unfolding the claim's settings
theorem learner_loop_simple_holds : LearnerLoopSimple := by
  intro α _ _ _ σ _ Ω _ μ _ M O P η₀ εcov εd δ pAP k Q A L δ' ι ρ s τ J V Kp G PB RB cap
    hL hη hη₀ hε hε1 hεd hεd1 hδ hδ1 hpAP hL1 hk0 hk1 hk2
  have hQ0 : (0 : ℝ) ≤ Q := Nat.cast_nonneg _
  have hδ'0 : 0 < δ' := by show 0 < δ / (2 * (Q + 1)); positivity
  have hδ'1 : δ' ≤ 1 := by
    have : δ / (2 * (Q + 1)) ≤ δ := div_le_self hδ.le (by linarith)
    exact this.trans hδ1
  obtain ⟨settings, h1, h2, h3, h4⟩ := loop_simple_checks M O P η₀ εcov εd δ' pAP k 0 0 hL hη hη₀
    hε hε1 hεd hεd1 hδ'0 hδ'1 hpAP hL1 hk0 hk1 hk2 le_rfl
  refine ⟨settings, h1, h2, h3, fun Dsf hDsf hlen hpAPD hatom => ?_⟩
  refine (measureReal_mono ?_ (measure_ne_top _ _)).trans
    ((h4 Dsf hDsf hlen hpAPD hatom).trans (le_of_eq ?_))
  · intro p hp hgood
    apply hp
    obtain ⟨hl, o, ho, hc⟩ := hgood
    exact ⟨hl, o, ho, hc.1⟩
  · show ((Fintype.card σ : ℝ) + 1) * (2 * (δ / (2 * ((Fintype.card σ : ℝ) + 1))) + 128 * 0 / εd)
      = δ
    field_simp
    ring

end OrthoDFA
