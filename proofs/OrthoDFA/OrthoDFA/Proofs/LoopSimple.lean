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
  64000 * Jm ^ 2 * Real.log ((Jm + 2) * capPB η₀ ι εcov δ pAP cLog Jm
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
    have hpool256 : (256 : ℝ) ≤ pool :=
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
  calc 64000 * (pops.card : ℝ) ^ 2
        * Real.log (((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ))
      ≤ 64000 * Jm ^ 2 * Real.log (((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ)) := by
        gcongr
    _ ≤ 64000 * Jm ^ 2 * Real.log ((Jm + 2) * PB / (δ / 2 * α * pAP * τ)) := by
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
    let W : ℝ := B₃ * (L + 1) * Q + Q * (Q + 1) ^ 2 * A + (Q + 1) * A ^ Fintype.card σ / ε
    let PB : ℝ := (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
      + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
    let RB : ℝ := 64000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
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
    (L + 1) * ε < 3 / 8 →
    prefixMax P.D k ≤ cap * ε / 1536 →
    1536 * Real.log (16 * (Q + 1) * Kc / δ') * prefixMax P.D k ≤ ε →
    ∃ settings : LoopSettings (Fintype.card σ + 1),
      settings.level3.walkStart = k ∧ settings.level3.disagreementRate = ε ∧
      (settings.level3.probeBudget : ℝ) ≤ B₃ ∧ settings.avoidDepth = k ∧
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
  intro Q A L ι s τ J S B₃ W PB RB cap Kc
    hL hη hη₀ hε hε1 hεd hεd12 hδ'0 hδ'1 hpAP hL1 hk0 hLε hk1 hk2
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
  have hW1 : 1 ≤ W := by
    show 1 ≤ B₃ * (L + 1) * Q + Q * (Q + 1) ^ 2 * A + (Q + 1) * A ^ Fintype.card σ / ε
    have h1 : (1 : ℝ) ≤ (Q + 1) ^ 2 := one_le_pow₀ (by linarith)
    have : 1 ≤ Q * (Q + 1) ^ 2 * A :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hQ1 h1) hA1
    have : 0 ≤ B₃ * (L + 1) * Q := by
      have : 0 ≤ B₃ := hbud0.trans hbud
      positivity
    have : 0 ≤ (Q + 1) * A ^ Fintype.card σ / ε := by positivity
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
    refine ⟨⟨fun _ => ∅, η₀, ι, εcov / 2, δ' / 4, C, C.walkStart⟩, rfl, rfl, hbud, rfl,
      fun _ => by simp, ?_⟩
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
  have hcap0 : 0 < cap := by
    show 0 < min (min (δ' * τ ^ 2 / (2 ^ 22 * (J + 1) ^ 2 * (RB + 1)))
      (δ' * ι ^ 2 / (2 ^ 11 * (J + 1) * (RB + 1))))
      (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
        (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1)))))
    positivity
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
      ≤ δ' * rateScale η₀ ι (εcov / 2) ^ 2 := by
    rw [hK2]
    change _ ≤ δ' * τ ^ 2
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
    linarith only [h]
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
  have hprem : RoundPremises σ α P.L C cross := by
    refine ⟨hδ'0, hδ'1, by show 0 < ε / 48; positivity, le_refl _, by
      show ε < 1; linarith, ?_, hcr.le, ?_⟩
    · show 36 * (S' : ℝ) / ε * Real.log (16 * S' * (S' + 1) / δ') + S' ≤ (budget : ℝ)
      rw [hSS]
      exact Nat.le_ceil _
    · show cross * (budget * ((P.L : ℝ) + 1) * Fintype.card σ
        + Fintype.card σ * ((Fintype.card σ : ℝ) + 1) ^ 2 * Fintype.card α) ≤ δ' / 8
      rw [← hcrW]
      refine mul_le_mul_of_nonneg_left ?_ hcr.le
      show (budget : ℝ) * (L + 1) * Q + Q * (Q + 1) ^ 2 * A
        ≤ B₃ * (L + 1) * Q + Q * (Q + 1) ^ 2 * A + (Q + 1) * A ^ Fintype.card σ / ε
      have : (budget : ℝ) * (L + 1) * Q ≤ B₃ * (L + 1) * Q := by gcongr
      have : 0 ≤ (Q + 1) * A ^ Fintype.card σ / ε := by positivity
      linarith
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
  have hAeq : Aₛ = (Q + 1) * (288 / ε + 16 * (L + 1)) := by
    show ((Fintype.card σ : ℝ) + 1) * (6 / (ε / 48) + 16 * ((P.L : ℝ) + 1)) = _
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
      _ = (288 + 16 * (L + 1) * ε) / 4608 := by
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
  have hshort : ((shortStrings (α := α) (Fintype.card σ)).card : ℝ) * cross / ε ≤ δ' / 8 := by
    have h1 := card_shortStrings_le (α := α) (Fintype.card σ)
    rw [← hcrW]
    have h2 : ((shortStrings (α := α) (Fintype.card σ)).card : ℝ) / ε
        ≤ (Q + 1) * A ^ Fintype.card σ / ε := by
      gcongr
    have h3 : (Q + 1) * A ^ Fintype.card σ / ε ≤ W := by
      show _ ≤ B₃ * (L + 1) * Q + Q * (Q + 1) ^ 2 * A + (Q + 1) * A ^ Fintype.card σ / ε
      have : 0 ≤ B₃ * (L + 1) * Q := by
        have : 0 ≤ B₃ := hbud0.trans hbud
        positivity
      have : 0 ≤ Q * (Q + 1) ^ 2 * A := by positivity
      linarith
    calc ((shortStrings (α := α) (Fintype.card σ)).card : ℝ) * cross / ε
        = cross * (((shortStrings (α := α) (Fintype.card σ)).card : ℝ) / ε) := by ring
      _ ≤ cross * W := mul_le_mul_of_nonneg_left (h2.trans h3) hcr.le
  have hcore := loop_core_holds M O P (Fintype.card σ + 1) (P.L + Fintype.card σ + 3)
    ⌊PB⌋₊ ⌊RB⌋₊ η₀ ι (δ' / 4) (εcov / 2) δ' ι pAP cap (ι / 2) (pAP / 2) (ε / 1536) ε cross Tu C
  dsimp only at hcore
  rw [hcexp] at hcore
  obtain ⟨settings, -, -, -, -, hC3, hzd, hnp, hmain⟩ := hcore hL hη hη₀ hι0
    (by linarith) (by positivity) (by linarith) (by positivity) (by linarith) hδ'0 hδ'1 hι0 hpAP
    hL1 (by omega) (by omega) hcap0 (by linarith) (by linarith) (by linarith) (by linarith)
    (fun pops hpops => by
      rw [show pAP - pAP / 2 = pAP / 2 by ring]
      exact Nat.le_floor (hcounts pops hpops).1)
    (fun pops hpops => by
      rw [show pAP - pAP / 2 = pAP / 2 by ring]
      exact Nat.le_floor (hcounts pops hpops).2)
    hq1 hq2 hq3 hqm hqms hk0
    (by show prefixMax P.D k ≤ cap * (ε / 48) / 2; linarith [hk1])
    hεd (by positivity) hmr (by linarith) hcr hprem hTu0 hcond
  refine ⟨settings, by rw [hC3], by rw [hC3], by rw [hC3]; exact hbud, hzd, hnp, ?_⟩
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
    show ((Fintype.card σ : ℝ) + 1) * (δ' + δ' / 4 + (δ' + δ' / 8
      + ((shortStrings (α := α) (Fintype.card σ)).card : ℝ) * cross / ε)) ≤ (Q + 1) * (5 * δ' / 2)
    have : δ' + δ' / 4 + (δ' + δ' / 8
        + ((shortStrings (α := α) (Fintype.card σ)).card : ℝ) * cross / ε) ≤ 5 * δ' / 2 := by
      linarith
    exact mul_le_mul_of_nonneg_left this hQ0

end OrthoDFA
