import OrthoDFA.Proofs.LoopSimple
import OrthoDFA.Proofs.ZeroCore

/-!
# The loop walked from `0` at explicit settings

`zero_simple_checks` takes `zero_core` at `K = |Q| + 1`, the probes walked from `0` and avoiding
the clustering's prefixes at the depth `a`, level 5 read at the depth `kd`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

open Random (prefixMax le_prefixMax prefixMax_pos)

set_option maxHeartbeats 8000000 in -- the settings and the cap meet `zero_core`'s conditions
/-- The loop walked from `0`, where the start is equivalent to a covered state: its first round
ends consistent, at most `ε + ε + ε/96` of the probes disagreeing and its family misreading at
most `(1025/1024) εcov` and leaving undecided at most `(3/8) εcov` of `P.D`, but for chance
`3δ'`. -/
theorem zero_simple_checks {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*}
    [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α) (L : ℕ)
    (η₀ εcov ε δ' pAP c ι θ : ℝ) (kd a : ℕ) (q : σ) :
    let Q : ℝ := Fintype.card σ
    let A : ℝ := Fintype.card α
    let N : ℝ := ((L + Fintype.card σ + 3 : ℕ) : ℝ) + 1 - ((L + 1) / 2 : ℕ)
    let s : ℝ := 1 / 2 - η₀
    let τ : ℝ := min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι
    let J : ℝ := Q + 2
    let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
    let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
    let W : ℝ := B₃ * (P.L + 1) * Q + Q * (Q + 1) ^ 2 * A + (Q + 1) * A ^ Fintype.card σ / ε
    let PB : ℝ := 128 * (Real.log (2 / τ) + 4 * Real.log (8 * W / δ')) / (s ^ 2 * pAP)
      + 16 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
    let RB : ℝ := 60000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * pAP * τ))
      / (s ^ 4 * τ ^ 2)
    let cap : ℝ := min (min (δ' * τ ^ 2 / (2 ^ 22 * (J + 1) ^ 2 * (RB + 1)))
        (δ' * ι ^ 2 / (2 ^ 9 * (J + 1) * (RB + 1))))
      (min (δ' * pAP / (2 ^ 14 * (J + 1) * (PB + 1))) (1 / (6 * J * (RB + 1))))
    let Kc : ℝ := (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ * (Q + 3) ^ ((Fintype.card σ + 2)
      * Fintype.card α)
    let H : ℝ := ∑ i ∈ Finset.Ico 0 kd, A ^ i
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
    O.η ≤ η₀ → η₀ < 1 / 2 →
    0 < εcov → εcov ≤ 1 → 0 < ε → ε ≤ 1 / 2 → 0 < δ' → δ' ≤ 1 → 0 < pAP →
    P.D = windowStrings ((L + 1) / 2) (L + Fintype.card σ + 3) → P.L = L + Fintype.card σ + 3 →
    0 < c → q ∈ coveredStates M L c →
    (∀ w : List α, M.evalFrom q w ∈ M.accept ↔ M.evalFrom M.start w ∈ M.accept) →
    0 < ι → ι ≤ εcov / 8 → 0 ≤ θ →
    3 * ι * N * A ^ Fintype.card σ / c ≤ 3 / 2 * θ →
    48 * (P.L + 1) * (Q + 1) * θ ≤ 1 →
    18 * θ * (Q + 1) ≤ ε / 96 →
    3 / 2 * θ * (B₃ * (P.L + 1) * Q + Q * (Q + 1) ^ 2 * A + H * (Q + 1)) ≤ δ' / 8 →
    8 * prefixMax P.D kd * Real.log (16 * Kc / δ') ≤ ε / 96 →
    1 ≤ a → a + Fintype.card σ < (L + 1) / 2 →
    (∀ t, P.D.real {t} ≤ cap) →
    6 * J * (RB + 1) * prefixMax P.D a ≤ ε / 96 →
    ∃ settings : LoopSettings (Fintype.card σ + 1),
      settings.level3.walkStart = 0 ∧ settings.level3.disagreementRate = ε ∧
      (settings.level3.probeBudget : ℝ) ≤ B₃ ∧ settings.avoidDepth = a ∧
      (∀ pops, ∑ B ∈ settings.schedule pops, B.npref ≤ 2 * ⌊RB⌋₊) ∧
      ∀ Dsf : Measure (FreeMonoid α), IsProbabilityMeasure Dsf →
        Dsf {v | v.toList.length ≠ P.L + Fintype.card σ + 3} = 0 →
        pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
        (∀ v, Dsf.real {v} ≤ cap) →
        (loopMeasure μ α (Fintype.card σ + 1) settings.level3.probeBudget).real
            {p | let rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p)
              ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
                  o.Checks M O P.D 0 (1025 / 1024 * εcov) (3 / 8 * εcov) (ε + ε + ε / 96))
                ∧ RoundsFresh settings O P.D Dsf (P.L + settings.level3.probeBudget + 2)
                  (P.L + Fintype.card σ + 3) p)}
          ≤ 3 * δ' := by
  intro Q A N s τ J S B₃ W PB RB cap Kc H
    hL hη hη₀ hε hε1 hεd hεd12 hδ'0 hδ'1 hpAP hPD hPL hc hq hstart hι0 hι8 hθ0 hθc hθL hθε hθδ
    hkd ha hak hDat hmr
  classical
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hA1 : (1 : ℝ) ≤ A := by
    show (1 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast Fintype.card_pos
  have hL1' : (1 : ℝ) ≤ P.L := by rw [hPL]; push_cast; linarith
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
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
  set C : Random.Level3Settings := ⟨0, δ', ε / 48, ε, budget⟩ with hCd
  -- the band's far side
  have hW1 : 1 ≤ W := by
    show 1 ≤ B₃ * (P.L + 1) * Q + Q * (Q + 1) ^ 2 * A + (Q + 1) * A ^ Fintype.card σ / ε
    have h1 : (1 : ℝ) ≤ (Q + 1) ^ 2 := one_le_pow₀ (by linarith)
    have : 1 ≤ Q * (Q + 1) ^ 2 * A :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hQ1 h1) hA1
    have : 0 ≤ B₃ * (P.L + 1) * Q := by
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
  · refine ⟨⟨fun _ => ∅, η₀, ι, εcov / 2, δ' / 4, C, a⟩, rfl, rfl, hbud, rfl,
      fun _ => by simp, ?_⟩
    intro Dsf hDsf _ hpAPD _
    have := measureReal_le_one (μ := Dsf) (s := {v : FreeMonoid α | ∀ p, p * v ∈ O.L ↔ p ∈ O.L})
    linarith
  have hτb : budgetScale η₀ ι (εcov / 2) = τ := by
    show min (min (εcov / 2) (1 / 2) * gateA η₀ ^ 2) ι
      = min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι
    rw [min_eq_left (by linarith : εcov / 2 ≤ 1 / 2)]
    rfl
  have hPBe : capPB η₀ ι (εcov / 2) δ' pAP cLog J = PB := by
    show _ = 128 * (Real.log (2 / τ) + cLog) / ((1 / 2 - η₀) ^ 2 * pAP)
      + 16 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
    unfold capPB
    rw [hτb]
  have hRBe : capRB η₀ ι (εcov / 2) (δ' / 4) δ' pAP cLog J = RB := by
    unfold capRB
    rw [hPBe, hτb]
  have hJ : (((Fintype.card σ + 1 + 1 : ℕ)) : ℝ) = J := by
    push_cast; show _ = (Fintype.card σ : ℝ) + 2; ring
  have hcounts : ∀ pops : Finset (Option (Fin (Fintype.card σ + 1))), none ∈ pops →
      (poolCount η₀ pops ι (εcov / 2) (δ' / 2) pAP (Real.exp (-cLog)) : ℝ) ≤ PB
        ∧ (prefCount η₀ pops ι (εcov / 2) (δ' / 2) (δ' / 4) pAP (Real.exp (-cLog)) 0 : ℝ)
          ≤ RB := by
    intro pops hpops
    have hcard : (pops.card : ℝ) ≤ J := by
      have h1 := Finset.card_le_univ pops
      simp only [Fintype.card_option, Fintype.card_fin] at h1
      rw [← hJ]
      exact_mod_cast h1
    have h := counts_le η₀ pops ι (εcov / 2) (δ' / 4) δ' pAP cLog J hη0 hη₀ ⟨none, hpops⟩
      hcard hι0 (by positivity) (by linarith) hδ'0 hδ'1 (by positivity) (by linarith)
      hpAP hpAP1 hcL0
    rw [hPBe, hRBe] at h
    exact h
  obtain ⟨hPB0', hRB0'⟩ := hcounts {none} (Finset.mem_singleton_self _)
  have hPB0 : 0 ≤ PB := (Nat.cast_nonneg _).trans hPB0'
  have hRB0 : 0 ≤ RB := (Nat.cast_nonneg _).trans hRB0'
  have hJ0 : 0 < J := by show 0 < Q + 2; linarith
  have hτ0 : 0 < τ := by
    have ha' : 0 < (1 - 2 * η₀) / (1 - η₀) := gateA_pos hη0 hη₀
    exact lt_min (mul_pos (by positivity) (pow_pos ha' 2)) hι0
  have hcap0 : 0 < cap := by
    show 0 < min (min (δ' * τ ^ 2 / (2 ^ 22 * (J + 1) ^ 2 * (RB + 1)))
        (δ' * ι ^ 2 / (2 ^ 9 * (J + 1) * (RB + 1))))
      (min (δ' * pAP / (2 ^ 14 * (J + 1) * (PB + 1))) (1 / (6 * J * (RB + 1))))
    positivity
  have hmul : ∀ {x b a : ℝ}, 0 < b → x ≤ b → cap ≤ a / b → x * cap ≤ a := fun hb hx h => by
    rw [le_div_iff₀ hb] at h
    exact (mul_le_mul_of_nonneg_right hx hcap0.le).trans (by linarith)
  have hNSr : ((⌊PB⌋₊ : ℕ) : ℝ) ≤ PB := Nat.floor_le hPB0
  have hNPr : ((⌊RB⌋₊ : ℕ) : ℝ) ≤ RB := Nat.floor_le hRB0
  have hK2 : (((Fintype.card σ + 1 : ℕ) : ℝ)) + 2 = J + 1 := by
    push_cast; show _ = (Fintype.card σ : ℝ) + 2 + 1; ring
  have hK1 : (((Fintype.card σ + 1 : ℕ) : ℝ)) + 1 = J := by
    push_cast; show _ = (Fintype.card σ : ℝ) + 2; ring
  have hq1 : 2 ^ 22 * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 2) ^ 2 * ((⌊RB⌋₊ : ℕ) : ℝ) * cap
      ≤ δ' * rateScale η₀ ι (εcov / 2) ^ 2 := by
    rw [hK2]
    change _ ≤ δ' * τ ^ 2
    exact hmul (b := 2 ^ 22 * (J + 1) ^ 2 * (RB + 1)) (by positivity) (by gcongr; linarith)
      ((min_le_left _ _).trans (min_le_left _ _))
  have hq2 : 2 ^ 9 * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 2) * ((⌊RB⌋₊ : ℕ) : ℝ) * cap
      ≤ δ' * ι ^ 2 := by
    rw [hK2]
    exact hmul (b := 2 ^ 9 * (J + 1) * (RB + 1)) (by positivity) (by gcongr; linarith)
      ((min_le_left _ _).trans (min_le_right _ _))
  have hq3 : 2 ^ 14 * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 2) * ((⌊PB⌋₊ : ℕ) : ℝ) * cap
      ≤ δ' * pAP := by
    rw [hK2]
    exact hmul (b := 2 ^ 14 * (J + 1) * (PB + 1)) (by positivity) (by gcongr; linarith)
      ((min_le_right _ _).trans (min_le_left _ _))
  -- level 3's premises
  have hKc : (((1 + (Fintype.card σ + 1) ^ 3 * Fintype.card α) ^ Fintype.card σ
      * (Fintype.card σ + 3) ^ ((Fintype.card σ + 2) * Fintype.card α) : ℕ) : ℝ) = Kc := by
    push_cast; rfl
  have hprem : RoundPremises σ α P.L C θ (2 * prefixMax P.D kd) cross := by
    refine ⟨hθ0, hδ'0, hδ'1, hθL, ?_, le_refl _, by
      show ε < 1; linarith, ?_, hcr.le, ?_⟩
    · show 2 * (9 * θ * ((Fintype.card σ : ℝ) + 1)
        + 2 * (2 * prefixMax P.D kd) * Real.log (16 * (((1 + (Fintype.card σ + 1) ^ 3
          * Fintype.card α) ^ Fintype.card σ * (Fintype.card σ + 3) ^ ((Fintype.card σ + 2)
            * Fintype.card α) : ℕ) : ℝ) / δ')) ≤ ε / 48
      rw [hKc]
      have e : 2 * (9 * θ * ((Fintype.card σ : ℝ) + 1) + 2 * (2 * prefixMax P.D kd)
          * Real.log (16 * Kc / δ')) = 18 * θ * (Q + 1)
            + 8 * prefixMax P.D kd * Real.log (16 * Kc / δ') := by
        show _ = 18 * θ * ((Fintype.card σ : ℝ) + 1) + _
        ring
      rw [e]
      linarith
    · show 36 * (S' : ℝ) / ε * Real.log (16 * S' * (S' + 1) / δ') + S' ≤ (budget : ℝ)
      rw [hSS]
      exact Nat.le_ceil _
    · show cross * (budget * ((P.L : ℝ) + 1) * Fintype.card σ
        + Fintype.card σ * ((Fintype.card σ : ℝ) + 1) ^ 2 * Fintype.card α) ≤ δ' / 8
      rw [← hcrW]
      refine mul_le_mul_of_nonneg_left ?_ hcr.le
      show (budget : ℝ) * (P.L + 1) * Q + Q * (Q + 1) ^ 2 * A
        ≤ B₃ * (P.L + 1) * Q + Q * (Q + 1) ^ 2 * A + (Q + 1) * A ^ Fintype.card σ / ε
      have : (budget : ℝ) * (P.L + 1) * Q ≤ B₃ * (P.L + 1) * Q := by gcongr
      have : 0 ≤ (Q + 1) * A ^ Fintype.card σ / ε := by positivity
      linarith
  have hshort' : ((shortStrings (α := α) (Fintype.card σ)).card : ℝ) * cross / ε ≤ δ' / 8 := by
    have h1 := card_shortStrings_le (α := α) (Fintype.card σ)
    rw [← hcrW]
    have h2 : ((shortStrings (α := α) (Fintype.card σ)).card : ℝ) / ε
        ≤ (Q + 1) * A ^ Fintype.card σ / ε := by
      gcongr
    have h3 : (Q + 1) * A ^ Fintype.card σ / ε ≤ W := by
      show _ ≤ B₃ * (P.L + 1) * Q + Q * (Q + 1) ^ 2 * A + (Q + 1) * A ^ Fintype.card σ / ε
      have : 0 ≤ B₃ * (P.L + 1) * Q := by
        have : 0 ≤ B₃ := hbud0.trans hbud
        positivity
      have : 0 ≤ Q * (Q + 1) ^ 2 * A := by positivity
      linarith
    calc ((shortStrings (α := α) (Fintype.card σ)).card : ℝ) * cross / ε
        = cross * (((shortStrings (α := α) (Fintype.card σ)).card : ℝ) / ε) := by ring
      _ ≤ cross * W := mul_le_mul_of_nonneg_left (h2.trans h3) hcr.le
  have hpma0 : 0 ≤ prefixMax P.D a := (prefixMax_pos P.D a).le
  have hmr' : 6 * (((Fintype.card σ + 1 : ℕ) : ℝ) + 1) * ((⌊RB⌋₊ : ℕ) : ℝ) * prefixMax P.D a
      ≤ ε / 96 := by
    rw [hK1]
    refine le_trans ?_ hmr
    have : 6 * J * ((⌊RB⌋₊ : ℕ) : ℝ) ≤ 6 * J * (RB + 1) := by gcongr; linarith
    exact mul_le_mul_of_nonneg_right this hpma0
  have hshortD : P.D {x | x.toList.length < (L + 1) / 2} = 0 := by
    rw [hPD]
    refine measure_mono_null (fun x hx => ?_) (ae_iff.1 (windowStrings_length
      ((L + 1) / 2) (L + Fintype.card σ + 3)))
    simp only [Set.mem_setOf_eq] at hx ⊢
    omega
  have hθc' : (2 * ι + ι) * ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
      * (Fintype.card α : ℝ) ^ Fintype.card σ / c ≤ 3 / 2 * θ := by
    have e : (2 * ι + ι) * ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
        * (Fintype.card α : ℝ) ^ Fintype.card σ / c = 3 * ι * N * A ^ Fintype.card σ / c := by
      show _ = 3 * ι * (((L + Fintype.card σ + 3 : ℕ) : ℝ) + 1 - ((L + 1) / 2 : ℕ))
        * (Fintype.card α : ℝ) ^ Fintype.card σ / c
      ring
    rw [e]; exact hθc
  have hLs : P.L + Fintype.card σ < P.L + Fintype.card σ + 3 := by omega
  obtain ⟨settings, -, -, -, -, hC3, hzd, hnp, hmain⟩ := zero_core M O P L
    (Fintype.card σ + 1) (P.L + Fintype.card σ + 3) ⌊PB⌋₊ ⌊RB⌋₊ ((L + 1) / 2) kd a η₀ ι
    (δ' / 4) (εcov / 2) δ' ι pAP cap (ε / 96) ε cross θ c C q hPD hL hη hη₀ hι0 (by linarith)
    (by positivity) (by linarith) (by positivity) (by linarith) hδ'0 hδ'1 hι0 hpAP
    (by rw [hPL]; omega) le_rfl hLs hcap0
    (fun pops hpops => by rw [hcexp]; exact Nat.le_floor (hcounts pops hpops).1)
    (fun pops hpops => by rw [hcexp]; exact Nat.le_floor (hcounts pops hpops).2)
    hq1 hq2 hq3 hDat hshortD ha (Nat.zero_le a) hak hεd hmr' (by linarith) hcr hprem hc hq hstart
    hθc'
  refine ⟨settings, by rw [hC3], by rw [hC3], by rw [hC3]; exact hbud, hzd, hnp, ?_⟩
  intro Dsf hDsf hlen hpAPD hatom
  have h := hmain Dsf hDsf hlen hpAPD hatom
  have hw0 : settings.level3.walkStart = 0 := by rw [hC3]
  refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) (h.trans ?_)
  · intro draws hd hgood
    apply hd
    obtain ⟨⟨h1, o, ho, hag⟩, hfr⟩ := hgood
    refine ⟨⟨h1, o, ho, checks_mono (by linarith) (by linarith) hag⟩, hfr⟩
  · have hex : (if C.walkStart < kd then 3 / 2 * θ * (C.probeBudget * (P.L + 1)
        * Fintype.card σ + Fintype.card σ * (Fintype.card σ + 1) ^ 2 * Fintype.card α
          + (∑ i ∈ Finset.Ico C.walkStart kd, (Fintype.card α : ℝ) ^ i)
            * (Fintype.card σ + 1)) else 0) ≤ δ' / 8 := by
      split_ifs
      · refine le_trans ?_ hθδ
        show 3 / 2 * θ * ((budget : ℝ) * (P.L + 1) * Q + Q * (Q + 1) ^ 2 * A + H * (Q + 1))
          ≤ 3 / 2 * θ * (B₃ * (P.L + 1) * Q + Q * (Q + 1) ^ 2 * A + H * (Q + 1))
        have : (budget : ℝ) * (P.L + 1) * Q ≤ B₃ * (P.L + 1) * Q := by gcongr
        have : 0 ≤ 3 / 2 * θ := by positivity
        nlinarith
      · positivity
    show δ' + δ' / 4 + (δ' + _ + δ' / 8
      + ((shortStrings (α := α) (Fintype.card σ)).card : ℝ) * cross / ε) ≤ 3 * δ'
    linarith

end OrthoDFA

end
