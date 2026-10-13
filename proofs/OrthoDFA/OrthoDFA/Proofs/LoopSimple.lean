import OrthoDFA.Proofs.LearnerLoop
import OrthoDFA.Proofs.Budget

/-!
# The loop's claim at the theorem's own settings

`capFloor` is a closed form under the clustering's cap `qualityCap` over at most `Jm`
populations: the suffix pool and the prefix count are held by `poolCount_le` and
`prefCount_le_poly`, and the ladder by the prefix count. `learner_loop_simple_holds` takes
`LoopCore` at `S = 0`, `K = |Q| + 1`, suffixes of length `L + |Q| + 3`, the gate at `1/4`, half of
`εcov` for the clustering, `indecisionLimit = slack = ι`, the cap at `capFloor` and the failure
split evenly between the clustering's and level 5's `|Q| + 1` rounds.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

/-- A closed form under `qualityCap` over at most `Jm` populations, the band's far side at
`exp (-cLog)`. -/
def capFloor (η₀ ι εcov α δ pAP cLog slack Jm : ℝ) : ℝ :=
  let s := 1 / 2 - η₀
  let τ := min εcov (min s ι)
  let PB := 128 * (Real.log (2 / τ) + cLog) / (s ^ 2 * pAP)
    + 16 * Real.log ((Jm + 2) / (δ / 2)) / pAP ^ 2
  let RB := 6000 * Jm ^ 2 * Real.log ((Jm + 2) * PB / (δ / 2 * α * pAP * τ)) / (s ^ 4 * τ ^ 3)
  min (δ / 2 / (64 * (Jm + 3) ^ 3 * (RB ^ 2 + PB ^ 2 + 1)))
    (min (slack ^ 2 / (64 * (Jm * RB + 1))) (δ * slack ^ 2 / (16 * (RB + 2) * (Jm + 1))))

/-- `capFloor`'s bound on the suffix pool. -/
def capPB (η₀ ι εcov δ pAP cLog Jm : ℝ) : ℝ :=
  128 * (Real.log (2 / min εcov (min (1 / 2 - η₀) ι)) + cLog) / ((1 / 2 - η₀) ^ 2 * pAP)
    + 16 * Real.log ((Jm + 2) / (δ / 2)) / pAP ^ 2

/-- `capFloor`'s bound on the prefix count. -/
def capRB (η₀ ι εcov α δ pAP cLog Jm : ℝ) : ℝ :=
  6000 * Jm ^ 2 * Real.log ((Jm + 2) * capPB η₀ ι εcov δ pAP cLog Jm
    / (δ / 2 * α * pAP * min εcov (min (1 / 2 - η₀) ι)))
    / ((1 / 2 - η₀) ^ 4 * min εcov (min (1 / 2 - η₀) ι) ^ 3)

theorem capFloor_le_qualityCap {J : Type*} [Fintype J] (η₀ : ℝ) (pops : Finset J)
    (ι εcov α δ pAP cLog slack Jm : ℝ) (hη0 : 0 ≤ η₀) (hη : η₀ < 1 / 2)
    (hpop : pops.Nonempty) (hcard : (pops.card : ℝ) ≤ Jm) (hι : 0 < ι) (hε : 0 < εcov)
    (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hα : 0 < α) (hα1 : α < 1) (hpAP : 0 < pAP)
    (hpAP1 : pAP ≤ 1) (hc : 0 ≤ cLog) (hslack : 0 < slack) :
    0 < capFloor η₀ ι εcov α δ pAP cLog slack Jm
      ∧ capFloor η₀ ι εcov α δ pAP cLog slack Jm
        ≤ qualityCap η₀ pops ι εcov α δ pAP (Real.exp (-cLog)) slack
      ∧ (poolCount η₀ pops ι εcov (δ / 2) pAP (Real.exp (-cLog)) : ℝ)
        ≤ capPB η₀ ι εcov δ pAP cLog Jm
      ∧ (prefCount η₀ pops ι εcov (δ / 2) α pAP (Real.exp (-cLog)) : ℝ)
        ≤ capRB η₀ ι εcov α δ pAP cLog Jm := by
  have hsig : 0 < sig η₀ := by unfold sig; linarith
  have hδ2 : 0 < δ / 2 := by positivity
  have hδ21 : δ / 2 ≤ 1 := by linarith
  have hc1 : (1 : ℝ) ≤ pops.card := Nat.one_le_cast.2 (Finset.card_pos.2 hpop)
  have hJm : (1 : ℝ) ≤ Jm := hc1.trans hcard
  set s := 1 / 2 - η₀ with hs
  have hs0 : 0 < s := by linarith
  have hsig' : sig η₀ = s := rfl
  set τ := min εcov (min s ι) with hτ
  have hτ' : budgetScale η₀ ι εcov = τ := rfl
  have hτ0 : 0 < τ := lt_min hε (lt_min hs0 hι)
  have hτ1 : τ ≤ 1 := (min_le_left _ _).trans hε1
  have hcross : 0 < Real.exp (-cLog) := Real.exp_pos _
  have hcross1 : Real.exp (-cLog) ≤ 1 := Real.exp_le_one_iff.2 (by linarith)
  set PB := 128 * (Real.log (2 / τ) + cLog) / (s ^ 2 * pAP)
    + 16 * Real.log ((Jm + 2) / (δ / 2)) / pAP ^ 2 with hPB
  set RB := 6000 * Jm ^ 2 * Real.log ((Jm + 2) * PB / (δ / 2 * α * pAP * τ)) / (s ^ 4 * τ ^ 3)
    with hRB
  set pool := (poolCount η₀ pops ι εcov (δ / 2) pAP (Real.exp (-cLog)) : ℝ) with hpool
  set pref := (prefCount η₀ pops ι εcov (δ / 2) α pAP (Real.exp (-cLog)) : ℝ) with hpref
  -- the pool
  have hpool256 : (256 : ℝ) ≤ pool :=
    poolCount_ge pops η₀ ι εcov (δ / 2) pAP _ hsig hη0 hι hε hε1 hpAP hpAP1
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
    rw [le_div_iff₀ (by positivity)]
    have h1 : δ / 2 * α * pAP * τ ≤ 1 := by
      have : δ / 2 * α ≤ 1 := by nlinarith
      have : δ / 2 * α * pAP ≤ 1 := by nlinarith
      nlinarith
    nlinarith
  have hprefB : pref ≤ RB := by
    have h := prefCount_le_poly pops η₀ ι εcov (δ / 2) α pAP (Real.exp (-cLog)) hsig hη0 hpop hι
      hε hε1 hδ2 hδ21 hα hα1 hpAP hpAP1
    rw [hsig', hτ'] at h
    refine h.trans ?_
    unfold budgetLog
    rw [hτ']
    refine div_le_div_of_nonneg_right ?_ (by positivity)
    have hl0 : 0 ≤ Real.log (((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ)) :=
      Real.log_nonneg hlogarg
    have hlog : Real.log (((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ))
        ≤ Real.log ((Jm + 2) * PB / (δ / 2 * α * pAP * τ)) := by
      refine Real.log_le_log (by linarith) (div_le_div_of_nonneg_right ?_ (by positivity))
      exact mul_le_mul (by linarith) hpoolB (by positivity) (by positivity)
    have hsq : (pops.card : ℝ) ^ 2 ≤ Jm ^ 2 := pow_le_pow_left₀ (by positivity) hcard 2
    calc 6000 * (pops.card : ℝ) ^ 2
          * Real.log (((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ))
        ≤ 6000 * Jm ^ 2 * Real.log (((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ)) := by
          gcongr
      _ ≤ 6000 * Jm ^ 2 * Real.log ((Jm + 2) * PB / (δ / 2 * α * pAP * τ)) := by
          gcongr
  have hpref0 : 0 ≤ pref := Nat.cast_nonneg _
  have hlad : ((ladderLen η₀ pops ι εcov (δ / 2) α pAP (Real.exp (-cLog)) : ℕ) : ℝ) ≤ RB + 1 := by
    unfold ladderLen
    push_cast
    have := Nat.log_le_self 2 (prefCount η₀ pops ι εcov (δ / 2) α pAP (Real.exp (-cLog)))
    have h' : ((Nat.log 2 (prefCount η₀ pops ι εcov (δ / 2) α pAP (Real.exp (-cLog))) : ℕ) : ℝ)
        ≤ pref := by rw [hpref]; exact_mod_cast this
    linarith
  have hRB0 : 0 ≤ RB := hpref0.trans hprefB
  have hPB0 : 0 ≤ PB := by linarith
  refine ⟨?_, ?_, hpoolB, hprefB⟩
  · unfold capFloor
    dsimp only
    rw [← hs, ← hτ, ← hPB, ← hRB]
    positivity
  unfold capFloor qualityCap collisionCap
  dsimp only
  rw [← hs, ← hτ, ← hPB, ← hRB]
  refine min_le_min ?_ (min_le_min ?_ ?_)
  · refine div_le_div_of_nonneg_left hδ2.le (by positivity) ?_
    have h3 : ((pops.card : ℝ) + 3) ^ 3 ≤ (Jm + 3) ^ 3 := pow_le_pow_left₀ (by positivity)
      (by linarith) 3
    have hp2 : pref ^ 2 ≤ RB ^ 2 := pow_le_pow_left₀ hpref0 hprefB 2
    have hq2 : pool ^ 2 ≤ PB ^ 2 := pow_le_pow_left₀ (by linarith) hpoolB 2
    have : pref ^ 2 + pool ^ 2 + 1 ≤ RB ^ 2 + PB ^ 2 + 1 := by linarith
    calc 64 * ((pops.card : ℝ) + 3) ^ 3 * (pref ^ 2 + pool ^ 2 + 1)
        ≤ 64 * (Jm + 3) ^ 3 * (pref ^ 2 + pool ^ 2 + 1) := by gcongr
      _ ≤ 64 * (Jm + 3) ^ 3 * (RB ^ 2 + PB ^ 2 + 1) := by gcongr
  · refine div_le_div_of_nonneg_left (by positivity) (by positivity) ?_
    have : (pops.card : ℝ) * pref ≤ Jm * RB := mul_le_mul hcard hprefB hpref0 (by linarith)
    linarith
  · refine div_le_div_of_nonneg_left (by positivity) (by positivity) ?_
    have h1 : ((ladderLen η₀ pops ι εcov (δ / 2) α pAP (Real.exp (-cLog)) : ℕ) : ℝ) + 1
        ≤ RB + 2 := by linarith
    have h2 : (pops.card : ℝ) + 1 ≤ Jm + 1 := by linarith
    calc 16 * (((ladderLen η₀ pops ι εcov (δ / 2) α pAP (Real.exp (-cLog)) : ℕ) : ℝ) + 1)
          * ((pops.card : ℝ) + 1)
        ≤ 16 * (RB + 2) * (Jm + 1) := by gcongr

end OrthoDFA

namespace OrthoDFA

open MeasureTheory

lemma agrees_mono {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} {K : ℕ} {O : Oracle μ (FreeMonoid α)} {D : Measure (FreeMonoid α)} {k : ℕ}
    {εr εu εd εr' εu' : ℝ} (hr : εr ≤ εr') (hu : εu ≤ εu') {o : LoopRound Ω α K}
    (h : o.Agrees O D k εr εu εd) : o.Agrees O D k εr' εu' εd :=
  ⟨h.1, h.2.1.trans hr, h.2.2.1.trans hu, h.2.2.2⟩

set_option maxHeartbeats 2000000 in -- the settings and the cap meet `LoopCore`'s conditions
theorem learner_loop_simple_holds : LearnerLoopSimple := by
  intro α _ _ _ σ _ Ω _ μ _ M O P η₀ εcov εd δ pAP k Q A L δ' ι ρ s τ J PB RB cap Kp X G
    hL hη hη₀ hε hε1 hεd hεd1 hδ hδ1 hpAP hL1 hk0 hk1 hk2
  classical
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hA1 : (1 : ℝ) ≤ A := by
    show (1 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast Fintype.card_pos
  have hL1' : (1 : ℝ) ≤ L := by show (1 : ℝ) ≤ (P.L : ℝ); exact_mod_cast hL1
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hδ'0 : 0 < δ' := by show 0 < δ / (3 * (Q + 1)); positivity
  have hδ'1 : δ' ≤ 1 := by
    have : δ / (3 * (Q + 1)) ≤ δ := div_le_self hδ.le (by linarith)
    exact this.trans hδ1
  have hι0 : 0 < ι := by
    show 0 < min (εcov / 8) (1 / (3072 * (L + 1) * (Q + 1) * Q))
    exact lt_min (by positivity) (by positivity)
  have hι8 : ι ≤ εcov / 8 := min_le_left _ _
  have hιL : ι ≤ 1 / (3072 * (L + 1) * (Q + 1) * Q) := min_le_right _ _
  have hρ0 : 0 < ρ := by show 0 < εd / (256 * (Q + 1) * (Q * A + 1)); positivity
  have hKp : ((Fintype.card σ : ℝ) + ((0 : ℕ) : ℝ) + 1) * (2 * ((Fintype.card σ : ℝ)
      + ((0 : ℕ) : ℝ) + 3) * Fintype.card α + 1) * (10 / (εd / 2) ^ 3 + 270 / (εd / 2) ^ 2
      + 1400 / (εd / 2)) + 8 * ((Fintype.card σ : ℝ) + ((0 : ℕ) : ℝ) + 3) * Fintype.card α
      * (P.L + 1) = Kp := by
    show _ = ((Fintype.card σ : ℝ) + 1) * (2 * ((Fintype.card σ : ℝ) + 3) * Fintype.card α + 1)
      * (80 / εd ^ 3 + 1080 / εd ^ 2 + 2800 / εd)
      + 8 * ((Fintype.card σ : ℝ) + 3) * Fintype.card α * ((P.L : ℝ) + 1)
    simp only [Nat.cast_zero, add_zero]
    field_simp
    ring
  by_cases hpAP1 : pAP ≤ 1
  swap
  · -- no suffix distribution finds that many absorbing suffixes
    obtain ⟨s4, hwalk, hdis, hbud, -, -, hl4⟩ :=
      level5_guarantee_all M (εd / 2) P.L δ' 0 k hδ'0 hδ'1 (by positivity) (by linarith) hL1
    have := hl4 μ
    clear this hl4
    refine ⟨⟨fun _ => ∅, ι, 1 / 4, s4⟩, hwalk, hdis, ?_, ?_⟩
    · refine hbud.trans (le_of_eq ?_)
      simp only [hKp]
      simp only [add_zero]
      rfl
    · intro Dsf hDsf _ hpAPD _
      have := measureReal_le_one (μ := Dsf) (s := {v : FreeMonoid α | ∀ p, p * v ∈ O.L ↔ p ∈ O.L})
      linarith
  have hJ : (((Fintype.card σ + 1 + 1 : ℕ)) : ℝ) = J := by
    push_cast; show _ = (Fintype.card σ : ℝ) + 2; ring
  have eρ : ((0 : ℕ) + 1) * εd / (256 * ((Fintype.card σ : ℝ) + ((0 : ℕ) : ℝ) + 1)
      * ((Fintype.card σ : ℝ) * Fintype.card α + ((0 : ℕ) : ℝ) + 1)) = ρ := by
    show _ = εd / (256 * ((Fintype.card σ : ℝ) + 1) * ((Fintype.card σ : ℝ) * Fintype.card α + 1))
    simp only [Nat.cast_zero, add_zero, zero_add, one_mul]
  have hcl : (8 * ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + ((0 : ℕ) : ℝ) + 1)
      / (((0 : ℕ) + 1) * εd / (256 * ((Fintype.card σ : ℝ) + ((0 : ℕ) : ℝ) + 1)
        * ((Fintype.card σ : ℝ) * Fintype.card α + ((0 : ℕ) : ℝ) + 1))))
      = 8 * (L + 1) * (Q + 1) / ρ := by
    rw [eρ]
    show _ = 8 * ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + 1) / ρ
    simp only [Nat.cast_zero, add_zero]
  set cLog : ℝ := 8 * (L + 1) * (Q + 1) / ρ with hcLog
  have hfloor : ∀ pops : Finset (Option (Fin (Fintype.card σ + 1))), none ∈ pops →
      0 < capFloor η₀ ι (εcov / 2) (1 / 4) δ' pAP cLog ι J
        ∧ capFloor η₀ ι (εcov / 2) (1 / 4) δ' pAP cLog ι J
          ≤ qualityCap η₀ pops ι (εcov / 2) (1 / 4) δ' pAP (Real.exp (-cLog)) ι
        ∧ (poolCount η₀ pops ι (εcov / 2) (δ' / 2) pAP (Real.exp (-cLog)) : ℝ) ≤ PB
        ∧ (prefCount η₀ pops ι (εcov / 2) (δ' / 2) (1 / 4) pAP (Real.exp (-cLog)) : ℝ) ≤ RB := by
    intro pops hpops
    have hcard : (pops.card : ℝ) ≤ J := by
      have h1 := Finset.card_le_univ pops
      simp only [Fintype.card_option, Fintype.card_fin] at h1
      rw [← hJ]
      exact_mod_cast h1
    exact capFloor_le_qualityCap η₀ pops ι (εcov / 2) (1 / 4) δ' pAP cLog ι
      J hη0 hη₀ ⟨none, hpops⟩ hcard hι0 (by positivity) (by linarith) hδ'0 hδ'1 (by norm_num)
      (by norm_num) hpAP hpAP1 (by positivity) hι0
  obtain ⟨hcf0, -, hPB0', hRB0'⟩ := hfloor {none} (Finset.mem_singleton_self _)
  have hPB0 : 0 ≤ PB := (Nat.cast_nonneg _).trans hPB0'
  have hRB0 : 0 ≤ RB := (Nat.cast_nonneg _).trans hRB0'
  have hJ0 : 0 < J := by show 0 < Q + 2; linarith
  have hcap0 : 0 < cap := lt_min hcf0 (lt_min (by positivity) (by positivity))
  have hcapF : cap ≤ capFloor η₀ ι (εcov / 2) (1 / 4) δ' pAP cLog ι J := min_le_left _ _
  have hcapP : cap ≤ 1 / (3 * J * (RB + 1) * (L + 1)) :=
    (min_le_right _ _).trans (min_le_left _ _)
  have hcapC : cap ≤ δ' / (J * (PB ^ 2 + 16 * J ^ 2 * RB ^ 2 + 1)) :=
    (min_le_right _ _).trans (min_le_right _ _)
  have hθ : 1024 * ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + ((0 : ℕ) : ℝ) + 1)
      * Fintype.card σ * (2 * ι + ι) ≤ 1 := by
    have hpos : 0 < 3072 * (L + 1) * (Q + 1) * Q := by positivity
    have h := mul_le_mul_of_nonneg_left hιL hpos.le
    rw [mul_one_div_cancel hpos.ne'] at h
    have e : 1024 * ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + ((0 : ℕ) : ℝ) + 1)
        * Fintype.card σ * (2 * ι + ι) = 3072 * (L + 1) * (Q + 1) * Q * ι := by
      show _ = 3072 * ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + 1) * (Fintype.card σ : ℝ) * ι
      simp only [Nat.cast_zero, add_zero]
      ring
    rw [e]
    exact h
  have hcore := loop_core_holds M O P (Fintype.card σ + 1) 0 (P.L + Fintype.card σ + 3)
    ⌊PB⌋₊ ⌊RB⌋₊ η₀ ι (1 / 4) (εcov / 2) δ' ι pAP (εd) δ' cap k
  dsimp only at hcore
  have hNSr : ((⌊PB⌋₊ : ℕ) : ℝ) ≤ PB := Nat.floor_le hPB0
  have hNPr : ((⌊RB⌋₊ : ℕ) : ℝ) ≤ RB := Nat.floor_le hRB0
  obtain ⟨settings, -, -, hwalk, hdis, hbud, hmain⟩ := hcore hL hη hη₀ hι0 (by linarith)
    (by norm_num) (by norm_num) (by positivity) (by linarith) hδ'0 hδ'1 hι0 hpAP hεd hεd1 hδ'0
    hδ'1 hL1 (by omega) (by omega) hθ hcap0
    (fun pops hpops => by rw [hcl]; exact hcapF.trans (hfloor pops hpops).2.1)
    (fun pops hpops => by
      rw [hcl]
      exact Nat.le_floor (hfloor pops hpops).2.2.1)
    (fun pops hpops => by
      rw [hcl]
      exact Nat.le_floor (hfloor pops hpops).2.2.2)
    (by
      have hJ' : (((Fintype.card σ + 1 : ℕ) : ℝ) + 1) = J := by
        push_cast; show _ = (Fintype.card σ : ℝ) + 2; ring
      rw [hJ']
      have hpos : 0 < 3 * J * (RB + 1) * (L + 1) := by positivity
      calc 3 * J * (⌊RB⌋₊ : ℝ) * ((P.L : ℝ) + 1) * cap
          ≤ 3 * J * (RB + 1) * (L + 1) * cap :=
            mul_le_mul_of_nonneg_right (by gcongr; linarith) hcap0.le
        _ ≤ 3 * J * (RB + 1) * (L + 1) * (1 / (3 * J * (RB + 1) * (L + 1))) :=
            mul_le_mul_of_nonneg_left hcapP hpos.le
        _ = 1 := by field_simp)
    (by
      have hJ' : (((Fintype.card σ + 1 : ℕ) : ℝ) + 1) = J := by
        push_cast; show _ = (Fintype.card σ : ℝ) + 2; ring
      have hK : (((Fintype.card σ + 1 : ℕ) : ℝ)) ≤ J := by
        push_cast; show _ ≤ (Fintype.card σ : ℝ) + 2; linarith
      rw [hJ']
      have hpos : 0 < J * (PB ^ 2 + 16 * J ^ 2 * RB ^ 2 + 1) := by positivity
      calc (((Fintype.card σ + 1 : ℕ) : ℝ)) * ((⌊PB⌋₊ : ℝ) ^ 2 + 16 * J ^ 2 * (⌊RB⌋₊ : ℝ) ^ 2)
            * cap
          ≤ J * (PB ^ 2 + 16 * J ^ 2 * RB ^ 2 + 1) * cap := by
            refine mul_le_mul_of_nonneg_right ?_ hcap0.le
            have h1 : (⌊PB⌋₊ : ℝ) ^ 2 ≤ PB ^ 2 := pow_le_pow_left₀ (Nat.cast_nonneg _) hNSr 2
            have h2 : (⌊RB⌋₊ : ℝ) ^ 2 ≤ RB ^ 2 := pow_le_pow_left₀ (Nat.cast_nonneg _) hNPr 2
            have h3 : (⌊PB⌋₊ : ℝ) ^ 2 + 16 * J ^ 2 * (⌊RB⌋₊ : ℝ) ^ 2
                ≤ PB ^ 2 + 16 * J ^ 2 * RB ^ 2 + 1 := by nlinarith
            exact mul_le_mul hK h3 (by positivity) hJ0.le
        _ ≤ J * (PB ^ 2 + 16 * J ^ 2 * RB ^ 2 + 1) * (δ' / (J * (PB ^ 2 + 16 * J ^ 2 * RB ^ 2
              + 1))) := mul_le_mul_of_nonneg_left hcapC hpos.le
        _ = δ' := by field_simp)
    hk0 (by rw [eρ]; exact hk1) (by
      have e2 : 24 * unionLog (α := α) σ 0 δ' * prefixMax P.D k
          = 96 * (Q + 2) * A * Real.log ((Q + 3) * A / δ') * prefixMax P.D k := by
        unfold unionLog
        show _ = 96 * ((Fintype.card σ : ℝ) + 2) * (Fintype.card α : ℝ)
          * Real.log (((Fintype.card σ : ℝ) + 3) * (Fintype.card α : ℝ) / δ') * prefixMax P.D k
        simp only [Nat.cast_zero, add_zero]
        ring
      rw [eρ, e2]
      exact hk2)
  refine ⟨settings, hwalk, hdis, ?_, ?_⟩
  · refine hbud.trans (le_of_eq ?_)
    show _ = Kp * (Real.log (4 * (2 ^ (Fintype.card σ + 5) * A + 7 * Kp ^ 2) / δ')
      + 2 * Real.log (2 * Real.log (4 * (2 ^ (Fintype.card σ + 5) * A + 7 * Kp ^ 2) / δ')))
    simp only [Nat.cast_zero, add_zero]
    rfl
  · intro Dsf hDsf hlen hpAPD hatom
    have h := hmain Dsf hDsf (by simpa using hlen) hpAPD hatom
    have : IsProbabilityMeasure
        (loopMeasure μ P.D Dsf (Fintype.card σ + 1) settings.level5.probeBudget) := by
      unfold loopMeasure; infer_instance
    refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) (h.trans (le_of_eq ?_))
    · intro draws hd hgood
      apply hd
      obtain ⟨h1, o, ho, hag⟩ := hgood
      refine ⟨h1, o, ho, agrees_mono ?_ ?_ hag⟩
      · linarith
      · linarith
    · show ((Fintype.card σ : ℝ) + 1) * (2 * (δ / (3 * ((Fintype.card σ : ℝ) + 1)))
        + δ / (3 * ((Fintype.card σ : ℝ) + 1))) = δ
      field_simp
      ring

end OrthoDFA
