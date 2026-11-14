import Mathlib.Analysis.SpecialFunctions.Log.Base
import RLStar.Proofs.LearnerLoop
import RLStar.Proofs.Budget

/-!
# The loop's claim at explicit settings

`capPB` and `capRB` bound the suffix pool and the prefix count over at most `Jm` populations, by
`poolCount_le` and `prefCount_le_split_of_le` (`counts_le`). `loop_simple_checks` takes `LoopCore`
at `K = |Q| + 1`, with the failure split evenly between the clustering's and the lazy round's
`|Q| + 1` rounds. The lazy round's knobs `E, J, b, Nr, Nh, W, Tu` are its premises: they depend on
the cap, and the cap on them, through a log.
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory

/-- A bound on the suffix pool, the clustering's `crossLimit` at `exp (-cLog)`. -/
def capPB (η₀ ι εcov δ pAP cLog Jm : ℝ) : ℝ :=
  2 * (25 * (max (4 * Real.log (20 / budgetScale η₀ ι εcov)) cLog + 12) / (8 * (1 / 2 - η₀) ^ 2)
      + 5 / (1 / 2 - η₀) + 7) / pAP
    + 60 * Real.log (3 / (δ / 2)) / pAP

/-- A bound on the prefix count: the populations' group at the count rule's scale `ι`, the rest at
`budgetScale`. -/
def capRB (η₀ ι εcov α δ pAP cLog Jm : ℝ) : ℝ :=
  2 ^ 22 * Jm * Real.log (128 * (capPB η₀ ι εcov δ pAP cLog Jm + 1) / δ)
      / ((1 / 2 - η₀) ^ 4 * ι ^ 2)
    + 2 ^ 27 * Real.log ((Jm + 2) * capPB η₀ ι εcov δ pAP cLog Jm
      / (δ / 2 * α * pAP * budgetScale η₀ ι εcov)) / budgetScale η₀ ι εcov ^ 2

theorem counts_le {J : Type*} [Fintype J] (η₀ : ℝ) (pops : Finset J)
    (ι εcov α δ pAP cLog Jm : ℝ) (hη0 : 0 ≤ η₀) (hη : η₀ < 1 / 2)
    (hpop : pops.Nonempty) (hcard : (pops.card : ℝ) ≤ Jm) (hι : 0 < ι) (hι4 : ι ≤ εcov / 4)
    (hι1 : ι ≤ 1 / 2) (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hα : 0 < α) (hα1 : α < 1) (hpAP : 0 < pAP)
    (hpAP1 : pAP ≤ 1) (hc : 0 ≤ cLog) :
    (poolCount η₀ pops ι εcov (δ / 2) pAP (Real.exp (-cLog)) : ℝ)
        ≤ capPB η₀ ι εcov δ pAP cLog Jm
      ∧ (prefCount η₀ pops ι εcov (εcov / 512) (δ / 2) α pAP (Real.exp (-cLog)) 0 : ℝ)
        ≤ capRB η₀ ι εcov α δ pAP cLog Jm := by
  have hε : 0 < εcov := by linarith
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
  -- the pool
  have hpoolB : pool ≤ PB := by
    have h := poolCount_le pops η₀ ι εcov (δ / 2) pAP (Real.exp (-cLog)) hsig hη0 hι hε hδ2
      hδ21 hpAP hpAP1 hcross hcross1
    rw [hsig', hτ'] at h
    have e : Real.log (1 / Real.exp (-cLog)) = cLog := by
      rw [one_div, ← Real.exp_neg, neg_neg, Real.log_exp]
    rw [e] at h
    exact h
  have hpool64 : (64 : ℝ) ≤ pool :=
    poolCount_ge pops η₀ ι εcov (δ / 2) pAP _ hsig hη0 hι hε hpAP hpAP1
  -- the prefix count
  have hlogarg : 1 ≤ ((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ) := by
    rw [le_div_iff₀ (by positivity)]
    have h1 : δ / 2 * α * pAP * τ ≤ 1 := by
      have : δ / 2 * α ≤ 1 := by nlinarith
      have : δ / 2 * α * pAP ≤ 1 := by nlinarith
      nlinarith
    nlinarith
  refine ⟨hpoolB, ?_⟩
  have h := prefCount_le_split_of_le (hsz := 0) pops η₀ ι εcov (εcov / 512) (δ / 2) α pAP
    (Real.exp (-cLog)) hsig hη0 hpop hι hι4 hι1 (by linarith) hδ2 hδ21 hα hα1 hpAP hpAP1
  rw [Nat.cast_zero, mul_zero, zero_div, add_zero] at h
  rw [hsig', hτ'] at h
  refine h.trans ?_
  unfold budgetLog levelLog capRB
  rw [hτ', ← hPB, ← hpool]
  have hJm : 0 ≤ Jm := (Nat.cast_nonneg _).trans hcard
  have hlev : Real.log (64 * (pool + 1) / (δ / 2)) ≤ Real.log (128 * (PB + 1) / δ) := by
    rw [show 64 * (pool + 1) / (δ / 2) = 128 * (pool + 1) / δ by field_simp; ring]
    exact Real.log_le_log (by positivity) (div_le_div_of_nonneg_right (by linarith) hδ.le)
  have hlev0 : 0 ≤ Real.log (64 * (pool + 1) / (δ / 2)) :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ2]; nlinarith)
  have hl0 : 0 ≤ Real.log (((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ)) :=
    Real.log_nonneg hlogarg
  have hlog : Real.log (((pops.card : ℝ) + 2) * pool / (δ / 2 * α * pAP * τ))
      ≤ Real.log ((Jm + 2) * PB / (δ / 2 * α * pAP * τ)) := by
    refine Real.log_le_log (by linarith) (div_le_div_of_nonneg_right ?_ (by positivity))
    exact mul_le_mul (by linarith) hpoolB (by positivity) (by linarith)
  refine add_le_add (div_le_div_of_nonneg_right ?_ (by positivity))
    (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hlog (by norm_num)) (by positivity))
  exact mul_le_mul (mul_le_mul_of_nonneg_left hcard (by norm_num)) hlev hlev0 (by positivity)

end OrthoDFA

namespace OrthoDFA

open MeasureTheory

open Random (prefixMax le_prefixMax prefixMax_pos)

lemma agrees_mono {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} {O : Oracle μ (FreeMonoid α)} {D : Measure (FreeMonoid α)}
    {εr εu εr' εu' : ℝ} (hr : εr ≤ εr') (hu : εu ≤ εu') {o : SpecRound α}
    (h : o.Agrees O D εr εu) : o.Agrees O D εr' εu' :=
  ⟨h.1, h.2.1.trans hr, h.2.2.trans hu⟩

lemma checks_mono {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} {M : _root_.DFA α σ} {O : Oracle μ (FreeMonoid α)}
    {D : Measure (FreeMonoid α)} {k : ℕ} {εr εu εd εr' εu' εd' : ℝ}
    (hr : εr ≤ εr') (hu : εu ≤ εu') (hd : εd ≤ εd') {o : SpecRound α}
    (h : o.Checks M O D k εr εu εd) : o.Checks M O D k εr' εu' εd' :=
  ⟨agrees_mono hr hu h.1, h.2.1.trans hd, h.2.2⟩

/-- The round's miss and scan tail at scan cap `2T`, `T ≥ 8 log(16(K + 1)/δ)` and `32 T cap ≤ 1`. -/
lemma scan_tail_le {Kn T : ℕ} {cap δ δₘ : ℝ} (hδ : 0 < δ) (hcap0 : 0 ≤ cap)
    (hρ : 32 * (T : ℝ) * cap ≤ 1) (hT : 8 * Real.log (16 * ((Kn : ℝ) + 1) / δ) ≤ T)
    (hδm : (Kn : ℝ) * δₘ ≤ δ / 4) :
    (Kn : ℝ) * δₘ + ((Kn : ℝ) + 1) * (2 * (((2 * T).choose T : ℕ) * ((T : ℝ) * (2 * cap))
      ^ (2 * T - T)) + 2 * Real.exp (-((T : ℝ) / 8))) ≤ δ / 2 := by
  have h1 := choose_mul_pow_le (M := 2 * T) (T := T) (ρ := 2 * cap) (by positivity) le_rfl
    (by linarith)
  have hl2 := Real.log_two_gt_d9
  have h2 : (1 / 2 : ℝ) ^ (2 * T) ≤ Real.exp (-((T : ℝ) / 8)) := by
    calc (1 / 2 : ℝ) ^ (2 * T) ≤ (1 / 2) ^ T :=
          pow_le_pow_of_le_one (by norm_num) (by norm_num) (by omega)
      _ = Real.exp (T * Real.log (1 / 2)) := by
          rw [Real.exp_nat_mul, Real.exp_log (by norm_num)]
      _ ≤ _ := by
          rw [one_div, Real.log_inv]
          refine Real.exp_le_exp.2 ?_
          have : (0 : ℝ) ≤ T := Nat.cast_nonneg _
          nlinarith
  have hK0 : (0 : ℝ) < (Kn : ℝ) + 1 := by positivity
  have h3 : Real.exp (-((T : ℝ) / 8)) ≤ δ / (16 * ((Kn : ℝ) + 1)) := by
    rw [← Real.exp_log (by positivity : 0 < δ / (16 * ((Kn : ℝ) + 1)))]
    refine Real.exp_le_exp.2 ?_
    rw [Real.log_div hδ.ne' (by positivity), ← neg_sub, ← Real.log_div (by positivity) hδ.ne']
    linarith
  have h4 : 2 * (((2 * T).choose T : ℕ) * ((T : ℝ) * (2 * cap)) ^ (2 * T - T))
      + 2 * Real.exp (-((T : ℝ) / 8)) ≤ δ / (4 * ((Kn : ℝ) + 1)) := by
    have : δ / (4 * ((Kn : ℝ) + 1)) = 4 * (δ / (16 * ((Kn : ℝ) + 1))) := by
      field_simp; ring
    rw [this]
    linarith
  have h5 := mul_le_mul_of_nonneg_left h4 hK0.le
  rw [show ((Kn : ℝ) + 1) * (δ / (4 * ((Kn : ℝ) + 1))) = δ / 4 by field_simp] at h5
  linarith

set_option maxHeartbeats 8000000 in -- the settings and the cap meet `LoopCore`'s conditions
/-- The loop's claim with each round's failure `δ'` free and the last round's disagreeing probes
at most `ε + ε/96 + tw`, level 3 running at disagreement rate `ε` over `E` epochs and top-ups of
`J` blocks of `b` probes, one-string lists, `Nh` heavy draws, each lazy round asking at most `Nr`
strings and the band's `ζ = δ'/(8W)`; its family misreads at most `(1025/1024) εcov` and leaves
undecided at most `(3/8) εcov` of `P.D`. Every round the filled draws run is good, and no epoch's
tree has more than `|Q| + 2` leaves. -/
theorem loop_simple_checks {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*}
    [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (η₀ εcov ε δ' pAP ι W Tu tw : ℝ) (k E J b Nr Nh : ℕ) :
    let Q : ℝ := Fintype.card σ
    let A : ℝ := Fintype.card α
    let L : ℝ := P.L
    let s : ℝ := 1 / 2 - η₀
    let τ : ℝ := min ((εcov / 2) * (1 - 2 * η₀) ^ 2) ι
    let Jp : ℝ := Q + 2
    let S : ℝ := Q * ((Q + 2) * (A + 1) + 1) + 2 * (A + 1) + 1
    let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) * (E * (E + 1)) / δ') + S + 1
    let PB : ℝ := (50 * (max (Real.log (20 / τ)) (Real.log (8 * W / δ')) + 3) / s ^ 2 + 20 / s + 28)
        / pAP
      + 120 * Real.log (3 / (δ' / 2)) / pAP
    let FV : ℝ := (50 * (max (Real.log (20 / τ)) (Real.log (8 * W / δ')) + 3) / s ^ 2 + 20 / s + 28)
        / 4
    let RB : ℝ := 2 ^ 22 * Jp * Real.log (128 * (PB + 1) / δ') / (s ^ 4 * ι ^ 2)
      + 2 ^ 27 * Real.log ((Jp + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ)) / τ ^ 2
    let cap : ℝ := min (min (min (s * min (εcov / 2 * s) ι / (2 ^ 20 * (RB + 1)))
          (ι ^ 2 / (2 ^ 28 * Real.log (128 * Jp * (PB + 1) * (3 + 2 * Real.logb 2 (RB + 1)) / δ'))))
        (ι / 2 / (8 * Jp * (RB + 1))))
      (min (min (pAP / (2 ^ 18 * (PB + 1))) (ι / (2 ^ 24 * Jp ^ 2 * (RB + 1))))
        (min (pAP / (2 * Jp * (PB + 2))) (1 / (6 * Jp * (RB + 1)))))
    let θ : ℝ := cap * ε / 192
    let Kt : ℝ := (1 + (Q + 1) ^ 3 * (A + 1)) ^ Fintype.card σ
      * (Q + 3) ^ ((Fintype.card σ + 2) * (Fintype.card α + 1))
    let SC : ℝ := 8 * (Q + 4) * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2
    let HR : ℝ := Real.log (4 * (Q + 1) * (SC + 1) / δ') / (ε / 384) + 1
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
    4 * (1 / 2 - η₀) ≤ O.gap → η₀ < 1 / 2 →
    0 < εcov → εcov ≤ 1 →
    2 * εcov ≤ P.D.real {p | p ∈ O.L} → 2 * εcov ≤ P.D.real {p | p ∉ O.L} →
    0 < ε → ε ≤ 1 / 2 → 0 < δ' → δ' < 1 → 0 < pAP → pAP ≤ 1 → 1 ≤ P.L →
    1 ≤ k → (∀ᵐ x ∂P.D, k ≤ x.toList.length) → (2 + Nat.clog 2 P.L) * ε < 3 / 8 →
    prefixMax P.D k ≤ cap * ε / 12288 →
    0 < ι → ι ≤ εcov / 8 → 1 ≤ W →
    Real.log (4 * k / (θ * δ')) / θ ≤ Nh →
    cap * Real.log (8 * (Q + 1) * Kt * (E * (E + 1)) / (δ' / 8)) ≤ 1 / 8 →
    2 * k / (ε / 48) + 1 ≤ E → k + 2 ≤ J →
    E * J * (1 - ε / 192) ^ b ≤ δ' / 4 →
    E * ((B₃ + J * b) * (Random.Kappa.probeSifts (P.L + k) * (Q + 3))
      + (Q + 2) * ((Q + 2) * (A + 1) * (Q + 2))) ≤ Nr →
    Nr ≤ W → (L + 1) * (Q + 2) ≤ tw * W → 0 < tw → tw ≤ ε / 12288 →
    0 < Tu →
    2 * (Q + 1) * (288 / ε + 16 * (2 + Nat.clog 2 (P.L + k))) * Tu + Q * (3 * ι) / Tu < 7 / 8 →
    ∃ settings : LoopSettings (Fintype.card σ + 1),
      settings.level3.walkStart = k ∧ settings.level3.disagreementRate = ε ∧
      (settings.level3.probeBudget : ℝ) ≤ B₃ ∧ (settings.scanCap : ℝ) ≤ SC ∧
      (settings.listSize = 1 ∧ settings.heavyDraws = Nh) ∧
      (settings.level3.harvestReadCap : ℝ) ≤ HR ∧ settings.level3.harvestRate = ε / 48 ∧
      settings.epochs = E ∧ settings.blocks = J
      ∧ settings.blockSize = b ∧
      (∀ pops, ∑ n ∈ (settings.schedule pops).image State.npref, n ≤ 2 * ⌊RB⌋₊) ∧
      (∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin (Fintype.card σ + 1)))),
        (roundPrefs (settings.schedule pops) pops x).card ≤ 6 * (Fintype.card σ + 1 + 1) * ⌊RB⌋₊
          ∧ (roundSuffs (settings.schedule pops) x).card ≤ ⌊PB⌋₊ + 1) ∧
      (∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ ⌊FV⌋₊ + 1) ∧
      ∀ Dsf : Measure (FreeMonoid α), IsProbabilityMeasure Dsf →
        Dsf {v | v.toList.length ≠ P.L + Fintype.card σ + 3} = 0 →
        pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
        (∀ v, Dsf.real {v} ≤ cap) →
        (settings.measure μ α).real
            {p | let rounds := idealRounds settings O P.D Dsf (persistDraws p) (Fintype.card σ + 1)
              ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
                  o.Checks M O P.D k (1025 / 1024 * εcov) (3 / 8 * εcov) (ε + ε / 96 + tw))
                ∧ (∀ o ∈ rounds, o.result.peak ≤ Fintype.card σ + 2)
                ∧ RoundsFresh settings O P.D Dsf
                  (P.L + Fintype.card σ + 3) p
                ∧ ∀ r (hr : r < Fintype.card σ + 1),
                  AllHarvested (idealRounds settings O P.D Dsf (persistDraws p) r) →
                    RoundGood settings O P.D Dsf (specHistOf (idealRounds settings O P.D Dsf
                      (persistDraws p) r)) (persistDraws p ⟨r, hr⟩))}
          ≤ (Q + 1) * (5 * δ') := by
  intro Q A L s τ Jp S B₃ PB FV RB cap θ Kt SC HR
    hL hgap hη₀ hε hε1 hbalL hbalN hεd hεd12 hδ'0 hδ'1' hpAP hpAP1 hL1 hk0 hkD _ hk1 hι0 hι8
    hW1 hNh hslack hE hJ hb hNr hNrW htwW htw htwε hTu hslope
  have hδ'1 : δ' ≤ 1 := hδ'1'.le
  classical
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hA1 : (1 : ℝ) ≤ A := by
    show (1 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast Fintype.card_pos
  have hη0 : 0 ≤ η₀ := by have := gap_le_one O; linarith
  have hW0 : 0 < W := by linarith
  -- level 3's settings
  set S' : ℕ := Fintype.card σ * ((Fintype.card σ + 2) * (Fintype.card α + 1) + 1)
    + 2 * (Fintype.card α + 1) + 1 with hS'
  have hSS : (S' : ℝ) = S := by
    show ((Fintype.card σ * ((Fintype.card σ + 2) * (Fintype.card α + 1) + 1)
      + 2 * (Fintype.card α + 1) + 1 : ℕ) : ℝ) = _
    push_cast; rfl
  have hS1 : (1 : ℝ) ≤ S := by
    show 1 ≤ Q * ((Q + 2) * (A + 1) + 1) + 2 * (A + 1) + 1
    have : 0 ≤ Q * ((Q + 2) * (A + 1) + 1) := by positivity
    linarith
  have hE1 : (1 : ℝ) ≤ E := by
    have : 0 ≤ 2 * (k : ℝ) / (ε / 48) := by positivity
    linarith
  have hlogS : 0 ≤ Real.log (16 * S * (S + 1) * (E * (E + 1)) / δ') := by
    refine Real.log_nonneg ?_
    rw [le_div_iff₀ hδ'0, one_mul]
    have h1 : 1 ≤ S * (S + 1) := by nlinarith
    have h2 : 1 ≤ (E : ℝ) * (E + 1) := by nlinarith
    have h3 : 1 ≤ S * (S + 1) * ((E : ℝ) * (E + 1)) := one_le_mul_of_one_le_of_one_le h1 h2
    have e : 16 * S * (S + 1) * ((E : ℝ) * (E + 1)) = 16 * (S * (S + 1) * (E * (E + 1))) := by
      ring
    rw [e]
    linarith
  set budget : ℕ := ⌈36 * S / ε * Real.log (16 * S * (S + 1) * (E * (E + 1)) / δ') + S⌉₊
    with hbudd
  have hbud : (budget : ℝ) ≤ B₃ := by
    have := Nat.ceil_lt_add_one (show 0 ≤ 36 * S / ε
      * Real.log (16 * S * (S + 1) * (E * (E + 1)) / δ') + S by positivity)
    show (budget : ℝ) ≤ 36 * S / ε * Real.log (16 * S * (S + 1) * (E * (E + 1)) / δ') + S + 1
    linarith
  -- the band's far side
  set cross : ℝ := δ' / (8 * W) with hcrossd
  have hcr : 0 < cross := div_pos hδ'0 (by linarith)
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
  have hτb : budgetScale η₀ ι (εcov / 2) = τ := by
    show min (min (εcov / 2) (1 / 2) * gateA η₀ ^ 2) ι
      = min ((εcov / 2) * (1 - 2 * η₀) ^ 2) ι
    rw [min_eq_left (by linarith : εcov / 2 ≤ 1 / 2)]
    rfl
  have hPBe : capPB η₀ ι (εcov / 2) δ' (pAP / 2) cLog Jp = PB := by
    show _ = (50 * (max (Real.log (20 / τ)) (Real.log (8 * W / δ')) + 3) / (1 / 2 - η₀) ^ 2 + 20 /
        (1 / 2 - η₀) + 28) / pAP
      + 120 * Real.log (3 / (δ' / 2)) / pAP
    unfold capPB
    rw [hτb, hcLog, ← mul_max_of_nonneg _ _ (by norm_num : (0 : ℝ) ≤ 4)]
    field_simp
    ring
  have hRBe : capRB η₀ ι (εcov / 2) (δ' / 4) δ' (pAP / 2) cLog Jp = RB := by
    unfold capRB
    rw [hPBe, hτb]
  have hJp : (((Fintype.card σ + 1 + 1 : ℕ)) : ℝ) = Jp := by
    push_cast; show _ = (Fintype.card σ : ℝ) + 2; ring
  have hcounts : ∀ pops : Finset (Option (Fin (Fintype.card σ + 1))), none ∈ pops →
      (poolCount η₀ pops ι (εcov / 2) (δ' / 2) (pAP / 2) (Real.exp (-cLog)) : ℝ) ≤ PB
        ∧ (prefCount η₀ pops ι (εcov / 2) (εcov / 2 / 512) (δ' / 2) (δ' / 4) (pAP / 2)
          (Real.exp (-cLog)) 0 : ℝ) ≤ RB := by
    intro pops hpops
    have hcard : (pops.card : ℝ) ≤ Jp := by
      have h1 := Finset.card_le_univ pops
      simp only [Fintype.card_option, Fintype.card_fin] at h1
      rw [← hJp]
      exact_mod_cast h1
    have h := counts_le η₀ pops ι (εcov / 2) (δ' / 4) δ' (pAP / 2) cLog Jp hη0 hη₀ ⟨none, hpops⟩
      hcard hι0 (by linarith) (by linarith) hδ'0 hδ'1 (by positivity) (by linarith)
      (by positivity) (by linarith) hcL0
    rw [hPBe, hRBe] at h
    exact h
  have hfam : ∀ pops : Finset (Option (Fin (Fintype.card σ + 1))),
      famCount η₀ pops ι (εcov / 2) (δ' / 2) (Real.exp (-cLog)) ≤ ⌊FV⌋₊ := by
    intro pops
    have hsig : 0 < sig η₀ := by unfold sig; linarith
    have h := famCount_succ_le pops η₀ ι (εcov / 2) (δ' / 2) (Real.exp (-cLog)) hsig hη0 hι0
      (by linarith) (Real.exp_pos _) (Real.exp_le_one_iff.2 (by linarith))
    have e1 : Real.log (1 / Real.exp (-cLog)) = cLog := by
      rw [one_div, ← Real.exp_neg, neg_neg, Real.log_exp]
    have e : 25 * (max (4 * Real.log (20 / budgetScale η₀ ι (εcov / 2)))
        (Real.log (1 / Real.exp (-cLog))) + 12) / (8 * sig η₀ ^ 2) + 5 / sig η₀ + 7 = FV := by
      rw [e1, hτb, hcLog, ← mul_max_of_nonneg _ _ (by norm_num : (0 : ℝ) ≤ 4)]
      show _ = (50 * (max (Real.log (20 / τ)) (Real.log (8 * W / δ')) + 3) / (1 / 2 - η₀) ^ 2
        + 20 / (1 / 2 - η₀) + 28) / 4
      unfold sig
      field_simp
      ring
    rw [e] at h
    exact Nat.le_floor (by linarith)
  obtain ⟨hPB0', hRB0'⟩ := hcounts {none} (Finset.mem_singleton_self _)
  have hPB0 : 0 ≤ PB := (Nat.cast_nonneg _).trans hPB0'
  have hRB0 : 0 ≤ RB := (Nat.cast_nonneg _).trans hRB0'
  -- the scan cap `2T`
  have hJ3 : (3 : ℝ) ≤ Jp := by show 3 ≤ Q + 2; linarith
  have hlog16 : 0 < Real.log (16 * Jp / δ') :=
    Real.log_pos (by rw [one_lt_div hδ'0]; linarith)
  set Tsc : ℕ := max (4 * ((Fintype.card σ + 1 + 3) * ⌊RB⌋₊)) ⌈8 * Real.log (16 * Jp / δ')⌉₊
    with hTscd
  have hTsc8 : 8 * Real.log (16 * Jp / δ') ≤ Tsc :=
    (Nat.le_ceil _).trans (by exact_mod_cast le_max_right _ _)
  have hMSC : ((2 * Tsc : ℕ) : ℝ) ≤ SC := by
    have hT : Tsc ≤ 4 * ((Fintype.card σ + 1 + 3) * ⌊RB⌋₊) + ⌈8 * Real.log (16 * Jp / δ')⌉₊ := by
      simp only [hTscd]; omega
    have hTR : (Tsc : ℝ) ≤ 4 * ((((Fintype.card σ + 1 + 3 : ℕ)) : ℝ) * ((⌊RB⌋₊ : ℕ) : ℝ))
        + ((⌈8 * Real.log (16 * Jp / δ')⌉₊ : ℕ) : ℝ) := by exact_mod_cast hT
    have hc := Nat.ceil_lt_add_one (show 0 ≤ 8 * Real.log (16 * Jp / δ') by positivity)
    have hK3 : ((Fintype.card σ + 1 + 3 : ℕ) : ℝ) = Q + 4 := by push_cast; show _ = Q + 4; ring
    rw [hK3] at hTR
    have hNP : ((⌊RB⌋₊ : ℕ) : ℝ) ≤ RB := Nat.floor_le hRB0
    have hM : ((2 * Tsc : ℕ) : ℝ) = 2 * Tsc := by push_cast; ring
    show ((2 * Tsc : ℕ) : ℝ) ≤ 8 * (Q + 4) * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2
    have hQ4 : 0 ≤ Q + 4 := by linarith
    have := mul_le_mul_of_nonneg_left hNP hQ4
    change ((2 * Tsc : ℕ) : ℝ) ≤ 8 * (Q + 4) * RB + 16 * Real.log (16 * Jp / δ') + 2
    linarith only [hTR, hc, this, hM]
  have hTsc0 : 0 < Tsc := by
    have : (0 : ℝ) < Tsc := lt_of_lt_of_le (by positivity) hTsc8
    exact_mod_cast this
  have hQ1' : (0 : ℝ) < Q + 1 := by linarith
  set C : Random.Level3Settings := ⟨k, δ', ε, budget,
    Random.harvestN (ε / 384) (δ' / (4 * (Q + 1)) / ((2 * Tsc : ℕ) : ℝ))⟩ with hCd
  have hHR : (C.harvestReadCap : ℝ) ≤ HR := by
    show ((Random.harvestN (ε / 384) (δ' / (4 * (Q + 1)) / ((2 * Tsc : ℕ) : ℝ)) : ℕ) : ℝ)
      ≤ Real.log (4 * (Q + 1) * (SC + 1) / δ') / (ε / 384) + 1
    unfold Random.harvestN
    have hT1 : (1 : ℝ) ≤ ((2 * Tsc : ℕ) : ℝ) := by exact_mod_cast (by omega : 1 ≤ 2 * Tsc)
    have he : 1 / (δ' / (4 * (Q + 1)) / ((2 * Tsc : ℕ) : ℝ))
        = 4 * (Q + 1) * ((2 * Tsc : ℕ) : ℝ) / δ' := by
      field_simp
    rw [he]
    have hlog0 : 0 ≤ Real.log (4 * (Q + 1) * ((2 * Tsc : ℕ) : ℝ) / δ') := by
      refine Real.log_nonneg ?_
      rw [le_div_iff₀ hδ'0]
      have h := mul_le_mul (show (1 : ℝ) ≤ 4 * (Q + 1) by linarith) hT1 (by norm_num)
        (by linarith)
      linarith only [h, hδ'1']
    have hle : Real.log (4 * (Q + 1) * ((2 * Tsc : ℕ) : ℝ) / δ')
        ≤ Real.log (4 * (Q + 1) * (SC + 1) / δ') :=
      Real.log_le_log (by positivity) (div_le_div_of_nonneg_right
        (mul_le_mul_of_nonneg_left (by linarith [hMSC]) (by linarith)) hδ'0.le)
    have hc := Nat.ceil_lt_add_one (div_nonneg hlog0 (by positivity : (0 : ℝ) ≤ ε / 384))
    have := div_le_div_of_nonneg_right hle (by positivity : (0 : ℝ) ≤ ε / 384)
    linarith
  have hJ0 : 0 < Jp := by show 0 < Q + 2; linarith
  have hτ0 : 0 < τ := by
    have ha : 0 < (1 - 2 * η₀) := gateA_pos hη0 hη₀
    exact lt_min (mul_pos (by positivity) (pow_pos ha 2)) hι0
  have hs0 : 0 < s := by show 0 < 1 / 2 - η₀; linarith
  have hlogpos : 0 < Real.log (128 * Jp * (PB + 1) * (3 + 2 * Real.logb 2 (RB + 1)) / δ') := by
    have hlb : 0 ≤ Real.logb 2 (RB + 1) := Real.logb_nonneg one_lt_two (by linarith)
    have hJ1 : (1 : ℝ) ≤ Jp := by linarith
    have h1 : 1 ≤ Jp * (PB + 1) := one_le_mul_of_one_le_of_one_le hJ1 (by linarith)
    have h2 : 3 ≤ Jp * (PB + 1) * (3 + 2 * Real.logb 2 (RB + 1)) := by nlinarith
    refine Real.log_pos ?_
    rw [lt_div_iff₀ hδ'0]
    nlinarith
  have hcap0 : 0 < cap := by
    show 0 < min (min (min (s * min (εcov / 2 * s) ι / (2 ^ 20 * (RB + 1)))
          (ι ^ 2 / (2 ^ 28 * Real.log (128 * Jp * (PB + 1) * (3 + 2 * Real.logb 2 (RB + 1))
            / δ'))))
        (ι / 2 / (8 * Jp * (RB + 1))))
      (min (min (pAP / (2 ^ 18 * (PB + 1))) (ι / (2 ^ 24 * Jp ^ 2 * (RB + 1))))
        (min (pAP / (2 * Jp * (PB + 2))) (1 / (6 * Jp * (RB + 1)))))
    refine lt_min (lt_min (lt_min (by positivity)
        (div_pos (by positivity) (mul_pos (by norm_num) hlogpos)))
      (by positivity)) (by positivity)
  -- the cap's terms
  have hmul : ∀ {x b a : ℝ}, 0 < b → x ≤ b → cap ≤ a / b → x * cap ≤ a := fun hb hx h => by
    rw [le_div_iff₀ hb] at h
    exact (mul_le_mul_of_nonneg_right hx hcap0.le).trans (by linarith)
  have hNSr : ((⌊PB⌋₊ : ℕ) : ℝ) ≤ PB := Nat.floor_le hPB0
  have hNPr : ((⌊RB⌋₊ : ℕ) : ℝ) ≤ RB := Nat.floor_le hRB0
  have hK1 : (((Fintype.card σ + 1 : ℕ) : ℝ)) + 1 = Jp := by
    push_cast; show _ = (Fintype.card σ : ℝ) + 2; ring
  have hnlog : ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ) ≤ Real.logb 2 (RB + 1) := by
    rcases Nat.eq_zero_or_pos ⌊RB⌋₊ with h0 | hpos
    · rw [h0]; simp only [Nat.log_zero_right, Nat.cast_zero]
      exact Real.logb_nonneg (by norm_num) (by linarith)
    · have h1 := Real.natLog_le_logb ⌊RB⌋₊ 2
      push_cast at h1
      refine h1.trans (Real.logb_le_logb_of_le (by norm_num) (by exact_mod_cast hpos) ?_)
      linarith
  have hq1a : 2 ^ 20 * ((⌊RB⌋₊ : ℕ) : ℝ) * cap
      ≤ (1 / 2 - η₀) * min (εcov / 2 * (1 / 2 - η₀)) ι := by
    change _ ≤ s * min (εcov / 2 * s) ι
    exact hmul (b := 2 ^ 20 * (RB + 1)) (by positivity) (by gcongr; linarith)
      (((min_le_left _ _).trans (min_le_left _ _)).trans (min_le_left _ _))
  have hq1c : 2 ^ 24 * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 4) * ((⌊RB⌋₊ : ℕ) : ℝ) * cap
      ≤ min (εcov / 2) ι := by
    have hK4 : (((Fintype.card σ + 1 : ℕ) : ℝ)) + 4 = Jp + 3 := by
      push_cast; show _ = (Fintype.card σ : ℝ) + 2 + 3; ring
    have hJ2 : Jp + 3 ≤ Jp ^ 2 := by nlinarith
    rw [hK4, min_eq_right (by linarith : ι ≤ εcov / 2)]
    exact hmul (b := 2 ^ 24 * Jp ^ 2 * (RB + 1)) (by positivity) (by gcongr; linarith)
      (((min_le_right _ _).trans (min_le_left _ _)).trans (min_le_right _ _))
  have hl0 : (0 : ℝ) ≤ ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ) := Nat.cast_nonneg _
  have hhash : 2 ^ 28 * Real.log (128 * Jp * ((((⌊PB⌋₊ : ℕ) : ℝ)) + 1)
      * (3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ)) / δ') * cap ≤ ι ^ 2 := by
    have hlg : Real.log (128 * Jp * ((((⌊PB⌋₊ : ℕ) : ℝ)) + 1)
          * (3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ)) / δ')
        ≤ Real.log (128 * Jp * (PB + 1) * (3 + 2 * Real.logb 2 (RB + 1)) / δ') :=
      Real.log_le_log (by positivity) (div_le_div_of_nonneg_right (by gcongr) hδ'0.le)
    exact hmul (b := 2 ^ 28
        * Real.log (128 * Jp * (PB + 1) * (3 + 2 * Real.logb 2 (RB + 1)) / δ'))
      (mul_pos (by norm_num) hlogpos) (by gcongr)
      (((min_le_left _ _).trans (min_le_left _ _)).trans (min_le_right _ _))
  have hq1b : 2 ^ 28 * Real.log (128 * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 1)
      * ((((⌊PB⌋₊ : ℕ) : ℝ)) + 1)
      * (3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ)) / δ') * cap ≤ min (εcov / 2) ι ^ 2 := by
    rw [hK1, min_eq_right (by linarith : ι ≤ εcov / 2)]
    exact hhash
  have hq2a : 4 * Real.log (128 * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 1)
      * ((((⌊PB⌋₊ : ℕ) : ℝ)) + 1)
      * (3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ)) / δ') * cap ≤ (ι - ι / 2) ^ 2 := by
    rw [hK1, show ι - ι / 2 = ι / 2 by ring]
    have h0 : 0 ≤ Real.log (128 * Jp * ((((⌊PB⌋₊ : ℕ) : ℝ)) + 1)
        * (3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ)) / δ') * cap := by
      refine mul_nonneg (Real.log_nonneg ?_) hcap0.le
      rw [le_div_iff₀ hδ'0]
      have hJ1 : (1 : ℝ) ≤ Jp := by linarith
      have hP1 : (1 : ℝ) ≤ ((⌊PB⌋₊ : ℕ) : ℝ) + 1 := by
        linarith [(Nat.cast_nonneg ⌊PB⌋₊ : (0 : ℝ) ≤ ((⌊PB⌋₊ : ℕ) : ℝ))]
      have h3 : (1 : ℝ) ≤ 3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ) := by linarith
      have : 1 ≤ Jp * (((⌊PB⌋₊ : ℕ) : ℝ) + 1) * (3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ)) :=
        one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hJ1 hP1) h3
      nlinarith
    nlinarith [hhash, h0]
  have hq2b : 8 * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 1) * ((⌊RB⌋₊ : ℕ) : ℝ) * cap
      ≤ ι - ι / 2 := by
    rw [hK1, show ι - ι / 2 = ι / 2 by ring]
    exact hmul (b := 8 * Jp * (RB + 1)) (by positivity) (by gcongr; linarith)
      ((min_le_left _ _).trans (min_le_right _ _))
  have hq3 : 2 ^ 17 * ((⌊PB⌋₊ : ℕ) : ℝ) * cap ≤ pAP - pAP / 2 := by
    have h := hmul (x := 2 ^ 18 * ((⌊PB⌋₊ : ℕ) : ℝ)) (a := pAP)
      (b := 2 ^ 18 * (PB + 1)) (by positivity) (by gcongr; linarith)
      (((min_le_right _ _).trans (min_le_left _ _)).trans (min_le_left _ _))
    linarith only [h]
  have hqm : 6 * (((Fintype.card σ + 1 : ℕ) : ℝ)) * ((((Fintype.card σ + 1 : ℕ) : ℝ)) + 1)
      * ((⌊RB⌋₊ : ℕ) : ℝ) * cap ≤ ι / 2 := by
    rw [hK1]
    have h := hmul (x := 12 * (((Fintype.card σ + 1 : ℕ) : ℝ)) * Jp * ((⌊RB⌋₊ : ℕ) : ℝ))
      (a := ι) (b := 2 ^ 24 * Jp ^ 2 * (RB + 1)) (by positivity)
      (by
        have : 0 ≤ Jp ^ 2 * (RB + 1) := by positivity
        calc 12 * (((Fintype.card σ + 1 : ℕ) : ℝ)) * Jp * ((⌊RB⌋₊ : ℕ) : ℝ)
            ≤ 12 * Jp * Jp * (RB + 1) := by gcongr <;> linarith [Nat.floor_le hRB0]
          _ = 12 * (Jp ^ 2 * (RB + 1)) := by ring
          _ ≤ 2 ^ 24 * Jp ^ 2 * (RB + 1) := by linarith)
      (((min_le_right _ _).trans (min_le_left _ _)).trans (min_le_right _ _))
    linarith only [h]
  have hqms : (((Fintype.card σ + 1 : ℕ) : ℝ)) * (((⌊PB⌋₊ : ℕ) : ℝ) + 1) * cap ≤ pAP / 2 := by
    have h := hmul (x := 2 * (((Fintype.card σ + 1 : ℕ) : ℝ)) * (((⌊PB⌋₊ : ℕ) : ℝ) + 1))
      (a := pAP) (b := 2 * Jp * (PB + 2)) (by positivity) (by gcongr <;> linarith)
      (((min_le_right _ _).trans (min_le_right _ _)).trans (min_le_left _ _))
    linarith only [h]
  have hpm0 : 0 ≤ prefixMax P.D k := (prefixMax_pos P.D k).le
  -- the probes' avoided mass
  have hcapJ : 6 * Jp ^ 2 * (RB + 1) * cap ≤ 1 := by
    have h0 : 0 ≤ Jp ^ 2 * (RB + 1) := by positivity
    have h := hmul (x := 6 * Jp ^ 2 * (RB + 1)) (a := ι) (b := 2 ^ 24 * Jp ^ 2 * (RB + 1))
      (by positivity) (by linarith only [h0])
      (((min_le_right _ _).trans (min_le_left _ _)).trans (min_le_right _ _))
    linarith only [h, hι8, hε1]
  have hmr : 6 * (((Fintype.card σ + 1 : ℕ) : ℝ) + 1) * ((⌊RB⌋₊ : ℕ) : ℝ) * prefixMax P.D k
      ≤ ε / (12288 * Jp) := by
    rw [hK1]
    calc 6 * Jp * ((⌊RB⌋₊ : ℕ) : ℝ) * prefixMax P.D k
        ≤ 6 * Jp * (RB + 1) * (cap * ε / 12288) :=
          mul_le_mul (mul_le_mul_of_nonneg_left (by linarith [Nat.floor_le hRB0])
            (by positivity)) hk1 hpm0 (by positivity)
      _ = 6 * Jp ^ 2 * (RB + 1) * cap * (ε / (12288 * Jp)) := by field_simp
      _ ≤ 1 * (ε / (12288 * Jp)) := mul_le_mul_of_nonneg_right hcapJ (by positivity)
      _ = ε / (12288 * Jp) := one_mul _
  have hmr12 : ε / (12288 * Jp) ≤ ε / 12288 :=
    div_le_div_of_nonneg_left hεd.le (by norm_num) (by linarith)
  -- the scan tail's atoms: `32 T cap ≤ 1`
  have hρ : 32 * (Tsc : ℝ) * cap ≤ 1 := by
    have hιl : ι ≤ 1 := by linarith only [hι8, hε1]
    have hι2 : ι ^ 2 ≤ 1 := by nlinarith only [hιl, hι0]
    rcases le_total (4 * ((Fintype.card σ + 1 + 3) * ⌊RB⌋₊)) ⌈8 * Real.log (16 * Jp / δ')⌉₊
      with h | h
    · rw [hTscd, max_eq_right h]
      have hc := Nat.ceil_lt_add_one (show 0 ≤ 8 * Real.log (16 * Jp / δ') by positivity)
      set Λ := Real.log (128 * Jp * (3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ)) / δ') with hΛd
      have hX : 1 ≤ 3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ) := by linarith only [hl0]
      have hJX : 3 ≤ Jp * (3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ)) := by nlinarith only [hX, hJ3]
      have hΛ1 : 1 ≤ Λ := by
        rw [hΛd, Real.le_log_iff_exp_le (by positivity)]
        have := Real.exp_one_lt_d9
        rw [le_div_iff₀ hδ'0]
        nlinarith only [this, hJX, hδ'1, hδ'0]
      have hlg : Real.log (16 * Jp / δ') ≤ Λ :=
        Real.log_le_log (by positivity)
          (div_le_div_of_nonneg_right (by nlinarith only [hX, hJ3]) hδ'0.le)
      have hce : ((⌈8 * Real.log (16 * Jp / δ')⌉₊ : ℕ) : ℝ) ≤ 9 * Λ := by
        linarith only [hc, hlg, hΛ1]
      have h1 := mul_le_mul_of_nonneg_right hce hcap0.le
      have h2 : 0 ≤ Λ * cap := mul_nonneg (by linarith only [hΛ1]) hcap0.le
      have hΛP : Λ ≤ Real.log (128 * Jp * ((((⌊PB⌋₊ : ℕ) : ℝ)) + 1)
          * (3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ)) / δ') := by
        rw [hΛd]
        refine Real.log_le_log (by positivity) (div_le_div_of_nonneg_right ?_ hδ'0.le)
        have hP1 : (1 : ℝ) ≤ ((⌊PB⌋₊ : ℕ) : ℝ) + 1 := by
          linarith [(Nat.cast_nonneg ⌊PB⌋₊ : (0 : ℝ) ≤ ((⌊PB⌋₊ : ℕ) : ℝ))]
        have hX0 : 0 ≤ 128 * Jp * (3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ)) := by positivity
        calc 128 * Jp * (3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ))
            = 128 * Jp * (3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ)) * 1 := by ring
          _ ≤ 128 * Jp * (3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ)) * (((⌊PB⌋₊ : ℕ) : ℝ) + 1) :=
            mul_le_mul_of_nonneg_left hP1 hX0
          _ = 128 * Jp * ((((⌊PB⌋₊ : ℕ) : ℝ)) + 1) * (3 + 2 * ((Nat.log 2 ⌊RB⌋₊ : ℕ) : ℝ)) := by
            ring
      have h3 : 2 ^ 28 * Λ * cap ≤ ι ^ 2 :=
        le_trans (mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hΛP (by norm_num))
          hcap0.le) hhash
      nlinarith only [h1, h2, h3, hι2]
    · rw [hTscd, max_eq_left h]
      push_cast
      have h0 : 0 ≤ ((⌊RB⌋₊ : ℕ) : ℝ) * cap := mul_nonneg (Nat.cast_nonneg _) hcap0.le
      have h1 : 0 ≤ (Fintype.card σ : ℝ) * (((⌊RB⌋₊ : ℕ) : ℝ) * cap) :=
        mul_nonneg (Nat.cast_nonneg _) h0
      have h2 := hq1c
      push_cast at h2
      have h3 : min (εcov / 2) ι ≤ 1 := (min_le_right _ _).trans hιl
      nlinarith only [h0, h1, h2, h3]
  -- the lazy round's premises
  have hθ0 : 0 < θ := by positivity
  have hcap4 : cap ≤ 1 / 4 := by
    have hJ1 : (1 : ℝ) ≤ Jp := by linarith
    have h1 : (1 : ℝ) ≤ Jp ^ 2 * (RB + 1) :=
      one_le_mul_of_one_le_of_one_le (one_le_pow₀ hJ1) (by linarith)
    have h6 : (6 : ℝ) ≤ 6 * Jp ^ 2 * (RB + 1) := by linarith only [h1]
    have := mul_le_mul_of_nonneg_right h6 hcap0.le
    linarith only [this, hcapJ]
  have hθ1 : θ ≤ 1 := by
    show cap * ε / 192 ≤ 1
    have := mul_le_mul hcap4 hεd12 hεd.le (by norm_num : (0 : ℝ) ≤ 1 / 4)
    linarith only [this]
  have hprem : LazyPremises σ α C E J (2 * θ) := by
    unfold LazyPremises
    dsimp only
    refine ⟨show (0 : ℝ) < ε / 48 by positivity, ?_, show ε / 48 ≤ ε / 48 from le_rfl, hδ'0,
      hδ'1', hεd, show ε < 1 by linarith, ?_, hE, hJ⟩
    · show 8 * (2 * (cap * ε / 192)) ≤ ε / 48
      have := mul_le_mul_of_nonneg_right hcap4 hεd.le
      linarith only [this]
    · show 36 * (S' : ℝ) / ε * Real.log (16 * S' * (S' + 1) * (E * (E + 1)) / δ') + S'
        ≤ (budget : ℝ)
      rw [hSS]
      exact Nat.le_ceil _
  have hNr' : lazyReadCap C E J b P.L (Fintype.card σ) (Fintype.card α) ≤ Nr := by
    have h : (lazyReadCap C E J b P.L (Fintype.card σ) (Fintype.card α) : ℝ) ≤ Nr := by
      refine le_trans ?_ hNr
      unfold lazyReadCap
      push_cast
      show (E : ℝ) * (((budget : ℝ) + J * b) * (Random.Kappa.probeSifts (P.L + k) * (Q + 3))
          + (Q + 2) * ((Q + 2) * (A + 1) * (Q + 2)))
        ≤ E * ((B₃ + J * b) * (Random.Kappa.probeSifts (P.L + k) * (Q + 3))
          + (Q + 2) * ((Q + 2) * (A + 1) * (Q + 2)))
      exact mul_le_mul_of_nonneg_left (add_le_add (mul_le_mul_of_nonneg_right
        (add_le_add hbud le_rfl)
        (mul_nonneg (Nat.cast_nonneg _)
          (show (0 : ℝ) ≤ Q + 3 from add_nonneg (Nat.cast_nonneg _) (by norm_num)))) le_rfl)
        (Nat.cast_nonneg _)
    exact_mod_cast h
  have hcr1 : cross ≤ 1 := by
    rw [hcrossd, div_le_one (by linarith)]
    linarith
  have hslope1 : lazySlope σ cross C P.L
      ≤ 2 * (Q + 1) * (288 / ε + 16 * (2 + Nat.clog 2 (P.L + k))) := by
    unfold lazySlope
    show (1 + cross) * ((Fintype.card σ : ℝ) + 1)
        * (6 / (ε / 48) + 16 * (2 + (Nat.clog 2 (P.L + k) : ℝ))) ≤ _
    rw [show 6 / (ε / 48) = 288 / ε by field_simp; ring]
    have h0 : 0 ≤ 288 / ε + 16 * (2 + (Nat.clog 2 (P.L + k) : ℝ)) := by positivity
    have h1 : (1 + cross) * ((Fintype.card σ : ℝ) + 1) ≤ 2 * (Q + 1) :=
      mul_le_mul_of_nonneg_right (by linarith) (by positivity)
    exact mul_le_mul_of_nonneg_right h1 h0
  have hslack1 : lazySlack σ α C E (2 * θ) ≤ 1 / 16 := by
    have hX : 0 < 8 * (Q + 1) * Kt * (E * (E + 1)) := by
      have hKt : 0 < Kt := by positivity
      have hE0 : (0 : ℝ) < E := by linarith
      positivity
    have hlog : Real.log (8 * (Q + 1) * Kt * (E * (E + 1)) / δ')
        ≤ Real.log (8 * (Q + 1) * Kt * (E * (E + 1)) / (δ' / 8)) :=
      Real.log_le_log (by positivity) (div_le_div_of_nonneg_left hX.le (by positivity)
        (by linarith))
    have e : lazySlack σ α C E (2 * θ)
        = cap * Real.log (8 * (Q + 1) * Kt * (E * (E + 1)) / δ') / 2 := by
      unfold lazySlack
      dsimp only
      push_cast
      show 2 * (cap * ε / 192) * Real.log (8 * (Q + 1) * Kt * (E * (E + 1)) / δ') / (ε / 48) = _
      generalize Real.log (8 * (Q + 1) * Kt * (E * (E + 1)) / δ') = Lg
      field_simp
      ring
    rw [e]
    have := mul_le_mul_of_nonneg_left hlog hcap0.le
    linarith
  have hcond : lazySlope σ cross C P.L * Tu + lazySlack σ α C E (2 * θ)
      + (2 * tw + ε / (12288 * Jp)) / (C.harvestRate / 4)
      + Fintype.card σ * (2 * ι + ι) / Tu < 1 := by
    have e : (Fintype.card σ : ℝ) * (2 * ι + ι) = Q * (3 * ι) := by
      show (Fintype.card σ : ℝ) * (2 * ι + ι) = (Fintype.card σ : ℝ) * (3 * ι)
      ring
    rw [e]
    have h1 := mul_le_mul_of_nonneg_right hslope1 hTu.le
    have h2 : (2 * tw + ε / (12288 * Jp)) / (C.harvestRate / 4) ≤ 3 / 64 := by
      show (2 * tw + ε / (12288 * Jp)) / (ε / 48 / 4) ≤ 3 / 64
      rw [div_le_iff₀ (by positivity)]
      linarith
    linarith
  have hcore := loop_core_holds M O P (Fintype.card σ + 1) (P.L + Fintype.card σ + 3)
    ⌊PB⌋₊ ⌊RB⌋₊ ⌊FV⌋₊ η₀ ι (δ' / 4) (εcov / 2) δ' ι pAP cap (ι / 2) (pAP / 2)
    (ε / (12288 * Jp)) cross Tu C (2 * Tsc) Tsc E J b Nr Nh θ tw (ε / 384) (δ' / (4 * (Q + 1)))
  dsimp only at hcore
  rw [hcexp] at hcore
  obtain ⟨settings, -, -, -, -, hC3, hscan, hlist, hheavy, hep, hbl, hbs, hnp, hreads, hks,
    hmain⟩ :=
    hcore hL hgap hη₀ hι0 (by linarith) (by positivity) (by linarith) (by positivity)
    (by linarith) (by linarith) (by linarith) hδ'0 hδ'1 hι0 hpAP
    hL1 (by omega) (by omega) hcap0 (by linarith) (by linarith) (by linarith) (by linarith)
    (fun pops hpops => by
      rw [show pAP - pAP / 2 = pAP / 2 by ring]
      exact Nat.le_floor (hcounts pops hpops).1)
    (fun pops hpops => by
      rw [show pAP - pAP / 2 = pAP / 2 by ring]
      exact Nat.le_floor (hcounts pops hpops).2)
    (fun pops _ => hfam pops)
    hq1a hq1c hq1b hq2a hq2b hq3 hqm hqms hk0 hkD hmr (by linarith) hcr hprem hθ0
    (hk1.trans (div_le_div_of_nonneg_left (by positivity) (by norm_num) (by norm_num)))
    (le_of_eq (by show 4 * (cap * ε / 192) = cap * (ε / 48); ring))
    (by show 2 * tw ≤ ε / 48; linarith) hNr' htw hTu hcond rfl (by positivity)
    (by show ε / 384 ≤ ε / 48 / 8; linarith) (div_pos hδ'0 (by linarith)) (by omega)
    (le_max_left _ _) (by omega)
  refine ⟨settings, by rw [hC3], by rw [hC3], by rw [hC3]; exact hbud,
    by rw [hscan]; exact hMSC, ⟨hlist, hheavy⟩, by rw [hC3]; exact hHR, by rw [hC3]; rfl,
    hep, hbl, hbs, hnp, hreads, hks, ?_⟩
  intro Dsf hDsf hlen hpAPD hatom
  have h := hmain Dsf hDsf hlen hpAPD hatom
  refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) (h.trans ?_)
  · intro p hd hgood
    apply hd
    obtain ⟨⟨h1, o, ho, hag⟩, hlv, hfr, hrg⟩ := hgood
    exact ⟨⟨h1, o, ho, checks_mono (by linarith) (by linarith)
      (by show ε + tw + ε / (12288 * Jp) ≤ ε + ε / 96 + tw; linarith) hag⟩, hlv, hfr, hrg⟩
  · have hKQ : ((Fintype.card σ + 1 : ℕ) : ℝ) = Q + 1 := by push_cast; rfl
    have hbsc := scan_tail_le (Kn := Fintype.card σ + 1) (T := Tsc) (δₘ := δ' / (4 * (Q + 1)))
      hδ'0 hcap0.le hρ (by rw [hK1]; exact hTsc8)
      (le_of_eq (by rw [hKQ]; field_simp))
    have hheavyB : (C.walkStart : ℝ) / θ * (1 - θ) ^ Nh ≤ δ' / 4 := by
      show (k : ℝ) / θ * (1 - θ) ^ Nh ≤ δ' / 4
      have hk0' : (1 : ℝ) ≤ k := by exact_mod_cast hk0
      have harg : 0 < 4 * k / (θ * δ') := by positivity
      have h1 : Real.log (4 * k / (θ * δ')) ≤ θ * Nh := by
        rw [div_le_iff₀ hθ0] at hNh; linarith
      have h2 : (1 - θ) ^ Nh ≤ Real.exp (-(θ * Nh)) := by
        calc (1 - θ) ^ Nh ≤ Real.exp (-θ) ^ Nh :=
              pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp (-θ)]) Nh
          _ = Real.exp (-(θ * Nh)) := by rw [← Real.exp_nat_mul]; ring_nf
      have h3 : Real.exp (-(θ * Nh)) ≤ Real.exp (-Real.log (4 * k / (θ * δ'))) :=
        Real.exp_le_exp.2 (by linarith)
      rw [Real.exp_neg (Real.log _), Real.exp_log harg] at h3
      calc (k : ℝ) / θ * (1 - θ) ^ Nh ≤ k / θ * (4 * k / (θ * δ'))⁻¹ :=
            mul_le_mul_of_nonneg_left (h2.trans h3) (by positivity)
        _ = δ' / 4 := by field_simp
    have hlr : lazyRoundBound σ C E J b cross Nr P.L tw ≤ 5 / 2 * δ' := by
      unfold lazyRoundBound
      have e1 : C.δ = δ' := rfl
      have e2 : C.harvestRate = ε / 48 := rfl
      rw [e1, e2, show ε / 48 / 4 = ε / 192 by ring]
      have h1 : cross * (Nr : ℝ) ≤ δ' / 8 := by
        rw [← hcrW]; exact mul_le_mul_of_nonneg_left hNrW hcr.le
      have h2 : ((P.L : ℝ) + 1) * (Fintype.card σ + 2) * cross / tw ≤ δ' / 8 := by
        rw [div_le_iff₀ htw]
        calc ((P.L : ℝ) + 1) * (Fintype.card σ + 2) * cross ≤ tw * W * cross :=
              mul_le_mul_of_nonneg_right htwW hcr.le
          _ = tw * (cross * W) := by ring
          _ = δ' / 8 * tw := by rw [hcrW]; ring
      linarith
    have hQ0 : (0 : ℝ) ≤ (Fintype.card σ : ℝ) + 1 := by positivity
    refine mul_le_mul_of_nonneg_left ?_ hQ0
    linarith

end OrthoDFA
