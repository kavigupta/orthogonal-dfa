import Mathlib.Analysis.SpecialFunctions.Log.Base
import RLStar.Proofs.LearnerLoop
import RLStar.Proofs.Budget

/-!
# The loop's claim at explicit settings

`capPB` and `capRB` bound the suffix pool and the prefix count over at most `Jm` populations, by
`poolCount_le` and `prefCount_le_split_of_le` (`counts_le`). `loop_simple_checks` takes `LoopCore`
at `K = |Q| + 1`, with the failure split evenly between the clustering's and the lazy round's
`|Q| + 1` rounds. The lazy round's knobs `nh, E, J, b, Nr, W, Tu` are its premises: they depend on
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

/-- The loop's claim with each round's failure `δ'` free and the last round's disagreeing probes
at most `ε + ε/96 + tw`, level 3 running at disagreement rate `ε` over `E` epochs and top-ups of
`J` blocks of `b` probes, lists of `nh` strings, each lazy round asking at most `Nr` strings and
the band's `ζ = δ'/(8W)`; its family misreads at most `(1025/1024) εcov` and leaves undecided at
most `(3/8) εcov` of `P.D`. Every round the filled draws run is good, and no epoch's tree has more
than `|Q| + 2` leaves. -/
theorem loop_simple_checks {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*}
    [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (η₀ εcov ε δ' pAP ι W Tu tw : ℝ) (k nh E J b Nr : ℕ) :
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
    let SC : ℝ := 8 * (Q + 4) * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
    4 * (1 / 2 - η₀) ≤ O.gap → η₀ < 1 / 2 →
    0 < εcov → εcov ≤ 1 →
    2 * εcov ≤ P.D.real {p | p ∈ O.L} → 2 * εcov ≤ P.D.real {p | p ∉ O.L} →
    0 < ε → ε ≤ 1 / 2 → 0 < δ' → δ' < 1 → 0 < pAP → pAP ≤ 1 → 1 ≤ P.L →
    1 ≤ k → (∀ᵐ x ∂P.D, k ≤ x.toList.length) → (2 + Nat.clog 2 P.L) * ε < 3 / 8 →
    prefixMax P.D k ≤ cap * ε / 1536 →
    0 < ι → ι ≤ εcov / 8 → 1 ≤ W →
    0 < nh → 2 ≤ nh * cap → 16 * nh * prefixMax P.D k ≤ ε / 48 →
    2 * k / (ε / 48) + 1 ≤ E → nh * (k + 1) + 1 ≤ J →
    E * J * (1 - ε / 192) ^ b ≤ δ' / 4 →
    E * ((B₃ + J * b) * (Random.Kappa.probeSifts (P.L + k) * (Q + 3))
      + (Q + 2) * ((Q + 2) * (A + 1) * (Q + 2))) ≤ Nr →
    Nr ≤ W → (L + 1) * (Q + 2) ≤ tw * W → 0 < tw →
    0 < Tu →
    lazySlope σ Nr nh * Tu + lazySlack σ nh (δ' / 4) + Q * (3 * ι) / Tu < 1 →
    ∃ settings : LoopSettings (Fintype.card σ + 1),
      settings.level3.walkStart = k ∧ settings.level3.disagreementRate = ε ∧
      (settings.level3.probeBudget : ℝ) ≤ B₃ ∧ (settings.scanCap : ℝ) ≤ SC ∧
      settings.listSize = nh ∧ settings.epochs = E ∧ settings.blocks = J
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
          ≤ (Q + 1) * (4 * δ') := by
  sorry -- closes by: main's settings at C.δ = δ'/4, cross = δ'/(8W), δH = δ'/4; loop_core_holds

end OrthoDFA
