import RLStar.Claims.RLStar
import RLStar.Proofs.TrieMisread
import RLStar.Proofs.Trie
import RLStar.Proofs.LoopSimple
import RLStar.Proofs.LoopReadsArith
import RLStar.Proofs.LazyDFA

/-!
# The lazy loop over `D`'s first `L` letters

`loop_simple_checks` at `P.D` the law of `D`'s first `L` letters leaves a last round passing its
checks, once `D` holds at most `p₀` on any first `k` letters. The learner's DFA reads each walk
start's sift off the last round's table, so it classifies a string as the trie DFA over `Option α`
classifies the padded string (`lazyClassifies_iff`), and misjudges at most the last round's
disagreeing probes plus the strings its family misreads (`trieSinglePieceCorrect`). Those are at
most the family's mean plus `t` but for the union bound of `TrieMisread`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

open Random (prefixMax le_prefixMax prefixMax_pos)

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ]

variable [Nonempty α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

variable [Nonempty α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

lemma misMean_split (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    {T : Measure (FreeMonoid α)} [IsFiniteMeasure T] {L : ℕ}
    (hT : T {x | x.toList.length ≠ L} = 0) :
    misMean O F kl kh T = ∫ x, misreadProb O F kl kh x ∂T
      + ∫ x, readProb O F kl kh x .undecided ∂T := by
  unfold misMean
  rw [integral_eq_sum_vector hT, integral_eq_sum_vector hT, integral_eq_sum_vector hT,
    ← Finset.sum_add_distrib]
  simp only [mul_add]

set_option maxHeartbeats 1000000 in -- the settings meet `loop_simple_checks`'s conditions
/-- The lazy loop over `D`'s first `L` letters: its DFA misjudges at most `ξT` of them, reading
at most `lazyReads` strings, but for chance `δ`. -/
theorem lazy_loop_core (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L k : ℕ)
    (η₀ c δ pAP ξ ξT t Ae p₀ ι W Tu tw : ℝ) (nh E J b Nr : ℕ) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] :
    let Q : ℝ := Fintype.card σ
    let A : ℝ := Fintype.card α
    let kN : ℕ := (L + 1) / 2
    let L₁ : ℕ := L + Fintype.card σ + 3
    let Ls : ℕ := L₁ + Fintype.card σ + 3
    let N : ℝ := L₁ + 1 - kN
    let ε : ℝ := c / (4 * (Ae + 1) * N)
    let εcov : ℝ := min (c / (10 * N)) (ξ / (8 * N))
    let δ' : ℝ := δ / (8 * (Q + 1))
    let s : ℝ := 1 / 2 - η₀
    let τ : ℝ := min ((εcov / 2) * (1 - 2 * η₀) ^ 2) ι
    let Jp : ℝ := Q + 2
    let S : ℝ := Q * ((Q + 2) * (A + 1) + 1) + 2 * (A + 1) + 1
    let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) * (E * (E + 1)) / δ') + S + 1
    let PB : ℝ := (50 * (max (Real.log (20 / τ)) (Real.log (8 * W / δ')) + 3) / s ^ 2 + 20 / s + 28)
        / pAP
      + 120 * Real.log (3 / (δ' / 2)) / pAP
    let RB : ℝ := 2 ^ 22 * Jp * Real.log (128 * (PB + 1) / δ') / (s ^ 4 * ι ^ 2)
      + 2 ^ 27 * Real.log ((Jp + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ)) / τ ^ 2
    let capP : ℝ := min (min (min (s * min (εcov / 2 * s) ι / (2 ^ 20 * (RB + 1)))
          (ι ^ 2 / (2 ^ 28 * Real.log (128 * Jp * (PB + 1) * (3 + 2 * Real.logb 2 (RB + 1)) / δ'))))
        (ι / 2 / (8 * Jp * (RB + 1))))
      (min (min (pAP / (2 ^ 18 * (PB + 1))) (ι / (2 ^ 24 * Jp ^ 2 * (RB + 1))))
        (min (pAP / (2 * Jp * (PB + 2))) (1 / (6 * Jp * (RB + 1)))))
    let SC : ℝ := 8 * (Q + 4) * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2
    let FV : ℝ := (50 * (max (Real.log (20 / τ)) (Real.log (8 * W / δ')) + 3) / s ^ 2 + 20 / s + 28)
        / 4
    let nX : ℝ := k * (6 * Jp * RB + 2 * (nh + b))
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
    4 * (1 / 2 - η₀) ≤ O.gap → η₀ < 1 / 2 →
    2 * εcov ≤ (prefixLaw D L).real {p | p ∈ O.L} → 2 * εcov ≤ (prefixLaw D L).real {p | p ∉ O.L} →
    0 < c → c ≤ 1 → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ →
    2 ≤ Fintype.card α → A ≤ Ae → 1 ≤ k → k ≤ L → 0 ≤ t →
    (∀ᵐ w ∂D, Ls ≤ w.toList.length) → 0 < p₀ → (∀ u, D.real {w | prefixOf w k = u} ≤ p₀) →
    pAP ≤ (prefixLaw D Ls).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
    p₀ ≤ capP * ε / 1536 →
    6 * Jp * RB * p₀ ≤ 1 / 2 →
    2 * ((⌊FV⌋₊ : ℝ) + 1) ^ 3 * (A ^ Ls + 1) ^ ⌊FV⌋₊ * Real.exp (-2 * t ^ 2 / p₀) ≤ δ / 8 →
    0 < ι → ι ≤ εcov / 8 → 1 ≤ W →
    0 < nh → 2 ≤ nh * capP → 16 * nh * p₀ ≤ ε / 48 →
    2 * k / (ε / 48) + 1 ≤ E → nh * (k + 1) + 1 ≤ J →
    E * J * (1 - ε / 192) ^ b ≤ δ' / 4 →
    E * ((B₃ + J * b) * (Random.Kappa.probeSifts (L₁ + k) * (Q + 3))
      + (Q + 2) * ((Q + 2) * (A + 1) * (Q + 2))) ≤ Nr →
    Nr ≤ W → (L₁ + 1) * (Q + 2) ≤ tw * W → 0 < tw → 0 < Tu →
    lazySlope σ Nr nh * Tu + lazySlack σ nh (δ' / 4) + Q * (3 * ι) / Tu < 1 →
    ε + ε / 96 + tw + (1025 / 1024 + 3 / 8) * εcov + t ≤ ξT →
    ∃ settings : LoopSettings (Fintype.card σ + 1),
      settings.level3.walkStart = k ∧ (settings.level3.probeBudget : ℝ) ≤ B₃ ∧
      (settings.scanCap : ℝ) ≤ SC ∧ settings.listSize = nh ∧ settings.epochs = E
      ∧ settings.blocks = J ∧ settings.blockSize = b ∧
      (∀ (p : settings.Draws Ω α) rounds,
          indecisiveLStarLoop settings O (prefixLaw D L) (prefixLaw D Ls) p = some rounds →
        rounds.length ≤ Fintype.card σ + 1
          ∧ ((rounds.map fun o => o.prefs.card).sum : ℝ) ≤ (Q + 1) * (6 * (Q + 2) * RB)
          ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ) ≤ (Q + 1) * (PB + 1)) ∧
      (settings.measure μ α).real
        {p | ¬ RLStarSucceeds settings O (prefixLaw D L) (prefixLaw D Ls)
          (prefixLaw D L) M ξT (lazyReads (Q + 1) (6 * (Q + 2) * RB) (PB + 1) FV Nr A nX Q) p}
        ≤ δ := by
  intro Q A kN L₁ Ls N ε εcov δ' s τ Jp S B₃ PB RB capP SC FV nX
    hL hgap hη₀ hbalL hbalN hc hc1 hδ hδ1 hpAP hξ hA2 hAe hk0 hkL ht hlong hp₀0 hpreD hpAPu
    hk1 hk3 hk6 hι0 hι8 hW1 hnh hnhc hnhp hE hJ hb hNr hNrW htwW htw hTu hslope hξT
  classical
  have hη0 : 0 ≤ η₀ := by have := gap_le_one O; linarith
  have hA2' : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have hA0 : (0 : ℝ) < A := by linarith
  have hAe2 : (2 : ℝ) ≤ Ae := hA2'.trans hAe
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hQ0 : (0 : ℝ) ≤ Q := by linarith
  have hkL₁ : kN ≤ L₁ := by show (L + 1) / 2 ≤ L + Fintype.card σ + 3; omega
  have hNe : N = ((L₁ : ℕ) : ℝ) + 1 - kN := rfl
  have hN1 : (1 : ℝ) ≤ N := by
    have : (kN : ℝ) ≤ L₁ := by exact_mod_cast hkL₁
    rw [hNe]; linarith
  have hwN : 2 + (Nat.clog 2 L₁ : ℝ) ≤ 2 * N := by
    rw [hNe]
    have h1 : 1 ≤ L₁ := by change 1 ≤ L + Fintype.card σ + 3; omega
    have h2 : Nat.clog 2 L₁ ≤ L₁ - 1 :=
      Nat.clog_le_of_le_pow (by have := (L₁ - 1).lt_two_pow_self; omega)
    have h3 : 2 * kN ≤ L₁ + 1 := by show 2 * ((L + 1) / 2) ≤ L + Fintype.card σ + 3 + 1; omega
    have : 2 + Nat.clog 2 L₁ + 2 * kN ≤ 2 * (L₁ + 1) := by omega
    have : 2 + (Nat.clog 2 L₁ : ℝ) + 2 * kN ≤ 2 * (L₁ + 1) := by exact_mod_cast this
    linarith
  obtain ⟨hε0, hε12, hεc0, hεcov1, -, -, -, -, hιL⟩ :=
    dfa_rates (ℓ := 2 + (Nat.clog 2 L₁ : ℝ)) hQ1 hAe2 hN1 (by positivity) hwN hc hc1 hξ rfl rfl
      rfl
  have hδ'0 : 0 < δ' := by show 0 < δ / (8 * (Q + 1)); positivity
  have hδ'1' : δ' < 1 := by
    show δ / (8 * (Q + 1)) < 1
    rw [div_lt_one (by positivity)]
    linarith
  have hkLs : k ≤ Ls := by show k ≤ L + Fintype.card σ + 3 + Fintype.card σ + 3; omega
  have hLLs : L ≤ Ls := by show L ≤ L + Fintype.card σ + 3 + Fintype.card σ + 3; omega
  have hTL : prefixLaw D L {x | x.toList.length ≠ L} = 0 :=
    prefixLaw_length D (hlong.mono fun w hw => hLLs.trans hw)
  have hlenL : ∀ᵐ x ∂(prefixLaw D L), x.toList.length ≤ L₁ :=
    ae_iff.2 (measure_mono_null (fun x hx => by
      simp only [Set.mem_ofPred_eq, not_le] at hx ⊢
      change L + Fintype.card σ + 3 < x.toList.length at hx
      omega) hTL)
  let P : PrefixDistribution α := ⟨prefixLaw D L, inferInstance, L₁, hlenL⟩
  have hPD : P.D = prefixLaw D L := rfl
  have hpre : ∀ u, P.D.real {x | prefixOf x k = u} ≤ p₀ := prefixLaw_prefix_le D hkL hpreD
  have hpm : prefixMax P.D k ≤ p₀ := by
    unfold prefixMax
    exact ciSup_le hpre
  have hp1 : pAP ≤ 1 := hpAPu.trans measureReal_le_one
  have hkD : ∀ᵐ x ∂P.D, k ≤ x.toList.length := by
    sorry -- closes by: prefixLaw_length, k ≤ L
  have hnhcap : 2 ≤ (nh : ℝ) * capP := hnhc
  obtain ⟨settings, hwalk, -, hbud, hSC, hnh', hE', hJ', hb', hnp, hab, hks, hmain⟩ :=
    loop_simple_checks M O P η₀ εcov ε δ' pAP ι W Tu tw k nh E J b Nr
    hL hgap hη₀ hεc0 hεcov1 hbalL hbalN hε0 hε12 hδ'0 hδ'1' hpAP hp1
    (by show 1 ≤ L + Fintype.card σ + 3; omega)
    hk0 hkD hιL (hpm.trans hk1) hι0 hι8 hW1 hnh hnhcap
    ((mul_le_mul_of_nonneg_left hpm (by positivity)).trans hnhp) hE hJ hb hNr hNrW htwW htw hTu
    hslope
  have hcap0 : 0 < capP := by
    by_contra hc0
    push Not at hc0
    have : capP * ε / 1536 ≤ 0 := by
      have := mul_nonpos_of_nonpos_of_nonneg hc0 hε0.le
      linarith
    linarith
  set Dsf : Measure (FreeMonoid α) := prefixLaw D Ls with hDsfd
  have hatom : ∀ v : FreeMonoid α, Dsf.real {v} ≤ capP := fun v => by
    refine (prefixLaw_singleton_le D hkLs hpreD v).trans ?_
    have : capP * ε ≤ capP * 1 := mul_le_mul_of_nonneg_left (by linarith) hcap0.le
    linarith
  have hD := hmain Dsf inferInstance (prefixLaw_length D hlong) hpAPu hatom
  set μL := settings.measure μ α
  set Bloop : Set (settings.Draws Ω α) :=
    {p | let rounds := idealRounds settings O P.D Dsf (persistDraws p) (Fintype.card σ + 1)
      ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
          o.Checks M O P.D k (1025 / 1024 * εcov) (3 / 8 * εcov) (ε + ε / 96 + tw))
        ∧ (∀ o ∈ rounds, o.result.peak ≤ Fintype.card σ + 2)
        ∧ RoundsFresh settings O P.D Dsf (P.L + Fintype.card σ + 3) p
        ∧ ∀ r (hr : r < Fintype.card σ + 1),
          AllHarvested (idealRounds settings O P.D Dsf (persistDraws p) r) →
            RoundGood settings O P.D Dsf (specHistOf (idealRounds settings O P.D Dsf
              (persistDraws p) r)) (persistDraws p ⟨r, hr⟩))} with hBloopd
  set Bmis : Set Ω := {ω | ∃ (F : Finset (FreeMonoid α)) (kl kh : ℕ),
      (∀ v ∈ F, v.toList.length = Ls) ∧ F.card ≤ ⌊FV⌋₊
        ∧ misMean O F kl kh (prefixLaw D L) + t < misFrac O F kl kh (prefixLaw D L) ω} with hBmisd
  have hlenT : ∀ (p : settings.Draws Ω α) rounds,
      indecisiveLStarLoop settings O (prefixLaw D L) (prefixLaw D Ls) p = some rounds →
        rounds.length ≤ Fintype.card σ + 1
          ∧ ((rounds.map fun o => o.prefs.card).sum : ℝ) ≤ (Q + 1) * (6 * (Q + 2) * RB)
          ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ) ≤ (Q + 1) * (PB + 1) := by
    sorry -- closes by: indecisiveLStarLoop_some, length_loopRounds_le, loop_reads_le, hab
  refine ⟨settings, hwalk, hbud, hSC, hnh', hE', hJ', hb', hlenT, ?_⟩
  -- off both, the loop's last round classifies through its table, reading few strings
  have hsub : {p | ¬ RLStarSucceeds settings O (prefixLaw D L) (prefixLaw D Ls)
      (prefixLaw D L) M ξT (lazyReads (Q + 1) (6 * (Q + 2) * RB) (PB + 1) FV Nr A nX Q) p}
      ⊆ Bloop ∪ Prod.fst ⁻¹' Bmis := by
    sorry -- closes by: idealRounds_mem_loop, lazyClassifies_iff, trieSinglePieceCorrect, misMean
  have hBl : μL.real Bloop ≤ δ / 2 := by
    refine hD.trans (le_of_eq ?_)
    show (Q + 1) * (4 * (δ / (8 * (Q + 1)))) = δ / 2
    field_simp
    ring
  have hmis : μL.real (Prod.fst ⁻¹' Bmis) ≤ δ / 8 := by
    have e : μL.real (Prod.fst ⁻¹' Bmis) = μ.real Bmis := by
      change (μ.prod (Measure.pi fun _ : Fin (Fintype.card σ + 1) =>
        settings.roundLaw α)).real _ = _
      rw [← Set.prod_univ, measureReal_def, measureReal_def, Measure.prod_prod, measure_univ,
        mul_one]
    rw [e]
    exact (misread_union_le O hTL (fun v => prefixLaw_singleton_le D hkL hpreD _) Ls ⌊FV⌋₊
      ht).trans hk6
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans ?_
  have u1 := measureReal_union_le (μ := μL) Bloop (Prod.fst ⁻¹' Bmis)
  linarith

end OrthoDFA

end
