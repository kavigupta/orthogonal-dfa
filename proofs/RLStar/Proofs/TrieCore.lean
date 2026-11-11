import RLStar.Claims.RLStar
import RLStar.Proofs.TrieMisread
import RLStar.Proofs.Trie
import RLStar.Proofs.TrieAgree
import RLStar.Proofs.LoopDFA

/-!
# The loop with local walk starts over `D`'s first `L` letters

`loop_simple_checks` at `P.D` the law of `D`'s first `L` letters leaves a last round passing its
checks, once `D` holds at most `p₀` on any first `k` letters. Its probes and harvested batches are
at least `walkStart` long (`TrieStage`), so the loop with local walk starts below `walkStart` letters
runs that loop's rounds and reads (`trieLoopAgrees_holds`), and its DFA misjudges at most the last round's disagreeing probes plus
the strings its family misreads (`trieSinglePieceCorrect`). Those are at most the family's mean
plus `t` but for the union bound of `TrieMisread`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

open Random (prefixMax le_prefixMax prefixMax_pos completeTrie)

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ]

lemma length_trieLoopRoundsQ_le {K : ℕ} (settings : LoopSettings K) (cap : ℕ)
    (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget)
    (rd : FreeMonoid α → ℝ) :
    ∀ r rs, (trieLoopRoundsQ settings cap D Dsf draws r).eval rd = some rs → rs.length ≤ r
  | 0, rs, h => by
    simp only [trieLoopRoundsQ, Query.eval_pure, Option.some.injEq] at h
    subst h
    simp
  | r + 1, rs, h => by
    rw [trieLoopRoundsQ, Query.eval_bind] at h
    rcases hr : (trieLoopRoundsQ settings cap D Dsf draws r).eval rd with _ | rs'
    · rw [hr] at h
      simp at h
    have ih := length_trieLoopRoundsQ_le settings cap D Dsf draws rd r rs' hr
    rw [hr] at h
    simp only at h
    split_ifs at h with hc
    · rw [Query.eval_map] at h
      obtain ⟨o, -, rfl⟩ := Option.map_eq_some_iff.1 h
      simp only [List.length_append, List.length_singleton]
      omega
    · simp only [Query.eval_pure, Option.some.injEq] at h
      subst h
      omega

lemma trieRoundQ_prefs {K : ℕ} (settings : LoopSettings K) (cap : ℕ)
    (D Dsf : Measure (FreeMonoid α)) (h : TrieLoopHist α)
    (d : RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget)
    (rd : FreeMonoid α → ℝ) (o : TrieLoopRound α)
    (ho : (trieRoundQ settings cap D Dsf h d).eval rd = some o) :
    ∃ (pops : Finset (Option (Fin K))) (x : Run Unit (FreeMonoid α) (Option (Fin K))),
      o.prefs = roundPrefs (settings.schedule pops) pops x
        ∧ o.suffs = roundSuffs (settings.schedule pops) x := by
  unfold trieRoundQ at ho
  rw [Query.eval_bind] at ho
  split at ho
  · simp at ho
  · dsimp only at ho
    split_ifs at ho
    · rw [Query.eval_bind] at ho
      split_ifs at ho
      · rw [Query.eval_map] at ho
        cases ho
        exact ⟨_, _, rfl, rfl⟩
      · simp at ho
    · simp at ho

lemma trieLoopRoundsQ_prefs {K : ℕ} (settings : LoopSettings K) (cap : ℕ)
    (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget)
    (rd : FreeMonoid α → ℝ) :
    ∀ r rs, (trieLoopRoundsQ settings cap D Dsf draws r).eval rd = some rs →
      ∀ o ∈ rs, ∃ (pops : Finset (Option (Fin K))) (x : Run Unit (FreeMonoid α) (Option (Fin K))),
        o.prefs = roundPrefs (settings.schedule pops) pops x
          ∧ o.suffs = roundSuffs (settings.schedule pops) x
  | 0, rs, h => by
    simp only [trieLoopRoundsQ, Query.eval_pure, Option.some.injEq] at h
    subst h
    simp
  | r + 1, rs, h => by
    rw [trieLoopRoundsQ, Query.eval_bind] at h
    rcases hr : (trieLoopRoundsQ settings cap D Dsf draws r).eval rd with _ | rs'
    · rw [hr] at h
      simp at h
    have ih := trieLoopRoundsQ_prefs settings cap D Dsf draws rd r rs' hr
    rw [hr] at h
    simp only at h
    split_ifs at h with hc
    · rw [Query.eval_map] at h
      obtain ⟨o, ho, rfl⟩ := Option.map_eq_some_iff.1 h
      intro o' ho'
      rcases List.mem_append.1 ho' with h1 | h1
      · exact ih o' h1
      · rw [List.mem_singleton] at h1
        subst h1
        exact trieRoundQ_prefs settings cap D Dsf _ _ rd _ ho
    · simp only [Query.eval_pure, Option.some.injEq] at h
      subst h
      exact ih

lemma trie_rounds_counts {Ω : Type*} {K : ℕ} (settings : LoopSettings K) (cap : ℕ)
    (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget)
    (rd : FreeMonoid α → ℝ) (ω : Ω) (a b : ℕ)
    (hab : ∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b)
    (rounds : List (TrieLoopRound α))
    (h : (trieLoopRoundsQ settings cap D Dsf draws K).eval rd = some rounds) :
    rounds.length ≤ K ∧ (rounds.map fun o => o.prefs.card).sum ≤ K * a
      ∧ (rounds.map fun o => o.suffs.card).sum ≤ K * b := by
  have hlen := length_trieLoopRoundsQ_le settings cap D Dsf draws rd K rounds h
  have hro := trieLoopRoundsQ_prefs settings cap D Dsf draws rd K rounds h
  have hsum : ∀ (f : TrieLoopRound α → ℕ) (c : ℕ), (∀ o ∈ rounds, f o ≤ c) →
      (rounds.map f).sum ≤ K * c := fun f c hf => by
    refine (List.sum_le_card_nsmul _ c fun n hn => ?_).trans ?_
    · obtain ⟨o, ho, rfl⟩ := List.mem_map.1 hn
      exact hf o ho
    · rw [List.length_map, smul_eq_mul]
      exact Nat.mul_le_mul_right _ hlen
  refine ⟨hlen, hsum _ a fun o ho => ?_, hsum _ b fun o ho => ?_⟩
  · obtain ⟨pops, x, e, -⟩ := hro o ho
    rw [e]
    exact (hab pops (ω, x.2)).1
  · obtain ⟨pops, x, -, e⟩ := hro o ho
    rw [e]
    exact (hab pops (ω, x.2)).2

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

set_option maxHeartbeats 8000000 in -- the settings meet `loop_simple_checks`'s conditions
/-- The loop with local walk starts below `k` letters, over `D`'s first `L` letters: its DFA
misjudges at most `ξT` of them, reading at most `learnerReads` strings, but for chance `δ`. -/
theorem trie_loop_core (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L k cap : ℕ)
    (η₀ c δ pAP ξ ξT t Ae p₀ : ℝ) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] :
    let Q : ℝ := Fintype.card σ
    let A : ℝ := Fintype.card α
    let kN : ℕ := (L + 1) / 2
    let L₁ : ℕ := L + Fintype.card σ + 3
    let Ls : ℕ := L₁ + Fintype.card σ + 3
    let N : ℝ := L₁ + 1 - kN
    let ε : ℝ := c / (4 * (Ae + 1) * N)
    let εcov : ℝ := min (c / (10 * N)) (ξ / (8 * N))
    let δ' : ℝ := δ / (8 * (Q + 1))
    let ι : ℝ := min (εcov / 8) (ε / (4608 * Q * (Q + 1)))
    let s : ℝ := 1 / 2 - η₀
    let τ : ℝ := min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι
    let J : ℝ := Q + 2
    let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
    let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
    let Wm : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
    let W : ℝ := max (2 * Real.exp 2 * Wm) 1
    let PB : ℝ := (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
      + 120 * Real.log (3 / (δ' / 2)) / pAP
    let RB : ℝ := 2 ^ 20 * J * Real.log (128 * (PB + 1) / δ') / (s ^ 4 * ι ^ 2)
      + 2 ^ 27 * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ)) / τ ^ 2
    let capP : ℝ := min (min (min (s * min (εcov / 2 * s) ι / (2 ^ 20 * (RB + 1)))
          (ι ^ 2 / (2 ^ 28 * Real.log (128 * J * (3 + 2 * Real.logb 2 (RB + 1)) / δ'))))
        (ι / 2 / (8 * J * (RB + 1))))
      (min (min (δ' * pAP / (2 ^ 16 * (PB + 1))) (ι / (2 ^ 24 * J ^ 2 * (RB + 1))))
        (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1)))))
    let Kc : ℝ := (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ
      * (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α)
    let SC : ℝ := 8 * (Q + 4) * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2
    let HN : ℝ := 192 / ε * Real.log (4 * (Q + 1) * SC / δ') + 1
    let FV : ℝ := (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / 4
    let SH : ℝ := 24 * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2
    let HB : ℝ := 384 / ε * ((Q + 1) * SH * Real.log 2 + Real.log (2 / δ')) + 1
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
    O.η ≤ η₀ → η₀ < 1 / 2 → 0 < c → c ≤ 1 → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ →
    2 ≤ Fintype.card α → A ≤ Ae → 1 ≤ k → k ≤ L → cap ≤ k → 0 ≤ t →
    (∀ᵐ w ∂D, Ls ≤ w.toList.length) → 0 < p₀ → (∀ u, D.real {w | prefixOf w k = u} ≤ p₀) →
    pAP ≤ (prefixLaw D Ls).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
    p₀ ≤ capP * ε / 1536 →
    1536 * Real.log (16 * (Q + 1) * Kc / δ') * p₀ ≤ ε →
    6 * J * RB * p₀ ≤ 1 / 2 →
    16 * B₃ ^ 2 * p₀ ≤ δ' →
    2 * ((⌊FV⌋₊ : ℝ) + 1) ^ 3 * (A ^ Ls + 1) ^ ⌊FV⌋₊ * Real.exp (-2 * t ^ 2 / p₀) ≤ δ / 8 →
    ε + ε / 96 + (1025 / 1024 + 3 / 8) * εcov + t ≤ ξT →
    ∃ settings : LoopSettings (Fintype.card σ + 1),
      settings.level3.walkStart = k ∧ (settings.level3.probeBudget : ℝ) ≤ B₃ ∧
      (settings.level3.harvestReadCap : ℝ) ≤ HN ∧ (settings.scanCap : ℝ) ≤ SC ∧
      (∀ (p : LoopDraws Ω α (Fintype.card σ + 1) settings.level3.harvestReadCap
            settings.level3.probeBudget) rounds,
          indecisiveLStarTrieLoop settings cap O (prefixLaw D L) (prefixLaw D Ls) p
            = some rounds →
        rounds.length ≤ Fintype.card σ + 1
          ∧ ((rounds.map fun o => o.prefs.card).sum : ℝ) ≤ (Q + 1) * (6 * (Q + 2) * RB)
          ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ) ≤ (Q + 1) * (PB + 1)) ∧
      (loopMeasure μ α (Fintype.card σ + 1) settings.level3.harvestReadCap
          settings.level3.probeBudget).real
        {p | ¬ RLStarSucceeds settings cap O (prefixLaw D L) (prefixLaw D Ls)
          (prefixLaw D L) M ξT
          (learnerReads (Q + 1) (6 * (Q + 2) * RB) (PB + 1) FV B₃ Q (⌈HB⌉₊ : ℝ)
            ((3 + Nat.clog 2 L₁ : ℕ) : ℝ) A 0) p}
        ≤ δ := by
  intro Q A kN L₁ Ls N ε εcov δ' ι s τ J S B₃ Wm W PB RB capP Kc SC HN FV SH HB
    hL hη hη₀ hc hc1 hδ hδ1 hpAP hξ hA2 hAe hk0 hkL hcap ht hlong hp₀0 hpreD hpAPu hk1 hk2 hk3 hk4
    hk6 hξT
  classical
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
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
  obtain ⟨hε0, hε12, hεc0, hεcov1, hι0, hι8, -, -, hιL⟩ :=
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
  have hKc1 : (1 : ℝ) ≤ Kc := kc_ge_one hQ0 (by linarith)
  have hlogK : 0 ≤ Real.log (16 * (Q + 1) * Kc / δ') :=
    Real.log_nonneg (by
      rw [le_div_iff₀ hδ'0]
      nlinarith only [hδ'1', hQ0, hKc1, mul_nonneg hQ0 (sub_nonneg.2 hKc1)])
  obtain ⟨settings, hwalk, -, hbud, hHN, hSC, hnp, hab, hks, hmain⟩ :=
    loop_simple_checks M O P (fun _ => ∅) η₀ εcov ε δ' pAP k
    hL hη hη₀ hεc0 hεcov1 hε0 hε12 hδ'0 hδ'1' hpAP hp1
    (by show 1 ≤ L + Fintype.card σ + 3; omega)
    hk0 hιL (hpm.trans hk1)
    ((mul_le_mul_of_nonneg_left hpm (mul_nonneg (by norm_num) hlogK)).trans hk2)
    (((mul_le_mul_of_nonneg_left hpm (by positivity)).trans hk4).trans hδ'1'.le)
  have hs0 : 0 < s := by show 0 < 1 / 2 - η₀; linarith
  have hs1 : s ≤ 1 / 2 := by show 1 / 2 - η₀ ≤ 1 / 2; linarith
  have hτ0 : 0 < τ := lt_min (mul_pos (div_pos hεc0 (by norm_num))
    (pow_pos (gateA_pos hη0 hη₀) 2)) hι0
  have hτ1 : τ ≤ 1 := (min_le_right _ _).trans (by linarith)
  have hW1 : 1 ≤ W := le_max_right _ _
  have hδ'1 : δ' ≤ 1 := hδ'1'.le
  have hRB0 : 0 ≤ RB := rb_nonneg hQ0 hs0 hs1 hpAP hp1 hδ'0 hδ'1 hι0 hτ0 hτ1
    (Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith))
  have hm0 : 0 ≤ max (Real.log (20 / τ)) (Real.log (8 * W / δ')) :=
    le_max_of_le_left (Real.log_nonneg (by rw [le_div_iff₀ hτ0]; linarith))
  have hPB0 : 0 ≤ PB := by
    have hl : 0 ≤ Real.log (3 / (δ' / 2)) :=
      Real.log_nonneg (by rw [le_div_iff₀ (by positivity)]; linarith)
    show 0 ≤ (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
      + 120 * Real.log (3 / (δ' / 2)) / pAP
    positivity
  have hFV0 : 0 ≤ FV := by
    show 0 ≤ (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / 4
    positivity
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
  set T := settings.level3.probeBudget
  set NH := settings.level3.harvestReadCap
  set μL := loopMeasure μ α (Fintype.card σ + 1) NH T
  set Nh : ℕ := 2 * max (4 * (3 * ⌊RB⌋₊)) ⌈8 * Real.log (16 * J / δ')⌉₊
  have hpreW : ∀ u, P.D.real {x | prefixOf x settings.level3.walkStart = u} ≤ p₀ := by
    rw [hwalk]; exact hpre
  have hkW : ∀ᵐ x ∂P.D, settings.level3.walkStart ≤ x.toList.length := by
    rw [hwalk]
    exact ae_iff.2 (measure_mono_null (fun x hx => by
      simp only [Set.mem_ofPred_eq, not_le] at hx ⊢
      omega) hTL)
  set Bloop : Set (Ω × (Fin (Fintype.card σ + 1)
      → RoundDraws α (Fintype.card σ + 1) NH T)) :=
    {p | let rounds := idealRounds settings O P.D Dsf (persistDraws p) (Fintype.card σ + 1)
      ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
          o.Checks M O P.D k (1025 / 1024 * εcov) (3 / 8 * εcov) (ε + ε / 96) (fun _ => ∅)
            (ε / (32 * Q)))
        ∧ (∀ o ∈ rounds, o.result.hypothesis.tree.leaves.length ≤ Fintype.card σ + 2)
        ∧ RoundsFresh settings O P.D Dsf (P.L + Fintype.card σ + 3) p
        ∧ (∀ r (hr : r < Fintype.card σ + 1),
          AllHarvested (idealRounds settings O P.D Dsf (persistDraws p) r) →
            RoundGood settings O P.D Dsf (specHistOf (idealRounds settings O P.D Dsf
              (persistDraws p) r)) (persistDraws p ⟨r, hr⟩))
        ∧ RoundsWalkWithin settings O P.D Dsf p Nh ⌈HB⌉₊)} with hBloopd
  set B3 : Fin (Fintype.card σ + 1) → Set (Ω × (Fin (Fintype.card σ + 1)
      → RoundDraws α (Fintype.card σ + 1) NH T)) := fun r =>
    {p | HistOK settings O P.D Dsf p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf p.1 p.2 r).length = r
        ∧ Returns settings O P.D Dsf (decodeHist
            (histCodes settings O P.D Dsf p.1 p.2 r)) (p.1, (p.2 r).1)
        ∧ ¬ L3Good settings.level3.walkStart P.D
          (roundKeys settings P.D Dsf (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r))
            (p.1, (p.2 r).1)) (p.2 r).2} with hB3d
  set BT : Fin (Fintype.card σ + 1) → Set (Ω × (Fin (Fintype.card σ + 1)
      → RoundDraws α (Fintype.card σ + 1) NH T)) := fun r =>
    {p | HistOK settings O P.D Dsf p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf p.1 p.2 r).length = r
        ∧ ∃ i : Fin (Fintype.card σ + 1), ¬ BatchesGood settings.level3.walkStart P.D NH
          settings.scanCap ((p.2 r).1.2.1 (some i))} with hBTd
  set Bmis : Set Ω := {ω | ∃ (F : Finset (FreeMonoid α)) (kl kh : ℕ),
      (∀ v ∈ F, v.toList.length = Ls) ∧ F.card ≤ ⌊FV⌋₊
        ∧ misMean O F kl kh (prefixLaw D L) + t < misFrac O F kl kh (prefixLaw D L) ω} with hBmisd
  have hlenT : ∀ (p : LoopDraws Ω α (Fintype.card σ + 1) NH T) rounds,
      indecisiveLStarTrieLoop settings cap O (prefixLaw D L) (prefixLaw D Ls) p
        = some rounds → rounds.length ≤ Fintype.card σ + 1
          ∧ ((rounds.map fun o => o.prefs.card).sum : ℝ) ≤ (Q + 1) * (6 * (Q + 2) * RB)
          ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ) ≤ (Q + 1) * (PB + 1) := by
    intro p rounds h
    obtain ⟨hlr, hpr, hsu⟩ := trie_rounds_counts settings cap _ _ p.2 _ p.1 _ _ hab rounds h
    have hQ : Q = (Fintype.card σ : ℝ) := rfl
    have e : ∀ f : TrieLoopRound α → ℕ, (rounds.map fun o => (f o : ℝ)).sum
        = ((rounds.map f).sum : ℝ) := fun f => by
      rw [Nat.cast_list_sum, List.map_map]; rfl
    refine ⟨hlr, (e _).trans_le ((Nat.cast_le.2 hpr).trans ?_),
      (e _).trans_le ((Nat.cast_le.2 hsu).trans ?_)⟩
    · calc (((Fintype.card σ + 1) * (6 * (Fintype.card σ + 1 + 1) * ⌊RB⌋₊) : ℕ) : ℝ)
          = (Q + 1) * (6 * (Q + 2) * (⌊RB⌋₊ : ℝ)) := by rw [hQ]; push_cast; ring
        _ ≤ (Q + 1) * (6 * (Q + 2) * RB) :=
          mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left (Nat.floor_le hRB0)
            (by linarith only [hQ0])) (by linarith only [hQ0])
    · calc (((Fintype.card σ + 1) * (⌊PB⌋₊ + 1) : ℕ) : ℝ)
          = (Q + 1) * ((⌊PB⌋₊ : ℝ) + 1) := by rw [hQ]; push_cast; ring
        _ ≤ (Q + 1) * (PB + 1) :=
          mul_le_mul_of_nonneg_left (by linarith only [Nat.floor_le hPB0]) (by linarith only [hQ0])
  refine ⟨settings, hwalk, hbud, hHN, hSC, hlenT, ?_⟩
  have hsub : {p | ¬ RLStarSucceeds settings cap O (prefixLaw D L) (prefixLaw D Ls)
        (prefixLaw D L) M ξT
        (learnerReads (Q + 1) (6 * (Q + 2) * RB) (PB + 1) FV B₃ Q (⌈HB⌉₊ : ℝ)
          ((3 + Nat.clog 2 L₁ : ℕ) : ℝ) A 0) p}
      ⊆ ((Bloop ∪ ⋃ r, B3 r) ∪ ⋃ r, BT r) ∪ Prod.fst ⁻¹' Bmis := by
    intro p hp
    by_contra hn
    simp only [Set.mem_union, Set.mem_iUnion, Set.mem_preimage, not_or, not_exists] at hn
    obtain ⟨⟨⟨hnl, hn3⟩, hnT⟩, hnm⟩ := hn
    obtain ⟨⟨hlen, o, ho, hch⟩, hlv, hfr, hrg, hwk⟩ := not_not.1 hnl
    apply hp
    set rd : FreeMonoid α → ℝ := fun w => O.mq w p.1
    set draws := persistDraws p
    have hloop : loopRounds settings O P.D Dsf draws (Fintype.card σ + 1)
        = some (idealRounds settings O P.D Dsf draws (Fintype.card σ + 1)) :=
      loopRounds_eq_some settings O P.D Dsf draws hrg _
    obtain ⟨e1, e2⟩ := eval_loopRoundsQ O p.1 settings P.D Dsf p.2 (Fintype.card σ + 1)
    rw [show persistDraws (p.1, p.2) = draws from rfl, hloop, Option.map_some] at e1
    have hread : ∀ o ∈ idealRounds settings O P.D Dsf draws (Fintype.card σ + 1),
        o.ReadsAt O p.1 := e2 _ hloop
    have hprem := trieRoundGood_of O P Dsf settings p hrg
      (fun r hr h => hn3 ⟨r, hr⟩ h) (fun r hr h => hnT ⟨r, hr⟩ h)
    obtain ⟨t1, t2⟩ := trieLoopAgrees_holds rd settings cap P.D Dsf p.2 (hcap.trans_eq hwalk.symm) hprem
    rw [e1, Option.map_some] at t1
    have hcons : o.result.ending = .consistent := hch.1.1
    have hoK : o ∈ idealRounds settings O P.D Dsf draws (Fintype.card σ + 1) :=
      List.mem_of_getLast? ho
    have hro : o.read = o.data.read O p.1 := hread o hoK
    -- the family: strings of length `Ls`, at most `⌊FV⌋₊` of them
    obtain ⟨r, hr, -, tt, hst, -, hOk, -, hF, -, -⟩ :=
      stageSel_last settings O P Dsf (P.L + Fintype.card σ + 3) p hfr o ho hcons
    have hlenF : ∀ v ∈ o.family, v.toList.length = Ls := fun v hv =>
      hOk.1.1 ⟨r, hr⟩ le_rfl v (hOk.2.1 (hF ▸ hv))
    have hcardF : o.family.card ≤ ⌊FV⌋₊ := by
      obtain ⟨t', h4, ht'⟩ := Option.map_eq_some_iff.1 hst
      obtain ⟨rfl, -⟩ := Prod.mk.inj ht'
      obtain ⟨c4, B4, F4, K4⟩ := t'
      obtain ⟨-, -, -, hret, hB4, hF4, -⟩ :=
        (stage4_eq_some settings O P.D Dsf r hr p.1 p.2 c4 B4 F4 K4).1 h4
      have hmem := (Classical.epsilon_spec hret).1
      change roundState settings O P.D Dsf (decodeHist c4) (p.1, (p.2 ⟨r, hr⟩).1)
        ∈ _ at hmem
      rw [hB4] at hmem
      have hk : B4.k ≤ ⌊FV⌋₊ + 1 := hks _ B4 hmem
      have hc := card_clusterAt_le O.mq (loopPopulations (Fintype.card σ + 1)
        (decodeHist c4).pops.length) (idealRun P.D Dsf settings.level3.walkStart NH
          (decodeHist c4) (p.1, (p.2 ⟨r, hr⟩).1)) B4
      rw [hF4] at hc
      have hF' : F4 = o.family := hF
      rw [← hF']
      omega
    refine ⟨⟨_, t1, ⟨by simpa using hlen, liftRound settings.level3.walkStart o.data,
      by simp [List.getLast?_map, Option.mem_def.1 ho], hcons, ?_⟩, ?_⟩, ?_⟩
    · show (prefixLaw D L).real {w | ¬ o.result.hypothesis.TrieClassifies (o.data.read O p.1)
          (completeTrie settings.level3.walkStart) w (M.eval w.toList ∈ M.accept)} ≤ ξT
      rw [← hro, hwalk]
      have eL : {w | ¬ o.result.hypothesis.TrieClassifies o.read (completeTrie k) w
            (M.eval w.toList ∈ M.accept)}
          = {w | ¬ o.result.hypothesis.TrieClassifies o.read (completeTrie k) w (w ∈ O.L)} := by
        ext w
        simp only [Set.mem_setOf_eq, hL]
      rw [eL]
      refine (trieSinglePieceCorrect o.read o.result.hypothesis (completeTrie k)
        (rootOne_of_classSet _ _ hch.2.2.1) O.L (prefixLaw D L)).trans ?_
      have hdis : (prefixLaw D L).real
          (Random.trieDisagreeSet o.read (completeTrie k) o.result.hypothesis)
          ≤ ε + ε / 96 := by
        refine le_trans ?_ hch.2.1
        have hsub : Random.trieDisagreeSet o.read (completeTrie k) o.result.hypothesis
            ⊆ disagreeSet o.read k o.result.hypothesis ∪ {x | x.toList.length ≠ L} := by
          intro x hx
          by_cases hxl : x.toList.length = L
          · left
            have := Random.walkStartIn_completeTrie (K := k) (x := x) (by omega)
            simp only [Random.trieDisagreeSet, Set.mem_setOf_eq, this] at hx
            exact hx
          · exact Or.inr hxl
        refine (measureReal_mono hsub (measure_ne_top _ _)).trans
          ((measureReal_union_le _ _).trans ?_)
        have h0 : (prefixLaw D L).real {x : FreeMonoid α | x.toList.length ≠ L} = 0 := by
          rw [measureReal_def, hTL, ENNReal.toReal_zero]
        rw [h0, add_zero]
      have hmis : (prefixLaw D L).real (Random.misreadSet o.read O.L)
          ≤ (1025 / 1024 + 3 / 8) * εcov + t := by
        have e : (prefixLaw D L).real (Random.misreadSet o.read O.L)
            = misFrac O o.family o.state.lo (o.state.hi + 1) (prefixLaw D L) p.1 := by
          rw [hro]; rfl
        rw [e]
        have h1 : misFrac O o.family o.state.lo (o.state.hi + 1) (prefixLaw D L) p.1
            ≤ misMean O o.family o.state.lo (o.state.hi + 1) (prefixLaw D L) + t :=
          not_lt.1 fun h => hnm ⟨o.family, o.state.lo, o.state.hi + 1, hlenF, hcardF, h⟩
        have h2 := hch.1.2.1
        have h3 := hch.1.2.2
        rw [misMean_split O _ _ _ hTL] at h1
        change ∫ x, misreadProb O o.family o.state.lo (o.state.hi + 1) x ∂(prefixLaw D L)
          ≤ 1025 / 1024 * εcov at h2
        change ∫ x, readProb O o.family o.state.lo (o.state.hi + 1) x .undecided
          ∂(prefixLaw D L) ≤ 3 / 8 * εcov at h3
        linarith
      linarith
    · intro o' ho'
      obtain ⟨o'', ho'', rfl⟩ := List.mem_map.1 ho'
      obtain ⟨o₃, ho₃, rfl⟩ := List.mem_map.1 ho''
      exact hlv o₃ ho₃
    · refine ⟨(trieLoopRoundsQ settings cap P.D Dsf p.2 (Fintype.card σ + 1)).reads rd, ?_,
        fun o' ho' => (Query.eval_congr _ _ ho').1⟩
      rw [t2]
      have hlv' : ∀ o ∈ (idealRounds settings O P.D Dsf draws (Fintype.card σ + 1)).map
          SpecRound.data, o.result.hypothesis.tree.leaves.length ≤ Fintype.card σ + 2 := by
        intro o' ho'
        obtain ⟨o'', ho'', rfl⟩ := List.mem_map.1 ho'
        exact hlv o'' ho''
      have h := card_reads_loopRoundsQ_le O p.1 settings P.D Dsf p.2 P.length_le hab hks e1 hlv'
        (harvestProbesWithin_of O settings P.D Dsf p Nh ⌈HB⌉₊ hrg hwk) le_rfl
      have h' := (Nat.cast_le (α := ℝ)).2 h
      refine (h'.trans (le_of_eq (by rw [zero_mul, add_zero]))).trans
        ((reads_le_learnerReads (Fintype.card σ + 1) (6 * (Fintype.card σ + 1 + 1) * ⌊RB⌋₊)
          (⌊PB⌋₊ + 1) ⌊FV⌋₊ T ⌈HB⌉₊ L₁ (Fintype.card α) 0 (Fintype.card σ)).trans ?_)
      have hQ : Q = (Fintype.card σ : ℝ) := rfl
      have ha : ((6 * (Fintype.card σ + 1 + 1) * ⌊RB⌋₊ : ℕ) : ℝ) ≤ 6 * (Q + 2) * RB := by
        have e : ((6 * (Fintype.card σ + 1 + 1) * ⌊RB⌋₊ : ℕ) : ℝ) = 6 * (Q + 2) * (⌊RB⌋₊ : ℝ) := by
          rw [hQ]; push_cast; ring
        rw [e]
        exact mul_le_mul_of_nonneg_left (Nat.floor_le hRB0) (by linarith)
      have hb : ((⌊PB⌋₊ + 1 : ℕ) : ℝ) ≤ PB + 1 := by push_cast; linarith [Nat.floor_le hPB0]
      exact learnerReads_mono (by positivity) (by positivity) (by positivity) (by positivity)
        (by positivity) (by positivity) (by positivity) (by positivity) (by positivity)
        (by norm_num) (by rw [hQ]; push_cast; rfl) ha hb (Nat.floor_le hFV0) hbud
        le_rfl le_rfl le_rfl le_rfl (by norm_num)
  have hKQ : ((Fintype.card σ + 1 : ℕ) : ℝ) = Q + 1 := by push_cast; rfl
  have hBl : μL.real Bloop ≤ δ / 2 := by
    refine hD.trans (le_of_eq ?_)
    show (Q + 1) * (4 * (δ / (8 * (Q + 1)))) = δ / 2
    field_simp
    ring
  have hK3 : ((6 * (Fintype.card σ + 1 + 1) * ⌊RB⌋₊ : ℕ) : ℝ) * p₀ ≤ 1 / 2 := by
    refine le_trans ?_ hk3
    have e : ((6 * (Fintype.card σ + 1 + 1) * ⌊RB⌋₊ : ℕ) : ℝ) = 6 * J * (⌊RB⌋₊ : ℝ) := by
      show _ = 6 * ((Fintype.card σ : ℝ) + 2) * _
      push_cast; ring
    rw [e]
    gcongr
    exact Nat.floor_le hRB0
  have h3 : μL.real (⋃ r, B3 r) ≤ 0 :=
    (measureReal_iUnion_fintype_le _).trans (Finset.sum_nonpos fun r _ =>
      trie_l3_round_le O P Dsf settings hnp hpreW hkW hK3 r.1 r.2)
  have hTop : μL.real (⋃ r, BT r) ≤ 0 :=
    (measureReal_iUnion_fintype_le _).trans (Finset.sum_nonpos fun r _ =>
      trie_batches_round_le O P Dsf settings hkW r.1 r.2)
  have hmis : μL.real (Prod.fst ⁻¹' Bmis) ≤ δ / 8 := by
    have e : μL.real (Prod.fst ⁻¹' Bmis) = μ.real Bmis := by
      change (μ.prod (Measure.pi fun _ : Fin (Fintype.card σ + 1) =>
        roundLaw α (Fintype.card σ + 1) NH T)).real _ = _
      rw [← Set.prod_univ, measureReal_def, measureReal_def, Measure.prod_prod, measure_univ,
        mul_one]
    rw [e]
    exact (misread_union_le O hTL (fun v => prefixLaw_singleton_le D hkL hpreD _) Ls ⌊FV⌋₊
      ht).trans hk6
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans ?_
  have u1 := measureReal_union_le (μ := μL) ((Bloop ∪ ⋃ r, B3 r) ∪ ⋃ r, BT r) (Prod.fst ⁻¹' Bmis)
  have u2 := measureReal_union_le (μ := μL) (Bloop ∪ ⋃ r, B3 r) (⋃ r, BT r)
  have u3 := measureReal_union_le (μ := μL) Bloop (⋃ r, B3 r)
  linarith

end OrthoDFA

end
