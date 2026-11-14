import RLStar.Claims.RLStar
import RLStar.Proofs.TrieMisread
import RLStar.Proofs.Trie
import RLStar.Proofs.LoopSimple
import RLStar.Proofs.LoopReadsArith
import RLStar.Proofs.LazyDFA
import RLStar.Proofs.LazyTable
import RLStar.Proofs.LazyShape

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

section Last

open Query

private lemma epochQ_root (read : FreeMonoid α → ARU) (C : Random.Level3Settings) (K : ℕ)
    (X : Finset (FreeMonoid α)) {P : ℕ} (d : Fin P → FreeMonoid α) :
    RootOne ((Random.epochQ C K X d).eval read).hypothesis.tree := by
  unfold Random.epochQ
  rw [eval_subst]
  exact rootOne_of_classSet _ _
    (Random.Spur.tree_classSet_of_leaves _ _ _ (Nat.le_add_right _ 2))

private lemma lazyEpochsQ_root (read : FreeMonoid α → ARU) (C : Random.Level3Settings)
    (K nh : ℕ) {J b : ℕ} : ∀ (e pk : ℕ) (ds : List (Random.EpochDraws α C.probeBudget J b))
      (X : Finset (FreeMonoid α)) (ls : Random.LazyLists α),
      ((Random.lazyEpochsQ C K nh e pk ds X ls).eval read).ending = .consistent →
      RootOne ((Random.lazyEpochsQ C K nh e pk ds X ls).eval read).hypothesis.tree
  | _, _, [], X, ls, h => by simp [Random.lazyEpochsQ] at h
  | e, pk, d :: ds, X, ls, h => by
    simp only [Random.lazyEpochsQ, eval_bind] at h ⊢
    have hr := epochQ_root read (Random.epochSettings C e) K X d.1
    generalize (Random.epochQ (Random.epochSettings C e) K X d.1).eval read = res at h hr ⊢
    rcases res with ⟨hyp, _ | s | _⟩
    · exact hr
    · simp only [eval_bind] at h ⊢
      generalize (Random.topUpQ K nh hyp s (List.ofFn d.2) X (ls s)).eval read = t at h ⊢
      rcases t with _ | ⟨_ | _, rest⟩
      · simp at h
      · simp at h
      · exact lazyEpochsQ_root read C K nh _ _ ds _ _ h
    · simp at h

/-- A consistent lazy round of lists of one string from the trie of `P` keeps its trie shorter than
`K`, roots its tree at the empty string, and keeps the trie of at most `K |P|` strings. -/
private lemma lazyRoundQ_consistent (read : FreeMonoid α → ARU) (C : Random.Level3Settings)
    (K : ℕ) {E J b : ℕ} (P : Finset (FreeMonoid α))
    (d : Fin E → Random.EpochDraws α C.probeBudget J b)
    (hc : ((Random.lazyRoundQ C K 1 (Random.prefixTrie K P) d).eval read).ending
      = .consistent) :
    (∀ u ∈ ((Random.lazyRoundQ C K 1 (Random.prefixTrie K P) d).eval read).expanded,
        u.toList.length < K)
      ∧ RootOne ((Random.lazyRoundQ C K 1 (Random.prefixTrie K P) d).eval read).hypothesis.tree
      ∧ ((Random.lazyRoundQ C K 1 (Random.prefixTrie K P) d).eval read).expanded.card
        ≤ K * P.card := by
  obtain ⟨hok, -, hcard, -, hlist, -, -, -⟩ :=
    lazyRoundShape_holds read C K 1 E J b (Random.prefixTrie K P) d one_pos (trieOK_prefixTrie K P)
  refine ⟨hok.1, lazyEpochsQ_root read C K 1 0 0 (List.ofFn d) (Random.prefixTrie K P)
    (fun _ => []) hc, ?_⟩
  have h1 := hlist hc .ends
  have h2 := hlist hc .middles
  have h4 : (((Random.lazyRoundQ C K 1 (Random.prefixTrie K P) d).eval read).lists .ends).length
      + (((Random.lazyRoundQ C K 1 (Random.prefixTrie K P) d).eval read).lists .middles).length
      = 0 := by omega
  rw [h4, mul_zero, add_zero] at hcard
  exact hcard.trans (card_prefixTrie_le K P)

private lemma lazyReads_mono {K a b f Nr A nX q K' a' b' f' Nr' A' nX' q' : ℝ} (hK0 : 0 ≤ K)
    (ha0 : 0 ≤ a) (hb0 : 0 ≤ b) (hf0 : 0 ≤ f) (hNr : 0 ≤ Nr) (hA0 : 0 ≤ A) (hnX0 : 0 ≤ nX)
    (hq0 : 0 ≤ q) (hK : K ≤ K') (ha : a ≤ a') (hb : b ≤ b') (hf : f ≤ f') (hN : Nr ≤ Nr')
    (hA : A ≤ A') (hnX : nX ≤ nX') (hq : q ≤ q') :
    lazyReads K a b f Nr A nX q ≤ lazyReads K' a' b' f' Nr' A' nX' q' := by
  unfold lazyReads
  have h1 : a * (b + 1) ≤ a' * (b' + 1) := mul_le_mul ha (by linarith) (by linarith) (by linarith)
  have h2 : f * Nr ≤ f' * Nr' := mul_le_mul hf hN hNr (by linarith)
  have h3 : K * (a * (b + 1) + f * Nr) ≤ K' * (a' * (b' + 1) + f' * Nr') :=
    mul_le_mul hK (add_le_add h1 h2)
      (add_nonneg (mul_nonneg ha0 (by linarith)) (mul_nonneg hf0 hNr)) (by linarith)
  have h4 : A * nX + 1 ≤ A' * nX' + 1 := by
    linarith [mul_le_mul hA hnX hnX0 (by linarith : 0 ≤ A')]
  have h5 : (A * nX + 1) * (q + 2) ≤ (A' * nX' + 1) * (q' + 2) :=
    mul_le_mul h4 (by linarith) (by linarith) (by linarith [mul_nonneg hA0 hnX0])
  have h6 : f * ((A * nX + 1) * (q + 2)) ≤ f' * ((A' * nX' + 1) * (q' + 2)) :=
    mul_le_mul hf h5 (mul_nonneg (by linarith [mul_nonneg hA0 hnX0]) (by linarith))
      (by linarith)
  linarith

end Last

variable [Nonempty α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

omit [DecidableEq α] [Nonempty α] [IsProbabilityMeasure μ] in
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
/-- The lazy loop over `D`'s first `L` letters: its DFA misjudges at most `ξT` of them from a trie
of at most `nX` strings, reading at most `lazyReads` strings, but for chance `δ`. -/
theorem lazy_loop_core (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L k : ℕ)
    (η₀ c δ pAP ξ ξT t p₀ ι W Tu tw : ℝ) (E J b Nr Nh : ℕ) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] :
    let Q : ℝ := Fintype.card σ
    let A : ℝ := Fintype.card α
    let kN : ℕ := (L + 1) / 2
    let L₁ : ℕ := L + Fintype.card σ + 3
    let Ls : ℕ := L₁ + Fintype.card σ + 3
    let N : ℝ := L₁ + 1 - kN
    let ε : ℝ := c / 8
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
    let Kt : ℝ := (1 + (Q + 1) ^ 3 * (A + 1)) ^ Fintype.card σ
      * (Q + 3) ^ ((Fintype.card σ + 2) * (Fintype.card α + 1))
    let capP : ℝ := min (min (min (s * min (εcov / 2 * s) ι / (2 ^ 20 * (RB + 1)))
          (ι ^ 2 / (2 ^ 28 * Real.log (128 * Jp * (PB + 1) * (3 + 2 * Real.logb 2 (RB + 1)) / δ'))))
        (ι / 2 / (8 * Jp * (RB + 1))))
      (min (min (pAP / (2 ^ 18 * (PB + 1))) (ι / (2 ^ 24 * Jp ^ 2 * (RB + 1))))
        (min (pAP / (2 * Jp * (PB + 2)))
          (min (1 / (6 * Jp * (RB + 1)))
            (1 / (8 * Real.log (8 * (Q + 1) * Kt * (E * (E + 1)) / (δ' / 8)))))))
    let θ : ℝ := capP * ε / 192
    let SC : ℝ := 8 * (Q + 4) * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2
    let HR : ℝ := Real.log (4 * (Q + 1) * (SC + 1) / δ') / (ε / 384) + 1
    let FV : ℝ := (50 * (max (Real.log (20 / τ)) (Real.log (8 * W / δ')) + 3) / s ^ 2 + 20 / s + 28)
        / 4
    let nX : ℝ := k * (6 * Jp * RB + Nh)
    let Ns : ℝ := Nr + (Q + 1) * (SC * (HR * (Random.Kappa.probeSifts (L₁ + k) * (Q + 1))))
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
    4 * (1 / 2 - η₀) ≤ O.gap → η₀ < 1 / 2 →
    2 * εcov ≤ (prefixLaw D L).real {p | p ∈ O.L} → 2 * εcov ≤ (prefixLaw D L).real {p | p ∉ O.L} →
    0 < c → c ≤ 1 → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ → 1 ≤ k → k ≤ L → 0 ≤ t →
    (∀ᵐ w ∂D, Ls ≤ w.toList.length) → 0 < p₀ → (∀ u, D.real {w | prefixOf w k = u} ≤ p₀) →
    pAP ≤ (prefixLaw D Ls).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
    p₀ ≤ capP * ε / 12288 →
    2 * ((⌊FV⌋₊ : ℝ) + 1) ^ 3 * (A ^ Ls + 1) ^ ⌊FV⌋₊ * Real.exp (-2 * t ^ 2 / p₀) ≤ δ / 8 →
    0 < ι → ι ≤ εcov / 8 → 1 ≤ W →
    Real.log (4 * k / (θ * δ')) / θ ≤ Nh →
    2 * k / (ε / 48) + 1 ≤ E → k + 2 ≤ J →
    E * J * (1 - ε / 192) ^ b ≤ δ' / 4 →
    E * ((B₃ + J * b) * (Random.Kappa.probeSifts (L₁ + k) * (Q + 3))
      + (Q + 2) * ((Q + 2) * (A + 1) * (Q + 2))) ≤ Nr →
    Nr ≤ W → (L₁ + 1) * (Q + 2) ≤ tw * W → 0 < tw → tw ≤ ε / 12288 → 0 < Tu →
    2 * (Q + 1) * (288 / ε + 16 * (2 + Nat.clog 2 (L₁ + k))) * Tu + Q * (3 * ι) / Tu < 7 / 8 →
    ε + ε / 96 + tw + (1025 / 1024 + 3 / 8) * εcov + t ≤ ξT →
    ∃ settings : LoopSettings (Fintype.card σ + 1),
      settings.level3.walkStart = k ∧ (settings.level3.probeBudget : ℝ) ≤ B₃ ∧
      (settings.scanCap : ℝ) ≤ SC ∧ settings.listSize = 1 ∧ settings.heavyDraws = Nh
      ∧ settings.epochs = E
      ∧ settings.blocks = J ∧ settings.blockSize = b ∧
      (∀ (p : settings.Draws Ω α) rounds,
          indecisiveLStarLoop settings O (prefixLaw D L) (prefixLaw D Ls) p = some rounds →
        rounds.length ≤ Fintype.card σ + 1
          ∧ ((rounds.map fun o => o.prefs.card).sum : ℝ) ≤ (Q + 1) * (6 * (Q + 2) * RB)
          ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ) ≤ (Q + 1) * (PB + 1)) ∧
      (settings.measure μ α).real
        {p | ¬ RLStarSucceeds settings O (prefixLaw D L) (prefixLaw D Ls)
          (prefixLaw D L) M ξT nX (lazyReads (Q + 1) (6 * (Q + 2) * RB) (PB + 1) FV Ns A nX Q) p}
        ≤ δ := by
  intro Q A kN L₁ Ls N ε εcov δ' s τ Jp S B₃ PB RB Kt capP θ SC HR FV nX Ns
    hL hgap hη₀ hbalL hbalN hc hc1 hδ hδ1 hpAP hξ hk0 hkL ht hlong hp₀0 hpreD hpAPu
    hk1 hk6 hι0 hι8 hW1 hNh hE hJ hb hNr hNrW htwW htw htwε hTu hslope hξT
  classical
  have hη0 : 0 ≤ η₀ := by have := gap_le_one O; linarith
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hQ0 : (0 : ℝ) ≤ Q := by linarith
  have hkL₁ : kN ≤ L₁ := by show (L + 1) / 2 ≤ L + Fintype.card σ + 3; omega
  have hNe : N = ((L₁ : ℕ) : ℝ) + 1 - kN := rfl
  have hN1 : (1 : ℝ) ≤ N := by
    have : (kN : ℝ) ≤ L₁ := by exact_mod_cast hkL₁
    rw [hNe]; linarith
  have hN0 : 0 < N := by linarith
  have hε0 : 0 < ε := div_pos hc (by norm_num)
  have hε12 : ε ≤ 1 / 2 := (div_le_iff₀ (by norm_num)).2 (by linarith)
  have hεc0 : 0 < εcov := lt_min (div_pos hc (by positivity)) (div_pos hξ (by positivity))
  have hεcov1 : εcov ≤ 1 := by
    refine (min_le_left _ _).trans ?_
    rw [div_le_one (by positivity)]
    linarith
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
  have hkD : ∀ᵐ x ∂P.D, k ≤ x.toList.length :=
    ae_iff.2 (measure_mono_null (fun x hx => by
      simp only [Set.mem_ofPred_eq, not_le] at hx ⊢
      omega) hTL)
  obtain ⟨settings, hwalk, -, hbud, hSC, ⟨hnh', hhd⟩, hHR, -, hE', hJ', hb', hnp, hab, hks,
      hmain⟩ :=
    loop_simple_checks M O P η₀ εcov ε δ' pAP ι W Tu tw k E J b Nr Nh
    hL hgap hη₀ hεc0 hεcov1 hbalL hbalN hε0 hε12 hδ'0 hδ'1' hpAP hp1
    (by show 1 ≤ L + Fintype.card σ + 3; omega)
    hk0 hkD (hpm.trans hk1) hι0 hι8 hW1 hNh hE hJ hb hNr hNrW htwW htw htwε hTu hslope
  have hs0 : 0 < s := by show 0 < 1 / 2 - η₀; linarith
  have hs1 : s ≤ 1 / 2 := by show 1 / 2 - η₀ ≤ 1 / 2; linarith
  have hτ0 : 0 < τ := lt_min (mul_pos (div_pos hεc0 (by norm_num))
    (pow_pos (gateA_pos hη0 hη₀) 2)) hι0
  have hτ1 : τ ≤ 1 := (min_le_right _ _).trans (by linarith)
  have hδ'1 : δ' ≤ 1 := hδ'1'.le
  have hRB0 : 0 ≤ RB := rb_nonneg hQ0 hs0 hs1 hpAP hp1 hδ'0 hδ'1 hι0 hτ0 hτ1
    (Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith))
  have hm0 : 0 ≤ max (Real.log (20 / τ)) (Real.log (8 * W / δ')) :=
    le_max_of_le_left (Real.log_nonneg (by rw [le_div_iff₀ hτ0]; linarith))
  have hPB0 : 0 ≤ PB := by
    have hl : 0 ≤ Real.log (3 / (δ' / 2)) :=
      Real.log_nonneg (by rw [le_div_iff₀ (by positivity)]; linarith)
    show 0 ≤ (50 * (max (Real.log (20 / τ)) (Real.log (8 * W / δ')) + 3) / s ^ 2 + 20 / s + 28) /
        pAP
      + 120 * Real.log (3 / (δ' / 2)) / pAP
    positivity
  have hFV0 : 0 ≤ FV := by
    show 0 ≤ (50 * (max (Real.log (20 / τ)) (Real.log (8 * W / δ')) + 3) / s ^ 2 + 20 / s + 28) / 4
    positivity
  have hcap0 : 0 < capP := by
    by_contra hc0
    push Not at hc0
    have : capP * ε / 12288 ≤ 0 := by
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
    intro p rounds h
    obtain ⟨rs, hrs, rfl⟩ := indecisiveLStarLoop_some O h
    obtain ⟨hlr, hpr, hsu⟩ := loop_reads_le_op settings O (prefixLaw D L) (prefixLaw D Ls)
      (persistDraws p) _ _ hab rs hrs
    have hQ : Q = (Fintype.card σ : ℝ) := rfl
    have hpr' : ((rs.map SpecRound.data).map fun o => o.prefs.card).sum
        ≤ (Fintype.card σ + 1) * (6 * (Fintype.card σ + 1 + 1) * ⌊RB⌋₊) := by
      rw [List.map_map]; exact hpr
    have hsu' : ((rs.map SpecRound.data).map fun o => o.suffs.card).sum
        ≤ (Fintype.card σ + 1) * (⌊PB⌋₊ + 1) := by
      rw [List.map_map]; exact hsu
    have e : ∀ f : LoopRound α → ℕ, ((rs.map SpecRound.data).map fun o => (f o : ℝ)).sum
        = ((((rs.map SpecRound.data).map f).sum : ℕ) : ℝ) := fun f => by
      rw [Nat.cast_list_sum]; simp [Function.comp_def]
    refine ⟨by rw [List.length_map]; exact hlr, (e fun o => o.prefs.card).trans_le ?_,
      (e fun o => o.suffs.card).trans_le ?_⟩
    · calc ((((rs.map SpecRound.data).map fun o => o.prefs.card).sum : ℕ) : ℝ)
          ≤ (((Fintype.card σ + 1) * (6 * (Fintype.card σ + 1 + 1) * ⌊RB⌋₊) : ℕ) : ℝ) :=
            Nat.cast_le.2 hpr'
        _ = (Q + 1) * (6 * (Q + 2) * (⌊RB⌋₊ : ℝ)) := by rw [hQ]; push_cast; ring
        _ ≤ (Q + 1) * (6 * (Q + 2) * RB) :=
          mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left (Nat.floor_le hRB0)
            (by linarith only [hQ0])) (by linarith only [hQ0])
    · calc ((((rs.map SpecRound.data).map fun o => o.suffs.card).sum : ℕ) : ℝ)
          ≤ (((Fintype.card σ + 1) * (⌊PB⌋₊ + 1) : ℕ) : ℝ) := Nat.cast_le.2 hsu'
        _ = (Q + 1) * ((⌊PB⌋₊ : ℝ) + 1) := by rw [hQ]; push_cast; ring
        _ ≤ (Q + 1) * (PB + 1) :=
          mul_le_mul_of_nonneg_left (by linarith only [Nat.floor_le hPB0]) (by linarith only [hQ0])
  refine ⟨settings, hwalk, hbud, hSC, hnh', hhd, hE', hJ', hb', hlenT, ?_⟩
  -- off both, the loop's last round classifies through its table, reading few strings
  have hsub : {p | ¬ RLStarSucceeds settings O (prefixLaw D L) (prefixLaw D Ls)
      (prefixLaw D L) M ξT nX (lazyReads (Q + 1) (6 * (Q + 2) * RB) (PB + 1) FV Ns A nX Q) p}
      ⊆ Bloop ∪ Prod.fst ⁻¹' Bmis := by
    intro p hp
    by_contra hn
    simp only [Set.mem_union, Set.mem_preimage, not_or] at hn
    obtain ⟨hnl, hnm⟩ := hn
    obtain ⟨⟨hlen, o, ho, hch⟩, hlv, hfr, hrg⟩ := not_not.1 hnl
    apply hp
    have hloop : loopRounds settings O P.D Dsf (persistDraws p) (Fintype.card σ + 1)
        = some (idealRounds settings O P.D Dsf (persistDraws p) (Fintype.card σ + 1)) :=
      idealRounds_mem_loop settings O Dsf P.D (persistDraws p) hrg
    have hind : indecisiveLStarLoop settings O (prefixLaw D L) (prefixLaw D Ls) p
        = some ((idealRounds settings O P.D Dsf (persistDraws p) (Fintype.card σ + 1)).map
          SpecRound.data) := by
      have := indecisiveLStarLoop_eq O settings P.D Dsf p
      rw [hloop, Option.map_some] at this
      exact this
    have hoK : o ∈ idealRounds settings O P.D Dsf (persistDraws p) (Fintype.card σ + 1) :=
      List.mem_of_getLast? ho
    have hro : o.read = o.data.read O p.1 :=
      loopRounds_read O settings P.D Dsf p _ (Option.mem_def.2 hloop) o hoK
    have hcons : o.result.ending = .consistent := hch.1.1
    obtain ⟨hh, x, u, hox⟩ :=
      (loopRounds_roundOf settings O P.D Dsf (persistDraws p) _ _ hloop).2 o hoK
    have hres : (Random.lazyRoundQ settings.level3 settings.level3.walkStart settings.listSize
        (Random.prefixTrie settings.level3.walkStart (o.prefs ∪ heavySet P.D u.1))
        (lazyProbes (avoidLaw P.D settings.level3.walkStart
          (clusterKeys (settings.schedule (loopPopulations (Fintype.card σ + 1) hh.pops.length))
            (loopPopulations (Fintype.card σ + 1) hh.pops.length) settings.level3.walkStart x))
          u.2)).eval o.read = o.result := by
      subst hox; rfl
    rw [hnh'] at hres
    have hfacts := lazyRoundQ_consistent o.read settings.level3 settings.level3.walkStart
      (o.prefs ∪ heavySet P.D u.1)
      (lazyProbes (avoidLaw P.D settings.level3.walkStart
        (clusterKeys (settings.schedule (loopPopulations (Fintype.card σ + 1) hh.pops.length))
          (loopPopulations (Fintype.card σ + 1) hh.pops.length) settings.level3.walkStart x)) u.2)
      (by rw [hres]; exact hcons)
    rw [hres, hwalk] at hfacts
    obtain ⟨hXK, hroot, hXc⟩ := hfacts
    have hHc : (heavySet P.D u.1).card ≤ Nh := by
      unfold heavySet
      refine Finset.card_image_le.trans ?_
      rw [Finset.card_univ, Fintype.card_fin, hhd]
    have hpk : ∀ o' ∈ (idealRounds settings O P.D Dsf (persistDraws p)
        (Fintype.card σ + 1)).map SpecRound.data, o'.result.peak ≤ Fintype.card σ + 2 := by
      intro o' ho'
      obtain ⟨o'', ho'', rfl⟩ := List.mem_map.1 ho'
      exact hlv o'' ho''
    have hlast : ((idealRounds settings O P.D Dsf (persistDraws p) (Fintype.card σ + 1)).map
        SpecRound.data).getLast? = some o.data := by
      rw [List.getLast?_map, Option.mem_def.1 ho, Option.map_some]
    obtain ⟨nXn, hnXn⟩ : ∃ n : ℕ, n = k * (6 * (Fintype.card σ + 1 + 1) * ⌊RB⌋₊ + Nh) :=
      ⟨_, rfl⟩
    have hX : ∀ o' ∈ ((idealRounds settings O P.D Dsf (persistDraws p)
        (Fintype.card σ + 1)).map SpecRound.data).getLast?, o'.result.expanded.card ≤ nXn := by
      intro o' ho'
      rw [hlast, Option.mem_some_iff] at ho'
      subst ho'
      rw [hnXn]
      refine hXc.trans (Nat.mul_le_mul_left _ ((Finset.card_union_le _ _).trans
        (Nat.add_le_add ?_ hHc)))
      rw [hox]
      exact (hab _ x).1
    have hQ : Q = (Fintype.card σ : ℝ) := rfl
    have hnX : (nXn : ℝ) ≤ nX := by
      have e : (nXn : ℝ) = k * (6 * (Q + 2) * (⌊RB⌋₊ : ℝ) + Nh) := by
        rw [hnXn, hQ]; push_cast; ring
      rw [e]
      show (k : ℝ) * (6 * (Q + 2) * (⌊RB⌋₊ : ℝ) + Nh) ≤ k * (6 * (Q + 2) * RB + Nh)
      have : 6 * (Q + 2) * (⌊RB⌋₊ : ℝ) ≤ 6 * (Q + 2) * RB :=
        mul_le_mul_of_nonneg_left (Nat.floor_le hRB0) (by linarith)
      exact mul_le_mul_of_nonneg_left (by linarith) (Nat.cast_nonneg _)
    refine ⟨⟨_, hind, ⟨by simpa using hlen, o.data, hlast, hcons, ?_,
      (Nat.cast_le.2 (hX o.data (Option.mem_def.2 hlast))).trans hnX⟩, hpk⟩, ?_⟩
    · show (prefixLaw D L).real {w | ¬ o.result.hypothesis.LazyClassifies
          ((Random.lazyTableQ settings.level3.walkStart o.result.hypothesis.tree
            o.result.expanded).eval (o.data.read O p.1)) o.result.expanded w
            (M.eval w.toList ∈ M.accept)} ≤ ξT
      rw [← hro, hwalk]
      have eL : {w | ¬ o.result.hypothesis.LazyClassifies
            ((Random.lazyTableQ k o.result.hypothesis.tree o.result.expanded).eval o.read)
            o.result.expanded w (M.eval w.toList ∈ M.accept)}
          = {w | ¬ o.result.hypothesis.LazyClassifies
            ((Random.lazyTableQ k o.result.hypothesis.tree o.result.expanded).eval o.read)
            o.result.expanded w (w ∈ O.L)} := by
        ext w
        simp only [Set.mem_ofPred_eq, hL]
      rw [eL]
      refine (lazyTableCorrect_holds o.read o.result.hypothesis k o.result.expanded hXK hroot
        O.L (prefixLaw D L) hkD).trans ?_
      have hdis : (prefixLaw D L).real
          (Random.lazyDisagree o.read k o.result.expanded o.result.hypothesis)
          ≤ ε + ε / 96 + tw := hch.2.1
      have hmis : (prefixLaw D L).real (Random.misreadSet o.read O.L)
          ≤ (1025 / 1024 + 3 / 8) * εcov + t := by
        have e : (prefixLaw D L).real (Random.misreadSet o.read O.L)
            = misFrac O o.family o.state.lo (o.state.hi + 1) (prefixLaw D L) p.1 := by
          rw [hro]; rfl
        rw [e]
        obtain ⟨hlenF, -, -, -⟩ :=
          family_of_fresh settings O P.D Dsf P.length_le (P.L + Fintype.card σ + 3) p hfr o hoK
        have hcardF : o.family.card ≤ ⌊FV⌋₊ :=
          loopRounds_small settings O P.D Dsf (persistDraws p) hks _ _ hloop o hoK
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
    · refine ⟨(learnerQ settings (prefixLaw D L) (prefixLaw D Ls) p.2).reads
        (fun w => O.mq w p.1), ?_, fun o' ho' => (Query.eval_congr _ _ ho').1⟩
      have hNrS : lazyReadCap settings.level3 settings.epochs settings.blocks settings.blockSize
          L₁ (Fintype.card σ) (Fintype.card α) ≤ Nr := by
        have h : (lazyReadCap settings.level3 settings.epochs settings.blocks settings.blockSize
            L₁ (Fintype.card σ) (Fintype.card α) : ℝ) ≤ Nr := by
          refine le_trans ?_ hNr
          unfold lazyReadCap
          rw [hE', hJ', hb', hwalk]
          push_cast
          show (E : ℝ) * (((settings.level3.probeBudget : ℝ) + J * b)
              * (Random.Kappa.probeSifts (L₁ + k) * (Q + 3))
              + (Q + 2) * ((Q + 2) * (A + 1) * (Q + 2)))
            ≤ E * ((B₃ + J * b) * (Random.Kappa.probeSifts (L₁ + k) * (Q + 3))
              + (Q + 2) * ((Q + 2) * (A + 1) * (Q + 2)))
          exact mul_le_mul_of_nonneg_left (add_le_add (mul_le_mul_of_nonneg_right
            (add_le_add hbud le_rfl)
            (mul_nonneg (Nat.cast_nonneg _)
              (show (0 : ℝ) ≤ Q + 3 from add_nonneg (Nat.cast_nonneg _) (by norm_num)))) le_rfl)
            (Nat.cast_nonneg _)
        exact_mod_cast h
      have h := card_reads_learnerQ_le O settings (prefixLaw D L) (prefixLaw D Ls) p hlenL hab
        hks hNrS hind hpk hX
      refine ((Nat.cast_le (α := ℝ)).2 h).trans ((reads_le_lazyReads (Fintype.card σ + 1)
        (6 * (Fintype.card σ + 1 + 1) * ⌊RB⌋₊) (⌊PB⌋₊ + 1) ⌊FV⌋₊
        (Nr + (Fintype.card σ + 1) * (settings.scanCap * (settings.level3.harvestReadCap
          * (Random.Kappa.probeSifts (L₁ + settings.level3.walkStart) * (Fintype.card σ + 1)))))
        (Fintype.card α) nXn (Fintype.card σ)).trans ?_)
      have ha : ((6 * (Fintype.card σ + 1 + 1) * ⌊RB⌋₊ : ℕ) : ℝ) ≤ 6 * (Q + 2) * RB := by
        have e : ((6 * (Fintype.card σ + 1 + 1) * ⌊RB⌋₊ : ℕ) : ℝ)
            = 6 * (Q + 2) * (⌊RB⌋₊ : ℝ) := by
          rw [hQ]; push_cast; ring
        rw [e]
        exact mul_le_mul_of_nonneg_left (Nat.floor_le hRB0) (by linarith)
      have hNs : (((Nr + (Fintype.card σ + 1) * (settings.scanCap
          * (settings.level3.harvestReadCap * (Random.Kappa.probeSifts
            (L₁ + settings.level3.walkStart) * (Fintype.card σ + 1))))) : ℕ) : ℝ) ≤ Ns := by
        rw [hwalk]
        push_cast
        show (Nr : ℝ) + (Q + 1) * ((settings.scanCap : ℝ) * ((settings.level3.harvestReadCap : ℝ)
            * (Random.Kappa.probeSifts (L₁ + k) * (Q + 1))))
          ≤ Nr + (Q + 1) * (SC * (HR * (Random.Kappa.probeSifts (L₁ + k) * (Q + 1))))
        have hu : (0 : ℝ) ≤ Random.Kappa.probeSifts (L₁ + k) * (Q + 1) := by positivity
        have h1 := mul_le_mul_of_nonneg_right hHR hu
        have h2 := mul_le_mul hSC h1 (by positivity)
          ((Nat.cast_nonneg _).trans hSC)
        have h3 := mul_le_mul_of_nonneg_left h2 (show (0 : ℝ) ≤ Q + 1 by positivity)
        exact add_le_add le_rfl h3
      exact lazyReads_mono (by positivity) (by positivity) (by positivity) (by positivity)
        (by positivity) (by positivity) (by positivity) (by positivity)
        (by push_cast; exact le_rfl) ha (by push_cast; linarith [Nat.floor_le hPB0])
        (Nat.floor_le hFV0) hNs le_rfl hnX le_rfl
  have hBl : μL.real Bloop ≤ 5 * δ / 8 := by
    refine hD.trans (le_of_eq ?_)
    show (Q + 1) * (5 * (δ / (8 * (Q + 1)))) = 5 * δ / 8
    field_simp
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
