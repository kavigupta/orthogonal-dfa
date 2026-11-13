import RLStar.Proofs.LoopE2Pre
import RLStar.Proofs.BandKappa

/-!
# A round's lazy round on one noise

The lazy round reads, at probes avoiding the clustering's prefixes, only strings the clustering
did not. Where the clustering's reads are re-noised it reads a fresh family on probes from `D`
conditioned to avoid them, so `lazyFail_le` holds there (`lazy_round_bad_exists`), and the actual
run is the same run. Transferred back to `D` it ends well, and off a collision it does so with
the chance it has on fresh noise (`loop_round_le_persist`).
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

instance : Countable State :=
  Function.Injective.countable (f := fun B : State => (B.nsuff, B.npref, B.k, B.lo, B.hi, B.sc,
    B.scd)) fun a b h => by
    cases a; cases b
    simp only [Prod.mk.injEq] at h
    obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h
    subst h1 h2 h3 h4 h5 h6 h7
    rfl

/-- `O` with a rate no string has set to zero. -/
def Oracle.clamped (O : Oracle μ (FreeMonoid α)) : Oracle μ (FreeMonoid α) :=
  { O with
    ηIn := max O.ηIn 0
    ηOut := max O.ηOut 0
    noise_mean_in := fun w hw => by
      have h := O.rate_nonneg w
      unfold Oracle.rate at h
      rw [if_pos hw] at h
      rw [O.noise_mean_in w hw, max_eq_left h]
    noise_mean_out := fun w hw => by
      have h := O.rate_nonneg w
      unfold Oracle.rate at h
      rw [if_neg hw] at h
      rw [O.noise_mean_out w hw, max_eq_left h] }

omit [IsProbabilityMeasure μ] in
lemma popLaw_null {ν : Measure (FreeMonoid α)} {A : Set (FreeMonoid α)} (hA : ν A = 0)
    (h1 : (1 : FreeMonoid α) ∉ A) : popLaw ν A = 0 := by
  unfold popLaw
  split_ifs
  · rw [Measure.dirac_apply' _ (measurableSet_of_countable A), Set.indicator_of_notMem h1]
  · rw [Measure.smul_apply, hA, smul_zero]

omit [DecidableEq α] in
lemma renoise_mq_of_mem (O : Oracle μ (FreeMonoid α)) {R : Set (FreeMonoid α)}
    {w : FreeMonoid α} (hw : w ∈ R) (p : Ω × Ω) : (O.renoise R).mq w p = O.mq w p.2 := by
  classical
  simp only [Oracle.mq, Oracle.label, Oracle.renoise, renoiseNoise, if_pos hw]

/-- The strings a lazy round from `CP`'s trie reads through `F` at probes avoiding `CP`'s first `k`
letters. -/
def lazyOffReads (k : ℕ) (CP F : Finset (FreeMonoid α)) : Set (FreeMonoid α) :=
  {w | ∃ z, LazyKeyOff k CP z ∧ ∃ v ∈ F, w = z * v}

lemma avoidLaw_len (D : Measure (FreeMonoid α)) {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (k : ℕ) (K₀ : Finset (FreeMonoid α)) : ∀ᵐ x ∂(avoidLaw D k K₀), x.toList.length ≤ L := by
  refine ae_iff.2 (popLaw_null (measure_mono_null (fun x hx => hx)
    (Measure.restrict_le_self.absolutelyContinuous (ae_iff.1 hlen))) ?_)
  simp

section Bad

variable [Nonempty α] {σ : Type*} [Fintype σ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
  (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) (P : PrefixDistribution α)
  (settings : LoopSettings K)

include hL in
/-- At a state, family and prefixes `CP`, the round's bad draws: where the lazy round fails at
re-noised clustering reads. Its chance is at most `lazyFail_le`'s bound, it is fixed by the
answers at the strings the lazy round reads off `CP`, and off it the actual round meets its
claim. -/
theorem lazy_round_bad_exists (m crossLimit δH tw : ℝ) (Nr : ℕ)
    (hk : 1 ≤ settings.level3.walkStart)
    (hKw : ∀ᵐ x ∂P.D, settings.level3.walkStart ≤ x.toList.length)
    (hη1 : O.η ≤ 1) (hcross : 0 < crossLimit)
    (hprem : LazyPremises σ α settings.level3 settings.listSize settings.epochs settings.blocks
      (2 * prefixMax P.D settings.level3.walkStart))
    (hNr : lazyReadCap settings.level3 settings.epochs settings.blocks settings.blockSize P.L
      (Fintype.card σ) (Fintype.card α) ≤ Nr)
    (hδH : 0 < δH) (htw : 0 < tw) (hm1 : m ≤ 1 / 2)
    (B : State) (F CP : Finset (FreeMonoid α)) (hsf : SuffixFree F) (hcard : F.card + 1 ≤ B.k)
    (hband : ∀ N, N + 1 ≤ B.k → ∀ p r : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 →
      RarerBounded N B.lo (B.hi + 1) p r crossLimit)
    (hK₀ : (CP.image fun p => prefixOf p settings.level3.walkStart).card
      * prefixMax P.D settings.level3.walkStart ≤ m) :
    ∃ Bd : Set (Ω × settings.Uniforms), MeasurableSet Bd
      ∧ (μ.prod (lazyLaw settings.epochs settings.level3.probeBudget settings.blocks
          settings.blockSize)) Bd
        ≤ ENNReal.ofReal (lazyRoundBound σ settings.level3 settings.epochs settings.blocks
          settings.blockSize crossLimit Nr P.L δH tw)
      ∧ (∀ v ω ω', (∀ w ∈ lazyZone (max (P.L + settings.level3.probeBudget + 2)
          (lazyProbesLen (probesOf settings P.D
            (CP.image fun p => prefixOf p settings.level3.walkStart) v)
            + 1 + settings.level3.probeBudget + 2)) settings.level3.walkStart CP F,
          (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
        (ω, v) ∈ Bd → (ω', v) ∈ Bd)
      ∧ ∀ Sf ω v, (ω, v) ∉ Bd →
        RoundSpec M O P (lazySlope σ Nr settings.listSize) (lazySlack σ settings.listSize δH)
          (settings.level3.disagreementRate + tw + m) settings.level3.walkStart
          (mkRound settings O P.D B F (CP.image fun p => prefixOf p settings.level3.walkStart)
            CP Sf (ω, v)) := by
  classical
  set C := settings.level3 with hCdef
  set k := C.walkStart
  set K₀ := CP.image fun p => prefixOf p k
  set Oc := O.clamped
  have hηIn : 0 ≤ 1 - Oc.ηIn ∧ 1 - Oc.ηIn ≤ 1 := by
    sorry -- closes by: main's hηIn (clamped noise rates)
  have hηOut : 0 ≤ Oc.ηOut ∧ Oc.ηOut ≤ 1 := by
    sorry -- closes by: main's hηOut
  have hb := hband F.card hcard (1 - Oc.ηIn) Oc.ηOut hηIn.1 hηIn.2 hηOut.1 hηOut.2
  let fam : SuffixFamily μ α σ :=
    ⟨Oc, M, F, B.lo, B.hi + 1, crossLimit, hL, hsf, hb⟩
  set R : Set (FreeMonoid α) := (lazyOffReads k CP F)ᶜ
  let fam' := fam.renoise R
  set D' := avoidLaw P.D k K₀
  set X₀ := Random.prefixTrie k CP
  have hD'len : ∀ᵐ x ∂D', k ≤ x.toList.length ∧ x.toList.length ≤ P.L := by
    sorry -- closes by: avoidLaw_len, popLaw_null on hKw
  have hp₁ : ∀ u, D'.real {x | prefixOf x k = u} ≤ 2 * prefixMax P.D k := by
    sorry -- closes by: main's hpm, prefixMax_avoidLaw_le, real_compl_avoiding_le, hK₀
  have h4 := lazyFail_le fam' hcross.le settings X₀ (trieOK_prefixTrie k CP) D'
    (2 * prefixMax P.D k) δH tw Nr P.L hD'len hp₁ hprem hNr hδH htw
  set Sf := lazyFail fam' settings X₀ D' (lazySlope σ Nr settings.listSize)
    (lazySlack σ settings.listSize δH) tw
  have hSm : MeasurableSet Sf := measurableSet_lazyFail fam' settings X₀ D' _ _ tw
  obtain ⟨ω₀, hω₀⟩ := exists_noise_le (μ := μ) (lazyDrawLaw settings D') Sf hSm
  set ν := lazyLaw settings.epochs C.probeBudget settings.blocks settings.blockSize
  have hpr : MeasurePreserving (probesOf settings P.D K₀) ν (lazyDrawLaw settings D') := by
    sorry -- closes by: measurePreserving_pi of map_drawFrom, as main's hpr
  have hmp : MeasurePreserving (fun q : Ω × settings.Uniforms =>
      (q.1, probesOf settings P.D K₀ q.2)) (μ.prod ν) (μ.prod (lazyDrawLaw settings D')) :=
    (MeasurePreserving.id μ).prod hpr
  have hSω : MeasurableSet {q : Ω × (Fin settings.epochs →
      Random.EpochDraws α C.probeBudget settings.blocks settings.blockSize) |
      ((q.1, ω₀), q.2) ∈ Sf} :=
    ((measurable_fst.prodMk measurable_const).prodMk measurable_snd) hSm
  -- the draws, null under `ν`, with a probe shorter than `k`
  set Sh : Set (Ω × settings.Uniforms) :=
    {q | ¬ LazyDrawsAll (fun x => k ≤ x.toList.length) (probesOf settings P.D K₀ q.2)}
  have hShm : MeasurableSet Sh := by
    sorry -- closes by: measurable_probesOf, countable codomain
  have hSh0 : (μ.prod ν) Sh = 0 := by
    sorry -- closes by: drawFrom_pos_or_one, hD'len, volume of {1} is 0
  refine ⟨(fun q : Ω × settings.Uniforms => (q.1, probesOf settings P.D K₀ q.2)) ⁻¹'
    {q | ((q.1, ω₀), q.2) ∈ Sf} ∪ Sh, (hmp.measurable hSω).union hShm, ?_, ?_, ?_⟩
  · calc (μ.prod ν) ((fun q : Ω × settings.Uniforms => (q.1, probesOf settings P.D K₀ q.2)) ⁻¹'
          {q | ((q.1, ω₀), q.2) ∈ Sf} ∪ Sh)
        ≤ (μ.prod ν) ((fun q : Ω × settings.Uniforms => (q.1, probesOf settings P.D K₀ q.2)) ⁻¹'
          {q | ((q.1, ω₀), q.2) ∈ Sf}) + (μ.prod ν) Sh := measure_union_le _ _
      _ = (μ.prod (lazyDrawLaw settings D')) {q | ((q.1, ω₀), q.2) ∈ Sf} := by
        rw [hSh0, add_zero, hmp.measure_preimage hSω.nullMeasurableSet]
      _ ≤ _ := hω₀.trans h4
  · intro v ω ω' h hω
    sorry -- closes by: main's hread via renoise_mq_of_mem/notMem, lazyFail_congr_up
  · intro Sf' ω v hnot
    simp only [Set.mem_union, Set.mem_preimage, Set.mem_setOf_eq, not_or] at hnot
    obtain ⟨hgood, hlong⟩ := hnot
    obtain ⟨hE, hpk⟩ := not_not.1 hgood
    have hres : (mkRound settings O P.D B F K₀ CP Sf' (ω, v)).result
        = lazyRun fam' settings X₀ ((ω, ω₀), probesOf settings P.D K₀ v) := by
      sorry -- closes by: renoise_read_of_notMem on LazyKeyOff, reads_lazyRoundQ_keyOff, eval_congr
    have hdist : fam'.readDist = familyReadDist O M F B.lo (B.hi + 1) := by
      rw [SuffixFamily.renoise_readDist]
      rfl
    refine ⟨?_, ?_⟩
    · sorry -- closes by: hres, hdist; lazyDisagree off avoiding mass m, LooseEndsWell.mono
    · rw [hres]
      exact hpk

end Bad

end OrthoDFA
