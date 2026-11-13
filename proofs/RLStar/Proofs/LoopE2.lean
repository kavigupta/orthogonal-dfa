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
    have h1 : O.ηIn ≤ O.η := le_max_left _ _
    exact ⟨by simp only [Oc, Oracle.clamped]; have := le_max_left O.ηIn 0
              exact sub_nonneg.2 (max_le (by linarith) (by norm_num)),
      by simp only [Oc, Oracle.clamped]; linarith [le_max_right O.ηIn 0]⟩
  have hηOut : 0 ≤ Oc.ηOut ∧ Oc.ηOut ≤ 1 := by
    have h1 : O.ηOut ≤ O.η := le_max_right _ _
    exact ⟨by simp only [Oc, Oracle.clamped]; exact le_max_right _ _,
      by simp only [Oc, Oracle.clamped]; exact max_le (by linarith) (by norm_num)⟩
  have hb := hband F.card hcard (1 - Oc.ηIn) Oc.ηOut hηIn.1 hηIn.2 hηOut.1 hηOut.2
  let fam : SuffixFamily μ α σ :=
    ⟨Oc, M, F, B.lo, B.hi + 1, crossLimit, hL, hsf, hb⟩
  set R : Set (FreeMonoid α) := (lazyOffReads k CP F)ᶜ
  let fam' := fam.renoise R
  set D' := avoidLaw P.D k K₀
  set X₀ := Random.prefixTrie k CP
  set A := avoiding k (↑K₀ : Set (FreeMonoid α))
  set m' := P.D.real Aᶜ
  have hm' : m' ≤ m := (real_compl_avoiding_le P.D k K₀).trans hK₀
  have hm'0 : 0 ≤ m' := measureReal_nonneg
  have hA : P.D.real A = 1 - m' := by
    rw [show m' = P.D.real Aᶜ from rfl, probReal_compl_eq_one_sub (measurableSet_of_countable A)]
    ring
  have hApos : 0 < P.D.real A := by rw [hA]; linarith
  have hA0 : P.D.restrict A Set.univ ≠ 0 := by
    rw [Measure.restrict_apply_univ]
    intro h0
    rw [measureReal_def, h0, ENNReal.toReal_zero] at hApos
    exact lt_irrefl _ hApos
  have hD'eq : D' = (P.D.restrict A Set.univ)⁻¹ • P.D.restrict A := by
    show popLaw (P.D.restrict A) = _
    unfold popLaw
    rw [if_neg (not_or.2 ⟨hA0, measure_ne_top _ _⟩)]
  have hAv : ∀ᵐ x ∂D', x ∈ A := by
    rw [hD'eq]
    exact Measure.smul_absolutelyContinuous.ae_le (ae_restrict_mem (measurableSet_of_countable A))
  have hD'len : ∀ᵐ x ∂D', k ≤ x.toList.length ∧ x.toList.length ≤ P.L := by
    have hac : D' ≪ P.D := by
      rw [hD'eq]
      exact Measure.smul_absolutelyContinuous.trans Measure.restrict_le_self.absolutelyContinuous
    exact hac.ae_le (hKw.and P.length_le)
  have hp₁ : ∀ u, D'.real {x | prefixOf x k = u} ≤ 2 * prefixMax P.D k := by
    have hpm0 : 0 ≤ prefixMax P.D k := (measureReal_nonneg).trans (le_prefixMax P.D k 1)
    have hpm : prefixMax D' k ≤ 2 * prefixMax P.D k := by
      refine (prefixMax_avoidLaw_le P.D k K₀ (by linarith)).trans ?_
      rw [div_le_iff₀ (by linarith)]
      nlinarith
    exact fun u => (le_prefixMax D' k u).trans hpm
  have h4 := lazyFail_le fam' hcross.le settings X₀ (trieOK_prefixTrie k CP) D'
    (2 * prefixMax P.D k) δH tw Nr P.L hD'len hp₁ hprem hNr hδH htw
  set Sf := lazyFail fam' settings X₀ D' (lazySlope σ Nr settings.listSize)
    (lazySlack σ settings.listSize δH) tw
  have hSm : MeasurableSet Sf := measurableSet_lazyFail fam' settings X₀ D' _ _ tw
  obtain ⟨ω₀, hω₀⟩ := exists_noise_le (μ := μ) (lazyDrawLaw settings D') Sf hSm
  set ν := lazyLaw settings.epochs C.probeBudget settings.blocks settings.blockSize
  have hpr : MeasurePreserving (probesOf settings P.D K₀) ν (lazyDrawLaw settings D') :=
    measurePreserving_pi _ _ fun _ =>
      (measurePreserving_pi _ _ fun _ => ⟨measurable_drawFrom _, map_drawFrom _⟩).prod
        (measurePreserving_pi _ _ fun _ => measurePreserving_pi _ _ fun _ =>
          ⟨measurable_drawFrom _, map_drawFrom _⟩)
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
  have hShm : MeasurableSet Sh :=
    ((measurable_probesOf settings P.D K₀).comp measurable_snd)
      (Set.to_countable {d | ¬ LazyDrawsAll (fun x => k ≤ x.toList.length) d}).measurableSet
  have hSh0 : (μ.prod ν) Sh = 0 := by
    have hk' : ∀ᵐ x ∂D', k ≤ x.toList.length := hD'len.mono fun x hx => hx.1
    have hall : ∀ᵐ d ∂(lazyDrawLaw settings D'),
        LazyDrawsAll (fun x => k ≤ x.toList.length) d := by
      set ρ := (Measure.pi fun _ : Fin C.probeBudget => D').prod
        (Measure.pi fun _ : Fin settings.blocks => Measure.pi fun _ : Fin settings.blockSize => D')
      have he : ∀ e, MeasurePreserving (fun d : Fin settings.epochs →
          Random.EpochDraws α C.probeBudget settings.blocks settings.blockSize => d e)
          (lazyDrawLaw settings D') ρ := fun e => measurePreserving_eval (fun _ => ρ) e
      have h1 : ∀ i, MeasurePreserving (fun p : Random.EpochDraws α C.probeBudget settings.blocks
          settings.blockSize => p.1 i) ρ D' := fun i =>
        (measurePreserving_eval (fun _ => D') i).comp measurePreserving_fst
      have h2 : ∀ j t, MeasurePreserving (fun p : Random.EpochDraws α C.probeBudget settings.blocks
          settings.blockSize => p.2 j t) ρ D' := fun j t =>
        (measurePreserving_eval (fun _ => D') t).comp
          ((measurePreserving_eval (fun _ => Measure.pi fun _ => D') j).comp measurePreserving_snd)
      unfold LazyDrawsAll
      refine ae_all_iff.2 fun e => (ae_all_iff.2 fun i => ?_).and
        (ae_all_iff.2 fun j => ae_all_iff.2 fun t => ?_)
      · exact ((h1 i).comp (he e)).quasiMeasurePreserving.ae hk'
      · exact ((h2 j t).comp (he e)).quasiMeasurePreserving.ae hk'
    exact ae_iff.1
      (measurePreserving_snd.quasiMeasurePreserving.ae (hpr.quasiMeasurePreserving.ae hall))
  have hdraws : ∀ v, LazyDrawsAll (fun x => k ≤ x.toList.length) (probesOf settings P.D K₀ v) →
      LazyDrawsAll (fun x => k ≤ x.toList.length ∧ Avoids k ↑K₀ x)
        (probesOf settings P.D K₀ v) := fun v hv e =>
    ⟨fun i => ⟨(hv e).1 i, drawFrom_avoids P.D hk K₀ _⟩,
      fun j t => ⟨(hv e).2 j t, drawFrom_avoids P.D hk K₀ _⟩⟩
  have hoff : ∀ z, LazyKeyOff k CP z → ∀ p : Ω × Ω, fam'.read p z = fam.read p.1 z :=
    fun z hz => renoise_read_of_notMem fam R fun v hv hR => hR ⟨z, hz, v, hv, rfl⟩
  refine ⟨(fun q : Ω × settings.Uniforms => (q.1, probesOf settings P.D K₀ q.2)) ⁻¹'
    {q | ((q.1, ω₀), q.2) ∈ Sf} ∪ Sh, (hmp.measurable hSω).union hShm, ?_, ?_, ?_⟩
  · calc (μ.prod ν) ((fun q : Ω × settings.Uniforms => (q.1, probesOf settings P.D K₀ q.2)) ⁻¹'
          {q | ((q.1, ω₀), q.2) ∈ Sf} ∪ Sh)
        ≤ (μ.prod ν) ((fun q : Ω × settings.Uniforms => (q.1, probesOf settings P.D K₀ q.2)) ⁻¹'
          {q | ((q.1, ω₀), q.2) ∈ Sf}) + (μ.prod ν) Sh := measure_union_le _ _
      _ = (μ.prod (lazyDrawLaw settings D')) {q | ((q.1, ω₀), q.2) ∈ Sf} := by
        rw [hSh0, add_zero, hmp.measure_preimage hSω.nullMeasurableSet]
      _ ≤ _ := hω₀.trans h4
  · -- the bad draws read only the zone
    intro v ω ω' h hω
    rcases hω with hω | hω
    · by_cases hsh : (ω', v) ∈ Sh
      · exact Or.inr hsh
      refine Or.inl ?_
      change ((ω, ω₀), probesOf settings P.D K₀ v) ∈ Sf at hω
      show ((ω', ω₀), probesOf settings P.D K₀ v) ∈ Sf
      have hlen : ∀ᵐ x ∂D', x.toList.length ≤ P.L ∧ k ≤ x.toList.length ∧ Avoids k ↑K₀ x :=
        (hD'len.and hAv).mono fun x hx => ⟨hx.1.2, hx.1.1, hx.2⟩
      refine lazyFail_congr_up fam' settings CP D' hlen _ _ tw (probesOf settings P.D K₀ v)
        (hdraws v (not_not.1 hsh)) (fun z hz hzl => ?_) hω
      rw [hoff z hz, hoff z hz]
      refine familyRead_congr Oc (Z := {z}) (fun w hw => ?_) (Finset.mem_singleton_self z)
      obtain ⟨⟨z', v'⟩, hzv, rfl⟩ := Finset.mem_image.1 hw
      obtain ⟨hz', hv'⟩ := Finset.mem_product.1 hzv
      rw [Finset.mem_singleton] at hz'
      subst hz'
      exact h _ (Finset.mem_filter.2 ⟨Finset.mem_image.2 ⟨(z', v'),
        Finset.mem_product.2 ⟨mem_shortStrings.2 hzl, hv'⟩, rfl⟩, z', hz, hzl, v', hv', rfl⟩)
    · exact Or.inr hω
  · intro Sf' ω v hnot
    simp only [Set.mem_union, Set.mem_preimage, Set.mem_ofPred_eq, not_or] at hnot
    obtain ⟨hgood, hlong⟩ := hnot
    obtain ⟨hE, hpk⟩ := not_not.1 hgood
    have hd := hdraws v (not_not.1 hlong)
    have hres : (mkRound settings O P.D B F K₀ CP Sf' (ω, v)).result
        = lazyRun fam' settings X₀ ((ω, ω₀), probesOf settings P.D K₀ v) := by
      have hreads := reads_lazyRoundQ_keyOff C k settings.listSize CP (probesOf settings P.D K₀ v)
        (fun z => fam'.read (ω, ω₀) z) rfl hd
      exact (Query.eval_congr (fun z => fam'.read (ω, ω₀) z) _ fun z hz =>
        (hoff z (hreads z hz).1 (ω, ω₀)).symm).1
    have hdist : fam'.readDist = familyReadDist O M F B.lo (B.hi + 1) := by
      rw [SuffixFamily.renoise_readDist]
      rfl
    refine ⟨?_, ?_⟩
    · set r := lazyRun fam' settings X₀ ((ω, ω₀), probesOf settings P.D K₀ v)
      obtain ⟨hsub, hmid⟩ := lazyRoundQ_expanded_mids (fun z => fam'.read (ω, ω₀) z) C k
        settings.listSize X₀ (probesOf settings P.D K₀ v)
      set S := Random.lazyDisagree (fun z => familyRead O.mq F B.lo (B.hi + 1) z ω) k r.expanded
        r.hypothesis
      set S' := Random.lazyDisagree (fun z => fam'.read (ω, ω₀) z) k r.expanded r.hypothesis
      have hU : (fun q => fam'.readDist q .undecided)
          = fun q => familyReadDist O M F B.lo (B.hi + 1) q .undecided := by rw [hdist]
      show LooseEndsWell M (fun q => familyReadDist O M F B.lo (B.hi + 1) q .undecided)
        (lazySlope σ Nr settings.listSize) (lazySlack σ settings.listSize δH) P.D
        (C.disagreementRate + tw + m)
        (Random.lazyDisagree (fun z => familyRead O.mq F B.lo (B.hi + 1) z ω) k
          (mkRound settings O P.D B F K₀ CP Sf' (ω, v)).result.expanded
          (mkRound settings O P.D B F K₀ CP Sf' (ω, v)).result.hypothesis)
        (mkRound settings O P.D B F K₀ CP Sf' (ω, v)).result.ending
      rw [← hU, hres]
      refine LooseEndsWell.of_real (fun hle => ?_) hE
      have hSS : (S ∩ A : Set (FreeMonoid α)) ≤ᵐ[P.D] (S' ∩ A : Set (FreeMonoid α)) := by
        filter_upwards [hKw] with x hkx hx
        exact ⟨(mem_lazyDisagree_congr hsub hmid hkx hx.2 fun z hz _ =>
          (hoff z hz (ω, ω₀)).symm).1 hx.1, hx.2⟩
      have h1 : P.D.real S ≤ P.D.real (S ∩ A) + m' :=
        (measureReal_mono (fun x hx => by
          by_cases hxA : x ∈ A
          exacts [Or.inl ⟨hx, hxA⟩, Or.inr hxA]) (measure_ne_top _ _)).trans
          (measureReal_union_le _ _)
      have h2 : P.D.real (S ∩ A) ≤ P.D.real (S' ∩ A) := by
        rw [measureReal_def, measureReal_def]
        exact ENNReal.toReal_mono (measure_ne_top _ _) (measure_mono_ae hSS)
      have h3 : P.D.real (S' ∩ A) ≤ D'.real S' := by
        have hD' : D'.real S' = P.D.real (S' ∩ A) / P.D.real A := by
          show (popLaw (P.D.restrict A)).real S' = _
          rw [popLaw_real hA0 (measure_ne_top _ _),
            measureReal_restrict_apply (measurableSet_of_countable _),
            measureReal_restrict_apply_univ]
        rw [hD', le_div_iff₀ hApos]
        have := measureReal_nonneg (μ := P.D) (s := S' ∩ A)
        nlinarith
      linarith
    · rw [hres]
      exact hpk

end Bad

end OrthoDFA
