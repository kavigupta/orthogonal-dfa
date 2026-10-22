import OrthoDFA.Proofs.LoopE2Pre
import OrthoDFA.Proofs.BandWide

/-!
# A round's level 3 on one noise

Level 3 reads, at probes avoiding the clustering's prefixes, only strings the clustering did not.
Where the clustering's reads are re-noised it reads a fresh family on probes from `D` conditioned
to avoid them, so level 5's guarantee holds there (`round_bad_exists`), and the actual run is the
same run. Transferred back to `D` it ends well, and off a collision it does so with the chance it
has on fresh noise (`loop_round_le_persist`).
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

/-- The strings level 3 reads at a probe whose first `k` letters avoid `K₀`. -/
def offReads (k : ℕ) (K₀ F : Finset (FreeMonoid α)) : Set (FreeMonoid α) :=
  {w | ∃ z, KeyOff k ↑K₀ z ∧ ∃ v ∈ F, w = z * v}

open Classical in
/-- Of them, those at strings at most `N'` long. -/
def roundZone (N' k : ℕ) (K₀ F : Finset (FreeMonoid α)) : Finset (FreeMonoid α) :=
  (readStrings (shortStrings N') F).filter (· ∈ offReads k K₀ F)

lemma avoidLaw_len (D : Measure (FreeMonoid α)) {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (k : ℕ) (K₀ : Finset (FreeMonoid α)) : ∀ᵐ x ∂(avoidLaw D k K₀), x.toList.length ≤ L := by
  refine ae_iff.2 (popLaw_null (measure_mono_null (fun x hx => hx)
    (Measure.restrict_le_self.absolutelyContinuous (ae_iff.1 hlen))) ?_)
  simp

section Bad

/-- Level 5's premises at `f = 0`, one spurious leaf tolerated within level 3's `|Q| + 2`, for
probes at most `L` long, wrong reads at `εw + κ U` and no length-`walkStart` prefix above `p₀`. -/
def RoundPremises (σ α : Type*) [Fintype σ] [Fintype α] (L : ℕ) (C : Random.Level3Settings)
    (εw κ p₀ : ℝ) : Prop :=
  let nQ := Fintype.card σ
  let nα := Fintype.card α
  let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
  let w : ℕ := min (L + 1) (3 + Nat.clog 2 L) * (nQ + 1)
  let c : ℝ := εw * max w (nQ + 2) + κ
  let b : ℝ := C.probeBudget * w + (nQ + 2) * nα * (nQ + 2) * (nQ + 1)
  let m : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / C.δ) + 1 + (nQ + 2) * nα * (nQ + 1)
  0 < C.δ ∧ C.δ < 1 ∧ 0 < C.harvestRate
    ∧ C.harvestRate ≤ C.disagreementRate / 48 ∧ C.disagreementRate < 1
    ∧ 36 * S / C.disagreementRate * Real.log (16 * S * (S + 1) / C.δ) + S ≤ C.probeBudget
    ∧ 0 ≤ εw ∧ 0 ≤ κ ∧ ∃ t t' : ℝ, 0 ≤ t ∧ 0 ≤ t' ∧ t * c ≤ 1 - Real.exp (-(t' * c))
      ∧ C.probeBudget * (C.probeBudget - 1 : ℝ) / 2 * p₀
        + Real.exp (t' * (εw * b + κ * m)) / (1 + t) ^ 2 ≤ C.δ / 8

variable [Nonempty α] {σ : Type*} [Fintype σ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
  (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) (P : PrefixDistribution α)
  (settings : LoopSettings K)

include hL in
/-- At a state and family, the round's bad draws: where level 3 fails at re-noised clustering
reads. Its chance is at most `δ + rarerTail T (2 prefixMax) ((|Q| + 1)(ζ² + ζ)) u`, it is fixed
by the answers at the strings level 3 reads off the clustering's prefixes, and off it the actual
round meets its claim. -/
theorem round_bad_exists (m u crossLimit : ℝ) (hk : 1 ≤ settings.level3.walkStart)
    (hη1 : O.η ≤ 1) (hcross : 0 < crossLimit)
    (hprem : RoundPremises σ α P.L settings.level3 (crossLimit ^ 2) crossLimit
      (2 * prefixMax P.D settings.level3.walkStart))
    (hm1 : m ≤ 1 / 2)
    (B : State) (F K₀ : Finset (FreeMonoid α)) (hsf : SuffixFree F) (hcard : F.card + 1 ≤ B.k)
    (hband : ∀ N, N + 1 ≤ B.k → ∀ p r : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 →
      BandPasses N B.lo (B.hi + 1) p r (crossLimit ^ 2) crossLimit crossLimit)
    (hK₀ : K₀.card * prefixMax P.D settings.level3.walkStart ≤ m) :
    ∃ Bd : Set (Ω × (Fin settings.level3.probeBudget → unitInterval)), MeasurableSet Bd
      ∧ (μ.prod (Measure.pi fun _ => volume)) Bd
        ≤ ENNReal.ofReal (settings.level3.δ
          + rarerTail settings.level3.probeBudget (2 * prefixMax P.D settings.level3.walkStart)
            ((Fintype.card σ + 1) * (crossLimit ^ 2 + crossLimit)) u)
      ∧ (∀ v ω ω', (∀ w ∈ roundZone (max (P.L + settings.level3.probeBudget + 2)
          (probesLen (probesOf settings P.D K₀ v) + 1 + settings.level3.probeBudget + 2))
          settings.level3.walkStart K₀ F, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
        (ω, v) ∈ Bd → (ω', v) ∈ Bd)
      ∧ ∀ x₀ Pr Sf ω v, (ω, v) ∉ Bd →
        RoundSpec M O P (roundSlope σ P.L settings.level3)
          (roundSlack σ α settings.level3 (2 * prefixMax P.D settings.level3.walkStart)) m
          (settings.level3.disagreementRate + u + m)
          (settings.level3.harvestRate / 2) settings.level3.walkStart
          (mkRound settings O P.D x₀ B F K₀ Pr Sf (ω, v)) := by
  classical
  obtain ⟨hδ0, hδ1, hy0, hyρ, hρ1, hP, hcr0, hκ0, t, t', ht, ht', htc, hfin⟩ := hprem
  set C := settings.level3 with hCdef
  set k := C.walkStart
  set T := C.probeBudget
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
    ⟨Oc, M, F, B.lo, B.hi + 1, crossLimit ^ 2, crossLimit, crossLimit, hL, hsf, hb⟩
  set R : Set (FreeMonoid α) := (offReads k K₀ F)ᶜ
  let fam' := fam.renoise R
  -- the conditioned probes
  set D' := avoidLaw P.D k K₀
  set m' := P.D.real (avoiding k ↑K₀)ᶜ
  have hm' : m' ≤ m := (real_compl_avoiding_le P.D k K₀).trans hK₀
  have hm'0 : 0 ≤ m' := measureReal_nonneg
  have hD'len := avoidLaw_len P.D P.length_le k K₀
  let P' : PrefixDistribution α := ⟨D', inferInstance, P.L, hD'len⟩
  have hpm0 : 0 ≤ prefixMax P.D k := (measureReal_nonneg).trans (le_prefixMax P.D k 1)
  have hpm : prefixMax D' k ≤ 2 * prefixMax P.D k := by
    refine (prefixMax_avoidLaw_le P.D k K₀ (by linarith)).trans ?_
    rw [div_le_iff₀ (by linarith)]
    nlinarith
  have hpre' : ∀ v, P'.D.real {x | prefixOf x C.walkStart = v} ≤ 2 * prefixMax P.D k :=
    fun v => (le_prefixMax D' k v).trans hpm
  have h4 := level3Fail_le fam' hcr0 hκ0 P' C (2 * prefixMax P.D k) u t t' hpre' hδ0 hδ1 hy0
    hyρ hρ1 hP ht ht' htc hfin
  have hA0 : 0 ≤ roundSlope σ P.L C := by unfold roundSlope; positivity
  have hsl0 : 0 ≤ roundSlack σ α C (2 * prefixMax P.D k) := by
    unfold roundSlack
    refine div_nonneg (mul_nonneg (by linarith) (Real.log_nonneg ?_)) hy0.le
    rw [le_div_iff₀ hδ0]
    have : (1 : ℝ) ≤ ((1 + (Fintype.card σ + 1) ^ 3 * Fintype.card α) ^ Fintype.card σ
        * (Fintype.card σ + 3) ^ ((Fintype.card σ + 2) * Fintype.card α) : ℕ) := by
      exact_mod_cast Nat.one_le_iff_ne_zero.2 (by positivity)
    have : (0 : ℝ) ≤ Fintype.card σ := Nat.cast_nonneg _
    nlinarith
  set Sf := level3Fail fam' C D' (roundSlope σ P.L C) (roundSlack σ α C (2 * prefixMax P.D k)) u
  have hSm : MeasurableSet Sf := measurableSet_level3Fail fam' C D' hA0 hsl0 u
  obtain ⟨ω₀, hω₀⟩ := exists_noise_le (μ := μ) (Measure.pi fun _ : Fin T => D') Sf hSm
  have hpr : MeasurePreserving (probesOf settings P.D K₀)
      (Measure.pi fun _ : Fin T => (volume : Measure unitInterval)) (Measure.pi fun _ => D') :=
    measurePreserving_pi _ _ fun _ => ⟨measurable_drawFrom _, map_drawFrom _⟩
  have hmp : MeasurePreserving (fun q : Ω × (Fin T → unitInterval) =>
      (q.1, probesOf settings P.D K₀ q.2)) (μ.prod (Measure.pi fun _ => volume))
      (μ.prod (Measure.pi fun _ => D')) := (MeasurePreserving.id μ).prod hpr
  have hSω : MeasurableSet {q : Ω × (Fin T → FreeMonoid α) | ((q.1, ω₀), q.2) ∈ Sf} :=
    ((measurable_fst.prodMk measurable_const).prodMk measurable_snd) hSm
  refine ⟨(fun q : Ω × (Fin T → unitInterval) => (q.1, probesOf settings P.D K₀ q.2)) ⁻¹'
    {q | ((q.1, ω₀), q.2) ∈ Sf}, hmp.measurable hSω, ?_, ?_, ?_⟩
  · rw [hmp.measure_preimage hSω.nullMeasurableSet]
    exact hω₀.trans h4
  · -- the bad draws read only the zone
    intro v ω ω' h hω
    have hread : ∀ z : FreeMonoid α, z.toList.length ≤ max (P.L + T + 2)
        (probesLen (probesOf settings P.D K₀ v) + 1 + T + 2) →
        fam'.read (ω, ω₀) z = fam'.read (ω', ω₀) z := by
      intro z hz
      unfold SuffixFamily.read familyRead voteCount
      congr 2
      refine Finset.filter_congr fun v' hv => ?_
      by_cases hzv : z * v' ∈ R
      · show (Oc.renoise R).mq (z * v') (ω, ω₀) = 1 ↔ (Oc.renoise R).mq (z * v') (ω', ω₀) = 1
        rw [renoise_mq_of_mem Oc hzv, renoise_mq_of_mem Oc hzv]
      · show (Oc.renoise R).mq (z * v') (ω, ω₀) = 1 ↔ (Oc.renoise R).mq (z * v') (ω', ω₀) = 1
        rw [renoise_mq_of_notMem Oc hzv, renoise_mq_of_notMem Oc hzv]
        refine h _ (Finset.mem_filter.2 ⟨Finset.mem_image.2 ⟨(z, v'), Finset.mem_product.2
          ⟨mem_shortStrings.2 hz, hv⟩, rfl⟩, not_not.1 hzv⟩)
    exact level3Fail_congr_up fam' C D' hD'len _ _ u (probesOf settings P.D K₀ v) hread hω
  · -- off the bad draws, the actual round meets its claim
    intro x₀ Pr Sf' ω v hnot
    set xs := probesOf settings P.D K₀ v
    set read' : FreeMonoid α → ARU := fun z => fam'.read (ω, ω₀) z
    have hgood : ¬ ((ω, ω₀), xs) ∈ Sf := hnot
    obtain ⟨hE, hcls⟩ := not_not.1 hgood
    set read : FreeMonoid α → ARU := fun z => familyRead O.mq F B.lo (B.hi + 1) z ω
    have hoff : ∀ z, KeyOff k ↑K₀ z → read z = read' z := fun z hz =>
      (renoise_read_of_notMem fam R (fun v hv hR => hR ⟨z, hz, v, hv, rfl⟩) (ω, ω₀)).symm
    have hxs : ∀ i, Avoids k ↑K₀ (xs i) := fun i => drawFrom_avoids P.D hk K₀ (v i)
    have hres : Random.indecisiveLStarLevel3 C read xs = Random.indecisiveLStarLevel3 C read' xs :=
      Random.level3_congr_zone C xs fun i j hj hjx m _ => hoff _ (keyOff_of_prefix (hxs i) hj hjx m)
    have hdist : fam'.readDist = familyReadDist O M F B.lo (B.hi + 1) := by
      rw [SuffixFamily.renoise_readDist]
      rfl
    set H := (Random.indecisiveLStarLevel3 C read xs).hypothesis with hHd
    have hq := looseEndsWell_of_endsWell P.D M
      (fun q => familyReadDist O M F B.lo (B.hi + 1) q .undecided) hA0 hsl0
      (k := k) (K₀ := ↑K₀) (read := read) (read' := read') hoff H
      (disagreeSet M (familyReadDist O M F B.lo (B.hi + 1)) read k H)
      (disagreeSet M (familyReadDist O M F B.lo (B.hi + 1)) read' k H)
      (fun x hx h => by
        have hp : probeOrAgree read H.tree H.edges k x = probeOrAgree read' H.tree H.edges k x :=
          Random.probeOrAgree_congr_zone H.edges k fun j hj hjx m _ =>
            hoff _ (keyOff_of_prefix hx hj hjx m)
        have e1 : x ∈ Random.searchSet read k H ↔ x ∈ Random.searchSet read' k H := by
          simp only [Random.searchSet, Set.mem_ofPred_eq]; rw [hp]
        have e2 : x ∈ Random.endsSet read k H ↔ x ∈ Random.endsSet read' k H := by
          simp only [Random.endsSet, Set.mem_ofPred_eq]; rw [hp]
        rcases h with (h | h) | ⟨hk', m, hm, hw⟩
        · exact .inl (.inl (e1.1 h))
        · exact .inl (.inr (e2.1 h))
        · refine .inr ⟨hk', m, hm, ?_⟩
          have := hoff _ (keyOff_of_prefix hx hk' le_rfl m)
          rw [prefixOf_length] at this
          beta_reduce at hw ⊢
          rw [← this]
          exact hw)
      (C.disagreementRate + u) C.harvestRate hy0.le
      (Random.indecisiveLStarLevel3 C read xs).ending (by linarith) (by
        have hH : H = (Random.indecisiveLStarLevel3 C read' xs).hypothesis := by rw [hHd, hres]
        rw [hres, hH, ← hdist]
        exact hE)
    refine ⟨?_, ?_⟩
    · have hq' := hq.mono (m' := m) (ε' := C.disagreementRate + u + m)
        (y' := C.harvestRate / 2) hm' (by linarith) (by nlinarith)
      exact hq'
    · show (Random.indecisiveLStarLevel3 C read xs).hypothesis.tree ∈ _
      rw [hres]
      exact hcls

end Bad

end OrthoDFA
