import OrthoDFA.Proofs.LoopE2Pre
import OrthoDFA.Proofs.BandKappa
import OrthoDFA.Proofs.HarvestConc

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

lemma drawFrom_avoidLaw_len (D : Measure (FreeMonoid α)) {L : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (k : ℕ) (K₀ : Finset (FreeMonoid α)) (u : unitInterval) :
    (drawFrom (avoidLaw D k K₀) u).toList.length ≤ L := by
  rcases drawFrom_pos_or_one (avoidLaw D k K₀) u with h | h
  · by_contra hlong
    push_neg at hlong
    have hae := avoidLaw_len D hlen k K₀
    rw [ae_iff] at hae
    have hmem : drawFrom (avoidLaw D k K₀) u ∈ {x : FreeMonoid α | ¬ x.toList.length ≤ L} := by
      simp only [Set.mem_setOf_eq, not_le]; exact hlong
    have hnull : avoidLaw D k K₀ {drawFrom (avoidLaw D k K₀) u} = 0 :=
      measure_mono_null (Set.singleton_subset_iff.2 hmem) hae
    rw [measureReal_def, hnull] at h
    simp at h
  · rw [h]; simp

section Bad

/-- Level 5's premises at `f = 0`, one spurious leaf tolerated within level 3's `|Q| + 2`, for
wrong reads at `κ U` and no length-`walkStart` prefix above `p₀`. -/
def RoundPremises (σ α : Type*) [Fintype σ] [Fintype α] (C : Random.Level3Settings)
    (κ p₀ : ℝ) : Prop :=
  let nQ := Fintype.card σ
  let nα := Fintype.card α
  let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
  let m : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / C.δ) + 1 + (nQ + 2) * nα * (nQ + 1)
  0 < C.δ ∧ C.δ < 1 ∧ 0 < C.harvestRate
    ∧ C.harvestRate ≤ C.disagreementRate / 48 ∧ C.disagreementRate < 1
    ∧ 36 * S / C.disagreementRate * Real.log (16 * S * (S + 1) / C.δ) + S ≤ C.probeBudget
    ∧ 0 ≤ κ ∧ ∃ t t' : ℝ, 0 ≤ t ∧ 0 ≤ t' ∧ t * κ ≤ 1 - Real.exp (-(t' * κ))
      ∧ C.probeBudget * (C.probeBudget - 1 : ℝ) / 2 * p₀
        + Real.exp (t' * (κ * m)) / (1 + t) ^ 2 ≤ C.δ / 8

variable [Nonempty α] {σ : Type*} [Fintype σ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
  (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) (P : PrefixDistribution α)
  (settings : LoopSettings K)

/-- The Hoeffding slack for a round's empirical harvest: small enough that the normalized sample
stays within the good bound's `m/(hr/2)` slack and the atom bound `ca`. -/
def roundGamma (m ca pm hr : ℝ) : ℝ := min (m * hr / (hr + m)) ((ca * hr - 2 * pm) / (1 + ca))

section GammaFacts
variable {m ca pm hr : ℝ}

lemma roundGamma_pos (hm : 0 < m) (hca : 0 ≤ ca) (hhr : 0 < hr) (hatom : 2 * pm < ca * hr) :
    0 < roundGamma m ca pm hr := by
  unfold roundGamma
  refine lt_min (by positivity) ?_
  exact div_pos (by linarith) (by linarith)

lemma roundGamma_lt (hm : 0 < m) (hca : 0 ≤ ca) (hhr : 0 < hr) :
    roundGamma m ca pm hr < hr := by
  unfold roundGamma
  refine lt_of_le_of_lt (min_le_left _ _) ?_
  rw [div_lt_iff₀ (by linarith)]
  nlinarith

lemma roundGamma_good (hm : 0 < m) (hca : 0 ≤ ca) (hhr : 0 < hr) (hatom : 2 * pm < ca * hr) :
    2 * roundGamma m ca pm hr / (hr - roundGamma m ca pm hr) ≤ m / (hr / 2) := by
  have hg0 := roundGamma_pos hm hca hhr hatom
  have hglt := roundGamma_lt hm hca hhr (pm := pm)
  have hle : roundGamma m ca pm hr ≤ m * hr / (hr + m) := min_le_left _ _
  rw [div_le_div_iff₀ (by linarith) (by positivity)]
  rw [le_div_iff₀ (by linarith) ] at hle
  nlinarith [hle, hg0, hglt]

lemma roundGamma_atom (hm : 0 < m) (hca : 0 ≤ ca) (hhr : 0 < hr) (hatom : 2 * pm < ca * hr) :
    (2 * pm + roundGamma m ca pm hr) / (hr - roundGamma m ca pm hr) ≤ ca := by
  have hg0 := roundGamma_pos hm hca hhr hatom
  have hglt := roundGamma_lt hm hca hhr (pm := pm)
  have hle : roundGamma m ca pm hr ≤ (ca * hr - 2 * pm) / (1 + ca) := min_le_right _ _
  rw [div_le_iff₀ (by linarith)]
  rw [le_div_iff₀ (by linarith)] at hle
  nlinarith [hle, hg0, hglt]

/-- The normalized sample ratio `(rS+γ)/(rU-γ)`, when `rS ≤ G·rU` and `hr0 ≤ rU`, stays within
`G` plus the good slack. -/
lemma empirical_ratio_le {rS rU G hr0 γ slk : ℝ} (hSU : rS ≤ rU) (hgood : rS ≤ G * rU)
    (hr : hr0 ≤ rU) (hγ : 0 < γ) (hγr : γ < hr0) (hslack : 2 * γ / (hr0 - γ) ≤ slk) :
    (rS + γ) / (rU - γ) ≤ G + slk := by
  have hrU : 0 < rU := by linarith
  have hrU0 : 0 < rU - γ := by linarith
  have hhr0 : 0 < hr0 - γ := by linarith
  have hrat : rS / rU ≤ G := by rw [div_le_iff₀ hrU]; linarith
  have hstep : (rS + γ) / (rU - γ) ≤ rS / rU + 2 * γ / (rU - γ) := by
    rw [div_add_div _ _ (ne_of_gt hrU) (ne_of_gt hrU0), div_le_div_iff₀ hrU0 (by positivity)]
    nlinarith [mul_nonneg (mul_nonneg hrU0.le hγ.le) (show (0:ℝ) ≤ rU - rS by linarith)]
  have hmono : 2 * γ / (rU - γ) ≤ 2 * γ / (hr0 - γ) := by gcongr
  linarith [hstep, hrat, hmono, hslack]

/-- A single atom's normalized sample stays within `ca` when its ideal mass is at most `b` and the
slack bound holds. -/
lemma empirical_atom_le {rS rU b hr0 γ ca : ℝ} (hS0 : 0 ≤ rS) (hSb : rS ≤ b) (hr : hr0 ≤ rU)
    (hγ : 0 < γ) (hγr : γ < hr0) (hb : (b + γ) / (hr0 - γ) ≤ ca) :
    (rS + γ) / (rU - γ) ≤ ca := by
  have hrU0 : 0 < rU - γ := by linarith
  have hhr0 : 0 < hr0 - γ := by linarith
  calc (rS + γ) / (rU - γ) ≤ (b + γ) / (rU - γ) := by gcongr
    _ ≤ (b + γ) / (hr0 - γ) := by gcongr <;> linarith
    _ ≤ ca := hb

end GammaFacts

set_option maxHeartbeats 1000000 in
include hL in
/-- At a state and family, the round's bad draws: where level 3 fails at re-noised clustering
reads. Its chance is at most `δ + |Q||Σ| misledTail T (2 prefixMax) (2(|Q| + 1)ζ) u`, it is fixed
by the answers at the strings level 3 reads off the clustering's prefixes, and off it the actual
round meets its claim. -/
theorem round_bad_exists (E : σ → Set (FreeMonoid α)) (m u crossLimit Tu ca : ℝ) (hu : 0 ≤ u)
    (hTu : 0 ≤ Tu)
    (hk : 1 ≤ settings.level3.walkStart)
    (hη1 : O.η ≤ 1) (hcross : 0 < crossLimit)
    (hprem : RoundPremises σ α settings.level3 crossLimit
      (2 * prefixMax P.D settings.level3.walkStart))
    (hm1 : m ≤ 1 / 2)
    (B : State) (F K₀ : Finset (FreeMonoid α)) (hsf : SuffixFree F) (hcard : F.card + 1 ≤ B.k)
    (hband : ∀ N, N + 1 ≤ B.k → ∀ p r : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 →
      RarerBounded N B.lo (B.hi + 1) p r crossLimit)
    (hK₀ : K₀.card * prefixMax P.D settings.level3.walkStart ≤ m) (hm0 : 0 < m)
    (hcap : 2 * prefixMax P.D settings.level3.walkStart
      < ca * settings.level3.harvestRate) :
    ∃ Bd : Set (Ω × (Fin settings.level3.probeBudget → unitInterval)), MeasurableSet Bd
      ∧ (μ.prod (Measure.pi fun _ => volume)) Bd
        ≤ ENNReal.ofReal (settings.level3.δ + Fintype.card σ * Fintype.card α
          * misledTail settings.level3.probeBudget (2 * prefixMax P.D settings.level3.walkStart)
            (2 * (Fintype.card σ + 1) * crossLimit) u)
          + ENNReal.ofReal (2 * ((Finset.card (hypClass (α := α) P.L (Fintype.card σ)) : ℝ)
              * ((Finset.card (shortStrings (α := α) (P.L + Fintype.card σ)) : ℝ) + 1))
            * (2 * Real.exp (-2 * (settings.level3.probeBudget : ℝ)
              * roundGamma m ca (prefixMax P.D settings.level3.walkStart)
                  settings.level3.harvestRate ^ 2)))
      ∧ (∀ v ω ω', (∀ w ∈ roundZone (max (P.L + settings.level3.probeBudget + 2)
          (probesLen (probesOf settings P.D K₀ v) + 1 + settings.level3.probeBudget + 2))
          settings.level3.walkStart K₀ F, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
        (ω, v) ∈ Bd → (ω', v) ∈ Bd)
      ∧ ∀ x₀ Pr Sf ω v, (ω, v) ∉ Bd →
        RoundSpec M O P (roundSlope σ P.L settings.level3)
          (roundSlack σ α settings.level3 (2 * prefixMax P.D settings.level3.walkStart)) m
          (settings.level3.disagreementRate + m)
          (settings.level3.harvestRate / 2) Tu ca settings.level3.walkStart E (u + m)
          (mkRound settings O P.D x₀ B F K₀ Pr Sf (ω, v)) := by
  classical
  obtain ⟨hδ0, hδ1, hy0, hyρ, hρ1, hP, hκ0, t, t', ht, ht', htc, hfin⟩ := hprem
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
    ⟨Oc, M, F, B.lo, B.hi + 1, crossLimit, hL, hsf, hb⟩
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
  have h4 := level3Fail_le fam' hκ0 P' C E (2 * prefixMax P.D k) u t t' hpre' hδ0 hδ1 hy0
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
  set Sf := level3Fail fam' C D' (roundSlope σ P.L C) (roundSlack σ α C (2 * prefixMax P.D k)) E u
  have hSm : MeasurableSet Sf := measurableSet_level3Fail fam' C D' hA0 hsl0 E u
  -- the concentration bad set: where the empirical harvest overshoots the ideal by more than `γc`
  have hca0 : 0 < ca := by
    by_contra h
    push_neg at h
    nlinarith [hpm0, hy0, hcap, mul_nonpos_of_nonpos_of_nonneg h hy0.le]
  set γc := roundGamma m ca (prefixMax P.D k) C.harvestRate with hγc
  have hγ0 : 0 < γc := roundGamma_pos hm0 hca0.le hy0 hcap
  have hγlt : γc < C.harvestRate := roundGamma_lt hm0 hca0.le hy0
  have hdis0 : 0 < C.disagreementRate := by linarith [hy0, hyρ]
  have hT1 : 1 ≤ T := by
    set S : ℕ := Fintype.card σ * ((Fintype.card σ + 2) * Fintype.card α + 1) + 2 * Fintype.card α + 1
      with hS
    have hS1 : 1 ≤ S := by rw [hS]; exact Nat.le_add_left 1 _
    have hSR1 : (1 : ℝ) ≤ (S : ℝ) := by exact_mod_cast hS1
    have harg : (1 : ℝ) ≤ 16 * (S : ℝ) * ((S : ℝ) + 1) / C.δ := by
      rw [le_div_iff₀ hδ0]; nlinarith [hSR1, hδ1, hδ0]
    have hfirst : 0 ≤ 36 * (S : ℝ) / C.disagreementRate
        * Real.log (16 * (S : ℝ) * ((S : ℝ) + 1) / C.δ) :=
      mul_nonneg (div_nonneg (by positivity) hdis0.le) (Real.log_nonneg harg)
    have : (1 : ℝ) ≤ (C.probeBudget : ℝ) := by linarith [hP, hfirst, hSR1]
    exact_mod_cast this
  let Hf : HypothesisDFA α × Option (FreeMonoid α) → HypothesisDFA α := fun i => i.1
  let Sset : HypothesisDFA α × Option (FreeMonoid α) → Set (FreeMonoid α) := fun i =>
    (i.2).elim {z | familyReadDist O M F B.lo (B.hi + 1) (M.eval z.toList) .undecided ≤ Tu}
      (fun t => {t})
  let csI : Finset (HypothesisDFA α × Option (FreeMonoid α)) :=
    hypClass P.L (Fintype.card σ) ×ˢ
      insert none ((shortStrings (P.L + Fintype.card σ)).image some)
  let CBset : Random.Level3Site → Set ((Ω × Ω) × (Fin T → FreeMonoid α)) := fun site =>
    {p | ∃ i ∈ csI,
      γc < (Random.harvestLaw (fun z => familyRead fam'.O.mq F B.lo (B.hi + 1) z p.1) D' k
          (Hf i) site).real Set.univ ∧
        ((Random.harvestLaw (fun z => familyRead fam'.O.mq F B.lo (B.hi + 1) z p.1) D' k
            (Hf i) site).real (Sset i) + γc)
          / ((Random.harvestLaw (fun z => familyRead fam'.O.mq F B.lo (B.hi + 1) z p.1) D' k
            (Hf i) site).real Set.univ - γc)
          < (popLaw (Random.harvestSample (fun z => familyRead fam'.O.mq F B.lo (B.hi + 1) z p.1) k
            (Hf i) site p.2)).real (Sset i)}
  have hCBm : ∀ site, MeasurableSet (CBset site) := fun site =>
    Random.measurableSet_harvestSample_prod_bad k fam'.O F B.lo (B.hi + 1) site D' csI Hf Sset γc
  have hCBle : ∀ site, ((μ.prod μ).prod (Measure.pi fun _ : Fin T => D')) (CBset site)
      ≤ ENNReal.ofReal ((csI.card : ℝ) * (2 * Real.exp (-2 * (T : ℝ) * γc ^ 2))) := fun site =>
    Random.harvestSample_prod_biUnion_le D' k (μ.prod μ) csI T hT1 γc hγ0
      (fun p z => familyRead fam'.O.mq F B.lo (B.hi + 1) z p) Hf site Sset (hCBm site)
  have hcard : (csI.card : ℝ)
      = (Finset.card (hypClass (α := α) P.L (Fintype.card σ)) : ℝ)
        * ((Finset.card (shortStrings (α := α) (P.L + Fintype.card σ)) : ℝ) + 1) := by
    show ((hypClass (α := α) P.L (Fintype.card σ) ×ˢ
      insert none ((shortStrings (α := α) (P.L + Fintype.card σ)).image some)).card : ℝ) = _
    rw [Finset.card_product, Finset.card_insert_of_notMem (by simp),
      Finset.card_image_of_injective _ (Option.some_injective _)]
    push_cast; ring
  obtain ⟨ω₀, hω₀⟩ := exists_noise_le (μ := μ) (Measure.pi fun _ : Fin T => D')
    (Sf ∪ (CBset .ends ∪ CBset .middles)) (hSm.union ((hCBm _).union (hCBm _)))
  have hpr : MeasurePreserving (probesOf settings P.D K₀)
      (Measure.pi fun _ : Fin T => (volume : Measure unitInterval)) (Measure.pi fun _ => D') :=
    measurePreserving_pi _ _ fun _ => ⟨measurable_drawFrom _, map_drawFrom _⟩
  have hmp : MeasurePreserving (fun q : Ω × (Fin T → unitInterval) =>
      (q.1, probesOf settings P.D K₀ q.2)) (μ.prod (Measure.pi fun _ => volume))
      (μ.prod (Measure.pi fun _ => D')) := (MeasurePreserving.id μ).prod hpr
  have hSω : MeasurableSet {q : Ω × (Fin T → FreeMonoid α) |
      ((q.1, ω₀), q.2) ∈ Sf ∪ (CBset .ends ∪ CBset .middles)} :=
    ((measurable_fst.prodMk measurable_const).prodMk measurable_snd)
      (hSm.union ((hCBm _).union (hCBm _)))
  refine ⟨(fun q : Ω × (Fin T → unitInterval) => (q.1, probesOf settings P.D K₀ q.2)) ⁻¹'
    {q | ((q.1, ω₀), q.2) ∈ Sf ∪ (CBset .ends ∪ CBset .middles)}, hmp.measurable hSω,
    ?_, ?_, ?_⟩
  · rw [hmp.measure_preimage hSω.nullMeasurableSet]
    refine hω₀.trans ((measure_union_le _ _).trans (add_le_add h4 ?_))
    refine (measure_union_le _ _).trans ((add_le_add (hCBle _) (hCBle _)).trans ?_)
    rw [← ENNReal.ofReal_add (by positivity) (by positivity), hcard]
    exact ENNReal.ofReal_le_ofReal (le_of_eq (by ring))
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
    rcases hω with hSf | hCB
    · exact Or.inl (level3Fail_congr_up fam' C D' hD'len _ _ E u
        (probesOf settings P.D K₀ v) hread hSf)
    · exact Or.inr (by sorry)
  · -- off the bad draws, the actual round meets its claim
    intro x₀ Pr Sf' ω v hnot
    set xs := probesOf settings P.D K₀ v
    set read' : FreeMonoid α → ARU := fun z => fam'.read (ω, ω₀) z with hrd
    have hgood : ¬ ((ω, ω₀), xs) ∈ Sf := fun h => hnot (Or.inl h)
    have hnotConc : ((ω, ω₀), xs) ∉ (CBset .ends ∪ CBset .middles) := fun h => hnot (Or.inr h)
    obtain ⟨hE, hcls, hfew⟩ := not_not.1 hgood
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
    have hH : H = (Random.indecisiveLStarLevel3 C read' xs).hypothesis := by rw [hHd, hres]
    have hq := looseEndsWell_of_endsWell P.D M
      (fun q => familyReadDist O M F B.lo (B.hi + 1) q .undecided) hA0 hsl0
      (k := k) (K₀ := ↑K₀) (read := read) (read' := read') hoff H
      (disagreeSet read k H) (disagreeSet read' k H)
      (fun x hx h => by
        have hp : probeOrAgree read H.tree H.edges k x = probeOrAgree read' H.tree H.edges k x :=
          Random.probeOrAgree_congr_zone H.edges k fun j hj hjx m _ =>
            hoff _ (keyOff_of_prefix hx hj hjx m)
        simpa only [disagreeSet, Set.mem_union, Random.searchSet, Random.endsSet,
          Set.mem_ofPred_eq, hp] using h)
      C.disagreementRate C.harvestRate hy0.le
      (Random.indecisiveLStarLevel3 C read xs).ending (by linarith) (by
        rw [hres, hH, ← hdist]
        exact hE)
    -- a misled avoiding probe reads alike under both
    have hmis : ∀ a, ∀ x ∈ avoiding k (↑K₀ : Set (FreeMonoid α)),
        x ∈ misledSet M (familyReadDist O M F B.lo (B.hi + 1)) read k H.tree a →
        x ∈ misledSet M (familyReadDist O M F B.lo (B.hi + 1)) read' k H.tree a := by
      intro a x hx ⟨hkx, hr⟩
      have hoff' : ∀ z, prefixOf z k = prefixOf x k → k ≤ z.toList.length → read z = read' z :=
        fun z hz hkz => hoff z ⟨hkz, by rw [hz]; exact hx hkx⟩
      have hrr : ∀ z, prefixOf z k = prefixOf x k → k ≤ z.toList.length →
          z ∈ rarerSet M (familyReadDist O M F B.lo (B.hi + 1)) read k H.tree →
          z ∈ rarerSet M (familyReadDist O M F B.lo (B.hi + 1)) read' k H.tree := by
        rintro z hz hkz ⟨-, m, hm, hw⟩
        refine ⟨hkz, m, hm, ?_⟩
        have := hoff' (z * m) (by rw [prefixOf_mul_of_le hkz, hz]) (by simp; omega)
        beta_reduce at hw ⊢
        rw [← this]
        exact hw
      refine ⟨hkx, hr.imp (hrr x rfl hkx) (hrr _ ?_ (by simp; omega))⟩
      exact prefixOf_mul_of_le hkx _
    refine ⟨?_, ?_, fun q a => ?_, ?_⟩
    · have hq' := hq.mono (m' := m) (ε' := C.disagreementRate + m)
        (y' := C.harvestRate / 2) hm' (by linarith) (by nlinarith)
      exact hq'
    · show (Random.indecisiveLStarLevel3 C read xs).hypothesis.tree ∈ _
      rw [hres]
      exact hcls
    · have h := real_inter_le_of_avoiding P.D (E := E q) (hmis a) hu (by linarith)
        (by rw [hH, ← hdist]; exact hfew q a)
      exact h.trans (by linarith)
    · -- the empirical population's good/atom bounds, via concentration
      intro site hsite
      have hsite' : (Random.indecisiveLStarLevel3 C read xs).ending = .harvest site := hsite
      have hend : (Random.indecisiveLStarLevel3 C read' xs).ending = .harvest site := by
        rw [← hres]; exact hsite'
      have hcong : ∀ z ∈ Random.harvestReads k H xs, read z = read' z := by
        intro z hz
        rw [Random.harvestReads, Finset.mem_biUnion] at hz
        obtain ⟨i, -, hzi⟩ := hz
        obtain ⟨hkx, hpre⟩ := Random.pot_pre hzi
        have hkz : k ≤ z.toList.length := by
          have hlen : (prefixOf z k).toList.length = k := by
            rw [hpre, length_prefixOf, min_eq_left hkx]
          rw [length_prefixOf] at hlen; omega
        exact hoff z ⟨hkz, by rw [hpre]; exact hxs i hkx⟩
      have hbridge : Random.harvestSampleList read k H site (List.ofFn xs)
          = Random.harvestSample read' k H site xs := by
        rw [Random.harvestSampleList_ofFn, Random.harvestSample_congr_reads read k H site xs hcong]
      have hEh := hE
      rw [← hrd] at hEh
      simp only [EndsWell, hend] at hEh
      obtain ⟨hgoodT, hspread, hyield⟩ := hEh
      rw [← hH] at hgoodT hspread hyield
      have hUrate : C.harvestRate ≤ (Random.harvestLaw read' D' k H site).real Set.univ := hyield
      have hUγ : γc < (Random.harvestLaw read' D' k H site).real Set.univ :=
        lt_of_lt_of_le hγlt hUrate
      have hUfin : (Random.harvestLaw read' D' k H site) Set.univ ≠ ⊤ :=
        Random.harvestLaw_univ_ne_top read' D' k H site
      have hHmem : H ∈ hypClass (α := α) P.L (Fintype.card σ) :=
        hypothesis_mem_hypClass C read xs
          (fun j => drawFrom_avoidLaw_len P.D P.length_le k K₀ (v j)) (by rw [hres]; exact hcls)
      have hns : ((ω, ω₀), xs) ∉ CBset site := by
        cases site with
        | ends => exact fun h => hnotConc (Or.inl h)
        | middles => exact fun h => hnotConc (Or.inr h)
      show (popLaw (Random.harvestSampleList read k H site (List.ofFn xs))).real
            {z | familyReadDist O M F B.lo (B.hi + 1) (M.eval z.toList) .undecided ≤ Tu}
            ≤ roundSlope σ P.L C * Tu + roundSlack σ α C (2 * prefixMax P.D k)
              + m / (C.harvestRate / 2)
          ∧ ∀ t, (popLaw (Random.harvestSampleList read k H site (List.ofFn xs))).real {t} ≤ ca
      rw [hbridge]
      rw [hdist] at hgoodT
      refine ⟨?_, fun t => ?_⟩
      · have hmemG : ((H, none) : HypothesisDFA α × Option (FreeMonoid α)) ∈ csI :=
          Finset.mk_mem_product hHmem (Finset.mem_insert_self _ _)
        have hpopG : (popLaw (Random.harvestSample read' k H site xs)).real (Sset (H, none))
            ≤ ((Random.harvestLaw read' D' k H site).real (Sset (H, none)) + γc)
              / ((Random.harvestLaw read' D' k H site).real Set.univ - γc) := by
          by_contra hcon
          push_neg at hcon
          exact hns ⟨(H, none), hmemG, hUγ, hcon⟩
        refine hpopG.trans ?_
        refine empirical_ratio_le (measureReal_mono (Set.subset_univ _) hUfin) (hgoodT Tu hTu)
          hUrate hγ0 hγlt ?_
        rw [hγc]; exact roundGamma_good hm0 hca0.le hy0 hcap
      · by_cases ht : t ∈ shortStrings (α := α) (P.L + Fintype.card σ)
        · have hmemA : ((H, some t) : HypothesisDFA α × Option (FreeMonoid α)) ∈ csI :=
            Finset.mk_mem_product hHmem
              (Finset.mem_insert_of_mem (Finset.mem_image_of_mem some ht))
          have hpopA : (popLaw (Random.harvestSample read' k H site xs)).real (Sset (H, some t))
              ≤ ((Random.harvestLaw read' D' k H site).real (Sset (H, some t)) + γc)
                / ((Random.harvestLaw read' D' k H site).real Set.univ - γc) := by
            by_contra hcon
            push_neg at hcon
            exact hns ⟨(H, some t), hmemA, hUγ, hcon⟩
          refine hpopA.trans ?_
          refine empirical_atom_le (b := 2 * prefixMax P.D k) measureReal_nonneg ?_ hUrate hγ0
            hγlt ?_
          · refine (hspread t).trans ?_
            have hseteq : {p : FreeMonoid α | p.toList.take k = t.toList.take k}
                = {x | prefixOf x k = prefixOf t k} := by
              ext p
              simp only [Set.mem_setOf_eq, prefixOf]
              constructor
              · intro h; rw [h]
              · intro h; have := congrArg FreeMonoid.toList h; simpa using this
            rw [hseteq]; exact hpre' (prefixOf t k)
          · rw [hγc]; exact roundGamma_atom hm0 hca0.le hy0 hcap
        · have htlen : P.L + Fintype.card σ < t.toList.length := by
            by_contra h; push_neg at h; exact ht (mem_shortStrings.2 h)
          have ht1 : t ≠ 1 := by intro h; rw [h] at htlen; simp at htlen
          have hclsH : H.tree ∈ Random.classSet (Fintype.card σ) := by rw [hHd, hres]; exact hcls
          have hnull : Random.harvestSample read' k H site xs {t} = 0 := by
            rw [← Random.harvestSampleList_ofFn read' k H site xs]
            refine measure_mono_null (Set.singleton_subset_iff.2 htlen)
              (Random.harvestSampleList_short read' k (Random.classSet_props _ _ hclsH).1 site
                (List.ofFn xs) fun x hx => ?_)
            obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
            exact drawFrom_avoidLaw_len P.D P.length_le k K₀ (v i)
          have h1 : (1 : FreeMonoid α) ∉ ({t} : Set (FreeMonoid α)) := by
            simp only [Set.mem_singleton_iff]; exact Ne.symm ht1
          rw [measureReal_def, popLaw_null hnull h1]
          simpa using hca0.le

end Bad

end OrthoDFA
