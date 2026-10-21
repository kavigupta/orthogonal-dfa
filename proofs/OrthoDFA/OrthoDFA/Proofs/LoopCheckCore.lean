import OrthoDFA.Proofs.LoopCheckAcc
import OrthoDFA.Proofs.LoopCheckHead
import OrthoDFA.Proofs.LoopCheckSched

/-!
# The checked loop ends at its first round with an accurate DFA

A run fails where the first round's clustering misses its claim on the uniform pool over a fresh
noise or at the run's own (`hfresh`, `huni`), draws by a uniform equal to `1`, which has chance
zero (`restLt_ae`), fails to end consistent at a good stage (`check_round0_le`), or ends
consistent at a hypothesis failing more than the check's rate (`acc_round0_le`). Off those, the
family reads every state undecided at most `u` of the time, through the uniform pool, the start's
re-rooting making every state read as a covered one (`undecided_le_of_covered`); the round ends
consistent, so the loop stops there, and its DFA from where `ε` sifts misjudges at most the check's
rate, what the check strings avoid, and the family's misreads of the uniform pool
(`check_first_core`).
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory Random

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

omit [DecidableEq α] [IsProbabilityMeasure μ] in
lemma popOff_none (D : Measure (FreeMonoid α)) (pops : List (Measure (FreeMonoid α)))
    (T : Finset (FreeMonoid α)) : popOff D pops (none : Option (Fin K)) T = offLaw D T := by
  unfold popOff
  split_ifs <;> rfl

/-- Off its avoided strings, a law is at most its conditioned law plus what it avoids. -/
lemma real_le_avoidLaw (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (K₀ : Finset (FreeMonoid α)) (S : Set (FreeMonoid α)) :
    D.real S ≤ (avoidLaw D k K₀).real S + D.real (avoiding k ↑K₀)ᶜ := by
  classical
  set A := avoiding (α := α) k ↑K₀
  have hsplit : D.real S ≤ D.real (S ∩ A) + D.real Aᶜ :=
    (measureReal_mono (fun x hx => by
      by_cases h : x ∈ A
      · exact Or.inl ⟨hx, h⟩
      · exact Or.inr h)).trans (measureReal_union_le _ _)
  refine hsplit.trans (add_le_add ?_ le_rfl)
  by_cases h0 : D A = 0
  · rw [measureReal_def, measure_mono_null Set.inter_subset_right h0, ENNReal.toReal_zero]
    exact measureReal_nonneg
  · rw [avoidLaw_eq D k K₀]
    unfold popLaw
    rw [if_neg (by rw [Measure.restrict_apply_univ]; exact fun h => h.elim h0 (measure_ne_top _ _))]
    have e : (((D.restrict A) Set.univ)⁻¹ • D.restrict A).real S
        = (D A).toReal⁻¹ * (D (S ∩ A)).toReal := by
      rw [measureReal_def, Measure.smul_apply, smul_eq_mul, Measure.restrict_apply
        (measurableSet_of_countable S), Measure.restrict_apply_univ, ENNReal.toReal_mul,
        ENNReal.toReal_inv]
    rw [e]
    have hA1 : (D A).toReal ≤ 1 := measureReal_le_one
    have hA0 : 0 < (D A).toReal := ENNReal.toReal_pos h0 (measure_ne_top _ _)
    rw [le_inv_mul_iff₀ hA0]
    exact mul_le_of_le_one_left ENNReal.toReal_nonneg hA1

/-- The first round's clustering draws, as the clustering's own law. -/
theorem measurePreserving_round0 (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α) (T n : ℕ)
    (hK : 0 < K) :
    MeasurePreserving (fun p : Ω × (Fin K → CheckRoundDraws α K T n) =>
        clusterDraws D Dsf h ((p.1, (p.2 ⟨0, hK⟩).1) : ClusterDraws Ω α K))
      (checkLoopMeasure μ α K T n)
      (runMeasure μ (fun j => popOff D h.pops j h.prefs) (offLaw Dsf h.suffs)) := by
  have hp : MeasurePreserving (fun p : Ω × (Fin K → CheckRoundDraws α K T n) =>
      ((p.1, (p.2 ⟨0, hK⟩).1) : ClusterDraws Ω α K)) (checkLoopMeasure μ α K T n)
      (clusterLaw μ α K) := by
    have h1 : MeasurePreserving (fun x : CheckRoundDraws α K T n => x.1) (checkRoundLaw α K T n)
        (restLaw α K) := by
      unfold checkRoundLaw; exact measurePreserving_fst
    exact (MeasurePreserving.id μ).prod (h1.comp (measurePreserving_eval
      (fun _ : Fin K => checkRoundLaw α K T n) ⟨0, hK⟩))
  exact (measurePreserving_clusterDraws D Dsf h).comp hp

/-- The rounds after a first round that does not end in a harvest are that round alone. -/
theorem checkRounds_of_first {K : ℕ} (settings : CheckLoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Du Dsf Dc : Measure (FreeMonoid α)) (h₀ : LoopHist α)
    (p : Ω × (Fin K → CheckRoundDraws α K settings.level3.probeBudget settings.check.size))
    (hK : 0 < K)
    (hn : (checkRound settings O D Du Dsf Dc h₀ (p.1, p.2 ⟨0, hK⟩)).result.ending.harvested = none) :
    ∀ r, 1 ≤ r → checkRounds settings O D Du Dsf Dc h₀ p r
      = [checkRound settings O D Du Dsf Dc h₀ (p.1, p.2 ⟨0, hK⟩)]
  | 0, h => absurd h (by omega)
  | 1, _ => by
    have e : h₀.append (histOf D settings.level3.walkStart ([] : List (LoopRound Ω α K))) = h₀ := by
      simp [LoopHist.append, histOf]
    simp only [checkRounds, List.not_mem_nil, false_imp_iff, implies_true, and_true, hK,
      dif_pos, List.nil_append, e]
  | r + 2, _ => by
    have ih := checkRounds_of_first settings O D Du Dsf Dc h₀ p hK hn (r + 1) (by omega)
    rw [checkRounds]
    split_ifs with hc
    · exact absurd hn (hc.2 _ (by rw [ih]; exact List.mem_singleton_self _))
    · exact ih

section Core

variable [Nonempty α] {σ : Type*} [Fintype σ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
  (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)

/-- The clustering's claim on the uniform pool over a fresh noise. -/
def FreshSpec (settings : LoopSettings K) (pops : Finset (Option (Fin K)))
    (Du : Measure (FreeMonoid α)) (slack : ℝ) (x : Run Ω (FreeMonoid α) (Option (Fin K))) : Prop :=
  (∃ B ∈ settings.schedule pops, x ∈ ret O.mq pops none settings.η₀ settings.indecisionLimit
      settings.εcov settings.gateLevel B)
    ∧ ∀ B ∈ settings.schedule pops, x ∈ ret O.mq pops none settings.η₀ settings.indecisionLimit
      settings.εcov settings.gateLevel B →
      ∫ p, miscutProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p ∂Du
          ≤ 513 / 512 * settings.εcov + slack
        ∧ ∫ p, undecidedProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p ∂Du
          ≤ 2 * settings.indecisionLimit + slack

set_option maxHeartbeats 1000000 in
include hL in
/-- The checked loop, started from `h₀`, ends at its first round, consistent, with its DFA from
where `ε` sifts misjudging at most the check's rate, the check strings' avoided mass and the
family's misreads of the uniform pool, but for the chances its parts fail. -/
theorem check_first_core (settings : CheckLoopSettings K) (hK : 0 < K)
    (P Pc : PrefixDistribution α) (Du Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Du]
    [IsProbabilityMeasure Dsf]
    (h₀ : LoopHist α) (L Ls NP NK h : ℕ) (c slack crossLimit p₀ u ph ηc cN m₀ Nw e₁ e₂ : ℝ)
    -- the walk and the depths
    (hz : settings.avoidDepth = settings.level3.walkStart)
    (hk : 1 ≤ settings.level3.walkStart) (ha : 1 ≤ settings.checkDepth)
    (hak : settings.checkDepth + Fintype.card σ + 1 ≤ settings.level3.walkStart)
    (hη1 : O.η ≤ 1) (hcross : 0 < crossLimit)
    -- the schedule
    (hband : ∀ pops, ∀ B ∈ settings.schedule pops, B.lo ≤ B.hi ∧ 2 ≤ B.k ∧ ∀ N, N + 1 ≤ B.k →
      ∀ p r ε₂ κ : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → 0 ≤ κ →
        BandPasses N B.lo (B.hi + 1) p r crossLimit ε₂ κ)
    (hNP : ∀ pops, ∑ B ∈ settings.schedule pops, B.npref ≤ 2 * NP)
    (hNK : 6 * (K + 1) * NP ≤ NK)
    (hfresh : (runMeasure μ (fun j => popOff Du h₀.pops j h₀.prefs) (offLaw Dsf h₀.suffs)).real
      {x | ¬ FreshSpec O settings.toLoopSettings (loopPopulations K h₀.pops.length)
        (popOff Du h₀.pops (none : Option (Fin K)) h₀.prefs) slack (streams (fun _ => ∅) none x)} ≤ e₁)
    (huni : (runMeasure μ (fun j => popOff Du h₀.pops j h₀.prefs) (offLaw Dsf h₀.suffs)).real
      {x | ¬ UniSpec O (loopPopulations K h₀.pops.length)
        (settings.schedule (loopPopulations K h₀.pops.length)) settings.η₀
        settings.indecisionLimit settings.εcov settings.gateLevel
        (popOff Du h₀.pops (none : Option (Fin K)) h₀.prefs) (streams (fun _ => ∅) none x)} ≤ e₂)
    -- the lengths the clustering reads
    (hDlen : Du {q | ¬ (settings.level3.walkStart ≤ q.toList.length ∧ q.toList.length < Ls)} = 0)
    (hpops : ∀ ν ∈ h₀.pops, IsProbabilityMeasure ν
      ∧ ν {q | ¬ (settings.level3.walkStart ≤ q.toList.length ∧ q.toList.length < Ls)} = 0)
    (hDsf : Dsf {v | v.toList.length ≠ Ls} = 0)
    (hpm : Du.real ↑h₀.prefs ≤ m₀) (hm₀ : m₀ ≤ 1 / 2) (hsm : Dsf.real ↑h₀.suffs ≤ 1 / 2)
    -- the start's re-rooting and the uniform pool's occupancy
    {qs : σ} (hq : CoveredAccurate M L c qs) {ps : FreeMonoid α} (hps : M.eval ps.toList = qs)
    (hLs : (L + 1) / 2 ≤ Ls) (hLs2 : (L + 1) / 2 + Ls ≤ 2 * L)
    (hLσ : (L + 1) / 2 + Fintype.card σ * Fintype.card σ ≤ 2 * L)
    (hσ : Fintype.card σ ≤ (L + 1) / 2)
    (hcov : ∀ q ∈ coveredStates M L c, cN ≤ Du.real {z | M.eval z.toList = q})
    (hu : 2 * settings.indecisionLimit + slack < u * (cN - m₀))
    -- the check strings
    (hNKk : NK * prefixMax P.D settings.level3.walkStart ≤ 1 / 2)
    (hp₀ : 2 * prefixMax P.D settings.level3.walkStart ≤ p₀)
    (hNKa : NK * prefixMax Pc.D settings.checkDepth ≤ 1 / 2)
    (hph : 2 * prefixMax Pc.D h ≤ ph)
    (hDc : ∀ S : Set (FreeMonoid α), Pc.D.real S ≤ Nw * Du.real S) (hNw : 0 ≤ Nw) :
    let C := settings.level3
    let Kk := settings.check
    let y := C.harvestRate
    let ε := C.disagreementRate
    let δ := C.δ
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    let Kt : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα)
    let A : ℝ := (nQ + 1) * (6 / y + 16 * (P.L + 1))
    let η := p₀ * Real.log (16 * (nQ + 1) * Kt / δ)
    let Ms : ℝ := (nQ + 1) * nα ^ nQ
    let θ₀ : ℝ := (Pc.L + 1) * Ms * u
    let q : ℝ := 2 * θ₀ + ηc
    0 < δ → 0 < y → y ≤ ε / 48 → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) / δ) + S ≤ C.probeBudget →
    0 ≤ u → 0 < ph → h ≤ Pc.L →
    0 ≤ ηc → 8 * θ₀ ≤ 1 → q ≤ 1 → 0 < Kk.rate → Kk.rate < 1 → 0 < Kk.δ →
    crossLimit * C.probeBudget * ((P.L + 1) * nQ + Kk.size * ((Pc.L + 1) * nQ)
      + (Kk.size + 1) * ((nQ + 1) ^ 2 * nα)) ≤ δ / 8 →
    A * u + η / y < 1 →
    (checkLoopMeasure μ α K C.probeBudget Kk.size).real
      {p | ¬ ∃ o, indecisiveLStarCheckLoop settings O P.D Du Dsf Pc.D h₀ p = [o]
          ∧ o.result.ending = .consistent
          ∧ ∃ s, o.result.hypothesis.tree.sift o.read 1 = .inl s
            ∧ Pc.D.real {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)}
              ≤ Kk.rate + NK * prefixMax Pc.D settings.checkDepth
                + Nw * (513 / 512 * settings.εcov + 2 * settings.indecisionLimit + m₀)}
      ≤ e₁ + e₂ + (9 * δ / 8 + 2 * (∑ j ∈ Finset.range S, Kk.silentBase q ^ Kk.size
            / Kk.level j + (h + 1) * nα ^ h * Ms * u + Real.exp (-(ηc / ph)))) + Kk.δ / 8 := by
  intro C Kk y ε δ nQ nα S Kt A η Ms θ₀ q hδ hy0 hyρ hρ1 hP hu0 hph0 hhL hηc h8 hq1 hr0 hr1 hKδ
    hw hAu
  classical
  set T := C.probeBudget
  set n := Kk.size
  set k := C.walkStart
  set a := settings.checkDepth
  set ls := settings.toLoopSettings
  set pops := loopPopulations K h₀.pops.length
  set sched := settings.schedule pops
  set r0 : Fin K := ⟨0, hK⟩
  set Dn := popOff Du h₀.pops (none : Option (Fin K)) h₀.prefs
  have hDn : Dn = offLaw Du h₀.prefs := popOff_none Du h₀.pops h₀.prefs
  have : IsProbabilityMeasure (checkLoopMeasure μ α K T n) := by
    unfold checkLoopMeasure; infer_instance
  set μL := checkLoopMeasure μ α K T n
  set xr : Ω × (Fin K → CheckRoundDraws α K T n) → Run Ω (FreeMonoid α) (Option (Fin K)) :=
    fun p => clusterRun Du Dsf h₀ (p.1, (p.2 r0).1)
  set bound := 9 * δ / 8 + 2 * (∑ j ∈ Finset.range S, Kk.silentBase q ^ n / Kk.level j
    + (h + 1) * nα ^ h * Ms * u + Real.exp (-(ηc / ph)))
  have hpm' : Du.real ↑h₀.prefs ≤ 1 / 2 := hpm.trans hm₀
  -- the bad events
  set E₁ := {p : Ω × (Fin K → CheckRoundDraws α K T n) |
    ¬ FreshSpec O ls pops Dn slack (xr p)}
  set E₂ := {p : Ω × (Fin K → CheckRoundDraws α K T n) |
    ¬ UniSpec O pops sched settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel Dn
      (xr p)}
  set E₃ := {p : Ω × (Fin K → CheckRoundDraws α K T n) | ¬ RestLt (p.2 r0).1}
  set E₄ := {p : Ω × (Fin K → CheckRoundDraws α K T n) | ∃ t, stage0 ls O Du Dsf h₀ a
      (p.1, (p.2 r0).1) = some t
    ∧ GoodStage M O sched u NK t ∧ LensOK ls Du Dsf h₀ k Ls (p.2 r0).1
    ∧ (checkRound settings O P.D Du Dsf Pc.D h₀ (p.1, p.2 r0)).result.ending ≠ .consistent}
  set E₅ := {p : Ω × (Fin K → CheckRoundDraws α K T n) | ∃ t, stage0 ls O Du Dsf h₀ a
      (p.1, (p.2 r0).1) = some t ∧ (p.1, (p.2 r0).2) ∈ accBad settings O P.D Pc.D t}
  have hmp := measurePreserving_round0 (μ := μ) Du Dsf h₀ T n hK
  have hE₁ : μL.real E₁ ≤ e₁ := by
    have hm : MeasurableSet {x : Draws Ω (FreeMonoid α) (Option (Fin K)) |
        ¬ FreshSpec O ls pops Dn slack (streams (fun _ => ∅) none x)} :=
      (measurable_streams _ none (measurableSet_specAt O pops sched settings.η₀
        settings.indecisionLimit settings.εcov settings.gateLevel (fun B F =>
          ∫ p, miscutProb O B.lo (B.hi + 1) F p ∂Dn ≤ 513 / 512 * settings.εcov + slack
          ∧ ∫ p, undecidedProb O B.lo (B.hi + 1) F p ∂Dn
            ≤ 2 * settings.indecisionLimit + slack))).compl
    have e : E₁ = (fun p : Ω × (Fin K → CheckRoundDraws α K T n) =>
        clusterDraws Du Dsf h₀ ((p.1, (p.2 r0).1) : ClusterDraws Ω α K)) ⁻¹'
          {x | ¬ FreshSpec O ls pops Dn slack (streams (fun _ => ∅) none x)} := rfl
    rw [e, measureReal_def, hmp.measure_preimage hm.nullMeasurableSet, ← measureReal_def]
    exact hfresh
  have hE₂ : μL.real E₂ ≤ e₂ := by
    have : IsProbabilityMeasure Dn := by rw [hDn]; infer_instance
    have hm : MeasurableSet {x : Draws Ω (FreeMonoid α) (Option (Fin K)) |
        ¬ UniSpec O pops sched settings.η₀ settings.indecisionLimit settings.εcov
          settings.gateLevel Dn (streams (fun _ => ∅) none x)} :=
      (measurable_streams _ none (measurableSet_uniSpec O pops sched settings.η₀
        settings.indecisionLimit settings.εcov settings.gateLevel Dn)).compl
    have e : E₂ = (fun p : Ω × (Fin K → CheckRoundDraws α K T n) =>
        clusterDraws Du Dsf h₀ ((p.1, (p.2 r0).1) : ClusterDraws Ω α K)) ⁻¹'
          {x | ¬ UniSpec O pops sched settings.η₀ settings.indecisionLimit settings.εcov
            settings.gateLevel Dn (streams (fun _ => ∅) none x)} := rfl
    rw [e, measureReal_def, hmp.measure_preimage hm.nullMeasurableSet, ← measureReal_def]
    exact huni
  have hE₃ : μL.real E₃ = 0 := by
    have hpr : MeasurePreserving (fun p : Ω × (Fin K → CheckRoundDraws α K T n) => (p.2 r0).1)
        μL (restLaw α K) :=
      measurePreserving_fst.comp ((measurePreserving_eval _ r0).comp measurePreserving_snd)
    have hm : MeasurableSet {c : ClusterRest α K | ¬ RestLt c} := by
      have : {c : ClusterRest α K | ¬ RestLt c} = ({c | RestLt c})ᶜ := rfl
      rw [this]
      refine MeasurableSet.compl ?_
      have h1 : MeasurableSet {u : unitInterval | (u : ℝ) < 1} :=
        measurableSet_lt continuous_subtype_val.measurable measurable_const
      have : {c : ClusterRest α K | RestLt c} = (⋂ m : ℕ, {c | ((c.1 m : unitInterval) : ℝ) < 1})
          ∩ ((⋂ j, ⋂ m : ℕ, {c | ((c.2.1 j m : unitInterval) : ℝ) < 1})
            ∩ ⋂ j, ⋂ m : ℕ, {c | ((c.2.2.1 j m : unitInterval) : ℝ) < 1}) := by
        ext c; simp [RestLt]
      rw [this]
      refine (MeasurableSet.iInter fun m => ?_).inter ((MeasurableSet.iInter fun j =>
        MeasurableSet.iInter fun m => ?_).inter (MeasurableSet.iInter fun j =>
          MeasurableSet.iInter fun m => ?_))
      · exact measurableSet_lt (continuous_subtype_val.measurable.comp
          ((measurable_pi_apply m).comp measurable_fst)) measurable_const
      · exact measurableSet_lt (continuous_subtype_val.measurable.comp ((measurable_pi_apply m).comp
          ((measurable_pi_apply j).comp (measurable_fst.comp measurable_snd)))) measurable_const
      · exact measurableSet_lt (continuous_subtype_val.measurable.comp ((measurable_pi_apply m).comp
          ((measurable_pi_apply j).comp (measurable_fst.comp (measurable_snd.comp
            measurable_snd))))) measurable_const
    have e : E₃ = (fun p : Ω × (Fin K → CheckRoundDraws α K T n) => (p.2 r0).1) ⁻¹'
        {c | ¬ RestLt c} := rfl
    rw [e, measureReal_def, hpr.measure_preimage hm.nullMeasurableSet, restLt_ae,
      ENNReal.toReal_zero]
  have hE₄ : μL.real E₄ ≤ bound := by
    have h := check_round0_le M O hL settings P Pc Du Dsf h₀ hK Ls NK h crossLimit p₀ u ph ηc hz hk ha
      hak hη1 hcross (fun pops B hB => (hband pops B hB).2.2) hNKk hp₀ hNKa hph hδ hy0 hyρ hρ1 hP
      hu0 hph0 hhL hηc h8 hq1 hr0 hr1 hKδ hw hAu
    exact ENNReal.toReal_le_of_le_ofReal (by
      have hq0 : 0 ≤ q := by positivity
      have hsb : 0 ≤ Kk.silentBase q := by
        unfold CheckSettings.silentBase
        have : 0 ≤ (1 - Kk.rate) / (1 - Kk.rate / 2) := div_nonneg (by linarith) (by linarith)
        have : 0 ≤ 1 - q := by linarith
        positivity
      have : 0 ≤ ∑ j ∈ Finset.range S, Kk.silentBase q ^ n / Kk.level j :=
        Finset.sum_nonneg fun j _ => div_nonneg (pow_nonneg hsb _) (by
          unfold CheckSettings.level; positivity)
      positivity) h
  have hE₅ : μL.real E₅ ≤ Kk.δ / 8 :=
    ENNReal.toReal_le_of_le_ofReal (by positivity)
      (acc_round0_le settings O P.D Du Dsf Pc.D h₀ hK hKδ hr0 hr1)
  -- off the bad events, the loop stops at its first round with an accurate DFA
  have hsub : {p | ¬ ∃ o, indecisiveLStarCheckLoop settings O P.D Du Dsf Pc.D h₀ p = [o]
      ∧ o.result.ending = .consistent
      ∧ ∃ s, o.result.hypothesis.tree.sift o.read 1 = .inl s
        ∧ Pc.D.real {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)}
          ≤ Kk.rate + NK * prefixMax Pc.D a
            + Nw * (513 / 512 * settings.εcov + 2 * settings.indecisionLimit + m₀)}
      ⊆ E₁ ∪ E₂ ∪ E₃ ∪ E₄ ∪ E₅ := by
    intro p hp
    by_contra hno
    simp only [Set.mem_union, not_or] at hno
    obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩ := hno
    have hfr : FreshSpec O ls pops Dn slack (xr p) := not_not.1 h1
    have hus : UniSpec O pops sched settings.η₀ settings.indecisionLimit settings.εcov
        settings.gateLevel Dn (xr p) := not_not.1 h2
    have hlt : RestLt (p.2 r0).1 := not_not.1 h3
    have hl := lensOK_of_restLt ls Du Dsf h₀ k Ls hDlen hpops hDsf hpm' hsm (p.2 r0).1 hlt
    set cd : ClusterDraws Ω α K := (p.1, (p.2 r0).1)
    have hret : Returns ls O Du Dsf h₀ cd := hfr.1
    set B := roundState ls O Du Dsf h₀ cd
    set F := clusterAt O.mq pops (clusterRun Du Dsf h₀ cd) B
    set K₀ := roundKeys ls Du Dsf h₀ cd
    set Kc := clusterKeys sched pops a (clusterRun Du Dsf h₀ cd)
    have hst : stage0 ls O Du Dsf h₀ a cd = some (B, F, K₀, Kc) :=
      (stage0_eq_some ls O Du Dsf h₀ a cd B F K₀ Kc).2 ⟨hret, rfl, rfl, rfl, rfl⟩
    have hBs : B ∈ sched := (Classical.epsilon_spec hret).1
    have hBret : xr p ∈ ret O.mq pops none settings.η₀ settings.indecisionLimit settings.εcov
        settings.gateLevel B := (Classical.epsilon_spec hret).2
    have hFs : F ⊆ suffsOf ls Du Dsf h₀ (p.2 r0).1 :=
      clusterAt_subset_suffsOf ls O Du Dsf h₀ p.1 (p.2 r0).1 hBs
    have hlenF : ∀ v ∈ F, v.toList.length = Ls := fun v hv => hl.1 v (hFs hv)
    have hF1 : (1 : FreeMonoid α) ∉ F := fun h' => by
      have := hlenF 1 h'
      have hσ0 : 0 < Fintype.card σ := Fintype.card_pos_iff.2 ⟨M.start⟩
      simp at this
      omega
    -- every state reads undecided at most `u` of the time
    have hUz : ∀ z : FreeMonoid α,
        familyReadDist O M F B.lo (B.hi + 1) (M.eval z.toList) .undecided ≤ u := by
      have hgate := (hfr.2 B hBs hBret).2
      have hcovu : ∀ q' ∈ coveredStates M L c,
          familyReadDist O M F B.lo (B.hi + 1) q' .undecided ≤ u := by
        intro q' hq'
        have hmass : cN - m₀ ≤ Dn.real {z | M.eval z.toList = q'} := by
          rw [hDn]
          have := le_offLaw_real hpm' {z : FreeMonoid α | M.eval z.toList = q'}
          linarith [hcov q' hq']
        have : IsProbabilityMeasure Dn := by rw [hDn]; infer_instance
        refine good_of_heavy M O hL hF1 B.lo B.hi Dn q' ?_ hgate
        calc 2 * settings.indecisionLimit + slack < u * (cN - m₀) := hu
          _ ≤ u * Dn.real {z | M.eval z.toList = q'} := mul_le_mul_of_nonneg_left hmass hu0
      exact undecided_le_of_covered M O hL hq hps hlenF hLs hLs2 hLσ hσ B.lo (B.hi + 1) hcovu
    have hgood : GoodStage M O sched u NK (B, F, K₀, Kc) := by
      refine ⟨hBs, ?_, ?_, hUz, ?_, ?_⟩
      · intro v hv v' hv' hsuf
        apply FreeMonoid.toList.injective
        exact hsuf.eq_of_length ((hlenF v hv).trans (hlenF v' hv').symm)
      · obtain ⟨-, hk2, -⟩ := hband _ B hBs
        have h1 := clusterAt_card_add_one (rule := lloydClusterer) O pops B (clusterRun Du Dsf h₀ cd)
        have h2 := lloydClusterer.card_le (fun w => O.mq w (oracleNoise (clusterRun Du Dsf h₀ cd)) = 1)
          (prefixesAt pops B.npref (clusterRun Du Dsf h₀ cd))
          (screenedAt O.mq pops B (clusterRun Du Dsf h₀ cd)) B.k (by omega)
        show F.card + 1 ≤ B.k
        simp only [F]
        rw [clusterAt_eq]
        unfold clusterFull at h1
        omega
      · show K₀.card ≤ NK
        refine Finset.card_image_le.trans ((card_roundPrefs_le _ _ _).trans ?_)
        have hpc : pops.card ≤ K + 1 := by
          have := Finset.card_le_univ pops
          simpa using this
        calc 3 * pops.card * ∑ B ∈ sched, B.npref ≤ 3 * (K + 1) * (2 * NP) :=
              Nat.mul_le_mul (Nat.mul_le_mul_left 3 hpc) (hNP pops)
          _ = 6 * (K + 1) * NP := by ring
          _ ≤ NK := hNK
      · show Kc.card ≤ NK
        refine Finset.card_image_le.trans ((card_roundPrefs_le _ _ _).trans ?_)
        have hpc : pops.card ≤ K + 1 := by
          have := Finset.card_le_univ pops
          simpa using this
        calc 3 * pops.card * ∑ B ∈ sched, B.npref ≤ 3 * (K + 1) * (2 * NP) :=
              Nat.mul_le_mul (Nat.mul_le_mul_left 3 hpc) (hNP pops)
          _ = 6 * (K + 1) * NP := by ring
          _ ≤ NK := hNK
    -- the round ends consistent
    set o := checkRound settings O P.D Du Dsf Pc.D h₀ (p.1, p.2 r0)
    have hcons : o.result.ending = .consistent := by
      by_contra hc
      exact h4 ⟨_, hst, hgood, hl, hc⟩
    have hrounds : indecisiveLStarCheckLoop settings O P.D Du Dsf Pc.D h₀ p = [o] :=
      checkRounds_of_first settings O P.D Du Dsf Pc.D h₀ p hK (by rw [hcons]; rfl) K hK
    -- its check passed at its rate
    have hres : o.result = indecisiveLStarCheck C Kk
        (fun z => familyRead O.mq F B.lo (B.hi + 1) z p.1)
        (checkDrawsOf Kk P.D Pc.D settings.avoidDepth a K₀ Kc (p.2 r0).2) := rfl
    have hread : o.read = fun z => familyRead O.mq F B.lo (B.hi + 1) z p.1 := rfl
    have hrate : (avoidLaw Pc.D a Kc).real {x | startFails o.read o.result.hypothesis.tree
        o.result.hypothesis.edges x = true} ≤ Kk.rate := by
      by_contra hc
      push Not at hc
      refine h5 ⟨_, hst, ?_⟩
      refine ⟨by rw [← hres]; exact hcons, ?_⟩
      rw [← hres]
      exact hc
    rw [hread] at hrate
    apply hp
    refine ⟨o, hrounds, hcons, ?_⟩
    have hx₀ : ∃ x, startFails (fun z => familyRead O.mq F B.lo (B.hi + 1) z p.1)
        o.result.hypothesis.tree o.result.hypothesis.edges x = false := by
      by_contra hall
      push Not at hall
      have : (avoidLaw Pc.D a Kc).real {x | startFails
          (fun z => familyRead O.mq F B.lo (B.hi + 1) z p.1) o.result.hypothesis.tree
          o.result.hypothesis.edges x = true} = 1 := by
        rw [show {x | startFails (fun z => familyRead O.mq F B.lo (B.hi + 1) z p.1)
          o.result.hypothesis.tree o.result.hypothesis.edges x = true}
          = Set.univ from Set.eq_univ_of_forall fun x => by
            simpa using hall x, probReal_univ]
      linarith
    obtain ⟨s, hs, hsub⟩ := check_dfa_of_reads M O hL hF1 (hband _ B hBs).1 p.1 o.result.hypothesis
      (by rw [hres]; exact rootOne_check _ C Kk _) hx₀
    refine ⟨s, hs, ?_⟩
    have hcut := (hus.2 B hBs hBret)
    have hfam : familyAt O.mq pops (xr p) B = insert 1 F := rfl
    rw [hfam] at hcut
    calc Pc.D.real {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)}
        ≤ Pc.D.real {x | startFails (fun z => familyRead O.mq F B.lo (B.hi + 1) z p.1)
            o.result.hypothesis.tree o.result.hypothesis.edges x = true}
          + Pc.D.real ({p' | ¬ cutCorrect O B.lo (B.hi + 1) (insert 1 F) p' p.1}
            ∪ {p' | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p' p.1}) :=
          (measureReal_mono hsub (measure_ne_top _ _)).trans (measureReal_union_le _ _)
      _ ≤ (Kk.rate + NK * prefixMax Pc.D a)
          + Nw * (513 / 512 * settings.εcov + 2 * settings.indecisionLimit + m₀) := by
        refine add_le_add ?_ ?_
        · refine (real_le_avoidLaw Pc.D a Kc _).trans (add_le_add hrate ?_)
          refine (real_compl_avoiding_le Pc.D a Kc).trans ?_
          have hKc : (Kc.card : ℝ) ≤ NK := by exact_mod_cast hgood.2.2.2.2.2
          exact mul_le_mul_of_nonneg_right hKc (measureReal_nonneg.trans (le_prefixMax Pc.D a 1))
        · refine (hDc _).trans (mul_le_mul_of_nonneg_left ?_ hNw)
          have h1 := le_offLaw_real hpm' ({p' | ¬ cutCorrect O B.lo (B.hi + 1) (insert 1 F) p' p.1}
            ∪ {p' | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p' p.1})
          rw [← hDn] at h1
          have h2 := measureReal_union_le (μ := Dn)
            {p' | ¬ cutCorrect O B.lo (B.hi + 1) (insert 1 F) p' p.1}
            {p' | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p' p.1}
          have hc1 : Dn.real {p' | ¬ cutCorrect O B.lo (B.hi + 1) (insert 1 F) p' p.1}
              ≤ 513 / 512 * settings.εcov := hcut.1
          have hc2 : Dn.real {p' | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 F) p' p.1}
              ≤ 2 * settings.indecisionLimit := hcut.2
          linarith
      _ = _ := by ring
  calc μL.real {p | ¬ ∃ o, indecisiveLStarCheckLoop settings O P.D Du Dsf Pc.D h₀ p = [o]
        ∧ o.result.ending = .consistent
        ∧ ∃ s, o.result.hypothesis.tree.sift o.read 1 = .inl s
          ∧ Pc.D.real {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)}
            ≤ Kk.rate + NK * prefixMax Pc.D a
              + Nw * (513 / 512 * settings.εcov + 2 * settings.indecisionLimit + m₀)}
      ≤ μL.real (E₁ ∪ E₂ ∪ E₃ ∪ E₄ ∪ E₅) := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ μL.real E₁ + μL.real E₂ + μL.real E₃ + μL.real E₄ + μL.real E₅ := by
        refine (measureReal_union_le _ _).trans (add_le_add ?_ le_rfl)
        refine (measureReal_union_le _ _).trans (add_le_add ?_ le_rfl)
        refine (measureReal_union_le _ _).trans (add_le_add ?_ le_rfl)
        exact measureReal_union_le _ _
    _ ≤ e₁ + e₂ + 0 + bound + Kk.δ / 8 := by
        gcongr
        exact le_of_eq hE₃
    _ = _ := by ring

end Core

end OrthoDFA
