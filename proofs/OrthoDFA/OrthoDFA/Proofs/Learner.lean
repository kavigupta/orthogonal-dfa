import OrthoDFA.Check
import OrthoDFA.Proofs.Termination
import OrthoDFA.Proofs.LearnerProbes

/-!
# The learner is correct: the proof

`Termination` bounds the chance no round returns: a run on which no round returns breaks some
round's spec (`exists_bad_round`).  `ReturnAccuracy`'s cell bounds the chance a returning round
has an inaccurate hypothesis.  Each round's bound conditions on the earlier rounds' draws and on
the noise at every string read so far (`readsUpTo`); on such a cell the populations, and so the
clustering's laws, are fixed.

* A yield test keeping a harvest of yield below `π₀`, or dropping one of yield `π₁`, is charged
  once per round, and off it every kept harvest has yield at least `π₀`.
* The clustering's spec is `cluster_cell_le` against `clustering_quality_bad`, which needs the
  harvests' yields.
* Past the gate, the hypothesis mislabels what the family cuts well only where it disagrees with
  the cut, which the gate rarely lets through (Hoeffding on its draws), or where the cut itself
  errs, which is rare on a cell (Markov).
* The gate refuses a family cutting little badly only if the stage neither agrees with it nor
  harvests enough, its premise, or the gate's draws mislead it (Hoeffding), or the yield test
  drops a harvest of yield `π₁`.
* The check's spec, and its power at return, are `CheckGuarantee`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S] {R : Type*} [Fintype R]

namespace LearnerProof

/-! ## The populations' laws -/

omit [Fintype R] in
lemma reaching_compl_eq_zero (H : DFA S R) (Dsamp : Measure S) {Pre : Set S}
    (h0 : Dsamp Preᶜ = 0) (h : R) : reaching H Dsamp h Preᶜ = 0 := by
  unfold reaching
  split_ifs
  · exact h0
  · rw [cond_apply (measurableSet_of_countable _)]
    rw [measure_mono_null Set.inter_subset_right h0, mul_zero]

/-- What a harvest reads, how much, how spread its reads and keeps are over the sampler, and
that it keeps prefixes it did not read. -/
def HvOK (Dsamp : Measure S) (Pre : Set S) (Th : ℕ) (κh κa : ℝ) (hv : Harvester S)
    (rd : S → (S → ℝ) → Finset S) : Prop :=
  HarvReads hv rd ∧ (∀ p f, (rd p f).card ≤ Th) ∧ (∀ f t, Dsamp.real {p | t ∈ rd p f} ≤ κh)
    ∧ (∀ f a, Dsamp.real {p | hv p f = some a} ≤ κa)
    ∧ ∀ p f a, hv p f = some a → a ∈ Pre ∧ a ∉ rd p f

/-- Every harvest kept in `past` keeps at least `π₀` of the sampler. -/
def GoodYield (O : Oracle μ S) (Dsamp : Measure S) (π₀ : ℝ) (past : List (Outcome S R)) : Prop :=
  ∀ o ∈ past, ∀ hv, o.2.2.2 = some hv → π₀ ≤ harvestYield O hv Dsamp

lemma popMeasure_compl_eq_zero {K : ℕ} (O : Oracle μ S) (Dsamp : Measure S)
    (past : List (Outcome S R)) {Pre : Set S} (h0 : Dsamp Preᶜ = 0)
    (hPre : ∀ o ∈ past, ∀ hv, o.2.2.2 = some hv → ∀ p f a, hv p f = some a → a ∈ Pre)
    (j : Pop K R) : popMeasure O Dsamp past j Preᶜ = 0 := by
  rcases j with _ | ⟨i, _ | h⟩
  · exact h0
  · simp only [popMeasure]
    rcases hget : past[i.val]? with _ | ⟨H, g, fl, _ | hv⟩
    · exact h0
    · exact h0
    · simp only
      split_ifs
      · exact harvestLaw_compl O hv Dsamp (hPre _ (List.mem_of_getElem? hget) hv rfl)
      · exact h0
  · simp only [popMeasure]
    cases past[i.val]? with
    | none => exact h0
    | some o => exact reaching_compl_eq_zero o.1 Dsamp h0 h

lemma popMeasure_atom_le {K : ℕ} (O : Oracle μ S) (Dsamp : Measure S)
    [IsProbabilityMeasure Dsamp] (past : List (Outcome S R)) {ε κ κa π₀ : ℝ} (hε : 0 < ε)
    (hε1 : ε ≤ 1) (hR : (1 : ℝ) ≤ Fintype.card R) (hκ : ∀ a, Dsamp.real {a} ≤ κ) (hπ₀ : 0 < π₀)
    (hκa : κa ≤ π₀ * (κ * Fintype.card R / ε))
    (hharv : ∀ o ∈ past, ∀ hv, o.2.2.2 = some hv → π₀ ≤ harvestYield O hv Dsamp
      ∧ (∃ rd, HarvReads hv rd) ∧ ∀ f a, Dsamp.real {p | hv p f = some a} ≤ κa) :
    ∀ j ∈ populationsAfter (K := K) Dsamp ε past, ∀ a,
      (popMeasure O Dsamp past j).real {a} ≤ κ * Fintype.card R / ε := by
  classical
  intro j hj a
  have hκ0 : 0 ≤ κ := le_trans measureReal_nonneg (hκ a)
  have hR0 : (0 : ℝ) < Fintype.card R := by linarith
  have hsamp : Dsamp.real {a} ≤ κ * Fintype.card R / ε :=
    calc Dsamp.real {a} ≤ κ := hκ a
      _ = κ * 1 := (mul_one κ).symm
      _ ≤ κ * (Fintype.card R / ε) := by
          gcongr
          rw [le_div_iff₀ hε]; linarith
      _ = κ * Fintype.card R / ε := by ring
  rcases j with _ | ⟨i, _ | h⟩
  · exact hsamp
  · simp only [popMeasure]
    rcases hget : past[i.val]? with _ | ⟨H, g, fl, _ | hv⟩
    · exact hsamp
    · exact hsamp
    · simp only
      split_ifs with hy
      · obtain ⟨hπ, ⟨rd, hrd⟩, hka⟩ := hharv _ (List.mem_of_getElem? hget) hv rfl
        refine (harvestLaw_atom_le O hv hrd Dsamp hka hy a).trans ?_
        calc κa / harvestYield O hv Dsamp ≤ κa / π₀ := by
              have : 0 ≤ κa := le_trans measureReal_nonneg (hka (fun _ => 0) a)
              exact div_le_div_of_nonneg_left this hπ₀ hπ
          _ ≤ κ * Fintype.card R / ε := by rw [div_le_iff₀ hπ₀]; linarith
      · exact hsamp
  · obtain ⟨o, ho, hm⟩ := (Finset.mem_filter.1 hj).2
    have ho' : past[i.val]? = some o := by simpa using ho
    have hpm : popMeasure O Dsamp past (some (i, some h)) = reaching o.1 Dsamp h := by
      simp [popMeasure, ho']
    rw [hpm]
    set m := Dsamp.real {v | o.1.state v = h}
    have hmpos : 0 < m := lt_of_lt_of_le (div_pos hε hR0) hm
    have h0 : Dsamp {v | o.1.state v = h} ≠ 0 := fun h0 => by
      simp [m, measureReal_def, h0] at hmpos
    have h1 := reaching_real_apply o.1 Dsamp h0 {a}
    have h2 : Dsamp.real ({v | o.1.state v = h} ∩ {a}) ≤ κ :=
      (measureReal_mono Set.inter_subset_right).trans (hκ a)
    have h3 : (reaching o.1 Dsamp h).real {a} ≤ κ / m := by
      rw [le_div_iff₀ hmpos]; linarith
    calc (reaching o.1 Dsamp h).real {a} ≤ κ / m := h3
      _ ≤ κ / (ε / Fintype.card R) := div_le_div_of_nonneg_left hκ0 (div_pos hε hR0) hm
      _ = κ * Fintype.card R / ε := by field_simp

/-! ## The learner's space -/

section Rounds

variable {K : ℕ} {Xs : Type*} [MeasurableSpace Xs]

noncomputable def checkLaw (Dsamp Dsf : Measure S) (N m : ℕ) :
    Measure ((Fin N → S) × (Fin m → S)) :=
  (Measure.pi fun _ : Fin N => Dsamp).prod (Measure.pi fun _ : Fin m => Dsf)

/-- The gate's draws, the yield test's and the check's. -/
noncomputable def tailLaw (Dsamp Dsf : Measure S) (G Y N m : ℕ) :
    Measure ((Fin G → S) × (Fin Y → S) × ((Fin N → S) × (Fin m → S))) :=
  (Measure.pi fun _ : Fin G => Dsamp).prod
    ((Measure.pi fun _ : Fin Y => Dsamp).prod (checkLaw Dsamp Dsf N m))

noncomputable def roundLaw (Dsamp Dsf : Measure S) (νs : Measure Xs) (L : Learner Ω S R Xs K) :
    Measure (RoundDraws S K R Xs L.M L.G L.Y L.N L.m) :=
  (Measure.pi fun _ : Fin L.M => Dsf).prod
    (((Measure.pi fun _ : Pop K R => Measure.pi fun _ : Fin L.M => Dsamp).prod
        (Measure.pi fun _ : Pop K R => Measure.pi fun _ : Fin L.M => Dsamp)).prod
      (νs.prod (tailLaw Dsamp Dsf L.G L.Y L.N L.m)))

noncomputable def streamLaw (Dsamp : Measure S) : Measure (R → ℕ → S) :=
  Measure.pi fun _ : R => Measure.infinitePi fun _ : ℕ => Dsamp

variable (Dsamp Dsf : Measure S) [IsProbabilityMeasure Dsamp] [IsProbabilityMeasure Dsf]
  (νs : Measure Xs) [IsProbabilityMeasure νs] (L : Learner Ω S R Xs K)

instance isProbabilityMeasure_checkLaw (N m : ℕ) :
    IsProbabilityMeasure (checkLaw Dsamp Dsf N m) := by
  unfold checkLaw; infer_instance

instance isProbabilityMeasure_tailLaw (G Y N m : ℕ) :
    IsProbabilityMeasure (tailLaw Dsamp Dsf G Y N m) := by
  unfold tailLaw; infer_instance

instance isProbabilityMeasure_roundLaw : IsProbabilityMeasure (roundLaw Dsamp Dsf νs L) := by
  unfold roundLaw; infer_instance

instance isProbabilityMeasure_streamLaw : IsProbabilityMeasure (streamLaw (R := R) Dsamp) := by
  unfold streamLaw; infer_instance

omit [IsProbabilityMeasure μ] [IsProbabilityMeasure Dsamp] [IsProbabilityMeasure Dsf]
  [IsProbabilityMeasure νs] in
lemma learnerMeasure_eq : learnerMeasure (μ := μ) Dsamp Dsf νs L
    = μ.prod ((Measure.pi fun _ : Fin K => roundLaw Dsamp Dsf νs L).prod
      (Measure.pi fun _ : Fin K => streamLaw (R := R) Dsamp)) := rfl

omit [MeasurableSpace Ω] in
lemma measurePreserving_clusterPart :
    MeasurePreserving
      (fun d : RoundDraws S K R Xs L.M L.G L.Y L.N L.m => ((d.1, d.2.1) : ClusterPart S K R L.M))
      (roundLaw Dsamp Dsf νs L) (clusterLaw Dsamp Dsf L.M) :=
  (MeasurePreserving.id _).prod measurePreserving_fst

omit [MeasurableSpace Ω] [IsProbabilityMeasure Dsamp] [IsProbabilityMeasure Dsf] in
/-- The clustering's draws, against the stage's, the gate's, the yield test's and the
check's. -/
lemma measurePreserving_clusterSplit :
    MeasurePreserving (MeasurableEquiv.prodAssoc.symm :
        RoundDraws S K R Xs L.M L.G L.Y L.N L.m → ClusterPart S K R L.M × (Xs × _))
      (roundLaw Dsamp Dsf νs L)
      ((clusterLaw Dsamp Dsf L.M).prod (νs.prod (tailLaw Dsamp Dsf L.G L.Y L.N L.m))) :=
  MeasurePreserving.symm _ (measurePreserving_prodAssoc _ _ _)

omit [MeasurableSpace Ω] [IsProbabilityMeasure Dsamp] [IsProbabilityMeasure Dsf] in
/-- The clustering's and the stage's draws, against the gate's, the yield test's and the
check's. -/
lemma measurePreserving_checkSplit :
    MeasurePreserving (fun d : RoundDraws S K R Xs L.M L.G L.Y L.N L.m =>
        MeasurableEquiv.prodAssoc.symm (MeasurableEquiv.prodAssoc.symm d))
      (roundLaw Dsamp Dsf νs L)
      (((clusterLaw Dsamp Dsf L.M).prod νs).prod (tailLaw Dsamp Dsf L.G L.Y L.N L.m)) :=
  (MeasurePreserving.symm _ (measurePreserving_prodAssoc _ _ _)).comp
    (measurePreserving_clusterSplit Dsamp Dsf νs L)

end Rounds

/-! ## Each round's specs -/

omit [IsProbabilityMeasure μ] in
/-- A bound on the pinned cell's conditional law, as a bound on its restriction. -/
lemma restrict_prod_le (O : Oracle μ S) (U : Finset S) (b : S → ℝ) {Y : Type*}
    [MeasurableSpace Y] (ν : Measure Y) [IsProbabilityMeasure ν] [IsFiniteMeasure μ]
    (G : Set (Ω × Y)) {β : ℝ} (hβ : ((μ[|pinned O U b]).prod ν).real G ≤ β) :
    ((μ.restrict (pinned O U b)).prod ν) G ≤ μ (pinned O U b) * ENNReal.ofReal β := by
  set P := pinned O U b
  by_cases hP0 : μ P = 0
  · rw [Measure.restrict_eq_zero.2 hP0, Measure.zero_prod, Measure.coe_zero, Pi.zero_apply]
    exact zero_le
  have hsm : μ.restrict P = μ P • μ[|P] := by
    rw [ProbabilityTheory.cond, smul_smul, ENNReal.mul_inv_cancel hP0 (measure_ne_top _ _),
      one_smul]
  have := cond_isProbabilityMeasure (μ := μ) hP0
  rw [hsm, Measure.prod_smul_left, Measure.smul_apply, smul_eq_mul]
  gcongr
  exact (ENNReal.le_ofReal_iff_toReal_le (measure_ne_top _ _)
    (le_trans measureReal_nonneg hβ)).2 hβ

lemma add_share_le {r K per x : ℕ} (hr : r < K) (hx : x ≤ per) : r * per + x ≤ K * per :=
  calc r * per + x ≤ r * per + per := Nat.add_le_add_left hx _
    _ = (r + 1) * per := (Nat.succ_mul r per).symm
    _ ≤ K * per := Nat.mul_le_mul_right _ hr

/-- The draws of the rounds before `r`, with `x0` after them. -/
noncomputable def extendDraws {K : ℕ} {X : Type*} (x0 : X) (r : Fin K)
    (a : {i : Fin K // i < r} → X) : Fin K → X :=
  fun i => if h : i < r then a ⟨i, h⟩ else x0

/-! ## The gate's draws, and the cut's errors on a cell -/

section Gate

variable {Q : Type*}

omit [Fintype R] in
lemma measurableSet_cutAgrees (O : Oracle μ S) (B : State) (F : Finset S) (H : DFA S R) (p : S) :
    MeasurableSet {ω | cutAgrees O B F H p ω} :=
  measurable_voteCount O F p (MeasurableSet.of_discrete (s := {n : ℕ |
    (B.hi < n ∧ H.state p ∈ H.accept) ∨ (n ≤ B.lo ∧ H.state p ∉ H.accept)}))

omit [Fintype R] in
lemma measurable_cutDisagreement (O : Oracle μ S) (B : State) (F : Finset S) (H : DFA S R)
    (Dsamp : Measure S) [IsFiniteMeasure Dsamp] :
    Measurable fun ω => cutDisagreement O B F H Dsamp ω := by
  have hE : MeasurableSet {q : Ω × S | ¬ cutAgrees O B F H q.2 q.1} := by
    have : {q : Ω × S | ¬ cutAgrees O B F H q.2 q.1}
        = ⋃ p, {ω | ¬ cutAgrees O B F H p ω} ×ˢ {p} := by
      ext ⟨ω, p⟩; simp
    rw [this]
    exact MeasurableSet.iUnion fun p =>
      (measurableSet_cutAgrees O B F H p).compl.prod (measurableSet_singleton p)
  exact (measurable_measure_prodMk_left hE).ennreal_toReal

/-- Over the gate's draws, given the noise: how many of them land in `W ω`. -/
lemma gate_prod_le (μ' : Measure Ω) [SFinite μ'] (Dsamp : Measure S) [IsProbabilityMeasure Dsamp]
    (G : ℕ) (W : Ω → Set S) [∀ ω, DecidablePred (· ∈ W ω)]
    (hW : ∀ p, MeasurableSet {ω | p ∈ W ω}) (E : Set Ω) (hE : MeasurableSet E) (cnt : ℕ → Prop)
    {β : ℝ}
    (hb : ∀ ω ∈ E, (Measure.pi fun _ : Fin G => Dsamp).real
      {g | cnt (Finset.univ.filter fun i => g i ∈ W ω).card} ≤ β) :
    (μ'.prod (Measure.pi fun _ : Fin G => Dsamp))
        {q | q.1 ∈ E ∧ cnt (Finset.univ.filter fun i => q.2 i ∈ W q.1).card}
      ≤ μ' E * ENNReal.ofReal β := by
  classical
  set ν := Measure.pi fun _ : Fin G => Dsamp
  set Z := {q : Ω × (Fin G → S) | q.1 ∈ E ∧ cnt (Finset.univ.filter fun i => q.2 i ∈ W q.1).card}
  have hcnt : ∀ g : Fin G → S,
      Measurable fun ω => (Finset.univ.filter fun i => g i ∈ W ω).card := by
    intro g
    have : (fun ω => (Finset.univ.filter fun i => g i ∈ W ω).card)
        = fun ω => ∑ i, (if g i ∈ W ω then 1 else 0) := by
      funext ω; rw [Finset.card_filter]
    rw [this]
    exact Finset.measurable_sum _ fun i _ =>
      Measurable.ite (hW (g i)) measurable_const measurable_const
  have hZ : MeasurableSet Z := by
    have : Z = ⋃ g, (E ∩ {ω | cnt (Finset.univ.filter fun i => g i ∈ W ω).card}) ×ˢ {g} := by
      ext ⟨ω, g⟩
      simp only [Z, Set.mem_iUnion, Set.mem_prod, Set.mem_inter_iff, Set.mem_ofPred_eq,
        Set.mem_singleton_iff]
      exact ⟨fun h => ⟨g, h, rfl⟩, fun ⟨g', h, hg⟩ => hg ▸ h⟩
    rw [this]
    exact MeasurableSet.iUnion fun g =>
      (hE.inter (hcnt g MeasurableSet.of_discrete)).prod (measurableSet_singleton g)
  rw [Measure.prod_apply hZ]
  calc ∫⁻ ω, ν (Prod.mk ω ⁻¹' Z) ∂μ'
      ≤ ∫⁻ ω, E.indicator (fun _ => ENNReal.ofReal β) ω ∂μ' := by
        refine lintegral_mono fun ω => ?_
        by_cases hω : ω ∈ E
        · rw [Set.indicator_of_mem hω]
          have : Prod.mk ω ⁻¹' Z = {g | cnt (Finset.univ.filter fun i => g i ∈ W ω).card} := by
            ext g; simp [Z, hω]
          rw [this, ← ofReal_measureReal]
          exact ENNReal.ofReal_le_ofReal (hb ω hω)
        · rw [Set.indicator_of_notMem hω]
          have : Prod.mk ω ⁻¹' Z = ∅ := by ext g; simp [Z, hω]
          rw [this, measure_empty]
    _ = μ' E * ENNReal.ofReal β := by rw [lintegral_indicator_const hE, mul_comm]

/-- The gate lets through a hypothesis disagreeing with the cut on more than `ζ₂`. -/
lemma pi_pass_le (Dsamp : Measure S) [IsProbabilityMeasure Dsamp] (G gth : ℕ) (W : Set S)
    [DecidablePred (· ∈ W)]
    {ζ₂ : ℝ} (hg : (gth : ℝ) ≤ ζ₂ * G) (hW : ζ₂ < Dsamp.real W) :
    (Measure.pi fun _ : Fin G => Dsamp).real
        {g | (Finset.univ.filter fun i => g i ∈ W).card ≤ gth}
      ≤ Real.exp (-2 * G * (ζ₂ - gth / G) ^ 2) := by
  classical
  rcases Nat.eq_zero_or_pos G with rfl | hG
  · rw [show -2 * ((0 : ℕ) : ℝ) * (ζ₂ - gth / ((0 : ℕ) : ℝ)) ^ 2 = 0 by simp, Real.exp_zero]
    exact measureReal_le_one
  have hG' : (0 : ℝ) < G := by exact_mod_cast hG
  have hgG : (gth : ℝ) / G ≤ ζ₂ := by rw [div_le_iff₀ hG']; linarith
  set d := Dsamp.real W
  have ht : 0 ≤ d - gth / G := by linarith
  refine (measureReal_mono (fun g hg' => ?_) (measure_ne_top _ _)).trans
    ((pi_hits_lower Dsamp G W d (d - gth / G) measureReal_nonneg ht le_rfl).trans ?_)
  · simp only [Set.mem_ofPred_eq] at hg' ⊢
    rw [sub_sub_cancel, mul_div_cancel₀ _ hG'.ne']
    norm_cast
    convert hg'
  · refine Real.exp_le_exp.2 ?_
    have h1 : 0 ≤ ζ₂ - gth / G := by linarith
    have h2 : (ζ₂ - gth / G) ^ 2 ≤ (d - gth / G) ^ 2 := by nlinarith
    nlinarith

/-- The gate refuses a hypothesis disagreeing with the cut on at most `ζ₀`. -/
lemma pi_refuse_le (Dsamp : Measure S) [IsProbabilityMeasure Dsamp] (G gth : ℕ) (W : Set S)
    [DecidablePred (· ∈ W)]
    {ζ₀ : ℝ} (hg : ζ₀ * G ≤ gth) (hW : Dsamp.real W ≤ ζ₀) :
    (Measure.pi fun _ : Fin G => Dsamp).real
        {g | gth < (Finset.univ.filter fun i => g i ∈ W).card}
      ≤ Real.exp (-2 * G * (gth / G - ζ₀) ^ 2) := by
  classical
  rcases Nat.eq_zero_or_pos G with rfl | hG
  · rw [show {g : Fin 0 → S | gth < (Finset.univ.filter fun i => g i ∈ W).card} = ∅ from
      Set.eq_empty_of_forall_notMem fun g hg => by simp at hg, measureReal_empty]
    exact (Real.exp_pos _).le
  have hG' : (0 : ℝ) < G := by exact_mod_cast hG
  have hgG : ζ₀ ≤ (gth : ℝ) / G := by rw [le_div_iff₀ hG']; linarith
  set d := Dsamp.real W
  have hd1 : d ≤ 1 := measureReal_le_one
  have hWc : 1 - d ≤ Dsamp.real Wᶜ := by
    rw [measureReal_compl MeasurableSet.of_discrete, probReal_univ]
  have ht : 0 ≤ (gth : ℝ) / G - d := by linarith
  refine (measureReal_mono (fun g hg' => ?_) (measure_ne_top _ _)).trans
    ((pi_hits_lower Dsamp G Wᶜ (1 - d) (gth / G - d) (by linarith) ht hWc).trans ?_)
  · simp only [Set.mem_ofPred_eq] at hg' ⊢
    have hsum := Finset.card_filter_add_card_filter_not (s := (Finset.univ : Finset (Fin G)))
      (fun i => g i ∈ W)
    rw [Finset.card_univ, Fintype.card_fin] at hsum
    have hc : ((Finset.univ.filter fun i => ¬ g i ∈ W).card : ℝ)
        = G - (Finset.univ.filter fun i => g i ∈ W).card := by
      rw [eq_sub_iff_add_eq, add_comm]; exact_mod_cast hsum
    convert_to ((Finset.univ.filter fun i => ¬ g i ∈ W).card : ℝ) ≤ _
    · simp
    rw [hc]
    have hlt : (gth : ℝ) + 1 ≤ (Finset.univ.filter fun i => g i ∈ W).card := by
      exact_mod_cast hg'
    have : (G : ℝ) * (1 - d - (gth / G - d)) = G - gth := by field_simp; ring
    rw [this]
    linarith
  · refine Real.exp_le_exp.2 ?_
    have h1 : 0 ≤ (gth : ℝ) / G - ζ₀ := by linarith
    have h2 : ((gth : ℝ) / G - ζ₀) ^ 2 ≤ (gth / G - d) ^ 2 := by nlinarith
    nlinarith

omit [IsProbabilityMeasure μ] [Fintype R] in
/-- A hypothesis mislabels a string the family cuts well only where it disagrees with the cut or
the cut errs. -/
lemma mislabelled_le (A : DFA S Q) (O : Oracle μ S) (hL : O.L = {w | A.state w ∈ A.accept})
    (tolerance : ℝ) (B : State) (F : Finset S) (H : DFA S R) (Dsamp : Measure S)
    [IsFiniteMeasure Dsamp] (ω : Ω) :
    mislabelledWellCut A O tolerance B F H Dsamp
      ≤ cutDisagreement O B F H Dsamp ω
        + Dsamp.real {p | miscutProb O B.lo B.hi F p ≤ tolerance
          ∧ ¬ cutCorrect O B.lo B.hi F p ω} := by
  classical
  refine (measureReal_mono (fun p hp => ?_) (measure_ne_top _ _)).trans (measureReal_union_le _ _)
  obtain ⟨hmis, hmc, -⟩ := hp
  by_cases hc : cutAgrees O B F H p ω
  · right
    refine ⟨hmc, fun hcc => ?_⟩
    rcases hc with ⟨hv, hH⟩ | ⟨hv, hH⟩
    · have h1 := hcc.1 hv
      by_cases hp : p ∈ O.L
      · rw [hL] at hp
        exact (hmis.1 hp) hH
      · simp [Oracle.label, hp] at h1
    · have h1 := hcc.2 hv
      by_cases hp : p ∈ O.L
      · simp [Oracle.label, hp] at h1
      · rw [hL] at hp
        exact hp (hmis.2 hH)
  · exact Or.inl hc

/-- Markov on a cell: off the pinned strings the cut errs where it cuts well no more often than
under fresh noise. -/
lemma cut_err_le (O : Oracle μ S) (Dsamp : Measure S) [IsProbabilityMeasure Dsamp]
    (U : Finset S) (b : S → ℝ) (lo hi : ℕ) (F : Finset S) {τ κ ζ₁ : ℝ} (hζ₁ : 0 < ζ₁)
    (hτ : 0 ≤ τ) (hκ : ∀ a, Dsamp.real {a} ≤ κ) :
    μ ({ω | ζ₁ < Dsamp.real {p | miscutProb O lo hi F p ≤ τ ∧ ¬ cutCorrect O lo hi F p ω}}
        ∩ pinned O U b)
      ≤ μ (pinned O U b) * ENNReal.ofReal ((τ + U.card * F.card * κ) / ζ₁) := by
  classical
  have hκ0 : 0 ≤ κ := le_trans measureReal_nonneg (hκ 1)
  set G : Set (Ω × S) := {q | miscutProb O lo hi F q.2 ≤ τ ∧ ¬ cutCorrect O lo hi F q.2 q.1}
  set RV : S → Finset S := fun p => F.image (p * ·)
  have hsec : ∀ p, (fun ω => (ω, p)) ⁻¹' G
      = if miscutProb O lo hi F p ≤ τ then {ω | ¬ cutCorrect O lo hi F p ω} else ∅ := by
    intro p
    split_ifs with hp
    · ext ω; simp [G, hp]
    · ext ω; simp [G, hp]
  have hGv : ∀ p, MeasurableSet[noiseAlg O ↑(RV p)] ((fun ω => (ω, p)) ⁻¹' G) := by
    intro p
    rw [hsec p]
    split_ifs
    · exact measurableSet_filter_pred_map O (T := ↑(RV p)) (A := F) (fun v => p * v)
        (fun v hv => Finset.mem_coe.2 (Finset.mem_image_of_mem _ hv))
        (fun W => ¬ ((hi < W.card → O.label p = 1) ∧ (W.card ≤ lo → O.label p = 0)))
    · exact @MeasurableSet.empty _ (noiseAlg O ↑(RV p))
  have hG : MeasurableSet G := by
    have : G = ⋃ p, ((fun ω => (ω, p)) ⁻¹' G) ×ˢ {p} := by
      ext ⟨ω, p⟩
      simp only [Set.mem_iUnion, Set.mem_prod, Set.mem_preimage, Set.mem_singleton_iff]
      exact ⟨fun h => ⟨p, h, rfl⟩, fun ⟨p', h, hp⟩ => hp ▸ h⟩
    rw [this]
    exact MeasurableSet.iUnion fun p => (noiseAlg_le O _ _ (hGv p)).prod (measurableSet_singleton p)
  have hpin := pinned_prod_le O U b Dsamp RV hG hGv
  have hfresh : (μ.prod Dsamp) G ≤ ENNReal.ofReal τ := by
    rw [Measure.prod_apply_symm hG]
    calc ∫⁻ p, μ ((fun ω => (ω, p)) ⁻¹' G) ∂Dsamp ≤ ∫⁻ _p, ENNReal.ofReal τ ∂Dsamp := by
          refine lintegral_mono fun p => ?_
          rw [hsec p]
          split_ifs with hp
          · rw [← ofReal_measureReal]
            exact ENNReal.ofReal_le_ofReal hp
          · rw [measure_empty]; exact zero_le
      _ = ENNReal.ofReal τ := by rw [lintegral_const, measure_univ, mul_one]
  have hhit : Dsamp {p | ¬ Disjoint U (RV p)} ≤ ENNReal.ofReal (U.card * F.card * κ) := by
    calc Dsamp {p | ¬ Disjoint U (RV p)} ≤ Dsamp (⋃ v ∈ F, {p | p * v ∈ U}) :=
          measure_mono fun p hp => by
            obtain ⟨w, hwU, hwR⟩ := Finset.not_disjoint_iff.1 hp
            obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hwR
            exact Set.mem_biUnion hv hwU
      _ ≤ ∑ v ∈ F, Dsamp {p | p * v ∈ U} := measure_biUnion_finset_le _ _
      _ ≤ ∑ _v ∈ F, ENNReal.ofReal (U.card * κ) :=
          Finset.sum_le_sum fun v _ => measure_mul_mem_le Dsamp hκ U v
      _ = ENNReal.ofReal (U.card * F.card * κ) := by
          rw [Finset.sum_const, nsmul_eq_mul, ← ENNReal.ofReal_natCast,
            ← ENNReal.ofReal_mul (Nat.cast_nonneg _)]
          congr 1; ring
  set f : Ω → ℝ≥0∞ := fun ω => Dsamp (Prod.mk ω ⁻¹' G)
  have hf : Measurable f := measurable_measure_prodMk_left hG
  rw [← Measure.restrict_apply' (measurableSet_pinned O U b)]
  calc (μ.restrict (pinned O U b))
        {ω | ζ₁ < Dsamp.real {p | miscutProb O lo hi F p ≤ τ ∧ ¬ cutCorrect O lo hi F p ω}}
      ≤ (μ.restrict (pinned O U b)) {ω | ENNReal.ofReal ζ₁ ≤ f ω} := measure_mono fun ω hω => by
        have : ζ₁ < (f ω).toReal := hω
        exact (ENNReal.ofReal_le_ofReal this.le).trans
          (ENNReal.ofReal_toReal (measure_ne_top _ _)).le
    _ ≤ (∫⁻ ω, f ω ∂(μ.restrict (pinned O U b))) / ENNReal.ofReal ζ₁ :=
        meas_ge_le_lintegral_div hf.aemeasurable (ENNReal.ofReal_pos.2 hζ₁).ne'
          ENNReal.ofReal_ne_top
    _ = ((μ.restrict (pinned O U b)).prod Dsamp) G / ENNReal.ofReal ζ₁ := by
        rw [Measure.prod_apply hG]
    _ ≤ μ (pinned O U b) * (ENNReal.ofReal τ + ENNReal.ofReal (U.card * F.card * κ))
          / ENNReal.ofReal ζ₁ := by
        gcongr
        exact hpin.trans (by gcongr)
    _ = μ (pinned O U b) * ENNReal.ofReal ((τ + U.card * F.card * κ) / ζ₁) := by
        rw [ENNReal.ofReal_div_of_pos hζ₁, ← ENNReal.ofReal_add hτ (by positivity), mul_div_assoc]

end Gate


section Specs

variable {Q : Type*} {K : ℕ} {Xs : Type*} [MeasurableSpace Xs] [Countable Xs]
  [MeasurableSingletonClass Xs]
  (A : DFA S Q) (O : Oracle μ S) (Dsamp Dsf : Measure S) [IsProbabilityMeasure Dsamp]
  [IsProbabilityMeasure Dsf] (νs : Measure Xs) [IsProbabilityMeasure νs] (L : Learner Ω S R Xs K)
  (schedule : Finset (Pop K R) → Finset State) (indecisionLimit α : ℝ)
  (stageReads : Finset S → State → ClusterDraws S K R → Ω → Xs → Finset S)
  (harvestReads : Finset S → State → ClusterDraws S K R → Ω → Xs → S → (S → ℝ) → Finset S)

omit [IsProbabilityMeasure μ] [Fintype R] [MeasurableSpace Xs] [Countable Xs]
  [MeasurableSingletonClass Xs] [IsProbabilityMeasure Dsamp] [IsProbabilityMeasure Dsf] in
/-- A harvest the stage returns has the premises' properties, with the reads of one such call. -/
lemma hvOK_of_harvIn {Pre : Set S} {Th : ℕ} {κh κa : ℝ}
    (hhv : ∀ F B c ω s p (f f' : S → ℝ), (∀ t ∈ harvestReads F B c ω s p f, f t = f' t) →
      harvestReads F B c ω s p f' = harvestReads F B c ω s p f
        ∧ L.harvest F B c ω s p f' = L.harvest F B c ω s p f)
    (hTh : ∀ F B c ω s p f, (harvestReads F B c ω s p f).card ≤ Th)
    (hκh : ∀ F B c ω s f t, Dsamp.real {p | t ∈ harvestReads F B c ω s p f} ≤ κh)
    (hκa : ∀ F B c ω s f a, Dsamp.real {p | L.harvest F B c ω s p f = some a} ≤ κa)
    (hout : ∀ F B c ω s p f a, L.harvest F B c ω s p f = some a →
      a ∈ Pre ∧ a ∉ harvestReads F B c ω s p f)
    {hv : Harvester S} (hin : HarvIn L hv) :
    HvOK Dsamp Pre Th κh κa hv (hvReads L harvestReads hv) := by
  classical
  have hspec := hin.choose_spec
  refine ⟨fun p f f' h => hvReads_congr L harvestReads hhv hin p h,
    fun p f => card_hvReads_le L harvestReads hTh hv p f, fun f t => ?_, fun f a => ?_,
    fun p f a h => ?_⟩
  · simp only [hvReads, dif_pos hin]
    exact hκh _ _ _ _ _ f t
  · rw [← hspec]
    exact hκa _ _ _ _ _ f a
  · simp only [hvReads, dif_pos hin]
    rw [← hspec] at h
    exact hout _ _ _ _ _ p f a h

/-- The clustering's spec at round `r`, while every harvest kept keeps at least `π₀`. -/
theorem cluster_round_le (tolerance εcov : ℝ) {Pre : Set S} {P Ts Th T : ℕ}
    {κ κh κa κf π₀ δc : ℝ}
    (hreads : ∀ F B c s, ReadsOnly O (fun ω => stageReads F B c ω s)
      (fun ω => (L.stage F B c ω s, L.harvest F B c ω s)))
    (hhv : ∀ F B c ω s p (f f' : S → ℝ), (∀ t ∈ harvestReads F B c ω s p f, f t = f' t) →
      harvestReads F B c ω s p f' = harvestReads F B c ω s p f
        ∧ L.harvest F B c ω s p f' = L.harvest F B c ω s p f)
    (hTs : ∀ F B c ω s, (stageReads F B c ω s).card ≤ Ts)
    (hTh : ∀ F B c ω s p f, (harvestReads F B c ω s p f).card ≤ Th)
    (hTK : K * (2 * Fintype.card (Pop K R) * L.M * (L.M + 1) + Ts
      + 2 * Fintype.card (Pop K R) * L.M * Th + L.G * (L.M + 1) + L.Y * Th
      + L.N * (L.m + 1)) ≤ T)
    (hM : 1 ≤ L.M) (hPM : P ≤ L.M)
    (hsched : ∀ pops, ∀ B ∈ schedule pops, B.npref ≤ P ∧ B.nsuff ≤ L.M)
    (hε : 0 < L.heavy) (hε1 : L.heavy ≤ 1) (hR : (1 : ℝ) ≤ Fintype.card R)
    (hκ : ∀ a, Dsamp.real {a} ≤ κ)
    (hP : (P : ℝ) ≤ L.M * L.heavy / Fintype.card R) (hPπ : (P : ℝ) ≤ L.M * π₀) (hπ₀ : 0 < π₀)
    (hκh : ∀ F B c ω s f t, Dsamp.real {p | t ∈ harvestReads F B c ω s p f} ≤ κh)
    (hκa : ∀ F B c ω s f a, Dsamp.real {p | L.harvest F B c ω s p f = some a} ≤ κa)
    (hout : ∀ F B c ω s p f a, L.harvest F B c ω s p f = some a →
      a ∈ Pre ∧ a ∉ harvestReads F B c ω s p f)
    (hκaπ : κa ≤ π₀ * (κ * Fintype.card R / L.heavy)) (hκf : ∀ a, Dsf.real {a} ≤ κf)
    (hclus : ∀ past : List (Outcome S R), PastIn L past → GoodYield O Dsamp π₀ past →
      (runMeasure μ (popMeasure (K := K) O Dsamp past) Dsf).real {x | ¬ QualityGood A O
        (populationsAfter Dsamp L.heavy past) (popMeasure O Dsamp past)
        (schedule (populationsAfter Dsamp L.heavy past)) indecisionLimit α tolerance εcov x} ≤ δc)
    (r : Fin K) :
    (learnerMeasure (μ := μ) Dsamp Dsf νs L).real
      {θ | GoodYield O Dsamp π₀ (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r)
        ∧ ¬ QualityGood A O
          (populationsAfter Dsamp L.heavy (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r))
          (popMeasure O Dsamp (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r))
          (schedule (populationsAfter Dsamp L.heavy (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r)))
          indecisionLimit α tolerance εcov
          ((θ.1, clusterDraws O θ.1 (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r)
            ((θ.2.1 r).1, (θ.2.1 r).2.1)) : Run Ω S (Pop K R))}
      ≤ δc + 2 * Fintype.card (Pop K R)
          * Real.exp (-2 * (L.M * L.heavy / Fintype.card R - P) ^ 2 / L.M)
        + 2 * Fintype.card (Pop K R) * Real.exp (-2 * (L.M * π₀ - P) ^ 2 / L.M)
        + 2 * Fintype.card (Pop K R) * L.M * (L.M + 1) * T * (κ * Fintype.card R / L.heavy)
        + 2 * Fintype.card (Pop K R) * L.M
          * (T + 2 * Fintype.card (Pop K R) * L.M * (L.M + 1)
            + 2 * Fintype.card (Pop K R) * L.M * Th) * κh
        + 2 * Fintype.card (Pop K R) * L.M * (L.M + 1) * Th * κf := by
  classical
  obtain ⟨ω0⟩ := nonempty_of_isProbabilityMeasure μ
  obtain ⟨s0⟩ := nonempty_of_isProbabilityMeasure νs
  have hκ0 : 0 ≤ κ := le_trans measureReal_nonneg (hκ 1)
  have hκh0 : 0 ≤ κh := le_trans measureReal_nonneg
    (hκh ∅ ⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩ ((fun _ => 1, fun _ _ => 1), fun _ _ => 1)
      ω0 s0 (fun _ => 0) 1)
  have hκf0 : 0 ≤ κf := le_trans measureReal_nonneg (hκf 1)
  have hδc0 : 0 ≤ δc := le_trans measureReal_nonneg
    (hclus [] (fun o ho => by simp at ho) (fun o ho => by simp at ho))
  set δc' := δc + 2 * Fintype.card (Pop K R)
      * Real.exp (-2 * (L.M * L.heavy / Fintype.card R - P) ^ 2 / L.M)
    + 2 * Fintype.card (Pop K R) * Real.exp (-2 * (L.M * π₀ - P) ^ 2 / L.M)
    + 2 * Fintype.card (Pop K R) * L.M * (L.M + 1) * T * (κ * Fintype.card R / L.heavy)
    + 2 * Fintype.card (Pop K R) * L.M
      * (T + 2 * Fintype.card (Pop K R) * L.M * (L.M + 1)
        + 2 * Fintype.card (Pop K R) * L.M * Th) * κh
    + 2 * Fintype.card (Pop K R) * L.M * (L.M + 1) * Th * κf with hδc'
  have hδc'0 : 0 ≤ δc' := by positivity
  set F : List (Outcome S R) →
      Set ((Ω × RoundDraws S K R Xs L.M L.G L.Y L.N L.m) × (R → ℕ → S)) := fun c =>
    {q | PastIn L c ∧ GoodYield O Dsamp π₀ c ∧ ¬ QualityGood A O
      (populationsAfter Dsamp L.heavy c) (popMeasure O Dsamp c)
      (schedule (populationsAfter Dsamp L.heavy c)) indecisionLimit α tolerance εcov
      ((q.1.1, clusterDraws O q.1.1 c (q.1.2.1, q.1.2.2.1)) : Run Ω S (Pop K R))}
  have hF : ∀ c U b, U.card ≤ T →
      (((μ.prod (roundLaw Dsamp Dsf νs L)).prod (streamLaw (R := R) Dsamp))
        (F c ∩ {q | q.1.1 ∈ pinned O U b})) ≤ μ (pinned O U b) * ENNReal.ofReal δc' := by
    intro c U b hU
    by_cases hc : PastIn L c ∧ GoodYield O Dsamp π₀ c
    swap
    · rw [show F c = ∅ from Set.eq_empty_of_forall_notMem fun q hq => hc ⟨hq.1, hq.2.1⟩,
        Set.empty_inter, measure_empty]
      exact zero_le
    obtain ⟨hpast, hgood⟩ := hc
    have hok : ∀ o ∈ c, ∀ hv, o.2.2.2 = some hv →
        HvOK Dsamp Pre Th κh κa hv (hvReads L harvestReads hv) := fun o ho hv hov =>
      hvOK_of_harvIn Dsamp L harvestReads hhv hTh hκh hκa hout (hpast o ho hv hov)
    set E : Set (Ω × ClusterPart S K R L.M) := {q | ¬ QualityGood A O
      (populationsAfter Dsamp L.heavy c) (popMeasure O Dsamp c)
      (schedule (populationsAfter Dsamp L.heavy c)) indecisionLimit α tolerance εcov
      ((q.1, clusterDraws O q.1 c q.2) : Run Ω S (Pop K R))}
    have hset : F c ∩ {q | q.1.1 ∈ pinned O U b}
        = ((Prod.map id (fun d : RoundDraws S K R Xs L.M L.G L.Y L.N L.m =>
            ((d.1, d.2.1) : ClusterPart S K R L.M))) ⁻¹' E
          ∩ (pinned O U b) ×ˢ Set.univ) ×ˢ Set.univ := by
      ext q
      simp only [F, E, hpast, hgood, true_and, Set.mem_inter_iff, Set.mem_ofPred_eq,
        Set.mem_prod, Set.mem_preimage, Prod.map, id, Set.mem_univ, and_true]
    rw [hset, Measure.prod_prod, measure_univ, mul_one,
      prod_restrict_apply _ (measurableSet_pinned O U b)]
    have hmp := (MeasurePreserving.id (μ.restrict (pinned O U b))).prod
      (measurePreserving_clusterPart Dsamp Dsf νs L)
    refine (Measure.le_map_apply hmp.measurable.aemeasurable E).trans ?_
    rw [hmp.map_eq]
    refine (cluster_cell_le A O Dsamp Dsf c (hvReads L harvestReads)
      (schedule (populationsAfter Dsamp L.heavy c))
      indecisionLimit α tolerance εcov hM hPM (hsched _) hε hκ0 hP hPπ hπ₀
      (fun o ho hv hov => ⟨(hok o ho hv hov).1, hgood o ho hv hov, (hok o ho hv hov).2.1,
        (hok o ho hv hov).2.2.1, fun p f a h => ((hok o ho hv hov).2.2.2.2 p f a h).2⟩) hκf
      (popMeasure_atom_le O Dsamp c hε hε1 hR hκ hπ₀ hκaπ fun o ho hv hov =>
        ⟨hgood o ho hv hov, ⟨_, (hok o ho hv hov).1⟩, (hok o ho hv hov).2.2.2.1⟩)
      (hclus c hpast hgood) U b).trans ?_
    have hUT : (U.card : ℝ) ≤ T := by exact_mod_cast hU
    rw [hδc']
    gcongr
  have hloop := loop_round_le O (roundLaw Dsamp Dsf νs L) (streamLaw (R := R) Dsamp) r
    (fun ω x => history O Dsamp schedule indecisionLimit α L ω x r)
    (fun ω x => readsUpTo O Dsamp schedule indecisionLimit α L stageReads harvestReads ω x r)
    (fun ω x x' hx => history_dep O Dsamp schedule indecisionLimit α L stageReads harvestReads ω
      x x' r fun i hi => hx i hi)
    (fun x ω ω' h => history_congr hreads hhv x r h)
    (fun ω x => (card_readsUpTo_le hTs hTh ω x hM r r.isLt.le).trans
      ((Nat.mul_le_mul_right _ r.isLt.le).trans hTK)) F _ hF
  rw [measureReal_def]
  refine ENNReal.toReal_le_of_le_ofReal hδc'0 ((measure_mono fun θ hθ => ?_).trans hloop)
  exact ⟨history_pastIn O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r, hθ.1, hθ.2⟩

variable {O Dsamp Dsf νs L schedule indecisionLimit α stageReads harvestReads} in
omit [MeasurableSpace Xs] [Countable Xs] [MeasurableSingletonClass Xs] [IsProbabilityMeasure Dsamp]
  [IsProbabilityMeasure μ] in
lemma history_extend (x0 : RoundDraws S K R Xs L.M L.G L.Y L.N L.m) (r : Fin K) (ω : Ω)
    (x : Fin K → RoundDraws S K R Xs L.M L.G L.Y L.N L.m) :
    history O Dsamp schedule indecisionLimit α L ω
        (extendDraws x0 r fun i : {i : Fin K // i < r} => x i) r
      = history O Dsamp schedule indecisionLimit α L ω x r
    ∧ readsUpTo O Dsamp schedule indecisionLimit α L stageReads harvestReads ω
        (extendDraws x0 r fun i : {i : Fin K // i < r} => x i) r
      = readsUpTo O Dsamp schedule indecisionLimit α L stageReads harvestReads ω x r :=
  history_dep O Dsamp schedule indecisionLimit α L stageReads harvestReads ω _ _ r fun i hi => by
    simp [extendDraws, show i < r from hi]

/-- The stage's spec at round `r`, where the family cuts little badly: the family it is handed is
fixed by the cell. -/
theorem stage_round_le (tolerance ζ₀ x₀ π₁ θh δs : ℝ) {Ts Th T : ℕ}
    (hreads : ∀ F B c s, ReadsOnly O (fun ω => stageReads F B c ω s)
      (fun ω => (L.stage F B c ω s, L.harvest F B c ω s)))
    (hhv : ∀ F B c ω s p (f f' : S → ℝ), (∀ t ∈ harvestReads F B c ω s p f, f t = f' t) →
      harvestReads F B c ω s p f' = harvestReads F B c ω s p f
        ∧ L.harvest F B c ω s p f' = L.harvest F B c ω s p f)
    (hTs : ∀ F B c ω s, (stageReads F B c ω s).card ≤ Ts)
    (hTh : ∀ F B c ω s p f, (harvestReads F B c ω s p f).card ≤ Th)
    (hTK : K * (2 * Fintype.card (Pop K R) * L.M * (L.M + 1) + Ts
      + 2 * Fintype.card (Pop K R) * L.M * Th + L.G * (L.M + 1) + L.Y * Th
      + L.N * (L.m + 1)) ≤ T)
    (hM : 1 ≤ L.M)
    (hstage : ∀ F B c (U : Finset S) b, U.card ≤ T →
      badlyCut O tolerance B F Dsamp < x₀ →
      ((μ[|pinned O U b]).prod νs).real
          {q | ζ₀ < cutDisagreement O B F (L.stage F B c q.1 q.2) Dsamp q.1
            ∧ ¬ (π₁ ≤ harvestYield O (L.harvest F B c q.1 q.2) Dsamp
              ∧ ∃ qq, (∃ p, A.state p = qq
                  ∧ tolerance < undecidedProb O B.lo B.hi F p)
                ∧ θh < stateMass A (harvestLaw O (L.harvest F B c q.1 q.2) Dsamp) qq)}
        ≤ δs)
    (hδs0 : 0 ≤ δs) (r : Fin K) :
    (learnerMeasure (μ := μ) Dsamp Dsf νs L).real
      {θ | badlyCut O tolerance (roundState O Dsamp L.heavy schedule indecisionLimit α
              (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
              ((θ.2.1 r).1, (θ.2.1 r).2.1)) (roundFamily O Dsamp L.heavy schedule indecisionLimit α
              (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
              ((θ.2.1 r).1, (θ.2.1 r).2.1)) Dsamp < x₀
        ∧ ζ₀ < cutDisagreement O (roundState O Dsamp L.heavy schedule indecisionLimit α
              (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
              ((θ.2.1 r).1, (θ.2.1 r).2.1)) (roundFamily O Dsamp L.heavy schedule indecisionLimit α
              (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
              ((θ.2.1 r).1, (θ.2.1 r).2.1)) (roundHyp O Dsamp schedule indecisionLimit α L
              (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
              ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp θ.1
        ∧ ¬ (π₁ ≤ harvestYield O (roundHarv O Dsamp schedule indecisionLimit α L
              (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
              ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp
          ∧ ∃ qq, (∃ p, A.state p = qq ∧ tolerance < undecidedProb O
              (roundState O Dsamp L.heavy schedule indecisionLimit α
                (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
                ((θ.2.1 r).1, (θ.2.1 r).2.1)).lo
              (roundState O Dsamp L.heavy schedule indecisionLimit α
                (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
                ((θ.2.1 r).1, (θ.2.1 r).2.1)).hi
              (roundFamily O Dsamp L.heavy schedule indecisionLimit α
                (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
                ((θ.2.1 r).1, (θ.2.1 r).2.1)) p)
            ∧ θh < stateMass A (harvestLaw O (roundHarv O Dsamp schedule indecisionLimit α L
              (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
              ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp) qq)} ≤ δs := by
  classical
  obtain ⟨x0⟩ := nonempty_of_isProbabilityMeasure (roundLaw Dsamp Dsf νs L)
  set hist : Ω → ({i : Fin K // i < r} → RoundDraws S K R Xs L.M L.G L.Y L.N L.m) →
      List (Outcome S R) := fun ω a =>
    history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r a) r
  set stage : Ω → ({i : Fin K // i < r} → RoundDraws S K R Xs L.M L.G L.Y L.N L.m)
      × ClusterPart S K R L.M → State × Finset S × ClusterDraws S K R := fun ω ae =>
    (roundState O Dsamp L.heavy schedule indecisionLimit α (hist ω ae.1) ω ae.2,
      roundFamily O Dsamp L.heavy schedule indecisionLimit α (hist ω ae.1) ω ae.2,
      clusterDraws O ω (hist ω ae.1) ae.2)
  set queried : Ω → ({i : Fin K // i < r} → RoundDraws S K R Xs L.M L.G L.Y L.N L.m)
      × ClusterPart S K R L.M → Finset S := fun ω ae =>
    readsUpTo O Dsamp schedule indecisionLimit α L stageReads harvestReads ω
        (extendDraws x0 r ae.1) r
      ∪ drawReads O L harvestReads ω (hist ω ae.1) ae.2
      ∪ clusterReads O Dsamp ω L.heavy (hist ω ae.1) ae.2
  have hreads' : ∀ ae, ReadsOnly O (fun ω => queried ω ae) (fun ω => stage ω ae) := by
    rintro ⟨a, e⟩ ω ω' h
    obtain ⟨h1, h2⟩ := history_congr hreads hhv (extendDraws x0 r a) r
      fun w hw => h w (Finset.mem_union_left _ (Finset.mem_union_left _ hw))
    have h3 : hist ω' a = hist ω a := h2
    obtain ⟨hdr, hd⟩ := clusterDraws_congr hhv
      (history_pastIn O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r a) r) e
      (ω := ω) (ω' := ω') fun w hw => h w (Finset.mem_union_left _ (Finset.mem_union_right _ hw))
    have hc : ∀ w ∈ clusterReads O Dsamp ω L.heavy (hist ω a) e, O.noise w ω = O.noise w ω' :=
      fun w hw => h w (Finset.mem_union_right _ hw)
    have hcr : clusterReads O Dsamp ω' L.heavy (hist ω a) e
        = clusterReads O Dsamp ω L.heavy (hist ω a) e := by
      unfold clusterReads entries
      rw [hd]
    refine ⟨?_, ?_⟩
    · change _ ∪ drawReads O L harvestReads ω' (hist ω' a) e
        ∪ clusterReads O Dsamp ω' L.heavy (hist ω' a) e = _
      rw [h1, h3, hdr, hcr]
    · change (roundState O Dsamp L.heavy schedule indecisionLimit α (hist ω' a) ω' e, _, _) = _
      rw [h3, roundState_congr O Dsamp schedule L.heavy indecisionLimit α _ e hd hc,
        roundFamily_congr O Dsamp schedule L.heavy indecisionLimit α _ e hd hc]
      exact congrArg _ (congrArg _ hd)
  have hT : ∀ ω ae, (queried ω ae).card ≤ T := by
    rintro ω ⟨a, e⟩
    have hlen := length_history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r a) r
    rw [min_eq_left r.isLt.le] at hlen
    have h1 := card_readsUpTo_le (O := O) (Dsamp := Dsamp) (schedule := schedule)
      (indecisionLimit := indecisionLimit) (α := α) hTs hTh ω (extendDraws x0 r a) hM r r.isLt.le
    have h2 := card_drawReads_le O L harvestReads hTh ω (hist ω a) e
    have h3 := card_clusterReads_le_share O Dsamp ω L.heavy (hist ω a) e hM
      (by rw [hlen]; exact r.isLt)
    refine (Finset.card_union_le _ _).trans ((Nat.add_le_add_right (Finset.card_union_le _ _)
      _).trans ?_)
    refine le_trans ?_ hTK
    refine le_trans ?_ (add_share_le r.isLt (le_refl _))
    nlinarith
  set F : State × Finset S × ClusterDraws S K R →
      Set (Ω × ((Xs × ((Fin L.G → S) × (Fin L.Y → S) × ((Fin L.N → S) × (Fin L.m → S)))) × (R → ℕ → S))) :=
    fun c => {q | badlyCut O tolerance c.1 c.2.1 Dsamp < x₀
      ∧ (ζ₀ < cutDisagreement O c.1 c.2.1 (L.stage c.2.1 c.1 c.2.2 q.1 q.2.1.1) Dsamp q.1
        ∧ ¬ (π₁ ≤ harvestYield O (L.harvest c.2.1 c.1 c.2.2 q.1 q.2.1.1) Dsamp
          ∧ ∃ qq, (∃ p, A.state p = qq ∧ tolerance < undecidedProb O c.1.lo c.1.hi c.2.1 p)
            ∧ θh < stateMass A (harvestLaw O (L.harvest c.2.1 c.1 c.2.2 q.1 q.2.1.1) Dsamp) qq))}
  have hF : ∀ c U b, U.card ≤ T →
      (μ.prod ((νs.prod (tailLaw Dsamp Dsf L.G L.Y L.N L.m)).prod (streamLaw (R := R) Dsamp)))
        (F c ∩ {q | q.1 ∈ pinned O U b}) ≤ μ (pinned O U b) * ENNReal.ofReal δs := by
    intro c U b hU
    by_cases hbc : badlyCut O tolerance c.1 c.2.1 Dsamp < x₀
    swap
    · rw [show F c = ∅ from Set.eq_empty_of_forall_notMem fun q hq => hbc hq.1,
        Set.empty_inter, measure_empty]
      exact zero_le
    have hmp : MeasurePreserving (Prod.map id fun w : (Xs × ((Fin L.G → S) × (Fin L.Y → S) × ((Fin L.N → S) × (Fin L.m → S)))) × (R → ℕ → S) => w.1.1)
        (μ.prod ((νs.prod (tailLaw Dsamp Dsf L.G L.Y L.N L.m)).prod (streamLaw (R := R) Dsamp)))
        (μ.prod νs) :=
      (MeasurePreserving.id μ).prod (measurePreserving_fst.comp measurePreserving_fst)
    set G : Set (Ω × Xs) :=
      {p | ζ₀ < cutDisagreement O c.1 c.2.1 (L.stage c.2.1 c.1 c.2.2 p.1 p.2) Dsamp p.1
        ∧ ¬ (π₁ ≤ harvestYield O (L.harvest c.2.1 c.1 c.2.2 p.1 p.2) Dsamp
          ∧ ∃ qq, (∃ p, A.state p = qq ∧ tolerance < undecidedProb O c.1.lo c.1.hi c.2.1 p)
            ∧ θh < stateMass A (harvestLaw O (L.harvest c.2.1 c.1 c.2.2 p.1 p.2) Dsamp) qq)}
    have hset : F c ∩ {q | q.1 ∈ pinned O U b}
        = Prod.map id (fun w : (Xs × ((Fin L.G → S) × (Fin L.Y → S) × ((Fin L.N → S) × (Fin L.m → S)))) × (R → ℕ → S) => w.1.1)
          ⁻¹' (G ∩ (pinned O U b) ×ˢ Set.univ) := by
      ext q
      simp only [F, G, hbc, true_and, Set.mem_inter_iff, Set.mem_ofPred_eq, Set.mem_preimage,
        Prod.map, id, Set.mem_prod, Set.mem_univ, and_true]
    rw [hset]
    refine (Measure.le_map_apply hmp.measurable.aemeasurable _).trans ?_
    rw [hmp.map_eq, prod_restrict_apply _ (measurableSet_pinned O U b)]
    exact restrict_prod_le O U b νs G (hstage _ _ _ U b hU hbc)
  have hsplit := round_split_le O (roundLaw Dsamp Dsf νs L) (clusterLaw Dsamp Dsf L.M)
    (νs.prod (tailLaw Dsamp Dsf L.G L.Y L.N L.m)) (streamLaw (R := R) Dsamp) _
    (measurePreserving_clusterSplit Dsamp Dsf νs L) r stage queried hreads' hT F _ hF
  rw [measureReal_def]
  refine ENNReal.toReal_le_of_le_ofReal hδs0 ((measure_mono fun θ hθ => ?_).trans hsplit)
  simp only [Set.mem_ofPred_eq, F, stage, hist]
  rw [(history_extend (stageReads := stageReads) (harvestReads := harvestReads) x0 r θ.1
    θ.2.1).1]
  exact hθ

/-- Round `r`'s cut, hypothesis and harvest, from the earlier rounds' draws and its own
clustering and stage draws. -/
noncomputable def hypStage (x0 : RoundDraws S K R Xs L.M L.G L.Y L.N L.m) (r : Fin K) (ω : Ω)
    (ae : ({i : Fin K // i < r} → RoundDraws S K R Xs L.M L.G L.Y L.N L.m)
      × (ClusterPart S K R L.M × Xs)) : State × Finset S × DFA S R × Harvester S :=
  (roundState O Dsamp L.heavy schedule indecisionLimit α
      (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r ae.1) r) ω ae.2.1,
    roundFamily O Dsamp L.heavy schedule indecisionLimit α
      (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r ae.1) r) ω ae.2.1,
    roundHyp O Dsamp schedule indecisionLimit α L
      (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r ae.1) r) ω ae.2.1 ae.2.2,
    roundHarv O Dsamp schedule indecisionLimit α L
      (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r ae.1) r) ω ae.2.1 ae.2.2)

/-- What round `r` has read by its gate. -/
noncomputable def hypReads (x0 : RoundDraws S K R Xs L.M L.G L.Y L.N L.m) (r : Fin K) (ω : Ω)
    (ae : ({i : Fin K // i < r} → RoundDraws S K R Xs L.M L.G L.Y L.N L.m)
      × (ClusterPart S K R L.M × Xs)) : Finset S :=
  readsUpTo O Dsamp schedule indecisionLimit α L stageReads harvestReads ω
      (extendDraws x0 r ae.1) r
    ∪ drawReads O L harvestReads ω
      (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r ae.1) r) ae.2.1
    ∪ clusterReads O Dsamp ω L.heavy
      (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r ae.1) r) ae.2.1
    ∪ stageReads
      (roundFamily O Dsamp L.heavy schedule indecisionLimit α
        (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r ae.1) r) ω ae.2.1)
      (roundState O Dsamp L.heavy schedule indecisionLimit α
        (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r ae.1) r) ω ae.2.1)
      (clusterDraws O ω
        (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r ae.1) r) ae.2.1)
      ω ae.2.2

variable {O Dsamp Dsf νs L schedule indecisionLimit α stageReads harvestReads} in
omit [MeasurableSpace Xs] [Countable Xs] [MeasurableSingletonClass Xs]
  [IsProbabilityMeasure Dsamp] in
lemma hypStage_readsOnly
    (hreads : ∀ F B c s, ReadsOnly O (fun ω => stageReads F B c ω s)
      (fun ω => (L.stage F B c ω s, L.harvest F B c ω s)))
    (hhv : ∀ F B c ω s p (f f' : S → ℝ), (∀ t ∈ harvestReads F B c ω s p f, f t = f' t) →
      harvestReads F B c ω s p f' = harvestReads F B c ω s p f
        ∧ L.harvest F B c ω s p f' = L.harvest F B c ω s p f)
    (x0 : RoundDraws S K R Xs L.M L.G L.Y L.N L.m) (r : Fin K) (ae) :
    ReadsOnly O (fun ω => hypReads O Dsamp L schedule indecisionLimit α stageReads harvestReads
        x0 r ω ae)
      (fun ω => hypStage O Dsamp L schedule indecisionLimit α x0 r ω ae) := by
  intro ω ω' h
  obtain ⟨a, e, s⟩ := ae
  unfold hypReads at h
  obtain ⟨h1, h2⟩ := history_congr hreads hhv (extendDraws x0 r a) r
    fun w hw => h w (by simp only [Finset.mem_union]; tauto)
  set past := history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r a) r
  obtain ⟨hdr, hd⟩ := clusterDraws_congr hhv
    (history_pastIn O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r a) r) e
    (ω := ω) (ω' := ω') fun w hw => h w (by simp only [Finset.mem_union]; tauto)
  have hc : ∀ w ∈ clusterReads O Dsamp ω L.heavy past e, O.noise w ω = O.noise w ω' :=
    fun w hw => h w (by simp only [Finset.mem_union]; tauto)
  have hcr : clusterReads O Dsamp ω' L.heavy past e = clusterReads O Dsamp ω L.heavy past e := by
    unfold clusterReads entries
    rw [hd]
  have hB := roundState_congr O Dsamp schedule L.heavy indecisionLimit α past e hd hc
  have hF := roundFamily_congr O Dsamp schedule L.heavy indecisionLimit α past e hd hc
  have hs := hreads (roundFamily O Dsamp L.heavy schedule indecisionLimit α past ω e)
    (roundState O Dsamp L.heavy schedule indecisionLimit α past ω e) (clusterDraws O ω past e) s
    ω ω' fun w hw => h w (by simp only [Finset.mem_union]; tauto)
  refine ⟨?_, ?_⟩
  · simp only [hypReads]
    rw [h1, h2, hdr, hcr, hB, hF, hd]
    exact congrArg _ hs.1
  · simp only [hypStage, roundHyp, roundHarv]
    rw [h2, hB, hF, hd]
    exact congrArg _ (congrArg _ (Prod.ext (congrArg Prod.fst hs.2) (congrArg Prod.snd hs.2)))

variable {O Dsamp Dsf νs L schedule indecisionLimit α stageReads harvestReads} in
omit [MeasurableSpace Xs] [Countable Xs] [MeasurableSingletonClass Xs]
  [IsProbabilityMeasure Dsamp] in
lemma hypReads_card {Ts Th T : ℕ}
    (hTs : ∀ F B c ω s, (stageReads F B c ω s).card ≤ Ts)
    (hTh : ∀ F B c ω s p f, (harvestReads F B c ω s p f).card ≤ Th)
    (hTK : K * (2 * Fintype.card (Pop K R) * L.M * (L.M + 1) + Ts
      + 2 * Fintype.card (Pop K R) * L.M * Th + L.G * (L.M + 1) + L.Y * Th
      + L.N * (L.m + 1)) ≤ T)
    (hM : 1 ≤ L.M) (x0 : RoundDraws S K R Xs L.M L.G L.Y L.N L.m) (r : Fin K) (ω : Ω) (ae) :
    (hypReads O Dsamp L schedule indecisionLimit α stageReads harvestReads x0 r ω ae).card
      ≤ T := by
  obtain ⟨a, e, s⟩ := ae
  have hlen := length_history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r a) r
  rw [min_eq_left r.isLt.le] at hlen
  have h1 := card_readsUpTo_le (O := O) (Dsamp := Dsamp) (schedule := schedule)
    (indecisionLimit := indecisionLimit) (α := α) hTs hTh ω (extendDraws x0 r a) hM r r.isLt.le
  have h2 := card_drawReads_le O L harvestReads hTh ω
    (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r a) r) e
  have h3 := card_clusterReads_le_share O Dsamp ω L.heavy
    (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r a) r) e hM
    (by rw [hlen]; exact r.isLt)
  have h4 := hTs (roundFamily O Dsamp L.heavy schedule indecisionLimit α
        (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r a) r) ω e)
      (roundState O Dsamp L.heavy schedule indecisionLimit α
        (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r a) r) ω e)
      (clusterDraws O ω
        (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r a) r) e) ω s
  unfold hypReads
  refine (Finset.card_union_le _ _).trans ((Nat.add_le_add_right ((Finset.card_union_le _ _).trans
    (Nat.add_le_add_right (Finset.card_union_le _ _) _)) _).trans ?_)
  refine le_trans ?_ hTK
  refine le_trans ?_ (add_share_le r.isLt (le_refl _))
  nlinarith

omit [MeasurableSpace Xs] [Countable Xs] [MeasurableSingletonClass Xs]
  [IsProbabilityMeasure Dsamp] [IsProbabilityMeasure Dsf] in
lemma card_roundFamily_le (past : List (Outcome S R)) (ω : Ω) (e : ClusterPart S K R L.M) :
    (roundFamily O Dsamp L.heavy schedule indecisionLimit α past ω e).card ≤ L.M + 1 :=
  (Finset.card_le_card ((clusterAt_subset O _ _ _).trans
    (poolAt_subset_clusterPool O ω past e _))).trans ((Finset.card_insert_le _ _).trans
      (Nat.add_le_add_right (Finset.card_image_le.trans (Finset.card_range _).le) 1))
/-- The gate at round `r` lets through a hypothesis that mislabels more than `ζ₂ + ζ₁` of what
the family cuts well. -/
theorem pass_round_le (tolerance ζ₁ ζ₂ : ℝ) {κ : ℝ} {Ts Th T : ℕ}
    (hreads : ∀ F B c s, ReadsOnly O (fun ω => stageReads F B c ω s)
      (fun ω => (L.stage F B c ω s, L.harvest F B c ω s)))
    (hhv : ∀ F B c ω s p (f f' : S → ℝ), (∀ t ∈ harvestReads F B c ω s p f, f t = f' t) →
      harvestReads F B c ω s p f' = harvestReads F B c ω s p f
        ∧ L.harvest F B c ω s p f' = L.harvest F B c ω s p f)
    (hTs : ∀ F B c ω s, (stageReads F B c ω s).card ≤ Ts)
    (hTh : ∀ F B c ω s p f, (harvestReads F B c ω s p f).card ≤ Th)
    (hTK : K * (2 * Fintype.card (Pop K R) * L.M * (L.M + 1) + Ts
      + 2 * Fintype.card (Pop K R) * L.M * Th + L.G * (L.M + 1) + L.Y * Th
      + L.N * (L.m + 1)) ≤ T)
    (hM : 1 ≤ L.M) (hL : O.L = {v | A.state v ∈ A.accept})
    (hκ : ∀ a, Dsamp.real {a} ≤ κ) (htol : 0 ≤ tolerance) (hζ₁ : 0 < ζ₁)
    (hg : (L.gth : ℝ) ≤ ζ₂ * L.G) (r : Fin K) :
    (learnerMeasure (μ := μ) Dsamp Dsf νs L).real
      {θ | roundGate O Dsamp schedule indecisionLimit α L
          (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1 (θ.2.1 r).2.2.2.1
        ∧ ζ₂ + ζ₁ < mislabelledWellCut A O tolerance
          (roundState O Dsamp L.heavy schedule indecisionLimit α
            (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1))
          (roundFamily O Dsamp L.heavy schedule indecisionLimit α
            (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1))
          (roundHyp O Dsamp schedule indecisionLimit α L
            (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp}
      ≤ Real.exp (-2 * L.G * (ζ₂ - L.gth / L.G) ^ 2)
        + (tolerance + T * (L.M + 1) * κ) / ζ₁ := by
  classical
  obtain ⟨x0⟩ := nonempty_of_isProbabilityMeasure (roundLaw Dsamp Dsf νs L)
  have hκ0 : 0 ≤ κ := le_trans measureReal_nonneg (hκ 1)
  set e2 := Real.exp (-2 * L.G * (ζ₂ - L.gth / L.G) ^ 2)
  set Mk := (tolerance + T * (L.M + 1) * κ) / ζ₁
  have hMk : 0 ≤ Mk := div_nonneg (by positivity) hζ₁.le
  set Tl := (Fin L.G → S) × (Fin L.Y → S) × ((Fin L.N → S) × (Fin L.m → S))
  set F : State × Finset S × DFA S R × Harvester S → Set (Ω × (Tl × (R → ℕ → S))) := fun c =>
    {q | c.2.1.card ≤ L.M + 1
      ∧ (Finset.univ.filter fun i => ¬ cutAgrees O c.1 c.2.1 c.2.2.1 (q.2.1.1 i) q.1).card ≤ L.gth
      ∧ ζ₂ + ζ₁ < mislabelledWellCut A O tolerance c.1 c.2.1 c.2.2.1 Dsamp}
  have hF : ∀ c U b, U.card ≤ T →
      (μ.prod ((tailLaw Dsamp Dsf L.G L.Y L.N L.m).prod (streamLaw (R := R) Dsamp)))
        (F c ∩ {q | q.1 ∈ pinned O U b}) ≤ μ (pinned O U b) * ENNReal.ofReal (e2 + Mk) := by
    intro c U b hU
    by_cases hFc : c.2.1.card ≤ L.M + 1
    swap
    · rw [show F c = ∅ from Set.eq_empty_of_forall_notMem fun q hq => hFc hq.1,
        Set.empty_inter, measure_empty]
      exact zero_le
    set W : Ω → Set S := fun ω => {p | ¬ cutAgrees O c.1 c.2.1 c.2.2.1 p ω}
    set Mset : Set Ω := {ω | ζ₁ < Dsamp.real {p | miscutProb O c.1.lo c.1.hi c.2.1 p ≤ tolerance
      ∧ ¬ cutCorrect O c.1.lo c.1.hi c.2.1 p ω}}
    set E : Set Ω := pinned O U b ∩ {ω | ζ₂ < cutDisagreement O c.1 c.2.1 c.2.2.1 Dsamp ω}
    have hE : MeasurableSet E := (measurableSet_pinned O U b).inter
      (measurableSet_lt measurable_const (measurable_cutDisagreement O c.1 c.2.1 c.2.2.1 Dsamp))
    have hsub : F c ∩ {q | q.1 ∈ pinned O U b}
        ⊆ Prod.map id (fun w : Tl × (R → ℕ → S) => w.1.1) ⁻¹'
            {q | q.1 ∈ E ∧ (Finset.univ.filter fun i => q.2 i ∈ W q.1).card ≤ L.gth}
          ∪ (Mset ∩ pinned O U b) ×ˢ Set.univ := by
      rintro q ⟨⟨-, hg', hmis⟩, hq⟩
      have := mislabelled_le A O hL tolerance c.1 c.2.1 c.2.2.1 Dsamp q.1
      by_cases hD : ζ₂ < cutDisagreement O c.1 c.2.1 c.2.2.1 Dsamp q.1
      · exact Or.inl ⟨⟨hq, hD⟩, hg'⟩
      · refine Or.inr ⟨⟨?_, hq⟩, trivial⟩
        change ζ₁ < _
        linarith [not_lt.1 hD]
    have hmp : MeasurePreserving (Prod.map id fun w : Tl × (R → ℕ → S) => w.1.1)
        (μ.prod ((tailLaw Dsamp Dsf L.G L.Y L.N L.m).prod (streamLaw (R := R) Dsamp)))
        (μ.prod (Measure.pi fun _ : Fin L.G => Dsamp)) :=
      (MeasurePreserving.id μ).prod (measurePreserving_fst.comp measurePreserving_fst)
    have h1 : (μ.prod ((tailLaw Dsamp Dsf L.G L.Y L.N L.m).prod (streamLaw (R := R) Dsamp)))
        (Prod.map id (fun w : Tl × (R → ℕ → S) => w.1.1) ⁻¹'
          {q | q.1 ∈ E ∧ (Finset.univ.filter fun i => q.2 i ∈ W q.1).card ≤ L.gth})
        ≤ μ (pinned O U b) * ENNReal.ofReal e2 := by
      refine (Measure.le_map_apply hmp.measurable.aemeasurable _).trans ?_
      rw [hmp.map_eq]
      refine (gate_prod_le μ Dsamp L.G W
        (fun p => (measurableSet_cutAgrees O c.1 c.2.1 c.2.2.1 p).compl) E hE (fun n => n ≤ L.gth)
        fun ω hω => pi_pass_le Dsamp L.G L.gth (W ω) hg hω.2).trans ?_
      gcongr
      exact Set.inter_subset_left
    have h2 : (μ.prod ((tailLaw Dsamp Dsf L.G L.Y L.N L.m).prod (streamLaw (R := R) Dsamp)))
        ((Mset ∩ pinned O U b) ×ˢ Set.univ) ≤ μ (pinned O U b) * ENNReal.ofReal Mk := by
      rw [Measure.prod_prod, measure_univ, mul_one]
      refine (cut_err_le O Dsamp U b c.1.lo c.1.hi c.2.1 hζ₁ htol hκ).trans ?_
      have hUT : (U.card : ℝ) ≤ T := by exact_mod_cast hU
      have hFM : (c.2.1.card : ℝ) ≤ L.M + 1 := by exact_mod_cast hFc
      have : (U.card : ℝ) * c.2.1.card * κ ≤ T * (L.M + 1) * κ := by gcongr
      have hxy : (tolerance + U.card * c.2.1.card * κ) / ζ₁ ≤ Mk :=
        div_le_div_of_nonneg_right (by linarith) hζ₁.le
      gcongr
    calc (μ.prod ((tailLaw Dsamp Dsf L.G L.Y L.N L.m).prod (streamLaw (R := R) Dsamp)))
          (F c ∩ {q | q.1 ∈ pinned O U b})
        ≤ μ (pinned O U b) * ENNReal.ofReal e2 + μ (pinned O U b) * ENNReal.ofReal Mk :=
          (measure_mono hsub).trans ((measure_union_le _ _).trans (add_le_add h1 h2))
      _ = μ (pinned O U b) * ENNReal.ofReal (e2 + Mk) := by
          rw [← mul_add, ← ENNReal.ofReal_add (Real.exp_pos _).le hMk]
  have hsplit := round_split_le O (roundLaw Dsamp Dsf νs L)
    ((clusterLaw Dsamp Dsf L.M).prod νs) (tailLaw Dsamp Dsf L.G L.Y L.N L.m)
    (streamLaw (R := R) Dsamp) _
    (measurePreserving_checkSplit Dsamp Dsf νs L) r
    (hypStage O Dsamp L schedule indecisionLimit α x0 r)
    (hypReads O Dsamp L schedule indecisionLimit α stageReads harvestReads x0 r)
    (hypStage_readsOnly hreads hhv x0 r) (hypReads_card hTs hTh hTK hM x0 r) F _ hF
  rw [measureReal_def]
  refine ENNReal.toReal_le_of_le_ofReal (add_nonneg (Real.exp_pos _).le hMk)
    ((measure_mono fun θ hθ => ?_).trans hsplit)
  simp only [Set.mem_ofPred_eq, F, hypStage]
  rw [(history_extend (stageReads := stageReads) (harvestReads := harvestReads) x0 r θ.1 θ.2.1).1]
  exact ⟨card_roundFamily_le O Dsamp L schedule indecisionLimit α _ _ _, hθ.1, hθ.2⟩

/-- The gate at round `r` refuses a hypothesis that disagrees with the cut on at most `ζ₀`. -/
theorem refuse_round_le (ζ₀ : ℝ) {Ts Th T : ℕ}
    (hreads : ∀ F B c s, ReadsOnly O (fun ω => stageReads F B c ω s)
      (fun ω => (L.stage F B c ω s, L.harvest F B c ω s)))
    (hhv : ∀ F B c ω s p (f f' : S → ℝ), (∀ t ∈ harvestReads F B c ω s p f, f t = f' t) →
      harvestReads F B c ω s p f' = harvestReads F B c ω s p f
        ∧ L.harvest F B c ω s p f' = L.harvest F B c ω s p f)
    (hTs : ∀ F B c ω s, (stageReads F B c ω s).card ≤ Ts)
    (hTh : ∀ F B c ω s p f, (harvestReads F B c ω s p f).card ≤ Th)
    (hTK : K * (2 * Fintype.card (Pop K R) * L.M * (L.M + 1) + Ts
      + 2 * Fintype.card (Pop K R) * L.M * Th + L.G * (L.M + 1) + L.Y * Th
      + L.N * (L.m + 1)) ≤ T)
    (hM : 1 ≤ L.M) (hg : ζ₀ * L.G ≤ L.gth) (r : Fin K) :
    (learnerMeasure (μ := μ) Dsamp Dsf νs L).real
      {θ | ¬ roundGate O Dsamp schedule indecisionLimit α L
          (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1 (θ.2.1 r).2.2.2.1
        ∧ cutDisagreement O
          (roundState O Dsamp L.heavy schedule indecisionLimit α
            (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1))
          (roundFamily O Dsamp L.heavy schedule indecisionLimit α
            (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1))
          (roundHyp O Dsamp schedule indecisionLimit α L
            (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp θ.1 ≤ ζ₀}
      ≤ Real.exp (-2 * L.G * (L.gth / L.G - ζ₀) ^ 2) := by
  classical
  obtain ⟨x0⟩ := nonempty_of_isProbabilityMeasure (roundLaw Dsamp Dsf νs L)
  set e1 := Real.exp (-2 * L.G * (L.gth / L.G - ζ₀) ^ 2)
  set Tl := (Fin L.G → S) × (Fin L.Y → S) × ((Fin L.N → S) × (Fin L.m → S))
  set F : State × Finset S × DFA S R × Harvester S → Set (Ω × (Tl × (R → ℕ → S))) := fun c =>
    {q | ¬ (Finset.univ.filter fun i => ¬ cutAgrees O c.1 c.2.1 c.2.2.1 (q.2.1.1 i) q.1).card
        ≤ L.gth
      ∧ cutDisagreement O c.1 c.2.1 c.2.2.1 Dsamp q.1 ≤ ζ₀}
  have hF : ∀ c U b, U.card ≤ T →
      (μ.prod ((tailLaw Dsamp Dsf L.G L.Y L.N L.m).prod (streamLaw (R := R) Dsamp)))
        (F c ∩ {q | q.1 ∈ pinned O U b}) ≤ μ (pinned O U b) * ENNReal.ofReal e1 := by
    intro c U b _
    set W : Ω → Set S := fun ω => {p | ¬ cutAgrees O c.1 c.2.1 c.2.2.1 p ω}
    set E : Set Ω := pinned O U b ∩ {ω | cutDisagreement O c.1 c.2.1 c.2.2.1 Dsamp ω ≤ ζ₀}
    have hE : MeasurableSet E := (measurableSet_pinned O U b).inter
      (measurableSet_le (measurable_cutDisagreement O c.1 c.2.1 c.2.2.1 Dsamp) measurable_const)
    have hsub : F c ∩ {q | q.1 ∈ pinned O U b}
        ⊆ Prod.map id (fun w : Tl × (R → ℕ → S) => w.1.1) ⁻¹'
            {q | q.1 ∈ E ∧ L.gth < (Finset.univ.filter fun i => q.2 i ∈ W q.1).card} := by
      rintro q ⟨⟨hg', hD⟩, hq⟩
      exact ⟨⟨hq, hD⟩, not_le.1 hg'⟩
    have hmp : MeasurePreserving (Prod.map id fun w : Tl × (R → ℕ → S) => w.1.1)
        (μ.prod ((tailLaw Dsamp Dsf L.G L.Y L.N L.m).prod (streamLaw (R := R) Dsamp)))
        (μ.prod (Measure.pi fun _ : Fin L.G => Dsamp)) :=
      (MeasurePreserving.id μ).prod (measurePreserving_fst.comp measurePreserving_fst)
    refine (measure_mono hsub).trans ((Measure.le_map_apply hmp.measurable.aemeasurable _).trans ?_)
    rw [hmp.map_eq]
    refine (gate_prod_le μ Dsamp L.G W
      (fun p => (measurableSet_cutAgrees O c.1 c.2.1 c.2.2.1 p).compl) E hE (fun n => L.gth < n)
      fun ω hω => pi_refuse_le Dsamp L.G L.gth (W ω) hg hω.2).trans ?_
    gcongr
    exact Set.inter_subset_left
  have hsplit := round_split_le O (roundLaw Dsamp Dsf νs L)
    ((clusterLaw Dsamp Dsf L.M).prod νs) (tailLaw Dsamp Dsf L.G L.Y L.N L.m)
    (streamLaw (R := R) Dsamp) _
    (measurePreserving_checkSplit Dsamp Dsf νs L) r
    (hypStage O Dsamp L schedule indecisionLimit α x0 r)
    (hypReads O Dsamp L schedule indecisionLimit α stageReads harvestReads x0 r)
    (hypStage_readsOnly hreads hhv x0 r) (hypReads_card hTs hTh hTK hM x0 r) F _ hF
  rw [measureReal_def]
  refine ENNReal.toReal_le_of_le_ofReal (Real.exp_pos _).le
    ((measure_mono fun θ hθ => ?_).trans hsplit)
  simp only [Set.mem_ofPred_eq, F, hypStage]
  rw [(history_extend (stageReads := stageReads) (harvestReads := harvestReads) x0 r θ.1 θ.2.1).1]
  exact hθ

/-- The check's spec at round `r`: it fails a state with a small minority rarely. -/
theorem check_round_le (wₛ : ℝ) {κ : ℝ} {Ts Th T : ℕ}
    (hreads : ∀ F B c s, ReadsOnly O (fun ω => stageReads F B c ω s)
      (fun ω => (L.stage F B c ω s, L.harvest F B c ω s)))
    (hhv : ∀ F B c ω s p (f f' : S → ℝ), (∀ t ∈ harvestReads F B c ω s p f, f t = f' t) →
      harvestReads F B c ω s p f' = harvestReads F B c ω s p f
        ∧ L.harvest F B c ω s p f' = L.harvest F B c ω s p f)
    (hTs : ∀ F B c ω s, (stageReads F B c ω s).card ≤ Ts)
    (hTh : ∀ F B c ω s p f, (harvestReads F B c ω s p f).card ≤ Th)
    (hTK : K * (2 * Fintype.card (Pop K R) * L.M * (L.M + 1) + Ts
      + 2 * Fintype.card (Pop K R) * L.M * Th + L.G * (L.M + 1) + L.Y * Th
      + L.N * (L.m + 1)) ≤ T)
    (hM : 1 ≤ L.M) (hκ : ∀ a, Dsamp.real {a} ≤ κ) (hε : 0 < L.heavy)
    (hw : 0 ≤ wₛ)
    (hcheck : ∀ (H : DFA S R) (h : R) (U : Finset S) (b : S → ℝ),
      L.heavy / Fintype.card R ≤ Dsamp.real {v | H.state v = h} → U.card ≤ T →
      ((μ[|pinned O U b]).prod (checkLaw Dsamp Dsf L.N L.m)).real
          {z | checkFails O H L.n L.t h z.2 z.1}
        ≤ L.m * Real.exp (-2 * L.t ^ 2 / L.n) + 2 * L.n * minorityShare A H Dsamp h
          + κ * Fintype.card R / L.heavy * (2 * L.n ^ 2 + 2 * L.n * (L.m + 1) * T))
    (r : Fin K) :
    (learnerMeasure (μ := μ) Dsamp Dsf νs L).real
      {θ | ∃ h, L.heavy / Fintype.card R ≤ Dsamp.real {v | (roundHyp O Dsamp schedule
          indecisionLimit α L (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1).state v = h}
        ∧ checkFails O (roundHyp O Dsamp schedule indecisionLimit α L
          (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) L.n L.t h (θ.2.1 r).2.2.2.2.2 θ.1
        ∧ minorityShare A (roundHyp O Dsamp schedule indecisionLimit α L
          (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp h < wₛ}
      ≤ Fintype.card R * (L.m * Real.exp (-2 * L.t ^ 2 / L.n) + 2 * L.n * wₛ
        + κ * Fintype.card R / L.heavy * (2 * L.n ^ 2 + 2 * L.n * (L.m + 1) * T)) := by
  classical
  obtain ⟨x0⟩ := nonempty_of_isProbabilityMeasure (roundLaw Dsamp Dsf νs L)
  have hκ0 : 0 ≤ κ := le_trans measureReal_nonneg (hκ 1)
  set b0 := L.m * Real.exp (-2 * L.t ^ 2 / L.n) + 2 * L.n * wₛ
    + κ * Fintype.card R / L.heavy * (2 * L.n ^ 2 + 2 * L.n * (L.m + 1) * T)
  have hb0 : 0 ≤ b0 := by positivity
  set Chk := (Fin L.N → S) × (Fin L.m → S)
  set Tl := (Fin L.G → S) × (Fin L.Y → S) × Chk
  set F : State × Finset S × DFA S R × Harvester S → Set (Ω × (Tl × (R → ℕ → S))) := fun c =>
    {q | ∃ h, L.heavy / Fintype.card R ≤ Dsamp.real {v | c.2.2.1.state v = h}
      ∧ checkFails O c.2.2.1 L.n L.t h q.2.1.2.2 q.1 ∧ minorityShare A c.2.2.1 Dsamp h < wₛ}
  have hF : ∀ c U b, U.card ≤ T →
      (μ.prod ((tailLaw Dsamp Dsf L.G L.Y L.N L.m).prod (streamLaw (R := R) Dsamp)))
        (F c ∩ {q | q.1 ∈ pinned O U b})
        ≤ μ (pinned O U b) * ENNReal.ofReal (Fintype.card R * b0) := by
    intro c U b hU
    set H := c.2.2.1
    set Bad := Finset.univ.filter fun h =>
      L.heavy / Fintype.card R ≤ Dsamp.real {v | H.state v = h} ∧ minorityShare A H Dsamp h < wₛ
    set C : R → Set (Ω × Chk) := fun h => {z | checkFails O H L.n L.t h z.2 z.1}
    have hmp : MeasurePreserving (Prod.map id fun w : Tl × (R → ℕ → S) => w.1.2.2)
        (μ.prod ((tailLaw Dsamp Dsf L.G L.Y L.N L.m).prod (streamLaw (R := R) Dsamp)))
        (μ.prod (checkLaw Dsamp Dsf L.N L.m)) :=
      (MeasurePreserving.id μ).prod
        ((measurePreserving_snd.comp measurePreserving_snd).comp measurePreserving_fst)
    have hsub : F c ∩ {q | q.1 ∈ pinned O U b} ⊆ ⋃ h ∈ Bad,
        Prod.map id (fun w : Tl × (R → ℕ → S) => w.1.2.2) ⁻¹'
          (C h ∩ (pinned O U b) ×ˢ Set.univ) := by
      rintro q ⟨⟨h, hh, hf, hm⟩, hq⟩
      exact Set.mem_biUnion (Finset.mem_filter.2 ⟨Finset.mem_univ _, hh, hm⟩)
        ⟨hf, hq, Set.mem_univ _⟩
    have hone : ∀ h ∈ Bad,
        (μ.prod ((tailLaw Dsamp Dsf L.G L.Y L.N L.m).prod (streamLaw (R := R) Dsamp)))
        (Prod.map id (fun w : Tl × (R → ℕ → S) => w.1.2.2) ⁻¹'
          (C h ∩ (pinned O U b) ×ˢ Set.univ))
        ≤ μ (pinned O U b) * ENNReal.ofReal b0 := by
      intro h hh
      obtain ⟨hheavyh, hmin⟩ := (Finset.mem_filter.1 hh).2
      refine (Measure.le_map_apply hmp.measurable.aemeasurable _).trans ?_
      rw [hmp.map_eq, prod_restrict_apply _ (measurableSet_pinned O U b)]
      refine restrict_prod_le O U b _ (C h) ?_
      refine (hcheck H h U b hheavyh hU).trans ?_
      have : 2 * (L.n : ℝ) * minorityShare A H Dsamp h ≤ 2 * L.n * wₛ :=
        mul_le_mul_of_nonneg_left hmin.le (by positivity)
      simp only [b0]
      linarith
    calc (μ.prod ((tailLaw Dsamp Dsf L.G L.Y L.N L.m).prod (streamLaw (R := R) Dsamp)))
          (F c ∩ {q | q.1 ∈ pinned O U b})
        ≤ ∑ h ∈ Bad, (μ.prod ((tailLaw Dsamp Dsf L.G L.Y L.N L.m).prod (streamLaw (R := R) Dsamp)))
          (Prod.map id (fun w : Tl × (R → ℕ → S) => w.1.2.2) ⁻¹'
            (C h ∩ (pinned O U b) ×ˢ Set.univ)) :=
          (measure_mono hsub).trans (measure_biUnion_finset_le _ _)
      _ ≤ ∑ _h ∈ Bad, μ (pinned O U b) * ENNReal.ofReal b0 := Finset.sum_le_sum hone
      _ ≤ ∑ _h : R, μ (pinned O U b) * ENNReal.ofReal b0 :=
          Finset.sum_le_sum_of_subset (Finset.filter_subset _ _)
      _ = μ (pinned O U b) * ENNReal.ofReal (Fintype.card R * b0) := by
          rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, ENNReal.ofReal_mul (by positivity),
            ENNReal.ofReal_natCast]
          ring
  have hsplit := round_split_le O (roundLaw Dsamp Dsf νs L)
    ((clusterLaw Dsamp Dsf L.M).prod νs) (tailLaw Dsamp Dsf L.G L.Y L.N L.m)
    (streamLaw (R := R) Dsamp) _
    (measurePreserving_checkSplit Dsamp Dsf νs L) r
    (hypStage O Dsamp L schedule indecisionLimit α x0 r)
    (hypReads O Dsamp L schedule indecisionLimit α stageReads harvestReads x0 r)
    (hypStage_readsOnly hreads hhv x0 r) (hypReads_card hTs hTh hTK hM x0 r) F _ hF
  rw [measureReal_def]
  refine ENNReal.toReal_le_of_le_ofReal (by positivity) ((measure_mono fun θ hθ => ?_).trans hsplit)
  simp only [Set.mem_ofPred_eq, F, hypStage]
  rw [(history_extend (stageReads := stageReads) (harvestReads := harvestReads) x0 r θ.1 θ.2.1).1]
  exact hθ

/-- `ReturnAccuracy`'s round at round `r`: a check passing every heavy state, but a hypothesis
inaccurate once denoised. -/
theorem return_round_le (w δ β : ℝ) {η₀ κ : ℝ} {Ts Th T : ℕ}
    (hreads : ∀ F B c s, ReadsOnly O (fun ω => stageReads F B c ω s)
      (fun ω => (L.stage F B c ω s, L.harvest F B c ω s)))
    (hhv : ∀ F B c ω s p (f f' : S → ℝ), (∀ t ∈ harvestReads F B c ω s p f, f t = f' t) →
      harvestReads F B c ω s p f' = harvestReads F B c ω s p f
        ∧ L.harvest F B c ω s p f' = L.harvest F B c ω s p f)
    (hTs : ∀ F B c ω s, (stageReads F B c ω s).card ≤ Ts)
    (hTh : ∀ F B c ω s p f, (harvestReads F B c ω s p f).card ≤ Th)
    (hTK : K * (2 * Fintype.card (Pop K R) * L.M * (L.M + 1) + Ts
      + 2 * Fintype.card (Pop K R) * L.M * Th + L.G * (L.M + 1) + L.Y * Th
      + L.N * (L.m + 1)) ≤ T)
    (hM : 1 ≤ L.M) (hL : O.L = {v | A.state v ∈ A.accept})
    (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hw0 : 0 ≤ w) (hγ : 0 < denoiseMargin η₀ w)
    (hε : 0 < L.heavy) (hδ : 0 < δ) (hκ : ∀ a, Dsamp.real {a} ≤ κ)
    (hn : Real.log (2 * Fintype.card R / δ) / (2 * denoiseMargin η₀ w ^ 2) ≤ L.nd)
    (hrep : κ * (Fintype.card R : ℝ) ^ 2 * L.nd * (L.nd + 2 * T) ≤ δ * L.heavy) (hβ : 0 ≤ β)
    (hcheck : ∀ (H : DFA S R) (h : R) (U : Finset S) (b : S → ℝ),
      L.heavy / Fintype.card R ≤ Dsamp.real {v | H.state v = h} → U.card ≤ T →
      w ≤ minorityShare A H Dsamp h →
      ((μ[|pinned O U b]).prod (checkLaw Dsamp Dsf L.N L.m)).real
        {z | ¬ checkFails O H L.n L.t h z.2 z.1} ≤ β)
    (r : Fin K) :
    (learnerMeasure (μ := μ) Dsamp Dsf νs L).real
      {θ | (∀ h, ¬ (L.heavy / Fintype.card R ≤ Dsamp.real {v | (roundHyp O Dsamp schedule
            indecisionLimit α L (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1).state v = h}
          ∧ checkFails O (roundHyp O Dsamp schedule indecisionLimit α L
            (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) L.n L.t h (θ.2.1 r).2.2.2.2.2 θ.1))
        ∧ accuracy A (roundHyp O Dsamp schedule indecisionLimit α L
            (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1)
          (fun h => denoisedLabel O L.nd (hitsOf (roundHyp O Dsamp schedule indecisionLimit α L
            (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) h (θ.2.2 r h)) θ.1) Dsamp
          < 1 - w - L.heavy}
      ≤ δ + Fintype.card R * β := by
  classical
  obtain ⟨x0⟩ := nonempty_of_isProbabilityMeasure (roundLaw Dsamp Dsf νs L)
  set Chk := (Fin L.N → S) × (Fin L.m → S)
  set Tl := (Fin L.G → S) × (Fin L.Y → S) × Chk
  set F : State × Finset S × DFA S R × Harvester S → Set (Ω × (Tl × (R → ℕ → S))) := fun c =>
    {q | (∀ h, ¬ (L.heavy / Fintype.card R ≤ Dsamp.real {v | c.2.2.1.state v = h}
        ∧ checkFails O c.2.2.1 L.n L.t h q.2.1.2.2 q.1))
      ∧ accuracy A c.2.2.1 (fun h => denoisedLabel O L.nd (hitsOf c.2.2.1 h (q.2.2 h)) q.1) Dsamp
        < 1 - w - L.heavy}
  have hF : ∀ c U b, U.card ≤ T →
      (μ.prod ((tailLaw Dsamp Dsf L.G L.Y L.N L.m).prod (streamLaw (R := R) Dsamp)))
        (F c ∩ {q | q.1 ∈ pinned O U b})
        ≤ μ (pinned O U b) * ENNReal.ofReal (δ + Fintype.card R * β) := by
    intro c U b hU
    set H := c.2.2.1
    set pass : R → Set (Ω × Tl) := fun h =>
      {p | ¬ (L.heavy / Fintype.card R ≤ Dsamp.real {v | H.state v = h}
        ∧ checkFails O H L.n L.t h p.2.2.2 p.1)}
    rw [← (measurePreserving_prodAssoc μ (tailLaw Dsamp Dsf L.G L.Y L.N L.m)
      (streamLaw (R := R) Dsamp)).map_eq, MeasurableEquiv.map_apply]
    have hset : MeasurableEquiv.prodAssoc ⁻¹' (F c ∩ {q | q.1 ∈ pinned O U b})
        = {q : (Ω × Tl) × (R → ℕ → S) | (∀ h, q.1 ∈ pass h)
          ∧ accuracy A H (fun h => denoisedLabel O L.nd (hitsOf H h (q.2 h)) q.1.1) Dsamp
            < 1 - w - L.heavy} ∩ {q | q.1.1 ∈ pinned O U b} := by
      ext q
      rfl
    rw [hset]
    refine cell_le A H O Dsamp (tailLaw Dsamp Dsf L.G L.Y L.N L.m) pass U b hL hη hη₀ hw0 hγ hε hδ
      hκ hn hrep hβ hU fun h hh hw' => ?_
    have hmp : MeasurePreserving (Prod.map id fun w : Tl => w.2.2)
        ((μ[|pinned O U b]).prod (tailLaw Dsamp Dsf L.G L.Y L.N L.m))
        ((μ[|pinned O U b]).prod (checkLaw Dsamp Dsf L.N L.m)) :=
      (MeasurePreserving.id _).prod (measurePreserving_snd.comp measurePreserving_snd)
    have hpass : pass h = Prod.map id (fun w : Tl => w.2.2) ⁻¹'
        {z | ¬ checkFails O H L.n L.t h z.2 z.1} := by
      ext z
      simp only [pass, Set.mem_ofPred_eq, Set.mem_preimage, Prod.map, id, hh, true_and]
    rw [hpass, measureReal_def]
    refine le_trans (ENNReal.toReal_mono (measure_ne_top _ _)
      ((Measure.le_map_apply hmp.measurable.aemeasurable _).trans (le_of_eq (by
        rw [hmp.map_eq])))) ?_
    exact hcheck H h U b hh hU hw'
  have hsplit := round_split_le O (roundLaw Dsamp Dsf νs L)
    ((clusterLaw Dsamp Dsf L.M).prod νs) (tailLaw Dsamp Dsf L.G L.Y L.N L.m)
    (streamLaw (R := R) Dsamp) _
    (measurePreserving_checkSplit Dsamp Dsf νs L) r
    (hypStage O Dsamp L schedule indecisionLimit α x0 r)
    (hypReads O Dsamp L schedule indecisionLimit α stageReads harvestReads x0 r)
    (hypStage_readsOnly hreads hhv x0 r) (hypReads_card hTs hTh hTK hM x0 r) F _ hF
  have hR0 : (0 : ℝ) ≤ Fintype.card R := Nat.cast_nonneg _
  rw [measureReal_def]
  refine ENNReal.toReal_le_of_le_ofReal (by positivity)
    ((measure_mono fun θ hθ => ?_).trans hsplit)
  simp only [Set.mem_ofPred_eq, F, hypStage]
  rw [(history_extend (stageReads := stageReads) (harvestReads := harvestReads) x0 r θ.1 θ.2.1).1]
  exact hθ


omit [MeasurableSpace Xs] [Countable Xs] [MeasurableSingletonClass Xs] [IsProbabilityMeasure Dsf]
  [IsProbabilityMeasure Dsamp] [IsProbabilityMeasure μ] [Fintype R] in
lemma hvReads_spread {κh : ℝ}
    (hκh : ∀ F B c ω s f t, Dsamp.real {p | t ∈ harvestReads F B c ω s p f} ≤ κh)
    {hv : Harvester S} (hin : HarvIn L hv) (f : S → ℝ) (t : S) :
    Dsamp.real {p | t ∈ hvReads L harvestReads hv p f} ≤ κh := by
  classical
  simp only [hvReads, dif_pos hin]
  exact hκh _ _ _ _ _ f t

/-- The yield test at round `r` keeps a harvest of yield below `π₀`, or drops one of yield at
least `π₁`. -/
theorem yield_round_le (π₀ π₁ : ℝ) {Ts Th T : ℕ} {κh : ℝ}
    (hreads : ∀ F B c s, ReadsOnly O (fun ω => stageReads F B c ω s)
      (fun ω => (L.stage F B c ω s, L.harvest F B c ω s)))
    (hhv : ∀ F B c ω s p (f f' : S → ℝ), (∀ t ∈ harvestReads F B c ω s p f, f t = f' t) →
      harvestReads F B c ω s p f' = harvestReads F B c ω s p f
        ∧ L.harvest F B c ω s p f' = L.harvest F B c ω s p f)
    (hTs : ∀ F B c ω s, (stageReads F B c ω s).card ≤ Ts)
    (hTh : ∀ F B c ω s p f, (harvestReads F B c ω s p f).card ≤ Th)
    (hTK : K * (2 * Fintype.card (Pop K R) * L.M * (L.M + 1) + Ts
      + 2 * Fintype.card (Pop K R) * L.M * Th + L.G * (L.M + 1) + L.Y * Th
      + L.N * (L.m + 1)) ≤ T)
    (hM : 1 ≤ L.M)
    (hκh : ∀ F B c ω s f t, Dsamp.real {p | t ∈ harvestReads F B c ω s p f} ≤ κh)
    (h0 : π₀ * L.Y ≤ L.yth) (h1 : (L.yth : ℝ) ≤ π₁ * L.Y) (r : Fin K) :
    (learnerMeasure (μ := μ) Dsamp Dsf νs L).real
      {θ | (L.yth ≤ keptCount O (roundHarv O Dsamp schedule indecisionLimit α L (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) (θ.2.1 r).2.2.2.2.1 θ.1
            ∧ harvestYield O (roundHarv O Dsamp schedule indecisionLimit α L (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp < π₀)
        ∨ (¬ L.yth ≤ keptCount O (roundHarv O Dsamp schedule indecisionLimit α L (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) (θ.2.1 r).2.2.2.2.1 θ.1
            ∧ π₁ ≤ harvestYield O (roundHarv O Dsamp schedule indecisionLimit α L (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp)}
      ≤ Real.exp (-2 * L.Y * (L.yth / L.Y - π₀) ^ 2)
        + Real.exp (-2 * L.Y * (π₁ - L.yth / L.Y) ^ 2) + L.Y * (T + L.Y * Th) * κh := by
  classical
  obtain ⟨x0⟩ := nonempty_of_isProbabilityMeasure (roundLaw Dsamp Dsf νs L)
  obtain ⟨ω0⟩ := nonempty_of_isProbabilityMeasure μ
  obtain ⟨s0⟩ := nonempty_of_isProbabilityMeasure νs
  have hκh0 : 0 ≤ κh := le_trans measureReal_nonneg
    (hκh ∅ ⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩ ((fun _ => 1, fun _ _ => 1), fun _ _ => 1)
      ω0 s0 (fun _ => 0) 1)
  set bnd := Real.exp (-2 * L.Y * (L.yth / L.Y - π₀) ^ 2)
    + Real.exp (-2 * L.Y * (π₁ - L.yth / L.Y) ^ 2) + L.Y * (T + L.Y * Th) * κh with hbnd_def
  have hbnd : 0 ≤ bnd := by positivity
  set Tl := (Fin L.G → S) × (Fin L.Y → S) × ((Fin L.N → S) × (Fin L.m → S))
  set F : State × Finset S × DFA S R × Harvester S → Set (Ω × (Tl × (R → ℕ → S))) := fun c =>
    {q | HarvIn L c.2.2.2 ∧ ((L.yth ≤ keptCount O c.2.2.2 q.2.1.2.1 q.1
        ∧ harvestYield O c.2.2.2 Dsamp < π₀)
      ∨ (¬ L.yth ≤ keptCount O c.2.2.2 q.2.1.2.1 q.1 ∧ π₁ ≤ harvestYield O c.2.2.2 Dsamp))}
  have hF : ∀ c U b, U.card ≤ T →
      (μ.prod ((tailLaw Dsamp Dsf L.G L.Y L.N L.m).prod (streamLaw (R := R) Dsamp)))
        (F c ∩ {q | q.1 ∈ pinned O U b}) ≤ μ (pinned O U b) * ENNReal.ofReal bnd := by
    intro c U b hU
    by_cases hin : HarvIn L c.2.2.2
    swap
    · rw [show F c = ∅ from Set.eq_empty_of_forall_notMem fun q hq => hin hq.1,
        Set.empty_inter, measure_empty]
      exact zero_le
    set Z : Set (Ω × (Fin L.Y → S)) :=
      {q | (L.yth ≤ keptCount O c.2.2.2 q.2 q.1 ∧ harvestYield O c.2.2.2 Dsamp < π₀)
          ∨ (¬ L.yth ≤ keptCount O c.2.2.2 q.2 q.1 ∧ π₁ ≤ harvestYield O c.2.2.2 Dsamp)}
    have hmp : MeasurePreserving (Prod.map id fun w : Tl × (R → ℕ → S) => w.1.2.1)
        (μ.prod ((tailLaw Dsamp Dsf L.G L.Y L.N L.m).prod (streamLaw (R := R) Dsamp)))
        (μ.prod (Measure.pi fun _ : Fin L.Y => Dsamp)) :=
      (MeasurePreserving.id μ).prod
        ((measurePreserving_fst.comp measurePreserving_snd).comp measurePreserving_fst)
    have hsub : F c ∩ {q | q.1 ∈ pinned O U b}
        ⊆ Prod.map id (fun w : Tl × (R → ℕ → S) => w.1.2.1) ⁻¹'
          (Z ∩ {q | q.1 ∈ pinned O U b}) := fun q hq => ⟨hq.1.2, hq.2⟩
    refine (measure_mono hsub).trans
      ((Measure.le_map_apply hmp.measurable.aemeasurable _).trans ?_)
    rw [hmp.map_eq]
    refine (yield_cell_le O Dsamp c.2.2.2
      (fun p f f' h => hvReads_congr L harvestReads hhv hin p h)
      (card_hvReads_le L harvestReads hTh c.2.2.2)
      (hvReads_spread Dsamp L harvestReads hκh hin) L.Y L.yth h0 h1 U b).trans ?_
    have hUT : (U.card : ℝ) ≤ T := by exact_mod_cast hU
    rw [hbnd_def]
    gcongr
  have hsplit := round_split_le O (roundLaw Dsamp Dsf νs L)
    ((clusterLaw Dsamp Dsf L.M).prod νs) (tailLaw Dsamp Dsf L.G L.Y L.N L.m)
    (streamLaw (R := R) Dsamp) _
    (measurePreserving_checkSplit Dsamp Dsf νs L) r
    (hypStage O Dsamp L schedule indecisionLimit α x0 r)
    (hypReads O Dsamp L schedule indecisionLimit α stageReads harvestReads x0 r)
    (hypStage_readsOnly hreads hhv x0 r) (hypReads_card hTs hTh hTK hM x0 r) F _ hF
  rw [measureReal_def]
  refine ENNReal.toReal_le_of_le_ofReal hbnd ((measure_mono fun θ hθ => ?_).trans hsplit)
  simp only [Set.mem_ofPred_eq, F, hypStage]
  rw [(history_extend (stageReads := stageReads) (harvestReads := harvestReads) x0 r θ.1
    θ.2.1).1]
  exact ⟨⟨(_, _, _, _, _), rfl⟩, hθ⟩

/-! ## The rounds' outputs, as `Termination` takes them -/

omit [IsProbabilityMeasure μ] [MeasurableSpace Xs] [Countable Xs] [MeasurableSingletonClass Xs]
  [IsProbabilityMeasure Dsamp] [IsProbabilityMeasure Dsf] in
lemma outAt_heavy (H0 : DFA S R) (r : ℕ) (ω : Ω)
    (d : Fin K → RoundDraws S K R Xs L.M L.G L.Y L.N L.m)
    (h : R) (hh : h ∈ (outAt O Dsamp schedule indecisionLimit α L H0 r ω d).2.2.1) :
    L.heavy / Fintype.card R
      ≤ Dsamp.real {v | (outAt O Dsamp schedule indecisionLimit α L H0 r ω d).1.state v = h} := by
  classical
  unfold outAt at hh ⊢
  split_ifs at hh ⊢ with hr
  · rw [round_eq] at hh ⊢
    exact (Finset.mem_filter.1 hh).2.1
  · simp at hh

omit [IsProbabilityMeasure μ] [MeasurableSpace Xs] [Countable Xs] [MeasurableSingletonClass Xs]
  [IsProbabilityMeasure Dsamp] [IsProbabilityMeasure Dsf] in
/-- Round `r`'s harvest is kept exactly when the yield test passes it. -/
lemma outAt_kept (H0 : DFA S R) (r : ℕ) (hr : r < K) (ω : Ω)
    (d : Fin K → RoundDraws S K R Xs L.M L.G L.Y L.N L.m) (hv : Harvester S)
    (h : (outAt O Dsamp schedule indecisionLimit α L H0 r ω d).2.2.2 = some hv) :
    hv = roundHarv O Dsamp schedule indecisionLimit α L
        (history O Dsamp schedule indecisionLimit α L ω d r) ω ((d ⟨r, hr⟩).1, (d ⟨r, hr⟩).2.1)
        (d ⟨r, hr⟩).2.2.1
      ∧ L.yth ≤ keptCount O hv (d ⟨r, hr⟩).2.2.2.2.1 ω := by
  classical
  unfold outAt at h
  rw [dif_pos hr, round_eq] at h
  simp only at h
  split_ifs at h with hk
  obtain rfl := (Option.some.inj h).symm
  exact ⟨rfl, hk⟩

omit [IsProbabilityMeasure μ] [MeasurableSpace Xs] [Countable Xs] [MeasurableSingletonClass Xs]
  [IsProbabilityMeasure Dsamp] [IsProbabilityMeasure Dsf] in
/-- A pool of round `r` is one of its populations' laws, while the harvests kept have positive
yield. -/
lemma pools_subset (H0 : DFA S R) (d : Fin K → RoundDraws S K R Xs L.M L.G L.Y L.N L.m) (ω : Ω)
    (y : Fin K → R → ℕ → S) (r : ℕ) (hr : r ≤ K)
    (hy : ∀ o ∈ history O Dsamp schedule indecisionLimit α L ω d r, ∀ hv, o.2.2.2 = some hv →
      0 < harvestYield O hv Dsamp) {D' : Measure S}
    (hD : D' ∈ poolsAt (Finset.univ : Finset Unit) (fun _ => Dsamp) Dsamp L.heavy
      (fun i θ => (outAt O Dsamp schedule indecisionLimit α L H0 i θ.1 θ.2.1).1)
      (fun i θ => match (outAt O Dsamp schedule indecisionLimit α L H0 i θ.1 θ.2.1).2.2.2 with
        | some hv => harvestLaw O hv Dsamp
        | none => Dsamp)
      (fun i θ => ((outAt O Dsamp schedule indecisionLimit α L H0 i θ.1 θ.2.1).2.2.2).isSome)
      r (ω, d, y)) :
    ∃ j ∈ populationsAfter (K := K) Dsamp L.heavy
        (history O Dsamp schedule indecisionLimit α L ω d r),
      D' = popMeasure O Dsamp (history O Dsamp schedule indecisionLimit α L ω d r) j := by
  classical
  rcases hD with ⟨_, _, rfl⟩ | ⟨i, hi, h, hh, rfl⟩ | ⟨i, hi, hk, rfl⟩
  · exact ⟨none, Finset.mem_filter.2 ⟨Finset.mem_univ _, trivial⟩, rfl⟩
  · have hiK : i < K := by omega
    have hget := history_getElem? O Dsamp schedule indecisionLimit α L H0 ω d r hr i
    rw [if_pos hi] at hget
    refine ⟨some (⟨i, hiK⟩, some h), Finset.mem_filter.2 ⟨Finset.mem_univ _, ?_⟩, ?_⟩
    · exact ⟨_, hget, hh⟩
    · simp [popMeasure, hget]
  · have hiK : i < K := by omega
    have hget := history_getElem? O Dsamp schedule indecisionLimit α L H0 ω d r hr i
    rw [if_pos hi] at hget
    obtain ⟨hv, hhv⟩ := Option.isSome_iff_exists.1 hk
    refine ⟨some (⟨i, hiK⟩, none), Finset.mem_filter.2 ⟨Finset.mem_univ _, ?_⟩, ?_⟩
    · exact ⟨_, hget, hk⟩
    · have hypos := hy _ (List.mem_of_getElem? hget) hv hhv
      simp only [popMeasure, hget]
      generalize outAt O Dsamp schedule indecisionLimit α L H0 i ω d = o at hhv hypos ⊢
      obtain ⟨H1, g1, f1, v1⟩ := o
      simp only at hhv
      subst hhv
      simp only [if_pos hypos]

end Specs

end LearnerProof

end OrthoDFA

universe u₁ u₂ u₃ u₄ u₅

namespace OrthoDFA

open MeasureTheory ProbabilityTheory LearnerProof

set_option maxHeartbeats 1600000 in
-- Matching each round's event against its spec's unfolds the rounds many times over.
/-- `CheckGuarantee` at the universes the learner lives in. -/
theorem learner_correct_of_check :
    CheckGuarantee.{u₁, u₂, u₃, u₄} → LearnerCorrect.{u₁, u₂, u₃, u₄, u₅} := by
  intro hcheck Ω _ μ _ S _ Q R _ _ A O Pre Suf K η₀ indecisionLimit εcov α δc pAP tolerance
    hL hη hη₀ hflat hpAP hind hind1 hα hα1 hεcov hεcov1 hδc hδc1 htol
  classical
  choose capOf hcapOf hbadOf using fun (pops : Finset (Pop K R))
    (hpops : pops.Nonempty) => clustering_quality_bad (μ := μ) A O pops Pre Suf (δ := δc) hL hη
      hη₀ hpops hflat hpAP hind hind1 hα hα1 hεcov hεcov1 hδc htol
  set capF : Finset (Pop K R) → ℝ := fun pops =>
    if h : pops.Nonempty then capOf pops h else 1 with hcapFdef
  have hcapF : ∀ pops, 0 < capF pops := fun pops => by
    simp only [hcapFdef]
    split_ifs with h
    · exact hcapOf pops h
    · exact one_pos
  refine ⟨Finset.univ.inf' Finset.univ_nonempty capF,
    (Finset.lt_inf'_iff _).2 fun pops _ => hcapF pops, ?_⟩
  intro Dsamp Dsf ε κ hDsamp hDsf hPre hSuf hκ hpAPD hε hε1 hcapR hcapSf
  have hcapLe : ∀ pops (h : pops.Nonempty),
      Finset.univ.inf' Finset.univ_nonempty capF ≤ capOf pops h := fun pops h => by
    refine (Finset.inf'_le capF (Finset.mem_univ pops)).trans (le_of_eq ?_)
    simp only [hcapFdef, dif_pos h]
  obtain ⟨sched, hsched⟩ : ∃ sched : Finset (Pop K R) → Finset State,
      sched = fun pops => stoppable η₀ pops indecisionLimit εcov (δc / 2) α pAP
        (κ * Fintype.card R / ε) (collisionMass Dsf) := ⟨_, rfl⟩
  refine ⟨sched, ?_⟩
  intro Xs _ _ _ νs L stageReads harvestReads P Ts Th ζ₀ ζ₁ ζ₂ x₀ wₛ w δs δ κf κh κa π₀ π₁ θh
    hνs hheavy hκf hschedB hP hPπ hreads hTs hhv hTh hκh hκa hout hπ₀ hy0 hy1 hκaπ hθC hθU J T
    hstage hδs0 hg0 hg2 hζ₂ hζ₁ hcC hcU hgC hgU hK ht hNn ht2 hw hγ hδ hnd hrep repeats β
  subst hheavy
  obtain ⟨ω0⟩ := nonempty_of_isProbabilityMeasure μ
  obtain ⟨s0⟩ := nonempty_of_isProbabilityMeasure νs
  set H0 : DFA S R := L.stage ∅ ⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩
    ((fun _ => 1, fun _ _ => 1), fun _ _ => 1) ω0 s0
  have hR1 : 1 ≤ Fintype.card R := Fintype.card_pos_iff.2 ⟨H0.start⟩
  have hQ1 : 1 ≤ Fintype.card Q := Fintype.card_pos_iff.2 ⟨A.start⟩
  have hRr : (1 : ℝ) ≤ Fintype.card R := by exact_mod_cast hR1
  have hQr : (1 : ℝ) ≤ Fintype.card Q := by exact_mod_cast hQ1
  have hKr : (2 * Fintype.card Q + 1 : ℝ) ≤ K := by exact_mod_cast hK
  have hκ0 : 0 ≤ κ := le_trans measureReal_nonneg (hκ 1)
  have hκf0 : 0 ≤ κf := le_trans measureReal_nonneg (hκf 1)
  have hκh0 : 0 ≤ κh := le_trans measureReal_nonneg
    (hκh ∅ ⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩ ((fun _ => 1, fun _ _ => 1), fun _ _ => 1)
      ω0 s0 (fun _ => 0) 1)
  have hws : 0 ≤ wₛ := by
    have h1 : 0 < (wₛ - (ζ₂ + ζ₁) * Fintype.card R / L.heavy) / Fintype.card Q :=
      lt_of_le_of_lt (by positivity) hcC
    have h2 : 0 < wₛ - (ζ₂ + ζ₁) * Fintype.card R / L.heavy :=
      (div_pos_iff_of_pos_right (by linarith)).1 h1
    have h3 : 0 ≤ (ζ₂ + ζ₁) * Fintype.card R / L.heavy := by
      have : 0 ≤ ζ₂ + ζ₁ := by linarith
      positivity
    linarith
  have hpAP1 : pAP ≤ 1 := hpAPD.trans measureReal_le_one
  have hrep0 : 0 ≤ repeats := by positivity
  have hβ0 : 0 ≤ β := by
    have : 0 ≤ (1 - pAP) ^ L.m := pow_nonneg (by linarith) _
    positivity
  set tail := Real.exp (-2 * (L.M * L.heavy / Fintype.card R - P) ^ 2 / L.M)
  set tailH := Real.exp (-2 * (L.M * π₀ - P) ^ 2 / L.M)
  set eLow := Real.exp (-2 * L.Y * (L.yth / L.Y - π₀) ^ 2)
  set eHigh := Real.exp (-2 * L.Y * (π₁ - L.yth / L.Y) ^ 2)
  set ovY := L.Y * (T + L.Y * Th) * κh
  set cpin := 2 * J * L.M * (L.M + 1) * T * (κ * Fintype.card R / L.heavy)
  set cov := 2 * J * L.M * (T + 2 * J * L.M * (L.M + 1) + 2 * J * L.M * Th) * κh
  set cfr := 2 * J * L.M * (L.M + 1) * Th * κf
  set e1 := Real.exp (-2 * L.G * (L.gth / L.G - ζ₀) ^ 2)
  set e2 := Real.exp (-2 * L.G * (ζ₂ - L.gth / L.G) ^ 2)
  set Mk := (tolerance + T * (L.M + 1) * κ) / ζ₁
  set δa := (Fintype.card R : ℝ)
    * (L.m * Real.exp (-2 * L.t ^ 2 / L.n) + 2 * L.n * wₛ + repeats)
  set δc' := δc + 2 * J * tail + 2 * J * tailH + cpin + cov + cfr with hδc'
  have hδc0 : 0 ≤ δc := hδc.le
  have hcpin : 0 ≤ cpin := by positivity
  have hcov : 0 ≤ cov := by positivity
  have hcfr : 0 ≤ cfr := by positivity
  have htail0 : 0 ≤ tail := (Real.exp_pos _).le
  have htailH0 : 0 ≤ tailH := (Real.exp_pos _).le
  have heLow : 0 ≤ eLow := (Real.exp_pos _).le
  have heHigh : 0 ≤ eHigh := (Real.exp_pos _).le
  have hovY : 0 ≤ ovY := by positivity
  have hδc'0 : 0 ≤ δc' := by positivity
  have hδa0 : 0 ≤ δa := by positivity
  have he1 : 0 ≤ e1 := (Real.exp_pos _).le
  have he2 : 0 ≤ e2 := (Real.exp_pos _).le
  have hMk : 0 ≤ Mk := div_nonneg (by positivity) hζ₁.le
  have hJ0 : (0 : ℝ) ≤ J := Nat.cast_nonneg _
  have hgoal : ∀ x : ℝ, x ≤ K * (eLow + eHigh + ovY)
        + (2 * Fintype.card Q + 1) * (δc' + (e2 + Mk) + δs + e1 + δa)
        + K * (δ + Fintype.card R * β) →
      x ≤ K * (δc + 2 * J * tail + 2 * J * tailH + eLow + cpin + cov + cfr + δs + e1 + eHigh
        + ovY + e2 + Mk + δa + δ + Fintype.card R * β) := by
    intro x hx
    have : (2 * Fintype.card Q + 1) * (δc' + (e2 + Mk) + δs + e1 + δa)
        ≤ K * (δc' + (e2 + Mk) + δs + e1 + δa) :=
      mul_le_mul_of_nonneg_right hKr (by positivity)
    calc x ≤ K * (eLow + eHigh + ovY) + K * (δc' + (e2 + Mk) + δs + e1 + δa)
          + K * (δ + Fintype.card R * β) := by linarith
      _ = _ := by rw [hδc']; ring
  have := hDsamp
  have := hDsf
  have : IsProbabilityMeasure (learnerMeasure (μ := μ) Dsamp Dsf νs L) := by
    rw [learnerMeasure_eq]; infer_instance
  by_cases hM0 : L.M = 0
  · refine measureReal_le_one.trans ?_
    have htail : tail = 1 := by simp [tail, hM0]
    have hJ : (1 : ℝ) ≤ J := by
      exact_mod_cast Fintype.card_pos_iff.2 ⟨(none : Pop K R)⟩
    have hK1 : (1 : ℝ) ≤ K := le_trans (by linarith) hKr
    have hpos : 0 ≤ δc + 2 * J * tail + 2 * J * tailH + eLow + cpin + cov + cfr + δs + e1
        + eHigh + ovY + e2 + Mk + δa + δ + Fintype.card R * β := by positivity
    have hbig : 1 ≤ δc + 2 * J * tail + 2 * J * tailH + eLow + cpin + cov + cfr + δs + e1
        + eHigh + ovY + e2 + Mk + δa + δ + Fintype.card R * β := by
      have : 0 ≤ (Fintype.card R : ℝ) * β := mul_nonneg (Nat.cast_nonneg _) hβ0
      have : 0 ≤ 2 * (J : ℝ) * tailH := by positivity
      rw [htail]
      linarith
    exact hbig.trans (le_mul_of_one_le_left hpos hK1)
  have hM1 : 1 ≤ L.M := Nat.one_le_iff_ne_zero.2 hM0
  have hPM : P ≤ L.M := by
    have h1 : (L.M : ℝ) * L.heavy / Fintype.card R ≤ L.M := by
      rw [div_le_iff₀ (by linarith)]
      exact mul_le_mul_of_nonneg_left (hε1.trans hRr) (Nat.cast_nonneg _)
    exact_mod_cast hP.trans h1
  have hclus : ∀ past : List (Outcome S R), PastIn L past → GoodYield O Dsamp π₀ past →
      (runMeasure μ (popMeasure (K := K) O Dsamp past) Dsf).real {x | ¬ QualityGood A O
        (populationsAfter Dsamp L.heavy past) (popMeasure O Dsamp past)
        (sched (populationsAfter Dsamp L.heavy past)) indecisionLimit α tolerance εcov x}
        ≤ δc := by
    intro past hpast hgood
    have hok : ∀ o ∈ past, ∀ hv, o.2.2.2 = some hv →
        HvOK Dsamp Pre Th κh κa hv (hvReads L harvestReads hv) := fun o ho hv hov =>
      hvOK_of_harvIn Dsamp L harvestReads hhv hTh hκh hκa hout (hpast o ho hv hov)
    have hpops : (populationsAfter (K := K) Dsamp L.heavy past).Nonempty :=
      ⟨none, Finset.mem_filter.2 ⟨Finset.mem_univ _, trivial⟩⟩
    have hprob := isProbabilityMeasure_popMeasure (K := K) O Dsamp past
      (fun o ho hv hov => ⟨_, (hok o ho hv hov).1⟩)
    have hatom := popMeasure_atom_le (K := K) O Dsamp past hε hε1 hRr hκ hπ₀ hκaπ
      fun o ho hv hov => ⟨hgood o ho hv hov, ⟨_, (hok o ho hv hov).1⟩, (hok o ho hv hov).2.2.2.1⟩
    rw [hsched]
    exact hbadOf _ hpops (popMeasure O Dsamp past) Dsf hprob hDsf
      (fun j _ => popMeasure_compl_eq_zero O Dsamp past hPre
        (fun o ho hv hov p f a h => ((hok o ho hv hov).2.2.2.2 p f a h).1) j) hSuf
      (hpAPD.trans (measureReal_mono (fun v hv => hv.2) (measure_ne_top _ _)))
      (κ * Fintype.card R / L.heavy)
      (fun j hj => by have := hprob j; exact collisionMass_le _ (hatom j hj))
      (hcapR.trans (hcapLe _ hpops)) (hcapSf.trans (hcapLe _ hpops))
  -- Each round's events, as its lemma bounds them.
  let RDs := RoundDraws S K R Xs L.M L.G L.Y L.N L.m
  let past : (Ω × (Fin K → RDs) × (Fin K → R → ℕ → S)) → Fin K → List (Outcome S R) :=
    fun θ r => history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r
  let YieldE : Fin K → Set (Ω × (Fin K → RDs) × (Fin K → R → ℕ → S)) := fun r =>
    {θ | (L.yth ≤ keptCount O (roundHarv O Dsamp sched indecisionLimit α L
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) (θ.2.1 r).2.2.2.2.1 θ.1
          ∧ harvestYield O (roundHarv O Dsamp sched indecisionLimit α L
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp < π₀)
      ∨ (¬ L.yth ≤ keptCount O (roundHarv O Dsamp sched indecisionLimit α L
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) (θ.2.1 r).2.2.2.2.1 θ.1
          ∧ π₁ ≤ harvestYield O (roundHarv O Dsamp sched indecisionLimit α L
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp)}
  have hY : ∀ r, (learnerMeasure (μ := μ) Dsamp Dsf νs L).real (YieldE r)
      ≤ eLow + eHigh + ovY := fun r =>
    yield_round_le O Dsamp Dsf νs L sched indecisionLimit α stageReads harvestReads π₀ π₁
      hreads hhv hTs hTh le_rfl hM1 hκh hy0 hy1 r
  -- Off every round's yield test going wrong, the harvests kept keep at least `π₀`.
  have hGY : ∀ θ : Ω × (Fin K → RDs) × (Fin K → R → ℕ → S), (∀ r, θ ∉ YieldE r) →
      ∀ r ≤ K, GoodYield O Dsamp π₀ (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) := by
    intro θ hg r hr o ho hv hov
    obtain ⟨i, hi⟩ := List.mem_iff_getElem?.1 ho
    rw [history_getElem? O Dsamp sched indecisionLimit α L H0 θ.1 θ.2.1 r hr] at hi
    split_ifs at hi with hir
    obtain rfl := Option.some.inj hi
    have hiK : i < K := by omega
    obtain ⟨rfl, hk⟩ := outAt_kept O Dsamp L sched indecisionLimit α H0 i hiK θ.1 θ.2.1 hv hov
    by_contra hlt
    exact hg ⟨i, hiK⟩ (Or.inl ⟨hk, not_le.1 hlt⟩)
  -- The rounds' other events.
  let BadE : Fin K → Set (Ω × (Fin K → RDs) × (Fin K → R → ℕ → S)) := fun r =>
    {θ | GoodYield O Dsamp π₀ (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r)
        ∧ ¬ QualityGood A O
          (populationsAfter Dsamp L.heavy
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r))
          (popMeasure O Dsamp (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r))
          (sched (populationsAfter Dsamp L.heavy
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r)))
          indecisionLimit α tolerance εcov
          ((θ.1, clusterDraws O θ.1 (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r)
            ((θ.2.1 r).1, (θ.2.1 r).2.1)) : Run Ω S (Pop K R))}
    ∪ {θ | roundGate O Dsamp sched indecisionLimit α L
          (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1 (θ.2.1 r).2.2.2.1
        ∧ ζ₂ + ζ₁ < mislabelledWellCut A O tolerance
          (roundState O Dsamp L.heavy sched indecisionLimit α
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1))
          (roundFamily O Dsamp L.heavy sched indecisionLimit α
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1))
          (roundHyp O Dsamp sched indecisionLimit α L
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp}
    ∪ {θ | badlyCut O tolerance
          (roundState O Dsamp L.heavy sched indecisionLimit α
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1))
          (roundFamily O Dsamp L.heavy sched indecisionLimit α
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1)) Dsamp < x₀
        ∧ ζ₀ < cutDisagreement O
          (roundState O Dsamp L.heavy sched indecisionLimit α
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1))
          (roundFamily O Dsamp L.heavy sched indecisionLimit α
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1))
          (roundHyp O Dsamp sched indecisionLimit α L
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp θ.1
        ∧ ¬ (π₁ ≤ harvestYield O (roundHarv O Dsamp sched indecisionLimit α L
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp
          ∧ ∃ qq, (∃ p, A.state p = qq ∧ tolerance < undecidedProb O
              (roundState O Dsamp L.heavy sched indecisionLimit α
                (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
                ((θ.2.1 r).1, (θ.2.1 r).2.1)).lo
              (roundState O Dsamp L.heavy sched indecisionLimit α
                (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
                ((θ.2.1 r).1, (θ.2.1 r).2.1)).hi
              (roundFamily O Dsamp L.heavy sched indecisionLimit α
                (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
                ((θ.2.1 r).1, (θ.2.1 r).2.1)) p)
            ∧ θh < stateMass A (harvestLaw O (roundHarv O Dsamp sched indecisionLimit α L
              (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
              ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp) qq)}
    ∪ {θ | ¬ roundGate O Dsamp sched indecisionLimit α L
          (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1 (θ.2.1 r).2.2.2.1
        ∧ cutDisagreement O
          (roundState O Dsamp L.heavy sched indecisionLimit α
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1))
          (roundFamily O Dsamp L.heavy sched indecisionLimit α
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1))
          (roundHyp O Dsamp sched indecisionLimit α L
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp θ.1 ≤ ζ₀}
    ∪ {θ | ∃ h, L.heavy / Fintype.card R ≤ Dsamp.real {v | (roundHyp O Dsamp sched
          indecisionLimit α L (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1).state v = h}
        ∧ checkFails O (roundHyp O Dsamp sched indecisionLimit α L
          (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) L.n L.t h (θ.2.1 r).2.2.2.2.2 θ.1
        ∧ minorityShare A (roundHyp O Dsamp sched indecisionLimit α L
          (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp h < wₛ}
  have hBad : ∀ r, (learnerMeasure (μ := μ) Dsamp Dsf νs L).real (BadE r)
      ≤ δc' + (e2 + Mk) + δs + e1 + δa := by
    intro r
    have h1 := cluster_round_le A O Dsamp Dsf νs L sched indecisionLimit α stageReads
      harvestReads tolerance εcov (Pre := Pre) hreads hhv hTs hTh le_rfl hM1 hPM hschedB hε hε1
      hRr hκ hP hPπ hπ₀ hκh hκa hout hκaπ hκf hclus r
    have h2 := pass_round_le A O Dsamp Dsf νs L sched indecisionLimit α stageReads harvestReads
      tolerance ζ₁ ζ₂ hreads hhv hTs hTh le_rfl hM1 hL hκ htol.le hζ₁ hg2 r
    have h3 := stage_round_le A O Dsamp Dsf νs L sched indecisionLimit α stageReads harvestReads
      tolerance ζ₀ x₀ π₁ θh δs hreads hhv hTs hTh le_rfl hM1 hstage hδs0 r
    have h4 := refuse_round_le O Dsamp Dsf νs L sched indecisionLimit α stageReads harvestReads
      ζ₀ hreads hhv hTs hTh le_rfl hM1 hg0 r
    have h5 := check_round_le A O Dsamp Dsf νs L sched indecisionLimit α stageReads harvestReads
      wₛ hreads hhv hTs hTh le_rfl hM1 hκ hε hws
      (fun H h U b hh hU => (hcheck A O Dsamp Dsf Pre Suf H h U b L.N L.m L.n T η₀ L.heavy κ L.t
        pAP w hDsamp hDsf hL hflat hPre hSuf hκ hε hh hU ht).1) r
    refine (measureReal_union_le _ _).trans ?_
    refine (add_le_add (measureReal_union_le _ _) le_rfl).trans ?_
    refine (add_le_add (add_le_add (measureReal_union_le _ _) le_rfl) le_rfl).trans ?_
    refine (add_le_add (add_le_add (add_le_add (measureReal_union_le _ _) le_rfl) le_rfl)
      le_rfl).trans ?_
    rw [hδc']
    linarith
  -- A round that returns an inaccurate hypothesis.
  let RetE : Fin K → Set (Ω × (Fin K → RDs) × (Fin K → R → ℕ → S)) := fun r =>
    {θ | ∃ o ∈ (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 K)[r.val]?,
      o.2.1 ∧ o.2.2.1 = ∅ ∧ accuracy A o.1
        (fun h => denoisedLabel O L.nd (hitsOf o.1 h (θ.2.2 r h)) θ.1) Dsamp < 1 - w - L.heavy}
  have hRet : ∀ r : Fin K, (learnerMeasure (μ := μ) Dsamp Dsf νs L).real (RetE r)
      ≤ δ + Fintype.card R * β := by
    intro r
    refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) (return_round_le A O Dsamp Dsf νs L
      sched indecisionLimit α stageReads harvestReads w δ β hreads hhv hTs hTh le_rfl hM1 hL hη
      hη₀ hw hγ hε hδ hκ hnd hrep hβ0 (fun H h U b hh hU hw' => (hcheck A O Dsamp Dsf Pre Suf H h
        U b L.N L.m L.n T η₀ L.heavy κ L.t pAP w hDsamp hDsf hL hflat hPre hSuf hκ hε hh hU
        ht).2 hη hη₀ hpAPD hw' hNn ht2) r)
    intro θ hθ
    obtain ⟨o, ho, -, ho2, hacc⟩ := hθ
    rw [history_getElem? O Dsamp sched indecisionLimit α L H0 θ.1 θ.2.1 K le_rfl, if_pos r.isLt,
      Option.mem_def, Option.some.injEq] at ho
    subst ho
    have hout' : outAt O Dsamp sched indecisionLimit α L H0 r θ.1 θ.2.1
        = round O Dsamp sched indecisionLimit α L
          (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1 (θ.2.1 r) := dif_pos r.isLt
    rw [hout', LearnerProof.round_eq] at ho2 hacc
    refine ⟨fun h hh => ?_, hacc⟩
    exact (Finset.filter_eq_empty_iff.1 ho2) (Finset.mem_univ h) hh
  -- `Termination`'s view of the rounds.
  set out : ℕ → Ω × (Fin K → RDs) × (Fin K → R → ℕ → S) → Outcome S R := fun r θ =>
    outAt O Dsamp sched indecisionLimit α L H0 r θ.1 θ.2.1 with hout
  set fam : ℕ → Ω × (Fin K → RDs) × (Fin K → R → ℕ → S) → State × Finset S := fun r θ =>
    if h : r < K then
      (roundState O Dsamp L.heavy sched indecisionLimit α
          (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 ⟨r, h⟩).1, (θ.2.1 ⟨r, h⟩).2.1),
        roundFamily O Dsamp L.heavy sched indecisionLimit α
          (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 ⟨r, h⟩).1, (θ.2.1 ⟨r, h⟩).2.1))
    else (⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩, ∅) with hfam
  have hout_lt : ∀ r (hr : r < K) θ, out r θ = round O Dsamp sched indecisionLimit α L
      (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1 (θ.2.1 ⟨r, hr⟩) :=
    fun r hr θ => dif_pos hr
  have hfam_lt : ∀ r (hr : r < K) θ, fam r θ = (roundState O Dsamp L.heavy sched indecisionLimit α
        (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
        ((θ.2.1 ⟨r, hr⟩).1, (θ.2.1 ⟨r, hr⟩).2.1),
      roundFamily O Dsamp L.heavy sched indecisionLimit α
        (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
        ((θ.2.1 ⟨r, hr⟩).1, (θ.2.1 ⟨r, hr⟩).2.1)) :=
    fun r hr θ => dif_pos hr
  set Rng := Finset.univ.filter fun r : Fin K => r.val < 2 * Fintype.card Q + 1
  refine hgoal _ ?_
  refine le_trans (measureReal_mono (s₂ := (⋃ r : Fin K, YieldE r) ∪ (⋃ r ∈ Rng, BadE r)
    ∪ ⋃ r : Fin K, RetE r) ?_ (measure_ne_top _ _)) ?_
  · intro θ hθ
    simp only [Set.mem_ofPred_eq] at hθ
    by_cases hg : ∃ r, θ ∈ YieldE r
    · exact Or.inl (Or.inl (Set.mem_iUnion.2 hg))
    push Not at hg
    rcases hθ with hall | ⟨r, hr⟩
    swap
    · exact Or.inr (Set.mem_iUnion.2 ⟨r, hr⟩)
    refine Or.inl (Or.inr ?_)
    have hall' : ∀ r < 2 * Fintype.card Q + 1, ¬ (out r θ).2.1 ∨ (out r θ).2.2.1.Nonempty := by
      intro r hr
      have hrK : r < K := by omega
      have := history_getElem? O Dsamp sched indecisionLimit α L H0 θ.1 θ.2.1 K le_rfl r
      rw [if_pos hrK] at this
      have hno := hall _ (List.mem_of_getElem? this)
      by_cases hg' : (out r θ).2.1
      · exact Or.inr (Finset.nonempty_iff_ne_empty.2 fun h => hno ⟨hg', h⟩)
      · exact Or.inl hg'
    obtain ⟨r, hr, h⟩ := exists_bad_round A O Dsamp (Finset.univ : Finset Unit) (fun _ => Dsamp)
      fam (fun r θ => (out r θ).1) (fun r θ => (out r θ).2.1) (fun r θ => (out r θ).2.2.1)
      (fun r θ => match (out r θ).2.2.2 with
        | some hv => harvestLaw O hv Dsamp
        | none => Dsamp)
      (fun r θ => ((out r θ).2.2.2).isSome) (tolerance := tolerance) (εcov := εcov)
      (indecisionLimit := indecisionLimit) (ε := L.heavy) (ζ := ζ₂ + ζ₁) (x₀ := x₀) (θh := θh)
      (wₛ := wₛ) hDsamp hε (by linarith) htol hεcov.le hcC hcU hgC hgU hθC hθU
      (fun r θ h hh => outAt_heavy O Dsamp L sched indecisionLimit α H0 r θ.1 θ.2.1 h hh) θ hall'
    have hrK : r < K := by omega
    refine Set.mem_biUnion (Finset.mem_filter.2 ⟨Finset.mem_univ ⟨r, hrK⟩, hr⟩) ?_
    have hgy := hGY θ hg r hrK.le
    have hro := hout_lt r hrK θ
    rw [LearnerProof.round_eq] at hro
    rcases h with hC | hS | hG | hA
    · refine Or.inl (Or.inl (Or.inl (Or.inl ⟨hgy, fun hgood => ?_⟩)))
      obtain ⟨q, hq⟩ := hC
      set pst := history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r
      set e : ClusterPart S K R L.M := ((θ.2.1 ⟨r, hrK⟩).1, (θ.2.1 ⟨r, hrK⟩).2.1)
      obtain ⟨B, hBs, hret⟩ := hgood.1
      have hspec := Classical.epsilon_spec
        (p := fun B => B ∈ sched (populationsAfter Dsamp L.heavy pst)
        ∧ ((θ.1, clusterDraws O θ.1 pst e) : Run Ω S (Pop K R))
          ∈ ret O.mq (populationsAfter Dsamp L.heavy pst) indecisionLimit α B) ⟨B, hBs, hret⟩
      have hgq := hgood.2 _ hspec.1 hspec.2 q
      rw [hfam_lt r hrK] at hq
      have hy : ∀ o ∈ pst, ∀ hv, o.2.2.2 = some hv → 0 < harvestYield O hv Dsamp :=
        fun o ho hv hov => lt_of_lt_of_le hπ₀ (hgy o ho hv hov)
      rcases hq with ⟨⟨D', hD', hm⟩, p, hp, hbad⟩ | ⟨⟨D', hD', hm⟩, p, hp, hbad⟩
      · obtain ⟨j, hj, rfl⟩ := pools_subset O Dsamp L sched indecisionLimit α H0 θ.2.1 θ.1 θ.2.2 r
          hrK.le hy hD'
        exact absurd (hgq.1 ⟨j, hj, hm⟩ p hp) (not_le.2 hbad)
      · obtain ⟨j, hj, rfl⟩ := pools_subset O Dsamp L sched indecisionLimit α H0 θ.2.1 θ.1 θ.2.2 r
          hrK.le hy hD'
        exact absurd (hgq.2 ⟨j, hj, hm⟩ p hp) (not_le.2 hbad)
    · refine Or.inl (Or.inl (Or.inl (Or.inr ?_)))
      simp only [hfam_lt r hrK, hro] at hS
      exact hS
    · simp only [hfam_lt r hrK, hro] at hG
      obtain ⟨hgate, hbc, hnk⟩ := hG
      by_cases hD : ζ₀ < cutDisagreement O
          (roundState O Dsamp L.heavy sched indecisionLimit α
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 ⟨r, hrK⟩).1, (θ.2.1 ⟨r, hrK⟩).2.1))
          (roundFamily O Dsamp L.heavy sched indecisionLimit α
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 ⟨r, hrK⟩).1, (θ.2.1 ⟨r, hrK⟩).2.1))
          (roundHyp O Dsamp sched indecisionLimit α L
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 ⟨r, hrK⟩).1, (θ.2.1 ⟨r, hrK⟩).2.1) (θ.2.1 ⟨r, hrK⟩).2.2.1) Dsamp θ.1
      · refine Or.inl (Or.inl (Or.inr ⟨hbc, hD, fun ⟨hy1', qq, hqq, hm⟩ => hnk ?_⟩))
        have hk : L.yth ≤ keptCount O (roundHarv O Dsamp sched indecisionLimit α L
            (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 ⟨r, hrK⟩).1, (θ.2.1 ⟨r, hrK⟩).2.1) (θ.2.1 ⟨r, hrK⟩).2.2.1)
            (θ.2.1 ⟨r, hrK⟩).2.2.2.2.1 θ.1 := by
          by_contra hk
          exact hg ⟨r, hrK⟩ (Or.inr ⟨hk, hy1'⟩)
        simp only [if_pos hk, Option.isSome_some, true_and]
        exact ⟨qq, hqq, hm⟩
      · exact Or.inl (Or.inr ⟨hgate, not_lt.1 hD⟩)
    · refine Or.inr ?_
      obtain ⟨h, hh, hmin⟩ := hA
      simp only [hro, Finset.mem_filter] at hh hmin
      exact ⟨h, hh.2.1, hh.2.2, hmin⟩
  refine (measureReal_union_le _ _).trans (add_le_add ((measureReal_union_le _ _).trans
    (add_le_add ?_ ?_)) ?_)
  · refine (measureReal_iUnion_fintype_le _).trans ?_
    calc ∑ r : Fin K, (learnerMeasure (μ := μ) Dsamp Dsf νs L).real (YieldE r)
        ≤ ∑ _r : Fin K, (eLow + eHigh + ovY) := Finset.sum_le_sum fun r _ => hY r
      _ = K * (eLow + eHigh + ovY) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  · refine (measureReal_biUnion_finset_le _ _).trans ?_
    calc ∑ r ∈ Rng, (learnerMeasure (μ := μ) Dsamp Dsf νs L).real (BadE r)
        ≤ ∑ _r ∈ Rng, (δc' + (e2 + Mk) + δs + e1 + δa) := Finset.sum_le_sum fun r _ => hBad r
      _ = Rng.card * (δc' + (e2 + Mk) + δs + e1 + δa) := by
          rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ (2 * Fintype.card Q + 1) * (δc' + (e2 + Mk) + δs + e1 + δa) := by
          have : (Rng.card : ℝ) ≤ 2 * Fintype.card Q + 1 := by
            exact_mod_cast card_fin_lt_le (2 * Fintype.card Q + 1)
          exact mul_le_mul_of_nonneg_right this (by positivity)
  · refine (measureReal_iUnion_fintype_le _).trans ?_
    calc ∑ r : Fin K, (learnerMeasure (μ := μ) Dsamp Dsf νs L).real (RetE r)
        ≤ ∑ _r : Fin K, (δ + Fintype.card R * β) := Finset.sum_le_sum fun r _ => hRet r
      _ = K * (δ + Fintype.card R * β) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end OrthoDFA
