import OrthoDFA.Check
import OrthoDFA.Proofs.Termination
import OrthoDFA.Proofs.LearnerClustering

/-!
# The learner is correct: the proof

`Termination` bounds the chance every round fails, given each round's three specs, and
`ReturnAccuracy`'s cell bound the chance a passing round returns an inaccurate hypothesis.  Each
round's bound conditions on the earlier rounds' draws and on the noise at every string read so
far (`readsUpTo`); on such a cell the populations, and so the clustering's laws, are fixed.

* The clustering's spec is `cluster_cell_le` against `clustering_quality_bad`.
* The stage's spec is its premise, with the family it was handed fixed by the cell.
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

omit [Fintype R] in
lemma popMeasure_compl_eq_zero {K : ℕ} (Dsamp : Measure S) (past : List (DFA S R × Finset R))
    {Pre : Set S} (h0 : Dsamp Preᶜ = 0) (j : Option (Fin K × R)) :
    popMeasure Dsamp past j Preᶜ = 0 := by
  rcases j with _ | ⟨i, h⟩
  · exact h0
  · simp only [popMeasure]
    cases past[i.val]? with
    | none => exact h0
    | some o => exact reaching_compl_eq_zero o.1 Dsamp h0 h

/-- The failed states are heavy. -/
def Heavy {K : ℕ} (Dsamp : Measure S) (ε : ℝ) (past : List (DFA S R × Finset R)) : Prop :=
  ∀ (i : Fin K) (h : R) o, past[i.val]? = some o → h ∈ o.2 →
    ε / Fintype.card R ≤ Dsamp.real {v | o.1.state v = h}

lemma popMeasure_atom_le {K : ℕ} (Dsamp : Measure S) [IsProbabilityMeasure Dsamp]
    (past : List (DFA S R × Finset R)) {ε κ : ℝ} (hε : 0 < ε) (hε1 : ε ≤ 1)
    (hR : (1 : ℝ) ≤ Fintype.card R) (hκ : ∀ a, Dsamp.real {a} ≤ κ)
    (hheavy : Heavy (K := K) Dsamp ε past) :
    ∀ j ∈ populationsAfter (K := K) past, ∀ a,
      (popMeasure Dsamp past j).real {a} ≤ κ * Fintype.card R / ε := by
  classical
  intro j hj a
  have hκ0 : 0 ≤ κ := le_trans measureReal_nonneg (hκ a)
  have hR0 : (0 : ℝ) < Fintype.card R := by linarith
  rcases j with _ | ⟨i, h⟩
  · calc Dsamp.real {a} ≤ κ := hκ a
      _ = κ * 1 := (mul_one κ).symm
      _ ≤ κ * (Fintype.card R / ε) := by
          gcongr
          rw [le_div_iff₀ hε]; linarith
      _ = κ * Fintype.card R / ε := by ring
  · obtain ⟨o, ho, hho⟩ := (Finset.mem_filter.1 hj).2
    have ho' : past[i.val]? = some o := by simpa using ho
    have hpm : popMeasure Dsamp past (some (i, h)) = reaching o.1 Dsamp h := by
      simp [popMeasure, ho']
    rw [hpm]
    set m := Dsamp.real {v | o.1.state v = h}
    have hm : ε / Fintype.card R ≤ m := hheavy i h o ho' hho
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

noncomputable def roundLaw (Dsamp Dsf : Measure S) (νs : Measure Xs) (L : Learner Ω S R Xs K) :
    Measure (RoundDraws S K R Xs L.M L.N L.m) :=
  (Measure.pi fun _ : Fin L.M => Dsf).prod
    (((Measure.pi fun _ : Option (Fin K × R) => Measure.pi fun _ : Fin L.M => Dsamp).prod
        (Measure.pi fun _ : Option (Fin K × R) => Measure.pi fun _ : Fin L.M => Dsamp)).prod
      (νs.prod (checkLaw Dsamp Dsf L.N L.m)))

noncomputable def streamLaw (Dsamp : Measure S) : Measure (R → ℕ → S) :=
  Measure.pi fun _ : R => Measure.infinitePi fun _ : ℕ => Dsamp

variable (Dsamp Dsf : Measure S) [IsProbabilityMeasure Dsamp] [IsProbabilityMeasure Dsf]
  (νs : Measure Xs) [IsProbabilityMeasure νs] (L : Learner Ω S R Xs K)

instance isProbabilityMeasure_checkLaw (N m : ℕ) :
    IsProbabilityMeasure (checkLaw Dsamp Dsf N m) := by
  unfold checkLaw; infer_instance

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
      (fun d : RoundDraws S K R Xs L.M L.N L.m => ((d.1, d.2.1) : ClusterPart S K R L.M))
      (roundLaw Dsamp Dsf νs L) (clusterLaw Dsamp Dsf L.M) :=
  (MeasurePreserving.id _).prod measurePreserving_fst

omit [MeasurableSpace Ω] [IsProbabilityMeasure Dsamp] [IsProbabilityMeasure Dsf] in
/-- The clustering's draws, against the stage's and the check's. -/
lemma measurePreserving_clusterSplit :
    MeasurePreserving (MeasurableEquiv.prodAssoc.symm :
        RoundDraws S K R Xs L.M L.N L.m → ClusterPart S K R L.M × (Xs × _))
      (roundLaw Dsamp Dsf νs L)
      ((clusterLaw Dsamp Dsf L.M).prod (νs.prod (checkLaw Dsamp Dsf L.N L.m))) :=
  MeasurePreserving.symm _ (measurePreserving_prodAssoc _ _ _)

omit [MeasurableSpace Ω] [IsProbabilityMeasure Dsamp] [IsProbabilityMeasure Dsf] in
/-- The clustering's and the stage's draws, against the check's. -/
lemma measurePreserving_checkSplit :
    MeasurePreserving (fun d : RoundDraws S K R Xs L.M L.N L.m =>
        MeasurableEquiv.prodAssoc.symm (MeasurableEquiv.prodAssoc.symm d))
      (roundLaw Dsamp Dsf νs L)
      (((clusterLaw Dsamp Dsf L.M).prod νs).prod (checkLaw Dsamp Dsf L.N L.m)) :=
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

section Specs

variable {Q : Type*} {K : ℕ} {Xs : Type*} [MeasurableSpace Xs] [Countable Xs]
  [MeasurableSingletonClass Xs]
  (A : DFA S Q) (O : Oracle μ S) (Dsamp Dsf : Measure S) [IsProbabilityMeasure Dsamp]
  [IsProbabilityMeasure Dsf] (νs : Measure Xs) [IsProbabilityMeasure νs] (L : Learner Ω S R Xs K)
  (schedule : Finset (Option (Fin K × R)) → Finset State) (indecisionLimit α : ℝ)
  (stageReads : Finset S → State → ClusterDraws S K R → Ω → Xs → Finset S)

/-- The clustering's spec at round `r`, on the rounds whose failed states were heavy. -/
theorem cluster_round_le (tolerance εcov : ℝ) {P Ts T : ℕ} {ε κ δc : ℝ}
    (hreads : ∀ F B c s, ReadsOnly O (fun ω => stageReads F B c ω s) (fun ω => L.stage F B c ω s))
    (hTs : ∀ F B c ω s, (stageReads F B c ω s).card ≤ Ts)
    (hTK : K * (2 * Fintype.card (Option (Fin K × R)) * L.M * (L.M + 1) + Ts + L.N * (L.m + 1))
      ≤ T)
    (hM : 1 ≤ L.M) (hR : 1 ≤ Fintype.card R) (hPM : P ≤ L.M)
    (hsched : ∀ pops, ∀ B ∈ schedule pops, B.npref ≤ P ∧ B.nsuff ≤ L.M)
    (hε : 0 < ε) (hε1 : ε ≤ 1) (hκ : ∀ a, Dsamp.real {a} ≤ κ)
    (hP : (P : ℝ) ≤ L.M * ε / Fintype.card R)
    (hclus : ∀ past : List (DFA S R × Finset R), Heavy (K := K) Dsamp ε past →
      (runMeasure μ (popMeasure (K := K) Dsamp past) Dsf).real {x | ¬ QualityGood A O
        (populationsAfter past) (popMeasure Dsamp past) (schedule (populationsAfter past))
        indecisionLimit α tolerance εcov x} ≤ δc)
    (r : Fin K) :
    (learnerMeasure (μ := μ) Dsamp Dsf νs L).real
      {θ | Heavy (K := K) Dsamp ε (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r)
        ∧ ¬ QualityGood A O
          (populationsAfter (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r))
          (popMeasure Dsamp (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r))
          (schedule (populationsAfter (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r)))
          indecisionLimit α tolerance εcov
          ((θ.1, clusterDraws (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r)
            ((θ.2.1 r).1, (θ.2.1 r).2.1)) : Run Ω S (Option (Fin K × R)))}
      ≤ δc + 2 * Fintype.card (Option (Fin K × R))
          * Real.exp (-2 * (L.M * ε / Fintype.card R - P) ^ 2 / L.M)
        + 2 * Fintype.card (Option (Fin K × R)) * L.M * (L.M + 1) * T
          * (κ * Fintype.card R / ε) := by
  classical
  have hκ0 : 0 ≤ κ := le_trans measureReal_nonneg (hκ 1)
  have hR0 : (0 : ℝ) < Fintype.card R := by exact_mod_cast hR
  have hδc0 : 0 ≤ δc := le_trans measureReal_nonneg (hclus [] fun i h o ho => by simp at ho)
  set δc' := δc + 2 * Fintype.card (Option (Fin K × R))
      * Real.exp (-2 * (L.M * ε / Fintype.card R - P) ^ 2 / L.M)
    + 2 * Fintype.card (Option (Fin K × R)) * L.M * (L.M + 1) * T * (κ * Fintype.card R / ε)
    with hδc'
  have hδc'0 : 0 ≤ δc' := by positivity
  set F : List (DFA S R × Finset R) →
      Set ((Ω × RoundDraws S K R Xs L.M L.N L.m) × (R → ℕ → S)) := fun c =>
    {q | Heavy (K := K) Dsamp ε c ∧ ¬ QualityGood A O (populationsAfter c) (popMeasure Dsamp c)
      (schedule (populationsAfter c)) indecisionLimit α tolerance εcov
      ((q.1.1, clusterDraws c (q.1.2.1, q.1.2.2.1)) : Run Ω S (Option (Fin K × R)))}
  have hF : ∀ c U b, U.card ≤ T →
      (((μ.prod (roundLaw Dsamp Dsf νs L)).prod (streamLaw Dsamp))
        (F c ∩ {q | q.1.1 ∈ pinned O U b})) ≤ μ (pinned O U b) * ENNReal.ofReal δc' := by
    intro c U b hU
    by_cases hc : Heavy (K := K) Dsamp ε c
    swap
    · rw [show F c = ∅ from Set.eq_empty_of_forall_notMem fun q hq => hc hq.1,
        Set.empty_inter, measure_empty]
      exact zero_le
    set E : Set (Ω × ClusterPart S K R L.M) := {q | ¬ QualityGood A O (populationsAfter c)
      (popMeasure Dsamp c) (schedule (populationsAfter c)) indecisionLimit α tolerance εcov
      ((q.1, clusterDraws c q.2) : Run Ω S (Option (Fin K × R)))}
    have hset : F c ∩ {q | q.1.1 ∈ pinned O U b}
        = ((Prod.map id (fun d : RoundDraws S K R Xs L.M L.N L.m =>
            ((d.1, d.2.1) : ClusterPart S K R L.M))) ⁻¹' E
          ∩ (pinned O U b) ×ˢ Set.univ) ×ˢ Set.univ := by
      ext q
      simp only [F, E, hc, true_and, Set.mem_inter_iff, Set.mem_ofPred_eq, Set.mem_prod,
        Set.mem_preimage, Prod.map, id, Set.mem_univ, and_true]
    rw [hset, Measure.prod_prod, measure_univ, mul_one,
      prod_restrict_apply _ (measurableSet_pinned O U b)]
    have hmp := (MeasurePreserving.id (μ.restrict (pinned O U b))).prod
      (measurePreserving_clusterPart Dsamp Dsf νs L)
    refine (Measure.le_map_apply hmp.measurable.aemeasurable E).trans ?_
    rw [hmp.map_eq]
    refine (cluster_cell_le A O Dsamp Dsf c (schedule (populationsAfter c)) indecisionLimit α
      tolerance εcov hM hPM (hsched _) hε hκ0 hc hP
      (popMeasure_atom_le Dsamp c hε hε1 (by exact_mod_cast hR) hκ hc) (hclus c hc) U b).trans ?_
    have hUT : (U.card : ℝ) ≤ T := by exact_mod_cast hU
    rw [hδc']
    gcongr
  have hloop := loop_round_le O (roundLaw Dsamp Dsf νs L) (streamLaw Dsamp) r
    (fun ω x => history O Dsamp schedule indecisionLimit α L ω x r)
    (fun ω x => readsUpTo O Dsamp schedule indecisionLimit α L stageReads ω x r)
    (fun ω x x' hx => history_dep O Dsamp schedule indecisionLimit α L stageReads ω x x' r
      fun i hi => hx i hi)
    (fun x ω ω' h => history_congr hreads x r h)
    (fun ω x => (card_readsUpTo_le hTs ω x hM hR r r.isLt.le).trans
      ((Nat.mul_le_mul_right _ r.isLt.le).trans hTK)) F _ hF
  rw [measureReal_def]
  exact ENNReal.toReal_le_of_le_ofReal hδc'0 hloop

variable {O Dsamp Dsf νs L schedule indecisionLimit α stageReads} in
omit [MeasurableSpace Xs] [Countable Xs] [MeasurableSingletonClass Xs] [IsProbabilityMeasure Dsamp]
  [IsProbabilityMeasure μ] in
lemma history_extend (x0 : RoundDraws S K R Xs L.M L.N L.m) (r : Fin K) (ω : Ω)
    (x : Fin K → RoundDraws S K R Xs L.M L.N L.m) :
    history O Dsamp schedule indecisionLimit α L ω
        (extendDraws x0 r fun i : {i : Fin K // i < r} => x i) r
      = history O Dsamp schedule indecisionLimit α L ω x r
    ∧ readsUpTo O Dsamp schedule indecisionLimit α L stageReads ω
        (extendDraws x0 r fun i : {i : Fin K // i < r} => x i) r
      = readsUpTo O Dsamp schedule indecisionLimit α L stageReads ω x r :=
  history_dep O Dsamp schedule indecisionLimit α L stageReads ω _ _ r fun i hi => by
    simp [extendDraws, show i < r from hi]

/-- The stage's spec at round `r`: the family it is handed is fixed by the cell. -/
theorem stage_round_le (tolerance ζ δs : ℝ) {Ts T : ℕ}
    (hreads : ∀ F B c s, ReadsOnly O (fun ω => stageReads F B c ω s) (fun ω => L.stage F B c ω s))
    (hTs : ∀ F B c ω s, (stageReads F B c ω s).card ≤ Ts)
    (hTK : K * (2 * Fintype.card (Option (Fin K × R)) * L.M * (L.M + 1) + Ts + L.N * (L.m + 1))
      ≤ T)
    (hM : 1 ≤ L.M) (hR : 1 ≤ Fintype.card R)
    (hstage : ∀ F B c (U : Finset S) b, U.card ≤ T →
      ((μ[|pinned O U b]).prod νs).real
          {q | ζ < mislabelledWellCut A O tolerance B F (L.stage F B c q.1 q.2) Dsamp}
        ≤ δs)
    (r : Fin K) :
    (learnerMeasure (μ := μ) Dsamp Dsf νs L).real
      {θ | ζ < mislabelledWellCut A O tolerance
        (roundState O schedule indecisionLimit α
          (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 r).1, (θ.2.1 r).2.1))
        (roundFamily O schedule indecisionLimit α
          (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 r).1, (θ.2.1 r).2.1))
        (roundHyp O schedule indecisionLimit α L
          (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp} ≤ δs := by
  classical
  obtain ⟨x0⟩ := nonempty_of_isProbabilityMeasure (roundLaw Dsamp Dsf νs L)
  have hδs0 : 0 ≤ δs := le_trans measureReal_nonneg (hstage ∅ ⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩
    ((fun _ => 1, fun _ _ => 1), fun _ _ => 1) ∅ (fun _ => 0) (by simp))
  set hist : Ω → ({i : Fin K // i < r} → RoundDraws S K R Xs L.M L.N L.m) →
      List (DFA S R × Finset R) := fun ω a =>
    history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r a) r
  set stage : Ω → ({i : Fin K // i < r} → RoundDraws S K R Xs L.M L.N L.m)
      × ClusterPart S K R L.M → State × Finset S × ClusterDraws S K R := fun ω ae =>
    (roundState O schedule indecisionLimit α (hist ω ae.1) ω ae.2,
      roundFamily O schedule indecisionLimit α (hist ω ae.1) ω ae.2,
      clusterDraws (hist ω ae.1) ae.2)
  set queried : Ω → ({i : Fin K // i < r} → RoundDraws S K R Xs L.M L.N L.m)
      × ClusterPart S K R L.M → Finset S := fun ω ae =>
    readsUpTo O Dsamp schedule indecisionLimit α L stageReads ω (extendDraws x0 r ae.1) r
      ∪ clusterReads (hist ω ae.1) ae.2
  have hreads' : ∀ ae, ReadsOnly O (fun ω => queried ω ae) (fun ω => stage ω ae) := by
    rintro ⟨a, e⟩ ω ω' h
    obtain ⟨h1, h2⟩ := history_congr hreads (extendDraws x0 r a) r
      fun w hw => h w (Finset.mem_union_left _ hw)
    have hc : ∀ w ∈ clusterReads (hist ω a) e, O.noise w ω = O.noise w ω' :=
      fun w hw => h w (Finset.mem_union_right _ hw)
    have h3 : hist ω' a = hist ω a := h2
    refine ⟨?_, ?_⟩
    · change _ ∪ clusterReads (hist ω' a) e = _
      rw [h1, h3]
    · change (roundState O schedule indecisionLimit α (hist ω' a) ω' e, _, _) = _
      rw [h3, roundState_congr O schedule indecisionLimit α _ e hc,
        roundFamily_congr O schedule indecisionLimit α _ e hc]
  have hT : ∀ ω ae, (queried ω ae).card ≤ T := by
    rintro ω ⟨a, e⟩
    have hlen := length_history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r a) r
    rw [min_eq_left r.isLt.le] at hlen
    refine (Finset.card_union_le _ _).trans ?_
    refine (Nat.add_le_add (card_readsUpTo_le hTs ω _ hM hR r r.isLt.le)
      (card_clusterReads_le_share _ e hM hR (by rw [hlen]; exact r.isLt))).trans ?_
    exact le_trans (add_share_le r.isLt (by omega)) hTK
  set F : State × Finset S × ClusterDraws S K R →
      Set (Ω × ((Xs × ((Fin L.N → S) × (Fin L.m → S))) × (R → ℕ → S))) := fun c =>
    {q | ζ < mislabelledWellCut A O tolerance c.1 c.2.1 (L.stage c.2.1 c.1 c.2.2 q.1 q.2.1.1) Dsamp}
  have hF : ∀ c U b, U.card ≤ T →
      (μ.prod ((νs.prod (checkLaw Dsamp Dsf L.N L.m)).prod (streamLaw Dsamp)))
        (F c ∩ {q | q.1 ∈ pinned O U b}) ≤ μ (pinned O U b) * ENNReal.ofReal δs := by
    intro c U b hU
    have hmp : MeasurePreserving (Prod.map id fun w : (Xs × ((Fin L.N → S) × (Fin L.m → S)))
        × (R → ℕ → S) => w.1.1)
        (μ.prod ((νs.prod (checkLaw Dsamp Dsf L.N L.m)).prod (streamLaw Dsamp))) (μ.prod νs) :=
      (MeasurePreserving.id μ).prod (measurePreserving_fst.comp measurePreserving_fst)
    set G : Set (Ω × Xs) :=
      {p | ζ < mislabelledWellCut A O tolerance c.1 c.2.1 (L.stage c.2.1 c.1 c.2.2 p.1 p.2) Dsamp}
    have hset : F c ∩ {q | q.1 ∈ pinned O U b}
        = Prod.map id (fun w : (Xs × ((Fin L.N → S) × (Fin L.m → S))) × (R → ℕ → S) => w.1.1)
          ⁻¹' (G ∩ (pinned O U b) ×ˢ Set.univ) := by
      ext q
      simp [F, G]
    rw [hset]
    refine (Measure.le_map_apply hmp.measurable.aemeasurable _).trans ?_
    rw [hmp.map_eq, prod_restrict_apply _ (measurableSet_pinned O U b)]
    exact restrict_prod_le O U b νs G (hstage _ _ _ U b hU)
  have hsplit := round_split_le O (roundLaw Dsamp Dsf νs L) (clusterLaw Dsamp Dsf L.M)
    (νs.prod (checkLaw Dsamp Dsf L.N L.m)) (streamLaw Dsamp) _
    (measurePreserving_clusterSplit Dsamp Dsf νs L) r stage queried hreads' hT F _ hF
  rw [measureReal_def]
  refine ENNReal.toReal_le_of_le_ofReal hδs0 ((measure_mono fun θ hθ => ?_).trans hsplit)
  simp only [Set.mem_ofPred_eq, F, stage, hist]
  rw [(history_extend (stageReads := stageReads) x0 r θ.1 θ.2.1).1]
  exact hθ

/-- Round `r`'s hypothesis, from the earlier rounds' draws and its own clustering and stage
draws. -/
noncomputable def hypStage (x0 : RoundDraws S K R Xs L.M L.N L.m) (r : Fin K) (ω : Ω)
    (ae : ({i : Fin K // i < r} → RoundDraws S K R Xs L.M L.N L.m) × (ClusterPart S K R L.M × Xs)) :
    DFA S R :=
  roundHyp O schedule indecisionLimit α L
    (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r ae.1) r) ω ae.2.1 ae.2.2

/-- What round `r` has read by its check. -/
noncomputable def hypReads (x0 : RoundDraws S K R Xs L.M L.N L.m) (r : Fin K) (ω : Ω)
    (ae : ({i : Fin K // i < r} → RoundDraws S K R Xs L.M L.N L.m) × (ClusterPart S K R L.M × Xs)) :
    Finset S :=
  readsUpTo O Dsamp schedule indecisionLimit α L stageReads ω (extendDraws x0 r ae.1) r
    ∪ clusterReads (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r ae.1) r)
      ae.2.1
    ∪ stageReads
      (roundFamily O schedule indecisionLimit α
        (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r ae.1) r) ω ae.2.1)
      (roundState O schedule indecisionLimit α
        (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r ae.1) r) ω ae.2.1)
      (clusterDraws (history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r ae.1) r)
        ae.2.1) ω ae.2.2

variable {O Dsamp Dsf νs L schedule indecisionLimit α stageReads} in
omit [MeasurableSpace Xs] [Countable Xs] [MeasurableSingletonClass Xs]
  [IsProbabilityMeasure Dsamp] in
lemma hypStage_readsOnly
    (hreads : ∀ F B c s, ReadsOnly O (fun ω => stageReads F B c ω s) (fun ω => L.stage F B c ω s))
    (x0 : RoundDraws S K R Xs L.M L.N L.m) (r : Fin K) (ae) :
    ReadsOnly O (fun ω => hypReads O Dsamp L schedule indecisionLimit α stageReads x0 r ω ae)
      (fun ω => hypStage O Dsamp L schedule indecisionLimit α x0 r ω ae) := by
  intro ω ω' h
  obtain ⟨a, e, s⟩ := ae
  obtain ⟨h1, h2⟩ := history_congr hreads (extendDraws x0 r a) r
    fun w hw => h w (Finset.mem_union_left _ (Finset.mem_union_left _ hw))
  set past := history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r a) r
  have hc : ∀ w ∈ clusterReads past e, O.noise w ω = O.noise w ω' :=
    fun w hw => h w (Finset.mem_union_left _ (Finset.mem_union_right _ hw))
  have hB := roundState_congr O schedule indecisionLimit α past e hc
  have hF := roundFamily_congr O schedule indecisionLimit α past e hc
  have hs := hreads (roundFamily O schedule indecisionLimit α past ω e)
    (roundState O schedule indecisionLimit α past ω e) (clusterDraws past e) s ω ω'
    fun w hw => h w (Finset.mem_union_right _ hw)
  have hs1 : stageReads (roundFamily O schedule indecisionLimit α past ω e)
      (roundState O schedule indecisionLimit α past ω e) (clusterDraws past e) ω' s
    = stageReads (roundFamily O schedule indecisionLimit α past ω e)
      (roundState O schedule indecisionLimit α past ω e) (clusterDraws past e) ω s := hs.1
  refine ⟨?_, ?_⟩
  · simp only [hypReads]
    rw [h1, h2, hB, hF, hs1]
  · simp only [hypStage, roundHyp]
    rw [h2, hB, hF]
    exact hs.2

variable {O Dsamp Dsf νs L schedule indecisionLimit α stageReads} in
omit [MeasurableSpace Xs] [Countable Xs] [MeasurableSingletonClass Xs] [IsProbabilityMeasure Dsamp]
  [IsProbabilityMeasure μ] in
lemma hypReads_card {Ts T : ℕ} (hTs : ∀ F B c ω s, (stageReads F B c ω s).card ≤ Ts)
    (hTK : K * (2 * Fintype.card (Option (Fin K × R)) * L.M * (L.M + 1) + Ts + L.N * (L.m + 1))
      ≤ T)
    (hM : 1 ≤ L.M) (hR : 1 ≤ Fintype.card R) (x0 : RoundDraws S K R Xs L.M L.N L.m)
    (r : Fin K) (ω : Ω) (ae) :
    (hypReads O Dsamp L schedule indecisionLimit α stageReads x0 r ω ae).card ≤ T := by
  obtain ⟨a, e, s⟩ := ae
  have hlen := length_history O Dsamp schedule indecisionLimit α L ω (extendDraws x0 r a) r
  rw [min_eq_left r.isLt.le] at hlen
  refine (Finset.card_union_le _ _).trans ?_
  refine (Nat.add_le_add ((Finset.card_union_le _ _).trans (Nat.add_le_add
    (card_readsUpTo_le hTs ω _ hM hR r r.isLt.le)
    (card_clusterReads_le_share _ e hM hR (by rw [hlen]; exact r.isLt)))) (hTs _ _ _ _ _)).trans ?_
  rw [Nat.add_assoc]
  exact le_trans (add_share_le r.isLt (by omega)) hTK

/-- The check's spec at round `r`: it fails a state with a small minority rarely. -/
theorem check_round_le (wₛ : ℝ) {ε κ : ℝ} {Ts T : ℕ}
    (hreads : ∀ F B c s, ReadsOnly O (fun ω => stageReads F B c ω s) (fun ω => L.stage F B c ω s))
    (hTs : ∀ F B c ω s, (stageReads F B c ω s).card ≤ Ts)
    (hTK : K * (2 * Fintype.card (Option (Fin K × R)) * L.M * (L.M + 1) + Ts + L.N * (L.m + 1))
      ≤ T)
    (hM : 1 ≤ L.M) (hR : 1 ≤ Fintype.card R) (hκ : ∀ a, Dsamp.real {a} ≤ κ) (hε : 0 < ε)
    (hheavy : L.heavy = ε) (hw : 0 ≤ wₛ)
    (hcheck : ∀ (H : DFA S R) (h : R) (U : Finset S) (b : S → ℝ),
      ε / Fintype.card R ≤ Dsamp.real {v | H.state v = h} → U.card ≤ T →
      ((μ[|pinned O U b]).prod (checkLaw Dsamp Dsf L.N L.m)).real
          {z | checkFails O H L.n L.t h z.2 z.1}
        ≤ L.m * Real.exp (-2 * L.t ^ 2 / L.n) + 2 * L.n * minorityShare A H Dsamp h
          + κ * Fintype.card R / ε * (2 * L.n ^ 2 + 2 * L.n * (L.m + 1) * T))
    (r : Fin K) :
    (learnerMeasure (μ := μ) Dsamp Dsf νs L).real
      {θ | ∃ h, L.heavy / Fintype.card R ≤ Dsamp.real {v | (roundHyp O schedule indecisionLimit α L
          (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1).state v = h}
        ∧ checkFails O (roundHyp O schedule indecisionLimit α L
          (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) L.n L.t h (θ.2.1 r).2.2.2 θ.1
        ∧ minorityShare A (roundHyp O schedule indecisionLimit α L
          (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) Dsamp h < wₛ}
      ≤ Fintype.card R * (L.m * Real.exp (-2 * L.t ^ 2 / L.n) + 2 * L.n * wₛ
        + κ * Fintype.card R / ε * (2 * L.n ^ 2 + 2 * L.n * (L.m + 1) * T)) := by
  classical
  obtain ⟨x0⟩ := nonempty_of_isProbabilityMeasure (roundLaw Dsamp Dsf νs L)
  have hκ0 : 0 ≤ κ := le_trans measureReal_nonneg (hκ 1)
  set b0 := L.m * Real.exp (-2 * L.t ^ 2 / L.n) + 2 * L.n * wₛ
    + κ * Fintype.card R / ε * (2 * L.n ^ 2 + 2 * L.n * (L.m + 1) * T)
  have hb0 : 0 ≤ b0 := by positivity
  set Chk := (Fin L.N → S) × (Fin L.m → S)
  set F : DFA S R → Set (Ω × (Chk × (R → ℕ → S))) := fun H =>
    {q | ∃ h, L.heavy / Fintype.card R ≤ Dsamp.real {v | H.state v = h}
      ∧ checkFails O H L.n L.t h q.2.1 q.1 ∧ minorityShare A H Dsamp h < wₛ}
  have hF : ∀ H U b, U.card ≤ T →
      (μ.prod ((checkLaw Dsamp Dsf L.N L.m).prod (streamLaw Dsamp)))
        (F H ∩ {q | q.1 ∈ pinned O U b})
        ≤ μ (pinned O U b) * ENNReal.ofReal (Fintype.card R * b0) := by
    intro H U b hU
    set Bad := Finset.univ.filter fun h => L.heavy / Fintype.card R ≤ Dsamp.real {v | H.state v = h}
      ∧ minorityShare A H Dsamp h < wₛ
    set C : R → Set (Ω × Chk) := fun h => {z | checkFails O H L.n L.t h z.2 z.1}
    have hmp : MeasurePreserving (Prod.map id fun w : Chk × (R → ℕ → S) => w.1)
        (μ.prod ((checkLaw Dsamp Dsf L.N L.m).prod (streamLaw Dsamp)))
        (μ.prod (checkLaw Dsamp Dsf L.N L.m)) :=
      (MeasurePreserving.id μ).prod measurePreserving_fst
    have hsub : F H ∩ {q | q.1 ∈ pinned O U b} ⊆ ⋃ h ∈ Bad,
        Prod.map id (fun w : Chk × (R → ℕ → S) => w.1) ⁻¹' (C h ∩ (pinned O U b) ×ˢ Set.univ) := by
      rintro q ⟨⟨h, hh, hf, hm⟩, hq⟩
      exact Set.mem_biUnion (Finset.mem_filter.2 ⟨Finset.mem_univ _, hh, hm⟩)
        ⟨hf, hq, Set.mem_univ _⟩
    have hone : ∀ h ∈ Bad, (μ.prod ((checkLaw Dsamp Dsf L.N L.m).prod (streamLaw Dsamp)))
        (Prod.map id (fun w : Chk × (R → ℕ → S) => w.1) ⁻¹' (C h ∩ (pinned O U b) ×ˢ Set.univ))
        ≤ μ (pinned O U b) * ENNReal.ofReal b0 := by
      intro h hh
      obtain ⟨hheavyh, hmin⟩ := (Finset.mem_filter.1 hh).2
      rw [hheavy] at hheavyh
      refine (Measure.le_map_apply hmp.measurable.aemeasurable _).trans ?_
      rw [hmp.map_eq, prod_restrict_apply _ (measurableSet_pinned O U b)]
      refine restrict_prod_le O U b _ (C h) ?_
      refine (hcheck H h U b hheavyh hU).trans ?_
      have : 2 * (L.n : ℝ) * minorityShare A H Dsamp h ≤ 2 * L.n * wₛ :=
        mul_le_mul_of_nonneg_left hmin.le (by positivity)
      simp only [b0]
      linarith
    calc (μ.prod ((checkLaw Dsamp Dsf L.N L.m).prod (streamLaw Dsamp)))
          (F H ∩ {q | q.1 ∈ pinned O U b})
        ≤ ∑ h ∈ Bad, (μ.prod ((checkLaw Dsamp Dsf L.N L.m).prod (streamLaw Dsamp)))
          (Prod.map id (fun w : Chk × (R → ℕ → S) => w.1) ⁻¹' (C h ∩ (pinned O U b) ×ˢ Set.univ)) :=
          (measure_mono hsub).trans (measure_biUnion_finset_le _ _)
      _ ≤ ∑ _h ∈ Bad, μ (pinned O U b) * ENNReal.ofReal b0 := Finset.sum_le_sum hone
      _ ≤ ∑ _h : R, μ (pinned O U b) * ENNReal.ofReal b0 :=
          Finset.sum_le_sum_of_subset (Finset.filter_subset _ _)
      _ = μ (pinned O U b) * ENNReal.ofReal (Fintype.card R * b0) := by
          rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, ENNReal.ofReal_mul (by positivity),
            ENNReal.ofReal_natCast]
          ring
  have hsplit := round_split_le O (roundLaw Dsamp Dsf νs L)
    ((clusterLaw Dsamp Dsf L.M).prod νs) (checkLaw Dsamp Dsf L.N L.m) (streamLaw Dsamp) _
    (measurePreserving_checkSplit Dsamp Dsf νs L) r
    (hypStage O Dsamp L schedule indecisionLimit α x0 r)
    (hypReads O Dsamp L schedule indecisionLimit α stageReads x0 r)
    (hypStage_readsOnly hreads x0 r) (hypReads_card hTs hTK hM hR x0 r) F _ hF
  rw [measureReal_def]
  refine ENNReal.toReal_le_of_le_ofReal (by positivity) ((measure_mono fun θ hθ => ?_).trans hsplit)
  simp only [Set.mem_ofPred_eq, F, hypStage]
  rw [(history_extend (stageReads := stageReads) x0 r θ.1 θ.2.1).1]
  exact hθ

/-- `ReturnAccuracy`'s round at round `r`: a check passing every heavy state, but a hypothesis
inaccurate once denoised. -/
theorem return_round_le (w δ β : ℝ) {η₀ ε κ : ℝ} {Ts T : ℕ}
    (hreads : ∀ F B c s, ReadsOnly O (fun ω => stageReads F B c ω s) (fun ω => L.stage F B c ω s))
    (hTs : ∀ F B c ω s, (stageReads F B c ω s).card ≤ Ts)
    (hTK : K * (2 * Fintype.card (Option (Fin K × R)) * L.M * (L.M + 1) + Ts + L.N * (L.m + 1))
      ≤ T)
    (hM : 1 ≤ L.M) (hR : 1 ≤ Fintype.card R) (hL : O.L = {v | A.state v ∈ A.accept})
    (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hw0 : 0 ≤ w) (hγ : 0 < denoiseMargin η₀ w) (hε : 0 < ε)
    (hδ : 0 < δ) (hκ : ∀ a, Dsamp.real {a} ≤ κ)
    (hn : Real.log (2 * Fintype.card R / δ) / (2 * denoiseMargin η₀ w ^ 2) ≤ L.nd)
    (hrep : κ * (Fintype.card R : ℝ) ^ 2 * L.nd * (L.nd + 2 * T) ≤ δ * ε) (hβ : 0 ≤ β)
    (hheavy : L.heavy = ε)
    (hcheck : ∀ (H : DFA S R) (h : R) (U : Finset S) (b : S → ℝ),
      ε / Fintype.card R ≤ Dsamp.real {v | H.state v = h} → U.card ≤ T →
      w ≤ minorityShare A H Dsamp h →
      ((μ[|pinned O U b]).prod (checkLaw Dsamp Dsf L.N L.m)).real
        {z | ¬ checkFails O H L.n L.t h z.2 z.1} ≤ β)
    (r : Fin K) :
    (learnerMeasure (μ := μ) Dsamp Dsf νs L).real
      {θ | (∀ h, ¬ (L.heavy / Fintype.card R ≤ Dsamp.real {v | (roundHyp O schedule
            indecisionLimit α L (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1).state v = h}
          ∧ checkFails O (roundHyp O schedule indecisionLimit α L
            (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) L.n L.t h (θ.2.1 r).2.2.2 θ.1))
        ∧ accuracy A (roundHyp O schedule indecisionLimit α L
            (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1)
          (fun h => denoisedLabel O L.nd (hitsOf (roundHyp O schedule indecisionLimit α L
            (history O Dsamp schedule indecisionLimit α L θ.1 θ.2.1 r) θ.1
            ((θ.2.1 r).1, (θ.2.1 r).2.1) (θ.2.1 r).2.2.1) h (θ.2.2 r h)) θ.1) Dsamp
          < 1 - w - ε}
      ≤ δ + Fintype.card R * β := by
  classical
  obtain ⟨x0⟩ := nonempty_of_isProbabilityMeasure (roundLaw Dsamp Dsf νs L)
  set Chk := (Fin L.N → S) × (Fin L.m → S)
  set F : DFA S R → Set (Ω × (Chk × (R → ℕ → S))) := fun H =>
    {q | (∀ h, ¬ (L.heavy / Fintype.card R ≤ Dsamp.real {v | H.state v = h}
        ∧ checkFails O H L.n L.t h q.2.1 q.1))
      ∧ accuracy A H (fun h => denoisedLabel O L.nd (hitsOf H h (q.2.2 h)) q.1) Dsamp
        < 1 - w - ε}
  have hF : ∀ H U b, U.card ≤ T →
      (μ.prod ((checkLaw Dsamp Dsf L.N L.m).prod (streamLaw Dsamp)))
        (F H ∩ {q | q.1 ∈ pinned O U b})
        ≤ μ (pinned O U b) * ENNReal.ofReal (δ + Fintype.card R * β) := by
    intro H U b hU
    set pass : R → Set (Ω × Chk) := fun h =>
      {p | ¬ (L.heavy / Fintype.card R ≤ Dsamp.real {v | H.state v = h}
        ∧ checkFails O H L.n L.t h p.2 p.1)}
    rw [← (measurePreserving_prodAssoc μ (checkLaw Dsamp Dsf L.N L.m) (streamLaw Dsamp)).map_eq,
      MeasurableEquiv.map_apply]
    have hset : MeasurableEquiv.prodAssoc ⁻¹' (F H ∩ {q | q.1 ∈ pinned O U b})
        = {q : (Ω × Chk) × (R → ℕ → S) | (∀ h, q.1 ∈ pass h)
          ∧ accuracy A H (fun h => denoisedLabel O L.nd (hitsOf H h (q.2 h)) q.1.1) Dsamp
            < 1 - w - ε} ∩ {q | q.1.1 ∈ pinned O U b} := by
      ext q
      rfl
    rw [hset]
    refine cell_le A H O Dsamp (checkLaw Dsamp Dsf L.N L.m) pass U b hL hη hη₀ hw0 hγ hε hδ hκ
      hn hrep hβ hU fun h hh hw' => ?_
    have hpass : pass h = {z | ¬ checkFails O H L.n L.t h z.2 z.1} := by
      ext z
      simp only [pass, Set.mem_ofPred_eq, hheavy, hh, true_and]
    rw [hpass]
    exact hcheck H h U b hh hU hw'
  have hsplit := round_split_le O (roundLaw Dsamp Dsf νs L)
    ((clusterLaw Dsamp Dsf L.M).prod νs) (checkLaw Dsamp Dsf L.N L.m) (streamLaw Dsamp) _
    (measurePreserving_checkSplit Dsamp Dsf νs L) r
    (hypStage O Dsamp L schedule indecisionLimit α x0 r)
    (hypReads O Dsamp L schedule indecisionLimit α stageReads x0 r)
    (hypStage_readsOnly hreads x0 r) (hypReads_card hTs hTK hM hR x0 r) F _ hF
  have hR0 : (0 : ℝ) ≤ Fintype.card R := Nat.cast_nonneg _
  rw [measureReal_def]
  refine ENNReal.toReal_le_of_le_ofReal (by positivity)
    ((measure_mono fun θ hθ => ?_).trans hsplit)
  simp only [Set.mem_ofPred_eq, F, hypStage]
  rw [(history_extend (stageReads := stageReads) x0 r θ.1 θ.2.1).1]
  exact hθ

/-! ## The rounds' outputs, as `Termination` takes them -/

omit [IsProbabilityMeasure μ] [MeasurableSpace Xs] [Countable Xs] [MeasurableSingletonClass Xs]
  [IsProbabilityMeasure Dsamp] [IsProbabilityMeasure Dsf] in
lemma outAt_heavy (H0 : DFA S R) (r : ℕ) (ω : Ω) (d : Fin K → RoundDraws S K R Xs L.M L.N L.m)
    (h : R) (hh : h ∈ (outAt O Dsamp schedule indecisionLimit α L H0 r ω d).2) :
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
lemma history_heavy {ε : ℝ} (hheavy : L.heavy = ε) (H0 : DFA S R) (ω : Ω)
    (d : Fin K → RoundDraws S K R Xs L.M L.N L.m) (r : ℕ) (hr : r ≤ K) :
    Heavy (K := K) Dsamp ε (history O Dsamp schedule indecisionLimit α L ω d r) := by
  intro i h o ho hh
  rw [history_getElem? O Dsamp schedule indecisionLimit α L H0 ω d r hr] at ho
  split_ifs at ho
  · obtain rfl := Option.some.inj ho
    rw [← hheavy]
    exact outAt_heavy O Dsamp L schedule indecisionLimit α H0 i ω d h hh

omit [IsProbabilityMeasure μ] [MeasurableSpace Xs] [Countable Xs] [MeasurableSingletonClass Xs]
  [IsProbabilityMeasure Dsamp] [IsProbabilityMeasure Dsf] in
/-- A pool of round `r` is one of its populations' laws. -/
lemma pools_subset (H0 : DFA S R) (d : Fin K → RoundDraws S K R Xs L.M L.N L.m) (ω : Ω)
    (y : Fin K → R → ℕ → S) (r : ℕ) (hr : r ≤ K) {D' : Measure S}
    (hD : D' ∈ poolsAt (Finset.univ : Finset Unit) (fun _ => Dsamp) Dsamp
      (fun i θ => (outAt O Dsamp schedule indecisionLimit α L H0 i θ.1 θ.2.1).1)
      (fun i θ => (outAt O Dsamp schedule indecisionLimit α L H0 i θ.1 θ.2.1).2) r (ω, d, y)) :
    ∃ j ∈ populationsAfter (K := K) (history O Dsamp schedule indecisionLimit α L ω d r),
      D' = popMeasure Dsamp (history O Dsamp schedule indecisionLimit α L ω d r) j := by
  classical
  rcases hD with ⟨_, _, rfl⟩ | ⟨i, hi, h, hh, rfl⟩
  · exact ⟨none, Finset.mem_filter.2 ⟨Finset.mem_univ _, trivial⟩, rfl⟩
  · have hiK : i < K := by omega
    have hget := history_getElem? O Dsamp schedule indecisionLimit α L H0 ω d r hr i
    rw [if_pos hi] at hget
    refine ⟨some (⟨i, hiK⟩, h), Finset.mem_filter.2 ⟨Finset.mem_univ _, ?_⟩, ?_⟩
    · exact ⟨_, hget, hh⟩
    · simp [popMeasure, hget]

end Specs

end LearnerProof

end OrthoDFA

universe u₁ u₂ u₃ u₄ u₅

namespace OrthoDFA

open MeasureTheory ProbabilityTheory LearnerProof

/-- `CheckGuarantee` at the universes the learner lives in. -/
theorem learner_correct_of_check :
    CheckGuarantee.{u₁, u₂, u₃, u₄} → LearnerCorrect.{u₁, u₂, u₃, u₄, u₅} := by
  intro hcheck Ω _ μ _ S _ Q R _ _ A O Pre K η₀ indecisionLimit εcov α δc pAP tolerance
    hL hη hη₀ hflat hpAP hind hind1 hα hα1 hεcov hεcov1 hδc hδc1 htol
  classical
  choose capOf hcapOf hbadOf using fun (pops : Finset (Option (Fin K × R)))
    (hpops : pops.Nonempty) => clustering_quality_bad (μ := μ) A O pops Pre (δ := δc) hL hη hη₀
      hpops hflat hpAP hind hind1 hα hα1 hεcov hεcov1 hδc htol
  set capF : Finset (Option (Fin K × R)) → ℝ := fun pops =>
    if h : pops.Nonempty then capOf pops h else 1 with hcapFdef
  have hcapF : ∀ pops, 0 < capF pops := fun pops => by
    simp only [hcapFdef]
    split_ifs with h
    · exact hcapOf pops h
    · exact one_pos
  refine ⟨Finset.univ.inf' Finset.univ_nonempty capF,
    (Finset.lt_inf'_iff _).2 fun pops _ => hcapF pops, ?_⟩
  intro Dsamp Dsf ε κ hDsamp hDsf hPre hκ hpAPD hε hε1 hcapR hcapSf
  have hcapLe : ∀ pops (h : pops.Nonempty),
      Finset.univ.inf' Finset.univ_nonempty capF ≤ capOf pops h := fun pops h => by
    refine (Finset.inf'_le capF (Finset.mem_univ pops)).trans (le_of_eq ?_)
    simp only [hcapFdef, dif_pos h]
  obtain ⟨sched, hsched⟩ : ∃ sched : Finset (Option (Fin K × R)) → Finset State,
      sched = fun pops => stoppable η₀ pops indecisionLimit εcov (δc / 2) α pAP
        (κ * Fintype.card R / ε) (collisionMass Dsf) := ⟨_, rfl⟩
  refine ⟨sched, ?_⟩
  intro Xs _ _ _ νs L stageReads P Ts ζ wₛ w δs δ hνs hheavy hschedB hP hreads hTs T hstage hζ hcC
    hcU hK ht hNn ht2 hw hγ hδ hnd hrep repeats β
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
  have hδs0 : 0 ≤ δs := le_trans measureReal_nonneg (hstage ∅ ⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩
    ((fun _ => 1, fun _ _ => 1), fun _ _ => 1) ∅ (fun _ => 0) (by simp))
  have hws : 0 ≤ wₛ := by
    have h1 : 0 < (wₛ - ζ * Fintype.card R / ε) / Fintype.card Q :=
      lt_of_le_of_lt (by positivity) hcC
    have h2 : 0 < wₛ - ζ * Fintype.card R / ε :=
      (div_pos_iff_of_pos_right (by linarith)).1 h1
    have h3 : 0 ≤ ζ * Fintype.card R / ε := by positivity
    linarith
  have hpAP1 : pAP ≤ 1 := hpAPD.trans measureReal_le_one
  have hrep0 : 0 ≤ repeats := by positivity
  have hβ0 : 0 ≤ β := by
    have : 0 ≤ (1 - pAP) ^ L.m := pow_nonneg (by linarith) _
    positivity
  set tail := Real.exp (-2 * (L.M * ε / Fintype.card R - P) ^ 2 / L.M)
  set δc' := δc + 2 * Fintype.card (Option (Fin K × R)) * tail
    + 2 * Fintype.card (Option (Fin K × R)) * L.M * (L.M + 1) * T * (κ * Fintype.card R / ε)
  set δa := (Fintype.card R : ℝ)
    * (L.m * Real.exp (-2 * L.t ^ 2 / L.n) + 2 * L.n * wₛ + repeats)
  have hδc'0 : 0 ≤ δc' := by positivity
  have hδa0 : 0 ≤ δa := by positivity
  have hgoal : ∀ x : ℝ, x ≤ (2 * Fintype.card Q + 1) * (δc' + δs + δa)
        + K * (δ + Fintype.card R * β) →
      x ≤ K * (δc' + δs + δa + δ + Fintype.card R * β) := by
    intro x hx
    have : (2 * Fintype.card Q + 1) * (δc' + δs + δa) ≤ K * (δc' + δs + δa) :=
      mul_le_mul_of_nonneg_right hKr (by positivity)
    linarith
  have := hDsamp
  have := hDsf
  have : IsProbabilityMeasure (learnerMeasure (μ := μ) Dsamp Dsf νs L) := by
    rw [learnerMeasure_eq]; infer_instance
  by_cases hM0 : L.M = 0
  · refine measureReal_le_one.trans ?_
    have htail : tail = 1 := by simp [tail, hM0]
    have hJ : (1 : ℝ) ≤ Fintype.card (Option (Fin K × R)) := by
      exact_mod_cast Fintype.card_pos_iff.2 ⟨(none : Option (Fin K × R))⟩
    have hK1 : (1 : ℝ) ≤ K := le_trans (by linarith) hKr
    have hbig : 1 ≤ δc' + δs + δa + δ + Fintype.card R * β := by
      have : 0 ≤ 2 * (Fintype.card (Option (Fin K × R)) : ℝ) * L.M * (L.M + 1) * T
          * (κ * Fintype.card R / ε) := by
        positivity
      have : 0 ≤ (Fintype.card R : ℝ) * β := by positivity
      simp only [δc', htail]
      linarith
    have hpos : 0 ≤ δc' + δs + δa + δ + Fintype.card R * β := le_trans zero_le_one hbig
    calc (1 : ℝ) ≤ δc' + δs + δa + δ + Fintype.card R * β := hbig
      _ ≤ K * (δc' + δs + δa + δ + Fintype.card R * β) := le_mul_of_one_le_left hpos hK1
  have hM1 : 1 ≤ L.M := Nat.one_le_iff_ne_zero.2 hM0
  have hPM : P ≤ L.M := by
    have h1 : (L.M : ℝ) * ε / Fintype.card R ≤ L.M := by
      rw [div_le_iff₀ (by linarith)]
      exact mul_le_mul_of_nonneg_left (hε1.trans hRr) (Nat.cast_nonneg _)
    exact_mod_cast hP.trans h1
  have hclus : ∀ past : List (DFA S R × Finset R), Heavy (K := K) Dsamp ε past →
      (runMeasure μ (popMeasure (K := K) Dsamp past) Dsf).real {x | ¬ QualityGood A O
        (populationsAfter past) (popMeasure Dsamp past) (sched (populationsAfter past))
        indecisionLimit α tolerance εcov x} ≤ δc := by
    intro past hpast
    have hpops : (populationsAfter (K := K) past).Nonempty :=
      ⟨none, Finset.mem_filter.2 ⟨Finset.mem_univ _, trivial⟩⟩
    have hprob := isProbabilityMeasure_popMeasure (K := K) Dsamp past
    have hatom := popMeasure_atom_le Dsamp past hε hε1 hRr hκ hpast
    rw [hsched]
    exact hbadOf _ hpops (popMeasure Dsamp past) Dsf hprob hDsf
      (fun j _ => popMeasure_compl_eq_zero Dsamp past hPre j)
      (hpAPD.trans (measureReal_mono (fun v hv => hv.2) (measure_ne_top _ _)))
      (κ * Fintype.card R / ε)
      (fun j hj => by have := hprob j; exact collisionMass_le _ (hatom j hj))
      (hcapR.trans (hcapLe _ hpops)) (hcapSf.trans (hcapLe _ hpops))
  -- `Termination` over the learner's space.
  set out : ℕ → Ω × (Fin K → RoundDraws S K R Xs L.M L.N L.m) × (Fin K → R → ℕ → S) →
      DFA S R × Finset R := fun r θ =>
    outAt O Dsamp sched indecisionLimit α L H0 r θ.1 θ.2.1 with hout
  set fam : ℕ → Ω × (Fin K → RoundDraws S K R Xs L.M L.N L.m) × (Fin K → R → ℕ → S) →
      State × Finset S := fun r θ =>
    if h : r < K then
      (roundState O sched indecisionLimit α
          (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 ⟨r, h⟩).1, (θ.2.1 ⟨r, h⟩).2.1),
        roundFamily O sched indecisionLimit α
          (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 ⟨r, h⟩).1, (θ.2.1 ⟨r, h⟩).2.1))
    else (⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩, ∅) with hfam
  have hout_lt : ∀ r (hr : r < K) θ, out r θ = round O Dsamp sched indecisionLimit α L
      (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1 (θ.2.1 ⟨r, hr⟩) :=
    fun r hr θ => dif_pos hr
  have hterm := termination_le (learnerMeasure (μ := μ) Dsamp Dsf νs L) A O Dsamp
    (Finset.univ : Finset Unit) (fun _ => Dsamp) fam (fun r θ => (out r θ).1)
    (fun r θ => (out r θ).2) (tolerance := tolerance) (εcov := εcov)
    (indecisionLimit := indecisionLimit) (ε := ε) (ζ := ζ) (wₛ := wₛ) (δc := δc') (δs := δs)
    (δa := δa) hDsamp hε hζ htol hεcov.le hcC hcU ?hC ?hS ?hA ?hmass
  case hmass =>
    intro r θ h hh
    rw [← hheavy]
    exact outAt_heavy O Dsamp L sched indecisionLimit α H0 r θ.1 θ.2.1 h hh
  case hC =>
    intro r hr
    have hrK : r < K := by omega
    refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) (cluster_round_le A O Dsamp Dsf νs
      L sched indecisionLimit α stageReads tolerance εcov hreads hTs le_rfl hM1 hR1 hPM hschedB hε
      hε1 hκ hP hclus ⟨r, hrK⟩)
    intro θ hθ
    obtain ⟨q, hq⟩ := hθ
    set past := history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r
    set e : ClusterPart S K R L.M := ((θ.2.1 ⟨r, hrK⟩).1, (θ.2.1 ⟨r, hrK⟩).2.1)
    refine ⟨history_heavy O Dsamp L sched indecisionLimit α hheavy H0 θ.1 θ.2.1 r hrK.le,
      fun hgood => ?_⟩
    obtain ⟨B, hBs, hret⟩ := hgood.1
    have hspec := Classical.epsilon_spec (p := fun B => B ∈ sched (populationsAfter past)
      ∧ ((θ.1, clusterDraws past e) : Run Ω S (Option (Fin K × R)))
        ∈ ret O.mq (populationsAfter past) indecisionLimit α B) ⟨B, hBs, hret⟩
    have hg := hgood.2 _ hspec.1 hspec.2 q
    have hf : fam r θ = (roundState O sched indecisionLimit α past θ.1 e,
        roundFamily O sched indecisionLimit α past θ.1 e) := dif_pos hrK
    rw [hf] at hq
    rcases hq with ⟨⟨D', hD', hm⟩, p, hp, hbad⟩ | ⟨⟨D', hD', hm⟩, p, hp, hbad⟩
    · obtain ⟨j, hj, rfl⟩ := pools_subset O Dsamp L sched indecisionLimit α H0 θ.2.1 θ.1 θ.2.2 r
        hrK.le hD'
      exact absurd (hg.1 ⟨j, hj, hm⟩ p hp) (not_le.2 hbad)
    · obtain ⟨j, hj, rfl⟩ := pools_subset O Dsamp L sched indecisionLimit α H0 θ.2.1 θ.1 θ.2.2 r
        hrK.le hD'
      exact absurd (hg.2 ⟨j, hj, hm⟩ p hp) (not_le.2 hbad)
  case hS =>
    intro r hr
    have hrK : r < K := by omega
    refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) (stage_round_le A O Dsamp Dsf νs L
      sched indecisionLimit α stageReads tolerance ζ δs hreads hTs le_rfl hM1 hR1 hstage ⟨r, hrK⟩)
    intro θ hθ
    have hf : fam r θ = (roundState O sched indecisionLimit α
          (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 ⟨r, hrK⟩).1, (θ.2.1 ⟨r, hrK⟩).2.1),
        roundFamily O sched indecisionLimit α
          (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1
          ((θ.2.1 ⟨r, hrK⟩).1, (θ.2.1 ⟨r, hrK⟩).2.1)) := dif_pos hrK
    simp only [Set.mem_ofPred_eq, hf, hout_lt r hrK, LearnerProof.round_eq] at hθ
    exact hθ
  case hA =>
    intro r hr
    have hrK : r < K := by omega
    refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) (check_round_le A O Dsamp Dsf νs L
      sched indecisionLimit α stageReads wₛ hreads hTs le_rfl hM1 hR1 hκ hε hheavy hws
      (fun H h U b hh hU => (hcheck A O Dsamp Dsf Pre H h U b L.N L.m L.n T η₀ ε κ L.t pAP w
        hDsamp hDsf hL hflat hPre hκ hε hh hU ht).1) ⟨r, hrK⟩)
    intro θ hθ
    obtain ⟨h, hh, hmin⟩ := hθ
    simp only [hout_lt r hrK, LearnerProof.round_eq, Finset.mem_filter] at hh hmin
    exact ⟨h, hh.2.1, hh.2.2, hmin⟩
  -- A round that passes but returns an inaccurate hypothesis.
  have hRet : ∀ r : Fin K, (learnerMeasure (μ := μ) Dsamp Dsf νs L).real
      {θ | ∃ o ∈ (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 K)[r.val]?, o.2 = ∅
        ∧ accuracy A o.1 (fun h => denoisedLabel O L.nd (hitsOf o.1 h (θ.2.2 r h)) θ.1) Dsamp
          < 1 - w - ε} ≤ δ + Fintype.card R * β := by
    intro r
    refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) (return_round_le A O Dsamp Dsf νs L
      sched indecisionLimit α stageReads w δ β hreads hTs le_rfl hM1 hR1 hL hη hη₀ hw hγ hε hδ hκ
      hnd hrep hβ0 hheavy (fun H h U b hh hU hw' => (hcheck A O Dsamp Dsf Pre H h U b L.N L.m L.n
        T η₀ ε κ L.t pAP w hDsamp hDsf hL hflat hPre hκ hε hh hU ht).2 hη hη₀ hpAPD hw' hNn ht2) r)
    intro θ hθ
    obtain ⟨o, ho, ho2, hacc⟩ := hθ
    rw [history_getElem? O Dsamp sched indecisionLimit α L H0 θ.1 θ.2.1 K le_rfl, if_pos r.isLt,
      Option.mem_def, Option.some.injEq] at ho
    subst ho
    have hout : outAt O Dsamp sched indecisionLimit α L H0 r θ.1 θ.2.1
        = round O Dsamp sched indecisionLimit α L
          (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 r) θ.1 (θ.2.1 r) := dif_pos r.isLt
    rw [hout, LearnerProof.round_eq] at ho2 hacc
    refine ⟨fun h hh => ?_, hacc⟩
    exact (Finset.filter_eq_empty_iff.1 ho2) (Finset.mem_univ h) hh
  refine hgoal _ ?_
  refine le_trans (measureReal_mono (s₂ := {θ | ∀ r < 2 * Fintype.card Q + 1, (out r θ).2.Nonempty}
    ∪ ⋃ r : Fin K, {θ | ∃ o ∈ (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 K)[r.val]?,
      o.2 = ∅ ∧ accuracy A o.1 (fun h => denoisedLabel O L.nd (hitsOf o.1 h (θ.2.2 r h)) θ.1)
        Dsamp < 1 - w - ε}) ?_ (measure_ne_top _ _)) ?_
  · intro θ hθ
    simp only [Set.mem_ofPred_eq] at hθ
    rcases hθ with hall | ⟨r, hr⟩
    · refine Or.inl fun r hr => hall _ ?_
      have hrK : r < K := by omega
      have := history_getElem? O Dsamp sched indecisionLimit α L H0 θ.1 θ.2.1 K le_rfl r
      rw [if_pos hrK] at this
      exact List.mem_of_getElem? this
    · exact Or.inr (Set.mem_iUnion.2 ⟨r, hr⟩)
  refine (measureReal_union_le _ _).trans (add_le_add hterm ?_)
  refine (measureReal_iUnion_fintype_le _).trans ?_
  calc ∑ r : Fin K, (learnerMeasure (μ := μ) Dsamp Dsf νs L).real
        {θ | ∃ o ∈ (history O Dsamp sched indecisionLimit α L θ.1 θ.2.1 K)[r.val]?, o.2 = ∅
          ∧ accuracy A o.1 (fun h => denoisedLabel O L.nd (hitsOf o.1 h (θ.2.2 r h)) θ.1) Dsamp
            < 1 - w - ε}
      ≤ ∑ _r : Fin K, (δ + Fintype.card R * β) := Finset.sum_le_sum fun r _ => hRet r
    _ = K * (δ + Fintype.card R * β) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end OrthoDFA
