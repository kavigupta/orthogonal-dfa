import OrthoDFA.Proofs.LoopCheckFirst
import OrthoDFA.Proofs.ProbeAgree
import OrthoDFA.Proofs.Level3CheckSound

/-!
# The checked round's DFA, from its check and its family's cut

The checked round's tree reads `ε` at its root, so a string whose probe from `0` agrees is
classified as read by the DFA started where `ε` sifts (`ProbeAgreeCorrect`), and a decided read on
the wrong side is a seeded miscut or a seeded indecision (`misread_cut`). So the DFA misjudges at
most the strings failing the check from `0` and those the family misreads (`check_dfa_of_reads`).
The check certifies its rate over its own strings but for chance `δ/8`, whatever the reads
(`Level3CheckSound`); in the loop, given the noise and the clustering, so too
(`acc_round0_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Random

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The tree's root reads the string itself. -/
def RootOne (T : DTree α) : Prop := ∃ r a, T = .node 1 r a

omit [DecidableEq α] in
theorem rootOne_splitAt {T : DTree α} (h : RootOne T) (d : FreeMonoid α) (p : List Bool) :
    RootOne (T.splitAt d p) := by
  obtain ⟨r, a, rfl⟩ := h
  rcases p with _ | ⟨_ | _, p⟩ <;> exact ⟨_, _, rfl⟩

omit [DecidableEq α] in
theorem rootOne_of_classSet : ∀ (n : ℕ) (T : DTree α), T ∈ Random.classSet n → RootOne T
  | 0, T, h => by
    classical
    simp only [Random.classSet, Finset.mem_singleton] at h
    exact ⟨_, _, h⟩
  | n + 1, T, h => by
    classical
    simp only [Random.classSet, Finset.mem_union, Finset.mem_biUnion, Finset.mem_image] at h
    rcases h with h | ⟨T₀, hT₀, _, _, rfl⟩
    · exact rootOne_of_classSet n T h
    · exact rootOne_splitAt (rootOne_of_classSet n T₀ hT₀) _ _

theorem rootOne_check (read : FreeMonoid α → ARU) (C : Level3Settings) (K : CheckSettings)
    (ds : Fin C.probeBudget → CheckDraw α K) :
    RootOne (indecisiveLStarCheck C K read ds).hypothesis.tree := by
  obtain ⟨h1, -⟩ := runChecked_tree read C K (X := (List.ofFn ds).map Prod.fst)
    (Y := (List.ofFn ds).flatMap fun d => List.ofFn d.2) (List.ofFn ds) initialState
    (fun d hd => ⟨List.mem_map_of_mem hd, fun i => List.mem_flatMap.2 ⟨d, hd, by simp⟩⟩)
    (checkRunInv_start read C _ _)
  exact rootOne_of_classSet _ _ h1.inv.cls

omit [Fintype α] [DecidableEq α] in
/-- A string passing the check from `0` neither searches nor stops undecided at its ends. -/
theorem not_dis_of_startFails {read : FreeMonoid α → ARU} {H : HypothesisDFA α} {x : FreeMonoid α}
    (h : startFails read H.tree H.edges x = false) :
    x ∉ searchSet read 0 H ∪ endsSet read 0 H := by
  have hp : probeOrAgree read H.tree H.edges 0 x = probe read H.tree H.edges 0 x := by
    simp [probeOrAgree]
  unfold startFails at h
  simp only [Set.mem_union, searchSet, endsSet, Set.mem_ofPred_eq, hp, not_or]
  generalize probe read H.tree H.edges 0 x = o at h
  rcases o with _ | _ | _ | _ | ⟨zs, b⟩ <;> simp_all [endsString]
  rcases zs with _ | _ <;> rcases b <;> simp_all

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

omit [DecidableEq α] in
open scoped Classical in
/-- A read on the wrong side is, with the seed's vote, a miscut or undecided. -/
lemma misread_cut (O : Oracle μ (FreeMonoid α)) {F : Finset (FreeMonoid α)} (hF : 1 ∉ F)
    {lo hi : ℕ} (hlh : lo ≤ hi) (p : FreeMonoid α) (ω : Ω)
    (hω : familyRead O.mq F lo (hi + 1) p ω = if p ∈ O.L then .reject else .accept) :
    ¬ cutCorrect O lo (hi + 1) (insert 1 F) p ω ∨ ¬ decided O.mq lo (hi + 1) (insert 1 F) p ω := by
  classical
  simp only [familyRead] at hω
  simp only [cutCorrect, decided]
  have hb := voteCount_insert O.mq hF p ω
  have hb01 : (if O.mq (p * 1) ω = 1 then 1 else 0) ≤ 1 := by split_ifs <;> omega
  set v := voteCount O.mq F p ω
  set vf := voteCount O.mq (insert 1 F) p ω
  by_cases hp : p ∈ O.L
  · have hl : O.label p = 1 := by simp [Oracle.label, hp]
    rw [if_pos hp] at hω
    have hv : v ≤ lo := by
      unfold readOf at hω
      split_ifs at hω with h1 h2
      · exact h2
    by_cases hlo : vf ≤ lo
    · left
      intro hcc
      have := hcc.2 hlo
      rw [hl] at this
      norm_num at this
    · right
      omega
  · have hl : O.label p = 0 := by simp [Oracle.label, hp]
    rw [if_neg hp] at hω
    have hv : hi + 1 ≤ v := by
      unfold readOf at hω
      split_ifs at hω with h1 h2
      · exact h1
    by_cases hhi : hi + 1 < vf
    · left
      intro hcc
      have := hcc.1 hhi
      rw [hl] at this
      norm_num at this
    · right
      omega

/-- Where some string passes the check from `0`, the DFA started where `ε` sifts misjudges only
strings failing the check or misread by the family with its seed. -/
theorem check_dfa_of_reads {σ : Type*} (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) {F : Finset (FreeMonoid α)} (hF : 1 ∉ F)
    {lo hi : ℕ} (hlh : lo ≤ hi) (ω : Ω) (H : HypothesisDFA α) (hroot : RootOne H.tree)
    (hx₀ : ∃ x, startFails (fun z => familyRead O.mq F lo (hi + 1) z ω) H.tree H.edges x = false) :
    ∃ s, H.tree.sift (fun z => familyRead O.mq F lo (hi + 1) z ω) 1 = .inl s
      ∧ {x | ¬ H.Classifies s x (M.eval x.toList ∈ M.accept)}
        ⊆ {x | startFails (fun z => familyRead O.mq F lo (hi + 1) z ω) H.tree H.edges x = true}
          ∪ ({p | ¬ cutCorrect O lo (hi + 1) (insert 1 F) p ω}
            ∪ {p | ¬ decided O.mq lo (hi + 1) (insert 1 F) p ω}) := by
  classical
  set read := fun z => familyRead O.mq F lo (hi + 1) z ω
  have hpa := probeAgreeCorrect (α := α)
  obtain ⟨x₀, hx₀⟩ := hx₀
  obtain ⟨-, h0⟩ := hpa read H 0 x₀ (Nat.zero_le _) (not_dis_of_startFails hx₀)
  obtain ⟨s, hs, -⟩ := h0 rfl hroot
  refine ⟨s, hs, fun x hx => ?_⟩
  by_contra hn
  simp only [Set.mem_union, Set.mem_ofPred_eq, not_or, Bool.not_eq_true] at hn
  obtain ⟨hnd, hnc, hnu⟩ := hn
  obtain ⟨-, h1⟩ := hpa read H 0 x (Nat.zero_le _) (not_dis_of_startFails hnd)
  obtain ⟨s', hs', hdec, hcl⟩ := h1 rfl hroot
  rw [hs] at hs'
  cases hs'
  have hright : (read x = .accept ↔ M.eval x.toList ∈ M.accept) := by
    rw [← hL x]
    by_contra hw
    have hwrong : familyRead O.mq F lo (hi + 1) x ω = if x ∈ O.L then .reject else .accept := by
      show read x = _
      rcases hr : read x with _ | _ | _
      · rw [hr] at hw; split_ifs with hx' <;> simp_all
      · rw [hr] at hw; split_ifs with hx' <;> simp_all
      · exact absurd hr hdec
    rcases misread_cut O hF hlh x ω hwrong with h | h
    · exact hnc h
    · exact hnu h
  apply hx
  obtain ⟨q, hq, hqb⟩ := hcl
  exact ⟨q, hq, hqb.trans hright⟩

variable [IsProbabilityMeasure μ]

/-- A property of the noise and the checked round's result is measurable in the noise where it is
at each result. -/
theorem measurableSet_of_checkResult' (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α))
    (kl kh : ℕ) (C : Level3Settings) (K : CheckSettings) (ds : Fin C.probeBudget → CheckDraw α K)
    (Ψ : Ω → Level3Result α → Prop) (hΨ : ∀ R, MeasurableSet {ω | Ψ ω R}) :
    MeasurableSet {ω | Ψ ω (indecisiveLStarCheck C K (fun z => familyRead O.mq F kl kh z ω) ds)} := by
  classical
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  set A := readStrings (shortStrings (drawsLen K ds + C.probeBudget + 1)) F
  set res : Ω → Level3Result α := fun ω =>
    indecisiveLStarCheck C K (fun z => familyRead O.mq F kl kh z ω) ds
  have hres : ∀ ω ω', (∀ p ∈ A, (O.mq p ω = 1 ↔ O.mq p ω' = 1)) → res ω = res ω' :=
    fun ω ω' h => checkResult_congr_short C K ds fun z hz =>
      familyRead_congr O h (mem_shortStrings.2 hz)
  set rep : Finset (FreeMonoid α) → Level3Result α := fun U =>
    if h : ∃ ω, A.filter (fun p => O.mq p ω = 1) = U then res h.choose else res ω₀
  have hrepω : ∀ ω, res ω = rep (A.filter (fun p => O.mq p ω = 1)) := by
    intro ω
    have h : ∃ ω', A.filter (fun p => O.mq p ω' = 1) = A.filter (fun p => O.mq p ω = 1) :=
      ⟨ω, rfl⟩
    simp only [rep, dif_pos h]
    refine hres _ _ fun p hp => ?_
    have := Finset.ext_iff.1 h.choose_spec p
    simp only [Finset.mem_filter, hp, true_and] at this
    exact this.symm
  have hset : {ω | Ψ ω (res ω)} = ⋃ U ∈ A.powerset,
      ({ω | A.filter (fun p => O.mq p ω = 1) = U} ∩ {ω | Ψ ω (rep U)}) := by
    ext ω
    simp only [Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_inter_iff, Finset.mem_powerset,
      exists_prop]
    constructor
    · intro h
      refine ⟨_, Finset.filter_subset _ _, rfl, ?_⟩
      rw [← hrepω ω]
      exact h
    · rintro ⟨U, -, hU, h⟩
      rw [← hU, ← hrepω ω] at h
      exact h
  show MeasurableSet {ω | Ψ ω (res ω)}
  rw [hset]
  refine Finset.measurableSet_biUnion _ fun U _ => MeasurableSet.inter ?_ (hΨ _)
  exact measurableSet_of_reads O A _ fun ω ω' h (hω : A.filter _ = U) => by
    show A.filter _ = U
    rw [← hω]
    exact Finset.filter_congr fun p hp => (h p hp).symm

/-- The share of `Dc`'s strings failing the check from `0` is measurable in the noise. -/
theorem measurable_startFails_mass (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α))
    (kl kh : ℕ) (Dc : Measure (FreeMonoid α)) [IsFiniteMeasure Dc] (T : DTree α) (E : Edges α) :
    Measurable fun ω => Dc.real
      {x | startFails (fun z => familyRead O.mq F kl kh z ω) T E x = true} := by
  classical
  refine (measurable_measure_of_mem Dc _ fun x => ?_).ennreal_toReal
  refine measurableSet_of_reads O (readStrings (pot 0 x T) F) _ fun ω ω' h hω => ?_
  simp only [Set.mem_ofPred_eq] at hω ⊢
  rwa [Eps.startFails_congr (r' := fun z => familyRead O.mq F kl kh z ω) fun z hz =>
    familyRead_congr O (fun p hp => (h p hp).symm) hz]

section Acc

variable {K : ℕ} (settings : CheckLoopSettings K) (O : Oracle μ (FreeMonoid α))
  (D Dsf Dc : Measure (FreeMonoid α)) [IsProbabilityMeasure D] [IsProbabilityMeasure Dc]
  (h₀ : LoopHist α)

/-- At a stage, the noise and round uniforms where the round ends consistent at a hypothesis
failing more than the check's rate of its strings from `0`. -/
def accBad (t : State × Finset (FreeMonoid α) × Finset (FreeMonoid α) × Finset (FreeMonoid α)) :
    Set (Ω × (Fin settings.level3.probeBudget → unitInterval × (Fin settings.check.size → unitInterval))) :=
  {q | (indecisiveLStarCheck settings.level3 settings.check
        (fun z => familyRead O.mq t.2.1 t.1.lo (t.1.hi + 1) z q.1)
        (checkDrawsOf settings.check D Dc settings.avoidDepth settings.checkDepth t.2.2.1 t.2.2.2 q.2)).ending
      = .consistent
    ∧ settings.check.rate < (avoidLaw Dc settings.checkDepth t.2.2.2).real
      {x | startFails (fun z => familyRead O.mq t.2.1 t.1.lo (t.1.hi + 1) z q.1)
        (indecisiveLStarCheck settings.level3 settings.check
          (fun z => familyRead O.mq t.2.1 t.1.lo (t.1.hi + 1) z q.1)
          (checkDrawsOf settings.check D Dc settings.avoidDepth settings.checkDepth t.2.2.1 t.2.2.2
            q.2)).hypothesis.tree
        (indecisiveLStarCheck settings.level3 settings.check
          (fun z => familyRead O.mq t.2.1 t.1.lo (t.1.hi + 1) z q.1)
          (checkDrawsOf settings.check D Dc settings.avoidDepth settings.checkDepth t.2.2.1 t.2.2.2
            q.2)).hypothesis.edges x = true}}

omit [IsProbabilityMeasure D] [IsProbabilityMeasure Dc] in
theorem measurableSet_accBad
    (t : State × Finset (FreeMonoid α) × Finset (FreeMonoid α) × Finset (FreeMonoid α)) :
    MeasurableSet (accBad settings O D Dc t) := by
  classical
  set C := settings.level3
  set Kk := settings.check
  set Dc' := avoidLaw Dc settings.checkDepth t.2.2.2
  have hS : MeasurableSet {q : Ω × (Fin C.probeBudget → CheckDraw α Kk) |
      (indecisiveLStarCheck C Kk (fun z => familyRead O.mq t.2.1 t.1.lo (t.1.hi + 1) z q.1)
        q.2).ending = .consistent
      ∧ Kk.rate < Dc'.real {x | startFails (fun z => familyRead O.mq t.2.1 t.1.lo (t.1.hi + 1) z q.1)
        (indecisiveLStarCheck C Kk (fun z => familyRead O.mq t.2.1 t.1.lo (t.1.hi + 1) z q.1)
          q.2).hypothesis.tree
        (indecisiveLStarCheck C Kk (fun z => familyRead O.mq t.2.1 t.1.lo (t.1.hi + 1) z q.1)
          q.2).hypothesis.edges x = true}} := by
    refine measurableSet_prod_countable_right _ fun ds => ?_
    refine measurableSet_of_checkResult' O t.2.1 t.1.lo (t.1.hi + 1) C Kk ds
      (fun ω R => R.ending = .consistent ∧ Kk.rate < Dc'.real
        {x | startFails (fun z => familyRead O.mq t.2.1 t.1.lo (t.1.hi + 1) z ω)
          R.hypothesis.tree R.hypothesis.edges x = true}) fun R => ?_
    exact (MeasurableSet.const _).inter (measurableSet_lt measurable_const
      (measurable_startFails_mass O t.2.1 t.1.lo (t.1.hi + 1) Dc' _ _))
  exact (measurable_fst.prodMk ((measurable_checkDrawsOf Kk D Dc _ _ _ _).comp measurable_snd)) hS

/-- The first round ends consistent at a hypothesis failing more than the check's rate of its
strings from `0` with chance at most `δ/8`, whatever its family. -/
theorem acc_round0_le (hK : 0 < K) (hKδ : 0 < settings.check.δ) (hr0 : 0 < settings.check.rate)
    (hr1 : settings.check.rate < 1) :
    (checkLoopMeasure μ α K settings.level3.probeBudget settings.check.size)
      {p | ∃ t, stage0 settings.toLoopSettings O D Dsf h₀ settings.checkDepth
          (p.1, (p.2 ⟨0, hK⟩).1) = some t
        ∧ (p.1, (p.2 ⟨0, hK⟩).2) ∈ accBad settings O D Dc t}
      ≤ ENNReal.ofReal (settings.check.δ / 8) := by
  classical
  set T := settings.level3.probeBudget
  set n := settings.check.size
  set r0 : Fin K := ⟨0, hK⟩
  set ls := settings.toLoopSettings
  set a := settings.checkDepth
  set X₂ := Fin T → unitInterval × (Fin n → unitInterval)
  set ν₂ : Measure X₂ := Measure.pi fun _ : Fin T => (volume : Measure unitInterval).prod
    (Measure.pi fun _ : Fin n => (volume : Measure unitInterval))
  set Y := Option (State × Finset (FreeMonoid α) × Finset (FreeMonoid α) × Finset (FreeMonoid α))
  set S := {p : Ω × (Fin K → CheckRoundDraws α K T n) | ∃ t, stage0 ls O D Dsf h₀ a
      (p.1, (p.2 r0).1) = some t ∧ (p.1, (p.2 r0).2) ∈ accBad settings O D Dc t}
  have hx0 : Measurable fun x : Fin K → CheckRoundDraws α K T n => (x r0).1 :=
    measurable_fst.comp (measurable_pi_apply r0)
  have hS : MeasurableSet S := by
    have : S = ⋃ t : State × Finset (FreeMonoid α) × Finset (FreeMonoid α) × Finset (FreeMonoid α),
        {p | stage0 ls O D Dsf h₀ a (p.1, (p.2 r0).1) = some t}
          ∩ (fun p : Ω × (Fin K → CheckRoundDraws α K T n) => (p.1, (p.2 r0).2)) ⁻¹'
            accBad settings O D Dc t := by
      ext p
      simp only [S, Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_preimage]
    rw [this]
    exact MeasurableSet.iUnion fun t =>
      ((measurable_fst.prodMk (hx0.comp measurable_snd))
        (measurableSet_stage0 ls O D Dsf h₀ a (some t))).inter
      ((measurable_fst.prodMk (measurable_snd.comp ((measurable_pi_apply r0).comp
        measurable_snd))) (measurableSet_accBad settings O D Dc t))
  have hsec : ∀ ω, (Measure.pi fun _ : Fin K => checkRoundLaw α K T n) (Prod.mk ω ⁻¹' S)
      ≤ ENNReal.ofReal (settings.check.δ / 8) := by
    intro ω
    have hσ : MeasurePreserving (id : CheckRoundDraws α K T n → CheckRoundDraws α K T n)
        (checkRoundLaw α K T n) ((restLaw α K).prod ν₂) := MeasurePreserving.id _
    set Bad : Y → Set X₂ := fun y => match y with
      | some t => {v | (ω, v) ∈ accBad settings O D Dc t}
      | none => ∅
    have hBad : ∀ y, ν₂ (Bad y) ≤ ENNReal.ofReal (settings.check.δ / 8) := by
      intro y
      rcases y with _ | t
      · simp [Bad]
      · have hdm := measurePreserving_checkDrawsOf (T := T) settings.check D Dc settings.avoidDepth
          a t.2.2.1 t.2.2.2
        set rd := fun z => familyRead O.mq t.2.1 t.1.lo (t.1.hi + 1) z ω
        have hX : MeasurableSet {ds : Fin T → CheckDraw α settings.check |
            (indecisiveLStarCheck settings.level3 settings.check rd ds).ending = .consistent
            ∧ settings.check.rate < (avoidLaw Dc a t.2.2.2).real {x | startFails rd
              (indecisiveLStarCheck settings.level3 settings.check rd ds).hypothesis.tree
              (indecisiveLStarCheck settings.level3 settings.check rd ds).hypothesis.edges x
                = true}} := (Set.to_countable _).measurableSet
        have e : Bad (some t) = checkDrawsOf settings.check D Dc settings.avoidDepth a t.2.2.1 t.2.2.2
            ⁻¹' {ds : Fin T → CheckDraw α settings.check |
            (indecisiveLStarCheck settings.level3 settings.check rd ds).ending = .consistent
            ∧ settings.check.rate < (avoidLaw Dc a t.2.2.2).real {x | startFails rd
              (indecisiveLStarCheck settings.level3 settings.check rd ds).hypothesis.tree
              (indecisiveLStarCheck settings.level3 settings.check rd ds).hypothesis.edges x
                = true}} := rfl
        rw [e, hdm.measure_preimage hX.nullMeasurableSet, ← ofReal_measureReal]
        exact ENNReal.ofReal_le_ofReal (level3CheckSound_holds rd _ _ settings.level3 settings.check
          hKδ hr0 hr1)
    have h := round_stage_le (checkRoundLaw α K T n) (restLaw α K) ν₂ id hσ id measurable_id
      (fun w => rfl) r0 (fun x => stage0 ls O D Dsf h₀ a (ω, (x r0).1))
      (fun x x' _ h1 => by
        change (x r0).1 = (x' r0).1 at h1
        simp only [h1])
      (fun y => hx0 (measurable_prodMk_left (measurableSet_stage0 ls O D Dsf h₀ a y)))
      Bad (ENNReal.ofReal (settings.check.δ / 8)) hBad
    refine (measure_mono fun g hg => ?_).trans h
    obtain ⟨t, ht, hf⟩ := hg
    show (g r0).2 ∈ Bad (stage0 ls O D Dsf h₀ a (ω, (g r0).1))
    rw [ht]
    exact hf
  unfold checkLoopMeasure
  rw [Measure.prod_apply hS]
  calc ∫⁻ ω, (Measure.pi fun _ : Fin K => checkRoundLaw α K T n) (Prod.mk ω ⁻¹' S) ∂μ
      ≤ ∫⁻ _ω, ENNReal.ofReal (settings.check.δ / 8) ∂μ := lintegral_mono hsec
    _ = ENNReal.ofReal (settings.check.δ / 8) := by rw [lintegral_const, measure_univ, mul_one]

end Acc

end OrthoDFA
