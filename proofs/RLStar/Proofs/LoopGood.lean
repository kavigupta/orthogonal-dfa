import RLStar.Proofs.LoopColl
import RLStar.Proofs.ScanTail

/-!
# A round's draws scan within the cap

A round is good (`RoundGood`) where its draws scan within `scanCap` and its clustering returns at
a scheduled state. The draws scan short with the chance of `clusterLaw_scanShort_le`, charged in
one round's draws after any history the earlier rounds hand on (`loop_good_round_le_persist`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {K : ℕ}


section Good

variable {μ : Measure Ω} [IsProbabilityMeasure μ]

instance : IsProbabilityMeasure (clusterLaw μ α K) := by
  unfold clusterLaw; infer_instance

omit [IsProbabilityMeasure μ] in
lemma card_loopPopulations_le (K r : ℕ) : (loopPopulations K r).card ≤ K + 1 := by
  unfold loopPopulations
  refine (Finset.card_insert_le _ _).trans (Nat.add_le_add_right ?_ 1)
  exact Finset.card_image_le.trans ((Finset.card_filter_le _ _).trans (by simp))

omit [IsProbabilityMeasure μ] [DecidableEq α] in
/-- No population's sides falling short and a scheduled state returning, the round is good. -/
lemma roundGood_of (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : SpecHist α)
    (d : LoopRoundDraws Ω α settings)
    {T A NP : ℕ}
    (hNP : ∀ B ∈ settings.schedule (loopPopulations K h.pops.length), B.npref ≤ NP)
    (hNA : (2 + (loopPopulations K h.pops.length).card) * NP ≤ A) (hAT : A ≤ T)
    (h2 : ¬ ScanShort (loopPopulations K h.pops.length) settings.scanCap T A
      (idealDraws D Dsf h (d.1, d.2.1)))
    (h3 : ∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
      idealRun D Dsf h (d.1, d.2.1)
        ∈ ret O.mq (loopPopulations K h.pops.length) none settings.η₀ settings.indecisionLimit
          settings.εcov settings.gateLevel B) :
    RoundGood settings O D Dsf h d :=
  ⟨scanWithin_of_not_scanShort _ _ hNP hNA hAT _ h2, h3⟩

omit [IsProbabilityMeasure μ] [DecidableEq α] [MeasurableSpace Ω] in
/-- Falling short reads no noise. -/
lemma scanShort_noise (pops : Finset (Option (Fin K))) (M T A : ℕ) (ω ω' : Ω)
    (s s' : ((ℕ → FreeMonoid α) × (Option (Fin K) → ℕ → FreeMonoid α))
      × (Option (Fin K) → ℕ → FreeMonoid α)) (g g' : FreeMonoid α → Bool) (hs : s = s')
    (hg : g = g') : ScanShort pops M T A ((ω, s), g) ↔ ScanShort pops M T A ((ω', s'), g') := by
  subst hs hg
  unfold ScanShort
  refine exists_congr fun j => and_congr_right fun _ => ?_
  rcases j with _ | i <;> exact Iff.rfl

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
  (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]

omit [IsProbabilityMeasure Dsf] in
/-- Round `r`'s draws, after any history the earlier rounds hand on whose populations hold at
most `cap` on a string and at most `m` of the earlier prefixes, fall short within the scan cap `M`
with chance at most

    (K + 1) (2 C(M, T) (2 T cap)^(M-T) + 2 exp(-T/8)),

where `4 (K + 3) NP ≤ T ≤ M`. -/
theorem loop_good_round_le_persist (cap m : ℝ) {T NP : ℕ} (hm : m ≤ 1 / 2)
    (hA : 4 * ((K + 3) * NP) ≤ T) (hT : T ≤ settings.scanCap) (hDat : ∀ t, P.D.real {t} ≤ cap)
    (r : ℕ) (hr : r < K) :
    (settings.measure μ α).real
      {p | HistOK settings O P.D Dsf p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf p.1 p.2 r).length = r
        ∧ (∀ ℓ ∈ (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)).pops,
            ∀ t, (listLaw ℓ).real {t} ≤ cap)
        ∧ (∀ j ∈ loopPopulations K r, (loopDistributions P.D (decodeHist
            (histCodes settings O P.D Dsf p.1 p.2 r)).pops j).real
              ↑(decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)).prefs ≤ m)
        ∧ ScanShort (loopPopulations K r) settings.scanCap T ((K + 3) * NP)
            (idealDraws P.D Dsf
              (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)) (p.1, (p.2 ⟨r, hr⟩).1))}
      ≤ (K + 1) * (2 * (settings.scanCap.choose T * (T * (2 * cap)) ^ (settings.scanCap - T))
          + 2 * Real.exp (-(T / 8 : ℝ))) := by
  classical
  set M := settings.scanCap
  set A := (K + 3) * NP
  set dec := decodeHist (α := α) with hdec
  set b : ℝ := (K + 1) * (2 * (M.choose T * (T * (2 * cap)) ^ (M - T))
    + 2 * Real.exp (-(T / 8 : ℝ)))
  have hcap0 : 0 ≤ cap := le_trans measureReal_nonneg (hDat 1)
  have hb0 : 0 ≤ b := by positivity
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  set ν := settings.roundLaw α
  have hσ : MeasurePreserving (fun d : settings.RoundDraws α => ((() : Unit), d)) ν
      ((Measure.dirac ()).prod ν) :=
    ⟨measurable_const.prodMk measurable_id, by rw [Measure.dirac_prod]⟩
  set Y := Option (List (HarvestCode α))
  set W : Y → (Fin K → settings.RoundDraws α) → Finset (FreeMonoid α) := fun y x => match y with
    | some c => histReads settings P.D Dsf c x r ω₀
    | none => ∅
  set G : Y → Prop := fun y => ∃ c, y = some c ∧ c.length = r
  set Good : List (HarvestCode α) → Prop := fun c => c.length = r
    ∧ (∀ ℓ ∈ (dec c).pops, ∀ t, (listLaw ℓ).real {t} ≤ cap)
    ∧ (∀ j ∈ loopPopulations K r, (loopDistributions P.D (dec c).pops j).real ↑(dec c).prefs ≤ m)
  set Bad : Y → Set (Ω × settings.RoundDraws α) := fun y => match y with
    | some c => {q | Good c
        ∧ ScanShort (loopPopulations K r) M T A (idealDraws P.D Dsf (dec c) (q.1, q.2.1))}
    | none => ∅
  have hmarg : MeasurePreserving (fun q : Ω × settings.RoundDraws α =>
      ((q.1, q.2.1) : ClusterDraws Ω α K)) (μ.prod ν) (clusterLaw μ α K) :=
    (MeasurePreserving.id μ).prod measurePreserving_fst
  have hmid : ∀ c : List (HarvestCode α), Measurable
      fun q : Ω × settings.RoundDraws α => idealDraws P.D Dsf (dec c) (q.1, q.2.1) := fun c =>
    (measurable_idealDraws P.D Dsf (dec c)).comp hmarg.measurable
  have hmain := persist_stage_le O ν (Measure.dirac ()) ν (fun d => ((() : Unit), d)) hσ Prod.snd
    measurable_snd (fun w => Prod.ext rfl rfl) ⟨r, hr⟩ (stageH settings O P.D Dsf r)
    (fun ω x x' hxx' _ => stageH_congr_draws settings O P.D Dsf r ω x x' fun i hi => hxx' i hi)
    (measurableSet_stageH settings O P.D Dsf r) W G
    (fun y x ω ω' hG hs h => by
      obtain ⟨c, rfl, -⟩ := hG
      exact stageH_stop settings O P.D Dsf r c x ω ω' ω₀ hs h)
    Bad (fun y hy => by
      rcases y with _ | c
      · rfl
      · exact Set.eq_empty_of_forall_notMem fun q hq => hy ⟨c, rfl, hq.1.1⟩)
    (fun y => by
      rcases y with _ | c
      · exact MeasurableSet.empty
      · exact (MeasurableSet.const _).inter
          (hmid c (measurableSet_scanShort (loopPopulations K r) M T A)))
    (fun _ _ => ∅) (fun y d ω ω' _ hq => by
      rcases y with _ | c
      · exact hq
      · exact ⟨hq.1, (scanShort_noise _ _ _ _ _ _ _ _ _ _ rfl rfl).1 hq.2⟩)
    (fun _ => Set.univ) (fun _ => MeasurableSet.univ)
    (fun y x _ _ => Finset.disjoint_empty_right _)
    (ENNReal.ofReal b) (fun y => by
      rcases y with _ | c
      · show (μ.prod ν) ∅ ≤ _
        simp
      · by_cases hH : Good c
        swap
        · have : Bad (some c) = ∅ := Set.eq_empty_of_forall_notMem fun q hq => hH hq.1
          rw [this, measure_empty]; exact zero_le
        obtain ⟨hcr, hpg, hpm⟩ := hH
        set h := dec c
        have hlen : h.pops.length = r := by
          simp only [h, dec, decodeHist, List.length_map, hcr]
        -- each population off the earlier prefixes holds at most `2 cap` on a string
        have hat : ∀ j ∈ loopPopulations K r, ∀ t,
            (offLaw (loopDistributions P.D h.pops j) h.prefs).real {t} ≤ 2 * cap := by
          intro j hj t
          refine (offLaw_atom_le ((hpm j hj).trans hm) t).trans ?_
          have : (loopDistributions P.D h.pops j).real {t} ≤ cap := by
            cases j with
            | none => exact hDat t
            | some i =>
              have hi : i.val < h.pops.length := by rw [hlen]; exact lt_of_some_mem_loopPops hj
              simp only [loopDistributions, List.getElem?_eq_getElem hi]
              exact hpg _ (List.getElem_mem _) t
          linarith
        have hscan := clusterLaw_scanShort_le (μ := μ) P.D Dsf h (loopPopulations K r)
          (M := M) (T := T) (A := A) hat hA hT
        have hcard : ((loopPopulations K r).card : ℝ) ≤ K + 1 := by
          exact_mod_cast card_loopPopulations_le K r
        have hscanE : (μ.prod ν) {q : Ω × settings.RoundDraws α |
            ScanShort (loopPopulations K r) M T A (idealDraws P.D Dsf h (q.1, q.2.1))}
            = clusterLaw μ α K
              {c | ScanShort (loopPopulations K r) M T A (idealDraws P.D Dsf h c)} :=
          hmarg.measure_preimage (((measurable_idealDraws P.D Dsf h)
            (measurableSet_scanShort (loopPopulations K r) M T A)).nullMeasurableSet)
        calc (μ.prod ν) (Bad (some c))
            ≤ (μ.prod ν) {q : Ω × settings.RoundDraws α |
                ScanShort (loopPopulations K r) M T A (idealDraws P.D Dsf h (q.1, q.2.1))} :=
              measure_mono fun q hq => hq.2
          _ = ENNReal.ofReal ((clusterLaw μ α K).real
                {c | ScanShort (loopPopulations K r) M T A (idealDraws P.D Dsf h c)}) := by
              rw [hscanE, ofReal_measureReal (measure_ne_top _ _)]
          _ ≤ ENNReal.ofReal b := by
              refine ENNReal.ofReal_le_ofReal (hscan.trans ?_)
              have hq0 : 0 ≤ 2 * (M.choose T * (T * (2 * cap)) ^ (M - T))
                  + 2 * Real.exp (-(T / 8 : ℝ)) := by positivity
              exact mul_le_mul_of_nonneg_right hcard hq0)
  refine ENNReal.toReal_le_of_le_ofReal hb0 ((measure_mono ?_).trans hmain)
  rintro p ⟨hok, hlen, hpg, hpm, hbad⟩
  have hst : stageH settings O P.D Dsf r p.1 p.2
      = some (histCodes settings O P.D Dsf p.1 p.2 r) := by
    simp only [stageH, if_pos hok]
  show (p.1, p.2 ⟨r, hr⟩) ∈ Bad (stageH settings O P.D Dsf r p.1 p.2)
    ∧ p.2 ∈ (fun _ : Y => (Set.univ : Set (Fin K → settings.RoundDraws α)))
      (stageH settings O P.D Dsf r p.1 p.2)
  rw [hst]
  exact ⟨⟨⟨hlen, hpg, hpm⟩, hbad⟩, Set.mem_univ _⟩

omit [IsProbabilityMeasure μ] [IsProbabilityMeasure Dsf] in
/-- Where every round the filled draws run is good, the loop runs those rounds. -/
theorem idealRounds_mem_loop (D : Measure (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α settings)
    (hg : ∀ r (hr : r < K), AllHarvested (idealRounds settings O D Dsf draws r) →
      RoundGood settings O D Dsf (specHistOf (idealRounds settings O D Dsf draws r)) (draws ⟨r, hr⟩)) :
    loopRounds settings O D Dsf draws K = some (idealRounds settings O D Dsf draws K) :=
  loopRounds_eq_some settings O D Dsf draws hg K

end Good

section Reads

variable {μ : Measure Ω} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
  (D Dsf : Measure (FreeMonoid α))
  (draws : Fin K → LoopRoundDraws Ω α settings)

/-- Each round the loop runs is a round after some history and clustering run. -/
lemma loopRounds_roundOf (n : ℕ) : ∀ rounds, loopRounds settings O D Dsf draws n = some rounds →
    rounds.length ≤ n ∧ ∀ o ∈ rounds, ∃ (h : SpecHist α) (x : Run Ω (FreeMonoid α) (Option (Fin K)))
      (u : settings.Uniforms), o = roundOf settings O D h x u := by
  induction n with
  | zero =>
    intro rounds hr
    rw [loopRounds, Option.some.injEq] at hr
    subst hr
    simp
  | succ r ih =>
    intro rounds hr
    rw [loopRounds] at hr
    obtain ⟨rs, hrs, hr⟩ := Option.mem_bind_iff.1 hr
    obtain ⟨hl, hrr⟩ := ih rs hrs
    split_ifs at hr with hc
    · obtain ⟨o, ho, rfl⟩ := Option.mem_map.1 hr
      refine ⟨by rw [List.length_append, List.length_singleton]; omega, fun o' ho' => ?_⟩
      rcases List.mem_append.1 ho' with h1 | h1
      · exact hrr o' h1
      · rw [List.mem_singleton] at h1
        subst h1
        unfold loopRound at ho
        dsimp only at ho
        split_ifs at ho
        · exact ⟨_, _, _, (Option.mem_some_iff.1 ho).symm⟩
        · cases ho
    · rw [Option.mem_some_iff] at hr
      subst hr
      exact ⟨by omega, hrr⟩

/-- The rounds the loop runs number at most `K`, each reading at most `a` prefixes and `b`
suffixes where every clustering run does. -/
theorem loop_reads_le_op (a b : ℕ)
    (h : ∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b) :
    ∀ rounds, loopRounds settings O D Dsf draws K = some rounds → rounds.length ≤ K
      ∧ (rounds.map fun o => o.prefs.card).sum ≤ K * a
      ∧ (rounds.map fun o => o.suffs.card).sum ≤ K * b := by
  intro rounds hr
  obtain ⟨hlen, hro⟩ := loopRounds_roundOf settings O D Dsf draws K rounds hr
  have hsum : ∀ (f : SpecRound α → ℕ) (c : ℕ), (∀ o ∈ rounds, f o ≤ c) →
      (rounds.map f).sum ≤ K * c := fun f c hf => by
    refine (List.sum_le_card_nsmul _ c fun n hn => ?_).trans ?_
    · obtain ⟨o, ho, rfl⟩ := List.mem_map.1 hn
      exact hf o ho
    · rw [List.length_map, smul_eq_mul]
      exact Nat.mul_le_mul_right _ hlen
  refine ⟨hlen, hsum _ a fun o ho => ?_, hsum _ b fun o ho => ?_⟩
  · obtain ⟨hh, x, u, rfl⟩ := hro o ho
    exact (h _ x).1
  · obtain ⟨hh, x, u, rfl⟩ := hro o ho
    exact (h _ x).2

end Reads


end OrthoDFA
