import RLStar.Proofs.TrieBirthday
import RLStar.Proofs.TrieLoop

/-!
# Each round's probes and batches, after the rounds before it

Round `r`'s level-3 uniforms are fresh given what reaches its level 3 (`stage4`), and its
harvested populations' batches fresh given the earlier rounds (`stageH`), so each draws a string
shorter than `walkStart` with chance `0` (`trie_l3_round_le`, `trie_batches_round_le`). Where every
round the filled draws run is good and none draws one, the loop with local walk starts meets
`TrieLoopAgrees`'s premise (`trieRoundGood_of`).
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory

open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

section Round

variable (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α) (Dsf : Measure (FreeMonoid α))
  [IsProbabilityMeasure Dsf] (settings : LoopSettings K)

/-- Round `r`'s level-3 probes, where every earlier clustering and this one returned, are not at
least `walkStart` long with chance `0`. -/
theorem trie_l3_round_le {NP : ℕ} (hNP : ∀ pops, ∑ B ∈ settings.schedule pops, B.npref ≤ 2 * NP)
    {p₀ : ℝ} (hpre : ∀ u, P.D.real {x | prefixOf x settings.level3.walkStart = u} ≤ p₀)
    (hk : ∀ᵐ x ∂P.D, settings.level3.walkStart ≤ x.toList.length)
    (hK : ((6 * (K + 1) * NP : ℕ) : ℝ) * p₀ ≤ 1 / 2) (r : ℕ) (hr : r < K) :
    (loopMeasure μ α K settings.level3.harvestReadCap settings.level3.probeBudget).real
      {p | HistOK settings O P.D Dsf p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf p.1 p.2 r).length = r
        ∧ Returns settings O P.D Dsf (decodeHist
            (histCodes settings O P.D Dsf p.1 p.2 r)) (p.1, (p.2 ⟨r, hr⟩).1)
        ∧ ¬ L3Good settings.level3.walkStart P.D
          (roundKeys settings P.D Dsf (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r))
            (p.1, (p.2 ⟨r, hr⟩).1)) (p.2 ⟨r, hr⟩).2}
      ≤ 0 := by
  classical
  set T := settings.level3.probeBudget
  set k := settings.level3.walkStart
  set dec := decodeHist (α := α) with hdec
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  set rK : Fin K := ⟨r, hr⟩
  set ν := roundLaw α K settings.level3.harvestReadCap T
  have hσ : MeasurePreserving (id : RoundDraws α K settings.level3.harvestReadCap T →
      RoundDraws α K settings.level3.harvestReadCap T) ν
      ((restLaw α K settings.level3.harvestReadCap).prod
        (Measure.pi fun _ : Fin T => (volume : Measure unitInterval))) :=
    MeasurePreserving.id _
  have hp0 : 0 ≤ p₀ := measureReal_nonneg.trans (hpre 1)
  set Y := Option (List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α))
  set Good : List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α) →
      Prop := fun t => t.1.length = r ∧ t.2.2.2.card ≤ 6 * (K + 1) * NP with hGood
  set Bad : Y → Set (Ω × (Fin T → unitInterval)) := fun y => match y with
    | some t => if Good t then Set.univ ×ˢ {u | ¬ L3Good k P.D t.2.2.2 u} else ∅
    | none => ∅ with hBad
  set W : Y → (Fin K → RoundDraws α K settings.level3.harvestReadCap T) →
      Finset (FreeMonoid α) := fun y x => match y with
    | some t => histReads settings P.D Dsf t.1 x r ω₀
        ∪ schedReads (settings.schedule (loopPopulations K (dec t.1).pops.length))
          (loopPopulations K (dec t.1).pops.length)
          ((ω₀, runOf settings P.D Dsf (dec t.1) (x rK).1) :
            Run Ω (FreeMonoid α) (Option (Fin K)))
    | none => ∅
  set G : Y → Prop := fun y => ∃ t, y = some t ∧ t.1.length = r
  have hmain := persist_stage_le O ν (restLaw α K settings.level3.harvestReadCap)
    (Measure.pi fun _ : Fin T => (volume : Measure unitInterval)) id hσ id measurable_id
    (fun w => rfl) rK (stage4 settings O P.D Dsf r hr)
    (fun ω x x' hxx' h1 => stage4_congr_draws settings O P.D Dsf r hr ω x x'
      (fun i hi => hxx' i hi) h1)
    (measurableSet_stage4 settings O P.D Dsf P.length_le r hr) W G
    (fun y x ω ω' hG hs h => by
      obtain ⟨⟨c, B, F, K₀⟩, rfl, -⟩ := hG
      exact stage4_stop settings O P.D Dsf P.length_le r hr c B F K₀ x ω ω' ω₀ hs h)
    Bad (fun y hy => by
      rcases y with _ | t
      · rfl
      · show (if Good t then _ else ∅) = ∅
        rw [if_neg fun h => hy ⟨t, rfl, h.1⟩])
    (fun y => by
      rcases y with _ | t
      · exact MeasurableSet.empty
      · show MeasurableSet (if Good t then _ else ∅)
        split_ifs
        · exact MeasurableSet.univ.prod (measurableSet_l3Good k P.D t.2.2.2).compl
        · exact MeasurableSet.empty)
    (fun _ _ => ∅) (fun y u ω ω' _ hq => by
      rcases y with _ | t
      · exact hq
      · change (ω, u) ∈ (if Good t then _ else ∅) at hq
        show (ω', u) ∈ (if Good t then _ else ∅)
        split_ifs at hq ⊢
        · exact ⟨Set.mem_univ _, hq.2⟩
        · exact hq)
    (fun _ => Set.univ) (fun _ => MeasurableSet.univ)
    (fun y x _ _ => Finset.disjoint_empty_right _)
    (ENNReal.ofReal 0) (fun y => by
      rcases y with _ | t
      · show (μ.prod _) ∅ ≤ _
        simp
      · show (μ.prod _) (if Good t then _ else ∅) ≤ _
        split_ifs with h
        · rw [Measure.prod_prod, measure_univ, one_mul]
          refine (pi_not_l3Good_le P.D k t.2.2.2 hpre hk ?_).le
          have h1 : (t.2.2.2.card : ℝ) ≤ ((6 * (K + 1) * NP : ℕ) : ℝ) := by exact_mod_cast h.2
          exact (mul_le_mul_of_nonneg_right h1 hp0).trans hK
        · simp)
  refine ENNReal.toReal_le_of_le_ofReal le_rfl ((measure_mono ?_).trans hmain)
  rintro p ⟨hok, hlen, hret, hnot⟩
  set c := histCodes settings O P.D Dsf p.1 p.2 r
  set H := dec c
  set cd : ClusterDraws Ω α K settings.level3.harvestReadCap := (p.1, (p.2 rK).1)
  set B := roundState settings O P.D Dsf H cd
  set F := clusterAt O.mq (loopPopulations K H.pops.length)
    (idealRun P.D Dsf k settings.level3.harvestReadCap H cd) B
  set K₀ := roundKeys settings P.D Dsf H cd
  have hst : stage4 settings O P.D Dsf r hr p.1 p.2 = some (c, B, F, K₀) :=
    (stage4_eq_some settings O P.D Dsf r hr p.1 p.2 c B F K₀).2
      ⟨hok, rfl, hlen, hret, rfl, rfl, rfl⟩
  have hK₀ : K₀.card ≤ 6 * (K + 1) * NP := by
    rw [show K₀ = _ from roundKeys_eq_prefsOf settings P.D Dsf H p.1 (p.2 rK).1]
    refine Finset.card_image_le.trans ((card_roundPrefs_le _ _ _).trans ?_)
    have hpops : (loopPopulations K H.pops.length).card ≤ K + 1 := by
      have := Finset.card_le_univ (loopPopulations K H.pops.length)
      simpa using this
    have := hNP (loopPopulations K H.pops.length)
    calc 3 * (loopPopulations K H.pops.length).card
          * ∑ B ∈ settings.schedule (loopPopulations K H.pops.length), B.npref
        ≤ 3 * (K + 1) * (2 * NP) := Nat.mul_le_mul (Nat.mul_le_mul_left 3 hpops) this
      _ = 6 * (K + 1) * NP := by ring
  show (p.1, (id (p.2 rK)).2) ∈ Bad (stage4 settings O P.D Dsf r hr p.1 p.2)
    ∧ p.2 ∈ (fun _ : Y => (Set.univ : Set (Fin K → RoundDraws α K
      settings.level3.harvestReadCap T))) (stage4 settings O P.D Dsf r hr p.1 p.2)
  rw [hst]
  refine ⟨?_, Set.mem_univ _⟩
  show (p.1, (p.2 rK).2) ∈ (if Good (c, B, F, K₀) then _ else ∅)
  rw [if_pos ⟨hlen, hK₀⟩]
  exact ⟨Set.mem_univ _, hnot⟩

omit [IsProbabilityMeasure Dsf] in
lemma restLaw_map_pop (N : ℕ) (i : Fin K) :
    MeasurePreserving (fun e : ClusterRest α K N => e.2.1 (some i)) (restLaw α K N)
      (Measure.infinitePi fun _ : ℕ => drawLaw N) := by
  refine ⟨(measurable_pi_apply _).comp (measurable_fst.comp measurable_snd), ?_⟩
  rw [show (fun e : ClusterRest α K N => e.2.1 (some i))
      = (fun v : Option (Fin K) → ℕ → unitInterval × (Fin N → unitInterval) => v (some i))
        ∘ Prod.fst ∘ Prod.snd from rfl,
    ← Measure.map_map (measurable_pi_apply _) (measurable_fst.comp measurable_snd),
    ← Measure.map_map measurable_fst measurable_snd, restLaw, Measure.map_snd_prod,
    measure_univ, one_smul, Measure.map_fst_prod, measure_univ, one_smul]
  exact (measurePreserving_eval _ (some i)).map_eq

/-- Round `r`'s harvested populations' batches, after earlier rounds whose clusterings returned,
fail `BatchesGood` with chance `0`. -/
theorem trie_batches_round_le (hk : ∀ᵐ x ∂P.D, settings.level3.walkStart ≤ x.toList.length)
    (r : ℕ) (hr : r < K) :
    (loopMeasure μ α K settings.level3.harvestReadCap settings.level3.probeBudget).real
      {p | HistOK settings O P.D Dsf p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf p.1 p.2 r).length = r
        ∧ ∃ i : Fin K, ¬ BatchesGood settings.level3.walkStart P.D settings.level3.harvestReadCap
          settings.scanCap ((p.2 ⟨r, hr⟩).1.2.1 (some i))}
      ≤ 0 := by
  classical
  set k := settings.level3.walkStart
  set N := settings.level3.harvestReadCap
  set M := settings.scanCap
  set PB := settings.level3.probeBudget
  set dec := decodeHist (α := α) with hdec
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  set ν := roundLaw α K N PB
  have hσ : MeasurePreserving (fun d : RoundDraws α K N PB => ((() : Unit), d)) ν
      ((Measure.dirac ()).prod ν) :=
    ⟨measurable_const.prodMk measurable_id, by rw [Measure.dirac_prod]⟩
  set Y := Option (List (HarvestCode α))
  set W : Y → (Fin K → RoundDraws α K N PB) → Finset (FreeMonoid α) := fun y x => match y with
    | some c => histReads settings P.D Dsf c x r ω₀
    | none => ∅
  set G : Y → Prop := fun y => ∃ c, y = some c ∧ c.length = r
  set Good : List (HarvestCode α) → Prop := fun c => c.length = r
  set Bad : Y → Set (Ω × RoundDraws α K N PB) := fun y => match y with
    | some c => {q | Good c ∧ ∃ i : Fin K, ¬ BatchesGood k P.D N M (q.2.1.2.1 (some i))}
    | none => ∅
  have hpop : ∀ i : Fin K, Measurable fun q : Ω × RoundDraws α K N PB => q.2.1.2.1 (some i) :=
    fun i => (measurable_pi_apply _).comp
      (measurable_fst.comp (measurable_snd.comp (measurable_fst.comp measurable_snd)))
  have hmBad : MeasurableSet {q : Ω × RoundDraws α K N PB |
      ∃ i : Fin K, ¬ BatchesGood k P.D N M (q.2.1.2.1 (some i))} := by
    rw [Set.setOf_exists]
    exact MeasurableSet.iUnion fun i => hpop i (measurableSet_batchesGood k P.D N M).compl
  have hmain := persist_stage_le O ν (Measure.dirac ()) ν (fun d => ((() : Unit), d)) hσ Prod.snd
    measurable_snd (fun w => Prod.ext rfl rfl) ⟨r, hr⟩ (stageH settings O P.D Dsf r)
    (fun ω x x' hxx' _ => stageH_congr_draws settings O P.D Dsf r ω x x' fun i hi => hxx' i hi)
    (measurableSet_stageH settings O P.D Dsf P.length_le r) W G
    (fun y x ω ω' hG hs h => by
      obtain ⟨c, rfl, -⟩ := hG
      exact stageH_stop settings O P.D Dsf P.length_le r c x ω ω' ω₀ hs h)
    Bad (fun y hy => by
      rcases y with _ | c
      · rfl
      · exact Set.eq_empty_of_forall_notMem fun q hq => hy ⟨c, rfl, hq.1⟩)
    (fun y => by
      rcases y with _ | c
      · exact MeasurableSet.empty
      · exact (MeasurableSet.const _).inter hmBad)
    (fun _ _ => ∅) (fun y d ω ω' _ hq => by
      rcases y with _ | c
      · exact hq
      · exact hq)
    (fun _ => Set.univ) (fun _ => MeasurableSet.univ)
    (fun y x _ _ => Finset.disjoint_empty_right _)
    (ENNReal.ofReal 0) (fun y => by
      rcases y with _ | c
      · show (μ.prod ν) ∅ ≤ _
        simp
      · have hone : ∀ i : Fin K, (μ.prod ν) {q : Ω × RoundDraws α K N PB |
            ¬ BatchesGood k P.D N M (q.2.1.2.1 (some i))} = 0 := by
          intro i
          have hp : MeasurePreserving (fun q : Ω × RoundDraws α K N PB => q.2.1.2.1 (some i))
              (μ.prod ν) (Measure.infinitePi fun _ : ℕ => drawLaw N) :=
            (restLaw_map_pop N i).comp (measurePreserving_fst.comp measurePreserving_snd)
          rw [show {q : Ω × RoundDraws α K N PB |
              ¬ BatchesGood k P.D N M (q.2.1.2.1 (some i))}
              = (fun q : Ω × RoundDraws α K N PB => q.2.1.2.1 (some i)) ⁻¹'
                {u | ¬ BatchesGood k P.D N M u} from rfl,
            hp.measure_preimage (show MeasurableSet {u | ¬ BatchesGood k P.D N M u} from
              (measurableSet_batchesGood k P.D N M).compl).nullMeasurableSet]
          exact batches_bad_le P.D k N M hk
        refine (measure_mono fun q hq => Set.mem_iUnion.2 hq.2).trans ?_
        rw [measure_iUnion_null hone]
        exact zero_le)
  refine ENNReal.toReal_le_of_le_ofReal le_rfl ((measure_mono ?_).trans hmain)
  rintro p ⟨hok, hlen, hbad⟩
  have hst : stageH settings O P.D Dsf r p.1 p.2
      = some (histCodes settings O P.D Dsf p.1 p.2 r) := by
    simp only [stageH, if_pos hok]
  show (p.1, p.2 ⟨r, hr⟩) ∈ Bad (stageH settings O P.D Dsf r p.1 p.2)
    ∧ p.2 ∈ (fun _ : Y => (Set.univ : Set (Fin K → RoundDraws α K N PB)))
      (stageH settings O P.D Dsf r p.1 p.2)
  rw [hst]
  exact ⟨⟨hlen, hbad⟩, Set.mem_univ _⟩

end Round

section Bridge

variable (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α) (Dsf : Measure (FreeMonoid α))
  (settings : LoopSettings K)
  (p : Ω × (Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget))

omit [IsProbabilityMeasure μ] in
lemma histAlike_of_harvested {r : ℕ} (hr : r ≤ K)
    (hall : AllHarvested (idealRounds settings O P.D Dsf (persistDraws p) r)) :
    (histCodes settings O P.D Dsf p.1 p.2 r).length = r
      ∧ HistAlike (lenBound P.D) (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r))
        (specHistOf (idealRounds settings O P.D Dsf (persistDraws p) r)) := by
  set draws := persistDraws p
  have hdec := decodeHist_alike (L := lenBound P.D) (N := codeLen settings P.D) settings.level3
    le_rfl (idealRounds settings O P.D Dsf draws r) (idealRounds_result settings O P.D Dsf draws r)
    hall
  have hc : histCodes settings O P.D Dsf p.1 p.2 r
      = codesOf (codeLen settings P.D) (idealRounds settings O P.D Dsf draws r) := rfl
  refine ⟨?_, by rw [hc]; exact hdec⟩
  rw [hc, length_codesOf (codeLen settings P.D) P.D settings.level3.walkStart,
    length_harvestsOf hall, length_eq_of_allHarvested settings O P.D Dsf draws hr hall]

omit [IsProbabilityMeasure μ] in
lemma ret_of_alike {h h' : SpecHist α} (hh : HistAlike (lenBound P.D) h h')
    (c : ClusterDraws Ω α K settings.level3.harvestReadCap)
    (hret : ∃ B ∈ settings.schedule (loopPopulations K h'.pops.length),
      idealRun P.D Dsf settings.level3.walkStart settings.level3.harvestReadCap h' c
        ∈ ret O.mq (loopPopulations K h'.pops.length) none settings.η₀ settings.indecisionLimit
          settings.εcov settings.gateLevel B) :
    ∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
      idealRun P.D Dsf settings.level3.walkStart settings.level3.harvestReadCap h c
        ∈ ret O.mq (loopPopulations K h.pops.length) none settings.η₀ settings.indecisionLimit
          settings.εcov settings.gateLevel B := by
  rw [hh.length, idealRun_alike hh P.D Dsf (ae_lenBound P.length_le)]
  exact hret

omit [IsProbabilityMeasure μ] in
lemma histOK_of_good
    (hrg : ∀ r (hr : r < K), AllHarvested (idealRounds settings O P.D Dsf (persistDraws p) r) →
      RoundGood settings O P.D Dsf (specHistOf (idealRounds settings O P.D Dsf (persistDraws p) r))
        (persistDraws p ⟨r, hr⟩)) :
    ∀ r, r ≤ K → AllHarvested (idealRounds settings O P.D Dsf (persistDraws p) r) →
      HistOK settings O P.D Dsf p.1 p.2 r
  | 0, _, _ => trivial
  | r + 1, hr, hall => by
    have hall' : AllHarvested (idealRounds settings O P.D Dsf (persistDraws p) r) := fun o ho =>
      hall o ((idealRounds_prefix settings O P.D Dsf (persistDraws p) r).subset ho)
    refine ⟨histOK_of_good hrg r (by omega) hall', fun hK _ => ?_⟩
    obtain ⟨-, hh⟩ := histAlike_of_harvested O P Dsf settings p (by omega) hall'
    exact ret_of_alike O P Dsf settings hh _ (hrg r hK hall').2.2

omit [IsProbabilityMeasure μ] in
/-- Where every round the filled draws run is good, and no round's probes or batches draw a string
shorter than `walkStart`, the loop with local walk starts meets `TrieLoopAgrees`'s premise. -/
theorem trieRoundGood_of
    (hrg : ∀ r (hr : r < K), AllHarvested (idealRounds settings O P.D Dsf (persistDraws p) r) →
      RoundGood settings O P.D Dsf (specHistOf (idealRounds settings O P.D Dsf (persistDraws p) r))
        (persistDraws p ⟨r, hr⟩))
    (hL3 : ∀ r (hr : r < K), ¬ (HistOK settings O P.D Dsf p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf p.1 p.2 r).length = r
        ∧ Returns settings O P.D Dsf (decodeHist
            (histCodes settings O P.D Dsf p.1 p.2 r)) (p.1, (p.2 ⟨r, hr⟩).1)
        ∧ ¬ L3Good settings.level3.walkStart P.D
          (roundKeys settings P.D Dsf (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r))
            (p.1, (p.2 ⟨r, hr⟩).1)) (p.2 ⟨r, hr⟩).2))
    (hTop : ∀ r (hr : r < K), ¬ (HistOK settings O P.D Dsf p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf p.1 p.2 r).length = r
        ∧ ∃ i : Fin K, ¬ BatchesGood settings.level3.walkStart P.D settings.level3.harvestReadCap
          settings.scanCap ((p.2 ⟨r, hr⟩).1.2.1 (some i)))) :
    ∀ r (hr : r < K) rs,
      (loopRoundsQ settings P.D Dsf p.2 r).eval (fun w => O.mq w p.1) = some rs →
      (∀ o ∈ rs, o.result.ending.harvested ≠ none) →
      TrieRoundGood settings P.D Dsf (histOf rs) (p.2 ⟨r, hr⟩) (fun w => O.mq w p.1) := by
  intro r hr rs hrs hharv
  set draws := persistDraws p
  obtain ⟨e1, e2⟩ := eval_loopRoundsQ O p.1 settings P.D Dsf p.2 r
  have hloop : loopRounds settings O P.D Dsf draws r
      = some (idealRounds settings O P.D Dsf draws r) :=
    loopRounds_eq_some settings O P.D Dsf draws hrg r
  have hread : ∀ o ∈ idealRounds settings O P.D Dsf draws r, o.ReadsAt O p.1 :=
    e2 _ hloop
  rw [show persistDraws (p.1, p.2) = draws from rfl, hloop] at e1
  rw [e1, Option.map_some, Option.some.injEq] at hrs
  subst hrs
  have hall : AllHarvested (idealRounds settings O P.D Dsf draws r) := fun o ho =>
    hharv o.data (List.mem_map_of_mem ho)
  obtain ⟨hlen, hh⟩ := histAlike_of_harvested O P Dsf settings p hr.le hall
  have hok := histOK_of_good O P Dsf settings p hrg r hr.le hall
  have hgood := hrg r hr hall
  have hret : Returns settings O P.D Dsf (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r))
      (p.1, (p.2 ⟨r, hr⟩).1) := ret_of_alike O P Dsf settings hh _ hgood.2.2
  refine ⟨fun i => ?_, fun o ho => ?_⟩
  · by_contra hbad
    exact hTop r hr ⟨hok, hlen, i, hbad⟩
  · obtain ⟨f1, -⟩ := eval_roundQ O p.1 settings P.D Dsf
      (histOf ((idealRounds settings O P.D Dsf draws r).map SpecRound.data)) (p.2 ⟨r, hr⟩)
    rw [← specHistOf_eq O p.1 hread] at f1
    rw [f1, loopRound_eq_some settings O P.D Dsf _ (p.1, p.2 ⟨r, hr⟩) hgood.1 hgood.2.1 hgood.2.2,
      Option.map_some, Option.some.injEq] at ho
    subst ho
    have hkeys : ((roundOf settings O P.D (specHistOf (idealRounds settings O P.D Dsf draws r))
        (idealRun P.D Dsf settings.level3.walkStart settings.level3.harvestReadCap
          (specHistOf (idealRounds settings O P.D Dsf draws r)) (p.1, (p.2 ⟨r, hr⟩).1))
        (p.2 ⟨r, hr⟩).2).data.prefs).image (fun q => prefixOf q settings.level3.walkStart)
        = roundKeys settings P.D Dsf (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r))
          (p.1, (p.2 ⟨r, hr⟩).1) := by
      show clusterKeys _ _ _ _ = clusterKeys _ _ _ _
      rw [hh.length, idealRun_alike hh P.D Dsf (ae_lenBound P.length_le)]
    rw [hkeys]
    by_contra hbad
    exact hL3 r hr ⟨hok, hlen, hret, hbad⟩

end Bridge

end OrthoDFA

end
