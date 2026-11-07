import OrthoDFA.Proofs.LoopColl
import OrthoDFA.Proofs.ScanTail
import OrthoDFA.Proofs.ProbeTotal

/-!
# A round's draws hit and scan within the cap

A round is good (`RoundGood`) where its harvested draws all hit, its filled draws scan within
`scanCap`, and its clustering returns at a scheduled state.  The draws miss (`MissAt`) with chance
at most `K δₘ` at the cap `harvestN y (δₘ / M)`, each harvested population yielding at least `y`
off the earlier rounds' prefixes (`restLaw_missAt_le`); they scan short with the chance of
`clusterLaw_scanShort_le`.  Both charged in one round's draws after any history the earlier rounds
hand on (`loop_good_round_le_persist`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {K : ℕ}

section Miss

/-- Some harvested population's first `M` draws miss. -/
def MissAt (D : Measure (FreeMonoid α)) (k N M : ℕ) (pops : List (SpecSampler α))
    (T : Finset (FreeMonoid α)) (u : Option (Fin K) → ℕ → unitInterval × (Fin N → unitInterval)) :
    Prop :=
  ∃ i : Fin K, popDraws D k N M pops T (some i) (u (some i)) = none

omit [DecidableEq α] in
lemma clusterDraws_isSome (D Dsf : Measure (FreeMonoid α)) (k N M : ℕ) (h : SpecHist α)
    (c : ClusterDraws Ω α K N) (hc : ¬ MissAt D k N M h.pops h.prefs c.2.2.1) :
    (clusterDraws D Dsf k N M h c).isSome := by
  classical
  have hall : ∀ j, (popDraws D k N M h.pops h.prefs j (c.2.2.1 j)).isSome := by
    intro j
    rcases j with _ | i
    · rfl
    · exact Option.isSome_iff_ne_none.2 fun he => hc ⟨i, he⟩
  simp only [clusterDraws, allSome, dif_pos hall, Option.map_some, Option.isSome_some]

lemma popDraws_alike {L : ℕ} {pops pops' : List (SpecSampler α)}
    (hp : List.Forall₂ (SamplerAlike L) pops pops') (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (hD : ∀ᵐ x ∂D, x.toList.length ≤ L) (k N M : ℕ)
    (T : Finset (FreeMonoid α)) (j : Option (Fin K))
    (u : ℕ → unitInterval × (Fin N → unitInterval)) :
    popDraws D k N M pops T j u = popDraws D k N M pops' T j u := by
  rcases j with _ | i
  · rfl
  simp only [popDraws]
  have hlen := hp.length_eq
  by_cases hi : i.val < pops.length
  · rw [List.getElem?_eq_getElem hi, List.getElem?_eq_getElem (hlen ▸ hi)]
    have hs := SamplerAlike.of_forall₂ hp hi
    simp only [Random.drawManyFirst]
    have he : ∀ n : Fin M, Random.harvestFirst pops[i.val].read k pops[i.val].hyp pops[i.val].site
        ↑T (List.ofFn fun t : Fin N => drawFrom D ((u n).2 t))
        = Random.harvestFirst (pops'[i.val]'(hlen ▸ hi)).read k (pops'[i.val]'(hlen ▸ hi)).hyp
          (pops'[i.val]'(hlen ▸ hi)).site ↑T (List.ofFn fun t : Fin N => drawFrom D ((u n).2 t)) :=
      fun n => harvestFirst_congr_zone hs k (↑T) _ fun x hx => by
        obtain ⟨t, rfl⟩ := List.mem_ofFn.1 hx; exact drawFrom_length_le D hD _
    simp only [he]
  · rw [List.getElem?_eq_none (not_lt.1 hi), List.getElem?_eq_none (hlen ▸ not_lt.1 hi)]

lemma missAt_alike {L : ℕ} {pops pops' : List (SpecSampler α)}
    (hp : List.Forall₂ (SamplerAlike L) pops pops') (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (hD : ∀ᵐ x ∂D, x.toList.length ≤ L) (k N M : ℕ)
    (T : Finset (FreeMonoid α)) (u : Option (Fin K) → ℕ → unitInterval × (Fin N → unitInterval)) :
    MissAt D k N M pops T u ↔ MissAt D k N M pops' T u := by
  simp only [MissAt, popDraws_alike hp D hD]

omit [DecidableEq α] in
lemma measurableSet_popDraws_none (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N M : ℕ) (pops : List (SpecSampler α)) (T : Finset (FreeMonoid α)) (i : Fin K) :
    MeasurableSet {u : ℕ → unitInterval × (Fin N → unitInterval) |
      popDraws D k N M pops T (some i) u = none} := by
  rcases hs : pops[i.val]? with _ | s
  · have : {u : ℕ → unitInterval × (Fin N → unitInterval) |
        popDraws D k N M pops T (some i) u = none} = ∅ := by
      ext u; simp [popDraws, hs]
    rw [this]; exact MeasurableSet.empty
  · have : {u : ℕ → unitInterval × (Fin N → unitInterval) |
        popDraws D k N M pops T (some i) u = none}
        = probeBlocks D N M ⁻¹'
          {yss | Random.drawManyFirst s.read k s.hyp s.site ↑T yss = none} := by
      ext u; simp only [Set.mem_ofPred_eq, Set.mem_preimage, popDraws_some D k N M pops T i s hs]
    rw [this]
    exact measurable_probeBlocks D N M (Set.to_countable _).measurableSet

omit [DecidableEq α] in
lemma measurableSet_missAt (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N M : ℕ) (pops : List (SpecSampler α)) (T : Finset (FreeMonoid α)) :
    MeasurableSet {u : Option (Fin K) → ℕ → unitInterval × (Fin N → unitInterval) |
      MissAt D k N M pops T u} := by
  have : {u : Option (Fin K) → ℕ → unitInterval × (Fin N → unitInterval) |
      MissAt D k N M pops T u}
      = ⋃ i : Fin K, (fun u => u (some i)) ⁻¹'
        {v | popDraws D k N M pops T (some i) v = none} := by
    ext u; simp [MissAt]
  rw [this]
  exact MeasurableSet.iUnion fun i =>
    measurable_pi_apply _ (measurableSet_popDraws_none D k N M pops T i)

omit [DecidableEq α] in
/-- At the cap `harvestN y (δₘ / M)`, every harvested population yielding at least `y` off `T`,
some harvested population's `M` draws miss with chance at most `K δₘ`. -/
theorem restLaw_missAt_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k M : ℕ)
    (pops : List (SpecSampler α)) (T : Finset (FreeMonoid α)) {y δₘ : ℝ} (hy0 : 0 < y)
    (hy1 : y ≤ 1) (hδ : 0 < δₘ) (hM : 0 < M)
    (hyield : ∀ s ∈ pops, y ≤ (Random.harvestLaw s.read D k s.hyp s.site).real (↑T)ᶜ) :
    (restLaw α K (Random.harvestN y (δₘ / M))).real
        {e | MissAt D k (Random.harvestN y (δₘ / M)) M pops T e.2.1}
      ≤ K * δₘ := by
  classical
  set N := Random.harvestN y (δₘ / M)
  have hmarg : ∀ i : Fin K, Measure.map (fun e : ClusterRest α K N => e.2.1 (some i)) (restLaw α K N)
      = Measure.infinitePi fun _ : ℕ => drawLaw N := by
    intro i
    rw [show (fun e : ClusterRest α K N => e.2.1 (some i))
        = (fun v : Option (Fin K) → ℕ → unitInterval × (Fin N → unitInterval) => v (some i))
          ∘ Prod.fst ∘ Prod.snd from rfl,
      ← Measure.map_map (measurable_pi_apply _) (measurable_fst.comp measurable_snd),
      ← Measure.map_map measurable_fst measurable_snd, restLaw, Measure.map_snd_prod,
      measure_univ, one_smul, Measure.map_fst_prod, measure_univ, one_smul]
    exact (measurePreserving_eval _ (some i)).map_eq
  have hone : ∀ i : Fin K, (restLaw α K N).real
      {e | popDraws D k N M pops T (some i) (e.2.1 (some i)) = none} ≤ δₘ := by
    intro i
    have hm : Measurable fun e : ClusterRest α K N => e.2.1 (some i) :=
      (measurable_pi_apply _).comp (measurable_fst.comp measurable_snd)
    rw [measureReal_def, show {e : ClusterRest α K N |
        popDraws D k N M pops T (some i) (e.2.1 (some i)) = none}
        = (fun e : ClusterRest α K N => e.2.1 (some i)) ⁻¹'
          {v | popDraws D k N M pops T (some i) v = none} from rfl,
      ← Measure.map_apply hm (measurableSet_popDraws_none D k N M pops T i), hmarg i]
    refine ENNReal.toReal_le_of_le_ofReal hδ.le ?_
    rcases hs : pops[i.val]? with _ | s
    · have : {v : ℕ → unitInterval × (Fin N → unitInterval) |
          popDraws D k N M pops T (some i) v = none} = ∅ := by
        ext u; simp [popDraws, hs]
      rw [this, measure_empty]; exact zero_le
    · exact popDraws_miss_le D k M pops T i s hs hy0 hy1 hδ hM
        (hyield s (List.mem_of_getElem? hs))
  have hsub : {e : ClusterRest α K N | MissAt D k N M pops T e.2.1}
      = ⋃ i : Fin K, {e | popDraws D k N M pops T (some i) (e.2.1 (some i)) = none} := by
    ext e; simp [MissAt]
  rw [hsub]
  calc _ ≤ ∑ i : Fin K, (restLaw α K N).real
        {e | popDraws D k N M pops T (some i) (e.2.1 (some i)) = none} :=
        measureReal_iUnion_fintype_le _
    _ ≤ ∑ _i : Fin K, δₘ := Finset.sum_le_sum fun i _ => hone i
    _ = K * δₘ := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end Miss

section Good

variable {μ : Measure Ω} [IsProbabilityMeasure μ]

instance (N : ℕ) : IsProbabilityMeasure (clusterLaw μ α K N) := by
  unfold clusterLaw; infer_instance

omit [IsProbabilityMeasure μ] in
lemma card_loopPopulations_le (K r : ℕ) : (loopPopulations K r).card ≤ K + 1 := by
  unfold loopPopulations
  refine (Finset.card_insert_le _ _).trans (Nat.add_le_add_right ?_ 1)
  exact Finset.card_image_le.trans ((Finset.card_filter_le _ _).trans (by simp))

omit [IsProbabilityMeasure μ] in
/-- No harvested draw missing, no population's sides falling short and a scheduled state
returning, the round is good. -/
lemma roundGood_of (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : SpecHist α)
    (d : LoopRoundDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget)
    {T A NP : ℕ}
    (hNP : ∀ B ∈ settings.schedule (loopPopulations K h.pops.length), B.npref ≤ NP)
    (hNA : (2 + (loopPopulations K h.pops.length).card) * NP ≤ A) (hAT : A ≤ T)
    (h1 : ¬ MissAt D settings.level3.walkStart settings.level3.harvestReadCap settings.scanCap
      h.pops h.prefs d.2.1.2.1)
    (h2 : ¬ ScanShort (loopPopulations K h.pops.length) settings.scanCap T A
      (idealDraws D Dsf settings.level3.walkStart settings.level3.harvestReadCap h (d.1, d.2.1)))
    (h3 : ∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
      idealRun D Dsf settings.level3.walkStart settings.level3.harvestReadCap h (d.1, d.2.1)
        ∈ ret O.mq (loopPopulations K h.pops.length) none settings.η₀ settings.indecisionLimit
          settings.εcov settings.gateLevel B) :
    RoundGood settings O D Dsf h d :=
  ⟨clusterDraws_isSome D Dsf _ _ _ h (d.1, d.2.1) h1,
    scanWithin_of_not_scanShort _ _ hNP hNA hAT _ h2, h3⟩

omit [IsProbabilityMeasure μ] [DecidableEq α] in
/-- Falling short reads no noise. -/
lemma scanShort_noise (pops : Finset (Option (Fin K))) (M T A : ℕ) (ω ω' : Ω)
    (s s' : ((ℕ → FreeMonoid α) × (Option (Fin K) → ℕ → FreeMonoid α))
      × (Option (Fin K) → ℕ → FreeMonoid α)) (g g' : FreeMonoid α → Bool) (hs : s = s')
    (hg : g = g') : ScanShort pops M T A ((ω, s), g) ↔ ScanShort pops M T A ((ω', s'), g') := by
  subst hs hg
  unfold ScanShort
  refine exists_congr fun j => and_congr_right fun _ => ?_
  rcases j with _ | i <;> exact Iff.rfl

/-- A harvested population yields at least `(1 - m) ν(univ)` off `T`, `T` holding at most `m`
of it. -/
lemma harvestLaw_compl_ge (s : SpecSampler α) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (k : ℕ) (T : Finset (FreeMonoid α)) {y m : ℝ} (hy : 0 < y)
    (hmass : 2 * y ≤ (Random.harvestLaw s.read D k s.hyp s.site).real Set.univ) (hm : m ≤ 1 / 2)
    (hT : (popLaw (Random.harvestLaw s.read D k s.hyp s.site)).real ↑T ≤ m) :
    y ≤ (Random.harvestLaw s.read D k s.hyp s.site).real (↑T)ᶜ := by
  set ν := Random.harvestLaw s.read D k s.hyp s.site
  have htop : ν Set.univ ≠ ⊤ := Random.harvestLaw_univ_ne_top _ D k _ _
  haveI : IsFiniteMeasure ν := ⟨lt_top_iff_ne_top.2 htop⟩
  have hpos : 0 < ν.real Set.univ := by linarith
  have h0 : ν Set.univ ≠ 0 := fun h0 => by
    rw [measureReal_def, h0, ENNReal.toReal_zero] at hpos; exact lt_irrefl _ hpos
  rw [popLaw_real h0 htop, div_le_iff₀ hpos] at hT
  rw [measureReal_compl (Finset.measurableSet _)]
  nlinarith

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
  (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]

/-- Round `r`'s draws, after any history the earlier rounds hand on whose populations hold at
most `cap` on a string, yield at least `2y` before the earlier prefixes and hold at most `m` of
them, miss, fall short or walk more than `B` probes in all with chance at most

    K δₘ + (K + 1) (2 C(M, T) (2 T cap)^(M-T) + 2 exp(-T/8)) + 2^(K M) (1 - y/2)^(B + 1),

at the cap `harvestN y (δₘ / M)`, `M` the scan cap and `4 (K + 3) NP ≤ T ≤ M`. -/
theorem loop_good_round_le_persist (cap m y δₘ : ℝ) {T NP : ℕ} (B : ℕ)
    (hcap : settings.level3.harvestReadCap = Random.harvestN y (δₘ / settings.scanCap))
    (hy0 : 0 < y) (hy1 : y ≤ 1) (hδ : 0 < δₘ) (hM : 0 < settings.scanCap) (hm : m ≤ 1 / 2)
    (hA : 4 * ((K + 3) * NP) ≤ T) (hT : T ≤ settings.scanCap)
    (hDat : ∀ t, P.D.real {t} ≤ cap) (r : ℕ) (hr : r < K) :
    (loopMeasure μ α K settings.level3.harvestReadCap settings.level3.probeBudget).real
      {p | HistOK settings O P.D Dsf p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf p.1 p.2 r).length = r
        ∧ (∀ ν' ∈ samplerLaws P.D settings.level3.walkStart (decodeHist
            (histCodes settings O P.D Dsf p.1 p.2 r)).pops, ∀ t, ν'.real {t} ≤ cap)
        ∧ (∀ s ∈ (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)).pops,
            2 * y ≤ (Random.harvestLaw s.read P.D settings.level3.walkStart s.hyp s.site).real
              Set.univ)
        ∧ (∀ j ∈ loopPopulations K r, (loopDistributions P.D settings.level3.walkStart (decodeHist
            (histCodes settings O P.D Dsf p.1 p.2 r)).pops j).real
              ↑(decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)).prefs ≤ m)
        ∧ (MissAt P.D settings.level3.walkStart settings.level3.harvestReadCap settings.scanCap
            (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)).pops
            (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)).prefs (p.2 ⟨r, hr⟩).1.2.1
          ∨ ScanShort (loopPopulations K r) settings.scanCap T ((K + 3) * NP)
            (idealDraws P.D Dsf settings.level3.walkStart settings.level3.harvestReadCap
              (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)) (p.1, (p.2 ⟨r, hr⟩).1))
          ∨ B < harvestWalk P.D settings.level3.walkStart settings.level3.harvestReadCap
            settings.scanCap (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)).pops
            (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)).prefs (p.2 ⟨r, hr⟩).1.2.1)}
      ≤ K * δₘ + (K + 1) * (2 * (settings.scanCap.choose T * (T * (2 * cap)) ^ (settings.scanCap - T))
          + 2 * Real.exp (-(T / 8 : ℝ))) + 2 ^ (K * settings.scanCap) * (1 - y / 2) ^ (B + 1) := by
  classical
  set k := settings.level3.walkStart
  set N := settings.level3.harvestReadCap
  set M := settings.scanCap
  set A := (K + 3) * NP
  set PB := settings.level3.probeBudget
  set dec := decodeHist (α := α) with hdec
  set b : ℝ := K * δₘ + (K + 1) * (2 * (M.choose T * (T * (2 * cap)) ^ (M - T))
    + 2 * Real.exp (-(T / 8 : ℝ))) + 2 ^ (K * M) * (1 - y / 2) ^ (B + 1)
  have hcap0 : 0 ≤ cap := le_trans measureReal_nonneg (hDat 1)
  have hy2 : 0 ≤ 1 - y / 2 := by linarith
  have hb0 : 0 ≤ b := add_nonneg (by positivity) (mul_nonneg (by positivity) (pow_nonneg hy2 _))
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
    ∧ (∀ ν' ∈ samplerLaws P.D k (dec c).pops, ∀ t, ν'.real {t} ≤ cap)
    ∧ (∀ s ∈ (dec c).pops, 2 * y ≤ (Random.harvestLaw s.read P.D k s.hyp s.site).real Set.univ)
    ∧ (∀ j ∈ loopPopulations K r, (loopDistributions P.D k (dec c).pops j).real ↑(dec c).prefs ≤ m)
  set Bad : Y → Set (Ω × RoundDraws α K N PB) := fun y => match y with
    | some c => {q | Good c ∧ (MissAt P.D k N M (dec c).pops (dec c).prefs q.2.1.2.1
        ∨ ScanShort (loopPopulations K r) M T A (idealDraws P.D Dsf k N (dec c) (q.1, q.2.1))
        ∨ B < harvestWalk P.D k N M (dec c).pops (dec c).prefs q.2.1.2.1)}
    | none => ∅
  have hmarg : MeasurePreserving (fun q : Ω × RoundDraws α K N PB =>
      ((q.1, q.2.1) : ClusterDraws Ω α K N)) (μ.prod ν) (clusterLaw μ α K N) :=
    (MeasurePreserving.id μ).prod measurePreserving_fst
  have hmid : ∀ c : List (HarvestCode α), Measurable
      fun q : Ω × RoundDraws α K N PB => idealDraws P.D Dsf k N (dec c) (q.1, q.2.1) := fun c =>
    (measurable_idealDraws P.D Dsf k N (dec c)).comp hmarg.measurable
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
      · exact Set.eq_empty_of_forall_notMem fun q hq => hy ⟨c, rfl, hq.1.1⟩)
    (fun y => by
      rcases y with _ | c
      · exact MeasurableSet.empty
      · exact (MeasurableSet.const _).inter
          ((measurable_fst.comp (measurable_snd.comp (measurable_fst.comp measurable_snd))
            (measurableSet_missAt P.D k N M (dec c).pops (dec c).prefs)).union
          ((hmid c (measurableSet_scanShort (loopPopulations K r) M T A)).union
            (((measurable_harvestWalk P.D k N M (dec c).pops (dec c).prefs).comp
              (measurable_fst.comp (measurable_snd.comp (measurable_fst.comp measurable_snd))))
              (MeasurableSet.of_discrete (s := Set.Ioi B))))))
    (fun _ _ => ∅) (fun y d ω ω' _ hq => by
      rcases y with _ | c
      · exact hq
      · exact ⟨hq.1, hq.2.imp id fun h' =>
          h'.imp (fun h' => (scanShort_noise _ _ _ _ _ _ _ _ _ _ rfl rfl).1 h') id⟩)
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
        obtain ⟨hcr, hpg, hyl, hpm⟩ := hH
        set h := dec c
        have hlen : h.pops.length = r := by
          simp only [h, dec, decodeHist, List.length_map, hcr]
        -- each population off the earlier prefixes holds at most `2 cap` on a string
        have hat : ∀ j ∈ loopPopulations K r, ∀ t,
            (offLaw (loopDistributions P.D k h.pops j) h.prefs).real {t} ≤ 2 * cap := by
          intro j hj t
          refine (offLaw_atom_le ((hpm j hj).trans hm) t).trans ?_
          have : (loopDistributions P.D k h.pops j).real {t} ≤ cap := by
            cases j with
            | none => exact hDat t
            | some i =>
              have hi : i.val < (samplerLaws P.D k h.pops).length := by
                rw [samplerLaws_length, hlen]; exact lt_of_some_mem_loopPops hj
              rw [loopDistributions_eq_popDists]
              simp only [popDists]
              rw [List.getD_eq_getElem (hn := hi)]
              exact hpg _ (List.getElem_mem _) t
          linarith
        -- each harvested population yields at least `y` off the earlier prefixes
        have hyield : ∀ s ∈ h.pops,
            y ≤ (Random.harvestLaw s.read P.D k s.hyp s.site).real (↑h.prefs)ᶜ := by
          intro s hs
          obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hs
          have hiK : i < K := by omega
          have hmem : some (⟨i, hiK⟩ : Fin K) ∈ loopPopulations K r := by
            simp only [loopPopulations, Finset.mem_insert, reduceCtorEq, false_or,
              Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and, Option.some.injEq]
            exact ⟨⟨i, hiK⟩, by simp; omega, rfl⟩
          have hT := hpm _ hmem
          simp only [loopDistributions, List.getElem?_eq_getElem hi] at hT
          exact harvestLaw_compl_ge _ P.D k h.prefs hy0 (hyl _ (List.getElem_mem _)) hm hT
        have hmiss := restLaw_missAt_le (K := K) P.D k M h.pops h.prefs hy0 hy1 hδ hM hyield
        rw [← hcap] at hmiss
        have hscan := clusterLaw_scanShort_le (μ := μ) P.D Dsf k N h (loopPopulations K r)
          (M := M) (T := T) (A := A) hat hA hT
        have hcard : ((loopPopulations K r).card : ℝ) ≤ K + 1 := by
          exact_mod_cast card_loopPopulations_le K r
        have hmissE : (μ.prod ν) {q : Ω × RoundDraws α K N PB |
            MissAt P.D k N M h.pops h.prefs q.2.1.2.1}
            = restLaw α K N {e | MissAt P.D k N M h.pops h.prefs e.2.1} := by
          have hp : MeasurePreserving (fun q : Ω × RoundDraws α K N PB => q.2.1) (μ.prod ν)
              (restLaw α K N) := measurePreserving_fst.comp measurePreserving_snd
          exact hp.measure_preimage (((measurable_fst.comp measurable_snd) (measurableSet_missAt
            P.D k N M h.pops h.prefs)).nullMeasurableSet)
        have hscanE : (μ.prod ν) {q : Ω × RoundDraws α K N PB |
            ScanShort (loopPopulations K r) M T A (idealDraws P.D Dsf k N h (q.1, q.2.1))}
            = clusterLaw μ α K N {c | ScanShort (loopPopulations K r) M T A
              (idealDraws P.D Dsf k N h c)} :=
          hmarg.measure_preimage (((measurable_idealDraws P.D Dsf k N h)
            (measurableSet_scanShort (loopPopulations K r) M T A)).nullMeasurableSet)
        have hwalk := restLaw_harvestWalk_le (K := K) P.D k N M h.pops h.prefs hy0 hy1 hyield B
        have hwalkE : (μ.prod ν) {q : Ω × RoundDraws α K N PB |
            B < harvestWalk P.D k N M h.pops h.prefs q.2.1.2.1}
            = restLaw α K N {e | B < harvestWalk P.D k N M h.pops h.prefs e.2.1} := by
          have hp : MeasurePreserving (fun q : Ω × RoundDraws α K N PB => q.2.1) (μ.prod ν)
              (restLaw α K N) := measurePreserving_fst.comp measurePreserving_snd
          exact hp.measure_preimage ((((measurable_harvestWalk P.D k N M h.pops h.prefs).comp
            (measurable_fst.comp measurable_snd)) (MeasurableSet.of_discrete
              (s := Set.Ioi B))).nullMeasurableSet)
        calc (μ.prod ν) (Bad (some c))
            ≤ (μ.prod ν) {q : Ω × RoundDraws α K N PB |
                MissAt P.D k N M h.pops h.prefs q.2.1.2.1}
              + ((μ.prod ν) {q : Ω × RoundDraws α K N PB |
                ScanShort (loopPopulations K r) M T A (idealDraws P.D Dsf k N h (q.1, q.2.1))}
              + (μ.prod ν) {q : Ω × RoundDraws α K N PB |
                B < harvestWalk P.D k N M h.pops h.prefs q.2.1.2.1}) :=
              (measure_mono fun q hq => hq.2).trans ((measure_union_le _ _).trans
                (add_le_add le_rfl (measure_union_le _ _)))
          _ = ENNReal.ofReal ((restLaw α K N).real {e | MissAt P.D k N M h.pops h.prefs e.2.1})
              + (ENNReal.ofReal ((clusterLaw μ α K N).real {c | ScanShort (loopPopulations K r) M T A
                (idealDraws P.D Dsf k N h c)})
              + ENNReal.ofReal ((restLaw α K N).real
                {e | B < harvestWalk P.D k N M h.pops h.prefs e.2.1})) := by
              rw [hmissE, hscanE, hwalkE, ofReal_measureReal (measure_ne_top _ _),
                ofReal_measureReal (measure_ne_top _ _), ofReal_measureReal (measure_ne_top _ _)]
          _ ≤ ENNReal.ofReal b := by
              rw [← ENNReal.ofReal_add measureReal_nonneg measureReal_nonneg,
                ← ENNReal.ofReal_add measureReal_nonneg
                  (add_nonneg measureReal_nonneg measureReal_nonneg)]
              refine ENNReal.ofReal_le_ofReal ?_
              have hq0 : 0 ≤ 2 * (M.choose T * (T * (2 * cap)) ^ (M - T))
                  + 2 * Real.exp (-(T / 8 : ℝ)) := by positivity
              have := mul_le_mul_of_nonneg_right hcard hq0
              linarith)
  refine ENNReal.toReal_le_of_le_ofReal hb0 ((measure_mono ?_).trans hmain)
  rintro p ⟨hok, hlen, hpg, hyl, hpm, hbad⟩
  have hst : stageH settings O P.D Dsf r p.1 p.2
      = some (histCodes settings O P.D Dsf p.1 p.2 r) := by
    simp only [stageH, if_pos hok]
  show (p.1, p.2 ⟨r, hr⟩) ∈ Bad (stageH settings O P.D Dsf r p.1 p.2)
    ∧ p.2 ∈ (fun _ : Y => (Set.univ : Set (Fin K → RoundDraws α K N PB)))
      (stageH settings O P.D Dsf r p.1 p.2)
  rw [hst]
  exact ⟨⟨⟨hlen, hpg, hyl, hpm⟩, hbad⟩, Set.mem_univ _⟩

omit [IsProbabilityMeasure μ] [IsProbabilityMeasure Dsf] in
/-- Where every round the filled draws run is good, the loop runs those rounds. -/
theorem idealRounds_mem_loop (D : Measure (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α K settings.level3.harvestReadCap
      settings.level3.probeBudget)
    (hg : ∀ r (hr : r < K), AllHarvested (idealRounds settings O D Dsf draws r) →
      RoundGood settings O D Dsf (specHistOf (idealRounds settings O D Dsf draws r)) (draws ⟨r, hr⟩)) :
    loopRounds settings O D Dsf draws K = some (idealRounds settings O D Dsf draws K) :=
  loopRounds_eq_some settings O D Dsf draws hg K

end Good

section Reads

variable {μ : Measure Ω} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
  (D Dsf : Measure (FreeMonoid α))
  (draws : Fin K → LoopRoundDraws Ω α K settings.level3.harvestReadCap
    settings.level3.probeBudget)

/-- Each round the loop runs is a round after some history and clustering run. -/
lemma loopRounds_roundOf (n : ℕ) : ∀ rounds, loopRounds settings O D Dsf draws n = some rounds →
    rounds.length ≤ n ∧ ∀ o ∈ rounds, ∃ (h : SpecHist α) (x : Run Ω (FreeMonoid α) (Option (Fin K)))
      (u : Fin settings.level3.probeBudget → unitInterval), o = roundOf settings O D h x u := by
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
        obtain ⟨dr, -, hdr⟩ := Option.mem_bind_iff.1 ho
        split_ifs at hdr
        · exact ⟨_, _, _, (Option.mem_some_iff.1 hdr).symm⟩
        · cases hdr
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

section Walk

/-- Round `r`'s harvested draws' probes in all. -/
def roundWalk {μ : Measure Ω}
    (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
    (p : Ω × (Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget))
    (r : ℕ) (hr : r < K) : ℕ :=
  harvestWalk D settings.level3.walkStart settings.level3.harvestReadCap settings.scanCap
    (specHistOf (idealRounds settings O D Dsf (persistDraws p) r)).pops
    (specHistOf (idealRounds settings O D Dsf (persistDraws p) r)).prefs
    (persistDraws p ⟨r, hr⟩).2.1.2.1

/-- Every round whose harvested draws all land walks at most `B` probes in all. -/
def RoundsWalkWithin {μ : Measure Ω}
    (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
    (p : Ω × (Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget))
    (B : ℕ) : Prop :=
  ∀ r (hr : r < K), AllHarvested (idealRounds settings O D Dsf (persistDraws p) r) →
    roundWalk settings O D Dsf p r hr ≤ B

end Walk

end OrthoDFA
