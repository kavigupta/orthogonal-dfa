import RLStar.Proofs.LoopColl
import RLStar.Proofs.ScanTail

/-!
# A round's draws hit and scan within the cap

A round is good (`RoundGood`) where its harvested draws all hit (`PopsHit`), its filled draws scan
within `scanCap`, and its clustering returns at a scheduled state. The draws miss with chance at
most `K δₘ` at the cap `harvestN y (δₘ / M)`, each harvested population yielding at least `y` off
the earlier rounds' prefixes (`restLaw_popsHit_le`); they scan short with the chance of
`clusterLaw_scanShort_le`. Both are charged in one round's draws after any history the earlier
rounds hand on (`loop_good_round_le_persist`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {K : ℕ}


section Miss

omit [DecidableEq α] in
lemma measurableSet_popsHit (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k N M : ℕ)
    (pops : List (SpecSampler α)) (T : Finset (FreeMonoid α)) :
    MeasurableSet {c : ClusterRest α K N | PopsHit D k N M pops T c} := by
  classical
  have hu : ∀ i : Fin K, ∀ n, MeasurableSet {c : ClusterRest α K N |
      (popHit D k N pops T (some i) (c.2.1 (some i) n)).isSome} := by
    intro i n
    rcases hs : pops[i.val]? with _ | s
    · have : {c : ClusterRest α K N | (popHit D k N pops T (some i) (c.2.1 (some i) n)).isSome}
          = ∅ := by ext c; simp [popHit, hs]
      rw [this]; exact MeasurableSet.empty
    · have : {c : ClusterRest α K N | (popHit D k N pops T (some i) (c.2.1 (some i) n)).isSome}
          = (fun c : ClusterRest α K N => fun t => drawFrom D ((c.2.1 (some i) n).2 t)) ⁻¹'
            {xs | (Random.lazyHarvestFirst s.read k s.trie s.hyp s.site T
              (List.ofFn xs)).isSome} := by
        ext c; simp [popHit, hs]
      rw [this]
      exact (measurable_pi_lambda _ fun t => (measurable_drawFrom D).comp
        ((measurable_pi_apply t).comp (measurable_snd.comp ((measurable_pi_apply n).comp
          ((measurable_pi_apply (some i)).comp (measurable_fst.comp measurable_snd))))))
        (Set.to_countable _).measurableSet
  have e : {c : ClusterRest α K N | PopsHit D k N M pops T c}
      = ⋂ i : Fin K, ⋂ n : ℕ, ({_c : ClusterRest α K N | i.val < pops.length ∧ n < M}ᶜ
        ∪ {c | (popHit D k N pops T (some i) (c.2.1 (some i) n)).isSome}) := by
    ext c
    simp only [PopsHit, Set.mem_ofPred_eq, Set.mem_iInter, Set.mem_union, Set.mem_compl_iff,
      not_and]
    exact ⟨fun h i n => or_iff_not_imp_left.2 fun h' =>
        h i (not_not.1 fun hi => h' fun hi' => absurd hi' hi) n
          (not_not.1 fun hn => h' fun _ hn' => absurd hn' hn),
      fun h i hi n hn => (h i n).resolve_left fun h' => h' hi hn⟩
  rw [e]
  exact MeasurableSet.iInter fun i => MeasurableSet.iInter fun n =>
    (MeasurableSet.const _).compl.union (hu i n)

omit [DecidableEq α] in
/-- A draw's sampler over `N = harvestN y δ` probes, yielding at least `y` off `T`, misses with
chance at most `δ`. -/
lemma drawLaw_lazyHarvestFirst_miss_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N : ℕ) (s : SpecSampler α) (T : Finset (FreeMonoid α)) {y δ : ℝ} (hy0 : 0 < y)
    (hy1 : y ≤ 1) (hδ : 0 < δ) (hN : N = Random.harvestN y δ)
    (hyield : y ≤ (Random.lazyHarvestLaw s.read D k s.trie s.hyp s.site).real (↑T)ᶜ) :
    drawLaw N {u | Random.lazyHarvestFirst s.read k s.trie s.hyp s.site T
      (List.ofFn fun t => drawFrom D (u.2 t)) = none} ≤ ENNReal.ofReal δ := by
  classical
  subst hN
  set N := Random.harvestN y δ
  set D' := D.map (Random.padIn k s.trie)
  have hpad : MeasurePreserving (fun v : unitInterval => Random.padIn k s.trie (drawFrom D v))
      volume D' :=
    (⟨measurable_of_countable _, rfl⟩ : MeasurePreserving (Random.padIn k s.trie) D D').comp
      ⟨measurable_drawFrom _, map_drawFrom _⟩
  have hf : MeasurePreserving (fun u : unitInterval × (Fin N → unitInterval) =>
      fun t => Random.padIn k s.trie (drawFrom D (u.2 t))) (drawLaw N)
      (Measure.pi fun _ : Fin N => D') :=
    (measurePreserving_pi _ _ fun _ => hpad).comp measurePreserving_snd
  have hset : {u : unitInterval × (Fin N → unitInterval) | Random.lazyHarvestFirst s.read k s.trie
      s.hyp s.site T (List.ofFn fun t => drawFrom D (u.2 t)) = none}
      = (fun u : unitInterval × (Fin N → unitInterval) =>
        fun t => Random.padIn k s.trie (drawFrom D (u.2 t))) ⁻¹'
        {xs | Random.harvestFirst (Random.padRead s.read) k s.hyp s.site (Random.unpad ⁻¹' ↑T)
          (List.ofFn xs) = none} := by
    ext u
    simp [Random.lazyHarvestFirst, List.map_ofFn, Function.comp_def]
  rw [hset, hf.measure_preimage (Set.to_countable _).measurableSet.nullMeasurableSet]
  refine Random.harvestCodeLaw_miss_le (Random.padRead s.read) D' k s.hyp s.site
    (Random.unpad ⁻¹' ↑T) hy0 hy1 hδ ?_
  have e : (Random.lazyHarvestLaw s.read D k s.trie s.hyp s.site).real (↑T)ᶜ
      = (Random.harvestLaw (Random.padRead s.read) D' k s.hyp s.site).real
        (Random.unpad ⁻¹' ↑T)ᶜ := by
    unfold Random.lazyHarvestLaw
    rw [map_measureReal_apply (measurable_of_countable _) (Set.to_countable _).measurableSet,
      Set.preimage_compl]
  rw [← e]
  exact hyield

omit [DecidableEq α] in
/-- At the cap `N = harvestN y (δₘ / M)`, every harvested population yielding at least `y` off
`T`, some harvested population's `M` draws miss with chance at most `K δₘ`. -/
theorem restLaw_popsHit_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k N M : ℕ)
    (pops : List (SpecSampler α)) (T : Finset (FreeMonoid α)) {y δₘ : ℝ} (hy0 : 0 < y)
    (hy1 : y ≤ 1) (hδ : 0 < δₘ) (hM : 0 < M) (hN : N = Random.harvestN y (δₘ / M))
    (hyield : ∀ s ∈ pops, y ≤ (Random.lazyHarvestLaw s.read D k s.trie s.hyp s.site).real (↑T)ᶜ) :
    (restLaw α K N).real {c | ¬ PopsHit D k N M pops T c} ≤ K * δₘ := by
  classical
  have hδM : 0 < δₘ / M := div_pos hδ (by exact_mod_cast hM)
  have hmarg : ∀ (i : Fin K) (n : ℕ), MeasurePreserving
      (fun c : ClusterRest α K N => c.2.1 (some i) n) (restLaw α K N) (drawLaw N) := by
    intro i n
    have h1 : MeasurePreserving (fun c : ClusterRest α K N => c.2.1)
        (restLaw α K N) (Measure.pi fun _ : Option (Fin K) =>
          Measure.infinitePi fun _ : ℕ => drawLaw N) :=
      measurePreserving_fst.comp measurePreserving_snd
    have h2 : MeasurePreserving (fun f : ℕ → unitInterval × (Fin N → unitInterval) => f n)
        (Measure.infinitePi fun _ : ℕ => drawLaw N) (drawLaw N) :=
      measurePreserving_eval_infinitePi (fun _ : ℕ => drawLaw N) n
    exact (h2.comp (measurePreserving_eval _ (some i))).comp h1
  have hone : ∀ (i : Fin K) (n : ℕ), (restLaw α K N).real
      {c | i.val < pops.length ∧ popHit D k N pops T (some i) (c.2.1 (some i) n) = none}
      ≤ δₘ / M := by
    intro i n
    refine ENNReal.toReal_le_of_le_ofReal hδM.le ?_
    rcases hs : pops[i.val]? with _ | s
    · have : {c : ClusterRest α K N | i.val < pops.length
          ∧ popHit D k N pops T (some i) (c.2.1 (some i) n) = none} = ∅ := by
        ext c
        simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_and]
        intro hi
        rw [List.getElem?_eq_getElem hi] at hs
        cases hs
      rw [this, measure_empty]; exact bot_le
    · have hsub : {c : ClusterRest α K N | i.val < pops.length
          ∧ popHit D k N pops T (some i) (c.2.1 (some i) n) = none}
          ⊆ (fun c : ClusterRest α K N => c.2.1 (some i) n) ⁻¹'
            {u | Random.lazyHarvestFirst s.read k s.trie s.hyp s.site T
              (List.ofFn fun t => drawFrom D (u.2 t)) = none} := by
        intro c hc
        simpa [popHit, hs] using hc.2
      refine (measure_mono hsub).trans ?_
      have hU : MeasurableSet {u : unitInterval × (Fin N → unitInterval) |
          Random.lazyHarvestFirst s.read k s.trie s.hyp s.site T
            (List.ofFn fun t => drawFrom D (u.2 t)) = none} :=
        (measurable_pi_lambda _ fun t => (measurable_drawFrom D).comp
          ((measurable_pi_apply t).comp measurable_snd))
          (Set.to_countable {xs : Fin N → FreeMonoid α | Random.lazyHarvestFirst s.read k s.trie
            s.hyp s.site T (List.ofFn xs) = none}).measurableSet
      rw [(hmarg i n).measure_preimage hU.nullMeasurableSet]
      exact drawLaw_lazyHarvestFirst_miss_le D k N s T hy0 hy1 hδM hN
        (hyield s (List.mem_of_getElem? hs))
  have hsub : {c : ClusterRest α K N | ¬ PopsHit D k N M pops T c}
      ⊆ ⋃ i : Fin K, ⋃ n ∈ Finset.range M,
        {c | i.val < pops.length ∧ popHit D k N pops T (some i) (c.2.1 (some i) n) = none} := by
    intro c hc
    simp only [PopsHit, not_forall] at hc
    obtain ⟨i, hi, n, hn, hnone⟩ := hc
    simp only [Set.mem_iUnion, Finset.mem_range]
    exact ⟨i, n, hn, hi, by simpa using hnone⟩
  calc _ ≤ (restLaw α K N).real (⋃ i : Fin K, ⋃ n ∈ Finset.range M,
        {c | i.val < pops.length ∧ popHit D k N pops T (some i) (c.2.1 (some i) n) = none}) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ i : Fin K, (restLaw α K N).real (⋃ n ∈ Finset.range M,
        {c | i.val < pops.length ∧ popHit D k N pops T (some i) (c.2.1 (some i) n) = none}) :=
        measureReal_iUnion_fintype_le _
    _ ≤ ∑ _i : Fin K, ∑ _n ∈ Finset.range M, δₘ / M := Finset.sum_le_sum fun i _ =>
        (measureReal_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun n _ => hone i n)
    _ = K * δₘ := by
        have hM' : (M : ℝ) ≠ 0 := by exact_mod_cast hM.ne'
        simp only [Finset.sum_const, Finset.card_range, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul]
        field_simp

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

omit [IsProbabilityMeasure μ] [DecidableEq α] in
/-- No harvested draw missing, no population's sides falling short and a scheduled state
returning, the round is good. -/
lemma roundGood_of (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : SpecHist α)
    (d : LoopRoundDraws Ω α settings)
    {T A NP : ℕ}
    (hNP : ∀ B ∈ settings.schedule (loopPopulations K h.pops.length), B.npref ≤ NP)
    (hNA : (2 + (loopPopulations K h.pops.length).card) * NP ≤ A) (hAT : A ≤ T)
    (h1 : PopsHit D settings.level3.walkStart settings.level3.harvestReadCap settings.scanCap
      h.pops h.prefs d.2.1)
    (h2 : ¬ ScanShort (loopPopulations K h.pops.length) settings.scanCap T A
      (idealDraws D Dsf settings.level3.walkStart settings.level3.harvestReadCap h (d.1, d.2.1)))
    (h3 : ∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
      idealRun D Dsf settings.level3.walkStart settings.level3.harvestReadCap h (d.1, d.2.1)
        ∈ ret O.mq (loopPopulations K h.pops.length) none settings.η₀ settings.indecisionLimit
          settings.εcov settings.gateLevel B) :
    RoundGood settings O D Dsf h d :=
  ⟨h1, scanWithin_of_not_scanShort _ _ hNP hNA hAT _ h2, h3⟩

omit [IsProbabilityMeasure μ] in
/-- A harvested population yields at least `y` off `T`, where its law yields `2y` and `T` holds at
most `m ≤ 1/2` of it as a probability. -/
lemma lazyHarvestLaw_compl_ge (s : SpecSampler α) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (k : ℕ) (T : Finset (FreeMonoid α)) {y m : ℝ} (hy : 0 < y)
    (hmass : 2 * y ≤ (Random.lazyHarvestLaw s.read D k s.trie s.hyp s.site).real Set.univ)
    (hm : m ≤ 1 / 2)
    (hT : (popLaw (Random.lazyHarvestLaw s.read D k s.trie s.hyp s.site)).real ↑T ≤ m) :
    y ≤ (Random.lazyHarvestLaw s.read D k s.trie s.hyp s.site).real (↑T)ᶜ := by
  set ν := Random.lazyHarvestLaw s.read D k s.trie s.hyp s.site
  have htop : ν Set.univ ≠ ⊤ := lazyHarvestLaw_univ_ne_top _ D k _ _ _
  have hpos : 0 < ν.real Set.univ := by linarith
  have h0 : ν Set.univ ≠ 0 := fun h0 => by
    rw [measureReal_def, h0, ENNReal.toReal_zero] at hpos; exact lt_irrefl _ hpos
  rw [popLaw_real h0 htop, div_le_iff₀ hpos] at hT
  rw [measureReal_compl (Finset.measurableSet _)]
  nlinarith

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
most `cap` on a string and at most `m` of the earlier prefixes, and whose samplers each yield at
least `ys` off those prefixes, miss a harvested draw or fall short within the scan cap `M` with
chance at most

    K δₘ + (K + 1) (2 C(M, T) (2 T cap)^(M-T) + 2 exp(-T/8)),

where `4 (K + 3) NP ≤ T ≤ M` and the sampler's cap is `harvestN ys (δₘ / M)`. -/
theorem loop_good_round_le_persist (cap m ys δₘ : ℝ) {T NP : ℕ} (hm : m ≤ 1 / 2)
    (hA : 4 * ((K + 3) * NP) ≤ T) (hT : T ≤ settings.scanCap) (hDat : ∀ t, P.D.real {t} ≤ cap)
    (hN : settings.level3.harvestReadCap = Random.harvestN ys (δₘ / settings.scanCap))
    (hy0 : 0 < ys) (hy1 : ys ≤ 1) (hδₘ : 0 < δₘ) (hM : 0 < settings.scanCap)
    (r : ℕ) (hr : r < K) :
    (settings.measure μ α).real
      {p | HistOK settings O P.D Dsf p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf p.1 p.2 r).length = r
        ∧ (∀ ν' ∈ samplerLaws P.D settings.level3.walkStart (decodeHist
            (histCodes settings O P.D Dsf p.1 p.2 r)).pops, ∀ t, ν'.real {t} ≤ cap)
        ∧ (∀ j ∈ loopPopulations K r, (loopDistributions P.D settings.level3.walkStart (decodeHist
            (histCodes settings O P.D Dsf p.1 p.2 r)).pops j).real
              ↑(decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)).prefs ≤ m)
        ∧ (∀ s ∈ (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)).pops,
            ys ≤ (Random.lazyHarvestLaw s.read P.D settings.level3.walkStart s.trie s.hyp
              s.site).real (↑(decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)).prefs)ᶜ)
        ∧ (¬ PopsHit P.D settings.level3.walkStart settings.level3.harvestReadCap
              settings.scanCap (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)).pops
              (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)).prefs (p.2 ⟨r, hr⟩).1
          ∨ ScanShort (loopPopulations K r) settings.scanCap T ((K + 3) * NP)
            (idealDraws P.D Dsf settings.level3.walkStart settings.level3.harvestReadCap
              (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)) (p.1, (p.2 ⟨r, hr⟩).1)))}
      ≤ K * δₘ + (K + 1) * (2 * (settings.scanCap.choose T * (T * (2 * cap))
          ^ (settings.scanCap - T)) + 2 * Real.exp (-(T / 8 : ℝ))) := by
  classical
  set M := settings.scanCap
  set A := (K + 3) * NP
  set k := settings.level3.walkStart
  set N := settings.level3.harvestReadCap
  set dec := decodeHist (α := α) with hdec
  set b : ℝ := K * δₘ + (K + 1) * (2 * (M.choose T * (T * (2 * cap)) ^ (M - T))
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
    ∧ (∀ ν' ∈ samplerLaws P.D k (dec c).pops, ∀ t, ν'.real {t} ≤ cap)
    ∧ (∀ j ∈ loopPopulations K r, (loopDistributions P.D k (dec c).pops j).real ↑(dec c).prefs
      ≤ m)
    ∧ (∀ s ∈ (dec c).pops,
      ys ≤ (Random.lazyHarvestLaw s.read P.D k s.trie s.hyp s.site).real (↑(dec c).prefs)ᶜ)
  set Bad : Y → Set (Ω × settings.RoundDraws α) := fun y => match y with
    | some c => {q | Good c
        ∧ (¬ PopsHit P.D k N M (dec c).pops (dec c).prefs q.2.1
          ∨ ScanShort (loopPopulations K r) M T A (idealDraws P.D Dsf k N (dec c) (q.1, q.2.1)))}
    | none => ∅
  have hmarg : MeasurePreserving (fun q : Ω × settings.RoundDraws α =>
      ((q.1, q.2.1) : ClusterDraws Ω α K N)) (μ.prod ν) (clusterLaw μ α K N) :=
    (MeasurePreserving.id μ).prod measurePreserving_fst
  have hrest : MeasurePreserving (fun q : Ω × settings.RoundDraws α => q.2.1) (μ.prod ν)
      (restLaw α K N) := measurePreserving_snd.comp hmarg
  have hmid : ∀ c : List (HarvestCode α), Measurable
      fun q : Ω × settings.RoundDraws α => idealDraws P.D Dsf k N (dec c) (q.1, q.2.1) := fun c =>
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
          ((hrest.measurable (measurableSet_popsHit P.D k N M (dec c).pops (dec c).prefs)).compl
            |>.union (hmid c (measurableSet_scanShort (loopPopulations K r) M T A))))
    (fun _ _ => ∅) (fun y d ω ω' _ hq => by
      rcases y with _ | c
      · exact hq
      · exact ⟨hq.1, hq.2.imp id (scanShort_noise _ _ _ _ _ _ _ _ _ _ rfl rfl).1⟩)
    (fun _ => Set.univ) (fun _ => MeasurableSet.univ)
    (fun y x _ _ => Finset.disjoint_empty_right _)
    (ENNReal.ofReal b) (fun y => by
      rcases y with _ | c
      · show (μ.prod ν) ∅ ≤ _
        simp
      · by_cases hH : Good c
        swap
        · have : Bad (some c) = ∅ := Set.eq_empty_of_forall_notMem fun q hq => hH hq.1
          rw [this, measure_empty]; exact bot_le
        obtain ⟨hcr, hpg, hpm, hyl⟩ := hH
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
              have hi : i.val < h.pops.length := by rw [hlen]; exact lt_of_some_mem_loopPops hj
              simp only [loopDistributions, List.getElem?_eq_getElem hi]
              exact hpg _ (List.mem_map.2 ⟨_, List.getElem_mem hi, rfl⟩) t
          linarith
        have hscan := clusterLaw_scanShort_le (μ := μ) P.D Dsf k N h (loopPopulations K r)
          (M := M) (T := T) (A := A) hat hA hT
        have hmiss := restLaw_popsHit_le (K := K) P.D k N M h.pops h.prefs hy0 hy1 hδₘ hM hN hyl
        have hcard : ((loopPopulations K r).card : ℝ) ≤ K + 1 := by
          exact_mod_cast card_loopPopulations_le K r
        have hscanE : (μ.prod ν) {q : Ω × settings.RoundDraws α |
            ScanShort (loopPopulations K r) M T A (idealDraws P.D Dsf k N h (q.1, q.2.1))}
            = clusterLaw μ α K N
              {c | ScanShort (loopPopulations K r) M T A (idealDraws P.D Dsf k N h c)} :=
          hmarg.measure_preimage (((measurable_idealDraws P.D Dsf k N h)
            (measurableSet_scanShort (loopPopulations K r) M T A)).nullMeasurableSet)
        have hmissE : (μ.prod ν) {q : Ω × settings.RoundDraws α |
            ¬ PopsHit P.D k N M h.pops h.prefs q.2.1}
            = restLaw α K N {c | ¬ PopsHit P.D k N M h.pops h.prefs c} :=
          hrest.measure_preimage
            (measurableSet_popsHit P.D k N M h.pops h.prefs).compl.nullMeasurableSet
        calc (μ.prod ν) (Bad (some c))
            ≤ (μ.prod ν) {q : Ω × settings.RoundDraws α |
                ¬ PopsHit P.D k N M h.pops h.prefs q.2.1}
              + (μ.prod ν) {q : Ω × settings.RoundDraws α |
                ScanShort (loopPopulations K r) M T A (idealDraws P.D Dsf k N h (q.1, q.2.1))} :=
              (measure_mono fun q hq => hq.2).trans (measure_union_le _ _)
          _ = ENNReal.ofReal ((restLaw α K N).real {c | ¬ PopsHit P.D k N M h.pops h.prefs c})
              + ENNReal.ofReal ((clusterLaw μ α K N).real
                {c | ScanShort (loopPopulations K r) M T A (idealDraws P.D Dsf k N h c)}) := by
              rw [hmissE, hscanE, ofReal_measureReal (measure_ne_top _ _),
                ofReal_measureReal (measure_ne_top _ _)]
          _ ≤ ENNReal.ofReal b := by
              rw [← ENNReal.ofReal_add measureReal_nonneg measureReal_nonneg]
              refine ENNReal.ofReal_le_ofReal ?_
              have hq0 : 0 ≤ 2 * (M.choose T * (T * (2 * cap)) ^ (M - T))
                  + 2 * Real.exp (-(T / 8 : ℝ)) := by positivity
              have := mul_le_mul_of_nonneg_right hcard hq0
              linarith)
  refine ENNReal.toReal_le_of_le_ofReal hb0 ((measure_mono ?_).trans hmain)
  rintro p ⟨hok, hlen, hpg, hpm, hyl, hbad⟩
  have hst : stageH settings O P.D Dsf r p.1 p.2
      = some (histCodes settings O P.D Dsf p.1 p.2 r) := by
    simp only [stageH, if_pos hok]
  show (p.1, p.2 ⟨r, hr⟩) ∈ Bad (stageH settings O P.D Dsf r p.1 p.2)
    ∧ p.2 ∈ (fun _ : Y => (Set.univ : Set (Fin K → settings.RoundDraws α)))
      (stageH settings O P.D Dsf r p.1 p.2)
  rw [hst]
  exact ⟨⟨⟨hlen, hpg, hpm, hyl⟩, hbad⟩, Set.mem_univ _⟩

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
