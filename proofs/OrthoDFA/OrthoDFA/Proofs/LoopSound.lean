import OrthoDFA.Proofs.LoopSoundDFA

/-!
# Every consistent round's DFA is accurate, whatever the populations

A run fails where some round, given the earlier rounds, misses its clustering's cut of the uniform
pool at the run's noise (`loop_uni_round_le_persist`), collides (`loop_coll_null_uni`), or ends
consistent at a hypothesis disagreeing on more than its rate (`loop_consistent_round_le`). Off
those, every earlier clustering returned, every round reads nothing an earlier one read, and a
round that ends consistent has its DFA from where `ε` sifts within its rate, what its probes avoid,
and its family's cut (`dfa_of_reads`). Nothing is asked of the harvests' populations.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Random

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

section Loop

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
  (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]

/-- Round `r` collides, where its populations are short and the uniform pool and the suffixes
are mostly unread, with chance zero. -/
theorem loop_coll_null_uni (Ls N : ℕ) (hN : P.L + settings.level3.probeBudget + 2 ≤ N)
    (hLs : P.L < Ls) (hDsf : Dsf {v | v.toList.length ≠ Ls} = 0) (r : ℕ) (hr : r < K) :
    (loopMeasure μ α K settings.level3.probeBudget)
      {p | (histCodes settings O P.D Dsf N p.1 p.2 r).length = r
        ∧ (∀ ν ∈ (decodeHist P.D settings.level3.walkStart
            (histCodes settings O P.D Dsf N p.1 p.2 r)).pops, ν {q | Ls ≤ q.toList.length} = 0)
        ∧ P.D.real ↑(decodeHist P.D settings.level3.walkStart
            (histCodes settings O P.D Dsf N p.1 p.2 r)).prefs ≤ 1 / 2
        ∧ Dsf.real ↑(decodeHist P.D settings.level3.walkStart
            (histCodes settings O P.D Dsf N p.1 p.2 r)).suffs ≤ 1 / 2
        ∧ ¬ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf N p.1 p.2 r) p.2 r hr}
      = 0 :=
  measure_mono_null (by
    intro p hp hlt
    obtain ⟨hlen, hsupp, hpm, hsm, hnc⟩ := hp
    exact hnc (noColl_of_allLt settings O P Dsf Ls N hN hLs hDsf p.1 p.2 r hr hlen hsupp hpm hsm
      hlt))
    (allLt_ae r)

omit [IsProbabilityMeasure μ] [IsProbabilityMeasure Dsf] in
/-- An earlier round's prefixes are among the data's. -/
lemma prefsOf_sub_hist (N : ℕ) (hN : P.L + settings.level3.probeBudget + 2 ≤ N) (ω : Ω)
    (g : Fin K → RoundDraws α K settings.level3.probeBudget) (r : ℕ) (i : Fin K)
    (hir : i.1 < (histCodes settings O P.D Dsf N ω g r).length) :
    prefsOf settings P.D Dsf (histAt settings P.D (histCodes settings O P.D Dsf N ω g r) i) (g i).1
      ⊆ (decodeHist P.D settings.level3.walkStart (histCodes settings O P.D Dsf N ω g r)).prefs := by
  intro p hp
  obtain ⟨hiK, code, hget, h1, -⟩ :=
    histCodes_getElem settings O P.D Dsf N P.length_le hN ω g r i hir
  have hmem : code ∈ histCodes settings O P.D Dsf N ω g r := List.mem_of_getElem? hget
  simp only [decodeHist]
  refine (mem_foldr_union _ _ p).2 ⟨code, hmem, ?_⟩
  rw [h1]
  exact hp

omit [IsProbabilityMeasure μ] [IsProbabilityMeasure Dsf] in
/-- Every earlier clustering returned. -/
lemma histOK_of_returns (N : ℕ) (ω : Ω) (g : Fin K → RoundDraws α K settings.level3.probeBudget) :
    ∀ i, (∀ j < i, ∀ hj : j < K, (histCodes settings O P.D Dsf N ω g j).length = j →
      ∃ B ∈ settings.schedule (loopPopulations K
          (decodeHist P.D settings.level3.walkStart (histCodes settings O P.D Dsf N ω g j)).pops.length),
        clusterRun P.D Dsf (decodeHist P.D settings.level3.walkStart
          (histCodes settings O P.D Dsf N ω g j)) (ω, (g ⟨j, hj⟩).1)
          ∈ ret O.mq (loopPopulations K (decodeHist P.D settings.level3.walkStart
            (histCodes settings O P.D Dsf N ω g j)).pops.length) none settings.η₀
            settings.indecisionLimit settings.εcov settings.gateLevel B) →
      HistOK settings O P.D Dsf N ω g i
  | 0, _ => trivial
  | i + 1, h => ⟨histOK_of_returns N ω g i fun j hj => h j (by omega),
      fun hK hlen => h i (by omega) hK hlen⟩

/-- Where the probes are walked from `0`, the clustering's schedule holds the uniform pool's cut at
the run's noise, and the rounds read few prefixes and suffixes of a uniform pool and suffixes with
small atoms, every round that ends consistent has its DFA from where `ε` sifts misjudging at most
`ε + mA + (513/512) εcov + 2ι + m` of the probes, but for chance `K (δc + δ/8)`. -/
theorem loop_sound_core {σ : Type*} [Fintype σ] (M : _root_.DFA α σ)
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) (Ls N NPr NSr : ℕ)
    (δc capU capSf m ms mA pAP : ℝ)
    (hN : P.L + settings.level3.probeBudget + 2 ≤ N) (hLs : P.L < Ls)
    (hDsf : Dsf {v | v.toList.length ≠ Ls} = 0)
    (hpAP : pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L})
    (hDat : ∀ t, P.D.real {t} ≤ capU) (hsfat : ∀ t, Dsf.real {t} ≤ capSf)
    (hband : ∀ pops, ∀ B ∈ settings.schedule pops, B.lo ≤ B.hi)
    (hpref : ∀ pops (x : Run Unit (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ NPr)
    (hsuff : ∀ pops (x : Run Unit (FreeMonoid α) (Option (Fin K))),
      (roundSuffs (settings.schedule pops) x).card ≤ NSr)
    (hk0 : settings.level3.walkStart = 0) (hδ : 0 < settings.level3.δ)
    (hε0 : 0 < settings.level3.disagreementRate) (hδc : 0 ≤ δc)
    (hmP : K * NPr * capU ≤ m) (hm : m ≤ 1 / 2) (hmS : K * NSr * capSf ≤ ms) (hms : ms ≤ 1 / 2)
    (hmA : NPr * prefixMax P.D settings.avoidDepth ≤ mA)
    (he : settings.level3.disagreementRate + mA < 1)
    (huni : ∀ pops : Finset (Option (Fin K)), none ∈ pops →
      ∀ (L : Option (Fin K) → Measure (FreeMonoid α)) (Lsf : Measure (FreeMonoid α)),
        (∀ j, IsProbabilityMeasure (L j)) → IsProbabilityMeasure Lsf →
        (∀ j ∈ pops, L j {p | Ls ≤ p.toList.length} = 0) →
        (∀ t, (L none).real {t} ≤ 2 * capU) →
        Lsf {v | v.toList.length ≠ Ls} = 0 →
        pAP - ms ≤ Lsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
        (∀ t, Lsf.real {t} ≤ 2 * capSf) →
        (runMeasure μ L Lsf).real
          {x | ¬ UniSpec O pops (settings.schedule pops) settings.η₀ settings.indecisionLimit
            settings.εcov settings.gateLevel (L none) (streams (fun _ => ∅) none x)} ≤ δc) :
    (loopMeasure μ α K settings.level3.probeBudget).real
      {p | ∃ o ∈ indecisiveLStarLoop settings O P.D Dsf (persistDraws p),
        o.result.ending = .consistent
          ∧ ¬ ∃ s, o.result.hypothesis.tree.sift o.read 1 = .inl s
            ∧ P.D.real {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)}
              ≤ settings.level3.disagreementRate + mA + 513 / 512 * settings.εcov
                + 2 * settings.indecisionLimit + m}
      ≤ K * (δc + settings.level3.δ / 8) := by
  classical
  set T := settings.level3.probeBudget
  set C := settings.level3
  set ε := C.disagreementRate
  set dec := decodeHist P.D settings.level3.walkStart with hdecd
  set μL := loopMeasure μ α K T
  set cH : ℕ → Ω × (Fin K → RoundDraws α K T) → List (HarvestCode α) := fun r p =>
    histCodes settings O P.D Dsf N p.1 p.2 r
  have hε1 : ε < 1 := by
    have : 0 ≤ mA := (mul_nonneg (Nat.cast_nonneg _) (prefixMax_pos P.D _).le).trans hmA
    linarith
  -- the rounds' bad events
  set EA : (r : ℕ) → r < K → Set (Ω × (Fin K → RoundDraws α K T)) := fun r hr =>
    {p | HistOK settings O P.D Dsf N p.1 p.2 r
      ∧ (cH r p).length = r
      ∧ (∀ ν' ∈ (dec (cH r p)).pops, ν' {q | Ls ≤ q.toList.length} = 0)
      ∧ P.D.real ↑(dec (cH r p)).prefs ≤ m
      ∧ Dsf.real ↑(dec (cH r p)).suffs ≤ ms
      ∧ ¬ UniSpec O (loopPopulations K r) (settings.schedule (loopPopulations K r))
          settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel
          (offLaw P.D (dec (cH r p)).prefs)
          (clusterRun P.D Dsf (dec (cH r p)) (p.1, (p.2 ⟨r, hr⟩).1))
      ∧ NoColl settings P.D Dsf Ls (cH r p) p.2 r hr
      ∧ ∀ i : Fin K, i.1 < r → prefsOf settings P.D Dsf (histAt settings P.D (cH r p) i) (p.2 i).1
        ⊆ (dec (cH r p)).prefs}
  set EB : (r : ℕ) → r < K → Set (Ω × (Fin K → RoundDraws α K T)) := fun r hr =>
    {p | (cH r p).length = r
      ∧ (∀ ν ∈ (dec (cH r p)).pops, ν {q | Ls ≤ q.toList.length} = 0)
      ∧ P.D.real ↑(dec (cH r p)).prefs ≤ 1 / 2
      ∧ Dsf.real ↑(dec (cH r p)).suffs ≤ 1 / 2
      ∧ ¬ NoColl settings P.D Dsf Ls (cH r p) p.2 r hr}
  set EC : (r : ℕ) → r < K → Set (Ω × (Fin K → RoundDraws α K T)) := fun r hr =>
    {p | ∃ t, stage4 settings O P.D Dsf N r hr p.1 p.2 = some t
      ∧ FalseConsistent O C t.2.2.1 t.2.1.lo (t.2.1.hi + 1)
        (avoidLaw P.D settings.avoidDepth t.2.2.2) p.1 (probesOf settings P.D t.2.2.2 (p.2 ⟨r, hr⟩).2)}
  have hA : ∀ r (hr : r < K), μL.real (EA r hr) ≤ δc := fun r hr =>
    loop_uni_round_le_persist settings O P Dsf Ls N δc capU capSf m ms pAP hδc hN hm hms huni hLs
      hDat hDsf hpAP hsfat r hr
  have hB : ∀ r (hr : r < K), μL.real (EB r hr) = 0 := fun r hr => by
    rw [measureReal_def, loop_coll_null_uni settings O P Dsf Ls N hN hLs hDsf r hr,
      ENNReal.toReal_zero]
  have hC : ∀ r (hr : r < K), μL.real (EC r hr) ≤ C.δ / 8 := fun r hr =>
    loop_consistent_round_le settings O P Dsf N hN hδ hε0 hε1 r hr
  set Ev : ℕ → Set (Ω × (Fin K → RoundDraws α K T)) := fun r =>
    if hr : r < K then (EA r hr ∪ EB r hr) ∪ EC r hr else ∅ with hEvd
  have hEv : ∀ r, μL.real (Ev r) ≤ δc + C.δ / 8 := by
    intro r
    simp only [hEvd]
    split_ifs with hr
    · calc μL.real ((EA r hr ∪ EB r hr) ∪ EC r hr)
          ≤ μL.real (EA r hr) + μL.real (EB r hr) + μL.real (EC r hr) :=
            (measureReal_union_le _ _).trans (add_le_add (measureReal_union_le _ _) le_rfl)
        _ ≤ δc + 0 + C.δ / 8 := add_le_add (add_le_add (hA r hr) (le_of_eq (hB r hr))) (hC r hr)
        _ = δc + C.δ / 8 := by ring
    · rw [measureReal_empty]
      positivity
  -- the counts
  have hcodes : ∀ (ω : Ω) (g : Fin K → RoundDraws α K T) (r : ℕ),
      ∀ code ∈ histCodes settings O P.D Dsf N ω g r,
        code.2.2.2.1.card ≤ NPr ∧ code.2.2.2.2.card ≤ NSr := by
    intro ω g r code hcode
    obtain ⟨i, hi, hget⟩ := List.getElem_of_mem hcode
    obtain ⟨hiK, code', hget', h1, h2⟩ :=
      histCodes_getElem settings O P.D Dsf N P.length_le hN ω g r i hi
    rw [List.getElem?_eq_getElem hi, hget, Option.some.injEq] at hget'
    subst hget'
    rw [h1, h2]
    exact ⟨hpref _ _, hsuff _ _⟩
  have hmass : ∀ p (r : ℕ), (cH r p).length ≤ K →
      P.D.real ↑(dec (cH r p)).prefs ≤ m ∧ Dsf.real ↑(dec (cH r p)).suffs ≤ ms := by
    intro p r hlen
    have hP := card_foldr_union_le (fun c' : HarvestCode α => c'.2.2.2.1) (cH r p) NPr
      fun x hx => (hcodes p.1 p.2 r x hx).1
    have hS := card_foldr_union_le (fun c' : HarvestCode α => c'.2.2.2.2) (cH r p) NSr
      fun x hx => (hcodes p.1 p.2 r x hx).2
    have hU0 : 0 ≤ capU := measureReal_nonneg.trans (hDat 1)
    have hS0 : 0 ≤ capSf := measureReal_nonneg.trans (hsfat 1)
    have hlenR : ((cH r p).length : ℝ) ≤ K := by exact_mod_cast hlen
    refine ⟨(measureReal_le_card_mul _ hDat).trans ?_, (measureReal_le_card_mul _ hsfat).trans ?_⟩
    · have hc : ((dec (cH r p)).prefs.card : ℝ) ≤ (cH r p).length * NPr := by exact_mod_cast hP
      calc ((dec (cH r p)).prefs.card : ℝ) * capU ≤ (cH r p).length * NPr * capU :=
            mul_le_mul_of_nonneg_right hc hU0
        _ ≤ K * NPr * capU := by gcongr
        _ ≤ m := hmP
    · have hc : ((dec (cH r p)).suffs.card : ℝ) ≤ (cH r p).length * NSr := by exact_mod_cast hS
      calc ((dec (cH r p)).suffs.card : ℝ) * capSf ≤ (cH r p).length * NSr * capSf :=
            mul_le_mul_of_nonneg_right hc hS0
        _ ≤ K * NSr * capSf := by gcongr
        _ ≤ ms := hmS
  -- a failing run fails some round's claim
  have hsub : {p : Ω × (Fin K → RoundDraws α K T) |
      ∃ o ∈ indecisiveLStarLoop settings O P.D Dsf (persistDraws p),
        o.result.ending = .consistent
          ∧ ¬ ∃ s, o.result.hypothesis.tree.sift o.read 1 = .inl s
            ∧ P.D.real {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)}
              ≤ ε + mA + 513 / 512 * settings.εcov + 2 * settings.indecisionLimit + m}
      ⊆ ⋃ r ∈ Finset.range K, Ev r := by
    intro p hp
    obtain ⟨o, ho, hco, hbad⟩ := hp
    by_contra hno
    simp only [Set.mem_iUnion, Finset.mem_range, not_exists] at hno
    set Lr := loopRounds settings O P.D Dsf (persistDraws p) K with hLr
    have hoL : o ∈ Lr := ho
    obtain ⟨r, hrL, rfl⟩ := List.getElem_of_mem hoL
    have hKL : Lr.length ≤ K := length_loopRounds_le settings O P.D Dsf _ K
    -- each round up to `r` meets its claims
    have key : ∀ i (hi : i < Lr.length), ∃ hK : i < K,
        HistOK settings O P.D Dsf N p.1 p.2 i
        ∧ (cH i p).length = i
        ∧ NoColl settings P.D Dsf Ls (cH i p) p.2 i hK
        ∧ UniSpec O (loopPopulations K i) (settings.schedule (loopPopulations K i))
            settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel
            (offLaw P.D (dec (cH i p)).prefs)
            (clusterRun P.D Dsf (dec (cH i p)) (p.1, (p.2 ⟨i, hK⟩).1))
        ∧ P.D.real ↑(dec (cH i p)).prefs ≤ m := by
      intro i
      induction i using Nat.strong_induction_on with
      | _ i ih =>
        intro hi
        obtain ⟨hK, hlen, hdec, e⟩ := persist_bridge settings O P.D Dsf P.length_le hN p i hi
        have hnot : p ∉ Ev i := hno i (by omega)
        simp only [hEvd, dif_pos hK, Set.mem_union, not_or] at hnot
        obtain ⟨⟨hnA, hnB⟩, -⟩ := hnot
        have hpl : (dec (cH i p)).pops.length = i := by
          simp only [hdecd, decodeHist, List.length_map]; exact hlen
        have hok : HistOK settings O P.D Dsf N p.1 p.2 i := by
          refine histOK_of_returns settings O P Dsf N p.1 p.2 i fun j hj hjK hjl => ?_
          obtain ⟨hK', -, -, -, hu, -⟩ := ih j hj (by omega)
          have hplj : (dec (cH j p)).pops.length = j := by
            simp only [hdecd, decodeHist, List.length_map]; exact hjl
          show ∃ B ∈ settings.schedule (loopPopulations K (dec (cH j p)).pops.length), _
          rw [hplj]
          exact hu.1
        obtain ⟨hmP', hmS'⟩ := hmass p i (by rw [hlen]; omega)
        have hsupp : ∀ ν' ∈ (dec (cH i p)).pops, ν' {q | Ls ≤ q.toList.length} = 0 := by
          show ∀ ν' ∈ (decodeHist P.D settings.level3.walkStart (cH i p)).pops, _
          rw [hdec, histOf_pops]
          intro ν' hν
          obtain ⟨o', ho', hoν⟩ := List.mem_filterMap.1 hν
          obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem ho'
          simp only [List.length_take] at hj
          rw [List.getElem_take] at hoν
          have hji : j < i := by omega
          obtain ⟨hKj, -, hlenj, hncj, -, -⟩ := ih j hji (by omega)
          obtain ⟨hKj', -, -, ej⟩ := persist_bridge settings O P.D Dsf P.length_le hN p j (by omega)
          unfold LoopRound.population at hoν
          rcases hh : (Lr[j]'(by omega)).result.ending.harvested with _ | h
          · rw [hh] at hoν; cases hoν
          rw [hh, Option.map_some, Option.some.injEq] at hoν
          subst hoν
          refine popLaw_null (le_antisymm ((Measure.restrict_apply'
            (measurableSet_of_countable _)).le.trans (measure_mono_null (fun z hz => ?_)
              (measure_empty (μ := Random.harvestLaw _ P.D _ _ h))).le) bot_le) ?_
          · obtain ⟨hzl, ⟨v, hv, hv1⟩, hzs⟩ := hz
            have hlv := hzs v hv hv1
            have hhj : histAt settings P.D (cH j p) j = dec (cH j p) := by
              simp only [histAt, List.take_of_length_le (le_of_eq hlenj), hdecd]
            have hvs : v ∈ suffsOf settings P.D Dsf (histAt settings P.D (cH j p) j)
                (p.2 ⟨j, hKj⟩).1 := by
              rw [hhj]
              rw [ej] at hv
              exact Finset.mem_erase.2 ⟨hv1, hv⟩
            have := hncj.1 ⟨j, hKj⟩ le_rfl v hvs
            simp only [Set.mem_ofPred_eq] at hzl
            omega
          · simp only [Set.mem_ofPred_eq, FreeMonoid.toList_one, List.length_nil, not_le]
            omega
        have hnc : NoColl settings P.D Dsf Ls (cH i p) p.2 i hK := by
          by_contra h
          exact hnB ⟨hlen, hsupp, hmP'.trans hm, hmS'.trans hms, h⟩
        have hcons : ∀ i' : Fin K, i'.1 < i → prefsOf settings P.D Dsf
            (histAt settings P.D (cH i p) i') (p.2 i').1 ⊆ (dec (cH i p)).prefs :=
          fun i' hi' => prefsOf_sub_hist settings O P Dsf N hN p.1 p.2 i i' (by rw [hlen]; exact hi')
        have hu : UniSpec O (loopPopulations K i) (settings.schedule (loopPopulations K i))
            settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel
            (offLaw P.D (dec (cH i p)).prefs)
            (clusterRun P.D Dsf (dec (cH i p)) (p.1, (p.2 ⟨i, hK⟩).1)) := by
          by_contra h
          exact hnA ⟨hok, hlen, hsupp, hmP', hmS', h, hnc, hcons⟩
        exact ⟨hK, hok, hlen, hnc, hu, hmP'⟩
    -- round `r` itself
    obtain ⟨hK, hok, hlen, hnc, hu, hmP'⟩ := key r hrL
    obtain ⟨hK2, -, hdec, e⟩ := persist_bridge settings O P.D Dsf P.length_le hN p r hrL
    have hpl : (dec (cH r p)).pops.length = r := by
      simp only [hdecd, decodeHist, List.length_map]; exact hlen
    set H := dec (cH r p)
    set d : LoopRoundDraws Ω α K T := (p.1, p.2 ⟨r, hK⟩)
    have hret : ∃ B ∈ settings.schedule (loopPopulations K H.pops.length),
        clusterRun P.D Dsf H (d.1, d.2.1) ∈ ret O.mq (loopPopulations K H.pops.length) none
          settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B := by
      rw [hpl]; exact hu.1
    obtain ⟨hBs, hBr⟩ := loopRound_returns settings O P.D Dsf H d hret
    set o' := loopRound settings O P.D Dsf H d
    have hoo : Lr[r] = o' := e
    rw [hoo] at hco hbad
    -- what reached level 3
    have hst : stage4 settings O P.D Dsf N r hK p.1 p.2 = some (cH r p, o'.state, o'.family,
        roundKeys settings P.D Dsf H (d.1, d.2.1)) := by
      rw [stage4_eq_some]
      exact ⟨hok, rfl, hlen, hret, rfl, rfl, rfl⟩
    have hnC : p ∉ EC r hK := by
      have := hno r (by omega)
      simp only [hEvd, dif_pos hK, Set.mem_union, not_or] at this
      exact this.2
    have hrate : ¬ FalseConsistent O C o'.family o'.state.lo (o'.state.hi + 1)
        (avoidLaw P.D settings.avoidDepth (roundKeys settings P.D Dsf H (d.1, d.2.1))) p.1
        (probesOf settings P.D (roundKeys settings P.D Dsf H (d.1, d.2.1)) (p.2 ⟨r, hK⟩).2) :=
      fun h => hnC ⟨_, hst, h⟩
    set K₀ := roundKeys settings P.D Dsf H (d.1, d.2.1)
    set read := fun z => familyRead O.mq o'.family o'.state.lo (o'.state.hi + 1) z p.1
    have hres : o'.result = indecisiveLStarLevel3 C read (probesOf settings P.D K₀ (p.2 ⟨r, hK⟩).2) :=
      rfl
    have hdis' : (avoidLaw P.D settings.avoidDepth K₀).real
        (searchSet read C.walkStart o'.result.hypothesis
          ∪ endsSet read C.walkStart o'.result.hypothesis) ≤ ε := by
      by_contra h
      push Not at h
      exact hrate ⟨by rw [← hres]; exact hco, by rw [← hres]; exact h⟩
    have hK₀ : (K₀.card : ℝ) ≤ NPr := by
      have h1 : K₀.card ≤ (prefsOf settings P.D Dsf H d.2.1).card := by
        rw [show K₀ = _ from roundKeys_eq_prefsOf settings P.D Dsf H d.1 d.2.1]
        exact Finset.card_image_le
      exact_mod_cast h1.trans (hpref _ _)
    have hav : P.D.real (avoiding settings.avoidDepth ↑K₀)ᶜ ≤ mA :=
      (real_compl_avoiding_le P.D settings.avoidDepth K₀).trans
        ((mul_le_mul_of_nonneg_right hK₀ (prefixMax_pos P.D _).le).trans hmA)
    have hdis : P.D.real (searchSet read 0 o'.result.hypothesis
        ∪ endsSet read 0 o'.result.hypothesis) ≤ ε + mA := by
      rw [← hk0]
      exact (real_le_avoidLaw P.D settings.avoidDepth K₀ _).trans (add_le_add hdis' hav)
    rw [hpl] at hBs hBr
    obtain ⟨hcut, hund⟩ := hu.2 o'.state hBs hBr
    have hfam : o'.family = clusterAt O.mq (loopPopulations K H.pops.length)
        (clusterRun P.D Dsf H (d.1, d.2.1)) o'.state := rfl
    have hF1 : (1 : FreeMonoid α) ∉ o'.family := by
      rw [hfam]
      unfold clusterAt
      exact Finset.notMem_erase _ _
    have hfa : familyAt O.mq (loopPopulations K r) (clusterRun P.D Dsf H (d.1, d.2.1)) o'.state
        = insert 1 o'.family := by
      rw [hfam, hpl]
      rfl
    have hx1 : (clusterRun P.D Dsf H (d.1, d.2.1)).1 = p.1 := rfl
    rw [hfa, hx1] at hcut hund
    obtain ⟨s, hs, hacc⟩ := dfa_of_reads M O hL hF1 (hband _ _ hBs) p.1 o'.result.hypothesis
      (by rw [hres]; exact rootOne_level3 _ _ _) P.D H.prefs (hmP'.trans hm) hdis he hcut hund
    refine hbad ⟨s, hs, hacc.trans ?_⟩
    linarith
  calc μL.real _ ≤ μL.real (⋃ r ∈ Finset.range K, Ev r) := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ r ∈ Finset.range K, μL.real (Ev r) := measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _r ∈ Finset.range K, (δc + C.δ / 8) := Finset.sum_le_sum fun r _ => hEv r
    _ = K * (δc + C.δ / 8) := by rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]

end Loop

end OrthoDFA

end
