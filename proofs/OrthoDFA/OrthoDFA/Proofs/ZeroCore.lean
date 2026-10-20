import OrthoDFA.Proofs.LearnerLoop
import OrthoDFA.Proofs.ZeroLoop
import OrthoDFA.Proofs.ZeroStart

/-!
# The loop walked from before its avoidance depth stops at its first round

Where the start is equivalent to a covered state, the first round's family, held to the
clustering's bound on the window's undecided reads, leaves no state undecided more than `1.5θ` of
the time (`undecided_le_of_start`). So level 3 cannot end well in a harvest, and the first round
ends consistent but for the chance its clustering fails, the chance its level 3 fails
(`loop_round_zero`), and null events (`zero_core`).
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

section Long

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
  (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]

omit [IsProbabilityMeasure μ] in
lemma histCodes_zero (N : ℕ) (ω : Ω) (g : Fin K → RoundDraws α K settings.level3.probeBudget) :
    histCodes settings O P.D Dsf N ω g 0 = [] := rfl

/-- The first round's clustering reads a prefix shorter than every string `D` gives mass with
chance zero. -/
theorem longPrefs_null (ℓ₀ : ℕ) (hshort : P.D {x | x.toList.length < ℓ₀} = 0) (hr : 0 < K) :
    (loopMeasure μ α K settings.level3.probeBudget)
      {p | ¬ ∀ q ∈ prefsOf settings P.D Dsf (histAt settings P.D
          (histCodes settings O P.D Dsf (P.L + settings.level3.probeBudget + 2) p.1 p.2 0) 0)
          (p.2 ⟨0, hr⟩).1, ℓ₀ ≤ q.toList.length} = 0 := by
  classical
  refine measure_mono_null (fun p hp => ?_) (allLt_ae (μ := μ) (K := K)
    (T := settings.level3.probeBudget) 0)
  simp only [Set.mem_setOf_eq] at hp ⊢
  intro hlt
  refine hp fun q hq => ?_
  rw [histCodes_zero] at hq
  obtain ⟨j, hj, n, hqn⟩ := mem_prefsOf_draw settings P.D Dsf hq
  have hj' : j = none := by
    rcases j with _ | j
    · rfl
    · exfalso
      have := lt_of_some_mem_loopPops hj
      simp [decodeHist] at this
  subst hj'
  have hD : loopDistributions (K := K) P.D (histAt settings P.D [] 0).pops none = P.D := rfl
  have hT : (histAt settings P.D [] 0).prefs = ∅ := rfl
  rw [hD, hT] at hqn
  have h0 : P.D (↑(∅ : Finset (FreeMonoid α)))ᶜ ≠ 0 := by
    simp only [Finset.coe_empty, Set.compl_empty, measure_univ]; exact one_ne_zero
  obtain ⟨u, hu, rfl⟩ : ∃ u : unitInterval, (u : ℝ) < 1 ∧
      q = drawFrom (offLaw P.D ∅) u := by
    rcases hqn with h | h
    · exact ⟨_, (hlt ⟨0, hr⟩ le_rfl).2.1 none n, h⟩
    · exact ⟨_, (hlt ⟨0, hr⟩ le_rfl).2.2 none n, h⟩
  obtain ⟨-, hpos⟩ := drawFrom_offLaw h0 hu
  have := mem_of_pos (A := {q : FreeMonoid α | ℓ₀ ≤ q.toList.length})
    (by
      refine measure_mono_null (fun x hx => ?_) hshort
      simp only [Set.mem_compl_iff, Set.mem_setOf_eq, not_le] at hx ⊢
      exact hx) hpos
  exact this

end Long

section Core

variable [Nonempty α] {σ : Type*} [Fintype σ]

set_option maxHeartbeats 4000000 in -- the first round's three bounds
/-- The loop walked from before its avoidance depth, where the start is equivalent to a covered
state: its first round ends consistent, its probes disagreeing at most
`C.disagreementRate + u + mr`, but for chance `δc + gate` its clustering fails and level 5's
chance at the depth `kd`, plus `δ/8 + #(strings of at most |Q| letters) cross / u`, that its
level 3 does. -/
theorem zero_core (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (L K Ls NS NP ℓ₀ kd a : ℕ) (η₀ ι gate εcov δc slack pAP cap mr u cross θ c : ℝ)
    (C : Random.Level3Settings) (q : σ)
    (hPD : P.D = windowStrings ((L + 1) / 2) (L + Fintype.card σ + 3))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)
    (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hι : 0 < ι) (hι1 : ι ≤ 1 / 2)
    (hg : 0 < gate) (hg1 : gate < 1 / 2) (hε : 0 < εcov) (hε1 : εcov ≤ 1 / 2)
    (hδc : 0 < δc) (hδc1 : δc ≤ 1) (hsl : 0 < slack) (hpAP : 0 < pAP) (hL1 : 1 ≤ P.L)
    (hK : Fintype.card σ + 1 ≤ K) (hLs : P.L + Fintype.card σ < Ls) (hcap : 0 < cap)
    (hNSb : ∀ pops : Finset (Option (Fin K)), none ∈ pops →
      poolCount η₀ pops ι εcov (δc / 2) pAP (cross ^ 4) ≤ NS)
    (hNPb : ∀ pops : Finset (Option (Fin K)), none ∈ pops →
      prefCount η₀ pops ι εcov (δc / 2) gate pAP (cross ^ 4) 0 ≤ NP)
    (hq1 : 2 ^ 22 * ((K : ℝ) + 2) ^ 2 * NP * cap ≤ δc * rateScale η₀ ι εcov ^ 2)
    (hq2 : 2 ^ 9 * ((K : ℝ) + 2) * NP * cap ≤ δc * slack ^ 2)
    (hq3 : 2 ^ 14 * ((K : ℝ) + 2) * NS * cap ≤ δc * pAP)
    (hDat : ∀ t, P.D.real {t} ≤ cap) (hshort : P.D {x | x.toList.length < ℓ₀} = 0)
    (ha : 1 ≤ a) (hka : C.walkStart ≤ a) (hlong : a + Fintype.card σ < ℓ₀)
    (hu : 0 < u) (hmr : 6 * (K + 1) * NP * prefixMax P.D a ≤ mr) (hmr1 : mr ≤ 1 / 2)
    (hcr : 0 < cross) (hprem : RoundPremises σ α P.L C θ (2 * prefixMax P.D kd) cross)
    (hc : 0 < c) (hq : q ∈ coveredStates M L c)
    (hstart : ∀ w : List α, M.evalFrom q w ∈ M.accept ↔ M.evalFrom M.start w ∈ M.accept)
    (hθc : (2 * ι + slack) * ((L + Fintype.card σ + 3 : ℕ) + 1 - ((L + 1) / 2 : ℕ))
      * (Fintype.card α : ℝ) ^ Fintype.card σ / c ≤ 3 / 2 * θ) :
    ∃ settings : LoopSettings K,
      settings.η₀ = η₀ ∧ settings.indecisionLimit = ι ∧ settings.εcov = εcov ∧
      settings.gateLevel = gate ∧ settings.level3 = C ∧ settings.avoidDepth = a ∧
      (∀ pops, ∑ B ∈ settings.schedule pops, B.npref ≤ 2 * NP) ∧
      ∀ Dsf : Measure (FreeMonoid α), IsProbabilityMeasure Dsf →
        Dsf {v | v.toList.length ≠ Ls} = 0 →
        pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
        (∀ v, Dsf.real {v} ≤ cap) →
        (loopMeasure μ α K settings.level3.probeBudget).real
            {p | let rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p)
              ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
                  o.Checks M O P.D C.walkStart (513 / 512 * εcov + 2 * ι + 2 * slack)
                    (2 * ι + slack) (C.disagreementRate + u + mr))
                ∧ RoundsFresh settings O P.D Dsf (P.L + settings.level3.probeBudget + 2) Ls p)}
          ≤ δc + gate + (C.δ
            + (if C.walkStart < kd then 3 / 2 * θ * (C.probeBudget * (P.L + 1) * Fintype.card σ
              + Fintype.card σ * (Fintype.card σ + 1) ^ 2 * Fintype.card α
                + (∑ i ∈ Finset.Ico C.walkStart kd, (Fintype.card α : ℝ) ^ i)
                  * (Fintype.card σ + 1)) else 0)
            + C.δ / 8 + (shortStrings (α := α) (Fintype.card σ)).card * cross / u) := by
  classical
  have hδ : 0 < C.δ := hprem.2.1
  obtain ⟨schedule₀, hband₀, hsize₀, hsum₀, hclust₀⟩ := clustering_schedule_exists O K
    {p : FreeMonoid α | p.toList.length < Ls} {v : FreeMonoid α | v.toList.length = Ls}
    η₀ ι gate cross εcov δc slack pAP hη hη₀ (flat_of_length Ls) hpAP
    hι hι1 hg hg1 hcr hε hε1 hδc hδc1 hsl
  set schedule : Finset (Option (Fin K)) → Finset State := fun pops =>
    if none ∈ pops then schedule₀ pops else ∅ with hschedd
  have hband : ∀ pops, ∀ B ∈ schedule pops, B.lo ≤ B.hi ∧ 2 ≤ B.k ∧ ∀ N, N + 1 ≤ B.k →
      ∀ p r ε₂ κ : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → 0 ≤ κ →
        BandPasses N B.lo (B.hi + 1) p r cross ε₂ κ := fun pops B hB => by
    simp only [hschedd] at hB
    split_ifs at hB
    · exact hband₀ pops B hB
    · simp at hB
  set settings : LoopSettings K := ⟨schedule, η₀, ι, εcov, gate, C, a⟩ with hset
  have hNPsum : ∀ pops, ∑ B ∈ settings.schedule pops, B.npref ≤ 2 * NP := fun pops => by
    simp only [hset, hschedd]
    split_ifs with hp
    · exact (hsum₀ pops).trans (Nat.mul_le_mul_left 2 (hNPb pops hp))
    · simp
  refine ⟨settings, rfl, rfl, rfl, rfl, rfl, rfl, hNPsum, ?_⟩
  intro Dsf hDsf hDsfLen hpAPDsf hatom
  set k := C.walkStart
  set T := C.probeBudget
  set N := P.L + T + 2
  have hN : P.L + settings.level3.probeBudget + 2 ≤ N := le_rfl
  set dec := decodeHist P.D k with hdecd
  set μL := loopMeasure μ α K T
  have hr : 0 < K := by omega
  have hsizes : ∀ pops, ∀ B ∈ settings.schedule pops, B.nsuff = poolCount η₀ pops ι εcov (δc / 2)
      pAP (cross ^ 4) ∧ B.npref ≤ NP ∧ none ∈ pops := fun pops B hB => by
    simp only [hset, hschedd] at hB
    split_ifs at hB with hp
    · exact ⟨(hsize₀ pops B hB).1, (hsize₀ pops B hB).2.trans (hNPb pops hp), hp⟩
    · simp at hB
  have hLs' : P.L < Ls := by omega
  have hpopc : ∀ pops : Finset (Option (Fin K)), ((pops.card : ℕ) : ℝ) + 1 ≤ K + 2 := by
    intro pops
    have : pops.card ≤ K + 1 := by
      have := Finset.card_le_univ pops
      simpa using this
    have : ((pops.card : ℕ) : ℝ) ≤ K + 1 := by exact_mod_cast this
    linarith
  have hη1 : O.η ≤ 1 := by linarith
  have hclust : ∀ pops : Finset (Option (Fin K)), none ∈ pops →
      ∀ (L : Option (Fin K) → Measure (FreeMonoid α)) (Lsf : Measure (FreeMonoid α)),
        (∀ j, IsProbabilityMeasure (L j)) → IsProbabilityMeasure Lsf →
        (∀ j ∈ pops, L j {p | Ls ≤ p.toList.length} = 0) →
        (∀ j ∈ pops, ∀ t, (L j).real {t} ≤ 2 * cap) →
        Lsf {v | v.toList.length ≠ Ls} = 0 →
        pAP - 0 ≤ Lsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
        (∀ t, Lsf.real {t} ≤ 2 * cap) →
        (runMeasure μ L Lsf).real
          {x | ¬ ((∃ B ∈ settings.schedule pops, streams (fun _ => ∅) none x
                ∈ ret O.mq pops none settings.η₀ settings.indecisionLimit settings.εcov
                  settings.gateLevel B)
            ∧ ∀ B ∈ settings.schedule pops, streams (fun _ => ∅) none x
              ∈ ret O.mq pops none settings.η₀ settings.indecisionLimit settings.εcov
                settings.gateLevel B →
              ∫ p, miscutProb O B.lo (B.hi + 1)
                  (familyAt O.mq pops (streams (fun _ => ∅) none x) B) p ∂(L none)
                  ≤ 513 / 512 * εcov + (slack - 0)
                ∧ ∀ j ∈ pops, ∫ p, undecidedProb O B.lo (B.hi + 1)
                  (familyAt O.mq pops (streams (fun _ => ∅) none x) B) p
                  ∂(L j) ≤ 2 * settings.indecisionLimit + (slack - 0))} ≤ δc + gate := by
    intro pops hpops L Lsf hLp hLsfp hLs hLat hLsfLen hpAPL hLsfat
    have hs : settings.schedule pops = schedule₀ pops := if_pos hpops
    rw [hs, sub_zero]
    rw [sub_zero] at hpAPL
    have hLat0 : ∀ j t, 0 ≤ (L j).real {t} := fun _ _ => measureReal_nonneg
    have hcolS : collisionMass Lsf ≤ 2 * cap := collisionMass_le_of_atoms Lsf hLsfat
    have hcolS0 : 0 ≤ collisionMass Lsf := tsum_nonneg fun t => sq_nonneg _
    have hNP0 : (0 : ℝ) ≤ NP := Nat.cast_nonneg _
    have hNS0 : (0 : ℝ) ≤ NS := Nat.cast_nonneg _
    refine hclust₀ pops hpops L Lsf hLp hLsfp (fun j hj => ?_) ?_ hpAPL
      (fun B hB j hj t => ?_) (fun B hB j hj t => ?_) (fun B hB => ?_)
    · refine measure_mono_null (fun p hp => ?_) (hLs j hj)
      simp only [Set.mem_compl_iff, Set.mem_setOf_eq, not_lt] at hp ⊢
      exact hp
    · refine measure_mono_null (fun v hv => ?_) hLsfLen
      simpa using hv
    · have hnp : (B.npref : ℝ) ≤ NP := by
        exact_mod_cast (hsizes pops B (by rw [hs]; exact hB)).2.1
      have hc1 := hpopc pops
      have ha' : ((pops.card : ℝ) + 1) ^ 2 ≤ ((K : ℝ) + 2) ^ 2 :=
        pow_le_pow_left₀ (add_nonneg (Nat.cast_nonneg _) zero_le_one) hc1 2
      have hb := mul_le_mul ha' hnp (Nat.cast_nonneg _) (by positivity)
      have hcc := mul_le_mul hb (hLat j hj t) (hLat0 j t) (by positivity)
      calc 2 ^ 21 * ((pops.card : ℝ) + 1) ^ 2 * (B.npref : ℝ) * (L j).real {t}
          = 2 ^ 21 * (((pops.card : ℝ) + 1) ^ 2 * (B.npref : ℝ) * (L j).real {t}) := by ring
        _ ≤ 2 ^ 21 * (((K : ℝ) + 2) ^ 2 * NP * (2 * cap)) :=
            mul_le_mul_of_nonneg_left hcc (by norm_num)
        _ = 2 ^ 22 * ((K : ℝ) + 2) ^ 2 * NP * cap := by ring
        _ ≤ _ := hq1
    · have hnp : (B.npref : ℝ) ≤ NP := by
        exact_mod_cast (hsizes pops B (by rw [hs]; exact hB)).2.1
      have hc1 := hpopc pops
      have hb := mul_le_mul hc1 hnp (Nat.cast_nonneg _) (by positivity)
      have hcc := mul_le_mul hb (hLat j hj t) (hLat0 j t) (by positivity)
      calc 2 ^ 8 * ((pops.card : ℝ) + 1) * (B.npref : ℝ) * (L j).real {t}
          = 2 ^ 8 * (((pops.card : ℝ) + 1) * (B.npref : ℝ) * (L j).real {t}) := by ring
        _ ≤ 2 ^ 8 * (((K : ℝ) + 2) * NP * (2 * cap)) :=
            mul_le_mul_of_nonneg_left hcc (by norm_num)
        _ = 2 ^ 9 * ((K : ℝ) + 2) * NP * cap := by ring
        _ ≤ _ := hq2
    · have hns : (B.nsuff : ℝ) ≤ NS := by
        rw [(hsizes pops B (by rw [hs]; exact hB)).1]
        exact_mod_cast hNSb pops hpops
      have hc1 := hpopc pops
      have hb := mul_le_mul hc1 hns (Nat.cast_nonneg _) (by positivity)
      have hcc := mul_le_mul hb hcolS hcolS0 (by positivity)
      calc 2 ^ 13 * ((pops.card : ℝ) + 1) * (B.nsuff : ℝ) * collisionMass Lsf
          = 2 ^ 13 * (((pops.card : ℝ) + 1) * (B.nsuff : ℝ) * collisionMass Lsf) := by ring
        _ ≤ 2 ^ 13 * (((K : ℝ) + 2) * NS * (2 * cap)) :=
            mul_le_mul_of_nonneg_left hcc (by norm_num)
        _ = 2 ^ 14 * ((K : ℝ) + 2) * NS * cap := by ring
        _ ≤ _ := hq3
  -- the first round's bad events
  set rK : Fin K := ⟨0, hr⟩
  set E1 := {p : Ω × (Fin K → RoundDraws α K T) | HistOK settings O P.D Dsf N p.1 p.2 0
        ∧ (histCodes settings O P.D Dsf N p.1 p.2 0).length = 0
        ∧ (∀ ν' ∈ (dec (histCodes settings O P.D Dsf N p.1 p.2 0)).pops,
          (∀ t, ν'.real {t} ≤ cap) ∧ ν' {q | Ls ≤ q.toList.length} = 0)
        ∧ (∀ j ∈ loopPopulations K 0, (loopDistributions P.D (dec
            (histCodes settings O P.D Dsf N p.1 p.2 0)).pops j).real
              ↑(dec (histCodes settings O P.D Dsf N p.1 p.2 0)).prefs ≤ 0)
        ∧ Dsf.real ↑(dec (histCodes settings O P.D Dsf N p.1 p.2 0)).suffs ≤ 0
        ∧ ¬ ClusterSpec settings O P.D (513 / 512 * εcov) slack (dec
            (histCodes settings O P.D Dsf N p.1 p.2 0)).pops
          (clusterRun P.D Dsf (dec (histCodes settings O P.D Dsf N p.1 p.2 0)) (p.1, (p.2 rK).1))
        ∧ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf N p.1 p.2 0) p.2 0 hr}
  set E2 := {p : Ω × (Fin K → RoundDraws α K T) | HistOK settings O P.D Dsf N p.1 p.2 0
        ∧ (histCodes settings O P.D Dsf N p.1 p.2 0).length = 0
        ∧ Returns settings O P.D Dsf (decodeHist P.D settings.level3.walkStart
            (histCodes settings O P.D Dsf N p.1 p.2 0)) (p.1, (p.2 ⟨0, hr⟩).1)
        ∧ (∀ q, familyReadDist O M (loopRound settings O P.D Dsf (decodeHist P.D
            settings.level3.walkStart (histCodes settings O P.D Dsf N p.1 p.2 0))
              (p.1, p.2 ⟨0, hr⟩)).family (loopRound settings O P.D Dsf (decodeHist P.D
            settings.level3.walkStart (histCodes settings O P.D Dsf N p.1 p.2 0))
              (p.1, p.2 ⟨0, hr⟩)).state.lo ((loopRound settings O P.D Dsf (decodeHist P.D
            settings.level3.walkStart (histCodes settings O P.D Dsf N p.1 p.2 0))
              (p.1, p.2 ⟨0, hr⟩)).state.hi + 1) q .undecided ≤ 3 / 2 * θ)
        ∧ ¬ ZeroSpec M O P (settings.level3.disagreementRate + u + mr)
            settings.level3.walkStart
            (loopRound settings O P.D Dsf (decodeHist P.D settings.level3.walkStart
              (histCodes settings O P.D Dsf N p.1 p.2 0)) (p.1, p.2 ⟨0, hr⟩))
        ∧ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf N p.1 p.2 0) p.2 0 hr
        ∧ ∀ q ∈ prefsOf settings P.D Dsf (histAt settings P.D
            (histCodes settings O P.D Dsf N p.1 p.2 0) 0) (p.2 ⟨0, hr⟩).1,
          ℓ₀ ≤ q.toList.length}
  set E3 := {p : Ω × (Fin K → RoundDraws α K T) |
        (histCodes settings O P.D Dsf N p.1 p.2 0).length = 0
        ∧ (∀ ν ∈ (decodeHist P.D settings.level3.walkStart
            (histCodes settings O P.D Dsf N p.1 p.2 0)).pops, ν {q | Ls ≤ q.toList.length} = 0)
        ∧ (∀ j ∈ loopPopulations K 0, (loopDistributions P.D (decodeHist P.D
            settings.level3.walkStart (histCodes settings O P.D Dsf N p.1 p.2 0)).pops j).real
              ↑(decodeHist P.D settings.level3.walkStart
                (histCodes settings O P.D Dsf N p.1 p.2 0)).prefs ≤ 1 / 2)
        ∧ Dsf.real ↑(decodeHist P.D settings.level3.walkStart
            (histCodes settings O P.D Dsf N p.1 p.2 0)).suffs ≤ 1 / 2
        ∧ ¬ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf N p.1 p.2 0) p.2 0 hr}
  set E4 := {p : Ω × (Fin K → RoundDraws α K T) | ¬ ∀ q ∈ prefsOf settings P.D Dsf (histAt settings
          P.D (histCodes settings O P.D Dsf (P.L + settings.level3.probeBudget + 2) p.1 p.2 0) 0)
          (p.2 ⟨0, hr⟩).1, ℓ₀ ≤ q.toList.length}
  have h1 : μL.real E1 ≤ δc + gate :=
    loop_cluster_round_le_persist settings O P Dsf Ls N (513 / 512 * εcov) (δc + gate) slack cap
      cap 0 0 pAP (by linarith) hN (by norm_num) (by norm_num) hclust hLs' hDat hDsfLen
      hpAPDsf hatom 0 hr
  set δ3 : ℝ := C.δ + (if C.walkStart < kd then 3 / 2 * θ * (C.probeBudget * (P.L + 1)
      * Fintype.card σ + Fintype.card σ * (Fintype.card σ + 1) ^ 2 * Fintype.card α
        + (∑ i ∈ Finset.Ico C.walkStart kd, (Fintype.card α : ℝ) ^ i)
          * (Fintype.card σ + 1)) else 0)
      + C.δ / 8 + (shortStrings (α := α) (Fintype.card σ)).card * cross / u with hδ3d
  have h2 : μL.real E2 ≤ δ3 :=
    loop_round_zero M O hL P Dsf settings Ls NP N ℓ₀ kd θ mr u cross hN hNPsum hband ha hka hlong
      hη1 hcr hprem hu hmr hmr1 hr
  have h3 : μL.real E3 = 0 := by
    rw [measureReal_def, loop_coll_null settings O P Dsf Ls N hN hLs' hDsfLen 0 hr,
      ENNReal.toReal_zero]
  have h4 : μL.real E4 = 0 := by
    rw [measureReal_def, longPrefs_null settings O P Dsf ℓ₀ hshort hr, ENNReal.toReal_zero]
  -- a failing run fails one of them
  have hsub : {p : Ω × (Fin K → RoundDraws α K T) |
      let rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p)
      ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
          o.Checks M O P.D C.walkStart (513 / 512 * εcov + 2 * ι + 2 * slack) (2 * ι + slack)
            (C.disagreementRate + u + mr))
        ∧ RoundsFresh settings O P.D Dsf N Ls p)}
      ⊆ ((E1 ∪ E2) ∪ E3) ∪ E4 := by
    intro p hp
    by_contra hno
    simp only [Set.mem_union, not_or] at hno
    obtain ⟨⟨⟨hn1, hn2⟩, hn3⟩, hn4⟩ := hno
    have hc0 : histCodes settings O P.D Dsf N p.1 p.2 0 = [] := rfl
    have hd0 : dec (histCodes settings O P.D Dsf N p.1 p.2 0) = ⟨[], ∅, ∅⟩ := rfl
    have hd1 : decodeHist P.D settings.level3.walkStart (histCodes settings O P.D Dsf N p.1 p.2 0)
        = ⟨[], ∅, ∅⟩ := rfl
    have hok : HistOK settings O P.D Dsf N p.1 p.2 0 := trivial
    have hlen : (histCodes settings O P.D Dsf N p.1 p.2 0).length = 0 := rfl
    have hlp : ∀ q ∈ prefsOf settings P.D Dsf (histAt settings P.D
        (histCodes settings O P.D Dsf N p.1 p.2 0) 0) (p.2 ⟨0, hr⟩).1, ℓ₀ ≤ q.toList.length := by
      by_contra h; exact hn4 h
    have hnc : NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf N p.1 p.2 0) p.2 0 hr := by
      by_contra h
      refine hn3 ⟨hlen, fun ν hν => ?_, fun j _ => ?_, ?_, h⟩
      · rw [hd1] at hν; simp at hν
      · rw [hd1]; simp
      · rw [hd1]; simp
    have hcs : ClusterSpec settings O P.D (513 / 512 * εcov) slack
        (dec (histCodes settings O P.D Dsf N p.1 p.2 0)).pops
        (clusterRun P.D Dsf (dec (histCodes settings O P.D Dsf N p.1 p.2 0)) (p.1, (p.2 rK).1)) := by
      by_contra h
      refine hn1 ⟨hok, hlen, fun ν hν => ?_, fun j _ => ?_, ?_, h, hnc⟩
      · rw [hd0] at hν; simp at hν
      · rw [hd0]; simp
      · rw [hd0]; simp
    set H := dec (histCodes settings O P.D Dsf N p.1 p.2 0)
    set o := loopRound settings O P.D Dsf H (p.1, p.2 rK)
    obtain ⟨hmem, hmis, hund⟩ := loopRound_quality settings O P.D Dsf H (p.1, p.2 rK) hcs
    have hlh : o.state.lo ≤ o.state.hi := (hband _ _ hmem).1
    have h1F : 1 ∉ o.family := one_notMem_loopRound_family settings O P.D Dsf H (p.1, p.2 rK)
    have hnone : none ∈ loopPopulations K H.pops.length := by simp [loopPopulations]
    have hund' := hund none hnone
    simp only [loopDistributions] at hund'
    -- no state reads undecided more than `1.5θ` of the time
    have hnob : ∀ q', familyReadDist O M o.family o.state.lo (o.state.hi + 1) q' .undecided
        ≤ 3 / 2 * θ := by
      intro q'
      exact (undecided_le_of_start M O hL L hc hq hstart h1F o.state.lo o.state.hi
        (u := 2 * ι + slack) (hPD ▸ hund') q').trans hθc
    have hzs : ZeroSpec M O P (C.disagreementRate + u + mr) k o := by
      by_contra h
      exact hn2 ⟨hok, hlen, hcs.1, hnob, h, hnc, hlp⟩
    obtain ⟨hcons, hdis, hcls⟩ := hzs
    -- the loop is that one round
    set Lr := loopRounds settings O P.D Dsf (persistDraws p) K with hLr
    have hpos : 0 < Lr.length := by
      rcases loopRounds_stops settings O P.D Dsf (persistDraws p) with h | ⟨o', ho', -⟩
      · rw [← hLr] at h; omega
      · rw [← hLr] at ho'
        rcases hl : Lr with _ | ⟨x, xs⟩
        · rw [hl] at ho'; simp at ho'
        · simp
    obtain ⟨hK0, -, -, e⟩ := persist_bridge settings O P.D Dsf P.length_le hN p 0 hpos
    have he0 : Lr[0] = o := e
    have hl1 : Lr.length = 1 := by
      by_contra hne
      have h1 : 0 + 1 < Lr.length := by omega
      have h2 : Lr[0].result.ending.harvested ≠ none :=
        harvested_of_lt settings O P.D Dsf (persistDraws p) K 0 h1
      rw [he0, hcons] at h2
      exact h2 rfl
    -- its checks
    have hb := (integrable_of_unit P.D (fun p => miscutProb O o.state.lo
      (o.state.hi + 1) (insert 1 o.family) p) (fun _ => measureReal_nonneg)
      (fun _ => measureReal_le_one)).add (integrable_of_unit P.D (fun p => undecidedProb O
        o.state.lo (o.state.hi + 1) (insert 1 o.family) p)
        (fun _ => measureReal_nonneg) (fun _ => measureReal_le_one))
    have hchk : o.Checks M O P.D k (513 / 512 * εcov + 2 * ι + 2 * slack) (2 * ι + slack)
        (C.disagreementRate + u + mr) := by
      refine ⟨⟨hcons, ?_, ?_⟩, hdis, hcls⟩
      · calc ∫ p, misreadProb O o.family o.state.lo (o.state.hi + 1) p ∂P.D
            ≤ ∫ p, (miscutProb O o.state.lo (o.state.hi + 1) (insert 1 o.family) p
                + undecidedProb O o.state.lo (o.state.hi + 1) (insert 1 o.family) p) ∂P.D :=
              integral_mono (integrable_of_unit P.D _ (fun _ => measureReal_nonneg)
                (fun _ => measureReal_le_one)) hb (fun p => misreadProb_le O h1F hlh p)
          _ = ∫ p, miscutProb O o.state.lo (o.state.hi + 1) (insert 1 o.family) p ∂P.D
              + ∫ p, undecidedProb O o.state.lo (o.state.hi + 1) (insert 1 o.family) p ∂P.D :=
              integral_add (integrable_of_unit P.D _ (fun _ => measureReal_nonneg)
                (fun _ => measureReal_le_one)) (integrable_of_unit P.D _
                  (fun _ => measureReal_nonneg) (fun _ => measureReal_le_one))
          _ ≤ 513 / 512 * εcov + 2 * ι + 2 * slack := by
              have := hmis
              linarith
      · calc ∫ p, readProb O o.family o.state.lo (o.state.hi + 1) p .undecided ∂P.D
            ≤ ∫ p, undecidedProb O o.state.lo (o.state.hi + 1) (insert 1 o.family) p ∂P.D :=
              integral_mono (integrable_of_unit P.D _ (fun _ => measureReal_nonneg)
                (fun _ => measureReal_le_one)) (integrable_of_unit P.D _
                  (fun _ => measureReal_nonneg) (fun _ => measureReal_le_one))
                (fun p => readProb_undecided_le O h1F _ _ p)
          _ ≤ 2 * ι + slack := hund'
    apply hp
    refine ⟨⟨?_, o, ?_, hchk⟩, fun i hi => ?_⟩
    · show Lr.length ≤ _
      omega
    · show o ∈ Lr.getLast?
      rw [List.getLast?_eq_getElem?, hl1]
      simp [← he0]
    · have hi0 : i = 0 := by
        have : i < Lr.length := hi
        omega
      subst hi0
      exact ⟨hr, hok, hcs.1, hnc⟩
  calc μL.real _ ≤ μL.real (((E1 ∪ E2) ∪ E3) ∪ E4) := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ μL.real E1 + μL.real E2 + μL.real E3 + μL.real E4 :=
        (measureReal_union_le _ _).trans (add_le_add ((measureReal_union_le _ _).trans
          (add_le_add (measureReal_union_le _ _) le_rfl)) le_rfl)
    _ ≤ δc + gate + δ3 + 0 + 0 := by rw [h3, h4]; linarith
    _ = _ := by ring

end Core

end OrthoDFA

end
