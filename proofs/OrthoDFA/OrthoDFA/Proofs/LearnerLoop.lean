import OrthoDFA.Proofs.LoopColl
import OrthoDFA.Proofs.ClusteringQuality
import Mathlib.Data.List.GetD

/-!
# The loop's theorem

The whole run reads one noise. Round `r` fails the clustering's claim, given the earlier rounds,
but for chance `δc` (`loop_cluster_round_le_persist`), level 5's but for chance `δ`
(`loop_level5_round_le_persist`), and repeats a string an earlier round drew but for chance `δc`
(`loop_coll_le`). A round meeting level 5's claim hands on a population putting at most
`prefixMax / (ρ/4)` on any string, within the cap. Where every one of the first `|Q| + 1` rounds
meets both claims, the loop stops in agreement (`loop_agrees_of_specs`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-! ## The clustering's schedule -/

/-- One schedule and collision cap serve every set of the loop's populations, and every state of
the schedule reads a family of at most `k - 1` members at a band whose far side `crossLimit`
bounds, whatever the read's two rates. -/
theorem clustering_schedule_exists {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) (K : ℕ) (Pre Suf : Set (FreeMonoid α))
    (η₀ indecisionLimit gateLevel crossLimit εcov δc slack pAP : ℝ)
    (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hflat : Flat Pre Suf) (hpAP : 0 < pAP)
    (hι : 0 < indecisionLimit) (hι1 : indecisionLimit ≤ 1 / 2) (hg : 0 < gateLevel)
    (hg1 : gateLevel < 1 / 2) (hcross : 0 < crossLimit) (hε : 0 < εcov) (hε1 : εcov ≤ 1)
    (hδc : 0 < δc) (hδc1 : δc ≤ 1) (hslack : 0 < slack) (cap : ℝ)
    (hcap : ∀ pops : Finset (Option (Fin K)), none ∈ pops →
      cap ≤ qualityCap η₀ pops indecisionLimit εcov gateLevel δc pAP
        (Real.exp (-2 / crossLimit)) slack) :
    ∃ schedule : Finset (Option (Fin K)) → Finset State,
      (∀ pops, ∀ B ∈ schedule pops, B.lo ≤ B.hi ∧ 2 ≤ B.k ∧ ∀ N, N + 1 ≤ B.k → ∀ p r ε₂ κ : ℝ,
        0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → 0 ≤ κ →
        BandPasses N B.lo (B.hi + 1) p r crossLimit ε₂ κ)
      ∧ (∀ pops, ∀ B ∈ schedule pops,
        B.nsuff = poolCount η₀ pops indecisionLimit εcov (δc / 2) pAP (Real.exp (-2 / crossLimit))
        ∧ B.npref ≤ prefCount η₀ pops indecisionLimit εcov (δc / 2) gateLevel pAP
          (Real.exp (-2 / crossLimit)))
      ∧ ∀ pops : Finset (Option (Fin K)), none ∈ pops →
        ∀ (D : Option (Fin K) → Measure (FreeMonoid α)) (Dsf : Measure (FreeMonoid α)),
          (∀ j, IsProbabilityMeasure (D j)) → IsProbabilityMeasure Dsf →
          (∀ j ∈ pops, D j Preᶜ = 0) → Dsf Sufᶜ = 0 →
          pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
          (∀ j ∈ pops, collisionMass (D j) ≤ cap) → collisionMass Dsf ≤ cap →
          (runMeasure μ D Dsf).real
            {x | ¬ ((∃ B ∈ schedule pops, x ∈ ret O.mq pops none indecisionLimit gateLevel B)
              ∧ ∀ B ∈ schedule pops, x ∈ ret O.mq pops none indecisionLimit gateLevel B →
                ∫ p, miscutProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p ∂(D none)
                    ≤ εcov + slack
                  ∧ ∀ j ∈ pops, ∫ p, undecidedProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p
                    ∂(D j) ≤ 2 * indecisionLimit + slack)} ≤ δc := by
  classical
  obtain ⟨c, hc⟩ : ∃ c : ℝ, c = Real.exp (-2 / crossLimit) := ⟨_, rfl⟩
  have hc0 : 0 < c := hc ▸ Real.exp_pos _
  rw [← hc] at hcap
  refine ⟨fun pops =>
    stoppable η₀ pops indecisionLimit εcov (δc / 2) gateLevel pAP c cap, ?_, ?_, ?_⟩
  · intro pops B hB
    obtain ⟨hlohi, hk2, hband⟩ := band_of_mem_schedule (le_trans (eta_nonneg O) hη) hη₀
      (Finset.mem_of_mem_filter _ hB)
    refine ⟨hlohi, hk2, fun N hN p r ε₂ κ hp0 hp1 hr0 hr1 _ =>
      bandPasses_of_wide hp0 hp1 hr0 hr1 hlohi ?_⟩
    have hlog : Real.log (1 / c) = 2 / crossLimit := by
      rw [hc, one_div, ← Real.exp_neg, Real.log_exp]; ring
    rw [hlog, mul_div_assoc', div_le_iff₀ hcross] at hband
    have hNk : (N : ℝ) + 1 ≤ B.k := by exact_mod_cast hN
    have hlh : (B.lo : ℝ) ≤ B.hi := by exact_mod_cast hlohi
    have h2 : ((B.hi : ℝ) - B.lo) ^ 2 ≤ ((B.hi : ℝ) + 1 - B.lo) ^ 2 := by nlinarith
    nlinarith [mul_le_mul_of_nonneg_left h2 hcross.le]
  · intro pops B hB
    have hs := Finset.mem_of_mem_filter _ hB
    refine ⟨by rw [← hc]; exact nsuff_of_mem_schedule hs, ?_⟩
    obtain ⟨i, -, hi⟩ := Finset.mem_image.1 hs
    rw [← hi, ← hc]
    exact Nat.div_le_self _ _
  · intro pops hpops D Dsf hD hDsf hsupp hsuppSf hpAPDsf hcolD hcolSf
    obtain ⟨-, hq⟩ := clustering_quality_bad O pops none Pre Suf η₀ indecisionLimit εcov gateLevel
      δc pAP c slack hη hη₀ hpops hflat hpAP hι hι1 hg hg1 hε hε1 hδc hδc1 hc0 hslack
    obtain ⟨-, hbad⟩ := hq D Dsf hD hDsf hsupp hsuppSf hpAPDsf cap hcolD
      (hcap pops hpops) (hcolSf.trans (hcap pops hpops))
    refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) hbad
    intro x hx hgood
    apply hx
    obtain ⟨⟨B, hret⟩, hall⟩ := hgood
    refine ⟨⟨B.val, B.property, hret⟩, fun B' hB' hret' => ?_⟩
    exact ⟨(hall ⟨B', hB'⟩ hret' none hpops).1, fun j hj => (hall ⟨B', hB'⟩ hret' j hj).2⟩

/-! ## The theorem -/

section Pops

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ}

omit [MeasurableSpace Ω] in
/-- A round's population is supported on strings shorter than `Ls`. -/
lemma population_short (P : PrefixDistribution α) (C : Level5Settings) (hC : 2 ≤ C.maxLeaves)
    {Ls : ℕ} (hLs : P.L + C.maxLeaves ≤ Ls) (k : ℕ) (o : LoopRound Ω α K) {T : ℕ}
    (xs : Fin T → FreeMonoid α) (ho : o.result = indecisiveLStarLevel5 C o.read xs)
    {ν : Measure (FreeMonoid α)} (hν : o.population P.D k = some ν) :
    ν {p | Ls ≤ p.toList.length} = 0 := by
  unfold LoopRound.population at hν
  rcases hh : o.result.ending.harvested with _ | h
  · rw [hh] at hν; cases hν
  rw [hh, Option.map_some, Option.some.injEq] at hν
  subst hν
  have hend : (indecisiveLStarLevel5 C o.read xs).ending = .harvest h := by
    rw [← ho]
    revert hh
    cases o.result.ending <;> simp [RoundEnd.harvested]
  refine popLaw_null (measure_mono_null (fun p hp => ?_)
    (harvestLaw_short C (fun z => (o.read z).cut) P.length_le k
      (harvest_short C hC o.read xs hend))) ?_
  · simp only [Set.mem_ofPred_eq] at hp ⊢
    omega
  · simp only [Set.mem_ofPred_eq, FreeMonoid.toList_one, List.length_nil, not_le]
    omega

end Pops

lemma flat_of_length (Ls : ℕ) :
    Flat {p : FreeMonoid α | p.toList.length < Ls} {v : FreeMonoid α | v.toList.length = Ls} := by
  intro p hp p' hp' v hv v' hv' e
  have hp : p.toList.length < Ls := hp
  have hp' : p'.toList.length < Ls := hp'
  have hv : v = 1 ∨ v.toList.length = Ls := hv
  have hv' : v' = 1 ∨ v'.toList.length = Ls := hv'
  have hl : p.toList ++ v.toList = p'.toList ++ v'.toList := by
    rw [← FreeMonoid.toList_mul, ← FreeMonoid.toList_mul, e]
  have hlen : p.toList.length + v.toList.length = p'.toList.length + v'.toList.length := by
    rw [← List.length_append, ← List.length_append, hl]
  have hpp : p.toList.length = p'.toList.length := by
    rcases hv with rfl | hv <;> rcases hv' with rfl | hv' <;>
      first | omega | (simp only [FreeMonoid.toList_one, List.length_nil] at hlen; omega)
  exact FreeMonoid.toList.injective (List.append_inj_left hl hpp)


lemma le_prefixMax_of_spread {D : Measure (FreeMonoid α)} [IsProbabilityMeasure D] {k : ℕ}
    {ν : Measure (FreeMonoid α)} (h : Spread D k ν) (t : FreeMonoid α) :
    ν.real {t} ≤ prefixMax D k := by
  refine (h t).trans (le_of_eq_of_le ?_ (le_prefixMax D k (prefixOf t k)))
  congr 1


lemma real_singleton_le_prefixMax (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (t : FreeMonoid α) : D.real {t} ≤ prefixMax D k :=
  (measureReal_mono (fun x hx => by rw [Set.mem_singleton_iff.1 hx]; rfl)
    (measure_ne_top _ _)).trans (le_prefixMax D k (prefixOf t k))

lemma popLaw_atom_le {ν : Measure (FreeMonoid α)} {q y : ℝ} (hy : 0 < y)
    (hyν : y ≤ ν.real Set.univ) (htop : ν Set.univ ≠ ∞) (hq : ∀ t, ν.real {t} ≤ q) (t : FreeMonoid α) :
    (popLaw ν).real {t} ≤ q / y := by
  have hpos : 0 < ν.real Set.univ := lt_of_lt_of_le hy hyν
  have h0 : ν Set.univ ≠ 0 := fun h => by
    rw [measureReal_def, h, ENNReal.toReal_zero] at hpos; exact lt_irrefl _ hpos
  have hq0 : 0 ≤ q := measureReal_nonneg.trans (hq 1)
  rw [popLaw_real h0 htop]
  calc ν.real {t} / ν.real Set.univ ≤ q / ν.real Set.univ :=
        div_le_div_of_nonneg_right (hq t) hpos.le
    _ ≤ q / y := div_le_div_of_nonneg_left hq0 hy hyν

lemma collisionMass_le_of_atoms {ν : Measure (FreeMonoid α)} [IsProbabilityMeasure ν] {q : ℝ}
    (hq : ∀ t, ν.real {t} ≤ q) : collisionMass ν ≤ q := by
  have hq0 : 0 ≤ q := measureReal_nonneg.trans (hq 1)
  unfold collisionMass
  refine tsum_le_of_sum_le' hq0 fun s => ?_
  calc ∑ t ∈ s, ν.real {t} ^ 2 ≤ ∑ t ∈ s, q * ν.real {t} :=
        Finset.sum_le_sum fun t _ => by
          rw [sq]; exact mul_le_mul_of_nonneg_right (hq t) measureReal_nonneg
    _ = q * ν.real s := by rw [← Finset.mul_sum, sum_measureReal_singleton]
    _ ≤ q * 1 := mul_le_mul_of_nonneg_left measureReal_le_one hq0
    _ = q := mul_one _

section Bridge

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}
  (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D : Measure (FreeMonoid α))
  {L N : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hC : 2 ≤ settings.level5.maxLeaves)
  (hN : L + settings.level5.maxLeaves ≤ N)

omit [IsProbabilityMeasure μ] in
include hlen hC hN in
/-- Round `i` of the persistent run is the round after the data the rounds before it hand on. -/
lemma persist_bridge (p : Ω × (Fin K → RoundDraws α K settings.level5.probeBudget)) (i : ℕ)
    (hi : i < (loopRounds settings O D (persistDraws p) K).length) :
    ∃ hK : i < K, (histCodes settings O D N p.1 p.2 i).length = i
      ∧ (histCodes settings O D N p.1 p.2 i).map (decodePop D settings.level5.walkStart)
        = harvestsOf D settings.level5.walkStart
          ((loopRounds settings O D (persistDraws p) K).take i)
      ∧ (loopRounds settings O D (persistDraws p) K)[i] = loopRound settings O D
        ((histCodes settings O D N p.1 p.2 i).map (decodePop D settings.level5.walkStart))
        (p.1, p.2 ⟨i, hK⟩) := by
  set draws := persistDraws p
  obtain ⟨hK, e⟩ := getElem_loopRounds settings O D draws K i hi
  have hiK : i ≤ K := (hi.trans_le (length_loopRounds_le settings O D draws K)).le
  have htake := take_loopRounds settings O D draws hiK
  have hdec := codesOf_decode settings.level5 hlen hC hN settings.level5.walkStart
    (loopRounds settings O D draws i) (loopRounds_result settings O D draws i)
  have hall : AllHarvested (loopRounds settings O D draws i) := by
    intro o ho
    rw [← htake] at ho
    obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem ho
    rw [List.getElem_take]
    simp only [List.length_take] at hj
    exact harvested_of_lt settings O D draws K j (by omega)
  have hc : histCodes settings O D N p.1 p.2 i = codesOf N (loopRounds settings O D draws i) := rfl
  refine ⟨hK, ?_, ?_, ?_⟩
  · rw [hc, length_codesOf N D settings.level5.walkStart, length_harvestsOf hall,
      length_eq_of_allHarvested settings O D draws hiK hall]
  · rw [hc, hdec, htake]
  · rw [e, hc, hdec, htake]
    rfl

include hlen hC hN in
/-- Every earlier clustering returned, where each met its claim. -/
lemma histOK_of_specs (εcov slack : ℝ)
    (p : Ω × (Fin K → RoundDraws α K settings.level5.probeBudget)) :
    ∀ i, i ≤ (loopRounds settings O D (persistDraws p) K).length →
      (∀ j (hj : j < (loopRounds settings O D (persistDraws p) K).length), j < i →
        ClusterSpec settings O D εcov slack (harvestsOf D settings.level5.walkStart
          ((loopRounds settings O D (persistDraws p) K).take j))
          (loopRounds settings O D (persistDraws p) K)[j].run) →
      HistOK settings O D N p.1 p.2 i
  | 0, _, _ => trivial
  | i + 1, hi, h => by
    refine ⟨histOK_of_specs εcov slack p i (by omega) fun j hj hji => h j hj (by omega),
      fun hK hlen' => ?_⟩
    obtain ⟨hK', -, hdec, e⟩ := persist_bridge settings O D hlen hC hN p i (by omega)
    have hs := (h i (by omega) (by omega)).1
    rw [← hdec, e] at hs
    exact hs

end Bridge

/-- The loop's claim at a cap `cap` and walk start `k`, `NS` and `NP` bounding the schedule's
suffix pool and prefix count. -/
def LoopCore : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (K S Ls NS NP : ℕ) (η₀ indecisionLimit gateLevel εcov δc slack pAP εd δ cap : ℝ) (k : ℕ),
  let n : ℝ := Fintype.card σ + S
  let A : ℝ := Fintype.card α
  let ρ : ℝ := (S + 1) * εd / (256 * (n + 1) * (Fintype.card σ * A + S + 1))
  let Kp : ℝ := (n + 1) * (2 * (n + 3) * A + 1) * (80 / εd ^ 3 + 1080 / εd ^ 2 + 2800 / εd)
    + 8 * (n + 3) * A * (P.L + 1)
  let X : ℝ := 4 * (2 ^ (Fintype.card σ + S + 5) * A + 7 * Kp ^ 2) / δ
  let G : ℝ := Real.log X + 2 * Real.log (2 * Real.log X)
  let c : ℝ := Real.exp (-(8 * (P.L + 1) * (n + 1) / ρ))
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 →
  0 < indecisionLimit → indecisionLimit ≤ 1 / 2 →
  0 < gateLevel → gateLevel < 1 / 2 →
  0 < εcov → εcov ≤ 1 →
  0 < δc → δc ≤ 1 →
  0 < slack → 0 < pAP →
  0 < εd → εd ≤ 1 → 0 < δ → δ ≤ 1 → 1 ≤ P.L →
  Fintype.card σ + 1 ≤ K →
  P.L + Fintype.card σ + S + 3 ≤ Ls →
  1024 * (P.L + 1) * (n + 1) * Fintype.card σ * (2 * indecisionLimit + slack) ≤ 1 →
  0 < cap →
  (∀ pops : Finset (Option (Fin K)), none ∈ pops →
    cap ≤ qualityCap η₀ pops indecisionLimit εcov gateLevel δc pAP c slack) →
  (∀ pops : Finset (Option (Fin K)), none ∈ pops →
    poolCount η₀ pops indecisionLimit εcov (δc / 2) pAP c ≤ NS) →
  (∀ pops : Finset (Option (Fin K)), none ∈ pops →
    prefCount η₀ pops indecisionLimit εcov (δc / 2) gateLevel pAP c ≤ NP) →
  3 * (K + 1) * NP * (P.L + 1) * cap ≤ 1 →
  K * (NS ^ 2 + 16 * (K + 1) ^ 2 * NP ^ 2) * cap ≤ δc →
  1 ≤ k → prefixMax P.D k ≤ cap * ρ / 4 →
  24 * unionLog (α := α) σ S δ * prefixMax P.D k ≤ ρ →
  ∃ settings : LoopSettings K,
    settings.indecisionLimit = indecisionLimit ∧ settings.gateLevel = gateLevel ∧
    settings.level5.walkStart = k ∧ settings.level5.disagreementRate = εd / 2 ∧
    (settings.level5.probeBudget : ℝ) ≤ Kp * G ∧
    ∀ Dsf : Measure (FreeMonoid α), IsProbabilityMeasure Dsf →
      Dsf {v | v.toList.length ≠ Ls} = 0 →
      pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
      (∀ v, Dsf.real {v} ≤ cap) →
      (loopMeasure μ P.D Dsf K settings.level5.probeBudget).real
          {p | let rounds := indecisiveLStarLoop settings O P.D (persistDraws p)
            ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
                o.Agrees O P.D k (εcov + 2 * indecisionLimit + 2 * slack)
                  (2 * indecisionLimit + slack) εd)}
        ≤ (Fintype.card σ + 1) * (2 * δc + δ)

set_option maxHeartbeats 1000000 in -- the three round bounds and the induction
theorem loop_core_holds : LoopCore := by
  intro α _ _ _ σ _ Ω _ μ _ M O P K S Ls NS NP η₀ ι gate εcov δc slack pAP εd δ cap k n A ρ Kp X G
    c hL hη hη₀ hι hι1 hg hg1 hε hε1 hδc hδc1 hsl hpAP hεd hεd1 hδ hδ1 hL1 hK hLs hθ hcap hcapq
    hNSb hNPb hcapP hcapC hk0 hk1 hk2
  classical
  have hA1 : (1 : ℝ) ≤ A := by
    show (1 : ℝ) ≤ Fintype.card α
    exact_mod_cast Fintype.card_pos
  have hn0 : (0 : ℝ) ≤ n := by positivity
  have hρ0 : 0 < ρ := by positivity
  have hρ1 : ρ ≤ εd / 256 := by
    have h1 : ((S : ℝ) + 1) ≤ (n + 1) * (Fintype.card σ * A + S + 1) := by
      have : (0 : ℝ) ≤ Fintype.card σ * A := by positivity
      nlinarith
    show (S + 1) * εd / (256 * (n + 1) * (Fintype.card σ * A + S + 1)) ≤ εd / 256
    rw [div_le_div_iff₀ (by positivity) (by norm_num)]
    nlinarith
  set cross : ℝ := ρ / (4 * (P.L + 1) * (n + 1)) with hcrossd
  have hcr : 0 < cross := by positivity
  have hc : Real.exp (-2 / cross) = c := by
    show Real.exp (-2 / cross) = Real.exp (-(8 * (P.L + 1) * (n + 1) / ρ))
    congr 1
    rw [hcrossd]
    field_simp
    ring
  obtain ⟨schedule₀, hband₀, hsize₀, hclust₀⟩ := clustering_schedule_exists O K
    {p : FreeMonoid α | p.toList.length < Ls} {v : FreeMonoid α | v.toList.length = Ls}
    η₀ ι gate cross εcov δc slack pAP hη hη₀ (flat_of_length Ls) hpAP hι hι1 hg hg1 hcr hε hε1
    hδc hδc1 hsl cap (fun pops hp => by rw [hc]; exact hcapq pops hp)
  -- the loop's populations always hold `D`'s
  set schedule : Finset (Option (Fin K)) → Finset State := fun pops =>
    if none ∈ pops then schedule₀ pops else ∅ with hschedd
  have hband : ∀ pops, ∀ B ∈ schedule pops, B.lo ≤ B.hi ∧ 2 ≤ B.k ∧ ∀ N, N + 1 ≤ B.k →
      ∀ p r ε₂ κ : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → 0 ≤ κ →
        BandPasses N B.lo (B.hi + 1) p r cross ε₂ κ := fun pops B hB => by
    simp only [hschedd] at hB
    split_ifs at hB
    · exact hband₀ pops B hB
    · simp at hB
  obtain ⟨s4, hwalk, hdis, hbud, hmax, hrec, hl4⟩ :=
    level5_guarantee_all M (εd / 2) P.L δ S k hδ hδ1 (by positivity) (by linarith) hL1
  set settings : LoopSettings K := ⟨schedule, ι, gate, s4⟩ with hset
  refine ⟨settings, rfl, rfl, hwalk, hdis, ?_, ?_⟩
  · refine hbud.trans (le_of_eq ?_)
    have e : ((Fintype.card σ : ℝ) + S + 1) * (2 * ((Fintype.card σ : ℝ) + S + 3)
        * Fintype.card α + 1) * (10 / (εd / 2) ^ 3 + 270 / (εd / 2) ^ 2 + 1400 / (εd / 2))
        + 8 * ((Fintype.card σ : ℝ) + S + 3) * Fintype.card α * (P.L + 1) = Kp := by
      simp only [Kp, n, A]
      field_simp
      ring
    simp only [e]
    rfl
  intro Dsf hDsf hDsfLen hpAPDsf hatom
  subst hwalk
  set k := s4.walkStart
  set T := s4.probeBudget
  set N := P.L + s4.maxLeaves
  have hC : 2 ≤ s4.maxLeaves := by rw [hmax]; omega
  have hN : P.L + settings.level5.maxLeaves ≤ N := le_rfl
  set dec := decodePop P.D k with hdecd
  set θ := loopTheta σ ι slack
  set μL := loopMeasure μ P.D Dsf K T
  have hpm0 : 0 ≤ prefixMax P.D k := measureReal_nonneg.trans (le_prefixMax P.D k 1)
  have hρ4 : ρ / 4 ≤ 1 := by linarith
  have hcap4 : cap * ρ / 4 ≤ cap := by
    calc cap * ρ / 4 = cap * (ρ / 4) := by ring
      _ ≤ cap * 1 := mul_le_mul_of_nonneg_left hρ4 hcap.le
      _ = cap := mul_one _
  have hNS : ∀ pops, schedNS (settings.schedule pops) ≤ NS := fun pops =>
    Finset.sup_le fun B hB => by
      simp only [hset, hschedd] at hB
      split_ifs at hB with hp
      · rw [(hsize₀ pops B hB).1, hc]; exact hNSb pops hp
      · simp at hB
  have hNP : ∀ pops, schedNP (settings.schedule pops) ≤ NP := fun pops =>
    Finset.sup_le fun B hB => by
      simp only [hset, hschedd] at hB
      split_ifs at hB with hp
      · exact (hsize₀ pops B hB).2.trans (by rw [hc]; exact hNPb pops hp)
      · simp at hB
  have hcolSf : collisionMass Dsf ≤ cap := collisionMass_le_of_atoms hatom
  have hDat : ∀ t, P.D.real {t} ≤ cap := fun t =>
    (real_singleton_le_prefixMax P.D k t).trans (hk1.trans hcap4)
  have hcolD : collisionMass P.D ≤ cap := collisionMass_le_of_atoms hDat
  have hLs' : P.L < Ls := by omega
  -- level 5's yield, at rate `εd / 2`
  set y : ℝ := (S + 1) * (εd / 2) / (256 * ((Fintype.card σ : ℝ) + S + 1)
    * ((Fintype.card σ : ℝ) * Fintype.card α + S + 1)) with hyd
  have hyρ : y = ρ / 2 := by
    simp only [hyd, ρ, n, A]
    ring
  have hy0 : 0 < y := by rw [hyρ]; positivity
  set m : ℝ := 3 * (K + 1) * NP * prefixMax P.D k with hmd
  have hm0 : 0 ≤ m := by positivity
  have hmρ : (P.L + 1) * m ≤ ρ / 4 := by
    calc (P.L + 1) * m = ((P.L : ℝ) + 1) * (3 * (K + 1) * NP) * prefixMax P.D k := by
          simp only [hmd]; ring
      _ ≤ ((P.L : ℝ) + 1) * (3 * (K + 1) * NP) * (cap * ρ / 4) :=
          mul_le_mul_of_nonneg_left hk1 (by positivity)
      _ = 3 * (K + 1) * NP * (P.L + 1) * cap * (ρ / 4) := by ring
      _ ≤ 1 * (ρ / 4) := mul_le_mul_of_nonneg_right hcapP (by positivity)
      _ = ρ / 4 := one_mul _
  have hmle : m ≤ ρ / 4 := by
    have h1 : m ≤ (P.L + 1) * m := by
      have : (1 : ℝ) ≤ P.L + 1 := by linarith [(Nat.cast_nonneg P.L : (0 : ℝ) ≤ P.L)]
      calc m = 1 * m := (one_mul _).symm
        _ ≤ (P.L + 1) * m := mul_le_mul_of_nonneg_right this hm0
    linarith
  -- each round's three bounds
  have hη1 : O.η ≤ 1 := by linarith
  have hθ0 : 0 ≤ θ := by unfold θ loopTheta; positivity
  have hθ' : 384 * (P.L + 1) * (Fintype.card σ + S + 1) * θ ≤ 1 := by
    have e : 384 * ((P.L : ℝ) + 1) * (Fintype.card σ + S + 1) * θ
        = 1024 * (P.L + 1) * (n + 1) * Fintype.card σ * (2 * ι + slack) := by
      simp only [θ, loopTheta, n]; ring
    rw [e]; exact hθ
  have hε' : (P.L + 1) * (Fintype.card σ + S + 1) * cross
      + 3 * unionLog (α := α) σ S δ * (2 * prefixMax P.D k) ≤ y := by
    have e1 : ((P.L : ℝ) + 1) * (Fintype.card σ + S + 1) * cross = ρ / 4 := by
      rw [hcrossd]; simp only [n]; field_simp
    rw [e1, hyρ]
    linarith
  have h1 : ∀ r (hr : r < K), μL.real {p | HistOK settings O P.D N p.1 p.2 r
        ∧ (histCodes settings O P.D N p.1 p.2 r).length = r
        ∧ (∀ ν' ∈ (histCodes settings O P.D N p.1 p.2 r).map dec,
          collisionMass ν' ≤ cap ∧ ν' {q | Ls ≤ q.toList.length} = 0)
        ∧ ¬ ClusterSpec settings O P.D εcov slack ((histCodes settings O P.D N p.1 p.2 r).map dec)
          (clusterRun ((histCodes settings O P.D N p.1 p.2 r).map dec) (p.1, (p.2 ⟨r, hr⟩).1))
        ∧ NoCollD settings P.D Ls NS NP (histCodes settings O P.D N p.1 p.2 r) p.2 r hr} ≤ δc :=
    fun r hr => loop_cluster_round_le_persist settings O P Dsf Ls NS NP N εcov δc slack cap
      hδc.le hC hN hNS hNP (fun pops hpops D hD hDlen hDcol => by
        rw [show settings.schedule pops = schedule₀ pops from if_pos hpops]
        exact hclust₀ pops hpops D Dsf hD hDsf (fun j hj => by
          refine measure_mono_null (fun p hp => ?_) (hDlen j hj)
          simp only [Set.mem_compl_iff, Set.mem_ofPred_eq, not_lt] at hp ⊢
          exact hp)
          (by
            refine measure_mono_null (fun v hv => ?_) hDsfLen
            simp only [Set.mem_compl_iff, Set.mem_ofPred_eq] at hv ⊢
            exact hv) hpAPDsf hDcol hcolSf) hLs' hcolD r hr
  have h2 : ∀ r (hr : r < K), μL.real {p | HistOK settings O P.D N p.1 p.2 r
        ∧ (histCodes settings O P.D N p.1 p.2 r).length = r
        ∧ Returns settings O ((histCodes settings O P.D N p.1 p.2 r).map dec) (p.1, (p.2 ⟨r, hr⟩).1)
        ∧ ¬ Level5Spec M O P θ εd (y / 2) k
            (loopRound settings O P.D ((histCodes settings O P.D N p.1 p.2 r).map dec)
              (p.1, p.2 ⟨r, hr⟩))
        ∧ NoCollD settings P.D Ls NS NP (histCodes settings O P.D N p.1 p.2 r) p.2 r hr} ≤ δ :=
    fun r hr => loop_level5_round_le_persist M O hL P Dsf settings S Ls NS NP N εd δ cross θ y m
      hmax hN hNS hNP hband hk0 hrec hdis hη1 hδ hδ1 hy0 hcr hθ0 hθ' hε' le_rfl
      (by linarith) (by linarith) (by rw [hyρ]; linarith)
      (fun P' hP' O' θ' => hl4 (μ.prod μ) P' hP' O' θ') r hr
  have h3 : ∀ r (hr : r < K), μL.real {p | HistOK settings O P.D N p.1 p.2 r
        ∧ (histCodes settings O P.D N p.1 p.2 r).length = r
        ∧ (∀ ν ∈ (histCodes settings O P.D N p.1 p.2 r).map dec,
          ν {q | Ls ≤ q.toList.length} = 0 ∧ ∀ t, ν.real {t} ≤ cap)
        ∧ ¬ NoCollD settings P.D Ls NS NP (histCodes settings O P.D N p.1 p.2 r) p.2 r hr} ≤ δc := by
    intro r hr
    refine (loop_coll_le settings O P Dsf Ls NS NP N cap cap hC hN hLs' hDsfLen hatom hDat r hr
      ).trans ?_
    have hrK : (r : ℝ) ≤ K := by exact_mod_cast hr.le
    have : (r : ℝ) * NS ^ 2 * cap + 16 * r * (K + 1) ^ 2 * NP ^ 2 * cap
        = r * (NS ^ 2 + 16 * (K + 1) ^ 2 * NP ^ 2) * cap := by ring
    rw [this]
    refine le_trans ?_ hcapC
    exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hrK (by positivity)) hcap.le
  -- the rounds' bad events
  set E1 : (r : ℕ) → r < K → Set (Ω × (Fin K → RoundDraws α K T)) := fun r hr =>
    {p | HistOK settings O P.D N p.1 p.2 r
        ∧ (histCodes settings O P.D N p.1 p.2 r).length = r
        ∧ (∀ ν' ∈ (histCodes settings O P.D N p.1 p.2 r).map dec,
          collisionMass ν' ≤ cap ∧ ν' {q | Ls ≤ q.toList.length} = 0)
        ∧ ¬ ClusterSpec settings O P.D εcov slack ((histCodes settings O P.D N p.1 p.2 r).map dec)
          (clusterRun ((histCodes settings O P.D N p.1 p.2 r).map dec) (p.1, (p.2 ⟨r, hr⟩).1))
        ∧ NoCollD settings P.D Ls NS NP (histCodes settings O P.D N p.1 p.2 r) p.2 r hr}
  set E2 : (r : ℕ) → r < K → Set (Ω × (Fin K → RoundDraws α K T)) := fun r hr =>
    {p | HistOK settings O P.D N p.1 p.2 r
        ∧ (histCodes settings O P.D N p.1 p.2 r).length = r
        ∧ Returns settings O ((histCodes settings O P.D N p.1 p.2 r).map dec) (p.1, (p.2 ⟨r, hr⟩).1)
        ∧ ¬ Level5Spec M O P θ εd (y / 2) k
            (loopRound settings O P.D ((histCodes settings O P.D N p.1 p.2 r).map dec)
              (p.1, p.2 ⟨r, hr⟩))
        ∧ NoCollD settings P.D Ls NS NP (histCodes settings O P.D N p.1 p.2 r) p.2 r hr}
  set E3 : (r : ℕ) → r < K → Set (Ω × (Fin K → RoundDraws α K T)) := fun r hr =>
    {p | HistOK settings O P.D N p.1 p.2 r
        ∧ (histCodes settings O P.D N p.1 p.2 r).length = r
        ∧ (∀ ν ∈ (histCodes settings O P.D N p.1 p.2 r).map dec,
          ν {q | Ls ≤ q.toList.length} = 0 ∧ ∀ t, ν.real {t} ≤ cap)
        ∧ ¬ NoCollD settings P.D Ls NS NP (histCodes settings O P.D N p.1 p.2 r) p.2 r hr}
  set Ev : ℕ → Set (Ω × (Fin K → RoundDraws α K T)) := fun r =>
    if hr : r < K then (E1 r hr ∪ E2 r hr) ∪ E3 r hr else ∅ with hEvd
  have hEv : ∀ r, μL.real (Ev r) ≤ 2 * δc + δ := by
    intro r
    simp only [hEvd]
    split_ifs with hr
    · calc μL.real ((E1 r hr ∪ E2 r hr) ∪ E3 r hr)
          ≤ μL.real (E1 r hr) + μL.real (E2 r hr) + μL.real (E3 r hr) :=
            (measureReal_union_le _ _).trans (add_le_add (measureReal_union_le _ _) le_rfl)
        _ ≤ δc + δ + δc := add_le_add (add_le_add (h1 r hr) (h2 r hr)) (h3 r hr)
        _ = 2 * δc + δ := by ring
    · rw [measureReal_empty]; positivity
  -- a failing run fails one of the first `|Q| + 1` rounds' claims
  have hsub : {p : Ω × (Fin K → RoundDraws α K T) |
      let rounds := indecisiveLStarLoop settings O P.D (persistDraws p)
      ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
          o.Agrees O P.D k (εcov + 2 * ι + 2 * slack) (2 * ι + slack) εd)}
      ⊆ ⋃ r ∈ Finset.range (Fintype.card σ + 1), Ev r := by
    intro p hp
    by_contra hno
    simp only [Set.mem_iUnion, Finset.mem_range, not_exists] at hno
    set Lr := loopRounds settings O P.D (persistDraws p) K with hLr
    have hspec : ∀ i, i ≤ Fintype.card σ → ∀ hi : i < Lr.length,
        ClusterSpec settings O P.D εcov slack (harvestsOf P.D k (Lr.take i)) Lr[i].run
          ∧ Level5Spec M O P θ εd (y / 2) k Lr[i] := by
      intro i
      induction i using Nat.strong_induction_on with
      | _ i ih =>
        intro hiσ hi
        obtain ⟨hK', hlen, hdec, e⟩ := persist_bridge settings O P.D P.length_le hC hN p i hi
        have hnot : p ∉ Ev i := hno i (by omega)
        simp only [hEvd, dif_pos hK', Set.mem_union, not_or] at hnot
        obtain ⟨⟨hn1, hn2⟩, hn3⟩ := hnot
        have hok : HistOK settings O P.D N p.1 p.2 i :=
          histOK_of_specs settings O P.D P.length_le hC hN εcov slack p i hi.le
            fun j hj hji => (ih j hji (by omega) hj).1
        have hpops : ∀ ν ∈ (histCodes settings O P.D N p.1 p.2 i).map dec,
            collisionMass ν ≤ cap ∧ ν {q | Ls ≤ q.toList.length} = 0 ∧ ∀ t, ν.real {t} ≤ cap := by
          rw [hdec]
          intro ν hν
          obtain ⟨o, ho, hoν⟩ := List.mem_filterMap.1 hν
          obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem ho
          simp only [List.length_take] at hj
          rw [List.getElem_take] at hoν
          have hji : j < i := by omega
          have h4 := (ih j hji (by omega) (by omega)).2
          obtain ⟨T', xs, hxs⟩ := loopRounds_result settings O P.D (persistDraws p) K _
            (List.getElem_mem (show j < Lr.length by omega))
          have hsupp := population_short P settings.level5 hC (Ls := Ls) (by rw [hmax]; omega) k _ xs hxs hoν
          unfold LoopRound.population at hoν
          rcases hh : (Lr[j]'(by omega)).result.ending.harvested with _ | h
          · rw [hh] at hoν; cases hoν
          rw [hh, Option.map_some, Option.some.injEq] at hoν
          subst hoν
          have he : (Lr[j]'(by omega)).result.ending = .harvest h := by
            revert hh
            cases (Lr[j]'(by omega)).result.ending <;> simp [RoundEnd.harvested]
          simp only [Level5Spec] at h4
          rw [he] at h4
          obtain ⟨-, hsp, hyl⟩ := h4
          have hy2 : y / 2 = ρ / 4 := by rw [hyρ]; ring
          have hatc : ∀ t, (popLaw (harvestLaw P.D (fun z => ((Lr[j]'(by omega)).read z).cut) k
              h)).real {t} ≤ cap := fun t =>
            (popLaw_atom_le (by positivity : (0 : ℝ) < y / 2) hyl
              (harvestLaw_univ_ne_top P.D _ k P.length_le h) (le_prefixMax_of_spread hsp) t).trans
              (by rw [div_le_iff₀ (by positivity), hy2]; linarith)
          exact ⟨collisionMass_le_of_atoms hatc, hsupp, hatc⟩
        have hnc : NoCollD settings P.D Ls NS NP (histCodes settings O P.D N p.1 p.2 i) p.2 i hK' := by
          by_contra h
          exact hn3 ⟨hok, hlen, fun ν hν => ⟨(hpops ν hν).2.1, (hpops ν hν).2.2⟩, h⟩
        have hcs : ClusterSpec settings O P.D εcov slack
            ((histCodes settings O P.D N p.1 p.2 i).map dec)
            (clusterRun ((histCodes settings O P.D N p.1 p.2 i).map dec) (p.1, (p.2 ⟨i, hK'⟩).1)) := by
          by_contra h
          exact hn1 ⟨hok, hlen, fun ν hν => ⟨(hpops ν hν).1, (hpops ν hν).2.1⟩, h, hnc⟩
        have h4 : Level5Spec M O P θ εd (y / 2) k
            (loopRound settings O P.D ((histCodes settings O P.D N p.1 p.2 i).map dec)
              (p.1, p.2 ⟨i, hK'⟩)) := by
          by_contra h
          exact hn2 ⟨hok, hlen, hcs.1, h, hnc⟩
        rw [← hdec, e]
        exact ⟨hcs, h4⟩
    exact hp (loop_agrees_of_specs M O hL P settings (persistDraws p) k rfl εcov slack εd (y / 2)
      hK (fun pops B hB => (hband pops B hB).1) fun i hi hiσ => hspec i hiσ hi)
  calc μL.real _ ≤ μL.real (⋃ r ∈ Finset.range (Fintype.card σ + 1), Ev r) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ r ∈ Finset.range (Fintype.card σ + 1), μL.real (Ev r) := measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _r ∈ Finset.range (Fintype.card σ + 1), (2 * δc + δ) :=
        Finset.sum_le_sum fun r _ => hEv r
    _ = (Fintype.card σ + 1) * (2 * δc + δ) := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        push_cast
        ring



theorem learner_loop_correct_holds : LearnerLoopCorrect := by
  intro α _ _ _ σ _ Ω _ μ _ M O P K S Ls η₀ ι gate εcov δc slack pAP εd δ n A ρ Kp X G
    hL hη hη₀ hι hι1 hg hg1 hε hε1 hδc hδc1 hsl hpAP hεd hεd1 hδ hδ1 hL1 hK hLs hθ
  classical
  have hρ0 : 0 < ρ := by positivity
  set c : ℝ := Real.exp (-(8 * (P.L + 1) * (n + 1) / ρ)) with hc
  have hc0 : 0 < c := Real.exp_pos _
  -- one cap within every set of populations' own
  set capAt : Finset (Option (Fin K)) → ℝ := fun pops =>
    if none ∈ pops then qualityCap η₀ pops ι εcov gate δc pAP c slack else 1 with hcapAt
  have hcapAt0 : ∀ pops, 0 < capAt pops := by
    intro pops
    simp only [hcapAt]
    split_ifs with h
    · exact (clustering_quality_bad O pops none {p : FreeMonoid α | p.toList.length < Ls}
        {v : FreeMonoid α | v.toList.length = Ls} η₀ ι εcov gate δc pAP c slack hη hη₀ h
        (flat_of_length Ls) hpAP hι hι1 hg hg1 hε hε1 hδc hδc1 hc0 hsl).1
    · norm_num
  set capQ : ℝ := (Finset.univ : Finset (Finset (Option (Fin K)))).inf' ⟨∅, Finset.mem_univ _⟩ capAt
  have hcapQ0 : 0 < capQ := (Finset.lt_inf'_iff _).2 fun pops _ => hcapAt0 pops
  have hcapQq : ∀ pops : Finset (Option (Fin K)), none ∈ pops →
      capQ ≤ qualityCap η₀ pops ι εcov gate δc pAP c slack := by
    intro pops h
    have := Finset.inf'_le capAt (Finset.mem_univ pops)
    simp only [hcapAt, if_pos h] at this
    exact this
  -- the schedule's suffix pool and prefix count, over every set of populations
  set NS : ℕ := (Finset.univ : Finset (Finset (Option (Fin K)))).sup fun pops =>
    poolCount η₀ pops ι εcov (δc / 2) pAP c
  set NP : ℕ := (Finset.univ : Finset (Finset (Option (Fin K)))).sup fun pops =>
    prefCount η₀ pops ι εcov (δc / 2) gate pAP c
  set Q : ℝ := (NS : ℝ) ^ 2 + 16 * (K + 1) ^ 2 * NP ^ 2
  set cap : ℝ := min capQ (min (1 / (3 * (K + 1) * (NP + 1) * (P.L + 1)))
    (δc / ((K + 1) * (Q + 1))))
  have hQ0 : 0 ≤ Q := by positivity
  have hcap0 : 0 < cap := lt_min hcapQ0 (lt_min (by positivity) (by positivity))
  have hcapP : 3 * (K + 1) * NP * (P.L + 1) * cap ≤ 1 := by
    have h1 : cap ≤ 1 / (3 * (K + 1) * (NP + 1) * (P.L + 1)) :=
      (min_le_right _ _).trans (min_le_left _ _)
    have hpos : (0 : ℝ) < 3 * (K + 1) * (NP + 1) * (P.L + 1) := by positivity
    calc 3 * (K + 1) * (NP : ℝ) * (P.L + 1) * cap
        ≤ 3 * (K + 1) * (NP + 1) * (P.L + 1) * cap :=
          mul_le_mul_of_nonneg_right (by gcongr; linarith) hcap0.le
      _ ≤ 3 * (K + 1) * (NP + 1) * (P.L + 1) * (1 / (3 * (K + 1) * (NP + 1) * (P.L + 1))) :=
          mul_le_mul_of_nonneg_left h1 hpos.le
      _ = 1 := by field_simp
  have hcapC : K * Q * cap ≤ δc := by
    have h1 : cap ≤ δc / ((K + 1) * (Q + 1)) := (min_le_right _ _).trans (min_le_right _ _)
    have hpos : (0 : ℝ) < (K + 1) * (Q + 1) := by positivity
    calc (K : ℝ) * Q * cap ≤ (K + 1) * (Q + 1) * cap :=
          mul_le_mul_of_nonneg_right (by gcongr <;> linarith) hcap0.le
      _ ≤ (K + 1) * (Q + 1) * (δc / ((K + 1) * (Q + 1))) := mul_le_mul_of_nonneg_left h1 hpos.le
      _ = δc := by field_simp
  refine ⟨cap, hcap0, fun k hk0 hk1 hk2 => ?_⟩
  exact loop_core_holds M O P K S Ls NS NP η₀ ι gate εcov δc slack pAP εd δ cap k hL hη hη₀ hι
    hι1 hg hg1 hε hε1 hδc hδc1 hsl hpAP hεd hεd1 hδ hδ1 hL1 hK hLs hθ hcap0
    (fun pops h => (min_le_left _ _).trans (hcapQq pops h))
    (fun pops _ => Finset.le_sup (f := fun pops => poolCount η₀ pops ι εcov (δc / 2) pAP c)
      (Finset.mem_univ pops))
    (fun pops _ => Finset.le_sup (f := fun pops => prefCount η₀ pops ι εcov (δc / 2) gate pAP c)
      (Finset.mem_univ pops))
    hcapP hcapC hk0 hk1 hk2

end OrthoDFA
