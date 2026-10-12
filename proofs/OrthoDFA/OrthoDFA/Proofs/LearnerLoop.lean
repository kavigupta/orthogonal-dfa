import OrthoDFA.Proofs.LoopTermination
import OrthoDFA.Proofs.Level4Correct
import OrthoDFA.Proofs.ClusteringQuality

/-!
# The loop's theorem

Each round meets the clustering's claim but for chance `δc` and, given that, level 4's but for
chance `2δ`; on the rounds' draws these are the one-round claims, at the populations and the
family that round was handed (`loop_cluster_round_le`, `loop_level4_round_le`). Where every one
of the first `|Q| + 1` rounds meets both, the loop stops in agreement (`loop_agrees_of_specs`)
unless it made a harvest too concentrated for the clustering.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-! ## The one-round claims, uniform in what the earlier rounds handed on -/

/-- One schedule and collision cap serve every set of the loop's populations, and every state of
the schedule reads a family of at most `k - 1` members at a band whose far side `crossLimit`
bounds, whatever the read's two rates. -/
theorem clustering_schedule_exists {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) (K : ℕ) (Pre Suf : Set (FreeMonoid α))
    (η₀ indecisionLimit gateLevel crossLimit εcov δc slack pAP : ℝ)
    (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hflat : Flat Pre Suf) (hpAP : 0 < pAP)
    (hι : 0 < indecisionLimit) (hι1 : indecisionLimit ≤ 1 / 2) (hg : 0 < gateLevel)
    (hg1 : gateLevel < 1 / 2) (hcross : 0 < crossLimit) (hε : 0 < εcov) (hε1 : εcov ≤ 1)
    (hδc : 0 < δc) (hδc1 : δc ≤ 1) (hslack : 0 < slack) :
    ∃ cap : ℝ, 0 < cap ∧ ∃ schedule : Finset (Option (Fin K)) → Finset State,
      (∀ pops, ∀ B ∈ schedule pops, B.lo ≤ B.hi ∧ ∀ N, N + 1 ≤ B.k → ∀ p r ε₂ κ : ℝ,
        0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → 0 ≤ κ →
        BandPasses N B.lo (B.hi + 1) p r crossLimit ε₂ κ)
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
  sorry

/-- Level 4's settings depend only on the sizes, so one choice serves every family's oracle. -/
theorem level4_settings_uniform [Nonempty α] (σ : Type*) [Fintype σ] (M : _root_.DFA α σ)
    (P : PrefixDistribution α) (εd δ : ℝ) (S k : ℕ) :
    let n : ℝ := Fintype.card σ + S
    let A : ℝ := Fintype.card α
    let ρ : ℝ := (S + 1) * εd / (256 * (n + 1) * (Fintype.card σ * A + S + 1))
    let Kp : ℝ := (n + 1) * (2 * (n + 3) * A + 1) * (10 / εd ^ 3 + 270 / εd ^ 2 + 1400 / εd)
      + 8 * (n + 3) * A * (P.L + 1)
    let X : ℝ := 4 * (2 ^ (Fintype.card σ + S + 4) * A + 7 * Kp ^ 2) / δ
    let G : ℝ := Real.log X + 2 * Real.log (2 * Real.log X)
    ∃ settings : Level4Settings, settings.walkStart = k ∧ settings.disagreementRate = εd ∧
      (settings.probeBudget : ℝ) ≤ Kp * G ∧ settings.maxLeaves = Fintype.card σ + S + 3 ∧
      ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
        (O : TriOracle μ M) (θ : ℝ),
        let V := undecidedCount P M O.readDist
        let bad := fun q => ¬ O.Good θ q
        0 < δ → δ ≤ 1 → 0 < εd → εd ≤ 1 → 1 ≤ P.L → 0 ≤ θ →
        384 * (P.L + 1) * (n + 1) * θ ≤ 1 →
        0 < O.κ → 4 * O.κ * (8 * (n + 3) * (P.L + 1) * Real.sqrt (A * ρ / 3) + 1) ≤ ρ →
        0 ≤ O.ε → (P.L + 1) * (n + 1) * O.ε + 3 * unionLog (α := α) σ S δ * prefixMax P.D k ≤ ρ →
        ∫⁻ ω, (Measure.pi fun _ : Fin settings.probeBudget => P.D)
            {draws | let result := indecisiveLStarLevel4 settings (O.read · ω) draws
              ¬ EndsWell M bad P.D settings.disagreementRate
                (searchingProbes (O.read · ω) settings.walkStart result.hypothesis)
                P.firstAnchor P.L (level4Replay (O.read · ω) result.hypothesis)
                ((V - 2 * Real.sqrt (P.prefixBound * V * Real.log (1 / δ)))
                  / (P.L + 1 - P.firstAnchor : ℕ))
                result.ending} ∂μ
          ≤ ENNReal.ofReal (2 * δ) := by
  sorry

/-! ## The one-round claims on the loop's draws -/

section Rounds

variable {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] {K : ℕ}

/-- Round `r` meets the clustering's claim but for chance `δc`, unless an earlier round harvested
a list too concentrated for it. Given the earlier rounds, the round's run is drawn as
`runMeasure` draws it at their harvests. -/
theorem loop_cluster_round_le (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (P : PrefixDistribution α) (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]
    (settings : LoopSettings K) (S Ls : ℕ) (εcov δc slack pAP cap : ℝ)
    (hmax : settings.level4.maxLeaves = Fintype.card σ + S + 3)
    (hLs : P.L + Fintype.card σ + S + 3 ≤ Ls)
    (hDsf : Dsf {v | v.toList.length ≠ Ls} = 0)
    (hpAPDsf : pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L})
    (hcolD : collisionMass P.D ≤ cap) (hcolSf : collisionMass Dsf ≤ cap)
    (hclust : ∀ pops : Finset (Option (Fin K)), none ∈ pops →
      ∀ (D : Option (Fin K) → Measure (FreeMonoid α)),
        (∀ j, IsProbabilityMeasure (D j)) →
        (∀ j ∈ pops, D j {p | Ls ≤ p.toList.length} = 0) →
        (∀ j ∈ pops, collisionMass (D j) ≤ cap) →
        (runMeasure μ D Dsf).real
          {x | ¬ ((∃ B ∈ settings.schedule pops,
                x ∈ ret O.mq pops none settings.indecisionLimit settings.gateLevel B)
            ∧ ∀ B ∈ settings.schedule pops,
              x ∈ ret O.mq pops none settings.indecisionLimit settings.gateLevel B →
              ∫ p, miscutProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p ∂(D none)
                  ≤ εcov + slack
                ∧ ∀ j ∈ pops, ∫ p, undecidedProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p
                  ∂(D j) ≤ 2 * settings.indecisionLimit + slack)} ≤ δc)
    (r : ℕ) :
    (loopMeasure μ P.D Dsf K settings.level4.probeBudget).real
      {draws | ∃ hr : r < (loopRounds settings O draws K).length,
        (∀ o ∈ (loopRounds settings O draws K).take r, ∀ zs, o.result.ending = .harvest zs →
          collisionMass (harvestMeasure zs) ≤ cap)
        ∧ ¬ ClusterSpec settings O P.D εcov slack
          (harvestsOf ((loopRounds settings O draws K).take r))
          ((loopRounds settings O draws K)[r]'hr).run} ≤ δc := by
  sorry

/-- Round `r`, meeting the clustering's claim, meets level 4's but for chance `2δ`. Given the
earlier rounds and the round's clustering, level 4 reads a suffix family drawn afresh. -/
theorem loop_level4_round_le [Nonempty α] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)
    (P : PrefixDistribution α) (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]
    (settings : LoopSettings K) (S k Ls : ℕ) (εcov slack εd δ crossLimit : ℝ)
    (hLs : P.L + Fintype.card σ + S + 3 ≤ Ls)
    (hDsf : Dsf {v | v.toList.length ≠ Ls} = 0)
    (hband : ∀ pops, ∀ B ∈ settings.schedule pops, B.lo ≤ B.hi ∧ ∀ N, N + 1 ≤ B.k →
      ∀ p r ε₂ κ : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → 0 ≤ κ →
        BandPasses N B.lo (B.hi + 1) p r crossLimit ε₂ κ)
    (hwalk : settings.level4.walkStart = k) (hdis : settings.level4.disagreementRate = εd)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hεd : 0 < εd) (hεd1 : εd ≤ 1) (hL1 : 1 ≤ P.L)
    (hcross : 0 < crossLimit)
    (hθ : 384 * (P.L + 1) * (Fintype.card σ + S + 1)
      * loopTheta σ settings.indecisionLimit slack ≤ 1)
    (hθ0 : 0 ≤ loopTheta σ settings.indecisionLimit slack)
    (hε : (P.L + 1) * (Fintype.card σ + S + 1) * crossLimit
      + 3 * unionLog (α := α) σ S δ * prefixMax P.D k
        ≤ (S + 1) * εd / (256 * (Fintype.card σ + S + 1)
          * (Fintype.card σ * Fintype.card α + S + 1)))
    (hlevel4 : ∀ (O' : TriOracle μ M) (θ : ℝ),
      let n : ℝ := Fintype.card σ + S
      let A : ℝ := Fintype.card α
      let ρ : ℝ := (S + 1) * εd / (256 * (n + 1) * (Fintype.card σ * A + S + 1))
      let V := undecidedCount P M O'.readDist
      let bad := fun q => ¬ O'.Good θ q
      0 < δ → δ ≤ 1 → 0 < εd → εd ≤ 1 → 1 ≤ P.L → 0 ≤ θ →
      384 * (P.L + 1) * (n + 1) * θ ≤ 1 →
      0 < O'.κ → 4 * O'.κ * (8 * (n + 3) * (P.L + 1) * Real.sqrt (A * ρ / 3) + 1) ≤ ρ →
      0 ≤ O'.ε → (P.L + 1) * (n + 1) * O'.ε + 3 * unionLog (α := α) σ S δ * prefixMax P.D k
        ≤ ρ →
      ∫⁻ ω, (Measure.pi fun _ : Fin settings.level4.probeBudget => P.D)
          {draws | let result := indecisiveLStarLevel4 settings.level4 (O'.read · ω) draws
            ¬ EndsWell M bad P.D settings.level4.disagreementRate
              (searchingProbes (O'.read · ω) settings.level4.walkStart result.hypothesis)
              P.firstAnchor P.L (level4Replay (O'.read · ω) result.hypothesis)
              ((V - 2 * Real.sqrt (P.prefixBound * V * Real.log (1 / δ)))
                / (P.L + 1 - P.firstAnchor : ℕ))
              result.ending} ∂μ
        ≤ ENNReal.ofReal (2 * δ))
    (r : ℕ) :
    (loopMeasure μ P.D Dsf K settings.level4.probeBudget).real
      {draws | ∃ hr : r < (loopRounds settings O draws K).length,
        ClusterSpec settings O P.D εcov slack
            (harvestsOf ((loopRounds settings O draws K).take r))
            ((loopRounds settings O draws K)[r]'hr).run
          ∧ ¬ Level4Spec M O P (loopTheta σ settings.indecisionLimit slack) δ εd k
            ((loopRounds settings O draws K)[r]'hr)} ≤ 2 * δ := by
  sorry

end Rounds

/-! ## The theorem -/

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

theorem learner_loop_correct_holds : LearnerLoopCorrect := by
  intro α _ _ _ σ _ Ω _ μ _ M O P K S k Ls η₀ ι gate cross εcov δc slack pAP εd δ n A ρ Kp X G
    hL hη hη₀ hι hι1 hg hg1 hcr hε hε1 hδc hδc1 hsl hpAP hεd hεd1 hδ hδ1 hL1 hK hLs hθ hcross
  classical
  obtain ⟨cap, hcap, schedule, hband, hclust⟩ := clustering_schedule_exists O K
    {p : FreeMonoid α | p.toList.length < Ls} {v : FreeMonoid α | v.toList.length = Ls}
    η₀ ι gate cross εcov δc slack pAP hη hη₀ (flat_of_length Ls) hpAP hι hι1 hg hg1 hcr hε hε1
    hδc hδc1 hsl
  obtain ⟨s4, hwalk, hdis, hbud, hmax, hl4⟩ := level4_settings_uniform σ M P εd δ S k
  set settings : LoopSettings K := ⟨schedule, ι, gate, s4⟩ with hset
  refine ⟨cap, hcap, settings, rfl, rfl, hwalk, hdis, hbud, ?_⟩
  intro Dsf hDsf hDsfLen hpAPDsf hcolD hcolSf
  have hθ' : 384 * (P.L + 1) * (Fintype.card σ + S + 1) * loopTheta σ ι slack ≤ 1 := by
    have e : 384 * ((P.L : ℝ) + 1) * (Fintype.card σ + S + 1) * loopTheta σ ι slack
        = 512 * (P.L + 1) * (n + 1) * Fintype.card σ * (2 * ι + slack) := by
      simp only [loopTheta, n]; ring
    rw [e]; exact hθ
  have hθ0 : 0 ≤ loopTheta σ ι slack := by
    unfold loopTheta; positivity
  set E1 : ℕ → Set (Fin K → LoopRoundDraws Ω α K s4.probeBudget) := fun r =>
    {draws | ∃ hr : r < (loopRounds settings O draws K).length,
      (∀ o ∈ (loopRounds settings O draws K).take r, ∀ zs, o.result.ending = .harvest zs →
        collisionMass (harvestMeasure zs) ≤ cap)
      ∧ ¬ ClusterSpec settings O P.D εcov slack
        (harvestsOf ((loopRounds settings O draws K).take r))
        ((loopRounds settings O draws K)[r]'hr).run} with hE1
  set E2 : ℕ → Set (Fin K → LoopRoundDraws Ω α K s4.probeBudget) := fun r =>
    {draws | ∃ hr : r < (loopRounds settings O draws K).length,
      ClusterSpec settings O P.D εcov slack
          (harvestsOf ((loopRounds settings O draws K).take r))
          ((loopRounds settings O draws K)[r]'hr).run
        ∧ ¬ Level4Spec M O P (loopTheta σ ι slack) δ εd k
          ((loopRounds settings O draws K)[r]'hr)} with hE2
  have h1 : ∀ r, (loopMeasure μ P.D Dsf K s4.probeBudget).real (E1 r) ≤ δc := fun r =>
    loop_cluster_round_le M O P Dsf settings S Ls εcov δc slack pAP cap hmax hLs hDsfLen hpAPDsf
      hcolD hcolSf (fun pops hpops D hD hDlen hDcol =>
        hclust pops hpops D Dsf hD hDsf (fun j hj => by
          refine measure_mono_null (fun p hp => ?_) (hDlen j hj)
          simp only [Set.mem_compl_iff, Set.mem_ofPred_eq, not_lt] at hp ⊢
          exact hp)
          (by
            refine measure_mono_null (fun v hv => ?_) hDsfLen
            simp only [Set.mem_compl_iff, Set.mem_ofPred_eq] at hv ⊢
            exact hv) hpAPDsf hDcol hcolSf) r
  have h2 : ∀ r, (loopMeasure μ P.D Dsf K s4.probeBudget).real (E2 r) ≤ 2 * δ := fun r =>
    loop_level4_round_le M O hL P Dsf settings S k Ls εcov slack εd δ cross hLs hDsfLen hband
      hwalk hdis hδ hδ1 hεd hεd1 hL1 hcr hθ' hθ0 hcross
      (fun O' θ => hl4 O' θ) r
  have hsub : {draws : Fin K → LoopRoundDraws Ω α K s4.probeBudget |
      let rounds := indecisiveLStarLoop settings O draws
      ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
            o.Agrees O P.D k (εcov + 2 * ι + 2 * slack) (2 * ι + slack) εd)
        ∨ ∃ o ∈ rounds, ∃ zs, o.result.ending = .harvest zs
            ∧ cap < collisionMass (harvestMeasure zs))}
      ⊆ ⋃ r ∈ Finset.range (Fintype.card σ + 1), (E1 r ∪ E2 r) := by
    intro draws hd
    simp only [Set.mem_ofPred_eq, not_or, not_and, not_exists, not_lt] at hd
    by_contra hno
    simp only [Set.mem_iUnion, Set.mem_union, Finset.mem_range, not_exists, not_or] at hno
    have hspec : ∀ i (hi : i < (loopRounds settings O draws K).length), i ≤ Fintype.card σ →
        ClusterSpec settings O P.D εcov slack
            (harvestsOf ((loopRounds settings O draws K).take i))
            (loopRounds settings O draws K)[i].run
          ∧ Level4Spec M O P (loopTheta σ settings.indecisionLimit slack) δ εd k
            (loopRounds settings O draws K)[i] := by
      intro i hi hiσ
      have hcol : ∀ o ∈ (loopRounds settings O draws K).take i, ∀ zs,
          o.result.ending = .harvest zs → collisionMass (harvestMeasure zs) ≤ cap :=
        fun o ho zs hzs => hd.2 o (List.mem_of_mem_take ho) zs hzs
      have hc : ClusterSpec settings O P.D εcov slack
          (harvestsOf ((loopRounds settings O draws K).take i))
          (loopRounds settings O draws K)[i].run := by
        by_contra hc
        exact (hno i (by omega)).1 ⟨hi, hcol, hc⟩
      refine ⟨hc, ?_⟩
      by_contra h4
      exact (hno i (by omega)).2 ⟨hi, hc, h4⟩
    have := loop_agrees_of_specs M O hL P settings draws k εcov slack εd δ hK
      (fun pops B hB => (hband pops B hB).1) hspec
    obtain ⟨o, ho, hag⟩ := this.2
    exact hd.1 this.1 o ho hag
  have : IsProbabilityMeasure (loopMeasure μ P.D Dsf K s4.probeBudget) := by
    unfold loopMeasure clusterLaw
    infer_instance
  calc (loopMeasure μ P.D Dsf K s4.probeBudget).real _
      ≤ (loopMeasure μ P.D Dsf K s4.probeBudget).real
          (⋃ r ∈ Finset.range (Fintype.card σ + 1), (E1 r ∪ E2 r)) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ r ∈ Finset.range (Fintype.card σ + 1),
          (loopMeasure μ P.D Dsf K s4.probeBudget).real (E1 r ∪ E2 r) :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _r ∈ Finset.range (Fintype.card σ + 1), (δc + 2 * δ) :=
        Finset.sum_le_sum fun r _ => (measureReal_union_le _ _).trans (add_le_add (h1 r) (h2 r))
    _ = (Fintype.card σ + 1) * (δc + 2 * δ) := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        push_cast
        ring

end OrthoDFA

end
