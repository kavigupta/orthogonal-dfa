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

omit [IsProbabilityMeasure μ] in
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

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
  (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]

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
  sorry -- closes by: main's scan half, clusterLaw_scanShort_le, offLaw_atom_le, persistence

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
