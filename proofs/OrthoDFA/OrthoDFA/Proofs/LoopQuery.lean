import OrthoDFA.Proofs.QueryLevel3
import OrthoDFA.Proofs.ClusterCongr
import OrthoDFA.Proofs.LoopGood

/-!
# The loop program

`loopQ`'s run against the answers at one noise is `loopRounds` at that noise in every round, each
round with its family read dropped (`indecisiveLStarLoop_eq`), and each of those rounds reads its
family at that noise (`loopRounds_read`). Against any answers, with probes at most `L` long, `T`
the probe budget, `M` the scan cap, `N` the harvest cap, and each round's clustering reading at
most `a` prefixes and `b` suffixes, a round asks at most

    K b M N (3 + ⌈log₂ L⌉)(T + 1) + a (b + 1) + b T (probeSifts L (T + 1) + (T + 2)|Σ|(T + 2))

strings (`card_reads_roundQ_le`), and the loop at most `K` times that (`card_reads_loopQ_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Query

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} {K : ℕ}

/-! ## The program's rounds as the functions' -/

section Bridge

variable (O : Oracle μ (FreeMonoid α)) (ω : Ω)

/-- A sampler read at the noise `ω`. -/
def HarvestSampler.spec (s : HarvestSampler α) : SpecSampler α :=
  ⟨fun z => familyRead O.mq s.family s.kl s.kh z ω, s.hyp, s.site⟩

/-- A history's samplers read at the noise `ω`. -/
def LoopHist.spec (h : LoopHist α) : SpecHist α :=
  ⟨h.pops.map (HarvestSampler.spec O ω), h.prefs, h.suffs⟩

/-- A round without its family read. -/
def SpecRound.data (o : SpecRound α) : LoopRound α :=
  ⟨o.state, o.family, o.result, o.prefs, o.suffs⟩

/-- The round reads its family at the noise `ω`. -/
def SpecRound.ReadsAt (o : SpecRound α) : Prop :=
  o.read = o.data.read O ω

lemma sampler_spec (o : SpecRound α) (ho : o.ReadsAt O ω) :
    o.sampler = o.data.sampler.map (HarvestSampler.spec O ω) := by
  simp only [SpecRound.sampler, LoopRound.sampler, SpecRound.data]
  rcases o.result.ending.harvested with _ | site
  · rfl
  · rw [ho]
    rfl

lemma specHistOf_eq {rs : List (SpecRound α)} (h : ∀ o ∈ rs, o.ReadsAt O ω) :
    specHistOf rs = (histOf (rs.map SpecRound.data)).spec O ω := by
  have hp : rs.filterMap (fun o => o.sampler)
      = rs.filterMap fun o => o.data.sampler.map (HarvestSampler.spec O ω) :=
    List.filterMap_congr fun o ho => sampler_spec O ω o (h o ho)
  simp only [specHistOf, histOf, LoopHist.spec, hp, ← List.map_filterMap, List.filterMap_map,
    List.foldr_map]
  rfl

lemma eval_popDrawsQ (D : Measure (FreeMonoid α)) (k N M : ℕ) (pops : List (HarvestSampler α))
    (T : Finset (FreeMonoid α)) (j : Option (Fin K))
    (u : ℕ → unitInterval × (Fin N → unitInterval)) :
    (popDrawsQ D k N M pops T j u).eval (fun w => O.mq w ω)
      = popDraws D k N M (pops.map (HarvestSampler.spec O ω)) T j u := by
  rcases j with _ | i
  · rfl
  simp only [popDrawsQ, popDraws, List.getElem?_map]
  rcases pops[i.val]? with _ | s
  · rfl
  simp only [Option.map_some, eval_subst]
  rw [← eval_drawManyFirstQ]
  congr 1
  funext z
  exact eval_familyReadQ_mq O.mq ω _ _ _ z

/-- A draw's noise folded away. -/
def Draws.flat {S J : Type*} (dr : Draws Ω S J) : Draws Unit S J := (((), dr.1.2), dr.2)

lemma eval_clusterDrawsQ (D Dsf : Measure (FreeMonoid α)) (k N M : ℕ) (h : LoopHist α)
    (c : ClusterRest α K N) :
    (clusterDrawsQ D Dsf k N M h c).eval (fun w => O.mq w ω)
      = (clusterDraws D Dsf k N M (h.spec O ω) (ω, c)).map Draws.flat := by
  unfold clusterDrawsQ clusterDraws
  rw [eval_map, eval_pi]
  simp only [eval_popDrawsQ, Option.map_map]
  rfl

lemma clusterDraws_noise {D Dsf : Measure (FreeMonoid α)} {k N M : ℕ} {h : SpecHist α}
    {c : ClusterDraws Ω α K N} {dr : Draws Ω (FreeMonoid α) (Option (Fin K))}
    (hdr : clusterDraws D Dsf k N M h c = some dr) : dr.1.1 = c.1 := by
  unfold clusterDraws at hdr
  obtain ⟨tab, -, rfl⟩ := Option.map_eq_some_iff.1 hdr
  rfl

open scoped Classical in
lemma eval_roundQ (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (d : RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget) :
    (roundQ settings D Dsf h d).eval (fun w => O.mq w ω)
      = (loopRound settings O D Dsf (h.spec O ω) (ω, d)).map SpecRound.data
      ∧ ∀ o ∈ loopRound settings O D Dsf (h.spec O ω) (ω, d), o.ReadsAt O ω := by
  have hlen : (h.spec O ω).pops.length = h.pops.length := by simp [LoopHist.spec]
  unfold roundQ loopRound
  simp only [hlen]
  rw [eval_bind, eval_clusterDrawsQ]
  rcases hdr : clusterDraws D Dsf settings.level3.walkStart settings.level3.harvestReadCap
    settings.scanCap (h.spec O ω) (ω, d.1) with _ | dr
  · simp
  have hω : dr.1.1 = ω := clusterDraws_noise hdr
  simp only [Option.map_some, Option.bind_some]
  set ps := loopPopulations K h.pops.length
  set sched := settings.schedule ps
  set xs := streams (fun _ => ∅) none dr with hxs
  set xc := streams (fun _ => ∅) none dr.flat with hxc
  have hx : xc = ((), xs.2) := rfl
  have hxs1 : xs.1 = ω := hω
  have hscan : ScanWithin sched ps settings.scanCap dr.flat ↔ ScanWithin sched ps settings.scanCap dr :=
    Iff.rfl
  by_cases hsc : ScanWithin sched ps settings.scanCap dr
  swap
  · simp only [hscan, hsc, false_and, ↓reduceIte]
    simp
  simp only [hscan, hsc, true_and, ↓reduceIte, eval_bind]
  set t := (Query.table (clusterReads sched ps xc).toList).eval (fun w => O.mq w ω)
  have ht : ∀ w ∈ clusterReads sched ps xc, t w = O.mq w ω := fun w hw =>
    eval_table_of_mem _ (Finset.mem_toList.2 hw)
  have hret : ∀ B ∈ sched, (xc ∈ ret (fun w _ => t w) ps none settings.η₀
      settings.indecisionLimit settings.εcov settings.gateLevel B
      ↔ xs ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit settings.εcov
        settings.gateLevel B) := by
    intro B hB
    rw [mem_ret_flat O.mq _ _ _ _ _ _ B xs, hxs1, hx]
    refine (ret_eqOn _ _ sched ps none _ _ _ _ hB _ fun w hw => ?_).symm
    rw [ht w (by rwa [hx])]
  have hexc : (∃ B ∈ sched, xc ∈ ret (fun w _ => t w) ps none settings.η₀
      settings.indecisionLimit settings.εcov settings.gateLevel B)
      ↔ ∃ B ∈ sched, xs ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit
        settings.εcov settings.gateLevel B :=
    exists_congr fun B => ⟨fun h => ⟨h.1, (hret B h.1).1 h.2⟩, fun h => ⟨h.1, (hret B h.1).2 h.2⟩⟩
  have hpred : (fun B => B ∈ sched ∧ xc ∈ ret (fun w _ => t w) ps none settings.η₀
      settings.indecisionLimit settings.εcov settings.gateLevel B)
      = fun B => B ∈ sched ∧ xs ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit
        settings.εcov settings.gateLevel B :=
    funext fun B => propext ⟨fun h => ⟨h.1, (hret B h.1).1 h.2⟩,
      fun h => ⟨h.1, (hret B h.1).2 h.2⟩⟩
  by_cases hex : ∃ B ∈ sched, xs ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit
      settings.εcov settings.gateLevel B
  swap
  · simp only [hexc, hex, ↓reduceIte]
    simp
  simp only [hexc, hex, ↓reduceIte]
  rw [hpred]
  set B := Classical.epsilon fun B => B ∈ sched ∧ xs ∈ ret O.mq ps none settings.η₀
    settings.indecisionLimit settings.εcov settings.gateLevel B with hBdef
  have hBs : B ∈ sched := (Classical.epsilon_spec (p := fun B => B ∈ sched ∧ xs ∈ ret O.mq ps
    none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B) hex).1
  have hF : clusterAt (fun w _ => t w) ps xc B = clusterAt O.mq ps xs B := by
    rw [clusterAt_flat O.mq ps B xs, hxs1, ← hx]
    refine clusterAt_eqOn_sched _ _ sched ps hBs xc fun w hw => ?_
    rw [ht w hw]
  have hread : (fun q => (familyReadQ (clusterAt O.mq ps xs B) B.lo (B.hi + 1) q).eval
      (fun w => O.mq w ω)) = fun z => familyRead O.mq (clusterAt O.mq ps xs B) B.lo (B.hi + 1) z
        xs.1 := by
    funext z
    rw [hxs1]
    exact eval_familyReadQ_mq O.mq ω _ _ _ z
  rw [hF, eval_map, eval_subst, hread, eval_level3Q]
  refine ⟨?_, fun o ho => ?_⟩
  · simp only [Option.map_some, roundOf, SpecRound.data, hlen]
    rfl
  rw [Option.mem_some_iff] at ho
  subst ho
  simp only [SpecRound.ReadsAt, roundOf, SpecRound.data, hxs1, hlen]
  rfl

/-- The program's first `r` rounds are the functions' at the noise `ω`, and each reads its family
there. -/
theorem eval_loopRoundsQ (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (rd : Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget) :
    ∀ r, (loopRoundsQ settings D Dsf rd r).eval (fun w => O.mq w ω)
        = (loopRounds settings O D Dsf (persistDraws (ω, rd)) r).map (List.map SpecRound.data)
      ∧ ∀ rs ∈ loopRounds settings O D Dsf (persistDraws (ω, rd)) r, ∀ o ∈ rs, o.ReadsAt O ω
  | 0 => ⟨rfl, by simp [loopRounds]⟩
  | r + 1 => by
    obtain ⟨ih1, ih2⟩ := eval_loopRoundsQ settings D Dsf rd r
    rw [loopRoundsQ, loopRounds, eval_bind, ih1]
    rcases hrs : loopRounds settings O D Dsf (persistDraws (ω, rd)) r with _ | rs
    · simp
    have hR := ih2 rs hrs
    simp only [Option.map_some, Option.bind_some]
    have hcond : (∀ o ∈ rs.map SpecRound.data, o.result.ending.harvested ≠ none)
        ↔ ∀ o ∈ rs, o.result.ending.harvested ≠ none := by
      simp [SpecRound.data]
    by_cases hc : r < K ∧ ∀ o ∈ rs, o.result.ending.harvested ≠ none
    · rw [dif_pos ⟨hc.1, hcond.2 hc.2⟩, dif_pos hc, eval_map]
      obtain ⟨e1, e2⟩ := eval_roundQ O ω settings D Dsf (histOf (rs.map SpecRound.data))
        (rd ⟨r, hc.1⟩)
      rw [← specHistOf_eq O ω hR] at e1 e2
      have hp : persistDraws (ω, rd) ⟨r, hc.1⟩ = (ω, rd ⟨r, hc.1⟩) := rfl
      rw [hp, e1]
      refine ⟨?_, fun rs' hrs' o ho => ?_⟩
      · rcases loopRound settings O D Dsf (specHistOf rs) (ω, rd ⟨r, hc.1⟩) with _ | o <;> simp
      · obtain ⟨o', ho', rfl⟩ := Option.mem_map.1 hrs'
        rcases List.mem_append.1 ho with ho | ho
        · exact hR o ho
        · rw [List.mem_singleton] at ho
          subst ho
          exact e2 _ ho'
    · rw [dif_neg (fun h => hc ⟨h.1, hcond.1 h.2⟩), dif_neg hc]
      refine ⟨rfl, fun rs' hrs' => ?_⟩
      rw [Option.mem_some_iff] at hrs'
      subst hrs'
      exact hR

/-- The loop's rounds at `p` are the functions' at `p.1` in every round. -/
theorem indecisiveLStarLoop_eq (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (p : Ω × (Fin K → RoundDraws α K settings.level3.harvestReadCap
      settings.level3.probeBudget)) :
    indecisiveLStarLoop settings O D Dsf p
      = (loopRounds settings O D Dsf (persistDraws p) K).map (List.map SpecRound.data) :=
  (eval_loopRoundsQ O p.1 settings D Dsf p.2 K).1

theorem loopRounds_read (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (p : Ω × (Fin K → RoundDraws α K settings.level3.harvestReadCap
      settings.level3.probeBudget)) :
    ∀ rs ∈ loopRounds settings O D Dsf (persistDraws p) K, ∀ o ∈ rs, o.read = o.data.read O p.1 :=
  (eval_loopRoundsQ O p.1 settings D Dsf p.2 K).2

end Bridge

end OrthoDFA

end
