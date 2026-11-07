import OrthoDFA.Proofs.QueryLevel3
import OrthoDFA.Proofs.ClusterCongr
import OrthoDFA.Proofs.LoopGood
import OrthoDFA.Code.LoopDFA

/-!
# The loop program

`loopQ`'s run against the answers at one noise is `loopRounds` at that noise in every round, each
round with its family read dropped (`indecisiveLStarLoop_eq`), and each of those rounds reads its
family at that noise (`loopRounds_read`). Against any answers, with probes at most `L` long, `T`
the probe budget, `M` the scan cap, `N` the harvest cap, `leafCap ≤ n + 2`, each round's
clustering reading at most `a` prefixes and `b` suffixes and each family at most `f` members, a
round asks at most

    K f M N (3 + ⌈log₂ L⌉)(n + 1) + a (b + 1) + f (T probeSifts L (n + 1) + (n + 1)(n + 2)²|Σ|)

strings (`card_reads_roundQ_le`): its harvested draws, its clustering and its level 3. The loop
asks at most `K` times that (`card_reads_loopQ_le`).
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
  have hscan :
      ScanWithin sched ps settings.scanCap dr.flat ↔ ScanWithin sched ps settings.scanCap dr :=
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
  have hres : (roundOf settings O D (h.spec O ω) xs d.2).result
      = Random.indecisiveLStarLevel3 settings.level3
        (fun z => familyRead O.mq (clusterAt O.mq ps xs B) B.lo (B.hi + 1) z xs.1)
        (fun i => drawFrom (avoidLaw D settings.level3.walkStart
          (clusterKeys sched ps settings.level3.walkStart xc)) (d.2 i)) := by
    simp only [roundOf, hlen]; rfl
  rw [← hres]
  by_cases hc : (roundOf settings O D (h.spec O ω) xs d.2).result.hypothesis.tree.leaves.length
      ≤ settings.level3.leafCap
  swap
  · simp only [hc, and_false, ↓reduceIte]
    simp
  simp only [hc, and_true, ↓reduceIte]
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

lemma indecisiveLStarLoop_some {settings : LoopSettings K} {D Dsf : Measure (FreeMonoid α)}
    {p : Ω × (Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget)}
    {rounds : List (LoopRound α)} (h : indecisiveLStarLoop settings O D Dsf p = some rounds) :
    ∃ rs, loopRounds settings O D Dsf (persistDraws p) K = some rs
      ∧ rounds = rs.map SpecRound.data := by
  rw [indecisiveLStarLoop_eq] at h
  obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 h
  exact ⟨rs, hrs, rfl⟩

theorem loopRounds_read (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (p : Ω × (Fin K → RoundDraws α K settings.level3.harvestReadCap
      settings.level3.probeBudget)) :
    ∀ rs ∈ loopRounds settings O D Dsf (persistDraws p) K, ∀ o ∈ rs, o.read = o.data.read O p.1 :=
  (eval_loopRoundsQ O p.1 settings D Dsf p.2 K).2

end Bridge

/-! ## What the program asks -/

section Reads

/-- A law that is `ν` scaled, or a point mass at `1`, keeps what `ν` almost surely has and `1`
has. -/
lemma ae_popLaw {ν : Measure (FreeMonoid α)} {P : FreeMonoid α → Prop} (hν : ∀ᵐ x ∂ν, P x)
    (h1 : P 1) : ∀ᵐ x ∂(popLaw ν), P x := by
  unfold popLaw
  split_ifs
  · exact (ae_dirac_iff (MeasurableSet.of_discrete)).2 h1
  · rw [ae_iff, Measure.smul_apply, ae_iff.1 hν, smul_zero]

/-- The bound on a round's reads, its trees at most `n + 2` leaves. -/
def roundCallBound (K a b f T M N L A n : ℕ) : ℕ :=
  K * (f * (M * (N * ((3 + Nat.clog 2 L) * (n + 1))))) + a * (b + 1)
    + f * (T * (Random.Kappa.probeSifts L * (n + 1)) + (n + 1) * ((n + 2) * A * (n + 2)))

/-- The family has at most `k - 1` members. -/
lemma card_clusterAt_le (mq : FreeMonoid α → Ω → ℝ) (pops : Finset (Option (Fin K)))
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) (B : State) :
    (clusterAt mq pops x B).card ≤ B.k - 1 := by
  unfold clusterAt nearestRows
  exact (Finset.card_le_card (Finset.erase_insert_subset _ _)).trans
    (leastLossSubset_card_le _ _ _)

variable (o : FreeMonoid α → ℝ)

theorem card_reads_popDrawsQ_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N M : ℕ) (pops : List (HarvestSampler α)) (T : Finset (FreeMonoid α)) (j : Option (Fin K))
    (u : ℕ → unitInterval × (Fin N → unitInterval)) {L f P : ℕ}
    (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hpops : ∀ s ∈ pops, (Random.mids s.hyp.tree).card ≤ P + 1 ∧ s.family.card ≤ f) :
    ((popDrawsQ D k N M pops T j u).reads o).card
      ≤ if j = none then 0 else f * (M * (N * ((3 + Nat.clog 2 L) * (P + 1)))) := by
  rcases j with _ | i
  · simp [popDrawsQ]
  simp only [popDrawsQ, reduceCtorEq, if_false]
  rcases hs : pops[i.val]? with _ | s
  · simp
  obtain ⟨hm, hb⟩ := hpops s (List.mem_of_getElem? hs)
  refine (card_reads_subst_le o _ _ (n := f)
    fun q _ => (card_reads_familyReadQ_le _ _ _ q o).trans hb).trans ?_
  rw [Nat.mul_comm _ f]
  refine Nat.mul_le_mul_left _ ((card_reads_drawManyFirstQ_le _ _ _ _ _ _
    fun n t => drawFrom_length_le _ hD _).trans ?_)
  gcongr

theorem card_reads_clusterDrawsQ_le (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N M : ℕ) (h : LoopHist α) (c : ClusterRest α K N) {L f P : ℕ}
    (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hpops : ∀ s ∈ h.pops, (Random.mids s.hyp.tree).card ≤ P + 1 ∧ s.family.card ≤ f) :
    ((clusterDrawsQ D Dsf k N M h c).reads o).card
      ≤ K * (f * (M * (N * ((3 + Nat.clog 2 L) * (P + 1))))) := by
  unfold clusterDrawsQ
  rw [reads_map, reads_pi]
  refine (Finset.card_biUnion_le).trans ?_
  rw [Fintype.sum_option]
  have := fun j => card_reads_popDrawsQ_le o D k N M h.pops h.prefs j (c.2.1 j) hD hpops
  refine (Nat.add_le_add (this none) (Finset.sum_le_sum fun i _ => this (some i))).trans ?_
  simp

open scoped Classical in
/-- A round reads at most `roundCallBound` strings, against any answers. -/
theorem card_reads_roundQ_le (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (h : LoopHist α)
    (d : RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget) {L a b f : ℕ}
    (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hab : ∀ pops (x : Run Unit (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b)
    (hk : ∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ f + 1) {n : ℕ}
    (hn : settings.level3.leafCap ≤ n + 2)
    (hpops : ∀ s ∈ h.pops, (Random.mids s.hyp.tree).card ≤ n + 1 ∧ s.family.card ≤ f) :
    ((roundQ settings D Dsf h d).reads o).card
      ≤ roundCallBound K a b f settings.level3.probeBudget settings.scanCap
        settings.level3.harvestReadCap L (Fintype.card α) n := by
  have hA := card_reads_clusterDrawsQ_le o D Dsf settings.level3.walkStart
    settings.level3.harvestReadCap settings.scanCap h d.1 hD hpops
  unfold roundQ roundCallBound
  rw [reads_bind]
  refine (Finset.card_union_le _ _).trans ?_
  rw [Nat.add_assoc]
  refine Nat.add_le_add hA ?_
  rcases (clusterDrawsQ D Dsf settings.level3.walkStart settings.level3.harvestReadCap
    settings.scanCap h d.1).eval o with _ | dr
  · simp
  simp only
  set ps := loopPopulations K h.pops.length
  set sched := settings.schedule ps
  set x := streams (fun _ => ∅) none dr
  obtain ⟨hpa, hsb⟩ := hab ps x
  split_ifs with hsc
  swap
  · simp
  rw [reads_bind]
  refine (Finset.card_union_le _ _).trans (Nat.add_le_add ?_ ?_)
  · rw [reads_table]
    refine (List.toFinset_card_le _).trans ?_
    rw [Finset.length_toList]
    refine Finset.card_image_le.trans ?_
    rw [Finset.card_product]
    exact Nat.mul_le_mul hpa ((Finset.card_insert_le _ _).trans (Nat.add_le_add_right hsb 1))
  split_ifs with hex
  swap
  · simp
  set t := (Query.table (clusterReads sched ps x).toList).eval o
  have hBs := (Classical.epsilon_spec (p := fun B => B ∈ sched ∧ x ∈ ret (fun w _ => t w) ps
    none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B)
    (by obtain ⟨B, hB, hr⟩ := hex; exact ⟨B, hB, hr⟩)).1
  have hF := (card_clusterAt_le (fun w _ => t w) ps x _).trans
    (Nat.sub_le_of_le_add (hk ps _ hBs))
  rw [reads_map]
  refine (card_reads_subst_le _ _ _ (n := f) fun q _ =>
    (card_reads_familyReadQ_le _ _ _ q _).trans hF).trans ?_
  rw [Nat.mul_comm _ f]
  exact Nat.mul_le_mul_left _ (card_reads_level3Q_le _ _ n hn _ fun i =>
    drawFrom_length_le _ (ae_popLaw (ae_restrict_of_ae hD) (by simp)) _)

section Small

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
  (draws : Fin K → LoopRoundDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget)

/-- Every round the loop runs, with `leafCap ≤ n + 2`, has a hypothesis of at most `n + 1`
midfixes and a family of at most `f` members. -/
lemma loopRounds_small {f n : ℕ} (hk : ∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ f + 1)
    (hn : settings.level3.leafCap ≤ n + 2) (r : ℕ) :
    ∀ rs ∈ loopRounds settings O D Dsf draws r, ∀ o ∈ rs,
      (Random.mids o.result.hypothesis.tree).card ≤ n + 1 ∧ o.family.card ≤ f := by
  induction r with
  | zero =>
    intro rounds hr
    rw [loopRounds, Option.mem_def, Option.some.injEq] at hr
    subst hr
    simp
  | succ r ih =>
    intro rounds hr
    rw [loopRounds] at hr
    obtain ⟨rs, hrs, hr⟩ := Option.mem_bind_iff.1 hr
    have hrr := ih rs hrs
    split_ifs at hr with hc
    · obtain ⟨o, ho, rfl⟩ := Option.mem_map.1 hr
      intro o' ho'
      rcases List.mem_append.1 ho' with h1 | h1
      · exact hrr o' h1
      · rw [List.mem_singleton] at h1
        subst h1
        unfold loopRound at ho
        obtain ⟨dr, -, hdr⟩ := Option.mem_bind_iff.1 ho
        split_ifs at hdr with hcond
        · rw [← Option.mem_some_iff.1 hdr]
          obtain ⟨B, hB, hr⟩ := hcond.2.1
          have hBs := (Classical.epsilon_spec (p := fun B => B ∈ settings.schedule
            (loopPopulations K (specHistOf rs).pops.length) ∧ streams (fun _ => ∅) none dr ∈ ret
              O.mq (loopPopulations K (specHistOf rs).pops.length) none settings.η₀
              settings.indecisionLimit settings.εcov settings.gateLevel B) ⟨B, hB, hr⟩).1
          exact ⟨Nat.le_of_succ_le_succ ((Random.mids_card _).trans (hcond.2.2.trans hn)),
            (card_clusterAt_le _ _ _ _).trans (Nat.sub_le_of_le_add (hk _ _ hBs))⟩
        · cases hdr
    · rw [Option.mem_some_iff] at hr
      subst hr
      exact hrr

end Small

/-- The program's first `r` rounds, against the answers at one noise, read at most `r` times
`roundCallBound` strings. -/
theorem card_reads_loopRoundsQ_le (O : Oracle μ (FreeMonoid α)) (ω : Ω)
    (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (rd : Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget)
    {L a b f : ℕ} (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hab : ∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b)
    (hk : ∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ f + 1) {n : ℕ}
    (hn : settings.level3.leafCap ≤ n + 2) :
    ∀ r, ((loopRoundsQ settings D Dsf rd r).reads fun w => O.mq w ω).card
      ≤ r * roundCallBound K a b f settings.level3.probeBudget settings.scanCap
        settings.level3.harvestReadCap L (Fintype.card α) n
  | 0 => by simp [loopRoundsQ]
  | r + 1 => by
    have ih := card_reads_loopRoundsQ_le O ω settings D Dsf rd hD hab hk hn r
    rw [loopRoundsQ, reads_bind, Nat.succ_mul]
    refine (Finset.card_union_le _ _).trans (Nat.add_le_add ih ?_)
    rw [(eval_loopRoundsQ O ω settings D Dsf rd r).1]
    rcases hrs : loopRounds settings O D Dsf (persistDraws (ω, rd)) r with _ | rs
    · simp
    have hsm := loopRounds_small settings O D Dsf _ hk hn r rs hrs
    simp only [Option.map_some]
    split_ifs with hc
    swap
    · simp
    rw [reads_map]
    refine card_reads_roundQ_le _ settings D Dsf _ _ hD (fun pops x => hab pops (ω, x.2)) hk hn
      fun s hs => ?_
    obtain ⟨o, ho, hos⟩ := List.mem_filterMap.1 hs
    obtain ⟨o', ho', rfl⟩ := List.mem_map.1 ho
    obtain ⟨site, -, rfl⟩ := Option.map_eq_some_iff.1 hos
    exact hsm o' ho'

theorem card_reads_loopQ_le (O : Oracle μ (FreeMonoid α)) (settings : LoopSettings K)
    (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (p : Ω × (Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget))
    {L a b f : ℕ} (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hab : ∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b)
    (hk : ∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ f + 1) {n : ℕ}
    (hn : settings.level3.leafCap ≤ n + 2) :
    ((loopQ settings D Dsf p.2).reads fun w => O.mq w p.1).card
      ≤ K * roundCallBound K a b f settings.level3.probeBudget settings.scanCap
        settings.level3.harvestReadCap L (Fintype.card α) n :=
  card_reads_loopRoundsQ_le O p.1 settings D Dsf p.2 hD hab hk hn K

end Reads

/-! ## The learner -/

section Learner

omit [Fintype α] [DecidableEq α] in
/-- The leaf picked depends on the read only at the held-out strings. -/
lemma pickLeaf_congr (H : HypothesisDFA α) {read read' : FreeMonoid α → ARU} {n : ℕ}
    (xs : Fin n → FreeMonoid α) (h : ∀ i, read (xs i) = read' (xs i)) :
    pickLeaf H read xs = pickLeaf H read' xs := by
  have : leafScore H read xs = leafScore H read' xs := by
    funext s
    unfold leafScore
    congr 2
    funext i
    rw [h i]
  simp only [pickLeaf, this]

omit [DecidableEq α] in
lemma eval_pickLeafQ (H : HypothesisDFA α) {n : ℕ} (xs : Fin n → FreeMonoid α)
    (read : FreeMonoid α → ARU) : (pickLeafQ H xs).eval read = pickLeaf H read xs := by
  unfold pickLeafQ
  rw [eval_map]
  exact pickLeaf_congr H xs fun i => eval_table_of_mem _ (List.mem_ofFn.2 ⟨i, rfl⟩)

omit [DecidableEq α] in
lemma card_reads_pickLeafQ_le (H : HypothesisDFA α) {n : ℕ} (xs : Fin n → FreeMonoid α)
    (read : FreeMonoid α → ARU) : ((pickLeafQ H xs).reads read).card ≤ n := by
  unfold pickLeafQ
  rw [reads_map, reads_table]
  exact (List.toFinset_card_le _).trans (List.length_ofFn).le

variable (O : Oracle μ (FreeMonoid α))

/-- The leaf the last of `rounds` starts at, picked on `xs` as its family reads them at `ω`. -/
def startLeaf (ω : Ω) (rounds : List (LoopRound α)) {n : ℕ} (xs : Fin n → FreeMonoid α) :
    List Bool :=
  match rounds.getLast? with
  | some o => pickLeaf o.result.hypothesis (o.read O ω) xs
  | none => []

/-- The learner returns the loop's rounds and the leaf its last round starts at. -/
theorem indecisiveLStar_eq (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (p : LoopDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget) {n : ℕ}
    (xs : Fin n → FreeMonoid α) :
    indecisiveLStar settings O D Dsf p xs
      = (indecisiveLStarLoop settings O D Dsf p).map fun rounds =>
        (rounds, startLeaf O p.1 rounds xs) := by
  unfold indecisiveLStar learnerQ
  rw [eval_bind]
  change _ = (indecisiveLStarLoop settings O D Dsf p).map _
  unfold indecisiveLStarLoop
  rcases (loopQ settings D Dsf p.2).eval (fun w => O.mq w p.1) with _ | rounds
  · rfl
  simp only [Option.map_some, startLeaf]
  rcases rounds.getLast? with _ | o
  · rfl
  simp only
  rw [eval_map, eval_subst]
  congr 2
  rw [show (fun q => (familyReadQ o.family o.state.lo (o.state.hi + 1) q).eval
      (fun w => O.mq w p.1)) = o.read O p.1 from funext fun z =>
    eval_familyReadQ_mq O.mq p.1 _ _ _ z]
  exact eval_pickLeafQ _ _ _

/-- The learner asks the loop's strings and, at the last round's family, `|F|` strings for each
held-out string. -/
theorem card_reads_learnerQ_le (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    [IsProbabilityMeasure D]
    (p : Ω × (Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget))
    {n : ℕ} (xs : Fin n → FreeMonoid α) {L a b f : ℕ} (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hab : ∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b)
    (hk : ∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ f + 1) {q : ℕ}
    (hq : settings.level3.leafCap ≤ q + 2) :
    ((learnerQ settings D Dsf p.2 xs).reads fun w => O.mq w p.1).card
      ≤ K * roundCallBound K a b f settings.level3.probeBudget settings.scanCap
          settings.level3.harvestReadCap L (Fintype.card α) q + n * f := by
  have hloop := card_reads_loopQ_le O settings D Dsf p hD hab hk hq
  unfold learnerQ
  rw [reads_bind]
  refine (Finset.card_union_le _ _).trans (Nat.add_le_add hloop ?_)
  rw [show (loopQ settings D Dsf p.2).eval (fun w => O.mq w p.1) = _ from
    (eval_loopRoundsQ O p.1 settings D Dsf p.2 K).1]
  rcases hrs : loopRounds settings O D Dsf (persistDraws (p.1, p.2)) K with _ | rs
  · simp
  have hsm := loopRounds_small settings O D Dsf _ hk hq K rs hrs
  simp only [Option.map_some, List.getLast?_map]
  rcases hl : rs.getLast? with _ | o
  · simp
  simp only [Option.map_some]
  rw [reads_map]
  have hF := (hsm o (List.mem_of_getLast? hl)).2
  refine (card_reads_subst_le _ _ _ (n := f) fun q _ =>
    (card_reads_familyReadQ_le _ _ _ q _).trans hF).trans ?_
  exact Nat.mul_le_mul_right _ (card_reads_pickLeafQ_le _ _ _)

end Learner

end OrthoDFA

end
