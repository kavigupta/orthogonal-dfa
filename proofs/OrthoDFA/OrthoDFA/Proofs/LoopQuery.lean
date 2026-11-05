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

/-- The bound on a round's reads. -/
def roundCallBound (K a b T M N L A : ℕ) : ℕ :=
  K * (b * (M * (N * ((3 + Nat.clog 2 L) * (T + 1))))) + a * (b + 1)
    + b * (T * (Random.Kappa.probeSifts L * (T + 1) + (T + 2) * A * (T + 2)))

variable (o : FreeMonoid α → ℝ)

theorem card_reads_popDrawsQ_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N M : ℕ) (pops : List (HarvestSampler α)) (T : Finset (FreeMonoid α)) (j : Option (Fin K))
    (u : ℕ → unitInterval × (Fin N → unitInterval)) {L b P : ℕ}
    (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hpops : ∀ s ∈ pops, (Random.mids s.hyp.tree).card ≤ P + 1 ∧ s.family.card ≤ b) :
    ((popDrawsQ D k N M pops T j u).reads o).card
      ≤ if j = none then 0 else b * (M * (N * ((3 + Nat.clog 2 L) * (P + 1)))) := by
  rcases j with _ | i
  · simp [popDrawsQ]
  simp only [popDrawsQ, reduceCtorEq, if_false]
  rcases hs : pops[i.val]? with _ | s
  · simp
  obtain ⟨hm, hb⟩ := hpops s (List.mem_of_getElem? hs)
  refine (card_reads_subst_le o _ _ (n := b)
    fun q _ => (card_reads_familyReadQ_le _ _ _ q o).trans hb).trans ?_
  rw [Nat.mul_comm _ b]
  refine Nat.mul_le_mul_left _ ((card_reads_drawManyFirstQ_le _ _ _ _ _ _
    fun n t => drawFrom_length_le _ hD _).trans ?_)
  gcongr

theorem card_reads_clusterDrawsQ_le (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N M : ℕ) (h : LoopHist α) (c : ClusterRest α K N) {L b P : ℕ}
    (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hpops : ∀ s ∈ h.pops, (Random.mids s.hyp.tree).card ≤ P + 1 ∧ s.family.card ≤ b) :
    ((clusterDrawsQ D Dsf k N M h c).reads o).card
      ≤ K * (b * (M * (N * ((3 + Nat.clog 2 L) * (P + 1))))) := by
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
    (d : RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget) {L a b : ℕ}
    (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hab : ∀ pops (x : Run Unit (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b)
    (hpops : ∀ s ∈ h.pops, (Random.mids s.hyp.tree).card ≤ settings.level3.probeBudget + 1
      ∧ s.family.card ≤ b) :
    ((roundQ settings D Dsf h d).reads o).card
      ≤ roundCallBound K a b settings.level3.probeBudget settings.scanCap
        settings.level3.harvestReadCap L (Fintype.card α) := by
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
  have hF := (Finset.card_le_card
    (clusterAt_sub_roundSuffs (fun w _ => t w) sched ps hBs x)).trans hsb
  rw [reads_map]
  refine (card_reads_subst_le _ _ _ (n := b) fun q _ =>
    (card_reads_familyReadQ_le _ _ _ q _).trans hF).trans ?_
  rw [Nat.mul_comm _ b]
  exact Nat.mul_le_mul_left _ (card_reads_level3Q_le _ _ _ fun i =>
    drawFrom_length_le _ (ae_popLaw (ae_restrict_of_ae hD) (by simp)) _)

/-- Level 3's hypothesis over `T` probes has at most `T + 1` midfixes. -/
lemma mids_level3_le (read : FreeMonoid α → ARU) (C : Random.Level3Settings) {T : ℕ}
    (xs : Fin T → FreeMonoid α) :
    (Random.mids (Random.indecisiveLStarLevel3 C read xs).hypothesis.tree).card ≤ T + 1 := by
  have := (Random.runProbes_tree read C (List.ofFn xs)).2.1
  have hm := Random.mids_card (Random.runProbes read C Random.initialState (List.ofFn xs)).1.tree
  rw [List.length_ofFn] at this
  show (Random.mids (Random.runProbes read C Random.initialState (List.ofFn xs)).1.tree).card
    ≤ T + 1
  omega

section Small

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
  (draws : Fin K → LoopRoundDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget)

/-- Every round the loop runs has a hypothesis of at most `T + 1` midfixes and a family of at
most `b` members. -/
lemma loopRounds_small {b : ℕ}
    (hb : ∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundSuffs (settings.schedule pops) x).card ≤ b) (n : ℕ) :
    ∀ rs ∈ loopRounds settings O D Dsf draws n, ∀ o ∈ rs,
      (Random.mids o.result.hypothesis.tree).card ≤ settings.level3.probeBudget + 1
        ∧ o.family.card ≤ b := by
  induction n with
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
          obtain ⟨B, hB, hr⟩ := hcond.2
          have hBs := (Classical.epsilon_spec (p := fun B => B ∈ settings.schedule
            (loopPopulations K (specHistOf rs).pops.length) ∧ streams (fun _ => ∅) none dr ∈ ret
              O.mq (loopPopulations K (specHistOf rs).pops.length) none settings.η₀
              settings.indecisionLimit settings.εcov settings.gateLevel B) ⟨B, hB, hr⟩).1
          exact ⟨mids_level3_le _ _ _,
            (Finset.card_le_card (clusterAt_sub_roundSuffs _ _ _ hBs _)).trans (hb _ _)⟩
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
    {L a b : ℕ} (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hab : ∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b) :
    ∀ r, ((loopRoundsQ settings D Dsf rd r).reads fun w => O.mq w ω).card
      ≤ r * roundCallBound K a b settings.level3.probeBudget settings.scanCap
        settings.level3.harvestReadCap L (Fintype.card α)
  | 0 => by simp [loopRoundsQ]
  | r + 1 => by
    have ih := card_reads_loopRoundsQ_le O ω settings D Dsf rd hD hab r
    rw [loopRoundsQ, reads_bind, Nat.succ_mul]
    refine (Finset.card_union_le _ _).trans (Nat.add_le_add ih ?_)
    rw [(eval_loopRoundsQ O ω settings D Dsf rd r).1]
    rcases hrs : loopRounds settings O D Dsf (persistDraws (ω, rd)) r with _ | rs
    · simp
    have hsm := loopRounds_small settings O D Dsf _ (fun pops x => (hab pops x).2) r rs hrs
    simp only [Option.map_some]
    split_ifs with hc
    swap
    · simp
    rw [reads_map]
    refine card_reads_roundQ_le _ settings D Dsf _ _ hD (fun pops x => hab pops (ω, x.2))
      fun s hs => ?_
    obtain ⟨o, ho, hos⟩ := List.mem_filterMap.1 hs
    obtain ⟨o', ho', rfl⟩ := List.mem_map.1 ho
    obtain ⟨site, -, rfl⟩ := Option.map_eq_some_iff.1 hos
    exact hsm o' ho'

theorem card_reads_loopQ_le (O : Oracle μ (FreeMonoid α)) (settings : LoopSettings K)
    (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (p : Ω × (Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget))
    {L a b : ℕ} (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hab : ∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b) :
    ((loopQ settings D Dsf p.2).reads fun w => O.mq w p.1).card
      ≤ K * roundCallBound K a b settings.level3.probeBudget settings.scanCap
        settings.level3.harvestReadCap L (Fintype.card α) :=
  card_reads_loopRoundsQ_le O p.1 settings D Dsf p.2 hD hab K

end Reads

end OrthoDFA

end
