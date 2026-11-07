import OrthoDFA.Proofs.QueryLevel3
import OrthoDFA.Proofs.ClusterCongr
import OrthoDFA.Proofs.LoopGood
import OrthoDFA.Code.LoopDFA
import OrthoDFA.LowLevelClaims.LoopDFA

/-!
# The loop program

`loopQ`'s run against the answers at one noise is `loopRounds` at that noise in every round, each
round with its family read dropped (`indecisiveLStarLoop_eq`), and each of those rounds reads its
family at that noise (`loopRounds_read`). Against any answers, with probes at most `L` long, `T`
the probe budget, each round's clustering reading at most `a` prefixes and `b` suffixes and each
family at most `f` members, a round whose tree and whose samplers' trees have at most `n + 2`
leaves and whose harvested draws find their strings within `W` probes in all asks at most

    f W (3 + ⌈log₂ L⌉)(n + 1) + a (b + 1) + f (T probeSifts L (n + 1) + n (n + 2)²|Σ|)

strings (`card_reads_roundQ_le`): its harvested draws, its clustering and its level 3. A loop
returning rounds whose trees have at most `n + 2` leaves and that meet `HarvestProbesWithin` at
`W` asks at most `K` times that (`card_reads_loopRoundsQ_le`); the loop's rounds meet it where
every round is good and walks at most `W` probes (`harvestProbesWithin_of`).
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
    rfl
  rw [hF, eval_map, eval_subst, hread]
  have hres : (roundOf settings O D (h.spec O ω) xs d.2).result
      = (Random.level3Q settings.level3 (fun i => drawFrom (avoidLaw D settings.level3.walkStart
          (clusterKeys sched ps settings.level3.walkStart xc)) (d.2 i))).eval
        (fun z => familyRead O.mq (clusterAt O.mq ps xs B) B.lo (B.hi + 1) z xs.1) := by
    simp only [roundOf, hlen]; rfl
  rw [← hres]
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

/-- The bound on a round's reads, its harvested draws walking at most `W` probes in all and its
trees at most `n + 2` leaves. -/
def roundCallBound (a b f T W L A n : ℕ) : ℕ :=
  f * (W * ((3 + Nat.clog 2 L) * (n + 1))) + a * (b + 1)
    + f * (T * (Random.Kappa.probeSifts L * (n + 1)) + n * ((n + 2) * A * (n + 2)))

/-- The family has at most `k - 1` members. -/
lemma card_clusterAt_le (mq : FreeMonoid α → Ω → ℝ) (pops : Finset (Option (Fin K)))
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) (B : State) :
    (clusterAt mq pops x B).card ≤ B.k - 1 := by
  unfold clusterAt nearestRows
  exact (Finset.card_le_card (Finset.erase_insert_subset _ _)).trans
    (leastLossSubset_card_le _ _ _)

variable (o : FreeMonoid α → ℝ)

/-- Draw `n` harvesting within its first `t n` probes, a harvested population's draws sift
`∑ₙ t n` probes, each a family read of at most `f` strings at `3 + ⌈log₂ L⌉` reads of each mid. -/
theorem card_reads_popDrawsQ_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N M : ℕ) (pops : List (HarvestSampler α)) (T : Finset (FreeMonoid α)) (i : Fin K)
    (u : ℕ → unitInterval × (Fin N → unitInterval)) {L f P : ℕ}
    (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hpops : ∀ s ∈ pops, (Random.mids s.hyp.tree).card ≤ P + 1 ∧ s.family.card ≤ f)
    (t : Fin M → ℕ)
    (ht : ∀ s, pops[i.val]? = some s → ∀ n : Fin M,
      (Random.harvestFirst (fun z => (familyReadQ s.family s.kl s.kh z).eval o) k s.hyp s.site ↑T
        ((List.ofFn fun j => drawFrom D ((u n).2 j)).take (t n))).isSome) :
    ((popDrawsQ D k N M pops T (some i) u).reads o).card
      ≤ f * ((∑ n, t n) * ((3 + Nat.clog 2 L) * (P + 1))) := by
  simp only [popDrawsQ]
  rcases hs : pops[i.val]? with _ | s
  · simp
  obtain ⟨hm, hb⟩ := hpops s (List.mem_of_getElem? hs)
  refine (card_reads_subst_le o _ _ (n := f)
    fun q _ => (card_reads_familyReadQ_le _ _ _ q o).trans hb).trans ?_
  rw [Nat.mul_comm _ f]
  refine Nat.mul_le_mul_left _ ((card_reads_drawManyFirstQ_take_le _ _ _ _ _ _
    (fun n j => drawFrom_length_le _ hD _) t (ht s hs)).trans ?_)
  gcongr

theorem card_reads_clusterDrawsQ_le (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N M : ℕ) (h : LoopHist α) (c : ClusterRest α K N) {L f P W : ℕ}
    (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hpops : ∀ s ∈ h.pops, (Random.mids s.hyp.tree).card ≤ P + 1 ∧ s.family.card ≤ f)
    (t : Fin K → Fin M → ℕ) (hW : ∑ i, ∑ n, t i n ≤ W)
    (ht : ∀ i s, h.pops[i.val]? = some s → ∀ n : Fin M,
      (Random.harvestFirst (fun z => (familyReadQ s.family s.kl s.kh z).eval o) k s.hyp s.site
        ↑h.prefs ((List.ofFn fun j => drawFrom D ((c.2.1 (some i) n).2 j)).take (t i n))).isSome) :
    ((clusterDrawsQ D Dsf k N M h c).reads o).card
      ≤ f * (W * ((3 + Nat.clog 2 L) * (P + 1))) := by
  unfold clusterDrawsQ
  rw [reads_map, reads_pi]
  refine (Finset.card_biUnion_le).trans ?_
  rw [Fintype.sum_option]
  have h0 : ((popDrawsQ (K := K) D k N M h.pops h.prefs none (c.2.1 none)).reads o).card = 0 := by
    simp [popDrawsQ]
  rw [h0, zero_add]
  refine (Finset.sum_le_sum fun i _ => card_reads_popDrawsQ_le o D k N M h.pops h.prefs i
    (c.2.1 (some i)) hD hpops (t i) (ht i)).trans ?_
  rw [← Finset.mul_sum, ← Finset.sum_mul]
  exact Nat.mul_le_mul_left _ (Nat.mul_le_mul_right _ hW)

open scoped Classical in
/-- A round whose tree, against the answers `o`, has at most `n + 2` leaves reads at most
`roundCallBound` strings. -/
theorem card_reads_roundQ_le (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (h : LoopHist α)
    (d : RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget) {L a b f : ℕ}
    (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hab : ∀ pops (x : Run Unit (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b)
    (hk : ∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ f + 1) {n : ℕ}
    (hpops : ∀ s ∈ h.pops, (Random.mids s.hyp.tree).card ≤ n + 1 ∧ s.family.card ≤ f) {W : ℕ}
    (t : Fin K → Fin settings.scanCap → ℕ) (hW : ∑ i, ∑ n, t i n ≤ W)
    (ht : ∀ i s, h.pops[i.val]? = some s → ∀ n : Fin settings.scanCap,
      (Random.harvestFirst (fun z => (familyReadQ s.family s.kl s.kh z).eval o)
        settings.level3.walkStart s.hyp s.site ↑h.prefs
        ((List.ofFn fun j => drawFrom D ((d.1.2.1 (some i) n).2 j)).take (t i n))).isSome)
    (hleaf : ∀ r ∈ (roundQ settings D Dsf h d).eval o,
      r.result.hypothesis.tree.leaves.length ≤ n + 2) :
    ((roundQ settings D Dsf h d).reads o).card
      ≤ roundCallBound a b f settings.level3.probeBudget W L (Fintype.card α) n := by
  have hA := card_reads_clusterDrawsQ_le o D Dsf settings.level3.walkStart
    settings.level3.harvestReadCap settings.scanCap h d.1 hD hpops t hW ht
  unfold roundQ at hleaf
  unfold roundQ roundCallBound
  rw [eval_bind] at hleaf
  rw [reads_bind]
  refine (Finset.card_union_le _ _).trans ?_
  rw [Nat.add_assoc]
  refine Nat.add_le_add hA ?_
  rcases hdr : (clusterDrawsQ D Dsf settings.level3.walkStart settings.level3.harvestReadCap
    settings.scanCap h d.1).eval o with _ | dr
  · simp
  rw [hdr] at hleaf
  simp only at hleaf ⊢
  set ps := loopPopulations K h.pops.length
  set sched := settings.schedule ps
  set x := streams (fun _ => ∅) none dr
  obtain ⟨hpa, hsb⟩ := hab ps x
  split_ifs at hleaf ⊢ with hsc
  swap
  · simp
  rw [eval_bind] at hleaf
  rw [reads_bind]
  refine (Finset.card_union_le _ _).trans (Nat.add_le_add ?_ ?_)
  · rw [reads_table]
    refine (List.toFinset_card_le _).trans ?_
    rw [Finset.length_toList]
    refine Finset.card_image_le.trans ?_
    rw [Finset.card_product]
    exact Nat.mul_le_mul hpa ((Finset.card_insert_le _ _).trans (Nat.add_le_add_right hsb 1))
  split_ifs at hleaf ⊢ with hex
  swap
  · simp
  simp only [eval_map, eval_subst, Option.mem_def, Option.some.injEq, forall_eq'] at hleaf
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
  exact Nat.mul_le_mul_left _ (card_reads_level3Q_le _ _ n _ (fun i =>
    drawFrom_length_le _ (ae_popLaw (ae_restrict_of_ae hD) (by simp)) _) hleaf)

section Small

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
  (draws : Fin K → LoopRoundDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget)

/-- Every round the loop runs has a family of at most `f` members. -/
lemma loopRounds_small {f : ℕ} (hk : ∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ f + 1)
    (r : ℕ) : ∀ rs ∈ loopRounds settings O D Dsf draws r, ∀ o ∈ rs, o.family.card ≤ f := by
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
          obtain ⟨B, hB, hr⟩ := hcond.2
          have hBs := (Classical.epsilon_spec (p := fun B => B ∈ settings.schedule
            (loopPopulations K (specHistOf rs).pops.length) ∧ streams (fun _ => ∅) none dr ∈ ret
              O.mq (loopPopulations K (specHistOf rs).pops.length) none settings.η₀
              settings.indecisionLimit settings.εcov settings.gateLevel B) ⟨B, hB, hr⟩).1
          exact (card_clusterAt_le _ _ _ _).trans (Nat.sub_le_of_le_add (hk _ _ hBs))
        · cases hdr
    · rw [Option.mem_some_iff] at hr
      subst hr
      exact hrr

end Small

section Prefix

variable (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
  (rd : Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget)
  (o : FreeMonoid α → ℝ)

lemma loopRoundsQ_prefix_succ (r : ℕ) {rs' : List (LoopRound α)}
    (h : (loopRoundsQ settings D Dsf rd (r + 1)).eval o = some rs') :
    ∃ rs, (loopRoundsQ settings D Dsf rd r).eval o = some rs ∧ rs <+: rs' := by
  rw [loopRoundsQ, eval_bind] at h
  rcases hrs : (loopRoundsQ settings D Dsf rd r).eval o with _ | rs
  · rw [hrs] at h; simp at h
  rw [hrs] at h
  refine ⟨rs, rfl, ?_⟩
  simp only at h
  split_ifs at h with hc
  · rw [eval_map] at h
    obtain ⟨ro, -, rfl⟩ := Option.map_eq_some_iff.1 h
    exact List.prefix_append _ _
  · simp at h
    subst h
    exact List.prefix_refl _

/-- A run's earlier rounds are a prefix of its later ones. -/
lemma loopRoundsQ_prefix {r s : ℕ} (hrs : r ≤ s) {rs' : List (LoopRound α)}
    (h : (loopRoundsQ settings D Dsf rd s).eval o = some rs') :
    ∃ rs, (loopRoundsQ settings D Dsf rd r).eval o = some rs ∧ rs <+: rs' := by
  induction s generalizing rs' with
  | zero =>
    obtain rfl : r = 0 := by omega
    exact ⟨rs', h, List.prefix_refl _⟩
  | succ s ih =>
    rcases Nat.lt_or_ge r (s + 1) with hlt | hge
    · obtain ⟨rs, h1, h2⟩ := loopRoundsQ_prefix_succ settings D Dsf rd o s h
      obtain ⟨rs0, h3, h4⟩ := ih (by omega) h1
      exact ⟨rs0, h3, h4.trans h2⟩
    · obtain rfl : r = s + 1 := by omega
      exact ⟨rs', h, List.prefix_refl _⟩

/-- The first `r ≤ K` rounds, all harvesting, are `r` rounds. -/
lemma loopRoundsQ_length {r : ℕ} (hr : r ≤ K) {rs : List (LoopRound α)}
    (h : (loopRoundsQ settings D Dsf rd r).eval o = some rs)
    (hall : ∀ ro ∈ rs, ro.result.ending.harvested ≠ none) : rs.length = r := by
  induction r generalizing rs with
  | zero =>
    simp [loopRoundsQ] at h
    subst h
    rfl
  | succ r ih =>
    rw [loopRoundsQ, eval_bind] at h
    rcases hrs : (loopRoundsQ settings D Dsf rd r).eval o with _ | rs0
    · rw [hrs] at h; simp at h
    rw [hrs] at h
    simp only at h
    split_ifs at h with hc
    · rw [eval_map] at h
      obtain ⟨ro, -, rfl⟩ := Option.map_eq_some_iff.1 h
      rw [List.length_append, List.length_singleton,
        ih (by omega) hrs fun x hx => hall x (List.mem_append_left _ hx)]
    · simp at h
      subst h
      exact absurd ⟨by omega, hall⟩ hc

end Prefix

/-- The program's first `r ≤ K` rounds, against the answers at the noise `ω`, where the loop
returns `rounds` whose harvested draws walk at most `W` probes a round, read at most `r` times
`roundCallBound` strings. -/
theorem card_reads_loopRoundsQ_le (O : Oracle μ (FreeMonoid α)) (ω : Ω)
    (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (rd : Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget)
    {L a b f : ℕ} (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hab : ∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b)
    (hk : ∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ f + 1) {n : ℕ}
    {rounds : List (LoopRound α)} {W : ℕ}
    (hout : (loopQ settings D Dsf rd).eval (fun w => O.mq w ω) = some rounds)
    (hlv : ∀ o ∈ rounds, o.result.hypothesis.tree.leaves.length ≤ n + 2)
    (hW : HarvestProbesWithin settings O D (ω, rd) rounds W) {r : ℕ} (hr : r ≤ K) :
    ((loopRoundsQ settings D Dsf rd r).reads fun w => O.mq w ω).card
      ≤ r * roundCallBound a b f settings.level3.probeBudget W L (Fintype.card α) n := by
  have hout' : (loopRoundsQ settings D Dsf rd K).eval (fun w => O.mq w ω) = some rounds := hout
  induction r with
  | zero => simp [loopRoundsQ]
  | succ r ih =>
    rw [loopRoundsQ, reads_bind, Nat.succ_mul]
    refine (Finset.card_union_le _ _).trans (Nat.add_le_add (ih (by omega)) ?_)
    obtain ⟨rs', hrs', hpre⟩ := loopRoundsQ_prefix settings D Dsf rd _ hr hout'
    rw [loopRoundsQ, eval_bind] at hrs'
    rcases hrs : (loopRoundsQ settings D Dsf rd r).eval (fun w => O.mq w ω) with _ | rs
    · simp
    rw [hrs] at hrs'
    simp only at hrs' ⊢
    split_ifs with hc
    swap
    · simp
    rw [dif_pos hc, eval_map] at hrs'
    obtain ⟨ro, hoo, rfl⟩ := Option.map_eq_some_iff.1 hrs'
    have hlen := loopRoundsQ_length settings D Dsf rd _ hc.1.le hrs hc.2
    have htake : rounds.take r = rs := by
      have e := List.prefix_iff_eq_take.1 ((List.prefix_append rs [ro]).trans hpre)
      rw [hlen] at e
      exact e.symm
    have hlt : r < rounds.length := by
      have := hpre.length_le
      rw [List.length_append, List.length_singleton, hlen] at this
      omega
    obtain ⟨t, htW, ht⟩ := hW r hc.1 hlt
    rw [htake] at ht
    have e := (eval_loopRoundsQ O ω settings D Dsf rd r).1
    rw [hrs] at e
    obtain ⟨rs0, hrs0, rfl⟩ := Option.map_eq_some_iff.1 e.symm
    have hsm := loopRounds_small settings O D Dsf _ hk r rs0 hrs0
    rw [reads_map]
    refine card_reads_roundQ_le _ settings D Dsf _ _ hD (fun pops x => hab pops (ω, x.2)) hk
      (fun s hs => ?_) t htW (fun i s hs m => ?_) fun x hx => ?_
    · obtain ⟨o, ho, hos⟩ := List.mem_filterMap.1 hs
      obtain ⟨o', ho', rfl⟩ := List.mem_map.1 ho
      obtain ⟨site, -, rfl⟩ := Option.map_eq_some_iff.1 hos
      exact ⟨Nat.le_of_succ_le_succ ((Random.mids_card _).trans
        (hlv _ (hpre.subset (List.mem_append_left _ ho)))), hsm o' ho'⟩
    · have h1 := ht i s hs m
      rwa [show (fun z => (familyReadQ s.family s.kl s.kh z).eval fun w => O.mq w ω)
        = fun z => familyRead O.mq s.family s.kl s.kh z ω from rfl]
    · rw [Option.mem_def, hoo, Option.some.injEq] at hx
      subst hx
      exact hlv _ (hpre.subset (List.mem_append_right _ (List.mem_singleton_self _)))

/-- Where every round the filled draws run is good and its harvested draws walk at most `B`
probes, the loop's rounds meet `HarvestProbesWithin` at `B`. -/
theorem harvestProbesWithin_of (O : Oracle μ (FreeMonoid α)) (settings : LoopSettings K)
    (D Dsf : Measure (FreeMonoid α))
    (p : Ω × (Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget))
    (B : ℕ)
    (hg : ∀ r (hr : r < K), AllHarvested (idealRounds settings O D Dsf (persistDraws p) r) →
      RoundGood settings O D Dsf (specHistOf (idealRounds settings O D Dsf (persistDraws p) r))
        (persistDraws p ⟨r, hr⟩))
    (hw : RoundsWalkWithin settings O D Dsf p B) :
    HarvestProbesWithin settings O D p
      ((idealRounds settings O D Dsf (persistDraws p) K).map SpecRound.data) B := by
  intro r hr hlt
  rw [List.length_map] at hlt
  have htk : (idealRounds settings O D Dsf (persistDraws p) K).take r
      = idealRounds settings O D Dsf (persistDraws p) r :=
    take_idealRounds settings O D Dsf (persistDraws p) hr.le
  have hall : AllHarvested (idealRounds settings O D Dsf (persistDraws p) r) := by
    intro o ho
    rw [← htk] at ho
    obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ho
    rw [List.getElem_take]
    simp only [List.length_take] at hi
    exact harvested_of_lt settings O D Dsf (persistDraws p) K i (by omega)
  set ir := idealRounds settings O D Dsf (persistDraws p) with hird
  have hloop := idealRounds_mem_loop settings O Dsf D (persistDraws p) hg
  have hspec : specHistOf (ir r) = (histOf ((ir r).map SpecRound.data)).spec O p.1 :=
    specHistOf_eq O p.1 fun o ho =>
      loopRounds_read O settings D Dsf p (ir K) hloop o
        (List.mem_of_mem_take (by rw [htk]; exact ho))
  rw [← List.map_take, htk]
  set h := histOf ((ir r).map SpecRound.data) with hhd
  set u := (p.2 ⟨r, hr⟩).1.2.1 with hud
  have hwr := hw r hr hall
  unfold roundWalk at hwr
  rw [hspec] at hwr
  have hc := (hg r hr hall).1
  rw [hspec] at hc
  refine ⟨fun i n => match h.pops[i.val]? with
    | some s => walkLen (hitSet (s.spec O p.1) settings.level3.walkStart h.prefs)
        (List.ofFn (probeBlocks D settings.level3.harvestReadCap settings.scanCap (u (some i)) n))
    | none => 0, le_trans (le_of_eq ?_) hwr, fun i s hs n => ?_⟩
  · unfold harvestWalk
    refine Finset.sum_congr rfl fun i _ => ?_
    simp only [LoopHist.spec, List.getElem?_map]
    rcases h.pops[i.val]? with _ | s
    · simp [popWalk]
    · rfl
  · simp only [hs]
    refine harvestFirst_take_walkLen _ _ _ _ _ _ ?_
    unfold clusterDraws at hc
    rw [Option.isSome_map] at hc
    unfold allSome at hc
    split_ifs at hc with hal
    · have hj := hal (some i)
      simp only [popDraws, LoopHist.spec, List.getElem?_map, hs, Option.map_some] at hj
      unfold Random.drawManyFirst at hj
      split_ifs at hj with hd
      · exact hd n
      · simp at hj
    · simp at hc

end Reads

/-! ## The learner -/

section Learner

omit [DecidableEq α] in
lemma card_reads_pickLeafQ_le (H : HypothesisDFA α) {n : ℕ} (xs : Fin n → FreeMonoid α)
    (read : FreeMonoid α → ARU) : ((pickLeafQ H xs).reads read).card ≤ n := by
  unfold pickLeafQ
  rw [reads_map, reads_table]
  exact (List.toFinset_card_le _).trans (List.length_ofFn).le

variable (O : Oracle μ (FreeMonoid α))

/-- The learner, its rounds' trees at most `q + 2` leaves and their harvested draws walking at most
`W` probes a round, asks the loop's strings and, at the last round's family, `|F|` strings for each
held-out string. -/
theorem card_reads_learnerQ_le (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    [IsProbabilityMeasure D]
    (p : Ω × (Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget))
    {n : ℕ} (xs : Fin n → FreeMonoid α) {L a b f : ℕ} (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hab : ∀ pops (x : Run Ω (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b)
    (hk : ∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ f + 1) {q : ℕ}
    {rounds : List (LoopRound α)} {W : ℕ}
    (hout : indecisiveLStarLoop settings O D Dsf p = some rounds)
    (hlv : ∀ o ∈ rounds, o.result.hypothesis.tree.leaves.length ≤ q + 2)
    (hW : HarvestProbesWithin settings O D p rounds W) :
    ((learnerQ settings D Dsf p.2 xs).reads fun w => O.mq w p.1).card
      ≤ K * roundCallBound a b f settings.level3.probeBudget W L (Fintype.card α) q + n * f := by
  have hloop := card_reads_loopRoundsQ_le O p.1 settings D Dsf p.2 hD hab hk hout hlv hW le_rfl
  unfold learnerQ
  rw [reads_bind]
  refine (Finset.card_union_le _ _).trans (Nat.add_le_add hloop ?_)
  rw [show (loopQ settings D Dsf p.2).eval (fun w => O.mq w p.1) = _ from
    (eval_loopRoundsQ O p.1 settings D Dsf p.2 K).1]
  rcases hrs : loopRounds settings O D Dsf (persistDraws (p.1, p.2)) K with _ | rs
  · simp
  have hsm := loopRounds_small settings O D Dsf _ hk K rs hrs
  simp only [Option.map_some, List.getLast?_map]
  rcases hl : rs.getLast? with _ | o
  · simp
  simp only [Option.map_some]
  rw [reads_map]
  have hF := hsm o (List.mem_of_getLast? hl)
  refine (card_reads_subst_le _ _ _ (n := f) fun q _ =>
    (card_reads_familyReadQ_le _ _ _ q _).trans hF).trans ?_
  exact Nat.mul_le_mul_right _ (card_reads_pickLeafQ_le _ _ _)

end Learner

end OrthoDFA

end
