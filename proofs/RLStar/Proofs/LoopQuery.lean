import RLStar.Proofs.QueryLevel3
import RLStar.Proofs.ClusterCongr
import RLStar.Proofs.LoopGood
import RLStar.Code.LoopDFA
import RLStar.LowLevelClaims.LoopDFA

/-!
# The loop program

`loopQ`'s run against the answers at one noise is `loopRounds` at that noise in every round, each
round with its family read dropped (`indecisiveLStarLoop_eq`), and each of those rounds reads its
family at that noise (`loopRounds_read`). Against any answers, with probes at most `L` long, `T`
the probe budget, each round's clustering reading at most `a` prefixes and `b` suffixes and each
family at most `f` members, a round whose tree and whose samplers' trees have at most `n + 2`
leaves and whose harvested draws find their strings within `W` probes in all asks at most

    f W (3 + ⌈log₂ L⌉)(n + 1) + a (b + 1) + f (T probeSifts L (n + 1) + n (n + 2)²|Σ|)

strings (`card_reads_roundQ_le`): its harvested draws, its clustering and its level 3. A population
draws until its draws hold its samples, and only those draws are walked (`eval_popDrawsQ`). A loop
returning rounds whose trees have at most `n + 2` leaves and that meet `HarvestProbesWithin` at
`W` asks at most `K` times that (`card_reads_loopRoundsQ_le`); the loop's rounds meet it where
every round is good and its harvested populations' first `Nh` draws hold their samples and walk at
most `W` probes in all (`harvestProbesWithin_of`), so each population stops within `Nh` draws.
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

lemma allSome_eq_some {ι β : Type*} {f : ι → Option β} {g : ι → β} (h : allSome f = some g)
    (i : ι) : f i = some (g i) := by
  unfold allSome at h
  split_ifs at h with hall
  · rw [Option.some.injEq] at h
    subst h
    exact (Option.some_get _).symm

lemma eval_popDrawsQ (D : Measure (FreeMonoid α)) (k N M : ℕ) (pops : List (HarvestSampler α))
    (T : Finset (FreeMonoid α)) (stop : List (FreeMonoid α) → Prop) (j : Option (Fin K))
    (u : ℕ → unitInterval × (Fin N → unitInterval)) :
    (popDrawsQ D k N M pops T stop j u).eval (fun w => O.mq w ω)
      = popDrawList D k N M (pops.map (HarvestSampler.spec O ω)) T stop j u := by
  rcases j with _ | i
  · rfl
  simp only [popDrawsQ, popDrawList, List.getElem?_map]
  rcases pops[i.val]? with _ | s
  · rfl
  have hr : (fun z => (familyReadQ s.family s.kl s.kh z).eval fun w => O.mq w ω)
      = (s.spec O ω).read := funext fun z => rfl
  simp only [Option.map_some, eval_subst, eval_extendUntil, eval_harvestFirstQ, hr]
  rfl

lemma eval_drawListsQ (D : Measure (FreeMonoid α)) (k N M : ℕ) (sched : Finset State)
    (h : LoopHist α) (c : ClusterRest α K N) :
    (drawListsQ D k N M sched h c).eval (fun w => O.mq w ω)
      = drawLists D k N M sched (h.spec O ω) c := by
  unfold drawListsQ drawLists
  rw [eval_map, eval_pi]
  simp only [eval_popDrawsQ]
  rfl

/-- A draw's noise folded away. -/
def Draws.flat {S J : Type*} (dr : Draws Ω S J) : Draws Unit S J := (((), dr.1.2), dr.2)

open scoped Classical in
lemma eval_roundQ (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (d : settings.RoundDraws α) :
    (roundQ settings D Dsf h d).eval (fun w => O.mq w ω)
      = (loopRound settings O D Dsf (h.spec O ω) (ω, d)).map SpecRound.data
      ∧ ∀ o ∈ loopRound settings O D Dsf (h.spec O ω) (ω, d), o.ReadsAt O ω := by
  have hlen : (h.spec O ω).pops.length = h.pops.length := by simp [LoopHist.spec]
  unfold roundQ loopRound
  simp only [hlen]
  rw [eval_bind, eval_drawListsQ]
  rcases hdr : drawLists D settings.level3.walkStart settings.level3.harvestReadCap
    settings.scanCap (settings.schedule (loopPopulations K h.pops.length)) (h.spec O ω) d.1
    with _ | tabs
  · simp
  simp only [Option.bind_some]
  set ps := loopPopulations K h.pops.length
  set sched := settings.schedule ps
  set dr := drawsOf ω D Dsf settings.scanCap h.prefs h.suffs d.1 tabs with hdrd
  have hflat : drawsOf () D Dsf settings.scanCap h.prefs h.suffs d.1 tabs = dr.flat := rfl
  have hspec : drawsOf (ω, d).1 D Dsf settings.scanCap (h.spec O ω).prefs (h.spec O ω).suffs
      (ω, d).2.1 tabs = dr := rfl
  rw [hflat, hspec]
  set xs := streams (fun _ => ∅) none dr with hxs
  set xc := streams (fun _ => ∅) none dr.flat with hxc
  have hx : xc = ((), xs.2) := rfl
  have hxs1 : xs.1 = ω := rfl
  have hscan : ScanWithin sched ps (fun j => (tabs j).length) dr.flat
      ↔ ScanWithin sched ps (fun j => (tabs j).length) dr :=
    Iff.rfl
  by_cases hsc : ScanWithin sched ps (fun j => (tabs j).length) dr
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
    refine (ret_eqOn _ _ sched ps none (Finset.mem_insert_self _ _) _ _ _ _ hB _
      fun w hw => ?_).symm
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
    (rd : Fin K → settings.RoundDraws α) :
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
    have hcond : (∀ o ∈ rs.map SpecRound.data, o.result.ending.IsHarvest)
        ↔ ∀ o ∈ rs, o.result.ending.IsHarvest := by
      simp [SpecRound.data]
    by_cases hc : r < K ∧ ∀ o ∈ rs, o.result.ending.IsHarvest
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
    (p : Ω × (Fin K → settings.RoundDraws α)) :
    indecisiveLStarLoop settings O D Dsf p
      = (loopRounds settings O D Dsf (persistDraws p) K).map (List.map SpecRound.data) :=
  (eval_loopRoundsQ O p.1 settings D Dsf p.2 K).1

lemma indecisiveLStarLoop_some {settings : LoopSettings K} {D Dsf : Measure (FreeMonoid α)}
    {p : Ω × (Fin K → settings.RoundDraws α)}
    {rounds : List (LoopRound α)} (h : indecisiveLStarLoop settings O D Dsf p = some rounds) :
    ∃ rs, loopRounds settings O D Dsf (persistDraws p) K = some rs
      ∧ rounds = rs.map SpecRound.data := by
  rw [indecisiveLStarLoop_eq] at h
  obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 h
  exact ⟨rs, hrs, rfl⟩

theorem loopRounds_read (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (p : Ω × (Fin K → settings.RoundDraws α)) :
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

/-- Draw `n` harvesting within its first `t n` probes, a harvested population's draws `l` sift
`∑_{n < |l|} t n` probes, each a family read of at most `f` strings at `3 + ⌈log₂ L⌉` reads of
each mid. -/
theorem card_reads_popDrawsQ_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N M : ℕ) (pops : List (HarvestSampler α)) (T : Finset (FreeMonoid α))
    (stop : List (FreeMonoid α) → Prop) (i : Fin K)
    (u : ℕ → unitInterval × (Fin N → unitInterval)) {L f P : ℕ}
    (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hpops : ∀ s ∈ pops, (Random.mids s.hyp.tree).card ≤ P + 1 ∧ s.family.card ≤ f)
    {l : List (FreeMonoid α)} (hl : (popDrawsQ D k N M pops T stop (some i) u).eval o = some l)
    (t : ℕ → ℕ)
    (ht : ∀ s, pops[i.val]? = some s → ∀ n < l.length,
      (Random.harvestFirst (fun z => (familyReadQ s.family s.kl s.kh z).eval o) k s.hyp s.site ↑T
        ((List.ofFn fun j => drawFrom D ((u n).2 j)).take (t n))).isSome) :
    ((popDrawsQ D k N M pops T stop (some i) u).reads o).card
      ≤ f * ((∑ n ∈ Finset.range l.length, t n) * ((3 + Nat.clog 2 L) * (P + 1))) := by
  classical
  simp only [popDrawsQ] at hl ⊢
  rcases hs : pops[i.val]? with _ | s
  · simp
  rw [hs] at hl
  obtain ⟨hm, hb⟩ := hpops s (List.mem_of_getElem? hs)
  simp only at hl ⊢
  rw [eval_subst] at hl
  refine (card_reads_subst_le o _ _ (n := f)
    fun q _ => (card_reads_familyReadQ_le _ _ _ q o).trans hb).trans ?_
  rw [Nat.mul_comm _ f]
  refine Nat.mul_le_mul_left _ ?_
  have := card_reads_extendUntil_le _ _ _ (fun n => t n * ((3 + Nat.clog 2 L) * (P + 1))) M [] l
    hl fun m _ hml => (card_reads_harvestFirstQ_take_le _ _ _ _ _ _ (t m)
      (fun y hy => by obtain ⟨j, rfl⟩ := List.mem_ofFn.1 hy; exact drawFrom_length_le _ hD _)
      (ht s hs m hml)).trans (Nat.mul_le_mul_left _ (Nat.mul_le_mul_left _ hm))
  rwa [List.length_nil, ← Finset.range_eq_Ico, ← Finset.sum_mul] at this

/-- The draws `tabs`, draw `n` of population `i` harvesting within its first `t i n` probes, ask at
most `f W` sifts, `W` the probes in all. -/
theorem card_reads_drawListsQ_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N M : ℕ) (sched : Finset State) (h : LoopHist α) (c : ClusterRest α K N) {L f P W : ℕ}
    (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hpops : ∀ s ∈ h.pops, (Random.mids s.hyp.tree).card ≤ P + 1 ∧ s.family.card ≤ f)
    {tabs : Option (Fin K) → List (FreeMonoid α)}
    (htabs : (drawListsQ D k N M sched h c).eval o = some tabs)
    (t : Fin K → ℕ → ℕ) (hW : ∑ i, ∑ n ∈ Finset.range (tabs (some i)).length, t i n ≤ W)
    (ht : ∀ i s, h.pops[i.val]? = some s → ∀ n < (tabs (some i)).length,
      (Random.harvestFirst (fun z => (familyReadQ s.family s.kl s.kh z).eval o) k s.hyp s.site
        ↑h.prefs ((List.ofFn fun j => drawFrom D ((c.2.1 (some i) n).2 j)).take (t i n))).isSome) :
    ((drawListsQ D k N M sched h c).reads o).card
      ≤ f * (W * ((3 + Nat.clog 2 L) * (P + 1))) := by
  unfold drawListsQ at htabs ⊢
  rw [eval_map, eval_pi] at htabs
  rw [reads_map, reads_pi]
  refine (Finset.card_biUnion_le).trans ?_
  rw [Fintype.sum_option]
  have h0 : ((popDrawsQ (K := K) D k N M h.pops h.prefs
      (DrawsHold sched c.2.2.2 (uniTable D M h.prefs (c.2.1 none))) none (c.2.1 none)).reads o).card
      = 0 := by
    simp [popDrawsQ]
  rw [h0, zero_add]
  refine (Finset.sum_le_sum fun i _ => card_reads_popDrawsQ_le o D k N M h.pops h.prefs _ i
    (c.2.1 (some i)) hD hpops (allSome_eq_some htabs (some i)) (t i) (ht i)).trans ?_
  rw [← Finset.mul_sum, ← Finset.sum_mul]
  exact Nat.mul_le_mul_left _ (Nat.mul_le_mul_right _ hW)

open scoped Classical in
/-- A round that returns, against the answers `o`, a tree of at most `n + 2` leaves reads at most
`roundCallBound` strings. -/
theorem card_reads_roundQ_le (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (h : LoopHist α)
    (d : settings.RoundDraws α) {L a b f : ℕ}
    (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hab : ∀ pops (x : Run Unit (FreeMonoid α) (Option (Fin K))),
      (roundPrefs (settings.schedule pops) pops x).card ≤ a
        ∧ (roundSuffs (settings.schedule pops) x).card ≤ b)
    (hk : ∀ pops, ∀ B ∈ settings.schedule pops, B.k ≤ f + 1) {n : ℕ}
    (hpops : ∀ s ∈ h.pops, (Random.mids s.hyp.tree).card ≤ n + 1 ∧ s.family.card ≤ f) {W : ℕ}
    (hW : ∀ tabs, (drawListsQ D settings.level3.walkStart settings.level3.harvestReadCap
        settings.scanCap (settings.schedule (loopPopulations K h.pops.length)) h d.1).eval o
        = some tabs →
      ∃ t : Fin K → ℕ → ℕ, ∑ i, ∑ n ∈ Finset.range (tabs (some i)).length, t i n ≤ W
        ∧ ∀ i s, h.pops[i.val]? = some s → ∀ n < (tabs (some i)).length,
          (Random.harvestFirst (fun z => (familyReadQ s.family s.kl s.kh z).eval o)
            settings.level3.walkStart s.hyp s.site ↑h.prefs
            ((List.ofFn fun j => drawFrom D ((d.1.2.1 (some i) n).2 j)).take (t i n))).isSome)
    (hsome : ((roundQ settings D Dsf h d).eval o).isSome)
    (hleaf : ∀ r ∈ (roundQ settings D Dsf h d).eval o,
      r.result.hypothesis.tree.leaves.length ≤ n + 2) :
    ((roundQ settings D Dsf h d).reads o).card
      ≤ roundCallBound a b f settings.level3.probeBudget W L (Fintype.card α) n := by
  unfold roundQ at hleaf hsome
  unfold roundQ roundCallBound
  rw [eval_bind] at hleaf hsome
  rw [reads_bind]
  refine (Finset.card_union_le _ _).trans ?_
  rw [Nat.add_assoc]
  rcases hdr : (drawListsQ D settings.level3.walkStart settings.level3.harvestReadCap
    settings.scanCap (settings.schedule (loopPopulations K h.pops.length)) h d.1).eval o
    with _ | tabs
  · rw [hdr] at hsome
    simp at hsome
  obtain ⟨t, htW, ht⟩ := hW tabs hdr
  refine Nat.add_le_add (card_reads_drawListsQ_le o D settings.level3.walkStart
    settings.level3.harvestReadCap settings.scanCap _ h d.1 hD hpops hdr t htW ht) ?_
  rw [hdr] at hleaf
  simp only at hleaf ⊢
  set ps := loopPopulations K h.pops.length
  set sched := settings.schedule ps
  set dr := drawsOf () D Dsf settings.scanCap h.prefs h.suffs d.1 tabs
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
  (draws : Fin K → LoopRoundDraws Ω α settings)

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
        obtain ⟨tabs, -, hdr⟩ := Option.mem_bind_iff.1 ho
        dsimp only at hdr
        split_ifs at hdr with hcond
        · rw [← Option.mem_some_iff.1 hdr]
          obtain ⟨B, hB, hr⟩ := hcond.2
          have hBs := (Classical.epsilon_spec (p := fun B => B ∈ settings.schedule
            (loopPopulations K (specHistOf rs).pops.length) ∧ streams (fun _ => ∅) none
              (drawsOf (draws ⟨r, hc.1⟩).1 D Dsf settings.scanCap (specHistOf rs).prefs
                (specHistOf rs).suffs (draws ⟨r, hc.1⟩).2.1 tabs) ∈ ret
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
  (rd : Fin K → settings.RoundDraws α)
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
    (hall : ∀ ro ∈ rs, ro.result.ending.IsHarvest) : rs.length = r := by
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
    (rd : Fin K → settings.RoundDraws α)
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
    have hWr := hW r hc.1 hlt
    rw [htake] at hWr
    have e := (eval_loopRoundsQ O ω settings D Dsf rd r).1
    rw [hrs] at e
    obtain ⟨rs0, hrs0, rfl⟩ := Option.map_eq_some_iff.1 e.symm
    have hsm := loopRounds_small settings O D Dsf _ hk r rs0 hrs0
    rw [reads_map]
    refine card_reads_roundQ_le _ settings D Dsf _ _ hD (fun pops x => hab pops (ω, x.2)) hk
      (fun s hs => ?_) (fun tabs htabs => ?_) (by rw [hoo]; rfl) fun x hx => ?_
    · obtain ⟨o, ho, hos⟩ := List.mem_filterMap.1 hs
      obtain ⟨o', ho', rfl⟩ := List.mem_map.1 ho
      obtain ⟨site, -, rfl⟩ := Option.map_eq_some_iff.1 hos
      exact ⟨Nat.le_of_succ_le_succ ((Random.mids_card _).trans
        (hlv _ (hpre.subset (List.mem_append_left _ ho)))), hsm o' ho'⟩
    · obtain ⟨t, htW, ht⟩ := hWr tabs htabs
      refine ⟨t, htW, fun i s hs m hm => ?_⟩
      have h1 := ht i s hs m hm
      rwa [show (fun z => (familyReadQ s.family s.kl s.kh z).eval fun w => O.mq w ω)
        = fun z => familyRead O.mq s.family s.kl s.kh z ω from rfl]
    · rw [Option.mem_def, hoo, Option.some.injEq] at hx
      subst hx
      exact hlv _ (hpre.subset (List.mem_append_right _ (List.mem_singleton_self _)))

/-- Where every round the filled draws run is good and its harvested populations' first `Nh`
filled draws hold their samples and walk at most `B` probes (`DrawsWithin`), the loop's rounds meet
`HarvestProbesWithin` at `B`. -/
theorem harvestProbesWithin_of (O : Oracle μ (FreeMonoid α)) (settings : LoopSettings K)
    (D Dsf : Measure (FreeMonoid α))
    (p : Ω × (Fin K → settings.RoundDraws α))
    (Nh B : ℕ)
    (hg : ∀ r (hr : r < K), AllHarvested (idealRounds settings O D Dsf (persistDraws p) r) →
      RoundGood settings O D Dsf (specHistOf (idealRounds settings O D Dsf (persistDraws p) r))
        (persistDraws p ⟨r, hr⟩))
    (hw : RoundsWalkWithin settings O D Dsf p Nh B) :
    HarvestProbesWithin settings O D p
      ((idealRounds settings O D Dsf (persistDraws p) K).map SpecRound.data) B := by
  classical
  intro r hr hlt tabs htabs
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
  rw [← List.map_take, htk] at htabs ⊢
  set h := histOf ((ir r).map SpecRound.data) with hhd
  set u := (p.2 ⟨r, hr⟩).1.2.1 with hud
  have hlen : (h.spec O p.1).pops.length = h.pops.length := by simp [LoopHist.spec]
  obtain ⟨hdw, hwr⟩ := hw r hr hall
  rw [hspec] at hdw hwr
  have hc := (hg r hr hall).1
  rw [hspec] at hc
  rw [eval_drawListsQ, ← hlen] at htabs
  obtain ⟨tabs', hdl, hag, -, hpop⟩ := drawLists_of_hit
    (sched := settings.schedule (loopPopulations K (h.spec O p.1).pops.length)) hc
  obtain rfl : tabs = tabs' := (Option.some.inj (hdl.symm.trans htabs)).symm
  have hstep : ∀ (i : Fin K) (s : HarvestSampler α), h.pops[i.val]? = some s →
      ∀ n < settings.scanCap,
      (Random.harvestFirst (fun z => familyRead O.mq s.family s.kl s.kh z p.1)
        settings.level3.walkStart s.hyp s.site ↑h.prefs
        (List.ofFn fun j => drawFrom D ((u (some i) n).2 j))).isSome := by
    intro i s hs n hn
    unfold clusterDraws at hc
    rw [Option.isSome_map] at hc
    unfold allSome at hc
    split_ifs at hc with hal
    · have hj := hal (some i)
      simp only [popDraws, LoopHist.spec, List.getElem?_map, hs, Option.map_some] at hj
      unfold Random.drawManyFirst at hj
      split_ifs at hj with hd
      · exact hd ⟨n, hn⟩
      · simp at hj
    · simp at hc
  have hspi : ∀ (i : Fin K) (s : HarvestSampler α), h.pops[i.val]? = some s →
      (h.spec O p.1).pops[i.val]? = some (s.spec O p.1) := fun i s hs => by
    simp [LoopHist.spec, hs]
  have hcount : ∀ (i : Fin K) (s : HarvestSampler α), h.pops[i.val]? = some s →
      (tabs (some i)).length ≤ Nh := by
    intro i s hs
    by_contra hlt'
    push_neg at hlt'
    have hi : i.val < (h.spec O p.1).pops.length := by
      rw [hlen]; exact (List.getElem?_eq_some_iff.1 hs).1
    refine (hpop i _ (hspi i s hs)).2.2 Nh hlt' ?_
    have e : (tabs (some i)).take Nh = (List.range Nh).map
        ((idealDraws D Dsf (h.spec O p.1)
          ((persistDraws p ⟨r, hr⟩).1, (persistDraws p ⟨r, hr⟩).2.1)).1.2.1.2 (some i)) := by
      refine List.ext_getElem (by simp; omega) fun n h1 h2 => ?_
      simp only [List.length_take] at h1
      have hn : n < (tabs (some i)).length := by omega
      rw [List.getElem_take, List.getElem_map, List.getElem_range, ← hag.2.2.2.1 (some i) n hn]
      simp only [drawsOf, padList]
      rw [List.getD_eq_getElem _ _ hn]
    rw [e]
    exact hdw i hi
  refine ⟨fun i n => match h.pops[i.val]? with
    | some s => walkLen (hitSet (s.spec O p.1) settings.level3.walkStart h.prefs)
        (List.ofFn fun j => drawFrom D ((u (some i) n).2 j))
    | none => 0, le_trans ?_ hwr, fun i s hs n hn => ?_⟩
  · unfold harvestWalk
    refine Finset.sum_le_sum fun i _ => ?_
    simp only [LoopHist.spec, List.getElem?_map]
    rcases hs : h.pops[i.val]? with _ | s
    · simp
    simp only [Option.map_some, popWalk]
    exact (Finset.sum_le_sum_of_subset (Finset.range_mono (hcount i s hs))).trans
      (Fin.sum_univ_eq_sum_range (fun n => walkLen (hitSet (s.spec O p.1)
        settings.level3.walkStart h.prefs) (List.ofFn fun j => drawFrom D ((u (some i) n).2 j)))
        Nh).symm.le
  · simp only [hs]
    exact harvestFirst_take_walkLen _ _ _ _ _ _ (hstep i s hs n
      (hn.trans_le (hpop i _ (hspi i s hs)).1))

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
    (p : Ω × (Fin K → settings.RoundDraws α))
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
