import OrthoDFA.Proofs.LoopPersist

/-!
# What fixes a round, and the rounds before it

Over its draws, a round's clustering reads the oracle only at its prefixes followed by its pool's
suffixes or by nothing, `schedReads`: at every state of the schedule its tests read no further,
since the gate's pool is capped. Its level 4 and the harvest data it hands on read only short
strings followed by the pool's suffixes. So a round's harvest data are fixed by the answers at
`roundReads` (`popCode_congr`), and the earlier rounds' harvest data by the answers at the union of
their rounds' sets, `histReads` (`codes_congr`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} {K : ℕ}

section Sets

variable {S : Type*} [Stringlike S] {J : Type*} [Fintype J]

lemma prefixesAt_mono (pops : Finset J) {m m' : ℕ} (h : m ≤ m') (x : Run Ω S J) :
    prefixesAt pops m x ⊆ prefixesAt pops m' x := by
  intro p hp
  obtain ⟨j, hj, hp⟩ := Finset.mem_biUnion.1 hp
  obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hp
  exact Finset.mem_biUnion.2 ⟨j, hj, Finset.mem_image.2 ⟨i, Finset.mem_range.2
    ((Finset.mem_range.1 hi).trans_le h), rfl⟩⟩

lemma certOf_mono (j : J) {m m' : ℕ} (h : m ≤ m') (x : Run Ω S J) :
    certOf j m x ⊆ certOf j m' x := by
  intro p hp
  obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hp
  exact Finset.mem_image.2 ⟨i, Finset.mem_range.2 ((Finset.mem_range.1 hi).trans_le h), rfl⟩

lemma poolAt_mono {M M' : ℕ} (h : M ≤ M') (x : Run Ω S J) : poolAt M x ⊆ poolAt M' x := by
  intro p hp
  rcases Finset.mem_insert.1 hp with rfl | hp
  · exact Finset.mem_insert_self _ _
  · obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hp
    exact Finset.mem_insert_of_mem (Finset.mem_image.2 ⟨i, Finset.mem_range.2
      ((Finset.mem_range.1 hi).trans_le h), rfl⟩)

lemma readSet_mono' {P P' C C' : Finset S} (hP : P ⊆ P') (hC : C ⊆ C') :
    readSet P C ⊆ readSet P' C' := fun w hw => by
  classical
  obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hw
  obtain ⟨hp, hv⟩ := Finset.mem_product.1 hpv
  exact mem_readSet (hP hp) (hC hv)

/-- The schedule's largest prefix count. -/
def schedNP (sched : Finset State) : ℕ := sched.sup State.npref

/-- The schedule's largest suffix count. -/
def schedNS (sched : Finset State) : ℕ := sched.sup State.nsuff

/-- Every prefix the clustering reads at a state of `sched`: each population's draws, and its
certification draws to twice the count. -/
def schedPrefs (sched : Finset State) (pops : Finset J) (x : Run Ω S J) : Finset S :=
  prefixesAt pops (schedNP sched) x ∪ pops.biUnion fun j => certOf j (2 * schedNP sched) x

/-- Every string the clustering reads at a state of `sched`. -/
def schedReads (sched : Finset State) (pops : Finset J) (x : Run Ω S J) : Finset S :=
  readSet (schedPrefs sched pops x) (insert 1 (poolAt (schedNS sched) x))

end Sets

section Congr

variable [IsProbabilityMeasure μ]

/-- The clustering's state test at `B`, at two noises the reads of `schedReads` cannot tell
apart. -/
lemma ret_congr (O : Oracle μ (FreeMonoid α)) (sched : Finset State)
    (pops : Finset (Option (Fin K))) (uni : Option (Fin K)) (huni : uni ∈ pops) (ι a : ℝ)
    {B : State} (hB : B ∈ sched) {ω ω' : Ω}
    (dr : ((ℕ → FreeMonoid α) × (Option (Fin K) → ℕ → FreeMonoid α))
      × (Option (Fin K) → ℕ → FreeMonoid α))
    (h : ∀ w ∈ schedReads sched pops ((ω, dr) : Run Ω (FreeMonoid α) (Option (Fin K))),
      (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    ((ω, dr) : Run Ω (FreeMonoid α) (Option (Fin K))) ∈ ret O.mq pops uni ι a B
      ↔ ((ω', dr) : Run Ω (FreeMonoid α) (Option (Fin K))) ∈ ret O.mq pops uni ι a B := by
  classical
  set x : Run Ω (FreeMonoid α) (Option (Fin K)) := (ω, dr)
  have hnp : B.npref ≤ schedNP sched := Finset.le_sup (f := State.npref) hB
  have hns : B.nsuff ≤ schedNS sched := Finset.le_sup (f := State.nsuff) hB
  set P := prefixesAt pops B.npref x
  set C := poolAt B.nsuff x
  have hone : (1 : FreeMonoid α) ∈ C := Finset.mem_insert_self _ _
  have hCs : C ⊆ insert 1 (poolAt (schedNS sched) x) :=
    (poolAt_mono hns x).trans (Finset.subset_insert _ _)
  have hPs : P ⊆ schedPrefs sched pops x :=
    (prefixesAt_mono pops hnp x).trans Finset.subset_union_left
  have hcert : ∀ j ∈ pops, ∀ e ≤ B.npref, certOf j (B.npref + e) x ⊆ schedPrefs sched pops x :=
    fun j hj e he => (certOf_mono j (by omega) x).trans
      ((Finset.subset_biUnion_of_mem (fun j => certOf j (2 * schedNP sched) x) hj).trans
        Finset.subset_union_right)
  set cl : Ω → Finset (FreeMonoid α) := fun ω'' =>
    clusterOf lloydClusterer O B.sc B.scd P C B.k ω''
  have hcl : cl ω = cl ω' :=
    clusterOf_congr_mq O B.sc B.scd P C B.k hone fun w hw => h w (readSet_mono' hPs hCs hw)
  have hins : ∀ v ∈ insert 1 (cl ω), v ∈ insert 1 (poolAt (schedNS sched) x) := by
    intro v hv
    rcases Finset.mem_insert.1 hv with rfl | hv
    · exact Finset.mem_insert_self _ _
    · exact hCs (clusterOf_subset O B.sc B.scd P C B.k ω hone hv)
  have hread : ∀ Q ⊆ schedPrefs sched pops x, ∀ p ∈ Q, ∀ v ∈ insert 1 (cl ω),
      (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1) :=
    fun Q hQ p hp v hv => h _ (mem_readSet (hQ hp) (hins v hv))
  have hseed : ∀ Q ⊆ schedPrefs sched pops x, ∀ p ∈ Q, (O.mq p ω = 1 ↔ O.mq p ω' = 1) :=
    fun Q hQ p hp => by
      have := hread Q hQ p hp 1 (Finset.mem_insert_self _ _)
      simpa using this
  have hret : ∀ ω'' : Ω, ((ω'', dr) : Run Ω (FreeMonoid α) (Option (Fin K)))
      ∈ ret O.mq pops uni ι a B ↔ (B.k ≤ (cl ω'').card + 1
        ∧ (∀ j ∈ pops, (((certOf j B.npref x).filter fun p => ¬ decided O.mq B.lo (B.hi + 1)
          (insert 1 (cl ω'')) p ω'').card : ℝ) ≤ ι * (certOf j B.npref x).card)
        ∧ ∃ e ≤ B.npref, admitted O.mq B.lo B.hi a (cl ω'') (certOf uni (B.npref + e) x) ω'') := by
    intro ω''
    simp only [ret, familyAt, clusterAt_eq, clusterAt_eq_clusterOf, Set.mem_setOf_eq]
    rfl
  rw [hret ω, hret ω', ← hcl]
  refine and_congr Iff.rfl (and_congr ?_ ?_)
  · refine forall₂_congr fun j hj => ?_
    have hc : (certOf j B.npref x) ⊆ schedPrefs sched pops x := by
      have := hcert j hj 0 (Nat.zero_le _); simpa using this
    have hf : (certOf j B.npref x).filter (fun p => ¬ decided O.mq B.lo (B.hi + 1)
          (insert 1 (cl ω)) p ω)
        = (certOf j B.npref x).filter (fun p => ¬ decided O.mq B.lo (B.hi + 1)
          (insert 1 (cl ω)) p ω') :=
      Finset.filter_congr fun p hp => by
        rw [decided, decided, voteCount_alike O (ω := ω) (ω' := ω')
          fun v hv => hread _ hc p hp v hv]
    rw [hf]
  · refine exists_congr fun e => and_congr_right fun he => ?_
    have hc := hcert uni huni e he
    unfold admitted
    rw [agreeCount_alike O (lo := B.lo) (hi := B.hi) (F := cl ω) (P := certOf uni (B.npref + e) x)
      (ω := ω) (ω' := ω')
      (fun p hp v hv => hread _ hc p hp v (Finset.mem_insert_of_mem hv))
      (fun p hp => hseed _ hc p hp)]

lemma clusterAt_congr_sched (O : Oracle μ (FreeMonoid α)) (sched : Finset State)
    (pops : Finset (Option (Fin K))) {B : State} (hB : B ∈ sched) {ω ω' : Ω}
    (dr : ((ℕ → FreeMonoid α) × (Option (Fin K) → ℕ → FreeMonoid α))
      × (Option (Fin K) → ℕ → FreeMonoid α))
    (h : ∀ w ∈ schedReads sched pops ((ω, dr) : Run Ω (FreeMonoid α) (Option (Fin K))),
      (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    clusterAt O.mq pops ((ω, dr) : Run Ω (FreeMonoid α) (Option (Fin K))) B
      = clusterAt O.mq pops ((ω', dr) : Run Ω (FreeMonoid α) (Option (Fin K))) B := by
  classical
  have hnp : B.npref ≤ schedNP sched := Finset.le_sup (f := State.npref) hB
  have hns : B.nsuff ≤ schedNS sched := Finset.le_sup (f := State.nsuff) hB
  rw [clusterAt_eq, clusterAt_eq, clusterAt_eq_clusterOf, clusterAt_eq_clusterOf]
  exact clusterOf_congr_mq O B.sc B.scd _ _ B.k (Finset.mem_insert_self _ _) fun w hw =>
    h w (readSet_mono' ((prefixesAt_mono pops hnp _).trans Finset.subset_union_left)
      ((poolAt_mono hns _).trans (Finset.subset_insert _ _)) hw)

end Congr

section Round

variable [IsProbabilityMeasure μ]

lemma readStrings_mono {Z Z' F F' : Finset (FreeMonoid α)} (hZ : Z ⊆ Z') (hF : F ⊆ F') :
    readStrings Z F ⊆ readStrings Z' F' := fun w hw => by
  obtain ⟨⟨z, v⟩, hzv, rfl⟩ := Finset.mem_image.1 hw
  obtain ⟨hz, hv⟩ := Finset.mem_product.1 hzv
  exact Finset.mem_image.2 ⟨(z, v), Finset.mem_product.2 ⟨hZ hz, hF hv⟩, rfl⟩

lemma shortStrings_mono {N N' : ℕ} (h : N ≤ N') :
    shortStrings (α := α) N ⊆ shortStrings N' := fun z hz =>
  mem_shortStrings.2 ((mem_shortStrings.1 hz).trans h)

/-- The draws a round's clustering makes after `harvests`, its noise aside. -/
def runOf (harvests : List (Measure (FreeMonoid α))) (c : ClusterRest α K) :
    ((ℕ → FreeMonoid α) × (Option (Fin K) → ℕ → FreeMonoid α))
      × (Option (Fin K) → ℕ → FreeMonoid α) :=
  (clusterRun (Ω := Unit) harvests ((), c)).2

omit [IsProbabilityMeasure μ] in
lemma clusterRun_eq (harvests : List (Measure (FreeMonoid α))) (ω : Ω) (c : ClusterRest α K) :
    clusterRun harvests (ω, c) = (ω, runOf harvests c) := rfl

/-- Every string a round after `harvests` reads: its clustering's, and through its pool's suffixes
the strings at most `max N (probes + 1 + maxLeaves)` long, which its level 4 over its probes and the
data it hands on read. -/
def roundReads (settings : LoopSettings K) (D : Measure (FreeMonoid α)) (N : ℕ)
    (harvests : List (Measure (FreeMonoid α))) (ω : Ω)
    (d : RoundDraws α K settings.level4.probeBudget) : Finset (FreeMonoid α) :=
  schedReads (settings.schedule (loopPopulations K harvests.length))
      (loopPopulations K harvests.length) ((ω, runOf harvests d.1) :
        Run Ω (FreeMonoid α) (Option (Fin K)))
    ∪ readStrings (shortStrings (max N (probesLen (probesOf settings D
        (clusterKeys (settings.schedule (loopPopulations K harvests.length))
          (loopPopulations K harvests.length) settings.level4.walkStart
          ((ω, runOf harvests d.1) : Run Ω (FreeMonoid α) (Option (Fin K))))
        d.2) + 1 + settings.level4.maxLeaves)))
      ((poolAt (schedNS (settings.schedule (loopPopulations K harvests.length)))
        ((ω, runOf harvests d.1) : Run Ω (FreeMonoid α) (Option (Fin K)))).erase 1)

omit [IsProbabilityMeasure μ] in
lemma schedReads_noise (sched : Finset State) (pops : Finset (Option (Fin K))) (ω ω' : Ω)
    (dr : ((ℕ → FreeMonoid α) × (Option (Fin K) → ℕ → FreeMonoid α))
      × (Option (Fin K) → ℕ → FreeMonoid α)) :
    schedReads sched pops ((ω, dr) : Run Ω (FreeMonoid α) (Option (Fin K)))
      = schedReads sched pops ((ω', dr) : Run Ω (FreeMonoid α) (Option (Fin K))) := rfl

theorem clusterSpec_congr (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D' : Measure (FreeMonoid α)) (εcov slack : ℝ) (harvests : List (Measure (FreeMonoid α)))
    {ω ω' : Ω} (c : ClusterRest α K)
    (h : ∀ w ∈ schedReads (settings.schedule (loopPopulations K harvests.length))
      (loopPopulations K harvests.length) (clusterRun harvests (ω, c)),
      (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    ClusterSpec settings O D' εcov slack harvests (clusterRun harvests (ω, c))
      ↔ ClusterSpec settings O D' εcov slack harvests (clusterRun harvests (ω', c)) := by
  set pops := loopPopulations K harvests.length
  set sched := settings.schedule pops
  have hnone : none ∈ pops := Finset.mem_insert_self _ _
  have hret : ∀ B ∈ sched, (clusterRun harvests (ω, c) ∈ ret O.mq pops none
      settings.indecisionLimit settings.gateLevel B ↔ clusterRun harvests (ω', c) ∈ ret O.mq pops
        none settings.indecisionLimit settings.gateLevel B) := fun B hB =>
    ret_congr O sched pops none hnone _ _ hB (runOf harvests c) h
  have hfam : ∀ B ∈ sched, familyAt O.mq pops (clusterRun harvests (ω, c)) B
      = familyAt O.mq pops (clusterRun harvests (ω', c)) B := fun B hB => by
    unfold familyAt
    exact congrArg (insert 1) (clusterAt_congr_sched O sched pops hB (runOf harvests c) h)
  unfold ClusterSpec
  refine and_congr ?_ ?_
  · exact exists_congr fun B => and_congr_right fun hB => hret B hB
  · refine forall₂_congr fun B hB => ?_
    rw [hret B hB, hfam B hB]

theorem roundState_congr (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (harvests : List (Measure (FreeMonoid α))) {ω ω' : Ω} (c : ClusterRest α K)
    (h : ∀ w ∈ schedReads (settings.schedule (loopPopulations K harvests.length))
      (loopPopulations K harvests.length) (clusterRun harvests (ω, c)),
      (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    roundState settings O harvests (ω, c) = roundState settings O harvests (ω', c) := by
  have hp : (fun B => B ∈ settings.schedule (loopPopulations K harvests.length)
      ∧ clusterRun harvests (ω, c) ∈ ret O.mq (loopPopulations K harvests.length) none
        settings.indecisionLimit settings.gateLevel B)
      = (fun B => B ∈ settings.schedule (loopPopulations K harvests.length)
      ∧ clusterRun harvests (ω', c) ∈ ret O.mq (loopPopulations K harvests.length) none
        settings.indecisionLimit settings.gateLevel B) :=
    funext fun B => propext (and_congr_right fun hB =>
      ret_congr O _ _ none (Finset.mem_insert_self _ _) _ _ hB (runOf harvests c) h)
  show Classical.epsilon _ = Classical.epsilon _
  rw [hp]

theorem popCode_congr (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (hC : 2 ≤ settings.level4.maxLeaves) (D : Measure (FreeMonoid α)) (N : ℕ)
    (harvests : List (Measure (FreeMonoid α))) {ω ω' : Ω}
    (d : RoundDraws α K settings.level4.probeBudget)
    (hret : ∃ B ∈ settings.schedule (loopPopulations K harvests.length),
      clusterRun harvests (ω, d.1) ∈ ret O.mq (loopPopulations K harvests.length) none
        settings.indecisionLimit settings.gateLevel B)
    (h : ∀ w ∈ roundReads settings D N harvests ω d, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    popCode N (loopRound settings O D harvests (ω, d))
      = popCode N (loopRound settings O D harvests (ω', d)) := by
  classical
  set pops := loopPopulations K harvests.length
  set sched := settings.schedule pops
  have hs : ∀ w ∈ schedReads sched pops (clusterRun harvests (ω, d.1)),
      (O.mq w ω = 1 ↔ O.mq w ω' = 1) := fun w hw => h w (Finset.mem_union_left _ hw)
  have hst := roundState_congr settings O harvests d.1 hs
  have hmem : roundState settings O harvests (ω, d.1) ∈ sched :=
    (Classical.epsilon_spec (p := fun B => B ∈ sched ∧ clusterRun harvests (ω, d.1) ∈ ret O.mq pops
      none settings.indecisionLimit settings.gateLevel B) (by
        obtain ⟨B, hB, hr⟩ := hret; exact ⟨B, hB, hr⟩)).1
  set B := roundState settings O harvests (ω, d.1)
  have hF := clusterAt_congr_sched O sched pops hmem (runOf harvests d.1) hs
  set F := clusterAt O.mq pops (clusterRun harvests (ω, d.1)) B
  rw [loopRound_eq_mkRound, loopRound_eq_mkRound]
  simp only []
  rw [← hst]
  change popCode N (roundAt settings O _ B F (ω, _)) = popCode N (roundAt settings O _ B
    (clusterAt O.mq pops (clusterRun harvests (ω', d.1)) B) (ω', _))
  rw [← show F = clusterAt O.mq pops (clusterRun harvests (ω', d.1)) B from hF]
  have hroundKeys : roundKeys settings harvests (ω', d.1) = roundKeys settings harvests (ω, d.1) :=
    rfl
  rw [hroundKeys]
  set xs := probesOf settings D (roundKeys settings harvests (ω, d.1)) d.2
  -- the family is among the pool's suffixes
  have hFpool : F ⊆ (poolAt (schedNS sched) (clusterRun harvests (ω, d.1))).erase 1 := by
    intro v hv
    have hns : B.nsuff ≤ schedNS sched := Finset.le_sup (f := State.nsuff) hmem
    have hv1 : v ≠ 1 := (Finset.mem_erase.1 hv).1
    unfold F at hv
    rw [clusterAt_eq, clusterAt_eq_clusterOf] at hv
    exact Finset.mem_erase.2 ⟨hv1, poolAt_mono hns _ (clusterOf_subset O B.sc B.scd _ _ B.k ω
      (Finset.mem_insert_self _ _) hv)⟩
  have hsub : ∀ M, M ≤ max N (probesLen xs + 1 + settings.level4.maxLeaves) →
      readStrings (shortStrings M) F ⊆ roundReads settings D N harvests ω d := fun M hM =>
    ((readStrings_mono (shortStrings_mono hM) hFpool)).trans (by
      intro w hw
      exact Finset.mem_union_right _ hw)
  have h1 : ∀ p ∈ readStrings (shortStrings (probesLen xs + 1 + settings.level4.maxLeaves)) F,
      (O.mq p ω = 1 ↔ O.mq p ω' = 1) := fun p hp => h p (hsub _ (le_max_right _ _) hp)
  have h2 : ∀ p ∈ readStrings (shortStrings N) F, (O.mq p ω = 1 ↔ O.mq p ω' = 1) :=
    fun p hp => h p (hsub _ (le_max_left _ _) hp)
  have hres := level4_result_congr O settings.level4 hC F B.lo (B.hi + 1) xs h1
  have hgraph : readGraph N (fun z => familyRead O.mq F B.lo (B.hi + 1) z ω)
      = readGraph N (fun z => familyRead O.mq F B.lo (B.hi + 1) z ω') := by
    unfold readGraph
    refine Finset.image_congr fun z hz => ?_
    simp only [Prod.mk.injEq, true_and]
    exact familyRead_congr O h2 hz
  unfold popCode roundAt
  dsimp only
  rw [hres, hgraph]

end Round

section History

variable [IsProbabilityMeasure μ] (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
  (D : Measure (FreeMonoid α)) (N : ℕ)

/-- The earlier rounds' harvest data, at the noise `ω`. -/
def histCodes (ω : Ω) (g : Fin K → RoundDraws α K settings.level4.probeBudget) (r : ℕ) :
    List (HarvestCode α) :=
  codesOf N (loopRounds settings O D (fun i => (ω, g i)) r)

/-- Every earlier round's clustering, at the noise `ω`, returns at a state of its schedule. -/
def HistOK (ω : Ω) (g : Fin K → RoundDraws α K settings.level4.probeBudget) : ℕ → Prop
  | 0 => True
  | r + 1 => HistOK ω g r ∧ ∀ h : r < K, (histCodes settings O D N ω g r).length = r →
      ∃ B ∈ settings.schedule (loopPopulations K ((histCodes settings O D N ω g r).map
          (decodePop D settings.level4.walkStart)).length),
        clusterRun ((histCodes settings O D N ω g r).map (decodePop D settings.level4.walkStart))
          (ω, (g ⟨r, h⟩).1) ∈ ret O.mq (loopPopulations K ((histCodes settings O D N ω g r).map
            (decodePop D settings.level4.walkStart)).length) none settings.indecisionLimit
            settings.gateLevel B

/-- Every string the rounds before `r` read, given their harvest data `c`. -/
def histReads (c : List (HarvestCode α)) (g : Fin K → RoundDraws α K settings.level4.probeBudget)
    (r : ℕ) (ω : Ω) : Finset (FreeMonoid α) :=
  (Finset.range r).biUnion fun i => if h : i < K ∧ i ≤ c.length then
    roundReads settings D N ((c.take i).map (decodePop D settings.level4.walkStart)) ω
      (g ⟨i, h.1⟩) else ∅

omit [IsProbabilityMeasure μ] in
lemma histReads_mono {c c' : List (HarvestCode α)} {r r' : ℕ}
    (g : Fin K → RoundDraws α K settings.level4.probeBudget) (ω : Ω) (hr : r ≤ r')
    (hc : c <+: c') (hlen : ∀ i < r, i ≤ c.length → i ≤ c'.length) :
    histReads settings D N c g r ω ⊆ histReads settings D N c' g r' ω := by
  intro w hw
  obtain ⟨i, hi, hw⟩ := Finset.mem_biUnion.1 hw
  have hir := Finset.mem_range.1 hi
  split_ifs at hw with h
  · refine Finset.mem_biUnion.2 ⟨i, Finset.mem_range.2 (by omega), ?_⟩
    rw [dif_pos ⟨h.1, hlen i hir h.2⟩]
    obtain ⟨l, rfl⟩ := hc
    rw [List.take_append_of_le_length h.2]
    exact hw
  · exact absurd hw (Finset.notMem_empty _)

/-- The earlier rounds are fixed by the answers at the strings they read. -/
theorem hist_congr {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hC : 2 ≤ settings.level4.maxLeaves) (hN : L + settings.level4.maxLeaves ≤ N)
    (g : Fin K → RoundDraws α K settings.level4.probeBudget) :
    ∀ (r : ℕ) (ω ω' : Ω), HistOK settings O D N ω g r →
      (∀ w ∈ histReads settings D N (histCodes settings O D N ω g r) g r ω,
        (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
      HistOK settings O D N ω' g r ∧ histCodes settings O D N ω' g r = histCodes settings O D N ω g r
  | 0, ω, ω', _, _ => ⟨trivial, rfl⟩
  | r + 1, ω, ω', hok, h => by
    obtain ⟨hok', hret⟩ := hok
    set dec := decodePop D settings.level4.walkStart
    set c := histCodes settings O D N ω g r
    set y := histCodes settings O D N ω g (r + 1)
    -- the earlier codes extend to the later
    have hpre : c <+: y := by
      simp only [c, y, histCodes]
      rw [loopRounds_succ]
      split_ifs
      · rw [codesOf_append_single]; exact List.prefix_append _ _
      · exact List.prefix_refl _
    have hlenle : c.length ≤ y.length := hpre.length_le
    obtain ⟨hokω', hc'⟩ := hist_congr hlen hC hN g r ω ω' hok' fun w hw => h w
      ((histReads_mono settings D N g ω (Nat.le_succ r) hpre fun i _ hi => hi.trans hlenle) hw)
    by_cases hrK : r < K
    · have hy : y = if c.length = r then c ++ (popCode N (loopRound settings O D (c.map dec)
          (ω, g ⟨r, hrK⟩))).toList else c :=
        codesOf_loopRounds_succ settings O D hlen hC hN (fun i => (ω, g i)) r hrK
      have hy' : histCodes settings O D N ω' g (r + 1) = if c.length = r then
          c ++ (popCode N (loopRound settings O D (c.map dec) (ω', g ⟨r, hrK⟩))).toList else c := by
        have := codesOf_loopRounds_succ settings O D hlen hC hN (fun i => (ω', g i)) r hrK
        rw [show codesOf N (loopRounds settings O D (fun i => (ω', g i)) r) = c from hc'] at this
        exact this
      by_cases hcl : c.length = r
      · have hr := hret hrK hcl
        -- round `r`'s reads are among the history's
        have hround : ∀ w ∈ roundReads settings D N (c.map dec) ω (g ⟨r, hrK⟩),
            (O.mq w ω = 1 ↔ O.mq w ω' = 1) := by
          intro w hw
          refine h w (Finset.mem_biUnion.2 ⟨r, Finset.mem_range.2 (Nat.lt_succ_self r), ?_⟩)
          have hry : r ≤ y.length := hcl ▸ hlenle
          have hyt : y.take r = c := by
            obtain ⟨l, hl⟩ := hpre
            rw [← hl, List.take_append_of_le_length (le_of_eq hcl.symm),
              List.take_of_length_le (le_of_eq hcl)]
          simp only [dif_pos (show r < K ∧ r ≤ y.length from ⟨hrK, hry⟩)]
          rw [hyt]
          exact hw
        have hpc := popCode_congr settings O hC D N (c.map dec) (g ⟨r, hrK⟩) hr hround
        refine ⟨⟨hokω', fun h' hl' => ?_⟩, ?_⟩
        · rw [hc'] at hl' ⊢
          obtain ⟨B, hB, hrB⟩ := hr
          refine ⟨B, hB, ?_⟩
          have hs : ∀ w ∈ schedReads (settings.schedule (loopPopulations K (c.map dec).length))
              (loopPopulations K (c.map dec).length) (clusterRun (c.map dec) (ω, (g ⟨r, hrK⟩).1)),
              (O.mq w ω = 1 ↔ O.mq w ω' = 1) := fun w hw => hround w (Finset.mem_union_left _ hw)
          exact (ret_congr O _ _ none (Finset.mem_insert_self _ _) _ _ hB
            (runOf (c.map dec) (g ⟨r, hrK⟩).1) hs).1 hrB
        · rw [hy', hy, if_pos hcl, if_pos hcl, hpc]
      · refine ⟨⟨hokω', fun h' hl' => by rw [hc'] at hl'; exact absurd hl' hcl⟩, ?_⟩
        rw [hy', hy, if_neg hcl, if_neg hcl]
    · have hst : ∀ ω'' : Ω, histCodes settings O D N ω'' g (r + 1)
          = histCodes settings O D N ω'' g r := by
        intro ω''
        simp only [histCodes]
        rw [loopRounds_succ, dif_neg (fun h' => hrK h'.1)]
      refine ⟨⟨hokω', fun h' => absurd h' hrK⟩, ?_⟩
      rw [hst ω']
      simp only [y]
      rw [hst ω]
      exact hc'

end History

end OrthoDFA
