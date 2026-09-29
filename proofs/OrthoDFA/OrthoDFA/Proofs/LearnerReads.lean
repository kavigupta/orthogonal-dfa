import OrthoDFA.Learner
import OrthoDFA.Proofs.LearnerCells

/-!
# What each round of the learner reads

A round reads the noise at the clustering's strings, which its draws fix, then at whatever the
stage asks for, then at the check's strings, which its draws fix again.  So the history of the
first `r` rounds is decided by the noise at `readsUpTo r`, and that set is decided the same way.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

namespace LearnerProof

/-! ## The clustering's reads -/

section Clustering

variable {J : Type*} [Fintype J]

lemma readSet_mono {P P' C C' : Finset S} (hP : P ⊆ P') (hC : C ⊆ C') :
    readSet P C ⊆ readSet P' C' := by
  classical
  intro w hw
  obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hw
  obtain ⟨hp, hv⟩ := Finset.mem_product.1 hpv
  exact mem_readSet (hP hp) (hC hv)

lemma card_readSet_le (P C : Finset S) : (readSet P C).card ≤ P.card * C.card := by
  classical
  unfold readSet
  exact Finset.card_image_le.trans (Finset.card_product P C).le

lemma mq_iff_of_noise (O : Oracle μ S) {w : S} {ω ω' : Ω} (h : O.noise w ω = O.noise w ω') :
    (O.mq w ω = 1 ↔ O.mq w ω' = 1) := by rw [mq_congr O h]

set_option linter.unusedFintypeInType false in
/-- The gate reads the certification prefixes at the family's strings and at themselves. -/
lemma ret_congr (O : Oracle μ S) (pops : Finset J) (il α : ℝ) (B : State)
    (d : ((ℕ → S) × (J → ℕ → S)) × (J → ℕ → S)) {ω ω' : Ω}
    (h : ∀ w ∈ readSet (prefixesAt pops B.npref ((ω, d) : Run Ω S J)
        ∪ pops.biUnion (fun j => certOf j B.npref ((ω, d) : Run Ω S J)))
        (poolAt B.nsuff ((ω, d) : Run Ω S J)), O.noise w ω = O.noise w ω') :
    ((ω, d) : Run Ω S J) ∈ ret O.mq pops il α B ↔ ((ω', d) : Run Ω S J) ∈ ret O.mq pops il α B := by
  classical
  have hF : clusterAt O.mq pops ((ω', d) : Run Ω S J) B = clusterAt O.mq pops (ω, d) B :=
    (clusterAt_congr O pops B d fun w hw =>
      h w (readSet_mono Finset.subset_union_left subset_rfl hw)).symm
  set F := clusterAt O.mq pops ((ω, d) : Run Ω S J) B
  have hFpool : F ⊆ poolAt B.nsuff ((ω, d) : Run Ω S J) := clusterAt_subset O pops B _
  have hcert : ∀ j ∈ pops, ∀ p ∈ certOf j B.npref ((ω, d) : Run Ω S J), ∀ v ∈ poolAt B.nsuff
      ((ω, d) : Run Ω S J), (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1) := fun j hj p hp v hv =>
    mq_iff_of_noise O (h _ (mem_readSet (Finset.mem_union_right _
      (Finset.mem_biUnion.2 ⟨j, hj, hp⟩)) hv))
  have hvote : ∀ j ∈ pops, ∀ p ∈ certOf j B.npref ((ω, d) : Run Ω S J),
      voteCount O.mq (F.erase 1) p ω = voteCount O.mq (F.erase 1) p ω' := fun j hj p hp =>
    voteCount_congr O _ p fun v hv => hcert j hj p hp v (hFpool (Finset.mem_of_mem_erase hv))
  have hself : ∀ j ∈ pops, ∀ p ∈ certOf j B.npref ((ω, d) : Run Ω S J),
      (O.mq p ω = 1 ↔ O.mq p ω' = 1) := fun j hj p hp => by
    simpa using hcert j hj p hp 1 (one_mem_poolAt _ _)
  have hdec : ∀ j ∈ pops, (certOf j B.npref ((ω, d) : Run Ω S J)).filter (fun p =>
        ¬ decided O.mq B.lo (B.hi - 1) (F.erase 1) p ω)
      = (certOf j B.npref ((ω, d) : Run Ω S J)).filter (fun p =>
        ¬ decided O.mq B.lo (B.hi - 1) (F.erase 1) p ω') := fun j hj =>
    Finset.filter_congr fun p hp => by unfold decided; rw [hvote j hj p hp]
  have hagree : ∀ j ∈ pops, agreeCount O.mq B.lo B.hi (F.erase 1)
        (certOf j B.npref ((ω, d) : Run Ω S J)) ω
      = agreeCount O.mq B.lo B.hi (F.erase 1) (certOf j B.npref ((ω, d) : Run Ω S J)) ω' := by
    intro j hj
    have hsides : cutSides O.mq B.lo B.hi (F.erase 1) (certOf j B.npref ((ω, d) : Run Ω S J)) ω
        = cutSides O.mq B.lo B.hi (F.erase 1) (certOf j B.npref ((ω, d) : Run Ω S J)) ω' := by
      unfold cutSides
      rw [Finset.filter_congr fun p hp => by rw [hvote j hj p hp],
        Finset.filter_congr (s := certOf j B.npref ((ω, d) : Run Ω S J))
          (p := fun p => B.hi - 1 < voteCount O.mq (F.erase 1) p ω
            ∨ voteCount O.mq (F.erase 1) p ω ≤ B.lo) fun p hp => by rw [hvote j hj p hp]]
    unfold agreeCount
    rw [hsides, Finset.filter_congr fun p hp => hself j hj p hp]
  change _ ∧ _ ∧ _ ↔ _ ∧ _ ∧ _
  rw [hF]
  refine and_congr Iff.rfl (and_congr ?_ ?_)
  · refine forall₂_congr fun j hj => ?_
    change (((certOf j B.npref ((ω, d) : Run Ω S J)).filter (fun p =>
        ¬ decided O.mq B.lo (B.hi - 1) (F.erase 1) p ω)).card : ℝ) ≤ _ ↔ _
    rw [hdec j hj]
    rfl
  · refine forall₂_congr fun j hj => ?_
    change admitted O.mq B.lo B.hi B.gmin α (F.erase 1) (certOf j B.npref ((ω, d) : Run Ω S J)) ω
      ↔ admitted O.mq B.lo B.hi B.gmin α (F.erase 1) (certOf j B.npref ((ω, d) : Run Ω S J)) ω'
    unfold admitted
    rw [hagree j hj]

end Clustering

/-! ## The learner's rounds -/

section Learner

variable {K : ℕ} {R Xs : Type*} [Fintype R]

/-- The clustering's part of a round's draws: suffixes, then prefixes and certification
prefixes per population. -/
abbrev ClusterPart (S : Type*) (K : ℕ) (R : Type*) (M : ℕ) :=
  (Fin M → S) × ((Option (Fin K × R) → Fin M → S) × (Option (Fin K × R) → Fin M → S))

noncomputable def clusterDraws {M : ℕ} (past : List (Outcome S R))
    (e : ClusterPart S K R M) : ClusterDraws S K R :=
  ((padded e.1, fun j => populationDraws past j (e.2.1 j)),
    fun j => populationDraws past j (e.2.2 j))

noncomputable def roundState (O : Oracle μ S) (Dsamp : Measure S) (heavy : ℝ)
    (schedule : Finset (Option (Fin K × R)) → Finset State) (indecisionLimit α : ℝ) {M : ℕ}
    (past : List (Outcome S R)) (ω : Ω) (e : ClusterPart S K R M) : State :=
  Classical.epsilon fun B => B ∈ schedule (populationsAfter Dsamp heavy past)
    ∧ ((ω, clusterDraws past e) : Run Ω S (Option (Fin K × R)))
      ∈ ret O.mq (populationsAfter Dsamp heavy past) indecisionLimit α B

noncomputable def roundFamily (O : Oracle μ S) (Dsamp : Measure S) (heavy : ℝ)
    (schedule : Finset (Option (Fin K × R)) → Finset State) (indecisionLimit α : ℝ) {M : ℕ}
    (past : List (Outcome S R)) (ω : Ω) (e : ClusterPart S K R M) : Finset S :=
  clusterAt O.mq (populationsAfter Dsamp heavy past)
    ((ω, clusterDraws past e) : Run Ω S (Option (Fin K × R)))
    (roundState O Dsamp heavy schedule indecisionLimit α past ω e)

noncomputable def roundHyp (O : Oracle μ S) (Dsamp : Measure S)
    (schedule : Finset (Option (Fin K × R)) → Finset State) (indecisionLimit α : ℝ)
    (L : Learner Ω S R Xs K) (past : List (Outcome S R)) (ω : Ω)
    (e : ClusterPart S K R L.M) (s : Xs) : DFA S R :=
  L.stage (roundFamily O Dsamp L.heavy schedule indecisionLimit α past ω e)
    (roundState O Dsamp L.heavy schedule indecisionLimit α past ω e) (clusterDraws past e) ω s

open scoped Classical in
/-- Whether round `past.length` gets past the gate on the draws `g`. -/
noncomputable def roundGate (O : Oracle μ S) (Dsamp : Measure S)
    (schedule : Finset (Option (Fin K × R)) → Finset State) (indecisionLimit α : ℝ)
    (L : Learner Ω S R Xs K) (past : List (Outcome S R)) (ω : Ω)
    (e : ClusterPart S K R L.M) (s : Xs) (g : Fin L.G → S) : Prop :=
  (Finset.univ.filter fun i => ¬ cutAgrees O
    (roundState O Dsamp L.heavy schedule indecisionLimit α past ω e)
    (roundFamily O Dsamp L.heavy schedule indecisionLimit α past ω e)
    (roundHyp O Dsamp schedule indecisionLimit α L past ω e s) (g i) ω).card ≤ L.gth

open scoped Classical in
omit [IsProbabilityMeasure μ] in
lemma round_eq (O : Oracle μ S) (Dsamp : Measure S)
    (schedule : Finset (Option (Fin K × R)) → Finset State) (indecisionLimit α : ℝ)
    (L : Learner Ω S R Xs K) (past : List (Outcome S R)) (ω : Ω)
    (d : RoundDraws S K R Xs L.M L.G L.N L.m) :
    round O Dsamp schedule indecisionLimit α L past ω d
      = (roundHyp O Dsamp schedule indecisionLimit α L past ω (d.1, d.2.1) d.2.2.1,
        roundGate O Dsamp schedule indecisionLimit α L past ω (d.1, d.2.1) d.2.2.1 d.2.2.2.1,
        Finset.univ.filter fun h => L.heavy / Fintype.card R
            ≤ Dsamp.real {v | (roundHyp O Dsamp schedule indecisionLimit α L past ω (d.1, d.2.1)
              d.2.2.1).state v = h}
          ∧ checkFails O (roundHyp O Dsamp schedule indecisionLimit α L past ω (d.1, d.2.1)
            d.2.2.1) L.n L.t h d.2.2.2.2 ω) := by
  unfold round
  congr

/-! ### The strings a round reads -/

lemma padded_of_le {M : ℕ} (a : Fin M → S) {i : ℕ} (hi : M ≤ i) : padded a i = 1 := by
  simp [padded, not_lt.2 hi]

omit [Fintype R] in
lemma hitsIn_of_le (H : DFA S R) (h : R) {M : ℕ} (a : Fin M → S) {i : ℕ} (hi : M ≤ i) :
    hitsIn H h a i = 1 := by
  classical
  unfold hitsIn
  refine List.getD_eq_default _ _ ?_
  exact (List.length_filter_le _ _).trans (by simpa using hi)

omit [Fintype R] in
lemma populationDraws_of_le (past : List (Outcome S R)) (j : Option (Fin K × R))
    {M : ℕ} (a : Fin M → S) {i : ℕ} (hi : M ≤ i) : populationDraws past j a i = 1 := by
  unfold populationDraws
  rcases j with _ | ⟨i', h⟩
  · exact padded_of_le a hi
  · simp only
    cases past[i'.val]? with
    | none => exact padded_of_le a hi
    | some o => exact hitsIn_of_le _ _ a hi

/-- Every population's first `M` prefixes and certification prefixes. -/
noncomputable def entries (Dsamp : Measure S) (heavy : ℝ) {M : ℕ} (past : List (Outcome S R))
    (e : ClusterPart S K R M) : Finset S :=
  (populationsAfter Dsamp heavy past).biUnion fun j =>
    (Finset.range M).image (populationDraws past j (e.2.1 j))
      ∪ (Finset.range M).image (populationDraws past j (e.2.2 j))

/-- The pool is the first `M` suffixes; a draw beyond them, of any population, is `ε`. -/
noncomputable def clusterReads (Dsamp : Measure S) (heavy : ℝ) {M : ℕ}
    (past : List (Outcome S R)) (e : ClusterPart S K R M) : Finset S :=
  readSet (insert 1 (entries Dsamp heavy past e)) (insert 1 ((Finset.range M).image (padded e.1)))

noncomputable def checkReads {N m : ℕ} (x : (Fin N → S) × (Fin m → S)) : Finset S :=
  readSet (Finset.univ.image x.1) (insert 1 (Finset.univ.image x.2))

/-- The gate reads its draws at the pool's suffixes, which hold the family. -/
noncomputable def gateReads {M G : ℕ} (e : Fin M → S) (g : Fin G → S) : Finset S :=
  readSet (Finset.univ.image g) (insert 1 ((Finset.range M).image (padded e)))

omit [MeasurableSpace Ω] in
lemma prefixesAt_subset_entries (Dsamp : Measure S) (heavy : ℝ) {M : ℕ} (past : List (Outcome S R))
    (e : ClusterPart S K R M) (ω : Ω) (n : ℕ) :
    prefixesAt (populationsAfter Dsamp heavy past) n
        ((ω, clusterDraws past e) : Run Ω S (Option (Fin K × R)))
      ⊆ insert 1 (entries Dsamp heavy past e) := by
  classical
  intro p hp
  obtain ⟨j, hj, hp⟩ := Finset.mem_biUnion.1 hp
  obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hp
  by_cases hi : i < M
  · exact Finset.mem_insert_of_mem (Finset.mem_biUnion.2 ⟨j, hj, Finset.mem_union_left _
      (Finset.mem_image.2 ⟨i, Finset.mem_range.2 hi, rfl⟩)⟩)
  · rw [show prefixDraw j i ((ω, clusterDraws past e) : Run Ω S (Option (Fin K × R)))
      = populationDraws past j (e.2.1 j) i from rfl, populationDraws_of_le _ _ _ (not_lt.1 hi)]
    exact Finset.mem_insert_self _ _

omit [MeasurableSpace Ω] in
lemma certs_subset_entries (Dsamp : Measure S) (heavy : ℝ) {M : ℕ} (past : List (Outcome S R))
    (e : ClusterPart S K R M) (ω : Ω) (n : ℕ) :
    (populationsAfter Dsamp heavy past).biUnion (fun j => certOf j n
        ((ω, clusterDraws past e) : Run Ω S (Option (Fin K × R))))
      ⊆ insert 1 (entries Dsamp heavy past e) := by
  classical
  intro p hp
  obtain ⟨j, hj, hp⟩ := Finset.mem_biUnion.1 hp
  obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hp
  by_cases hi : i < M
  · exact Finset.mem_insert_of_mem (Finset.mem_biUnion.2 ⟨j, hj, Finset.mem_union_right _
      (Finset.mem_image.2 ⟨i, Finset.mem_range.2 hi, rfl⟩)⟩)
  · rw [show certPrefix j i ((ω, clusterDraws past e) : Run Ω S (Option (Fin K × R)))
      = populationDraws past j (e.2.2 j) i from rfl, populationDraws_of_le _ _ _ (not_lt.1 hi)]
    exact Finset.mem_insert_self _ _

omit [MeasurableSpace Ω] [Fintype R] in
lemma poolAt_subset_clusterPool {M : ℕ} (past : List (Outcome S R))
    (e : ClusterPart S K R M) (ω : Ω) (n : ℕ) :
    poolAt n ((ω, clusterDraws past e) : Run Ω S (Option (Fin K × R)))
      ⊆ insert 1 ((Finset.range M).image (padded e.1)) := by
  classical
  intro v hv
  rcases Finset.mem_insert.1 hv with rfl | hv
  · exact Finset.mem_insert_self _ _
  obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hv
  by_cases hi : i < M
  · exact Finset.mem_insert_of_mem (Finset.mem_image.2 ⟨i, Finset.mem_range.2 hi, rfl⟩)
  · rw [show suffixDraw i ((ω, clusterDraws past e) : Run Ω S (Option (Fin K × R)))
      = padded e.1 i from rfl, padded_of_le _ (not_lt.1 hi)]
    exact Finset.mem_insert_self _ _

/-- At any state, the gate and the family read only `clusterReads`. -/
lemma cluster_congr {M : ℕ} (O : Oracle μ S) (Dsamp : Measure S) (heavy : ℝ)
    (indecisionLimit α : ℝ)
    (past : List (Outcome S R)) (e : ClusterPart S K R M) {ω ω' : Ω}
    (h : ∀ w ∈ clusterReads Dsamp heavy past e, O.noise w ω = O.noise w ω') (B : State) :
    (((ω, clusterDraws past e) : Run Ω S (Option (Fin K × R)))
        ∈ ret O.mq (populationsAfter Dsamp heavy past) indecisionLimit α B
      ↔ ((ω', clusterDraws past e) : Run Ω S (Option (Fin K × R)))
        ∈ ret O.mq (populationsAfter Dsamp heavy past) indecisionLimit α B)
    ∧ clusterAt O.mq (populationsAfter Dsamp heavy past) ((ω, clusterDraws past e) : Run Ω S _) B
      = clusterAt O.mq (populationsAfter Dsamp heavy past)
        ((ω', clusterDraws past e) : Run Ω S _) B := by
  classical
  have hsub : ∀ n n', readSet (prefixesAt (populationsAfter Dsamp heavy past) n
        ((ω, clusterDraws past e) : Run Ω S (Option (Fin K × R)))
        ∪ (populationsAfter Dsamp heavy past).biUnion (fun j => certOf j n
          ((ω, clusterDraws past e) : Run Ω S (Option (Fin K × R)))))
        (poolAt n' ((ω, clusterDraws past e) : Run Ω S (Option (Fin K × R))))
      ⊆ clusterReads Dsamp heavy past e := fun n n' =>
    readSet_mono (Finset.union_subset (prefixesAt_subset_entries Dsamp heavy past e ω n)
      (certs_subset_entries Dsamp heavy past e ω n)) (poolAt_subset_clusterPool past e ω n')
  refine ⟨ret_congr O _ indecisionLimit α B _ fun w hw => h w (hsub _ _ hw), ?_⟩
  exact clusterAt_congr O _ B _ fun w hw =>
    h w (hsub _ _ (readSet_mono Finset.subset_union_left subset_rfl hw))

lemma roundState_congr {M : ℕ} (O : Oracle μ S) (Dsamp : Measure S) (heavy : ℝ)
    (schedule : Finset (Option (Fin K × R)) → Finset State) (indecisionLimit α : ℝ)
    (past : List (Outcome S R)) (e : ClusterPart S K R M) {ω ω' : Ω}
    (h : ∀ w ∈ clusterReads Dsamp heavy past e, O.noise w ω = O.noise w ω') :
    roundState O Dsamp heavy schedule indecisionLimit α past ω' e
      = roundState O Dsamp heavy schedule indecisionLimit α past ω e := by
  unfold roundState
  congr 1
  funext B
  exact propext (and_congr Iff.rfl
    (cluster_congr O Dsamp heavy indecisionLimit α past e h B).1.symm)

lemma roundFamily_congr {M : ℕ} (O : Oracle μ S) (Dsamp : Measure S) (heavy : ℝ)
    (schedule : Finset (Option (Fin K × R)) → Finset State) (indecisionLimit α : ℝ)
    (past : List (Outcome S R)) (e : ClusterPart S K R M) {ω ω' : Ω}
    (h : ∀ w ∈ clusterReads Dsamp heavy past e, O.noise w ω = O.noise w ω') :
    roundFamily O Dsamp heavy schedule indecisionLimit α past ω' e
      = roundFamily O Dsamp heavy schedule indecisionLimit α past ω e := by
  unfold roundFamily
  rw [roundState_congr O Dsamp heavy schedule indecisionLimit α past e h]
  exact (cluster_congr O Dsamp heavy indecisionLimit α past e h _).2.symm

lemma leaning_congr (O : Oracle μ S) (n : ℕ) (members : List S) (v : S) {ω ω' : Ω}
    (h : ∀ k < 2 * n, O.noise (members.getD k 1 * v) ω = O.noise (members.getD k 1 * v) ω'
      ∧ O.noise (members.getD k 1) ω = O.noise (members.getD k 1) ω') :
    leaning O n members v ω = leaning O n members v ω' := by
  unfold leaning
  refine Finset.sum_congr rfl fun k hk => ?_
  have hk := Finset.mem_range.1 hk
  obtain ⟨h1, h2⟩ := h (2 * k) (by omega)
  obtain ⟨h3, h4⟩ := h (2 * k + 1) (by omega)
  rw [mq_congr O h1, mq_congr O h2, mq_congr O h3, mq_congr O h4]

omit [Fintype R] in
/-- The check reads its members at themselves and at its suffixes. -/
lemma checkFails_congr (O : Oracle μ S) (H : DFA S R) (n : ℕ) (t : ℝ) {N m : ℕ} (h : R)
    (x : (Fin N → S) × (Fin m → S)) {ω ω' : Ω}
    (hω : ∀ w ∈ checkReads x, O.noise w ω = O.noise w ω') :
    checkFails O H n t h x ω ↔ checkFails O H n t h x ω' := by
  classical
  have hmem : ∀ k < (membersOf H h x.1).length, (membersOf H h x.1).getD k 1
      ∈ Finset.univ.image x.1 := by
    intro k hk
    rw [List.getD_eq_getElem _ _ hk]
    have := List.mem_of_mem_filter (List.getElem_mem hk)
    obtain ⟨i, hi⟩ := List.mem_ofFn.1 this
    rw [← hi]
    exact Finset.mem_image_of_mem _ (Finset.mem_univ i)
  have hlean : 2 * n ≤ (membersOf H h x.1).length → ∀ j,
      leaning O n (membersOf H h x.1) (x.2 j) ω = leaning O n (membersOf H h x.1) (x.2 j) ω' :=
    fun hlen j => leaning_congr O n _ _ fun k hk =>
      ⟨hω _ (mem_readSet (hmem k (by omega)) (Finset.mem_insert_of_mem
          (Finset.mem_image_of_mem _ (Finset.mem_univ j)))),
        by simpa using hω _ (mem_readSet (hmem k (by omega)) (Finset.mem_insert_self 1 _))⟩
  unfold checkFails
  constructor
  · rintro ⟨hlen, j, hj, hl⟩
    exact ⟨hlen, j, hj, (hlean hlen j) ▸ hl⟩
  · rintro ⟨hlen, j, hj, hl⟩
    exact ⟨hlen, j, hj, (hlean hlen j).symm ▸ hl⟩

variable (O : Oracle μ S) (Dsamp : Measure S)
  (schedule : Finset (Option (Fin K × R)) → Finset State) (indecisionLimit α : ℝ)
  (L : Learner Ω S R Xs K) (stageReads : Finset S → State → ClusterDraws S K R → Ω → Xs → Finset S)

noncomputable def roundReads (past : List (Outcome S R)) (ω : Ω)
    (d : RoundDraws S K R Xs L.M L.G L.N L.m) : Finset S :=
  clusterReads Dsamp L.heavy past (d.1, d.2.1)
    ∪ stageReads (roundFamily O Dsamp L.heavy schedule indecisionLimit α past ω (d.1, d.2.1))
      (roundState O Dsamp L.heavy schedule indecisionLimit α past ω (d.1, d.2.1))
      (clusterDraws past (d.1, d.2.1)) ω d.2.2.1
    ∪ gateReads d.1 d.2.2.2.1
    ∪ checkReads d.2.2.2.2

variable {O Dsamp schedule indecisionLimit α L stageReads} in
lemma round_congr
    (hreads : ∀ F B c s, ReadsOnly O (fun ω => stageReads F B c ω s) (fun ω => L.stage F B c ω s))
    (past : List (Outcome S R)) (d : RoundDraws S K R Xs L.M L.G L.N L.m) {ω ω' : Ω}
    (h : ∀ w ∈ roundReads O Dsamp schedule indecisionLimit α L stageReads past ω d,
      O.noise w ω = O.noise w ω') :
    roundReads O Dsamp schedule indecisionLimit α L stageReads past ω' d
        = roundReads O Dsamp schedule indecisionLimit α L stageReads past ω d
      ∧ round O Dsamp schedule indecisionLimit α L past ω' d
        = round O Dsamp schedule indecisionLimit α L past ω d := by
  classical
  have hc : ∀ w ∈ clusterReads Dsamp L.heavy past (d.1, d.2.1), O.noise w ω = O.noise w ω' :=
    fun w hw => h w (Finset.mem_union_left _ (Finset.mem_union_left _
      (Finset.mem_union_left _ hw)))
  have hB := roundState_congr O Dsamp L.heavy schedule indecisionLimit α past _ hc
  have hF := roundFamily_congr O Dsamp L.heavy schedule indecisionLimit α past _ hc
  have hs := hreads (roundFamily O Dsamp L.heavy schedule indecisionLimit α past ω (d.1, d.2.1))
    (roundState O Dsamp L.heavy schedule indecisionLimit α past ω (d.1, d.2.1))
    (clusterDraws past (d.1, d.2.1)) d.2.2.1 ω ω' fun w hw =>
      h w (Finset.mem_union_left _ (Finset.mem_union_left _ (Finset.mem_union_right _ hw)))
  have hH : roundHyp O Dsamp schedule indecisionLimit α L past ω' (d.1, d.2.1) d.2.2.1
      = roundHyp O Dsamp schedule indecisionLimit α L past ω (d.1, d.2.1) d.2.2.1 := by
    unfold roundHyp
    rw [hB, hF]
    exact hs.2
  have hs1 : stageReads (roundFamily O Dsamp L.heavy schedule indecisionLimit α past ω
        (d.1, d.2.1)) (roundState O Dsamp L.heavy schedule indecisionLimit α past ω (d.1, d.2.1))
        (clusterDraws past (d.1, d.2.1)) ω' d.2.2.1
      = stageReads (roundFamily O Dsamp L.heavy schedule indecisionLimit α past ω (d.1, d.2.1))
        (roundState O Dsamp L.heavy schedule indecisionLimit α past ω (d.1, d.2.1))
        (clusterDraws past (d.1, d.2.1)) ω d.2.2.1 := hs.1
  have hg : roundGate O Dsamp schedule indecisionLimit α L past ω' (d.1, d.2.1) d.2.2.1 d.2.2.2.1
      ↔ roundGate O Dsamp schedule indecisionLimit α L past ω (d.1, d.2.1) d.2.2.1 d.2.2.2.1 := by
    unfold roundGate
    rw [hB, hF, hH]
    have hsub := (clusterAt_subset O (populationsAfter Dsamp L.heavy past) _ _).trans
      (poolAt_subset_clusterPool past (d.1, d.2.1) ω
      (roundState O Dsamp L.heavy schedule indecisionLimit α past ω (d.1, d.2.1)).nsuff)
    have hv : ∀ i, voteCount O.mq (roundFamily O Dsamp L.heavy schedule indecisionLimit α past ω
          (d.1, d.2.1)) (d.2.2.2.1 i) ω'
        = voteCount O.mq (roundFamily O Dsamp L.heavy schedule indecisionLimit α past ω
          (d.1, d.2.1)) (d.2.2.2.1 i) ω := fun i =>
      (voteCount_congr O _ _ fun v hv => by
        rw [mq_congr O (h _ (Finset.mem_union_left _ (Finset.mem_union_right _
          (mem_readSet (Finset.mem_image_of_mem _ (Finset.mem_univ i)) (hsub hv)))))]).symm
    have hc : ∀ i, cutAgrees O (roundState O Dsamp L.heavy schedule indecisionLimit α past ω
          (d.1, d.2.1)) (roundFamily O Dsamp L.heavy schedule indecisionLimit α past ω
          (d.1, d.2.1)) (roundHyp O Dsamp schedule indecisionLimit α L past ω (d.1, d.2.1)
          d.2.2.1) (d.2.2.2.1 i) ω'
        ↔ cutAgrees O (roundState O Dsamp L.heavy schedule indecisionLimit α past ω
          (d.1, d.2.1)) (roundFamily O Dsamp L.heavy schedule indecisionLimit α past ω
          (d.1, d.2.1)) (roundHyp O Dsamp schedule indecisionLimit α L past ω (d.1, d.2.1)
          d.2.2.1) (d.2.2.2.1 i) ω := fun i => by
      unfold cutAgrees
      rw [hv i]
    exact iff_of_eq (congrArg (fun t => t.card ≤ L.gth)
      (Finset.filter_congr fun i _ => not_congr (hc i)))
  refine ⟨?_, ?_⟩
  · unfold roundReads
    rw [hB, hF, hs1]
  · rw [round_eq, round_eq, hH, propext hg]
    congr 2
    refine Finset.filter_congr fun r _ => and_congr Iff.rfl ?_
    exact (checkFails_congr O _ L.n L.t r d.2.2.2.2 fun w hw =>
      h w (Finset.mem_union_right _ hw)).symm

/-- The strings the first `r` rounds read. -/
noncomputable def readsUpTo (ω : Ω) (d : Fin K → RoundDraws S K R Xs L.M L.G L.N L.m) :
    ℕ → Finset S
  | 0 => ∅
  | r + 1 =>
    if h : r < K then
      readsUpTo ω d r ∪ roundReads O Dsamp schedule indecisionLimit α L stageReads
        (history O Dsamp schedule indecisionLimit α L ω d r) ω (d ⟨r, h⟩)
    else readsUpTo ω d r

variable {O Dsamp schedule indecisionLimit α L stageReads} in
/-- The first `r` rounds read the noise only at `readsUpTo r`. -/
lemma history_congr
    (hreads : ∀ F B c s, ReadsOnly O (fun ω => stageReads F B c ω s) (fun ω => L.stage F B c ω s))
    (d : Fin K → RoundDraws S K R Xs L.M L.G L.N L.m) (r : ℕ) {ω ω' : Ω}
    (h : ∀ w ∈ readsUpTo O Dsamp schedule indecisionLimit α L stageReads ω d r,
      O.noise w ω = O.noise w ω') :
    readsUpTo O Dsamp schedule indecisionLimit α L stageReads ω' d r
        = readsUpTo O Dsamp schedule indecisionLimit α L stageReads ω d r
      ∧ history O Dsamp schedule indecisionLimit α L ω' d r
        = history O Dsamp schedule indecisionLimit α L ω d r := by
  induction r with
  | zero => exact ⟨rfl, rfl⟩
  | succ r ih =>
    by_cases hr : r < K
    · simp only [readsUpTo, history, dif_pos hr] at h ⊢
      obtain ⟨h1, h2⟩ := ih fun w hw => h w (Finset.mem_union_left _ hw)
      rw [h1, h2]
      obtain ⟨h3, h4⟩ := round_congr (Dsamp := Dsamp) hreads
        (history O Dsamp schedule indecisionLimit α L ω d r) (d ⟨r, hr⟩)
        fun w hw => h w (Finset.mem_union_right _ hw)
      rw [h3, h4]
      exact ⟨rfl, rfl⟩
    · simp only [readsUpTo, history, dif_neg hr] at h ⊢
      exact ih h

omit [IsProbabilityMeasure μ] in
/-- The first `r` rounds use only the first `r` rounds' draws. -/
lemma history_dep (ω : Ω) (d d' : Fin K → RoundDraws S K R Xs L.M L.G L.N L.m) (r : ℕ)
    (h : ∀ i : Fin K, i.val < r → d i = d' i) :
    history O Dsamp schedule indecisionLimit α L ω d r
        = history O Dsamp schedule indecisionLimit α L ω d' r
      ∧ readsUpTo O Dsamp schedule indecisionLimit α L stageReads ω d r
        = readsUpTo O Dsamp schedule indecisionLimit α L stageReads ω d' r := by
  induction r with
  | zero => exact ⟨rfl, rfl⟩
  | succ r ih =>
    obtain ⟨h1, h2⟩ := ih fun i hi => h i (by omega)
    by_cases hr : r < K
    · simp only [readsUpTo, history, dif_pos hr]
      rw [h1, h2, h ⟨r, hr⟩ (by simp)]
      exact ⟨rfl, rfl⟩
    · simp only [readsUpTo, history, dif_neg hr]
      exact ⟨h1, h2⟩

omit [IsProbabilityMeasure μ] in
lemma length_history (ω : Ω) (d : Fin K → RoundDraws S K R Xs L.M L.G L.N L.m) (r : ℕ) :
    (history O Dsamp schedule indecisionLimit α L ω d r).length = min r K := by
  induction r with
  | zero => simp [history]
  | succ r ih =>
    by_cases hr : r < K
    · simp only [history, dif_pos hr, List.length_append, ih, List.length_singleton]
      omega
    · simp only [history, dif_neg hr, ih]
      omega

/-- Round `r`'s output, and nothing past the last round. -/
noncomputable def outAt (H0 : DFA S R) (r : ℕ) (ω : Ω)
    (d : Fin K → RoundDraws S K R Xs L.M L.G L.N L.m) : Outcome S R :=
  if h : r < K then
    round O Dsamp schedule indecisionLimit α L (history O Dsamp schedule indecisionLimit α L ω d r)
      ω (d ⟨r, h⟩)
  else (H0, True, ∅)

omit [IsProbabilityMeasure μ] in
lemma history_getElem? (H0 : DFA S R) (ω : Ω) (d : Fin K → RoundDraws S K R Xs L.M L.G L.N L.m)
    (r : ℕ) (hr : r ≤ K) (i : ℕ) :
    (history O Dsamp schedule indecisionLimit α L ω d r)[i]?
      = if i < r then some (outAt O Dsamp schedule indecisionLimit α L H0 i ω d) else none := by
  induction r generalizing i with
  | zero => simp [history]
  | succ r ih =>
    have hr' : r < K := hr
    have hlen := length_history O Dsamp schedule indecisionLimit α L ω d r
    rw [min_eq_left (by omega)] at hlen
    simp only [history, dif_pos hr']
    rw [List.getElem?_append]
    split_ifs with h1 h2 h2
    · rw [ih (by omega), if_pos (by omega)]
    · omega
    · have hi : i = r := by omega
      subst hi
      simp [hlen, outAt, dif_pos hr']
    · rw [List.getElem?_singleton]
      simp only [hlen]
      split_ifs <;> first | rfl | omega

/-! ### How many strings the rounds read -/

lemma card_fin_lt_le (n : ℕ) : (Finset.univ.filter fun i : Fin K => i.val < n).card ≤ n := by
  calc (Finset.univ.filter fun i : Fin K => i.val < n).card
      ≤ (Finset.range n).card := Finset.card_le_card_of_injOn Fin.val
        (fun i hi => by simpa using (Finset.mem_filter.1 hi).2)
        (fun i _ j _ hij => Fin.ext hij)
    _ = n := Finset.card_range n

omit [Fintype R] in
lemma card_populationsAfter_le [Fintype R] (heavy : ℝ) (past : List (Outcome S R)) :
    (populationsAfter (K := K) Dsamp heavy past).card ≤ 1 + past.length * Fintype.card R := by
  classical
  have hsub : populationsAfter (K := K) Dsamp heavy past ⊆ insert none (Finset.image some
      ((Finset.univ.filter fun i : Fin K => i.val < past.length) ×ˢ (Finset.univ : Finset R))) := by
    intro j hj
    rcases j with _ | ⟨i, h⟩
    · exact Finset.mem_insert_self _ _
    · obtain ⟨o, ho, -⟩ := (Finset.mem_filter.1 hj).2
      refine Finset.mem_insert_of_mem (Finset.mem_image.2 ⟨(i, h), ?_, rfl⟩)
      refine Finset.mem_product.2 ⟨Finset.mem_filter.2 ⟨Finset.mem_univ _, ?_⟩, Finset.mem_univ _⟩
      change i.val < past.length
      by_contra hi
      rw [List.getElem?_eq_none (by omega)] at ho
      simp at ho
  calc (populationsAfter (K := K) Dsamp heavy past).card
      ≤ (insert none (Finset.image some ((Finset.univ.filter fun i : Fin K =>
          i.val < past.length) ×ˢ (Finset.univ : Finset R)))).card := Finset.card_le_card hsub
    _ ≤ 1 + ((Finset.univ.filter fun i : Fin K => i.val < past.length)
          ×ˢ (Finset.univ : Finset R)).card := by
        rw [add_comm]
        exact (Finset.card_insert_le _ _).trans (Nat.add_le_add_right Finset.card_image_le 1)
    _ ≤ 1 + past.length * Fintype.card R := by
        rw [Finset.card_product, Finset.card_univ]
        exact Nat.add_le_add_left (Nat.mul_le_mul_right _ (card_fin_lt_le _)) 1

lemma card_clusterReads_le (heavy : ℝ) {M : ℕ} (past : List (Outcome S R))
    (e : ClusterPart S K R M) :
    (clusterReads Dsamp heavy past e).card
      ≤ ((populationsAfter (K := K) Dsamp heavy past).card * (2 * M) + 1) * (M + 1) := by
  classical
  refine (card_readSet_le _ _).trans (Nat.mul_le_mul ?_ ?_)
  · refine (Finset.card_insert_le _ _).trans (Nat.add_le_add_right ?_ 1)
    refine Finset.card_biUnion_le.trans ?_
    calc ∑ j ∈ populationsAfter Dsamp heavy past,
          ((Finset.range M).image (populationDraws past j (e.2.1 j))
            ∪ (Finset.range M).image (populationDraws past j (e.2.2 j))).card
        ≤ ∑ _j ∈ populationsAfter (K := K) Dsamp heavy past, 2 * M :=
          Finset.sum_le_sum fun j _ => by
          refine (Finset.card_union_le _ _).trans ?_
          have h1 := (Finset.card_image_le (s := Finset.range M)
            (f := populationDraws past j (e.2.1 j))).trans (Finset.card_range M).le
          have h2 := (Finset.card_image_le (s := Finset.range M)
            (f := populationDraws past j (e.2.2 j))).trans (Finset.card_range M).le
          omega
      _ = _ := by rw [Finset.sum_const, smul_eq_mul]
  · refine (Finset.card_insert_le _ _).trans (Nat.add_le_add_right ?_ 1)
    exact Finset.card_image_le.trans (Finset.card_range M).le

lemma card_gateReads_le {M G : ℕ} (e : Fin M → S) (g : Fin G → S) :
    (gateReads e g).card ≤ G * (M + 1) := by
  classical
  refine (card_readSet_le _ _).trans (Nat.mul_le_mul ?_ ?_)
  · exact Finset.card_image_le.trans (by simp)
  · exact (Finset.card_insert_le _ _).trans (Nat.add_le_add_right
      (Finset.card_image_le.trans (by simp)) 1)

lemma card_checkReads_le {N m : ℕ} (x : (Fin N → S) × (Fin m → S)) :
    (checkReads x).card ≤ N * (m + 1) := by
  classical
  refine (card_readSet_le _ _).trans (Nat.mul_le_mul ?_ ?_)
  · exact Finset.card_image_le.trans (by simp)
  · exact (Finset.card_insert_le _ _).trans (Nat.add_le_add_right
      (Finset.card_image_le.trans (by simp)) 1)

/-- A round's clustering reads fit its share of `T`, as its populations are at most those of
the rounds before it. -/
lemma card_clusterReads_le_share (heavy : ℝ) {M : ℕ} (past : List (Outcome S R))
    (e : ClusterPart S K R M) (hM : 1 ≤ M) (hR : 1 ≤ Fintype.card R) (hlen : past.length < K) :
    (clusterReads Dsamp heavy past e).card
      ≤ 2 * Fintype.card (Option (Fin K × R)) * M * (M + 1) := by
  refine (card_clusterReads_le Dsamp heavy past e).trans (Nat.mul_le_mul_right _ ?_)
  have hpop := card_populationsAfter_le (K := K) Dsamp heavy past
  rw [Fintype.card_option, Fintype.card_prod, Fintype.card_fin]
  set c := Fintype.card R
  set a := (populationsAfter (K := K) Dsamp heavy past).card
  set r := past.length
  have h1 : a * (2 * M) ≤ (1 + r * c) * (2 * M) := Nat.mul_le_mul_right _ hpop
  have h2 : r * c + c ≤ K * c := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hlen
  have h3 : 1 ≤ c * M := Nat.one_le_iff_ne_zero.2 (Nat.mul_ne_zero (by omega) (by omega))
  nlinarith

variable {O Dsamp schedule indecisionLimit α L stageReads} in
omit [IsProbabilityMeasure μ] in
lemma card_roundReads_le {Ts : ℕ} (hTs : ∀ F B c ω s, (stageReads F B c ω s).card ≤ Ts)
    (past : List (Outcome S R)) (ω : Ω) (d : RoundDraws S K R Xs L.M L.G L.N L.m)
    (hM : 1 ≤ L.M) (hR : 1 ≤ Fintype.card R) (hlen : past.length < K) :
    (roundReads O Dsamp schedule indecisionLimit α L stageReads past ω d).card
      ≤ 2 * Fintype.card (Option (Fin K × R)) * L.M * (L.M + 1) + Ts + L.G * (L.M + 1)
        + L.N * (L.m + 1) := by
  refine (Finset.card_union_le _ _).trans (Nat.add_le_add ((Finset.card_union_le _ _).trans
    (Nat.add_le_add ((Finset.card_union_le _ _).trans
      (Nat.add_le_add (card_clusterReads_le_share Dsamp L.heavy past _ hM hR hlen)
        (hTs _ _ _ _ _))) (card_gateReads_le _ _))) (card_checkReads_le _))

variable {O Dsamp schedule indecisionLimit α L stageReads} in
omit [IsProbabilityMeasure μ] in
lemma card_readsUpTo_le {Ts : ℕ} (hTs : ∀ F B c ω s, (stageReads F B c ω s).card ≤ Ts)
    (ω : Ω) (d : Fin K → RoundDraws S K R Xs L.M L.G L.N L.m) (hM : 1 ≤ L.M)
    (hR : 1 ≤ Fintype.card R) (r : ℕ) (hr : r ≤ K) :
    (readsUpTo O Dsamp schedule indecisionLimit α L stageReads ω d r).card
      ≤ r * (2 * Fintype.card (Option (Fin K × R)) * L.M * (L.M + 1) + Ts + L.G * (L.M + 1)
        + L.N * (L.m + 1)) := by
  induction r with
  | zero => simp [readsUpTo]
  | succ r ih =>
    have hr' : r < K := hr
    simp only [readsUpTo, dif_pos hr']
    have hlen := length_history O Dsamp schedule indecisionLimit α L ω d r
    refine (Finset.card_union_le _ _).trans ?_
    rw [Nat.succ_mul]
    exact Nat.add_le_add (ih (by omega))
      (card_roundReads_le hTs _ ω _ hM hR (by rw [hlen]; omega))

end Learner

end LearnerProof

end OrthoDFA
