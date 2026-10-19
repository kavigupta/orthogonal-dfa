import OrthoDFA.Proofs.RawEvents

/-!
# The round's verdict is measurable

The gate reads the oracle at finitely many strings, so its verdict at fixed samples is
measurable in the noise; over the run it decomposes along the samples' countably many values.
-/

namespace OrthoDFA

open MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S] {rule : Clusterer S}
variable {J : Type*} [Fintype J]

variable (O : Oracle μ S)

open scoped Classical in
lemma refApart_congr (A R : Finset S) {ω ω' : Ω}
    (h : ∀ p ∈ A, (O.mq p ω = 1 ↔ O.mq p ω' = 1)
      ∧ ∀ v ∈ R, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1)) :
    refApart O.mq A R ω = refApart O.mq A R ω' := by
  unfold refApart
  split_ifs with hR
  · have hf : ∀ v ∈ R, (A.filter (fun p => ¬ ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)))).card
        = (A.filter (fun p => ¬ ((O.mq (p * v) ω' = 1) ↔ (O.mq p ω' = 1)))).card :=
      fun v hv => congrArg Finset.card (Finset.filter_congr (fun p hp => by
        rw [(h p hp).1, (h p hp).2 v hv]))
    exact le_antisymm
      (Finset.le_inf' _ _ (fun v hv => (Finset.inf'_le _ hv).trans_eq (hf v hv)))
      (Finset.le_inf' _ _ (fun v hv => (Finset.inf'_le _ hv).trans_eq (hf v hv).symm))
  · rfl

open scoped Classical in
/-- The gate reads the sample, the family's votes on it and the references' reads on it. -/
lemma certifiesAt_congr (η₀ lim a : ℝ) (lo hi : ℕ) (F R P : Finset S) {ω ω' : Ω}
    (h : ∀ w ∈ readSet P (insert 1 (F ∪ R)), (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    certifiesAt O.mq η₀ lim a lo hi F R P ω → certifiesAt O.mq η₀ lim a lo hi F R P ω' := by
  classical
  have hPC : ∀ p ∈ P, ∀ u ∈ insert 1 (F ∪ R), (O.mq (p * u) ω = 1 ↔ O.mq (p * u) ω' = 1) :=
    fun p hp u hu => h _ (mem_readSet hp hu)
  have hP : ∀ p ∈ P, (O.mq p ω = 1 ↔ O.mq p ω' = 1) := fun p hp => by
    simpa using hPC p hp 1 (Finset.mem_insert_self _ _)
  have hvc : ∀ p ∈ P, voteCount O.mq F p ω = voteCount O.mq F p ω' := fun p hp =>
    voteCount_congr O F p (fun u hu => hPC p hp u
      (Finset.mem_insert_of_mem (Finset.mem_union_left _ hu)))
  have hacc : acceptSide O.mq hi F P ω = acceptSide O.mq hi F P ω' :=
    Finset.filter_congr (fun p hp => by rw [hvc p hp])
  have hrej : rejectSide O.mq lo F P ω = rejectSide O.mq lo F P ω' :=
    Finset.filter_congr (fun p hp => by rw [hvc p hp])
  have hside : ∀ (accept : Bool) (A : Finset S), A ⊆ P → ∀ a' : ℝ,
      sideCap O.mq accept A R a' η₀ ω = sideCap O.mq accept A R a' η₀ ω' := by
    intro accept A hA a'
    have hag : againstOn O.mq accept A ω = againstOn O.mq accept A ω' := by
      unfold againstOn
      exact congrArg Finset.card (Finset.filter_congr (fun p hp => by rw [hP p (hA hp)]))
    have hra := refApart_congr O A R (ω := ω) (ω' := ω') (fun p hp => ⟨hP p (hA hp),
      fun v hv => hPC p (hA hp) v (Finset.mem_insert_of_mem (Finset.mem_union_right _ hv))⟩)
    unfold sideCap
    rw [hag, hra]
  have hmis : miscutBound O.mq (sizeLevel a P.card) lo hi F R P η₀ ω
      = miscutBound O.mq (sizeLevel a P.card) lo hi F R P η₀ ω' := by
    unfold miscutBound
    rw [hside true (acceptSide O.mq hi F P ω) (Finset.filter_subset _ _),
      hside false (rejectSide O.mq lo F P ω) (Finset.filter_subset _ _), hacc, hrej]
  unfold certifiesAt
  rw [hmis]
  exact id

lemma measurableSet_certifiesAt (η₀ lim a : ℝ) (lo hi : ℕ) (F R P : Finset S) :
    MeasurableSet {ω | certifiesAt O.mq η₀ lim a lo hi F R P ω} :=
  measurableSet_of_readCongr O _ _ (fun _ _ h => certifiesAt_congr O η₀ lim a lo hi F R P h)

open scoped Classical in
/-- An event about the pool, the family and the noise, at fixed samples, is measurable when it
is for each family. -/
lemma measurableSet_cluster_noise (B : State) (P C : Finset S) (hone : (1 : S) ∈ C)
    (Φ : Finset S → Ω → Prop) (hΦ : ∀ A, MeasurableSet {ω | Φ A ω}) :
    MeasurableSet {ω | Φ (clusterOf rule O B.sc B.scd P C B.k ω) ω} := by
  classical
  have hcov : {ω | Φ (clusterOf rule O B.sc B.scd P C B.k ω) ω}
      = ⋃ A ∈ C.powerset, ({ω | clusterOf rule O B.sc B.scd P C B.k ω = A} ∩ {ω | Φ A ω}) := by
    ext ω
    simp only [Set.mem_ofPred_eq, Set.mem_iUnion, Finset.mem_powerset, Set.mem_inter_iff,
      exists_prop]
    exact ⟨fun h => ⟨_, clusterOf_subset O _ _ P C _ ω hone, rfl, h⟩,
      fun ⟨A, _, hc, hA⟩ => hc ▸ hA⟩
  rw [hcov]
  exact MeasurableSet.biUnion (Finset.countable_toSet _)
    (fun A _ => (measurableSet_clusterOf O _ _ P C _ hone A).inter (hΦ A))

open scoped Classical in
/-- An event about the tables, the pool, the family and the noise is measurable when it is for
each family. -/
lemma measurableSet_family_noise (populations : Finset J) (B : State)
    (Φ : Finset S → Ω → Prop) (hΦ : ∀ A, MeasurableSet {ω | Φ A ω}) :
    MeasurableSet {x : Run Ω S J | Φ (clusterBy rule O.mq populations x B) (oracleNoise x)} := by
  classical
  have hR : ∀ P C : Finset S, MeasurableSet (if (1 : S) ∈ C then
      oracleNoise ⁻¹' {ω | Φ (clusterOf rule O B.sc B.scd P C B.k ω) ω}
      else (∅ : Set (Run Ω S J))) := by
    intro P C
    split_ifs with hone
    · exact measurable_nz (measurableSet_cluster_noise O B P C hone Φ hΦ)
    · exact MeasurableSet.empty
  have hrw : {x : Run Ω S J | Φ (clusterBy rule O.mq populations x B) (oracleNoise x)}
      = {x : Run Ω S J | x ∈ (fun P C => if (1 : S) ∈ C then
          oracleNoise ⁻¹' {ω | Φ (clusterOf rule O B.sc B.scd P C B.k ω) ω} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)} := by
    ext x
    simp only [Set.mem_ofPred_eq, if_pos (one_mem_poolAt B.nsuff x), Set.mem_preimage]
    rfl
  rw [hrw]
  exact measurableSet_of_run_data populations B _ hR

open scoped Classical in
/-- As `measurableSet_family_noise`, reading population `j`'s certification sample at `m`. -/
lemma measurableSet_family_cert (populations : Finset J) (j : J) (B : State) (m : ℕ)
    (Φ : Finset S → Finset S → Finset S → Ω → Prop)
    (hΦ : ∀ C A A₀, MeasurableSet {ω | Φ C A A₀ ω}) :
    MeasurableSet {x : Run Ω S J | Φ (poolAt B.nsuff x)
      (certOf (prefixesAt populations B.npref x) j m x)
      (clusterBy rule O.mq populations x B) (oracleNoise x)} := by
  classical
  have hR : ∀ P C A : Finset S, MeasurableSet (if (1 : S) ∈ C then
      oracleNoise ⁻¹' {ω | Φ C A (clusterOf rule O B.sc B.scd P C B.k ω) ω}
      else (∅ : Set (Run Ω S J))) := by
    intro P C A
    split_ifs with hone
    · exact measurable_nz (measurableSet_cluster_noise O B P C hone (Φ C A) (hΦ C A))
    · exact MeasurableSet.empty
  have hrw : {x : Run Ω S J | Φ (poolAt B.nsuff x)
      (certOf (prefixesAt populations B.npref x) j m x)
      (clusterBy rule O.mq populations x B) (oracleNoise x)}
      = {x : Run Ω S J | x ∈ (fun P C A => if (1 : S) ∈ C then
          oracleNoise ⁻¹' {ω | Φ C A (clusterOf rule O B.sc B.scd P C B.k ω) ω} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (certOf (prefixesAt populations B.npref x) j m x)} := by
    ext x
    simp only [Set.mem_ofPred_eq, if_pos (one_mem_poolAt B.nsuff x), Set.mem_preimage]
    rfl
  rw [hrw]
  exact measurableSet_of_run_data_certAt populations j B m _ hR

lemma measurableSet_keptAt (populations : Finset J) (j : J) (m : ℕ) :
    MeasurableSet {x : Run Ω S J | keptAt populations j m x} := by
  classical
  have h1 : MeasurableSet {x : Run Ω S J | (prefixesOf j m x).card = m} := by
    have : {x : Run Ω S J | (prefixesOf j m x).card = m}
        = ⋃ F ∈ {F : Finset S | F.card = m}, {x : Run Ω S J | prefixesOf j m x = F} := by
      ext x; simp
    rw [this]
    exact MeasurableSet.biUnion (Set.to_countable _)
      (fun F _ => measurable_prefixStream j (measurableSet_freshDraws ∅ m F))
  have h2 : MeasurableSet {x : Run Ω S J |
      (certOf (prefixesAt populations m x) j m x).card = m} := by
    have : {x : Run Ω S J | (certOf (prefixesAt populations m x) j m x).card = m}
        = ⋃ F ∈ {F : Finset S | F.card = m},
          {x : Run Ω S J | certOf (prefixesAt populations m x) j m x = F} := by
      ext x; simp
    rw [this]
    exact MeasurableSet.biUnion (Set.to_countable _)
      (fun F _ => measurableSet_certOf populations m j m F)
  exact h1.inter h2

open scoped Classical in
lemma measurableSet_retBy (populations : Finset J) (uni : J) (η₀ indecisionLimit εcov α : ℝ)
    (B : State) :
    MeasurableSet (retBy rule O.mq populations uni η₀ indecisionLimit εcov α B) := by
  classical
  have h1 : MeasurableSet {x : Run Ω S J | B.k ≤ (clusterBy rule O.mq populations x B).card + 1} :=
    measurableSet_family_noise O populations B (fun A _ => B.k ≤ A.card + 1)
      (fun A => MeasurableSet.const _)
  have hfnr : ∀ j, MeasurableSet {x : Run Ω S J |
      (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
          ¬ decided O.mq B.lo (B.hi + 1) (familyBy rule O.mq populations x B) p
            (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * (certOf (prefixesAt populations B.npref x) j B.npref x).card} :=
    fun j => measurableSet_family_cert O populations j B B.npref
      (fun _ A A₀ ω => ((A.filter (fun p => ¬ decided O.mq B.lo (B.hi + 1) (insert 1 A₀) p ω)).card
        : ℝ) ≤ indecisionLimit * A.card)
      (fun _ A A₀ => measurableSet_indecisionCount O B.lo (B.hi + 1) (insert 1 A₀) A _)
  have hgate : ∀ j e, MeasurableSet {x : Run Ω S J |
      gateAt rule O.mq populations uni η₀ εcov α B j e x} :=
    fun j e => measurableSet_family_cert O populations j B (B.npref + e)
      (fun C A A₀ ω => certifiesAt O.mq η₀ (gateLimit uni εcov j) (gateLevel α populations B)
        B.lo B.hi A₀ (C \ insert 1 A₀) A ω)
      (fun C A A₀ => measurableSet_certifiesAt O _ _ _ _ _ _ _ _)
  have hrw : retBy rule O.mq populations uni η₀ indecisionLimit εcov α B
      = {x : Run Ω S J | B.k ≤ (clusterBy rule O.mq populations x B).card + 1}
        ∩ (⋂ j ∈ populations, ({x | keptAt populations j B.npref x}ᶜ
          ∪ {x : Run Ω S J |
            (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
              ¬ decided O.mq B.lo (B.hi + 1) (familyBy rule O.mq populations x B) p
                (oracleNoise x))).card : ℝ)
            ≤ indecisionLimit * (certOf (prefixesAt populations B.npref x) j B.npref x).card}))
        ∩ (⋂ j ∈ populations, ({x | keptAt populations j B.npref x}ᶜ
          ∪ ⋃ e ∈ Finset.range (B.npref + 1),
            {x : Run Ω S J | gateAt rule O.mq populations uni η₀ εcov α B j e x})) := by
    ext x
    simp only [retBy, Set.mem_ofPred_eq, Set.mem_inter_iff, Set.mem_iInter, Set.mem_union,
      Set.mem_compl_iff, Set.mem_iUnion, Finset.mem_range, exists_prop, and_assoc]
    refine and_congr Iff.rfl (and_congr ?_ ?_)
    · exact forall₂_congr (fun j _ => imp_iff_not_or)
    · refine forall₂_congr (fun j _ => ?_)
      rw [imp_iff_not_or]
      exact or_congr Iff.rfl (exists_congr (fun e => by
        constructor
        · rintro ⟨he, h⟩; exact ⟨by omega, h⟩
        · rintro ⟨he, h⟩; exact ⟨by omega, h⟩))
  rw [hrw]
  refine (h1.inter (Finset.measurableSet_biInter _ (fun j _ =>
    (measurableSet_keptAt populations j B.npref).compl.union (hfnr j)))).inter
    (Finset.measurableSet_biInter _ (fun j _ =>
      (measurableSet_keptAt populations j B.npref).compl.union
        (Finset.measurableSet_biUnion _ (fun e _ => hgate j e))))

end OrthoDFA
