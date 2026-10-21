import OrthoDFA.Proofs.RawNamed

/-!
# Pricing events of the draws

An event read off the sequences the round reads is priced at every assignment of the hash by
the stream-level lemmas; an event about the hash on a window the draws pick is priced at every
draw by the coin tails; and the uniform pool's share of a named population's window by Markov's
inequality.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory Thin
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S] {rule : Clusterer S}
variable {J : Type*} [Fintype J]

section Support

lemma ae_tableSeqOf_mem (D : Measure S) [IsProbabilityMeasure D] {Pre : Set S}
    (hD : D Preᶜ = 0) (L : List S) (hL : ∀ s ∈ L, s ∈ Pre) (A : Set S) :
    ∀ᵐ y ∂(Measure.map (fun p : (ℕ → S) × (ℕ → S) => tableSeqOf L A p.1 p.2)
      ((iidStream D).prod (iidStream D))), ∀ i, y i ∈ Pre := by
  rw [ae_map_iff (measurable_tableSeqOf L A).aemeasurable (measurableSet_forall_mem Pre)]
  have h1 : ∀ᵐ p ∂((iidStream D).prod (iidStream D)), ∀ i, p.1 i ∈ Pre :=
    Measure.quasiMeasurePreserving_fst.ae (ae_iidStream_mem D hD)
  have h2 : ∀ᵐ p ∂((iidStream D).prod (iidStream D)), ∀ i, p.2 i ∈ Pre :=
    Measure.quasiMeasurePreserving_snd.ae (ae_iidStream_mem D hD)
  filter_upwards [h1, h2] with p hp1 hp2 i
  rcases tableSeqOf_spec L A p.1 p.2 with h | h
  · rw [h i]; exact hp2 0
  · rcases h i with h' | ⟨k, -, hk⟩
    · exact hL _ h'
    · rw [← hk]; exact hp1 k

lemma ae_reserveSeqOf_mem (D : Measure S) [IsProbabilityMeasure D] {Pre : Set S}
    (hD : D Preᶜ = 0) (R : Set S) :
    ∀ᵐ y ∂(Measure.map (fun p : (ℕ → S) × (ℕ → S) => reserveSeqOf R p.1 p.2)
      ((iidStream D).prod (iidStream D))), ∀ i, y i ∈ Pre := by
  rw [ae_map_iff (measurable_reserveSeqOf R).aemeasurable (measurableSet_forall_mem Pre)]
  have h1 : ∀ᵐ p ∂((iidStream D).prod (iidStream D)), ∀ i, p.1 i ∈ Pre :=
    Measure.quasiMeasurePreserving_fst.ae (ae_iidStream_mem D hD)
  have h2 : ∀ᵐ p ∂((iidStream D).prod (iidStream D)), ∀ i, p.2 i ∈ Pre :=
    Measure.quasiMeasurePreserving_snd.ae (ae_iidStream_mem D hD)
  filter_upwards [h1, h2] with p hp1 hp2 i
  rcases reserveSeqOf_spec R p.1 p.2 with h | h
  · rw [h i]; exact hp2 1
  · obtain ⟨k, -, hk⟩ := h i
    rw [← hk]; exact hp1 k

lemma ae_sideLawT_mem (H : J → Finset S) (uni : J) (D : J → Measure S)
    [∀ j, IsProbabilityMeasure (D j)] (g : S → Bool) (populations : Finset J) {Pre : Set S}
    (hD : ∀ j ∈ populations, D j Preᶜ = 0) (hH : ∀ j, ∀ p ∈ H j, p ∈ Pre) :
    ∀ j ∈ populations, ∀ᵐ y ∂(sideLawT H uni D g j), ∀ i, y i ∈ Pre := by
  intro j hj
  unfold sideLawT
  split_ifs
  · exact ae_iidStream_mem (D j) (hD j hj)
  · exact ae_tableSeqOf_mem (D j) (hD j hj) _ (fun s hs => hH j s (Finset.mem_toList.1 hs)) _

lemma ae_sideLawC_mem (H : J → Finset S) (uni : J) (D : J → Measure S)
    [∀ j, IsProbabilityMeasure (D j)] (g : S → Bool) (populations : Finset J) {Pre : Set S}
    (hD : ∀ j ∈ populations, D j Preᶜ = 0) :
    ∀ j ∈ populations, ∀ᵐ y ∂(sideLawC H uni D g j), ∀ i, y i ∈ Pre := by
  intro j hj
  unfold sideLawC
  split_ifs
  · exact ae_iidStream_mem (D j) (hD j hj)
  · exact ae_reserveSeqOf_mem (D j) (hD j hj) _

lemma sideLawT_uni (H : J → Finset S) (uni : J) (D : J → Measure S) (g : S → Bool) :
    sideLawT H uni D g uni = iidStream (D uni) := by
  simp [sideLawT]

lemma sideLawC_uni (H : J → Finset S) (uni : J) (D : J → Measure S) (g : S → Bool) :
    sideLawC H uni D g uni = iidStream (D uni) := by
  simp [sideLawC]

end Support

section Slices

/-- An event about the sequences the round reads, priced at every assignment of the hash. -/
lemma runMeasure_streams_le (H : J → Finset S) (uni : J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (E : Set (Run Ω S J))
    (hE : MeasurableSet E) {b : ℝ}
    (h : ∀ g, (streamMeasure μ (sideLawT H uni D g) (sideLawC H uni D g) Dsf).real E ≤ b) :
    (runMeasure μ D Dsf).real (streams H uni ⁻¹' E) ≤ b := by
  have hb : 0 ≤ b := le_trans measureReal_nonneg (h (fun _ => true))
  have hE' : MeasurableSet (streams (Ω := Ω) H uni ⁻¹' E) := measurable_streams H uni hE
  refine ENNReal.toReal_le_of_le_ofReal hb
    (runMeasure_le_of_slices D Dsf _ hE' (ENNReal.ofReal b) (fun g => ?_))
  have hm : Measurable (fun x : Run Ω S J => streams H uni (x, g)) :=
    (measurable_streams H uni).comp (measurable_id.prodMk measurable_const)
  show baseRun μ D Dsf ((fun x => streams H uni (x, g)) ⁻¹' E) ≤ _
  rw [← Measure.map_apply hm hE, map_streams H uni D Dsf g, ← ofReal_measureReal]
  exact ENNReal.ofReal_le_ofReal (h g)

/-- An event about the hash, priced at every draw. -/
lemma runMeasure_coins_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (E : Set (Draws Ω S J))
    (hE : MeasurableSet E) {b : ℝ} (hb : 0 ≤ b)
    (h : ∀ x, (coins S).real {g | (x, g) ∈ E} ≤ b) : (runMeasure μ D Dsf).real E ≤ b := by
  refine ENNReal.toReal_le_of_le_ofReal hb
    (runMeasure_le_of_coin_slices D Dsf E hE (ENNReal.ofReal b) (fun x => ?_))
  rw [← ofReal_measureReal]
  exact ENNReal.ofReal_le_ofReal (h x)

/-- An event about the draws alone. -/
lemma runMeasure_base_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (E : Set (Run Ω S J))
    (hE : MeasurableSet E) {b : ℝ} (h : (baseRun μ D Dsf).real E ≤ b) :
    (runMeasure μ D Dsf).real {x | x.1 ∈ E} ≤ b := by
  have he : {x : Draws Ω S J | x.1 ∈ E} = E ×ˢ Set.univ := by ext x; simp
  rw [he, measureReal_def, runMeasure_eq_prod, Measure.prod_prod, measure_univ, mul_one,
    ← measureReal_def]
  exact h

end Slices

section Measurability

variable (O : Oracle μ S)

open scoped Classical in
/-- An event about the tables, the pool and the family is measurable. -/
lemma measurableSet_cluster_data (populations : Finset J) (B : State)
    (Φ : Finset S → Finset S → Finset S → Prop) :
    MeasurableSet {x : Run Ω S J | Φ (prefixesAt populations B.npref x) (poolAt B.nsuff x)
      (clusterBy rule O.mq populations x B)} := by
  classical
  have hR : ∀ P C : Finset S, MeasurableSet (if (1 : S) ∈ C then
      oracleNoise ⁻¹' {ω | Φ P C (clusterOf rule O B.sc B.scd P C B.k ω)}
      else (∅ : Set (Run Ω S J))) := by
    intro P C
    split_ifs with hone
    · refine measurable_nz ?_
      have hcov : {ω | Φ P C (clusterOf rule O B.sc B.scd P C B.k ω)}
          = ⋃ A ∈ C.powerset.filter (fun A => Φ P C A),
              {ω | clusterOf rule O B.sc B.scd P C B.k ω = A} := by
        ext ω
        simp only [Set.mem_ofPred_eq, Set.mem_iUnion, Finset.mem_filter, Finset.mem_powerset,
          exists_prop]
        exact ⟨fun h => ⟨_, ⟨clusterOf_subset O _ _ P C _ ω hone, h⟩, rfl⟩,
          fun ⟨A, ⟨_, hA⟩, hc⟩ => hc ▸ hA⟩
      rw [hcov]
      exact MeasurableSet.biUnion (Finset.countable_toSet _)
        (fun A _ => measurableSet_clusterOf O _ _ P C _ hone A)
    · exact MeasurableSet.empty
  have hrw : {x : Run Ω S J | Φ (prefixesAt populations B.npref x) (poolAt B.nsuff x)
      (clusterBy rule O.mq populations x B)}
      = {x : Run Ω S J | x ∈ (fun P C => if (1 : S) ∈ C then
          oracleNoise ⁻¹' {ω | Φ P C (clusterOf rule O B.sc B.scd P C B.k ω)} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)} := by
    ext x
    simp only [Set.mem_ofPred_eq, if_pos (one_mem_poolAt B.nsuff x), Set.mem_preimage]
    rfl
  rw [hrw]
  exact measurableSet_of_run_data populations B _ hR

lemma measurableSet_stalled (populations : Finset J) (B : State) (kmin kmax : ℕ) :
    MeasurableSet (stalled rule O populations B kmin kmax) :=
  measurableSet_cluster_data O populations B (fun _ _ A => ¬ (kmin ≤ A.card ∧ A.card ≤ kmax))

lemma measurableSet_dirty (populations : Finset J) (B : State) (Dj : Measure S) (Δp : ℝ) :
    MeasurableSet {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B,
      flipMass O Dj v ≤ Δp} :=
  measurableSet_cluster_data O populations B (fun _ _ A => ¬ ∀ v ∈ A, flipMass O Dj v ≤ Δp)

lemma measurableSet_noClean (populations : Finset J) (B : State) :
    MeasurableSet {x : Run Ω S J | ¬ ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
      ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0} := by
  classical
  have hR : ∀ P C : Finset S, MeasurableSet
      {_x : Run Ω S J | ¬ ∃ w₀ ∈ C.erase 1, ∀ p ∈ P, O.flip w₀ p = 0} := by
    intro P C
    by_cases h : ∃ w₀ ∈ C.erase 1, ∀ p ∈ P, O.flip w₀ p = 0
    · simp only [h, not_true_eq_false, Set.setOf_false, MeasurableSet.empty]
    · simp only [h, not_false_eq_true, Set.setOf_true, MeasurableSet.univ]
  exact measurableSet_of_run_data populations B _ hR

end Measurability

lemma measurableSet_notPoolGood (M r : ℕ) :
    MeasurableSet {x : Run Ω S J | ¬ poolGood M r x} :=
  measurable_suffixStream (measurableSet_acceptedAt ∅ (M + r) (fun F => ¬ M ≤ F.card))

lemma measurableSet_notTabGood (j : J) (m r : ℕ) :
    MeasurableSet {x : Run Ω S J | ¬ tabGood j m r x} :=
  measurable_prefixStream j (measurableSet_acceptedAt ∅ (m + r) (fun F => ¬ m ≤ F.card))

lemma measurableSet_notCertGood (populations : Finset J) (j : J) (m r : ℕ) :
    MeasurableSet {x : Run Ω S J | ¬ certGood populations j m r x} := by
  classical
  have hcov : {x : Run Ω S J | ¬ certGood populations j m r x}
      = ⋃ P : Finset S, {x : Run Ω S J | prefixesAt populations m x = P}
          ∩ (fun x : Run Ω S J => (fun i => certPrefix j i x)) ⁻¹'
            {y | ¬ m ≤ (acceptedAt P y (m + r)).card} := by
    ext x
    simp only [Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_preimage,
      certGood]
    exact ⟨fun h => ⟨_, rfl, h⟩, fun ⟨P, hP, h⟩ => hP ▸ h⟩
  rw [hcov]
  exact MeasurableSet.iUnion (fun P => (measurableSet_prefixesAt populations m P).inter
    (measurable_certStream j (measurableSet_acceptedAt P (m + r) (fun F => ¬ m ≤ F.card))))

section Window

lemma measurable_prefixStreamOf (j : J) :
    Measurable (fun r : Run Ω S J => r.2.1.2 j) :=
  (measurable_pi_apply j).comp measurable_prfs

lemma measurableSet_window_of (Q : Set S) (K : ℕ) (j : J) (z : Finset S) :
    MeasurableSet {r : Run Ω S J | window Q K (r.2.1.2 j) = z} :=
  measurable_prefixStreamOf j (measurableSet_window_eq Q K (fun z' => z' = z))

omit [Stringlike S] in
/-- More than half of a window's coins plus `G` on the side `b` has chance at most
`exp(-2G²/|V|)`. -/
theorem coins_window_side (V : Finset S) (b : Bool) {g₀ : ℝ} (hg₀ : 0 ≤ g₀) (hV : 0 < V.card) :
    (coins S).real {g | (V.card : ℝ) / 2 + g₀ ≤ ((V.filter (fun s => g s = b)).card : ℝ)}
      ≤ Real.exp (-2 * g₀ ^ 2 / V.card) := by
  have hVr : (0 : ℝ) < V.card := by exact_mod_cast hV
  have h := coins_count_upper V b (τ := g₀ / V.card) (by positivity)
  refine le_trans (measureReal_mono (fun g hg => ?_) (measure_ne_top _ _)) (le_trans h ?_)
  · simp only [Set.mem_ofPred_eq] at hg ⊢
    have e : (V.card : ℝ) * (1 / 2 + g₀ / V.card) = (V.card : ℝ) / 2 + g₀ := by
      field_simp
    linarith
  · refine Real.exp_le_exp.2 (le_of_eq ?_)
    field_simp

open scoped Classical in
/-- The coins on a window of at most `2N` strings put `G` more than half of it on a side with
chance at most `2 exp(-G²/N)`. -/
theorem coins_window_card_le (V : Finset S) (N G : ℕ) (hN : 0 < N) (hV : V.card ≤ 2 * N) :
    (coins S).real {g | (V.card : ℝ) / 2 + G ≤ ((V.filter (fun s => g s = true)).card : ℝ)
        ∨ (V.card : ℝ) / 2 + G ≤ ((V.filter (fun s => g s = false)).card : ℝ)}
      ≤ 2 * Real.exp (-((G : ℝ) ^ 2 / N)) := by
  have hNR : (0 : ℝ) < N := by exact_mod_cast hN
  rcases Nat.eq_zero_or_pos G with hG | hG
  · rw [hG]
    simp only [Nat.cast_zero, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow,
      zero_div, neg_zero, Real.exp_zero, mul_one]
    linarith [measureReal_le_one (μ := coins S) (s := {g : S → Bool | (V.card : ℝ) / 2 + 0
      ≤ ((V.filter (fun s => g s = true)).card : ℝ)
        ∨ (V.card : ℝ) / 2 + 0 ≤ ((V.filter (fun s => g s = false)).card : ℝ)})]
  rcases Nat.eq_zero_or_pos V.card with hV0 | hVpos
  · have hGR : (0 : ℝ) < G := by exact_mod_cast hG
    have hempty : {g : S → Bool | (V.card : ℝ) / 2 + G
        ≤ ((V.filter (fun s => g s = true)).card : ℝ)
        ∨ (V.card : ℝ) / 2 + G ≤ ((V.filter (fun s => g s = false)).card : ℝ)} = ∅ := by
      ext g
      have hVe : V = ∅ := Finset.card_eq_zero.1 hV0
      simp only [hVe, Finset.card_empty, Nat.cast_zero, zero_div, zero_add, Finset.filter_empty,
        Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_or, not_le]
      exact ⟨hGR, hGR⟩
    rw [hempty, measureReal_empty]
    positivity
  have hVR : (0 : ℝ) < V.card := by exact_mod_cast hVpos
  have hexp : Real.exp (-2 * (G : ℝ) ^ 2 / V.card) ≤ Real.exp (-((G : ℝ) ^ 2 / N)) := by
    refine Real.exp_le_exp.2 ?_
    have hV2 : (V.card : ℝ) ≤ 2 * N := by exact_mod_cast hV
    rw [show -2 * (G : ℝ) ^ 2 / V.card = -(2 * (G : ℝ) ^ 2 / V.card) by ring, neg_le_neg_iff,
      div_le_div_iff₀ hNR hVR]
    nlinarith [sq_nonneg (G : ℝ)]
  rw [Set.setOf_or]
  refine le_trans (measureReal_union_le _ _) ?_
  have h1 := coins_window_side V true (Nat.cast_nonneg G) hVpos
  have h2 := coins_window_side V false (Nat.cast_nonneg G) hVpos
  linarith

end Window

section Hits

lemma measureReal_window_le (D : Measure S) [IsProbabilityMeasure D] {ρ : ℝ}
    (hρ : ∀ t, D.real {t} ≤ ρ) (Z : Finset S) : D.real ↑Z ≤ Z.card * ρ := by
  have hZ : (↑Z : Set S) = ⋃ s ∈ Z, ({s} : Set S) := by ext; simp
  rw [hZ]
  calc D.real (⋃ s ∈ Z, ({s} : Set S)) ≤ ∑ s ∈ Z, D.real {s} := measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _s ∈ Z, ρ := Finset.sum_le_sum (fun s _ => hρ s)
    _ = Z.card * ρ := by rw [Finset.sum_const, nsmul_eq_mul]

lemma map_prefixStreams_base (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] :
    Measure.map (fun r : Run Ω S J => r.2.1.2) (baseRun μ D Dsf)
      = Measure.pi (fun j => iidStream (D j)) := by
  rw [show (fun r : Run Ω S J => r.2.1.2) = Prod.snd ∘ (draws : Run Ω S J → _) from rfl,
    ← Measure.map_map measurable_snd measurable_draws, baseRun, map_drawBlock,
    Measure.map_snd_prod]
  simp only [measure_univ, one_smul]

/-- One of the uniform pool's draws lands in a named population's window, a fixed set of at
most `K` strings the uniform pool's draws are independent of, with chance at most `Kρ`. -/
lemma measureReal_draw_in_window (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (uni j : J) (hju : j ≠ uni)
    {ρ : ℝ} (hρ : ∀ t, (D uni).real {t} ≤ ρ) (Q : Set S) (K i : ℕ) :
    (baseRun μ D Dsf).real {r : Run Ω S J | r.2.1.2 uni i ∈ window Q K (r.2.1.2 j)}
      ≤ K * ρ := by
  classical
  have hρ0 : 0 ≤ ρ := le_trans measureReal_nonneg (hρ 1)
  set A' : Set ((ℕ → S) × (ℕ → S)) := {p | p.1 i ∈ window Q K p.2} with hA'
  have hA'm : MeasurableSet A' := by
    have : A' = ⋃ z : Finset S, {p : (ℕ → S) × (ℕ → S) | p.1 i ∈ z}
        ∩ Prod.snd ⁻¹' {w | window Q K w = z} := by
      ext p
      simp only [hA', Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_preimage]
      exact ⟨fun h => ⟨_, h, rfl⟩, fun ⟨z, h, hz⟩ => hz ▸ h⟩
    rw [this]
    refine MeasurableSet.iUnion (fun z => MeasurableSet.inter ?_
      (measurable_snd (measurableSet_window_eq Q K (fun z' => z' = z))))
    exact ((measurable_pi_apply i).comp measurable_fst) (Set.to_countable _).measurableSet
  have hpair : Measurable (fun q : J → ℕ → S => (q uni, q j)) := by fun_prop
  have hind : IndepFun (fun q : J → ℕ → S => q uni) (fun q => q j)
      (Measure.pi (fun j => iidStream (D j))) :=
    (iIndepFun_pi (X := fun _ (y : ℕ → S) => y) (μ := fun j => iidStream (D j))
      (fun _ => measurable_id.aemeasurable)).indepFun hju.symm
  have hlaw : Measure.map (fun q : J → ℕ → S => (q uni, q j))
      (Measure.pi (fun j => iidStream (D j))) = (iidStream (D uni)).prod (iidStream (D j)) := by
    rw [(indepFun_iff_map_prod_eq_prod_map_map (measurable_pi_apply uni).aemeasurable
      (measurable_pi_apply j).aemeasurable).1 hind,
      (measurePreserving_eval (fun j => iidStream (D j)) uni).map_eq,
      (measurePreserving_eval (fun j => iidStream (D j)) j).map_eq]
  have hset : {r : Run Ω S J | r.2.1.2 uni i ∈ window Q K (r.2.1.2 j)}
      = (fun r : Run Ω S J => r.2.1.2) ⁻¹' ((fun q : J → ℕ → S => (q uni, q j)) ⁻¹' A') := rfl
  have hval : baseRun μ D Dsf {r : Run Ω S J | r.2.1.2 uni i ∈ window Q K (r.2.1.2 j)}
      ≤ ENNReal.ofReal (K * ρ) := by
    rw [hset, ← Measure.map_apply (by fun_prop) (hpair hA'm), map_prefixStreams_base,
      ← Measure.map_apply hpair hA'm, hlaw, Measure.prod_apply_symm hA'm]
    refine le_trans (lintegral_mono (g := fun _ => ENNReal.ofReal (K * ρ)) (fun w => ?_)) ?_
    · have hpre : (fun a : ℕ → S => (a, w)) ⁻¹' A' = (fun a : ℕ → S => a i) ⁻¹' ↑(window Q K w) :=
        rfl
      rw [hpre, ← Measure.map_apply (measurable_pi_apply i) (Set.to_countable _).measurableSet,
        (measurePreserving_eval_infinitePi (fun _ : ℕ => D uni) i).map_eq, ← ofReal_measureReal]
      refine ENNReal.ofReal_le_ofReal (le_trans (measureReal_window_le (D uni) hρ _) ?_)
      exact mul_le_mul_of_nonneg_right (by exact_mod_cast window_card_le Q K w) hρ0
    · simp
  have hK : 0 ≤ (K : ℝ) * ρ := by positivity
  exact ENNReal.toReal_le_of_le_ofReal hK hval

open scoped Classical in
/-- The uniform pool's first `n` distinct draws land `a > 0` times in a named population's window,
a set of at most `K` strings drawn independently of them, no more often than `Bin(n, Kρ)` reaches
`⌈a⌉`. -/
theorem measureReal_hits_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (uni j : J) (hju : j ≠ uni)
    {ρ : ℝ} (hρ : ∀ t, (D uni).real {t} ≤ ρ) (Q : Set S) (K n : ℕ) {a : ℝ} (ha : 0 < a) :
    (baseRun μ D Dsf).real {r : Run Ω S J |
        a ≤ ((window Q K (r.2.1.2 j) ∩ acceptedAt ∅ (r.2.1.2 uni) n).card : ℝ)}
      ≤ rejTail n (K * ρ) (⌈a⌉₊ - 1) := by
  classical
  have hρ0 : 0 ≤ ρ := le_trans measureReal_nonneg (hρ 1)
  have hk1 : 1 ≤ ⌈a⌉₊ := Nat.one_le_iff_ne_zero.2 (Nat.ceil_pos.2 ha).ne'
  set k : ℕ := ⌈a⌉₊ - 1 with hk
  have hkk : k + 1 = ⌈a⌉₊ := by omega
  set A' : Set ((ℕ → S) × (ℕ → S)) :=
    {p | k + 1 ≤ ((window Q K p.2 ∩ acceptedAt ∅ p.1 n).card)} with hA'
  have hA'm : MeasurableSet A' := by
    have : A' = ⋃ z : Finset S, Prod.fst ⁻¹' {s : ℕ → S | k + 1 ≤ (z ∩ acceptedAt ∅ s n).card}
        ∩ Prod.snd ⁻¹' {w | window Q K w = z} := by
      ext p
      simp only [hA', Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_preimage]
      exact ⟨fun h => ⟨_, h, rfl⟩, fun ⟨z, h, hz⟩ => hz ▸ h⟩
    rw [this]
    exact MeasurableSet.iUnion (fun z => MeasurableSet.inter
      (measurable_fst (measurableSet_acceptedAt ∅ n (fun A => k + 1 ≤ (z ∩ A).card)))
      (measurable_snd (measurableSet_window_eq Q K (fun z' => z' = z))))
  have hpair : Measurable (fun q : J → ℕ → S => (q uni, q j)) := by fun_prop
  have hind : IndepFun (fun q : J → ℕ → S => q uni) (fun q => q j)
      (Measure.pi (fun j => iidStream (D j))) :=
    (iIndepFun_pi (X := fun _ (y : ℕ → S) => y) (μ := fun j => iidStream (D j))
      (fun _ => measurable_id.aemeasurable)).indepFun hju.symm
  have hlaw : Measure.map (fun q : J → ℕ → S => (q uni, q j))
      (Measure.pi (fun j => iidStream (D j))) = (iidStream (D uni)).prod (iidStream (D j)) := by
    rw [(indepFun_iff_map_prod_eq_prod_map_map (measurable_pi_apply uni).aemeasurable
      (measurable_pi_apply j).aemeasurable).1 hind,
      (measurePreserving_eval (fun j => iidStream (D j)) uni).map_eq,
      (measurePreserving_eval (fun j => iidStream (D j)) j).map_eq]
  have hsub : {r : Run Ω S J |
        a ≤ ((window Q K (r.2.1.2 j) ∩ acceptedAt ∅ (r.2.1.2 uni) n).card : ℝ)}
      ⊆ (fun r : Run Ω S J => r.2.1.2) ⁻¹' ((fun q : J → ℕ → S => (q uni, q j)) ⁻¹' A') := by
    intro r hr
    show k + 1 ≤ (window Q K (r.2.1.2 j) ∩ acceptedAt ∅ (r.2.1.2 uni) n).card
    have hr' : a ≤ ((window Q K (r.2.1.2 j) ∩ acceptedAt ∅ (r.2.1.2 uni) n).card : ℝ) := hr
    have := Nat.ceil_le.2 hr'
    omega
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  have hval : baseRun μ D Dsf
        ((fun r : Run Ω S J => r.2.1.2) ⁻¹' ((fun q : J → ℕ → S => (q uni, q j)) ⁻¹' A'))
      ≤ ENNReal.ofReal (rejTail n (K * ρ) k) := by
    rw [← Measure.map_apply (by fun_prop) (hpair hA'm), map_prefixStreams_base,
      ← Measure.map_apply hpair hA'm, hlaw, Measure.prod_apply_symm hA'm]
    refine le_trans (lintegral_mono (g := fun _ => ENNReal.ofReal (rejTail n (K * ρ) k))
      (fun w => ?_)) ?_
    · set W : Finset S := window Q K w with hW
      have hmp := measurePreserving_finRestrict (D uni) n
      have hpre : (fun s : ℕ → S => (s, w)) ⁻¹' A'
          = (fun s : ℕ → S => fun i : Fin n => s i.val) ⁻¹'
            {z : Fin n → S | k + 1 ≤ (((Finset.univ : Finset (Fin n)).image z \ ∅).filter
              (· ∈ (W : Set S))).card} := by
        ext s
        simp only [Set.mem_preimage, Set.mem_ofPred_eq, hA', acceptedAt,
          image_range_eq_image_univ n s]
        rw [Finset.inter_comm, ← Finset.filter_mem_eq_inter]
        rfl
      rw [hpre, hmp.measure_preimage (Set.to_countable _).measurableSet.nullMeasurableSet,
        ← ofReal_measureReal]
      refine ENNReal.ofReal_le_ofReal ?_
      have key := (successiveDrawsTails_holds (D uni) n ∅ 1 (↑W : Set S) le_rfl
        (fun F _ => measureReal_le_one)).2.1 (k + 1)
      have hW1 : (D uni).real (↑W : Set S) ≤ min (K * ρ) 1 := by
        refine le_min (le_trans (measureReal_window_le (D uni) hρ W) ?_) measureReal_le_one
        exact mul_le_mul_of_nonneg_right (by exact_mod_cast window_card_le Q K w) hρ0
      refine le_trans (le_of_eq ?_)
        (le_trans key (binomSfGe_mono measureReal_nonneg (min_le_right _ _) hW1 n (k + 1)))
      congr 1
      ext z
      simp only [Set.mem_ofPred_eq]
      congr!
    · simp
  have hK : 0 ≤ rejTail n (K * ρ) k := rejTail_nonneg _ (by positivity) _
  exact ENNReal.toReal_le_of_le_ofReal hK hval

end Hits

end OrthoDFA
