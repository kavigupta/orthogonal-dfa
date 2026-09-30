import OrthoDFA.Proofs.Adaptive

/-!
# Lloyd's clustering is a `Clusterer`

`clusterAround`, the clustering the claim is stated for, meets `Clusterer`'s conditions, so the
proof, which is carried out for any `Clusterer`, covers it.
-/

namespace OrthoDFA

variable {Ω : Type*} [MeasurableSpace Ω] {S : Type*} [Stringlike S]

section leastLoss
variable {S : Type*} [DecidableEq S] (ℓ : S → ℝ) (cands : Finset S) (k : ℕ)

lemma leastLossSubset_mem (hk : k ≤ cands.card) :
    leastLossSubset ℓ cands k ∈ cands.powersetCard k := by
  have h : (cands.powersetCard k).Nonempty := Finset.powersetCard_nonempty.mpr hk
  rw [leastLossSubset, dif_pos h]
  exact (Finset.exists_min_image (cands.powersetCard k) (fun T => ∑ x ∈ T, ℓ x) h).choose_spec.1

lemma leastLossSubset_subset (hk : k ≤ cands.card) : leastLossSubset ℓ cands k ⊆ cands :=
  (Finset.mem_powersetCard.mp (leastLossSubset_mem ℓ cands k hk)).1

lemma leastLossSubset_card (hk : k ≤ cands.card) : (leastLossSubset ℓ cands k).card = k :=
  (Finset.mem_powersetCard.mp (leastLossSubset_mem ℓ cands k hk)).2

/-- The defining property: every chosen element has loss ≤ every unchosen candidate.
Proved from minimality of the argmin by the swap `v ↦ w`. -/
lemma leastLossSubset_least (hk : k ≤ cands.card) :
    ∀ v ∈ leastLossSubset ℓ cands k, ∀ w ∈ cands, w ∉ leastLossSubset ℓ cands k →
      ℓ v ≤ ℓ w := by
  intro v hv w hw hwnot
  by_contra hlt
  push_neg at hlt
  have h : (cands.powersetCard k).Nonempty := Finset.powersetCard_nonempty.mpr hk
  have hspec :=
    (Finset.exists_min_image (cands.powersetCard k) (fun T => ∑ x ∈ T, ℓ x) h).choose_spec
  have hTeq : leastLossSubset ℓ cands k
      = (Finset.exists_min_image (cands.powersetCard k) (fun T => ∑ x ∈ T, ℓ x) h).choose := by
    rw [leastLossSubset, dif_pos h]
  have hwnoterase : w ∉ (leastLossSubset ℓ cands k).erase v :=
    fun hh => hwnot (Finset.mem_of_mem_erase hh)
  set T' := insert w ((leastLossSubset ℓ cands k).erase v) with hT'
  have hT'mem : T' ∈ cands.powersetCard k := by
    rw [Finset.mem_powersetCard]
    refine ⟨?_, ?_⟩
    · rw [hT', Finset.insert_subset_iff]
      exact ⟨hw, (Finset.erase_subset v _).trans (leastLossSubset_subset ℓ cands k hk)⟩
    · have hkpos : 0 < k := by
        have hp := Finset.card_pos.mpr ⟨v, hv⟩
        rwa [leastLossSubset_card ℓ cands k hk] at hp
      rw [hT', Finset.card_insert_of_notMem hwnoterase, Finset.card_erase_of_mem hv,
        leastLossSubset_card ℓ cands k hk]
      omega
  have hmin := hspec.2 T' hT'mem
  rw [← hTeq] at hmin
  have hsum : ∑ x ∈ T', ℓ x = (∑ x ∈ leastLossSubset ℓ cands k, ℓ x) - ℓ v + ℓ w := by
    rw [hT', Finset.sum_insert hwnoterase, Finset.sum_erase_eq_sub hv]; ring
  rw [hsum] at hmin
  linarith
end leastLoss

lemma voteCount_congr' (mq : S → Ω → ℝ) (F : Finset S) (p : S) {ω ω' : Ω}
    (h : ∀ v ∈ F, (mq (p * v) ω = 1 ↔ mq (p * v) ω' = 1)) :
    voteCount mq F p ω = voteCount mq F p ω' := by
  classical
  unfold voteCount
  exact congrArg Finset.card (Finset.filter_congr (fun v hv => h v hv))

/-- The cluster never drifts off the seed.  `identify_cluster_around` stops the moment
`ε` would leave, so every cluster the loop proposes contains it — which is what lets the
gate read the split off `ε`'s own column. -/
lemma one_mem_clusterAround (mq : S → Ω → ℝ) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ) :
    (1 : S) ∈ clusterAround mq cn cd P cands ω k := by
  classical
  unfold clusterAround
  generalize k * P.card + 1 = n
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Function.iterate_succ_apply']
      unfold lloydStep
      split_ifs
      · exact Finset.mem_insert_self _ _
      · exact ih

lemma hammingLoss_congr (mq : S → Ω → ℝ) (F : Finset S) (cn cd : ℕ) {P cands : Finset S}
    (hF : F ⊆ cands) {ω ω' : Ω} {v : S} (hv : v ∈ cands)
    (h : ∀ w ∈ readSet P cands, (mq w ω = 1 ↔ mq w ω' = 1)) :
    hammingLoss mq F cn cd P ω v = hammingLoss mq F cn cd P ω' v := by
  classical
  unfold hammingLoss
  refine congrArg _ (congrArg Finset.card (Finset.filter_congr (fun p hp => ?_)))
  rw [voteCount_congr' mq F p (fun v' hv' => h _ (mem_readSet hp (hF hv')))]
  exact not_congr (iff_congr (h _ (mem_readSet hp hv)) Iff.rfl)

lemma leastLossSubset_subset' (l : S → ℝ) (cands : Finset S) (k : ℕ) :
    leastLossSubset l cands k ⊆ cands := by
  classical
  unfold leastLossSubset
  split_ifs with hne
  · exact (Finset.mem_powersetCard.mp (Finset.exists_min_image _ _ hne).choose_spec.1).1
  · exact Finset.empty_subset _

lemma clusterLoss_congr (mq : S → Ω → ℝ) (F : Finset S) (cn cd : ℕ) (P cands : Finset S)
    (hF : F ⊆ cands) {ω ω' : Ω} (h : ∀ w ∈ readSet P cands, (mq w ω = 1 ↔ mq w ω' = 1)) :
    clusterLoss mq F cn cd P cands ω = clusterLoss mq F cn cd P cands ω' := by
  classical
  funext v
  unfold clusterLoss
  split_ifs with hv
  · exact hammingLoss_congr mq F cn cd hF hv h
  · rfl

lemma lloydStep_congr (mq : S → Ω → ℝ) (cn cd : ℕ) (P cands : Finset S) (k : ℕ) {F : Finset S}
    (hF : F ⊆ cands) {ω ω' : Ω} (h : ∀ w ∈ readSet P cands, (mq w ω = 1 ↔ mq w ω' = 1)) :
    lloydStep mq cn cd P cands ω k F = lloydStep mq cn cd P cands ω' k F := by
  classical
  unfold lloydStep
  rw [clusterLoss_congr mq F cn cd P cands hF h]

lemma clusterLoss_nonneg (mq : S → Ω → ℝ) (F : Finset S) (cn cd : ℕ) (P cands : Finset S)
    (ω : Ω) (v : S) : 0 ≤ clusterLoss mq F cn cd P cands ω v := by
  classical
  unfold clusterLoss hammingLoss
  split_ifs
  · exact Nat.cast_nonneg _
  · exact le_rfl

open scoped Classical in
/-- The seed's own loss against its own column is zero, so the first step always ranks
it first — which is what stops the clustering from stalling at `{ε}`. -/
lemma clusterLoss_seed_zero (mq : S → Ω → ℝ) {cn cd : ℕ} (hcd : cn < cd) (P cands : Finset S)
    (ω : Ω) (hone : (1 : S) ∈ cands) :
    clusterLoss mq {(1 : S)} cn cd P cands ω 1 = 0 := by
  classical
  unfold clusterLoss
  rw [if_pos hone, hammingLoss]
  have hempty : P.filter (fun p => ¬ ((mq (p * 1) ω = 1)
      ↔ cn * ({(1 : S)} : Finset S).card < cd * voteCount mq {(1 : S)} p ω)) = ∅ := by
    refine Finset.filter_eq_empty_iff.2 (fun p _ => ?_)
    simp only [Classical.not_not, Finset.card_singleton, mul_one, mul_comm]
    unfold voteCount
    by_cases h : mq p ω = 1
    · have hp : ({(1 : S)} : Finset S).filter (fun v => mq (p * v) ω = 1) = {(1 : S)} := by
        refine Finset.filter_eq_self.2 (fun v hv => ?_)
        rw [Finset.mem_singleton.1 hv, mul_one]
        exact h
      rw [hp]
      simp only [Finset.card_singleton, mul_one]
      exact ⟨fun _ => hcd, fun _ => h⟩
    · have hp : ({(1 : S)} : Finset S).filter (fun v => mq (p * v) ω = 1) = ∅ := by
        refine Finset.filter_eq_empty_iff.2 (fun v hv => ?_)
        rw [Finset.mem_singleton.1 hv, mul_one]
        exact h
      rw [hp]
      simp only [Finset.card_empty, mul_zero]
      exact ⟨fun hc => absurd hc h, fun hc => absurd hc (by omega)⟩
  rw [hempty]
  simp

open scoped Classical in
/-- The first step takes its cohort: the seed ranks first and the rest are the `k−1` next. -/
lemma lloydStep_seed_card (mq : S → Ω → ℝ) {cn cd : ℕ} (hcd : cn < cd) (P cands : Finset S)
    (ω : Ω) (k : ℕ) (hone : (1 : S) ∈ cands) (hk : k ≤ cands.card) (hkpos : 0 < k) :
    (lloydStep mq cn cd P cands ω k {(1 : S)}).card = k := by
  classical
  have hkm : k - 1 ≤ (cands.erase 1).card := by
    rw [Finset.card_erase_of_mem hone]; omega
  have hguard : ∀ w ∈ cands, w ∉ insert (1 : S)
        (leastLossSubset (clusterLoss mq {(1 : S)} cn cd P cands ω) (cands.erase 1) (k - 1)) →
      ∀ v ∈ insert (1 : S)
        (leastLossSubset (clusterLoss mq {(1 : S)} cn cd P cands ω) (cands.erase 1) (k - 1)),
      clusterLoss mq {(1 : S)} cn cd P cands ω v
        ≤ clusterLoss mq {(1 : S)} cn cd P cands ω w := by
    intro w hw hwn v hv
    rcases Finset.mem_insert.1 hv with rfl | hv'
    · rw [clusterLoss_seed_zero mq hcd P cands ω hone]
      exact clusterLoss_nonneg mq _ cn cd P cands ω w
    · refine leastLossSubset_least _ (cands.erase 1) (k - 1) hkm v hv' w
        (Finset.mem_erase.2 ⟨fun hc => hwn (hc ▸ Finset.mem_insert_self _ _), hw⟩)
        (fun hc => hwn (Finset.mem_insert_of_mem hc))
  unfold lloydStep
  rw [if_pos hguard, Finset.card_insert_of_notMem (fun hc =>
    (Finset.mem_erase.1 (leastLossSubset_subset' _ _ _ hc)).1 rfl),
    leastLossSubset_card _ _ _ hkm]
  omega

open scoped Classical in
/-- A step from a `k`-member family keeps `k` members. -/
lemma lloydStep_card_keep (mq : S → Ω → ℝ) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (hone : (1 : S) ∈ cands) (hk : k ≤ cands.card) (hkpos : 0 < k) {F : Finset S}
    (hF : F.card = k) : (lloydStep mq cn cd P cands ω k F).card = k := by
  classical
  have hkm : k - 1 ≤ (cands.erase 1).card := by
    rw [Finset.card_erase_of_mem hone]; omega
  unfold lloydStep
  split_ifs
  · rw [Finset.card_insert_of_notMem (fun hc =>
      (Finset.mem_erase.1 (leastLossSubset_subset' _ _ _ hc)).1 rfl),
      leastLossSubset_card _ _ _ hkm]
    omega
  · exact hF

open scoped Classical in
/-- The clustering does not stall.  The seed's loss against its own column is zero, so
the first step is taken and every later one either keeps its `k` members or retakes `k`. -/
theorem clusterAround_card (mq : S → Ω → ℝ) {cn cd : ℕ} (hcd : cn < cd) (P cands : Finset S)
    (ω : Ω) (k : ℕ) (hone : (1 : S) ∈ cands) (hk : k ≤ cands.card) (hkpos : 0 < k) :
    (clusterAround mq cn cd P cands ω k).card = k := by
  classical
  unfold clusterAround
  have hiter : ∀ (n : ℕ) (F : Finset S), F.card = k →
      ((lloydStep mq cn cd P cands ω k)^[n] F).card = k := by
    intro n
    induction n with
    | zero => intro F hF; rwa [Function.iterate_zero_apply]
    | succ n ih =>
        intro F hF
        rw [Function.iterate_succ_apply]
        exact ih _ (lloydStep_card_keep mq cn cd P cands ω k hone hk hkpos hF)
  rw [show k * P.card + 1 = (k * P.card) + 1 from rfl, Function.iterate_succ_apply]
  exact hiter _ _ (lloydStep_seed_card mq hcd P cands ω k hone hk hkpos)

lemma leastLossSubset_card_le (l : S → ℝ) (cands : Finset S) (k : ℕ) :
    (leastLossSubset l cands k).card ≤ k := by
  unfold leastLossSubset
  split_ifs with h
  · exact le_of_eq (Finset.mem_powersetCard.1
      (Finset.exists_min_image (cands.powersetCard k) (fun T => ∑ x ∈ T, l x) h).choose_spec.1).2
  · simp

open scoped Classical in
/-- The family never outgrows the round. -/
lemma clusterAround_card_le (mq : S → Ω → ℝ) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (hk : 0 < k) : (clusterAround mq cn cd P cands ω k).card ≤ k := by
  classical
  unfold clusterAround
  have hstep : ∀ F : Finset S, F.card ≤ k → (lloydStep mq cn cd P cands ω k F).card ≤ k := by
    intro F hF
    unfold lloydStep
    split_ifs
    · refine le_trans (Finset.card_insert_le _ _) ?_
      have := leastLossSubset_card_le (clusterLoss mq F cn cd P cands ω) (cands.erase 1) (k - 1)
      omega
    · exact hF
  have hiter : ∀ (n : ℕ) (F : Finset S), F.card ≤ k →
      ((lloydStep mq cn cd P cands ω k)^[n] F).card ≤ k := by
    intro n
    induction n with
    | zero => intro F hF; rwa [Function.iterate_zero_apply]
    | succ n ih =>
        intro F hF
        rw [Function.iterate_succ_apply]
        exact ih _ (hstep F hF)
  exact hiter _ _ (by rw [Finset.card_singleton]; exact hk)

lemma lloydStep_subset (mq : S → Ω → ℝ) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (hone : (1 : S) ∈ cands) {F : Finset S} (hF : F ⊆ cands) :
    lloydStep mq cn cd P cands ω k F ⊆ cands := by
  classical
  unfold lloydStep
  split_ifs
  · exact Finset.insert_subset hone
      (le_trans (leastLossSubset_subset' _ _ _) (Finset.erase_subset _ _))
  · exact hF

lemma lloydIterate_subset (mq : S → Ω → ℝ) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (hone : (1 : S) ∈ cands) :
    ∀ (n : ℕ) (F : Finset S), F ⊆ cands → (lloydStep mq cn cd P cands ω k)^[n] F ⊆ cands := by
  intro n
  induction n with
  | zero => intro F hF; rw [Function.iterate_zero_apply]; exact hF
  | succ n ih =>
      intro F hF
      rw [Function.iterate_succ_apply]
      exact ih _ (lloydStep_subset mq cn cd P cands ω k hone hF)

lemma lloydIterate_congr (mq : S → Ω → ℝ) (cn cd : ℕ) (P cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands)
    {ω ω' : Ω} (h : ∀ w ∈ readSet P cands, (mq w ω = 1 ↔ mq w ω' = 1)) :
    ∀ (n : ℕ) (F : Finset S), F ⊆ cands →
      (lloydStep mq cn cd P cands ω k)^[n] F = (lloydStep mq cn cd P cands ω' k)^[n] F := by
  intro n
  induction n with
  | zero => intro F _; rw [Function.iterate_zero_apply, Function.iterate_zero_apply]
  | succ n ih =>
      intro F hF
      calc (lloydStep mq cn cd P cands ω k)^[n + 1] F
          = (lloydStep mq cn cd P cands ω k)^[n] (lloydStep mq cn cd P cands ω k F) :=
            Function.iterate_succ_apply _ _ _
        _ = (lloydStep mq cn cd P cands ω k)^[n] (lloydStep mq cn cd P cands ω' k F) :=
            congrArg (fun z => (lloydStep mq cn cd P cands ω k)^[n] z)
              (lloydStep_congr mq cn cd P cands k hF h)
        _ = (lloydStep mq cn cd P cands ω' k)^[n] (lloydStep mq cn cd P cands ω' k F) :=
            ih _ (lloydStep_subset mq cn cd P cands ω' k hone hF)
        _ = (lloydStep mq cn cd P cands ω' k)^[n + 1] F :=
            (Function.iterate_succ_apply _ _ _).symm

/-- The cluster reads only `readSet`.  Two noise draws agreeing at `p · v` for every
representative prefix and candidate suffix give the same family — so neither the family nor
any vote cast with it is decided by the oracle's bit at a bare prefix, which is the bit the
gate scores. -/
lemma clusterAround_congr_mq (mq : S → Ω → ℝ) (cn cd : ℕ) (P cands : Finset S) (k : ℕ)
    {ω ω' : Ω} (hone : (1 : S) ∈ cands)
    (h : ∀ w ∈ readSet P cands, (mq w ω = 1 ↔ mq w ω' = 1)) :
    clusterAround mq cn cd P cands ω k = clusterAround mq cn cd P cands ω' k :=
  lloydIterate_congr mq cn cd P cands k hone h _ _ (by simpa using hone)

lemma clusterAround_subset (mq : S → Ω → ℝ) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (hone : (1 : S) ∈ cands) : clusterAround mq cn cd P cands ω k ⊆ cands := by
  classical
  unfold clusterAround
  exact lloydIterate_subset mq cn cd P cands ω k hone _ _ (by simpa using hone)


/-- A read table as an oracle: the "noise" is the table itself. -/
noncomputable def tableRead (w : S) (reads : S → Prop) : ℝ := by
  classical
  exact if reads w then 1 else 0

lemma tableRead_eq_one (w : S) (reads : S → Prop) : tableRead w reads = 1 ↔ reads w := by
  classical
  unfold tableRead
  split_ifs with h <;> simp [h]

/-- Lloyd's iteration at the half-family centre, over the reads it is handed. -/
noncomputable def lloydClusterer : Clusterer S :=
  letI : MeasurableSpace (S → Prop) := ⊤
  { pick := fun reads P cands k => clusterAround tableRead 1 2 P cands reads k
    seed_mem := fun reads P cands k _ => one_mem_clusterAround _ 1 2 P cands reads k
    subset := fun reads P cands k hone => clusterAround_subset _ 1 2 P cands reads k hone
    card_le := fun reads P cands k hk => clusterAround_card_le _ 1 2 P cands reads k hk
    card_eq := fun reads P cands k hone hk hkpos =>
      clusterAround_card _ Nat.one_lt_two P cands reads k hone hk hkpos
    congr := fun reads reads' P cands k hone h => by
      refine clusterAround_congr_mq _ 1 2 P cands k hone (fun w hw => ?_)
      obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hw
      obtain ⟨hp, hv⟩ := Finset.mem_product.1 hpv
      rw [tableRead_eq_one, tableRead_eq_one]
      exact h p hp v hv }

open scoped Classical in
lemma clusterAround_eq_tableRead (mq : S → Ω → ℝ) (cn cd : ℕ) (P cands : Finset S) (ω : Ω)
    (k : ℕ) :
    clusterAround mq cn cd P cands ω k
      = letI : MeasurableSpace (S → Prop) := ⊤
        clusterAround tableRead cn cd P cands (fun w => mq w ω = 1) k := by
  have hloss : ∀ F, clusterLoss mq F cn cd P cands ω
      = letI : MeasurableSpace (S → Prop) := ⊤
        clusterLoss tableRead F cn cd P cands (fun w => mq w ω = 1) := by
    intro F
    funext v
    simp only [clusterLoss, hammingLoss, voteCount, tableRead_eq_one]
    congr
  have hstep : lloydStep mq cn cd P cands ω k
      = letI : MeasurableSpace (S → Prop) := ⊤
        lloydStep tableRead cn cd P cands (fun w => mq w ω = 1) k := by
    funext F
    simp only [lloydStep, hloss]
  unfold clusterAround
  rw [hstep]

lemma clusterAt_eq {J : Type*} (mq : S → Ω → ℝ) (populations : Finset J)
    (x : Run Ω S J) (B : State) :
    clusterAt mq populations x B = clusterBy lloydClusterer mq populations x B := by
  rw [clusterAt, clusterBy, clusterAround_eq_tableRead]
  rfl

lemma ret_eq {J : Type*} (mq : S → Ω → ℝ) (populations : Finset J)
    (indecisionLimit α : ℝ) (B : State) :
    ret (Ω := Ω) mq populations indecisionLimit α B
      = retBy lloydClusterer mq populations indecisionLimit α B := by
  have h : clusterAt (Ω := Ω) mq populations
      = fun x B => clusterBy lloydClusterer mq populations x B := by
    funext x B
    exact clusterAt_eq mq populations x B
  unfold ret
  rw [h]
  rfl
end OrthoDFA
