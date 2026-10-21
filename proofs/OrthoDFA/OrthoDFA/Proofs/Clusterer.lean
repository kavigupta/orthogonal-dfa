import OrthoDFA.Clustering

/-!
# Clustering rules

Validity holds for any rule that picks the family the way `Clusterer` says, and termination for
`nearestClusterer`, the rule `clusterAt` runs.
-/

namespace OrthoDFA

variable {Ω : Type*} [MeasurableSpace Ω] {S : Type*} [Stringlike S] {J : Type*} [Fintype J]

/-- A way of picking the family out of the screened candidates from their reads: `reads w` says
the oracle answered accept at `w`, and `P`, `cands` and `k` are the prefixes, the candidates and
the size asked for.

`identify_cluster_around` is one for each boundary it is handed and each state of `pst.rng`.
It reads only the candidates' columns on the representative prefixes, puts `ε` first, and
returns `count` rows once the pool holds that many.  How it groups the pool, which group it
anchors on, and how it fills the family are its own business, which is why the proof is carried
out for every such rule.  A boundary or generator state handed in from outside the round is
independent of the round's bits, so the claim holds with it fixed first. -/
structure Clusterer (S : Type*) [Stringlike S] where
  pick : (S → Prop) → Finset S → Finset S → ℕ → Finset S
  seed_mem : ∀ reads P cands k, (1 : S) ∈ cands → (1 : S) ∈ pick reads P cands k
  subset : ∀ reads P cands k, (1 : S) ∈ cands → pick reads P cands k ⊆ cands
  card_le : ∀ reads P cands k, 0 < k → (pick reads P cands k).card ≤ k
  card_eq : ∀ reads P cands k, (1 : S) ∈ cands → k ≤ cands.card → 0 < k →
    (pick reads P cands k).card = k
  congr : ∀ reads reads' P cands k, (1 : S) ∈ cands →
    (∀ p ∈ P, ∀ v ∈ cands, (reads (p * v) ↔ reads' (p * v))) →
    pick reads P cands k = pick reads' P cands k


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

section nearest
variable {S : Type*} [DecidableEq S]

lemma leastLossSubset_subset' (l : S → ℝ) (cands : Finset S) (k : ℕ) :
    leastLossSubset l cands k ⊆ cands := by
  classical
  unfold leastLossSubset
  split_ifs with hne
  · exact (Finset.mem_powersetCard.mp (Finset.exists_min_image _ _ hne).choose_spec.1).1
  · exact Finset.empty_subset _

lemma leastLossSubset_card_le (l : S → ℝ) (cands : Finset S) (k : ℕ) :
    (leastLossSubset l cands k).card ≤ k := by
  unfold leastLossSubset
  split_ifs with h
  · exact le_of_eq (Finset.mem_powersetCard.1
      (Finset.exists_min_image (cands.powersetCard k) (fun T => ∑ x ∈ T, l x) h).choose_spec.1).2
  · simp

/-- The chosen subset depends on the loss only on the candidates. -/
lemma leastLossSubset_congr {ℓ ℓ' : S → ℝ} {cands : Finset S} {k : ℕ}
    (h : ∀ x ∈ cands, ℓ x = ℓ' x) : leastLossSubset ℓ cands k = leastLossSubset ℓ' cands k := by
  have hf : ∀ T ∈ cands.powersetCard k, (∑ x ∈ T, ℓ x) = ∑ x ∈ T, ℓ' x := fun T hT =>
    Finset.sum_congr rfl (fun x hx => h x ((Finset.mem_powersetCard.1 hT).1 hx))
  unfold leastLossSubset
  split_ifs with hne
  · have hp : (fun T => T ∈ cands.powersetCard k ∧ ∀ T' ∈ cands.powersetCard k,
          (∑ x ∈ T, ℓ x) ≤ ∑ x ∈ T', ℓ x)
        = (fun T => T ∈ cands.powersetCard k ∧ ∀ T' ∈ cands.powersetCard k,
          (∑ x ∈ T, ℓ' x) ≤ ∑ x ∈ T', ℓ' x) := by
      funext T
      apply propext
      constructor
      · rintro ⟨hT, hmin⟩
        exact ⟨hT, fun T' hT' => by rw [← hf T hT, ← hf T' hT']; exact hmin T' hT'⟩
      · rintro ⟨hT, hmin⟩
        exact ⟨hT, fun T' hT' => by rw [hf T hT, hf T' hT']; exact hmin T' hT'⟩
    have key : ∀ (p q : Finset S → Prop) (hpq : p = q) (h1 : ∃ x, p x) (h2 : ∃ x, q x),
        h1.choose = h2.choose := by
      intro p q hpq h1 h2; subst hpq; rfl
    exact key _ _ (by simpa only [exists_prop] using hp) _ _
  · rfl

end nearest

open scoped Classical in
/-- How many prefixes of `P` the reads put `p · v` and `p` in different classes on. -/
noncomputable def readLoss (reads : S → Prop) (P : Finset S) (v : S) : ℝ :=
  ((P.filter (fun p => ¬ (reads (p * v) ↔ reads p))).card : ℝ)

open scoped Classical in
/-- `nearestRows` over the reads it is handed. -/
noncomputable def nearestClusterer : Clusterer S where
  pick := fun reads P cands k => insert 1 (leastLossSubset (readLoss reads P) (cands.erase 1) (k - 1))
  seed_mem := fun _ _ _ _ _ => Finset.mem_insert_self _ _
  subset := fun reads P cands k hone => Finset.insert_subset hone
    ((leastLossSubset_subset' _ _ _).trans (Finset.erase_subset _ _))
  card_le := fun reads P cands k hk => by
    refine le_trans (Finset.card_insert_le _ _) ?_
    have := leastLossSubset_card_le (readLoss reads P) (cands.erase 1) (k - 1)
    omega
  card_eq := fun reads P cands k hone hk hkpos => by
    have hc : k - 1 ≤ (cands.erase 1).card := by
      rw [Finset.card_erase_of_mem hone]; omega
    have h1 : (1 : S) ∉ leastLossSubset (readLoss reads P) (cands.erase 1) (k - 1) := fun h =>
      (Finset.notMem_erase 1 cands) (leastLossSubset_subset _ _ _ hc h)
    rw [Finset.card_insert_of_notMem h1, leastLossSubset_card _ _ _ hc]
    omega
  congr := fun reads reads' P cands k hone h => by
    have hl : ∀ v ∈ cands.erase 1, readLoss reads P v = readLoss reads' P v := by
      intro v hv
      unfold readLoss
      refine congrArg (fun t : Finset S => (t.card : ℝ)) (Finset.filter_congr (fun p hp => ?_))
      have h1 := h p hp v (Finset.mem_of_mem_erase hv)
      have h0 := h p hp 1 hone
      rw [mul_one] at h0
      rw [h1, h0]
    rw [leastLossSubset_congr hl]

/-- The cluster without its seed. -/
noncomputable def clusterBy (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J)
    (x : Run Ω S J) (B : State) : Finset S :=
  (rule.pick (fun w => mq w (oracleNoise x) = 1) (prefixesAt populations B.npref x)
    (screenedAt mq populations B x) B.k).erase 1

/-- `familyAt` with the family picked by `rule`. -/
noncomputable def familyBy (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J)
    (x : Run Ω S J) (B : State) : Finset S :=
  insert 1 (clusterBy rule mq populations x B)

/-- `refsAt` with the family picked by `rule`. -/
noncomputable def refsBy (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J)
    (x : Run Ω S J) (B : State) : Finset S :=
  poolAt B.nsuff x \ familyBy rule mq populations x B

/-- The gate at population `j`, at a look of `npref + e` draws, with the family picked by
`rule`. -/
def gateAt (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J) (uni : J)
    (η₀ εcov α : ℝ) (B : State) (j : J) (e : ℕ) (x : Run Ω S J) : Prop :=
  certifiesAt mq η₀ (gateLimit uni εcov j) (gateLevel α populations B) B.lo B.hi
    (clusterBy rule mq populations x B) (refsBy rule mq populations x B)
    (certOf (prefixesAt populations B.npref x) j (B.npref + e) x)
    (drawsTaken (prefixesAt populations B.npref x) (fun i => certPrefix j i x) (B.npref + e))
    (oracleNoise x)

open scoped Classical in
/-- `ret` with the family picked by `rule`. -/
noncomputable def retBy (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J) (uni : J)
    (η₀ indecisionLimit εcov α : ℝ) (B : State) : Set (Run Ω S J) :=
  {x | B.k ≤ (clusterBy rule mq populations x B).card + 1
    ∧ (∀ j ∈ populations,
      (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
          ¬ decided mq B.lo (B.hi + 1) (familyBy rule mq populations x B) p
            (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * (certOf (prefixesAt populations B.npref x) j B.npref x).card)
    ∧ ∀ j ∈ populations, ∃ e ≤ B.npref, gateAt rule mq populations uni η₀ εcov α B j e x}

open scoped Classical in
/-- `retBy` with every population's gate read at its first `npref` draws: what the loop is shown
to reach. -/
noncomputable def retAll (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J)
    (uni : J) (η₀ indecisionLimit εcov α : ℝ) (B : State) : Set (Run Ω S J) :=
  {x | B.k ≤ (clusterBy rule mq populations x B).card + 1
    ∧ (∀ j ∈ populations,
      (((certOf (prefixesAt populations B.npref x) j B.npref x).filter (fun p =>
          ¬ decided mq B.lo (B.hi + 1) (familyBy rule mq populations x B) p
            (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * (certOf (prefixesAt populations B.npref x) j B.npref x).card)
    ∧ ∀ j ∈ populations, gateAt rule mq populations uni η₀ εcov α B j 0 x}

lemma retAll_subset_retBy (rule : Clusterer S) (mq : S → Ω → ℝ) {populations : Finset J}
    {uni : J} (η₀ indecisionLimit εcov α : ℝ) (B : State) :
    retAll rule mq populations uni η₀ indecisionLimit εcov α B
      ⊆ retBy rule mq populations uni η₀ indecisionLimit εcov α B :=
  fun _ ⟨h1, h2, h3⟩ => ⟨h1, h2, fun j hj => ⟨0, Nat.zero_le _, h3 j hj⟩⟩

end OrthoDFA
