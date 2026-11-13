import RLStar.Proofs.LazyFail

/-!
# A lazy round's claim is fixed by the short reads

Over probes at most `L` long, the lazy round's claim reads the family only at `LazyKeyOff`
strings at most `max (L + T + 2) (probes + 1 + T + 2)` long, `T` the probe budget
(`lazyFail_congr_up`). The clustering reads at most three prefixes per population per prefix the
schedule draws, and at most one suffix more than the pool.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {K : ℕ}

section Reads

variable {σ : Type*} [Fintype σ] {Ω' : Type*} [MeasurableSpace Ω'] {μ' : Measure Ω'}
  [IsProbabilityMeasure μ']

/-- Over probes at least `k` and at most `L` long whose first `k` letters avoid `CP`'s, whether the
lazy round from `CP`'s trie fails its claim is fixed by the family's reads of the `LazyKeyOff`
strings at most `max (L + T + 2) (probes + 1 + T + 2)` long. -/
theorem lazyFail_congr_up (fam : SuffixFamily μ' α σ) (s : LoopSettings K)
    (CP : Finset (FreeMonoid α)) (D' : Measure (FreeMonoid α)) {L : ℕ}
    (hlen : ∀ᵐ x ∂D', x.toList.length ≤ L ∧ s.level3.walkStart ≤ x.toList.length
      ∧ Avoids s.level3.walkStart ↑(CP.image fun p => prefixOf p s.level3.walkStart) x)
    (A sl tw : ℝ)
    (d : Fin s.epochs → Random.EpochDraws α s.level3.probeBudget s.blocks s.blockSize)
    (hd : LazyDrawsAll (fun x => s.level3.walkStart ≤ x.toList.length
      ∧ Avoids s.level3.walkStart ↑(CP.image fun p => prefixOf p s.level3.walkStart) x) d)
    {ω ω' : Ω'}
    (h : ∀ z, LazyKeyOff s.level3.walkStart CP z →
      z.toList.length ≤ max (L + s.level3.probeBudget + 2)
        (lazyProbesLen d + 1 + s.level3.probeBudget + 2) → fam.read ω z = fam.read ω' z) :
    (ω, d) ∈ lazyFail fam s (Random.prefixTrie s.level3.walkStart CP) D' A sl tw →
      (ω', d) ∈ lazyFail fam s (Random.prefixTrie s.level3.walkStart CP) D' A sl tw := by
  sorry -- closes by: reads_lazyRoundQ_keyOff, Query.eval_congr; lazyDisagree's reads likewise

end Reads

section Prefs

lemma freshDraws_mono {S : Type*} [Stringlike S] (T : Finset S) (y : ℕ → S) {m m' : ℕ}
    (h : m ≤ m') (h' : ∃ n, m' ≤ (acceptedAt T y n).card) :
    freshDraws T y m ⊆ freshDraws T y m' := by
  have h'' : ∃ n, m ≤ (acceptedAt T y n).card := by
    obtain ⟨n, hn⟩ := h'; exact ⟨n, h.trans hn⟩
  rw [freshDraws_eq T y m h'', freshDraws_eq T y m' h']
  exact acceptedAt_mono T y (Nat.find_mono fun n hn => h.trans hn)

/-- Draws at the counts `n` to `2n` hold at most `2n` strings: the larger counts' draws extend the
smaller's, and a count the stream cannot fill draws nothing. -/
lemma card_biUnion_freshDraws_le {S : Type*} [Stringlike S] (T : Finset S) (y : ℕ → S) (n : ℕ) :
    ((Finset.range (n + 1)).biUnion fun e => freshDraws T y (n + e)).card ≤ 2 * n := by
  classical
  set P : ℕ → Prop := fun e => ∃ k, n + e ≤ (acceptedAt T y k).card
  set E := Nat.findGreatest P n
  by_cases hE : P E
  · have hsub : ((Finset.range (n + 1)).biUnion fun e => freshDraws T y (n + e))
        ⊆ freshDraws T y (n + E) := by
      intro v hv
      obtain ⟨e, he, hv⟩ := Finset.mem_biUnion.1 hv
      by_cases hPe : P e
      · have hle : e ≤ E := Nat.le_findGreatest (Nat.lt_succ_iff.1 (Finset.mem_range.1 he)) hPe
        exact freshDraws_mono T y (by omega) hE hv
      · rw [freshDraws_of_not T y _ hPe] at hv
        simp at hv
    refine (Finset.card_le_card hsub).trans ((card_freshDraws_le T y _).trans ?_)
    have := Nat.findGreatest_le (P := P) n
    omega
  · have hnone : ∀ e ≤ n, ¬ P e := fun e he hPe => hE (Nat.findGreatest_spec he hPe)
    have : ((Finset.range (n + 1)).biUnion fun e => freshDraws T y (n + e)) = ∅ := by
      refine Finset.eq_empty_of_forall_notMem fun v hv => ?_
      obtain ⟨e, he, hv⟩ := Finset.mem_biUnion.1 hv
      rw [freshDraws_of_not T y _ (hnone e (Nat.lt_succ_iff.1 (Finset.mem_range.1 he)))] at hv
      simp at hv
    rw [this]
    simp

variable {Ω : Type*} [MeasurableSpace Ω]

/-- The prefixes a round's clustering reads number at most three per population per prefix the
schedule draws; states at one prefix count read the same ones. -/
lemma card_roundPrefs_le (sched : Finset State) (pops : Finset (Option (Fin K)))
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) :
    (roundPrefs sched pops x).card ≤ 3 * pops.card * ∑ n ∈ sched.image State.npref, n := by
  classical
  set g : ℕ → Finset (FreeMonoid α) := fun n => prefixesAt pops n x
    ∪ pops.biUnion fun j => (Finset.range (n + 1)).biUnion fun e =>
      certOf (prefixesAt pops n x) j (n + e) x with hg
  have hR : roundPrefs sched pops x = (sched.image State.npref).biUnion g := by
    unfold roundPrefs
    ext p
    simp only [Finset.mem_biUnion, Finset.mem_image, hg]
    constructor
    · rintro ⟨B, hB, hp⟩; exact ⟨B.npref, ⟨B, hB, rfl⟩, hp⟩
    · rintro ⟨n, ⟨B, hB, rfl⟩, hp⟩; exact ⟨B, hB, hp⟩
  rw [hR]
  refine Finset.card_biUnion_le.trans ?_
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun n _ => (Finset.card_union_le _ _).trans ?_
  have h1 : (prefixesAt pops n x).card ≤ pops.card * n := by
    unfold prefixesAt
    refine Finset.card_biUnion_le.trans ?_
    calc ∑ j ∈ pops, (prefixesOf j n x).card ≤ ∑ _j ∈ pops, n :=
          Finset.sum_le_sum fun j _ => card_freshDraws_le _ _ _
      _ = _ := by rw [Finset.sum_const, smul_eq_mul]
  have h2 : (pops.biUnion fun j => (Finset.range (n + 1)).biUnion fun e =>
      certOf (prefixesAt pops n x) j (n + e) x).card ≤ pops.card * (2 * n) := by
    refine Finset.card_biUnion_le.trans ?_
    calc ∑ j ∈ pops, ((Finset.range (n + 1)).biUnion fun e =>
          certOf (prefixesAt pops n x) j (n + e) x).card
        ≤ ∑ _j ∈ pops, 2 * n :=
          Finset.sum_le_sum fun j _ => card_biUnion_freshDraws_le _ _ _
      _ = _ := by rw [Finset.sum_const, smul_eq_mul]
  nlinarith

/-- Where every state of the schedule draws `n` suffixes, the clustering reads at most `n + 1`. -/
lemma card_roundSuffs_le (sched : Finset State) {n : ℕ} (hn : ∀ B ∈ sched, B.nsuff = n)
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) : (roundSuffs sched x).card ≤ n + 1 := by
  classical
  have hsub : roundSuffs sched x ⊆ poolAt n x := by
    intro v hv
    obtain ⟨B, hB, hv⟩ := Finset.mem_biUnion.1 hv
    rwa [hn B hB] at hv
  refine (Finset.card_le_card hsub).trans ((Finset.card_insert_le _ _).trans ?_)
  exact Nat.succ_le_succ (card_freshDraws_le _ _ _)

end Prefs

end OrthoDFA
