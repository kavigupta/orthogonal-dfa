import RLStar.Proofs.LazyFail
import RLStar.Proofs.TrieTable
import Mathlib.Data.List.ReduceOption

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

omit [DecidableEq α] in
lemma LooseEndsWell.of_real {σ : Type*} {M : _root_.DFA α σ} {U : σ → ℝ} {A sl : ℝ}
    {D D' : Measure (FreeMonoid α)} {ε ε' : ℝ} {S S' : Set (FreeMonoid α)}
    (h : D'.real S' ≤ ε' → D.real S ≤ ε) {e : Random.LazyEnd α}
    (hw : LooseEndsWell M U A sl D' ε' S' e) : LooseEndsWell M U A sl D ε S e := by
  cases e with
  | consistent => exact h hw
  | harvest _ _ => exact hw
  | failed => exact hw

omit [DecidableEq α] in
private lemma unpad_prefixOf_padIn_mul (K : ℕ) (X : Finset (FreeMonoid α)) (x : FreeMonoid α)
    (j : ℕ) (m : FreeMonoid (Option α)) :
    Random.unpad (prefixOf (Random.padIn K X x) j * m)
      = prefixOf x (j - (K - Random.walkStartIn X x)) * Random.unpad m := by
  apply FreeMonoid.toList.injective
  simp only [Random.unpad, prefixOf, Random.padIn, FreeMonoid.toList_mul,
    FreeMonoid.toList_ofList, List.reduceOption_append, List.take_append, List.length_replicate,
    List.take_replicate, List.reduceOption_replicate_none, ← List.map_take]
  simp [List.reduceOption]

omit [Fintype α] [DecidableEq α] in
private lemma length_unpad_le (m : FreeMonoid (Option α)) :
    (Random.unpad m).toList.length ≤ m.toList.length := by
  simp only [Random.unpad, FreeMonoid.toList_ofList, List.reduceOption]
  exact List.length_filterMap_le _ _

/-- Over a probe whose first `k` letters avoid `CP`'s, walked from a trie holding `CP`'s, membership
in `lazyDisagree` reads only `LazyKeyOff` strings at most `n` longer than the probe, `n` bounding
the midfixes. -/
theorem mem_lazyDisagree_congr {k n : ℕ} {CP X : Finset (FreeMonoid α)}
    (hX : Random.prefixTrie k CP ⊆ X) {H : HypothesisDFA (Option α)}
    (hH : Random.MidsLe n H.tree) {read read' : FreeMonoid α → ARU} {x : FreeMonoid α}
    (hkx : k ≤ x.toList.length) (hx : Avoids k ↑(CP.image fun p => prefixOf p k) x)
    (h : ∀ z, LazyKeyOff k CP z → z.toList.length ≤ x.toList.length + n → read z = read' z) :
    x ∈ Random.lazyDisagree read k X H ↔ x ∈ Random.lazyDisagree read' k X H := by
  classical
  obtain ⟨hle, -, hout⟩ := Random.walkStartIn_spec X x
  have hp : probeOrAgree (Random.padRead read) H.tree H.edges k (Random.padIn k X x)
      = probeOrAgree (Random.padRead read') H.tree H.edges k (Random.padIn k X x) := by
    refine Random.probeOrAgree_congr_zone H.edges k fun j hkj _ m hm => ?_
    show read (Random.unpad _) = read' (Random.unpad _)
    rw [unpad_prefixOf_padIn_mul]
    set ws := Random.walkStartIn X x
    set j' := j - (k - ws)
    have hj' : min ws k ≤ j' := by omega
    have hmlen : (Random.unpad m).toList.length ≤ n := (length_unpad_le m).trans (hH m hm)
    have hlen : (prefixOf x j' * Random.unpad m).toList.length
        = min j' x.toList.length + (Random.unpad m).toList.length := by
      simp only [FreeMonoid.toList_mul, List.length_append, length_prefixOf]
    refine h _ ⟨min ws k, min_le_right _ _, by omega, fun p hp hpl hpe => ?_⟩ (by omega)
    have hpz : prefixOf (prefixOf x j' * Random.unpad m) (min ws k) = prefixOf x (min ws k) := by
      rw [prefixOf_mul_of_le (by rw [length_prefixOf]; omega),
        prefixOf_prefixOf_of_le x hj']
    rw [hpz] at hpe
    by_cases hws : ws < k
    · rw [min_eq_left hws.le] at hpe hpl
      refine hout (by omega) (hpe ▸ hX ?_)
      exact Finset.mem_biUnion.2 ⟨p, hp, Finset.mem_image.2 ⟨ws, Finset.mem_range.2 (by omega),
        rfl⟩⟩
    · rw [min_eq_right (not_lt.1 hws)] at hpe hpl
      exact hx hkx (Finset.mem_coe.2 (Finset.mem_image.2 ⟨p, hp, hpe⟩))
  simp only [Random.lazyDisagree, Set.mem_ofPred_eq, Set.mem_union, Random.searchSet,
    Random.endsSet, hp]

omit [DecidableEq α] in
private lemma topUpQ_retract_sub (read : FreeMonoid α → ARU) (H : HypothesisDFA (Option α))
    (s : Random.Level3Site) (K nh : ℕ) {b : ℕ} :
    ∀ (Bs : List (Fin b → FreeMonoid α)) (X : Finset (FreeMonoid α)) (ℓ : List (FreeMonoid α))
      {ℓ' : List (FreeMonoid α)} {X' : Finset (FreeMonoid α)}
      {rest : List (Fin b → FreeMonoid α)},
      (Random.topUpQ K nh H s Bs X ℓ).eval read = some (.retract ℓ' X', rest) → X ⊆ X'
  | [], _, _, _, _, _, h => by simp [Random.topUpQ] at h
  | B :: Bs, X, ℓ, ℓ', X', rest, h => by
    simp only [Random.topUpQ, Query.eval_bind] at h
    split_ifs at h with hfull hret
    · simp at h
    · simp only [Query.eval_pure, Option.some.injEq, Prod.mk.injEq,
        Random.TopUpEnd.retract.injEq] at h
      exact h.1.2 ▸ subset_rfl
    · exact Finset.subset_union_left.trans (topUpQ_retract_sub read H s K nh Bs _ _ h)

private lemma lazyEpochsQ_expanded_mids (read : FreeMonoid α → ARU) (C : Random.Level3Settings)
    (K nh : ℕ) {J b : ℕ} : ∀ (e pk : ℕ) (ds : List (Random.EpochDraws α C.probeBudget J b))
      (X : Finset (FreeMonoid α)) (ls : Random.LazyLists α),
      X ⊆ ((Random.lazyEpochsQ C K nh e pk ds X ls).eval read).expanded
        ∧ Random.MidsLe C.probeBudget
          ((Random.lazyEpochsQ C K nh e pk ds X ls).eval read).hypothesis.tree
  | _, _, [], X, ls => ⟨subset_rfl, fun m hm => by simp [Random.lazyEpochsQ, Random.mids] at hm⟩
  | e, pk, d :: ds, X, ls => by
    have hmid : Random.MidsLe C.probeBudget
        ((Random.epochQ (Random.epochSettings C e) K X d.1).eval read).hypothesis.tree := by
      have := (Random.runProbes_tree (fun z => (Random.unpadQ z).eval read)
        { Random.epochSettings C e with walkStart := K }
        (List.ofFn fun i => Random.padIn K X (d.1 i))).2.2
      rw [List.length_ofFn] at this
      simp only [Random.epochQ, Query.eval_subst]
      change Random.MidsLe _ (Random.indecisiveLStarLevel3 _ _ _).hypothesis.tree
      rw [Random.indecisiveLStarLevel3_tree]
      exact this
    simp only [Random.lazyEpochsQ, Query.eval_bind]
    generalize (Random.epochQ (Random.epochSettings C e) K X d.1).eval read = res at hmid ⊢
    rcases res with ⟨hyp, _ | s | _⟩
    · exact ⟨subset_rfl, hmid⟩
    · simp only [Query.eval_bind]
      have hT : ∀ ℓ' X' rest, (Random.topUpQ K nh hyp s (List.ofFn d.2) X (ls s)).eval read
          = some (.retract ℓ' X', rest) → X ⊆ X' := fun _ _ _ h =>
        topUpQ_retract_sub read hyp s K nh (List.ofFn d.2) X (ls s) h
      generalize (Random.topUpQ K nh hyp s (List.ofFn d.2) X (ls s)).eval read = o at hT ⊢
      rcases o with _ | ⟨_ | ⟨ℓ, X'⟩, rest⟩
      · exact ⟨subset_rfl, hmid⟩
      · exact ⟨subset_rfl, hmid⟩
      · obtain ⟨h1, h2⟩ := lazyEpochsQ_expanded_mids read C K nh (e + 1) _ ds X' _
        exact ⟨(hT _ _ _ rfl).trans h1, h2⟩
    · exact ⟨subset_rfl, hmid⟩

/-- A lazy round's trie holds its first, and its hypothesis's midfixes are at most `T` long. -/
theorem lazyRoundQ_expanded_mids (read : FreeMonoid α → ARU) (C : Random.Level3Settings)
    (K nh : ℕ) {E J b : ℕ} (X₀ : Finset (FreeMonoid α))
    (d : Fin E → Random.EpochDraws α C.probeBudget J b) :
    X₀ ⊆ ((Random.lazyRoundQ C K nh X₀ d).eval read).expanded
      ∧ Random.MidsLe C.probeBudget ((Random.lazyRoundQ C K nh X₀ d).eval read).hypothesis.tree :=
  lazyEpochsQ_expanded_mids read C K nh 0 0 (List.ofFn d) X₀ fun _ => []

variable {σ : Type*} [Fintype σ] {Ω' : Type*} [MeasurableSpace Ω'] {μ' : Measure Ω'}
  [IsProbabilityMeasure μ']

omit [IsProbabilityMeasure μ'] in
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
  classical
  set k := s.level3.walkStart
  set X₀ := Random.prefixTrie k CP
  have hreads := reads_lazyRoundQ_keyOff s.level3 k s.listSize CP d (fun z => fam.read ω z) rfl hd
  have hrun : lazyRun fam s X₀ (ω', d) = lazyRun fam s X₀ (ω, d) :=
    (Query.eval_congr (fun z => fam.read ω z) _ fun z hz =>
      (h z (hreads z hz).1 (le_max_of_le_right (hreads z hz).2)).symm).1
  obtain ⟨hsub, hmid⟩ := lazyRoundQ_expanded_mids (fun z => fam.read ω z) s.level3 k s.listSize X₀ d
  set r := lazyRun fam s X₀ (ω, d)
  have hdis : D'.real (Random.lazyDisagree (fun z => fam.read ω z) k r.expanded r.hypothesis)
      = D'.real (Random.lazyDisagree (fun z => fam.read ω' z) k r.expanded r.hypothesis) := by
    refine measureReal_congr ?_
    filter_upwards [hlen] with x hx
    exact propext (mem_lazyDisagree_congr hsub hmid hx.2.1 hx.2.2 fun z hz hzl =>
      h z hz (le_max_of_le_left (by omega)))
  intro hω
  simp only [lazyFail, Set.mem_ofPred_eq] at hω ⊢
  rw [hrun]
  exact fun hg => hω ⟨hg.1.of_real fun h' => hdis ▸ h', hg.2⟩

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

omit [DecidableEq α] [MeasurableSpace Ω] in
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

omit [DecidableEq α] [MeasurableSpace Ω] in
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
