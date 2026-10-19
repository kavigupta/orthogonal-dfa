import OrthoDFA.Proofs.LoopRenoise

/-!
# Level 3's claim is fixed by the short reads

Over probes at most `L` long, level 3's claim reads the family only at strings at most
`max (L + T + 2) (probes + 1 + T + 2)` long, `T` the probes (`level3Fail_congr_up`). The clustering
reads at most three prefixes per population per prefix the schedule draws, and at most one suffix
more than the pool.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {K : ℕ}

section Reads

variable {σ : Type*} [Fintype σ] {Ω' : Type*} [MeasurableSpace Ω'] {μ' : Measure Ω'}
  [IsProbabilityMeasure μ']

/-- Over probes at most `L` long, whether level 3 fails its claim is fixed by the family's reads
of the strings at most `max (L + T + 2) (probes + 1 + T + 2)` long, `T` the probes. -/
theorem level3Fail_congr_up (fam : SuffixFamily μ' α σ) (C : Random.Level3Settings)
    (D' : Measure (FreeMonoid α)) {L : ℕ} (hlen : ∀ᵐ x ∂D', x.toList.length ≤ L) (θ u : ℝ)
    (xs : Fin C.probeBudget → FreeMonoid α) {ω ω' : Ω'}
    (h : ∀ z, z.toList.length ≤ max (L + C.probeBudget + 2)
      (probesLen xs + 1 + C.probeBudget + 2) → fam.read ω z = fam.read ω' z) :
    (ω, xs) ∈ level3Fail fam C D' θ u → (ω', xs) ∈ level3Fail fam C D' θ u := by
  classical
  have hlenP : ∀ x : FreeMonoid α, ∀ j, j ≤ x.toList.length → ∀ m : FreeMonoid α,
      (prefixOf x j * m).toList.length ≤ x.toList.length + m.toList.length := by
    intro x j _ m
    have : (prefixOf x j).toList.length ≤ x.toList.length := by simp [prefixOf]
    simp only [FreeMonoid.toList_mul, List.length_append]
    omega
  have hres : Random.indecisiveLStarLevel3 C (fun z => fam.read ω z) xs
      = Random.indecisiveLStarLevel3 C (fun z => fam.read ω' z) xs :=
    Random.level3_congr_zone C xs fun i j _ hj m hm => h _ (by
      have := hlenP (xs i) j hj m
      have := le_probesLen xs i
      omega)
  set H := (Random.indecisiveLStarLevel3 C (fun z => fam.read ω z) xs).hypothesis with hHd
  have hmid : Random.MidsLe C.probeBudget H.tree := by
    have := (Random.runProbes_tree (fun z => fam.read ω z) C (List.ofFn xs)).2.2
    rwa [List.length_ofFn] at this
  have hz : ∀ x : FreeMonoid α, x.toList.length ≤ L → ∀ j, j ≤ x.toList.length →
      ∀ m ∈ Random.mids H.tree, fam.read ω (prefixOf x j * m) = fam.read ω' (prefixOf x j * m) :=
    fun x hx j hj m hm => h _ (by
      have := hlenP x j hj m
      have := hmid m hm
      omega)
  have hnull : D' {x : FreeMonoid α | x.toList.length ≤ L}ᶜ = 0 := by
    rw [ae_iff] at hlen; convert hlen using 2; ext x; simp
  have hlaw := Random.harvestLaw_congr_zone (r := fun z => fam.read ω z)
    (r' := fun z => fam.read ω' z) D' C.walkStart H {x | x.toList.length ≤ L} hnull
    (fun x hx j _ hj m hm => hz x hx j hj m hm)
  have hdis : disagreeSet fam.M fam.readDist (fun z => fam.read ω z) C.walkStart H
      =ᵐ[D'] disagreeSet fam.M fam.readDist (fun z => fam.read ω' z) C.walkStart H := by
    filter_upwards [hlen] with x hx
    have hp : probeOrAgree (fun z => fam.read ω z) H.tree H.edges C.walkStart x
        = probeOrAgree (fun z => fam.read ω' z) H.tree H.edges C.walkStart x :=
      Random.probeOrAgree_congr_zone H.edges C.walkStart fun j _ hj m hm => hz x hx j hj m hm
    have hr : ∀ m ∈ Random.mids H.tree, fam.read ω (x * m) = fam.read ω' (x * m) := by
      intro m hm
      have := hz x hx x.toList.length le_rfl m hm
      rwa [prefixOf_length] at this
    have e1 : x ∈ Random.searchSet (fun z => fam.read ω z) C.walkStart H
        ↔ x ∈ Random.searchSet (fun z => fam.read ω' z) C.walkStart H := by
      simp only [Random.searchSet, Set.mem_ofPred_eq]; rw [hp]
    have e2 : x ∈ Random.endsSet (fun z => fam.read ω z) C.walkStart H
        ↔ x ∈ Random.endsSet (fun z => fam.read ω' z) C.walkStart H := by
      simp only [Random.endsSet, Set.mem_ofPred_eq]; rw [hp]
    have e3 : x ∈ rarerSet fam.M fam.readDist (fun z => fam.read ω z) C.walkStart H.tree
        ↔ x ∈ rarerSet fam.M fam.readDist (fun z => fam.read ω' z) C.walkStart H.tree := by
      constructor
      · rintro ⟨hk, m, hm, hw⟩
        exact ⟨hk, m, hm, by beta_reduce at hw ⊢; rw [← hr m hm]; exact hw⟩
      · rintro ⟨hk, m, hm, hw⟩
        exact ⟨hk, m, hm, by beta_reduce at hw ⊢; rw [hr m hm]; exact hw⟩
    apply propext
    show x ∈ disagreeSet fam.M fam.readDist (fun z => fam.read ω z) C.walkStart H
      ↔ x ∈ disagreeSet fam.M fam.readDist (fun z => fam.read ω' z) C.walkStart H
    simp only [disagreeSet, Set.mem_union, e1, e2, e3]
  have hsuff : ∀ E : RoundEnd Random.Level3Site,
      EndsWell
        { target := fam.M,
          badStates := fun q => ¬ fam.readDist q .undecided ≤ 3 / 2 * θ,
          probes := D',
          disagreeRate := C.disagreementRate + u,
          disagree := disagreeSet fam.M fam.readDist (fun z => fam.read ω' z) C.walkStart H,
          walkStart := C.walkStart,
          harvestLaw := Random.harvestLaw (fun z => fam.read ω' z) D' C.walkStart H,
          yield := C.harvestRate,
          ending := E } →
      EndsWell
        { target := fam.M,
          badStates := fun q => ¬ fam.readDist q .undecided ≤ 3 / 2 * θ,
          probes := D',
          disagreeRate := C.disagreementRate + u,
          disagree := disagreeSet fam.M fam.readDist (fun z => fam.read ω z) C.walkStart H,
          walkStart := C.walkStart,
          harvestLaw := Random.harvestLaw (fun z => fam.read ω z) D' C.walkStart H,
          yield := C.harvestRate,
          ending := E } := by
    intro E hE
    rcases E with _ | site | _
    · show D'.real _ ≤ _
      rw [measureReal_congr hdis]
      exact hE
    · simp only [EndsWell] at hE ⊢
      rw [hlaw site]
      exact hE
    · exact (show False from hE).elim
  intro hω
  simp only [level3Fail, Set.mem_ofPred_eq] at hω ⊢
  rw [← hres]
  exact fun hg => hω ⟨hsuff _ hg.1, hg.2⟩

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
schedule draws. -/
lemma card_roundPrefs_le (sched : Finset State) (pops : Finset (Option (Fin K)))
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) :
    (roundPrefs sched pops x).card ≤ 3 * pops.card * ∑ B ∈ sched, B.npref := by
  classical
  unfold roundPrefs
  refine Finset.card_biUnion_le.trans ?_
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun B _ => (Finset.card_union_le _ _).trans ?_
  have h1 : (prefixesAt pops B.npref x).card ≤ pops.card * B.npref := by
    unfold prefixesAt
    refine Finset.card_biUnion_le.trans ?_
    calc ∑ j ∈ pops, (prefixesOf j B.npref x).card ≤ ∑ _j ∈ pops, B.npref :=
          Finset.sum_le_sum fun j _ => card_freshDraws_le _ _ _
      _ = _ := by rw [Finset.sum_const, smul_eq_mul]
  have h2 : (pops.biUnion fun j => (Finset.range (B.npref + 1)).biUnion fun e =>
      certOf (prefixesAt pops B.npref x) j (B.npref + e) x).card ≤ pops.card * (2 * B.npref) := by
    refine Finset.card_biUnion_le.trans ?_
    calc ∑ j ∈ pops, ((Finset.range (B.npref + 1)).biUnion fun e =>
          certOf (prefixesAt pops B.npref x) j (B.npref + e) x).card
        ≤ ∑ _j ∈ pops, 2 * B.npref :=
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
