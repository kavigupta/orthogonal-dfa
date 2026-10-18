import OrthoDFA.Proofs.Level5Cond
import OrthoDFA.Proofs.Level5Event

/-!
# The noise event past the short strings

Walked from `k`, a probe reads at its positions before `K` only `headStrings T k K`: strings of
length `k` to `K − 1`, each followed by a midfix. Its other reads begin with its first `K`
letters. So once the reads of `headStrings` are fixed, the probes' other reads are independent
across their length-`K` prefixes, and the edge and middles fields counted off `headStrings`
concentrate as before at `p₀`, the largest mass of a length-`K` prefix, whatever the fixed reads
are (`measure_le_of_fixed`). The start reads only `headStrings` when `k < K`. Where no string of
`headStrings` at a good read-state reads undecided, the fields off them are the fields
(`noiseAt_of_deep`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

attribute [local instance] ARU.fintype

section Head

/-- The strings of length `k` to `K − 1`, each followed by a midfix of `T`. -/
noncomputable def headStrings (T : DTree α) (k K : ℕ) : Finset (FreeMonoid α) :=
  ((Finset.Ico k K).biUnion lenK).biUnion fun u => T.midfixes.image (u * ·)

theorem mem_headStrings {T : DTree α} {k K : ℕ} {u m : FreeMonoid α} (hk : k ≤ u.toList.length)
    (hK : u.toList.length < K) (hm : m ∈ T.midfixes) : u * m ∈ headStrings T k K :=
  Finset.mem_biUnion.2 ⟨u, Finset.mem_biUnion.2 ⟨u.toList.length, Finset.mem_Ico.2 ⟨hk, hK⟩,
    mem_lenK.2 rfl⟩, Finset.mem_image_of_mem _ hm⟩

theorem headStrings_card_le (T : DTree α) (k K : ℕ) :
    (headStrings T k K).card ≤ (∑ i ∈ Finset.Ico k K, Fintype.card α ^ i) * T.midfixes.card := by
  unfold headStrings
  refine Finset.card_biUnion_le.trans ?_
  calc _ ≤ ∑ _u ∈ (Finset.Ico k K).biUnion lenK, T.midfixes.card :=
        Finset.sum_le_sum fun _ _ => Finset.card_image_le
    _ = ((Finset.Ico k K).biUnion lenK).card * T.midfixes.card := by
        rw [Finset.sum_const, smul_eq_mul]
    _ ≤ _ := by
        gcongr
        refine Finset.card_biUnion_le.trans (Finset.sum_le_sum fun i _ => ?_)
        unfold lenK
        exact Finset.card_image_le.trans (by simp)

theorem headStrings_empty {T : DTree α} {k K : ℕ} (h : K ≤ k) : headStrings T k K = ∅ := by
  simp [headStrings, Finset.Ico_eq_empty_of_le h]

/-- A probe's read off `headStrings` begins with its first `K` letters. -/
theorem prefix_of_notMem_head {T : DTree α} {k K : ℕ} {x z : FreeMonoid α}
    (hz : z ∈ probeStrings T k x) (hn : z ∉ headStrings T k K) : prefixOf z K = prefixOf x K := by
  obtain ⟨j, hj, hz⟩ := Finset.mem_biUnion.1 hz
  obtain ⟨m, hm, rfl⟩ := Finset.mem_image.1 hz
  obtain ⟨hkj, hjx⟩ := Finset.mem_Icc.1 hj
  have hlen : (prefixOf x j).toList.length = j := by simp [prefixOf]; omega
  by_cases hjK : j < K
  · exact absurd (mem_headStrings (by omega) (by omega) hm) hn
  apply FreeMonoid.toList.injective
  simp only [prefixOf, FreeMonoid.toList_ofList, FreeMonoid.toList_mul]
  rw [List.take_append_of_le_length (by simp; omega), List.take_take, min_eq_left (by omega)]

end Head

namespace ReadModel

variable {σ : Type*} (G : ReadModel α σ)

open scoped Classical in
/-- `undecAt` at good read-states, off the strings `Z`. -/
noncomputable def undecOff (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : EdgeTable α)
    (e : List Bool × α) (Z : Finset (FreeMonoid α)) (x : FreeMonoid α) : ℕ :=
  ((edgeHarvestBy (fun z => (rd z).cut) T edges k x e).filter
    fun b => G.Good (G.M.eval b.toList) ∧ b ∉ Z).length

/-- `startAt` at good read-states, off the strings `Z`. -/
def startOff (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (Z : Finset (FreeMonoid α)) :
    Set (FreeMonoid α) :=
  {x | k ≤ x.toList.length ∧ ∃ b, T.siftCut (fun z => (rd z).cut) (prefixOf x k) = .inr b
    ∧ G.Good (G.M.eval b.toList) ∧ b ∉ Z}

/-- `ptAt` at good read-states, off the strings `Z`. -/
def ptOff (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : EdgeTable α)
    (Z : Finset (FreeMonoid α)) : Set (FreeMonoid α) :=
  {x | ∃ b, middleHarvestBy (fun z => (rd z).cut) T edges k x = [b] ∧ G.Good (G.M.eval b.toList)
    ∧ b ∉ Z}

/-- No string of `Z` at a good read-state reads undecided. -/
def Decided (rd : FreeMonoid α → ARU) (Z : Finset (FreeMonoid α)) : Prop :=
  ∀ z ∈ Z, G.Good (G.M.eval z.toList) → rd z ≠ .undecided

end ReadModel

/-- The noise event at one tree and edge map. -/
structure NoiseAt {σ : Type*} (G : ReadModel α σ) (D : Measure (FreeMonoid α))
    (C : Level5Settings) (θg θgs θgpt : ℝ) (rd : FreeMonoid α → ARU) (T : DTree α)
    (edges : EdgeTable α) : Prop where
  goodEdge : ∀ e, ∫ x, (G.undecAt rd C.walkStart T edges e G.Good x : ℝ) ∂D
      ≤ θg * ∫ x, (edgeReadsBy (fun z => (rd z).cut) T edges C.walkStart x e : ℝ) ∂D
        + C.edgeTraffic * (C.edgeUndecidedRate - 8 * θg) / 4
  goodStart : D.real (G.startAt rd C.walkStart T G.Good) ≤ θgs
  goodPT : D.real (G.ptAt rd C.walkStart T edges G.Good)
      ≤ θgpt * D.real (searchAt rd C.walkStart T edges) + θgpt * C.searchShare / 2

theorem NoiseEvent.at {σ : Type*} {G : ReadModel α σ} {D : Measure (FreeMonoid α)}
    {C : Level5Settings} {S : ℕ} {θg θgs θgpt : ℝ} {rd : FreeMonoid α → ARU}
    (hE : NoiseEvent G D C S θg θgs θgpt rd) {T : DTree α} {edges : EdgeTable α}
    (hT : G.InClass S T) (he : EdgesInto T edges) : NoiseAt G D C θg θgs θgpt rd T edges :=
  ⟨hE.goodEdge T edges hT he, hE.goodStart T hT, hE.goodPT T edges hT he⟩

/-- The noise event off the strings each tree reads at positions before `K`. -/
structure DeepEvent {σ : Type*} (G : ReadModel α σ) (D : Measure (FreeMonoid α))
    (C : Level5Settings) (S K : ℕ) (θg θgs θgpt : ℝ) (rd : FreeMonoid α → ARU) : Prop where
  goodEdge : ∀ T edges, G.InClass S T → EdgesInto T edges → ∀ e,
    ∫ x, (G.undecOff rd C.walkStart T edges e (headStrings T C.walkStart K) x : ℝ) ∂D
      ≤ θg * ∫ x, (edgeReadsBy (fun z => (rd z).cut) T edges C.walkStart x e : ℝ) ∂D
        + C.edgeTraffic * (C.edgeUndecidedRate - 8 * θg) / 4
  goodStart : ∀ T, G.InClass S T →
    D.real (G.startOff rd C.walkStart T (headStrings T C.walkStart K)) ≤ θgs
  goodPT : ∀ T edges, G.InClass S T → EdgesInto T edges →
    D.real (G.ptOff rd C.walkStart T edges (headStrings T C.walkStart K))
      ≤ θgpt * D.real (searchAt rd C.walkStart T edges) + θgpt * C.searchShare / 2

section Full

variable {σ : Type*} (G : ReadModel α σ) {rd : FreeMonoid α → ARU} {Z : Finset (FreeMonoid α)}

theorem cut_of_mem_edgeHarvestBy {T : DTree α} {edges : EdgeTable α} {k : ℕ} {x b : FreeMonoid α}
    {e : List Bool × α} (hb : b ∈ edgeHarvestBy (fun z => (rd z).cut) T edges k x e) :
    rd b = .undecided := by
  unfold edgeHarvestBy at hb
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hb
  have := (List.mem_filter.1 hq).2
  simp only [Bool.decide_and, Bool.and_eq_true, decide_eq_true_eq] at this
  have h := this.2
  revert h
  cases rd q.2 <;> simp [ARU.cut]

theorem undecAt_eq_off (hdec : G.Decided rd Z) (k : ℕ) (T : DTree α) (edges : EdgeTable α)
    (e : List Bool × α) (x : FreeMonoid α) :
    G.undecAt rd k T edges e G.Good x = G.undecOff rd k T edges e Z x := by
  unfold ReadModel.undecAt ReadModel.undecOff
  congr 1
  refine List.filter_congr fun b hb => ?_
  have hu := cut_of_mem_edgeHarvestBy hb
  by_cases hg : G.Good (G.M.eval b.toList)
  · have hbZ : b ∉ Z := fun h => hdec b h hg hu
    simp [hg, hbZ]
  · simp [hg]

theorem startAt_eq_off (hdec : G.Decided rd Z) (k : ℕ) (T : DTree α) :
    G.startAt rd k T G.Good = G.startOff rd k T Z := by
  ext x
  simp only [ReadModel.startAt, ReadModel.startOff, Set.mem_ofPred_eq]
  constructor
  · rintro ⟨hk, b, hs, hg⟩
    refine ⟨hk, b, hs, hg, fun hbZ => hdec b hbZ hg ?_⟩
    obtain ⟨m, -, rfl, hc⟩ := siftCut_inr T _ b hs
    revert hc
    cases rd (prefixOf x k * m) <;> simp [ARU.cut]
  · rintro ⟨hk, b, hs, hg, -⟩
    exact ⟨hk, b, hs, hg⟩

theorem ptAt_eq_off (hdec : G.Decided rd Z) (k : ℕ) (T : DTree α) (edges : EdgeTable α) :
    G.ptAt rd k T edges G.Good = G.ptOff rd k T edges Z := by
  ext x
  simp only [ReadModel.ptAt, ReadModel.ptOff, Set.mem_ofPred_eq]
  constructor
  · rintro ⟨b, hb, hg⟩
    refine ⟨b, hb, hg, fun hbZ => hdec b hbZ hg ?_⟩
    unfold middleHarvestBy at hb
    rcases ho : probeBy (fun z => (rd z).cut) T edges k x with _ | _ | _ | j | _ | _ <;>
      rw [ho] at hb <;> simp only [reduceCtorEq] at hb
    all_goals first
      | (simp at hb)
      | skip
    all_goals
      rcases hs : T.siftCut (fun z => (rd z).cut) (prefixOf x j) with q | b' <;> rw [hs] at hb <;>
        simp only [Sum.elim_inl, Sum.elim_inr, List.cons.injEq, reduceCtorEq, and_true,
          List.nil_eq] at hb
      subst hb
      obtain ⟨m, -, rfl, hc⟩ := siftCut_inr T _ b' hs
      revert hc
      cases rd (prefixOf x j * m) <;> simp [ARU.cut]
  · rintro ⟨b, hb, hg, -⟩
    exact ⟨b, hb, hg⟩

/-- Where no string `T` reads before `K` at a good read-state reads undecided, the deep event is
the noise event at `T`. -/
theorem noiseAt_of_deep {D : Measure (FreeMonoid α)} {C : Level5Settings} {S K : ℕ}
    {θg θgs θgpt : ℝ} (hE : DeepEvent G D C S K θg θgs θgpt rd) {T : DTree α}
    {edges : EdgeTable α} (hT : G.InClass S T) (he : EdgesInto T edges)
    (hdec : G.Decided rd (headStrings T C.walkStart K)) : NoiseAt G D C θg θgs θgpt rd T edges := by
  refine ⟨fun e => ?_, ?_, ?_⟩
  · simp only [undecAt_eq_off G hdec]
    exact hE.goodEdge T edges hT he e
  · rw [startAt_eq_off G hdec]
    exact hE.goodStart T hT
  · rw [ptAt_eq_off G hdec]
    exact hE.goodPT T edges hT he

end Full

end OrthoDFA
