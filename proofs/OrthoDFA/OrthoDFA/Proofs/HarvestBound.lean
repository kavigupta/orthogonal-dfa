import OrthoDFA.Proofs.RoundAtK

/-!
# A harvest class's claim

`triple_holds_le`, for any class whose draws each run a computation `G` asking strings beginning
with their first `k` letters, and harvest the reads `harv` of its result, the first of which the
cut leaves undecided being a tagged first read of its string.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {β : Type*}

/-- What a class's computation and harvest must satisfy. -/
structure HarvestSpec (G : DTree α → Edges α → FreeMonoid α → Qry α β)
    (harv : β → List (FreeMonoid α)) (k : ℕ) : Prop where
  asks : ∀ t e x, (G t e x).AsksIn fun y => ∃ i, k ≤ i ∧ ∃ m ∈ t.mids, y = prefixOf x i * m
  first : ∀ (R : CutReads α) t e x, harv ((G t e x).run R.cut) ≠ [] →
    ∃ b ∈ harv ((G t e x).run R.cut), R.cut b = none
      ∧ ∃ r, ∃ hr : r < ((G t e x).trace R.cut).length, ((G t e x).trace R.cut)[r] = (b, true)
        ∧ ∀ i, ∀ hi : i < r, (((G t e x).trace R.cut)[i]'(hi.trans hr)).1 ≠ b
  form : ∀ (R : CutReads α) t e x b, b ∈ harv ((G t e x).run R.cut) →
    ∃ i, k ≤ i ∧ ∃ m, b = prefixOf x i * m

omit [Fintype α] [DecidableEq α] in
/-- In a trace whose untagged reads before a block are decided and whose block is tagged, an
undecided string read in the block is first read tagged. -/
theorem first_of_split (cut : FreeMonoid α → Option Bool)
    {pre blk post : List (FreeMonoid α × Bool)}
    (hpre : ∀ e ∈ pre, e.2 = false → cut e.1 ≠ none) (hblk : ∀ e ∈ blk, e.2 = true)
    {b : FreeMonoid α} (hb : (b, true) ∈ blk) (hcut : cut b = none) :
    ∃ r, ∃ hr : r < (pre ++ blk ++ post).length, (pre ++ blk ++ post)[r] = (b, true)
      ∧ ∀ i, ∀ hi : i < r, ((pre ++ blk ++ post)[i]'(hi.trans hr)).1 ≠ b := by
  classical
  set L := pre ++ blk ++ post
  have hex : ∃ e ∈ L, (fun e : FreeMonoid α × Bool => decide (e.1 = b)) e :=
    ⟨_, List.mem_append_left _ (List.mem_append_right _ hb), by simp⟩
  set r := L.findIdx fun e => decide (e.1 = b)
  have hr : r < L.length := List.findIdx_lt_length_of_exists hex
  have hrb : (L[r]).1 = b := by
    have := List.findIdx_getElem (w := hr)
    simpa using this
  refine ⟨r, hr, ?_, fun i hi => by simpa using List.not_of_lt_findIdx hi⟩
  refine Prod.ext hrb ?_
  by_contra hg
  rw [Bool.not_eq_true] at hg
  by_cases hlt : r < pre.length
  · have : L[r] = pre[r] := by
      simp only [L]
      rw [List.getElem_append_left (by simp; omega), List.getElem_append_left hlt]
    exact hpre _ (List.getElem_mem hlt) (this ▸ hg) (by rw [← this, hrb]; exact hcut)
  push Not at hlt
  obtain ⟨j, hj, hjb⟩ := List.mem_iff_getElem.1 hb
  have hpos : L[pre.length + j]'(by simp [L]; omega) = (b, true) := by
    simp only [L]
    rw [List.getElem_append_left (by simp; omega), List.getElem_append_right (by omega)]
    simpa using hjb
  have hrle : r ≤ pre.length + j := by
    by_contra hgt
    push Not at hgt
    have := List.not_of_lt_findIdx hgt
    rw [hpos] at this
    simp at this
  have hblkr : L[r] = blk[r - pre.length]'(by omega) := by
    simp only [L]
    rw [List.getElem_append_left (by simp; omega), List.getElem_append_right hlt]
  have htag := hblk _ (List.getElem_mem (l := blk) (n := r - pre.length) (by omega))
  rw [← hblkr, hg] at htag
  exact absurd htag (by decide)

omit [Fintype α] [DecidableEq α] in
/-- A first tagged read stays first when more reads follow. -/
theorem first_append {L E : List (FreeMonoid α × Bool)} {b : FreeMonoid α}
    (h : ∃ r, ∃ hr : r < L.length, L[r] = (b, true)
      ∧ ∀ i, ∀ hi : i < r, (L[i]'(hi.trans hr)).1 ≠ b) :
    ∃ r, ∃ hr : r < (L ++ E).length, (L ++ E)[r] = (b, true)
      ∧ ∀ i, ∀ hi : i < r, ((L ++ E)[i]'(hi.trans hr)).1 ≠ b := by
  obtain ⟨r, hr, hrb, hf⟩ := h
  refine ⟨r, by simp; omega, by rw [List.getElem_append_left hr]; exact hrb, fun i hi => ?_⟩
  rw [List.getElem_append_left (hi.trans hr)]
  exact hf i hi

section Fresh

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- `fresh_first_le`, for any computation `Gq` run on the hypothesis `st ω`. -/
theorem fresh_first_le_gen [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) (B : State)
    (F V : Finset (FreeMonoid α)) (hV : SuffixFree V) (hFV : F ⊆ V) {γ : Type*}
    (Gq : γ → Qry α β) (st : Ω → γ) (Tp : Ω → Finset (FreeMonoid α))
    (hst : ∀ ω ω', (∀ y ∈ vBits V (Tp ω), O.noise y ω = O.noise y ω') →
      st ω' = st ω ∧ Tp ω' = Tp ω)
    (C₀ : Set Ω)
    (hC₀ : ∀ ω ω', (∀ y ∈ vBits V (Tp ω), O.noise y ω = O.noise y ω') → (ω' ∈ C₀ ↔ ω ∈ C₀))
    (good : FreeMonoid α → Prop) {u : ℝ≥0∞}
    (hgood : ∀ z, good z → μ {ω | (readsAt O B F ω).cut z = none} ≤ u) :
    μ {ω | ω ∈ C₀ ∧ FirstBad ((Gq (st ω)).trace (readsAt O B F ω).cut) (Tp ω)
        (readsAt O B F ω).cut good}
      ≤ u * ∫⁻ ω, C₀.indicator
          (fun ω => ((((Gq (st ω)).trace (readsAt O B F ω).cut).countP (·.2) : ℕ) : ℝ≥0∞)) ω
          ∂μ := by
  classical
  set L : Ω → List (FreeMonoid α × Bool) := fun ω => (Gq (st ω)).trace (readsAt O B F ω).cut
    with hLdef
  set T : ℕ → Ω → Finset (FreeMonoid α) := fun r ω =>
    Tp ω ∪ (((L ω).take r).map Prod.fst).toFinset
  set Z : ℕ → Ω → Finset (FreeMonoid α) := fun r ω =>
    if ω ∈ C₀ then (((L ω)[r]?.filter (·.2)).map Prod.fst).toFinset else ∅
  set Fl : FreeMonoid α → Set Ω := fun z =>
    if good z then {ω | (readsAt O B F ω).cut z = none} else ∅
  have hFl : ∀ z, MeasurableSet[noiseAlg O ↑(F.image (z * ·))] (Fl z) := by
    intro z
    by_cases hz : good z
    · rw [show Fl z = {ω | (readsAt O B F ω).cut z = none} from if_pos hz]
      exact measurableSet_cut_none O B F z
    · rw [show Fl z = ∅ from if_neg hz]
      exact @MeasurableSet.empty Ω (noiseAlg O ↑(F.image (z * ·)))
  have hφ : ∀ z, μ (Fl z) ≤ u := by
    intro z
    by_cases hz : good z
    · simp only [Fl, if_pos hz]; exact hgood z hz
    · simp [Fl, if_neg hz]
  have hTZ : ∀ r ω ω', (∀ y ∈ vBits V (T r ω), O.noise y ω = O.noise y ω') →
      T r ω' = T r ω ∧ Z r ω' = Z r ω := by
    intro r ω ω' hag
    have hT : ∀ y ∈ vBits V (Tp ω), O.noise y ω = O.noise y ω' :=
      fun y hy => hag y (vBits_mono Finset.subset_union_left hy)
    obtain ⟨hst', hTp'⟩ := hst ω ω' hT
    have hC := hC₀ ω ω' hT
    have htake : (L ω').take (r + 1) = (L ω).take (r + 1) := by
      simp only [hLdef, hst']
      refine Qry.trace_congr _ r fun i hi hir => ?_
      have hi' : i < (L ω).length := hi
      refine cut_readsAt_congr O B hFV fun y hy => hag y (vBits_mono ?_ hy)
      intro w hw
      rw [Finset.mem_singleton] at hw
      subst hw
      refine Finset.mem_union_right _ ?_
      simp only [List.mem_toFinset, List.mem_map]
      exact ⟨_, List.mem_iff_getElem.2 ⟨i, by simp; omega, by simp [List.getElem_take]; rfl⟩, rfl⟩
    have hsplit : ∀ l : List (FreeMonoid α × Bool), l.take r = (l.take (r + 1)).take r :=
      fun l => by rw [List.take_take, min_eq_left (by omega)]
    have h1 : (L ω').take r = (L ω).take r := by
      rw [hsplit (L ω'), htake, ← hsplit]
    have h2 : (L ω')[r]? = (L ω)[r]? := by
      rw [← List.getElem?_take_of_lt (Nat.lt_succ_self r), htake,
        List.getElem?_take_of_lt (Nat.lt_succ_self r)]
    refine ⟨by simp only [T, hTp', h1], ?_⟩
    simp only [Z, h2]
    by_cases hω : ω ∈ C₀
    · rw [if_pos hω, if_pos (hC.2 hω)]
    · rw [if_neg hω, if_neg fun h => hω (hC.1 h)]
  have hsub : {ω | ω ∈ C₀ ∧ FirstBad (L ω) (Tp ω) (readsAt O B F ω).cut good}
      ⊆ ⋃ r, {ω | ∃ z ∈ Z r ω, z ∉ T r ω ∧ ω ∈ Fl z} := by
    rintro ω ⟨hC, r, hr, htag, hTp, hfirst, hcut, hgd⟩
    refine Set.mem_iUnion.2 ⟨r, (L ω)[r].1, ?_, ?_, ?_⟩
    · simp only [Z, if_pos hC, List.getElem?_eq_getElem hr, Option.filter, htag, if_true,
        Option.map_some, Option.toFinset_some, Finset.mem_singleton]
    · simp only [T, Finset.mem_union, List.mem_toFinset, List.mem_map, not_or, not_exists,
        not_and]
      refine ⟨hTp, fun e he hfe => ?_⟩
      obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 he
      simp only [List.length_take] at hi
      have := hfirst i (by omega)
      simp only [List.getElem_take] at hfe
      exact this hfe
    · simp only [Fl, if_pos hgd]; exact hcut
  have hcard : ∀ r ω, ((Z r ω \ T r ω).card : ℝ≥0∞)
      ≤ C₀.indicator (fun ω => if ((L ω)[r]?.any fun e : FreeMonoid α × Bool => e.2) = true
          then (1 : ℝ≥0∞) else 0) ω := by
    intro r ω
    refine (Nat.cast_le.2 (Finset.card_le_card Finset.sdiff_subset)).trans ?_
    by_cases hC : ω ∈ C₀
    · rw [Set.indicator_of_mem hC]
      simp only [Z, if_pos hC]
      rcases (L ω)[r]? with _ | ⟨w, _ | _⟩ <;> simp [Option.filter]
    · simp [Z, hC]
  calc μ {ω | ω ∈ C₀ ∧ FirstBad (L ω) (Tp ω) (readsAt O B F ω).cut good}
      ≤ μ (⋃ r, {ω | ∃ z ∈ Z r ω, z ∉ T r ω ∧ ω ∈ Fl z}) := measure_mono hsub
    _ ≤ ∑' r, μ {ω | ∃ z ∈ Z r ω, z ∉ T r ω ∧ ω ∈ Fl z} := measure_iUnion_le _
    _ ≤ ∑' r, u * ∫⁻ ω, ((Z r ω \ T r ω).card : ℝ≥0∞) ∂μ := ENNReal.tsum_le_tsum fun r =>
        cell_bound_rel O hV (T r) (Z r) (hTZ r) hFV Fl hFl hφ
    _ = u * ∑' r, ∫⁻ ω, ((Z r ω \ T r ω).card : ℝ≥0∞) ∂μ := ENNReal.tsum_mul_left
    _ ≤ u * ∫⁻ ω, ∑' r, C₀.indicator
          (fun ω => if ((L ω)[r]?.any fun e : FreeMonoid α × Bool => e.2) = true
            then (1 : ℝ≥0∞) else 0) ω ∂μ := by
        gcongr
        exact (ENNReal.tsum_le_tsum fun r => lintegral_mono (hcard r)).trans
          (tsum_lintegral_le _)
    _ = u * ∫⁻ ω, C₀.indicator
          (fun ω => (((L ω).countP (·.2) : ℕ) : ℝ≥0∞)) ω ∂μ := by
        congr 1
        refine lintegral_congr fun ω => ?_
        by_cases hC : ω ∈ C₀
        · simp only [Set.indicator_of_mem hC, tsum_tagged]
        · simp [Set.indicator_of_notMem hC]

end Fresh

section Class

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}
variable (G : DTree α → Edges α → FreeMonoid α → Qry α β) (harv : β → List (FreeMonoid α))

/-- The class's harvest is nonempty, of strings off `Tp` at states read undecided less than `u`
of the time. -/
def GFresh (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (u : ℝ) (cut : FreeMonoid α → Option Bool) (t : DTree α)
    (edges : Edges α) (Tp : Finset (FreeMonoid α)) (x : FreeMonoid α) : Prop :=
  harv ((G t edges x).run cut) ≠ []
    ∧ ∀ b ∈ harv ((G t edges x).run cut), stateIndecision A O B F (A.state b) < u ∧ b ∉ Tp

/-- How many of the class's reads are tagged. -/
noncomputable def gTags (cut : FreeMonoid α → Option Bool) (t : DTree α) (edges : Edges α)
    (x : FreeMonoid α) : ℕ :=
  ((G t edges x).trace cut).countP fun e => e.2

open scoped Classical in
/-- A draw's share of the class's fluctuation. -/
noncomputable def gContrib (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (u : ℝ) (cut : FreeMonoid α → Option Bool) (t : DTree α)
    (edges : Edges α) (Tp : Finset (FreeMonoid α)) (x : FreeMonoid α) : ℝ :=
  (if GFresh G harv A O B F u cut t edges Tp x then 1 else 0) - u * gTags G cut t edges x

variable {G harv} {k : ℕ}

theorem gFresh_firstBad (hG : HarvestSpec G harv k) (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) {u : ℝ}
    {R : CutReads α} {t : DTree α} {edges : Edges α}
    {Tp : Finset (FreeMonoid α)} {x : FreeMonoid α}
    (h : GFresh G harv A O B F u R.cut t edges Tp x) :
    FirstBad ((G t edges x).trace R.cut) Tp R.cut
      fun z => stateIndecision A O B F (A.state z) < u := by
  obtain ⟨b, hb, hcut, r, hr, hrb, hfirst⟩ := hG.first R t edges x h.1
  obtain ⟨hg, hT⟩ := h.2 b hb
  refine ⟨r, hr, by rw [hrb], by rw [hrb]; exact hT, fun i hi => by rw [hrb]; exact hfirst i hi,
    by rw [hrb]; exact hcut, by rw [hrb]; exact hg⟩

/-- Under the cell's reads, a draw's share depends only on the bits it can read off the cell's. -/
theorem measurable_gContrib_of [IsProbabilityMeasure μ] (hG : HarvestSpec G harv k)
    (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F V : Finset (FreeMonoid α)) (u : ℝ) (c : Finset (FreeMonoid α) × Finset (FreeMonoid α))
    (t : DTree α) (edges : Edges α) (Tp : Finset (FreeMonoid α)) (x : FreeMonoid α)
    (T : Set (FreeMonoid α)) (hT : ∀ w, (∃ i, k ≤ i ∧ ∃ m ∈ t.mids, w = prefixOf x i * m) →
      (↑(F.image (w * ·)) : Set (FreeMonoid α)) \ ↑(vBits V c.1) ⊆ T) :
    MeasurableSet[noiseAlg O T]
        {ω | GFresh G harv A O B F u (cellReads O B F V c ω).cut t edges Tp x}
      ∧ (∀ n, MeasurableSet[noiseAlg O T]
        {ω | gTags G (cellReads O B F V c ω).cut t edges x = n})
      ∧ Measurable[noiseAlg O T]
        fun ω => gContrib G harv A O B F u (cellReads O B F V c ω).cut t edges Tp x := by
  classical
  set cut : Ω → FreeMonoid α → Option Bool := fun ω => (cellReads O B F V c ω).cut
  have hS : ∀ w, (∃ i, k ≤ i ∧ ∃ m ∈ t.mids, w = prefixOf x i * m) →
      ∀ o, MeasurableSet[noiseAlg O T] {ω | cut ω w = o} :=
    fun w hw o => noiseAlg_mono O (hT w hw) _ (measurableSet_cellCut O B F V c w o)
  have hrt : ∀ A' : Set (β × List (FreeMonoid α × Bool)),
      MeasurableSet[noiseAlg O T] {ω | ((G t edges x).run (cut ω),
        (G t edges x).trace (cut ω)) ∈ A'} :=
    Qry.measurableSet_run_trace cut hS _ (hG.asks t edges x)
  have hFT : MeasurableSet[noiseAlg O T]
      {ω | GFresh G harv A O B F u (cellReads O B F V c ω).cut t edges Tp x} :=
    hrt {p | harv p.1 ≠ [] ∧ ∀ b ∈ harv p.1, stateIndecision A O B F (A.state b) < u ∧ b ∉ Tp}
  have hcount : ∀ n, MeasurableSet[noiseAlg O T]
      {ω | gTags G (cellReads O B F V c ω).cut t edges x = n} := fun n =>
    hrt {p | p.2.countP (fun e => e.2) = n}
  have hpair : Measurable[noiseAlg O T] fun ω =>
      (decide (GFresh G harv A O B F u (cellReads O B F V c ω).cut t edges Tp x),
        gTags G (cellReads O B F V c ω).cut t edges x) := by
    refine @measurable_to_countable' _ _ _ _ (noiseAlg O T) _ fun y => ?_
    have : (fun ω => (decide (GFresh G harv A O B F u (cellReads O B F V c ω).cut t edges Tp x),
        gTags G (cellReads O B F V c ω).cut t edges x)) ⁻¹' {y}
        = {ω | decide (GFresh G harv A O B F u (cellReads O B F V c ω).cut t edges Tp x) = y.1}
          ∩ {ω | gTags G (cellReads O B F V c ω).cut t edges x = y.2} := by
      ext ω; simp [Prod.ext_iff]
    rw [this]
    refine MeasurableSet.inter ?_ (hcount y.2)
    obtain ⟨b, n⟩ := y
    cases b
    · have h' : {ω | decide (GFresh G harv A O B F u (cellReads O B F V c ω).cut t edges Tp x)
          = false} = {ω | GFresh G harv A O B F u (cellReads O B F V c ω).cut t edges Tp x}ᶜ := by
        ext ω; simp
      rw [h']; exact hFT.compl
    · have h' : {ω | decide (GFresh G harv A O B F u (cellReads O B F V c ω).cut t edges Tp x)
          = true} = {ω | GFresh G harv A O B F u (cellReads O B F V c ω).cut t edges Tp x} := by
        ext ω; simp
      rw [h']; exact hFT
  have hg : Measurable fun p : Bool × ℕ => (if p.1 then (1 : ℝ) else 0) - u * p.2 :=
    measurable_of_countable _
  refine ⟨hFT, hcount, ?_⟩
  have := hg.comp hpair
  convert this using 1
  ext ω
  simp [gContrib]

theorem image_sub_drawBits (F V : Finset (FreeMonoid α))
    (c : Finset (FreeMonoid α) × Finset (FreeMonoid α)) (k : ℕ) (x : FreeMonoid α)
    {w : FreeMonoid α} (hw : ∃ i, k ≤ i ∧ ∃ m, w = prefixOf x i * m) :
    (↑(F.image (w * ·)) : Set (FreeMonoid α)) \ ↑(vBits V c.1) ⊆ drawBits V c k x := by
  rintro z ⟨hz, hzv⟩
  refine ⟨?_, hzv⟩
  obtain ⟨i, hi, m, rfl⟩ := hw
  simp only [Finset.coe_image, Set.mem_image, Finset.mem_coe] at hz
  obtain ⟨v, -, rfl⟩ := hz
  simp only [Set.mem_ofPred_eq, FreeMonoid.toList_mul, prefixOf, FreeMonoid.toList_ofList]
  exact (List.take_prefix_take_left hi).trans (List.prefix_append _ _ |>.trans
    (List.prefix_append _ _))

/-- Under the cell's reads, a draw's share depends only on the bits beginning with its first `k`
letters. -/
theorem measurable_gContrib [IsProbabilityMeasure μ] (hG : HarvestSpec G harv k)
    (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F V : Finset (FreeMonoid α)) (u : ℝ) (c : Finset (FreeMonoid α) × Finset (FreeMonoid α))
    (t : DTree α) (edges : Edges α) (Tp : Finset (FreeMonoid α)) (x : FreeMonoid α) :
    MeasurableSet[noiseAlg O (drawBits V c k x)]
        {ω | GFresh G harv A O B F u (cellReads O B F V c ω).cut t edges Tp x}
      ∧ (∀ n, MeasurableSet[noiseAlg O (drawBits V c k x)]
        {ω | gTags G (cellReads O B F V c ω).cut t edges x = n})
      ∧ Measurable[noiseAlg O (drawBits V c k x)]
        fun ω => gContrib G harv A O B F u (cellReads O B F V c ω).cut t edges Tp x :=
  measurable_gContrib_of hG A O B F V u c t edges Tp x _
    fun w ⟨i, hi, m, _, he⟩ => image_sub_drawBits F V c k x ⟨i, hi, m, he⟩

end Class

section GenPass

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
variable (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (K : StageKnobs α)
  (k : ℕ) (seed probes : List (FreeMonoid α)) (Pf : CutReads α → KState α)

/-- The pass `Pf` on noise `ω`. -/
noncomputable def passOf (ω : Ω) : KState α := Pf (readsAt O B F ω)

/-- The strings the pass `Pf` on noise `ω` can read. -/
noncomputable def readsOf (ω : Ω) : Finset (FreeMonoid α) :=
  passReadSet k seed probes (passOf O B F Pf ω).tree

/-- The cell of the pass `Pf`: its read set `c.1`, with those reads' bits patterned `c.2`. -/
def cellOf (c : Finset (FreeMonoid α) × Finset (FreeMonoid α)) : Set Ω :=
  pcell O (K.suffixes F) c ∩ cleanAll O ∩ {ω | readsOf O B F k seed probes Pf ω = c.1}

/-- The pass `Pf` is decided by the oracle's bits at what it can read against the tree it ends
with. -/
def PassDetermined : Prop :=
  ∀ ω ω' : Ω, (∀ y ∈ vBits (K.suffixes F) (readsOf O B F k seed probes Pf ω),
    O.noise y ω = O.noise y ω') → passOf O B F Pf ω' = passOf O B F Pf ω

theorem runPassK_passDetermined :
    PassDetermined (μ := μ) O B F K k seed probes
      fun R => runPassK K R k (initialK K R seed) probes :=
  fun ω ω' h => runPassK_determined O B F K k seed probes ω ω' h

variable {O B F K k seed probes Pf}

theorem cellOf_const [IsProbabilityMeasure μ] (hD : PassDetermined O B F K k seed probes Pf)
    {c : Finset (FreeMonoid α) × Finset (FreeMonoid α)} {ω₀ ω : Ω}
    (h₀ : ω₀ ∈ cellOf O B F K k seed probes Pf c) (hω : ω ∈ pcell O (K.suffixes F) c)
    (hcl : ω ∈ cleanAll O) :
    passOf O B F Pf ω = passOf O B F Pf ω₀ ∧ ω ∈ cellOf O B F K k seed probes Pf c := by
  obtain ⟨⟨⟨hp₀, hc₀⟩, -⟩, hT₀⟩ := h₀
  have hT₀' : readsOf O B F k seed probes Pf ω₀ = c.1 := hT₀
  have hag : ∀ y ∈ vBits (K.suffixes F) (readsOf O B F k seed probes Pf ω₀),
      O.noise y ω₀ = O.noise y ω := by
    rw [hT₀']
    exact noise_eq_of_pattern O hc₀ hω.2 (hp₀.trans hω.1.symm)
  have hs := hD ω₀ ω hag
  refine ⟨hs, ⟨⟨hω, hcl⟩, ?_⟩⟩
  change readsOf O B F k seed probes Pf ω = c.1
  rw [← hT₀']
  simp only [readsOf, hs]

theorem measurableSet_cellOf [IsProbabilityMeasure μ] (hD : PassDetermined O B F K k seed probes Pf)
    (c : Finset (FreeMonoid α) × Finset (FreeMonoid α)) :
    MeasurableSet (cellOf O B F K k seed probes Pf c) := by
  by_cases h : (cellOf O B F K k seed probes Pf c).Nonempty
  · obtain ⟨ω₀, h₀⟩ := h
    have he : cellOf O B F K k seed probes Pf c = pcell O (K.suffixes F) c ∩ cleanAll O := by
      ext ω; constructor
      · intro hω; exact ⟨hω.1.1, hω.1.2⟩
      · rintro ⟨hP, hc'⟩; exact (cellOf_const hD h₀ hP hc').2
    rw [he]
    exact (noiseAlg_le O _ _ ((measurableSet_noisePattern O _ c.2).inter
      (measurableSet_noiseClean O _))).inter (measurableSet_cleanAll O)
  · rw [Set.not_nonempty_iff_eq_empty.1 h]; exact MeasurableSet.empty

end GenPass

section Cells

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}
variable (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (K : StageKnobs α)
  (k : ℕ) (seed probes : List (FreeMonoid α)) (Pf : CutReads α → KState α)
variable {G : DTree α → Edges α → FreeMonoid α → Qry α β} {harv : β → List (FreeMonoid α)}

/-- Inside a cell of the pass, a draw's share is pulled down on average: its fresh harvests are
outnumbered by `uGood` times its tagged reads. -/
theorem cell_mean_le_gen [IsProbabilityMeasure μ] (hG : HarvestSpec G harv k)
    (bnd : DTree α → ℕ → ℕ)
    (hbnd : ∀ (R : CutReads α) t e x, gTags G R.cut t e x ≤ bnd t x.toList.length)
    (A : DFA (FreeMonoid α) Q) {uGood : ℝ}
    (hu : 0 ≤ uGood) (hV : SuffixFree (K.suffixes F))
    {c : Finset (FreeMonoid α) × Finset (FreeMonoid α)} {ω₀ : Ω}
    (hD : PassDetermined O B F K k seed probes Pf)
    (h₀ : ω₀ ∈ cellOf O B F K k seed probes Pf c) (x : FreeMonoid α) :
    μ.real (pcell O (K.suffixes F) c)
      * ∫ ω, gContrib G harv A O B F uGood (cellReads O B F (K.suffixes F) c ω).cut
          (passOf O B F Pf ω₀).tree (passOf O B F Pf ω₀).edges c.1 x ∂μ
      ≤ 0 := by
  classical
  set V := K.suffixes F
  set s₀ := passOf O B F Pf ω₀
  set P := pcell O V c
  set C := cellOf O B F K k seed probes Pf c
  set FTc : Ω → Prop := fun ω =>
    GFresh G harv A O B F uGood (cellReads O B F V c ω).cut s₀.tree s₀.edges c.1 x
  set tc : Ω → ℕ := fun ω => gTags G (cellReads O B F V c ω).cut s₀.tree s₀.edges x
  set h : Ω → ℝ := fun ω =>
    gContrib G harv A O B F uGood (cellReads O B F V c ω).cut s₀.tree s₀.edges c.1 x
  obtain ⟨hFTm, htcm, hhm⟩ := measurable_gContrib hG A O B F V uGood c s₀.tree s₀.edges c.1 x
  have hle := noiseAlg_le O
  have hFTm' : MeasurableSet {ω | FTc ω} := hle _ _ hFTm
  have htcm' : Measurable fun ω => (tc ω : ℝ) := by
    refine Measurable.comp (g := fun n : ℕ => (n : ℝ)) measurable_from_top ?_
    exact measurable_to_countable' fun n => hle _ _ (htcm n)
  have hhm' : Measurable h := hhm.mono (hle _) le_rfl
  have hPm : MeasurableSet[noiseAlg O ↑(vBits V c.1)] P :=
    (measurableSet_noisePattern O _ c.2).inter (measurableSet_noiseClean O _)
  have hPm' : MeasurableSet P := hle _ _ hPm
  have hbound : ∀ ω, (tc ω : ℝ) ≤ bnd s₀.tree x.toList.length := fun ω => by
    exact_mod_cast hbnd _ _ _ x
  have hint_tc : Integrable (fun ω => (tc ω : ℝ)) μ :=
    Integrable.of_bound htcm'.aestronglyMeasurable _ (ae_of_all _ fun ω => by
      rw [Real.norm_of_nonneg (Nat.cast_nonneg _)]; exact hbound ω)
  have hint_h : Integrable h μ :=
    Integrable.of_bound hhm'.aestronglyMeasurable (1 + uGood * bnd s₀.tree x.toList.length)
      (ae_of_all _ fun ω => by
        have h0 : (0 : ℝ) ≤ tc ω := Nat.cast_nonneg _
        have h1 := hbound ω
        simp only [h, gContrib, Real.norm_eq_abs]
        split_ifs <;> rw [abs_le] <;> constructor <;> nlinarith)
  -- `P` is independent of the draw's share
  have hdisj : Disjoint (↑(vBits V c.1) : Set (FreeMonoid α)) (drawBits V c k x) :=
    Set.disjoint_left.2 fun z hz hz' => hz'.2 hz
  have hPi : Measurable[noiseAlg O ↑(vBits V c.1)] (P.indicator (1 : Ω → ℝ)) :=
    Measurable.indicator measurable_const hPm
  have hind : (P.indicator (1 : Ω → ℝ)) ⟂ᵢ[μ] h := indepFun_of_noiseAlg O hdisj hPi hhm
  have e1 : ∫ ω, P.indicator 1 ω * h ω ∂μ = μ.real P * ∫ ω, h ω ∂μ := by
    have := hind.integral_mul_eq_mul_integral (hPi.mono (hle _) le_rfl).aestronglyMeasurable
      hhm'.aestronglyMeasurable
    simp only [Pi.mul_apply] at this
    rw [this, integral_indicator_one hPm']
  -- `P` and the cell agree off a null set
  have hCP : C ⊆ P := fun ω hω => hω.1.1
  have hcl := measure_cleanAll_compl O (μ := μ)
  have hae : ∀ᵐ ω ∂μ, P.indicator (1 : Ω → ℝ) ω = C.indicator 1 ω := by
    have : ∀ᵐ ω ∂μ, ω ∈ cleanAll O := ae_iff.2 hcl
    filter_upwards [this] with ω hω
    by_cases hP : ω ∈ P
    · rw [Set.indicator_of_mem hP,
        Set.indicator_of_mem (cellOf_const hD h₀ hP hω).2]
    · rw [Set.indicator_of_notMem hP, Set.indicator_of_notMem fun h => hP (hCP h)]
  have e2 : ∫ ω, P.indicator 1 ω * h ω ∂μ = ∫ ω, C.indicator 1 ω * h ω ∂μ :=
    integral_congr_ae (hae.mono fun ω hω => by
      change P.indicator 1 ω * h ω = C.indicator 1 ω * h ω
      rw [hω])
  -- on the cell, the cell's reads are the oracle's
  have hon : ∀ ω ∈ C, cellReads O B F V c ω = readsAt O B F ω
      ∧ passOf O B F Pf ω = s₀ ∧ readsOf O B F k seed probes Pf ω = c.1 := by
    intro ω hω
    exact ⟨cellReads_eq O B F V c hω.1.1.1 hω.1.2,
      (cellOf_const hD h₀ hω.1.1 hω.1.2).1, hω.2⟩
  -- the fresh triples, by `fresh_first_le`
  set st : Ω → DTree α × Edges α := fun ω =>
    ((passOf O B F Pf ω).tree, (passOf O B F Pf ω).edges)
  set C₀ := P ∩ {ω | readsOf O B F k seed probes Pf ω = c.1}
  have hst : ∀ ω ω', (∀ y ∈ vBits V (readsOf O B F k seed probes Pf ω),
      O.noise y ω = O.noise y ω') → st ω' = st ω
        ∧ readsOf O B F k seed probes Pf ω' = readsOf O B F k seed probes Pf ω := by
    intro ω ω' hag
    have hs := hD ω ω' hag
    exact ⟨by simp only [st, hs], by simp only [readsOf, hs]⟩
  have hC₀ : ∀ ω ω', (∀ y ∈ vBits V (readsOf O B F k seed probes Pf ω),
      O.noise y ω = O.noise y ω') → (ω' ∈ C₀ ↔ ω ∈ C₀) := by
    intro ω ω' hag
    have hT := (hst ω ω' hag).2
    by_cases hω : readsOf O B F k seed probes Pf ω = c.1
    · rw [hω] at hag
      have hpat : noisePattern O (vBits V c.1) ω' = noisePattern O (vBits V c.1) ω :=
        Finset.filter_congr fun y hy => by rw [hag y hy]
      have hcln : ω' ∈ noiseClean O (vBits V c.1) ↔ ω ∈ noiseClean O (vBits V c.1) := by
        simp only [noiseClean, Set.mem_ofPred_eq]
        exact forall₂_congr fun y hy => by rw [hag y hy]
      simp only [C₀, P, pcell, Set.mem_inter_iff, Set.mem_ofPred_eq, hpat, hcln, hT]
    · simp only [C₀, Set.mem_inter_iff, Set.mem_ofPred_eq, hT, hω, and_false]
  have hfresh := fresh_first_le_gen O B F V hV (K.family_sub F)
    (fun p : DTree α × Edges α => G p.1 p.2 x) st (readsOf O B F k seed probes Pf) hst C₀ hC₀
    (fun z => stateIndecision A O B F (A.state z) < uGood) (u := ENNReal.ofReal uGood)
    fun z hz => good_le O B F A z hz
  have hsub : C ∩ {ω | FTc ω} ⊆ {ω | ω ∈ C₀
      ∧ FirstBad ((G (st ω).1 (st ω).2 x).trace (readsAt O B F ω).cut)
      (readsOf O B F k seed probes Pf ω) (readsAt O B F ω).cut
      fun z => stateIndecision A O B F (A.state z) < uGood} := by
    rintro ω ⟨hω, hft⟩
    obtain ⟨hr, hs, hT⟩ := hon ω hω
    refine ⟨⟨hω.1.1, hT⟩, ?_⟩
    have hft' : GFresh G harv A O B F uGood (readsAt O B F ω).cut (st ω).1 (st ω).2
        (readsOf O B F k seed probes Pf ω) x := by
      simp only [st, hs, hT]; simpa [FTc, hr] using hft
    exact gFresh_firstBad hG A O B F hft'
  have hlin : ∫⁻ ω, C₀.indicator (fun ω =>
      ((((G (st ω).1 (st ω).2 x).trace (readsAt O B F ω).cut).countP (·.2) : ℕ)
      : ℝ≥0∞)) ω ∂μ = ∫⁻ ω, ENNReal.ofReal (C.indicator (fun ω => (tc ω : ℝ)) ω) ∂μ := by
    have : ∀ᵐ ω ∂μ, ω ∈ cleanAll O := ae_iff.2 hcl
    refine lintegral_congr_ae (this.mono fun ω hcl' => ?_)
    beta_reduce
    by_cases hω : ω ∈ C₀
    · have hC : ω ∈ C := (cellOf_const hD h₀ hω.1 hcl').2
      obtain ⟨hr, hs, -⟩ := hon ω hC
      rw [Set.indicator_of_mem hω, Set.indicator_of_mem hC, ENNReal.ofReal_natCast]
      simp only [st, hs, tc, gTags, hr]
    · have hC : ω ∉ C := fun h => hω ⟨h.1.1, h.2⟩
      rw [Set.indicator_of_notMem hω, Set.indicator_of_notMem hC, ENNReal.ofReal_zero]
  have hCm : MeasurableSet C := by
    have : C = P ∩ cleanAll O := by
      ext ω; constructor
      · intro hω; exact ⟨hω.1.1, hω.1.2⟩
      · rintro ⟨hP, hc'⟩; exact (cellOf_const hD h₀ hP hc').2
    rw [this]; exact hPm'.inter (measurableSet_cleanAll O)
  have hint_ctc : Integrable (C.indicator fun ω => (tc ω : ℝ)) μ := hint_tc.indicator hCm
  have hFTbound : μ.real (C ∩ {ω | FTc ω})
      ≤ uGood * ∫ ω, C.indicator (fun ω => (tc ω : ℝ)) ω ∂μ := by
    have h1 := (measure_mono hsub).trans hfresh
    rw [hlin, ← ofReal_integral_eq_lintegral_ofReal hint_ctc
      (ae_of_all _ fun ω => Set.indicator_nonneg (fun ω _ => Nat.cast_nonneg _) ω),
      ← ENNReal.ofReal_mul hu] at h1
    rw [measureReal_def]
    exact ENNReal.toReal_le_of_le_ofReal (mul_nonneg hu (integral_nonneg fun ω =>
      Set.indicator_nonneg (fun _ _ => Nat.cast_nonneg _) ω)) h1
  -- assemble
  have e3 : ∫ ω, C.indicator 1 ω * h ω ∂μ
      = μ.real (C ∩ {ω | FTc ω}) - uGood * ∫ ω, C.indicator (fun ω => (tc ω : ℝ)) ω ∂μ := by
    have hpt : (fun ω => C.indicator 1 ω * h ω) = fun ω =>
        (C ∩ {ω | FTc ω}).indicator (fun _ => (1 : ℝ)) ω
          - uGood * C.indicator (fun ω => (tc ω : ℝ)) ω := by
      funext ω
      by_cases hC : ω ∈ C
      · rw [Set.indicator_of_mem hC, Set.indicator_of_mem hC, Pi.one_apply, one_mul]
        by_cases hf : FTc ω
        · rw [Set.indicator_of_mem (show ω ∈ C ∩ {ω | FTc ω} from ⟨hC, hf⟩)]
          simp only [h, gContrib]
          rw [if_pos hf]
        · rw [Set.indicator_of_notMem (show ω ∉ C ∩ {ω | FTc ω} from fun h' => hf h'.2)]
          simp only [h, gContrib]
          rw [if_neg hf]
      · rw [Set.indicator_of_notMem hC, Set.indicator_of_notMem hC,
          Set.indicator_of_notMem (show ω ∉ C ∩ {ω | FTc ω} from fun h' => hC h'.1)]
        simp
    rw [hpt, integral_sub ((integrable_const (1 : ℝ)).indicator (hCm.inter hFTm'))
      (hint_ctc.const_mul _), integral_indicator_const _ (hCm.inter hFTm'), integral_const_mul,
      smul_eq_mul, mul_one]
  rw [← e1, e2, e3]
  linarith


end Cells

section Hoeffding

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}
variable (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (K : StageKnobs α)
  (k : ℕ) (seed probes : List (FreeMonoid α)) (Pf : CutReads α → KState α)
variable {G : DTree α → Edges α → FreeMonoid α → Qry α β} {harv : β → List (FreeMonoid α)}

/-- The bits a draw's class can read off the cell's. -/
noncomputable def clsBits (V : Finset (FreeMonoid α))
    (c : Finset (FreeMonoid α) × Finset (FreeMonoid α)) (t : DTree α) (x : FreeMonoid α) :
    Finset (FreeMonoid α) :=
  vBits F ((((Finset.Icc k (max k x.toList.length)).image (prefixOf x)) ×ˢ t.midfixes).image
    fun z => z.1 * z.2) \ vBits V c.1

omit [Fintype α] [DecidableEq α] in
theorem prefixOf_clip (x : FreeMonoid α) {i : ℕ} (hi : k ≤ i) :
    prefixOf x (min i (max k x.toList.length)) = prefixOf x i := by
  unfold prefixOf
  congr 1
  rw [List.take_eq_take_iff]
  omega

theorem image_sub_clsBits (V : Finset (FreeMonoid α))
    (c : Finset (FreeMonoid α) × Finset (FreeMonoid α)) (t : DTree α) (x : FreeMonoid α)
    {w : FreeMonoid α} (hw : ∃ i, k ≤ i ∧ ∃ m ∈ t.mids, w = prefixOf x i * m) :
    (↑(F.image (w * ·)) : Set (FreeMonoid α)) \ ↑(vBits V c.1)
      ⊆ ↑(clsBits F k V c t x) := by
  rintro z ⟨hz, hzv⟩
  obtain ⟨i, hi, m, hm, rfl⟩ := hw
  simp only [Finset.coe_image, Set.mem_image, Finset.mem_coe] at hz
  obtain ⟨v, hv, rfl⟩ := hz
  simp only [clsBits, Finset.coe_sdiff, Set.mem_diff, Finset.mem_coe]
  refine ⟨?_, hzv⟩
  simp only [vBits, Finset.mem_biUnion, Finset.mem_image, Finset.mem_product, Prod.exists]
  refine ⟨prefixOf x i * m, ⟨prefixOf x (min i (max k x.toList.length)), m,
    ⟨⟨min i (max k x.toList.length), Finset.mem_Icc.2 ⟨by omega, min_le_right _ _⟩, rfl⟩,
      (DTree.mem_midfixes_iff _).2 hm⟩, by rw [prefixOf_clip k x hi]⟩, v, hv, rfl⟩

theorem clsBits_sub_drawBits (V : Finset (FreeMonoid α))
    (c : Finset (FreeMonoid α) × Finset (FreeMonoid α)) (t : DTree α) {x : FreeMonoid α}
    (hx : k ≤ x.toList.length) : ↑(clsBits F k V c t x) ⊆ drawBits V c k x := by
  intro z hz
  simp only [clsBits, Finset.coe_sdiff, Set.mem_diff, Finset.mem_coe] at hz
  refine ⟨?_, hz.2⟩
  obtain ⟨w, hw, hzw⟩ := Finset.mem_biUnion.1 hz.1
  obtain ⟨⟨y, m⟩, hym, rfl⟩ := Finset.mem_image.1 hw
  obtain ⟨hy, -⟩ := Finset.mem_product.1 hym
  obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hy
  obtain ⟨v, -, rfl⟩ := Finset.mem_image.1 hzw
  have hki := (Finset.mem_Icc.1 hi).1
  simp only [Set.mem_ofPred_eq, FreeMonoid.toList_mul, prefixOf, FreeMonoid.toList_ofList]
  exact (List.take_prefix_take_left hki).trans (List.prefix_append _ _ |>.trans
    (List.prefix_append _ _))

/-- Hoeffding inside a cell of the pass: the draws' shares, grouped by their first `k` letters,
are independent and each group moves the sum by at most its mass times the scale. -/
theorem cell_tail_hoeff [IsProbabilityMeasure μ] (hG : HarvestSpec G harv k)
    (bnd : DTree α → ℕ → ℕ)
    (hbnd : ∀ (R : CutReads α) t e x, gTags G R.cut t e x ≤ bnd t x.toList.length)
    (A : DFA (FreeMonoid α) Q) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {L : ℕ}
    (hkL : k ≤ L) {uGood ε : ℝ} (hu : 0 ≤ uGood) (hε : 0 < ε) (hV : SuffixFree (K.suffixes F))
    {c : Finset (FreeMonoid α) × Finset (FreeMonoid α)} {ω₀ : Ω}
    (hD : PassDetermined O B F K k seed probes Pf)
    (h₀ : ω₀ ∈ cellOf O B F K k seed probes Pf c) :
    μ.real (cellOf O B F K k seed probes Pf c ∩ {ω | ε * (1 + uGood
        * bnd (passOf O B F Pf ω).tree L) < ∑ x ∈ wordsOf (α := α) L, D.real {x}
          * gContrib G harv A O B F uGood (readsAt O B F ω).cut
            (passOf O B F Pf ω).tree (passOf O B F Pf ω).edges
            (readsOf O B F k seed probes Pf ω) x})
      ≤ μ.real (cellOf O B F K k seed probes Pf c) * Real.exp (-2 * ε ^ 2 / prefixMax D k) := by
  classical
  set V := K.suffixes F
  set s₀ := passOf O B F Pf ω₀
  set P := pcell O V c
  set C := cellOf O B F K k seed probes Pf c
  set d₀ : ℝ := (bnd s₀.tree L : ℝ)
  set R : ℝ := 1 + uGood * d₀
  set X := wordsOf (α := α) L
  set h : FreeMonoid α → Ω → ℝ := fun x ω =>
    gContrib G harv A O B F uGood (cellReads O B F V c ω).cut s₀.tree s₀.edges c.1 x
  have hle := noiseAlg_le O
  have hd₀ : 0 ≤ d₀ := Nat.cast_nonneg _
  have hR : 0 < R := by positivity
  have hhm : ∀ x, Measurable[noiseAlg O ↑(clsBits F k V c s₀.tree x)] (h x) := fun x =>
    (measurable_gContrib_of hG A O B F V uGood c s₀.tree s₀.edges c.1 x _
      fun w hw => image_sub_clsBits F k V c s₀.tree x hw).2.2
  have hhm' : ∀ x, Measurable (h x) := fun x => (hhm x).mono (hle _) le_rfl
  have hhb : ∀ x ∈ X, ∀ ω, -(uGood * d₀) ≤ h x ω ∧ h x ω ≤ 1 := by
    intro x hx ω
    have hl := mem_wordsOf.1 hx
    have ht : (gTags G (cellReads O B F V c ω).cut s₀.tree s₀.edges x : ℝ) ≤ d₀ := by
      have := hbnd (cellReads O B F V c ω) s₀.tree s₀.edges x
      rw [hl] at this; simp only [d₀]; exact_mod_cast this
    have ht0 : (0 : ℝ) ≤ gTags G (cellReads O B F V c ω).cut s₀.tree s₀.edges x :=
      Nat.cast_nonneg _
    simp only [h, gContrib]
    split_ifs <;> constructor <;> nlinarith
  have hhi : ∀ x ∈ X, Integrable (h x) μ := fun x hx =>
    Integrable.of_bound (hhm' x).aestronglyMeasurable R (ae_of_all _ fun ω => by
      obtain ⟨h1, h2⟩ := hhb x hx ω
      rw [Real.norm_eq_abs, abs_le]; constructor <;> nlinarith)
  -- the cell agrees with the pattern off a null set
  have hcl := measure_cleanAll_compl O (μ := μ)
  have hPm : MeasurableSet[noiseAlg O ↑(vBits V c.1)] P :=
    (measurableSet_noisePattern O _ c.2).inter (measurableSet_noiseClean O _)
  have hPm' : MeasurableSet P := hle _ _ hPm
  have hCeq : C = P ∩ cleanAll O := by
    ext ω; constructor
    · intro (hω : ω ∈ C); exact ⟨hω.1.1, hω.1.2⟩
    · rintro ⟨hP, hc'⟩; exact (cellOf_const hD h₀ hP hc').2
  have hCP : μ.real C = μ.real P := by
    rw [hCeq, measureReal_def, measure_inter_conull hcl, ← measureReal_def]
  rcases eq_or_lt_of_le (measureReal_nonneg : 0 ≤ μ.real P) with hP0 | hP0
  · have : μ.real C = 0 := by rw [hCP, ← hP0]
    refine (measureReal_mono Set.inter_subset_left (measure_ne_top _ _)).trans ?_
    rw [this]; simp
  -- the means are at most zero
  set m : FreeMonoid α → ℝ := fun x => ∫ ω, h x ω ∂μ
  have hm0 : ∀ x, m x ≤ 0 := fun x => by
    have := cell_mean_le_gen O B F K k seed probes Pf hG bnd hbnd A hu hV hD h₀ x
    exact nonpos_of_mul_nonpos_right this hP0
  -- the groups by first `k` letters
  set Ps := X.image (prefixOf · k)
  set grp : FreeMonoid α → Finset (FreeMonoid α) := fun p => X.filter fun x => prefixOf x k = p
  set Ablk : FreeMonoid α → Finset (FreeMonoid α) := fun p =>
    (grp p).biUnion (clsBits F k V c s₀.tree)
  set Gp : FreeMonoid α → Ω → ℝ := fun p ω => ∑ x ∈ grp p, D.real {x} * h x ω
  set Dp : FreeMonoid α → ℝ := fun p => ∑ x ∈ grp p, D.real {x}
  have hDp0 : ∀ p, 0 ≤ Dp p := fun p => Finset.sum_nonneg fun _ _ => measureReal_nonneg
  have hGb : ∀ p ω, Gp p ω ∈ Set.Icc (-(Dp p * (uGood * d₀))) (Dp p) := by
    intro p ω
    have hx : ∀ x ∈ grp p, x ∈ X := fun x hx => (Finset.mem_filter.1 hx).1
    constructor
    · rw [show -(Dp p * (uGood * d₀)) = ∑ x ∈ grp p, D.real {x} * -(uGood * d₀) by
        simp only [Dp, Finset.sum_mul, mul_neg, Finset.sum_neg_distrib]]
      exact Finset.sum_le_sum fun x hxg => mul_le_mul_of_nonneg_left (hhb x (hx x hxg) ω).1
        measureReal_nonneg
    · rw [show Dp p = ∑ x ∈ grp p, D.real {x} * 1 by simp [Dp]]
      exact Finset.sum_le_sum fun x hxg => mul_le_mul_of_nonneg_left (hhb x (hx x hxg) ω).2
        measureReal_nonneg
  have hAm : ∀ p, noiseAlg O ↑(Ablk p)
      = ⨆ w ∈ Ablk p, MeasurableSpace.comap (O.noise w) inferInstance := by
    intro p; rfl
  have hGm : ∀ p, Measurable[noiseAlg O ↑(Ablk p)] (Gp p) := by
    intro p
    refine Finset.measurable_sum _ fun x hx => ((hhm x).mono (noiseAlg_mono O ?_) le_rfl).const_mul _
    intro z hz
    exact Finset.mem_coe.2 (Finset.mem_biUnion.2 ⟨x, hx, hz⟩)
  have hdisj : Pairwise fun p p' => Disjoint (Ablk p) (Ablk p') := by
    intro p p' hpp'
    rw [Finset.disjoint_left]
    intro z hz hz'
    obtain ⟨x, hx, hzx⟩ := Finset.mem_biUnion.1 hz
    obtain ⟨y, hy, hzy⟩ := Finset.mem_biUnion.1 hz'
    obtain ⟨hxX, rfl⟩ := Finset.mem_filter.1 hx
    obtain ⟨hyX, rfl⟩ := Finset.mem_filter.1 hy
    have hxl := mem_wordsOf.1 hxX
    have hyl := mem_wordsOf.1 hyX
    have h1 := clsBits_sub_drawBits F k V c s₀.tree (x := x) (by omega) hzx
    have h2 := clsBits_sub_drawBits F k V c s₀.tree (x := y) (by omega) hzy
    exact Set.disjoint_left.1 (disjoint_drawBits k (V := V) (c := c) (by omega) (by omega)
      hpp') h1 h2
  have hind : iIndepFun Gp μ := iIndepFun_blocks O.noise_meas O.noise_indep Ablk hdisj Gp
    fun p => hAm p ▸ hGm p
  -- centred
  set Y : FreeMonoid α → Ω → ℝ := fun p ω => Gp p ω - μ[Gp p]
  have hYind : iIndepFun Y μ := by
    have := hind.comp (fun p (x : ℝ) => x - ∫ ω, Gp p ω ∂μ) (fun p => measurable_id.sub_const _)
    exact this
  have hYsub : ∀ p ∈ Ps, ProbabilityTheory.HasSubgaussianMGF (Y p)
      ((‖Dp p - -(Dp p * (uGood * d₀))‖₊ / 2) ^ 2) μ := fun p _ =>
    ProbabilityTheory.hasSubgaussianMGF_of_mem_Icc
      ((hGm p).mono (hle _) le_rfl).aemeasurable (ae_of_all _ (hGb p))
  have hεR : 0 ≤ ε * R := by positivity
  have hmain := ProbabilityTheory.HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hYind hYsub hεR
  -- the sum regrouped
  set W : Ω → ℝ := fun ω => ∑ x ∈ X, D.real {x} * h x ω
  set M : ℝ := ∑ x ∈ X, D.real {x} * m x
  have hM : M ≤ 0 := Finset.sum_nonpos fun x _ => mul_nonpos_of_nonneg_of_nonpos measureReal_nonneg
    (hm0 x)
  have hgrp : ∀ (g : FreeMonoid α → ℝ), ∑ x ∈ X, g x = ∑ p ∈ Ps, ∑ x ∈ grp p, g x := fun g =>
    (Finset.sum_fiberwise_of_maps_to (fun x hx => Finset.mem_image_of_mem _ hx) g).symm
  have hmean : ∀ p, μ[Gp p] = ∑ x ∈ grp p, D.real {x} * m x := by
    intro p
    simp only [Gp]
    rw [integral_finsetSum _ fun x hx => (hhi x (Finset.mem_filter.1 hx).1).const_mul _]
    exact Finset.sum_congr rfl fun x _ => integral_const_mul _ _
  have hYsum : ∀ ω, ∑ p ∈ Ps, Y p ω = W ω - M := by
    intro ω
    simp only [Y, Finset.sum_sub_distrib, hmean, W, M]
    rw [hgrp, hgrp]
  -- the exponent
  have hDpp : ∀ p ∈ Ps, Dp p ≤ prefixMax D k := by
    intro p hp
    obtain ⟨x, hx, rfl⟩ := Finset.mem_image.1 hp
    have := sum_same_prefix_le D hkL (mem_wordsOf.1 hx)
    refine le_of_eq_of_le ?_ this
    rw [← Finset.sum_filter]
  have hDsum : ∑ p ∈ Ps, Dp p ≤ 1 := by
    rw [← hgrp, sum_measureReal_singleton]; exact measureReal_le_one
  have hpm : 0 ≤ prefixMax D k :=
    Real.iSup_nonneg fun p => by split_ifs <;> simp [measureReal_nonneg]
  have hsq : ∑ p ∈ Ps, Dp p ^ 2 ≤ prefixMax D k := by
    calc ∑ p ∈ Ps, Dp p ^ 2 ≤ ∑ p ∈ Ps, prefixMax D k * Dp p :=
          Finset.sum_le_sum fun p hp => by
            rw [sq]; exact mul_le_mul_of_nonneg_right (hDpp p hp) (hDp0 p)
      _ = prefixMax D k * ∑ p ∈ Ps, Dp p := by rw [Finset.mul_sum]
      _ ≤ prefixMax D k * 1 := mul_le_mul_of_nonneg_left hDsum hpm
      _ = prefixMax D k := mul_one _
  have hc : ∀ p, ((‖Dp p - -(Dp p * (uGood * d₀))‖₊ / 2) ^ 2 : NNReal).toReal
      = R ^ 2 / 4 * Dp p ^ 2 := by
    intro p
    have : Dp p - -(Dp p * (uGood * d₀)) = Dp p * R := by simp only [R]; ring
    rw [this]
    push_cast
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg (hDp0 p) hR.le)]
    ring
  have htail : μ.real {ω | ε * R ≤ W ω - M} ≤ Real.exp (-2 * ε ^ 2 / prefixMax D k) := by
    rcases eq_or_lt_of_le (Finset.sum_nonneg fun p (_ : p ∈ Ps) => sq_nonneg (Dp p)) with h0 | h0
    · -- no mass in any group: the sum is its mean
      have hD0 : ∀ p ∈ Ps, Dp p = 0 := fun p hp =>
        pow_eq_zero_iff (n := 2) (by norm_num) |>.1
          ((Finset.sum_eq_zero_iff_of_nonneg fun p _ => sq_nonneg (Dp p)).1 h0.symm p hp)
      have hY0 : ∀ ω, ∑ p ∈ Ps, Y p ω = 0 := by
        intro ω
        refine Finset.sum_eq_zero fun p hp => ?_
        have := hGb p ω
        have hμ := hGb p
        rw [hD0 p hp] at this
        have hG0 : ∀ ω', Gp p ω' = 0 := fun ω' => by
          have := hGb p ω'; rw [hD0 p hp] at this
          simp only [zero_mul, neg_zero, Set.mem_Icc] at this; linarith [this.1, this.2]
        simp only [Y, hG0, integral_zero, sub_zero]
      have : {ω | ε * R ≤ W ω - M} = ∅ := by
        ext ω
        simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_le]
        rw [← hYsum, hY0]
        positivity
      rw [this, measureReal_empty]; positivity
    · refine le_of_eq_of_le ?_ (hmain.trans (Real.exp_le_exp.2 ?_))
      · congr 1; ext ω; simp only [Set.mem_ofPred_eq, hYsum]
      · simp only [NNReal.coe_sum, hc]
        rw [← Finset.mul_sum]
        rw [show -(ε * R) ^ 2 / (2 * (R ^ 2 / 4 * ∑ p ∈ Ps, Dp p ^ 2))
            = -2 * ε ^ 2 / ∑ p ∈ Ps, Dp p ^ 2 by field_simp; ring]
        have h2 : 2 * ε ^ 2 / prefixMax D k ≤ 2 * ε ^ 2 / ∑ p ∈ Ps, Dp p ^ 2 :=
          div_le_div_of_nonneg_left (by positivity) h0 hsq
        have e1 : -2 * ε ^ 2 / ∑ p ∈ Ps, Dp p ^ 2 = -(2 * ε ^ 2 / ∑ p ∈ Ps, Dp p ^ 2) := by ring
        have e2 : -2 * ε ^ 2 / prefixMax D k = -(2 * ε ^ 2 / prefixMax D k) := by ring
        rw [e1, e2]
        linarith
  -- on the cell
  have hon : ∀ ω ∈ C, (∑ x ∈ X, D.real {x} * gContrib G harv A O B F uGood (readsAt O B F ω).cut
      (passOf O B F Pf ω).tree (passOf O B F Pf ω).edges
      (readsOf O B F k seed probes Pf ω) x) = W ω
      ∧ bnd (passOf O B F Pf ω).tree L = bnd s₀.tree L := by
    intro ω hω
    have hr := cellReads_eq O B F V c hω.1.1.1 hω.1.2
    have hs := (cellOf_const hD h₀ hω.1.1 hω.1.2).1
    have hT : readsOf O B F k seed probes Pf ω = c.1 := hω.2
    refine ⟨Finset.sum_congr rfl fun x _ => ?_, by rw [hs]⟩
    simp only [h, hr, hs, hT, s₀]
  have hsub : C ∩ {ω | ε * (1 + uGood * bnd (passOf O B F Pf ω).tree L)
      < ∑ x ∈ X, D.real {x} * gContrib G harv A O B F uGood (readsAt O B F ω).cut
        (passOf O B F Pf ω).tree (passOf O B F Pf ω).edges
        (readsOf O B F k seed probes Pf ω) x} ⊆ P ∩ {ω | ε * R ≤ W ω - M} := by
    rintro ω ⟨hω, hlt⟩
    obtain ⟨hsum, hdep⟩ := hon ω hω
    simp only [Set.mem_ofPred_eq, hsum, hdep] at hlt
    exact ⟨(hCeq ▸ hω).1, by simp only [Set.mem_ofPred_eq, R, d₀]; linarith⟩
  -- the pattern is independent of the groups
  set Tall := Ps.biUnion Ablk
  have hWm : Measurable[noiseAlg O ↑Tall] fun ω => W ω - M := by
    have : ∀ ω, W ω - M = ∑ p ∈ Ps, Y p ω := fun ω => (hYsum ω).symm
    simp_rw [this]
    refine Finset.measurable_sum _ fun p hp => ((hGm p).mono (noiseAlg_mono O ?_) le_rfl).sub_const _
    intro z hz; exact Finset.mem_coe.2 (Finset.mem_biUnion.2 ⟨p, hp, hz⟩)
  have hdT : Disjoint (↑(vBits V c.1) : Set (FreeMonoid α)) ↑Tall := by
    rw [Set.disjoint_left]
    intro z hz hz'
    obtain ⟨p, -, hzp⟩ := Finset.mem_biUnion.1 hz'
    obtain ⟨x, -, hzx⟩ := Finset.mem_biUnion.1 hzp
    exact (Finset.mem_sdiff.1 hzx).2 hz
  have hindep := indep_noiseAlg O hdT (μ := μ)
  have hEm : MeasurableSet[noiseAlg O ↑Tall] {ω | ε * R ≤ W ω - M} :=
    measurableSet_le measurable_const hWm
  have hPE : μ.real (P ∩ {ω | ε * R ≤ W ω - M}) = μ.real P * μ.real {ω | ε * R ≤ W ω - M} := by
    rw [measureReal_def, measureReal_def, measureReal_def,
      (ProbabilityTheory.Indep_iff _ _ _).1 hindep _ _ hPm hEm, ENNReal.toReal_mul]
  calc _ ≤ μ.real (P ∩ {ω | ε * R ≤ W ω - M}) := measureReal_mono hsub (measure_ne_top _ _)
    _ = μ.real P * μ.real {ω | ε * R ≤ W ω - M} := hPE
    _ ≤ μ.real P * Real.exp (-2 * ε ^ 2 / prefixMax D k) :=
        mul_le_mul_of_nonneg_left htail measureReal_nonneg
    _ = μ.real C * Real.exp (-2 * ε ^ 2 / prefixMax D k) := by rw [hCP]

end Hoeffding

section Fail

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}
variable {G : DTree α → Edges α → FreeMonoid α → Qry α β} {harv : β → List (FreeMonoid α)}
  {k : ℕ}

/-- Draws harvesting a string the pass may have read share their first `k` letters with it. -/
theorem gen_touched_le (hG : HarvestSpec G harv k) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] {L : ℕ} (hkL : k ≤ L) (hlen : ∀ᵐ x ∂D, x.toList.length = L)
    (R : CutReads α) (t : DTree α) (edges : Edges α)
    (Tp : Finset (FreeMonoid α)) :
    D.real {x | ∃ b ∈ harv ((G t edges x).run R.cut), b ∈ Tp}
      ≤ (kPrefixes k Tp).card * prefixMax D k := by
  classical
  have hsub : {x | ∃ b ∈ harv ((G t edges x).run R.cut), b ∈ Tp}
      ⊆ (⋃ p ∈ kPrefixes k Tp, {x | p.toList <+: x.toList}) ∪ {x | ¬ x.toList.length = L} := by
    rintro x ⟨b, hb, hT⟩
    by_cases hx : x.toList.length = L
    swap
    · exact .inr hx
    left
    obtain ⟨i, hi, m, rfl⟩ := hG.form R t edges x b hb
    have hlen' : k ≤ (prefixOf x i).toList.length := by simp [prefixOf]; omega
    have hbk : k ≤ (prefixOf x i * m).toList.length := by
      simp only [FreeMonoid.toList_mul, List.length_append]; omega
    refine Set.mem_biUnion (Finset.mem_image_of_mem _ (Finset.mem_filter.2 ⟨hT, hbk⟩)) ?_
    simp only [Set.mem_ofPred_eq, prefixOf, FreeMonoid.toList_ofList, FreeMonoid.toList_mul]
    rw [List.take_append_of_le_length (by simpa [prefixOf] using hlen'), List.take_take,
      min_eq_left hi]
    exact List.take_prefix _ _
  have hnull : D.real {x : FreeMonoid α | ¬ x.toList.length = L} = 0 := by
    rw [measureReal_def, ae_iff.1 hlen, ENNReal.toReal_zero]
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans
    ((measureReal_union_le _ _).trans ?_)
  rw [hnull, add_zero]
  refine (measureReal_biUnion_finset_le _ _).trans ?_
  calc ∑ p ∈ kPrefixes k Tp, D.real {x | p.toList <+: x.toList}
      ≤ ∑ _p ∈ kPrefixes k Tp, prefixMax D k :=
        Finset.sum_le_sum fun p hp => by
          obtain ⟨w, hw, rfl⟩ := Finset.mem_image.1 hp
          refine le_trans (le_of_eq ?_) (le_ciSup (f := fun p : FreeMonoid α =>
            if p.toList.length = k then D.real {x | p.toList <+: x.toList} else 0)
            ⟨1, by rintro _ ⟨p, rfl⟩; simp only []; split_ifs <;> simp [measureReal_le_one]⟩
            (prefixOf w k))
          rw [if_pos (length_prefixOf (Finset.mem_filter.1 hw).2)]
    _ = (kPrefixes k Tp).card * prefixMax D k := by
        rw [Finset.sum_const, nsmul_eq_mul]

open scoped Classical in
/-- Where the class's claim fails, the draws' shares sum past the fluctuation allowed. -/
theorem gen_fail_sum (hG : HarvestSpec G harv k) (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) {u ε M : ℝ}
    (R : CutReads α) (t : DTree α) (edges : Edges α)
    (Tp : Finset (FreeMonoid α)) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {L : ℕ}
    (hkL : k ≤ L) (hlen : ∀ᵐ x ∂D, x.toList.length = L)
    (h : ¬ D.real {x | harv ((G t edges x).run R.cut) ≠ []
          ∧ ∀ b ∈ harv ((G t edges x).run R.cut), stateIndecision A O B F (A.state b) < u}
      ≤ u * ∫ x, (gTags G R.cut t edges x : ℝ) ∂D + (kPrefixes k Tp).card * prefixMax D k
        + ε * (1 + u * M)) :
    ε * (1 + u * M)
      < ∑ x ∈ wordsOf (α := α) L, D.real {x} * gContrib G harv A O B F u R.cut t edges Tp x := by
  set X := wordsOf (α := α) L
  have h1 : D.real {x | harv ((G t edges x).run R.cut) ≠ []
      ∧ ∀ b ∈ harv ((G t edges x).run R.cut), stateIndecision A O B F (A.state b) < u}
      ≤ D.real {x | GFresh G harv A O B F u R.cut t edges Tp x}
        + D.real {x | ∃ b ∈ harv ((G t edges x).run R.cut), b ∈ Tp} := by
    refine (measureReal_mono (fun x hx => ?_) (measure_ne_top _ _)).trans
      (measureReal_union_le _ _)
    obtain ⟨hne, hg⟩ := hx
    by_cases hT : ∃ b ∈ harv ((G t edges x).run R.cut), b ∈ Tp
    · exact .inr hT
    · push Not at hT
      exact .inl ⟨hne, fun b hb => ⟨hg b hb, hT b hb⟩⟩
  have h2 := gen_touched_le hG D hkL hlen R t edges Tp
  have h3 : D.real {x | GFresh G harv A O B F u R.cut t edges Tp x}
      = ∑ x ∈ X, D.real {x} * (if GFresh G harv A O B F u R.cut t edges Tp x then 1 else 0) := by
    rw [real_eq_sum_words D hlen]
    refine Finset.sum_congr rfl fun x _ => ?_
    simp only [Set.mem_ofPred_eq]
    split_ifs <;> simp
  have h4 := integral_eq_sum_words D hlen fun x => (gTags G R.cut t edges x : ℝ)
  have h6 : ∑ x ∈ X, D.real {x} * gContrib G harv A O B F u R.cut t edges Tp x
      = ∑ x ∈ X, D.real {x} * (if GFresh G harv A O B F u R.cut t edges Tp x then 1 else 0)
        - u * ∑ x ∈ X, D.real {x} * (gTags G R.cut t edges x : ℝ) := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun x _ => ?_
    simp only [gContrib]
    ring
  rw [h6, ← h3]
  rw [h4] at h
  push Not at h
  linarith

end Fail

section Holds

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}
variable {G : DTree α → Edges α → FreeMonoid α → Qry α β} {harv : β → List (FreeMonoid α)}
  {k : ℕ}

/-- A class's claim over the oracle's noise. -/
theorem harvest_holds_le [IsProbabilityMeasure μ] (hG : HarvestSpec G harv k)
    (bnd : DTree α → ℕ → ℕ)
    (hbnd : ∀ (R : CutReads α) t e x, gTags G R.cut t e x ≤ bnd t x.toList.length)
    (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (K : StageKnobs α) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] {L : ℕ} (seed probes : List (FreeMonoid α))
    (Pf : CutReads α → KState α) (hD : PassDetermined O B F K k seed probes Pf) {u ε : ℝ}
    (hu : 0 ≤ u) (hε : 0 < ε) (hkL : k ≤ L) (hlen : ∀ᵐ x ∂D, x.toList.length = L)
    (hV : SuffixFree (K.suffixes F)) :
    μ.real {ω | ¬ D.real {x | harv ((G (passOf O B F Pf ω).tree
          (passOf O B F Pf ω).edges x).run (readsAt O B F ω).cut) ≠ []
        ∧ ∀ b ∈ harv ((G (passOf O B F Pf ω).tree
          (passOf O B F Pf ω).edges x).run (readsAt O B F ω).cut),
          stateIndecision A O B F (A.state b) < u}
      ≤ u * ∫ x, (gTags G (readsAt O B F ω).cut (passOf O B F Pf ω).tree
          (passOf O B F Pf ω).edges x : ℝ) ∂D
        + (kPrefixes k (passReadSet k seed probes (passOf O B F Pf ω).tree)).card
          * prefixMax D k
        + ε * (1 + u * bnd (passOf O B F Pf ω).tree L)}
      ≤ Real.exp (-2 * ε ^ 2 / prefixMax D k) := by
  classical
  set V := K.suffixes F
  have hr : 0 ≤ Real.exp (-2 * ε ^ 2 / prefixMax D k) := (Real.exp_pos _).le
  set Ev : Set Ω := {ω | ε * (1 + u * bnd (passOf O B F Pf ω).tree L)
    < ∑ x ∈ wordsOf (α := α) L, D.real {x} * gContrib G harv A O B F u (readsAt O B F ω).cut
      (passOf O B F Pf ω).tree (passOf O B F Pf ω).edges
      (readsOf O B F k seed probes Pf ω) x}
  have hsub : {ω | ¬ D.real {x | harv ((G (passOf O B F Pf ω).tree
          (passOf O B F Pf ω).edges x).run (readsAt O B F ω).cut) ≠ []
        ∧ ∀ b ∈ harv ((G (passOf O B F Pf ω).tree
          (passOf O B F Pf ω).edges x).run (readsAt O B F ω).cut),
          stateIndecision A O B F (A.state b) < u}
      ≤ u * ∫ x, (gTags G (readsAt O B F ω).cut (passOf O B F Pf ω).tree
          (passOf O B F Pf ω).edges x : ℝ) ∂D
        + (kPrefixes k (passReadSet k seed probes (passOf O B F Pf ω).tree)).card
          * prefixMax D k
        + ε * (1 + u * bnd (passOf O B F Pf ω).tree L)} ⊆ Ev := fun ω hω =>
    gen_fail_sum hG A O B F _ _ _ _ D hkL hlen hω
  set cells := fun c : Finset (FreeMonoid α) × Finset (FreeMonoid α) =>
    cellOf O B F K k seed probes Pf c
  have hcover : Ev ⊆ (cleanAll O)ᶜ ∪ ⋃ c, cells c ∩ Ev := by
    intro ω hω
    by_cases hcl : ω ∈ cleanAll O
    · refine .inr (Set.mem_iUnion.2 ⟨(readsOf O B F k seed probes Pf ω,
        noisePattern O (vBits V (readsOf O B F k seed probes Pf ω)) ω), ?_, hω⟩)
      exact ⟨⟨⟨rfl, fun y _ => hcl y⟩, hcl⟩, rfl⟩
    · exact .inl hcl
  have hcell : ∀ c, μ (cells c ∩ Ev) ≤ μ (cells c) * ENNReal.ofReal (Real.exp (-2 * ε ^ 2 / prefixMax D k)) := by
    intro c
    by_cases hne : (cells c).Nonempty
    · obtain ⟨ω₀, h₀⟩ := hne
      have := cell_tail_hoeff O B F K k seed probes Pf hG bnd hbnd A D hkL hu hε hV hD h₀
      rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def,
        ← ENNReal.ofReal_toReal (measure_ne_top μ (cells c)), ← measureReal_def,
        ← ENNReal.ofReal_mul measureReal_nonneg]
      refine ENNReal.ofReal_le_ofReal (le_trans this (le_of_eq ?_))
      ring
    · rw [Set.not_nonempty_iff_eq_empty.1 hne]; simp
  have hdisj : Pairwise (Function.onFun Disjoint cells) := by
    intro c c' hcc'
    rw [Function.onFun, Set.disjoint_left]
    intro ω hω hω'
    apply hcc'
    have h1 : c.1 = c'.1 := hω.2.symm.trans hω'.2
    have h2 : c.2 = c'.2 := by
      have a : noisePattern O (vBits V c.1) ω = c.2 := hω.1.1.1
      have b : noisePattern O (vBits V c'.1) ω = c'.2 := hω'.1.1.1
      rw [h1] at a
      exact a.symm.trans b
    exact Prod.ext h1 h2
  have hmeas : ∀ c, MeasurableSet (cells c) := measurableSet_cellOf hD
  have htot : μ Ev ≤ ENNReal.ofReal (Real.exp (-2 * ε ^ 2 / prefixMax D k)) := by
    calc μ Ev ≤ μ ((cleanAll O)ᶜ ∪ ⋃ c, cells c ∩ Ev) := measure_mono hcover
      _ ≤ μ (cleanAll O)ᶜ + μ (⋃ c, cells c ∩ Ev) := measure_union_le _ _
      _ = μ (⋃ c, cells c ∩ Ev) := by rw [measure_cleanAll_compl O, zero_add]
      _ ≤ ∑' c, μ (cells c ∩ Ev) := measure_iUnion_le _
      _ ≤ ∑' c, μ (cells c) * ENNReal.ofReal (Real.exp (-2 * ε ^ 2 / prefixMax D k)) :=
          ENNReal.tsum_le_tsum hcell
      _ = (∑' c, μ (cells c)) * ENNReal.ofReal (Real.exp (-2 * ε ^ 2 / prefixMax D k)) := ENNReal.tsum_mul_right
      _ = μ (⋃ c, cells c) * ENNReal.ofReal (Real.exp (-2 * ε ^ 2 / prefixMax D k)) := by
          rw [measure_iUnion hdisj hmeas]
      _ ≤ 1 * ENNReal.ofReal (Real.exp (-2 * ε ^ 2 / prefixMax D k)) := by gcongr; exact prob_le_one
      _ = _ := one_mul _
  calc μ.real _ ≤ μ.real Ev := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ Real.exp (-2 * ε ^ 2 / prefixMax D k) := by
        rw [measureReal_def]; exact ENNReal.toReal_le_of_le_ofReal hr htot

end Holds

end OrthoDFA
