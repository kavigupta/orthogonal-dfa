import RLStar.Proofs.LoopRenoise
import RLStar.Proofs.LazyZone
import RLStar.Proofs.QueryCutoff
import RLStar.Proofs.QueryBadTail
import RLStar.Proofs.ListSlope
import RLStar.Proofs.UnaskedFresh
import RLStar.Proofs.AdaptiveReads
import RLStar.Proofs.LazyRound
import RLStar.Proofs.LazyEpoch
import RLStar.Proofs.LazyTopUp
import RLStar.Proofs.TrieMass

/-!
# A lazy round read through a suffix family

The lazy round from a trie `X₀`, reading a suffix family over probes from `D'`, fails its claim
(`lazyFail`) with chance at most

    δ + E J (1 − y/4)^b + κ Nr + δH + (Lp + 1)(|Q| + 2) κ / tw

(`lazyFail_le`): some string it asks reads on its state's rarer side (`κ Nr`, the round asking at
most `Nr`); else the round reads as if no read were wrong, so `LazyRoundCorrect` holds
(`δ + E J (1 − y/4)^b`); its lists put at most `A·T + sl` on the states read undecided at most
`T` of the time, `A = 2|Q| Nr / (nh log 2)` and `sl = |Q| log(|Q|/δH) / (nh log 2)`, but for
`δH` (`ListSlope`); and more than `tw` of `D'` reaches a string read wrong but never asked with
chance at most `(Lp + 1)(|Q| + 2) κ / tw` (`UnaskedFresh`, Markov).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {K : ℕ}

/-- `LazyRoundCorrect`'s premises for rounds of `nh`-string lists, `E` epochs and `J` blocks, no
`walkStart`-letter prefix above `p₁`. -/
def LazyPremises (σ α : Type*) [Fintype σ] [Fintype α] (C : Random.Level3Settings) (nh E J : ℕ)
    (p₁ : ℝ) : Prop :=
  let nQ := Fintype.card σ
  let nα := Fintype.card α
  let S : ℕ := nQ * ((nQ + 2) * (nα + 1) + 1) + 2 * (nα + 1) + 1
  0 < nh ∧ 8 * nh * p₁ ≤ C.harvestRate ∧ C.harvestRate ≤ 1
    ∧ 0 < C.δ ∧ C.δ < 1 ∧ 0 < C.disagreementRate ∧ C.disagreementRate < 1
    ∧ 36 * (S : ℝ) / C.disagreementRate * Real.log (16 * S * (S + 1) * (E * (E + 1)) / C.δ) + S
      ≤ C.probeBudget
    ∧ 2 * (C.walkStart : ℝ) / C.harvestRate + 1 ≤ E
    ∧ nh * (C.walkStart + 1) + 1 ≤ J

/-- A lazy round's draws from `D'`. -/
def lazyDrawLaw (s : LoopSettings K) (D' : Measure (FreeMonoid α)) :
    Measure (Fin s.epochs → Random.EpochDraws α s.level3.probeBudget s.blocks s.blockSize) :=
  Measure.pi fun _ => (Measure.pi fun _ : Fin s.level3.probeBudget => D').prod
    (Measure.pi fun _ : Fin s.blocks => Measure.pi fun _ : Fin s.blockSize => D')

instance (E P J b : ℕ) : IsProbabilityMeasure (lazyLaw E P J b) := by
  unfold lazyLaw; infer_instance

instance (s : LoopSettings K) (D' : Measure (FreeMonoid α)) [IsProbabilityMeasure D'] :
    IsProbabilityMeasure (lazyDrawLaw s D') := by
  unfold lazyDrawLaw; infer_instance

variable {σ : Type*} [Fintype σ] {Ω' : Type*} [MeasurableSpace Ω'] {μ' : Measure Ω'}

/-- The lazy round from `X₀` reading `fam` at the noise and draws `d`. -/
def lazyRun (fam : SuffixFamily μ' α σ) (s : LoopSettings K) (X₀ : Finset (FreeMonoid α))
    (d : Ω' × (Fin s.epochs → Random.EpochDraws α s.level3.probeBudget s.blocks s.blockSize)) :
    Random.LazyResult α :=
  (Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d.2).eval fun z => fam.read d.1 z

/-- Where the lazy round from `X₀`, reading `fam` over probes from `D'`, fails, ends consistent
with more than `ε + tw` of `D'` in `lazyDisagree`, harvests a list putting more than `A·T + sl` on
the states read undecided at most `T` of the time, or has an epoch past `|Q| + 2` leaves. -/
def lazyFail (fam : SuffixFamily μ' α σ) (s : LoopSettings K) (X₀ : Finset (FreeMonoid α))
    (D' : Measure (FreeMonoid α)) (A sl tw : ℝ) :
    Set (Ω' × (Fin s.epochs → Random.EpochDraws α s.level3.probeBudget s.blocks s.blockSize)) :=
  {d | let r := lazyRun fam s X₀ d
    ¬ (LooseEndsWell fam.M (fun q => fam.readDist q .undecided) A sl D'
        (s.level3.disagreementRate + tw)
        (Random.lazyDisagree (fun z => fam.read d.1 z) s.level3.walkStart r.expanded
          r.hypothesis) r.ending
      ∧ r.peak ≤ Fintype.card σ + 2)}

/-- The slope `A = 2|Q| Nr / (nh log 2)` of a lazy round's lists. -/
def lazySlope (σ : Type*) [Fintype σ] (Nr nh : ℕ) : ℝ :=
  2 * Fintype.card σ * Nr / (nh * Real.log 2)

/-- The slack `sl = |Q| log(|Q|/δH) / (nh log 2)` of a lazy round's lists. -/
def lazySlack (σ : Type*) [Fintype σ] (nh : ℕ) (δH : ℝ) : ℝ :=
  Fintype.card σ * Real.log (Fintype.card σ / δH) / (nh * Real.log 2)

/-- `lazyFail_le`'s bound. -/
def lazyRoundBound (σ : Type*) [Fintype σ] (C : Random.Level3Settings) (E J b : ℕ) (κ : ℝ)
    (Nr L : ℕ) (δH tw : ℝ) : ℝ :=
  C.δ + E * J * (1 - C.harvestRate / 4) ^ b + κ * Nr + δH + (L + 1) * (Fintype.card σ + 2) * κ / tw

/-! ## Measurability -/

omit [Fintype α] [DecidableEq α] in
/-- Under answers measurable in the noise, a condition measurable at each result and reads set
holds on a measurable set of noise. -/
private theorem measurableSet_eval_reads {Ω Q R β : Type*} [MeasurableSpace Ω] [DecidableEq Q]
    [Countable R] (r : Ω → Q → R) (hr : ∀ q a, MeasurableSet {ω | r ω q = a})
    (p : Query Q R β) :
    ∀ Φ : Ω → β × Finset Q → Prop, (∀ v, MeasurableSet {ω | Φ ω v}) →
      MeasurableSet {ω | Φ ω (p.eval (r ω), p.reads (r ω))} := by
  induction p with
  | pure b => exact fun Φ hΦ => hΦ (b, ∅)
  | ask q k ih =>
    intro Φ hΦ
    have e : {ω | Φ ω ((Query.ask q k).eval (r ω), (Query.ask q k).reads (r ω))}
        = ⋃ a, {ω | r ω q = a} ∩ {ω | Φ ω ((k a).eval (r ω), insert q ((k a).reads (r ω)))} := by
      ext ω
      simp only [Query.eval_ask, Query.reads_ask, Set.mem_ofPred_eq, Set.mem_iUnion,
        Set.mem_inter_iff]
      exact ⟨fun h => ⟨_, rfl, h⟩, by rintro ⟨a, rfl, h⟩; exact h⟩
    rw [e]
    exact MeasurableSet.iUnion fun a => (hr q a).inter
      (ih a (fun ω v => Φ ω (v.1, insert q v.2)) fun v => hΦ _)

omit [Fintype α] [DecidableEq α] in
/-- A probe's membership in `lazyDisagree` is fixed by its outcome. -/
private theorem mem_lazyDisagree_of_probe {read read' : FreeMonoid α → ARU} {K : ℕ}
    {X : Finset (FreeMonoid α)} {H : HypothesisDFA (Option α)} {x : FreeMonoid α}
    (h : probeOrAgree (Random.padRead read) H.tree H.edges K (Random.padIn K X x)
      = probeOrAgree (Random.padRead read') H.tree H.edges K (Random.padIn K X x)) :
    x ∈ Random.lazyDisagree read K X H ↔ x ∈ Random.lazyDisagree read' K X H := by
  simp only [Random.lazyDisagree, Random.searchSet, Random.endsSet, Set.mem_union,
    Set.mem_ofPred_eq, h]

omit [Fintype α] [DecidableEq α] in
private theorem measurableSet_mem_lazyDisagree {Ω : Type*} [MeasurableSpace Ω]
    (rd : Ω → FreeMonoid α → ARU) (hrd : ∀ z a, MeasurableSet {ω | rd ω z = a}) (K : ℕ)
    (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α)) (x : FreeMonoid α) :
    MeasurableSet {ω | x ∈ Random.lazyDisagree (rd ω) K X H} := by
  classical
  let Φ : Ω → Outcome (Option α) × Finset (FreeMonoid (Option α)) → Prop := fun _ v =>
    ∃ read, probeOrAgree (Random.padRead read) H.tree H.edges K (Random.padIn K X x) = v.1
      ∧ x ∈ Random.lazyDisagree read K X H
  have e : {ω | x ∈ Random.lazyDisagree (rd ω) K X H}
      = {ω | Φ ω ((probeOrAgreeQ H.tree H.edges K (Random.padIn K X x)).eval
          (Random.padRead (rd ω)),
        (probeOrAgreeQ H.tree H.edges K (Random.padIn K X x)).reads (Random.padRead (rd ω)))} := by
    ext ω
    exact ⟨fun h => ⟨rd ω, rfl, h⟩, fun ⟨read, he, hm⟩ => (mem_lazyDisagree_of_probe he).1 hm⟩
  rw [e]
  exact measurableSet_eval_reads (fun ω => Random.padRead (rd ω)) (fun z a => hrd _ a)
    (probeOrAgreeQ H.tree H.edges K (Random.padIn K X x)) Φ fun _ => MeasurableSet.const _

omit [DecidableEq α] in
private theorem measurable_real_lazyDisagree {Ω : Type*} [MeasurableSpace Ω]
    (rd : Ω → FreeMonoid α → ARU) (hrd : ∀ z a, MeasurableSet {ω | rd ω z = a}) (K : ℕ)
    (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α)) (D' : Measure (FreeMonoid α)) :
    Measurable fun ω => D'.real (Random.lazyDisagree (rd ω) K X H) :=
  ENNReal.measurable_toReal.comp (measurable_measure_of_mem D' _
    (measurableSet_mem_lazyDisagree rd hrd K X H))

theorem measurableSet_lazyFail [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ)
    (s : LoopSettings K) (X₀ : Finset (FreeMonoid α)) (D' : Measure (FreeMonoid α))
    (A sl tw : ℝ) : MeasurableSet (lazyFail fam s X₀ D' A sl tw) := by
  have hrd : ∀ z a, MeasurableSet {ω | fam.read ω z = a} := fun z a =>
    fam.read_measurable z (measurableSet_singleton a)
  refine measurableSet_prod_countable_right _ fun d => ?_
  refine measurableSet_eval_reads (fun ω => fam.read ω) hrd
    (Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d)
    (fun ω v => ¬ (LooseEndsWell fam.M (fun q => fam.readDist q .undecided) A sl D'
        (s.level3.disagreementRate + tw)
        (Random.lazyDisagree (fam.read ω) s.level3.walkStart v.1.expanded v.1.hypothesis)
        v.1.ending ∧ v.1.peak ≤ Fintype.card σ + 2)) fun v => ?_
  rcases v with ⟨r, S⟩
  rcases r with ⟨H, X, ls, e, pk⟩
  cases e with
  | consistent =>
    exact ((measurableSet_le (measurable_real_lazyDisagree _ hrd _ X H D') measurable_const).inter
      (MeasurableSet.const _)).compl
  | harvest s' ℓ => simp only [LooseEndsWell]; exact MeasurableSet.const _
  | failed => simp only [LooseEndsWell]; exact MeasurableSet.const _

/-! ## Wrong reads and the cleaned read -/

/-- A string's read on its state's rarer side. -/
private def wrongRead (fam : SuffixFamily μ' α σ) (z : FreeMonoid α) : ARU :=
  if readSide (fam.readDist (fam.M.eval z.toList)) then .reject else .accept

/-- The family's read with each wrong read made undecided. -/
private def cleanRead (fam : SuffixFamily μ' α σ) (ω : Ω') (z : FreeMonoid α) : ARU :=
  if fam.read ω z = wrongRead fam z then .undecided else fam.read ω z

omit [Fintype σ] [DecidableEq α] in
private theorem readsNoWrong_cleanRead (fam : SuffixFamily μ' α σ) (ω : Ω') :
    ReadsNoWrong fam.M (fun q => readSide (fam.readDist q)) (cleanRead fam ω) := by
  intro z
  show cleanRead fam ω z ≠ wrongRead fam z
  unfold cleanRead
  by_cases h : fam.read ω z = wrongRead fam z
  · rw [if_pos h]
    unfold wrongRead
    split_ifs <;> simp
  · rw [if_neg h]
    exact h

omit [Fintype σ] [DecidableEq α] in
private theorem cleanRead_eq (fam : SuffixFamily μ' α σ) {ω : Ω'} {z : FreeMonoid α}
    (h : fam.read ω z ≠ wrongRead fam z) : cleanRead fam ω z = fam.read ω z := by
  simp [cleanRead, h]

omit [Fintype σ] [DecidableEq α] in
private theorem measurableSet_read [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ)
    (z : FreeMonoid α) (a : ARU) :
    MeasurableSet {ω | fam.read ω z = a} :=
  fam.read_measurable z (measurableSet_singleton a)

omit [Fintype σ] [DecidableEq α] in
private theorem measurableSet_cleanRead [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ)
    (z : FreeMonoid α) (a : ARU) :
    MeasurableSet {ω | cleanRead fam ω z = a} := by
  have hm : Measurable fun ω => cleanRead fam ω z :=
    (measurable_from_top
      (f := fun r : ARU => if r = wrongRead fam z then ARU.undecided else r)).comp
        (fam.read_measurable z)
  exact hm (measurableSet_singleton a)

omit [Fintype σ] [DecidableEq α] in
private theorem readDist_le_one' [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ) (q : σ)
    (r : ARU) : fam.readDist q r ≤ 1 := by
  classical
  unfold SuffixFamily.readDist
  split_ifs
  · exact measureReal_le_one
  · exact zero_le_one

omit [Fintype σ] in
private theorem wrong_le_real [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ)
    (hκ : 0 ≤ fam.κ) (z : FreeMonoid α) : μ'.real {ω | fam.read ω z = wrongRead fam z} ≤ fam.κ := by
  rw [fam.reads_by_state z]
  exact (fam.readDist_wrong _).trans (mul_le_of_le_one_right hκ (readDist_le_one' fam _ _))

omit [Fintype α] [DecidableEq α] in
private theorem le_ofReal_of_real_le {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsFiniteMeasure μ] {S : Set Ω} {c : ℝ} (h : μ.real S ≤ c) : μ S ≤ ENNReal.ofReal c := by
  rw [← ofReal_measureReal (measure_ne_top μ S)]
  exact ENNReal.ofReal_le_ofReal h

omit [Fintype σ] in
private theorem wrong_le [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ) (hκ : 0 ≤ fam.κ)
    (z : FreeMonoid α) : μ' {ω | fam.read ω z = wrongRead fam z} ≤ ENNReal.ofReal fam.κ :=
  le_ofReal_of_real_le (wrong_le_real fam hκ z)

omit [Fintype α] [DecidableEq α] in
/-- A bound on every section over the draws bounds the product. -/
private theorem prod_le_of_sections [IsProbabilityMeasure μ'] {Y : Type*} [MeasurableSpace Y]
    (ν : Measure Y) [IsProbabilityMeasure ν] {S : Set (Ω' × Y)} (hS : MeasurableSet S) {c : ℝ≥0∞}
    (h : ∀ y, μ' {ω | (ω, y) ∈ S} ≤ c) : (μ'.prod ν) S ≤ c := by
  rw [Measure.prod_apply_symm hS]
  calc ∫⁻ y, μ' ((fun x => (x, y)) ⁻¹' S) ∂ν ≤ ∫⁻ _, c ∂ν := lintegral_mono fun y => h y
    _ = c := by rw [lintegral_const, measure_univ, mul_one]

/-! ## G: some asked string reads wrong -/

/-- Some string the round, cut off past `Nr` strings, asks reads wrong. -/
private def lazyG (fam : SuffixFamily μ' α σ) (s : LoopSettings K) (X₀ : Finset (FreeMonoid α))
    (Nr : ℕ) :
    Set (Ω' × (Fin s.epochs → Random.EpochDraws α s.level3.probeBudget s.blocks s.blockSize)) :=
  {d | ∃ z ∈ ((Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d.2).cutoff Nr).reads
      (fam.read d.1), fam.read d.1 z = wrongRead fam z}

omit [Fintype σ] in
private theorem lazyG_le [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ) (hκ : 0 ≤ fam.κ)
    (s : LoopSettings K) (X₀ : Finset (FreeMonoid α)) (Nr : ℕ)
    (ν : Measure (Fin s.epochs → Random.EpochDraws α s.level3.probeBudget s.blocks s.blockSize))
    [IsProbabilityMeasure ν] :
    (μ'.prod ν) (lazyG fam s X₀ Nr) ≤ ENNReal.ofReal (fam.κ * Nr) := by
  have hmeas : MeasurableSet (lazyG fam s X₀ Nr) := by
    refine measurableSet_prod_countable_right _ fun d => ?_
    refine measurableSet_eval_reads (fun ω => fam.read ω) (measurableSet_read fam)
      ((Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d).cutoff Nr)
      (fun ω v => ∃ z ∈ v.2, fam.read ω z = wrongRead fam z) fun v => ?_
    have e : {ω | ∃ z ∈ v.2, fam.read ω z = wrongRead fam z}
        = ⋃ z ∈ v.2, {ω | fam.read ω z = wrongRead fam z} := by
      ext ω
      simp
    rw [e]
    exact Finset.measurableSet_biUnion _ fun z _ => measurableSet_read fam z _
  refine prod_le_of_sections ν hmeas fun d => ?_
  set p := (Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d).cutoff Nr
  have h := adaptiveReadsBound_holds μ' (fun z ω => fam.read ω z) (fun z => {wrongRead fam z})
    fam.κ (Query.nthAsk p) (fun _ => 1) Nr (fun z => fam.read_measurable z) fam.read_indep
    (fun z => wrong_le_real fam hκ z) (fun j r r' _ h => by
      ext z
      simp only [Query.mem_nthAsk]
      rw [Query.newAsks_getElem?_congr p ∅ j (by simp) fun i hi z hz =>
        (h i hi z (Query.mem_nthAsk.2 hz)).symm])
    (fun j r _ => Query.card_nthAsk_le p j r)
  simp only [Nat.cast_one, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h
  refine (measure_mono fun ω hω => ?_).trans h
  obtain ⟨z, hz, hw⟩ := hω
  obtain ⟨hnd, hfs⟩ := Query.newAsks_spec (fam.read ω) p ∅
  rw [Finset.sdiff_empty] at hfs
  have hzL : z ∈ Query.newAsks (fam.read ω) ∅ p := List.mem_toFinset.1 (by rw [hfs]; exact hz)
  have hlen : (Query.newAsks (fam.read ω) ∅ p).length ≤ Nr := by
    rw [← List.toFinset_card_of_nodup hnd, hfs]
    exact Query.card_reads_cutoff_le Nr _ _
  exact ⟨_, (List.idxOf_lt_length_of_mem hzL).trans_le hlen, z,
    Query.mem_nthAsk.2 (List.getElem?_idxOf hzL), hw⟩

/-! ## V: the round under the cleaned read -/

/-- The round fails, ends consistent with more than `ε` of `D'` in its `lazyDisagree`, or has an
epoch past `|Q| + 2` leaves. -/
private def VBad (rd : FreeMonoid α → ARU) (nQ : ℕ) (ε : ℝ) (D' : Measure (FreeMonoid α))
    (k : ℕ) (r : Random.LazyResult α) : Prop :=
  nQ + 2 < r.peak ∨
    match r.ending with
    | .consistent => ε < D'.real (Random.lazyDisagree rd k r.expanded r.hypothesis)
    | .harvest _ _ => False
    | .failed => True

/-- `VBad` of the round under the cleaned read. -/
private def lazyV (fam : SuffixFamily μ' α σ) (s : LoopSettings K) (X₀ : Finset (FreeMonoid α))
    (D' : Measure (FreeMonoid α)) :
    Set (Ω' × (Fin s.epochs → Random.EpochDraws α s.level3.probeBudget s.blocks s.blockSize)) :=
  {d | VBad (cleanRead fam d.1) (Fintype.card σ) s.level3.disagreementRate D' s.level3.walkStart
    ((Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d.2).eval (cleanRead fam d.1))}

private theorem lazyV_le [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ)
    (s : LoopSettings K) (X₀ : Finset (FreeMonoid α))
    (hX₀ : Random.TrieOK s.level3.walkStart X₀) (D' : Measure (FreeMonoid α))
    [IsProbabilityMeasure D'] (p₁ : ℝ) (hK : ∀ᵐ x ∂D', s.level3.walkStart ≤ x.toList.length)
    (hp₁ : ∀ u, D'.real {x | prefixOf x s.level3.walkStart = u} ≤ p₁)
    (hprem : LazyPremises σ α s.level3 s.listSize s.epochs s.blocks p₁) :
    (μ'.prod (lazyDrawLaw s D')) (lazyV fam s X₀ D')
      ≤ ENNReal.ofReal (s.level3.δ
        + s.epochs * s.blocks * (1 - s.level3.harvestRate / 4) ^ s.blockSize) := by
  obtain ⟨-, h8, -, hδ, hδ1, hε, hε1, hP, hE, hJ⟩ := hprem
  have hmeas : MeasurableSet (lazyV fam s X₀ D') := by
    refine measurableSet_prod_countable_right _ fun d => ?_
    refine measurableSet_eval_reads (cleanRead fam) (measurableSet_cleanRead fam)
      (Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d)
      (fun ω v => VBad (cleanRead fam ω) (Fintype.card σ) s.level3.disagreementRate D'
        s.level3.walkStart v.1) fun v => ?_
    rcases v with ⟨⟨H, X, ls, e, pk⟩, S⟩
    cases e with
    | consistent =>
      exact (MeasurableSet.const _).union (measurableSet_lt measurable_const
        (measurable_real_lazyDisagree _ (measurableSet_cleanRead fam) _ X H D'))
    | harvest => simp only [VBad]; exact MeasurableSet.const _
    | failed => simp only [VBad]; exact MeasurableSet.const _
  rw [Measure.prod_apply hmeas]
  refine (lintegral_mono fun ω => ?_).trans_eq (by rw [lintegral_const, measure_univ, mul_one])
  exact le_ofReal_of_real_le (lazyRoundCorrect_of epochCorrect_holds topUpCorrect_holds
    trieMassBound_holds walkStartMoves_holds fam.M (fun q => readSide (fam.readDist q))
    (cleanRead fam ω) s.level3 D' s.level3.walkStart s.listSize s.epochs s.blocks s.blockSize p₁
    X₀ (readsNoWrong_cleanRead fam ω) hX₀ hK hp₁ h8 hδ hδ1 hε hε1 hP hE hJ)

/-! ## H1: the listed strings -/

/-- The list a result harvests, if any. -/
private def harvestList : Option (Random.LazyResult α) → List (FreeMonoid α)
  | some r => match r.ending with
    | .harvest _ ℓ => ℓ
    | _ => []
  | none => []

/-- The round, cut off past `Nr` strings, lists more than `ListSlope`'s count of strings at the
states read undecided at most `T` of the time, for some `T`. -/
private def lazyH (fam : SuffixFamily μ' α σ) (s : LoopSettings K) (X₀ : Finset (FreeMonoid α))
    (Nr : ℕ) (δH : ℝ) :
    Set (Ω' × (Fin s.epochs → Random.EpochDraws α s.level3.probeBudget s.blocks s.blockSize)) :=
  {d | ∃ T, 0 ≤ T ∧
    (2 * Fintype.card σ * Nr * T + Fintype.card σ * Real.log (Fintype.card σ / δH)) / Real.log 2
      < ((harvestList (((Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d.2).cutoff
          Nr).eval (fam.read d.1))).filter
        fun w => fam.readDist (fam.M.eval w.toList) .undecided ≤ T).length}

private theorem lazyH_le [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ)
    (s : LoopSettings K) (X₀ : Finset (FreeMonoid α))
    (hX₀ : Random.TrieOK s.level3.walkStart X₀) (hnh : 0 < s.listSize) (Nr : ℕ) {δH : ℝ}
    (hδH : 0 < δH)
    (ν : Measure (Fin s.epochs → Random.EpochDraws α s.level3.probeBudget s.blocks s.blockSize))
    [IsProbabilityMeasure ν] :
    (μ'.prod ν) (lazyH fam s X₀ Nr δH) ≤ ENNReal.ofReal δH := by
  have hmeas : MeasurableSet (lazyH fam s X₀ Nr δH) := by
    refine measurableSet_prod_countable_right _ fun d => ?_
    exact measurableSet_eval_reads (fun ω => fam.read ω) (measurableSet_read fam)
      ((Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d).cutoff Nr)
      (fun _ v => ∃ T, 0 ≤ T ∧ (2 * Fintype.card σ * Nr * T
          + Fintype.card σ * Real.log (Fintype.card σ / δH)) / Real.log 2
        < ((harvestList v.1).filter
          fun w => fam.readDist (fam.M.eval w.toList) .undecided ≤ T).length)
      fun _ => MeasurableSet.const _
  refine prod_le_of_sections ν hmeas fun d => ?_
  set q := Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d
  have hlist : ∀ r : FreeMonoid α → ARU, (harvestList ((q.cutoff Nr).eval r)).Nodup
      ∧ ∀ w ∈ harvestList ((q.cutoff Nr).eval r), w ∈ (q.cutoff Nr).reads r ∧ r w = .undecided := by
    intro r
    rcases hq : (q.cutoff Nr).eval r with _ | b
    · simp [harvestList]
    obtain ⟨rfl, hreads⟩ := Query.eval_cutoff_eq_some hq
    obtain ⟨-, -, -, hnd, -, hharv, -, hlisted⟩ := lazyRoundShape_holds r s.level3
      s.level3.walkStart s.listSize s.epochs s.blocks s.blockSize X₀ d hnh hX₀
    rw [hreads]
    rcases he : (q.eval r).ending with _ | ⟨s', ℓ⟩ | _
    · simp [harvestList, he]
    · obtain ⟨hl, -⟩ := hharv s' ℓ he
      simp only [harvestList, he]
      exact ⟨hl ▸ hnd s', fun w hw => hlisted s' w (hl ▸ hw)⟩
    · simp [harvestList, he]
  have h := listSlope_holds μ' (fun z ω => fam.read ω z) (fun z => fam.M.eval z.toList)
    (fun q => fam.readDist q .undecided) (q.cutoff Nr) harvestList Nr δH
    (fun z => fam.read_measurable z) fam.read_indep (fun z => (fam.reads_by_state z _).le)
    (fun r => Query.card_reads_cutoff_le Nr q r) hlist hδH
  exact le_ofReal_of_real_le h

omit [DecidableEq α] in
/-- For `ℓ` without repeats and `0 < n ≤ |ℓ|`,
    listLaw ℓ {z | P z} ≤ |{z ∈ ℓ | P z}| / n. -/
private theorem listLaw_real_le {ℓ : List (FreeMonoid α)} (hnd : ℓ.Nodup) {n : ℕ} (hn : 0 < n)
    (hnℓ : n ≤ ℓ.length) (P : FreeMonoid α → Prop) [DecidablePred P] :
    (listLaw ℓ).real {z | P z} ≤ (ℓ.filter fun z => decide (P z)).length / n := by
  have hlen : ℓ.toFinset.card = ℓ.length := List.toFinset_card_of_nodup hnd
  have hν : (∑ z ∈ ℓ.toFinset, Measure.dirac z) Set.univ = ℓ.length := by
    rw [Measure.finsetSum_apply]
    simp only [measure_univ, Finset.sum_const, nsmul_eq_mul, mul_one, hlen]
  have hS : (∑ z ∈ ℓ.toFinset, Measure.dirac z) {z | P z}
      = (ℓ.filter fun z => decide (P z)).length := by
    rw [Measure.finsetSum_apply]
    simp only [Measure.dirac_apply, Set.indicator_apply, Set.mem_ofPred_eq, Pi.one_apply]
    have hfil : ℓ.toFinset.filter P = (ℓ.filter fun z => decide (P z)).toFinset := by
      ext z
      simp
    rw [Finset.sum_boole, hfil, List.toFinset_card_of_nodup (hnd.filter _)]
  have hne : ¬ ((∑ z ∈ ℓ.toFinset, Measure.dirac z) Set.univ = 0
      ∨ (∑ z ∈ ℓ.toFinset, Measure.dirac z) Set.univ = ∞) := by
    rw [hν]
    simp only [Nat.cast_eq_zero, ENNReal.natCast_ne_top, or_false]
    omega
  unfold listLaw popLaw
  rw [if_neg hne, measureReal_def, Measure.smul_apply, smul_eq_mul, hν, hS, ENNReal.toReal_mul,
    ENNReal.toReal_inv, ENNReal.toReal_natCast, ENNReal.toReal_natCast, inv_mul_eq_div]
  exact div_le_div_of_nonneg_left (Nat.cast_nonneg _) (Nat.cast_pos.2 hn) (Nat.cast_le.2 hnℓ)

/-! ## W: unasked wrong reads in a probe's reach -/

/-- The strings a probe `x` walked under `T` may read, pads dropped: a prefix of `x`, then a
midfix of `T`. -/
private def lazyPot (T : DTree (Option α)) (x : FreeMonoid α) : Finset (FreeMonoid α) :=
  (Finset.range (x.toList.length + 1) ×ˢ Random.mids T).image fun im =>
    prefixOf x im.1 * Random.unpad im.2

omit [DecidableEq α] in
private theorem card_lazyPot_le (T : DTree (Option α)) (x : FreeMonoid α) :
    (lazyPot T x).card ≤ (x.toList.length + 1) * (Random.mids T).card := by
  unfold lazyPot
  exact Finset.card_image_le.trans (by rw [Finset.card_product, Finset.card_range])

omit [DecidableEq α] in
private theorem unpad_mul (a b : FreeMonoid (Option α)) :
    Random.unpad (a * b) = Random.unpad a * Random.unpad b := by
  simp [Random.unpad, List.reduceOption_append]

omit [Fintype α] [DecidableEq α] in
private theorem unpad_prefixOf_padIn (K : ℕ) (X : Finset (FreeMonoid α)) (x : FreeMonoid α)
    (p : ℕ) :
    Random.unpad (prefixOf (Random.padIn K X x) p)
      = prefixOf x (p - (K - Random.walkStartIn X x)) := by
  simp only [Random.unpad, prefixOf, Random.padIn, FreeMonoid.toList_ofList, List.take_append,
    List.length_replicate, List.take_replicate, List.reduceOption_append,
    List.reduceOption_replicate_none, List.nil_append, ← List.map_take]
  simp [List.reduceOption]

/-- Reads agreeing on what a probe's walk may read put it in `lazyDisagree` alike. -/
private theorem lazyDisagree_congr {read read' : FreeMonoid α → ARU} (K : ℕ)
    (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α)) {x : FreeMonoid α}
    (h : ∀ z ∈ lazyPot H.tree x, read z = read' z) :
    x ∈ Random.lazyDisagree read K X H ↔ x ∈ Random.lazyDisagree read' K X H := by
  refine mem_lazyDisagree_of_probe (Random.probeOrAgree_congr_zone H.edges K fun j _ hj m hm => ?_)
  simp only [Random.padRead, unpad_mul, unpad_prefixOf_padIn]
  refine h (prefixOf x (j - (K - Random.walkStartIn X x)) * Random.unpad m)
    (Finset.mem_image.2 ⟨(j - (K - Random.walkStartIn X x), m),
      Finset.mem_product.2 ⟨Finset.mem_range.2 ?_, hm⟩, rfl⟩)
  have : (Random.padIn K X x).toList.length = K - Random.walkStartIn X x + x.toList.length := by
    simp [Random.padIn]
  omega

/-- The probes whose walk under `r`'s tree, of at most `|Q| + 2` leaves, may read a string
outside `S` read wrong. -/
private def dirty (fam : SuffixFamily μ' α σ) (ω : Ω') (r : Random.LazyResult α)
    (S : Finset (FreeMonoid α)) : Set (FreeMonoid α) :=
  {x | r.hypothesis.tree.leaves.length ≤ Fintype.card σ + 2
    ∧ ∃ z ∈ lazyPot r.hypothesis.tree x, z ∉ S ∧ fam.read ω z = wrongRead fam z}

omit [DecidableEq α] in
private theorem measurableSet_mem_dirty [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ)
    (r : Random.LazyResult α) (S : Finset (FreeMonoid α)) (x : FreeMonoid α) :
    MeasurableSet {ω | x ∈ dirty fam ω r S} := by
  have e : {ω | x ∈ dirty fam ω r S}
      = {_ω | r.hypothesis.tree.leaves.length ≤ Fintype.card σ + 2}
        ∩ ⋃ z ∈ lazyPot r.hypothesis.tree x,
          {_ω | z ∉ S} ∩ {ω | fam.read ω z = wrongRead fam z} := by
    ext ω
    simp only [dirty, Set.mem_ofPred_eq, Set.mem_inter_iff, Set.mem_iUnion, exists_prop]
  rw [e]
  exact (MeasurableSet.const _).inter (Finset.measurableSet_biUnion _ fun z _ =>
    (MeasurableSet.const _).inter (measurableSet_read fam z _))

/-- More than `tw` of `D'` may read a string the round never asked and that reads wrong. -/
private def lazyW (fam : SuffixFamily μ' α σ) (s : LoopSettings K) (X₀ : Finset (FreeMonoid α))
    (D' : Measure (FreeMonoid α)) (tw : ℝ) :
    Set (Ω' × (Fin s.epochs → Random.EpochDraws α s.level3.probeBudget s.blocks s.blockSize)) :=
  {d | tw < D'.real (dirty fam d.1
    ((Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d.2).eval (fam.read d.1))
    ((Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d.2).reads (fam.read d.1)))}

private theorem lazyW_le [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ) (hκ : 0 ≤ fam.κ)
    (s : LoopSettings K) (X₀ : Finset (FreeMonoid α)) (D' : Measure (FreeMonoid α))
    [IsProbabilityMeasure D'] (Lp : ℕ) (hLp : ∀ᵐ x ∂D', x.toList.length ≤ Lp) {tw : ℝ}
    (htw : 0 < tw)
    (ν : Measure (Fin s.epochs → Random.EpochDraws α s.level3.probeBudget s.blocks s.blockSize))
    [IsProbabilityMeasure ν] :
    (μ'.prod ν) (lazyW fam s X₀ D' tw)
      ≤ ENNReal.ofReal ((Lp + 1) * (Fintype.card σ + 2) * fam.κ / tw) := by
  have hmeas : MeasurableSet (lazyW fam s X₀ D' tw) := by
    refine measurableSet_prod_countable_right _ fun d => ?_
    exact measurableSet_eval_reads (fun ω => fam.read ω) (measurableSet_read fam)
      (Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d)
      (fun ω v => tw < D'.real (dirty fam ω v.1 v.2)) fun v => measurableSet_lt measurable_const
        (ENNReal.measurable_toReal.comp (measurable_measure_of_mem D' _
          (measurableSet_mem_dirty fam v.1 v.2)))
  refine prod_le_of_sections ν hmeas fun d => ?_
  set p := Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d
  set Dt : Ω' → Set (FreeMonoid α) := fun ω =>
    dirty fam ω (p.eval (fam.read ω)) (p.reads (fam.read ω)) with hDt
  have hmemd : ∀ x, MeasurableSet {ω | x ∈ Dt ω} := fun x =>
    measurableSet_eval_reads (fun ω => fam.read ω) (measurableSet_read fam) p
      (fun ω v => x ∈ dirty fam ω v.1 v.2) fun v => measurableSet_mem_dirty fam v.1 v.2 x
  set c : ℕ := (Lp + 1) * (Fintype.card σ + 1) with hc
  have hx : ∀ x : FreeMonoid α, x.toList.length ≤ Lp →
      μ' {ω | x ∈ Dt ω} ≤ c * ENNReal.ofReal fam.κ := by
    intro x hxL
    set A : FreeMonoid α → Set (Random.LazyResult α) := fun z =>
      {r | r.hypothesis.tree.leaves.length ≤ Fintype.card σ + 2
        ∧ z ∈ lazyPot r.hypothesis.tree x} with hA
    have hmA : ∀ z, MeasurableSet {ω | p.eval (fam.read ω) ∈ A z} := fun z =>
      measurableSet_eval_reads (fun ω => fam.read ω) (measurableSet_read fam) p
        (fun _ v => v.1 ∈ A z) fun _ => MeasurableSet.const _
    have hpt : ∀ ω, ∑' z, {ω | p.eval (fam.read ω) ∈ A z}.indicator (1 : Ω' → ℝ≥0∞) ω ≤ c := by
      intro ω
      set T := (p.eval (fam.read ω)).hypothesis.tree
      rw [tsum_eq_sum (s := lazyPot T x) fun z hz =>
        Set.indicator_of_notMem (fun h => hz h.2) _]
      by_cases hl : T.leaves.length ≤ Fintype.card σ + 2
      · calc ∑ z ∈ lazyPot T x, {ω | p.eval (fam.read ω) ∈ A z}.indicator (1 : Ω' → ℝ≥0∞) ω
            ≤ ∑ _z ∈ lazyPot T x, (1 : ℝ≥0∞) :=
              Finset.sum_le_sum fun z _ =>
                Set.indicator_apply_le' (fun _ => le_rfl) (fun _ => zero_le_one)
          _ = (lazyPot T x).card := by rw [Finset.sum_const, nsmul_eq_mul, mul_one]
          _ ≤ c := by
              have h1 := card_lazyPot_le T x
              have h2 := Random.mids_card T
              have h3 : (lazyPot T x).card ≤ c :=
                h1.trans (Nat.mul_le_mul (by omega) (by omega))
              exact_mod_cast h3
      · rw [Finset.sum_eq_zero fun z _ => Set.indicator_of_notMem (fun h => hl h.1) _]
        exact zero_le
    have hcount : ∑' z, μ' {ω | p.eval (fam.read ω) ∈ A z} ≤ c :=
      calc ∑' z, μ' {ω | p.eval (fam.read ω) ∈ A z}
          = ∑' z, ∫⁻ ω, {ω | p.eval (fam.read ω) ∈ A z}.indicator 1 ω ∂μ' :=
            tsum_congr fun z => (lintegral_indicator_one (hmA z)).symm
        _ = ∫⁻ ω, ∑' z, {ω | p.eval (fam.read ω) ∈ A z}.indicator 1 ω ∂μ' :=
            (lintegral_tsum fun z => (measurable_one.indicator (hmA z)).aemeasurable).symm
        _ ≤ ∫⁻ _, (c : ℝ≥0∞) ∂μ' := lintegral_mono hpt
        _ = c := by rw [lintegral_const, measure_univ, mul_one]
    calc μ' {ω | x ∈ Dt ω}
        ≤ μ' (⋃ z, {ω | z ∉ p.reads (fam.read ω) ∧ p.eval (fam.read ω) ∈ A z
            ∧ fam.read ω z ∈ ({wrongRead fam z} : Set ARU)}) :=
          measure_mono fun ω hω => by
            obtain ⟨hl, z, hz, hzS, hw⟩ := hω
            exact Set.mem_iUnion.2 ⟨z, hzS, ⟨hl, hz⟩, hw⟩
      _ ≤ ∑' z, μ' {ω | z ∉ p.reads (fam.read ω) ∧ p.eval (fam.read ω) ∈ A z
            ∧ fam.read ω z ∈ ({wrongRead fam z} : Set ARU)} := measure_iUnion_le _
      _ = ∑' z, μ' {ω | z ∉ p.reads (fam.read ω) ∧ p.eval (fam.read ω) ∈ A z}
            * μ' {ω | fam.read ω z ∈ ({wrongRead fam z} : Set ARU)} :=
          tsum_congr fun z => unaskedFresh_holds μ' (fun z ω => fam.read ω z) p z (A z)
            {wrongRead fam z} (fun z => fam.read_measurable z) fam.read_indep
      _ ≤ ∑' z, μ' {ω | p.eval (fam.read ω) ∈ A z} * ENNReal.ofReal fam.κ :=
          ENNReal.tsum_le_tsum fun z =>
            mul_le_mul' (measure_mono fun ω h => h.2) (wrong_le fam hκ z)
      _ = (∑' z, μ' {ω | p.eval (fam.read ω) ∈ A z}) * ENNReal.ofReal fam.κ :=
          ENNReal.tsum_mul_right
      _ ≤ c * ENNReal.ofReal fam.κ := mul_le_mul' hcount le_rfl
  have hint : ∫⁻ ω, D' (Dt ω) ∂μ' ≤ c * ENNReal.ofReal fam.κ := by
    have hS : MeasurableSet {q : Ω' × FreeMonoid α | q.2 ∈ Dt q.1} :=
      measurableSet_prod_countable_right _ fun x => hmemd x
    calc ∫⁻ ω, D' (Dt ω) ∂μ' = (μ'.prod D') {q | q.2 ∈ Dt q.1} := (Measure.prod_apply hS).symm
      _ = ∫⁻ x, μ' {ω | x ∈ Dt ω} ∂D' := Measure.prod_apply_symm hS
      _ ≤ ∫⁻ _, c * ENNReal.ofReal fam.κ ∂D' :=
          lintegral_mono_ae (hLp.mono fun x hxL => hx x hxL)
      _ = c * ENNReal.ofReal fam.κ := by rw [lintegral_const, measure_univ, mul_one]
  have hDm : Measurable fun ω => D' (Dt ω) := measurable_measure_of_mem D' Dt hmemd
  calc μ' {ω | (ω, d) ∈ lazyW fam s X₀ D' tw}
      ≤ μ' {ω | ENNReal.ofReal tw ≤ D' (Dt ω)} := by
        refine measure_mono fun ω hω => ?_
        have hω' : tw < (D' (Dt ω)).toReal := hω
        exact (ENNReal.ofReal_le_ofReal hω'.le).trans ENNReal.ofReal_toReal_le
    _ ≤ (∫⁻ ω, D' (Dt ω) ∂μ') / ENNReal.ofReal tw :=
        meas_ge_le_lintegral_div hDm.aemeasurable (ENNReal.ofReal_pos.2 htw).ne'
          ENNReal.ofReal_ne_top
    _ ≤ (c * ENNReal.ofReal fam.κ) / ENNReal.ofReal tw := ENNReal.div_le_div_right hint _
    _ = ENNReal.ofReal (c * fam.κ / tw) := by
        rw [ENNReal.ofReal_div_of_pos htw, ENNReal.ofReal_mul (Nat.cast_nonneg _),
          ENNReal.ofReal_natCast]
    _ ≤ ENNReal.ofReal ((Lp + 1) * (Fintype.card σ + 2) * fam.κ / tw) := by
        refine ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_right
          (mul_le_mul_of_nonneg_right ?_ hκ) htw.le)
        rw [hc]
        push_cast
        exact mul_le_mul_of_nonneg_left (by linarith) (by positivity)

theorem lazyFail_le [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ) (hκ : 0 ≤ fam.κ)
    (s : LoopSettings K) (X₀ : Finset (FreeMonoid α))
    (hX₀ : Random.TrieOK s.level3.walkStart X₀) (D' : Measure (FreeMonoid α))
    [IsProbabilityMeasure D'] (p₁ δH tw : ℝ) (Nr Lp : ℕ)
    (hlen : ∀ᵐ x ∂D', s.level3.walkStart ≤ x.toList.length ∧ x.toList.length ≤ Lp)
    (hp₁ : ∀ u, D'.real {x | prefixOf x s.level3.walkStart = u} ≤ p₁)
    (hprem : LazyPremises σ α s.level3 s.listSize s.epochs s.blocks p₁)
    (hNr : lazyReadCap s.level3 s.epochs s.blocks s.blockSize Lp (Fintype.card σ)
      (Fintype.card α) ≤ Nr)
    (hδH : 0 < δH) (htw : 0 < tw) :
    (μ'.prod (lazyDrawLaw s D'))
        (lazyFail fam s X₀ D' (lazySlope σ Nr s.listSize) (lazySlack σ s.listSize δH) tw)
      ≤ ENNReal.ofReal
        (lazyRoundBound σ s.level3 s.epochs s.blocks s.blockSize fam.κ Nr Lp δH tw) := by
  have hnh : 0 < s.listSize := hprem.1
  have hy1 : s.level3.harvestRate ≤ 1 := hprem.2.2.1
  have hδ : 0 < s.level3.δ := hprem.2.2.2.1
  have hK : ∀ᵐ x ∂D', s.level3.walkStart ≤ x.toList.length := hlen.mono fun x h => h.1
  have hLp : ∀ᵐ x ∂D', x.toList.length ≤ Lp := hlen.mono fun x h => h.2
  set π := lazyDrawLaw s D'
  -- the draws are at most `Lp` long but on a null set
  set N : Set (Ω' × (Fin s.epochs → Random.EpochDraws α s.level3.probeBudget s.blocks
      s.blockSize)) := Set.univ ×ˢ {d | ¬ LazyDrawsAll (fun x => x.toList.length ≤ Lp) d} with hNdef
  have hN : (μ'.prod π) N = 0 := by
    have hpe : ∀ e, MeasurePreserving (fun d : Fin s.epochs → Random.EpochDraws α
        s.level3.probeBudget s.blocks s.blockSize => d e) π
        ((Measure.pi fun _ : Fin s.level3.probeBudget => D').prod
          (Measure.pi fun _ : Fin s.blocks => Measure.pi fun _ : Fin s.blockSize => D')) :=
      fun e => measurePreserving_eval _ e
    have hae : ∀ᵐ d ∂π, LazyDrawsAll (fun x => x.toList.length ≤ Lp) d := by
      unfold LazyDrawsAll
      refine ae_all_iff.2 fun e => (ae_all_iff.2 fun i => ?_).and
        (ae_all_iff.2 fun j => ae_all_iff.2 fun t => ?_)
      · exact ((measurePreserving_eval (fun _ => D') i).comp
          (measurePreserving_fst.comp (hpe e))).quasiMeasurePreserving.ae hLp
      · exact ((measurePreserving_eval (fun _ => D') t).comp
          ((measurePreserving_eval (fun _ => Measure.pi fun _ : Fin s.blockSize => D') j).comp
            (measurePreserving_snd.comp (hpe e)))).quasiMeasurePreserving.ae hLp
    rw [hNdef, Measure.prod_prod, ae_iff.1 hae, mul_zero]
  have hsub : lazyFail fam s X₀ D' (lazySlope σ Nr s.listSize) (lazySlack σ s.listSize δH) tw
      ⊆ lazyV fam s X₀ D' ∪ lazyG fam s X₀ Nr ∪ lazyH fam s X₀ Nr δH ∪ lazyW fam s X₀ D' tw
        ∪ N := by
    rintro ⟨ω, d⟩ hfail
    by_contra hno
    simp only [Set.mem_union, not_or] at hno
    obtain ⟨⟨⟨⟨hV, hG⟩, hH⟩, hW⟩, hN'⟩ := hno
    apply hfail
    set p := Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d with hp
    have hL : LazyDrawsAll (fun x => x.toList.length ≤ Lp) d := by
      by_contra h
      exact hN' (Set.mk_mem_prod (Set.mem_univ _) h)
    have hV' : ¬ VBad (cleanRead fam ω) (Fintype.card σ) s.level3.disagreementRate D'
        s.level3.walkStart (p.eval (cleanRead fam ω)) := hV
    simp only [VBad, not_or, not_lt] at hV'
    obtain ⟨hpk, hend⟩ := hV'
    have hcard : (p.reads (cleanRead fam ω)).card ≤ Nr :=
      (card_reads_lazyRoundQ_le s.level3 s.level3.walkStart s.listSize X₀ d (cleanRead fam ω) rfl
        (Fintype.card σ) Lp hL hpk).trans hNr
    obtain ⟨hcl1, -⟩ := Query.eval_cutoff hcard
    -- off G, the cut-off run reads as the cleaned one
    have hnw : ∀ z ∈ (p.cutoff Nr).reads (fam.read ω), fam.read ω z ≠ wrongRead fam z :=
      fun z hz hw => hG ⟨z, hz, hw⟩
    obtain ⟨he1, -⟩ := Query.eval_congr (fam.read ω) (p.cutoff Nr) fun z hz =>
      cleanRead_eq fam (hnw z hz)
    have hsome : (p.cutoff Nr).eval (fam.read ω) = some (p.eval (cleanRead fam ω)) := by
      rw [← he1]
      exact hcl1
    obtain ⟨hr, hreads⟩ := Query.eval_cutoff_eq_some hsome
    show LooseEndsWell fam.M (fun q => fam.readDist q .undecided) (lazySlope σ Nr s.listSize)
        (lazySlack σ s.listSize δH) D' (s.level3.disagreementRate + tw)
        (Random.lazyDisagree (fam.read ω) s.level3.walkStart (p.eval (fam.read ω)).expanded
          (p.eval (fam.read ω)).hypothesis) (p.eval (fam.read ω)).ending
      ∧ (p.eval (fam.read ω)).peak ≤ Fintype.card σ + 2
    rw [hr] at hpk hend
    refine ⟨?_, hpk⟩
    obtain ⟨-, -, -, hnd, -, hharv, -, -⟩ := lazyRoundShape_holds (fam.read ω) s.level3
      s.level3.walkStart s.listSize s.epochs s.blocks s.blockSize X₀ d hnh hX₀
    obtain ⟨-, -, -, -, -, -, hleaves, -⟩ := lazyRoundShape_holds (cleanRead fam ω) s.level3
      s.level3.walkStart s.listSize s.epochs s.blocks s.blockSize X₀ d hnh hX₀
    rw [← hp] at hnd hharv hleaves
    rw [hr] at hleaves
    rcases hrend : (p.eval (fam.read ω)).ending with _ | ⟨s', ℓ⟩ | _
    · -- consistent: W bounds where the real read's disagreements leave the cleaned read's
      simp only [hrend, not_lt] at hend
      have hleaf : (p.eval (fam.read ω)).hypothesis.tree.leaves.length ≤ Fintype.card σ + 2 :=
        (hleaves (by rw [hrend]; simp)).trans hpk
      have hWd : D'.real (dirty fam ω (p.eval (fam.read ω)) (p.reads (fam.read ω))) ≤ tw :=
        not_lt.1 hW
      have hsub' : Random.lazyDisagree (fam.read ω) s.level3.walkStart
            (p.eval (fam.read ω)).expanded (p.eval (fam.read ω)).hypothesis
          ⊆ Random.lazyDisagree (cleanRead fam ω) s.level3.walkStart
            (p.eval (fam.read ω)).expanded (p.eval (fam.read ω)).hypothesis
            ∪ dirty fam ω (p.eval (fam.read ω)) (p.reads (fam.read ω)) := by
        intro x hx
        by_cases hxd : x ∈ dirty fam ω (p.eval (fam.read ω)) (p.reads (fam.read ω))
        · exact Or.inr hxd
        refine Or.inl ((lazyDisagree_congr (read := fam.read ω) (read' := cleanRead fam ω) _ _ _
          fun z hz => ?_).1 hx)
        by_cases hw : fam.read ω z = wrongRead fam z
        · exact (hxd ⟨hleaf, z, hz, fun hzS => hnw z (by rw [hreads]; exact hzS) hw, hw⟩).elim
        · exact (cleanRead_eq fam hw).symm
      show D'.real _ ≤ _
      calc _ ≤ _ := measureReal_mono hsub'
        _ ≤ _ := measureReal_union_le _ _
        _ ≤ _ := add_le_add hend hWd
    · -- a harvest: H1 bounds its list
      obtain ⟨hl, hnl⟩ := hharv s' ℓ hrend
      have hndℓ : ℓ.Nodup := hl ▸ hnd s'
      intro T hT
      have hlist : harvestList ((p.cutoff Nr).eval (fam.read ω)) = ℓ := by
        rw [hsome, hr]
        simp [harvestList, hrend]
      have hcnt : (((ℓ.filter fun w => fam.readDist (fam.M.eval w.toList) .undecided ≤ T).length
          : ℕ) : ℝ) ≤ (2 * Fintype.card σ * Nr * T
            + Fintype.card σ * Real.log (Fintype.card σ / δH)) / Real.log 2 := by
        refine not_lt.1 fun hlt => hH ?_
        have h' : (2 * Fintype.card σ * Nr * T
            + Fintype.card σ * Real.log (Fintype.card σ / δH)) / Real.log 2
            < ((harvestList ((p.cutoff Nr).eval (fam.read ω))).filter
              fun w => fam.readDist (fam.M.eval w.toList) .undecided ≤ T).length := by
          rw [hlist]
          exact hlt
        exact ⟨T, hT, h'⟩
      have hl2 : Real.log 2 ≠ 0 := (Real.log_pos one_lt_two).ne'
      have hnh' : (s.listSize : ℝ) ≠ 0 := by positivity
      calc (listLaw ℓ).real {z | fam.readDist (fam.M.eval z.toList) .undecided ≤ T}
          ≤ (ℓ.filter fun w => fam.readDist (fam.M.eval w.toList) .undecided ≤ T).length
              / s.listSize := listLaw_real_le hndℓ hnh hnl _
        _ ≤ (2 * Fintype.card σ * Nr * T + Fintype.card σ * Real.log (Fintype.card σ / δH))
              / Real.log 2 / s.listSize := div_le_div_of_nonneg_right hcnt (Nat.cast_nonneg _)
        _ = lazySlope σ Nr s.listSize * T + lazySlack σ s.listSize δH := by
            unfold lazySlope lazySlack
            field_simp
    · simp [hrend] at hend
  have hκN : 0 ≤ fam.κ * Nr := mul_nonneg hκ (Nat.cast_nonneg _)
  have hV0 : 0 ≤ s.level3.δ
      + s.epochs * s.blocks * (1 - s.level3.harvestRate / 4) ^ s.blockSize := by
    have : 0 ≤ 1 - s.level3.harvestRate / 4 := by linarith
    exact add_nonneg hδ.le (mul_nonneg (by positivity) (pow_nonneg this _))
  have hW0 : 0 ≤ (Lp + 1) * (Fintype.card σ + 2) * fam.κ / tw :=
    div_nonneg (mul_nonneg (by positivity) hκ) htw.le
  calc (μ'.prod π) (lazyFail fam s X₀ D' (lazySlope σ Nr s.listSize)
        (lazySlack σ s.listSize δH) tw)
      ≤ (μ'.prod π) (lazyV fam s X₀ D') + (μ'.prod π) (lazyG fam s X₀ Nr)
        + (μ'.prod π) (lazyH fam s X₀ Nr δH) + (μ'.prod π) (lazyW fam s X₀ D' tw)
        + (μ'.prod π) N := by
        refine (measure_mono hsub).trans ((measure_union_le _ _).trans (add_le_add
          ((measure_union_le _ _).trans (add_le_add ((measure_union_le _ _).trans
            (add_le_add (measure_union_le _ _) le_rfl)) le_rfl)) le_rfl))
    _ ≤ ENNReal.ofReal (s.level3.δ
          + s.epochs * s.blocks * (1 - s.level3.harvestRate / 4) ^ s.blockSize)
        + ENNReal.ofReal (fam.κ * Nr) + ENNReal.ofReal δH
        + ENNReal.ofReal ((Lp + 1) * (Fintype.card σ + 2) * fam.κ / tw) + 0 := by
        rw [hN]
        exact add_le_add (add_le_add (add_le_add (add_le_add
          (lazyV_le fam s X₀ hX₀ D' p₁ hK hp₁ hprem) (lazyG_le fam hκ s X₀ Nr π))
          (lazyH_le fam s X₀ hX₀ hnh Nr hδH π)) (lazyW_le fam hκ s X₀ D' Lp hLp htw π)) le_rfl
    _ = ENNReal.ofReal
          (lazyRoundBound σ s.level3 s.epochs s.blocks s.blockSize fam.κ Nr Lp δH tw) := by
        rw [add_zero, ← ENNReal.ofReal_add hV0 hκN,
          ← ENNReal.ofReal_add (add_nonneg hV0 hκN) hδH.le,
          ← ENNReal.ofReal_add (add_nonneg (add_nonneg hV0 hκN) hδH.le) hW0]
        rfl

end OrthoDFA

end
