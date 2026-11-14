import RLStar.Proofs.LoopRenoise
import RLStar.Proofs.LazyZone
import RLStar.Proofs.QueryCutoff
import RLStar.Proofs.QueryBadTail
import RLStar.Proofs.UnaskedFresh
import RLStar.Proofs.AdaptiveReads
import RLStar.Proofs.LazyRound
import RLStar.Proofs.LazyEpoch
import RLStar.Proofs.LazyTopUp
import RLStar.Proofs.TrieMass
import RLStar.Proofs.PadOracle
import RLStar.Proofs.HeavyTrie

/-!
# A lazy round read through a suffix family

The lazy round from a trie `X₀`, reading a suffix family over probes from `D'`, fails its claim
(`lazyFail`) with chance at most

    2δ + E J (1 − y/4)^b + κ Nr + (Lp + 1)(|Q| + 2) κ / tw

(`lazyFail_le`). Some string it asks reads on its state's rarer side with chance at most `κ Nr`,
the round asking at most `Nr` (`lazyG_le`); off that the round reads as if no read were wrong, so
`LazyRoundCorrect` holds but for `δ + E J (1 − y/4)^b` (`lazyV_le`) and Level 3 tames the
harvest's law but for `δ` (`cleanHarvest_le`); and more than `tw` of `D'` reaches a string read
wrong but never asked with chance at most `(Lp + 1)(|Q| + 2) κ / tw` (`lazyW_le`, by
`UnaskedFresh` and Markov). Off all these the round reads as its cleaned self, its harvest's law
moved by at most `tw`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {K : ℕ}

/-- `LazyRoundCorrect`'s premises for rounds of one-string lists, `E` epochs and `J` blocks, no
`walkStart`-letter prefix above `p₁`. -/
def LazyPremises (σ α : Type*) [Fintype σ] [Fintype α] (C : Random.Level3Settings) (E J : ℕ)
    (p₁ : ℝ) : Prop :=
  let nQ := Fintype.card σ
  let nα := Fintype.card α
  let S : ℕ := nQ * ((nQ + 2) * (nα + 1) + 1) + 2 * (nα + 1) + 1
  0 < C.harvestRate ∧ 8 * p₁ ≤ C.harvestRate ∧ C.harvestRate ≤ C.disagreementRate / 48
    ∧ 0 < C.δ ∧ C.δ < 1 ∧ 0 < C.disagreementRate ∧ C.disagreementRate < 1
    ∧ 36 * (S : ℝ) / C.disagreementRate * Real.log (16 * S * (S + 1) * (E * (E + 1)) / C.δ) + S
      ≤ C.probeBudget
    ∧ 2 * (C.walkStart : ℝ) / C.harvestRate + 1 ≤ E
    ∧ C.walkStart + 2 ≤ J

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
with more than `ε + tw` of `D'` in `lazyDisagree`, harvests with a law yielding less than `y`,
putting more than `pc` on a string or more than `A·T + sl` of itself, past `m`, on the states read
undecided at most `T` of the time, or has an epoch past `|Q| + 2` leaves. -/
def lazyFail (fam : SuffixFamily μ' α σ) (s : LoopSettings K) (X₀ : Finset (FreeMonoid α))
    (D' : Measure (FreeMonoid α)) (A sl m pc y tw : ℝ) :
    Set (Ω' × (Fin s.epochs → Random.EpochDraws α s.level3.probeBudget s.blocks s.blockSize)) :=
  {d | let r := lazyRun fam s X₀ d
    ¬ (LooseEndsWell fam.M (fun q => fam.readDist q .undecided) A sl m pc y D'
        (s.level3.disagreementRate + tw)
        (Random.lazyDisagree (fun z => fam.read d.1 z) s.level3.walkStart r.expanded r.hypothesis)
        (Random.lazyHarvestLaw (fun z => fam.read d.1 z) D' s.level3.walkStart r.expanded
          r.hypothesis)
        r.ending
      ∧ r.peak ≤ Fintype.card σ + 2)}

/-- Level 3's slope at padded probes at most `Lp + walkStart` long, times `1 + κ`. -/
def lazySlope (σ : Type*) [Fintype σ] (κ : ℝ) (C : Random.Level3Settings) (Lp : ℕ) : ℝ :=
  (1 + κ) * (Fintype.card σ + 1) * (6 / C.harvestRate + 16 * (2 + Nat.clog 2 (Lp + C.walkStart)))

/-- Level 3's slack `η / y` at cells of mass at most `p₀`, at the least epoch level. -/
def lazySlack (σ α : Type*) [Fintype σ] [Fintype α] (C : Random.Level3Settings) (E : ℕ)
    (p₀ : ℝ) : ℝ :=
  let nQ := Fintype.card σ
  let nα := Fintype.card α
  let Kt : ℕ := (1 + (nQ + 1) ^ 3 * (nα + 1)) ^ nQ * (nQ + 3) ^ ((nQ + 2) * (nα + 1))
  p₀ * Real.log (8 * (nQ + 1) * Kt * (E * (E + 1)) / C.δ) / C.harvestRate

/-- `lazyFail_le`'s bound. -/
def lazyRoundBound (σ : Type*) [Fintype σ] (C : Random.Level3Settings) (E J b : ℕ) (κ : ℝ)
    (Nr L : ℕ) (tw : ℝ) : ℝ :=
  2 * C.δ + E * J * (1 - C.harvestRate / 4) ^ b + κ * Nr
    + (L + 1) * (Fintype.card σ + 2) * κ / tw

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

omit [Fintype α] [DecidableEq α] in
private theorem measurableSet_probe {Ω : Type*} [MeasurableSpace Ω]
    (rd : Ω → FreeMonoid α → ARU) (hrd : ∀ z a, MeasurableSet {ω | rd ω z = a}) (K : ℕ)
    (H : HypothesisDFA (Option α)) (y : FreeMonoid (Option α)) (P : Outcome (Option α) → Prop) :
    MeasurableSet {ω | P (probeOrAgree (Random.padRead (rd ω)) H.tree H.edges K y)} := by
  classical
  exact measurableSet_eval_reads (fun ω => Random.padRead (rd ω)) (fun _ a => hrd _ a)
    (probeOrAgreeQ H.tree H.edges K y) (fun _ v => P v.1) fun _ => MeasurableSet.const _

omit [DecidableEq α] in
private theorem measurable_real_lazyHarvestLaw {Ω : Type*} [MeasurableSpace Ω]
    (rd : Ω → FreeMonoid α → ARU) (hrd : ∀ z a, MeasurableSet {ω | rd ω z = a})
    (D' : Measure (FreeMonoid α)) (K : ℕ) (X : Finset (FreeMonoid α))
    (H : HypothesisDFA (Option α)) (site : Random.Level3Site) (S : Set (FreeMonoid α)) :
    Measurable fun ω => (Random.lazyHarvestLaw (rd ω) D' K X H site).real S := by
  classical
  refine ENNReal.measurable_toReal.comp ?_
  have e : ∀ ω, Random.lazyHarvestLaw (rd ω) D' K X H site S
      = ∑' y, (((Random.siteHarvest (Random.padRead (rd ω)) K H site y).filter
          (· ∈ Random.unpad ⁻¹' S)).length : ℝ≥0∞) * (D'.map (Random.padIn K X)) {y} := by
    intro ω
    unfold Random.lazyHarvestLaw
    rw [Measure.map_apply (measurable_of_countable _) (Set.to_countable _).measurableSet,
      Random.harvestLaw_apply, lintegral_countable']
  simp_rw [e]
  refine Measurable.ennreal_tsum fun y => Measurable.mul_const ?_ _
  refine measurable_from_nat.comp (measurable_to_countable' fun n => ?_)
  cases site with
  | ends =>
    exact measurableSet_probe rd hrd K H y
      (fun o => ((Random.endsString o).filter (· ∈ Random.unpad ⁻¹' S)).length = n)
  | middles =>
    exact measurableSet_probe rd hrd K H y
      (fun o => ((Random.middleString o).filter (· ∈ Random.unpad ⁻¹' S)).length = n)

omit [DecidableEq α] in
private theorem measurableSet_harvestWell {Ω : Type*} [MeasurableSpace Ω] {τ : Type*}
    [Fintype τ] (M : _root_.DFA α τ) (U : τ → ℝ) (A sl m pc y : ℝ)
    (law : Ω → Measure (FreeMonoid α)) (hlaw : ∀ S, Measurable fun ω => (law ω).real S) :
    MeasurableSet {ω | (∀ T, 0 ≤ T → (law ω).real {z | U (M.eval z.toList) ≤ T}
        ≤ (A * T + sl) * (law ω).real Set.univ + m)
      ∧ (∀ t, (law ω).real {t} ≤ pc) ∧ y ≤ (law ω).real Set.univ} := by
  classical
  set v : Ω → (Finset τ → ℝ) × ℝ := fun ω =>
    (fun Q => (law ω).real {z | M.eval z.toList ∈ Q}, (law ω).real Set.univ)
  have hv : Measurable v := (measurable_pi_lambda _ fun Q => hlaw _).prodMk (hlaw _)
  set C : Set ((Finset τ → ℝ) × ℝ) := ⋂ T : ℝ, ⋂ _ : 0 ≤ T,
    {p | p.1 (Finset.univ.filter fun q => U q ≤ T) ≤ (A * T + sl) * p.2 + m}
  have hC : IsClosed C := isClosed_iInter fun T => isClosed_iInter fun _ =>
    isClosed_le ((continuous_apply _).comp continuous_fst)
      ((continuous_const.mul continuous_snd).add continuous_const)
  have hset : ∀ T, {z : FreeMonoid α | U (M.eval z.toList) ≤ T}
      = {z | M.eval z.toList ∈ Finset.univ.filter fun q => U q ≤ T} := fun T => by
    ext z; simp
  have h1 : MeasurableSet {ω | ∀ T, 0 ≤ T → (law ω).real {z | U (M.eval z.toList) ≤ T}
      ≤ (A * T + sl) * (law ω).real Set.univ + m} := by
    have := hv hC.measurableSet
    convert this using 1
    ext ω
    simp only [Set.mem_preimage, C, Set.mem_iInter, Set.mem_ofPred_eq, v, hset]
  have h2 : MeasurableSet {ω | ∀ t, (law ω).real {t} ≤ pc} := by
    rw [Set.ofPred_forall]
    exact MeasurableSet.iInter fun t => measurableSet_le (hlaw _) measurable_const
  exact h1.inter (h2.inter (measurableSet_le measurable_const (hlaw _)))

theorem measurableSet_lazyFail [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ)
    (s : LoopSettings K) (X₀ : Finset (FreeMonoid α)) (D' : Measure (FreeMonoid α))
    (A sl m pc y tw : ℝ) : MeasurableSet (lazyFail fam s X₀ D' A sl m pc y tw) := by
  have hrd : ∀ z a, MeasurableSet {ω | fam.read ω z = a} := fun z a =>
    fam.read_measurable z (measurableSet_singleton a)
  refine measurableSet_prod_countable_right _ fun d => ?_
  refine measurableSet_eval_reads (fun ω => fam.read ω) hrd
    (Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d)
    (fun ω v => ¬ (LooseEndsWell fam.M (fun q => fam.readDist q .undecided) A sl m pc y D'
        (s.level3.disagreementRate + tw)
        (Random.lazyDisagree (fam.read ω) s.level3.walkStart v.1.expanded v.1.hypothesis)
        (Random.lazyHarvestLaw (fam.read ω) D' s.level3.walkStart v.1.expanded v.1.hypothesis)
        v.1.ending ∧ v.1.peak ≤ Fintype.card σ + 2)) fun v => ?_
  rcases v with ⟨r, S⟩
  rcases r with ⟨H, X, ls, e, pk⟩
  cases e with
  | consistent =>
    exact ((measurableSet_le (measurable_real_lazyDisagree _ hrd _ X H D') measurable_const).inter
      (MeasurableSet.const _)).compl
  | harvest s' ℓ =>
    exact ((measurableSet_harvestWell fam.M (fun q => fam.readDist q .undecided) A sl m pc y
      (fun ω => Random.lazyHarvestLaw (fam.read ω) D' s.level3.walkStart X H s')
      fun S => measurable_real_lazyHarvestLaw _ hrd D' _ X H s' S).inter
        (MeasurableSet.const _)).compl
  | failed => simp only [LooseEndsWell]; exact MeasurableSet.const _

/-! ## Wrong reads -/

omit [Fintype σ] [DecidableEq α] in
private theorem measurableSet_read [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ)
    (z : FreeMonoid α) (a : ARU) :
    MeasurableSet {ω | fam.read ω z = a} :=
  fam.read_measurable z (measurableSet_singleton a)

omit [Fintype σ] [DecidableEq α] in
private theorem measurableSet_cleanRead [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ)
    (z : FreeMonoid α) (a : ARU) :
    MeasurableSet {ω | fam.cleanRead ω z = a} :=
  fam.measurable_cleanRead z (measurableSet_singleton a)

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
    (hκ : 0 ≤ fam.κ) (z : FreeMonoid α) : μ'.real {ω | fam.read ω z = fam.wrongRead z} ≤ fam.κ := by
  rw [fam.reads_by_state z]
  exact (fam.readDist_wrong _).trans (mul_le_of_le_one_right hκ (readDist_le_one' fam _ _))

omit [Fintype α] [DecidableEq α] in
private theorem le_ofReal_of_real_le {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsFiniteMeasure μ] {S : Set Ω} {c : ℝ} (h : μ.real S ≤ c) : μ S ≤ ENNReal.ofReal c := by
  rw [← ofReal_measureReal (measure_ne_top μ S)]
  exact ENNReal.ofReal_le_ofReal h

omit [Fintype σ] in
private theorem wrong_le [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ) (hκ : 0 ≤ fam.κ)
    (z : FreeMonoid α) : μ' {ω | fam.read ω z = fam.wrongRead z} ≤ ENNReal.ofReal fam.κ :=
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
def lazyG (fam : SuffixFamily μ' α σ) (s : LoopSettings K) (X₀ : Finset (FreeMonoid α))
    (Nr : ℕ) :
    Set (Ω' × (Fin s.epochs → Random.EpochDraws α s.level3.probeBudget s.blocks s.blockSize)) :=
  {d | ∃ z ∈ ((Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d.2).cutoff Nr).reads
      (fam.read d.1), fam.read d.1 z = fam.wrongRead z}

omit [Fintype σ] in
theorem lazyG_le [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ) (hκ : 0 ≤ fam.κ)
    (s : LoopSettings K) (X₀ : Finset (FreeMonoid α)) (Nr : ℕ)
    (ν : Measure (Fin s.epochs → Random.EpochDraws α s.level3.probeBudget s.blocks s.blockSize))
    [IsProbabilityMeasure ν] :
    (μ'.prod ν) (lazyG fam s X₀ Nr) ≤ ENNReal.ofReal (fam.κ * Nr) := by
  have hmeas : MeasurableSet (lazyG fam s X₀ Nr) := by
    refine measurableSet_prod_countable_right _ fun d => ?_
    refine measurableSet_eval_reads (fun ω => fam.read ω) (measurableSet_read fam)
      ((Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d).cutoff Nr)
      (fun ω v => ∃ z ∈ v.2, fam.read ω z = fam.wrongRead z) fun v => ?_
    have e : {ω | ∃ z ∈ v.2, fam.read ω z = fam.wrongRead z}
        = ⋃ z ∈ v.2, {ω | fam.read ω z = fam.wrongRead z} := by
      ext ω
      simp
    rw [e]
    exact Finset.measurableSet_biUnion _ fun z _ => measurableSet_read fam z _
  refine prod_le_of_sections ν hmeas fun d => ?_
  set p := (Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d).cutoff Nr
  have h := adaptiveReadsBound_holds μ' (fun z ω => fam.read ω z) (fun z => {fam.wrongRead z})
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
def VBad (rd : FreeMonoid α → ARU) (nQ : ℕ) (ε : ℝ) (D' : Measure (FreeMonoid α))
    (k : ℕ) (r : Random.LazyResult α) : Prop :=
  nQ + 2 < r.peak ∨
    match r.ending with
    | .consistent => ε < D'.real (Random.lazyDisagree rd k r.expanded r.hypothesis)
    | .harvest _ _ => False
    | .failed => True

/-- `VBad` of the round under the cleaned read. -/
def lazyV (fam : SuffixFamily μ' α σ) (s : LoopSettings K) (X₀ : Finset (FreeMonoid α))
    (D' : Measure (FreeMonoid α)) :
    Set (Ω' × (Fin s.epochs → Random.EpochDraws α s.level3.probeBudget s.blocks s.blockSize)) :=
  {d | VBad (fam.cleanRead d.1) (Fintype.card σ) s.level3.disagreementRate D' s.level3.walkStart
    ((Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d.2).eval (fam.cleanRead d.1))}

theorem lazyV_le [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ)
    (s : LoopSettings K) (X₀ : Finset (FreeMonoid α))
    (hX₀ : Random.TrieOK s.level3.walkStart X₀) (D' : Measure (FreeMonoid α))
    [IsProbabilityMeasure D'] (p₁ : ℝ) (hK : ∀ᵐ x ∂D', s.level3.walkStart ≤ x.toList.length)
    (hp₁ : ∀ u, D'.real {x | prefixOf x s.level3.walkStart = u} ≤ p₁) (hnh : s.listSize = 1)
    (hprem : LazyPremises σ α s.level3 s.epochs s.blocks p₁) :
    (μ'.prod (lazyDrawLaw s D')) (lazyV fam s X₀ D')
      ≤ ENNReal.ofReal (s.level3.δ
        + s.epochs * s.blocks * (1 - s.level3.harvestRate / 4) ^ s.blockSize) := by
  obtain ⟨-, h8, -, hδ, hδ1, hε, hε1, hP, hE, hJ⟩ := hprem
  have hmeas : MeasurableSet (lazyV fam s X₀ D') := by
    refine measurableSet_prod_countable_right _ fun d => ?_
    refine measurableSet_eval_reads fam.cleanRead (measurableSet_cleanRead fam)
      (Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d)
      (fun ω v => VBad (fam.cleanRead ω) (Fintype.card σ) s.level3.disagreementRate D'
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
    (fam.cleanRead ω) s.level3 D' s.level3.walkStart s.listSize s.epochs s.blocks s.blockSize p₁
    X₀ (fam.readsNoWrong_cleanRead ω) hX₀ hK hp₁ (by rw [hnh]; simpa using h8) hδ hδ1 hε hε1 hP
    hE (by rw [hnh]; omega))

/-! ## W: unasked wrong reads in a probe's reach -/

/-- The strings a probe `x` walked under `T` may read, pads dropped: a prefix of `x`, then a
midfix of `T`. -/
def lazyPot (T : DTree (Option α)) (x : FreeMonoid α) : Finset (FreeMonoid α) :=
  (Finset.range (x.toList.length + 1) ×ˢ Random.mids T).image fun im =>
    prefixOf x im.1 * Random.unpad im.2

omit [DecidableEq α] in
private theorem card_lazyPot_le (T : DTree (Option α)) (x : FreeMonoid α) :
    (lazyPot T x).card ≤ (x.toList.length + 1) * (Random.mids T).card := by
  unfold lazyPot
  exact Finset.card_image_le.trans (by rw [Finset.card_product, Finset.card_range])

/-- Reads agreeing on what a probe's walk may read walk it alike. -/
private theorem probe_congr_pot {read read' : FreeMonoid α → ARU} (K : ℕ)
    (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α)) {x : FreeMonoid α}
    (h : ∀ z ∈ lazyPot H.tree x, read z = read' z) :
    probeOrAgree (Random.padRead read) H.tree H.edges K (Random.padIn K X x)
      = probeOrAgree (Random.padRead read') H.tree H.edges K (Random.padIn K X x) := by
  refine Random.probeOrAgree_congr_zone H.edges K fun j _ hj m hm => ?_
  simp only [Random.padRead, Random.unpad_mul, Random.unpad_prefixOf_padIn]
  refine h (prefixOf x (j - (K - Random.walkStartIn X x)) * Random.unpad m)
    (Finset.mem_image.2 ⟨(j - (K - Random.walkStartIn X x), m),
      Finset.mem_product.2 ⟨Finset.mem_range.2 ?_, hm⟩, rfl⟩)
  have : (Random.padIn K X x).toList.length = K - Random.walkStartIn X x + x.toList.length := by
    simp [Random.padIn]
  omega

private theorem lazyDisagree_congr {read read' : FreeMonoid α → ARU} (K : ℕ)
    (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α)) {x : FreeMonoid α}
    (h : ∀ z ∈ lazyPot H.tree x, read z = read' z) :
    x ∈ Random.lazyDisagree read K X H ↔ x ∈ Random.lazyDisagree read' K X H :=
  mem_lazyDisagree_of_probe (probe_congr_pot K X H h)

/-- Reads agreeing on each walk's reach off `B` harvest laws at most `D'(B)` apart on a set. -/
private theorem lazyHarvestLaw_real_le {read read' : FreeMonoid α → ARU}
    (D' : Measure (FreeMonoid α)) [IsProbabilityMeasure D'] (K : ℕ) (X : Finset (FreeMonoid α))
    (H : HypothesisDFA (Option α)) (s : Random.Level3Site) {B : Set (FreeMonoid α)}
    (h : ∀ x ∉ B, ∀ z ∈ lazyPot H.tree x, read z = read' z) (S : Set (FreeMonoid α)) :
    (Random.lazyHarvestLaw read D' K X H s).real S
      ≤ (Random.lazyHarvestLaw read' D' K X H s).real S + D'.real B := by
  classical
  have e : ∀ r : FreeMonoid α → ARU, Random.lazyHarvestLaw r D' K X H s S
      = ∫⁻ x, (((Random.siteHarvest (Random.padRead r) K H s (Random.padIn K X x)).filter
          (· ∈ Random.unpad ⁻¹' S)).length : ℝ≥0∞) ∂D' := by
    intro r
    unfold Random.lazyHarvestLaw
    rw [Measure.map_apply (measurable_of_countable _) (Set.to_countable _).measurableSet,
      Random.harvestLaw_apply,
      lintegral_map (measurable_of_countable _) (measurable_of_countable _)]
  have hpt : ∀ x, (((Random.siteHarvest (Random.padRead read) K H s (Random.padIn K X x)).filter
        (· ∈ Random.unpad ⁻¹' S)).length : ℝ≥0∞)
      ≤ (((Random.siteHarvest (Random.padRead read') K H s (Random.padIn K X x)).filter
        (· ∈ Random.unpad ⁻¹' S)).length : ℝ≥0∞) + B.indicator 1 x := by
    intro x
    by_cases hx : x ∈ B
    · rw [Set.indicator_of_mem hx, Pi.one_apply]
      refine le_add_left ?_
      exact_mod_cast (List.length_filter_le _ _).trans
        (Random.siteHarvest_length (Random.padRead read) K H s _)
    · have hs : Random.siteHarvest (Random.padRead read) K H s (Random.padIn K X x)
          = Random.siteHarvest (Random.padRead read') K H s (Random.padIn K X x) := by
        cases s <;> simp only [Random.siteHarvest, probe_congr_pot K X H (h x hx)]
      rw [hs]
      exact le_self_add
  have hB : MeasurableSet B := (Set.to_countable B).measurableSet
  have hle : Random.lazyHarvestLaw read D' K X H s S
      ≤ Random.lazyHarvestLaw read' D' K X H s S + D' B := by
    rw [e, e, ← lintegral_indicator_one hB, ← lintegral_add_right _ (measurable_one.indicator hB)]
    exact lintegral_mono hpt
  rw [measureReal_def, measureReal_def, measureReal_def,
    ← ENNReal.toReal_add (measure_ne_top _ _) (measure_ne_top _ _)]
  exact ENNReal.toReal_mono (ENNReal.add_ne_top.2 ⟨measure_ne_top _ _, measure_ne_top _ _⟩) hle

/-- The probes whose walk under `r`'s tree, of at most `|Q| + 2` leaves, may read a string
outside `S` read wrong. -/
def dirty (fam : SuffixFamily μ' α σ) (ω : Ω') (r : Random.LazyResult α)
    (S : Finset (FreeMonoid α)) : Set (FreeMonoid α) :=
  {x | r.hypothesis.tree.leaves.length ≤ Fintype.card σ + 2
    ∧ ∃ z ∈ lazyPot r.hypothesis.tree x, z ∉ S ∧ fam.read ω z = fam.wrongRead z}

omit [DecidableEq α] in
private theorem measurableSet_mem_dirty [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ)
    (r : Random.LazyResult α) (S : Finset (FreeMonoid α)) (x : FreeMonoid α) :
    MeasurableSet {ω | x ∈ dirty fam ω r S} := by
  have e : {ω | x ∈ dirty fam ω r S}
      = {_ω | r.hypothesis.tree.leaves.length ≤ Fintype.card σ + 2}
        ∩ ⋃ z ∈ lazyPot r.hypothesis.tree x,
          {_ω | z ∉ S} ∩ {ω | fam.read ω z = fam.wrongRead z} := by
    ext ω
    simp only [dirty, Set.mem_ofPred_eq, Set.mem_inter_iff, Set.mem_iUnion, exists_prop]
  rw [e]
  exact (MeasurableSet.const _).inter (Finset.measurableSet_biUnion _ fun z _ =>
    (MeasurableSet.const _).inter (measurableSet_read fam z _))

/-- More than `tw` of `D'` may read a string the round never asked and that reads wrong. -/
def lazyW (fam : SuffixFamily μ' α σ) (s : LoopSettings K) (X₀ : Finset (FreeMonoid α))
    (D' : Measure (FreeMonoid α)) (tw : ℝ) :
    Set (Ω' × (Fin s.epochs → Random.EpochDraws α s.level3.probeBudget s.blocks s.blockSize)) :=
  {d | tw < D'.real (dirty fam d.1
    ((Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d.2).eval (fam.read d.1))
    ((Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d.2).reads (fam.read d.1)))}

theorem lazyW_le [IsProbabilityMeasure μ'] (fam : SuffixFamily μ' α σ) (hκ : 0 ≤ fam.κ)
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
            ∧ fam.read ω z ∈ ({fam.wrongRead z} : Set ARU)}) :=
          measure_mono fun ω hω => by
            obtain ⟨hl, z, hz, hzS, hw⟩ := hω
            exact Set.mem_iUnion.2 ⟨z, hzS, ⟨hl, hz⟩, hw⟩
      _ ≤ ∑' z, μ' {ω | z ∉ p.reads (fam.read ω) ∧ p.eval (fam.read ω) ∈ A z
            ∧ fam.read ω z ∈ ({fam.wrongRead z} : Set ARU)} := measure_iUnion_le _
      _ = ∑' z, μ' {ω | z ∉ p.reads (fam.read ω) ∧ p.eval (fam.read ω) ∈ A z}
            * μ' {ω | fam.read ω z ∈ ({fam.wrongRead z} : Set ARU)} :=
          tsum_congr fun z => unaskedFresh_holds μ' (fun z ω => fam.read ω z) p z (A z)
            {fam.wrongRead z} (fun z => fam.read_measurable z) fam.read_indep
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
    (s : LoopSettings K) (hnh : s.listSize = 1) (X₀ : Finset (FreeMonoid α))
    (hX₀ : Random.TrieOK s.level3.walkStart X₀) (D' : Measure (FreeMonoid α))
    [IsProbabilityMeasure D'] (p₁ θ tw : ℝ) (Nr Lp : ℕ)
    (hlen : ∀ᵐ x ∂D', s.level3.walkStart ≤ x.toList.length ∧ x.toList.length ≤ Lp)
    (hp₁ : ∀ u, D'.real {x | prefixOf x s.level3.walkStart = u} ≤ p₁)
    (hlight : ∀ c : FreeMonoid α, c.toList.length < s.level3.walkStart → c ∉ X₀ →
      D'.real {x | prefixOf x c.toList.length = c} ≤ θ)
    (hprem : LazyPremises σ α s.level3 s.epochs s.blocks p₁)
    (hNr : lazyReadCap s.level3 s.epochs s.blocks s.blockSize Lp (Fintype.card σ)
      (Fintype.card α) ≤ Nr)
    (htw : 0 < tw) :
    (μ'.prod (lazyDrawLaw s D'))
        (lazyFail fam s X₀ D' (lazySlope σ fam.κ s.level3 Lp)
          (lazySlack σ α s.level3 s.epochs (max θ p₁)) (2 * tw) (max θ p₁)
          (s.level3.harvestRate - tw) tw)
      ≤ ENNReal.ofReal
        (lazyRoundBound σ s.level3 s.epochs s.blocks s.blockSize fam.κ Nr Lp tw) := by
  have hprem' := hprem
  obtain ⟨hy, -, hy48, hδ, hδ1, hε, hε1, hP, hE, -⟩ := hprem'
  have hK : ∀ᵐ x ∂D', s.level3.walkStart ≤ x.toList.length := hlen.mono fun x h => h.1
  have hLp : ∀ᵐ x ∂D', x.toList.length ≤ Lp := hlen.mono fun x h => h.2
  have hp₀ : 0 ≤ max θ p₁ := le_max_of_le_right (measureReal_nonneg.trans (hp₁ 1))
  set π := lazyDrawLaw s D'
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
  set A := lazySlope σ fam.κ s.level3 Lp with hAdef
  set sl := lazySlack σ α s.level3 s.epochs (max θ p₁) with hsldef
  have hA : 0 ≤ A := mul_nonneg (mul_nonneg (by linarith) (by positivity))
    (add_nonneg (div_nonneg (by norm_num) hy.le) (by positivity))
  have hsl : 0 ≤ sl := by
    rw [hsldef]
    unfold lazySlack
    dsimp only
    refine div_nonneg (mul_nonneg hp₀ (Real.log_nonneg ?_)) hy.le
    rw [le_div_iff₀ hδ, one_mul]
    have hE1 : (1 : ℝ) ≤ s.epochs := by
      have : 0 ≤ 2 * (s.level3.walkStart : ℝ) / s.level3.harvestRate := by positivity
      linarith
    have hKt : (1 : ℝ) ≤ (((1 + (Fintype.card σ + 1) ^ 3 * (Fintype.card α + 1)) ^ Fintype.card σ
        * (Fintype.card σ + 3) ^ ((Fintype.card σ + 2) * (Fintype.card α + 1)) : ℕ) : ℝ) := by
      exact_mod_cast Nat.one_le_iff_ne_zero.2 (by positivity)
    have hQ : (1 : ℝ) ≤ Fintype.card σ + 1 := by linarith [(Fintype.card σ).cast_nonneg (α := ℝ)]
    have h1 := one_le_mul_of_one_le_of_one_le hQ hKt
    have h2 := one_le_mul_of_one_le_of_one_le h1
      (one_le_mul_of_one_le_of_one_le hE1 (by linarith : (1 : ℝ) ≤ s.epochs + 1))
    nlinarith
  have hsub : lazyFail fam s X₀ D' A sl (2 * tw) (max θ p₁) (s.level3.harvestRate - tw) tw
      ⊆ lazyV fam s X₀ D' ∪ lazyG fam s X₀ Nr ∪ fam.cleanHarvestBad s.level3 s.listSize X₀ D' A sl
        ∪ lazyW fam s X₀ D' tw ∪ N := by
    rintro ⟨ω, d⟩ hfail
    by_contra hno
    simp only [Set.mem_union, not_or] at hno
    obtain ⟨⟨⟨⟨hV, hG⟩, hH⟩, hW⟩, hN'⟩ := hno
    apply hfail
    set p := Random.lazyRoundQ s.level3 s.level3.walkStart s.listSize X₀ d with hp
    have hL : LazyDrawsAll (fun x => x.toList.length ≤ Lp) d := by
      by_contra h
      exact hN' (Set.mk_mem_prod (Set.mem_univ _) h)
    have hV' : ¬ VBad (fam.cleanRead ω) (Fintype.card σ) s.level3.disagreementRate D'
        s.level3.walkStart (p.eval (fam.cleanRead ω)) := hV
    have hH' : ¬ ∃ s' ℓ, (p.eval (fam.cleanRead ω)).ending = .harvest s' ℓ
        ∧ ¬ Random.HarvestTame fam.M (fun q => fam.readDist q .undecided) A sl
          s.level3.harvestRate (Random.lazyHarvestLaw (fam.cleanRead ω) D' s.level3.walkStart X₀
            (p.eval (fam.cleanRead ω)).hypothesis s') := hH
    simp only [VBad, not_or, not_lt] at hV'
    obtain ⟨hpk, hend⟩ := hV'
    have hcard : (p.reads (fam.cleanRead ω)).card ≤ Nr :=
      (card_reads_lazyRoundQ_le s.level3 s.level3.walkStart s.listSize X₀ d (fam.cleanRead ω) rfl
        (Fintype.card σ) Lp hL hpk).trans hNr
    obtain ⟨hcl1, -⟩ := Query.eval_cutoff hcard
    have hnw : ∀ z ∈ (p.cutoff Nr).reads (fam.read ω), fam.read ω z ≠ fam.wrongRead z :=
      fun z hz hw => hG ⟨z, hz, hw⟩
    obtain ⟨he1, -⟩ := Query.eval_congr (fam.read ω) (p.cutoff Nr) fun z hz =>
      fam.cleanRead_eq (hnw z hz)
    have hsome : (p.cutoff Nr).eval (fam.read ω) = some (p.eval (fam.cleanRead ω)) := by
      rw [← he1]
      exact hcl1
    obtain ⟨hr, hreads⟩ := Query.eval_cutoff_eq_some hsome
    have hexp : (p.eval (fam.read ω)).expanded = X₀ := by
      rw [hp, hnh]
      exact (Random.lazyRoundQ_one (fam.read ω) s.level3 s.level3.walkStart X₀ d).1
    show LooseEndsWell fam.M (fun q => fam.readDist q .undecided) A sl (2 * tw) (max θ p₁)
        (s.level3.harvestRate - tw) D' (s.level3.disagreementRate + tw)
        (Random.lazyDisagree (fam.read ω) s.level3.walkStart (p.eval (fam.read ω)).expanded
          (p.eval (fam.read ω)).hypothesis)
        (Random.lazyHarvestLaw (fam.read ω) D' s.level3.walkStart (p.eval (fam.read ω)).expanded
          (p.eval (fam.read ω)).hypothesis)
        (p.eval (fam.read ω)).ending
      ∧ (p.eval (fam.read ω)).peak ≤ Fintype.card σ + 2
    rw [hr] at hpk hend hH'
    refine ⟨?_, hpk⟩
    obtain ⟨-, -, -, -, -, -, hleaves, -⟩ := lazyRoundShape_holds (fam.read ω) s.level3
      s.level3.walkStart s.listSize s.epochs s.blocks s.blockSize X₀ d
      (by rw [hnh]; exact Nat.one_pos) hX₀
    rw [← hp] at hleaves
    have hWd : D'.real (dirty fam ω (p.eval (fam.read ω)) (p.reads (fam.read ω))) ≤ tw :=
      not_lt.1 hW
    have hagree : (p.eval (fam.read ω)).ending ≠ .failed →
        ∀ x ∉ dirty fam ω (p.eval (fam.read ω)) (p.reads (fam.read ω)),
          ∀ z ∈ lazyPot (p.eval (fam.read ω)).hypothesis.tree x,
            fam.read ω z = fam.cleanRead ω z := by
      intro hf x hxd z hz
      by_cases hw : fam.read ω z = fam.wrongRead z
      · exact (hxd ⟨(hleaves hf).trans hpk, z, hz,
          fun hzS => hnw z (by rw [hreads]; exact hzS) hw, hw⟩).elim
      · exact (fam.cleanRead_eq hw).symm
    rcases hrend : (p.eval (fam.read ω)).ending with _ | ⟨s', ℓ⟩ | _
    · simp only [hrend, not_lt] at hend
      have hcl := hagree (by rw [hrend]; simp)
      have hsub' : Random.lazyDisagree (fam.read ω) s.level3.walkStart
            (p.eval (fam.read ω)).expanded (p.eval (fam.read ω)).hypothesis
          ⊆ Random.lazyDisagree (fam.cleanRead ω) s.level3.walkStart
            (p.eval (fam.read ω)).expanded (p.eval (fam.read ω)).hypothesis
            ∪ dirty fam ω (p.eval (fam.read ω)) (p.reads (fam.read ω)) := by
        intro x hx
        by_cases hxd : x ∈ dirty fam ω (p.eval (fam.read ω)) (p.reads (fam.read ω))
        · exact Or.inr hxd
        exact Or.inl ((lazyDisagree_congr _ _ _ (hcl x hxd)).1 hx)
      show D'.real _ ≤ _
      calc _ ≤ _ := measureReal_mono hsub'
        _ ≤ _ := measureReal_union_le _ _
        _ ≤ _ := add_le_add hend hWd
    · have hcl := hagree (by rw [hrend]; simp)
      have hT : Random.HarvestTame fam.M (fun q => fam.readDist q .undecided) A sl
          s.level3.harvestRate (Random.lazyHarvestLaw (fam.cleanRead ω) D' s.level3.walkStart X₀
            (p.eval (fam.read ω)).hypothesis s') :=
        not_not.1 fun h => hH' ⟨s', ℓ, hrend, h⟩
      obtain ⟨hT1, hTy⟩ := hT
      have h1 : ∀ S, (Random.lazyHarvestLaw (fam.read ω) D' s.level3.walkStart X₀
            (p.eval (fam.read ω)).hypothesis s').real S
          ≤ (Random.lazyHarvestLaw (fam.cleanRead ω) D' s.level3.walkStart X₀
            (p.eval (fam.read ω)).hypothesis s').real S + tw := fun S =>
        (lazyHarvestLaw_real_le D' _ X₀ _ s' hcl S).trans (by linarith)
      have h2 : ∀ S, (Random.lazyHarvestLaw (fam.cleanRead ω) D' s.level3.walkStart X₀
            (p.eval (fam.read ω)).hypothesis s').real S
          ≤ (Random.lazyHarvestLaw (fam.read ω) D' s.level3.walkStart X₀
            (p.eval (fam.read ω)).hypothesis s').real S + tw := fun S =>
        (lazyHarvestLaw_real_le D' _ X₀ _ s' (fun x hx z hz => (hcl x hx z hz).symm) S).trans
          (by linarith)
      rw [hexp]
      simp only [LooseEndsWell]
      refine ⟨fun T hT0 => ?_, fun t => ?_, ?_⟩
      · have hc0 : 0 ≤ A * T + sl := add_nonneg (mul_nonneg hA hT0) hsl
        by_cases hc1 : A * T + sl ≤ 1
        · calc _ ≤ _ := h1 _
            _ ≤ (A * T + sl) * (Random.lazyHarvestLaw (fam.cleanRead ω) D' s.level3.walkStart X₀
                (p.eval (fam.read ω)).hypothesis s').real Set.univ + tw := by linarith [hT1 T hT0]
            _ ≤ (A * T + sl) * ((Random.lazyHarvestLaw (fam.read ω) D' s.level3.walkStart X₀
                (p.eval (fam.read ω)).hypothesis s').real Set.univ + tw) + tw := by
                linarith [mul_le_mul_of_nonneg_left (h2 Set.univ) hc0]
            _ ≤ _ := by linarith [mul_le_of_le_one_left htw.le hc1]
        · calc _ ≤ (Random.lazyHarvestLaw (fam.read ω) D' s.level3.walkStart X₀
                (p.eval (fam.read ω)).hypothesis s').real Set.univ :=
                measureReal_mono (Set.subset_univ _)
            _ ≤ (A * T + sl) * (Random.lazyHarvestLaw (fam.read ω) D' s.level3.walkStart X₀
                (p.eval (fam.read ω)).hypothesis s').real Set.univ :=
                le_mul_of_one_le_left measureReal_nonneg (not_le.1 hc1).le
            _ ≤ _ := by linarith
      · exact (lazyHarvestLaw_atom_le (fam.read ω) D' _ X₀ _ s' θ hX₀.1 hK hlight t).trans
          (max_le_max le_rfl (ciSup_le hp₁))
      · linarith [h2 Set.univ]
    · simp [hrend] at hend
  have hκN : 0 ≤ fam.κ * Nr := mul_nonneg hκ (Nat.cast_nonneg _)
  have hV0 : 0 ≤ s.level3.δ
      + s.epochs * s.blocks * (1 - s.level3.harvestRate / 4) ^ s.blockSize := by
    have : 0 ≤ 1 - s.level3.harvestRate / 4 := by linarith
    exact add_nonneg hδ.le (mul_nonneg (by positivity) (pow_nonneg this _))
  have hW0 : 0 ≤ (Lp + 1) * (Fintype.card σ + 2) * fam.κ / tw :=
    div_nonneg (mul_nonneg (by positivity) hκ) htw.le
  calc (μ'.prod π) (lazyFail fam s X₀ D' A sl (2 * tw) (max θ p₁) (s.level3.harvestRate - tw) tw)
      ≤ (μ'.prod π) (lazyV fam s X₀ D') + (μ'.prod π) (lazyG fam s X₀ Nr)
        + (μ'.prod π) (fam.cleanHarvestBad s.level3 s.listSize X₀ D' A sl)
        + (μ'.prod π) (lazyW fam s X₀ D' tw) + (μ'.prod π) N := by
        refine (measure_mono hsub).trans ((measure_union_le _ _).trans (add_le_add
          ((measure_union_le _ _).trans (add_le_add ((measure_union_le _ _).trans
            (add_le_add (measure_union_le _ _) le_rfl)) le_rfl)) le_rfl))
    _ ≤ ENNReal.ofReal (s.level3.δ
          + s.epochs * s.blocks * (1 - s.level3.harvestRate / 4) ^ s.blockSize)
        + ENNReal.ofReal (fam.κ * Nr) + ENNReal.ofReal s.level3.δ
        + ENNReal.ofReal ((Lp + 1) * (Fintype.card σ + 2) * fam.κ / tw) + 0 := by
        rw [hN]
        exact add_le_add (add_le_add (add_le_add (add_le_add
          (lazyV_le fam s X₀ hX₀ D' p₁ hK hp₁ hnh hprem) (lazyG_le fam hκ s X₀ Nr π))
          (fam.cleanHarvest_le hκ s.level3 s.listSize hnh X₀ hX₀.1 D' Lp hlen hp₁ hlight hδ hδ1
            hy hε1 hP)) (lazyW_le fam hκ s X₀ D' Lp hLp htw π)) le_rfl
    _ = ENNReal.ofReal
          (lazyRoundBound σ s.level3 s.epochs s.blocks s.blockSize fam.κ Nr Lp tw) := by
        rw [add_zero, ← ENNReal.ofReal_add hV0 hκN,
          ← ENNReal.ofReal_add (add_nonneg hV0 hκN) hδ.le,
          ← ENNReal.ofReal_add (add_nonneg (add_nonneg hV0 hκN) hδ.le) hW0]
        congr 1
        unfold lazyRoundBound
        ring

end OrthoDFA

end
