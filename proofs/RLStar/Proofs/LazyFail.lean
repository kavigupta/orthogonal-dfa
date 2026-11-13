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
/-- A program's result and reads, under answers measurable in the noise, meet a condition
measurable at each result and reads in a measurable set of noise. -/
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
  sorry -- closes by: G (AdaptiveReadsBound), V (LazyRoundCorrect), H1 (ListSlope), W (UnaskedFresh)

end OrthoDFA

end
