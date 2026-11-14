import RLStar.Proofs.PadCanon
import RLStar.Proofs.LoopSpec
import RLStar.Proofs.RandomCorrect
import RLStar.Proofs.FamilyOracle
import RLStar.Proofs.QueryLevel3
import Mathlib.Probability.ProductMeasure

/-!
# The clean read over the alphabet with a pad, as a read oracle

The clean read makes each read on its state's rarer side undecided, so it is never wrong, and a
state reads it undecided with chance `U' ≤ (1 + κ) U`. Over the alphabet with a pad, a canonical
string (`padIn K X (unpad z) = z`) reads cleanly as its pads dropped, and any other string reads a
fresh draw with the same law (`padOracle`). Canonical strings are distinct once their pads are
dropped, so these reads are independent: a `ReadOracle` for the target looping on pads.

Level 3 on probes padded from `X` reads only canonical strings, and so does the harvest's sampler
under its hypothesis (`RLStar.Proofs.PadCanon`), so `Level3RoundCorrect` on this oracle speaks of
the epoch under the clean read (`epochBad_le`). With lists of one string the round's trie stays
`X`, and its harvest is its harvesting epoch's (`lazyRoundQ_one`), so the round harvests a law
that is not tame with chance at most `δ`, the epochs' levels summed (`cleanHarvest_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] {σ : Type*} {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-! ## Measurability in the noise -/

/-- Under answers measurable in the noise, a condition measurable at each result holds on a
measurable set of noise. -/
theorem measurableSet_query_eval {Ω' Q R β : Type*} [MeasurableSpace Ω'] [Countable R]
    (r : Ω' → Q → R) (hr : ∀ q a, MeasurableSet {ω | r ω q = a}) (p : Query Q R β) :
    ∀ Φ : Ω' → β → Prop, (∀ b, MeasurableSet {ω | Φ ω b}) →
      MeasurableSet {ω | Φ ω (p.eval (r ω))} := by
  induction p with
  | pure b => exact fun Φ hΦ => hΦ b
  | ask q k ih =>
    intro Φ hΦ
    have e : {ω | Φ ω ((Query.ask q k).eval (r ω))}
        = ⋃ a, {ω | r ω q = a} ∩ {ω | Φ ω ((k a).eval (r ω))} := by
      ext ω
      simp only [Query.eval_ask, Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_inter_iff]
      exact ⟨fun h => ⟨_, rfl, h⟩, by rintro ⟨a, rfl, h⟩; exact h⟩
    rw [e]
    exact MeasurableSet.iUnion fun a => (hr q a).inter (ih a Φ hΦ)

private theorem measurableSet_of_sections {X Y : Type*} [MeasurableSpace X] [MeasurableSpace Y]
    [Countable Y] [MeasurableSingletonClass Y] (S : Set (X × Y))
    (h : ∀ y, MeasurableSet {x | (x, y) ∈ S}) : MeasurableSet S := by
  have e : S = ⋃ y, {x | (x, y) ∈ S} ×ˢ {y} := by
    ext ⟨x, y⟩
    simp
  rw [e]
  exact MeasurableSet.iUnion fun y => (h y).prod (measurableSet_singleton y)

namespace Random

/-! ## The target looping on pads -/

omit [Fintype α] in
theorem padded_evalFrom {σ : Type*} (M : DFA α σ) (l : List (Option α)) :
    ∀ q, M.padded.evalFrom q l = M.evalFrom q l.reduceOption := by
  induction l with
  | nil => intro q; rfl
  | cons c l ih =>
    intro q
    cases c with
    | none => rw [List.reduceOption_cons_of_none, DFA.evalFrom_cons]; exact ih q
    | some a =>
      rw [List.reduceOption_cons_of_some, DFA.evalFrom_cons, DFA.evalFrom_cons]
      exact ih _

omit [Fintype α] in
theorem padded_eval {σ : Type*} (M : DFA α σ) (z : FreeMonoid (Option α)) :
    M.padded.eval z.toList = M.eval (unpad z).toList := by
  change M.padded.evalFrom M.start z.toList = M.evalFrom M.start _
  rw [padded_evalFrom]
  rfl

/-! ## Lists of one string -/

section One

variable [DecidableEq α]

omit [DecidableEq α] in
theorem length_blockUpdate_ge (K : ℕ) (X : Finset (FreeMonoid α)) :
    ∀ (ws ℓ : List (FreeMonoid α)), ℓ.length ≤ (blockUpdate K X ℓ ws).1.length
  | [], _ => le_rfl
  | w :: ws, ℓ => by
    by_cases hw : w ∈ ℓ
    · simp only [blockUpdate, if_pos hw]
      exact length_blockUpdate_ge K X ws ℓ
    · simp only [blockUpdate, if_neg hw]
      have := length_blockUpdate_ge K X ws (ℓ ++ [w])
      rw [List.length_append] at this
      omega

omit [DecidableEq α] in
/-- A block fills the list of one string, or grows nothing from an empty list and retracts with
`X` unchanged. -/
theorem topUpQ_one (read : FreeMonoid α → ARU) (K : ℕ) (H : HypothesisDFA (Option α))
    (s : Level3Site) {b : ℕ} (Bs : List (Fin b → FreeMonoid α)) (X : Finset (FreeMonoid α))
    (ℓ : List (FreeMonoid α)) :
    (topUpQ K 1 H s Bs X ℓ).eval read = none
      ∨ (∃ ℓ' Bs', (topUpQ K 1 H s Bs X ℓ).eval read = some (.full ℓ', Bs'))
      ∨ ∃ Bs', (topUpQ K 1 H s Bs X ℓ).eval read = some (.retract [] X, Bs') := by
  rcases Bs with _ | ⟨B, Bs⟩
  · exact Or.inl rfl
  right
  rw [topUpQ, Query.eval_bind]
  generalize (blockQ K H s X (List.ofFn B)).eval read = ws
  by_cases h1 : 1 ≤ (blockUpdate K X ℓ ws).1.length
  · left
    exact ⟨(blockUpdate K X ℓ ws).1, Bs, by simp [if_pos h1]⟩
  · right
    have hℓ := length_blockUpdate_ge K X ws ℓ
    obtain rfl : ℓ = [] := List.eq_nil_of_length_eq_zero (by omega)
    obtain rfl : ws = [] := by
      rcases ws with _ | ⟨w, ws⟩
      · rfl
      · exfalso
        apply h1
        have e : blockUpdate K X [] (w :: ws) = blockUpdate K X [w] ws := by simp [blockUpdate]
        rw [e]
        simpa using length_blockUpdate_ge K X ws [w]
    exact ⟨Bs, by simp [blockUpdate]⟩

/-- With lists of one string, the epochs keep `X`, and a harvest is the harvesting epoch's. -/
theorem lazyEpochsQ_one (read : FreeMonoid α → ARU) (C : Level3Settings) (K : ℕ) {J b : ℕ} :
    ∀ (ds : List (EpochDraws α C.probeBudget J b)) (e pk : ℕ) (X : Finset (FreeMonoid α))
      (ls : LazyLists α),
      ((lazyEpochsQ C K 1 e pk ds X ls).eval read).expanded = X
      ∧ ∀ s ℓ, ((lazyEpochsQ C K 1 e pk ds X ls).eval read).ending = .harvest s ℓ →
        ∃ i, ∃ hi : i < ds.length,
          ((epochQ (epochSettings C (e + i)) K X (ds[i]'hi).1).eval read).ending = .harvest s
          ∧ ((epochQ (epochSettings C (e + i)) K X (ds[i]'hi).1).eval read).hypothesis
            = ((lazyEpochsQ C K 1 e pk ds X ls).eval read).hypothesis
  | [], _, _, _, _ => ⟨rfl, fun _ _ h => by simp [lazyEpochsQ] at h⟩
  | d :: ds, e, pk, X, ls => by
    have ih := lazyEpochsQ_one read C K ds (e + 1)
    have h0 : ∀ res, (epochQ (epochSettings C e) K X d.1).eval read = res →
        (epochQ (epochSettings C (e + 0)) K X ((d :: ds)[0]'(by simp)).1).eval read = res :=
      fun _ h => h
    simp only [lazyEpochsQ, Query.eval_bind]
    generalize hres : (epochQ (epochSettings C e) K X d.1).eval read = res
    have h0 := h0 res hres
    rcases res with ⟨H, _ | s | _⟩
    · simp
    · simp only [Query.eval_bind]
      rcases topUpQ_one read K H s (List.ofFn d.2) X (ls s) with h | ⟨ℓ, Bs, h⟩ | ⟨Bs, h⟩
      · rw [h]
        simp
      · rw [h]
        refine ⟨rfl, fun s' ℓ' he => ?_⟩
        simp at he
        obtain ⟨rfl, -⟩ := he
        exact ⟨0, by simp, by rw [h0], by rw [h0]; rfl⟩
      · rw [h]
        dsimp only
        refine ⟨(ih _ X _).1, fun s' ℓ' he => ?_⟩
        obtain ⟨i, hi, hs, hH⟩ := (ih _ X _).2 s' ℓ' he
        refine ⟨i + 1, by simpa using hi, ?_⟩
        simp only [List.getElem_cons_succ]
        rw [show e + (i + 1) = e + 1 + i by omega]
        exact ⟨hs, hH⟩
    · simp

/-- With lists of one string, the round keeps its trie, and a harvest is the harvesting
epoch's. -/
theorem lazyRoundQ_one (read : FreeMonoid α → ARU) (C : Level3Settings) (K : ℕ) {E J b : ℕ}
    (X₀ : Finset (FreeMonoid α)) (draws : Fin E → EpochDraws α C.probeBudget J b) :
    ((lazyRoundQ C K 1 X₀ draws).eval read).expanded = X₀
      ∧ ∀ s ℓ, ((lazyRoundQ C K 1 X₀ draws).eval read).ending = .harvest s ℓ → ∃ e : Fin E,
        ((epochQ (epochSettings C e) K X₀ (draws e).1).eval read).ending = .harvest s
        ∧ ((epochQ (epochSettings C e) K X₀ (draws e).1).eval read).hypothesis
          = ((lazyRoundQ C K 1 X₀ draws).eval read).hypothesis := by
  unfold lazyRoundQ
  obtain ⟨h1, h2⟩ := lazyEpochsQ_one read C K (List.ofFn draws) 0 0 X₀ fun _ => []
  refine ⟨h1, fun s ℓ h => ?_⟩
  obtain ⟨i, hi, hs, hH⟩ := h2 s ℓ h
  rw [List.length_ofFn] at hi
  refine ⟨⟨i, hi⟩, ?_⟩
  simpa [List.getElem_ofFn] using And.intro hs hH

theorem eval_epochQ (read : FreeMonoid α → ARU) (C : Level3Settings) (K : ℕ)
    (X : Finset (FreeMonoid α)) {P : ℕ} (draws : Fin P → FreeMonoid α) :
    (epochQ C K X draws).eval read
      = indecisiveLStarLevel3 { C with walkStart := K } (padRead read)
          fun i => padIn K X (draws i) := by
  rw [epochQ, Query.eval_subst]
  rfl

end One

/-! ## The harvest's law -/

section Law

variable [DecidableEq α]

/-- Where `X`'s strings are shorter than `K ≤ |x|` on `D`, reads agreeing on the canonical strings
give the same harvest from probes padded from `X`, under a tree whose midfixes have only letters
of the alphabet. -/
theorem harvestLaw_padIn_congr {r r' : FreeMonoid (Option α) → ARU} {K : ℕ}
    {X : Finset (FreeMonoid α)} (h : ∀ z, padIn K X (unpad z) = z → r z = r' z)
    (hX : ∀ c ∈ X, c.toList.length < K) (D : Measure (FreeMonoid α))
    (hD : ∀ᵐ x ∂D, K ≤ x.toList.length) {H : HypothesisDFA (Option α)}
    (hT : MidsIn (fun c : Option α => c.isSome) H.tree) (s : Level3Site) :
    harvestLaw r (D.map (padIn K X)) K H s = harvestLaw r' (D.map (padIn K X)) K H s := by
  classical
  ext S _
  rw [harvestLaw_apply, harvestLaw_apply,
    lintegral_map (measurable_of_countable _) (measurable_of_countable _),
    lintegral_map (measurable_of_countable _) (measurable_of_countable _)]
  refine lintegral_congr_ae (hD.mono fun x hx => ?_)
  cases s <;> simp only [siteHarvest, probeOrAgree_padIn_congr h hX hx hT H.edges]

omit [DecidableEq α] in
theorem measurable_lazyHarvestLaw {Ω' : Type*} [MeasurableSpace Ω'] (rd : Ω' → FreeMonoid α → ARU)
    (hrd : ∀ z a, MeasurableSet {ω | rd ω z = a}) (D : Measure (FreeMonoid α)) [IsFiniteMeasure D]
    (K : ℕ) (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α)) (s : Level3Site)
    (S : Set (FreeMonoid α)) : Measurable fun ω => lazyHarvestLaw (rd ω) D K X H s S := by
  classical
  have e : (fun ω => lazyHarvestLaw (rd ω) D K X H s S) = fun ω =>
      ∫⁻ z, (((siteHarvest (padRead (rd ω)) K H s z).filter (· ∈ unpad ⁻¹' S)).length : ℝ≥0∞)
        ∂(D.map (padIn K X)) := by
    funext ω
    rw [lazyHarvestLaw, Measure.map_apply (measurable_of_countable _)
      (Set.to_countable _).measurableSet, harvestLaw_apply]
  rw [e]
  refine Measurable.lintegral_prod_right' (f := fun p : Ω' × FreeMonoid (Option α) =>
    (((siteHarvest (padRead (rd p.1)) K H s p.2).filter (· ∈ unpad ⁻¹' S)).length : ℝ≥0∞)) ?_
  refine measurable_from_prod_countable_left fun z => ?_
  intro T _
  have := measurableSet_query_eval (fun ω => padRead (rd ω)) (fun z a => hrd (unpad z) a)
    (siteHarvestQ K H s z)
    (fun _ l => (((l.filter (· ∈ unpad ⁻¹' S)).length : ℝ≥0∞) ∈ T)) fun _ => MeasurableSet.const _
  simp only [eval_siteHarvestQ] at this
  exact this

omit [DecidableEq α] in
theorem lazyHarvestLaw_real_states (rd : FreeMonoid α → ARU) (D : Measure (FreeMonoid α))
    (K : ℕ) (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α)) (s : Level3Site)
    (M : DFA α σ) (U : σ → ℝ) (t : ℝ) :
    (lazyHarvestLaw rd D K X H s).real {z | U (M.eval z.toList) ≤ t}
      = (harvestLaw (padRead rd) (D.map (padIn K X)) K H s).real
          {z | U (M.padded.eval z.toList) ≤ t} := by
  rw [lazyHarvestLaw, map_measureReal_apply (measurable_of_countable _)
    (Set.to_countable _).measurableSet]
  congr 1
  ext z
  simp [padded_eval]

omit [DecidableEq α] in
theorem lazyHarvestLaw_real_univ (rd : FreeMonoid α → ARU) (D : Measure (FreeMonoid α))
    (K : ℕ) (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α)) (s : Level3Site) :
    (lazyHarvestLaw rd D K X H s).real Set.univ
      = (harvestLaw (padRead rd) (D.map (padIn K X)) K H s).real Set.univ := by
  rw [lazyHarvestLaw, map_measureReal_apply (measurable_of_countable _) MeasurableSet.univ,
    Set.preimage_univ]

omit [DecidableEq α] in
theorem lazyHarvestLaw_univ_ne_top (rd : FreeMonoid α → ARU) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (K : ℕ) (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α))
    (s : Level3Site) : lazyHarvestLaw rd D K X H s Set.univ ≠ ⊤ := by
  classical
  have : IsProbabilityMeasure (D.map (padIn K X)) :=
    Measure.isProbabilityMeasure_map (measurable_of_countable _).aemeasurable
  rw [lazyHarvestLaw, Measure.map_apply (measurable_of_countable _) MeasurableSet.univ,
    Set.preimage_univ]
  exact harvestLaw_univ_ne_top _ _ _ _ _

end Law

/-! ## Tame harvests -/

/-- `ν` puts at most `A·T + sl` of its mass on the states read undecided at most `T` of the time,
and yields `y`. -/
def HarvestTame (M : DFA α σ) (U : σ → ℝ) (A sl y : ℝ) (ν : Measure (FreeMonoid α)) : Prop :=
  (∀ T, 0 ≤ T → ν.real {z | U (M.eval z.toList) ≤ T} ≤ (A * T + sl) * ν.real Set.univ)
    ∧ y ≤ ν.real Set.univ

/-- `ν` puts more than `A·U q + sl` of its mass on the states read undecided at most as often as
some state `q`, or yields less than `y`. -/
def StateBad (M : DFA α σ) (U : σ → ℝ) (A sl y : ℝ) (ν : Measure (FreeMonoid α)) : Prop :=
  (∃ q, 0 ≤ U q ∧ (A * U q + sl) * ν.real Set.univ < ν.real {z | U (M.eval z.toList) ≤ U q})
    ∨ ν.real Set.univ < y

/-- A law not tame at the rates `U` is bad at a state for rates `U' ≤ c U`. -/
theorem stateBad_of_not_tame [Fintype σ] {M : DFA α σ} {U U' : σ → ℝ} {c A A' sl sl' y : ℝ}
    {ν : Measure (FreeMonoid α)} (hν : ν Set.univ ≠ ⊤) (hc : 0 ≤ c) (hA' : 0 ≤ A')
    (hA : c * A' ≤ A) (hsl' : 0 ≤ sl') (hsl : sl' ≤ sl) (hU'0 : ∀ q, 0 ≤ U' q)
    (hU' : ∀ q, U' q ≤ c * U q) (h : ¬ HarvestTame M U A sl y ν) : StateBad M U' A' sl' y ν := by
  by_contra hb
  simp only [StateBad, not_or, not_exists, not_and, not_lt] at hb
  obtain ⟨hq, hy⟩ := hb
  have hT := sublevel_of_states M U' ν hA' hsl' (fun z => hU'0 _) hq
  refine h ⟨fun T hT0 => ?_, hy⟩
  have hm : 0 ≤ ν.real Set.univ := measureReal_nonneg
  have hsub : {z : FreeMonoid α | U (M.eval z.toList) ≤ T}
      ⊆ {z | U' (M.eval z.toList) ≤ c * T} := fun z hz =>
    (hU' _).trans (mul_le_mul_of_nonneg_left hz hc)
  have h1 := measureReal_mono hsub (measure_ne_top_of_subset (Set.subset_univ _) hν)
  have h2 := hT (c * T) (mul_nonneg hc hT0)
  have h3 : A' * (c * T) + sl' ≤ A * T + sl := by nlinarith
  nlinarith [mul_le_mul_of_nonneg_right h3 hm]

theorem measurableSet_stateBad [Fintype σ] [DecidableEq α] {Ω' : Type*} [MeasurableSpace Ω']
    (rd : Ω' → FreeMonoid α → ARU) (hrd : ∀ z a, MeasurableSet {ω | rd ω z = a})
    (D : Measure (FreeMonoid α)) [IsFiniteMeasure D] (K : ℕ) (X : Finset (FreeMonoid α))
    (H : HypothesisDFA (Option α)) (s : Level3Site) (M : DFA α σ) (U : σ → ℝ) (A sl y : ℝ) :
    MeasurableSet {ω | StateBad M U A sl y (lazyHarvestLaw (rd ω) D K X H s)} := by
  have hm : ∀ S, Measurable fun ω => (lazyHarvestLaw (rd ω) D K X H s).real S := fun S =>
    ENNReal.measurable_toReal.comp (measurable_lazyHarvestLaw rd hrd D K X H s S)
  simp only [StateBad, Set.ofPred_or, Set.ofPred_exists, Set.ofPred_and]
  exact (MeasurableSet.iUnion fun q => (MeasurableSet.const _).inter
    (measurableSet_lt ((hm _).const_mul _) (hm _))).union (measurableSet_lt (hm _) measurable_const)

end Random

/-! ## The clean read -/

namespace SuffixFamily

variable [IsProbabilityMeasure μ] (fam : SuffixFamily μ α σ)

/-- A string's read on its state's rarer side. -/
def wrongRead (z : FreeMonoid α) : ARU :=
  if readSide (fam.readDist (fam.M.eval z.toList)) then .reject else .accept

/-- The family's read, with a read on its state's rarer side made undecided. -/
def cleanRead (ω : Ω) (z : FreeMonoid α) : ARU :=
  if fam.read ω z = fam.wrongRead z then .undecided else fam.read ω z

/-- A state's chance of a clean undecided read: undecided, or on its rarer side. -/
def cleanUndecided (q : σ) : ℝ :=
  fam.readDist q .undecided
    + fam.readDist q (if readSide (fam.readDist q) then .reject else .accept)

/-- The family's noise, and a fresh read of every string over the alphabet with a pad, drawn
with the clean law of the string's pads dropped. -/
def padNoise : Measure (Ω × (FreeMonoid (Option α) → ARU)) :=
  μ.prod (Measure.infinitePi fun z => μ.map fun ω => fam.cleanRead ω (Random.unpad z))

open scoped Classical in
/-- A canonical string reads cleanly as its pads dropped; any other string reads its fresh draw. -/
def padCleanRead (K : ℕ) (X : Finset (FreeMonoid α)) (z : FreeMonoid (Option α))
    (v : Ω × (FreeMonoid (Option α) → ARU)) : ARU :=
  if Random.padIn K X (Random.unpad z) = z then fam.cleanRead v.1 (Random.unpad z) else v.2 z

omit [IsProbabilityMeasure μ] in
theorem wrongRead_ne_undecided (z : FreeMonoid α) : fam.wrongRead z ≠ .undecided := by
  unfold wrongRead
  split_ifs <;> simp

omit [IsProbabilityMeasure μ] in
theorem cleanRead_ne_wrong (ω : Ω) (z : FreeMonoid α) : fam.cleanRead ω z ≠ fam.wrongRead z := by
  unfold cleanRead
  split_ifs with h
  · exact (fam.wrongRead_ne_undecided z).symm
  · exact h

omit [IsProbabilityMeasure μ] in
theorem cleanRead_eq {ω : Ω} {z : FreeMonoid α} (h : fam.read ω z ≠ fam.wrongRead z) :
    fam.cleanRead ω z = fam.read ω z := by
  simp [cleanRead, h]

omit [IsProbabilityMeasure μ] in
theorem readsNoWrong_cleanRead (ω : Ω) :
    ReadsNoWrong fam.M (fun q => readSide (fam.readDist q)) (fam.cleanRead ω) :=
  fun z => fam.cleanRead_ne_wrong ω z

theorem measurable_cleanRead (z : FreeMonoid α) : Measurable fun ω => fam.cleanRead ω z :=
  (measurable_from_top
    (f := fun r : ARU => if r = fam.wrongRead z then ARU.undecided else r)).comp
      (fam.read_measurable z)

theorem cleanRead_indep : iIndepFun (fun z ω => fam.cleanRead ω z) μ := by
  classical
  exact fam.read_indep.comp (fun z (r : ARU) => if r = fam.wrongRead z then ARU.undecided else r)
    fun _ => measurable_from_top

theorem real_cleanRead_undecided (z : FreeMonoid α) :
    μ.real {ω | fam.cleanRead ω z = .undecided} = fam.cleanUndecided (fam.M.eval z.toList) := by
  classical
  have e : {ω | fam.cleanRead ω z = .undecided}
      = {ω | fam.read ω z = .undecided} ∪ {ω | fam.read ω z = fam.wrongRead z} := by
    ext ω
    simp only [cleanRead, Set.mem_ofPred_eq, Set.mem_union]
    by_cases h : fam.read ω z = fam.wrongRead z
    · simp [h]
    · simp [h]
  rw [e, measureReal_union (Set.disjoint_left.2 fun ω h1 h2 =>
      fam.wrongRead_ne_undecided z ((Set.mem_ofPred_eq ▸ h2).symm.trans h1))
      (show MeasurableSet {ω | fam.read ω z = fam.wrongRead z} from
        fam.read_measurable z (measurableSet_singleton _)),
    fam.reads_by_state, fam.reads_by_state]
  rfl

theorem cleanUndecided_nonneg (q : σ) : 0 ≤ fam.cleanUndecided q := by
  classical
  exact add_nonneg (fam.readDist_nonneg q _) (fam.readDist_nonneg q _)

theorem cleanUndecided_le (q : σ) :
    fam.cleanUndecided q ≤ (1 + fam.κ) * fam.readDist q .undecided := by
  classical
  have := fam.readDist_wrong q
  unfold cleanUndecided
  linarith

instance (z : FreeMonoid (Option α)) :
    IsProbabilityMeasure (μ.map fun ω => fam.cleanRead ω (Random.unpad z)) :=
  Measure.isProbabilityMeasure_map (fam.measurable_cleanRead _).aemeasurable

instance : IsProbabilityMeasure fam.padNoise := by
  unfold padNoise
  infer_instance

/-- A string's read has the clean law of its pads dropped. -/
theorem padNoise_read (K : ℕ) (X : Finset (FreeMonoid α)) (z : FreeMonoid (Option α))
    (A : Set ARU) :
    fam.padNoise (fam.padCleanRead K X z ⁻¹' A)
      = μ ((fun ω => fam.cleanRead ω (Random.unpad z)) ⁻¹' A) := by
  classical
  unfold padNoise
  by_cases hz : Random.padIn K X (Random.unpad z) = z
  · have e : fam.padCleanRead K X z ⁻¹' A
        = ((fun ω => fam.cleanRead ω (Random.unpad z)) ⁻¹' A) ×ˢ Set.univ := by
      ext v
      simp [padCleanRead, hz]
    rw [e, Measure.prod_prod, measure_univ, mul_one]
  · have e : fam.padCleanRead K X z ⁻¹' A = Set.univ ×ˢ ((fun ξ => ξ z) ⁻¹' A) := by
      ext v
      simp [padCleanRead, hz]
    rw [e, Measure.prod_prod, measure_univ, one_mul,
      ← Measure.map_apply (measurable_pi_apply z) MeasurableSpace.measurableSet_top,
      Measure.infinitePi_map_eval,
      Measure.map_apply (fam.measurable_cleanRead _) MeasurableSpace.measurableSet_top]

theorem padCleanRead_indep (K : ℕ) (X : Finset (FreeMonoid α)) :
    iIndepFun (fam.padCleanRead K X) fam.padNoise := by
  classical
  rw [iIndepFun_iff_measure_inter_preimage_eq_mul]
  intro S A _
  set Sc := S.filter fun z => Random.padIn K X (Random.unpad z) = z with hSc
  set Sn := S.filter fun z => ¬ Random.padIn K X (Random.unpad z) = z with hSn
  have hinter : (⋂ z ∈ S, fam.padCleanRead K X z ⁻¹' A z)
      = (⋂ w ∈ Sc.image Random.unpad,
          (fun ω => fam.cleanRead ω w) ⁻¹' A (Random.padIn K X w)) ×ˢ Set.pi (↑Sn) A := by
    ext ⟨ω, ξ⟩
    simp only [Set.mem_iInter, Set.mem_preimage, Set.mem_prod, Set.mem_pi, Finset.mem_coe,
      Finset.mem_image, Finset.mem_filter, hSc, hSn, padCleanRead]
    constructor
    · intro h
      refine ⟨?_, fun z ⟨hzS, hz⟩ => by simpa [hz] using h z hzS⟩
      rintro w ⟨z, ⟨hzS, hz⟩, rfl⟩
      have := h z hzS
      rw [if_pos hz] at this
      rwa [hz]
    · rintro ⟨h1, h2⟩ z hzS
      by_cases hz : Random.padIn K X (Random.unpad z) = z
      · rw [if_pos hz]
        have := h1 (Random.unpad z) ⟨z, ⟨hzS, hz⟩, rfl⟩
        rwa [hz] at this
      · rw [if_neg hz]
        exact h2 z ⟨hzS, hz⟩
  have hinj : Set.InjOn Random.unpad (Sc : Set (FreeMonoid (Option α))) := fun z hz z' hz' h => by
    rw [← (Finset.mem_filter.1 hz).2, ← (Finset.mem_filter.1 hz').2, h]
  simp_rw [fam.padNoise_read K X]
  rw [hinter, padNoise, Measure.prod_prod,
    Measure.infinitePi_pi _ (fun _ _ => MeasurableSpace.measurableSet_top),
    iIndepFun_iff_measure_inter_preimage_eq_mul.1 fam.cleanRead_indep _
      (fun _ _ => MeasurableSpace.measurableSet_top),
    Finset.prod_image hinj,
    ← Finset.prod_filter_mul_prod_filter_not S fun z => Random.padIn K X (Random.unpad z) = z]
  congr 1
  · refine Finset.prod_congr rfl fun z hz => ?_
    rw [(Finset.mem_filter.1 hz).2]
  · refine Finset.prod_congr rfl fun z _ => ?_
    rw [Measure.map_apply (fam.measurable_cleanRead _) MeasurableSpace.measurableSet_top]

/-- The clean read over the alphabet with a pad: never wrong, each state undecided with its clean
chance. -/
def padOracle (K : ℕ) (X : Finset (FreeMonoid α)) :
    Random.ReadOracle fam.padNoise fam.M.padded (fun q => readSide (fam.readDist q))
      fam.cleanUndecided where
  read := fam.padCleanRead K X
  measurable z := by
    classical
    unfold padCleanRead
    split_ifs
    · exact (fam.measurable_cleanRead _).comp measurable_fst
    · exact (measurable_pi_apply z).comp measurable_snd
  indep := fam.padCleanRead_indep K X
  undecided z := by
    rw [measureReal_def, show {v | fam.padCleanRead K X z v = .undecided}
        = fam.padCleanRead K X z ⁻¹' {.undecided} from rfl, fam.padNoise_read,
      ← measureReal_def, Random.padded_eval]
    exact fam.real_cleanRead_undecided _
  wrong z := by
    rw [show {v | fam.padCleanRead K X z v
          = if readSide (fam.readDist (fam.M.padded.eval z.toList)) then .reject else .accept}
        = fam.padCleanRead K X z ⁻¹' {fam.wrongRead (Random.unpad z)} by
          ext v
          simp [wrongRead, Random.padded_eval],
      fam.padNoise_read]
    convert measure_empty (μ := μ)
    ext ω
    simpa using fam.cleanRead_ne_wrong ω (Random.unpad z)

/-! ## An epoch, and the round, under the clean read -/

section Epoch

variable [Fintype σ] [DecidableEq α]

/-- The clean epoch on the probes harvests a law bad at a state of the clean rates. -/
def epochBad (C' : Random.Level3Settings) (K : ℕ) (X₀ : Finset (FreeMonoid α))
    (D' : Measure (FreeMonoid α)) (A sl : ℝ) {P : ℕ} : Set (Ω × (Fin P → FreeMonoid α)) :=
  {p | ∃ s, ((Random.epochQ C' K X₀ p.2).eval (fam.cleanRead p.1)).ending = .harvest s
    ∧ Random.StateBad fam.M fam.cleanUndecided A sl C'.harvestRate
      (Random.lazyHarvestLaw (fam.cleanRead p.1) D' K X₀
        ((Random.epochQ C' K X₀ p.2).eval (fam.cleanRead p.1)).hypothesis s)}

theorem measurableSet_epochBad (C' : Random.Level3Settings) (K : ℕ) (X₀ : Finset (FreeMonoid α))
    (D' : Measure (FreeMonoid α)) [IsFiniteMeasure D'] (A sl : ℝ) {P : ℕ} :
    MeasurableSet (fam.epochBad C' K X₀ D' A sl (P := P)) := by
  have hr : ∀ z a, MeasurableSet {ω | fam.cleanRead ω z = a} := fun z a =>
    fam.measurable_cleanRead z (measurableSet_singleton a)
  refine measurableSet_of_sections _ fun x => ?_
  refine measurableSet_query_eval fam.cleanRead hr (Random.epochQ C' K X₀ x)
    (fun ω res => ∃ s, res.ending = .harvest s ∧ Random.StateBad fam.M fam.cleanUndecided A sl
      C'.harvestRate (Random.lazyHarvestLaw (fam.cleanRead ω) D' K X₀ res.hypothesis s))
    fun res => ?_
  rcases res with ⟨H, _ | s | _⟩
  · simp
  · simpa using Random.measurableSet_stateBad fam.cleanRead hr D' K X₀ H s fam.M
      fam.cleanUndecided A sl C'.harvestRate
  · simp

/-- `Level3RoundCorrect` on the clean read over the alphabet with a pad: the clean epoch on probes
padded from `X₀` harvests a law bad at a state of the clean rates with chance at most `δ`. -/
theorem epochBad_le (C' : Random.Level3Settings) (X₀ : Finset (FreeMonoid α))
    (hX₀ : ∀ c ∈ X₀, c.toList.length < C'.walkStart) (D' : Measure (FreeMonoid α))
    [IsProbabilityMeasure D'] {θ p₁ : ℝ} (Lp : ℕ)
    (hlen : ∀ᵐ x ∂D', C'.walkStart ≤ x.toList.length ∧ x.toList.length ≤ Lp)
    (hp₁ : ∀ u, D'.real {x | prefixOf x C'.walkStart = u} ≤ p₁)
    (hlight : ∀ c : FreeMonoid α, c.toList.length < C'.walkStart → c ∉ X₀ →
      D'.real {x | prefixOf x c.toList.length = c} ≤ θ)
    (hδ : 0 < C'.δ) (hy : 0 < C'.harvestRate) (hε1 : C'.disagreementRate < 1) :
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * (nα + 1) + 1) + 2 * (nα + 1) + 1
    let Kt : ℕ := (1 + (nQ + 1) ^ 3 * (nα + 1)) ^ nQ * (nQ + 3) ^ ((nQ + 2) * (nα + 1))
    36 * (S : ℝ) / C'.disagreementRate * Real.log (16 * S * (S + 1) / C'.δ) + S
        ≤ C'.probeBudget →
    (μ.prod (Measure.pi fun _ : Fin C'.probeBudget => D'))
        (fam.epochBad C' C'.walkStart X₀ D'
          ((nQ + 1) * (6 / C'.harvestRate + 16 * (2 + Nat.clog 2 (Lp + C'.walkStart))))
          (max θ p₁ * Real.log (8 * (nQ + 1) * Kt / C'.δ) / C'.harvestRate))
      ≤ ENNReal.ofReal C'.δ := by
  intro nQ nα S Kt hP
  have hπ : Measurable (Random.padIn C'.walkStart X₀) := measurable_of_countable _
  have hlenP : ∀ᵐ z ∂(D'.map (Random.padIn C'.walkStart X₀)),
      z.toList.length ≤ Lp + C'.walkStart := by
    rw [ae_map_iff hπ.aemeasurable (Set.to_countable _).measurableSet]
    refine hlen.mono fun x hx => ?_
    rw [Random.length_padIn]
    omega
  let P : PrefixDistribution (Option α) :=
    ⟨D'.map (Random.padIn C'.walkStart X₀), Measure.isProbabilityMeasure_map hπ.aemeasurable,
      Lp + C'.walkStart, hlenP⟩
  have key := Random.level3RoundCorrect_holds fam.M.padded (fun q => readSide (fam.readDist q))
    fam.cleanUndecided fam.padNoise (fam.padOracle C'.walkStart X₀) P C' (max θ p₁)
  dsimp only at key
  simp only [Fintype.card_option] at key
  have h := key (Random.real_padCell_le D' hX₀ (hlen.mono fun x hx => hx.1) hlight hp₁) hδ hy
    le_rfl hε1 hP
  have hB := fam.measurableSet_epochBad C' C'.walkStart X₀ D'
    ((nQ + 1) * (6 / C'.harvestRate + 16 * (2 + Nat.clog 2 (Lp + C'.walkStart))))
    (max θ p₁ * Real.log (8 * (nQ + 1) * Kt / C'.δ) / C'.harvestRate) (P := C'.probeBudget)
  have hf : MeasurePreserving
      (Prod.map Prod.fst id : (Ω × (FreeMonoid (Option α) → ARU))
        × (Fin C'.probeBudget → FreeMonoid α) → Ω × (Fin C'.probeBudget → FreeMonoid α))
      (fam.padNoise.prod (Measure.pi fun _ : Fin C'.probeBudget => D'))
      (μ.prod (Measure.pi fun _ : Fin C'.probeBudget => D')) :=
    (measurePreserving_fst (μ := μ)).prod (MeasurePreserving.id _)
  rw [← hf.measure_preimage hB.nullMeasurableSet, Measure.prod_apply (hf.measurable hB)]
  refine (lintegral_mono fun v => ?_).trans h
  obtain ⟨ω, ξ⟩ := v
  have hG : ∀ᵐ x ∂(Measure.pi fun _ : Fin C'.probeBudget => D'),
      ∀ i, C'.walkStart ≤ (x i).toList.length :=
    Filter.eventually_all.2 fun i =>
      (Measure.tendsto_eval_ae_ae (μ := fun _ : Fin C'.probeBudget => D') (i := i)).eventually
        (p := fun y : FreeMonoid α => C'.walkStart ≤ y.toList.length) (hlen.mono fun x hx => hx.1)
  have hF : Measurable fun (x : Fin C'.probeBudget → FreeMonoid α) i =>
      Random.padIn C'.walkStart X₀ (x i) :=
    measurable_pi_lambda _ fun i => hπ.comp (measurable_pi_apply i)
  refine (measure_mono_ae (hG.mono fun x hx hmem => ?_)).trans
    ((Measure.le_map_apply hF.aemeasurable _).trans_eq
      (by rw [Measure.pi_map_pi fun _ => hπ.aemeasurable]))
  obtain ⟨s, hend, hbad⟩ := hmem
  have hend' : ((Random.epochQ C' C'.walkStart X₀ x).eval (fam.cleanRead ω)).ending
      = .harvest s := hend
  have hbad' : Random.StateBad fam.M fam.cleanUndecided
      ((nQ + 1) * (6 / C'.harvestRate + 16 * (2 + Nat.clog 2 (Lp + C'.walkStart))))
      (max θ p₁ * Real.log (8 * (nQ + 1) * Kt / C'.δ) / C'.harvestRate) C'.harvestRate
      (Random.lazyHarvestLaw (fam.cleanRead ω) D' C'.walkStart X₀
        ((Random.epochQ C' C'.walkStart X₀ x).eval (fam.cleanRead ω)).hypothesis s) := hbad
  have hcanon : ∀ z, Random.padIn C'.walkStart X₀ (Random.unpad z) = z →
      (fam.padOracle C'.walkStart X₀).read z (ω, ξ) = Random.padRead (fam.cleanRead ω) z :=
    fun z hz => by
      show fam.padCleanRead C'.walkStart X₀ z (ω, ξ) = _
      simp only [padCleanRead, hz, if_true]
      rfl
  obtain ⟨hrun, hmids⟩ := Random.level3_padIn_congr hcanon hX₀ C' rfl x hx
  have hres : Random.indecisiveLStarLevel3 C'
      (fun z => (fam.padOracle C'.walkStart X₀).read z (ω, ξ))
      (fun i => Random.padIn C'.walkStart X₀ (x i))
      = (Random.epochQ C' C'.walkStart X₀ x).eval (fam.cleanRead ω) :=
    hrun.trans (Random.eval_epochQ _ C' C'.walkStart X₀ x).symm
  rw [hres] at hmids
  have hlaw := Random.harvestLaw_padIn_congr hcanon hX₀ D' (hlen.mono fun x hx => hx.1) hmids s
  show ¬ EndsWell _
  simp only [hres, EndsWell, hend']
  rintro ⟨hT, -, hY⟩
  rcases hbad' with ⟨q, hq0, hq⟩ | hy'
  · have := hT _ hq0
    rw [Random.lazyHarvestLaw_real_states, Random.lazyHarvestLaw_real_univ, ← hlaw] at hq
    exact absurd this (not_le.2 hq)
  · rw [Random.lazyHarvestLaw_real_univ, ← hlaw] at hy'
    exact absurd hY (not_le.2 hy')

/-- The clean round, with lists of one string, harvests a law not tame at the family's rates. -/
def cleanHarvestBad (C : Random.Level3Settings) (nh : ℕ) {E J b : ℕ} (X₀ : Finset (FreeMonoid α))
    (D' : Measure (FreeMonoid α)) (A sl : ℝ) :
    Set (Ω × (Fin E → Random.EpochDraws α C.probeBudget J b)) :=
  {d | ∃ s ℓ, ((Random.lazyRoundQ C C.walkStart nh X₀ d.2).eval (fam.cleanRead d.1)).ending
      = .harvest s ℓ
    ∧ ¬ Random.HarvestTame fam.M (fun q => fam.readDist q .undecided) A sl C.harvestRate
      (Random.lazyHarvestLaw (fam.cleanRead d.1) D' C.walkStart X₀
        ((Random.lazyRoundQ C C.walkStart nh X₀ d.2).eval (fam.cleanRead d.1)).hypothesis s)}

omit [Fintype σ] [DecidableEq α] [IsProbabilityMeasure μ] in
theorem sum_epochLevels_le (δ : ℝ) (hδ : 0 ≤ δ) (E : ℕ) :
    ∑ e : Fin E, δ / (((e : ℕ) + 1) * ((e : ℕ) + 2) : ℝ) ≤ δ := by
  have key : ∀ n : ℕ, ∑ e ∈ Finset.range n, δ / (((e : ℝ) + 1) * ((e : ℝ) + 2))
      = δ - δ / (n + 1) := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      rw [Finset.sum_range_succ, ih]
      push_cast
      field_simp
      ring
  rw [Fin.sum_univ_eq_sum_range (fun e : ℕ => δ / (((e : ℝ) + 1) * ((e : ℝ) + 2))) E, key]
  have : 0 ≤ δ / ((E : ℝ) + 1) := by positivity
  linarith

/-- With lists of one string, the clean round harvests a law not tame at slope `(1 + κ) A₃` and
slack `η / y`, at the least epoch level, with chance at most `δ`, the epochs' levels summed. -/
theorem cleanHarvest_le (hκ : 0 ≤ fam.κ) (C : Random.Level3Settings) {E J b : ℕ} (nh : ℕ)
    (hnh : nh = 1) (X₀ : Finset (FreeMonoid α)) (hX₀ : ∀ c ∈ X₀, c.toList.length < C.walkStart)
    (D' : Measure (FreeMonoid α)) [IsProbabilityMeasure D'] {θ p₁ : ℝ} (Lp : ℕ)
    (hlen : ∀ᵐ x ∂D', C.walkStart ≤ x.toList.length ∧ x.toList.length ≤ Lp)
    (hp₁ : ∀ u, D'.real {x | prefixOf x C.walkStart = u} ≤ p₁)
    (hlight : ∀ c : FreeMonoid α, c.toList.length < C.walkStart → c ∉ X₀ →
      D'.real {x | prefixOf x c.toList.length = c} ≤ θ)
    (hδ : 0 < C.δ) (hδ1 : C.δ < 1) (hy : 0 < C.harvestRate) (hε1 : C.disagreementRate < 1) :
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * (nα + 1) + 1) + 2 * (nα + 1) + 1
    let Kt : ℕ := (1 + (nQ + 1) ^ 3 * (nα + 1)) ^ nQ * (nQ + 3) ^ ((nQ + 2) * (nα + 1))
    36 * (S : ℝ) / C.disagreementRate * Real.log (16 * S * (S + 1) * (E * (E + 1)) / C.δ) + S
        ≤ C.probeBudget →
    (μ.prod (Measure.pi fun _ : Fin E => (Measure.pi fun _ : Fin C.probeBudget => D').prod
        (Measure.pi fun _ : Fin J => Measure.pi fun _ : Fin b => D')))
        (fam.cleanHarvestBad C nh X₀ D'
          ((1 + fam.κ) * (nQ + 1) * (6 / C.harvestRate + 16 * (2 + Nat.clog 2 (Lp + C.walkStart))))
          (max θ p₁ * Real.log (8 * (nQ + 1) * Kt * (E * (E + 1)) / C.δ) / C.harvestRate))
      ≤ ENNReal.ofReal C.δ := by
  intro nQ nα S Kt hP
  subst hnh
  have hp₀ : 0 ≤ max θ p₁ := le_max_of_le_right (measureReal_nonneg.trans (hp₁ 1))
  have hA₃ : (0 : ℝ) ≤ (nQ + 1) * (6 / C.harvestRate + 16 * (2 + Nat.clog 2 (Lp + C.walkStart))) :=
    by positivity
  have hKt : (1 : ℝ) ≤ Kt := by exact_mod_cast Nat.one_le_iff_ne_zero.2 (by positivity)
  have hS0 : (0 : ℝ) ≤ 36 * S / C.disagreementRate := by
    have : 0 < C.disagreementRate := by
      have := hy
      unfold Random.Level3Settings.harvestRate at this
      linarith
    positivity
  have hlev : ∀ e : Fin E, (((e : ℕ) : ℝ) + 1) * ((e : ℕ) + 2) ≤ E * (E + 1) := fun e => by
    have : ((e : ℕ) : ℝ) + 1 ≤ E := by exact_mod_cast e.isLt
    nlinarith
  have hpos : ∀ e : Fin E, (0 : ℝ) < (((e : ℕ) : ℝ) + 1) * ((e : ℕ) + 2) := fun e => by positivity
  have hsle : ∀ e : Fin E,
      max θ p₁ * Real.log (8 * (nQ + 1) * Kt / (C.δ / ((((e : ℕ) : ℝ) + 1) * ((e : ℕ) + 2))))
          / C.harvestRate
        ≤ max θ p₁ * Real.log (8 * (nQ + 1) * Kt * (E * (E + 1)) / C.δ) / C.harvestRate := by
    intro e
    have hle := hlev e
    refine div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left
      (Real.log_le_log (by have := hpos e; positivity) ?_) hp₀) hy.le
    rw [div_div_eq_mul_div]
    gcongr
  have hsle0 : ∀ e : Fin E, 0 ≤ max θ p₁
      * Real.log (8 * (nQ + 1) * Kt / (C.δ / ((((e : ℕ) : ℝ) + 1) * ((e : ℕ) + 2))))
      / C.harvestRate := by
    intro e
    refine div_nonneg (mul_nonneg hp₀ (Real.log_nonneg ?_)) hy.le
    have he : C.δ / ((((e : ℕ) : ℝ) + 1) * ((e : ℕ) + 2)) ≤ C.δ :=
      div_le_self hδ.le (by nlinarith [hpos e])
    rw [le_div_iff₀ (div_pos hδ (hpos e))]
    nlinarith
  have hsub : fam.cleanHarvestBad C 1 X₀ D'
      ((1 + fam.κ) * (nQ + 1) * (6 / C.harvestRate + 16 * (2 + Nat.clog 2 (Lp + C.walkStart))))
      (max θ p₁ * Real.log (8 * (nQ + 1) * Kt * (E * (E + 1)) / C.δ) / C.harvestRate)
      ⊆ ⋃ e : Fin E, Prod.map id (fun dr : Fin E → Random.EpochDraws α C.probeBudget J b =>
          (dr e).1) ⁻¹'
          fam.epochBad (P := C.probeBudget) (Random.epochSettings C e) C.walkStart X₀ D'
            ((nQ + 1) * (6 / C.harvestRate + 16 * (2 + Nat.clog 2 (Lp + C.walkStart))))
            (max θ p₁
              * Real.log (8 * (nQ + 1) * Kt / (C.δ / ((((e : ℕ) : ℝ) + 1) * ((e : ℕ) + 2))))
              / C.harvestRate) := by
    rintro ⟨ω, dr⟩ ⟨s, ℓ, hend, hnt⟩
    obtain ⟨-, hep⟩ := Random.lazyRoundQ_one (fam.cleanRead ω) C C.walkStart X₀ dr
    obtain ⟨e, he1, he2⟩ := hep s ℓ hend
    refine Set.mem_iUnion.2 ⟨e, s, he1, ?_⟩
    dsimp only [Prod.map, id]
    rw [he2]
    exact Random.stateBad_of_not_tame (Random.lazyHarvestLaw_univ_ne_top _ _ _ _ _ _)
      (by linarith) hA₃ (le_of_eq (by ring)) (hsle0 e) (hsle e) fam.cleanUndecided_nonneg
      fam.cleanUndecided_le hnt
  refine (measure_mono hsub).trans ((measure_iUnion_fintype_le _ _).trans ?_)
  refine (Finset.sum_le_sum fun e _ => ?_).trans
    ((ENNReal.ofReal_sum_of_nonneg fun e _ => (div_pos hδ (hpos e)).le).symm.le.trans
      (ENNReal.ofReal_le_ofReal (sum_epochLevels_le C.δ hδ.le E)))
  have hpe : MeasurePreserving
      (Prod.map id (fun dr : Fin E → Random.EpochDraws α C.probeBudget J b => (dr e).1))
      (μ.prod (Measure.pi fun _ : Fin E => (Measure.pi fun _ : Fin C.probeBudget => D').prod
        (Measure.pi fun _ : Fin J => Measure.pi fun _ : Fin b => D')))
      (μ.prod (Measure.pi fun _ : Fin C.probeBudget => D')) :=
    (MeasurePreserving.id μ).prod (measurePreserving_fst.comp (measurePreserving_eval _ e))
  refine (hpe.measure_preimage_le _).trans ?_
  refine fam.epochBad_le (Random.epochSettings C e) X₀ hX₀ D' Lp hlen hp₁ hlight
    (div_pos hδ (hpos e)) hy hε1 ?_
  change 36 * (S : ℝ) / C.disagreementRate
      * Real.log (16 * S * (S + 1) / (C.δ / ((((e : ℕ) : ℝ) + 1) * ((e : ℕ) + 2)))) + S
    ≤ C.probeBudget
  have hlog : Real.log (16 * S * (S + 1) / (C.δ / ((((e : ℕ) : ℝ) + 1) * ((e : ℕ) + 2))))
      ≤ Real.log (16 * S * (S + 1) * (E * (E + 1)) / C.δ) := by
    have hle := hlev e
    apply Real.log_le_log (by have := hpos e; positivity)
    rw [div_div_eq_mul_div]
    gcongr
  have := mul_le_mul_of_nonneg_left hlog hS0
  linarith

end Epoch

end SuffixFamily

end OrthoDFA

end
