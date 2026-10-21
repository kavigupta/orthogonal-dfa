import OrthoDFA.Proofs.GateLoop
import OrthoDFA.Proofs.SideMeas
import OrthoDFA.Proofs.Convert
import OrthoDFA.Proofs.CoinWindow

/-!
# From the draws to the streams

`runMeasure` is the law of the draws: every population's streams, and a coin per string.  For a
fixed assignment of the coins, the sequences the round reads have a product law, the law of the
stream-level lemmas; the coins themselves are sliced off last.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory Thin
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]
variable {J : Type*} [Fintype J]

/-- The draws without the coins: every population's two streams i.i.d. of its law. -/
noncomputable abbrev baseRun (μ : Measure Ω) (D : J → Measure S) (Dsf : Measure S) :
    Measure (Run Ω S J) :=
  streamMeasure μ (fun j => iidStream (D j)) (fun j => iidStream (D j)) Dsf

lemma runMeasure_eq_prod (D : J → Measure S) (Dsf : Measure S) :
    runMeasure μ D Dsf = (baseRun μ D Dsf).prod (coins S) := rfl

lemma measurable_tableSeqOf_side (L : List S) (Hs : Finset S) :
    Measurable (fun p : (S → Bool) × ((ℕ → S) × (ℕ → S)) =>
      tableSeqOf L (onSide Hs true p.1) p.2.1 p.2.2) := by
  unfold tableSeqOf
  have hsub := measurable_sub_onSide Hs true
  have h1 : Measurable (fun p : (S → Bool) × ((ℕ → S) × (ℕ → S)) =>
      nthDrawIn (onSide Hs true p.1) (p.2.2 0) p.2.1 0) := by
    show Measurable ((fun f : ℕ → S => f 0)
      ∘ (fun q : ((S → Bool) × (ℕ → S)) × S => nthDrawIn (onSide Hs true q.1.1) q.2 q.1.2)
      ∘ (fun p : (S → Bool) × ((ℕ → S) × (ℕ → S)) => ((p.1, p.2.1), p.2.2 0)))
    exact (measurable_pi_apply 0).comp (hsub.comp ((measurable_fst.prodMk
      (measurable_fst.comp measurable_snd)).prodMk
      ((measurable_pi_apply 0).comp (measurable_snd.comp measurable_snd))))
  have h2 : Measurable (fun p : (S → Bool) × ((ℕ → S) × (ℕ → S)) =>
      L.headD (nthDrawIn (onSide Hs true p.1) (p.2.2 0) p.2.1 0)) :=
    (measurable_of_countable (fun s : S => L.headD s)).comp h1
  exact (measurable_prependStream L).comp (hsub.comp ((measurable_fst.prodMk
      (measurable_fst.comp measurable_snd)).prodMk h2))

lemma measurable_reserveSeqOf_side (Hs : Finset S) :
    Measurable (fun p : (S → Bool) × ((ℕ → S) × (ℕ → S)) =>
      reserveSeqOf (onSide Hs false p.1) p.2.1 p.2.2) := by
  unfold reserveSeqOf
  have hsub := measurable_sub_onSide Hs false
  have h1 : Measurable (fun p : (S → Bool) × ((ℕ → S) × (ℕ → S)) =>
      nthDrawIn (onSide Hs false p.1) (p.2.2 1) p.2.1 0) := by
    show Measurable ((fun f : ℕ → S => f 0)
      ∘ (fun q : ((S → Bool) × (ℕ → S)) × S => nthDrawIn (onSide Hs false q.1.1) q.2 q.1.2)
      ∘ (fun p : (S → Bool) × ((ℕ → S) × (ℕ → S)) => ((p.1, p.2.1), p.2.2 1)))
    exact (measurable_pi_apply 0).comp (hsub.comp ((measurable_fst.prodMk
      (measurable_fst.comp measurable_snd)).prodMk
      ((measurable_pi_apply 1).comp (measurable_snd.comp measurable_snd))))
  exact hsub.comp ((measurable_fst.prodMk (measurable_fst.comp measurable_snd)).prodMk h1)

open scoped Classical in
lemma measurable_streams (H : J → Finset S) (uni : J) :
    Measurable (streams (Ω := Ω) H uni) := by
  unfold streams
  refine (measurable_fst.comp measurable_fst).prodMk (Measurable.prodMk
    ((measurable_fst.comp (measurable_fst.comp (measurable_snd.comp measurable_fst))).prodMk
      (measurable_pi_lambda _ (fun j => ?_)))
    (measurable_pi_lambda _ (fun j => ?_)))
  · by_cases hj : j = uni
    · simp only [hj, if_true]
      exact (measurable_pi_apply uni).comp
        (measurable_snd.comp (measurable_fst.comp (measurable_snd.comp measurable_fst)))
    · simp only [hj, if_false]
      exact (measurable_tableSeqOf_side (H j).toList (allHarvest H)).comp
        (by fun_prop : Measurable (fun x : Draws Ω S J => (x.2, (x.1.2.1.2 j, x.1.2.2 j))))
  · by_cases hj : j = uni
    · simp only [hj, if_true]
      exact (measurable_pi_apply uni).comp
        (measurable_snd.comp (measurable_snd.comp measurable_fst))
    · simp only [hj, if_false]
      exact (measurable_reserveSeqOf_side (allHarvest H)).comp
        (by fun_prop : Measurable (fun x : Draws Ω S J => (x.2, (x.1.2.1.2 j, x.1.2.2 j))))

/-- A bound on every coin assignment's slice bounds the draws. -/
lemma runMeasure_le_of_slices (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (E : Set (Draws Ω S J)) (hE : MeasurableSet E) (b : ℝ≥0∞)
    (h : ∀ g, baseRun μ D Dsf {x | (x, g) ∈ E} ≤ b) : runMeasure μ D Dsf E ≤ b := by
  rw [runMeasure_eq_prod, Measure.prod_apply_symm hE]
  exact le_trans (lintegral_mono (fun g => h g)) (by simp)

/-- A bound on every draw's slice of the coins bounds the draws. -/
lemma runMeasure_le_of_coin_slices (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (E : Set (Draws Ω S J)) (hE : MeasurableSet E) (b : ℝ≥0∞)
    (h : ∀ x, coins S {g | (x, g) ∈ E} ≤ b) : runMeasure μ D Dsf E ≤ b := by
  rw [runMeasure_eq_prod, Measure.prod_apply hE]
  exact le_trans (lintegral_mono (fun x => h x)) (by simp)

open scoped Classical in
/-- Population `j`'s table sequence law under the coins `g`. -/
noncomputable def sideLawT (H : J → Finset S) (uni : J) (D : J → Measure S) (g : S → Bool)
    (j : J) : Measure (ℕ → S) :=
  if j = uni then iidStream (D j)
  else Measure.map (fun p : (ℕ → S) × (ℕ → S) =>
    tableSeqOf (H j).toList (onSide (allHarvest H) true g) p.1 p.2)
    ((iidStream (D j)).prod (iidStream (D j)))

open scoped Classical in
/-- Population `j`'s certification sequence law under the coins `g`. -/
noncomputable def sideLawC (H : J → Finset S) (uni : J) (D : J → Measure S) (g : S → Bool)
    (j : J) : Measure (ℕ → S) :=
  if j = uni then iidStream (D j)
  else Measure.map (fun p : (ℕ → S) × (ℕ → S) =>
    reserveSeqOf (onSide (allHarvest H) false g) p.1 p.2)
    ((iidStream (D j)).prod (iidStream (D j)))

lemma measurable_tableSeqOf_at (L : List S) (Hs : Finset S) (g : S → Bool) :
    Measurable (fun p : (ℕ → S) × (ℕ → S) => tableSeqOf L (onSide Hs true g) p.1 p.2) :=
  (measurable_tableSeqOf_side L Hs).comp (measurable_const.prodMk measurable_id)

lemma measurable_reserveSeqOf_at (Hs : Finset S) (g : S → Bool) :
    Measurable (fun p : (ℕ → S) × (ℕ → S) => reserveSeqOf (onSide Hs false g) p.1 p.2) :=
  (measurable_reserveSeqOf_side Hs).comp (measurable_const.prodMk measurable_id)

instance (H : J → Finset S) (uni : J) (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    (g : S → Bool) (j : J) : IsProbabilityMeasure (sideLawT H uni D g j) := by
  unfold sideLawT
  split_ifs
  · infer_instance
  · exact Measure.isProbabilityMeasure_map (measurable_tableSeqOf_at _ _ g).aemeasurable

instance (H : J → Finset S) (uni : J) (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    (g : S → Bool) (j : J) : IsProbabilityMeasure (sideLawC H uni D g j) := by
  unfold sideLawC
  split_ifs
  · infer_instance
  · exact Measure.isProbabilityMeasure_map (measurable_reserveSeqOf_at _ g).aemeasurable

lemma onSide_disjoint (Hs : Finset S) (g : S → Bool) :
    Disjoint (onSide Hs true g) (onSide Hs false g) := by
  rw [Set.disjoint_left]
  rintro s ⟨h1, -⟩ ⟨h2, -⟩
  rw [h1] at h2
  exact Bool.noConfusion h2

/-- A pair law that is a product of probability measures is the product of its marginals. -/
lemma eq_prod_marginals {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (ν : Measure (α × β)) (μ₁ : Measure α) (μ₂ : Measure β) [IsProbabilityMeasure μ₁]
    [IsProbabilityMeasure μ₂] (h : ν = μ₁.prod μ₂) :
    ν = (ν.map Prod.fst).prod (ν.map Prod.snd) := by
  rw [h, Measure.map_fst_prod, Measure.map_snd_prod]
  simp

open scoped Classical in
/-- One population's two sequences have the product law of `sideLawT` and `sideLawC`. -/
lemma map_pair_eq (H : J → Finset S) (uni : J) (D : J → Measure S)
    [∀ j, IsProbabilityMeasure (D j)] (g : S → Bool) (j : J) :
    Measure.map (fun p : (ℕ → S) × (ℕ → S) =>
        (if j = uni then p.1
          else tableSeqOf (H j).toList (onSide (allHarvest H) true g) p.1 p.2,
         if j = uni then p.2 else reserveSeqOf (onSide (allHarvest H) false g) p.1 p.2))
      ((iidStream (D j)).prod (iidStream (D j)))
      = (sideLawT H uni D g j).prod (sideLawC H uni D g j) := by
  by_cases hj : j = uni
  · simp only [hj, if_true, sideLawT, sideLawC]
    exact Measure.map_id
  · simp only [hj, if_false, sideLawT, sideLawC]
    obtain ⟨μ₁, μ₂, h1, h2, hlaw, -, -⟩ := map_sides (D j) (H j).toList
      (onSide (allHarvest H) true g) (onSide (allHarvest H) false g)
      (onSide_disjoint (allHarvest H) g)
    have hm := (measurable_tableSeqOf_at (H j).toList (allHarvest H) g).prodMk
      (measurable_reserveSeqOf_at (allHarvest H) g)
    rw [eq_prod_marginals _ μ₁ μ₂ hlaw, Measure.map_map measurable_fst hm,
      Measure.map_map measurable_snd hm]
    rfl

open scoped Classical in
/-- For fixed coins, the sequences the round reads have the stream-level product law. -/
theorem map_streams (H : J → Finset S) (uni : J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (g : S → Bool) :
    Measure.map (fun x : Run Ω S J => streams H uni (x, g)) (baseRun μ D Dsf)
      = streamMeasure μ (sideLawT H uni D g) (sideLawC H uni D g) Dsf := by
  set Ψ : J → (ℕ → S) × (ℕ → S) → (ℕ → S) × (ℕ → S) := fun j p =>
    (if j = uni then p.1 else tableSeqOf (H j).toList (onSide (allHarvest H) true g) p.1 p.2,
     if j = uni then p.2 else reserveSeqOf (onSide (allHarvest H) false g) p.1 p.2) with hΨ
  have hmΨ : ∀ j, Measurable (Ψ j) := by
    intro j
    by_cases hj : j = uni
    · simp only [hΨ, hj, if_true]; exact measurable_id
    · simp only [hΨ, hj, if_false]
      exact (measurable_tableSeqOf_at _ _ g).prodMk (measurable_reserveSeqOf_at _ g)
  set e := MeasurableEquiv.arrowProdEquivProdArrow (ℕ → S) (ℕ → S) J with he
  set Φ : (J → ℕ → S) × (J → ℕ → S) → (J → ℕ → S) × (J → ℕ → S) :=
    fun q => e (fun j => Ψ j (q.1 j, q.2 j)) with hΦ
  have hmΦ : Measurable Φ := by
    rw [hΦ]
    exact e.measurable.comp (measurable_pi_lambda _ (fun j => (hmΨ j).comp
      (((measurable_pi_apply j).comp measurable_fst).prodMk
        ((measurable_pi_apply j).comp measurable_snd))))
  have hΦlaw : Measure.map Φ ((Measure.pi fun j => iidStream (D j)).prod
      (Measure.pi fun j => iidStream (D j)))
      = (Measure.pi (sideLawT H uni D g)).prod (Measure.pi (sideLawC H uni D g)) := by
    have h1 := (measurePreserving_arrowProdEquivProdArrow (ℕ → S) (ℕ → S) J
      (fun j => iidStream (D j)) (fun j => iidStream (D j))).symm
    have h2 : Measure.map (fun V (j : J) => Ψ j (V j))
        (Measure.pi fun j => (iidStream (D j)).prod (iidStream (D j)))
        = Measure.pi fun j => (sideLawT H uni D g j).prod (sideLawC H uni D g j) := by
      rw [Measure.pi_map_pi (fun j => (hmΨ j).aemeasurable)]
      congr 1
      funext j
      exact map_pair_eq H uni D g j
    have h3 := (measurePreserving_arrowProdEquivProdArrow (ℕ → S) (ℕ → S) J
      (sideLawT H uni D g) (sideLawC H uni D g))
    have hcomp : Φ = e ∘ (fun V (j : J) => Ψ j (V j)) ∘ e.symm := by
      funext q
      simp only [hΦ, Function.comp, he]
      rfl
    have hF : MeasurePreserving (fun V (j : J) => Ψ j (V j))
        (Measure.pi fun j => (iidStream (D j)).prod (iidStream (D j)))
        (Measure.pi fun j => (sideLawT H uni D g j).prod (sideLawC H uni D g j)) :=
      ⟨measurable_pi_lambda _ (fun j => (hmΨ j).comp (measurable_pi_apply j)), h2⟩
    rw [hcomp]
    exact (h3.comp (hF.comp h1)).map_eq
  -- the run: noise and suffixes untouched, the streams mapped by `Φ`
  have hrun : (fun x : Run Ω S J => streams H uni (x, g))
      = Prod.map id (fun d : ((ℕ → S) × (J → ℕ → S)) × (J → ℕ → S) =>
          ((d.1.1, (Φ (d.1.2, d.2)).1), (Φ (d.1.2, d.2)).2)) := by
    funext x
    simp only [streams, Prod.map, hΦ, hΨ, he]
    rfl
  have hmInner : Measurable (fun d : ((ℕ → S) × (J → ℕ → S)) × (J → ℕ → S) =>
      ((d.1.1, (Φ (d.1.2, d.2)).1), (Φ (d.1.2, d.2)).2)) := by
    have hq : Measurable (fun d : ((ℕ → S) × (J → ℕ → S)) × (J → ℕ → S) =>
        Φ (d.1.2, d.2)) := hmΦ.comp ((measurable_snd.comp measurable_fst).prodMk measurable_snd)
    exact ((measurable_fst.comp measurable_fst).prodMk (measurable_fst.comp hq)).prodMk
      (measurable_snd.comp hq)
  rw [hrun, baseRun, streamMeasure, streamMeasure,
    ← Measure.map_prod_map _ _ measurable_id hmInner, Measure.map_id]
  congr 1
  -- the inner block
  have hassoc := measurePreserving_prodAssoc (Measure.infinitePi fun _ : ℕ => Dsf)
    (Measure.pi fun j => iidStream (D j)) (Measure.pi fun j => iidStream (D j))
  have hinner : (fun d : ((ℕ → S) × (J → ℕ → S)) × (J → ℕ → S) =>
      ((d.1.1, (Φ (d.1.2, d.2)).1), (Φ (d.1.2, d.2)).2))
      = MeasurableEquiv.prodAssoc.symm ∘ Prod.map id Φ ∘ MeasurableEquiv.prodAssoc := by
    funext d
    rfl
  rw [hinner, ← Measure.map_map MeasurableEquiv.prodAssoc.symm.measurable
    ((measurable_id.prodMap hmΦ).comp MeasurableEquiv.prodAssoc.measurable),
    ← Measure.map_map (measurable_id.prodMap hmΦ) MeasurableEquiv.prodAssoc.measurable,
    hassoc.map_eq, ← Measure.map_prod_map _ _ measurable_id hmΦ, Measure.map_id, hΦlaw]
  exact (measurePreserving_prodAssoc (Measure.infinitePi fun _ : ℕ => Dsf)
    (Measure.pi (sideLawT H uni D g)) (Measure.pi (sideLawC H uni D g))).symm.map_eq

end OrthoDFA
