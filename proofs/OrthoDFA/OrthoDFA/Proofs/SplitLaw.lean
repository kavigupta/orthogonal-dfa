import OrthoDFA.Proofs.Windows

/-!
# The law of one population's two sides

A named population reads its table sequence and its reserve sequence off one i.i.d. stream `w`,
with a spare stream `z` for the pads.  The two sequences are independent, and where a side has
mass its sequence is i.i.d. of the conditional law.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory Thin
open scoped ENNReal

variable {S : Type*} [Stringlike S]

lemma sub_eq_of_unbounded (A : Set S) (w : ℕ → S) (h : ∀ k, ∃ n, k < (drawsIn A w n).length)
    (d d' : S) : nthDrawIn A d w = nthDrawIn A d' w := by
  funext k
  obtain ⟨n, hn⟩ := h k
  rw [sub_eq_getD A d w hn, sub_eq_getD A d' w hn, List.getD_eq_getElem _ _ hn,
    List.getD_eq_getElem _ _ hn]

lemma sub_eq_const_of_none (A : Set S) (w : ℕ → S) (h : ∀ n, w n ∉ A) (d : S) :
    nthDrawIn A d w = fun _ => d := by
  classical
  have hnil : ∀ n, drawsIn A w n = [] := by
    intro n
    induction n with
    | zero => exact entries_zero A w
    | succ n ih => rw [entries_succ, ih, if_neg (h n)]; rfl
  funext k
  rw [nthDrawIn, dif_neg (by simp [hnil])]

lemma measurable_sub_pad (A : Set S) : Measurable (fun p : S × (ℕ → S) => nthDrawIn A p.1 p.2) := by
  have h : Measurable (fun p : (ℕ → S) × S => nthDrawIn A p.2 p.1) :=
    measurable_from_prod_countable_left (fun d => measurable_sub A d)
  exact h.comp measurable_swap

lemma measurable_tableSeqOf (L : List S) (A : Set S) :
    Measurable (fun p : (ℕ → S) × (ℕ → S) => tableSeqOf L A p.1 p.2) := by
  unfold tableSeqOf
  refine (measurable_prependStream L).comp ?_
  have h1 : Measurable (fun p : (ℕ → S) × (ℕ → S) => nthDrawIn A (p.2 0) p.1 0) := by
    show Measurable ((fun f : ℕ → S => f 0) ∘ (fun p : S × (ℕ → S) => nthDrawIn A p.1 p.2)
      ∘ (fun p : (ℕ → S) × (ℕ → S) => (p.2 0, p.1)))
    exact (measurable_pi_apply 0).comp ((measurable_sub_pad A).comp
      (((measurable_pi_apply 0).comp measurable_snd).prodMk measurable_fst))
  have h2 : Measurable (fun p : (ℕ → S) × (ℕ → S) => L.headD (nthDrawIn A (p.2 0) p.1 0)) :=
    (measurable_of_countable (fun s : S => L.headD s)).comp h1
  exact (measurable_sub_pad A).comp (h2.prodMk measurable_fst)

lemma measurable_reserveSeqOf (R : Set S) :
    Measurable (fun p : (ℕ → S) × (ℕ → S) => reserveSeqOf R p.1 p.2) := by
  unfold reserveSeqOf
  have h1 : Measurable (fun p : (ℕ → S) × (ℕ → S) => nthDrawIn R (p.2 1) p.1 0) := by
    show Measurable ((fun f : ℕ → S => f 0) ∘ (fun p : S × (ℕ → S) => nthDrawIn R p.1 p.2)
      ∘ (fun p : (ℕ → S) × (ℕ → S) => (p.2 1, p.1)))
    exact (measurable_pi_apply 0).comp ((measurable_sub_pad R).comp
      (((measurable_pi_apply 1).comp measurable_snd).prodMk measurable_fst))
  exact (measurable_sub_pad R).comp (h1.prodMk measurable_fst)

/-- No mass on `A`: no draw lands there. -/
lemma ae_none_of_zero (D : Measure S) [IsProbabilityMeasure D] (A : Set S) (hA : D A = 0) :
    ∀ᵐ w ∂(iidS D), ∀ n, w n ∉ A := by
  have h := ae_forall_mem D Aᶜ (by rwa [compl_compl])
  filter_upwards [h] with w hw n
  exact hw n


/-- `((a, b), (c, d)) ↦ ((a, c), (b, d))` carries the product law to the product law. -/
lemma rearrange_measurePreserving {α β γ δ : Type*} [MeasurableSpace α] [MeasurableSpace β]
    [MeasurableSpace γ] [MeasurableSpace δ] (μa : Measure α) (μb : Measure β) (μc : Measure γ)
    (μd : Measure δ) [SFinite μa] [SFinite μb] [SFinite μc] [SFinite μd] :
    MeasurePreserving (fun q : (α × β) × (γ × δ) => ((q.1.1, q.2.1), (q.1.2, q.2.2)))
      ((μa.prod μb).prod (μc.prod μd)) ((μa.prod μc).prod (μb.prod μd)) := by
  have h1 := measurePreserving_prodAssoc μa μb (μc.prod μd)
  have h2 := (MeasurePreserving.id μa).prod (measurePreserving_prodAssoc μb μc μd).symm
  have h3 := (MeasurePreserving.id μa).prod
    ((Measure.measurePreserving_swap (μ := μb) (ν := μc)).prod (MeasurePreserving.id μd))
  have h4 := (MeasurePreserving.id μa).prod (measurePreserving_prodAssoc μc μb μd)
  have h5 := (measurePreserving_prodAssoc μa μc (μb.prod μd)).symm
  have h := h5.comp (h4.comp (h3.comp (h2.comp h1)))
  convert h using 1
  funext q
  rfl

/-- The two sides of one population: independent, and i.i.d. of the conditional laws where a side
has mass. -/
theorem map_sides (D : Measure S) [IsProbabilityMeasure D] (L : List S) (A R : Set S)
    (hAR : Disjoint A R) :
    ∃ μ₁ μ₂ : Measure (ℕ → S), IsProbabilityMeasure μ₁ ∧ IsProbabilityMeasure μ₂
      ∧ Measure.map (fun p : (ℕ → S) × (ℕ → S) => (tableSeqOf L A p.1 p.2, reserveSeqOf R p.1 p.2))
          ((iidS D).prod (iidS D)) = μ₁.prod μ₂
      ∧ (D A ≠ 0 → μ₁ = Measure.map (prependStream L) (iidS (D[|A])))
      ∧ (D R ≠ 0 → μ₂ = iidS (D[|R])) := by
  classical
  have hne : Nonempty S := by
    by_contra h
    rw [not_nonempty_iff] at h
    have := measure_univ (μ := D)
    rw [Set.univ_eq_empty_iff.2 h, measure_empty] at this
    exact zero_ne_one this
  obtain ⟨d⟩ := hne
  obtain ⟨μu, μv, hμu, hμv, hlaw, hu, hv⟩ := map_sub_pair_prod D A R hAR d d
  -- what each side is, almost surely, given its subsequence and its spare draw
  set Φ₁ : (ℕ → S) × S → ℕ → S := fun q =>
    if D A ≠ 0 then prependStream L q.1 else prependStream L (fun _ => L.headD q.2) with hΦ₁
  set Φ₂ : (ℕ → S) × S → ℕ → S := fun q => if D R ≠ 0 then q.1 else (fun _ => q.2) with hΦ₂
  have hmΦ₁ : Measurable Φ₁ := by
    rw [hΦ₁]
    split_ifs
    · exact (measurable_prependStream L).comp measurable_fst
    · exact (measurable_prependStream L).comp
        (measurable_pi_lambda _ (fun _ =>
          (measurable_of_countable (fun s : S => L.headD s)).comp measurable_snd))
  have hmΦ₂ : Measurable Φ₂ := by
    rw [hΦ₂]
    split_ifs
    · exact measurable_fst
    · exact measurable_pi_lambda _ (fun _ => measurable_snd)
  have hT : ∀ᵐ w ∂(iidS D), ∀ z : ℕ → S, tableSeqOf L A w z = Φ₁ (nthDrawIn A d w, z 0) := by
    by_cases hA : D A = 0
    · filter_upwards [ae_none_of_zero D A hA] with w hw z
      simp only [hΦ₁, hA, ne_eq, not_true_eq_false, if_false, tableSeqOf]
      rw [sub_eq_const_of_none A w hw, sub_eq_const_of_none A w hw]
    · filter_upwards [ae_entries_unbounded D A hA] with w hw z
      simp only [hΦ₁, hA, ne_eq, not_false_eq_true, if_true, tableSeqOf]
      rw [sub_eq_of_unbounded A w hw _ d]
  have hC : ∀ᵐ w ∂(iidS D), ∀ z : ℕ → S, reserveSeqOf R w z = Φ₂ (nthDrawIn R d w, z 1) := by
    by_cases hR : D R = 0
    · filter_upwards [ae_none_of_zero D R hR] with w hw z
      simp only [hΦ₂, hR, ne_eq, not_true_eq_false, if_false, reserveSeqOf]
      rw [sub_eq_const_of_none R w hw, sub_eq_const_of_none R w hw]
    · filter_upwards [ae_entries_unbounded D R hR] with w hw z
      simp only [hΦ₂, hR, ne_eq, not_false_eq_true, if_true, reserveSeqOf]
      rw [sub_eq_of_unbounded R w hw _ d]
  -- the a.s. form, as a function of independent pieces
  set Ψ : ((ℕ → S) × (ℕ → S)) × (S × S) → (ℕ → S) × (ℕ → S) :=
    fun q => (Φ₁ (q.1.1, q.2.1), Φ₂ (q.1.2, q.2.2)) with hΨ
  set G : (ℕ → S) × (ℕ → S) → ((ℕ → S) × (ℕ → S)) × (S × S) :=
    fun p => ((nthDrawIn A d p.1, nthDrawIn R d p.1), (p.2 0, p.2 1)) with hG
  have hmG : Measurable G := by
    rw [hG]
    exact (((measurable_sub A d).comp measurable_fst).prodMk
      ((measurable_sub R d).comp measurable_fst)).prodMk
      (((measurable_pi_apply 0).comp measurable_snd).prodMk
        ((measurable_pi_apply 1).comp measurable_snd))
  have hmΨ : Measurable Ψ := by
    rw [hΨ]
    exact (hmΦ₁.comp ((measurable_fst.comp measurable_fst).prodMk
      (measurable_fst.comp measurable_snd))).prodMk
      (hmΦ₂.comp ((measurable_snd.comp measurable_fst).prodMk
        (measurable_snd.comp measurable_snd)))
  have hae : (fun p : (ℕ → S) × (ℕ → S) => (tableSeqOf L A p.1 p.2, reserveSeqOf R p.1 p.2))
      =ᵐ[(iidS D).prod (iidS D)] Ψ ∘ G := by
    refine (Measure.ae_prod_mem_iff_ae_ae_mem ?_).2 ?_
    · exact measurableSet_eq_fun ((measurable_tableSeqOf L A).prodMk (measurable_reserveSeqOf R))
        (hmΨ.comp hmG)
    · filter_upwards [hT, hC] with w hw1 hw2
      exact Filter.Eventually.of_forall (fun z => by
        show (tableSeqOf L A w z, reserveSeqOf R w z) = (Φ₁ (nthDrawIn A d w, z 0), Φ₂ (nthDrawIn R d w, z 1))
        rw [hw1 z, hw2 z])
  -- the pieces' law
  have hz : Measure.map (fun z : ℕ → S => (z 0, z 1)) (iidS D) = D.prod D := by
    refine (Measure.prod_eq (fun s t hs ht => ?_)).symm
    rw [Measure.map_apply (by fun_prop) (hs.prod ht)]
    have e : (fun z : ℕ → S => (z 0, z 1)) ⁻¹' (s ×ˢ t)
        = Set.pi ↑({0, 1} : Finset ℕ) (fun i => if i = 0 then s else t) := by
      ext z
      simp
    rw [e, Measure.infinitePi_pi _ (fun i _ => by split_ifs <;> assumption),
      Finset.prod_pair (by norm_num)]
    simp
  have hGlaw : Measure.map G ((iidS D).prod (iidS D)) = (μu.prod μv).prod (D.prod D) := by
    rw [hG, show (fun p : (ℕ → S) × (ℕ → S) => ((nthDrawIn A d p.1, nthDrawIn R d p.1), (p.2 0, p.2 1)))
        = Prod.map (fun w => (nthDrawIn A d w, nthDrawIn R d w)) (fun z : ℕ → S => (z 0, z 1)) from rfl,
      ← Measure.map_prod_map _ _ ((measurable_sub A d).prodMk (measurable_sub R d))
        (by fun_prop), hlaw, hz]
  refine ⟨Measure.map Φ₁ (μu.prod D), Measure.map Φ₂ (μv.prod D),
    Measure.isProbabilityMeasure_map hmΦ₁.aemeasurable,
    Measure.isProbabilityMeasure_map hmΦ₂.aemeasurable, ?_, ?_, ?_⟩
  · rw [Measure.map_congr hae, ← Measure.map_map hmΨ hmG, hGlaw,
      Measure.map_prod_map _ _ hmΦ₁ hmΦ₂]
    have hswap : Measure.map (fun q : ((ℕ → S) × (ℕ → S)) × (S × S) =>
        ((q.1.1, q.2.1), (q.1.2, q.2.2))) ((μu.prod μv).prod (D.prod D))
        = (μu.prod D).prod (μv.prod D) := by
      exact (rearrange_measurePreserving μu μv D D).map_eq
    rw [← hswap, Measure.map_map (hmΦ₁.prodMap hmΦ₂) (by fun_prop)]
    rfl
  · intro hA
    rw [hu hA, hΦ₁]
    simp only [ne_eq, hA, not_false_eq_true, if_true]
    rw [show (fun q : (ℕ → S) × S => prependStream L q.1) = prependStream L ∘ Prod.fst from rfl,
      ← Measure.map_map (measurable_prependStream L) measurable_fst, Measure.map_fst_prod,
      measure_univ, one_smul]
  · intro hR
    rw [hv hR, hΦ₂]
    simp only [ne_eq, hR, not_false_eq_true, if_true]
    rw [show (fun q : (ℕ → S) × S => q.1) = Prod.fst from rfl, Measure.map_fst_prod,
      measure_univ, one_smul]

end OrthoDFA
