import OrthoDFA.Proofs.Level5Deep

/-!
# The deep event's tails

Each field is bounded as the noise event's is, at the largest mass of a length-`K` prefix: once
the reads of `headStrings` are fixed, the strings off them that a probe reads begin with its first
`K` letters, the probe's first reads off them are fresh, and `block_tail` applies; fixing the
reads costs nothing (`measure_le_of_fixed`). The per-probe means count reads off `headStrings`
only, so they need the laws only there.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

attribute [local instance] ARU.fintype

section Mean

variable {σ : Type*} (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ]

/-- `edge_first_mean` with the undecided reads counted only at strings satisfying `Q`, whose
undecided chance is at most `θ'`. -/
theorem edge_first_mean_str {β : Type*} (read : FreeMonoid α → Ω → ARU)
    (hmeas : ∀ w, Measurable (read w)) (hind : iIndepFun read μ) (Q : FreeMonoid α → Prop)
    [DecidablePred Q] (θ' : ℝ) (hQ : ∀ w, Q w → μ.real {ω | read w ω = .undecided} ≤ θ')
    (e : List Bool × α) (t : ProbeQuery α β) :
    ∫⁻ ω, (((t.firsts (fun y => read y ω) ∅).filter fun q =>
        q.1.2 = some e ∧ read q.2 ω = .undecided ∧ Q q.2).length : ENNReal) ∂μ
      ≤ ENNReal.ofReal θ' * ∫⁻ ω,
        (((t.firsts (fun y => read y ω) ∅).filter fun q => q.1.2 = some e).length : ENNReal)
            ∂μ := by
  have hcount : ∀ (rd : FreeMonoid α → ARU) (f : ReadLabel α → FreeMonoid α → ARU → Prop)
      [∀ ℓ w v, Decidable (f ℓ w v)] (c : ENNReal),
      QTree.fsum (fun ℓ w v => if f ℓ w v then c else 0) rd ∅ t
        = c * ((t.firsts rd ∅).filter fun q => f q.1 q.2 (rd q.2)).length := by
    intro rd f _ c
    unfold QTree.fsum
    induction t.firsts rd ∅ with
    | nil => simp
    | cons q l ih =>
      simp only [List.map_cons, List.sum_cons, ih, List.filter_cons]
      by_cases h : f q.1 q.2 (rd q.2)
      · simp [h, mul_add, add_comm]
      · simp [h]
  have h1 := QTree.firsts_lintegral_le (μ := μ)
    (fun ℓ w v => if ℓ.2 = some e ∧ v = .undecided ∧ Q w then 1 else 0)
    (fun ℓ _ _ => if ℓ.2 = some e then ENNReal.ofReal θ' else 0) t read ∅ hmeas hind
    (fun ℓ w _ => by
      by_cases hl : ℓ.2 = some e
      · by_cases hp : Q w
        · simp only [hl, hp, true_and, and_true, if_true]
          rw [lintegral_ite_read read hmeas, one_mul, lintegral_const, measure_univ, mul_one,
            ← ofReal_measureReal (measure_ne_top _ _)]
          exact ENNReal.ofReal_le_ofReal (hQ w hp)
        · simp [hp]
      · simp [hl])
  simp only [hcount, one_mul] at h1
  rw [← lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
  convert h1 using 3 <;> simp

open scoped Classical in
theorem undecOff_eq (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : EdgeTable α)
    (d : List Bool × α) (Z : Finset (FreeMonoid α)) (x : FreeMonoid α) :
    G.undecOff rd k T edges d Z x = (((probeQuery T edges k x).firsts rd ∅).filter fun q =>
      q.1.2 = some d ∧ rd q.2 = .undecided
        ∧ (G.Good (G.M.eval q.2.toList) ∧ q.2 ∉ Z)).length := by
  unfold ReadModel.undecOff edgeHarvestBy firstsBy
  rw [ofCut_rd, List.filter_map, List.length_map, List.filter_filter]
  congr 1
  refine List.filter_congr fun q _ => ?_
  have hc : ((rd q.2).cut = none) ↔ rd q.2 = .undecided := by
    cases rd q.2 <;> simp [ARU.cut]
  by_cases h1 : q.1.2 = some d <;> by_cases h2 : rd q.2 = .undecided
  · simp [h1, h2, ARU.cut]
  · have h2' : ¬ (rd q.2).cut = none := fun h => h2 (hc.1 h)
    simp [h1, h2, h2']
  · simp [h1]
  · simp [h1]

theorem undecOff_le (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : EdgeTable α)
    (d : List Bool × α) (Z : Finset (FreeMonoid α)) (x : FreeMonoid α) :
    G.undecOff rd k T edges d Z x ≤ (edgeHarvestBy (fun z => (rd z).cut) T edges k x d).length :=
  List.length_filter_le _ _

open scoped Classical in
/-- `undecAt_mean_real` off `Z`, the laws needed only off `Z`. -/
theorem undecOff_mean_real (read : FreeMonoid α → Ω → ARU) (hmeas : ∀ w, Measurable (read w))
    (hind : iIndepFun read μ) (Z : Finset (FreeMonoid α)) (hθ : 0 ≤ G.θ)
    (hQ : ∀ w, w ∉ Z → G.Good (G.M.eval w.toList) →
      μ.real {ω | read w ω = .undecided} ≤ 3 / 2 * G.θ)
    (T : DTree α) (edges : EdgeTable α) (k : ℕ) (d : List Bool × α) (x : FreeMonoid α) :
    ∫ ω, (G.undecOff (read · ω) k T edges d Z x : ℝ) ∂μ
      ≤ 3 / 2 * G.θ * ∫ ω, (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ) ∂μ := by
  have h := edge_first_mean_str read hmeas hind
    (fun w => G.Good (G.M.eval w.toList) ∧ w ∉ Z) (3 / 2 * G.θ) (fun w hw => hQ w hw.2 hw.1) d
    (probeQuery T edges k x)
  have hmU : Measurable fun ω => (G.undecOff (read · ω) k T edges d Z x : ℝ) :=
    measurable_of_strings hmeas (prefStrings T x)
      (fun rd => (G.undecOff rd k T edges d Z x : ℝ)) fun rd rd' h => by
        simp only [ReadModel.undecOff, edgeHarvestBy_congr_pref (edges := edges) (k := k) (T := T)
          (x := x) (cut := fun z => (rd z).cut) (cut' := fun z => (rd' z).cut) fun z hz => by
            rw [h z hz]]
  have hmN : Measurable fun ω => (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ) :=
    measurable_of_strings hmeas (prefStrings T x)
      (fun rd => (edgeReadsBy (fun z => (rd z).cut) T edges k x d : ℝ)) fun rd rd' h => by
        simp only [edgeReadsBy_congr_pref (edges := edges) (k := k) (T := T) (x := x)
          (cut := fun z => (rd z).cut) (cut' := fun z => (rd' z).cut) fun z hz => by
            rw [h z hz]]
  have hbN : ∀ ω, (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ)
      ≤ (x.toList.length + 1) * T.midfixes.card := fun ω => by
    have := edgeReadsBy_le (fun z => (read z ω).cut) T edges k x d
    exact_mod_cast this
  have hbU : ∀ ω, (G.undecOff (read · ω) k T edges d Z x : ℝ)
      ≤ (x.toList.length + 1) * T.midfixes.card := fun ω => by
    have h1 := undecOff_le G (read · ω) k T edges d Z x
    have h2 := edgeHarvestBy_le_edgeReadsBy (fun z => (read z ω).cut) T edges k x d
    have h3 := edgeReadsBy_le (fun z => (read z ω).cut) T edges k x d
    have := h1.trans (h2.trans h3)
    exact_mod_cast this
  have hiU : Integrable (fun ω => (G.undecOff (read · ω) k T edges d Z x : ℝ)) μ :=
    Integrable.of_bound hmU.aestronglyMeasurable _ (Filter.Eventually.of_forall fun ω => by
      rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]; exact hbU ω)
  have hiN : Integrable (fun ω => (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ)) μ :=
    Integrable.of_bound hmN.aestronglyMeasurable _ (Filter.Eventually.of_forall fun ω => by
      rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]; exact hbN ω)
  have e1 : ∫⁻ ω, (((((probeQuery T edges k x).firsts (fun y => read y ω) ∅).filter fun q =>
      q.1.2 = some d ∧ read q.2 ω = .undecided
        ∧ (G.Good (G.M.eval q.2.toList) ∧ q.2 ∉ Z)).length : ℕ) : ENNReal) ∂μ
      = ENNReal.ofReal (∫ ω, (G.undecOff (read · ω) k T edges d Z x : ℝ) ∂μ) := by
    rw [ofReal_integral_eq_lintegral_ofReal hiU
      (Filter.Eventually.of_forall fun ω => Nat.cast_nonneg _)]
    congr 1; funext ω
    rw [undecOff_eq G (read · ω), ENNReal.ofReal_natCast]
  have e2 : ∫⁻ ω, (((((probeQuery T edges k x).firsts (fun y => read y ω) ∅).filter fun q =>
      q.1.2 = some d).length : ℕ) : ENNReal) ∂μ
      = ENNReal.ofReal (∫ ω, (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ) ∂μ) := by
    rw [ofReal_integral_eq_lintegral_ofReal hiN
      (Filter.Eventually.of_forall fun ω => Nat.cast_nonneg _)]
    congr 1; funext ω
    rw [edgeReadsBy_eq (read · ω), ENNReal.ofReal_natCast]
  rw [e1, e2, ← ENNReal.ofReal_mul (by positivity)] at h
  have hI : 0 ≤ ∫ ω, (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ) ∂μ :=
    integral_nonneg fun ω => Nat.cast_nonneg _
  exact (ENNReal.ofReal_le_ofReal_iff (by positivity)).1 h

open scoped Classical in
/-- `pt_mean_le` off `Z`, the laws needed only off `Z`. -/
theorem pt_mean_off (read : FreeMonoid α → Ω → ARU) (hmeas : ∀ z, Measurable (read z))
    (hind : iIndepFun read μ) (Z : Finset (FreeMonoid α))
    (hQ : ∀ w, w ∉ Z → G.Good (G.M.eval w.toList) →
      μ.real {ω | read w ω = .undecided} ≤ 3 / 2 * G.θ) (hθ : 0 ≤ G.θ)
    (T : DTree α) (edges : EdgeTable α) {k : ℕ} {x : FreeMonoid α} (hkx : k ≤ x.toList.length) :
    μ.real {ω | x ∈ G.ptOff (read · ω) k T edges Z}
      ≤ (x.toList.length + 1) * T.midfixes.card * (3 / 2 * G.θ)
        * μ.real {ω | x ∈ searchAt (read · ω) k T edges} := by
  set J := Finset.Icc k x.toList.length
  set Srch : Set Ω := {ω | (probeBy (fun y => (read y ω).cut) T edges k x).IsSearch}
  set Gd : ℕ → FreeMonoid α → Prop := fun j m =>
    G.Good (G.M.eval (prefixOf x j * m).toList) ∧ prefixOf x j * m ∉ Z
  have hsrch : {ω | x ∈ searchAt (read · ω) k T edges} = Srch := by
    ext ω; simp only [Set.mem_ofPred_eq, Srch]; exact mem_searchAt
  have hsub : {ω | x ∈ G.ptOff (read · ω) k T edges Z}
      ⊆ ⋃ j ∈ J, ⋃ m ∈ T.midfixes.filter (Gd j),
        Srch ∩ {ω | read (prefixOf x j * m) ω = .undecided} := by
    intro ω hω
    obtain ⟨b, hb, hg, hbZ⟩ := hω
    unfold middleHarvestBy at hb
    rcases ho : probeBy (fun y => (read y ω).cut) T edges k x with _ | _ | _ | j | _ | _ <;>
      rw [ho] at hb <;> simp only [reduceCtorEq] at hb
    all_goals first
      | (simp at hb)
      | skip
    all_goals
      rcases hs : T.siftCut (fun y => (read y ω).cut) (prefixOf x j) with q
          | b' <;> rw [hs] at hb <;>
        simp only [Sum.elim_inl, Sum.elim_inr, List.cons.injEq, reduceCtorEq, and_true,
          List.nil_eq] at hb
      subst hb
      obtain ⟨m, hm, rfl, hcm⟩ := siftCut_inr T (prefixOf x j) b' hs
      obtain ⟨ps, hi, -, hbr⟩ := probeBy_search _ ho trivial
      have hjk : k < j := (bracketAt_mid _ _ _ _ _ _ hbr).1
      set j' := min j x.toList.length
      have hpj : prefixOf x j = prefixOf x j' := by
        simp only [prefixOf, j']
        rcases le_total j x.toList.length with h | h
        · rw [min_eq_left h]
        · rw [min_eq_right h, List.take_of_length_le h, List.take_of_length_le le_rfl]
      have hj' : j' ∈ J := Finset.mem_Icc.2 ⟨le_min hjk.le hkx, min_le_right _ _⟩
      rw [hpj] at hg hcm hbZ
      refine Set.mem_biUnion hj' (Set.mem_biUnion (Finset.mem_filter.2 ⟨hm, hg, hbZ⟩) ⟨?_, ?_⟩)
      · show (probeBy (fun y => (read y ω).cut) T edges k x).IsSearch
        rw [ho]; trivial
      · simp only [Set.mem_ofPred_eq]
        revert hcm
        cases read (prefixOf x j' * m) ω <;> simp [ARU.cut]
  rw [hsrch]
  calc μ.real {ω | x ∈ G.ptOff (read · ω) k T edges Z}
      ≤ μ.real (⋃ j ∈ J, ⋃ m ∈ T.midfixes.filter (Gd j),
          Srch ∩ {ω | read (prefixOf x j * m) ω = .undecided}) :=
        measureReal_mono hsub (measure_ne_top μ _)
    _ ≤ ∑ j ∈ J, ∑ m ∈ T.midfixes.filter (Gd j),
          μ.real (Srch ∩ {ω | read (prefixOf x j * m) ω = .undecided}) := by
        refine (measureReal_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun j _ => ?_)
        exact measureReal_biUnion_finset_le _ _
    _ ≤ ∑ j ∈ J, ∑ m ∈ T.midfixes.filter (Gd j), μ.real Srch * (3 / 2 * G.θ) := by
        refine Finset.sum_le_sum fun j _ => Finset.sum_le_sum fun m hm => ?_
        refine (search_und_le read hmeas hind T edges hkx _).trans ?_
        have hm' := (Finset.mem_filter.1 hm).2
        exact mul_le_mul_of_nonneg_left (hQ _ hm'.2 hm'.1) measureReal_nonneg
    _ ≤ ∑ _j ∈ J, ∑ _m ∈ T.midfixes, μ.real Srch * (3 / 2 * G.θ) := by
        refine Finset.sum_le_sum fun j _ => Finset.sum_le_sum_of_subset_of_nonneg
          (Finset.filter_subset _ _) fun _ _ _ => mul_nonneg measureReal_nonneg (by positivity)
    _ = J.card * T.midfixes.card * (3 / 2 * G.θ) * μ.real Srch := by
        simp only [Finset.sum_const, nsmul_eq_mul]; ring
    _ ≤ _ := by
        have hJ : (J.card : ℝ) ≤ x.toList.length + 1 := by
          simp only [J, Nat.card_Icc]; exact_mod_cast (by omega : _ ≤ x.toList.length + 1)
        gcongr

end Mean

section One

variable {σ : Type*} (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)

theorem ovr_agree {Z : Finset (FreeMonoid α)} {c : Z → ARU} {S : Finset (FreeMonoid α)}
    {rd rd' : FreeMonoid α → ARU} (h : ∀ z ∈ S \ Z, rd z = rd' z) :
    ∀ z ∈ S, ovr Z rd c z = ovr Z rd' c z := by
  intro z hz
  by_cases hzZ : z ∈ Z
  · simp only [ovr, dif_pos hzZ]
  · simp only [ovr, dif_neg hzZ]; exact h z (Finset.mem_sdiff.2 ⟨hz, hzZ⟩)

omit [IsProbabilityMeasure μ] in
theorem hQ_ovr (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.readDist (G.M.eval z.toList) r)
    (Z : Finset (FreeMonoid α)) (c : Z → ARU) :
    ∀ w, w ∉ Z → G.Good (G.M.eval w.toList) →
      μ.real {ω | ovr Z (read · ω) c w = .undecided} ≤ 3 / 2 * G.θ := by
  intro w hw hg
  simp only [ovr, dif_neg hw]
  rw [hlaw]
  exact hg

/-- `goodEdge_one` off `headStrings T k K`, at the largest mass of a length-`K` prefix. -/
theorem deepEdge_one (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.readDist (G.M.eval z.toList) r) (hθ : 0 ≤ G.θ)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {k K L : ℕ} {p₀ η : ℝ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hp₀ : 0 < p₀)
    (hpmax : ∀ u, D.real {x | prefixOf x K = u} ≤ p₀) (hη : 0 ≤ η) (T : DTree α)
    (edges : EdgeTable α) (d : List Bool × α) :
    μ {ω | 2 * (3 / 2 * G.θ) * ∫ x, (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ) ∂D + η
        < ∫ x, (G.undecOff (read · ω) k T edges d (headStrings T k K) x : ℝ) ∂D}
      ≤ ENNReal.ofReal (Real.exp (-(η / (2 * p₀ * (L + 1)
          * (1 + 4 * (3 / 2 * G.θ) * T.midfixes.card))))) := by
  set Z := headStrings T k K
  set uf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    if k ≤ x.toList.length then (G.undecOff rd k T edges d Z x : ℝ) else 0
  set tf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    if k ≤ x.toList.length then (edgeReadsBy (fun z => (rd z).cut) T edges k x d : ℝ) else 0
  have hagr : ∀ x, k ≤ x.toList.length → ∀ rd rd' : FreeMonoid α → ARU,
      (∀ z ∈ probeStrings T k x, rd z = rd' z) →
      ∀ z ∈ probeStrings T k x, (rd z).cut = (rd' z).cut := fun x _ rd rd' h z hz => by
    rw [h z hz]
  have huf : ∀ x rd rd', (∀ z ∈ probeStrings T k x, rd z = rd' z) → uf rd x = uf rd' x := by
    intro x rd rd' h
    by_cases hkx : k ≤ x.toList.length
    · simp only [uf, if_pos hkx, ReadModel.undecOff,
        edgeHarvestBy_congr_strings (edges := edges) hkx (hagr x hkx rd rd' h)]
    · simp only [uf, if_neg hkx]
  have htf : ∀ x rd rd', (∀ z ∈ probeStrings T k x, rd z = rd' z) → tf rd x = tf rd' x := by
    intro x rd rd' h
    by_cases hkx : k ≤ x.toList.length
    · simp only [tf, if_pos hkx, edgeReadsBy_congr_strings (edges := edges) hkx
        (hagr x hkx rd rd' h)]
    · simp only [tf, if_neg hkx]
  have hub : ∀ rd, ∀ x ∈ lenRange 0 L, 0 ≤ uf rd x ∧ uf rd x ≤ L + 1 := by
    intro rd x hx
    obtain ⟨-, hxL⟩ := mem_lenRange.1 hx
    by_cases hkx : k ≤ x.toList.length
    swap
    · simp only [uf, if_neg hkx]; constructor <;> positivity
    simp only [uf, if_pos hkx]
    refine ⟨Nat.cast_nonneg _, ?_⟩
    have h1 := undecOff_le G rd k T edges d Z x
    have h2 := edgeHarvestBy_length_le (fun z => (rd z).cut) T edges k x d
    have : G.undecOff rd k T edges d Z x ≤ L + 1 := by omega
    exact_mod_cast this
  have htb : ∀ rd, ∀ x ∈ lenRange 0 L, 0 ≤ tf rd x
      ∧ tf rd x ≤ (L + 1) * (T.midfixes.card : ℝ) := by
    intro rd x hx
    obtain ⟨-, hxL⟩ := mem_lenRange.1 hx
    by_cases hkx : k ≤ x.toList.length
    swap
    · simp only [tf, if_neg hkx]; constructor <;> positivity
    simp only [tf, if_pos hkx]
    refine ⟨Nat.cast_nonneg _, ?_⟩
    have h1 := edgeReadsBy_le (fun z => (rd z).cut) T edges k x d
    have : edgeReadsBy (fun z => (rd z).cut) T edges k x d ≤ (L + 1) * T.midfixes.card :=
      h1.trans (Nat.mul_le_mul_right _ (by omega))
    exact_mod_cast this
  have hM : ∀ (f : FreeMonoid α → ℝ), (∀ x, x.toList.length ≤ L → 0 ≤ f x ∧
      f x ≤ (L + 1) * (T.midfixes.card + 1)) →
      ∀ x, x.toList.length ≤ L → |f x| ≤ (L + 1) * (T.midfixes.card + 1) := by
    intro f hf x hx
    rw [abs_of_nonneg (hf x hx).1]; exact (hf x hx).2
  have hUs : ∀ rd, ∫ x, (G.undecOff rd k T edges d Z x : ℝ) ∂D
      = ∑ x ∈ lenRange 0 L, D.real {x} * uf rd x := by
    intro rd
    rw [integral_eq_sum_len D hlen _ ((L + 1) * (T.midfixes.card + 1)) (hM _ fun x hx => by
      refine ⟨Nat.cast_nonneg _, ?_⟩
      have h1 := undecOff_le G rd k T edges d Z x
      have h2 := edgeHarvestBy_length_le (fun z => (rd z).cut) T edges k x d
      have : (G.undecOff rd k T edges d Z x : ℝ) ≤ L + 1 := by
        have : G.undecOff rd k T edges d Z x ≤ L + 1 := by omega
        exact_mod_cast this
      nlinarith [(Nat.cast_nonneg T.midfixes.card : (0 : ℝ) ≤ _)])]
    refine Finset.sum_congr rfl fun x _ => ?_
    by_cases hkx : k ≤ x.toList.length
    · simp only [uf, if_pos hkx]
    · have hs' : x.toList.length < k := by omega
      simp [uf, hkx, ReadModel.undecOff, edgeHarvestBy_short _ T edges hs']
  have hNs : ∀ rd, ∫ x, (edgeReadsBy (fun z => (rd z).cut) T edges k x d : ℝ) ∂D
      = ∑ x ∈ lenRange 0 L, D.real {x} * tf rd x := by
    intro rd
    rw [integral_eq_sum_len D hlen _ ((L + 1) * (T.midfixes.card + 1)) (hM _ fun x hx => by
      refine ⟨Nat.cast_nonneg _, ?_⟩
      have h1 := edgeReadsBy_le (fun z => (rd z).cut) T edges k x d
      have : (edgeReadsBy (fun z => (rd z).cut) T edges k x d : ℝ)
          ≤ (L + 1) * T.midfixes.card := by
        have : edgeReadsBy (fun z => (rd z).cut) T edges k x d ≤ (L + 1) * T.midfixes.card :=
          h1.trans (Nat.mul_le_mul_right _ (by omega))
        exact_mod_cast this
      nlinarith)]
    refine Finset.sum_congr rfl fun x _ => ?_
    by_cases hkx : k ≤ x.toList.length
    · simp only [tf, if_pos hkx]
    · simp [tf, hkx, edgeReadsBy_short _ T edges (show x.toList.length < k by omega)]
  set F : (FreeMonoid α → ARU) → Prop := fun rd =>
    2 * (3 / 2 * G.θ) * ∑ x ∈ lenRange 0 L, D.real {x} * tf rd x + η
      < ∑ x ∈ lenRange 0 L, D.real {x} * uf rd x
  set W : Finset (FreeMonoid α) := (lenRange 0 L : Finset (FreeMonoid α)).biUnion
    (probeStrings T k ·)
  have hF : ∀ rd rd', (∀ z ∈ W, rd z = rd' z) → (F rd ↔ F rd') := by
    intro rd rd' h
    have hx : ∀ x ∈ lenRange 0 L, ∀ z ∈ probeStrings T k x, rd z = rd' z := fun x hx z hz =>
      h z (Finset.mem_biUnion.2 ⟨x, hx, hz⟩)
    have e1 : ∑ x ∈ lenRange 0 L, D.real {x} * uf rd x = ∑ x ∈ lenRange 0 L, D.real {x} * uf rd' x :=
      Finset.sum_congr rfl fun x hxL => by rw [huf x rd rd' (hx x hxL)]
    have e2 : ∑ x ∈ lenRange 0 L, D.real {x} * tf rd x = ∑ x ∈ lenRange 0 L, D.real {x} * tf rd' x :=
      Finset.sum_congr rfl fun x hxL => by rw [htf x rd rd' (hx x hxL)]
    simp only [F, e1, e2]
  have hfix : ∀ c : Z → ARU, μ {ω | F (ovr Z (read · ω) c)}
      ≤ ENNReal.ofReal (Real.exp (-(η / (2 * p₀ * (L + 1)
          * (1 + 4 * (3 / 2 * G.θ) * T.midfixes.card))))) := by
    intro c
    have hTB := block_tail (μ := μ) (read := read) hmeas hind D (k := K) (L := L) hp₀ hpmax hη
      (by positivity : (0 : ℝ) ≤ 3 / 2 * G.θ) (by positivity : (0 : ℝ) < L + 1)
      (Nat.cast_nonneg T.midfixes.card) (fun x => probeStrings T k x \ Z)
      (fun x z hz => prefix_of_notMem_head (Finset.mem_sdiff.1 hz).1 (Finset.mem_sdiff.1 hz).2)
      (fun rd x => uf (ovr Z rd c) x) (fun rd x => tf (ovr Z rd c) x)
      (fun x rd rd' h => huf x _ _ (ovr_agree h)) (fun x rd rd' h => htf x _ _ (ovr_agree h))
      (fun rd x hx => hub _ x hx) (fun rd x hx => htb _ x hx) (fun x hx => by
        by_cases hkx : k ≤ x.toList.length
        swap
        · simp only [uf, tf, if_neg hkx, integral_zero, mul_zero, le_refl]
        simp only [uf, tf, if_pos hkx]
        exact undecOff_mean_real G (fun z ω => ovr Z (read · ω) c z) (measurable_ovr hmeas Z c)
          (iIndepFun_ovr hind Z c) Z hθ (hQ_ovr G read hlaw Z c) T edges k d x)
    exact hTB
  calc μ _ ≤ μ {ω | F (read · ω)} := by
        refine measure_mono fun ω hω => ?_
        simp only [Set.mem_ofPred_eq, F] at hω ⊢
        rw [← hUs, ← hNs]
        exact hω
    _ ≤ _ := measure_le_of_fixed hmeas hind Z W F hF _ hfix


theorem undecOff_econgr {T : DTree α} {e e' : EdgeTable α} (he : EdgesInto T e)
    (hg : TgtAgree T e e') (rd : FreeMonoid α → ARU) (k : ℕ) (d : List Bool × α)
    (Z : Finset (FreeMonoid α)) (x : FreeMonoid α) :
    G.undecOff rd k T e d Z x = G.undecOff rd k T e' d Z x := by
  simp only [ReadModel.undecOff]
  rw [edgeHarvestBy_econgr (k := k) (x := x) he hg (fun z => (rd z).cut)]

open scoped Classical in
/-- `pt_one` off `headStrings T k K`, at the largest mass of a length-`K` prefix. -/
theorem deepPT_one (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.readDist (G.M.eval z.toList) r) (hθ : 0 ≤ G.θ)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {k K L : ℕ} {p₀ η θ₀ : ℝ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hp₀ : 0 < p₀)
    (hpmax : ∀ u, D.real {x | prefixOf x K = u} ≤ p₀) (hη : 0 ≤ η) (T : DTree α)
    (hT : (L + 1) * T.midfixes.card * (3 / 2 * G.θ) ≤ θ₀) (edges : EdgeTable α) :
    μ {ω | 2 * θ₀ * D.real (searchAt (read · ω) k T edges) + η
        < D.real (G.ptOff (read · ω) k T edges (headStrings T k K))}
      ≤ ENNReal.ofReal (Real.exp (-(η / (2 * p₀ * (1 + 4 * θ₀ * 1))))) := by
  have hθ₀ : 0 ≤ θ₀ := le_trans (by positivity) hT
  set Z := headStrings T k K
  set uf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    if k ≤ x.toList.length ∧ x ∈ G.ptOff rd k T edges Z then 1 else 0
  set tf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    if k ≤ x.toList.length ∧ x ∈ searchAt rd k T edges then 1 else 0
  have hagr : ∀ x, k ≤ x.toList.length → ∀ rd rd' : FreeMonoid α → ARU,
      (∀ z ∈ probeStrings T k x, rd z = rd' z) →
      PrefAgreeK (fun z => (rd z).cut) (fun z => (rd' z).cut) T x k := fun x hkx rd rd' h =>
    prefAgreeK_of_strings hkx fun z hz => by simp only [h z hz]
  have huf : ∀ x rd rd', (∀ z ∈ probeStrings T k x, rd z = rd' z) → uf rd x = uf rd' x := by
    intro x rd rd' h
    by_cases hkx : k ≤ x.toList.length
    · simp only [uf, ReadModel.ptOff, Set.mem_ofPred_eq,
        middleHarvestBy_congrK (edges := edges) (hagr x hkx rd rd' h)]
      exact if_congr Iff.rfl rfl rfl
    · simp only [uf, hkx, false_and, if_false]
  have htf : ∀ x rd rd', (∀ z ∈ probeStrings T k x, rd z = rd' z) → tf rd x = tf rd' x := by
    intro x rd rd' h
    by_cases hkx : k ≤ x.toList.length
    · simp only [tf, searchAt_eq, Set.mem_ofPred_eq,
        probeBy_congrK (edges := edges) (hagr x hkx rd rd' h)]
      exact if_congr Iff.rfl rfl rfl
    · simp only [tf, hkx, false_and, if_false]
  have hb01 : ∀ (P : Prop) [Decidable P], 0 ≤ (if P then (1 : ℝ) else 0)
      ∧ (if P then (1 : ℝ) else 0) ≤ 1 := by
    intro P _; split_ifs <;> norm_num
  have hub : ∀ rd, ∀ x ∈ lenRange 0 L, 0 ≤ uf rd x ∧ uf rd x ≤ 1 := fun rd x _ => hb01 _
  have htb : ∀ rd, ∀ x ∈ lenRange 0 L, 0 ≤ tf rd x ∧ tf rd x ≤ 1 * 1 := fun rd x _ => by
    rw [mul_one]; exact hb01 _
  have hPs : ∀ rd, ∑ x ∈ lenRange 0 L, D.real {x} * uf rd x
      = D.real (G.ptOff rd k T edges Z) := by
    intro rd
    rw [measureReal_eq_sum_len D hlen]
    refine Finset.sum_congr rfl fun x _ => ?_
    by_cases hkx : k ≤ x.toList.length
    · simp only [uf, hkx, true_and]
    · have hs' : x.toList.length < k := by omega
      have : x ∉ G.ptOff rd k T edges Z := by
        simp [ReadModel.ptOff, middleHarvestBy_short _ T edges hs']
      simp [uf, hkx, this]
  have hSs : ∀ rd, ∑ x ∈ lenRange 0 L, D.real {x} * tf rd x
      = D.real (searchAt rd k T edges) := by
    intro rd
    rw [measureReal_eq_sum_len D hlen]
    refine Finset.sum_congr rfl fun x _ => ?_
    by_cases hkx : k ≤ x.toList.length
    · simp only [tf, hkx, true_and]
    · have hs' : x.toList.length < k := by omega
      have : x ∉ searchAt rd k T edges := by
        simp [searchAt_eq, probeBy_short _ T edges hs']
      simp [tf, hkx, this]
  set F : (FreeMonoid α → ARU) → Prop := fun rd =>
    2 * θ₀ * ∑ x ∈ lenRange 0 L, D.real {x} * tf rd x + η
      < ∑ x ∈ lenRange 0 L, D.real {x} * uf rd x
  set W : Finset (FreeMonoid α) := (lenRange 0 L : Finset (FreeMonoid α)).biUnion
    (probeStrings T k ·)
  have hF : ∀ rd rd', (∀ z ∈ W, rd z = rd' z) → (F rd ↔ F rd') := by
    intro rd rd' h
    have hx : ∀ x ∈ lenRange 0 L, ∀ z ∈ probeStrings T k x, rd z = rd' z := fun x hx z hz =>
      h z (Finset.mem_biUnion.2 ⟨x, hx, hz⟩)
    have e1 : ∑ x ∈ lenRange 0 L, D.real {x} * uf rd x
        = ∑ x ∈ lenRange 0 L, D.real {x} * uf rd' x :=
      Finset.sum_congr rfl fun x hxL => by rw [huf x rd rd' (hx x hxL)]
    have e2 : ∑ x ∈ lenRange 0 L, D.real {x} * tf rd x
        = ∑ x ∈ lenRange 0 L, D.real {x} * tf rd' x :=
      Finset.sum_congr rfl fun x hxL => by rw [htf x rd rd' (hx x hxL)]
    simp only [F, e1, e2]
  have hset : ∀ (A : Set Ω), MeasurableSet A →
      ∫ ω, (if ω ∈ A then (1 : ℝ) else 0) ∂μ = μ.real A := fun A hA => by
    rw [← integral_indicator_one hA]; congr 1
  have hfix : ∀ c : Z → ARU, μ {ω | F (ovr Z (read · ω) c)}
      ≤ ENNReal.ofReal (Real.exp (-(η / (2 * p₀ * (1 + 4 * θ₀ * 1))))) := by
    intro c
    set rc : FreeMonoid α → Ω → ARU := fun z ω => ovr Z (read · ω) c z
    have hmc := measurable_ovr hmeas Z c
    have hic := iIndepFun_ovr hind Z c
    refine le_trans (block_tail (μ := μ) (read := read) hmeas hind D (k := K) (L := L) hp₀ hpmax hη hθ₀
      one_pos zero_le_one (fun x => probeStrings T k x \ Z)
      (fun x z hz => prefix_of_notMem_head (Finset.mem_sdiff.1 hz).1 (Finset.mem_sdiff.1 hz).2)
      (fun rd x => uf (ovr Z rd c) x) (fun rd x => tf (ovr Z rd c) x)
      (fun x rd rd' h => huf x _ _ (ovr_agree h)) (fun x rd rd' h => htf x _ _ (ovr_agree h))
      (fun rd x hx => hub _ x hx) (fun rd x hx => htb _ x hx) (fun x hx => by
        obtain ⟨-, hxL⟩ := mem_lenRange.1 hx
        by_cases hkx : k ≤ x.toList.length
        swap
        · simp only [uf, tf, hkx, false_and, if_false, integral_zero, mul_zero, le_refl]
        have hmP : MeasurableSet {ω | x ∈ G.ptOff (rc · ω) k T edges Z} := by
          have := measurable_of_strings hmc (probeStrings T k x)
            (fun rd => decide (x ∈ G.ptOff rd k T edges Z)) fun rd rd' h => by
              simp only [ReadModel.ptOff, Set.mem_ofPred_eq,
                middleHarvestBy_congrK (edges := edges) (hagr x hkx rd rd' h)]
              exact decide_eq_decide.2 Iff.rfl
          convert this (measurableSet_singleton true) using 1
          ext ω; simp only [Set.mem_ofPred_eq, Set.mem_preimage, Set.mem_singleton_iff,
            decide_eq_true_eq]; rfl
        have hmS : MeasurableSet {ω | x ∈ searchAt (rc · ω) k T edges} := by
          have := measurable_of_strings hmc (probeStrings T k x)
            (fun rd => decide (x ∈ searchAt rd k T edges)) fun rd rd' h => by
              simp only [searchAt_eq, Set.mem_ofPred_eq,
                probeBy_congrK (edges := edges) (hagr x hkx rd rd' h)]
              exact decide_eq_decide.2 Iff.rfl
          convert this (measurableSet_singleton true) using 1
          ext ω; simp only [Set.mem_ofPred_eq, Set.mem_preimage, Set.mem_singleton_iff,
            decide_eq_true_eq]; rfl
        have e1 : ∫ ω, uf (ovr Z (read · ω) c) x ∂μ
            = μ.real {ω | x ∈ G.ptOff (rc · ω) k T edges Z} := by
          simp only [uf, hkx, true_and]; exact hset _ hmP
        have e2 : ∫ ω, tf (ovr Z (read · ω) c) x ∂μ
            = μ.real {ω | x ∈ searchAt (rc · ω) k T edges} := by
          simp only [tf, hkx, true_and]; exact hset _ hmS
        rw [e1, e2]
        refine (pt_mean_off G rc hmc hic Z (hQ_ovr G read hlaw Z c) hθ T edges hkx).trans ?_
        have hxL' : (x.toList.length : ℝ) + 1 ≤ L + 1 := by
          have : (x.toList.length : ℝ) ≤ L := Nat.cast_le.2 hxL
          linarith
        have h2 : ((x.toList.length : ℝ) + 1) * T.midfixes.card * (3 / 2 * G.θ) ≤ θ₀ :=
          le_trans (mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hxL'
            (Nat.cast_nonneg _)) (by positivity)) hT
        exact mul_le_mul_of_nonneg_right h2 measureReal_nonneg)) (le_of_eq (by congr 4; ring))
  calc μ _ ≤ μ {ω | F (read · ω)} := by
        refine measure_mono fun ω hω => ?_
        simp only [Set.mem_ofPred_eq, F] at hω ⊢
        rw [hPs, hSs]
        exact hω
    _ ≤ _ := measure_le_of_fixed hmeas hind Z W F hF _ hfix

end One

section DeepUnion

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)

/-- `goodEdge_le` off each tree's `headStrings`, at the largest mass of a length-`K` prefix. -/
theorem deepEdge_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.readDist (G.M.eval z.toList) r) (hθ : 0 ≤ G.θ)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k K L S : ℕ) {p₀ η θg : ℝ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hp₀ : 0 < p₀)
    (hpmax : ∀ u, D.real {x | prefixOf x K = u} ≤ p₀) (hη : 0 ≤ η) (hθg : 2 * (3 / 2 * G.θ) ≤ θg) :
    μ {ω | ¬ ∀ T, G.InClass S T → ∀ edges, EdgesInto T edges → ∀ d,
        ∫ x, (G.undecOff (read · ω) k T edges d (headStrings T k K) x : ℝ) ∂D
          ≤ θg * ∫ x, (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ) ∂D + η}
      ≤ ENNReal.ofReal ((classSet (Fintype.card σ + S) : Finset (DTree α)).card
        * ((Fintype.card σ + S + 3) ^ ((Fintype.card σ + S + 2) * Fintype.card α)
          * ((Fintype.card σ + S + 2) * Fintype.card α))
        * Real.exp (-(η / (2 * p₀ * (L + 1)
          * (1 + 4 * (3 / 2 * G.θ) * (Fintype.card σ + S + 1)))))) := by
  set n := Fintype.card σ + S
  set ε : ℝ := Real.exp (-(η / (2 * p₀ * (L + 1) * (1 + 4 * (3 / 2 * G.θ) * (n + 1)))))
  set B : DTree α → EdgeTable α → List Bool × α → Set Ω := fun T e d => {ω |
    2 * (3 / 2 * G.θ) * ∫ x, (edgeReadsBy (fun z => (read z ω).cut) T e k x d : ℝ) ∂D + η
      < ∫ x, (G.undecOff (read · ω) k T e d (headStrings T k K) x : ℝ) ∂D}
  have hsub : {ω | ¬ ∀ T, G.InClass S T → ∀ edges, EdgesInto T edges → ∀ d,
      ∫ x, (G.undecOff (read · ω) k T edges d (headStrings T k K) x : ℝ) ∂D
        ≤ θg * ∫ x, (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ) ∂D + η}
      ⊆ ⋃ T ∈ (classSet n : Finset (DTree α)), ⋃ e ∈ edgeMaps T,
          ⋃ d ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α), B T e d := by
    intro ω hω
    simp only [Set.mem_setOf_eq, not_forall, not_le] at hω
    obtain ⟨T, hT, edges, he, d, hlt⟩ := hω
    obtain ⟨e', he'm, hagr, -⟩ := exists_edgeMaps he
    have hU : ∀ x, G.undecOff (read · ω) k T edges d (headStrings T k K) x
        = G.undecOff (read · ω) k T e' d (headStrings T k K) x := fun x =>
      undecOff_econgr G he hagr _ k d _ x
    have hN : ∀ x, edgeReadsBy (fun z => (read z ω).cut) T edges k x d
        = edgeReadsBy (fun z => (read z ω).cut) T e' k x d := by
      intro x
      rw [edgeReadsBy_econgr (k := k) (x := x) he hagr (fun z => (read z ω).cut)]
    have hd : d.1 ∈ T.leaves := by
      by_contra hd
      have h0 : ∀ x, G.undecOff (read · ω) k T edges d (headStrings T k K) x = 0 := by
        intro x
        simp only [ReadModel.undecOff]
        have : edgeHarvestBy (fun z => (read z ω).cut) T edges k x d = [] :=
          by_contra fun hne => hd (edgeHarvestBy_mem _ T k x he hne)
        simp [this]
      simp only [h0, Nat.cast_zero, integral_zero] at hlt
      have : 0 ≤ ∫ x, (edgeReadsBy (fun z => (read z ω).cut) T edges k x d : ℝ) ∂D :=
        integral_nonneg fun x => Nat.cast_nonneg _
      have : 0 ≤ θg := le_trans (by positivity) hθg
      nlinarith
    refine Set.mem_biUnion (inClass_mem G hT) (Set.mem_biUnion he'm (Set.mem_biUnion
      (Finset.mem_product.2 ⟨List.mem_toFinset.2 hd, Finset.mem_univ _⟩) ?_))
    simp only [B, Set.mem_setOf_eq]
    simp only [hU, hN] at hlt
    have : 0 ≤ ∫ x, (edgeReadsBy (fun z => (read z ω).cut) T e' k x d : ℝ) ∂D :=
      integral_nonneg fun x => Nat.cast_nonneg _
    nlinarith [mul_le_mul_of_nonneg_right hθg this]
  have hone : ∀ T ∈ (classSet n : Finset (DTree α)), ∀ e d, μ (B T e d) ≤ ENNReal.ofReal ε := by
    intro T hT e d
    have hm : (T.midfixes.card : ℝ) ≤ n + 1 := by
      have := midfixes_card T
      have := classSet_paths n T hT
      exact_mod_cast (by omega : T.midfixes.card ≤ n + 1)
    refine (deepEdge_one G read hmeas hind hlaw hθ D hlen hp₀ hpmax hη T e d).trans
      (ENNReal.ofReal_le_ofReal ?_)
    simp only [ε]
    gcongr
  calc μ _ ≤ μ (⋃ T ∈ (classSet n : Finset (DTree α)), ⋃ e ∈ edgeMaps T,
        ⋃ d ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α), B T e d) := measure_mono hsub
    _ ≤ ∑ T ∈ (classSet n : Finset (DTree α)), ∑ e ∈ edgeMaps T,
          ∑ d ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α), μ (B T e d) := by
        refine (measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun T _ => ?_)
        refine (measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun e _ => ?_)
        exact measure_biUnion_finset_le _ _
    _ ≤ ∑ T ∈ (classSet n : Finset (DTree α)),
          (((n + 3) ^ ((n + 2) * Fintype.card α) * ((n + 2) * Fintype.card α) : ℕ) : ENNReal)
            * ENNReal.ofReal ε := by
        refine Finset.sum_le_sum fun T hT => ?_
        have hp := classSet_paths n T hT
        calc ∑ e ∈ edgeMaps T, ∑ d ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α), μ (B T e d)
            ≤ ∑ e ∈ edgeMaps T, ∑ d ∈ T.leaves.toFinset ×ˢ (Finset.univ : Finset α),
                ENNReal.ofReal ε := Finset.sum_le_sum fun e _ =>
              Finset.sum_le_sum fun d _ => hone T hT e d
          _ = ((edgeMaps T).card * (T.leaves.toFinset ×ˢ (Finset.univ : Finset α)).card : ℕ)
                * ENNReal.ofReal ε := by
              simp only [Finset.sum_const, nsmul_eq_mul]; push_cast; ring
          _ ≤ _ := by
              refine mul_le_mul' ?_ le_rfl
              have h1 : (edgeMaps T).card ≤ (n + 3) ^ ((n + 2) * Fintype.card α) :=
                calc (edgeMaps T).card
                    ≤ (T.leaves.length + 1) ^ (T.leaves.length * Fintype.card α) := edgeMaps_card T
                  _ ≤ (n + 3) ^ (T.leaves.length * Fintype.card α) :=
                    Nat.pow_le_pow_left (by omega) _
                  _ ≤ (n + 3) ^ ((n + 2) * Fintype.card α) :=
                    Nat.pow_le_pow_right (by omega) (Nat.mul_le_mul_right _ hp)
              have h2 : (T.leaves.toFinset ×ˢ (Finset.univ : Finset α)).card
                  ≤ (n + 2) * Fintype.card α := by
                simp only [Finset.card_product, Finset.card_univ]
                exact Nat.mul_le_mul_right _ ((List.toFinset_card_le _).trans hp)
              exact_mod_cast Nat.mul_le_mul h1 h2
    _ = _ := by
        rw [Finset.sum_const, nsmul_eq_mul, ← ENNReal.ofReal_natCast,
          ← ENNReal.ofReal_natCast ((n + 3) ^ ((n + 2) * Fintype.card α)
            * ((n + 2) * Fintype.card α)), ← ENNReal.ofReal_mul (by positivity),
          ← ENNReal.ofReal_mul (by positivity)]
        congr 1
        simp only [n, ε]
        push_cast
        ring

end DeepUnion

section DeepPTUnion

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)

/-- `goodPT_le` off each tree's `headStrings`, at the largest mass of a length-`K` prefix. -/
theorem deepPT_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.readDist (G.M.eval z.toList) r) (hθ : 0 ≤ G.θ)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k K L S : ℕ) {p₀ η θgpt : ℝ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hp₀ : 0 < p₀)
    (hpmax : ∀ u, D.real {x | prefixOf x K = u} ≤ p₀) (hη : 0 ≤ η)
    (hθg : 2 * ((L + 1) * (Fintype.card σ + S + 1) * (3 / 2 * G.θ)) ≤ θgpt) :
    μ {ω | ¬ ∀ T edges, G.InClass S T → EdgesInto T edges →
        D.real (G.ptOff (read · ω) k T edges (headStrings T k K))
          ≤ θgpt * D.real (searchAt (read · ω) k T edges) + η}
      ≤ ENNReal.ofReal ((classSet (Fintype.card σ + S) : Finset (DTree α)).card
        * (Fintype.card σ + S + 3) ^ ((Fintype.card σ + S + 2) * Fintype.card α)
        * Real.exp (-(η / (2 * p₀ * (1 + 4 * ((L + 1) * (Fintype.card σ + S + 1)
          * (3 / 2 * G.θ)) * 1))))) := by
  set n := Fintype.card σ + S
  set θ₀ : ℝ := (L + 1) * (n + 1) * (3 / 2 * G.θ)
  have hθ₀ : 0 ≤ θ₀ := by positivity
  have h2θ : 2 * θ₀ ≤ θgpt := by simp only [θ₀, n]; push_cast; linarith
  set ε : ℝ := Real.exp (-(η / (2 * p₀ * (1 + 4 * θ₀ * 1))))
  set B : DTree α → EdgeTable α → Set Ω := fun T e => {ω |
    2 * θ₀ * D.real (searchAt (read · ω) k T e) + η
      < D.real (G.ptOff (read · ω) k T e (headStrings T k K))}
  have hsub : {ω | ¬ ∀ T edges, G.InClass S T → EdgesInto T edges →
      D.real (G.ptOff (read · ω) k T edges (headStrings T k K))
        ≤ θgpt * D.real (searchAt (read · ω) k T edges) + η}
      ⊆ ⋃ T ∈ (classSet n : Finset (DTree α)), ⋃ e ∈ edgeMaps T, B T e := by
    intro ω hω
    simp only [Set.mem_setOf_eq, not_forall, not_le] at hω
    obtain ⟨T, edges, hT, he, hlt⟩ := hω
    obtain ⟨e', he'm, hagr, -⟩ := exists_edgeMaps he
    have hP : G.ptOff (read · ω) k T edges (headStrings T k K) = G.ptOff (read · ω) k T e' (headStrings T k K) := by
      ext x
      simp only [ReadModel.ptOff, Set.mem_ofPred_eq]
      rw [middleHarvestBy_econgr (cut := fun z => (read z ω).cut) (k := k) (x := x) he hagr]
    have hS : searchAt (read · ω) k T edges = searchAt (read · ω) k T e' := by
      ext x
      simp only [searchAt_eq, Set.mem_setOf_eq]
      rw [probeBy_econgr (cut := fun z => (read z ω).cut) (k := k) (x := x) he hagr]
    refine Set.mem_biUnion (inClass_mem G hT) (Set.mem_biUnion he'm ?_)
    simp only [B, Set.mem_setOf_eq]
    rw [hP, hS] at hlt
    have : 0 ≤ D.real (searchAt (read · ω) k T e') := measureReal_nonneg
    nlinarith [mul_le_mul_of_nonneg_right h2θ this]
  have hone : ∀ T ∈ (classSet n : Finset (DTree α)), ∀ e, μ (B T e) ≤ ENNReal.ofReal ε := by
    intro T hT e
    have hm : (T.midfixes.card : ℝ) ≤ n + 1 := by
      have := midfixes_card T
      have := classSet_paths n T hT
      exact_mod_cast (by omega : T.midfixes.card ≤ n + 1)
    have hT' : ((L : ℝ) + 1) * T.midfixes.card * (3 / 2 * G.θ) ≤ θ₀ :=
      mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hm (by positivity)) (by positivity)
    exact deepPT_one G read hmeas hind hlaw hθ D hlen hp₀ hpmax hη T hT' e
  calc μ _ ≤ μ (⋃ T ∈ (classSet n : Finset (DTree α)), ⋃ e ∈ edgeMaps T, B T e) :=
        measure_mono hsub
    _ ≤ ∑ T ∈ (classSet n : Finset (DTree α)), ∑ e ∈ edgeMaps T, μ (B T e) := by
        refine (measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun T _ => ?_)
        exact measure_biUnion_finset_le _ _
    _ ≤ ∑ T ∈ (classSet n : Finset (DTree α)),
          (((n + 3) ^ ((n + 2) * Fintype.card α) : ℕ) : ENNReal) * ENNReal.ofReal ε := by
        refine Finset.sum_le_sum fun T hT => ?_
        have hp := classSet_paths n T hT
        calc ∑ e ∈ edgeMaps T, μ (B T e) ≤ ∑ e ∈ edgeMaps T, ENNReal.ofReal ε :=
              Finset.sum_le_sum fun e _ => hone T hT e
          _ = ((edgeMaps T).card : ENNReal) * ENNReal.ofReal ε := by
              simp only [Finset.sum_const, nsmul_eq_mul]
          _ ≤ _ := by
              refine mul_le_mul' ?_ le_rfl
              have h1 : (edgeMaps T).card ≤ (n + 3) ^ ((n + 2) * Fintype.card α) :=
                calc (edgeMaps T).card
                    ≤ (T.leaves.length + 1) ^ (T.leaves.length * Fintype.card α) := edgeMaps_card T
                  _ ≤ (n + 3) ^ (T.leaves.length * Fintype.card α) :=
                    Nat.pow_le_pow_left (by omega) _
                  _ ≤ (n + 3) ^ ((n + 2) * Fintype.card α) :=
                    Nat.pow_le_pow_right (by omega) (Nat.mul_le_mul_right _ hp)
              exact_mod_cast h1
    _ = _ := by
        rw [Finset.sum_const, nsmul_eq_mul, ← ENNReal.ofReal_natCast,
          ← ENNReal.ofReal_natCast ((n + 3) ^ ((n + 2) * Fintype.card α)),
          ← ENNReal.ofReal_mul (by positivity), ← ENNReal.ofReal_mul (by positivity)]
        congr 1
        simp only [n, ε, θ₀]
        push_cast
        ring

end DeepPTUnion

section DeepStart

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)

omit [Fintype α] in
theorem prefixOf_prefixOf_le (x : FreeMonoid α) {j k : ℕ} (h : k ≤ j) :
    prefixOf (prefixOf x j) k = prefixOf x k := by
  apply FreeMonoid.toList.injective
  simp only [prefixOf, FreeMonoid.toList_ofList, List.take_take, min_eq_left h]

/-- `goodStart_le` off each tree's `headStrings`: when `k < K` the start reads only them, and
otherwise a length-`k` prefix has at most the mass of its length-`K` prefix. -/
theorem deepStart_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.readDist (G.M.eval z.toList) r)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k K n : ℕ) {p₀ l θgs : ℝ}
    (hθ : 0 ≤ G.θ) (hp₀ : 0 < p₀) (hpmax : ∀ u, D.real {x | prefixOf x K = u} ≤ p₀) (hl : 0 ≤ l)
    (hθgs : 0 ≤ θgs) :
    μ {ω | ¬ ∀ T ∈ (classSet n : Finset (DTree α)),
        D.real (G.startOff (read · ω) k T (headStrings T k K)) ≤ θgs}
      ≤ ENNReal.ofReal ((classSet n : Finset (DTree α)).card
        * Real.exp ((n + 1) * (3 / 2 * G.θ) * (Real.exp (l * p₀) - 1) / p₀ - l * θgs)) := by
  rcases lt_or_ge k K with hkK | hKk
  · have h0 : {ω | ¬ ∀ T ∈ (classSet n : Finset (DTree α)),
        D.real (G.startOff (read · ω) k T (headStrings T k K)) ≤ θgs} = ∅ := by
      ext ω
      simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_not]
      intro T _
      have he : G.startOff (read · ω) k T (headStrings T k K) = ∅ := by
        ext x
        simp only [ReadModel.startOff, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false,
          not_and, not_exists]
        intro hk b hs _ hbZ
        obtain ⟨m, hm, rfl, -⟩ := siftCut_inr T _ b hs
        exact hbZ (mem_headStrings (by simp [prefixOf]; omega) (by simp [prefixOf]; omega) hm)
      rw [he, measureReal_empty]
      exact hθgs
    rw [h0, measure_empty]
    exact bot_le
  · have hpk : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀ := fun u =>
      (measureReal_mono (fun x (hx : prefixOf x k = u) =>
        (show prefixOf x K = prefixOf u K by rw [← hx, prefixOf_prefixOf_le x hKk]))
        (measure_ne_top _ _)).trans (hpmax _)
    refine le_trans (measure_mono fun ω hω => ?_)
      (goodStart_le G read hmeas hind hlaw D k n hθ hp₀ hpk hl)
    simp only [Set.mem_ofPred_eq, not_forall] at hω ⊢
    obtain ⟨T, hT, hlt⟩ := hω
    refine ⟨T, hT, fun h => hlt (le_trans (measureReal_mono ?_ (measure_ne_top _ _)) h)⟩
    rintro x ⟨hk, b, hs, hg, -⟩
    exact ⟨hk, b, hs, hg⟩

/-- `noiseEvent_le` for the deep event, at the largest mass of a length-`K` prefix. -/
theorem deepEvent_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.readDist (G.M.eval z.toList) r) (hθ : 0 ≤ G.θ)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (C : Level5Settings) (S K L : ℕ)
    {p₀ θg θgs θgpt l : ℝ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hp₀ : 0 < p₀)
    (hpmax : ∀ u, D.real {x | prefixOf x K = u} ≤ p₀) (hl : 0 ≤ l)
    (hφe : 0 ≤ C.edgeTraffic) (hφpt : 0 ≤ C.searchShare) (hθe : 8 * θg ≤ C.edgeUndecidedRate)
    (hθg : 2 * (3 / 2 * G.θ) ≤ θg)
    (hθgpt : 2 * ((L + 1) * (Fintype.card σ + S + 1) * (3 / 2 * G.θ)) ≤ θgpt) (hθgs : 0 ≤ θgs) :
    μ {ω | ¬ DeepEvent G D C S K θg θgs θgpt (read · ω)}
      ≤ ENNReal.ofReal ((classSet (Fintype.card σ + S) : Finset (DTree α)).card
          * ((Fintype.card σ + S + 3) ^ ((Fintype.card σ + S + 2) * Fintype.card α)
            * ((Fintype.card σ + S + 2) * Fintype.card α))
          * Real.exp (-(C.edgeTraffic * (C.edgeUndecidedRate - 8 * θg) / 4 / (2 * p₀ * (L + 1)
            * (1 + 4 * (3 / 2 * G.θ) * (Fintype.card σ + S + 1))))))
        + ENNReal.ofReal ((classSet (Fintype.card σ + S) : Finset (DTree α)).card
          * Real.exp ((Fintype.card σ + S + 1) * (3 / 2 * G.θ) * (Real.exp (l * p₀) - 1) / p₀
            - l * θgs))
        + ENNReal.ofReal ((classSet (Fintype.card σ + S) : Finset (DTree α)).card
          * (Fintype.card σ + S + 3) ^ ((Fintype.card σ + S + 2) * Fintype.card α)
          * Real.exp (-(θgpt * C.searchShare / 2 / (2 * p₀
              * (1 + 4 * ((L + 1) * (Fintype.card σ + S + 1)
            * (3 / 2 * G.θ)) * 1))))) := by
  have hθg0 : 0 ≤ θg := le_trans (by positivity) hθg
  have hθgpt0 : 0 ≤ θgpt := le_trans (by positivity) hθgpt
  have hE := deepEdge_le G read hmeas hind hlaw hθ D C.walkStart K L S hlen hp₀ hpmax
    (η := C.edgeTraffic * (C.edgeUndecidedRate - 8 * θg) / 4)
    (mul_nonneg (mul_nonneg hφe (by linarith)) (by norm_num)) hθg
  have hS := deepStart_le G read hmeas hind hlaw D C.walkStart K (Fintype.card σ + S) (θgs := θgs)
    hθ hp₀ hpmax hl hθgs
  rw [Nat.cast_add] at hS
  have hP := deepPT_le G read hmeas hind hlaw hθ D C.walkStart K L S hlen hp₀ hpmax
    (η := θgpt * C.searchShare / 2) (by positivity) hθgpt
  refine le_trans (measure_mono ?_) (le_trans (measure_union_le _ _) (add_le_add
    (le_trans (measure_union_le _ _) (add_le_add hE hS)) hP))
  intro ω hω
  simp only [Set.mem_ofPred_eq] at hω
  by_contra hno
  simp only [Set.mem_union, Set.mem_ofPred_eq, not_or, not_not] at hno
  obtain ⟨⟨he, hs⟩, hp⟩ := hno
  exact hω ⟨fun T edges hT hed e => he T hT edges hed e,
    fun T hT => hs T (inClass_mem G hT), hp⟩

end DeepStart

end OrthoDFA
