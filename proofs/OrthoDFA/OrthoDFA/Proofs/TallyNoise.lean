import OrthoDFA.Proofs.TallyClass
import OrthoDFA.Proofs.StartAtK

/-!
# The noise event's start field

The reads are independent across strings, each with its state's law. A probe's start sift reads
only its length-`k` prefix followed by the tree's midfixes, so the sifts of distinct length-`k`
prefixes read disjoint strings and are independent. The good read-states' undecided start reads
then have mass a weighted sum of independent indicators, each weight at most `p₀`, the largest
mass of a length-`k` prefix, and each indicator on with chance at most the midfixes' count times
`1.5θ`. `goodStart_holds` bounds its upper tail by the exponential moment, over the class.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- An index for each read. -/
def ARU.toFin3 : ARU → Fin 3
  | .accept => 0
  | .reject => 1
  | .undecided => 2

instance : Finite ARU :=
  Finite.of_injective ARU.toFin3 fun a b h => by cases a <;> cases b <;> simp_all [ARU.toFin3]

section Sift

theorem sift_congr_mid {cut cut' : FreeMonoid α → Option Bool} :
    ∀ (T : DTree α) (u : FreeMonoid α), (∀ m ∈ T.midfixes, cut (u * m) = cut' (u * m)) →
      T.sift cut u = T.sift cut' u
  | .leaf, _, _ => rfl
  | .node m r a, u, h => by
    have hm := h m (by simp [DTree.midfixes])
    have hr := sift_congr_mid r u fun m' hm' => h m' (by simp [DTree.midfixes, hm'])
    have ha := sift_congr_mid a u fun m' hm' => h m' (by simp [DTree.midfixes, hm'])
    simp only [DTree.sift, DTree.route] at hr ha ⊢
    rw [hm]
    rcases cut' (u * m) with _ | _ | _ <;> simp [hr, ha]

theorem sift_inr {cut : FreeMonoid α → Option Bool} :
    ∀ (T : DTree α) (u b : FreeMonoid α), T.sift cut u = .inr b →
      ∃ m ∈ T.midfixes, b = u * m ∧ cut (u * m) = none
  | .leaf, _, _, h => by simp [DTree.sift, DTree.route] at h
  | .node m r a, u, b, h => by
    simp only [DTree.sift, DTree.route] at h
    rcases hc : cut (u * m) with _ | _ | _ <;> simp only [hc] at h
    · simp only [Sum.inr.injEq] at h
      exact ⟨m, by simp [DTree.midfixes], h.symm, hc⟩
    · rcases hr : (r.route cut u).2 with q | b' <;> rw [hr] at h
      · simp at h
      · simp only [Sum.map_inr, id, Sum.inr.injEq] at h
        subst h
        obtain ⟨m', hm', h1, h2⟩ := sift_inr r u b' hr
        exact ⟨m', by simp [DTree.midfixes, hm'], h1, h2⟩
    · rcases ha : (a.route cut u).2 with q | b' <;> rw [ha] at h
      · simp at h
      · simp only [Sum.map_inr, id, Sum.inr.injEq] at h
        subst h
        obtain ⟨m', hm', h1, h2⟩ := sift_inr a u b' ha
        exact ⟨m', by simp [DTree.midfixes, hm'], h1, h2⟩

theorem midfixes_card : ∀ T : DTree α, T.midfixes.card + 1 ≤ T.paths.length
  | .leaf => by simp [DTree.midfixes, DTree.paths]
  | .node m r a => by
    classical
    have hr := midfixes_card r
    have ha := midfixes_card a
    simp only [DTree.midfixes, DTree.paths, List.length_append, List.length_map]
    have := Finset.card_insert_le m (r.midfixes ∪ a.midfixes)
    have := Finset.card_union_le r.midfixes a.midfixes
    omega

end Sift

section Blocks

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  (read : FreeMonoid α → Ω → ARU)

open scoped Classical in
/-- Functions of reads of pairwise disjoint finite sets of strings: the moment of their product
is the product of their moments. -/
theorem lintegral_prod_blocks (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    {ι : Type*} (S : ι → Finset (FreeMonoid α)) (g : ∀ u, (S u → ARU) → ENNReal) :
    ∀ U : Finset ι, (∀ u ∈ U, ∀ u' ∈ U, u ≠ u' → Disjoint (S u) (S u')) →
      ∫⁻ ω, ∏ u ∈ U, g u (fun z => read z ω) ∂μ
        = ∏ u ∈ U, ∫⁻ ω, g u (fun z => read z ω) ∂μ := by
  intro U
  induction U using Finset.induction_on with
  | empty => intro _; simp
  | insert a U ha ih =>
    intro hdisj
    rw [Finset.prod_insert ha]
    simp_rw [Finset.prod_insert ha]
    rw [← ih fun u hu u' hu' h => hdisj u (Finset.mem_insert_of_mem hu) u'
        (Finset.mem_insert_of_mem hu') h]
    set B := U.biUnion S
    have hdB : Disjoint (S a) B := by
      rw [Finset.disjoint_biUnion_right]
      intro u hu
      exact hdisj a (Finset.mem_insert_self a U) u (Finset.mem_insert_of_mem hu)
        (fun h => ha (h ▸ hu))
    have hI := hind.indepFun_finset (S a) B hdB hmeas
    set H : (B → ARU) → ENNReal := fun v =>
      ∏ u ∈ U.attach, g u.1 (fun z => v ⟨z.1, Finset.mem_biUnion.2 ⟨u.1, u.2, z.2⟩⟩)
    have hH : ∀ ω, ∏ u ∈ U, g u (fun z => read z ω) = H (fun i => read i ω) := by
      intro ω
      rw [← Finset.prod_attach U]
    have hmA : Measurable fun ω (i : S a) => read i ω := measurable_pi_lambda _ fun i => hmeas i
    have hmB : Measurable fun ω (i : B) => read i ω := measurable_pi_lambda _ fun i => hmeas i
    have hI' := hI.comp (φ := g a) (ψ := H) (measurable_of_countable _) (measurable_of_countable _)
    simp_rw [hH]
    exact lintegral_mul_eq_lintegral_mul_lintegral_of_indepFun
      (f := fun ω => g a (fun z => read z ω)) (g := fun ω => H (fun i => read i ω))
      ((measurable_of_countable (g a)).comp hmA) ((measurable_of_countable H).comp hmB) hI'

end Blocks

section Start

/-- The strings of length `k`. -/
noncomputable def lenK (k : ℕ) : Finset (FreeMonoid α) :=
  (Finset.univ : Finset (Fin k → α)).image fun v => FreeMonoid.ofList (List.ofFn v)

theorem mem_lenK {k : ℕ} {u : FreeMonoid α} : u ∈ lenK k ↔ u.toList.length = k := by
  constructor
  · intro h
    obtain ⟨v, -, rfl⟩ := Finset.mem_image.1 h
    simp
  · intro h
    refine Finset.mem_image.2 ⟨fun i => u.toList.get ⟨i, by omega⟩, Finset.mem_univ _, ?_⟩
    apply FreeMonoid.toList.injective
    simp only [FreeMonoid.toList_ofList]
    apply List.ext_get (by simp [h])
    intro i h1 h2
    simp

theorem exp_sub_one_le {w p₀ l : ℝ} (hp₀ : 0 < p₀) (hw0 : 0 ≤ w) (hw : w ≤ p₀) (hl : 0 ≤ l) :
    Real.exp (l * w) - 1 ≤ w / p₀ * (Real.exp (l * p₀) - 1) := by
  have ht0 : 0 ≤ w / p₀ := div_nonneg hw0 hp₀.le
  have ht1 : w / p₀ ≤ 1 := (div_le_one hp₀).2 hw
  have h := convexOn_exp.2 (Set.mem_univ (l * p₀)) (Set.mem_univ 0) ht0 (sub_nonneg.2 ht1)
    (by ring)
  simp only [smul_eq_mul, mul_zero, add_zero, Real.exp_zero, mul_one] at h
  have : w / p₀ * (l * p₀) = l * w := by field_simp
  rw [this] at h
  linarith

open scoped Classical in
/-- Over the class, the good read-states' undecided start reads exceed `θgs` of the probes with
chance at most the class's size times `exp(π (e^{λ p₀} − 1)/p₀ − λ θgs)`, where
`π = (n + 1)·1.5θ` bounds a start sift's chance of stopping undecided at a good read-state. -/
theorem goodStart_le {σ : Type*} [Fintype σ] (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)
    (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.dist (G.M.eval z.toList) r)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k n : ℕ) {p₀ l θgs : ℝ}
    (hθ : 0 ≤ G.θ) (hk : ∀ᵐ x ∂D, k ≤ x.toList.length) (hp₀ : 0 < p₀)
    (hpmax : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (hl : 0 ≤ l) :
    μ {ω | ¬ ∀ T ∈ (classSet n : Finset (DTree α)),
        D.real (G.startAt (read · ω) k T G.Good) ≤ θgs}
      ≤ ENNReal.ofReal ((classSet n : Finset (DTree α)).card
        * Real.exp ((n + 1) * (3 / 2 * G.θ) * (Real.exp (l * p₀) - 1) / p₀ - l * θgs)) := by
  set π : ℝ := (n + 1) * (3 / 2 * G.θ)
  have hπ0 : 0 ≤ π := by positivity
  set U := (lenK k : Finset (FreeMonoid α))
  set w : FreeMonoid α → ℝ := fun u => D.real {x | prefixOf x k = u}
  have hw0 : ∀ u, 0 ≤ w u := fun u => measureReal_nonneg
  have hwsum : ∑ u ∈ U, w u ≤ 1 := by
    rw [← measureReal_biUnion_finset (fun u _ u' _ h => Set.disjoint_left.2 fun x h1 h2 =>
      h (h1.symm.trans h2)) (fun u _ => (Set.to_countable _).measurableSet)]
    have := measureReal_mono (μ := D) (Set.subset_univ (⋃ u ∈ U, {x | prefixOf x k = u}))
    rwa [probReal_univ] at this
  have hsplit : ∀ P : FreeMonoid α → Prop, D.real {x | P (prefixOf x k)}
      = ∑ u ∈ U, w u * if P u then 1 else 0 := by
    intro P
    have hae : {x | P (prefixOf x k)} =ᵐ[D] ⋃ u ∈ U.filter P, {x | prefixOf x k = u} := by
      filter_upwards [hk] with x hx
      show (x ∈ {x | P (prefixOf x k)}) = (x ∈ ⋃ u ∈ U.filter P, {x | prefixOf x k = u})
      rw [Set.mem_iUnion₂]
      apply propext
      simp only [Set.mem_ofPred_eq, Finset.mem_filter, exists_prop]
      constructor
      · intro h
        refine ⟨prefixOf x k, ⟨mem_lenK.2 ?_, h⟩, rfl⟩
        simp [prefixOf, List.length_take, hx]
      · rintro ⟨u, ⟨-, hu⟩, rfl⟩
        exact hu
    rw [measureReal_congr hae, measureReal_biUnion_finset (fun u _ u' _ h =>
      Set.disjoint_left.2 fun x h1 h2 => h (h1.symm.trans h2))
      (fun u _ => (Set.to_countable _).measurableSet), Finset.sum_filter]
    exact Finset.sum_congr rfl fun u _ => by by_cases h : P u <;> simp [w, h]
  -- the bound for one tree
  have hone : ∀ T ∈ (classSet n : Finset (DTree α)),
      μ {ω | θgs < D.real (G.startAt (read · ω) k T G.Good)}
        ≤ ENNReal.ofReal (Real.exp (π * (Real.exp (l * p₀) - 1) / p₀ - l * θgs)) := by
    intro T hT
    set Sb : FreeMonoid α → Finset (FreeMonoid α) := fun u => T.midfixes.image (u * ·)
    set bad : FreeMonoid α → (FreeMonoid α → Option Bool) → Prop := fun u cut =>
      ∃ b, T.sift cut u = .inr b ∧ G.Good (G.M.eval b.toList)
    set cutOf : ∀ u, (Sb u → ARU) → FreeMonoid α → Option Bool := fun u v z =>
      if h : z ∈ Sb u then (v ⟨z, h⟩).cut else none
    have hcutOf : ∀ u ω, bad u (cutOf u fun z => read z ω) ↔ bad u fun z => (read z ω).cut := by
      intro u ω
      simp only [bad]
      rw [sift_congr_mid (cut' := fun z => (read z ω).cut) T u fun m hm => by
        simp only [cutOf]
        rw [dif_pos (show u * m ∈ Sb u from Finset.mem_image_of_mem (u * ·) hm)]]
    set E : FreeMonoid α → Set Ω := fun u => {ω | bad u fun z => (read z ω).cut}
    have hEm : ∀ u, MeasurableSet (E u) := by
      intro u
      have hmA : Measurable fun ω (i : Sb u) => read i ω :=
        measurable_pi_lambda _ fun i => hmeas i
      have : E u = (fun ω (i : Sb u) => read i ω) ⁻¹' {v | bad u (cutOf u v)} := by
        ext ω; simp only [E, Set.mem_preimage, Set.mem_ofPred_eq, hcutOf]
      rw [this]
      exact hmA (Set.to_countable _).measurableSet
    have hEp : ∀ u, μ.real (E u) ≤ π := by
      intro u
      have hsub : E u ⊆ ⋃ m ∈ T.midfixes.filter (fun m => G.Good (G.M.eval (u * m).toList)),
          {ω | read (u * m) ω = .undecided} := by
        rintro ω ⟨b, hb, hg⟩
        obtain ⟨m, hm, rfl, hc⟩ := sift_inr T u b hb
        refine Set.mem_biUnion (Finset.mem_filter.2 ⟨hm, hg⟩) ?_
        simp only [Set.mem_ofPred_eq]
        revert hc
        cases read (u * m) ω <;> simp [ARU.cut]
      refine (measureReal_mono hsub (measure_ne_top μ _)).trans
        ((measureReal_biUnion_finset_le _ _).trans ?_)
      have hcard : T.midfixes.card ≤ n + 1 := by
        have := midfixes_card T; have := classSet_paths n T hT; omega
      calc _ ≤ ∑ m ∈ T.midfixes.filter (fun m => G.Good (G.M.eval (u * m).toList)),
            3 / 2 * G.θ := Finset.sum_le_sum fun m hm => by
              rw [hlaw]; exact (Finset.mem_filter.1 hm).2
        _ = _ := by rw [Finset.sum_const, nsmul_eq_mul]
        _ ≤ π := by
          simp only [π]
          gcongr
          exact_mod_cast (Finset.card_filter_le _ _).trans hcard
    have hdisj : ∀ u ∈ U, ∀ u' ∈ U, u ≠ u' → Disjoint (Sb u) (Sb u') := by
      intro u hu u' hu' hne
      rw [Finset.disjoint_left]
      intro z h1 h2
      obtain ⟨m, -, rfl⟩ := Finset.mem_image.1 h1
      obtain ⟨m', -, he⟩ := Finset.mem_image.1 h2
      apply hne
      have h1 := mem_lenK.1 hu
      have h2 := mem_lenK.1 hu'
      apply FreeMonoid.toList.injective
      have := congrArg (fun z : FreeMonoid α => z.toList.take k) he
      simp only [FreeMonoid.toList_mul, List.take_left' h1, List.take_left' h2] at this
      exact this.symm
    set Z : Ω → ℝ := fun ω => ∑ u ∈ U, w u * if ω ∈ E u then 1 else 0
    have hZ : ∀ ω, D.real (G.startAt (read · ω) k T G.Good) = Z ω := by
      intro ω
      exact hsplit fun u => bad u fun z => (read z ω).cut
    set g : ∀ u, (Sb u → ARU) → ENNReal := fun u v =>
      if bad u (cutOf u v) then ENNReal.ofReal (Real.exp (l * w u)) else 1
    have hexpZ : ∀ ω, ENNReal.ofReal (Real.exp (l * Z ω)) = ∏ u ∈ U, g u fun z => read z ω := by
      intro ω
      rw [show l * Z ω = ∑ u ∈ U, l * (w u * if ω ∈ E u then 1 else 0) by
        simp only [Z, Finset.mul_sum], Real.exp_sum,
        ENNReal.ofReal_prod_of_nonneg fun u _ => (Real.exp_pos _).le]
      refine Finset.prod_congr rfl fun u _ => ?_
      simp only [g, hcutOf, E, Set.mem_ofPred_eq]
      by_cases h : bad u fun z => (read z ω).cut
      · rw [if_pos h, if_pos h, mul_one]
      · rw [if_neg h, if_neg h, mul_zero, mul_zero, Real.exp_zero, ENNReal.ofReal_one]
    have hmom : ∀ u, ∫⁻ ω, g u (fun z => read z ω) ∂μ
        ≤ ENNReal.ofReal (Real.exp (π * (Real.exp (l * w u) - 1))) := by
      intro u
      have he1 : 1 ≤ Real.exp (l * w u) := Real.one_le_exp (mul_nonneg hl (hw0 u))
      have hpt : ∀ ω, g u (fun z => read z ω)
          = 1 + (E u).indicator (fun _ => ENNReal.ofReal (Real.exp (l * w u) - 1)) ω := by
        intro ω
        simp only [g, hcutOf]
        by_cases h : bad u fun z => (read z ω).cut
        · rw [if_pos h, Set.indicator_of_mem (show ω ∈ E u from h), ENNReal.ofReal_sub _ zero_le_one,
            ENNReal.ofReal_one, add_tsub_cancel_of_le (by simpa using he1)]
        · rw [if_neg h, Set.indicator_of_notMem (show ω ∉ E u from h), add_zero]
      simp_rw [hpt]
      rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
        lintegral_indicator (hEm u), setLIntegral_const, ← ofReal_measureReal,
        ← ENNReal.ofReal_mul (by linarith), ← ENNReal.ofReal_one,
        ← ENNReal.ofReal_add zero_le_one (mul_nonneg (by linarith) measureReal_nonneg)]
      refine ENNReal.ofReal_le_ofReal ?_
      have := hEp u
      have h2 := Real.add_one_le_exp (π * (Real.exp (l * w u) - 1))
      nlinarith [measureReal_nonneg (μ := μ) (s := E u)]
    have hprod : ∫⁻ ω, ENNReal.ofReal (Real.exp (l * Z ω)) ∂μ
        ≤ ENNReal.ofReal (Real.exp (π * (Real.exp (l * p₀) - 1) / p₀)) := by
      simp_rw [hexpZ]
      rw [lintegral_prod_blocks read hmeas hind Sb g U hdisj]
      calc _ ≤ ∏ u ∈ U, ENNReal.ofReal (Real.exp (π * (Real.exp (l * w u) - 1))) :=
            Finset.prod_le_prod' fun u _ => hmom u
        _ = ENNReal.ofReal (Real.exp (∑ u ∈ U, π * (Real.exp (l * w u) - 1))) := by
            rw [Real.exp_sum, ENNReal.ofReal_prod_of_nonneg fun u _ => (Real.exp_pos _).le]
        _ ≤ _ := by
            refine ENNReal.ofReal_le_ofReal (Real.exp_le_exp.2 ?_)
            calc ∑ u ∈ U, π * (Real.exp (l * w u) - 1)
                ≤ ∑ u ∈ U, π * (w u / p₀ * (Real.exp (l * p₀) - 1)) :=
                  Finset.sum_le_sum fun u _ => mul_le_mul_of_nonneg_left
                    (exp_sub_one_le hp₀ (hw0 u) (hpmax u) hl) hπ0
              _ = π * (Real.exp (l * p₀) - 1) / p₀ * ∑ u ∈ U, w u := by
                  rw [Finset.mul_sum]; refine Finset.sum_congr rfl fun u _ => ?_; ring
              _ ≤ π * (Real.exp (l * p₀) - 1) / p₀ * 1 := by
                  have : 0 ≤ π * (Real.exp (l * p₀) - 1) / p₀ := by
                    have := Real.one_le_exp (mul_nonneg hl hp₀.le)
                    have : 0 ≤ Real.exp (l * p₀) - 1 := by linarith
                    positivity
                  exact mul_le_mul_of_nonneg_left hwsum this
              _ = _ := mul_one _
    have hZm : Measurable fun ω => ENNReal.ofReal (Real.exp (l * Z ω)) := by
      have : ∀ u, Measurable fun ω => (if ω ∈ E u then (1 : ℝ) else 0) := fun u =>
        Measurable.ite (hEm u) measurable_const measurable_const
      exact ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp (measurable_const.mul
        (Finset.measurable_sum _ fun u _ => measurable_const.mul (this u))))
    calc μ {ω | θgs < D.real (G.startAt (read · ω) k T G.Good)}
        ≤ μ {ω | ENNReal.ofReal (Real.exp (l * θgs)) ≤ ENNReal.ofReal (Real.exp (l * Z ω))} := by
          refine measure_mono fun ω hω => ?_
          simp only [Set.mem_ofPred_eq, hZ] at hω ⊢
          exact ENNReal.ofReal_le_ofReal (Real.exp_le_exp.2 (mul_le_mul_of_nonneg_left hω.le hl))
      _ ≤ (∫⁻ ω, ENNReal.ofReal (Real.exp (l * Z ω)) ∂μ) / ENNReal.ofReal (Real.exp (l * θgs)) :=
          meas_ge_le_lintegral_div hZm.aemeasurable (by simp [Real.exp_pos])
            ENNReal.ofReal_ne_top
      _ ≤ ENNReal.ofReal (Real.exp (π * (Real.exp (l * p₀) - 1) / p₀))
            / ENNReal.ofReal (Real.exp (l * θgs)) := by gcongr
      _ = _ := by
          rw [← ENNReal.ofReal_div_of_pos (Real.exp_pos _), ← Real.exp_sub]
  calc μ {ω | ¬ ∀ T ∈ (classSet n : Finset (DTree α)),
        D.real (G.startAt (read · ω) k T G.Good) ≤ θgs}
      ≤ μ (⋃ T ∈ (classSet n : Finset (DTree α)),
          {ω | θgs < D.real (G.startAt (read · ω) k T G.Good)}) := by
        refine measure_mono fun ω hω => ?_
        simp only [Set.mem_ofPred_eq, not_forall, not_le] at hω
        obtain ⟨T, hT, h⟩ := hω
        exact Set.mem_biUnion hT h
    _ ≤ ∑ T ∈ (classSet n : Finset (DTree α)),
          μ {ω | θgs < D.real (G.startAt (read · ω) k T G.Good)} := measure_biUnion_finset_le _ _
    _ ≤ ∑ _T ∈ (classSet n : Finset (DTree α)),
          ENNReal.ofReal (Real.exp (π * (Real.exp (l * p₀) - 1) / p₀ - l * θgs)) :=
        Finset.sum_le_sum hone
    _ = _ := by
        rw [Finset.sum_const, nsmul_eq_mul, ENNReal.ofReal_mul (Nat.cast_nonneg _),
          ENNReal.ofReal_natCast]

end Start

end OrthoDFA
