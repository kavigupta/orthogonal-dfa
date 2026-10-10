import OrthoDFA.Proofs.TallyQuery
import OrthoDFA.Proofs.TallyPTNoise
import OrthoDFA.Proofs.TallyRace

/-!
# The noise event's records field

A probe's chance of a record that is not true is at most `κ` times its mean undecided strings
plus `ε` times its first reads (`spurious_mean`). Over the probes' length-`k` prefixes the
field's tail is bounded as the edge field's is, at twice `κ`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

attribute [local instance] ARU.fintype

section Congr

variable (T : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α)

theorem recordBy_congrK {rd rd' : FreeMonoid α → ARU}
    (h : PrefAgreeK (fun z => (rd z).cut) (fun z => (rd' z).cut) T x k) :
    recordBy (fun z => (rd z).cut) k (T, edges) x
      = recordBy (fun z => (rd' z).cut) k (T, edges) x := by
  unfold recordBy
  simp only []
  rw [probeBy_congrK (edges := edges) h]
  rcases ho : probeBy (fun z => (rd' z).cut) T edges k x with _ | _ | _ | _ | ⟨ps, fd⟩ | _ | _
    <;> try rfl
  have hfd := (probeBy_edge T edges k x rd' ho).1
  simp only [h fd hfd.le]

theorem twinsBy_congrK {cut cut' : FreeMonoid α → Option Bool} (h : PrefAgreeK cut cut' T x k) :
    twinsBy cut T edges k x = twinsBy cut' T edges k x := by
  unfold twinsBy startHarvBy
  rw [h k le_rfl, edgeHarvBy_congrK (edges := edges) h]

theorem recordBy_econgr {e e' : Edges α} (he : EdgesInto T e) (hg : TgtAgree T e e')
    (cut : FreeMonoid α → Option Bool) :
    recordBy cut k (T, e) x = recordBy cut k (T, e') x := by
  unfold recordBy
  simp only []
  rw [probeBy_econgr (cut := cut) (k := k) (x := x) he hg]

theorem twinsBy_econgr {e e' : Edges α} (he : EdgesInto T e) (hg : TgtAgree T e e')
    (cut : FreeMonoid α → Option Bool) :
    twinsBy cut T e k x = twinsBy cut T e' k x := by
  unfold twinsBy
  rw [edgeHarvBy_econgr (cut := cut) (k := k) (x := x) he hg]

end Congr

section Count

theorem route_mid (cut : FreeMonoid α → Option Bool) :
    ∀ (t : DTree α) (u q : FreeMonoid α), q ∈ (t.route cut u).1 → ∃ m ∈ t.midfixes, q = u * m
  | .leaf, _, _, h => by simp [DTree.route] at h
  | .node m r a, u, q, h => by
    simp only [DTree.route] at h
    split at h
    · simp only [List.mem_singleton] at h
      exact ⟨m, by simp [DTree.midfixes], h⟩
    · simp only [List.mem_cons] at h
      rcases h with h | h
      · exact ⟨m, by simp [DTree.midfixes], h⟩
      · obtain ⟨m', hm', rfl⟩ := route_mid cut a u q h
        exact ⟨m', by simp [DTree.midfixes, hm'], rfl⟩
    · simp only [List.mem_cons] at h
      rcases h with h | h
      · exact ⟨m, by simp [DTree.midfixes], h⟩
      · obtain ⟨m', hm', rfl⟩ := route_mid cut r u q h
        exact ⟨m', by simp [DTree.midfixes, hm'], rfl⟩

variable (T : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) (rd : FreeMonoid α → ARU)

/-- The strings any sift of a prefix of `x` reads. -/
noncomputable def prefStrings : Finset (FreeMonoid α) :=
  (Finset.range (x.toList.length + 1)).biUnion fun j => T.midfixes.image (prefixOf x j * ·)

theorem route_mem_prefStrings {i : ℕ} {q : FreeMonoid α}
    (h : q ∈ (T.route (fun z => (rd z).cut) (prefixOf x i)).1) : q ∈ prefStrings T x := by
  obtain ⟨m, hm, rfl⟩ := route_mid _ T _ q h
  refine Finset.mem_biUnion.2 ⟨min i x.toList.length, Finset.mem_range.2 (by omega),
    Finset.mem_image.2 ⟨m, hm, ?_⟩⟩
  congr 1
  apply FreeMonoid.toList.injective
  simp only [prefixOf, FreeMonoid.toList_ofList]
  rcases le_total i x.toList.length with h | h
  · rw [min_eq_left h]
  · rw [min_eq_right h, List.take_of_length_le h, List.take_of_length_le le_rfl]

theorem prefStrings_card :
    (prefStrings T x).card ≤ (x.toList.length + 1) * T.midfixes.card := by
  unfold prefStrings
  refine Finset.card_biUnion_le.trans ?_
  calc ∑ j ∈ Finset.range (x.toList.length + 1), (T.midfixes.image (prefixOf x j * ·)).card
      ≤ ∑ _j ∈ Finset.range (x.toList.length + 1), T.midfixes.card :=
        Finset.sum_le_sum fun j _ => Finset.card_image_le
    _ = _ := by simp

theorem recordQ_firsts_le :
    ((recordQ T edges k x).firsts rd ∅).length ≤ (x.toList.length + 1) * T.midfixes.card := by
  obtain ⟨hnd, hkeys⟩ := QTree.firsts_keys_eq rd (recordQ T edges k x) ∅
  rw [← List.length_map (f := Prod.snd), ← List.toFinset_card_of_nodup hnd, hkeys,
    Finset.sdiff_empty]
  refine le_trans (Finset.card_le_card fun w hw => ?_) (prefStrings_card T x)
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 (List.mem_toFinset.1 hw)
  rw [asks_recordQ, List.mem_append] at hq
  rcases hq with hq | hq
  · exact route_mem_prefStrings T x rd (asks_probeQ T edges k x rd q hq).1
  · rcases ho : probeBy (fun z => (rd z).cut) T edges k x with _ | _ | _ | _ | ⟨ps, fd⟩ | _ | _
      <;> rw [ho] at hq <;> simp only [List.not_mem_nil] at hq
    simp only [List.mem_append, List.mem_map] at hq
    rcases hq with ⟨w', hw', rfl⟩ | ⟨w', hw', rfl⟩
    · exact route_mem_prefStrings T x rd hw'
    · exact route_mem_prefStrings T x rd hw'

end Count

section Blocks

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {read : FreeMonoid α → Ω → ARU}

theorem measurable_of_strings {β : Type*} [MeasurableSpace β] (hmeas : ∀ z, Measurable (read z))
    (S : Finset (FreeMonoid α)) (f : (FreeMonoid α → ARU) → β)
    (hf : ∀ rd rd', (∀ z ∈ S, rd z = rd' z) → f rd = f rd') :
    Measurable fun ω => f (read · ω) := by
  have : (fun ω => f (read · ω))
      = (fun v : S → ARU => f (rdOf S v)) ∘ fun ω (i : S) => read i ω := by
    funext ω
    exact hf _ _ fun z hz => by simp only [rdOf, dif_pos hz]
  rw [this]
  exact (measurable_of_countable _).comp (measurable_pi_lambda _ fun i => hmeas i)


/-- Probe quantities each read through strings sharing the probe's length-`k` prefix, the first's
mean at most `θ₀` of the second's: their `D`-weighted sums exceed twice `θ₀` by `η` with chance at
most `exp(−η / (2 p₀ B (1 + 4θ₀R)))`. -/
theorem block_tail (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {k L : ℕ} {p₀ η θ₀ B R : ℝ}
    (hp₀ : 0 < p₀)
    (hpmax : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (hη : 0 ≤ η) (hθ₀ : 0 ≤ θ₀) (hB : 0 < B)
    (hR : 0 ≤ R) (Sx : FreeMonoid α → Finset (FreeMonoid α))
    (hSx : ∀ x z, z ∈ Sx x → k ≤ x.toList.length → prefixOf z k = prefixOf x k)
    (uf tf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ)
    (huf : ∀ x rd rd', (∀ z ∈ Sx x, rd z = rd' z) → uf rd x = uf rd' x)
    (htf : ∀ x rd rd', (∀ z ∈ Sx x, rd z = rd' z) → tf rd x = tf rd' x)
    (hub : ∀ rd, ∀ x ∈ lenRange k L, 0 ≤ uf rd x ∧ uf rd x ≤ B)
    (htb : ∀ rd, ∀ x ∈ lenRange k L, 0 ≤ tf rd x ∧ tf rd x ≤ B * R)
    (hmean : ∀ x ∈ lenRange k L,
      ∫ ω, uf (read · ω) x ∂μ ≤ θ₀ * ∫ ω, tf (read · ω) x ∂μ) :
    μ {ω | 2 * θ₀ * ∑ x ∈ lenRange k L, D.real {x} * tf (read · ω) x + η
        < ∑ x ∈ lenRange k L, D.real {x} * uf (read · ω) x}
      ≤ ENNReal.ofReal (Real.exp (-(η / (2 * p₀ * B * (1 + 4 * θ₀ * R))))) := by
  set c : ℝ := 1 / (2 * p₀ * B * (1 + 4 * θ₀ * R))
  have hc : 0 < c := by positivity
  set X : Finset (FreeMonoid α) := lenRange k L
  set U : Finset (FreeMonoid α) := lenK k
  set Xu : FreeMonoid α → Finset (FreeMonoid α) := fun u => X.filter fun x => prefixOf x k = u
  set w : FreeMonoid α → ℝ := fun x => D.real {x}
  set Sb : FreeMonoid α → Finset (FreeMonoid α) := fun u => (Xu u).biUnion Sx
  have hXk : ∀ x ∈ X, k ≤ x.toList.length := fun x hx => (mem_lenRange.1 hx).1
  have hmapsto : ∀ x ∈ X, prefixOf x k ∈ U := by
    intro x hx
    refine mem_lenK.2 ?_
    simp [prefixOf, hXk x hx]
  have hsum : ∀ f : FreeMonoid α → ℝ, ∑ x ∈ X, w x * f x = ∑ u ∈ U, ∑ x ∈ Xu u, w x * f x :=
    fun f => (Finset.sum_fiberwise_of_maps_to hmapsto _).symm
  have hblk : ∀ u, ∀ x ∈ Xu u, ∀ rd rd' : FreeMonoid α → ARU, (∀ z ∈ Sb u, rd z = rd' z) →
      ∀ z ∈ Sx x, rd z = rd' z := fun u x hx rd rd' h z hz =>
    h z (Finset.mem_biUnion.2 ⟨x, hx, hz⟩)
  have hrd : ∀ u ω, ∀ z ∈ Sb u, (read z ω) = rdOf (Sb u) (fun i => read i ω) z := by
    intro u ω z hz
    simp only [rdOf, dif_pos hz]
  have hmf : ∀ (f : (FreeMonoid α → ARU) → FreeMonoid α → ℝ),
      (∀ x rd rd', (∀ z ∈ Sx x, rd z = rd' z) → f rd x = f rd' x) →
      ∀ x, Measurable fun ω => f (read · ω) x := fun f hf x =>
    measurable_of_strings hmeas (Sx x) (fun rd => f rd x) fun rd rd' h => hf x rd rd' h
  have hw0 : ∀ x, 0 ≤ w x := fun x => measureReal_nonneg
  have hwsum : ∀ u, ∑ x ∈ Xu u, w x ≤ p₀ := by
    intro u
    refine le_trans ?_ (hpmax u)
    rw [← measureReal_biUnion_finset (fun x _ x' _ h => Set.disjoint_singleton.2 h)
      (fun x _ => measurableSet_singleton x)]
    refine measureReal_mono ?_
    intro y hy
    simp only [Set.mem_iUnion, Set.mem_singleton_iff, exists_prop] at hy
    obtain ⟨x, hx, rfl⟩ := hy
    exact (Finset.mem_filter.1 hx).2
  set Y : FreeMonoid α → Ω → ℝ := fun u ω =>
    ∑ x ∈ Xu u, w x * (uf (read · ω) x - 2 * θ₀ * tf (read · ω) x)
  have hev : {ω | 2 * θ₀ * ∑ x ∈ X, w x * tf (read · ω) x + η < ∑ x ∈ X, w x * uf (read · ω) x}
      ⊆ {ω | η < ∑ u ∈ U, Y u ω} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    rw [hsum, hsum] at hω
    have : ∑ u ∈ U, Y u ω = ∑ u ∈ U, ∑ x ∈ Xu u, w x * uf (read · ω) x
        - 2 * θ₀ * ∑ u ∈ U, ∑ x ∈ Xu u, w x * tf (read · ω) x := by
      simp only [Y, Finset.mul_sum, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun u _ => Finset.sum_congr rfl fun x _ => ?_
      ring
    rw [this]
    linarith
  have hXuX : ∀ u, ∀ x ∈ Xu u, x ∈ X := fun u x hx => (Finset.mem_filter.1 hx).1
  have hint : ∀ (u : FreeMonoid α) (f : (FreeMonoid α → ARU) → FreeMonoid α → ℝ) (M : ℝ),
      (∀ x rd rd', (∀ z ∈ Sx x, rd z = rd' z) → f rd x = f rd' x) →
      (∀ rd, ∀ x ∈ X, 0 ≤ f rd x ∧ f rd x ≤ M) →
      ∫ ω, ∑ x ∈ Xu u, w x * f (read · ω) x ∂μ = ∑ x ∈ Xu u, w x * ∫ ω, f (read · ω) x ∂μ := by
    intro u f M hf hfb
    rw [integral_finset_sum _ fun x hx => (Integrable.of_bound (hmf f hf x).aestronglyMeasurable
      M (Filter.Eventually.of_forall fun ω => by
        rw [Real.norm_eq_abs, abs_of_nonneg (hfb _ x (hXuX u x hx)).1]
        exact (hfb _ x (hXuX u x hx)).2)).const_mul _]
    exact Finset.sum_congr rfl fun x _ => integral_const_mul _ _
  have hblock : ∀ u, ∫⁻ ω, ENNReal.ofReal (Real.exp (c * Y u ω)) ∂μ ≤ 1 := by
    intro u
    set Uf : Ω → ℝ := fun ω => ∑ x ∈ Xu u, w x * uf (read · ω) x
    set Nf : Ω → ℝ := fun ω => ∑ x ∈ Xu u, w x * tf (read · ω) x
    have hY : ∀ ω, c * Y u ω = c * (Uf ω - 2 * θ₀ * Nf ω) := by
      intro ω
      have : Y u ω = Uf ω - 2 * θ₀ * Nf ω := by
        simp only [Y, Uf, Nf, Finset.mul_sum, ← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun x _ => by ring
      rw [this]
    simp_rw [hY]
    have hUm : Measurable Uf := Finset.measurable_sum _ fun x _ =>
      measurable_const.mul (hmf uf huf x)
    have hNm : Measurable Nf := Finset.measurable_sum _ fun x _ =>
      measurable_const.mul (hmf tf htf x)
    refine mgf_real Uf Nf hUm hNm hθ₀ (by positivity : 0 ≤ p₀ * B) hR hc ?_ ?_ ?_ ?_
    · simp only [c]; field_simp; linarith
    · intro ω
      constructor
      · exact Finset.sum_nonneg fun x hx => mul_nonneg (hw0 x) (hub _ x (hXuX u x hx)).1
      · calc Uf ω ≤ ∑ x ∈ Xu u, w x * B := Finset.sum_le_sum fun x hx =>
              mul_le_mul_of_nonneg_left (hub _ x (hXuX u x hx)).2 (hw0 x)
          _ = (∑ x ∈ Xu u, w x) * B := by rw [Finset.sum_mul]
          _ ≤ p₀ * B := mul_le_mul_of_nonneg_right (hwsum u) hB.le
    · intro ω
      constructor
      · exact Finset.sum_nonneg fun x hx => mul_nonneg (hw0 x) (htb _ x (hXuX u x hx)).1
      · calc Nf ω ≤ ∑ x ∈ Xu u, w x * (B * R) := Finset.sum_le_sum fun x hx =>
              mul_le_mul_of_nonneg_left (htb _ x (hXuX u x hx)).2 (hw0 x)
          _ = (∑ x ∈ Xu u, w x) * (B * R) := by rw [Finset.sum_mul]
          _ ≤ p₀ * (B * R) := mul_le_mul_of_nonneg_right (hwsum u) (by positivity)
          _ = p₀ * B * R := by ring
    · rw [hint u uf B huf hub, hint u tf (B * R) htf htb, Finset.mul_sum]
      exact Finset.sum_le_sum fun x hx => by
        have := hmean x (hXuX u x hx)
        nlinarith [hw0 x]
  have hdisj : ∀ u ∈ U, ∀ u' ∈ U, u ≠ u' → Disjoint (Sb u) (Sb u') := by
    intro u _ u' _ hne
    rw [Finset.disjoint_left]
    intro z h1 h2
    apply hne
    have key : ∀ v, z ∈ Sb v → prefixOf z k = v := by
      intro v hz
      obtain ⟨x, hx, hz⟩ := Finset.mem_biUnion.1 hz
      obtain ⟨hxX, hxv⟩ := Finset.mem_filter.1 hx
      rw [← hxv]
      exact hSx x z hz (hXk x hxX)
    rw [← key u h1, key u' h2]
  set gb : ∀ u, (Sb u → ARU) → ENNReal := fun u v => ENNReal.ofReal (Real.exp (c *
    ∑ x ∈ Xu u, w x * (uf (rdOf (Sb u) v) x - 2 * θ₀ * tf (rdOf (Sb u) v) x)))
  have hgb : ∀ u, ∀ ω, gb u (fun i => read i ω) = ENNReal.ofReal (Real.exp (c * Y u ω)) := by
    intro u ω
    simp only [gb, Y]
    congr 3
    refine Finset.sum_congr rfl fun x hx => ?_
    rw [huf x _ _ (hblk u x hx _ _ fun z hz => hrd u ω z hz),
      htf x _ _ (hblk u x hx _ _ fun z hz => hrd u ω z hz)]
  have hprod : ∫⁻ ω, ENNReal.ofReal (Real.exp (c * ∑ u ∈ U, Y u ω)) ∂μ ≤ 1 := by
    have he : ∀ ω, ENNReal.ofReal (Real.exp (c * ∑ u ∈ U, Y u ω))
        = ∏ u ∈ U, gb u (fun i => read i ω) := by
      intro ω
      rw [Finset.mul_sum, Real.exp_sum, ENNReal.ofReal_prod_of_nonneg fun u _ =>
        (Real.exp_pos _).le]
      exact Finset.prod_congr rfl fun u _ => (hgb u ω).symm
    simp_rw [he]
    rw [lintegral_prod_blocks read hmeas hind Sb gb U hdisj]
    calc ∏ u ∈ U, ∫⁻ ω, gb u (fun i => read i ω) ∂μ ≤ ∏ _u ∈ U, (1 : ENNReal) := by
          refine Finset.prod_le_prod' fun u _ => ?_
          simp_rw [hgb]
          exact hblock u
      _ = 1 := Finset.prod_const_one
  have hYm : Measurable fun ω => ENNReal.ofReal (Real.exp (c * ∑ u ∈ U, Y u ω)) := by
    refine ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp (measurable_const.mul
      (Finset.measurable_sum _ fun u _ => Finset.measurable_sum _ fun x _ =>
        measurable_const.mul ((hmf uf huf x).sub (measurable_const.mul (hmf tf htf x))))))
  calc μ _ ≤ μ {ω | η < ∑ u ∈ U, Y u ω} := measure_mono hev
    _ ≤ μ {ω | ENNReal.ofReal (Real.exp (c * η))
          ≤ ENNReal.ofReal (Real.exp (c * ∑ u ∈ U, Y u ω))} := by
        refine measure_mono fun ω hω => ?_
        simp only [Set.mem_setOf_eq] at hω ⊢
        exact ENNReal.ofReal_le_ofReal (Real.exp_le_exp.2
          (mul_le_mul_of_nonneg_left hω.le hc.le))
    _ ≤ (∫⁻ ω, ENNReal.ofReal (Real.exp (c * ∑ u ∈ U, Y u ω)) ∂μ)
          / ENNReal.ofReal (Real.exp (c * η)) :=
        meas_ge_le_lintegral_div hYm.aemeasurable (by simp [Real.exp_pos]) ENNReal.ofReal_ne_top
    _ ≤ 1 / ENNReal.ofReal (Real.exp (c * η)) := by gcongr
    _ = _ := by
        rw [← ENNReal.ofReal_one, ← ENNReal.ofReal_div_of_pos (Real.exp_pos _), one_div,
          ← Real.exp_neg]
        congr 2
        simp only [c]
        field_simp

end Blocks

section Mean

variable {σ : Type*} (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] {read : FreeMonoid α → Ω → ARU}

theorem untrue_iff_congrK {T : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {rd rd' : FreeMonoid α → ARU}
    (h : PrefAgreeK (fun z => (rd z).cut) (fun z => (rd' z).cut) T x k) :
    x ∈ G.untrueAt rd k T edges ↔ x ∈ G.untrueAt rd' k T edges := by
  simp only [ReadModel.untrueAt, Set.mem_setOf_eq, recordBy_congrK T edges k x h]

open scoped Classical in
/-- The records field in mean at one probe, over real expectations. -/
theorem untrue_mean_real (hmeas : ∀ w, Measurable (read w)) (hind : iIndepFun read μ)
    (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.dist (G.M.eval w.toList) r) (hκ : 0 ≤ G.κ)
    (hε : 0 ≤ G.ε) (T : DTree α) (edges : Edges α) (he : EdgesInto T edges) {k : ℕ}
    {x : FreeMonoid α} (hkx : k ≤ x.toList.length) :
    μ.real {ω | x ∈ G.untrueAt (read · ω) k T edges}
      ≤ G.κ * ∫ ω, (twinsBy (fun z => (read z ω).cut) T edges k x : ℝ) ∂μ
        + G.ε * ((x.toList.length + 1) * T.midfixes.card) := by
  have hagr : ∀ rd rd' : FreeMonoid α → ARU, (∀ z ∈ probeStrings T k x, rd z = rd' z) →
      PrefAgreeK (fun z => (rd z).cut) (fun z => (rd' z).cut) T x k := fun rd rd' h =>
    prefAgreeK_of_strings hkx fun z hz => by simp only [h z hz]
  have hmU : MeasurableSet {ω | x ∈ G.untrueAt (read · ω) k T edges} := by
    have := measurable_of_strings hmeas (probeStrings T k x)
      (fun rd => decide (x ∈ G.untrueAt rd k T edges)) fun rd rd' h => by
        simp only [untrue_iff_congrK G (hagr rd rd' h)]
    convert this (measurableSet_singleton true) using 1
    ext ω; simp
  have hmN : Measurable fun ω => (twinsBy (fun z => (read z ω).cut) T edges k x : ℝ) :=
    measurable_of_strings hmeas (probeStrings T k x)
      (fun rd => (twinsBy (fun z => (rd z).cut) T edges k x : ℝ)) fun rd rd' h => by
        simp only [twinsBy_congrK T edges k x (hagr rd rd' h)]
  have hNi : Integrable (fun ω => (twinsBy (fun z => (read z ω).cut) T edges k x : ℝ)) μ :=
    Integrable.of_bound hmN.aestronglyMeasurable (x.toList.length + 1)
      (Filter.Eventually.of_forall fun ω => by
        rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]
        exact_mod_cast twinsBy_le _ T edges k x)
  have h := spurious_mean G T edges k x he hmeas hind hlaw hκ
  have hl : ∫⁻ ω, (G.untrueAt (fun y => read y ω) k T edges).indicator 1 x ∂μ
      = μ {ω | x ∈ G.untrueAt (read · ω) k T edges} := by
    rw [← lintegral_indicator_one hmU]
    congr 1
  have hN : ∫⁻ ω, (twinsBy (fun z => (read z ω).cut) T edges k x : ENNReal) ∂μ
      = ENNReal.ofReal (∫ ω, (twinsBy (fun z => (read z ω).cut) T edges k x : ℝ) ∂μ) := by
    rw [ofReal_integral_eq_lintegral_ofReal hNi
      (Filter.Eventually.of_forall fun ω => Nat.cast_nonneg _)]
    congr 1; funext ω; rw [ENNReal.ofReal_natCast]
  have hF : ∫⁻ ω, (((recordQ T edges k x).firsts (fun y => read y ω) ∅).length : ENNReal) ∂μ
      ≤ (((x.toList.length + 1) * T.midfixes.card : ℕ) : ENNReal) := by
    calc _ ≤ ∫⁻ _ω, (((x.toList.length + 1) * T.midfixes.card : ℕ) : ENNReal) ∂μ :=
          lintegral_mono fun ω =>
            (Nat.cast_le (α := ENNReal)).2 (recordQ_firsts_le T edges k x (fun y => read y ω))
      _ = _ := by simp
  rw [hl, hN] at h
  have hI0 : 0 ≤ ∫ ω, (twinsBy (fun z => (read z ω).cut) T edges k x : ℝ) ∂μ :=
    integral_nonneg fun ω => Nat.cast_nonneg _
  refine ENNReal.toReal_le_of_le_ofReal (by positivity) (h.trans ?_)
  have hF' : ∫⁻ ω, (((recordQ T edges k x).firsts (fun y => read y ω) ∅).length : ENNReal) ∂μ
      ≤ ENNReal.ofReal ((x.toList.length + 1) * T.midfixes.card) := by
    refine hF.trans_eq ?_
    rw [← ENNReal.ofReal_natCast]
    push_cast
    rfl
  rw [ENNReal.ofReal_add (by positivity) (by positivity), ENNReal.ofReal_mul hκ,
    ENNReal.ofReal_mul hε]
  exact add_le_add le_rfl (mul_le_mul_of_nonneg_left hF' zero_le)

end Mean

section SpurOne

variable {σ : Type*} (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] {read : FreeMonoid α → Ω → ARU}

open scoped Classical in
theorem measureReal_eq_sum_lenRange (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    {k L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hk : ∀ᵐ x ∂D, k ≤ x.toList.length)
    (S : Set (FreeMonoid α)) :
    D.real S = ∑ x ∈ lenRange k L, D.real {x} * if x ∈ S then 1 else 0 := by
  have h1 : D.real S = ∫ x, (if x ∈ S then (1 : ℝ) else 0) ∂D := by
    rw [← integral_indicator_one (Set.to_countable S).measurableSet]
    congr 1
  rw [h1, integral_eq_sum_lenRange D hlen hk _ fun x _ => by split_ifs <;> norm_num <;> positivity]

theorem probeStrings_prefix {T : DTree α} {k : ℕ} {x z : FreeMonoid α}
    (hz : z ∈ probeStrings T k x) : prefixOf z k = prefixOf x k := by
  obtain ⟨j, hj, hz⟩ := Finset.mem_biUnion.1 hz
  obtain ⟨m, -, rfl⟩ := Finset.mem_image.1 hz
  have hjk := (Finset.mem_Icc.1 hj).1
  have hjx := (Finset.mem_Icc.1 hj).2
  apply FreeMonoid.toList.injective
  simp only [prefixOf, FreeMonoid.toList_ofList, FreeMonoid.toList_mul]
  rw [List.take_append_of_le_length (by simp; omega), List.take_take, min_eq_left hjk]

open scoped Classical in
/-- At one tree and edge map, records that are not true exceed twice `κ` times the undecided
strings and twice `ε` times the most first reads a probe makes by `η` with chance at most this. -/
theorem spurious_one (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.dist (G.M.eval z.toList) r) (hκ : 0 < G.κ)
    (hε : 0 ≤ G.ε) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {k L : ℕ} {p₀ η : ℝ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hk : ∀ᵐ x ∂D, k ≤ x.toList.length)
    (hp₀ : 0 < p₀) (hpmax : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (hη : 0 ≤ η)
    (T : DTree α) (edges : Edges α) (he : EdgesInto T edges) :
    μ {ω | 2 * G.κ * ∫ x, (twinsBy (fun z => (read z ω).cut) T edges k x : ℝ) ∂D
          + 2 * (G.ε * ((L + 1) * T.midfixes.card)) + η
        < D.real (G.untrueAt (read · ω) k T edges)}
      ≤ ENNReal.ofReal (Real.exp (-(η / (2 * p₀ * (1 + 4 * (G.κ * (L + 1)
          + G.ε * ((L + 1) * T.midfixes.card))))))) := by
  set s : ℝ := G.ε * ((L + 1) * T.midfixes.card) / G.κ
  have hs : 0 ≤ s := by positivity
  have hκs : G.κ * s = G.ε * ((L + 1) * T.midfixes.card) := by
    simp only [s]; field_simp
  set uf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    if k ≤ x.toList.length ∧ x ∈ G.untrueAt rd k T edges then 1 else 0
  set tf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    if k ≤ x.toList.length then (twinsBy (fun z => (rd z).cut) T edges k x : ℝ) + s else 0
  have hagr : ∀ x, k ≤ x.toList.length → ∀ rd rd' : FreeMonoid α → ARU,
      (∀ z ∈ probeStrings T k x, rd z = rd' z) →
      PrefAgreeK (fun z => (rd z).cut) (fun z => (rd' z).cut) T x k := fun x hkx rd rd' h =>
    prefAgreeK_of_strings hkx fun z hz => by simp only [h z hz]
  have huf : ∀ x rd rd', (∀ z ∈ probeStrings T k x, rd z = rd' z) → uf rd x = uf rd' x := by
    intro x rd rd' h
    by_cases hkx : k ≤ x.toList.length
    · simp only [uf, untrue_iff_congrK G (hagr x hkx rd rd' h)]
    · simp only [uf, hkx, false_and, if_false]
  have htf : ∀ x rd rd', (∀ z ∈ probeStrings T k x, rd z = rd' z) → tf rd x = tf rd' x := by
    intro x rd rd' h
    by_cases hkx : k ≤ x.toList.length
    · simp only [tf, if_pos hkx, twinsBy_congrK T edges k x (hagr x hkx rd rd' h)]
    · simp only [tf, if_neg hkx]
  have hub : ∀ rd, ∀ x ∈ lenRange k L, 0 ≤ uf rd x ∧ uf rd x ≤ 1 := by
    intro rd x _
    simp only [uf]; split_ifs <;> norm_num
  have htb : ∀ rd, ∀ x ∈ lenRange k L, 0 ≤ tf rd x ∧ tf rd x ≤ 1 * (L + 1 + s) := by
    intro rd x hx
    obtain ⟨hkx, hxL⟩ := mem_lenRange.1 hx
    simp only [tf, if_pos hkx, one_mul]
    have h1 := twinsBy_le (fun z => (rd z).cut) T edges k x
    have h2 : ((twinsBy (fun z => (rd z).cut) T edges k x : ℕ) : ℝ) ≤ L + 1 := by
      exact_mod_cast h1.trans (by omega)
    constructor <;> linarith [(Nat.cast_nonneg _ : (0 : ℝ) ≤ twinsBy (fun z => (rd z).cut) T edges k x)]
  have hmean : ∀ x ∈ lenRange k L,
      ∫ ω, uf (read · ω) x ∂μ ≤ G.κ * ∫ ω, tf (read · ω) x ∂μ := by
    intro x hx
    obtain ⟨hkx, hxL⟩ := mem_lenRange.1 hx
    have hmU : MeasurableSet {ω | x ∈ G.untrueAt (read · ω) k T edges} := by
      have := measurable_of_strings hmeas (probeStrings T k x)
        (fun rd => decide (x ∈ G.untrueAt rd k T edges)) fun rd rd' h => by
          simp only [untrue_iff_congrK G (hagr x hkx rd rd' h)]
      convert this (measurableSet_singleton true) using 1
      ext ω; simp
    have hmN : Measurable fun ω => (twinsBy (fun z => (read z ω).cut) T edges k x : ℝ) :=
      measurable_of_strings hmeas (probeStrings T k x)
        (fun rd => (twinsBy (fun z => (rd z).cut) T edges k x : ℝ)) fun rd rd' h => by
          simp only [twinsBy_congrK T edges k x (hagr x hkx rd rd' h)]
    have hNi : Integrable (fun ω => (twinsBy (fun z => (read z ω).cut) T edges k x : ℝ)) μ :=
      Integrable.of_bound hmN.aestronglyMeasurable (x.toList.length + 1)
        (Filter.Eventually.of_forall fun ω => by
          rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]
          exact_mod_cast twinsBy_le _ T edges k x)
    have hU : ∫ ω, uf (read · ω) x ∂μ = μ.real {ω | x ∈ G.untrueAt (read · ω) k T edges} := by
      simp only [uf, hkx, true_and]
      rw [← integral_indicator_one hmU]
      congr 1
    have hN : ∫ ω, tf (read · ω) x ∂μ
        = ∫ ω, (twinsBy (fun z => (read z ω).cut) T edges k x : ℝ) ∂μ + s := by
      simp only [tf, if_pos hkx]
      rw [integral_add hNi (integrable_const s), integral_const, probReal_univ, one_smul]
    rw [hU, hN]
    have h1 := untrue_mean_real G hmeas hind hlaw hκ.le hε T edges he hkx
    have h2 : G.ε * ((x.toList.length + 1) * T.midfixes.card)
        ≤ G.ε * ((L + 1) * T.midfixes.card) := by
      have : (x.toList.length : ℝ) ≤ L := Nat.cast_le.2 hxL
      gcongr
    nlinarith
  have hTB := block_tail (μ := μ) (read := read) hmeas hind D (L := L) hp₀ hpmax hη hκ.le
    one_pos (by positivity : (0 : ℝ) ≤ L + 1 + s) (probeStrings T k ·)
    (fun x z hz _ => probeStrings_prefix hz) uf tf huf htf hub htb hmean
  have hw1 : ∑ x ∈ lenRange k L, D.real {x} = 1 := by
    have := measureReal_eq_sum_lenRange D hlen hk Set.univ
    simpa using this.symm
  have hUs : ∀ ω, ∑ x ∈ lenRange k L, D.real {x} * uf (read · ω) x
      = D.real (G.untrueAt (read · ω) k T edges) := by
    intro ω
    rw [measureReal_eq_sum_lenRange D hlen hk]
    refine Finset.sum_congr rfl fun x hx => ?_
    simp only [uf, (mem_lenRange.1 hx).1, true_and]
  have hNs : ∀ ω, ∑ x ∈ lenRange k L, D.real {x} * tf (read · ω) x
      = ∫ x, (twinsBy (fun z => (read z ω).cut) T edges k x : ℝ) ∂D + s := by
    intro ω
    rw [integral_eq_sum_lenRange D hlen hk _ fun x hx => by
      rw [abs_of_nonneg (Nat.cast_nonneg _)]
      have := twinsBy_le (fun z => (read z ω).cut) T edges k x
      exact_mod_cast this.trans (by omega)]
    rw [← mul_one s, ← hw1, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun x hx => ?_
    simp only [tf, (mem_lenRange.1 hx).1, if_true]
    ring
  calc μ _ ≤ μ {ω | 2 * G.κ * ∑ x ∈ lenRange k L, D.real {x} * tf (read · ω) x + η
        < ∑ x ∈ lenRange k L, D.real {x} * uf (read · ω) x} := by
        refine measure_mono fun ω hω => ?_
        simp only [Set.mem_setOf_eq] at hω ⊢
        rw [hUs, hNs]
        nlinarith
    _ ≤ _ := hTB
    _ = _ := by
        congr 3
        rw [mul_one, ← hκs]
        ring

end SpurOne

section SpurUnion

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)

open scoped Classical in
/-- The records field's tail over the class and its edge maps, at `κs ≥ 2κ` and
`ρ ≥ 2ε (L + 1)(|Q| + S + 1)`. -/
theorem spurious_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.dist (G.M.eval z.toList) r) (hκ : 0 < G.κ)
    (hε : 0 ≤ G.ε) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k L S : ℕ)
    {p₀ ρ κs : ℝ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hk : ∀ᵐ x ∂D, k ≤ x.toList.length)
    (hp₀ : 0 < p₀) (hpmax : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (hκs : 2 * G.κ ≤ κs)
    (hρ : 2 * (G.ε * ((L + 1) * (Fintype.card σ + S + 1))) ≤ ρ) :
    μ {ω | ¬ ∀ T edges, G.InClass S T → EdgesInto T edges →
        D.real (G.untrueAt (read · ω) k T edges)
          ≤ κs * ∫ x, (twinsBy (fun z => (read z ω).cut) T edges k x : ℝ) ∂D + ρ}
      ≤ ENNReal.ofReal ((classSet (Fintype.card σ + S) : Finset (DTree α)).card
        * (Fintype.card σ + S + 3) ^ ((Fintype.card σ + S + 2) * Fintype.card α)
        * Real.exp (-((ρ - 2 * (G.ε * ((L + 1) * (Fintype.card σ + S + 1))))
          / (2 * p₀ * (1 + 4 * (G.κ * (L + 1) + G.ε * ((L + 1) * (Fintype.card σ + S + 1)))))))) := by
  set n := Fintype.card σ + S
  set η : ℝ := ρ - 2 * (G.ε * ((L + 1) * (n + 1)))
  have hη : 0 ≤ η := by simp only [η, n]; push_cast; linarith
  set ε' : ℝ := Real.exp (-(η / (2 * p₀ * (1 + 4 * (G.κ * (L + 1) + G.ε * ((L + 1) * (n + 1)))))))
  set B : DTree α → Edges α → Set Ω := fun T e => {ω |
    2 * G.κ * ∫ x, (twinsBy (fun z => (read z ω).cut) T e k x : ℝ) ∂D
      + 2 * (G.ε * ((L + 1) * T.midfixes.card)) + η < D.real (G.untrueAt (read · ω) k T e)}
  have hmid : ∀ T ∈ (classSet n : Finset (DTree α)), (T.midfixes.card : ℝ) ≤ n + 1 := by
    intro T hT
    have := midfixes_card T
    have := classSet_paths n T hT
    exact_mod_cast (by omega : T.midfixes.card ≤ n + 1)
  have hsub : {ω | ¬ ∀ T edges, G.InClass S T → EdgesInto T edges →
      D.real (G.untrueAt (read · ω) k T edges)
        ≤ κs * ∫ x, (twinsBy (fun z => (read z ω).cut) T edges k x : ℝ) ∂D + ρ}
      ⊆ ⋃ T ∈ (classSet n : Finset (DTree α)), ⋃ e ∈ (edgeMaps T).filter (EdgesInto T ·),
          B T e := by
    intro ω hω
    simp only [Set.mem_setOf_eq, not_forall, not_le] at hω
    obtain ⟨T, edges, hT, he, hlt⟩ := hω
    obtain ⟨e', he'm, hagr, he'⟩ := exists_edgeMaps he
    have hU : G.untrueAt (read · ω) k T edges = G.untrueAt (read · ω) k T e' := by
      ext x
      simp only [ReadModel.untrueAt, Set.mem_setOf_eq,
        recordBy_econgr T k x he hagr (fun z => (read z ω).cut)]
    have hN : ∀ x, twinsBy (fun z => (read z ω).cut) T edges k x
        = twinsBy (fun z => (read z ω).cut) T e' k x := fun x =>
      twinsBy_econgr T k x he hagr _
    have hTn := inClass_mem G hT
    refine Set.mem_biUnion hTn (Set.mem_biUnion (Finset.mem_filter.2 ⟨he'm, he'⟩) ?_)
    simp only [B, Set.mem_setOf_eq]
    simp only [hU, hN] at hlt
    have hI : 0 ≤ ∫ x, (twinsBy (fun z => (read z ω).cut) T e' k x : ℝ) ∂D :=
      integral_nonneg fun x => Nat.cast_nonneg _
    have hm := hmid T hTn
    have : G.ε * ((L + 1) * T.midfixes.card) ≤ G.ε * ((L + 1) * (n + 1)) := by gcongr
    simp only [η]
    nlinarith [mul_le_mul_of_nonneg_right hκs hI]
  have hone : ∀ T ∈ (classSet n : Finset (DTree α)), ∀ e ∈ (edgeMaps T).filter (EdgesInto T ·),
      μ (B T e) ≤ ENNReal.ofReal ε' := by
    intro T hT e he
    refine (spurious_one G hmeas hind hlaw hκ hε D hlen hk hp₀ hpmax hη T e
      (Finset.mem_filter.1 he).2).trans (ENNReal.ofReal_le_ofReal ?_)
    simp only [ε']
    gcongr
    exact hmid T hT
  calc μ _ ≤ μ (⋃ T ∈ (classSet n : Finset (DTree α)),
        ⋃ e ∈ (edgeMaps T).filter (EdgesInto T ·), B T e) := measure_mono hsub
    _ ≤ ∑ T ∈ (classSet n : Finset (DTree α)), ∑ e ∈ (edgeMaps T).filter (EdgesInto T ·),
          μ (B T e) := by
        refine (measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun T _ => ?_)
        exact measure_biUnion_finset_le _ _
    _ ≤ ∑ T ∈ (classSet n : Finset (DTree α)),
          (((n + 3) ^ ((n + 2) * Fintype.card α) : ℕ) : ENNReal) * ENNReal.ofReal ε' := by
        refine Finset.sum_le_sum fun T hT => ?_
        have hp := classSet_paths n T hT
        calc ∑ e ∈ (edgeMaps T).filter (EdgesInto T ·), μ (B T e)
            ≤ ∑ e ∈ (edgeMaps T).filter (EdgesInto T ·), ENNReal.ofReal ε' :=
              Finset.sum_le_sum fun e he => hone T hT e he
          _ = (((edgeMaps T).filter (EdgesInto T ·)).card : ENNReal) * ENNReal.ofReal ε' := by
              simp only [Finset.sum_const, nsmul_eq_mul]
          _ ≤ _ := by
              refine mul_le_mul' ?_ le_rfl
              have h1 : ((edgeMaps T).filter (EdgesInto T ·)).card
                  ≤ (n + 3) ^ ((n + 2) * Fintype.card α) :=
                calc ((edgeMaps T).filter (EdgesInto T ·)).card ≤ (edgeMaps T).card :=
                      Finset.card_filter_le _ _
                  _ ≤ (T.paths.length + 1) ^ (T.paths.length * Fintype.card α) := edgeMaps_card T
                  _ ≤ (n + 3) ^ (T.paths.length * Fintype.card α) :=
                    Nat.pow_le_pow_left (by omega) _
                  _ ≤ (n + 3) ^ ((n + 2) * Fintype.card α) :=
                    Nat.pow_le_pow_right (by omega) (Nat.mul_le_mul_right _ hp)
              exact_mod_cast h1
    _ = _ := by
        rw [Finset.sum_const, nsmul_eq_mul, ← ENNReal.ofReal_natCast,
          ← ENNReal.ofReal_natCast ((n + 3) ^ ((n + 2) * Fintype.card α)),
          ← ENNReal.ofReal_mul (by positivity), ← ENNReal.ofReal_mul (by positivity)]
        congr 1
        simp only [n, ε', η]
        push_cast
        ring

end SpurUnion

end OrthoDFA
