import OrthoDFA.Proofs.Level4Firsts
import OrthoDFA.Proofs.Level4PTNoise
import OrthoDFA.Proofs.Level4Race

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

variable (T : DTree α) (edges : EdgeTable α) (k : ℕ) (x : FreeMonoid α)

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

theorem recordBy_econgr {e e' : EdgeTable α} (he : EdgesInto T e) (hg : TgtAgree T e e')
    (cut : FreeMonoid α → Option Bool) :
    recordBy cut k (T, e) x = recordBy cut k (T, e') x := by
  unfold recordBy
  simp only []
  rw [probeBy_econgr (cut := cut) (k := k) (x := x) he hg]

end Congr

section Count

variable (T : DTree α) (edges : EdgeTable α) (k : ℕ) (x : FreeMonoid α) (rd : FreeMonoid α → ARU)

theorem recordQ_firsts_le :
    ((recordQ T edges k x).firsts rd ∅).length ≤ (x.toList.length + 1) * T.midfixes.card := by
  obtain ⟨hnd, hkeys⟩ := QTree.firsts_keys_eq rd (recordQ T edges k x) ∅
  rw [← List.length_map (f := Prod.snd), ← List.toFinset_card_of_nodup hnd, hkeys,
    Finset.sdiff_empty]
  refine le_trans (Finset.card_le_card fun w hw => ?_) (prefStrings_card T x)
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 (List.mem_toFinset.1 hw)
  rw [asks_recordQ, List.mem_append] at hq
  rcases hq with hq | hq
  · exact route_mem_prefStrings (asks_probeQuery T edges k x rd q hq).1
  · rcases ho : probeBy (fun z => (rd z).cut) T edges k x with _ | _ | _ | _ | ⟨ps, fd⟩ | _ | _
      <;> rw [ho] at hq <;> simp only [List.not_mem_nil] at hq
    simp only [List.mem_append, List.mem_map] at hq
    rcases hq with ⟨w', hw', rfl⟩ | ⟨w', hw', rfl⟩
    · exact route_mem_prefStrings hw'
    · exact route_mem_prefStrings hw'

end Count

section Mean

variable {σ : Type*} (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] {read : FreeMonoid α → Ω → ARU}

theorem untrue_iff_congrK {T : DTree α} {edges : EdgeTable α} {k : ℕ} {x : FreeMonoid α}
    {rd rd' : FreeMonoid α → ARU}
    (h : PrefAgreeK (fun z => (rd z).cut) (fun z => (rd' z).cut) T x k) :
    x ∈ G.untrueAt rd k T edges ↔ x ∈ G.untrueAt rd' k T edges := by
  simp only [ReadModel.untrueAt, Set.mem_setOf_eq, recordBy_congrK T edges k x h]

open scoped Classical in
/-- The records field in mean at one probe, over real expectations. -/
theorem untrue_mean_real (hmeas : ∀ w, Measurable (read w)) (hind : iIndepFun read μ)
    (hlaw : ∀ w r, μ.real {ω | read w ω = r} = G.readDist (G.M.eval w.toList) r) (hκ : 0 ≤ G.κ)
    (hε : 0 ≤ G.ε) (T : DTree α) (edges : EdgeTable α) (he : EdgesInto T edges) {k : ℕ}
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
        rw [twinsBy_congr_strings (edges := edges) hkx fun z hz => by rw [h z hz]]
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
/-- At one tree and edge map, records that are not true exceed twice `κ` times the undecided
strings and twice `ε` times the most first reads a probe makes by `η` with chance at most this. -/
theorem spurious_one (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.readDist (G.M.eval z.toList) r) (hκ : 0 < G.κ)
    (hε : 0 ≤ G.ε) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {k L : ℕ} {p₀ η : ℝ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hp₀ : 0 < p₀) (hpmax : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (hη : 0 ≤ η)
    (T : DTree α) (edges : EdgeTable α) (he : EdgesInto T edges) :
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
    (twinsBy (fun z => (rd z).cut) T edges k x : ℝ) + s
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
    · simp only [tf, twinsBy_congr_strings (edges := edges) (T := T) (x := x)
        (cut := fun z => (rd z).cut) (cut' := fun z => (rd' z).cut) hkx
        fun z hz => by rw [h z hz]]
    · have hs' : x.toList.length < k := by omega
      simp only [tf, twinsBy_short _ T edges hs']
  have hub : ∀ rd, ∀ x ∈ lenRange 0 L, 0 ≤ uf rd x ∧ uf rd x ≤ 1 := by
    intro rd x _
    simp only [uf]; split_ifs <;> norm_num
  have htb : ∀ rd, ∀ x ∈ lenRange 0 L, 0 ≤ tf rd x ∧ tf rd x ≤ 1 * (L + 1 + s) := by
    intro rd x hx
    obtain ⟨-, hxL⟩ := mem_lenRange.1 hx
    simp only [tf, one_mul]
    have h1 := twinsBy_le (fun z => (rd z).cut) T edges k x
    have h2 : ((twinsBy (fun z => (rd z).cut) T edges k x : ℕ) : ℝ) ≤ L + 1 := by
      exact_mod_cast h1.trans (by omega)
    have h3 : (0 : ℝ) ≤ twinsBy (fun z => (rd z).cut) T edges k x := Nat.cast_nonneg _
    constructor <;> linarith
  have hmean : ∀ x ∈ lenRange 0 L,
      ∫ ω, uf (read · ω) x ∂μ ≤ G.κ * ∫ ω, tf (read · ω) x ∂μ := by
    intro x hx
    obtain ⟨-, hxL⟩ := mem_lenRange.1 hx
    by_cases hkx : k ≤ x.toList.length
    swap
    · have hs' : x.toList.length < k := by omega
      simp only [uf, hkx, false_and, if_false, integral_zero, tf, twinsBy_short _ T edges hs',
        Nat.cast_zero, zero_add, integral_const, probReal_univ, one_smul]
      positivity
    have hmU : MeasurableSet {ω | x ∈ G.untrueAt (read · ω) k T edges} := by
      have := measurable_of_strings hmeas (probeStrings T k x)
        (fun rd => decide (x ∈ G.untrueAt rd k T edges)) fun rd rd' h => by
          simp only [untrue_iff_congrK G (hagr x hkx rd rd' h)]
      convert this (measurableSet_singleton true) using 1
      ext ω; simp
    have hmN : Measurable fun ω => (twinsBy (fun z => (read z ω).cut) T edges k x : ℝ) :=
      measurable_of_strings hmeas (probeStrings T k x)
        (fun rd => (twinsBy (fun z => (rd z).cut) T edges k x : ℝ)) fun rd rd' h => by
          rw [twinsBy_congr_strings (edges := edges) hkx fun z hz => by rw [h z hz]]
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
      simp only [tf]
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
    (fun x z hz => probeStrings_prefix hz) uf tf huf htf hub htb hmean
  have hw1 : ∑ x ∈ lenRange 0 L, D.real {x} = 1 := by
    have := measureReal_eq_sum_len D hlen Set.univ
    simpa using this.symm
  have hUs : ∀ ω, ∑ x ∈ lenRange 0 L, D.real {x} * uf (read · ω) x
      = D.real (G.untrueAt (read · ω) k T edges) := by
    intro ω
    rw [measureReal_eq_sum_len D hlen]
    refine Finset.sum_congr rfl fun x _ => ?_
    by_cases hkx : k ≤ x.toList.length
    · simp only [uf, hkx, true_and]
    · have hs' : x.toList.length < k := by omega
      have : x ∉ G.untrueAt (read · ω) k T edges := by
        simp [ReadModel.untrueAt, recordBy_short _ T edges hs']
      simp [uf, hkx, this]
  have hNs : ∀ ω, ∑ x ∈ lenRange 0 L, D.real {x} * tf (read · ω) x
      = ∫ x, (twinsBy (fun z => (read z ω).cut) T edges k x : ℝ) ∂D + s := by
    intro ω
    rw [integral_eq_sum_len D hlen _ (L + 1) fun x hx => by
      rw [abs_of_nonneg (Nat.cast_nonneg _)]
      have := twinsBy_le (fun z => (read z ω).cut) T edges k x
      exact_mod_cast this.trans (by omega)]
    rw [← mul_one s, ← hw1, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun x _ => ?_
    simp only [tf]
    ring
  calc μ _ ≤ μ {ω | 2 * G.κ * ∑ x ∈ lenRange 0 L, D.real {x} * tf (read · ω) x + η
        < ∑ x ∈ lenRange 0 L, D.real {x} * uf (read · ω) x} := by
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
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.readDist (G.M.eval z.toList) r) (hκ : 0 < G.κ)
    (hε : 0 ≤ G.ε) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k L S : ℕ)
    {p₀ ρ κs : ℝ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hp₀ : 0 < p₀) (hpmax : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (hκs : 2 * G.κ ≤ κs)
    (hρ : 2 * (G.ε * ((L + 1) * (Fintype.card σ + S + 1))) ≤ ρ) :
    μ {ω | ¬ ∀ T edges, G.InClass S T → EdgesInto T edges →
        D.real (G.untrueAt (read · ω) k T edges)
          ≤ κs * ∫ x, (twinsBy (fun z => (read z ω).cut) T edges k x : ℝ) ∂D + ρ}
      ≤ ENNReal.ofReal ((classSet (Fintype.card σ + S) : Finset (DTree α)).card
        * (Fintype.card σ + S + 3) ^ ((Fintype.card σ + S + 2) * Fintype.card α)
        * Real.exp (-((ρ - 2 * (G.ε * ((L + 1) * (Fintype.card σ + S + 1))))
          / (2 * p₀ * (1 + 4 * (G.κ * (L + 1)
            + G.ε * ((L + 1) * (Fintype.card σ + S + 1)))))))) := by
  set n := Fintype.card σ + S
  set η : ℝ := ρ - 2 * (G.ε * ((L + 1) * (n + 1)))
  have hη : 0 ≤ η := by simp only [η, n]; push_cast; linarith
  set ε' : ℝ := Real.exp (-(η / (2 * p₀ * (1 + 4 * (G.κ * (L + 1) + G.ε * ((L + 1) * (n + 1)))))))
  set B : DTree α → EdgeTable α → Set Ω := fun T e => {ω |
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
      twinsBy_econgr (k := k) (x := x) he hagr _
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
    refine (spurious_one G hmeas hind hlaw hκ hε D hlen hp₀ hpmax hη T e
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
                  _ ≤ (T.leaves.length + 1) ^ (T.leaves.length * Fintype.card α) := edgeMaps_card T
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
        simp only [n, ε', η]
        push_cast
        ring

end SpurUnion

end OrthoDFA
