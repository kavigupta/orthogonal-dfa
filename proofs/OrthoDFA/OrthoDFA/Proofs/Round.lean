import OrthoDFA.Round
import OrthoDFA.Proofs.Replay

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
variable {μ : Measure Ω} {Q : Type*}

omit [Fintype α] [DecidableEq α] in
theorem take_eq_iff_prefix {x t : FreeMonoid α} {i : ℕ} (hi : i ≤ t.toList.length) :
    x.toList.take i = t.toList.take i ↔ t.toList.take i <+: x.toList := by
  constructor
  · intro h; rw [← h]; exact List.take_prefix _ _
  · intro h
    have hlen : (t.toList.take i).length = i := List.length_take_of_le hi
    have := List.prefix_iff_eq_take.1 h
    rw [hlen] at this
    exact this.symm

/-- `RoundOutcome`'s yield gives `d ≤ L · P(harvest)`, its harvest spread gives
`P(t harvested) ≤ ∑ min(i+1,L)/L · D(first i letters are t's)`, and for `i ≤ |t|` that last chance
is `L · ν(t's first i letters) ≤ L · κ · V(its state)`. -/
theorem harvestSpread_of (A : DFA (FreeMonoid α) Q) (R : CutReads α) (H : Hypothesis α)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (L : ℕ) (κ : ℝ)
    (hb : R.B.lo ≤ R.B.hi)
    (hκ : ∀ y, prefixWeight D L y ≤ κ * stateWeight A D L (A.state y)) :
    HarvestSpread A R H D L κ := by
  intro t
  rcases Nat.eq_zero_or_pos L with hL | hL
  · subst hL
    simp [anchorLaw]
  obtain ⟨hy, hs, -⟩ := round_outcome_holds R H D L hb
  have hLr : (0 : ℝ) < L := by exact_mod_cast hL
  set Y := (D.prod (anchorLaw L)).real {q | (replay R H q.1 q.2).2 ≠ []}
  set d := D.real {x | DFAandDTDisagree R H x}
  set Pt := (D.prod (anchorLaw L)).real {q | t ∈ (replay R H q.1 q.2).2}
  set X := κ * ∑ i ∈ Finset.range (t.toList.length + 1),
    ((min (i + 1) L : ℕ) : ℝ) * stateWeight A D L (A.state (prefixOf t i))
  have hterm : ∀ i ∈ Finset.range (t.toList.length + 1),
      ((min (i + 1) L : ℕ) : ℝ) / L * D.real {p | p.toList.take i = t.toList.take i}
        ≤ κ * (((min (i + 1) L : ℕ) : ℝ) * stateWeight A D L (A.state (prefixOf t i))) := by
    intro i hi
    have hi' : i ≤ t.toList.length := Nat.lt_succ_iff.1 (Finset.mem_range.1 hi)
    have hset : {p : FreeMonoid α | p.toList.take i = t.toList.take i}
        = {p | (prefixOf t i).toList <+: p.toList} := by
      ext p; exact take_eq_iff_prefix hi'
    have hν := hκ (prefixOf t i)
    rw [prefixWeight, div_le_iff₀ hLr] at hν
    rw [hset]
    have hm : (0 : ℝ) ≤ ((min (i + 1) L : ℕ) : ℝ) := Nat.cast_nonneg _
    calc ((min (i + 1) L : ℕ) : ℝ) / L * D.real {p | (prefixOf t i).toList <+: p.toList}
        ≤ ((min (i + 1) L : ℕ) : ℝ) / L * (κ * stateWeight A D L (A.state (prefixOf t i)) * L) :=
          mul_le_mul_of_nonneg_left hν (div_nonneg hm hLr.le)
      _ = κ * (((min (i + 1) L : ℕ) : ℝ) * stateWeight A D L (A.state (prefixOf t i))) := by
          field_simp
  have hPt : Pt ≤ X := by
    refine le_trans (hs t) ?_
    simp only [X, Finset.mul_sum]
    exact Finset.sum_le_sum hterm
  have hPt0 : 0 ≤ Pt := measureReal_nonneg
  have hd : d ≤ L * Y := by
    have := hy; rw [div_le_iff₀ hLr] at this; linarith
  have hd0 : 0 ≤ d := measureReal_nonneg
  have hX0 : 0 ≤ X := le_trans hPt0 hPt
  calc d * Pt ≤ d * X := mul_le_mul_of_nonneg_left hPt hd0
    _ ≤ (L * Y) * X := mul_le_mul_of_nonneg_right hd hX0
    _ = L * κ * (∑ i ∈ Finset.range (t.toList.length + 1),
          ((min (i + 1) L : ℕ) : ℝ) * stateWeight A D L (A.state (prefixOf t i))) * Y := by ring

variable (R : CutReads α)

omit [Fintype α] [DecidableEq α] in
theorem le_length_of_suffixDisagree (H : Hypothesis α) {x : FreeMonoid α} {e : ℕ}
    (h : SuffixDisagree R H x e) : e ≤ x.toList.length := by
  by_contra hlt
  apply h
  have hle : x.toList.length ≤ e := by omega
  have hp : prefixOf x e = x := by
    simp [prefixOf, List.take_of_length_le hle]
  rw [List.drop_of_length_le hle, hp]
  rfl

/-- A replay anchored no earlier than `e` harvests when the draw, walked from where the
middle-of-band reading puts its first `e` letters, disagrees with the tree. -/
theorem replay_harvests_of_suffix (H : Hypothesis α) (hb : R.B.lo ≤ R.B.hi) {x : FreeMonoid α}
    {e : ℕ} (hd : SuffixDisagree R H x e) : (replay R H x e).2 ≠ [] := by
  have he := le_length_of_suffixDisagree R H hd
  obtain ⟨rest, hrest⟩ := filter_range_ge he
  rcases hε : H.tree.sift R.cut (prefixOf x e) with p | b
  · have hp : midPath R H (prefixOf x e) = p := midPath_of_sift R H hb hε
    have hs := anchorSearch_cons_of_sift R H.tree (is := rest) hε
    have hend : ((x.toList.drop e).scanl H.step p).getD (x.toList.length - e) []
        = (x.toList.drop e).foldl H.step p := by
      rw [← List.length_drop, scanl_getD_length]
    rcases hx : H.tree.sift R.cut x with actual | b
    · have ha : actual ≠ (x.toList.drop e).foldl H.step p := by
        intro h
        apply hd
        rw [midPath_of_sift R H hb hx, hp, h]
      simp only [replay, hrest, hs, hx, hend, ha, if_false]
      split <;> simp
    · simp [replay, hrest, hs, hx]
  · obtain ⟨l, hl⟩ := replay_snd R H x e
    rw [hl, hrest]
    simp [anchorSearch, hε]

/-- An anchor at which a disagreeing draw is still in sync walks on as the draw's walk from `ε`
does, so it ends in the same disagreement. -/
theorem suffixDisagree_of_inSync (H : Hypothesis α) {x : FreeMonoid α} {e : ℕ}
    (hs : InSync R H x e) (hd : DFAandDTDisagree R H x) : SuffixDisagree R H x e := by
  intro h
  apply hd
  rw [← List.take_append_drop e x.toList, List.foldl_append, hs]
  exact h

theorem prod_sum_dirac (D : Measure (FreeMonoid α)) [SFinite D] {S : Set (FreeMonoid α × ℕ)}
    (hS : MeasurableSet S) (n : ℕ) :
    (D.prod (∑ k ∈ Finset.range n, Measure.dirac k)) S
      = ∑ k ∈ Finset.range n, D ((fun x => (x, k)) ⁻¹' S) := by
  induction n with
  | zero => simp
  | succ n ih =>
    have hm : Measurable fun x : FreeMonoid α => (x, n) := measurable_id.prodMk measurable_const
    rw [Finset.sum_range_succ, Measure.prod_add, Measure.add_apply, ih, Finset.sum_range_succ,
      Measure.prod_dirac, Measure.map_apply hm hS]

theorem anchored_yield_holds : AnchoredYield := by
  intro α _ _ R H D _ L hb
  set S := {q : FreeMonoid α × ℕ | (replay R H q.1 q.2).2 ≠ []}
  have hS : MeasurableSet S := S.to_countable.measurableSet
  have hsum : (D.prod (anchorLaw L)) S
      = (L : ℝ≥0∞)⁻¹ * ∑ k ∈ Finset.range L, D ((fun x => (x, k)) ⁻¹' S) := by
    rw [anchorLaw, Measure.prod_smul_right, Measure.smul_apply, prod_sum_dirac D hS, smul_eq_mul]
  have hle : ∀ k, D {x | SuffixDisagree R H x k} ≤ D ((fun x => (x, k)) ⁻¹' S) := fun k =>
    measure_mono fun x hx => replay_harvests_of_suffix R H hb hx
  have hsumle : ∑ e ∈ Finset.range L, D {x | SuffixDisagree R H x e}
      ≤ ∑ k ∈ Finset.range L, D ((fun x => (x, k)) ⁻¹' S) :=
    Finset.sum_le_sum fun k _ => hle k
  have htop : ∀ k, D {x | SuffixDisagree R H x k} ≠ ⊤ := fun _ => measure_ne_top _ _
  rw [measureReal_def, hsum, div_eq_inv_mul, ENNReal.toReal_mul, ENNReal.toReal_inv,
    ENNReal.toReal_natCast]
  simp only [measureReal_def]
  rw [← ENNReal.toReal_sum fun k _ => htop k]
  gcongr
  · exact ENNReal.sum_ne_top.2 fun _ _ => measure_ne_top _ _

theorem inSync_yield_holds : InSyncYield := by
  intro α _ _ R H D _ L hb
  refine le_trans ?_ (anchored_yield_holds R H D L hb)
  apply div_le_div_of_nonneg_right _ (Nat.cast_nonneg L)
  exact Finset.sum_le_sum fun e _ =>
    measureReal_mono (fun x hx => suffixDisagree_of_inSync R H hx.1 hx.2)

/-- If at least `θ` of a population's undecided mass sits on strings undecided at least `uHi` of
the time, its size-biased indecision `∫ u² / ∫ u` is at least `θ · uHi`.  This is what makes an
edge population that #408 selects at factor `f` indecisive: then `θ ≥ 1 - 1/f`. -/
theorem share_bad_indecisive {β : Type*} [MeasurableSpace β] (X : Measure β) [IsFiniteMeasure X]
    (u : β → ℝ) (hum : Measurable u) (hu0 : ∀ x, 0 ≤ u x) (hu1 : ∀ x, u x ≤ 1)
    (bad : Set β) (hb : MeasurableSet bad) {uHi θ : ℝ} (hHi0 : 0 ≤ uHi)
    (hHi : ∀ x ∈ bad, uHi ≤ u x) (hθ : θ * ∫ x, u x ∂X ≤ ∫ x in bad, u x ∂X) :
    θ * uHi * ∫ x, u x ∂X ≤ ∫ x, u x ^ 2 ∂X := by
  have hint : Integrable u X :=
    Integrable.of_bound hum.aestronglyMeasurable 1 (Filter.Eventually.of_forall fun x => by
      rw [Real.norm_eq_abs, abs_of_nonneg (hu0 x)]; exact hu1 x)
  have hint2 : Integrable (fun x => u x ^ 2) X :=
    Integrable.of_bound (hum.pow_const 2).aestronglyMeasurable 1
      (Filter.Eventually.of_forall fun x => by
        rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
        nlinarith [hu0 x, hu1 x])
  have hbad : uHi * ∫ x in bad, u x ∂X ≤ ∫ x in bad, u x ^ 2 ∂X := by
    rw [← integral_const_mul]
    refine setIntegral_mono_on (hint.const_mul uHi).integrableOn hint2.integrableOn hb ?_
    intro x hx
    have := hHi x hx
    nlinarith [hu0 x]
  have hsq : ∫ x in bad, u x ^ 2 ∂X ≤ ∫ x, u x ^ 2 ∂X :=
    setIntegral_le_integral hint2 (Filter.Eventually.of_forall fun x => sq_nonneg _)
  nlinarith

/-- A rollover chain's enrichment: over `k` fresh links each reading badly read strings at least
`uHi` and clean ones at most `c`, the bad part keeps at least `uHi^k` of its mass and the clean part
at most `c^k`, so the bad-to-clean odds grow by `(uHi / c)^k`. -/
theorem rolled_odds {β : Type*} [MeasurableSpace β] (X : Measure β) [IsFiniteMeasure X]
    (g : ℕ → β → ℝ) (k : ℕ) (hm : ∀ j, Measurable (g j)) (hg0 : ∀ j x, 0 ≤ g j x)
    (hg1 : ∀ j x, g j x ≤ 1) (bad : Set β) (hb : MeasurableSet bad) {uHi c : ℝ}
    (hHi0 : 0 ≤ uHi) (hHi : ∀ j, ∀ x ∈ bad, uHi ≤ g j x) (hc : ∀ j, ∀ x ∈ badᶜ, g j x ≤ c) :
    uHi ^ k * X.real bad ≤ ∫ x in bad, ∏ j ∈ Finset.range k, g j x ∂X
      ∧ ∫ x in badᶜ, ∏ j ∈ Finset.range k, g j x ∂X ≤ c ^ k * X.real badᶜ := by
  have hpm : Measurable fun x => ∏ j ∈ Finset.range k, g j x :=
    Finset.measurable_prod _ fun j _ => hm j
  have hp0 : ∀ x, 0 ≤ ∏ j ∈ Finset.range k, g j x := fun x =>
    Finset.prod_nonneg fun j _ => hg0 j x
  have hp1 : ∀ x, ∏ j ∈ Finset.range k, g j x ≤ 1 := fun x =>
    Finset.prod_le_one (fun j _ => hg0 j x) fun j _ => hg1 j x
  have hint : Integrable (fun x => ∏ j ∈ Finset.range k, g j x) X :=
    Integrable.of_bound hpm.aestronglyMeasurable 1 (Filter.Eventually.of_forall fun x => by
      rw [Real.norm_eq_abs, abs_of_nonneg (hp0 x)]; exact hp1 x)
  constructor
  · have : ∫ x in bad, uHi ^ k ∂X ≤ ∫ x in bad, ∏ j ∈ Finset.range k, g j x ∂X := by
      refine setIntegral_mono_on (integrableOn_const (measure_ne_top _ _)) hint.integrableOn hb ?_
      intro x hx
      have := Finset.prod_le_prod (s := Finset.range k) (fun j _ => hHi0) fun j _ => hHi j x hx
      simpa [Finset.prod_const, Finset.card_range] using this
    simpa [setIntegral_const, smul_eq_mul, mul_comm] using this
  · have hbc : MeasurableSet badᶜ := hb.compl
    have : ∫ x in badᶜ, ∏ j ∈ Finset.range k, g j x ∂X ≤ ∫ x in badᶜ, c ^ k ∂X := by
      refine setIntegral_mono_on hint.integrableOn (integrableOn_const (measure_ne_top _ _)) hbc ?_
      intro x hx
      have := Finset.prod_le_prod (s := Finset.range k) (fun j _ => hg0 j x) fun j _ => hc j x hx
      simpa [Finset.prod_const, Finset.card_range] using this
    simpa [setIntegral_const, smul_eq_mul, mul_comm] using this

theorem withDensity_real_eq {β : Type*} [MeasurableSpace β] (X : Measure β) [IsFiniteMeasure X]
    (g : β → ℝ) (hm : Measurable g) (h0 : ∀ x, 0 ≤ g x) {S : Set β} (hS : MeasurableSet S) :
    (X.withDensity fun x => ENNReal.ofReal (g x)).real S = ∫ x in S, g x ∂X := by
  rw [measureReal_def, withDensity_apply _ hS,
    integral_eq_lintegral_of_nonneg_ae (Filter.Eventually.of_forall h0) hm.aestronglyMeasurable]

theorem stateIndecision_nonneg (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α))
    (B : State) (F : Finset (FreeMonoid α)) (q : Q) : 0 ≤ stateIndecision A O B F q :=
  Real.sSup_nonneg fun _ ⟨_, _, h⟩ => h ▸ measureReal_nonneg

theorem stateIndecision_le_one [IsProbabilityMeasure μ] (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (q : Q) :
    stateIndecision A O B F q ≤ 1 :=
  Real.sSup_le (fun _ ⟨_, _, h⟩ => h ▸ measureReal_le_one) zero_le_one

theorem list_prod_ge {l : List ℝ} {b : ℝ} (hb : 0 ≤ b) (h : ∀ y ∈ l, b ≤ y) :
    b ^ l.length ≤ l.prod := by
  induction l with
  | nil => simp
  | cons y l ih =>
    rw [List.prod_cons, List.length_cons, pow_succ]
    have hy := h y (List.mem_cons_self ..)
    have ih' := ih fun z hz => h z (List.mem_cons_of_mem _ hz)
    calc b ^ l.length * b ≤ l.prod * y :=
          mul_le_mul ih' hy hb (le_trans (pow_nonneg hb _) ih')
      _ = y * l.prod := mul_comm _ _

theorem list_prod_le {l : List ℝ} {b : ℝ} (h : ∀ y ∈ l, 0 ≤ y ∧ y ≤ b) :
    l.prod ≤ b ^ l.length := by
  induction l with
  | nil => simp
  | cons y l ih =>
    rw [List.prod_cons, List.length_cons, pow_succ]
    have hy := h y (List.mem_cons_self ..)
    have ih' := ih fun z hz => h z (List.mem_cons_of_mem _ hz)
    have h0 : 0 ≤ l.prod := List.prod_nonneg fun z hz => (h z (List.mem_cons_of_mem _ hz)).1
    calc y * l.prod ≤ b * b ^ l.length := mul_le_mul hy.2 ih' h0 (le_trans hy.1 hy.2)
      _ = b ^ l.length * b := mul_comm _ _

/-- After `k ≥ rolloverRounds − 1` links the bad-to-clean odds are at least
`(uHi/(r·a))^k · π/(1−π) ≥ f·r·a/(uHi − f·r·a)`, so the bad share is at least `f·r·a/uHi` and the
chain's rate reaches #408's promotion rate. -/
theorem rollover_promotes [IsProbabilityMeasure μ] (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (X : Measure (FreeMonoid α)) [IsFiniteMeasure X] (c : α)
    (links : List (State × Finset (FreeMonoid α))) (bad : Set (FreeMonoid α))
    {f a uHi π : ℝ} {r : ℕ} (hf : 1 ≤ f) (hra : 0 < r * a) (hT : f * r * a < uHi)
    (hπ0 : 0 < π) (hπ1 : π < 1)
    (hHi : ∀ l ∈ links, ∀ x ∈ bad,
      uHi ≤ stateIndecision A O l.1 l.2 (A.state (x * FreeMonoid.of c)))
    (hc : ∀ l ∈ links, ∀ x ∉ bad,
      stateIndecision A O l.1 l.2 (A.state (x * FreeMonoid.of c)) ≤ r * a)
    (hπ : π * X.real Set.univ ≤ X.real bad)
    (hk : rolloverRounds f a uHi π r ≤ links.length + 1) :
    f * r * a / uHi * (rolledLaw A O X c links).real Set.univ
      ≤ (rolledLaw A O X c links).real bad := by
  set k := links.length
  set T := f * r * a with hTdef
  have hT0 : 0 < T := by rw [hTdef, mul_assoc]; exact mul_pos (by linarith) hra
  have hHi0 : 0 < uHi := lt_trans hT0 hT
  have hraT : r * a ≤ T := by rw [hTdef, mul_assoc]; nlinarith
  set u := fun (l : State × Finset (FreeMonoid α)) (x : FreeMonoid α) =>
    stateIndecision A O l.1 l.2 (A.state (x * FreeMonoid.of c))
  set G : FreeMonoid α → ℝ := fun x => (links.map fun l => u l x).prod with hG
  have hmem : ∀ x, ∀ y ∈ links.map (fun l => u l x), 0 ≤ y ∧ y ≤ 1 := by
    intro x y hy
    obtain ⟨l, -, rfl⟩ := List.mem_map.1 hy
    exact ⟨stateIndecision_nonneg A O _ _ _, stateIndecision_le_one A O _ _ _⟩
  have hG0 : ∀ x, 0 ≤ G x := fun x => List.prod_nonneg fun y hy => (hmem x y hy).1
  have hG1 : ∀ x, G x ≤ 1 := fun x => by simpa using list_prod_le (hmem x)
  have hGm : Measurable G := measurable_from_top
  have hint : Integrable G X :=
    Integrable.of_bound hGm.aestronglyMeasurable 1 (Filter.Eventually.of_forall fun x => by
      rw [Real.norm_eq_abs, abs_of_nonneg (hG0 x)]; exact hG1 x)
  have hbadm : MeasurableSet bad := MeasurableSpace.measurableSet_top
  have hlaw : rolledLaw A O X c links = X.withDensity fun x => ENNReal.ofReal (G x) := rfl
  rw [hlaw, withDensity_real_eq X G hGm hG0 MeasurableSet.univ,
    withDensity_real_eq X G hGm hG0 hbadm, setIntegral_univ, ← integral_add_compl hbadm hint]
  set B := ∫ x in bad, G x ∂X
  set C := ∫ x in badᶜ, G x ∂X
  have hBpt : ∀ x ∈ bad, uHi ^ k ≤ G x := fun x hx => by
    have := list_prod_ge hHi0.le (l := links.map fun l => u l x)
      (fun y hy => by obtain ⟨l, hl, rfl⟩ := List.mem_map.1 hy; exact hHi l hl x hx)
    rwa [List.length_map] at this
  have hCpt : ∀ x ∈ badᶜ, G x ≤ (r * a) ^ k := fun x hx => by
    have := list_prod_le (l := links.map fun l => u l x) (b := r * a)
      (fun y hy => by
        obtain ⟨l, hl, rfl⟩ := List.mem_map.1 hy
        exact ⟨stateIndecision_nonneg A O _ _ _, hc l hl x hx⟩)
    rwa [List.length_map] at this
  have hB : uHi ^ k * X.real bad ≤ B := by
    have h := setIntegral_mono_on (f := fun _ : FreeMonoid α => uHi ^ k) (g := G)
      (integrableOn_const (measure_ne_top _ _)) hint.integrableOn hbadm hBpt
    rw [setIntegral_const, smul_eq_mul, mul_comm] at h
    exact h
  have hC : C ≤ (r * a) ^ k * X.real badᶜ := by
    have h := setIntegral_mono_on (f := G) (g := fun _ : FreeMonoid α => (r * a) ^ k)
      hint.integrableOn (integrableOn_const (measure_ne_top _ _)) hbadm.compl hCpt
    rw [setIntegral_const, smul_eq_mul, mul_comm] at h
    exact h
  have hβ : 1 < uHi / (r * a) := by rw [one_lt_div hra]; linarith
  have hZ0 : 0 < T * (1 - π) / ((uHi - T) * π) :=
    div_pos (mul_pos hT0 (by linarith)) (mul_pos (by linarith) hπ0)
  have hlog : Real.logb (uHi / (r * a)) (T * (1 - π) / ((uHi - T) * π)) ≤ k := by
    have h1 := Nat.le_ceil (Real.logb (uHi / (r * a)) (T * (1 - π) / ((uHi - T) * π)))
    have h2 : ⌈Real.logb (uHi / (r * a)) (T * (1 - π) / ((uHi - T) * π))⌉₊ ≤ k := by
      have := hk; unfold rolloverRounds at this; rw [← hTdef] at this; omega
    exact le_trans h1 (by exact_mod_cast h2)
  have hZ : T * (1 - π) / ((uHi - T) * π) ≤ (uHi / (r * a)) ^ k := by
    calc T * (1 - π) / ((uHi - T) * π)
        = (uHi / (r * a)) ^ Real.logb (uHi / (r * a)) (T * (1 - π) / ((uHi - T) * π)) :=
          (Real.rpow_logb (by linarith) hβ.ne' hZ0).symm
      _ ≤ (uHi / (r * a)) ^ (k : ℝ) := Real.rpow_le_rpow_of_exponent_le hβ.le hlog
      _ = (uHi / (r * a)) ^ k := Real.rpow_natCast _ _
  rw [div_pow, le_div_iff₀ (pow_pos hra k), div_mul_eq_mul_div,
    div_le_iff₀ (mul_pos (by linarith) hπ0)] at hZ
  have hU : 0 ≤ X.real Set.univ := measureReal_nonneg
  have hcompl : X.real badᶜ = X.real Set.univ - X.real bad := measureReal_compl hbadm
  have hCx : X.real badᶜ ≤ (1 - π) * X.real Set.univ := by rw [hcompl]; linarith
  have hrak : 0 ≤ (r * a) ^ k := pow_nonneg hra.le k
  have hHik : 0 ≤ uHi ^ k := pow_nonneg hHi0.le k
  have e1 : T * C ≤ T * ((r * a) ^ k * ((1 - π) * X.real Set.univ)) :=
    mul_le_mul_of_nonneg_left (le_trans hC (mul_le_mul_of_nonneg_left hCx hrak)) hT0.le
  have e2 : T * ((r * a) ^ k * ((1 - π) * X.real Set.univ))
      ≤ uHi ^ k * ((uHi - T) * π) * X.real Set.univ := by
    have := mul_le_mul_of_nonneg_right hZ hU
    nlinarith
  have e3 : uHi ^ k * ((uHi - T) * π) * X.real Set.univ ≤ (uHi - T) * B := by
    have hd : 0 ≤ uHi - T := by linarith
    have := mul_le_mul_of_nonneg_left hπ (mul_nonneg hHik hd)
    have := mul_le_mul_of_nonneg_left hB hd
    nlinarith
  rw [div_mul_eq_mul_div, div_le_iff₀ hHi0]
  nlinarith

/-- A chain whose source holds predecessors of a badly read state advances whenever the round's
family reads every other draw's extension cleanly, as the gap premise has it. -/
theorem chainAdvances_of_mass [IsProbabilityMeasure μ] (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (R : CutReads α) (sources : List (Measure (FreeMonoid α)))
    (hfin : ∀ X ∈ sources, IsFiniteMeasure X) {a uHi : ℝ} (hHi0 : 0 ≤ uHi)
    (hgap : GapPremise A O R.B R.F a uHi)
    (hmass : ∃ X ∈ sources, ∃ c : α, 0 < X.real (badAlong A O R c uHi)) :
    ChainAdvances A O R sources uHi a := by
  obtain ⟨X, hX, c, hpos⟩ := hmass
  have := hfin X hX
  refine ⟨X, hX, c, hpos, ?_⟩
  set g := fun x : FreeMonoid α => stateIndecision A O R.B R.F (A.state (x * FreeMonoid.of c))
  have hm : Measurable g := measurable_from_top
  have hlaw : rolledLaw A O X c [(R.B, R.F)] = X.withDensity fun x => ENNReal.ofReal (g x) := by
    simp [rolledLaw, g]
  have hbad : MeasurableSet (badAlong A O R c uHi) := MeasurableSpace.measurableSet_top
  have hodds := rolled_odds X (fun _ => g) 1 (fun _ => hm)
    (fun _ x => stateIndecision_nonneg A O R.B R.F _) (fun _ x => stateIndecision_le_one A O R.B R.F _)
    _ hbad hHi0 (fun _ x hx => hx) (fun _ x hx => by
      rcases hgap (A.state (x * FreeMonoid.of c)) with h | h
      · exact h
      · exact absurd h hx)
  simp only [Finset.prod_range_one, pow_one] at hodds
  rw [hlaw, withDensity_real_eq X g hm (fun x => stateIndecision_nonneg A O R.B R.F _) hbad,
    withDensity_real_eq X g hm (fun x => stateIndecision_nonneg A O R.B R.F _) hbad.compl]
  exact hodds

theorem anchorLaw_singleton {L k : ℕ} (hk : k < L) : anchorLaw L {k} = (L : ℝ≥0∞)⁻¹ := by
  simp only [anchorLaw, Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply,
    Measure.dirac_apply, Set.indicator_apply, Set.mem_singleton_iff, Pi.one_apply, smul_eq_mul]
  rw [Finset.sum_ite_eq' (Finset.range L) k fun _ => (1 : ℝ≥0∞), if_pos (Finset.mem_range.2 hk),
    mul_one]

/-- A string `x` shorter than `L` that `D` begins with is in the `ν` root with chance at least
`ν(x) = D(x is a prefix) / L`. -/
theorem nuRoot_pos (D : Measure (FreeMonoid α)) [IsFiniteMeasure D] {L : ℕ} {x : FreeMonoid α}
    (hx : x.toList.length < L) (hD : 0 < D.real {p | x.toList <+: p.toList})
    {S : Set (FreeMonoid α)} (hS : x ∈ S) : 0 < (nuRoot D L).real S := by
  have hm : Measurable fun q : FreeMonoid α × ℕ => prefixOf q.1 q.2 := measurable_of_countable _
  have hsub : {p : FreeMonoid α | x.toList <+: p.toList} ×ˢ ({x.toList.length} : Set ℕ)
      ⊆ (fun q : FreeMonoid α × ℕ => prefixOf q.1 q.2) ⁻¹' S := by
    rintro ⟨p, k⟩ ⟨hp, hk⟩
    simp only [Set.mem_singleton_iff] at hk
    subst hk
    obtain ⟨t, ht⟩ := hp
    show prefixOf p x.toList.length ∈ S
    have : prefixOf p x.toList.length = x := by
      simp [prefixOf, ← ht]
    rwa [this]
  have hpos : 0 < (D.prod (anchorLaw L))
      ({p : FreeMonoid α | x.toList <+: p.toList} ×ˢ ({x.toList.length} : Set ℕ)) := by
    rw [Measure.prod_prod, anchorLaw_singleton hx]
    refine ENNReal.mul_pos ?_ (ENNReal.inv_ne_zero.2 (ENNReal.natCast_ne_top L))
    intro h0
    rw [measureReal_def, h0, ENNReal.toReal_zero] at hD
    exact lt_irrefl _ hD
  rw [measureReal_def, nuRoot, Measure.map_apply hm (Set.to_countable S).measurableSet]
  exact ENNReal.toReal_pos (lt_of_lt_of_le hpos (measure_mono hsub)).ne'
    (measure_ne_top _ _)

/-- With the `ν` root among the sources, a round advances a chain whenever some string read
before position `L` leads by `c` to a badly read state. -/
theorem chainAdvances_of_visited [IsProbabilityMeasure μ] (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (R : CutReads α) (D : Measure (FreeMonoid α)) [IsFiniteMeasure D]
    (L : ℕ) (others : List (Measure (FreeMonoid α))) (hfin : ∀ X ∈ others, IsFiniteMeasure X)
    {a uHi : ℝ} (hHi0 : 0 ≤ uHi) (hgap : GapPremise A O R.B R.F a uHi) {x : FreeMonoid α}
    {c : α} (hx : x.toList.length < L) (hD : 0 < D.real {p | x.toList <+: p.toList})
    (hbad : uHi ≤ stateIndecision A O R.B R.F (A.state (x * FreeMonoid.of c))) :
    ChainAdvances A O R (nuRoot D L :: others) uHi a := by
  refine chainAdvances_of_mass A O R _ ?_ hHi0 hgap ⟨nuRoot D L, List.mem_cons_self .., c,
    nuRoot_pos D hx hD (S := badAlong A O R c uHi) hbad⟩
  intro X hX
  rcases List.mem_cons.1 hX with rfl | h
  · unfold nuRoot; infer_instance
  · exact hfin X h

/-- A chain filtered by any chance `g` advances once its source holds strings with `g ≥ hi` and
every other string has `g ≤ lo`. -/
theorem chainAdvancesBy_of_mass (X : Measure (FreeMonoid α)) [IsFiniteMeasure X]
    (g : FreeMonoid α → ℝ) (h0 : ∀ x, 0 ≤ g x) (h1 : ∀ x, g x ≤ 1) {lo hi : ℝ} (hhi : 0 ≤ hi)
    (hgap : ∀ x, g x ≤ lo ∨ hi ≤ g x) (hpos : 0 < X.real {x | hi ≤ g x}) :
    ChainAdvancesBy X g lo hi := by
  have hm : Measurable g := measurable_from_top
  have hbad : MeasurableSet {x | hi ≤ g x} := MeasurableSpace.measurableSet_top
  have hodds := rolled_odds X (fun _ => g) 1 (fun _ => hm) (fun _ => h0) (fun _ => h1) _ hbad hhi
    (fun _ x hx => hx) (fun _ x hx => (hgap x).resolve_right hx)
  simp only [Finset.prod_range_one, pow_one] at hodds
  refine ⟨hpos, ?_⟩
  rw [withDensity_real_eq X g hm h0 hbad, withDensity_real_eq X g hm h0 hbad.compl]
  exact hodds

theorem edgeDisagreeProb_nonneg (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (H : Hypothesis α) (x : FreeMonoid α) (c : α) :
    0 ≤ edgeDisagreeProb O B F H x c := measureReal_nonneg

theorem edgeDisagreeProb_le_one [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α))
    (B : State) (F : Finset (FreeMonoid α)) (H : Hypothesis α) (x : FreeMonoid α) (c : α) :
    edgeDisagreeProb O B F H x c ≤ 1 := measureReal_le_one

/-- With the `ν` root among the sources, a round advances a disagreement chain whenever some
string read before position `L` is read off its edge by `c` at least `wHi` of the time. -/
theorem chainAdvancesEither_of_visited [IsProbabilityMeasure μ] (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (R : CutReads α) (H : Hypothesis α)
    (D : Measure (FreeMonoid α)) [IsFiniteMeasure D] (L : ℕ)
    (others : List (Measure (FreeMonoid α))) {a uHi η wHi : ℝ} (hHi0 : 0 ≤ uHi) (hw0 : 0 ≤ wHi)
    (hgap : GapPremise A O R.B R.F a uHi) (hegap : EdgeGapPremise O R.B R.F H η wHi)
    {x : FreeMonoid α} {c : α} (hx : x.toList.length < L)
    (hD : 0 < D.real {p | x.toList <+: p.toList})
    (hbad : uHi ≤ stateIndecision A O R.B R.F (A.state (x * FreeMonoid.of c))
      ∨ wHi ≤ edgeDisagreeProb O R.B R.F H x c) :
    ChainAdvancesEither A O R H (nuRoot D L :: others) a uHi η wHi := by
  have : IsFiniteMeasure (nuRoot D L) := by unfold nuRoot; infer_instance
  refine ⟨nuRoot D L, List.mem_cons_self .., c, ?_⟩
  rcases hbad with h | h
  · exact .inl (chainAdvancesBy_of_mass _ _ (fun _ => stateIndecision_nonneg A O _ _ _)
      (fun _ => stateIndecision_le_one A O _ _ _) hHi0 (fun y => hgap _)
      (nuRoot_pos D hx hD (S := {y | uHi ≤ _}) h))
  · exact .inr (chainAdvancesBy_of_mass _ _ (fun _ => edgeDisagreeProb_nonneg O _ _ H _ c)
      (fun _ => edgeDisagreeProb_le_one O _ _ H _ c) hw0 (fun y => hegap y c)
      (nuRoot_pos D hx hD (S := {y | wHi ≤ _}) h))

/-- A round that advances no chain, with the `ν` root among its sources, reads every edge out of a
visited string cleanly: its successor is not badly read, and it lands on the hypothesis's edge but
for chance `η`. -/
theorem visited_clean_of_not_advances [IsProbabilityMeasure μ] (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (R : CutReads α) (H : Hypothesis α)
    (D : Measure (FreeMonoid α)) [IsFiniteMeasure D] (L : ℕ)
    (others : List (Measure (FreeMonoid α))) {a uHi η wHi : ℝ} (hHi0 : 0 ≤ uHi) (hw0 : 0 ≤ wHi)
    (hgap : GapPremise A O R.B R.F a uHi) (hegap : EdgeGapPremise O R.B R.F H η wHi)
    (hnot : ¬ ChainAdvancesEither A O R H (nuRoot D L :: others) a uHi η wHi)
    {x : FreeMonoid α} (c : α) (hx : x.toList.length < L)
    (hD : 0 < D.real {p | x.toList <+: p.toList}) :
    stateIndecision A O R.B R.F (A.state (x * FreeMonoid.of c)) < uHi
      ∧ edgeDisagreeProb O R.B R.F H x c ≤ η := by
  by_contra h
  apply hnot
  refine chainAdvancesEither_of_visited A O R H D L others hHi0 hw0 hgap hegap (c := c) hx hD ?_
  rcases not_and_or.1 h with h | h
  · exact .inl (not_lt.1 h)
  · exact .inr ((hegap x c).resolve_left h)

/-- Believed true.  A draw the gate counts against the hypothesis leaves its walk first at some
position `j ≤ L`, through a middle reading of `x[:j-1]` or `x[:j]` off its state's side: with every
visited edge read off at most `η < 1 − 2(N+2)φ`, the sides' readings agree with the edge, so
some node read on those two strings' paths (at most `N + 2` deep) flipped.  Reads the pass did not
make are fresh given the pass, each flipping at most `φ`, so the expected disagreement mass is at
most `(L+1)·2(N+2)·φ` and Markov gives the first term.  The pass's own node reads are among
`passReadBound S` (seed and probe prefixes, by a letter, at the final tree's midfixes, by a
letter); read adaptively, each is fresh when read, so the chance any flipped is at most
`passReadBound S · φ`.  Unproved: that the pass is determined by those reads (structural, by
induction over `probeStep`), and the adaptive-read independence (as `probe_couple` on #400). -/
theorem gate_flip_bound [IsProbabilityMeasure μ] (S : RoundSetting α μ Q)
    {ε η φ uHi : ℝ} (hε : 0 < ε) (hlen : ∀ᵐ x ∂S.D, x.toList.length = S.L)
    (hflip : MidFlipPremise S.A S.O S.B S.F uHi φ) (hη : η + 2 * (S.N + 2) * φ < 1) :
    (μ.prod (Measure.pi fun _ : Fin S.N => S.D)).real {θ |
        let R := readsAt S.O S.B S.F θ.1
        let s := roundEnd S.K S.O S.B S.F S.seed θ
        (∀ y c, y.toList.length < S.L → 0 < S.D.real {p | y.toList <+: p.toList} →
            edgeDisagreeProb S.O S.B S.F s.hyp y c ≤ η)
          ∧ (∀ q, stateIndecision S.A S.O S.B S.F q < uHi)
          ∧ ¬ S.D.real {x | DFAandDTDisagree R s.hyp x} ≤ ε}
      ≤ (S.L + 1) * (2 * (S.N + 2)) * φ / ε + passReadBound S * φ := by
  sorry

/-- `RoundTetrachotomyBoth` with its error made explicit: but for the gate's flips, a round whose
hypothesis meets the edge gap ends in (1) agreement within `ε`, (2) a population the next gate
must act on, (3) halving, or (4') a chain advanced. -/
theorem round_tetrachotomy_both [IsProbabilityMeasure μ] (S : RoundSetting α μ Q)
    (ε : ℝ) (nH nS : ℕ) (σ : ℝ) (live : List (Measure (FreeMonoid α)))
    {f a uHi η wHi φ : ℝ} {r : ℕ}
    (hv : S.Valid) (hgap : GapPremise S.A S.O S.B S.F a uHi) (hHi0 : 0 ≤ uHi) (hw0 : 0 ≤ wHi)
    (hε : 0 < ε) (hlen : ∀ᵐ x ∂S.D, x.toList.length = S.L)
    (hflip : MidFlipPremise S.A S.O S.B S.F uHi φ)
    (hbad : BadVisited S.A S.O S.B S.F S.D S.L uHi) (hη : η + 2 * (S.N + 2) * φ < 1) :
    (μ.prod (Measure.pi fun _ : Fin S.N => S.D)).real {θ |
        let R := readsAt S.O S.B S.F θ.1
        let s := roundEnd S.K S.O S.B S.F S.seed θ
        EdgeGapPremise S.O S.B S.F s.hyp η wHi
          ∧ ¬ S.D.real {x | DFAandDTDisagree R s.hyp x} ≤ ε
          ∧ ¬ (HarvestSpread S.A R s.hyp S.D S.L S.κ
            ∧ (PopulationIndecisive S.A S.O R s.hyp S.D S.L S.τ
              ∨ WrongEdgeHarvest S.A S.O R s.hyp S.D S.L nH nS σ a
              ∨ ∃ l ∈ s.hyp.tree.paths, ∃ c : α, EdgeSelected S.A S.O R s.hyp S.D l c f a r
                  ∧ EdgePopulationIndecisive S.A S.O R (S.D[|settlesAt R s.hyp l]) c S.τ))
          ∧ ¬ s.halves S.τ
          ∧ ¬ ChainAdvancesEither S.A S.O R s.hyp
              (nuRoot S.D S.L :: live ++ s.hyp.tree.paths.map fun l => S.D[|settlesAt R s.hyp l])
              a uHi η wHi}
      ≤ (S.L + 1) * (2 * (S.N + 2)) * φ / ε + passReadBound S * φ := by
  have hD : IsProbabilityMeasure S.D := hv.2.1
  refine le_trans (measureReal_mono ?_) (gate_flip_bound S hε hlen hflip hη)
  rintro θ ⟨hegap, hfail, -, -, hnot⟩
  have hclean := fun (y : FreeMonoid α) (c : α) (hy : y.toList.length < S.L)
      (hy' : 0 < S.D.real {p | y.toList <+: p.toList}) =>
    visited_clean_of_not_advances S.A S.O (readsAt S.O S.B S.F θ.1)
      (roundEnd S.K S.O S.B S.F S.seed θ).hyp S.D S.L _ hHi0 hw0 hgap hegap hnot c hy hy'
  refine ⟨fun y c hy hy' => (hclean y c hy hy').2, fun q => ?_, hfail⟩
  by_contra hq
  obtain ⟨y, c, hy, hy', rfl⟩ := hbad q (not_lt.1 hq)
  exact absurd (hclean y c hy hy').1 hq

end OrthoDFA
