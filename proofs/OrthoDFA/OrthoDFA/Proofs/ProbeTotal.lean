import OrthoDFA.Proofs.LoopCodes
import Mathlib.MeasureTheory.Integral.Pi

/-!
# The probes a round's harvested draws walk

A draw walks its block's probes up to its first harvest (`walkLen`). Each probe harvests with
chance at least `y`, the probes i.i.d., so at `c = 2/(2 - y)`

    E[c^walk] ≤ 2

(`integral_walkLen_le`). A round's `K` populations each draw `M` times over independent blocks, so
with `W` the probes its harvested draws walk in all, Markov's inequality on `c^W` gives

    P(W > B) ≤ 2^(K M) (1 - y/2)^(B + 1)

(`restLaw_harvestWalk_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

section Walk

variable {X : Type*}

open scoped Classical in
/-- How many of `xs` a walk takes that stops at the first in `hit`. -/
def walkLen (hit : Set X) : List X → ℕ
  | [] => 0
  | x :: xs => if x ∈ hit then 1 else walkLen hit xs + 1

lemma walkLen_le_length (hit : Set X) : ∀ xs : List X, walkLen hit xs ≤ xs.length
  | [] => le_rfl
  | x :: xs => by
    have := walkLen_le_length hit xs
    simp only [walkLen, List.length_cons]
    split_ifs <;> omega

lemma walkLen_congr {hit hit' : Set X} :
    ∀ {xs : List X}, (∀ x ∈ xs, x ∈ hit ↔ x ∈ hit') → walkLen hit xs = walkLen hit' xs
  | [], _ => rfl
  | x :: xs, h => by
    have ih := walkLen_congr (xs := xs) fun y hy => h y (List.mem_cons_of_mem _ hy)
    have hx := h x List.mem_cons_self
    simp only [walkLen, ih]
    by_cases h1 : x ∈ hit
    · rw [if_pos h1, if_pos (hx.1 h1)]
    · rw [if_neg h1, if_neg fun h2 => h1 (hx.2 h2)]

variable [MeasurableSpace X] [MeasurableSingletonClass X] [Countable X]

/-- Each probe in `hit` with chance at least `y`, `x₁, …, x_N` i.i.d. from `D`:

    E[(2/(2 - y))^walkLen(x₁, …, x_N)] ≤ 2. -/
theorem integral_walkLen_le (D : Measure X) [IsProbabilityMeasure D] (hit : Set X) {y : ℝ}
    (hy0 : 0 ≤ y) (hy1 : y ≤ 1) (hhit : y ≤ D.real hit) (N : ℕ) :
    ∫ xs, (2 / (2 - y)) ^ walkLen hit (List.ofFn xs) ∂(Measure.pi fun _ : Fin N => D) ≤ 2 := by
  set c : ℝ := 2 / (2 - y) with hc
  have h2y : 0 < 2 - y := by linarith
  have hc1 : 1 ≤ c := by rw [hc, le_div_iff₀ h2y]; linarith
  have hcy : c * (2 - y) = 2 := by rw [hc]; field_simp
  have hmeas : MeasurableSet hit := (Set.to_countable hit).measurableSet
  induction N with
  | zero =>
    have e : ∀ xs : Fin 0 → X, c ^ walkLen hit (List.ofFn xs) = 1 := fun xs => by
      simp [walkLen]
    simp only [e, integral_const, probReal_univ, smul_eq_mul, mul_one]
    norm_num
  | succ N ih =>
    set ν := Measure.pi fun _ : Fin N => D
    set g : (Fin N → X) → ℝ := fun t => c ^ walkLen hit (List.ofFn t)
    set a : X → ℝ := hit.indicator fun _ => c
    set b : X → ℝ := hitᶜ.indicator fun _ => c
    have hg0 : ∀ t, 0 ≤ g t := fun t => pow_nonneg (by linarith) _
    have hgb : ∀ t, ‖g t‖ ≤ c ^ N := fun t => by
      rw [Real.norm_of_nonneg (hg0 t)]
      exact pow_le_pow_right₀ hc1 ((walkLen_le_length hit _).trans (by simp))
    have hpt : ∀ z : X × (Fin N → X),
        c ^ walkLen hit (List.ofFn ((MeasurableEquiv.piFinSuccAbove (fun _ => X) 0).symm z))
          = a z.1 + b z.1 * g z.2 := by
      rintro ⟨x, t⟩
      have hl : List.ofFn ((MeasurableEquiv.piFinSuccAbove (fun _ => X) 0).symm (x, t))
          = x :: List.ofFn t := by
        rw [List.ofFn_succ]
        simp [MeasurableEquiv.piFinSuccAbove_symm_apply, Fin.insertNthEquiv, Fin.insertNth_zero']
      rw [hl]
      simp only [walkLen, a, b, g]
      by_cases hx : x ∈ hit
      · rw [if_pos hx, Set.indicator_of_mem hx,
          Set.indicator_of_notMem (Set.notMem_compl_iff.2 hx)]
        ring
      · rw [if_neg hx, Set.indicator_of_notMem hx, Set.indicator_of_mem (Set.mem_compl hx)]
        ring
    have hmp := (measurePreserving_piFinSuccAbove (fun _ : Fin (N + 1) => D) 0).symm
    have hint_a : Integrable (fun z : X × (Fin N → X) => a z.1) (D.prod ν) :=
      Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable c
        (Filter.Eventually.of_forall fun z => by
          simp only [a, Set.indicator]
          split_ifs <;> simp [abs_of_nonneg (by linarith : (0 : ℝ) ≤ c)] <;> linarith)
    have hint_bg : Integrable (fun z : X × (Fin N → X) => b z.1 * g z.2) (D.prod ν) :=
      Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable (c * c ^ N)
        (Filter.Eventually.of_forall fun z => by
          rw [norm_mul]
          have hb : ‖b z.1‖ ≤ c := by
            simp only [b, Set.indicator]
            split_ifs <;> simp [abs_of_nonneg (by linarith : (0 : ℝ) ≤ c)] <;> linarith
          exact mul_le_mul hb (hgb z.2) (norm_nonneg _) (by linarith))
    have hsplit : ∫ xs, c ^ walkLen hit (List.ofFn xs) ∂(Measure.pi fun _ : Fin (N + 1) => D)
        = (∫ x, a x ∂D) + (∫ x, b x ∂D) * ∫ t, g t ∂ν := by
      rw [← hmp.integral_comp' (g := fun xs => c ^ walkLen hit (List.ofFn xs))]
      simp only [hpt]
      rw [integral_add hint_a hint_bg, integral_prod_mul b g]
      congr 1
      have := integral_prod_mul (μ := D) (ν := ν) a (fun _ => (1 : ℝ))
      simp only [mul_one, integral_const, probReal_univ, smul_eq_mul] at this
      exact this
    have ha : ∫ x, a x ∂D = D.real hit * c := by
      simp only [a]; rw [integral_indicator_const _ hmeas, smul_eq_mul]
    have hb : ∫ x, b x ∂D = (1 - D.real hit) * c := by
      simp only [b]
      rw [integral_indicator_const _ hmeas.compl, smul_eq_mul, measureReal_compl hmeas,
        probReal_univ]
    have hp1 : D.real hit ≤ 1 := measureReal_le_one
    rw [hsplit, ha, hb]
    have hI : ∫ t, g t ∂ν ≤ 2 := ih
    have h1p : 0 ≤ (1 - D.real hit) * c := mul_nonneg (by linarith) (by linarith)
    calc D.real hit * c + (1 - D.real hit) * c * ∫ t, g t ∂ν
        ≤ D.real hit * c + (1 - D.real hit) * c * 2 := by gcongr
      _ = c * (2 - D.real hit) := by ring
      _ ≤ c * (2 - y) := by gcongr
      _ = 2 := hcy

end Walk

section Round

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The probes on which the sampler `s` harvests a string off `T`. -/
def hitSet (s : SpecSampler α) (k : ℕ) (T : Finset (FreeMonoid α)) : Set (FreeMonoid α) :=
  {x | (Random.harvestFirst s.read k s.hyp s.site ↑T [x]).isSome}

/-- The probes a population's first `M` draws walk, draw `n` over its block of `v n`; none for the
probes' own population. -/
def popWalk (D : Measure (FreeMonoid α)) (k N M : ℕ) (s : Option (SpecSampler α))
    (T : Finset (FreeMonoid α)) (v : ℕ → unitInterval × (Fin N → unitInterval)) : ℕ :=
  match s with
  | some s => ∑ n : Fin M, walkLen (hitSet s k T) (List.ofFn (probeBlocks D N M v n))
  | none => 0

/-- The probes the harvested populations' first `M` draws walk, population `i`'s over the blocks
of `u (some i)`. -/
def harvestWalk {K : ℕ} (D : Measure (FreeMonoid α)) (k N M : ℕ) (pops : List (SpecSampler α))
    (T : Finset (FreeMonoid α)) (u : Option (Fin K) → ℕ → unitInterval × (Fin N → unitInterval)) :
    ℕ :=
  ∑ i : Fin K, popWalk D k N M pops[i.val]? T (u (some i))

omit [DecidableEq α] in
/-- A probe harvests off `T` with the chance the sampler's law puts off `T`. -/
lemma real_hitSet (s : SpecSampler α) (D : Measure (FreeMonoid α)) (k : ℕ)
    (T : Finset (FreeMonoid α)) :
    D.real (hitSet s k T) = (Random.harvestLaw s.read D k s.hyp s.site).real (↑T)ᶜ := by
  classical
  have h := Random.harvestLaw_restrict_apply s.read D k s.hyp s.site (↑T) Set.univ
  rw [Measure.restrict_apply_univ] at h
  rw [measureReal_def, measureReal_def, h]
  congr 2
  ext x
  simp only [hitSet, Random.harvestFirst, List.findSome?_cons, List.findSome?_nil,
    Set.mem_setOf_eq, Set.mem_univ, and_true]
  rcases (Random.siteHarvest s.read k s.hyp s.site x).head? with _ | v
  · simp
  · by_cases hv : v ∈ T <;> simp [Option.filter, hv]

lemma findSome_take_walkLen {X Y : Type*} (f : X → Option Y) :
    ∀ xs : List X, (xs.findSome? f).isSome →
      ((xs.take (walkLen {x | ([x].findSome? f).isSome} xs)).findSome? f).isSome
  | [], h => h
  | x :: xs, h => by
    have ih := findSome_take_walkLen f xs
    simp only [walkLen, Set.mem_setOf_eq]
    cases hfx : f x with
    | some b =>
      have hx : ([x].findSome? f).isSome := by simp [hfx]
      rw [if_pos hx]
      simp [hfx]
    | none =>
      have hx : ¬ ([x].findSome? f).isSome := by simp [hfx]
      rw [if_neg hx, List.take_succ_cons]
      simp only [List.findSome?_cons, hfx] at h ⊢
      exact ih h

omit [Fintype α] [DecidableEq α] in
/-- A walk that ends in a harvest has its harvest among the probes it takes. -/
lemma harvestFirst_take_walkLen (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α)
    (site : Random.Level3Site) (T : Set (FreeMonoid α)) (xs : List (FreeMonoid α))
    (h : (Random.harvestFirst read k H site T xs).isSome) :
    (Random.harvestFirst read k H site T
      (xs.take (walkLen {x | (Random.harvestFirst read k H site T [x]).isSome} xs))).isSome :=
  findSome_take_walkLen _ xs h

omit [DecidableEq α] in
lemma measurable_popWalk (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k N M : ℕ)
    (s : Option (SpecSampler α)) (T : Finset (FreeMonoid α)) :
    Measurable (popWalk D k N M s T) := by
  rcases s with _ | s
  · exact measurable_const
  · exact (measurable_of_countable (fun z : Fin M → Fin N → FreeMonoid α =>
      ∑ n : Fin M, walkLen (hitSet s k T) (List.ofFn (z n)))).comp (measurable_probeBlocks D N M)

omit [DecidableEq α] in
lemma measurable_harvestWalk {K : ℕ} (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N M : ℕ) (pops : List (SpecSampler α)) (T : Finset (FreeMonoid α)) :
    Measurable (harvestWalk (K := K) D k N M pops T) :=
  Finset.measurable_sum _ fun i _ =>
    (measurable_popWalk D k N M _ T).comp (measurable_pi_apply (some i))

omit [DecidableEq α] in
lemma harvestWalk_le {K : ℕ} (D : Measure (FreeMonoid α)) (k N M : ℕ)
    (pops : List (SpecSampler α)) (T : Finset (FreeMonoid α))
    (u : Option (Fin K) → ℕ → unitInterval × (Fin N → unitInterval)) :
    harvestWalk D k N M pops T u ≤ K * (M * N) := by
  unfold harvestWalk
  refine (Finset.sum_le_sum fun i _ => (?_ : popWalk D k N M pops[i.val]? T (u (some i))
    ≤ M * N)).trans (by simp)
  rcases pops[i.val]? with _ | s
  · exact Nat.zero_le _
  · calc ∑ n : Fin M, walkLen (hitSet s k T) (List.ofFn (probeBlocks D N M (u (some i)) n))
        ≤ ∑ _n : Fin M, N :=
          Finset.sum_le_sum fun n _ => (walkLen_le_length _ _).trans (by simp)
      _ = M * N := by simp

omit [DecidableEq α] in
/-- Each probe of a harvested population harvesting with chance at least `y`, its `M` draws'
walks `W_i` satisfy

    E[(2/(2 - y))^W_i] ≤ 2^M. -/
lemma integral_popWalk_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k N M : ℕ)
    (s : Option (SpecSampler α)) (T : Finset (FreeMonoid α)) {y : ℝ} (hy0 : 0 ≤ y) (hy1 : y ≤ 1)
    (hyield : ∀ s' ∈ s, y ≤ (Random.harvestLaw s'.read D k s'.hyp s'.site).real (↑T)ᶜ) :
    ∫ v, (2 / (2 - y)) ^ popWalk D k N M s T v ∂(Measure.infinitePi fun _ : ℕ => drawLaw N)
      ≤ 2 ^ M := by
  set c : ℝ := 2 / (2 - y) with hc
  rcases s with _ | s
  · simp only [popWalk, pow_zero, integral_const, probReal_univ, smul_eq_mul, mul_one]
    exact one_le_pow₀ (by norm_num)
  have hy := (hyield s rfl).trans_eq (real_hitSet s D k T).symm
  set H : (Fin M → Fin N → FreeMonoid α) → ℝ := fun z =>
    ∏ n : Fin M, c ^ walkLen (hitSet s k T) (List.ofFn (z n))
  have he : ∀ v, c ^ popWalk D k N M (some s) T v = H (probeBlocks D N M v) := fun v => by
    simp only [popWalk, H, Finset.prod_pow_eq_pow_sum]
  simp only [he]
  rw [← integral_map (measurable_probeBlocks D N M).aemeasurable
    (measurable_of_countable _).aestronglyMeasurable, map_probeBlocks]
  simp only [H]
  rw [integral_fintype_prod_eq_prod (f := fun _ z => c ^ walkLen (hitSet s k T) (List.ofFn z))]
  calc ∏ _n : Fin M, ∫ z, c ^ walkLen (hitSet s k T) (List.ofFn z) ∂(Measure.pi fun _ : Fin N => D)
      ≤ ∏ _n : Fin M, (2 : ℝ) :=
        Finset.prod_le_prod (fun n _ => integral_nonneg fun z =>
          pow_nonneg (by rw [hc]; exact div_nonneg (by norm_num) (by linarith)) _)
          fun n _ => integral_walkLen_le D (hitSet s k T) hy0 hy1 hy N
    _ = 2 ^ M := by simp

omit [DecidableEq α] in
/-- Each probe of each harvested population harvesting with chance at least `y`, the probes its
`K` populations' first `M` draws walk number more than `B` with chance

    P(W > B) ≤ 2^(K M) (1 - y/2)^(B + 1). -/
theorem restLaw_harvestWalk_le {K : ℕ} (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N M : ℕ) (pops : List (SpecSampler α)) (T : Finset (FreeMonoid α)) {y : ℝ} (hy0 : 0 < y)
    (hy1 : y ≤ 1) (hyield : ∀ s ∈ pops, y ≤ (Random.harvestLaw s.read D k s.hyp s.site).real (↑T)ᶜ)
    (B : ℕ) :
    (restLaw α K N).real {e | B < harvestWalk D k N M pops T e.2.1}
      ≤ 2 ^ (K * M) * (1 - y / 2) ^ (B + 1) := by
  set c : ℝ := 2 / (2 - y) with hc
  have h2y : 0 < 2 - y := by linarith
  have hc1 : 1 ≤ c := by rw [hc, le_div_iff₀ h2y]; linarith
  have hc0 : 0 < c := by linarith
  set ν := Measure.infinitePi fun _ : ℕ => drawLaw N
  set π := Measure.pi fun _ : Option (Fin K) => ν
  set W := harvestWalk (K := K) D k N M pops T
  set F : Option (Fin K) → (ℕ → unitInterval × (Fin N → unitInterval)) → ℝ := fun j v =>
    match j with
    | none => 1
    | some i => c ^ popWalk D k N M pops[i.val]? T v
  have hprod : ∀ u, c ^ W u = ∏ j, F j (u j) := fun u => by
    rw [Fintype.prod_option]
    simp only [F, one_mul, W, harvestWalk, Finset.prod_pow_eq_pow_sum]
  have hpop : ∀ i : Fin K, ∀ s' ∈ pops[i.val]?,
      y ≤ (Random.harvestLaw s'.read D k s'.hyp s'.site).real (↑T)ᶜ :=
    fun i s' hs' => hyield s' (List.mem_of_getElem? hs')
  have hF : ∀ j, 0 ≤ ∫ v, F j v ∂ν := fun j => integral_nonneg fun v => by
    rcases j with _ | i
    · simp [F]
    · exact pow_nonneg hc0.le _
  have hmgf : ∫ u, c ^ W u ∂π ≤ 2 ^ (K * M) := by
    simp only [hprod]
    rw [integral_fintype_prod_eq_prod (f := F), Fintype.prod_option]
    have h0 : ∫ v, F none v ∂ν = 1 := by simp [F]
    rw [h0, one_mul, pow_mul', ← Fin.prod_const]
    exact Finset.prod_le_prod (fun i _ => hF (some i))
      fun i _ => integral_popWalk_le D k N M _ T hy0.le hy1 (hpop i)
  have hWm : Measurable W := measurable_harvestWalk D k N M pops T
  have hint : Integrable (fun u => c ^ W u) π :=
    Integrable.of_bound
      ((measurable_of_countable (fun n : ℕ => c ^ n)).comp hWm).aestronglyMeasurable
      (c ^ (K * (M * N))) (Filter.Eventually.of_forall fun u => by
        rw [Real.norm_of_nonneg (pow_nonneg hc0.le _)]
        exact pow_le_pow_right₀ hc1 (harvestWalk_le D k N M pops T u))
  have hmk := mul_meas_ge_le_integral_of_nonneg
    (Filter.Eventually.of_forall fun u => pow_nonneg hc0.le (W u)) hint (c ^ (B + 1))
  have hsub : {u | B < W u} ⊆ {u | c ^ (B + 1) ≤ c ^ W u} := fun u hu =>
    pow_le_pow_right₀ hc1 (Nat.succ_le_of_lt hu)
  have hπ : π.real {u | B < W u} ≤ 2 ^ (K * M) * (1 - y / 2) ^ (B + 1) := by
    have hcB : 0 < c ^ (B + 1) := pow_pos hc0 _
    have h1 : π.real {u | B < W u} ≤ π.real {u | c ^ (B + 1) ≤ c ^ W u} :=
      measureReal_mono hsub (measure_ne_top _ _)
    have h2 : π.real {u | c ^ (B + 1) ≤ c ^ W u} ≤ 2 ^ (K * M) / c ^ (B + 1) := by
      rw [le_div_iff₀ hcB, mul_comm]
      exact hmk.trans hmgf
    have h3 : (2 : ℝ) ^ (K * M) / c ^ (B + 1) = 2 ^ (K * M) * (1 - y / 2) ^ (B + 1) := by
      rw [div_eq_mul_inv, ← inv_pow]
      congr 2
      rw [hc]
      field_simp
    linarith
  have hmp : MeasurePreserving (fun e : ClusterRest α K N => e.2.1) (restLaw α K N) π :=
    measurePreserving_fst.comp measurePreserving_snd
  have hS : MeasurableSet {u | B < W u} := hWm (MeasurableSet.of_discrete (s := Set.Ioi B))
  exact (hmp.measureReal_preimage hS.nullMeasurableSet).trans_le hπ

/-- Alike samplers walk alike, on probes at most `L` long. -/
lemma harvestWalk_alike {K L : ℕ} {pops pops' : List (SpecSampler α)}
    (hp : List.Forall₂ (SamplerAlike L) pops pops') (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (hD : ∀ᵐ x ∂D, x.toList.length ≤ L) (k N M : ℕ)
    (T : Finset (FreeMonoid α)) (u : Option (Fin K) → ℕ → unitInterval × (Fin N → unitInterval)) :
    harvestWalk D k N M pops T u = harvestWalk D k N M pops' T u := by
  unfold harvestWalk
  refine Finset.sum_congr rfl fun i _ => ?_
  have hlen := hp.length_eq
  by_cases hi : i.val < pops.length
  · rw [List.getElem?_eq_getElem hi, List.getElem?_eq_getElem (hlen ▸ hi)]
    have hs := SamplerAlike.of_forall₂ hp hi
    simp only [popWalk]
    refine Finset.sum_congr rfl fun n _ => walkLen_congr fun x hx => ?_
    obtain ⟨t, rfl⟩ := List.mem_ofFn.1 hx
    simp only [hitSet, Set.mem_setOf_eq]
    rw [harvestFirst_congr_zone hs k (↑T) _ fun z hz => by
      rw [List.mem_singleton.1 hz]; exact drawFrom_length_le D hD _]
  · rw [List.getElem?_eq_none (not_lt.1 hi), List.getElem?_eq_none (hlen ▸ not_lt.1 hi)]

end Round

end OrthoDFA

end
