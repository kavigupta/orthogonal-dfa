import OrthoDFA.Proofs.Level4Avoid
import OrthoDFA.Proofs.LoopLaw

/-!
# From the avoiding probes back to all of them

Level 4 draws its probes from `D` conditioned on avoiding a set `K₀` of first `k` letters, which
removes `m` of `D`. Two reads agreeing wherever `KeyOff` holds search and harvest the same at every
avoiding probe, so a claim about one read over the conditioned probes is a claim about the other
over all of `D` up to the removed mass: the disagreement grows by `m`, the sampler's yield shrinks
by the factor `1 − m`, and the removed probes, each harvesting at most `L + 1` strings, leave more
than a quarter of the sampler at bad states once `(L + 1) m ≤ (1 − m) y`
(`quarterEndsWell_of_endsWell`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

open scoped Classical in
/-- `EndsWell` with a quarter of the sampler's law at bad states, rather than a half. -/
def QuarterEndsWell {σ H : Type*} (M : _root_.DFA α σ) (bad : σ → Prop)
    (D : Measure (FreeMonoid α)) (ε : ℝ) (disagree : Set (FreeMonoid α)) (k : ℕ)
    (law : H → Measure (FreeMonoid α)) (y : ℝ) : RoundEnd H → Prop
  | .consistent => D.real disagree ≤ ε
  | .harvest h => (law h).real Set.univ < 4 * (law h).real {z | bad (M.eval z.toList)}
      ∧ Spread D k (law h) ∧ Yields (law h) y
  | .failed => False

/-- The probes level 4 may draw. -/
def avoiding (k : ℕ) (K₀ : Set (FreeMonoid α)) : Set (FreeMonoid α) := {x | Avoids k K₀ x}

open scoped Classical in
/-- Over the versions, the strings in `S` a probe harvests, summed over `ν`. -/
def siteSum (ν : Measure (FreeMonoid α)) (cut : FreeMonoid α → Option Bool) (k : ℕ)
    (h : Level4Harvest α) (S : Set (FreeMonoid α)) : ℝ :=
  (h.versions.map fun v => ∫ x,
    (((siteHarvest cut k h.site v x).filter (· ∈ S)).length : ℝ) ∂ν).sum

variable {k : ℕ} {K₀ : Set (FreeMonoid α)}

omit [Fintype α] [DecidableEq α] in
lemma list_sum_map_add' {β : Type*} (l : List β) (f g : β → ℝ) :
    (l.map fun v => f v + g v).sum = (l.map f).sum + (l.map g).sum := by
  induction l with
  | nil => simp
  | cons a l ih => simp [ih]; ring

lemma harvestLaw_real_siteSum (ν : Measure (FreeMonoid α)) [IsFiniteMeasure ν] {L : ℕ}
    (hlen : ∀ᵐ x ∂ν, x.toList.length ≤ L) (cut : FreeMonoid α → Option Bool)
    (h : Level4Harvest α) (S : Set (FreeMonoid α)) :
    (harvestLaw ν cut k h).real S = siteSum ν cut k h S / h.versions.length := by
  classical
  rw [harvestLaw_real ν cut k hlen h S]
  rfl

lemma siteSum_split (D : Measure (FreeMonoid α)) [IsFiniteMeasure D] {L : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (cut : FreeMonoid α → Option Bool)
    (h : Level4Harvest α) (S A : Set (FreeMonoid α)) :
    siteSum D cut k h S = siteSum (D.restrict A) cut k h S + siteSum (D.restrict Aᶜ) cut k h S := by
  classical
  unfold siteSum
  rw [← list_sum_map_add']
  congr 1
  refine List.map_congr_left fun v _ => ?_
  exact (integral_add_compl (measurableSet_of_countable A)
    (integrable_siteCount D cut k hlen h.site v _)).symm

lemma siteSum_restrict_congr (D : Measure (FreeMonoid α)) {cut cut' : FreeMonoid α → Option Bool}
    (hc : AgreeOff k K₀ cut cut') (h : Level4Harvest α) (S : Set (FreeMonoid α)) :
    siteSum (D.restrict (avoiding k K₀)) cut k h S
      = siteSum (D.restrict (avoiding k K₀)) cut' k h S := by
  classical
  unfold siteSum
  congr 1
  refine List.map_congr_left fun v _ => ?_
  refine setIntegral_congr_fun (measurableSet_of_countable _) fun x hx => ?_
  simp only [siteHarvest_congr_off hc h.site v hx]

lemma siteSum_smul (ν : Measure (FreeMonoid α)) (c : ENNReal) (cut : FreeMonoid α → Option Bool)
    (h : Level4Harvest α) (S : Set (FreeMonoid α)) :
    siteSum (c • ν) cut k h S = c.toReal * siteSum ν cut k h S := by
  classical
  unfold siteSum
  simp only [integral_smul_measure, smul_eq_mul]
  induction h.versions with
  | nil => simp
  | cons a l ih => simp [ih]; ring

lemma siteSum_nonneg (ν : Measure (FreeMonoid α)) (cut : FreeMonoid α → Option Bool)
    (h : Level4Harvest α) (S : Set (FreeMonoid α)) : 0 ≤ siteSum ν cut k h S :=
  List.sum_nonneg fun a ha => by
    obtain ⟨v, -, rfl⟩ := List.mem_map.1 ha
    exact integral_nonneg fun _ => Nat.cast_nonneg _

lemma siteSum_le (ν : Measure (FreeMonoid α)) [IsFiniteMeasure ν] {L : ℕ}
    (hlen : ∀ᵐ x ∂ν, x.toList.length ≤ L) (cut : FreeMonoid α → Option Bool)
    (h : Level4Harvest α) (S : Set (FreeMonoid α)) :
    siteSum ν cut k h S ≤ h.versions.length * ((L + 1) * ν.real Set.univ) := by
  classical
  unfold siteSum
  have hv : ∀ v : DTree α × EdgeTable α, ∫ x,
      (((siteHarvest cut k h.site v x).filter (· ∈ S)).length : ℝ) ∂ν ≤ (L + 1) * ν.real Set.univ := by
    intro v
    have := integral_mono_ae (integrable_siteCount ν cut k hlen h.site v (· ∈ S))
      (integrable_const ((L : ℝ) + 1)) (by
        filter_upwards [hlen] with x hx
        have h1 := List.length_filter_le (· ∈ S) (siteHarvest cut k h.site v x)
        have h2 := siteHarvest_length_le cut k x h.site v
        exact_mod_cast (by omega : ((siteHarvest cut k h.site v x).filter (· ∈ S)).length ≤ L + 1))
    rw [integral_const, smul_eq_mul] at this
    linarith
  induction h.versions with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    push_cast
    nlinarith [hv a, ih]

lemma searchingProbes_congr_off {read read' : FreeMonoid α → ARU}
    (hc : AgreeOff k K₀ (fun z => (read z).cut) (fun z => (read' z).cut)) (H : Hypothesis α)
    {x : FreeMonoid α} (hx : x ∈ avoiding k K₀) :
    x ∈ searchingProbes read k H ↔ x ∈ searchingProbes read' k H := by
  unfold searchingProbes
  simp only [Set.mem_setOf_eq, probeBy_congr_off hc H.tree (Function.curry H.edges) hx]

/-- A claim at the avoiding probes, through a read agreeing off `K₀`, is a weaker claim at all of
`D`. -/
theorem quarterEndsWell_of_endsWell (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) {σ : Type*} (M : _root_.DFA α σ)
    (bad : σ → Prop) {read read' : FreeMonoid α → ARU}
    (hc : AgreeOff k K₀ (fun z => (read z).cut) (fun z => (read' z).cut)) (H : Hypothesis α)
    (ε y : ℝ) (hy : 0 ≤ y) (e : RoundEnd (Level4Harvest α))
    (hm : D.real (avoiding k K₀)ᶜ ≤ 1 / 2)
    (hmc : (L + 1) * D.real (avoiding k K₀)ᶜ ≤ (1 - D.real (avoiding k K₀)ᶜ) * y)
    (h : EndsWell M bad (popLaw (D.restrict (avoiding k K₀))) ε (searchingProbes read' k H) k
      (harvestLaw (popLaw (D.restrict (avoiding k K₀))) (fun z => (read' z).cut) k) y e) :
    QuarterEndsWell M bad D (ε + D.real (avoiding k K₀)ᶜ) (searchingProbes read k H) k
      (harvestLaw D (fun z => (read z).cut) k) ((1 - D.real (avoiding k K₀)ᶜ) * y) e := by
  classical
  set A := avoiding (α := α) k K₀
  set m := D.real Aᶜ with hmd
  have hA : D.real A = 1 - m := by
    rw [hmd, probReal_compl_eq_one_sub (measurableSet_of_countable A)]; ring
  have hApos : 0 < D.real A := by rw [hA]; linarith
  have hA0 : D.restrict A Set.univ ≠ 0 := by
    rw [Measure.restrict_apply_univ]
    intro h0
    rw [measureReal_def, h0, ENNReal.toReal_zero] at hApos
    exact lt_irrefl _ hApos
  have hAtop : D.restrict A Set.univ ≠ ⊤ := measure_ne_top _ _
  -- the conditioned probes
  set D' := popLaw (D.restrict A) with hD'
  have hD'eq : D' = (D.restrict A Set.univ)⁻¹ • D.restrict A := by
    rw [hD']; unfold popLaw; rw [if_neg (by tauto)]
  have hinv : ((D.restrict A Set.univ)⁻¹).toReal = (1 - m)⁻¹ := by
    rw [ENNReal.toReal_inv, Measure.restrict_apply_univ]
    rw [show (D A).toReal = D.real A from rfl, hA]
  have hlen' : ∀ᵐ x ∂D', x.toList.length ≤ L := by
    rw [hD'eq]
    exact Measure.ae_smul_measure (ae_restrict_of_ae hlen) _
  have hlenA : ∀ᵐ x ∂(D.restrict A), x.toList.length ≤ L := ae_restrict_of_ae hlen
  have hlenC : ∀ᵐ x ∂(D.restrict Aᶜ), x.toList.length ≤ L := ae_restrict_of_ae hlen
  set cut := fun z => (read z).cut
  set cut' := fun z => (read' z).cut
  -- the sums at `D` against those at `D'`
  have hsum : ∀ (hv : Level4Harvest α) (S : Set (FreeMonoid α)),
      siteSum D cut k hv S = (1 - m) * siteSum D' cut' k hv S + siteSum (D.restrict Aᶜ) cut k hv S := by
    intro hv S
    have hc' : siteSum (D.restrict A) cut k hv S = siteSum (D.restrict A) cut' k hv S :=
      siteSum_restrict_congr D hc hv S
    rw [siteSum_split D hlen cut hv S A, hc', hD'eq, siteSum_smul, hinv]
    field_simp [show (1 - m) ≠ 0 by linarith]
  cases e with
  | consistent =>
    show D.real (searchingProbes read k H) ≤ ε + m
    have h' : D'.real (searchingProbes read' k H) ≤ ε := h
    rw [hD'eq] at h'
    simp only [measureReal_def, Measure.smul_apply, smul_eq_mul, ENNReal.toReal_mul] at h'
    rw [hinv, Measure.restrict_apply (measurableSet_of_countable _), ← measureReal_def] at h'
    have hsplit := measureReal_inter_add_sdiff₀ (μ := D) (s := searchingProbes read k H)
      (measurableSet_of_countable A).nullMeasurableSet (measure_ne_top _ _)
    have hinter : searchingProbes read k H ∩ A = searchingProbes read' k H ∩ A := by
      ext x
      simp only [Set.mem_inter_iff]
      exact ⟨fun ⟨h1, h2⟩ => ⟨(searchingProbes_congr_off hc H h2).1 h1, h2⟩,
        fun ⟨h1, h2⟩ => ⟨(searchingProbes_congr_off hc H h2).2 h1, h2⟩⟩
    have hdiff : D.real (searchingProbes read k H \ A) ≤ m :=
      measureReal_mono (fun x hx => hx.2) (measure_ne_top _ _)
    have h1m : 0 < 1 - m := by linarith
    rw [hinter] at hsplit
    have hε : 0 ≤ ε := le_trans (mul_nonneg (inv_nonneg.2 h1m.le) measureReal_nonneg) h'
    have : D.real (searchingProbes read' k H ∩ A) ≤ ε * (1 - m) := by
      rw [inv_mul_le_iff₀ h1m] at h'
      linarith
    have hm0 : 0 ≤ m := measureReal_nonneg
    have hεm : ε * (1 - m) ≤ ε := by nlinarith
    linarith
  | harvest hv =>
    obtain ⟨hbad, -, hyield⟩ := h
    unfold Yields at hyield
    set Bset := {z : FreeMonoid α | bad (M.eval z.toList)}
    rw [harvestLaw_real_siteSum D' hlen' cut' hv Set.univ,
      harvestLaw_real_siteSum D' hlen' cut' hv Bset] at hbad
    rw [harvestLaw_real_siteSum D' hlen' cut' hv Set.univ] at hyield
    set N : ℝ := (hv.versions.length : ℝ) with hNd
    set U' := siteSum D' cut' k hv Set.univ
    set B' := siteSum D' cut' k hv Bset
    have hN : 0 < N := by
      rcases Nat.eq_zero_or_pos hv.versions.length with h0 | h0
      · exfalso
        have : N = 0 := by simp [N, h0]
        rw [this, div_zero, div_zero, mul_zero] at hbad
        exact lt_irrefl _ hbad
      · rw [hNd]; exact_mod_cast h0
    have hbad' : U' < 2 * B' := by
      have h2 : U' / N < (2 * B') / N := by rwa [mul_div_assoc]
      exact (div_lt_div_iff_of_pos_right hN).1 h2
    have hyN : y * N ≤ U' := (le_div_iff₀ hN).1 hyield
    have hRU := siteSum_le (k := k) (D.restrict Aᶜ) hlenC cut hv Set.univ
    rw [measureReal_restrict_apply_univ] at hRU
    have hRB := siteSum_nonneg (k := k) (D.restrict Aᶜ) cut hv Bset
    set RU := siteSum (D.restrict Aᶜ) cut k hv Set.univ
    set RB := siteSum (D.restrict Aᶜ) cut k hv Bset
    have h1m : 0 < 1 - m := by linarith
    -- the removed probes harvest no more than the kept ones' share
    have hRU0' := siteSum_nonneg (k := k) (D.restrict Aᶜ) cut hv Set.univ
    have hR : RU ≤ (1 - m) * U' := by
      have e1 : N * ((L + 1) * m) ≤ N * ((1 - m) * y) := mul_le_mul_of_nonneg_left hmc hN.le
      have e2 : (1 - m) * (y * N) ≤ (1 - m) * U' := mul_le_mul_of_nonneg_left hyN h1m.le
      nlinarith only [hRU, e1, e2]
    have hB2 : (1 - m) * U' < (1 - m) * (2 * B') := mul_lt_mul_of_pos_left hbad' h1m
    show (harvestLaw D cut k hv).real Set.univ < 4 * (harvestLaw D cut k hv).real Bset
      ∧ Spread D k (harvestLaw D cut k hv) ∧ (1 - m) * y ≤ (harvestLaw D cut k hv).real Set.univ
    rw [harvestLaw_real_siteSum D hlen cut hv Set.univ, harvestLaw_real_siteSum D hlen cut hv Bset,
      hsum, hsum]
    refine ⟨?_, harvestLaw_spread D cut k hv, ?_⟩
    · rw [← mul_div_assoc, div_lt_div_iff_of_pos_right hN]
      nlinarith only [hR, hB2, hRB]
    · rw [le_div_iff₀ hN]
      have e2 : (1 - m) * (y * N) ≤ (1 - m) * U' := mul_le_mul_of_nonneg_left hyN h1m.le
      nlinarith only [e2, hRU0', hRB]
  | failed => exact h

end OrthoDFA
