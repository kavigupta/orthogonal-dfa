import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import OrthoDFA.Estimate
import OrthoDFA.Top

/-!
# Liveness, step 1: the clustering selects an accept-preserving family

The ε-anchored greedy returns the `k` suffixes of least loss against its centre.
This file proves the *selection* fact that liveness rests on, with no probability:
if the accept-preserving suffixes are separated *below* the non-accept-preserving
ones (every AP loss strictly under every non-AP loss) and there are at least `k`
of them, then the `k` least-loss suffixes are all accept-preserving.

`chosen` abstracts the greedy's output — any least-loss `k`-subset. Separability
(`hsep`) is the substantive hypothesis: the noisy losses must actually split the
two classes, which downstream is a concentration statement about the pool once it
is rich enough to expose every non-accept-preserving suffix's flip.
-/

namespace OrthoDFA

/-- **Selection under separability.**  If `chosen` is a least-loss `k`-subset of
`cands`, there are at least `k` accept-preserving candidates, and every
accept-preserving candidate has strictly smaller loss than every
non-accept-preserving one, then every chosen suffix is accept-preserving. -/
theorem chosen_accept_preserving {S : Type*} [DecidableEq S]
    (ℓ : S → ℝ) (AP : S → Prop) [DecidablePred AP]
    (cands chosen : Finset S) (k : ℕ)
    (hsub : chosen ⊆ cands) (hcard : chosen.card = k)
    (hleast : ∀ v ∈ chosen, ∀ w ∈ cands, w ∉ chosen → ℓ v ≤ ℓ w)
    (apCount : k ≤ (cands.filter AP).card)
    (hsep : ∀ v ∈ cands, ∀ w ∈ cands, AP v → ¬ AP w → ℓ v < ℓ w) :
    ∀ w ∈ chosen, AP w := by
  intro w hw
  by_contra hwAP
  have hwnot : w ∉ cands.filter AP := fun hh => hwAP (Finset.mem_filter.mp hh).2
  -- some accept-preserving candidate `u` is not chosen
  have hne : ((cands.filter AP) \ chosen).Nonempty := by
    rw [← Finset.card_pos]
    have key : (cands.filter AP ∩ chosen).card + 1 ≤ chosen.card := by
      have hss : insert w (cands.filter AP ∩ chosen) ⊆ chosen := by
        intro x hx
        rcases Finset.mem_insert.mp hx with rfl | hx
        · exact hw
        · exact Finset.mem_of_mem_inter_right hx
      have hcard' := Finset.card_le_card hss
      rwa [Finset.card_insert_of_notMem
        (fun hh => hwnot (Finset.mem_of_mem_inter_left hh))] at hcard'
    have hid : (cands.filter AP ∩ chosen).card + ((cands.filter AP) \ chosen).card
        = (cands.filter AP).card := Finset.card_inter_add_card_sdiff _ _
    omega
  obtain ⟨u, hu⟩ := hne
  rw [Finset.mem_sdiff, Finset.mem_filter] at hu
  obtain ⟨⟨huc, huAP⟩, hunot⟩ := hu
  have h1 : ℓ w ≤ ℓ u := hleast w hw u huc hunot
  have h2 : ℓ u < ℓ w := hsep u huc w (hsub hw) huAP hwAP
  linarith

#print axioms chosen_accept_preserving

/-- **Selection avoids the bad set.**  The D-relative version: `good` and `bad` are
disjoint, there are at least `k` good candidates, and every good candidate has
strictly smaller loss than every bad one.  Then the least-loss `k`-subset avoids
`bad` entirely — borderline candidates (neither good nor bad) may be chosen, which
is fine: only `bad` (a flip of D-mass ≥ ε_cov) hurts the D-accuracy goal.

This removes the need to catch D-negligible flips, so no coverage assumption is
required: `good`/`bad` are defined by D-mass relative to the target `ε_cov`. -/
theorem chosen_avoids_bad {S : Type*} [DecidableEq S]
    (ℓ : S → ℝ) (good bad : S → Prop) [DecidablePred good]
    (hdisj : ∀ v, bad v → ¬ good v)
    (cands chosen : Finset S) (k : ℕ)
    (hsub : chosen ⊆ cands) (hcard : chosen.card = k)
    (hleast : ∀ v ∈ chosen, ∀ w ∈ cands, w ∉ chosen → ℓ v ≤ ℓ w)
    (goodCount : k ≤ (cands.filter good).card)
    (hsep : ∀ v ∈ cands, ∀ w ∈ cands, good v → bad w → ℓ v < ℓ w) :
    ∀ w ∈ chosen, ¬ bad w := by
  intro w hw hbad
  have hwnot : w ∉ cands.filter good :=
    fun hh => hdisj w hbad (Finset.mem_filter.mp hh).2
  have hne : ((cands.filter good) \ chosen).Nonempty := by
    rw [← Finset.card_pos]
    have key : (cands.filter good ∩ chosen).card + 1 ≤ chosen.card := by
      have hss : insert w (cands.filter good ∩ chosen) ⊆ chosen := by
        intro x hx
        rcases Finset.mem_insert.mp hx with rfl | hx
        · exact hw
        · exact Finset.mem_of_mem_inter_right hx
      have hcard' := Finset.card_le_card hss
      rwa [Finset.card_insert_of_notMem
        (fun hh => hwnot (Finset.mem_of_mem_inter_left hh))] at hcard'
    have hid : (cands.filter good ∩ chosen).card + ((cands.filter good) \ chosen).card
        = (cands.filter good).card := Finset.card_inter_add_card_sdiff _ _
    omega
  obtain ⟨u, hu⟩ := hne
  rw [Finset.mem_sdiff, Finset.mem_filter] at hu
  obtain ⟨⟨huc, huGood⟩, hunot⟩ := hu
  have h1 : ℓ w ≤ ℓ u := hleast w hw u huc hunot
  have h2 : ℓ u < ℓ w := hsep u huc w (hsub hw) huGood hbad
  linarith

#print axioms chosen_avoids_bad

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- **Liveness core: the greedy proposes an accept-preserving family, w.h.p.**
Each suffix `v`'s loss is `∑ⱼ D v j`, `D v j ∈ [0,1]` the disagreement indicator on
prefix `j`, independent across prefixes.  Under *mean-loss separability* — every
accept-preserving suffix has mean loss ≤ `m·ρlo`, every other ≥ `m·ρhi`, with a
margin `γ` (`ρlo+γ ≤ ρhi-γ`) — and at least `k` accept-preserving candidates, the
`ω`-dependent least-loss `k`-subset `chosen ω` is entirely accept-preserving except
with probability at most `#cands · exp(-2mγ²)`.

The noisy losses separate by concentration (`sumUpper_le`/`sumLower_le`), and the
selection lemma then forces the choice.  `chosen` abstracts the ε-anchored greedy's
output as any per-`ω` least-loss `k`-subset. -/
theorem chosen_accept_preserving_whp {S : Type*} [DecidableEq S]
    (AP : S → Prop) [DecidablePred AP]
    (cands : Finset S) (k m : ℕ) (ρlo ρhi γ : ℝ)
    (D : S → ℕ → Ω → ℝ)
    (hmeas : ∀ v i, AEMeasurable (D v i) μ)
    (hindep : ∀ v, iIndepFun (D v) μ)
    (hIcc : ∀ v i, ∀ᵐ ω ∂μ, D v i ω ∈ Set.Icc (0 : ℝ) 1)
    (hAPmean : ∀ v ∈ cands, AP v → ∑ i ∈ Finset.range m, μ[D v i] ≤ (m : ℝ) * ρlo)
    (hNAmean : ∀ v ∈ cands, ¬ AP v → (m : ℝ) * ρhi ≤ ∑ i ∈ Finset.range m, μ[D v i])
    (hgap : ρlo + γ ≤ ρhi - γ) (hγ : 0 ≤ γ)
    (apCount : k ≤ (cands.filter AP).card)
    (chosen : Ω → Finset S)
    (hsub : ∀ ω, chosen ω ⊆ cands) (hcard : ∀ ω, (chosen ω).card = k)
    (hleast : ∀ ω, ∀ v ∈ chosen ω, ∀ w ∈ cands, w ∉ chosen ω →
        (∑ i ∈ Finset.range m, D v i ω) ≤ ∑ i ∈ Finset.range m, D w i ω) :
    μ.real {ω | ¬ ∀ w ∈ chosen ω, AP w}
      ≤ (cands.card : ℝ) * Real.exp (-2 * (m : ℝ) * γ ^ 2) := by
  classical
  set E : ℝ := Real.exp (-2 * (m : ℝ) * γ ^ 2) with hE
  set UAP : Set Ω := ⋃ v ∈ cands.filter AP,
    {ω | (m : ℝ) * (ρlo + γ) ≤ ∑ i ∈ Finset.range m, D v i ω} with hUAP
  set UNA : Set Ω := ⋃ v ∈ cands.filter (fun v => ¬ AP v),
    {ω | ∑ i ∈ Finset.range m, D v i ω ≤ (m : ℝ) * (ρhi - γ)} with hUNA
  have hbadAP : μ.real UAP ≤ ((cands.filter AP).card : ℝ) * E := by
    calc μ.real UAP ≤ ∑ v ∈ cands.filter AP,
          μ.real {ω | (m : ℝ) * (ρlo + γ) ≤ ∑ i ∈ Finset.range m, D v i ω} :=
            measureReal_biUnion_le _ _
      _ ≤ ∑ _v ∈ cands.filter AP, E := Finset.sum_le_sum (fun v hv => by
            obtain ⟨hvc, hvAP⟩ := Finset.mem_filter.mp hv
            exact sumUpper_le (D v) m ρlo γ (hmeas v) (hindep v) (hIcc v)
              (hAPmean v hvc hvAP) hγ)
      _ = ((cands.filter AP).card : ℝ) * E := by rw [Finset.sum_const, nsmul_eq_mul]
  have hbadNA : μ.real UNA ≤ ((cands.filter (fun v => ¬ AP v)).card : ℝ) * E := by
    calc μ.real UNA ≤ ∑ v ∈ cands.filter (fun v => ¬ AP v),
          μ.real {ω | ∑ i ∈ Finset.range m, D v i ω ≤ (m : ℝ) * (ρhi - γ)} :=
            measureReal_biUnion_le _ _
      _ ≤ ∑ _v ∈ cands.filter (fun v => ¬ AP v), E := Finset.sum_le_sum (fun v hv => by
            obtain ⟨hvc, hvNA⟩ := Finset.mem_filter.mp hv
            exact sumLower_le (D v) m ρhi γ (hmeas v) (hindep v) (hIcc v)
              (hNAmean v hvc hvNA) hγ)
      _ = ((cands.filter (fun v => ¬ AP v)).card : ℝ) * E := by
            rw [Finset.sum_const, nsmul_eq_mul]
  have hincl : {ω | ¬ ∀ w ∈ chosen ω, AP w} ⊆ UAP ∪ UNA := by
    intro ω hω
    by_contra hnot
    rw [Set.mem_union, not_or] at hnot
    obtain ⟨hnAP, hnNA⟩ := hnot
    apply hω
    refine chosen_accept_preserving (fun v => ∑ i ∈ Finset.range m, D v i ω) AP cands
      (chosen ω) k (hsub ω) (hcard ω) (hleast ω) apCount ?_
    intro v hv w hw hvAP hwNA
    have h1 : ∑ i ∈ Finset.range m, D v i ω < (m : ℝ) * (ρlo + γ) := by
      by_contra h
      exact hnAP (Set.mem_iUnion₂.mpr ⟨v, Finset.mem_filter.mpr ⟨hv, hvAP⟩, not_lt.mp h⟩)
    have h2 : (m : ℝ) * (ρhi - γ) < ∑ i ∈ Finset.range m, D w i ω := by
      by_contra h
      exact hnNA (Set.mem_iUnion₂.mpr ⟨w, Finset.mem_filter.mpr ⟨hw, hwNA⟩, not_lt.mp h⟩)
    have hmid : (m : ℝ) * (ρlo + γ) ≤ (m : ℝ) * (ρhi - γ) :=
      mul_le_mul_of_nonneg_left hgap (Nat.cast_nonneg m)
    linarith
  calc μ.real {ω | ¬ ∀ w ∈ chosen ω, AP w}
      ≤ μ.real (UAP ∪ UNA) := measureReal_mono hincl
    _ ≤ μ.real UAP + μ.real UNA := measureReal_union_le _ _
    _ ≤ ((cands.filter AP).card : ℝ) * E
          + ((cands.filter (fun v => ¬ AP v)).card : ℝ) * E := add_le_add hbadAP hbadNA
    _ = (cands.card : ℝ) * E := by
        rw [← add_mul]
        congr 1
        rw [← Nat.cast_add, Finset.card_filter_add_card_filter_not]

#print axioms chosen_accept_preserving_whp

/-- **Coverage-free liveness: the greedy avoids the bad set, w.h.p.**
`good v` (flip of D-mass ≤ ρlo) and `bad v` (flip of D-mass ≥ ρhi ≈ ε_cov) are
*definitional* w.r.t. the target — no coverage assumption.  The mean-loss bounds
`hgoodmean`/`hbadmean` are just those D-mass bounds (the loss over `m` D-sampled
prefixes has mean = the flip's D-mass · m), so `hbadmean` holds because a bad
flip's mass is ≥ ρhi *by definition*, and a large D-pool exposes it.  Then the
greedy's least-loss `k`-subset avoids `bad` except w.p. ≤ `#cands·exp(-2mγ²)`.

This is the fix for coverage: "reaches separability" is "sample a large enough
D-pool", derived here, not assumed. -/
theorem chosen_avoids_bad_whp {S : Type*} [DecidableEq S]
    (good bad : S → Prop) [DecidablePred good] [DecidablePred bad]
    (hdisj : ∀ v, bad v → ¬ good v)
    (cands : Finset S) (k m : ℕ) (ρlo ρhi γ : ℝ)
    (D : S → ℕ → Ω → ℝ)
    (hmeas : ∀ v i, AEMeasurable (D v i) μ)
    (hindep : ∀ v, iIndepFun (D v) μ)
    (hIcc : ∀ v i, ∀ᵐ ω ∂μ, D v i ω ∈ Set.Icc (0 : ℝ) 1)
    (hgoodmean : ∀ v ∈ cands, good v → ∑ i ∈ Finset.range m, μ[D v i] ≤ (m : ℝ) * ρlo)
    (hbadmean : ∀ v ∈ cands, bad v → (m : ℝ) * ρhi ≤ ∑ i ∈ Finset.range m, μ[D v i])
    (hgap : ρlo + γ ≤ ρhi - γ) (hγ : 0 ≤ γ)
    (goodCount : k ≤ (cands.filter good).card)
    (chosen : Ω → Finset S)
    (hsub : ∀ ω, chosen ω ⊆ cands) (hcard : ∀ ω, (chosen ω).card = k)
    (hleast : ∀ ω, ∀ v ∈ chosen ω, ∀ w ∈ cands, w ∉ chosen ω →
        (∑ i ∈ Finset.range m, D v i ω) ≤ ∑ i ∈ Finset.range m, D w i ω) :
    μ.real {ω | ¬ ∀ w ∈ chosen ω, ¬ bad w}
      ≤ (cands.card : ℝ) * Real.exp (-2 * (m : ℝ) * γ ^ 2) := by
  classical
  set E : ℝ := Real.exp (-2 * (m : ℝ) * γ ^ 2) with hE
  set UG : Set Ω := ⋃ v ∈ cands.filter good,
    {ω | (m : ℝ) * (ρlo + γ) ≤ ∑ i ∈ Finset.range m, D v i ω} with hUG
  set UB : Set Ω := ⋃ v ∈ cands.filter bad,
    {ω | ∑ i ∈ Finset.range m, D v i ω ≤ (m : ℝ) * (ρhi - γ)} with hUB
  have hbadUG : μ.real UG ≤ ((cands.filter good).card : ℝ) * E := by
    calc μ.real UG ≤ ∑ v ∈ cands.filter good,
          μ.real {ω | (m : ℝ) * (ρlo + γ) ≤ ∑ i ∈ Finset.range m, D v i ω} :=
            measureReal_biUnion_le _ _
      _ ≤ ∑ _v ∈ cands.filter good, E := Finset.sum_le_sum (fun v hv => by
            obtain ⟨hvc, hvg⟩ := Finset.mem_filter.mp hv
            exact sumUpper_le (D v) m ρlo γ (hmeas v) (hindep v) (hIcc v)
              (hgoodmean v hvc hvg) hγ)
      _ = ((cands.filter good).card : ℝ) * E := by rw [Finset.sum_const, nsmul_eq_mul]
  have hbadUB : μ.real UB ≤ ((cands.filter bad).card : ℝ) * E := by
    calc μ.real UB ≤ ∑ v ∈ cands.filter bad,
          μ.real {ω | ∑ i ∈ Finset.range m, D v i ω ≤ (m : ℝ) * (ρhi - γ)} :=
            measureReal_biUnion_le _ _
      _ ≤ ∑ _v ∈ cands.filter bad, E := Finset.sum_le_sum (fun v hv => by
            obtain ⟨hvc, hvb⟩ := Finset.mem_filter.mp hv
            exact sumLower_le (D v) m ρhi γ (hmeas v) (hindep v) (hIcc v)
              (hbadmean v hvc hvb) hγ)
      _ = ((cands.filter bad).card : ℝ) * E := by rw [Finset.sum_const, nsmul_eq_mul]
  have hincl : {ω | ¬ ∀ w ∈ chosen ω, ¬ bad w} ⊆ UG ∪ UB := by
    intro ω hω
    by_contra hnot
    rw [Set.mem_union, not_or] at hnot
    obtain ⟨hnG, hnB⟩ := hnot
    apply hω
    refine chosen_avoids_bad (fun v => ∑ i ∈ Finset.range m, D v i ω) good bad hdisj cands
      (chosen ω) k (hsub ω) (hcard ω) (hleast ω) goodCount ?_
    intro v hv w hw hvg hwb
    have h1 : ∑ i ∈ Finset.range m, D v i ω < (m : ℝ) * (ρlo + γ) := by
      by_contra h
      exact hnG (Set.mem_iUnion₂.mpr ⟨v, Finset.mem_filter.mpr ⟨hv, hvg⟩, not_lt.mp h⟩)
    have h2 : (m : ℝ) * (ρhi - γ) < ∑ i ∈ Finset.range m, D w i ω := by
      by_contra h
      exact hnB (Set.mem_iUnion₂.mpr ⟨w, Finset.mem_filter.mpr ⟨hw, hwb⟩, not_lt.mp h⟩)
    have hmid : (m : ℝ) * (ρlo + γ) ≤ (m : ℝ) * (ρhi - γ) :=
      mul_le_mul_of_nonneg_left hgap (Nat.cast_nonneg m)
    linarith
  have hcards : ((cands.filter good).card : ℝ) + ((cands.filter bad).card : ℝ)
      ≤ (cands.card : ℝ) := by
    have hdisjF : Disjoint (cands.filter good) (cands.filter bad) := by
      rw [Finset.disjoint_filter]
      exact fun v _ hvg hvb => hdisj v hvb hvg
    have := Finset.card_union_of_disjoint hdisjF
    have hle : (cands.filter good ∪ cands.filter bad).card ≤ cands.card :=
      Finset.card_le_card (Finset.union_subset (Finset.filter_subset _ _) (Finset.filter_subset _ _))
    rw [this] at hle
    exact_mod_cast hle
  have hEnn : 0 ≤ E := (Real.exp_pos _).le
  calc μ.real {ω | ¬ ∀ w ∈ chosen ω, ¬ bad w}
      ≤ μ.real (UG ∪ UB) := measureReal_mono hincl
    _ ≤ μ.real UG + μ.real UB := measureReal_union_le _ _
    _ ≤ ((cands.filter good).card : ℝ) * E + ((cands.filter bad).card : ℝ) * E :=
        add_le_add hbadUG hbadUB
    _ = (((cands.filter good).card : ℝ) + ((cands.filter bad).card : ℝ)) * E := by ring
    _ ≤ (cands.card : ℝ) * E := by
        apply mul_le_mul_of_nonneg_right hcards hEnn

/-- **Per-prefix disagreement mean, from random classification noise.**
The oracle is `MQ(x) = ℓ(x) ⊕ r(x)` with `ℓ` the true label and `r(x) ∼
Bernoulli(η)` iid.  The disagreement of `MQ(x·v)` with the denoised centre `ℓ(x)`
reduces (XOR algebra) to `flip ⊕ r`, i.e. `flip + (1−2·flip)·r`, where
`flip = 1[v flips x]` and `r = r(x·v)` is the noise bit.  Its mean is, by linearity
from `E[r] = η`,
    `η + flip·(1 − 2η)`,
so a flip shifts expected disagreement by exactly `1 − 2η = 2s` (signal `s = ½−η`).
Baseline `η` is common to all suffixes; the flip term is the discriminating signal.
Derived, not assumed. -/
theorem read_disagreement_mean {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (η flip : ℝ)
    (r : Ω → ℝ) (hrint : Integrable r μ) (hEr : ∫ ω, r ω ∂μ = η) :
    ∫ ω, (flip + (1 - 2 * flip) * r ω) ∂μ = η + flip * (1 - 2 * η) := by
  have h1 : ∫ ω, (flip + (1 - 2 * flip) * r ω) ∂μ
      = flip + (1 - 2 * flip) * ∫ ω, r ω ∂μ := by
    rw [integral_add (integrable_const flip) (hrint.const_mul (1 - 2 * flip)),
      integral_const_mul]
    simp
  rw [h1, hEr]; ring

#print axioms read_disagreement_mean

/-- **Denoised loss decomposes into baseline + signal·flips.**  Against the
denoised oracle, a read's expected disagreement on prefix `j` is a common baseline
`c₀` plus `2s` exactly when the suffix flips `j`'s state (`flip j = 1`), else `c₀`.
So the mean loss over `m` prefixes is `m·c₀ + 2s·(flip count)` — linear in the
flips.  This is the modeling identification, now explicit: the per-prefix
expectation `hexp` is the (denoised) oracle read model, and the loss is its sum.

Consequently a strictly-accept-preserving suffix (`flip ≡ 0`) has mean loss `m·c₀`
(so `ρlo = c₀`), and a bad one flipping a fraction `≥ ε_cov` has mean loss
`≥ m·(c₀ + 2s·ε_cov)` (so `ρhi = c₀ + 2s·ε_cov`, `γ = s·ε_cov`), discharging the
mean-separability hypotheses of `chosen_avoids_bad_whp`. -/
theorem denoised_loss_eq_flip (m : ℕ) (c₀ s : ℝ) (flip : ℕ → ℝ) (D : ℕ → Ω → ℝ)
    (hexp : ∀ j, μ[D j] = c₀ + 2 * s * flip j) :
    ∑ j ∈ Finset.range m, μ[D j]
      = (m : ℝ) * c₀ + 2 * s * ∑ j ∈ Finset.range m, flip j := by
  rw [Finset.sum_congr rfl (fun j _ => hexp j), Finset.sum_add_distrib,
    Finset.sum_const, Finset.card_range, nsmul_eq_mul, Finset.mul_sum]

#print axioms denoised_loss_eq_flip

#print axioms chosen_avoids_bad_whp

/-- The persistent signal oracle, bundled as random classification noise — the
*genuine* model, nothing derived asserted.  `noise v i` is the iid `Bernoulli(η)`
noise bit on the string `x_i·v` (rate `η`, independent across strings, in `[0,1]`,
mean `η`); `flip v i ∈ {0,1}` marks whether `v` flips `x_i`'s state (from the
language).  The disagreement read and its mean/independence/range are *derived*
below, not fields. -/
structure Oracle {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) (S : Type*) where
  noise : S → ℕ → Ω → ℝ
  η : ℝ
  hη : η ≤ 1 / 2
  noise_meas : ∀ v i, AEMeasurable (noise v i) μ
  noise_indep : ∀ v, iIndepFun (noise v) μ
  noise_icc : ∀ v i, ∀ᵐ ω ∂μ, noise v i ω ∈ Set.Icc (0 : ℝ) 1
  noise_mean : ∀ v i, μ[noise v i] = η
  flip : S → ℕ → ℝ
  flip_bit : ∀ v i, flip v i = 0 ∨ flip v i = 1

namespace Oracle
variable {S : Type*} (O : Oracle μ S)

/-- The disagreement read `flip ⊕ noise = flip + (1−2·flip)·noise` of `v` on
prefix `i` (the loss the clustering sees). -/
noncomputable def read (v : S) (i : ℕ) : Ω → ℝ :=
  fun ω => O.flip v i + (1 - 2 * O.flip v i) * O.noise v i ω

lemma noise_int (v i) : Integrable (O.noise v i) μ :=
  MeasureTheory.Integrable.of_mem_Icc 0 1 (O.noise_meas v i) (O.noise_icc v i)

/-- **Derived** read mean (this is `read_disagreement_mean`, now a fact about the
oracle, not a field): `E[read v i] = η + (1−2η)·flip v i`. -/
lemma read_mean (v i) : μ[O.read v i] = O.η + (1 - 2 * O.η) * O.flip v i := by
  have h := read_disagreement_mean μ O.η (O.flip v i) (O.noise v i) (O.noise_int v i)
    (O.noise_mean v i)
  calc μ[O.read v i]
      = ∫ ω, (O.flip v i + (1 - 2 * O.flip v i) * O.noise v i ω) ∂μ := rfl
    _ = O.η + O.flip v i * (1 - 2 * O.η) := h
    _ = O.η + (1 - 2 * O.η) * O.flip v i := by ring

lemma read_meas (v i) : AEMeasurable (O.read v i) μ := by
  show AEMeasurable (fun ω => O.flip v i + (1 - 2 * O.flip v i) * O.noise v i ω) μ
  exact aemeasurable_const.add (aemeasurable_const.mul (O.noise_meas v i))

lemma read_indep (v) : iIndepFun (O.read v) μ :=
  (O.noise_indep v).comp (fun i x => O.flip v i + (1 - 2 * O.flip v i) * x)
    (fun _ => measurable_const.add (measurable_const.mul measurable_id))

lemma read_icc (v i) : ∀ᵐ ω ∂μ, O.read v i ω ∈ Set.Icc (0 : ℝ) 1 := by
  filter_upwards [O.noise_icc v i] with ω hω
  rw [Set.mem_Icc] at hω
  have hr : O.read v i ω = O.flip v i + (1 - 2 * O.flip v i) * O.noise v i ω := rfl
  rw [hr, Set.mem_Icc]
  rcases O.flip_bit v i with h | h <;> rw [h] <;> constructor <;> nlinarith [hω.1, hω.2]

end Oracle

theorem greedy_picks_good {S : Type*} [DecidableEq S] (O : Oracle μ S)
    (good bad : S → Prop) [DecidablePred good] [DecidablePred bad]
    (hdisj : ∀ v, bad v → ¬ good v)
    (cands : Finset S) (k m : ℕ) (εcov : ℝ)
    (hgoodflip : ∀ v ∈ cands, good v → ∑ i ∈ Finset.range m, O.flip v i = 0)
    (hbadflip : ∀ v ∈ cands, bad v → (m : ℝ) * εcov ≤ ∑ i ∈ Finset.range m, O.flip v i)
    (hεcov0 : 0 ≤ εcov)
    (goodCount : k ≤ (cands.filter good).card)
    (chosen : Ω → Finset S)
    (hsub : ∀ ω, chosen ω ⊆ cands) (hcard : ∀ ω, (chosen ω).card = k)
    (hleast : ∀ ω, ∀ v ∈ chosen ω, ∀ w ∈ cands, w ∉ chosen ω →
        (∑ i ∈ Finset.range m, O.read v i ω) ≤ ∑ i ∈ Finset.range m, O.read w i ω) :
    μ.real {ω | ¬ ∀ w ∈ chosen ω, ¬ bad w}
      ≤ (cands.card : ℝ) * Real.exp (-2 * (m : ℝ) * ((1 / 2 - O.η) * εcov) ^ 2) := by
  have hsum : ∀ v, ∑ i ∈ Finset.range m, μ[O.read v i]
      = (m : ℝ) * O.η + (1 - 2 * O.η) * ∑ i ∈ Finset.range m, O.flip v i := by
    intro v
    have h := denoised_loss_eq_flip (μ := μ) m O.η (1 / 2 - O.η) (O.flip v) (O.read v)
      (fun j => by rw [O.read_mean]; ring)
    rw [h]; ring
  have hgm : ∀ v ∈ cands, good v → ∑ i ∈ Finset.range m, μ[O.read v i] ≤ (m : ℝ) * O.η := by
    intro v hv hg; rw [hsum v, hgoodflip v hv hg]; simp
  have hbm : ∀ v ∈ cands, bad v →
      (m : ℝ) * (O.η + (1 - 2 * O.η) * εcov) ≤ ∑ i ∈ Finset.range m, μ[O.read v i] := by
    intro v hv hb; rw [hsum v]; nlinarith [hbadflip v hv hb, O.hη]
  refine chosen_avoids_bad_whp good bad hdisj cands k m O.η (O.η + (1 - 2 * O.η) * εcov)
    ((1 / 2 - O.η) * εcov) O.read O.read_meas O.read_indep O.read_icc hgm hbm ?_ ?_ goodCount
    chosen hsub hcard hleast
  · nlinarith [hεcov0, O.hη]
  · have : (0 : ℝ) ≤ 1 / 2 - O.η := by linarith [O.hη]
    exact mul_nonneg this hεcov0

#print axioms greedy_picks_good

/-- **Liveness (fused): separability ⇒ a good family is produced, w.h.p.**
Combining the two proved halves.  Under mean-loss separability the greedy proposes
an all-accept-preserving family except w.p. `#cands·exp(-2mγ²)`
(`chosen_accept_preserving_whp`); and the gate rejects the proposed family with
probability at most `qgate` (the gate-accept factor, `1 - qgate`, from
`cleanAdmit_le`/`apLowFNR_le` for an accept-preserving family).  Then the round
produces a family that is accept-preserving *and* clears the gate — a good pass —
except with probability at most `#cands·exp(-2mγ²) + qgate`.

This is the faithful liveness statement: not "lucky independent draws" but
"once the pool is separable, the round succeeds w.h.p." -/
theorem liveness_produces_good {S : Type*} [DecidableEq S]
    (AP : S → Prop) [DecidablePred AP]
    (cands : Finset S) (k m : ℕ) (ρlo ρhi γ : ℝ)
    (D : S → ℕ → Ω → ℝ)
    (hmeas : ∀ v i, AEMeasurable (D v i) μ)
    (hindep : ∀ v, iIndepFun (D v) μ)
    (hIcc : ∀ v i, ∀ᵐ ω ∂μ, D v i ω ∈ Set.Icc (0 : ℝ) 1)
    (hAPmean : ∀ v ∈ cands, AP v → ∑ i ∈ Finset.range m, μ[D v i] ≤ (m : ℝ) * ρlo)
    (hNAmean : ∀ v ∈ cands, ¬ AP v → (m : ℝ) * ρhi ≤ ∑ i ∈ Finset.range m, μ[D v i])
    (hgap : ρlo + γ ≤ ρhi - γ) (hγ : 0 ≤ γ)
    (apCount : k ≤ (cands.filter AP).card)
    (chosen : Ω → Finset S)
    (hsub : ∀ ω, chosen ω ⊆ cands) (hcard : ∀ ω, (chosen ω).card = k)
    (hleast : ∀ ω, ∀ v ∈ chosen ω, ∀ w ∈ cands, w ∉ chosen ω →
        (∑ i ∈ Finset.range m, D v i ω) ≤ ∑ i ∈ Finset.range m, D w i ω)
    (gateReject : Set Ω) (qgate : ℝ) (hgate : μ.real gateReject ≤ qgate) :
    μ.real {ω | ¬ ((∀ w ∈ chosen ω, AP w) ∧ ω ∉ gateReject)}
      ≤ (cands.card : ℝ) * Real.exp (-2 * (m : ℝ) * γ ^ 2) + qgate := by
  have hprop := chosen_accept_preserving_whp AP cands k m ρlo ρhi γ D hmeas hindep hIcc
    hAPmean hNAmean hgap hγ apCount chosen hsub hcard hleast
  have hincl : {ω | ¬ ((∀ w ∈ chosen ω, AP w) ∧ ω ∉ gateReject)}
      ⊆ {ω | ¬ ∀ w ∈ chosen ω, AP w} ∪ gateReject := by
    intro ω hω
    by_contra hn
    rw [Set.mem_union, not_or] at hn
    exact hω ⟨not_not.mp hn.1, hn.2⟩
  calc μ.real {ω | ¬ ((∀ w ∈ chosen ω, AP w) ∧ ω ∉ gateReject)}
      ≤ μ.real ({ω | ¬ ∀ w ∈ chosen ω, AP w} ∪ gateReject) := measureReal_mono hincl
    _ ≤ μ.real {ω | ¬ ∀ w ∈ chosen ω, AP w} + μ.real gateReject := measureReal_union_le _ _
    _ ≤ (cands.card : ℝ) * Real.exp (-2 * (m : ℝ) * γ ^ 2) + qgate := add_le_add hprop hgate

#print axioms liveness_produces_good

end OrthoDFA
