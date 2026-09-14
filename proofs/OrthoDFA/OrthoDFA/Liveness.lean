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
