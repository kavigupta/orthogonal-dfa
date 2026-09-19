import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import OrthoDFA.Estimate
import OrthoDFA.Model
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

/-- Selection under separability.  If `chosen` is a least-loss `k`-subset of
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

/-- Selection avoids the bad set.  The D-relative version: `good` and `bad` are
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

section leastLoss
variable {S : Type*} [DecidableEq S] (ℓ : S → ℝ) (cands : Finset S) (k : ℕ)

lemma leastLossSubset_mem (hk : k ≤ cands.card) :
    leastLossSubset ℓ cands k ∈ cands.powersetCard k := by
  have h : (cands.powersetCard k).Nonempty := Finset.powersetCard_nonempty.mpr hk
  rw [leastLossSubset, dif_pos h]
  exact (Finset.exists_min_image (cands.powersetCard k) (fun T => ∑ x ∈ T, ℓ x) h).choose_spec.1

lemma leastLossSubset_subset (hk : k ≤ cands.card) : leastLossSubset ℓ cands k ⊆ cands :=
  (Finset.mem_powersetCard.mp (leastLossSubset_mem ℓ cands k hk)).1

lemma leastLossSubset_card (hk : k ≤ cands.card) : (leastLossSubset ℓ cands k).card = k :=
  (Finset.mem_powersetCard.mp (leastLossSubset_mem ℓ cands k hk)).2

/-- The defining property: every chosen element has loss ≤ every unchosen candidate.
Proved from minimality of the argmin by the swap `v ↦ w`. -/
lemma leastLossSubset_least (hk : k ≤ cands.card) :
    ∀ v ∈ leastLossSubset ℓ cands k, ∀ w ∈ cands, w ∉ leastLossSubset ℓ cands k →
      ℓ v ≤ ℓ w := by
  intro v hv w hw hwnot
  by_contra hlt
  push_neg at hlt
  have h : (cands.powersetCard k).Nonempty := Finset.powersetCard_nonempty.mpr hk
  have hspec :=
    (Finset.exists_min_image (cands.powersetCard k) (fun T => ∑ x ∈ T, ℓ x) h).choose_spec
  have hTeq : leastLossSubset ℓ cands k
      = (Finset.exists_min_image (cands.powersetCard k) (fun T => ∑ x ∈ T, ℓ x) h).choose := by
    rw [leastLossSubset, dif_pos h]
  have hwnoterase : w ∉ (leastLossSubset ℓ cands k).erase v :=
    fun hh => hwnot (Finset.mem_of_mem_erase hh)
  set T' := insert w ((leastLossSubset ℓ cands k).erase v) with hT'
  have hT'mem : T' ∈ cands.powersetCard k := by
    rw [Finset.mem_powersetCard]
    refine ⟨?_, ?_⟩
    · rw [hT', Finset.insert_subset_iff]
      exact ⟨hw, (Finset.erase_subset v _).trans (leastLossSubset_subset ℓ cands k hk)⟩
    · have hkpos : 0 < k := by
        have hp := Finset.card_pos.mpr ⟨v, hv⟩
        rwa [leastLossSubset_card ℓ cands k hk] at hp
      rw [hT', Finset.card_insert_of_notMem hwnoterase, Finset.card_erase_of_mem hv,
        leastLossSubset_card ℓ cands k hk]
      omega
  have hmin := hspec.2 T' hT'mem
  rw [← hTeq] at hmin
  have hsum : ∑ x ∈ T', ℓ x = (∑ x ∈ leastLossSubset ℓ cands k, ℓ x) - ℓ v + ℓ w := by
    rw [hT', Finset.sum_insert hwnoterase, Finset.sum_erase_eq_sub hv]; ring
  rw [hsum] at hmin
  linarith
end leastLoss

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Liveness core: the greedy proposes an accept-preserving family, w.h.p.
Each suffix `v`'s loss is `∑ⱼ D v j`, `D v j ∈ [0,1]` the disagreement indicator on
prefix `j`, independent across prefixes.  Under *mean-loss separability* — every
accept-preserving suffix has mean loss ≤ `m·ρlo`, every other ≥ `m·ρhi`, with a
margin `γ` (`ρlo+γ ≤ ρhi-γ`) — and at least `k` accept-preserving candidates, the
`ω`-dependent least-loss `k`-subset `chosen ω` is entirely accept-preserving except
with probability at most `#cands · exp(-2mγ²)`.

The noisy losses separate by concentration (`sumUpper_le`/`sumLower_le`), and the
selection lemma then forces the choice.  `chosen` abstracts the ε-anchored greedy's
output as any per-`ω` least-loss `k`-subset. -/
theorem chosen_accept_preserving_whp {S ι : Type*} [DecidableEq S]
    (AP : S → Prop) [DecidablePred AP]
    (cands : Finset S) (k : ℕ) (idx : Finset ι) (ρlo ρhi γ : ℝ)
    (D : S → ι → Ω → ℝ)
    (hmeas : ∀ v i, AEMeasurable (D v i) μ)
    (hindep : ∀ v, iIndepFun (D v) μ)
    (hIcc : ∀ v i, ∀ᵐ ω ∂μ, D v i ω ∈ Set.Icc (0 : ℝ) 1)
    (hAPmean : ∀ v ∈ cands, AP v → ∑ i ∈ idx, μ[D v i] ≤ (idx.card : ℝ) * ρlo)
    (hNAmean : ∀ v ∈ cands, ¬ AP v → (idx.card : ℝ) * ρhi ≤ ∑ i ∈ idx, μ[D v i])
    (hgap : ρlo + γ ≤ ρhi - γ) (hγ : 0 ≤ γ)
    (apCount : k ≤ (cands.filter AP).card)
    (chosen : Ω → Finset S)
    (hsub : ∀ ω, chosen ω ⊆ cands) (hcard : ∀ ω, (chosen ω).card = k)
    (hleast : ∀ ω, ∀ v ∈ chosen ω, ∀ w ∈ cands, w ∉ chosen ω →
        (∑ i ∈ idx, D v i ω) ≤ ∑ i ∈ idx, D w i ω) :
    μ.real {ω | ¬ ∀ w ∈ chosen ω, AP w}
      ≤ (cands.card : ℝ) * Real.exp (-2 * (idx.card : ℝ) * γ ^ 2) := by
  classical
  set E : ℝ := Real.exp (-2 * (idx.card : ℝ) * γ ^ 2) with hE
  set UAP : Set Ω := ⋃ v ∈ cands.filter AP,
    {ω | (idx.card : ℝ) * (ρlo + γ) ≤ ∑ i ∈ idx, D v i ω} with hUAP
  set UNA : Set Ω := ⋃ v ∈ cands.filter (fun v => ¬ AP v),
    {ω | ∑ i ∈ idx, D v i ω ≤ (idx.card : ℝ) * (ρhi - γ)} with hUNA
  have hbadAP : μ.real UAP ≤ ((cands.filter AP).card : ℝ) * E := by
    calc μ.real UAP ≤ ∑ v ∈ cands.filter AP,
          μ.real {ω | (idx.card : ℝ) * (ρlo + γ) ≤ ∑ i ∈ idx, D v i ω} :=
            measureReal_biUnion_le _ _
      _ ≤ ∑ _v ∈ cands.filter AP, E := Finset.sum_le_sum (fun v hv => by
            obtain ⟨hvc, hvAP⟩ := Finset.mem_filter.mp hv
            exact sumUpper_le (D v) idx ρlo γ (hmeas v) (hindep v) (hIcc v)
              (hAPmean v hvc hvAP) hγ)
      _ = ((cands.filter AP).card : ℝ) * E := by rw [Finset.sum_const, nsmul_eq_mul]
  have hbadNA : μ.real UNA ≤ ((cands.filter (fun v => ¬ AP v)).card : ℝ) * E := by
    calc μ.real UNA ≤ ∑ v ∈ cands.filter (fun v => ¬ AP v),
          μ.real {ω | ∑ i ∈ idx, D v i ω ≤ (idx.card : ℝ) * (ρhi - γ)} :=
            measureReal_biUnion_le _ _
      _ ≤ ∑ _v ∈ cands.filter (fun v => ¬ AP v), E := Finset.sum_le_sum (fun v hv => by
            obtain ⟨hvc, hvNA⟩ := Finset.mem_filter.mp hv
            exact sumLower_le (D v) idx ρhi γ (hmeas v) (hindep v) (hIcc v)
              (hNAmean v hvc hvNA) hγ)
      _ = ((cands.filter (fun v => ¬ AP v)).card : ℝ) * E := by
            rw [Finset.sum_const, nsmul_eq_mul]
  have hincl : {ω | ¬ ∀ w ∈ chosen ω, AP w} ⊆ UAP ∪ UNA := by
    intro ω hω
    by_contra hnot
    rw [Set.mem_union, not_or] at hnot
    obtain ⟨hnAP, hnNA⟩ := hnot
    apply hω
    refine chosen_accept_preserving (fun v => ∑ i ∈ idx, D v i ω) AP cands
      (chosen ω) k (hsub ω) (hcard ω) (hleast ω) apCount ?_
    intro v hv w hw hvAP hwNA
    have h1 : ∑ i ∈ idx, D v i ω < (idx.card : ℝ) * (ρlo + γ) := by
      by_contra h
      exact hnAP (Set.mem_iUnion₂.mpr ⟨v, Finset.mem_filter.mpr ⟨hv, hvAP⟩, not_lt.mp h⟩)
    have h2 : (idx.card : ℝ) * (ρhi - γ) < ∑ i ∈ idx, D w i ω := by
      by_contra h
      exact hnNA (Set.mem_iUnion₂.mpr ⟨w, Finset.mem_filter.mpr ⟨hw, hwNA⟩, not_lt.mp h⟩)
    have hmid : (idx.card : ℝ) * (ρlo + γ) ≤ (idx.card : ℝ) * (ρhi - γ) :=
      mul_le_mul_of_nonneg_left hgap (Nat.cast_nonneg idx.card)
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

/-- The "reads fail to separate the classes" event: some good candidate's loss
reaches the upper band `m(ρlo+γ)`, or some bad candidate's loss drops to the lower
band `m(ρhi-γ)`.  Its complement is the separation *trigger*: off `dSepCompl` the
noisy losses split good strictly below bad, so the greedy avoids `bad`. -/
def dSepCompl {S ι : Type*} [DecidableEq S] (good bad : S → Prop)
    [DecidablePred good] [DecidablePred bad]
    (cands : Finset S) (idx : Finset ι) (ρlo ρhi γ : ℝ) (D : S → ι → Ω → ℝ) : Set Ω :=
  (⋃ v ∈ cands.filter good, {ω | (idx.card : ℝ) * (ρlo + γ) ≤ ∑ i ∈ idx, D v i ω}) ∪
  (⋃ v ∈ cands.filter bad, {ω | ∑ i ∈ idx, D v i ω ≤ (idx.card : ℝ) * (ρhi - γ)})

/-- The separation trigger fires w.h.p.  Under mean-loss separability, the reads
fail to separate the classes (`dSepCompl`) with probability at most
`#cands·exp(-2mγ²)`. -/
lemma dSepCompl_prob {S ι : Type*} [DecidableEq S] (good bad : S → Prop)
    [DecidablePred good] [DecidablePred bad]
    (hdisj : ∀ v, bad v → ¬ good v)
    (cands : Finset S) (idx : Finset ι) (ρlo ρhi γ : ℝ) (D : S → ι → Ω → ℝ)
    (hmeas : ∀ v i, AEMeasurable (D v i) μ)
    (hindep : ∀ v, iIndepFun (D v) μ)
    (hIcc : ∀ v i, ∀ᵐ ω ∂μ, D v i ω ∈ Set.Icc (0 : ℝ) 1)
    (hgoodmean : ∀ v ∈ cands, good v → ∑ i ∈ idx, μ[D v i] ≤ (idx.card : ℝ) * ρlo)
    (hbadmean : ∀ v ∈ cands, bad v → (idx.card : ℝ) * ρhi ≤ ∑ i ∈ idx, μ[D v i])
    (hγ : 0 ≤ γ) :
    μ.real (dSepCompl good bad cands idx ρlo ρhi γ D)
      ≤ (cands.card : ℝ) * Real.exp (-2 * (idx.card : ℝ) * γ ^ 2) := by
  classical
  set E : ℝ := Real.exp (-2 * (idx.card : ℝ) * γ ^ 2) with hE
  have hbadUG : μ.real (⋃ v ∈ cands.filter good,
      {ω | (idx.card : ℝ) * (ρlo + γ) ≤ ∑ i ∈ idx, D v i ω})
      ≤ ((cands.filter good).card : ℝ) * E := by
    calc μ.real (⋃ v ∈ cands.filter good,
          {ω | (idx.card : ℝ) * (ρlo + γ) ≤ ∑ i ∈ idx, D v i ω})
        ≤ ∑ v ∈ cands.filter good,
          μ.real {ω | (idx.card : ℝ) * (ρlo + γ) ≤ ∑ i ∈ idx, D v i ω} :=
            measureReal_biUnion_le _ _
      _ ≤ ∑ _v ∈ cands.filter good, E := Finset.sum_le_sum (fun v hv => by
            obtain ⟨hvc, hvg⟩ := Finset.mem_filter.mp hv
            exact sumUpper_le (D v) idx ρlo γ (hmeas v) (hindep v) (hIcc v)
              (hgoodmean v hvc hvg) hγ)
      _ = ((cands.filter good).card : ℝ) * E := by rw [Finset.sum_const, nsmul_eq_mul]
  have hbadUB : μ.real (⋃ v ∈ cands.filter bad,
      {ω | ∑ i ∈ idx, D v i ω ≤ (idx.card : ℝ) * (ρhi - γ)})
      ≤ ((cands.filter bad).card : ℝ) * E := by
    calc μ.real (⋃ v ∈ cands.filter bad,
          {ω | ∑ i ∈ idx, D v i ω ≤ (idx.card : ℝ) * (ρhi - γ)})
        ≤ ∑ v ∈ cands.filter bad,
          μ.real {ω | ∑ i ∈ idx, D v i ω ≤ (idx.card : ℝ) * (ρhi - γ)} :=
            measureReal_biUnion_le _ _
      _ ≤ ∑ _v ∈ cands.filter bad, E := Finset.sum_le_sum (fun v hv => by
            obtain ⟨hvc, hvb⟩ := Finset.mem_filter.mp hv
            exact sumLower_le (D v) idx ρhi γ (hmeas v) (hindep v) (hIcc v)
              (hbadmean v hvc hvb) hγ)
      _ = ((cands.filter bad).card : ℝ) * E := by rw [Finset.sum_const, nsmul_eq_mul]
  have hcards : ((cands.filter good).card : ℝ) + ((cands.filter bad).card : ℝ)
      ≤ (cands.card : ℝ) := by
    have hdisjF : Disjoint (cands.filter good) (cands.filter bad) := by
      rw [Finset.disjoint_filter]
      exact fun v _ hvg hvb => hdisj v hvb hvg
    have hu := Finset.card_union_of_disjoint hdisjF
    have hle : (cands.filter good ∪ cands.filter bad).card ≤ cands.card :=
      Finset.card_le_card (Finset.union_subset (Finset.filter_subset _ _) (Finset.filter_subset _ _))
    rw [hu] at hle
    exact_mod_cast hle
  have hEnn : 0 ≤ E := (Real.exp_pos _).le
  calc μ.real (dSepCompl good bad cands idx ρlo ρhi γ D)
      ≤ μ.real (⋃ v ∈ cands.filter good,
            {ω | (idx.card : ℝ) * (ρlo + γ) ≤ ∑ i ∈ idx, D v i ω})
          + μ.real (⋃ v ∈ cands.filter bad,
            {ω | ∑ i ∈ idx, D v i ω ≤ (idx.card : ℝ) * (ρhi - γ)}) :=
        measureReal_union_le _ _
    _ ≤ ((cands.filter good).card : ℝ) * E + ((cands.filter bad).card : ℝ) * E :=
        add_le_add hbadUG hbadUB
    _ = (((cands.filter good).card : ℝ) + ((cands.filter bad).card : ℝ)) * E := by ring
    _ ≤ (cands.card : ℝ) * E := mul_le_mul_of_nonneg_right hcards hEnn

/-- Off `dSepCompl`, the greedy avoids `bad`.  For a fixed `ω` outside the
separation-failure event, the good losses are strictly below the bad losses, so the
selection lemma forces the least-loss `k`-subset to avoid `bad`. -/
lemma avoids_bad_of_not_mem_dSepCompl {S ι : Type*} [DecidableEq S]
    (good bad : S → Prop) [DecidablePred good] [DecidablePred bad]
    (hdisj : ∀ v, bad v → ¬ good v)
    (cands : Finset S) (k : ℕ) (idx : Finset ι) (ρlo ρhi γ : ℝ) (D : S → ι → Ω → ℝ)
    (hgap : ρlo + γ ≤ ρhi - γ)
    (goodCount : k ≤ (cands.filter good).card)
    (chosen : Finset S) (hsub : chosen ⊆ cands) (hcard : chosen.card = k)
    {ω : Ω}
    (hleast : ∀ v ∈ chosen, ∀ w ∈ cands, w ∉ chosen →
        (∑ i ∈ idx, D v i ω) ≤ ∑ i ∈ idx, D w i ω)
    (hω : ω ∉ dSepCompl good bad cands idx ρlo ρhi γ D) :
    ∀ w ∈ chosen, ¬ bad w := by
  rw [dSepCompl, Set.mem_union, not_or] at hω
  obtain ⟨hnG, hnB⟩ := hω
  refine chosen_avoids_bad (fun v => ∑ i ∈ idx, D v i ω) good bad hdisj cands
    chosen k hsub hcard hleast goodCount ?_
  intro v hv w hw hvg hwb
  have h1 : ∑ i ∈ idx, D v i ω < (idx.card : ℝ) * (ρlo + γ) := by
    by_contra h
    exact hnG (Set.mem_iUnion₂.mpr ⟨v, Finset.mem_filter.mpr ⟨hv, hvg⟩, not_lt.mp h⟩)
  have h2 : (idx.card : ℝ) * (ρhi - γ) < ∑ i ∈ idx, D w i ω := by
    by_contra h
    exact hnB (Set.mem_iUnion₂.mpr ⟨w, Finset.mem_filter.mpr ⟨hw, hwb⟩, not_lt.mp h⟩)
  have hmid : (idx.card : ℝ) * (ρlo + γ) ≤ (idx.card : ℝ) * (ρhi - γ) :=
    mul_le_mul_of_nonneg_left hgap (Nat.cast_nonneg idx.card)
  linarith

/-- Coverage-free liveness: the greedy avoids the bad set, w.h.p.
`good v` (flip of D-mass ≤ ρlo) and `bad v` (flip of D-mass ≥ ρhi ≈ ε_cov) are
*definitional* w.r.t. the target — no coverage assumption.  The greedy's least-loss
`k`-subset avoids `bad` except w.p. ≤ `#cands·exp(-2mγ²)`.  A three-line corollary
of the separation trigger: the failure set sits inside `dSepCompl`. -/
theorem chosen_avoids_bad_whp {S ι : Type*} [DecidableEq S]
    (good bad : S → Prop) [DecidablePred good] [DecidablePred bad]
    (hdisj : ∀ v, bad v → ¬ good v)
    (cands : Finset S) (k : ℕ) (idx : Finset ι) (ρlo ρhi γ : ℝ)
    (D : S → ι → Ω → ℝ)
    (hmeas : ∀ v i, AEMeasurable (D v i) μ)
    (hindep : ∀ v, iIndepFun (D v) μ)
    (hIcc : ∀ v i, ∀ᵐ ω ∂μ, D v i ω ∈ Set.Icc (0 : ℝ) 1)
    (hgoodmean : ∀ v ∈ cands, good v → ∑ i ∈ idx, μ[D v i] ≤ (idx.card : ℝ) * ρlo)
    (hbadmean : ∀ v ∈ cands, bad v → (idx.card : ℝ) * ρhi ≤ ∑ i ∈ idx, μ[D v i])
    (hgap : ρlo + γ ≤ ρhi - γ) (hγ : 0 ≤ γ)
    (goodCount : k ≤ (cands.filter good).card)
    (chosen : Ω → Finset S)
    (hsub : ∀ ω, chosen ω ⊆ cands) (hcard : ∀ ω, (chosen ω).card = k)
    (hleast : ∀ ω, ∀ v ∈ chosen ω, ∀ w ∈ cands, w ∉ chosen ω →
        (∑ i ∈ idx, D v i ω) ≤ ∑ i ∈ idx, D w i ω) :
    μ.real {ω | ¬ ∀ w ∈ chosen ω, ¬ bad w}
      ≤ (cands.card : ℝ) * Real.exp (-2 * (idx.card : ℝ) * γ ^ 2) := by
  have hsubset : {ω | ¬ ∀ w ∈ chosen ω, ¬ bad w}
      ⊆ dSepCompl good bad cands idx ρlo ρhi γ D := by
    intro ω hω
    by_contra hnot
    exact hω (avoids_bad_of_not_mem_dSepCompl good bad hdisj cands k idx ρlo ρhi γ D hgap
      goodCount (chosen ω) (hsub ω) (hcard ω) (hleast ω) hnot)
  exact (measureReal_mono hsubset).trans
    (dSepCompl_prob good bad hdisj cands idx ρlo ρhi γ D hmeas hindep hIcc hgoodmean hbadmean hγ)

/-- Per-prefix disagreement mean, from random classification noise.
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

/-- Denoised loss decomposes into baseline + signal·flips.  Against the
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

namespace Oracle
variable {S : Type*} [MeasurableSpace S] [Mul S] (O : Oracle μ S)

/-- Bit-valuedness, by construction: an indicator is `0` or `1`. -/
lemma label_bit (w : S) : O.label w = 0 ∨ O.label w = 1 := by
  by_cases h : w ∈ O.L <;> simp [Oracle.label, Set.indicator_apply, h]

/-- Two strings get the same bit exactly when they agree on membership. -/
lemma label_eq_iff (x y : S) : O.label x = O.label y ↔ (x ∈ O.L ↔ y ∈ O.L) := by
  by_cases hx : x ∈ O.L <;> by_cases hy : y ∈ O.L <;>
    simp [Oracle.label, Set.indicator_apply, hx, hy]

/-- Accept-preservation, as membership rather than as an equality of bits. -/
lemma apSet_eq : {v : S | ∀ p, p * v ∈ O.L ↔ p ∈ O.L}
    = {v : S | ∀ p, O.label (p * v) = O.label p} :=
  Set.ext fun v => forall_congr' fun p => (O.label_eq_iff (p * v) p).symm

lemma label_meas : Measurable O.label :=
  (measurable_one.indicator O.L_meas)

/-- Joint measurability in the query string and the sample, which is what composing the
oracle with a randomly drawn string needs.  Derived, because for a countable `S` the preimage
splits as `⋃ w, {w} ×ˢ (noise w)⁻¹(B)`. -/
lemma noise_meas_prod [Countable S] [MeasurableSingletonClass S] :
    Measurable (fun z : S × Ω => O.noise z.1 z.2) :=
  measurable_from_prod_countable_right O.noise_meas

/-- Whether `v` flips `p`'s acceptance:

    flip v p = ℓ(p·v) ⊕ ℓ(p) = ℓ(p·v) + ℓ(p) − 2·ℓ(p·v)·ℓ(p)

so it is `0` exactly when `v` is accept-preserving at `p`. -/
noncomputable def flip (v p : S) : ℝ :=
  O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p

/-- Derived bit-valuedness of `flip`: an XOR of two bits is a bit. -/
lemma flip_bit (v p : S) : O.flip v p = 0 ∨ O.flip v p = 1 := by
  rcases O.label_bit (p * v) with h1 | h1 <;> rcases O.label_bit p with h2 | h2 <;>
    · rw [Oracle.flip, h1, h2]; norm_num

/-- Derived boundedness: a `{0,1}` bit lies in `[0,1]` (what the Hoeffding
bounds consume). -/
lemma noise_icc (w) : ∀ᵐ ω ∂μ, O.noise w ω ∈ Set.Icc (0 : ℝ) 1 := by
  filter_upwards [O.noise_bit w] with ω hω
  rcases hω with h | h <;> rw [Set.mem_Icc, h] <;> constructor <;> norm_num

/-- The disagreement read of `v` on the `i`-th prefix,

    read v i = flip v (pref i) ⊕ noise (pref i · v)

the noise being that of the concatenated query string. -/
noncomputable def read {ι : Type*} (pref : ι → S) (v : S) (i : ι) : Ω → ℝ :=
  fun ω => O.flip v (pref i) + (1 - 2 * O.flip v (pref i)) * O.noise (pref i * v) ω

variable {ι : Type*} (pref : ι → S)

lemma noise_int (w) : Integrable (O.noise w) μ :=
  MeasureTheory.Integrable.of_mem_Icc 0 1 (O.noise_meas w).aemeasurable (O.noise_icc w)

open scoped Classical in
/-- The noise rate at one string. -/
noncomputable def rate (w : S) : ℝ := if w ∈ O.L then O.ηIn else O.ηOut

lemma noise_mean (w : S) : μ[O.noise w] = O.rate w := by
  unfold rate
  split_ifs with h
  · exact O.noise_mean_in w h
  · exact O.noise_mean_out w h

lemma rate_nonneg (w : S) : 0 ≤ O.rate w := by
  rw [← O.noise_mean w]
  refine integral_nonneg_of_ae ?_
  filter_upwards [O.noise_icc w] with ω hω
  exact hω.1

lemma rate_le_eta (w : S) : O.rate w ≤ O.η := by
  unfold rate Oracle.η
  split_ifs
  · exact le_max_left _ _
  · exact le_max_right _ _

/-- Two strings in the same class share a rate. -/
lemma rate_eq_of_label_eq {x y : S} (h : O.label x = O.label y) : O.rate x = O.rate y := by
  have hxy := (O.label_eq_iff x y).1 h
  unfold rate
  by_cases hx : x ∈ O.L
  · rw [if_pos hx, if_pos (hxy.1 hx)]
  · rw [if_neg hx, if_neg (fun hy => hx (hxy.2 hy))]

/-- `E[read v i] = r + (1−2r)·flip v (pref i)` at the concatenated string's rate `r`. -/
lemma read_mean (v : S) (i : ι) :
    μ[O.read pref v i]
      = O.rate (pref i * v) + (1 - 2 * O.rate (pref i * v)) * O.flip v (pref i) := by
  have h := read_disagreement_mean μ (O.rate (pref i * v)) (O.flip v (pref i))
    (O.noise (pref i * v)) (O.noise_int _) (O.noise_mean _)
  calc μ[O.read pref v i]
      = ∫ ω, (O.flip v (pref i) + (1 - 2 * O.flip v (pref i)) * O.noise (pref i * v) ω) ∂μ := rfl
    _ = O.rate (pref i * v) + O.flip v (pref i) * (1 - 2 * O.rate (pref i * v)) := h
    _ = O.rate (pref i * v) + (1 - 2 * O.rate (pref i * v)) * O.flip v (pref i) := by ring

lemma rate_eq_eta (hsym : O.ηIn = O.ηOut) (w : S) : O.rate w = O.η := by
  unfold rate Oracle.η
  rw [hsym, max_self]
  split_ifs <;> rfl

lemma read_mean_sym (hsym : O.ηIn = O.ηOut) (v : S) (i : ι) :
    μ[O.read pref v i] = O.η + (1 - 2 * O.η) * O.flip v (pref i) := by
  rw [O.read_mean, O.rate_eq_eta hsym]

/-- A read sits at its prefix's rate, and `1 − 2η` above it where `v` flips. -/
lemma read_mean_bounds (v : S) (i : ι) :
    O.rate (pref i) + (1 - 2 * O.η) * O.flip v (pref i) ≤ μ[O.read pref v i]
      ∧ (O.flip v (pref i) = 0 → μ[O.read pref v i] = O.rate (pref i)) := by
  rw [O.read_mean]
  have hr0 := O.rate_nonneg (pref i * v)
  have hr1 := O.rate_le_eta (pref i * v)
  have hp0 := O.rate_nonneg (pref i)
  have hp1 := O.rate_le_eta (pref i)
  rcases O.flip_bit v (pref i) with h | h
  · have hlab : O.label (pref i * v) = O.label (pref i) := by
      have := h
      unfold Oracle.flip at this
      rcases O.label_bit (pref i * v) with h1 | h1 <;> rcases O.label_bit (pref i) with h2 | h2 <;>
        rw [h1, h2] at this ⊢ <;> norm_num at this
    rw [h, O.rate_eq_of_label_eq hlab]
    exact ⟨le_of_eq (by ring), fun _ => by ring⟩
  · rw [h]
    exact ⟨by nlinarith, fun h' => absurd h' (by norm_num)⟩

lemma read_meas (v : S) (i : ι) : Measurable (O.read pref v i) := by
  show Measurable
    (fun ω => O.flip v (pref i) + (1 - 2 * O.flip v (pref i)) * O.noise (pref i * v) ω)
  exact measurable_const.add (measurable_const.mul (O.noise_meas _))

/-- Derived per-suffix independence across prefixes.  The reads of `v` across
distinct prefixes hit *distinct* query strings (`pref` injective, composed with
right-cancellation `mul_left_injective`), so they are an injective reindexing of the
per-string iid noise — independent by `iIndepFun.precomp`. -/
lemma read_indep [IsRightCancelMul S] (hpref : Function.Injective pref) (v : S) :
    iIndepFun (O.read pref v) μ := by
  have hinj : Function.Injective (fun i => pref i * v) :=
    (mul_left_injective v).comp hpref
  have h1 : iIndepFun (fun i => O.noise (pref i * v)) μ := O.noise_indep.precomp hinj
  exact h1.comp (fun i x => O.flip v (pref i) + (1 - 2 * O.flip v (pref i)) * x)
    (fun _ => measurable_const.add (measurable_const.mul measurable_id))

lemma read_icc (v : S) (i : ι) : ∀ᵐ ω ∂μ, O.read pref v i ω ∈ Set.Icc (0 : ℝ) 1 := by
  filter_upwards [O.noise_icc (pref i * v)] with ω hω
  rw [Set.mem_Icc] at hω
  have hr : O.read pref v i ω
      = O.flip v (pref i) + (1 - 2 * O.flip v (pref i)) * O.noise (pref i * v) ω := rfl
  rw [hr, Set.mem_Icc]
  rcases O.flip_bit v (pref i) with h | h <;> rw [h] <;> constructor <;> nlinarith [hω.1, hω.2]

end Oracle

theorem greedy_picks_good {S : Type*} [DecidableEq S] [MeasurableSpace S] [Mul S] [IsRightCancelMul S]
    (O : Oracle μ S) (hη : O.η ≤ 1 / 2) (pref : ℕ → S) (hpref : Function.Injective pref)
    (good bad : S → Prop) [DecidablePred good] [DecidablePred bad]
    (hdisj : ∀ v, bad v → ¬ good v)
    (cands : Finset S) (k m : ℕ) (εcov : ℝ)
    (hgoodflip : ∀ v ∈ cands, good v → ∑ i ∈ Finset.range m, O.flip v (pref i) = 0)
    (hbadflip : ∀ v ∈ cands, bad v → (m : ℝ) * εcov ≤ ∑ i ∈ Finset.range m, O.flip v (pref i))
    (hεcov0 : 0 ≤ εcov)
    (goodCount : k ≤ (cands.filter good).card)
    (chosen : Ω → Finset S)
    (hsub : ∀ ω, chosen ω ⊆ cands) (hcard : ∀ ω, (chosen ω).card = k)
    (hleast : ∀ ω, ∀ v ∈ chosen ω, ∀ w ∈ cands, w ∉ chosen ω →
        (∑ i ∈ Finset.range m, O.read pref v i ω)
          ≤ ∑ i ∈ Finset.range m, O.read pref w i ω) :
    μ.real {ω | ¬ ∀ w ∈ chosen ω, ¬ bad w}
      ≤ (cands.card : ℝ) * Real.exp (-2 * (m : ℝ) * ((1 / 2 - O.η) * εcov) ^ 2) := by
  set c : ℝ := (∑ i ∈ Finset.range m, O.rate (pref i)) / (m : ℝ) with hc
  have hmc : (m : ℝ) * c = ∑ i ∈ Finset.range m, O.rate (pref i) := by
    rcases Nat.eq_zero_or_pos m with hm | hm
    · subst hm; simp
    · rw [hc]; field_simp
  have hgm : ∀ v ∈ cands, good v →
      ∑ i ∈ Finset.range m, μ[O.read pref v i] ≤ (m : ℝ) * c := by
    intro v hv hg
    have hz : ∀ i ∈ Finset.range m, O.flip v (pref i) = 0 := by
      have hnn : ∀ i ∈ Finset.range m, 0 ≤ O.flip v (pref i) := fun i _ => by
        rcases O.flip_bit v (pref i) with h | h <;> rw [h] <;> norm_num
      exact (Finset.sum_eq_zero_iff_of_nonneg hnn).1 (hgoodflip v hv hg)
    rw [hmc]
    exact le_of_eq (Finset.sum_congr rfl
      (fun i hi => (O.read_mean_bounds pref v i).2 (hz i hi)))
  have hbm : ∀ v ∈ cands, bad v →
      (m : ℝ) * (c + (1 - 2 * O.η) * εcov) ≤ ∑ i ∈ Finset.range m, μ[O.read pref v i] := by
    intro v hv hb
    have hlow := Finset.sum_le_sum (fun i (_ : i ∈ Finset.range m) =>
      (O.read_mean_bounds pref v i).1)
    rw [Finset.sum_add_distrib, ← Finset.mul_sum] at hlow
    have h2η : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith
    have := mul_le_mul_of_nonneg_left (hbadflip v hv hb) h2η
    nlinarith [hmc]
  have hgapb : c + (1 / 2 - O.η) * εcov
      ≤ (c + (1 - 2 * O.η) * εcov) - (1 / 2 - O.η) * εcov := by nlinarith [hεcov0]
  have hγb : (0 : ℝ) ≤ (1 / 2 - O.η) * εcov :=
    mul_nonneg (by linarith) hεcov0
  have h := chosen_avoids_bad_whp good bad hdisj cands k (Finset.range m) c
    (c + (1 - 2 * O.η) * εcov) ((1 / 2 - O.η) * εcov) (O.read pref)
    (fun v i => (O.read_meas pref v i).aemeasurable)
    (O.read_indep pref hpref) (O.read_icc pref)
    (by simpa [Finset.card_range] using hgm) (by simpa [Finset.card_range] using hbm)
    hgapb hγb goodCount chosen hsub hcard hleast
  simpa [Finset.card_range] using h

#print axioms greedy_picks_good

namespace Oracle
variable {S : Type*} [MeasurableSpace S] [Mul S] [DecidableEq S] (O : Oracle μ S)

/-- Liveness (fused): separability ⇒ a good family is produced, w.h.p.
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
  have hprop : μ.real {ω | ¬ ∀ w ∈ chosen ω, AP w}
      ≤ (cands.card : ℝ) * Real.exp (-2 * (m : ℝ) * γ ^ 2) := by
    have h := chosen_accept_preserving_whp AP cands k (Finset.range m) ρlo ρhi γ D hmeas hindep
      hIcc (by simpa [Finset.card_range] using hAPmean)
      (by simpa [Finset.card_range] using hNAmean) hgap hγ apCount chosen hsub hcard hleast
    simpa [Finset.card_range] using h
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

end Oracle

end OrthoDFA
