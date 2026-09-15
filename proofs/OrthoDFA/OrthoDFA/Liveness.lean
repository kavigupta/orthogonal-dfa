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

open scoped Classical in
/-- The greedy's output: a least-total-loss `k`-subset of `cands` (argmin over
`k`-subsets of `∑ ℓ`).  This *defines* the greedy, so its subset/cardinality/
pairwise-least-loss properties become lemmas rather than assumptions. -/
noncomputable def leastLossSubset {S : Type*} (ℓ : S → ℝ) (cands : Finset S) (k : ℕ) :
    Finset S :=
  if h : (cands.powersetCard k).Nonempty then
    (Finset.exists_min_image (cands.powersetCard k) (fun T => ∑ x ∈ T, ℓ x) h).choose
  else ∅

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

/-- The **"reads fail to separate the classes"** event: some good candidate's loss
reaches the upper band `m(ρlo+γ)`, or some bad candidate's loss drops to the lower
band `m(ρhi-γ)`.  Its complement is the separation *trigger*: off `dSepCompl` the
noisy losses split good strictly below bad, so the greedy avoids `bad`. -/
def dSepCompl {S ι : Type*} [DecidableEq S] (good bad : S → Prop)
    [DecidablePred good] [DecidablePred bad]
    (cands : Finset S) (idx : Finset ι) (ρlo ρhi γ : ℝ) (D : S → ι → Ω → ℝ) : Set Ω :=
  (⋃ v ∈ cands.filter good, {ω | (idx.card : ℝ) * (ρlo + γ) ≤ ∑ i ∈ idx, D v i ω}) ∪
  (⋃ v ∈ cands.filter bad, {ω | ∑ i ∈ idx, D v i ω ≤ (idx.card : ℝ) * (ρhi - γ)})

/-- `dSepCompl` is measurable when the reads are. -/
lemma dSepCompl_measurable {S ι : Type*} [DecidableEq S] (good bad : S → Prop)
    [DecidablePred good] [DecidablePred bad]
    (cands : Finset S) (idx : Finset ι) (ρlo ρhi γ : ℝ) (D : S → ι → Ω → ℝ)
    (hD : ∀ v i, Measurable (D v i)) :
    MeasurableSet (dSepCompl good bad cands idx ρlo ρhi γ D) :=
  MeasurableSet.union
    (Finset.measurableSet_biUnion _ (fun v _ =>
      measurableSet_le measurable_const (Finset.measurable_sum _ (fun i _ => hD v i))))
    (Finset.measurableSet_biUnion _ (fun v _ =>
      measurableSet_le (Finset.measurable_sum _ (fun i _ => hD v i)) measurable_const))

/-- **The separation trigger fires w.h.p.**  Under mean-loss separability, the reads
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

/-- **Off `dSepCompl`, the greedy avoids `bad`.**  For a fixed `ω` outside the
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

/-- **Coverage-free liveness: the greedy avoids the bad set, w.h.p.**
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

/-- The persistent signal oracle: random classification noise on query strings.
Every field is a function of a **single** query string `w : S` — the oracle answers
membership on one string at a time.  `label w = 1[w ∈ L]` is the noiseless
membership bit and `noise w` the persistent RCN bit; the membership query is
`label ⊕ noise`.  Concatenation and the notion of one suffix flipping a prefix are
*not* in the oracle: strings are Mathlib's theory (`[Mul S]` concatenation,
`[IsRightCancelMul S]` right-cancellation), and `flip` is *derived* below. -/
structure Oracle {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    (S : Type*) [MeasurableSpace S] where
  /-- The noiseless membership bit `ℓ(w) = 1[w ∈ L]`. -/
  label : S → ℝ
  label_bit : ∀ w, label w = 0 ∨ label w = 1
  label_meas : Measurable label
  /-- The random classification noise, one persistent bit per query string. -/
  noise : S → Ω → ℝ
  /-- The noise level. -/
  η : ℝ
  hη : η ≤ 1 / 2
  /-- Noise is independent and identically distributed as `Bernoulli(η)`.  Measurability
  is *joint* in the query string and the sample, which is what lets the oracle be
  composed with a **randomly drawn** query string; the per-string version is derived. -/
  noise_meas : Measurable (fun z : S × Ω => noise z.1 z.2)
  noise_indep : iIndepFun noise μ
  noise_bit : ∀ w, ∀ᵐ ω ∂μ, noise w ω = 0 ∨ noise w ω = 1
  noise_mean : ∀ w, μ[noise w] = η

namespace Oracle
variable {S : Type*} [MeasurableSpace S] [Mul S] (O : Oracle μ S)

/-- **Derived** per-string measurability, from the joint version. -/
lemma noise_meas' (w : S) : Measurable (O.noise w) :=
  O.noise_meas.comp (measurable_const.prodMk measurable_id)

/-- Whether suffix `v` flips prefix `p`'s acceptance: the XOR `ℓ(p·v) ⊕ ℓ(p)` of the
two membership bits (`a ⊕ b = a + b − 2ab`).  **Derived** from the single-string
label, so it is `0` exactly when `p·v` and `p` agree — accept-preservation. -/
def flip (v p : S) : ℝ :=
  O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p

/-- **Derived** bit-valuedness of `flip`: an XOR of two bits is a bit. -/
lemma flip_bit (v p : S) : O.flip v p = 0 ∨ O.flip v p = 1 := by
  rcases O.label_bit (p * v) with h1 | h1 <;> rcases O.label_bit p with h2 | h2 <;>
    · rw [Oracle.flip, h1, h2]; norm_num

/-- **Derived** boundedness: a `{0,1}` bit lies in `[0,1]` (what the Hoeffding
bounds consume). -/
lemma noise_icc (w) : ∀ᵐ ω ∂μ, O.noise w ω ∈ Set.Icc (0 : ℝ) 1 := by
  filter_upwards [O.noise_bit w] with ω hω
  rcases hω with h | h <;> rw [Set.mem_Icc, h] <;> constructor <;> norm_num

/-- The disagreement read `flip ⊕ noise = flip + (1−2·flip)·noise` of suffix `v` on
the `i`-th prefix `pref i` (both strings), where the noise is that of the
concatenated query string `pref i · v` (`·` = the string `Mul`). -/
noncomputable def read (pref : ℕ → S) (v : S) (i : ℕ) : Ω → ℝ :=
  fun ω => O.flip v (pref i) + (1 - 2 * O.flip v (pref i)) * O.noise (pref i * v) ω

variable (pref : ℕ → S)

lemma noise_int (w) : Integrable (O.noise w) μ :=
  MeasureTheory.Integrable.of_mem_Icc 0 1 (O.noise_meas' w).aemeasurable (O.noise_icc w)

/-- **Derived** read mean (this is `read_disagreement_mean`, now a fact about the
oracle, not a field): `E[read v i] = η + (1−2η)·flip v (pref i)`. -/
lemma read_mean (v i) :
    μ[O.read pref v i] = O.η + (1 - 2 * O.η) * O.flip v (pref i) := by
  have h := read_disagreement_mean μ O.η (O.flip v (pref i)) (O.noise (pref i * v))
    (O.noise_int _) (O.noise_mean _)
  calc μ[O.read pref v i]
      = ∫ ω, (O.flip v (pref i) + (1 - 2 * O.flip v (pref i)) * O.noise (pref i * v) ω) ∂μ := rfl
    _ = O.η + O.flip v (pref i) * (1 - 2 * O.η) := h
    _ = O.η + (1 - 2 * O.η) * O.flip v (pref i) := by ring

lemma read_meas (v i) : Measurable (O.read pref v i) := by
  show Measurable
    (fun ω => O.flip v (pref i) + (1 - 2 * O.flip v (pref i)) * O.noise (pref i * v) ω)
  exact measurable_const.add (measurable_const.mul (O.noise_meas' _))

/-- **Derived** per-suffix independence across prefixes.  The reads of `v` across
distinct prefixes hit *distinct* query strings (`pref` injective, composed with
right-cancellation `mul_left_injective`), so they are an injective reindexing of the
per-string iid noise — independent by `iIndepFun.precomp`. -/
lemma read_indep [IsRightCancelMul S] (hpref : Function.Injective pref) (v) :
    iIndepFun (O.read pref v) μ := by
  have hinj : Function.Injective (fun i => pref i * v) :=
    (mul_left_injective v).comp hpref
  have h1 : iIndepFun (fun i => O.noise (pref i * v)) μ := O.noise_indep.precomp hinj
  exact h1.comp (fun i x => O.flip v (pref i) + (1 - 2 * O.flip v (pref i)) * x)
    (fun _ => measurable_const.add (measurable_const.mul measurable_id))

lemma read_icc (v i) : ∀ᵐ ω ∂μ, O.read pref v i ω ∈ Set.Icc (0 : ℝ) 1 := by
  filter_upwards [O.noise_icc (pref i * v)] with ω hω
  rw [Set.mem_Icc] at hω
  have hr : O.read pref v i ω
      = O.flip v (pref i) + (1 - 2 * O.flip v (pref i)) * O.noise (pref i * v) ω := rfl
  rw [hr, Set.mem_Icc]
  rcases O.flip_bit v (pref i) with h | h <;> rw [h] <;> constructor <;> nlinarith [hω.1, hω.2]

lemma read_int (v i) : Integrable (O.read pref v i) μ :=
  MeasureTheory.Integrable.of_mem_Icc 0 1 (O.read_meas pref v i).aemeasurable (O.read_icc pref v i)

end Oracle

theorem greedy_picks_good {S : Type*} [DecidableEq S] [MeasurableSpace S] [Mul S] [IsRightCancelMul S]
    (O : Oracle μ S) (pref : ℕ → S) (hpref : Function.Injective pref)
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
  have hsum : ∀ v, ∑ i ∈ Finset.range m, μ[O.read pref v i]
      = (m : ℝ) * O.η + (1 - 2 * O.η) * ∑ i ∈ Finset.range m, O.flip v (pref i) := by
    intro v
    have h := denoised_loss_eq_flip (μ := μ) m O.η (1 / 2 - O.η) (fun i => O.flip v (pref i))
      (O.read pref v) (fun j => by rw [O.read_mean]; ring)
    rw [h]; ring
  have hgm : ∀ v ∈ cands, good v →
      ∑ i ∈ Finset.range m, μ[O.read pref v i] ≤ (m : ℝ) * O.η := by
    intro v hv hg; rw [hsum v, hgoodflip v hv hg]; simp
  have hbm : ∀ v ∈ cands, bad v →
      (m : ℝ) * (O.η + (1 - 2 * O.η) * εcov) ≤ ∑ i ∈ Finset.range m, μ[O.read pref v i] := by
    intro v hv hb; rw [hsum v]; nlinarith [hbadflip v hv hb, O.hη]
  have hgapb : O.η + (1 / 2 - O.η) * εcov
      ≤ (O.η + (1 - 2 * O.η) * εcov) - (1 / 2 - O.η) * εcov := by nlinarith [hεcov0, O.hη]
  have hγb : (0 : ℝ) ≤ (1 / 2 - O.η) * εcov :=
    mul_nonneg (by linarith [O.hη]) hεcov0
  have h := chosen_avoids_bad_whp good bad hdisj cands k (Finset.range m) O.η
    (O.η + (1 - 2 * O.η) * εcov) ((1 / 2 - O.η) * εcov) (O.read pref)
    (fun v i => (O.read_meas pref v i).aemeasurable)
    (O.read_indep pref hpref) (O.read_icc pref)
    (by simpa [Finset.card_range] using hgm) (by simpa [Finset.card_range] using hbm)
    hgapb hγb goodCount chosen hsub hcard hleast
  simpa [Finset.card_range] using h

#print axioms greedy_picks_good

namespace Oracle
variable {S : Type*} [MeasurableSpace S] [Mul S] [DecidableEq S] (O : Oracle μ S)

/-- The round's **separation-failure** event for the greedy over the oracle reads:
`dSepCompl` at the greedy bands `ρlo = η`, `ρhi = η+(1−2η)εcov`, `γ = (½−η)εcov`
(here `ρlo+γ = ρhi-γ = η+(½−η)εcov`, a single threshold).  Its complement is the
*trigger* that forces the greedy to propose an all-good family. -/
noncomputable def sepFail (pref : ℕ → S) (good bad : S → Prop)
    [DecidablePred good] [DecidablePred bad] (cands : Finset S) (m : ℕ) (εcov : ℝ) : Set Ω :=
  dSepCompl good bad cands (Finset.range m) O.η (O.η + (1 - 2 * O.η) * εcov) ((1 / 2 - O.η) * εcov) (O.read pref)

/-- The separation trigger is measurable. -/
lemma sepFail_measurable (pref : ℕ → S) (good bad : S → Prop)
    [DecidablePred good] [DecidablePred bad] (cands : Finset S) (m : ℕ) (εcov : ℝ) :
    MeasurableSet (O.sepFail pref good bad cands m εcov) :=
  dSepCompl_measurable good bad cands (Finset.range m) O.η (O.η + (1 - 2 * O.η) * εcov) ((1 / 2 - O.η) * εcov)
    (O.read pref) (fun v i => O.read_meas pref v i)

/-- **Off `sepFail`, the greedy avoids `bad`.**  Derived from the selection lemma:
the two bands coincide, so a good candidate's loss below it and a bad candidate's
above it order good strictly under bad. -/
lemma not_bad_of_not_mem_sepFail (pref : ℕ → S) (good bad : S → Prop)
    [DecidablePred good] [DecidablePred bad] (hdisj : ∀ v, bad v → ¬ good v)
    (cands : Finset S) (k m : ℕ) (εcov : ℝ)
    (goodCount : k ≤ (cands.filter good).card)
    (chosen : Finset S) (hsub : chosen ⊆ cands) (hcard : chosen.card = k) {ω : Ω}
    (hleast : ∀ v ∈ chosen, ∀ w ∈ cands, w ∉ chosen →
        (∑ i ∈ Finset.range m, O.read pref v i ω) ≤ ∑ i ∈ Finset.range m, O.read pref w i ω)
    (hω : ω ∉ O.sepFail pref good bad cands m εcov) :
    ∀ w ∈ chosen, ¬ bad w :=
  avoids_bad_of_not_mem_dSepCompl good bad hdisj cands k (Finset.range m) O.η (O.η + (1 - 2 * O.η) * εcov)
    ((1 / 2 - O.η) * εcov) (O.read pref) (le_of_eq (by ring)) goodCount chosen hsub hcard
    hleast hω

/-- **The separation trigger fires w.h.p.**  From flip-mass separability (`good`
flips nothing, `bad` flips `≥ εcov`), the reads fail to separate with probability at
most `#cands·exp(-2m((½−η)εcov)²)` — the `denoised_loss` conversion of the flip
bounds fed to `dSepCompl_prob`. -/
lemma sepFail_prob [IsRightCancelMul S] (pref : ℕ → S) (hpref : Function.Injective pref)
    (good bad : S → Prop) [DecidablePred good] [DecidablePred bad]
    (hdisj : ∀ v, bad v → ¬ good v)
    (cands : Finset S) (m : ℕ) (εcov : ℝ) (hεcov0 : 0 ≤ εcov)
    (hgoodflip : ∀ v ∈ cands, good v → ∑ i ∈ Finset.range m, O.flip v (pref i) = 0)
    (hbadflip : ∀ v ∈ cands, bad v → (m : ℝ) * εcov ≤ ∑ i ∈ Finset.range m, O.flip v (pref i)) :
    μ.real (O.sepFail pref good bad cands m εcov)
      ≤ (cands.card : ℝ) * Real.exp (-2 * (m : ℝ) * ((1 / 2 - O.η) * εcov) ^ 2) := by
  have hsum : ∀ v, ∑ i ∈ Finset.range m, μ[O.read pref v i]
      = (m : ℝ) * O.η + (1 - 2 * O.η) * ∑ i ∈ Finset.range m, O.flip v (pref i) := by
    intro v
    have h := denoised_loss_eq_flip (μ := μ) m O.η (1 / 2 - O.η) (fun i => O.flip v (pref i))
      (O.read pref v) (fun j => by rw [O.read_mean]; ring)
    rw [h]; ring
  have hgm : ∀ v ∈ cands, good v →
      ∑ i ∈ Finset.range m, μ[O.read pref v i] ≤ (m : ℝ) * O.η := by
    intro v hv hg; rw [hsum v, hgoodflip v hv hg]; simp
  have hbm : ∀ v ∈ cands, bad v →
      (m : ℝ) * (O.η + (1 - 2 * O.η) * εcov) ≤ ∑ i ∈ Finset.range m, μ[O.read pref v i] := by
    intro v hv hb; rw [hsum v]; nlinarith [hbadflip v hv hb, O.hη]
  have h := dSepCompl_prob good bad hdisj cands (Finset.range m) O.η
    (O.η + (1 - 2 * O.η) * εcov) ((1 / 2 - O.η) * εcov) (O.read pref)
    (fun v i => (O.read_meas pref v i).aemeasurable)
    (O.read_indep pref hpref) (O.read_icc pref)
    (by simpa [Finset.card_range] using hgm) (by simpa [Finset.card_range] using hbm)
    (mul_nonneg (by linarith [O.hη]) hεcov0)
  simpa [Oracle.sepFail, Finset.card_range] using h

/-- **Soundness — no bad candidate clears certification.**  The gate certifies on
fresh test prefixes `cpref` with the agreement reads `1 − read`.  A candidate that
flips `≥ εcov` of the test prefixes has agreement-mean `≤ (1−η)−(1−2η)εcov` — the
drift level `β+τ`, *derived* from `η`, `εcov` (no free gate parameters) — so it
clears the admit threshold with probability `≤ α` (`certErr_bound`); union-bounded
over the pool, `≤ #cands·α`. -/
lemma cert_sound [IsRightCancelMul S] (cpref : ℕ → S) (hcpref : Function.Injective cpref)
    (cands : Finset S) (n : ℕ) (εcov α : ℝ)
    (hn : 0 < n) (hα0 : 0 < α) (hα1 : α ≤ 1) (hεcov0 : 0 ≤ εcov) :
    μ.real {ω | ∃ v ∈ cands,
        ((n : ℝ) * εcov ≤ ∑ j ∈ Finset.range n, O.flip v (cpref j))
        ∧ (n : ℝ) * (((1 - O.η) - (1 - 2 * O.η) * εcov) + certMargin n α)
            ≤ ∑ j ∈ Finset.range n, (1 - O.read cpref v j ω)}
      ≤ (cands.card : ℝ) * α := by
  classical
  set cbad : S → Prop := fun v => (n : ℝ) * εcov ≤ ∑ j ∈ Finset.range n, O.flip v (cpref j)
    with hcbad
  set adm : S → Set Ω := fun v => {ω | (n : ℝ) * (((1 - O.η) - (1 - 2 * O.η) * εcov)
      + certMargin n α) ≤ ∑ j ∈ Finset.range n, (1 - O.read cpref v j ω)} with hadm
  have hadmle : ∀ v ∈ cands.filter cbad, μ.real (adm v) ≤ α := by
    intro v hv
    obtain ⟨_, hb⟩ := Finset.mem_filter.mp hv
    have hmeas : ∀ j, AEMeasurable (fun ω => 1 - O.read cpref v j ω) μ :=
      fun j => ((O.read_meas cpref v j).const_sub 1).aemeasurable
    have hindep : iIndepFun (fun j ω => 1 - O.read cpref v j ω) μ :=
      (O.read_indep cpref hcpref v).comp (fun _ => fun x : ℝ => 1 - x)
        (fun _ => measurable_const.sub measurable_id)
    have hIcc : ∀ j, ∀ᵐ ω ∂μ, (fun ω => 1 - O.read cpref v j ω) ω ∈ Set.Icc (0 : ℝ) 1 := by
      intro j; filter_upwards [O.read_icc cpref v j] with ω hω
      rw [Set.mem_Icc] at hω ⊢; exact ⟨by linarith [hω.2], by linarith [hω.1]⟩
    have hstep : ∀ j, μ[fun ω => 1 - O.read cpref v j ω] = 1 - μ[O.read cpref v j] := by
      intro j
      rw [integral_sub (integrable_const 1) (O.read_int cpref v j), integral_const]; simp
    have hread := denoised_loss_eq_flip (μ := μ) n O.η (1 / 2 - O.η)
      (fun j => O.flip v (cpref j)) (O.read cpref v) (fun j => by rw [O.read_mean]; ring)
    have hmean : ∑ j ∈ Finset.range n, μ[fun ω => 1 - O.read cpref v j ω]
        ≤ (n : ℝ) * (((1 - O.η) - (1 - 2 * O.η) * εcov) + 0) := by
      rw [Finset.sum_congr rfl (fun j _ => hstep j), Finset.sum_sub_distrib,
        Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one, hread]
      nlinarith [hb, O.hη]
    have h := certErr_bound (fun j ω => 1 - O.read cpref v j ω) n
      ((1 - O.η) - (1 - 2 * O.η) * εcov) 0 α hmeas hindep hIcc hmean hn hα0 hα1
    simpa only [add_zero] using h
  have hset : {ω | ∃ v ∈ cands, cbad v ∧ ω ∈ adm v} = ⋃ v ∈ cands.filter cbad, adm v := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Finset.mem_filter, exists_prop]
    exact ⟨fun ⟨v, hv, hb, ha⟩ => ⟨v, ⟨hv, hb⟩, ha⟩, fun ⟨v, ⟨hv, hb⟩, ha⟩ => ⟨v, hv, hb, ha⟩⟩
  show μ.real {ω | ∃ v ∈ cands, cbad v ∧ ω ∈ adm v} ≤ (cands.card : ℝ) * α
  calc μ.real {ω | ∃ v ∈ cands, cbad v ∧ ω ∈ adm v}
      = μ.real (⋃ v ∈ cands.filter cbad, adm v) := by rw [hset]
    _ ≤ ∑ v ∈ cands.filter cbad, μ.real (adm v) := measureReal_biUnion_le _ _
    _ ≤ ∑ _v ∈ cands.filter cbad, α := Finset.sum_le_sum hadmle
    _ = ((cands.filter cbad).card : ℝ) * α := by rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ (cands.card : ℝ) * α :=
        mul_le_mul_of_nonneg_right (by exact_mod_cast Finset.card_filter_le _ _) hα0.le

end Oracle

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

end OrthoDFA
