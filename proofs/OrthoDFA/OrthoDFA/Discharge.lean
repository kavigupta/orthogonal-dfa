import OrthoDFA.Hoeffding
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Certification bound, discharged from Mathlib

The certification kernel (`admitProb_le` in the interface) as a theorem, proved
from `wrongDecisive_le`.  A side of `n` prefixes, independent `[0,1]` reads, whose
average mean is at most `β+τ` (drifted at or past tolerance), is admitted — its
sum clearing the Chernoff admit threshold `n((β+τ)+certMargin n α)` — with
probability at most `α`.

Note: this is the Chernoff-threshold certificate.  A certification count spans `n`
*different* prefixes of mixed true class, so it is a sum of *non-identical*
Bernoullis — not a binomial.  The implementation tests it against a `Bin(n, β+τ)`
reference (the exact-binomial tail), which admits marginally more readily;
justifying *that* non-i.i.d. tail's domination by the exact binomial needs
Hoeffding's 1956 theorem, which is not in Mathlib.  The Chernoff reference used
here (MGF domination via AM–GM) carries the same `≤ α` guarantee.  This is a real
mathematical subtlety — distinct from the vote's fpr/fnr, which is genuinely
binomial (`k` i.i.d. reads of one string) and only bounded conservatively here to
keep the trusted base at Lean core.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Chernoff admit-threshold margin: the per-read excess over `β+τ` at which the
sub-Gaussian tail of `n` reads equals the level `α`. -/
noncomputable def certMargin (n : ℕ) (α : ℝ) : ℝ :=
  Real.sqrt (Real.log (1 / α) / (2 * (n : ℝ)))

/-- **T1 (discharged).**  A drifted side is admitted with probability at most `α`. -/
theorem certErr_bound
    (Xc : ℕ → Ω → ℝ) (n : ℕ) (β τ α : ℝ)
    (hmeas : ∀ i, AEMeasurable (Xc i) μ)
    (h_indep : iIndepFun Xc μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, Xc i ω ∈ Set.Icc (0 : ℝ) 1)
    (hmean_sum : ∑ i ∈ Finset.range n, μ[Xc i] ≤ (n : ℝ) * (β + τ))
    (hn : 0 < n) (hα0 : 0 < α) (hα1 : α ≤ 1) :
    μ.real {ω | (n : ℝ) * ((β + τ) + certMargin n α)
        ≤ ∑ i ∈ Finset.range n, Xc i ω} ≤ α := by
  set t := certMargin n α with ht
  have htnn : 0 ≤ t := Real.sqrt_nonneg _
  have hbound :=
    wrongDecisive_le Xc (Finset.range n) (β + τ) t hmeas h_indep hIcc
      (by simpa [Finset.card_range] using hmean_sum) htnn
  rw [Finset.card_range] at hbound
  refine hbound.trans_eq ?_
  have hnne : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  have h1α : (1 : ℝ) ≤ 1 / α := by rw [le_div_iff₀ hα0, one_mul]; exact hα1
  have harg : (0 : ℝ) ≤ Real.log (1 / α) / (2 * (n : ℝ)) := by
    apply div_nonneg (Real.log_nonneg h1α)
    positivity
  have ht2 : t ^ 2 = Real.log (1 / α) / (2 * (n : ℝ)) := by
    rw [ht, certMargin]; exact Real.sq_sqrt harg
  rw [ht2]
  have hexp : -2 * (n : ℝ) * (Real.log (1 / α) / (2 * (n : ℝ))) = Real.log α := by
    rw [one_div, Real.log_inv]; field_simp
  rw [hexp, Real.exp_log hα0]

/-- **Decisiveness / placement (non-member side).**  Reads independent, each in
`[0,1]`, with total mean at most `k(β-s)` (a non-member under an accept-preserving
family: per-read mean `β-s`).  The vote failing to decide REJECT (sum reaching
`k(β-τ)`, i.e. indecisive or wrongly accepted) has probability at most
`exp(-2k(s-τ)²)`. -/
theorem misplacedNonmember_le {ι : Type*}
    (X : ι → Ω → ℝ) (idx : Finset ι) (β s τ : ℝ)
    (hmeas : ∀ i, AEMeasurable (X i) μ)
    (h_indep : iIndepFun X μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    (hmean_sum : ∑ i ∈ idx, μ[X i] ≤ (idx.card : ℝ) * (β - s))
    (hτs : τ ≤ s) :
    μ.real {ω | (idx.card : ℝ) * (β - τ) ≤ ∑ i ∈ idx, X i ω}
      ≤ Real.exp (-2 * (idx.card : ℝ) * (s - τ) ^ 2) := by
  have h := wrongDecisive_le X idx (β - s) (s - τ) hmeas h_indep hIcc hmean_sum (by linarith)
  have heq : (β - s) + (s - τ) = β - τ := by ring
  simpa only [heq] using h

/-- **Decisiveness / placement (member side).**  Reads independent, each in
`[0,1]`, with total mean at least `k(β+s)` (a member under an accept-preserving
family).  The vote failing to decide ACCEPT (sum at most `k(β+τ)`) has probability
at most `exp(-2k(s-τ)²)`.  Proved from the non-member side via the reflection
`X ↦ 1-X`. -/
theorem misplacedMember_le {ι : Type*}
    (X : ι → Ω → ℝ) (idx : Finset ι) (β s τ : ℝ)
    (hmeas : ∀ i, AEMeasurable (X i) μ)
    (h_indep : iIndepFun X μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    (hmean_sum : (idx.card : ℝ) * (β + s) ≤ ∑ i ∈ idx, μ[X i])
    (hτs : τ ≤ s) :
    μ.real {ω | ∑ i ∈ idx, X i ω ≤ (idx.card : ℝ) * (β + τ)}
      ≤ Real.exp (-2 * (idx.card : ℝ) * (s - τ) ^ 2) := by
  set X' : ι → Ω → ℝ := fun i ω => 1 - X i ω with hX'
  have hmeas' : ∀ i, AEMeasurable (X' i) μ := fun i => (hmeas i).const_sub 1
  have hindep' : iIndepFun X' μ :=
    h_indep.comp (fun _ => fun x : ℝ => 1 - x) (fun _ => measurable_const.sub measurable_id)
  have hIcc' : ∀ i, ∀ᵐ ω ∂μ, X' i ω ∈ Set.Icc (0 : ℝ) 1 := by
    intro i; filter_upwards [hIcc i] with ω hω
    simp only [hX', Set.mem_Icc]; constructor <;> [linarith [hω.2]; linarith [hω.1]]
  have hint : ∀ i, Integrable (X i) μ := fun i =>
    MeasureTheory.Integrable.of_mem_Icc 0 1 (hmeas i) (hIcc i)
  have hmeanX' : ∀ i, μ[X' i] = 1 - μ[X i] := by
    intro i; simp only [hX']
    rw [integral_sub (integrable_const (1 : ℝ)) (hint i), integral_const]
    have huniv : μ.real Set.univ = 1 := by
      simp [MeasureTheory.measureReal_def, measure_univ]
    rw [huniv]; ring
  have hmean_sum' : ∑ i ∈ idx, μ[X' i] ≤ (idx.card : ℝ) * ((1 - β) - s) := by
    have hrw : ∑ i ∈ idx, μ[X' i] = (idx.card : ℝ) - ∑ i ∈ idx, μ[X i] := by
      rw [Finset.sum_congr rfl (fun i _ => hmeanX' i), Finset.sum_sub_distrib,
        Finset.sum_const]
      simp [nsmul_eq_mul]
    rw [hrw]; nlinarith [hmean_sum]
  have h := wrongDecisive_le X' idx ((1 - β) - s) (s - τ) hmeas' hindep' hIcc' hmean_sum'
    (by linarith)
  refine le_trans (le_of_eq ?_) h
  congr 1
  ext ω
  simp only [Set.mem_setOf_eq, hX', Finset.sum_sub_distrib, Finset.sum_const,
    nsmul_eq_mul, mul_one]
  constructor
  · intro hle; nlinarith [hle]
  · intro hge; nlinarith [hge]

end OrthoDFA
#print axioms OrthoDFA.certErr_bound
#print axioms OrthoDFA.misplacedNonmember_le
#print axioms OrthoDFA.misplacedMember_le
