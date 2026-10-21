import OrthoDFA.Proofs.Nearest
import Mathlib.Algebra.Order.Chebyshev

/-!
# The count rule

The family is the `k` candidates whose reads disagree with the seed's on the fewest prefixes of
the table.  Termination needs only that the family's flips are few on average, and that is what
this rule delivers: an accept-preserving row and a flipping one are compared through the same
seed reads, so the seed's errors shift both alike, and what is left of each row's count is a sum
of independent terms, one per prefix, sub-Gaussian at a quarter each.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped NNReal

section Signed

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

lemma integrable_of_bit_comp {e : Ω → ℝ} (he : Measurable e)
    (hebit : ∀ᵐ ω ∂μ, e ω = 0 ∨ e ω = 1) (f : ℝ → ℝ) (hf : Measurable f) :
    Integrable (fun ω => f (e ω)) μ := by
  refine MeasureTheory.Integrable.of_bound (C := |f 0| + |f 1|) (hf.comp he).aestronglyMeasurable ?_
  filter_upwards [hebit] with ω hω
  rcases hω with h | h <;> rw [h, Real.norm_eq_abs] <;> linarith [abs_nonneg (f 0), abs_nonneg (f 1)]

/-- With `e` a bit independent of `b ∈ [0, 1]`,

    E[exp(s(e − E e) + t(1 − 2e)(b − E b))] ≤ exp((s² + t²)/8). -/
theorem mgf_signed_le {e b : Ω → ℝ} (he : Measurable e) (hb : Measurable b)
    (hebit : ∀ᵐ ω ∂μ, e ω = 0 ∨ e ω = 1) (hbI : ∀ᵐ ω ∂μ, b ω ∈ Set.Icc (0 : ℝ) 1)
    (hind : IndepFun e b μ) (s t : ℝ) :
    μ[fun ω => Real.exp (s * (e ω - μ[e])) * Real.exp (t * ((1 - 2 * e ω) * (b ω - μ[b])))]
      ≤ Real.exp ((s ^ 2 + t ^ 2) / 8) := by
  set em : ℝ := μ[e] with hem
  set bm : ℝ := μ[b] with hbm
  have heI : ∀ᵐ ω ∂μ, e ω ∈ Set.Icc (0 : ℝ) 1 := by
    filter_upwards [hebit] with ω hω
    rcases hω with h | h <;> simp [h]
  -- Hoeffding's lemma for each bit
  have hsubB := hasSubgaussianMGF_of_mem_Icc (X := b) (μ := μ) hb.aemeasurable hbI
  have hsubE := hasSubgaussianMGF_of_mem_Icc (X := e) (μ := μ) he.aemeasurable heI
  have hc : ((‖(1 : ℝ) - 0‖₊ / 2) ^ 2 : ℝ≥0) = (1 / 4 : ℝ≥0) := by norm_num
  rw [hc] at hsubB hsubE
  have hB : ∀ u : ℝ, μ[fun ω => Real.exp (u * (b ω - bm))] ≤ Real.exp (u ^ 2 / 8) := by
    intro u
    have := hsubB.mgf_le u
    unfold mgf at this
    refine le_trans this (le_of_eq ?_)
    congr 1; push_cast; ring
  have hE : μ[fun ω => Real.exp (s * (e ω - em))] ≤ Real.exp (s ^ 2 / 8) := by
    have := hsubE.mgf_le s
    unfold mgf at this
    refine le_trans this (le_of_eq ?_)
    congr 1; push_cast; ring
  -- split on the seed's bit
  set g₀ : ℝ → ℝ := fun x => Real.exp (s * (x - em)) * (1 - x) with hg₀
  set g₁ : ℝ → ℝ := fun x => Real.exp (s * (x - em)) * x with hg₁
  have hsplit : (fun ω => Real.exp (s * (e ω - em)) * Real.exp (t * ((1 - 2 * e ω) * (b ω - bm))))
      =ᵐ[μ] fun ω => g₀ (e ω) * Real.exp (t * (b ω - bm))
        + g₁ (e ω) * Real.exp (-t * (b ω - bm)) := by
    filter_upwards [hebit] with ω hω
    rcases hω with h | h <;> simp only [hg₀, hg₁, h] <;> ring_nf
  have hmg₀ : Measurable g₀ := by fun_prop
  have hmg₁ : Measurable g₁ := by fun_prop
  have hint₀ := integrable_of_bit_comp he hebit g₀ hmg₀
  have hint₁ := integrable_of_bit_comp he hebit g₁ hmg₁
  have hexpB : ∀ u : ℝ, Integrable (fun ω => Real.exp (u * (b ω - bm))) μ := fun u =>
    hsubB.integrable_exp_mul u
  have hprod : ∀ (g : ℝ → ℝ) (u : ℝ), Measurable g →
      μ[fun ω => g (e ω) * Real.exp (u * (b ω - bm))]
        = μ[fun ω => g (e ω)] * μ[fun ω => Real.exp (u * (b ω - bm))] := by
    intro g u hg
    exact hind.integral_fun_comp_mul_comp (f := g) (g := fun y => Real.exp (u * (y - bm)))
      he.aemeasurable hb.aemeasurable hg.aestronglyMeasurable
      (by fun_prop : Measurable fun y => Real.exp (u * (y - bm))).aestronglyMeasurable
  have hg₀0 : 0 ≤ μ[fun ω => g₀ (e ω)] := by
    refine integral_nonneg_of_ae ?_
    filter_upwards [hebit] with ω hω
    rcases hω with h | h <;> simp [hg₀, h, (Real.exp_pos _).le]
  have hg₁0 : 0 ≤ μ[fun ω => g₁ (e ω)] := by
    refine integral_nonneg_of_ae ?_
    filter_upwards [hebit] with ω hω
    rcases hω with h | h <;> simp [hg₁, h, (Real.exp_pos _).le]
  have hsum : μ[fun ω => g₀ (e ω)] + μ[fun ω => g₁ (e ω)]
      = μ[fun ω => Real.exp (s * (e ω - em))] := by
    rw [← integral_add hint₀ hint₁]
    refine integral_congr_ae ?_
    filter_upwards with ω
    simp only [hg₀, hg₁]; ring
  have hprodInt : ∀ (g : ℝ → ℝ) (u : ℝ), Measurable g →
      Integrable (fun ω => g (e ω) * Real.exp (u * (b ω - bm))) μ := by
    intro g u hg
    refine (hexpB u).bdd_mul (c := |g 0| + |g 1|) (hg.comp he).aestronglyMeasurable ?_
    filter_upwards [hebit] with ω hω
    rcases hω with h | h <;> rw [h, Real.norm_eq_abs] <;>
      linarith [abs_nonneg (g 0), abs_nonneg (g 1)]
  rw [integral_congr_ae hsplit, integral_add (hprodInt g₀ t hmg₀) (hprodInt g₁ (-t) hmg₁),
    hprod g₀ t hmg₀, hprod g₁ (-t) hmg₁]
  calc μ[fun ω => g₀ (e ω)] * μ[fun ω => Real.exp (t * (b ω - bm))]
        + μ[fun ω => g₁ (e ω)] * μ[fun ω => Real.exp (-t * (b ω - bm))]
      ≤ μ[fun ω => g₀ (e ω)] * Real.exp (t ^ 2 / 8)
        + μ[fun ω => g₁ (e ω)] * Real.exp ((-t) ^ 2 / 8) := by
        gcongr
        · exact hB t
        · exact hB (-t)
    _ = μ[fun ω => Real.exp (s * (e ω - em))] * Real.exp (t ^ 2 / 8) := by
        rw [← hsum]; ring_nf
    _ ≤ Real.exp (s ^ 2 / 8) * Real.exp (t ^ 2 / 8) :=
        mul_le_mul_of_nonneg_right hE (Real.exp_pos _).le
    _ = Real.exp ((s ^ 2 + t ^ 2) / 8) := by rw [← Real.exp_add]; ring_nf

/-- The sign-flipped read, shifted by `κ` of the seed's own bit, is sub-Gaussian at
`(1 + κ²)/4`. -/
theorem subG_signed {e b : Ω → ℝ} (he : Measurable e) (hb : Measurable b)
    (hebit : ∀ᵐ ω ∂μ, e ω = 0 ∨ e ω = 1) (hbI : ∀ᵐ ω ∂μ, b ω ∈ Set.Icc (0 : ℝ) 1)
    (hind : IndepFun e b μ) (κ : ℝ) :
    HasSubgaussianMGF (fun ω => (1 - 2 * e ω) * (b ω - μ[b]) + κ * (e ω - μ[e]))
      ((1 + κ ^ 2) / 4).toNNReal μ where
  integrable_exp_mul t := by
    have hbd : ∀ᵐ ω ∂μ, (1 - 2 * e ω) * (b ω - μ[b]) + κ * (e ω - μ[e])
        ∈ Set.Icc (-(1 + |κ| * 2)) (1 + |κ| * 2) := by
      have hbm0 : 0 ≤ μ[b] := integral_nonneg_of_ae (by filter_upwards [hbI] with ω h; exact h.1)
      have hbm1 : μ[b] ≤ 1 := by
        have : μ[b] ≤ μ[fun _ => (1 : ℝ)] :=
          integral_mono_ae (MeasureTheory.Integrable.of_mem_Icc 0 1 hb.aemeasurable hbI)
            (integrable_const _) (by filter_upwards [hbI] with ω h; exact h.2)
        simpa using this
      have hem0 : 0 ≤ μ[e] := integral_nonneg_of_ae (by
        filter_upwards [hebit] with ω h; rcases h with h | h <;> simp [h])
      have hem1 : μ[e] ≤ 1 := by
        have : μ[e] ≤ μ[fun _ => (1 : ℝ)] :=
          integral_mono_ae (integrable_of_bit_comp he hebit id measurable_id)
            (integrable_const _) (by filter_upwards [hebit] with ω h; rcases h with h | h <;> simp [h])
        simpa using this
      filter_upwards [hebit, hbI] with ω h1 h2
      have hk1 : |κ * (e ω - μ[e])| ≤ |κ| * 2 := by
        rw [abs_mul]
        refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg κ)
        rcases h1 with h | h <;> rw [h, abs_le] <;> constructor <;> linarith
      have hk2 : |(1 - 2 * e ω) * (b ω - μ[b])| ≤ 1 := by
        rw [abs_mul]
        have : |1 - 2 * e ω| = 1 := by rcases h1 with h | h <;> rw [h] <;> norm_num
        rw [this, one_mul, abs_le]; constructor <;> linarith [h2.1, h2.2]
      have := abs_add_le ((1 - 2 * e ω) * (b ω - μ[b])) (κ * (e ω - μ[e]))
      rw [Set.mem_Icc]
      constructor <;> [linarith [neg_abs_le ((1 - 2 * e ω) * (b ω - μ[b]) + κ * (e ω - μ[e]))];
        linarith [le_abs_self ((1 - 2 * e ω) * (b ω - μ[b]) + κ * (e ω - μ[e]))]]
    exact integrable_exp_mul_of_mem_Icc (by fun_prop) hbd
  mgf_le t := by
    unfold mgf
    have h := mgf_signed_le (μ := μ) he hb hebit hbI hind (t * κ) t
    have heq : (fun ω => Real.exp (t * ((1 - 2 * e ω) * (b ω - μ[b]) + κ * (e ω - μ[e]))))
        = fun ω => Real.exp (t * κ * (e ω - μ[e])) * Real.exp (t * ((1 - 2 * e ω) * (b ω - μ[b]))) := by
      funext ω; rw [← Real.exp_add]; ring_nf
    rw [heq]
    refine le_trans h (le_of_eq ?_)
    rw [Real.coe_toNNReal _ (by positivity)]
    ring_nf

end Signed

section Rows

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

/-- What the seed's read of `p` predicts a clean row disagrees with it at:

    r + e(1 − 2r),

`r` the prefix's rate and `e` the seed's noise bit at `p`. -/
noncomputable def seedRef (O : Oracle μ S) (p : S) (ω : Ω) : ℝ :=
  O.rate p + O.noise p ω * (1 - 2 * O.rate p)

/-- What a flip at `p` adds to a row's mean disagreement: `(1 − r' − r)(1 − 2r)`. -/
noncomputable def flipShift (O : Oracle μ S) (v p : S) : ℝ :=
  O.flip v p * ((1 - O.rate (p * v) - O.rate p) * (1 - 2 * O.rate p))

/-- A row's disagreement at `p` less `seedRef` and `flipShift`. -/
noncomputable def rowTerm (O : Oracle μ S) (v p : S) (ω : Ω) : ℝ :=
  (1 - 2 * O.flip v p) * ((1 - 2 * O.noise p ω) * (O.noise (p * v) ω - O.rate (p * v))
    + 2 * O.flip v p * (1 - O.rate (p * v) - O.rate p) * (O.noise p ω - O.rate p))

lemma seedLoss_split (O : Oracle μ S) (v p : S) :
    ∀ᵐ ω ∂μ, seedLoss O 1 2 v p ω = seedRef O p ω + flipShift O v p + rowTerm O v p ω := by
  filter_upwards [seedLoss_eq_disagree O (by norm_num : (1 : ℕ) < 2) v p, O.noise_bit p,
    O.noise_bit (p * v)] with ω hω he hb
  rw [hω]
  unfold seedRef flipShift rowTerm Oracle.mq Oracle.flip
  rcases O.label_bit (p * v) with hx | hx <;> rcases O.label_bit p with hy | hy
  · have hr := O.rate_eq_of_label_eq (hx.trans hy.symm)
    rw [hx, hy, hr]
    rcases he with he | he <;> rcases hb with hb | hb <;> rw [he, hb] <;> ring
  · rw [hx, hy]
    rcases he with he | he <;> rcases hb with hb | hb <;> rw [he, hb] <;> ring
  · rw [hx, hy]
    rcases he with he | he <;> rcases hb with hb | hb <;> rw [he, hb] <;> ring
  · have hr := O.rate_eq_of_label_eq (hx.trans hy.symm)
    rw [hx, hy, hr]
    rcases he with he | he <;> rcases hb with hb | hb <;> rw [he, hb] <;> ring

/-- The row's term at `p` is sub-Gaussian at a quarter, and at a quarter more where `v`
flips `p`. -/
theorem subG_rowTerm (O : Oracle μ S) (v p : S) (hv : p * v ≠ p) :
    HasSubgaussianMGF (rowTerm O v p) (1 / 4 + O.flip v p).toNNReal μ := by
  have hind : IndepFun (O.noise p) (O.noise (p * v)) μ := O.noise_indep.indepFun hv.symm
  have hbI := O.noise_icc (p * v)
  set G : ℝ := 1 - O.rate (p * v) - O.rate p with hG
  have hle1 : ∀ w, O.rate w ≤ 1 := by
    intro w
    rw [← O.noise_mean w]
    have : μ[O.noise w] ≤ μ[fun _ => (1 : ℝ)] :=
      integral_mono_ae (O.noise_int w) (integrable_const _)
        (by filter_upwards [O.noise_icc w] with ω h; exact h.2)
    simpa using this
  have hG1 : G ^ 2 ≤ 1 := by
    have := O.rate_nonneg (p * v); have := O.rate_nonneg p
    have := hle1 (p * v); have := hle1 p
    nlinarith
  have key := subG_signed (μ := μ) (O.noise_meas p) (O.noise_meas (p * v)) (O.noise_bit p) hbI
    hind (2 * O.flip v p * G)
  rw [O.noise_mean p, O.noise_mean (p * v)] at key
  have hmono : ∀ {X : Ω → ℝ} {c d : ℝ≥0}, HasSubgaussianMGF X c μ → c ≤ d →
      HasSubgaussianMGF X d μ := by
    intro X c d h hcd
    exact ⟨h.integrable_exp_mul, fun t => le_trans (h.mgf_le t)
      (Real.exp_le_exp.2 (by gcongr))⟩
  rcases O.flip_bit v p with hf | hf
  · have e : rowTerm O v p = fun ω => (1 - 2 * O.noise p ω) * (O.noise (p * v) ω - O.rate (p * v))
        + 2 * O.flip v p * G * (O.noise p ω - O.rate p) := by
      funext ω; unfold rowTerm; rw [hf]; ring
    rw [e]
    refine hmono key ?_
    rw [hf]; simp
  · have e : rowTerm O v p = -fun ω => (1 - 2 * O.noise p ω) * (O.noise (p * v) ω - O.rate (p * v))
        + 2 * O.flip v p * G * (O.noise p ω - O.rate p) := by
      funext ω; unfold rowTerm; rw [hf]; simp only [Pi.neg_apply]; ring
    rw [e]
    refine hmono key.neg ?_
    rw [hf]
    refine Real.toNNReal_le_toNNReal ?_
    nlinarith

end Rows

section Tails

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

/-- The disagreement `seedRef` predicts over the table: every clean row's mean, read off the
seed's own noise. -/
noncomputable def seedBase (O : Oracle μ S) (P : Finset S) (ω : Ω) : ℝ :=
  ∑ p ∈ P, seedRef O p ω

lemma measurable_rowTerm_block (O : Oracle μ S) (v p : S) :
    Measurable[⨆ w ∈ ({p, p * v} : Finset S), MeasurableSpace.comap (O.noise w) inferInstance]
      (rowTerm O v p) := by
  have h1 : Measurable[⨆ w ∈ ({p, p * v} : Finset S),
      MeasurableSpace.comap (O.noise w) inferInstance] (O.noise p) :=
    (comap_measurable (O.noise p)).mono (le_iSup₂ (f := fun w (_ : w ∈ ({p, p * v} : Finset S)) =>
      MeasurableSpace.comap (O.noise w) inferInstance) p (by simp)) le_rfl
  have h2 : Measurable[⨆ w ∈ ({p, p * v} : Finset S),
      MeasurableSpace.comap (O.noise w) inferInstance] (O.noise (p * v)) :=
    (comap_measurable (O.noise (p * v))).mono (le_iSup₂ (f := fun w (_ : w ∈ ({p, p * v} :
      Finset S)) => MeasurableSpace.comap (O.noise w) inferInstance) (p * v) (by simp)) le_rfl
  unfold rowTerm
  exact measurable_const.mul (((measurable_const.sub (measurable_const.mul h1)).mul
    (h2.sub measurable_const)).add ((measurable_const.mul measurable_const).mul
      (h1.sub measurable_const)))

/-- A row's terms are independent across the table's prefixes: each reads `p` and `p · v`, and
those pairs are disjoint, as in `seedLoss_indep`. -/
lemma rowTerm_indep {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    {P : Finset S} (hP : ∀ p ∈ P, p ∈ Pre) (v : S) (hvS : v ∈ insert 1 Suf) :
    iIndepFun (fun p : {p // p ∈ P} => rowTerm O v p.val) μ := by
  refine iIndepFun_blocks (X := O.noise) (fun w => O.noise_meas w) O.noise_indep
    (fun p : {p // p ∈ P} => {p.val, p.val * v}) ?_ _ (fun p => measurable_rowTerm_block O v p.val)
  intro a b hab
  refine Finset.disjoint_left.2 (fun w hw hw' => ?_)
  simp only [Finset.mem_insert, Finset.mem_singleton] at hw hw'
  rcases hw with rfl | rfl <;> rcases hw' with hb | hb
  · exact hab (Subtype.ext hb)
  · by_cases hv1 : v = 1
    · exact hab (Subtype.ext (by rw [hb, hv1, mul_one]))
    · exact flat_ne_of_ne_one hflat (hP _ b.property) (hP _ a.property) hvS hv1 hb.symm
  · by_cases hv1 : v = 1
    · exact hab (Subtype.ext (by rw [← hb, hv1, mul_one]))
    · exact flat_ne_of_ne_one hflat (hP _ a.property) (hP _ b.property) hvS hv1 hb
  · exact hab (Subtype.ext (mul_right_cancel hb))

lemma screenCount_split (O : Oracle μ S) (P : Finset S) (v : S) :
    ∀ᵐ ω ∂μ, (screenCount O.mq P v ω : ℝ) - seedBase O P ω
      = ∑ p ∈ P, flipShift O v p + ∑ p ∈ P, rowTerm O v p ω := by
  have hall : ∀ᵐ ω ∂μ, ∀ p ∈ P,
      seedLoss O 1 2 v p ω = seedRef O p ω + flipShift O v p + rowTerm O v p ω :=
    (Filter.eventually_all_finset P).2 (fun p _ => seedLoss_split O v p)
  filter_upwards [hall] with ω hω
  rw [screenCount_eq_sum O (by norm_num : (1 : ℕ) < 2) P v ω, Finset.sum_congr rfl hω,
    Finset.sum_add_distrib, Finset.sum_add_distrib, seedBase]
  ring

/-- Sub-Gaussian at a quarter per prefix and one more per flip, so the row's sum of terms
deviates by `u` with chance at most `exp(−2u²/(#P + 4Φ))`, `Φ` the row's flips on the table. -/
theorem rowTerm_sum_tail {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    {P : Finset S} (hP : ∀ p ∈ P, p ∈ Pre) (v : S) (hvS : v ∈ insert 1 Suf) (hv : v ≠ 1)
    (u : ℝ) (hu : 0 ≤ u) :
    μ.real {ω | u ≤ ∑ p ∈ P, rowTerm O v p ω}
        ≤ Real.exp (-(2 * u ^ 2 / ((P.card : ℝ) + 4 * ∑ p ∈ P, O.flip v p)))
      ∧ μ.real {ω | ∑ p ∈ P, rowTerm O v p ω ≤ -u}
        ≤ Real.exp (-(2 * u ^ 2 / ((P.card : ℝ) + 4 * ∑ p ∈ P, O.flip v p))) := by
  classical
  have hind := rowTerm_indep hflat O hP v hvS
  have hsub : ∀ p ∈ (Finset.univ : Finset {p // p ∈ P}),
      HasSubgaussianMGF (rowTerm O v p.val) (1 / 4 + O.flip v p.val).toNNReal μ :=
    fun p _ => subG_rowTerm O v p.val (mul_ne_self p.val v hv)
  have hc : ((∑ p ∈ (Finset.univ : Finset {p // p ∈ P}), (1 / 4 + O.flip v p.val).toNNReal :
      ℝ≥0) : ℝ) = (P.card : ℝ) / 4 + ∑ p ∈ P, O.flip v p := by
    push_cast
    rw [Finset.sum_congr rfl (fun p _ => Real.coe_toNNReal _
      (by linarith [(flip_icc O v p.val).1]))]
    rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_coe,
      Finset.sum_coe_sort P (fun p => O.flip v p)]
    simp; ring
  have hsumeq : ∀ ω, ∑ p ∈ (Finset.univ : Finset {p // p ∈ P}), rowTerm O v p.val ω
      = ∑ p ∈ P, rowTerm O v p ω := fun ω => Finset.sum_coe_sort P (fun p => rowTerm O v p ω)
  have hexp : -u ^ 2 / (2 * ((P.card : ℝ) / 4 + ∑ p ∈ P, O.flip v p))
      = -(2 * u ^ 2 / ((P.card : ℝ) + 4 * ∑ p ∈ P, O.flip v p)) := by
    rw [neg_div]; congr 1; field_simp; ring
  constructor
  · have h := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hind hsub hu
    rw [hc, hexp] at h
    simpa only [hsumeq] using h
  · have hind' : iIndepFun (fun p : {p // p ∈ P} => fun ω => -rowTerm O v p.val ω) μ :=
      hind.comp (fun _ x => -x) (fun _ => measurable_neg)
    have hsub' : ∀ p ∈ (Finset.univ : Finset {p // p ∈ P}),
        HasSubgaussianMGF (fun ω => -rowTerm O v p.val ω) (1 / 4 + O.flip v p.val).toNNReal μ :=
      fun p hp => (hsub p hp).neg
    have h := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hind' hsub' hu
    rw [hc, hexp] at h
    refine le_trans (measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)) h
    simp only [Set.mem_setOf_eq] at hω ⊢
    rw [Finset.sum_neg_distrib, hsumeq]
    linarith

/-- A clean row's flip shifts vanish, and a flipping row's add at least `gap·(1 − 2η)` a flip. -/
lemma flipShift_sum_ge (O : Oracle μ S) (P : Finset S) (v : S) (hsig : O.η ≤ 1 / 2) :
    O.gap * (1 - 2 * O.η) * ∑ p ∈ P, O.flip v p ≤ ∑ p ∈ P, flipShift O v p := by
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum (fun p _ => ?_)
  unfold flipShift
  rcases O.flip_bit v p with h | h
  · rw [h]; simp
  · rw [h, ← gap_of_flip O h]
    have hgap0 : 0 ≤ O.gap := by linarith [gap_ge O]
    have := O.rate_le_eta p
    nlinarith

lemma flipShift_sum_clean (O : Oracle μ S) (P : Finset S) (v : S)
    (hclean : ∀ p ∈ P, O.flip v p = 0) : ∑ p ∈ P, flipShift O v p = 0 :=
  Finset.sum_eq_zero (fun p hp => by unfold flipShift; rw [hclean p hp, zero_mul])

end Tails

section Select

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

/-- Markov over weighted events: `μ{T ≤ ∑ wᵢ 1[Aᵢ]} ≤ ∑ wᵢ μ(Aᵢ) / T`. -/
lemma markov_weighted {ι : Type*} (I : Finset ι) (w : ι → ℝ) (hw : ∀ i ∈ I, 0 ≤ w i)
    (A : ι → Set Ω) (hA : ∀ i ∈ I, MeasurableSet (A i)) {T : ℝ} (hT : 0 < T) :
    μ.real {ω | T ≤ ∑ i ∈ I, w i * (A i).indicator 1 ω} ≤ (∑ i ∈ I, w i * μ.real (A i)) / T := by
  classical
  set f : Ω → ℝ := fun ω => ∑ i ∈ I, w i * (A i).indicator 1 ω with hf
  have hInt : ∀ i ∈ I, Integrable (fun ω => w i * (A i).indicator (1 : Ω → ℝ) ω) μ := by
    intro i hi
    have h1 : Integrable ((A i).indicator (1 : Ω → ℝ)) μ :=
      (integrable_const (1 : ℝ)).indicator (hA i hi)
    exact h1.const_mul (w i)
  have hint : Integrable f μ := integrable_finset_sum I hInt
  have hnn : 0 ≤ᵐ[μ] f := Filter.Eventually.of_forall (fun ω => Finset.sum_nonneg (fun i hi =>
    mul_nonneg (hw i hi) (Set.indicator_nonneg (fun _ _ => zero_le_one) ω)))
  have hmk := mul_meas_ge_le_integral_of_nonneg hnn hint T
  have hI : ∫ ω, f ω ∂μ = ∑ i ∈ I, w i * μ.real (A i) := by
    rw [hf]
    simp only
    rw [integral_finset_sum I hInt]
    refine Finset.sum_congr rfl (fun i hi => ?_)
    rw [integral_const_mul, integral_indicator_one (hA i hi)]
  rw [hI] at hmk
  rw [le_div_iff₀ hT]
  linarith

lemma measurable_screenCount_real (O : Oracle μ S) (P : Finset S) (v : S) :
    Measurable (fun ω => (screenCount O.mq P v ω : ℝ)) := by
  have e : (fun ω => (screenCount O.mq P v ω : ℝ)) = fun ω => ∑ p ∈ P, seedLoss O 1 2 v p ω := by
    funext ω; exact screenCount_eq_sum O (by norm_num) P v ω
  rw [e]
  exact Finset.measurable_sum P (fun p _ => seedLoss_meas O 1 2 v p)

lemma measurable_seedBase (O : Oracle μ S) (P : Finset S) : Measurable (seedBase O P) := by
  unfold seedBase seedRef
  exact Finset.measurable_sum P (fun p _ =>
    measurable_const.add ((O.noise_meas p).mul measurable_const))

/-- The `k` least of a loss sit below any level that `k` of the candidates sit below. -/
lemma leastLossSubset_lt {ℓ : S → ℝ} {cands T : Finset S} {k : ℕ} (hk : k ≤ cands.card)
    (hT : T ⊆ cands) (hTk : k ≤ T.card) {τ : ℝ} (hTτ : ∀ w ∈ T, ℓ w < τ) :
    ∀ v ∈ leastLossSubset ℓ cands k, ℓ v < τ := by
  classical
  intro v hv
  by_contra hge
  push_neg at hge
  have hvT : v ∉ T := fun h => absurd (hTτ v h) (not_lt.2 hge)
  by_cases hsub : T ⊆ leastLossSubset ℓ cands k
  · have h1 : T ⊆ (leastLossSubset ℓ cands k).erase v := fun w hw =>
      Finset.mem_erase.2 ⟨fun h => hvT (h ▸ hw), hsub hw⟩
    have h2 := Finset.card_le_card h1
    rw [Finset.card_erase_of_mem hv, leastLossSubset_card ℓ cands k hk] at h2
    have : 0 < k := lt_of_lt_of_le (Finset.card_pos.2 ⟨v, hv⟩)
      (le_of_eq (leastLossSubset_card ℓ cands k hk))
    omega
  · obtain ⟨w, hwT, hwF⟩ := Finset.not_subset.1 hsub
    have := leastLossSubset_least ℓ cands k hk v hv w (hT hwT) hwF
    linarith [hTτ w hwT]

/-- The count rule's family flips little on average.  At a fixed table `P` and pool `C`, with
`k` of the pool's rows clean on the table (`A`), the `k` rows closest to the seed's reads flip

    ∑_{v ∈ F} Φ_v ≤ k Φ₀ + T

but for chance at most

    #A·e^{−2u²/n} / (#A − k + 1) + (1/T) ∑_{v : Φ_v ≥ Φ₀} Φ_v e^{−2(γΦ_v − u)²/(n + 4Φ_v)},

`Φ_v` a row's flips on the table, `n = #P` and `γ = gap·(1 − 2η)`: every clean row but `k − 1`
sits within `u` above the seed's own prediction, so every member does, and a row flipping
`Φ_v` gets there only in its lower tail. -/
theorem countRule_flips {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (hsig : O.η ≤ 1 / 2) {P C A : Finset S} (hP : ∀ p ∈ P, p ∈ Pre)
    (hV : ∀ v ∈ C, v ∈ insert 1 Suf) (hA : A ⊆ C.erase 1)
    (hAclean : ∀ w ∈ A, ∀ p ∈ P, O.flip w p = 0) (k : ℕ) (hkA : k ≤ A.card)
    (u Φ₀ T : ℝ) (hu : 0 ≤ u) (hΦ₀ : 0 ≤ Φ₀) (hT : 0 < T)
    (hheavy : ∀ v ∈ C.erase 1, Φ₀ ≤ ∑ p ∈ P, O.flip v p →
      u ≤ O.gap * (1 - 2 * O.η) * ∑ p ∈ P, O.flip v p) :
    μ.real {ω | (k : ℝ) * Φ₀ + T < ∑ v ∈ leastLossSubset (fun v => (screenCount O.mq P v ω : ℝ))
        (C.erase 1) k, ∑ p ∈ P, O.flip v p}
      ≤ (A.card : ℝ) * Real.exp (-(2 * u ^ 2 / (P.card : ℝ))) / ((A.card : ℝ) - k + 1)
        + (∑ v ∈ (C.erase 1).filter (fun v => Φ₀ ≤ ∑ p ∈ P, O.flip v p),
            (∑ p ∈ P, O.flip v p) * Real.exp (-(2 * (O.gap * (1 - 2 * O.η)
              * ∑ p ∈ P, O.flip v p - u) ^ 2 / ((P.card : ℝ) + 4 * ∑ p ∈ P, O.flip v p)))) / T := by
  classical
  set Φ : S → ℝ := fun v => ∑ p ∈ P, O.flip v p with hΦ
  set γ : ℝ := O.gap * (1 - 2 * O.η) with hγ
  set D : S → Ω → ℝ := fun v ω => (screenCount O.mq P v ω : ℝ) with hD
  set Bs : Ω → ℝ := seedBase O P with hBs
  set H : Finset S := (C.erase 1).filter (fun v => Φ₀ ≤ Φ v) with hH
  have hΦ0 : ∀ v, 0 ≤ Φ v := fun v => Finset.sum_nonneg (fun p _ => (flip_icc O v p).1)
  have hkC : k ≤ (C.erase 1).card := le_trans hkA (Finset.card_le_card hA)
  -- the events
  set Up : S → Set Ω := fun w => {ω | u ≤ D w ω - Bs ω} with hUp
  set Lo : S → Set Ω := fun v => {ω | D v ω - Bs ω < u} with hLo
  have hmUp : ∀ w, MeasurableSet (Up w) := fun w =>
    measurableSet_le measurable_const ((measurable_screenCount_real O P w).sub
      (measurable_seedBase O P))
  have hmLo : ∀ v, MeasurableSet (Lo v) := fun v =>
    measurableSet_lt ((measurable_screenCount_real O P v).sub (measurable_seedBase O P))
      measurable_const
  -- the deterministic step
  have hsub : {ω | (k : ℝ) * Φ₀ + T < ∑ v ∈ leastLossSubset (fun v => D v ω) (C.erase 1) k, Φ v}
      ⊆ {ω | (A.card : ℝ) - k + 1 ≤ ∑ w ∈ A, (1 : ℝ) * (Up w).indicator 1 ω}
        ∪ {ω | T ≤ ∑ v ∈ H, Φ v * (Lo v).indicator 1 ω} := by
    intro ω hω
    by_contra hnot
    simp only [Set.mem_union, Set.mem_setOf_eq, not_or, not_le] at hnot hω
    obtain ⟨h1, h2⟩ := hnot
    set A' : Finset S := A.filter (fun w => D w ω < Bs ω + u) with hA'
    have hneg : ((A.filter (fun w => ¬ (D w ω < Bs ω + u))).card : ℝ)
        = ∑ w ∈ A, (1 : ℝ) * (Up w).indicator 1 ω := by
      rw [Finset.card_filter, Nat.cast_sum]
      refine Finset.sum_congr rfl (fun w _ => ?_)
      by_cases h : D w ω < Bs ω + u
      · have : ω ∉ Up w := fun hh => by
          simp only [hUp, Set.mem_setOf_eq] at hh; linarith
        simp [h, Set.indicator_of_notMem this]
      · have : ω ∈ Up w := by
          simp only [hUp, Set.mem_setOf_eq]; linarith [not_lt.1 h]
        simp [h, Set.indicator_of_mem this]
    have hsplit := Finset.card_filter_add_card_filter_not (s := A) (fun w => D w ω < Bs ω + u)
    have hA'k : k ≤ A'.card := by
      have hR : ((A'.card : ℕ) : ℝ) + ((A.filter (fun w => ¬ (D w ω < Bs ω + u))).card : ℝ)
          = (A.card : ℝ) := by exact_mod_cast hsplit
      have : (k : ℝ) < (A'.card : ℝ) + 1 := by linarith
      exact Nat.lt_succ_iff.1 (by exact_mod_cast this)
    have hbelow := leastLossSubset_lt (ℓ := fun v => D v ω) hkC
      ((Finset.filter_subset _ _).trans hA) hA'k (fun w hw => (Finset.mem_filter.1 hw).2)
    set F : Finset S := leastLossSubset (fun v => D v ω) (C.erase 1) k with hF
    have hFsub : F ⊆ C.erase 1 := leastLossSubset_subset _ _ _ hkC
    have hFcard : F.card = k := leastLossSubset_card _ _ _ hkC
    have hlight : ∑ v ∈ F.filter (fun v => ¬ Φ₀ ≤ Φ v), Φ v ≤ (k : ℝ) * Φ₀ := by
      calc ∑ v ∈ F.filter (fun v => ¬ Φ₀ ≤ Φ v), Φ v
          ≤ ∑ v ∈ F.filter (fun v => ¬ Φ₀ ≤ Φ v), Φ₀ :=
            Finset.sum_le_sum (fun v hv => le_of_lt (not_le.1 (Finset.mem_filter.1 hv).2))
        _ ≤ ∑ v ∈ F, Φ₀ :=
            Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) (fun _ _ _ => hΦ₀)
        _ = (k : ℝ) * Φ₀ := by rw [Finset.sum_const, hFcard, nsmul_eq_mul]
    have hheavyF : ∑ v ∈ F.filter (fun v => Φ₀ ≤ Φ v), Φ v
        ≤ ∑ v ∈ H, Φ v * (Lo v).indicator 1 ω := by
      have e : ∑ v ∈ F.filter (fun v => Φ₀ ≤ Φ v), Φ v
          = ∑ v ∈ F.filter (fun v => Φ₀ ≤ Φ v), Φ v * (Lo v).indicator 1 ω := by
        refine Finset.sum_congr rfl (fun v hv => ?_)
        have hvLo : ω ∈ Lo v := by
          simp only [hLo, Set.mem_setOf_eq]
          have := hbelow v (Finset.mem_filter.1 hv).1
          linarith
        rw [Set.indicator_of_mem hvLo, Pi.one_apply, mul_one]
      rw [e]
      refine Finset.sum_le_sum_of_subset_of_nonneg ?_ (fun v _ _ => mul_nonneg (hΦ0 v)
        (Set.indicator_nonneg (fun _ _ => zero_le_one) ω))
      intro v hv
      obtain ⟨hvF, hvh⟩ := Finset.mem_filter.1 hv
      exact Finset.mem_filter.2 ⟨hFsub hvF, hvh⟩
    have htot := Finset.sum_filter_add_sum_filter_not F (fun v => Φ₀ ≤ Φ v) Φ
    linarith
  have hD1 : (A.card : ℝ) - k + 1 > 0 := by
    have : (k : ℝ) ≤ A.card := by exact_mod_cast hkA
    linarith
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) ?_
  have hm1 := markov_weighted (μ := μ) A (fun _ => (1 : ℝ)) (fun _ _ => zero_le_one) Up
    (fun w _ => hmUp w) hD1
  have hm2 := markov_weighted (μ := μ) H Φ (fun v _ => hΦ0 v) Lo (fun v _ => hmLo v) hT
  -- the clean rows' upper tails
  have hup : ∀ w ∈ A, μ.real (Up w) ≤ Real.exp (-(2 * u ^ 2 / (P.card : ℝ))) := by
    intro w hw
    have hwC := hA hw
    have hw1 := Finset.ne_of_mem_erase hwC
    have hwS := hV w (Finset.mem_of_mem_erase hwC)
    have hcl : ∑ p ∈ P, O.flip w p = 0 := Finset.sum_eq_zero (hAclean w hw)
    have htail := (rowTerm_sum_tail hflat O hP w hwS hw1 u hu).1
    rw [hcl, mul_zero, add_zero] at htail
    refine le_trans (measureReal_le_of_ae_imp ?_) htail
    filter_upwards [screenCount_split O P w] with ω hω hU
    simp only [hUp, Set.mem_setOf_eq] at hU ⊢
    rw [hD, hBs] at hU
    simp only at hU
    rw [hω, flipShift_sum_clean O P w (hAclean w hw), zero_add] at hU
    exact hU
  -- the heavy rows' lower tails
  have hlo : ∀ v ∈ H, μ.real (Lo v) ≤ Real.exp (-(2 * (γ * Φ v - u) ^ 2
      / ((P.card : ℝ) + 4 * Φ v))) := by
    intro v hv
    obtain ⟨hvC, hvh⟩ := Finset.mem_filter.1 hv
    have hv1 := Finset.ne_of_mem_erase hvC
    have hvS := hV v (Finset.mem_of_mem_erase hvC)
    have hge := hheavy v hvC hvh
    have htail := (rowTerm_sum_tail hflat O hP v hvS hv1 (γ * Φ v - u) (by linarith)).2
    refine le_trans (measureReal_le_of_ae_imp ?_) htail
    filter_upwards [screenCount_split O P v] with ω hω hL
    simp only [hLo, Set.mem_setOf_eq] at hL ⊢
    rw [hD, hBs] at hL
    simp only at hL
    rw [hω] at hL
    have := flipShift_sum_ge O P v hsig
    linarith
  have hs1 : ∑ w ∈ A, (1 : ℝ) * μ.real (Up w)
      ≤ (A.card : ℝ) * Real.exp (-(2 * u ^ 2 / (P.card : ℝ))) := by
    calc ∑ w ∈ A, (1 : ℝ) * μ.real (Up w)
        ≤ ∑ _w ∈ A, Real.exp (-(2 * u ^ 2 / (P.card : ℝ))) :=
          Finset.sum_le_sum (fun w hw => by rw [one_mul]; exact hup w hw)
      _ = (A.card : ℝ) * Real.exp (-(2 * u ^ 2 / (P.card : ℝ))) := by
          rw [Finset.sum_const, nsmul_eq_mul]
  have hs2 : ∑ v ∈ H, Φ v * μ.real (Lo v)
      ≤ ∑ v ∈ H, Φ v * Real.exp (-(2 * (γ * Φ v - u) ^ 2 / ((P.card : ℝ) + 4 * Φ v))) :=
    Finset.sum_le_sum (fun v hv => mul_le_mul_of_nonneg_left (hlo v hv) (hΦ0 v))
  have h1 : (∑ w ∈ A, (1 : ℝ) * μ.real (Up w)) / ((A.card : ℝ) - k + 1)
      ≤ (A.card : ℝ) * Real.exp (-(2 * u ^ 2 / (P.card : ℝ))) / ((A.card : ℝ) - k + 1) :=
    div_le_div_of_nonneg_right hs1 hD1.le
  have h2 : (∑ v ∈ H, Φ v * μ.real (Lo v)) / T
      ≤ (∑ v ∈ H, Φ v * Real.exp (-(2 * (γ * Φ v - u) ^ 2 / ((P.card : ℝ) + 4 * Φ v)))) / T :=
    div_le_div_of_nonneg_right hs2 hT.le
  linarith

end Select

section Heavy

/-- A heavy row's charge, `Φ·exp(−2(γΦ − u)²/(n + 4Φ))`, is largest at the least heavy mass
`Φ₀`, once `γ²Φ₀ ≥ 2`, `4Φ₀ ≤ n` and `n ≤ γΦ₀(γΦ₀ − u)`. -/
lemma heavy_term_le {γ u n Φ₀ Φ : ℝ} (hu : 0 ≤ u) (hΦ₀ : 0 < Φ₀) (hΦ : Φ₀ ≤ Φ)
    (h1 : 2 ≤ Φ₀ * γ ^ 2) (h2 : 4 * Φ₀ ≤ n) (ha₀ : u ≤ γ * Φ₀)
    (h3 : n ≤ Φ₀ * γ * (γ * Φ₀ - u)) :
    Φ * Real.exp (-(2 * (γ * Φ - u) ^ 2 / (n + 4 * Φ)))
      ≤ Φ₀ * Real.exp (-(2 * (γ * Φ₀ - u) ^ 2 / (n + 4 * Φ₀))) := by
  set d : ℝ := Φ - Φ₀ with hd
  have hd0 : 0 ≤ d := by linarith
  set a₀ : ℝ := γ * Φ₀ - u with ha₀def
  have ha₀0 : 0 ≤ a₀ := by linarith
  have hn0 : 0 < n := by linarith
  set N₀ : ℝ := n + 4 * Φ₀ with hN₀
  have hN₀pos : 0 < N₀ := by linarith
  have hNpos : 0 < n + 4 * Φ := by linarith
  have hγ0 : 0 < γ := by
    by_contra h
    push_neg at h
    have : γ * Φ₀ ≤ 0 := mul_nonpos_of_nonpos_of_nonneg h hΦ₀.le
    have : Φ₀ * γ * (γ * Φ₀ - u) ≤ 0 := by nlinarith
    linarith
  -- the exponent grows by at least `d/Φ₀`
  have hgrow : d / Φ₀ ≤ 2 * (γ * Φ - u) ^ 2 / (n + 4 * Φ) - 2 * a₀ ^ 2 / N₀ := by
    have hΦeq : n + 4 * Φ = N₀ + 4 * d := by rw [hN₀, hd]; ring
    have haeq : γ * Φ - u = a₀ + γ * d := by rw [ha₀def, hd]; ring
    rw [hΦeq, haeq, div_sub_div _ _ (by linarith) hN₀pos.ne', div_le_div_iff₀ hΦ₀ (by positivity)]
    -- `(N₀ + 4d) N₀ ≤ 2Φ₀ (2a₀γN₀ + γ²dN₀ − 4a₀²)`, times `d`
    have hc : N₀ ^ 2 ≤ 2 * Φ₀ * (2 * a₀ * γ * N₀ - 4 * a₀ ^ 2) := by
      have hA : 4 * a₀ ^ 2 ≤ 4 * a₀ * (γ * Φ₀) := by nlinarith
      have hB : N₀ ≤ 2 * n := by linarith
      have hC : N₀ ^ 2 ≤ 4 * n ^ 2 := by nlinarith
      have hD : 4 * n ^ 2 ≤ 4 * n * (Φ₀ * γ * a₀) := by nlinarith
      nlinarith
    have hdk : 4 * d * N₀ ≤ 2 * Φ₀ * (γ ^ 2 * d * N₀) := by
      have := mul_le_mul_of_nonneg_right h1 (by positivity : (0 : ℝ) ≤ 2 * d * N₀)
      nlinarith
    have key : (N₀ + 4 * d) * N₀ ≤ 2 * Φ₀ * (2 * a₀ * γ * N₀ + γ ^ 2 * d * N₀ - 4 * a₀ ^ 2) := by
      nlinarith
    have expand : (2 * (a₀ + γ * d) ^ 2 * N₀ - (N₀ + 4 * d) * (2 * a₀ ^ 2)) * Φ₀
        = d * (2 * Φ₀ * (2 * a₀ * γ * N₀ + γ ^ 2 * d * N₀ - 4 * a₀ ^ 2)) := by ring
    rw [expand]
    have := mul_le_mul_of_nonneg_left key hd0
    nlinarith
  have hratio : Φ ≤ Φ₀ * Real.exp (d / Φ₀) := by
    have := Real.add_one_le_exp (d / Φ₀)
    have e : Φ = Φ₀ * (d / Φ₀ + 1) := by rw [hd]; field_simp; ring
    rw [e]
    exact mul_le_mul_of_nonneg_left this hΦ₀.le
  calc Φ * Real.exp (-(2 * (γ * Φ - u) ^ 2 / (n + 4 * Φ)))
      ≤ Φ₀ * Real.exp (d / Φ₀) * Real.exp (-(2 * (γ * Φ - u) ^ 2 / (n + 4 * Φ))) :=
        mul_le_mul_of_nonneg_right hratio (Real.exp_pos _).le
    _ = Φ₀ * Real.exp (d / Φ₀ - 2 * (γ * Φ - u) ^ 2 / (n + 4 * Φ)) := by
        rw [mul_assoc, ← Real.exp_add]; ring_nf
    _ ≤ Φ₀ * Real.exp (-(2 * a₀ ^ 2 / N₀)) :=
        mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 (by linarith)) hΦ₀.le

end Heavy

section Table

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

/-- `countRule_flips` at the schedule's levels.  On a table of `n` prefixes, `m ≤ n ≤ Jm`, with
`a ≥ κ` of the pool's rows clean, the `κ` rows nearest the seed flip at most `κ·Δ·m` of it but
for chance

    a e^{−2u²}/(a − κ + 1) + (#C/κ)·(θ/(1 − θ))·e^{−(20/11)x²},

once the flip signal of a heavy row, `γθΔm` with `γ = gap(1 − 2η)`, clears `(u + x)√(Jm)`.
The heavy rows are those flipping `θΔm` or more, each at its own tail; the clean rows sit within
`u√(Jm)` of the seed's prediction. -/
theorem countRule_table {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (hsig : O.η ≤ 1 / 2) {P C A : Finset S} (hP : ∀ p ∈ P, p ∈ Pre)
    (hV : ∀ v ∈ C, v ∈ insert 1 Suf) (hA : A ⊆ C.erase 1)
    (hAclean : ∀ w ∈ A, ∀ p ∈ P, O.flip w p = 0) (κ : ℕ) (hκ : 0 < κ) (a : ℝ)
    (haA : a ≤ A.card) (hκa : (κ : ℝ) ≤ a)
    (m J : ℝ) (hm : 0 < m) (hJ : 1 ≤ J) (hmP : m ≤ P.card) (hPJ : (P.card : ℝ) ≤ J * m)
    (Δ θ u x : ℝ) (hΔ0 : 0 < Δ) (hΔ : 40 * Δ ≤ 1) (hθ0 : 0 < θ) (hθ1 : θ < 1) (hu : 0 ≤ u)
    (hx : 1 ≤ x)
    (hsignal : (u + x) * Real.sqrt (J * m) ≤ O.gap * (1 - 2 * O.η) * (θ * Δ * m))
    (hsq : 2 ≤ θ * Δ * m * (O.gap * (1 - 2 * O.η)) ^ 2) :
    μ.real {ω | (κ : ℝ) * (Δ * m) < ∑ v ∈ leastLossSubset (fun v => (screenCount O.mq P v ω : ℝ))
        (C.erase 1) κ, ∑ p ∈ P, O.flip v p}
      ≤ a * Real.exp (-(2 * u ^ 2)) / (a - κ + 1)
        + (C.card : ℝ) * θ / ((1 - θ) * κ) * Real.exp (-(20 / 11 * x ^ 2)) := by
  classical
  set γ : ℝ := O.gap * (1 - 2 * O.η) with hγ
  set n : ℝ := (P.card : ℝ) with hn
  set R : ℝ := Real.sqrt (J * m) with hR
  have hR0 : 0 < R := Real.sqrt_pos.2 (by positivity)
  have hRsq : R ^ 2 = J * m := Real.sq_sqrt (by positivity)
  have hn0 : 0 < n := lt_of_lt_of_le hm hmP
  set Φ₀ : ℝ := θ * Δ * m with hΦ₀
  have hΦ₀0 : 0 < Φ₀ := by positivity
  have hκR : (0 : ℝ) < κ := by exact_mod_cast hκ
  have hkA : κ ≤ A.card := by exact_mod_cast le_trans hκa haA
  have hγΦ : (u + x) * R ≤ γ * Φ₀ := hsignal
  have hγ0 : 0 < γ := by
    by_contra h
    push_neg at h
    have : γ * Φ₀ ≤ 0 := mul_nonpos_of_nonpos_of_nonneg h hΦ₀0.le
    nlinarith [mul_pos (by linarith : (0 : ℝ) < u + x) hR0]
  -- `countRule_flips` with `u√(Jm)` and the heavy rows at `Φ₀`
  have hT : 0 < (1 - θ) * κ * (Δ * m) := by
    have : 0 < 1 - θ := by linarith
    positivity
  have hmain := countRule_flips (μ := μ) hflat O hsig hP hV hA hAclean κ hkA (u * R) Φ₀
    ((1 - θ) * κ * (Δ * m)) (by positivity) hΦ₀0.le hT
    (fun v _ hv => by nlinarith [mul_le_mul_of_nonneg_left hv hγ0.le])
  have hset : {ω | (κ : ℝ) * (Δ * m) < ∑ v ∈ leastLossSubset (fun v => (screenCount O.mq P v ω : ℝ))
        (C.erase 1) κ, ∑ p ∈ P, O.flip v p}
      ⊆ {ω | (κ : ℝ) * Φ₀ + (1 - θ) * κ * (Δ * m) < ∑ v ∈ leastLossSubset
        (fun v => (screenCount O.mq P v ω : ℝ)) (C.erase 1) κ, ∑ p ∈ P, O.flip v p} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    have : (κ : ℝ) * Φ₀ + (1 - θ) * κ * (Δ * m) = (κ : ℝ) * (Δ * m) := by rw [hΦ₀]; ring
    linarith
  refine le_trans (measureReal_mono hset (measure_ne_top _ _)) (le_trans hmain ?_)
  gcongr ?_ + ?_
  · -- the clean rows
    have hA1 : (0 : ℝ) < (A.card : ℝ) - κ + 1 := by
      have : (κ : ℝ) ≤ A.card := by exact_mod_cast hkA
      linarith
    have hexp : Real.exp (-(2 * (u * R) ^ 2 / n)) ≤ Real.exp (-(2 * u ^ 2)) := by
      refine Real.exp_le_exp.2 (neg_le_neg ?_)
      rw [le_div_iff₀ hn0, mul_pow, hRsq]
      have : n ≤ J * m := hPJ
      nlinarith [sq_nonneg u]
    -- `t ↦ t/(t − κ + 1)` falls on `t ≥ κ`
    have hmono : (A.card : ℝ) / ((A.card : ℝ) - κ + 1) ≤ a / (a - κ + 1) := by
      rw [div_le_div_iff₀ hA1 (by linarith)]
      have hk1 : (1 : ℝ) ≤ κ := by exact_mod_cast hκ
      nlinarith [mul_nonneg (sub_nonneg.2 haA) (sub_nonneg.2 hk1)]
    calc (A.card : ℝ) * Real.exp (-(2 * (u * R) ^ 2 / n)) / ((A.card : ℝ) - κ + 1)
        ≤ (A.card : ℝ) * Real.exp (-(2 * u ^ 2)) / ((A.card : ℝ) - κ + 1) := by
          gcongr
      _ = (A.card : ℝ) / ((A.card : ℝ) - κ + 1) * Real.exp (-(2 * u ^ 2)) := by ring
      _ ≤ a / (a - κ + 1) * Real.exp (-(2 * u ^ 2)) :=
          mul_le_mul_of_nonneg_right hmono (Real.exp_pos _).le
      _ = a * Real.exp (-(2 * u ^ 2)) / (a - κ + 1) := by ring
  · -- the heavy rows, each at the least heavy mass
    have hΦm : Φ₀ ≤ Δ * m := by
      rw [hΦ₀]
      have := mul_le_mul_of_nonneg_right hθ1.le (by positivity : (0 : ℝ) ≤ Δ * m)
      linarith
    have hΔm : 40 * (Δ * m) ≤ m := by nlinarith only [hΔ, hm]
    have h4 : 4 * Φ₀ ≤ n := by linarith only [hΔm, hΦm, hmP, hΦ₀0]
    have hxR : 0 ≤ x * R := by positivity
    have huxR : (u + x) * R = u * R + x * R := by ring
    have ha₀ : u * R ≤ γ * Φ₀ := by linarith only [hγΦ, huxR, hxR]
    have h1 : x * R ≤ γ * Φ₀ - u * R := by linarith only [hγΦ, huxR]
    have hx3 : n ≤ Φ₀ * γ * (γ * Φ₀ - u * R) := by
      have h2 : (u + x) * R ≤ Φ₀ * γ := by linarith only [hγΦ]
      have h3 : (u + x) * R * (x * R) ≤ Φ₀ * γ * (γ * Φ₀ - u * R) :=
        mul_le_mul h2 h1 hxR (by positivity)
      have hux : (1 : ℝ) ≤ (u + x) * x := by nlinarith only [hu, hx]
      have hJm : n ≤ J * m := hPJ
      have h5 : J * m ≤ (u + x) * x * (J * m) := by
        nlinarith only [hux, mul_pos (by linarith only [hJ] : (0 : ℝ) < J) hm]
      have e : (u + x) * R * (x * R) = (u + x) * x * R ^ 2 := by ring
      rw [e, hRsq] at h3
      linarith only [h3, h5, hJm]
    have hterm : ∀ v ∈ (C.erase 1).filter (fun v => Φ₀ ≤ ∑ p ∈ P, O.flip v p),
        (∑ p ∈ P, O.flip v p) * Real.exp (-(2 * (γ * ∑ p ∈ P, O.flip v p - u * R) ^ 2
          / (n + 4 * ∑ p ∈ P, O.flip v p)))
        ≤ Φ₀ * Real.exp (-(20 / 11 * x ^ 2)) := by
      intro v hv
      have hvΦ := (Finset.mem_filter.1 hv).2
      refine le_trans (heavy_term_le (by positivity) hΦ₀0 hvΦ (by linarith only [hsq]) h4 ha₀
        hx3) ?_
      refine mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 (neg_le_neg ?_)) hΦ₀0.le
      -- `γΦ₀ − u√(Jm) ≥ x√(Jm)` and `n + 4Φ₀ ≤ (11/10) Jm`
      have h2 : n + 4 * Φ₀ ≤ 11 / 10 * (J * m) := by
        have : m ≤ J * m := by nlinarith only [hJ, hm]
        linarith only [this, hPJ, hΔm, hΦm]
      rw [le_div_iff₀ (by linarith only [hn0, hΦ₀0])]
      have h3 : (x * R) ^ 2 ≤ (γ * Φ₀ - u * R) ^ 2 :=
        pow_le_pow_left₀ hxR h1 2
      rw [mul_pow, hRsq] at h3
      have hx2 : 0 ≤ x ^ 2 := sq_nonneg x
      nlinarith only [h3, h2, hx2, mul_le_mul_of_nonneg_left h2 hx2]
    have hsum : ∑ v ∈ (C.erase 1).filter (fun v => Φ₀ ≤ ∑ p ∈ P, O.flip v p),
        (∑ p ∈ P, O.flip v p) * Real.exp (-(2 * (γ * ∑ p ∈ P, O.flip v p - u * R) ^ 2
          / (n + 4 * ∑ p ∈ P, O.flip v p)))
        ≤ (C.card : ℝ) * (Φ₀ * Real.exp (-(20 / 11 * x ^ 2))) := by
      refine le_trans (Finset.sum_le_sum hterm) ?_
      rw [Finset.sum_const, nsmul_eq_mul]
      have : (((C.erase 1).filter (fun v => Φ₀ ≤ ∑ p ∈ P, O.flip v p)).card : ℝ) ≤ C.card := by
        exact_mod_cast le_trans (Finset.card_filter_le _ _) (Finset.card_erase_le)
      exact mul_le_mul_of_nonneg_right this (by positivity)
    rw [div_le_iff₀ hT]
    calc _ ≤ (C.card : ℝ) * (Φ₀ * Real.exp (-(20 / 11 * x ^ 2))) := hsum
      _ = (C.card : ℝ) * θ / ((1 - θ) * κ) * Real.exp (-(20 / 11 * x ^ 2))
            * ((1 - θ) * κ * (Δ * m)) := by
          have h1θ : (1 - θ) ≠ 0 := by linarith only [hθ1]
          have hκ0 : (κ : ℝ) ≠ 0 := hκR.ne'
          rw [hΦ₀]; field_simp

end Table

section Mass

open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]
variable {J : Type*} [Fintype J]

open scoped Classical in
/-- The prefix draws on which `v`'s first `m` flips fall short of their mean by more than
`√(2 m Φ L)`, `Φ` its flip mass. -/
noncomputable def massUnder (O : Oracle μ S) (D : J → Measure S) (j : J) (m : ℕ) (L : ℝ)
    (v : S) : Set (J → ℕ → S) :=
  {q | ∑ i : Fin m, O.flip v (q j i.val)
    < (m : ℝ) * flipMass O (D j) v - Real.sqrt (2 * ((m : ℝ) * flipMass O (D j) v) * L)}

lemma measurableSet_massUnder (O : Oracle μ S) (D : J → Measure S) (j : J) (m : ℕ) (L : ℝ)
    (v : S) : MeasurableSet (massUnder O D j m L v) :=
  measurableSet_lt (Finset.measurable_sum _ (fun i _ => (flip_meas O v).comp
    ((measurable_pi_apply i.val).comp (measurable_pi_apply j)))) measurable_const

lemma measure_massUnder_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (j : J) (m : ℕ) (L : ℝ) (hL : 0 ≤ L) (v : S) :
    (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) (massUnder O D j m L v)
      ≤ ENNReal.ofReal (Real.exp (-L)) := by
  classical
  rw [← runMeasure_prefix_apply (μ := μ) D Dsf _ (measurableSet_massUnder O D j m L v)]
  set φ : ℝ := flipMass O (D j) v with hφ
  have hφ0 : 0 ≤ φ := integral_nonneg (fun p => (flip_icc O v p).1)
  rcases (eq_or_lt_of_le (mul_nonneg (Nat.cast_nonneg m) hφ0) : (0 : ℝ) = (m : ℝ) * φ ∨ _)
    with h0 | hpos
  · have hempty : {x : Run Ω S J | prefixStreams x ∈ massUnder O D j m L v} = ∅ := by
      ext x
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, massUnder, not_lt]
      rw [← hφ, ← h0]
      simp only [mul_zero, zero_mul, Real.sqrt_zero, sub_zero]
      exact Finset.sum_nonneg (fun i _ => (flip_icc O v _).1)
    rw [hempty, measure_empty]
    exact bot_le
  have hcard : ((Finset.univ : Finset (Fin m)).card : ℝ) = (m : ℝ) := by simp
  have hmean : ∑ i ∈ (Finset.univ : Finset (Fin m)),
      (Measure.pi fun _ : Fin m => D j)[fun q : Fin m → S => O.flip v (q i)] = (m : ℝ) * φ := by
    rw [Finset.sum_congr rfl (fun i _ => pi_flip_mean (D j) O m v i), Finset.sum_const,
      nsmul_eq_mul, hcard]
  have htail := sumLower_rel (μ := Measure.pi fun _ : Fin m => D j)
    (fun (i : Fin m) (q : Fin m → S) => O.flip v (q i)) (Finset.univ : Finset (Fin m))
    ((m : ℝ) * φ) (Real.sqrt (2 * ((m : ℝ) * φ) * L))
    (fun i => (flip_meas O v).comp (measurable_pi_apply i))
    (iIndepFun_pi (fun _ => (flip_meas O v).aemeasurable))
    (fun i => Filter.Eventually.of_forall (fun q => flip_icc O v (q i)))
    hpos (le_of_eq hmean.symm) (Real.sqrt_nonneg _)
  have hexp : (Real.sqrt (2 * ((m : ℝ) * φ) * L)) ^ 2 / (2 * ((m : ℝ) * φ)) = L := by
    have hne : (m : ℝ) * φ ≠ 0 := hpos.ne'
    rw [Real.sq_sqrt (by positivity)]
    rw [div_eq_iff (by positivity)]
    ring
  rw [hexp] at htail
  have hsub : {x : Run Ω S J | prefixStreams x ∈ massUnder O D j m L v}
      ⊆ (fun x : Run Ω S J => (fun i : Fin m => prefixDraw j i.val x)) ⁻¹'
        {q : Fin m → S | ∑ i : Fin m, O.flip v (q i)
          ≤ (m : ℝ) * φ - Real.sqrt (2 * ((m : ℝ) * φ) * L)} := by
    intro x hx
    simp only [Set.mem_setOf_eq, massUnder] at hx
    exact le_of_lt hx
  have hmeasPrf : Measurable (fun x : Run Ω S J => (fun i : Fin m => prefixDraw j i.val x)) :=
    measurable_pi_lambda _ (fun i : Fin m => measurable_prf j i.val)
  refine le_trans (measure_mono hsub) ?_
  rw [Measure.map_apply hmeasPrf (measurableSet_le (by fun_prop) measurable_const)
    |>.symm.trans (congrArg (fun ν : Measure (Fin m → S) => ν _) (map_prefixBlock D Dsf j m))]
  rw [← ENNReal.ofReal_toReal (measure_ne_top _ _), ← measureReal_def]
  exact ENNReal.ofReal_le_ofReal htail

/-- Every pool member's first `m` flips read its flip mass to within `√(2 m Φ L)`, off an
`M e^{−L}` set. -/
theorem pool_massUnder_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (j : J) (m M : ℕ) (L : ℝ) (hL : 0 ≤ L) :
    (runMeasure μ D Dsf).real {x : Run Ω S J | ¬ ∀ v ∈ poolAt M x,
        (m : ℝ) * flipMass O (D j) v - Real.sqrt (2 * ((m : ℝ) * flipMass O (D j) v) * L)
          ≤ ∑ i : Fin m, O.flip v (prefixDraw j i.val x)}
      ≤ (M : ℝ) * Real.exp (-L) := by
  classical
  have hseed : flipMass O (D j) (1 : S) = 0 := by
    have : ∀ p : S, O.flip (1 : S) p = 0 := by
      intro p
      show O.label (p * 1) + O.label p - 2 * O.label (p * 1) * O.label p = 0
      rw [mul_one]
      rcases O.label_bit p with hl | hl <;> rw [hl] <;> ring
    simp [flipMass, this]
  have hsub : {x : Run Ω S J | ¬ ∀ v ∈ poolAt M x,
        (m : ℝ) * flipMass O (D j) v - Real.sqrt (2 * ((m : ℝ) * flipMass O (D j) v) * L)
          ≤ ∑ i : Fin m, O.flip v (prefixDraw j i.val x)}
      ⊆ {x : Run Ω S J | ∃ v ∈ freshDraws ∅ (fun i => suffixDraw i x) M,
          prefixStreams x ∈ massUnder O D j m L v} := by
    intro x hx
    simp only [Set.mem_setOf_eq, not_forall, not_le] at hx
    obtain ⟨v, hv, hlt⟩ := hx
    rcases Finset.mem_insert.1 hv with rfl | hv'
    · rw [hseed] at hlt
      simp only [mul_zero, zero_mul, Real.sqrt_zero, sub_zero] at hlt
      exact absurd hlt (not_lt.2 (Finset.sum_nonneg (fun i _ => (flip_icc O _ _).1)))
    · exact ⟨v, hv', hlt⟩
  have hbound := le_trans (measure_mono hsub) (runMeasure_pool_selection_le (μ := μ) D Dsf _
    (measurableSet_massUnder O D j m L) M _
    (fun v => measure_massUnder_le (μ := μ) D Dsf O j m L hL v))
  rw [measureReal_def]
  calc _ ≤ ((M : ℝ≥0∞) * ENNReal.ofReal (Real.exp (-L))).toReal :=
        ENNReal.toReal_mono (by simp [ENNReal.mul_eq_top]) hbound
    _ = (M : ℝ) * Real.exp (-L) := by
        rw [ENNReal.toReal_mul, ENNReal.toReal_ofReal (Real.exp_nonneg _)]
        simp

/-- Read back through the rows' own deviations: if `S − a√S ≤ C` and `C + a√Δ ≤ Δ`, then
`S ≤ Δ`. -/
lemma le_of_sub_sqrt_le {S C a Δ : ℝ} (hS : 0 ≤ S) (ha : 0 ≤ a) (hC : 0 ≤ C) (hΔ : 0 < Δ)
    (h1 : S - a * Real.sqrt S ≤ C) (h2 : C + a * Real.sqrt Δ ≤ Δ) : S ≤ Δ := by
  by_contra hlt
  push_neg at hlt
  have hΔ0 : 0 ≤ Δ := hΔ.le
  set r : ℝ := Real.sqrt S with hr
  set q : ℝ := Real.sqrt Δ with hq
  have hr2 : r ^ 2 = S := Real.sq_sqrt hS
  have hq2 : q ^ 2 = Δ := Real.sq_sqrt hΔ0
  have hq0 : 0 ≤ q := Real.sqrt_nonneg _
  have hrq : q < r := Real.sqrt_lt_sqrt hΔ0 hlt
  have hqa : a * q ≤ q ^ 2 := by rw [hq2]; linarith
  -- `q ≥ a` unless `q = 0`
  have hqpos : 0 < q := Real.sqrt_pos.2 hΔ
  have key : r ^ 2 - a * r > q ^ 2 - a * q := by
    have hqa' : a ≤ q := by nlinarith
    nlinarith
  rw [hr2, hq2] at key
  linarith

/-- `(∑ √xᵢ)² ≤ #s · ∑ xᵢ`. -/
lemma sum_sqrt_le {ι : Type*} (s : Finset ι) (f : ι → ℝ) (hf : ∀ i ∈ s, 0 ≤ f i) :
    ∑ i ∈ s, Real.sqrt (f i) ≤ Real.sqrt ((s.card : ℝ) * ∑ i ∈ s, f i) := by
  have h := sq_sum_le_card_mul_sum_sq (s := s) (f := fun i => Real.sqrt (f i))
  have e : ∑ i ∈ s, Real.sqrt (f i) ^ 2 = ∑ i ∈ s, f i :=
    Finset.sum_congr rfl (fun i hi => Real.sq_sqrt (hf i hi))
  rw [e] at h
  exact Real.le_sqrt_of_sq_le h

end Mass

section Run

open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]
variable {J : Type*} [Fintype J]

lemma measurable_screenCount (O : Oracle μ S) (P : Finset S) (v : S) :
    Measurable (fun ω => screenCount O.mq P v ω) :=
  measurable_to_countable' (fun n => noiseAlg_le O Set.univ _
    (measurableSet_screenCount_pred O P v (fun k => k = n)))

open scoped Classical in
/-- The noise on which the `κ` rows nearest the seed flip more than `b` of the table. -/
lemma measurableSet_nearFlip (O : Oracle μ S) (P C : Finset S) (κ : ℕ) (b : ℝ) :
    MeasurableSet {ω : Ω | b < ∑ v ∈ leastLossSubset (fun v => (screenCount O.mq P v ω : ℝ))
      (C.erase 1) κ, ∑ p ∈ P, O.flip v p} := by
  set C' : Finset S := C.erase 1 with hC'
  set Ψ : Ω → ({v // v ∈ C'} → ℕ) := fun ω v => screenCount O.mq P v.val ω with hΨ
  have hΨm : Measurable Ψ := measurable_pi_lambda _ (fun v => measurable_screenCount O P v.val)
  set ext : ({v // v ∈ C'} → ℕ) → S → ℝ := fun t v => if h : v ∈ C' then (t ⟨v, h⟩ : ℝ) else 0
    with hext
  have heq : {ω : Ω | b < ∑ v ∈ leastLossSubset (fun v => (screenCount O.mq P v ω : ℝ)) C' κ,
      ∑ p ∈ P, O.flip v p}
      = Ψ ⁻¹' {t | b < ∑ v ∈ leastLossSubset (ext t) C' κ, ∑ p ∈ P, O.flip v p} := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_preimage]
    rw [leastLossSubset_congr (ℓ := fun v => (screenCount O.mq P v ω : ℝ)) (ℓ' := ext (Ψ ω))
      (fun v hv => by simp [hext, hΨ, hv])]
  rw [heq]
  exact hΨm (Set.to_countable _).measurableSet


open scoped Classical in
/-- A screen whose cutoff is at least its resolution admits every candidate. -/
lemma screened_eq_of_le (mq : S → Ω → ℝ) {sc scd : ℕ} (h : scd ≤ sc) (P cands : Finset S)
    (ω : Ω) : screened mq sc scd P cands ω = cands := by
  refine Finset.filter_true_of_mem (fun v _ => ?_)
  have h1 : screenCount mq P v ω ≤ P.card := Finset.card_filter_le _ _
  calc scd * screenCount mq P v ω ≤ scd * P.card := Nat.mul_le_mul_left _ h1
    _ ≤ sc * P.card := Nat.mul_le_mul_right _ h
    _ ≤ scd * screenBase mq P cands ω + sc * P.card := Nat.le_add_left _ _

open scoped Classical in
lemma clusterBy_nearest_eq (O : Oracle μ S) (populations : Finset J) (B : State)
    (hscreen : B.scd ≤ B.sc) (x : Run Ω S J) :
    clusterBy nearestClusterer O.mq populations x B
      = leastLossSubset (fun v => (screenCount O.mq (prefixesAt populations B.npref x) v
          (oracleNoise x) : ℝ)) ((poolAt B.nsuff x).erase 1) (B.k - 1) := by
  have hscr : screenedAt O.mq populations B x = poolAt B.nsuff x :=
    screened_eq_of_le O.mq hscreen _ _ _
  unfold clusterBy
  rw [hscr]
  show (insert 1 (leastLossSubset (readLoss (fun w => O.mq w (oracleNoise x) = 1)
    (prefixesAt populations B.npref x)) ((poolAt B.nsuff x).erase 1) (B.k - 1))).erase 1 = _
  rw [← screenCount_eq_readLoss]
  refine Finset.erase_insert (fun h => ?_)
  exact Finset.notMem_erase 1 _ (leastLossSubset_subset' _ _ _ h)

set_option maxHeartbeats 1000000 in
/-- The count rule's family flips little on average at population `j`: but for the table or
the pool redrawing too often, the pool short of accept-preserving rows, the family's table
flips (`countRule_table`) and a member's first `m` flips understating its mass, its members'
flip mass averages at most `Δ + c + g`. -/
theorem measureReal_famHeavy_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (hsuppSf : Dsf Sufᶜ = 0) (j : J)
    (hj : j ∈ populations) (B : State) (hsig : O.η ≤ 1 / 2) (hscreen : B.scd ≤ B.sc)
    (hk2 : 2 ≤ B.k) (hkM : B.k ≤ B.nsuff) (hmpos : 0 < B.npref)
    (Δ c g θ u x L tap a pAP ρ ρsf : ℝ) (r rs : ℕ)
    (hΔ0 : 0 < Δ) (hΔ : 40 * Δ ≤ 1) (hc0 : 0 ≤ c) (hr : (r : ℝ) ≤ c * B.npref) (hg : 0 < g)
    (hθ0 : 0 < θ) (hθ1 : θ < 1) (hu : 0 ≤ u) (hx : 1 ≤ x) (hL : 0 ≤ L) (htap : 0 ≤ tap)
    (hsignal : (u + x) * Real.sqrt ((populations.card : ℝ) * B.npref)
      ≤ O.gap * (1 - 2 * O.η) * (θ * Δ * B.npref))
    (hsq : 2 ≤ θ * Δ * B.npref * (O.gap * (1 - 2 * O.η)) ^ 2)
    (ha : a ≤ (B.nsuff : ℝ) * (pAP - tap) - rs - 1) (hκa : ((B.k - 1 : ℕ) : ℝ) ≤ a)
    (hpAP0 : 0 ≤ pAP) (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hmass : 2 * L * (Δ + c + g) ≤ B.npref * g ^ 2)
    (hρD : collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf) :
    (runMeasure μ D Dsf).real {x : Run Ω S J | ¬ ∑ v ∈ clusterBy nearestClusterer O.mq populations x B,
        flipMass O (D j) v ≤ ((clusterBy nearestClusterer O.mq populations x B).card : ℝ)
          * (Δ + c + g)}
      ≤ ((B.npref + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1))
        + ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))
        + Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2)
        + (a * Real.exp (-(2 * u ^ 2)) / (a - ((B.k - 1 : ℕ) : ℝ) + 1)
          + ((B.nsuff : ℝ) + 1) * θ / ((1 - θ) * ((B.k - 1 : ℕ) : ℝ))
            * Real.exp (-(20 / 11 * x ^ 2)))
        + (B.nsuff : ℝ) * Real.exp (-L) := by
  classical
  set m : ℕ := B.npref with hmdef
  set M : ℕ := B.nsuff with hMdef
  set κ : ℕ := B.k - 1 with hκdef
  have hκpos : 0 < κ := by omega
  have hmR : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hmpos
  set Jc : ℝ := (populations.card : ℝ) with hJc
  have hJ1 : (1 : ℝ) ≤ Jc := by
    have : 1 ≤ populations.card := Finset.card_pos.2 ⟨j, hj⟩
    rw [hJc]; exact_mod_cast this
  set APs : Finset S → Finset S := fun C =>
    (C.erase 1).filter (fun v => ∀ p, O.label (p * v) = O.label p) with hAPs
  -- the table events, by their table and pool
  set R : Finset S → Finset S → Set (Run Ω S J) := fun P C =>
    if (m : ℝ) ≤ P.card ∧ (P.card : ℝ) ≤ Jc * m ∧ a ≤ ((APs C).card : ℝ) ∧ C.card ≤ M + 1 then
      oracleNoise ⁻¹' {ω | (κ : ℝ) * (Δ * m) < ∑ v ∈ leastLossSubset
        (fun v => (screenCount O.mq P v ω : ℝ)) (C.erase 1) κ, ∑ p ∈ P, O.flip v p}
    else ∅ with hR
  have hRm : ∀ P C, MeasurableSet (R P C) := by
    intro P C
    simp only [hR]
    split_ifs
    · exact measurable_nz (measurableSet_nearFlip O P C κ _)
    · exact MeasurableSet.empty
  set Ecr : Set (Run Ω S J) :=
    {x | x ∈ R (prefixesAt populations B.npref x) (poolAt B.nsuff x)} with hEcr
  set E1 : Set (Run Ω S J) := {x | ¬ tabGood j m r x} with hE1
  set E2 : Set (Run Ω S J) := {x | ¬ poolGood M rs x} with hE2
  set E3 : Set (Run Ω S J) := apShort O M pAP tap with hE3
  set E5 : Set (Run Ω S J) := {x | ¬ ∀ v ∈ poolAt M x,
    (m : ℝ) * flipMass O (D j) v - Real.sqrt (2 * ((m : ℝ) * flipMass O (D j) v) * L)
      ≤ ∑ i : Fin m, O.flip v (prefixDraw j i.val x)} with hE5
  set A₀ : ℝ := Real.sqrt (2 * L / m) with hA₀
  have hA₀0 : 0 ≤ A₀ := Real.sqrt_nonneg _
  have hsqrt : ∀ φ : ℝ, 0 ≤ φ →
      Real.sqrt (2 * ((m : ℝ) * φ) * L) = (m : ℝ) * (A₀ * Real.sqrt φ) := by
    intro φ hφ
    have e : (m : ℝ) * (A₀ * Real.sqrt φ) = Real.sqrt ((m : ℝ) ^ 2 * (2 * L / m * φ)) := by
      rw [Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq hmR.le, Real.sqrt_mul (by positivity), hA₀]
    rw [e]
    congr 1
    field_simp
  have hsub : {x : Run Ω S J | ¬ ∑ v ∈ clusterBy nearestClusterer O.mq populations x B,
        flipMass O (D j) v ≤ ((clusterBy nearestClusterer O.mq populations x B).card : ℝ)
          * (Δ + c + g)}
      ⊆ E1 ∪ E2 ∪ E3 ∪ Ecr ∪ E5 := by
    intro x hx
    by_contra hnot
    simp only [Set.mem_union, not_or] at hnot
    obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, hcr⟩, h5⟩ := hnot
    simp only [hE1, hE2, hE5, Set.mem_setOf_eq, not_not] at h1 h2 h5
    set P : Finset S := prefixesAt populations B.npref x with hP
    set C : Finset S := poolAt B.nsuff x with hC
    set ω : Ω := oracleNoise x with hω
    have hcl := clusterBy_nearest_eq O populations B hscreen x
    rw [← hP, ← hC, ← hω] at hcl
    simp only [Set.mem_setOf_eq] at hx
    rw [hcl] at hx
    set F : Finset S := leastLossSubset (fun v => (screenCount O.mq P v ω : ℝ)) (C.erase 1) κ
      with hF
    -- the pool
    have hCle : C.card ≤ M + 1 :=
      le_trans (Finset.card_insert_le _ _)
        (Nat.add_le_add_right (card_freshDraws_le ∅ (fun i => suffixDraw i x) B.nsuff) 1)
    have hAPc : a ≤ ((APs C).card : ℝ) := by
      have hfr := ap_fresh_ge O M rs x h2
      have hshort : (M : ℝ) * (pAP - tap) < (((Finset.univ : Finset (Fin M)).filter
          (fun i : Fin M => ∀ p, O.label (p * suffixDraw i.val x) = O.label p)).card : ℝ) :=
        not_le.1 h3
      have hsubAP : ((freshDraws ∅ (fun i => suffixDraw i x) M).filter
          (fun v => ∀ p, O.label (p * v) = O.label p)).erase 1 ⊆ APs C := by
        intro v hv
        obtain ⟨hv1, hv⟩ := Finset.mem_erase.1 hv
        obtain ⟨hvF, hap⟩ := Finset.mem_filter.1 hv
        exact Finset.mem_filter.2 ⟨Finset.mem_erase.2 ⟨hv1, Finset.mem_insert_of_mem hvF⟩, hap⟩
      have hc1 := Finset.card_le_card hsubAP
      have hc2 := Finset.pred_card_le_card_erase (s := (freshDraws ∅ (fun i => suffixDraw i x)
        M).filter (fun v => ∀ p, O.label (p * v) = O.label p)) (a := 1)
      have : (((freshDraws ∅ (fun i => suffixDraw i x) M).filter
          (fun v => ∀ p, O.label (p * v) = O.label p)).card : ℝ) - 1 ≤ ((APs C).card : ℝ) := by
        have h' : ((freshDraws ∅ (fun i => suffixDraw i x) M).filter
            (fun v => ∀ p, O.label (p * v) = O.label p)).card ≤ (APs C).card + 1 := by omega
        have : (((freshDraws ∅ (fun i => suffixDraw i x) M).filter
            (fun v => ∀ p, O.label (p * v) = O.label p)).card : ℝ) ≤ (APs C).card + 1 := by
          exact_mod_cast h'
        linarith
      linarith
    have hκC : κ ≤ (C.erase 1).card := by
      have h' : ((κ : ℕ) : ℝ) ≤ ((APs C).card : ℝ) := le_trans hκa hAPc
      have h'' : κ ≤ (APs C).card := by exact_mod_cast h'
      exact le_trans h'' (Finset.card_filter_le _ _)
    have hFcard : F.card = κ := leastLossSubset_card _ _ _ hκC
    have hFsub : F ⊆ C.erase 1 := leastLossSubset_subset _ _ _ hκC
    -- the table
    have hmP : (m : ℝ) ≤ P.card := by
      exact_mod_cast npref_le_card_of_tabGood populations j hj m r x h1
    have hPJ : (P.card : ℝ) ≤ Jc * m := by
      rw [hP, hJc]; exact_mod_cast card_prefixesAt_le populations m x
    -- off `Ecr`, the family flips little of the table
    have hcond : (m : ℝ) ≤ P.card ∧ (P.card : ℝ) ≤ Jc * m ∧ a ≤ ((APs C).card : ℝ)
        ∧ C.card ≤ M + 1 := ⟨hmP, hPJ, hAPc, hCle⟩
    have hΦF : ∑ v ∈ F, ∑ p ∈ P, O.flip v p ≤ (κ : ℝ) * (Δ * m) := by
      by_contra hlt
      apply hcr
      show x ∈ R P C
      simp only [hR, if_pos hcond, Set.mem_preimage, Set.mem_setOf_eq]
      exact not_le.1 hlt
    -- each member's mass, read through its first draws
    have hmem : ∀ v ∈ F, (m : ℝ) * flipMass O (D j) v - (m : ℝ) * (A₀ * Real.sqrt
        (flipMass O (D j) v)) ≤ ∑ p ∈ P, O.flip v p + r := by
      intro v hv
      have hvC : v ∈ C := Finset.mem_of_mem_erase (hFsub hv)
      have hφ0 : 0 ≤ flipMass O (D j) v := integral_nonneg (fun p => (flip_icc O v p).1)
      have hh := h5 v hvC
      rw [hsqrt _ hφ0] at hh
      obtain ⟨-, -, hsum⟩ := freshDraws_of_good ∅ (fun i => prefixDraw j i x) m r h1
      have hdraw := (hsum (fun p => O.flip v p) (fun p => (flip_icc O v p).1)
        (fun p => (flip_icc O v p).2)).1
      rw [Fin.sum_univ_eq_sum_range (fun i => O.flip v (prefixDraw j i x))] at hh
      have hle1 := sum_prefixesOf_le_prefixesAt O populations j hj m x v
      have hof : ∑ p ∈ prefixesOf j m x, O.flip v p
          = ∑ p ∈ freshDraws ∅ (fun i => prefixDraw j i x) m, O.flip v p := rfl
      linarith
    -- summed over the family, and read back
    set Sm : ℝ := ∑ v ∈ F, flipMass O (D j) v with hSm
    have hSm0 : 0 ≤ Sm := Finset.sum_nonneg (fun v _ => integral_nonneg
      (fun p => (flip_icc O v p).1))
    have hκR : (0 : ℝ) < κ := by exact_mod_cast hκpos
    have hroot : ∑ v ∈ F, Real.sqrt (flipMass O (D j) v) ≤ (κ : ℝ) * Real.sqrt (Sm / κ) := by
      have h := sum_sqrt_le F (fun v => flipMass O (D j) v)
        (fun v _ => integral_nonneg (fun p => (flip_icc O v p).1))
      rw [hFcard] at h
      refine le_trans h (le_of_eq ?_)
      rw [show (κ : ℝ) * Sm = (κ : ℝ) ^ 2 * (Sm / κ) by field_simp, Real.sqrt_mul (by positivity),
        Real.sqrt_sq hκR.le]
    have hsumm := Finset.sum_le_sum hmem
    rw [Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum,
      Finset.sum_add_distrib, Finset.sum_const, hFcard, nsmul_eq_mul] at hsumm
    set S₁ : ℝ := Sm / κ with hS₁
    have hS₁0 : 0 ≤ S₁ := by positivity
    have hstep : S₁ - A₀ * Real.sqrt S₁ ≤ Δ + c := by
      have hr1 := mul_le_mul_of_nonneg_left hroot (mul_nonneg hmR.le hA₀0)
      have h1' : (m : ℝ) * Sm - (m : ℝ) * (A₀ * ((κ : ℝ) * Real.sqrt S₁))
          ≤ (κ : ℝ) * (Δ * m) + κ * r := by
        have e : (m : ℝ) * A₀ * ((κ : ℝ) * Real.sqrt S₁)
            = (m : ℝ) * (A₀ * ((κ : ℝ) * Real.sqrt S₁)) := by ring
        rw [e] at hr1
        linarith only [hsumm, hr1, hΦF]
      have hSmeq : Sm = (κ : ℝ) * S₁ := by rw [hS₁]; field_simp
      rw [hSmeq] at h1'
      have hkr : (κ : ℝ) * r ≤ (κ : ℝ) * (c * m) := mul_le_mul_of_nonneg_left hr hκR.le
      have h2' : (m : ℝ) * (κ : ℝ) * (S₁ - A₀ * Real.sqrt S₁) ≤ (m : ℝ) * (κ : ℝ) * (Δ + c) := by
        have e1 : (m : ℝ) * (κ : ℝ) * (S₁ - A₀ * Real.sqrt S₁)
            = (m : ℝ) * ((κ : ℝ) * S₁) - (m : ℝ) * (A₀ * ((κ : ℝ) * Real.sqrt S₁)) := by ring
        have e2 : (m : ℝ) * (κ : ℝ) * (Δ + c) = (κ : ℝ) * (Δ * m) + (κ : ℝ) * (c * m) := by ring
        rw [e1, e2]
        linarith only [h1', hkr]
      exact le_of_mul_le_mul_left h2' (by positivity)
    have hfin : S₁ ≤ Δ + c + g := by
      refine le_of_sub_sqrt_le hS₁0 hA₀0 (by linarith) (by linarith) hstep ?_
      -- `A₀ √(Δ + c + g) ≤ g`, which `2L(Δ + c + g) ≤ m g²` is
      have hq : A₀ * Real.sqrt (Δ + c + g) = Real.sqrt (2 * L * (Δ + c + g) / m) := by
        rw [hA₀, ← Real.sqrt_mul (by positivity)]; congr 1; ring
      rw [hq]
      have : Real.sqrt (2 * L * (Δ + c + g) / m) ≤ g := by
        rw [Real.sqrt_le_left hg.le, div_le_iff₀ hmR]
        linarith only [hmass]
      linarith only [this]
    apply hx
    rw [hFcard]
    have hSmeq : Sm = (κ : ℝ) * S₁ := by rw [hS₁]; field_simp
    rw [hSmeq]
    exact mul_le_mul_of_nonneg_left hfin hκR.le
  -- the table event, sliced at the draws
  have hEcr : (runMeasure μ D Dsf).real Ecr
      ≤ a * Real.exp (-(2 * u ^ 2)) / (a - ((B.k - 1 : ℕ) : ℝ) + 1)
        + ((B.nsuff : ℝ) + 1) * θ / ((1 - θ) * ((B.k - 1 : ℕ) : ℝ))
          * Real.exp (-(20 / 11 * x ^ 2)) := by
    set E : ℝ := a * Real.exp (-(2 * u ^ 2)) / (a - ((B.k - 1 : ℕ) : ℝ) + 1)
        + ((B.nsuff : ℝ) + 1) * θ / ((1 - θ) * ((B.k - 1 : ℕ) : ℝ))
          * Real.exp (-(20 / 11 * x ^ 2)) with hEdef
    have hκR : (0 : ℝ) < κ := by exact_mod_cast hκpos
    have hE0 : 0 ≤ E := by
      have h1 : 0 ≤ a - ((B.k - 1 : ℕ) : ℝ) + 1 := by linarith
      have h2 : 0 < 1 - θ := by linarith
      have h3 : 0 ≤ a := le_trans (Nat.cast_nonneg _) hκa
      positivity
    have hEnn : runMeasure μ D Dsf Ecr ≤ ENNReal.ofReal E := by
      refine runMeasure_slice_le D Dsf _ (measurableSet_of_run_data populations B R hRm) _ ?_
      filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
        ae_sfx_mem_Suf D Dsf Suf hsuppSf] with d hd hdS
      set Pd : Finset S := populations.biUnion
        (fun j => freshDraws ∅ (fun i => d.1.2 j i) B.npref) with hPd
      set Cd : Finset S := insert 1 (freshDraws ∅ (fun i => d.1.1 i) B.nsuff) with hCd
      have hV : ∀ v ∈ Cd, v ∈ insert 1 Suf := pool_mem_Suf B.nsuff (fun i => d.1.1 i) hdS
      have hPre : ∀ q ∈ Pd, q ∈ Pre :=
        table_mem_Pre populations B.npref (fun j i => d.1.2 j i) hd
      change μ {ω : Ω | ((ω, d) : Run Ω S J) ∈ R Pd Cd} ≤ _
      by_cases hcond : (m : ℝ) ≤ Pd.card ∧ (Pd.card : ℝ) ≤ Jc * m ∧ a ≤ ((APs Cd).card : ℝ)
          ∧ Cd.card ≤ M + 1
      · have hset : {ω : Ω | ((ω, d) : Run Ω S J) ∈ R Pd Cd}
            = {ω | (κ : ℝ) * (Δ * m) < ∑ v ∈ leastLossSubset
              (fun v => (screenCount O.mq Pd v ω : ℝ)) (Cd.erase 1) κ, ∑ p ∈ Pd, O.flip v p} := by
          ext ω
          simp only [hR, if_pos hcond, Set.mem_preimage, Set.mem_setOf_eq]
          rfl
        rw [hset, ← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
        refine ENNReal.ofReal_le_ofReal ?_
        obtain ⟨hmP, hPJ, hAPc, hCle⟩ := hcond
        have htab := countRule_table (μ := μ) hflat O hsig hPre hV (A := APs Cd)
          (Finset.filter_subset _ _)
          (fun w hw p _ => by
            have hap := (Finset.mem_filter.1 hw).2 p
            show O.label (p * w) + O.label p - 2 * O.label (p * w) * O.label p = 0
            rw [hap]
            rcases O.label_bit p with hl | hl <;> rw [hl] <;> ring)
          κ hκpos a hAPc hκa m Jc hmR hJ1 hmP hPJ Δ θ u x hΔ0 hΔ hθ0 hθ1 hu hx hsignal hsq
        refine le_trans htab ?_
        rw [hEdef]
        have hC1 : (Cd.card : ℝ) ≤ (B.nsuff : ℝ) + 1 := by exact_mod_cast hCle
        have h1θ : 0 < (1 - θ) * (κ : ℝ) := mul_pos (by linarith) hκR
        have := mul_le_mul_of_nonneg_right
          (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hC1 hθ0.le) h1θ.le)
          (Real.exp_pos (-(20 / 11 * x ^ 2))).le
        linarith only [this]
      · have hset : {ω : Ω | ((ω, d) : Run Ω S J) ∈ R Pd Cd} = ∅ := by
          ext ω
          simp only [hR, if_neg hcond, Set.mem_empty_iff_false, Set.mem_setOf_eq]
        rw [hset, measure_empty]
        exact bot_le
    rw [measureReal_def]
    calc (runMeasure μ D Dsf Ecr).toReal ≤ (ENNReal.ofReal E).toReal :=
          ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
      _ = E := ENNReal.toReal_ofReal hE0
  have b1 := measureReal_tabBad_le (μ := μ) D Dsf j m r ρ hρD hρ0
  have b2 := measureReal_poolBad_le (μ := μ) D Dsf M rs ρsf hρsf hρsf0
  have b3 := measureReal_apShort_le (μ := μ) D Dsf O M pAP tap hpAP0 htap hpAPBound
  have b5 := pool_massUnder_le (μ := μ) D Dsf O j m M L hL
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  have u1 := measureReal_union_le (μ := runMeasure μ D Dsf) E1 E2
  have u2 := measureReal_union_le (μ := runMeasure μ D Dsf) (E1 ∪ E2) E3
  have u3 := measureReal_union_le (μ := runMeasure μ D Dsf) (E1 ∪ E2 ∪ E3) Ecr
  have u4 := measureReal_union_le (μ := runMeasure μ D Dsf) (E1 ∪ E2 ∪ E3 ∪ Ecr) E5
  linarith

end Run

end OrthoDFA
