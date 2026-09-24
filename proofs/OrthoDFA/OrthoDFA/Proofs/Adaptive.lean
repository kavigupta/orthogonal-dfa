import OrthoDFA.Proofs.Basics
import OrthoDFA.Proofs.Schedule
import OrthoDFA.Proofs.BinomTail
import OrthoDFA.Proofs.Grouping
import Mathlib.Probability.ProductMeasure
import Mathlib.Probability.Independence.InfinitePi

/-!
# The adaptive clustering loop: the proof

`ClusteringCorrect`, in `OrthoDFA.Proofs.Schedule`, is what this file proves.  Two parts, composed by `sound_and_terminating`:

* `validity_of_returned` — whatever is returned is valid, whenever it is returned;
* `loop_terminates` — the loop returns at some round;

each except w.p. `δ/2`.

They meet at `PassableAt`, the arithmetic a round has to satisfy for both of its tests to
pass, and `exists_passable` shows the computed schedule reaches such a state.  That
arithmetic is solved in order: the miscut budget `lcut` below the indecision limit, the flip
budget `Δ` below it at the flip fraction the vote absorbs, the screen's two margins below
`Δ(1−2η)²`, then the
prefix count large enough for every exponential — the share's own among them — then `α` at
the gate's tail, then the pool at `k/(pAP − t)`.

The union bound over `stoppable` is a finite sum: a rung carries `δ·npref/(8·N)` and the
ladder's counts halve, so no summable weight over all budgets is needed and no state has to
be encoded as a number.  `vote_mem_grid` is what lets the state be a `State` at all — a
threshold enters every event only through the count it cuts at.

The gate's null is a coin flip, so it only has to pass: validity is carried by the family
itself — `ret` asks it to be full-size, the screen keeps it clean, and the vote then cuts
right on all but a small fraction of the certification sample (`measureReal_validMiss_le`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

/-! ## The derived configuration

`build_pst` does not take the family size; it *computes* it:

```python
n, eps = population_size_and_evidence_margin(
    signal_strength=min_signal_strength, acceptable_fpr=0.01, acceptable_fnr=0.01)
config = SearchConfig(suffix_family_size=n, evidence_margin=eps, ...)
```

and `min_signal_strength` is `½ − η`, already carried by the `Oracle`.  So the family
size and the evidence margin are derived from the oracle's signal together with the
two acceptable rates — they are not free parameters. -/

lemma mq_meas (O : Oracle μ S) (p : S) : Measurable (O.mq p) := by
  show Measurable (fun ω => O.label p + (1 - 2 * O.label p) * O.noise p ω)
  exact measurable_const.add (measurable_const.mul (O.noise_meas p))

lemma mq_indep (O : Oracle μ S) : iIndepFun (fun p : S => O.mq p) μ :=
  O.noise_indep.comp (fun p x => O.label p + (1 - 2 * O.label p) * x)
    (fun _ => measurable_const.add (measurable_const.mul measurable_id))

lemma mq_bit (O : Oracle μ S) (p : S) : ∀ᵐ ω ∂μ, O.mq p ω = 0 ∨ O.mq p ω = 1 := by
  filter_upwards [O.noise_bit p] with ω hω
  rcases O.label_bit p with hl | hl <;> rcases hω with hn | hn <;>
    norm_num [Oracle.mq, hl, hn]

lemma mq_icc (O : Oracle μ S) (p : S) : ∀ᵐ ω ∂μ, O.mq p ω ∈ Set.Icc (0 : ℝ) 1 := by
  filter_upwards [mq_bit O p] with ω hω
  rcases hω with h | h <;> rw [h] <;> norm_num

/-- `E[O.mq p] = r + (1−2r)·ℓ(p)` at `p`'s own rate `r`. -/
lemma mq_mean (O : Oracle μ S) (p : S) :
    μ[O.mq p] = O.rate p + (1 - 2 * O.rate p) * O.label p := by
  have : μ[O.mq p] = ∫ ω, (O.label p + (1 - 2 * O.label p) * O.noise p ω) ∂μ := rfl
  rw [this, integral_add (integrable_const _) ((O.noise_int p).const_mul _), integral_const,
    integral_const_mul, O.noise_mean p]
  simp only [measureReal_def, measure_univ, ENNReal.toReal_one, smul_eq_mul, one_mul]
  ring

/-- The midpoint of the two clean class means, `(ηOut + (1 − ηIn))/2`. -/
noncomputable def Oracle.mid (O : Oracle μ S) : ℝ := (1 + O.ηOut - O.ηIn) / 2

/-- Half the distance between them, `((1 − ηIn) − ηOut)/2`. -/
noncomputable def Oracle.hgap (O : Oracle μ S) : ℝ := (1 - O.ηIn - O.ηOut) / 2

lemma mid_sub_hgap (O : Oracle μ S) : O.mid - O.hgap = O.ηOut := by
  unfold Oracle.mid Oracle.hgap; ring

lemma mid_add_hgap (O : Oracle μ S) : O.mid + O.hgap = 1 - O.ηIn := by
  unfold Oracle.mid Oracle.hgap; ring

lemma hgap_nonneg (O : Oracle μ S) (hη : O.η ≤ 1 / 2) : 0 ≤ O.hgap := by
  unfold Oracle.η at hη
  have hin : O.ηIn ≤ 1 / 2 := le_trans (le_max_left _ _) hη
  have hout : O.ηOut ≤ 1 / 2 := le_trans (le_max_right _ _) hη
  unfold Oracle.hgap
  linarith

/-- Each query sits exactly at its own class's mean: `mid − hgap` off the language and
`mid + hgap` on it. -/
lemma mq_mean_centred (O : Oracle μ S) (w : S) :
    μ[O.mq w] = (O.mid - O.hgap) + 2 * O.hgap * O.label w := by
  rw [mq_mean]
  unfold Oracle.rate Oracle.label Oracle.mid Oracle.hgap
  by_cases h : w ∈ O.L
  · rw [if_pos h, Set.indicator_of_mem h, Pi.one_apply]; ring
  · rw [if_neg h, Set.indicator_of_notMem h]; ring

lemma eta_nonneg (O : Oracle μ S) : 0 ≤ O.η :=
  le_trans (O.rate_nonneg (1 : S)) (O.rate_le_eta 1)

lemma mq_integrable (O : Oracle μ S) (w : S) : Integrable (O.mq w) μ :=
  MeasureTheory.Integrable.of_mem_Icc 0 1 (mq_meas O w).aemeasurable (mq_icc O w)

open scoped Classical in
/-- The gate's hit count *is* the sum of the column's reads: they are `0/1`. -/
lemma hits_eq_sum (O : Oracle μ S) (A : Finset S) :
    ∀ᵐ ω ∂μ, ((A.filter (fun p => O.mq p ω = 1)).card : ℝ) = ∑ p ∈ A, O.mq p ω := by
  filter_upwards [(ae_ball_iff A.countable_toSet).2 (fun p _ => mq_bit O p)] with ω hω
  rw [← Finset.sum_filter_add_sum_filter_not A (fun p => O.mq p ω = 1)]
  have h1 : ∑ p ∈ A.filter (fun p => O.mq p ω = 1), O.mq p ω
      = ((A.filter (fun p => O.mq p ω = 1)).card : ℝ) := by
    rw [Finset.sum_congr rfl (fun p hp => (Finset.mem_filter.mp hp).2), Finset.sum_const,
      nsmul_eq_mul, mul_one]
  have h0 : ∑ p ∈ A.filter (fun p => ¬ (O.mq p ω = 1)), O.mq p ω = 0 := by
    refine Finset.sum_eq_zero (fun p hp => ?_)
    obtain ⟨hpA, hne⟩ := Finset.mem_filter.mp hp
    rcases hω p hpA with h | h
    · exact h
    · exact absurd h hne
  rw [h1, h0, add_zero]

/-! ## The run space

`runMeasure` is a concrete space with a concrete measure, so the marginals the concentration
arguments consume (`map_firstDraws`, `map_drawBlock`) are lemmas rather than hypotheses.

On the deduplication gap `OrthoDFA.Clustering` records: do not close it by modelling
the without-replacement law as "i.i.d. conditioned on the block being injective".  Those
conditioned laws are inconsistent across `n` — for `Dsf = (½,¼,¼)` the first marginal of the
`n = 2` law puts mass `⅖` on the first atom, not `½` — so no space carries them all and
everything built on them would be vacuous. -/

open scoped Classical in
/-- Reading the first `n` coordinates of an i.i.d. stream. -/
lemma measurePreserving_finRestrict (D : Measure S) [IsProbabilityMeasure D] (n : ℕ) :
    MeasurePreserving (fun (p : ℕ → S) (i : Fin n) => p i.val)
      (Measure.infinitePi fun _ : ℕ => D) (Measure.pi fun _ : Fin n => D) where
  measurable := by fun_prop
  map_eq := by
    refine (Measure.pi_eq (μ := fun _ : Fin n => D) fun t ht => ?_).symm
    have hpre : (fun (p : ℕ → S) (i : Fin n) => p i.val) ⁻¹' Set.univ.pi t
        = Set.pi ↑(Finset.range n) (fun j => if h : j < n then t ⟨j, h⟩ else Set.univ) := by
      ext p
      simp only [Set.mem_preimage, Set.mem_pi, Set.mem_univ, forall_const, Finset.coe_range,
        Set.mem_Iio]
      refine ⟨fun h j hj => by rw [dif_pos hj]; exact h ⟨j, hj⟩, fun h i => ?_⟩
      simpa [dif_pos i.isLt] using h i.val i.isLt
    rw [Measure.map_apply (by fun_prop) (MeasurableSet.univ_pi ht), hpre,
      Measure.infinitePi_pi]
    · rw [← Fin.prod_univ_eq_prod_range]
      exact Finset.prod_congr rfl fun i _ => by simp [dif_pos i.isLt]
    · intro j _
      split_ifs with h
      exacts [ht ⟨j, h⟩, .univ]

variable {J : Type*} [Fintype J]
variable {η₀ : ℝ}
variable {indecisionLimit : ℝ}

/-! The whole blocks of a run, as opposed to `suffixDraw`/`prefixDraw`/`certPrefix`, which pick out one draw.
The marginal arguments slice the product at these boundaries. -/

/-- The draw block: the suffix stream paired with every population's prefix stream. -/
def draws (x : Run Ω S J) : (ℕ → S) × (J → ℕ → S) := x.2.1

/-- The prefix streams. -/
def prefixStreams (x : Run Ω S J) : J → ℕ → S := x.2.1.2

/-- The certification stream. -/
def certStream (x : Run Ω S J) : J → ℕ → S := x.2.2

@[fun_prop]
lemma measurable_draws : Measurable (draws : Run Ω S J → (ℕ → S) × (J → ℕ → S)) :=
  measurable_fst.comp measurable_snd

@[fun_prop]
lemma measurable_prfs : Measurable (prefixStreams : Run Ω S J → J → ℕ → S) :=
  measurable_snd.comp (measurable_fst.comp measurable_snd)

@[fun_prop]
lemma measurable_certs : Measurable (certStream : Run Ω S J → J → ℕ → S) :=
  measurable_snd.comp measurable_snd

lemma measurable_nz : Measurable (oracleNoise : Run Ω S J → Ω) := measurable_fst

lemma measurable_sfx (i : ℕ) : Measurable (suffixDraw (Ω := Ω) (S := S) (J := J) i) := by
  unfold suffixDraw; fun_prop

lemma measurable_prf (j : J) (i : ℕ) : Measurable (prefixDraw (Ω := Ω) (S := S) j i) := by
  unfold prefixDraw; fun_prop

lemma measurable_cert (j : J) (i : ℕ) : Measurable (certPrefix (Ω := Ω) (S := S) j i) := by
  unfold certPrefix; fun_prop

/-- The joint law of the first `n` draws.  This is what the concentration arguments
consume, and it is a theorem about `runMeasure`, not a hypothesis about an abstract space. -/
lemma map_firstDraws (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (n : ℕ) :
    Measure.map (fun x : Run Ω S J => ((fun i : Fin n => suffixDraw i.val x),
        (fun (j : J) (i : Fin n) => prefixDraw j i.val x))) (runMeasure μ D Dsf)
      = (Measure.pi fun _ : Fin n => Dsf).prod
          (Measure.pi (fun j : J => Measure.pi fun _ : Fin n => D j)) := by
  have h1 : (fun x : Run Ω S J => ((fun i : Fin n => suffixDraw i.val x),
      (fun (j : J) (i : Fin n) => prefixDraw j i.val x)))
      = (fun y : (ℕ → S) × (J → ℕ → S) => ((fun i : Fin n => y.1 i.val),
          (fun (j : J) (i : Fin n) => y.2 j i.val))) ∘ (draws : Run Ω S J → _) := rfl
  rw [h1, ← Measure.map_map (by fun_prop) (by fun_prop), runMeasure]
  rw [show (draws : Run Ω S J → _) = Prod.fst ∘ Prod.snd from rfl,
    ← Measure.map_map measurable_fst measurable_snd, Measure.map_snd_prod]
  simp only [measure_univ, one_smul]
  rw [Measure.map_fst_prod]
  simp only [measure_univ, one_smul]
  exact ((measurePreserving_finRestrict Dsf n).prod
    (measurePreserving_pi _ _ fun j => measurePreserving_finRestrict (D j) n)).map_eq

section Loop

variable {D : J → Measure S} {Dsf : Measure S}

/-! ## The loop, with the schedule left to the algorithm

The growth schedule is not a user parameter — it is something the algorithm optimizes as
it sees fit (`sample_suffix_family` alternates suffix- and prefix-growth based on whether
the FNR improved, and even the suffix increment is the *screened* `kept` count).  So the
statement fixes no schedule.  Instead a history records the sequence of budget states
the loop has passed through, and the guarantee is uniform over *all* histories and *all*
budgets — whatever the algorithm chooses, it is covered.  Histories are countable, so the
union bound still closes. -/

/-- The budget the loop is heading for: the top of the ladder. -/
noncomputable def solvedState (η₀ : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ α pAP : ℝ) : State :=
  solvedStateAt η₀ populations indecisionLimit εcov δ pAP
    (prefCount η₀ populations indecisionLimit εcov δ α pAP)

lemma ladderLen_pos (η₀ : ℝ) (populations : Finset J) (εcov δ α pAP : ℝ) :
    0 < ladderLen η₀ populations indecisionLimit εcov δ α pAP := by
  rw [ladderLen]; omega

lemma solvedState_mem_schedule (η₀ : ℝ) (populations : Finset J)
    (εcov δ α pAP : ℝ) :
    solvedState η₀ populations indecisionLimit εcov δ α pAP
      ∈ schedule η₀ populations indecisionLimit εcov δ α pAP := by
  rw [schedule, solvedState]
  refine Finset.mem_image.2 ⟨0, Finset.mem_range.2 (ladderLen_pos _ _ _ _ _ _), ?_⟩
  norm_num

/-- Every rung draws the same pool: only the prefix count halves along the ladder. -/
lemma nsuff_of_mem_schedule {η₀ : ℝ} {populations : Finset J} {εcov δ α pAP : ℝ} {B : State}
    (hB : B ∈ schedule η₀ populations indecisionLimit εcov δ α pAP) :
    B.nsuff = poolCount η₀ populations indecisionLimit εcov δ pAP := by
  rw [schedule] at hB
  obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hB
  rfl

lemma prefCount_pos (η₀ : ℝ) (populations : Finset J) (εcov δ α pAP : ℝ) :
    0 < prefCount η₀ populations indecisionLimit εcov δ α pAP := by
  rw [prefCount]; omega

/-- Collapsing an image can only drop terms, never add them. -/
lemma sum_image_le_sum {β : Type*} [DecidableEq β] (g : β → ℝ) (hg : ∀ b, 0 ≤ g b)
    (f : ℕ → β) (s : Finset ℕ) :
    ∑ b ∈ s.image f, g b ≤ ∑ i ∈ s, g (f i) := by
  classical
  refine Finset.induction_on s (by simp) ?_
  intro a t ha ih
  rw [Finset.image_insert, Finset.sum_insert ha]
  by_cases h : f a ∈ t.image f
  · rw [Finset.insert_eq_of_mem h]
    linarith [ih, hg (f a)]
  · rw [Finset.sum_insert h]
    linarith [ih]

/-- The ladder's prefix counts halve, so they sum to under twice the top rung's however long
the ladder is.  This is what lets the rungs be paid in proportion to their counts. -/
lemma schedule_npref_sum_le (η₀ : ℝ) (populations : Finset J) (εcov δ α pAP : ℝ) :
    ∑ B ∈ schedule η₀ populations indecisionLimit εcov δ α pAP, (B.npref : ℝ)
      ≤ 2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP : ℝ) := by
  classical
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP with hmdef
  set L : ℕ := ladderLen η₀ populations indecisionLimit εcov δ α pAP with hLdef
  calc ∑ B ∈ schedule η₀ populations indecisionLimit εcov δ α pAP, (B.npref : ℝ)
      ≤ ∑ i ∈ Finset.range L,
          ((solvedStateAt η₀ populations indecisionLimit εcov δ pAP (m / 2 ^ i)).npref : ℝ) := by
        rw [schedule, ← hmdef, ← hLdef]
        exact sum_image_le_sum _ (fun B => Nat.cast_nonneg _) _ _
    _ ≤ ∑ i ∈ Finset.range L, (m : ℝ) * (1 / 2) ^ i := by
        refine Finset.sum_le_sum (fun i _ => ?_)
        show ((m / 2 ^ i : ℕ) : ℝ) ≤ (m : ℝ) * (1 / 2) ^ i
        have h2 : ((2 ^ i : ℕ) : ℝ) = (2 : ℝ) ^ i := by push_cast; ring
        refine le_trans Nat.cast_div_le (le_of_eq ?_)
        rw [h2, div_pow, one_pow, mul_one_div]
    _ = (m : ℝ) * ∑ i ∈ Finset.range L, (1 / 2 : ℝ) ^ i := by rw [Finset.mul_sum]
    _ ≤ (m : ℝ) * 2 :=
        mul_le_mul_of_nonneg_left (sum_geometric_two_le _) (Nat.cast_nonneg _)
    _ = 2 * (m : ℝ) := by ring

lemma sig_pos (η₀ : ℝ) (hsig : η₀ < 1 / 2) : 0 < sig η₀ := by
  rw [sig]; linarith

lemma cutBudget_pos (η₀ : ℝ) {εcov : ℝ} (hsig : η₀ < 1 / 2) (hε : 0 < εcov)
    (hind : 0 < indecisionLimit) : 0 < cutBudget η₀ indecisionLimit εcov := by
  rw [cutBudget]
  exact lt_min (div_pos hε (by norm_num))
    (lt_min (div_pos (sig_pos η₀ hsig) (by norm_num)) (by linarith))

lemma cutBudget_le (η₀ indecisionLimit εcov : ℝ) :
    cutBudget η₀ indecisionLimit εcov ≤ εcov / 16
    ∧ cutBudget η₀ indecisionLimit εcov ≤ sig η₀ / 64
    ∧ cutBudget η₀ indecisionLimit εcov ≤ indecisionLimit / 2 :=
  ⟨min_le_left _ _, (min_le_right _ _).trans (min_le_left _ _),
    (min_le_right _ _).trans (min_le_right _ _)⟩

lemma voteSlack_pos (η₀ : ℝ) (hsig : η₀ < 1 / 2) : 0 < voteSlack η₀ := by
  rw [voteSlack]
  have := sig_pos η₀ hsig
  positivity

lemma flipFrac_pos (η₀ : ℝ) (hsig : η₀ < 1 / 2) : 0 < flipFrac η₀ := by
  rw [flipFrac]
  have := sig_pos η₀ hsig
  exact div_pos (by linarith) (by linarith)

lemma flipFrac_lt_one (η₀ : ℝ) (hsig : η₀ < 1 / 2) : flipFrac η₀ < 1 := by
  rw [flipFrac, sig, div_lt_one (by linarith)]
  linarith

/-- The split `flipFrac`/`voteSlack` was solved against. -/
lemma flipFrac_voteSlack (η₀ : ℝ) (hsig : η₀ < 1 / 2) :
    (1 - η₀) * flipFrac η₀ + voteSlack η₀ = sig η₀ := by
  have hne : (1 : ℝ) - η₀ ≠ 0 := by intro h; rw [sub_eq_zero] at h; linarith [h.symm]
  rw [flipFrac, voteSlack]
  field_simp
  ring

lemma famCount_pos (η₀ : ℝ) (populations : Finset J) (εcov δ : ℝ) :
    0 < famCount η₀ populations indecisionLimit εcov δ := by
  rw [famCount]; omega

/-- The family's size is even, so its centre is a count. -/
lemma famCount_half (η₀ : ℝ) (populations : Finset J) (εcov δ : ℝ) :
    (⌈(famCount η₀ populations indecisionLimit εcov δ : ℝ) / 2⌉₊ : ℝ)
      = (famCount η₀ populations indecisionLimit εcov δ : ℝ) / 2 := by
  rw [famCount]
  rw [show ((2 * (⌈Real.log (2 / cutBudget η₀ indecisionLimit εcov)
        / (4 * voteSlack η₀ ^ 2)⌉₊ + 1) : ℕ) : ℝ) / 2
      = ((⌈Real.log (2 / cutBudget η₀ indecisionLimit εcov)
        / (4 * voteSlack η₀ ^ 2)⌉₊ + 1 : ℕ) : ℝ) by
    push_cast; ring]
  rw [Nat.ceil_natCast]

/-- To put `exp (-a)` under `ε` it is enough that `a` clears `log (1/ε)`. -/
lemma exp_neg_le_of_log_le {a ε : ℝ} (hε : 0 < ε) (h : Real.log (1 / ε) ≤ a) :
    Real.exp (-a) ≤ ε := by
  have h1 : Real.exp (-a) ≤ Real.exp (-Real.log (1 / ε)) := Real.exp_le_exp.2 (by linarith)
  have h2 : Real.exp (-Real.log (1 / ε)) = ε := by
    rw [← Real.log_inv, one_div, inv_inv, Real.exp_log hε]
  linarith [h1, h2.le, h2.ge]

lemma famCount_tail (η₀ : ℝ) (populations : Finset J) {εcov δ : ℝ} (hsig : η₀ < 1 / 2)
    (hε : 0 < εcov) (hind : 0 < indecisionLimit) :
    Real.exp (-2 * (famCount η₀ populations indecisionLimit εcov δ : ℝ) * voteSlack η₀ ^ 2)
      ≤ cutBudget η₀ indecisionLimit εcov / 2 := by
  have hs := voteSlack_pos η₀ hsig
  have hcut := cutBudget_pos η₀ hsig hε hind
  have hhalf : (⌈Real.log (2 / cutBudget η₀ indecisionLimit εcov) / (4 * voteSlack η₀ ^ 2)⌉₊ : ℝ)
      ≤ (famCount η₀ populations indecisionLimit εcov δ : ℝ) / 2 := by
    rw [famCount]; push_cast; linarith
  have h4 := le_trans (Nat.le_ceil (Real.log (2 / cutBudget η₀ indecisionLimit εcov)
    / (4 * voteSlack η₀ ^ 2))) hhalf
  rw [div_le_iff₀ (by positivity : (0 : ℝ) < 4 * voteSlack η₀ ^ 2)] at h4
  rw [show -2 * (famCount η₀ populations indecisionLimit εcov δ : ℝ) * voteSlack η₀ ^ 2
      = -(2 * (famCount η₀ populations indecisionLimit εcov δ : ℝ) * voteSlack η₀ ^ 2) by ring]
  refine exp_neg_le_of_log_le (by positivity) ?_
  rw [one_div_div]
  linarith

lemma flipBudget_pos (η₀ : ℝ) (populations : Finset J) {εcov δ : ℝ}
    (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) :
    0 < flipBudget η₀ populations indecisionLimit εcov δ := by
  rw [flipBudget]
  exact div_pos (mul_pos (cutBudget_pos η₀ hsig hε hind) (flipFrac_pos η₀ hsig))
    (mul_pos (by norm_num) hcard)

lemma screenMargin_pos (η₀ : ℝ) (populations : Finset J) {εcov δ : ℝ}
    (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) :
    0 < screenMargin η₀ populations indecisionLimit εcov δ := by
  rw [screenMargin]
  have h1 := flipBudget_pos η₀ populations (δ := δ) hsig hε hind hcard
  have h2 := sig_pos η₀ hsig
  positivity

lemma one_mem_poolAt (M : ℕ) (x : Run Ω S J) : (1 : S) ∈ poolAt M x :=
  Finset.mem_insert_self _ _

/-- The family's vote on a prefix: the mean membership query over the family. -/
noncomputable def vote (O : Oracle μ S) (F : Finset S) (p : S) (ω : Ω) : ℝ :=
  (∑ v ∈ F, O.mq (p * v) ω) / F.card

open scoped Classical in
/-- Votes live on a grid.  Every membership query is `0` or `1`, so a family of `k`
suffixes votes in `{0, 1/k, …, 1}`.

This is what collapses the union over boundaries.  Every comparison the algorithm makes
against a real-valued threshold — the cluster centre's `cn/cd`, the gate's `lo` and `hi` —
comes down to which grid cell that threshold sits in, an integer in `{0, …, k+1}`, so
`State` can carry the counts instead. -/
lemma vote_mem_grid (O : Oracle μ S) (F : Finset S) (p : S) :
    ∀ᵐ ω ∂μ, ∃ j : ℕ, j ≤ F.card ∧ vote O F p ω = (j : ℝ) / F.card := by
  filter_upwards [(ae_ball_iff F.countable_toSet).2 (fun v _ => mq_bit O (p * v))] with ω hω
  refine ⟨(F.filter (fun v => O.mq (p * v) ω = 1)).card,
    Finset.card_le_card (Finset.filter_subset _ _), ?_⟩
  have hsum : ∑ v ∈ F, O.mq (p * v) ω
      = ((F.filter (fun v => O.mq (p * v) ω = 1)).card : ℝ) := by
    rw [← Finset.sum_filter_add_sum_filter_not F (fun v => O.mq (p * v) ω = 1)]
    have h1 : ∑ v ∈ F.filter (fun v => O.mq (p * v) ω = 1), O.mq (p * v) ω
        = ((F.filter (fun v => O.mq (p * v) ω = 1)).card : ℝ) := by
      rw [Finset.sum_congr rfl (fun v hv => (Finset.mem_filter.mp hv).2), Finset.sum_const,
        nsmul_eq_mul, mul_one]
    have h0 : ∑ v ∈ F.filter (fun v => ¬ (O.mq (p * v) ω = 1)), O.mq (p * v) ω = 0 := by
      refine Finset.sum_eq_zero (fun v hv => ?_)
      obtain ⟨hvF, hne⟩ := Finset.mem_filter.mp hv
      rcases hω v hvF with h | h
      · exact h
      · exact absurd h hne
    rw [h1, h0, add_zero]
  show (∑ v ∈ F, O.mq (p * v) ω) / F.card = _
  rw [hsum]

/-- The cluster never drifts off the seed.  `identify_cluster_around` stops the moment
`ε` would leave, so every family the loop proposes contains it — which is what lets the
gate read the split off `ε`'s own column. -/
lemma one_mem_clusterAround (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ) :
    (1 : S) ∈ clusterAround O.mq cn cd P cands ω k := by
  classical
  unfold clusterAround
  generalize k * P.card + 1 = n
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Function.iterate_succ_apply']
      unfold lloydStep
      split_ifs
      · exact Finset.mem_insert_self _ _
      · exact ih

lemma screened_subset (O : Oracle μ S) (sc scd : ℕ) (P cands : Finset S) (ω : Ω) :
    screened O.mq sc scd P cands ω ⊆ cands :=
  Finset.filter_subset _ _

lemma screenCount_one (O : Oracle μ S) (P : Finset S) (ω : Ω) : screenCount O.mq P 1 ω = 0 := by
  classical
  unfold screenCount
  rw [Finset.card_eq_zero]
  exact Finset.filter_eq_empty_iff.2 (fun p _ => by simp [mul_one])

open scoped Classical in
lemma one_mem_screened (O : Oracle μ S) (sc scd : ℕ) (P cands : Finset S) (ω : Ω)
    (hone : (1 : S) ∈ cands) : (1 : S) ∈ screened O.mq sc scd P cands ω := by
  classical
  refine Finset.mem_filter.2 ⟨hone, ?_⟩
  show scd * screenCount O.mq P 1 ω ≤ scd * screenBase O.mq P cands ω + sc * P.card
  rw [screenCount_one]
  simpa using Nat.zero_le _

lemma screenedAt_subset (O : Oracle μ S) (populations : Finset J) (B : State)
    (x : Run Ω S J) : screenedAt O.mq populations B x ⊆ poolAt B.nsuff x :=
  screened_subset _ _ _ _ _ _

open scoped Classical in
/-- The seed always survives: it is the reference, so its disagreement count is zero. -/
lemma one_mem_screenedAt (O : Oracle μ S) (populations : Finset J) (B : State)
    (x : Run Ω S J) : (1 : S) ∈ screenedAt O.mq populations B x :=
  one_mem_screened O B.sc B.scd _ _ _ (one_mem_poolAt B.nsuff x)

/-- The seed's column is read at a different string from the split.  The gate counts
`O.mq p`, the oracle at `p`; the split reads `p · v` for the family members `v`.  With `ε`
dropped from the family those strings are all distinct from `p`, so the persistent oracle's
bits at them are independent of the bit being scored.

This is what makes the cut usable in `agree_sound_of_wrong`.  That lemma needs the cut
fixed, but it is `ω`-dependent; it is determined by the reads at `p · v`, and those are
independent of the reads at `p`, so conditioning on the votes fixes the cut without
disturbing the law of the seed column.  Without dropping `ε` the cut would be partly
determined by the very bit the gate counts, and no conditioning would separate them. -/
lemma mul_ne_self (p v : S) (hv : v ≠ 1) : p * v ≠ p := fun h =>
  hv (mul_left_cancel (a := p) (by rw [h, mul_one]))

/-! ### Bounding an event whose index set is chosen elsewhere

`agree_sound_of_wrong` bounds a *fixed* cut, but the cut the gate scores is `ω`-dependent.
The way through is not a union bound over the possible cuts — that would cost `2^m` — but
the observation that the cut is decided by randomness independent of the bits being scored.
Decomposing over the cut's values then pays nothing: the probabilities of the values sum to
one, not to `2^m`. -/

/-- A worst-case bound survives an independently chosen index.  If `sel ω` always lands
in the finite set `T` and, for each value `t`, the event `sel ω = t` is independent of
`Bad t`, then a bound `E` holding for every fixed `t` holds for `Bad (sel ω)` itself.

The decomposition over `t` costs nothing: the probabilities of the values sum to one, not to
`#T`.  That is the whole reason the gate's `ω`-dependent accept side is usable — a union
bound over its possible values would cost `2^m`. -/
theorem measureReal_selection_le {β : Type*} [DecidableEq β] (T : Finset β)
    (sel : Ω → β) (hsel : ∀ ω, sel ω ∈ T)
    (hmeasSel : ∀ t, MeasurableSet {ω | sel ω = t})
    (Bad : β → Set Ω) (hmeasBad : ∀ t, MeasurableSet (Bad t)) (E : ℝ)
    (hindep : ∀ t ∈ T, μ.real ({ω | sel ω = t} ∩ Bad t) = μ.real {ω | sel ω = t} * μ.real (Bad t))
    (hbad : ∀ t ∈ T, μ.real (Bad t) ≤ E) (hE : 0 ≤ E) :
    μ.real {ω | ω ∈ Bad (sel ω)} ≤ E := by
  classical
  have hdisj : (T : Set β).PairwiseDisjoint (fun t => {ω | sel ω = t} ∩ Bad t) := by
    intro a _ b _ hab
    simp only [Function.onFun, Set.disjoint_left]
    rintro ω ⟨ha, -⟩ ⟨hb, -⟩
    exact hab (ha.symm.trans hb)
  have hdisj' : (T : Set β).PairwiseDisjoint (fun t => {ω | sel ω = t}) := by
    intro a _ b _ hab
    simp only [Function.onFun, Set.disjoint_left]
    exact fun ω ha hb => hab (ha.symm.trans hb)
  have hcover : {ω | ω ∈ Bad (sel ω)} = ⋃ t ∈ T, ({ω | sel ω = t} ∩ Bad t) := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Finset.mem_coe, exists_prop]
    exact ⟨fun h => ⟨sel ω, hsel ω, rfl, h⟩, fun ⟨t, _, he, hb⟩ => he ▸ hb⟩
  have htotal : ∑ t ∈ T, μ.real {ω | sel ω = t} = 1 := by
    have huniv : (Set.univ : Set Ω) = ⋃ t ∈ T, {ω | sel ω = t} := by
      ext ω
      simp only [Set.mem_univ, Set.mem_iUnion, Finset.mem_coe, Set.mem_setOf_eq, exists_prop,
        true_iff]
      exact ⟨sel ω, hsel ω, rfl⟩
    have hb := measureReal_biUnion_finset (μ := μ) hdisj' (fun t _ => hmeasSel t)
    rw [← hb, ← huniv, measureReal_def, measure_univ, ENNReal.toReal_one]
  calc μ.real {ω | ω ∈ Bad (sel ω)}
      = ∑ t ∈ T, μ.real ({ω | sel ω = t} ∩ Bad t) := by
        rw [hcover, measureReal_biUnion_finset hdisj (fun t _ => (hmeasSel t).inter (hmeasBad t))]
    _ = ∑ t ∈ T, μ.real {ω | sel ω = t} * μ.real (Bad t) := Finset.sum_congr rfl hindep
    _ ≤ ∑ t ∈ T, μ.real {ω | sel ω = t} * E :=
        Finset.sum_le_sum (fun t ht =>
          mul_le_mul_of_nonneg_left (hbad t ht) measureReal_nonneg)
    _ = E := by rw [← Finset.sum_mul, htotal, one_mul]

/-! ### What the clustering reads

`measureReal_selection_le` needs the gate's accept side decided by randomness independent of
the bits the gate scores.  These lemmas pin down which bits decide it: the clustering and
the votes read the oracle only at `p · v` for a representative prefix `p` and a candidate
suffix `v` — never at a bare prefix. -/

open scoped Classical in
/-- The query strings the clustering reads. -/
noncomputable def readSet (P cands : Finset S) : Finset S :=
  (P ×ˢ cands).image (fun z => z.1 * z.2)

lemma mem_readSet {P cands : Finset S} {p v : S} (hp : p ∈ P) (hv : v ∈ cands) :
    p * v ∈ readSet P cands := by
  classical
  exact Finset.mem_image.2 ⟨(p, v), Finset.mem_product.2 ⟨hp, hv⟩, rfl⟩

lemma mq_congr (O : Oracle μ S) {w : S} {ω ω' : Ω} (h : O.noise w ω = O.noise w ω') :
    O.mq w ω = O.mq w ω' := by simp [Oracle.mq, h]

lemma voteCount_congr (O : Oracle μ S) (F : Finset S) (p : S) {ω ω' : Ω}
    (h : ∀ v ∈ F, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1)) :
    voteCount O.mq F p ω = voteCount O.mq F p ω' := by
  classical
  unfold voteCount
  exact congrArg Finset.card (Finset.filter_congr (fun v hv => h v hv))

lemma hammingLoss_congr (O : Oracle μ S) (F : Finset S) (cn cd : ℕ) {P cands : Finset S}
    (hF : F ⊆ cands) {ω ω' : Ω} {v : S} (hv : v ∈ cands)
    (h : ∀ w ∈ readSet P cands, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    hammingLoss O.mq F cn cd P ω v = hammingLoss O.mq F cn cd P ω' v := by
  classical
  unfold hammingLoss
  refine congrArg _ (congrArg Finset.card (Finset.filter_congr (fun p hp => ?_)))
  rw [voteCount_congr O F p (fun v' hv' => h _ (mem_readSet hp (hF hv')))]
  exact not_congr (iff_congr (h _ (mem_readSet hp hv)) Iff.rfl)

lemma leastLossSubset_subset' (l : S → ℝ) (cands : Finset S) (k : ℕ) :
    leastLossSubset l cands k ⊆ cands := by
  classical
  unfold leastLossSubset
  split_ifs with hne
  · exact (Finset.mem_powersetCard.mp (Finset.exists_min_image _ _ hne).choose_spec.1).1
  · exact Finset.empty_subset _

lemma clusterLoss_congr (O : Oracle μ S) (F : Finset S) (cn cd : ℕ) (P cands : Finset S)
    (hF : F ⊆ cands) {ω ω' : Ω} (h : ∀ w ∈ readSet P cands, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    clusterLoss O.mq F cn cd P cands ω = clusterLoss O.mq F cn cd P cands ω' := by
  classical
  funext v
  unfold clusterLoss
  split_ifs with hv
  · exact hammingLoss_congr O F cn cd hF hv h
  · rfl

lemma lloydStep_congr (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (k : ℕ) {F : Finset S}
    (hF : F ⊆ cands) {ω ω' : Ω} (h : ∀ w ∈ readSet P cands, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    lloydStep O.mq cn cd P cands ω k F = lloydStep O.mq cn cd P cands ω' k F := by
  classical
  unfold lloydStep
  rw [clusterLoss_congr O F cn cd P cands hF h]

lemma clusterLoss_nonneg (O : Oracle μ S) (F : Finset S) (cn cd : ℕ) (P cands : Finset S)
    (ω : Ω) (v : S) : 0 ≤ clusterLoss O.mq F cn cd P cands ω v := by
  classical
  unfold clusterLoss hammingLoss
  split_ifs
  · exact Nat.cast_nonneg _
  · exact le_rfl

open scoped Classical in
/-- The seed's own loss against its own column is zero, so the first step always ranks
it first — which is what stops the clustering from stalling at `{ε}`. -/
lemma clusterLoss_seed_zero (O : Oracle μ S) {cn cd : ℕ} (hcd : cn < cd) (P cands : Finset S)
    (ω : Ω) (hone : (1 : S) ∈ cands) :
    clusterLoss O.mq {(1 : S)} cn cd P cands ω 1 = 0 := by
  classical
  unfold clusterLoss
  rw [if_pos hone, hammingLoss]
  have hempty : P.filter (fun p => ¬ ((O.mq (p * 1) ω = 1)
      ↔ cn * ({(1 : S)} : Finset S).card < cd * voteCount O.mq {(1 : S)} p ω)) = ∅ := by
    refine Finset.filter_eq_empty_iff.2 (fun p _ => ?_)
    simp only [Classical.not_not, Finset.card_singleton, mul_one, mul_comm]
    unfold voteCount
    by_cases h : O.mq p ω = 1
    · have hp : ({(1 : S)} : Finset S).filter (fun v => O.mq (p * v) ω = 1) = {(1 : S)} := by
        refine Finset.filter_eq_self.2 (fun v hv => ?_)
        rw [Finset.mem_singleton.1 hv, mul_one]
        exact h
      rw [hp]
      simp only [Finset.card_singleton, mul_one]
      exact ⟨fun _ => hcd, fun _ => h⟩
    · have hp : ({(1 : S)} : Finset S).filter (fun v => O.mq (p * v) ω = 1) = ∅ := by
        refine Finset.filter_eq_empty_iff.2 (fun v hv => ?_)
        rw [Finset.mem_singleton.1 hv, mul_one]
        exact h
      rw [hp]
      simp only [Finset.card_empty, mul_zero]
      exact ⟨fun hc => absurd hc h, fun hc => absurd hc (by omega)⟩
  rw [hempty]
  simp

open scoped Classical in
/-- The first step takes its cohort: the seed ranks first and the rest are the `k−1` next. -/
lemma lloydStep_seed_card (O : Oracle μ S) {cn cd : ℕ} (hcd : cn < cd) (P cands : Finset S)
    (ω : Ω) (k : ℕ) (hone : (1 : S) ∈ cands) (hk : k ≤ cands.card) (hkpos : 0 < k) :
    (lloydStep O.mq cn cd P cands ω k {(1 : S)}).card = k := by
  classical
  have hkm : k - 1 ≤ (cands.erase 1).card := by
    rw [Finset.card_erase_of_mem hone]; omega
  have hguard : ∀ w ∈ cands, w ∉ insert (1 : S)
        (leastLossSubset (clusterLoss O.mq {(1 : S)} cn cd P cands ω) (cands.erase 1) (k - 1)) →
      ∀ v ∈ insert (1 : S)
        (leastLossSubset (clusterLoss O.mq {(1 : S)} cn cd P cands ω) (cands.erase 1) (k - 1)),
      clusterLoss O.mq {(1 : S)} cn cd P cands ω v
        ≤ clusterLoss O.mq {(1 : S)} cn cd P cands ω w := by
    intro w hw hwn v hv
    rcases Finset.mem_insert.1 hv with rfl | hv'
    · rw [clusterLoss_seed_zero O hcd P cands ω hone]
      exact clusterLoss_nonneg O _ cn cd P cands ω w
    · refine leastLossSubset_least _ (cands.erase 1) (k - 1) hkm v hv' w
        (Finset.mem_erase.2 ⟨fun hc => hwn (hc ▸ Finset.mem_insert_self _ _), hw⟩)
        (fun hc => hwn (Finset.mem_insert_of_mem hc))
  unfold lloydStep
  rw [if_pos hguard, Finset.card_insert_of_notMem (fun hc =>
    (Finset.mem_erase.1 (leastLossSubset_subset' _ _ _ hc)).1 rfl),
    leastLossSubset_card _ _ _ hkm]
  omega

open scoped Classical in
/-- A step from a `k`-member family keeps `k` members. -/
lemma lloydStep_card_keep (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (hone : (1 : S) ∈ cands) (hk : k ≤ cands.card) (hkpos : 0 < k) {F : Finset S}
    (hF : F.card = k) : (lloydStep O.mq cn cd P cands ω k F).card = k := by
  classical
  have hkm : k - 1 ≤ (cands.erase 1).card := by
    rw [Finset.card_erase_of_mem hone]; omega
  unfold lloydStep
  split_ifs
  · rw [Finset.card_insert_of_notMem (fun hc =>
      (Finset.mem_erase.1 (leastLossSubset_subset' _ _ _ hc)).1 rfl),
      leastLossSubset_card _ _ _ hkm]
    omega
  · exact hF

open scoped Classical in
/-- The clustering does not stall.  The seed's loss against its own column is zero, so
the first step is taken and every later one either keeps its `k` members or retakes `k`. -/
theorem clusterAround_card (O : Oracle μ S) {cn cd : ℕ} (hcd : cn < cd) (P cands : Finset S)
    (ω : Ω) (k : ℕ) (hone : (1 : S) ∈ cands) (hk : k ≤ cands.card) (hkpos : 0 < k) :
    (clusterAround O.mq cn cd P cands ω k).card = k := by
  classical
  unfold clusterAround
  have hiter : ∀ (n : ℕ) (F : Finset S), F.card = k →
      ((lloydStep O.mq cn cd P cands ω k)^[n] F).card = k := by
    intro n
    induction n with
    | zero => intro F hF; rwa [Function.iterate_zero_apply]
    | succ n ih =>
        intro F hF
        rw [Function.iterate_succ_apply]
        exact ih _ (lloydStep_card_keep O cn cd P cands ω k hone hk hkpos hF)
  rw [show k * P.card + 1 = (k * P.card) + 1 from rfl, Function.iterate_succ_apply]
  exact hiter _ _ (lloydStep_seed_card O hcd P cands ω k hone hk hkpos)

lemma leastLossSubset_card_le (l : S → ℝ) (cands : Finset S) (k : ℕ) :
    (leastLossSubset l cands k).card ≤ k := by
  unfold leastLossSubset
  split_ifs with h
  · exact le_of_eq (Finset.mem_powersetCard.1
      (Finset.exists_min_image (cands.powersetCard k) (fun T => ∑ x ∈ T, l x) h).choose_spec.1).2
  · simp

open scoped Classical in
/-- The family never outgrows the round. -/
lemma clusterAround_card_le (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (hk : 0 < k) : (clusterAround O.mq cn cd P cands ω k).card ≤ k := by
  classical
  unfold clusterAround
  have hstep : ∀ F : Finset S, F.card ≤ k → (lloydStep O.mq cn cd P cands ω k F).card ≤ k := by
    intro F hF
    unfold lloydStep
    split_ifs
    · refine le_trans (Finset.card_insert_le _ _) ?_
      have := leastLossSubset_card_le (clusterLoss O.mq F cn cd P cands ω) (cands.erase 1) (k - 1)
      omega
    · exact hF
  have hiter : ∀ (n : ℕ) (F : Finset S), F.card ≤ k →
      ((lloydStep O.mq cn cd P cands ω k)^[n] F).card ≤ k := by
    intro n
    induction n with
    | zero => intro F hF; rwa [Function.iterate_zero_apply]
    | succ n ih =>
        intro F hF
        rw [Function.iterate_succ_apply]
        exact ih _ (hstep F hF)
  exact hiter _ _ (by rw [Finset.card_singleton]; exact hk)

lemma lloydStep_subset (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (hone : (1 : S) ∈ cands) {F : Finset S} (hF : F ⊆ cands) :
    lloydStep O.mq cn cd P cands ω k F ⊆ cands := by
  classical
  unfold lloydStep
  split_ifs
  · exact Finset.insert_subset hone
      (le_trans (leastLossSubset_subset' _ _ _) (Finset.erase_subset _ _))
  · exact hF

lemma lloydIterate_subset (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (hone : (1 : S) ∈ cands) :
    ∀ (n : ℕ) (F : Finset S), F ⊆ cands → (lloydStep O.mq cn cd P cands ω k)^[n] F ⊆ cands := by
  intro n
  induction n with
  | zero => intro F hF; rw [Function.iterate_zero_apply]; exact hF
  | succ n ih =>
      intro F hF
      rw [Function.iterate_succ_apply]
      exact ih _ (lloydStep_subset O cn cd P cands ω k hone hF)

lemma lloydIterate_congr (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands)
    {ω ω' : Ω} (h : ∀ w ∈ readSet P cands, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    ∀ (n : ℕ) (F : Finset S), F ⊆ cands →
      (lloydStep O.mq cn cd P cands ω k)^[n] F = (lloydStep O.mq cn cd P cands ω' k)^[n] F := by
  intro n
  induction n with
  | zero => intro F _; rw [Function.iterate_zero_apply, Function.iterate_zero_apply]
  | succ n ih =>
      intro F hF
      calc (lloydStep O.mq cn cd P cands ω k)^[n + 1] F
          = (lloydStep O.mq cn cd P cands ω k)^[n] (lloydStep O.mq cn cd P cands ω k F) :=
            Function.iterate_succ_apply _ _ _
        _ = (lloydStep O.mq cn cd P cands ω k)^[n] (lloydStep O.mq cn cd P cands ω' k F) :=
            congrArg (fun z => (lloydStep O.mq cn cd P cands ω k)^[n] z)
              (lloydStep_congr O cn cd P cands k hF h)
        _ = (lloydStep O.mq cn cd P cands ω' k)^[n] (lloydStep O.mq cn cd P cands ω' k F) :=
            ih _ (lloydStep_subset O cn cd P cands ω' k hone hF)
        _ = (lloydStep O.mq cn cd P cands ω' k)^[n + 1] F :=
            (Function.iterate_succ_apply _ _ _).symm

/-- The cluster reads only `readSet`.  Two noise draws agreeing at `p · v` for every
representative prefix and candidate suffix give the same family — so neither the family nor
any vote cast with it is decided by the oracle's bit at a bare prefix, which is the bit the
gate scores. -/
lemma clusterAround_congr_mq (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (k : ℕ)
    {ω ω' : Ω} (hone : (1 : S) ∈ cands)
    (h : ∀ w ∈ readSet P cands, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    clusterAround O.mq cn cd P cands ω k = clusterAround O.mq cn cd P cands ω' k :=
  lloydIterate_congr O cn cd P cands k hone h _ _ (by simpa using hone)

/-- The screen reads only `readSet`.  Its two reads at a prefix are `p · v` and
`p · ε = p`, and the seed is a candidate, so both are already there. -/
lemma screenCount_congr (O : Oracle μ S) {P cands : Finset S} (hone : (1 : S) ∈ cands)
    {v : S} (hv : v ∈ cands) {ω ω' : Ω}
    (h : ∀ w ∈ readSet P cands, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    screenCount O.mq P v ω = screenCount O.mq P v ω' := by
  classical
  unfold screenCount
  refine congrArg Finset.card (Finset.filter_congr (fun p hp => ?_))
  have h1 := h _ (mem_readSet hp hv)
  have h0 : (O.mq p ω = 1 ↔ O.mq p ω' = 1) := by
    have := h _ (mem_readSet hp hone)
    rwa [mul_one] at this
  exact not_congr (iff_congr h1 h0)

open scoped Classical in
/-- The floor reads only `readSet` too: it is an infimum of counts that each do. -/
lemma screenBase_congr (O : Oracle μ S) {P cands : Finset S} (hone : (1 : S) ∈ cands)
    {ω ω' : Ω} (h : ∀ w ∈ readSet P cands, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    screenBase O.mq P cands ω = screenBase O.mq P cands ω' := by
  classical
  unfold screenBase
  by_cases hne : (cands.erase 1).Nonempty
  · rw [dif_pos hne, dif_pos hne]
    exact Finset.inf'_congr hne rfl
      (fun v hv => screenCount_congr O hone (Finset.mem_of_mem_erase hv) h)
  · rw [dif_neg hne, dif_neg hne]

lemma screenedAt_congr (O : Oracle μ S) (populations : Finset J) (B : State)
    (d : ((ℕ → S) × (J → ℕ → S)) × (J → ℕ → S)) {ω ω' : Ω}
    (h : ∀ w ∈ readSet (prefixesAt populations B.npref ((ω, d) : Run Ω S J))
      (poolAt B.nsuff ((ω, d) : Run Ω S J)), (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    screenedAt O.mq populations B (ω, d) = screenedAt O.mq populations B (ω', d) := by
  classical
  unfold screenedAt
  refine Finset.filter_congr (fun v hv => ?_)
  show B.scd * screenCount O.mq (prefixesAt populations B.npref ((ω, d) : Run Ω S J)) v ω
      ≤ B.scd * screenBase O.mq (prefixesAt populations B.npref ((ω, d) : Run Ω S J))
          (poolAt B.nsuff ((ω, d) : Run Ω S J)) ω
        + B.sc * (prefixesAt populations B.npref ((ω, d) : Run Ω S J)).card
    ↔ B.scd * screenCount O.mq (prefixesAt populations B.npref ((ω', d) : Run Ω S J)) v ω'
      ≤ B.scd * screenBase O.mq (prefixesAt populations B.npref ((ω', d) : Run Ω S J))
          (poolAt B.nsuff ((ω', d) : Run Ω S J)) ω'
        + B.sc * (prefixesAt populations B.npref ((ω', d) : Run Ω S J)).card
  rw [screenCount_congr O (one_mem_poolAt _ _) hv h,
    screenBase_congr O (one_mem_poolAt _ _) h]
  rfl

/-! ### The prefix alphabet

The gate scores the oracle's bit at a prefix `p`.  Everything that decides *which side* `p`
falls on — the family, and `p`'s own vote — is read at strings `q · v` with `v ≠ ε`.  For
the gate's null to be honest those must be different strings, and that is a property of
where prefixes come from, not of the algorithm. -/

/-- No prefix is another prefix extended. -/
lemma flat_eq_one {Pre : Set S} (hflat : Flat Pre) {p p' : S} (hp : p ∈ Pre) (hp' : p' ∈ Pre)
    {v : S} (h : p * v = p') : v = 1 := by
  have hpp : p = p' := hflat p hp p' hp' v 1 (by rw [h, mul_one])
  exact mul_left_cancel (a := p) (by rw [h, hpp, mul_one])

/-- On a flat alphabet the gate's query string is never one of the split's. -/
lemma flat_ne_of_ne_one {Pre : Set S} (hflat : Flat Pre) {p p' : S} (hp : p ∈ Pre)
    (hp' : p' ∈ Pre) {v : S} (hv : v ≠ 1) : p * v ≠ p' :=
  fun h => hv (flat_eq_one hflat hp hp' h)

lemma clusterAround_subset (O : Oracle μ S) (cn cd : ℕ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (hone : (1 : S) ∈ cands) : clusterAround O.mq cn cd P cands ω k ⊆ cands := by
  classical
  unfold clusterAround
  exact lloydIterate_subset O cn cd P cands ω k hone _ _ (by simpa using hone)

lemma clusterAt_subset (O : Oracle μ S) (populations : Finset J) (B : State)
    (x : Run Ω S J) : clusterAt O.mq populations x B ⊆ poolAt B.nsuff x :=
  fun v hv => screenedAt_subset O populations B x
    (clusterAround_subset O B.cn B.cd _ _ (oracleNoise x) B.k (one_mem_screenedAt O populations B x) hv)

/-- The family is decided by the bits on `readSet` — the screen's reads and the
clustering's alike. -/
lemma clusterAt_congr (O : Oracle μ S) (populations : Finset J) (B : State)
    (d : ((ℕ → S) × (J → ℕ → S)) × (J → ℕ → S)) {ω ω' : Ω}
    (h : ∀ w ∈ readSet (prefixesAt populations B.npref ((ω, d) : Run Ω S J))
      (poolAt B.nsuff ((ω, d) : Run Ω S J)), O.noise w ω = O.noise w ω') :
    clusterAt O.mq populations (ω, d) B = clusterAt O.mq populations (ω', d) B := by
  classical
  have hbit : ∀ w ∈ readSet (prefixesAt populations B.npref ((ω, d) : Run Ω S J))
      (poolAt B.nsuff ((ω, d) : Run Ω S J)), (O.mq w ω = 1 ↔ O.mq w ω' = 1) :=
    fun w hw => by rw [mq_congr O (h w hw)]
  have hcands : screenedAt O.mq populations B (ω, d) = screenedAt O.mq populations B (ω', d) :=
    screenedAt_congr O populations B d hbit
  unfold clusterAt
  rw [show oracleNoise ((ω, d) : Run Ω S J) = ω from rfl, show oracleNoise ((ω', d) : Run Ω S J) = ω' from rfl,
    ← hcands]
  refine clusterAround_congr_mq O B.cn B.cd _ _ B.k (one_mem_screenedAt O populations B (ω, d))
    (fun w hw => hbit w ?_)
  exact Finset.mem_image.2 (by
    obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hw
    obtain ⟨hp, hvs⟩ := Finset.mem_product.1 hpv
    exact ⟨(p, v), Finset.mem_product.2 ⟨hp,
      screenedAt_subset O populations B (ω, d) hvs⟩, rfl⟩)

open scoped Classical in
/-- Dropping the seed costs the vote one count.  `1 ∈ F` always, so the full family's
vote at `p` is the erased family's plus the seed's own read there. -/
lemma voteCount_le_erase_succ (O : Oracle μ S) (F : Finset S) (p : S) (ω : Ω) :
    voteCount O.mq F p ω ≤ voteCount O.mq (F.erase 1) p ω + 1 := by
  classical
  unfold voteCount
  have hsub : F.filter (fun v => O.mq (p * v) ω = 1)
      ⊆ insert (1 : S) ((F.erase 1).filter (fun v => O.mq (p * v) ω = 1)) := by
    intro v hv
    obtain ⟨hvF, hvm⟩ := Finset.mem_filter.1 hv
    by_cases h1 : v = 1
    · rw [h1]
      exact Finset.mem_insert_self _ _
    · exact Finset.mem_insert_of_mem (Finset.mem_filter.2 ⟨Finset.mem_erase.2 ⟨h1, hvF⟩, hvm⟩)
  exact le_trans (Finset.card_le_card hsub) (Finset.card_insert_le _ _)

/-- The gate's own query strings are not read by the clustering.  A prefix is never
`p · v` for a prefix `p` and any suffix, on a flat alphabet — so the bits the gate scores
are untouched by everything that decides which side each prefix falls on. -/
lemma disjoint_readSet {Pre : Set S} (hflat : Flat Pre) {P cands C : Finset S}
    (hP : ∀ p ∈ P, p ∈ Pre) (hC : ∀ p ∈ C, p ∈ Pre) (hPC : Disjoint P C) :
    Disjoint C (readSet P cands) := by
  classical
  refine Finset.disjoint_left.2 (fun z hz hmem => ?_)
  obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hmem
  obtain ⟨hp, -⟩ := Finset.mem_product.1 hpv
  have hv1 : v = 1 := flat_eq_one hflat (hP p hp) (hC _ hz) rfl
  rw [hv1, mul_one] at hz
  exact (Finset.disjoint_left.1 hPC hp) hz

/-- The same, for the gate's own prefixes against their own votes: those read `p · v` with
`v ≠ ε`, which is never a prefix. -/
lemma disjoint_readSet_erase {Pre : Set S} (hflat : Flat Pre) {cands C : Finset S}
    (hC : ∀ p ∈ C, p ∈ Pre) : Disjoint C (readSet C (cands.erase 1)) := by
  classical
  refine Finset.disjoint_left.2 (fun z hz hmem => ?_)
  obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hmem
  obtain ⟨hp, hv⟩ := Finset.mem_product.1 hpv
  exact (Finset.mem_erase.1 hv).1 (flat_eq_one hflat (hC p hp) (hC _ hz) rfl)

/-- A population prefix the table does not hold is read nowhere by the clustering.  Its
query strings `p · v` collide with the clustering's `q · v'` only if `p = q`. -/
lemma disjoint_image_readSet {Pre : Set S} (hflat : Flat Pre) {P cands : Finset S} {p : S}
    (hP : ∀ q ∈ P, q ∈ Pre) (hp : p ∈ Pre) (hpP : p ∉ P) :
    Disjoint (↑(cands.image (fun v => p * v)) : Set S) (↑(readSet P cands) : Set S) := by
  classical
  rw [Finset.disjoint_coe, Finset.disjoint_left]
  rintro z hz hmem
  obtain ⟨v, -, rfl⟩ := Finset.mem_image.1 hz
  obtain ⟨⟨q, v'⟩, hqv, hq⟩ := Finset.mem_image.1 hmem
  obtain ⟨hqP, -⟩ := Finset.mem_product.1 hqv
  exact hpP (hflat p hp q (hP q hqP) v v' hq.symm ▸ hqP)

/-! ### Independence of the score from the side

Both events are structurally measurable over disjoint sub-families of the oracle's bits, so
this needs no factorisation theorem: the cut is decided by the bits at `p · v` over the
family, the score reads the bits at the certification prefixes themselves, and
`disjoint_readSet` separates them. -/

/-- The σ-algebra the oracle's bits on a set of query strings generate. -/
def noiseAlg (O : Oracle μ S) (T : Set S) : MeasurableSpace Ω :=
  ⨆ w ∈ T, MeasurableSpace.comap (O.noise w) inferInstance

lemma noiseAlg_le (O : Oracle μ S) (T : Set S) : noiseAlg O T ≤ ‹MeasurableSpace Ω› :=
  iSup₂_le (fun w _ => (O.noise_meas w).comap_le)

lemma measurableSet_noise_preimage (O : Oracle μ S) {T : Set S} {w : S} (hw : w ∈ T)
    {s : Set ℝ} (hs : MeasurableSet s) :
    MeasurableSet[noiseAlg O T] (O.noise w ⁻¹' s) := by
  have hle : MeasurableSpace.comap (O.noise w) inferInstance ≤ noiseAlg O T :=
    le_iSup₂ (f := fun w (_ : w ∈ T) => MeasurableSpace.comap (O.noise w) inferInstance) w hw
  exact hle _ ⟨s, hs, rfl⟩

lemma measurableSet_mq_eq_one (O : Oracle μ S) {T : Set S} {w : S} (hw : w ∈ T) :
    MeasurableSet[noiseAlg O T] {ω | O.mq w ω = 1} := by
  have hpre : {ω | O.mq w ω = 1}
      = O.noise w ⁻¹' {r : ℝ | O.label w + (1 - 2 * O.label w) * r = 1} := rfl
  rw [hpre]
  exact measurableSet_noise_preimage O hw
    (measurableSet_eq_fun (by fun_prop) measurable_const)

/-- Two independent blocks of the oracle's bits. -/
lemma indep_noiseAlg (O : Oracle μ S) {T T' : Set S} (h : Disjoint T T') :
    Indep (noiseAlg O T) (noiseAlg O T') μ :=
  indep_iSup_of_disjoint (fun w => (O.noise_meas w).comap_le) O.noise_indep h

open scoped Classical in
/-- A block's bits decide which of its strings satisfy any condition they decide. -/
lemma measurableSet_filter_fiber' (O : Oracle μ S) {T : Set S} {A : Finset S}
    (Pr : S → Ω → Prop) [inst : ∀ ω, DecidablePred (fun p => Pr p ω)]
    (hPr : ∀ p ∈ A, MeasurableSet[noiseAlg O T] {ω | Pr p ω}) (U : Finset S) :
    MeasurableSet[noiseAlg O T] {ω | A.filter (fun p => Pr p ω) = U} := by
  classical
  have hfib : {ω | A.filter (fun p => Pr p ω) = U}
      = if U ⊆ A then (⋂ p ∈ U, {ω | Pr p ω}) ∩ ⋂ p ∈ A \ U, {ω | Pr p ω}ᶜ
        else ∅ := by
    split_ifs with hUA
    · ext ω
      simp only [Set.mem_setOf_eq, Set.mem_inter_iff, Set.mem_iInter, Set.mem_compl_iff,
        Finset.mem_sdiff]
      constructor
      · rintro rfl
        exact ⟨fun p hp => (Finset.mem_filter.1 hp).2,
          fun p hp => fun hc => hp.2 (Finset.mem_filter.2 ⟨hp.1, hc⟩)⟩
      · rintro ⟨h1, h2⟩
        ext p
        simp only [Finset.mem_filter]
        exact ⟨fun ⟨hpA, hpm⟩ => by_contra (fun hpU => h2 p ⟨hpA, hpU⟩ hpm),
          fun hpU => ⟨hUA hpU, h1 p hpU⟩⟩
    · ext ω
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
      exact fun hc => hUA (hc ▸ Finset.filter_subset _ _)
  rw [hfib]
  split_ifs with hUA
  · apply MeasurableSet.inter
    · apply Finset.measurableSet_biInter
      exact fun p hp => hPr p (hUA hp)
    · apply Finset.measurableSet_biInter
      exact fun p hp => (hPr p (Finset.mem_sdiff.1 hp).1).compl
  · exact (noiseAlg O T).measurableSet_empty

lemma measurableSet_noise_eq_one (O : Oracle μ S) {T : Set S} {w : S} (hw : w ∈ T) :
    MeasurableSet[noiseAlg O T] {ω | O.noise w ω = 1} :=
  measurableSet_noise_preimage O hw (measurableSet_singleton 1)

open scoped Classical in
/-- Which of a block's prefixes read as accepting is decided by that block's bits. -/
lemma measurableSet_filter_fiber (O : Oracle μ S) {T : Set S} {A : Finset S} (hA : ↑A ⊆ T)
    (U : Finset S) :
    MeasurableSet[noiseAlg O T] {ω | A.filter (fun p => O.mq p ω = 1) = U} :=
  measurableSet_filter_fiber' O _ (fun p hp => measurableSet_mq_eq_one O (hA hp)) U

open scoped Classical in
/-- The bit pattern a block's strings show. -/
noncomputable def noisePattern (O : Oracle μ S) (Q : Finset S) (ω : Ω) : Finset S :=
  Q.filter (fun w => O.noise w ω = 1)

lemma noisePattern_mem (O : Oracle μ S) (Q : Finset S) (ω : Ω) :
    noisePattern O Q ω ∈ Q.powerset :=
  Finset.mem_powerset.2 (Finset.filter_subset _ _)

lemma measurableSet_noisePattern (O : Oracle μ S) (Q : Finset S) (t : Finset S) :
    MeasurableSet[noiseAlg O ↑Q] {ω | noisePattern O Q ω = t} :=
  measurableSet_filter_fiber' O _ (fun w hw => measurableSet_noise_eq_one O hw) t

open scoped Classical in
/-- The runs on which a block's bits are all genuinely `0` or `1`. -/
def noiseClean (O : Oracle μ S) (Q : Finset S) : Set Ω :=
  {ω | ∀ w ∈ Q, O.noise w ω = 0 ∨ O.noise w ω = 1}

lemma measurableSet_noiseClean (O : Oracle μ S) (Q : Finset S) :
    MeasurableSet[noiseAlg O ↑Q] (noiseClean O Q) := by
  classical
  have hset : noiseClean O Q = ⋂ w ∈ Q, (O.noise w ⁻¹' ({0, 1} : Set ℝ)) := by
    ext ω; simp [noiseClean, Set.mem_iInter]
  rw [hset]
  apply Finset.measurableSet_biInter
  exact fun w hw => measurableSet_noise_preimage O hw
    ((measurableSet_singleton 0).union (measurableSet_singleton 1))

lemma noiseClean_ae (O : Oracle μ S) (Q : Finset S) : μ.real (noiseClean O Q)ᶜ = 0 := by
  classical
  have hae : ∀ᵐ ω ∂μ, ω ∈ noiseClean O Q := by
    filter_upwards [(ae_ball_iff Q.countable_toSet).2 (fun w _ => O.noise_bit w)] with ω hω
    exact fun w hw => hω w hw
  have hz : μ (noiseClean O Q)ᶜ = 0 := by
    have h1 := MeasureTheory.ae_iff.1 hae
    rwa [show {a | a ∉ noiseClean O Q} = (noiseClean O Q)ᶜ from rfl] at h1
  simp [measureReal_def, hz]

/-- On a clean run the bit pattern pins the bits down. -/
lemma noise_eq_of_pattern (O : Oracle μ S) {Q : Finset S} {ω ω' : Ω}
    (hω : ω ∈ noiseClean O Q) (hω' : ω' ∈ noiseClean O Q)
    (h : noisePattern O Q ω = noisePattern O Q ω') :
    ∀ w ∈ Q, O.noise w ω = O.noise w ω' := by
  classical
  intro w hw
  have hiff : (O.noise w ω = 1) ↔ (O.noise w ω' = 1) := by
    constructor
    · intro h1
      have : w ∈ noisePattern O Q ω := Finset.mem_filter.2 ⟨hw, h1⟩
      rw [h] at this
      exact (Finset.mem_filter.1 this).2
    · intro h1
      have : w ∈ noisePattern O Q ω' := Finset.mem_filter.2 ⟨hw, h1⟩
      rw [← h] at this
      exact (Finset.mem_filter.1 this).2
  rcases hω w hw with h0 | h1
  · rcases hω' w hw with h0' | h1'
    · rw [h0, h0']
    · exact absurd (hiff.2 h1') (by rw [h0]; norm_num)
  · rcases hω' w hw with h0' | h1'
    · exact absurd (hiff.1 h1) (by rw [h0']; norm_num)
    · rw [h1, h1']

open scoped Classical in
/-- Hence any condition on that count is: it takes finitely many values. -/
lemma measurableSet_filter_pred (O : Oracle μ S) {T : Set S} {A : Finset S} (hA : ↑A ⊆ T)
    (P : Finset S → Prop) :
    MeasurableSet[noiseAlg O T] {ω | P (A.filter (fun p => O.mq p ω = 1))} := by
  classical
  have hcover : {ω | P (A.filter (fun p => O.mq p ω = 1))}
      = ⋃ U ∈ A.powerset.filter P, {ω | A.filter (fun p => O.mq p ω = 1) = U} := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Finset.mem_coe, Finset.mem_filter,
      Finset.mem_powerset, exists_prop]
    refine ⟨fun h => ⟨_, ⟨Finset.filter_subset _ _, h⟩, rfl⟩, ?_⟩
    rintro ⟨U, ⟨-, hPU⟩, rfl⟩
    exact hPU
  rw [hcover]
  apply Finset.measurableSet_biUnion
  exact fun U _ => measurableSet_filter_fiber O hA U

open scoped Classical in
/-- The same for an arbitrary per-element predicate on the bits. -/
lemma measurableSet_filter_pred' (O : Oracle μ S) {T : Set S} {A : Finset S}
    (Pr : S → Ω → Prop) [inst : ∀ ω, DecidablePred (fun p => Pr p ω)]
    (hPr : ∀ p ∈ A, MeasurableSet[noiseAlg O T] {ω | Pr p ω}) (Q : Finset S → Prop) :
    MeasurableSet[noiseAlg O T] {ω | Q (A.filter (fun p => Pr p ω))} := by
  classical
  have hcover : {ω | Q (A.filter (fun p => Pr p ω))}
      = ⋃ U ∈ A.powerset.filter Q, {ω | A.filter (fun p => Pr p ω) = U} := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Finset.mem_coe, Finset.mem_filter,
      Finset.mem_powerset, exists_prop]
    refine ⟨fun h => ⟨_, ⟨Finset.filter_subset _ _, h⟩, rfl⟩, ?_⟩
    rintro ⟨U, ⟨-, hQU⟩, rfl⟩
    exact hQU
  rw [hcover]
  exact Finset.measurableSet_biUnion _ (fun U _ => measurableSet_filter_fiber' O Pr hPr U)

open scoped Classical in
/-- The same when the reads are taken at shifted strings `r v` rather than at `v` itself:
the vote at a population prefix reads `p · v`, not `v`. -/
lemma measurableSet_filter_pred_map (O : Oracle μ S) {T : Set S} {A : Finset S} (r : S → S)
    (hA : ∀ v ∈ A, r v ∈ T) (P : Finset S → Prop) :
    MeasurableSet[noiseAlg O T] {ω | P (A.filter (fun v => O.mq (r v) ω = 1))} := by
  classical
  have hcover : {ω | P (A.filter (fun v => O.mq (r v) ω = 1))}
      = ⋃ U ∈ A.powerset.filter P, {ω | A.filter (fun v => O.mq (r v) ω = 1) = U} := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Finset.mem_coe, Finset.mem_filter,
      Finset.mem_powerset, exists_prop]
    refine ⟨fun h => ⟨_, ⟨Finset.filter_subset _ _, h⟩, rfl⟩, ?_⟩
    rintro ⟨U, ⟨-, hPU⟩, rfl⟩
    exact hPU
  rw [hcover]
  refine Finset.measurableSet_biUnion _ (fun U _ => ?_)
  exact measurableSet_filter_fiber' O _ (fun v hv => measurableSet_mq_eq_one O (hA v hv)) U

open scoped Classical in
/-- Congruence becomes measurability.  A side decided by a block's bits is, on the clean
runs, a union of that block's pattern fibres. -/
lemma measurableSet_side_clean (O : Oracle μ S) (Q : Finset S) {β : Type*} [DecidableEq β]
    (side : Ω → β)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → side ω = side ω')
    (A₀ : β) :
    MeasurableSet[noiseAlg O ↑Q] ({ω | side ω = A₀} ∩ noiseClean O Q) := by
  classical
  have hcover : {ω | side ω = A₀} ∩ noiseClean O Q
      = ⋃ t ∈ Q.powerset.filter
          (fun t => ∃ ω, ω ∈ noiseClean O Q ∧ noisePattern O Q ω = t ∧ side ω = A₀),
        ({ω | noisePattern O Q ω = t} ∩ noiseClean O Q) := by
    ext ω
    simp only [Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_iUnion, Finset.mem_coe,
      Finset.mem_filter, Finset.mem_powerset, exists_prop]
    constructor
    · rintro ⟨hsd, hcl⟩
      exact ⟨noisePattern O Q ω,
        ⟨Finset.mem_powerset.1 (noisePattern_mem O Q ω), ω, hcl, rfl, hsd⟩, rfl, hcl⟩
    · rintro ⟨t, ⟨-, ω₀, hcl₀, hpat₀, hsd₀⟩, hpat, hcl⟩
      refine ⟨?_, hcl⟩
      rw [← hsd₀]
      exact hcongr ω ω₀ (noise_eq_of_pattern O hcl hcl₀ (hpat.trans hpat₀.symm))
  rw [hcover]
  apply Finset.measurableSet_biUnion
  exact fun t _ => (measurableSet_noisePattern O Q t).inter (measurableSet_noiseClean O Q)

open scoped Classical in
/-- A worst case survives the scoring rule being chosen elsewhere.  The read set is fixed at
`C`, and the selection is an arbitrary value the block `Q` determines — for the gate it is the
pair (accept side, decided set), which no single `Finset S` records but which the votes fix
all the same.  Conditioning is free: the selection is `Q`-measurable, the score is
`C`-measurable, and `C` and `Q` are disjoint. -/
theorem selection_read_bound (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    {β : Type*} [DecidableEq β] (T : Finset β) (t₀ : β) (ht₀ : t₀ ∈ T)
    (sel : Ω → β) (hsel : ∀ ω, sel ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → sel ω = sel ω')
    (P : β → Finset S → Prop) (E : ℝ) (hE : 0 ≤ E)
    (hbad : ∀ t ∈ T, μ.real {ω | P t (C.filter (fun p => O.mq p ω = 1))} ≤ E) :
    μ.real {ω | P (sel ω) (C.filter (fun p => O.mq p ω = 1))} ≤ E := by
  classical
  set Bad : β → Set Ω :=
    fun t => {ω | P t (C.filter (fun p => O.mq p ω = 1))} with hBaddef
  set sel' : Ω → β := fun ω => if ω ∈ noiseClean O Q then sel ω else t₀ with hsel'def
  have hmeasBad : ∀ t, MeasurableSet (Bad t) := fun t =>
    noiseAlg_le O Set.univ _ (measurableSet_filter_pred O (T := Set.univ) (by simp) _)
  have hmeasBadC : ∀ t, MeasurableSet[noiseAlg O ↑C] (Bad t) := fun t =>
    measurableSet_filter_pred O (le_refl (↑C : Set S)) _
  have hsel2 : ∀ ω, sel' ω ∈ T := by
    intro ω
    rw [hsel'def]
    by_cases hc : ω ∈ noiseClean O Q
    · simpa [hc] using hsel ω
    · simpa [hc] using ht₀
  have hsplit : ∀ t, {ω | sel' ω = t}
      = ({ω | sel ω = t} ∩ noiseClean O Q) ∪ (if t = t₀ then (noiseClean O Q)ᶜ else ∅) := by
    intro t
    ext ω
    by_cases hc : ω ∈ noiseClean O Q <;> by_cases he : t = t₀ <;>
      simp [hsel'def, hc, he, Set.mem_setOf_eq, eq_comm (a := t₀)]
  have hmeasSelQ : ∀ t, MeasurableSet[noiseAlg O ↑Q] {ω | sel' ω = t} := by
    intro t
    rw [hsplit t]
    refine MeasurableSet.union (measurableSet_side_clean O Q sel hcongr t) ?_
    split_ifs
    · exact (measurableSet_noiseClean O Q).compl
    · exact (noiseAlg O ↑Q).measurableSet_empty
  have hmeasSel : ∀ t, MeasurableSet {ω | sel' ω = t} := fun t =>
    noiseAlg_le O ↑Q _ (hmeasSelQ t)
  have hindep : ∀ t ∈ T,
      μ.real ({ω | sel' ω = t} ∩ Bad t) = μ.real {ω | sel' ω = t} * μ.real (Bad t) := by
    intro t _
    have hI := (indep_noiseAlg O hdisj.symm).indepSet_of_measurableSet (hmeasSelQ t)
      (hmeasBadC t)
    have := hI.measure_inter_eq_mul
    simp only [measureReal_def, this, ENNReal.toReal_mul]
  have hmain := measureReal_selection_le (μ := μ) T sel' hsel2 hmeasSel Bad hmeasBad E
    hindep (fun t ht => hbad t ht) hE
  have hsub : {ω | P (sel ω) (C.filter (fun p => O.mq p ω = 1))}
      ⊆ {ω | ω ∈ Bad (sel' ω)} ∪ (noiseClean O Q)ᶜ := by
    intro ω hω
    by_cases hc : ω ∈ noiseClean O Q
    · refine Or.inl ?_
      change ω ∈ Bad (sel' ω)
      rw [hsel'def]
      simp only [hc, if_pos]
      exact hω
    · exact Or.inr hc
  calc μ.real {ω | P (sel ω) (C.filter (fun p => O.mq p ω = 1))}
      ≤ μ.real ({ω | ω ∈ Bad (sel' ω)} ∪ (noiseClean O Q)ᶜ) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ μ.real {ω | ω ∈ Bad (sel' ω)} + μ.real (noiseClean O Q)ᶜ := measureReal_union_le _ _
    _ = μ.real {ω | ω ∈ Bad (sel' ω)} := by rw [noiseClean_ae O Q, add_zero]
    _ ≤ E := hmain

/-! ### The certification draws see the wrong set

The family and its wrong-set live in the run's first factor; the certification draws are the
second and are independent of them.  So with the wrong-set held fixed this is plain
Hoeffding over i.i.d. draws — no noise enters, which is the second thing the fresh stream
buys. -/

lemma measurableSet_of_countable (W : Set S) : MeasurableSet W :=
  (Set.to_countable W).measurableSet

open scoped Classical in
/-- A wrong set of mass `≥ εcov` is hit by all but `t` of that fraction of the draws,
except with probability `exp(−2mt²)`.

The margin `t` is free for the same reason the gate's rates are: it trades against what the
gate then has to resolve.  Charging it a fixed `εcov/2`, as an earlier version did, costs
the gate a factor of four in the drift it must see and sixteen in the prefixes that takes. -/
lemma cert_hits_wrongSet (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    (j : J) (m : ℕ) (W : Set S) (εcov t : ℝ) (hεcov : 0 ≤ εcov) (ht : 0 ≤ t)
    (hW : εcov ≤ (D j).real W) :
    (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j).real
        {c | (((Finset.range m).filter (fun i => c j i ∈ W)).card : ℝ)
          ≤ (m : ℝ) * (εcov - t)}
      ≤ Real.exp (-2 * (m : ℝ) * t ^ 2) := by
  classical
  set ν : Measure (J → ℕ → S) := Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j with hνdef
  have hWm : MeasurableSet W := measurableSet_of_countable W
  set ind : S → ℝ := W.indicator 1 with hinddef
  have hindm : Measurable ind := measurable_const.indicator hWm
  set X : ℕ → (J → ℕ → S) → ℝ := fun i c => ind (c j i) with hXdef
  have hevalj : MeasurePreserving (fun c : J → ℕ → S => c j) ν
      (Measure.infinitePi fun _ : ℕ => D j) :=
    measurePreserving_eval (fun j : J => Measure.infinitePi fun _ : ℕ => D j) j
  have hcoord : iIndepFun (fun (i : ℕ) (s : ℕ → S) => s i)
      (Measure.infinitePi fun _ : ℕ => D j) := iIndepFun_infinitePi (fun _ => measurable_id)
  have hindep : iIndepFun X ν :=
    (iIndepFun_comp_measurePreserving hevalj (fun _ => measurable_pi_apply _) hcoord).comp
      (fun _ => ind) (fun _ => hindm)
  have hmeas : ∀ i, AEMeasurable (X i) ν := fun i =>
    (hindm.comp ((measurable_pi_apply _).comp (measurable_pi_apply _))).aemeasurable
  have hicc : ∀ i, ∀ᵐ c ∂ν, X i c ∈ Set.Icc (0 : ℝ) 1 := by
    intro i
    filter_upwards with c
    by_cases h : c j i ∈ W
    · simp [hXdef, hinddef, Set.indicator_of_mem h]
    · simp [hXdef, hinddef, Set.indicator_of_notMem h]
  have hmean : ∀ i, ν[X i] = (D j).real W := by
    intro i
    have hmp : MeasurePreserving (fun c : J → ℕ → S => c j i) ν (D j) :=
      (measurePreserving_eval_infinitePi _ i).comp hevalj
    calc ν[X i] = ∫ s, ind s ∂(D j) := by
          rw [← hmp.map_eq,
            integral_map hmp.measurable.aemeasurable hindm.aestronglyMeasurable]
      _ = (D j).real W := by rw [hinddef, integral_indicator_one hWm]
  have hsum : ((Finset.range m).card : ℝ) * εcov ≤ ∑ i ∈ Finset.range m, ν[X i] := by
    rw [Finset.sum_congr rfl (fun i _ => hmean i), Finset.sum_const, nsmul_eq_mul,
      Finset.card_range]
    have : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
    nlinarith [hW]
  have hmain := sumLower_le X (Finset.range m) εcov t hmeas hindep hicc hsum ht
  have hcount : ∀ c : J → ℕ → S, ∑ i ∈ Finset.range m, X i c
      = (((Finset.range m).filter (fun i => c j i ∈ W)).card : ℝ) := by
    intro c
    rw [← Finset.sum_filter_add_sum_filter_not (Finset.range m) (fun i => c j i ∈ W)]
    have h1 : ∑ i ∈ (Finset.range m).filter (fun i => c j i ∈ W), X i c
        = (((Finset.range m).filter (fun i => c j i ∈ W)).card : ℝ) := by
      have hone : ∀ i ∈ (Finset.range m).filter (fun i => c j i ∈ W), X i c = (1 : ℝ) := by
        intro i hi
        simp [hXdef, hinddef, Set.indicator_of_mem (Finset.mem_filter.1 hi).2]
      rw [Finset.sum_congr rfl hone, Finset.sum_const, nsmul_eq_mul, mul_one]
    have h0 : ∑ i ∈ (Finset.range m).filter (fun i => ¬ (c j i ∈ W)), X i c = 0 :=
      Finset.sum_eq_zero (fun i hi => by
        simp [hXdef, hinddef, Set.indicator_of_notMem (Finset.mem_filter.1 hi).2])
    rw [h1, h0, add_zero]
  refine le_trans (le_trans (measureReal_mono ?_ (measure_ne_top _ _)) hmain) ?_
  · intro c hc
    simp only [Set.mem_setOf_eq] at hc ⊢
    rw [hcount c, Finset.card_range]
    linarith [hc]
  · rw [Finset.card_range]

open scoped Classical in
/-- The upper-tail twin: a set of mass at most `q` is hit at most `m(q + t)` times.  This
is what prices the certification prefixes the family flips too much of, whose set is chosen
by the noise while the draws are not. -/
lemma cert_hits_upper (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    (j : J) (m : ℕ) (W : Set S) (q t : ℝ) (ht : 0 ≤ t)
    (hW : (D j).real W ≤ q) :
    (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j).real
        {c | (m : ℝ) * (q + t)
          ≤ (((Finset.range m).filter (fun i => c j i ∈ W)).card : ℝ)}
      ≤ Real.exp (-2 * (m : ℝ) * t ^ 2) := by
  classical
  set ν : Measure (J → ℕ → S) := Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j with hνdef
  have hWm : MeasurableSet W := measurableSet_of_countable W
  set ind : S → ℝ := W.indicator 1 with hinddef
  have hindm : Measurable ind := measurable_const.indicator hWm
  set X : ℕ → (J → ℕ → S) → ℝ := fun i c => ind (c j i) with hXdef
  have hevalj : MeasurePreserving (fun c : J → ℕ → S => c j) ν
      (Measure.infinitePi fun _ : ℕ => D j) :=
    measurePreserving_eval (fun j : J => Measure.infinitePi fun _ : ℕ => D j) j
  have hcoord : iIndepFun (fun (i : ℕ) (s : ℕ → S) => s i)
      (Measure.infinitePi fun _ : ℕ => D j) := iIndepFun_infinitePi (fun _ => measurable_id)
  have hindep : iIndepFun X ν :=
    (iIndepFun_comp_measurePreserving hevalj (fun _ => measurable_pi_apply _) hcoord).comp
      (fun _ => ind) (fun _ => hindm)
  have hmeas : ∀ i, AEMeasurable (X i) ν := fun i =>
    (hindm.comp ((measurable_pi_apply _).comp (measurable_pi_apply _))).aemeasurable
  have hicc : ∀ i, ∀ᵐ c ∂ν, X i c ∈ Set.Icc (0 : ℝ) 1 := by
    intro i
    filter_upwards with c
    by_cases h : c j i ∈ W
    · simp [hXdef, hinddef, Set.indicator_of_mem h]
    · simp [hXdef, hinddef, Set.indicator_of_notMem h]
  have hmean : ∀ i, ν[X i] = (D j).real W := by
    intro i
    have hmp : MeasurePreserving (fun c : J → ℕ → S => c j i) ν (D j) :=
      (measurePreserving_eval_infinitePi _ i).comp hevalj
    calc ν[X i] = ∫ s, ind s ∂(D j) := by
          rw [← hmp.map_eq,
            integral_map hmp.measurable.aemeasurable hindm.aestronglyMeasurable]
      _ = (D j).real W := by rw [hinddef, integral_indicator_one hWm]
  have hsum : ∑ i ∈ Finset.range m, ν[X i] ≤ ((Finset.range m).card : ℝ) * q := by
    rw [Finset.sum_congr rfl (fun i _ => hmean i), Finset.sum_const, nsmul_eq_mul,
      Finset.card_range]
    have : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
    nlinarith [hW]
  have hmain := sumUpper_le X (Finset.range m) q t hmeas hindep hicc hsum ht
  have hcount : ∀ c : J → ℕ → S, ∑ i ∈ Finset.range m, X i c
      = (((Finset.range m).filter (fun i => c j i ∈ W)).card : ℝ) := by
    intro c
    rw [← Finset.sum_filter_add_sum_filter_not (Finset.range m) (fun i => c j i ∈ W)]
    have h1 : ∑ i ∈ (Finset.range m).filter (fun i => c j i ∈ W), X i c
        = (((Finset.range m).filter (fun i => c j i ∈ W)).card : ℝ) := by
      have hone : ∀ i ∈ (Finset.range m).filter (fun i => c j i ∈ W), X i c = (1 : ℝ) := by
        intro i hi
        simp [hXdef, hinddef, Set.indicator_of_mem (Finset.mem_filter.1 hi).2]
      rw [Finset.sum_congr rfl hone, Finset.sum_const, nsmul_eq_mul, mul_one]
    have h0 : ∑ i ∈ (Finset.range m).filter (fun i => ¬ (c j i ∈ W)), X i c = 0 :=
      Finset.sum_eq_zero (fun i hi => by
        simp [hXdef, hinddef, Set.indicator_of_notMem (Finset.mem_filter.1 hi).2])
    rw [h1, h0, add_zero]
  refine le_trans (le_trans (measureReal_mono ?_ (measure_ne_top _ _)) hmain) ?_
  · intro c hc
    simp only [Set.mem_setOf_eq] at hc ⊢
    rw [hcount c, Finset.card_range]
    linarith [hc]
  · rw [Finset.card_range]

/-! ### The accept-preserving gate

`AcceptPreservingGate` runs after the FNR test, right before the family is returned.  It
splits the prefixes by the family's *own* cut and counts, on the seed's own column, how
many read as accepting — membership of `p · ε` is membership of `p`, which is why the gate
is read off `ε` and why `one_mem_clusterAround` matters.  A family is admitted only when
each side reads as its own class (`drift_verdict`).

The cutoffs are `B.lo` and `B.hi`, and the binomial nulls are the rates they cut at,
`lo/k` and `hi/k`, and `PassableAt` is where the budgets they have to keep are written
out. -/

open scoped Classical in
/-- The miscut count read off the cut's two sides. -/
noncomputable def miscutOf (O : Oracle μ S) (A Dset : Finset S) : ℕ :=
  (A.filter (fun p => ¬ (O.label p = 1))).card
    + ((Dset \ A).filter (fun p => ¬ (O.label p = 0))).card

open scoped Classical in
/-- The per-prefix agreement indicator: the seed's read where the cut accepts, its
complement where the cut rejects.  Its mean is `1 − r` where the cut is right and `r` where
it is wrong, `r` the prefix's rate. -/
noncomputable def agreeVar (O : Oracle μ S) (A : Finset S) (p : S) (ω : Ω) : ℝ :=
  if p ∈ A then O.mq p ω else 1 - O.mq p ω

lemma agreeVar_meas (O : Oracle μ S) (A : Finset S) (p : S) :
    AEMeasurable (agreeVar O A p) μ := by
  classical
  by_cases h : p ∈ A
  · have hfun : agreeVar O A p = O.mq p := by funext ω; simp [agreeVar, h]
    rw [hfun]
    exact (mq_meas O p).aemeasurable
  · have hfun : agreeVar O A p = fun ω => 1 - O.mq p ω := by funext ω; simp [agreeVar, h]
    rw [hfun]
    exact (measurable_const.sub (mq_meas O p)).aemeasurable

lemma agreeVar_indep (O : Oracle μ S) (A : Finset S) :
    iIndepFun (fun p => agreeVar O A p) μ := by
  classical
  have hg : ∀ p : S, Measurable (fun x : ℝ => if p ∈ A then x else 1 - x) := by
    intro p
    by_cases h : p ∈ A
    · simp only [h, if_true]
      exact measurable_id'
    · simp only [h, if_false]
      exact measurable_const.sub measurable_id'
  exact (mq_indep O).comp (fun p x => if p ∈ A then x else 1 - x) hg

lemma agreeVar_icc (O : Oracle μ S) (A : Finset S) (p : S) :
    ∀ᵐ ω ∂μ, agreeVar O A p ω ∈ Set.Icc (0 : ℝ) 1 := by
  classical
  filter_upwards [mq_icc O p] with ω hω
  rw [agreeVar]
  by_cases h : p ∈ A
  · simpa [h] using hω
  · simp only [h, if_false]
    exact ⟨by linarith [hω.2], by linarith [hω.1]⟩

open scoped Classical in
/-- The agreement count is the sum of the per-prefix indicators. -/
lemma agreeOf_eq_sum (O : Oracle μ S) (A Dset : Finset S) (hAD : A ⊆ Dset) :
    ∀ᵐ ω ∂μ, ((agreeOf A Dset (Dset.filter (fun p => O.mq p ω = 1)) : ℕ) : ℝ)
      = ∑ p ∈ Dset, agreeVar O A p ω := by
  classical
  filter_upwards [hits_eq_sum O A, hits_eq_sum O (Dset \ A)] with ω hA hR
  have hsplit : ∑ p ∈ Dset \ A, agreeVar O A p ω + ∑ p ∈ A, agreeVar O A p ω
      = ∑ p ∈ Dset, agreeVar O A p ω := Finset.sum_sdiff hAD
  have hAY : ∑ p ∈ A, agreeVar O A p ω = ∑ p ∈ A, O.mq p ω :=
    Finset.sum_congr rfl (fun p hp => by simp [agreeVar, hp])
  have hRY : ∑ p ∈ Dset \ A, agreeVar O A p ω
      = ((Dset \ A).card : ℝ) - ∑ p ∈ Dset \ A, O.mq p ω := by
    rw [Finset.sum_congr rfl (fun p hp => by
      simp [agreeVar, (Finset.mem_sdiff.1 hp).2] :
        ∀ p ∈ Dset \ A, agreeVar O A p ω = 1 - O.mq p ω)]
    rw [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul, mul_one]
  have hAcap : A.filter (fun p => O.mq p ω = 1)
      = A ∩ Dset.filter (fun p => O.mq p ω = 1) := by
    ext q
    constructor
    · intro hq
      obtain ⟨hqA, hq1⟩ := Finset.mem_filter.1 hq
      exact Finset.mem_inter.2 ⟨hqA, Finset.mem_filter.2 ⟨hAD hqA, hq1⟩⟩
    · intro hq
      obtain ⟨hqA, hqD⟩ := Finset.mem_inter.1 hq
      exact Finset.mem_filter.2 ⟨hqA, (Finset.mem_filter.1 hqD).2⟩
  have hRsdiff : (Dset \ A).filter (fun p => ¬ O.mq p ω = 1)
      = (Dset \ A) \ Dset.filter (fun p => O.mq p ω = 1) := by
    ext q
    constructor
    · intro hq
      obtain ⟨hqR, hne⟩ := Finset.mem_filter.1 hq
      exact Finset.mem_sdiff.2 ⟨hqR, fun hc => hne (Finset.mem_filter.1 hc).2⟩
    · intro hq
      obtain ⟨hqR, hne⟩ := Finset.mem_sdiff.1 hq
      refine Finset.mem_filter.2 ⟨hqR, fun hc => hne ?_⟩
      exact Finset.mem_filter.2 ⟨(Finset.mem_sdiff.1 hqR).1, hc⟩
  have hRcount : ((Dset \ A).card : ℝ) - ∑ p ∈ Dset \ A, O.mq p ω
      = ((((Dset \ A) \ Dset.filter (fun p => O.mq p ω = 1)).card : ℕ) : ℝ) := by
    rw [← hR, ← hRsdiff]
    have hc : (((Dset \ A).filter (fun p => O.mq p ω = 1)).card : ℝ)
        + (((Dset \ A).filter (fun p => ¬ O.mq p ω = 1)).card : ℝ)
        = ((Dset \ A).card : ℝ) := by
      exact_mod_cast Finset.card_filter_add_card_filter_not (s := Dset \ A)
        (fun p => O.mq p ω = 1)
    linarith
  rw [← hsplit, hAY, hRY, hRcount, ← hA, hAcap, agreeOf]
  push_cast
  ring

open scoped Classical in
/-- The agreement statistic's mean: at least `1 − η` per decided prefix the cut gets right.  A
wrong one is charged its whole `1 − η`, since at a rate near `0` it reads as agreeing almost
never. -/
lemma agree_mean_ge (O : Oracle μ S) (A Dset : Finset S) (hAD : A ⊆ Dset) :
    (Dset.card : ℝ) * (1 - O.η) - (1 - O.η) * ((miscutOf O A Dset : ℕ) : ℝ)
      ≤ ∑ p ∈ Dset, μ[agreeVar O A p] := by
  classical
  have hnn : ∀ p, 0 ≤ μ[agreeVar O A p] := fun p =>
    integral_nonneg_of_ae ((agreeVar_icc O A p).mono fun ω h => h.1)
  have hA : (1 - O.η) * ((A.filter (fun p => O.label p = 1)).card : ℝ)
      ≤ ∑ p ∈ A, μ[agreeVar O A p] := by
    refine le_trans ?_ (Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset (fun p => O.label p = 1) A)
      (fun p _ _ => hnn p))
    rw [mul_comm, ← nsmul_eq_mul, ← Finset.sum_const]
    refine Finset.sum_le_sum (fun p hp => ?_)
    obtain ⟨hpA, hl⟩ := Finset.mem_filter.1 hp
    have hfun : agreeVar O A p = O.mq p := by funext ω; simp [agreeVar, hpA]
    rw [hfun, mq_mean, hl]
    linarith [O.rate_le_eta p]
  have hR : (1 - O.η) * (((Dset \ A).filter (fun p => O.label p = 0)).card : ℝ)
      ≤ ∑ p ∈ Dset \ A, μ[agreeVar O A p] := by
    refine le_trans ?_ (Finset.sum_le_sum_of_subset_of_nonneg
      (Finset.filter_subset (fun p => O.label p = 0) (Dset \ A)) (fun p _ _ => hnn p))
    rw [mul_comm, ← nsmul_eq_mul, ← Finset.sum_const]
    refine Finset.sum_le_sum (fun p hp => ?_)
    obtain ⟨hpR, hl⟩ := Finset.mem_filter.1 hp
    have hnot : p ∉ A := (Finset.mem_sdiff.1 hpR).2
    have hfun : agreeVar O A p = fun ω => 1 - O.mq p ω := by funext ω; simp [agreeVar, hnot]
    rw [hfun, integral_sub (integrable_const 1) (mq_integrable O p), integral_const,
      mq_mean O p, hl]
    simp only [measureReal_def, measure_univ, ENNReal.toReal_one, smul_eq_mul, one_mul]
    linarith [O.rate_le_eta p]
  have hcA : ((A.filter (fun p => O.label p = 1)).card : ℝ)
      + ((A.filter (fun p => ¬ (O.label p = 1))).card : ℝ) = (A.card : ℝ) := by
    exact_mod_cast Finset.card_filter_add_card_filter_not (s := A) (fun p => O.label p = 1)
  have hcR : (((Dset \ A).filter (fun p => O.label p = 0)).card : ℝ)
      + (((Dset \ A).filter (fun p => ¬ (O.label p = 0))).card : ℝ) = ((Dset \ A).card : ℝ) := by
    exact_mod_cast Finset.card_filter_add_card_filter_not (s := Dset \ A)
      (fun p => O.label p = 0)
  have hcards : ((Dset \ A).card : ℝ) + (A.card : ℝ) = (Dset.card : ℝ) := by
    exact_mod_cast Finset.card_sdiff_add_card_eq_card hAD
  have hsplit : ∑ p ∈ Dset \ A, μ[agreeVar O A p] + ∑ p ∈ A, μ[agreeVar O A p]
      = ∑ p ∈ Dset, μ[agreeVar O A p] := Finset.sum_sdiff hAD
  have hmis : ((miscutOf O A Dset : ℕ) : ℝ)
      = ((A.filter (fun p => ¬ (O.label p = 1))).card : ℝ)
        + (((Dset \ A).filter (fun p => ¬ (O.label p = 0))).card : ℝ) := by
    rw [miscutOf]; push_cast; ring
  have hid : (1 - O.η) * ((A.filter (fun p => O.label p = 1)).card : ℝ)
      + (1 - O.η) * (((Dset \ A).filter (fun p => O.label p = 0)).card : ℝ)
      = (Dset.card : ℝ) * (1 - O.η) - (1 - O.η) * ((miscutOf O A Dset : ℕ) : ℝ) := by
    rw [hmis, ← hcards, ← hcA, ← hcR]; ring
  linarith [hA, hR, hid, hsplit]

/-- A cut that is mostly right reads as agreeing often enough.  Unlike the two
side-wise tests this replaces, the denominator is the whole decided set, so the bound does
not degrade when one side of the cut is small. -/
lemma agree_sound_of_wrong (O : Oracle μ S) (A Dset : Finset S) (hAD : A ⊆ Dset)
    (θ τ w : ℝ) (hτ : 0 ≤ τ) (hsig : O.η ≤ 1 / 2)
    (hw : ((miscutOf O A Dset : ℕ) : ℝ) ≤ w)
    (hθ : (Dset.card : ℝ) * (θ + τ)
      ≤ (Dset.card : ℝ) * (1 - O.η) - (1 - O.η) * w) :
    μ.real {ω | ((agreeOf A Dset (Dset.filter (fun p => O.mq p ω = 1)) : ℕ) : ℝ)
        ≤ (Dset.card : ℝ) * θ}
      ≤ Real.exp (-2 * (Dset.card : ℝ) * τ ^ 2) := by
  classical
  have h1η : (0 : ℝ) ≤ 1 - O.η := by linarith
  have hmean : (Dset.card : ℝ) * (θ + τ) ≤ ∑ p ∈ Dset, μ[agreeVar O A p] := by
    nlinarith [agree_mean_ge O A Dset hAD, mul_le_mul_of_nonneg_left hw h1η]
  have hsum := sumLower_le (fun p => agreeVar O A p) Dset (θ + τ) τ
    (agreeVar_meas O A) (agreeVar_indep O A) (agreeVar_icc O A) hmean hτ
  have hsub : {ω | ((agreeOf A Dset (Dset.filter (fun p => O.mq p ω = 1)) : ℕ) : ℝ)
        ≤ (Dset.card : ℝ) * θ}
      ≤ᵐ[μ] {ω | ∑ p ∈ Dset, agreeVar O A p ω ≤ (Dset.card : ℝ) * ((θ + τ) - τ)} := by
    filter_upwards [agreeOf_eq_sum O A Dset hAD] with ω hω hmem
    change ∑ p ∈ Dset, agreeVar O A p ω ≤ (Dset.card : ℝ) * ((θ + τ) - τ)
    have hrw : (Dset.card : ℝ) * ((θ + τ) - τ) = (Dset.card : ℝ) * θ := by ring
    rw [hrw, ← hω]
    exact hmem
  exact le_trans (ENNReal.toReal_mono (measure_ne_top _ _) (measure_mono_ae hsub)) hsum

open scoped Classical in
/-- The agreement count only reads the decided prefixes, so it does not matter whether the
hit set is taken over the sample or over the decided part of it. -/
lemma agreeOf_filter_of_subset (O : Oracle μ S) (A Dset C : Finset S) (hAD : A ⊆ Dset)
    (hDC : Dset ⊆ C) (ω : Ω) :
    agreeOf A Dset (C.filter (fun p => O.mq p ω = 1))
      = agreeOf A Dset (Dset.filter (fun p => O.mq p ω = 1)) := by
  classical
  have h1 : A ∩ C.filter (fun p => O.mq p ω = 1)
      = A ∩ Dset.filter (fun p => O.mq p ω = 1) := by
    ext q
    constructor
    · intro hq
      obtain ⟨hqA, hqC⟩ := Finset.mem_inter.1 hq
      exact Finset.mem_inter.2 ⟨hqA,
        Finset.mem_filter.2 ⟨hAD hqA, (Finset.mem_filter.1 hqC).2⟩⟩
    · intro hq
      obtain ⟨hqA, hqD⟩ := Finset.mem_inter.1 hq
      exact Finset.mem_inter.2 ⟨hqA,
        Finset.mem_filter.2 ⟨hDC (hAD hqA), (Finset.mem_filter.1 hqD).2⟩⟩
  have h2 : (Dset \ A) \ C.filter (fun p => O.mq p ω = 1)
      = (Dset \ A) \ Dset.filter (fun p => O.mq p ω = 1) := by
    ext q
    constructor
    · intro hq
      obtain ⟨hqR, hne⟩ := Finset.mem_sdiff.1 hq
      refine Finset.mem_sdiff.2 ⟨hqR, fun hc => hne ?_⟩
      exact Finset.mem_filter.2 ⟨hDC (Finset.mem_filter.1 hc).1, (Finset.mem_filter.1 hc).2⟩
    · intro hq
      obtain ⟨hqR, hne⟩ := Finset.mem_sdiff.1 hq
      refine Finset.mem_sdiff.2 ⟨hqR, fun hc => hne ?_⟩
      exact Finset.mem_filter.2 ⟨(Finset.mem_sdiff.1 hqR).1, (Finset.mem_filter.1 hc).2⟩
  rw [agreeOf, agreeOf, h1, h2]

open scoped Classical in
/-- The gate's bound, with the cut chosen by the votes.  `agree_sound_of_wrong` prices a
*fixed* cut; this prices the cut the run produces, by conditioning on the votes — which read
`Q`, disjoint from the prefixes the statistic scores. -/
theorem gate_agree_bound (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    (lo hi : ℕ) (fam : Ω → Finset S)
    (hQ : ∀ ω, ∀ p ∈ C, ∀ v ∈ fam ω, p * v ∈ Q)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (θ τ w : ℝ) (n₀ : ℕ) (hτ : 0 ≤ τ) (hsig : O.η ≤ 1 / 2)
    (hθ : ∀ n : ℕ, n₀ ≤ n → n ≤ C.card →
      (n : ℝ) * (θ + τ) ≤ (n : ℝ) * (1 - O.η) - (1 - O.η) * w) :
    μ.real {ω | n₀ ≤ (cutSides O.mq lo hi (fam ω) C ω).2.card
        ∧ ((miscutOf O (cutSides O.mq lo hi (fam ω) C ω).1
              (cutSides O.mq lo hi (fam ω) C ω).2 : ℕ) : ℝ) ≤ w
        ∧ ((agreeOf (cutSides O.mq lo hi (fam ω) C ω).1 (cutSides O.mq lo hi (fam ω) C ω).2
              (C.filter (fun p => O.mq p ω = 1)) : ℕ) : ℝ)
            ≤ ((cutSides O.mq lo hi (fam ω) C ω).2.card : ℝ) * θ}
      ≤ Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := by
  classical
  have hvc : ∀ (ω ω' : Ω), (∀ w ∈ Q, O.noise w ω = O.noise w ω') → ∀ p ∈ C,
      voteCount O.mq (fam ω) p ω = voteCount O.mq (fam ω') p ω' := by
    intro ω ω' h p hp
    rw [← hcongr ω ω' h]
    exact voteCount_congr O _ p (fun v hv => by rw [mq_congr O (h _ (hQ ω p hp v hv))])
  have hselcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') →
      cutSides O.mq lo hi (fam ω) C ω = cutSides O.mq lo hi (fam ω') C ω' := by
    intro ω ω' h
    rw [cutSides, cutSides]
    exact Prod.ext (Finset.filter_congr (fun p hp => by rw [hvc ω ω' h p hp]))
      (Finset.filter_congr (fun p hp => by rw [hvc ω ω' h p hp]))
  set Pr : (Finset S × Finset S) → Finset S → Prop := fun t U =>
    t.1 ⊆ t.2 ∧ n₀ ≤ t.2.card
      ∧ ((miscutOf O t.1 t.2 : ℕ) : ℝ) ≤ w
      ∧ ((agreeOf t.1 t.2 U : ℕ) : ℝ) ≤ (t.2.card : ℝ) * θ with hPr
  have hsel : ∀ ω, cutSides O.mq lo hi (fam ω) C ω ∈ C.powerset ×ˢ C.powerset := fun ω =>
    Finset.mem_product.2 ⟨Finset.mem_powerset.2 (Finset.filter_subset _ _),
      Finset.mem_powerset.2 (Finset.filter_subset _ _)⟩
  have hAD : ∀ ω, (cutSides O.mq lo hi (fam ω) C ω).1 ⊆ (cutSides O.mq lo hi (fam ω) C ω).2 := by
    intro ω q hq
    exact Finset.mem_filter.2 ⟨(Finset.mem_filter.1 hq).1, Or.inl (Finset.mem_filter.1 hq).2⟩
  have hbad : ∀ t ∈ C.powerset ×ˢ C.powerset,
      μ.real {ω | Pr t (C.filter (fun p => O.mq p ω = 1))}
        ≤ Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) := by
    rintro ⟨A, Dset⟩ hmem
    by_cases hADt : A ⊆ Dset
    · by_cases hn : n₀ ≤ Dset.card
      · by_cases hwc : ((miscutOf O A Dset : ℕ) : ℝ) ≤ w
        · have hDC : Dset ⊆ C := Finset.mem_powerset.1 (Finset.mem_product.1 hmem).2
          have hDcard : Dset.card ≤ C.card := Finset.card_le_card hDC
          have hsub : {ω | Pr (A, Dset) (C.filter (fun p => O.mq p ω = 1))}
              ⊆ {ω | ((agreeOf A Dset (Dset.filter (fun p => O.mq p ω = 1)) : ℕ) : ℝ)
                    ≤ (Dset.card : ℝ) * θ} := by
            intro ω hω
            have h4 := hω.2.2.2
            rwa [agreeOf_filter_of_subset O A Dset C hADt hDC ω] at h4
          refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
          refine le_trans (agree_sound_of_wrong O A Dset hADt θ τ w hτ hsig hwc
            (hθ Dset.card hn hDcard)) ?_
          refine Real.exp_le_exp.2 ?_
          have hc : (n₀ : ℝ) ≤ (Dset.card : ℝ) := by exact_mod_cast hn
          nlinarith [sq_nonneg τ]
        · have hz : {ω | Pr (A, Dset) (C.filter (fun p => O.mq p ω = 1))} = (∅ : Set Ω) := by
            ext ω; simp [hPr, hwc]
          rw [hz]; simpa using Real.exp_nonneg _
      · have hz : {ω | Pr (A, Dset) (C.filter (fun p => O.mq p ω = 1))} = (∅ : Set Ω) := by
          ext ω; simp [hPr, hn]
        rw [hz]; simpa using Real.exp_nonneg _
    · have hz : {ω | Pr (A, Dset) (C.filter (fun p => O.mq p ω = 1))} = (∅ : Set Ω) := by
        ext ω; simp [hPr, hADt]
      rw [hz]; simpa using Real.exp_nonneg _
  have hmain := selection_read_bound O C Q hdisj (C.powerset ×ˢ C.powerset) (∅, ∅)
    (Finset.mem_product.2 ⟨Finset.empty_mem_powerset C, Finset.empty_mem_powerset C⟩)
    (fun ω => cutSides O.mq lo hi (fam ω) C ω) hsel hselcongr Pr _ (Real.exp_nonneg _) hbad
  refine le_trans (measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)) hmain
  exact ⟨hAD ω, hω.1, hω.2.1, hω.2.2⟩

open scoped Classical in
/-- Both of the gate's wrong-counts are charged to the same mis-cut set.  A prefix the
cut accepts but the oracle rejects, and one the cut rejects but the oracle accepts, are each
a prefix where the cut is wrong — and the two kinds are disjoint. -/
lemma miscutOf_le_cutWrong (O : Oracle μ S) (lo hi : ℕ) (F C : Finset S) (ω : Ω) :
    miscutOf O (cutSides O.mq lo hi F C ω).1 (cutSides O.mq lo hi F C ω).2
      ≤ (C.filter (fun p => ¬ cutCorrect O lo (hi - 1) F p ω)).card := by
  classical
  have hsubA : ((cutSides O.mq lo hi F C ω).1).filter (fun p => ¬ (O.label p = 1))
      ⊆ C.filter (fun p => ¬ cutCorrect O lo (hi - 1) F p ω) := by
    intro q hq
    obtain ⟨hqA, hlab⟩ := Finset.mem_filter.1 hq
    obtain ⟨hqC, hv⟩ := Finset.mem_filter.1 hqA
    exact Finset.mem_filter.2 ⟨hqC, fun hcc => hlab (hcc.1 hv)⟩
  have hsubR : (((cutSides O.mq lo hi F C ω).2 \ (cutSides O.mq lo hi F C ω).1).filter
        (fun p => ¬ (O.label p = 0)))
      ⊆ C.filter (fun p => ¬ cutCorrect O lo (hi - 1) F p ω) := by
    intro q hq
    obtain ⟨hqR, hlab⟩ := Finset.mem_filter.1 hq
    obtain ⟨hqD, hqA⟩ := Finset.mem_sdiff.1 hqR
    obtain ⟨hqC, hv⟩ := Finset.mem_filter.1 hqD
    have hnot : ¬ (hi - 1 < voteCount O.mq F q ω) := fun h =>
      hqA (Finset.mem_filter.2 ⟨hqC, h⟩)
    exact Finset.mem_filter.2 ⟨hqC, fun hcc => hlab (hcc.2 (hv.resolve_left hnot))⟩
  have hdj : Disjoint (((cutSides O.mq lo hi F C ω).1).filter (fun p => ¬ (O.label p = 1)))
      (((cutSides O.mq lo hi F C ω).2 \ (cutSides O.mq lo hi F C ω).1).filter
        (fun p => ¬ (O.label p = 0))) := by
    refine Finset.disjoint_left.2 (fun q hq hq' => ?_)
    exact (Finset.mem_sdiff.1 (Finset.mem_filter.1 hq').1).2 (Finset.mem_filter.1 hq).1
  rw [miscutOf, ← Finset.card_union_of_disjoint hdj]
  exact Finset.card_le_card (Finset.union_subset hsubA hsubR)

/-- The family at a reachable state is invalid: on some population its cut is wrong on
more than an `εcov` fraction. -/
def FailAt (O : Oracle μ S) (populations : Finset J) (D : J → Measure S) (εcov : ℝ)
    (B : State) : Set (Run Ω S J) :=
  {x | ¬ ∀ j ∈ populations, 1 - εcov
        ≤ (D j).real {p | cutCorrect O B.lo B.hi (clusterAt O.mq populations x B) p (oracleNoise x)}}

lemma voteCount_mono (O : Oracle μ S) {F F' : Finset S} (h : F ⊆ F') (p : S) (ω : Ω) :
    voteCount O.mq F p ω ≤ voteCount O.mq F' p ω := by
  classical
  exact Finset.card_le_card (Finset.filter_subset_filter _ h)

/-- Dropping the seed moves the count by at most one, so the seed-dropped family read with
its accept threshold one lower is right wherever the full family decides. -/
lemma cutCorrect_of_erase (O : Oracle μ S) (lo hi : ℕ) (hhi : 1 ≤ hi) (F : Finset S) (p : S)
    (ω : Ω) (h : cutCorrect O lo (hi - 1) (F.erase 1) p ω) : cutCorrect O lo hi F p ω := by
  obtain ⟨hacc, hrej⟩ := h
  refine ⟨fun hlt => hacc ?_,
    fun hle => hrej (le_trans (voteCount_mono O (Finset.erase_subset _ _) p ω) hle)⟩
  have hstep := voteCount_le_erase_succ O F p ω
  omega

/-- Hoeffding's bound on the binomial upper tail, as a fact about `binomSfGe`.  Validity
needs the tails to force the counts; termination needs the counts to force the tails, which
is this direction. -/
theorem binomSfGe_le (n j : ℕ) (θ τ : ℝ) (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (hτ : 0 ≤ τ)
    (h : (n : ℝ) * (θ + τ) ≤ j) :
    binomSfGe n θ j ≤ Real.exp (-2 * (n : ℝ) * τ ^ 2) := by
  classical
  set p : unitInterval := ⟨θ, hθ0, hθ1⟩ with hp
  have hbin : binomSfGe n θ j = (ProbabilityTheory.binomial n p).real ↑(Finset.Icc j n) := by
    rw [binomial_real_finset]
    rfl
  rw [hbin]
  refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) (binomial_real_ge_le n p τ hτ)
  intro i hi
  simp only [Finset.coe_Icc, Set.mem_Icc] at hi
  have : (j : ℝ) ≤ (i : ℝ) := by exact_mod_cast hi.1
  exact le_trans h this

open scoped Classical in
/-- A mostly-correct cut on sides that carry prefixes is admitted.  The fractional form:
the cut has to be right on all but `w` of the certification sample, not on all of it, and
`miscut_frac_le` is what supplies that `w`.  Both sides draw their wrong-member budget from
the same count, since a side member carrying the wrong label *is* a mis-cut prefix. -/
theorem admitted_whp (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    (lo hi : ℕ) (α τ w wi : ℝ) (n₀ nlo : ℕ) (fam : Ω → Finset S)
    (hQ : ∀ ω, ∀ p ∈ C, ∀ v ∈ fam ω, p * v ∈ Q)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (hτ : 0 ≤ τ) (hsig : O.η ≤ 1 / 2)
    (hnloC : (nlo : ℝ) ≤ (C.card : ℝ) - wi)
    (hga : ∀ n : ℕ, nlo ≤ n → n ≤ C.card →
      (n : ℝ) * (1 / 2 + τ + τ) ≤ (n : ℝ) * (1 - O.η) - (1 - O.η) * w)
    (hα : Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) ≤ α) :
    μ.real {ω | (((C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ) ≤ w)
        ∧ (((C.filter (fun p => ¬ decided O.mq lo (hi - 1) (fam ω) p ω)).card : ℝ) ≤ wi)
        ∧ ¬ admitted O.mq lo hi n₀ α (fam ω) C ω}
      ≤ 2 * Real.exp (-2 * (nlo : ℝ) * τ ^ 2) := by
  classical
  have hsub : {ω | (((C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ)
        ≤ w) ∧ (((C.filter (fun p => ¬ decided O.mq lo (hi - 1) (fam ω) p ω)).card : ℝ) ≤ wi)
        ∧ ¬ admitted O.mq lo hi n₀ α (fam ω) C ω}
      ⊆ {ω | nlo ≤ (cutSides O.mq lo hi (fam ω) C ω).2.card
          ∧ ((miscutOf O (cutSides O.mq lo hi (fam ω) C ω).1
                (cutSides O.mq lo hi (fam ω) C ω).2 : ℕ) : ℝ) ≤ w
          ∧ ((agreeOf (cutSides O.mq lo hi (fam ω) C ω).1 (cutSides O.mq lo hi (fam ω) C ω).2
                (C.filter (fun p => O.mq p ω = 1)) : ℕ) : ℝ)
              ≤ ((cutSides O.mq lo hi (fam ω) C ω).2.card : ℝ) * (1 / 2 + τ)} := by
    rintro ω ⟨hw, hwi, hadm⟩
    have hdec : (nlo : ℝ) ≤ ((cutSides O.mq lo hi (fam ω) C ω).2.card : ℝ) := by
      have hcompl := Finset.card_filter_add_card_filter_not (s := C)
        (fun p => decided O.mq lo (hi - 1) (fam ω) p ω)
      have hc : (((C.filter (fun p => decided O.mq lo (hi - 1) (fam ω) p ω)).card : ℝ))
          + (((C.filter (fun p => ¬ decided O.mq lo (hi - 1) (fam ω) p ω)).card : ℝ))
          = (C.card : ℝ) := by exact_mod_cast hcompl
      have heq : (cutSides O.mq lo hi (fam ω) C ω).2
          = C.filter (fun p => decided O.mq lo (hi - 1) (fam ω) p ω) := by
        show C.filter (fun p => hi - 1 < voteCount O.mq (fam ω) p ω
              ∨ voteCount O.mq (fam ω) p ω ≤ lo)
            = C.filter (fun p => decided O.mq lo (hi - 1) (fam ω) p ω)
        exact Finset.filter_congr (fun p _ => Iff.rfl)
      rw [heq]
      linarith
    have hdecN : nlo ≤ (cutSides O.mq lo hi (fam ω) C ω).2.card := by exact_mod_cast hdec
    rw [admitted] at hadm
    push_neg at hadm
    obtain ⟨hn, hbin⟩ := hadm
    refine ⟨hdecN, ?_, ?_⟩
    · refine le_trans ?_ hw
      exact_mod_cast miscutOf_le_cutWrong O lo hi (fam ω) C ω
    · by_contra hc
      push_neg at hc
      refine absurd ?_ (not_le.2 hbin)
      have hcount : ((agreeCount O.mq lo hi (fam ω) C ω).2 : ℝ)
          * (1 / 2 + τ) ≤ ((agreeCount O.mq lo hi (fam ω) C ω).1 : ℝ) := le_of_lt hc
      refine le_trans (binomSfGe_le _ _ _ τ (by norm_num) (by norm_num) hτ hcount) ?_
      refine le_trans (Real.exp_le_exp.2 ?_) hα
      have hnR : (n₀ : ℝ) ≤ ((agreeCount O.mq lo hi (fam ω) C ω).2 : ℝ) := by exact_mod_cast hn
      nlinarith [sq_nonneg τ]
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  have hmain := gate_agree_bound O C Q hdisj lo hi fam hQ hcongr (1 / 2 + τ) τ w nlo
    hτ hsig (by intro n h1 h2; have := hga n h1 h2; linarith)
  linarith [Real.exp_nonneg (-2 * (nlo : ℝ) * τ ^ 2), hmain]

/-- Two independent draws, from possibly different populations, coincide with probability
`∑ₐ D₁{a}·D₂{a}`. -/
lemma prod_diagonal_eq (D₁ D₂ : Measure S) [IsProbabilityMeasure D₁] [IsProbabilityMeasure D₂] :
    (D₁.prod D₂).real {q : S × S | q.1 = q.2} = ∑' a : S, D₁.real {a} * D₂.real {a} := by
  classical
  have hdiag : {q : S × S | q.1 = q.2} = ⋃ a : S, {((a, a) : S × S)} := by
    ext q; simp [Prod.ext_iff, eq_comm]
  have hdisj : Pairwise (Function.onFun Disjoint (fun a : S => ({(a, a)} : Set (S × S)))) := by
    intro a b hab
    simp only [Function.onFun, Set.disjoint_singleton]
    exact fun h => hab (congrArg Prod.fst h)
  rw [measureReal_def, hdiag, measure_iUnion hdisj (fun a => measurableSet_singleton _),
    ENNReal.tsum_toReal_eq (fun a => by
      simp only [← Set.singleton_prod_singleton, Measure.prod_prod]
      exact ENNReal.mul_ne_top (measure_ne_top _ _) (measure_ne_top _ _))]
  exact tsum_congr (fun a => by
    rw [← Set.singleton_prod_singleton, Measure.prod_prod, ENNReal.toReal_mul, measureReal_def,
      measureReal_def])

/-- Draws from two populations collide no more often than within one.  By `ab ≤ (a²+b²)/2`
pointwise, so the certification stream's cross-collisions with the table stream are paid for
by the same `ρ`. -/
lemma cross_collision_le (D₁ D₂ : Measure S) [IsProbabilityMeasure D₁] [IsProbabilityMeasure D₂]
    (ρ : ℝ) (h₁ : collisionMass D₁ ≤ ρ) (h₂ : collisionMass D₂ ≤ ρ)
    (hs₁ : Summable (fun a : S => D₁.real {a} ^ 2)) (hs₂ : Summable (fun a : S => D₂.real {a} ^ 2)) :
    (D₁.prod D₂).real {q : S × S | q.1 = q.2} ≤ ρ := by
  rw [prod_diagonal_eq D₁ D₂]
  have hle : ∀ a : S, D₁.real {a} * D₂.real {a}
      ≤ (D₁.real {a} ^ 2 + D₂.real {a} ^ 2) / 2 := by
    intro a; nlinarith [sq_nonneg (D₁.real {a} - D₂.real {a})]
  have hsum : Summable (fun a : S => (D₁.real {a} ^ 2 + D₂.real {a} ^ 2) / 2) :=
    (hs₁.add hs₂).div_const 2
  have hprod : Summable (fun a : S => D₁.real {a} * D₂.real {a}) :=
    Summable.of_nonneg_of_le
      (fun a => mul_nonneg measureReal_nonneg measureReal_nonneg) hle hsum
  calc ∑' a : S, D₁.real {a} * D₂.real {a}
      ≤ ∑' a : S, (D₁.real {a} ^ 2 + D₂.real {a} ^ 2) / 2 := hprod.tsum_le_tsum hle hsum
    _ = (collisionMass D₁ + collisionMass D₂) / 2 := by
        rw [tsum_div_const, hs₁.tsum_add hs₂, collisionMass, collisionMass]
    _ ≤ ρ := by linarith

open scoped Classical in
/-- Two fixed coordinates of an i.i.d. block coincide with exactly the collision mass. -/
lemma pi_coord_eq (Dj : Measure S) [IsProbabilityMeasure Dj] {m : ℕ} {i i' : Fin m}
    (hii : i ≠ i') :
    (Measure.pi fun _ : Fin m => Dj).real {p : Fin m → S | p i = p i'} = collisionMass Dj := by
  have hset : {p : Fin m → S | p i = p i'}
      = ⋃ a : S, Set.univ.pi (fun z => if z = i ∨ z = i' then ({a} : Set S) else Set.univ) := by
    ext p
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_univ_pi]
    refine ⟨fun h => ⟨p i, fun z => ?_⟩, fun ⟨a, ha⟩ => ?_⟩
    · by_cases hz : z = i ∨ z = i'
      · rw [if_pos hz]
        rcases hz with hz | hz <;> rw [hz] <;> simp [h]
      · rw [if_neg hz]; trivial
    · have h1 := ha i; have h2 := ha i'
      rw [if_pos (Or.inl rfl)] at h1
      rw [if_pos (Or.inr rfl)] at h2
      rw [Set.mem_singleton_iff] at h1 h2
      rw [h1, h2]
  have hdisj : Pairwise (Function.onFun Disjoint (fun a : S =>
      Set.univ.pi (fun z => if z = i ∨ z = i' then ({a} : Set S) else Set.univ))) := by
    intro a b hab
    simp only [Function.onFun, Set.disjoint_left]
    intro p ha hb
    have h1 := ha i; have h2 := hb i
    rw [Set.mem_univ_pi] at ha hb
    have h1' := ha i; have h2' := hb i
    rw [if_pos (Or.inl rfl), Set.mem_singleton_iff] at h1' h2'
    exact hab (h1'.symm.trans h2')
  have hmeas : ∀ a : S, MeasurableSet
      (Set.univ.pi (fun z => if z = i ∨ z = i' then ({a} : Set S) else Set.univ)) := by
    intro a
    refine MeasurableSet.univ_pi (fun z => ?_)
    by_cases hz : z = i ∨ z = i'
    · rw [if_pos hz]; exact measurableSet_singleton _
    · rw [if_neg hz]; exact MeasurableSet.univ
  have hbox : ∀ a : S, (Measure.pi fun _ : Fin m => Dj)
      (Set.univ.pi (fun z => if z = i ∨ z = i' then ({a} : Set S) else Set.univ))
      = Dj {a} * Dj {a} := by
    intro a
    rw [Measure.pi_pi]
    have hfac : ∀ z : Fin m, Dj (if z = i ∨ z = i' then ({a} : Set S) else Set.univ)
        = if z ∈ ({i, i'} : Finset (Fin m)) then Dj {a} else 1 := by
      intro z
      by_cases hz : z = i ∨ z = i'
      · rw [if_pos hz, if_pos (by simpa using hz)]
      · rw [if_neg hz, if_neg (by simpa using hz), measure_univ]
    rw [Finset.prod_congr rfl (fun z _ => hfac z), Finset.prod_ite_mem, Finset.univ_inter,
      Finset.prod_const, Finset.card_pair hii, sq]
  rw [measureReal_def, hset, measure_iUnion hdisj hmeas, collisionMass,
    ENNReal.tsum_toReal_eq (fun a => by
      rw [hbox a]; exact ENNReal.mul_ne_top (measure_ne_top _ _) (measure_ne_top _ _))]
  exact tsum_congr (fun a => by rw [hbox a, ENNReal.toReal_mul, sq, measureReal_def])

open scoped Classical in
/-- The certification block is distinct except for the collision mass.  This is what
the `ρ` premise buys: below this event the `m` draws are distinct strings, so their noise
bits are independent and the gate's binomial null is honest. -/
lemma pi_not_injective_le (Dj : Measure S) [IsProbabilityMeasure Dj] (m : ℕ) (ρ : ℝ)
    (hρ : collisionMass Dj ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (Measure.pi fun _ : Fin m => Dj).real {p : Fin m → S | ¬ Function.Injective p}
      ≤ (m : ℝ) ^ 2 * ρ := by
  classical
  set κ := (Finset.univ : Finset (Fin m × Fin m)).filter (fun q => q.1 ≠ q.2) with hκ
  have hsub : {p : Fin m → S | ¬ Function.Injective p}
      ⊆ ⋃ q ∈ κ, {p : Fin m → S | p q.1 = p q.2} := by
    intro p hp
    simp only [Set.mem_setOf_eq, Function.Injective, not_forall] at hp
    obtain ⟨a, b, hab, hne⟩ := hp
    exact Set.mem_biUnion (show (a, b) ∈ κ by simp [hκ, hne]) hab
  calc (Measure.pi fun _ : Fin m => Dj).real {p : Fin m → S | ¬ Function.Injective p}
      ≤ (Measure.pi fun _ : Fin m => Dj).real (⋃ q ∈ κ, {p : Fin m → S | p q.1 = p q.2}) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ q ∈ κ, (Measure.pi fun _ : Fin m => Dj).real {p : Fin m → S | p q.1 = p q.2} :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _q ∈ κ, ρ := Finset.sum_le_sum (fun q hq => by
        rw [pi_coord_eq Dj (Finset.mem_filter.mp hq).2]; exact hρ)
    _ = (κ.card : ℝ) * ρ := by rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ (m : ℝ) ^ 2 * ρ := by
        refine mul_le_mul_of_nonneg_right ?_ hρ0
        have h1 : κ.card ≤ (Finset.univ : Finset (Fin m × Fin m)).card :=
          Finset.card_le_card (Finset.filter_subset _ _)
        have h2 : (Finset.univ : Finset (Fin m × Fin m)).card = m * m := by simp
        rw [h2] at h1
        calc (κ.card : ℝ) ≤ ((m * m : ℕ) : ℝ) := by exact_mod_cast h1
          _ = (m : ℝ) ^ 2 := by push_cast; ring

lemma stoppable_npref_sum_le (η₀ : ℝ) (populations : Finset J)
    (εcov δ α pAP ρ ρsf : ℝ) :
    ∑ B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP ρ ρsf, (B.npref : ℝ)
      ≤ 2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP : ℝ) := by
  classical
  refine le_trans (Finset.sum_le_sum_of_subset_of_nonneg ?_ (fun _ _ _ => Nat.cast_nonneg _))
    (schedule_npref_sum_le η₀ populations εcov δ α pAP)
  rw [stoppable]
  exact Finset.filter_subset _ _

lemma capped_of_mem_stoppable {η₀ : ℝ} {populations : Finset J}
    {εcov δ α pAP ρ ρsf : ℝ} {B : State}
    (hB : B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP ρ ρsf) :
    Capped η₀ populations indecisionLimit εcov δ ρ ρsf pAP (prefCount η₀ populations
      indecisionLimit εcov δ α pAP) B := by
  classical
  exact (Finset.mem_filter.1 hB).2

open scoped Classical in
/-- The first step's loss at one prefix, in the exact form `hammingLoss` uses. -/
noncomputable def seedLoss (O : Oracle μ S) (cn cd : ℕ) (v p : S) (ω : Ω) : ℝ :=
  if ((O.mq (p * v) ω = 1) ↔ cn * ({(1 : S)} : Finset S).card
      < cd * voteCount O.mq {(1 : S)} p ω) then 0 else 1

lemma seedLoss_icc (O : Oracle μ S) (cn cd : ℕ) (v p : S) :
    ∀ᵐ ω ∂μ, seedLoss O cn cd v p ω ∈ Set.Icc (0 : ℝ) 1 := by
  filter_upwards with ω
  unfold seedLoss
  split_ifs <;> norm_num

lemma measurable_voteCount (O : Oracle μ S) (F : Finset S) (p : S) :
    Measurable (fun ω => voteCount O.mq F p ω) := by
  classical
  have hfun : (fun ω => voteCount O.mq F p ω)
      = fun ω => ∑ v ∈ F, (if O.mq (p * v) ω = 1 then 1 else 0) := by
    funext ω; unfold voteCount; rw [Finset.card_filter]
  rw [hfun]
  exact Finset.measurable_sum _ (fun v _ =>
    Measurable.ite (measurableSet_eq_fun (mq_meas O _) measurable_const)
      measurable_const measurable_const)

lemma seedLoss_meas (O : Oracle μ S) (cn cd : ℕ) (v p : S) :
    Measurable (seedLoss O cn cd v p) := by
  classical
  unfold seedLoss
  refine Measurable.ite ?_ measurable_const measurable_const
  exact MeasurableSet.iff (measurableSet_eq_fun (mq_meas O _) measurable_const)
    (measurableSet_lt measurable_const (measurable_const.mul (measurable_voteCount O _ p)))

/-- On a proper centre ratio the first step's loss is the disagreement with the seed's own
column: `cn < cd` makes `cn·1 < cd·voteCount {ε}` say exactly `O.mq p = 1`. -/
lemma seedLoss_eq_disagree (O : Oracle μ S) {cn cd : ℕ} (hcd : cn < cd) (v p : S) :
    ∀ᵐ ω ∂μ, seedLoss O cn cd v p ω
      = O.mq (p * v) ω + O.mq p ω - 2 * (O.mq (p * v) ω * O.mq p ω) := by
  classical
  filter_upwards [mq_bit O (p * v), mq_bit O p] with ω h1 h0
  have hvc : voteCount O.mq {(1 : S)} p ω = if O.mq p ω = 1 then 1 else 0 := by
    unfold voteCount
    rcases h0 with h | h <;> simp [voteCount, Finset.filter_singleton, mul_one, h]
  unfold seedLoss
  rw [hvc]
  rcases h1 with h1 | h1 <;> rcases h0 with h0 | h0 <;>
    · rw [h1, h0]
      norm_num [hcd, Nat.not_lt.2 (Nat.zero_le cn)]

lemma mq_mul_integrable (O : Oracle μ S) (w w' : S) :
    Integrable (fun ω => O.mq w ω * O.mq w' ω) μ := by
  refine MeasureTheory.Integrable.of_mem_Icc 0 1
    (((mq_meas O w).mul (mq_meas O w')).aemeasurable) ?_
  filter_upwards [mq_icc O w, mq_icc O w'] with ω h1 h0
  rw [Set.mem_Icc] at h1 h0 ⊢
  exact ⟨mul_nonneg h1.1 h0.1, by nlinarith [h1.1, h1.2, h0.1, h0.2]⟩

/-- The first step's mean separates by the square of the signal.  Two noisy reads are
compared, so an accept-preserving candidate disagrees at `2r(1−r)` for the prefix's rate `r`,
and one that flips at `p` at `(1 − r − r')(1 − 2r)` more, `r'` the other class's rate — at
least `(1−2η)²`.  This is exactly the statistic `_screen_cohort` tests against, and the square
is why its power is weaker than a comparison against the truth would be. -/
lemma seedLoss_mean (O : Oracle μ S) {cn cd : ℕ} (hcd : cn < cd) (v p : S) (hv : p * v ≠ p)
    (hsig : O.η ≤ 1 / 2) :
    2 * O.rate p * (1 - O.rate p) + O.flip v p * (1 - 2 * O.η) ^ 2 ≤ μ[seedLoss O cn cd v p]
      ∧ (O.flip v p = 0 → μ[seedLoss O cn cd v p] = 2 * O.rate p * (1 - O.rate p)) := by
  have hprod : μ[fun ω => O.mq (p * v) ω * O.mq p ω] = μ[O.mq (p * v)] * μ[O.mq p] :=
    ProbabilityTheory.IndepFun.integral_mul_eq_mul_integral
      ((mq_indep O).indepFun hv) (mq_meas O _).aestronglyMeasurable
      (mq_meas O _).aestronglyMeasurable
  have hsplit : μ[seedLoss O cn cd v p]
      = μ[O.mq (p * v)] + μ[O.mq p] - 2 * (μ[O.mq (p * v)] * μ[O.mq p]) := by
    rw [integral_congr_ae (seedLoss_eq_disagree O hcd v p),
      integral_sub (f := fun ω => O.mq (p * v) ω + O.mq p ω)
        (g := fun ω => 2 * (O.mq (p * v) ω * O.mq p ω))
        ((mq_integrable O _).add (mq_integrable O _))
        ((mq_mul_integrable O (p * v) p).const_mul 2),
      integral_add (mq_integrable O _) (mq_integrable O _), integral_const_mul, hprod]
  have hflip : O.flip v p = O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p := rfl
  have hrx0 := O.rate_nonneg (p * v)
  have hrx1 := O.rate_le_eta (p * v)
  have hry0 := O.rate_nonneg p
  have hry1 := O.rate_le_eta p
  have key : (1 - 2 * O.η) ^ 2 ≤ (1 - O.rate (p * v) - O.rate p) * (1 - 2 * O.rate p) := by
    rw [sq]
    exact mul_le_mul (by linarith) (by linarith) (by linarith) (by linarith)
  rw [hsplit, mq_mean, mq_mean, hflip]
  rcases O.label_bit (p * v) with hx | hx <;> rcases O.label_bit p with hy | hy
  · rw [O.rate_eq_of_label_eq (hx.trans hy.symm), hx, hy]
    exact ⟨le_of_eq (by ring), fun _ => by ring⟩
  · rw [hx, hy]
    exact ⟨by nlinarith [key], fun h => by norm_num at h⟩
  · rw [hx, hy]
    exact ⟨by nlinarith [key], fun h => by norm_num at h⟩
  · rw [O.rate_eq_of_label_eq (hx.trans hy.symm), hx, hy]
    exact ⟨le_of_eq (by ring), fun _ => by ring⟩

/-- A clean candidate's mean screen count on the table `P`. -/
noncomputable def cleanLoss (O : Oracle μ S) (P : Finset S) : ℝ :=
  ∑ p ∈ P, 2 * O.rate p * (1 - O.rate p)

/-- `sumUpper_le` against a total mean rather than a per-index one. -/
lemma sumUpper_le_total {ι : Type*} (X : ι → Ω → ℝ) (idx : Finset ι) (T γ : ℝ)
    (hmeas : ∀ i, AEMeasurable (X i) μ) (h_indep : iIndepFun X μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    (hmean : ∑ i ∈ idx, μ[X i] ≤ T) (hγ : 0 ≤ γ) :
    μ.real {ω | T + (idx.card : ℝ) * γ ≤ ∑ i ∈ idx, X i ω}
      ≤ Real.exp (-2 * (idx.card : ℝ) * γ ^ 2) := by
  rcases Nat.eq_zero_or_pos idx.card with h0 | hpos
  · simp only [h0, Nat.cast_zero, mul_zero, zero_mul, Real.exp_zero]
    exact measureReal_le_one
  have hc : (0 : ℝ) < (idx.card : ℝ) := by exact_mod_cast hpos
  have h := sumUpper_le X idx (T / idx.card) γ hmeas h_indep hIcc
    (by rw [mul_div_cancel₀ _ hc.ne']; exact hmean) hγ
  rwa [mul_add, mul_div_cancel₀ _ hc.ne'] at h

/-- `sumLower_le` against a total mean rather than a per-index one. -/
lemma sumLower_le_total {ι : Type*} (X : ι → Ω → ℝ) (idx : Finset ι) (T γ : ℝ)
    (hmeas : ∀ i, AEMeasurable (X i) μ) (h_indep : iIndepFun X μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    (hmean : T ≤ ∑ i ∈ idx, μ[X i]) (hγ : 0 ≤ γ) :
    μ.real {ω | ∑ i ∈ idx, X i ω ≤ T - (idx.card : ℝ) * γ}
      ≤ Real.exp (-2 * (idx.card : ℝ) * γ ^ 2) := by
  rcases Nat.eq_zero_or_pos idx.card with h0 | hpos
  · simp only [h0, Nat.cast_zero, mul_zero, zero_mul, Real.exp_zero]
    exact measureReal_le_one
  have hc : (0 : ℝ) < (idx.card : ℝ) := by exact_mod_cast hpos
  have h := sumLower_le X idx (T / idx.card) γ hmeas h_indep hIcc
    (by rw [mul_div_cancel₀ _ hc.ne']; exact hmean) hγ
  rwa [mul_sub, mul_div_cancel₀ _ hc.ne'] at h

open scoped Classical in
/-- The screen's statistic is the summed `seedLoss`, exactly rather than almost everywhere:
`voteCount` over the singleton `{ε}` is a card, so it is a bit by construction. -/
lemma screenCount_eq_sum (O : Oracle μ S) {cn cd : ℕ} (hcd : cn < cd) (P : Finset S) (v : S)
    (ω : Ω) : ((screenCount O.mq P v ω : ℝ)) = ∑ p ∈ P, seedLoss O cn cd v p ω := by
  classical
  have hcond : ∀ p : S, (cn * ({(1 : S)} : Finset S).card < cd * voteCount O.mq {(1 : S)} p ω)
      ↔ (O.mq p ω = 1) := by
    intro p
    have hvc : voteCount O.mq {(1 : S)} p ω = if O.mq p ω = 1 then 1 else 0 := by
      unfold voteCount
      by_cases h : O.mq p ω = 1 <;> simp [Finset.filter_singleton, mul_one, h]
    rw [hvc]
    by_cases h : O.mq p ω = 1 <;> simp [h, hcd]
  unfold screenCount seedLoss
  rw [Finset.sum_ite]
  simp only [Finset.sum_const, smul_zero, zero_add, nsmul_eq_mul, mul_one]
  refine congrArg (fun n : ℕ => (n : ℝ)) (congrArg Finset.card ?_)
  ext p
  simp only [Finset.mem_filter, hcond p]

/-- The seed's own reads are independent across prefixes.  Each prefix contributes a
function of two strings, `p` and `p · v`, and those pairs are pairwise disjoint: `p · v` by
right-cancellation, the bare prefixes by distinctness, and the two kinds from each other by
flatness. -/
lemma seedLoss_indep {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S) (cn cd : ℕ)
    {P : Finset S} (hP : ∀ p ∈ P, p ∈ Pre) (v : S) :
    iIndepFun (fun p : {p // p ∈ P} => seedLoss O cn cd v p.val) μ := by
  refine iIndepFun_blocks (X := O.noise) (fun w => O.noise_meas w) O.noise_indep
    (fun p : {p // p ∈ P} => {p.val, p.val * v}) ?_ _ ?_
  · intro a b hab
    refine Finset.disjoint_left.2 (fun w hw hw' => ?_)
    simp only [Finset.mem_insert, Finset.mem_singleton] at hw hw'
    rcases hw with rfl | rfl <;> rcases hw' with hb | hb
    · exact hab (Subtype.ext hb)
    · by_cases hv1 : v = 1
      · exact hab (Subtype.ext (by rw [hb, hv1, mul_one]))
      · exact flat_ne_of_ne_one hflat (hP _ b.property) (hP _ a.property) hv1 hb.symm
    · by_cases hv1 : v = 1
      · exact hab (Subtype.ext (by rw [← hb, hv1, mul_one]))
      · exact flat_ne_of_ne_one hflat (hP _ a.property) (hP _ b.property) hv1 hb
    · exact hab (Subtype.ext (mul_right_cancel hb))
  · intro p
    have hsup : (⨆ w ∈ ({p.val, p.val * v} : Finset S),
        MeasurableSpace.comap (O.noise w) inferInstance)
        = noiseAlg O ↑({p.val, p.val * v} : Finset S) := by
      unfold noiseAlg
      exact iSup_congr (fun w => by simp)
    rw [hsup]
    set T : Set S := ↑({p.val, p.val * v} : Finset S) with hT
    have hA : MeasurableSet[noiseAlg O T] {ω | O.mq (p.val * v) ω = 1} :=
      measurableSet_mq_eq_one O (by simp [hT])
    have hC : MeasurableSet[noiseAlg O T] {ω | cn * ({(1 : S)} : Finset S).card
        < cd * voteCount O.mq {(1 : S)} p.val ω} :=
      measurableSet_filter_pred_map O (T := T) (A := {(1 : S)}) (fun w => p.val * w)
        (by intro w hw; simp only [Finset.mem_singleton] at hw; simp [hw, hT])
        (fun U => cn * ({(1 : S)} : Finset S).card < cd * U.card)
    have hQ : MeasurableSet[noiseAlg O T] {ω | (O.mq (p.val * v) ω = 1)
        ↔ cn * ({(1 : S)} : Finset S).card < cd * voteCount O.mq {(1 : S)} p.val ω} := by
      have hrw : {ω | (O.mq (p.val * v) ω = 1)
          ↔ cn * ({(1 : S)} : Finset S).card < cd * voteCount O.mq {(1 : S)} p.val ω}
          = ({ω | O.mq (p.val * v) ω = 1} ∩ {ω | cn * ({(1 : S)} : Finset S).card
              < cd * voteCount O.mq {(1 : S)} p.val ω})
            ∪ ({ω | O.mq (p.val * v) ω = 1}ᶜ ∩ {ω | cn * ({(1 : S)} : Finset S).card
              < cd * voteCount O.mq {(1 : S)} p.val ω}ᶜ) := by
        ext ω
        by_cases h1 : O.mq (p.val * v) ω = 1 <;>
          by_cases h2 : cn * ({(1 : S)} : Finset S).card
            < cd * voteCount O.mq {(1 : S)} p.val ω <;>
          simp [h1, h2]
      rw [hrw]
      exact ((hA.inter hC).union (hA.compl.inter hC.compl))
    unfold seedLoss
    exact Measurable.ite hQ measurable_const measurable_const

/-! ### The first Lloyd step

Its centre is `{ε}`, so its loss is `seedLoss`, whose mean separates a candidate that never
flips from one that flips on a `Δ` fraction by at least `Δ(1−2η)²` — the screen's
statistic.  The
ranking is *not* what bounds the family's flip mass: `clusterAt_flip_bound` reads that off
the screen, which every candidate has already passed.  What the ranking has to
deliver is only that the seed survives it, and `lloydStep`'s tie-break gives that outright.
-/

/-! ### Why Part 1 does not come from the clustering

Inferring validity from the cluster's loss concentration needs a union bound over every
candidate: taking the best of `M` candidates buys `√(2 log M)` of the loss's spread for free,
so the ranking is decided by luck unless `m ≳ (¼−s²)·log M / (2s²Δ²)`, and the persistent
oracle's fixed bits put the per-candidate error floor at the prefix collision entropy, so
the route fails outright once the pool outgrows `exp(c/ρ)`.

Part 1 reads the family's vote on the certification sample instead, draws the family was
never selected from.  What the clustering still has to deliver is only that the family is
clean (`measureReal_dirtyMember_le`, off the screen) and has the round's size. -/

open scoped Classical in
/-- The seed's disagreements with `v` over `P`, as a sum over `P`'s own subtype. -/
lemma screenCount_eq_attach (O : Oracle μ S) {cn cd : ℕ} (hcd : cn < cd) (P : Finset S)
    (v : S) (ω : Ω) :
    ∑ i : {p // p ∈ P}, seedLoss O cn cd v i.val ω = (screenCount O.mq P v ω : ℝ) := by
  rw [screenCount_eq_sum O hcd P v ω]
  exact Finset.sum_attach P (fun p => seedLoss O cn cd v p ω)

open scoped Classical in
/-- A clean candidate's disagreement count, against a real threshold: its mean is exactly
`cleanLoss`, so it exceeds that by `γ` per prefix only in the upper tail. -/
theorem screenCount_upper {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S) {cn cd : ℕ}
    (hcd : cn < cd) {P : Finset S} (hP : ∀ p ∈ P, p ∈ Pre) (v : S) (hv : v ≠ 1)
    (γ : ℝ) (hγ : 0 ≤ γ) (hsig : O.η ≤ 1 / 2) (hclean : ∀ p ∈ P, O.flip v p = 0) :
    μ.real {ω | cleanLoss O P + (P.card : ℝ) * γ ≤ (screenCount O.mq P v ω : ℝ)}
      ≤ Real.exp (-2 * (P.card : ℝ) * γ ^ 2) := by
  classical
  have hmean : ∑ i ∈ (Finset.univ : Finset {p // p ∈ P}), μ[seedLoss O cn cd v i.val]
      ≤ cleanLoss O P := by
    rw [cleanLoss, ← Finset.sum_coe_sort P (fun p => 2 * O.rate p * (1 - O.rate p))]
    exact le_of_eq (Finset.sum_congr rfl (fun i _ =>
      (seedLoss_mean O hcd v i.val (mul_ne_self i.val v hv) hsig).2 (hclean i.val i.property)))
  have htail := sumUpper_le_total (fun (i : {p // p ∈ P}) => seedLoss O cn cd v i.val)
    Finset.univ _ γ (fun i => (seedLoss_meas O cn cd v i.val).aemeasurable)
    (seedLoss_indep hflat O cn cd hP v) (fun i => seedLoss_icc O cn cd v i.val) hmean hγ
  simp only [Finset.card_univ, Fintype.card_coe] at htail
  refine le_trans (measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)) htail
  change _ ≤ ∑ i : {p // p ∈ P}, seedLoss O cn cd v i.val ω
  rw [screenCount_eq_attach O hcd P v ω]
  exact hω

open scoped Classical in
/-- Any candidate's disagreement count sits at least `cleanLoss` in mean — the flip term is
nonnegative — so it falls `γ` per prefix below that only in the lower tail.  This is what puts
a floor under `screenBase`. -/
theorem screenCount_lower {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S) {cn cd : ℕ}
    (hcd : cn < cd) {P : Finset S} (hP : ∀ p ∈ P, p ∈ Pre) (v : S) (hv : v ≠ 1)
    (Δ γ : ℝ) (hγ : 0 ≤ γ) (hsig : O.η ≤ 1 / 2)
    (hflip : Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p) :
    μ.real {ω | (screenCount O.mq P v ω : ℝ)
        ≤ cleanLoss O P + (P.card : ℝ) * (Δ * (1 - 2 * O.η) ^ 2 - γ)}
      ≤ Real.exp (-2 * (P.card : ℝ) * γ ^ 2) := by
  classical
  have hmean : cleanLoss O P + (P.card : ℝ) * (Δ * (1 - 2 * O.η) ^ 2)
      ≤ ∑ i ∈ (Finset.univ : Finset {p // p ∈ P}), μ[seedLoss O cn cd v i.val] := by
    have hle := Finset.sum_le_sum (fun (i : {p // p ∈ P}) (_ : i ∈ Finset.univ) =>
      (seedLoss_mean O hcd v i.val (mul_ne_self i.val v hv) hsig).1)
    rw [Finset.sum_add_distrib, ← Finset.sum_mul,
      Finset.sum_coe_sort P (fun p => 2 * O.rate p * (1 - O.rate p)),
      Finset.sum_coe_sort P (fun p => O.flip v p)] at hle
    rw [cleanLoss]
    nlinarith [mul_le_mul_of_nonneg_right hflip (sq_nonneg (1 - 2 * O.η))]
  have htail := sumLower_le_total (fun (i : {p // p ∈ P}) => seedLoss O cn cd v i.val)
    Finset.univ _ γ (fun i => (seedLoss_meas O cn cd v i.val).aemeasurable)
    (seedLoss_indep hflat O cn cd hP v) (fun i => seedLoss_icc O cn cd v i.val) hmean hγ
  simp only [Finset.card_univ, Fintype.card_coe] at htail
  refine le_trans (measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)) htail
  change ∑ i : {p // p ∈ P}, seedLoss O cn cd v i.val ω ≤ _
  rw [screenCount_eq_attach O hcd P v ω]
  have h : (screenCount O.mq P v ω : ℝ)
      ≤ cleanLoss O P + (P.card : ℝ) * (Δ * (1 - 2 * O.η) ^ 2 - γ) := hω
  linarith

/-! ### The iterate, and what actually bounds its flips

The first step's centre is the seed's own column, which the screen already controls.
Every step after that ranks candidates against the *current* centre's majority vote, and
that ranking is by agreement with the centre's drift rather than by flipping little: writing
`Dset` for the centre's error set, at a single rate `η` a candidate scores mean
`η·#P + (1−2η)·#(Φ_v Δ Dset)`, so
one that flips exactly `Dset` scores zero.  Chasing the bound through the majority vote gives
`d' ≤ (2 / c) · d` with `c = (s + eps) / (2 * s)`, about `3` at the usual settings — no
contraction.

None of that matters, because the ranking is not what bounds the flips: the screen is.
A suffix that fails it never becomes a fully observed column and so is never a clustering
candidate at all, and the iterate can only choose among what is left. -/

/-- The family flips no more than the screened pool does. -/
theorem clusterAt_flip_bound (O : Oracle μ S) (populations : Finset J) (B : State)
    (x : Run Ω S J) (Δ : ℝ)
    (hscreen : ∀ v ∈ screenedAt O.mq populations B x,
      ¬ (Δ * ((prefixesAt populations B.npref x).card : ℝ)
        ≤ ∑ p ∈ prefixesAt populations B.npref x, O.flip v p)) :
    ∀ w ∈ clusterAt O.mq populations x B,
      ¬ (Δ * ((prefixesAt populations B.npref x).card : ℝ)
        ≤ ∑ p ∈ prefixesAt populations B.npref x, O.flip w p) :=
  fun w hw => hscreen w (clusterAround_subset O B.cn B.cd _ _ (oracleNoise x) B.k
    (one_mem_screenedAt O populations B x) hw)

/-! ### From flip mass to a correct cut

The band is what turns "few members flip" into "the cut is right".  A rejecting prefix is
accepted only when the vote clears the centre, and members preserving at `p` contribute only
through noise, so with a fraction `f` flipping the vote's mean is at most `η + (1 − η)·f` — a
flip can cost a whole read when the rates are lopsided.  The vote therefore absorbs a flipping
`flipFrac` of the family, and Markov over the members' flip masses charges the misclassified
mass at `Δ/flipFrac`. -/

open scoped Classical in
/-- The number of the family's members that flip at a prefix, as a real. -/
noncomputable def flipCount (O : Oracle μ S) (F : Finset S) (p : S) : ℝ :=
  ((F.filter (fun v => O.flip v p = 1)).card : ℝ)

lemma flipCount_eq_sum (O : Oracle μ S) (F : Finset S) (p : S) :
    flipCount O F p = ∑ v ∈ F, O.flip v p := by
  classical
  unfold flipCount
  rw [← Finset.sum_filter_add_sum_filter_not F (fun v => O.flip v p = 1)]
  have h1 : ∑ v ∈ F.filter (fun v => O.flip v p = 1), O.flip v p
      = ((F.filter (fun v => O.flip v p = 1)).card : ℝ) := by
    rw [Finset.sum_congr rfl (fun v hv => (Finset.mem_filter.1 hv).2), Finset.sum_const,
      nsmul_eq_mul, mul_one]
  have h0 : ∑ v ∈ F.filter (fun v => ¬ (O.flip v p = 1)), O.flip v p = 0 := by
    refine Finset.sum_eq_zero (fun v hv => ?_)
    obtain ⟨-, hne⟩ := Finset.mem_filter.1 hv
    rcases O.flip_bit v p with h | h
    · exact h
    · exact absurd h hne
  rw [h1, h0, add_zero]

lemma flip_integrable (O : Oracle μ S) (Dj : Measure S) [IsProbabilityMeasure Dj] (v : S) :
    Integrable (fun p => O.flip v p) Dj :=
  MeasureTheory.Integrable.of_mem_Icc 0 1 (flip_meas O v).aemeasurable
    (Filter.Eventually.of_forall (fun p => flip_icc O v p))

/-- Markov over the family's flip masses.  If no member flips more than a `Δ` mass of
the population, the mass of prefixes where a `c` fraction of the family flips is at most
`Δ / c`.

This is what turns per-member flip mass into misclassified mass.  The round puts `c` at the
flip fraction the vote's margin absorbs, a constant, so nothing here scales with `|F|` — a
union bound over the members would have cost a factor of the family's size. -/
theorem flipCount_mass_le (O : Oracle μ S) (Dj : Measure S) [IsProbabilityMeasure Dj]
    (F : Finset S) (Δ c : ℝ) (hc : 0 < c) (hFne : 0 < F.card)
    (hF : ∀ v ∈ F, flipMass O Dj v ≤ Δ) :
    Dj.real {p | c * (F.card : ℝ) ≤ flipCount O F p} ≤ Δ / c := by
  classical
  have hcard : (0 : ℝ) < (F.card : ℝ) := by exact_mod_cast hFne
  have hsum : ∀ p, flipCount O F p = ∑ v ∈ F, O.flip v p := flipCount_eq_sum O F
  have hint : Integrable (fun p => flipCount O F p) Dj := by
    refine ((integrable_finset_sum F (fun v _ => flip_integrable O Dj v)).congr ?_)
    exact Filter.Eventually.of_forall (fun p => (hsum p).symm)
  have hnn : 0 ≤ᵐ[Dj] fun p => flipCount O F p :=
    Filter.Eventually.of_forall (fun p => Nat.cast_nonneg _)
  have hInt : ∫ p, flipCount O F p ∂Dj = ∑ v ∈ F, flipMass O Dj v := by
    rw [integral_congr_ae (Filter.Eventually.of_forall hsum),
      integral_finset_sum _ (fun v _ => flip_integrable O Dj v)]
    rfl
  have hmark := mul_meas_ge_le_integral_of_nonneg hnn hint (c * (F.card : ℝ))
  rw [hInt] at hmark
  have hle : ∑ v ∈ F, flipMass O Dj v ≤ (F.card : ℝ) * Δ := by
    calc ∑ v ∈ F, flipMass O Dj v ≤ ∑ _v ∈ F, Δ := Finset.sum_le_sum hF
      _ = (F.card : ℝ) * Δ := by rw [Finset.sum_const, nsmul_eq_mul]
  have hstep : (F.card : ℝ) * (c * Dj.real {p | c * (F.card : ℝ) ≤ flipCount O F p})
      ≤ (F.card : ℝ) * Δ := by
    calc (F.card : ℝ) * (c * Dj.real {p | c * (F.card : ℝ) ≤ flipCount O F p})
        = c * (F.card : ℝ) * Dj.real {p | c * (F.card : ℝ) ≤ flipCount O F p} := by ring
      _ ≤ ∑ v ∈ F, flipMass O Dj v := hmark
      _ ≤ (F.card : ℝ) * Δ := hle
  have := le_of_mul_le_mul_left hstep hcard
  rw [le_div_iff₀ hc]
  linarith

open scoped Classical in
/-- The vote as a sum of reads, which is what concentrates; a.e. equal to `voteCount`. -/
noncomputable def voteSum (O : Oracle μ S) (F : Finset S) (p : S) (ω : Ω) : ℝ :=
  ∑ v ∈ F, O.mq (p * v) ω

lemma voteCount_eq_voteSum (O : Oracle μ S) (F : Finset S) (p : S) :
    ∀ᵐ ω ∂μ, ((voteCount O.mq F p ω : ℝ)) = voteSum O F p ω := by
  classical
  filter_upwards [(ae_ball_iff F.countable_toSet).2 (fun v _ => mq_bit O (p * v))] with ω hω
  unfold voteCount voteSum
  rw [← Finset.sum_filter_add_sum_filter_not F (fun v => O.mq (p * v) ω = 1)]
  have h1 : ∑ v ∈ F.filter (fun v => O.mq (p * v) ω = 1), O.mq (p * v) ω
      = ((F.filter (fun v => O.mq (p * v) ω = 1)).card : ℝ) := by
    rw [Finset.sum_congr rfl (fun v hv => (Finset.mem_filter.1 hv).2), Finset.sum_const,
      nsmul_eq_mul, mul_one]
  have h0 : ∑ v ∈ F.filter (fun v => ¬ (O.mq (p * v) ω = 1)), O.mq (p * v) ω = 0 := by
    refine Finset.sum_eq_zero (fun v hv => ?_)
    obtain ⟨hvF, hne⟩ := Finset.mem_filter.1 hv
    rcases hω v hvF with h | h
    · exact h
    · exact absurd h hne
  rw [h1, h0, add_zero]

lemma mq_indep_shift (O : Oracle μ S) (p : S) :
    iIndepFun (fun v : S => O.mq (p * v)) μ :=
  (mq_indep O).precomp (mul_right_injective p)

/-- Upper tail on a rejecting prefix.  A clean member reads accepting at most at rate `η` and a
flipping one at most always, so a flip fraction of `f` lifts the vote's mean only to
`η + (1 − η)·f`. -/
theorem voteSum_upper (O : Oracle μ S) (F : Finset S) (p : S) (hp : O.label p = 0)
    (f γ : ℝ) (hsig : O.η ≤ 1 / 2) (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ) :
    μ.real {ω | (F.card : ℝ) * ((O.η + (1 - O.η) * f) + γ) ≤ voteSum O F p ω}
      ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  have hper : ∀ v ∈ F, μ[O.mq (p * v)] ≤ O.η + (1 - O.η) * O.flip v p := by
    intro v _
    have hlab : O.label (p * v) = O.flip v p := by
      show O.label (p * v) = O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p
      rw [hp]; ring
    rw [mq_mean, hlab]
    have hr0 := O.rate_nonneg (p * v)
    have hr1 := O.rate_le_eta (p * v)
    rcases O.flip_bit v p with h | h <;> rw [h] <;> linarith
  have hmean : ∑ v ∈ F, μ[O.mq (p * v)] ≤ (F.card : ℝ) * (O.η + (1 - O.η) * f) := by
    refine le_trans (Finset.sum_le_sum hper) ?_
    rw [Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum,
      ← flipCount_eq_sum]
    have h1 : (0 : ℝ) ≤ 1 - O.η := by linarith
    nlinarith [mul_le_mul_of_nonneg_left hf h1]
  exact sumUpper_le (fun v : S => O.mq (p * v)) F (O.η + (1 - O.η) * f) γ
    (fun v => (mq_meas O _).aemeasurable) (mq_indep_shift O p) (fun v => mq_icc O _) hmean hγ

/-- Lower tail on an accepting prefix.  Mirror of `voteSum_upper`: a clean member reads
accepting at rate at least `1 − η` and a flipping one possibly never, so the vote's mean only
falls to `(1 − η)(1 − f)`. -/
theorem voteSum_lower (O : Oracle μ S) (F : Finset S) (p : S) (hp : O.label p = 1)
    (f γ : ℝ) (hsig : O.η ≤ 1 / 2) (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ) :
    μ.real {ω | voteSum O F p ω ≤ (F.card : ℝ) * (((1 - O.η) * (1 - f)) - γ)}
      ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  have hper : ∀ v ∈ F, (1 - O.η) * (1 - O.flip v p) ≤ μ[O.mq (p * v)] := by
    intro v _
    have hlab : O.label (p * v) = 1 - O.flip v p := by
      show O.label (p * v) = 1 - (O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p)
      rw [hp]; ring
    rw [mq_mean, hlab]
    have hr0 := O.rate_nonneg (p * v)
    have hr1 := O.rate_le_eta (p * v)
    rcases O.flip_bit v p with h | h <;> rw [h] <;> linarith
  have hmean : (F.card : ℝ) * ((1 - O.η) * (1 - f)) ≤ ∑ v ∈ F, μ[O.mq (p * v)] := by
    refine le_trans ?_ (Finset.sum_le_sum hper)
    rw [← Finset.mul_sum, Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul, mul_one,
      ← flipCount_eq_sum]
    have h1 : (0 : ℝ) ≤ 1 - O.η := by linarith
    nlinarith [mul_le_mul_of_nonneg_left hf h1]
  exact sumLower_le (fun v : S => O.mq (p * v)) F ((1 - O.η) * (1 - f)) γ
    (fun v => (mq_meas O _).aemeasurable) (mq_indep_shift O p) (fun v => mq_icc O _) hmean hγ

/-- `voteSum_upper` recentred: a clean member sits at `mid − hgap` and a flipping one at the
other class's mean `mid + hgap`, so a flip fraction of `f` lifts the vote's mean by exactly
`2·hgap·f` — a flip costs the displacement between the classes, not a whole bit. -/
theorem voteSum_upper_gap (O : Oracle μ S) (F : Finset S) (p : S) (hp : O.label p = 0)
    (f γ : ℝ) (hgap0 : 0 ≤ O.hgap) (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ) :
    μ.real {ω | (F.card : ℝ) * (((O.mid - O.hgap) + 2 * O.hgap * f) + γ) ≤ voteSum O F p ω}
      ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  have hper : ∀ v ∈ F, μ[O.mq (p * v)] = (O.mid - O.hgap) + 2 * O.hgap * O.flip v p := by
    intro v _
    have hlab : O.label (p * v) = O.flip v p := by
      show O.label (p * v) = O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p
      rw [hp]; ring
    rw [mq_mean_centred, hlab]
  have hmean : ∑ v ∈ F, μ[O.mq (p * v)]
      ≤ (F.card : ℝ) * ((O.mid - O.hgap) + 2 * O.hgap * f) := by
    rw [Finset.sum_congr rfl hper, Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul,
      ← Finset.mul_sum, ← flipCount_eq_sum]
    nlinarith [mul_le_mul_of_nonneg_left hf hgap0]
  exact sumUpper_le (fun v : S => O.mq (p * v)) F ((O.mid - O.hgap) + 2 * O.hgap * f) γ
    (fun v => (mq_meas O _).aemeasurable) (mq_indep_shift O p) (fun v => mq_icc O _) hmean hγ

/-- `voteSum_lower` recentred.  Mirror of `voteSum_upper_gap`. -/
theorem voteSum_lower_gap (O : Oracle μ S) (F : Finset S) (p : S) (hp : O.label p = 1)
    (f γ : ℝ) (hgap0 : 0 ≤ O.hgap) (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ) :
    μ.real {ω | voteSum O F p ω ≤ (F.card : ℝ) * (((O.mid + O.hgap) - 2 * O.hgap * f) - γ)}
      ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  have hper : ∀ v ∈ F, μ[O.mq (p * v)] = (O.mid + O.hgap) - 2 * O.hgap * O.flip v p := by
    intro v _
    have hlab : O.label (p * v) = 1 - O.flip v p := by
      show O.label (p * v) = 1 - (O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p)
      rw [hp]; ring
    rw [mq_mean_centred, hlab]; ring
  have hmean : (F.card : ℝ) * ((O.mid + O.hgap) - 2 * O.hgap * f)
      ≤ ∑ v ∈ F, μ[O.mq (p * v)] := by
    rw [Finset.sum_congr rfl hper, Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul,
      ← Finset.mul_sum, ← flipCount_eq_sum]
    nlinarith [mul_le_mul_of_nonneg_left hf hgap0]
  exact sumLower_le (fun v : S => O.mq (p * v)) F ((O.mid + O.hgap) - 2 * O.hgap * f) γ
    (fun v => (mq_meas O _).aemeasurable) (mq_indep_shift O p) (fun v => mq_icc O _) hmean hγ

lemma measureReal_le_of_ae_imp {A B : Set Ω} (h : ∀ᵐ ω ∂μ, ω ∈ A → ω ∈ B) :
    μ.real A ≤ μ.real B :=
  ENNReal.toReal_mono (measure_ne_top μ B) (measure_mono_ae h)

/-- The cut is correct at a prefix the family barely flips.  Only the side the prefix
actually sits on can fail, so one tail — not two — pays for it. -/
theorem cutCorrect_whp (O : Oracle μ S) (F : Finset S) (p : S) (lo hi : ℕ) (f γ : ℝ)
    (hsig : O.η ≤ 1 / 2)
    (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ)
    (hhi : (F.card : ℝ) * ((O.η + (1 - O.η) * f) + γ) ≤ (hi : ℝ))
    (hlo : (lo : ℝ) < (F.card : ℝ) * (((1 - O.η) * (1 - f)) - γ)) :
    μ.real {ω | ¬ cutCorrect O lo hi F p ω} ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  rcases O.label_bit p with hp | hp
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_upper O F p hp f γ hsig hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hacc : ¬ (hi < voteCount O.mq F p ω → O.label p = 1) := by
      intro hacc
      exact hbad ⟨hacc, fun _ => hp⟩
    have hgt : hi < voteCount O.mq F p ω := by
      by_contra hc
      exact hacc (fun h => absurd h hc)
    have : (hi : ℝ) < (voteCount O.mq F p ω : ℝ) := by exact_mod_cast hgt
    show (F.card : ℝ) * ((O.η + (1 - O.η) * f) + γ) ≤ voteSum O F p ω
    rw [← heq]; linarith
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_lower O F p hp f γ hsig hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hrej : ¬ (voteCount O.mq F p ω ≤ lo → O.label p = 0) := by
      intro hrej
      exact hbad ⟨fun _ => hp, hrej⟩
    have hle : voteCount O.mq F p ω ≤ lo := by
      by_contra hc
      exact hrej (fun h => absurd h hc)
    have : (voteCount O.mq F p ω : ℝ) ≤ (lo : ℝ) := by exact_mod_cast hle
    show voteSum O F p ω ≤ (F.card : ℝ) * (((1 - O.η) * (1 - f)) - γ)
    rw [← heq]; linarith

/-- `cutCorrect_whp` recentred, relaying `voteSum_upper_gap`/`voteSum_lower_gap`. -/
theorem cutCorrect_whp_gap (O : Oracle μ S) (F : Finset S) (p : S) (lo hi : ℕ) (f γ : ℝ)
    (hgap0 : 0 ≤ O.hgap)
    (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ)
    (hhi : (F.card : ℝ) * (((O.mid - O.hgap) + 2 * O.hgap * f) + γ) ≤ (hi : ℝ))
    (hlo : (lo : ℝ) < (F.card : ℝ) * (((O.mid + O.hgap) - 2 * O.hgap * f) - γ)) :
    μ.real {ω | ¬ cutCorrect O lo hi F p ω} ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  rcases O.label_bit p with hp | hp
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_upper_gap O F p hp f γ hgap0 hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hacc : ¬ (hi < voteCount O.mq F p ω → O.label p = 1) := by
      intro hacc
      exact hbad ⟨hacc, fun _ => hp⟩
    have hgt : hi < voteCount O.mq F p ω := by
      by_contra hc
      exact hacc (fun h => absurd h hc)
    have : (hi : ℝ) < (voteCount O.mq F p ω : ℝ) := by exact_mod_cast hgt
    show (F.card : ℝ) * (((O.mid - O.hgap) + 2 * O.hgap * f) + γ) ≤ voteSum O F p ω
    rw [← heq]; linarith
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_lower_gap O F p hp f γ hgap0 hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hrej : ¬ (voteCount O.mq F p ω ≤ lo → O.label p = 0) := by
      intro hrej
      exact hbad ⟨fun _ => hp, hrej⟩
    have hle : voteCount O.mq F p ω ≤ lo := by
      by_contra hc
      exact hrej (fun h => absurd h hc)
    have : (voteCount O.mq F p ω : ℝ) ≤ (lo : ℝ) := by exact_mod_cast hle
    show voteSum O F p ω ≤ (F.card : ℝ) * (((O.mid + O.hgap) - 2 * O.hgap * f) - γ)
    rw [← heq]; linarith

/-! ### From a per-prefix bound to a population bound

The population's prefixes are countable, so the mass of bad ones is a sum rather than an
integral over a product, and the exchange is `lintegral_tsum` — no product measure and no
joint measurability in the pair. -/

lemma tsum_singleton_eq (Dj : Measure S) (A : Set S) : ∑' a : A, Dj {(a : S)} = Dj A := by
  have h := tsum_measure_preimage_singleton (μ := Dj) A.to_countable
    (f := (id : S → S)) (fun y _ => measurableSet_singleton y)
  simpa using h

lemma measure_setOf_eq_tsum (Dj : Measure S) (A : Set S) :
    Dj A = ∑' a : S, A.indicator (fun a => Dj {a}) a := by
  rw [← tsum_singleton_eq Dj A]
  exact tsum_subtype A (fun a => Dj {a})

/-- The expected bad mass is the worst per-prefix bound.  Each prefix's failure has
`μ`-probability at most `E`, and the prefix masses sum to one. -/
lemma badMass_eq_tsum (Dj : Measure S) (Bad : S → Set Ω) (ω : Ω) :
    Dj {p | ω ∈ Bad p} = ∑' p : S, (Bad p).indicator (fun _ => Dj {p}) ω := by
  rw [measure_setOf_eq_tsum]
  exact tsum_congr (fun p => by by_cases hb : ω ∈ Bad p <;> simp [hb])

lemma measurable_badMass (Dj : Measure S) (Bad : S → Set Ω)
    (hmeas : ∀ p, MeasurableSet (Bad p)) : Measurable (fun ω => Dj {p | ω ∈ Bad p}) := by
  simp only [badMass_eq_tsum Dj Bad]
  exact Measurable.ennreal_tsum (fun p => measurable_const.indicator (hmeas p))

/-- The law of the draws alone. -/
noncomputable def drawMeasure (D : J → Measure S) (Dsf : Measure S) :
    Measure ((((ℕ → S) × (J → ℕ → S))) × (J → ℕ → S)) :=
  (((Measure.infinitePi fun _ : ℕ => Dsf).prod
      (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j))).prod
    (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j)

instance (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] : IsProbabilityMeasure (drawMeasure D Dsf) := by
  unfold drawMeasure; infer_instance

lemma runMeasure_eq_prod (D : J → Measure S) (Dsf : Measure S) :
    runMeasure μ D Dsf = μ.prod (drawMeasure D Dsf) := rfl

/-- One table coordinate has the population's own law. -/
lemma map_drawCoord (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (j : J) (i : ℕ) :
    Measure.map (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J → ℕ → S)) => d.1.2 j i)
        (drawMeasure D Dsf) = D j := by
  have hstep : (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J → ℕ → S)) => d.1.2 j i)
      = (fun p : ℕ → S => p i) ∘ ((fun q : J → ℕ → S => q j) ∘ (Prod.snd ∘ Prod.fst)) := rfl
  rw [hstep, ← Measure.map_map (by fun_prop) (by fun_prop),
    ← Measure.map_map (by fun_prop) (by fun_prop),
    ← Measure.map_map measurable_snd measurable_fst, drawMeasure, Measure.map_fst_prod]
  simp only [measure_univ, one_smul]
  rw [Measure.map_snd_prod]
  simp only [measure_univ, one_smul]
  rw [(measurePreserving_eval (fun j : J => Measure.infinitePi fun _ : ℕ => D j) j).map_eq,
    (measurePreserving_eval_infinitePi (fun _ : ℕ => D j) i).map_eq]

/-- A bound at almost every fixed draw is a bound on the run.  The clustering's prefixes
and candidates are draws, so its guarantees are stated for the noise at a fixed table; this
is what lifts them.  The `a.e.` is what lets the table be assumed inside the flat set. -/
lemma runMeasure_slice_le (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (A : Set (Run Ω S J)) (hA : MeasurableSet A) (E : ℝ≥0∞)
    (h : ∀ᵐ d ∂(drawMeasure D Dsf), μ {ω | ((ω, d) : Run Ω S J) ∈ A} ≤ E) :
    runMeasure μ D Dsf A ≤ E := by
  rw [runMeasure_eq_prod, Measure.prod_apply_symm hA]
  exact le_trans (lintegral_mono_ae h) (by simp)

/-- The table's prefixes land in the flat set, since that is where the populations live. -/
lemma ae_draws_mem_Pre (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (Pre : Set S)
    (populations : Finset J) (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) :
    ∀ᵐ d ∂(drawMeasure D Dsf), ∀ j ∈ populations, ∀ i : ℕ, d.1.2 j i ∈ Pre := by
  have hmeasPre : MeasurableSet (Preᶜ : Set S) := (Set.to_countable _).measurableSet
  have hcoord : ∀ z : J × ℕ, ∀ᵐ d ∂(drawMeasure D Dsf), z.1 ∈ populations → d.1.2 z.1 z.2 ∈ Pre := by
    rintro ⟨j, i⟩
    by_cases hj : j ∈ populations
    · have hz : drawMeasure D Dsf {d | ¬ (j ∈ populations → d.1.2 j i ∈ Pre)} = 0 := by
        have hset : {d : ((((ℕ → S) × (J → ℕ → S))) × (J → ℕ → S))
            | ¬ (j ∈ populations → d.1.2 j i ∈ Pre)}
            = (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J → ℕ → S)) => d.1.2 j i) ⁻¹' Preᶜ := by
          ext d; simp [hj]
        rw [hset, ← Measure.map_apply (by fun_prop) hmeasPre, map_drawCoord D Dsf j i]
        exact hsupp j hj
      exact ae_iff.2 hz
    · filter_upwards with d hjj
      exact absurd hjj hj
  filter_upwards [ae_all_iff.2 hcoord] with d hd j hj i
  exact hd (j, i) hj

/-- A bound at every fixed noise-and-table slice is a bound on the run.  The
certification draws are the last factor, so they can be sliced off on their own — which is
what lets the gate be judged on prefixes the family was never selected from. -/
lemma runMeasure_slice_cert_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (A : Set (Run Ω S J)) (hA : MeasurableSet A) (E : ℝ≥0∞)
    (h : ∀ y : Ω × ((ℕ → S) × (J → ℕ → S)),
      (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j)
        {c | ((y.1, (y.2, c)) : Run Ω S J) ∈ A} ≤ E) :
    runMeasure μ D Dsf A ≤ E := by
  set νsq : Measure ((ℕ → S) × (J → ℕ → S)) :=
    (Measure.infinitePi fun _ : ℕ => Dsf).prod
      (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) with hνsq
  set νc : Measure (J → ℕ → S) := Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j with hνc
  have hmap : Measure.map (MeasurableEquiv.prodAssoc : (Ω × ((ℕ → S) × (J → ℕ → S)))
      × (J → ℕ → S) ≃ᵐ Ω × (((ℕ → S) × (J → ℕ → S)) × (J → ℕ → S)))
      ((μ.prod νsq).prod νc) = runMeasure μ D Dsf :=
    (measurePreserving_prodAssoc μ νsq νc).map_eq
  have hpre : runMeasure μ D Dsf A = ((μ.prod νsq).prod νc)
      ((MeasurableEquiv.prodAssoc : (Ω × ((ℕ → S) × (J → ℕ → S)))
        × (J → ℕ → S) ≃ᵐ Run Ω S J) ⁻¹' A) := by
    rw [← hmap, Measure.map_apply (MeasurableEquiv.prodAssoc).measurable hA]
  rw [hpre, Measure.prod_apply ((MeasurableEquiv.prodAssoc).measurable hA)]
  calc ∫⁻ y, νc (Prod.mk y ⁻¹' (MeasurableEquiv.prodAssoc ⁻¹' A)) ∂(μ.prod νsq)
      ≤ ∫⁻ _, E ∂(μ.prod νsq) := lintegral_mono (fun y => h y)
    _ = E := by simp

/-- The certification stream's own law. -/
lemma map_certStream (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] :
    Measure.map (certStream : Run Ω S J → _) (runMeasure μ D Dsf)
      = Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j := by
  rw [show (certStream : Run Ω S J → _) = Prod.snd ∘ Prod.snd from rfl,
    ← Measure.map_map measurable_snd measurable_snd, runMeasure, Measure.map_snd_prod]
  simp only [measure_univ, one_smul]
  rw [Measure.map_snd_prod]
  simp

/-- One certification coordinate has the population's own law. -/
lemma map_certCoord (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (j : J) (i : ℕ) :
    Measure.map (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J → ℕ → S)) => d.2 j i)
        (drawMeasure D Dsf) = D j := by
  have hstep : (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J → ℕ → S)) => d.2 j i)
      = (fun c : J → ℕ → S => c j i) ∘ Prod.snd := rfl
  rw [hstep, ← Measure.map_map (by fun_prop) measurable_snd, drawMeasure, Measure.map_snd_prod]
  simp only [measure_univ, one_smul]
  exact ((measurePreserving_eval_infinitePi (fun _ : ℕ => D j) i).comp
    (measurePreserving_eval (fun j : J => Measure.infinitePi fun _ : ℕ => D j) j)).map_eq

/-- The certification prefixes land in the flat set too, for the same reason the table's
do. -/
lemma ae_cert_mem_Pre (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (Pre : Set S)
    (populations : Finset J) (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) :
    ∀ᵐ d ∂(drawMeasure D Dsf), ∀ j ∈ populations, ∀ i : ℕ, d.2 j i ∈ Pre := by
  have hmeasPre : MeasurableSet (Preᶜ : Set S) := (Set.to_countable _).measurableSet
  have hcoord : ∀ z : J × ℕ, ∀ᵐ d ∂(drawMeasure D Dsf),
      z.1 ∈ populations → d.2 z.1 z.2 ∈ Pre := by
    rintro ⟨j, i⟩
    by_cases hj : j ∈ populations
    · have hz : drawMeasure D Dsf {d | ¬ (j ∈ populations → d.2 j i ∈ Pre)} = 0 := by
        have hset : {d : ((((ℕ → S) × (J → ℕ → S))) × (J → ℕ → S))
            | ¬ (j ∈ populations → d.2 j i ∈ Pre)}
            = (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J → ℕ → S)) => d.2 j i) ⁻¹' Preᶜ := by
          ext d; simp [hj]
        rw [hset, ← Measure.map_apply (by fun_prop) hmeasPre, map_certCoord D Dsf j i]
        exact hsupp j hj
      exact ae_iff.2 hz
    · filter_upwards with d hjj
      exact absurd hjj hj
  filter_upwards [ae_all_iff.2 hcoord] with d hd j hj i
  exact hd (j, i) hj

open scoped Classical in
lemma map_certBlock (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (j : J) (m : ℕ) :
    Measure.map (fun x : Run Ω S J => (fun i : Fin m => certPrefix j i.val x)) (runMeasure μ D Dsf)
      = Measure.pi (fun _ : Fin m => D j) := by
  have hstep : (fun x : Run Ω S J => (fun i : Fin m => certPrefix j i.val x))
      = (fun s : ℕ → S => (fun i : Fin m => s i.val))
        ∘ ((fun c : J → ℕ → S => c j) ∘ (certStream : Run Ω S J → _)) := rfl
  rw [hstep, ← Measure.map_map (by fun_prop) (by fun_prop),
    ← Measure.map_map (by fun_prop) (by fun_prop), map_certStream D Dsf,
    (measurePreserving_eval (fun j : J => Measure.infinitePi fun _ : ℕ => D j) j).map_eq]
  refine (Measure.pi_eq (μ := fun _ : Fin m => D j) fun t ht => ?_).symm
  have hpre : (fun s : ℕ → S => (fun i : Fin m => s i.val)) ⁻¹' Set.univ.pi t
      = Set.pi ↑(Finset.range m) (fun i => if h : i < m then t ⟨i, h⟩ else Set.univ) := by
    ext s
    simp only [Set.mem_preimage, Set.mem_pi, Set.mem_univ, forall_const, Finset.coe_range,
      Set.mem_Iio]
    constructor
    · intro h i hi
      rw [dif_pos hi]
      exact h ⟨i, hi⟩
    · intro h i
      have := h i.val i.isLt
      simpa [dif_pos i.isLt] using this
  rw [Measure.map_apply (by fun_prop) (MeasurableSet.univ_pi ht), hpre, Measure.infinitePi_pi]
  · rw [← Fin.prod_univ_eq_prod_range]
    exact Finset.prod_congr rfl fun i _ => by simp [dif_pos i.isLt]
  · intro i _
    split_ifs with h
    exacts [ht ⟨i, h⟩, .univ]

/-- A table prefix and a certification prefix are drawn from independent streams, so their
joint law is the product — which is what lets `cross_collision_le` price a collision between
the two. -/
lemma map_prefCertPair (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (j' j : J) (i i' : ℕ) :
    Measure.map (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J → ℕ → S)) => (d.1.2 j' i, d.2 j i'))
        (drawMeasure D Dsf) = (D j').prod (D j) := by
  have hf : Measurable (fun y : (ℕ → S) × (J → ℕ → S) => y.2 j' i) := by fun_prop
  have hg : Measurable (fun c : J → ℕ → S => c j i') := by fun_prop
  have hmapf : Measure.map (fun y : (ℕ → S) × (J → ℕ → S) => y.2 j' i)
      ((Measure.infinitePi fun _ : ℕ => Dsf).prod
        (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j)) = D j' := by
    rw [show (fun y : (ℕ → S) × (J → ℕ → S) => y.2 j' i)
        = (fun q : J → ℕ → S => q j' i) ∘ Prod.snd from rfl,
      ← Measure.map_map (by fun_prop) measurable_snd, Measure.map_snd_prod]
    simp only [measure_univ, one_smul]
    rw [show (fun q : J → ℕ → S => q j' i)
        = (fun r : ℕ → S => r i) ∘ (fun q : J → ℕ → S => q j') from rfl,
      ← Measure.map_map (by fun_prop) (by fun_prop),
      (measurePreserving_eval (fun j : J => Measure.infinitePi fun _ : ℕ => D j) j').map_eq,
      (measurePreserving_eval_infinitePi (fun _ : ℕ => D j') i).map_eq]
  have hmapg : Measure.map (fun c : J → ℕ → S => c j i')
      (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) = D j :=
    ((measurePreserving_eval_infinitePi (fun _ : ℕ => D j) i').comp
      (measurePreserving_eval (fun j : J => Measure.infinitePi fun _ : ℕ => D j) j)).map_eq
  rw [show (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J → ℕ → S)) => (d.1.2 j' i, d.2 j i'))
      = Prod.map (fun y : (ℕ → S) × (J → ℕ → S) => y.2 j' i)
        (fun c : J → ℕ → S => c j i') from rfl,
    drawMeasure, ← Measure.map_prod_map _ _ hf hg, hmapf, hmapg]

/-! ### Unioning over a drawn pool

The candidates are drawn, so a union bound over them is a union over an `x`-dependent set.
What makes it cost `M` rather than everything is that a candidate's index is drawn from the
suffix stream while the event it indexes lives on the prefix streams — a different factor of
the same product — so slicing costs nothing. -/

lemma map_drawBlock (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] :
    Measure.map (draws : Run Ω S J → _) (runMeasure μ D Dsf)
      = (Measure.infinitePi fun _ : ℕ => Dsf).prod
          (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) := by
  rw [show (draws : Run Ω S J → _) = Prod.fst ∘ Prod.snd from rfl,
    ← Measure.map_map measurable_fst measurable_snd, runMeasure, Measure.map_snd_prod]
  simp only [measure_univ, one_smul]
  rw [Measure.map_fst_prod]
  simp only [measure_univ, one_smul]

/-- A drawn index costs nothing.  The event is indexed by a suffix draw and decided by
the prefix draws, so the worst case over candidates bounds the run. -/
theorem runMeasure_draw_selection_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (C : S → Set (J → ℕ → S)) (hC : ∀ v, MeasurableSet (C v)) (i : ℕ) (E : ℝ≥0∞)
    (hbad : ∀ v, (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) (C v) ≤ E) :
    runMeasure μ D Dsf {x : Run Ω S J | prefixStreams x ∈ C (suffixDraw i x)} ≤ E := by
  classical
  set νs : Measure (ℕ → S) := Measure.infinitePi fun _ : ℕ => Dsf with hνs
  set νq : Measure (J → ℕ → S) := Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j
    with hνq
  have hmeasSet : MeasurableSet {y : (ℕ → S) × (J → ℕ → S) | y.2 ∈ C (y.1 i)} := by
    have hcov : {y : (ℕ → S) × (J → ℕ → S) | y.2 ∈ C (y.1 i)}
        = ⋃ a : S, ({y : (ℕ → S) × (J → ℕ → S) | y.1 i = a} ∩ {y | y.2 ∈ C a}) := by
      ext y
      simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff]
      exact ⟨fun h => ⟨y.1 i, rfl, h⟩, fun ⟨a, ha, h⟩ => ha ▸ h⟩
    rw [hcov]
    exact MeasurableSet.iUnion (fun a =>
      (measurableSet_eq_fun ((measurable_pi_apply i).comp measurable_fst) measurable_const).inter
        (measurable_snd (hC a)))
  have hpre : {x : Run Ω S J | prefixStreams x ∈ C (suffixDraw i x)}
      = (draws : Run Ω S J → _) ⁻¹' {y : (ℕ → S) × (J → ℕ → S) | y.2 ∈ C (y.1 i)} := rfl
  rw [hpre, ← Measure.map_apply (by fun_prop) hmeasSet, map_drawBlock D Dsf,
    Measure.prod_apply hmeasSet]
  calc ∫⁻ a, νq (Prod.mk a ⁻¹' {y : (ℕ → S) × (J → ℕ → S) | y.2 ∈ C (y.1 i)}) ∂νs
      = ∫⁻ a, νq (C (a i)) ∂νs := by rfl
    _ ≤ ∫⁻ _, E ∂νs := lintegral_mono (fun a => hbad (a i))
    _ = E := by simp

/-! ### Level 2: the empirical flip rate is honest

The clustering only ever sees a candidate's flips on the prefixes actually drawn.  Those
are `m` i.i.d. draws from the population, so the count concentrates around `m · flipMass`
and a candidate that looks clean on them is clean. -/

lemma map_prefixBlock (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (j : J) (m : ℕ) :
    Measure.map (fun x : Run Ω S J => (fun i : Fin m => prefixDraw j i.val x)) (runMeasure μ D Dsf)
      = Measure.pi (fun _ : Fin m => D j) := by
  have hstep : (fun x : Run Ω S J => (fun i : Fin m => prefixDraw j i.val x))
      = (fun y : ((Fin m → S) × (J → Fin m → S)) => y.2 j)
        ∘ (fun x : Run Ω S J => ((fun i : Fin m => suffixDraw i.val x),
            (fun (j : J) (i : Fin m) => prefixDraw j i.val x))) := rfl
  have hmeasBlock : Measurable (fun x : Run Ω S J => ((fun i : Fin m => suffixDraw i.val x),
      (fun (j : J) (i : Fin m) => prefixDraw j i.val x))) :=
    (measurable_pi_lambda _ (fun i : Fin m => measurable_sfx i.val)).prodMk
      (measurable_pi_lambda _ (fun j : J =>
        measurable_pi_lambda _ (fun i : Fin m => measurable_prf j i.val)))
  rw [hstep, ← Measure.map_map (by fun_prop) hmeasBlock, map_firstDraws D Dsf m,
    show (fun y : ((Fin m → S) × (J → Fin m → S)) => y.2 j)
      = (fun q : J → Fin m → S => q j) ∘ Prod.snd from rfl,
    ← Measure.map_map (by fun_prop) measurable_snd, Measure.map_snd_prod]
  simp only [measure_univ, one_smul]
  exact (measurePreserving_eval (fun j : J => Measure.pi fun _ : Fin m => D j) j).map_eq

lemma pi_flip_mean (Dj : Measure S) [IsProbabilityMeasure Dj] (O : Oracle μ S) (m : ℕ)
    (v : S) (i : Fin m) :
    (Measure.pi fun _ : Fin m => Dj)[fun q : Fin m → S => O.flip v (q i)]
      = flipMass O Dj v := by
  have hmap : Measure.map (fun q : Fin m → S => q i) (Measure.pi fun _ : Fin m => Dj) = Dj :=
    (measurePreserving_eval (fun _ : Fin m => Dj) i).map_eq
  calc (Measure.pi fun _ : Fin m => Dj)[fun q : Fin m → S => O.flip v (q i)]
      = ∫ w, O.flip v w ∂(Measure.map (fun q : Fin m → S => q i)
          (Measure.pi fun _ : Fin m => Dj)) := by
        rw [integral_map (measurable_pi_apply i).aemeasurable
          (flip_meas O v).aestronglyMeasurable]
    _ = flipMass O Dj v := by rw [hmap]; rfl

/-- Level 2 for one population.  A candidate the drawn prefixes say is clean really is. -/
theorem prefix_flip_lower (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (j : J) (m : ℕ) (v : S) (g : ℝ) (hg : 0 ≤ g) :
    (runMeasure μ D Dsf).real {x : Run Ω S J | ∑ i : Fin m, O.flip v (prefixDraw j i.val x)
        ≤ (m : ℝ) * (flipMass O (D j) v - g)}
      ≤ Real.exp (-2 * (m : ℝ) * g ^ 2) := by
  classical
  have hcard : ((Finset.univ : Finset (Fin m)).card : ℝ) = (m : ℝ) := by simp
  have htail := sumLower_le (μ := Measure.pi fun _ : Fin m => D j)
    (fun (i : Fin m) (q : Fin m → S) => O.flip v (q i)) (Finset.univ : Finset (Fin m))
    (flipMass O (D j) v) g
    (fun i => ((flip_meas O v).comp (measurable_pi_apply i)).aemeasurable)
    (iIndepFun_pi (fun _ => (flip_meas O v).aemeasurable))
    (fun i => Filter.Eventually.of_forall (fun q => flip_icc O v (q i)))
    (by rw [Finset.sum_congr rfl (fun i _ => pi_flip_mean (D j) O m v i), Finset.sum_const,
      nsmul_eq_mul, hcard]) hg
  rw [hcard] at htail
  have hpre : {x : Run Ω S J | ∑ i : Fin m, O.flip v (prefixDraw j i.val x)
      ≤ (m : ℝ) * (flipMass O (D j) v - g)}
      = (fun x : Run Ω S J => (fun i : Fin m => prefixDraw j i.val x)) ⁻¹'
        {q : Fin m → S | ∑ i : Fin m, O.flip v (q i) ≤ (m : ℝ) * (flipMass O (D j) v - g)} := rfl
  have hmeasPrf : Measurable (fun x : Run Ω S J => (fun i : Fin m => prefixDraw j i.val x)) :=
    measurable_pi_lambda _ (fun i : Fin m => measurable_prf j i.val)
  rw [hpre, measureReal_def,
    Measure.map_apply hmeasPrf (measurableSet_le (by fun_prop) measurable_const)
      |>.symm.trans (congrArg (fun ν : Measure (Fin m → S) => ν _) (map_prefixBlock D Dsf j m)),
    ← measureReal_def]
  exact htail

lemma runMeasure_prefix_apply (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (C : Set (J → ℕ → S)) (hC : MeasurableSet C) :
    runMeasure μ D Dsf {x : Run Ω S J | prefixStreams x ∈ C}
      = (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) C := by
  have hpre : {x : Run Ω S J | prefixStreams x ∈ C} = (draws : Run Ω S J → _) ⁻¹' (Prod.snd ⁻¹' C) :=
    rfl
  rw [hpre, ← Measure.map_apply (by fun_prop) (measurable_snd hC), map_drawBlock D Dsf,
    ← Measure.map_apply measurable_snd hC, Measure.map_snd_prod]
  simp

open scoped Classical in
/-- The candidates the drawn prefixes understate: really flipping more than `Δ`, but
empirically inside `Δ - g`. -/
noncomputable def understated (O : Oracle μ S) (D : J → Measure S) (j : J) (m : ℕ)
    (Δ g : ℝ) (v : S) : Set (J → ℕ → S) :=
  {q | Δ < flipMass O (D j) v ∧
    ∑ i : Fin m, O.flip v (q j i.val) ≤ (m : ℝ) * (Δ - g)}

lemma measurableSet_understated (O : Oracle μ S) (D : J → Measure S) (j : J) (m : ℕ)
    (Δ g : ℝ) (v : S) : MeasurableSet (understated O D j m Δ g v) := by
  classical
  have hsum : MeasurableSet
      {q : J → ℕ → S | ∑ i : Fin m, O.flip v (q j i.val) ≤ (m : ℝ) * (Δ - g)} :=
    measurableSet_le
      (Finset.measurable_sum _ (fun i _ => (flip_meas O v).comp
        ((measurable_pi_apply i.val).comp (measurable_pi_apply j)))) measurable_const
  by_cases h : Δ < flipMass O (D j) v
  · simpa [understated, h] using hsum
  · simpa [understated, h] using MeasurableSet.empty

lemma measure_understated_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (j : J) (m : ℕ) (Δ g : ℝ) (hg : 0 ≤ g) (v : S) :
    (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) (understated O D j m Δ g v)
      ≤ ENNReal.ofReal (Real.exp (-2 * (m : ℝ) * g ^ 2)) := by
  rw [← runMeasure_prefix_apply (μ := μ) D Dsf _ (measurableSet_understated O D j m Δ g v)]
  have hsub : {x : Run Ω S J | prefixStreams x ∈ understated O D j m Δ g v}
      ⊆ {x : Run Ω S J | ∑ i : Fin m, O.flip v (prefixDraw j i.val x)
          ≤ (m : ℝ) * (flipMass O (D j) v - g)} := by
    rintro x ⟨hΔ, hcount⟩
    have hm : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
    exact le_trans hcount (by nlinarith)
  refine le_trans (measure_mono hsub) ?_
  rw [← ENNReal.ofReal_toReal (measure_ne_top (runMeasure μ D Dsf) _), ← measureReal_def]
  exact ENNReal.ofReal_le_ofReal (prefix_flip_lower D Dsf O j m v g hg)

/-- The whole drawn pool is honest at once.  A pool member the drawn prefixes say sits
inside `Δ - g` really flips at most `Δ` of the population, off an `M · exp(-2 m g²)` set. -/
theorem pool_flipMass_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (j : J) (m M : ℕ) (Δ g : ℝ) (hg : 0 ≤ g) (hΔ : 0 ≤ Δ) :
    (runMeasure μ D Dsf).real {x : Run Ω S J | ¬ ∀ v ∈ poolAt M x,
        (∑ i : Fin m, O.flip v (prefixDraw j i.val x) ≤ (m : ℝ) * (Δ - g)) → flipMass O (D j) v ≤ Δ}
      ≤ (M : ℝ) * Real.exp (-2 * (m : ℝ) * g ^ 2) := by
  classical
  set E : ℝ≥0∞ := ENNReal.ofReal (Real.exp (-2 * (m : ℝ) * g ^ 2)) with hE
  have hseed : flipMass O (D j) (1 : S) = 0 := by
    have : ∀ p : S, O.flip (1 : S) p = 0 := by
      intro p
      show O.label (p * 1) + O.label p - 2 * O.label (p * 1) * O.label p = 0
      rw [mul_one]
      rcases O.label_bit p with hl | hl <;> rw [hl] <;> ring
    simp [flipMass, this]
  have hsub : {x : Run Ω S J | ¬ ∀ v ∈ poolAt M x,
      (∑ i : Fin m, O.flip v (prefixDraw j i.val x) ≤ (m : ℝ) * (Δ - g)) → flipMass O (D j) v ≤ Δ}
      ⊆ ⋃ i ∈ Finset.range M, {x : Run Ω S J | prefixStreams x ∈ understated O D j m Δ g (suffixDraw i x)} := by
    intro x hx
    simp only [Set.mem_setOf_eq, not_forall] at hx
    obtain ⟨v, hv, hcount, hmass⟩ := hx
    rcases Finset.mem_insert.1 hv with rfl | hv'
    · exact absurd (hseed ▸ hΔ) hmass
    · obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hv'
      exact Set.mem_biUnion hi ⟨not_le.1 hmass, hcount⟩
  have hbound : runMeasure μ D Dsf {x : Run Ω S J | ¬ ∀ v ∈ poolAt M x,
      (∑ i : Fin m, O.flip v (prefixDraw j i.val x) ≤ (m : ℝ) * (Δ - g)) → flipMass O (D j) v ≤ Δ}
      ≤ (M : ℝ≥0∞) * E := by
    refine le_trans (measure_mono hsub) (le_trans (measure_biUnion_finset_le _ _) ?_)
    calc ∑ i ∈ Finset.range M,
          runMeasure μ D Dsf {x : Run Ω S J | prefixStreams x ∈ understated O D j m Δ g (suffixDraw i x)}
        ≤ ∑ _i ∈ Finset.range M, E :=
          Finset.sum_le_sum (fun i _ => runMeasure_draw_selection_le D Dsf _
            (measurableSet_understated O D j m Δ g) i E
            (fun v => measure_understated_le (μ := μ) D Dsf O j m Δ g hg v))
      _ = (M : ℝ≥0∞) * E := by rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf {x : Run Ω S J | ¬ ∀ v ∈ poolAt M x,
          (∑ i : Fin m, O.flip v (prefixDraw j i.val x) ≤ (m : ℝ) * (Δ - g))
            → flipMass O (D j) v ≤ Δ}).toReal
      ≤ ((M : ℝ≥0∞) * E).toReal :=
        ENNReal.toReal_mono (by simp [hE, ENNReal.mul_eq_top]) hbound
    _ = (M : ℝ) * Real.exp (-2 * (m : ℝ) * g ^ 2) := by
        rw [ENNReal.toReal_mul, hE, ENNReal.toReal_ofReal (Real.exp_nonneg _)]
        simp

/-! ### The run's set-valued data is measurable

The table, the pool and the family are `Finset`s read off finitely many draws and finitely
many oracle bits.  `Finset S` is countable, so every event about them is a countable union
of coordinate fibres — no factorisation theorem, the same move as `measurableSet_filter_pred`
one level up. -/

lemma measurableSet_finData {ι : Type*} [Fintype ι] (c : ι → Run Ω S J → S)
    (hc : ∀ z, Measurable (c z)) (g : (ι → S) → Finset S) (Q : Finset S) :
    MeasurableSet {x : Run Ω S J | g (fun z => c z x) = Q} := by
  classical
  have hcov : {x : Run Ω S J | g (fun z => c z x) = Q}
      = ⋃ t : {t : ι → S // g t = Q}, ⋂ z : ι, {x : Run Ω S J | c z x = t.val z} := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_iInter]
    refine ⟨fun h => ⟨⟨fun z => c z x, h⟩, fun z => rfl⟩, ?_⟩
    rintro ⟨⟨t, ht⟩, hx⟩
    rw [show (fun z => c z x) = t from funext hx]
    exact ht
  rw [hcov]
  exact MeasurableSet.iUnion (fun t => MeasurableSet.iInter (fun z =>
    (hc z) (measurableSet_singleton (t.val z))))

open scoped Classical in
lemma image_range_eq_image_univ {α : Type*} [DecidableEq α] (m : ℕ) (f : ℕ → α) :
    (Finset.range m).image f = (Finset.univ : Finset (Fin m)).image (fun i => f i.val) := by
  classical
  ext a
  simp only [Finset.mem_image, Finset.mem_range, Finset.mem_univ, true_and]
  exact ⟨fun ⟨i, hi, h⟩ => ⟨⟨i, hi⟩, h⟩, fun ⟨i, h⟩ => ⟨i.val, i.isLt, h⟩⟩

open scoped Classical in
lemma measurableSet_prefixesAt (populations : Finset J) (m : ℕ) (Q : Finset S) :
    MeasurableSet {x : Run Ω S J | prefixesAt populations m x = Q} := by
  classical
  have hrw : ∀ x : Run Ω S J, prefixesAt populations m x
      = populations.biUnion (fun j => (Finset.univ : Finset (Fin m)).image
          (fun i => (fun z : J × Fin m => prefixDraw z.1 z.2.val x) (j, i))) := by
    intro x
    exact Finset.biUnion_congr rfl (fun j _ => image_range_eq_image_univ m (fun i => prefixDraw j i x))
  simp only [hrw]
  exact measurableSet_finData (fun z : J × Fin m => prefixDraw z.1 z.2.val)
    (fun z => measurable_prf z.1 z.2.val)
    (fun t => populations.biUnion (fun j => (Finset.univ : Finset (Fin m)).image
      (fun i => t (j, i)))) Q

open scoped Classical in
lemma measurableSet_poolAt (M : ℕ) (C : Finset S) :
    MeasurableSet {x : Run Ω S J | poolAt M x = C} := by
  classical
  have hrw : ∀ x : Run Ω S J, poolAt M x
      = insert 1 ((Finset.univ : Finset (Fin M)).image
          (fun i => (fun z : Fin M => suffixDraw z.val x) i)) := by
    intro x
    exact congrArg (insert 1) (image_range_eq_image_univ M (fun i => suffixDraw i x))
  simp only [hrw]
  exact measurableSet_finData (fun z : Fin M => suffixDraw z.val) (fun z => measurable_sfx z.val)
    (fun t => insert 1 ((Finset.univ : Finset (Fin M)).image (fun i => t i))) C

open scoped Classical in
/-- Events about the table and the pool are measurable.  Decompose over their values,
of which there are countably many. -/
lemma measurableSet_of_run_data (populations : Finset J) (B : State)
    (R : Finset S → Finset S → Set (Run Ω S J)) (hR : ∀ P C, MeasurableSet (R P C)) :
    MeasurableSet {x : Run Ω S J | x ∈ R (prefixesAt populations B.npref x) (poolAt B.nsuff x)} := by
  classical
  have hcov : {x : Run Ω S J | x ∈ R (prefixesAt populations B.npref x) (poolAt B.nsuff x)}
      = ⋃ z : Finset S × Finset S, (({x : Run Ω S J | prefixesAt populations B.npref x = z.1}
          ∩ {x : Run Ω S J | poolAt B.nsuff x = z.2}) ∩ R z.1 z.2) := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff]
    refine ⟨fun h => ⟨(prefixesAt populations B.npref x, poolAt B.nsuff x), ⟨rfl, rfl⟩, h⟩, ?_⟩
    rintro ⟨⟨P, C⟩, ⟨hP, hC⟩, hx⟩
    simp only at hP hC
    rw [hP, hC]
    exact hx
  rw [hcov]
  exact MeasurableSet.iUnion (fun z =>
    ((measurableSet_prefixesAt populations B.npref z.1).inter
      (measurableSet_poolAt B.nsuff z.2)).inter (hR z.1 z.2))

lemma measureReal_le_one' (Dj : Measure S) [IsProbabilityMeasure Dj] (A : Set S) :
    Dj.real A ≤ 1 := by
  have h := measureReal_mono (μ := Dj) (Set.subset_univ A) (by finiteness)
  simpa using h

lemma summable_singleton_real (Dj : Measure S) [IsProbabilityMeasure Dj] :
    Summable (fun a : S => Dj.real {a}) := by
  classical
  refine summable_of_sum_le (c := 1) (fun a => measureReal_nonneg) (fun Q => ?_)
  rw [sum_measureReal_singleton]
  exact measureReal_le_one' Dj _

lemma summable_singleton_sq (Dj : Measure S) [IsProbabilityMeasure Dj] :
    Summable (fun a : S => Dj.real {a} ^ 2) := by
  refine Summable.of_nonneg_of_le (fun a => sq_nonneg _) (fun a => ?_) (summable_singleton_real Dj)
  nlinarith [measureReal_nonneg (μ := Dj) (s := ({a} : Set S)), measureReal_le_one' Dj ({a} : Set S)]

lemma map_suffixBlock (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (M : ℕ) :
    Measure.map (fun x : Run Ω S J => (fun i : Fin M => suffixDraw i.val x)) (runMeasure μ D Dsf)
      = Measure.pi (fun _ : Fin M => Dsf) := by
  have hstep : (fun x : Run Ω S J => (fun i : Fin M => suffixDraw i.val x))
      = Prod.fst ∘ (fun x : Run Ω S J => ((fun i : Fin M => suffixDraw i.val x),
          (fun (j : J) (i : Fin M) => prefixDraw j i.val x))) := rfl
  have hmeasBlock : Measurable (fun x : Run Ω S J => ((fun i : Fin M => suffixDraw i.val x),
      (fun (j : J) (i : Fin M) => prefixDraw j i.val x))) :=
    (measurable_pi_lambda _ (fun i : Fin M => measurable_sfx i.val)).prodMk
      (measurable_pi_lambda _ (fun j : J =>
        measurable_pi_lambda _ (fun i : Fin M => measurable_prf j i.val)))
  rw [hstep, ← Measure.map_map measurable_fst hmeasBlock, map_firstDraws D Dsf M,
    Measure.map_fst_prod]
  simp

/-- The suffix draws are distinct, except on an `M²ρ` set — the pool is interned, so the
indices that preserve acceptance only become that many *candidates* when they differ. -/
theorem suffix_not_injective_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (M : ℕ) (ρ : ℝ)
    (hρ : collisionMass Dsf ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runMeasure μ D Dsf).real
        {x : Run Ω S J | ¬ Function.Injective (fun i : Fin M => suffixDraw i.val x)}
      ≤ (M : ℝ) ^ 2 * ρ := by
  classical
  have hmeasSfx : Measurable (fun x : Run Ω S J => (fun i : Fin M => suffixDraw i.val x)) :=
    measurable_pi_lambda _ (fun i : Fin M => measurable_sfx i.val)
  have hmeasSet : MeasurableSet {q : Fin M → S | ¬ Function.Injective q} := by
    have hcov : {q : Fin M → S | ¬ Function.Injective q}
        = ⋃ z : {z : Fin M × Fin M // z.1 ≠ z.2}, {q : Fin M → S | q z.val.1 = q z.val.2} := by
      ext q
      simp only [Set.mem_setOf_eq, Set.mem_iUnion, Function.not_injective_iff]
      constructor
      · rintro ⟨a, b, hab, hne⟩; exact ⟨⟨(a, b), hne⟩, hab⟩
      · rintro ⟨⟨⟨a, b⟩, hne⟩, hab⟩; exact ⟨a, b, hab, hne⟩
    rw [hcov]
    exact MeasurableSet.iUnion (fun z =>
      measurableSet_eq_fun (measurable_pi_apply _) (measurable_pi_apply _))
  have hpre : {x : Run Ω S J | ¬ Function.Injective (fun i : Fin M => suffixDraw i.val x)}
      = (fun x : Run Ω S J => (fun i : Fin M => suffixDraw i.val x)) ⁻¹'
        {q : Fin M → S | ¬ Function.Injective q} := rfl
  rw [hpre, measureReal_def, Measure.map_apply hmeasSfx hmeasSet
    |>.symm.trans (congrArg (fun ν : Measure (Fin M → S) => ν _) (map_suffixBlock D Dsf M)),
    ← measureReal_def]
  exact pi_not_injective_le Dsf M ρ hρ hρ0

open scoped Classical in
/-- The vote is decisive at a prefix the family barely flips.  As in `cutCorrect_whp` a flip
fraction of `f` moves the mean by up to `(1 − η)·f`, so both thresholds shift by that. -/
theorem decided_whp (O : Oracle μ S) (F : Finset S) (p : S) (lo hi : ℕ) (f γ : ℝ) (hγ : 0 ≤ γ)
    (hsig : O.η ≤ 1 / 2)
    (hf : flipCount O F p ≤ (F.card : ℝ) * f)
    (hhi : (hi : ℝ) ≤ (F.card : ℝ) * (((1 - O.η) * (1 - f)) - γ))
    (hlo : (F.card : ℝ) * ((O.η + (1 - O.η) * f) + γ) ≤ (lo : ℝ) + 1) :
    μ.real {ω | ¬ decided O.mq lo hi F p ω} ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  rcases O.label_bit p with hp | hp
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_upper O F p hp f γ hsig hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hgt : lo < voteCount O.mq F p ω := by
      by_contra hc
      exact hbad (Or.inr (not_lt.1 hc))
    have hcast : (lo : ℝ) + 1 ≤ (voteCount O.mq F p ω : ℝ) := by exact_mod_cast hgt
    show (F.card : ℝ) * ((O.η + (1 - O.η) * f) + γ) ≤ voteSum O F p ω
    rw [← heq]
    linarith
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_lower O F p hp f γ hsig hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hle : voteCount O.mq F p ω ≤ hi := by
      by_contra hc
      exact hbad (Or.inl (not_le.1 hc))
    have hcast : (voteCount O.mq F p ω : ℝ) ≤ (hi : ℝ) := by exact_mod_cast hle
    show voteSum O F p ω ≤ (F.card : ℝ) * (((1 - O.η) * (1 - f)) - γ)
    rw [← heq]
    linarith

open scoped Classical in
/-- `decided_whp` recentred, relaying `voteSum_upper_gap`/`voteSum_lower_gap`. -/
theorem decided_whp_gap (O : Oracle μ S) (F : Finset S) (p : S) (lo hi : ℕ) (f γ : ℝ)
    (hγ : 0 ≤ γ) (hgap0 : 0 ≤ O.hgap)
    (hf : flipCount O F p ≤ (F.card : ℝ) * f)
    (hhi : (hi : ℝ) ≤ (F.card : ℝ) * (((O.mid + O.hgap) - 2 * O.hgap * f) - γ))
    (hlo : (F.card : ℝ) * (((O.mid - O.hgap) + 2 * O.hgap * f) + γ) ≤ (lo : ℝ) + 1) :
    μ.real {ω | ¬ decided O.mq lo hi F p ω} ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  rcases O.label_bit p with hp | hp
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_upper_gap O F p hp f γ hgap0 hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hgt : lo < voteCount O.mq F p ω := by
      by_contra hc
      exact hbad (Or.inr (not_lt.1 hc))
    have hcast : (lo : ℝ) + 1 ≤ (voteCount O.mq F p ω : ℝ) := by exact_mod_cast hgt
    show (F.card : ℝ) * (((O.mid - O.hgap) + 2 * O.hgap * f) + γ) ≤ voteSum O F p ω
    rw [← heq]
    linarith
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_lower_gap O F p hp f γ hgap0 hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hle : voteCount O.mq F p ω ≤ hi := by
      by_contra hc
      exact hbad (Or.inl (not_le.1 hc))
    have hcast : (voteCount O.mq F p ω : ℝ) ≤ (hi : ℝ) := by exact_mod_cast hle
    show voteSum O F p ω ≤ (F.card : ℝ) * (((O.mid + O.hgap) - 2 * O.hgap * f) - γ)
    rw [← heq]
    linarith

/-! ### From the clustering's empirical bound to the population's

The clustering scores a candidate on the *deduplicated* table, the sampler draws `m` times
with replacement, and `flipMass` is about a fresh draw.  On the event that a population's
draws are distinct the three agree; `pi_not_injective_le` prices the rest at `m²ρ`. -/

open scoped Classical in
lemma sum_eq_sum_prefixesOf (O : Oracle μ S) (j : J) (m : ℕ) (x : Run Ω S J) (v : S)
    (hinj : Function.Injective (fun i : Fin m => prefixDraw j i.val x)) :
    ∑ i : Fin m, O.flip v (prefixDraw j i.val x) = ∑ p ∈ prefixesOf j m x, O.flip v p := by
  classical
  unfold prefixesOf
  rw [Finset.sum_image ?_, ← Fin.sum_univ_eq_sum_range]
  intro a ha b hb hab
  have := hinj (show (fun i : Fin m => prefixDraw j i.val x) ⟨a, Finset.mem_range.1 ha⟩
    = (fun i : Fin m => prefixDraw j i.val x) ⟨b, Finset.mem_range.1 hb⟩ from hab)
  simpa using congrArg Fin.val this

open scoped Classical in
lemma sum_prefixesOf_le_prefixesAt (O : Oracle μ S) (populations : Finset J) (j : J)
    (hj : j ∈ populations) (m : ℕ) (x : Run Ω S J) (v : S) :
    ∑ p ∈ prefixesOf j m x, O.flip v p ≤ ∑ p ∈ prefixesAt populations m x, O.flip v p := by
  classical
  refine Finset.sum_le_sum_of_subset_of_nonneg ?_ (fun p _ _ => (flip_icc O v p).1)
  exact fun p hp => Finset.mem_biUnion.2 ⟨j, hj, hp⟩

open scoped Classical in
lemma card_prefixesAt_le (populations : Finset J) (m : ℕ) (x : Run Ω S J) :
    (prefixesAt populations m x).card ≤ populations.card * m := by
  classical
  refine le_trans (Finset.card_biUnion_le) ?_
  calc ∑ j ∈ populations, (prefixesOf j m x).card
      ≤ ∑ _j ∈ populations, m :=
        Finset.sum_le_sum (fun j _ => le_trans Finset.card_image_le (by simp))
    _ = populations.card * m := by rw [Finset.sum_const, smul_eq_mul]

/-- The population's own draws are distinct, except on an `m²ρ` set. -/
theorem prefix_not_injective_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (j : J) (m : ℕ) (ρ : ℝ)
    (hρ : collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runMeasure μ D Dsf).real
        {x : Run Ω S J | ¬ Function.Injective (fun i : Fin m => prefixDraw j i.val x)}
      ≤ (m : ℝ) ^ 2 * ρ := by
  classical
  have hmeasPrf : Measurable (fun x : Run Ω S J => (fun i : Fin m => prefixDraw j i.val x)) :=
    measurable_pi_lambda _ (fun i : Fin m => measurable_prf j i.val)
  have hpre : {x : Run Ω S J | ¬ Function.Injective (fun i : Fin m => prefixDraw j i.val x)}
      = (fun x : Run Ω S J => (fun i : Fin m => prefixDraw j i.val x)) ⁻¹'
        {p : Fin m → S | ¬ Function.Injective p} := rfl
  have hmeasSet : MeasurableSet {p : Fin m → S | ¬ Function.Injective p} := by
    have hcov : {p : Fin m → S | ¬ Function.Injective p}
        = ⋃ z : {z : Fin m × Fin m // z.1 ≠ z.2}, {p : Fin m → S | p z.val.1 = p z.val.2} := by
      ext p
      simp only [Set.mem_setOf_eq, Set.mem_iUnion, Function.not_injective_iff]
      constructor
      · rintro ⟨a, b, hab, hne⟩; exact ⟨⟨(a, b), hne⟩, hab⟩
      · rintro ⟨⟨⟨a, b⟩, hne⟩, hab⟩; exact ⟨a, b, hab, hne⟩
    rw [hcov]
    exact MeasurableSet.iUnion (fun z =>
      measurableSet_eq_fun (measurable_pi_apply _) (measurable_pi_apply _))
  rw [hpre, measureReal_def, Measure.map_apply hmeasPrf hmeasSet
    |>.symm.trans (congrArg (fun ν : Measure (Fin m → S) => ν _) (map_prefixBlock D Dsf j m)),
    ← measureReal_def]
  exact pi_not_injective_le (D j) m ρ hρ hρ0

/-- The certification draws are distinct too, on the same `m²ρ`.  Needed because the
gate's split counts *prefixes*, and a repeated draw is one prefix, not two. -/
theorem cert_not_injective_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (j : J) (m : ℕ) (ρ : ℝ)
    (hρ : collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runMeasure μ D Dsf).real
        {x : Run Ω S J | ¬ Function.Injective (fun i : Fin m => certPrefix j i.val x)}
      ≤ (m : ℝ) ^ 2 * ρ := by
  classical
  have hmeasPrf : Measurable (fun x : Run Ω S J => (fun i : Fin m => certPrefix j i.val x)) :=
    measurable_pi_lambda _ (fun i : Fin m => measurable_cert j i.val)
  have hpre : {x : Run Ω S J | ¬ Function.Injective (fun i : Fin m => certPrefix j i.val x)}
      = (fun x : Run Ω S J => (fun i : Fin m => certPrefix j i.val x)) ⁻¹'
        {p : Fin m → S | ¬ Function.Injective p} := rfl
  have hmeasSet : MeasurableSet {p : Fin m → S | ¬ Function.Injective p} := by
    have hcov : {p : Fin m → S | ¬ Function.Injective p}
        = ⋃ z : {z : Fin m × Fin m // z.1 ≠ z.2}, {p : Fin m → S | p z.val.1 = p z.val.2} := by
      ext p
      simp only [Set.mem_setOf_eq, Set.mem_iUnion, Function.not_injective_iff]
      constructor
      · rintro ⟨a, b, hab, hne⟩; exact ⟨⟨(a, b), hne⟩, hab⟩
      · rintro ⟨⟨⟨a, b⟩, hne⟩, hab⟩; exact ⟨a, b, hab, hne⟩
    rw [hcov]
    exact MeasurableSet.iUnion (fun z =>
      measurableSet_eq_fun (measurable_pi_apply _) (measurable_pi_apply _))
  rw [hpre, measureReal_def, Measure.map_apply hmeasPrf hmeasSet
    |>.symm.trans (congrArg (fun ν : Measure (Fin m → S) => ν _) (map_certBlock D Dsf j m)),
    ← measureReal_def]
  exact pi_not_injective_le (D j) m ρ hρ hρ0

lemma map_prefCertPairRun (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (j' j : J) (i i' : ℕ) :
    Measure.map (fun x : Run Ω S J => (prefixDraw j' i x, certPrefix j i' x)) (runMeasure μ D Dsf)
      = (D j').prod (D j) := by
  rw [show (fun x : Run Ω S J => (prefixDraw j' i x, certPrefix j i' x))
      = (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J → ℕ → S)) => (d.1.2 j' i, d.2 j i'))
        ∘ Prod.snd from rfl,
    ← Measure.map_map (by fun_prop) measurable_snd, runMeasure_eq_prod, Measure.map_snd_prod]
  simp only [measure_univ, one_smul]
  exact map_prefCertPair D Dsf j' j i i'

open scoped Classical in
/-- The gate's prefixes are fresh.  A certification draw repeating a table prefix costs
the same `ρ` as a repeat inside one stream, and there are `|populations|·m²` pairs. -/
theorem prefix_cert_disjoint_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (populations : Finset J) (j : J) (m : ℕ) (ρ : ℝ)
    (hρ : ∀ j' ∈ populations, collisionMass (D j') ≤ ρ) (hρj : collisionMass (D j) ≤ ρ)
    (hρ0 : 0 ≤ ρ) :
    (runMeasure μ D Dsf).real
        {x : Run Ω S J | ¬ Disjoint (prefixesAt populations m x) (certOf j m x)}
      ≤ (populations.card : ℝ) * (m : ℝ) ^ 2 * ρ := by
  classical
  set κ := populations ×ˢ (Finset.range m ×ˢ Finset.range m) with hκ
  have hsub : {x : Run Ω S J | ¬ Disjoint (prefixesAt populations m x) (certOf j m x)}
      ⊆ ⋃ z ∈ κ, {x : Run Ω S J | prefixDraw z.1 z.2.1 x = certPrefix j z.2.2 x} := by
    intro x hx
    obtain ⟨a, ha, ha'⟩ := Finset.not_disjoint_iff.1 hx
    obtain ⟨j', hj', hja⟩ := Finset.mem_biUnion.1 ha
    obtain ⟨i, hi, hia⟩ := Finset.mem_image.1 hja
    obtain ⟨i', hi', hia'⟩ := Finset.mem_image.1 ha'
    exact Set.mem_biUnion (show (j', i, i') ∈ κ by simp [hκ, hj', Finset.mem_range.1 hi,
      Finset.mem_range.1 hi']) (by simpa using hia.trans hia'.symm)
  have hone : ∀ z : J × ℕ × ℕ, z ∈ κ →
      (runMeasure μ D Dsf).real {x : Run Ω S J | prefixDraw z.1 z.2.1 x = certPrefix j z.2.2 x} ≤ ρ := by
    intro z hz
    have hj' : z.1 ∈ populations := (Finset.mem_product.1 hz).1
    have hdiag : MeasurableSet {q : S × S | q.1 = q.2} :=
      measurableSet_eq_fun measurable_fst measurable_snd
    have hpre : {x : Run Ω S J | prefixDraw z.1 z.2.1 x = certPrefix j z.2.2 x}
        = (fun x : Run Ω S J => (prefixDraw z.1 z.2.1 x, certPrefix j z.2.2 x)) ⁻¹' {q : S × S | q.1 = q.2} :=
      rfl
    rw [hpre, measureReal_def,
      Measure.map_apply ((measurable_prf z.1 z.2.1).prodMk (measurable_cert j z.2.2)) hdiag
        |>.symm.trans (congrArg (fun ν : Measure (S × S) => ν _)
          (map_prefCertPairRun D Dsf z.1 j z.2.1 z.2.2)),
      ← measureReal_def]
    exact cross_collision_le (D z.1) (D j) ρ (hρ z.1 hj') hρj
      (summable_singleton_sq (D z.1)) (summable_singleton_sq (D j))
  calc (runMeasure μ D Dsf).real
        {x : Run Ω S J | ¬ Disjoint (prefixesAt populations m x) (certOf j m x)}
      ≤ (runMeasure μ D Dsf).real (⋃ z ∈ κ, {x : Run Ω S J | prefixDraw z.1 z.2.1 x = certPrefix j z.2.2 x}) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ z ∈ κ, (runMeasure μ D Dsf).real {x : Run Ω S J | prefixDraw z.1 z.2.1 x = certPrefix j z.2.2 x} :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _z ∈ κ, ρ := Finset.sum_le_sum hone
    _ = (κ.card : ℝ) * ρ := by rw [Finset.sum_const, nsmul_eq_mul]
    _ = (populations.card : ℝ) * (m : ℝ) ^ 2 * ρ := by
        have hcard : κ.card = populations.card * (m * m) := by
          rw [hκ, Finset.card_product, Finset.card_product, Finset.card_range]
        rw [hcard]; push_cast; ring

/-- The certification prefixes are read by neither the clustering nor the split. -/
lemma disjoint_gateReads {Pre : Set S} (hflat : Flat Pre) (P C A : Finset S)
    (hP : ∀ q ∈ P, q ∈ Pre) (hA : ∀ p ∈ A, p ∈ Pre) (hPA : Disjoint P A) :
    Disjoint (↑A : Set S) (↑(readSet P C ∪ readSet A (C.erase 1)) : Set S) := by
  classical
  rw [Finset.coe_union, Set.disjoint_union_right]
  exact ⟨Finset.disjoint_coe.2 (disjoint_readSet hflat hP hA hPA),
    Finset.disjoint_coe.2 (disjoint_readSet_erase hflat hA)⟩

lemma measurableSet_of_fam {fam : Ω → Finset S} {T : Finset (Finset S)}
    (hfam : ∀ ω, fam ω ∈ T) (hfamMeas : ∀ A₀, MeasurableSet {ω | fam ω = A₀})
    (R : Finset S → Set Ω) (hR : ∀ A₀, MeasurableSet (R A₀)) :
    MeasurableSet {ω | ω ∈ R (fam ω)} := by
  classical
  have hcover : {ω | ω ∈ R (fam ω)} = ⋃ A₀ ∈ T, ({ω | fam ω = A₀} ∩ R A₀) := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Finset.mem_coe, exists_prop]
    refine ⟨fun h => ⟨fam ω, hfam ω, rfl, h⟩, ?_⟩
    rintro ⟨A₀, -, he, hr⟩
    exact he ▸ hr
  rw [hcover]
  exact Finset.measurableSet_biUnion _ (fun A₀ _ => (hfamMeas A₀).inter (hR A₀))

lemma measurableSet_cutCorrect (O : Oracle μ S) (lo hi : ℕ) (A₀ : Finset S) (p : S) :
    MeasurableSet {ω | ¬ cutCorrect O lo hi A₀ p ω} :=
  noiseAlg_le O Set.univ _ (measurableSet_filter_pred_map O (T := Set.univ) (fun v => p * v)
    (by simp) (fun U => ¬ ((hi < Finset.card U → O.label p = 1)
      ∧ (Finset.card U ≤ lo → O.label p = 0))))

open scoped Classical in
/-- A worst case survives any selection the block `Q` makes, so long as the scored event is
decided by a block `R` disjoint from it. -/
theorem selection_block_bound (O : Oracle μ S) (R Q : Finset S)
    (hdisj : Disjoint (↑R : Set S) (↑Q : Set S))
    {β : Type*} [DecidableEq β] (T : Finset β) (t₀ : β) (ht₀ : t₀ ∈ T)
    (sel : Ω → β) (hsel : ∀ ω, sel ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → sel ω = sel ω')
    (Bad : β → Set Ω) (hmeas : ∀ t, MeasurableSet (Bad t))
    (hmeasR : ∀ t ∈ T, MeasurableSet[noiseAlg O ↑R] (Bad t))
    (E : ℝ) (hE : 0 ≤ E) (hbad : ∀ t ∈ T, μ.real (Bad t) ≤ E) :
    μ.real {ω | ω ∈ Bad (sel ω)} ≤ E := by
  classical
  set sel' : Ω → β := fun ω => if ω ∈ noiseClean O Q then sel ω else t₀ with hsel'def
  have hsel2 : ∀ ω, sel' ω ∈ T := by
    intro ω
    rw [hsel'def]
    by_cases hc : ω ∈ noiseClean O Q
    · simpa [hc] using hsel ω
    · simpa [hc] using ht₀
  have hsplit : ∀ t, {ω | sel' ω = t}
      = ({ω | sel ω = t} ∩ noiseClean O Q) ∪ (if t = t₀ then (noiseClean O Q)ᶜ else ∅) := by
    intro t
    ext ω
    by_cases hc : ω ∈ noiseClean O Q <;> by_cases he : t = t₀ <;>
      simp [hsel'def, hc, he, Set.mem_setOf_eq, eq_comm (a := t₀)]
  have hmeasSelQ : ∀ t, MeasurableSet[noiseAlg O ↑Q] {ω | sel' ω = t} := by
    intro t
    rw [hsplit t]
    refine MeasurableSet.union (measurableSet_side_clean O Q sel hcongr t) ?_
    split_ifs
    · exact (measurableSet_noiseClean O Q).compl
    · exact (noiseAlg O ↑Q).measurableSet_empty
  have hmeasSel : ∀ t, MeasurableSet {ω | sel' ω = t} := fun t =>
    noiseAlg_le O ↑Q _ (hmeasSelQ t)
  have hindep : ∀ t ∈ T,
      μ.real ({ω | sel' ω = t} ∩ Bad t) = μ.real {ω | sel' ω = t} * μ.real (Bad t) := by
    intro t ht
    have hI := (indep_noiseAlg O hdisj.symm).indepSet_of_measurableSet (hmeasSelQ t)
      (hmeasR t ht)
    have := hI.measure_inter_eq_mul
    simp only [measureReal_def, this, ENNReal.toReal_mul]
  have hmain := measureReal_selection_le (μ := μ) T sel' hsel2 hmeasSel Bad hmeas E
    hindep hbad hE
  have hsub : {ω | ω ∈ Bad (sel ω)} ⊆ {ω | ω ∈ Bad (sel' ω)} ∪ (noiseClean O Q)ᶜ := by
    intro ω hω
    by_cases hc : ω ∈ noiseClean O Q
    · refine Or.inl ?_
      change ω ∈ Bad (sel' ω)
      rw [hsel'def]
      simp only [hc, if_pos]
      exact hω
    · exact Or.inr hc
  calc μ.real {ω | ω ∈ Bad (sel ω)}
      ≤ μ.real ({ω | ω ∈ Bad (sel' ω)} ∪ (noiseClean O Q)ᶜ) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ μ.real {ω | ω ∈ Bad (sel' ω)} + μ.real (noiseClean O Q)ᶜ := measureReal_union_le _ _
    _ = μ.real {ω | ω ∈ Bad (sel' ω)} := by rw [noiseClean_ae O Q, add_zero]
    _ ≤ E := hmain

open scoped Classical in
/-- Hoeffding on a per-prefix failure count.  Each prefix's failure is decided by its own
block of strings, the blocks are disjoint, so the failures are independent. -/
theorem count_frac_hoeffding (O : Oracle μ S) (C : Finset S) (blk : S → Finset S)
    (hblk : ∀ p ∈ C, ∀ q ∈ C, p ≠ q → Disjoint (blk p) (blk q))
    (Bad : S → Set Ω) (hmeasB : ∀ p, MeasurableSet[noiseAlg O ↑(blk p)] (Bad p))
    (E l : ℝ) (hEl : E ≤ l) (hper : ∀ p ∈ C, μ.real (Bad p) ≤ E) :
    μ.real {ω | l * (C.card : ℝ) < ((C.filter (fun p => ω ∈ Bad p)).card : ℝ)}
      ≤ Real.exp (-2 * (C.card : ℝ) * (l - E) ^ 2) := by
  classical
  set Y : {p // p ∈ C} → Ω → ℝ := fun i => (Bad i.val).indicator (fun _ => (1 : ℝ)) with hY
  have hmeasA : ∀ p, MeasurableSet (Bad p) := fun p => noiseAlg_le O _ _ (hmeasB p)
  have hind : iIndepFun Y μ := by
    refine iIndepFun_blocks (X := O.noise) O.noise_meas O.noise_indep
      (fun i : {p // p ∈ C} => blk i.val) ?_ Y (fun i => ?_)
    · intro a b hab
      exact hblk a.val a.property b.val b.property (fun h => hab (Subtype.ext h))
    · have hsup : (⨆ w ∈ blk i.val, MeasurableSpace.comap (O.noise w) inferInstance)
          = noiseAlg O ↑(blk i.val) := by
        unfold noiseAlg
        exact iSup_congr (fun w => by simp)
      rw [hsup]
      exact measurable_const.indicator (hmeasB i.val)
  have hIcc : ∀ i, ∀ᵐ ω ∂μ, Y i ω ∈ Set.Icc (0 : ℝ) 1 := fun i =>
    Filter.Eventually.of_forall (fun ω => by
      by_cases h : ω ∈ Bad i.val <;> simp [hY, h])
  have hmean : ∑ i ∈ (Finset.univ : Finset {p // p ∈ C}), μ[Y i]
      ≤ ((Finset.univ : Finset {p // p ∈ C}).card : ℝ) * E := by
    calc ∑ i ∈ (Finset.univ : Finset {p // p ∈ C}), μ[Y i]
        = ∑ i ∈ (Finset.univ : Finset {p // p ∈ C}), μ.real (Bad i.val) :=
          Finset.sum_congr rfl (fun i _ => integral_indicator_one (hmeasA i.val))
      _ ≤ ∑ _i ∈ (Finset.univ : Finset {p // p ∈ C}), E :=
          Finset.sum_le_sum (fun i _ => hper i.val i.property)
      _ = ((Finset.univ : Finset {p // p ∈ C}).card : ℝ) * E := by
          rw [Finset.sum_const, nsmul_eq_mul]
  have h := sumUpper_le Y Finset.univ E (l - E)
    (fun i => (measurable_const.indicator (hmeasA i.val)).aemeasurable) hind hIcc hmean
    (by linarith)
  have hcard : ((Finset.univ : Finset {p // p ∈ C}).card : ℝ) = (C.card : ℝ) := by
    simp
  rw [hcard] at h
  refine le_trans (measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)) h
  change (C.card : ℝ) * (E + (l - E)) ≤ ∑ i : {p // p ∈ C}, Y i ω
  have hsum : ∑ i : {p // p ∈ C}, Y i ω = ((C.filter (fun p => ω ∈ Bad p)).card : ℝ) := by
    rw [hY, Finset.sum_coe_sort C (fun p => (Bad p).indicator (fun _ => (1 : ℝ)) ω)]
    simp [Set.indicator_apply, Finset.sum_boole]
  rw [hsum, show E + (l - E) = l by ring]
  have h' : l * (C.card : ℝ) < ((C.filter (fun p => ω ∈ Bad p)).card : ℝ) := hω
  linarith

open scoped Classical in
/-- The table's own vote concentrates: off an `exp (−2·#P·(b − E)²)` set, every `κ`-subset
of the pool votes within `#F·γ` of its mean at all but a `b` fraction of the table's
prefixes.  Each prefix reads its own block `p · F`, which flatness keeps apart from the
others, so the per-prefix deviations are independent.

Unlike `frac_selected_le` this says nothing about the family the table selects; it is
uniform over the families instead, which is what the selection's own reads cost. -/
theorem tableVote_unif_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (P Cd : Finset S) (hP : ∀ p ∈ P, p ∈ Pre) (κ : ℕ) (γ b : ℝ) (hγ : 0 ≤ γ)
    (hb : 2 * Real.exp (-2 * (κ : ℝ) * γ ^ 2) ≤ b) :
    μ.real {ω | ∃ F ∈ Cd.powersetCard κ, b * (P.card : ℝ)
        < ((P.filter (fun p => ¬ |(voteCount O.mq F p ω : ℝ)
            - ∑ v ∈ F, μ[O.mq (p * v)]| ≤ (F.card : ℝ) * γ)).card : ℝ)}
      ≤ (Cd.card.choose κ : ℝ)
        * Real.exp (-2 * (P.card : ℝ) * (b - 2 * Real.exp (-2 * (κ : ℝ) * γ ^ 2)) ^ 2) := by
  classical
  have hper : ∀ (F : Finset S) (p : S),
      μ.real {ω | ¬ |(voteCount O.mq F p ω : ℝ) - ∑ v ∈ F, μ[O.mq (p * v)]|
          ≤ (F.card : ℝ) * γ} ≤ 2 * Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
    intro F p
    have hup := sumUpper_le_total (fun v : S => O.mq (p * v)) F
      (∑ v ∈ F, μ[O.mq (p * v)]) γ (fun _ => (mq_meas O _).aemeasurable)
      (mq_indep_shift O p) (fun _ => mq_icc O _) le_rfl hγ
    have hlo := sumLower_le_total (fun v : S => O.mq (p * v)) F
      (∑ v ∈ F, μ[O.mq (p * v)]) γ (fun _ => (mq_meas O _).aemeasurable)
      (mq_indep_shift O p) (fun _ => mq_icc O _) le_rfl hγ
    have hsub : ∀ᵐ ω ∂μ, ω ∈ {ω | ¬ |(voteCount O.mq F p ω : ℝ)
          - ∑ v ∈ F, μ[O.mq (p * v)]| ≤ (F.card : ℝ) * γ} →
        ω ∈ ({ω | ∑ v ∈ F, μ[O.mq (p * v)] + (F.card : ℝ) * γ ≤ ∑ v ∈ F, O.mq (p * v) ω}
          ∪ {ω | ∑ v ∈ F, O.mq (p * v) ω
              ≤ ∑ v ∈ F, μ[O.mq (p * v)] - (F.card : ℝ) * γ}) := by
      filter_upwards [voteCount_eq_voteSum O F p] with ω heq hω
      have hω' : (F.card : ℝ) * γ ≤ |voteSum O F p ω - ∑ v ∈ F, μ[O.mq (p * v)]| := by
        rw [← heq]; exact le_of_lt (not_le.mp hω)
      rcases le_abs.mp hω' with h | h
      · exact Or.inl (show _ ≤ voteSum O F p ω by
          simp only [voteSum] at h ⊢; linarith)
      · exact Or.inr (show voteSum O F p ω ≤ _ by
          simp only [voteSum] at h ⊢; linarith)
    calc μ.real {ω | ¬ |(voteCount O.mq F p ω : ℝ) - ∑ v ∈ F, μ[O.mq (p * v)]|
            ≤ (F.card : ℝ) * γ}
        ≤ μ.real ({ω | ∑ v ∈ F, μ[O.mq (p * v)] + (F.card : ℝ) * γ
              ≤ ∑ v ∈ F, O.mq (p * v) ω}
            ∪ {ω | ∑ v ∈ F, O.mq (p * v) ω
              ≤ ∑ v ∈ F, μ[O.mq (p * v)] - (F.card : ℝ) * γ}) :=
          measureReal_le_of_ae_imp hsub
      _ ≤ _ + _ := measureReal_union_le _ _
      _ ≤ 2 * Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by linarith
  have hcount : ∀ F ∈ Cd.powersetCard κ,
      μ.real {ω | b * (P.card : ℝ) < ((P.filter (fun p => ¬ |(voteCount O.mq F p ω : ℝ)
            - ∑ v ∈ F, μ[O.mq (p * v)]| ≤ (F.card : ℝ) * γ)).card : ℝ)}
        ≤ Real.exp (-2 * (P.card : ℝ) * (b - 2 * Real.exp (-2 * (κ : ℝ) * γ ^ 2)) ^ 2) := by
    intro F hF
    have hFκ : F.card = κ := (Finset.mem_powersetCard.mp hF).2
    have h := count_frac_hoeffding O P (fun p => F.image (fun v => p * v)) ?_
      (fun p => {ω | ¬ |(voteCount O.mq F p ω : ℝ) - ∑ v ∈ F, μ[O.mq (p * v)]|
        ≤ (F.card : ℝ) * γ}) ?_
      (2 * Real.exp (-2 * (κ : ℝ) * γ ^ 2)) b hb (fun p _ => by rw [← hFκ]; exact hper F p)
    · simpa only [Set.mem_setOf_eq] using h
    · intro p hp q hq hpq
      rw [Finset.disjoint_left]
      intro z hz hz'
      obtain ⟨v, -, rfl⟩ := Finset.mem_image.1 hz
      obtain ⟨v', -, hv'⟩ := Finset.mem_image.1 hz'
      exact hpq (hflat p (hP p hp) q (hP q hq) v v' hv'.symm)
    · intro p
      exact measurableSet_filter_pred_map O (fun v => p * v)
        (fun v hv => Finset.mem_coe.2 (Finset.mem_image_of_mem _ hv))
        (fun W => ¬ |((W.card : ℕ) : ℝ) - ∑ v ∈ F, μ[O.mq (p * v)]| ≤ (F.card : ℝ) * γ)
  have hunion : {ω | ∃ F ∈ Cd.powersetCard κ, b * (P.card : ℝ)
        < ((P.filter (fun p => ¬ |(voteCount O.mq F p ω : ℝ)
            - ∑ v ∈ F, μ[O.mq (p * v)]| ≤ (F.card : ℝ) * γ)).card : ℝ)}
      = ⋃ F ∈ Cd.powersetCard κ, {ω | b * (P.card : ℝ)
        < ((P.filter (fun p => ¬ |(voteCount O.mq F p ω : ℝ)
            - ∑ v ∈ F, μ[O.mq (p * v)]| ≤ (F.card : ℝ) * γ)).card : ℝ)} := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, exists_prop]
  rw [hunion]
  refine le_trans (measureReal_biUnion_finset_le _ _)
    (le_trans (Finset.sum_le_sum hcount) ?_)
  rw [Finset.sum_const, nsmul_eq_mul, Finset.card_powersetCard]

open scoped Classical in
/-- A per-prefix failure count on the certification sample, for the family the table
selects, off an `exp (−2·#C·(l − E)²)` set.  Given the family, the vote at each sample prefix
reads its own block `p · A₀`, and flatness keeps those blocks apart. -/
theorem frac_selected_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hCPre : ∀ p ∈ C, p ∈ Pre)
    (hPC : Disjoint P C)
    (T : Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (Ev : Finset S → S → Ω → Prop)
    (hEv : ∀ (A₀ : Finset S) (p : S) (U : Set S), (∀ v ∈ A₀, p * v ∈ U) →
      MeasurableSet[noiseAlg O U] {ω | Ev A₀ p ω})
    (E l : ℝ) (hE : 0 ≤ E) (hEl : E ≤ l)
    (hbad : ∀ p ∈ C, ∀ A₀ ∈ T, μ.real {ω | Ev A₀ p ω} ≤ E) :
    μ.real {ω | l * (C.card : ℝ) < ((C.filter (fun p => Ev (fam ω) p ω)).card : ℝ)}
      ≤ Real.exp (-2 * (C.card : ℝ) * (l - E) ^ 2) := by
  classical
  set R : Finset S := C.biUnion (fun p => cands.image (fun v => p * v)) with hR
  have hdisjR : Disjoint (↑R : Set S) (↑(readSet P cands) : Set S) := by
    rw [Finset.disjoint_coe, hR, Finset.disjoint_biUnion_left]
    intro p hp
    exact Finset.disjoint_coe.1 (disjoint_image_readSet hflat hP (hCPre p hp)
      (Finset.disjoint_right.1 hPC hp))
  set Bad : Finset S → Set Ω := fun A₀ =>
    {ω | l * (C.card : ℝ) < ((C.filter (fun p => Ev A₀ p ω)).card : ℝ)} with hBad
  have hmeasU : ∀ A₀, MeasurableSet (Bad A₀) := fun A₀ =>
    noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O (fun p ω => Ev A₀ p ω)
      (fun p _ => hEv A₀ p Set.univ (fun _ _ => Set.mem_univ _))
      (fun U => l * (C.card : ℝ) < (U.card : ℝ)))
  have hmeasR : ∀ A₀ ∈ T, MeasurableSet[noiseAlg O ↑R] (Bad A₀) := fun A₀ hA₀ =>
    measurableSet_filter_pred' O (fun p ω => Ev A₀ p ω)
      (fun p hp => hEv A₀ p ↑R (fun v hv => Finset.mem_coe.2 (Finset.mem_biUnion.2
        ⟨p, hp, Finset.mem_image_of_mem _ (hTC A₀ hA₀ hv)⟩)))
      (fun U => l * (C.card : ℝ) < (U.card : ℝ))
  have hfixed : ∀ A₀ ∈ T, μ.real (Bad A₀) ≤ Real.exp (-2 * (C.card : ℝ) * (l - E) ^ 2) := by
    intro A₀ hA₀
    have h := count_frac_hoeffding O C (fun p => A₀.image (fun v => p * v)) ?_
      (fun p => {ω | Ev A₀ p ω})
      (fun p => hEv A₀ p _ (fun v hv => Finset.mem_coe.2 (Finset.mem_image_of_mem _ hv)))
      E l hEl (fun p hp => hbad p hp A₀ hA₀)
    · exact le_of_eq_of_le (by rfl) h
    · intro p hp q hq hpq
      rw [Finset.disjoint_left]
      intro z hz hz'
      obtain ⟨v, -, rfl⟩ := Finset.mem_image.1 hz
      obtain ⟨v', -, hv'⟩ := Finset.mem_image.1 hz'
      exact hpq (hflat p (hCPre p hp) q (hCPre q hq) v v' hv'.symm)
  exact selection_block_bound O R (readSet P cands) hdisjR T t₀ ht₀ fam hfam hcongr Bad
    hmeasU hmeasR _ (Real.exp_nonneg _) hfixed

open scoped Classical in
/-- The indecision rate on the certification sample is below the limit, off a Hoeffding
tail. -/
theorem indecision_frac_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hCPre : ∀ p ∈ C, p ∈ Pre)
    (hPC : Disjoint P C) (lo ha : ℕ)
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E l : ℝ) (hE : 0 ≤ E) (hEl : E ≤ l)
    (hbad : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p → μ.real {ω | ¬ decided O.mq lo ha A₀ p ω} ≤ E) :
    μ.real {ω | l * (C.card : ℝ)
        < ((C.filter (fun p => fam ω ∈ good p ∧ ¬ decided O.mq lo ha (fam ω) p ω)).card : ℝ)}
      ≤ Real.exp (-2 * (C.card : ℝ) * (l - E) ^ 2) := by
  classical
  have h := frac_selected_le hflat O P cands C hP hCPre hPC T t₀ ht₀ hTC fam hfam hcongr
    (fun A₀ p ω => A₀ ∈ good p ∧ ¬ decided O.mq lo ha A₀ p ω) (fun A₀ p U hU => by
      by_cases hg : A₀ ∈ good p
      · have hset : {ω | A₀ ∈ good p ∧ ¬ decided O.mq lo ha A₀ p ω}
            = {ω | ¬ decided O.mq lo ha A₀ p ω} := by ext ω; simp [hg]
        rw [hset]
        exact measurableSet_filter_pred_map O (T := U) (fun v => p * v) hU
          (fun W => ¬ (ha < Finset.card W ∨ Finset.card W ≤ lo))
      · simp [hg])
    E l hE hEl (fun p hp A₀ hA₀ => by
      by_cases hg : A₀ ∈ good p
      · simpa [hg] using hbad p hp A₀ hA₀ hg
      · simpa [hg] using hE)
  convert h using 8

open scoped Classical in
/-- The same for the rate at which the cut is wrong. -/
theorem miscut_frac_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hCPre : ∀ p ∈ C, p ∈ Pre)
    (hPC : Disjoint P C) (lo ha : ℕ)
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E l : ℝ) (hE : 0 ≤ E) (hEl : E ≤ l)
    (hbad : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p → μ.real {ω | ¬ cutCorrect O lo ha A₀ p ω} ≤ E) :
    μ.real {ω | l * (C.card : ℝ)
        < ((C.filter (fun p => fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω)).card : ℝ)}
      ≤ Real.exp (-2 * (C.card : ℝ) * (l - E) ^ 2) := by
  classical
  have h := frac_selected_le hflat O P cands C hP hCPre hPC T t₀ ht₀ hTC fam hfam hcongr
    (fun A₀ p ω => A₀ ∈ good p ∧ ¬ cutCorrect O lo ha A₀ p ω) (fun A₀ p U hU => by
      by_cases hg : A₀ ∈ good p
      · have hset : {ω | A₀ ∈ good p ∧ ¬ cutCorrect O lo ha A₀ p ω}
            = {ω | ¬ cutCorrect O lo ha A₀ p ω} := by ext ω; simp [hg]
        rw [hset]
        exact measurableSet_filter_pred_map O (T := U) (fun v => p * v) hU
          (fun W => ¬ ((ha < Finset.card W → O.label p = 1)
            ∧ (Finset.card W ≤ lo → O.label p = 0)))
      · simp [hg])
    E l hE hEl (fun p hp A₀ hA₀ => by
      by_cases hg : A₀ ∈ good p
      · simpa [hg] using hbad p hp A₀ hA₀ hg
      · simpa [hg] using hE)
  convert h using 8

open scoped Classical in
/-- At a fixed table, the state returns.  Each failure count splits into the prefixes
the family is light for — bounded by `frac_selected_le` — and the heavy ones, whose number is
itself a fraction.  Lightness at a *fresh* prefix only holds for most of them, never all, so
carrying it as a per-prefix condition rather than a global guard is what makes the argument
available at all.

The certification sample's class counts are what the sides are measured against — an
all-accepting population leaves the reject side empty and nothing can admit — so they stay
as hypotheses about the draws rather than becoming a knob. -/
theorem ret_at_whp {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hCPre : ∀ p ∈ C, p ∈ Pre)
    (hPC : Disjoint P C) (Q : Finset S) (hQsup : readSet P cands ⊆ Q)
    (hdisjQ : Disjoint (↑C : Set S) (↑Q : Set S))
    (lo hi : ℕ) (α τ l lcut : ℝ) (n₀ nlo : ℕ)
    (hnloC : (nlo : ℝ) ≤ (C.card : ℝ) - 2 * l * (C.card : ℝ))
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (hQ : ∀ ω, ∀ p ∈ C, ∀ v ∈ fam ω, p * v ∈ Q)
    (E : ℝ) (hE : 0 ≤ E) (hElcut : E ≤ lcut) (hlcl : lcut ≤ l)
    (hτ : 0 ≤ τ) (hsig : O.η ≤ 1 / 2)
    (hdec : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p →
      μ.real {ω | ¬ decided O.mq lo (hi - 1) A₀ p ω} ≤ E)
    (hcut : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p →
      μ.real {ω | ¬ cutCorrect O lo (hi - 1) A₀ p ω} ≤ E)
    (hga : ∀ n : ℕ, nlo ≤ n → n ≤ C.card →
      (n : ℝ) * (1 / 2 + τ + τ)
        ≤ (n : ℝ) * (1 - O.η) - (1 - O.η) * (2 * lcut * (C.card : ℝ)))
    (hα : Real.exp (-2 * (n₀ : ℝ) * τ ^ 2) ≤ α) :
    μ.real {ω | ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) ≤ lcut * (C.card : ℝ)
        ∧ ¬ ((((C.filter (fun p => ¬ decided O.mq lo (hi - 1) (fam ω) p ω)).card : ℝ)
                ≤ 2 * l * (C.card : ℝ))
            ∧ admitted O.mq lo hi n₀ α (fam ω) C ω)}
      ≤ Real.exp (-2 * (C.card : ℝ) * (l - E) ^ 2)
        + (Real.exp (-2 * (C.card : ℝ) * (lcut - E) ^ 2)
          + 2 * Real.exp (-2 * (nlo : ℝ) * τ ^ 2)) := by
  classical
  have hsplit : ∀ (ω : Ω) (Q : S → Prop) [DecidablePred Q],
      (C.filter (fun p => Q p)).card
        ≤ (C.filter (fun p => fam ω ∈ good p ∧ Q p)).card
          + (C.filter (fun p => fam ω ∉ good p)).card := by
    intro ω Q _
    refine le_trans (Finset.card_le_card ?_) (Finset.card_union_le _ _)
    intro p hp
    obtain ⟨hpC, hq⟩ := Finset.mem_filter.1 hp
    by_cases hg : fam ω ∈ good p
    · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpC, hg, hq⟩)
    · exact Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hpC, hg⟩)
  have hsub : {ω | ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) ≤ lcut * (C.card : ℝ)
      ∧ ¬ ((((C.filter (fun p => ¬ decided O.mq lo (hi - 1) (fam ω) p ω)).card : ℝ)
              ≤ 2 * l * (C.card : ℝ))
          ∧ admitted O.mq lo hi n₀ α (fam ω) C ω)}
      ⊆ {ω | l * (C.card : ℝ)
            < ((C.filter (fun p => fam ω ∈ good p
                ∧ ¬ decided O.mq lo (hi - 1) (fam ω) p ω)).card : ℝ)}
        ∪ ({ω | lcut * (C.card : ℝ)
              < ((C.filter (fun p => fam ω ∈ good p
                  ∧ ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ)}
          ∪ {ω | (((C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ)
                ≤ 2 * lcut * (C.card : ℝ))
              ∧ (((C.filter (fun p => ¬ decided O.mq lo (hi - 1) (fam ω) p ω)).card : ℝ)
                ≤ 2 * l * (C.card : ℝ))
              ∧ ¬ admitted O.mq lo hi n₀ α (fam ω) C ω}) := by
    rintro ω ⟨hheavy, hbad⟩
    by_cases hindL : ((C.filter (fun p => fam ω ∈ good p
        ∧ ¬ decided O.mq lo (hi - 1) (fam ω) p ω)).card : ℝ) ≤ l * (C.card : ℝ)
    · by_cases hmisL : ((C.filter (fun p => fam ω ∈ good p
          ∧ ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ) ≤ lcut * (C.card : ℝ)
      · have hind : (((C.filter (fun p => ¬ decided O.mq lo (hi - 1) (fam ω) p ω)).card : ℝ))
            ≤ 2 * l * (C.card : ℝ) := by
          have := hsplit ω (fun p => ¬ decided O.mq lo (hi - 1) (fam ω) p ω)
          have hc : (((C.filter (fun p => ¬ decided O.mq lo (hi - 1) (fam ω) p ω)).card : ℝ))
              ≤ ((C.filter (fun p => fam ω ∈ good p
                  ∧ ¬ decided O.mq lo (hi - 1) (fam ω) p ω)).card : ℝ)
                + ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) := by exact_mod_cast this
          have hscale : lcut * (C.card : ℝ) ≤ l * (C.card : ℝ) :=
            mul_le_mul_of_nonneg_right hlcl (Nat.cast_nonneg _)
          linarith
        have hmis : (((C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ))
            ≤ 2 * lcut * (C.card : ℝ) := by
          have := hsplit ω (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)
          have hc : (((C.filter (fun p => ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ))
              ≤ ((C.filter (fun p => fam ω ∈ good p
                  ∧ ¬ cutCorrect O lo (hi - 1) (fam ω) p ω)).card : ℝ)
                + ((C.filter (fun p => fam ω ∉ good p)).card : ℝ) := by exact_mod_cast this
          linarith
        have hadm : ¬ admitted O.mq lo hi n₀ α (fam ω) C ω := fun h => hbad ⟨hind, h⟩
        exact Or.inr (Or.inr ⟨hmis, hind, hadm⟩)
      · exact Or.inr (Or.inl (not_le.1 hmisL))
    · exact Or.inl (not_le.1 hindL)
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) (add_le_add ?_ ?_)
  · exact indecision_frac_le hflat O P cands C hP hCPre hPC lo (hi - 1) T good t₀ ht₀ hTC
      fam hfam hcongr E l hE (le_trans hElcut hlcl) hdec
  · refine le_trans (measureReal_union_le _ _) (add_le_add ?_ ?_)
    · exact miscut_frac_le hflat O P cands C hP hCPre hPC lo (hi - 1) T good t₀ ht₀ hTC
        fam hfam hcongr E lcut hE hElcut hcut
    · exact admitted_whp O C Q hdisjQ lo hi α τ (2 * lcut * (C.card : ℝ))
        (2 * l * (C.card : ℝ)) n₀ nlo
        fam hQ (fun ω ω' h => hcongr ω ω' (fun w hw => h w (hQsup hw)))
        hτ hsig hnloC hga hα

lemma measurableSet_screenCount_pred (O : Oracle μ S) (P : Finset S) (v : S)
    (Q : ℕ → Prop) :
    MeasurableSet[noiseAlg O Set.univ]
      {ω | Q (screenCount O.mq P v ω)} := by
  have hPr : ∀ p ∈ P, MeasurableSet[noiseAlg O Set.univ]
      {ω | ¬ ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1))} := by
    intro p _
    have h1 := measurableSet_mq_eq_one O (T := Set.univ) (w := p * v) (Set.mem_univ _)
    have h0 := measurableSet_mq_eq_one O (T := Set.univ) (w := p) (Set.mem_univ _)
    have hiff : {ω | (O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)}
        = ({ω | O.mq (p * v) ω = 1} ∩ {ω | O.mq p ω = 1})
          ∪ ({ω | O.mq (p * v) ω = 1}ᶜ ∩ {ω | O.mq p ω = 1}ᶜ) := by
      ext ω
      by_cases ha : O.mq (p * v) ω = 1 <;> by_cases hb : O.mq p ω = 1 <;> simp [ha, hb]
    have : MeasurableSet[noiseAlg O Set.univ] {ω | (O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)} := by
      rw [hiff]
      exact ((h1.inter h0).union (h1.compl.inter h0.compl))
    exact this.compl
  exact measurableSet_filter_pred' O (fun p ω => ¬ ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)))
      hPr (fun U => Q U.card)

open scoped Classical in
lemma measurableSet_screenBase_le (O : Oracle μ S) (P cands : Finset S) (n : ℕ) :
    MeasurableSet[noiseAlg O Set.univ] {ω | screenBase O.mq P cands ω ≤ n} := by
  classical
  by_cases hne : (cands.erase 1).Nonempty
  · have hrw : {ω | screenBase O.mq P cands ω ≤ n}
        = ⋃ w ∈ cands.erase 1, {ω | screenCount O.mq P w ω ≤ n} := by
      ext ω
      simp only [screenBase, dif_pos hne, Set.mem_setOf_eq, Set.mem_iUnion, Finset.mem_coe,
        exists_prop]
      exact Finset.inf'_le_iff (f := fun w => screenCount O.mq P w ω) hne
    rw [hrw]
    exact MeasurableSet.biUnion (cands.erase 1).countable_toSet
      (fun w _ => measurableSet_screenCount_pred O P w (fun c => c ≤ n))
  · have hrw : {ω | screenBase O.mq P cands ω ≤ n} = Set.univ := by
      ext ω; simp [screenBase, dif_neg hne]
    rw [hrw]; exact MeasurableSet.univ

open scoped Classical in
lemma measurableSet_le_screenBase (O : Oracle μ S) (P cands : Finset S) (n : ℕ) :
    MeasurableSet[noiseAlg O Set.univ] {ω | n ≤ screenBase O.mq P cands ω} := by
  classical
  by_cases hne : (cands.erase 1).Nonempty
  · have hrw : {ω | n ≤ screenBase O.mq P cands ω}
        = ⋂ w ∈ cands.erase 1, {ω | n ≤ screenCount O.mq P w ω} := by
      ext ω
      simp only [screenBase, dif_pos hne, Set.mem_setOf_eq, Set.mem_iInter, Finset.mem_coe]
      exact Finset.le_inf'_iff (f := fun w => screenCount O.mq P w ω) hne
    rw [hrw]
    exact MeasurableSet.biInter (cands.erase 1).countable_toSet
      (fun w _ => measurableSet_screenCount_pred O P w (fun c => n ≤ c))
  · have hrw : {ω | n ≤ screenBase O.mq P cands ω} = {ω | n ≤ 0} := by
      ext ω; simp [screenBase, dif_neg hne]
    rw [hrw]
    by_cases h : n ≤ 0
    · simpa [h] using MeasurableSet.univ (α := Ω) (m := noiseAlg O Set.univ)
    · simpa [h] using MeasurableSet.empty (α := Ω) (m := noiseAlg O Set.univ)

open scoped Classical in
/-- The screen's own comparison, now relative to the pool's floor. -/
lemma measurableSet_screenRate (O : Oracle μ S) (P cands : Finset S) (v : S) (sc scd : ℕ) :
    MeasurableSet[noiseAlg O Set.univ]
      {ω | scd * screenCount O.mq P v ω
        ≤ scd * screenBase O.mq P cands ω + sc * P.card} := by
  classical
  have hrw : {ω | scd * screenCount O.mq P v ω
        ≤ scd * screenBase O.mq P cands ω + sc * P.card}
      = ⋃ n : ℕ,
          (({ω | screenBase O.mq P cands ω ≤ n} ∩ {ω | n ≤ screenBase O.mq P cands ω})
            ∩ {ω | scd * screenCount O.mq P v ω ≤ scd * n + sc * P.card}) := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff]
    constructor
    · intro h
      exact ⟨screenBase O.mq P cands ω, ⟨le_rfl, le_rfl⟩, h⟩
    · rintro ⟨n, ⟨h1, h2⟩, h3⟩
      have hb : screenBase O.mq P cands ω = n := le_antisymm h1 h2
      rw [hb]; exact h3
  rw [hrw]
  refine MeasurableSet.iUnion (fun n => ?_)
  exact ((measurableSet_screenBase_le O P cands n).inter
    (measurableSet_le_screenBase O P cands n)).inter
    (measurableSet_screenCount_pred O P v (fun c => scd * c ≤ scd * n + sc * P.card))

lemma measurableSet_screenRate' (O : Oracle μ S) (P cands : Finset S) (v : S) (sc scd : ℕ) :
    MeasurableSet {ω | scd * screenCount O.mq P v ω
      ≤ scd * screenBase O.mq P cands ω + sc * P.card} :=
  noiseAlg_le O Set.univ _ (measurableSet_screenRate O P cands v sc scd)

/-- `clusterAt` with the draws fixed: the family is a function of the noise alone. -/
noncomputable def clusterOf (O : Oracle μ S) (cn cd sc scd : ℕ) (P cands : Finset S) (k : ℕ)
    (ω : Ω) : Finset S :=
  clusterAround O.mq cn cd P (screened O.mq sc scd P cands ω) ω k

lemma clusterAt_eq_clusterOf (O : Oracle μ S) (populations : Finset J) (B : State)
    (x : Run Ω S J) :
    clusterAt O.mq populations x B
      = clusterOf O B.cn B.cd B.sc B.scd (prefixesAt populations B.npref x) (poolAt B.nsuff x) B.k (oracleNoise x) :=
  rfl

lemma clusterOf_subset (O : Oracle μ S) (cn cd sc scd : ℕ) (P cands : Finset S) (k : ℕ) (ω : Ω)
    (hone : (1 : S) ∈ cands) : clusterOf O cn cd sc scd P cands k ω ⊆ cands :=
  fun v hv => screened_subset O sc scd P cands ω
    (clusterAround_subset O cn cd P _ ω k (one_mem_screened O sc scd P cands ω hone) hv)

lemma clusterOf_congr_mq (O : Oracle μ S) (cn cd sc scd : ℕ) (P cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands) {ω ω' : Ω}
    (hbit : ∀ w ∈ readSet P cands, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    clusterOf O cn cd sc scd P cands k ω = clusterOf O cn cd sc scd P cands k ω' := by
  classical
  have hscr : screened O.mq sc scd P cands ω = screened O.mq sc scd P cands ω' :=
    Finset.filter_congr (fun v hv => by
      rw [screenCount_congr O hone hv hbit, screenBase_congr O hone hbit])
  have hsub : ∀ w ∈ readSet P (screened O.mq sc scd P cands ω'), (O.mq w ω = 1 ↔ O.mq w ω' = 1) := by
    intro w hw
    obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hw
    obtain ⟨hp, hvs⟩ := Finset.mem_product.1 hpv
    exact hbit _ (mem_readSet hp (screened_subset O sc scd P cands ω' hvs))
  unfold clusterOf
  rw [hscr, clusterAround_congr_mq O cn cd P _ k (one_mem_screened O sc scd P cands ω' hone) hsub]

open scoped Classical in
lemma measurableSet_clusterOf (O : Oracle μ S) (cn cd sc scd : ℕ) (P cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands) (A₀ : Finset S) :
    MeasurableSet {ω | clusterOf O cn cd sc scd P cands k ω = A₀} := by
  classical
  set Pred : Finset S → Prop := fun U => ∃ ω', (readSet P cands).filter
    (fun w => O.mq w ω' = 1) = U ∧ clusterOf O cn cd sc scd P cands k ω' = A₀ with hPred
  have hcov : {ω | clusterOf O cn cd sc scd P cands k ω = A₀}
      = {ω | Pred ((readSet P cands).filter (fun w => O.mq w ω = 1))} := by
    ext ω
    simp only [Set.mem_setOf_eq, hPred]
    refine ⟨fun h => ⟨ω, rfl, h⟩, ?_⟩
    rintro ⟨ω', hU, hA⟩
    refine (clusterOf_congr_mq O cn cd sc scd P cands k hone (fun w hw => ?_)).trans hA
    have := Finset.ext_iff.1 hU w
    simp only [Finset.mem_filter, hw, true_and] at this
    exact this.symm
  rw [hcov]
  exact noiseAlg_le O Set.univ _
    (measurableSet_filter_pred O (T := Set.univ) (by simp) Pred)

lemma measurable_badMassReal (O : Oracle μ S) (Dj : Measure S) (lo hi : ℕ) (A₀ : Finset S) :
    Measurable (fun ω => Dj.real {p | ¬ cutCorrect O lo hi A₀ p ω}) :=
  ENNReal.measurable_toReal.comp (measurable_badMass Dj
    (fun p => {ω | ¬ cutCorrect O lo hi A₀ p ω}) (fun p => measurableSet_cutCorrect O lo hi A₀ p))

/-- The runs at one state whose screen lets through a candidate the table says flips more
than `Δ` of it. -/
noncomputable def screenBad (O : Oracle μ S) (populations : Finset J) (B : State) (Δ : ℝ) :
    Set (Run Ω S J) :=
  {x | B.npref ≤ (prefixesAt populations B.npref x).card
    ∧ (∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
        ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0)
    ∧ ¬ ∀ v ∈ screenedAt O.mq populations B x,
      ¬ (Δ * ((prefixesAt populations B.npref x).card : ℝ)
        ≤ ∑ p ∈ prefixesAt populations B.npref x, O.flip v p)}

open scoped Classical in
lemma measurableSet_screenBad (O : Oracle μ S) (populations : Finset J) (B : State) (Δ : ℝ) :
    MeasurableSet (screenBad O populations B Δ) := by
  classical
  have hClean : MeasurableSet {x : Run Ω S J | ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
      ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0} := by
    have hR : ∀ P C : Finset S, MeasurableSet
        {_x : Run Ω S J | ∃ w₀ ∈ C.erase 1, ∀ p ∈ P, O.flip w₀ p = 0} := by
      intro P C
      by_cases h : ∃ w₀ ∈ C.erase 1, ∀ p ∈ P, O.flip w₀ p = 0
      · simpa [h] using (MeasurableSet.univ : MeasurableSet (Set.univ : Set (Run Ω S J)))
      · simpa [h] using (MeasurableSet.empty : MeasurableSet (∅ : Set (Run Ω S J)))
    exact measurableSet_of_run_data populations B _ hR
  have hRest : MeasurableSet {x : Run Ω S J
      | B.npref ≤ (prefixesAt populations B.npref x).card
        ∧ ¬ ∀ v ∈ screenedAt O.mq populations B x,
          ¬ (Δ * ((prefixesAt populations B.npref x).card : ℝ)
            ≤ ∑ p ∈ prefixesAt populations B.npref x, O.flip v p)} := by
    have hR : ∀ P C : Finset S, MeasurableSet
        {x : Run Ω S J | B.npref ≤ P.card ∧ ∃ v ∈ C, B.scd * screenCount O.mq P v (oracleNoise x)
              ≤ B.scd * screenBase O.mq P C (oracleNoise x) + B.sc * P.card
          ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p} := by
      intro P C
      by_cases hm : B.npref ≤ P.card
      swap
      · have : {x : Run Ω S J | B.npref ≤ P.card ∧ ∃ v ∈ C, B.scd * screenCount O.mq P v (oracleNoise x)
                ≤ B.scd * screenBase O.mq P C (oracleNoise x) + B.sc * P.card
            ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p} = (∅ : Set (Run Ω S J)) := by
          ext x; simp [hm]
        rw [this]; exact MeasurableSet.empty
      have hsimp : {x : Run Ω S J | B.npref ≤ P.card ∧ ∃ v ∈ C, B.scd * screenCount O.mq P v (oracleNoise x)
              ≤ B.scd * screenBase O.mq P C (oracleNoise x) + B.sc * P.card
          ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p}
          = {x : Run Ω S J | ∃ v ∈ C, B.scd * screenCount O.mq P v (oracleNoise x)
                ≤ B.scd * screenBase O.mq P C (oracleNoise x) + B.sc * P.card
            ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p} := by
        ext x; simp [hm]
      rw [hsimp]
      have hcov : {x : Run Ω S J | ∃ v ∈ C, B.scd * screenCount O.mq P v (oracleNoise x)
              ≤ B.scd * screenBase O.mq P C (oracleNoise x) + B.sc * P.card
          ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p}
          = ⋃ v ∈ C.filter (fun v => Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p),
              oracleNoise ⁻¹' {ω : Ω | B.scd * screenCount O.mq P v ω
                ≤ B.scd * screenBase O.mq P C ω + B.sc * P.card} := by
        ext x
        simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_preimage, Finset.mem_coe,
          Finset.mem_filter, exists_prop]
        exact ⟨fun ⟨v, hv, h1, h2⟩ => ⟨v, ⟨hv, h2⟩, h1⟩,
          fun ⟨v, ⟨hv, h2⟩, h1⟩ => ⟨v, hv, h1, h2⟩⟩
      rw [hcov]
      exact Finset.measurableSet_biUnion _
        (fun v _ => measurable_nz (measurableSet_screenRate' O P C v B.sc B.scd))
    have hrw : {x : Run Ω S J
        | B.npref ≤ (prefixesAt populations B.npref x).card
          ∧ ¬ ∀ v ∈ screenedAt O.mq populations B x,
            ¬ (Δ * ((prefixesAt populations B.npref x).card : ℝ)
              ≤ ∑ p ∈ prefixesAt populations B.npref x, O.flip v p)}
        = {x : Run Ω S J | x ∈ (fun P C => {x : Run Ω S J | B.npref ≤ P.card ∧ ∃ v ∈ C,
            B.scd * screenCount O.mq P v (oracleNoise x)
                ≤ B.scd * screenBase O.mq P C (oracleNoise x) + B.sc * P.card
              ∧ Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p})
          (prefixesAt populations B.npref x) (poolAt B.nsuff x)} := by
      ext x
      simp only [screened, screenedAt, Set.mem_setOf_eq, not_forall, Finset.mem_filter,
        not_not, exists_prop, and_congr_right_iff]
      intro _
      exact ⟨fun ⟨v, hv, h⟩ => ⟨v, hv.1, hv.2, h⟩, fun ⟨v, hv, h1, h2⟩ => ⟨v, ⟨hv, h1⟩, h2⟩⟩
    rw [hrw]
    exact measurableSet_of_run_data populations B _ hR
  have hsplit : screenBad O populations B Δ
      = {x : Run Ω S J | ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
          ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
        ∩ {x : Run Ω S J
          | B.npref ≤ (prefixesAt populations B.npref x).card
            ∧ ¬ ∀ v ∈ screenedAt O.mq populations B x,
              ¬ (Δ * ((prefixesAt populations B.npref x).card : ℝ)
                ≤ ∑ p ∈ prefixesAt populations B.npref x, O.flip v p)} := by
    ext x
    exact ⟨fun ⟨h0, hw, hb⟩ => ⟨hw, h0, hb⟩, fun ⟨hw, h0, hb⟩ => ⟨h0, hw, hb⟩⟩
  rw [hsplit]
  exact hClean.inter hRest

/-- The relative screen's soundness at a fixed table.  A candidate flipping a `Δ` mass sits
at least `Δ(1−2η)²` above the pool's floor, so it clears a cutoff of `Δ(1−2η)² − 2γ` only in
a tail — one `γ` for its own count, one for the floor, whose upper bound is what a clean
candidate `w₀` in the pool buys. -/
theorem screen_tail_rel {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S) {cn cd : ℕ}
    (hcd : cn < cd) {P cands : Finset S} (hP : ∀ p ∈ P, p ∈ Pre) (v : S) (hv : v ≠ 1)
    (w₀ : S) (hw₀ : w₀ ∈ cands.erase 1) (hw₀clean : ∀ p ∈ P, O.flip w₀ p = 0)
    (Δ γ : ℝ) (sc scd : ℕ) (hγ : 0 ≤ γ) (hsig : O.η ≤ 1 / 2) (hscd : 0 < scd)
    (hflip : Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p)
    (hsc : (sc : ℝ) ≤ (scd : ℝ) * (Δ * (1 - 2 * O.η) ^ 2 - 2 * γ)) :
    μ.real {ω | scd * screenCount O.mq P v ω
        ≤ scd * screenBase O.mq P cands ω + sc * P.card}
      ≤ 2 * Real.exp (-2 * (P.card : ℝ) * γ ^ 2) := by
  classical
  have hscdR : (0 : ℝ) < (scd : ℝ) := by exact_mod_cast hscd
  have hPc : (0 : ℝ) ≤ (P.card : ℝ) := Nat.cast_nonneg _
  have hsub : {ω | scd * screenCount O.mq P v ω
        ≤ scd * screenBase O.mq P cands ω + sc * P.card}
      ⊆ {ω | cleanLoss O P + (P.card : ℝ) * γ ≤ (screenCount O.mq P w₀ ω : ℝ)}
        ∪ {ω | (screenCount O.mq P v ω : ℝ)
            ≤ cleanLoss O P + (P.card : ℝ) * (Δ * (1 - 2 * O.η) ^ 2 - γ)} := by
    intro ω hω
    by_contra hnot
    simp only [Set.mem_union, not_or, Set.mem_setOf_eq, not_le] at hnot
    obtain ⟨hwU, hvL⟩ := hnot
    have hne : (cands.erase 1).Nonempty := ⟨w₀, hw₀⟩
    have hbase : (screenBase O.mq P cands ω : ℝ) ≤ (screenCount O.mq P w₀ ω : ℝ) := by
      have : screenBase O.mq P cands ω ≤ screenCount O.mq P w₀ ω := by
        rw [screenBase, dif_pos hne]
        exact Finset.inf'_le _ hw₀
      exact_mod_cast this
    have hωR : (scd : ℝ) * (screenCount O.mq P v ω : ℝ)
        ≤ (scd : ℝ) * (screenBase O.mq P cands ω : ℝ) + (sc : ℝ) * (P.card : ℝ) := by
      exact_mod_cast hω
    nlinarith [hsc, hbase, hwU, hvL, hscdR, hPc]
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) ?_
  have h1 := screenCount_upper hflat O hcd hP w₀ (Finset.ne_of_mem_erase hw₀) γ hγ hsig
    hw₀clean
  have h2 := screenCount_lower hflat O hcd hP v hv Δ γ hγ hsig hflip
  linarith [h1, h2]

/-- The relative screen's liveness.  A clean candidate's count is within `γ` of `cleanLoss`
and the floor is within `γ` below it, so `2γ` of cutoff admits it.  The floor's lower bound
is a union bound over the pool, which is where the `#cands` factor comes from. -/
theorem screen_pass_rel {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S) {cn cd : ℕ}
    (hcd : cn < cd) {P cands : Finset S} (hP : ∀ p ∈ P, p ∈ Pre) (v : S) (hv : v ≠ 1)
    (hvc : v ∈ cands)
    (γ : ℝ) (sc scd : ℕ) (hγ : 0 ≤ γ) (hsig : O.η ≤ 1 / 2) (hscd : 0 < scd)
    (hclean : ∀ p ∈ P, O.flip v p = 0)
    (hsc : (scd : ℝ) * (2 * γ) ≤ (sc : ℝ)) :
    μ.real {ω | ¬ (scd * screenCount O.mq P v ω
        ≤ scd * screenBase O.mq P cands ω + sc * P.card)}
      ≤ ((cands.card : ℝ) + 1) * Real.exp (-2 * (P.card : ℝ) * γ ^ 2) := by
  classical
  have hne : (cands.erase 1).Nonempty := ⟨v, Finset.mem_erase.2 ⟨hv, hvc⟩⟩
  have hscdR : (0 : ℝ) < (scd : ℝ) := by exact_mod_cast hscd
  have hPc : (0 : ℝ) ≤ (P.card : ℝ) := Nat.cast_nonneg _
  have hsub : {ω | ¬ (scd * screenCount O.mq P v ω
        ≤ scd * screenBase O.mq P cands ω + sc * P.card)}
      ⊆ {ω | cleanLoss O P + (P.card : ℝ) * γ ≤ (screenCount O.mq P v ω : ℝ)}
        ∪ ⋃ w ∈ cands.erase 1,
            {ω | (screenCount O.mq P w ω : ℝ) ≤ cleanLoss O P + (P.card : ℝ) * (0 * (1 - 2 * O.η) ^ 2 - γ)} := by
    intro ω hω
    by_contra hnot
    simp only [Set.mem_union, not_or, Set.mem_iUnion, not_exists, Set.mem_setOf_eq,
      not_le, exists_prop, not_and] at hnot
    obtain ⟨hvU, hBl⟩ := hnot
    have hbase : cleanLoss O P + (P.card : ℝ) * (0 * (1 - 2 * O.η) ^ 2 - γ)
        < (screenBase O.mq P cands ω : ℝ) := by
      rw [screenBase, dif_pos hne]
      obtain ⟨w, hw, hweq⟩ :=
        Finset.exists_mem_eq_inf' hne (fun w => screenCount O.mq P w ω)
      rw [hweq]
      exact hBl w hw
    have hωR : (scd : ℝ) * (screenBase O.mq P cands ω : ℝ) + (sc : ℝ) * (P.card : ℝ)
        < (scd : ℝ) * (screenCount O.mq P v ω : ℝ) := by exact_mod_cast not_le.1 hω
    nlinarith [hsc, hbase, hvU, hωR, hscdR, hPc]
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) ?_
  have h1 := screenCount_upper hflat O hcd hP v hv γ hγ hsig hclean
  have h2 : μ.real (⋃ w ∈ cands.erase 1,
        {ω | (screenCount O.mq P w ω : ℝ)
          ≤ cleanLoss O P + (P.card : ℝ) * (0 * (1 - 2 * O.η) ^ 2 - γ)})
      ≤ (cands.card : ℝ) * Real.exp (-2 * (P.card : ℝ) * γ ^ 2) := by
    refine le_trans (measureReal_biUnion_finset_le _ _) ?_
    have hper : ∀ w ∈ cands.erase 1,
        μ.real {ω | (screenCount O.mq P w ω : ℝ)
          ≤ cleanLoss O P + (P.card : ℝ) * (0 * (1 - 2 * O.η) ^ 2 - γ)}
          ≤ Real.exp (-2 * (P.card : ℝ) * γ ^ 2) :=
      fun w hw => screenCount_lower hflat O hcd hP w (Finset.ne_of_mem_erase hw) 0 γ hγ hsig
        (by rw [zero_mul]; exact Finset.sum_nonneg (fun p _ => by
          rcases O.flip_bit w p with h | h <;> rw [h] <;> norm_num))
    refine le_trans (Finset.sum_le_sum hper) ?_
    rw [Finset.sum_const, nsmul_eq_mul]
    have hcardle : ((cands.erase 1).card : ℝ) ≤ (cands.card : ℝ) := by
      exact_mod_cast Finset.card_erase_le
    exact mul_le_mul_of_nonneg_right hcardle (Real.exp_nonneg _)
  linarith [h1, h2]

/-- The screen rarely lets a badly-flipping candidate through.  One `screen_tail_rel` per
pool member, at a fixed table: the flip counts are deterministic once the draws are, so this
is a plain union bound and costs `M + 1`. -/
theorem measureReal_screenBad_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (B : State) (hcd : B.cn < B.cd)
    (hsig : O.η ≤ 1 / 2) (hmpos : 0 < B.npref) (Δ γ : ℝ) (hΔ : 0 < Δ) (hγ : 0 ≤ γ)
    (hscd : 0 < B.scd)
    (hsc : (B.sc : ℝ) ≤ (B.scd : ℝ) * (Δ * (1 - 2 * O.η) ^ 2 - 2 * γ)) :
    (runMeasure μ D Dsf).real (screenBad O populations B Δ)
      ≤ ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2) := by
  classical
  set E : ℝ := ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2) with hEdef
  have hE0 : 0 ≤ E := by positivity
  have hEnn : runMeasure μ D Dsf (screenBad O populations B Δ) ≤ ENNReal.ofReal E := by
    refine runMeasure_slice_le D Dsf _ (measurableSet_screenBad O populations B Δ) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp] with d hd
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.npref).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.nsuff).image (fun i => d.1.1 i)) with hCd
    have hP : ∀ q ∈ Pd, q ∈ Pre := by
      intro q hq
      obtain ⟨j', hj', hq'⟩ := Finset.mem_biUnion.1 hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hq'
      exact hd j' hj' i
    by_cases hcl : ∃ w₀ ∈ Cd.erase 1, ∀ p ∈ Pd, O.flip w₀ p = 0
    swap
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ screenBad O populations B Δ}
          = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
        rintro ⟨-, hw, -⟩
        exact hcl (by simpa [hPd, hCd, prefixesAt, prefixesOf, poolAt, suffixDraw,
          prefixDraw] using hw)
      rw [hsec]
      simpa using ENNReal.ofReal_le_ofReal hE0
    obtain ⟨w₀, hw₀, hw₀c⟩ := hcl
    by_cases hm : B.npref ≤ Pd.card
    · have hPpos : 0 < Pd.card := lt_of_lt_of_le hmpos hm
      have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ screenBad O populations B Δ}
          ⊆ ⋃ v ∈ Cd.filter (fun v => Δ * (Pd.card : ℝ) ≤ ∑ p ∈ Pd, O.flip v p),
              {ω : Ω | B.scd * screenCount O.mq Pd v ω ≤ B.scd * screenBase O.mq Pd Cd ω + B.sc * Pd.card} := by
        rintro ω ⟨-, -, hbad⟩
        simp only [not_forall, not_not] at hbad
        obtain ⟨v, hv, hflip⟩ := hbad
        obtain ⟨hvC, hvs⟩ := Finset.mem_filter.1 hv
        exact Set.mem_biUnion (Finset.mem_filter.2 ⟨hvC, hflip⟩) hvs
      refine le_trans (measure_mono hsec) (le_trans (measure_biUnion_finset_le _ _) ?_)
      have hper : ∀ v ∈ Cd.filter (fun v => Δ * (Pd.card : ℝ) ≤ ∑ p ∈ Pd, O.flip v p),
          μ {ω : Ω | B.scd * screenCount O.mq Pd v ω ≤ B.scd * screenBase O.mq Pd Cd ω + B.sc * Pd.card}
            ≤ ENNReal.ofReal (2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2)) := by
        intro v hv
        obtain ⟨-, hflip⟩ := Finset.mem_filter.1 hv
        have hv1 : v ≠ 1 := by
          intro h
          rw [h] at hflip
          have hz : ∑ p ∈ Pd, O.flip (1 : S) p = 0 := by
            refine Finset.sum_eq_zero (fun p _ => ?_)
            show O.label (p * 1) + O.label p - 2 * O.label (p * 1) * O.label p = 0
            rw [mul_one]
            rcases O.label_bit p with hl | hl <;> rw [hl] <;> ring
          rw [hz] at hflip
          exact absurd hflip (not_le.2 (mul_pos hΔ (by exact_mod_cast hPpos)))
        have hcardR : (B.npref : ℝ) ≤ (Pd.card : ℝ) := by exact_mod_cast hm
        have htail := screen_tail_rel hflat O hcd hP v hv1 w₀ hw₀ hw₀c
          Δ γ B.sc B.scd hγ hsig hscd hflip hsc
        rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
        refine ENNReal.ofReal_le_ofReal (le_trans htail ?_)
        have hmono : Real.exp (-2 * (Pd.card : ℝ) * γ ^ 2)
            ≤ Real.exp (-2 * (B.npref : ℝ) * γ ^ 2) := by
          refine Real.exp_le_exp.2 ?_
          nlinarith [sq_nonneg γ]
        linarith [hmono]
      refine le_trans (Finset.sum_le_sum hper) ?_
      rw [Finset.sum_const, nsmul_eq_mul, hEdef]
      refine le_trans (mul_le_mul' (le_refl _) (le_refl _)) ?_
      rw [← ENNReal.ofReal_natCast (Cd.filter _).card, ← ENNReal.ofReal_mul (by positivity)]
      refine ENNReal.ofReal_le_ofReal ?_
      have hCardR : ((Cd.filter (fun v => Δ * (Pd.card : ℝ) ≤ ∑ p ∈ Pd, O.flip v p)).card : ℝ)
          ≤ (B.nsuff : ℝ) + 1 := by
        have hrangeC : (Finset.range B.nsuff).card ≤ B.nsuff := by simp
        have hCard : Cd.card ≤ B.nsuff + 1 :=
          le_trans (Finset.card_insert_le _ _)
            (by simpa using le_trans Finset.card_image_le hrangeC)
        have := le_trans (Finset.card_filter_le
          Cd (fun v => Δ * (Pd.card : ℝ) ≤ ∑ p ∈ Pd, O.flip v p)) hCard
        exact_mod_cast this
      have hn0 : (0 : ℝ) ≤ (B.nsuff : ℝ) := Nat.cast_nonneg _
      have hex : (0 : ℝ) ≤ Real.exp (-2 * (B.npref : ℝ) * γ ^ 2) := Real.exp_nonneg _
      have hstep := mul_le_mul_of_nonneg_right hCardR
        (by positivity : (0:ℝ) ≤ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2))
      nlinarith [hstep, hn0, hex]
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ screenBad O populations B Δ} = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
        rintro ⟨h1, -⟩
        exact hm h1
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (screenBad O populations B Δ)).toReal
      ≤ (ENNReal.ofReal E).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = E := ENNReal.toReal_ofReal hE0

open scoped Classical in
/-- A block of i.i.d. draws hits a set about as often as its mass.  The lower tail, for
the suffix pool: a mass-`q` set is hit at least `M(q − t)` times. -/
lemma pi_hits_lower (Dsf : Measure S) [IsProbabilityMeasure Dsf] (M : ℕ) (W : Set S)
    (q t : ℝ) (hq : 0 ≤ q) (ht : 0 ≤ t) (hW : q ≤ Dsf.real W) :
    (Measure.pi fun _ : Fin M => Dsf).real
        {r : Fin M → S | (((Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W)).card : ℝ)
          ≤ (M : ℝ) * (q - t)}
      ≤ Real.exp (-2 * (M : ℝ) * t ^ 2) := by
  classical
  set ν : Measure (Fin M → S) := Measure.pi fun _ : Fin M => Dsf with hνdef
  have hWm : MeasurableSet W := measurableSet_of_countable W
  set ind : S → ℝ := W.indicator 1 with hinddef
  have hindm : Measurable ind := measurable_const.indicator hWm
  set X : Fin M → (Fin M → S) → ℝ := fun i r => ind (r i) with hXdef
  have hindep : iIndepFun X ν := iIndepFun_pi (fun _ => hindm.aemeasurable)
  have hmeas : ∀ i, AEMeasurable (X i) ν := fun i =>
    (hindm.comp (measurable_pi_apply _)).aemeasurable
  have hicc : ∀ i, ∀ᵐ r ∂ν, X i r ∈ Set.Icc (0 : ℝ) 1 := by
    intro i
    filter_upwards with r
    by_cases h : r i ∈ W
    · simp [hXdef, hinddef, Set.indicator_of_mem h]
    · simp [hXdef, hinddef, Set.indicator_of_notMem h]
  have hmean : ∀ i, ν[X i] = Dsf.real W := by
    intro i
    have hmp : MeasurePreserving (fun r : Fin M → S => r i) ν Dsf :=
      measurePreserving_eval (fun _ : Fin M => Dsf) i
    calc ν[X i] = ∫ s, ind s ∂Dsf := by
          rw [← hmp.map_eq,
            integral_map hmp.measurable.aemeasurable hindm.aestronglyMeasurable]
      _ = Dsf.real W := by rw [hinddef, integral_indicator_one hWm]
  have hsum : ((Finset.univ : Finset (Fin M)).card : ℝ) * q ≤ ∑ i, ν[X i] := by
    rw [Finset.sum_congr rfl (fun i _ => hmean i), Finset.sum_const, nsmul_eq_mul]
    have hc : (0 : ℝ) ≤ ((Finset.univ : Finset (Fin M)).card : ℝ) := Nat.cast_nonneg _
    nlinarith [hW]
  have hmain := sumLower_le X (Finset.univ : Finset (Fin M)) q t hmeas hindep hicc hsum ht
  have hcount : ∀ r : Fin M → S, ∑ i, X i r
      = (((Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W)).card : ℝ) := by
    intro r
    rw [← Finset.sum_filter_add_sum_filter_not (Finset.univ : Finset (Fin M))
      (fun i => r i ∈ W)]
    have h1 : ∑ i ∈ (Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W), X i r
        = ((((Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W)).card : ℝ)) := by
      have hone : ∀ i ∈ (Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W),
          X i r = (1 : ℝ) := by
        intro i hi
        simp [hXdef, hinddef, Set.indicator_of_mem (Finset.mem_filter.1 hi).2]
      rw [Finset.sum_congr rfl hone, Finset.sum_const, nsmul_eq_mul, mul_one]
    have h0 : ∑ i ∈ (Finset.univ : Finset (Fin M)).filter (fun i => ¬ (r i ∈ W)), X i r = 0 :=
      Finset.sum_eq_zero (fun i hi => by
        simp [hXdef, hinddef, Set.indicator_of_notMem (Finset.mem_filter.1 hi).2])
    rw [h1, h0, add_zero]
  have hcard : ((Finset.univ : Finset (Fin M)).card : ℝ) = (M : ℝ) := by simp
  refine le_trans (le_trans (measureReal_mono ?_ (measure_ne_top _ _)) hmain) ?_
  · intro r hr
    simp only [Set.mem_setOf_eq] at hr ⊢
    rw [hcount r, hcard]
    linarith [hr]
  · rw [hcard]

open scoped Classical in
/-- The pool is short of accept-preserving suffixes.  It reads the suffix draws and nothing
else, so it is one event for the whole ladder rather than one per rung. -/
noncomputable def apShort (O : Oracle μ S) (M : ℕ) (pAP t : ℝ) : Set (Run Ω S J) :=
  {x : Run Ω S J | (((Finset.univ : Finset (Fin M)).filter
      (fun i : Fin M => ∀ p, O.label (p * suffixDraw i.val x) = O.label p)).card : ℝ)
    ≤ (M : ℝ) * (pAP - t)}

open scoped Classical in
/-- The pool holds accept-preserving suffixes.  Findability says a draw is
accept-preserving with probability at least `pAP`, so `M` draws hold about `pAP·M` of them. -/
theorem measureReal_apShort_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S) (M : ℕ)
    (pAP t : ℝ) (hpAP0 : 0 ≤ pAP) (ht : 0 ≤ t)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p}) :
    (runMeasure μ D Dsf).real (apShort O M pAP t)
      ≤ Real.exp (-2 * (M : ℝ) * t ^ 2) := by
  classical
  rw [apShort]
  set W : Set S := {v | ∀ p, O.label (p * v) = O.label p} with hW
  have hmeasHits : Measurable (fun r : Fin M → S =>
      (((Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W)).card : ℝ)) := by
    have hWm : MeasurableSet W := (Set.to_countable _).measurableSet
    have hrw : (fun r : Fin M → S =>
        (((Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W)).card : ℝ))
        = fun r => ∑ i : Fin M, W.indicator (fun _ => (1 : ℝ)) (r i) := by
      funext r
      rw [Finset.card_filter, Nat.cast_sum]
      refine Finset.sum_congr rfl (fun i _ => ?_)
      by_cases h : r i ∈ W <;> simp [h]
    rw [hrw]
    exact Finset.measurable_sum _ (fun i _ =>
      (measurable_const.indicator hWm).comp (measurable_pi_apply i))
  have hmeasSet : MeasurableSet {r : Fin M → S |
      (((Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W)).card : ℝ)
        ≤ (M : ℝ) * (pAP - t)} := measurableSet_le hmeasHits measurable_const
  have hpre : {x : Run Ω S J |
      (((Finset.univ : Finset (Fin M)).filter
        (fun i => ∀ p, O.label (p * suffixDraw i.val x) = O.label p)).card : ℝ)
        ≤ (M : ℝ) * (pAP - t)}
      = (fun x : Run Ω S J => (fun i : Fin M => suffixDraw i.val x)) ⁻¹'
        {r : Fin M → S | (((Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W)).card : ℝ)
          ≤ (M : ℝ) * (pAP - t)} := rfl
  have hmeasBlock : Measurable (fun x : Run Ω S J => (fun i : Fin M => suffixDraw i.val x)) :=
    measurable_pi_lambda _ (fun i : Fin M => measurable_sfx i.val)
  rw [hpre, measureReal_def, Measure.map_apply hmeasBlock hmeasSet
    |>.symm.trans (congrArg (fun ν : Measure (Fin M → S) => ν _) (map_suffixBlock D Dsf M)),
    ← measureReal_def]
  exact pi_hits_lower Dsf M W pAP t hpAP0 ht hpAPBound

open scoped Classical in
/-- The runs where an accept-preserving candidate the pool holds is thrown out by the
screen.  Off this event the pool's accept-preserving draws all survive to be clustered. -/
noncomputable def screenFail (O : Oracle μ S) (populations : Finset J) (B : State) :
    Set (Run Ω S J) :=
  {x | B.npref ≤ (prefixesAt populations B.npref x).card
    ∧ ¬ ∀ v ∈ poolAt B.nsuff x, v ≠ 1 → (∀ p, O.label (p * v) = O.label p) →
      B.scd * screenCount O.mq (prefixesAt populations B.npref x) v (oracleNoise x)
        ≤ B.scd * screenBase O.mq (prefixesAt populations B.npref x) (poolAt B.nsuff x)
            (oracleNoise x)
          + B.sc * (prefixesAt populations B.npref x).card}

open scoped Classical in
lemma measurableSet_screenFail (O : Oracle μ S) (populations : Finset J) (B : State) :
    MeasurableSet (screenFail O populations B) := by
  classical
  have hR : ∀ P C : Finset S, MeasurableSet (if B.npref ≤ P.card then
      {x : Run Ω S J | ¬ ∀ v ∈ C, v ≠ 1 → (∀ p, O.label (p * v) = O.label p) →
        B.scd * screenCount O.mq P v (oracleNoise x)
          ≤ B.scd * screenBase O.mq P C (oracleNoise x) + B.sc * P.card} else ∅) := by
    intro P C
    split_ifs with hm
    · have hset : {x : Run Ω S J | ¬ ∀ v ∈ C, v ≠ 1 → (∀ p, O.label (p * v) = O.label p) →
          B.scd * screenCount O.mq P v (oracleNoise x)
            ≤ B.scd * screenBase O.mq P C (oracleNoise x) + B.sc * P.card}
          = oracleNoise ⁻¹' (⋃ v ∈ C.filter (fun v => v ≠ 1 ∧ ∀ p, O.label (p * v) = O.label p),
            {ω : Ω | ¬ (B.scd * screenCount O.mq P v ω ≤ B.scd * screenBase O.mq P C ω + B.sc * P.card)}) := by
        ext x
        simp only [Set.mem_setOf_eq, Set.mem_preimage, Set.mem_iUnion, Finset.mem_coe,
          Finset.mem_filter, exists_prop, not_forall]
        constructor
        · rintro ⟨v, hv, hv1, hap, hfail⟩
          exact ⟨v, ⟨hv, hv1, hap⟩, hfail⟩
        · rintro ⟨v, ⟨hv, hv1, hap⟩, hfail⟩
          exact ⟨v, hv, hv1, hap, hfail⟩
      rw [hset]
      exact measurable_nz (Finset.measurableSet_biUnion _
        (fun v _ => (measurableSet_screenRate' O P C v B.sc B.scd).compl))
    · exact MeasurableSet.empty
  have hrw : screenFail O populations B
      = {x : Run Ω S J | x ∈ (fun P C => if B.npref ≤ P.card then
          {x : Run Ω S J | ¬ ∀ v ∈ C, v ≠ 1 → (∀ p, O.label (p * v) = O.label p) →
            B.scd * screenCount O.mq P v (oracleNoise x)
            ≤ B.scd * screenBase O.mq P C (oracleNoise x) + B.sc * P.card} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)} := by
    ext x
    simp only [screenFail, Set.mem_setOf_eq]
    by_cases hm : B.npref ≤ (prefixesAt populations B.npref x).card
    · rw [if_pos hm]
      exact ⟨fun h => h.2, fun h => ⟨hm, h⟩⟩
    · rw [if_neg hm]
      exact ⟨fun h => absurd h.1 hm, fun h => absurd h (Set.notMem_empty x)⟩
  rw [hrw]
  exact measurableSet_of_run_data populations B _ hR

open scoped Classical in
/-- The pool holds a clean non-seed reference.  Findability puts accept-preserving draws in
it, and distinctness leaves at most one of them at the seed, so two suffice. -/
theorem noCleanRef_subset (O : Oracle μ S) (populations : Finset J) (B : State) (pAP t : ℝ)
    (hcount : (2 : ℝ) ≤ (B.nsuff : ℝ) * (pAP - t)) :
    {x : Run Ω S J | ¬ ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
        ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
      ⊆ apShort O B.nsuff pAP t
        ∪ {x | ¬ Function.Injective (fun i : Fin B.nsuff => suffixDraw i.val x)} := by
  classical
  set A : Set (Run Ω S J) := apShort O B.nsuff pAP t with hA
  set N : Set (Run Ω S J) :=
    {x | ¬ Function.Injective (fun i : Fin B.nsuff => suffixDraw i.val x)} with hN
  have hsub : {x : Run Ω S J | ¬ ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
      ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0} ⊆ A ∪ N := by
    intro x hx
    by_contra hnot
    simp only [Set.mem_union, not_or, hA, hN, apShort, Set.mem_setOf_eq, not_not] at hnot
    obtain ⟨hcnt, hinj⟩ := hnot
    set I : Finset (Fin B.nsuff) := (Finset.univ : Finset (Fin B.nsuff)).filter
      (fun i => ∀ p, O.label (p * suffixDraw i.val x) = O.label p) with hI
    have hI2 : 2 ≤ I.card := by
      have : (2 : ℝ) ≤ (I.card : ℝ) := le_trans hcount (le_of_lt (not_le.1 hcnt))
      exact_mod_cast this
    obtain ⟨i, hi, j, hj, hij⟩ := Finset.one_lt_card.1 (by omega : 1 < I.card)
    have hne : suffixDraw i.val x ≠ suffixDraw j.val x := fun h => hij (hinj h)
    have hpick : ∃ k ∈ I, suffixDraw k.val x ≠ 1 := by
      by_cases h : suffixDraw i.val x = 1
      · exact ⟨j, hj, fun hj1 => hne (by rw [h, hj1])⟩
      · exact ⟨i, hi, h⟩
    obtain ⟨k, hk, hk1⟩ := hpick
    refine hx ⟨suffixDraw k.val x, Finset.mem_erase.2 ⟨hk1, ?_⟩, ?_⟩
    · exact Finset.mem_insert_of_mem (Finset.mem_image.2
        ⟨k.val, Finset.mem_range.2 k.isLt, rfl⟩)
    · intro p _
      have hap := (Finset.mem_filter.1 hk).2 p
      change O.label (p * suffixDraw k.val x) + O.label p
        - 2 * O.label (p * suffixDraw k.val x) * O.label p = 0
      rw [hap]
      rcases O.label_bit p with hl | hl <;> rw [hl] <;> ring
  exact hsub

/-- With findability charged elsewhere, what is left of the clean reference is distinctness
of the pool's draws. -/
theorem measureReal_noCleanRef_off_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (populations : Finset J) (B : State) (pAP t ρsf : ℝ)
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hcount : (2 : ℝ) ≤ (B.nsuff : ℝ) * (pAP - t)) :
    (runMeasure μ D Dsf).real ({x : Run Ω S J |
        ¬ ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
          ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
        \ apShort O B.nsuff pAP t)
      ≤ (B.nsuff : ℝ) ^ 2 * ρsf := by
  classical
  refine le_trans (measureReal_mono ?_ (measure_ne_top _ _))
    (suffix_not_injective_le D Dsf B.nsuff ρsf hρsf hρsf0)
  rintro x ⟨hx, hxA⟩
  rcases noCleanRef_subset O populations B pAP t hcount hx with h | h
  · exact absurd h hxA
  · exact h

theorem measureReal_noCleanRef_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (populations : Finset J) (B : State) (pAP t ρsf : ℝ)
    (hpAP0 : 0 ≤ pAP) (ht : 0 ≤ t)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hcount : (2 : ℝ) ≤ (B.nsuff : ℝ) * (pAP - t)) :
    (runMeasure μ D Dsf).real {x : Run Ω S J |
        ¬ ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
          ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
      ≤ Real.exp (-2 * (B.nsuff : ℝ) * t ^ 2) + (B.nsuff : ℝ) ^ 2 * ρsf := by
  refine le_trans (measureReal_mono (noCleanRef_subset O populations B pAP t hcount)
    (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) ?_
  gcongr
  · exact measureReal_apShort_le D Dsf O B.nsuff pAP t hpAP0 ht hpAPBound
  · exact suffix_not_injective_le D Dsf B.nsuff ρsf hρsf hρsf0

/-- The screen keeps the accept-preserving candidates.  One `screen_pass_rel` per pool
member; the cutoff sits above the clean rate by `γ`. -/
theorem measureReal_screenFail_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (B : State) (hcd : B.cn < B.cd)
    (γ : ℝ) (hγ : 0 ≤ γ) (hscd : 0 < B.scd)
    (hsig : O.η ≤ 1 / 2)
    (hsc : (B.scd : ℝ) * (2 * γ) ≤ (B.sc : ℝ)) :
    (runMeasure μ D Dsf).real (screenFail O populations B)
      ≤ ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2) := by
  classical
  set E : ℝ := ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2) with hEdef
  have hE0 : 0 ≤ E := by positivity
  have hEnn : runMeasure μ D Dsf (screenFail O populations B) ≤ ENNReal.ofReal E := by
    refine runMeasure_slice_le D Dsf _ (measurableSet_screenFail O populations B) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp] with d hd
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.npref).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.nsuff).image (fun i => d.1.1 i)) with hCd
    have hP : ∀ q ∈ Pd, q ∈ Pre := by
      intro q hq
      obtain ⟨j', hj', hq'⟩ := Finset.mem_biUnion.1 hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hq'
      exact hd j' hj' i
    have hPle : Pd.card ≤ populations.card * B.npref := by
      rw [hPd]
      refine le_trans Finset.card_biUnion_le ?_
      calc ∑ j ∈ populations, ((Finset.range B.npref).image (fun i => d.1.2 j i)).card
          ≤ ∑ _j ∈ populations, B.npref :=
            Finset.sum_le_sum (fun j _ => le_trans Finset.card_image_le (by simp))
        _ = populations.card * B.npref := by rw [Finset.sum_const, smul_eq_mul]
    by_cases hm : B.npref ≤ Pd.card
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ screenFail O populations B}
          ⊆ ⋃ v ∈ Cd.filter (fun v => v ≠ 1 ∧ ∀ p, O.label (p * v) = O.label p),
              {ω : Ω | ¬ (B.scd * screenCount O.mq Pd v ω ≤ B.scd * screenBase O.mq Pd Cd ω + B.sc * Pd.card)} := by
        rintro ω ⟨-, hbad⟩
        simp only [not_forall] at hbad
        obtain ⟨v, hv, hv1, hap, hfail⟩ := hbad
        exact Set.mem_biUnion (Finset.mem_filter.2 ⟨hv, hv1, hap⟩) hfail
      refine le_trans (measure_mono hsec) (le_trans (measure_biUnion_finset_le _ _) ?_)
      have hper : ∀ v ∈ Cd.filter (fun v => v ≠ 1 ∧ ∀ p, O.label (p * v) = O.label p),
          μ {ω : Ω | ¬ (B.scd * screenCount O.mq Pd v ω ≤ B.scd * screenBase O.mq Pd Cd ω + B.sc * Pd.card)}
            ≤ ENNReal.ofReal (((B.nsuff : ℝ) + 2) * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2)) := by
        intro v hv
        obtain ⟨hvC, hv1, hap⟩ := Finset.mem_filter.1 hv
        have hclean : ∀ p ∈ Pd, O.flip v p = 0 := by
          intro p _
          show O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p = 0
          rw [hap p]
          rcases O.label_bit p with hl | hl <;> rw [hl] <;> ring
        have htail := screen_pass_rel (cands := Cd) hflat O hcd hP v hv1 hvC γ B.sc B.scd hγ
          hsig hscd hclean hsc
        have hcardR : (B.npref : ℝ) ≤ (Pd.card : ℝ) := by exact_mod_cast hm
        have hrangeC : (Finset.range B.nsuff).card ≤ B.nsuff := by simp
        have hCard : Cd.card ≤ B.nsuff + 1 :=
          le_trans (Finset.card_insert_le _ _)
            (by simpa using le_trans Finset.card_image_le hrangeC)
        have hCdc : (Cd.card : ℝ) ≤ (B.nsuff : ℝ) + 1 := by exact_mod_cast hCard
        rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
        refine ENNReal.ofReal_le_ofReal (le_trans htail ?_)
        have hmono : Real.exp (-2 * (Pd.card : ℝ) * γ ^ 2)
            ≤ Real.exp (-2 * (B.npref : ℝ) * γ ^ 2) := by
          refine Real.exp_le_exp.2 ?_
          nlinarith [sq_nonneg γ]
        have hexp0 : (0 : ℝ) ≤ Real.exp (-2 * (Pd.card : ℝ) * γ ^ 2) := Real.exp_nonneg _
        refine mul_le_mul ?_ hmono hexp0 (by positivity)
        linarith [hCdc]
      refine le_trans (Finset.sum_le_sum hper) ?_
      rw [Finset.sum_const, nsmul_eq_mul, hEdef]
      rw [← ENNReal.ofReal_natCast (Cd.filter _).card, ← ENNReal.ofReal_mul (by positivity)]
      refine ENNReal.ofReal_le_ofReal ?_
      have hrangeC : (Finset.range B.nsuff).card ≤ B.nsuff := by simp
      have hCard : Cd.card ≤ B.nsuff + 1 :=
        le_trans (Finset.card_insert_le _ _)
          (by simpa using le_trans Finset.card_image_le hrangeC)
      have hCardR : ((Cd.filter (fun v => v ≠ 1 ∧ ∀ p, O.label (p * v) = O.label p)).card : ℝ)
          ≤ (B.nsuff : ℝ) + 1 := by
        have := le_trans (Finset.card_filter_le
          Cd (fun v => v ≠ 1 ∧ ∀ p, O.label (p * v) = O.label p)) hCard
        exact_mod_cast this
      have hn0 : (0 : ℝ) ≤ (B.nsuff : ℝ) := Nat.cast_nonneg _
      have hex : (0 : ℝ) ≤ Real.exp (-2 * (B.npref : ℝ) * γ ^ 2) := Real.exp_nonneg _
      have hstep := mul_le_mul_of_nonneg_right hCardR
        (by positivity : (0:ℝ) ≤ ((B.nsuff : ℝ) + 2) * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2))
      nlinarith [hstep, hn0, hex]
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ screenFail O populations B}
          = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
        rintro ⟨h1, -⟩
        exact hm h1
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (screenFail O populations B)).toReal
      ≤ (ENNReal.ofReal E).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = E := ENNReal.toReal_ofReal hE0

open scoped Classical in
/-- The screened pool is big enough to cluster.  Findability puts `k` accept-preserving
suffixes in the pool, the screen keeps them, and distinct draws keep them distinct — so the
clustering has `k` candidates to choose from and does not stall. -/
theorem measureReal_smallScreen_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (B : State) (hcd : B.cn < B.cd)
    (j₀ : J) (hj₀ : j₀ ∈ populations)
    (γ pAP t ρsf ρ : ℝ) (hγ : 0 ≤ γ) (hpAP0 : 0 ≤ pAP) (ht : 0 ≤ t)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hscd : 0 < B.scd) (hsig : O.η ≤ 1 / 2)
    (hsc : (B.scd : ℝ) * (2 * γ) ≤ (B.sc : ℝ))
    (hcount : (B.k : ℝ) ≤ (B.nsuff : ℝ) * (pAP - t))
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hρj : collisionMass (D j₀) ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runMeasure μ D Dsf).real
        {x : Run Ω S J | ¬ (B.k ≤ (screenedAt O.mq populations B x).card)}
      ≤ (B.nsuff : ℝ) ^ 2 * ρsf + (Real.exp (-2 * (B.nsuff : ℝ) * t ^ 2)
        + ((B.npref : ℝ) ^ 2 * ρ
          + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2))) := by
  classical
  set E1 : Set (Run Ω S J) :=
    {x | ¬ Function.Injective (fun i : Fin B.nsuff => suffixDraw i.val x)} with hE1
  set E2 : Set (Run Ω S J) := {x | (((Finset.univ : Finset (Fin B.nsuff)).filter
    (fun i => ∀ p, O.label (p * suffixDraw i.val x) = O.label p)).card : ℝ)
      ≤ (B.nsuff : ℝ) * (pAP - t)} with hE2
  set E3 : Set (Run Ω S J) :=
    {x | ¬ Function.Injective (fun i : Fin B.npref => prefixDraw j₀ i.val x)} with hE3
  set E4 : Set (Run Ω S J) := screenFail O populations B with hE4
  have hsub : {x : Run Ω S J | ¬ (B.k ≤ (screenedAt O.mq populations B x).card)}
      ⊆ E1 ∪ (E2 ∪ (E3 ∪ E4)) := by
    intro x hx
    by_contra hcon
    simp only [Set.mem_union, not_or] at hcon
    obtain ⟨h1, h2, h3, h4⟩ := hcon
    have hinj : Function.Injective (fun i : Fin B.nsuff => suffixDraw i.val x) := by
      by_contra h; exact h1 h
    have hapcount : (B.nsuff : ℝ) * (pAP - t)
        < (((Finset.univ : Finset (Fin B.nsuff)).filter
          (fun i => ∀ p, O.label (p * suffixDraw i.val x) = O.label p)).card : ℝ) := not_le.1 h2
    have hmP : B.npref ≤ (prefixesAt populations B.npref x).card := by
      have hinjP : Function.Injective (fun i : Fin B.npref => prefixDraw j₀ i.val x) := by
        by_contra h; exact h3 h
      have hinjOn : Set.InjOn (fun i => prefixDraw j₀ i x) ↑(Finset.range B.npref) := by
        intro a ha b hb hab
        have := hinjP (show (fun i : Fin B.npref => prefixDraw j₀ i.val x)
            ⟨a, Finset.mem_range.1 (by simpa using ha)⟩
          = (fun i : Fin B.npref => prefixDraw j₀ i.val x)
            ⟨b, Finset.mem_range.1 (by simpa using hb)⟩ from hab)
        simpa using congrArg Fin.val this
      have hcardOf : (prefixesOf j₀ B.npref x).card = B.npref := by
        unfold prefixesOf
        rw [Finset.card_image_of_injOn hinjOn, Finset.card_range]
      calc B.npref = (prefixesOf j₀ B.npref x).card := hcardOf.symm
        _ ≤ (prefixesAt populations B.npref x).card :=
            Finset.card_le_card (fun q hq => Finset.mem_biUnion.2 ⟨j₀, hj₀, hq⟩)
    have hscreen : ∀ v ∈ poolAt B.nsuff x, v ≠ 1 → (∀ p, O.label (p * v) = O.label p) →
        B.scd * screenCount O.mq (prefixesAt populations B.npref x) v (oracleNoise x)
          ≤ B.scd * screenBase O.mq (prefixesAt populations B.npref x)
              (poolAt B.nsuff x) (oracleNoise x)
            + B.sc * (prefixesAt populations B.npref x).card := by
      by_contra h
      exact h4 ⟨hmP, h⟩
    -- the accept-preserving draws, as distinct strings, all survive the screen
    set I : Finset (Fin B.nsuff) := (Finset.univ : Finset (Fin B.nsuff)).filter
      (fun i => ∀ p, O.label (p * suffixDraw i.val x) = O.label p) with hI
    have hImg : I.image (fun i => suffixDraw i.val x) ⊆ screenedAt O.mq populations B x := by
      intro v hv
      obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hv
      have hpool : suffixDraw i.val x ∈ poolAt B.nsuff x :=
        Finset.mem_insert_of_mem (Finset.mem_image.2 ⟨i.val,
          Finset.mem_range.2 i.isLt, rfl⟩)
      refine Finset.mem_filter.2 ⟨hpool, ?_⟩
      by_cases hv1 : suffixDraw i.val x = 1
      · rw [hv1, screenCount_one]
        exact Nat.zero_le _
      · exact hscreen _ hpool hv1 (Finset.mem_filter.1 hi).2
    have hcardI : I.card = (I.image (fun i => suffixDraw i.val x)).card :=
      (Finset.card_image_of_injective I hinj).symm
    have hk : (B.k : ℝ) ≤ ((I.image (fun i => suffixDraw i.val x)).card : ℝ) := by
      rw [← hcardI]
      exact le_trans hcount (le_of_lt hapcount)
    have : B.k ≤ (screenedAt O.mq populations B x).card := by
      have hcast : ((I.image (fun i => suffixDraw i.val x)).card : ℝ)
          ≤ ((screenedAt O.mq populations B x).card : ℝ) := by
        exact_mod_cast Finset.card_le_card hImg
      have : (B.k : ℝ) ≤ ((screenedAt O.mq populations B x).card : ℝ) := le_trans hk hcast
      exact_mod_cast this
    exact hx this
  calc (runMeasure μ D Dsf).real {x : Run Ω S J | ¬ (B.k ≤ (screenedAt O.mq populations B x).card)}
      ≤ (runMeasure μ D Dsf).real (E1 ∪ (E2 ∪ (E3 ∪ E4))) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ (runMeasure μ D Dsf).real E1 + ((runMeasure μ D Dsf).real E2
        + ((runMeasure μ D Dsf).real E3 + (runMeasure μ D Dsf).real E4)) := by
        have h34 := measureReal_union_le (μ := runMeasure μ D Dsf) E3 E4
        have h234 := measureReal_union_le (μ := runMeasure μ D Dsf) E2 (E3 ∪ E4)
        have hall := measureReal_union_le (μ := runMeasure μ D Dsf) E1 (E2 ∪ (E3 ∪ E4))
        linarith
    _ ≤ (B.nsuff : ℝ) ^ 2 * ρsf + (Real.exp (-2 * (B.nsuff : ℝ) * t ^ 2)
        + ((B.npref : ℝ) ^ 2 * ρ
          + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2))) := by
        gcongr
        · exact suffix_not_injective_le D Dsf B.nsuff ρsf hρsf hρsf0
        · exact measureReal_apShort_le D Dsf O B.nsuff pAP t hpAP0 ht hpAPBound
        · exact prefix_not_injective_le D Dsf j₀ B.npref ρ hρj hρ0
        · exact measureReal_screenFail_le hflat O populations D Dsf hsupp B hcd γ hγ hscd hsig
            hsc

open scoped Classical in
/-- Every member the clustering keeps is clean.  Three things have to go right: the
population's draws distinct, the screen holding, and the drawn prefixes not understating a
candidate's flip mass.  The Lloyd ranking does not appear — the family is a subset of what
the screen left, so it inherits the bound. -/
theorem measureReal_dirtyMember_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (j : J) (hj : j ∈ populations)
    (B : State) (hcd : B.cn < B.cd) (hsig : O.η ≤ 1 / 2) (hmpos : 0 < B.npref)
    (Δ γ g ρ : ℝ) (hΔ : 0 < Δ) (hγ : 0 ≤ γ) (hg : 0 ≤ g) (hρ0 : 0 ≤ ρ)
    (hρD : collisionMass (D j) ≤ ρ)
    (hscd : 0 < B.scd)
    (hsc : (B.sc : ℝ) ≤ (B.scd : ℝ) * (Δ * (1 - 2 * O.η) ^ 2 - 2 * γ)) :
    (runMeasure μ D Dsf).real ({x : Run Ω S J | ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
          ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
        ∩ {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O.mq populations x B,
          flipMass O (D j) v ≤ (populations.card : ℝ) * Δ + g})
      ≤ (B.npref : ℝ) ^ 2 * ρ + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2)
        + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * g ^ 2) := by
  classical
  set Δp : ℝ := (populations.card : ℝ) * Δ + g with hΔp
  set E1 : Set (Run Ω S J) :=
    {x | ¬ Function.Injective (fun i : Fin B.npref => prefixDraw j i.val x)} with hE1
  set E2 : Set (Run Ω S J) := screenBad O populations B Δ with hE2
  set E3 : Set (Run Ω S J) := {x | ¬ ∀ v ∈ poolAt B.nsuff x,
    (∑ i : Fin B.npref, O.flip v (prefixDraw j i.val x) ≤ (B.npref : ℝ) * (Δp - g)) →
      flipMass O (D j) v ≤ Δp} with hE3
  have hsub : ({x : Run Ω S J | ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
        ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
      ∩ {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O.mq populations x B,
        flipMass O (D j) v ≤ Δp}) ⊆ (E1 ∪ E2) ∪ E3 := by
    rintro x ⟨hclean, hx⟩
    by_contra hnot
    simp only [Set.mem_union, not_or] at hnot
    obtain ⟨⟨h1, h2⟩, h3⟩ := hnot
    have hinjP : Function.Injective (fun i : Fin B.npref => prefixDraw j i.val x) := by
      by_contra h; exact h1 h
    have hcardPre : B.npref ≤ (prefixesAt populations B.npref x).card := by
      have hinjOn : Set.InjOn (fun i => prefixDraw j i x) ↑(Finset.range B.npref) := by
        intro a ha b hb hab
        have := hinjP (show (fun i : Fin B.npref => prefixDraw j i.val x)
            ⟨a, Finset.mem_range.1 (by simpa using ha)⟩
          = (fun i : Fin B.npref => prefixDraw j i.val x)
            ⟨b, Finset.mem_range.1 (by simpa using hb)⟩ from hab)
        simpa using congrArg Fin.val this
      have hcardOf : (prefixesOf j B.npref x).card = B.npref := by
        unfold prefixesOf
        rw [Finset.card_image_of_injOn hinjOn, Finset.card_range]
      calc B.npref = (prefixesOf j B.npref x).card := hcardOf.symm
        _ ≤ (prefixesAt populations B.npref x).card :=
            Finset.card_le_card (fun q hq => Finset.mem_biUnion.2 ⟨j, hj, hq⟩)
    -- the screen held, so nothing it left flips much of the table
    have hscreen : ∀ v ∈ screenedAt O.mq populations B x,
        ¬ (Δ * ((prefixesAt populations B.npref x).card : ℝ)
          ≤ ∑ p ∈ prefixesAt populations B.npref x, O.flip v p) := by
      by_contra h
      exact h2 ⟨hcardPre, hclean, h⟩
    have hrank := clusterAt_flip_bound O populations B x Δ hscreen
    obtain ⟨v, hv, hbad⟩ : ∃ v ∈ clusterAt O.mq populations x B, ¬ (flipMass O (D j) v ≤ Δp) := by
      by_contra h
      exact hx (fun v hv => by
        by_contra hc
        exact h ⟨v, hv, hc⟩)
    have hvpool : v ∈ poolAt B.nsuff x := clusterAt_subset O populations B x hv
    have hcount : ∑ i : Fin B.npref, O.flip v (prefixDraw j i.val x) ≤ (B.npref : ℝ) * (Δp - g) := by
      rw [sum_eq_sum_prefixesOf O j B.npref x v hinjP]
      have hle1 := sum_prefixesOf_le_prefixesAt O populations j hj B.npref x v
      have hle2 := not_le.1 (hrank v hv)
      have hle3 : ((prefixesAt populations B.npref x).card : ℝ)
          ≤ (populations.card : ℝ) * (B.npref : ℝ) := by
        exact_mod_cast card_prefixesAt_le populations B.npref x
      have hΔ0 : (0 : ℝ) ≤ Δ := le_of_lt hΔ
      have hrw : (B.npref : ℝ) * (Δp - g) = (populations.card : ℝ) * Δ * (B.npref : ℝ) := by
        rw [hΔp]; ring
      rw [hrw]
      nlinarith
    exact hbad (by
      by_contra h
      exact h3 (by
        simp only [hE3, Set.mem_setOf_eq, not_forall]
        exact ⟨v, hvpool, hcount, h⟩))
  calc (runMeasure μ D Dsf).real ({x : Run Ω S J | ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
          ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
        ∩ {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O.mq populations x B,
          flipMass O (D j) v ≤ Δp})
      ≤ (runMeasure μ D Dsf).real ((E1 ∪ E2) ∪ E3) := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ((runMeasure μ D Dsf).real E1 + (runMeasure μ D Dsf).real E2) + (runMeasure μ D Dsf).real E3 := by
        have h12 := measureReal_union_le (μ := runMeasure μ D Dsf) E1 E2
        have h123 := measureReal_union_le (μ := runMeasure μ D Dsf) (E1 ∪ E2) E3
        linarith
    _ ≤ _ := by
        gcongr
        · exact prefix_not_injective_le D Dsf j B.npref ρ hρD hρ0
        · exact measureReal_screenBad_le hflat O populations D Dsf hsupp B hcd hsig hmpos Δ γ
            hΔ hγ hscd hsc
        · exact pool_flipMass_le D Dsf O j B.npref B.nsuff Δp g hg (by positivity)

lemma measurableSet_cutCorrect' (O : Oracle μ S) (lo hi : ℕ) (A₀ : Finset S) (p : S) :
    MeasurableSet[noiseAlg O Set.univ] {ω | ¬ cutCorrect O lo hi A₀ p ω} :=
  measurableSet_filter_pred_map O (T := Set.univ) (fun v => p * v)
    (by simp) (fun U => ¬ ((hi < Finset.card U → O.label p = 1)
      ∧ (Finset.card U ≤ lo → O.label p = 0)))

open scoped Classical in
/-- Events about the table, the pool and the certification draws are measurable: all three
take countably many values. -/
lemma measurableSet_of_run_data_cert (populations : Finset J) (j : J) (B : State)
    (R : Finset S → Finset S → (Fin B.npref → S) → Set (Run Ω S J))
    (hR : ∀ P C t, MeasurableSet (R P C t)) :
    MeasurableSet {x : Run Ω S J | x ∈ R (prefixesAt populations B.npref x) (poolAt B.nsuff x)
      (fun i : Fin B.npref => certPrefix j i.val x)} := by
  classical
  have hcov : {x : Run Ω S J | x ∈ R (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (fun i : Fin B.npref => certPrefix j i.val x)}
      = ⋃ z : Finset S × Finset S × (Fin B.npref → S),
          ((({x : Run Ω S J | prefixesAt populations B.npref x = z.1}
            ∩ {x : Run Ω S J | poolAt B.nsuff x = z.2.1})
            ∩ ⋂ i : Fin B.npref, {x : Run Ω S J | certPrefix j i.val x = z.2.2 i})
          ∩ R z.1 z.2.1 z.2.2) := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_iInter]
    refine ⟨fun h => ⟨(prefixesAt populations B.npref x, poolAt B.nsuff x,
      fun i : Fin B.npref => certPrefix j i.val x), ⟨⟨rfl, rfl⟩, fun _ => rfl⟩, h⟩, ?_⟩
    rintro ⟨⟨P, C, t⟩, ⟨⟨hP, hC⟩, ht⟩, hx⟩
    simp only at hP hC ht
    rw [hP, hC, show (fun i : Fin B.npref => certPrefix j i.val x) = t from funext ht]
    exact hx
  rw [hcov]
  exact MeasurableSet.iUnion (fun z =>
    (((measurableSet_prefixesAt populations B.npref z.1).inter
      (measurableSet_poolAt B.nsuff z.2.1)).inter
        (MeasurableSet.iInter (fun i : Fin B.npref =>
          measurableSet_eq_fun (measurable_cert j i.val) measurable_const))).inter
      (hR z.1 z.2.1 z.2.2))

lemma certOf_eq_image (j : J) (m : ℕ) (x : Run Ω S J) :
    certOf j m x = (Finset.univ : Finset (Fin m)).image (fun i => certPrefix j i.val x) :=
  image_range_eq_image_univ m (fun i => certPrefix j i x)

open scoped Classical in
/-- The certification sample misses a wrong cut.  The family is fixed before the sample
is drawn, so a cut wrong on `εcov` of the population is hit `εcov·m` times up to the
Hoeffding slack `t`. -/
noncomputable def hitShort (O : Oracle μ S) (populations : Finset J) (Dj : Measure S) (j : J)
    (B : State) (εcov t : ℝ) : Set (Run Ω S J) :=
  {x | Function.Injective (fun i : Fin B.npref => certPrefix j i.val x)
    ∧ εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi (clusterAt O.mq populations x B) p (oracleNoise x)}
    ∧ (((certOf j B.npref x).filter (fun p =>
        ¬ cutCorrect O B.lo B.hi (clusterAt O.mq populations x B) p (oracleNoise x))).card : ℝ)
      ≤ (B.npref : ℝ) * (εcov - t)}

open scoped Classical in
lemma measurableSet_hitShort (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (j : J) (B : State) (εcov t : ℝ) :
    MeasurableSet (hitShort O populations Dj j B εcov t) := by
  classical
  have hR : ∀ (P C : Finset S) (tt : Fin B.npref → S), MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | Function.Injective tt
        ∧ εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
            (clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)) p (oracleNoise x)}
        ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
            ¬ cutCorrect O B.lo B.hi (clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)) p (oracleNoise x))).card
              : ℝ)
          ≤ (B.npref : ℝ) * (εcov - t)} else ∅) := by
    intro P C tt
    split_ifs with hone
    · by_cases hinj : Function.Injective tt
      · have hω : MeasurableSet {ω : Ω |
            εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
                (clusterOf O B.cn B.cd B.sc B.scd P C B.k ω) p ω}
            ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                ¬ cutCorrect O B.lo B.hi (clusterOf O B.cn B.cd B.sc B.scd P C B.k ω) p ω)).card : ℝ)
              ≤ (B.npref : ℝ) * (εcov - t)} := by
          refine measurableSet_of_fam (T := C.powerset)
            (fun ω => Finset.mem_powerset.2 (clusterOf_subset O B.cn B.cd B.sc B.scd P C B.k ω hone))
            (fun A₀ => measurableSet_clusterOf O B.cn B.cd B.sc B.scd P C B.k hone A₀)
            (fun A₀ => {ω | εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi A₀ p ω}
              ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                  ¬ cutCorrect O B.lo B.hi A₀ p ω)).card : ℝ) ≤ (B.npref : ℝ) * (εcov - t)})
            (fun A₀ => MeasurableSet.inter
              (measurableSet_le measurable_const (measurable_badMassReal O Dj B.lo B.hi A₀))
              (noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O
                (fun p ω => ¬ cutCorrect O B.lo B.hi A₀ p ω)
                (fun p _ => measurableSet_cutCorrect' O B.lo B.hi A₀ p)
                (fun U => ((U.card : ℝ) ≤ (B.npref : ℝ) * (εcov - t))))))
        have hset : {x : Run Ω S J | Function.Injective tt
            ∧ εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
                (clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)) p (oracleNoise x)}
            ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                ¬ cutCorrect O B.lo B.hi
                  (clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)) p (oracleNoise x))).card : ℝ)
              ≤ (B.npref : ℝ) * (εcov - t)}
            = oracleNoise ⁻¹' {ω : Ω |
              εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
                  (clusterOf O B.cn B.cd B.sc B.scd P C B.k ω) p ω}
              ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                  ¬ cutCorrect O B.lo B.hi (clusterOf O B.cn B.cd B.sc B.scd P C B.k ω) p ω)).card : ℝ)
                ≤ (B.npref : ℝ) * (εcov - t)} := by
          ext x
          simp only [Set.mem_setOf_eq, Set.mem_preimage, hinj, true_and]
        rw [hset]
        exact measurable_nz hω
      · simp [hinj]
    · exact MeasurableSet.empty
  have hrw : hitShort O populations Dj j B εcov t
      = {x : Run Ω S J | x ∈ (fun P C tt => if (1 : S) ∈ C then
          {x : Run Ω S J | Function.Injective tt
            ∧ εcov ≤ Dj.real {p | ¬ cutCorrect O B.lo B.hi
                (clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)) p (oracleNoise x)}
            ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                ¬ cutCorrect O B.lo B.hi
                  (clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)) p (oracleNoise x))).card : ℝ)
              ≤ (B.npref : ℝ) * (εcov - t)} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (fun i : Fin B.npref => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), hitShort,
      ← certOf_eq_image j B.npref x, ← clusterAt_eq_clusterOf O populations B x]
    tauto
  rw [hrw]
  exact measurableSet_of_run_data_cert populations j B _ hR

lemma measurableSet_voteCount_gt (O : Oracle μ S) (F : Finset S) (n : ℕ) (p : S) :
    MeasurableSet[noiseAlg O Set.univ] {ω | n < voteCount O.mq F p ω} :=
  measurableSet_filter_pred_map O (T := Set.univ) (fun v => p * v) (by simp)
    (fun V => n < V.card)

lemma measurableSet_voteCount_le (O : Oracle μ S) (F : Finset S) (n : ℕ) (p : S) :
    MeasurableSet[noiseAlg O Set.univ] {ω | voteCount O.mq F p ω ≤ n} :=
  measurableSet_filter_pred_map O (T := Set.univ) (fun v => p * v) (by simp)
    (fun V => V.card ≤ n)

open scoped Classical in
lemma measurableSet_sideOf_gt (O : Oracle μ S) (A F : Finset S) (n : ℕ) (U : Finset S) :
    MeasurableSet {ω | A.filter (fun p => n < voteCount O.mq F p ω) = U} :=
  noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O
    (fun p ω => n < voteCount O.mq F p ω)
    (fun p _ => measurableSet_voteCount_gt O F n p) (fun V => V = U))

open scoped Classical in
lemma measurableSet_decOf (O : Oracle μ S) (A F : Finset S) (lo hi : ℕ) (U : Finset S) :
    MeasurableSet {ω | A.filter (fun p => hi - 1 < voteCount O.mq F p ω
        ∨ voteCount O.mq F p ω ≤ lo) = U} :=
  noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O
    (fun p ω => hi - 1 < voteCount O.mq F p ω ∨ voteCount O.mq F p ω ≤ lo)
    (fun p _ => (measurableSet_voteCount_gt O F (hi - 1) p).union
      (measurableSet_voteCount_le O F lo p)) (fun V => V = U))

/-- Distinct draws are counted once each, so the sample's hit count is the draw count. -/
lemma card_filter_certOf (j : J) (m : ℕ) (x : Run Ω S J) (Q : S → Prop)
    (instA : DecidablePred Q) (instB : DecidablePred (fun i : ℕ => Q (certPrefix j i x)))
    (hinj : Function.Injective (fun i : Fin m => certPrefix j i.val x)) :
    (@Finset.filter _ (fun i : ℕ => Q (certPrefix j i x)) instB (Finset.range m)).card
      = (@Finset.filter _ Q instA (certOf j m x)).card := by
  refine Finset.card_nbij (fun i => certPrefix j i x) (fun i hi => ?_) (fun a ha b hb hab => ?_)
    (fun p hp => ?_)
  · obtain ⟨hi, hQ⟩ := Finset.mem_filter.1 hi
    exact Finset.mem_filter.2 ⟨Finset.mem_image.2 ⟨i, hi, rfl⟩, hQ⟩
  · simp only [Finset.coe_filter, Set.mem_setOf_eq, Finset.mem_range] at ha hb
    have := hinj (show (fun i : Fin m => certPrefix j i.val x) ⟨a, ha.1⟩
      = (fun i : Fin m => certPrefix j i.val x) ⟨b, hb.1⟩ from hab)
    simpa using congrArg Fin.val this
  · simp only [Finset.coe_filter, Set.mem_setOf_eq] at hp
    obtain ⟨hpC, hQ⟩ := hp
    unfold certOf at hpC
    obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hpC
    exact ⟨i, by simp only [Finset.coe_filter, Set.mem_setOf_eq, Finset.mem_range,
      Finset.mem_range.1 hi, true_and]; exact hQ, rfl⟩

open scoped Classical in
/-- A cut wrong on the population is wrong on the sample.  The family is a function of
the noise and the table, so the certification draws are independent of it and plain
Hoeffding applies. -/
theorem measureReal_hitShort_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (populations : Finset J) (j : J) (B : State) (εcov t : ℝ) (hε : 0 ≤ εcov) (ht : 0 ≤ t) :
    (runMeasure μ D Dsf).real (hitShort O populations (D j) j B εcov t)
      ≤ Real.exp (-2 * (B.npref : ℝ) * t ^ 2) := by
  classical
  set E : ℝ := Real.exp (-2 * (B.npref : ℝ) * t ^ 2) with hEdef
  have hEnn : runMeasure μ D Dsf (hitShort O populations (D j) j B εcov t) ≤ ENNReal.ofReal E := by
    refine runMeasure_slice_cert_le D Dsf _
      (measurableSet_hitShort O populations (D j) j B εcov t) _ ?_
    intro y
    set F : Finset S := clusterAt O.mq populations ((y.1, (y.2, fun _ _ => (1 : S))) : Run Ω S J) B
      with hF
    set W : Set S := {p | ¬ cutCorrect O B.lo B.hi F p y.1} with hW
    have hFeq : ∀ c : J → ℕ → S,
        clusterAt O.mq populations ((y.1, (y.2, c)) : Run Ω S J) B = F := fun c => rfl
    by_cases hmass : εcov ≤ (D j).real W
    · have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
            ∈ hitShort O populations (D j) j B εcov t}
          ⊆ {c | (((Finset.range B.npref).filter (fun i => c j i ∈ W)).card : ℝ)
            ≤ (B.npref : ℝ) * (εcov - t)} := by
        rintro c ⟨hinj, -, hcount⟩
        show (((Finset.range B.npref).filter (fun i => c j i ∈ W)).card : ℝ)
          ≤ (B.npref : ℝ) * (εcov - t)
        calc (((Finset.range B.npref).filter (fun i => c j i ∈ W)).card : ℝ)
            = (((certOf j B.npref ((y.1, (y.2, c)) : Run Ω S J)).filter
                (fun p => ¬ cutCorrect O B.lo B.hi F p y.1)).card : ℝ) :=
              congrArg (fun n : ℕ => (n : ℝ)) (card_filter_certOf j B.npref
                ((y.1, (y.2, c)) : Run Ω S J)
                (fun p => ¬ cutCorrect O B.lo B.hi F p y.1) _ _ hinj)
          _ ≤ (B.npref : ℝ) * (εcov - t) := hcount
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) _),
        ← measureReal_def]
      refine ENNReal.ofReal_le_ofReal ?_
      rw [hEdef]
      have hcert := cert_hits_wrongSet D j B.npref W εcov t hε ht hmass
      convert hcert using 3
      funext c
      congr!
    · have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
          ∈ hitShort O populations (D j) j B εcov t} = (∅ : Set (J → ℕ → S)) := by
        ext c
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨-, hm, -⟩
        rw [hFeq c] at hm
        exact hmass hm
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (hitShort O populations (D j) j B εcov t)).toReal
      ≤ (ENNReal.ofReal E).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = E := ENNReal.toReal_ofReal (Real.exp_nonneg _)

open scoped Classical in
/-- A class thin on the certification sample is thin on the population: the class is fixed
before the sample is drawn, so this is `cert_hits_wrongSet` at the class's own set. -/
theorem measureReal_thinClass_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (j : J) (B : State) (b εcov t : ℝ) (hε : 0 ≤ εcov) (ht : 0 ≤ t) :
    (runMeasure μ D Dsf).real
        {x : Run Ω S J | Function.Injective (fun i : Fin B.npref => certPrefix j i.val x)
          ∧ εcov ≤ (D j).real {p | O.label p = b}
          ∧ (((certOf j B.npref x).filter (fun p => O.label p = b)).card : ℝ)
            ≤ (B.npref : ℝ) * (εcov - t)}
      ≤ Real.exp (-2 * (B.npref : ℝ) * t ^ 2) := by
  classical
  have hmeas : MeasurableSet {x : Run Ω S J |
      Function.Injective (fun i : Fin B.npref => certPrefix j i.val x)
        ∧ εcov ≤ (D j).real {p | O.label p = b}
        ∧ (((certOf j B.npref x).filter (fun p => O.label p = b)).card : ℝ)
          ≤ (B.npref : ℝ) * (εcov - t)} := by
    have hR : ∀ (P C : Finset S) (tt : Fin B.npref → S), MeasurableSet
        {_x : Run Ω S J | Function.Injective tt ∧ εcov ≤ (D j).real {p | O.label p = b}
          ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter
              (fun p => O.label p = b)).card : ℝ) ≤ (B.npref : ℝ) * (εcov - t)} := by
      intro P C tt
      by_cases h : Function.Injective tt ∧ εcov ≤ (D j).real {p | O.label p = b}
          ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter
              (fun p => O.label p = b)).card : ℝ) ≤ (B.npref : ℝ) * (εcov - t)
      · simp [h]
      · simp [h]
    have hrw : {x : Run Ω S J |
        Function.Injective (fun i : Fin B.npref => certPrefix j i.val x)
          ∧ εcov ≤ (D j).real {p | O.label p = b}
          ∧ (((certOf j B.npref x).filter (fun p => O.label p = b)).card : ℝ)
            ≤ (B.npref : ℝ) * (εcov - t)}
        = {x : Run Ω S J | x ∈ (fun (_P _C : Finset S) (tt : Fin B.npref → S) =>
            {_x : Run Ω S J | Function.Injective tt ∧ εcov ≤ (D j).real {p | O.label p = b}
              ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter
                  (fun p => O.label p = b)).card : ℝ) ≤ (B.npref : ℝ) * (εcov - t)})
          (prefixesAt (∅ : Finset J) B.npref x) (poolAt B.nsuff x)
          (fun i : Fin B.npref => certPrefix j i.val x)} := by
      ext x
      simp only [Set.mem_setOf_eq, ← certOf_eq_image j B.npref x]
    rw [hrw]
    exact measurableSet_of_run_data_cert (∅ : Finset J) j B _ hR
  have hEnn : runMeasure μ D Dsf {x : Run Ω S J |
      Function.Injective (fun i : Fin B.npref => certPrefix j i.val x)
        ∧ εcov ≤ (D j).real {p | O.label p = b}
        ∧ (((certOf j B.npref x).filter (fun p => O.label p = b)).card : ℝ)
          ≤ (B.npref : ℝ) * (εcov - t)}
      ≤ ENNReal.ofReal (Real.exp (-2 * (B.npref : ℝ) * t ^ 2)) := by
    refine runMeasure_slice_cert_le D Dsf _ hmeas _ ?_
    intro y
    by_cases hmass : εcov ≤ (D j).real {p | O.label p = b}
    · have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J) ∈ {x : Run Ω S J |
            Function.Injective (fun i : Fin B.npref => certPrefix j i.val x)
              ∧ εcov ≤ (D j).real {p | O.label p = b}
              ∧ (((certOf j B.npref x).filter (fun p => O.label p = b)).card : ℝ)
                ≤ (B.npref : ℝ) * (εcov - t)}}
          ⊆ {c | (((Finset.range B.npref).filter
              (fun i => c j i ∈ {p | O.label p = b})).card : ℝ)
            ≤ (B.npref : ℝ) * (εcov - t)} := by
        rintro c ⟨hinj, -, hcount⟩
        change (((Finset.range B.npref).filter
          (fun i => c j i ∈ {p | O.label p = b})).card : ℝ) ≤ (B.npref : ℝ) * (εcov - t)
        calc (((Finset.range B.npref).filter
              (fun i => c j i ∈ {p | O.label p = b})).card : ℝ)
            = (((certOf j B.npref ((y.1, (y.2, c)) : Run Ω S J)).filter
                (fun p => O.label p = b)).card : ℝ) :=
              congrArg (fun n : ℕ => (n : ℝ)) (card_filter_certOf j B.npref
                ((y.1, (y.2, c)) : Run Ω S J) (fun p => O.label p = b) _ _ hinj)
          _ ≤ (B.npref : ℝ) * (εcov - t) := hcount
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top
          (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) _), ← measureReal_def]
      refine ENNReal.ofReal_le_ofReal ?_
      have hcert := cert_hits_wrongSet D j B.npref {p | O.label p = b} εcov t hε ht hmass
      convert hcert using 3
    · refine le_trans (measure_mono (show _ ⊆ (∅ : Set (J → ℕ → S)) from ?_)) (by simp)
      rintro c ⟨-, hm, -⟩
      exact absurd hm hmass
  rw [measureReal_def]
  calc (runMeasure μ D Dsf {x : Run Ω S J |
        Function.Injective (fun i : Fin B.npref => certPrefix j i.val x)
          ∧ εcov ≤ (D j).real {p | O.label p = b}
          ∧ (((certOf j B.npref x).filter (fun p => O.label p = b)).card : ℝ)
            ≤ (B.npref : ℝ) * (εcov - t)}).toReal
      ≤ (ENNReal.ofReal (Real.exp (-2 * (B.npref : ℝ) * t ^ 2))).toReal :=
        ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = Real.exp (-2 * (B.npref : ℝ) * t ^ 2) := ENNReal.toReal_ofReal (Real.exp_nonneg _)

/-- The sample counts draws, the table counts strings: a repeated draw is one string. -/
lemma card_filter_certOf_le (j : J) (m : ℕ) (x : Run Ω S J) (Q : S → Prop)
    (instA : DecidablePred Q) (instB : DecidablePred (fun i : ℕ => Q (certPrefix j i x))) :
    (@Finset.filter _ Q instA (certOf j m x)).card
      ≤ (@Finset.filter _ (fun i : ℕ => Q (certPrefix j i x)) instB (Finset.range m)).card := by
  refine Finset.card_le_card_of_surjOn (fun i => certPrefix j i x) (fun p hp => ?_)
  simp only [Finset.coe_filter, Set.mem_setOf_eq] at hp
  obtain ⟨hpC, hQ⟩ := hp
  unfold certOf at hpC
  obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hpC
  exact ⟨i, by simp only [Finset.coe_filter, Set.mem_setOf_eq, Finset.mem_range,
    Finset.mem_range.1 hi, true_and]; exact hQ, rfl⟩

open scoped Classical in
/-- The sample is not mostly prefixes the family flips.  The heavy set is chosen by the
noise and the table; the certification draws are neither, so its hit count is the upper
Hoeffding tail at its mass. -/
noncomputable def heavyHits (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (j : J) (B : State) (f q t : ℝ) : Set (Run Ω S J) :=
  {x | Dj.real {p | ¬ (flipCount O ((clusterAt O.mq populations x B).erase 1) p
        ≤ (((clusterAt O.mq populations x B).erase 1).card : ℝ) * f)} ≤ q
    ∧ (B.npref : ℝ) * (q + t)
        ≤ (((certOf j B.npref x).filter (fun p =>
          ¬ (flipCount O ((clusterAt O.mq populations x B).erase 1) p
            ≤ (((clusterAt O.mq populations x B).erase 1).card : ℝ) * f))).card : ℝ)}

open scoped Classical in
lemma measurableSet_heavyHits (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (j : J) (B : State) (f q t : ℝ) :
    MeasurableSet (heavyHits O populations Dj j B f q t) := by
  classical
  have hR : ∀ (P C : Finset S) (tt : Fin B.npref → S), MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | Dj.real {p | ¬ (flipCount O
            ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p
          ≤ (((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1).card : ℝ) * f)} ≤ q
        ∧ (B.npref : ℝ) * (q + t)
            ≤ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
              ¬ (flipCount O ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p
                ≤ (((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1).card : ℝ) * f))).card
                  : ℝ)} else ∅) := by
    intro P C tt
    split_ifs with hone
    · refine measurable_nz (measurableSet_of_fam (T := C.powerset)
        (fun ω => Finset.mem_powerset.2 (clusterOf_subset O B.cn B.cd B.sc B.scd P C B.k ω hone))
        (fun A₀ => measurableSet_clusterOf O B.cn B.cd B.sc B.scd P C B.k hone A₀)
        (fun A₀ => {_ω : Ω |
          Dj.real {p | ¬ (flipCount O (A₀.erase 1) p ≤ ((A₀.erase 1).card : ℝ) * f)} ≤ q
          ∧ (B.npref : ℝ) * (q + t)
              ≤ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                ¬ (flipCount O (A₀.erase 1) p ≤ ((A₀.erase 1).card : ℝ) * f))).card : ℝ)})
        (fun A₀ => ?_))
      by_cases hcond : Dj.real {p | ¬ (flipCount O (A₀.erase 1) p
            ≤ ((A₀.erase 1).card : ℝ) * f)} ≤ q
          ∧ (B.npref : ℝ) * (q + t)
            ≤ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
              ¬ (flipCount O (A₀.erase 1) p ≤ ((A₀.erase 1).card : ℝ) * f))).card : ℝ)
      · simpa [hcond] using MeasurableSet.univ
      · simpa [hcond] using MeasurableSet.empty
    · exact MeasurableSet.empty
  have hrw : heavyHits O populations Dj j B f q t
      = {x : Run Ω S J | x ∈ (fun P C tt => if (1 : S) ∈ C then
          {x : Run Ω S J | Dj.real {p | ¬ (flipCount O
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p
              ≤ (((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1).card : ℝ) * f)} ≤ q
            ∧ (B.npref : ℝ) * (q + t)
                ≤ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                  ¬ (flipCount O ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p
                    ≤ (((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1).card : ℝ)
                      * f))).card : ℝ)} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (fun i : Fin B.npref => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), heavyHits,
      ← certOf_eq_image j B.npref x, ← clusterAt_eq_clusterOf O populations B x]
    tauto
  rw [hrw]
  exact measurableSet_of_run_data_cert populations j B _ hR

open scoped Classical in
/-- The flip-heavy prefixes are a `q` fraction of the population, so the sample sees at most
`m(q + t)` of them. -/
theorem measureReal_heavyHits_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (populations : Finset J) (j : J) (B : State) (f q t : ℝ) (ht : 0 ≤ t) :
    (runMeasure μ D Dsf).real (heavyHits O populations (D j) j B f q t)
      ≤ Real.exp (-2 * (B.npref : ℝ) * t ^ 2) := by
  classical
  have hEnn : runMeasure μ D Dsf (heavyHits O populations (D j) j B f q t)
      ≤ ENNReal.ofReal (Real.exp (-2 * (B.npref : ℝ) * t ^ 2)) := by
    refine runMeasure_slice_cert_le D Dsf _
      (measurableSet_heavyHits O populations (D j) j B f q t) _ ?_
    intro y
    set F : Finset S :=
      (clusterAt O.mq populations ((y.1, (y.2, fun _ _ => (1 : S))) : Run Ω S J) B).erase 1 with hF
    set W : Set S := {p | ¬ (flipCount O F p ≤ (F.card : ℝ) * f)} with hW
    have hFeq : ∀ c : J → ℕ → S,
        (clusterAt O.mq populations ((y.1, (y.2, c)) : Run Ω S J) B).erase 1 = F := fun c => rfl
    by_cases hmass : (D j).real W ≤ q
    · have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
            ∈ heavyHits O populations (D j) j B f q t}
          ⊆ {c | (B.npref : ℝ) * (q + t)
            ≤ (((Finset.range B.npref).filter (fun i => c j i ∈ W)).card : ℝ)} := by
        rintro c ⟨-, hcount⟩
        refine le_trans hcount ?_
        exact_mod_cast card_filter_certOf_le j B.npref ((y.1, (y.2, c)) : Run Ω S J)
          (fun p => p ∈ W) _ _
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) _),
        ← measureReal_def]
      refine ENNReal.ofReal_le_ofReal ?_
      have hcert := cert_hits_upper D j B.npref W q t ht hmass
      convert hcert using 3
      funext c
      congr!
    · have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
          ∈ heavyHits O populations (D j) j B f q t} = (∅ : Set (J → ℕ → S)) := by
        ext c
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨hm, -⟩
        exact hmass hm
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (heavyHits O populations (D j) j B f q t)).toReal
      ≤ (ENNReal.ofReal (Real.exp (-2 * (B.npref : ℝ) * t ^ 2))).toReal :=
        ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = Real.exp (-2 * (B.npref : ℝ) * t ^ 2) := ENNReal.toReal_ofReal (Real.exp_nonneg _)

lemma measurableSet_decided' (O : Oracle μ S) (lo ha : ℕ) (A₀ : Finset S) (p : S) :
    MeasurableSet[noiseAlg O Set.univ] {ω | ¬ decided O.mq lo ha A₀ p ω} :=
  measurableSet_filter_pred_map O (T := Set.univ) (fun v => p * v)
    (by simp) (fun U => ¬ (ha < Finset.card U ∨ Finset.card U ≤ lo))

open scoped Classical in
lemma measurableSet_indecisionCount (O : Oracle μ S) (lo ha : ℕ) (A₀ A : Finset S) (r : ℝ) :
    MeasurableSet {ω | (((A.filter (fun p => ¬ decided O.mq lo ha A₀ p ω)).card : ℝ) ≤ r)} :=
  noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O
    (fun p ω => ¬ decided O.mq lo ha A₀ p ω)
    (fun p _ => measurableSet_decided' O lo ha A₀ p) (fun U => ((U.card : ℝ) ≤ r)))

open scoped Classical in
/-- The gate's verdict is measurable: each side is a fibre of the votes, and the count it
scores is a fibre of the bits at the certification prefixes. -/
lemma measurableSet_admittedFixed (O : Oracle μ S) (lo hi n₀ : ℕ) (F A : Finset S)
    (α : ℝ) : MeasurableSet {ω | admitted O.mq lo hi n₀ α F A ω} := by
  classical
  have hcover : {ω | admitted O.mq lo hi n₀ α F A ω}
      = ⋃ t ∈ A.powerset ×ˢ A.powerset,
          ({ω | cutSides O.mq lo hi F A ω = t}
            ∩ {ω | n₀ ≤ t.2.card →
                binomSfGe t.2.card (1 / 2 : ℝ)
                  (agreeOf t.1 t.2 (A.filter (fun p => O.mq p ω = 1))) ≤ α}) := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Finset.mem_coe,
      exists_prop]
    constructor
    · intro h
      refine ⟨cutSides O.mq lo hi F A ω, ?_, rfl, ?_⟩
      · exact Finset.mem_product.2 ⟨Finset.mem_powerset.2 (Finset.filter_subset _ _),
          Finset.mem_powerset.2 (Finset.filter_subset _ _)⟩
      · simpa [admitted, agreeCount] using h
    · rintro ⟨t, -, hEq, hb⟩
      simpa [admitted, agreeCount, hEq] using hb
  rw [hcover]
  refine Finset.measurableSet_biUnion _ (fun t _ => MeasurableSet.inter ?_ ?_)
  · have hpair : {ω | cutSides O.mq lo hi F A ω = t}
        = {ω | A.filter (fun p => hi - 1 < voteCount O.mq F p ω) = t.1}
          ∩ {ω | A.filter (fun p => hi - 1 < voteCount O.mq F p ω
                ∨ voteCount O.mq F p ω ≤ lo) = t.2} := by
      ext ω
      constructor
      · intro h
        exact ⟨congrArg Prod.fst h, congrArg Prod.snd h⟩
      · rintro ⟨h1, h2⟩
        exact Prod.ext h1 h2
    rw [hpair]
    exact (measurableSet_sideOf_gt O A F (hi - 1) t.1).inter
      (measurableSet_decOf O A F lo hi t.2)
  · by_cases h : n₀ ≤ t.2.card
    · have hrw : {ω : Ω | n₀ ≤ t.2.card →
          binomSfGe t.2.card (1 / 2 : ℝ)
            (agreeOf t.1 t.2 (A.filter (fun p => O.mq p ω = 1))) ≤ α}
          = {ω : Ω | binomSfGe t.2.card (1 / 2 : ℝ)
            (agreeOf t.1 t.2 (A.filter (fun p => O.mq p ω = 1))) ≤ α} := by
        ext ω; simp [h]
      rw [hrw]
      exact noiseAlg_le O Set.univ _ (measurableSet_filter_pred O (T := Set.univ) (by simp)
        (fun V => binomSfGe t.2.card (1 / 2 : ℝ) (agreeOf t.1 t.2 V) ≤ α))
    · have hrw : {ω : Ω | n₀ ≤ t.2.card →
          binomSfGe t.2.card (1 / 2 : ℝ)
            (agreeOf t.1 t.2 (A.filter (fun p => O.mq p ω = 1))) ≤ α}
          = (Set.univ : Set Ω) := by
        ext ω; simp [h]
      rw [hrw]
      exact MeasurableSet.univ

/-- The family is usable at `p`: at most an `f` fraction of it flips there, and the family
is neither too small for the thresholds nor larger than the round allows. -/
def famGood (O : Oracle μ S) (f : ℝ) (kmin kmax : ℕ) (F : Finset S) (p : S) : Prop :=
  flipCount O F p ≤ (F.card : ℝ) * f ∧ kmin ≤ F.card ∧ F.card ≤ kmax

open scoped Classical in
/-- The draws were good and the state still did not return.  Everything the state needs
of its draws — the certification prefixes fresh and nonempty, and the family light on all
but an `l` fraction of them — holds, and the round's two tests still fail.
This is the event `ret_at_whp` prices; the rest of the lift charges the draws. -/
noncomputable def retMiss (O : Oracle μ S) (populations : Finset J) (j : J) (B : State)
    (α l lcut f : ℝ) (n₀ kmin kmax : ℕ) : Set (Run Ω S J) :=
  {x | Disjoint (prefixesAt populations B.npref x) (certOf j B.npref x)
    ∧ (certOf j B.npref x).card = B.npref
    ∧ 0 < (certOf j B.npref x).card
    ∧ (((certOf j B.npref x).filter (fun p =>
        ¬ famGood O f kmin kmax ((clusterAt O.mq populations x B).erase 1) p)).card : ℝ)
        ≤ lcut * ((certOf j B.npref x).card : ℝ)
    ∧ ¬ ((((certOf j B.npref x).filter (fun p => ¬ decided O.mq B.lo (B.hi - 1)
            ((clusterAt O.mq populations x B).erase 1) p (oracleNoise x))).card : ℝ)
          ≤ 2 * l * ((certOf j B.npref x).card : ℝ)
        ∧ admitted O.mq B.lo B.hi B.gmin α ((clusterAt O.mq populations x B).erase 1)
            (certOf j B.npref x) (oracleNoise x))}

open scoped Classical in
lemma measurableSet_retMiss (O : Oracle μ S) (populations : Finset J) (j : J) (B : State)
    (α l lcut f : ℝ) (n₀ kmin kmax : ℕ) :
    MeasurableSet (retMiss O populations j B α l lcut f n₀ kmin kmax) := by
  classical
  have hR : ∀ (P C : Finset S) (tt : Fin B.npref → S), MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.npref)).image tt)
        ∧ ((Finset.univ : Finset (Fin B.npref)).image tt).card = B.npref
        ∧ 0 < ((Finset.univ : Finset (Fin B.npref)).image tt).card
        ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
            ¬ famGood O f kmin kmax
              ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p)).card : ℝ)
            ≤ lcut * (((Finset.univ : Finset (Fin B.npref)).image tt).card : ℝ)
        ∧ ¬ (((((Finset.univ : Finset (Fin B.npref)).image tt).filter
                (fun p => ¬ decided O.mq B.lo (B.hi - 1)
                  ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p (oracleNoise x))).card : ℝ)
              ≤ 2 * l * (((Finset.univ : Finset (Fin B.npref)).image tt).card : ℝ)
            ∧ admitted O.mq B.lo B.hi B.gmin α
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1)
                ((Finset.univ : Finset (Fin B.npref)).image tt) (oracleNoise x))} else ∅) := by
    intro P C tt
    split_ifs with hone
    · set A : Finset S := (Finset.univ : Finset (Fin B.npref)).image tt with hA
      by_cases hdraw : Disjoint P A ∧ A.card = B.npref ∧ 0 < A.card
      · have hω : MeasurableSet {ω : Ω |
            ((A.filter (fun p => ¬ famGood O f kmin kmax
              ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1) p)).card : ℝ)
              ≤ lcut * (A.card : ℝ)
            ∧ ¬ (((A.filter (fun p => ¬ decided O.mq B.lo (B.hi - 1)
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1) p ω)).card : ℝ)
                  ≤ 2 * l * (A.card : ℝ)
                ∧ admitted O.mq B.lo B.hi B.gmin α
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1) A ω)} := by
          refine measurableSet_of_fam (T := C.powerset)
            (fun ω => Finset.mem_powerset.2 (clusterOf_subset O B.cn B.cd B.sc B.scd P C B.k ω hone))
            (fun A₀ => measurableSet_clusterOf O B.cn B.cd B.sc B.scd P C B.k hone A₀)
            (fun A₀ => {ω : Ω |
              ((A.filter (fun p => ¬ famGood O f kmin kmax (A₀.erase 1) p)).card : ℝ)
                ≤ lcut * (A.card : ℝ)
              ∧ ¬ (((A.filter (fun p => ¬ decided O.mq B.lo (B.hi - 1) (A₀.erase 1) p ω)).card : ℝ)
                    ≤ 2 * l * (A.card : ℝ)
                  ∧ admitted O.mq B.lo B.hi B.gmin α (A₀.erase 1) A ω)})
            (fun A₀ => ?_)
          by_cases hlight : ((A.filter (fun p =>
              ¬ famGood O f kmin kmax (A₀.erase 1) p)).card : ℝ) ≤ lcut * (A.card : ℝ)
          · have hrw : {ω : Ω |
                ((A.filter (fun p => ¬ famGood O f kmin kmax (A₀.erase 1) p)).card : ℝ)
                  ≤ lcut * (A.card : ℝ)
                ∧ ¬ (((A.filter (fun p =>
                      ¬ decided O.mq B.lo (B.hi - 1) (A₀.erase 1) p ω)).card : ℝ)
                      ≤ 2 * l * (A.card : ℝ)
                    ∧ admitted O.mq B.lo B.hi B.gmin α (A₀.erase 1) A ω)}
                = ({ω : Ω | ((A.filter (fun p =>
                      ¬ decided O.mq B.lo (B.hi - 1) (A₀.erase 1) p ω)).card : ℝ)
                        ≤ 2 * l * (A.card : ℝ)}
                  ∩ {ω : Ω | admitted O.mq B.lo B.hi B.gmin α (A₀.erase 1) A ω})ᶜ := by
              ext ω
              simp only [Set.mem_setOf_eq, Set.mem_compl_iff, Set.mem_inter_iff, hlight, true_and]
            rw [hrw]
            exact ((measurableSet_indecisionCount O B.lo (B.hi - 1) (A₀.erase 1) A
              (2 * l * (A.card : ℝ))).inter
                (measurableSet_admittedFixed O B.lo B.hi B.gmin (A₀.erase 1) A α)).compl
          · have hz : {ω : Ω |
                ((A.filter (fun p => ¬ famGood O f kmin kmax (A₀.erase 1) p)).card : ℝ)
                  ≤ lcut * (A.card : ℝ)
                ∧ ¬ (((A.filter (fun p =>
                      ¬ decided O.mq B.lo (B.hi - 1) (A₀.erase 1) p ω)).card : ℝ)
                      ≤ 2 * l * (A.card : ℝ)
                    ∧ admitted O.mq B.lo B.hi B.gmin α (A₀.erase 1) A ω)} = (∅ : Set Ω) := by
              ext ω
              simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
              rintro ⟨h, -⟩
              exact hlight h
            rw [hz]
            exact MeasurableSet.empty
        have hset : {x : Run Ω S J | Disjoint P A ∧ A.card = B.npref ∧ 0 < A.card
            ∧ ((A.filter (fun p => ¬ famGood O f kmin kmax
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p)).card : ℝ)
                  ≤ lcut * (A.card : ℝ)
            ∧ ¬ (((A.filter (fun p => ¬ decided O.mq B.lo (B.hi - 1)
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p (oracleNoise x))).card : ℝ)
                  ≤ 2 * l * (A.card : ℝ)
                ∧ admitted O.mq B.lo B.hi B.gmin α
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) A (oracleNoise x))}
            = oracleNoise ⁻¹' {ω : Ω |
              ((A.filter (fun p => ¬ famGood O f kmin kmax
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1) p)).card : ℝ)
                ≤ lcut * (A.card : ℝ)
              ∧ ¬ (((A.filter (fun p => ¬ decided O.mq B.lo (B.hi - 1)
                      ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1) p ω)).card : ℝ)
                    ≤ 2 * l * (A.card : ℝ)
                  ∧ admitted O.mq B.lo B.hi B.gmin α
                      ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1) A ω)} := by
          ext x
          constructor
          · rintro ⟨-, -, -, hl, hn⟩
            exact ⟨hl, hn⟩
          · rintro ⟨hl, hn⟩
            exact ⟨hdraw.1, hdraw.2.1, hdraw.2.2, hl, hn⟩
        rw [hset]
        exact measurable_nz hω
      · have hempty : {x : Run Ω S J | Disjoint P A ∧ A.card = B.npref ∧ 0 < A.card
            ∧ ((A.filter (fun p => ¬ famGood O f kmin kmax
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p)).card : ℝ)
                  ≤ lcut * (A.card : ℝ)
            ∧ ¬ (((A.filter (fun p => ¬ decided O.mq B.lo (B.hi - 1)
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p (oracleNoise x))).card : ℝ)
                  ≤ 2 * l * (A.card : ℝ)
                ∧ admitted O.mq B.lo B.hi B.gmin α
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) A (oracleNoise x))}
            = (∅ : Set (Run Ω S J)) := by
          ext x
          simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
          rintro ⟨h1, h2, h3, -⟩
          exact hdraw ⟨h1, h2, h3⟩
        rw [hempty]
        exact MeasurableSet.empty
    · exact MeasurableSet.empty
  have hrw : retMiss O populations j B α l lcut f n₀ kmin kmax
      = {x : Run Ω S J | x ∈ (fun P C tt => if (1 : S) ∈ C then
          {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.npref)).image tt)
            ∧ ((Finset.univ : Finset (Fin B.npref)).image tt).card = B.npref
            ∧ 0 < ((Finset.univ : Finset (Fin B.npref)).image tt).card
            ∧ ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                ¬ famGood O f kmin kmax
                  ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p)).card : ℝ)
                ≤ lcut * (((Finset.univ : Finset (Fin B.npref)).image tt).card : ℝ)
            ∧ ¬ (((((Finset.univ : Finset (Fin B.npref)).image tt).filter
                    (fun p => ¬ decided O.mq B.lo (B.hi - 1)
                      ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p (oracleNoise x))).card : ℝ)
                  ≤ 2 * l * (((Finset.univ : Finset (Fin B.npref)).image tt).card : ℝ)
                ∧ admitted O.mq B.lo B.hi B.gmin α
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1)
                    ((Finset.univ : Finset (Fin B.npref)).image tt) (oracleNoise x))} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (fun i : Fin B.npref => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), retMiss,
      ← certOf_eq_image j B.npref x, ← clusterAt_eq_clusterOf O populations B x]
    constructor
    · rintro ⟨h1, h2, h3, h4, h5⟩
      exact ⟨h1, h2, h3, h4, h5⟩
    · rintro ⟨h1, h2, h3, h4, h5⟩
      exact ⟨h1, h2, h3, h4, h5⟩
  rw [hrw]
  exact measurableSet_of_run_data_cert populations j B _ hR

open scoped Classical in
/-- The lift of `ret_at_whp`.  At every table the state's own round fails only as often
as the two fractional counts and the two gate tails allow; the draws are charged elsewhere. -/
theorem measureReal_retMiss_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (j : J) (hj : j ∈ populations) (B : State)
    (α τ l lcut f E : ℝ) (kmin kmax nlo : ℕ)
    (hE : 0 ≤ E) (hElcut : E ≤ lcut) (hlcl : lcut ≤ l) (hτ : 0 ≤ τ)
    (hl1 : 2 * l ≤ 1) (hnloB : (nlo : ℝ) ≤ (1 - 2 * l) * (B.npref : ℝ))
    (hsig : O.η ≤ 1 / 2)
    (hdec : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ decided O.mq B.lo (B.hi - 1) F p ω} ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo (B.hi - 1) F p ω} ≤ E)
    (hga : ∀ n c : ℕ, nlo ≤ n → n ≤ c → c ≤ B.npref →
      (n : ℝ) * (1 / 2 + τ + τ)
        ≤ (n : ℝ) * (1 - O.η) - (1 - O.η) * (2 * lcut * (c : ℝ)))
    (hα : Real.exp (-2 * (B.gmin : ℝ) * τ ^ 2) ≤ α) :
    (runMeasure μ D Dsf).real (retMiss O populations j B α l lcut f B.gmin kmin kmax)
      ≤ Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
        + (Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
          + 2 * Real.exp (-2 * (nlo : ℝ) * τ ^ 2)) := by
  classical
  set R : ℝ := Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
    + (Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
      + 2 * Real.exp (-2 * (nlo : ℝ) * τ ^ 2)) with hR
  have hR0 : 0 ≤ R := by rw [hR]; positivity
  have hEnn : runMeasure μ D Dsf (retMiss O populations j B α l lcut f B.gmin kmin kmax)
      ≤ ENNReal.ofReal R := by
    refine runMeasure_slice_le D Dsf _
      (measurableSet_retMiss O populations j B α l lcut f B.gmin kmin kmax) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
      ae_cert_mem_Pre D Dsf Pre populations hsupp] with d hdP hdC
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.npref).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.nsuff).image (fun i => d.1.1 i)) with hCd
    set Ad : Finset S := (Finset.range B.npref).image (fun i => d.2 j i) with hAd
    have hP : ∀ q ∈ Pd, q ∈ Pre := by
      intro q hq
      obtain ⟨j', hj', hq'⟩ := Finset.mem_biUnion.1 hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hq'
      exact hdP j' hj' i
    have hA : ∀ p ∈ Ad, p ∈ Pre := by
      intro p hp
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hp
      exact hdC j hj i
    have hAcard : Ad.card ≤ B.npref := by
      rw [hAd]
      exact le_trans Finset.card_image_le (by simp)
    by_cases hdraw : Disjoint Pd Ad ∧ Ad.card = B.npref ∧ 0 < Ad.card
    · obtain ⟨hdisj, hAdm, hCpos⟩ := hdraw
      set fam : Ω → Finset S :=
        fun ω => (clusterAt O.mq populations ((ω, d) : Run Ω S J) B).erase 1 with hfamdef
      set T : Finset (Finset S) := Cd.powerset with hT
      set good : S → Finset (Finset S) := fun p => T.filter (fun t =>
        flipCount O t p ≤ (t.card : ℝ) * f ∧ kmin ≤ t.card ∧ t.card ≤ kmax) with hgood
      have hfamT : ∀ ω, fam ω ∈ T := fun ω => Finset.mem_powerset.2
        (fun v hv => clusterAt_subset O populations B _ (Finset.mem_erase.1 hv).2)
      have hmain := ret_at_whp hflat O Pd Cd Ad hP hA hdisj
        (readSet Pd Cd ∪ readSet Ad (Cd.erase 1)) Finset.subset_union_left
        (disjoint_gateReads hflat Pd Cd Ad hP hA hdisj)
        B.lo B.hi α τ l lcut B.gmin nlo
        (by
          have hcard : (Ad.card : ℝ) = (B.npref : ℝ) := by exact_mod_cast hAdm
          rw [hcard]
          nlinarith [hnloB])
        T good ∅ (Finset.mem_powerset.2 (Finset.empty_subset _))
        (fun t ht => Finset.mem_powerset.1 ht) fam hfamT
        (fun ω ω' h => congrArg (fun t : Finset S => t.erase 1)
          (clusterAt_congr O populations B d h))
        (fun ω p hp v hv => Finset.mem_union_right _ (mem_readSet hp
          (Finset.mem_erase.2 ⟨(Finset.mem_erase.1 hv).1,
            clusterAt_subset O populations B _ (Finset.mem_erase.1 hv).2⟩)))
        E hE hElcut hlcl hτ hsig
        (fun p _ A₀ _ hgp => hdec A₀ p (Finset.mem_filter.1 hgp).2.1
          (Finset.mem_filter.1 hgp).2.2.1 (Finset.mem_filter.1 hgp).2.2.2)
        (fun p _ A₀ _ hgp => hcut A₀ p (Finset.mem_filter.1 hgp).2.1
          (Finset.mem_filter.1 hgp).2.2.1 (Finset.mem_filter.1 hgp).2.2.2)
        (fun n hn hnc => hga n Ad.card hn hnc hAcard)
        hα
      have hsec : {ω : Ω | ((ω, d) : Run Ω S J)
            ∈ retMiss O populations j B α l lcut f B.gmin kmin kmax}
          ⊆ {ω : Ω | ((Ad.filter (fun p => fam ω ∉ good p)).card : ℝ) ≤ lcut * (Ad.card : ℝ)
            ∧ ¬ ((((Ad.filter (fun p =>
                    ¬ decided O.mq B.lo (B.hi - 1) (fam ω) p ω)).card : ℝ)
                  ≤ 2 * l * (Ad.card : ℝ))
              ∧ admitted O.mq B.lo B.hi B.gmin α (fam ω) Ad ω)} := by
        rintro ω ⟨-, -, -, hlight, hbad⟩
        refine ⟨le_trans (le_of_eq ?_) hlight, hbad⟩
        refine congrArg (fun t : Finset S => (t.card : ℝ)) (Finset.filter_congr ?_)
        intro p _
        simp only [hgood, Finset.mem_filter, hfamT ω, true_and]
        exact Iff.rfl
      have hcardR : (Ad.card : ℝ) = (B.npref : ℝ) := by exact_mod_cast hAdm
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
      exact ENNReal.ofReal_le_ofReal (hmain.trans (le_of_eq (by rw [hcardR])))
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J)
          ∈ retMiss O populations j B α l lcut f B.gmin kmin kmax} = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨h1, h2, h3, -⟩
        exact hdraw ⟨h1, h2, h3⟩
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (retMiss O populations j B α l lcut f B.gmin kmin kmax)).toReal
      ≤ (ENNReal.ofReal R).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = R := ENNReal.toReal_ofReal hR0

open scoped Classical in
lemma measurableSet_miscutCount (O : Oracle μ S) (f : ℝ) (kmin kmax lo ha : ℕ)
    (A₀ A : Finset S) (r : ℝ) :
    MeasurableSet {ω | r < ((A.filter (fun p => famGood O f kmin kmax A₀ p
        ∧ ¬ cutCorrect O lo ha A₀ p ω)).card : ℝ)} := by
  classical
  refine noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O
    (fun p ω => famGood O f kmin kmax A₀ p ∧ ¬ cutCorrect O lo ha A₀ p ω)
    (fun p _ => ?_) (fun U => r < (U.card : ℝ)))
  by_cases hg : famGood O f kmin kmax A₀ p
  · simpa [hg] using measurableSet_cutCorrect' O lo ha A₀ p
  · simp [hg]

open scoped Classical in
/-- The certification draws are fresh, and still the vote misfires on more than an `lcut`
fraction of the sample at prefixes the family barely flips. -/
noncomputable def validMiss (O : Oracle μ S) (populations : Finset J) (j : J) (B : State)
    (lcut f : ℝ) (kmin kmax : ℕ) : Set (Run Ω S J) :=
  {x | Disjoint (prefixesAt populations B.npref x) (certOf j B.npref x)
    ∧ (certOf j B.npref x).card = B.npref
    ∧ lcut * ((certOf j B.npref x).card : ℝ)
      < (((certOf j B.npref x).filter (fun p =>
          famGood O f kmin kmax ((clusterAt O.mq populations x B).erase 1) p
          ∧ ¬ cutCorrect O B.lo (B.hi - 1) ((clusterAt O.mq populations x B).erase 1) p
            (oracleNoise x))).card : ℝ)}

open scoped Classical in
lemma measurableSet_validMiss (O : Oracle μ S) (populations : Finset J) (j : J) (B : State)
    (lcut f : ℝ) (kmin kmax : ℕ) :
    MeasurableSet (validMiss O populations j B lcut f kmin kmax) := by
  classical
  have hR : ∀ (P C : Finset S) (tt : Fin B.npref → S), MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.npref)).image tt)
        ∧ ((Finset.univ : Finset (Fin B.npref)).image tt).card = B.npref
        ∧ lcut * (((Finset.univ : Finset (Fin B.npref)).image tt).card : ℝ)
          < ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
              famGood O f kmin kmax
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p
              ∧ ¬ cutCorrect O B.lo (B.hi - 1)
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p
                (oracleNoise x))).card : ℝ)} else ∅) := by
    intro P C tt
    split_ifs with hone
    · set A : Finset S := (Finset.univ : Finset (Fin B.npref)).image tt with hA
      by_cases hdraw : Disjoint P A ∧ A.card = B.npref
      · have hω : MeasurableSet {ω : Ω | lcut * (A.card : ℝ)
            < ((A.filter (fun p => famGood O f kmin kmax
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1) p
              ∧ ¬ cutCorrect O B.lo (B.hi - 1)
                ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1) p ω)).card : ℝ)} :=
          measurableSet_of_fam (T := C.powerset)
            (fun ω => Finset.mem_powerset.2
              (clusterOf_subset O B.cn B.cd B.sc B.scd P C B.k ω hone))
            (fun A₀ => measurableSet_clusterOf O B.cn B.cd B.sc B.scd P C B.k hone A₀)
            (fun A₀ => {ω : Ω | lcut * (A.card : ℝ)
              < ((A.filter (fun p => famGood O f kmin kmax (A₀.erase 1) p
                ∧ ¬ cutCorrect O B.lo (B.hi - 1) (A₀.erase 1) p ω)).card : ℝ)})
            (fun A₀ => measurableSet_miscutCount O f kmin kmax B.lo (B.hi - 1) (A₀.erase 1) A
              (lcut * (A.card : ℝ)))
        have hset : {x : Run Ω S J | Disjoint P A ∧ A.card = B.npref
            ∧ lcut * (A.card : ℝ)
              < ((A.filter (fun p => famGood O f kmin kmax
                  ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p
                ∧ ¬ cutCorrect O B.lo (B.hi - 1)
                  ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p
                  (oracleNoise x))).card : ℝ)}
            = oracleNoise ⁻¹' {ω : Ω | lcut * (A.card : ℝ)
              < ((A.filter (fun p => famGood O f kmin kmax
                  ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1) p
                ∧ ¬ cutCorrect O B.lo (B.hi - 1)
                  ((clusterOf O B.cn B.cd B.sc B.scd P C B.k ω).erase 1) p ω)).card : ℝ)} := by
          ext x
          exact ⟨fun h => h.2.2, fun h => ⟨hdraw.1, hdraw.2, h⟩⟩
        rw [hset]
        exact measurable_nz hω
      · have hempty : {x : Run Ω S J | Disjoint P A ∧ A.card = B.npref
            ∧ lcut * (A.card : ℝ)
              < ((A.filter (fun p => famGood O f kmin kmax
                  ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p
                ∧ ¬ cutCorrect O B.lo (B.hi - 1)
                  ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p
                  (oracleNoise x))).card : ℝ)} = (∅ : Set (Run Ω S J)) := by
          ext x
          simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
          rintro ⟨h1, h2, -⟩
          exact hdraw ⟨h1, h2⟩
        rw [hempty]
        exact MeasurableSet.empty
    · exact MeasurableSet.empty
  have hrw : validMiss O populations j B lcut f kmin kmax
      = {x : Run Ω S J | x ∈ (fun P C tt => if (1 : S) ∈ C then
          {x : Run Ω S J | Disjoint P ((Finset.univ : Finset (Fin B.npref)).image tt)
            ∧ ((Finset.univ : Finset (Fin B.npref)).image tt).card = B.npref
            ∧ lcut * (((Finset.univ : Finset (Fin B.npref)).image tt).card : ℝ)
              < ((((Finset.univ : Finset (Fin B.npref)).image tt).filter (fun p =>
                  famGood O f kmin kmax
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p
                  ∧ ¬ cutCorrect O B.lo (B.hi - 1)
                    ((clusterOf O B.cn B.cd B.sc B.scd P C B.k (oracleNoise x)).erase 1) p
                    (oracleNoise x))).card : ℝ)} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (fun i : Fin B.npref => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), validMiss,
      ← certOf_eq_image j B.npref x, ← clusterAt_eq_clusterOf O populations B x]
    constructor
    · rintro ⟨h1, h2, h3⟩
      exact ⟨h1, h2, h3⟩
    · rintro ⟨h1, h2, h3⟩
      exact ⟨h1, h2, h3⟩
  rw [hrw]
  exact measurableSet_of_run_data_cert populations j B _ hR

open scoped Classical in
/-- Hoeffding over the certification sample: the family is chosen from the table's reads, the
sample's reads are at other strings, so the light prefixes misfire independently, each at most
`E`. -/
theorem measureReal_validMiss_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (j : J) (hj : j ∈ populations) (B : State)
    (lcut f E : ℝ) (kmin kmax : ℕ) (hE : 0 ≤ E) (hElcut : E ≤ lcut)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo (B.hi - 1) F p ω} ≤ E) :
    (runMeasure μ D Dsf).real (validMiss O populations j B lcut f kmin kmax)
      ≤ Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2) := by
  classical
  set R : ℝ := Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2) with hR
  have hR0 : 0 ≤ R := Real.exp_nonneg _
  have hEnn : runMeasure μ D Dsf (validMiss O populations j B lcut f kmin kmax)
      ≤ ENNReal.ofReal R := by
    refine runMeasure_slice_le D Dsf _
      (measurableSet_validMiss O populations j B lcut f kmin kmax) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
      ae_cert_mem_Pre D Dsf Pre populations hsupp] with d hdP hdC
    set Pd : Finset S := populations.biUnion
      (fun j => (Finset.range B.npref).image (fun i => d.1.2 j i)) with hPd
    set Cd : Finset S := insert 1 ((Finset.range B.nsuff).image (fun i => d.1.1 i)) with hCd
    set Ad : Finset S := (Finset.range B.npref).image (fun i => d.2 j i) with hAd
    have hP : ∀ q ∈ Pd, q ∈ Pre := by
      intro q hq
      obtain ⟨j', hj', hq'⟩ := Finset.mem_biUnion.1 hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hq'
      exact hdP j' hj' i
    have hA : ∀ p ∈ Ad, p ∈ Pre := by
      intro p hp
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hp
      exact hdC j hj i
    by_cases hdraw : Disjoint Pd Ad ∧ Ad.card = B.npref
    · obtain ⟨hdisj, hAdm⟩ := hdraw
      set fam : Ω → Finset S :=
        fun ω => (clusterAt O.mq populations ((ω, d) : Run Ω S J) B).erase 1 with hfamdef
      set T : Finset (Finset S) := Cd.powerset with hT
      set good : S → Finset (Finset S) := fun p => T.filter (fun t =>
        flipCount O t p ≤ (t.card : ℝ) * f ∧ kmin ≤ t.card ∧ t.card ≤ kmax) with hgood
      have hfamT : ∀ ω, fam ω ∈ T := fun ω => Finset.mem_powerset.2
        (fun v hv => clusterAt_subset O populations B _ (Finset.mem_erase.1 hv).2)
      have hmain := miscut_frac_le hflat O Pd Cd Ad hP hA hdisj B.lo (B.hi - 1) T good ∅
        (Finset.mem_powerset.2 (Finset.empty_subset _))
        (fun t ht => Finset.mem_powerset.1 ht) fam hfamT
        (fun ω ω' h => congrArg (fun t : Finset S => t.erase 1)
          (clusterAt_congr O populations B d h))
        E lcut hE hElcut
        (fun p _ A₀ _ hgp => hcut A₀ p (Finset.mem_filter.1 hgp).2.1
          (Finset.mem_filter.1 hgp).2.2.1 (Finset.mem_filter.1 hgp).2.2.2)
      have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ validMiss O populations j B lcut f kmin kmax}
          ⊆ {ω : Ω | lcut * (Ad.card : ℝ) < ((Ad.filter (fun p => fam ω ∈ good p
              ∧ ¬ cutCorrect O B.lo (B.hi - 1) (fam ω) p ω)).card : ℝ)} := by
        rintro ω ⟨-, -, hmis⟩
        refine lt_of_lt_of_le hmis (le_of_eq ?_)
        refine congrArg (fun t : Finset S => (t.card : ℝ)) (Finset.filter_congr ?_)
        intro p _
        simp only [hgood, Finset.mem_filter, hfamT ω, true_and, famGood]
        rfl
      have hcardR : (Ad.card : ℝ) = (B.npref : ℝ) := by exact_mod_cast hAdm
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def]
      exact ENNReal.ofReal_le_ofReal (hmain.trans (le_of_eq (by rw [hcardR])))
    · have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ validMiss O populations j B lcut f kmin kmax}
          = (∅ : Set Ω) := by
        ext ω
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨h1, h2, -⟩
        exact hdraw ⟨h1, h2⟩
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (validMiss O populations j B lcut f kmin kmax)).toReal
      ≤ (ENNReal.ofReal R).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = R := ENNReal.toReal_ofReal hR0

open scoped Classical in
set_option maxHeartbeats 1000000 in
/-- Part 1 at one state and one population.  A family of the round's size that is wrong on
more than `εcov` of the population is seen to be by the certification sample, and the wrong
prefixes the sample holds are then either ones the family flips — few, once no member flips
much — or ones where the vote of a barely-flipped family misfires — few, by Hoeffding. -/
theorem measureReal_validFail_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (j : J) (hj : j ∈ populations) (B : State)
    (hkpos : 0 < B.k) (hhi : 1 ≤ B.hi)
    (εcov lcut f E Δp ρ th : ℝ) (hε0 : 0 ≤ εcov) (hE : 0 ≤ E) (hElcut : E ≤ lcut)
    (hΔp : 0 ≤ Δp) (hth : 0 ≤ th) (hf0 : 0 < f)
    (hbudget : Δp / f + th + lcut ≤ 3 * εcov / 4)
    (hρ : ∀ j' ∈ populations, collisionMass (D j') ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      B.k - 1 ≤ F.card → F.card ≤ B.k - 1 →
      μ.real {ω | ¬ cutCorrect O B.lo (B.hi - 1) F p ω} ≤ E) :
    (runMeasure μ D Dsf).real
        ({x : Run Ω S J | B.k ≤ (clusterAt O.mq populations x B).card
            ∧ ¬ (1 - εcov ≤ (D j).real
              {p | cutCorrect O B.lo B.hi (clusterAt O.mq populations x B) p (oracleNoise x)})}
          \ {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O.mq populations x B, flipMass O (D j) v ≤ Δp})
      ≤ ((populations.card : ℝ) + 1) * (B.npref : ℝ) ^ 2 * ρ
        + (Real.exp (-2 * (B.npref : ℝ) * (εcov / 4) ^ 2)
          + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
            + Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2))) := by
  classical
  set E1 : Set (Run Ω S J) :=
    {x | ¬ Function.Injective (fun i : Fin B.npref => certPrefix j i.val x)} with hE1
  set E2 : Set (Run Ω S J) :=
    {x | ¬ Disjoint (prefixesAt populations B.npref x) (certOf j B.npref x)} with hE2
  set E3 : Set (Run Ω S J) := hitShort O populations (D j) j B εcov (εcov / 4) with hE3
  set E7 : Set (Run Ω S J) := heavyHits O populations (D j) j B f (Δp / f) th with hE7
  set E8 : Set (Run Ω S J) := validMiss O populations j B lcut f (B.k - 1) (B.k - 1) with hE8
  have hsub : ({x : Run Ω S J | B.k ≤ (clusterAt O.mq populations x B).card
        ∧ ¬ (1 - εcov ≤ (D j).real
          {p | cutCorrect O B.lo B.hi (clusterAt O.mq populations x B) p (oracleNoise x)})}
      \ {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O.mq populations x B, flipMass O (D j) v ≤ Δp})
      ⊆ (E1 ∪ E2) ∪ (E3 ∪ (E7 ∪ E8)) := by
    rintro x ⟨⟨hsize, hfail⟩, hclean⟩
    simp only [Set.mem_setOf_eq, not_not] at hclean
    by_cases h1 : Function.Injective (fun i : Fin B.npref => certPrefix j i.val x)
    swap
    · exact Or.inl (Or.inl h1)
    by_cases h2 : Disjoint (prefixesAt populations B.npref x) (certOf j B.npref x)
    swap
    · exact Or.inl (Or.inr h2)
    have hcard : ((certOf j B.npref x).card : ℝ) = (B.npref : ℝ) := by
      have hinjOn : Set.InjOn (fun i => certPrefix j i x) ↑(Finset.range B.npref) := by
        intro a ha b hb hab
        have := h1 (show (fun i : Fin B.npref => certPrefix j i.val x)
            ⟨a, Finset.mem_range.1 (by simpa using ha)⟩
          = (fun i : Fin B.npref => certPrefix j i.val x)
            ⟨b, Finset.mem_range.1 (by simpa using hb)⟩ from hab)
        simpa using congrArg Fin.val this
      have h := congrArg (fun n : ℕ => (n : ℝ))
        (show (certOf j B.npref x).card = B.npref by
          unfold certOf
          rw [Finset.card_image_of_injOn hinjOn, Finset.card_range])
      exact h
    -- the family is exactly the round's size
    have hone : (1 : S) ∈ clusterAt O.mq populations x B :=
      one_mem_clusterAround O B.cn B.cd _ _ (oracleNoise x) B.k
    have hle : (clusterAt O.mq populations x B).card ≤ B.k :=
      clusterAround_card_le O B.cn B.cd _ _ (oracleNoise x) B.k hkpos
    have herase : ((clusterAt O.mq populations x B).erase 1).card = B.k - 1 := by
      rw [Finset.card_erase_of_mem hone, le_antisymm hle hsize]
    have hpop : εcov ≤ (D j).real
        {p | ¬ cutCorrect O B.lo B.hi (clusterAt O.mq populations x B) p (oracleNoise x)} := by
      rw [show {p : S | ¬ cutCorrect O B.lo B.hi
            (clusterAt O.mq populations x B) p (oracleNoise x)}
          = {p : S | cutCorrect O B.lo B.hi
            (clusterAt O.mq populations x B) p (oracleNoise x)}ᶜ from rfl,
        measureReal_compl (measurableSet_of_countable _), measureReal_def, measure_univ,
        ENNReal.toReal_one]
      push_neg at hfail
      linarith
    by_cases hshort : (((certOf j B.npref x).filter (fun p =>
          ¬ cutCorrect O B.lo B.hi (clusterAt O.mq populations x B) p (oracleNoise x))).card : ℝ)
        ≤ (B.npref : ℝ) * (εcov - εcov / 4)
    · exact Or.inr (Or.inl ⟨h1, hpop, hshort⟩)
    push_neg at hshort
    -- the mass the family flips, by Markov over its members
    have hmass : (D j).real {p | ¬ (flipCount O
          ((clusterAt O.mq populations x B).erase 1) p
        ≤ ((((clusterAt O.mq populations x B).erase 1).card : ℝ)) * f)} ≤ Δp / f := by
      set F : Finset S := (clusterAt O.mq populations x B).erase 1 with hFdef
      rcases Nat.eq_zero_or_pos F.card with hF0 | hFpos
      · have hz : {p : S | ¬ (flipCount O F p ≤ ((F.card : ℝ)) * f)} = (∅ : Set S) := by
          ext p
          simp only [Set.mem_empty_iff_false, iff_false, not_not]
          have hFe : F = ∅ := Finset.card_eq_zero.1 hF0
          simp [flipCount, hFe]
        rw [hz]
        simp only [measureReal_empty]
        exact div_nonneg hΔp hf0.le
      · have hsetle : {p : S | ¬ (flipCount O F p ≤ ((F.card : ℝ)) * f)}
            ⊆ {p | f * (F.card : ℝ) ≤ flipCount O F p} := by
          intro p hp
          have hlt := not_le.1 hp
          rw [Set.mem_setOf_eq, mul_comm]
          linarith
        refine le_trans (measureReal_mono hsetle (measure_ne_top _ _)) ?_
        exact flipCount_mass_le O (D j) F Δp f hf0 hFpos
          (fun v hv => hclean v (Finset.mem_erase.1 hv).2)
    by_cases h7 : (B.npref : ℝ) * (Δp / f + th)
        ≤ (((certOf j B.npref x).filter (fun p =>
          ¬ (flipCount O ((clusterAt O.mq populations x B).erase 1) p
            ≤ ((((clusterAt O.mq populations x B).erase 1).card : ℝ)) * f))).card : ℝ)
    · exact Or.inr (Or.inr (Or.inl ⟨hmass, h7⟩))
    replace h7 := not_le.1 h7
    refine Or.inr (Or.inr (Or.inr ⟨h2, ?_, ?_⟩))
    · exact_mod_cast hcard
    · -- every wrong prefix is heavy, or light and misread by the seed-dropped vote
      have hcover : ((certOf j B.npref x).filter (fun p =>
            ¬ cutCorrect O B.lo B.hi (clusterAt O.mq populations x B) p (oracleNoise x)))
          ⊆ ((certOf j B.npref x).filter (fun p =>
              ¬ (flipCount O ((clusterAt O.mq populations x B).erase 1) p
                ≤ ((((clusterAt O.mq populations x B).erase 1).card : ℝ)) * f)))
            ∪ ((certOf j B.npref x).filter (fun p =>
              famGood O f (B.k - 1) (B.k - 1) ((clusterAt O.mq populations x B).erase 1) p
              ∧ ¬ cutCorrect O B.lo (B.hi - 1) ((clusterAt O.mq populations x B).erase 1) p
                (oracleNoise x))) := by
        intro p hp
        obtain ⟨hpC, hbad⟩ := Finset.mem_filter.1 hp
        by_cases hh : flipCount O ((clusterAt O.mq populations x B).erase 1) p
            ≤ ((((clusterAt O.mq populations x B).erase 1).card : ℝ)) * f
        · refine Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hpC, ⟨hh, ?_, ?_⟩, ?_⟩)
          · rw [herase]
          · rw [herase]
          · exact fun hc => hbad (cutCorrect_of_erase O B.lo B.hi hhi _ p (oracleNoise x) hc)
        · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hpC, hh⟩)
      have hcnt := le_trans (Finset.card_le_card hcover) (Finset.card_union_le _ _)
      have hcntR : (((certOf j B.npref x).filter (fun p =>
            ¬ cutCorrect O B.lo B.hi (clusterAt O.mq populations x B) p (oracleNoise x))).card : ℝ)
          ≤ (((certOf j B.npref x).filter (fun p =>
              ¬ (flipCount O ((clusterAt O.mq populations x B).erase 1) p
                ≤ ((((clusterAt O.mq populations x B).erase 1).card : ℝ)) * f))).card : ℝ)
            + (((certOf j B.npref x).filter (fun p =>
              famGood O f (B.k - 1) (B.k - 1) ((clusterAt O.mq populations x B).erase 1) p
              ∧ ¬ cutCorrect O B.lo (B.hi - 1) ((clusterAt O.mq populations x B).erase 1) p
                (oracleNoise x))).card : ℝ) := by exact_mod_cast hcnt
      rw [hcard]
      have hm0 : (0 : ℝ) ≤ (B.npref : ℝ) := Nat.cast_nonneg _
      have hb := mul_le_mul_of_nonneg_left hbudget hm0
      linarith [hshort, h7, hcntR, hb]
  calc (runMeasure μ D Dsf).real
        ({x : Run Ω S J | B.k ≤ (clusterAt O.mq populations x B).card
            ∧ ¬ (1 - εcov ≤ (D j).real
              {p | cutCorrect O B.lo B.hi (clusterAt O.mq populations x B) p (oracleNoise x)})}
          \ {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O.mq populations x B, flipMass O (D j) v ≤ Δp})
      ≤ (runMeasure μ D Dsf).real ((E1 ∪ E2) ∪ (E3 ∪ (E7 ∪ E8))) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ((runMeasure μ D Dsf).real E1 + (runMeasure μ D Dsf).real E2)
        + ((runMeasure μ D Dsf).real E3 + ((runMeasure μ D Dsf).real E7
          + (runMeasure μ D Dsf).real E8)) := by
        have h12 := measureReal_union_le (μ := runMeasure μ D Dsf) E1 E2
        have h78 := measureReal_union_le (μ := runMeasure μ D Dsf) E7 E8
        have h378 := measureReal_union_le (μ := runMeasure μ D Dsf) E3 (E7 ∪ E8)
        have hall := measureReal_union_le (μ := runMeasure μ D Dsf) (E1 ∪ E2) (E3 ∪ (E7 ∪ E8))
        linarith
    _ ≤ ((B.npref : ℝ) ^ 2 * ρ + (populations.card : ℝ) * (B.npref : ℝ) ^ 2 * ρ)
        + (Real.exp (-2 * (B.npref : ℝ) * (εcov / 4) ^ 2)
          + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
            + Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2))) := by
        gcongr
        · exact cert_not_injective_le D Dsf j B.npref ρ (hρ j hj) hρ0
        · exact prefix_cert_disjoint_le D Dsf populations j B.npref ρ hρ (hρ j hj) hρ0
        · exact measureReal_hitShort_le D Dsf O populations j B εcov (εcov / 4) hε0
            (by positivity)
        · exact measureReal_heavyHits_le D Dsf O populations j B f (Δp / f) th hth
        · exact measureReal_validMiss_le hflat O populations D Dsf hsupp j hj B lcut f E
            (B.k - 1) (B.k - 1) hE hElcut hcut
    _ = ((populations.card : ℝ) + 1) * (B.npref : ℝ) ^ 2 * ρ
        + (Real.exp (-2 * (B.npref : ℝ) * (εcov / 4) ^ 2)
          + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
            + Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2))) := by ring

open scoped Classical in
/-- The round's own test at one population: the family's size, the FNR count and the gate. -/
noncomputable def retAt (O : Oracle μ S) (populations : Finset J)
    (indecisionLimit α : ℝ) (B : State) (j : J) : Set (Run Ω S J) :=
  {x | B.k ≤ (clusterAt O.mq populations x B).card
    ∧ (((certOf j B.npref x).filter (fun p => ¬ decided O.mq B.lo (B.hi - 1)
          ((clusterAt O.mq populations x B).erase 1) p (oracleNoise x))).card : ℝ)
        ≤ indecisionLimit * ((certOf j B.npref x).card : ℝ)
    ∧ admitted O.mq B.lo B.hi B.gmin α ((clusterAt O.mq populations x B).erase 1)
        (certOf j B.npref x) (oracleNoise x)}

lemma mem_ret_of_retAt (O : Oracle μ S) (populations : Finset J) (hpop : populations.Nonempty)
    (indecisionLimit α : ℝ) (B : State) (x : Run Ω S J)
    (h : ∀ j ∈ populations, x ∈ retAt O populations indecisionLimit α B j) :
    x ∈ ret O.mq populations indecisionLimit α B := by
  obtain ⟨j₀, hj₀⟩ := hpop
  exact ⟨(h j₀ hj₀).1, fun j hj => (h j hj).2.1, fun j hj => (h j hj).2.2⟩

open scoped Classical in
/-- The runs where the clustering stalls on the seed, or overshoots the round's size.  The
gate refuses a stalled family through whichever side the population populates, so this is the liveness
half's obligation, not the round's. -/
noncomputable def stalled (O : Oracle μ S) (populations : Finset J) (B : State)
    (kmin kmax : ℕ) : Set (Run Ω S J) :=
  {x | ¬ (kmin ≤ ((clusterAt O.mq populations x B).erase 1).card
      ∧ ((clusterAt O.mq populations x B).erase 1).card ≤ kmax)}

open scoped Classical in
/-- A pool of `k` screened candidates is a family of `k`.  The seed is one of them and
the ranking keeps it, so the family the gate sees has `k − 1` members besides the seed. -/
lemma stalled_subset (O : Oracle μ S) (populations : Finset J) (B : State)
    (hcd : B.cn < B.cd) (hkpos : 0 < B.k) :
    stalled O populations B (B.k - 1) (B.k - 1)
      ⊆ {x : Run Ω S J | ¬ (B.k ≤ (screenedAt O.mq populations B x).card)} := by
  intro x hx
  by_contra hk
  simp only [Set.mem_setOf_eq, Classical.not_not] at hk
  have hcard : (clusterAt O.mq populations x B).card = B.k :=
    clusterAround_card O hcd (prefixesAt populations B.npref x) (screenedAt O.mq populations B x)
      (oracleNoise x) B.k (one_mem_screenedAt O populations B x) hk hkpos
  have hone : (1 : S) ∈ clusterAt O.mq populations x B :=
    one_mem_clusterAround O B.cn B.cd _ _ (oracleNoise x) B.k
  have herase : ((clusterAt O.mq populations x B).erase 1).card = B.k - 1 := by
    rw [Finset.card_erase_of_mem hone, hcard]
  exact hx ⟨le_of_eq herase.symm, le_of_eq herase⟩

open scoped Classical in
/-- The liveness half's obligation, discharged: the clustering has a family of the round's
own size except on the events `measureReal_smallScreen_le` prices. -/
theorem measureReal_stalled_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (B : State) (hcd : B.cn < B.cd)
    (hkpos : 0 < B.k) (j₀ : J) (hj₀ : j₀ ∈ populations)
    (γ pAP t ρsf ρ : ℝ) (hγ : 0 ≤ γ) (hpAP0 : 0 ≤ pAP) (ht : 0 ≤ t)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hscd : 0 < B.scd) (hsig : O.η ≤ 1 / 2)
    (hsc : (B.scd : ℝ) * (2 * γ) ≤ (B.sc : ℝ))
    (hcount : (B.k : ℝ) ≤ (B.nsuff : ℝ) * (pAP - t))
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hρj : collisionMass (D j₀) ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runMeasure μ D Dsf).real (stalled O populations B (B.k - 1) (B.k - 1))
      ≤ (B.nsuff : ℝ) ^ 2 * ρsf + (Real.exp (-2 * (B.nsuff : ℝ) * t ^ 2)
        + ((B.npref : ℝ) ^ 2 * ρ
          + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2))) :=
  le_trans (measureReal_mono (stalled_subset O populations B hcd hkpos) (measure_ne_top _ _))
    (measureReal_smallScreen_le hflat O populations D Dsf hsupp B hcd j₀ hj₀ γ pAP t ρsf ρ
      hγ hpAP0 ht hpAPBound hscd hsig hsc hcount hρsf hρsf0 hρj hρ0)

open scoped Classical in
set_option maxHeartbeats 1000000 in
/-- The round returns at one population.  Everything outside `retMiss` is a fact about
the draws: the certification prefixes repeat, or meet the table, or under-represent a class,
or the family is dirty and the sample sees it.  The cluster's own size is the liveness
half's business and is carried as `Estall`. -/
theorem measureReal_notRetAt_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (j : J) (hj : j ∈ populations) (B : State)
    (hmpos : 0 < B.npref) (hsig : O.η ≤ 1 / 2)
    (α τ l lcut f E Δp ρ th : ℝ) (kmin kmax nlo : ℕ) (hkmin : B.k ≤ kmin + 1)
    (hE : 0 ≤ E) (hElcut : E ≤ lcut) (hlcl : lcut ≤ l) (hτ : 0 ≤ τ)
    (hl1 : 2 * l ≤ 1) (hnloB : (nlo : ℝ) ≤ (1 - 2 * l) * (B.npref : ℝ))
    (hρ : ∀ j' ∈ populations, collisionMass (D j') ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hΔp : 0 ≤ Δp) (hth : 0 ≤ th) (hf0 : 0 < f)
    (hheavy : Δp / f + th ≤ lcut)
    (Estall : ℝ) (hstall : (runMeasure μ D Dsf).real (stalled O populations B kmin kmax) ≤ Estall)
    (Edirty : ℝ) (hdirty : (runMeasure μ D Dsf).real
      {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O.mq populations x B, flipMass O (D j) v ≤ Δp}
        ≤ Edirty)
    (hdec : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ decided O.mq B.lo (B.hi - 1) F p ω} ≤ E)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      kmin ≤ F.card → F.card ≤ kmax →
      μ.real {ω | ¬ cutCorrect O B.lo (B.hi - 1) F p ω} ≤ E)
    (hga : ∀ n c : ℕ, nlo ≤ n → n ≤ c → c ≤ B.npref →
      (n : ℝ) * (1 / 2 + τ + τ)
        ≤ (n : ℝ) * (1 - O.η) - (1 - O.η) * (2 * lcut * (c : ℝ)))
    (hα : Real.exp (-2 * (B.gmin : ℝ) * τ ^ 2) ≤ α) :
    (runMeasure μ D Dsf).real
        {x : Run Ω S J | x ∉ retAt O populations (2 * l) α B j}
      ≤ ((populations.card : ℝ) + 1) * (B.npref : ℝ) ^ 2 * ρ
        + (Estall + (Edirty + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
          + (Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
            + (Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
              + 2 * Real.exp (-2 * (nlo : ℝ) * τ ^ 2)))))) := by
  classical
  set E1 : Set (Run Ω S J) :=
    {x | ¬ Function.Injective (fun i : Fin B.npref => certPrefix j i.val x)} with hE1
  set E2 : Set (Run Ω S J) :=
    {x | ¬ Disjoint (prefixesAt populations B.npref x) (certOf j B.npref x)} with hE2
  set E5 : Set (Run Ω S J) := stalled O populations B kmin kmax with hE5
  set E6 : Set (Run Ω S J) :=
    {x | ¬ ∀ v ∈ clusterAt O.mq populations x B, flipMass O (D j) v ≤ Δp} with hE6
  set E7 : Set (Run Ω S J) := heavyHits O populations (D j) j B f (Δp / f) th with hE7
  set E8 : Set (Run Ω S J) := retMiss O populations j B α l lcut f B.gmin kmin kmax with hE8
  have hsub : {x : Run Ω S J | x ∉ retAt O populations (2 * l) α B j}
      ⊆ (E1 ∪ E2) ∪ (E5 ∪ (E6 ∪ (E7 ∪ E8))) := by
    intro x hx
    by_cases h1 : Function.Injective (fun i : Fin B.npref => certPrefix j i.val x)
    · by_cases h2 : Disjoint (prefixesAt populations B.npref x) (certOf j B.npref x)
      · by_cases h5 : kmin ≤ ((clusterAt O.mq populations x B).erase 1).card
            ∧ ((clusterAt O.mq populations x B).erase 1).card ≤ kmax
        · by_cases h6 : ∀ v ∈ clusterAt O.mq populations x B, flipMass O (D j) v ≤ Δp
          · -- the certification sample is good and the family is clean
            have hcard : ((certOf j B.npref x).card : ℝ) = (B.npref : ℝ) := by
              have hinjOn : Set.InjOn (fun i => certPrefix j i x) ↑(Finset.range B.npref) := by
                intro a ha b hb hab
                have := h1 (show (fun i : Fin B.npref => certPrefix j i.val x)
                    ⟨a, Finset.mem_range.1 (by simpa using ha)⟩
                  = (fun i : Fin B.npref => certPrefix j i.val x)
                    ⟨b, Finset.mem_range.1 (by simpa using hb)⟩ from hab)
                simpa using congrArg Fin.val this
              unfold certOf
              rw [Finset.card_image_of_injOn hinjOn, Finset.card_range]
            have hCpos : 0 < (certOf j B.npref x).card := by
              have : (0 : ℝ) < ((certOf j B.npref x).card : ℝ) := by
                rw [hcard]; exact_mod_cast hmpos
              exact_mod_cast this
            -- Markov over the family: the mean flip fraction is at most `Δp`, so the mass
            -- where an `f` fraction flips is `Δp / f`, with no family-size factor.
            have hmass : (D j).real {p | ¬ (flipCount O
                  ((clusterAt O.mq populations x B).erase 1) p
                ≤ ((((clusterAt O.mq populations x B).erase 1).card : ℝ)) * f)}
                ≤ Δp / f := by
              set F : Finset S := (clusterAt O.mq populations x B).erase 1 with hFdef
              rcases Nat.eq_zero_or_pos F.card with hF0 | hFpos
              · have hz : {p : S | ¬ (flipCount O F p ≤ ((F.card : ℝ)) * f)}
                    = (∅ : Set S) := by
                  ext p
                  simp only [Set.mem_empty_iff_false, iff_false, not_not]
                  have hFe : F = ∅ := Finset.card_eq_zero.1 hF0
                  simp [flipCount, hFe]
                rw [hz]
                simp only [measureReal_empty]
                exact div_nonneg hΔp hf0.le
              · have hsetle : {p : S | ¬ (flipCount O F p ≤ ((F.card : ℝ)) * f)}
                    ⊆ {p | f * (F.card : ℝ) ≤ flipCount O F p} := by
                  intro p hp
                  have hlt := not_le.1 hp
                  rw [Set.mem_setOf_eq, mul_comm]
                  linarith
                refine le_trans (measureReal_mono hsetle (measure_ne_top _ _)) ?_
                exact flipCount_mass_le O (D j) F Δp f hf0 hFpos
                  (fun v hv => h6 v (Finset.mem_erase.1 hv).2)
            by_cases h7 : (B.npref : ℝ) * (Δp / f + th)
                ≤ (((certOf j B.npref x).filter (fun p =>
                  ¬ (flipCount O ((clusterAt O.mq populations x B).erase 1) p
                    ≤ ((((clusterAt O.mq populations x B).erase 1).card : ℝ)) * f))).card : ℝ)
            · exact Or.inr (Or.inr (Or.inr (Or.inl ⟨hmass, h7⟩)))
            · refine Or.inr (Or.inr (Or.inr (Or.inr ⟨h2, ?_, hCpos, ?_, ?_⟩)))
              · exact_mod_cast hcard
              · have hlt := not_le.1 h7
                have hle : (((certOf j B.npref x).filter (fun p =>
                    ¬ famGood O f kmin kmax ((clusterAt O.mq populations x B).erase 1) p)).card : ℝ)
                    = (((certOf j B.npref x).filter (fun p =>
                      ¬ (flipCount O ((clusterAt O.mq populations x B).erase 1) p
                        ≤ ((((clusterAt O.mq populations x B).erase 1).card : ℝ))
                          * f))).card : ℝ) := by
                  refine congrArg (fun t : Finset S => (t.card : ℝ))
                    (Finset.filter_congr (fun p _ => ?_))
                  unfold famGood
                  simp only [h5.1, h5.2, and_true, true_and]
                rw [hle, hcard]
                have hm0 : (0 : ℝ) ≤ (B.npref : ℝ) := Nat.cast_nonneg _
                nlinarith
              · intro hgood
                refine hx ⟨?_, hgood⟩
                have hone : (1 : S) ∈ clusterAt O.mq populations x B :=
                  one_mem_clusterAround O B.cn B.cd _ _ (oracleNoise x) B.k
                have hcard := Finset.card_erase_add_one hone
                omega
          · exact Or.inr (Or.inr (Or.inl h6))
        · exact Or.inr (Or.inl h5)
      · exact Or.inl (Or.inr h2)
    · exact Or.inl (Or.inl h1)
  calc (runMeasure μ D Dsf).real {x : Run Ω S J | x ∉ retAt O populations (2 * l) α B j}
      ≤ (runMeasure μ D Dsf).real ((E1 ∪ E2) ∪ (E5 ∪ (E6 ∪ (E7 ∪ E8)))) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ((runMeasure μ D Dsf).real E1 + (runMeasure μ D Dsf).real E2)
        + ((runMeasure μ D Dsf).real E5 + ((runMeasure μ D Dsf).real E6
          + ((runMeasure μ D Dsf).real E7 + (runMeasure μ D Dsf).real E8))) := by
        have h12 := measureReal_union_le (μ := runMeasure μ D Dsf) E1 E2
        have h78 := measureReal_union_le (μ := runMeasure μ D Dsf) E7 E8
        have h678 := measureReal_union_le (μ := runMeasure μ D Dsf) E6 (E7 ∪ E8)
        have h5678 := measureReal_union_le (μ := runMeasure μ D Dsf) E5 (E6 ∪ (E7 ∪ E8))
        have hall := measureReal_union_le (μ := runMeasure μ D Dsf) (E1 ∪ E2)
          (E5 ∪ (E6 ∪ (E7 ∪ E8)))
        linarith
    _ ≤ ((B.npref : ℝ) ^ 2 * ρ + (populations.card : ℝ) * (B.npref : ℝ) ^ 2 * ρ)
        + (Estall + (Edirty + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
          + (Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
            + (Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
              + 2 * Real.exp (-2 * (nlo : ℝ) * τ ^ 2)))))) := by
        gcongr
        · exact cert_not_injective_le D Dsf j B.npref ρ (hρ j hj) hρ0
        · exact prefix_cert_disjoint_le D Dsf populations j B.npref ρ hρ (hρ j hj) hρ0
        · exact measureReal_heavyHits_le D Dsf O populations j B f (Δp / f) th hth
        · exact measureReal_retMiss_le hflat O populations D Dsf hsupp j hj B α τ l lcut f E
            kmin kmax nlo hE hElcut hlcl hτ hl1 hnloB hsig hdec hcut hga hα
    _ = ((populations.card : ℝ) + 1) * (B.npref : ℝ) ^ 2 * ρ
        + (Estall + (Edirty + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
          + (Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
            + (Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
              + 2 * Real.exp (-2 * (nlo : ℝ) * τ ^ 2)))))) := by ring

/-- Part 1, reduced to one state.  The ladder is finite, so the union over the states the
loop may stop at is a finite sum: a rung carries `δ·npref/(8·N)`, and the counts sum to at
most `2N`.  No summable weight over all budgets is needed, and so no encoding of a budget as
a number.

`G` is what the rungs share.  The pool's findability does not mention the prefixes, so it is
one event for the whole ladder; charging it once rather than per rung is what keeps the pool
count free of the ladder's length, which the prefix count determines.

What remains of Part 1 is `hper`: at one rung, off `G`, a family that passes both gates is
valid on every population except with probability that rung's share. -/
theorem validity_of_ladder (O : Oracle μ S) (populations : Finset J)
    (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (indecisionLimit εcov α δ : ℝ) (hδ : 0 ≤ δ) (s : Finset State) (N : ℕ) (hN : 0 < N)
    (hsum : ∑ B ∈ s, (B.npref : ℝ) ≤ 2 * N)
    (G : Set (Run Ω S J)) (hG : (runMeasure μ D Dsf).real G ≤ δ / 4)
    (hper : ∀ B ∈ s, (runMeasure μ D Dsf).real
      ((ret O.mq populations indecisionLimit α B ∩ FailAt O populations D εcov B) \ G)
        ≤ δ * (B.npref : ℝ) / (8 * N)) :
    (runMeasure μ D Dsf).real (⋃ B : {B : State // B ∈ s},
        ret O.mq populations indecisionLimit α B.val
          ∩ FailAt O populations D εcov B.val) ≤ δ / 2 := by
  classical
  have hNR : (0 : ℝ) < (N : ℕ) := by exact_mod_cast hN
  have hsplit : (⋃ B : {B : State // B ∈ s},
        ret O.mq populations indecisionLimit α B.val
          ∩ FailAt O populations D εcov B.val)
      ⊆ (⋃ B : {B : State // B ∈ s},
          (ret O.mq populations indecisionLimit α B.val
            ∩ FailAt O populations D εcov B.val) \ G) ∪ G := by
    intro x hx
    obtain ⟨B, hB⟩ := Set.mem_iUnion.1 hx
    by_cases hg : x ∈ G
    · exact Or.inr hg
    · exact Or.inl (Set.mem_iUnion.2 ⟨B, hB, hg⟩)
  refine le_trans (measureReal_mono hsplit (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) ?_
  have hmain : (runMeasure μ D Dsf).real (⋃ B : {B : State // B ∈ s},
      (ret O.mq populations indecisionLimit α B.val
        ∩ FailAt O populations D εcov B.val) \ G) ≤ δ / 4 := by
    rw [Set.iUnion_subtype]
    calc (runMeasure μ D Dsf).real (⋃ B, ⋃ (_ : B ∈ s),
          (ret O.mq populations indecisionLimit α B ∩ FailAt O populations D εcov B) \ G)
        ≤ ∑ B ∈ s, (runMeasure μ D Dsf).real
            ((ret O.mq populations indecisionLimit α B
              ∩ FailAt O populations D εcov B) \ G) :=
          measureReal_biUnion_finset_le _ _
      _ ≤ ∑ B ∈ s, δ * (B.npref : ℝ) / (8 * (N : ℕ)) := Finset.sum_le_sum hper
      _ = (δ / (8 * (N : ℕ))) * ∑ B ∈ s, (B.npref : ℝ) := by
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl (fun B _ => by ring)
      _ ≤ (δ / (8 * (N : ℕ))) * (2 * (N : ℕ)) :=
          mul_le_mul_of_nonneg_left hsum (by positivity)
      _ = δ / 4 := by field_simp; ring
  linarith [hmain, hG]

/-! ### How `per_state_le` gets its bound

Validity is read off how the family was built, not off the gate.  Where the noise rate
depends on the class, a correct cut's agreement with the seed's column sits anywhere from
`1 − η` to `1` according to the population's mix of classes, so no null separates it from a
cut drifted on an `εcov` fraction.  What does bound the drift is the screen: it keeps no
candidate that flips more than the flip budget of the table, Markov over the family turns
that into a bound on the mass where many members flip, and the vote is right elsewhere.

`measureReal_validFail_le` carries that through the certification sample, which the family
never saw: a cut wrong on `εcov` of the population is wrong on `3εcov/4` of the sample
(`hitShort`), few of those can be prefixes the family flips (`heavyHits`), and few of the
rest can be misread by a family that barely flips there (`validMiss`, Hoeffding). -/

/-- The vote's two clean means, pushed by a flipping `flipFrac` of the family and read
`voteSlack` in, still straddle the centre `κ/2`.

Both sides are tight at `O.η = η₀`: the two means sum to `1`, so the budget `flipFrac` and
`voteSlack` split spends one side exactly when it spends the other. -/
lemma vote_shifts (O : Oracle μ S) {η₀ : ℝ} (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) {κ : ℝ}
    (hκ0 : 0 ≤ κ) :
    κ * ((O.η + (1 - O.η) * flipFrac η₀) + voteSlack η₀) ≤ κ / 2
      ∧ κ / 2 ≤ κ * (((1 - O.η) * (1 - flipFrac η₀)) - voteSlack η₀) := by
  have hsplit := flipFrac_voteSlack η₀ hη₀
  have hsv : sig η₀ = 1 / 2 - η₀ := rfl
  have hF1 := flipFrac_lt_one η₀ hη₀
  have hu : (O.η + (1 - O.η) * flipFrac η₀) + voteSlack η₀ ≤ 1 / 2 := by
    nlinarith [mul_nonneg (sub_nonneg.2 hηle) (by linarith : (0 : ℝ) ≤ 1 - flipFrac η₀)]
  have hl : 1 / 2 ≤ ((1 - O.η) * (1 - flipFrac η₀)) - voteSlack η₀ := by nlinarith [hu]
  constructor
  · nlinarith [mul_le_mul_of_nonneg_left hu hκ0]
  · nlinarith [mul_le_mul_of_nonneg_left hl hκ0]

/-- The same straddle read off the half-gap rather than off the worse rate: the two clean
means are `c ± hgap` for `c = (1 + ηOut − ηIn)/2` and `hgap = (1 − ηIn − ηOut)/2`, a flipping
member answers at the other mean and so moves the vote by `2·hgap`, and the centre is held at
the rational `cn/cd` rather than at `c`.

The flip fraction is the constant `3/8` whatever the asymmetry: at the true centre the margin
is `hgap` and one unit of flip fraction spends `2·hgap`. -/
lemma vote_shifts_gap {c hgap κ : ℝ} {cn cd : ℕ}
    (hκ0 : 0 ≤ κ) (hκs : 8 ≤ κ * hgap)
    (hcbelow : (cn : ℝ) / cd ≤ c) (hcabove : c ≤ (cn : ℝ) / cd + hgap / 8) :
    κ * ((c - hgap) + 2 * hgap * (3 / 8) + hgap / 8) ≤ κ * ((cn : ℝ) / cd)
      ∧ κ * ((cn : ℝ) / cd) + 1
        ≤ κ * ((c + hgap) - 2 * hgap * (3 / 8) - hgap / 8) := by
  constructor
  · nlinarith [mul_le_mul_of_nonneg_left hcabove hκ0]
  · nlinarith [mul_le_mul_of_nonneg_left hcbelow hκ0]

/-- `rung_facts`' threshold clauses at a centre held at `cn/cd`, with `lo = ⌈κ·cn/cd⌉ − 1`
and `hi = ⌈κ·cn/cd⌉ + 1`.  The slack `κ·hgap/8` covers the rounding to a count as well as the
gap between `c` and `cn/cd`. -/
lemma rung_thresholds_gap {c hgap : ℝ} {κ cn cd : ℕ}
    (hκpos : 0 < κ) (hκs : 8 ≤ (κ : ℝ) * hgap)
    (hcn : 0 < cn) (hcd : cn < cd)
    (hcbelow : (cn : ℝ) / cd ≤ c) (hcabove : c ≤ (cn : ℝ) / cd + hgap / 8) :
    ⌈(κ : ℝ) * cn / cd⌉₊ - 1 < ⌈(κ : ℝ) * cn / cd⌉₊ + 1
    ∧ (((⌈(κ : ℝ) * cn / cd⌉₊ + 1) - 1 : ℕ) : ℝ)
        ≤ (κ : ℝ) * ((c + hgap) - 2 * hgap * (3 / 8) - hgap / 8)
    ∧ (κ : ℝ) * ((c - hgap) + 2 * hgap * (3 / 8) + hgap / 8)
        ≤ ((⌈(κ : ℝ) * cn / cd⌉₊ - 1 : ℕ) : ℝ) + 1
    ∧ (κ : ℝ) * ((c - hgap) + 2 * hgap * (3 / 8) + hgap / 8)
        ≤ (((⌈(κ : ℝ) * cn / cd⌉₊ + 1) - 1 : ℕ) : ℝ)
    ∧ ((⌈(κ : ℝ) * cn / cd⌉₊ - 1 : ℕ) : ℝ)
        < (κ : ℝ) * ((c + hgap) - 2 * hgap * (3 / 8) - hgap / 8) := by
  have hκR : (0 : ℝ) < (κ : ℝ) := by exact_mod_cast hκpos
  have hcdR : (0 : ℝ) < (cd : ℝ) := by
    exact_mod_cast lt_of_le_of_lt (Nat.zero_le cn) hcd
  have hcnR : (0 : ℝ) < (cn : ℝ) := by exact_mod_cast hcn
  set x : ℝ := (κ : ℝ) * cn / cd with hxdef
  have hxpos : 0 < x := by rw [hxdef]; positivity
  have hxeq : x = (κ : ℝ) * ((cn : ℝ) / cd) := by rw [hxdef]; ring
  obtain ⟨hlowShift, hhiShift⟩ :=
    vote_shifts_gap (c := c) (κ := (κ : ℝ)) (cn := cn) (cd := cd) hκR.le hκs hcbelow hcabove
  rw [← hxeq] at hlowShift hhiShift
  have hceil1 : 1 ≤ ⌈x⌉₊ := Nat.one_le_ceil_iff.2 hxpos
  have hceilx : (⌈x⌉₊ : ℝ) ≤ x + 1 := le_of_lt (Nat.ceil_lt_add_one hxpos.le)
  have hxceil : x ≤ (⌈x⌉₊ : ℝ) := Nat.le_ceil x
  refine ⟨by omega, ?_, ?_, ?_, ?_⟩
  · rw [show (⌈x⌉₊ + 1 - 1 : ℕ) = ⌈x⌉₊ from by omega]
    linarith
  · rw [Nat.cast_sub hceil1]
    push_cast
    linarith
  · rw [show (⌈x⌉₊ + 1 - 1 : ℕ) = ⌈x⌉₊ from by omega]
    linarith
  · rw [Nat.cast_sub hceil1]
    push_cast
    linarith

/-- What every rung of the ladder satisfies whatever its prefix count: only the prefix count
and the skip guard change down the ladder, and none of these read either. -/
lemma rung_facts (O : Oracle μ S) (populations : Finset J) {εcov δ pAP : ℝ} (mi : ℕ)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hεcov : 0 < εcov)
    (hind : 0 < indecisionLimit) (hcard : (0 : ℝ) < (populations.card : ℝ)) :
    0 < (solvedStateAt η₀ populations indecisionLimit εcov δ pAP mi).k
    ∧ (solvedStateAt η₀ populations indecisionLimit εcov δ pAP mi).cn
      < (solvedStateAt η₀ populations indecisionLimit εcov δ pAP mi).cd
    ∧ 0 < (solvedStateAt η₀ populations indecisionLimit εcov δ pAP mi).scd
    ∧ (solvedStateAt η₀ populations indecisionLimit εcov δ pAP mi).lo
      < (solvedStateAt η₀ populations indecisionLimit εcov δ pAP mi).hi
    ∧ ((solvedStateAt η₀ populations indecisionLimit εcov δ pAP mi).sc : ℝ)
      ≤ ((solvedStateAt η₀ populations indecisionLimit εcov δ pAP mi).scd : ℝ)
        * (flipBudget η₀ populations indecisionLimit εcov δ * (1 - 2 * O.η) ^ 2
          - 2 * (screenMargin η₀ populations indecisionLimit εcov δ / 2))
    ∧ (((solvedStateAt η₀ populations indecisionLimit εcov δ pAP mi).k - 1 : ℕ) : ℝ)
        * ((O.η + (1 - O.η) * flipFrac η₀) + voteSlack η₀)
      ≤ (((solvedStateAt η₀ populations indecisionLimit εcov δ pAP mi).hi - 1 : ℕ) : ℝ)
    ∧ ((solvedStateAt η₀ populations indecisionLimit εcov δ pAP mi).lo : ℝ)
      < (((solvedStateAt η₀ populations indecisionLimit εcov δ pAP mi).k - 1 : ℕ) : ℝ)
        * (((1 - O.η) * (1 - flipFrac η₀)) - voteSlack η₀)
    ∧ Real.exp (-2 * (((solvedStateAt η₀ populations indecisionLimit εcov δ pAP mi).k - 1 : ℕ) : ℝ)
        * voteSlack η₀ ^ 2) ≤ cutBudget η₀ indecisionLimit εcov / 2 := by
  classical
  have hs : 0 < sig η₀ := sig_pos η₀ hη₀
  have hsval : sig η₀ = 1 / 2 - η₀ := rfl
  have hΔ : 0 < flipBudget η₀ populations indecisionLimit εcov δ :=
    flipBudget_pos η₀ populations hη₀ hεcov hind hcard
  have hγ : 0 < screenMargin η₀ populations indecisionLimit εcov δ :=
    screenMargin_pos η₀ populations hη₀ hεcov hind hcard
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ with hκdef
  have hκpos : 0 < κ := famCount_pos η₀ populations εcov δ
  have hκR : (0 : ℝ) < (κ : ℝ) := by exact_mod_cast hκpos
  set x : ℝ := (κ : ℝ) / 2 with hxdef
  have hxpos : 0 < x := by
    rw [hxdef]
    exact div_pos hκR (by norm_num)
  have hceil1 : 1 ≤ ⌈x⌉₊ := Nat.one_le_ceil_iff.2 hxpos
  have hxexact : (⌈x⌉₊ : ℝ) = x := by rw [hxdef, hκdef]; exact famCount_half η₀ populations εcov δ
  have hκ0 : (0 : ℝ) ≤ (κ : ℝ) := Nat.cast_nonneg _
  obtain ⟨hlowShift, hhiShift⟩ := vote_shifts O hηle hη₀ hκ0
  set B : State := solvedStateAt η₀ populations indecisionLimit εcov δ pAP mi with hBdef
  have hBk : B.k = κ + 1 := rfl
  have hBcn : B.cn = 1 := rfl
  have hBcd : B.cd = 2 := rfl
  have hBlo : B.lo = ⌈x⌉₊ - 1 := rfl
  have hBhi : B.hi = ⌈x⌉₊ + 1 := rfl
  have hBscd : B.scd = ⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 := rfl
  have hBsc : B.sc = ⌈((⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) :
    ℝ)
      * screenMargin η₀ populations indecisionLimit εcov δ⌉₊ := rfl
  clear_value B κ x
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hBk]; omega
  · rw [hBcn, hBcd]; norm_num
  · rw [hBscd]; omega
  · rw [hBlo, hBhi]; omega
  · rw [hBsc, hBscd]
    have hone : (15 : ℝ) ≤ 2 * ((⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ +
      1 : ℕ) : ℝ)
        * screenMargin η₀ populations indecisionLimit εcov δ := by
      have hcl : (15 : ℝ) / (2 * screenMargin η₀ populations indecisionLimit εcov δ)
          ≤ ((⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) : ℝ) := by
        refine le_trans (Nat.le_ceil _) ?_
        push_cast
        linarith
      rw [div_le_iff₀ (by positivity)] at hcl
      linarith
    have hceilub : (⌈((⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) :
      ℝ)
        * screenMargin η₀ populations indecisionLimit εcov δ⌉₊ : ℝ)
        ≤ ((⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) : ℝ)
          * screenMargin η₀ populations indecisionLimit εcov δ + 1 :=
      le_of_lt (Nat.ceil_lt_add_one (by positivity))
    have hmonoF : flipBudget η₀ populations indecisionLimit εcov δ * (1 - 2 * η₀) ^ 2
        ≤ flipBudget η₀ populations indecisionLimit εcov δ * (1 - 2 * O.η) ^ 2 := by
      refine mul_le_mul_of_nonneg_left ?_ hΔ.le
      have h1 : (0 : ℝ) ≤ 1 - 2 * η₀ := by linarith
      nlinarith [h1, hηle]
    have hFeq : flipBudget η₀ populations indecisionLimit εcov δ * (1 - 2 * η₀) ^ 2
        = 32 / 15 * screenMargin η₀ populations indecisionLimit εcov δ := by
      rw [screenMargin, hsval]; ring
    rw [hFeq] at hmonoF
    have hS0 : (0 : ℝ) ≤ ((⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 :
      ℕ) : ℝ) :=
      Nat.cast_nonneg _
    have hFS := mul_le_mul_of_nonneg_left hmonoF hS0
    nlinarith [hceilub, hone, hFS, hγ.le, hS0]
  · rw [hBhi, hBk, show (κ + 1 - 1 : ℕ) = κ from by omega,
      show (⌈x⌉₊ + 1 - 1 : ℕ) = ⌈x⌉₊ from by omega]
    linarith [hxexact, hlowShift]
  · rw [hBlo, hBk, show (κ + 1 - 1 : ℕ) = κ from by omega, Nat.cast_sub hceil1]
    push_cast
    linarith [hxexact, hhiShift]
  · rw [hBk, show (κ + 1 - 1 : ℕ) = κ from by omega, hκdef]
    exact famCount_tail η₀ populations hη₀ hεcov hind

open scoped Classical in
set_option maxHeartbeats 1000000 in
/-- Part 1 at one state.  Off the pool's findability, a returned family is valid on every
population except where the certification draws, the screen or the vote fail, charged per
population, and where the pool's own draws collide, charged once. -/
theorem per_state_le {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (hsig : O.η ≤ 1 / 2)
    (indecisionLimit εcov α ρ ρsf pAP : ℝ) (hε0 : 0 ≤ εcov)
    (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (B : State) (hkpos : 0 < B.k) (hhi : 1 ≤ B.hi) (hcd : B.cn < B.cd) (hmpos : 0 < B.npref)
    (hscd : 0 < B.scd) (hcount : (2 : ℝ) ≤ (B.nsuff : ℝ) * (pAP - pAP / 2))
    (Δ γ g th f lcut E : ℝ) (hΔ : 0 < Δ) (hγ : 0 ≤ γ) (hg : 0 ≤ g) (hth : 0 ≤ th)
    (hf0 : 0 < f) (hE : 0 ≤ E) (hE2 : E ≤ lcut / 2)
    (hsc : (B.sc : ℝ) ≤ (B.scd : ℝ) * (Δ * (1 - 2 * O.η) ^ 2 - 2 * γ))
    (hbudget : ((populations.card : ℝ) * Δ + g) / f + th + lcut ≤ 3 * εcov / 4)
    (hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
      B.k - 1 ≤ F.card → F.card ≤ B.k - 1 →
      μ.real {ω | ¬ cutCorrect O B.lo (B.hi - 1) F p ω} ≤ E) :
    (runMeasure μ D Dsf).real ((ret O.mq populations indecisionLimit α B
        ∩ FailAt O populations D εcov B) \ apShort O B.nsuff pAP (pAP / 2))
      ≤ (populations.card : ℝ) * (((populations.card : ℝ) + 1) * (B.npref : ℝ) ^ 2 * ρ
        + (Real.exp (-2 * (B.npref : ℝ) * (εcov / 4) ^ 2)
          + (((B.npref : ℝ) ^ 2 * ρ
              + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2)
              + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * g ^ 2))
            + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
              + Real.exp (-2 * (B.npref : ℝ) * (lcut / 2) ^ 2)))))
        + (B.nsuff : ℝ) ^ 2 * ρsf := by
  classical
  set Δp : ℝ := (populations.card : ℝ) * Δ + g with hΔp
  set CR : Set (Run Ω S J) := {x | ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
      ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0} with hCR
  set V : J → Set (Run Ω S J) := fun j => {x : Run Ω S J |
      B.k ≤ (clusterAt O.mq populations x B).card
      ∧ ¬ (1 - εcov ≤ (D j).real
        {p | cutCorrect O B.lo B.hi (clusterAt O.mq populations x B) p (oracleNoise x)})}
    with hV
  set Dy : J → Set (Run Ω S J) := fun j =>
    {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O.mq populations x B, flipMass O (D j) v ≤ Δp} with hDy
  have hsub : ((ret O.mq populations indecisionLimit α B ∩ FailAt O populations D εcov B)
        \ apShort O B.nsuff pAP (pAP / 2))
      ⊆ ({x : Run Ω S J | ¬ ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
            ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
          \ apShort O B.nsuff pAP (pAP / 2))
        ∪ ⋃ j ∈ populations, ((V j \ Dy j) ∪ (CR ∩ Dy j)) := by
    rintro x ⟨⟨hret, hfail⟩, hG⟩
    by_cases hc : x ∈ CR
    · refine Or.inr ?_
      simp only [FailAt, Set.mem_setOf_eq, not_forall] at hfail
      obtain ⟨j, hj, hfj⟩ := hfail
      refine Set.mem_biUnion hj ?_
      by_cases hd : x ∈ Dy j
      · exact Or.inr ⟨hc, hd⟩
      · exact Or.inl ⟨⟨hret.1, hfj⟩, hd⟩
    · exact Or.inl ⟨hc, hG⟩
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) ?_
  have h1 := measureReal_noCleanRef_off_le D Dsf O populations B pAP (pAP / 2) ρsf hρsf hρsf0
    hcount
  have h2 : (runMeasure μ D Dsf).real (⋃ j ∈ populations, ((V j \ Dy j) ∪ (CR ∩ Dy j)))
      ≤ ∑ _j ∈ populations, (((populations.card : ℝ) + 1) * (B.npref : ℝ) ^ 2 * ρ
        + (Real.exp (-2 * (B.npref : ℝ) * (εcov / 4) ^ 2)
          + (((B.npref : ℝ) ^ 2 * ρ
              + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2)
              + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * g ^ 2))
            + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
              + Real.exp (-2 * (B.npref : ℝ) * (lcut / 2) ^ 2))))) := by
    refine le_trans (measureReal_biUnion_finset_le _ _) (Finset.sum_le_sum (fun j hj => ?_))
    refine le_trans (measureReal_union_le _ _) ?_
    have hv := measureReal_validFail_le hflat O populations D Dsf hsupp j hj B hkpos hhi εcov lcut f E
      Δp ρ th hε0 hE (by linarith) (by rw [hΔp]; positivity) hth hf0 hbudget hρ hρ0 hcut
    have htail : Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
        ≤ Real.exp (-2 * (B.npref : ℝ) * (lcut / 2) ^ 2) := by
      refine Real.exp_le_exp.2 ?_
      have hm : (0 : ℝ) ≤ (B.npref : ℝ) := Nat.cast_nonneg _
      have h1 : (lcut / 2) ^ 2 ≤ (lcut - E) ^ 2 := by nlinarith
      nlinarith [mul_le_mul_of_nonneg_left h1 hm]
    have hd := measureReal_dirtyMember_le hflat O populations D Dsf hsupp j hj B hcd hsig hmpos
      Δ γ g ρ hΔ hγ hg hρ0 (hρ j hj) hscd hsc
    have hv' : (runMeasure μ D Dsf).real (V j \ Dy j)
        ≤ ((populations.card : ℝ) + 1) * (B.npref : ℝ) ^ 2 * ρ
          + (Real.exp (-2 * (B.npref : ℝ) * (εcov / 4) ^ 2)
            + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
              + Real.exp (-2 * (B.npref : ℝ) * (lcut / 2) ^ 2))) :=
      le_trans hv (by linarith [htail])
    have hd' : (runMeasure μ D Dsf).real (CR ∩ Dy j)
        ≤ (B.npref : ℝ) ^ 2 * ρ + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2)
          + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * g ^ 2) := hd
    linarith [hv', hd']
  rw [Finset.sum_const, nsmul_eq_mul] at h2
  linarith [h1, h2]

open scoped Classical in
/-- Part 1 — whatever is returned is valid, whenever it is returned.

Except with probability `δ/2`, no rung the loop may stop at returns an invalid family. -/
theorem validity_of_returned {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (indecisionLimit εcov α : ℝ) (hindLim : 0 < indecisionLimit)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (ρ ρsf : ℝ) (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ)
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ)
    (pAP : ℝ) (hpAP0 : 0 ≤ pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hfind : Real.exp (-2 * (poolCount η₀ populations indecisionLimit εcov δ pAP : ℝ) * (pAP / 2) ^
      2)
      ≤ δ / 4) :
    (runMeasure μ D Dsf).real
        (⋃ B : {B : State // B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP ρ ρsf},
          ret O.mq populations indecisionLimit α B.val
            ∩ FailAt O populations D εcov B.val) ≤ δ / 2 := by
  classical
  obtain ⟨j₀, hj₀⟩ := hpop
  have hρ0 : 0 ≤ ρ := le_trans (tsum_nonneg (fun a => sq_nonneg _)) (hρ j₀ hj₀)
  have hcard : (0 : ℝ) < (populations.card : ℝ) := by
    exact_mod_cast Finset.card_pos.2 ⟨j₀, hj₀⟩
  refine validity_of_ladder O populations D Dsf indecisionLimit εcov α δ hδ.le _
    (prefCount η₀ populations indecisionLimit εcov δ α pAP) (prefCount_pos _ _ _ _ _ _)
    (stoppable_npref_sum_le η₀ populations εcov δ α pAP ρ ρsf)
    (apShort O (poolCount η₀ populations indecisionLimit εcov δ pAP) pAP (pAP / 2))
    (le_trans (measureReal_apShort_le D Dsf O _ pAP (pAP / 2) hpAP0 (by linarith) hpAPBound)
      hfind) (fun B hB => ?_)
  have hcap := capped_of_mem_stoppable hB
  have hsched := (Finset.mem_filter.1 hB).1
  have hM : B.nsuff = poolCount η₀ populations indecisionLimit εcov δ pAP := nsuff_of_mem_schedule
    hsched
  rw [schedule] at hsched
  obtain ⟨i, -, hBeq⟩ := Finset.mem_image.1 hsched
  obtain ⟨hk, hcd, hscd, hlohi, hsc, hhi, hlo, hEc⟩ := rung_facts O populations
    (δ := δ) (pAP := pAP) (prefCount η₀ populations indecisionLimit εcov δ α pAP / 2 ^ i)
    hηle hη₀ hεcov hindLim hcard
  rw [hBeq] at hk hcd hscd hlohi hsc hhi hlo hEc
  have hΔ : 0 < flipBudget η₀ populations indecisionLimit εcov δ :=
    flipBudget_pos η₀ populations hη₀ hεcov hindLim hcard
  have hγ : 0 < screenMargin η₀ populations indecisionLimit εcov δ :=
    screenMargin_pos η₀ populations hη₀ hεcov hindLim hcard
  have hs : 0 < sig η₀ := sig_pos η₀ hη₀
  have hbudget : ((populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ
      + (populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ) / flipFrac η₀
      + cutBudget η₀ indecisionLimit εcov / 4 + cutBudget η₀ indecisionLimit εcov ≤ 3 * εcov / 4 :=
        by
    have hne : (populations.card : ℝ) ≠ 0 := ne_of_gt hcard
    have hFne : flipFrac η₀ ≠ 0 := ne_of_gt (flipFrac_pos η₀ hη₀)
    have hid : ((populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ
        + (populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ) / flipFrac η₀
        = cutBudget η₀ indecisionLimit εcov * (2 / 3) := by
      rw [flipBudget]
      field_simp
      ring
    rw [hid]
    linarith [(cutBudget_le η₀ indecisionLimit εcov).1]
  have hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * flipFrac η₀ →
      B.k - 1 ≤ F.card → F.card ≤ B.k - 1 →
      μ.real {ω | ¬ cutCorrect O B.lo (B.hi - 1) F p ω}
        ≤ Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * voteSlack η₀ ^ 2) := by
    intro F p hf hmin hmax
    have hcardF : F.card = B.k - 1 := le_antisymm hmax hmin
    refine le_trans (cutCorrect_whp O F p _ _ (flipFrac η₀) (voteSlack η₀) hsig.le hf
      (voteSlack_pos η₀ hη₀).le
      ?_ ?_)
      (le_of_eq ?_)
    · rw [hcardF]; exact hhi
    · rw [hcardF]; exact hlo
    · rw [hcardF]
  rw [← hM]
  refine le_trans (per_state_le hflat O populations D Dsf hsupp hsig.le indecisionLimit εcov α
    ρ ρsf pAP hεcov.le hρ hρ0 hρsf hρsf0 B hk (by omega) hcd hcap.mpos hscd
    (by have := hcap.found; linarith)
    (flipBudget η₀ populations indecisionLimit εcov δ) (screenMargin η₀ populations indecisionLimit
      εcov δ / 2)
    ((populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ) (cutBudget η₀
      indecisionLimit εcov / 4)
    (flipFrac η₀) (cutBudget η₀ indecisionLimit εcov) (Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) *
      voteSlack η₀ ^ 2))
    hΔ (by linarith) (mul_nonneg hcard.le hΔ.le)
    (by linarith [cutBudget_pos η₀ hη₀ hεcov hindLim])
    (flipFrac_pos η₀ hη₀) (Real.exp_nonneg _) hEc hsc hbudget hcut)
    (le_trans (le_of_eq ?_) hcap.share)
  rfl

/-- What one round at one population can cost: the draws, the family's size and cleanliness,
the sample's two class counts and its heavy fraction, and the round's own two tests. -/
noncomputable def roundFail (populations : Finset J)
    (l lcut τ th E γscr γdirty gdirty tap ρ ρsf : ℝ) (n₀ : ℕ) (B : State) : ℝ :=
  ((populations.card : ℝ) + 1) * (B.npref : ℝ) ^ 2 * ρ
    + ((((B.nsuff : ℝ) ^ 2 * ρsf + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2)
          + ((B.npref : ℝ) ^ 2 * ρ + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γscr ^ 2))))
        + (((B.npref : ℝ) ^ 2 * ρ + (((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γdirty ^ 2)
            + ((B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * gdirty ^ 2)
              + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2) + (B.nsuff : ℝ) ^ 2 * ρsf))))
          + (Real.exp (-2 * (B.npref : ℝ) * th ^ 2)
            + (Real.exp (-2 * (B.npref : ℝ) * (l - E) ^ 2)
              + (Real.exp (-2 * (B.npref : ℝ) * (lcut - E) ^ 2)
                + 2 * Real.exp (-2 * (n₀ : ℝ) * τ ^ 2)))))))

/-- A state whose round can pass.  Every clause is an inequality among the state's
budgets, the oracle's rates, the suffix distribution's findability and the error budget —
no probability enters, and nothing here is a free parameter of the algorithm.  Reaching
such a state is what the computed ladder is for.

A population that never rejects is fine: the gate skips a sample below `gmin`, and
`admitted` is an implication, so a skipped test is passed over rather than failed.  Nothing
here asks a population to carry both labels. -/
def PassableAt (O : Oracle μ S) (η₀ : ℝ) (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    (indecisionLimit εcov α δ ρ ρsf pAP : ℝ) (B : State) : Prop :=
  ∃ (τ th tap γdec γscr γdirty gdirty Δ lcut f : ℝ),
    0 < B.npref ∧ 2 ≤ B.k ∧ B.cn < B.cd ∧ 0 < indecisionLimit ∧ indecisionLimit ≤ 1 / 2
    ∧ εcov ≤ 1
    ∧ 0 ≤ τ ∧ 0 ≤ th ∧ 0 ≤ tap ∧ 0 ≤ γdec ∧ 0 ≤ γscr ∧ 0 ≤ γdirty ∧ 0 ≤ gdirty
    ∧ 0 < Δ ∧ 0 < f ∧ 0 ≤ pAP ∧ 0 < lcut ∧ lcut ≤ indecisionLimit / 2
    -- the pool's findability
    ∧ pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p}
    ∧ collisionMass Dsf ≤ ρsf
    -- the certification sample holds few heavy prefixes: Markov over the family, so the
    -- flip mass is charged at the fraction the vote absorbs and not at the family's size
    ∧ (((populations.card : ℝ) * Δ + gdirty) / f + th ≤ lcut)
    -- the screen's rate sits above the clean one and below the dirty one
    ∧ 0 < B.scd
    ∧ ((B.scd : ℝ) * (2 * γscr) ≤ (B.sc : ℝ))
    ∧ ((B.sc : ℝ)
        ≤ (B.scd : ℝ) * (Δ * (1 - 2 * O.η) ^ 2 - 2 * γdirty))
    -- the pool holds a family
    ∧ ((B.k : ℝ) ≤ (B.nsuff : ℝ) * (pAP - tap))
    -- the thresholds decide, and decide right, on a family of the round's size an `f`
    -- fraction of which flips, so both means move toward the centre by up to `(1 − η)·f`
    ∧ (((B.hi - 1 : ℕ) : ℝ)
        ≤ ((B.k - 1 : ℕ) : ℝ) * (((1 - O.η) * (1 - f)) - γdec))
    ∧ (((B.k - 1 : ℕ) : ℝ) * ((O.η + (1 - O.η) * f) + γdec) ≤ (B.lo : ℝ) + 1)
    ∧ (((B.k - 1 : ℕ) : ℝ) * ((O.η + (1 - O.η) * f) + γdec) ≤ ((B.hi - 1 : ℕ) : ℝ))
    ∧ ((B.lo : ℝ) < ((B.k - 1 : ℕ) : ℝ) * (((1 - O.η) * (1 - f)) - γdec))
    -- the gate's margin clears its null on every decided count the FNR test allows
    ∧ (∀ n c : ℕ, ⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊ ≤ n → n ≤ c → c ≤ B.npref →
        (n : ℝ) * (1 / 2 + τ + τ)
          ≤ (n : ℝ) * (1 - O.η) - (1 - O.η) * (2 * lcut * (c : ℝ)))
    ∧ (Real.exp (-2 * (B.gmin : ℝ) * τ ^ 2) ≤ α)
    -- the vote misfires at under half the cut budget, so the sample's count of misfires
    -- concentrates below the budget
    ∧ (Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * γdec ^ 2) ≤ lcut / 2)
    -- and the whole round, over every population, fits in the budget
    ∧ ((populations.card : ℝ)
        * roundFail populations (indecisionLimit / 2) lcut τ th
            (Real.exp (-2 * ((B.k - 1 : ℕ) : ℝ) * γdec ^ 2)) γscr γdirty gdirty tap ρ ρsf
            ⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊ B
      ≤ δ / 2)

lemma mul_self_add_le_cube {x : ℝ} (hx : 0 ≤ x) : x * (x + 3) ≤ (x + 3) ^ 3 := by
  have h : (x + 3) ^ 3 - x * (x + 3) = x ^ 3 + 8 * x ^ 2 + 24 * x + 27 := by ring
  linarith [h, pow_nonneg hx 3, sq_nonneg x, hx]

lemma le_cube_of_nonneg {x : ℝ} (hx : 0 ≤ x) : x ≤ (x + 3) ^ 3 := by
  have h : (x + 3) ^ 3 - x = x ^ 3 + 9 * x ^ 2 + 26 * x + 27 := by ring
  linarith [h, pow_nonneg hx 3, sq_nonneg x, hx]

lemma exp_tail_anti {m a b : ℝ} (hm : 0 ≤ m) (ha : 0 ≤ a) (hab : a ≤ b) :
    Real.exp (-2 * m * b ^ 2) ≤ Real.exp (-2 * m * a ^ 2) := by
  refine Real.exp_le_exp.2 ?_
  have h := mul_le_mul_of_nonneg_left (pow_le_pow_left₀ ha hab 2) hm
  nlinarith

/-- A count that clears `log (c/ε) / (2γ²)` kills the tail it was read off. -/
lemma tail_le_of_count {γ ε c : ℝ} {n : ℕ} (hγ : 0 < γ) (hε : 0 < ε) (hc : 0 < c)
    (hn : Real.log (c / ε) / (2 * γ ^ 2) ≤ (n : ℝ)) :
    c * Real.exp (-2 * (n : ℝ) * γ ^ 2) ≤ ε := by
  have hγ2 : (0 : ℝ) < 2 * γ ^ 2 := by positivity
  have h1 : Real.log (c / ε) ≤ 2 * (n : ℝ) * γ ^ 2 := by
    rw [div_le_iff₀ hγ2] at hn
    linarith
  have h2 : Real.exp (-2 * (n : ℝ) * γ ^ 2) ≤ ε / c := by
    rw [show (-2 * (n : ℝ) * γ ^ 2) = -(2 * (n : ℝ) * γ ^ 2) from by ring]
    refine exp_neg_le_of_log_le (by positivity) ?_
    rwa [one_div_div]
  calc c * Real.exp (-2 * (n : ℝ) * γ ^ 2) ≤ c * (ε / c) :=
        mul_le_mul_of_nonneg_left h2 hc.le
    _ = ε := by field_simp

/-! ### What the solved budget's fields buy

Each of these says that a field, read off its condition, meets it.  They are the analytic
content of the construction; everything else about `PassableAt` is arithmetic on the
definitions. -/

/-- The gate's floor clears its own tail at the error rate `α`. -/
lemma solved_alpha (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP : ℝ}
    (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hα : 0 < α)
    (hα1 : α < 1 / 2)
    (hpAP : 0 < pAP)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) :
    Real.exp (-2 * ((solvedState η₀ populations indecisionLimit εcov δ α pAP).gmin : ℝ)
      * (sig η₀ * εcov / 4) ^ 2) ≤ α := by
  have hs : 0 < sig η₀ := sig_pos η₀ hsig
  have hτ : (0 : ℝ) < sig η₀ * εcov / 4 := by positivity
  have hτ2 : (0 : ℝ) < (sig η₀ * εcov / 4) ^ 2 := pow_pos hτ 2
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP with hmdef
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  -- the prefix count clears the gate's tail
  have hmlog : 64 * Real.log (1 / α) / (εcov * (sig η₀ * εcov / 4) ^ 2) ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈64 * Real.log (1 / α) / (εcov * (sig η₀ * εcov / 4) ^ 2)⌉₊ ≤ m := by
      rw [hmdef, prefCount]; omega
    exact_mod_cast hle
  -- and the floor keeps a fraction of it
  have hsizeR : 64 / εcov ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈64 / εcov⌉₊ ≤ m := by rw [hmdef, prefCount]; omega
    exact_mod_cast hle
  have hsize : 2 ≤ εcov * (m : ℝ) / 32 := by
    rw [div_le_iff₀ hε] at hsizeR
    linarith
  have hgmin : εcov * (m : ℝ) / 64
      ≤ ((solvedState η₀ populations indecisionLimit εcov δ α pAP).gmin : ℝ) := by
    show εcov * (m : ℝ) / 64 ≤ (⌊εcov * (m : ℝ) / 32⌋₊ : ℝ)
    have hlt := Nat.lt_floor_add_one (εcov * (m : ℝ) / 32)
    linarith
  rw [show (-2 * ((solvedState η₀ populations indecisionLimit εcov δ α pAP).gmin : ℝ)
        * (sig η₀ * εcov / 4) ^ 2)
      = -(2 * ((solvedState η₀ populations indecisionLimit εcov δ α pAP).gmin : ℝ)
        * (sig η₀ * εcov / 4) ^ 2) from by ring]
  refine exp_neg_le_of_log_le hα ?_
  have hlog0 : 0 ≤ Real.log (1 / α) := Real.log_nonneg (by rw [le_div_iff₀ hα]; linarith)
  rw [div_le_iff₀ (by positivity)] at hmlog
  nlinarith [mul_le_mul_of_nonneg_right hgmin hτ2.le, hmR, hlog0]

/-- The pool is deep enough to hold accept-preserving suffixes.  This is the one event the
ladder shares, so it is charged once against `δ/4` rather than per rung. -/
lemma solved_findability (η₀ : ℝ) (populations : Finset J) {εcov δ pAP : ℝ}
    (hδ : 0 < δ) (hpAP : 0 < pAP) (hcard : (1 : ℝ) ≤ (populations.card : ℝ)) :
    Real.exp (-2 * (poolCount η₀ populations indecisionLimit εcov δ pAP : ℝ) * (pAP / 2) ^ 2) ≤ δ /
      32 := by
  have ctap : Real.log (1 / (δ / 32)) / (2 * (pAP / 2) ^ 2)
      ≤ ((poolCount η₀ populations indecisionLimit εcov δ pAP : ℕ) : ℝ) := by
    have hmono : Real.log (1 / (δ / 32))
        ≤ Real.log (128 * (populations.card : ℝ) / δ) := by
      refine Real.log_le_log (by positivity) ?_
      rw [one_div_div, div_le_div_iff₀ hδ hδ]
      nlinarith
    refine le_trans (div_le_div_of_nonneg_right hmono (by positivity)) ?_
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (pAP / 2) ^ 2)⌉₊
        ≤ poolCount η₀ populations indecisionLimit εcov δ pAP := by rw [poolCount]; omega
    exact_mod_cast hle
  have h := tail_le_of_count (γ := pAP / 2) (c := 1) (by positivity)
    (by positivity : (0 : ℝ) < δ / 32) (by norm_num) ctap
  linarith

/-- The top rung's tails, each at `δ/(128·|populations|)`, and its collision terms. -/
lemma solved_tails (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ ρsf : ℝ}
    (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hpAP : 0 < pAP)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ) (hρsf0 : 0 ≤ ρsf)
    (hρsmall : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP)
    (hρsfsmall : ρsf ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP) :
    Real.exp (-2 * (poolCount η₀ populations indecisionLimit εcov δ pAP : ℝ) * (pAP / 2) ^ 2)
        ≤ δ / (128 * (populations.card : ℝ))
    ∧ ((poolCount η₀ populations indecisionLimit εcov δ pAP : ℝ) + 2) ^ 2
        * Real.exp (-2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP : ℝ)
          * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2)
        ≤ δ / (128 * (populations.card : ℝ))
    ∧ (poolCount η₀ populations indecisionLimit εcov δ pAP : ℝ)
        * Real.exp (-2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP : ℝ)
          * ((populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ) ^ 2)
        ≤ δ / (128 * (populations.card : ℝ))
    ∧ Real.exp (-2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP : ℝ) * (cutBudget η₀
      indecisionLimit εcov / 4) ^ 2)
        ≤ δ / (128 * (populations.card : ℝ))
    ∧ (populations.card : ℝ) * (((populations.card : ℝ) + 3)
        * (prefCount η₀ populations indecisionLimit εcov δ α pAP : ℝ) ^ 2 * ρ) ≤ δ / 64
    ∧ (populations.card : ℝ) * ((poolCount η₀ populations indecisionLimit εcov δ pAP : ℝ) ^ 2 * ρsf)
        ≤ δ / 64 := by
  have hs : 0 < sig η₀ := sig_pos η₀ hsig
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hsig hε hind
  have hΔ : 0 < flipBudget η₀ populations indecisionLimit εcov δ :=
    flipBudget_pos η₀ populations hsig hε hind hcard
  have hγ : 0 < screenMargin η₀ populations indecisionLimit εcov δ :=
    screenMargin_pos η₀ populations hsig hε hind hcard
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  have hε₀ : 0 < ε₀ := by rw [hε₀def]; positivity
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP with hmdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP with hMdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ with hκdef
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  have hMR : (0 : ℝ) ≤ (M : ℝ) := Nat.cast_nonneg _
  -- each count clears the logarithm its tail asks for
  have ctap : Real.log (1 / ε₀) / (2 * (pAP / 2) ^ 2) ≤ (M : ℝ) := by
    have heq : (1 : ℝ) / ε₀ = 128 * (populations.card : ℝ) / δ := by
      rw [hε₀def]; field_simp
    rw [heq]
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (pAP / 2) ^ 2)⌉₊ ≤ M := by
      rw [hMdef, poolCount]; omega
    exact_mod_cast hle
  have cscr : Real.log ((((M : ℝ) + 2) ^ 2) / ε₀)
      / (2 * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2) ≤ (m : ℝ) := by
    have heq : (((M : ℝ) + 2) ^ 2) / ε₀
        = 128 * (populations.card : ℝ) * ((M : ℝ) + 2) ^ 2 / δ := by
      rw [hε₀def]; field_simp; try ring
    rw [heq]
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * (populations.card : ℝ) * ((M : ℝ) + 2) ^ 2 / δ)
        / (2 * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2)⌉₊ ≤ m := by
      rw [hmdef, prefCount, ← hMdef]; omega
    exact_mod_cast hle
  have cdirty : Real.log (((M : ℝ) + 1) / ε₀)
      / (2 * ((populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ) ^ 2)
      ≤ (m : ℝ) := by
    have heq : ((M : ℝ) + 1) / ε₀
        = 128 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ := by
      rw [hε₀def]; field_simp; try ring
    rw [heq]
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * (populations.card : ℝ) * ((M : ℝ) + 1) / δ)
        / (2 * ((populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ) ^ 2)⌉₊ ≤
          m := by
      rw [hmdef, prefCount, ← hMdef]; omega
    exact_mod_cast hle
  have cth : Real.log (1 / ε₀) / (2 * (cutBudget η₀ indecisionLimit εcov / 4) ^ 2) ≤ (m : ℝ) := by
    have heq : (1 : ℝ) / ε₀ = 128 * (populations.card : ℝ) / δ := by
      rw [hε₀def]; field_simp
    rw [heq]
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈Real.log (128 * (populations.card : ℝ) / δ)
        / (2 * (cutBudget η₀ indecisionLimit εcov / 4) ^ 2)⌉₊ ≤ m := by
      rw [hmdef, prefCount]; omega
    exact_mod_cast hle
  -- the tails those counts kill
  have ttap : Real.exp (-2 * (M : ℝ) * (pAP / 2) ^ 2) ≤ ε₀ := by
    have h := tail_le_of_count (γ := pAP / 2) (c := 1) (by positivity) hε₀ (by norm_num) ctap
    linarith
  have tscr : ((M : ℝ) + 2) ^ 2
      * Real.exp (-2 * (m : ℝ) * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2)
      ≤ ε₀ := tail_le_of_count (by positivity) hε₀ (by positivity) cscr
  have tdirty : ((M : ℝ) + 1) * Real.exp (-2 * (m : ℝ)
      * ((populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ) ^ 2) ≤ ε₀ :=
    tail_le_of_count (by positivity) hε₀ (by linarith) cdirty
  have tdirty' : (M : ℝ) * Real.exp (-2 * (m : ℝ)
      * ((populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ) ^ 2) ≤ ε₀ := by
    refine le_trans ?_ tdirty
    exact mul_le_mul_of_nonneg_right (by linarith) (Real.exp_nonneg _)
  have tth : Real.exp (-2 * (m : ℝ) * (cutBudget η₀ indecisionLimit εcov / 4) ^ 2) ≤ ε₀ := by
    have h := tail_le_of_count (γ := cutBudget η₀ indecisionLimit εcov / 4) (c := 1) (by
      positivity) hε₀
      (by norm_num) cth
    linarith
  -- the collision terms, with the state's code kept opaque
  have hden : (0 : ℝ) < 64 * ((populations.card : ℝ) + 3) ^ 3
      * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) := by positivity
  have hcapval : collisionCap η₀ populations indecisionLimit εcov δ α pAP
      = δ / (64 * ((populations.card : ℝ) + 3) ^ 3
        * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1)) := by
    rw [collisionCap, ← hmdef, ← hMdef]
  have hcoll1 : (populations.card : ℝ) * (((populations.card : ℝ) + 3) * (m : ℝ) ^ 2 * ρ)
      ≤ δ / 64 := by
    have hstep : (populations.card : ℝ) * (((populations.card : ℝ) + 3) * (m : ℝ) ^ 2)
        ≤ ((populations.card : ℝ) + 3) ^ 3 * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) := by
      have h1 : (populations.card : ℝ) * ((populations.card : ℝ) + 3)
          ≤ ((populations.card : ℝ) + 3) ^ 3 := mul_self_add_le_cube hcard.le
      have h2 : (m : ℝ) ^ 2 ≤ (m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1 := by
        nlinarith [sq_nonneg (M : ℝ)]
      calc (populations.card : ℝ) * (((populations.card : ℝ) + 3) * (m : ℝ) ^ 2)
          = ((populations.card : ℝ) * ((populations.card : ℝ) + 3)) * (m : ℝ) ^ 2 := by ring
        _ ≤ ((populations.card : ℝ) + 3) ^ 3 * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) :=
            mul_le_mul h1 h2 (by positivity) (by positivity)
    have hro : ρ ≤ δ / (64 * ((populations.card : ℝ) + 3) ^ 3
        * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1)) := by rwa [hcapval] at hρsmall
    rw [le_div_iff₀ hden] at hro
    have h3 : (populations.card : ℝ) * (((populations.card : ℝ) + 3) * (m : ℝ) ^ 2) * ρ
        ≤ (((populations.card : ℝ) + 3) ^ 3 * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1)) * ρ :=
      mul_le_mul_of_nonneg_right hstep hρ0
    linarith
  have hcoll2 : (populations.card : ℝ) * ((M : ℝ) ^ 2 * ρsf) ≤ δ / 64 := by
    have hstep : (populations.card : ℝ) * (M : ℝ) ^ 2
        ≤ ((populations.card : ℝ) + 3) ^ 3 * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1) := by
      have h1 : (populations.card : ℝ) ≤ ((populations.card : ℝ) + 3) ^ 3 :=
        le_cube_of_nonneg hcard.le
      have h2 : (M : ℝ) ^ 2 ≤ (m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1 := by
        nlinarith [sq_nonneg (m : ℝ)]
      exact mul_le_mul h1 h2 (by positivity) (by positivity)
    have hro : ρsf ≤ δ / (64 * ((populations.card : ℝ) + 3) ^ 3
        * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1)) := by rwa [hcapval] at hρsfsmall
    rw [le_div_iff₀ hden] at hro
    have h3 : (populations.card : ℝ) * (M : ℝ) ^ 2 * ρsf
        ≤ (((populations.card : ℝ) + 3) ^ 3 * ((m : ℝ) ^ 2 + (M : ℝ) ^ 2 + 1)) * ρsf :=
      mul_le_mul_of_nonneg_right hstep hρsf0
    linarith
  exact ⟨ttap, tscr, tdirty', tth, hcoll1, hcoll2⟩

set_option maxHeartbeats 1000000 in
/-- The round's failure, over every population, fits the error budget. -/
lemma solved_roundFail (η₀ : ℝ) (populations : Finset J)
    {indecisionLimit εcov δ α pAP ρ ρsf : ℝ}
    (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hpAP : 0 < pAP) (hind : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ) (hρsf0 : 0 ≤ ρsf)
    (hρsmall : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP)
    (hρsfsmall : ρsf ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP) :
    (populations.card : ℝ)
        * roundFail populations (indecisionLimit / 2) (cutBudget η₀ indecisionLimit εcov)
            (sig η₀ * εcov / 4)
            (cutBudget η₀ indecisionLimit εcov / 4)
            (Real.exp (-2
              * (((solvedState η₀ populations indecisionLimit εcov δ α pAP).k - 1 : ℕ) : ℝ)
              * voteSlack η₀ ^ 2))
            (screenMargin η₀ populations indecisionLimit εcov δ / 2)
            (screenMargin η₀ populations indecisionLimit εcov δ / 2)
            ((populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ)
            (pAP / 2) ρ ρsf
            ⌊(1 - indecisionLimit)
              * ((solvedState η₀ populations indecisionLimit εcov δ α pAP).npref : ℝ)⌋₊
            (solvedState η₀ populations indecisionLimit εcov δ α pAP)
      ≤ δ / 2 := by
  have hs : 0 < sig η₀ := sig_pos η₀ hsig
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hsig hε hind
  have hcutlim := (cutBudget_le η₀ indecisionLimit εcov).2.2
  have hτ : (0 : ℝ) < sig η₀ * εcov / 4 := by positivity
  obtain ⟨ttap, tscr, tdirty', tth, hcoll1, hcoll2⟩ :=
    solved_tails η₀ populations hsig hε hind hδ hpAP hcard hρ0 hρsf0 hρsmall hρsfsmall
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  have hε₀ : 0 < ε₀ := by rw [hε₀def]; positivity
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP with hmdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP with hMdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ with hκdef
  have hBm : (solvedState η₀ populations indecisionLimit εcov δ α pAP).npref = m := rfl
  have hBM : (solvedState η₀ populations indecisionLimit εcov δ α pAP).nsuff = M := rfl
  have hBκ : ((solvedState η₀ populations indecisionLimit εcov δ α pAP).k - 1 : ℕ) = κ := by
    show κ + 1 - 1 = κ
    omega
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  -- the gate's floor, and the tail it kills
  have hsizeR : 64 / εcov ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have hle : ⌈64 / εcov⌉₊ ≤ m := by rw [hmdef, prefCount]; omega
    exact_mod_cast hle
  have hgmin : εcov * (m : ℝ) / 64
      ≤ ((solvedState η₀ populations indecisionLimit εcov δ α pAP).gmin : ℝ) := by
    show εcov * (m : ℝ) / 64 ≤ (⌊εcov * (m : ℝ) / 32⌋₊ : ℝ)
    have hlt := Nat.lt_floor_add_one (εcov * (m : ℝ) / 32)
    rw [div_le_iff₀ hε] at hsizeR
    linarith
  have cgate : Real.log (2 / ε₀) / (2 * (sig η₀ * εcov / 4) ^ 2)
      ≤ ((solvedState η₀ populations indecisionLimit εcov δ α pAP).gmin : ℝ) := by
    have heq : (2 : ℝ) / ε₀ = 256 * (populations.card : ℝ) / δ := by
      rw [hε₀def]; field_simp; try ring
    rw [heq]
    have hmlog : 64 * Real.log (256 * (populations.card : ℝ) / δ)
        / (εcov * (sig η₀ * εcov / 4) ^ 2) ≤ (m : ℝ) := by
      refine le_trans (Nat.le_ceil _) ?_
      have hle : ⌈64 * Real.log (256 * (populations.card : ℝ) / δ)
          / (εcov * (sig η₀ * εcov / 4) ^ 2)⌉₊ ≤ m := by
        rw [hmdef, prefCount]; omega
      exact_mod_cast hle
    have hlog0 : 0 ≤ Real.log (256 * (populations.card : ℝ) / δ) := by
      refine Real.log_nonneg ?_
      have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
        have h1 : 1 ≤ populations.card := by exact_mod_cast hcard
        exact_mod_cast h1
      rw [le_div_iff₀ hδ]
      linarith
    rw [div_le_iff₀ (by positivity)] at hmlog ⊢
    linarith [mul_le_mul_of_nonneg_right hgmin
      (by positivity : (0 : ℝ) ≤ 2 * (sig η₀ * εcov / 4) ^ 2)]
  have tgate : 2 * Real.exp (-2
      * ((solvedState η₀ populations indecisionLimit εcov δ α pAP).gmin : ℝ)
      * (sig η₀ * εcov / 4) ^ 2) ≤ ε₀ :=
    tail_le_of_count hτ hε₀ (by norm_num) cgate
  -- the gate's tail, now floored at the decided count rather than at `gmin`
  have hfloorge : ((solvedState η₀ populations indecisionLimit εcov δ α pAP).gmin : ℝ)
      ≤ ((⌊(1 - indecisionLimit) * (m : ℝ)⌋₊ : ℕ) : ℝ) := by
    have hg : (solvedState η₀ populations indecisionLimit εcov δ α pAP).gmin = ⌊εcov * (m : ℝ) /
      32⌋₊ := rfl
    rw [hg]
    have hgap : (0 : ℝ) ≤ 1 - indecisionLimit - εcov / 32 := by linarith
    have hmono : ⌊εcov * (m : ℝ) / 32⌋₊ ≤ ⌊(1 - indecisionLimit) * (m : ℝ)⌋₊ :=
      Nat.floor_mono (by nlinarith [mul_nonneg hgap hmR])
    exact_mod_cast hmono
  have tgate' : 2 * Real.exp (-2 * ((⌊(1 - indecisionLimit) * (m : ℝ)⌋₊ : ℕ) : ℝ)
      * (sig η₀ * εcov / 4) ^ 2) ≤ ε₀ := by
    refine le_trans ?_ tgate
    have hexp : Real.exp (-2 * ((⌊(1 - indecisionLimit) * (m : ℝ)⌋₊ : ℕ) : ℝ)
          * (sig η₀ * εcov / 4) ^ 2)
        ≤ Real.exp (-2 * ((solvedState η₀ populations indecisionLimit εcov δ α pAP).gmin : ℝ)
          * (sig η₀ * εcov / 4) ^ 2) := by
      refine Real.exp_le_exp.2 ?_
      nlinarith [hfloorge, sq_nonneg (sig η₀ * εcov / 4)]
    linarith
  -- the vote's misfire rate sits under half the cut budget, so both count tails are no worse
  -- than the threshold's
  have hEc : Real.exp (-2 * (κ : ℝ) * voteSlack η₀ ^ 2) ≤ cutBudget η₀ indecisionLimit εcov / 2 :=
    famCount_tail η₀ populations hsig hε hind
  have hE0 : 0 ≤ Real.exp (-2 * (κ : ℝ) * voteSlack η₀ ^ 2) := Real.exp_nonneg _
  have tl : Real.exp (-2 * (m : ℝ) * (indecisionLimit / 2
      - Real.exp (-2 * (κ : ℝ) * voteSlack η₀ ^ 2)) ^ 2) ≤ ε₀ :=
    le_trans (exp_tail_anti hmR (by linarith) (by linarith)) tth
  have tlc : Real.exp (-2 * (m : ℝ) * (cutBudget η₀ indecisionLimit εcov
      - Real.exp (-2 * (κ : ℝ) * voteSlack η₀ ^ 2)) ^ 2) ≤ ε₀ :=
    le_trans (exp_tail_anti hmR (by linarith) (by linarith)) tth
  have hcε : (populations.card : ℝ) * ε₀ = δ / 128 := by
    rw [hε₀def]
    field_simp
  rw [roundFail, hBm, hBM, hBκ]
  refine le_trans (mul_le_mul_of_nonneg_left
    (show _ ≤ 10 * ε₀ + (((populations.card : ℝ) + 3) * (m : ℝ) ^ 2 * ρ
        + 2 * (M : ℝ) ^ 2 * ρsf) from ?_) hcard.le) ?_
  · have tscr' : ((M : ℝ) + 2) ^ 2
        * Real.exp (-2 * (m : ℝ) * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^ 2) ≤
          ε₀ := tscr
    linarith [ttap, tscr', tdirty', tth, tl, tlc, tgate']
  · have hexp : (populations.card : ℝ)
        * (10 * ε₀ + (((populations.card : ℝ) + 3) * (m : ℝ) ^ 2 * ρ + 2 * (M : ℝ) ^ 2 * ρsf))
        = 10 * ((populations.card : ℝ) * ε₀)
          + ((populations.card : ℝ) * (((populations.card : ℝ) + 3) * (m : ℝ) ^ 2 * ρ)
            + 2 * ((populations.card : ℝ) * ((M : ℝ) ^ 2 * ρsf))) := by ring
    rw [hexp, hcε]
    linarith [hcoll1, hcoll2]

set_option maxHeartbeats 1000000 in
/-- The top rung carries its own share of the error budget, `δ/8`. -/
lemma solved_share (η₀ : ℝ) (populations : Finset J) {εcov δ α pAP ρ ρsf : ℝ}
    (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hδ : 0 < δ) (hpAP : 0 < pAP)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) (hρ0 : 0 ≤ ρ) (hρsf0 : 0 ≤ ρsf)
    (hρsmall : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP)
    (hρsfsmall : ρsf ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP) :
    stateFail η₀ populations indecisionLimit εcov δ ρ ρsf (solvedState η₀ populations
      indecisionLimit εcov δ α pAP)
      ≤ δ / 8 := by
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hsig hε hind
  obtain ⟨-, tscr, tdirty', tth, hcoll1, hcoll2⟩ :=
    solved_tails η₀ populations hsig hε hind hδ hpAP hcard hρ0 hρsf0 hρsmall hρsfsmall
  set ε₀ : ℝ := δ / (128 * (populations.card : ℝ)) with hε₀def
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP with hmdef
  set M : ℕ := poolCount η₀ populations indecisionLimit εcov δ pAP with hMdef
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ with hκdef
  have hBm : (solvedState η₀ populations indecisionLimit εcov δ α pAP).npref = m := rfl
  have hBM : (solvedState η₀ populations indecisionLimit εcov δ α pAP).nsuff = M := rfl
  have hBκ : ((solvedState η₀ populations indecisionLimit εcov δ α pAP).k - 1 : ℕ) = κ := by
    show κ + 1 - 1 = κ
    omega
  have hmR : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  -- the coverage tail sits under the threshold's, whose margin is smaller
  have tcov : Real.exp (-2 * (m : ℝ) * (εcov / 4) ^ 2) ≤ ε₀ := by
    refine le_trans (Real.exp_le_exp.2 ?_) tth
    have h : cutBudget η₀ indecisionLimit εcov / 4 ≤ εcov / 4 := by
      linarith [(cutBudget_le η₀ indecisionLimit εcov).1]
    nlinarith [mul_le_mul h h (by linarith) (by linarith)]
  have hcε : (populations.card : ℝ) * ε₀ = δ / 128 := by
    rw [hε₀def]
    field_simp
  have hcoll1' : (populations.card : ℝ) * ((((populations.card : ℝ) + 1) * (m : ℝ) ^ 2 * ρ)
      + (m : ℝ) ^ 2 * ρ) ≤ δ / 64 := by
    refine le_trans (mul_le_mul_of_nonneg_left ?_ hcard.le) hcoll1
    nlinarith [mul_nonneg (sq_nonneg (m : ℝ)) hρ0]
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    have h1 : 1 ≤ populations.card := by exact_mod_cast hcard
    exact_mod_cast h1
  have hcoll2' : (M : ℝ) ^ 2 * ρsf ≤ δ / 64 :=
    le_trans (le_mul_of_one_le_left (by positivity) hcard1) hcoll2
  rw [stateFail, hBm, hBM]
  have hsum : (((populations.card : ℝ) + 1) * (m : ℝ) ^ 2 * ρ
      + (Real.exp (-2 * (m : ℝ) * (εcov / 4) ^ 2)
        + (((m : ℝ) ^ 2 * ρ
            + ((M : ℝ) + 2) ^ 2
              * Real.exp (-2 * (m : ℝ) * (screenMargin η₀ populations indecisionLimit εcov δ / 2) ^
                2)
            + (M : ℝ) * Real.exp (-2 * (m : ℝ)
              * ((populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ) ^ 2))
          + (Real.exp (-2 * (m : ℝ) * (cutBudget η₀ indecisionLimit εcov / 4) ^ 2)
            + Real.exp (-2 * (m : ℝ) * (cutBudget η₀ indecisionLimit εcov / 2) ^ 2)))))
      ≤ ((((populations.card : ℝ) + 1) * (m : ℝ) ^ 2 * ρ) + (m : ℝ) ^ 2 * ρ) + 5 * ε₀ := by
    have tfam : Real.exp (-2 * (m : ℝ) * (cutBudget η₀ indecisionLimit εcov / 2) ^ 2) ≤ ε₀ :=
      le_trans (exp_tail_anti hmR (by linarith) (by linarith)) tth
    linarith [tcov, tscr, tdirty', tth, tfam]
  have hscale := mul_le_mul_of_nonneg_left hsum hcard.le
  have hexp : (populations.card : ℝ) * ((((populations.card : ℝ) + 1) * (m : ℝ) ^ 2 * ρ
        + (m : ℝ) ^ 2 * ρ) + 5 * ε₀)
      = (populations.card : ℝ) * ((((populations.card : ℝ) + 1) * (m : ℝ) ^ 2 * ρ)
        + (m : ℝ) ^ 2 * ρ) + 5 * ((populations.card : ℝ) * ε₀) := by ring
  linarith

set_option maxHeartbeats 1000000 in
/-- The computed ladder holds a state whose round can pass.

`PassableAt` is arithmetic: every clause is an inequality among the state's budgets, the
oracle's rates, the suffix distribution's findability and the error budget.  The witness is
`solvedState`, the ladder's top rung, and every clause is discharged from the closed forms
that define it — no reachability is assumed.

The order the constants come out in: the miscut budget
`lcut = cutBudget η₀ indecisionLimit εcov` under the indecision limit, the flip budget
`Δ = flipBudget` under `lcut` at the flip fraction the vote absorbs, the screen's two
margins at
`screenMargin` (so the rate window is non-empty), then the prefix count large enough for
every exponential — the share's own among them — then `α` at the gate's own tail
`exp(−2·gmin·τ²)`,
then the pool at `k/(pAP − t)`.  The collision masses enter as `m²ρ`, which is what
`collisionCap` bounds. -/
theorem exists_passable (O : Oracle μ S) (populations : Finset J) (D : J → Measure S)
    (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (indecisionLimit εcov α δ ρ ρsf pAP : ℝ)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (hεcov : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hαpos : 0 < α) (hα : α < 1 / 2) (hindLim : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2)
    (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP)
    (hρsfcap : ρsf ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP) :
    ∃ B : State, B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP ρ ρsf
      ∧ PassableAt O η₀ populations D Dsf indecisionLimit εcov α δ ρ ρsf pAP B := by
  classical
  have hcard : (0 : ℝ) < (populations.card : ℝ) := by
    exact_mod_cast Finset.card_pos.2 hpop
  have hη0 : 0 ≤ O.η := eta_nonneg O
  have hs : 0 < sig η₀ := sig_pos η₀ hη₀
  have hsval : sig η₀ = 1 / 2 - η₀ := rfl
  have hcut : 0 < cutBudget η₀ indecisionLimit εcov := cutBudget_pos η₀ hη₀ hεcov hindLim
  have hΔ : 0 < flipBudget η₀ populations indecisionLimit εcov δ :=
    flipBudget_pos η₀ populations hη₀ hεcov hindLim hcard
  have hγ : 0 < screenMargin η₀ populations indecisionLimit εcov δ :=
    screenMargin_pos η₀ populations hη₀ hεcov hindLim hcard
  set κ : ℕ := famCount η₀ populations indecisionLimit εcov δ with hκdef
  have hκpos : 0 < κ := famCount_pos η₀ populations εcov δ
  have hκR : (0 : ℝ) < (κ : ℝ) := by exact_mod_cast hκpos
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP with hmdef
  have hmpos : 0 < m := by rw [hmdef, prefCount]; omega
  have hmR : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hmpos
  set B : State := solvedState η₀ populations indecisionLimit εcov δ α pAP with hBdef
  -- the fields, as the definitions give them
  have hBm : B.npref = m := rfl
  have hBk : B.k = κ + 1 := rfl
  have hBkappa : (B.k - 1 : ℕ) = κ := by rw [hBk]; omega
  have hBgmin : B.gmin = ⌊εcov * (m : ℝ) / 32⌋₊ := rfl
  -- the gate's floor, from both sides
  have hsizeR : 64 / εcov ≤ (m : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    have : ⌈64 / εcov⌉₊ ≤ m := by rw [hmdef, prefCount]; omega
    exact_mod_cast this
  have hsize : 2 ≤ εcov * (m : ℝ) / 32 := by
    rw [div_le_iff₀ hεcov] at hsizeR
    linarith
  have hgminLe : (B.gmin : ℝ) ≤ εcov * (m : ℝ) / 32 := by
    rw [hBgmin]
    exact Nat.floor_le (by positivity)
  have hgminGe : εcov * (m : ℝ) / 64 ≤ (B.gmin : ℝ) := by
    rw [hBgmin]
    have hlt := Nat.lt_floor_add_one (εcov * (m : ℝ) / 32)
    linarith
  set x : ℝ := (κ : ℝ) / 2 with hxdef
  have hxpos : 0 < x := by
    rw [hxdef]
    exact div_pos hκR (by norm_num)
  have hceil1 : 1 ≤ ⌈x⌉₊ := Nat.one_le_ceil_iff.2 hxpos
  have hxexact : (⌈x⌉₊ : ℝ) = x := by rw [hxdef, hκdef]; exact famCount_half η₀ populations εcov δ
  have hκ0 : (0 : ℝ) ≤ (κ : ℝ) := Nat.cast_nonneg _
  obtain ⟨hlowShift, hhiShift⟩ := vote_shifts O hηle hη₀ hκ0
  -- the fields, named before the definitions are made opaque
  have hBM : B.nsuff = poolCount η₀ populations indecisionLimit εcov δ pAP := rfl
  have hBcn : B.cn = 1 := rfl
  have hBcd : B.cd = 2 := rfl
  have hBlo : B.lo = ⌈x⌉₊ - 1 := rfl
  have hBhi : B.hi = ⌈x⌉₊ + 1 := rfl
  have hBscd : B.scd = ⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 := rfl
  have hBsc : B.sc = ⌈((⌈15 / (2 * screenMargin η₀ populations indecisionLimit εcov δ)⌉₊ + 1 : ℕ) :
    ℝ)
      * screenMargin η₀ populations indecisionLimit εcov δ⌉₊ := rfl
  have hMceil : 2 * ((κ : ℝ) + 1) / pAP ≤ ((B.nsuff : ℕ) : ℝ) := by
    rw [hBM, poolCount, ← hκdef]
    push_cast
    linarith [Nat.le_ceil (2 * ((κ : ℝ) + 1) / pAP),
      Nat.cast_nonneg (α := ℝ)
        ⌈Real.log (128 * (populations.card : ℝ) / δ) / (2 * (pAP / 2) ^ 2)⌉₊]
  clear_value B κ m x
  refine ⟨B, Finset.mem_filter.2 ⟨?_, ⟨?_, ?_, ?_, ?_, ?_⟩⟩,
    sig η₀ * εcov / 4, cutBudget η₀ indecisionLimit εcov / 4, pAP / 2,
    voteSlack η₀, screenMargin η₀ populations indecisionLimit εcov δ / 2,
    screenMargin η₀ populations indecisionLimit εcov δ / 2,
    (populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ,
    flipBudget η₀ populations indecisionLimit εcov δ, cutBudget η₀ indecisionLimit εcov, flipFrac
      η₀,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  -- in the schedule
  · rw [hBdef]
    exact solvedState_mem_schedule η₀ populations εcov δ α pAP
  -- Capped
  · rw [hBlo, hBhi]
    omega
  · rw [hBcn, hBcd]
    norm_num
  · rw [hBm]; exact hmpos
  · have hκ1 : (1 : ℝ) ≤ (κ : ℝ) := by exact_mod_cast hκpos
    have h := mul_le_mul_of_nonneg_right hMceil (by positivity : (0 : ℝ) ≤ pAP / 2)
    have heq : 2 * ((κ : ℝ) + 1) / pAP * (pAP / 2) = (κ : ℝ) + 1 := by
      field_simp
    rw [heq] at h
    linarith
  · have hval : δ * (m : ℝ) / (8 * (m : ℝ)) = δ / 8 := by
      rw [mul_comm (8 : ℝ) (m : ℝ), mul_comm δ (m : ℝ),
        mul_div_mul_left _ _ (ne_of_gt hmR)]
    rw [hBm, ← hmdef, hval, hBdef]
    exact solved_share η₀ populations hη₀ hεcov hindLim hδ hpAPPositive hcard hρ0 hρsf0 hρcap
      hρsfcap
  -- PassableAt
  · rw [hBm]; exact hmpos
  · rw [hBk]; omega
  · rw [hBcn, hBcd]
    norm_num
  · exact hindLim
  · exact hind1
  · exact hε1
  · exact div_nonneg (mul_nonneg hs.le hεcov.le) (by norm_num)
  · linarith
  · linarith
  · exact (voteSlack_pos η₀ hη₀).le
  · linarith [hγ.le]
  · linarith [hγ.le]
  · exact mul_nonneg hcard.le hΔ.le
  · exact hΔ
  · exact flipFrac_pos η₀ hη₀
  · linarith
  · exact hcut
  · exact (cutBudget_le η₀ indecisionLimit εcov).2.2
  · exact hpAPBound
  · exact hρsf
  -- the family's flips fit the cut budget
  · have hne : ((populations.card : ℝ)) ≠ 0 := ne_of_gt hcard
    have hFne : flipFrac η₀ ≠ 0 := ne_of_gt (flipFrac_pos η₀ hη₀)
    have hid : ((populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ
        + (populations.card : ℝ) * flipBudget η₀ populations indecisionLimit εcov δ) / flipFrac η₀
        = cutBudget η₀ indecisionLimit εcov * (2 / 3) := by
      rw [flipBudget]
      field_simp
      ring
    rw [hid]
    linarith
  -- the screen's rate
  · rw [hBscd]
    omega
  · rw [hBsc, hBscd]
    have h2 : (2 : ℝ) * (screenMargin η₀ populations indecisionLimit εcov δ / 2)
        = screenMargin η₀ populations indecisionLimit εcov δ := by ring
    rw [h2]
    exact Nat.le_ceil _
  · rw [hBdef]
    exact (rung_facts O populations (δ := δ) (pAP := pAP) (prefCount η₀ populations indecisionLimit
      εcov δ α pAP)
      hηle hη₀ hεcov hindLim hcard).2.2.2.2.1
  -- the pool holds a family
  · rw [hBk]
    have hceil := hMceil
    rw [div_le_iff₀ hpAPPositive] at hceil
    push_cast
    linarith
  -- the thresholds decide, and decide right
  · rw [hBhi, hBk, show (⌈x⌉₊ + 1 - 1 : ℕ) = ⌈x⌉₊ from by omega,
      show (κ + 1 - 1 : ℕ) = κ from by omega]
    linarith [hxexact, hhiShift]
  · rw [hBlo, hBk, show (κ + 1 - 1 : ℕ) = κ from by omega, Nat.cast_sub hceil1]
    push_cast
    linarith [hxexact, hlowShift]
  · rw [hBhi, hBk, show (κ + 1 - 1 : ℕ) = κ from by omega,
      show (⌈x⌉₊ + 1 - 1 : ℕ) = ⌈x⌉₊ from by omega]
    linarith [hxexact, hlowShift]
  · rw [hBlo, hBk, show (κ + 1 - 1 : ℕ) = κ from by omega, Nat.cast_sub hceil1]
    push_cast
    linarith [hxexact, hhiShift]
  -- the gate's two sides clear their thresholds
  · intro n c hn₀ hnc hcm
    have hcR : (c : ℝ) ≤ (m : ℝ) := by rw [← hBm]; exact_mod_cast hcm
    have hnR : ((⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊ : ℕ) : ℝ) ≤ (n : ℝ) := by
      exact_mod_cast hn₀
    have hfloor : (1 - indecisionLimit) * (m : ℝ) - 1
        ≤ ((⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊ : ℕ) : ℝ) := by
      rw [hBm]
      linarith [Nat.lt_floor_add_one ((1 - indecisionLimit) * (m : ℝ))]
    have hlow : (1 - indecisionLimit) * (m : ℝ) - 1 ≤ (n : ℝ) := le_trans hfloor hnR
    have hm64 : (64 : ℝ) ≤ (m : ℝ) := by
      have h1 : (64 : ℝ) / εcov ≤ (m : ℝ) := hsizeR
      rw [div_le_iff₀ hεcov] at h1
      nlinarith
    clear hBsc hBscd hBlo hBhi hBM hMceil hBcn hBcd hBdef hBk hBkappa hBgmin hBm
      hxexact hlowShift hhiShift hgminLe hgminGe hnR hfloor hsize hsizeR
      hρcap hρsfcap hρ hρsf hpAPBound
    have hcn : (c : ℝ) ≤ 4 * (n : ℝ) := by nlinarith [hcR, hlow, hm64, hind1]
    have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg _
    have hc0 : (0 : ℝ) ≤ (c : ℝ) := Nat.cast_nonneg _
    have hss : sig η₀ ≤ 1 / 2 - O.η := by rw [sig]; linarith
    -- a wrong cut can cost the gate a whole read, so the cut budget is held under `s`
    have ha : 2 * cutBudget η₀ indecisionLimit εcov * (c : ℝ) ≤ sig η₀ * (n : ℝ) / 8 := by
      nlinarith [mul_le_mul (cutBudget_le η₀ indecisionLimit εcov).2.1 hcn hc0
        (by linarith : (0 : ℝ) ≤ sig η₀ / 64)]
    have h1 : (1 - O.η) * (2 * cutBudget η₀ indecisionLimit εcov * (c : ℝ))
        ≤ sig η₀ * (n : ℝ) / 8 := by
      have hb : (0 : ℝ) ≤ 2 * cutBudget η₀ indecisionLimit εcov * (c : ℝ) := by positivity
      nlinarith [mul_le_mul_of_nonneg_right (by linarith : 1 - O.η ≤ 1) hb]
    have h2 : (n : ℝ) * (sig η₀ * εcov / 4 + sig η₀ * εcov / 4) ≤ (n : ℝ) * (sig η₀ / 2) := by
      refine mul_le_mul_of_nonneg_left ?_ hn0
      nlinarith [hs]
    have h3 : (n : ℝ) * (1 / 2 + sig η₀) ≤ (n : ℝ) * (1 - O.η) :=
      mul_le_mul_of_nonneg_left (by linarith) hn0
    nlinarith [h1, h2, h3]
  · rw [hBdef]
    exact solved_alpha η₀ populations hη₀ hεcov hε1 hδ hαpos hα hpAPPositive hcard
  · rw [hBdef]
    exact (rung_facts O populations (δ := δ) (pAP := pAP) (prefCount η₀ populations indecisionLimit
      εcov δ α pAP)
      hηle hη₀ hεcov hindLim hcard).2.2.2.2.2.2.2
  · rw [hBdef]
    exact solved_roundFail η₀ populations hη₀ hεcov hε1 hδ hδ1 hpAPPositive hindLim hind1
      hcard hρ0 hρsf0 hρcap hρsfcap

set_option maxHeartbeats 1000000 in
/-- Part 2 — the loop terminates: except with probability `δ/2`, some reachable state
passes both tests.

Each growth step draws fresh suffixes, accept-preserving with probability `≥ pAP`; once the
pool holds enough of them and the prefix count is large enough, every population's vote is
decisive on all but `indecisionLimit` of its mass.  The per-step trigger is block-local, so
`geometric_miss_triggered` gives `(1 − p)^N` and `geom_le` drives it under `δ/2`.

Here `admitted` has to pass, not merely be sound — the gate's power, the complement of the
`α` it spends — which is why `pAP > 0` is needed rather than merely useful.
`ACCEPT_PRESERVING_GIVE_UP = 20` caps the refusals in the code; the statement carries no
cap.

Nothing here asks a population to carry both labels: `admitted` is an implication, so a
sample below `gmin` is skipped rather than failed and an all-accepting population is
admitted.  What the round pays for that is coverage, which is why `gmin ≤ εcov·m/32`.

`cutBudget` is held under `indecisionLimit/2` by its own definition, so the family size is
solved against a single misfire rate rather than the max of the round's two. -/
theorem loop_terminates {Pre : Set S} (hflat : Flat Pre) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (indecisionLimit εcov α : ℝ) (ρ pAP δ : ℝ)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (hεcov : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hαpos : 0 < α) (hα : α < 1 / 2) (hindLim : 0 < indecisionLimit)
    (hind1 : indecisionLimit ≤ 1 / 2)
    (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP)
    (hρsf : collisionMass Dsf ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP) :
    (runMeasure μ D Dsf).real {x | ∀ B : {B : State //
        B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP ρ (collisionMass Dsf)},
      x ∉ ret O.mq populations indecisionLimit α B.val} ≤ δ / 2 := by
  classical
  obtain ⟨B, hB, hpass⟩ := exists_passable O populations D Dsf indecisionLimit εcov α δ ρ
    (collisionMass Dsf) pAP hsig hpop hηle hη₀ hεcov hε1 hδ hδ1 hαpos hα hindLim
    hind1
    hpAPPositive
    hpAPBound hρ hρ0 le_rfl (tsum_nonneg (fun a => sq_nonneg _)) hρcap hρsf
  obtain ⟨τ, th, tap, γdec, γscr, γdirty, gdirty, Δ, lcut, f, hmpos, hkpos, hcd, hindLim,
    hind1', hε1, hτ, hth, htap, hγdec, hγscr, hγdirty, hgdirty, hΔ, hf0, hpAP0, hlcut, hlcl,
    hpAPBound, hρsf, hheavy, hscd, hscLow, hscHigh, hcount,
    hhiUp, hloUp, hhiLo, hloLo, hga, hα, hEhalf, hbudget⟩ := hpass
  set l : ℝ := indecisionLimit / 2 with hl
  have hlpos : 0 < l := by rw [hl]; linarith
  have hl2 : 2 * l = indecisionLimit := by rw [hl]; ring
  set κ : ℕ := B.k - 1 with hκ
  set E : ℝ := Real.exp (-2 * (κ : ℝ) * γdec ^ 2) with hE
  obtain ⟨j₀, hj₀⟩ := hpop
  have hρsf0 : (0 : ℝ) ≤ collisionMass Dsf := tsum_nonneg (fun a => sq_nonneg _)
  -- the whole failure at the one state, population by population
  have hsub : {x : Run Ω S J | ∀ B' : {B : State //
          B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP ρ (collisionMass Dsf)},
        x ∉ ret O.mq populations indecisionLimit α B'.val}
      ⊆ ⋃ j ∈ populations,
        {x : Run Ω S J | x ∉ retAt O populations indecisionLimit α B j} := by
    intro x hx
    by_contra hc
    simp only [Set.mem_iUnion, not_exists, exists_prop, Set.mem_setOf_eq, not_and] at hc
    refine hx ⟨B, hB⟩ (mem_ret_of_retAt O populations ⟨j₀, hj₀⟩ indecisionLimit α B x
      (fun j hj => ?_))
    by_contra hcj
    exact hc j hj hcj
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_biUnion_finset_le _ _) ?_
  have hper : ∀ j ∈ populations,
      (runMeasure μ D Dsf).real
          {x : Run Ω S J | x ∉ retAt O populations indecisionLimit α B j}
        ≤ roundFail populations l lcut τ th E γscr γdirty gdirty tap ρ (collisionMass Dsf) ⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊ B := by
    intro j hj
    have hstall := measureReal_stalled_le hflat O populations D Dsf hsupp B hcd (by omega) j₀ hj₀
      γscr pAP tap (collisionMass Dsf) ρ hγscr hpAP0 htap hpAPBound hscd hsig.le hscLow hcount hρsf hρsf0
      (hρ j₀ hj₀) hρ0
    have hdirtyI := measureReal_dirtyMember_le hflat O populations D Dsf hsupp j hj B hcd hsig.le
      hmpos Δ γdirty gdirty ρ hΔ hγdirty hgdirty hρ0 (hρ j hj) hscd hscHigh
    have hnoclean := measureReal_noCleanRef_le D Dsf O populations B pAP tap
      (collisionMass Dsf) hpAP0 htap hpAPBound le_rfl hρsf0
      (by
        have hk2 : (2 : ℝ) ≤ (B.k : ℝ) := by exact_mod_cast hkpos
        linarith [hcount, hk2])
    have hdirty : (runMeasure μ D Dsf).real
        {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O.mq populations x B,
          flipMass O (D j) v ≤ (populations.card : ℝ) * Δ + gdirty}
        ≤ ((B.npref : ℝ) ^ 2 * ρ
            + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γdirty ^ 2)
            + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * gdirty ^ 2))
          + (Real.exp (-2 * (B.nsuff : ℝ) * tap ^ 2)
            + (B.nsuff : ℝ) ^ 2 * collisionMass Dsf) := by
      have hcov : {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O.mq populations x B,
            flipMass O (D j) v ≤ (populations.card : ℝ) * Δ + gdirty}
          ⊆ ({x : Run Ω S J | ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
                ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
              ∩ {x : Run Ω S J | ¬ ∀ v ∈ clusterAt O.mq populations x B,
                flipMass O (D j) v ≤ (populations.card : ℝ) * Δ + gdirty})
            ∪ {x : Run Ω S J | ¬ ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
                ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0} := by
        intro x hx
        by_cases hc : ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
            ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0
        · exact Or.inl ⟨hc, hx⟩
        · exact Or.inr hc
      refine le_trans (measureReal_mono hcov (measure_ne_top _ _)) ?_
      refine le_trans (measureReal_union_le _ _) ?_
      gcongr
    have hdec : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
        κ ≤ F.card → F.card ≤ κ →
        μ.real {ω | ¬ decided O.mq B.lo (B.hi - 1) F p ω} ≤ E := by
      intro F p hfl hmin hmax
      have hcardF : F.card = κ := le_antisymm hmax hmin
      refine le_trans (decided_whp O F p B.lo (B.hi - 1) f γdec hγdec hsig.le hfl ?_ ?_) ?_
      · rw [hcardF]; exact hhiUp
      · rw [hcardF]; exact hloUp
      · rw [hE, hcardF]
    have hcut : ∀ (F : Finset S) (p : S), flipCount O F p ≤ (F.card : ℝ) * f →
        κ ≤ F.card → F.card ≤ κ →
        μ.real {ω | ¬ cutCorrect O B.lo (B.hi - 1) F p ω} ≤ E := by
      intro F p hfl hmin hmax
      have hcardF : F.card = κ := le_antisymm hmax hmin
      refine le_trans (cutCorrect_whp O F p B.lo (B.hi - 1) f γdec hsig.le hfl hγdec ?_ ?_) ?_
      · rw [hcardF]; exact hhiLo
      · rw [hcardF]; exact hloLo
      · rw [hE, hcardF]
    have hl1' : 2 * l ≤ 1 := by rw [hl]; linarith [hind1']
    have hnloB' : ((⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊ : ℕ) : ℝ)
        ≤ (1 - 2 * l) * (B.npref : ℝ) := by
      have hmnn : (0 : ℝ) ≤ (B.npref : ℝ) := Nat.cast_nonneg _
      have hnn : (0 : ℝ) ≤ (1 - indecisionLimit) * (B.npref : ℝ) := by nlinarith [hind1']
      have hfl := Nat.floor_le hnn
      rw [hl]
      linarith [hfl]
    have hmain := measureReal_notRetAt_le hflat O populations D Dsf hsupp j hj B
      hmpos hsig.le
      α τ l lcut f E ((populations.card : ℝ) * Δ + gdirty) ρ th κ κ
      ⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊ (by omega)
      (Real.exp_nonneg _) (by linarith) (by rw [hl]; exact hlcl) hτ
      hl1' hnloB'
      hρ hρ0
      (by positivity) hth hf0
      hheavy _ hstall _ hdirty hdec hcut hga hα
    rw [hl2] at hmain
    refine le_trans hmain (le_of_eq ?_)
    unfold roundFail
    ring
  have hsum : ∑ j ∈ populations, (runMeasure μ D Dsf).real
        {x : Run Ω S J | x ∉ retAt O populations indecisionLimit α B j}
      ≤ (populations.card : ℝ)
          * roundFail populations l lcut τ th E γscr γdirty gdirty tap ρ (collisionMass Dsf) ⌊(1 - indecisionLimit) * (B.npref : ℝ)⌋₊ B := by
    refine le_trans (Finset.sum_le_sum hper) (le_of_eq ?_)
    rw [Finset.sum_const, nsmul_eq_mul]
  linarith [hsum, hbudget]

#print axioms validity_of_returned
#print axioms loop_terminates

end Loop

/-- `validity_of_returned` (whatever the loop returns is good, whenever it is returned) and
`loop_terminates` (it returns), each except w.p. `δ/2`, glued by `sound_and_terminating`. -/
theorem clustering_correct : ClusteringCorrect := by
  intro Ω _ μ _ S _ J _ O populations D Dsf _ _ Pre η₀ indecisionLimit εcov α δ ρ pAP k
    hηle hη₀ hpop hflat hsupp hρ hpAPPositive hpAPBound hindLim hind1 hαpos hα hεcov
    hε1 hδ _hbudget hρcap hρsf
  have hsig : O.η < 1 / 2 := lt_of_le_of_lt hηle hη₀
  rw [O.apSet_eq] at hpAPBound
  by_cases hδ1 : δ ≤ 1
  case neg =>
    exact le_trans (by linarith [not_le.1 hδ1] : (1 : ℝ) - δ ≤ 0) measureReal_nonneg
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    exact_mod_cast Finset.card_pos.2 hpop
  have hfind : Real.exp (-2 * (poolCount η₀ populations indecisionLimit εcov δ pAP : ℝ)
      * (pAP / 2) ^ 2) ≤ δ / 4 :=
    le_trans (solved_findability η₀ populations hδ hpAPPositive hcard1) (by linarith)
  have h := sound_and_terminating (runMeasure μ D Dsf)
    (fun B : {B : State //
        B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP ρ (collisionMass Dsf)} =>
      ret O.mq populations indecisionLimit α B.val ∩ FailAt O populations D εcov B.val)
    (fun B : {B : State //
        B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP ρ (collisionMass Dsf)} =>
      ret O.mq populations indecisionLimit α B.val) δ
    (validity_of_returned hflat O populations D Dsf hsupp indecisionLimit εcov α hindLim hsig hpop
      hηle hη₀ ρ (collisionMass Dsf) hρ le_rfl (tsum_nonneg (fun a => sq_nonneg _))
      hεcov δ hδ pAP hpAPPositive.le hpAPBound hfind)
    (loop_terminates hflat O populations D Dsf hsupp indecisionLimit εcov α ρ pAP
      δ hsig hpop hηle hη₀ hεcov hε1 hδ hδ1 hαpos hα hindLim hind1
      hpAPPositive
      hpAPBound hρ (le_trans (tsum_nonneg (fun a => sq_nonneg _))
        (hρ hpop.choose hpop.choose_spec)) hρcap hρsf)
  refine le_trans h (le_of_eq ?_)
  congr 1
  ext x
  simp only [Set.mem_setOf_eq, Set.mem_inter_iff, FailAt, not_and, not_not]

#print axioms clustering_correct

end OrthoDFA
