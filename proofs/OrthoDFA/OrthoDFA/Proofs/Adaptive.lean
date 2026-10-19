import OrthoDFA.Proofs.Basics
import OrthoDFA.Proofs.Schedule
import OrthoDFA.Proofs.Hoeffding
import OrthoDFA.Proofs.Grouping
import OrthoDFA.Proofs.FreshDraws
import Mathlib.Probability.ProductMeasure
import Mathlib.Probability.Independence.InfinitePi

/-!
# The adaptive clustering loop: what a state's draws give

The facts `OrthoDFA.Proofs.GateLoop` assembles into `ClusteringCorrect`: the derived
configuration and the band, the table, pool and certification draws and their redraws, the
screen keeping the family clean, and the vote deciding, and deciding right, on all but a small
fraction of a sample.

The union bound over `stoppable` is a finite sum: a rung carries `δ·npref/(8·N)`, and
`δ·V/(16·npref)` more once it is at or above `V = validCount`, and the ladder's counts halve,
so no summable weight over all budgets is needed and no state has to be encoded as a number.
`vote_mem_grid` is what lets the state be a `State` at all — a threshold enters every event
only through the count it cuts at.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]
variable {rule : Clusterer S}

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

/-- How far a flip moves a read's mean, `1 − ηIn − ηOut`.  A class no string is in leaves its
rate unconstrained, hence the clamps. -/
noncomputable def Oracle.gap (O : Oracle μ S) : ℝ := 1 - max O.ηIn 0 - max O.ηOut 0

lemma gap_ge (O : Oracle μ S) : 1 - 2 * O.η ≤ O.gap := by
  have h0 := eta_nonneg O
  unfold Oracle.gap
  unfold Oracle.η at h0 ⊢
  have h1 : max O.ηIn 0 ≤ max O.ηIn O.ηOut := max_le (le_max_left _ _) h0
  have h2 : max O.ηOut 0 ≤ max O.ηIn O.ηOut := max_le (le_max_right _ _) h0
  linarith

lemma gap_le (O : Oracle μ S) : O.gap ≤ 1 - O.η := by
  unfold Oracle.gap Oracle.η
  rcases le_total O.ηIn O.ηOut with h | h
  · rw [max_eq_right h]; linarith [le_max_left O.ηOut 0, le_max_right O.ηIn 0]
  · rw [max_eq_left h]; linarith [le_max_left O.ηIn 0, le_max_right O.ηOut 0]

lemma gap_of_flip (O : Oracle μ S) {v p : S} (h : O.flip v p = 1) :
    O.gap = 1 - O.rate (p * v) - O.rate p := by
  have h1 := O.rate_nonneg (p * v)
  have h0 := O.rate_nonneg p
  unfold Oracle.gap
  unfold Oracle.rate at h1 h0 ⊢
  by_cases hx : p * v ∈ O.L <;> by_cases hy : p ∈ O.L
  · simp [Oracle.flip, Oracle.label, hx, hy] at h
    norm_num at h
  · rw [if_pos hx] at h1 ⊢
    rw [if_neg hy] at h0 ⊢
    linarith [max_eq_left h1, max_eq_left h0]
  · rw [if_neg hx] at h1 ⊢
    rw [if_pos hy] at h0 ⊢
    linarith [max_eq_left h1, max_eq_left h0]
  · simp [Oracle.flip, Oracle.label, hx, hy] at h

/-- A member's mean at a rejecting prefix: its own rate, or the other class's read where it
flips. -/
lemma mq_mean_le_flip (O : Oracle μ S) (v p : S) (hp : O.label p = 0) :
    μ[O.mq (p * v)] ≤ O.η + O.gap * O.flip v p := by
  have hlab : O.label (p * v) = O.flip v p := by
    show O.label (p * v) = O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p
    rw [hp]; ring
  rw [mq_mean, hlab]
  have hr1 := O.rate_le_eta (p * v)
  rcases O.flip_bit v p with h | h
  · rw [h]; linarith
  · have hg := gap_of_flip O h
    have hr0 := O.rate_le_eta p
    rw [h, hg]; linarith

lemma mq_mean_ge_flip (O : Oracle μ S) (v p : S) (hp : O.label p = 1) :
    1 - O.η - O.gap * O.flip v p ≤ μ[O.mq (p * v)] := by
  have hlab : O.label (p * v) = 1 - O.flip v p := by
    show O.label (p * v) = 1 - (O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p)
    rw [hp]; ring
  rw [mq_mean, hlab]
  have hr1 := O.rate_le_eta (p * v)
  rcases O.flip_bit v p with h | h
  · rw [h]; linarith
  · have hg := gap_of_flip O h
    have hr0 := O.rate_le_eta p
    rw [h, hg]; linarith

lemma mq_integrable (O : Oracle μ S) (w : S) : Integrable (O.mq w) μ :=
  MeasureTheory.Integrable.of_mem_Icc 0 1 (mq_meas O w).aemeasurable (mq_icc O w)

/-! ## The run space

`runMeasure` is a concrete space with a concrete measure, so the marginals the concentration
arguments consume (`map_firstDraws`, `map_drawBlock`) are lemmas rather than hypotheses.

The streams are i.i.d. and the sampler draws from them without replacement, redrawing a
repeat, so the concentration arguments read the streams and charge what the sampler redrew.  Do
not model the without-replacement law as "i.i.d. conditioned on the block being injective"
instead: those conditioned laws are inconsistent across `n` — for `Dsf = (½,¼,¼)` the first
marginal of the `n = 2` law puts mass `⅖` on the first atom, not `½` — so no space carries them
all. -/

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
variable {crossLimit : ℝ}

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
    (indecisionLimit εcov δ α pAP crossLimit : ℝ) : State :=
  solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit
    (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit)

lemma ladderLen_pos (η₀ : ℝ) (populations : Finset J) (εcov δ α pAP : ℝ) :
    0 < ladderLen η₀ populations indecisionLimit εcov δ α pAP crossLimit := by
  rw [ladderLen]; omega

lemma solvedState_mem_schedule (η₀ : ℝ) (populations : Finset J)
    (εcov δ α pAP : ℝ) :
    solvedState η₀ populations indecisionLimit εcov δ α pAP crossLimit
      ∈ schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit := by
  rw [schedule, solvedState]
  refine Finset.mem_image.2 ⟨0, Finset.mem_range.2 (ladderLen_pos _ _ _ _ _ _), ?_⟩
  norm_num

/-- Every rung draws the same pool: only the prefix count halves along the ladder. -/
lemma nsuff_of_mem_schedule {η₀ : ℝ} {populations : Finset J} {εcov δ α pAP : ℝ} {B : State}
    (hB : B ∈ schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit) :
    B.nsuff = poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit := by
  rw [schedule] at hB
  obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hB
  rfl

lemma prefCount_pos (η₀ : ℝ) (populations : Finset J) (εcov δ α pAP : ℝ) :
    0 < prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit := by
  rw [prefCount]; omega

/-- The ladder's prefix counts halve, so they sum to under twice the top rung's however long
the ladder is.  This is what lets the rungs be paid in proportion to their counts. -/
lemma schedule_npref_sum_le (η₀ : ℝ) (populations : Finset J) (εcov δ α pAP : ℝ) :
    ∑ B ∈ schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit, (B.npref : ℝ)
      ≤ 2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ) := by
  classical
  set m : ℕ := prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit with hmdef
  set L : ℕ := ladderLen η₀ populations indecisionLimit εcov δ α pAP crossLimit with hLdef
  calc ∑ B ∈ schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit, (B.npref : ℝ)
      ≤ ∑ i ∈ Finset.range L,
          ((solvedStateAt η₀ populations indecisionLimit εcov δ pAP crossLimit
            (m / 2 ^ i)).npref : ℝ) := by
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

/-- The rungs at or above `V` double going up, so `V` over their counts sums to under `2`: the
running sum never exceeds twice its last term, which is at most `1`. -/
lemma ladder_inv_sum_le (m V : ℕ) (n : ℕ) :
    ∑ i ∈ Finset.range (n + 1),
        (if V ≤ m / 2 ^ i then (V : ℝ) / ((m / 2 ^ i : ℕ) : ℝ) else 0) ≤ 2
      ∧ (V ≤ m / 2 ^ n → ∑ i ∈ Finset.range (n + 1),
        (if V ≤ m / 2 ^ i then (V : ℝ) / ((m / 2 ^ i : ℕ) : ℝ) else 0)
          ≤ 2 * ((V : ℝ) / ((m / 2 ^ n : ℕ) : ℝ))) := by
  have hle1 : ∀ i, V ≤ m / 2 ^ i → (V : ℝ) / ((m / 2 ^ i : ℕ) : ℝ) ≤ 1 := fun i h =>
    div_le_one_of_le₀ (by exact_mod_cast h) (Nat.cast_nonneg _)
  induction n with
  | zero =>
    simp only [zero_add, Finset.sum_range_one]
    split_ifs with h
    · exact ⟨by linarith [hle1 0 h], fun _ => by
        linarith [div_nonneg (Nat.cast_nonneg (α := ℝ) V)
          (Nat.cast_nonneg (α := ℝ) (m / 2 ^ 0))]⟩
    · exact ⟨by norm_num, fun h' => absurd h' h⟩
  | succ n ih =>
    rw [Finset.sum_range_succ]
    by_cases h : V ≤ m / 2 ^ (n + 1)
    · rw [if_pos h]
      have hhalf : 2 * (m / 2 ^ (n + 1)) ≤ m / 2 ^ n := by
        rw [pow_succ, ← Nat.div_div_eq_div_mul]
        exact Nat.mul_div_le _ _
      have hprev : V ≤ m / 2 ^ n := le_trans h (le_trans (by omega) hhalf)
      have hstep : (V : ℝ) / ((m / 2 ^ n : ℕ) : ℝ)
          ≤ (V : ℝ) / ((m / 2 ^ (n + 1) : ℕ) : ℝ) / 2 := by
        rcases Nat.eq_zero_or_pos V with hV | hV
        · simp [hV]
        have hpos : (0 : ℝ) < ((m / 2 ^ (n + 1) : ℕ) : ℝ) := by
          exact_mod_cast lt_of_lt_of_le hV h
        rw [div_div]
        refine div_le_div_of_nonneg_left (Nat.cast_nonneg _) (by positivity) ?_
        rw [mul_comm]; exact_mod_cast hhalf
      have hlast := ih.2 hprev
      have hone := hle1 (n + 1) h
      exact ⟨by linarith, fun _ => by linarith⟩
    · rw [if_neg h]
      exact ⟨by linarith [ih.1], fun h' => absurd h' h⟩

/-- `ladder_inv_sum_le` over the rungs themselves. -/
lemma schedule_inv_sum_le (η₀ : ℝ) (populations : Finset J) (εcov δ α pAP : ℝ) (V : ℕ)
    (s : Finset State) (hs : s ⊆ schedule η₀ populations indecisionLimit εcov δ α pAP crossLimit) :
    ∑ B ∈ s, (if V ≤ B.npref then (V : ℝ) / (B.npref : ℝ) else 0) ≤ 2 := by
  classical
  have hnn : ∀ B : State, 0 ≤ (if V ≤ B.npref then (V : ℝ) / (B.npref : ℝ) else 0) := by
    intro B; split_ifs <;> positivity
  refine le_trans (Finset.sum_le_sum_of_subset_of_nonneg hs (fun B _ _ => hnn B)) ?_
  rw [schedule]
  refine le_trans (sum_image_le_sum _ hnn _ _) ?_
  have hL := ladderLen_pos (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ α pAP
  obtain ⟨n, hn⟩ : ∃ n, ladderLen η₀ populations indecisionLimit εcov δ α pAP crossLimit = n + 1 :=
    ⟨_, (Nat.succ_pred_eq_of_pos hL).symm⟩
  rw [hn]
  exact (ladder_inv_sum_le _ V n).1

lemma sig_pos (η₀ : ℝ) (hsig : η₀ < 1 / 2) : 0 < sig η₀ := by
  rw [sig]; linarith

lemma cutBudget_pos (η₀ : ℝ) {εcov : ℝ} (hsig : η₀ < 1 / 2) (hε : 0 < εcov)
    (hind : 0 < indecisionLimit) : 0 < cutBudget η₀ indecisionLimit εcov := by
  rw [cutBudget]
  exact lt_min (div_pos hε (by norm_num))
    (lt_min (div_pos (sig_pos η₀ hsig) (by norm_num)) (lt_min (by linarith) (by
      have := gateKappa_pos hsig
      have : 0 < min εcov (1 / 2) := lt_min hε (by norm_num)
      positivity)))

lemma cutBudget_le (η₀ indecisionLimit εcov : ℝ) :
    cutBudget η₀ indecisionLimit εcov ≤ εcov / 8
    ∧ cutBudget η₀ indecisionLimit εcov ≤ sig η₀ / 10
    ∧ cutBudget η₀ indecisionLimit εcov ≤ indecisionLimit / 2 :=
  ⟨min_le_left _ _, (min_le_right _ _).trans (min_le_left _ _),
    (min_le_right _ _).trans ((min_le_right _ _).trans (min_le_left _ _))⟩

lemma cutBudget_le_gate (η₀ indecisionLimit εcov : ℝ) :
    cutBudget η₀ indecisionLimit εcov ≤ 32 * gateKappa η₀ * min εcov (1 / 2) / 65 :=
  (min_le_right _ _).trans ((min_le_right _ _).trans (min_le_right _ _))

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

lemma rejFrac_pos (η₀ : ℝ) {εcov : ℝ} (hsig : η₀ < 1 / 2) (hε : 0 < εcov)
    (hind : 0 < indecisionLimit) : 0 < rejFrac η₀ indecisionLimit εcov := by
  rw [rejFrac]
  have := cutBudget_pos η₀ hsig hε hind
  have := flipFrac_pos η₀ hsig
  positivity

/-- The redraws are a hundred-and-twenty-eighth of the cut budget, and of the flip fraction. -/
lemma rejFrac_le (η₀ : ℝ) {εcov : ℝ} (hsig : η₀ < 1 / 2) (hε : 0 < εcov)
    (hind : 0 < indecisionLimit) :
    rejFrac η₀ indecisionLimit εcov ≤ cutBudget η₀ indecisionLimit εcov / 128
    ∧ rejFrac η₀ indecisionLimit εcov
      = cutBudget η₀ indecisionLimit εcov / 128 * flipFrac η₀ := by
  have hc := cutBudget_pos η₀ hsig hε hind
  have hf := flipFrac_lt_one η₀ hsig
  refine ⟨?_, by rw [rejFrac]; ring⟩
  rw [rejFrac]
  nlinarith [flipFrac_pos η₀ hsig]

/-- A state that redraws a `c` fraction of its `m` draws pays `(m + r)²/(r + 1) ≤ 2m/c` for it
under Markov's inequality, linearly in `m`. -/
lemma rejTerm_le {c : ℝ} (hc0 : 0 < c) (hc : c ≤ 1 / 4) (m : ℕ) :
    ((m + ⌊c * m⌋₊ : ℕ) : ℝ) ^ 2 / ((⌊c * m⌋₊ : ℝ) + 1) ≤ 2 * (m : ℝ) / c := by
  have hm0 : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  have hr1 : ((⌊c * m⌋₊ : ℕ) : ℝ) ≤ c * m := Nat.floor_le (by positivity)
  have hr2 : c * m < ((⌊c * m⌋₊ : ℕ) : ℝ) + 1 := Nat.lt_floor_add_one _
  have hr0 : (0 : ℝ) ≤ ((⌊c * m⌋₊ : ℕ) : ℝ) := Nat.cast_nonneg _
  rw [div_le_div_iff₀ (by positivity) hc0]
  push_cast
  have hsq : ((m : ℝ) + ((⌊c * m⌋₊ : ℕ) : ℝ)) ^ 2 ≤ 2 * (m : ℝ) ^ 2 := by
    nlinarith [mul_le_mul_of_nonneg_left hc hm0]
  nlinarith [mul_le_mul_of_nonneg_right hsq hc0.le, mul_le_mul_of_nonneg_left hr2.le hm0]

/-- A tame population's redraws at a state of `m` prefixes, its table's and its certification
sample's, against `(3P + 4)·m·ρ/c`. -/
lemma redrawCost_le {c : ℝ} (hc0 : 0 < c) (hc : c ≤ 1 / 4) (m : ℕ) {P ρ : ℝ} (hP : 0 ≤ P)
    (hρ : 0 ≤ ρ) :
    ((m + ⌊c * m⌋₊ : ℕ) : ℝ) ^ 2 * ρ / (2 * ((⌊c * m⌋₊ : ℝ) + 1))
      + (((2 * m + ⌊c * m⌋₊ : ℕ) : ℝ) * (P * m * ρ)
        + ((2 * m + ⌊c * m⌋₊ : ℕ) : ℝ) ^ 2 * ρ / 2) / ((⌊c * m⌋₊ : ℝ) + 1)
      ≤ (3 * P + 4) * m * ρ / c := by
  have hm0 : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg _
  set r : ℝ := ((⌊c * m⌋₊ : ℕ) : ℝ) with hrdef
  have hr1 : r ≤ c * m := Nat.floor_le (by positivity)
  have hr2 : c * m < r + 1 := Nat.lt_floor_add_one _
  have hr0 : 0 ≤ r := Nat.cast_nonneg _
  have h1 := rejTerm_le hc0 hc m
  rw [← hrdef] at h1
  have t1 : ((m + ⌊c * m⌋₊ : ℕ) : ℝ) ^ 2 * ρ / (2 * (r + 1)) ≤ m * ρ / c := by
    rw [show ((m + ⌊c * m⌋₊ : ℕ) : ℝ) ^ 2 * ρ / (2 * (r + 1))
        = ρ / 2 * (((m + ⌊c * m⌋₊ : ℕ) : ℝ) ^ 2 / (r + 1)) by
          have : r + 1 ≠ 0 := by linarith
          field_simp]
    calc ρ / 2 * (((m + ⌊c * m⌋₊ : ℕ) : ℝ) ^ 2 / (r + 1)) ≤ ρ / 2 * (2 * (m : ℝ) / c) :=
          mul_le_mul_of_nonneg_left h1 (by positivity)
      _ = m * ρ / c := by ring
  have h2m : ((2 * m + ⌊c * m⌋₊ : ℕ) : ℝ) ≤ 9 / 4 * m := by
    push_cast
    rw [← hrdef]
    nlinarith
  have h2m0 : (0 : ℝ) ≤ ((2 * m + ⌊c * m⌋₊ : ℕ) : ℝ) := Nat.cast_nonneg _
  have hnum : ((2 * m + ⌊c * m⌋₊ : ℕ) : ℝ) * (P * m * ρ)
        + ((2 * m + ⌊c * m⌋₊ : ℕ) : ℝ) ^ 2 * ρ / 2
      ≤ (9 / 4 * P + 81 / 32) * ((m : ℝ) ^ 2 * ρ) := by
    have hPm : 0 ≤ P * m * ρ := by positivity
    have e1 := mul_le_mul_of_nonneg_right h2m hPm
    have e2 : ((2 * m + ⌊c * m⌋₊ : ℕ) : ℝ) ^ 2 ≤ (9 / 4 * m) ^ 2 :=
      pow_le_pow_left₀ h2m0 h2m 2
    have e3 := mul_le_mul_of_nonneg_right e2 (by positivity : (0 : ℝ) ≤ ρ / 2)
    nlinarith
  have t2 : (((2 * m + ⌊c * m⌋₊ : ℕ) : ℝ) * (P * m * ρ)
        + ((2 * m + ⌊c * m⌋₊ : ℕ) : ℝ) ^ 2 * ρ / 2) / (r + 1)
      ≤ (9 / 4 * P + 81 / 32) * (m * ρ / c) := by
    rw [div_le_iff₀ (by linarith)]
    have hk : 0 ≤ (9 / 4 * P + 81 / 32) * (m * ρ / c) := by positivity
    have e : (9 / 4 * P + 81 / 32) * (m * ρ / c) * (c * m)
        = (9 / 4 * P + 81 / 32) * ((m : ℝ) ^ 2 * ρ) := by field_simp
    nlinarith [mul_le_mul_of_nonneg_left hr2.le hk]
  have hmc : 0 ≤ (m : ℝ) * ρ / c := by positivity
  have e : (3 * P + 4) * m * ρ / c = (3 * P + 4) * (m * ρ / c) := by ring
  rw [e]
  nlinarith

/-- The fraction of the family the vote absorbs flipping, at this oracle's own rates.  A flip
moves a member's mean by `gap`, and the worse class's clean mean sits `sig O.η` from the centre:

    gap · flipFracOf O η₀ + voteSlack η₀ = sig O.η

The screen sees each flip at `gap·(1 − 2η)`, so the `gap` cancels between the two
(`flipFracOf_sep`); `flipFrac` is what is left once `gap` is bounded by `1 − η`. -/
noncomputable def flipFracOf (O : Oracle μ S) (η₀ : ℝ) : ℝ := (sig O.η - voteSlack η₀) / O.gap

lemma gap_pos (O : Oracle μ S) (hsig : O.η < 1 / 2) : 0 < O.gap := by
  linarith [gap_ge O]

lemma gap_mul_flipFracOf (O : Oracle μ S) (η₀ : ℝ) (hsig : O.η < 1 / 2) :
    O.gap * flipFracOf O η₀ = sig O.η - voteSlack η₀ := by
  rw [flipFracOf]
  field_simp [(gap_pos O hsig).ne']

lemma flipFrac_le_flipFracOf (O : Oracle μ S) {η₀ : ℝ} (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) :
    flipFrac η₀ ≤ flipFracOf O η₀ := by
  have hg := gap_pos O (lt_of_le_of_lt hηle hη₀)
  have hgle := gap_le O
  rw [flipFrac, flipFracOf, voteSlack, div_le_div_iff₀ (by linarith) hg]
  simp only [sig]
  have hs : 0 ≤ 1 / 2 - η₀ := by linarith
  nlinarith [mul_le_mul_of_nonneg_left hgle (by linarith : (0 : ℝ) ≤ 7 * (1 / 2 - η₀)),
    mul_nonneg (sub_nonneg.2 hηle) (by linarith : (0 : ℝ) ≤ 5 + 3 * (1 / 2 - η₀))]

/-- What a `flipFracOf` fraction flipping shows the screen, against what `screenMargin` was
solved at. -/
lemma flipFracOf_sep (O : Oracle μ S) {η₀ : ℝ} (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) :
    flipFrac η₀ * ((1 - η₀) * (1 - 2 * η₀))
      ≤ flipFracOf O η₀ * (O.gap * (1 - 2 * O.η)) := by
  have hsplit := flipFrac_voteSlack η₀ hη₀
  have hg := gap_mul_flipFracOf O η₀ (lt_of_le_of_lt hηle hη₀)
  rw [show flipFracOf O η₀ * (O.gap * (1 - 2 * O.η))
      = (O.gap * flipFracOf O η₀) * (1 - 2 * O.η) by ring, hg,
    show flipFrac η₀ * ((1 - η₀) * (1 - 2 * η₀)) = ((1 - η₀) * flipFrac η₀) * (1 - 2 * η₀) by ring]
  have hvs : voteSlack η₀ = 3 * (1 / 2 - η₀) / 10 := by rw [voteSlack, sig]
  simp only [sig] at hsplit ⊢
  exact mul_le_mul (by linarith) (by linarith) (by linarith) (by linarith)

lemma flipFracOf_pos (O : Oracle μ S) {η₀ : ℝ} (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) :
    0 < flipFracOf O η₀ :=
  lt_of_lt_of_le (flipFrac_pos η₀ hη₀) (flipFrac_le_flipFracOf O hηle hη₀)

lemma famCount_pos (η₀ : ℝ) (populations : Finset J) (εcov δ : ℝ) :
    0 < famCount η₀ populations indecisionLimit εcov δ crossLimit := by
  rw [famCount]; omega

/-- The family's size is even, so its centre is a count. -/
lemma famCount_half (η₀ : ℝ) (populations : Finset J) (εcov δ : ℝ) :
    (⌈(famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) / 2⌉₊ : ℝ)
      = (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) / 2 := by
  obtain ⟨n, hn⟩ : ∃ n : ℕ, famCount η₀ populations indecisionLimit εcov δ crossLimit = 2 * n :=
    ⟨_, rfl⟩
  rw [hn, Nat.cast_mul, Nat.cast_ofNat, mul_div_cancel_left₀ _ two_ne_zero, Nat.ceil_natCast]

/-- To put `exp (-a)` under `ε` it is enough that `a` clears `log (1/ε)`. -/
lemma exp_neg_le_of_log_le {a ε : ℝ} (hε : 0 < ε) (h : Real.log (1 / ε) ≤ a) :
    Real.exp (-a) ≤ ε := by
  have h1 : Real.exp (-a) ≤ Real.exp (-Real.log (1 / ε)) := Real.exp_le_exp.2 (by linarith)
  have h2 : Real.exp (-Real.log (1 / ε)) = ε := by
    rw [← Real.log_inv, one_div, inv_inv, Real.exp_log hε]
  linarith [h1, h2.le, h2.ge]

lemma famCount_tail (η₀ : ℝ) (populations : Finset J) {εcov δ : ℝ} (hsig : η₀ < 1 / 2)
    (hε : 0 < εcov) (hind : 0 < indecisionLimit) :
    Real.exp (-2 * (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ)
        * (voteSlack η₀ / 2) ^ 2)
      ≤ cutBudget η₀ indecisionLimit εcov / 2 := by
  have hs := voteSlack_pos η₀ hsig
  have hcut := cutBudget_pos η₀ hsig hε hind
  have hhalf : (⌈Real.log (2 / cutBudget η₀ indecisionLimit εcov) / voteSlack η₀ ^ 2⌉₊ : ℝ)
      ≤ (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) / 2 := by
    rw [famCount]; push_cast
    linarith [Nat.cast_nonneg (α := ℝ) ⌈Real.log (1 / crossLimit) / voteSlack η₀ ^ 2⌉₊,
      Nat.cast_nonneg (α := ℝ) ⌈1 / voteSlack η₀⌉₊]
  have h4 := le_trans (Nat.le_ceil (Real.log (2 / cutBudget η₀ indecisionLimit εcov)
    / voteSlack η₀ ^ 2)) hhalf
  rw [div_le_iff₀ (by positivity : (0 : ℝ) < voteSlack η₀ ^ 2)] at h4
  rw [show -2 * (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ)
        * (voteSlack η₀ / 2) ^ 2
      = -((famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) / 2
        * voteSlack η₀ ^ 2) by ring]
  refine exp_neg_le_of_log_le (by positivity) ?_
  rw [one_div_div]
  linarith

/-- The seed's term in `famCount`: the family outnumbers `1/voteSlack`. -/
lemma famCount_seed (η₀ : ℝ) (populations : Finset J) (εcov δ : ℝ) (hsig : η₀ < 1 / 2) :
    2 ≤ (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) * voteSlack η₀ := by
  have hs := voteSlack_pos η₀ hsig
  have hle : 1 / voteSlack η₀
      ≤ (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) / 2 := by
    refine le_trans (Nat.le_ceil _) ?_
    rw [famCount]; push_cast
    linarith [Nat.cast_nonneg (α := ℝ)
      ⌈Real.log (2 / cutBudget η₀ indecisionLimit εcov) / voteSlack η₀ ^ 2⌉₊,
      Nat.cast_nonneg (α := ℝ) ⌈Real.log (1 / crossLimit) / voteSlack η₀ ^ 2⌉₊]
  rw [div_le_iff₀ hs] at hle
  linarith

lemma decSlack_nonneg (η₀ : ℝ) (populations : Finset J) (εcov δ : ℝ) (hsig : η₀ < 1 / 2) :
    0 ≤ decSlack η₀ populations indecisionLimit εcov δ crossLimit := by
  have h := famCount_seed (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hsig
  have hs := voteSlack_pos η₀ hsig
  have hk : (0 : ℝ) < (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) + 1 := by
    positivity
  rw [decSlack, sub_nonneg, div_le_iff₀ (by positivity)]
  nlinarith

lemma decSlack_le (η₀ : ℝ) (populations : Finset J) (εcov δ : ℝ) :
    decSlack η₀ populations indecisionLimit εcov δ crossLimit ≤ voteSlack η₀ / 2 := by
  rw [decSlack]
  have : (0 : ℝ) ≤ 1 / (2 * ((famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) + 1))
    := by positivity
  linarith

/-- `famCount_tail` at the returned family's slack: the seed's term pays for the half read. -/
lemma famCount_tail_seed (η₀ : ℝ) (populations : Finset J) {εcov δ : ℝ} (hsig : η₀ < 1 / 2)
    (hε : 0 < εcov) (hind : 0 < indecisionLimit) :
    Real.exp (-2 * (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ)
        * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2)
      ≤ cutBudget η₀ indecisionLimit εcov / 2 := by
  have hs := voteSlack_pos η₀ hsig
  have hcut := cutBudget_pos η₀ hsig hε hind
  set κ : ℝ := (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) with hκdef
  set v : ℝ := voteSlack η₀ with hvdef
  have hκ0 : 0 ≤ κ := Nat.cast_nonneg _
  have hA : Real.log (2 / cutBudget η₀ indecisionLimit εcov) / v ^ 2 + 1 / v ≤ κ / 2 := by
    have h1 := Nat.le_ceil (Real.log (2 / cutBudget η₀ indecisionLimit εcov) / v ^ 2)
    have h2 := Nat.le_ceil (1 / v)
    have hf : κ / 2 = (⌈Real.log (2 / cutBudget η₀ indecisionLimit εcov) / v ^ 2⌉₊ : ℝ)
        + (⌈Real.log (1 / crossLimit) / v ^ 2⌉₊ : ℝ) + (⌈1 / v⌉₊ : ℝ) + 1 := by
      rw [hκdef, famCount]; push_cast; ring
    linarith [Nat.cast_nonneg (α := ℝ) ⌈Real.log (1 / crossLimit) / v ^ 2⌉₊]
  have hA' : Real.log (2 / cutBudget η₀ indecisionLimit εcov) + v ≤ κ * v ^ 2 / 2 := by
    have e1 : (Real.log (2 / cutBudget η₀ indecisionLimit εcov) / v ^ 2 + 1 / v) * v ^ 2
        = Real.log (2 / cutBudget η₀ indecisionLimit εcov) + v := by
      field_simp
    have := mul_le_mul_of_nonneg_right hA (by positivity : (0 : ℝ) ≤ v ^ 2)
    rw [e1] at this
    linarith
  have hd : decSlack η₀ populations indecisionLimit εcov δ crossLimit = v / 2 - 1 / (2 * (κ + 1)) := by
    rw [decSlack]
  have hk1 : (0 : ℝ) < κ + 1 := by linarith
  -- `2κ(v/2 − 1/(2(κ+1)))² ≥ κv²/2 − v`
  have hq : κ * v ^ 2 / 2 - v
      ≤ 2 * κ * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2 := by
    rw [hd]
    have hr : κ / (κ + 1) ≤ 1 := by rw [div_le_one hk1]; linarith
    have hr0 : 0 ≤ κ / (κ + 1) := div_nonneg hκ0 hk1.le
    have e : 2 * κ * (v / 2 - 1 / (2 * (κ + 1))) ^ 2
        = κ * v ^ 2 / 2 - v * (κ / (κ + 1)) + κ / (κ + 1) / (2 * (κ + 1)) := by
      field_simp
      ring
    rw [e]
    have : 0 ≤ κ / (κ + 1) / (2 * (κ + 1)) := by positivity
    nlinarith [mul_le_mul_of_nonneg_left hr hs.le]
  rw [show -2 * κ * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2
      = -(2 * κ * decSlack η₀ populations indecisionLimit εcov δ crossLimit ^ 2) by ring]
  refine exp_neg_le_of_log_le (by positivity) ?_
  rw [one_div_div]
  linarith

/-- The crossing term in `famCount`, in the form the band's width is checked against. -/
lemma famCount_cross (η₀ : ℝ) (populations : Finset J) (εcov δ : ℝ) (hsig : η₀ < 1 / 2) :
    2 * Real.log (1 / crossLimit)
      ≤ (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) * voteSlack η₀ ^ 2 := by
  have hs := voteSlack_pos η₀ hsig
  have hle : Real.log (1 / crossLimit) / voteSlack η₀ ^ 2
      ≤ (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) / 2 := by
    refine le_trans (Nat.le_ceil _) ?_
    rw [famCount]; push_cast
    linarith [Nat.cast_nonneg (α := ℝ)
      ⌈Real.log (2 / cutBudget η₀ indecisionLimit εcov) / voteSlack η₀ ^ 2⌉₊,
      Nat.cast_nonneg (α := ℝ) ⌈1 / voteSlack η₀⌉₊]
  rw [div_le_iff₀ (by positivity : (0 : ℝ) < voteSlack η₀ ^ 2)] at hle
  linarith

lemma bandHalf_le (η₀ : ℝ) (populations : Finset J) (εcov δ : ℝ) (hsig : η₀ < 1 / 2) :
    (bandHalf η₀ populations indecisionLimit εcov δ crossLimit : ℝ)
      ≤ (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) * voteSlack η₀ / 2 :=
  Nat.floor_le (by have := voteSlack_pos η₀ hsig; positivity)

/-- The band spans at least `κ·voteSlack/2` counts. -/
lemma bandHalf_wide (η₀ : ℝ) (populations : Finset J) (εcov δ : ℝ) :
    (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) * voteSlack η₀ / 2
      ≤ 2 * (bandHalf η₀ populations indecisionLimit εcov δ crossLimit : ℝ) + 1 := by
  have h := Nat.lt_floor_add_one
    ((famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) * voteSlack η₀ / 2)
  rw [← bandHalf] at h
  linarith [Nat.cast_nonneg (α := ℝ) (bandHalf η₀ populations indecisionLimit εcov δ crossLimit)]

/-- The band fits below the centre, so `lo` is not truncated. -/
lemma bandHalf_lt (η₀ : ℝ) (populations : Finset J) (εcov δ : ℝ) (hη0 : 0 ≤ η₀)
    (hsig : η₀ < 1 / 2) :
    bandHalf η₀ populations indecisionLimit εcov δ crossLimit + 1
      ≤ ⌈(famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) / 2⌉₊ := by
  have hκ : (0 : ℝ) < (famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) := by
    exact_mod_cast famCount_pos η₀ populations εcov δ
  have hv : voteSlack η₀ ≤ 3 / 20 := by rw [voteSlack, sig]; linarith
  have hb := bandHalf_le (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ hsig
  have hc := famCount_half (indecisionLimit := indecisionLimit) (crossLimit := crossLimit)
    η₀ populations εcov δ
  have hlt : (bandHalf η₀ populations indecisionLimit εcov δ crossLimit : ℝ)
      < (⌈(famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) / 2⌉₊ : ℝ) := by
    rw [hc]; nlinarith
  have : bandHalf η₀ populations indecisionLimit εcov δ crossLimit
      < ⌈(famCount η₀ populations indecisionLimit εcov δ crossLimit : ℝ) / 2⌉₊ := by
    exact_mod_cast hlt
  omega

lemma flipBudget_pos (η₀ : ℝ) (populations : Finset J) {εcov δ : ℝ}
    (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) :
    0 < flipBudget η₀ populations indecisionLimit εcov δ := by
  rw [flipBudget]
  exact div_pos
    (mul_pos (mul_pos (by norm_num) (cutBudget_pos η₀ hsig hε hind)) (flipFrac_pos η₀ hsig))
    (mul_pos (by norm_num) hcard)

lemma screenMargin_pos (η₀ : ℝ) (populations : Finset J) {εcov δ : ℝ}
    (hsig : η₀ < 1 / 2) (hε : 0 < εcov) (hind : 0 < indecisionLimit)
    (hcard : (0 : ℝ) < (populations.card : ℝ)) :
    0 < screenMargin η₀ populations indecisionLimit εcov δ := by
  rw [screenMargin]
  have h1 := flipBudget_pos η₀ populations (δ := δ) hsig hε hind hcard
  have h2 := sig_pos η₀ hsig
  have h3 : (0 : ℝ) < 1 - η₀ := by linarith
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
against a real-valued threshold — the gate's `lo` and `hi` —
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

The gate's sides are `ω`-dependent; they are determined by the reads at `p · v`, and those are
independent of the reads at `p`, so conditioning on the votes fixes the sides without
disturbing the law of the seed column.  Without dropping `ε` the sides would be partly
determined by the very bit the gate counts, and no conditioning would separate them. -/
lemma mul_ne_self (p v : S) (hv : v ≠ 1) : p * v ≠ p := fun h =>
  hv (mul_left_cancel (a := p) (by rw [h, mul_one]))

/-! ### Bounding an event whose index set is chosen elsewhere

A tail bound holds of a *fixed* cut, but the cut the gate scores is `ω`-dependent.
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
lemma flat_eq_one {Pre Suf : Set S} (hflat : Flat Pre Suf) {p p' : S} (hp : p ∈ Pre)
    (hp' : p' ∈ Pre) {v : S} (hv : v ∈ insert 1 Suf) (h : p * v = p') : v = 1 := by
  have hpp : p = p' := hflat p hp p' hp' v hv 1 (Set.mem_insert _ _) (by rw [h, mul_one])
  exact mul_left_cancel (a := p) (by rw [h, hpp, mul_one])

/-- On a flat alphabet the gate's query string is never one of the split's. -/
lemma flat_ne_of_ne_one {Pre Suf : Set S} (hflat : Flat Pre Suf) {p p' : S} (hp : p ∈ Pre)
    (hp' : p' ∈ Pre) {v : S} (hvS : v ∈ insert 1 Suf) (hv : v ≠ 1) : p * v ≠ p' :=
  fun h => hv (flat_eq_one hflat hp hp' hvS h)

lemma clusterAt_subset (O : Oracle μ S) (populations : Finset J) (B : State)
    (x : Run Ω S J) : clusterBy rule O.mq populations x B ⊆ poolAt B.nsuff x :=
  fun v hv => screenedAt_subset O populations B x
    (rule.subset _ _ _ B.k (one_mem_screenedAt O populations B x) (Finset.mem_of_mem_erase hv))

/-- The family with its seed. -/
noncomputable def clusterFull (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J)
    (x : Run Ω S J) (B : State) : Finset S :=
  rule.pick (fun w => O.mq w (oracleNoise x) = 1) (prefixesAt populations B.npref x)
    (screenedAt O.mq populations B x) B.k

lemma clusterAt_card_add_one (O : Oracle μ S) (populations : Finset J) (B : State)
    (x : Run Ω S J) :
    (clusterBy rule O.mq populations x B).card + 1 = (clusterFull rule O populations x B).card :=
  Finset.card_erase_add_one (rule.seed_mem _ _ _ B.k (one_mem_screenedAt O populations B x))

open scoped Classical in
/-- The seed adds its own read of `p` to the vote. -/
lemma voteCount_insert_one (mq : S → Ω → ℝ) (F : Finset S) (hF : (1 : S) ∉ F) (p : S) (ω : Ω) :
    voteCount mq (insert 1 F) p ω = voteCount mq F p ω + if mq p ω = 1 then 1 else 0 := by
  classical
  unfold voteCount
  rw [Finset.filter_insert, mul_one]
  split_ifs with h
  · rw [Finset.card_insert_of_notMem (fun hm => hF (Finset.mem_filter.1 hm).1)]
  · rfl

/-- Wherever the family with its seed decides, a cut one read wider on the accept side, the
family without it decides the same way. -/
lemma cutCorrect_insert_one (O : Oracle μ S) (lo hi : ℕ) (F : Finset S) (hF : (1 : S) ∉ F)
    (p : S) (ω : Ω) (h : cutCorrect O lo hi F p ω) : cutCorrect O lo (hi + 1) (insert 1 F) p ω := by
  classical
  have hv := voteCount_insert_one O.mq F hF p ω
  have hs : (if O.mq p ω = 1 then 1 else 0) ≤ 1 := by split_ifs <;> omega
  refine ⟨fun hacc => h.1 (by omega), fun hrej => h.2 (by omega)⟩

/-- Where the family with its seed decides, the family without it decides too. -/
lemma decided_of_insert_one (mq : S → Ω → ℝ) (lo hi : ℕ) (F : Finset S) (hF : (1 : S) ∉ F)
    (p : S) (ω : Ω) (h : decided mq lo (hi + 1) (insert 1 F) p ω) : decided mq lo hi F p ω := by
  classical
  have hv := voteCount_insert_one mq F hF p ω
  have hs : (if mq p ω = 1 then 1 else 0) ≤ 1 := by split_ifs <;> omega
  rcases h with h | h
  · exact Or.inl (by omega)
  · exact Or.inr (by omega)

/-- The family is decided by the bits on `readSet` — the screen's reads and the
clustering's alike. -/
lemma clusterAt_congr (O : Oracle μ S) (populations : Finset J) (B : State)
    (d : ((ℕ → S) × (J → ℕ → S)) × (J → ℕ → S)) {ω ω' : Ω}
    (h : ∀ w ∈ readSet (prefixesAt populations B.npref ((ω, d) : Run Ω S J))
      (poolAt B.nsuff ((ω, d) : Run Ω S J)), O.noise w ω = O.noise w ω') :
    clusterBy rule O.mq populations (ω, d) B = clusterBy rule O.mq populations (ω', d) B := by
  classical
  have hbit : ∀ w ∈ readSet (prefixesAt populations B.npref ((ω, d) : Run Ω S J))
      (poolAt B.nsuff ((ω, d) : Run Ω S J)), (O.mq w ω = 1 ↔ O.mq w ω' = 1) :=
    fun w hw => by rw [mq_congr O (h w hw)]
  have hcands : screenedAt O.mq populations B (ω, d) = screenedAt O.mq populations B (ω', d) :=
    screenedAt_congr O populations B d hbit
  unfold clusterBy
  rw [show oracleNoise ((ω, d) : Run Ω S J) = ω from rfl, show oracleNoise ((ω', d) : Run Ω S J) = ω' from rfl,
    ← hcands]
  refine congrArg (fun t : Finset S => t.erase 1) ?_
  exact rule.congr _ _ _ _ _ (one_mem_screenedAt O populations B (ω, d))
    (fun p hp v hv => hbit _ (mem_readSet hp (screenedAt_subset O populations B (ω, d) hv)))

/-- A population prefix the table does not hold is read nowhere by the clustering.  Its
query strings `p · v` collide with the clustering's `q · v'` only if `p = q`. -/
lemma disjoint_image_readSet {Pre Suf : Set S} (hflat : Flat Pre Suf) {P cands : Finset S}
    {p : S} (hP : ∀ q ∈ P, q ∈ Pre) (hV : ∀ v ∈ cands, v ∈ insert 1 Suf) (hp : p ∈ Pre)
    (hpP : p ∉ P) :
    Disjoint (↑(cands.image (fun v => p * v)) : Set S) (↑(readSet P cands) : Set S) := by
  classical
  rw [Finset.disjoint_coe, Finset.disjoint_left]
  rintro z hz hmem
  obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hz
  obtain ⟨⟨q, v'⟩, hqv, hq⟩ := Finset.mem_image.1 hmem
  obtain ⟨hqP, hv'⟩ := Finset.mem_product.1 hqv
  exact hpP (hflat p hp q (hP q hqP) v (hV v hv) v' (hV v' hv') hq.symm ▸ hqP)

/-! ### Independence of the score from the side

Both events are structurally measurable over disjoint sub-families of the oracle's bits, so
this needs no factorisation theorem: the cut is decided by the bits at `p · v` over the
family, the score reads the bits at the certification prefixes themselves, and `Flat`
separates them. -/

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


/-! ### Collisions between draws -/

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

lemma stoppable_npref_sum_le (η₀ : ℝ) (populations : Finset J)
    (εcov δ α pAP ρ : ℝ) :
    ∑ B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit ρ, (B.npref : ℝ)
      ≤ 2 * (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ) := by
  classical
  refine le_trans (Finset.sum_le_sum_of_subset_of_nonneg ?_ (fun _ _ _ => Nat.cast_nonneg _))
    (schedule_npref_sum_le η₀ populations εcov δ α pAP)
  rw [stoppable]
  exact Finset.filter_subset _ _

lemma capped_of_mem_stoppable {η₀ : ℝ} {populations : Finset J}
    {εcov δ α pAP ρ : ℝ} {B : State}
    (hB : B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit ρ) :
    Capped η₀ populations indecisionLimit εcov δ ρ pAP
      (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit)
      (validCount η₀ populations indecisionLimit εcov δ pAP crossLimit) B := by
  classical
  exact (Finset.mem_filter.1 hB).2

open scoped Classical in
/-- The screen's disagreement with the seed's column at one prefix, as a centre of `{ε}` reads it. -/
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

/-- Two noisy reads are compared, so an accept-preserving candidate disagrees at `2r(1−r)` for
the prefix's rate `r`, and one that flips at `p` at `(1 − r − r')(1 − 2r) = gap·(1 − 2r)` more,
`r'` the other class's rate. -/
lemma seedLoss_mean (O : Oracle μ S) {cn cd : ℕ} (hcd : cn < cd) (v p : S) (hv : p * v ≠ p)
    (hsig : O.η ≤ 1 / 2) :
    2 * O.rate p * (1 - O.rate p) + O.flip v p * (O.gap * (1 - 2 * O.η))
        ≤ μ[seedLoss O cn cd v p]
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
  have hgap0 : 0 ≤ O.gap := by linarith [gap_ge O]
  have key : O.flip v p = 1 →
      O.gap * (1 - 2 * O.η) ≤ (1 - O.rate (p * v) - O.rate p) * (1 - 2 * O.rate p) := by
    intro h1
    rw [← gap_of_flip O h1]
    exact mul_le_mul_of_nonneg_left (by linarith) hgap0
  rw [hsplit, mq_mean, mq_mean, hflip]
  rcases O.label_bit (p * v) with hx | hx <;> rcases O.label_bit p with hy | hy
  · rw [O.rate_eq_of_label_eq (hx.trans hy.symm), hx, hy]
    exact ⟨le_of_eq (by ring), fun _ => by ring⟩
  · have k := key (by rw [hflip, hx, hy]; norm_num)
    rw [hx, hy]
    exact ⟨by nlinarith [k], fun h => by norm_num at h⟩
  · have k := key (by rw [hflip, hx, hy]; norm_num)
    rw [hx, hy]
    exact ⟨by nlinarith [k], fun h => by norm_num at h⟩
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
lemma seedLoss_indep {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S) (cn cd : ℕ)
    {P : Finset S} (hP : ∀ p ∈ P, p ∈ Pre) (v : S) (hvS : v ∈ insert 1 Suf) :
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
      · exact flat_ne_of_ne_one hflat (hP _ b.property) (hP _ a.property) hvS hv1 hb.symm
    · by_cases hv1 : v = 1
      · exact hab (Subtype.ext (by rw [← hb, hv1, mul_one]))
      · exact flat_ne_of_ne_one hflat (hP _ a.property) (hP _ b.property) hvS hv1 hb
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

/-! ### The screen's statistic

`seedLoss` is the disagreement with the seed's own column, whose mean separates a candidate
that never flips from one that flips on a `Δ` fraction by at least `Δ·gap·(1−2η)`.  The
clustering is *not* what bounds the family's flip mass: `clusterAt_flip_bound` reads that off
the screen, which every candidate has already passed, whatever the `Clusterer` picks.
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
theorem screenCount_upper {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S) {cn cd : ℕ}
    (hcd : cn < cd) {P : Finset S} (hP : ∀ p ∈ P, p ∈ Pre) (v : S) (hvS : v ∈ insert 1 Suf)
    (hv : v ≠ 1) (γ : ℝ) (hγ : 0 ≤ γ) (hsig : O.η ≤ 1 / 2)
    (hclean : ∀ p ∈ P, O.flip v p = 0) :
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
    (seedLoss_indep hflat O cn cd hP v hvS) (fun i => seedLoss_icc O cn cd v i.val) hmean hγ
  simp only [Finset.card_univ, Fintype.card_coe] at htail
  refine le_trans (measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)) htail
  change _ ≤ ∑ i : {p // p ∈ P}, seedLoss O cn cd v i.val ω
  rw [screenCount_eq_attach O hcd P v ω]
  exact hω

open scoped Classical in
/-- Any candidate's disagreement count sits at least `cleanLoss` in mean — the flip term is
nonnegative — so it falls `γ` per prefix below that only in the lower tail.  This is what puts
a floor under `screenBase`. -/
theorem screenCount_lower {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S) {cn cd : ℕ}
    (hcd : cn < cd) {P : Finset S} (hP : ∀ p ∈ P, p ∈ Pre) (v : S) (hvS : v ∈ insert 1 Suf)
    (hv : v ≠ 1) (Δ γ : ℝ) (hγ : 0 ≤ γ) (hsig : O.η ≤ 1 / 2)
    (hflip : Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p) :
    μ.real {ω | (screenCount O.mq P v ω : ℝ)
        ≤ cleanLoss O P + (P.card : ℝ) * (Δ * (O.gap * (1 - 2 * O.η)) - γ)}
      ≤ Real.exp (-2 * (P.card : ℝ) * γ ^ 2) := by
  classical
  have hmean : cleanLoss O P + (P.card : ℝ) * (Δ * (O.gap * (1 - 2 * O.η)))
      ≤ ∑ i ∈ (Finset.univ : Finset {p // p ∈ P}), μ[seedLoss O cn cd v i.val] := by
    have hle := Finset.sum_le_sum (fun (i : {p // p ∈ P}) (_ : i ∈ Finset.univ) =>
      (seedLoss_mean O hcd v i.val (mul_ne_self i.val v hv) hsig).1)
    rw [Finset.sum_add_distrib, ← Finset.sum_mul,
      Finset.sum_coe_sort P (fun p => 2 * O.rate p * (1 - O.rate p)),
      Finset.sum_coe_sort P (fun p => O.flip v p)] at hle
    rw [cleanLoss]
    nlinarith [mul_le_mul_of_nonneg_right hflip
      (mul_nonneg (by linarith [gap_ge O] : (0 : ℝ) ≤ O.gap) (by linarith : (0 : ℝ) ≤ 1 - 2 * O.η))]
  have htail := sumLower_le_total (fun (i : {p // p ∈ P}) => seedLoss O cn cd v i.val)
    Finset.univ _ γ (fun i => (seedLoss_meas O cn cd v i.val).aemeasurable)
    (seedLoss_indep hflat O cn cd hP v hvS) (fun i => seedLoss_icc O cn cd v i.val) hmean hγ
  simp only [Finset.card_univ, Fintype.card_coe] at htail
  refine le_trans (measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)) htail
  change ∑ i : {p // p ∈ P}, seedLoss O cn cd v i.val ω ≤ _
  rw [screenCount_eq_attach O hcd P v ω]
  have h : (screenCount O.mq P v ω : ℝ)
      ≤ cleanLoss O P + (P.card : ℝ) * (Δ * (O.gap * (1 - 2 * O.η)) - γ) := hω
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
    ∀ w ∈ clusterBy rule O.mq populations x B,
      ¬ (Δ * ((prefixesAt populations B.npref x).card : ℝ)
        ≤ ∑ p ∈ prefixesAt populations B.npref x, O.flip w p) :=
  fun w hw => hscreen w (rule.subset _ _ _ B.k
    (one_mem_screenedAt O populations B x) (Finset.mem_of_mem_erase hw))

/-! ### From flip mass to a correct cut

The band is what turns "few members flip" into "the cut is right".  A rejecting prefix is
accepted only when the vote clears the centre, and members preserving at `p` contribute only
through noise, so with a fraction `f` flipping the vote's mean is at most `η + gap·f`.  The
vote stays decisive with a flipping `flipFracOf` of the family, and Markov over the members'
flip masses charges the heavy mass at `Δ/flipFracOf`. -/

open scoped Classical in
/-- The number of the family's members that flip at a prefix, as a real. -/
noncomputable def flipCount (O : Oracle μ S) (F : Finset S) (p : S) : ℝ :=
  ((F.filter (fun v => O.flip v p = 1)).card : ℝ)

/-- The seed never flips. -/
lemma flipCount_insert_one (O : Oracle μ S) (F : Finset S) (p : S) :
    flipCount O (insert 1 F) p ≤ flipCount O F p := by
  classical
  unfold flipCount
  rw [Finset.filter_insert]
  have h0 : O.flip 1 p ≠ 1 := by
    simp only [Oracle.flip, mul_one]
    rcases O.label_bit p with h | h <;> rw [h] <;> norm_num
  rw [if_neg h0]

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
flipping one `gap` higher, so a flip fraction of `f` lifts the vote's mean only to
`η + gap·f`. -/
theorem voteSum_upper (O : Oracle μ S) (F : Finset S) (p : S) (hp : O.label p = 0)
    (f γ : ℝ) (hsig : O.η ≤ 1 / 2) (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ) :
    μ.real {ω | (F.card : ℝ) * ((O.η + O.gap * f) + γ) ≤ voteSum O F p ω}
      ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  have hper : ∀ v ∈ F, μ[O.mq (p * v)] ≤ O.η + O.gap * O.flip v p :=
    fun v _ => mq_mean_le_flip O v p hp
  have hmean : ∑ v ∈ F, μ[O.mq (p * v)] ≤ (F.card : ℝ) * (O.η + O.gap * f) := by
    refine le_trans (Finset.sum_le_sum hper) ?_
    rw [Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum,
      ← flipCount_eq_sum]
    have h1 : (0 : ℝ) ≤ O.gap := by linarith [gap_ge O]
    nlinarith [mul_le_mul_of_nonneg_left hf h1]
  exact sumUpper_le (fun v : S => O.mq (p * v)) F (O.η + O.gap * f) γ
    (fun v => (mq_meas O _).aemeasurable) (mq_indep_shift O p) (fun v => mq_icc O _) hmean hγ

/-- Lower tail on an accepting prefix.  Mirror of `voteSum_upper`. -/
theorem voteSum_lower (O : Oracle μ S) (F : Finset S) (p : S) (hp : O.label p = 1)
    (f γ : ℝ) (hsig : O.η ≤ 1 / 2) (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ) :
    μ.real {ω | voteSum O F p ω ≤ (F.card : ℝ) * ((1 - O.η - O.gap * f) - γ)}
      ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  have hper : ∀ v ∈ F, 1 - O.η - O.gap * O.flip v p ≤ μ[O.mq (p * v)] :=
    fun v _ => mq_mean_ge_flip O v p hp
  have hmean : (F.card : ℝ) * (1 - O.η - O.gap * f) ≤ ∑ v ∈ F, μ[O.mq (p * v)] := by
    refine le_trans ?_ (Finset.sum_le_sum hper)
    rw [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum,
      ← flipCount_eq_sum]
    have h1 : (0 : ℝ) ≤ O.gap := by linarith [gap_ge O]
    nlinarith [mul_le_mul_of_nonneg_left hf h1]
  exact sumLower_le (fun v : S => O.mq (p * v)) F (1 - O.η - O.gap * f) γ
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

lemma measureReal_mq_eq_one (O : Oracle μ S) (w : S) :
    μ.real {ω | O.mq w ω = 1} = μ[O.mq w] := by
  have hmeas : MeasurableSet {ω | O.mq w ω = 1} :=
    measurableSet_eq_fun (mq_meas O w) measurable_const
  have hae : O.mq w =ᵐ[μ] ({ω | O.mq w ω = 1} : Set Ω).indicator 1 := by
    filter_upwards [mq_bit O w] with ω hω
    rcases hω with h | h
    · simp [h]
    · simp [h]
  rw [integral_congr_ae hae, integral_indicator_one hmeas]

/-- A vote whose mean lies above `hi` falls to `lo` or below, and one whose mean lies at or
below `lo` rises above `hi`, each with probability at most `exp (-2 (hi − lo)² / n)` for a
family of at most `n`. -/
theorem vote_cross_le (O : Oracle μ S) (F : Finset S) (p : S) {lo hi n : ℕ} (hlohi : lo ≤ hi)
    (hn : F.card ≤ n) (hn0 : 0 < n) :
    (hi < meanVote O F p →
      μ.real {ω | voteCount O.mq F p ω ≤ lo} ≤ Real.exp (-2 * ((hi : ℝ) - lo) ^ 2 / n))
    ∧ (meanVote O F p ≤ lo →
      μ.real {ω | hi < voteCount O.mq F p ω} ≤ Real.exp (-2 * ((hi : ℝ) - lo) ^ 2 / n)) := by
  classical
  have hmean : meanVote O F p = ∑ v ∈ F, μ[O.mq (p * v)] :=
    Finset.sum_congr rfl (fun v _ => measureReal_mq_eq_one O (p * v))
  have hd : (0 : ℝ) ≤ (hi : ℝ) - lo := by
    have : (lo : ℝ) ≤ hi := by exact_mod_cast hlohi
    linarith
  rcases Nat.eq_zero_or_pos F.card with h0 | hpos
  · obtain rfl : F = ∅ := Finset.card_eq_zero.1 h0
    refine ⟨fun h => ?_, fun _ => ?_⟩
    · rw [meanVote, Finset.sum_empty] at h
      exact absurd h (not_lt.2 (Nat.cast_nonneg _))
    · have : {ω | hi < voteCount O.mq ∅ p ω} = ∅ := by
        ext ω; simp [voteCount]
      rw [this, measureReal_empty]
      exact (Real.exp_pos _).le
  have hc : (0 : ℝ) < F.card := by exact_mod_cast hpos
  have hc0 : (F.card : ℝ) ≠ 0 := hc.ne'
  have hcn : (F.card : ℝ) ≤ n := by exact_mod_cast hn
  -- a deviation of `d` over the family's own size costs at least `hi − lo` over `n`
  have hexp : ∀ d : ℝ, (hi : ℝ) - lo ≤ d →
      Real.exp (-2 * (F.card : ℝ) * (d / F.card) ^ 2)
        ≤ Real.exp (-2 * ((hi : ℝ) - lo) ^ 2 / n) := by
    intro d hdd
    refine Real.exp_le_exp.2 ?_
    have e : -2 * (F.card : ℝ) * (d / F.card) ^ 2 = -2 * (d ^ 2 / F.card) := by
      field_simp
    have e' : -2 * ((hi : ℝ) - lo) ^ 2 / n = -2 * (((hi : ℝ) - lo) ^ 2 / n) := by ring
    have h1 : ((hi : ℝ) - lo) ^ 2 / n ≤ d ^ 2 / F.card :=
      div_le_div₀ (sq_nonneg _) (pow_le_pow_left₀ hd hdd 2) hc hcn
    rw [e, e']
    linarith
  constructor
  · intro hm
    set γ : ℝ := (meanVote O F p - lo) / F.card with hγ
    have hγ0 : 0 ≤ γ := div_nonneg (by linarith) hc.le
    have hT := sumLower_le_total (fun v : S => O.mq (p * v)) F (meanVote O F p) γ
      (fun v => (mq_meas O _).aemeasurable) (mq_indep_shift O p) (fun v => mq_icc O _)
      hmean.le hγ0
    have hcg : meanVote O F p - (F.card : ℝ) * γ = lo := by
      rw [hγ]; field_simp; ring
    rw [hcg] at hT
    refine le_trans (measureReal_le_of_ae_imp ?_) (hT.trans (hexp _ (by linarith)))
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq h
    have h' : (voteCount O.mq F p ω : ℝ) ≤ lo := by exact_mod_cast h
    rw [heq] at h'
    exact h'
  · intro hm
    set γ : ℝ := ((hi : ℝ) - meanVote O F p) / F.card with hγ
    have hγ0 : 0 ≤ γ := div_nonneg (by linarith) hc.le
    have hT := sumUpper_le_total (fun v : S => O.mq (p * v)) F (meanVote O F p) γ
      (fun v => (mq_meas O _).aemeasurable) (mq_indep_shift O p) (fun v => mq_icc O _)
      hmean.ge hγ0
    have hcg : meanVote O F p + (F.card : ℝ) * γ = hi := by
      rw [hγ]; field_simp; ring
    rw [hcg] at hT
    refine le_trans (measureReal_le_of_ae_imp ?_) (hT.trans (hexp _ (by linarith)))
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq h
    have h' : (hi : ℝ) < (voteCount O.mq F p ω : ℝ) := by exact_mod_cast h
    rw [heq] at h'
    exact h'.le

/-- The cut is correct at a prefix the family barely flips.  Only the side the prefix
actually sits on can fail, so one tail — not two — pays for it. -/
theorem cutCorrect_whp (O : Oracle μ S) (F : Finset S) (p : S) (lo hi : ℕ) (f γ : ℝ)
    (hsig : O.η ≤ 1 / 2)
    (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ)
    (hhi : (F.card : ℝ) * ((O.η + O.gap * f) + γ) ≤ (hi : ℝ) + 1)
    (hlo : (lo : ℝ) ≤ (F.card : ℝ) * ((1 - O.η - O.gap * f) - γ)) :
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
    have : (hi : ℝ) + 1 ≤ (voteCount O.mq F p ω : ℝ) := by exact_mod_cast hgt
    show (F.card : ℝ) * ((O.η + O.gap * f) + γ) ≤ voteSum O F p ω
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
    show voteSum O F p ω ≤ (F.card : ℝ) * ((1 - O.η - O.gap * f) - γ)
    rw [← heq]; linarith

/-- `cutCorrect_whp` recentred, relaying `voteSum_upper_gap`/`voteSum_lower_gap`. -/
theorem cutCorrect_whp_gap (O : Oracle μ S) (F : Finset S) (p : S) (lo hi : ℕ) (f γ : ℝ)
    (hgap0 : 0 ≤ O.hgap)
    (hf : flipCount O F p ≤ (F.card : ℝ) * f) (hγ : 0 ≤ γ)
    (hhi : (F.card : ℝ) * (((O.mid - O.hgap) + 2 * O.hgap * f) + γ) ≤ (hi : ℝ) + 1)
    (hlo : (lo : ℝ) ≤ (F.card : ℝ) * (((O.mid + O.hgap) - 2 * O.hgap * f) - γ)) :
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
    have : (hi : ℝ) + 1 ≤ (voteCount O.mq F p ω : ℝ) := by exact_mod_cast hgt
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
is what lifts them.  The `a.e.` is what lets the table and pool be assumed inside the flat
pair. -/
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

/-- One suffix coordinate has the suffix law. -/
lemma map_sfxCoord (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (i : ℕ) :
    Measure.map (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J → ℕ → S)) => d.1.1 i)
        (drawMeasure D Dsf) = Dsf := by
  have hstep : (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J → ℕ → S)) => d.1.1 i)
      = (fun p : ℕ → S => p i) ∘ (Prod.fst ∘ Prod.fst) := rfl
  rw [hstep, ← Measure.map_map (by fun_prop) (by fun_prop),
    ← Measure.map_map measurable_fst measurable_fst, drawMeasure, Measure.map_fst_prod]
  simp only [measure_univ, one_smul]
  rw [Measure.map_fst_prod]
  simp only [measure_univ, one_smul]
  rw [(measurePreserving_eval_infinitePi (fun _ : ℕ => Dsf) i).map_eq]

lemma ae_sfx_mem_Suf (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (Suf : Set S)
    (hsuppSf : Dsf Sufᶜ = 0) :
    ∀ᵐ d ∂(drawMeasure D Dsf), ∀ i : ℕ, d.1.1 i ∈ Suf := by
  have hmeasSuf : MeasurableSet (Sufᶜ : Set S) := (Set.to_countable _).measurableSet
  refine ae_all_iff.2 (fun i => ae_iff.2 ?_)
  rw [show {d : ((((ℕ → S) × (J → ℕ → S))) × (J → ℕ → S)) | ¬ d.1.1 i ∈ Suf}
      = (fun d : ((((ℕ → S) × (J → ℕ → S))) × (J → ℕ → S)) => d.1.1 i) ⁻¹' Sufᶜ from rfl,
    ← Measure.map_apply (by fun_prop) hmeasSuf, map_sfxCoord D Dsf i]
  exact hsuppSf

lemma pool_mem_Suf {Suf : Set S} (M : ℕ) (s : ℕ → S) (hs : ∀ i, s i ∈ Suf) :
    ∀ v ∈ insert 1 (freshDraws ∅ s M), v ∈ insert 1 Suf := by
  intro v hv
  rcases Finset.mem_insert.1 hv with rfl | hv
  · exact Set.mem_insert _ _
  · obtain ⟨i, rfl⟩ := mem_freshDraws hv
    exact Set.mem_insert_of_mem _ (hs i)

/-- The table's prefixes are drawn, so they lie wherever every draw does. -/
lemma table_mem_Pre {Pre : Set S} (populations : Finset J) (m : ℕ) (q : J → ℕ → S)
    (hq : ∀ j ∈ populations, ∀ i, q j i ∈ Pre) :
    ∀ p ∈ populations.biUnion (fun j => freshDraws ∅ (fun i => q j i) m), p ∈ Pre := by
  intro p hp
  obtain ⟨j, hj, hp'⟩ := Finset.mem_biUnion.1 hp
  obtain ⟨i, rfl⟩ := mem_freshDraws hp'
  exact hq j hj i

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

/-- Any event about a stream's first `n` draws is measurable: they take countably many
values. -/
lemma measurableSet_firstDraws (n : ℕ) (P : (Fin n → S) → Prop) :
    MeasurableSet {s : ℕ → S | P (fun i : Fin n => s i.val)} := by
  have hmeas : Measurable (fun (s : ℕ → S) (i : Fin n) => s i.val) := by fun_prop
  exact hmeas (Set.to_countable {t : Fin n → S | P t}).measurableSet

open scoped Classical in
lemma image_range_eq_image_univ {α : Type*} [DecidableEq α] (m : ℕ) (f : ℕ → α) :
    (Finset.range m).image f = (Finset.univ : Finset (Fin m)).image (fun i => f i.val) := by
  classical
  ext a
  simp only [Finset.mem_image, Finset.mem_range, Finset.mem_univ, true_and]
  exact ⟨fun ⟨i, hi, h⟩ => ⟨⟨i, hi⟩, h⟩, fun ⟨i, h⟩ => ⟨i.val, i.isLt, h⟩⟩

open scoped Classical in
lemma measurableSet_acceptedAt (T : Finset S) (n : ℕ) (Q : Finset S → Prop) :
    MeasurableSet {s : ℕ → S | Q (acceptedAt T s n)} := by
  classical
  have hrw : {s : ℕ → S | Q (acceptedAt T s n)}
      = {s : ℕ → S | (fun t : Fin n → S => Q ((Finset.univ : Finset (Fin n)).image t \ T))
          (fun i : Fin n => s i.val)} := by
    ext s
    simp only [Set.mem_setOf_eq, acceptedAt, image_range_eq_image_univ n s]
  rw [hrw]
  exact measurableSet_firstDraws (S := S) n
    (fun t : Fin n → S => Q ((Finset.univ : Finset (Fin n)).image t \ T))

/-- What the sampler returns is measurable in the stream: it stops at the first count that
offers `m`, and each count reads finitely many draws. -/
lemma measurableSet_freshDraws (T : Finset S) (m : ℕ) (F : Finset S) :
    MeasurableSet {s : ℕ → S | freshDraws T s m = F} := by
  classical
  have hcov : {s : ℕ → S | freshDraws T s m = F}
      = (⋃ n : ℕ, ({s : ℕ → S | m ≤ (acceptedAt T s n).card ∧ acceptedAt T s n = F}
          ∩ ⋂ k : Fin n, {s : ℕ → S | (acceptedAt T s k.val).card < m}))
        ∪ ((⋂ n : ℕ, {s : ℕ → S | (acceptedAt T s n).card < m}) ∩ {_s | F = ∅}) := by
    ext s
    simp only [Set.mem_setOf_eq, Set.mem_union, Set.mem_iUnion, Set.mem_inter_iff,
      Set.mem_iInter]
    by_cases h : ∃ n, m ≤ (acceptedAt T s n).card
    · rw [freshDraws_eq T s m h]
      constructor
      · intro hF
        exact Or.inl ⟨Nat.find h, ⟨Nat.find_spec h, hF⟩,
          fun k => not_le.1 (Nat.find_min h k.isLt)⟩
      · rintro (⟨n, ⟨hn, hF⟩, hmin⟩ | ⟨hall, -⟩)
        · have hle : Nat.find h ≤ n := Nat.find_min' h hn
          have hge : n ≤ Nat.find h := by
            by_contra hlt
            push_neg at hlt
            exact absurd (Nat.find_spec h) (not_le.2 (hmin ⟨Nat.find h, hlt⟩))
          rw [le_antisymm hle hge]
          exact hF
        · exact absurd (Nat.find_spec h) (not_le.2 (hall _))
    · rw [freshDraws_of_not T s m h]
      push_neg at h
      constructor
      · intro hF
        exact Or.inr ⟨h, hF.symm⟩
      · rintro (⟨n, ⟨hn, -⟩, -⟩ | ⟨-, hF⟩)
        · exact absurd hn (not_le.2 (h n))
        · exact hF.symm
  rw [hcov]
  refine MeasurableSet.union (MeasurableSet.iUnion (fun n => MeasurableSet.inter ?_ ?_))
    (MeasurableSet.inter (MeasurableSet.iInter (fun n => ?_)) ?_)
  · exact measurableSet_acceptedAt T n (fun A => m ≤ A.card ∧ A = F)
  · exact MeasurableSet.iInter (fun k => measurableSet_acceptedAt T k.val (fun A => A.card < m))
  · exact measurableSet_acceptedAt T n (fun A => A.card < m)
  · by_cases hF : F = ∅
    · simp [hF]
    · simp [hF]

lemma measurable_prefixStream (j : J) :
    Measurable (fun x : Run Ω S J => (fun i => prefixDraw j i x)) :=
  measurable_pi_lambda _ (fun i => measurable_prf j i)

lemma measurable_certStream (j : J) :
    Measurable (fun x : Run Ω S J => (fun i => certPrefix j i x)) :=
  measurable_pi_lambda _ (fun i => measurable_cert j i)

lemma measurable_suffixStream :
    Measurable (fun x : Run Ω S J => (fun i => suffixDraw i x)) :=
  measurable_pi_lambda _ (fun i => measurable_sfx i)

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

/-- A drawn pool costs its size.  The pool is read off the suffix stream and the event it
indexes off the prefix streams, so once the suffix stream is fixed the union over the pool is a
union over at most `M` fixed strings. -/
theorem runMeasure_pool_selection_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (C : S → Set (J → ℕ → S)) (hC : ∀ v, MeasurableSet (C v)) (M : ℕ) (E : ℝ≥0∞)
    (hbad : ∀ v, (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) (C v) ≤ E) :
    runMeasure μ D Dsf {x : Run Ω S J | ∃ v ∈ freshDraws ∅ (fun i => suffixDraw i x) M,
        prefixStreams x ∈ C v} ≤ M * E := by
  classical
  set νs : Measure (ℕ → S) := Measure.infinitePi fun _ : ℕ => Dsf with hνs
  set νq : Measure (J → ℕ → S) := Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j
    with hνq
  have hmeasSet : MeasurableSet
      {y : (ℕ → S) × (J → ℕ → S) | ∃ v ∈ freshDraws ∅ y.1 M, y.2 ∈ C v} := by
    have hcov : {y : (ℕ → S) × (J → ℕ → S) | ∃ v ∈ freshDraws ∅ y.1 M, y.2 ∈ C v}
        = ⋃ F : Finset S, ({y : (ℕ → S) × (J → ℕ → S) | freshDraws ∅ y.1 M = F}
            ∩ ⋃ v ∈ F, {y | y.2 ∈ C v}) := by
      ext y
      simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, exists_prop]
      exact ⟨fun ⟨v, hv, h⟩ => ⟨_, rfl, v, hv, h⟩, fun ⟨F, hF, v, hv, h⟩ => ⟨v, hF ▸ hv, h⟩⟩
    rw [hcov]
    exact MeasurableSet.iUnion (fun F => (measurable_fst (measurableSet_freshDraws ∅ M F)).inter
      (Finset.measurableSet_biUnion _ (fun v _ => measurable_snd (hC v))))
  have hpre : {x : Run Ω S J | ∃ v ∈ freshDraws ∅ (fun i => suffixDraw i x) M,
        prefixStreams x ∈ C v}
      = (draws : Run Ω S J → _) ⁻¹'
        {y : (ℕ → S) × (J → ℕ → S) | ∃ v ∈ freshDraws ∅ y.1 M, y.2 ∈ C v} := rfl
  rw [hpre, ← Measure.map_apply (by fun_prop) hmeasSet, map_drawBlock D Dsf,
    Measure.prod_apply hmeasSet]
  calc ∫⁻ a, νq (Prod.mk a ⁻¹'
        {y : (ℕ → S) × (J → ℕ → S) | ∃ v ∈ freshDraws ∅ y.1 M, y.2 ∈ C v}) ∂νs
      = ∫⁻ a, νq (⋃ v ∈ freshDraws ∅ a M, C v) ∂νs := by
        congr 1
        funext a
        congr 1
        ext q
        simp
    _ ≤ ∫⁻ _, (M : ℝ≥0∞) * E ∂νs := by
        refine lintegral_mono (fun a => ?_)
        refine le_trans (measure_biUnion_finset_le _ _) ?_
        calc ∑ v ∈ freshDraws ∅ a M, νq (C v) ≤ ∑ _v ∈ freshDraws ∅ a M, E :=
              Finset.sum_le_sum (fun v _ => hbad v)
          _ = ((freshDraws ∅ a M).card : ℝ≥0∞) * E := by rw [Finset.sum_const, nsmul_eq_mul]
          _ ≤ (M : ℝ≥0∞) * E := by
              gcongr
              exact_mod_cast card_freshDraws_le ∅ a M
    _ = (M : ℝ≥0∞) * E := by simp

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
      ⊆ {x : Run Ω S J | ∃ v ∈ freshDraws ∅ (fun i => suffixDraw i x) M,
          prefixStreams x ∈ understated O D j m Δ g v} := by
    intro x hx
    simp only [Set.mem_setOf_eq, not_forall] at hx
    obtain ⟨v, hv, hcount, hmass⟩ := hx
    rcases Finset.mem_insert.1 hv with rfl | hv'
    · exact absurd (hseed ▸ hΔ) hmass
    · exact ⟨v, hv', not_le.1 hmass, hcount⟩
  have hbound : runMeasure μ D Dsf {x : Run Ω S J | ¬ ∀ v ∈ poolAt M x,
      (∑ i : Fin m, O.flip v (prefixDraw j i.val x) ≤ (m : ℝ) * (Δ - g)) → flipMass O (D j) v ≤ Δ}
      ≤ (M : ℝ≥0∞) * E :=
    le_trans (measure_mono hsub) (runMeasure_pool_selection_le D Dsf _
      (measurableSet_understated O D j m Δ g) M E
      (fun v => measure_understated_le (μ := μ) D Dsf O j m Δ g hg v))
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

open scoped Classical in
lemma measurableSet_prefixesAt (populations : Finset J) (m : ℕ) (Q : Finset S) :
    MeasurableSet {x : Run Ω S J | prefixesAt populations m x = Q} := by
  classical
  have hcov : {x : Run Ω S J | prefixesAt populations m x = Q}
      = ⋃ t : {t : J → Finset S // populations.biUnion t = Q},
          ⋂ j ∈ populations, {x : Run Ω S J | prefixesOf j m x = t.val j} := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_iInter]
    refine ⟨fun h => ⟨⟨fun j => prefixesOf j m x, h⟩, fun j _ => rfl⟩, ?_⟩
    rintro ⟨⟨t, ht⟩, hx⟩
    rw [← ht]
    exact Finset.biUnion_congr rfl (fun j hj => hx j hj)
  rw [hcov]
  exact MeasurableSet.iUnion (fun t => Finset.measurableSet_biInter _ (fun j _ =>
    measurable_prefixStream j (measurableSet_freshDraws ∅ m (t.val j))))

open scoped Classical in
lemma measurableSet_poolAt (M : ℕ) (C : Finset S) :
    MeasurableSet {x : Run Ω S J | poolAt M x = C} := by
  classical
  have hcov : {x : Run Ω S J | poolAt M x = C}
      = ⋃ F : {F : Finset S // insert 1 F = C},
          {x : Run Ω S J | freshDraws ∅ (fun i => suffixDraw i x) M = F.val} := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion]
    refine ⟨fun h => ⟨⟨freshDraws ∅ (fun i => suffixDraw i x) M, h⟩, rfl⟩, ?_⟩
    rintro ⟨⟨F, hF⟩, hx⟩
    rw [poolAt, hx, hF]
  rw [hcov]
  exact MeasurableSet.iUnion (fun F =>
    measurable_suffixStream (measurableSet_freshDraws ∅ M F.val))

open scoped Classical in
lemma measurableSet_certOf (populations : Finset J) (m : ℕ) (j : J) (m' : ℕ) (A : Finset S) :
    MeasurableSet {x : Run Ω S J | certOf (prefixesAt populations m x) j m' x = A} := by
  classical
  have hcov : {x : Run Ω S J | certOf (prefixesAt populations m x) j m' x = A}
      = ⋃ P : Finset S, ({x : Run Ω S J | prefixesAt populations m x = P}
          ∩ {x : Run Ω S J | freshDraws P (fun i => certPrefix j i x) m' = A}) := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff]
    refine ⟨fun h => ⟨prefixesAt populations m x, rfl, h⟩, ?_⟩
    rintro ⟨P, hP, hx⟩
    rw [certOf, hP]
    exact hx
  rw [hcov]
  exact MeasurableSet.iUnion (fun P => (measurableSet_prefixesAt populations m P).inter
    (measurable_certStream j (measurableSet_freshDraws P m' A)))

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

open scoped Classical in
/-- The vote is decisive at a prefix the family barely flips.  As in `cutCorrect_whp` a flip
fraction of `f` moves the mean by up to `gap·f`, so both thresholds shift by that. -/
theorem decided_whp (O : Oracle μ S) (F : Finset S) (p : S) (lo hi : ℕ) (f γ : ℝ) (hγ : 0 ≤ γ)
    (hsig : O.η ≤ 1 / 2)
    (hf : flipCount O F p ≤ (F.card : ℝ) * f)
    (hhi : (hi : ℝ) ≤ (F.card : ℝ) * ((1 - O.η - O.gap * f) - γ))
    (hlo : (F.card : ℝ) * ((O.η + O.gap * f) + γ) ≤ (lo : ℝ) + 1) :
    μ.real {ω | ¬ decided O.mq lo hi F p ω} ≤ Real.exp (-2 * (F.card : ℝ) * γ ^ 2) := by
  classical
  rcases O.label_bit p with hp | hp
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_upper O F p hp f γ hsig hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hgt : lo < voteCount O.mq F p ω := by
      by_contra hc
      exact hbad (Or.inr (not_lt.1 hc))
    have hcast : (lo : ℝ) + 1 ≤ (voteCount O.mq F p ω : ℝ) := by exact_mod_cast hgt
    show (F.card : ℝ) * ((O.η + O.gap * f) + γ) ≤ voteSum O F p ω
    rw [← heq]
    linarith
  · refine le_trans (measureReal_le_of_ae_imp ?_) (voteSum_lower O F p hp f γ hsig hf hγ)
    filter_upwards [voteCount_eq_voteSum O F p] with ω heq hbad
    have hle : voteCount O.mq F p ω ≤ hi := by
      by_contra hc
      exact hbad (Or.inl (not_le.1 hc))
    have hcast : (voteCount O.mq F p ω : ℝ) ≤ (hi : ℝ) := by exact_mod_cast hle
    show voteSum O F p ω ≤ (F.card : ℝ) * ((1 - O.η - O.gap * f) - γ)
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

The clustering scores a candidate on the table, which holds every population's sample. -/

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
        Finset.sum_le_sum (fun j _ => card_freshDraws_le _ _ _)
    _ = populations.card * m := by rw [Finset.sum_const, smul_eq_mul]

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

/-! ### What the sampler redraws

The concentration arguments read the streams' first draws, repeats and all, and the sampler
keeps only fresh values.  What they need of it is that it finds its `m` within the first
`m + r` draws: then the sample is `m` strings among those draws, and a count over it is within
`r` of the count over the draws (`freshDraws_of_good`).  A redraw is a pair of equal draws, or a
certification draw the table holds, and each of those has chance at most the collision mass,
so Markov's inequality prices `r + 1` of them linearly in the draws. -/

/-- Markov's inequality for a count of events in two families. -/
lemma measureReal_count_ge_le {α ι κ : Type*} [MeasurableSpace α] (ν : Measure α)
    [IsFiniteMeasure ν] (s : Finset ι) (A : ι → Set α) (hA : ∀ i, MeasurableSet (A i))
    (t : Finset κ) (B : κ → Set α) (hB : ∀ k, MeasurableSet (B k)) {r : ℝ} (hr : 0 < r)
    [∀ x, DecidablePred (fun i => x ∈ A i)] [∀ x, DecidablePred (fun k => x ∈ B k)] :
    ν.real {x | r ≤ ((s.filter (fun i => x ∈ A i)).card : ℝ)
        + ((t.filter (fun k => x ∈ B k)).card : ℝ)}
      ≤ (∑ i ∈ s, ν.real (A i) + ∑ k ∈ t, ν.real (B k)) / r := by
  classical
  set f : α → ℝ := fun x => ∑ i ∈ s, (A i).indicator 1 x + ∑ k ∈ t, (B k).indicator 1 x
    with hf
  have hcountA : ∀ x, ∑ i ∈ s, (A i).indicator (1 : α → ℝ) x
      = ((s.filter (fun i => x ∈ A i)).card : ℝ) := by
    intro x
    rw [Finset.card_filter, Nat.cast_sum]
    refine Finset.sum_congr rfl (fun i _ => ?_)
    by_cases h : x ∈ A i <;> simp [h]
  have hcountB : ∀ x, ∑ k ∈ t, (B k).indicator (1 : α → ℝ) x
      = ((t.filter (fun k => x ∈ B k)).card : ℝ) := by
    intro x
    rw [Finset.card_filter, Nat.cast_sum]
    refine Finset.sum_congr rfl (fun k _ => ?_)
    by_cases h : x ∈ B k <;> simp [h]
  have hfx : ∀ x, f x = ((s.filter (fun i => x ∈ A i)).card : ℝ)
      + ((t.filter (fun k => x ∈ B k)).card : ℝ) := by
    intro x
    rw [hf]
    simp only
    rw [hcountA x, hcountB x]
  have hintA : ∀ i ∈ s, Integrable ((A i).indicator (1 : α → ℝ)) ν :=
    fun i _ => (integrable_const (1 : ℝ)).indicator (hA i)
  have hintB : ∀ k ∈ t, Integrable ((B k).indicator (1 : α → ℝ)) ν :=
    fun k _ => (integrable_const (1 : ℝ)).indicator (hB k)
  have hint : Integrable f ν :=
    (integrable_finsetSum s hintA).add (integrable_finsetSum t hintB)
  have hmarkov := mul_meas_ge_le_integral_of_nonneg (μ := ν) (f := f)
    (Filter.Eventually.of_forall (fun x => by
      simp only [Pi.zero_apply]
      rw [hfx x]
      positivity)) hint r
  have hintf : ∫ x, f x ∂ν = ∑ i ∈ s, ν.real (A i) + ∑ k ∈ t, ν.real (B k) := by
    rw [hf, integral_add (integrable_finsetSum s hintA) (integrable_finsetSum t hintB),
      integral_finsetSum s hintA, integral_finsetSum t hintB]
    congr 1
    · exact Finset.sum_congr rfl (fun i _ => integral_indicator_one (hA i))
    · exact Finset.sum_congr rfl (fun k _ => integral_indicator_one (hB k))
  have hset : {x | r ≤ ((s.filter (fun i => x ∈ A i)).card : ℝ)
      + ((t.filter (fun k => x ∈ B k)).card : ℝ)} = {x | r ≤ f x} := by
    ext x
    simp only [Set.mem_setOf_eq, hfx x]
  rw [hset, le_div_iff₀ hr, mul_comm, ← hintf]
  exact hmarkov

/-- Two draws of an i.i.d. block coincide with the collision mass. -/
lemma block_pair_eq {β : Type*} [MeasurableSpace β] (ν : Measure β) (Dj : Measure S)
    [IsProbabilityMeasure Dj] {n : ℕ} (Y : β → Fin n → S) (hY : Measurable Y)
    (hmap : Measure.map Y ν = Measure.pi fun _ : Fin n => Dj) {a b : Fin n} (hab : a ≠ b) :
    ν.real {x | Y x a = Y x b} = collisionMass Dj := by
  have hset : MeasurableSet {p : Fin n → S | p a = p b} :=
    measurableSet_eq_fun (measurable_pi_apply _) (measurable_pi_apply _)
  have hpre : {x | Y x a = Y x b} = Y ⁻¹' {p : Fin n → S | p a = p b} := rfl
  rw [hpre, measureReal_def, ← Measure.map_apply hY hset, hmap, ← measureReal_def]
  exact pi_coord_eq Dj hab

open scoped Classical in
/-- `r + 1` repeated pairs among an i.i.d. block's `n` draws have chance at most
`n²ρ / (2(r + 1))`. -/
lemma measureReal_repeats_le {β : Type*} [MeasurableSpace β] (ν : Measure β)
    [IsProbabilityMeasure ν] (Dj : Measure S) [IsProbabilityMeasure Dj] (y : ℕ → β → S)
    (hy : ∀ i, Measurable (y i)) (n : ℕ)
    (hmap : Measure.map (fun x => fun i : Fin n => y i.val x) ν = Measure.pi fun _ : Fin n => Dj)
    (ρ : ℝ) (hρ : collisionMass Dj ≤ ρ) (hρ0 : 0 ≤ ρ) (r : ℕ) :
    ν.real {x | ((r : ℝ) + 1) ≤ ((repeatPairs (fun i => y i x) n).card : ℝ)}
      ≤ (n : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1)) := by
  classical
  set I : Finset (ℕ × ℕ) :=
    ((Finset.range n) ×ˢ (Finset.range n)).filter (fun q => q.1 < q.2) with hI
  have hrep : ∀ x, repeatPairs (fun i => y i x) n = I.filter (fun q => x ∈ {x | y q.1 x = y q.2 x}) := by
    intro x
    ext q
    simp only [repeatPairs, hI, Finset.mem_filter, Set.mem_setOf_eq, and_assoc]
  have hIcard : 2 * (I.card : ℝ) ≤ (n : ℝ) ^ 2 := by
    have h := card_repeatPairs_le (fun _ : ℕ => (1 : S)) n
    have hI' : repeatPairs (fun _ : ℕ => (1 : S)) n = I := by
      ext q
      simp [repeatPairs, hI]
    rw [hI'] at h
    have : ((2 * I.card : ℕ) : ℝ) ≤ ((n * n : ℕ) : ℝ) := by exact_mod_cast h
    push_cast at this
    nlinarith
  have hpair : ∀ q ∈ I, ν.real {x | y q.1 x = y q.2 x} ≤ ρ := by
    intro q hq
    simp only [hI, Finset.mem_filter, Finset.mem_product, Finset.mem_range] at hq
    have hY : Measurable (fun x => fun i : Fin n => y i.val x) :=
      measurable_pi_lambda _ (fun i => hy i.val)
    have h := block_pair_eq ν Dj (fun x => fun i : Fin n => y i.val x) hY hmap
      (a := ⟨q.1, hq.1.1⟩) (b := ⟨q.2, hq.1.2⟩)
      (fun h => absurd (congrArg Fin.val h) (ne_of_lt hq.2))
    simp only at h
    rw [h]
    exact hρ
  have hmk := measureReal_count_ge_le ν I (fun q => {x | y q.1 x = y q.2 x})
    (fun q => measurableSet_eq_fun (hy q.1) (hy q.2)) (∅ : Finset Unit) (fun _ => ∅)
    (fun _ => MeasurableSet.empty) (r := (r : ℝ) + 1) (by positivity)
  simp only [Finset.filter_empty, Finset.card_empty, Nat.cast_zero, add_zero,
    Finset.sum_empty] at hmk
  have hset : {x | ((r : ℝ) + 1) ≤ ((repeatPairs (fun i => y i x) n).card : ℝ)}
      = {x | ((r : ℝ) + 1) ≤ ((I.filter (fun q => x ∈ {x | y q.1 x = y q.2 x})).card : ℝ)} := by
    ext x
    rw [Set.mem_setOf_eq, Set.mem_setOf_eq, hrep x]
  rw [hset]
  refine le_trans hmk ?_
  have hsum : ∑ q ∈ I, ν.real {x | y q.1 x = y q.2 x} ≤ (I.card : ℝ) * ρ := by
    calc ∑ q ∈ I, ν.real {x | y q.1 x = y q.2 x} ≤ ∑ _q ∈ I, ρ := Finset.sum_le_sum hpair
      _ = (I.card : ℝ) * ρ := by rw [Finset.sum_const, nsmul_eq_mul]
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith [mul_le_mul_of_nonneg_right hIcard hρ0]

/-- Population `j`'s table finds its `m` prefixes within its first `m + r` draws. -/
def tabGood (j : J) (m r : ℕ) (x : Run Ω S J) : Prop :=
  m ≤ (acceptedAt ∅ (fun i => prefixDraw j i x) (m + r)).card

/-- Population `j`'s certification sample finds its `m` prefixes within its first `m + r`
draws. -/
def certGood (populations : Finset J) (j : J) (m r : ℕ) (x : Run Ω S J) : Prop :=
  m ≤ (acceptedAt (prefixesAt populations m x) (fun i => certPrefix j i x) (m + r)).card

/-- The pool finds its `M` suffixes within its first `M + r` draws. -/
def poolGood (M r : ℕ) (x : Run Ω S J) : Prop :=
  M ≤ (acceptedAt ∅ (fun i => suffixDraw i x) (M + r)).card

open scoped Classical in
/-- A sampler that falls short redraws `r + 1` times, and drawing nothing from `T` each redraw
is a repeated pair. -/
lemma repeats_of_short (y : ℕ → S) (m r : ℕ)
    (h : ¬ m ≤ (acceptedAt ∅ y (m + r)).card) :
    ((r : ℝ) + 1) ≤ ((repeatPairs y (m + r)).card : ℝ) := by
  have h1 := le_card_acceptedAt_add ∅ y (m + r)
  have h2 : ((Finset.range (m + r)).filter (fun i => y i ∈ (∅ : Finset S))).card = 0 := by
    simp
  rw [h2] at h1
  have : r + 1 ≤ (repeatPairs y (m + r)).card := by omega
  exact_mod_cast this

theorem measureReal_tabBad_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (j : J) (m r : ℕ) (ρ : ℝ)
    (hρ : collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runMeasure μ D Dsf).real {x : Run Ω S J | ¬ tabGood j m r x}
      ≤ ((m + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1)) :=
  le_trans (measureReal_mono (fun x hx => repeats_of_short _ m r hx) (measure_ne_top _ _))
    (measureReal_repeats_le (runMeasure μ D Dsf) (D j) (fun i x => prefixDraw j i x)
      (fun i => measurable_prf j i) (m + r) (map_prefixBlock D Dsf j (m + r)) ρ hρ hρ0 r)

theorem measureReal_poolBad_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (M r : ℕ) (ρ : ℝ)
    (hρ : collisionMass Dsf ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runMeasure μ D Dsf).real {x : Run Ω S J | ¬ poolGood M r x}
      ≤ ((M + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1)) :=
  le_trans (measureReal_mono (fun x hx => repeats_of_short _ M r hx) (measure_ne_top _ _))
    (measureReal_repeats_le (runMeasure μ D Dsf) Dsf (fun i x => suffixDraw i x)
      (fun i => measurable_sfx i) (M + r) (map_suffixBlock D Dsf (M + r)) ρ hρ hρ0 r)

/-- A table draw and a certification draw are independent, so they coincide with chance at
most the larger collision mass. -/
lemma prefix_cert_eq_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (j' j : J) (a b : ℕ) (ρ : ℝ)
    (hρ' : collisionMass (D j') ≤ ρ) (hρ : collisionMass (D j) ≤ ρ) :
    (runMeasure μ D Dsf).real {x : Run Ω S J | certPrefix j a x = prefixDraw j' b x} ≤ ρ := by
  have hdiag : MeasurableSet {q : S × S | q.1 = q.2} :=
    measurableSet_eq_fun measurable_fst measurable_snd
  have hpre : {x : Run Ω S J | certPrefix j a x = prefixDraw j' b x}
      = (fun x : Run Ω S J => (prefixDraw j' b x, certPrefix j a x)) ⁻¹'
        {q : S × S | q.1 = q.2} := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_preimage]
    exact eq_comm
  rw [hpre, measureReal_def,
    Measure.map_apply ((measurable_prf j' b).prodMk (measurable_cert j a)) hdiag
      |>.symm.trans (congrArg (fun ν : Measure (S × S) => ν _)
        (map_prefCertPairRun D Dsf j' j b a)),
    ← measureReal_def]
  exact cross_collision_le (D j') (D j) ρ hρ' hρ
    (summable_singleton_sq (D j')) (summable_singleton_sq (D j))

open scoped Classical in
set_option maxHeartbeats 1000000 in
/-- With every table found in time, a certification sample that is not redraws `r + 1` times
at repeats or at table prefixes, which there are `(m + r)²(|populations| + ½)` chances of. -/
theorem measureReal_certBad_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (populations : Finset J)
    (j : J) (m r : ℕ) (ρ : ℝ) (hρ : ∀ j' ∈ populations, collisionMass (D j') ≤ ρ)
    (hρj : collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runMeasure μ D Dsf).real {x : Run Ω S J |
        (∀ j' ∈ populations, tabGood j' m r x) ∧ ¬ certGood populations j m r x}
      ≤ ((m + r : ℕ) : ℝ) ^ 2 * ((populations.card : ℝ) + 1 / 2) * ρ / ((r : ℝ) + 1) := by
  classical
  set n : ℕ := m + r with hn
  set K : Finset (ℕ × J × ℕ) := Finset.range n ×ˢ (populations ×ˢ Finset.range n) with hK
  set I : Finset (ℕ × ℕ) :=
    ((Finset.range n) ×ˢ (Finset.range n)).filter (fun q => q.1 < q.2) with hI
  set A : ℕ × J × ℕ → Set (Run Ω S J) :=
    fun z => {x | certPrefix j z.1 x = prefixDraw z.2.1 z.2.2 x} with hA
  set Bq : ℕ × ℕ → Set (Run Ω S J) :=
    fun q => {x | certPrefix j q.1 x = certPrefix j q.2 x} with hBq
  have hsub : {x : Run Ω S J |
        (∀ j' ∈ populations, tabGood j' m r x) ∧ ¬ certGood populations j m r x}
      ⊆ {x | ((r : ℝ) + 1) ≤ ((K.filter (fun z => x ∈ A z)).card : ℝ)
          + ((I.filter (fun q => x ∈ Bq q)).card : ℝ)} := by
    rintro x ⟨htab, hcert⟩
    set y : ℕ → S := fun i => certPrefix j i x with hy
    set T : Finset S := prefixesAt populations m x with hT
    have h1 := le_card_acceptedAt_add T y n
    have hshort : (acceptedAt T y n).card < m := not_le.1 hcert
    -- every table prefix is among its population's first `n` draws
    have hTsub : ∀ p ∈ T, ∃ j' ∈ populations, ∃ i' < n, prefixDraw j' i' x = p := by
      intro p hp
      obtain ⟨j', hj', hp'⟩ := Finset.mem_biUnion.1 hp
      have hacc := freshDraws_subset_acceptedAt ∅ (fun i => prefixDraw j' i x) (htab j' hj') hp'
      obtain ⟨i', hi', hpi⟩ := Finset.mem_image.1 (Finset.mem_sdiff.1 hacc).1
      exact ⟨j', hj', i', Finset.mem_range.1 hi', hpi⟩
    have hhits : ((Finset.range n).filter (fun i => y i ∈ T)).card
        ≤ (K.filter (fun z => x ∈ A z)).card := by
      refine Finset.card_le_card_of_surjOn (fun z => z.1) (fun i hi => ?_)
      obtain ⟨hin, hiT⟩ := Finset.mem_filter.1 hi
      obtain ⟨j', hj', i', hi', hpi⟩ := hTsub (y i) hiT
      refine ⟨(i, j', i'), ?_, rfl⟩
      simp only [hK, hA, Finset.coe_filter, Finset.mem_product, Finset.mem_range,
        Set.mem_setOf_eq]
      exact ⟨⟨Finset.mem_range.1 hin, hj', hi'⟩, hpi.symm⟩
    have hpairs : (repeatPairs y n).card = (I.filter (fun q => x ∈ Bq q)).card := by
      congr 1
      ext q
      simp only [repeatPairs, hI, hBq, Finset.mem_filter, Set.mem_setOf_eq, and_assoc, hy]
    have : r + 1 ≤ (K.filter (fun z => x ∈ A z)).card + (I.filter (fun q => x ∈ Bq q)).card := by
      omega
    show ((r : ℝ) + 1) ≤ ((K.filter (fun z => x ∈ A z)).card : ℝ)
      + ((I.filter (fun q => x ∈ Bq q)).card : ℝ)
    exact_mod_cast this
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_count_ge_le (runMeasure μ D Dsf) K A
    (fun z => measurableSet_eq_fun (measurable_cert j z.1) (measurable_prf z.2.1 z.2.2))
    I Bq (fun q => measurableSet_eq_fun (measurable_cert j q.1) (measurable_cert j q.2))
    (by positivity)) ?_
  have hKsum : ∑ z ∈ K, (runMeasure μ D Dsf).real (A z) ≤ (K.card : ℝ) * ρ := by
    calc ∑ z ∈ K, (runMeasure μ D Dsf).real (A z) ≤ ∑ _z ∈ K, ρ := by
          refine Finset.sum_le_sum (fun z hz => ?_)
          have hz' := (Finset.mem_product.1 (Finset.mem_product.1 hz).2).1
          exact prefix_cert_eq_le D Dsf z.2.1 j z.1 z.2.2 ρ (hρ _ hz') hρj
      _ = (K.card : ℝ) * ρ := by rw [Finset.sum_const, nsmul_eq_mul]
  have hIsum : ∑ q ∈ I, (runMeasure μ D Dsf).real (Bq q) ≤ (I.card : ℝ) * ρ := by
    calc ∑ q ∈ I, (runMeasure μ D Dsf).real (Bq q) ≤ ∑ _q ∈ I, ρ := by
          refine Finset.sum_le_sum (fun q hq => ?_)
          simp only [hI, Finset.mem_filter, Finset.mem_product, Finset.mem_range] at hq
          have hY : Measurable (fun x : Run Ω S J => fun i : Fin n => certPrefix j i.val x) :=
            measurable_pi_lambda _ (fun i => measurable_cert j i.val)
          have h := block_pair_eq (runMeasure μ D Dsf) (D j)
            (fun x : Run Ω S J => fun i : Fin n => certPrefix j i.val x) hY
            (map_certBlock D Dsf j n) (a := ⟨q.1, hq.1.1⟩) (b := ⟨q.2, hq.1.2⟩)
            (fun h => absurd (congrArg Fin.val h) (ne_of_lt hq.2))
          simp only at h
          rw [show Bq q = {x | certPrefix j q.1 x = certPrefix j q.2 x} from rfl, h]
          exact hρj
      _ = (I.card : ℝ) * ρ := by rw [Finset.sum_const, nsmul_eq_mul]
  have hKcard : (K.card : ℝ) = (n : ℝ) ^ 2 * (populations.card : ℝ) := by
    rw [hK, Finset.card_product, Finset.card_product, Finset.card_range]
    push_cast
    ring
  have hIcard : 2 * (I.card : ℝ) ≤ (n : ℝ) ^ 2 := by
    have h := card_repeatPairs_le (fun _ : ℕ => (1 : S)) n
    have hI' : repeatPairs (fun _ : ℕ => (1 : S)) n = I := by
      ext q
      simp [repeatPairs, hI]
    rw [hI'] at h
    have : ((2 * I.card : ℕ) : ℝ) ≤ ((n * n : ℕ) : ℝ) := by exact_mod_cast h
    push_cast at this
    nlinarith
  have hnum : ∑ z ∈ K, (runMeasure μ D Dsf).real (A z) + ∑ q ∈ I, (runMeasure μ D Dsf).real (Bq q)
      ≤ (n : ℝ) ^ 2 * ((populations.card : ℝ) + 1 / 2) * ρ := by
    have e : (n : ℝ) ^ 2 * ((populations.card : ℝ) + 1 / 2) * ρ
        = (K.card : ℝ) * ρ + (n : ℝ) ^ 2 * ρ / 2 := by
      rw [hKcard]
      ring
    rw [e]
    have : (I.card : ℝ) * ρ ≤ (n : ℝ) ^ 2 * ρ / 2 := by
      nlinarith [mul_le_mul_of_nonneg_right hIcard hρ0]
    linarith
  exact div_le_div_of_nonneg_right hnum (by positivity)

open scoped Classical in
/-- A certification sample is found in time unless some table is not, or it redraws too much
itself. -/
theorem measureReal_certNotGood_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (populations : Finset J)
    (j : J) (m r : ℕ) (ρ : ℝ) (hρ : ∀ j' ∈ populations, collisionMass (D j') ≤ ρ)
    (hρj : collisionMass (D j) ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runMeasure μ D Dsf).real {x : Run Ω S J | ¬ certGood populations j m r x}
      ≤ ((m + r : ℕ) : ℝ) ^ 2 * ((3 * (populations.card : ℝ) + 1) / 2) * ρ / ((r : ℝ) + 1) := by
  classical
  have hsub : {x : Run Ω S J | ¬ certGood populations j m r x}
      ⊆ (⋃ j' ∈ populations, {x : Run Ω S J | ¬ tabGood j' m r x})
        ∪ {x : Run Ω S J |
            (∀ j' ∈ populations, tabGood j' m r x) ∧ ¬ certGood populations j m r x} := by
    intro x hx
    by_cases htab : ∀ j' ∈ populations, tabGood j' m r x
    · exact Or.inr ⟨htab, hx⟩
    · push_neg at htab
      obtain ⟨j', hj', hc⟩ := htab
      exact Or.inl (Set.mem_biUnion hj' hc)
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) ?_
  have h1 : (runMeasure μ D Dsf).real (⋃ j' ∈ populations, {x : Run Ω S J | ¬ tabGood j' m r x})
      ≤ (populations.card : ℝ) * (((m + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1))) := by
    refine le_trans (measureReal_biUnion_finset_le _ _) ?_
    refine le_trans (Finset.sum_le_sum (fun j' hj' =>
      measureReal_tabBad_le D Dsf j' m r ρ (hρ j' hj') hρ0)) ?_
    rw [Finset.sum_const, nsmul_eq_mul]
  have h2 := measureReal_certBad_le (μ := μ) D Dsf populations j m r ρ hρ hρj hρ0
  have e : (populations.card : ℝ) * (((m + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1)))
      + ((m + r : ℕ) : ℝ) ^ 2 * ((populations.card : ℝ) + 1 / 2) * ρ / ((r : ℝ) + 1)
      = ((m + r : ℕ) : ℝ) ^ 2 * ((3 * (populations.card : ℝ) + 1) / 2) * ρ / ((r : ℝ) + 1) := by
    field_simp
    ring
  linarith

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
theorem tableVote_unif_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (P Cd : Finset S) (hP : ∀ p ∈ P, p ∈ Pre) (hV : ∀ v ∈ Cd, v ∈ insert 1 Suf) (κ : ℕ)
    (γ b : ℝ) (hγ : 0 ≤ γ)
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
    have hFV : ∀ v ∈ F, v ∈ insert 1 Suf := fun v hv => hV v ((Finset.mem_powersetCard.mp hF).1 hv)
    have h := count_frac_hoeffding O P (fun p => F.image (fun v => p * v)) ?_
      (fun p => {ω | ¬ |(voteCount O.mq F p ω : ℝ) - ∑ v ∈ F, μ[O.mq (p * v)]|
        ≤ (F.card : ℝ) * γ}) ?_
      (2 * Real.exp (-2 * (κ : ℝ) * γ ^ 2)) b hb (fun p _ => by rw [← hFκ]; exact hper F p)
    · simpa only [Set.mem_setOf_eq] using h
    · intro p hp q hq hpq
      rw [Finset.disjoint_left]
      intro z hz hz'
      obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hz
      obtain ⟨v', hv'F, hv'⟩ := Finset.mem_image.1 hz'
      exact hpq (hflat p (hP p hp) q (hP q hq) v (hFV v hv) v' (hFV v' hv'F) hv'.symm)
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
theorem frac_selected_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hV : ∀ v ∈ cands, v ∈ insert 1 Suf)
    (hCPre : ∀ p ∈ C, p ∈ Pre) (hPC : Disjoint P C)
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
    exact Finset.disjoint_coe.1 (disjoint_image_readSet hflat hP hV (hCPre p hp)
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
      obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hz
      obtain ⟨v', hv'A, hv'⟩ := Finset.mem_image.1 hz'
      exact hpq (hflat p (hCPre p hp) q (hCPre q hq) v (hV v (hTC A₀ hA₀ hv)) v'
        (hV v' (hTC A₀ hA₀ hv'A)) hv'.symm)
  exact selection_block_bound O R (readSet P cands) hdisjR T t₀ ht₀ fam hfam hcongr Bad
    hmeasU hmeasR _ (Real.exp_nonneg _) hfixed

open scoped Classical in
/-- The indecision rate on the certification sample is below the limit, off a Hoeffding
tail. -/
theorem indecision_frac_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hV : ∀ v ∈ cands, v ∈ insert 1 Suf)
    (hCPre : ∀ p ∈ C, p ∈ Pre) (hPC : Disjoint P C) (lo ha : ℕ)
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E l : ℝ) (hE : 0 ≤ E) (hEl : E ≤ l)
    (hbad : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p → μ.real {ω | ¬ decided O.mq lo ha A₀ p ω} ≤ E) :
    μ.real {ω | l * (C.card : ℝ)
        < ((C.filter (fun p => fam ω ∈ good p ∧ ¬ decided O.mq lo ha (fam ω) p ω)).card : ℝ)}
      ≤ Real.exp (-2 * (C.card : ℝ) * (l - E) ^ 2) := by
  classical
  have h := frac_selected_le hflat O P cands C hP hV hCPre hPC T t₀ ht₀ hTC fam hfam hcongr
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
/-- `indecision_frac_le` for the family with its seed added.  The seed's read is at the bare
prefix, which the clustering never reads either. -/
theorem indecision_frac_le_seed {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hV : ∀ v ∈ cands, v ∈ insert 1 Suf)
    (hCPre : ∀ p ∈ C, p ∈ Pre) (hPC : Disjoint P C) (hone : (1 : S) ∈ cands) (lo ha : ℕ)
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (hT1 : ∀ t ∈ T, (1 : S) ∉ t) (fam : Ω → Finset S)
    (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E l : ℝ) (hE : 0 ≤ E) (hEl : E ≤ l)
    (hbad : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p →
      μ.real {ω | ¬ decided O.mq lo ha (insert 1 A₀) p ω} ≤ E) :
    μ.real {ω | l * (C.card : ℝ)
        < ((C.filter (fun p => fam ω ∈ good p
            ∧ ¬ decided O.mq lo ha (insert 1 (fam ω)) p ω)).card : ℝ)}
      ≤ Real.exp (-2 * (C.card : ℝ) * (l - E) ^ 2) := by
  classical
  set T' : Finset (Finset S) := T.image (insert 1) with hT'
  set good' : S → Finset (Finset S) := fun p => T'.filter (fun A => A.erase 1 ∈ good p)
    with hgood'
  have h := indecision_frac_le hflat O P cands C hP hV hCPre hPC lo ha T' good' (insert 1 t₀)
    (Finset.mem_image_of_mem _ ht₀)
    (fun t ht => by
      obtain ⟨t₁, ht₁, rfl⟩ := Finset.mem_image.1 ht
      exact Finset.insert_subset hone (hTC t₁ ht₁))
    (fun ω => insert 1 (fam ω)) (fun ω => Finset.mem_image_of_mem _ (hfam ω))
    (fun ω ω' hw => by rw [hcongr ω ω' hw])
    E l hE hEl
    (fun p hp A hA hg => by
      obtain ⟨A₀, hA₀, rfl⟩ := Finset.mem_image.1 hA
      have hg' := (Finset.mem_filter.1 hg).2
      rw [Finset.erase_insert (hT1 A₀ hA₀)] at hg'
      exact hbad p hp A₀ hA₀ hg')
  refine le_trans (le_of_eq ?_) h
  congr 1
  ext ω
  simp only [Set.mem_setOf_eq]
  have hmem : ∀ p, (fam ω ∈ good p) ↔ (insert 1 (fam ω) ∈ good' p) := by
    intro p
    rw [hgood', Finset.mem_filter, Finset.erase_insert (hT1 _ (hfam ω))]
    exact ⟨fun hg => ⟨Finset.mem_image_of_mem _ (hfam ω), hg⟩, fun h => h.2⟩
  simp only [hmem]

open scoped Classical in
/-- The same for the rate at which the cut is wrong. -/
theorem miscut_frac_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hV : ∀ v ∈ cands, v ∈ insert 1 Suf)
    (hCPre : ∀ p ∈ C, p ∈ Pre) (hPC : Disjoint P C) (lo ha : ℕ)
    (T : Finset (Finset S)) (good : S → Finset (Finset S)) (t₀ : Finset S) (ht₀ : t₀ ∈ T)
    (hTC : ∀ t ∈ T, t ⊆ cands) (fam : Ω → Finset S) (hfam : ∀ ω, fam ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (E l : ℝ) (hE : 0 ≤ E) (hEl : E ≤ l)
    (hbad : ∀ p ∈ C, ∀ A₀ ∈ T, A₀ ∈ good p → μ.real {ω | ¬ cutCorrect O lo ha A₀ p ω} ≤ E) :
    μ.real {ω | l * (C.card : ℝ)
        < ((C.filter (fun p => fam ω ∈ good p ∧ ¬ cutCorrect O lo ha (fam ω) p ω)).card : ℝ)}
      ≤ Real.exp (-2 * (C.card : ℝ) * (l - E) ^ 2) := by
  classical
  have h := frac_selected_le hflat O P cands C hP hV hCPre hPC T t₀ ht₀ hTC fam hfam hcongr
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

/-- `clusterBy` with the draws fixed: the family is a function of the noise alone. -/
noncomputable def clusterOf (rule : Clusterer S) (O : Oracle μ S) (sc scd : ℕ) (P cands : Finset S)
    (k : ℕ) (ω : Ω) : Finset S :=
  (rule.pick (fun w => O.mq w ω = 1) P (screened O.mq sc scd P cands ω) k).erase 1

lemma clusterAt_eq_clusterOf (O : Oracle μ S) (populations : Finset J) (B : State)
    (x : Run Ω S J) :
    clusterBy rule O.mq populations x B
      = clusterOf rule O B.sc B.scd (prefixesAt populations B.npref x) (poolAt B.nsuff x) B.k (oracleNoise x) :=
  rfl

lemma clusterOf_subset (O : Oracle μ S) (sc scd : ℕ) (P cands : Finset S) (k : ℕ) (ω : Ω)
    (hone : (1 : S) ∈ cands) : clusterOf rule O sc scd P cands k ω ⊆ cands :=
  fun v hv => screened_subset O sc scd P cands ω
    (rule.subset _ _ _ k (one_mem_screened O sc scd P cands ω hone) (Finset.mem_of_mem_erase hv))

lemma clusterOf_congr_mq (O : Oracle μ S) (sc scd : ℕ) (P cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands) {ω ω' : Ω}
    (hbit : ∀ w ∈ readSet P cands, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    clusterOf rule O sc scd P cands k ω = clusterOf rule O sc scd P cands k ω' := by
  classical
  have hscr : screened O.mq sc scd P cands ω = screened O.mq sc scd P cands ω' :=
    Finset.filter_congr (fun v hv => by
      rw [screenCount_congr O hone hv hbit, screenBase_congr O hone hbit])
  unfold clusterOf
  rw [hscr]
  exact congrArg (fun t : Finset S => t.erase 1) (rule.congr _ _ _ _ _
    (one_mem_screened O sc scd P cands ω' hone)
    (fun p hp v hv => hbit _ (mem_readSet hp (screened_subset O sc scd P cands ω' hv))))

open scoped Classical in
lemma measurableSet_clusterOf (O : Oracle μ S) (sc scd : ℕ) (P cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands) (A₀ : Finset S) :
    MeasurableSet {ω | clusterOf rule O sc scd P cands k ω = A₀} := by
  classical
  set Pred : Finset S → Prop := fun U => ∃ ω', (readSet P cands).filter
    (fun w => O.mq w ω' = 1) = U ∧ clusterOf rule O sc scd P cands k ω' = A₀ with hPred
  have hcov : {ω | clusterOf rule O sc scd P cands k ω = A₀}
      = {ω | Pred ((readSet P cands).filter (fun w => O.mq w ω = 1))} := by
    ext ω
    simp only [Set.mem_setOf_eq, hPred]
    refine ⟨fun h => ⟨ω, rfl, h⟩, ?_⟩
    rintro ⟨ω', hU, hA⟩
    refine (clusterOf_congr_mq O sc scd P cands k hone (fun w hw => ?_)).trans hA
    have := Finset.ext_iff.1 hU w
    simp only [Finset.mem_filter, hw, true_and] at this
    exact this.symm
  rw [hcov]
  exact noiseAlg_le O Set.univ _
    (measurableSet_filter_pred O (T := Set.univ) (by simp) Pred)

open scoped Classical in
lemma measurableSet_insert_clusterOf (O : Oracle μ S) (sc scd : ℕ) (P cands : Finset S) (k : ℕ)
    (hone : (1 : S) ∈ cands) (A₀ : Finset S) :
    MeasurableSet {ω | insert 1 (clusterOf rule O sc scd P cands k ω) = A₀} := by
  classical
  have hcov : {ω | insert 1 (clusterOf rule O sc scd P cands k ω) = A₀}
      = ⋃ A ∈ cands.powerset.filter (fun A => insert 1 A = A₀),
          {ω | clusterOf rule O sc scd P cands k ω = A} := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Finset.mem_filter, Finset.mem_powerset,
      exists_prop]
    exact ⟨fun h => ⟨_, ⟨clusterOf_subset O sc scd P cands k ω hone, h⟩, rfl⟩,
      fun ⟨A, ⟨_, hA⟩, hc⟩ => hc ▸ hA⟩
  rw [hcov]
  exact MeasurableSet.biUnion (Finset.countable_toSet _)
    (fun A _ => measurableSet_clusterOf O sc scd P cands k hone A)

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
at least `Δ·gap·(1−2η)` above the pool's floor, so it clears a cutoff of `Δ·gap·(1−2η) − 2γ`
only in a tail — one `γ` for its own count, one for the floor, whose upper bound is what a clean
candidate `w₀` in the pool buys. -/
theorem screen_tail_rel {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S) {cn cd : ℕ}
    (hcd : cn < cd) {P cands : Finset S} (hP : ∀ p ∈ P, p ∈ Pre)
    (hV : ∀ v ∈ cands, v ∈ insert 1 Suf) (v : S) (hvS : v ∈ insert 1 Suf) (hv : v ≠ 1)
    (w₀ : S) (hw₀ : w₀ ∈ cands.erase 1) (hw₀clean : ∀ p ∈ P, O.flip w₀ p = 0)
    (Δ γ : ℝ) (sc scd : ℕ) (hγ : 0 ≤ γ) (hsig : O.η ≤ 1 / 2) (hscd : 0 < scd)
    (hflip : Δ * (P.card : ℝ) ≤ ∑ p ∈ P, O.flip v p)
    (hsc : (sc : ℝ) ≤ (scd : ℝ) * (Δ * (O.gap * (1 - 2 * O.η)) - 2 * γ)) :
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
            ≤ cleanLoss O P + (P.card : ℝ) * (Δ * (O.gap * (1 - 2 * O.η)) - γ)} := by
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
  have h1 := screenCount_upper hflat O hcd hP w₀ (hV w₀ (Finset.mem_of_mem_erase hw₀))
    (Finset.ne_of_mem_erase hw₀) γ hγ hsig hw₀clean
  have h2 := screenCount_lower hflat O hcd hP v hvS hv Δ γ hγ hsig hflip
  linarith [h1, h2]

/-- The relative screen's liveness.  A clean candidate's count is within `γ` of `cleanLoss`
and the floor is within `γ` below it, so `2γ` of cutoff admits it.  The floor's lower bound
is a union bound over the pool, which is where the `#cands` factor comes from. -/
theorem screen_pass_rel {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S) {cn cd : ℕ}
    (hcd : cn < cd) {P cands : Finset S} (hP : ∀ p ∈ P, p ∈ Pre)
    (hV : ∀ v ∈ cands, v ∈ insert 1 Suf) (v : S) (hv : v ≠ 1) (hvc : v ∈ cands)
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
            {ω | (screenCount O.mq P w ω : ℝ)
              ≤ cleanLoss O P + (P.card : ℝ) * (0 * (O.gap * (1 - 2 * O.η)) - γ)} := by
    intro ω hω
    by_contra hnot
    simp only [Set.mem_union, not_or, Set.mem_iUnion, not_exists, Set.mem_setOf_eq,
      not_le, exists_prop, not_and] at hnot
    obtain ⟨hvU, hBl⟩ := hnot
    have hbase : cleanLoss O P + (P.card : ℝ) * (0 * (O.gap * (1 - 2 * O.η)) - γ)
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
  have h1 := screenCount_upper hflat O hcd hP v (hV v hvc) hv γ hγ hsig hclean
  have h2 : μ.real (⋃ w ∈ cands.erase 1,
        {ω | (screenCount O.mq P w ω : ℝ)
          ≤ cleanLoss O P + (P.card : ℝ) * (0 * (O.gap * (1 - 2 * O.η)) - γ)})
      ≤ (cands.card : ℝ) * Real.exp (-2 * (P.card : ℝ) * γ ^ 2) := by
    refine le_trans (measureReal_biUnion_finset_le _ _) ?_
    have hper : ∀ w ∈ cands.erase 1,
        μ.real {ω | (screenCount O.mq P w ω : ℝ)
          ≤ cleanLoss O P + (P.card : ℝ) * (0 * (O.gap * (1 - 2 * O.η)) - γ)}
          ≤ Real.exp (-2 * (P.card : ℝ) * γ ^ 2) :=
      fun w hw => screenCount_lower hflat O hcd hP w (hV w (Finset.mem_of_mem_erase hw))
        (Finset.ne_of_mem_erase hw) 0 γ hγ hsig
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
theorem measureReal_screenBad_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (B : State)
    (hsig : O.η ≤ 1 / 2) (hmpos : 0 < B.npref) (Δ γ : ℝ) (hΔ : 0 < Δ) (hγ : 0 ≤ γ)
    (hscd : 0 < B.scd)
    (hsc : (B.sc : ℝ) ≤ (B.scd : ℝ) * (Δ * (O.gap * (1 - 2 * O.η)) - 2 * γ)) :
    (runMeasure μ D Dsf).real (screenBad O populations B Δ)
      ≤ ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2) := by
  classical
  set E : ℝ := ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2) with hEdef
  have hE0 : 0 ≤ E := by positivity
  have hEnn : runMeasure μ D Dsf (screenBad O populations B Δ) ≤ ENNReal.ofReal E := by
    refine runMeasure_slice_le D Dsf _ (measurableSet_screenBad O populations B Δ) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
      ae_sfx_mem_Suf D Dsf Suf hsuppSf] with d hd hdS
    set Pd : Finset S := populations.biUnion
      (fun j => freshDraws ∅ (fun i => d.1.2 j i) B.npref) with hPd
    set Cd : Finset S := insert 1 (freshDraws ∅ (fun i => d.1.1 i) B.nsuff) with hCd
    have hV : ∀ v ∈ Cd, v ∈ insert 1 Suf := pool_mem_Suf B.nsuff (fun i => d.1.1 i) hdS
    have hP : ∀ q ∈ Pd, q ∈ Pre :=
      table_mem_Pre populations B.npref (fun j i => d.1.2 j i) hd
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
        have htail := screen_tail_rel hflat O Nat.one_lt_two hP hV v (hV v (Finset.mem_filter.1 hv).1) hv1
          w₀ hw₀ hw₀c Δ γ B.sc B.scd hγ hsig hscd hflip hsc
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
        have hCard : Cd.card ≤ B.nsuff + 1 :=
          le_trans (Finset.card_insert_le _ _)
            (by have := card_freshDraws_le ∅ (fun i => d.1.1 i) B.nsuff; omega)
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
/-- The pool's accept-preserving draws, as strings: the sampler loses at most `r` of the
draws to redraws, so at least `#AP − r` of them are distinct and in the pool. -/
lemma ap_fresh_ge (O : Oracle μ S) (M r : ℕ) (x : Run Ω S J) (hgood : poolGood M r x) :
    ((((Finset.univ : Finset (Fin M)).filter
        (fun i : Fin M => ∀ p, O.label (p * suffixDraw i.val x) = O.label p)).card : ℝ))
      ≤ (((freshDraws ∅ (fun i => suffixDraw i x) M).filter
          (fun v => ∀ p, O.label (p * v) = O.label p)).card : ℝ) + r := by
  classical
  have h := (card_filter_of_good ∅ (fun i => suffixDraw i x) M r hgood
    (fun v => ∀ p, O.label (p * v) = O.label p)).1
  have hfin := card_filter_univ_fin M (fun i => ∀ p, O.label (p * suffixDraw i x) = O.label p)
  have hfin' : ((((Finset.univ : Finset (Fin M)).filter
      (fun i : Fin M => ∀ p, O.label (p * suffixDraw i.val x) = O.label p)).card : ℕ) : ℝ)
      = ((((Finset.range M).filter
          (fun i => ∀ p, O.label (p * suffixDraw i x) = O.label p)).card : ℕ) : ℝ) := by
    exact_mod_cast hfin
  rw [hfin']
  exact h

open scoped Classical in
/-- The pool holds a clean non-seed reference.  Findability puts accept-preserving draws in
it, the sampler loses at most `r` of them to redraws, and at most one of the rest is the seed,
so two more than `r` suffice. -/
theorem noCleanRef_subset (O : Oracle μ S) (populations : Finset J) (B : State) (pAP t : ℝ)
    (r : ℕ) (hcount : (2 : ℝ) + r ≤ (B.nsuff : ℝ) * (pAP - t)) :
    {x : Run Ω S J | ¬ ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
        ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
      ⊆ apShort O B.nsuff pAP t ∪ {x | ¬ poolGood B.nsuff r x} := by
  classical
  intro x hx
  by_contra hnot
  simp only [Set.mem_union, not_or, apShort, Set.mem_setOf_eq, not_not] at hnot
  obtain ⟨hcnt, hgood⟩ := hnot
  set W : Finset S := (freshDraws ∅ (fun i => suffixDraw i x) B.nsuff).filter
    (fun v => ∀ p, O.label (p * v) = O.label p) with hW
  have hfresh := ap_fresh_ge O B.nsuff r x hgood
  rw [← hW] at hfresh
  have h2 : 1 < W.card := by
    have : (2 : ℝ) < (W.card : ℝ) := by linarith [not_le.1 hcnt]
    have : 2 < W.card := by exact_mod_cast this
    omega
  obtain ⟨a, ha, b, hb, hab⟩ := Finset.one_lt_card.1 h2
  have hpick : ∃ w ∈ W, w ≠ 1 := by
    by_cases h : a = 1
    · exact ⟨b, hb, fun hb1 => hab (h.trans hb1.symm)⟩
    · exact ⟨a, ha, h⟩
  obtain ⟨w, hw, hw1⟩ := hpick
  refine hx ⟨w, Finset.mem_erase.2 ⟨hw1, Finset.mem_insert_of_mem (Finset.mem_filter.1 hw).1⟩, ?_⟩
  intro p _
  have hap := (Finset.mem_filter.1 hw).2 p
  change O.label (p * w) + O.label p - 2 * O.label (p * w) * O.label p = 0
  rw [hap]
  rcases O.label_bit p with hl | hl <;> rw [hl] <;> ring

theorem measureReal_noCleanRef_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (populations : Finset J) (B : State) (pAP t ρsf : ℝ) (r : ℕ)
    (hpAP0 : 0 ≤ pAP) (ht : 0 ≤ t)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hcount : (2 : ℝ) + r ≤ (B.nsuff : ℝ) * (pAP - t)) :
    (runMeasure μ D Dsf).real {x : Run Ω S J |
        ¬ ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
          ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
      ≤ Real.exp (-2 * (B.nsuff : ℝ) * t ^ 2)
        + ((B.nsuff + r : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((r : ℝ) + 1)) := by
  refine le_trans (measureReal_mono (noCleanRef_subset O populations B pAP t r hcount)
    (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_union_le _ _) ?_
  gcongr
  · exact measureReal_apShort_le D Dsf O B.nsuff pAP t hpAP0 ht hpAPBound
  · exact measureReal_poolBad_le D Dsf B.nsuff r ρsf hρsf hρsf0

/-- The screen keeps the accept-preserving candidates.  One `screen_pass_rel` per pool
member; the cutoff sits above the clean rate by `γ`. -/
theorem measureReal_screenFail_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (B : State)
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
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
      ae_sfx_mem_Suf D Dsf Suf hsuppSf] with d hd hdS
    set Pd : Finset S := populations.biUnion
      (fun j => freshDraws ∅ (fun i => d.1.2 j i) B.npref) with hPd
    set Cd : Finset S := insert 1 (freshDraws ∅ (fun i => d.1.1 i) B.nsuff) with hCd
    have hV : ∀ v ∈ Cd, v ∈ insert 1 Suf := pool_mem_Suf B.nsuff (fun i => d.1.1 i) hdS
    have hP : ∀ q ∈ Pd, q ∈ Pre :=
      table_mem_Pre populations B.npref (fun j i => d.1.2 j i) hd
    have hPle : Pd.card ≤ populations.card * B.npref := by
      rw [hPd]
      refine le_trans Finset.card_biUnion_le ?_
      calc ∑ j ∈ populations, (freshDraws ∅ (fun i => d.1.2 j i) B.npref).card
          ≤ ∑ _j ∈ populations, B.npref :=
            Finset.sum_le_sum (fun j _ => card_freshDraws_le _ _ _)
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
        have htail := screen_pass_rel (cands := Cd) hflat O Nat.one_lt_two hP hV v hv1 hvC γ B.sc B.scd hγ
          hsig hscd hclean hsc
        have hcardR : (B.npref : ℝ) ≤ (Pd.card : ℝ) := by exact_mod_cast hm
        have hCard : Cd.card ≤ B.nsuff + 1 :=
          le_trans (Finset.card_insert_le _ _)
            (by have := card_freshDraws_le ∅ (fun i => d.1.1 i) B.nsuff; omega)
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
      have hCard : Cd.card ≤ B.nsuff + 1 :=
        le_trans (Finset.card_insert_le _ _)
          (by have := card_freshDraws_le ∅ (fun i => d.1.1 i) B.nsuff; omega)
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
/-- The table of a population found in time holds `m` prefixes. -/
lemma npref_le_card_of_tabGood (populations : Finset J) (j : J) (hj : j ∈ populations) (m r : ℕ)
    (x : Run Ω S J) (h : tabGood j m r x) : m ≤ (prefixesAt populations m x).card := by
  have hcard : (prefixesOf j m x).card = m :=
    card_freshDraws ∅ (fun i => prefixDraw j i x) m ⟨m + r, h⟩
  calc m = (prefixesOf j m x).card := hcard.symm
    _ ≤ (prefixesAt populations m x).card :=
        Finset.card_le_card (fun q hq => Finset.mem_biUnion.2 ⟨j, hj, hq⟩)

open scoped Classical in
theorem measureReal_smallScreen_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (B : State)
    (j₀ : J) (hj₀ : j₀ ∈ populations)
    (γ pAP t ρsf ρ : ℝ) (rs rt : ℕ) (hγ : 0 ≤ γ) (hpAP0 : 0 ≤ pAP) (ht : 0 ≤ t)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hscd : 0 < B.scd) (hsig : O.η ≤ 1 / 2)
    (hsc : (B.scd : ℝ) * (2 * γ) ≤ (B.sc : ℝ))
    (hcount : (B.k : ℝ) + rs ≤ (B.nsuff : ℝ) * (pAP - t))
    (hρsf : collisionMass Dsf ≤ ρsf) (hρsf0 : 0 ≤ ρsf)
    (hρj : collisionMass (D j₀) ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runMeasure μ D Dsf).real
        {x : Run Ω S J | ¬ (B.k ≤ (screenedAt O.mq populations B x).card)}
      ≤ ((B.nsuff + rs : ℕ) : ℝ) ^ 2 * ρsf / (2 * ((rs : ℝ) + 1))
        + (Real.exp (-2 * (B.nsuff : ℝ) * t ^ 2)
        + (((B.npref + rt : ℕ) : ℝ) ^ 2 * ρ / (2 * ((rt : ℝ) + 1))
          + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2))) := by
  classical
  set E1 : Set (Run Ω S J) := {x | ¬ poolGood B.nsuff rs x} with hE1
  set E2 : Set (Run Ω S J) := apShort O B.nsuff pAP t with hE2
  set E3 : Set (Run Ω S J) := {x | ¬ tabGood j₀ B.npref rt x} with hE3
  set E4 : Set (Run Ω S J) := screenFail O populations B with hE4
  have hsub : {x : Run Ω S J | ¬ (B.k ≤ (screenedAt O.mq populations B x).card)}
      ⊆ E1 ∪ (E2 ∪ (E3 ∪ E4)) := by
    intro x hx
    by_contra hcon
    simp only [Set.mem_union, not_or] at hcon
    obtain ⟨h1, h2, h3, h4⟩ := hcon
    have hgood : poolGood B.nsuff rs x := by by_contra h; exact h1 h
    have htab : tabGood j₀ B.npref rt x := by by_contra h; exact h3 h
    have hapcount : (B.nsuff : ℝ) * (pAP - t)
        < (((Finset.univ : Finset (Fin B.nsuff)).filter
          (fun i => ∀ p, O.label (p * suffixDraw i.val x) = O.label p)).card : ℝ) := not_le.1 h2
    have hmP : B.npref ≤ (prefixesAt populations B.npref x).card :=
      npref_le_card_of_tabGood populations j₀ hj₀ B.npref rt x htab
    have hscreen : ∀ v ∈ poolAt B.nsuff x, v ≠ 1 → (∀ p, O.label (p * v) = O.label p) →
        B.scd * screenCount O.mq (prefixesAt populations B.npref x) v (oracleNoise x)
          ≤ B.scd * screenBase O.mq (prefixesAt populations B.npref x)
              (poolAt B.nsuff x) (oracleNoise x)
            + B.sc * (prefixesAt populations B.npref x).card := by
      by_contra h
      exact h4 ⟨hmP, h⟩
    -- the accept-preserving strings the pool holds all survive the screen
    set W : Finset S := (freshDraws ∅ (fun i => suffixDraw i x) B.nsuff).filter
      (fun v => ∀ p, O.label (p * v) = O.label p) with hW
    have hWsub : W ⊆ screenedAt O.mq populations B x := by
      intro v hv
      obtain ⟨hvF, hap⟩ := Finset.mem_filter.1 hv
      have hpool : v ∈ poolAt B.nsuff x := Finset.mem_insert_of_mem hvF
      refine Finset.mem_filter.2 ⟨hpool, ?_⟩
      by_cases hv1 : v = 1
      · rw [hv1, screenCount_one]
        exact Nat.zero_le _
      · exact hscreen v hpool hv1 hap
    have hfresh := ap_fresh_ge O B.nsuff rs x hgood
    rw [← hW] at hfresh
    have hk : (B.k : ℝ) ≤ (W.card : ℝ) := by linarith
    have hcast : (W.card : ℝ) ≤ ((screenedAt O.mq populations B x).card : ℝ) := by
      exact_mod_cast Finset.card_le_card hWsub
    have : (B.k : ℝ) ≤ ((screenedAt O.mq populations B x).card : ℝ) := le_trans hk hcast
    exact hx (by exact_mod_cast this)
  calc (runMeasure μ D Dsf).real {x : Run Ω S J | ¬ (B.k ≤ (screenedAt O.mq populations B x).card)}
      ≤ (runMeasure μ D Dsf).real (E1 ∪ (E2 ∪ (E3 ∪ E4))) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ (runMeasure μ D Dsf).real E1 + ((runMeasure μ D Dsf).real E2
        + ((runMeasure μ D Dsf).real E3 + (runMeasure μ D Dsf).real E4)) := by
        have h34 := measureReal_union_le (μ := runMeasure μ D Dsf) E3 E4
        have h234 := measureReal_union_le (μ := runMeasure μ D Dsf) E2 (E3 ∪ E4)
        have hall := measureReal_union_le (μ := runMeasure μ D Dsf) E1 (E2 ∪ (E3 ∪ E4))
        linarith
    _ ≤ _ := by
        gcongr
        · exact measureReal_poolBad_le D Dsf B.nsuff rs ρsf hρsf hρsf0
        · exact measureReal_apShort_le D Dsf O B.nsuff pAP t hpAP0 ht hpAPBound
        · exact measureReal_tabBad_le D Dsf j₀ B.npref rt ρ hρj hρ0
        · exact measureReal_screenFail_le hflat O populations D Dsf hsupp hsuppSf B γ hγ hscd
            hsig hsc

open scoped Classical in
/-- Every member the clustering keeps is clean.  Three things have to go right: the
population's table found in time, the screen holding, and the drawn prefixes not understating a
candidate's flip mass.  The table misses at most `r` of the population's first draws, which a
candidate can flip, so the bound carries `c ≥ r/m` more.  The Lloyd ranking does not appear — the
family is a subset of what the screen left, so it inherits the bound. -/
theorem measureReal_dirtyMember_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0) (j : J) (hj : j ∈ populations)
    (B : State) (hsig : O.η ≤ 1 / 2) (hmpos : 0 < B.npref)
    (Δ γ g c ρ : ℝ) (r : ℕ) (hΔ : 0 < Δ) (hγ : 0 ≤ γ) (hg : 0 ≤ g) (hc0 : 0 ≤ c) (hρ0 : 0 ≤ ρ)
    (hr : (r : ℝ) ≤ c * B.npref)
    (hρD : collisionMass (D j) ≤ ρ)
    (hscd : 0 < B.scd)
    (hsc : (B.sc : ℝ) ≤ (B.scd : ℝ) * (Δ * (O.gap * (1 - 2 * O.η)) - 2 * γ)) :
    (runMeasure μ D Dsf).real ({x : Run Ω S J | ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
          ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
        ∩ {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B,
          flipMass O (D j) v ≤ (populations.card : ℝ) * Δ + c + g})
      ≤ ((B.npref + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1))
        + ((B.nsuff : ℝ) + 2) ^ 2 * Real.exp (-2 * (B.npref : ℝ) * γ ^ 2)
        + (B.nsuff : ℝ) * Real.exp (-2 * (B.npref : ℝ) * g ^ 2) := by
  classical
  set Δp : ℝ := (populations.card : ℝ) * Δ + c + g with hΔp
  set E1 : Set (Run Ω S J) := {x | ¬ tabGood j B.npref r x} with hE1
  set E2 : Set (Run Ω S J) := screenBad O populations B Δ with hE2
  set E3 : Set (Run Ω S J) := {x | ¬ ∀ v ∈ poolAt B.nsuff x,
    (∑ i : Fin B.npref, O.flip v (prefixDraw j i.val x) ≤ (B.npref : ℝ) * (Δp - g)) →
      flipMass O (D j) v ≤ Δp} with hE3
  have hsub : ({x : Run Ω S J | ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
        ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
      ∩ {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B,
        flipMass O (D j) v ≤ Δp}) ⊆ (E1 ∪ E2) ∪ E3 := by
    rintro x ⟨hclean, hx⟩
    by_contra hnot
    simp only [Set.mem_union, not_or] at hnot
    obtain ⟨⟨h1, h2⟩, h3⟩ := hnot
    have htab : tabGood j B.npref r x := by by_contra h; exact h1 h
    have hcardPre : B.npref ≤ (prefixesAt populations B.npref x).card :=
      npref_le_card_of_tabGood populations j hj B.npref r x htab
    -- the screen held, so nothing it left flips much of the table
    have hscreen : ∀ v ∈ screenedAt O.mq populations B x,
        ¬ (Δ * ((prefixesAt populations B.npref x).card : ℝ)
          ≤ ∑ p ∈ prefixesAt populations B.npref x, O.flip v p) := by
      by_contra h
      exact h2 ⟨hcardPre, hclean, h⟩
    have hrank := clusterAt_flip_bound (rule := rule) O populations B x Δ hscreen
    obtain ⟨v, hv, hbad⟩ : ∃ v ∈ clusterBy rule O.mq populations x B,
        ¬ (flipMass O (D j) v ≤ Δp) := by
      by_contra h
      exact hx (fun v hv => by
        by_contra hc
        exact h ⟨v, hv, hc⟩)
    have hvpool : v ∈ poolAt B.nsuff x := clusterAt_subset O populations B x hv
    have hcount : ∑ i : Fin B.npref, O.flip v (prefixDraw j i.val x)
        ≤ (B.npref : ℝ) * (Δp - g) := by
      obtain ⟨-, -, hsum⟩ := freshDraws_of_good ∅ (fun i => prefixDraw j i x) B.npref r htab
      have hdraw := (hsum (fun p => O.flip v p) (fun p => (flip_icc O v p).1)
        (fun p => (flip_icc O v p).2)).1
      rw [Fin.sum_univ_eq_sum_range (fun i => O.flip v (prefixDraw j i x))]
      have hle1 := sum_prefixesOf_le_prefixesAt O populations j hj B.npref x v
      have hle2 := not_le.1 (hrank v hv)
      have hle3 : ((prefixesAt populations B.npref x).card : ℝ)
          ≤ (populations.card : ℝ) * (B.npref : ℝ) := by
        exact_mod_cast card_prefixesAt_le populations B.npref x
      have hΔ0 : (0 : ℝ) ≤ Δ := le_of_lt hΔ
      have hrw : (B.npref : ℝ) * (Δp - g)
          = (populations.card : ℝ) * Δ * (B.npref : ℝ) + c * (B.npref : ℝ) := by
        rw [hΔp]; ring
      have hof : ∑ p ∈ prefixesOf j B.npref x, O.flip v p
          = ∑ p ∈ freshDraws ∅ (fun i => prefixDraw j i x) B.npref, O.flip v p := rfl
      rw [hrw]
      nlinarith
    exact hbad (by
      by_contra h
      exact h3 (by
        simp only [hE3, Set.mem_setOf_eq, not_forall]
        exact ⟨v, hvpool, hcount, h⟩))
  calc (runMeasure μ D Dsf).real ({x : Run Ω S J | ∃ w₀ ∈ (poolAt B.nsuff x).erase 1,
          ∀ p ∈ prefixesAt populations B.npref x, O.flip w₀ p = 0}
        ∩ {x : Run Ω S J | ¬ ∀ v ∈ clusterBy rule O.mq populations x B,
          flipMass O (D j) v ≤ Δp})
      ≤ (runMeasure μ D Dsf).real ((E1 ∪ E2) ∪ E3) := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ((runMeasure μ D Dsf).real E1 + (runMeasure μ D Dsf).real E2)
        + (runMeasure μ D Dsf).real E3 := by
        have h12 := measureReal_union_le (μ := runMeasure μ D Dsf) E1 E2
        have h123 := measureReal_union_le (μ := runMeasure μ D Dsf) (E1 ∪ E2) E3
        linarith
    _ ≤ _ := by
        gcongr
        · exact measureReal_tabBad_le D Dsf j B.npref r ρ hρD hρ0
        · exact measureReal_screenBad_le hflat O populations D Dsf hsupp hsuppSf B hsig hmpos
            Δ γ hΔ hγ hscd hsc
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

open scoped Classical in
/-- Events about the table, the pool and the certification sample are measurable: all three
take countably many values. -/
lemma measurableSet_of_run_data_certSet (populations : Finset J) (j : J) (B : State)
    (R : Finset S → Finset S → Finset S → Set (Run Ω S J))
    (hR : ∀ P C A, MeasurableSet (R P C A)) :
    MeasurableSet {x : Run Ω S J | x ∈ R (prefixesAt populations B.npref x) (poolAt B.nsuff x)
      (certOf (prefixesAt populations B.npref x) j B.npref x)} := by
  classical
  have hcov : {x : Run Ω S J | x ∈ R (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (certOf (prefixesAt populations B.npref x) j B.npref x)}
      = ⋃ z : Finset S × Finset S × Finset S,
          ((({x : Run Ω S J | prefixesAt populations B.npref x = z.1}
            ∩ {x : Run Ω S J | poolAt B.nsuff x = z.2.1})
            ∩ {x : Run Ω S J | certOf (prefixesAt populations B.npref x) j B.npref x = z.2.2})
          ∩ R z.1 z.2.1 z.2.2) := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff]
    refine ⟨fun h => ⟨(prefixesAt populations B.npref x, poolAt B.nsuff x,
      certOf (prefixesAt populations B.npref x) j B.npref x), ⟨⟨rfl, rfl⟩, rfl⟩, h⟩, ?_⟩
    rintro ⟨⟨P, C, A⟩, ⟨⟨hP, hC⟩, hA⟩, hx⟩
    simp only at hP hC hA
    rw [hA, hC, hP]
    exact hx
  rw [hcov]
  exact MeasurableSet.iUnion (fun z =>
    (((measurableSet_prefixesAt populations B.npref z.1).inter
      (measurableSet_poolAt B.nsuff z.2.1)).inter
        (measurableSet_certOf populations B.npref j B.npref z.2.2)).inter
      (hR z.1 z.2.1 z.2.2))

/-- A count over the first `n` indices is measurable once each index's event is. -/
lemma measurableSet_card_filter_fin {α : Type*} [MeasurableSpace α] {n : ℕ}
    (Q : Fin n → α → Prop) [∀ ω, DecidablePred (fun i => Q i ω)]
    (hQ : ∀ i, MeasurableSet {ω | Q i ω}) (P : ℕ → Prop) :
    MeasurableSet {ω | P ((Finset.univ.filter (fun i => Q i ω)).card)} := by
  classical
  have hcov : {ω | P ((Finset.univ.filter (fun i => Q i ω)).card)}
      = ⋃ U ∈ (Finset.univ : Finset (Finset (Fin n))).filter (fun U => P U.card),
          ⋂ i : Fin n, {ω | Q i ω ↔ i ∈ U} := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_iInter, Finset.mem_filter,
      Finset.mem_univ, true_and, exists_prop]
    constructor
    · intro h
      exact ⟨_, h, fun i => by simp⟩
    · rintro ⟨U, hU, hi⟩
      have : Finset.univ.filter (fun i => Q i ω) = U := by
        ext i
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        exact hi i
      rw [this]
      exact hU
  rw [hcov]
  refine Finset.measurableSet_biUnion _ (fun U _ => MeasurableSet.iInter (fun i => ?_))
  by_cases hi : i ∈ U
  · have h : {ω | Q i ω ↔ i ∈ U} = {ω | Q i ω} := by
      ext ω
      simp [hi]
    rw [h]
    exact hQ i
  · have h : {ω | Q i ω ↔ i ∈ U} = {ω | Q i ω}ᶜ := by
      ext ω
      simp [hi]
    rw [h]
    exact (hQ i).compl

/-- The raw count over a stream's first `m` draws, read on `Fin m`. -/
lemma card_filter_fin_eq (m : ℕ) (Q : S → Prop) [DecidablePred Q] (y : ℕ → S) :
    ((Finset.univ : Finset (Fin m)).filter (fun i => Q (y i.val))).card
      = ((Finset.range m).filter (fun i => Q (y i))).card :=
  card_filter_univ_fin m (fun i => Q (y i))

open scoped Classical in
/-- The draws are not mostly prefixes the family flips.  The heavy set is chosen by the
noise and the table; the certification draws are neither, so their hit count is the upper
Hoeffding tail at its mass. -/
noncomputable def heavyHits (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J)
    (Dj : Measure S) (j : J) (B : State) (f q t : ℝ) : Set (Run Ω S J) :=
  {x | Dj.real {p | ¬ (flipCount O (clusterBy rule O.mq populations x B) p
        ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f)} ≤ q
    ∧ (B.npref : ℝ) * (q + t)
        ≤ (((Finset.univ : Finset (Fin B.npref)).filter (fun i =>
          ¬ (flipCount O (clusterBy rule O.mq populations x B) (certPrefix j i.val x)
            ≤ ((clusterBy rule O.mq populations x B).card : ℝ) * f))).card : ℝ)}

open scoped Classical in
lemma measurableSet_heavyHits (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (j : J) (B : State) (f q t : ℝ) :
    MeasurableSet (heavyHits rule O populations Dj j B f q t) := by
  classical
  have hR : ∀ (P C : Finset S) (tt : Fin B.npref → S), MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | Dj.real {p | ¬ (flipCount O
            (clusterOf rule O B.sc B.scd P C B.k (oracleNoise x)) p
          ≤ ((clusterOf rule O B.sc B.scd P C B.k (oracleNoise x)).card : ℝ) * f)} ≤ q
        ∧ (B.npref : ℝ) * (q + t)
            ≤ (((Finset.univ : Finset (Fin B.npref)).filter (fun i =>
              ¬ (flipCount O (clusterOf rule O B.sc B.scd P C B.k (oracleNoise x)) (tt i)
                ≤ ((clusterOf rule O B.sc B.scd P C B.k (oracleNoise x)).card : ℝ) * f))).card
                  : ℝ)} else ∅) := by
    intro P C tt
    split_ifs with hone
    · refine measurable_nz (measurableSet_of_fam (T := C.powerset)
        (fun ω => Finset.mem_powerset.2 (clusterOf_subset O B.sc B.scd P C B.k ω hone))
        (fun A₀ => measurableSet_clusterOf O B.sc B.scd P C B.k hone A₀)
        (fun A₀ => {_ω : Ω |
          Dj.real {p | ¬ (flipCount O A₀ p ≤ (A₀.card : ℝ) * f)} ≤ q
          ∧ (B.npref : ℝ) * (q + t)
              ≤ (((Finset.univ : Finset (Fin B.npref)).filter (fun i =>
                ¬ (flipCount O A₀ (tt i) ≤ (A₀.card : ℝ) * f))).card : ℝ)})
        (fun A₀ => ?_))
      by_cases hcond : Dj.real {p | ¬ (flipCount O A₀ p
            ≤ (A₀.card : ℝ) * f)} ≤ q
          ∧ (B.npref : ℝ) * (q + t)
            ≤ (((Finset.univ : Finset (Fin B.npref)).filter (fun i =>
              ¬ (flipCount O A₀ (tt i) ≤ (A₀.card : ℝ) * f))).card : ℝ)
      · simpa [hcond] using MeasurableSet.univ
      · simpa [hcond] using MeasurableSet.empty
    · exact MeasurableSet.empty
  have hrw : heavyHits rule O populations Dj j B f q t
      = {x : Run Ω S J | x ∈ (fun P C tt => if (1 : S) ∈ C then
          {x : Run Ω S J | Dj.real {p | ¬ (flipCount O
                (clusterOf rule O B.sc B.scd P C B.k (oracleNoise x)) p
              ≤ ((clusterOf rule O B.sc B.scd P C B.k (oracleNoise x)).card : ℝ) * f)} ≤ q
            ∧ (B.npref : ℝ) * (q + t)
                ≤ (((Finset.univ : Finset (Fin B.npref)).filter (fun i =>
                  ¬ (flipCount O (clusterOf rule O B.sc B.scd P C B.k (oracleNoise x)) (tt i)
                    ≤ ((clusterOf rule O B.sc B.scd P C B.k (oracleNoise x)).card : ℝ)
                      * f))).card : ℝ)} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (fun i : Fin B.npref => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), heavyHits,
      ← clusterAt_eq_clusterOf O populations B x]
  rw [hrw]
  exact measurableSet_of_run_data_cert populations j B _ hR

open scoped Classical in
/-- The flip-heavy prefixes are a `q` fraction of the population, so the first `m` draws hold at
most `m(q + t)` of them. -/
theorem measureReal_heavyHits_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (populations : Finset J) (j : J) (B : State) (f q t : ℝ) (ht : 0 ≤ t) :
    (runMeasure μ D Dsf).real (heavyHits rule O populations (D j) j B f q t)
      ≤ Real.exp (-2 * (B.npref : ℝ) * t ^ 2) := by
  classical
  have hEnn : runMeasure μ D Dsf (heavyHits rule O populations (D j) j B f q t)
      ≤ ENNReal.ofReal (Real.exp (-2 * (B.npref : ℝ) * t ^ 2)) := by
    refine runMeasure_slice_cert_le D Dsf _
      (measurableSet_heavyHits O populations (D j) j B f q t) _ ?_
    intro y
    set F : Finset S :=
      clusterBy rule O.mq populations ((y.1, (y.2, fun _ _ => (1 : S))) : Run Ω S J) B with hF
    set W : Set S := {p | ¬ (flipCount O F p ≤ (F.card : ℝ) * f)} with hW
    have hFeq : ∀ c : J → ℕ → S,
        clusterBy rule O.mq populations ((y.1, (y.2, c)) : Run Ω S J) B = F := fun c => rfl
    by_cases hmass : (D j).real W ≤ q
    · have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
            ∈ heavyHits rule O populations (D j) j B f q t}
          ⊆ {c | (B.npref : ℝ) * (q + t)
            ≤ (((Finset.range B.npref).filter (fun i => c j i ∈ W)).card : ℝ)} := by
        rintro c ⟨-, hcount⟩
        have h := card_filter_fin_eq B.npref (fun p => p ∈ W) (fun i => c j i)
        show (B.npref : ℝ) * (q + t)
          ≤ (((Finset.range B.npref).filter (fun i => c j i ∈ W)).card : ℝ)
        rw [← h]
        exact hcount
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top
          (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) _), ← measureReal_def]
      refine ENNReal.ofReal_le_ofReal ?_
      have hcert := cert_hits_upper D j B.npref W q t ht hmass
      convert hcert using 3
      funext c
      congr!
    · have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
          ∈ heavyHits rule O populations (D j) j B f q t} = (∅ : Set (J → ℕ → S)) := by
        ext c
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨hm, -⟩
        exact hmass hm
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (heavyHits rule O populations (D j) j B f q t)).toReal
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

lemma measurable_undecidedMassReal (O : Oracle μ S) (Dj : Measure S) (lo ha : ℕ)
    (A₀ : Finset S) : Measurable (fun ω => Dj.real {p | ¬ decided O.mq lo ha A₀ p ω}) :=
  ENNReal.measurable_toReal.comp (measurable_badMass Dj
    (fun p => {ω | ¬ decided O.mq lo ha A₀ p ω})
    (fun p => noiseAlg_le O Set.univ _ (measurableSet_decided' O lo ha A₀ p)))

open scoped Classical in
/-- The certification draws miss the prefixes the family's vote leaves undecided. -/
noncomputable def undecidedShort (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J)
    (Dj : Measure S) (j : J) (B : State) (l t : ℝ) : Set (Run Ω S J) :=
  {x | l ≤ Dj.real {p | ¬ decided O.mq B.lo (B.hi + 1) (familyBy rule O.mq populations x B) p
        (oracleNoise x)}
    ∧ (((Finset.univ : Finset (Fin B.npref)).filter (fun i => ¬ decided O.mq B.lo (B.hi + 1)
        (familyBy rule O.mq populations x B) (certPrefix j i.val x) (oracleNoise x))).card : ℝ)
      ≤ (B.npref : ℝ) * (l - t)}

open scoped Classical in
lemma measurableSet_undecidedShort (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (j : J) (B : State) (l t : ℝ) :
    MeasurableSet (undecidedShort rule O populations Dj j B l t) := by
  classical
  have hR : ∀ (P C : Finset S) (tt : Fin B.npref → S), MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | l ≤ Dj.real {p | ¬ decided O.mq B.lo (B.hi + 1)
            (insert 1 (clusterOf rule O B.sc B.scd P C B.k (oracleNoise x))) p (oracleNoise x)}
        ∧ (((Finset.univ : Finset (Fin B.npref)).filter (fun i =>
            ¬ decided O.mq B.lo (B.hi + 1)
              (insert 1 (clusterOf rule O B.sc B.scd P C B.k (oracleNoise x))) (tt i)
              (oracleNoise x))).card : ℝ)
          ≤ (B.npref : ℝ) * (l - t)} else ∅) := by
    intro P C tt
    split_ifs with hone
    · have hω : MeasurableSet {ω : Ω |
          l ≤ Dj.real {p | ¬ decided O.mq B.lo (B.hi + 1)
              (insert 1 (clusterOf rule O B.sc B.scd P C B.k ω)) p ω}
          ∧ (((Finset.univ : Finset (Fin B.npref)).filter (fun i =>
              ¬ decided O.mq B.lo (B.hi + 1)
                (insert 1 (clusterOf rule O B.sc B.scd P C B.k ω)) (tt i) ω)).card : ℝ)
            ≤ (B.npref : ℝ) * (l - t)} :=
        measurableSet_of_fam (T := C.powerset)
          (fun ω => Finset.mem_powerset.2
            (clusterOf_subset O B.sc B.scd P C B.k ω hone))
          (fun A₀ => measurableSet_clusterOf O B.sc B.scd P C B.k hone A₀)
          (fun A₀ => {ω | l ≤ Dj.real {p | ¬ decided O.mq B.lo (B.hi + 1) (insert 1 A₀) p ω}
            ∧ (((Finset.univ : Finset (Fin B.npref)).filter (fun i =>
                ¬ decided O.mq B.lo (B.hi + 1) (insert 1 A₀) (tt i) ω)).card : ℝ)
              ≤ (B.npref : ℝ) * (l - t)})
          (fun A₀ => MeasurableSet.inter
            (measurableSet_le measurable_const
              (measurable_undecidedMassReal O Dj B.lo (B.hi + 1) (insert 1 A₀)))
            (measurableSet_card_filter_fin
              (fun i ω => ¬ decided O.mq B.lo (B.hi + 1) (insert 1 A₀) (tt i) ω)
              (fun i => noiseAlg_le O Set.univ _
                (measurableSet_decided' O B.lo (B.hi + 1) (insert 1 A₀) (tt i)))
              (fun n => (n : ℝ) ≤ (B.npref : ℝ) * (l - t))))
      exact measurable_nz hω
    · exact MeasurableSet.empty
  have hrw : undecidedShort rule O populations Dj j B l t
      = {x : Run Ω S J | x ∈ (fun P C tt => if (1 : S) ∈ C then
          {x : Run Ω S J | l ≤ Dj.real {p | ¬ decided O.mq B.lo (B.hi + 1)
                (insert 1 (clusterOf rule O B.sc B.scd P C B.k (oracleNoise x))) p
                (oracleNoise x)}
            ∧ (((Finset.univ : Finset (Fin B.npref)).filter (fun i =>
                ¬ decided O.mq B.lo (B.hi + 1)
                  (insert 1 (clusterOf rule O B.sc B.scd P C B.k (oracleNoise x))) (tt i)
                  (oracleNoise x))).card : ℝ)
              ≤ (B.npref : ℝ) * (l - t)} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (fun i : Fin B.npref => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), undecidedShort, familyBy,
      ← clusterAt_eq_clusterOf O populations B x]
    rfl
  rw [hrw]
  exact measurableSet_of_run_data_cert populations j B _ hR

open scoped Classical in
/-- The family is fixed before the certification draws, so a large undecided mass is seen on
them. -/
theorem measureReal_undecidedShort_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (populations : Finset J) (j : J) (B : State) (l t : ℝ) (hl : 0 ≤ l) (ht : 0 ≤ t) :
    (runMeasure μ D Dsf).real (undecidedShort rule O populations (D j) j B l t)
      ≤ Real.exp (-2 * (B.npref : ℝ) * t ^ 2) := by
  classical
  set E : ℝ := Real.exp (-2 * (B.npref : ℝ) * t ^ 2) with hEdef
  have hEnn : runMeasure μ D Dsf (undecidedShort rule O populations (D j) j B l t)
      ≤ ENNReal.ofReal E := by
    refine runMeasure_slice_cert_le D Dsf _
      (measurableSet_undecidedShort O populations (D j) j B l t) _ ?_
    intro y
    set F : Finset S :=
      familyBy rule O.mq populations ((y.1, (y.2, fun _ _ => (1 : S))) : Run Ω S J) B with hF
    set W : Set S := {p | ¬ decided O.mq B.lo (B.hi + 1) F p y.1} with hW
    have hFeq : ∀ c : J → ℕ → S,
        familyBy rule O.mq populations ((y.1, (y.2, c)) : Run Ω S J) B = F := fun c => rfl
    by_cases hmass : l ≤ (D j).real W
    · have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
            ∈ undecidedShort rule O populations (D j) j B l t}
          ⊆ {c | (((Finset.range B.npref).filter (fun i => c j i ∈ W)).card : ℝ)
            ≤ (B.npref : ℝ) * (l - t)} := by
        rintro c ⟨-, hcount⟩
        have h := card_filter_fin_eq B.npref (fun p => p ∈ W) (fun i => c j i)
        show (((Finset.range B.npref).filter (fun i => c j i ∈ W)).card : ℝ)
          ≤ (B.npref : ℝ) * (l - t)
        rw [← h]
        exact hcount
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal
          (measure_ne_top (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) _),
        ← measureReal_def]
      refine ENNReal.ofReal_le_ofReal ?_
      rw [hEdef]
      have hcert := cert_hits_wrongSet D j B.npref W l t hl ht hmass
      convert hcert using 3
      funext c
      congr!
    · have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
          ∈ undecidedShort rule O populations (D j) j B l t} = (∅ : Set (J → ℕ → S)) := by
        ext c
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨hm, -⟩
        rw [hFeq c] at hm
        exact hmass hm
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (undecidedShort rule O populations (D j) j B l t)).toReal
      ≤ (ENNReal.ofReal E).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = E := ENNReal.toReal_ofReal (Real.exp_nonneg _)

/-- The family is usable at `p`: at most an `f` fraction of it flips there, and the family
is neither too small for the thresholds nor larger than the round allows. -/
def famGood (O : Oracle μ S) (f : ℝ) (kmin kmax : ℕ) (F : Finset S) (p : S) : Prop :=
  flipCount O F p ≤ (F.card : ℝ) * f ∧ kmin ≤ F.card ∧ F.card ≤ kmax

open scoped Classical in
/-- What a certification sample found in time says about its population's first draws: it is
`m` strings, and any count over it is within `r` of the count over the first `m` draws. -/
lemma certOf_of_good (populations : Finset J) (j : J) (m r : ℕ) (x : Run Ω S J)
    (h : certGood populations j m r x) :
    (certOf (prefixesAt populations m x) j m x).card = m
      ∧ ∀ (Q : S → Prop) [DecidablePred Q],
        (((Finset.univ : Finset (Fin m)).filter (fun i => Q (certPrefix j i.val x))).card : ℝ)
          ≤ (((certOf (prefixesAt populations m x) j m x).filter Q).card : ℝ) + r
        ∧ (((certOf (prefixesAt populations m x) j m x).filter Q).card : ℝ)
          ≤ (((Finset.univ : Finset (Fin m)).filter (fun i => Q (certPrefix j i.val x))).card : ℝ)
            + r := by
  refine ⟨(freshDraws_of_good _ _ m r h).1, fun Q _ => ?_⟩
  have hc := card_filter_of_good (prefixesAt populations m x) (fun i => certPrefix j i x) m r h Q
  have hfin := card_filter_fin_eq m Q (fun i => certPrefix j i x)
  have hfinR : (((Finset.univ : Finset (Fin m)).filter (fun i => Q (certPrefix j i.val x))).card
      : ℝ) = (((Finset.range m).filter (fun i => Q (certPrefix j i x))).card : ℝ) := by
    exact_mod_cast hfin
  rw [hfinR]
  exact hc

lemma collisionMass_le_of_atoms (Dj : Measure S) [IsProbabilityMeasure Dj] {ρ : ℝ}
    (h : ∀ t : S, Dj.real {t} ≤ ρ) : collisionMass Dj ≤ ρ := by
  classical
  have hs := summable_singleton_real Dj
  have hρ0 : 0 ≤ ρ := le_trans measureReal_nonneg (h 1)
  have h1 : ∑' t : S, Dj.real {t} ≤ 1 :=
    hs.tsum_le_of_sum_le (fun Q => by
      rw [sum_measureReal_singleton]; exact measureReal_le_one' Dj _)
  unfold collisionMass
  calc ∑' t : S, Dj.real {t} ^ 2 ≤ ∑' t : S, ρ * Dj.real {t} :=
        (summable_singleton_sq Dj).tsum_le_tsum (fun t => by
          have := measureReal_nonneg (μ := Dj) (s := ({t} : Set S))
          nlinarith [h t]) (hs.mul_left ρ)
    _ = ρ * ∑' t : S, Dj.real {t} := tsum_mul_left
    _ ≤ ρ := by nlinarith

lemma measurableSet_cert_mem_table (populations : Finset J) (j : J) (m a : ℕ) :
    MeasurableSet {x : Run Ω S J | certPrefix j a x ∈ prefixesAt populations m x} := by
  classical
  have hcov : {x : Run Ω S J | certPrefix j a x ∈ prefixesAt populations m x}
      = ⋃ Q : Finset S, {x : Run Ω S J | prefixesAt populations m x = Q}
          ∩ {x | certPrefix j a x ∈ (Q : Set S)} := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Finset.mem_coe]
    exact ⟨fun h => ⟨_, rfl, h⟩, fun ⟨Q, hQ, h⟩ => hQ ▸ h⟩
  rw [hcov]
  exact MeasurableSet.iUnion (fun Q => (measurableSet_prefixesAt populations m Q).inter
    (measurable_cert j a (measurableSet_of_countable _)))

/-- A certification draw lands among a state's table prefixes, drawn independently of it, with
chance at most `|populations|·m·ρ` when no string has more than `ρ` of its population. -/
lemma measureReal_cert_mem_table_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (populations : Finset J)
    (j : J) (m a : ℕ) {ρ : ℝ} (hρ : ∀ t, (D j).real {t} ≤ ρ) :
    (runMeasure μ D Dsf).real {x : Run Ω S J | certPrefix j a x ∈ prefixesAt populations m x}
      ≤ (populations.card : ℝ) * m * ρ := by
  classical
  have hρ0 : 0 ≤ ρ := le_trans measureReal_nonneg (hρ 1)
  set ν : Measure (J → ℕ → S) := Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j
    with hν
  have hmp : MeasurePreserving (fun c : J → ℕ → S => c j a) ν (D j) :=
    (measurePreserving_eval_infinitePi _ a).comp
      (measurePreserving_eval (fun j : J => Measure.infinitePi fun _ : ℕ => D j) j)
  have hE : runMeasure μ D Dsf {x : Run Ω S J | certPrefix j a x ∈ prefixesAt populations m x}
      ≤ ENNReal.ofReal ((populations.card : ℝ) * m * ρ) := by
    refine runMeasure_slice_cert_le D Dsf _ (measurableSet_cert_mem_table populations j m a) _ ?_
    intro y
    set T : Finset S := prefixesAt populations m ((y.1, (y.2, fun _ _ => (1 : S))) : Run Ω S J)
      with hT
    have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
          ∈ {x : Run Ω S J | certPrefix j a x ∈ prefixesAt populations m x}}
        = (fun c : J → ℕ → S => c j a) ⁻¹' (T : Set S) := by
      ext c
      rfl
    rw [hsec, hmp.measure_preimage (measurableSet_of_countable _).nullMeasurableSet,
      ← ENNReal.ofReal_toReal (measure_ne_top _ _), ← measureReal_def]
    refine ENNReal.ofReal_le_ofReal ?_
    rw [← sum_measureReal_singleton]
    have hcard : (T.card : ℝ) ≤ (populations.card : ℝ) * m := by
      exact_mod_cast card_prefixesAt_le populations m _
    calc ∑ t ∈ T, (D j).real {t} ≤ ∑ _t ∈ T, ρ := Finset.sum_le_sum (fun t _ => hρ t)
      _ = (T.card : ℝ) * ρ := by rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ (populations.card : ℝ) * m * ρ := mul_le_mul_of_nonneg_right hcard hρ0
  rw [measureReal_def]
  exact le_trans (ENNReal.toReal_mono ENNReal.ofReal_ne_top hE)
    (le_of_eq (ENNReal.toReal_ofReal (by positivity)))

/-- Population `j`'s certification sample finds `2m` strings outside the state's tables within
its first `2m + r` draws, so every look up to twice the state's count is found in time. -/
def certGood2 (populations : Finset J) (j : J) (m r : ℕ) (x : Run Ω S J) : Prop :=
  2 * m ≤ (acceptedAt (prefixesAt populations m x) (fun i => certPrefix j i x) (2 * m + r)).card

lemma certGood_of_certGood2 (populations : Finset J) (j : J) (m r : ℕ) (x : Run Ω S J)
    (h : certGood2 populations j m r x) : certGood populations j m r x := by
  have := card_acceptedAt_add_le (prefixesAt populations m x) (fun i => certPrefix j i x)
    (show m + r ≤ 2 * m + r by omega)
  unfold certGood2 at h
  unfold certGood
  omega

open scoped Classical in
set_option maxHeartbeats 1000000 in
/-- A certification sample that is not found in time redraws `r + 1` times at repeats or at
table prefixes; only `j`'s own atoms enter. -/
theorem measureReal_certBad2_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (populations : Finset J)
    (j : J) (m r : ℕ) (ρ : ℝ) (hρj : ∀ t, (D j).real {t} ≤ ρ) :
    (runMeasure μ D Dsf).real {x : Run Ω S J | ¬ certGood2 populations j m r x}
      ≤ (((2 * m + r : ℕ) : ℝ) * ((populations.card : ℝ) * m * ρ)
          + ((2 * m + r : ℕ) : ℝ) ^ 2 * ρ / 2) / ((r : ℝ) + 1) := by
  classical
  have hρ0 : 0 ≤ ρ := le_trans measureReal_nonneg (hρj 1)
  have hcm := collisionMass_le_of_atoms (D j) hρj
  set n : ℕ := 2 * m + r with hn
  set I : Finset (ℕ × ℕ) :=
    ((Finset.range n) ×ˢ (Finset.range n)).filter (fun q => q.1 < q.2) with hI
  set A : ℕ → Set (Run Ω S J) :=
    fun a => {x | certPrefix j a x ∈ prefixesAt populations m x} with hA
  set Bq : ℕ × ℕ → Set (Run Ω S J) :=
    fun q => {x | certPrefix j q.1 x = certPrefix j q.2 x} with hBq
  have hsub : {x : Run Ω S J | ¬ certGood2 populations j m r x}
      ⊆ {x | ((r : ℝ) + 1) ≤ (((Finset.range n).filter (fun a => x ∈ A a)).card : ℝ)
          + ((I.filter (fun q => x ∈ Bq q)).card : ℝ)} := by
    intro x hx
    set y : ℕ → S := fun i => certPrefix j i x with hy
    set T : Finset S := prefixesAt populations m x with hT
    have h1 := le_card_acceptedAt_add T y n
    have hshort : (acceptedAt T y n).card < 2 * m := not_le.1 hx
    have hhits : ((Finset.range n).filter (fun i => y i ∈ T)).card
        = ((Finset.range n).filter (fun a => x ∈ A a)).card := rfl
    have hpairs : (repeatPairs y n).card = (I.filter (fun q => x ∈ Bq q)).card := by
      congr 1
      ext q
      simp only [repeatPairs, hI, hBq, Finset.mem_filter, Set.mem_setOf_eq, and_assoc, hy]
    have : r + 1 ≤ ((Finset.range n).filter (fun a => x ∈ A a)).card
        + (I.filter (fun q => x ∈ Bq q)).card := by omega
    show ((r : ℝ) + 1) ≤ (((Finset.range n).filter (fun a => x ∈ A a)).card : ℝ)
      + ((I.filter (fun q => x ∈ Bq q)).card : ℝ)
    exact_mod_cast this
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_count_ge_le (runMeasure μ D Dsf) (Finset.range n) A
    (fun a => measurableSet_cert_mem_table populations j m a)
    I Bq (fun q => measurableSet_eq_fun (measurable_cert j q.1) (measurable_cert j q.2))
    (by positivity)) ?_
  have hAsum : ∑ a ∈ Finset.range n, (runMeasure μ D Dsf).real (A a)
      ≤ (n : ℝ) * ((populations.card : ℝ) * m * ρ) := by
    calc ∑ a ∈ Finset.range n, (runMeasure μ D Dsf).real (A a)
        ≤ ∑ _a ∈ Finset.range n, (populations.card : ℝ) * m * ρ :=
          Finset.sum_le_sum (fun a _ => measureReal_cert_mem_table_le D Dsf populations j m a hρj)
      _ = (n : ℝ) * ((populations.card : ℝ) * m * ρ) := by
          rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  have hIsum : ∑ q ∈ I, (runMeasure μ D Dsf).real (Bq q) ≤ (I.card : ℝ) * ρ := by
    calc ∑ q ∈ I, (runMeasure μ D Dsf).real (Bq q) ≤ ∑ _q ∈ I, ρ := by
          refine Finset.sum_le_sum (fun q hq => ?_)
          simp only [hI, Finset.mem_filter, Finset.mem_product, Finset.mem_range] at hq
          have hY : Measurable (fun x : Run Ω S J => fun i : Fin n => certPrefix j i.val x) :=
            measurable_pi_lambda _ (fun i => measurable_cert j i.val)
          have h := block_pair_eq (runMeasure μ D Dsf) (D j)
            (fun x : Run Ω S J => fun i : Fin n => certPrefix j i.val x) hY
            (map_certBlock D Dsf j n) (a := ⟨q.1, hq.1.1⟩) (b := ⟨q.2, hq.1.2⟩)
            (fun h => absurd (congrArg Fin.val h) (ne_of_lt hq.2))
          simp only at h
          rw [show Bq q = {x | certPrefix j q.1 x = certPrefix j q.2 x} from rfl, h]
          exact hcm
      _ = (I.card : ℝ) * ρ := by rw [Finset.sum_const, nsmul_eq_mul]
  have hIcard : 2 * (I.card : ℝ) ≤ (n : ℝ) ^ 2 := by
    have h := card_repeatPairs_le (fun _ : ℕ => (1 : S)) n
    have hI' : repeatPairs (fun _ : ℕ => (1 : S)) n = I := by
      ext q
      simp [repeatPairs, hI]
    rw [hI'] at h
    have : ((2 * I.card : ℕ) : ℝ) ≤ ((n * n : ℕ) : ℝ) := by exact_mod_cast h
    push_cast at this
    nlinarith
  refine div_le_div_of_nonneg_right ?_ (by positivity)
  nlinarith [mul_le_mul_of_nonneg_right hIcard hρ0]

/-- The tame populations' tables and certification samples are all found in time. -/
def drawsGoodOn (populations tame : Finset J) (m r : ℕ) (x : Run Ω S J) : Prop :=
  (∀ j ∈ tame, tabGood j m r x) ∧ ∀ j ∈ tame, certGood2 populations j m r x

open scoped Classical in
theorem measureReal_drawsBadOn_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (populations tame : Finset J)
    (m r : ℕ) (ρ : ℝ) (hρ : ∀ j ∈ tame, ∀ t, (D j).real {t} ≤ ρ) (hρ0 : 0 ≤ ρ) :
    (runMeasure μ D Dsf).real {x : Run Ω S J | ¬ drawsGoodOn populations tame m r x}
      ≤ (tame.card : ℝ) * (((m + r : ℕ) : ℝ) ^ 2 * ρ / (2 * ((r : ℝ) + 1))
        + (((2 * m + r : ℕ) : ℝ) * ((populations.card : ℝ) * m * ρ)
          + ((2 * m + r : ℕ) : ℝ) ^ 2 * ρ / 2) / ((r : ℝ) + 1)) := by
  classical
  have hsub : {x : Run Ω S J | ¬ drawsGoodOn populations tame m r x}
      ⊆ ⋃ j ∈ tame, ({x : Run Ω S J | ¬ tabGood j m r x}
        ∪ {x : Run Ω S J | ¬ certGood2 populations j m r x}) := by
    intro x hx
    simp only [drawsGoodOn, not_and_or, not_forall, Set.mem_setOf_eq] at hx
    rcases hx with ⟨j, hj, hc⟩ | ⟨j, hj, hc⟩
    · exact Set.mem_biUnion hj (Or.inl hc)
    · exact Set.mem_biUnion hj (Or.inr hc)
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  refine le_trans (measureReal_biUnion_finset_le _ _) ?_
  rw [← nsmul_eq_mul, ← Finset.sum_const]
  refine Finset.sum_le_sum (fun j hj => le_trans (measureReal_union_le _ _) (add_le_add ?_ ?_))
  · exact measureReal_tabBad_le D Dsf j m r ρ (collisionMass_le_of_atoms (D j) (hρ j hj)) hρ0
  · exact measureReal_certBad2_le D Dsf populations j m r ρ (hρ j hj)

end Loop

end OrthoDFA
